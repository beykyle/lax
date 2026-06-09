"""Compile-time orchestration for solver bundles.

This module owns the "Python side" of the library described in ``DESIGN.md §14``:
it validates the requested solver features, builds mesh/operator caches, precomputes
boundary data, binds pickle-safe runtime callables, and returns the final
``Solver`` bundle used by downstream code.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from lax.boundary import compute_boundary_values
from lax.boundary._types import (
    BoundaryValues,
    DirectGridObservable,
    DirectRMatrixKernel,
    DoubleFourierTransform,
    EigenpairAccessor,
    FourierTransform,
    FromGridVectorTransform,
    GreenFunctionObservable,
    GridMatrixTransform,
    GridVectorTransform,
    Integrator,
    InterpolatorBuilder,
    Mesh,
    OperatorMatrices,
    RMatrixObservable,
    Solver,
    SpectrumGridObservable,
    SpectrumKernel,
    SpectrumObservable,
    TransformMatrices,
    WavefunctionObservable,
)
from lax.meshes import build_mesh
from lax.solvers import (
    bind_direct_grid_observables,
    bind_grid_observables,
    bind_interpolators,
    bind_observables,
    make_rmatrix_direct_grid_observable,
    make_rmatrix_direct_kernel,
    make_spectrum_kernel,
)
from lax.transforms import (
    compute_B_grid,
    compute_F_momentum,
    make_double_fourier,
    make_fourier,
    make_integration,
    make_to_grid,
)
from lax.types import ChannelSpec, MeshSpec, Method


@dataclass(frozen=True)
class _CompileRequest:
    """Normalized user request after validation and default resolution."""

    channels: tuple[ChannelSpec, ...]
    operators: frozenset[str]
    solvers: frozenset[str]
    method: Method
    needs_spectrum: bool
    needs_boundary: bool
    keep_eigenvectors: bool


@dataclass(frozen=True)
class _TransformBundle:
    """Optional transform matrices and bound runtime helpers."""

    matrices: TransformMatrices
    to_grid_vector: GridVectorTransform | None
    from_grid_vector: FromGridVectorTransform | None
    to_grid_matrix: GridMatrixTransform | None
    fourier: FourierTransform | None
    double_fourier_transform: DoubleFourierTransform | None
    integrate: Integrator


@dataclass(frozen=True)
class _ObservableBundle:
    """Runtime observable entry points bound to compile-time caches."""

    spectrum: SpectrumKernel | None
    rmatrix: RMatrixObservable | None
    smatrix: SpectrumObservable | None
    phases: SpectrumObservable | None
    greens: GreenFunctionObservable | None
    wavefunction: WavefunctionObservable | None
    eigh: EigenpairAccessor | None
    rmatrix_grid: SpectrumGridObservable | None
    smatrix_grid: SpectrumGridObservable | None
    phases_grid: SpectrumGridObservable | None
    rmatrix_direct: DirectRMatrixKernel | None
    rmatrix_direct_grid: DirectGridObservable | None
    smatrix_direct_grid: DirectGridObservable | None
    phases_direct_grid: DirectGridObservable | None
    interpolate_rmatrix: InterpolatorBuilder | None
    interpolate_smatrix: InterpolatorBuilder | None
    interpolate_phases: InterpolatorBuilder | None


def compile(
    *,
    mesh: MeshSpec,
    channels: Iterable[ChannelSpec],
    operators: Iterable[str] = ("T+L",),
    solvers: Iterable[str] = ("spectrum", "rmatrix", "smatrix", "phases"),
    energies: jax.Array | None = None,
    energy_dependent: bool = False,
    method: Method | None = None,
    V_is_complex: bool = False,
    grid: jax.Array | None = None,
    momenta: jax.Array | None = None,
    z1z2: tuple[int, int] | None = None,
    dps: int = 40,
) -> Solver:
    """Build a compiled solver bundle for one mesh/channel definition.

    .. note::
        ``lax.compile`` shadows Python's built-in ``compile``.  Avoid
        ``from lax import compile`` in modules that also use the built-in.

    Parameters
    ----------
    mesh
        User-facing mesh specification. The chosen mesh family, regularization,
        scale, and any mesh-specific extras are resolved at compile time.
    channels
        Channel definitions baked into the compiled solver structure.
    operators
        Compile-time operator matrices to precompute. ``"T+L"`` is injected
        automatically whenever the requested solver path needs it.
    solvers
        Runtime entry points to expose on the returned :class:`~lax.Solver`.
        The potential passed to ``solver.spectrum(V)`` or
        ``solver.rmatrix_direct(V)`` must have shape ``(N_c, N_c, N)`` for a
        local potential or ``(N_c, N_c, N, N)`` for a non-local kernel, where
        ``N = mesh.n`` and ``N_c = len(channels)``.  Use
        :func:`lax.assemble_local` / :func:`lax.assemble_nonlocal` to build
        these arrays.
    energies
        Compile-time energy grid used for boundary-value-dependent observables and
        aligned-grid workflows.
    energy_dependent
        Whether the caller intends to provide an energy-dependent potential on the
        compile-time energy grid.  When ``True``, call
        ``jax.vmap(solver.spectrum)(V_grid)`` over the energy axis to get a
        batched ``Spectrum``, then use ``solver.phases_grid(spectra)`` and the
        ``solver.interpolate_*`` builders for off-grid evaluation.
    method
        Explicit solver method. When omitted, the method is chosen from
        ``V_is_complex`` and the active JAX backend.
    V_is_complex
        Whether the potential path is complex-valued.
    grid
        Optional radial grid used to precompute mesh-to-grid transforms.
        Accessible afterward as ``solver.grid_r``.
    momenta
        Optional momentum grid used to precompute Fourier transforms.
        Accessible afterward as ``solver.momenta``.
    z1z2
        Optional pair of charges passed to compile-time boundary-value evaluation.
    dps
        Decimal precision for the ``mpmath`` boundary-value calculation.

    Returns
    -------
    Solver
        A pickle-safe solver bundle containing compile-time caches and bound
        runtime observables.
    """

    request = _resolve_compile_request(
        channels=channels,
        operators=operators,
        solvers=solvers,
        energies=energies,
        energy_dependent=energy_dependent,
        method=method,
        V_is_complex=V_is_complex,
    )
    mesh_data, operator_matrices = _build_solver_mesh(
        mesh=mesh,
        operators=request.operators,
        needs_spectrum=request.needs_spectrum,
        method=request.method,
        grid=grid,
        momenta=momenta,
    )
    boundary, energies_array = _prepare_boundary_data(
        channels=request.channels,
        energies=energies,
        channel_radius=mesh.scale,
        z1z2=z1z2,
        dps=dps,
    )
    transforms = _prepare_transforms(
        mesh=mesh_data,
        channels=request.channels,
        grid=grid,
        momenta=momenta,
    )
    observables = _bind_solver_observables(
        request=request,
        mesh=mesh_data,
        operators=operator_matrices,
        energies=energies_array,
        boundary=boundary,
        has_energy_grid=energies is not None,
    )
    return _assemble_solver(
        request=request,
        mesh=mesh_data,
        operators=operator_matrices,
        energies=energies_array,
        boundary=boundary,
        transforms=transforms,
        observables=observables,
    )


def _resolve_compile_request(
    *,
    channels: Iterable[ChannelSpec],
    operators: Iterable[str],
    solvers: Iterable[str],
    energies: jax.Array | None,
    energy_dependent: bool,
    method: Method | None,
    V_is_complex: bool,
) -> _CompileRequest:
    """Normalize the requested features into one immutable compile plan.

    The rest of ``compile()`` operates on this normalized request rather than on
    raw user inputs. Keeping those decisions in one place makes the orchestration
    path easier to read and keeps feature gating consistent.
    """

    channels_tuple = tuple(channels)
    operators_set = set(operators)
    solvers_set = frozenset(solvers)
    needs_spectrum = bool(
        solvers_set & {"spectrum", "rmatrix", "smatrix", "phases", "greens", "wavefunction"}
    )
    needs_boundary = bool(solvers_set & {"smatrix", "phases", "rmatrix_direct"})
    if (needs_boundary or energy_dependent) and energies is None:
        msg = "`energies` is required for continuum solvers or energy-dependent potentials."
        raise ValueError(msg)
    if needs_spectrum or "rmatrix_direct" in solvers_set:
        operators_set.add("T+L")

    selected_method = method or _default_method(V_is_complex)
    if selected_method not in {"eigh", "eig", "linear_solve"}:
        msg = f"Method {selected_method!r} is not implemented in the MVP compile() path."
        raise ValueError(msg)
    if needs_spectrum and selected_method not in {"eigh", "eig"}:
        msg = f"Method {selected_method!r} is not implemented in the MVP spectrum path."
        raise ValueError(msg)

    return _CompileRequest(
        channels=channels_tuple,
        operators=frozenset(operators_set),
        solvers=solvers_set,
        method=selected_method,
        needs_spectrum=needs_spectrum,
        needs_boundary=needs_boundary,
        keep_eigenvectors=bool(solvers_set & {"greens", "wavefunction"}),
    )


def _build_solver_mesh(
    *,
    mesh: MeshSpec,
    operators: frozenset[str],
    needs_spectrum: bool,
    method: Method,
    grid: jax.Array | None,
    momenta: jax.Array | None,
) -> tuple[Mesh, OperatorMatrices]:
    """Build mesh/operator caches and reject unsupported feature combinations."""

    mesh_data, operator_matrices = build_mesh(
        family=mesh.family,
        regularization=mesh.regularization,
        n=mesh.n,
        scale=mesh.scale,
        operators=set(operators),
        **mesh.extras,
    )
    if mesh_data.n_intervals > 1 and needs_spectrum:
        msg = (
            "Subinterval propagation is defined only for local potentials on the direct "
            "linear-solve path. Spectrum-derived observables are not mathematically "
            "supported for propagated meshes."
        )
        raise ValueError(msg)
    if mesh_data.n_intervals > 1 and grid is not None:
        msg = (
            "Subinterval propagation is defined only for local direct linear-solve "
            "workflows. Radial-grid transforms require a global basis formulation, so "
            "they are not mathematically supported for propagated meshes."
        )
        raise ValueError(msg)
    if mesh_data.n_intervals > 1 and momenta is not None:
        msg = (
            "Subinterval propagation is defined only for local direct linear-solve "
            "workflows. Momentum transforms require a global basis formulation, so "
            "they are not mathematically supported for propagated meshes."
        )
        raise ValueError(msg)
    if mesh_data.n_intervals > 1 and method != "linear_solve":
        msg = (
            "Subinterval propagation is defined only for the local direct linear-solve "
            "formulation. Propagated meshes cannot be used with spectral eigensolvers."
        )
        raise ValueError(msg)
    return mesh_data, operator_matrices


def _prepare_boundary_data(
    *,
    channels: tuple[ChannelSpec, ...],
    energies: jax.Array | None,
    channel_radius: float,
    z1z2: tuple[int, int] | None,
    dps: int,
) -> tuple[BoundaryValues | None, jax.Array]:
    """Prepare compile-time boundary values and the canonical energy-grid array."""

    if energies is None:
        # The solver always stores an energy array so downstream code can rely on a
        # uniform bundle shape even when no boundary-valued observables are exposed.
        empty_energies = np.zeros((0,), dtype=np.float64)
        return None, _to_jax_array(empty_energies)

    energies_np = np.asarray(energies)
    boundary = compute_boundary_values(
        channels=channels,
        energies=energies_np,
        channel_radius=channel_radius,
        z1z2=z1z2,
        dps=dps,
    )
    return boundary, _to_jax_array(energies_np)


def _prepare_transforms(
    *,
    mesh: Mesh,
    channels: tuple[ChannelSpec, ...],
    grid: jax.Array | None,
    momenta: jax.Array | None,
) -> _TransformBundle:
    """Bind optional grid/Fourier transforms around one compiled mesh."""

    transforms = TransformMatrices()
    to_grid_vector_fn: GridVectorTransform | None = None
    from_grid_vector_fn: FromGridVectorTransform | None = None
    to_grid_matrix_fn: GridMatrixTransform | None = None
    fourier_fn: FourierTransform | None = None
    double_fourier_transform_fn: DoubleFourierTransform | None = None

    if grid is not None:
        grid_array = _to_jax_array(np.asarray(grid))
        basis_grid = compute_B_grid(mesh, grid_array)
        transforms = TransformMatrices(
            B_grid=basis_grid,
            grid_r=grid_array,
            F_momentum=transforms.F_momentum,
            momenta=transforms.momenta,
        )
        (
            to_grid_vector_fn,
            from_grid_vector_fn,
            to_grid_matrix_fn,
        ) = make_to_grid(mesh, basis_grid, grid_array)

    if momenta is not None:
        momenta_array = _to_jax_array(np.asarray(momenta))
        unique_angular_momenta = sorted({channel.l for channel in channels})
        matrices_by_l = {
            angular_momentum: compute_F_momentum(mesh, momenta_array, angular_momentum)
            for angular_momentum in unique_angular_momenta
        }
        fourier_stack = jnp.stack([matrices_by_l[channel.l] for channel in channels])
        transforms = TransformMatrices(
            B_grid=transforms.B_grid,
            grid_r=transforms.grid_r,
            F_momentum=fourier_stack,
            momenta=momenta_array,
        )
        fourier_fn = make_fourier(transforms)
        double_fourier_transform_fn = make_double_fourier(transforms)

    return _TransformBundle(
        matrices=transforms,
        to_grid_vector=to_grid_vector_fn,
        from_grid_vector=from_grid_vector_fn,
        to_grid_matrix=to_grid_matrix_fn,
        fourier=fourier_fn,
        double_fourier_transform=double_fourier_transform_fn,
        integrate=make_integration(mesh),
    )


def _bind_solver_observables(
    *,
    request: _CompileRequest,
    mesh: Mesh,
    operators: OperatorMatrices,
    energies: jax.Array,
    boundary: BoundaryValues | None,
    has_energy_grid: bool,
) -> _ObservableBundle:
    """Bind all runtime entry points requested for one compiled solver."""

    spectrum_fn: SpectrumKernel | None = None
    rmatrix_fn: RMatrixObservable | None = None
    smatrix_fn: SpectrumObservable | None = None
    phases_fn: SpectrumObservable | None = None
    greens_fn: GreenFunctionObservable | None = None
    wavefunction_fn: WavefunctionObservable | None = None
    eigh_fn: EigenpairAccessor | None = None
    rmatrix_grid_fn: SpectrumGridObservable | None = None
    smatrix_grid_fn: SpectrumGridObservable | None = None
    phases_grid_fn: SpectrumGridObservable | None = None

    if request.needs_spectrum:
        spectrum_fn = make_spectrum_kernel(
            mesh,
            operators,
            request.channels,
            method=request.method,
            keep_eigenvectors=request.keep_eigenvectors,
        )
        (
            bound_rmatrix,
            bound_smatrix,
            bound_phases,
            bound_greens,
            bound_wavefunction,
            bound_eigh,
        ) = bind_observables(mesh, request.channels, energies, boundary)

        # S- and phase-shift observables are built from the spectral R-matrix, so
        # the raw R observable remains available whenever matching is requested.
        rmatrix_fn = (
            bound_rmatrix if "rmatrix" in request.solvers or request.needs_boundary else None
        )
        smatrix_fn = (
            bound_smatrix if "smatrix" in request.solvers or "phases" in request.solvers else None
        )
        phases_fn = bound_phases if "phases" in request.solvers else None
        greens_fn = bound_greens if "greens" in request.solvers else None
        wavefunction_fn = bound_wavefunction if "wavefunction" in request.solvers else None
        eigh_fn = bound_eigh

        if has_energy_grid:
            (
                rmatrix_grid_fn,
                smatrix_grid_fn,
                phases_grid_fn,
            ) = bind_grid_observables(mesh, request.channels, energies, boundary)

    rmatrix_direct_fn: DirectRMatrixKernel | None = None
    rmatrix_direct_grid_fn: DirectGridObservable | None = None
    smatrix_direct_grid_fn: DirectGridObservable | None = None
    phases_direct_grid_fn: DirectGridObservable | None = None
    if "rmatrix_direct" in request.solvers:
        rmatrix_direct_fn = make_rmatrix_direct_kernel(
            mesh,
            operators,
            request.channels,
            energies,
            boundary,
        )
        if has_energy_grid:
            rmatrix_direct_grid_fn = make_rmatrix_direct_grid_observable(
                mesh,
                operators,
                request.channels,
                energies,
                boundary,
            )
            (
                smatrix_direct_grid_fn,
                phases_direct_grid_fn,
            ) = bind_direct_grid_observables(rmatrix_direct_grid_fn, boundary)

    interpolate_rmatrix_fn: InterpolatorBuilder | None = None
    interpolate_smatrix_fn: InterpolatorBuilder | None = None
    interpolate_phases_fn: InterpolatorBuilder | None = None
    if has_energy_grid:
        (
            interpolate_rmatrix_fn,
            interpolate_smatrix_fn,
            interpolate_phases_fn,
        ) = bind_interpolators(energies)

    return _ObservableBundle(
        spectrum=spectrum_fn,
        rmatrix=rmatrix_fn,
        smatrix=smatrix_fn,
        phases=phases_fn,
        greens=greens_fn,
        wavefunction=wavefunction_fn,
        eigh=eigh_fn,
        rmatrix_grid=rmatrix_grid_fn,
        smatrix_grid=smatrix_grid_fn,
        phases_grid=phases_grid_fn,
        rmatrix_direct=rmatrix_direct_fn,
        rmatrix_direct_grid=rmatrix_direct_grid_fn,
        smatrix_direct_grid=smatrix_direct_grid_fn,
        phases_direct_grid=phases_direct_grid_fn,
        interpolate_rmatrix=interpolate_rmatrix_fn,
        interpolate_smatrix=interpolate_smatrix_fn,
        interpolate_phases=interpolate_phases_fn,
    )


def _assemble_solver(
    *,
    request: _CompileRequest,
    mesh: Mesh,
    operators: OperatorMatrices,
    energies: jax.Array,
    boundary: BoundaryValues | None,
    transforms: _TransformBundle,
    observables: _ObservableBundle,
) -> Solver:
    """Assemble the final public solver bundle from normalized subcomponents."""

    expose_spectrum = (
        "spectrum" in request.solvers
        or observables.rmatrix is not None
        or observables.smatrix is not None
    )
    return Solver(
        mesh=mesh,
        operators=operators,
        channels=request.channels,
        energies=energies,
        boundary=boundary,
        transforms=transforms.matrices,
        method=request.method,
        spectrum=observables.spectrum if expose_spectrum else None,
        rmatrix=observables.rmatrix,
        smatrix=observables.smatrix,
        phases=observables.phases,
        greens=observables.greens,
        wavefunction=observables.wavefunction,
        eigh=observables.eigh,
        rmatrix_grid=observables.rmatrix_grid,
        smatrix_grid=observables.smatrix_grid,
        phases_grid=observables.phases_grid,
        rmatrix_direct=observables.rmatrix_direct,
        rmatrix_direct_grid=observables.rmatrix_direct_grid,
        smatrix_direct_grid=observables.smatrix_direct_grid,
        phases_direct_grid=observables.phases_direct_grid,
        interpolate_rmatrix=observables.interpolate_rmatrix,
        interpolate_smatrix=observables.interpolate_smatrix,
        interpolate_phases=observables.interpolate_phases,
        to_grid_vector=transforms.to_grid_vector,
        from_grid_vector=transforms.from_grid_vector,
        to_grid_matrix=transforms.to_grid_matrix,
        fourier=transforms.fourier,
        double_fourier_transform=transforms.double_fourier_transform,
        integrate=transforms.integrate,
    )


def _to_jax_array(values: np.ndarray) -> jax.Array:
    """Convert compile-time NumPy data to an explicitly typed JAX array."""

    array: jax.Array = jnp.asarray(values)
    return array


def _default_method(V_is_complex: bool) -> Method:
    """Choose the default solver method from potential type and active backend."""

    if not V_is_complex:
        return "eigh"
    return "eig" if jax.default_backend() == "cpu" else "linear_solve"


__all__ = ["compile"]

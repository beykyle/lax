"""Compile-time orchestration for solver bundles.

This module owns the "Python side" of the library described in ``DESIGN.md §14``:
it validates the requested solver features, builds mesh/operator caches, precomputes
boundary data, binds pickle-safe runtime callables, and returns the final
``Solver`` bundle used by downstream code.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import DTypeLike

from lax.boundary import compute_boundary_values_blocks
from lax.meshes import build_mesh
from lax.operators.interaction import (
    make_interaction_from_array,
    make_interaction_from_block,
    make_interaction_from_funcs,
    make_potential_builders,
)
from lax.solvers import (
    bind_grid_observables,
    bind_interpolators,
    bind_observables,
    make_direct_wavefunction_kernel,
    make_phases_direct_observable,
    make_rmatrix_direct_kernel,
    make_smatrix_direct_observable,
    make_spectrum_kernel,
)
from lax.solvers.assembly import take_block0
from lax.transforms import (
    compute_B_grid,
    compute_F_momentum,
    make_double_fourier,
    make_fourier,
    make_integration,
    make_to_grid,
)
from lax.types import (
    BoundaryValues,
    ChannelSpec,
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
    MeshSpec,
    Method,
    OperatorMatrices,
    PhasesDirectObservable,
    RMatrixObservable,
    SMatrixDirectObservable,
    Solver,
    SpectrumGridObservable,
    SpectrumKernel,
    SpectrumObservable,
    TransformMatrices,
    WavefunctionDirectObservable,
    WavefunctionObservable,
)


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
    # The symmetry-block set (DESIGN.md §15.5), or None for a channels=
    # compile.  When set, `channels` is the template block block_groups[0].
    block_groups: tuple[tuple[ChannelSpec, ...], ...] | None = None


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
    smatrix_direct: SMatrixDirectObservable | None
    phases_direct: PhasesDirectObservable | None
    wavefunction_direct: WavefunctionDirectObservable | None
    interaction_from_block: Callable[..., Any] | None
    interaction_from_array: Callable[..., Any] | None
    interaction_from_funcs: Callable[..., Any] | None
    local_potential: Callable[..., Any] | None
    nonlocal_potential: Callable[..., Any] | None
    interpolate_rmatrix: InterpolatorBuilder | None
    interpolate_smatrix: InterpolatorBuilder | None
    interpolate_phases: InterpolatorBuilder | None


def compile(
    *,
    mesh: MeshSpec,
    channels: Iterable[ChannelSpec] | None = None,
    blocks: Iterable[Iterable[ChannelSpec]] | None = None,
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
    mass_factor_grid: jax.Array | None = None,
    dtype: DTypeLike = jnp.float64,
    device: Any = None,
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
        Channel definitions for a single coupled-channel block, baked into the
        compiled solver structure.  Mutually exclusive with ``blocks``;
        exactly one must be given.
    blocks
        A batch of same-shaped symmetry blocks (independent ``(J, π)`` groups,
        partial waves, …); see DESIGN.md §15.5.  Each inner group must have
        the same length ``N_c``.  The compiled solver carries the block set on
        ``solver.blocks``, the boundary values gain a leading ``(N_b,)`` axis,
        and every observable output gains a corresponding leading block axis.
        Partial-wave batching is the ``N_c == 1`` case:
        ``blocks=[[ChannelSpec(l=0, …)], [ChannelSpec(l=1, …)], …]``.
        Mutually exclusive with ``channels``.
    operators
        Compile-time operator matrices to precompute. ``"T+L"`` is injected
        automatically whenever the requested solver path needs it.
    solvers
        Runtime entry points to expose on the returned :class:`~lax.Solver`.
        The potential passed to ``solver.spectrum(V)`` or ``solver.rmatrix_direct(V)``
        must be an :class:`~lax.Interaction`.  Build one with ``solver.local_potential(fn)``/``solver.nonlocal_potential(fn)``
        or the ``solver.interaction_from_{block,array,funcs}`` builders.
    energies
        Compile-time energy grid used for boundary-value-dependent observables and
        aligned-grid workflows.
    energy_dependent
        Whether the caller intends to provide an energy-dependent potential on
        the compile-time energy grid.  Build the potential with
        ``energy_dependent=True`` — ``solver.spectrum`` dispatches the energy
        axis internally and returns a batched ``Spectrum`` — then use
        ``solver.phases_grid(spectra)`` and the ``solver.interpolate_*``
        builders for off-grid evaluation.
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
    mass_factor_grid
        Per-energy (and optionally per-channel) ℏ²/2μ values in MeV·fm².
        Accepted shapes (all broadcast to the canonical ``(N_E, N_c)`` form):

        * ``None`` — use each channel's ``ChannelSpec.mass_factor`` uniformly.
        * scalar — the same value for all energies and channels.
        * shape ``(N_E,)`` — one value per energy, shared across channels.
        * shape ``(N_E, N_c)`` — fully independent per ``(energy, channel)`` pair.

        When provided, ``len(mass_factor_grid)`` along the first axis must equal
        ``len(energies)``.  The grid is used in two places:

        1. **Boundary values** — wave numbers and Sommerfeld parameters at each
           ``(energy, channel)`` pair use ``mass_factor_grid[ie, ic]``.
        2. **Aligned-grid direct observables** — the Hamiltonian is assembled
           with the per-energy per-channel mass factor at each grid point.
    dtype
        Floating-point precision for the baked arrays (mesh, operators,
        boundary values, energy grid, transforms).  Default ``jnp.float64``;
        complex caches use the matching complex dtype (``complex64`` for
        ``float32``).  x64 itself is enabled globally via ``jax.config``;
        ``dtype`` only selects the precision of the compile-time caches.
        Runtime kernels compute in the promoted dtype of the baked arrays and
        the supplied :class:`~lax.Interaction` — build interactions through the
        solver's own builders to stay in the requested precision.
    device
        Optional device (or device-platform string such as ``"cpu"``/``"gpu"``)
        on which to place the compiled solver's cached arrays via
        ``jax.device_put``.  ``None`` keeps JAX's default placement.

    Returns
    -------
    Solver
        A pickle-safe solver bundle containing compile-time caches and bound
        runtime observables.
    """

    request = _resolve_compile_request(
        channels=channels,
        blocks=blocks,
        operators=operators,
        solvers=solvers,
        energies=energies,
        energy_dependent=energy_dependent,
        method=method,
        V_is_complex=V_is_complex,
    )
    if request.block_groups is not None and momenta is not None:
        msg = (
            "`momenta` Fourier transforms are not supported with `blocks=`; "
            "compile per-block solvers for momentum-space work."
        )
        raise ValueError(msg)
    mesh_data, operator_matrices = _build_solver_mesh(
        mesh=mesh,
        operators=request.operators,
        needs_spectrum=request.needs_spectrum,
        method=request.method,
        grid=grid,
        momenta=momenta,
        in_blocks_mode=request.block_groups is not None,
    )
    mass_factor_grid_np: np.ndarray | None
    if mass_factor_grid is not None:
        if energies is None:
            msg = "`energies` is required when `mass_factor_grid` is provided."
            raise ValueError(msg)
        n_e = len(np.asarray(energies))
        n_c = len(request.channels)
        mass_factor_grid_np = _broadcast_mass_factor_grid(
            np.asarray(mass_factor_grid, dtype=np.float64), n_e, n_c
        )
    else:
        mass_factor_grid_np = None
    # Everything downstream runs the batched (blocks-always) machinery: a
    # channels= compile is the single block (channels,) with block_mode=False.
    block_mode = request.block_groups is not None
    blocks_resolved = request.block_groups if block_mode else (request.channels,)
    assert blocks_resolved is not None
    boundary, energies_array = _prepare_boundary_data(
        blocks=blocks_resolved,
        block_mode=block_mode,
        energies=energies,
        channel_radius=mesh.scale,
        z1z2=z1z2,
        dps=dps,
        mass_factor_grid=mass_factor_grid_np,
    )
    # Cast and place every compile-time cache before any kernel/observable
    # factory runs, so all derived arrays inherit dtype/device (§14.1).
    target_device = _resolve_device(device)
    mesh_data = _finalize_arrays(mesh_data, dtype, target_device)
    operator_matrices = _finalize_arrays(operator_matrices, dtype, target_device)
    boundary = _finalize_arrays(boundary, dtype, target_device)
    energies_array = _finalize_arrays(energies_array, dtype, target_device)
    grid = _finalize_arrays(grid, dtype, target_device) if grid is not None else None
    momenta = _finalize_arrays(momenta, dtype, target_device) if momenta is not None else None
    transforms = _prepare_transforms(
        mesh=mesh_data,
        channels=request.channels,
        grid=grid,
        momenta=momenta,
    )
    mass_factor_grid_jax = (
        _finalize_arrays(jnp.asarray(mass_factor_grid_np), dtype, target_device)
        if mass_factor_grid_np is not None
        else None
    )
    observables = _bind_solver_observables(
        request=request,
        blocks=blocks_resolved,
        block_mode=block_mode,
        mesh=mesh_data,
        operators=operator_matrices,
        energies=energies_array,
        boundary=boundary,
        has_energy_grid=energies is not None,
        mass_factor_grid=mass_factor_grid_jax,
    )
    return _assemble_solver(
        request=request,
        mesh=mesh_data,
        operators=operator_matrices,
        energies=energies_array,
        boundary=boundary,
        transforms=transforms,
        observables=observables,
        mass_factor_grid=mass_factor_grid_jax,
    )


def _resolve_compile_request(
    *,
    channels: Iterable[ChannelSpec] | None,
    blocks: Iterable[Iterable[ChannelSpec]] | None,
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

    if (channels is None) == (blocks is None):
        msg = "Pass exactly one of `channels` or `blocks` to lax.compile()."
        raise ValueError(msg)
    block_groups: tuple[tuple[ChannelSpec, ...], ...] | None
    if blocks is not None:
        block_groups = tuple(tuple(group) for group in blocks)
        if not block_groups:
            msg = "`blocks` must contain at least one symmetry block."
            raise ValueError(msg)
        n_c = len(block_groups[0])
        if n_c == 0 or any(len(group) != n_c for group in block_groups):
            msg = "All `blocks` must share the same non-zero channel shape N_c."
            raise ValueError(msg)
        # The template block: shared shape data (N_c, mesh pairing) comes from
        # here; per-block centrifugal/threshold/μ rows are stacked separately.
        channels_tuple = block_groups[0]
    else:
        block_groups = None
        channels_tuple = tuple(cast(Iterable[ChannelSpec], channels))
    operators_set = set(operators)
    solvers_set = frozenset(solvers)

    selected_method = method or _default_method(V_is_complex)
    if selected_method not in {"eigh", "eig", "linear_solve"}:
        msg = f"Method {selected_method!r} is not implemented in the MVP compile() path."
        raise ValueError(msg)

    # "wavefunction" is served by the spectral path under eigh/eig, but by the
    # direct wavefunction_direct kernel under linear_solve (§14, Example 16.8).
    wants_wavefunction = "wavefunction" in solvers_set
    wavefunction_via_spectrum = wants_wavefunction and selected_method in {
        "eigh",
        "eig",
    }
    wavefunction_via_direct = wants_wavefunction and selected_method == "linear_solve"

    needs_spectrum = (
        bool(solvers_set & {"spectrum", "rmatrix", "smatrix", "phases", "greens"})
        or wavefunction_via_spectrum
    )
    needs_boundary = bool(solvers_set & {"smatrix", "phases", "rmatrix_direct"})
    # wavefunction_direct indexes the compile-time energy grid, so it needs one.
    if (needs_boundary or energy_dependent or wavefunction_via_direct) and energies is None:
        msg = "`energies` is required for continuum solvers or energy-dependent potentials."
        raise ValueError(msg)
    if needs_spectrum or "rmatrix_direct" in solvers_set or wavefunction_via_direct:
        operators_set.add("T+L")

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
        keep_eigenvectors=bool(solvers_set & {"greens"}) or wavefunction_via_spectrum,
        block_groups=block_groups,
    )


def _build_solver_mesh(
    *,
    mesh: MeshSpec,
    operators: frozenset[str],
    needs_spectrum: bool,
    method: Method,
    grid: jax.Array | None,
    momenta: jax.Array | None,
    in_blocks_mode: bool = False,
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
    if mesh_data.n_intervals > 1 and in_blocks_mode:
        msg = (
            "Symmetry-block batching (`blocks=`) is not supported on propagated "
            "meshes; compile per-block solvers instead."
        )
        raise ValueError(msg)
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
    blocks: tuple[tuple[ChannelSpec, ...], ...],
    block_mode: bool,
    energies: jax.Array | None,
    channel_radius: float,
    z1z2: tuple[int, int] | None,
    dps: int,
    mass_factor_grid: np.ndarray | None = None,
) -> tuple[BoundaryValues | None, jax.Array]:
    """Prepare compile-time boundary values and the canonical energy-grid array.

    The boundary values are always computed stacked per symmetry block on a
    leading ``(N_b,)`` axis (DESIGN.md §15.5); for a ``channels=`` compile the
    ``N_b == 1`` axis is squeezed off for the public ``Solver.boundary`` shape.
    """

    if energies is None:
        # The solver always stores an energy array so downstream code can rely on a
        # uniform bundle shape even when no boundary-valued observables are exposed.
        empty_energies = np.zeros((0,), dtype=np.float64)
        return None, jnp.asarray(empty_energies)

    energies_np = np.asarray(energies)
    boundary = compute_boundary_values_blocks(
        block_groups=blocks,
        energies=energies_np,
        channel_radius=channel_radius,
        z1z2=z1z2,
        dps=dps,
        mass_factor_grid=mass_factor_grid,
    )
    if not block_mode:
        boundary = take_block0(boundary)
    return boundary, jnp.asarray(energies_np)


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
        grid_array = jnp.asarray(np.asarray(grid))
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
        momenta_array = jnp.asarray(np.asarray(momenta))
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
    blocks: tuple[tuple[ChannelSpec, ...], ...],
    block_mode: bool,
    mesh: Mesh,
    operators: OperatorMatrices,
    energies: jax.Array,
    boundary: BoundaryValues | None,
    has_energy_grid: bool,
    mass_factor_grid: jax.Array | None = None,
) -> _ObservableBundle:
    """Bind all runtime entry points requested for one compiled solver.

    Every kernel is batched over the symmetry-block axis (DESIGN.md §15.5);
    a ``channels=`` compile passes the single block ``(channels,)`` with
    ``block_mode=False``.
    """

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
            blocks,
            method=request.method,
            keep_eigenvectors=request.keep_eigenvectors,
            block_mode=block_mode,
        )
        (
            bound_rmatrix,
            bound_smatrix,
            bound_phases,
            bound_greens,
            bound_wavefunction,
            bound_eigh,
        ) = bind_observables(mesh, blocks, energies, boundary, block_mode=block_mode)

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
            ) = bind_grid_observables(
                mesh,
                blocks,
                energies,
                boundary,
                mass_factor_grid=mass_factor_grid,
                block_mode=block_mode,
            )

    rmatrix_direct_fn: DirectRMatrixKernel | None = None
    smatrix_direct_fn: SMatrixDirectObservable | None = None
    phases_direct_fn: PhasesDirectObservable | None = None
    wavefunction_direct_fn: WavefunctionDirectObservable | None = None
    if "rmatrix_direct" in request.solvers:
        rmatrix_direct_fn = make_rmatrix_direct_kernel(
            mesh,
            operators,
            blocks,
            energies,
            boundary,
            mass_factor_grid,
            block_mode=block_mode,
        )
        smatrix_direct_fn = make_smatrix_direct_observable(
            rmatrix_direct_fn, boundary, block_mode=block_mode
        )
        phases_direct_fn = make_phases_direct_observable(smatrix_direct_fn)
    # wavefunction_direct is available whenever the direct path is active
    # ("rmatrix_direct"), and is the binding for "wavefunction" under
    # method="linear_solve" (§14, Example 16.8) where no spectral path exists.
    wavefunction_via_direct = "wavefunction" in request.solvers and request.method == "linear_solve"
    if "rmatrix_direct" in request.solvers or wavefunction_via_direct:
        wavefunction_direct_fn = make_direct_wavefunction_kernel(
            mesh,
            operators,
            blocks,
            energies,
            mass_factor_grid,
            block_mode=block_mode,
        )

    interpolate_rmatrix_fn: InterpolatorBuilder | None = None
    interpolate_smatrix_fn: InterpolatorBuilder | None = None
    interpolate_phases_fn: InterpolatorBuilder | None = None
    if has_energy_grid:
        (
            interpolate_rmatrix_fn,
            interpolate_smatrix_fn,
            interpolate_phases_fn,
        ) = bind_interpolators(
            energies,
            n_blocks=len(blocks) if block_mode else None,
        )

    n_blocks = len(blocks) if block_mode else None
    interaction_from_block_fn = make_interaction_from_block(
        mesh, request.channels, energies, n_blocks=n_blocks
    )
    interaction_from_array_fn = make_interaction_from_array(
        mesh, request.channels, energies, n_blocks=n_blocks
    )
    interaction_from_funcs_fn = make_interaction_from_funcs(
        mesh, request.channels, energies, n_blocks=n_blocks
    )
    local_potential_fn, nonlocal_potential_fn = make_potential_builders(
        mesh, request.channels, energies, n_blocks=n_blocks
    )

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
        smatrix_direct=smatrix_direct_fn,
        phases_direct=phases_direct_fn,
        wavefunction_direct=wavefunction_direct_fn,
        interaction_from_block=interaction_from_block_fn,
        interaction_from_array=interaction_from_array_fn,
        interaction_from_funcs=interaction_from_funcs_fn,
        local_potential=local_potential_fn,
        nonlocal_potential=nonlocal_potential_fn,
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
    mass_factor_grid: jax.Array | None = None,
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
        mass_factor_grid=mass_factor_grid,
        blocks=request.block_groups,
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
        smatrix_direct=observables.smatrix_direct,
        phases_direct=observables.phases_direct,
        wavefunction_direct=observables.wavefunction_direct,
        interaction_from_block=observables.interaction_from_block,
        interaction_from_array=observables.interaction_from_array,
        interaction_from_funcs=observables.interaction_from_funcs,
        local_potential=observables.local_potential,
        nonlocal_potential=observables.nonlocal_potential,
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


def _resolve_device(device: Any) -> Any:
    """Resolve a device-platform string (e.g. ``"cpu"``) to a concrete device."""

    if device is None or isinstance(device, jax.Device):
        return device
    return cast(Any, jax.devices(device)[0])


def _finalize_arrays[T](tree: T, dtype: DTypeLike, device: Any) -> T:
    """Cast a compile-time cache pytree to ``dtype`` and place it on ``device``.

    Floating leaves are cast to ``dtype``; complex leaves to the matching
    complex dtype (``complex64`` for ``float32``); bool/int leaves are left
    alone.  Applied to every cache *before* the kernel/observable factories
    run, so all derived arrays (surface projectors, Gauss scaling, stacked
    centrifugal rows) inherit the precision and placement automatically.
    """

    target = jnp.dtype(dtype)
    if not jnp.issubdtype(target, jnp.floating):
        msg = f"`dtype` must be a floating-point dtype, got {target}."
        raise ValueError(msg)
    complex_target = jnp.dtype(
        jnp.complex64 if target == jnp.dtype(jnp.float32) else jnp.complex128
    )

    def finalize_leaf(leaf: object) -> object:
        if not isinstance(leaf, jax.Array | np.ndarray):
            return leaf
        array = jnp.asarray(leaf)
        if jnp.issubdtype(array.dtype, jnp.complexfloating):
            array = array.astype(complex_target)
        elif jnp.issubdtype(array.dtype, jnp.floating):
            array = array.astype(target)
        if device is not None:
            array = jax.device_put(array, device)
        return array

    return jax.tree.map(finalize_leaf, tree)


def _broadcast_mass_factor_grid(
    arr: np.ndarray,
    n_energies: int,
    n_channels: int,
) -> np.ndarray:
    """Broadcast user-supplied mass_factor_grid to canonical (N_E, N_c) shape.

    Accepts scalar, ``(N_E,)``, or ``(N_E, N_c)`` input.  Returns a
    C-contiguous ``float64`` array of shape ``(N_E, N_c)``.
    """

    if arr.ndim == 0:
        return np.full((n_energies, n_channels), float(arr), dtype=np.float64)
    if arr.ndim == 1:
        if arr.shape[0] != n_energies:
            msg = (
                f"`mass_factor_grid` length {arr.shape[0]} must equal "
                f"`energies` length {n_energies}."
            )
            raise ValueError(msg)
        return np.broadcast_to(arr[:, None], (n_energies, n_channels)).copy()
    if arr.ndim == 2:
        if arr.shape != (n_energies, n_channels):
            msg = (
                f"`mass_factor_grid` shape {arr.shape} must be "
                f"({n_energies}, {n_channels}) = (N_E, N_c)."
            )
            raise ValueError(msg)
        return arr.copy()
    msg = (
        "`mass_factor_grid` must be scalar, shape (N_E,), or shape (N_E, N_c), "
        f"got ndim={arr.ndim}."
    )
    raise ValueError(msg)


def _default_method(V_is_complex: bool) -> Method:
    """Choose the default solver method from potential type and active backend."""

    if not V_is_complex:
        return "eigh"
    return "eig" if jax.default_backend() == "cpu" else "linear_solve"


__all__ = ["compile"]

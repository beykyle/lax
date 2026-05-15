"""Public solver factory."""

from __future__ import annotations

from collections.abc import Iterable

import jax
import jax.numpy as jnp
import numpy as np

from lax.boundary import compute_boundary_values
from lax.boundary._types import Solver, TransformMatrices
from lax.meshes import build_mesh
from lax.solvers import bind_observables, make_rmatrix_direct_kernel, make_spectrum_kernel
from lax.transforms import (
    compute_B_grid,
    compute_F_momentum,
    make_double_fourier,
    make_fourier,
    make_integration,
    make_to_grid,
)
from lax.types import ChannelSpec, MeshSpec, Method


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
    """Build a solver bundle specialized to one mesh and channel structure."""

    channels_tuple = tuple(channels)
    operators_set = set(operators)
    solvers_set = set(solvers)
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

    mesh_data, operator_matrices = build_mesh(
        family=mesh.family,
        regularization=mesh.regularization,
        n=mesh.n,
        scale=mesh.scale,
        operators=operators_set,
        **mesh.extras,
    )

    if energies is not None:
        energies_np = np.asarray(energies)
        boundary = compute_boundary_values(
            channels=channels_tuple,
            energies=energies_np,
            channel_radius=mesh.scale,
            z1z2=z1z2,
            dps=dps,
        )
        energies_array = _to_jax_array(energies_np)
    else:
        boundary = None
        energies_array = _to_jax_array(np.zeros((0,), dtype=np.float64))

    transforms = TransformMatrices()
    to_grid_vector_fn = None
    from_grid_vector_fn = None
    to_grid_matrix_fn = None
    fourier_fn = None
    double_fourier_transform_fn = None
    if grid is not None:
        grid_array = _to_jax_array(np.asarray(grid))
        basis_grid = compute_B_grid(mesh_data, grid_array)
        transforms = TransformMatrices(
            B_grid=basis_grid,
            grid_r=grid_array,
            F_momentum=transforms.F_momentum,
            momenta=transforms.momenta,
        )
        to_grid_vector_fn, from_grid_vector_fn, to_grid_matrix_fn = make_to_grid(
            mesh_data,
            basis_grid,
            grid_array,
        )
    if momenta is not None:
        momenta_array = _to_jax_array(np.asarray(momenta))
        unique_angular_momenta = sorted({channel.l for channel in channels_tuple})
        matrices_by_l = {
            angular_momentum: compute_F_momentum(mesh_data, momenta_array, angular_momentum)
            for angular_momentum in unique_angular_momenta
        }
        fourier_stack = jnp.stack([matrices_by_l[channel.l] for channel in channels_tuple])
        transforms = TransformMatrices(
            B_grid=transforms.B_grid,
            grid_r=transforms.grid_r,
            F_momentum=fourier_stack,
            momenta=momenta_array,
        )
        fourier_fn = make_fourier(transforms)
        double_fourier_transform_fn = make_double_fourier(transforms)
    integrate_fn = make_integration(mesh_data)

    keep_eigenvectors = bool(solvers_set & {"greens", "wavefunction"})
    spectrum_fn = None
    if needs_spectrum:
        spectrum_fn = make_spectrum_kernel(
            mesh_data,
            operator_matrices,
            channels_tuple,
            method=selected_method,
            keep_eigenvectors=keep_eigenvectors,
        )

    rmatrix_fn = None
    smatrix_fn = None
    phases_fn = None
    greens_fn = None
    wavefunction_fn = None
    eigh_fn = None
    rmatrix_direct_fn = None
    if spectrum_fn is not None:
        (
            bound_rmatrix,
            bound_smatrix,
            bound_phases,
            bound_greens,
            bound_wavefunction,
            bound_eigh,
        ) = bind_observables(
            mesh_data,
            channels_tuple,
            energies_array,
            boundary,
        )
        rmatrix_fn = bound_rmatrix if "rmatrix" in solvers_set or needs_boundary else None
        smatrix_fn = bound_smatrix if "smatrix" in solvers_set or "phases" in solvers_set else None
        phases_fn = bound_phases if "phases" in solvers_set else None
        greens_fn = bound_greens if "greens" in solvers_set else None
        wavefunction_fn = bound_wavefunction if "wavefunction" in solvers_set else None
        eigh_fn = bound_eigh if keep_eigenvectors else bound_eigh

    if "rmatrix_direct" in solvers_set:
        rmatrix_direct_fn = make_rmatrix_direct_kernel(
            mesh_data,
            operator_matrices,
            channels_tuple,
            energies_array,
        )

    return Solver(
        mesh=mesh_data,
        operators=operator_matrices,
        channels=channels_tuple,
        energies=energies_array,
        boundary=boundary,
        transforms=transforms,
        method=selected_method,
        spectrum=spectrum_fn
        if "spectrum" in solvers_set or rmatrix_fn is not None or smatrix_fn is not None
        else None,
        rmatrix=rmatrix_fn,
        smatrix=smatrix_fn,
        phases=phases_fn,
        greens=greens_fn,
        wavefunction=wavefunction_fn,
        eigh=eigh_fn,
        rmatrix_direct=rmatrix_direct_fn,
        to_grid_vector=to_grid_vector_fn,
        from_grid_vector=from_grid_vector_fn,
        to_grid_matrix=to_grid_matrix_fn,
        fourier=fourier_fn,
        double_fourier_transform=double_fourier_transform_fn,
        integrate=integrate_fn,
    )


__all__ = ["compile"]


def _to_jax_array(values: np.ndarray) -> jax.Array:
    """Convert compile-time NumPy data to an explicitly typed JAX array."""

    array: jax.Array = jnp.asarray(values)  # pyright: ignore[reportUnknownMemberType] -- JAX stubs expose asarray imprecisely for NumPy inputs.
    return array


def _default_method(V_is_complex: bool) -> Method:
    """Choose the default continuum method from the backend and potential type."""

    if not V_is_complex:
        return "eigh"
    return "eig" if jax.default_backend() == "cpu" else "linear_solve"

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import lax as lm
from lax.meshes import build_mesh
from lax.meshes._basis_eval import basis_at
from lax.transforms.grid import compute_B_grid

pytest.importorskip("jax")
pytest.importorskip("scipy")


def test_compute_b_grid_matches_basis_evaluation() -> None:
    """`compute_B_grid` agrees with the registered Legendre-x basis evaluator."""

    mesh, _ = build_mesh("legendre", "x", n=5, scale=6.0, operators={"T+L"})
    grid = jnp.linspace(0.2, 5.8, 11)

    direct = np.asarray(basis_at(mesh, grid))
    computed = np.asarray(compute_B_grid(mesh, grid))

    assert computed.shape == (11, 5)
    assert np.allclose(computed, direct)


def test_compile_binds_to_grid_transforms() -> None:
    """`compile()` exposes vector and matrix grid transforms when a grid is requested."""

    solver = lm.compile(
        mesh=lm.MeshSpec("legendre", "x", n=5, scale=6.0),
        channels=(lm.ChannelSpec(l=0, threshold=0.0, mass_factor=2.0),),
        solvers=("spectrum",),
        grid=jnp.linspace(0.2, 5.8, 11),
    )

    assert solver.transforms.B_grid is not None
    assert solver.transforms.grid_r is not None
    assert solver.to_grid_vector is not None
    assert solver.from_grid_vector is not None
    assert solver.to_grid_matrix is not None

    values = jnp.asarray([1.0, -0.5, 0.25, 0.0, 0.5])
    kernel = jnp.eye(5)

    projected_values = np.asarray(solver.to_grid_vector(values))
    recovered_values = np.asarray(solver.from_grid_vector(projected_values))
    projected_kernel = np.asarray(solver.to_grid_matrix(kernel))

    assert projected_values.shape == (11,)
    assert recovered_values.shape == (5,)
    assert projected_kernel.shape == (11, 11)


def test_from_grid_vector_round_trip_on_nodal_grid() -> None:
    """`from_grid_vector` exactly inverts nodal Lagrange sampling on the mesh radii."""

    solver = lm.compile(
        mesh=lm.MeshSpec("legendre", "x", n=6, scale=6.0),
        channels=(lm.ChannelSpec(l=0, threshold=0.0, mass_factor=2.0),),
        solvers=(),
        grid=build_mesh("legendre", "x", n=6, scale=6.0, operators={"T+L"})[0].radii,
    )

    assert solver.to_grid_vector is not None
    assert solver.from_grid_vector is not None

    coefficients = jnp.asarray([1.0, -0.5, 0.75, -0.25, 0.125, 0.0])
    sampled = np.asarray(solver.to_grid_vector(coefficients))
    recovered = np.asarray(solver.from_grid_vector(jnp.asarray(sampled)))

    assert np.allclose(recovered, np.asarray(coefficients), atol=1.0e-12, rtol=1.0e-12)


def test_from_grid_vector_accepts_callable_profile() -> None:
    """`from_grid_vector` treats callable and sampled-array inputs equivalently."""

    grid = jnp.linspace(0.2, 5.8, 19)
    solver = lm.compile(
        mesh=lm.MeshSpec("legendre", "x", n=5, scale=6.0),
        channels=(lm.ChannelSpec(l=0, threshold=0.0, mass_factor=2.0),),
        solvers=(),
        grid=grid,
    )

    assert solver.to_grid_vector is not None
    assert solver.from_grid_vector is not None
    assert solver.transforms.grid_r is not None

    def profile(radii: jax.Array) -> jax.Array:
        return jnp.exp(-0.5 * radii) * radii

    expected = np.asarray(profile(solver.transforms.grid_r))
    callable_coefficients = np.asarray(solver.from_grid_vector(profile))
    sampled_coefficients = np.asarray(solver.from_grid_vector(jnp.asarray(expected)))
    reconstructed = np.asarray(solver.to_grid_vector(jnp.asarray(callable_coefficients)))

    assert np.allclose(callable_coefficients, sampled_coefficients, atol=1.0e-12, rtol=1.0e-12)
    assert np.allclose(reconstructed, np.asarray(solver.to_grid_vector(jnp.asarray(sampled_coefficients))))


def test_legendre_to_grid_preserves_norm_for_bound_state() -> None:
    """Legendre-x grid projection preserves the norm of a normalized eigenvector."""

    solver = lm.compile(
        mesh=lm.MeshSpec("legendre", "x", n=12, scale=8.0),
        channels=(lm.ChannelSpec(l=0, threshold=0.0, mass_factor=2.0),),
        operators=("T+L",),
        solvers=("spectrum", "wavefunction"),
        grid=jnp.linspace(1.0e-3, 7.999, 4000),
    )
    potential = jnp.asarray(
        [[[-2.5, -2.2, -1.8, -1.3, -0.9, -0.6, -0.35, -0.2, -0.1, -0.05, -0.02, -0.01]]]
    )

    assert solver.spectrum is not None
    assert solver.to_grid_vector is not None
    assert solver.transforms.grid_r is not None

    spectrum = solver.spectrum(potential)
    assert spectrum.eigenvectors is not None
    eigenvector = np.asarray(spectrum.eigenvectors)[:, 0]
    grid_values = np.asarray(solver.to_grid_vector(jnp.asarray(eigenvector)))
    radii = np.asarray(solver.transforms.grid_r)

    mesh_norm = float(np.vdot(eigenvector, eigenvector).real)
    grid_norm = float(np.trapezoid(np.abs(grid_values) ** 2, radii))

    assert abs(mesh_norm - 1.0) < 1.0e-12
    assert abs(grid_norm - mesh_norm) < 5.0e-3

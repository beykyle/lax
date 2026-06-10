"""Tests for make_interaction_from_{block,array,funcs} builder factories."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import lax as lm
from lax.operators.potential import assemble_local, assemble_nonlocal

pytest.importorskip("jax")


def _make_mesh_channels():
    mesh = lm.MeshSpec("legendre", "x", n=6, scale=5.0)
    channels = (lm.ChannelSpec(l=0, threshold=0.0, mass_factor=41.472),)
    energies = jnp.asarray([1.0, 3.0])
    solver = lm.compile(
        mesh=mesh,
        channels=channels,
        operators=("T+L",),
        solvers=("rmatrix_direct",),
        energies=energies,
        energy_dependent=True,
    )
    return solver


# ---------------------------------------------------------------------------
# make_interaction_from_block
# ---------------------------------------------------------------------------


def test_interaction_from_block_wraps_matrix() -> None:
    """Wrapping a (M, M) block stores it verbatim with energy_dependent=False."""

    solver = _make_mesh_channels()
    N = solver.mesh.n
    M = N * len(solver.channels)

    V_block = jnp.eye(M) * 0.5
    assert solver.interaction_from_block is not None
    interaction = solver.interaction_from_block(V_block, energy_dependent=False)

    assert interaction.energy_dependent is False
    assert np.allclose(np.asarray(interaction.block), np.asarray(V_block))


def test_interaction_from_block_energy_dependent() -> None:
    """Energy-dependent (N_E, M, M) blocks are stored with energy_dependent=True."""

    solver = _make_mesh_channels()
    N_E = len(solver.energies)
    M = solver.mesh.n * len(solver.channels)

    V_block = jnp.ones((N_E, M, M)) * 0.1
    assert solver.interaction_from_block is not None
    interaction = solver.interaction_from_block(V_block, energy_dependent=True)

    assert interaction.energy_dependent is True
    assert interaction.block.shape == (N_E, M, M)


def test_interaction_from_block_rejects_wrong_shape() -> None:
    """Wrong block shape raises ValueError."""

    solver = _make_mesh_channels()
    M = solver.mesh.n * len(solver.channels)

    with pytest.raises(ValueError, match="shape"):
        assert solver.interaction_from_block is not None
        solver.interaction_from_block(jnp.zeros((M + 1, M)), energy_dependent=False)


# ---------------------------------------------------------------------------
# make_interaction_from_array — local term round-trip
# ---------------------------------------------------------------------------


def test_interaction_from_array_local_matches_assemble_local() -> None:
    """Local term builds the same (M, M) diagonal block as assemble_local."""

    solver = _make_mesh_channels()
    radii = np.asarray(solver.mesh.radii)
    g = jnp.asarray(np.exp(-0.5 * radii))  # (N,) form-factor

    # Interaction builder: one local term with coupling A = [[1.0]]
    A = np.array([[1.0]])
    assert solver.interaction_from_array is not None
    interaction = solver.interaction_from_array(
        local=[(g, A)],
        energy_dependent=False,
    )

    # assemble_local returns (1, 1, N); the block should be diag(g)
    V_raw = assemble_local(solver.mesh, lambda r: g)  # (1, 1, N)
    expected = np.diag(np.asarray(g))

    assert np.allclose(np.asarray(interaction.block), expected, atol=1e-13)
    # Also verify it matches the V_raw diagonal
    assert np.allclose(np.asarray(interaction.block), np.diag(np.asarray(V_raw[0, 0])), atol=1e-13)


# ---------------------------------------------------------------------------
# make_interaction_from_array — nonlocal term round-trip
# ---------------------------------------------------------------------------


def test_interaction_from_array_nonlocal_matches_assemble_nonlocal() -> None:
    """Nonlocal term builds the same (M, M) block as assemble_nonlocal."""

    solver = _make_mesh_channels()
    radii = np.asarray(solver.mesh.radii)
    ri, rj = np.meshgrid(radii, radii, indexing="ij")
    K = jnp.asarray(np.exp(-0.5 * (ri + rj)))  # (N, N) kernel values

    A = np.array([[1.0]])
    assert solver.interaction_from_array is not None
    interaction = solver.interaction_from_array(
        nonlocal_=[(K, A)],
        energy_dependent=False,
    )

    # assemble_nonlocal applies sqrt(w_i * w_j) * a scaling; the block should match
    V_raw = assemble_nonlocal(
        solver.mesh, lambda r1, r2: jnp.asarray(np.exp(-0.5 * (np.asarray(r1) + np.asarray(r2))))
    )
    expected = np.asarray(V_raw[0, 0])  # (N, N)

    assert np.allclose(np.asarray(interaction.block), expected, atol=1e-13)


# ---------------------------------------------------------------------------
# make_interaction_from_array — symmetry validation
# ---------------------------------------------------------------------------


def test_interaction_from_array_rejects_asymmetric_coupling() -> None:
    """Asymmetric coupling matrix A raises ValueError."""

    solver = lm.compile(
        mesh=lm.MeshSpec("legendre", "x", n=4, scale=5.0),
        channels=(
            lm.ChannelSpec(l=0, threshold=0.0, mass_factor=41.472),
            lm.ChannelSpec(l=0, threshold=0.0, mass_factor=41.472),
        ),
        operators=("T+L",),
        solvers=("rmatrix_direct",),
        energies=jnp.asarray([1.0]),
        energy_dependent=True,
    )
    N = solver.mesh.n
    g = jnp.ones(N)
    A_asymmetric = np.array([[1.0, 2.0], [0.0, 1.0]])  # not symmetric

    assert solver.interaction_from_array is not None
    with pytest.raises(ValueError, match="symmetric"):
        solver.interaction_from_array(local=[(g, A_asymmetric)], energy_dependent=False)


# ---------------------------------------------------------------------------
# make_interaction_from_array — energy-dependent
# ---------------------------------------------------------------------------


def test_interaction_from_array_energy_dependent_shape() -> None:
    """energy_dependent=True produces block of shape (N_E, M, M)."""

    solver = _make_mesh_channels()
    N_E = len(solver.energies)
    N = solver.mesh.n

    g_grid = jnp.ones((N_E, N))  # (N_E, N) energy-dependent local form-factor
    A = np.array([[1.0]])

    assert solver.interaction_from_array is not None
    interaction = solver.interaction_from_array(
        local=[(g_grid, A)],
        energy_dependent=True,
    )

    M = N * len(solver.channels)
    assert interaction.energy_dependent is True
    assert interaction.block.shape == (N_E, M, M)


# ---------------------------------------------------------------------------
# make_interaction_from_funcs — matches interaction_from_array
# ---------------------------------------------------------------------------


def test_interaction_from_funcs_matches_from_array() -> None:
    """interaction_from_funcs evaluating a lambda equals interaction_from_array with sampled values."""

    solver = _make_mesh_channels()
    radii = np.asarray(solver.mesh.radii)

    def local_fn(r: jnp.ndarray) -> jnp.ndarray:
        return jnp.asarray(np.exp(-0.5 * np.asarray(r)))

    g_sampled = local_fn(jnp.asarray(radii))
    A = np.array([[1.0]])

    assert solver.interaction_from_funcs is not None
    assert solver.interaction_from_array is not None

    interaction_funcs = solver.interaction_from_funcs(
        local=[(local_fn, A)],
        energy_dependent=False,
    )
    interaction_array = solver.interaction_from_array(
        local=[(g_sampled, A)],
        energy_dependent=False,
    )

    assert np.allclose(
        np.asarray(interaction_funcs.block),
        np.asarray(interaction_array.block),
        atol=1e-13,
    )


# ---------------------------------------------------------------------------
# End-to-end: rmatrix_direct(interaction) matches rmatrix_direct(raw_V)
# ---------------------------------------------------------------------------


def test_rmatrix_direct_interaction_round_trip() -> None:
    """rmatrix_direct(Interaction) produces physically correct R-matrix values."""

    alpha, beta = 0.2316053, 1.3918324
    HBAR2_2MU = 41.472

    def yamaguchi_kernel(r1: jnp.ndarray, r2: jnp.ndarray) -> jnp.ndarray:
        return -2.0 * beta * (alpha + beta) ** 2 * jnp.exp(-beta * (r1 + r2)) * HBAR2_2MU

    solver = lm.compile(
        mesh=lm.MeshSpec("legendre", "x", n=10, scale=8.0),
        channels=(lm.ChannelSpec(l=0, threshold=0.0, mass_factor=HBAR2_2MU),),
        operators=("T+L",),
        solvers=("rmatrix_direct",),
        energies=jnp.asarray([0.1, 5.0]),
    )

    assert solver.rmatrix_direct is not None
    assert solver.potential is not None

    # Build via solver.potential (canonical API)
    V = solver.potential(yamaguchi_kernel)
    r_from_potential = np.asarray(solver.rmatrix_direct(V))

    # Build via interaction_from_block (pre-assembled with Gauss scaling applied manually)
    ri, rj = jnp.meshgrid(solver.mesh.radii, solver.mesh.radii, indexing="ij")
    wi, wj = jnp.meshgrid(solver.mesh.weights, solver.mesh.weights, indexing="ij")
    scaled_block = yamaguchi_kernel(ri, rj) * (jnp.sqrt(wi * wj) * solver.mesh.scale)
    assert solver.interaction_from_block is not None
    interaction = solver.interaction_from_block(scaled_block, energy_dependent=False)
    r_from_block = np.asarray(solver.rmatrix_direct(interaction))

    assert np.allclose(r_from_potential, r_from_block, atol=1.0e-12, rtol=1.0e-12)

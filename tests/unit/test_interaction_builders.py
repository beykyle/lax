"""Tests for make_interaction_from_{block,array,funcs} builder factories."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import lax as lm

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


def _is_block_diagonal(block: jnp.ndarray) -> bool:
    """Return True when ``block`` is (numerically) diagonal."""

    arr = np.asarray(block)
    return bool(np.allclose(arr, np.diag(np.diag(arr))))


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
    """Local term builds the correct (M, M) diagonal block."""

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

    # Local term with A=[[1]] should produce diag(g) as the (M, M) block
    expected = np.diag(np.asarray(g))
    assert np.allclose(np.asarray(interaction.block), expected, atol=1e-13)


# ---------------------------------------------------------------------------
# make_interaction_from_array — nonlocal term round-trip
# ---------------------------------------------------------------------------


def test_interaction_from_array_nonlocal_matches_assemble_nonlocal() -> None:
    """Nonlocal term builds the correct Gauss-scaled (M, M) block."""

    solver = _make_mesh_channels()
    radii = np.asarray(solver.mesh.radii)
    weights = np.asarray(solver.mesh.weights)
    a = float(solver.mesh.scale)
    ri, rj = np.meshgrid(radii, radii, indexing="ij")
    K = jnp.asarray(np.exp(-0.5 * (ri + rj)))  # (N, N) kernel values

    A = np.array([[1.0]])
    assert solver.interaction_from_array is not None
    interaction = solver.interaction_from_array(
        nonlocal_=[(K, A)],
        energy_dependent=False,
    )

    # Nonlocal term with A=[[1]] should produce K * sqrt(w_i * w_j) * a
    wi, wj = np.meshgrid(weights, weights, indexing="ij")
    gauss_scale = np.sqrt(wi * wj) * a
    expected = np.asarray(K) * gauss_scale
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
    assert solver.nonlocal_potential is not None

    # Build via solver.nonlocal_potential (canonical API)
    V = solver.nonlocal_potential(yamaguchi_kernel)
    r_from_potential = np.asarray(solver.rmatrix_direct(V))

    # Build via interaction_from_block (pre-assembled with Gauss scaling applied manually)
    ri, rj = jnp.meshgrid(solver.mesh.radii, solver.mesh.radii, indexing="ij")
    wi, wj = jnp.meshgrid(solver.mesh.weights, solver.mesh.weights, indexing="ij")
    scaled_block = yamaguchi_kernel(ri, rj) * (jnp.sqrt(wi * wj) * solver.mesh.scale)
    assert solver.interaction_from_block is not None
    interaction = solver.interaction_from_block(scaled_block, energy_dependent=False)
    r_from_block = np.asarray(solver.rmatrix_direct(interaction))

    assert np.allclose(r_from_potential, r_from_block, atol=1.0e-12, rtol=1.0e-12)


# ---------------------------------------------------------------------------
# Builders inside jax transformations (DESIGN Examples 16.3 / 16.4 patterns)
# ---------------------------------------------------------------------------


def _yamaguchi_solver() -> lm.Solver:
    return lm.compile(
        mesh=lm.MeshSpec("legendre", "x", n=8, scale=8.0),
        channels=(lm.ChannelSpec(l=0, threshold=0.0, mass_factor=41.472),),
        operators=("T+L",),
        solvers=("rmatrix_direct",),
        energies=jnp.asarray([0.1, 5.0]),
    )


def test_interaction_builder_under_vmap() -> None:
    """vmap over potential parameters builds a batched Interaction (Example 16.4)."""

    solver = _yamaguchi_solver()
    assert solver.interaction_from_funcs is not None

    def make_interaction(alpha: jax.Array, beta: jax.Array) -> lm.Interaction:
        def kernel(r1: jax.Array, r2: jax.Array) -> jax.Array:
            return -2.0 * beta * (alpha + beta) ** 2 * jnp.exp(-beta * (r1 + r2)) * 41.472

        assert solver.interaction_from_funcs is not None
        return solver.interaction_from_funcs(nonlocal_=[(kernel, jnp.ones((1, 1)))])

    alphas = jnp.asarray([0.2, 0.3])
    betas = jnp.asarray([1.4, 1.5])
    batched = jax.vmap(make_interaction)(alphas, betas)

    M = solver.mesh.n
    assert batched.block.shape == (2, M, M)
    # Each slice equals the eagerly built Interaction for those parameters.
    eager = make_interaction(alphas[1], betas[1])
    assert np.allclose(np.asarray(batched.block[1]), np.asarray(eager.block), atol=1e-13)


def test_interaction_builder_under_jit() -> None:
    """A jitted function may build the Interaction inside the trace (Example 16.3)."""

    solver = _yamaguchi_solver()

    @jax.jit
    def build_block(depth: jax.Array) -> jax.Array:
        assert solver.interaction_from_funcs is not None
        interaction = solver.interaction_from_funcs(
            nonlocal_=[(lambda r1, r2: -depth * jnp.exp(-(r1 + r2)), jnp.ones((1, 1)))]
        )
        return interaction.block

    block = build_block(jnp.asarray(5.0))
    assert block.shape == (solver.mesh.n, solver.mesh.n)


def test_interaction_builder_still_validates_eagerly() -> None:
    """Outside transformations, an asymmetric assembled block still raises."""

    solver = _yamaguchi_solver()
    N = solver.mesh.n
    asymmetric = jnp.triu(jnp.ones((N, N)))  # K(r, r') != K(r', r)

    assert solver.interaction_from_array is not None
    with pytest.raises(ValueError, match="not symmetric"):
        solver.interaction_from_array(nonlocal_=[(asymmetric, np.array([[1.0]]))])


# ---------------------------------------------------------------------------
# solver.local_potential / solver.nonlocal_potential entry points
# ---------------------------------------------------------------------------


def test_local_potential_builds_diagonal_block() -> None:
    """`solver.local_potential(fn)` assembles a local term (diagonal sub-block)."""

    solver = _make_mesh_channels()
    assert solver.local_potential is not None

    interaction = solver.local_potential(lambda r: -50.0 * jnp.exp(-((r / 2.0) ** 2)))
    assert not interaction.energy_dependent
    assert _is_block_diagonal(interaction.block), "local_potential produced a non-diagonal block"


def test_nonlocal_potential_builds_full_block() -> None:
    """`solver.nonlocal_potential(fn)` assembles a nonlocal term (full sub-block)."""

    solver = _make_mesh_channels()
    assert solver.nonlocal_potential is not None

    interaction = solver.nonlocal_potential(lambda r, rp: jnp.exp(-((r - rp) ** 2)))
    assert not interaction.energy_dependent
    assert not _is_block_diagonal(interaction.block), "nonlocal_potential produced a diagonal block"


def test_local_potential_energy_dependent_carries_energy_axis() -> None:
    """`solver.local_potential(fn(r, E), energy_dependent=True)` carries an (N_E,) axis."""

    solver = _make_mesh_channels()
    assert solver.local_potential is not None

    interaction = solver.local_potential(
        lambda r, e: -3.0 * jnp.exp(-((r / 2.0) ** 2)) + 0.02 * e, energy_dependent=True
    )
    assert interaction.energy_dependent
    assert interaction.block.shape[0] == len(solver.energies)

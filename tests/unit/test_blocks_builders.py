"""Tests for the block axis on the Interaction builders (DESIGN.md §15.5)."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import lax
from lax.meshes import build_mesh
from lax.operators.interaction import (
    make_interaction_from_array,
    make_interaction_from_block,
    make_interaction_from_funcs,
    make_potential_builders,
)
from lax.types import ChannelSpec, Interaction

HBAR2_2MU = 41.47
N = 8
N_E = 3
N_B = 2
N_C = 2
M = N_C * N


@pytest.fixture(scope="module")
def mesh_and_channels():
    mesh, _ = build_mesh(
        family="legendre",
        regularization="x",
        n=N,
        scale=10.0,
        operators={"T+L"},
    )
    channels = (
        ChannelSpec(l=0, threshold=0.0, mass_factor=HBAR2_2MU),
        ChannelSpec(l=2, threshold=1.0, mass_factor=HBAR2_2MU),
    )
    energies = jnp.linspace(1.0, 5.0, N_E)
    return mesh, channels, energies


def test_from_block_round_trips_block_shapes(mesh_and_channels) -> None:
    mesh, channels, energies = mesh_and_channels
    builder = make_interaction_from_block(mesh, channels, energies, n_blocks=N_B)
    rng = np.random.default_rng(0)

    block = jnp.asarray(rng.normal(size=(N_B, M, M)))
    interaction = builder(block, block_dependent=True)
    assert interaction.block_dependent is True
    assert interaction.energy_dependent is False
    assert interaction.block.shape == (N_B, M, M)

    block_e = jnp.asarray(rng.normal(size=(N_B, N_E, M, M)))
    interaction = builder(block_e, energy_dependent=True, block_dependent=True)
    assert interaction.block_dependent is True
    assert interaction.energy_dependent is True
    assert interaction.block.shape == (N_B, N_E, M, M)


def test_from_block_shape_errors(mesh_and_channels) -> None:
    mesh, channels, energies = mesh_and_channels
    builder = make_interaction_from_block(mesh, channels, energies, n_blocks=N_B)
    with pytest.raises(ValueError, match="Expected block shape"):
        builder(jnp.zeros((N_B + 1, M, M)), block_dependent=True)
    with pytest.raises(ValueError, match="Expected block shape"):
        builder(jnp.zeros((M, M)), block_dependent=True)


def test_block_dependent_requires_blocks_mode(mesh_and_channels) -> None:
    mesh, channels, energies = mesh_and_channels
    builder = make_interaction_from_block(mesh, channels, energies)
    with pytest.raises(TypeError, match="compiled with channels="):
        builder(jnp.zeros((N_B, M, M)), block_dependent=True)

    array_builder = make_interaction_from_array(mesh, channels, energies)
    with pytest.raises(TypeError, match="compiled with channels="):
        array_builder(local=[(jnp.ones((N_B, N)), np.eye(N_C))], block_dependent=True)


def test_from_array_block_constant_stack_matches_single(mesh_and_channels) -> None:
    mesh, channels, energies = mesh_and_channels
    rng = np.random.default_rng(1)
    coupling = np.array([[1.0, 0.3], [0.3, 0.5]])
    g_local = rng.normal(size=(N,))
    g_nonlocal = rng.normal(size=(N, N))
    g_nonlocal = 0.5 * (g_nonlocal + g_nonlocal.T)

    single = make_interaction_from_array(mesh, channels, energies)(
        local=[(jnp.asarray(g_local), coupling)],
        nonlocal_=[(jnp.asarray(g_nonlocal), coupling)],
    )

    stacked = make_interaction_from_array(mesh, channels, energies, n_blocks=N_B)(
        local=[(jnp.broadcast_to(g_local, (N_B, N)), coupling)],
        nonlocal_=[(jnp.broadcast_to(g_nonlocal, (N_B, N, N)), coupling)],
        block_dependent=True,
    )

    assert stacked.block.shape == (N_B, M, M)
    for b in range(N_B):
        np.testing.assert_array_equal(np.asarray(stacked.block[b]), np.asarray(single.block))


def test_from_array_block_energy_shapes(mesh_and_channels) -> None:
    mesh, channels, energies = mesh_and_channels
    rng = np.random.default_rng(2)
    coupling = np.eye(N_C)
    g = rng.normal(size=(N_B, N_E, N, N))
    g = 0.5 * (g + np.swapaxes(g, -1, -2))
    interaction = make_interaction_from_array(mesh, channels, energies, n_blocks=N_B)(
        nonlocal_=[(jnp.asarray(g), coupling)],
        energy_dependent=True,
        block_dependent=True,
    )
    assert interaction.block.shape == (N_B, N_E, M, M)
    assert interaction.energy_dependent and interaction.block_dependent

    with pytest.raises(ValueError, match="requires g shape"):
        make_interaction_from_array(mesh, channels, energies, n_blocks=N_B)(
            nonlocal_=[(jnp.asarray(g[:, 0]), coupling)],
            energy_dependent=True,
            block_dependent=True,
        )


def test_from_funcs_sequence_of_callables(mesh_and_channels) -> None:
    mesh, channels, energies = mesh_and_channels
    coupling = np.eye(N_C)
    funcs_builder = make_interaction_from_funcs(mesh, channels, energies, n_blocks=N_B)
    array_builder = make_interaction_from_array(mesh, channels, energies, n_blocks=N_B)

    def kernel_for_block(b: int):
        def kernel(ri: jnp.ndarray, rj: jnp.ndarray) -> jnp.ndarray:
            return -float(b + 1) * jnp.exp(-0.3 * (ri + rj))

        return kernel

    fns = [kernel_for_block(b) for b in range(N_B)]
    interaction = funcs_builder(nonlocal_=[(fns, coupling)], block_dependent=True)
    assert interaction.block.shape == (N_B, M, M)
    assert interaction.block_dependent is True

    r = mesh.radii
    ri, rj = jnp.meshgrid(r, r, indexing="ij")
    stacked = jnp.stack([fn(ri, rj) for fn in fns])
    expected = array_builder(nonlocal_=[(stacked, coupling)], block_dependent=True)
    np.testing.assert_array_equal(np.asarray(interaction.block), np.asarray(expected.block))


def test_from_funcs_bare_block_dependent_term_single_channel(mesh_and_channels) -> None:
    """A bare list of per-block callables gets the [[1.0]] coupling for N_c == 1."""

    mesh, _, energies = mesh_and_channels
    single = (ChannelSpec(l=0, threshold=0.0, mass_factor=HBAR2_2MU),)
    funcs_builder = make_interaction_from_funcs(mesh, single, energies, n_blocks=N_B)
    fns = [lambda ri, rj, b=b: -(b + 1.0) * jnp.exp(-(ri + rj)) for b in range(N_B)]
    bare = funcs_builder(nonlocal_=[fns], block_dependent=True)
    explicit = funcs_builder(nonlocal_=[(fns, np.ones((1, 1)))], block_dependent=True)
    assert bare.block.shape == (N_B, N, N)
    np.testing.assert_array_equal(np.asarray(bare.block), np.asarray(explicit.block))


def test_from_funcs_rejects_single_callable_when_block_dependent(
    mesh_and_channels,
) -> None:
    mesh, channels, energies = mesh_and_channels
    funcs_builder = make_interaction_from_funcs(mesh, channels, energies, n_blocks=N_B)
    with pytest.raises(TypeError, match="sequence of 2 callables"):
        funcs_builder(
            nonlocal_=[(lambda ri, rj: jnp.zeros_like(ri), np.eye(N_C))],
            block_dependent=True,
        )
    with pytest.raises(ValueError, match="one callable per symmetry block"):
        funcs_builder(
            nonlocal_=[([lambda ri, rj: jnp.zeros_like(ri)], np.eye(N_C))],
            block_dependent=True,
        )


def test_potential_builders_forward_block_dependent(mesh_and_channels) -> None:
    mesh, _, energies = mesh_and_channels
    single_channel = (ChannelSpec(l=0, threshold=0.0, mass_factor=HBAR2_2MU),)
    local_potential, nonlocal_potential = make_potential_builders(
        mesh, single_channel, energies, n_blocks=N_B
    )

    fns = [lambda r, b=b: -(b + 1.0) * jnp.exp(-r) for b in range(N_B)]
    interaction = local_potential(fns, block_dependent=True)
    assert interaction.block.shape == (N_B, N, N)
    assert interaction.block_dependent is True

    nl_fns = [lambda ri, rj, b=b: -(b + 1.0) * jnp.exp(-(ri + rj)) for b in range(N_B)]
    interaction = nonlocal_potential(nl_fns, block_dependent=True)
    assert interaction.block.shape == (N_B, N, N)


def test_channels_compiled_kernels_reject_block_dependent() -> None:
    energies = jnp.linspace(1.0, 5.0, N_E)
    solver = lax.compile(
        mesh=lax.MeshSpec("legendre", "x", n=N, scale=10.0),
        channels=(ChannelSpec(l=0, threshold=0.0, mass_factor=HBAR2_2MU),),
        solvers=("spectrum", "rmatrix_direct", "wavefunction"),
        energies=energies,
    )
    block_dep = Interaction(
        block=jnp.zeros((N_B, N, N)), energy_dependent=False, block_dependent=True
    )
    with pytest.raises(TypeError, match="re-compile with\\s+blocks="):
        solver.spectrum(block_dep)
    assert solver.rmatrix_direct is not None
    with pytest.raises(TypeError, match="re-compile\\s+with blocks="):
        solver.rmatrix_direct(block_dep)
    assert solver.wavefunction_direct is not None
    with pytest.raises(TypeError, match="re-compile with blocks="):
        solver.wavefunction_direct(block_dep, jnp.zeros(N), 0)

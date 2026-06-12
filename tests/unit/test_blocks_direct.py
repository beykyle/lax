"""Block-batched direct path: batched run ≡ per-block compiled solvers (§15.5)."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import lax
from lax.types import ChannelSpec
from tests.unit._blocks_helpers import (
    HBAR2_2MU,
    gaussian_kernel,
    gaussian_kernel_e,
    partial_wave_groups,
)

N = 16
RADIUS = 10.0
ENERGIES = jnp.linspace(2.0, 30.0, 5)
MESH = lax.MeshSpec("legendre", "x", n=N, scale=RADIUS)
DIRECT = ("rmatrix_direct",)

TIGHT = dict(rtol=1e-10, atol=1e-12)


def _compile_blocks(block_groups, **kwargs):
    return lax.compile(
        mesh=MESH,
        blocks=block_groups,
        solvers=DIRECT,
        energies=ENERGIES,
        **kwargs,
    )


def _compile_single(group, **kwargs):
    return lax.compile(
        mesh=MESH,
        channels=group,
        solvers=DIRECT,
        energies=ENERGIES,
        **kwargs,
    )


def _assert_direct_observables_match(blocks_solver, block_groups, interactions, **kwargs):
    """Compare every direct observable of the blocks solver against per-block compiles.

    ``interactions`` maps the blocks solver's Interaction to a list of
    per-block single-solver Interactions, as
    ``(blocks_interaction, [single_interaction_builder(solver) for each b])``.
    """

    blocks_interaction, single_builders = interactions
    r_blocks = blocks_solver.rmatrix_direct(blocks_interaction)
    s_blocks = blocks_solver.smatrix_direct(blocks_interaction)
    d_blocks = blocks_solver.phases_direct(blocks_interaction)
    n_b = len(block_groups)
    n_c = len(block_groups[0])
    assert r_blocks.shape == (n_b, len(ENERGIES), n_c, n_c)

    for b, group in enumerate(block_groups):
        single = _compile_single(group, **kwargs)
        interaction = single_builders[b](single)
        np.testing.assert_allclose(
            np.asarray(r_blocks[b]),
            np.asarray(single.rmatrix_direct(interaction)),
            err_msg=f"rmatrix_direct block {b}",
            **TIGHT,
        )
        np.testing.assert_allclose(
            np.asarray(s_blocks[b]),
            np.asarray(single.smatrix_direct(interaction)),
            err_msg=f"smatrix_direct block {b}",
            **TIGHT,
        )
        np.testing.assert_allclose(
            np.asarray(d_blocks[b]),
            np.asarray(single.phases_direct(interaction)),
            err_msg=f"phases_direct block {b}",
            **TIGHT,
        )


def test_partial_waves_block_independent_interaction() -> None:
    block_groups = partial_wave_groups()
    solver = _compile_blocks(block_groups)
    kernel = gaussian_kernel(20.0)
    interactions = (
        solver.nonlocal_potential(kernel),
        [lambda s, k=kernel: s.nonlocal_potential(k)] * len(block_groups),
    )
    _assert_direct_observables_match(solver, block_groups, interactions)


def test_partial_waves_block_dependent_interaction() -> None:
    block_groups = partial_wave_groups()
    solver = _compile_blocks(block_groups)
    kernels = [gaussian_kernel(10.0 * (b + 1)) for b in range(len(block_groups))]
    interactions = (
        solver.nonlocal_potential(kernels, block_dependent=True),
        [lambda s, k=k: s.nonlocal_potential(k) for k in kernels],
    )
    _assert_direct_observables_match(solver, block_groups, interactions)


def test_partial_waves_block_and_energy_dependent_interaction() -> None:
    block_groups = partial_wave_groups((0, 2))
    solver = _compile_blocks(block_groups)
    kernels = [gaussian_kernel_e(10.0 * (b + 1)) for b in range(len(block_groups))]
    interactions = (
        solver.nonlocal_potential(kernels, block_dependent=True, energy_dependent=True),
        [lambda s, k=k: s.nonlocal_potential(k, energy_dependent=True) for k in kernels],
    )
    _assert_direct_observables_match(solver, block_groups, interactions)


def test_partial_waves_coulomb() -> None:
    block_groups = partial_wave_groups()
    z1z2 = (2, 82)
    solver = _compile_blocks(block_groups, z1z2=z1z2)
    kernel = gaussian_kernel(15.0)
    interactions = (
        solver.nonlocal_potential(kernel),
        [lambda s, k=kernel: s.nonlocal_potential(k)] * len(block_groups),
    )
    _assert_direct_observables_match(solver, block_groups, interactions, z1z2=z1z2)


def test_coupled_channel_blocks() -> None:
    coupling = np.array([[1.0, 0.4], [0.4, 0.7]])
    block_groups = (
        (
            ChannelSpec(l=0, threshold=0.0, mass_factor=HBAR2_2MU),
            ChannelSpec(l=2, threshold=2.0, mass_factor=HBAR2_2MU),
        ),
        (
            ChannelSpec(l=1, threshold=0.0, mass_factor=HBAR2_2MU),
            ChannelSpec(l=3, threshold=2.0, mass_factor=HBAR2_2MU),
        ),
    )
    solver = _compile_blocks(block_groups)
    kernels = [gaussian_kernel(12.0), gaussian_kernel(18.0)]
    interactions = (
        solver.nonlocal_potential(kernels, coupling=coupling, block_dependent=True),
        [lambda s, k=k: s.nonlocal_potential(k, coupling=coupling) for k in kernels],
    )
    _assert_direct_observables_match(solver, block_groups, interactions)


@pytest.mark.parametrize(
    "mass_factor_grid",
    [
        np.full(len(ENERGIES), 40.0),  # energy-uniform override (fast path)
        np.linspace(40.0, 42.0, len(ENERGIES)),  # μ(E) (per-energy grid path)
    ],
    ids=["uniform_mu", "energy_dependent_mu"],
)
def test_partial_waves_with_mass_factor_grid(mass_factor_grid) -> None:
    block_groups = partial_wave_groups((0, 1))
    solver = _compile_blocks(block_groups, mass_factor_grid=mass_factor_grid)
    kernel = gaussian_kernel(20.0)
    interactions = (
        solver.nonlocal_potential(kernel),
        [lambda s, k=kernel: s.nonlocal_potential(k)] * len(block_groups),
    )
    _assert_direct_observables_match(
        solver, block_groups, interactions, mass_factor_grid=mass_factor_grid
    )


def test_wavefunction_direct_matches_per_block() -> None:
    block_groups = partial_wave_groups()
    solver = _compile_blocks(block_groups)
    kernels = [gaussian_kernel(10.0 * (b + 1)) for b in range(len(block_groups))]
    interaction = solver.nonlocal_potential(kernels, block_dependent=True)
    energy_index = 2
    sources = lax.make_wavefunction_source(solver, channel_index=0, energy_index=energy_index)
    assert sources.shape == (len(block_groups), N)
    psi_blocks = solver.wavefunction_direct(interaction, sources, energy_index)
    assert psi_blocks.shape == (len(block_groups), N)

    for b, group in enumerate(block_groups):
        single = _compile_single(group)
        source = lax.make_wavefunction_source(single, channel_index=0, energy_index=energy_index)
        np.testing.assert_array_equal(np.asarray(sources[b]), np.asarray(source))
        psi = single.wavefunction_direct(
            single.nonlocal_potential(kernels[b]), source, energy_index
        )
        np.testing.assert_allclose(
            np.asarray(psi_blocks[b]), np.asarray(psi), err_msg=f"block {b}", **TIGHT
        )


def test_wavefunction_direct_shared_source_broadcasts() -> None:
    block_groups = partial_wave_groups((0, 1))
    solver = _compile_blocks(block_groups)
    interaction = solver.nonlocal_potential(gaussian_kernel(20.0))
    shared = jnp.ones(N, dtype=jnp.complex128)
    psi = solver.wavefunction_direct(interaction, shared, 0)
    assert psi.shape == (2, N)
    stacked = solver.wavefunction_direct(interaction, jnp.stack([shared, shared]), 0)
    np.testing.assert_array_equal(np.asarray(psi), np.asarray(stacked))


def test_block_constant_stack_reduces_to_block_independent() -> None:
    block_groups = partial_wave_groups()
    solver = _compile_blocks(block_groups)
    kernel = gaussian_kernel(20.0)
    constant_stack = solver.nonlocal_potential([kernel] * len(block_groups), block_dependent=True)
    independent = solver.nonlocal_potential(kernel)
    np.testing.assert_allclose(
        np.asarray(solver.rmatrix_direct(constant_stack)),
        np.asarray(solver.rmatrix_direct(independent)),
        **TIGHT,
    )


def test_single_block_compile_still_carries_leading_axis() -> None:
    block_groups = partial_wave_groups((0,))
    solver = _compile_blocks(block_groups)
    interaction = solver.nonlocal_potential(gaussian_kernel(20.0))
    r = solver.rmatrix_direct(interaction)
    assert r.shape == (1, len(ENERGIES), 1, 1)
    assert solver.blocks == block_groups
    assert "1 blocks" in repr(solver)


def test_compile_validation_errors() -> None:
    group = (ChannelSpec(l=0, threshold=0.0, mass_factor=HBAR2_2MU),)
    with pytest.raises(ValueError, match="exactly one of"):
        lax.compile(mesh=MESH, channels=group, blocks=[group], solvers=DIRECT, energies=ENERGIES)
    with pytest.raises(ValueError, match="exactly one of"):
        lax.compile(mesh=MESH, solvers=DIRECT, energies=ENERGIES)
    with pytest.raises(ValueError, match="same non-zero channel shape"):
        lax.compile(
            mesh=MESH,
            blocks=[
                group,
                (
                    ChannelSpec(l=0, threshold=0.0, mass_factor=HBAR2_2MU),
                    ChannelSpec(l=2, threshold=0.0, mass_factor=HBAR2_2MU),
                ),
            ],
            solvers=DIRECT,
            energies=ENERGIES,
        )
    with pytest.raises(ValueError, match="not supported with `blocks="):
        lax.compile(
            mesh=MESH,
            blocks=[group],
            solvers=DIRECT,
            energies=ENERGIES,
            momenta=jnp.linspace(0.1, 2.0, 10),
        )
    with pytest.raises(ValueError, match="not supported on propagated meshes"):
        lax.compile(
            mesh=lax.MeshSpec("legendre", "x", n=N, scale=RADIUS, extras={"n_intervals": 4}),
            blocks=[group],
            solvers=DIRECT,
            energies=ENERGIES,
            method="linear_solve",
        )


def test_grid_transforms_work_in_blocks_mode() -> None:
    block_groups = partial_wave_groups((0, 1))
    grid = jnp.linspace(0.1, RADIUS - 0.1, 30)
    solver = _compile_blocks(block_groups, grid=grid)
    assert solver.to_grid_vector is not None
    values = solver.to_grid_vector(jnp.ones(N))
    assert values.shape == (30,)

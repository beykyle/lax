"""Block-batched spectral path: batched run ≡ per-block compiled solvers (§15.5)."""

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
SPECTRAL = ("spectrum", "rmatrix", "smatrix", "phases", "greens", "wavefunction")

TIGHT = dict(rtol=1e-10, atol=1e-12)


def _compile(solvers=SPECTRAL, **kwargs):
    return lax.compile(mesh=MESH, solvers=solvers, energies=ENERGIES, **kwargs)


def test_spectral_observables_match_per_block() -> None:
    block_groups = partial_wave_groups()
    n_b = len(block_groups)
    solver = _compile(blocks=block_groups)
    kernels = [gaussian_kernel(10.0 * (b + 1)) for b in range(n_b)]
    interaction = solver.nonlocal_potential(kernels, block_dependent=True)

    spectrum = solver.spectrum(interaction)
    assert spectrum.eigenvalues.shape == (n_b, N)
    assert spectrum.surface_amplitudes.shape == (n_b, N, 1)

    probe_energy = 12.5
    r_blocks = solver.rmatrix(spectrum, probe_energy)
    s_blocks = solver.smatrix(spectrum)
    d_blocks = solver.phases(spectrum)
    g_blocks = solver.greens(spectrum, probe_energy)
    assert r_blocks.shape == (n_b, 1, 1)
    assert s_blocks.shape == (n_b, len(ENERGIES), 1, 1)
    assert d_blocks.shape == (n_b, len(ENERGIES), 1)
    assert g_blocks.shape == (n_b, N, N)

    energy_index = 2
    sources = lax.make_wavefunction_source(solver, channel_index=0, energy_index=energy_index)
    psi_blocks = solver.wavefunction(spectrum, ENERGIES[energy_index], sources)
    assert psi_blocks.shape == (n_b, N)

    eigenvalues_blocks, eigenvectors_blocks = solver.eigh(spectrum)
    assert eigenvalues_blocks.shape == (n_b, N)
    assert eigenvectors_blocks.shape == (n_b, N, N)

    for b, group in enumerate(block_groups):
        single = _compile(channels=group)
        single_spectrum = single.spectrum(single.nonlocal_potential(kernels[b]))
        np.testing.assert_allclose(
            np.asarray(spectrum.eigenvalues[b]),
            np.asarray(single_spectrum.eigenvalues),
            err_msg=f"eigenvalues block {b}",
            **TIGHT,
        )
        np.testing.assert_allclose(
            np.asarray(r_blocks[b]),
            np.asarray(single.rmatrix(single_spectrum, probe_energy)),
            err_msg=f"rmatrix block {b}",
            **TIGHT,
        )
        np.testing.assert_allclose(
            np.asarray(s_blocks[b]),
            np.asarray(single.smatrix(single_spectrum)),
            err_msg=f"smatrix block {b}",
            **TIGHT,
        )
        np.testing.assert_allclose(
            np.asarray(d_blocks[b]),
            np.asarray(single.phases(single_spectrum)),
            err_msg=f"phases block {b}",
            **TIGHT,
        )
        np.testing.assert_allclose(
            np.asarray(g_blocks[b]),
            np.asarray(single.greens(single_spectrum, probe_energy)),
            err_msg=f"greens block {b}",
            **TIGHT,
        )
        source = lax.make_wavefunction_source(single, channel_index=0, energy_index=energy_index)
        np.testing.assert_allclose(
            np.asarray(psi_blocks[b]),
            np.asarray(single.wavefunction(single_spectrum, ENERGIES[energy_index], source)),
            err_msg=f"wavefunction block {b}",
            **TIGHT,
        )


def test_spectral_block_independent_interaction_broadcasts() -> None:
    block_groups = partial_wave_groups((0, 1))
    solver = _compile(blocks=block_groups)
    kernel = gaussian_kernel(20.0)
    spectrum = solver.spectrum(solver.nonlocal_potential(kernel))
    phases = solver.phases(spectrum)
    assert phases.shape == (2, len(ENERGIES), 1)
    # Blocks differ through their centrifugal even for a shared interaction.
    assert not np.allclose(np.asarray(phases[0]), np.asarray(phases[1]))

    for b, group in enumerate(block_groups):
        single = _compile(channels=group)
        single_spectrum = single.spectrum(single.nonlocal_potential(kernel))
        np.testing.assert_allclose(
            np.asarray(phases[b]),
            np.asarray(single.phases(single_spectrum)),
            err_msg=f"block {b}",
            **TIGHT,
        )


def test_energy_dependent_grid_observables_match_per_block() -> None:
    block_groups = partial_wave_groups((0, 2))
    solver = _compile(blocks=block_groups)
    kernels = [gaussian_kernel_e(10.0 * (b + 1)) for b in range(len(block_groups))]
    interaction = solver.nonlocal_potential(kernels, block_dependent=True, energy_dependent=True)

    spectra = solver.spectrum(interaction)
    assert spectra.eigenvalues.shape == (len(block_groups), len(ENERGIES), N)

    r_grid = solver.rmatrix_grid(spectra)
    s_grid = solver.smatrix_grid(spectra)
    d_grid = solver.phases_grid(spectra)
    assert d_grid.shape == (len(block_groups), len(ENERGIES), 1)

    for b, group in enumerate(block_groups):
        single = _compile(channels=group)
        single_spectra = single.spectrum(
            single.nonlocal_potential(kernels[b], energy_dependent=True)
        )
        np.testing.assert_allclose(
            np.asarray(r_grid[b]),
            np.asarray(single.rmatrix_grid(single_spectra)),
            err_msg=f"rmatrix_grid block {b}",
            **TIGHT,
        )
        np.testing.assert_allclose(
            np.asarray(s_grid[b]),
            np.asarray(single.smatrix_grid(single_spectra)),
            err_msg=f"smatrix_grid block {b}",
            **TIGHT,
        )
        np.testing.assert_allclose(
            np.asarray(d_grid[b]),
            np.asarray(single.phases_grid(single_spectra)),
            err_msg=f"phases_grid block {b}",
            **TIGHT,
        )


def test_eig_method_smoke() -> None:
    block_groups = partial_wave_groups((0, 1))
    solver = _compile(
        blocks=block_groups,
        solvers=("spectrum", "smatrix", "phases"),
        method="eig",
        V_is_complex=True,
    )

    def complex_kernel(ri: jnp.ndarray, rj: jnp.ndarray) -> jnp.ndarray:
        return (-20.0 - 4.0j) * jnp.exp(-0.25 * (ri - rj) ** 2 - 0.05 * (ri + rj) ** 2)

    spectrum = solver.spectrum(solver.nonlocal_potential(complex_kernel))
    s_blocks = solver.smatrix(spectrum)
    assert s_blocks.shape == (2, len(ENERGIES), 1, 1)

    for b, group in enumerate(block_groups):
        single = lax.compile(
            mesh=MESH,
            channels=group,
            solvers=("spectrum", "smatrix", "phases"),
            energies=ENERGIES,
            method="eig",
            V_is_complex=True,
        )
        single_spectrum = single.spectrum(single.nonlocal_potential(complex_kernel))
        np.testing.assert_allclose(
            np.asarray(s_blocks[b]),
            np.asarray(single.smatrix(single_spectrum)),
            err_msg=f"block {b}",
            **TIGHT,
        )


def test_mixed_mass_across_blocks_rejected_on_spectral_path() -> None:
    block_groups = (
        (ChannelSpec(l=0, threshold=0.0, mass_factor=HBAR2_2MU),),
        (ChannelSpec(l=1, threshold=0.0, mass_factor=38.0),),
    )
    with pytest.raises(ValueError, match="uniform mass_factor"):
        _compile(blocks=block_groups)
    # The direct path supports per-block μ via the per-block Q' scaling.
    solver = _compile(blocks=block_groups, solvers=("rmatrix_direct",))
    assert solver.rmatrix_direct is not None

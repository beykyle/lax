"""C4 acceptance tests: non-uniform μ(E) regimes on the spectral path (spec v0.1.5.1, T9)."""

from __future__ import annotations

import pickle

import jax.numpy as jnp
import numpy as np
import pytest

import lax
from tests.unit._blocks_helpers import HBAR2_2MU

N = 10
RADIUS = 8.0
ENERGIES = jnp.linspace(2.0, 10.0, 3)
N_E = len(ENERGIES)
MESH = lax.MeshSpec("legendre", "x", n=N, scale=RADIUS)
SPECTRAL = ("spectrum", "rmatrix", "smatrix", "phases", "greens", "wavefunction")
# A genuinely energy-dependent reduced mass (a few percent across the grid).
NONUNIFORM_MFG = jnp.asarray(HBAR2_2MU * (1.0 + 0.03 * np.arange(N_E)))

TIGHT = dict(rtol=1e-10, atol=1e-12)


def _solver(mass_factor_grid: jnp.ndarray | None) -> lax.Solver:
    return lax.compile(
        mesh=MESH,
        channels=(lax.ChannelSpec(l=0, threshold=0.0, mass_factor=HBAR2_2MU),),
        solvers=SPECTRAL,
        energies=ENERGIES,
        mass_factor_grid=mass_factor_grid,
    )


def _static_interaction(solver: lax.Solver) -> lax.Interaction:
    assert solver.local_potential is not None
    return solver.local_potential(lambda r: -8.0 * jnp.exp(-0.15 * r**2))


def test_static_observables_raise_on_nonuniform_mass_factor_grid() -> None:
    """All five static-regime spectral observables are stubbed out (C4)."""

    solver = _solver(NONUNIFORM_MFG)
    spectrum = solver.spectrum(_static_interaction(solver))  # type: ignore[misc]

    for name in ("rmatrix", "smatrix", "phases", "greens", "wavefunction"):
        observable = getattr(solver, name)
        assert observable is not None
        with pytest.raises(ValueError) as excinfo:
            observable(spectrum)
        message = str(excinfo.value)
        assert f"solver.{name} is unavailable" in message
        assert "non-uniform mass_factor_grid" in message
        assert "rmatrix_grid" in message
        assert "direct path" in message


def test_spectrum_is_energy_batched_even_for_static_potential() -> None:
    """Non-uniform μ(E) forces the energy-batched spectrum path for static V."""

    solver = _solver(NONUNIFORM_MFG)
    spectrum = solver.spectrum(_static_interaction(solver))  # type: ignore[misc]
    assert spectrum.eigenvalues.shape == (N_E, N)

    uniform = _solver(None)
    static_spectrum = uniform.spectrum(_static_interaction(uniform))  # type: ignore[misc]
    assert static_spectrum.eigenvalues.shape == (N,)


def test_grid_observables_match_per_energy_compiled_references() -> None:
    """`smatrix_grid` with μ(E) equals per-energy independently compiled solvers.

    This pins the spectrum-kernel fix: each per-energy Hamiltonian must be
    scaled by its own μ_e, not by a single uniform μ.
    """

    solver = _solver(NONUNIFORM_MFG)
    interaction = _static_interaction(solver)
    spectra = solver.spectrum(interaction)  # type: ignore[misc]
    assert solver.smatrix_grid is not None
    assert solver.rmatrix_grid is not None
    s_grid = solver.smatrix_grid(spectra)
    r_grid = solver.rmatrix_grid(spectra)

    for e in range(N_E):
        reference = lax.compile(
            mesh=MESH,
            channels=(lax.ChannelSpec(l=0, threshold=0.0, mass_factor=float(NONUNIFORM_MFG[e])),),
            solvers=SPECTRAL,
            energies=ENERGIES[e : e + 1],
        )
        ref_spectrum = reference.spectrum(_static_interaction(reference))  # type: ignore[misc]
        assert reference.smatrix is not None
        assert reference.rmatrix is not None
        np.testing.assert_allclose(
            np.asarray(s_grid[e]),
            np.asarray(reference.smatrix(ref_spectrum)[0]),
            err_msg=f"smatrix energy {e}",
            **TIGHT,
        )
        np.testing.assert_allclose(
            np.asarray(r_grid[e]),
            np.asarray(reference.rmatrix(ref_spectrum, ENERGIES[e])),
            err_msg=f"rmatrix energy {e}",
            **TIGHT,
        )


def test_uniform_grid_keeps_static_observables_live() -> None:
    """A constant mass_factor_grid is uniform — no stubs, results unchanged."""

    uniform_grid = jnp.full(N_E, HBAR2_2MU)
    with_grid = _solver(uniform_grid)
    without_grid = _solver(None)

    spectrum_g = with_grid.spectrum(_static_interaction(with_grid))  # type: ignore[misc]
    spectrum_n = without_grid.spectrum(_static_interaction(without_grid))  # type: ignore[misc]
    assert spectrum_g.eigenvalues.shape == spectrum_n.eigenvalues.shape == (N,)
    assert with_grid.smatrix is not None
    assert without_grid.smatrix is not None
    np.testing.assert_allclose(
        np.asarray(with_grid.smatrix(spectrum_g)),
        np.asarray(without_grid.smatrix(spectrum_n)),
        **TIGHT,
    )


def test_per_channel_nonuniform_grid_rejected_on_spectral_path() -> None:
    """Per-channel μ stays a direct-path feature; the spectral path raises."""

    channels = (
        lax.ChannelSpec(l=0, threshold=0.0, mass_factor=HBAR2_2MU),
        lax.ChannelSpec(l=0, threshold=1.0, mass_factor=HBAR2_2MU),
    )
    per_channel = jnp.stack([jnp.full(N_E, HBAR2_2MU), jnp.full(N_E, 1.1 * HBAR2_2MU)], axis=1)
    with pytest.raises(ValueError, match="per-channel-uniform"):
        lax.compile(
            mesh=MESH,
            channels=channels,
            solvers=("spectrum", "rmatrix"),
            energies=ENERGIES,
            mass_factor_grid=per_channel,
        )


def test_eigh_accessor_survives_nonuniform_grid() -> None:
    solver = _solver(NONUNIFORM_MFG)
    spectra = solver.spectrum(_static_interaction(solver))  # type: ignore[misc]
    assert solver.eigh is not None
    eigenvalues, eigenvectors = solver.eigh(spectra)
    assert eigenvalues.shape == (N_E, N)
    assert eigenvectors.shape == (N_E, N, N)


def test_nonuniform_solver_round_trips_through_pickle() -> None:
    solver = _solver(NONUNIFORM_MFG)
    restored = pickle.loads(pickle.dumps(solver))
    interaction = _static_interaction(solver)
    np.testing.assert_allclose(
        np.asarray(restored.spectrum(interaction).eigenvalues),  # type: ignore[misc]
        np.asarray(solver.spectrum(interaction).eigenvalues),  # type: ignore[misc]
        **TIGHT,
    )
    assert restored.smatrix is not None
    with pytest.raises(ValueError, match="non-uniform mass_factor_grid"):
        restored.smatrix(restored.spectrum(interaction))  # type: ignore[misc]

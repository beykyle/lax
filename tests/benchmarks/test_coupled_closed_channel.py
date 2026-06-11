from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import lax as lm
from lax.boundary import BoundaryValues
from lax.spectral.matching import _decouple_closed_channels, _project_open_channels

pytest.importorskip("jax")

ENERGIES = np.array([0.25, 0.75, 1.25, 1.75], dtype=np.float64)
CHANNEL_THRESHOLD = 1.0
MASS_FACTOR = 1.0


def _coupled_solver(method: str, solvers: tuple[str, ...]) -> lm.Solver:
    """Compile the mixed-threshold two-channel toy model."""

    return lm.compile(
        mesh=lm.MeshSpec("legendre", "x", n=24, scale=8.0),
        channels=(
            lm.ChannelSpec(l=0, threshold=0.0, mass_factor=MASS_FACTOR),
            lm.ChannelSpec(l=2, threshold=CHANNEL_THRESHOLD, mass_factor=MASS_FACTOR),
        ),
        operators=("T+L", "1/r^2"),
        solvers=solvers,
        energies=ENERGIES,
        method=method,
    )


def _single_channel_solver() -> lm.Solver:
    """Compile the decoupled open-channel limit."""

    return lm.compile(
        mesh=lm.MeshSpec("legendre", "x", n=24, scale=8.0),
        channels=(lm.ChannelSpec(l=0, threshold=0.0, mass_factor=MASS_FACTOR),),
        operators=("T+L",),
        solvers=("spectrum", "smatrix"),
        energies=ENERGIES,
        method="eigh",
    )


def _diagonal_open(radii: jax.Array) -> jax.Array:
    """Return the open-channel diagonal potential in MeV."""

    return -6.0 * jnp.exp(-((radii / 2.1) ** 2))


def _diagonal_closed(radii: jax.Array) -> jax.Array:
    """Return the closed-channel diagonal potential in MeV."""

    return -4.5 * jnp.exp(-((radii / 2.6) ** 2))


def _channel_coupling(radii: jax.Array) -> jax.Array:
    """Return the inter-channel coupling potential in MeV."""

    return -1.25 * jnp.exp(-((radii / 2.3) ** 2))


def _toy_interaction(solver: lm.Solver, *, coupled: bool) -> object:
    """Build the 2-channel toy Interaction, optionally without the channel coupling."""

    A00 = np.array([[1.0, 0.0], [0.0, 0.0]])
    A01 = np.array([[0.0, 1.0], [1.0, 0.0]])
    A11 = np.array([[0.0, 0.0], [0.0, 1.0]])
    assert solver.local_potential is not None
    interaction = solver.local_potential(_diagonal_open, coupling=A00) + solver.local_potential(
        _diagonal_closed, coupling=A11
    )
    if coupled:
        interaction = interaction + solver.local_potential(_channel_coupling, coupling=A01)
    return interaction


def _smatrix_from_direct_rmatrix(solver: lm.Solver, potential) -> np.ndarray:
    """Evaluate the physical S-matrix from the direct R-matrix kernel."""

    assert solver.rmatrix_direct is not None
    assert solver.boundary is not None

    r_values = solver.rmatrix_direct(potential)
    smatrices: list[np.ndarray] = []
    for energy_index in range(r_values.shape[0]):
        decoupled_r = _decouple_closed_channels(
            r_values[energy_index],
            solver.boundary.H_plus[energy_index],
            solver.boundary.H_plus_p[energy_index],
            solver.boundary.is_open[energy_index],
        )
        projected_r, projected_boundary = _project_open_channels(
            decoupled_r,
            solver.boundary.H_plus[energy_index],
            solver.boundary.H_minus[energy_index],
            solver.boundary.H_plus_p[energy_index],
            solver.boundary.H_minus_p[energy_index],
            solver.boundary.is_open[energy_index],
            solver.boundary.k[energy_index],
        )
        smatrix = lm.spectral.smatrix_from_R(
            projected_r,
            BoundaryValues(
                H_plus=projected_boundary.H_plus,
                H_minus=projected_boundary.H_minus,
                H_plus_p=projected_boundary.H_plus_p,
                H_minus_p=projected_boundary.H_minus_p,
                is_open=projected_boundary.is_open,
                k=projected_boundary.k,
            ),
        )
        smatrices.append(np.asarray(smatrix))
    return np.stack(smatrices)


@pytest.mark.benchmark
def test_coupled_closed_channel_spectral_and_direct_paths_agree() -> None:
    """A mixed open/closed coupled system agrees across the spectral and direct paths."""

    spectral_solver = _coupled_solver("eigh", ("spectrum", "smatrix"))
    direct_solver = _coupled_solver("linear_solve", ("rmatrix_direct",))
    spectral_V = _toy_interaction(spectral_solver, coupled=True)
    direct_V = _toy_interaction(direct_solver, coupled=True)

    assert spectral_solver.spectrum is not None
    assert spectral_solver.smatrix is not None

    spectral_smatrix = np.asarray(spectral_solver.smatrix(spectral_solver.spectrum(spectral_V)))
    direct_smatrix = _smatrix_from_direct_rmatrix(direct_solver, direct_V)
    below_threshold = ENERGIES < CHANNEL_THRESHOLD
    above_threshold = np.logical_not(below_threshold)

    assert spectral_smatrix.shape == (ENERGIES.size, 2, 2)
    assert np.allclose(spectral_smatrix, direct_smatrix, atol=1.0e-10, rtol=1.0e-10)
    assert np.all(np.isfinite(spectral_smatrix[below_threshold, 0, 0]))
    assert np.allclose(spectral_smatrix[below_threshold, 1, :], 0.0, atol=1.0e-12, rtol=1.0e-12)
    assert np.allclose(spectral_smatrix[below_threshold, :, 1], 0.0, atol=1.0e-12, rtol=1.0e-12)
    assert np.any(np.abs(spectral_smatrix[above_threshold, 0, 1]) > 1.0e-8)


@pytest.mark.benchmark
def test_coupled_closed_channel_decoupled_limit_matches_single_channel() -> None:
    """Removing the channel coupling recovers the one-channel open-channel result."""

    coupled_solver = _coupled_solver("eigh", ("spectrum", "smatrix"))
    single_channel_solver = _single_channel_solver()
    coupled_V = _toy_interaction(coupled_solver, coupled=False)
    assert single_channel_solver.local_potential is not None
    single_channel_V = single_channel_solver.local_potential(_diagonal_open)

    assert coupled_solver.spectrum is not None
    assert coupled_solver.smatrix is not None
    assert single_channel_solver.spectrum is not None
    assert single_channel_solver.smatrix is not None

    coupled_smatrix = np.asarray(coupled_solver.smatrix(coupled_solver.spectrum(coupled_V)))
    single_channel_smatrix = np.asarray(
        single_channel_solver.smatrix(single_channel_solver.spectrum(single_channel_V))
    )

    assert np.allclose(
        coupled_smatrix[:, 0, 0],
        single_channel_smatrix[:, 0, 0],
        atol=1.0e-10,
        rtol=1.0e-10,
    )

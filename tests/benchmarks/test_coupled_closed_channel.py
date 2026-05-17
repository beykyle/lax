from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import lax as lm
from lax.boundary import BoundaryValues
from lax.solvers.observables import _decouple_closed_channels, _project_open_channels

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


def _toy_potential(radii: jax.Array, channel_index: int, coupled_index: int) -> jax.Array:
    """Return a smooth two-channel local potential in MeV."""

    diagonal_open = -6.0 * jnp.exp(-(radii / 2.1) ** 2)
    diagonal_closed = -4.5 * jnp.exp(-(radii / 2.6) ** 2)
    coupling = -1.25 * jnp.exp(-(radii / 2.3) ** 2)

    if channel_index == coupled_index == 0:
        return diagonal_open
    if channel_index == coupled_index == 1:
        return diagonal_closed
    return coupling


def _decoupled_potential(radii: jax.Array, channel_index: int, coupled_index: int) -> jax.Array:
    """Return the same toy model with the inter-channel coupling removed."""

    if channel_index != coupled_index:
        return jnp.zeros_like(radii)
    return _toy_potential(radii, channel_index, coupled_index)


def _open_channel_potential(radii: jax.Array) -> jax.Array:
    """Return the open-channel diagonal potential used in the decoupled limit."""

    return _toy_potential(radii, 0, 0)


def _smatrix_from_direct_rmatrix(solver: lm.Solver, potential: jax.Array) -> np.ndarray:
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
        )
        smatrix = lm.spectral.smatrix_from_R(
            projected_r,
            BoundaryValues(
                H_plus=projected_boundary.H_plus,
                H_minus=projected_boundary.H_minus,
                H_plus_p=projected_boundary.H_plus_p,
                H_minus_p=projected_boundary.H_minus_p,
                is_open=projected_boundary.is_open,
            ),
        )
        smatrices.append(np.asarray(smatrix))
    return np.stack(smatrices)


@pytest.mark.benchmark
def test_coupled_closed_channel_spectral_and_direct_paths_agree() -> None:
    """A mixed open/closed coupled system agrees across the spectral and direct paths."""

    spectral_solver = _coupled_solver("eigh", ("spectrum", "smatrix"))
    direct_solver = _coupled_solver("linear_solve", ("rmatrix_direct",))
    potential = lm.assemble_local(spectral_solver.mesh, _toy_potential, n_channels=2)
    direct_potential = lm.assemble_local(direct_solver.mesh, _toy_potential, n_channels=2)

    assert spectral_solver.spectrum is not None
    assert spectral_solver.smatrix is not None

    spectral_smatrix = np.asarray(spectral_solver.smatrix(spectral_solver.spectrum(potential)))
    direct_smatrix = _smatrix_from_direct_rmatrix(direct_solver, direct_potential)
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
    coupled_potential = lm.assemble_local(coupled_solver.mesh, _decoupled_potential, n_channels=2)
    single_channel_potential = lm.assemble_local(
        single_channel_solver.mesh,
        _open_channel_potential,
    )

    assert coupled_solver.spectrum is not None
    assert coupled_solver.smatrix is not None
    assert single_channel_solver.spectrum is not None
    assert single_channel_solver.smatrix is not None

    coupled_smatrix = np.asarray(coupled_solver.smatrix(coupled_solver.spectrum(coupled_potential)))
    single_channel_smatrix = np.asarray(
        single_channel_solver.smatrix(single_channel_solver.spectrum(single_channel_potential))
    )

    assert np.allclose(
        coupled_smatrix[:, 0, 0],
        single_channel_smatrix[:, 0, 0],
        atol=1.0e-10,
        rtol=1.0e-10,
    )

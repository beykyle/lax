from __future__ import annotations

import jax
import numpy as np
import pytest

import lax as lm
from lax._descouvemont_utils import (
    alpha_c12_channels,
    alpha_c12_open_channel_count,
    alpha_c12_potential,
    first_column_amplitudes_and_phases,
)
from lax.boundary import BoundaryValues
from lax.solvers.observables import _decouple_closed_channels, _project_open_channels
from tests.benchmarks._descouvemont_cases import ALPHA_C12_REFERENCE_A11, AlphaC12Reference

pytest.importorskip("jax")


def _solver(reference: AlphaC12Reference, method: str, solvers: tuple[str, ...]) -> lm.Solver:
    """Compile the Descouvemont Example 4 α + 12C benchmark."""

    return lm.compile(
        mesh=lm.MeshSpec("legendre", "x", n=reference.n_basis, scale=reference.scale),
        channels=alpha_c12_channels(),
        operators=("T+L", "1/r^2"),
        solvers=solvers,
        energies=reference.energies,
        method=method,
        V_is_complex=True,
        z1z2=(2, 6),
    )


def _smatrix_from_direct_rmatrix(
    solver: lm.Solver, potential: jax.Array
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    """Evaluate the physical open-channel S-matrices from the direct R-matrix kernel."""

    assert solver.rmatrix_direct is not None
    assert solver.boundary is not None

    r_values = solver.rmatrix_direct(potential)
    smatrices: list[np.ndarray] = []
    projected_boundaries: list[np.ndarray] = []
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
        open_count = alpha_c12_open_channel_count(float(solver.energies[energy_index]))
        smatrices.append(np.asarray(smatrix)[:open_count, :open_count])
        projected_boundaries.append(np.asarray(projected_boundary.is_open))
    return tuple(smatrices), tuple(projected_boundaries)


@pytest.mark.benchmark
def test_descouvemont_closed_channel_example_matches_published_first_column_structure() -> None:
    """The α + 12C benchmark follows the published first-column convention and thresholds."""

    reference = ALPHA_C12_REFERENCE_A11
    solver = _solver(reference, "linear_solve", ("rmatrix_direct",))
    potential = lm.assemble_local(
        solver.mesh, alpha_c12_potential, n_channels=len(alpha_c12_channels())
    )
    smatrices, projected_boundaries = _smatrix_from_direct_rmatrix(solver, potential)

    for energy_index, energy in enumerate(reference.energies):
        open_count = alpha_c12_open_channel_count(float(energy))
        amplitudes, phases = first_column_amplitudes_and_phases(smatrices[energy_index], open_count)

        assert smatrices[energy_index].shape == (open_count, open_count)
        assert np.count_nonzero(projected_boundaries[energy_index]) == open_count
        assert amplitudes.shape[0] == reference.amplitudes[energy_index].shape[0]
        assert phases.shape[0] == reference.phases[energy_index].shape[0]
        assert np.all(np.isfinite(amplitudes))
        assert np.all(np.isfinite(phases))

    amplitudes_4_mev, phases_4_mev = first_column_amplitudes_and_phases(smatrices[0], open_count=1)
    assert np.allclose(amplitudes_4_mev, reference.amplitudes[0], atol=7.0e-3, rtol=7.0e-3)
    assert np.allclose(phases_4_mev, reference.phases[0], atol=3.0e-3, rtol=3.0e-3)


@pytest.mark.benchmark
def test_descouvemont_closed_channel_reduced_spectral_and_direct_paths_agree() -> None:
    """A reduced α + 12C setup agrees across the spectral and direct complex paths."""

    energies = np.asarray([4.0, 8.0], dtype=np.float64)
    channels = alpha_c12_channels()
    spectral_solver = lm.compile(
        mesh=lm.MeshSpec("legendre", "x", n=20, scale=11.0),
        channels=channels,
        operators=("T+L", "1/r^2"),
        solvers=("spectrum", "smatrix"),
        energies=energies,
        method="eig",
        V_is_complex=True,
        z1z2=(2, 6),
    )
    direct_solver = lm.compile(
        mesh=lm.MeshSpec("legendre", "x", n=20, scale=11.0),
        channels=channels,
        operators=("T+L", "1/r^2"),
        solvers=("rmatrix_direct",),
        energies=energies,
        method="linear_solve",
        V_is_complex=True,
        z1z2=(2, 6),
    )
    spectral_potential = lm.assemble_local(
        spectral_solver.mesh,
        alpha_c12_potential,
        n_channels=len(channels),
    )
    direct_potential = lm.assemble_local(
        direct_solver.mesh,
        alpha_c12_potential,
        n_channels=len(channels),
    )

    assert spectral_solver.spectrum is not None
    assert spectral_solver.smatrix is not None

    spectral_smatrices = np.asarray(
        spectral_solver.smatrix(spectral_solver.spectrum(spectral_potential))
    )
    direct_smatrices, _ = _smatrix_from_direct_rmatrix(direct_solver, direct_potential)
    for energy_index, energy in enumerate(energies):
        open_count = alpha_c12_open_channel_count(float(energy))
        assert np.allclose(
            spectral_smatrices[energy_index, :open_count, :open_count],
            direct_smatrices[energy_index],
            atol=1.0e-8,
            rtol=1.0e-8,
        )

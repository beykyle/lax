from __future__ import annotations

import numpy as np
import pytest

import lax as lm
from lax.boundary import BoundaryValues
from lax.models import (
    ALPHA_C12_ROTOR_MODEL,
    channels_from_rotor_model,
    first_column_amplitudes_and_phases,
    interaction_from_rotor_model,
    open_channel_count,
)
from tests.benchmarks._descouvemont_fixtures import (
    CoupledColumnReference,
    load_alpha_c12_references,
    load_alpha_c12_single_interval_demo,
)

pytest.importorskip("jax")


def _solver(reference: CoupledColumnReference, method: str, solvers: tuple[str, ...]) -> lm.Solver:
    """Compile the Descouvemont Example 4 α + 12C benchmark."""

    return lm.compile(
        mesh=lm.MeshSpec(
            "legendre",
            "x",
            n=reference.n_basis,
            scale=reference.scale,
            extras={"n_intervals": reference.n_intervals},
        ),
        channels=channels_from_rotor_model(ALPHA_C12_ROTOR_MODEL),
        operators=("T+L", "1/r^2"),
        solvers=solvers,
        energies=reference.energies,
        method=method,
        V_is_complex=True,
        z1z2=(2, 6),
    )


def _smatrix_from_direct_rmatrix(
    solver: lm.Solver, potential
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    """Evaluate the physical open-channel S-matrices from the direct R-matrix kernel."""

    assert solver.rmatrix_direct is not None

    r_values = solver.rmatrix_direct(potential)
    smatrices: list[np.ndarray] = []
    projected_boundaries: list[np.ndarray] = []
    for energy_index in range(r_values.shape[0]):
        boundary = _boundary_at_energy(solver, energy_index)
        smatrix = lm.spectral.open_channel_smatrix_from_R(
            r_values[energy_index],
            boundary,
        )
        open_count = open_channel_count(ALPHA_C12_ROTOR_MODEL, float(solver.energies[energy_index]))
        smatrices.append(np.asarray(smatrix)[:open_count, :open_count])
        projected_boundaries.append(np.asarray(boundary.is_open))
    return tuple(smatrices), tuple(projected_boundaries)


def _boundary_at_energy(solver: lm.Solver, energy_index: int) -> BoundaryValues:
    """Return the boundary-value slice for one compile-time energy."""

    assert solver.boundary is not None
    return BoundaryValues(
        H_plus=solver.boundary.H_plus[energy_index],
        H_minus=solver.boundary.H_minus[energy_index],
        H_plus_p=solver.boundary.H_plus_p[energy_index],
        H_minus_p=solver.boundary.H_minus_p[energy_index],
        is_open=solver.boundary.is_open[energy_index],
        k=solver.boundary.k[energy_index],
    )


@pytest.mark.benchmark
@pytest.mark.parametrize(
    "reference",
    load_alpha_c12_references(),
    ids=["a9-n25-ns4", "a10-n25-ns4", "a11-n20-ns4"],
)
def test_descouvemont_closed_channel_matches_published_first_column(
    reference: CoupledColumnReference,
) -> None:
    """Published Descouvemont Example 4 values remain visible in the suite."""

    solver = _solver(reference, "linear_solve", ("rmatrix_direct",))
    interaction = interaction_from_rotor_model(ALPHA_C12_ROTOR_MODEL, solver)
    smatrices, projected_boundaries = _smatrix_from_direct_rmatrix(solver, interaction)

    for energy_index, energy in enumerate(reference.energies):
        open_count = open_channel_count(ALPHA_C12_ROTOR_MODEL, float(energy))
        amplitudes, phases = first_column_amplitudes_and_phases(smatrices[energy_index], open_count)
        amplitudes_match = np.allclose(
            amplitudes,
            reference.amplitudes[energy_index],
            atol=7.0e-3,
            rtol=7.0e-3,
        )
        phases_match = np.allclose(
            phases,
            reference.phases[energy_index],
            atol=3.0e-3,
            rtol=3.0e-3,
        )

        assert smatrices[energy_index].shape == (open_count, open_count)
        assert np.count_nonzero(projected_boundaries[energy_index]) == open_count
        assert amplitudes.shape[0] == reference.amplitudes[energy_index].shape[0]
        assert phases.shape[0] == reference.phases[energy_index].shape[0]
        assert np.all(np.isfinite(amplitudes))
        assert np.all(np.isfinite(phases))
        assert amplitudes_match
        assert phases_match


@pytest.mark.benchmark
def test_descouvemont_closed_channel_demo_matches_full_precision_reference() -> None:
    """The single-interval notebook regression stays locked to the checked-in full-precision output."""

    reference = load_alpha_c12_single_interval_demo()
    solver = _solver(reference, "linear_solve", ("rmatrix_direct",))
    interaction = interaction_from_rotor_model(ALPHA_C12_ROTOR_MODEL, solver)
    smatrices, _ = _smatrix_from_direct_rmatrix(solver, interaction)

    for energy_index, energy in enumerate(reference.energies):
        open_count = open_channel_count(ALPHA_C12_ROTOR_MODEL, float(energy))
        amplitudes, phases = first_column_amplitudes_and_phases(smatrices[energy_index], open_count)
        # 1e-10 is cross-platform achievable for complex 80×80 LAPACK solves;
        # 1e-12 asked for bit-exact reproducibility across CPU/BLAS variants.
        amplitudes_match = np.allclose(
            amplitudes,
            reference.amplitudes[energy_index],
            atol=1.0e-10,
            rtol=1.0e-10,
        )
        phases_match = np.allclose(
            phases,
            reference.phases[energy_index],
            atol=1.0e-10,
            rtol=1.0e-10,
        )

        assert smatrices[energy_index].shape == (open_count, open_count)
        assert amplitudes_match
        assert phases_match


@pytest.mark.benchmark
def test_descouvemont_closed_channel_reduced_spectral_and_direct_paths_agree() -> None:
    """A reduced α + 12C setup agrees across the spectral and direct complex paths."""

    energies = np.asarray([4.0, 8.0], dtype=np.float64)
    channels = channels_from_rotor_model(ALPHA_C12_ROTOR_MODEL)
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
    spectral_V = interaction_from_rotor_model(ALPHA_C12_ROTOR_MODEL, spectral_solver)
    direct_V = interaction_from_rotor_model(ALPHA_C12_ROTOR_MODEL, direct_solver)

    assert spectral_solver.spectrum is not None
    assert spectral_solver.smatrix is not None

    spectral_smatrices = np.asarray(spectral_solver.smatrix(spectral_solver.spectrum(spectral_V)))
    direct_smatrices, _ = _smatrix_from_direct_rmatrix(direct_solver, direct_V)
    for energy_index, energy in enumerate(energies):
        open_count = open_channel_count(ALPHA_C12_ROTOR_MODEL, float(energy))
        assert np.allclose(
            spectral_smatrices[energy_index, :open_count, :open_count],
            direct_smatrices[energy_index],
            atol=1.0e-8,
            rtol=1.0e-8,
        )

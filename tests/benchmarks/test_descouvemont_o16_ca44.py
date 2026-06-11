from __future__ import annotations

import numpy as np
import pytest

import lax as lm
from lax.boundary import BoundaryValues
from lax.models import (
    O16_CA44_ROTOR_MODEL,
    channels_from_rotor_model,
    first_column_amplitudes_and_phases,
    interaction_from_rotor_model,
    open_channel_count,
)
from tests.benchmarks._descouvemont_fixtures import (
    CoupledColumnReference,
    load_o16_ca44_references,
)

pytest.importorskip("jax")

# Descouvemont reference data was prepared with the rounded e² = 1.44; use it library-wide
# for this module (boundary Sommerfeld parameter and the rotor-model Coulomb potential).
pytestmark = pytest.mark.usefixtures("legacy_coulomb_constant")


def _solver(reference: CoupledColumnReference, method: str, solvers: tuple[str, ...]) -> lm.Solver:
    """Compile the Descouvemont Example 3 16O + 44Ca benchmark."""

    return lm.compile(
        mesh=lm.MeshSpec(
            "legendre",
            "x",
            n=reference.n_basis,
            scale=reference.scale,
            extras={"n_intervals": reference.n_intervals},
        ),
        channels=channels_from_rotor_model(O16_CA44_ROTOR_MODEL),
        operators=("T+L", "1/r^2"),
        solvers=solvers,
        energies=reference.energies,
        method=method,
        V_is_complex=True,
        z1z2=(8, 20),
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
        open_count = open_channel_count(O16_CA44_ROTOR_MODEL, float(solver.energies[energy_index]))
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
    load_o16_ca44_references(),
    ids=["a12-n25-ns4", "a13-n25-ns4", "a14-n25-ns4", "a14-n50-ns2"],
)
def test_descouvemont_o16_ca44_matches_published_output(
    reference: CoupledColumnReference,
) -> None:
    """Published Descouvemont Example 3 values remain visible in the suite."""

    solver = _solver(reference, "linear_solve", ("rmatrix_direct",))
    interaction = interaction_from_rotor_model(O16_CA44_ROTOR_MODEL, solver)
    smatrices, projected_boundaries = _smatrix_from_direct_rmatrix(solver, interaction)

    for energy_index, energy in enumerate(reference.energies):
        open_count = open_channel_count(O16_CA44_ROTOR_MODEL, float(energy))
        amplitudes, phases = first_column_amplitudes_and_phases(smatrices[energy_index], open_count)

        assert smatrices[energy_index].shape == (open_count, open_count)
        assert np.count_nonzero(projected_boundaries[energy_index]) == open_count
        assert np.allclose(amplitudes, reference.amplitudes[energy_index], atol=5.0e-4, rtol=5.0e-4)
        assert np.allclose(phases, reference.phases[energy_index], atol=5.0e-4, rtol=5.0e-4)

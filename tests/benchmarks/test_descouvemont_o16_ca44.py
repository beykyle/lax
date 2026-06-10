from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import lax as lm
from lax.boundary import BoundaryValues
from lax.models import (
    O16_CA44_ROTOR_MODEL,
    channels_from_rotor_model,
    first_column_amplitudes_and_phases,
    make_rotor_coupled_optical_potential,
    open_channel_count,
)
from tests.benchmarks._descouvemont_fixtures import (
    CoupledColumnReference,
    load_o16_ca44_references,
)

pytest.importorskip("jax")


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


def _rotor_interaction(solver: lm.Solver, fn) -> object:
    """Build an Interaction for the 4-channel O16+Ca44 rotor model from fn(r, c, cp)."""
    n_c = len(channels_from_rotor_model(O16_CA44_ROTOR_MODEL))
    N = solver.mesh.n
    M = n_c * N
    r = solver.mesh.radii
    block = jnp.zeros((M, M), dtype=jnp.complex128)
    for c in range(n_c):
        for cp in range(n_c):
            g = fn(r, c, cp)
            block = block.at[c * N : (c + 1) * N, cp * N : (cp + 1) * N].set(jnp.diag(g))
    assert solver.interaction_from_block is not None
    return solver.interaction_from_block(block, energy_dependent=False)


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
    k_values = None if solver.boundary.k is None else solver.boundary.k[energy_index]
    return BoundaryValues(
        H_plus=solver.boundary.H_plus[energy_index],
        H_minus=solver.boundary.H_minus[energy_index],
        H_plus_p=solver.boundary.H_plus_p[energy_index],
        H_minus_p=solver.boundary.H_minus_p[energy_index],
        is_open=solver.boundary.is_open[energy_index],
        k=k_values,
    )


@pytest.mark.benchmark
@pytest.mark.parametrize(
    "reference",
    load_o16_ca44_references(),
    ids=["a12-n25-ns4", "a13-n25-ns4", "a14-n25-ns4", "a14-n50-ns2"],
)
def test_descouvemont_o16_ca44_matches_published_output(reference: CoupledColumnReference) -> None:
    """Published Descouvemont Example 3 values remain visible in the suite."""

    potential = make_rotor_coupled_optical_potential(O16_CA44_ROTOR_MODEL)
    solver = _solver(reference, "linear_solve", ("rmatrix_direct",))
    interaction = _rotor_interaction(solver, potential)
    smatrices, projected_boundaries = _smatrix_from_direct_rmatrix(solver, interaction)

    for energy_index, energy in enumerate(reference.energies):
        open_count = open_channel_count(O16_CA44_ROTOR_MODEL, float(energy))
        amplitudes, phases = first_column_amplitudes_and_phases(smatrices[energy_index], open_count)

        assert smatrices[energy_index].shape == (open_count, open_count)
        assert np.count_nonzero(projected_boundaries[energy_index]) == open_count
        assert np.allclose(amplitudes, reference.amplitudes[energy_index], atol=5.0e-4, rtol=5.0e-4)
        assert np.allclose(phases, reference.phases[energy_index], atol=5.0e-4, rtol=5.0e-4)

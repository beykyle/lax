from __future__ import annotations

import jax
import numpy as np
import pytest

import lax as lm
from lax._descouvemont_utils import np_j1_channels, reid_np_j1_potential
from lax.boundary import BoundaryValues
from tests.benchmarks._descouvemont_cases import (
    NP_J1_REFERENCE_A7,
    NP_J1_REFERENCE_A8,
    NpJ1Reference,
)

pytest.importorskip("jax")


def _solver(reference: NpJ1Reference, method: str, solvers: tuple[str, ...]) -> lm.Solver:
    """Compile the Descouvemont Example 2 n-p J=1+ benchmark."""

    return lm.compile(
        mesh=lm.MeshSpec("legendre", "x", n=reference.n_basis, scale=reference.scale),
        channels=np_j1_channels(),
        operators=("T+L", "1/r^2"),
        solvers=solvers,
        energies=reference.energies,
        method=method,
    )


def _smatrix_from_direct_rmatrix(solver: lm.Solver, potential: jax.Array) -> np.ndarray:
    """Evaluate the collision matrix from the direct R-matrix kernel."""

    assert solver.rmatrix_direct is not None
    assert solver.boundary is not None

    r_values = solver.rmatrix_direct(potential)
    smatrices = []
    for energy_index in range(r_values.shape[0]):
        boundary = BoundaryValues(
            H_plus=solver.boundary.H_plus[energy_index],
            H_minus=solver.boundary.H_minus[energy_index],
            H_plus_p=solver.boundary.H_plus_p[energy_index],
            H_minus_p=solver.boundary.H_minus_p[energy_index],
            is_open=solver.boundary.is_open[energy_index],
        )
        smatrix = lm.spectral.smatrix_from_R(r_values[energy_index], boundary)
        smatrices.append(np.asarray(smatrix))
    return np.stack(smatrices)


def _paper_observables(smatrices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the quantities printed in Descouvemont Appendix B."""

    phase_11 = []
    phase_22 = []
    for smatrix in smatrices:
        parameters = lm.spectral.coupled_channel_parameters_from_S(smatrix)
        phase_11.append(float(np.asarray(parameters.phase_2)))
        phase_22.append(float(np.asarray(parameters.phase_1)))
    eta_12 = np.abs(smatrices[:, 0, 1])
    return np.asarray(phase_11), np.asarray(phase_22), eta_12


@pytest.mark.benchmark
@pytest.mark.parametrize(
    ("reference", "atol"),
    [
        (NP_J1_REFERENCE_A7, 3.0e-3),
        (NP_J1_REFERENCE_A8, 3.0e-3),
    ],
    ids=["a7", "a8"],
)
def test_descouvemont_np_matches_appendix_b(
    reference: NpJ1Reference,
    atol: float,
) -> None:
    """The coupled n-p eigenphases reproduce Descouvemont Appendix B."""

    solver = _solver(reference, "eigh", ("spectrum", "smatrix"))
    potential = lm.assemble_local(solver.mesh, reid_np_j1_potential, n_channels=2)

    assert solver.spectrum is not None
    assert solver.smatrix is not None

    smatrices = np.asarray(solver.smatrix(solver.spectrum(potential)))
    phase_11, phase_22, eta_12 = _paper_observables(smatrices)

    assert np.allclose(phase_11, reference.phase_11, atol=atol, rtol=0.0)
    assert np.allclose(phase_22, reference.phase_22, atol=atol, rtol=0.0)
    assert np.all(eta_12 > 0.0)


@pytest.mark.benchmark
@pytest.mark.parametrize("reference", [NP_J1_REFERENCE_A7, NP_J1_REFERENCE_A8], ids=["a7", "a8"])
def test_descouvemont_np_spectral_and_direct_paths_agree(reference: NpJ1Reference) -> None:
    """The n-p spectral and direct coupled-channel paths agree."""

    spectral_solver = _solver(reference, "eigh", ("spectrum", "smatrix"))
    direct_solver = _solver(reference, "linear_solve", ("rmatrix_direct",))
    spectral_potential = lm.assemble_local(spectral_solver.mesh, reid_np_j1_potential, n_channels=2)
    direct_potential = lm.assemble_local(direct_solver.mesh, reid_np_j1_potential, n_channels=2)

    assert spectral_solver.spectrum is not None
    assert spectral_solver.smatrix is not None

    spectral_smatrices = np.asarray(
        spectral_solver.smatrix(spectral_solver.spectrum(spectral_potential))
    )
    direct_smatrices = _smatrix_from_direct_rmatrix(direct_solver, direct_potential)

    assert np.allclose(spectral_smatrices, direct_smatrices, atol=1.0e-10, rtol=1.0e-10)

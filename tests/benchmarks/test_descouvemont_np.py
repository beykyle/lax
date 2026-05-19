from __future__ import annotations

import jax
import numpy as np
import pytest

import lax as lm
from lax.boundary import BoundaryValues
from lax.models import reid_np_j1_channels, reid_np_j1_potential
from tests.benchmarks._descouvemont_fixtures import NpJ1Reference, load_np_j1_references

pytest.importorskip("jax")


def _solver(reference: NpJ1Reference, method: str, solvers: tuple[str, ...]) -> lm.Solver:
    """Compile the Descouvemont Example 2 n-p J=1+ benchmark."""

    return lm.compile(
        mesh=lm.MeshSpec(
            "legendre",
            "x",
            n=reference.n_basis,
            scale=reference.scale,
            extras={"n_intervals": reference.n_intervals},
        ),
        channels=reid_np_j1_channels(),
        operators=("T+L", "1/r^2"),
        solvers=solvers,
        energies=reference.energies,
        method=method,
    )


def _smatrix_from_direct_rmatrix(solver: lm.Solver, potential: jax.Array) -> np.ndarray:
    """Evaluate the collision matrix from the direct R-matrix kernel."""

    assert solver.rmatrix_direct is not None

    r_values = solver.rmatrix_direct(potential)
    smatrices = []
    for energy_index in range(r_values.shape[0]):
        smatrix = lm.spectral.open_channel_smatrix_from_R(
            r_values[energy_index],
            _boundary_at_energy(solver, energy_index),
        )
        smatrices.append(np.asarray(smatrix))
    return np.stack(smatrices)


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


def _paper_observables_from_direct(
    reference: NpJ1Reference,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the published observables from the direct R-matrix path."""

    solver = _solver(reference, "linear_solve", ("rmatrix_direct",))
    potential = lm.assemble_local(solver.mesh, reid_np_j1_potential, n_channels=2)
    smatrices = _smatrix_from_direct_rmatrix(solver, potential)
    return _paper_observables(smatrices)


@pytest.mark.benchmark
@pytest.mark.parametrize(
    "reference",
    load_np_j1_references(),
    ids=["a7-n60-ns1", "a7-n30-ns2", "a8-n25-ns3"],
)
def test_descouvemont_np_matches_appendix_b(reference: NpJ1Reference) -> None:
    """The coupled n-p eigenphases reproduce the published Appendix B values."""

    phase_11, phase_22, eta_12 = _paper_observables_from_direct(reference)

    assert np.allclose(phase_11, reference.phase_11, atol=3.0e-3, rtol=0.0)
    assert np.allclose(phase_22, reference.phase_22, atol=3.0e-3, rtol=0.0)


@pytest.mark.benchmark
@pytest.mark.parametrize(
    "reference",
    load_np_j1_references(),
    ids=["a7-n60-ns1", "a7-n30-ns2", "a8-n25-ns3"],
)
def test_descouvemont_np_eta12_matches_appendix_b(reference: NpJ1Reference) -> None:
    """The published `eta_12 = |S_12|` values reproduce Appendix B."""

    _, _, eta_12 = _paper_observables_from_direct(reference)

    assert np.allclose(eta_12, reference.eta_12, atol=3.0e-3, rtol=0.0)


@pytest.mark.benchmark
def test_descouvemont_np_spectral_and_direct_paths_agree() -> None:
    """The n-p spectral and direct coupled-channel paths agree."""

    reference = load_np_j1_references()[0]
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

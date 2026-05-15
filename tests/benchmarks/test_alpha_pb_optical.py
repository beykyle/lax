from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import lax as lm
from lax.boundary import BoundaryValues

pytest.importorskip("jax")

OPTICAL_ENERGIES = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float64)
APPENDIX_A_S = np.array(
    [
        1.0000e00 + 5.9801e-19j,
        1.0000e00 + 7.4950e-07j,
        9.9893e-01 + 9.0496e-03j,
        6.5081e-01 + 2.9560e-01j,
        6.4367e-02 + 4.1130e-02j,
    ],
    dtype=np.complex128,
)
ALPHA_PB_MASS_FACTOR = 20.736 / (4.0 * 208.0 / (4.0 + 208.0))


def _optical_potential(r: jax.Array, imag_depth: float) -> jax.Array:
    """α + 208Pb optical potential from Descouvemont eq. 47, in MeV."""

    v0 = 100.0
    radius = 1.1132 * (208.0 ** (1.0 / 3.0) + 4.0 ** (1.0 / 3.0))
    diffuseness = 0.5803
    woods_saxon = 1.0 / (1.0 + jnp.exp((r - radius) / diffuseness))
    coulomb = 2.0 * 82.0 * 1.44 / r
    return -v0 * woods_saxon - 1.0j * imag_depth * woods_saxon + coulomb


def _complex_solver(method: str, solvers: tuple[str, ...]):
    return lm.compile(
        mesh=lm.MeshSpec("legendre", "x", n=60, scale=14.0),
        channels=(lm.ChannelSpec(l=20, threshold=0.0, mass_factor=ALPHA_PB_MASS_FACTOR),),
        operators=("T+L",),
        solvers=solvers,
        energies=OPTICAL_ENERGIES,
        V_is_complex=True,
        method=method,
        z1z2=(2, 82),
    )


def _real_solver(method: str, solvers: tuple[str, ...]):
    return lm.compile(
        mesh=lm.MeshSpec("legendre", "x", n=60, scale=14.0),
        channels=(lm.ChannelSpec(l=20, threshold=0.0, mass_factor=ALPHA_PB_MASS_FACTOR),),
        operators=("T+L",),
        solvers=solvers,
        energies=OPTICAL_ENERGIES,
        method=method,
        z1z2=(2, 82),
    )


def _smatrix_from_direct_rmatrix(solver, potential: jax.Array) -> np.ndarray:
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


@pytest.mark.benchmark
def test_alpha_pb_optical_eig_matches_appendix_a() -> None:
    """The complex-spectrum path reproduces Descouvemont Appendix A."""

    solver = _complex_solver("eig", ("spectrum", "smatrix"))
    potential = lm.assemble_local(solver.mesh, lambda r: _optical_potential(r, imag_depth=10.0))

    assert solver.spectrum is not None
    assert solver.smatrix is not None

    smatrix = np.asarray(solver.smatrix(solver.spectrum(potential)))[:, 0, 0]

    assert np.allclose(smatrix, APPENDIX_A_S, atol=1.0e-4, rtol=1.0e-4)


@pytest.mark.benchmark
def test_alpha_pb_optical_direct_matches_appendix_a() -> None:
    """The direct linear-solve path reproduces Descouvemont Appendix A."""

    solver = _complex_solver("linear_solve", ("rmatrix_direct",))
    potential = lm.assemble_local(solver.mesh, lambda r: _optical_potential(r, imag_depth=10.0))
    smatrix = _smatrix_from_direct_rmatrix(solver, potential)[:, 0, 0]

    assert np.allclose(smatrix, APPENDIX_A_S, atol=1.0e-4, rtol=1.0e-4)


@pytest.mark.benchmark
def test_alpha_pb_optical_method_paths_agree() -> None:
    """The `eigh`, `eig`, and `linear_solve` paths stay consistent where they overlap."""

    real_spectrum_solver = _real_solver("eigh", ("spectrum", "smatrix"))
    real_direct_solver = _real_solver("linear_solve", ("rmatrix_direct",))
    complex_solver = _complex_solver("eig", ("spectrum", "smatrix"))
    complex_direct_solver = _complex_solver("linear_solve", ("rmatrix_direct",))
    real_potential = lm.assemble_local(
        real_spectrum_solver.mesh,
        lambda r: jnp.real(_optical_potential(r, imag_depth=0.0)),
    )
    complex_potential = lm.assemble_local(
        complex_solver.mesh,
        lambda r: _optical_potential(r, imag_depth=10.0),
    )

    assert real_spectrum_solver.spectrum is not None
    assert real_spectrum_solver.smatrix is not None
    assert complex_solver.spectrum is not None
    assert complex_solver.smatrix is not None

    real_smatrix = np.asarray(
        real_spectrum_solver.smatrix(real_spectrum_solver.spectrum(real_potential))
    )
    real_direct = _smatrix_from_direct_rmatrix(real_direct_solver, real_potential)
    complex_smatrix = np.asarray(complex_solver.smatrix(complex_solver.spectrum(complex_potential)))
    complex_direct = _smatrix_from_direct_rmatrix(complex_direct_solver, complex_potential)

    assert np.allclose(real_smatrix, real_direct, atol=1.0e-10, rtol=1.0e-10)
    assert np.allclose(complex_smatrix, complex_direct, atol=1.0e-10, rtol=1.0e-10)

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.special as sc

import lax as lm
from lax.boundary._types import Solver

pytest.importorskip("jax")
pytest.importorskip("scipy")

GRID_POINTS = 4000
GRID_MAX_RADIUS = 40.0
HBAR2_2MU = 0.5


def _hydrogen_solver(
    angular_momentum: int,
    *,
    with_grid: bool = False,
    momenta: jax.Array | None = None,
) -> Solver:
    """Compile the Laguerre hydrogen solver for one partial wave."""

    grid = jnp.linspace(0.0, GRID_MAX_RADIUS, GRID_POINTS) if with_grid else None
    solvers = ("spectrum", "wavefunction") if (with_grid or momenta is not None) else ("spectrum",)
    return lm.compile(
        mesh=lm.MeshSpec("laguerre", "x", n=30, scale=2.0),
        channels=(lm.ChannelSpec(l=angular_momentum, threshold=0.0, mass_factor=HBAR2_2MU),),
        operators=("T", "1/r"),
        solvers=solvers,
        grid=grid,
        momenta=momenta,
    )


def _hydrogen_potential(solver: Solver) -> jax.Array:
    """Return the Coulomb potential for one compiled hydrogen solver."""

    return jnp.asarray((-1.0 / solver.mesh.radii)[None, None, :])


def _hydrogen_radial_wavefunction(n: int, angular_momentum: int, radii: np.ndarray) -> np.ndarray:
    """Return the normalized internal hydrogen radial wavefunction `u_{nl}(r)`."""

    rho = 2.0 * radii / float(n)
    prefactor = (
        2.0
        / (n**2)
        * math.sqrt(math.factorial(n - angular_momentum - 1) / math.factorial(n + angular_momentum))
    )
    radial = (
        prefactor
        * np.exp(-0.5 * rho)
        * rho**angular_momentum
        * sc.eval_genlaguerre(n - angular_momentum - 1, 2 * angular_momentum + 1, rho)
    )
    return radii * radial


def _hydrogen_momentum_wavefunction(
    n: int,
    angular_momentum: int,
    momenta: np.ndarray,
) -> np.ndarray:
    """Return analytic hydrogen momentum-space amplitudes for low-lying states."""

    if n == 1 and angular_momentum == 0:
        return np.sqrt(2.0 / np.pi) * 2.0 / (1.0 + momenta**2)

    if n == 2 and angular_momentum == 0:
        denominator = momenta**2 + 0.25
        return np.sqrt(1.0 / np.pi) * (momenta**2 - 0.25) / (denominator**2)

    if n == 2 and angular_momentum == 1:
        denominator = momenta**2 + 0.25
        return np.sqrt(2.0 / (6.0 * np.pi)) * momenta / (denominator**2)

    msg = f"No analytic momentum-space hydrogen wavefunction is defined for (n, l)=({n}, {angular_momentum})."
    raise ValueError(msg)


def _normalized_and_aligned(
    numerical: np.ndarray,
    analytic: np.ndarray,
    radii: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize two wavefunctions and align the numerical sign to the analytic one."""

    numerical_norm = math.sqrt(float(np.trapezoid(np.abs(numerical) ** 2, radii)))
    analytic_norm = math.sqrt(float(np.trapezoid(np.abs(analytic) ** 2, radii)))
    normalized_numerical = numerical / numerical_norm
    normalized_analytic = analytic / analytic_norm
    overlap = float(np.trapezoid(normalized_numerical * normalized_analytic, radii))
    sign = -1.0 if overlap < 0.0 else 1.0
    return sign * normalized_numerical, normalized_analytic


def _normalized_and_aligned_on_grid(
    numerical: np.ndarray,
    analytic: np.ndarray,
    grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize two one-dimensional profiles on a shared grid and align the sign."""

    numerical_norm = math.sqrt(float(np.trapezoid(np.abs(numerical) ** 2, grid)))
    analytic_norm = math.sqrt(float(np.trapezoid(np.abs(analytic) ** 2, grid)))
    normalized_numerical = numerical / numerical_norm
    normalized_analytic = analytic / analytic_norm
    overlap = float(np.trapezoid(normalized_numerical * normalized_analytic, grid))
    sign = -1.0 if overlap < 0.0 else 1.0
    return sign * normalized_numerical, normalized_analytic


@pytest.mark.benchmark
def test_hydrogen_ground_state_laguerre_x() -> None:
    """Laguerre-x reproduces the hydrogen 1s energy. [DESIGN.md §16.3]"""

    solver = _hydrogen_solver(0)

    assert solver.spectrum is not None
    spectrum = solver.spectrum(_hydrogen_potential(solver))
    ground_state = float(np.asarray(spectrum.eigenvalues)[0]) * solver.channels[0].mass_factor

    assert abs(ground_state + 0.5) < 1.0e-10


@pytest.mark.benchmark
@pytest.mark.parametrize(
    ("angular_momentum", "principal_quantum_numbers"),
    [
        (0, (1, 2, 3)),
        (1, (2, 3)),
    ],
)
def test_hydrogen_bound_state_energies(
    angular_momentum: int,
    principal_quantum_numbers: tuple[int, ...],
) -> None:
    """Laguerre-x reproduces low-lying hydrogen energies across `s` and `p` waves."""

    solver = _hydrogen_solver(angular_momentum)

    assert solver.spectrum is not None
    spectrum = solver.spectrum(_hydrogen_potential(solver))
    physical_energies = np.asarray(spectrum.eigenvalues) * solver.channels[0].mass_factor
    expected = np.asarray([-0.5 / (n**2) for n in principal_quantum_numbers], dtype=np.float64)

    assert np.allclose(
        physical_energies[: len(principal_quantum_numbers)],
        expected,
        atol=1.0e-10,
        rtol=1.0e-10,
    )


@pytest.mark.benchmark
@pytest.mark.parametrize(
    ("angular_momentum", "state_index", "principal_quantum_number"),
    [
        (0, 0, 1),
        (0, 1, 2),
        (1, 0, 2),
    ],
)
def test_hydrogen_wavefunctions_match_analytic_radial_forms(
    angular_momentum: int,
    state_index: int,
    principal_quantum_number: int,
) -> None:
    """Laguerre-x bound-state wavefunctions agree with analytic hydrogen forms."""

    solver = _hydrogen_solver(angular_momentum, with_grid=True)

    assert solver.spectrum is not None
    assert solver.to_grid_vector is not None
    assert solver.transforms.grid_r is not None

    spectrum = solver.spectrum(_hydrogen_potential(solver))
    assert spectrum.eigenvectors is not None

    eigenvector = np.asarray(spectrum.eigenvectors)[:, state_index]
    numerical = np.asarray(solver.to_grid_vector(jnp.asarray(eigenvector)))
    radii = np.asarray(solver.transforms.grid_r)
    analytic = _hydrogen_radial_wavefunction(principal_quantum_number, angular_momentum, radii)
    aligned_numerical, normalized_analytic = _normalized_and_aligned(numerical, analytic, radii)

    relative_l2_error = np.linalg.norm(aligned_numerical - normalized_analytic) / np.linalg.norm(
        normalized_analytic
    )

    assert relative_l2_error < 3.0e-2


@pytest.mark.benchmark
@pytest.mark.parametrize(
    ("angular_momentum", "state_index", "principal_quantum_number"),
    [
        (0, 0, 1),
        (0, 1, 2),
        (1, 0, 2),
    ],
)
def test_hydrogen_wavefunctions_match_analytic_momentum_forms(
    angular_momentum: int,
    state_index: int,
    principal_quantum_number: int,
) -> None:
    """Laguerre-x bound-state momentum amplitudes agree with analytic hydrogen forms."""

    momenta = jnp.linspace(0.0, 6.0, 600)
    solver = _hydrogen_solver(angular_momentum, momenta=momenta)

    assert solver.spectrum is not None
    assert solver.fourier is not None
    assert solver.transforms.momenta is not None

    spectrum = solver.spectrum(_hydrogen_potential(solver))
    assert spectrum.eigenvectors is not None

    eigenvector = np.asarray(spectrum.eigenvectors)[:, state_index]
    numerical = np.asarray(solver.fourier(jnp.asarray(eigenvector)))
    momenta_np = np.asarray(solver.transforms.momenta)
    analytic = _hydrogen_momentum_wavefunction(
        principal_quantum_number,
        angular_momentum,
        momenta_np,
    )
    aligned_numerical, normalized_analytic = _normalized_and_aligned_on_grid(
        numerical,
        analytic,
        momenta_np,
    )

    relative_l2_error = np.linalg.norm(aligned_numerical - normalized_analytic) / np.linalg.norm(
        normalized_analytic
    )

    assert relative_l2_error < 5.0e-2


@pytest.mark.benchmark
def test_hydrogen_momentum_norm_matches_current_fourier_convention() -> None:
    """Hydrogen 1s raw Fourier norm matches the current partial-wave convention."""

    momenta = jnp.linspace(0.0, 6.0, 600)
    solver = _hydrogen_solver(0, with_grid=True, momenta=momenta)

    assert solver.spectrum is not None
    assert solver.to_grid_vector is not None
    assert solver.fourier is not None
    assert solver.transforms.grid_r is not None
    assert solver.transforms.momenta is not None

    spectrum = solver.spectrum(_hydrogen_potential(solver))
    assert spectrum.eigenvectors is not None

    eigenvector = jnp.asarray(np.asarray(spectrum.eigenvectors)[:, 0])
    numerical_r = np.asarray(solver.to_grid_vector(eigenvector))
    numerical_k = np.asarray(solver.fourier(eigenvector))
    radii = np.asarray(solver.transforms.grid_r)
    momenta_np = np.asarray(solver.transforms.momenta)
    analytic_k = _hydrogen_momentum_wavefunction(1, 0, momenta_np)

    r_norm = float(np.trapezoid(np.abs(numerical_r) ** 2, radii))
    numerical_k_norm = float(np.trapezoid(np.abs(numerical_k) ** 2, momenta_np))
    analytic_k_norm = float(np.trapezoid(np.abs(analytic_k) ** 2, momenta_np))

    assert abs(r_norm - 1.0) < 1.0e-6
    assert abs(numerical_k_norm - analytic_k_norm) < 2.0e-6

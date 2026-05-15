from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.integrate as si
import scipy.special as sc

import lax as lm
from lax.boundary._types import Solver
from lax.meshes._basis_eval import basis_at
from lax.transforms.fourier import compute_F_momentum

pytest.importorskip("jax")
pytest.importorskip("scipy")


type RadialProfile = Callable[[np.ndarray], np.ndarray]
type MomentumProfile = Callable[[np.ndarray], np.ndarray]


def _coefficients_for_profile(solver: Solver, profile: RadialProfile) -> jax.Array:
    """Return mesh coefficients that interpolate an analytic radial profile."""

    assert solver.from_grid_vector is not None
    return solver.from_grid_vector(
        lambda radii: jnp.asarray(profile(np.asarray(radii)))  # pyright: ignore[reportUnknownLambdaType] -- Callable-based sampling is part of the public helper API.
    )


def _analytic_fourier_cases() -> list[
    tuple[str, int, RadialProfile, MomentumProfile, float, float]
]:
    """Return analytic Legendre-x Fourier reference cases."""

    beta = 0.5
    gaussian_l0 = (
        "gaussian-l0",
        0,
        lambda radii: radii**2 * np.exp(-beta * radii**2),
        lambda momenta: np.exp(-(momenta**2) / (4.0 * beta)) / (2.0 * beta) ** 1.5,
        2.0e-6,
        2.0e-6,
    )
    ho_like_l0 = (
        "gaussian-polynomial-l0",
        0,
        lambda radii: radii**2 * (3.0 / (2.0 * beta) - radii**2) * np.exp(-beta * radii**2),
        lambda momenta: (
            ((momenta**2) / (4.0 * beta**2))
            * np.exp(-(momenta**2) / (4.0 * beta))
            / (2.0 * beta) ** 1.5
        ),
        5.0e-5,
        1.0e-4,
    )
    gaussian_l1 = (
        "gaussian-l1",
        1,
        lambda radii: radii**3 * np.exp(-beta * radii**2),
        lambda momenta: momenta * np.exp(-(momenta**2) / (4.0 * beta)) / (2.0 * beta) ** 2.5,
        3.0e-6,
        1.0e-5,
    )
    return [gaussian_l0, ho_like_l0, gaussian_l1]


def test_compute_f_momentum_matches_direct_quadrature() -> None:
    """`compute_F_momentum` matches the defining quadrature for Legendre-x."""

    mesh, _ = lm.meshes.build_mesh("legendre", "x", n=5, scale=6.0, operators={"T+L"})
    momenta = jnp.asarray([0.2, 0.5, 1.0])

    result = np.asarray(compute_F_momentum(mesh, momenta, angular_momentum=0, n_quad=300))

    x_q, w_q = np.polynomial.legendre.leggauss(300)
    r_q = 0.5 * mesh.scale * (x_q + 1.0)
    w_q = 0.5 * mesh.scale * w_q
    basis_values = np.asarray(basis_at(mesh, jnp.asarray(r_q)))
    expected = []
    for momentum in np.asarray(momenta):
        bessel = sc.spherical_jn(0, momentum * r_q)
        expected.append(np.sqrt(2.0 / np.pi) * (w_q @ (bessel[:, None] * basis_values)))

    assert np.allclose(result, np.stack(expected), atol=1.0e-10, rtol=1.0e-10)


def test_compute_f_momentum_matches_half_line_quadrature_for_laguerre() -> None:
    """`compute_F_momentum` matches direct half-line integration for Laguerre-x."""

    mesh, _ = lm.meshes.build_mesh("laguerre", "x", n=4, scale=2.0, operators={"T"})
    momenta = jnp.asarray([0.2, 0.5, 1.0])

    result = np.asarray(compute_F_momentum(mesh, momenta, angular_momentum=0, n_quad=400))

    expected = np.zeros_like(result)
    for momentum_index, momentum in enumerate(np.asarray(momenta)):
        for basis_index in range(mesh.n):

            def integrand(radius: float) -> float:
                basis_value = float(
                    np.asarray(basis_at(mesh, jnp.asarray([radius])))[0, basis_index]
                )
                bessel_value = float(sc.spherical_jn(0, momentum * radius))
                return np.sqrt(2.0 / np.pi) * bessel_value * basis_value

            expected[momentum_index, basis_index] = si.quad(
                integrand,
                0.0,
                np.inf,
                epsabs=1.0e-10,
                epsrel=1.0e-10,
                limit=200,
            )[0]

    assert np.allclose(result, expected, atol=2.0e-7, rtol=1.0e-7)


@pytest.mark.parametrize(
    ("case_name", "angular_momentum", "profile", "reference", "atol", "rtol"),
    _analytic_fourier_cases(),
)
def test_fourier_matches_analytic_profile(
    case_name: str,
    angular_momentum: int,
    profile: RadialProfile,
    reference: MomentumProfile,
    atol: float,
    rtol: float,
) -> None:
    """`solver.fourier` reproduces analytic Gaussian-family Hankel transforms."""

    del case_name
    momenta = jnp.linspace(0.0, 2.0, 21)
    mesh, _ = lm.meshes.build_mesh("legendre", "x", n=24, scale=18.0, operators={"T+L"})
    solver = lm.compile(
        mesh=lm.MeshSpec("legendre", "x", n=24, scale=18.0),
        channels=(lm.ChannelSpec(l=angular_momentum, threshold=0.0, mass_factor=2.0),),
        solvers=(),
        grid=mesh.radii,
        momenta=momenta,
    )

    assert solver.fourier is not None
    assert solver.from_grid_vector is not None

    coefficients = _coefficients_for_profile(solver, profile)
    transformed = np.asarray(solver.fourier(coefficients))
    expected = reference(np.asarray(momenta))

    assert np.allclose(transformed, expected, atol=atol, rtol=rtol)


def test_compile_binds_fourier_and_integrate() -> None:
    """`compile()` exposes Fourier, double-Fourier, and integration helpers."""

    solver = lm.compile(
        mesh=lm.MeshSpec("legendre", "x", n=5, scale=6.0),
        channels=(lm.ChannelSpec(l=0, threshold=0.0, mass_factor=2.0),),
        solvers=("spectrum",),
        momenta=jnp.asarray([0.2, 0.5, 1.0]),
    )

    assert solver.transforms.F_momentum is not None
    assert solver.transforms.momenta is not None
    assert solver.fourier is not None
    assert solver.double_fourier_transform is not None
    assert solver.integrate is not None

    values = jnp.asarray([1.0, -0.5, 0.25, 0.0, 0.5])
    kernel = jnp.diag(values)
    momentum_values = np.asarray(solver.fourier(values))
    transformed_kernel = np.asarray(solver.double_fourier_transform(kernel))
    norm = float(np.asarray(solver.integrate(values)))
    local_expectation = float(np.asarray(solver.integrate(values, jnp.linspace(1.0, 2.0, 5))))

    assert momentum_values.shape == (3,)
    assert transformed_kernel.shape == (3, 3)
    assert norm > 0.0
    assert local_expectation > 0.0


def test_double_fourier_transform_matches_same_channel_matrix_product() -> None:
    """`solver.double_fourier_transform` matches `F @ K @ F.T` for one channel."""

    solver = lm.compile(
        mesh=lm.MeshSpec("legendre", "x", n=5, scale=6.0),
        channels=(lm.ChannelSpec(l=0, threshold=0.0, mass_factor=2.0),),
        solvers=(),
        momenta=jnp.asarray([0.2, 0.5, 1.0]),
    )

    assert solver.transforms.F_momentum is not None
    assert solver.double_fourier_transform is not None

    kernel = jnp.asarray(
        [
            [1.0, 0.2, -0.1, 0.0, 0.0],
            [0.2, 0.8, 0.3, 0.1, 0.0],
            [-0.1, 0.3, 0.6, -0.2, 0.1],
            [0.0, 0.1, -0.2, 0.4, 0.2],
            [0.0, 0.0, 0.1, 0.2, 0.5],
        ]
    )
    F = np.asarray(solver.transforms.F_momentum[0])

    result = np.asarray(solver.double_fourier_transform(kernel))
    expected = F @ np.asarray(kernel) @ F.T

    assert np.allclose(result, expected, atol=1.0e-10, rtol=1.0e-10)


def test_double_fourier_transform_supports_mixed_channel_indices() -> None:
    """`solver.double_fourier_transform` uses different left/right channel matrices."""

    solver = lm.compile(
        mesh=lm.MeshSpec("legendre", "x", n=5, scale=6.0),
        channels=(
            lm.ChannelSpec(l=0, threshold=0.0, mass_factor=2.0),
            lm.ChannelSpec(l=1, threshold=0.0, mass_factor=2.0),
        ),
        solvers=(),
        momenta=jnp.asarray([0.2, 0.5, 1.0]),
    )

    assert solver.transforms.F_momentum is not None
    assert solver.double_fourier_transform is not None

    kernel = jnp.asarray(
        [
            [0.4, -0.1, 0.0, 0.2, 0.1],
            [0.0, 0.3, 0.2, -0.2, 0.0],
            [0.1, 0.0, 0.5, 0.1, -0.1],
            [0.2, -0.2, 0.0, 0.6, 0.3],
            [0.0, 0.1, -0.1, 0.3, 0.7],
        ]
    )
    F_left = np.asarray(solver.transforms.F_momentum[0])
    F_right = np.asarray(solver.transforms.F_momentum[1])

    result = np.asarray(
        solver.double_fourier_transform(
            kernel,
            left_channel_index=0,
            right_channel_index=1,
        )
    )
    expected = F_left @ np.asarray(kernel) @ F_right.T

    assert np.allclose(result, expected, atol=1.0e-10, rtol=1.0e-10)

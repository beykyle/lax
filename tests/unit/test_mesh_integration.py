from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest

import lax as lm
from lax.meshes import build_mesh

pytest.importorskip("jax")
pytest.importorskip("scipy")


def _physical_quadrature_weights(mesh: object) -> np.ndarray:
    """Return the physical quadrature weights for meshes with direct `dr = scale dx` mapping."""

    scale = float(getattr(mesh, "scale"))
    weights = np.asarray(getattr(mesh, "weights"))
    return scale * weights


def _nodal_coefficients(mesh: object, profile: np.ndarray) -> np.ndarray:
    """Return Lagrange-mesh coefficients for a sampled radial profile."""

    return np.sqrt(_physical_quadrature_weights(mesh)) * profile


@pytest.mark.parametrize("regularization", ["x", "x(1-x)", "x^3/2"])
def test_legendre_mesh_quadrature_integrates_polynomials_exactly(regularization: str) -> None:
    """Finite-interval Legendre meshes integrate low-degree polynomials exactly."""

    mesh, _ = build_mesh("legendre", regularization, n=6, scale=3.0, operators=set())
    radii = np.asarray(mesh.radii)
    weights = _physical_quadrature_weights(mesh)

    for power in (0, 1, 3, 5, 9, 11):
        approximate = float(np.sum(weights * radii**power))
        exact = mesh.scale ** (power + 1) / float(power + 1)
        assert math.isclose(approximate, exact, rel_tol=1e-9, abs_tol=5.0e-13)


def test_laguerre_x_mesh_quadrature_integrates_weighted_moments_exactly() -> None:
    """Laguerre-x meshes reproduce analytic exponential moments on the half-line."""

    mesh, _ = build_mesh("laguerre", "x", n=6, scale=2.0, operators=set())
    radii = np.asarray(mesh.radii)
    weights = _physical_quadrature_weights(mesh)
    scaled_exponential = np.exp(-radii / mesh.scale)

    for power in range(6):
        approximate = float(np.sum(weights * radii**power * scaled_exponential))
        exact = math.factorial(power) * mesh.scale ** (power + 1)
        assert math.isclose(approximate, exact, rel_tol=0.0, abs_tol=1.0e-11)


@pytest.mark.parametrize("regularization", ["x", "x(1-x)", "x^3/2"])
def test_solver_integrate_matches_polynomial_norms_for_legendre_meshes(
    regularization: str,
) -> None:
    """`solver.integrate` reproduces analytic polynomial norms on finite intervals."""

    solver = lm.compile(
        mesh=lm.MeshSpec("legendre", regularization, n=6, scale=3.0),
        channels=(lm.ChannelSpec(l=0, threshold=0.0, mass_factor=2.0),),
        solvers=(),
    )

    assert solver.integrate is not None

    radii = np.asarray(solver.mesh.radii)
    profile = radii**2
    coefficients = _nodal_coefficients(solver.mesh, profile)

    norm = float(np.asarray(solver.integrate(jnp.asarray(coefficients))))
    radial_moment = float(np.asarray(solver.integrate(jnp.asarray(coefficients), solver.mesh.radii)))

    assert math.isclose(norm, solver.mesh.scale**5 / 5.0, rel_tol=0.0, abs_tol=5.0e-13)
    assert math.isclose(
        radial_moment,
        solver.mesh.scale**6 / 6.0,
        rel_tol=0.0,
        abs_tol=5.0e-13,
    )


def test_solver_integrate_matches_exponential_moments_for_laguerre_x() -> None:
    """`solver.integrate` reproduces analytic Laguerre-x exponential moments."""

    solver = lm.compile(
        mesh=lm.MeshSpec("laguerre", "x", n=6, scale=2.0),
        channels=(lm.ChannelSpec(l=0, threshold=0.0, mass_factor=2.0),),
        solvers=(),
    )

    assert solver.integrate is not None

    radii = np.asarray(solver.mesh.radii)
    scale = float(solver.mesh.scale)
    profile = radii * np.exp(-0.5 * radii / scale)
    coefficients = _nodal_coefficients(solver.mesh, profile)

    norm = float(np.asarray(solver.integrate(jnp.asarray(coefficients))))
    radial_moment = float(np.asarray(solver.integrate(jnp.asarray(coefficients), solver.mesh.radii)))

    assert math.isclose(norm, 2.0 * scale**3, rel_tol=0.0, abs_tol=1.0e-11)
    assert math.isclose(radial_moment, 6.0 * scale**4, rel_tol=0.0, abs_tol=1.0e-10)

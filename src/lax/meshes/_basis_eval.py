"""Basis-function evaluation for grid and Fourier transforms."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import scipy.special as sc  # pyright: ignore[reportMissingTypeStubs] -- SciPy does not currently ship complete type stubs for special.
from numpy.polynomial import Legendre

from lax.boundary._types import Mesh

type BasisEvaluator = Callable[[Mesh, np.ndarray], np.ndarray]

_BASIS_EVALUATORS: dict[tuple[str, str], BasisEvaluator] = {}


def register_basis_evaluator(
    family: str, regularization: str
) -> Callable[[BasisEvaluator], BasisEvaluator]:
    """Register a basis evaluator for one mesh family/regularization pair."""

    def decorator(function: BasisEvaluator) -> BasisEvaluator:
        key = (family, regularization)
        if key in _BASIS_EVALUATORS:
            msg = f"Basis evaluator already registered for {key}."
            raise ValueError(msg)
        _BASIS_EVALUATORS[key] = function
        return function

    return decorator


def basis_at(mesh: Mesh, radii: jax.Array) -> jax.Array:
    """Evaluate mesh basis functions at one-dimensional physical radii."""

    radii_np = np.asarray(radii, dtype=np.float64)
    if radii_np.ndim != 1:
        msg = "`radii` must be one-dimensional for basis evaluation."
        raise ValueError(msg)

    key = (mesh.family, mesh.regularization)
    if key not in _BASIS_EVALUATORS:
        msg = f"No basis evaluator registered for {key}."
        raise ValueError(msg)

    values = _BASIS_EVALUATORS[key](mesh, radii_np)
    array: jax.Array = jnp.asarray(values)  # pyright: ignore[reportUnknownMemberType] -- JAX stubs expose asarray imprecisely for NumPy inputs.
    return array


@register_basis_evaluator("legendre", "x")
def legendre_x_basis_at(mesh: Mesh, radii: np.ndarray) -> np.ndarray:
    """Evaluate shifted-Legendre-x basis functions on a physical radial grid."""

    basis_size = mesh.n
    channel_radius = mesh.scale
    nodes = np.asarray(mesh.nodes)
    dimensionless_radii = radii / channel_radius
    legendre_arg = 2.0 * dimensionless_radii - 1.0
    legendre_values = sc.eval_legendre(basis_size, legendre_arg)

    parity = np.where(np.arange(basis_size) % 2 == 0, 1.0, -1.0)
    signs = -parity if basis_size % 2 == 1 else parity
    normalization = np.sqrt((1.0 - nodes) / nodes) / np.sqrt(channel_radius)

    numerator = (
        signs[None, :]
        * normalization[None, :]
        * dimensionless_radii[:, None]
        * legendre_values[:, None]
    )
    denominator = dimensionless_radii[:, None] - nodes[None, :]
    close_mask = np.isclose(denominator, 0.0)
    values = np.empty_like(numerator)
    np.divide(numerator, denominator, out=values, where=~close_mask)

    if np.any(close_mask):
        derivative = Legendre.basis(basis_size).deriv()
        derivative_values = derivative(2.0 * nodes - 1.0)
        limits = signs * normalization * nodes * 2.0 * derivative_values
        row_indices, col_indices = np.nonzero(close_mask)
        values[row_indices, col_indices] = limits[col_indices]

    return values


@register_basis_evaluator("laguerre", "x")
def laguerre_x_basis_at(mesh: Mesh, radii: np.ndarray) -> np.ndarray:
    """Evaluate regularized-Laguerre-x basis functions on a physical radial grid."""

    basis_size = mesh.n
    scale = mesh.scale
    nodes = np.asarray(mesh.nodes)
    weights = np.asarray(mesh.weights)
    dimensionless_radii = radii / scale
    laguerre_values = sc.eval_laguerre(basis_size, dimensionless_radii)

    derivative_values = -sc.eval_genlaguerre(basis_size - 1, 1.0, nodes)
    signs = np.sign(derivative_values)
    signs[signs == 0.0] = 1.0

    numerator = (
        signs[None, :]
        * dimensionless_radii[:, None]
        * laguerre_values[:, None]
        * np.exp(-0.5 * dimensionless_radii[:, None])
    )
    denominator = np.sqrt(scale * nodes)[None, :] * (
        dimensionless_radii[:, None] - nodes[None, :]
    )

    close_mask = np.isclose(dimensionless_radii[:, None], nodes[None, :])
    values = np.empty_like(numerator)
    np.divide(numerator, denominator, out=values, where=~close_mask)

    if np.any(close_mask):
        limits = 1.0 / np.sqrt(scale * weights)
        row_indices, col_indices = np.nonzero(close_mask)
        values[row_indices, col_indices] = limits[col_indices]

    return values


__all__ = ["basis_at", "register_basis_evaluator"]

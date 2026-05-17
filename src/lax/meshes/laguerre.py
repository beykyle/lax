"""Laguerre mesh builders."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import scipy.special as sc  # pyright: ignore[reportMissingTypeStubs] -- SciPy does not currently ship complete type stubs for special.

from lax.boundary._types import Mesh, OperatorMatrices

from ._registry import register


@register("laguerre", "x")
def build_laguerre_x(
    *,
    n: int,
    scale: float,
    operators: set[str],
    alpha: float = 0.0,
    **extras: object,
) -> tuple[Mesh, OperatorMatrices]:
    """Build Laguerre-x mesh data on `(0, ∞)`. [Baye eqs. 3.50, 3.61, 3.75-3.76]"""

    del extras
    if alpha != 0.0:
        msg = "Only alpha=0.0 is supported for the MVP Laguerre-x mesh."
        raise NotImplementedError(msg)

    h = float(scale)
    nodes, quadrature_weights = sc.roots_laguerre(n)
    weights = quadrature_weights * np.exp(nodes)
    radii = h * nodes

    kinetic = _laguerre_x_kinetic(nodes, h)
    mesh = Mesh(
        family="laguerre",
        regularization="x",
        n=n,
        scale=h,
        nodes=_to_jax_array(nodes),
        weights=_to_jax_array(weights),
        radii=_to_jax_array(radii),
        basis_at_boundary=_to_jax_array(np.zeros(n, dtype=np.float64)),
    )

    include_kinetic = bool({"T", "T+L", "TpL"} & operators)
    operators_out = OperatorMatrices(
        T=_to_jax_array(kinetic) if include_kinetic else None,
        TpL=_to_jax_array(kinetic) if include_kinetic else None,
        inv_r=_diagonal_operator(1.0 / radii) if {"1/r", "inv_r"} & operators else None,
        inv_r2=(
            _diagonal_operator(1.0 / (radii**2))
            if {"1/r^2", "1/r²", "inv_r2"} & operators
            else None
        ),
    )
    return mesh, operators_out


@register("laguerre", "modified_x^2")
def build_laguerre_modified_x2(
    *,
    n: int,
    scale: float,
    operators: set[str],
    alpha: float = 0.5,
    **extras: object,
) -> tuple[Mesh, OperatorMatrices]:
    """Build modified-Laguerre-x^2 mesh data. [Baye eqs. 3.82-3.84, 3.88-3.91]"""

    del extras
    if alpha != 0.5:
        msg = "Only alpha=0.5 is supported for the MVP modified Laguerre-x^2 mesh."
        raise NotImplementedError(msg)

    h = float(scale)
    squared_nodes, _ = sc.roots_genlaguerre(n, alpha)
    nodes = np.sqrt(squared_nodes)
    radii = h * nodes
    weights = _modified_laguerre_x2_weights(squared_nodes, nodes, n, alpha)
    kinetic = _modified_laguerre_x2_kinetic(nodes, h, n, alpha)

    mesh = Mesh(
        family="laguerre",
        regularization="modified_x^2",
        n=n,
        scale=h,
        nodes=_to_jax_array(nodes),
        weights=_to_jax_array(weights),
        radii=_to_jax_array(radii),
        basis_at_boundary=_to_jax_array(np.zeros(n, dtype=np.float64)),
    )

    include_kinetic = bool({"T", "T+L", "TpL"} & operators)
    operators_out = OperatorMatrices(
        T=_to_jax_array(kinetic) if include_kinetic else None,
        TpL=_to_jax_array(kinetic) if include_kinetic else None,
        inv_r=_diagonal_operator(1.0 / radii) if {"1/r", "inv_r"} & operators else None,
        inv_r2=(
            _diagonal_operator(1.0 / (radii**2))
            if {"1/r^2", "1/r²", "inv_r2"} & operators
            else None
        ),
    )
    return mesh, operators_out


def _laguerre_x_kinetic(nodes: np.ndarray, scale: float) -> np.ndarray:
    """Return the regularized-Laguerre kinetic matrix. [Baye eqs. 3.75, 3.76]"""

    n = nodes.size
    matrix = np.zeros((n, n), dtype=np.float64)

    diagonal = -(nodes**2 - 2.0 * (2.0 * n + 1.0) * nodes - 4.0) / (12.0 * nodes**2)
    np.fill_diagonal(matrix, diagonal)

    row_idx, col_idx = np.triu_indices(n, k=1)
    sign = np.where((row_idx - col_idx) % 2 == 0, 1.0, -1.0)
    off_diagonal = (
        sign
        * (nodes[row_idx] + nodes[col_idx])
        / (np.sqrt(nodes[row_idx] * nodes[col_idx]) * (nodes[row_idx] - nodes[col_idx]) ** 2)
    )
    matrix[row_idx, col_idx] = off_diagonal
    matrix[col_idx, row_idx] = off_diagonal
    return matrix / (scale**2)


def _modified_laguerre_x2_weights(
    squared_nodes: np.ndarray,
    nodes: np.ndarray,
    n: int,
    alpha: float,
) -> np.ndarray:
    """Return the modified-Laguerre-x^2 quadrature weights. [Baye eq. 3.84]"""

    polynomial_values = np.asarray(sc.eval_genlaguerre(n - 1, alpha, squared_nodes), dtype=np.float64)
    factorial_n = float(math.factorial(n))
    numerator = np.exp(squared_nodes) * np.exp(sc.gammaln(n + alpha + 1.0))
    denominator: np.ndarray = (
        2.0 ** (alpha - 1.0)
        * factorial_n
        * (n + alpha)
        * nodes**alpha
        * polynomial_values**2
    )
    return numerator / denominator


def _modified_laguerre_x2_kinetic(
    nodes: np.ndarray,
    scale: float,
    n: int,
    alpha: float,
) -> np.ndarray:
    """Return the modified-Laguerre-x^2 kinetic matrix. [Baye eqs. 3.88-3.91]"""

    basis_size = nodes.size
    matrix = np.zeros((basis_size, basis_size), dtype=np.float64)

    diagonal = (
        -nodes**2
        + 2.0 * (2.0 * n + alpha + 1.0)
        + (2.0 * alpha**2 - 2.0) / (nodes**2)
    ) / (3.0 * scale**2)
    np.fill_diagonal(matrix, diagonal)

    row_idx: np.ndarray
    col_idx: np.ndarray
    row_idx, col_idx = np.triu_indices(basis_size, k=1)
    sign = np.where((row_idx - col_idx) % 2 == 0, 1.0, -1.0)
    off_diagonal = (
        sign
        * 8.0
        * nodes[row_idx]
        * nodes[col_idx]
        / (scale**2 * (nodes[row_idx] ** 2 - nodes[col_idx] ** 2) ** 2)
    )
    matrix[row_idx, col_idx] = off_diagonal
    matrix[col_idx, row_idx] = off_diagonal
    return matrix


def _to_jax_array(values: np.ndarray) -> jax.Array:
    """Convert a NumPy array to a runtime JAX array with an explicit type."""

    array: jax.Array = jnp.asarray(values)  # pyright: ignore[reportUnknownMemberType] -- JAX stubs expose asarray imprecisely for NumPy inputs.
    return array


def _diagonal_operator(values: np.ndarray) -> jax.Array:
    """Construct a diagonal JAX operator from compile-time diagonal values."""

    diagonal = _to_jax_array(values)
    matrix: jax.Array = jnp.diag(diagonal)
    return matrix


__all__ = ["build_laguerre_modified_x2", "build_laguerre_x"]

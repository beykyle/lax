"""Laguerre mesh builders."""

from __future__ import annotations

import math

import numpy as np
import scipy.special as sc

from lax.meshes._registry import register
from lax.meshes._utils import diagonal_operator as _diagonal_operator
from lax.meshes._utils import to_jax_array as _to_jax_array
from lax.types import Mesh, OperatorMatrices


@register("laguerre", "x")
def build_laguerre_x(
    *,
    n: int,
    scale: float,
    operators: set[str],
    alpha: float = 0.0,
    **extras: object,
) -> tuple[Mesh, OperatorMatrices]:
    """Build Laguerre-``x`` mesh data on ``(0, ∞)``.

    Semi-infinite mesh with nodes ``r_i = h x_i`` where ``x_i`` are the
    Laguerre zeros and ``h`` is the scaling factor.  For ``α = 0`` the
    ``1/r`` operator is Gauss-exact, making this mesh ideal for Coulomb
    and hydrogen-like systems.  [Baye eqs. 3.50, 3.61, 3.75–3.76]

    Parameters
    ----------
    n
        Number of basis functions.
    scale
        Laguerre scale factor ``h`` in fm.
    operators
        Set of operator strings to precompute.  Supported:
        ``"T"``, ``"T+L"``, ``"1/r"``, ``"1/r^2"``.
    alpha
        Generalized Laguerre parameter ``α``.  Only ``α = 0`` is
        implemented in v1.
    **extras
        Currently unused; reserved for future extensions.

    Returns
    -------
    tuple[Mesh, OperatorMatrices]
        Compiled mesh and precomputed operator matrices.
    """

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
        n_intervals=1,
        basis_size_per_interval=n,
        nodes=_to_jax_array(nodes),
        weights=_to_jax_array(weights),
        radii=_to_jax_array(radii),
        basis_at_boundary=_to_jax_array(np.zeros(n, dtype=np.float64)),
        propagation=None,
    )

    include_kinetic = bool({"T", "T+L", "TpL"} & operators)
    operators_out = OperatorMatrices(
        T=_to_jax_array(kinetic) if include_kinetic else None,
        TpL=_to_jax_array(kinetic) if include_kinetic else None,
        inv_r=_diagonal_operator(1.0 / radii) if {"1/r", "inv_r"} & operators else None,
        inv_r2=_diagonal_operator(1.0 / (radii**2)),
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
    """Build modified Laguerre-``x²`` mesh data on ``(0, ∞)``.

    Uses the substitution ``t = x²`` so nodes are the positive roots of
    the generalized Laguerre polynomial in ``t``.  Well-suited for 3D
    harmonic-oscillator-like potentials.
    [Baye eqs. 3.82–3.84, 3.88–3.91]

    Parameters
    ----------
    n
        Number of basis functions.
    scale
        Scaling factor ``h`` in fm.
    operators
        Set of operator strings to precompute.  Supported:
        ``"T"``, ``"T+L"``, ``"1/r"``, ``"1/r^2"``.
    **extras
        Currently unused; reserved for future extensions.

    Returns
    -------
    tuple[Mesh, OperatorMatrices]
        Compiled mesh and precomputed operator matrices.
    """

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
        n_intervals=1,
        basis_size_per_interval=n,
        nodes=_to_jax_array(nodes),
        weights=_to_jax_array(weights),
        radii=_to_jax_array(radii),
        basis_at_boundary=_to_jax_array(np.zeros(n, dtype=np.float64)),
        propagation=None,
    )

    include_kinetic = bool({"T", "T+L", "TpL"} & operators)
    operators_out = OperatorMatrices(
        T=_to_jax_array(kinetic) if include_kinetic else None,
        TpL=_to_jax_array(kinetic) if include_kinetic else None,
        inv_r=_diagonal_operator(1.0 / radii) if {"1/r", "inv_r"} & operators else None,
        inv_r2=_diagonal_operator(1.0 / (radii**2)),
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

    polynomial_values = np.asarray(
        sc.eval_genlaguerre(n - 1, alpha, squared_nodes), dtype=np.float64
    )
    factorial_n = float(math.factorial(n))
    numerator = np.exp(squared_nodes) * np.exp(sc.gammaln(n + alpha + 1.0))
    denominator: np.ndarray = (
        2.0 ** (alpha - 1.0) * factorial_n * (n + alpha) * nodes**alpha * polynomial_values**2
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
        -(nodes**2) + 2.0 * (2.0 * n + alpha + 1.0) + (2.0 * alpha**2 - 2.0) / (nodes**2)
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


__all__ = ["build_laguerre_modified_x2", "build_laguerre_x"]

"""Shifted Legendre mesh builders."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from lax.boundary._types import Mesh, OperatorMatrices

from ._registry import register


@register("legendre", "x")
def build_legendre_x(
    *,
    n: int,
    scale: float,
    operators: set[str],
    **extras: object,
) -> tuple[Mesh, OperatorMatrices]:
    """Build shifted Legendre-x mesh data. [Desc. eqs. 22-24]"""

    del extras
    a = float(scale)

    x_raw: np.ndarray
    w_raw: np.ndarray
    x_raw, w_raw = np.polynomial.legendre.leggauss(n)
    nodes: np.ndarray = 0.5 * (x_raw + 1.0)
    weights: np.ndarray = 0.5 * w_raw
    radii: np.ndarray = a * nodes

    parity: np.ndarray = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    boundary_sign: np.ndarray = -parity if n % 2 == 1 else parity
    basis_at_boundary: np.ndarray = boundary_sign / np.sqrt(a * nodes * (1.0 - nodes))

    TpL: np.ndarray = _legendre_x_t_plus_l(nodes, a)
    D: np.ndarray = _legendre_x_derivative(nodes, a)

    nodes_array = _to_jax_array(nodes)
    weights_array = _to_jax_array(weights)
    radii_array = _to_jax_array(radii)
    boundary_array = _to_jax_array(basis_at_boundary)

    mesh = Mesh(
        family="legendre",
        regularization="x",
        n=n,
        scale=a,
        nodes=nodes_array,
        weights=weights_array,
        radii=radii_array,
        basis_at_boundary=boundary_array,
    )

    operators_out = OperatorMatrices(
        TpL=_to_jax_array(TpL) if {"T+L", "TpL"} & operators else None,
        D=_to_jax_array(D) if {"D", "d/dr"} & operators else None,
        inv_r=_diagonal_operator(1.0 / radii) if {"1/r", "inv_r"} & operators else None,
        inv_r2=(
            _diagonal_operator(1.0 / (radii**2))
            if {"1/r^2", "1/r²", "inv_r2"} & operators
            else None
        ),
    )
    return mesh, operators_out


@register("legendre", "x(1-x)")
def build_legendre_x_one_minus_x(
    *,
    n: int,
    scale: float,
    operators: set[str],
    **extras: object,
) -> tuple[Mesh, OperatorMatrices]:
    """Build shifted-Legendre-x(1-x) mesh data. [Baye eqs. 3.138, 3.142-3.144]"""

    del extras
    a = float(scale)
    nodes, weights, radii = _shifted_legendre_quadrature(n, a)
    kinetic = _legendre_x_one_minus_x_kinetic(nodes, a)

    mesh = Mesh(
        family="legendre",
        regularization="x(1-x)",
        n=n,
        scale=a,
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


@register("legendre", "x^3/2")
def build_legendre_x_three_halves(
    *,
    n: int,
    scale: float,
    operators: set[str],
    **extras: object,
) -> tuple[Mesh, OperatorMatrices]:
    """Build shifted-Legendre-x^3/2 mesh data. [Baye eqs. 3.130, 3.136, 3.137]"""

    del extras
    a = float(scale)
    nodes, weights, radii = _shifted_legendre_quadrature(n, a)
    basis_at_boundary = _legendre_boundary_values(nodes, a)
    t_plus_l = _legendre_x_three_halves_t_plus_l(nodes, a)

    mesh = Mesh(
        family="legendre",
        regularization="x^3/2",
        n=n,
        scale=a,
        nodes=_to_jax_array(nodes),
        weights=_to_jax_array(weights),
        radii=_to_jax_array(radii),
        basis_at_boundary=_to_jax_array(basis_at_boundary),
    )

    include_kinetic = bool({"T+L", "TpL"} & operators)
    operators_out = OperatorMatrices(
        TpL=_to_jax_array(t_plus_l) if include_kinetic else None,
        inv_r=_diagonal_operator(1.0 / radii) if {"1/r", "inv_r"} & operators else None,
        inv_r2=(
            _diagonal_operator(1.0 / (radii**2))
            if {"1/r^2", "1/r²", "inv_r2"} & operators
            else None
        ),
    )
    return mesh, operators_out


def _legendre_x_t_plus_l(nodes: np.ndarray, scale: float) -> np.ndarray:
    """Return the Bloch-augmented kinetic matrix. [Desc. eqs. 22, 23]"""

    n = nodes.size
    matrix = np.zeros((n, n), dtype=np.float64)

    diagonal = ((4.0 * n * (n + 1) + 3.0) * nodes * (1.0 - nodes) - 6.0 * nodes + 1.0) / (
        3.0 * nodes**2 * (1.0 - nodes) ** 2
    )
    np.fill_diagonal(matrix, diagonal)

    row_idx: np.ndarray
    col_idx: np.ndarray
    row_idx, col_idx = np.triu_indices(n, k=1)
    off_diagonal = (
        n * (n + 1)
        + 1.0
        + (nodes[row_idx] + nodes[col_idx] - 2.0 * nodes[row_idx] * nodes[col_idx])
        / (nodes[row_idx] - nodes[col_idx]) ** 2
        - 1.0 / (1.0 - nodes[row_idx])
        - 1.0 / (1.0 - nodes[col_idx])
    )
    off_diagonal /= np.sqrt(
        nodes[row_idx] * (1.0 - nodes[row_idx]) * nodes[col_idx] * (1.0 - nodes[col_idx])
    )
    off_diagonal *= np.where((row_idx + col_idx) % 2 == 1, -1.0, 1.0)
    matrix[row_idx, col_idx] = off_diagonal
    matrix[col_idx, row_idx] = off_diagonal
    return matrix / (scale**2)


def _legendre_x_derivative(nodes: np.ndarray, scale: float) -> np.ndarray:
    """Return the exact first-derivative matrix. [Baye eqs. 3.123, 3.124]"""

    n = nodes.size
    derivative = np.zeros((n, n), dtype=np.float64)
    np.fill_diagonal(derivative, 1.0 / (2.0 * nodes * (1.0 - nodes)))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            sign = -1.0 if (i - j) % 2 else 1.0
            derivative[i, j] = (
                sign
                * np.sqrt(nodes[i] * (1.0 - nodes[j]) / (nodes[j] * (1.0 - nodes[i])))
                / (nodes[i] - nodes[j])
            )
    return derivative / scale


def _shifted_legendre_quadrature(n: int, scale: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return shifted-Legendre nodes, weights, and physical radii on `(0, a)`."""

    x_raw: np.ndarray
    w_raw: np.ndarray
    x_raw, w_raw = np.polynomial.legendre.leggauss(n)
    nodes = 0.5 * (x_raw + 1.0)
    weights = 0.5 * w_raw
    radii = scale * nodes
    return nodes, weights, radii


def _legendre_boundary_values(nodes: np.ndarray, scale: float) -> np.ndarray:
    """Return basis values at the channel boundary `r = a`."""

    n = nodes.size
    parity = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    boundary_sign = -parity if n % 2 == 1 else parity
    return boundary_sign / np.sqrt(scale * nodes * (1.0 - nodes))


def _legendre_x_one_minus_x_kinetic(nodes: np.ndarray, scale: float) -> np.ndarray:
    """Return the confined shifted-Legendre-x(1-x) kinetic matrix. [Baye eqs. 3.142-3.144]"""

    n = nodes.size
    matrix = np.zeros((n, n), dtype=np.float64)

    radial_factor = nodes * (1.0 - nodes)
    diagonal = (n * (n + 1.0) + 1.0 / radial_factor) / (3.0 * scale**2 * radial_factor)
    np.fill_diagonal(matrix, diagonal)

    row_idx: np.ndarray
    col_idx: np.ndarray
    row_idx, col_idx = np.triu_indices(n, k=1)
    sign = np.where((row_idx - col_idx) % 2 == 0, 1.0, -1.0)
    rij = np.sqrt(
        nodes[row_idx]
        * (1.0 - nodes[row_idx])
        * nodes[col_idx]
        * (1.0 - nodes[col_idx])
    )
    off_diagonal = (
        sign
        * (nodes[row_idx] + nodes[col_idx] - 2.0 * nodes[row_idx] * nodes[col_idx])
        / (scale**2 * rij * (nodes[row_idx] - nodes[col_idx]) ** 2)
    )
    matrix[row_idx, col_idx] = off_diagonal
    matrix[col_idx, row_idx] = off_diagonal

    all_row, all_col = np.indices((n, n))
    correction_sign = np.where((all_row - all_col) % 2 == 0, 1.0, -1.0)
    rij_full = np.sqrt(
        nodes[all_row]
        * (1.0 - nodes[all_row])
        * nodes[all_col]
        * (1.0 - nodes[all_col])
    )
    correction = correction_sign * n * (n + 1.0) / (scale**2 * (2.0 * n + 1.0) * rij_full)
    return matrix - correction


def _legendre_x_three_halves_t_plus_l(nodes: np.ndarray, scale: float) -> np.ndarray:
    """Return the shifted-Legendre-x^3/2 `T+L` matrix. [Baye eqs. 3.136, 3.137]"""

    n = nodes.size
    matrix = np.zeros((n, n), dtype=np.float64)

    diagonal_numerator = (
        4.0 * n * (n + 1.0) * (3.0 + nodes) * (1.0 - nodes)
        + 3.0 * nodes**2
        - 30.0 * nodes
        + 7.0
    )
    diagonal = diagonal_numerator / (12.0 * scale**2 * nodes**2 * (1.0 - nodes) ** 2)
    np.fill_diagonal(matrix, diagonal)

    row_idx: np.ndarray
    col_idx: np.ndarray
    row_idx, col_idx = np.triu_indices(n, k=1)
    sign = np.where((row_idx + col_idx) % 2 == 0, 1.0, -1.0)
    denominator = np.sqrt(
        nodes[row_idx]
        * nodes[col_idx]
        * (1.0 - nodes[row_idx])
        * (1.0 - nodes[col_idx])
    )
    numerator = (
        n * (n + 1.0)
        + 1.5
        + (
            nodes[row_idx] ** 2
            + nodes[col_idx] ** 2
            - nodes[row_idx] * nodes[col_idx] * (nodes[row_idx] + nodes[col_idx])
        )
        / (nodes[row_idx] - nodes[col_idx]) ** 2
        - 1.0 / (1.0 - nodes[row_idx])
        - 1.0 / (1.0 - nodes[col_idx])
    )
    off_diagonal = sign * numerator / (scale**2 * denominator)
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


__all__ = [
    "build_legendre_x",
    "build_legendre_x_one_minus_x",
    "build_legendre_x_three_halves",
]

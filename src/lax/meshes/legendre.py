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


def _to_jax_array(values: np.ndarray) -> jax.Array:
    """Convert a NumPy array to a runtime JAX array with an explicit type."""

    array: jax.Array = jnp.asarray(values)  # pyright: ignore[reportUnknownMemberType] -- JAX stubs expose asarray imprecisely for NumPy inputs.
    return array


def _diagonal_operator(values: np.ndarray) -> jax.Array:
    """Construct a diagonal JAX operator from compile-time diagonal values."""

    diagonal = _to_jax_array(values)
    matrix: jax.Array = jnp.diag(diagonal)
    return matrix


__all__ = ["build_legendre_x"]

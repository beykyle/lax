"""Compile-time helpers for R-matrix subinterval propagation."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from lax.types import PropagationMatrices


def build_legendre_x_propagation(
    *,
    basis_size_per_interval: int,
    n_intervals: int,
    scale: float,
) -> PropagationMatrices:
    """Return the precomputed matrices used by Descouvemont-style R-matrix propagation.

    Divides the internal region ``[0, a]`` into ``n_intervals`` equal
    subintervals of width ``a / n_intervals``.  For each subinterval the
    per-interval kinetic matrix and Bloch surface-overlap matrices are built
    using the shifted Legendre-x formulae from Descouvemont.  The resulting
    :class:`PropagationMatrices` object is stored inside the :class:`Mesh`
    and consumed by ``_propagated_rmatrix_at_energy`` at runtime.

    Parameters
    ----------
    basis_size_per_interval
        Number of Legendre basis functions per subinterval.
    n_intervals
        Number of subintervals to divide the internal region into.
    scale
        Total channel radius ``a`` in fm.

    Returns
    -------
    PropagationMatrices
        All precomputed kinetic and boundary-overlap matrices for the
        propagation recursion.
    """

    nr = basis_size_per_interval
    ns = n_intervals
    interval_width = float(scale) / float(ns)
    x_raw: np.ndarray
    w_raw: np.ndarray
    x_raw, w_raw = np.polynomial.legendre.leggauss(nr)
    local_nodes = 0.5 * (x_raw + 1.0)
    local_weights = 0.5 * w_raw

    kinetic = np.zeros((ns, nr, nr), dtype=np.float64)
    blo0 = np.zeros((nr, nr), dtype=np.float64)
    blo1 = np.zeros((nr, nr), dtype=np.float64)
    blo2 = np.zeros((nr, nr), dtype=np.float64)
    q1 = np.zeros((ns, nr), dtype=np.float64)
    q2 = np.zeros((ns, nr), dtype=np.float64)

    for interval_index in range(ns):
        for row in range(nr):
            xi = local_nodes[row]
            xi2 = xi * (1.0 - xi)
            if interval_index == 0:
                xx = 4.0 * nr * (nr + 1.0) + 3.0 + (1.0 - 6.0 * xi) / xi2
                kinetic[interval_index, row, row] = xx / (3.0 * xi2)
                blo0[row, row] = 1.0 / xi2
            else:
                xlb = xi / (1.0 - xi) * (nr * (nr + 1.0) - 1.0 / (1.0 - xi))
                xla = (1.0 - xi) / xi * (-nr * (nr + 1.0) + 1.0 / xi)
                kinetic[interval_index, row, row] = (
                    (nr * nr + nr + 6.0 - 2.0 / xi2) / (3.0 * xi2) + xlb - xla
                )
                blo1[row, row] = (1.0 - xi) / xi
                blo2[row, row] = xi / (1.0 - xi)

            for column in range(row):
                xj = local_nodes[column]
                xj2 = xj * (1.0 - xj)
                if interval_index == 0:
                    xx = (
                        nr * (nr + 1.0)
                        + 1.0
                        + (xi + xj - 2.0 * xi * xj) / (xi - xj) ** 2
                        - 1.0 / (1.0 - xi)
                        - 1.0 / (1.0 - xj)
                    )
                    value = xx / np.sqrt(xi2 * xj2)
                    blo0_value = 1.0 / np.sqrt(xi2 * xj2)
                    if (row + column) % 2 == 1:
                        value = -value
                        blo0_value = -blo0_value
                    kinetic[interval_index, row, column] = value
                    kinetic[interval_index, column, row] = value
                    blo0[row, column] = blo0_value
                    blo0[column, row] = blo0_value
                else:
                    yy = (
                        np.sqrt(xj2 / xi2**3)
                        * (2.0 * xi * xj + 3.0 * xi - xj - 4.0 * xi**2)
                        / (xj - xi) ** 2
                    )
                    xlb = np.sqrt(xi * xj / (1.0 - xi) / (1.0 - xj)) * (
                        nr * (nr + 1.0) - 1.0 / (1.0 - xj)
                    )
                    xla = np.sqrt((1.0 - xi) * (1.0 - xj) / xi / xj) * (-nr * (nr + 1.0) + 1.0 / xj)
                    value = yy + xlb - xla
                    blo1_value = np.sqrt((1.0 - xi) * (1.0 - xj) / xi / xj)
                    blo2_value = np.sqrt(xi * xj / (1.0 - xi) / (1.0 - xj))
                    if (row + column) % 2 == 1:
                        value = -value
                        blo0[row, column] = -blo0[row, column]
                        blo0[column, row] = -blo0[column, row]
                        blo1_value = -blo1_value
                        blo2_value = -blo2_value
                    kinetic[interval_index, row, column] = value
                    kinetic[interval_index, column, row] = value
                    blo1[row, column] = blo1_value
                    blo1[column, row] = blo1_value
                    blo2[row, column] = blo2_value
                    blo2[column, row] = blo2_value

        if interval_index == 0:
            q2[interval_index] = 1.0 / np.sqrt(local_nodes * (1.0 - local_nodes))
        else:
            q2[interval_index] = np.sqrt(local_nodes / (1.0 - local_nodes))
            q1[interval_index] = -1.0 / q2[interval_index]

        if nr % 2 == 1:
            q2[interval_index] = -q2[interval_index]
        q1[interval_index, ::2] = -q1[interval_index, ::2]
        q2[interval_index, ::2] = -q2[interval_index, ::2]

    kinetic /= interval_width**2
    blo0 /= interval_width
    blo1 /= interval_width
    blo2 /= interval_width
    q1 /= np.sqrt(interval_width)
    q2 /= np.sqrt(interval_width)

    return PropagationMatrices(
        n_intervals=ns,
        basis_size_per_interval=nr,
        interval_width=interval_width,
        local_nodes=_to_jax_array(local_nodes),
        local_weights=_to_jax_array(local_weights),
        kinetic=_to_jax_array(kinetic),
        blo0=_to_jax_array(blo0),
        blo1=_to_jax_array(blo1),
        blo2=_to_jax_array(blo2),
        q1=_to_jax_array(q1),
        q2=_to_jax_array(q2),
    )


def _to_jax_array(values: np.ndarray) -> jax.Array:
    """Convert a NumPy array to a runtime JAX array with an explicit type."""

    array: jax.Array = jnp.asarray(values)
    return array


__all__ = ["build_legendre_x_propagation"]

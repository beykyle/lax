"""Shared helpers for compile-time mesh construction."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def to_jax_array(values: np.ndarray) -> jax.Array:
    """Convert a NumPy array to a runtime JAX array with an explicit type."""

    array: jax.Array = jnp.asarray(values)
    return array


def diagonal_operator(values: np.ndarray) -> jax.Array:
    """Construct a diagonal JAX operator from compile-time diagonal values."""

    matrix: jax.Array = jnp.diag(to_jax_array(values))
    return matrix


__all__ = ["diagonal_operator", "to_jax_array"]

"""Integration helpers for norms and expectation values in the mesh basis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import jax
import jax.numpy as jnp

from lax.boundary._types import Integrator, Mesh


@dataclass(frozen=True)
class _IntegrationHelper:
    """Pickle-safe mesh-basis integration helper."""

    def __call__(self, values: jax.Array, operator: jax.Array | None = None) -> jax.Array:
        """Integrate mesh coefficients with an optional operator insertion."""

        if operator is None:
            return cast(jax.Array, _INTEGRATE_NORM_JIT(values))
        if operator.ndim == 1:
            return cast(jax.Array, _INTEGRATE_DIAGONAL_JIT(values, operator))
        return cast(jax.Array, _INTEGRATE_MATRIX_JIT(values, operator))


def make_integration(mesh: Mesh) -> Integrator:
    """Return a JIT-compiled mesh-basis integration helper. [DESIGN.md §13.3]"""

    del mesh
    return _IntegrationHelper()


def _integrate_norm(values: jax.Array) -> jax.Array:
    """Return the mesh-basis norm."""

    return jnp.sum(jnp.abs(values) ** 2)


def _integrate_diagonal(values: jax.Array, operator: jax.Array) -> jax.Array:
    """Return the expectation value for a diagonal operator."""

    return jnp.sum(jnp.abs(values) ** 2 * operator)


def _integrate_matrix(values: jax.Array, operator: jax.Array) -> jax.Array:
    """Return the expectation value for a full operator matrix."""

    result: jax.Array = values.conj() @ operator @ values
    return result


_INTEGRATE_NORM_JIT = jax.jit(  # pyright: ignore[reportUnknownMemberType] -- JAX jit wrappers are not precisely typed at module scope.
    _integrate_norm
)
_INTEGRATE_DIAGONAL_JIT = jax.jit(  # pyright: ignore[reportUnknownMemberType] -- JAX jit wrappers are not precisely typed at module scope.
    _integrate_diagonal
)
_INTEGRATE_MATRIX_JIT = jax.jit(  # pyright: ignore[reportUnknownMemberType] -- JAX jit wrappers are not precisely typed at module scope.
    _integrate_matrix
)


__all__ = ["make_integration"]

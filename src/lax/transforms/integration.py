"""Integration helpers for norms and expectation values in the mesh basis."""

from __future__ import annotations

from typing import cast

import jax
import jax.numpy as jnp

from lax.boundary._types import Integrator, Mesh


def make_integration(mesh: Mesh) -> Integrator:
    """Return a JIT-compiled mesh-basis integration helper. [DESIGN.md §13.3]"""

    del mesh

    def integrate(values: jax.Array, operator: jax.Array | None = None) -> jax.Array:
        if operator is None:
            return jnp.sum(jnp.abs(values) ** 2)
        if operator.ndim == 1:
            return jnp.sum(jnp.abs(values) ** 2 * operator)
        result: jax.Array = values.conj() @ operator @ values
        return result

    return cast(
        Integrator,
        jax.jit(integrate),  # pyright: ignore[reportUnknownMemberType] -- JAX jit wrappers are not precisely typed.
    )


__all__ = ["make_integration"]

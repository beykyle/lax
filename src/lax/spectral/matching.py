"""Observable matching from R-matrix to S-matrix and phases."""

from __future__ import annotations

from typing import cast

import jax
import jax.numpy as jnp

from lax.boundary._types import BoundaryValues


def smatrix_from_R(R: jax.Array, boundary_at_energy: BoundaryValues) -> jax.Array:
    """Return the S-matrix at one energy from the boundary matching formula."""

    H_plus = jnp.diag(boundary_at_energy.H_plus)
    H_minus = jnp.diag(boundary_at_energy.H_minus)
    H_plus_p = jnp.diag(boundary_at_energy.H_plus_p)
    H_minus_p = jnp.diag(boundary_at_energy.H_minus_p)

    numerator = H_minus - R @ H_minus_p
    denominator = H_plus - R @ H_plus_p
    matrix = cast(
        jax.Array,
        jnp.linalg.solve(  # pyright: ignore[reportUnknownMemberType] -- JAX linalg solve stubs lose the transpose result type here.
            denominator.T,
            numerator.T,
        ).T,
    )
    return matrix


def phases_from_S(S: jax.Array) -> jax.Array:
    """Return channel phase shifts from the S-matrix eigenphases."""

    eigenvalues = cast(
        jax.Array,
        jnp.linalg.eigvals(  # pyright: ignore[reportUnknownMemberType] -- JAX eigvals stubs do not preserve a concrete return type.
            S
        ),
    )
    phases: jax.Array = 0.5 * jnp.angle(eigenvalues)
    return phases


__all__ = ["phases_from_S", "smatrix_from_R"]

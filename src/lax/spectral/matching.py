"""Observable matching from R-matrix to S-matrix and phases."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import jax
import jax.numpy as jnp

from lax.boundary._types import BoundaryValues


@dataclass(frozen=True)
class CoupledChannelParameters:
    """Blatt-Biedenharn-style eigenphases and mixing angle for a 2x2 S-matrix."""

    phase_1: jax.Array
    phase_2: jax.Array
    mixing_angle: jax.Array


def smatrix_from_R(R: jax.Array, boundary_at_energy: BoundaryValues) -> jax.Array:
    """Return the S-matrix at one energy from the boundary matching formula."""

    H_plus = jnp.diag(boundary_at_energy.H_plus)
    H_minus = jnp.diag(boundary_at_energy.H_minus)
    H_plus_p = jnp.diag(boundary_at_energy.H_plus_p)
    H_minus_p = jnp.diag(boundary_at_energy.H_minus_p)
    if boundary_at_energy.k is None:
        k = jnp.ones(R.shape[0], dtype=R.dtype)  # pyright: ignore[reportUnknownMemberType] -- JAX ones stubs are imprecise.
    else:
        k = boundary_at_energy.k.astype(R.dtype)
    sqrt_k = jnp.sqrt(k)
    K = jnp.diag(sqrt_k)
    Kinv = jnp.diag(1.0 / sqrt_k)
    normalized_R = K @ R @ Kinv

    numerator = H_minus - normalized_R @ H_minus_p
    denominator = H_plus - normalized_R @ H_plus_p
    matrix = cast(
        jax.Array,
        jnp.linalg.solve(  # pyright: ignore[reportUnknownMemberType] -- JAX linalg solve stubs lose the result type here.
            denominator,
            numerator,
        ),
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


def coupled_channel_parameters_from_S(S: jax.Array) -> CoupledChannelParameters:
    """Return ordered eigenphases and one mixing angle from a symmetric 2x2 S-matrix."""

    if S.shape != (2, 2):
        msg = f"Expected a 2x2 coupled-channel S-matrix, got shape {S.shape!r}."
        raise ValueError(msg)

    eigenvalues = cast(
        jax.Array,
        jnp.linalg.eigvals(  # pyright: ignore[reportUnknownMemberType] -- JAX eigvals stubs do not preserve a concrete return type.
            S
        ),
    )
    order = jnp.argsort(jnp.angle(eigenvalues))  # pyright: ignore[reportUnknownMemberType] -- JAX argsort stubs are imprecise.
    eigenvalues_ordered = eigenvalues[order]
    phase_1 = 0.5 * jnp.angle(eigenvalues_ordered[0])
    phase_2 = 0.5 * jnp.angle(eigenvalues_ordered[1])

    lambda_1 = eigenvalues_ordered[0]
    lambda_2 = eigenvalues_ordered[1]
    eigenvalue_gap = lambda_1 - lambda_2
    gap_magnitude = jnp.abs(eigenvalue_gap)
    safe_gap = jnp.where(  # pyright: ignore[reportUnknownMemberType] -- JAX where stubs are imprecise.
        gap_magnitude < 1.0e-12,
        jnp.asarray(1.0 + 0.0j, dtype=eigenvalue_gap.dtype),  # pyright: ignore[reportUnknownMemberType] -- JAX asarray stubs are imprecise for scalar literals.
        eigenvalue_gap,
    )

    cos2 = jnp.real((S[0, 0] - lambda_2) / safe_gap)
    sin2 = jnp.real((lambda_1 - S[0, 0]) / safe_gap)
    cos2_clamped = jnp.clip(cos2, 0.0, 1.0)  # pyright: ignore[reportUnknownMemberType] -- JAX clip stubs are imprecise.
    sin2_clamped = jnp.clip(sin2, 0.0, 1.0)  # pyright: ignore[reportUnknownMemberType] -- JAX clip stubs are imprecise.
    mixing_sign = jnp.sign(jnp.real(S[0, 1] / safe_gap))
    mixing_abs = jnp.arctan2(jnp.sqrt(sin2_clamped), jnp.sqrt(cos2_clamped))
    mixing_angle = jnp.where(  # pyright: ignore[reportUnknownMemberType] -- JAX where stubs are imprecise.
        gap_magnitude < 1.0e-12,
        0.0,
        mixing_sign * mixing_abs,
    )

    return CoupledChannelParameters(
        phase_1=phase_1,
        phase_2=phase_2,
        mixing_angle=mixing_angle,
    )


__all__ = [
    "CoupledChannelParameters",
    "coupled_channel_parameters_from_S",
    "phases_from_S",
    "smatrix_from_R",
]

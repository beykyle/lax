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


def open_channel_smatrix_from_R(R: jax.Array, boundary_at_energy: BoundaryValues) -> jax.Array:
    """Return the physical open-channel S-matrix at one energy.

    Parameters
    ----------
    R
        Full channel-space R-matrix at one energy.
    boundary_at_energy
        Boundary values for the same energy. Closed channels may be included.

    Returns
    -------
    jax.Array
        Physical S-matrix restricted to the open-channel subspace.
    """

    decoupled_r = _decouple_closed_channels(
        R,
        boundary_at_energy.H_plus,
        boundary_at_energy.H_plus_p,
        boundary_at_energy.is_open,
    )
    projected_r, projected_boundary = _project_open_channels(
        decoupled_r,
        boundary_at_energy.H_plus,
        boundary_at_energy.H_minus,
        boundary_at_energy.H_plus_p,
        boundary_at_energy.H_minus_p,
        boundary_at_energy.is_open,
        boundary_at_energy.k,
    )
    return smatrix_from_R(projected_r, projected_boundary)


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


def _decouple_closed_channels(
    rmatrix: jax.Array,
    h_plus: jax.Array,
    h_plus_p: jax.Array,
    is_open: jax.Array,
) -> jax.Array:
    """Fold closed-channel boundary conditions into an effective R-matrix."""

    bloch = _closed_channel_bloch(h_plus, h_plus_p, is_open)
    identity: jax.Array = jnp.eye(  # pyright: ignore[reportUnknownMemberType] -- JAX eye stubs are imprecise.
        rmatrix.shape[0],
        dtype=rmatrix.dtype,
    )
    correction = identity - rmatrix @ jnp.diag(bloch)
    return cast(
        jax.Array,
        jnp.linalg.solve(  # pyright: ignore[reportUnknownMemberType] -- JAX linalg solve stubs lose the result type here.
            correction.T,
            rmatrix.T,
        ).T,
    )


def _project_open_channels(
    rmatrix: jax.Array,
    h_plus: jax.Array,
    h_minus: jax.Array,
    h_plus_p: jax.Array,
    h_minus_p: jax.Array,
    is_open: jax.Array,
    k: jax.Array | None,
) -> tuple[jax.Array, BoundaryValues]:
    """Project a full-channel system onto the open-channel matching problem."""

    mask = is_open.astype(rmatrix.dtype)
    projected_r = rmatrix * mask[:, None] * mask[None, :]
    closed_dtype = h_plus.dtype
    ones: jax.Array = jnp.ones_like(  # pyright: ignore[reportUnknownMemberType] -- JAX ones_like stubs are imprecise.
        h_plus,
        dtype=closed_dtype,
    )
    mask_complex = is_open.astype(closed_dtype)
    if k is None:
        k_values = None
    else:
        ones_k: jax.Array = jnp.ones_like(k, dtype=k.dtype)  # pyright: ignore[reportUnknownMemberType] -- JAX ones_like stubs are imprecise.
        k_values = k * is_open.astype(k.dtype) + ones_k * (1 - is_open.astype(k.dtype))

    boundary_slice = BoundaryValues(
        H_plus=h_plus * mask_complex + ones * (1.0 - mask_complex),
        H_minus=h_minus * mask_complex,
        H_plus_p=h_plus_p * mask_complex,
        H_minus_p=h_minus_p * mask_complex,
        is_open=is_open,
        k=k_values,
    )
    return projected_r, boundary_slice


def _closed_channel_bloch(
    h_plus: jax.Array,
    h_plus_p: jax.Array,
    is_open: jax.Array,
) -> jax.Array:
    """Return the closed-channel Bloch boundary ratio ``B_c = H'_c / H_c``."""

    ratio = h_plus_p / h_plus
    closed_mask = jnp.logical_not(is_open)  # pyright: ignore[reportUnknownMemberType] -- JAX logical_not stubs are imprecise.
    zeros: jax.Array = jnp.zeros_like(  # pyright: ignore[reportUnknownMemberType] -- JAX zeros_like stubs are imprecise.
        ratio
    )
    return jnp.where(  # pyright: ignore[reportUnknownMemberType] -- JAX where stubs are imprecise.
        closed_mask,
        ratio,
        zeros,
    )


__all__ = [
    "CoupledChannelParameters",
    "coupled_channel_parameters_from_S",
    "open_channel_smatrix_from_R",
    "phases_from_S",
    "smatrix_from_R",
]

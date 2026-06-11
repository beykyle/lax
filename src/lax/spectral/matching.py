"""Observable matching from R-matrix to S-matrix and phases."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import jax
import jax.numpy as jnp

from lax.boundary._types import BoundaryValues


@dataclass(frozen=True)
class CoupledChannelParameters:
    """Blatt-Biedenharn eigenphases and mixing angle for a 2×2 S-matrix.

    Stores the bar-phase (ε-bar) parameterisation used by Blatt and
    Biedenharn (1952): the physical 2×2 S-matrix is diagonalised, giving two
    eigenphases ``phase_1``/``phase_2`` and one mixing angle ``mixing_angle``
    (the bar-epsilon parameter ε̄).

    Attributes
    ----------
    phase_1
        First eigenphase in radians (eigenvalue with smaller argument).
    phase_2
        Second eigenphase in radians (eigenvalue with larger argument).
    mixing_angle
        Bar-epsilon mixing angle ε̄ in radians.  Zero for an uncoupled channel.
    """

    phase_1: jax.Array
    phase_2: jax.Array
    mixing_angle: jax.Array


def smatrix_from_R(R: jax.Array, boundary_at_energy: BoundaryValues) -> jax.Array:
    """Return the full channel-space S-matrix at one energy.

    Applies the standard R-matrix matching formula [Descouvemont eqs. 16-17],
    normalised by channel wave numbers:

    .. code-block:: text

        S = (H⁻ - R̃ H⁻') / (H⁺ - R̃ H⁺')   where  R̃ = K R K⁻¹

    Parameters
    ----------
    R
        R-matrix at one energy, shape ``(N_c, N_c)``.
    boundary_at_energy
        Boundary values at the same energy.  For a multi-energy solver,
        pass a single energy's slice (shape ``(N_c,)`` per array field).

    Returns
    -------
    jax.Array
        S-matrix, shape ``(N_c, N_c)``, complex.
    """

    H_plus = jnp.diag(boundary_at_energy.H_plus)
    H_minus = jnp.diag(boundary_at_energy.H_minus)
    H_plus_p = jnp.diag(boundary_at_energy.H_plus_p)
    H_minus_p = jnp.diag(boundary_at_energy.H_minus_p)
    k = boundary_at_energy.k.astype(R.dtype)
    sqrt_k = jnp.sqrt(k)
    K = jnp.diag(sqrt_k)
    Kinv = jnp.diag(1.0 / sqrt_k)
    normalized_R = K @ R @ Kinv

    numerator = H_minus - normalized_R @ H_minus_p
    denominator = H_plus - normalized_R @ H_plus_p
    matrix = cast(
        jax.Array,
        jnp.linalg.solve(
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
    """Return channel phase shifts from the eigenphases of S.

    Computes ``δ_k = ½ arg(λ_k)`` where ``λ_k`` are the eigenvalues of S.
    For a diagonal (uncoupled) S-matrix these coincide with the standard
    single-channel phase shifts.

    Parameters
    ----------
    S
        S-matrix at one energy, shape ``(N_c, N_c)``.

    Returns
    -------
    jax.Array
        Phase shifts in radians, shape ``(N_c,)``.  For a single-channel
        solver index with ``[0]`` to get a scalar.
    """

    eigenvalues = cast(
        jax.Array,
        jnp.linalg.eigvals(S),
    )
    phases: jax.Array = 0.5 * jnp.angle(eigenvalues)
    return phases


def coupled_channel_parameters_from_S(S: jax.Array) -> CoupledChannelParameters:
    """Return Blatt-Biedenharn eigenphases and mixing angle from a 2×2 S-matrix.

    Diagonalises the symmetric 2×2 collision matrix and extracts the
    bar-phase parameterisation [Blatt & Biedenharn 1952].

    Parameters
    ----------
    S
        Symmetric 2×2 S-matrix at one energy.

    Returns
    -------
    CoupledChannelParameters
        Eigenphases and mixing angle.

    Raises
    ------
    ValueError
        If ``S.shape != (2, 2)``.
    """

    if S.shape != (2, 2):
        msg = f"Expected a 2x2 coupled-channel S-matrix, got shape {S.shape!r}."
        raise ValueError(msg)

    eigenvalues = cast(
        jax.Array,
        jnp.linalg.eigvals(S),
    )
    order = jnp.argsort(jnp.angle(eigenvalues))
    eigenvalues_ordered = eigenvalues[order]
    phase_1 = jnp.real(0.5 * jnp.angle(eigenvalues_ordered[0]))
    phase_2 = jnp.real(0.5 * jnp.angle(eigenvalues_ordered[1]))

    lambda_1 = eigenvalues_ordered[0]
    lambda_2 = eigenvalues_ordered[1]
    eigenvalue_gap = lambda_1 - lambda_2
    gap_magnitude = jnp.abs(eigenvalue_gap)
    safe_gap = jnp.where(
        gap_magnitude < 1.0e-12,
        jnp.asarray(1.0 + 0.0j, dtype=eigenvalue_gap.dtype),
        eigenvalue_gap,
    )

    cos2 = jnp.real((S[0, 0] - lambda_2) / safe_gap)
    sin2 = jnp.real((lambda_1 - S[0, 0]) / safe_gap)
    cos2_clamped = jnp.clip(cos2, 0.0, 1.0)
    sin2_clamped = jnp.clip(sin2, 0.0, 1.0)
    mixing_sign = jnp.sign(jnp.real(S[0, 1] / safe_gap))
    mixing_abs = jnp.arctan2(jnp.sqrt(sin2_clamped), jnp.sqrt(cos2_clamped))
    mixing_angle = jnp.where(
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
    identity: jax.Array = jnp.eye(
        rmatrix.shape[0],
        dtype=rmatrix.dtype,
    )
    correction = identity - rmatrix @ jnp.diag(bloch)
    return cast(
        jax.Array,
        jnp.linalg.solve(
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
    k: jax.Array,
) -> tuple[jax.Array, BoundaryValues]:
    """Project a full-channel system onto the open-channel matching problem."""

    mask = is_open.astype(rmatrix.dtype)
    projected_r = rmatrix * mask[:, None] * mask[None, :]
    closed_dtype = h_plus.dtype
    ones: jax.Array = jnp.ones_like(
        h_plus,
        dtype=closed_dtype,
    )
    mask_complex = is_open.astype(closed_dtype)
    ones_k: jax.Array = jnp.ones_like(k, dtype=k.dtype)
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
    closed_mask = jnp.logical_not(is_open)
    zeros: jax.Array = jnp.zeros_like(ratio)
    return jnp.where(
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

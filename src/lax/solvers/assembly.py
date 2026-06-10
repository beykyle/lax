"""Hamiltonian assembly helpers for compiled solver kernels."""

from __future__ import annotations

from typing import cast

import jax
import jax.numpy as jnp

from lax.boundary._types import Mesh, OperatorMatrices
from lax.types import ChannelSpec


def assemble_block_hamiltonian(
    mesh: Mesh,
    operators: OperatorMatrices,
    channels: tuple[ChannelSpec, ...],
    potential: jax.Array,
    mass_factor_override: float | jax.Array | None = None,
) -> jax.Array:
    """Assemble the Bloch-augmented block Hamiltonian in MeV units.

    Builds the ``(N_c·N, N_c·N)`` matrix whose eigendecomposition drives
    all spectral observables.  Diagonal blocks contain
    ``m_c·(T+L) + E_c·I``; off-diagonal blocks contain the coupling
    potential.  [DESIGN.md §11.5]

    Each channel's kinetic block is scaled by ``m_c`` (ℏ²/2μ in MeV·fm²)
    so that the assembled matrix is in MeV throughout.  The potential ``V``
    is added without rescaling — it must already be in MeV.

    Parameters
    ----------
    mesh
        Compiled mesh supplying ``TpL`` and ``inv_r2``.
    operators
        Precomputed operator matrices; ``TpL`` must be present.
    channels
        Channel definitions (``l``, ``threshold``, ``mass_factor``).
    potential
        Assembled potential in MeV.  Shape ``(N_c, N_c, N)`` for local
        or ``(N_c, N_c, N, N)`` for non-local.
    mass_factor_override
        When not ``None``, overrides ``channel.mass_factor`` for the
        kinetic scaling.  Accepts either a scalar (uniform override for
        all channels) or a JAX array of shape ``(N_c,)`` for per-channel
        values.  Supply a traced scalar when using an energy-dependent
        μ(E) so it vmaps correctly.

    Returns
    -------
    jax.Array
        Block Hamiltonian, shape ``(N_c·N, N_c·N)``, in MeV.
    """

    channel_count = len(channels)
    basis_size = mesh.n
    t_plus_l = _require_operator(operators.TpL, "TpL")
    inv_r2 = operators.inv_r2
    if inv_r2 is None:
        inv_r2 = _diagonal_from_vector(1.0 / (mesh.radii**2))

    blocks: list[jax.Array] = []
    for channel_index in range(channel_count):
        row_blocks: list[jax.Array] = []
        if mass_factor_override is not None:
            if jnp.ndim(mass_factor_override) == 0:
                m_c = mass_factor_override
            else:
                m_c = cast(jax.Array, mass_factor_override)[channel_index]
        else:
            m_c = channels[channel_index].mass_factor
        angular_momentum = channels[channel_index].l
        threshold = channels[channel_index].threshold
        for coupled_index in range(channel_count):
            block: jax.Array = jnp.zeros(
                (basis_size, basis_size),
                dtype=potential.dtype,
            )
            if channel_index == coupled_index:
                block = (
                    block
                    + m_c * (t_plus_l + angular_momentum * (angular_momentum + 1) * inv_r2)
                    + threshold
                    * jnp.eye(
                        basis_size,
                        dtype=potential.dtype,
                    )
                )
            if potential.ndim == 3:
                block = block + _diagonal_from_vector(potential[channel_index, coupled_index])
            elif potential.ndim == 2:
                # Pre-assembled (M, M) block (e.g. from Interaction.block): extract sub-block
                # by slicing rather than channel indexing.
                rs = channel_index * basis_size
                cs = coupled_index * basis_size
                block = block + potential[rs : rs + basis_size, cs : cs + basis_size]
            else:
                block = block + potential[channel_index, coupled_index]
            row_blocks.append(block)
        blocks.append(jnp.concatenate(row_blocks, axis=1))
    matrix: jax.Array = jnp.concatenate(blocks, axis=0)
    return matrix


def build_Q(mesh: Mesh, channels: tuple[ChannelSpec, ...]) -> jax.Array:
    """Return the surface projector matrix Q.

    ``Q[c·N + j, c'] = δ_{cc'} φ_j(a)`` — a block-diagonal matrix that
    picks out the boundary-surface component of each channel's eigenvectors.
    The surface amplitudes are then ``γ = U^T Q``.  [DESIGN.md §11.5]

    Parameters
    ----------
    mesh
        Compiled mesh supplying ``basis_at_boundary`` (shape ``(N,)``).
    channels
        Channel definitions; only the count is used.

    Returns
    -------
    jax.Array
        Surface projector, shape ``(N_c·N, N_c)``.
    """

    channel_count = len(channels)
    basis_size = mesh.n
    boundary = mesh.basis_at_boundary

    q: jax.Array = jnp.zeros(
        (channel_count * basis_size, channel_count),
        dtype=boundary.dtype,
    )
    for channel_index in range(channel_count):
        q = q.at[channel_index * basis_size : (channel_index + 1) * basis_size, channel_index].set(
            boundary
        )
    return q


def _diagonal_from_vector(values: jax.Array) -> jax.Array:
    """Construct a diagonal matrix from one vector."""

    matrix: jax.Array = jnp.diag(values)
    return matrix


def _require_operator(operator: jax.Array | None, name: str) -> jax.Array:
    """Require a precomputed operator matrix."""

    if operator is None:
        msg = f"OperatorMatrices.{name} is required for solver assembly."
        raise ValueError(msg)
    return operator


__all__ = ["assemble_block_hamiltonian", "build_Q"]

"""Hamiltonian assembly helpers for compiled solver kernels."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from lax.boundary._types import Mesh, OperatorMatrices
from lax.types import ChannelSpec


def assemble_block_hamiltonian(
    mesh: Mesh,
    operators: OperatorMatrices,
    channels: tuple[ChannelSpec, ...],
    potential: jax.Array,
) -> jax.Array:
    """Assemble the Bloch-augmented Hamiltonian in fm^-2 units. [DESIGN.md §11.5]"""

    channel_count = len(channels)
    basis_size = mesh.n
    t_plus_l = _require_operator(operators.TpL, "TpL")
    inv_r2 = operators.inv_r2
    if inv_r2 is None:
        inv_r2 = _diagonal_from_vector(1.0 / (mesh.radii**2))

    blocks: list[jax.Array] = []
    for channel_index in range(channel_count):
        row_blocks: list[jax.Array] = []
        mass_factor = channels[channel_index].mass_factor
        angular_momentum = channels[channel_index].l
        threshold = channels[channel_index].threshold / mass_factor
        for coupled_index in range(channel_count):
            block: jax.Array = jnp.zeros(  # pyright: ignore[reportUnknownMemberType] -- JAX array constructors have imprecise stubs.
                (basis_size, basis_size),
                dtype=potential.dtype,
            )
            if channel_index == coupled_index:
                block = (
                    block
                    + t_plus_l
                    + angular_momentum * (angular_momentum + 1) * inv_r2
                    + threshold
                    * jnp.eye(  # pyright: ignore[reportUnknownMemberType] -- JAX array constructors have imprecise stubs.
                        basis_size,
                        dtype=potential.dtype,
                    )
                )
            if potential.ndim == 3:
                block = (
                    block
                    + _diagonal_from_vector(potential[channel_index, coupled_index]) / mass_factor
                )
            else:
                block = block + potential[channel_index, coupled_index] / mass_factor
            row_blocks.append(block)
        blocks.append(jnp.concatenate(row_blocks, axis=1))
    matrix: jax.Array = jnp.concatenate(blocks, axis=0)
    return matrix


def build_Q(mesh: Mesh, channels: tuple[ChannelSpec, ...]) -> jax.Array:
    """Return the surface projector `Q[c·N+j, c'] = δ_cc' φ_j(a)`. [DESIGN.md §11.5]"""

    channel_count = len(channels)
    basis_size = mesh.n
    boundary = mesh.basis_at_boundary

    q: jax.Array = jnp.zeros(  # pyright: ignore[reportUnknownMemberType] -- JAX array constructors have imprecise stubs.
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

    matrix: jax.Array = jnp.diag(values)  # pyright: ignore[reportUnknownMemberType] -- JAX stubs for diag are imprecise.
    return matrix


def _require_operator(operator: jax.Array | None, name: str) -> jax.Array:
    """Require a precomputed operator matrix."""

    if operator is None:
        msg = f"OperatorMatrices.{name} is required for solver assembly."
        raise ValueError(msg)
    return operator


__all__ = ["assemble_block_hamiltonian", "build_Q"]

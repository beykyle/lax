"""Potential assembly helpers."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp

from lax.boundary._types import Mesh


def assemble_local(
    mesh: Mesh,
    potential_fn: Callable[..., jax.Array],
    n_channels: int = 1,
) -> jax.Array:
    """Assemble a local potential sampled on the mesh radii."""

    radii = mesh.radii
    if n_channels == 1:
        values = potential_fn(radii)
        return values[None, None, :]

    blocks: list[jax.Array] = []
    for channel_index in range(n_channels):
        row: list[jax.Array] = []
        for coupled_index in range(n_channels):
            row.append(potential_fn(radii, channel_index, coupled_index))
        blocks.append(jnp.stack(row))
    matrix: jax.Array = jnp.stack(blocks)
    return matrix


def assemble_nonlocal(
    mesh: Mesh,
    kernel_fn: Callable[..., jax.Array],
    n_channels: int = 1,
) -> jax.Array:
    """Assemble a Gauss-scaled non-local potential on the mesh."""

    radii = mesh.radii
    weights = mesh.weights
    radius_i, radius_j = jnp.meshgrid(radii, radii, indexing="ij")  # pyright: ignore[reportUnknownMemberType] -- JAX stubs for meshgrid are imprecise.
    weight_i, weight_j = jnp.meshgrid(weights, weights, indexing="ij")  # pyright: ignore[reportUnknownMemberType] -- JAX stubs for meshgrid are imprecise.
    scale = jnp.sqrt(weight_i * weight_j) * mesh.scale

    if n_channels == 1:
        kernel = kernel_fn(radius_i, radius_j)
        return (kernel * scale)[None, None, :, :]

    blocks: list[jax.Array] = []
    for channel_index in range(n_channels):
        row: list[jax.Array] = []
        for coupled_index in range(n_channels):
            row.append(kernel_fn(radius_i, radius_j, channel_index, coupled_index) * scale)
        blocks.append(jnp.stack(row))
    matrix: jax.Array = jnp.stack(blocks)
    return matrix


__all__ = ["assemble_local", "assemble_nonlocal"]

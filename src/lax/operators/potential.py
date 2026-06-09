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
    """Assemble a local potential sampled on the mesh radii.

    Parameters
    ----------
    mesh
        Compiled mesh (``solver.mesh``).
    potential_fn
        For single-channel: ``potential_fn(radii) -> (N,)`` array of potential
        values in MeV.  For coupled-channel: ``potential_fn(radii, c, c') ->
        (N,)`` array for the ``(c, c')`` block.
    n_channels
        Number of coupled channels ``N_c``.

    Returns
    -------
    jax.Array
        Shape ``(N_c, N_c, N)`` where ``N = mesh.n``.  Pass directly to
        ``solver.spectrum(V)`` or ``solver.rmatrix_direct(V)``.
    """

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
    """Assemble a Gauss-scaled non-local potential on the mesh.

    Applies the Gauss-quadrature weight scaling ``(λ_i λ_j)^{1/2} · a``
    required by the Lagrange-mesh non-local matrix element formula
    [Descouvemont eq. 26].

    Parameters
    ----------
    mesh
        Compiled mesh (``solver.mesh``).
    kernel_fn
        For single-channel: ``kernel_fn(r_i, r_j) -> (N, N)`` kernel values in
        MeV.  For coupled-channel: ``kernel_fn(r_i, r_j, c, c') -> (N, N)``
        for the ``(c, c')`` block.
    n_channels
        Number of coupled channels ``N_c``.

    Returns
    -------
    jax.Array
        Shape ``(N_c, N_c, N, N)`` where ``N = mesh.n``.  Pass directly to
        ``solver.spectrum(V)`` or ``solver.rmatrix_direct(V)``.
    """

    if mesh.propagation is not None:
        msg = (
            "Subinterval propagation is defined only for local potentials in the direct "
            "linear-solve formulation. Non-local kernels are not mathematically "
            "supported on propagated meshes."
        )
        raise ValueError(msg)

    radii = mesh.radii
    weights = mesh.weights
    radius_i, radius_j = jnp.meshgrid(radii, radii, indexing="ij")
    weight_i, weight_j = jnp.meshgrid(weights, weights, indexing="ij")
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

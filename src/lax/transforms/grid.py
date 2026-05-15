"""Radial-grid transforms built from precomputed basis-evaluation matrices."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np

from lax.boundary._types import (
    FromGridVectorTransform,
    GridMatrixTransform,
    GridVectorTransform,
    Mesh,
)
from lax.meshes._basis_eval import basis_at


def compute_B_grid(mesh: Mesh, radii: jax.Array) -> jax.Array:
    """Return `B[k, j] = f_j(r_k)` for one physical radial grid. [DESIGN.md §13.1]"""

    return basis_at(mesh, radii)


def make_to_grid(
    mesh: Mesh,
    basis_grid: jax.Array,
    radii: jax.Array,
) -> tuple[GridVectorTransform, FromGridVectorTransform, GridMatrixTransform]:
    """Return JIT-compiled grid projection helpers in both directions."""

    projection_matrix = _compute_from_grid_projection(mesh, basis_grid, radii)
    grid_r = radii

    def to_grid_vector(values: jax.Array) -> jax.Array:
        """Project mesh coefficients `(N,)` onto the radial grid `(M_r,)`."""

        result: jax.Array = basis_grid @ values
        return result

    projection_bound = cast(
        GridVectorTransform,
        jax.jit(to_grid_vector),  # pyright: ignore[reportUnknownMemberType] -- JAX jit wrappers are not precisely typed.
    )

    def from_grid_array(values: jax.Array) -> jax.Array:
        """Project sampled radial-grid values `(M_r,)` back to mesh coefficients `(N,)`."""

        result: jax.Array = projection_matrix @ values
        return result

    def to_grid_matrix(values: jax.Array) -> jax.Array:
        """Project a mesh-space kernel `(N, N)` onto the radial grid `(M_r, M_r)`."""

        result: jax.Array = basis_grid @ values @ basis_grid.T
        return result

    from_grid_array_bound = cast(
        FromGridVectorTransform,
        jax.jit(from_grid_array),  # pyright: ignore[reportUnknownMemberType] -- JAX jit wrappers are not precisely typed.
    )

    def from_grid_vector(values: jax.Array | Callable[[jax.Array], jax.Array]) -> jax.Array:
        """Project sampled radial-grid values or a callable profile onto the mesh basis."""

        sampled = values(grid_r) if callable(values) else values
        return from_grid_array_bound(sampled)

    return (
        projection_bound,
        cast(FromGridVectorTransform, from_grid_vector),
        cast(
            GridMatrixTransform,
            jax.jit(to_grid_matrix),  # pyright: ignore[reportUnknownMemberType] -- JAX jit wrappers are not precisely typed.
        ),
    )


def _compute_from_grid_projection(mesh: Mesh, basis_grid: jax.Array, radii: jax.Array) -> jax.Array:
    """Return the compile-time projection matrix from grid samples to mesh coefficients."""

    basis_grid_np = np.asarray(basis_grid, dtype=np.float64)
    radii_np = np.asarray(radii, dtype=np.float64)
    mesh_radii = np.asarray(mesh.radii, dtype=np.float64)

    if _is_nodal_lagrange_grid(mesh_radii, radii_np):
        diagonal = np.diag(basis_grid_np)
        projection = np.diag(1.0 / diagonal)
    elif basis_grid_np.shape[0] == basis_grid_np.shape[1]:
        projection = np.linalg.solve(
            basis_grid_np, np.eye(basis_grid_np.shape[0], dtype=np.float64)
        )
    else:
        projection = np.linalg.pinv(basis_grid_np)

    projection_array: jax.Array = jnp.asarray(projection)  # pyright: ignore[reportUnknownMemberType] -- JAX stubs expose asarray imprecisely for NumPy inputs.
    return projection_array


def _is_nodal_lagrange_grid(mesh_radii: np.ndarray, radii: np.ndarray) -> bool:
    """Return whether the configured grid matches the mesh nodal grid."""

    return mesh_radii.shape == radii.shape and np.allclose(mesh_radii, radii)


__all__ = ["compute_B_grid", "make_to_grid"]

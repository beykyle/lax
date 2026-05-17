"""Radial-grid transforms built from precomputed basis-evaluation matrices."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
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


@dataclass(frozen=True)
class _GridVectorProjection:
    """Pickle-safe mesh-to-grid vector projection."""

    basis_grid: jax.Array

    def __call__(self, values: jax.Array) -> jax.Array:
        """Project mesh coefficients onto the configured radial grid."""

        return cast(jax.Array, _TO_GRID_VECTOR_JIT(values, self.basis_grid))


@dataclass(frozen=True)
class _FromGridVectorProjection:
    """Pickle-safe grid-to-mesh vector projection."""

    projection_matrix: jax.Array
    grid_r: jax.Array

    def __call__(self, values: jax.Array | Callable[[jax.Array], jax.Array]) -> jax.Array:
        """Project samples or a callable profile from the grid onto the mesh basis."""

        sampled = values(self.grid_r) if callable(values) else values
        return cast(jax.Array, _FROM_GRID_ARRAY_JIT(sampled, self.projection_matrix))


@dataclass(frozen=True)
class _GridMatrixProjection:
    """Pickle-safe mesh-to-grid kernel projection."""

    basis_grid: jax.Array

    def __call__(self, values: jax.Array) -> jax.Array:
        """Project a mesh-space kernel onto the configured radial grid."""

        return cast(jax.Array, _TO_GRID_MATRIX_JIT(values, self.basis_grid))


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
    return (
        _GridVectorProjection(basis_grid=basis_grid),
        _FromGridVectorProjection(projection_matrix=projection_matrix, grid_r=radii),
        _GridMatrixProjection(basis_grid=basis_grid),
    )


def _to_grid_vector(values: jax.Array, basis_grid: jax.Array) -> jax.Array:
    """Project mesh coefficients `(N,)` onto the radial grid `(M_r,)`."""

    result: jax.Array = basis_grid @ values
    return result


def _from_grid_array(values: jax.Array, projection_matrix: jax.Array) -> jax.Array:
    """Project sampled radial-grid values `(M_r,)` back to mesh coefficients `(N,)`."""

    result: jax.Array = projection_matrix @ values
    return result


def _to_grid_matrix(values: jax.Array, basis_grid: jax.Array) -> jax.Array:
    """Project a mesh-space kernel `(N, N)` onto the radial grid `(M_r, M_r)`."""

    result: jax.Array = basis_grid @ values @ basis_grid.T
    return result


_TO_GRID_VECTOR_JIT = jax.jit(  # pyright: ignore[reportUnknownMemberType] -- JAX jit wrappers are not precisely typed at module scope.
    _to_grid_vector
)
_FROM_GRID_ARRAY_JIT = jax.jit(  # pyright: ignore[reportUnknownMemberType] -- JAX jit wrappers are not precisely typed at module scope.
    _from_grid_array
)
_TO_GRID_MATRIX_JIT = jax.jit(  # pyright: ignore[reportUnknownMemberType] -- JAX jit wrappers are not precisely typed at module scope.
    _to_grid_matrix
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

"""Momentum-space transforms built from precomputed basis-evaluation matrices."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import scipy.special as sc

from lax.meshes._basis_eval import basis_at
from lax.types import DoubleFourierTransform, FourierTransform, Mesh, TransformMatrices


@dataclass(frozen=True)
class _FourierProjection:
    """Pickle-safe Fourier-Bessel transform."""

    fourier_matrices: jax.Array

    def __call__(self, values: jax.Array, channel_index: int = 0) -> jax.Array:
        """Project mesh coefficients or kernels onto the momentum grid."""

        if values.ndim == 1:
            return cast(
                jax.Array,
                _FOURIER_VECTOR_JIT(values, self.fourier_matrices, channel_index),
            )
        return cast(
            jax.Array,
            _FOURIER_MATRIX_JIT(values, self.fourier_matrices, channel_index),
        )


@dataclass(frozen=True)
class _DoubleFourierProjection:
    """Pickle-safe double Fourier-Bessel transform."""

    fourier_matrices: jax.Array

    def __call__(
        self,
        values: jax.Array,
        left_channel_index: int = 0,
        right_channel_index: int | None = None,
    ) -> jax.Array:
        """Project a mesh-space kernel onto left/right momentum grids."""

        if values.ndim != 2:
            msg = "double_fourier_transform expects a rank-2 mesh-space kernel."
            raise ValueError(msg)

        resolved_right_channel_index = (
            left_channel_index if right_channel_index is None else right_channel_index
        )
        return cast(
            jax.Array,
            _DOUBLE_FOURIER_JIT(
                values,
                self.fourier_matrices,
                left_channel_index,
                resolved_right_channel_index,
            ),
        )


def compute_F_momentum(
    mesh: Mesh,
    momenta: jax.Array,
    angular_momentum: int,
    n_quad: int = 200,
) -> jax.Array:
    """Compute the Fourier-Bessel transform matrix ``F[k, j] = √(2/π) ∫ j_ℓ(k r) f_j(r) dr``.

    Evaluates the partial-wave Fourier transform matrix at compile time using
    an internal Gauss-Legendre quadrature.  [DESIGN.md §13.2]

    Parameters
    ----------
    mesh
        Compiled mesh supplying basis functions and physical scale.
    momenta
        Momentum grid in fm⁻¹, shape ``(M_k,)``.
    angular_momentum
        Orbital angular momentum ``ℓ`` selecting the spherical Bessel function
        ``j_ℓ``.
    n_quad
        Number of internal quadrature points.  200 is sufficient for meshes
        with ``n ≤ 60``; increase for very large or highly oscillatory bases.

    Returns
    -------
    jax.Array
        Transform matrix, shape ``(M_k, N)``.
    """

    radii, weights = _fourier_quadrature(mesh, n_quad)
    basis_values = np.asarray(
        basis_at(
            mesh,
            jnp.asarray(radii),
        )
    )
    momenta_np = np.asarray(momenta, dtype=np.float64)

    matrix = np.zeros((momenta_np.shape[0], mesh.n), dtype=np.float64)
    prefactor = np.sqrt(2.0 / np.pi)
    for momentum_index, momentum in enumerate(momenta_np):
        bessel = np.asarray(sc.spherical_jn(angular_momentum, momentum * radii))
        integrand = bessel[:, None] * basis_values
        matrix[momentum_index] = prefactor * (weights @ integrand)

    result: jax.Array = jnp.asarray(matrix)
    return result


def _fourier_quadrature(mesh: Mesh, n_quad: int) -> tuple[np.ndarray, np.ndarray]:
    """Return compile-time quadrature nodes and weights for one mesh family."""

    if mesh.family == "legendre":
        return _finite_interval_quadrature(mesh.scale, n_quad)
    if mesh.family == "laguerre":
        return _half_line_quadrature(mesh.scale, n_quad)
    msg = f"No Fourier quadrature is implemented for mesh family {mesh.family!r}."
    raise ValueError(msg)


def _finite_interval_quadrature(scale: float, n_quad: int) -> tuple[np.ndarray, np.ndarray]:
    """Return Gauss-Legendre quadrature on `(0, scale)`."""

    quadrature_nodes, quadrature_weights = np.polynomial.legendre.leggauss(n_quad)
    radii = 0.5 * scale * (quadrature_nodes + 1.0)
    weights = 0.5 * scale * quadrature_weights
    return radii, weights


def _half_line_quadrature(scale: float, n_quad: int) -> tuple[np.ndarray, np.ndarray]:
    """Return Gauss-Legendre quadrature mapped from `(-1, 1)` to `(0, ∞)`."""

    quadrature_nodes, quadrature_weights = np.polynomial.legendre.leggauss(n_quad)
    radii = scale * (1.0 + quadrature_nodes) / (1.0 - quadrature_nodes)
    jacobian = 2.0 * scale / (1.0 - quadrature_nodes) ** 2
    weights = quadrature_weights * jacobian
    return radii, weights


def make_fourier(transform_matrices: TransformMatrices) -> FourierTransform:
    """Return a JIT-compiled Fourier-Bessel transform bound to precomputed matrices.

    Parameters
    ----------
    transform_matrices
        Must have ``F_momentum`` populated (i.e. ``momenta`` was passed to
        :func:`lax.compile`).

    Returns
    -------
    FourierTransform
        Callable: ``fourier(values, channel_index=0) → momentum-space array``.

    Raises
    ------
    ValueError
        If ``transform_matrices.F_momentum`` is ``None``.
    """

    if transform_matrices.F_momentum is None:
        msg = "TransformMatrices.F_momentum is required to build the Fourier transform."
        raise ValueError(msg)

    return _FourierProjection(fourier_matrices=transform_matrices.F_momentum)


def make_double_fourier(transform_matrices: TransformMatrices) -> DoubleFourierTransform:
    """Return a JIT-compiled double Fourier-Bessel transform for mesh kernels.

    Parameters
    ----------
    transform_matrices
        Must have ``F_momentum`` populated.

    Returns
    -------
    DoubleFourierTransform
        Callable: ``double_fourier(values, left_channel_index, right_channel_index)
        → (M_k, M_k)`` kernel in momentum space.

    Raises
    ------
    ValueError
        If ``transform_matrices.F_momentum`` is ``None``.
    """

    if transform_matrices.F_momentum is None:
        msg = "TransformMatrices.F_momentum is required to build the double Fourier transform."
        raise ValueError(msg)

    return cast(
        DoubleFourierTransform,
        _DoubleFourierProjection(fourier_matrices=transform_matrices.F_momentum),
    )


def _fourier_vector(
    values: jax.Array, fourier_matrices: jax.Array, channel_index: int
) -> jax.Array:
    """Project mesh coefficients `(N,)` onto the momentum grid `(M_k,)`."""

    matrix = fourier_matrices[channel_index]
    result: jax.Array = matrix @ values
    return result


def _fourier_matrix(
    values: jax.Array, fourier_matrices: jax.Array, channel_index: int
) -> jax.Array:
    """Project a mesh kernel `(N, N)` onto the momentum grid `(M_k, M_k)`."""

    matrix = fourier_matrices[channel_index]
    result: jax.Array = matrix @ values @ matrix.T
    return result


def _double_fourier(
    values: jax.Array,
    fourier_matrices: jax.Array,
    left_channel_index: int,
    right_channel_index: int,
) -> jax.Array:
    """Project a mesh kernel onto left/right momentum grids."""

    left_matrix = fourier_matrices[left_channel_index]
    right_matrix = fourier_matrices[right_channel_index]
    result: jax.Array = left_matrix @ values @ right_matrix.T
    return result


_FOURIER_VECTOR_JIT = jax.jit(
    _fourier_vector,
    static_argnames=("channel_index",),
)
_FOURIER_MATRIX_JIT = jax.jit(
    _fourier_matrix,
    static_argnames=("channel_index",),
)
_DOUBLE_FOURIER_JIT = jax.jit(
    _double_fourier,
    static_argnames=("left_channel_index", "right_channel_index"),
)


__all__ = ["compute_F_momentum", "make_double_fourier", "make_fourier"]

"""Spectrum-kernel construction."""

from __future__ import annotations

from typing import cast

import jax
import jax.numpy as jnp
import numpy as np

from lax.boundary._types import Mesh, OperatorMatrices, SpectrumKernel
from lax.spectral.types import Spectrum
from lax.types import ChannelSpec, Method

from .assembly import assemble_block_hamiltonian, build_Q


def make_spectrum_kernel(
    mesh: Mesh,
    operators: OperatorMatrices,
    channels: tuple[ChannelSpec, ...],
    method: Method = "eigh",
    keep_eigenvectors: bool = False,
) -> SpectrumKernel:
    """Build a JIT-compiled `spectrum(V) -> Spectrum` kernel. [DESIGN.md §11.1]"""

    q = build_Q(mesh, channels)

    if method == "eigh":

        def spectrum(potential: jax.Array) -> Spectrum:
            hamiltonian = assemble_block_hamiltonian(mesh, operators, channels, potential)
            eigensystem = cast(
                tuple[jax.Array, jax.Array],
                jnp.linalg.eigh(  # pyright: ignore[reportUnknownMemberType] -- JAX linalg stubs lose tuple precision here.
                    hamiltonian
                ),
            )
            eigenvalues, eigenvectors = eigensystem
            surface_amplitudes: jax.Array = eigenvectors.T @ q
            return Spectrum(
                eigenvalues=eigenvalues,
                surface_amplitudes=surface_amplitudes,
                eigenvectors=eigenvectors if keep_eigenvectors else None,
                is_hermitian=True,
            )

        return cast(
            SpectrumKernel,
            jax.jit(spectrum),  # pyright: ignore[reportUnknownMemberType] -- JAX jit wrappers are not precisely typed.
        )

    if method == "eig":

        def spectrum(potential: jax.Array) -> Spectrum:
            hamiltonian = assemble_block_hamiltonian(mesh, operators, channels, potential)
            eigenvalues, eigenvectors = _eig_via_callback(hamiltonian)
            bilinear_norm = jnp.sqrt(jnp.diag(eigenvectors.T @ eigenvectors))
            eigenvectors_normalized = eigenvectors / bilinear_norm[None, :]
            surface_amplitudes: jax.Array = eigenvectors_normalized.T @ q
            return Spectrum(
                eigenvalues=eigenvalues,
                surface_amplitudes=surface_amplitudes,
                eigenvectors=eigenvectors_normalized if keep_eigenvectors else None,
                is_hermitian=False,
            )

        return cast(
            SpectrumKernel,
            jax.jit(spectrum),  # pyright: ignore[reportUnknownMemberType] -- JAX jit wrappers are not precisely typed.
        )

    msg = f"Method {method!r} is not implemented in the MVP spectrum kernel."
    raise ValueError(msg)


def _eig_via_callback(hamiltonian: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Return a complex eigensystem via host NumPy for complex-symmetric problems."""

    matrix_size = hamiltonian.shape[0]
    result_shape = (
        jax.ShapeDtypeStruct((matrix_size,), jnp.complex128),
        jax.ShapeDtypeStruct((matrix_size, matrix_size), jnp.complex128),
    )

    def numpy_eig(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values, vectors = np.linalg.eig(matrix.astype(np.complex128))
        return values.astype(np.complex128), vectors.astype(np.complex128)

    complex_hamiltonian = hamiltonian.astype(jnp.complex128)
    eigensystem = jax.pure_callback(  # pyright: ignore[reportUnknownMemberType] -- JAX callback stubs are imprecise.
        numpy_eig,
        result_shape,
        complex_hamiltonian,
    )
    return cast(tuple[jax.Array, jax.Array], eigensystem)


__all__ = ["make_spectrum_kernel"]

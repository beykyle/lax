"""Spectrum-kernel construction."""

from __future__ import annotations

from typing import cast

import jax

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

    if method != "eigh":
        msg = f"Method {method!r} is not implemented in the MVP spectrum kernel."
        raise ValueError(msg)

    q = build_Q(mesh, channels)

    def spectrum(potential: jax.Array) -> Spectrum:
        hamiltonian = assemble_block_hamiltonian(mesh, operators, channels, potential)
        eigensystem = cast(
            tuple[jax.Array, jax.Array],
            jax.numpy.linalg.eigh(  # pyright: ignore[reportUnknownMemberType] -- JAX linalg stubs lose tuple precision here.
                hamiltonian
            ),
        )
        eigenvalues, eigenvectors = eigensystem
        eigenvectors_t: jax.Array = jax.numpy.transpose(  # pyright: ignore[reportUnknownMemberType] -- JAX transpose stubs are imprecise.
            eigenvectors
        )
        surface_amplitudes: jax.Array = eigenvectors_t @ q
        return Spectrum(
            eigenvalues=eigenvalues,
            surface_amplitudes=surface_amplitudes,
            eigenvectors=eigenvectors if keep_eigenvectors else None,
            is_hermitian=True,
        )

    spectrum_jit = cast(
        SpectrumKernel,
        jax.jit(spectrum),  # pyright: ignore[reportUnknownMemberType] -- JAX jit wrappers are not precisely typed.
    )
    return spectrum_jit


__all__ = ["make_spectrum_kernel"]

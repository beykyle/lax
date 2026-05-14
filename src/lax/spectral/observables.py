"""Spectral sums built from a mesh-independent :class:`~lax.spectral.types.Spectrum`."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from lax.spectral.types import Spectrum


def rmatrix_from_spectrum(
    spectrum: Spectrum,
    energy: float | jax.Array,
    channel_radius: float,
    mass_factor: float,
) -> jax.Array:
    """Return the Wigner-Eisenbud R-matrix. [DESIGN.md §10.2]"""

    gamma = spectrum.surface_amplitudes
    energy_dimless = energy / mass_factor
    denom = 1.0 / (spectrum.eigenvalues - energy_dimless)
    matrix: jax.Array = (
        jnp.einsum(  # pyright: ignore[reportUnknownMemberType] -- JAX exposes an imprecise overloaded stub for einsum.
            "m,mc,md->cd",
            denom,
            gamma,
            gamma,
        )
        / channel_radius
    )
    return matrix


def greens_from_spectrum(
    spectrum: Spectrum,
    energy: float | jax.Array,
    mass_factor: float,
) -> jax.Array:
    """Return the resolvent `(H - E / mass_factor)^-1`. [DESIGN.md §10.2]"""

    eigenvectors = spectrum.eigenvectors
    if eigenvectors is None:
        msg = "Green's function evaluation requires stored eigenvectors."
        raise ValueError(msg)

    energy_dimless = energy / mass_factor
    denom = 1.0 / (spectrum.eigenvalues - energy_dimless)
    transpose = eigenvectors.conj().T if spectrum.is_hermitian else eigenvectors.T
    matrix: jax.Array = (eigenvectors * denom[None, :]) @ transpose
    return matrix


def wavefunction_internal_from_spectrum(
    spectrum: Spectrum,
    energy: float | jax.Array,
    source: jax.Array,
    mass_factor: float,
) -> jax.Array:
    """Return the internal wavefunction from the resolvent action."""

    greens = greens_from_spectrum(spectrum, energy=energy, mass_factor=mass_factor)
    values: jax.Array = greens @ source
    return values


__all__ = [
    "greens_from_spectrum",
    "rmatrix_from_spectrum",
    "wavefunction_internal_from_spectrum",
]

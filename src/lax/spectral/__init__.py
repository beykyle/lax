"""Mesh-independent spectral types, observables, and matching utilities."""

from lax.spectral.interpolation import pade_interpolate
from lax.spectral.matching import phases_from_S, smatrix_from_R
from lax.spectral.observables import (
    greens_from_spectrum,
    rmatrix_from_spectrum,
    wavefunction_internal_from_spectrum,
)
from lax.spectral.types import Spectrum

__all__ = [
    "Spectrum",
    "greens_from_spectrum",
    "pade_interpolate",
    "phases_from_S",
    "rmatrix_from_spectrum",
    "smatrix_from_R",
    "wavefunction_internal_from_spectrum",
]

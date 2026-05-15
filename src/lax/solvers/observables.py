"""Bound runtime observable closures for compiled solvers."""

from __future__ import annotations

from typing import cast

import jax

from lax.boundary._types import (
    BoundaryValues,
    EigenpairAccessor,
    GreenFunctionObservable,
    Mesh,
    RMatrixObservable,
    SpectrumObservable,
    WavefunctionObservable,
)
from lax.spectral.matching import phases_from_S, smatrix_from_R
from lax.spectral.observables import (
    greens_from_spectrum,
    rmatrix_from_spectrum,
    wavefunction_internal_from_spectrum,
)
from lax.spectral.types import Spectrum
from lax.types import ChannelSpec


def bind_observables(
    mesh: Mesh,
    channels: tuple[ChannelSpec, ...],
    energies: jax.Array,
    boundary: BoundaryValues | None,
) -> tuple[
    RMatrixObservable,
    SpectrumObservable | None,
    SpectrumObservable | None,
    GreenFunctionObservable,
    WavefunctionObservable,
    EigenpairAccessor,
]:
    """Bind solver observables to cached mesh and boundary data. [DESIGN.md §11.2]"""

    channel_radius = mesh.scale
    mass_factor = _uniform_mass_factor(channels)

    def rmatrix(spectrum: Spectrum, energy: float | jax.Array) -> jax.Array:
        return rmatrix_from_spectrum(
            spectrum,
            energy=energy,
            channel_radius=channel_radius,
            mass_factor=mass_factor,
        )

    rmatrix_bound = cast(
        RMatrixObservable,
        jax.jit(rmatrix),  # pyright: ignore[reportUnknownMemberType] -- JAX jit wrappers are not precisely typed.
    )
    smatrix = _bind_smatrix(rmatrix_bound, energies, boundary) if boundary is not None else None
    phases = _bind_phases(smatrix) if smatrix is not None else None

    def greens(spectrum: Spectrum, energy: float | jax.Array) -> jax.Array:
        return greens_from_spectrum(spectrum, energy=energy, mass_factor=mass_factor)

    def wavefunction(spectrum: Spectrum, energy: float | jax.Array, source: jax.Array) -> jax.Array:
        return wavefunction_internal_from_spectrum(
            spectrum,
            energy=energy,
            source=source,
            mass_factor=mass_factor,
        )

    def eigh(spectrum: Spectrum) -> tuple[jax.Array, jax.Array | None]:
        return spectrum.eigenvalues, spectrum.eigenvectors

    greens_bound = cast(
        GreenFunctionObservable,
        jax.jit(greens),  # pyright: ignore[reportUnknownMemberType] -- JAX jit wrappers are not precisely typed.
    )
    wavefunction_bound = cast(
        WavefunctionObservable,
        jax.jit(wavefunction),  # pyright: ignore[reportUnknownMemberType] -- JAX jit wrappers are not precisely typed.
    )
    eigh_bound = cast(
        EigenpairAccessor,
        jax.jit(eigh),  # pyright: ignore[reportUnknownMemberType] -- JAX jit wrappers are not precisely typed.
    )

    return rmatrix_bound, smatrix, phases, greens_bound, wavefunction_bound, eigh_bound


def _bind_smatrix(
    rmatrix: RMatrixObservable,
    energies: jax.Array,
    boundary: BoundaryValues,
) -> SpectrumObservable:
    """Bind the compile-time energy-grid S-matrix evaluator."""

    def smatrix(spectrum: Spectrum) -> jax.Array:
        def one_energy(
            energy: jax.Array,
            h_plus: jax.Array,
            h_minus: jax.Array,
            h_plus_p: jax.Array,
            h_minus_p: jax.Array,
            is_open: jax.Array,
        ) -> jax.Array:
            r = rmatrix(spectrum, energy)
            masked_r = _mask_rmatrix(r, is_open)
            boundary_slice = BoundaryValues(
                H_plus=h_plus * is_open.astype(h_plus.dtype),
                H_minus=h_minus * is_open.astype(h_minus.dtype),
                H_plus_p=h_plus_p * is_open.astype(h_plus_p.dtype),
                H_minus_p=h_minus_p * is_open.astype(h_minus_p.dtype),
                is_open=is_open,
            )
            return smatrix_from_R(masked_r, boundary_slice)

        result: jax.Array = jax.vmap(one_energy)(  # pyright: ignore[reportUnknownMemberType] -- JAX vmap wrappers are not precisely typed.
            energies,
            boundary.H_plus,
            boundary.H_minus,
            boundary.H_plus_p,
            boundary.H_minus_p,
            boundary.is_open,
        )
        return result

    return cast(
        SpectrumObservable,
        jax.jit(smatrix),  # pyright: ignore[reportUnknownMemberType] -- JAX jit wrappers are not precisely typed.
    )


def _bind_phases(
    smatrix: SpectrumObservable,
) -> SpectrumObservable:
    """Bind the compile-time energy-grid phase-shift evaluator."""

    def phases(spectrum: Spectrum) -> jax.Array:
        return jax.vmap(phases_from_S)(  # pyright: ignore[reportUnknownMemberType] -- JAX vmap wrappers are not precisely typed.
            smatrix(spectrum)
        )

    return cast(
        SpectrumObservable,
        jax.jit(phases),  # pyright: ignore[reportUnknownMemberType] -- JAX jit wrappers are not precisely typed.
    )


def _mask_rmatrix(rmatrix: jax.Array, is_open: jax.Array) -> jax.Array:
    """Mask closed-channel rows and columns in the R-matrix."""

    mask = is_open.astype(rmatrix.dtype)
    return rmatrix * mask[:, None] * mask[None, :]


def _uniform_mass_factor(channels: tuple[ChannelSpec, ...]) -> float:
    """Return the shared mass factor expected by the MVP observables."""

    mass_factor = channels[0].mass_factor
    for channel in channels[1:]:
        if channel.mass_factor != mass_factor:
            msg = "The MVP solver path requires a uniform mass_factor across channels."
            raise ValueError(msg)
    return mass_factor


__all__ = ["bind_observables"]

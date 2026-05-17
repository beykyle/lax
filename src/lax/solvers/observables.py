"""Bound runtime observable closures for compiled solvers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import jax
import jax.numpy as jnp

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


@dataclass(frozen=True)
class _RMatrixObservable:
    """Pickle-safe spectral R-matrix observable."""

    channel_radius: float
    mass_factor: float

    def __call__(self, spectrum: Spectrum, energy: float | jax.Array) -> jax.Array:
        """Evaluate the R-matrix at one energy."""

        return cast(
            jax.Array,
            _RMATRIX_JIT(spectrum, energy, self.channel_radius, self.mass_factor),
        )


@dataclass(frozen=True)
class _SMatrixObservable:
    """Pickle-safe S-matrix observable on the compile-time energy grid."""

    energies: jax.Array
    boundary: BoundaryValues
    channel_radius: float
    mass_factor: float

    def __call__(self, spectrum: Spectrum) -> jax.Array:
        """Evaluate the S-matrix on the compile-time energy grid."""

        return cast(
            jax.Array,
            _SMATRIX_JIT(
                spectrum,
                self.energies,
                self.boundary,
                self.channel_radius,
                self.mass_factor,
            ),
        )


@dataclass(frozen=True)
class _PhasesObservable:
    """Pickle-safe phase-shift observable on the compile-time energy grid."""

    energies: jax.Array
    boundary: BoundaryValues
    channel_radius: float
    mass_factor: float

    def __call__(self, spectrum: Spectrum) -> jax.Array:
        """Evaluate the phase shifts on the compile-time energy grid."""

        return cast(
            jax.Array,
            _PHASES_JIT(
                spectrum,
                self.energies,
                self.boundary,
                self.channel_radius,
                self.mass_factor,
            ),
        )


@dataclass(frozen=True)
class _GreenFunctionObservable:
    """Pickle-safe Green's-function observable."""

    mass_factor: float

    def __call__(self, spectrum: Spectrum, energy: float | jax.Array) -> jax.Array:
        """Evaluate the Green's function at one energy."""

        return cast(jax.Array, _GREENS_JIT(spectrum, energy, self.mass_factor))


@dataclass(frozen=True)
class _WavefunctionObservable:
    """Pickle-safe internal wavefunction observable."""

    mass_factor: float

    def __call__(self, spectrum: Spectrum, energy: float | jax.Array, source: jax.Array) -> jax.Array:
        """Evaluate the internal wavefunction at one energy."""

        return cast(
            jax.Array,
            _WAVEFUNCTION_JIT(spectrum, energy, source, self.mass_factor),
        )


@dataclass(frozen=True)
class _EigenpairAccessor:
    """Pickle-safe accessor for raw eigensystem data."""

    def __call__(self, spectrum: Spectrum) -> tuple[jax.Array, jax.Array | None]:
        """Return the stored eigenvalues and optional eigenvectors."""

        return cast(tuple[jax.Array, jax.Array | None], _EIGH_JIT(spectrum))


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
    rmatrix = _RMatrixObservable(channel_radius=channel_radius, mass_factor=mass_factor)
    smatrix = (
        _SMatrixObservable(
            energies=energies,
            boundary=boundary,
            channel_radius=channel_radius,
            mass_factor=mass_factor,
        )
        if boundary is not None
        else None
    )
    phases = (
        _PhasesObservable(
            energies=energies,
            boundary=boundary,
            channel_radius=channel_radius,
            mass_factor=mass_factor,
        )
        if boundary is not None
        else None
    )
    greens = _GreenFunctionObservable(mass_factor=mass_factor)
    wavefunction = _WavefunctionObservable(mass_factor=mass_factor)
    eigh = _EigenpairAccessor()

    return rmatrix, smatrix, phases, greens, wavefunction, eigh


def _rmatrix(
    spectrum: Spectrum,
    energy: float | jax.Array,
    channel_radius: float,
    mass_factor: float,
) -> jax.Array:
    """Return the spectral R-matrix at one energy."""

    return rmatrix_from_spectrum(
        spectrum,
        energy=energy,
        channel_radius=channel_radius,
        mass_factor=mass_factor,
    )


def _smatrix(
    spectrum: Spectrum,
    energies: jax.Array,
    boundary: BoundaryValues,
    channel_radius: float,
    mass_factor: float,
) -> jax.Array:
    """Return the compile-time energy-grid S-matrix."""

    def one_energy(
        energy: jax.Array,
        h_plus: jax.Array,
        h_minus: jax.Array,
        h_plus_p: jax.Array,
        h_minus_p: jax.Array,
        is_open: jax.Array,
    ) -> jax.Array:
        r = _rmatrix(spectrum, energy, channel_radius, mass_factor)
        decoupled_r = _decouple_closed_channels(r, h_plus, h_plus_p, is_open)
        projected_r, boundary_slice = _project_open_channels(
            decoupled_r,
            h_plus,
            h_minus,
            h_plus_p,
            h_minus_p,
            is_open,
        )
        return smatrix_from_R(projected_r, boundary_slice)

    result: jax.Array = jax.vmap(one_energy)(  # pyright: ignore[reportUnknownMemberType] -- JAX vmap wrappers are not precisely typed.
        energies,
        boundary.H_plus,
        boundary.H_minus,
        boundary.H_plus_p,
        boundary.H_minus_p,
        boundary.is_open,
    )
    return result


def _phases(
    spectrum: Spectrum,
    energies: jax.Array,
    boundary: BoundaryValues,
    channel_radius: float,
    mass_factor: float,
) -> jax.Array:
    """Return the compile-time energy-grid phase shifts."""

    return jax.vmap(phases_from_S)(  # pyright: ignore[reportUnknownMemberType] -- JAX vmap wrappers are not precisely typed.
        _smatrix(spectrum, energies, boundary, channel_radius, mass_factor)
    )


def _greens(spectrum: Spectrum, energy: float | jax.Array, mass_factor: float) -> jax.Array:
    """Return the Green's function at one energy."""

    return greens_from_spectrum(spectrum, energy=energy, mass_factor=mass_factor)


def _wavefunction(
    spectrum: Spectrum,
    energy: float | jax.Array,
    source: jax.Array,
    mass_factor: float,
) -> jax.Array:
    """Return the internal wavefunction at one energy."""

    return wavefunction_internal_from_spectrum(
        spectrum,
        energy=energy,
        source=source,
        mass_factor=mass_factor,
    )


def _eigh(spectrum: Spectrum) -> tuple[jax.Array, jax.Array | None]:
    """Return the stored eigenvalues and optional eigenvectors."""

    return spectrum.eigenvalues, spectrum.eigenvectors


_RMATRIX_JIT = jax.jit(  # pyright: ignore[reportUnknownMemberType] -- JAX jit wrappers are not precisely typed at module scope.
    _rmatrix,
    static_argnames=("channel_radius", "mass_factor"),
)
_SMATRIX_JIT = jax.jit(  # pyright: ignore[reportUnknownMemberType] -- JAX jit wrappers are not precisely typed at module scope.
    _smatrix,
    static_argnames=("channel_radius", "mass_factor"),
)
_PHASES_JIT = jax.jit(  # pyright: ignore[reportUnknownMemberType] -- JAX jit wrappers are not precisely typed at module scope.
    _phases,
    static_argnames=("channel_radius", "mass_factor"),
)
_GREENS_JIT = jax.jit(  # pyright: ignore[reportUnknownMemberType] -- JAX jit wrappers are not precisely typed at module scope.
    _greens,
    static_argnames=("mass_factor",),
)
_WAVEFUNCTION_JIT = jax.jit(  # pyright: ignore[reportUnknownMemberType] -- JAX jit wrappers are not precisely typed at module scope.
    _wavefunction,
    static_argnames=("mass_factor",),
)
_EIGH_JIT = jax.jit(_eigh)  # pyright: ignore[reportUnknownMemberType] -- JAX jit wrappers are not precisely typed at module scope.


def _decouple_closed_channels(
    rmatrix: jax.Array,
    h_plus: jax.Array,
    h_plus_p: jax.Array,
    is_open: jax.Array,
) -> jax.Array:
    """Fold closed-channel Whittaker boundary conditions into an effective R-matrix."""

    bloch = _closed_channel_bloch(h_plus, h_plus_p, is_open)
    identity: jax.Array = jnp.eye(  # pyright: ignore[reportUnknownMemberType] -- JAX eye stubs are imprecise.
        rmatrix.shape[0],
        dtype=rmatrix.dtype,
    )
    correction = identity - rmatrix @ jnp.diag(bloch)
    return cast(
        jax.Array,
        jnp.linalg.solve(  # pyright: ignore[reportUnknownMemberType] -- JAX linalg solve stubs lose the result type here.
            correction.T,
            rmatrix.T,
        ).T,
    )


def _project_open_channels(
    rmatrix: jax.Array,
    h_plus: jax.Array,
    h_minus: jax.Array,
    h_plus_p: jax.Array,
    h_minus_p: jax.Array,
    is_open: jax.Array,
) -> tuple[jax.Array, BoundaryValues]:
    """Project the decoupled system onto the open-channel S-matrix with fixed shapes."""

    mask = is_open.astype(rmatrix.dtype)
    projected_r = rmatrix * mask[:, None] * mask[None, :]
    closed_dtype = h_plus.dtype
    ones: jax.Array = jnp.ones_like(  # pyright: ignore[reportUnknownMemberType] -- JAX ones_like stubs are imprecise.
        h_plus,
        dtype=closed_dtype,
    )
    mask_complex = is_open.astype(closed_dtype)

    boundary_slice = BoundaryValues(
        H_plus=h_plus * mask_complex + ones * (1.0 - mask_complex),
        H_minus=h_minus * mask_complex,
        H_plus_p=h_plus_p * mask_complex,
        H_minus_p=h_minus_p * mask_complex,
        is_open=is_open,
    )
    return projected_r, boundary_slice


def _closed_channel_bloch(
    h_plus: jax.Array,
    h_plus_p: jax.Array,
    is_open: jax.Array,
) -> jax.Array:
    """Return the closed-channel Bloch boundary parameter `B_c = H'_c / H_c`."""

    ratio = h_plus_p / h_plus
    closed_mask = jnp.logical_not(is_open)  # pyright: ignore[reportUnknownMemberType] -- JAX logical_not stubs are imprecise.
    zeros: jax.Array = jnp.zeros_like(  # pyright: ignore[reportUnknownMemberType] -- JAX zeros_like stubs are imprecise.
        ratio
    )
    return jnp.where(  # pyright: ignore[reportUnknownMemberType] -- JAX where stubs are imprecise.
        closed_mask,
        ratio,
        zeros,
    )


def _uniform_mass_factor(channels: tuple[ChannelSpec, ...]) -> float:
    """Return the shared mass factor expected by the MVP observables."""

    mass_factor = channels[0].mass_factor
    for channel in channels[1:]:
        if channel.mass_factor != mass_factor:
            msg = "The MVP solver path requires a uniform mass_factor across channels."
            raise ValueError(msg)
    return mass_factor


__all__ = ["bind_observables"]

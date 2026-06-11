"""Pickle-safe observable wrappers bound to compiled solver caches.

The public ``Solver`` bundle stores module-level callable objects rather than local
closures so it can round-trip through stdlib ``pickle``. This module provides those
wrappers plus the helper functions that bind compile-time mesh, energy-grid, and
boundary data to the runtime spectral formulas from ``lax.spectral``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import jax
import jax.numpy as jnp

from lax.spectral.interpolation import pade_interpolate
from lax.spectral.matching import open_channel_smatrix_from_R, phases_from_S
from lax.spectral.observables import (
    greens_from_spectrum,
    rmatrix_from_spectrum,
    wavefunction_internal_from_spectrum,
)
from lax.spectral.types import Spectrum
from lax.types import (
    BoundaryValues,
    ChannelSpec,
    EigenpairAccessor,
    GreenFunctionObservable,
    InterpolatorBuilder,
    Mesh,
    RMatrixObservable,
    SpectrumGridObservable,
    SpectrumObservable,
    WavefunctionObservable,
)

from .assembly import uniform_mass_factor


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
        """Evaluate the phase shifts on the compile-time energy grid.

        Returns
        -------
        jax.Array
            Shape ``(N_E, N_c)`` in radians.  For a single-channel solver use
            ``result[:, 0]`` to obtain a 1-D energy curve.
        """

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

    def __call__(
        self, spectrum: Spectrum, energy: float | jax.Array, source: jax.Array
    ) -> jax.Array:
        """Evaluate the internal wavefunction at one energy."""

        return cast(
            jax.Array,
            _WAVEFUNCTION_JIT(spectrum, energy, source, self.mass_factor),
        )


@dataclass(frozen=True)
class _EigenpairAccessor:
    """Pickle-safe accessor for raw eigensystem data."""

    def __call__(self, spectrum: Spectrum) -> tuple[jax.Array, jax.Array]:
        """Return the stored eigenvalues and eigenvectors.

        Raises
        ------
        RuntimeError
            If eigenvectors were not retained at compile time.
        """

        return cast(tuple[jax.Array, jax.Array], _EIGH_JIT(spectrum))


@dataclass(frozen=True)
class _RMatrixGridObservable:
    """Pickle-safe aligned-grid spectral R-matrix observable."""

    energies: jax.Array
    channel_radius: float
    mass_factor_grid: jax.Array

    def __call__(self, spectra: Spectrum) -> jax.Array:
        """Evaluate `R(E_i; spec_i)` across the compile-time energy grid."""

        return cast(
            jax.Array,
            _RMATRIX_GRID_JIT(
                spectra,
                self.energies,
                self.channel_radius,
                self.mass_factor_grid,
            ),
        )


@dataclass(frozen=True)
class _SMatrixGridObservable:
    """Pickle-safe aligned-grid spectral S-matrix observable."""

    energies: jax.Array
    boundary: BoundaryValues
    channel_radius: float
    mass_factor_grid: jax.Array

    def __call__(self, spectra: Spectrum) -> jax.Array:
        """Evaluate `S(E_i; spec_i)` across the compile-time energy grid."""

        return cast(
            jax.Array,
            _SMATRIX_GRID_JIT(
                spectra,
                self.energies,
                self.boundary,
                self.channel_radius,
                self.mass_factor_grid,
            ),
        )


@dataclass(frozen=True)
class _PhasesGridObservable:
    """Pickle-safe aligned-grid spectral phase-shift observable."""

    energies: jax.Array
    boundary: BoundaryValues
    channel_radius: float
    mass_factor_grid: jax.Array

    def __call__(self, spectra: Spectrum) -> jax.Array:
        """Evaluate `δ(E_i; spec_i)` across the compile-time energy grid."""

        return cast(
            jax.Array,
            _PHASES_GRID_JIT(
                spectra,
                self.energies,
                self.boundary,
                self.channel_radius,
                self.mass_factor_grid,
            ),
        )


@dataclass(frozen=True)
class _InterpolatorBuilder:
    """Pickle-safe Padé interpolation builder bound to one energy grid."""

    energies: jax.Array

    def __call__(
        self,
        values: jax.Array,
        order: tuple[int, int] | None = None,
    ) -> Callable[[float | jax.Array], jax.Array]:
        """Build a Padé interpolator over the compile-time energy grid."""

        return pade_interpolate(values, self.energies, order=order)


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
    """Bind pointwise spectral observables to one compiled solver.

    Parameters
    ----------
    mesh
        Compiled mesh cache containing the channel radius and basis boundary values.
    channels
        Channel layout baked into the solver.
    energies
        Compile-time energy grid. It is stored on matching-dependent observables even
        though pointwise R and Green's functions can be evaluated at arbitrary energy.
    boundary
        Compile-time boundary values for matching-dependent observables. When absent,
        only spectrum-only observables are returned.

    Returns
    -------
    tuple
        Bound runtime entry points for the R-matrix, S-matrix, phase shifts, Green's
        function, internal wavefunction, and raw eigensystem access.
    """

    channel_radius = mesh.scale
    mass_factor = _uniform_mass_factor(channels)
    rmatrix = _RMatrixObservable(channel_radius=channel_radius, mass_factor=mass_factor)
    smatrix, phases = _matching_observables(
        energies=energies,
        boundary=boundary,
        channel_radius=channel_radius,
        mass_factor=mass_factor,
    )
    greens = _GreenFunctionObservable(mass_factor=mass_factor)
    wavefunction = _WavefunctionObservable(mass_factor=mass_factor)
    eigh = _EigenpairAccessor()

    return rmatrix, smatrix, phases, greens, wavefunction, eigh


def bind_grid_observables(
    mesh: Mesh,
    channels: tuple[ChannelSpec, ...],
    energies: jax.Array,
    boundary: BoundaryValues | None,
    mass_factor_grid: jax.Array | None = None,
) -> tuple[SpectrumGridObservable, SpectrumGridObservable | None, SpectrumGridObservable | None]:
    """Bind aligned-grid spectral observables for energy-dependent workflows.

    Parameters
    ----------
    mesh
        Compiled mesh cache.
    channels
        Channel layout baked into the solver.
    energies
        Compile-time energy grid aligned with the batched spectra.
    boundary
        Compile-time matching data. When absent, only the aligned-grid R-matrix
        observable is returned.
    mass_factor_grid
        Per-energy ℏ²/2μ values in MeV·fm², shape ``(N_E,)``, or ``None``
        for a constant mass factor.  When provided the spectral denominators
        ``ε_k − E_i/μ(E_i)`` use the correct per-energy value.

    Returns
    -------
    tuple
        Bound aligned-grid observables for ``R(E_i; spec_i)``, ``S(E_i; spec_i)``,
        and ``δ(E_i; spec_i)``.
    """

    channel_radius = mesh.scale
    mass_factor = _uniform_mass_factor(channels)
    _mfg = (
        jnp.full(len(energies), mass_factor)
        if mass_factor_grid is None
        else jnp.asarray(mass_factor_grid)
    )
    rmatrix_grid = _RMatrixGridObservable(
        energies=energies,
        channel_radius=channel_radius,
        mass_factor_grid=_mfg,
    )
    smatrix_grid, phases_grid = _matching_grid_observables(
        energies=energies,
        boundary=boundary,
        channel_radius=channel_radius,
        mass_factor_grid=_mfg,
    )
    return rmatrix_grid, smatrix_grid, phases_grid


def bind_interpolators(
    energies: jax.Array,
) -> tuple[InterpolatorBuilder, InterpolatorBuilder, InterpolatorBuilder]:
    """Bind Padé interpolation builders to one compile-time energy grid.

    A single builder type serves the R-matrix, S-matrix, and phase-shift
    interpolation paths; the three returned values are aliases kept for public API
    clarity on the ``Solver`` bundle.
    """

    builder = _InterpolatorBuilder(energies=energies)
    return builder, builder, builder


def _matching_observables(
    *,
    energies: jax.Array,
    boundary: BoundaryValues | None,
    channel_radius: float,
    mass_factor: float,
) -> tuple[SpectrumObservable | None, SpectrumObservable | None]:
    """Create matching-dependent observables when boundary data are available."""

    if boundary is None:
        return None, None
    return (
        _SMatrixObservable(
            energies=energies,
            boundary=boundary,
            channel_radius=channel_radius,
            mass_factor=mass_factor,
        ),
        _PhasesObservable(
            energies=energies,
            boundary=boundary,
            channel_radius=channel_radius,
            mass_factor=mass_factor,
        ),
    )


def _matching_grid_observables(
    *,
    energies: jax.Array,
    boundary: BoundaryValues | None,
    channel_radius: float,
    mass_factor_grid: jax.Array,
) -> tuple[SpectrumGridObservable | None, SpectrumGridObservable | None]:
    """Create aligned-grid matching observables when boundary data are available."""

    if boundary is None:
        return None, None
    return (
        _SMatrixGridObservable(
            energies=energies,
            boundary=boundary,
            channel_radius=channel_radius,
            mass_factor_grid=mass_factor_grid,
        ),
        _PhasesGridObservable(
            energies=energies,
            boundary=boundary,
            channel_radius=channel_radius,
            mass_factor_grid=mass_factor_grid,
        ),
    )


def _rmatrix(
    spectrum: Spectrum,
    energy: float | jax.Array,
    channel_radius: float,
    mass_factor: float | jax.Array,
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
    """Return the S-matrix on the compile-time energy grid.

    Vectorises the R-matrix spectral sum and the channel matching over the
    full compile-time energy grid via ``jax.vmap``.

    Parameters
    ----------
    spectrum
        Stored eigenpairs of the Bloch-augmented Hamiltonian.
    energies
        Compile-time energy grid in MeV, shape ``(N_E,)``.
    boundary
        Compile-time boundary values, shape ``(N_E, N_c)`` per field.
    channel_radius, mass_factor
        Channel radius in fm and ℏ²/2μ in MeV·fm².

    Returns
    -------
    jax.Array
        S-matrix on the compile-time grid, shape ``(N_E, N_c, N_c)``.
    """

    wave_numbers = _boundary_wave_numbers(boundary)

    def one_energy(
        energy: jax.Array,
        h_plus: jax.Array,
        h_minus: jax.Array,
        h_plus_p: jax.Array,
        h_minus_p: jax.Array,
        is_open: jax.Array,
        k: jax.Array,
    ) -> jax.Array:
        r = _rmatrix(spectrum, energy, channel_radius, mass_factor)
        return _match_one_energy(r, h_plus, h_minus, h_plus_p, h_minus_p, is_open, k)

    result: jax.Array = jax.vmap(one_energy)(
        energies,
        boundary.H_plus,
        boundary.H_minus,
        boundary.H_plus_p,
        boundary.H_minus_p,
        boundary.is_open,
        wave_numbers,
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

    return jax.vmap(phases_from_S)(
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


def _eigh(spectrum: Spectrum) -> tuple[jax.Array, jax.Array]:
    """Return the stored eigenvalues and eigenvectors.

    Raises
    ------
    RuntimeError
        If eigenvectors were not retained at compile time.  Re-compile with
        ``'greens'`` or ``'wavefunction'`` in ``solvers=`` to keep them.
    """

    if spectrum.eigenvectors is None:
        raise RuntimeError(
            "Eigenvectors were not retained at compile time. "
            "Re-compile with 'greens' or 'wavefunction' in solvers= to keep them."
        )
    return spectrum.eigenvalues, spectrum.eigenvectors


def _rmatrix_grid(
    spectra: Spectrum,
    energies: jax.Array,
    channel_radius: float,
    mass_factor_grid: jax.Array,
) -> jax.Array:
    """Return aligned-grid ``R(E_i; spec_i)`` samples."""

    def one_energy(spectrum: Spectrum, energy: jax.Array, mu: jax.Array) -> jax.Array:
        return _rmatrix(spectrum, energy, channel_radius, mu)

    return jax.vmap(one_energy)(spectra, energies, mass_factor_grid)


def _smatrix_grid(
    spectra: Spectrum,
    energies: jax.Array,
    boundary: BoundaryValues,
    channel_radius: float,
    mass_factor_grid: jax.Array,
) -> jax.Array:
    """Return aligned-grid ``S(E_i; spec_i)`` samples."""

    wave_numbers = _boundary_wave_numbers(boundary)

    def one_energy(
        spectrum: Spectrum,
        energy: jax.Array,
        h_plus: jax.Array,
        h_minus: jax.Array,
        h_plus_p: jax.Array,
        h_minus_p: jax.Array,
        is_open: jax.Array,
        k: jax.Array,
        mu: jax.Array,
    ) -> jax.Array:
        r = _rmatrix(spectrum, energy, channel_radius, mu)
        return _match_one_energy(r, h_plus, h_minus, h_plus_p, h_minus_p, is_open, k)

    return jax.vmap(one_energy)(
        spectra,
        energies,
        boundary.H_plus,
        boundary.H_minus,
        boundary.H_plus_p,
        boundary.H_minus_p,
        boundary.is_open,
        wave_numbers,
        mass_factor_grid,
    )


def _phases_grid(
    spectra: Spectrum,
    energies: jax.Array,
    boundary: BoundaryValues,
    channel_radius: float,
    mass_factor_grid: jax.Array,
) -> jax.Array:
    """Return aligned-grid ``δ(E_i; spec_i)`` samples."""

    return jax.vmap(phases_from_S)(
        _smatrix_grid(spectra, energies, boundary, channel_radius, mass_factor_grid)
    )


_RMATRIX_JIT = jax.jit(
    _rmatrix,
    static_argnames=("channel_radius", "mass_factor"),
)
_SMATRIX_JIT = jax.jit(
    _smatrix,
    static_argnames=("channel_radius", "mass_factor"),
)
_PHASES_JIT = jax.jit(
    _phases,
    static_argnames=("channel_radius", "mass_factor"),
)
_GREENS_JIT = jax.jit(
    _greens,
    static_argnames=("mass_factor",),
)
_WAVEFUNCTION_JIT = jax.jit(
    _wavefunction,
    static_argnames=("mass_factor",),
)
_EIGH_JIT = jax.jit(_eigh)
_RMATRIX_GRID_JIT = jax.jit(
    _rmatrix_grid,
    static_argnames=("channel_radius",),
)
_SMATRIX_GRID_JIT = jax.jit(
    _smatrix_grid,
    static_argnames=("channel_radius",),
)
_PHASES_GRID_JIT = jax.jit(
    _phases_grid,
    static_argnames=("channel_radius",),
)


def _match_one_energy(
    rmatrix: jax.Array,
    h_plus: jax.Array,
    h_minus: jax.Array,
    h_plus_p: jax.Array,
    h_minus_p: jax.Array,
    is_open: jax.Array,
    k: jax.Array,
) -> jax.Array:
    """Match one channel-space R-matrix sample to the physical S-matrix.

    Re-wraps the per-energy boundary arrays as a :class:`BoundaryValues`
    slice and delegates the closed-channel decoupling, open-channel
    projection, and matching to :func:`lax.spectral.matching.open_channel_smatrix_from_R`.
    """

    boundary_slice = BoundaryValues(
        H_plus=h_plus,
        H_minus=h_minus,
        H_plus_p=h_plus_p,
        H_minus_p=h_minus_p,
        is_open=is_open,
        k=k,
    )
    return open_channel_smatrix_from_R(rmatrix, boundary_slice)


def _boundary_wave_numbers(boundary: BoundaryValues) -> jax.Array:
    """Return wave numbers for matching."""

    return boundary.k


def _uniform_mass_factor(channels: tuple[ChannelSpec, ...]) -> float:
    """Return the shared mass factor expected by the spectral observables."""

    return uniform_mass_factor(channels, context="spectral observable path")


__all__ = [
    "bind_grid_observables",
    "bind_interpolators",
    "bind_observables",
]

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

from lax.boundary._types import (
    BoundaryValues,
    DirectGridObservable,
    EigenpairAccessor,
    GreenFunctionObservable,
    InterpolatorBuilder,
    Mesh,
    RMatrixObservable,
    SpectrumGridObservable,
    SpectrumObservable,
    WavefunctionObservable,
)
from lax.spectral.interpolation import pade_interpolate
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
    mass_factor: float
    mass_factor_grid: jax.Array | None = None

    def __call__(self, spectra: Spectrum) -> jax.Array:
        """Evaluate `R(E_i; spec_i)` across the compile-time energy grid."""

        if self.mass_factor_grid is not None:
            return cast(
                jax.Array,
                _RMATRIX_GRID_WITH_MU_JIT(
                    spectra,
                    self.energies,
                    self.channel_radius,
                    self.mass_factor_grid,
                ),
            )
        return cast(
            jax.Array,
            _RMATRIX_GRID_JIT(
                spectra,
                self.energies,
                self.channel_radius,
                self.mass_factor,
            ),
        )


@dataclass(frozen=True)
class _SMatrixGridObservable:
    """Pickle-safe aligned-grid spectral S-matrix observable."""

    energies: jax.Array
    boundary: BoundaryValues
    channel_radius: float
    mass_factor: float
    mass_factor_grid: jax.Array | None = None

    def __call__(self, spectra: Spectrum) -> jax.Array:
        """Evaluate `S(E_i; spec_i)` across the compile-time energy grid."""

        if self.mass_factor_grid is not None:
            return cast(
                jax.Array,
                _SMATRIX_GRID_WITH_MU_JIT(
                    spectra,
                    self.energies,
                    self.boundary,
                    self.channel_radius,
                    self.mass_factor_grid,
                ),
            )
        return cast(
            jax.Array,
            _SMATRIX_GRID_JIT(
                spectra,
                self.energies,
                self.boundary,
                self.channel_radius,
                self.mass_factor,
            ),
        )


@dataclass(frozen=True)
class _PhasesGridObservable:
    """Pickle-safe aligned-grid spectral phase-shift observable."""

    energies: jax.Array
    boundary: BoundaryValues
    channel_radius: float
    mass_factor: float
    mass_factor_grid: jax.Array | None = None

    def __call__(self, spectra: Spectrum) -> jax.Array:
        """Evaluate `δ(E_i; spec_i)` across the compile-time energy grid."""

        if self.mass_factor_grid is not None:
            return cast(
                jax.Array,
                _PHASES_GRID_WITH_MU_JIT(
                    spectra,
                    self.energies,
                    self.boundary,
                    self.channel_radius,
                    self.mass_factor_grid,
                ),
            )
        return cast(
            jax.Array,
            _PHASES_GRID_JIT(
                spectra,
                self.energies,
                self.boundary,
                self.channel_radius,
                self.mass_factor,
            ),
        )


@dataclass(frozen=True)
class _DirectSMatrixGridObservable:
    """Pickle-safe aligned-grid S-matrix observable for direct R-matrices."""

    rmatrix_direct_grid: DirectGridObservable
    boundary: BoundaryValues

    def __call__(self, potentials: jax.Array) -> jax.Array:
        """Evaluate `S(E_i; V_i)` from aligned direct R-matrix samples."""

        rmatrix_grid = self.rmatrix_direct_grid(potentials)
        return cast(jax.Array, _DIRECT_SMATRIX_GRID_JIT(rmatrix_grid, self.boundary))


@dataclass(frozen=True)
class _DirectPhasesGridObservable:
    """Pickle-safe aligned-grid phase observable for direct R-matrices."""

    smatrix_direct_grid: DirectGridObservable

    def __call__(self, potentials: jax.Array) -> jax.Array:
        """Evaluate `δ(E_i; V_i)` from aligned direct S-matrix samples."""

        smatrix_grid = self.smatrix_direct_grid(potentials)
        return cast(jax.Array, _DIRECT_PHASES_GRID_JIT(smatrix_grid))


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
    rmatrix_grid = _RMatrixGridObservable(
        energies=energies,
        channel_radius=channel_radius,
        mass_factor=mass_factor,
        mass_factor_grid=mass_factor_grid,
    )
    smatrix_grid, phases_grid = _matching_grid_observables(
        energies=energies,
        boundary=boundary,
        channel_radius=channel_radius,
        mass_factor=mass_factor,
        mass_factor_grid=mass_factor_grid,
    )
    return rmatrix_grid, smatrix_grid, phases_grid


def bind_direct_grid_observables(
    rmatrix_direct_grid: DirectGridObservable,
    boundary: BoundaryValues | None,
) -> tuple[DirectGridObservable | None, DirectGridObservable | None]:
    """Bind aligned-grid direct S-matrix and phase observables.

    Parameters
    ----------
    rmatrix_direct_grid
        Aligned-grid direct R-matrix observable.
    boundary
        Compile-time boundary values used to match the direct R-matrix onto the
        physical S-matrix.

    Returns
    -------
    tuple
        Bound aligned-grid direct S-matrix and phase observables, or ``(None, None)``
        when no boundary data are available.
    """

    if boundary is None:
        return None, None
    smatrix_direct_grid = _DirectSMatrixGridObservable(
        rmatrix_direct_grid=rmatrix_direct_grid,
        boundary=boundary,
    )
    phases_direct_grid = _DirectPhasesGridObservable(smatrix_direct_grid=smatrix_direct_grid)
    return smatrix_direct_grid, phases_direct_grid


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
    mass_factor: float,
    mass_factor_grid: jax.Array | None = None,
) -> tuple[SpectrumGridObservable | None, SpectrumGridObservable | None]:
    """Create aligned-grid matching observables when boundary data are available."""

    if boundary is None:
        return None, None
    return (
        _SMatrixGridObservable(
            energies=energies,
            boundary=boundary,
            channel_radius=channel_radius,
            mass_factor=mass_factor,
            mass_factor_grid=mass_factor_grid,
        ),
        _PhasesGridObservable(
            energies=energies,
            boundary=boundary,
            channel_radius=channel_radius,
            mass_factor=mass_factor,
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
    mass_factor: float,
) -> jax.Array:
    """Return aligned-grid `R(E_i; spec_i)` samples."""

    def one_energy(spectrum: Spectrum, energy: jax.Array) -> jax.Array:
        return _rmatrix(spectrum, energy, channel_radius, mass_factor)

    return jax.vmap(one_energy)(
        spectra,
        energies,
    )


def _smatrix_grid(
    spectra: Spectrum,
    energies: jax.Array,
    boundary: BoundaryValues,
    channel_radius: float,
    mass_factor: float,
) -> jax.Array:
    """Return aligned-grid `S(E_i; spec_i)` samples."""

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
    ) -> jax.Array:
        r = _rmatrix(spectrum, energy, channel_radius, mass_factor)
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
    )


def _phases_grid(
    spectra: Spectrum,
    energies: jax.Array,
    boundary: BoundaryValues,
    channel_radius: float,
    mass_factor: float,
) -> jax.Array:
    """Return aligned-grid `δ(E_i; spec_i)` samples."""

    return jax.vmap(phases_from_S)(
        _smatrix_grid(spectra, energies, boundary, channel_radius, mass_factor)
    )


def _rmatrix_grid_with_mu(
    spectra: Spectrum,
    energies: jax.Array,
    channel_radius: float,
    mass_factor_grid: jax.Array,
) -> jax.Array:
    """Return aligned-grid ``R(E_i; spec_i)`` samples with per-energy μ(E)."""

    def one_energy(spectrum: Spectrum, energy: jax.Array, mu: jax.Array) -> jax.Array:
        return _rmatrix(spectrum, energy, channel_radius, mu)

    return jax.vmap(one_energy)(spectra, energies, mass_factor_grid)


def _smatrix_grid_with_mu(
    spectra: Spectrum,
    energies: jax.Array,
    boundary: BoundaryValues,
    channel_radius: float,
    mass_factor_grid: jax.Array,
) -> jax.Array:
    """Return aligned-grid ``S(E_i; spec_i)`` samples with per-energy μ(E)."""

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


def _phases_grid_with_mu(
    spectra: Spectrum,
    energies: jax.Array,
    boundary: BoundaryValues,
    channel_radius: float,
    mass_factor_grid: jax.Array,
) -> jax.Array:
    """Return aligned-grid ``δ(E_i; spec_i)`` samples with per-energy μ(E)."""

    return jax.vmap(phases_from_S)(
        _smatrix_grid_with_mu(spectra, energies, boundary, channel_radius, mass_factor_grid)
    )


def _direct_smatrix_grid(rmatrix_grid: jax.Array, boundary: BoundaryValues) -> jax.Array:
    """Return aligned-grid `S(E_i; V_i)` samples from direct R-matrices."""

    wave_numbers = _boundary_wave_numbers(boundary)

    def one_energy(
        rmatrix: jax.Array,
        h_plus: jax.Array,
        h_minus: jax.Array,
        h_plus_p: jax.Array,
        h_minus_p: jax.Array,
        is_open: jax.Array,
        k: jax.Array,
    ) -> jax.Array:
        return _match_one_energy(rmatrix, h_plus, h_minus, h_plus_p, h_minus_p, is_open, k)

    return jax.vmap(one_energy)(
        rmatrix_grid,
        boundary.H_plus,
        boundary.H_minus,
        boundary.H_plus_p,
        boundary.H_minus_p,
        boundary.is_open,
        wave_numbers,
    )


def _direct_phases_grid(smatrix_grid: jax.Array) -> jax.Array:
    """Return aligned-grid `δ(E_i; V_i)` samples from direct S-matrices."""

    return jax.vmap(phases_from_S)(smatrix_grid)


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
    static_argnames=("channel_radius", "mass_factor"),
)
_SMATRIX_GRID_JIT = jax.jit(
    _smatrix_grid,
    static_argnames=("channel_radius", "mass_factor"),
)
_PHASES_GRID_JIT = jax.jit(
    _phases_grid,
    static_argnames=("channel_radius", "mass_factor"),
)
_DIRECT_SMATRIX_GRID_JIT = jax.jit(_direct_smatrix_grid)
_DIRECT_PHASES_GRID_JIT = jax.jit(_direct_phases_grid)
# μ(E)-aware aligned-grid kernels: mass_factor_grid is a traced JAX array in the vmap,
# so it is NOT in static_argnames.  channel_radius remains static.
_RMATRIX_GRID_WITH_MU_JIT = jax.jit(
    _rmatrix_grid_with_mu,
    static_argnames=("channel_radius",),
)
_SMATRIX_GRID_WITH_MU_JIT = jax.jit(
    _smatrix_grid_with_mu,
    static_argnames=("channel_radius",),
)
_PHASES_GRID_WITH_MU_JIT = jax.jit(
    _phases_grid_with_mu,
    static_argnames=("channel_radius",),
)


def _decouple_closed_channels(
    rmatrix: jax.Array,
    h_plus: jax.Array,
    h_plus_p: jax.Array,
    is_open: jax.Array,
) -> jax.Array:
    """Fold closed-channel Whittaker boundary conditions into an effective R-matrix."""

    bloch = _closed_channel_bloch(h_plus, h_plus_p, is_open)
    identity: jax.Array = jnp.eye(
        rmatrix.shape[0],
        dtype=rmatrix.dtype,
    )
    correction = identity - rmatrix @ jnp.diag(bloch)
    return cast(
        jax.Array,
        jnp.linalg.solve(
            correction.T,
            rmatrix.T,
        ).T,
    )


def _match_rmatrix(
    rmatrix: jax.Array,
    h_plus: jax.Array,
    h_minus: jax.Array,
    h_plus_p: jax.Array,
    h_minus_p: jax.Array,
    is_open: jax.Array,
    k: jax.Array | None,
) -> jax.Array:
    """Convert one channel-space R-matrix into the physical S-matrix."""

    decoupled_r = _decouple_closed_channels(rmatrix, h_plus, h_plus_p, is_open)
    projected_r, boundary_slice = _project_open_channels(
        decoupled_r,
        h_plus,
        h_minus,
        h_plus_p,
        h_minus_p,
        is_open,
        k,
    )
    return smatrix_from_R(projected_r, boundary_slice)


def _match_one_energy(
    rmatrix: jax.Array,
    h_plus: jax.Array,
    h_minus: jax.Array,
    h_plus_p: jax.Array,
    h_minus_p: jax.Array,
    is_open: jax.Array,
    k: jax.Array,
) -> jax.Array:
    """Match one energy sample after the caller has supplied a concrete ``k`` array."""

    return _match_rmatrix(rmatrix, h_plus, h_minus, h_plus_p, h_minus_p, is_open, k)


def _project_open_channels(
    rmatrix: jax.Array,
    h_plus: jax.Array,
    h_minus: jax.Array,
    h_plus_p: jax.Array,
    h_minus_p: jax.Array,
    is_open: jax.Array,
    k: jax.Array | None,
) -> tuple[jax.Array, BoundaryValues]:
    """Project the decoupled R-matrix and boundary values onto the open-channel subspace.

    Closed-channel rows and columns of R are zeroed via an ``is_open`` mask,
    and the corresponding Hankel function entries are replaced by 1 in
    ``H_plus`` (to avoid divide-by-zero in the matching formula) and by 0 in
    ``H_minus``, ``H_plus_p``, and ``H_minus_p``.  The shapes remain
    ``(N_c, N_c)`` / ``(N_c,)`` so JAX sees static shapes inside JIT.

    Parameters
    ----------
    rmatrix
        Full channel-space R-matrix, shape ``(N_c, N_c)``.
    h_plus, h_minus, h_plus_p, h_minus_p
        Boundary value arrays, shape ``(N_c,)``, complex.
    is_open
        Boolean mask for open channels, shape ``(N_c,)``.
    k
        Wave numbers in fm⁻¹, shape ``(N_c,)``, or ``None``.

    Returns
    -------
    tuple[jax.Array, BoundaryValues]
        Masked R-matrix and masked boundary slice for use in
        :func:`smatrix_from_R`.
    """

    mask = is_open.astype(rmatrix.dtype)
    projected_r = rmatrix * mask[:, None] * mask[None, :]
    closed_dtype = h_plus.dtype
    ones: jax.Array = jnp.ones_like(
        h_plus,
        dtype=closed_dtype,
    )
    mask_complex = is_open.astype(closed_dtype)
    if k is None:
        k_values = None
    else:
        ones_k: jax.Array = jnp.ones_like(k, dtype=k.dtype)
        k_values = k * is_open.astype(k.dtype) + ones_k * (1 - is_open.astype(k.dtype))

    boundary_slice = BoundaryValues(
        H_plus=h_plus * mask_complex + ones * (1.0 - mask_complex),
        H_minus=h_minus * mask_complex,
        H_plus_p=h_plus_p * mask_complex,
        H_minus_p=h_minus_p * mask_complex,
        is_open=is_open,
        k=k_values,
    )
    return projected_r, boundary_slice


def _closed_channel_bloch(
    h_plus: jax.Array,
    h_plus_p: jax.Array,
    is_open: jax.Array,
) -> jax.Array:
    """Return the closed-channel Bloch boundary parameter `B_c = H'_c / H_c`."""

    ratio = h_plus_p / h_plus
    closed_mask = jnp.logical_not(is_open)
    zeros: jax.Array = jnp.zeros_like(ratio)
    return jnp.where(
        closed_mask,
        ratio,
        zeros,
    )


def _boundary_wave_numbers(boundary: BoundaryValues) -> jax.Array:
    """Return concrete wave numbers for matching, defaulting closed tests to one."""

    if boundary.k is not None:
        return boundary.k
    return jnp.ones_like(
        boundary.is_open,
        dtype=jnp.float64,
    )


def _uniform_mass_factor(channels: tuple[ChannelSpec, ...]) -> float:
    """Return the shared mass factor expected by the MVP observables."""

    mass_factor = channels[0].mass_factor
    for channel in channels[1:]:
        if channel.mass_factor != mass_factor:
            msg = "The MVP solver path requires a uniform mass_factor across channels."
            raise ValueError(msg)
    return mass_factor


__all__ = [
    "bind_direct_grid_observables",
    "bind_grid_observables",
    "bind_interpolators",
    "bind_observables",
]

"""Pickle-safe observable wrappers bound to compiled solver caches.

The public ``Solver`` bundle stores module-level callable objects rather than local
closures so it can round-trip through stdlib ``pickle``. This module provides those
wrappers plus the helper functions that bind compile-time mesh, energy-grid, and
boundary data to the runtime spectral formulas from ``lax.spectral``.

Every observable is batched over the leading symmetry-block axis (DESIGN.md
§15.5).  A solver compiled with ``channels=`` is the ``N_b == 1`` case: its
(unbatched) :class:`Spectrum` is lifted onto a length-1 block axis, the batched
formula runs, and the block axis is squeezed off the result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NoReturn, cast

import jax
import jax.numpy as jnp

from lax.spectral.matching import open_channel_smatrix_from_R, phases_from_S
from lax.spectral.observables import (
    greens_from_spectrum,
    rmatrix_from_spectrum,
    wavefunction_grid_from_spectrum,
    wavefunction_internal_from_spectrum,
)
from lax.spectral.types import Spectrum
from lax.types import (
    BoundaryValues,
    ChannelSpec,
    EigenpairAccessor,
    GreenFunctionObservable,
    Mesh,
    RMatrixObservable,
    SpectrumGridObservable,
    SpectrumObservable,
    WavefunctionGridObservable,
    WavefunctionObservable,
)

from .assembly import add_block_axis, uniform_block_mass_factor


@dataclass(frozen=True)
class _NonuniformMassFactorStub:
    """Raising stand-in for static-regime observables on a non-uniform μ(E) solver.

    With a non-uniform ``mass_factor_grid`` the dimensionless Hamiltonian
    ``H/μ(E)`` varies with energy, so a static-regime evaluation would mix
    per-energy boundary values with a single-μ spectral denominator — silently
    wrong matching.  The stub keeps the observable slot non-``None`` (feature
    detection, ``__repr__``, pickling all behave) while turning any call into
    a clear error pointing at the legitimate paths.
    """

    observable: str

    def __call__(self, *args: object, **kwargs: object) -> NoReturn:
        raise ValueError(
            f"solver.{self.observable} is unavailable: this solver was compiled with a "
            "non-uniform mass_factor_grid, so static-regime spectral evaluation would mix "
            "per-energy boundary values with a single-mu spectral denominator. "
            "solver.spectrum(V) returns an energy-batched Spectrum on this solver - use "
            "the aligned-grid observables (rmatrix_grid / smatrix_grid / phases_grid / "
            "wavefunction_grid) or the direct path (method='linear_solve')."
        )


@dataclass(frozen=True)
class _RMatrixObservable:
    """Pickle-safe spectral R-matrix observable."""

    channel_radius: float
    mass_factor: float
    block_mode: bool

    def __call__(self, spectrum: Spectrum, energy: float | jax.Array) -> jax.Array:
        """Evaluate the R-matrix at one energy.

        Shape ``(N_c, N_c)`` — ``(N_b, N_c, N_c)`` in blocks mode.
        """

        if not self.block_mode:
            spectrum = add_block_axis(spectrum)
        result = cast(
            jax.Array,
            _RMATRIX_BLOCKS_JIT(spectrum, energy, self.channel_radius, self.mass_factor),
        )
        return result if self.block_mode else result[0]


@dataclass(frozen=True)
class _SMatrixObservable:
    """Pickle-safe S-matrix observable on the compile-time energy grid.

    ``boundary`` is always stacked — ``(N_b, N_E, N_c)`` fields, ``N_b == 1``
    for a ``channels=`` solver.
    """

    energies: jax.Array
    boundary: BoundaryValues
    channel_radius: float
    mass_factor: float
    block_mode: bool

    def __call__(self, spectrum: Spectrum) -> jax.Array:
        """Evaluate the S-matrix on the compile-time energy grid.

        Shape ``(N_E, N_c, N_c)`` — ``(N_b, N_E, N_c, N_c)`` in blocks mode.
        """

        if not self.block_mode:
            spectrum = add_block_axis(spectrum)
        result = cast(
            jax.Array,
            _SMATRIX_BLOCKS_JIT(
                spectrum,
                self.energies,
                self.boundary,
                self.channel_radius,
                self.mass_factor,
            ),
        )
        return result if self.block_mode else result[0]


@dataclass(frozen=True)
class _PhasesObservable:
    """Pickle-safe phase-shift observable on the compile-time energy grid.

    ``boundary`` is always stacked; see :class:`_SMatrixObservable`.
    """

    energies: jax.Array
    boundary: BoundaryValues
    channel_radius: float
    mass_factor: float
    block_mode: bool

    def __call__(self, spectrum: Spectrum) -> jax.Array:
        """Evaluate the phase shifts on the compile-time energy grid.

        Returns
        -------
        jax.Array
            Shape ``(N_E, N_c)`` in radians — ``(N_b, N_E, N_c)`` in blocks
            mode.  For a single-channel solver use ``result[:, 0]`` to obtain
            a 1-D energy curve.
        """

        if not self.block_mode:
            spectrum = add_block_axis(spectrum)
        result = cast(
            jax.Array,
            _PHASES_BLOCKS_JIT(
                spectrum,
                self.energies,
                self.boundary,
                self.channel_radius,
                self.mass_factor,
            ),
        )
        return result if self.block_mode else result[0]


@dataclass(frozen=True)
class _GreenFunctionObservable:
    """Pickle-safe Green's-function observable."""

    mass_factor: float
    block_mode: bool

    def __call__(self, spectrum: Spectrum, energy: float | jax.Array) -> jax.Array:
        """Evaluate the Green's function at one energy.

        Shape ``(M, M)`` — ``(N_b, M, M)`` in blocks mode.
        """

        if not self.block_mode:
            spectrum = add_block_axis(spectrum)
        result = cast(jax.Array, _GREENS_BLOCKS_JIT(spectrum, energy, self.mass_factor))
        return result if self.block_mode else result[0]


@dataclass(frozen=True)
class _WavefunctionObservable:
    """Pickle-safe internal wavefunction observable."""

    mass_factor: float
    n_blocks: int
    block_mode: bool

    def __call__(
        self, spectrum: Spectrum, energy: float | jax.Array, source: jax.Array
    ) -> jax.Array:
        """Evaluate the internal wavefunction at one energy.

        ``source`` is ``(M,)`` — in blocks mode also ``(N_b, M)`` for
        per-block sources.  Shape ``(M,)`` — ``(N_b, M)`` in blocks mode.
        """

        if not self.block_mode:
            spectrum = add_block_axis(spectrum)
        sources = (
            jnp.broadcast_to(source, (self.n_blocks, *source.shape)) if source.ndim == 1 else source
        )
        result = cast(
            jax.Array,
            _WAVEFUNCTION_BLOCKS_JIT(spectrum, energy, sources, self.mass_factor),
        )
        return result if self.block_mode else result[0]


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
    block_mode: bool

    def __call__(self, spectra: Spectrum) -> jax.Array:
        """Evaluate `R(E_i; spec_i)` across the compile-time energy grid."""

        if not self.block_mode:
            spectra = add_block_axis(spectra)
        result = cast(
            jax.Array,
            _RMATRIX_GRID_BLOCKS_JIT(
                spectra,
                self.energies,
                self.channel_radius,
                self.mass_factor_grid,
            ),
        )
        return result if self.block_mode else result[0]


@dataclass(frozen=True)
class _SMatrixGridObservable:
    """Pickle-safe aligned-grid spectral S-matrix observable.

    ``boundary`` is always stacked; see :class:`_SMatrixObservable`.
    """

    energies: jax.Array
    boundary: BoundaryValues
    channel_radius: float
    mass_factor_grid: jax.Array
    block_mode: bool

    def __call__(self, spectra: Spectrum) -> jax.Array:
        """Evaluate `S(E_i; spec_i)` across the compile-time energy grid."""

        if not self.block_mode:
            spectra = add_block_axis(spectra)
        result = cast(
            jax.Array,
            _SMATRIX_GRID_BLOCKS_JIT(
                spectra,
                self.energies,
                self.boundary,
                self.channel_radius,
                self.mass_factor_grid,
            ),
        )
        return result if self.block_mode else result[0]


@dataclass(frozen=True)
class _PhasesGridObservable:
    """Pickle-safe aligned-grid spectral phase-shift observable.

    ``boundary`` is always stacked; see :class:`_SMatrixObservable`.
    """

    energies: jax.Array
    boundary: BoundaryValues
    channel_radius: float
    mass_factor_grid: jax.Array
    block_mode: bool

    def __call__(self, spectra: Spectrum) -> jax.Array:
        """Evaluate `δ(E_i; spec_i)` across the compile-time energy grid."""

        if not self.block_mode:
            spectra = add_block_axis(spectra)
        result = cast(
            jax.Array,
            _PHASES_GRID_BLOCKS_JIT(
                spectra,
                self.energies,
                self.boundary,
                self.channel_radius,
                self.mass_factor_grid,
            ),
        )
        return result if self.block_mode else result[0]


def bind_observables(
    mesh: Mesh,
    blocks: tuple[tuple[ChannelSpec, ...], ...],
    energies: jax.Array,
    boundary: BoundaryValues | None,
    block_mode: bool = False,
    mass_factor_nonuniform: bool = False,
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
    blocks
        ``N_b`` symmetry blocks of ``N_c`` channels each.  A ``channels=``
        compile passes the single block ``(channels,)``.
    energies
        Compile-time energy grid. It is stored on matching-dependent observables even
        though pointwise R and Green's functions can be evaluated at arbitrary energy.
    boundary
        Compile-time boundary values for matching-dependent observables —
        mode-appropriate shape (stacked iff ``block_mode``). When absent, only
        spectrum-only observables are returned.
    block_mode
        ``True`` for a ``blocks=`` compile: outputs keep the leading
        ``(N_b,)`` axis.
    mass_factor_nonuniform
        ``True`` when the solver was compiled with a non-uniform
        ``mass_factor_grid``.  The five static-regime observables are then
        bound to :class:`_NonuniformMassFactorStub` instances that raise on
        call — the aligned-grid observables and the direct path are the
        legitimate μ(E) routes.  Raw eigenpair access stays live.

    Returns
    -------
    tuple
        Bound runtime entry points for the R-matrix, S-matrix, phase shifts, Green's
        function, internal wavefunction, and raw eigensystem access.
    """

    if mass_factor_nonuniform:
        return (
            _NonuniformMassFactorStub(observable="rmatrix"),
            _NonuniformMassFactorStub(observable="smatrix") if boundary is not None else None,
            _NonuniformMassFactorStub(observable="phases") if boundary is not None else None,
            _NonuniformMassFactorStub(observable="greens"),
            _NonuniformMassFactorStub(observable="wavefunction"),
            _EigenpairAccessor(),
        )

    channel_radius = mesh.scale
    mass_factor = uniform_block_mass_factor(blocks, context="spectral observable path")
    rmatrix = _RMatrixObservable(
        channel_radius=channel_radius, mass_factor=mass_factor, block_mode=block_mode
    )
    smatrix: SpectrumObservable | None = None
    phases: SpectrumObservable | None = None
    if boundary is not None:
        boundary_blocks = boundary if block_mode else add_block_axis(boundary)
        smatrix = _SMatrixObservable(
            energies=energies,
            boundary=boundary_blocks,
            channel_radius=channel_radius,
            mass_factor=mass_factor,
            block_mode=block_mode,
        )
        phases = _PhasesObservable(
            energies=energies,
            boundary=boundary_blocks,
            channel_radius=channel_radius,
            mass_factor=mass_factor,
            block_mode=block_mode,
        )
    greens = _GreenFunctionObservable(mass_factor=mass_factor, block_mode=block_mode)
    wavefunction = _WavefunctionObservable(
        mass_factor=mass_factor, n_blocks=len(blocks), block_mode=block_mode
    )
    eigh = _EigenpairAccessor()

    return rmatrix, smatrix, phases, greens, wavefunction, eigh


def bind_grid_observables(
    mesh: Mesh,
    blocks: tuple[tuple[ChannelSpec, ...], ...],
    energies: jax.Array,
    boundary: BoundaryValues | None,
    mass_factor_grid: jax.Array | None = None,
    block_mode: bool = False,
) -> tuple[SpectrumGridObservable, SpectrumGridObservable | None, SpectrumGridObservable | None]:
    """Bind aligned-grid spectral observables for energy-dependent workflows.

    Parameters
    ----------
    mesh
        Compiled mesh cache.
    blocks
        ``N_b`` symmetry blocks of ``N_c`` channels each.
    energies
        Compile-time energy grid aligned with the batched spectra.
    boundary
        Compile-time matching data — mode-appropriate shape (stacked iff
        ``block_mode``). When absent, only the aligned-grid R-matrix
        observable is returned.
    mass_factor_grid
        Per-energy ℏ²/2μ values in MeV·fm², shape ``(N_E,)``, or ``None``
        for a constant mass factor.  When provided the spectral denominators
        ``ε_k − E_i/μ(E_i)`` use the correct per-energy value.
    block_mode
        ``True`` for a ``blocks=`` compile: outputs keep the leading
        ``(N_b,)`` axis.

    Returns
    -------
    tuple
        Bound aligned-grid observables for ``R(E_i; spec_i)``, ``S(E_i; spec_i)``,
        and ``δ(E_i; spec_i)``.
    """

    channel_radius = mesh.scale
    mass_factor = uniform_block_mass_factor(blocks, context="spectral observable path")
    _mfg = (
        jnp.full(len(energies), mass_factor)
        if mass_factor_grid is None
        else jnp.asarray(mass_factor_grid)
    )
    rmatrix_grid = _RMatrixGridObservable(
        energies=energies,
        channel_radius=channel_radius,
        mass_factor_grid=_mfg,
        block_mode=block_mode,
    )
    smatrix_grid: SpectrumGridObservable | None = None
    phases_grid: SpectrumGridObservable | None = None
    if boundary is not None:
        boundary_blocks = boundary if block_mode else add_block_axis(boundary)
        smatrix_grid = _SMatrixGridObservable(
            energies=energies,
            boundary=boundary_blocks,
            channel_radius=channel_radius,
            mass_factor_grid=_mfg,
            block_mode=block_mode,
        )
        phases_grid = _PhasesGridObservable(
            energies=energies,
            boundary=boundary_blocks,
            channel_radius=channel_radius,
            mass_factor_grid=_mfg,
            block_mode=block_mode,
        )
    return rmatrix_grid, smatrix_grid, phases_grid


def spectrum_is_energy_batched(spectrum: Spectrum) -> bool:
    """Return whether a block-lifted ``Spectrum`` carries an energy batch axis.

    Apply *after* :func:`add_block_axis` so one rule serves both compile
    modes: block-lifted eigenvalues are ``(N_b, M)`` for a static-V spectrum
    and ``(N_b, N_E, M)`` for the energy-batched regime.  Ranks are static
    under ``jax.jit``.
    """

    return spectrum.eigenvalues.ndim >= 3


@dataclass(frozen=True)
class _WavefunctionGridObservable:
    """Pickle-safe wavefunction-grid observable, serving both evaluation regimes.

    The static-V regime ("diagonalize once, evaluate the grid") runs the fused
    einsum of :func:`wavefunction_grid_from_spectrum`; the energy-batched
    regime vmaps the per-energy Green's contraction over the aligned
    ``(spectra_e, E_e, source_e, μ_e)`` axes — the ``smatrix_grid`` pattern.
    The regime is detected from the ``Spectrum`` rank.
    """

    sources: jax.Array  # (N_b, N_E, N_c, M) — always block-stacked
    energies: jax.Array  # (N_E,)
    mass_factor: float
    mass_factor_grid: jax.Array  # (N_E,)
    n_energies: int
    block_mode: bool
    mass_factor_nonuniform: bool

    def __call__(self, spectrum: Spectrum, channel_index: int | None = 0) -> jax.Array:
        """Evaluate ψ for every ``(block, grid-energy)`` pair; see the protocol."""

        if spectrum.eigenvectors is None:
            raise RuntimeError(
                "Eigenvectors were not retained at compile time. "
                "Re-compile with 'greens' or 'wavefunction' in solvers= to keep them."
            )
        if not self.block_mode:
            spectrum = add_block_axis(spectrum)
        batched = spectrum_is_energy_batched(spectrum)
        if batched and spectrum.eigenvalues.shape[1] != self.n_energies:
            msg = (
                f"energy-batched Spectrum carries {spectrum.eigenvalues.shape[1]} energies "
                f"but the compile-time grid has N_E = {self.n_energies}."
            )
            raise ValueError(msg)
        sources = (
            self.sources
            if channel_index is None
            else self.sources[:, :, channel_index : channel_index + 1, :]
        )
        if batched:
            result = cast(
                jax.Array,
                _WAVEFUNCTION_GRID_ALIGNED_BLOCKS_JIT(
                    spectrum, self.energies, sources, self.mass_factor_grid
                ),
            )
        else:
            if self.mass_factor_nonuniform:
                raise ValueError(
                    "wavefunction_grid received a static-V Spectrum, but this solver was "
                    "compiled with a non-uniform mass_factor_grid: H/mu(E) varies with "
                    "energy, so diagonalize-once is invalid. Pass the energy-batched "
                    "Spectrum from this solver's spectrum(V) (it dispatches per energy), "
                    "or use wavefunction_direct_grid (method='linear_solve')."
                )
            result = cast(
                jax.Array,
                _WAVEFUNCTION_GRID_STATIC_BLOCKS_JIT(
                    spectrum, self.energies, sources, self.mass_factor
                ),
            )
        if channel_index is not None:
            result = result[:, :, 0, :]
        return result if self.block_mode else result[0]


def bind_wavefunction_grid_observable(
    blocks: tuple[tuple[ChannelSpec, ...], ...],
    energies: jax.Array,
    sources: jax.Array,
    mass_factor_grid: jax.Array | None = None,
    block_mode: bool = False,
    mass_factor_nonuniform: bool = False,
) -> WavefunctionGridObservable:
    """Bind the spectral wavefunction-grid observable to one compiled solver.

    Parameters
    ----------
    blocks
        ``N_b`` symmetry blocks of ``N_c`` channels each.
    energies
        Compile-time energy grid, shape ``(N_E,)``.
    sources
        Block-stacked baked source stack, shape ``(N_b, N_E, N_c, M)``.
    mass_factor_grid
        Reduced ``(N_E,)`` per-energy ℏ²/2μ, or ``None`` for a constant μ.
    block_mode
        ``True`` for a ``blocks=`` compile.
    mass_factor_nonuniform
        ``True`` when μ genuinely varies with energy — the static regime then
        raises (C4) and only the energy-batched regime is served.
    """

    mass_factor = uniform_block_mass_factor(blocks, context="spectral observable path")
    _mfg = (
        jnp.full(len(energies), mass_factor)
        if mass_factor_grid is None
        else jnp.asarray(mass_factor_grid)
    )
    return _WavefunctionGridObservable(
        sources=sources,
        energies=energies,
        mass_factor=mass_factor,
        mass_factor_grid=_mfg,
        n_energies=len(energies),
        block_mode=block_mode,
        mass_factor_nonuniform=mass_factor_nonuniform,
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
        boundary.k,
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
        boundary.k,
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


# --------------------------------------------------------------------------
# Symmetry-block batched layers (DESIGN.md §15.5): thin jax.vmap wrappers over
# the single-block formulas above.


def _rmatrix_blocks(
    spectra: Spectrum,
    energy: float | jax.Array,
    channel_radius: float,
    mass_factor: float,
) -> jax.Array:
    """Return per-block spectral R-matrices at one energy, shape ``(N_b, N_c, N_c)``."""

    def one_block(spectrum: Spectrum) -> jax.Array:
        return _rmatrix(spectrum, energy, channel_radius, mass_factor)

    return jax.vmap(one_block)(spectra)


def _smatrix_blocks(
    spectra: Spectrum,
    energies: jax.Array,
    boundary: BoundaryValues,
    channel_radius: float,
    mass_factor: float,
) -> jax.Array:
    """Return per-block S-matrices, shape ``(N_b, N_E, N_c, N_c)``.

    The block-batched :class:`Spectrum` and the ``(N_b,)``-stacked
    :class:`BoundaryValues` pytrees vmap jointly over the block axis.
    """

    def one_block(spectrum: Spectrum, boundary_b: BoundaryValues) -> jax.Array:
        return _smatrix(spectrum, energies, boundary_b, channel_radius, mass_factor)

    return jax.vmap(one_block)(spectra, boundary)


def _phases_blocks(
    spectra: Spectrum,
    energies: jax.Array,
    boundary: BoundaryValues,
    channel_radius: float,
    mass_factor: float,
) -> jax.Array:
    """Return per-block phase shifts, shape ``(N_b, N_E, N_c)``."""

    def one_block(spectrum: Spectrum, boundary_b: BoundaryValues) -> jax.Array:
        return _phases(spectrum, energies, boundary_b, channel_radius, mass_factor)

    return jax.vmap(one_block)(spectra, boundary)


def _greens_blocks(
    spectra: Spectrum,
    energy: float | jax.Array,
    mass_factor: float,
) -> jax.Array:
    """Return per-block Green's functions at one energy, shape ``(N_b, M, M)``."""

    def one_block(spectrum: Spectrum) -> jax.Array:
        return _greens(spectrum, energy, mass_factor)

    return jax.vmap(one_block)(spectra)


def _wavefunction_blocks(
    spectra: Spectrum,
    energy: float | jax.Array,
    sources: jax.Array,
    mass_factor: float,
) -> jax.Array:
    """Return per-block internal wavefunctions at one energy, shape ``(N_b, M)``."""

    def one_block(spectrum: Spectrum, source: jax.Array) -> jax.Array:
        return _wavefunction(spectrum, energy, source, mass_factor)

    return jax.vmap(one_block)(spectra, sources)


def _wavefunction_grid_aligned(
    spectra: Spectrum,
    energies: jax.Array,
    sources: jax.Array,
    mass_factor_grid: jax.Array,
) -> jax.Array:
    """Return aligned-grid wavefunctions ``ψ(E_i; spec_i)``, shape ``(N_E, K, M)``.

    The energy-batched regime: one Green's contraction per aligned
    ``(spectrum_e, E_e, source_e, μ_e)`` tuple — the ``_smatrix_grid``
    pattern.
    """

    def one_energy(
        spectrum: Spectrum, energy: jax.Array, source: jax.Array, mu: jax.Array
    ) -> jax.Array:
        greens = greens_from_spectrum(spectrum, energy=energy, mass_factor=mu)
        return jnp.einsum("mn,cn->cm", greens, source)

    return jax.vmap(one_energy)(spectra, energies, sources, mass_factor_grid)


def _wavefunction_grid_static_blocks(
    spectra: Spectrum,
    energies: jax.Array,
    sources: jax.Array,
    mass_factor: float,
) -> jax.Array:
    """Return per-block static-regime grid wavefunctions, shape ``(N_b, N_E, K, M)``."""

    def one_block(spectrum: Spectrum, source_stack: jax.Array) -> jax.Array:
        return wavefunction_grid_from_spectrum(spectrum, energies, source_stack, mass_factor)

    return jax.vmap(one_block)(spectra, sources)


def _wavefunction_grid_aligned_blocks(
    spectra: Spectrum,
    energies: jax.Array,
    sources: jax.Array,
    mass_factor_grid: jax.Array,
) -> jax.Array:
    """Return per-block aligned-grid wavefunctions, shape ``(N_b, N_E, K, M)``."""

    def one_block(spectra_b: Spectrum, source_stack: jax.Array) -> jax.Array:
        return _wavefunction_grid_aligned(spectra_b, energies, source_stack, mass_factor_grid)

    return jax.vmap(one_block)(spectra, sources)


def _rmatrix_grid_blocks(
    spectra: Spectrum,
    energies: jax.Array,
    channel_radius: float,
    mass_factor_grid: jax.Array,
) -> jax.Array:
    """Return per-block aligned-grid R samples, shape ``(N_b, N_E, N_c, N_c)``."""

    def one_block(spectra_b: Spectrum) -> jax.Array:
        return _rmatrix_grid(spectra_b, energies, channel_radius, mass_factor_grid)

    return jax.vmap(one_block)(spectra)


def _smatrix_grid_blocks(
    spectra: Spectrum,
    energies: jax.Array,
    boundary: BoundaryValues,
    channel_radius: float,
    mass_factor_grid: jax.Array,
) -> jax.Array:
    """Return per-block aligned-grid S samples, shape ``(N_b, N_E, N_c, N_c)``."""

    def one_block(spectra_b: Spectrum, boundary_b: BoundaryValues) -> jax.Array:
        return _smatrix_grid(spectra_b, energies, boundary_b, channel_radius, mass_factor_grid)

    return jax.vmap(one_block)(spectra, boundary)


def _phases_grid_blocks(
    spectra: Spectrum,
    energies: jax.Array,
    boundary: BoundaryValues,
    channel_radius: float,
    mass_factor_grid: jax.Array,
) -> jax.Array:
    """Return per-block aligned-grid phase shifts, shape ``(N_b, N_E, N_c)``."""

    def one_block(spectra_b: Spectrum, boundary_b: BoundaryValues) -> jax.Array:
        return _phases_grid(spectra_b, energies, boundary_b, channel_radius, mass_factor_grid)

    return jax.vmap(one_block)(spectra, boundary)


_EIGH_JIT = jax.jit(_eigh)
_RMATRIX_BLOCKS_JIT = jax.jit(
    _rmatrix_blocks,
    static_argnames=("channel_radius", "mass_factor"),
)
_SMATRIX_BLOCKS_JIT = jax.jit(
    _smatrix_blocks,
    static_argnames=("channel_radius", "mass_factor"),
)
_PHASES_BLOCKS_JIT = jax.jit(
    _phases_blocks,
    static_argnames=("channel_radius", "mass_factor"),
)
_GREENS_BLOCKS_JIT = jax.jit(
    _greens_blocks,
    static_argnames=("mass_factor",),
)
_WAVEFUNCTION_BLOCKS_JIT = jax.jit(
    _wavefunction_blocks,
    static_argnames=("mass_factor",),
)
_RMATRIX_GRID_BLOCKS_JIT = jax.jit(
    _rmatrix_grid_blocks,
    static_argnames=("channel_radius",),
)
_SMATRIX_GRID_BLOCKS_JIT = jax.jit(
    _smatrix_grid_blocks,
    static_argnames=("channel_radius",),
)
_PHASES_GRID_BLOCKS_JIT = jax.jit(
    _phases_grid_blocks,
    static_argnames=("channel_radius",),
)
_WAVEFUNCTION_GRID_STATIC_BLOCKS_JIT = jax.jit(
    _wavefunction_grid_static_blocks,
    static_argnames=("mass_factor",),
)
_WAVEFUNCTION_GRID_ALIGNED_BLOCKS_JIT = jax.jit(_wavefunction_grid_aligned_blocks)


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


__all__ = [
    "bind_grid_observables",
    "bind_observables",
]

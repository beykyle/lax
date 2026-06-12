"""Helpers for constructing scattering wavefunctions from a compiled solver.

The core observable is the *internal* wavefunction on the mesh, obtained by
contracting the Green's function with a source term:

    ψ_int = G(E) · source

where ``source`` encodes which incoming channel drives the reaction.  This
module provides :func:`make_wavefunction_source` (one ``(channel, energy)``
pair) and :func:`make_wavefunction_source_grid` (the full compile-time stack),
both slices of one shared builder so the two can never drift apart.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from lax.types import BoundaryValues, Mesh, Solver


def build_wavefunction_sources(
    mesh: Mesh,
    boundary: BoundaryValues,
    n_channels: int,
) -> jax.Array:
    """Build the full Descouvemont eq.-27 source stack from the boundary cache.

    ``sources[..., e, c, c·N:(c+1)·N] = φ_n(a) · H⁻[..., e, c]`` with every
    other channel block zero — the boundary cache already holds ``H⁻`` for
    every ``(energy, channel)`` pair, so the stack is fully determined at
    compile time.

    Closed channels are **not** masked: where ``boundary.is_open`` is false,
    ``H⁻`` is the Whittaker-decaying solution and the resulting source (and
    wavefunction) is mathematically defined but is *not* a scattering
    wavefunction.  Callers slice by ``boundary.is_open`` if they need
    open-channel-only results.

    Parameters
    ----------
    mesh
        Compiled single-interval mesh (supplies ``basis_at_boundary``).
    boundary
        Compile-time boundary values; ``H_minus`` has shape ``(N_E, N_c)`` —
        ``(N_b, N_E, N_c)`` in blocks mode.
    n_channels
        ``N_c``, the channels per block.

    Returns
    -------
    jax.Array
        Complex source stack of shape ``(N_E, N_c, N_c·N)`` —
        ``(N_b, N_E, N_c, N_c·N)`` in blocks mode — indexed
        ``[..., energy, incoming_channel, coefficient]``.
    """

    phi_a = mesh.basis_at_boundary  # (N,)
    n = mesh.n
    h_minus = boundary.H_minus  # (..., N_E, N_c), complex
    sources = jnp.zeros((*h_minus.shape, n_channels * n), dtype=h_minus.dtype)
    for c in range(n_channels):
        start = c * n
        sources = sources.at[..., c, start : start + n].set(phi_a * h_minus[..., c][..., None])
    return sources


def _resolved_sources(solver: Solver, caller: str) -> jax.Array:
    """Return the solver's baked source stack, building it on demand."""

    boundary: BoundaryValues | None = solver.boundary
    if boundary is None:
        msg = f"solver.boundary is None — re-compile with an energy grid to use {caller}."
        raise ValueError(msg)
    if solver.wavefunction_sources is not None:
        return solver.wavefunction_sources
    if solver.mesh.propagation is not None:
        msg = (
            "wavefunction sources are not supported on propagated multi-interval "
            "meshes — the boundary basis differs per interval. Use a "
            "single-interval mesh."
        )
        raise NotImplementedError(msg)
    return build_wavefunction_sources(solver.mesh, boundary, len(solver.channels))


def make_wavefunction_source(
    solver: Solver,
    channel_index: int,
    energy_index: int,
) -> jnp.ndarray:
    """Build the mesh-space source vector for one incoming channel.

    The source term drives the internal Green's function to produce the
    scattering wavefunction for a reaction incoming in channel ``c`` at
    energy ``E_i``.  Following Descouvemont [2] eq. 27, the source is:

    .. code-block:: text

        source[c·N : (c+1)·N] = φ_n(a) · H⁻_c(E_i)   (all other blocks zero)

    where ``φ_n(a)`` are the Lagrange-basis boundary values and ``H⁻`` is
    the incoming Coulomb/Whittaker function at the channel radius.  This is a
    slice of the compile-time stack built by
    :func:`build_wavefunction_sources` (baked on the solver when a
    wavefunction entry point was requested, rebuilt on demand otherwise).

    Parameters
    ----------
    solver
        Compiled solver bundle.  Must have been built with an energy grid
        (so ``solver.boundary`` is not ``None``).
    channel_index
        Index ``c`` of the incoming channel (0-based).
    energy_index
        Index into the compile-time energy grid (0-based).

    Returns
    -------
    jnp.ndarray
        Source vector of shape ``(N_c · N,)``, where ``N = solver.mesh.n``
        and ``N_c = len(solver.channels)``.  For a solver compiled with
        ``blocks=`` (DESIGN.md §15.5) the per-block sources are stacked on a
        leading block axis — shape ``(N_b, N_c · N)`` — matching the input
        expected by ``solver.wavefunction_direct`` in blocks mode.

    Raises
    ------
    ValueError
        If ``solver.boundary`` is ``None`` (no energy grid was compiled).

    Examples
    --------
    >>> import lax, lax.constants as C, jax.numpy as jnp
    >>> HBAR2_2MU = C.hbar2_over_2mu(1.008665, 1.008665)
    >>> energies  = jnp.linspace(1.0, 10.0, 20)
    >>> solver = lax.compile(
    ...     mesh=lax.MeshSpec("legendre", "x", n=20, scale=8.0),
    ...     channels=(lax.ChannelSpec(l=0, threshold=0.0, mass_factor=HBAR2_2MU),),
    ...     solvers=("spectrum", "wavefunction"),
    ...     energies=energies,
    ... )
    >>> V = solver.nonlocal_potential(lambda r1, r2: jnp.zeros_like(r1))
    >>> spec = solver.spectrum(V)
    >>> src  = lax.make_wavefunction_source(solver, channel_index=0, energy_index=5)
    >>> psi  = solver.wavefunction(spec, energies[5], src)

    For the direct (linear-solve) path — no eigendecomposition required — compile
    with ``solvers=("rmatrix_direct",)`` and use::

        interaction = solver.interaction_from_block(V[0, 0])  # (M, M) block
        psi = solver.wavefunction_direct(interaction, src, energy_index=5)
    """

    sources = _resolved_sources(solver, "make_wavefunction_source")
    if solver.blocks is not None:
        return sources[:, energy_index, channel_index]
    return sources[energy_index, channel_index]


def make_wavefunction_source_grid(
    solver: Solver,
    channel_index: int | None = None,
) -> jnp.ndarray:
    """Return the full compile-time source stack, or one incoming channel of it.

    Parameters
    ----------
    solver
        Compiled solver bundle with an energy grid.
    channel_index
        ``None`` (default) returns the all-channels stack; an ``int`` slices
        the corresponding incoming channel.

    Returns
    -------
    jnp.ndarray
        ``channel_index=None``: shape ``(N_E, N_c, N_c·N)`` —
        ``(N_b, N_E, N_c, N_c·N)`` in blocks mode.
        With an ``int`` ``channel_index`` the incoming-channel axis is sliced
        off: ``(N_E, N_c·N)`` / ``(N_b, N_E, N_c·N)``.

    Raises
    ------
    ValueError
        If ``solver.boundary`` is ``None`` (no energy grid was compiled).
    """

    sources = _resolved_sources(solver, "make_wavefunction_source_grid")
    if channel_index is None:
        return sources
    if solver.blocks is not None:
        return sources[:, :, channel_index]
    return sources[:, channel_index]


__all__ = [
    "build_wavefunction_sources",
    "make_wavefunction_source",
    "make_wavefunction_source_grid",
]

"""Helpers for constructing scattering wavefunctions from a compiled solver.

The core observable is the *internal* wavefunction on the mesh, obtained by
contracting the Green's function with a source term:

    ψ_int = G(E) · source

where ``source`` encodes which incoming channel drives the reaction.  This
module provides :func:`make_wavefunction_source` so users do not need to
construct the source vector by hand.
"""

from __future__ import annotations

import jax.numpy as jnp

from lax.types import BoundaryValues, Mesh, Solver


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
    the incoming Coulomb/Whittaker function at the channel radius.

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
    boundary: BoundaryValues | None = solver.boundary
    if boundary is None:
        raise ValueError(
            "solver.boundary is None — re-compile with an energy grid "
            "to use make_wavefunction_source."
        )

    mesh: Mesh = solver.mesh
    n_c = len(solver.channels)
    n = mesh.n

    # φ_n(a) for all basis functions  (N,)
    phi_a = mesh.basis_at_boundary

    if solver.blocks is not None:
        # Blocks mode: H⁻ carries a leading (N_b,) axis; build one source per
        # symmetry block, stacked on that axis.
        h_minus_b = boundary.H_minus[:, energy_index, channel_index]  # (N_b,)
        n_b = h_minus_b.shape[0]
        sources = jnp.zeros((n_b, n_c * n), dtype=jnp.complex128)
        start = channel_index * n
        return sources.at[:, start : start + n].set(phi_a[None] * h_minus_b[:, None])

    # H⁻ for the requested channel at the requested energy  (scalar, complex)
    h_minus_c = boundary.H_minus[energy_index, channel_index]

    # Build the source vector: non-zero only in the block for channel_index
    source = jnp.zeros(n_c * n, dtype=jnp.complex128)
    start = channel_index * n
    source = source.at[start : start + n].set(phi_a * h_minus_c)
    return source


__all__ = ["make_wavefunction_source"]

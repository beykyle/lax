"""Mesh-independent spectral types. See DESIGN.md §10.1.

Holds the two pure-data pytrees the spectral submodule operates on:
:class:`Spectrum` (eigenpairs + surface amplitudes) and
:class:`BoundaryValues` (Coulomb/Whittaker matching data). Both depend only
on JAX, keeping ``lax.spectral`` independent of the rest of the package.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class BoundaryValues:
    """Coulomb and Whittaker boundary values at the channel radius.

    Precomputed at compile time by ``mpmath`` for every ``(energy, channel)``
    pair.  Open channels use Coulomb Hankel functions; closed channels use
    Whittaker functions that decay exponentially into the barrier.

    Attributes
    ----------
    H_plus
        Outgoing Coulomb Hankel function ``H⁺ = G + iF`` at ``r = a``,
        shape ``(N_E, N_c)``, complex.
    H_minus
        Incoming Coulomb Hankel function ``H⁻ = G - iF`` at ``r = a``,
        shape ``(N_E, N_c)``, complex.
    H_plus_p
        ``ρ · d/dρ H⁺`` evaluated at ``ρ = ka``,
        shape ``(N_E, N_c)``, complex.
    H_minus_p
        ``ρ · d/dρ H⁻`` evaluated at ``ρ = ka``,
        shape ``(N_E, N_c)``, complex.
    is_open
        Boolean mask: ``True`` for open channels (``E > E_threshold``),
        shape ``(N_E, N_c)``.
    k
        Channel wave numbers ``k_c(E)`` in fm⁻¹, shape ``(N_E, N_c)``.
    """

    H_plus: jax.Array
    H_minus: jax.Array
    H_plus_p: jax.Array
    H_minus_p: jax.Array
    is_open: jax.Array
    k: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Spectrum:
    """Spectral decomposition of the Bloch-augmented Hamiltonian.

    Produced by ``solver.spectrum(V)`` and consumed by all downstream
    observables.  One ``Spectrum`` supports R-matrix, Green's function,
    S-matrix, and phase-shift evaluation at arbitrary energies via spectral
    sums — no further linear-algebra calls are needed.

    Attributes
    ----------
    eigenvalues
        Eigenvalues ε_k of H in fm⁻².  Shape ``(M,)`` where ``M = N_c · N``.
        Real-valued for Hermitian H (real potential); complex for
        complex-symmetric H (optical/absorptive potential).
    surface_amplitudes
        Reduced-width amplitudes γ_kc = (U^T Q)_kc.  Shape ``(M, N_c)``.
        Sufficient on their own for R-matrix and S-matrix evaluation.
    eigenvectors
        Full eigenvector matrix U, shape ``(M, M)``.  ``None`` if neither
        ``'greens'`` nor ``'wavefunction'`` was requested at compile time.
    is_hermitian
        Static flag.  ``True`` when H is Hermitian (real V, ``method='eigh'``),
        routing downstream code to use conjugate-transpose instead of
        plain transpose in spectral sums.
    """

    eigenvalues: jax.Array
    surface_amplitudes: jax.Array
    eigenvectors: jax.Array | None
    is_hermitian: bool = field(metadata={"static": True})


__all__ = ["BoundaryValues", "Spectrum"]

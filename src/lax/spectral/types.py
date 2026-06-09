"""Mesh-independent spectral decomposition type. See DESIGN.md §10.1."""

from __future__ import annotations

from dataclasses import dataclass, field

import jax


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


__all__ = ["Spectrum"]

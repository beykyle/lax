"""Mesh-independent spectral decomposition type. See DESIGN.md §10.1."""

from __future__ import annotations

from dataclasses import dataclass, field

import jax


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Spectrum:
    """Spectral decomposition of the Bloch-augmented Hamiltonian."""

    eigenvalues: jax.Array
    surface_amplitudes: jax.Array
    eigenvectors: jax.Array | None
    is_hermitian: bool = field(metadata={"static": True})


__all__ = ["Spectrum"]

"""Public type aliases and user-facing specifications. See DESIGN.md §6."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

type MeshFamily = Literal["legendre", "laguerre", "hermite", "jacobi"]
type Regularization = Literal[
    "none",
    "x",
    "x^3/2",
    "x(1-x)",
    "sqrt(1-x^2)",
    "modified_x^2",
]
type Method = Literal["eigh", "eig", "linear_solve", "lanczos"]

# Backward-compatible aliases for internal signatures and existing sketches.
type MeshFamilyT = MeshFamily
type RegularizationT = Regularization
type MethodT = Method


def _empty_extras() -> dict[str, object]:
    """Return a typed empty mapping for mesh-specific extra options."""

    return {}


@dataclass(frozen=True)
class MeshSpec:
    """User-facing mesh specification passed to :func:`lax.compile`."""

    family: MeshFamily
    regularization: Regularization
    n: int
    scale: float
    extras: dict[str, object] = field(default_factory=_empty_extras)


@dataclass(frozen=True)
class ChannelSpec:
    """One scattering channel baked into the compiled solver structure."""

    l: int
    threshold: float
    mass_factor: float = 1.0


__all__ = [
    "ChannelSpec",
    "MeshFamily",
    "MeshFamilyT",
    "MeshSpec",
    "Method",
    "MethodT",
    "Regularization",
    "RegularizationT",
]

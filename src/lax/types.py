"""Public type aliases and user-facing specifications. See ``DESIGN.md §6``."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

type MeshFamily = Literal["legendre", "laguerre"]
type Regularization = Literal[
    "x",
    "x^3/2",
    "x(1-x)",
    "modified_x^2",
]
type Method = Literal["eigh", "eig", "linear_solve"]

# Backward-compatible aliases for internal signatures and existing sketches.
type MeshFamilyT = MeshFamily
type RegularizationT = Regularization
type MethodT = Method


def _empty_extras() -> dict[str, object]:
    """Return a typed empty mapping for mesh-specific extra options."""

    return {}


@dataclass(frozen=True)
class MeshSpec:
    """User-facing mesh specification passed to :func:`lax.compile`.

    Attributes
    ----------
    family
        Mesh family registered in :mod:`lax.meshes`. The current public API
        supports ``"legendre"`` and ``"laguerre"``.
    regularization
        Endpoint regularization used by the chosen family. The currently
        supported combinations are:

        - Legendre: ``"x"``, ``"x(1-x)"``, ``"x^3/2"``
        - Laguerre: ``"x"``, ``"modified_x^2"``
    n
        Number of mesh basis functions.
    scale
        Physical length scale for the mesh. For finite-interval meshes this is
        the channel radius; for semi-infinite meshes it is the radial scaling
        factor described in ``DESIGN.md``.
    extras
        Mesh-specific compile-time options forwarded to the registered mesh
        builder.
    """

    family: MeshFamily
    regularization: Regularization
    n: int
    scale: float
    extras: dict[str, object] = field(default_factory=_empty_extras)


@dataclass(frozen=True)
class ChannelSpec:
    """One scattering channel baked into the compiled solver structure.

    Attributes
    ----------
    l
        Orbital angular momentum for the channel.
    threshold
        Channel threshold in MeV. Assembly code converts it to fm^-2 using
        ``mass_factor``.
    mass_factor
        Conversion factor ``ℏ² / 2μ`` in MeV·fm².  Defaults to ``1.0``,
        which is physically meaningless for any real nucleus — always set
        this explicitly.  Use :func:`lax.constants.hbar2_over_2mu` to
        compute the correct value from particle masses in AMU, e.g.
        ``lax.constants.hbar2_over_2mu(1.008665, 1.008665)`` ≈ 41.47 MeV·fm²
        for nucleon–nucleon systems.
    """

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

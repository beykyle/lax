"""Public type aliases and user-facing specifications. See ``DESIGN.md §6``."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import jax

type MeshFamily = Literal["legendre", "laguerre"]
type Regularization = Literal[
    "x",
    "x^3/2",
    "x(1-x)",
    "modified_x^2",
]
type Method = Literal["eigh", "eig", "linear_solve"]


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
    mass_factor: float | jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Interaction:
    """Assembled coupled-channel potential block in MeV.

    block : (M, M) or (N_E, M, M)  where M = N_c·N
        Local terms on the per-channel diagonal, non-local terms as full
        Gauss-scaled blocks. Symmetric. Mass-independent — per-channel mass
        factors are applied by the solver, never folded into this block.
        Excludes kinetic, centrifugal, threshold, and energy terms.
    energy_dependent : bool (static)
        True iff ``block`` has a leading (N_E,) axis aligned with the
        compile-time energy grid.
    """

    block: jax.Array
    energy_dependent: bool = field(metadata={"static": True})

    def __add__(self, other: object) -> Interaction:
        """Combine two Interaction blocks by summing their potential contributions.

        Energy-independent + energy-independent → energy-independent.
        Any combination involving an energy-dependent term → energy-dependent,
        broadcasting ``(M, M)`` to ``(1, M, M)`` before adding.
        """
        if not isinstance(other, Interaction):
            return NotImplemented
        if not self.energy_dependent and not other.energy_dependent:
            return Interaction(block=self.block + other.block, energy_dependent=False)
        b1 = self.block if self.energy_dependent else self.block[None]
        b2 = other.block if other.energy_dependent else other.block[None]
        return Interaction(block=b1 + b2, energy_dependent=True)

    def __radd__(self, other: object) -> Interaction:
        if other == 0:
            return self
        return NotImplemented


__all__ = [
    "ChannelSpec",
    "Interaction",
    "MeshFamily",
    "MeshSpec",
    "Method",
    "Regularization",
]

"""Reid soft-core interaction helpers for the coupled ``n-p`` example."""

from __future__ import annotations

from typing import Final

import jax
import jax.numpy as jnp

from lax.constants import hbar2_over_2mu
from lax.types import ChannelSpec

NN_MASS_FACTOR: Final[float] = hbar2_over_2mu(1.008665, 1.008665)

_REID_MU: Final[float] = 0.7
_REID_H: Final[float] = 10.463


def reid_np_j1_channels() -> tuple[ChannelSpec, ...]:
    """Return the coupled ``^3S_1``-``^3D_1`` channel pair for ``J=1``.

    Returns
    -------
    tuple[ChannelSpec, ...]
        Two-channel layout for the standard Reid soft-core triplet example.
    """

    return (
        ChannelSpec(l=0, threshold=0.0, mass_factor=NN_MASS_FACTOR),
        ChannelSpec(l=2, threshold=0.0, mass_factor=NN_MASS_FACTOR),
    )


def reid_soft_core_triplet_components(
    radii: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return the Reid soft-core triplet components in MeV.

    Parameters
    ----------
    radii
        Radial grid in fm.

    Returns
    -------
    tuple[jax.Array, jax.Array, jax.Array]
        Central, tensor, and spin-orbit terms in MeV.
    """

    x = _REID_MU * radii
    e1 = jnp.exp(-x) / x
    e2 = jnp.exp(-2.0 * x) / x
    e4 = jnp.exp(-4.0 * x) / x
    e6 = jnp.exp(-6.0 * x) / x

    v_central = -_REID_H * e1 + 105.468 * e2 - 3187.8 * e4 + 9924.3 * e6
    v_tensor = (
        -_REID_H * (1.0 + 3.0 / x + 3.0 / x**2) * e1
        + (_REID_H * (12.0 / x + 3.0 / x**2) + 351.77) * e4
        - 1673.5 * e6
    )
    v_spin_orbit = 708.91 * e4 - 2713.1 * e6
    return v_central, v_tensor, v_spin_orbit


def reid_np_j1_potential(
    radii: jax.Array,
    channel_index: int,
    coupled_index: int,
) -> jax.Array:
    """Return the coupled Reid soft-core ``n-p`` potential in MeV.

    Parameters
    ----------
    radii
        Radial grid in fm.
    channel_index
        Bra-channel index. ``0`` selects ``^3S_1`` and ``1`` selects ``^3D_1``.
    coupled_index
        Ket-channel index. ``0`` selects ``^3S_1`` and ``1`` selects ``^3D_1``.

    Returns
    -------
    jax.Array
        One matrix element of the coupled local potential in MeV.
    """

    v_central, v_tensor, v_spin_orbit = reid_soft_core_triplet_components(radii)
    if channel_index == coupled_index == 0:
        return v_central
    if channel_index == coupled_index == 1:
        return v_central - 2.0 * v_tensor - 3.0 * v_spin_orbit
    return 2.0 * jnp.sqrt(2.0) * v_tensor


__all__ = [
    "NN_MASS_FACTOR",
    "reid_np_j1_channels",
    "reid_np_j1_potential",
    "reid_soft_core_triplet_components",
]

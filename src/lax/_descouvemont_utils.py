"""Compatibility wrappers for the former private Descouvemont helper module.

The supported public API now lives under :mod:`lax.models`. This module remains as a
thin compatibility layer for older internal imports while callers migrate.
"""

from __future__ import annotations

from typing import Final

import jax

from lax.models import (
    ALPHA_C12_ROTOR_MODEL,
    NN_MASS_FACTOR,
    O16_CA44_ROTOR_MODEL,
    RotorCoupledOpticalModel,
    channels_from_rotor_model,
    first_column_amplitudes_and_phases,
    make_rotor_coupled_optical_potential,
    open_channel_count,
    reid_np_j1_channels,
    reid_np_j1_potential,
    reid_soft_core_triplet_components,
)
from lax.types import ChannelSpec

CoupledOpticalModel = RotorCoupledOpticalModel

_ALPHA_C12_POTENTIAL = make_rotor_coupled_optical_potential(ALPHA_C12_ROTOR_MODEL)
_O16_CA44_POTENTIAL = make_rotor_coupled_optical_potential(O16_CA44_ROTOR_MODEL)


def np_j1_channels() -> tuple[ChannelSpec, ...]:
    """Return the coupled ``n-p`` ``J=1`` channels."""

    return reid_np_j1_channels()


def alpha_c12_channels() -> tuple[ChannelSpec, ...]:
    """Return the legacy ``α + 12C`` channel tuple."""

    return channels_from_rotor_model(ALPHA_C12_ROTOR_MODEL)


def alpha_c12_open_channel_count(energy: float) -> int:
    """Return the number of open ``α + 12C`` channels at one energy."""

    return open_channel_count(ALPHA_C12_ROTOR_MODEL, energy)


def alpha_c12_potential(radii: jax.Array, channel_index: int, coupled_index: int) -> jax.Array:
    """Return the legacy ``α + 12C`` coupled optical potential."""

    return _ALPHA_C12_POTENTIAL(radii, channel_index, coupled_index)


def o16_ca44_channels() -> tuple[ChannelSpec, ...]:
    """Return the legacy ``16O + 44Ca`` channel tuple."""

    return channels_from_rotor_model(O16_CA44_ROTOR_MODEL)


def o16_ca44_open_channel_count(energy: float) -> int:
    """Return the number of open ``16O + 44Ca`` channels at one energy."""

    return open_channel_count(O16_CA44_ROTOR_MODEL, energy)


def o16_ca44_potential(radii: jax.Array, channel_index: int, coupled_index: int) -> jax.Array:
    """Return the legacy ``16O + 44Ca`` coupled optical potential."""

    return _O16_CA44_POTENTIAL(radii, channel_index, coupled_index)


__all__: Final[list[str]] = [
    "CoupledOpticalModel",
    "NN_MASS_FACTOR",
    "alpha_c12_channels",
    "alpha_c12_open_channel_count",
    "alpha_c12_potential",
    "first_column_amplitudes_and_phases",
    "np_j1_channels",
    "o16_ca44_channels",
    "o16_ca44_open_channel_count",
    "o16_ca44_potential",
    "reid_np_j1_potential",
    "reid_soft_core_triplet_components",
]

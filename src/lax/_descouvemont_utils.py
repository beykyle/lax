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
    """Return the coupled n-p ``J=1`` channels.

    .. deprecated::
        Use :func:`lax.models.reid_np_j1_channels` directly.

    Returns
    -------
    tuple[ChannelSpec, ...]
        ``(³S₁, ³D₁)`` channel pair.
    """

    return reid_np_j1_channels()


def alpha_c12_channels() -> tuple[ChannelSpec, ...]:
    """Return the α + ¹²C channel tuple.

    .. deprecated::
        Use ``lax.models.channels_from_rotor_model(lax.models.ALPHA_C12_ROTOR_MODEL)``
        directly.

    Returns
    -------
    tuple[ChannelSpec, ...]
        Channel definitions for the α + ¹²C rotor-coupled model.
    """

    return channels_from_rotor_model(ALPHA_C12_ROTOR_MODEL)


def alpha_c12_open_channel_count(energy: float) -> int:
    """Return the number of open α + ¹²C channels at one energy.

    .. deprecated::
        Use ``lax.models.open_channel_count(lax.models.ALPHA_C12_ROTOR_MODEL, energy)``
        directly.

    Parameters
    ----------
    energy
        Centre-of-mass energy in MeV.

    Returns
    -------
    int
        Number of channels with ``E > threshold``.
    """

    return open_channel_count(ALPHA_C12_ROTOR_MODEL, energy)


def alpha_c12_potential(radii: jax.Array, channel_index: int, coupled_index: int) -> jax.Array:
    """Return the α + ¹²C coupled rotor-optical potential at one set of radii.

    .. deprecated::
        Use ``lax.models.make_rotor_coupled_optical_potential(lax.models.ALPHA_C12_ROTOR_MODEL)``
        to get a callable, then call it directly.

    Parameters
    ----------
    radii
        Radial grid in fm.
    channel_index
        Bra-channel index.
    coupled_index
        Ket-channel index.

    Returns
    -------
    jax.Array
        Local potential values in MeV, shape ``(len(radii),)``.
    """

    return _ALPHA_C12_POTENTIAL(radii, channel_index, coupled_index)


def o16_ca44_channels() -> tuple[ChannelSpec, ...]:
    """Return the ¹⁶O + ⁴⁴Ca channel tuple.

    .. deprecated::
        Use ``lax.models.channels_from_rotor_model(lax.models.O16_CA44_ROTOR_MODEL)``
        directly.

    Returns
    -------
    tuple[ChannelSpec, ...]
        Channel definitions for the ¹⁶O + ⁴⁴Ca rotor-coupled model.
    """

    return channels_from_rotor_model(O16_CA44_ROTOR_MODEL)


def o16_ca44_open_channel_count(energy: float) -> int:
    """Return the number of open ¹⁶O + ⁴⁴Ca channels at one energy.

    .. deprecated::
        Use ``lax.models.open_channel_count(lax.models.O16_CA44_ROTOR_MODEL, energy)``
        directly.

    Parameters
    ----------
    energy
        Centre-of-mass energy in MeV.

    Returns
    -------
    int
        Number of channels with ``E > threshold``.
    """

    return open_channel_count(O16_CA44_ROTOR_MODEL, energy)


def o16_ca44_potential(radii: jax.Array, channel_index: int, coupled_index: int) -> jax.Array:
    """Return the ¹⁶O + ⁴⁴Ca coupled rotor-optical potential at one set of radii.

    .. deprecated::
        Use ``lax.models.make_rotor_coupled_optical_potential(lax.models.O16_CA44_ROTOR_MODEL)``
        to get a callable, then call it directly.

    Parameters
    ----------
    radii
        Radial grid in fm.
    channel_index
        Bra-channel index.
    coupled_index
        Ket-channel index.

    Returns
    -------
    jax.Array
        Local potential values in MeV, shape ``(len(radii),)``.
    """

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

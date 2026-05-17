from __future__ import annotations

import numpy as np
import pytest

from lax.boundary import compute_boundary_values
from lax.types import ChannelSpec

pytest.importorskip("jax")
pytest.importorskip("mpmath")


def test_compute_boundary_values_open_channel_wronskian() -> None:
    """Open-channel Coulomb values satisfy the Coulomb Wronskian identity."""

    channel = ChannelSpec(l=0, threshold=0.0, mass_factor=1.0)
    energies = np.array([1.5])
    radius = 3.0

    boundary = compute_boundary_values((channel,), energies=energies, channel_radius=radius)

    rho = np.sqrt(energies[0] / channel.mass_factor) * radius
    H_plus = np.asarray(boundary.H_plus)[0, 0]
    H_minus = np.asarray(boundary.H_minus)[0, 0]
    H_plus_p = np.asarray(boundary.H_plus_p)[0, 0]
    H_minus_p = np.asarray(boundary.H_minus_p)[0, 0]

    F = (H_plus - H_minus) / (2.0j)
    G = (H_plus + H_minus) / 2.0
    dF = (H_plus_p - H_minus_p) / (2.0j * rho)
    dG = (H_plus_p + H_minus_p) / (2.0 * rho)

    assert np.allclose(G * dF - F * dG, 1.0, atol=1.0e-10)


def test_compute_boundary_values_sets_is_open_mask() -> None:
    """Positive relative energies are open and negative ones are closed."""

    channel = ChannelSpec(l=0, threshold=0.0, mass_factor=1.0)
    boundary = compute_boundary_values(
        (channel,),
        energies=np.array([0.5, -0.5]),
        channel_radius=4.0,
    )

    assert np.array_equal(np.asarray(boundary.is_open)[:, 0], np.array([True, False]))


def test_compute_boundary_values_hankel_conjugation_for_real_eta() -> None:
    """For real eta, incoming and outgoing Coulomb waves are conjugates."""

    channel = ChannelSpec(l=1, threshold=0.0, mass_factor=1.0)
    boundary = compute_boundary_values(
        (channel,),
        energies=np.array([2.0]),
        channel_radius=5.0,
    )

    assert np.allclose(
        np.asarray(boundary.H_plus)[0, 0],
        np.conj(np.asarray(boundary.H_minus)[0, 0]),
        atol=1.0e-12,
    )


def test_compute_boundary_values_closed_channel_whittaker_ratio_is_finite() -> None:
    """Closed-channel Whittaker values provide a finite Bloch ratio `H'/H`."""

    channel = ChannelSpec(l=0, threshold=1.0, mass_factor=1.0)
    boundary = compute_boundary_values(
        (channel,),
        energies=np.array([0.25]),
        channel_radius=4.0,
    )

    value = np.asarray(boundary.H_plus)[0, 0]
    derivative = np.asarray(boundary.H_plus_p)[0, 0]

    assert np.asarray(boundary.is_open)[0, 0] is np.False_
    assert np.isfinite(value)
    assert np.isfinite(derivative)
    assert np.isfinite(derivative / value)

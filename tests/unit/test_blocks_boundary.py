"""Tests for the (N_b,)-stacked boundary values (DESIGN.md §15.5, Phase 11 item 33)."""

from __future__ import annotations

import numpy as np

from lax.boundary import compute_boundary_values, compute_boundary_values_blocks
from lax.types import ChannelSpec

HBAR2_2MU = 41.47
RADIUS = 10.0
FIELDS = ("H_plus", "H_minus", "H_plus_p", "H_minus_p", "is_open", "k")


def _assert_blocks_match_per_block(block_groups, energies, **kwargs) -> None:
    stacked = compute_boundary_values_blocks(
        block_groups, energies, channel_radius=RADIUS, **kwargs
    )
    for b, group in enumerate(block_groups):
        single = compute_boundary_values(
            channels=group, energies=energies, channel_radius=RADIUS, **kwargs
        )
        for field in FIELDS:
            np.testing.assert_array_equal(
                np.asarray(getattr(stacked, field)[b]),
                np.asarray(getattr(single, field)),
                err_msg=f"block {b}, field {field}",
            )


def test_stacked_boundary_matches_per_block_neutral() -> None:
    energies = np.linspace(1.0, 20.0, 5)
    block_groups = tuple(
        (ChannelSpec(l=ell, threshold=0.0, mass_factor=HBAR2_2MU),) for ell in range(4)
    )
    stacked = compute_boundary_values_blocks(block_groups, energies, channel_radius=RADIUS)
    assert stacked.H_plus.shape == (4, 5, 1)
    _assert_blocks_match_per_block(block_groups, energies)


def test_stacked_boundary_matches_per_block_coulomb() -> None:
    energies = np.linspace(2.0, 12.0, 4)
    block_groups = tuple(
        (ChannelSpec(l=ell, threshold=0.0, mass_factor=HBAR2_2MU),) for ell in (0, 1, 2)
    )
    _assert_blocks_match_per_block(block_groups, energies, z1z2=(2, 82))


def test_stacked_boundary_matches_per_block_closed_channels() -> None:
    energies = np.linspace(0.5, 4.0, 4)
    block_groups = (
        (
            ChannelSpec(l=0, threshold=0.0, mass_factor=HBAR2_2MU),
            ChannelSpec(l=2, threshold=2.0, mass_factor=HBAR2_2MU),
        ),
        (
            ChannelSpec(l=1, threshold=0.0, mass_factor=HBAR2_2MU),
            ChannelSpec(l=1, threshold=6.0, mass_factor=HBAR2_2MU),
        ),
    )
    stacked = compute_boundary_values_blocks(block_groups, energies, channel_radius=RADIUS)
    assert stacked.H_plus.shape == (2, 4, 2)
    # Second channel of block 1 is closed everywhere on this grid.
    assert not np.any(np.asarray(stacked.is_open[1, :, 1]))
    _assert_blocks_match_per_block(block_groups, energies)


def test_stacked_boundary_matches_per_block_with_mass_factor_grid() -> None:
    energies = np.linspace(1.0, 10.0, 3)
    mass_factor_grid = np.column_stack([np.linspace(40.0, 42.0, 3), np.linspace(38.0, 39.0, 3)])
    block_groups = (
        (
            ChannelSpec(l=0, threshold=0.0, mass_factor=HBAR2_2MU),
            ChannelSpec(l=2, threshold=1.0, mass_factor=HBAR2_2MU),
        ),
        (
            ChannelSpec(l=1, threshold=0.0, mass_factor=HBAR2_2MU),
            ChannelSpec(l=3, threshold=1.0, mass_factor=HBAR2_2MU),
        ),
    )
    _assert_blocks_match_per_block(block_groups, energies, mass_factor_grid=mass_factor_grid)


def test_identical_channels_share_slices_across_blocks() -> None:
    energies = np.linspace(1.0, 20.0, 4)
    shared = ChannelSpec(l=1, threshold=0.0, mass_factor=HBAR2_2MU)
    # Two blocks with the same ℓ (e.g. j = ℓ ± ½) must produce identical slices.
    block_groups = (
        (shared,),
        (shared,),
        (ChannelSpec(l=2, threshold=0.0, mass_factor=HBAR2_2MU),),
    )
    stacked = compute_boundary_values_blocks(block_groups, energies, channel_radius=RADIUS)
    for field in FIELDS:
        values = np.asarray(getattr(stacked, field))
        np.testing.assert_array_equal(values[0], values[1])
    assert not np.array_equal(np.asarray(stacked.H_plus[0]), np.asarray(stacked.H_plus[2]))

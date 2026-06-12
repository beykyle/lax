"""Bit-identity tests for the array-parameterized Hamiltonian assembly core.

The v1.5 symmetry-block axis (DESIGN.md §15.5) refactors
``assemble_block_hamiltonian`` onto an array-parameterized core so the
per-channel centrifugal/threshold/mass data can be vmapped over a leading
block axis.  These tests pin the refactor to exact equality with the
channels-keyed path — any weak-type or expression-order drift is a bug.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from lax.meshes import build_mesh
from lax.solvers.assembly import (
    assemble_block_hamiltonian,
    assemble_hamiltonian_arrays,
    block_group_arrays,
    channel_arrays,
)
from lax.types import ChannelSpec


def _two_channel_setup():
    mesh, operators = build_mesh(
        family="legendre",
        regularization="x",
        n=12,
        scale=10.0,
        operators={"T+L", "1/r^2"},
    )
    channels = (
        ChannelSpec(l=0, threshold=0.0, mass_factor=41.47),
        ChannelSpec(l=2, threshold=1.5, mass_factor=38.9),
    )
    return mesh, operators, channels


def test_channel_arrays_values() -> None:
    _, _, channels = _two_channel_setup()
    centrifugal, thresholds, mass_factors = channel_arrays(channels)
    np.testing.assert_array_equal(np.asarray(centrifugal), [0.0, 6.0])
    np.testing.assert_array_equal(np.asarray(thresholds), [0.0, 1.5])
    np.testing.assert_array_equal(np.asarray(mass_factors), [41.47, 38.9])


def test_array_assembly_matches_channels_assembly_exactly() -> None:
    mesh, operators, channels = _two_channel_setup()
    rng = np.random.default_rng(7)
    n = mesh.n
    n_c = len(channels)

    local = jnp.asarray(rng.normal(size=(n_c, n_c, n)))
    local = 0.5 * (local + jnp.swapaxes(local, 0, 1))
    nonlocal_ = jnp.asarray(rng.normal(size=(n_c, n_c, n, n)))
    block = jnp.asarray(rng.normal(size=(n_c * n, n_c * n)))

    for potential in (local, nonlocal_, block):
        reference = assemble_block_hamiltonian(mesh, operators, channels, potential)
        arrays = assemble_hamiltonian_arrays(mesh, operators, *channel_arrays(channels), potential)
        np.testing.assert_array_equal(np.asarray(reference), np.asarray(arrays))


def test_block_group_arrays_stacks_per_block_rows() -> None:
    _, _, channels = _two_channel_setup()
    other = (
        ChannelSpec(l=1, threshold=0.0, mass_factor=41.47),
        ChannelSpec(l=3, threshold=2.0, mass_factor=38.9),
    )
    centrifugal, thresholds, mass_factors = block_group_arrays((channels, other))
    assert centrifugal.shape == (2, 2)
    np.testing.assert_array_equal(np.asarray(centrifugal), [[0.0, 6.0], [2.0, 12.0]])
    np.testing.assert_array_equal(np.asarray(thresholds), [[0.0, 1.5], [0.0, 2.0]])
    np.testing.assert_array_equal(np.asarray(mass_factors), [[41.47, 38.9], [41.47, 38.9]])

"""Tests for the compile() dtype/device parameters (DESIGN.md §14.1)."""

from __future__ import annotations

import pickle

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import lax
from lax.types import ChannelSpec

HBAR2_2MU = 41.47
N = 12
RADIUS = 10.0
ENERGIES = jnp.linspace(2.0, 20.0, 4)
MESH = lax.MeshSpec("legendre", "x", n=N, scale=RADIUS)
CHANNELS = (ChannelSpec(l=1, threshold=0.0, mass_factor=HBAR2_2MU),)


def _kernel(ri: jnp.ndarray, rj: jnp.ndarray) -> jnp.ndarray:
    return -20.0 * jnp.exp(-0.25 * (ri - rj) ** 2 - 0.05 * (ri + rj) ** 2)


def _compile(**kwargs):
    return lax.compile(
        mesh=MESH,
        solvers=("spectrum", "smatrix", "phases", "rmatrix_direct"),
        energies=ENERGIES,
        **kwargs,
    )


def test_float32_caches_and_results() -> None:
    solver = _compile(channels=CHANNELS, dtype=jnp.float32)

    assert solver.mesh.nodes.dtype == jnp.float32
    assert solver.mesh.radii.dtype == jnp.float32
    assert solver.operators.TpL is not None
    assert solver.operators.TpL.dtype == jnp.float32
    assert solver.energies.dtype == jnp.float32
    assert solver.boundary is not None
    assert solver.boundary.H_plus.dtype == jnp.complex64
    assert solver.boundary.k.dtype == jnp.float32

    interaction = solver.nonlocal_potential(_kernel)
    assert interaction.block.dtype == jnp.float32

    assert solver.phases_direct is not None
    phases32 = solver.phases_direct(interaction)
    assert phases32.dtype == jnp.float32

    spectrum = solver.spectrum(interaction)
    assert spectrum.eigenvalues.dtype == jnp.float32
    spectral_phases32 = solver.phases(spectrum)
    assert spectral_phases32.dtype == jnp.float32

    reference = _compile(channels=CHANNELS)
    ref_interaction = reference.nonlocal_potential(_kernel)
    ref_phases = reference.phases_direct(ref_interaction)
    np.testing.assert_allclose(np.asarray(phases32), np.asarray(ref_phases), atol=2e-3, rtol=1e-3)
    np.testing.assert_allclose(
        np.asarray(spectral_phases32), np.asarray(ref_phases), atol=2e-3, rtol=1e-3
    )
    assert np.all(np.isfinite(np.asarray(phases32)))


def test_float32_blocks_mode() -> None:
    block_groups = tuple(
        (ChannelSpec(l=ell, threshold=0.0, mass_factor=HBAR2_2MU),) for ell in (0, 2)
    )
    solver = _compile(blocks=block_groups, dtype=jnp.float32)
    assert solver.boundary is not None
    assert solver.boundary.H_plus.dtype == jnp.complex64
    interaction = solver.nonlocal_potential(_kernel)
    assert solver.phases_direct is not None
    phases = solver.phases_direct(interaction)
    assert phases.shape == (2, len(ENERGIES), 1)
    assert phases.dtype == jnp.float32
    assert np.all(np.isfinite(np.asarray(phases)))


def test_device_placement() -> None:
    cpu = jax.devices("cpu")[0]
    solver = _compile(channels=CHANNELS, device="cpu")
    assert solver.mesh.nodes.devices() == {cpu}
    assert solver.energies.devices() == {cpu}
    assert solver.boundary is not None
    assert solver.boundary.H_plus.devices() == {cpu}

    explicit = _compile(channels=CHANNELS, device=cpu)
    assert explicit.mesh.nodes.devices() == {cpu}


def test_invalid_dtype_rejected() -> None:
    with pytest.raises(ValueError, match="floating-point dtype"):
        _compile(channels=CHANNELS, dtype=jnp.int32)


def test_float32_solver_pickle_round_trip() -> None:
    solver = _compile(channels=CHANNELS, dtype=jnp.float32)
    restored = pickle.loads(pickle.dumps(solver))
    interaction = solver.nonlocal_potential(_kernel)
    assert restored.phases_direct is not None
    assert solver.phases_direct is not None
    np.testing.assert_array_equal(
        np.asarray(restored.phases_direct(interaction)),
        np.asarray(solver.phases_direct(interaction)),
    )
    assert restored.mesh.nodes.dtype == jnp.float32

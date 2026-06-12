"""Shared fixtures for the symmetry-block (§15.5) equivalence test suites."""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp

from lax.types import ChannelSpec

HBAR2_2MU = 41.47

KernelFn = Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
KernelFnE = Callable[[jnp.ndarray, jnp.ndarray, float], jnp.ndarray]


def gaussian_kernel(scale: float) -> KernelFn:
    """Return a smooth non-local Gaussian kernel of the given strength (MeV)."""

    def kernel(ri: jnp.ndarray, rj: jnp.ndarray) -> jnp.ndarray:
        return -scale * jnp.exp(-0.25 * (ri - rj) ** 2 - 0.05 * (ri + rj) ** 2)

    return kernel


def gaussian_kernel_e(scale: float) -> KernelFnE:
    """Return an energy-dependent variant of :func:`gaussian_kernel`."""

    def kernel(ri: jnp.ndarray, rj: jnp.ndarray, energy: float) -> jnp.ndarray:
        return (
            -scale * (1.0 + 0.01 * energy) * jnp.exp(-0.25 * (ri - rj) ** 2 - 0.05 * (ri + rj) ** 2)
        )

    return kernel


def partial_wave_groups(
    ells: tuple[int, ...] = (0, 1, 2),
    mass_factor: float = HBAR2_2MU,
) -> tuple[tuple[ChannelSpec, ...], ...]:
    """Return single-channel symmetry blocks, one per partial wave ℓ."""

    return tuple((ChannelSpec(l=ell, threshold=0.0, mass_factor=mass_factor),) for ell in ells)

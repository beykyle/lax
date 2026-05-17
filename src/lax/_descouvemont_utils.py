from __future__ import annotations

import math
from typing import Final

import jax
import jax.numpy as jnp
import numpy as np

from lax._angular import wigner_3j, wigner_6j
from lax.types import ChannelSpec

type ChannelInfo = tuple[int, int, float]

NN_MASS_FACTOR: Final[float] = 41.472

REID_MU: Final[float] = 0.7
REID_H: Final[float] = 10.463

ALPHA_C12_MASS_FACTOR: Final[float] = 20.736 / (4.0 * 12.0 / (4.0 + 12.0))
ALPHA_C12_THRESHOLD_2: Final[float] = 4.44
ALPHA_C12_THRESHOLD_4: Final[float] = 14.08
ALPHA_C12_V0: Final[float] = 110.0
ALPHA_C12_W0: Final[float] = 20.0
ALPHA_C12_RADIUS: Final[float] = 1.2 * (4.0 ** (1.0 / 3.0) + 12.0 ** (1.0 / 3.0))
ALPHA_C12_DIFFUSENESS: Final[float] = 0.5
ALPHA_C12_RC: Final[float] = ALPHA_C12_RADIUS
ALPHA_C12_RV: Final[float] = 1.2 * 12.0 ** (1.0 / 3.0)
ALPHA_C12_BETA2: Final[float] = 0.58
ALPHA_C12_TOTAL_ANGULAR_MOMENTUM: Final[int] = 3
ALPHA_C12_LAMBDA: Final[int] = 2
ALPHA_C12_CHANNELS_INFO: Final[tuple[ChannelInfo, ...]] = (
    (3, 0, 0.0),
    (1, 2, ALPHA_C12_THRESHOLD_2),
    (3, 2, ALPHA_C12_THRESHOLD_2),
    (5, 2, ALPHA_C12_THRESHOLD_2),
    (1, 4, ALPHA_C12_THRESHOLD_4),
    (3, 4, ALPHA_C12_THRESHOLD_4),
    (5, 4, ALPHA_C12_THRESHOLD_4),
    (7, 4, ALPHA_C12_THRESHOLD_4),
)


def np_j1_channels() -> tuple[ChannelSpec, ...]:
    """Return the Descouvemont Example 2 n-p J=1+ channel tuple."""

    return (
        ChannelSpec(l=0, threshold=0.0, mass_factor=NN_MASS_FACTOR),
        ChannelSpec(l=2, threshold=0.0, mass_factor=NN_MASS_FACTOR),
    )


def reid_soft_core_triplet_components(radii: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return the original Reid soft-core T=0 central/tensor/spin-orbit terms in MeV."""

    x = REID_MU * radii
    e1 = jnp.exp(-x) / x
    e2 = jnp.exp(-2.0 * x) / x
    e4 = jnp.exp(-4.0 * x) / x
    e6 = jnp.exp(-6.0 * x) / x

    v_central = -REID_H * e1 + 105.468 * e2 - 3187.8 * e4 + 9924.3 * e6
    v_tensor = (
        -REID_H * (1.0 + 3.0 / x + 3.0 / x**2) * e1
        + (REID_H * (12.0 / x + 3.0 / x**2) + 351.77) * e4
        - 1673.5 * e6
    )
    v_spin_orbit = 708.91 * e4 - 2713.1 * e6
    return v_central, v_tensor, v_spin_orbit


def reid_np_j1_potential(radii: jax.Array, channel_index: int, coupled_index: int) -> jax.Array:
    """Return the Descouvemont Example 2 n-p J=1+ coupled potential in MeV."""

    v_central, v_tensor, v_spin_orbit = reid_soft_core_triplet_components(radii)
    if channel_index == coupled_index == 0:
        return v_central
    if channel_index == coupled_index == 1:
        return v_central - 2.0 * v_tensor - 3.0 * v_spin_orbit
    return 2.0 * jnp.sqrt(2.0) * v_tensor


def alpha_c12_channels() -> tuple[ChannelSpec, ...]:
    """Return the Descouvemont Example 4 α + 12C channels."""

    return tuple(
        ChannelSpec(l=angular_momentum, threshold=threshold, mass_factor=ALPHA_C12_MASS_FACTOR)
        for angular_momentum, _, threshold in ALPHA_C12_CHANNELS_INFO
    )


def alpha_c12_open_channel_count(energy: float) -> int:
    """Return the number of open α + 12C channels at one c.m. energy."""

    return sum(1 for _, _, threshold in ALPHA_C12_CHANNELS_INFO if energy >= threshold)


def alpha_c12_potential(radii: jax.Array, channel_index: int, coupled_index: int) -> jax.Array:
    """Return the Descouvemont Example 4 coupled α + 12C optical potential in MeV."""

    nuclear_shape = _woods_saxon(radii, ALPHA_C12_RADIUS, ALPHA_C12_DIFFUSENESS)
    derivative = _woods_saxon_derivative(radii, ALPHA_C12_RADIUS, ALPHA_C12_DIFFUSENESS)
    complex_depth = ALPHA_C12_V0 + 1.0j * ALPHA_C12_W0
    nuclear = -complex_depth * nuclear_shape

    result: jax.Array = jnp.zeros_like(  # pyright: ignore[reportUnknownMemberType] -- JAX zeros_like stubs are imprecise.
        nuclear,
        dtype=jnp.complex128,
    )
    if channel_index == coupled_index:
        result = result + nuclear + _uniform_sphere_coulomb(radii, ALPHA_C12_RC, z1=2, z2=6)

    coupling = _alpha_c12_coupling_coefficient(channel_index, coupled_index)
    if coupling != 0.0:
        result = result - coupling * ALPHA_C12_BETA2 * ALPHA_C12_RV * derivative * complex_depth
    return result


def first_column_amplitudes_and_phases(
    smatrix: np.ndarray, open_count: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return amplitudes and half-angles for the first open-channel column."""

    column = smatrix[:open_count, 0]
    return np.abs(column), 0.5 * np.angle(column)


def _woods_saxon(radii: jax.Array, radius: float, diffuseness: float) -> jax.Array:
    """Return the Woods-Saxon form factor."""

    return 1.0 / (1.0 + jnp.exp((radii - radius) / diffuseness))


def _woods_saxon_derivative(radii: jax.Array, radius: float, diffuseness: float) -> jax.Array:
    """Return the radial derivative of the Woods-Saxon form factor."""

    exponential = jnp.exp((radii - radius) / diffuseness)
    denominator = (1.0 + exponential) ** 2
    return exponential / (diffuseness * denominator)


def _uniform_sphere_coulomb(radii: jax.Array, radius: float, z1: int, z2: int) -> jax.Array:
    """Return the uniformly charged-sphere Coulomb potential in MeV."""

    prefactor = z1 * z2 * 1.44
    inside = prefactor * (3.0 - (radii / radius) ** 2) / (2.0 * radius)
    outside = prefactor / radii
    return jnp.where(radii <= radius, inside, outside)  # pyright: ignore[reportUnknownMemberType] -- JAX where stubs are imprecise.


def _alpha_c12_coupling_coefficient(channel_index: int, coupled_index: int) -> float:
    """Return the `example4.f` angular coefficient xfac for the α + 12C λ=2 coupling."""

    angular_momentum, spin, _ = ALPHA_C12_CHANNELS_INFO[channel_index]
    coupled_angular_momentum, coupled_spin, _ = ALPHA_C12_CHANNELS_INFO[coupled_index]
    wigner_i = wigner_3j(coupled_spin, ALPHA_C12_LAMBDA, spin, 0, 0, 0)
    wigner_l = wigner_3j(angular_momentum, ALPHA_C12_LAMBDA, coupled_angular_momentum, 0, 0, 0)
    six_j = wigner_6j(
        spin,
        angular_momentum,
        ALPHA_C12_TOTAL_ANGULAR_MOMENTUM,
        coupled_angular_momentum,
        coupled_spin,
        ALPHA_C12_LAMBDA,
    )
    if wigner_i == 0.0 or wigner_l == 0.0 or six_j == 0.0:
        return 0.0

    factor = math.sqrt(
        (2 * angular_momentum + 1)
        * (2 * coupled_angular_momentum + 1)
        * (2 * spin + 1)
        * (2 * coupled_spin + 1)
        * (2 * ALPHA_C12_LAMBDA + 1)
        / (4.0 * math.pi)
    )
    coefficient = wigner_i * wigner_l * six_j * factor
    if (abs(angular_momentum - coupled_angular_momentum) // 2) % 2 == 1:
        coefficient = -coefficient
    if (ALPHA_C12_TOTAL_ANGULAR_MOMENTUM + ALPHA_C12_LAMBDA) % 2 == 1:
        coefficient = -coefficient
    return coefficient


__all__: Final[list[str]] = [
    "alpha_c12_channels",
    "alpha_c12_open_channel_count",
    "alpha_c12_potential",
    "first_column_amplitudes_and_phases",
    "np_j1_channels",
    "reid_np_j1_potential",
    "reid_soft_core_triplet_components",
]

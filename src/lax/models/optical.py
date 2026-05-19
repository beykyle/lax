"""Reusable rotor-coupled optical-model helpers.

These utilities expose the general machinery behind the coupled optical examples in
the benchmark suite. Users can define their own rotor-coupled models, derive the
corresponding :class:`lax.types.ChannelSpec` objects, and build local potential
callbacks for :func:`lax.assemble_local`.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from lax._angular import wigner_3j, wigner_6j
from lax.types import ChannelSpec

type CoupledPotential = Callable[[jax.Array, int, int], jax.Array]


@dataclass(frozen=True)
class RotorChannel:
    """One channel in a rotor-coupled optical model.

    Parameters
    ----------
    orbital_angular_momentum
        Relative orbital angular momentum ``L`` for the channel.
    target_spin
        Rotor (target-state) spin coupled to the projectile.
    threshold
        Channel threshold in MeV.
    label
        Optional human-readable label for plots or tables.
    """

    orbital_angular_momentum: int
    target_spin: int
    threshold: float
    label: str | None = None


@dataclass(frozen=True)
class RotorCoupledOpticalModel:
    """Parameters for a rotor-coupled optical potential model.

    Parameters
    ----------
    mass_factor
        Conversion factor ``ℏ² / 2μ`` in MeV fm².
    potential_radius
        Woods-Saxon radius used for the diagonal optical potential.
    coupling_radius
        Radius that multiplies the derivative coupling term.
    coulomb_radius
        Radius of the uniformly charged sphere used for the Coulomb term.
    diffuseness
        Woods-Saxon diffuseness in fm.
    real_depth
        Real optical depth in MeV.
    imaginary_depth
        Imaginary optical depth in MeV.
    deformation
        Rotor deformation parameter, typically written ``β``.
    multipole
        Coupling multipole ``λ``.
    total_angular_momentum
        Total coupled angular momentum ``J``.
    projectile_charge
        Projectile charge ``Z_1``.
    target_charge
        Target charge ``Z_2``.
    channels
        Channel definitions included in the coupled model.
    """

    mass_factor: float
    potential_radius: float
    coupling_radius: float
    coulomb_radius: float
    diffuseness: float
    real_depth: float
    imaginary_depth: float
    deformation: float
    multipole: int
    total_angular_momentum: int
    projectile_charge: int
    target_charge: int
    channels: tuple[RotorChannel, ...]


def channels_from_rotor_model(model: RotorCoupledOpticalModel) -> tuple[ChannelSpec, ...]:
    """Return :class:`~lax.types.ChannelSpec` objects for a rotor model.

    Parameters
    ----------
    model
        Rotor-coupled optical model definition.

    Returns
    -------
    tuple[ChannelSpec, ...]
        Channel layout ready to pass to :func:`lax.compile`.
    """

    return tuple(
        ChannelSpec(
            l=channel.orbital_angular_momentum,
            threshold=channel.threshold,
            mass_factor=model.mass_factor,
        )
        for channel in model.channels
    )


def open_channel_count(model: RotorCoupledOpticalModel, energy: float) -> int:
    """Return the number of channels open at one center-of-mass energy.

    Parameters
    ----------
    model
        Rotor-coupled optical model definition.
    energy
        Center-of-mass energy in MeV.

    Returns
    -------
    int
        Number of channels with threshold below or equal to ``energy``.
    """

    return sum(1 for channel in model.channels if energy >= channel.threshold)


def rotor_coupled_optical_potential(
    model: RotorCoupledOpticalModel,
    radii: jax.Array,
    channel_index: int,
    coupled_index: int,
) -> jax.Array:
    """Return one matrix element of a rotor-coupled optical potential in MeV.

    Parameters
    ----------
    model
        Rotor-coupled optical model definition.
    radii
        Radial grid in fm.
    channel_index
        Bra-channel index.
    coupled_index
        Ket-channel index.

    Returns
    -------
    jax.Array
        Complex local potential evaluated on ``radii``.
    """

    nuclear_shape = woods_saxon_form_factor(
        radii,
        radius=model.potential_radius,
        diffuseness=model.diffuseness,
    )
    derivative = woods_saxon_derivative(
        radii,
        radius=model.potential_radius,
        diffuseness=model.diffuseness,
    )
    complex_depth = model.real_depth + 1.0j * model.imaginary_depth
    nuclear = -complex_depth * nuclear_shape

    result: jax.Array = jnp.zeros_like(  # pyright: ignore[reportUnknownMemberType] -- JAX zeros_like stubs are imprecise.
        nuclear,
        dtype=jnp.complex128,
    )
    if channel_index == coupled_index:
        result = (
            result
            + nuclear
            + uniform_sphere_coulomb_potential(
                radii,
                radius=model.coulomb_radius,
                projectile_charge=model.projectile_charge,
                target_charge=model.target_charge,
            )
        )

    coupling = rotor_coupling_coefficient(model, channel_index, coupled_index)
    if coupling != 0.0:
        result = (
            result
            - coupling * model.deformation * model.coupling_radius * derivative * complex_depth
        )
    return result


def make_rotor_coupled_optical_potential(model: RotorCoupledOpticalModel) -> CoupledPotential:
    """Bind a rotor model into a local-potential callback.

    Parameters
    ----------
    model
        Rotor-coupled optical model definition.

    Returns
    -------
    CoupledPotential
        Callback with signature ``(radii, channel_index, coupled_index)`` suitable
        for :func:`lax.assemble_local`.
    """

    def potential(radii: jax.Array, channel_index: int, coupled_index: int) -> jax.Array:
        return rotor_coupled_optical_potential(
            model,
            radii,
            channel_index,
            coupled_index,
        )

    return potential


def woods_saxon_form_factor(radii: jax.Array, radius: float, diffuseness: float) -> jax.Array:
    """Return the Woods-Saxon form factor.

    Parameters
    ----------
    radii
        Radial grid in fm.
    radius
        Woods-Saxon radius in fm.
    diffuseness
        Woods-Saxon diffuseness in fm.

    Returns
    -------
    jax.Array
        Dimensionless Woods-Saxon profile.
    """

    return 1.0 / (1.0 + jnp.exp((radii - radius) / diffuseness))


def woods_saxon_derivative(radii: jax.Array, radius: float, diffuseness: float) -> jax.Array:
    """Return the positive radial derivative factor used in rotor coupling.

    Parameters
    ----------
    radii
        Radial grid in fm.
    radius
        Woods-Saxon radius in fm.
    diffuseness
        Woods-Saxon diffuseness in fm.

    Returns
    -------
    jax.Array
        Derivative factor entering the deformation coupling term.
    """

    exponential = jnp.exp((radii - radius) / diffuseness)
    denominator = (1.0 + exponential) ** 2
    return exponential / (diffuseness * denominator)


def uniform_sphere_coulomb_potential(
    radii: jax.Array,
    radius: float,
    projectile_charge: int,
    target_charge: int,
) -> jax.Array:
    """Return the uniformly charged-sphere Coulomb potential in MeV.

    Parameters
    ----------
    radii
        Radial grid in fm.
    radius
        Sphere radius in fm.
    projectile_charge
        Projectile charge ``Z_1``.
    target_charge
        Target charge ``Z_2``.

    Returns
    -------
    jax.Array
        Coulomb potential in MeV.
    """

    prefactor = projectile_charge * target_charge * 1.44
    inside = prefactor * (3.0 - (radii / radius) ** 2) / (2.0 * radius)
    outside = prefactor / radii
    return jnp.where(radii <= radius, inside, outside)  # pyright: ignore[reportUnknownMemberType] -- JAX where stubs are imprecise.


def rotor_coupling_coefficient(
    model: RotorCoupledOpticalModel,
    channel_index: int,
    coupled_index: int,
) -> float:
    """Return the angular coupling coefficient for one channel pair.

    Parameters
    ----------
    model
        Rotor-coupled optical model definition.
    channel_index
        Bra-channel index.
    coupled_index
        Ket-channel index.

    Returns
    -------
    float
        Angular coupling coefficient multiplying the derivative term.
    """

    channel = model.channels[channel_index]
    coupled_channel = model.channels[coupled_index]
    wigner_i = wigner_3j(
        coupled_channel.target_spin,
        model.multipole,
        channel.target_spin,
        0,
        0,
        0,
    )
    wigner_l = wigner_3j(
        channel.orbital_angular_momentum,
        model.multipole,
        coupled_channel.orbital_angular_momentum,
        0,
        0,
        0,
    )
    six_j = wigner_6j(
        channel.target_spin,
        channel.orbital_angular_momentum,
        model.total_angular_momentum,
        coupled_channel.orbital_angular_momentum,
        coupled_channel.target_spin,
        model.multipole,
    )
    if wigner_i == 0.0 or wigner_l == 0.0 or six_j == 0.0:
        return 0.0

    factor = math.sqrt(
        (2 * channel.orbital_angular_momentum + 1)
        * (2 * coupled_channel.orbital_angular_momentum + 1)
        * (2 * channel.target_spin + 1)
        * (2 * coupled_channel.target_spin + 1)
        * (2 * model.multipole + 1)
        / (4.0 * math.pi)
    )
    coefficient = wigner_i * wigner_l * six_j * factor
    if (
        abs(channel.orbital_angular_momentum - coupled_channel.orbital_angular_momentum) // 2
    ) % 2 == 1:
        coefficient = -coefficient
    if (model.total_angular_momentum + model.multipole) % 2 == 1:
        coefficient = -coefficient
    return coefficient


def first_column_amplitudes_and_phases(
    smatrix: np.ndarray,
    open_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return amplitudes and half-angles for the first open-channel column.

    Parameters
    ----------
    smatrix
        Open-channel collision matrix.
    open_count
        Number of open channels represented in the matrix.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Absolute values and half-angles of the first column.
    """

    column = smatrix[:open_count, 0]
    return np.abs(column), 0.5 * np.angle(column)


__all__ = [
    "CoupledPotential",
    "RotorChannel",
    "RotorCoupledOpticalModel",
    "channels_from_rotor_model",
    "first_column_amplitudes_and_phases",
    "make_rotor_coupled_optical_potential",
    "open_channel_count",
    "rotor_coupled_optical_potential",
    "rotor_coupling_coefficient",
    "uniform_sphere_coulomb_potential",
    "woods_saxon_derivative",
    "woods_saxon_form_factor",
]

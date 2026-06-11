"""Reusable preset models built from :mod:`lax.models` primitives."""

from __future__ import annotations

from typing import Final

from lax.models.optical import RotorChannel, RotorCoupledOpticalModel

ALPHA_C12_ROTOR_MODEL: Final[RotorCoupledOpticalModel] = RotorCoupledOpticalModel(
    mass_factor=20.736 / (4.0 * 12.0 / (4.0 + 12.0)),
    potential_radius=1.2 * (4.0 ** (1.0 / 3.0) + 12.0 ** (1.0 / 3.0)),
    coupling_radius=1.2 * 12.0 ** (1.0 / 3.0),
    coulomb_radius=1.2 * (4.0 ** (1.0 / 3.0) + 12.0 ** (1.0 / 3.0)),
    diffuseness=0.5,
    real_depth=110.0,
    imaginary_depth=20.0,
    deformation=0.58,
    multipole=2,
    total_angular_momentum=3,
    projectile_charge=2,
    target_charge=6,
    channels=(
        RotorChannel(orbital_angular_momentum=3, target_spin=0, threshold=0.0, label="0+, L=3"),
        RotorChannel(orbital_angular_momentum=1, target_spin=2, threshold=4.44, label="2+, L=1"),
        RotorChannel(orbital_angular_momentum=3, target_spin=2, threshold=4.44, label="2+, L=3"),
        RotorChannel(orbital_angular_momentum=5, target_spin=2, threshold=4.44, label="2+, L=5"),
        RotorChannel(orbital_angular_momentum=1, target_spin=4, threshold=14.08, label="4+, L=1"),
        RotorChannel(orbital_angular_momentum=3, target_spin=4, threshold=14.08, label="4+, L=3"),
        RotorChannel(orbital_angular_momentum=5, target_spin=4, threshold=14.08, label="4+, L=5"),
        RotorChannel(orbital_angular_momentum=7, target_spin=4, threshold=14.08, label="4+, L=7"),
    ),
)

O16_CA44_ROTOR_MODEL: Final[RotorCoupledOpticalModel] = RotorCoupledOpticalModel(
    mass_factor=20.736 / (16.0 * 44.0 / (16.0 + 44.0)),
    potential_radius=1.2 * (16.0 ** (1.0 / 3.0) + 44.0 ** (1.0 / 3.0)),
    coupling_radius=1.2 * 44.0 ** (1.0 / 3.0),
    coulomb_radius=1.2 * (16.0 ** (1.0 / 3.0) + 44.0 ** (1.0 / 3.0)),
    diffuseness=0.5,
    real_depth=110.0,
    imaginary_depth=20.0,
    deformation=0.4,
    multipole=2,
    total_angular_momentum=30,
    projectile_charge=8,
    target_charge=20,
    channels=(
        RotorChannel(orbital_angular_momentum=30, target_spin=0, threshold=0.0, label="0+, L=30"),
        RotorChannel(
            orbital_angular_momentum=28,
            target_spin=2,
            threshold=1.156,
            label="2+, L=28",
        ),
        RotorChannel(
            orbital_angular_momentum=30,
            target_spin=2,
            threshold=1.156,
            label="2+, L=30",
        ),
        RotorChannel(
            orbital_angular_momentum=32,
            target_spin=2,
            threshold=1.156,
            label="2+, L=32",
        ),
    ),
)

__all__ = [
    "ALPHA_C12_ROTOR_MODEL",
    "O16_CA44_ROTOR_MODEL",
]

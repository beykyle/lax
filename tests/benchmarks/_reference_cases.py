from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np


@dataclass(frozen=True)
class SingleChannelCollisionReference:
    """Published single-channel collision-matrix outputs for one mesh setup."""

    scale: float
    n_basis: int
    n_intervals: int
    energies: np.ndarray
    collision_matrix: np.ndarray


@dataclass(frozen=True)
class YamaguchiReference:
    """Published Yamaguchi phase shifts for one mesh setup."""

    scale: float
    n_basis: int
    n_intervals: int
    energies: np.ndarray
    phases_deg: np.ndarray


ALPHA_PB_REFERENCE_A14_N60_NS1: Final[SingleChannelCollisionReference] = (
    SingleChannelCollisionReference(
        scale=14.0,
        n_basis=60,
        n_intervals=1,
        energies=np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float64),
        collision_matrix=np.array(
            [
                1.0000e00 + 5.9801e-19j,
                1.0000e00 + 7.4950e-07j,
                9.9893e-01 + 9.0496e-03j,
                6.5081e-01 + 2.9560e-01j,
                6.4367e-02 + 4.1130e-02j,
            ],
            dtype=np.complex128,
        ),
    )
)
ALPHA_PB_REFERENCE_A14_N15_NS5: Final[SingleChannelCollisionReference] = (
    SingleChannelCollisionReference(
        scale=14.0,
        n_basis=15,
        n_intervals=5,
        energies=np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float64),
        collision_matrix=np.array(
            [
                1.0000e00 + 5.9631e-19j,
                1.0000e00 + 7.4950e-07j,
                9.9893e-01 + 9.0496e-03j,
                6.5081e-01 + 2.9560e-01j,
                6.4367e-02 + 4.1130e-02j,
            ],
            dtype=np.complex128,
        ),
    )
)
ALPHA_PB_REFERENCE_A16_N15_NS5: Final[SingleChannelCollisionReference] = (
    SingleChannelCollisionReference(
        scale=16.0,
        n_basis=15,
        n_intervals=5,
        energies=np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float64),
        collision_matrix=np.array(
            [
                1.0000e00 + 2.8398e-17j,
                1.0000e00 + 4.4229e-06j,
                9.9879e-01 + 1.0301e-02j,
                6.5059e-01 + 2.9599e-01j,
                6.4381e-02 + 4.1206e-02j,
            ],
            dtype=np.complex128,
        ),
    )
)
ALPHA_PB_REFERENCES: Final[tuple[SingleChannelCollisionReference, ...]] = (
    ALPHA_PB_REFERENCE_A14_N60_NS1,
    ALPHA_PB_REFERENCE_A14_N15_NS5,
    ALPHA_PB_REFERENCE_A16_N15_NS5,
)

YAMAGUCHI_REFERENCE_A8_N10_NS1: Final[YamaguchiReference] = YamaguchiReference(
    scale=8.0,
    n_basis=10,
    n_intervals=1,
    energies=np.array([0.1, 10.0], dtype=np.float64),
    phases_deg=np.array([-15.077, 85.637], dtype=np.float64),
)
YAMAGUCHI_REFERENCE_A8_N15_NS1: Final[YamaguchiReference] = YamaguchiReference(
    scale=8.0,
    n_basis=15,
    n_intervals=1,
    energies=np.array([0.1, 10.0], dtype=np.float64),
    phases_deg=np.array([-15.077, 85.637], dtype=np.float64),
)
YAMAGUCHI_REFERENCE_A12_N15_NS1: Final[YamaguchiReference] = YamaguchiReference(
    scale=12.0,
    n_basis=15,
    n_intervals=1,
    energies=np.array([0.1, 10.0], dtype=np.float64),
    phases_deg=np.array([-15.079, 85.635], dtype=np.float64),
)
YAMAGUCHI_REFERENCES: Final[tuple[YamaguchiReference, ...]] = (
    YAMAGUCHI_REFERENCE_A8_N10_NS1,
    YAMAGUCHI_REFERENCE_A8_N15_NS1,
    YAMAGUCHI_REFERENCE_A12_N15_NS1,
)

__all__ = [
    "ALPHA_PB_REFERENCE_A14_N15_NS5",
    "ALPHA_PB_REFERENCE_A14_N60_NS1",
    "ALPHA_PB_REFERENCE_A16_N15_NS5",
    "ALPHA_PB_REFERENCES",
    "SingleChannelCollisionReference",
    "YAMAGUCHI_REFERENCE_A12_N15_NS1",
    "YAMAGUCHI_REFERENCE_A8_N10_NS1",
    "YAMAGUCHI_REFERENCE_A8_N15_NS1",
    "YAMAGUCHI_REFERENCES",
    "YamaguchiReference",
]

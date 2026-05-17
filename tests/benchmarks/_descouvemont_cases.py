from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np


@dataclass(frozen=True)
class NpJ1Reference:
    """Published Descouvemont Example 2 outputs for one channel radius."""

    scale: float
    n_basis: int
    energies: np.ndarray
    phase_11: np.ndarray
    phase_22: np.ndarray
    eta_12: np.ndarray


@dataclass(frozen=True)
class AlphaC12Reference:
    """Published Descouvemont Example 4 first-column outputs for one channel radius."""

    scale: float
    n_basis: int
    energies: np.ndarray
    amplitudes: tuple[np.ndarray, ...]
    phases: tuple[np.ndarray, ...]


NP_J1_REFERENCE_A7: Final[NpJ1Reference] = NpJ1Reference(
    scale=7.0,
    n_basis=60,
    energies=np.array([12.0, 24.0, 36.0, 48.0], dtype=np.float64),
    phase_11=np.array([1.4256, 1.1052, 0.90165, 0.74889], dtype=np.float64),
    phase_22=np.array([-0.048047, -0.11502, -0.16959, -0.21425], dtype=np.float64),
    eta_12=np.array([0.067922, 0.082249, 0.099708, 0.11575], dtype=np.float64),
)

NP_J1_REFERENCE_A8: Final[NpJ1Reference] = NpJ1Reference(
    scale=8.0,
    n_basis=75,
    energies=np.array([12.0, 24.0, 36.0, 48.0], dtype=np.float64),
    phase_11=np.array([1.4258, 1.1052, 0.90182, 0.74900], dtype=np.float64),
    phase_22=np.array([-0.049358, -0.11506, -0.17002, -0.21469], dtype=np.float64),
    eta_12=np.array([0.064110, 0.082470, 0.097677, 0.11440], dtype=np.float64),
)

ALPHA_C12_REFERENCE_A11: Final[AlphaC12Reference] = AlphaC12Reference(
    scale=11.0,
    n_basis=80,
    energies=np.array([4.0, 8.0, 12.0, 16.0, 20.0], dtype=np.float64),
    amplitudes=(
        np.array([0.61525], dtype=np.float64),
        np.array([0.18113, 0.066769, 0.048913, 0.024643], dtype=np.float64),
        np.array([0.087310, 0.042829, 0.040388, 0.073597], dtype=np.float64),
        np.array(
            [0.043124, 0.027057, 0.029530, 0.034450, 0.0029433, 0.0010489, 0.00018491, 0.000024997],
            dtype=np.float64,
        ),
        np.array(
            [0.028037, 0.019167, 0.019080, 0.020027, 0.0047582, 0.0063842, 0.0094671, 0.0058584],
            dtype=np.float64,
        ),
    ),
    phases=(
        np.array([-0.10040], dtype=np.float64),
        np.array([-1.0734, -0.15641, -0.10245, -0.40518], dtype=np.float64),
        np.array([1.0617, -1.4261, -0.93782, -0.72888], dtype=np.float64),
        np.array(
            [0.33174, 0.42457, 1.2632, -1.2680, -0.65924, -0.97341, -1.3553, 1.5188],
            dtype=np.float64,
        ),
        np.array(
            [-0.41741, -0.51378, 0.42378, 1.0520, 0.77293, 1.3273, 1.4670, 1.2880], dtype=np.float64
        ),
    ),
)


__all__: Final[list[str]] = [
    "ALPHA_C12_REFERENCE_A11",
    "AlphaC12Reference",
    "NP_J1_REFERENCE_A7",
    "NP_J1_REFERENCE_A8",
    "NpJ1Reference",
]

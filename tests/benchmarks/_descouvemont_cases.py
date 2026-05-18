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
class NpJ1Reference:
    """Published Descouvemont Example 2 outputs for one channel radius."""

    scale: float
    n_basis: int
    n_intervals: int
    energies: np.ndarray
    phase_11: np.ndarray
    phase_22: np.ndarray
    eta_12: np.ndarray


@dataclass(frozen=True)
class CoupledColumnReference:
    """Published first-column amplitudes and phases for one coupled-channel setup."""

    scale: float
    n_basis: int
    n_intervals: int
    energies: np.ndarray
    amplitudes: tuple[np.ndarray, ...]
    phases: tuple[np.ndarray, ...]


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

NP_J1_REFERENCE_A7_N60_NS1: Final[NpJ1Reference] = NpJ1Reference(
    scale=7.0,
    n_basis=60,
    n_intervals=1,
    energies=np.array([12.0, 24.0, 36.0, 48.0], dtype=np.float64),
    phase_11=np.array([1.4256, 1.1052, 0.90165, 0.74889], dtype=np.float64),
    phase_22=np.array([-0.048047, -0.11502, -0.16959, -0.21425], dtype=np.float64),
    eta_12=np.array([0.067922, 0.082249, 0.099708, 0.11575], dtype=np.float64),
)
NP_J1_REFERENCE_A7_N30_NS2: Final[NpJ1Reference] = NpJ1Reference(
    scale=7.0,
    n_basis=30,
    n_intervals=2,
    energies=np.array([12.0, 24.0, 36.0, 48.0], dtype=np.float64),
    phase_11=np.array([1.4256, 1.1052, 0.90165, 0.74889], dtype=np.float64),
    phase_22=np.array([-0.048047, -0.11502, -0.16959, -0.21425], dtype=np.float64),
    eta_12=np.array([0.067922, 0.082249, 0.099708, 0.11575], dtype=np.float64),
)
NP_J1_REFERENCE_A8_N25_NS3: Final[NpJ1Reference] = NpJ1Reference(
    scale=8.0,
    n_basis=25,
    n_intervals=3,
    energies=np.array([12.0, 24.0, 36.0, 48.0], dtype=np.float64),
    phase_11=np.array([1.4258, 1.1052, 0.90182, 0.74900], dtype=np.float64),
    phase_22=np.array([-0.049358, -0.11506, -0.17002, -0.21469], dtype=np.float64),
    eta_12=np.array([0.064110, 0.082470, 0.097677, 0.11440], dtype=np.float64),
)
NP_J1_REFERENCES: Final[tuple[NpJ1Reference, ...]] = (
    NP_J1_REFERENCE_A7_N60_NS1,
    NP_J1_REFERENCE_A7_N30_NS2,
    NP_J1_REFERENCE_A8_N25_NS3,
)

O16_CA44_REFERENCE_A12_N25_NS4: Final[CoupledColumnReference] = CoupledColumnReference(
    scale=12.0,
    n_basis=25,
    n_intervals=4,
    energies=np.array([34.0, 44.0], dtype=np.float64),
    amplitudes=(
        np.array([0.99408, 0.019823, 0.0099839, 0.0068090], dtype=np.float64),
        np.array([0.53784, 0.16169, 0.20842, 0.21203], dtype=np.float64),
    ),
    phases=(
        np.array([0.013878, -0.65505, -0.66877, -0.67803], dtype=np.float64),
        np.array([0.18330, 0.24799, 0.034390, -0.076706], dtype=np.float64),
    ),
)
O16_CA44_REFERENCE_A13_N25_NS4: Final[CoupledColumnReference] = CoupledColumnReference(
    scale=13.0,
    n_basis=25,
    n_intervals=4,
    energies=np.array([34.0, 44.0], dtype=np.float64),
    amplitudes=(
        np.array([0.99373, 0.020734, 0.010974, 0.0079570], dtype=np.float64),
        np.array([0.53759, 0.16176, 0.20848, 0.21180], dtype=np.float64),
    ),
    phases=(
        np.array([0.014771, -0.65504, -0.66884, -0.67773], dtype=np.float64),
        np.array([0.18357, 0.24807, 0.034413, -0.076101], dtype=np.float64),
    ),
)
O16_CA44_REFERENCE_A14_N25_NS4: Final[CoupledColumnReference] = CoupledColumnReference(
    scale=14.0,
    n_basis=25,
    n_intervals=4,
    energies=np.array([34.0, 44.0], dtype=np.float64),
    amplitudes=(
        np.array([0.99370, 0.020814, 0.011000, 0.0079145], dtype=np.float64),
        np.array([0.53757, 0.16177, 0.20848, 0.21177], dtype=np.float64),
    ),
    phases=(
        np.array([0.014832, -0.65501, -0.66880, -0.67762], dtype=np.float64),
        np.array([0.18360, 0.24808, 0.034417, -0.076024], dtype=np.float64),
    ),
)
O16_CA44_REFERENCE_A14_N50_NS2: Final[CoupledColumnReference] = CoupledColumnReference(
    scale=14.0,
    n_basis=50,
    n_intervals=2,
    energies=np.array([34.0, 44.0], dtype=np.float64),
    amplitudes=(
        np.array([0.99370, 0.020814, 0.011000, 0.0079145], dtype=np.float64),
        np.array([0.53757, 0.16177, 0.20848, 0.21177], dtype=np.float64),
    ),
    phases=(
        np.array([0.014832, -0.65501, -0.66880, -0.67762], dtype=np.float64),
        np.array([0.18360, 0.24808, 0.034417, -0.076024], dtype=np.float64),
    ),
)
O16_CA44_REFERENCES: Final[tuple[CoupledColumnReference, ...]] = (
    O16_CA44_REFERENCE_A12_N25_NS4,
    O16_CA44_REFERENCE_A13_N25_NS4,
    O16_CA44_REFERENCE_A14_N25_NS4,
    O16_CA44_REFERENCE_A14_N50_NS2,
)

ALPHA_C12_REFERENCE_A9_N25_NS4: Final[CoupledColumnReference] = CoupledColumnReference(
    scale=9.0,
    n_basis=25,
    n_intervals=4,
    energies=np.array([4.0, 8.0, 12.0, 16.0, 20.0], dtype=np.float64),
    amplitudes=(
        np.array([0.63596], dtype=np.float64),
        np.array([0.17408, 0.041658, 0.052298, 0.026680], dtype=np.float64),
        np.array([0.077759, 0.038135, 0.030539, 0.073661], dtype=np.float64),
        np.array(
            [0.042879, 0.027207, 0.029671, 0.034527, 0.0029332, 0.0010489, 0.00018585, 0.000025162],
            dtype=np.float64,
        ),
        np.array(
            [0.028051, 0.019170, 0.019106, 0.020175, 0.0047665, 0.0063974, 0.0094889, 0.0058624],
            dtype=np.float64,
        ),
    ),
    phases=(
        np.array([-0.094942], dtype=np.float64),
        np.array([-1.0478, -0.36134, -0.093976, -0.33158], dtype=np.float64),
        np.array([1.1344, 1.2203, -0.86494, -0.55400], dtype=np.float64),
        np.array(
            [0.33080, 0.42471, 1.2639, -1.2662, -0.66359, -0.97805, -1.3598, 1.5139],
            dtype=np.float64,
        ),
        np.array(
            [-0.41339, -0.51731, 0.42013, 1.0489, 0.77182, 1.3264, 1.4657, 1.2864], dtype=np.float64
        ),
    ),
)
ALPHA_C12_REFERENCE_A10_N25_NS4: Final[CoupledColumnReference] = CoupledColumnReference(
    scale=10.0,
    n_basis=25,
    n_intervals=4,
    energies=np.array([4.0, 8.0, 12.0, 16.0, 20.0], dtype=np.float64),
    amplitudes=(
        np.array([0.63211], dtype=np.float64),
        np.array([0.18026, 0.043834, 0.052228, 0.027052], dtype=np.float64),
        np.array([0.083944, 0.013480, 0.040382, 0.081238], dtype=np.float64),
        np.array(
            [0.043161, 0.027045, 0.029524, 0.034464, 0.0029445, 0.0010500, 0.00018528, 0.000025063],
            dtype=np.float64,
        ),
        np.array(
            [0.028039, 0.019165, 0.019073, 0.020003, 0.0047594, 0.0063858, 0.0094702, 0.0058594],
            dtype=np.float64,
        ),
    ),
    phases=(
        np.array([-0.10498], dtype=np.float64),
        np.array([-1.0365, -0.037119, -0.12120, -0.37472], dtype=np.float64),
        np.array([1.1559, -1.5076, -0.83169, -0.63383], dtype=np.float64),
        np.array(
            [0.33205, 0.42416, 1.2627, -1.2686, -0.65988, -0.97391, -1.3555, 1.5185],
            dtype=np.float64,
        ),
        np.array(
            [-0.41803, -0.51340, 0.42415, 1.0520, 0.77283, 1.3272, 1.4670, 1.2879], dtype=np.float64
        ),
    ),
)
ALPHA_C12_REFERENCE_A11_N20_NS4: Final[CoupledColumnReference] = CoupledColumnReference(
    scale=11.0,
    n_basis=20,
    n_intervals=4,
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
ALPHA_C12_REFERENCES: Final[tuple[CoupledColumnReference, ...]] = (
    ALPHA_C12_REFERENCE_A9_N25_NS4,
    ALPHA_C12_REFERENCE_A10_N25_NS4,
    ALPHA_C12_REFERENCE_A11_N20_NS4,
)

ALPHA_C12_NOTEBOOK_A11_FULL: Final[CoupledColumnReference] = CoupledColumnReference(
    scale=11.0,
    n_basis=80,
    n_intervals=1,
    energies=np.array([4.0, 8.0, 12.0, 16.0, 20.0], dtype=np.float64),
    amplitudes=(
        np.array([0.6217448887190808], dtype=np.float64),
        np.array(
            [0.1851917284406713, 0.0600333897606479, 0.0552189012752522, 0.0244432355603402],
            dtype=np.float64,
        ),
        np.array(
            [0.0819458991648443, 0.0407324260326619, 0.0467846909218332, 0.0592731336287996],
            dtype=np.float64,
        ),
        np.array(
            [
                0.0431244138658739,
                0.0270576026110178,
                0.0295302932352534,
                0.0344506493516375,
                0.0029433004657287,
                0.0010488597790414,
                0.0001849134170337,
                0.0000249965803525,
            ],
            dtype=np.float64,
        ),
        np.array(
            [
                0.0280376710788114,
                0.0191668551710252,
                0.0190798321648061,
                0.0200272164183306,
                0.0047584203762124,
                0.0063844247278095,
                0.0094672198166932,
                0.0058584273658399,
            ],
            dtype=np.float64,
        ),
    ),
    phases=(
        np.array([-0.0982795569787105], dtype=np.float64),
        np.array(
            [-1.0451019963391328, -0.2329704885801083, -0.1540728177674309, -0.3197238710532695],
            dtype=np.float64,
        ),
        np.array(
            [1.1706329109695930, 1.5537234823460864, -0.9401849599260977, -0.5721555739750408],
            dtype=np.float64,
        ),
        np.array(
            [
                0.3317379472167195,
                0.4245718729645078,
                1.2631841282413587,
                -1.2680459575609488,
                -0.6592612599635107,
                -0.9734268021131747,
                -1.3553509849314378,
                1.5187571668743516,
            ],
            dtype=np.float64,
        ),
        np.array(
            [
                -0.4174043041711418,
                -0.5137837228391653,
                0.4237725997802942,
                1.0519628434017500,
                0.7729187985731898,
                1.3272640594671419,
                1.4670187145653906,
                1.2879543368464175,
            ],
            dtype=np.float64,
        ),
    ),
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

__all__: Final[list[str]] = [
    "ALPHA_C12_NOTEBOOK_A11_FULL",
    "ALPHA_C12_REFERENCE_A10_N25_NS4",
    "ALPHA_C12_REFERENCE_A11_N20_NS4",
    "ALPHA_C12_REFERENCE_A9_N25_NS4",
    "ALPHA_C12_REFERENCES",
    "ALPHA_PB_REFERENCE_A14_N15_NS5",
    "ALPHA_PB_REFERENCE_A14_N60_NS1",
    "ALPHA_PB_REFERENCE_A16_N15_NS5",
    "ALPHA_PB_REFERENCES",
    "CoupledColumnReference",
    "NP_J1_REFERENCE_A7_N30_NS2",
    "NP_J1_REFERENCE_A7_N60_NS1",
    "NP_J1_REFERENCE_A8_N25_NS3",
    "NP_J1_REFERENCES",
    "NpJ1Reference",
    "O16_CA44_REFERENCE_A12_N25_NS4",
    "O16_CA44_REFERENCE_A13_N25_NS4",
    "O16_CA44_REFERENCE_A14_N25_NS4",
    "O16_CA44_REFERENCE_A14_N50_NS2",
    "O16_CA44_REFERENCES",
    "SingleChannelCollisionReference",
    "YAMAGUCHI_REFERENCE_A12_N15_NS1",
    "YAMAGUCHI_REFERENCE_A8_N10_NS1",
    "YAMAGUCHI_REFERENCE_A8_N15_NS1",
    "YAMAGUCHI_REFERENCES",
    "YamaguchiReference",
]

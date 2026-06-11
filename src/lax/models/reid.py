"""Reid soft-core interaction helpers for the coupled ``n-p`` example."""

from __future__ import annotations

from typing import Final

import jax
import jax.numpy as jnp
import numpy as np

from lax.constants import hbar2_over_2mu
from lax.types import ChannelSpec, Interaction, Solver

NN_MASS_FACTOR: Final[float] = hbar2_over_2mu(1.008665, 1.008665)

_REID_MU: Final[float] = 0.7
_REID_H: Final[float] = 10.463


def reid_np_j1_channels() -> tuple[ChannelSpec, ...]:
    """Return the coupled ``^3S_1``-``^3D_1`` channel pair for ``J=1``.

    Returns
    -------
    tuple[ChannelSpec, ...]
        Two-channel layout for the standard Reid soft-core triplet example.
    """

    return (
        ChannelSpec(l=0, threshold=0.0, mass_factor=NN_MASS_FACTOR),
        ChannelSpec(l=2, threshold=0.0, mass_factor=NN_MASS_FACTOR),
    )


def reid_soft_core_triplet_components(
    radii: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return the Reid soft-core triplet components in MeV.

    Parameters
    ----------
    radii
        Radial grid in fm.

    Returns
    -------
    tuple[jax.Array, jax.Array, jax.Array]
        Central, tensor, and spin-orbit terms in MeV.
    """

    x = _REID_MU * radii
    e1 = jnp.exp(-x) / x
    e2 = jnp.exp(-2.0 * x) / x
    e4 = jnp.exp(-4.0 * x) / x
    e6 = jnp.exp(-6.0 * x) / x

    v_central = -_REID_H * e1 + 105.468 * e2 - 3187.8 * e4 + 9924.3 * e6
    v_tensor = (
        -_REID_H * (1.0 + 3.0 / x + 3.0 / x**2) * e1
        + (_REID_H * (12.0 / x + 3.0 / x**2) + 351.77) * e4
        - 1673.5 * e6
    )
    v_spin_orbit = 708.91 * e4 - 2713.1 * e6
    return v_central, v_tensor, v_spin_orbit


def interaction_from_reid_np_j1(solver: Solver) -> Interaction:
    """Build an :class:`~lax.Interaction` for the coupled Reid soft-core ``n-p`` model.

    Decomposes the ``J=1`` triplet potential into its three physical terms via the
    §6.1 term-decomposition pattern and assembles them through
    :meth:`~lax.Solver.interaction_from_funcs`:

    * Central: ``v_central(r)`` on the diagonal channels.
    * Tensor: ``v_tensor(r)`` scaled by ``[[0, 2√2], [2√2, -2]]``.
    * Spin-orbit: ``v_spin_orbit(r)`` scaled by ``[[0, 0], [0, -3]]``.

    Parameters
    ----------
    solver
        Compiled two-channel solver (see :func:`reid_np_j1_channels`) whose
        :meth:`~lax.Solver.interaction_from_funcs` entry point is used to assemble
        the potential block.

    Returns
    -------
    Interaction
        Energy-independent assembled potential block ready for ``solver.spectrum``
        or ``solver.rmatrix_direct``.
    """

    tensor_coupling = 2.0 * np.sqrt(2.0)
    A_tensor = np.array([[0.0, tensor_coupling], [tensor_coupling, -2.0]])
    A_spin_orbit = np.array([[0.0, 0.0], [0.0, -3.0]])

    def _central(r: jax.Array) -> jax.Array:
        return reid_soft_core_triplet_components(r)[0]

    def _tensor(r: jax.Array) -> jax.Array:
        return reid_soft_core_triplet_components(r)[1]

    def _spin_orbit(r: jax.Array) -> jax.Array:
        return reid_soft_core_triplet_components(r)[2]

    assert solver.interaction_from_funcs is not None
    return solver.interaction_from_funcs(
        local=[
            (_central, np.eye(2)),
            (_tensor, A_tensor),
            (_spin_orbit, A_spin_orbit),
        ],
    )


__all__ = [
    "NN_MASS_FACTOR",
    "interaction_from_reid_np_j1",
    "reid_np_j1_channels",
    "reid_soft_core_triplet_components",
]

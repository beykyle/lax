from __future__ import annotations

import hypothesis.strategies as st
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings

import lax as lm
from lax.spectral import smatrix_from_R
from lax.types import BoundaryValues

pytest.importorskip("jax")

_N = 10
_SCALE = 8.0
_HBAR2_2MU = 41.472
_ENERGIES = jnp.asarray([1.0, 5.0, 10.0])

_SOLVER = lm.compile(
    mesh=lm.MeshSpec("legendre", "x", n=_N, scale=_SCALE),
    channels=(lm.ChannelSpec(l=0, threshold=0.0, mass_factor=_HBAR2_2MU),),
    operators=("T+L",),
    solvers=("spectrum", "rmatrix", "smatrix"),
    energies=_ENERGIES,
)


@st.composite
def _local_gaussian_potential(draw: st.DrawFn) -> jax.Array:
    depth = draw(st.floats(-200.0, -1.0, allow_nan=False, allow_infinity=False))
    width = draw(st.floats(0.5, _SCALE / 2, allow_nan=False, allow_infinity=False))
    r = np.asarray(_SOLVER.mesh.radii)
    V = depth * np.exp(-((r / width) ** 2)) * _HBAR2_2MU
    return jnp.asarray(V[None, None, :])  # shape (1, 1, N)


def _interaction(V: jax.Array) -> lm.Interaction:
    """Wrap a raw (1, 1, N) local potential as an Interaction (block = diag(V[0, 0]))."""

    return lm.Interaction(block=jnp.diag(V[0, 0]), energy_dependent=False)


@pytest.mark.property
@settings(deadline=None)
@given(V=_local_gaussian_potential())
def test_smatrix_is_unitary(V: jax.Array) -> None:
    """S-matrix diagonal elements lie on the unit circle for open channels."""
    assert _SOLVER.smatrix is not None
    assert _SOLVER.spectrum is not None
    spectrum = _SOLVER.spectrum(_interaction(V))
    S = np.asarray(_SOLVER.smatrix(spectrum))  # shape (N_E, 1, 1)
    assert _SOLVER.boundary is not None
    for i in range(len(_ENERGIES)):
        if not np.asarray(_SOLVER.boundary.is_open)[i, 0]:
            continue
        assert abs(abs(S[i, 0, 0]) - 1.0) < 1e-10, (
            f"energy index {i}: |S| = {abs(S[i, 0, 0])}, expected 1.0"
        )


@pytest.mark.property
@settings(deadline=None)
@given(V=_local_gaussian_potential())
def test_smatrix_agrees_with_per_energy_rmatrix_path(V: jax.Array) -> None:
    """S reconstructed from the per-energy R-matrix matches the grid S-matrix."""
    assert _SOLVER.spectrum is not None
    assert _SOLVER.rmatrix is not None
    assert _SOLVER.smatrix is not None
    assert _SOLVER.boundary is not None

    spectrum = _SOLVER.spectrum(_interaction(V))
    S_grid = np.asarray(_SOLVER.smatrix(spectrum))  # shape (N_E, 1, 1)

    for i, energy in enumerate(np.asarray(_ENERGIES)):
        if not np.asarray(_SOLVER.boundary.is_open)[i, 0]:
            continue
        R = _SOLVER.rmatrix(spectrum, float(energy))
        boundary_slice = BoundaryValues(
            H_plus=_SOLVER.boundary.H_plus[i],
            H_minus=_SOLVER.boundary.H_minus[i],
            H_plus_p=_SOLVER.boundary.H_plus_p[i],
            H_minus_p=_SOLVER.boundary.H_minus_p[i],
            is_open=_SOLVER.boundary.is_open[i],
            k=_SOLVER.boundary.k[i],
        )
        S_from_R = np.asarray(smatrix_from_R(R, boundary_slice))
        assert np.allclose(S_from_R, S_grid[i], atol=1e-10), (
            f"energy index {i}: S_from_R={S_from_R}, S_grid={S_grid[i]}"
        )

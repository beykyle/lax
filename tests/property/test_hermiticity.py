from __future__ import annotations

import hypothesis.strategies as st
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings

import lax as lm
from lax.solvers import assemble_block_hamiltonian

pytest.importorskip("jax")

_N = 10
_SCALE = 8.0
_HBAR2_2MU = 41.472
_ENERGIES = jnp.asarray([1.0, 5.0, 10.0])

_SOLVER = lm.compile(
    mesh=lm.MeshSpec("legendre", "x", n=_N, scale=_SCALE),
    channels=(lm.ChannelSpec(l=0, threshold=0.0, mass_factor=_HBAR2_2MU),),
    operators=("T+L",),
    solvers=("spectrum",),
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
def test_block_hamiltonian_is_real_symmetric(V: jax.Array) -> None:
    """Block Hamiltonian H = H^T for any real local potential."""
    H = np.asarray(assemble_block_hamiltonian(_SOLVER.mesh, _SOLVER.operators, _SOLVER.channels, V))
    assert np.allclose(H, H.T, atol=1e-10)


@pytest.mark.property
@settings(deadline=None)
@given(V=_local_gaussian_potential())
def test_eigenvalues_are_real_for_real_symmetric_potential(V: jax.Array) -> None:
    """eigh path returns real eigenvalues for a real symmetric Hamiltonian."""
    assert _SOLVER.spectrum is not None
    spectrum = _SOLVER.spectrum(_interaction(V))
    assert jnp.issubdtype(spectrum.eigenvalues.dtype, jnp.floating)

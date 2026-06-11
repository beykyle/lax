from __future__ import annotations

import hypothesis.strategies as st
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings

import lax as lm

pytest.importorskip("jax")

_N = 10
_SCALE = 8.0
_HBAR2_2MU = 41.472
_ENERGIES = jnp.asarray([1.0, 5.0, 10.0])

_SOLVER = lm.compile(
    mesh=lm.MeshSpec("legendre", "x", n=_N, scale=_SCALE),
    channels=(lm.ChannelSpec(l=0, threshold=0.0, mass_factor=_HBAR2_2MU),),
    operators=("T+L",),
    solvers=("spectrum", "smatrix"),
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
    """Wrap a raw ``(1, 1, N)`` local potential as an Interaction for spectrum().

    The assembled single-channel block is ``diag(V[0, 0])`` — identical to the
    assembly the kernel performed internally — so the result (and its gradient
    w.r.t. ``V``) is unchanged.
    """

    return lm.Interaction(block=jnp.diag(V[0, 0]), energy_dependent=False)


@pytest.mark.property
@settings(deadline=None)
@given(V=_local_gaussian_potential())
def test_spectrum_eigenvalues_are_differentiable(V: jax.Array) -> None:
    """jax.grad passes through solver.spectrum for any real local potential."""
    assert _SOLVER.spectrum is not None

    def loss(V: jax.Array) -> jax.Array:
        return jnp.sum(_SOLVER.spectrum(_interaction(V)).eigenvalues)

    grad = jax.grad(loss)(V)
    assert jnp.all(jnp.isfinite(grad))


@pytest.mark.property
@settings(deadline=None)
@given(V=_local_gaussian_potential())
def test_smatrix_is_differentiable(V: jax.Array) -> None:
    """jax.grad passes through the full spectrum → S-matrix pipeline."""
    assert _SOLVER.spectrum is not None
    assert _SOLVER.smatrix is not None

    def loss(V: jax.Array) -> jax.Array:
        spec = _SOLVER.spectrum(_interaction(V))
        return jnp.sum(_SOLVER.smatrix(spec).real)

    grad = jax.grad(loss)(V)
    assert jnp.all(jnp.isfinite(grad))

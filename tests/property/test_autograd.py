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


# ---------------------------------------------------------------------------
# T4 (spec v0.1.5.1) — gradients through the DWBA chain in miniature.
# Differentiable paths: eigh spectral (real V) and linear_solve direct
# (complex V).  The eig path is asserted to raise (C8): its spectra flow
# through jax.pure_callback → host np.linalg.eig, which has no JVP/VJP.

_WAVE_SOLVER = lm.compile(
    mesh=lm.MeshSpec("legendre", "x", n=_N, scale=_SCALE),
    channels=(lm.ChannelSpec(l=0, threshold=0.0, mass_factor=_HBAR2_2MU),),
    solvers=("spectrum", "wavefunction"),
    energies=_ENERGIES,
)
_DIRECT_SOLVER = lm.compile(
    mesh=lm.MeshSpec("legendre", "x", n=_N, scale=_SCALE),
    channels=(lm.ChannelSpec(l=0, threshold=0.0, mass_factor=_HBAR2_2MU),),
    solvers=("rmatrix_direct",),
    energies=_ENERGIES,
    V_is_complex=True,
    method="linear_solve",
)
_EIG_SOLVER = lm.compile(
    mesh=lm.MeshSpec("legendre", "x", n=_N, scale=_SCALE),
    channels=(lm.ChannelSpec(l=0, threshold=0.0, mass_factor=_HBAR2_2MU),),
    solvers=("spectrum", "wavefunction"),
    energies=_ENERGIES,
    V_is_complex=True,
    method="eig",
)


@pytest.mark.property
def test_wavefunction_grid_matrix_element_is_differentiable() -> None:
    """jax.grad through eigh spectrum → wavefunction_grid → matrix_element."""
    assert _WAVE_SOLVER.spectrum is not None
    assert _WAVE_SOLVER.wavefunction_grid is not None
    assert _WAVE_SOLVER.matrix_element is not None
    radii = _WAVE_SOLVER.mesh.radii
    operator = jnp.exp(-0.1 * radii**2)  # local transition operator at the nodes

    def loss(depth: jax.Array) -> jax.Array:
        V = lm.Interaction(
            block=jnp.diag(depth * jnp.exp(-((radii / 2.0) ** 2))), energy_dependent=False
        )
        spec = _WAVE_SOLVER.spectrum(V)
        psi = _WAVE_SOLVER.wavefunction_grid(spec)  # (N_E, N)
        element = _WAVE_SOLVER.matrix_element(psi, psi, operator, conjugate=False)
        return jnp.sum(jnp.abs(element) ** 2)

    grad = jax.grad(loss)(jnp.asarray(-25.0))
    assert jnp.isfinite(grad)


@pytest.mark.property
def test_wavefunction_direct_grid_matrix_element_is_differentiable() -> None:
    """jax.grad through linear_solve → wavefunction_direct_grid → matrix_element."""
    assert _DIRECT_SOLVER.wavefunction_direct_grid is not None
    assert _DIRECT_SOLVER.matrix_element is not None
    radii = _DIRECT_SOLVER.mesh.radii
    operator = jnp.exp(-0.1 * radii**2)

    def loss(depth: jax.Array) -> jax.Array:
        profile = (depth - 4.0j) * jnp.exp(-((radii / 2.0) ** 2))
        V = lm.Interaction(block=jnp.diag(profile), energy_dependent=False)
        psi = _DIRECT_SOLVER.wavefunction_direct_grid(V)  # (N_E, N)
        element = _DIRECT_SOLVER.matrix_element(psi, psi, operator, conjugate=False)
        return jnp.sum(jnp.abs(element) ** 2)

    grad = jax.grad(loss)(jnp.asarray(-25.0))
    assert jnp.isfinite(grad)


@pytest.mark.property
def test_eig_path_raises_on_differentiation() -> None:
    """C8: the eig spectral path cannot be differentiated (pure_callback).

    Guarded so a future custom-JVP upgrade of the eig path is noticed.
    """
    assert _EIG_SOLVER.spectrum is not None
    radii = _EIG_SOLVER.mesh.radii

    def loss(depth: jax.Array) -> jax.Array:
        profile = (depth - 4.0j) * jnp.exp(-((radii / 2.0) ** 2))
        V = lm.Interaction(block=jnp.diag(profile), energy_dependent=False)
        return jnp.sum(jnp.abs(_EIG_SOLVER.spectrum(V).eigenvalues) ** 2)

    with pytest.raises(ValueError, match="do not support JVP"):
        jax.grad(loss)(jnp.asarray(-25.0))

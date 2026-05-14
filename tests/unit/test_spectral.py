from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from lax.boundary._types import BoundaryValues
from lax.spectral import (
    Spectrum,
    greens_from_spectrum,
    phases_from_S,
    rmatrix_from_spectrum,
    smatrix_from_R,
)

pytest.importorskip("jax")


def test_rmatrix_from_spectrum_matches_direct_inverse() -> None:
    """Spectral R-matrix agrees with a direct inverse construction."""

    eigenvalues = jnp.array([1.25, 2.5])
    surface_amplitudes = jnp.array([[1.0], [2.0]])
    spectrum = Spectrum(
        eigenvalues=eigenvalues,
        surface_amplitudes=surface_amplitudes,
        eigenvectors=jnp.eye(2),
        is_hermitian=True,
    )

    energy = 0.5
    mass_factor = 2.0
    channel_radius = 8.0
    q = np.array([[1.0], [2.0]])
    direct = (
        q.T
        @ np.linalg.inv(np.diag(np.asarray(eigenvalues)) - (energy / mass_factor) * np.eye(2))
        @ q
    ) / channel_radius

    result = np.asarray(
        rmatrix_from_spectrum(
            spectrum,
            energy=energy,
            channel_radius=channel_radius,
            mass_factor=mass_factor,
        )
    )

    assert np.allclose(result, direct)


def test_greens_from_spectrum_matches_direct_inverse() -> None:
    """Spectral Green's function matches the direct matrix inverse."""

    theta = 0.37
    eigenvectors = np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ]
    )
    eigenvalues = np.array([0.8, 1.7])
    hamiltonian = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    spectrum = Spectrum(
        eigenvalues=jnp.asarray(eigenvalues),
        surface_amplitudes=jnp.asarray([[1.0], [0.0]]),
        eigenvectors=jnp.asarray(eigenvectors),
        is_hermitian=True,
    )

    energy = 0.2
    mass_factor = 2.0
    expected = np.linalg.inv(hamiltonian - (energy / mass_factor) * np.eye(2))
    result = np.asarray(greens_from_spectrum(spectrum, energy=energy, mass_factor=mass_factor))

    assert np.allclose(result, expected)


def test_smatrix_from_R_is_unitary_for_real_r() -> None:
    """Real R with conjugate boundary data yields a unitary S-matrix."""

    R = jnp.asarray([[0.4 + 0.0j]])
    boundary = BoundaryValues(
        H_plus=jnp.asarray([1.0 + 1.0j]),
        H_minus=jnp.asarray([1.0 - 1.0j]),
        H_plus_p=jnp.asarray([0.3 + 0.2j]),
        H_minus_p=jnp.asarray([0.3 - 0.2j]),
        is_open=jnp.asarray([True]),
    )

    S = np.asarray(smatrix_from_R(R, boundary))
    identity = S.conj().T @ S

    assert np.allclose(identity, np.eye(1), atol=1.0e-12)


def test_phases_from_S_returns_half_argument() -> None:
    """Phase extraction returns half the S-matrix eigenphase."""

    delta = 0.37
    S = jnp.asarray([[np.exp(2.0j * delta)]])

    phases = np.asarray(phases_from_S(S))

    assert np.allclose(phases, np.array([delta]))

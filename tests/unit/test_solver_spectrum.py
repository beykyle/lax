from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from lax.boundary import BoundaryValues
from lax.boundary._types import OperatorMatrices
from lax.meshes.legendre import build_legendre_x
from lax.solvers import assemble_block_hamiltonian, bind_observables, build_Q, make_spectrum_kernel
from lax.types import ChannelSpec

pytest.importorskip("jax")


def test_build_q_places_boundary_values_in_channel_blocks() -> None:
    """`build_Q` places the boundary vector in each channel block."""

    mesh, _ = build_legendre_x(n=4, scale=6.0, operators={"T+L"})
    channels = (
        ChannelSpec(l=0, threshold=0.0, mass_factor=2.0),
        ChannelSpec(l=1, threshold=1.0, mass_factor=2.0),
    )

    q = np.asarray(build_Q(mesh, channels))

    assert q.shape == (8, 2)
    assert np.allclose(q[:4, 0], np.asarray(mesh.basis_at_boundary))
    assert np.allclose(q[4:, 1], np.asarray(mesh.basis_at_boundary))
    assert np.allclose(q[:4, 1], 0.0)
    assert np.allclose(q[4:, 0], 0.0)


def test_assemble_block_hamiltonian_scales_threshold_and_local_potential() -> None:
    """Assembly preserves the fm^-2 threshold and local-potential scaling."""

    mesh, operators = build_legendre_x(n=3, scale=5.0, operators={"T+L", "1/r^2"})
    channels = (ChannelSpec(l=0, threshold=2.0, mass_factor=4.0),)
    local_potential = jnp.asarray([[[1.0, 3.0, 5.0]]])

    hamiltonian = np.asarray(assemble_block_hamiltonian(mesh, operators, channels, local_potential))
    expected = (
        np.asarray(operators.TpL)
        + np.eye(3) * (channels[0].threshold / channels[0].mass_factor)
        + np.diag(np.array([1.0, 3.0, 5.0]) / channels[0].mass_factor)
    )

    assert np.allclose(hamiltonian, expected)


def test_make_spectrum_kernel_matches_direct_eigh() -> None:
    """The compiled `eigh` kernel reproduces the direct Hamiltonian eigensystem."""

    mesh, operators = build_legendre_x(n=4, scale=8.0, operators={"T+L"})
    channels = (ChannelSpec(l=0, threshold=0.0, mass_factor=2.0),)
    potential = jnp.asarray([[[0.5, 1.0, 1.5, 2.0]]])

    kernel = make_spectrum_kernel(mesh, operators, channels, keep_eigenvectors=True)
    spectrum = kernel(potential)
    hamiltonian = assemble_block_hamiltonian(mesh, operators, channels, potential)
    expected_eigenvalues, expected_eigenvectors = np.linalg.eigh(np.asarray(hamiltonian))
    expected_surface_amplitudes = expected_eigenvectors.T @ np.asarray(build_Q(mesh, channels))

    assert np.allclose(np.asarray(spectrum.eigenvalues), expected_eigenvalues)
    assert np.allclose(
        np.abs(np.asarray(spectrum.surface_amplitudes)), np.abs(expected_surface_amplitudes)
    )
    assert spectrum.eigenvectors is not None


def test_bind_observables_matches_direct_spectral_helpers() -> None:
    """Bound observables agree with the direct spectral helpers."""

    mesh, operators = build_legendre_x(n=4, scale=7.0, operators={"T+L"})
    channels = (ChannelSpec(l=0, threshold=0.0, mass_factor=2.0),)
    potential = jnp.asarray([[[0.1, 0.2, 0.3, 0.4]]])
    boundary = BoundaryValues(
        H_plus=jnp.asarray([[1.0 + 1.0j]]),
        H_minus=jnp.asarray([[1.0 - 1.0j]]),
        H_plus_p=jnp.asarray([[0.25 + 0.15j]]),
        H_minus_p=jnp.asarray([[0.25 - 0.15j]]),
        is_open=jnp.asarray([[True]]),
    )
    spectrum = make_spectrum_kernel(mesh, operators, channels, keep_eigenvectors=True)(potential)

    rmatrix, smatrix, phases, greens, eigh = bind_observables(
        mesh,
        channels,
        energies=jnp.asarray([0.5]),
        boundary=boundary,
    )

    assert smatrix is not None
    assert phases is not None

    r_value = np.asarray(rmatrix(spectrum, 0.5))
    s_value = np.asarray(smatrix(spectrum))
    phase_value = np.asarray(phases(spectrum))
    green_value = np.asarray(greens(spectrum, 0.5))
    eigenvalues, eigenvectors = eigh(spectrum)

    assert r_value.shape == (1, 1)
    assert s_value.shape == (1, 1, 1)
    assert phase_value.shape == (1, 1)
    assert green_value.shape == (4, 4)
    assert eigenvectors is not None
    assert np.allclose(np.asarray(eigenvalues), np.asarray(spectrum.eigenvalues))


def test_make_spectrum_kernel_requires_supported_method() -> None:
    """The MVP solver kernel supports only the `eigh` path."""

    mesh, operators = build_legendre_x(n=3, scale=6.0, operators={"T+L"})
    channels = (ChannelSpec(l=0, threshold=0.0, mass_factor=1.0),)

    with pytest.raises(ValueError, match="not implemented"):
        make_spectrum_kernel(mesh, operators, channels, method="eig")


def test_assemble_block_hamiltonian_requires_t_plus_l() -> None:
    """Assembly fails clearly if `TpL` is missing."""

    mesh, _ = build_legendre_x(n=3, scale=5.0, operators=set())
    channels = (ChannelSpec(l=0, threshold=0.0, mass_factor=1.0),)

    with pytest.raises(ValueError, match="TpL"):
        assemble_block_hamiltonian(
            mesh,
            OperatorMatrices(),
            channels,
            jnp.asarray([[[0.0, 0.0, 0.0]]]),
        )

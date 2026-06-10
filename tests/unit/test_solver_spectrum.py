from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import lax as lm
from lax.boundary import BoundaryValues
from lax.boundary._types import OperatorMatrices
from lax.meshes.legendre import build_legendre_x
from lax.operators import make_interaction_from_array
from lax.solvers import (
    assemble_block_hamiltonian,
    bind_observables,
    build_Q,
    make_rmatrix_direct_kernel,
    make_spectrum_kernel,
)
from lax.solvers.observables import (
    _closed_channel_bloch,
    _decouple_closed_channels,
    _project_open_channels,
)
from lax.spectral.matching import smatrix_from_R
from lax.types import ChannelSpec, MeshSpec

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
    """Assembly uses the symmetric MeV form: m_c·(T+L) + E_threshold·I + V (untouched)."""

    mesh, operators = build_legendre_x(n=3, scale=5.0, operators={"T+L", "1/r^2"})
    channels = (ChannelSpec(l=0, threshold=2.0, mass_factor=4.0),)
    local_potential = jnp.asarray([[[1.0, 3.0, 5.0]]])

    hamiltonian = np.asarray(assemble_block_hamiltonian(mesh, operators, channels, local_potential))
    m_c = channels[0].mass_factor
    expected = (
        m_c * np.asarray(operators.TpL)
        + np.eye(3) * channels[0].threshold
        + np.diag(np.array([1.0, 3.0, 5.0]))
    )

    assert np.allclose(hamiltonian, expected)


def test_make_spectrum_kernel_matches_direct_eigh() -> None:
    """The compiled `eigh` kernel reproduces the direct Hamiltonian eigensystem."""

    mesh, operators = build_legendre_x(n=4, scale=8.0, operators={"T+L"})
    channels = (ChannelSpec(l=0, threshold=0.0, mass_factor=2.0),)
    potential = jnp.asarray([[[0.5, 1.0, 1.5, 2.0]]])

    kernel = make_spectrum_kernel(mesh, operators, channels, keep_eigenvectors=True)
    spectrum = kernel(potential)
    # Assembler returns H_MeV; spectrum path divides by m0 before eigh → fm⁻² eigenvalues.
    m0 = channels[0].mass_factor
    hamiltonian = np.asarray(assemble_block_hamiltonian(mesh, operators, channels, potential)) / m0
    expected_eigenvalues, expected_eigenvectors = np.linalg.eigh(hamiltonian)
    expected_surface_amplitudes = expected_eigenvectors.T @ np.asarray(build_Q(mesh, channels))

    assert np.allclose(np.asarray(spectrum.eigenvalues), expected_eigenvalues)
    assert np.allclose(
        np.abs(np.asarray(spectrum.surface_amplitudes)), np.abs(expected_surface_amplitudes)
    )
    assert spectrum.eigenvectors is not None


def test_make_spectrum_kernel_matches_complex_symmetric_eig() -> None:
    """The compiled `eig` kernel reproduces a complex-symmetric eigensystem."""

    mesh, operators = build_legendre_x(n=4, scale=8.0, operators={"T+L"})
    channels = (ChannelSpec(l=0, threshold=0.0, mass_factor=2.0),)
    potential = jnp.asarray([[[0.5 + 0.1j, 1.0 + 0.2j, 1.5 + 0.3j, 2.0 + 0.4j]]])

    spectrum = make_spectrum_kernel(
        mesh,
        operators,
        channels,
        method="eig",
        keep_eigenvectors=True,
    )(potential)
    # Assembler returns H_MeV; spectrum path divides by m0 before eig → fm⁻² eigenvalues.
    m0 = channels[0].mass_factor
    hamiltonian = np.asarray(assemble_block_hamiltonian(mesh, operators, channels, potential)) / m0
    expected_eigenvalues, expected_eigenvectors = np.linalg.eig(hamiltonian.astype(np.complex128))
    order = np.argsort(expected_eigenvalues.real)
    expected_eigenvalues = expected_eigenvalues[order]
    expected_eigenvectors = expected_eigenvectors[:, order]
    expected_norm = np.sqrt(np.diag(expected_eigenvectors.T @ expected_eigenvectors))
    expected_eigenvectors = expected_eigenvectors / expected_norm[None, :]
    result_eigenvalues = np.asarray(spectrum.eigenvalues)
    result_vectors = np.asarray(spectrum.eigenvectors)
    result_order = np.argsort(result_eigenvalues.real)
    result_eigenvalues = result_eigenvalues[result_order]
    result_vectors = result_vectors[:, result_order]

    assert spectrum.eigenvectors is not None
    assert spectrum.is_hermitian is False
    assert np.allclose(result_eigenvalues.real, expected_eigenvalues.real)
    assert np.allclose(result_eigenvalues.imag, expected_eigenvalues.imag)
    assert np.allclose(result_vectors.T @ result_vectors, np.eye(4), atol=1.0e-10)
    assert np.allclose(
        np.abs(result_vectors),
        np.abs(expected_eigenvectors),
        atol=1.0e-10,
    )


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

    rmatrix, smatrix, phases, greens, wavefunction, eigh = bind_observables(
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
    wavefunction_value = np.asarray(wavefunction(spectrum, 0.5, jnp.asarray([1.0, 0.0, 0.0, 0.0])))
    eigenvalues, eigenvectors = eigh(spectrum)

    assert r_value.shape == (1, 1)
    assert s_value.shape == (1, 1, 1)
    assert phase_value.shape == (1, 1)
    assert green_value.shape == (4, 4)
    assert wavefunction_value.shape == (4,)
    assert eigenvectors is not None
    assert np.allclose(np.asarray(eigenvalues), np.asarray(spectrum.eigenvalues))


def test_compile_binds_wavefunction_observable() -> None:
    """`compile()` exposes the wavefunction observable when requested."""

    solver = lm.compile(
        mesh=MeshSpec("legendre", "x", n=4, scale=7.0),
        channels=(ChannelSpec(l=0, threshold=0.0, mass_factor=2.0),),
        solvers=("spectrum", "wavefunction"),
    )
    potential = jnp.asarray([[[0.2, 0.1, 0.0, -0.1]]])

    assert solver.spectrum is not None
    assert solver.wavefunction is not None

    spectrum = solver.spectrum(potential)
    values = np.asarray(solver.wavefunction(spectrum, 0.5, jnp.asarray([1.0, 0.0, 0.0, 0.0])))

    assert values.shape == (4,)


def test_coupled_channel_rmatrix_matches_direct_solver() -> None:
    """The spectral and direct R-matrix solvers agree for coupled channels."""

    mesh, operators = build_legendre_x(n=4, scale=8.0, operators={"T+L", "1/r^2"})
    channels = (
        ChannelSpec(l=0, threshold=0.0, mass_factor=2.0),
        ChannelSpec(l=1, threshold=0.5, mass_factor=2.0),
    )
    potential = jnp.asarray(
        [
            [
                [0.4, 0.3, 0.2, 0.1],
                [0.05, 0.04, 0.03, 0.02],
            ],
            [
                [0.05, 0.04, 0.03, 0.02],
                [0.2, 0.1, 0.0, -0.1],
            ],
        ]
    )
    spectrum_kernel = make_spectrum_kernel(mesh, operators, channels, keep_eigenvectors=True)
    spectrum = spectrum_kernel(potential)
    rmatrix, smatrix, phases, _, _, _ = bind_observables(
        mesh,
        channels,
        energies=jnp.asarray([0.25]),
        boundary=BoundaryValues(
            H_plus=jnp.asarray([[1.0 + 1.0j, 1.1 + 0.9j]]),
            H_minus=jnp.asarray([[1.0 - 1.0j, 1.1 - 0.9j]]),
            H_plus_p=jnp.asarray([[0.2 + 0.1j, 0.3 + 0.15j]]),
            H_minus_p=jnp.asarray([[0.2 - 0.1j, 0.3 - 0.15j]]),
            is_open=jnp.asarray([[True, True]]),
            k=jnp.asarray([[1.0, 1.0]]),
        ),
    )
    array_builder = make_interaction_from_array(mesh, channels, jnp.asarray([0.25]))
    interaction = array_builder(
        local=[
            (potential[0, 0], np.array([[1.0, 0.0], [0.0, 0.0]])),
            (potential[0, 1], np.array([[0.0, 1.0], [1.0, 0.0]])),
            (potential[1, 1], np.array([[0.0, 0.0], [0.0, 1.0]])),
        ],
        energy_dependent=False,
    )
    direct = make_rmatrix_direct_kernel(mesh, operators, channels, jnp.asarray([0.25]), None)(
        interaction
    )

    assert smatrix is not None
    assert phases is not None
    assert np.allclose(np.asarray(rmatrix(spectrum, 0.25)), np.asarray(direct[0]), atol=1.0e-10)
    assert np.asarray(smatrix(spectrum)).shape == (1, 2, 2)
    assert np.asarray(phases(spectrum)).shape == (1, 2)


def test_closed_channel_decoupling_matches_direct_bloch_updated_rmatrix() -> None:
    """Closed-channel decoupling matches a direct solve with the Whittaker Bloch update."""

    mesh, operators = build_legendre_x(n=4, scale=8.0, operators={"T+L", "1/r^2"})
    channels = (
        ChannelSpec(l=0, threshold=0.0, mass_factor=2.0),
        ChannelSpec(l=1, threshold=1.5, mass_factor=2.0),
    )
    potential = jnp.asarray(
        [
            [
                [0.3, 0.2, 0.1, 0.0],
                [0.04, 0.03, 0.02, 0.01],
            ],
            [
                [0.04, 0.03, 0.02, 0.01],
                [0.1, 0.0, -0.1, -0.2],
            ],
        ]
    )
    energy = 0.25
    boundary = BoundaryValues(
        H_plus=jnp.asarray([[1.0 + 1.0j, 2.0 + 0.0j]]),
        H_minus=jnp.asarray([[1.0 - 1.0j, 2.0 + 0.0j]]),
        H_plus_p=jnp.asarray([[0.2 + 0.1j, 0.6 + 0.0j]]),
        H_minus_p=jnp.asarray([[0.2 - 0.1j, 0.6 + 0.0j]]),
        is_open=jnp.asarray([[True, False]]),
        k=jnp.asarray([[np.sqrt(energy / channels[0].mass_factor), 1.0]]),
    )
    spectrum_kernel = make_spectrum_kernel(mesh, operators, channels, keep_eigenvectors=True)
    spectrum = spectrum_kernel(potential)
    rmatrix, smatrix, _, _, _, _ = bind_observables(
        mesh,
        channels,
        energies=jnp.asarray([energy]),
        boundary=boundary,
    )

    assert smatrix is not None

    hamiltonian = np.asarray(assemble_block_hamiltonian(mesh, operators, channels, potential))
    q = np.asarray(build_Q(mesh, channels))
    m_c = channels[0].mass_factor
    q_prime = np.sqrt(m_c) * q
    # MeV form: C = H_MeV - E*I; R = Q'^T C^{-1} Q' / a where Q' = sqrt(m_c)*Q
    base_matrix = hamiltonian - np.eye(hamiltonian.shape[0]) * energy
    base_solved = np.linalg.solve(base_matrix, q_prime)
    raw_rmatrix = (q_prime.T @ base_solved) / mesh.scale
    bloch = np.asarray(
        _closed_channel_bloch(
            boundary.H_plus[0],
            boundary.H_plus_p[0],
            boundary.is_open[0],
        )
    )
    decoupled = np.asarray(
        _decouple_closed_channels(
            jnp.asarray(raw_rmatrix),
            boundary.H_plus[0],
            boundary.H_plus_p[0],
            boundary.is_open[0],
        )
    )

    bloch_matrix = np.diag(bloch)
    # Bloch correction in MeV form: scale by m_c
    updated_matrix = (
        hamiltonian
        - m_c * (q @ bloch_matrix @ q.T) / mesh.scale
        - np.eye(hamiltonian.shape[0]) * energy
    )
    solved = np.linalg.solve(updated_matrix, q_prime)
    direct_rmatrix = (q_prime.T @ solved) / mesh.scale

    assert np.allclose(decoupled[0, 0], direct_rmatrix[0, 0], atol=5.0e-5, rtol=5.0e-5)

    projected_rmatrix, projected_boundary = _project_open_channels(
        jnp.asarray(direct_rmatrix),
        boundary.H_plus[0],
        boundary.H_minus[0],
        boundary.H_plus_p[0],
        boundary.H_minus_p[0],
        boundary.is_open[0],
        boundary.k[0],
    )
    direct_smatrix = np.asarray(smatrix_from_R(projected_rmatrix, projected_boundary))
    bound_smatrix = np.asarray(smatrix(spectrum))

    assert np.allclose(bound_smatrix[0, 0, 0], direct_smatrix[0, 0], atol=5.0e-5, rtol=5.0e-5)


def test_compile_defaults_to_eig_for_complex_cpu_problems() -> None:
    """Complex potentials default to the spectral `eig` path on CPU."""

    solver = lm.compile(
        mesh=MeshSpec("legendre", "x", n=4, scale=8.0),
        channels=(ChannelSpec(l=0, threshold=0.0, mass_factor=2.0),),
        solvers=("spectrum",),
        V_is_complex=True,
    )

    assert solver.method == "eig"


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

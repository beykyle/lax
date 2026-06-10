from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import lax as lm
from lax.meshes import build_mesh
from lax.solvers import assemble_block_hamiltonian, build_Q, make_rmatrix_direct_kernel

pytest.importorskip("jax")

HBAR2_2MU = 41.472


def _make_energy_dependent_potential(solver: lm.Solver, energy: jax.Array) -> jax.Array:
    """Return a smooth energy-dependent local potential in MeV."""

    radii = solver.mesh.radii
    values = (-3.5 * jnp.exp(-((radii / 2.4) ** 2)) + 0.02 * energy) * HBAR2_2MU
    return values[None, None, :]


def test_make_rmatrix_direct_kernel_matches_manual_linear_solve() -> None:
    """The direct kernel matches a manual solve of `(H - E) X = Q`."""

    mesh = lm.MeshSpec("legendre", "x", n=4, scale=7.0)
    channels = (lm.ChannelSpec(l=0, threshold=0.0, mass_factor=2.0),)
    solver = lm.compile(
        mesh=mesh,
        channels=channels,
        operators=("T+L",),
        solvers=("rmatrix_direct",),
        energies=jnp.asarray([0.25, 0.75]),
    )
    potential = jnp.asarray([[[0.1, 0.2, 0.3, 0.4]]])

    kernel = make_rmatrix_direct_kernel(
        solver.mesh,
        solver.operators,
        solver.channels,
        solver.energies,
        None,
    )
    result = np.asarray(kernel(potential))

    # Manual computation using MeV form: H_MeV = m_c*(T+L) + V; C = H_MeV − E·I;
    # R = Q'^T C^{-1} Q' / a where Q' = sqrt(m_c) · Q.
    hamiltonian = np.asarray(
        assemble_block_hamiltonian(solver.mesh, solver.operators, solver.channels, potential)
    )
    q = np.asarray(build_Q(solver.mesh, solver.channels))
    m_c = channels[0].mass_factor
    q_prime = np.sqrt(m_c) * q
    expected = []
    for energy in np.asarray(solver.energies):
        matrix = hamiltonian - np.eye(hamiltonian.shape[0]) * energy
        expected.append((q_prime.T @ np.linalg.solve(matrix, q_prime)) / solver.mesh.scale)

    assert np.allclose(result, np.stack(expected))


def test_compile_exposes_direct_rmatrix_kernel() -> None:
    """`compile()` wires the direct R-matrix kernel into the solver bundle."""

    solver = lm.compile(
        mesh=lm.MeshSpec("legendre", "x", n=4, scale=8.0),
        channels=(lm.ChannelSpec(l=0, threshold=0.0, mass_factor=2.0),),
        solvers=("rmatrix_direct",),
        energies=jnp.asarray([0.5]),
        method="linear_solve",
    )

    assert solver.rmatrix_direct is not None
    assert solver.rmatrix_direct_grid is not None
    assert solver.smatrix_direct_grid is not None
    assert solver.phases_direct_grid is not None
    assert solver.interpolate_rmatrix is not None
    assert solver.interpolate_smatrix is not None
    assert solver.interpolate_phases is not None
    assert solver.spectrum is None


def test_compile_rejects_linear_solve_for_spectrum_path() -> None:
    """The MVP compiler does not build spectral observables with `linear_solve`."""

    with pytest.raises(ValueError, match="spectrum path"):
        lm.compile(
            mesh=lm.MeshSpec("legendre", "x", n=4, scale=8.0),
            channels=(lm.ChannelSpec(l=0, threshold=0.0, mass_factor=2.0),),
            solvers=("spectrum",),
            method="linear_solve",
        )


def test_compile_rejects_propagated_spectrum_path() -> None:
    """Propagated meshes remain a direct-only feature under the Phase 9 design."""

    with pytest.raises(ValueError, match="Spectrum-derived observables"):
        lm.compile(
            mesh=lm.MeshSpec("legendre", "x", n=4, scale=8.0, extras={"n_intervals": 2}),
            channels=(lm.ChannelSpec(l=0, threshold=0.0, mass_factor=2.0),),
            solvers=("spectrum",),
        )


def test_compile_rejects_propagated_grid_transforms() -> None:
    """Propagated meshes still reject grid and momentum transforms explicitly."""

    with pytest.raises(ValueError, match="Radial-grid transforms"):
        lm.compile(
            mesh=lm.MeshSpec("legendre", "x", n=4, scale=8.0, extras={"n_intervals": 2}),
            channels=(lm.ChannelSpec(l=0, threshold=0.0, mass_factor=2.0),),
            solvers=("rmatrix_direct",),
            energies=jnp.asarray([0.5]),
            grid=jnp.linspace(0.1, 7.9, 12),
        )


def test_compile_rejects_propagated_momentum_transforms() -> None:
    """Propagated meshes reject momentum transforms with a mathematical-consistency error."""

    with pytest.raises(ValueError, match="Momentum transforms"):
        lm.compile(
            mesh=lm.MeshSpec("legendre", "x", n=4, scale=8.0, extras={"n_intervals": 2}),
            channels=(lm.ChannelSpec(l=0, threshold=0.0, mass_factor=2.0),),
            solvers=("rmatrix_direct",),
            energies=jnp.asarray([0.5]),
            momenta=jnp.linspace(0.1, 2.0, 12),
        )


def test_compile_rejects_propagated_non_linear_solve_method() -> None:
    """Propagated meshes reject spectral eigensolver methods directly."""

    with pytest.raises(ValueError, match="local direct linear-solve formulation"):
        lm.compile(
            mesh=lm.MeshSpec("legendre", "x", n=4, scale=8.0, extras={"n_intervals": 2}),
            channels=(lm.ChannelSpec(l=0, threshold=0.0, mass_factor=2.0),),
            solvers=("rmatrix_direct",),
            energies=jnp.asarray([0.5]),
            method="eigh",
        )


def test_direct_rmatrix_matches_spectral_rmatrix_for_real_potential() -> None:
    """The direct and spectral R-matrix solvers agree for a real non-local potential."""

    alpha = 0.2316053
    beta = 1.3918324
    hbar2_2mu = 41.472
    energies = jnp.asarray([0.1, 10.0])

    def yamaguchi_kernel(r1: jax.Array, r2: jax.Array) -> jax.Array:
        return -2.0 * beta * (alpha + beta) ** 2 * jnp.exp(-beta * (r1 + r2)) * hbar2_2mu

    solver = lm.compile(
        mesh=lm.MeshSpec("legendre", "x", n=10, scale=8.0),
        channels=(lm.ChannelSpec(l=0, threshold=0.0, mass_factor=hbar2_2mu),),
        operators=("T+L",),
        solvers=("spectrum", "rmatrix", "rmatrix_direct"),
        energies=energies,
    )
    potential = lm.assemble_nonlocal(solver.mesh, yamaguchi_kernel)
    spectrum = solver.spectrum(potential)

    assert solver.rmatrix is not None
    assert solver.rmatrix_direct is not None

    spectral = np.stack(
        [np.asarray(solver.rmatrix(spectrum, float(energy))) for energy in np.asarray(energies)]
    )
    direct = np.asarray(solver.rmatrix_direct(potential))

    assert np.allclose(direct, spectral, atol=1.0e-10, rtol=1.0e-10)


def test_direct_rmatrix_grid_matches_manual_per_energy_solve() -> None:
    """`rmatrix_direct_grid` matches a manual per-energy linear solve with varying `V(E)`."""

    energies = jnp.asarray([0.25, 0.75, 1.25])
    solver = lm.compile(
        mesh=lm.MeshSpec("legendre", "x", n=4, scale=7.0),
        channels=(lm.ChannelSpec(l=0, threshold=0.0, mass_factor=2.0),),
        operators=("T+L",),
        solvers=("rmatrix_direct",),
        energies=energies,
        method="linear_solve",
        energy_dependent=True,
    )
    potentials = jax.vmap(lambda energy: _make_energy_dependent_potential(solver, energy))(energies)

    assert solver.rmatrix_direct_grid is not None

    result = np.asarray(solver.rmatrix_direct_grid(potentials))
    expected = []
    m_c = solver.channels[0].mass_factor
    q = np.asarray(build_Q(solver.mesh, solver.channels))
    q_prime = np.sqrt(m_c) * q
    for index, energy in enumerate(np.asarray(energies)):
        hamiltonian = np.asarray(
            assemble_block_hamiltonian(
                solver.mesh,
                solver.operators,
                solver.channels,
                potentials[index],
            )
        )
        # MeV form: C = H_MeV − E·I; R = Q'^T C^{-1} Q' / a.
        matrix = hamiltonian - np.eye(hamiltonian.shape[0]) * energy
        expected.append((q_prime.T @ np.linalg.solve(matrix, q_prime)) / solver.mesh.scale)

    assert np.allclose(result, np.stack(expected), atol=1.0e-10, rtol=1.0e-10)


def test_assemble_nonlocal_rejects_propagated_mesh() -> None:
    """Propagated meshes reject non-local kernel assembly explicitly."""

    mesh, _ = build_mesh("legendre", "x", n=4, scale=8.0, operators={"T+L"}, n_intervals=2)

    def nonlocal_kernel(r1: jax.Array, r2: jax.Array) -> jax.Array:
        return -2.0 * jnp.exp(-0.5 * (r1 + r2))

    with pytest.raises(ValueError, match="Non-local kernels"):
        lm.assemble_nonlocal(mesh, nonlocal_kernel)


def test_propagated_nonlocal_direct_rejects_inconsistent_request() -> None:
    """Propagated direct solves reject non-local potentials instead of emulating them."""

    propagated_solver = lm.compile(
        mesh=lm.MeshSpec("legendre", "x", n=4, scale=8.0, extras={"n_intervals": 2}),
        channels=(lm.ChannelSpec(l=0, threshold=0.0, mass_factor=2.0),),
        operators=("T+L",),
        solvers=("rmatrix_direct",),
        energies=jnp.asarray([0.3, 0.9]),
        method="linear_solve",
    )

    potential = jnp.ones((1, 1, 8, 8), dtype=jnp.float64)

    assert propagated_solver.rmatrix_direct is not None

    with pytest.raises(ValueError, match="Non-local propagated solves"):
        propagated_solver.rmatrix_direct(potential)


def test_direct_grid_observables_match_spectral_grid_for_real_energy_dependent_potential() -> None:
    """Direct aligned-grid `R/S/δ` agree with the spectral aligned-grid helpers."""

    energies = jnp.linspace(0.2, 2.0, 9)
    spectral_solver = lm.compile(
        mesh=lm.MeshSpec("legendre", "x", n=10, scale=8.0),
        channels=(lm.ChannelSpec(l=0, threshold=0.0, mass_factor=HBAR2_2MU),),
        operators=("T+L",),
        solvers=("spectrum", "smatrix", "phases"),
        energies=energies,
        energy_dependent=True,
    )
    direct_solver = lm.compile(
        mesh=lm.MeshSpec("legendre", "x", n=10, scale=8.0),
        channels=(lm.ChannelSpec(l=0, threshold=0.0, mass_factor=HBAR2_2MU),),
        operators=("T+L",),
        solvers=("rmatrix_direct",),
        energies=energies,
        method="linear_solve",
        energy_dependent=True,
    )

    assert spectral_solver.spectrum is not None
    assert spectral_solver.rmatrix_grid is not None
    assert spectral_solver.smatrix_grid is not None
    assert spectral_solver.phases_grid is not None
    assert direct_solver.rmatrix_direct_grid is not None
    assert direct_solver.smatrix_direct_grid is not None
    assert direct_solver.phases_direct_grid is not None

    spectral_potentials = jax.vmap(
        lambda energy: _make_energy_dependent_potential(spectral_solver, energy)
    )(energies)
    direct_potentials = jax.vmap(
        lambda energy: _make_energy_dependent_potential(direct_solver, energy)
    )(energies)
    spectra = jax.vmap(spectral_solver.spectrum)(spectral_potentials)

    spectral_r = np.asarray(spectral_solver.rmatrix_grid(spectra))
    spectral_s = np.asarray(spectral_solver.smatrix_grid(spectra))
    spectral_phases = np.asarray(spectral_solver.phases_grid(spectra))
    direct_r = np.asarray(direct_solver.rmatrix_direct_grid(direct_potentials))
    direct_s = np.asarray(direct_solver.smatrix_direct_grid(direct_potentials))
    direct_phases = np.asarray(direct_solver.phases_direct_grid(direct_potentials))

    assert np.allclose(direct_r, spectral_r, atol=1.0e-10, rtol=1.0e-10)
    assert np.allclose(direct_s, spectral_s, atol=1.0e-10, rtol=1.0e-10)
    assert np.allclose(direct_phases, spectral_phases, atol=1.0e-10, rtol=1.0e-10)

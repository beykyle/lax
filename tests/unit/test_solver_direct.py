from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import lax as lm
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

    hamiltonian = np.asarray(
        assemble_block_hamiltonian(solver.mesh, solver.operators, solver.channels, potential)
    )
    q = np.asarray(build_Q(solver.mesh, solver.channels))
    expected = []
    for energy in np.asarray(solver.energies):
        matrix = hamiltonian - np.eye(hamiltonian.shape[0]) * (energy / channels[0].mass_factor)
        expected.append((q.T @ np.linalg.solve(matrix, q)) / solver.mesh.scale)

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
    q = np.asarray(build_Q(solver.mesh, solver.channels))
    for index, energy in enumerate(np.asarray(energies)):
        hamiltonian = np.asarray(
            assemble_block_hamiltonian(
                solver.mesh,
                solver.operators,
                solver.channels,
                potentials[index],
            )
        )
        matrix = hamiltonian - np.eye(hamiltonian.shape[0]) * (
            energy / solver.channels[0].mass_factor
        )
        expected.append((q.T @ np.linalg.solve(matrix, q)) / solver.mesh.scale)

    assert np.allclose(result, np.stack(expected), atol=1.0e-10, rtol=1.0e-10)


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

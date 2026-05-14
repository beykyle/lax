from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import lax as lm
from lax.solvers import assemble_block_hamiltonian, build_Q, make_rmatrix_direct_kernel

pytest.importorskip("jax")


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

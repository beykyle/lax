from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import lax as lm

pytest.importorskip("jax")
pytest.importorskip("scipy")

HBAR2_2MU = 41.472


def test_energy_dependent_pade_flow_matches_dense_grid() -> None:
    """The §16.7 Padé workflow interpolates energy-dependent S-matrices accurately."""

    sparse_energies = jnp.linspace(0.2, 2.0, 11)
    dense_energies = jnp.linspace(0.2, 2.0, 25)

    sparse_solver = lm.compile(
        mesh=lm.MeshSpec("legendre", "x", n=12, scale=8.0),
        channels=(lm.ChannelSpec(l=0, threshold=0.0, mass_factor=HBAR2_2MU),),
        solvers=("spectrum",),
        energies=sparse_energies,
        energy_dependent=True,
    )
    dense_solver = lm.compile(
        mesh=lm.MeshSpec("legendre", "x", n=12, scale=8.0),
        channels=(lm.ChannelSpec(l=0, threshold=0.0, mass_factor=HBAR2_2MU),),
        solvers=("spectrum",),
        energies=dense_energies,
        energy_dependent=True,
    )

    def make_potential(solver: lm.Solver, energy: jax.Array) -> jax.Array:
        radii = solver.mesh.radii
        radius_i, radius_j = jnp.meshgrid(radii, radii, indexing="ij")
        weights_i, weights_j = jnp.meshgrid(solver.mesh.weights, solver.mesh.weights, indexing="ij")
        kernel = (
            -2.0 * 1.25 * (0.23 + 1.25) ** 2 * jnp.exp(-1.25 * (radius_i + radius_j))
            + 0.01 * energy
        ) * HBAR2_2MU
        return (kernel * jnp.sqrt(weights_i * weights_j) * solver.mesh.scale)[None, None, :, :]

    def s_at_own_energy(solver: lm.Solver, spectra: object, energies: jax.Array) -> jax.Array:
        assert solver.boundary is not None

        def one(spec_i: object, energy_i: jax.Array, boundary_i: object) -> jax.Array:
            r_value = lm.spectral.rmatrix_from_spectrum(
                spec_i,
                energy_i,
                channel_radius=solver.mesh.scale,
                mass_factor=solver.channels[0].mass_factor,
            )
            return lm.spectral.smatrix_from_R(r_value, boundary_i)

        return jax.vmap(one)(spectra, energies, solver.boundary)

    sparse_potentials = jax.vmap(lambda energy: make_potential(sparse_solver, energy))(
        sparse_energies
    )
    dense_potentials = jax.vmap(lambda energy: make_potential(dense_solver, energy))(dense_energies)
    sparse_spectra = jax.vmap(sparse_solver.spectrum)(sparse_potentials)
    dense_spectra = jax.vmap(dense_solver.spectrum)(dense_potentials)

    sparse_s = s_at_own_energy(sparse_solver, sparse_spectra, sparse_energies)
    dense_s = s_at_own_energy(dense_solver, dense_spectra, dense_energies)

    interpolant = lm.spectral.pade_interpolate(sparse_s, sparse_energies)
    interpolated = jax.vmap(interpolant)(dense_energies)

    assert np.allclose(np.asarray(interpolated), np.asarray(dense_s), atol=1.0e-4, rtol=1.0e-4)

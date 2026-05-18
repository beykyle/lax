from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import lax as lm

pytest.importorskip("jax")
pytest.importorskip("scipy")

HBAR2_2MU = 41.472


def _make_potential(solver: lm.Solver, energy: jax.Array) -> jax.Array:
    """Return a smooth energy-dependent non-local potential."""

    radii = solver.mesh.radii
    radius_i, radius_j = jnp.meshgrid(radii, radii, indexing="ij")
    weights_i, weights_j = jnp.meshgrid(solver.mesh.weights, solver.mesh.weights, indexing="ij")
    kernel = (
        -2.0 * 1.25 * (0.23 + 1.25) ** 2 * jnp.exp(-1.25 * (radius_i + radius_j)) + 0.01 * energy
    ) * HBAR2_2MU
    return (kernel * jnp.sqrt(weights_i * weights_j) * solver.mesh.scale)[None, None, :, :]


def _manual_smatrix_grid(solver: lm.Solver, spectra: object, energies: jax.Array) -> jax.Array:
    """Return aligned-grid S-matrices via the manual §16.7 pattern."""

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


def _manual_smatrix_from_fixed_spectrum(
    solver: lm.Solver, spectrum: object, energies: jax.Array
) -> jax.Array:
    """Return the compile-time grid S-matrix from one fixed spectrum."""

    assert solver.boundary is not None

    def one(energy_i: jax.Array, boundary_i: object) -> jax.Array:
        r_value = lm.spectral.rmatrix_from_spectrum(
            spectrum,
            energy_i,
            channel_radius=solver.mesh.scale,
            mass_factor=solver.channels[0].mass_factor,
        )
        return lm.spectral.smatrix_from_R(r_value, boundary_i)

    return jax.vmap(one)(energies, solver.boundary)


def test_energy_dependent_smatrix_grid_matches_manual_flow() -> None:
    """`smatrix_grid` matches the manual aligned-grid pattern from DESIGN.md §16.7."""

    energies = jnp.linspace(0.2, 2.0, 11)
    solver = lm.compile(
        mesh=lm.MeshSpec("legendre", "x", n=12, scale=8.0),
        channels=(lm.ChannelSpec(l=0, threshold=0.0, mass_factor=HBAR2_2MU),),
        solvers=("spectrum", "smatrix", "phases"),
        energies=energies,
        energy_dependent=True,
    )
    potentials = jax.vmap(lambda energy: _make_potential(solver, energy))(energies)

    assert solver.spectrum is not None
    assert solver.smatrix_grid is not None
    assert solver.phases_grid is not None

    spectra = jax.vmap(solver.spectrum)(potentials)
    manual_smatrix = _manual_smatrix_grid(solver, spectra, energies)
    bound_smatrix = solver.smatrix_grid(spectra)
    bound_phases = solver.phases_grid(spectra)

    assert np.allclose(
        np.asarray(bound_smatrix), np.asarray(manual_smatrix), atol=1.0e-10, rtol=1.0e-10
    )
    assert np.allclose(
        np.asarray(bound_phases),
        np.asarray(jax.vmap(lm.spectral.phases_from_S)(manual_smatrix)),
        atol=1.0e-10,
        rtol=1.0e-10,
    )


def test_energy_dependent_fixed_spectrum_semantics_are_unchanged() -> None:
    """`solver.smatrix(spec)` still means one fixed spectrum evaluated on the full grid."""

    energies = jnp.linspace(0.2, 2.0, 11)
    solver = lm.compile(
        mesh=lm.MeshSpec("legendre", "x", n=12, scale=8.0),
        channels=(lm.ChannelSpec(l=0, threshold=0.0, mass_factor=HBAR2_2MU),),
        solvers=("spectrum", "smatrix"),
        energies=energies,
        energy_dependent=True,
    )
    potentials = jax.vmap(lambda energy: _make_potential(solver, energy))(energies)

    assert solver.spectrum is not None
    assert solver.smatrix is not None
    assert solver.smatrix_grid is not None

    spectra = jax.vmap(solver.spectrum)(potentials)
    fixed_spectrum = jax.tree.map(lambda leaf: leaf[0], spectra)
    fixed_grid = solver.smatrix(fixed_spectrum)
    manual_fixed_grid = _manual_smatrix_from_fixed_spectrum(solver, fixed_spectrum, energies)
    aligned_grid = solver.smatrix_grid(spectra)

    assert np.allclose(
        np.asarray(fixed_grid), np.asarray(manual_fixed_grid), atol=1.0e-10, rtol=1.0e-10
    )
    assert not np.allclose(np.asarray(fixed_grid), np.asarray(aligned_grid))


def test_energy_dependent_pade_flow_matches_dense_grid() -> None:
    """The compile-bound Phase 9 Padé flow interpolates energy-dependent S-matrices accurately."""

    sparse_energies = jnp.linspace(0.2, 2.0, 11)
    dense_energies = jnp.linspace(0.2, 2.0, 25)

    sparse_solver = lm.compile(
        mesh=lm.MeshSpec("legendre", "x", n=12, scale=8.0),
        channels=(lm.ChannelSpec(l=0, threshold=0.0, mass_factor=HBAR2_2MU),),
        solvers=("spectrum", "smatrix", "phases"),
        energies=sparse_energies,
        energy_dependent=True,
    )
    dense_solver = lm.compile(
        mesh=lm.MeshSpec("legendre", "x", n=12, scale=8.0),
        channels=(lm.ChannelSpec(l=0, threshold=0.0, mass_factor=HBAR2_2MU),),
        solvers=("spectrum", "smatrix", "phases"),
        energies=dense_energies,
        energy_dependent=True,
    )

    sparse_potentials = jax.vmap(lambda energy: _make_potential(sparse_solver, energy))(
        sparse_energies
    )
    dense_potentials = jax.vmap(lambda energy: _make_potential(dense_solver, energy))(
        dense_energies
    )

    assert sparse_solver.spectrum is not None
    assert sparse_solver.smatrix_grid is not None
    assert sparse_solver.interpolate_smatrix is not None
    assert sparse_solver.interpolate_phases is not None
    assert dense_solver.spectrum is not None
    assert dense_solver.smatrix_grid is not None

    sparse_spectra = jax.vmap(sparse_solver.spectrum)(sparse_potentials)
    dense_spectra = jax.vmap(dense_solver.spectrum)(dense_potentials)

    sparse_s = sparse_solver.smatrix_grid(sparse_spectra)
    dense_s = dense_solver.smatrix_grid(dense_spectra)
    sparse_phases = jax.vmap(lm.spectral.phases_from_S)(sparse_s)

    interpolant = sparse_solver.interpolate_smatrix(sparse_s)
    interpolated = jax.vmap(interpolant)(dense_energies)
    phase_interpolant = sparse_solver.interpolate_phases(sparse_phases)
    recovered_phase_knots = jax.vmap(phase_interpolant)(sparse_energies)

    assert np.allclose(np.asarray(interpolated), np.asarray(dense_s), atol=1.0e-4, rtol=1.0e-4)
    assert np.allclose(
        np.asarray(recovered_phase_knots), np.asarray(sparse_phases), atol=1.0e-10, rtol=1.0e-10
    )

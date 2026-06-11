from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import lax as lm

pytest.importorskip("jax")
pytest.importorskip("scipy")

HBAR2_2MU = 41.472


def _spectra(solver: lm.Solver, energies: jax.Array) -> object:
    """Build the energy-dependent non-local Interaction and decompose it.

    The bare form factor ``g(r, r'; E)`` (shape ``(N_E, N, N)``) is passed to
    ``interaction_from_array``, which owns the Gauss scaling ``√(λ_i λ_j)·a`` and the
    coupling kron; ``spectrum`` then dispatches over the energy axis internally.
    """

    radii = solver.mesh.radii
    radius_i, radius_j = jnp.meshgrid(radii, radii, indexing="ij")

    def kernel(energy: jax.Array) -> jax.Array:  # bare g(r, r'; E), shape (N, N)
        return (
            -2.0 * 1.25 * (0.23 + 1.25) ** 2 * jnp.exp(-1.25 * (radius_i + radius_j))
            + 0.01 * energy
        ) * HBAR2_2MU

    g = jax.vmap(kernel)(energies)  # (N_E, N, N)
    assert solver.interaction_from_array is not None
    assert solver.spectrum is not None
    interaction = solver.interaction_from_array(
        nonlocal_=[(g, np.ones((1, 1)))], energy_dependent=True
    )
    return solver.spectrum(interaction)


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

    assert solver.spectrum is not None
    assert solver.smatrix_grid is not None
    assert solver.phases_grid is not None

    spectra = _spectra(solver, energies)
    manual_smatrix = _manual_smatrix_grid(solver, spectra, energies)
    bound_smatrix = solver.smatrix_grid(spectra)
    bound_phases = solver.phases_grid(spectra)

    assert np.allclose(
        np.asarray(bound_smatrix),
        np.asarray(manual_smatrix),
        atol=1.0e-10,
        rtol=1.0e-10,
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

    assert solver.spectrum is not None
    assert solver.smatrix is not None
    assert solver.smatrix_grid is not None

    spectra = _spectra(solver, energies)
    fixed_spectrum = jax.tree.map(lambda leaf: leaf[0], spectra)
    fixed_grid = solver.smatrix(fixed_spectrum)
    manual_fixed_grid = _manual_smatrix_from_fixed_spectrum(solver, fixed_spectrum, energies)
    aligned_grid = solver.smatrix_grid(spectra)

    assert np.allclose(
        np.asarray(fixed_grid),
        np.asarray(manual_fixed_grid),
        atol=1.0e-10,
        rtol=1.0e-10,
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

    assert sparse_solver.spectrum is not None
    assert sparse_solver.smatrix_grid is not None
    assert sparse_solver.interpolate_smatrix is not None
    assert sparse_solver.interpolate_phases is not None
    assert dense_solver.spectrum is not None
    assert dense_solver.smatrix_grid is not None

    sparse_spectra = _spectra(sparse_solver, sparse_energies)
    dense_spectra = _spectra(dense_solver, dense_energies)

    sparse_s = sparse_solver.smatrix_grid(sparse_spectra)
    dense_s = dense_solver.smatrix_grid(dense_spectra)
    sparse_phases = jax.vmap(lm.spectral.phases_from_S)(sparse_s)

    interpolant = sparse_solver.interpolate_smatrix(sparse_s)
    interpolated = jax.vmap(interpolant)(dense_energies)
    phase_interpolant = sparse_solver.interpolate_phases(sparse_phases)
    recovered_phase_knots = jax.vmap(phase_interpolant)(sparse_energies)

    assert np.allclose(np.asarray(interpolated), np.asarray(dense_s), atol=1.0e-4, rtol=1.0e-4)
    assert np.allclose(
        np.asarray(recovered_phase_knots),
        np.asarray(sparse_phases),
        atol=1.0e-10,
        rtol=1.0e-10,
    )


def test_constant_mass_factor_grid_reproduces_scalar_result() -> None:
    """A uniform mass_factor_grid produces bit-identical results to scalar mass_factor."""

    energies = jnp.linspace(0.2, 2.0, 11)
    mu_scalar = HBAR2_2MU

    # Solver compiled with scalar mass_factor (no grid)
    scalar_solver = lm.compile(
        mesh=lm.MeshSpec("legendre", "x", n=12, scale=8.0),
        channels=(lm.ChannelSpec(l=0, threshold=0.0, mass_factor=mu_scalar),),
        solvers=("spectrum", "smatrix", "phases"),
        energies=energies,
        energy_dependent=True,
    )

    # Solver compiled with constant mass_factor_grid = [mu, mu, ..., mu]
    mu_grid = jnp.full(len(energies), mu_scalar)
    grid_solver = lm.compile(
        mesh=lm.MeshSpec("legendre", "x", n=12, scale=8.0),
        channels=(lm.ChannelSpec(l=0, threshold=0.0, mass_factor=mu_scalar),),
        solvers=("spectrum", "smatrix", "phases"),
        energies=energies,
        energy_dependent=True,
        mass_factor_grid=mu_grid,
    )

    assert scalar_solver.spectrum is not None
    assert grid_solver.spectrum is not None
    assert scalar_solver.smatrix_grid is not None
    assert grid_solver.smatrix_grid is not None

    # Scalar path: spectrum uses ChannelSpec.mass_factor
    scalar_spectra = _spectra(scalar_solver, energies)
    scalar_phases = scalar_solver.phases_grid(scalar_spectra)

    # Grid path: mass factor is baked in at compile time; spectrum uses ChannelSpec.mass_factor
    # (which equals mu_scalar), and phases_grid uses the constant mass_factor_grid boundary.
    grid_spectra = _spectra(grid_solver, energies)
    grid_phases = grid_solver.phases_grid(grid_spectra)

    assert np.allclose(
        np.asarray(scalar_phases), np.asarray(grid_phases), atol=1.0e-10, rtol=1.0e-10
    ), "Constant mu_grid must reproduce scalar mass_factor result"


def test_varying_mass_factor_grid_changes_phases() -> None:
    """A non-trivial mu(E) produces different phases from constant mu."""

    energies = jnp.linspace(0.5, 3.0, 10)
    mu_base = HBAR2_2MU
    alpha = 0.05  # 5% variation over the energy range

    # Constant-mu solver
    const_solver = lm.compile(
        mesh=lm.MeshSpec("legendre", "x", n=12, scale=8.0),
        channels=(lm.ChannelSpec(l=0, threshold=0.0, mass_factor=mu_base),),
        solvers=("spectrum", "phases"),
        energies=energies,
        energy_dependent=True,
    )

    # Energy-dependent-mu solver: mu(E) = mu_base * (1 + alpha * E)
    mu_grid = mu_base * (1.0 + alpha * energies)
    mu_solver = lm.compile(
        mesh=lm.MeshSpec("legendre", "x", n=12, scale=8.0),
        channels=(lm.ChannelSpec(l=0, threshold=0.0, mass_factor=mu_base),),
        solvers=("spectrum", "phases"),
        energies=energies,
        energy_dependent=True,
        mass_factor_grid=mu_grid,
    )

    mu_grid_np = np.asarray(mu_grid)

    assert const_solver.spectrum is not None
    assert mu_solver.spectrum is not None
    assert const_solver.phases_grid is not None
    assert mu_solver.phases_grid is not None

    const_spectra = _spectra(const_solver, energies)
    const_phases = const_solver.phases_grid(const_spectra)

    # Mass factors are baked at compile time; spectrum always uses ChannelSpec.mass_factor.
    # The energy-dependent mu enters through the compiled boundary values (k_c, η_c) in
    # phases_grid, so mu_phases differ from const_phases even though the spectra agree.
    mu_spectra = _spectra(mu_solver, energies)
    mu_phases = mu_solver.phases_grid(mu_spectra)

    # Phases must differ — the energy-dependent mu changes the boundary matching (k_c, η_c).
    assert not np.allclose(np.asarray(const_phases), np.asarray(mu_phases), atol=1e-6), (
        "Varying mu(E) should produce different phases from constant mu"
    )
    # But neither result should be NaN
    assert np.all(np.isfinite(np.asarray(mu_phases))), "mu(E) phases must be finite"
    _ = mu_grid_np  # used in construction, checked above

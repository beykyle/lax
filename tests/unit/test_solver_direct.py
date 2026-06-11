from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import lax as lm
from lax.solvers import assemble_block_hamiltonian, build_Q, make_rmatrix_direct_kernel

pytest.importorskip("jax")

HBAR2_2MU = 41.472


def _energy_dep_V(r: jax.Array, E: float) -> jax.Array:
    """Smooth energy-dependent local potential in MeV used across several tests."""
    return (-3.5 * jnp.exp(-((r / 2.4) ** 2)) + 0.02 * E) * HBAR2_2MU


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
    g = jnp.asarray([0.1, 0.2, 0.3, 0.4])
    interaction = solver.interaction_from_array(
        local=[(g, np.ones((1, 1)))], energy_dependent=False
    )

    kernel = make_rmatrix_direct_kernel(
        solver.mesh,
        solver.operators,
        (solver.channels,),
        solver.energies,
        None,
    )
    result = np.asarray(kernel(interaction))

    # Manual computation using MeV form: H_MeV = m_c*(T+L) + V; C = H_MeV − E·I;
    # R = Q'^T C^{-1} Q' / a where Q' = sqrt(m_c) · Q.
    # assemble_block_hamiltonian with (1,1,4) raw array gives the same Hamiltonian.
    potential_raw = jnp.asarray([[[0.1, 0.2, 0.3, 0.4]]])
    hamiltonian = np.asarray(
        assemble_block_hamiltonian(solver.mesh, solver.operators, solver.channels, potential_raw)
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
    assert solver.smatrix_direct is not None
    assert solver.phases_direct is not None
    assert solver.local_potential is not None
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

    V = solver.nonlocal_potential(yamaguchi_kernel)
    spectrum = solver.spectrum(V)

    assert solver.rmatrix is not None
    assert solver.rmatrix_direct is not None

    spectral = np.stack(
        [np.asarray(solver.rmatrix(spectrum, float(energy))) for energy in np.asarray(energies)]
    )
    direct = np.asarray(solver.rmatrix_direct(V))

    assert np.allclose(direct, spectral, atol=1.0e-10, rtol=1.0e-10)


def test_direct_rmatrix_grid_matches_manual_per_energy_solve() -> None:
    """`rmatrix_direct(energy_dep_interaction)` matches a manual per-energy linear solve."""

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

    interaction = solver.local_potential(_energy_dep_V, energy_dependent=True)
    result = np.asarray(solver.rmatrix_direct(interaction))

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
                interaction.block[index],  # (M, M) per-energy block
            )
        )
        # MeV form: C = H_MeV − E·I; R = Q'^T C^{-1} Q' / a.
        matrix = hamiltonian - np.eye(hamiltonian.shape[0]) * energy
        expected.append((q_prime.T @ np.linalg.solve(matrix, q_prime)) / solver.mesh.scale)

    assert np.allclose(result, np.stack(expected), atol=1.0e-10, rtol=1.0e-10)


def test_rmatrix_direct_propagated_rejects_nonlocal_interaction() -> None:
    """Propagated-mesh direct solves reject non-local Interactions with a clear ValueError."""

    propagated_solver = lm.compile(
        mesh=lm.MeshSpec("legendre", "x", n=4, scale=8.0, extras={"n_intervals": 2}),
        channels=(lm.ChannelSpec(l=0, threshold=0.0, mass_factor=2.0),),
        operators=("T+L",),
        solvers=("rmatrix_direct",),
        energies=jnp.asarray([0.3, 0.9]),
        method="linear_solve",
    )

    def nonlocal_kernel(r1: jax.Array, r2: jax.Array) -> jax.Array:
        return -2.0 * jnp.exp(-0.5 * (r1 + r2))

    assert propagated_solver.nonlocal_potential is not None
    assert propagated_solver.rmatrix_direct is not None

    nonlocal_interaction = propagated_solver.nonlocal_potential(nonlocal_kernel)

    with pytest.raises(ValueError, match="Non-local propagated"):
        propagated_solver.rmatrix_direct(nonlocal_interaction)


def test_propagated_nonlocal_direct_rejects_non_interaction() -> None:
    """Propagated direct solves reject non-Interaction inputs with a clear TypeError."""

    propagated_solver = lm.compile(
        mesh=lm.MeshSpec("legendre", "x", n=4, scale=8.0, extras={"n_intervals": 2}),
        channels=(lm.ChannelSpec(l=0, threshold=0.0, mass_factor=2.0),),
        operators=("T+L",),
        solvers=("rmatrix_direct",),
        energies=jnp.asarray([0.3, 0.9]),
        method="linear_solve",
    )

    assert propagated_solver.rmatrix_direct is not None

    with pytest.raises(TypeError, match="Interaction"):
        propagated_solver.rmatrix_direct(jnp.ones((4, 4), dtype=jnp.float64))


def test_direct_grid_observables_match_spectral_grid_for_real_energy_dependent_potential() -> None:
    """Direct `R/S/δ` from `rmatrix_direct(energy_dep)` agree with spectral aligned-grid helpers."""

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
    assert direct_solver.rmatrix_direct is not None
    assert direct_solver.smatrix_direct is not None
    assert direct_solver.phases_direct is not None
    # deprecated aligned-grid observables are no longer wired
    spectral_interaction = spectral_solver.local_potential(_energy_dep_V, energy_dependent=True)
    direct_interaction = direct_solver.local_potential(_energy_dep_V, energy_dependent=True)

    # Spectral path: spectrum() dispatches the energy-dependent Interaction
    # internally over the per-energy block axis (no manual vmap needed).
    spectra = spectral_solver.spectrum(spectral_interaction)

    spectral_r = np.asarray(spectral_solver.rmatrix_grid(spectra))
    spectral_s = np.asarray(spectral_solver.smatrix_grid(spectra))
    spectral_phases = np.asarray(spectral_solver.phases_grid(spectra))

    direct_r = np.asarray(direct_solver.rmatrix_direct(direct_interaction))
    direct_s = np.asarray(direct_solver.smatrix_direct(direct_interaction))
    direct_phases = np.asarray(direct_solver.phases_direct(direct_interaction))

    assert np.allclose(direct_r, spectral_r, atol=1.0e-10, rtol=1.0e-10)
    assert np.allclose(direct_s, spectral_s, atol=1.0e-10, rtol=1.0e-10)
    assert np.allclose(direct_phases, spectral_phases, atol=1.0e-10, rtol=1.0e-10)


# ---------------------------------------------------------------------------
# Task 4: per-channel and energy-dependent mass_factor_grid
# ---------------------------------------------------------------------------


def test_mass_factor_grid_broadcast_scalar_reproduces_uniform() -> None:
    """Scalar mass_factor_grid broadcasts and reproduces the uniform compile."""

    energies = jnp.asarray([0.25, 0.75])
    m = 2.0
    solver_uniform = lm.compile(
        mesh=lm.MeshSpec("legendre", "x", n=4, scale=7.0),
        channels=(lm.ChannelSpec(l=0, threshold=0.0, mass_factor=m),),
        operators=("T+L",),
        solvers=("rmatrix_direct",),
        energies=energies,
        method="linear_solve",
        energy_dependent=True,
    )
    solver_grid = lm.compile(
        mesh=lm.MeshSpec("legendre", "x", n=4, scale=7.0),
        channels=(lm.ChannelSpec(l=0, threshold=0.0, mass_factor=m),),
        operators=("T+L",),
        solvers=("rmatrix_direct",),
        energies=energies,
        method="linear_solve",
        energy_dependent=True,
        mass_factor_grid=jnp.full((2,), m),  # (N_E,) — broadcasts to (N_E, N_c)
    )

    interaction_uniform = solver_uniform.local_potential(_energy_dep_V, energy_dependent=True)
    interaction_grid = solver_grid.local_potential(_energy_dep_V, energy_dependent=True)

    r_uniform = np.asarray(solver_uniform.rmatrix_direct(interaction_uniform))
    r_grid = np.asarray(solver_grid.rmatrix_direct(interaction_grid))

    assert np.allclose(r_uniform, r_grid, atol=1.0e-12, rtol=1.0e-12)


def test_mass_factor_grid_2d_reproduces_uniform() -> None:
    """(N_E, N_c) mass_factor_grid with uniform values reproduces the baseline."""

    energies = jnp.asarray([0.25, 0.75])
    m = 2.0
    solver_uniform = lm.compile(
        mesh=lm.MeshSpec("legendre", "x", n=4, scale=7.0),
        channels=(lm.ChannelSpec(l=0, threshold=0.0, mass_factor=m),),
        operators=("T+L",),
        solvers=("rmatrix_direct",),
        energies=energies,
        method="linear_solve",
        energy_dependent=True,
    )
    solver_grid = lm.compile(
        mesh=lm.MeshSpec("legendre", "x", n=4, scale=7.0),
        channels=(lm.ChannelSpec(l=0, threshold=0.0, mass_factor=m),),
        operators=("T+L",),
        solvers=("rmatrix_direct",),
        energies=energies,
        method="linear_solve",
        energy_dependent=True,
        mass_factor_grid=jnp.full((2, 1), m),  # explicit (N_E, N_c) shape
    )

    interaction_uniform = solver_uniform.local_potential(_energy_dep_V, energy_dependent=True)
    interaction_grid = solver_grid.local_potential(_energy_dep_V, energy_dependent=True)

    r_uniform = np.asarray(solver_uniform.rmatrix_direct(interaction_uniform))
    r_grid = np.asarray(solver_grid.rmatrix_direct(interaction_grid))

    assert np.allclose(r_uniform, r_grid, atol=1.0e-12, rtol=1.0e-12)


def test_mass_factor_grid_rejects_wrong_shape() -> None:
    """mass_factor_grid with mismatched N_E raises ValueError."""

    with pytest.raises(ValueError, match="must equal"):
        lm.compile(
            mesh=lm.MeshSpec("legendre", "x", n=4, scale=7.0),
            channels=(lm.ChannelSpec(l=0, threshold=0.0, mass_factor=2.0),),
            operators=("T+L",),
            solvers=("rmatrix_direct",),
            energies=jnp.asarray([0.5]),
            energy_dependent=True,
            mass_factor_grid=jnp.asarray([2.0, 2.0]),  # wrong length
        )


def test_per_channel_mass_factor_grid_decoupled_matches_single_channel() -> None:
    """Per-channel mass_factor_grid: each diagonal element matches the single-channel case.

    For a block-diagonal (decoupled) potential, R[c,c] depends only on channel
    c's mass factor.  We verify that giving channel 0 mass μ₁ and channel 1
    mass μ₂ reproduces the single-channel result for each channel independently.
    """

    energies = jnp.asarray([0.5, 1.0])
    m0, m1 = 2.0, 5.0
    n = 5
    scale = 7.0

    # Two-channel solver with per-channel mass factors at each energy.
    mu_grid = jnp.tile(jnp.asarray([[m0, m1]]), (len(energies), 1))  # (N_E, 2)
    two_ch = lm.compile(
        mesh=lm.MeshSpec("legendre", "x", n=n, scale=scale),
        channels=(
            lm.ChannelSpec(l=0, threshold=0.0, mass_factor=m0),
            lm.ChannelSpec(l=0, threshold=0.0, mass_factor=m1),
        ),
        operators=("T+L",),
        solvers=("rmatrix_direct",),
        energies=energies,
        method="linear_solve",
        energy_dependent=True,
        mass_factor_grid=mu_grid,
    )

    # Single-channel reference solvers.
    ch0_solver = lm.compile(
        mesh=lm.MeshSpec("legendre", "x", n=n, scale=scale),
        channels=(lm.ChannelSpec(l=0, threshold=0.0, mass_factor=m0),),
        operators=("T+L",),
        solvers=("rmatrix_direct",),
        energies=energies,
        method="linear_solve",
        energy_dependent=True,
    )
    ch1_solver = lm.compile(
        mesh=lm.MeshSpec("legendre", "x", n=n, scale=scale),
        channels=(lm.ChannelSpec(l=0, threshold=0.0, mass_factor=m1),),
        operators=("T+L",),
        solvers=("rmatrix_direct",),
        energies=energies,
        method="linear_solve",
        energy_dependent=True,
    )

    # Decoupled diagonal potentials (energy-independent in value, energy-dependent in API).
    def V_ch0_fn(r: jax.Array, E: float) -> jax.Array:
        return -0.5 * jnp.exp(-((r / 2.5) ** 2)) * m0

    def V_ch1_fn(r: jax.Array, E: float) -> jax.Array:
        return -0.3 * jnp.exp(-((r / 3.0) ** 2)) * m1

    A0 = np.array([[1.0, 0.0], [0.0, 0.0]])
    A1 = np.array([[0.0, 0.0], [0.0, 1.0]])

    V_two = two_ch.local_potential(
        V_ch0_fn, coupling=A0, energy_dependent=True
    ) + two_ch.local_potential(V_ch1_fn, coupling=A1, energy_dependent=True)
    V_ch0 = ch0_solver.local_potential(V_ch0_fn, energy_dependent=True)
    V_ch1 = ch1_solver.local_potential(V_ch1_fn, energy_dependent=True)

    r_two = np.asarray(two_ch.rmatrix_direct(V_two))  # (N_E, 2, 2)
    r_ch0 = np.asarray(ch0_solver.rmatrix_direct(V_ch0))  # (N_E, 1, 1)
    r_ch1 = np.asarray(ch1_solver.rmatrix_direct(V_ch1))  # (N_E, 1, 1)

    # Diagonal elements of decoupled two-channel solver must match single-channel results.
    assert np.allclose(r_two[:, 0, 0], r_ch0[:, 0, 0], atol=1.0e-10, rtol=1.0e-10)
    assert np.allclose(r_two[:, 1, 1], r_ch1[:, 0, 0], atol=1.0e-10, rtol=1.0e-10)
    # Off-diagonal must be zero for a decoupled potential.
    assert np.allclose(r_two[:, 0, 1], 0.0, atol=1.0e-10)
    assert np.allclose(r_two[:, 1, 0], 0.0, atol=1.0e-10)


# ---------------------------------------------------------------------------
# Task 5: wavefunction_direct round-trip vs spectral wavefunction
# ---------------------------------------------------------------------------


def test_wavefunction_direct_matches_spectral_wavefunction() -> None:
    """wavefunction_direct(interaction, source, i) equals spectral wavefunction(spec, E, source)."""

    alpha = 0.2316053
    beta = 1.3918324
    energies = jnp.asarray([1.0, 5.0])

    def yamaguchi_kernel(r1: jax.Array, r2: jax.Array) -> jax.Array:
        return -2.0 * beta * (alpha + beta) ** 2 * jnp.exp(-beta * (r1 + r2)) * HBAR2_2MU

    solver = lm.compile(
        mesh=lm.MeshSpec("legendre", "x", n=10, scale=8.0),
        channels=(lm.ChannelSpec(l=0, threshold=0.0, mass_factor=HBAR2_2MU),),
        operators=("T+L",),
        solvers=("spectrum", "wavefunction", "rmatrix_direct"),
        energies=energies,
    )

    assert solver.wavefunction is not None
    assert solver.wavefunction_direct is not None
    assert solver.nonlocal_potential is not None

    V = solver.nonlocal_potential(yamaguchi_kernel)
    spec = solver.spectrum(V)

    for energy_index in range(len(energies)):
        energy = float(energies[energy_index])
        src = lm.make_wavefunction_source(solver, channel_index=0, energy_index=energy_index)

        psi_spec = np.asarray(solver.wavefunction(spec, energy, src))
        psi_dir = np.asarray(solver.wavefunction_direct(V, src, energy_index))

        assert np.allclose(psi_spec, psi_dir, atol=1.0e-10, rtol=1.0e-10), (
            f"wavefunction_direct mismatch at energy_index={energy_index}"
        )


def test_wavefunction_under_linear_solve_binds_direct_kernel() -> None:
    """Under `method="linear_solve"`, `"wavefunction"` binds the direct kernel, not a spectral path."""

    solver = lm.compile(
        mesh=lm.MeshSpec("legendre", "x", n=10, scale=8.0),
        channels=(lm.ChannelSpec(l=0, threshold=0.0, mass_factor=HBAR2_2MU),),
        solvers=("rmatrix_direct", "wavefunction"),
        method="linear_solve",
        energies=jnp.asarray([0.5, 1.0, 2.0]),
    )
    assert solver.wavefunction_direct is not None
    # No spectral path exists under linear_solve.
    assert solver.wavefunction is None
    assert solver.spectrum is None


def test_wavefunction_only_under_linear_solve_binds_direct_kernel() -> None:
    """`solvers=("wavefunction",), method="linear_solve"` binds `wavefunction_direct`, no spectrum."""

    solver = lm.compile(
        mesh=lm.MeshSpec("legendre", "x", n=10, scale=8.0),
        channels=(lm.ChannelSpec(l=0, threshold=0.0, mass_factor=HBAR2_2MU),),
        solvers=("wavefunction",),
        method="linear_solve",
        energies=jnp.asarray([0.5, 1.0, 2.0]),
    )
    assert solver.wavefunction_direct is not None
    assert solver.wavefunction is None

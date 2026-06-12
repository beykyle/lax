"""F2 acceptance tests: wavefunction grids (spec v0.1.5.1, T6/T7/T8/T9-C2)."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import lax
from lax.spectral.types import Spectrum
from tests.unit._blocks_helpers import HBAR2_2MU, gaussian_kernel, partial_wave_groups

N = 10
RADIUS = 8.0
ENERGIES = jnp.linspace(2.0, 10.0, 4)
N_E = len(ENERGIES)
MESH = lax.MeshSpec("legendre", "x", n=N, scale=RADIUS)
WAVE = ("spectrum", "wavefunction")

TIGHT = dict(rtol=1e-10, atol=1e-12)


def _real_profile(r: jnp.ndarray) -> jnp.ndarray:
    return -10.0 * jnp.exp(-0.2 * r**2)


def _complex_profile(r: jnp.ndarray) -> jnp.ndarray:
    return (-10.0 - 4.0j) * jnp.exp(-0.2 * r**2)


def _channels_solver(
    threshold: float = 0.0,
    z1z2: tuple[int, int] | None = None,
    V_is_complex: bool = False,
    method: str | None = None,
) -> lax.Solver:
    return lax.compile(
        mesh=MESH,
        channels=(lax.ChannelSpec(l=0, threshold=threshold, mass_factor=HBAR2_2MU),),
        solvers=WAVE if method != "linear_solve" else ("rmatrix_direct",),
        energies=ENERGIES,
        z1z2=z1z2,
        V_is_complex=V_is_complex,
        method=method,
    )


def _loop_reference(
    solver: lax.Solver,
    spectrum: Spectrum,
    energy_batched: bool = False,
) -> np.ndarray:
    """The §4.1 reference loop: stack the single-energy wavefunction per index."""

    assert solver.wavefunction is not None
    psi = []
    for e in range(N_E):
        spec_e = (
            jax.tree_util.tree_map(lambda x, e=e: x[e], spectrum) if energy_batched else spectrum
        )
        source = lax.make_wavefunction_source(solver, channel_index=0, energy_index=e)
        psi.append(np.asarray(solver.wavefunction(spec_e, ENERGIES[e], source)))
    return np.stack(psi, axis=-2)


# ---------------------------------------------------------------------------
# T6 — wavefunction_grid ≡ looped wavefunction


@pytest.mark.parametrize(
    ("z1z2", "profile", "v_is_complex"),
    [
        (None, _real_profile, False),  # neutral, Hermitian metric (eigh)
        ((1, 20), _real_profile, False),  # charged, Hermitian metric
        (None, _complex_profile, True),  # neutral, complex-symmetric metric (eig)
    ],
    ids=["neutral-eigh", "charged-eigh", "neutral-eig"],
)
def test_static_regime_matches_loop(
    z1z2: tuple[int, int] | None,
    profile: Callable[[jnp.ndarray], jnp.ndarray],
    v_is_complex: bool,
) -> None:
    solver = _channels_solver(z1z2=z1z2, V_is_complex=v_is_complex)
    assert solver.local_potential is not None
    assert solver.wavefunction_grid is not None
    spectrum = solver.spectrum(solver.local_potential(profile))  # type: ignore[misc]

    psi_grid = solver.wavefunction_grid(spectrum)
    assert psi_grid.shape == (N_E, N)
    np.testing.assert_allclose(np.asarray(psi_grid), _loop_reference(solver, spectrum), **TIGHT)


def test_energy_dependent_regime_matches_loop() -> None:
    solver = _channels_solver()
    assert solver.interaction_from_array is not None
    assert solver.wavefunction_grid is not None
    radii = solver.mesh.radii
    profiles = jnp.stack([(-10.0 - 0.5 * float(e)) * jnp.exp(-0.2 * radii**2) for e in range(N_E)])
    interaction = solver.interaction_from_array(
        local=[(profiles, np.ones((1, 1)))], energy_dependent=True
    )
    spectra = solver.spectrum(interaction)  # type: ignore[misc]
    assert spectra.eigenvalues.shape == (N_E, N)

    psi_grid = solver.wavefunction_grid(spectra)
    np.testing.assert_allclose(
        np.asarray(psi_grid),
        _loop_reference(solver, spectra, energy_batched=True),
        **TIGHT,
    )


def test_closed_channel_entries_are_defined_and_loop_equivalent() -> None:
    """The grid straddles the threshold: closed entries are defined, not masked."""

    solver = _channels_solver(threshold=5.0)
    assert solver.boundary is not None
    is_open = np.asarray(solver.boundary.is_open)[:, 0]
    assert is_open.any() and not is_open.all()  # the grid genuinely straddles

    assert solver.local_potential is not None
    assert solver.wavefunction_grid is not None
    spectrum = solver.spectrum(solver.local_potential(_real_profile))  # type: ignore[misc]
    psi_grid = solver.wavefunction_grid(spectrum)
    assert bool(jnp.all(jnp.isfinite(psi_grid)))
    np.testing.assert_allclose(np.asarray(psi_grid), _loop_reference(solver, spectrum), **TIGHT)


def test_channel_index_none_stacks_all_incoming_channels() -> None:
    channels = (
        lax.ChannelSpec(l=0, threshold=0.0, mass_factor=HBAR2_2MU),
        lax.ChannelSpec(l=2, threshold=1.0, mass_factor=HBAR2_2MU),
    )
    solver = lax.compile(
        mesh=MESH,
        channels=channels,
        solvers=WAVE,
        energies=ENERGIES,
    )
    assert solver.nonlocal_potential is not None
    assert solver.wavefunction_grid is not None
    coupling = np.asarray([[1.0, 0.2], [0.2, 0.8]])
    interaction = solver.nonlocal_potential(gaussian_kernel(8.0), coupling=coupling)
    spectrum = solver.spectrum(interaction)  # type: ignore[misc]

    stacked = solver.wavefunction_grid(spectrum, channel_index=None)
    assert stacked.shape == (N_E, 2, 2 * N)
    for c in range(2):
        np.testing.assert_allclose(
            np.asarray(stacked[:, c]),
            np.asarray(solver.wavefunction_grid(spectrum, channel_index=c)),
            **TIGHT,
        )


def test_eigenvector_free_spectrum_raises() -> None:
    bare = lax.compile(
        mesh=MESH,
        channels=(lax.ChannelSpec(l=0, threshold=0.0, mass_factor=HBAR2_2MU),),
        solvers=("spectrum", "smatrix", "wavefunction"),
        energies=ENERGIES,
    )
    no_vectors = lax.compile(
        mesh=MESH,
        channels=(lax.ChannelSpec(l=0, threshold=0.0, mass_factor=HBAR2_2MU),),
        solvers=("spectrum", "smatrix"),
        energies=ENERGIES,
    )
    assert no_vectors.local_potential is not None
    assert bare.wavefunction_grid is not None
    spectrum = no_vectors.spectrum(no_vectors.local_potential(_real_profile))  # type: ignore[misc]
    with pytest.raises(RuntimeError, match="Eigenvectors were not retained"):
        bare.wavefunction_grid(spectrum)


# ---------------------------------------------------------------------------
# T7 — blocks equivalence


def test_blocked_wavefunction_grid_matches_per_block_solvers() -> None:
    block_groups = partial_wave_groups()
    n_b = len(block_groups)
    solver = lax.compile(mesh=MESH, blocks=block_groups, solvers=WAVE, energies=ENERGIES)
    assert solver.nonlocal_potential is not None
    assert solver.wavefunction_grid is not None
    kernels = [gaussian_kernel(10.0 * (b + 1)) for b in range(n_b)]
    interaction = solver.nonlocal_potential(kernels, block_dependent=True)
    spectrum = solver.spectrum(interaction)  # type: ignore[misc]

    psi_grid = solver.wavefunction_grid(spectrum)
    assert psi_grid.shape == (n_b, N_E, N)
    for b, group in enumerate(block_groups):
        single = lax.compile(mesh=MESH, channels=group, solvers=WAVE, energies=ENERGIES)
        assert single.nonlocal_potential is not None
        assert single.wavefunction_grid is not None
        single_spectrum = single.spectrum(single.nonlocal_potential(kernels[b]))  # type: ignore[misc]
        np.testing.assert_allclose(
            np.asarray(psi_grid[b]),
            np.asarray(single.wavefunction_grid(single_spectrum)),
            err_msg=f"wavefunction grid block {b}",
            **TIGHT,
        )


# ---------------------------------------------------------------------------
# T8 — direct-path grid


def test_direct_grid_matches_looped_wavefunction_direct() -> None:
    solver = _channels_solver(method="linear_solve")
    assert solver.local_potential is not None
    assert solver.wavefunction_direct is not None
    assert solver.wavefunction_direct_grid is not None
    interaction = solver.local_potential(_real_profile)

    psi_grid = solver.wavefunction_direct_grid(interaction)
    assert psi_grid.shape == (N_E, N)
    for e in range(N_E):
        source = lax.make_wavefunction_source(solver, channel_index=0, energy_index=e)
        np.testing.assert_allclose(
            np.asarray(psi_grid[e]),
            np.asarray(solver.wavefunction_direct(interaction, source, e)),
            err_msg=f"direct grid energy {e}",
            **TIGHT,
        )


def test_spectral_and_direct_grids_agree() -> None:
    spectral_solver = _channels_solver()
    direct_solver = _channels_solver(method="linear_solve")
    assert spectral_solver.local_potential is not None
    assert direct_solver.local_potential is not None
    assert spectral_solver.wavefunction_grid is not None
    assert direct_solver.wavefunction_direct_grid is not None

    spectrum = spectral_solver.spectrum(  # type: ignore[misc]
        spectral_solver.local_potential(_real_profile)
    )
    np.testing.assert_allclose(
        np.asarray(spectral_solver.wavefunction_grid(spectrum)),
        np.asarray(
            direct_solver.wavefunction_direct_grid(direct_solver.local_potential(_real_profile))
        ),
        rtol=1e-8,
        atol=1e-10,
    )


def test_blocked_direct_grid_matches_per_block() -> None:
    block_groups = partial_wave_groups((0, 1))
    solver = lax.compile(
        mesh=MESH,
        blocks=block_groups,
        solvers=("rmatrix_direct",),
        energies=ENERGIES,
        method="linear_solve",
    )
    assert solver.nonlocal_potential is not None
    assert solver.wavefunction_direct_grid is not None
    kernels = [gaussian_kernel(6.0 * (b + 1)) for b in range(len(block_groups))]
    interaction = solver.nonlocal_potential(kernels, block_dependent=True)
    psi_grid = solver.wavefunction_direct_grid(interaction)
    assert psi_grid.shape == (len(block_groups), N_E, N)

    for b, group in enumerate(block_groups):
        single = lax.compile(
            mesh=MESH,
            channels=group,
            solvers=("rmatrix_direct",),
            energies=ENERGIES,
            method="linear_solve",
        )
        assert single.nonlocal_potential is not None
        assert single.wavefunction_direct_grid is not None
        np.testing.assert_allclose(
            np.asarray(psi_grid[b]),
            np.asarray(single.wavefunction_direct_grid(single.nonlocal_potential(kernels[b]))),
            err_msg=f"direct grid block {b}",
            **TIGHT,
        )


# ---------------------------------------------------------------------------
# T9-C2 — propagated-mesh guard


def test_propagated_mesh_wavefunction_raises() -> None:
    with pytest.raises(NotImplementedError, match="propagated multi-interval"):
        lax.compile(
            mesh=lax.MeshSpec("legendre", "x", n=N, scale=RADIUS, extras={"n_intervals": 2}),
            channels=(lax.ChannelSpec(l=0, threshold=0.0, mass_factor=HBAR2_2MU),),
            solvers=("rmatrix_direct", "wavefunction"),
            energies=ENERGIES,
            method="linear_solve",
        )


# ---------------------------------------------------------------------------
# Pickle


def test_wavefunction_grids_round_trip_through_pickle() -> None:
    import pickle

    spectral_solver = _channels_solver()
    direct_solver = _channels_solver(method="linear_solve")
    restored_spectral = pickle.loads(pickle.dumps(spectral_solver))
    restored_direct = pickle.loads(pickle.dumps(direct_solver))

    assert spectral_solver.local_potential is not None
    interaction = spectral_solver.local_potential(_real_profile)
    spectrum = spectral_solver.spectrum(interaction)  # type: ignore[misc]
    np.testing.assert_allclose(
        np.asarray(restored_spectral.wavefunction_grid(spectrum)),
        np.asarray(spectral_solver.wavefunction_grid(spectrum)),
        **TIGHT,
    )
    assert direct_solver.local_potential is not None
    d_interaction = direct_solver.local_potential(_real_profile)
    np.testing.assert_allclose(
        np.asarray(restored_direct.wavefunction_direct_grid(d_interaction)),
        np.asarray(direct_solver.wavefunction_direct_grid(d_interaction)),
        **TIGHT,
    )

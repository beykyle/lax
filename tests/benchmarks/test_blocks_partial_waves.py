"""Partial-wave symmetry-block batching benchmark (DESIGN.md §15.5, Example 16.9).

Validates the ``blocks=`` axis on physically meaningful single-channel
(``N_c == 1``) partial-wave sets:

* the published α + ²⁰⁸Pb collision matrix (Descouvemont Appendix A, ℓ = 20)
  is reproduced by the corresponding slice of a multi-ℓ blocks compile,
  confirming the per-block boundary ``F_ℓ, G_ℓ``;
* a pure-Coulomb (zero nuclear potential) blocks compile yields vanishing
  nuclear phase shifts in every partial wave;
* the partial-wave-summed elastic cross section from one batched call equals
  the explicit per-ℓ compile loop.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import lax as lm
from tests.benchmarks._reference_cases import ALPHA_PB_REFERENCE_A14_N60_NS1
from tests.conftest import LEGACY_COULOMB_E2

pytest.importorskip("jax")

pytestmark = pytest.mark.usefixtures("legacy_coulomb_constant")

ALPHA_PB_MASS_FACTOR = 20.736 / (4.0 * 208.0 / (4.0 + 208.0))
NEUTRON_MASS_FACTOR = 41.47


def _alpha_pb_optical_potential(r: jax.Array) -> jax.Array:
    """α + ²⁰⁸Pb optical potential (Descouvemont eq. 47), imag depth 10 MeV."""

    v0 = 100.0
    radius = 1.1132 * (208.0 ** (1.0 / 3.0) + 4.0 ** (1.0 / 3.0))
    diffuseness = 0.5803
    woods_saxon = 1.0 / (1.0 + jnp.exp((r - radius) / diffuseness))
    coulomb = 2.0 * 82.0 * LEGACY_COULOMB_E2 / r
    return -v0 * woods_saxon - 1.0j * 10.0 * woods_saxon + coulomb


@pytest.mark.benchmark
def test_blocks_alpha_pb_slice_matches_published_collision_matrix() -> None:
    """The ℓ = 20 slice of a multi-ℓ blocks compile reproduces Appendix A."""

    reference = ALPHA_PB_REFERENCE_A14_N60_NS1
    ells = (18, 20, 22)
    solver = lm.compile(
        mesh=lm.MeshSpec("legendre", "x", n=reference.n_basis, scale=reference.scale),
        blocks=[
            [lm.ChannelSpec(l=ell, threshold=0.0, mass_factor=ALPHA_PB_MASS_FACTOR)] for ell in ells
        ],
        solvers=("rmatrix_direct",),
        energies=reference.energies,
        V_is_complex=True,
        method="linear_solve",
        z1z2=(2, 82),
    )
    interaction = solver.local_potential(_alpha_pb_optical_potential)
    assert solver.smatrix_direct is not None
    smatrix = solver.smatrix_direct(interaction)
    assert smatrix.shape == (len(ells), len(reference.energies), 1, 1)
    assert np.allclose(
        np.asarray(smatrix[ells.index(20), :, 0, 0]),
        reference.collision_matrix,
        atol=1.0e-4,
        rtol=1.0e-4,
    )


@pytest.mark.benchmark
def test_blocks_pure_coulomb_nuclear_phases_vanish() -> None:
    """Zero nuclear potential → S = 1 (relative to Coulomb) in every ℓ block.

    Any error in the per-block ℓ-dependent boundary ``F_ℓ, G_ℓ`` would show up
    as a spurious nuclear phase shift.
    """

    ells = tuple(range(6))
    energies = jnp.linspace(5.0, 40.0, 4)
    solver = lm.compile(
        mesh=lm.MeshSpec("legendre", "x", n=40, scale=12.0),
        blocks=[
            [lm.ChannelSpec(l=ell, threshold=0.0, mass_factor=ALPHA_PB_MASS_FACTOR)] for ell in ells
        ],
        solvers=("rmatrix_direct",),
        energies=energies,
        z1z2=(2, 82),
    )
    # The Coulomb tail itself must stay in the interaction (the boundary
    # matching assumes a pure Coulomb exterior, and the same 2·Z₁Z₂e²/r tail
    # is part of the interior Hamiltonian).
    interaction = solver.local_potential(lambda r: 2.0 * 82.0 * LEGACY_COULOMB_E2 / r)
    assert solver.phases_direct is not None
    phases = solver.phases_direct(interaction)
    assert phases.shape == (len(ells), len(energies), 1)
    np.testing.assert_allclose(np.asarray(phases), 0.0, atol=1.0e-7)


@pytest.mark.benchmark
def test_blocks_partial_wave_cross_section_matches_per_ell_loop() -> None:
    """One batched call reproduces the per-ℓ compile loop's summed cross section."""

    ells = tuple(range(9))
    energies = jnp.linspace(1.0, 20.0, 5)
    mesh = lm.MeshSpec("legendre", "x", n=40, scale=12.0)

    def woods_saxon(r: jax.Array) -> jax.Array:
        return -45.0 / (1.0 + jnp.exp((r - 4.0) / 0.65))

    solver = lm.compile(
        mesh=mesh,
        blocks=[
            [lm.ChannelSpec(l=ell, threshold=0.0, mass_factor=NEUTRON_MASS_FACTOR)] for ell in ells
        ],
        solvers=("rmatrix_direct",),
        energies=energies,
    )
    interaction = solver.local_potential(woods_saxon)
    assert solver.phases_direct is not None
    phases_blocks = np.asarray(solver.phases_direct(interaction))[:, :, 0]  # (N_b, N_E)

    phases_loop = []
    for ell in ells:
        single = lm.compile(
            mesh=mesh,
            channels=(lm.ChannelSpec(l=ell, threshold=0.0, mass_factor=NEUTRON_MASS_FACTOR),),
            solvers=("rmatrix_direct",),
            energies=energies,
        )
        assert single.phases_direct is not None
        phases_loop.append(
            np.asarray(single.phases_direct(single.local_potential(woods_saxon)))[:, 0]
        )
    phases_loop = np.stack(phases_loop)

    np.testing.assert_allclose(phases_blocks, phases_loop, rtol=1e-10, atol=1e-12)

    # Partial-wave-summed elastic cross section σ(E) = 4π/k² Σ_ℓ (2ℓ+1) sin²δ_ℓ.
    k_squared = np.asarray(energies) / NEUTRON_MASS_FACTOR  # fm⁻²
    weights = 2.0 * np.asarray(ells) + 1.0
    sigma_blocks = 4.0 * np.pi / k_squared * (weights[:, None] * np.sin(phases_blocks) ** 2).sum(0)
    sigma_loop = 4.0 * np.pi / k_squared * (weights[:, None] * np.sin(phases_loop) ** 2).sum(0)
    np.testing.assert_allclose(sigma_blocks, sigma_loop, rtol=1e-10)
    assert np.all(sigma_blocks > 0.0)

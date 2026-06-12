"""T5 (spec v0.1.5.1): cross-engine DWBA normalization anchor — fixture-gated.

Compares ``matrix_element(χp, χn, U₁) / (a·k_p·k_n)`` built from lax
wavefunctions against reference-engine node-sum ``T_lj`` golden data per
``(partial wave, energy)``.  The fixture is exported from an independent
reference R-matrix engine; this test activates automatically once the file is
dropped into ``tests/acceptance/data/dwba_pn_reference.npz``.

Fixture schema (``np.savez``):

==================  =========================  =====================================
key                 shape                      meaning
==================  =========================  =====================================
``energies``        ``(N_E,)``                 entrance CM energies, MeV
``exit_energies``   ``(N_E,)``                 exit CM energies, MeV
``ls``              ``(N_pw,)`` int            orbital ℓ per (ℓ, j) partial wave
``channel_radius``  scalar                     channel radius a, fm
``nbasis``          scalar int                 Lagrange-Legendre-x mesh size N
``mass_factor_p``   scalar                     entrance ℏ²/2μ, MeV·fm²
``mass_factor_n``   scalar                     exit ℏ²/2μ, MeV·fm²
``z1z2_p``          ``(2,)`` int               entrance charges (exit is neutral)
``U_p``, ``U_n``    ``(N_pw, N)`` complex      distorting potentials sampled on the
                                               lax mesh radii, MeV
``U1``              ``(N_pw, N)`` complex      isovector transition operator node
                                               samples, MeV (local)
``T_ref``           ``(N_pw, N_E)`` complex    reference node-sum T_lj
``rtol``/``atol``   scalars                    comparison tolerances
==================  =========================  =====================================

The C7 convention factor: the test first resolves the (single, constant)
relative normalization between the two engines from the data, asserts it is
constant across every ``(partial wave, energy)``, and then compares.  Once the
factor is pinned it must be documented on ``wavefunction_grid`` and frozen
here as an explicit constant.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import lax

FIXTURE = Path(__file__).parent / "data" / "dwba_pn_reference.npz"

pytestmark = pytest.mark.skipif(
    not FIXTURE.exists(),
    reason="reference-engine DWBA fixture not yet exported (see module docstring)",
)


def test_matrix_element_reproduces_reference_node_sum() -> None:
    import jax.numpy as jnp

    data = np.load(FIXTURE)
    energies = jnp.asarray(data["energies"])
    exit_energies = jnp.asarray(data["exit_energies"])
    ls = [int(ell) for ell in data["ls"]]
    radius = float(data["channel_radius"])
    nbasis = int(data["nbasis"])
    mesh = lax.MeshSpec("legendre", "x", n=nbasis, scale=radius)

    proton = lax.compile(
        mesh=mesh,
        blocks=[
            (lax.ChannelSpec(l=ell, threshold=0.0, mass_factor=float(data["mass_factor_p"])),)
            for ell in ls
        ],
        solvers=("spectrum", "wavefunction"),
        energies=energies,
        z1z2=(int(data["z1z2_p"][0]), int(data["z1z2_p"][1])),
        V_is_complex=True,
        method="eig",
    )
    neutron = lax.compile(
        mesh=mesh,
        blocks=[
            (lax.ChannelSpec(l=ell, threshold=0.0, mass_factor=float(data["mass_factor_n"])),)
            for ell in ls
        ],
        solvers=("spectrum", "wavefunction"),
        energies=exit_energies,
        V_is_complex=True,
        method="eig",
    )
    assert proton.interaction_from_array is not None
    assert neutron.interaction_from_array is not None
    assert proton.matrix_element is not None
    assert proton.wavefunction_grid is not None
    assert neutron.wavefunction_grid is not None
    assert proton.boundary is not None
    assert neutron.boundary is not None

    coupling = np.ones((1, 1))
    v_p = proton.interaction_from_array(
        local=[(jnp.asarray(data["U_p"]), coupling)], block_dependent=True
    )
    v_n = neutron.interaction_from_array(
        local=[(jnp.asarray(data["U_n"]), coupling)], block_dependent=True
    )
    u1 = proton.interaction_from_array(
        local=[(jnp.asarray(data["U1"]), coupling)], block_dependent=True
    )

    chi_p = proton.wavefunction_grid(proton.spectrum(v_p))  # type: ignore[misc]
    chi_n = neutron.wavefunction_grid(neutron.spectrum(v_n))  # type: ignore[misc]
    k_p = proton.boundary.k[:, :, 0]  # (N_pw, N_E)
    k_n = neutron.boundary.k[:, :, 0]

    element = proton.matrix_element(chi_p, chi_n, u1, conjugate=False)
    t_lax = np.asarray(element / (radius * k_p * k_n))
    t_ref = np.asarray(data["T_ref"])

    # C7: resolve the constant cross-engine convention factor from the data,
    # then require it to be the SAME for every (partial wave, energy).
    ratio = t_ref / t_lax
    factor = ratio.flat[np.argmax(np.abs(t_ref))]
    np.testing.assert_allclose(
        ratio,
        factor,
        rtol=float(data["rtol"]),
        atol=float(data["atol"]),
        err_msg="cross-engine convention factor is not constant (C7)",
    )
    np.testing.assert_allclose(
        t_lax * factor,
        t_ref,
        rtol=float(data["rtol"]),
        atol=float(data["atol"]),
    )

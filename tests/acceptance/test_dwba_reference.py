"""T5 (spec v0.1.5.1): cross-engine DWBA normalization anchor against jitr.

Reference data: jitr's ``xs.quasielastic_pn`` engine on the ⁴⁸Ca(p,n)⁴⁸Sc IAS
case (E_lab = 35 MeV, KDUQ defaults, lmax = 20, a = 16 fm, nbasis = 35) with
**classical kinematics**, exported by ``generate_dwba_reference.py`` (run it
inside the jitr virtualenv to regenerate).  The semi-relativistic variant of
the same case is validated against CHEX and Jon et al. data in jitr's
``examples/notebooks/chex_jitr_validation.ipynb``.

Classical kinematics make ``ħ²k²/2μ = Ecm`` and ``η = αZz·μ/(ħk)`` exact
identities, so each jitr channel maps onto lax with
``mass_factor = ħ²c²/(2μ)`` and ``energies = [Ecm]`` — no potential rescaling.

Two stages:

1. **S-matrix (normalization-free):** lax's per-(l, j) elastic S-matrices must
   reproduce jitr's ``Sp``/``Sn`` directly — this validates the whole
   convention mapping (grid, η, k, potential scale) independent of
   wavefunction normalization.
2. **T-matrix (C7):** ``matrix_element(χp, χn, U₁)/(a·k_p·k_n)`` must
   reproduce jitr's node-sum ``T_lj`` after two documented conversions:

   * **Source convention (per row):** jitr's internal wavefunction is driven
     by the matched exterior *derivative*,
     ``u_ext'(a) = (i/2)(H⁻' − S_lj·H⁺')`` (``rmatrix/core.py``,
     ``solution_coeffs_with_inverse``), while lax's is driven by the
     boundary *value* ``H⁻(a)``.  The driven solution is linear in that
     scalar, so each (l, j, channel) converts by the exact factor
     ``(i/2)(H⁻' − S·H⁺')/H⁻`` — built here from lax's own boundary cache
     and S-matrices.
   * **Global constant (C7 proper):** what remains after the per-row
     conversion is exactly ``k_p·k_n/a`` (verified to 4e-14): jitr's s = k·r
     coefficients relate to lax's r-space coefficients per channel as
     ``x_c = (k_c/√a)·[(i/2)(H⁻' − S·H⁺')/H⁻]·χ_c``, so

         T_jitr = conv_p·conv_n·matrix_element(χp, χn, U₁) / a²

     The factor is frozen analytically below and asserted constant across
     every (l, j).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import lax
import lax.constants as constants

FIXTURE = Path(__file__).parent / "data" / "dwba_pn_reference.npz"

pytestmark = pytest.mark.skipif(
    not FIXTURE.exists(),
    reason="jitr DWBA reference fixture not yet generated (see generate_dwba_reference.py)",
)


def _compile_channel(
    data: np.lib.npyio.NpzFile,
    ecm: float,
    mu: float,
    z1z2: tuple[int, int] | None,
) -> lax.Solver:
    import jax.numpy as jnp

    mass_factor = constants.HBARC**2 / (2.0 * mu)
    return lax.compile(
        mesh=lax.MeshSpec(
            "legendre", "x", n=int(data["nbasis"]), scale=float(data["channel_radius"])
        ),
        blocks=[
            (lax.ChannelSpec(l=int(ell), threshold=0.0, mass_factor=mass_factor),)
            for ell in data["ls"]
        ],
        solvers=("spectrum", "smatrix", "wavefunction"),
        energies=jnp.asarray([ecm]),
        z1z2=z1z2,
        V_is_complex=True,
        method="eig",
    )


def test_matrix_element_reproduces_jitr_node_sum() -> None:
    import jax.numpy as jnp

    data = np.load(FIXTURE)
    radius = float(data["channel_radius"])
    k_p = float(data["k_p"])
    k_n = float(data["k_n"])
    n_pw = len(data["ls"])

    # Classical-kinematics identity the mapping relies on.
    for ecm, mu, k in (
        (float(data["Ecm_p"]), float(data["mu_p"]), k_p),
        (float(data["Ecm_n"]), float(data["mu_n"]), k_n),
    ):
        assert np.isclose(constants.HBARC**2 / (2.0 * mu) * k**2, ecm, rtol=1e-10)

    proton = _compile_channel(
        data,
        float(data["Ecm_p"]),
        float(data["mu_p"]),
        (int(data["z1z2_p"][0]), int(data["z1z2_p"][1])),
    )
    neutron = _compile_channel(data, float(data["Ecm_n"]), float(data["mu_n"]), None)

    # Grid and wavenumber identity with the jitr engine.
    np.testing.assert_allclose(
        np.asarray(proton.mesh.radii), np.asarray(data["rgrid"]), rtol=1e-12, atol=1e-12
    )
    assert proton.boundary is not None and neutron.boundary is not None
    np.testing.assert_allclose(np.asarray(proton.boundary.k[:, 0, 0]), k_p, rtol=1e-10)
    np.testing.assert_allclose(np.asarray(neutron.boundary.k[:, 0, 0]), k_n, rtol=1e-10)

    assert proton.interaction_from_array is not None
    assert neutron.interaction_from_array is not None
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

    spectrum_p = proton.spectrum(v_p)  # type: ignore[misc]
    spectrum_n = neutron.spectrum(v_n)  # type: ignore[misc]

    # ------------------------------------------------------------------
    # Stage 1 — normalization-free S-matrix regression per (l, j).
    assert proton.smatrix is not None and neutron.smatrix is not None
    s_p = np.asarray(proton.smatrix(spectrum_p))[:, 0, 0, 0]
    s_n = np.asarray(neutron.smatrix(spectrum_n))[:, 0, 0, 0]
    # Measured cross-engine agreement is ~8e-13 absolute; assert with margin.
    np.testing.assert_allclose(s_p, np.asarray(data["S_p"]), rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(s_n, np.asarray(data["S_n"]), rtol=1e-9, atol=1e-10)

    # ------------------------------------------------------------------
    # Stage 2 — DWBA bilinear vs jitr's node sum, up to one constant (C7).
    assert proton.wavefunction_grid is not None
    assert neutron.wavefunction_grid is not None
    assert proton.matrix_element is not None
    chi_p = proton.wavefunction_grid(spectrum_p)  # (N_pw, 1, M)
    chi_n = neutron.wavefunction_grid(spectrum_n)
    element = proton.matrix_element(chi_p, chi_n, u1, conjugate=False)  # (N_pw, 1)
    t_lax = np.asarray(element)[:, 0] / (radius * k_p * k_n)
    t_ref = np.asarray(data["T_ref"])

    # Source-convention conversion per (l, j, channel): jitr drives with
    # u_ext'(a) = (i/2)(H-' - S H+'); lax drives with H-(a).
    def source_conversion(solver: lax.Solver, s_matrix: np.ndarray) -> np.ndarray:
        assert solver.boundary is not None
        h_minus = np.asarray(solver.boundary.H_minus)[:, 0, 0]
        h_minus_p = np.asarray(solver.boundary.H_minus_p)[:, 0, 0]
        h_plus_p = np.asarray(solver.boundary.H_plus_p)[:, 0, 0]
        return 0.5j * (h_minus_p - s_matrix * h_plus_p) / h_minus

    t_lax = t_lax * source_conversion(proton, s_p) * source_conversion(neutron, s_n)

    # The frozen C7 constant: k_p·k_n/a (resolved 2026-06-12 from this
    # fixture to a 4e-14 relative deviation; see the module docstring).
    factor = k_p * k_n / radius

    significant = np.abs(t_ref) > 1e-5 * np.abs(t_ref).max()
    assert significant.sum() > n_pw // 2
    ratio = t_ref[significant] / t_lax[significant]
    np.testing.assert_allclose(
        ratio,
        factor,
        rtol=float(data["rtol"]),
        err_msg="cross-engine convention factor drifted from k_p*k_n/a — C7",
    )
    np.testing.assert_allclose(
        t_lax * factor,
        t_ref,
        rtol=float(data["rtol"]),
        atol=float(data["atol"]),
    )

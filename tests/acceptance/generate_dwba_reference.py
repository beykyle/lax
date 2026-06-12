"""Generate the T5 cross-engine DWBA reference fixture from jitr.

Runs jitr's ``xs.quasielastic_pn.Workspace`` on the ⁴⁸Ca(p,n)⁴⁸Sc IAS case
(the setup of jitr's ``examples/notebooks/chex_jitr_validation.ipynb``, whose
semi-relativistic variant is validated against CHEX and Jon et al. data) and
exports the per-(l, j) T-matrix elements, elastic S-matrices, sampled KDUQ
potentials, and kinematics to ``tests/acceptance/data/dwba_pn_reference.npz``
for ``test_dwba_reference.py``.

**Classical kinematics** are used (instead of the notebook's semi-relativistic
prescription) so that ``ħ²k²/2μ = Ecm`` and ``η = αZz·μ/(ħk)`` hold as
identities — the lax solver then reproduces each channel exactly from
``(Ecm, ħ²/2μ, z1z2)`` with no potential rescaling.  The reference values
therefore differ ~1-2% from the CHEX-validated notebook numbers, but the
engine and workspace are identical.

Run inside the jitr virtualenv::

    cd ~/umich/jitr && .venv/bin/python <lax>/tests/acceptance/generate_dwba_reference.py
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import jitr
import numpy as np
from jitr.utils.kinematics import classical_kinematics, classical_kinematics_cm

# --- the notebook case, classical kinematics --------------------------------
PROTON = (1, 1)
NEUTRON = (1, 0)
CA48 = (48, 20)
SC48 = (48, 21)
E_LAB = 35.0  # MeV
E_IAS = 6.67  # MeV
CHANNEL_RADIUS_FM = 16.0
LMAX = 20
NBASIS = 35

OUTPUT = Path(__file__).parent / "data" / "dwba_pn_reference.npz"


def main() -> None:
    reaction = jitr.reactions.Reaction(
        target=CA48, projectile=PROTON, product=NEUTRON, residual=SC48
    )
    reaction_exit_channel = jitr.reactions.Reaction(
        target=reaction.residual, projectile=reaction.product, process="El"
    )

    # Classical kinematics, mirroring Reaction.kinematics / kinematics_exit
    # but with the non-relativistic constructors.
    kin_p = classical_kinematics(
        reaction.target.m0,
        reaction.projectile.m0,
        E_LAB,
        Zz=reaction.target.Z * reaction.projectile.Z,
    )
    ecm_n = kin_p.Ecm + reaction.Q - E_IAS
    kin_n = classical_kinematics_cm(
        reaction.residual.m0 + E_IAS,
        reaction.product.m0,
        ecm_n,
        Zz=reaction.residual.Z * reaction.product.Z,  # neutron: 0
    )

    solver = jitr.rmatrix.Solver(nbasis=NBASIS)
    workspace = jitr.xs.quasielastic_pn.Workspace(
        reaction,
        kin_p,
        kin_n,
        solver,
        np.linspace(0.01, np.pi, 10),  # angles are unused by tmatrix()
        LMAX,
        CHANNEL_RADIUS_FM,
        tmatrix_abs_tol=1.0e-16,  # no early partial-wave exit
    )
    rgrid = workspace.radial_grid()

    # KDUQ default-parameter potentials sampled at the classical kinematics.
    kd_p_params = np.array(list(jitr.optical_potentials.kduq.Global(PROTON).params.values()))
    kd_n_params = np.array(list(jitr.optical_potentials.kduq.Global(NEUTRON).params.values()))
    omp_p = jitr.optical_potentials.kduq.KDUQ(PROTON)
    omp_n = jitr.optical_potentials.kduq.KDUQ(NEUTRON)
    u_p_central, u_p_spin_orbit, u_p_coulomb = omp_p(rgrid, reaction, kin_p, *kd_p_params)
    u_n_central, u_n_spin_orbit, _ = omp_n(rgrid, reaction_exit_channel, kin_n, *kd_n_params)

    tpn, s_n, s_p = workspace.tmatrix(
        u_p_coulomb,
        u_p_central,
        u_p_spin_orbit,
        u_n_central,
        u_n_spin_orbit,
    )

    # Flatten the (lmax+1, 2) layout to N_pw = 2*lmax + 1 rows: l = 0 has only
    # j = 1/2 (l.s = 0); l >= 1 carries (j = l+1/2, j = l-1/2) with the l.s
    # eigenvalues the workspace itself precomputed.
    rows: list[tuple[int, float, int]] = [(0, 0.0, 0)]  # (l, l.s, ji)
    for ell in range(1, LMAX + 1):
        ldots = workspace.l_dot_s[ell - 1]
        rows.append((ell, float(ldots[0]), 0))
        rows.append((ell, float(ldots[1]), 1))

    ls = np.array([row[0] for row in rows], dtype=np.int64)
    ldots = np.array([row[1] for row in rows], dtype=np.float64)
    t_ref = np.array([tpn[row[0], row[2]] for row in rows])
    s_p_ref = np.array([s_p[row[0], row[2]] for row in rows])
    s_n_ref = np.array([s_n[row[0], row[2]] for row in rows])

    # Per-row composed potentials (raw MeV node values, exactly as the
    # workspace composes them; Coulomb enters the proton solve, not U1).
    iso = workspace.isovector_factor
    u_p_rows = u_p_central + u_p_coulomb + ldots[:, None] * u_p_spin_orbit
    u_n_rows = u_n_central + ldots[:, None] * u_n_spin_orbit
    u1_rows = -iso * (
        (u_n_central - u_p_central) + ldots[:, None] * (u_n_spin_orbit - u_p_spin_orbit)
    )

    try:
        commit = subprocess.run(
            ["git", "-C", str(Path(jitr.__file__).resolve().parents[3]), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        commit = "unknown"

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUTPUT,
        rgrid=np.asarray(rgrid),
        ls=ls,
        ldots=ldots,
        channel_radius=CHANNEL_RADIUS_FM,
        nbasis=NBASIS,
        Ecm_p=kin_p.Ecm,
        Ecm_n=kin_n.Ecm,
        mu_p=kin_p.mu,
        mu_n=kin_n.mu,
        k_p=kin_p.k,
        k_n=kin_n.k,
        eta_p=kin_p.eta,
        z1z2_p=np.array([reaction.projectile.Z, reaction.target.Z], dtype=np.int64),
        U_p=np.broadcast_to(u_p_rows, (len(rows), NBASIS)).copy(),
        U_n=np.broadcast_to(u_n_rows, (len(rows), NBASIS)).copy(),
        U1=np.broadcast_to(u1_rows, (len(rows), NBASIS)).copy(),
        T_ref=t_ref,
        S_p=s_p_ref,
        S_n=s_n_ref,
        rtol=1.0e-8,
        atol=1.0e-12,
        provenance=(
            f"jitr {getattr(jitr, '__version__', 'unknown')} commit {commit}; "
            "48Ca(p,n)48Sc IAS, E_lab=35 MeV, E_IAS=6.67 MeV, KDUQ defaults, "
            "lmax=20, a=16 fm, nbasis=35, CLASSICAL kinematics "
            "(the semi-relativistic variant of this case is CHEX/data-validated; "
            "see jitr examples/notebooks/chex_jitr_validation.ipynb)"
        ),
    )
    print(f"wrote {OUTPUT}")
    print(f"  kin_p: Ecm={kin_p.Ecm:.6f} k={kin_p.k:.6f} mu={kin_p.mu:.4f} eta={kin_p.eta:.6f}")
    print(f"  kin_n: Ecm={kin_n.Ecm:.6f} k={kin_n.k:.6f} mu={kin_n.mu:.4f} eta={kin_n.eta:.6f}")
    print(f"  |T| range: {np.abs(t_ref).min():.3e} .. {np.abs(t_ref).max():.3e}")


if __name__ == "__main__":
    main()

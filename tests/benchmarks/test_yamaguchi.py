"""
Yamaguchi non-local potential benchmark.

Reference values from Descouvemont [2], Example 5 / Appendix E:
  N=10, a=8 fm:
    E=0.1  MeV  → δ = -15.0770°  (rounded; full value -1.5077e+01 deg)
    E=10.0 MeV  → δ =  85.6370°

Reference values from Baye [1] (via Hesse et al.), reproduced by the direct
prototype in `prototype/test_yamaguchi.py`:
  N=20, a=15 fm:
    E=0.1  → -15.078689°
    E=10.0 →  85.634560°

The `a=8` case checks the finite-radius Descouvemont setup directly. The
large-radius Baye/Hesse value is reached once the channel radius is increased;
the prototype and the library both recover it at `a=15, N=20`.

This test is the keystone benchmark: it exercises the full chain
  assemble_nonlocal → spectrum → rmatrix → smatrix → phases.
"""

import numpy as np
import pytest

from tests.benchmarks._reference_cases import YAMAGUCHI_REFERENCES, YamaguchiReference

pytest.importorskip("jax")  # skip gracefully if JAX not installed
pytest.importorskip("scipy")

HBAR2_2MU = 41.472  # MeV·fm²   Descouvemont eq. 46
ALPHA = 0.2316053  # fm⁻¹
BETA = 1.3918324  # fm⁻¹

YAMAGUCHI_ASYMPTOTIC_CASES = [
    (15.0, 20, 0.1, -15.078689, 1e-5),
    (15.0, 20, 10.0, 85.634560, 1e-5),
]
COMPLEX_LIMIT_CASES = [(8.0, 10, 0.1), (8.0, 10, 10.0)]


def _yamaguchi_kernel(r1, r2):
    """Non-local kernel in MeV. Descouvemont eq. 53 × HBAR2_2MU."""

    import jax.numpy as jnp

    return -2.0 * BETA * (ALPHA + BETA) ** 2 * jnp.exp(-BETA * (r1 + r2)) * HBAR2_2MU


def _complex_yamaguchi_kernel(r1, r2, imag_strength):
    """Artificial optical variant used to test the `eig` real-potential limit."""

    return _yamaguchi_kernel(r1, r2) * (1.0 + 1.0j * imag_strength)


def _phase_from_direct_rmatrix(solver, potential):
    """Evaluate phase shifts from `solver.rmatrix_direct` on the compile-time energy grid."""

    import jax.numpy as jnp

    import lax as lm
    from lax.boundary import BoundaryValues

    assert solver.rmatrix_direct is not None
    assert solver.boundary is not None

    r_values = solver.rmatrix_direct(potential)
    phases = []
    for energy_index in range(r_values.shape[0]):
        boundary = BoundaryValues(
            H_plus=solver.boundary.H_plus[energy_index],
            H_minus=solver.boundary.H_minus[energy_index],
            H_plus_p=solver.boundary.H_plus_p[energy_index],
            H_minus_p=solver.boundary.H_minus_p[energy_index],
            is_open=solver.boundary.is_open[energy_index],
            k=solver.boundary.k[energy_index],
        )
        smatrix = lm.spectral.smatrix_from_R(r_values[energy_index], boundary)
        phases.append(lm.spectral.phases_from_S(smatrix))
    return jnp.stack(phases)


@pytest.mark.benchmark
@pytest.mark.parametrize(
    "reference",
    YAMAGUCHI_REFERENCES,
    ids=["a8-n10", "a8-n15", "a12-n15"],
)
def test_yamaguchi_phase_shifts(reference: YamaguchiReference) -> None:
    """Phase shifts match the published Descouvemont Example 5 values."""
    import jax.numpy as jnp

    import lax as lm

    solver = lm.compile(
        mesh=lm.MeshSpec("legendre", "x", n=reference.n_basis, scale=reference.scale),
        channels=(lm.ChannelSpec(l=0, threshold=0.0, mass_factor=HBAR2_2MU),),
        operators=("T+L",),
        solvers=("spectrum", "phases"),
        energies=jnp.asarray(reference.energies),
    )
    potential = lm.assemble_nonlocal(solver.mesh, _yamaguchi_kernel)
    spectrum = solver.spectrum(potential)
    phases_deg = np.asarray(solver.phases(spectrum))[:, 0] * (180.0 / np.pi)

    assert np.allclose(phases_deg, reference.phases_deg, atol=1.0e-2, rtol=0.0)


@pytest.mark.benchmark
@pytest.mark.parametrize(
    "reference",
    YAMAGUCHI_REFERENCES,
    ids=["a8-n10", "a8-n15", "a12-n15"],
)
def test_yamaguchi_phase_shifts_direct_rmatrix(reference: YamaguchiReference) -> None:
    """Direct R-matrix phase shifts match the published Descouvemont Example 5 values."""

    import lax as lm

    solver = lm.compile(
        mesh=lm.MeshSpec("legendre", "x", n=reference.n_basis, scale=reference.scale),
        channels=(lm.ChannelSpec(l=0, threshold=0.0, mass_factor=HBAR2_2MU),),
        operators=("T+L",),
        solvers=("rmatrix_direct",),
        energies=reference.energies,
        method="linear_solve",
    )
    potential = lm.assemble_nonlocal(solver.mesh, _yamaguchi_kernel)
    phases_deg = np.asarray(_phase_from_direct_rmatrix(solver, potential))[:, 0] * (180.0 / np.pi)

    assert np.allclose(phases_deg, reference.phases_deg, atol=1.0e-2, rtol=0.0)


@pytest.mark.benchmark
@pytest.mark.parametrize(
    "reference",
    YAMAGUCHI_REFERENCES,
    ids=["a8-n10", "a8-n15", "a12-n15"],
)
def test_yamaguchi_direct_matches_spectral(reference: YamaguchiReference) -> None:
    """Direct and spectral Yamaguchi phase shifts agree on each published mesh."""

    import lax as lm

    solver = lm.compile(
        mesh=lm.MeshSpec("legendre", "x", n=reference.n_basis, scale=reference.scale),
        channels=(lm.ChannelSpec(l=0, threshold=0.0, mass_factor=HBAR2_2MU),),
        operators=("T+L",),
        solvers=("spectrum", "rmatrix", "phases", "rmatrix_direct"),
        energies=reference.energies,
    )
    potential = lm.assemble_nonlocal(solver.mesh, _yamaguchi_kernel)
    spectrum = solver.spectrum(potential)
    spectral_delta = np.asarray(solver.phases(spectrum))[:, 0]
    direct_delta = np.asarray(_phase_from_direct_rmatrix(solver, potential))[:, 0]

    assert np.allclose(direct_delta, spectral_delta, atol=1.0e-10, rtol=1.0e-10)


@pytest.mark.parametrize("a,n,E,ref_deg,tol", YAMAGUCHI_ASYMPTOTIC_CASES)
def test_yamaguchi_large_radius_matches_baye_reference(a, n, E, ref_deg, tol):
    """The large-radius single-channel limit matches the Baye/Hesse reference values."""

    import jax.numpy as jnp

    import lax as lm

    solver = lm.compile(
        mesh=lm.MeshSpec("legendre", "x", n=n, scale=a),
        channels=(lm.ChannelSpec(l=0, threshold=0.0, mass_factor=HBAR2_2MU),),
        operators=("T+L",),
        solvers=("spectrum", "phases"),
        energies=jnp.array([E]),
    )
    potential = lm.assemble_nonlocal(solver.mesh, _yamaguchi_kernel)
    delta = float(solver.phases(solver.spectrum(potential))[0, 0]) * (180.0 / np.pi)

    assert abs(delta - ref_deg) < tol, (
        f"N={n}, a={a}, E={E}: δ={delta:.6f}° vs ref={ref_deg:.6f}° (tol={tol})"
    )


@pytest.mark.benchmark
@pytest.mark.parametrize("a,n,E", COMPLEX_LIMIT_CASES)
def test_complex_yamaguchi_eig_matches_real_limit(a, n, E):
    """The complex-spectrum path approaches the real Yamaguchi result as Im(V) -> 0."""

    import lax as lm

    imag_strength = 1.0e-8
    real_solver = lm.compile(
        mesh=lm.MeshSpec("legendre", "x", n=n, scale=a),
        channels=(lm.ChannelSpec(l=0, threshold=0.0, mass_factor=HBAR2_2MU),),
        operators=("T+L",),
        solvers=("spectrum", "phases"),
        energies=np.array([E]),
        method="eigh",
    )
    complex_solver = lm.compile(
        mesh=lm.MeshSpec("legendre", "x", n=n, scale=a),
        channels=(lm.ChannelSpec(l=0, threshold=0.0, mass_factor=HBAR2_2MU),),
        operators=("T+L",),
        solvers=("spectrum", "phases"),
        energies=np.array([E]),
        V_is_complex=True,
        method="eig",
    )
    real_potential = lm.assemble_nonlocal(real_solver.mesh, _yamaguchi_kernel)
    complex_potential = lm.assemble_nonlocal(
        complex_solver.mesh,
        lambda r1, r2: _complex_yamaguchi_kernel(r1, r2, imag_strength),
    )
    real_phase = float(real_solver.phases(real_solver.spectrum(real_potential))[0, 0])
    complex_phase = float(complex_solver.phases(complex_solver.spectrum(complex_potential))[0, 0])

    assert abs(complex_phase - real_phase) < 1.0e-7

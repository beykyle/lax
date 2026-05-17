from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("scipy")

HBAR2_2MU = 0.5


@pytest.mark.benchmark
def test_confined_hydrogen_ground_state_legendre_x_one_minus_x() -> None:
    """Legendre-x(1-x) reproduces the confined-hydrogen ground state. [DESIGN.md §19]"""

    import lax as lm

    solver = lm.compile(
        mesh=lm.MeshSpec("legendre", "x(1-x)", n=8, scale=2.0),
        channels=(lm.ChannelSpec(l=0, threshold=0.0, mass_factor=HBAR2_2MU),),
        operators=("T", "1/r"),
        solvers=("spectrum",),
    )

    assert solver.spectrum is not None
    potential = lm.assemble_local(solver.mesh, lambda r: -1.0 / r)
    spectrum = solver.spectrum(potential)
    ground_state = float(np.asarray(spectrum.eigenvalues)[0]) * HBAR2_2MU

    assert abs(ground_state + 0.125) < 1.0e-13


@pytest.mark.benchmark
def test_harmonic_oscillator_ground_state_modified_laguerre_x2() -> None:
    """Modified-Laguerre-x^2 reproduces the 3D harmonic-oscillator ground state."""

    import lax as lm

    solver = lm.compile(
        mesh=lm.MeshSpec("laguerre", "modified_x^2", n=20, scale=1.0),
        channels=(lm.ChannelSpec(l=0, threshold=0.0, mass_factor=1.0),),
        operators=("T",),
        solvers=("spectrum",),
    )

    assert solver.spectrum is not None
    potential = lm.assemble_local(solver.mesh, lambda r: 0.25 * r**2)
    spectrum = solver.spectrum(potential)
    ground_state = float(np.asarray(spectrum.eigenvalues)[0])

    assert abs(ground_state - 1.5) < 1.0e-12

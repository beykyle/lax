from __future__ import annotations

import jax
import numpy as np
import pytest
import scipy.special as sc

import lax as lm
from lax.meshes._basis_eval import basis_at

pytest.importorskip("jax")
pytest.importorskip("scipy")

HBAR2_2MU = 41.472
ALPHA = 0.2316053
BETA = 1.3918324


@pytest.mark.benchmark
def test_yamaguchi_fourier_matches_direct_integral() -> None:
    """Momentum-space transform of a Yamaguchi eigenvector matches direct quadrature."""

    import jax.numpy as jnp

    def yamaguchi_kernel(r1: jax.Array, r2: jax.Array) -> jax.Array:
        return -2.0 * BETA * (ALPHA + BETA) ** 2 * jnp.exp(-BETA * (r1 + r2)) * HBAR2_2MU

    solver = lm.compile(
        mesh=lm.MeshSpec("legendre", "x", n=20, scale=15.0),
        channels=(lm.ChannelSpec(l=0, threshold=0.0, mass_factor=HBAR2_2MU),),
        operators=("T+L",),
        solvers=("spectrum", "wavefunction"),
        momenta=jnp.linspace(0.1, 2.0, 20),
    )
    potential = lm.assemble_nonlocal(solver.mesh, yamaguchi_kernel)

    assert solver.spectrum is not None
    assert solver.fourier is not None
    assert solver.transforms.momenta is not None

    spectrum = solver.spectrum(potential)
    assert spectrum.eigenvectors is not None
    coefficients = np.asarray(spectrum.eigenvectors)[:, 0]
    transformed = np.asarray(solver.fourier(jnp.asarray(coefficients)))

    x_q, w_q = np.polynomial.legendre.leggauss(400)
    r_q = 0.5 * solver.mesh.scale * (x_q + 1.0)
    w_q = 0.5 * solver.mesh.scale * w_q
    basis_values = np.asarray(basis_at(solver.mesh, jnp.asarray(r_q)))
    position_values = basis_values @ coefficients
    expected = []
    for momentum in np.asarray(solver.transforms.momenta):
        bessel = sc.spherical_jn(0, momentum * r_q)
        expected.append(np.sqrt(2.0 / np.pi) * np.dot(w_q, bessel * position_values))

    assert np.allclose(transformed, np.asarray(expected), atol=1.0e-6, rtol=1.0e-6)

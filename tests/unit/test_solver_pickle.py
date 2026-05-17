from __future__ import annotations

import pickle

import jax.numpy as jnp
import numpy as np
import pytest

import lax as lm

pytest.importorskip("jax")


def test_compiled_solver_round_trips_through_pickle() -> None:
    """A compiled solver can be pickled and restored without changing results."""

    solver = lm.compile(
        mesh=lm.MeshSpec("legendre", "x", n=4, scale=7.0),
        channels=(lm.ChannelSpec(l=0, threshold=0.0, mass_factor=2.0),),
        solvers=("spectrum", "rmatrix", "smatrix", "phases", "greens", "wavefunction", "rmatrix_direct"),
        energies=jnp.asarray([0.5]),
        grid=jnp.linspace(0.5, 6.5, 7),
        momenta=jnp.linspace(0.1, 1.0, 6),
    )

    restored = pickle.loads(pickle.dumps(solver))
    potential = jnp.asarray([[[0.2, 0.1, 0.0, -0.1]]])
    source = jnp.asarray([1.0, 0.0, 0.0, 0.0])
    vector = jnp.asarray([0.3, -0.2, 0.1, 0.4])
    matrix = jnp.asarray(
        [
            [1.0, 0.2, 0.1, 0.0],
            [0.2, 0.8, 0.3, 0.1],
            [0.1, 0.3, 0.7, 0.2],
            [0.0, 0.1, 0.2, 0.6],
        ]
    )
    diagonal_operator = jnp.asarray([1.0, 1.5, 2.0, 2.5])

    for name in (
        "spectrum",
        "rmatrix",
        "smatrix",
        "phases",
        "greens",
        "wavefunction",
        "rmatrix_direct",
        "to_grid_vector",
        "from_grid_vector",
        "to_grid_matrix",
        "fourier",
        "double_fourier_transform",
        "integrate",
    ):
        assert getattr(restored, name) is not None

    assert solver.spectrum is not None
    assert solver.rmatrix is not None
    assert solver.smatrix is not None
    assert solver.phases is not None
    assert solver.greens is not None
    assert solver.wavefunction is not None
    assert solver.rmatrix_direct is not None
    assert solver.to_grid_vector is not None
    assert solver.from_grid_vector is not None
    assert solver.to_grid_matrix is not None
    assert solver.fourier is not None
    assert solver.double_fourier_transform is not None
    assert solver.integrate is not None

    fresh_spectrum = solver.spectrum(potential)
    restored_spectrum = restored.spectrum(potential)

    assert np.allclose(np.asarray(restored.mesh.nodes), np.asarray(solver.mesh.nodes))
    assert np.allclose(np.asarray(restored.mesh.weights), np.asarray(solver.mesh.weights))
    assert np.allclose(
        np.asarray(restored_spectrum.eigenvalues),
        np.asarray(fresh_spectrum.eigenvalues),
    )
    assert np.allclose(
        np.asarray(restored_spectrum.surface_amplitudes),
        np.asarray(fresh_spectrum.surface_amplitudes),
    )
    assert restored_spectrum.eigenvectors is not None
    assert fresh_spectrum.eigenvectors is not None
    assert np.allclose(
        np.abs(np.asarray(restored_spectrum.eigenvectors)),
        np.abs(np.asarray(fresh_spectrum.eigenvectors)),
    )
    assert np.allclose(
        np.asarray(restored.rmatrix(restored_spectrum, 0.5)),
        np.asarray(solver.rmatrix(fresh_spectrum, 0.5)),
    )
    assert np.allclose(
        np.asarray(restored.smatrix(restored_spectrum)),
        np.asarray(solver.smatrix(fresh_spectrum)),
    )
    assert np.allclose(
        np.asarray(restored.phases(restored_spectrum)),
        np.asarray(solver.phases(fresh_spectrum)),
    )
    assert np.allclose(
        np.asarray(restored.greens(restored_spectrum, 0.5)),
        np.asarray(solver.greens(fresh_spectrum, 0.5)),
    )
    assert np.allclose(
        np.asarray(restored.wavefunction(restored_spectrum, 0.5, source)),
        np.asarray(solver.wavefunction(fresh_spectrum, 0.5, source)),
    )
    assert np.allclose(
        np.asarray(restored.rmatrix_direct(potential)),
        np.asarray(solver.rmatrix_direct(potential)),
    )
    assert np.allclose(
        np.asarray(restored.to_grid_vector(vector)),
        np.asarray(solver.to_grid_vector(vector)),
    )
    assert np.allclose(
        np.asarray(restored.from_grid_vector(restored.to_grid_vector(vector))),
        np.asarray(solver.from_grid_vector(solver.to_grid_vector(vector))),
    )
    assert np.allclose(
        np.asarray(restored.to_grid_matrix(matrix)),
        np.asarray(solver.to_grid_matrix(matrix)),
    )
    assert np.allclose(
        np.asarray(restored.fourier(vector)),
        np.asarray(solver.fourier(vector)),
    )
    assert np.allclose(
        np.asarray(restored.double_fourier_transform(matrix)),
        np.asarray(solver.double_fourier_transform(matrix)),
    )
    assert np.allclose(
        np.asarray(restored.integrate(vector)),
        np.asarray(solver.integrate(vector)),
    )
    assert np.allclose(
        np.asarray(restored.integrate(vector, diagonal_operator)),
        np.asarray(solver.integrate(vector, diagonal_operator)),
    )

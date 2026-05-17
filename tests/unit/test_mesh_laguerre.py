from __future__ import annotations

import numpy as np
import pytest
import scipy.special as sc

from lax.meshes import build_mesh
from lax.meshes.laguerre import build_laguerre_modified_x2, build_laguerre_x

pytest.importorskip("jax")
pytest.importorskip("scipy")


def test_build_laguerre_x_nodes_weights_and_boundary() -> None:
    """Laguerre-x builder returns valid nodes, weights, and open-boundary values."""

    mesh, operators = build_laguerre_x(n=6, scale=2.0, operators={"T", "1/r", "1/r^2"})

    nodes = np.asarray(mesh.nodes)
    weights = np.asarray(mesh.weights)
    expected_nodes, quadrature_weights = sc.roots_laguerre(6)
    expected_weights = quadrature_weights * np.exp(expected_nodes)

    assert np.all(nodes > 0.0)
    assert np.all(weights > 0.0)
    assert np.allclose(nodes, expected_nodes)
    assert np.allclose(weights, expected_weights)
    assert np.allclose(np.asarray(mesh.radii), mesh.scale * nodes)
    assert np.allclose(np.asarray(mesh.basis_at_boundary), np.zeros(mesh.n))
    assert operators.T is not None
    assert operators.TpL is not None
    assert operators.inv_r is not None
    assert operators.inv_r2 is not None


def test_build_laguerre_x_kinetic_is_symmetric() -> None:
    """Laguerre-x kinetic matrix is symmetric."""

    _, operators = build_laguerre_x(n=10, scale=1.5, operators={"T"})

    assert operators.T is not None
    matrix = np.asarray(operators.T)
    assert np.allclose(matrix, matrix.T, atol=1.0e-12)


def test_build_laguerre_modified_x2_nodes_weights_and_boundary() -> None:
    """Modified-Laguerre-x^2 builder returns valid nodes, weights, and open-boundary values."""

    mesh, operators = build_laguerre_modified_x2(n=6, scale=1.0, operators={"T", "1/r^2"})

    squared_nodes, _ = sc.roots_genlaguerre(6, 0.5)
    expected_nodes = np.sqrt(squared_nodes)

    assert np.all(np.asarray(mesh.nodes) > 0.0)
    assert np.all(np.asarray(mesh.weights) > 0.0)
    assert np.allclose(np.asarray(mesh.nodes), expected_nodes)
    assert np.allclose(np.asarray(mesh.radii), expected_nodes)
    assert np.allclose(np.asarray(mesh.basis_at_boundary), np.zeros(mesh.n))
    assert operators.T is not None
    assert operators.TpL is not None
    assert operators.inv_r2 is not None


def test_build_laguerre_modified_x2_kinetic_is_symmetric() -> None:
    """Modified-Laguerre-x^2 kinetic matrix is symmetric."""

    _, operators = build_laguerre_modified_x2(n=10, scale=1.5, operators={"T"})

    assert operators.T is not None
    matrix = np.asarray(operators.T)
    assert np.allclose(matrix, matrix.T, atol=1.0e-12)


def test_build_mesh_dispatches_laguerre_x() -> None:
    """Registry dispatch returns the registered Laguerre-x builder."""

    mesh, operators = build_mesh("laguerre", "x", n=5, scale=2.0, operators={"T"})

    assert mesh.family == "laguerre"
    assert mesh.regularization == "x"
    assert operators.T is not None


def test_build_mesh_dispatches_laguerre_modified_x2() -> None:
    """Registry dispatch returns the registered modified-Laguerre-x^2 builder."""

    mesh, operators = build_mesh("laguerre", "modified_x^2", n=5, scale=1.0, operators={"T"})

    assert mesh.family == "laguerre"
    assert mesh.regularization == "modified_x^2"
    assert operators.T is not None


def test_build_laguerre_x_rejects_nonzero_alpha() -> None:
    """Laguerre-x builder rejects unsupported alpha values."""

    with pytest.raises(NotImplementedError, match="alpha=0.0"):
        build_laguerre_x(n=4, scale=1.0, operators={"T"}, alpha=1.0)


def test_build_laguerre_modified_x2_rejects_unsupported_alpha() -> None:
    """Modified-Laguerre-x^2 builder rejects unsupported alpha values."""

    with pytest.raises(NotImplementedError, match="alpha=0.5"):
        build_laguerre_modified_x2(n=4, scale=1.0, operators={"T"}, alpha=-0.5)

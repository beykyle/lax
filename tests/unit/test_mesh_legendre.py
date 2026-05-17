from __future__ import annotations

import numpy as np
import pytest

from lax.meshes import build_mesh
from lax.meshes.legendre import (
    build_legendre_x,
    build_legendre_x_one_minus_x,
    build_legendre_x_three_halves,
)

pytest.importorskip("jax")
pytest.importorskip("scipy")


def test_build_legendre_x_nodes_boundary_and_weights() -> None:
    """Legendre-x builder returns valid nodes, weights, and boundary values."""

    mesh, operators = build_legendre_x(n=6, scale=8.0, operators={"T+L", "1/r", "D"})

    nodes = np.asarray(mesh.nodes)
    weights = np.asarray(mesh.weights)
    boundary_sign = np.where(np.arange(mesh.n) % 2 == 0, 1.0, -1.0)
    expected_boundary = np.array(
        [
            sign / np.sqrt(mesh.scale * node * (1.0 - node))
            for sign, node in zip(boundary_sign, nodes, strict=True)
        ]
    )

    assert np.all((0.0 < nodes) & (nodes < 1.0))
    assert np.all(weights > 0.0)
    assert np.allclose(np.asarray(mesh.radii), mesh.scale * nodes)
    assert np.allclose(np.asarray(mesh.basis_at_boundary), expected_boundary)
    assert operators.TpL is not None
    assert operators.inv_r is not None
    assert operators.D is not None


def test_build_legendre_x_t_plus_l_is_symmetric() -> None:
    """Legendre-x T+L matrix is symmetric."""

    _, operators = build_legendre_x(n=12, scale=10.0, operators={"T+L"})

    assert operators.TpL is not None
    matrix = np.asarray(operators.TpL)
    assert np.allclose(matrix, matrix.T, atol=1.0e-12)


def test_build_legendre_x_one_minus_x_nodes_weights_and_boundary() -> None:
    """Legendre-x(1-x) builder returns confined-boundary mesh data."""

    mesh, operators = build_legendre_x_one_minus_x(n=6, scale=2.0, operators={"T", "1/r"})

    nodes = np.asarray(mesh.nodes)
    weights = np.asarray(mesh.weights)
    expected_x_raw, expected_w_raw = np.polynomial.legendre.leggauss(6)
    expected_nodes = 0.5 * (expected_x_raw + 1.0)
    expected_weights = 0.5 * expected_w_raw

    assert np.all((0.0 < nodes) & (nodes < 1.0))
    assert np.allclose(nodes, expected_nodes)
    assert np.allclose(weights, expected_weights)
    assert np.allclose(np.asarray(mesh.basis_at_boundary), np.zeros(mesh.n))
    assert operators.T is not None
    assert operators.TpL is not None
    assert operators.inv_r is not None


def test_build_legendre_x_one_minus_x_t_is_symmetric() -> None:
    """Legendre-x(1-x) kinetic matrix is symmetric."""

    _, operators = build_legendre_x_one_minus_x(n=10, scale=3.0, operators={"T"})

    assert operators.T is not None
    matrix = np.asarray(operators.T)
    assert np.allclose(matrix, matrix.T, atol=1.0e-12)


def test_build_legendre_x_three_halves_nodes_boundary_and_weights() -> None:
    """Legendre-x^3/2 builder returns valid boundary values and weights."""

    mesh, operators = build_legendre_x_three_halves(n=6, scale=8.0, operators={"T+L", "1/r^2"})

    nodes = np.asarray(mesh.nodes)
    weights = np.asarray(mesh.weights)
    boundary_sign = np.where(np.arange(mesh.n) % 2 == 0, 1.0, -1.0)
    expected_boundary = np.array(
        [
            sign / np.sqrt(mesh.scale * node * (1.0 - node))
            for sign, node in zip(boundary_sign, nodes, strict=True)
        ]
    )
    if mesh.n % 2 == 1:
        expected_boundary *= -1.0

    assert np.all((0.0 < nodes) & (nodes < 1.0))
    assert np.all(weights > 0.0)
    assert np.allclose(np.asarray(mesh.radii), mesh.scale * nodes)
    assert np.allclose(np.asarray(mesh.basis_at_boundary), expected_boundary)
    assert operators.TpL is not None
    assert operators.inv_r2 is not None


def test_build_legendre_x_three_halves_t_plus_l_is_symmetric() -> None:
    """Legendre-x^3/2 T+L matrix is symmetric."""

    _, operators = build_legendre_x_three_halves(n=12, scale=7.0, operators={"T+L"})

    assert operators.TpL is not None
    matrix = np.asarray(operators.TpL)
    assert np.allclose(matrix, matrix.T, atol=1.0e-12)


def test_build_mesh_dispatches_legendre_x() -> None:
    """Registry dispatch returns the registered Legendre-x builder."""

    mesh, operators = build_mesh("legendre", "x", n=5, scale=6.0, operators={"T+L"})

    assert mesh.family == "legendre"
    assert mesh.regularization == "x"
    assert operators.TpL is not None


def test_build_mesh_dispatches_legendre_x_one_minus_x() -> None:
    """Registry dispatch returns the registered Legendre-x(1-x) builder."""

    mesh, operators = build_mesh("legendre", "x(1-x)", n=5, scale=2.0, operators={"T"})

    assert mesh.family == "legendre"
    assert mesh.regularization == "x(1-x)"
    assert operators.T is not None


def test_build_mesh_dispatches_legendre_x_three_halves() -> None:
    """Registry dispatch returns the registered Legendre-x^3/2 builder."""

    mesh, operators = build_mesh("legendre", "x^3/2", n=5, scale=6.0, operators={"T+L"})

    assert mesh.family == "legendre"
    assert mesh.regularization == "x^3/2"
    assert operators.TpL is not None


def test_build_mesh_rejects_unknown_builder() -> None:
    """Registry rejects unsupported mesh families."""

    with pytest.raises(ValueError, match="No builder"):
        build_mesh("hermite", "x", n=4, scale=1.0, operators=set())

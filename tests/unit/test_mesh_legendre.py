from __future__ import annotations

import numpy as np
import pytest

from lax.meshes import build_mesh
from lax.meshes.legendre import build_legendre_x

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


def test_build_mesh_dispatches_legendre_x() -> None:
    """Registry dispatch returns the registered Legendre-x builder."""

    mesh, operators = build_mesh("legendre", "x", n=5, scale=6.0, operators={"T+L"})

    assert mesh.family == "legendre"
    assert mesh.regularization == "x"
    assert operators.TpL is not None


def test_build_mesh_rejects_unknown_builder() -> None:
    """Registry rejects unsupported mesh families."""

    with pytest.raises(ValueError, match="No builder"):
        build_mesh("laguerre", "x^3/2", n=4, scale=1.0, operators=set())

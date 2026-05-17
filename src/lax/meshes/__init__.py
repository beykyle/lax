"""Mesh builders and registry dispatch."""

from lax.meshes._registry import build_mesh, register
from lax.meshes.laguerre import build_laguerre_modified_x2, build_laguerre_x
from lax.meshes.legendre import (
    build_legendre_x,
    build_legendre_x_one_minus_x,
    build_legendre_x_three_halves,
)

__all__ = [
    "build_laguerre_modified_x2",
    "build_laguerre_x",
    "build_legendre_x",
    "build_legendre_x_one_minus_x",
    "build_legendre_x_three_halves",
    "build_mesh",
    "register",
]

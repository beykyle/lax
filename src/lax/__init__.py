"""
lax — JAX-compiled Lagrange-mesh solvers.

Import this package before importing jax.numpy directly so the x64 config
is applied before JAX's first array creation (see DESIGN.md §C.10).
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]

import lax.spectral as spectral  # noqa: E402
from lax.boundary._types import Solver  # noqa: E402
from lax.compile import compile  # noqa: E402
from lax.operators.potential import assemble_local, assemble_nonlocal  # noqa: E402
from lax.types import ChannelSpec, MeshSpec  # noqa: E402

__all__ = [
    "ChannelSpec",
    "MeshSpec",
    "Solver",
    "assemble_local",
    "assemble_nonlocal",
    "compile",
    "spectral",
]

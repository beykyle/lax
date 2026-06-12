"""
lax — JAX-compiled Lagrange-mesh solvers.

Import this package before importing jax.numpy directly so the x64 config
is applied before JAX's first array creation (see DESIGN.md §C.9).

Note: ``lax.compile`` shadows Python's built-in ``compile``.  Avoid
``from lax import compile`` in modules that also need the built-in.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]

import lax.constants as constants
import lax.models as models
import lax.spectral as spectral
from lax.compile import compile
from lax.types import ChannelSpec, Interaction, MeshSpec, Solver
from lax.wavefunction import make_wavefunction_source

__all__ = [
    "ChannelSpec",
    "Interaction",
    "MeshSpec",
    "Solver",
    "compile",
    "constants",
    "make_wavefunction_source",
    "models",
    "spectral",
]

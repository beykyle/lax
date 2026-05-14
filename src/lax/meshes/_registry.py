"""Mesh-builder registry and dispatch."""

from __future__ import annotations

from collections.abc import Callable

from lax.boundary._types import Mesh, OperatorMatrices
from lax.types import MeshFamily, Regularization

type Builder = Callable[..., tuple[Mesh, OperatorMatrices]]

_BUILDERS: dict[tuple[MeshFamily, Regularization], Builder] = {}


def register(family: MeshFamily, regularization: Regularization) -> Callable[[Builder], Builder]:
    """Register a mesh builder for a `(family, regularization)` pair."""

    def decorator(builder: Builder) -> Builder:
        key = (family, regularization)
        if key in _BUILDERS:
            msg = f"Builder already registered for {key}"
            raise ValueError(msg)
        _BUILDERS[key] = builder
        return builder

    return decorator


def build_mesh(
    family: MeshFamily,
    regularization: Regularization,
    n: int,
    scale: float,
    operators: set[str],
    **extras: object,
) -> tuple[Mesh, OperatorMatrices]:
    """Dispatch to the concrete mesh builder for the requested mesh kind."""

    key = (family, regularization)
    builder = _BUILDERS.get(key)
    if builder is None:
        available = sorted(_BUILDERS)
        msg = f"No builder for {key}. Available: {available}"
        raise ValueError(msg)
    return builder(n=n, scale=scale, operators=operators, **extras)


__all__ = ["Builder", "build_mesh", "register"]

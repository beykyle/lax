"""Mesh-builder registry and dispatch."""

from __future__ import annotations

from collections.abc import Callable

from lax.types import Mesh, MeshFamily, OperatorMatrices, Regularization

type Builder = Callable[..., tuple[Mesh, OperatorMatrices]]

_BUILDERS: dict[tuple[MeshFamily, Regularization], Builder] = {}


def register(family: MeshFamily, regularization: Regularization) -> Callable[[Builder], Builder]:
    """Register a mesh builder for a ``(family, regularization)`` pair.

    Use as a decorator on the builder function.  Raises ``ValueError`` if the
    key is already registered.

    Parameters
    ----------
    family
        Mesh family, e.g. ``"legendre"`` or ``"laguerre"``.
    regularization
        Endpoint regularization, e.g. ``"x"`` or ``"x(1-x)"``.

    Returns
    -------
    Callable[[Builder], Builder]
        Decorator that registers the builder and returns it unchanged.
    """

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
    """Dispatch to the concrete mesh builder for the requested mesh kind.

    Parameters
    ----------
    family
        Mesh family identifier.
    regularization
        Regularization identifier.
    n
        Number of mesh basis functions.
    scale
        Physical scale (channel radius or Laguerre scale) in fm.
    operators
        Set of operator names to precompute, e.g. ``{"T+L", "1/r"}``.
    **extras
        Additional keyword arguments forwarded to the concrete builder
        (e.g. ``n_intervals`` for propagated meshes).

    Returns
    -------
    tuple[Mesh, OperatorMatrices]
        Compiled mesh data and precomputed operator matrices.

    Raises
    ------
    ValueError
        If no builder is registered for ``(family, regularization)``.
    """

    key = (family, regularization)
    builder = _BUILDERS.get(key)
    if builder is None:
        available = sorted(_BUILDERS)
        msg = f"No builder for {key}. Available: {available}"
        raise ValueError(msg)
    return builder(n=n, scale=scale, operators=operators, **extras)


__all__ = ["Builder", "build_mesh", "register"]

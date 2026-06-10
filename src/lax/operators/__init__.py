"""Operator assembly helpers."""

from lax.operators.interaction import (
    make_interaction_from_array,
    make_interaction_from_block,
    make_interaction_from_funcs,
)

__all__ = [
    "make_interaction_from_array",
    "make_interaction_from_block",
    "make_interaction_from_funcs",
]

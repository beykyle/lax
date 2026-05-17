from __future__ import annotations

import math
from typing import Final


def wigner_3j(j1: int, j2: int, j3: int, m1: int, m2: int, m3: int) -> float:
    """Return the Wigner 3j symbol for small integer angular momenta."""

    if m1 + m2 + m3 != 0:
        return 0.0
    if not _triangle_allowed(j1, j2, j3):
        return 0.0
    if abs(m1) > j1 or abs(m2) > j2 or abs(m3) > j3:
        return 0.0

    prefactor = (-1.0) ** (j1 - j2 - m3)
    normalization = math.sqrt(
        _triangle_delta(j1, j2, j3)
        * math.factorial(j1 + m1)
        * math.factorial(j1 - m1)
        * math.factorial(j2 + m2)
        * math.factorial(j2 - m2)
        * math.factorial(j3 + m3)
        * math.factorial(j3 - m3)
    )

    lower = max(0, j2 - j3 - m1, j1 - j3 + m2)
    upper = min(j1 + j2 - j3, j1 - m1, j2 + m2)
    total = 0.0
    for index in range(lower, upper + 1):
        denominator = (
            math.factorial(index)
            * math.factorial(j1 + j2 - j3 - index)
            * math.factorial(j1 - m1 - index)
            * math.factorial(j2 + m2 - index)
            * math.factorial(j3 - j2 + m1 + index)
            * math.factorial(j3 - j1 - m2 + index)
        )
        total += (-1.0) ** index / denominator
    return prefactor * normalization * total


def wigner_6j(j1: int, j2: int, j3: int, l1: int, l2: int, l3: int) -> float:
    """Return the Wigner 6j symbol for small integer angular momenta."""

    if not (
        _triangle_allowed(j1, j2, j3)
        and _triangle_allowed(j1, l2, l3)
        and _triangle_allowed(l1, j2, l3)
        and _triangle_allowed(l1, l2, j3)
    ):
        return 0.0

    delta = math.sqrt(
        _triangle_delta(j1, j2, j3)
        * _triangle_delta(j1, l2, l3)
        * _triangle_delta(l1, j2, l3)
        * _triangle_delta(l1, l2, j3)
    )

    lower = max(j1 + j2 + j3, j1 + l2 + l3, l1 + j2 + l3, l1 + l2 + j3)
    upper = min(j1 + j2 + l1 + l2, j2 + j3 + l2 + l3, j3 + j1 + l3 + l1)
    total = 0.0
    for index in range(lower, upper + 1):
        denominator = (
            math.factorial(index - j1 - j2 - j3)
            * math.factorial(index - j1 - l2 - l3)
            * math.factorial(index - l1 - j2 - l3)
            * math.factorial(index - l1 - l2 - j3)
            * math.factorial(j1 + j2 + l1 + l2 - index)
            * math.factorial(j2 + j3 + l2 + l3 - index)
            * math.factorial(j3 + j1 + l3 + l1 - index)
        )
        total += (-1.0) ** index * math.factorial(index + 1) / denominator
    return delta * total


def _triangle_allowed(a: int, b: int, c: int) -> bool:
    """Return whether integer angular momenta satisfy the triangle rule."""

    return abs(a - b) <= c <= a + b


def _triangle_delta(a: int, b: int, c: int) -> float:
    """Return the triangle delta factor used in Wigner symbols."""

    return (
        math.factorial(a + b - c)
        * math.factorial(a - b + c)
        * math.factorial(-a + b + c)
        / math.factorial(a + b + c + 1)
    )


__all__: Final[list[str]] = ["wigner_3j", "wigner_6j"]

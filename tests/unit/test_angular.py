from __future__ import annotations

import math

from lax._angular import wigner_3j, wigner_6j


def test_wigner_3j_matches_known_closed_form_values() -> None:
    """Small-j Wigner 3j values match closed-form results."""

    assert math.isclose(wigner_3j(0, 0, 0, 0, 0, 0), 1.0, rel_tol=0.0, abs_tol=1.0e-15)
    assert math.isclose(
        wigner_3j(1, 1, 0, 0, 0, 0),
        -1.0 / math.sqrt(3.0),
        rel_tol=0.0,
        abs_tol=1.0e-15,
    )
    assert math.isclose(
        wigner_3j(2, 2, 0, 0, 0, 0),
        1.0 / math.sqrt(5.0),
        rel_tol=0.0,
        abs_tol=1.0e-15,
    )


def test_wigner_3j_returns_zero_for_selection_rule_violations() -> None:
    """Impossible Wigner 3j combinations vanish."""

    assert wigner_3j(1, 1, 1, 0, 0, 0) == 0.0
    assert wigner_3j(1, 1, 1, 1, 1, 1) == 0.0
    assert wigner_3j(1, 1, 3, 0, 0, 0) == 0.0


def test_wigner_3j_obeys_column_swap_symmetry() -> None:
    """Swapping the first two columns gives the standard phase factor."""

    direct = wigner_3j(1, 1, 1, 1, -1, 0)
    swapped = wigner_3j(1, 1, 1, -1, 1, 0)

    assert math.isclose(swapped, (-1.0) ** (1 + 1 + 1) * direct, rel_tol=0.0, abs_tol=1.0e-15)


def test_wigner_6j_matches_known_closed_form_values() -> None:
    """Small-j Wigner 6j values match closed-form results."""

    assert math.isclose(wigner_6j(0, 0, 0, 0, 0, 0), 1.0, rel_tol=0.0, abs_tol=1.0e-15)
    assert math.isclose(wigner_6j(1, 1, 0, 1, 1, 0), 1.0 / 3.0, rel_tol=0.0, abs_tol=1.0e-15)
    assert math.isclose(wigner_6j(2, 2, 0, 2, 2, 0), 1.0 / 5.0, rel_tol=0.0, abs_tol=1.0e-15)


def test_wigner_6j_returns_zero_for_selection_rule_violations() -> None:
    """Impossible Wigner 6j combinations vanish."""

    assert wigner_6j(1, 1, 3, 1, 1, 1) == 0.0
    assert wigner_6j(1, 2, 1, 1, 1, 4) == 0.0


def test_wigner_6j_is_invariant_under_column_permutations() -> None:
    """Column permutations leave the Wigner 6j value unchanged."""

    direct = wigner_6j(2, 1, 1, 1, 2, 2)
    swapped_columns = wigner_6j(1, 2, 1, 2, 1, 2)

    assert math.isclose(swapped_columns, direct, rel_tol=0.0, abs_tol=1.0e-15)

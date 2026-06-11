"""Tests for the Interaction symmetry-block axis (`block_dependent`, DESIGN.md §15.5)."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from lax.types import Interaction

M = 3
N_E = 4
N_B = 2


def _plain() -> Interaction:
    return Interaction(block=jnp.full((M, M), 1.0), energy_dependent=False)


def _energy() -> Interaction:
    block = jnp.arange(N_E, dtype=jnp.float64)[:, None, None] * jnp.ones((M, M))
    return Interaction(block=block, energy_dependent=True)


def _blocks() -> Interaction:
    block = jnp.arange(N_B, dtype=jnp.float64)[:, None, None] * jnp.ones((M, M)) + 10.0
    return Interaction(block=block, energy_dependent=False, block_dependent=True)


def _blocks_energy() -> Interaction:
    block = (
        jnp.arange(N_B * N_E, dtype=jnp.float64).reshape(N_B, N_E)[:, :, None, None]
        * jnp.ones((M, M))
        + 100.0
    )
    return Interaction(block=block, energy_dependent=True, block_dependent=True)


def test_default_flag_is_block_independent() -> None:
    interaction = _plain()
    assert interaction.block_dependent is False


def test_add_plain_plain_stays_plain() -> None:
    total = _plain() + _plain()
    assert total.energy_dependent is False
    assert total.block_dependent is False
    assert total.block.shape == (M, M)
    np.testing.assert_array_equal(np.asarray(total.block), 2.0)


def test_add_plain_energy_promotes_energy() -> None:
    total = _plain() + _energy()
    assert total.energy_dependent is True
    assert total.block_dependent is False
    assert total.block.shape == (N_E, M, M)
    np.testing.assert_array_equal(np.asarray(total.block[:, 0, 0]), 1.0 + np.arange(N_E))


def test_add_plain_blocks_promotes_block() -> None:
    total = _plain() + _blocks()
    assert total.energy_dependent is False
    assert total.block_dependent is True
    assert total.block.shape == (N_B, M, M)
    np.testing.assert_array_equal(np.asarray(total.block[:, 0, 0]), 11.0 + np.arange(N_B))


def test_add_energy_blocks_promotes_both() -> None:
    total = _energy() + _blocks()
    assert total.energy_dependent is True
    assert total.block_dependent is True
    assert total.block.shape == (N_B, N_E, M, M)
    expected = np.arange(N_E)[None, :] + np.arange(N_B)[:, None] + 10.0
    np.testing.assert_array_equal(np.asarray(total.block[:, :, 0, 0]), expected)


def test_add_blocks_blocks_energy_promotes_energy() -> None:
    total = _blocks() + _blocks_energy()
    assert total.energy_dependent is True
    assert total.block_dependent is True
    assert total.block.shape == (N_B, N_E, M, M)
    expected = np.arange(N_B)[:, None] + 10.0 + np.arange(N_B * N_E).reshape(N_B, N_E) + 100.0
    np.testing.assert_array_equal(np.asarray(total.block[:, :, 0, 0]), expected)


def test_add_plain_blocks_energy_promotes_both() -> None:
    total = _plain() + _blocks_energy()
    assert total.energy_dependent is True
    assert total.block_dependent is True
    assert total.block.shape == (N_B, N_E, M, M)


def test_add_is_commutative_in_shape_and_value() -> None:
    left = _energy() + _blocks()
    right = _blocks() + _energy()
    assert left.block.shape == right.block.shape
    np.testing.assert_array_equal(np.asarray(left.block), np.asarray(right.block))


def test_sum_with_radd_zero() -> None:
    terms = [_blocks(), _blocks(), _plain()]
    total = sum(terms, start=0)
    assert isinstance(total, Interaction)
    assert total.block_dependent is True
    assert total.block.shape == (N_B, M, M)

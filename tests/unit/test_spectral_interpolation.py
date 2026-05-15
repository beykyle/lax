from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lax.spectral import pade_interpolate

pytest.importorskip("jax")


def test_pade_interpolate_reproduces_scalar_rational_function() -> None:
    """Padé interpolation reproduces an exactly representable scalar rational function."""

    knots = jnp.linspace(-0.6, 0.6, 5)
    values = _scalar_rational(knots)

    interpolant = pade_interpolate(values, knots, order=(1, 3))
    query = jnp.linspace(-0.5, 0.5, 11)
    result = np.asarray(interpolant(query))
    expected = np.asarray(_scalar_rational(query))

    assert np.allclose(result, expected, atol=1.0e-10, rtol=1.0e-10)


def test_pade_interpolate_handles_matrix_valued_samples() -> None:
    """Padé interpolation preserves trailing matrix dimensions."""

    knots = jnp.linspace(-0.4, 0.4, 5)
    values = _matrix_rational(knots)

    interpolant = pade_interpolate(values, knots, order=(1, 3))
    query = jnp.asarray([0.0, 0.25])
    result = np.asarray(interpolant(query))
    expected = np.asarray(_matrix_rational(query))

    assert result.shape == (2, 2, 2)
    assert np.allclose(result, expected, atol=1.0e-10, rtol=1.0e-10)


def test_pade_interpolate_handles_complex_values() -> None:
    """Padé interpolation supports complex-valued observables."""

    knots = jnp.linspace(-0.3, 0.3, 5)
    values = _complex_rational(knots)

    interpolant = pade_interpolate(values, knots, order=(1, 3))
    query = jnp.asarray([0.1 - 0.05j, -0.2 + 0.1j])
    result = np.asarray(interpolant(query))
    expected = np.asarray(_complex_rational(query))

    assert np.allclose(result, expected, atol=1.0e-10, rtol=1.0e-10)


def test_pade_interpolate_rejects_invalid_order() -> None:
    """Padé interpolation rejects orders inconsistent with the number of knots."""

    knots = jnp.linspace(-0.5, 0.5, 5)
    values = _scalar_rational(knots)

    with pytest.raises(ValueError, match="p \\+ q \\+ 1 = N_E"):
        pade_interpolate(values, knots, order=(2, 1))


def _scalar_rational(energy: jax.Array) -> jax.Array:
    """Return a smooth scalar rational function of degree (1, 3)."""

    numerator = 1.2 + 0.4 * energy
    denominator = 1.0 - 0.3 * energy + 0.1 * energy**2 - 0.02 * energy**3
    return numerator / denominator


def _matrix_rational(energy: jax.Array) -> jax.Array:
    """Return a 2x2 matrix-valued rational function."""

    base = _scalar_rational(energy)
    quadratic = (0.5 - 0.2 * energy) / (1.0 + 0.15 * energy + 0.05 * energy**2 - 0.01 * energy**3)
    cubic = (0.8 + 0.1 * energy) / (1.0 - 0.05 * energy + 0.04 * energy**2 + 0.02 * energy**3)
    constant = jnp.ones_like(base) * 0.25
    return jnp.stack(
        (
            jnp.stack((base, quadratic), axis=-1),
            jnp.stack((cubic, constant), axis=-1),
        ),
        axis=-2,
    )


def _complex_rational(energy: jax.Array) -> jax.Array:
    """Return a complex scalar rational function of degree (1, 3)."""

    numerator = (1.0 + 0.2j) + (0.3 - 0.1j) * energy
    denominator = 1.0 + (-0.15 + 0.05j) * energy + 0.08 * energy**2 + 0.01j * energy**3
    return numerator / denominator

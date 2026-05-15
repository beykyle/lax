"""Padé interpolation utilities for energy-dependent observables. See DESIGN.md §12."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import jax
import jax.numpy as jnp


def pade_interpolate(
    values: jax.Array,
    knots: jax.Array,
    order: tuple[int, int] | None = None,
) -> Callable[[jax.Array | float], jax.Array]:
    """Return a JIT-compiled Padé interpolator over the leading energy axis.

    Parameters
    ----------
    values
        Samples on the compile-time energy grid. Shape `(N_E, ...)`.
    knots
        Energy grid corresponding to `values`. Shape `(N_E,)`.
    order
        Optional `(p, q)` Padé order. Must satisfy `p + q + 1 = N_E`.

    Returns
    -------
    Callable[[jax.Array | float], jax.Array]
        JIT-compiled callable evaluating the interpolant at scalar or batched
        query energies.
    """

    if values.ndim < 1:
        msg = "`values` must have at least one dimension with leading energy samples."
        raise ValueError(msg)
    if knots.ndim != 1:
        msg = "`knots` must be a one-dimensional energy grid."
        raise ValueError(msg)

    n_energies = knots.shape[0]
    if values.shape[0] != n_energies:
        msg = (
            "The leading axis of `values` must match `knots`: "
            f"got {values.shape[0]} samples and {n_energies} knots."
        )
        raise ValueError(msg)
    if n_energies < 2:
        msg = "Padé interpolation requires at least two energy samples."
        raise ValueError(msg)

    if order is None:
        order = ((n_energies - 1) // 2, n_energies // 2)
    p, q = order
    _validate_order(n_energies, p, q)

    trailing_shape = values.shape[1:]
    flattened_values: jax.Array = values.reshape(n_energies, -1)
    center: jax.Array = jnp.mean(knots)
    shifted_knots: jax.Array = knots - center

    a_coeffs_flat: jax.Array
    b_coeffs_flat: jax.Array
    a_coeffs_flat, b_coeffs_flat = jax.vmap(
        _solve_pade_system,
        in_axes=(1, None, None, None),
        out_axes=(0, 0),
    )(flattened_values, shifted_knots, p, q)

    def evaluate(energy: jax.Array | float) -> jax.Array:
        """Evaluate the fitted Padé approximant at one or more energies."""

        energy_array: jax.Array = jnp.asarray(  # pyright: ignore[reportUnknownMemberType] -- JAX stubs expose asarray imprecisely.
            energy
        )
        shifted_energy: jax.Array = energy_array - center
        flat_energy: jax.Array = jnp.reshape(  # pyright: ignore[reportUnknownMemberType] -- JAX reshape stubs are imprecise.
            shifted_energy,
            (-1,),
        )

        numerator_powers = jnp.stack(
            [flat_energy**k for k in range(p + 1)],
            axis=-1,
        )
        denominator_powers = jnp.stack(
            [flat_energy**k for k in range(q + 1)],
            axis=-1,
        )

        numerator_flat: jax.Array = numerator_powers @ a_coeffs_flat.T
        denominator_flat: jax.Array = denominator_powers @ b_coeffs_flat.T
        result_flat: jax.Array = numerator_flat / denominator_flat
        result: jax.Array = result_flat.reshape(*shifted_energy.shape, *trailing_shape)
        return result

    return cast(
        Callable[[jax.Array | float], jax.Array],
        jax.jit(evaluate),  # pyright: ignore[reportUnknownMemberType] -- JAX jit wrappers are not precisely typed.
    )


def _solve_pade_system(
    samples: jax.Array,
    shifted_knots: jax.Array,
    p: int,
    q: int,
) -> tuple[jax.Array, jax.Array]:
    """Solve the linearized Padé system for one sampled observable."""

    n_energies = shifted_knots.shape[0]
    dtype = jnp.result_type(samples, shifted_knots)
    matrix: jax.Array = jnp.zeros(  # pyright: ignore[reportUnknownMemberType] -- JAX array constructors have imprecise stubs.
        (n_energies, p + q + 1),
        dtype=dtype,
    )

    for k in range(p + 1):
        matrix = matrix.at[:, k].set(-(shifted_knots**k))
    for k in range(1, q + 1):
        matrix = matrix.at[:, p + k].set(samples * (shifted_knots**k))

    rhs = -samples.astype(dtype)
    coefficients = cast(
        jax.Array,
        jnp.linalg.solve(  # pyright: ignore[reportUnknownMemberType] -- JAX linalg solve stubs lose the result type here.
            matrix,
            rhs,
        ),
    )
    a_coeffs: jax.Array = coefficients[: p + 1]
    b_coeffs: jax.Array = jnp.concatenate(
        (
            jnp.ones(  # pyright: ignore[reportUnknownMemberType] -- JAX array constructors have imprecise stubs.
                (1,),
                dtype=coefficients.dtype,
            ),
            coefficients[p + 1 :],
        )
    )
    return a_coeffs, b_coeffs


def _validate_order(n_energies: int, p: int, q: int) -> None:
    """Validate the requested Padé order."""

    if p < 0 or q < 0:
        msg = f"Padé orders must be non-negative, got ({p}, {q})."
        raise ValueError(msg)
    if p + q + 1 != n_energies:
        msg = f"Padé order must satisfy p + q + 1 = N_E: got {p} + {q} + 1 != {n_energies}."
        raise ValueError(msg)


__all__ = ["pade_interpolate"]

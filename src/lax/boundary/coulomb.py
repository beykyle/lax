"""Compile-time boundary-value computation with mpmath."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import mpmath as mp
import numpy as np
import scipy.special as sc

from lax.boundary._types import BoundaryValues
from lax.types import ChannelSpec

# reportUnknownMemberType is suppressed globally in pyproject.toml (JAX/mpmath stubs).
# reportCallIssue, reportUnknownArgumentType, and reportUnknownVariableType are
# suppressed here because mpmath's entire API is dynamically typed.
# pyright: reportCallIssue=false, reportUnknownArgumentType=false, reportUnknownVariableType=false


def compute_boundary_values(
    channels: tuple[ChannelSpec, ...],
    energies: np.ndarray,
    channel_radius: float,
    z1z2: tuple[int, int] | None = None,
    dps: int = 40,
    mass_factor_grid: np.ndarray | None = None,
) -> BoundaryValues:
    """Compute Coulomb and Whittaker boundary values at the channel radius.

    Evaluates the Hankel functions ``H± = G_L ± iF_L`` and their
    ``ρ d/dρ`` derivatives at ``ρ = k·a`` for every ``(energy, channel)``
    pair using ``mpmath`` with ``dps`` decimal digits of precision.  Closed
    channels use the Whittaker function ``W_{-η, ℓ+1/2}(2|k|a)`` instead.

    This function runs at compile time (pure Python/NumPy) and is never
    traced by JAX.

    Parameters
    ----------
    channels
        Channel definitions specifying ``l``, ``threshold``, and
        ``mass_factor`` for each channel.
    energies
        Compile-time energy grid in MeV, shape ``(N_E,)``.
    channel_radius
        Channel radius ``a`` in fm.
    z1z2
        Charge product ``(Z_1, Z_2)`` for Coulomb scattering.  Pass
        ``None`` for neutral particles (``η = 0``).
    dps
        ``mpmath`` decimal precision.  The default of 40 provides ample
        guard digits against cancellation near resonances.
    mass_factor_grid
        Per-energy ℏ²/2μ values in MeV·fm², shape ``(N_E,)``.  When
        provided, overrides ``channel.mass_factor`` in the wave-number and
        Sommerfeld-parameter computation at each energy.  Pass ``None`` to
        use the scalar ``ChannelSpec.mass_factor`` uniformly.

    Returns
    -------
    BoundaryValues
        Boundary values for all ``(N_E, N_c)`` pairs.  Open-channel entries
        use Coulomb Hankel functions; closed-channel entries use Whittaker
        functions and the ``is_open`` mask is ``False``.
    """

    mp.mp.dps = dps
    n_energies = len(energies)
    n_channels = len(channels)

    H_plus = np.zeros((n_energies, n_channels), dtype=np.complex128)
    H_minus = np.zeros((n_energies, n_channels), dtype=np.complex128)
    H_plus_p = np.zeros((n_energies, n_channels), dtype=np.complex128)
    H_minus_p = np.zeros((n_energies, n_channels), dtype=np.complex128)
    is_open = np.zeros((n_energies, n_channels), dtype=bool)
    k_values = np.zeros((n_energies, n_channels), dtype=np.float64)

    for energy_index, energy in enumerate(energies):
        for channel_index, channel in enumerate(channels):
            # Use per-energy mass_factor when provided; fall back to ChannelSpec value.
            effective_mass_factor = (
                float(mass_factor_grid[energy_index])
                if mass_factor_grid is not None
                else channel.mass_factor
            )
            effective_channel = (
                ChannelSpec(
                    l=channel.l,
                    threshold=channel.threshold,
                    mass_factor=effective_mass_factor,
                )
                if mass_factor_grid is not None
                else channel
            )
            relative_energy = float(energy - effective_channel.threshold)
            if relative_energy > 0.0:
                _fill_open_channel(
                    H_plus,
                    H_minus,
                    H_plus_p,
                    H_minus_p,
                    is_open,
                    k_values,
                    energy_index,
                    channel_index,
                    effective_channel,
                    relative_energy,
                    channel_radius,
                    z1z2,
                )
            else:
                _fill_closed_channel(
                    H_plus,
                    H_minus,
                    H_plus_p,
                    H_minus_p,
                    is_open,
                    k_values,
                    energy_index,
                    channel_index,
                    effective_channel,
                    relative_energy,
                    channel_radius,
                    z1z2,
                )

    return BoundaryValues(
        H_plus=_to_jax_array(H_plus),
        H_minus=_to_jax_array(H_minus),
        H_plus_p=_to_jax_array(H_plus_p),
        H_minus_p=_to_jax_array(H_minus_p),
        is_open=_to_jax_array(is_open),
        k=_to_jax_array(k_values),
    )


def _fill_open_channel(
    H_plus: np.ndarray,
    H_minus: np.ndarray,
    H_plus_p: np.ndarray,
    H_minus_p: np.ndarray,
    is_open: np.ndarray,
    k_values: np.ndarray,
    energy_index: int,
    channel_index: int,
    channel: ChannelSpec,
    relative_energy: float,
    channel_radius: float,
    z1z2: tuple[int, int] | None,
) -> None:
    """Fill one open-channel boundary-value entry."""

    k = np.sqrt(relative_energy / channel.mass_factor)
    rho = k * channel_radius
    eta = _sommerfeld(z1z2, k, channel.mass_factor) if z1z2 is not None else 0.0

    def coulombf_at_rho(rho_value: float) -> object:
        return _mp_coulombf(channel.l, eta, rho_value)

    def coulombg_at_rho(rho_value: float) -> object:
        return _mp_coulombg(channel.l, eta, rho_value)

    if eta == 0.0:
        f_value, g_value, d_f, d_g = _neutral_open_channel_values(channel.l, rho)
    else:
        f_value = complex(coulombf_at_rho(rho))  # pyright: ignore[reportArgumentType] -- mpmath values are complex-convertible numeric objects.
        g_value = complex(coulombg_at_rho(rho))  # pyright: ignore[reportArgumentType] -- mpmath values are complex-convertible numeric objects.
        d_f, d_g = _open_channel_derivatives(channel.l, eta, rho, f_value, g_value)

    H_plus[energy_index, channel_index] = g_value + 1.0j * f_value
    H_minus[energy_index, channel_index] = g_value - 1.0j * f_value
    H_plus_p[energy_index, channel_index] = rho * (d_g + 1.0j * d_f)
    H_minus_p[energy_index, channel_index] = rho * (d_g - 1.0j * d_f)
    is_open[energy_index, channel_index] = True
    k_values[energy_index, channel_index] = k


def _fill_closed_channel(
    H_plus: np.ndarray,
    H_minus: np.ndarray,
    H_plus_p: np.ndarray,
    H_minus_p: np.ndarray,
    is_open: np.ndarray,
    k_values: np.ndarray,
    energy_index: int,
    channel_index: int,
    channel: ChannelSpec,
    relative_energy: float,
    channel_radius: float,
    z1z2: tuple[int, int] | None,
) -> None:
    """Fill one closed-channel boundary-value entry."""

    k = np.sqrt(-relative_energy / channel.mass_factor)
    rho = 2.0 * k * channel_radius
    eta = _sommerfeld(z1z2, k, channel.mass_factor) if z1z2 is not None else 0.0

    def whitw_at_rho(rho_value: float) -> object:
        return _mp_whittaker_w(-eta, channel.l + 0.5, rho_value)

    value = complex(whitw_at_rho(rho))  # pyright: ignore[reportArgumentType] -- mpmath values are complex-convertible numeric objects.
    derivative = _differentiate(whitw_at_rho, rho)

    H_plus[energy_index, channel_index] = value
    H_minus[energy_index, channel_index] = value
    H_plus_p[energy_index, channel_index] = rho * derivative
    H_minus_p[energy_index, channel_index] = rho * derivative
    is_open[energy_index, channel_index] = False
    k_values[energy_index, channel_index] = k


def _sommerfeld(z1z2: tuple[int, int], k: float, mass_factor: float) -> float:
    """Return the Sommerfeld parameter in the fm^-2 convention."""

    z1, z2 = z1z2
    return z1 * z2 * 1.44 / (2.0 * mass_factor * k)


def _mp_coulombf(l: int, eta: float, rho: float) -> object:
    """Evaluate the regular Coulomb function with mpmath."""

    value: object = mp.coulombf(  # pyright: ignore[reportUnknownArgumentType] -- mpmath APIs are dynamically typed.
        l,
        eta,
        rho,
    )
    return value


def _mp_coulombg(l: int, eta: float, rho: float) -> object:
    """Evaluate the irregular Coulomb function with mpmath."""

    value: object = mp.coulombg(  # pyright: ignore[reportUnknownArgumentType] -- mpmath APIs are dynamically typed.
        l,
        eta,
        rho,
    )
    return value


def _mp_whittaker_w(kappa: float, mu: float, rho: float) -> object:
    """Evaluate the Whittaker W function with mpmath."""

    value: object = mp.whitw(  # pyright: ignore[reportUnknownArgumentType] -- mpmath APIs are dynamically typed.
        kappa,
        mu,
        rho,
    )
    return value


def _differentiate(function: Callable[[float], object], rho: float) -> complex:
    """Differentiate an mpmath-callable function at one point."""

    return complex(
        mp.diff(  # pyright: ignore[reportUnknownArgumentType] -- mpmath differentiation is dynamically typed.
            function,
            rho,
        )
    )


def _open_channel_derivatives(
    l: int,
    eta: float,
    rho: float,
    F: complex,
    G: complex,
) -> tuple[complex, complex]:
    """Return dF/drho and dG/drho using Coulomb-function recurrences."""

    if eta == 0.0:
        dF = complex(sc.spherical_jn(l, rho) + rho * sc.spherical_jn(l, rho, derivative=True))
        dG = complex(-sc.spherical_yn(l, rho) - rho * sc.spherical_yn(l, rho, derivative=True))
        return dF, dG

    next_F = complex(_mp_coulombf(l + 1, eta, rho))  # pyright: ignore[reportArgumentType] -- mpmath values are complex-convertible numeric objects.
    next_G = complex(_mp_coulombg(l + 1, eta, rho))  # pyright: ignore[reportArgumentType] -- mpmath values are complex-convertible numeric objects.
    ratio = np.sqrt(1.0 + eta**2 / (l + 1) ** 2)
    shift = (l + 1) / rho + eta / (l + 1)
    return shift * F - ratio * next_F, shift * G - ratio * next_G


def _neutral_open_channel_values(l: int, rho: float) -> tuple[complex, complex, complex, complex]:
    """Return neutral-particle F, G and rho-derivatives from spherical Bessel functions."""

    F = complex(rho * sc.spherical_jn(l, rho))
    G = complex(-rho * sc.spherical_yn(l, rho))
    dF = complex(sc.spherical_jn(l, rho) + rho * sc.spherical_jn(l, rho, derivative=True))
    dG = complex(-sc.spherical_yn(l, rho) - rho * sc.spherical_yn(l, rho, derivative=True))
    return F, G, dF, dG


def _to_jax_array(values: np.ndarray) -> jax.Array:
    """Convert compile-time NumPy arrays to runtime JAX arrays."""

    array: jax.Array = jnp.asarray(values)
    return array


__all__ = ["compute_boundary_values"]

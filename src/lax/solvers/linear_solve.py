"""Per-energy direct R-matrix solver."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np

from lax.boundary._types import (
    BoundaryValues,
    DirectGridObservable,
    DirectRMatrixKernel,
    Mesh,
    OperatorMatrices,
    PropagationMatrices,
)
from lax.types import ChannelSpec

from .assembly import assemble_block_hamiltonian, build_Q


@dataclass(frozen=True)
class _DirectRMatrixKernel:
    """Pickle-safe direct R-matrix kernel backed by a module-level JIT function."""

    mesh: Mesh
    operators: OperatorMatrices
    channels: tuple[ChannelSpec, ...]
    energies: jax.Array
    q: jax.Array
    channel_radius: float
    matrix_size: int
    mass_factor: float
    boundary: BoundaryValues | None

    def __call__(self, potential: jax.Array) -> jax.Array:
        """Evaluate the direct R-matrix on the compile-time energy grid."""

        return cast(
            jax.Array,
            _RMATRIX_DIRECT_JIT(
                potential,
                self.mesh,
                self.operators,
                self.channels,
                self.energies,
                self.q,
                self.channel_radius,
                self.matrix_size,
                self.mass_factor,
                self.boundary,
            ),
        )


@dataclass(frozen=True)
class _DirectRMatrixGridObservable:
    """Pickle-safe aligned-grid direct R-matrix observable."""

    mesh: Mesh
    operators: OperatorMatrices
    channels: tuple[ChannelSpec, ...]
    energies: jax.Array
    q: jax.Array
    channel_radius: float
    matrix_size: int
    mass_factor: float
    boundary: BoundaryValues | None

    def __call__(self, potentials: jax.Array) -> jax.Array:
        """Evaluate `R(E_i; V_i)` across the compile-time energy grid."""

        return cast(
            jax.Array,
            _RMATRIX_DIRECT_GRID_JIT(
                potentials,
                self.mesh,
                self.operators,
                self.channels,
                self.energies,
                self.q,
                self.channel_radius,
                self.matrix_size,
                self.mass_factor,
                self.boundary,
            ),
        )


def make_rmatrix_direct_kernel(
    mesh: Mesh,
    operators: OperatorMatrices,
    channels: tuple[ChannelSpec, ...],
    energies: jax.Array,
    boundary: BoundaryValues | None,
) -> DirectRMatrixKernel:
    """Build a JIT-compiled ``rmatrix_direct(V) → R`` kernel for the compile-time grid.

    The returned kernel solves ``C(E) X = Q`` for each compile-time energy via
    ``jnp.linalg.solve``, bypassing the eigendecomposition.  Supports real and
    complex potentials, local and non-local, propagated and non-propagated meshes.
    [DESIGN.md §11.3]

    Parameters
    ----------
    mesh
        Compiled mesh; if ``mesh.propagation`` is not ``None``, a subinterval
        propagation recursion is used instead of a global solve.
    operators
        Precomputed operator matrices (``TpL`` required).
    channels
        Channel definitions.
    energies
        Compile-time energy grid in MeV, shape ``(N_E,)``.
    boundary
        Compile-time boundary values for S-matrix matching, or ``None`` if only
        the R-matrix is needed.

    Returns
    -------
    DirectRMatrixKernel
        JIT-compiled callable: ``kernel(V) → R`` with shape ``(N_E, N_c, N_c)``.
    """

    q = build_Q(mesh, channels)
    channel_radius = mesh.scale
    matrix_size = mesh.n * len(channels)
    mass_factor = _uniform_mass_factor(channels)
    return cast(
        DirectRMatrixKernel,
        _DirectRMatrixKernel(
            mesh=mesh,
            operators=operators,
            channels=channels,
            energies=energies,
            q=q,
            channel_radius=channel_radius,
            matrix_size=matrix_size,
            mass_factor=mass_factor,
            boundary=boundary,
        ),
    )


def make_rmatrix_direct_grid_observable(
    mesh: Mesh,
    operators: OperatorMatrices,
    channels: tuple[ChannelSpec, ...],
    energies: jax.Array,
    boundary: BoundaryValues | None,
) -> DirectGridObservable:
    """Build a JIT-compiled aligned-grid ``R(E_i; V_i)`` kernel.

    For energy-dependent potentials: given a stack of potentials
    ``V_grid[i]`` at compile-time energy ``E_i``, returns ``R[i]`` evaluated
    with that exact ``(V_i, E_i)`` pairing — the physically correct aligned-grid
    evaluation.

    Parameters
    ----------
    mesh
        Compiled mesh.
    operators
        Precomputed operator matrices.
    channels
        Channel definitions.
    energies
        Compile-time energy grid in MeV, shape ``(N_E,)``.
    boundary
        Compile-time boundary values, or ``None``.

    Returns
    -------
    DirectGridObservable
        JIT-compiled callable: ``kernel(V_grid) → R`` where ``V_grid`` has a
        leading ``N_E`` axis and ``R`` has shape ``(N_E, N_c, N_c)``.
    """

    q = build_Q(mesh, channels)
    channel_radius = mesh.scale
    matrix_size = mesh.n * len(channels)
    mass_factor = _uniform_mass_factor(channels)
    return cast(
        DirectGridObservable,
        _DirectRMatrixGridObservable(
            mesh=mesh,
            operators=operators,
            channels=channels,
            energies=energies,
            q=q,
            channel_radius=channel_radius,
            matrix_size=matrix_size,
            mass_factor=mass_factor,
            boundary=boundary,
        ),
    )


def _rmatrix_direct(
    potential: jax.Array,
    mesh: Mesh,
    operators: OperatorMatrices,
    channels: tuple[ChannelSpec, ...],
    energies: jax.Array,
    q: jax.Array,
    channel_radius: float,
    matrix_size: int,
    mass_factor: float,
    boundary: BoundaryValues | None,
) -> jax.Array:
    """Return the direct R-matrix across the compile-time energy grid.

    Dispatches to the propagated or non-propagated path depending on
    ``mesh.propagation``, and to the local or non-local potential path
    depending on ``potential.ndim``.

    Parameters
    ----------
    potential
        Assembled potential in MeV.  Local: ``(N_c, N_c, N)``; non-local:
        ``(N_c, N_c, N, N)``.
    mesh, operators, channels, energies, q, channel_radius, matrix_size, mass_factor, boundary
        Compile-time cached data forwarded from the kernel dataclass.

    Returns
    -------
    jax.Array
        R-matrix on the compile-time energy grid, shape ``(N_E, N_c, N_c)``.
    """

    if mesh.propagation is not None and potential.ndim == 3:
        if boundary is None:
            msg = "Boundary values are required for propagated direct R-matrix solves."
            raise ValueError(msg)
        propagation = mesh.propagation

        def propagated_one_energy(
            energy: jax.Array,
            h_plus: jax.Array,
            h_plus_p: jax.Array,
            is_open: jax.Array,
        ) -> jax.Array:
            return _propagated_rmatrix_at_energy(
                potential,
                propagation,
                channels,
                energy,
                h_plus,
                h_plus_p,
                is_open,
                mass_factor,
            )

        result: jax.Array = jax.vmap(propagated_one_energy)(
            energies,
            boundary.H_plus,
            boundary.H_plus_p,
            boundary.is_open,
        )
        return result

    if mesh.propagation is not None and potential.ndim == 4:
        msg = (
            "Subinterval propagation is defined only for local potentials in the direct "
            "linear-solve formulation. Non-local propagated solves are not mathematically "
            "supported."
        )
        raise ValueError(msg)

    hamiltonian = assemble_block_hamiltonian(
        mesh,
        operators,
        channels,
        potential,
    )

    def one_energy(energy: jax.Array) -> jax.Array:
        energy_dimless = energy / mass_factor
        matrix = hamiltonian - energy_dimless * jnp.eye(
            matrix_size,
            dtype=hamiltonian.dtype,
        )
        solved = cast(
            jax.Array,
            jnp.linalg.solve(
                matrix,
                q,
            ),
        )
        values: jax.Array = (q.T @ solved) / channel_radius
        return values

    result: jax.Array = jax.vmap(one_energy)(energies)
    return result


def _rmatrix_direct_grid(
    potentials: jax.Array,
    mesh: Mesh,
    operators: OperatorMatrices,
    channels: tuple[ChannelSpec, ...],
    energies: jax.Array,
    q: jax.Array,
    channel_radius: float,
    matrix_size: int,
    mass_factor: float,
    boundary: BoundaryValues | None,
) -> jax.Array:
    """Return aligned-grid ``R(E_i; V_i)`` samples for energy-dependent potentials.

    Evaluates each ``(V_i, E_i)`` pair independently — the diagonal of the
    ``(N_E, N_E)`` Cartesian product — so the result is physically correct for
    potentials that depend on energy.

    Parameters
    ----------
    potentials
        Per-energy potentials in MeV.  Local: ``(N_E, N_c, N_c, N)``; non-local:
        ``(N_E, N_c, N_c, N, N)``.
    mesh, operators, channels, energies, q, channel_radius, matrix_size, mass_factor, boundary
        Compile-time cached data.

    Returns
    -------
    jax.Array
        R-matrix samples, shape ``(N_E, N_c, N_c)``.
    """

    if mesh.propagation is not None and potentials.ndim == 4:
        if boundary is None:
            msg = "Boundary values are required for propagated direct R-matrix solves."
            raise ValueError(msg)
        propagation = mesh.propagation

        def propagated_one_energy(
            potential: jax.Array,
            energy: jax.Array,
            h_plus: jax.Array,
            h_plus_p: jax.Array,
            is_open: jax.Array,
        ) -> jax.Array:
            return _propagated_rmatrix_at_energy(
                potential,
                propagation,
                channels,
                energy,
                h_plus,
                h_plus_p,
                is_open,
                mass_factor,
            )

        result: jax.Array = jax.vmap(propagated_one_energy)(
            potentials,
            energies,
            boundary.H_plus,
            boundary.H_plus_p,
            boundary.is_open,
        )
        return result

    if mesh.propagation is not None and potentials.ndim == 5:
        msg = (
            "Subinterval propagation is defined only for local potentials in the direct "
            "linear-solve formulation. Non-local propagated solves are not mathematically "
            "supported."
        )
        raise ValueError(msg)

    def one_energy(potential: jax.Array, energy: jax.Array) -> jax.Array:
        hamiltonian = assemble_block_hamiltonian(
            mesh,
            operators,
            channels,
            potential,
        )
        energy_dimless = energy / mass_factor
        matrix = hamiltonian - energy_dimless * jnp.eye(
            matrix_size,
            dtype=hamiltonian.dtype,
        )
        solved = cast(
            jax.Array,
            jnp.linalg.solve(
                matrix,
                q,
            ),
        )
        return (q.T @ solved) / channel_radius

    return jax.vmap(one_energy)(
        potentials,
        energies,
    )


def _propagated_rmatrix_at_energy(
    potential: jax.Array,
    propagation: PropagationMatrices,
    channels: tuple[ChannelSpec, ...],
    energy: jax.Array,
    h_plus: jax.Array,
    h_plus_p: jax.Array,
    is_open: jax.Array,
    mass_factor: float,
) -> jax.Array:
    """Return the propagated effective R-matrix at one energy.

    Implements the Descouvemont R-matrix subinterval propagation algorithm:
    the internal region is divided into ``n_intervals`` subintervals; the
    local R-matrix for each interval is computed by a direct solve, then
    matched at the shared boundary using the Bloch overlap matrices ``blo0``,
    ``blo1``, ``blo2`` until the effective R-matrix at the outer surface is
    obtained.

    Parameters
    ----------
    potential
        Local potential in MeV, shape ``(N_c, N_c, N)``.
    propagation
        Precomputed subinterval matrices from :class:`PropagationMatrices`.
    channels
        Channel definitions.
    energy
        Scalar energy in MeV.
    h_plus, h_plus_p
        Outgoing Coulomb/Whittaker function and its derivative at the channel
        surface, shape ``(N_c,)``.
    is_open
        Boolean mask for open channels, shape ``(N_c,)``.
    mass_factor
        ℏ²/2μ in MeV·fm².

    Returns
    -------
    jax.Array
        Effective R-matrix, shape ``(N_c, N_c)``.
    """

    nr = propagation.basis_size_per_interval
    ns = propagation.n_intervals
    interval_width = propagation.interval_width
    channel_count = len(channels)
    matrix_size = channel_count * nr
    dtype = potential.dtype
    local_q1 = propagation.q1.astype(dtype)
    local_q2 = propagation.q2.astype(dtype)
    kinetic = propagation.kinetic.astype(dtype)
    blo0 = propagation.blo0.astype(dtype)
    blo1 = propagation.blo1.astype(dtype)
    blo2 = propagation.blo2.astype(dtype)
    local_nodes = propagation.local_nodes.astype(dtype)
    closed_ratio = jnp.where(
        is_open,
        jnp.zeros_like(h_plus, dtype=dtype),
        jnp.real(h_plus_p / h_plus).astype(dtype),
    )
    thresholds = np.asarray([channel.threshold for channel in channels], dtype=np.float64)
    threshold_array: jax.Array = jnp.asarray(
        thresholds,
        dtype=dtype,
    )
    qk_sq = jnp.abs((energy - threshold_array) / mass_factor)

    crma0 = jnp.zeros((channel_count, channel_count), dtype=dtype)
    for interval_index in range(ns):
        interval_start = interval_index * nr
        interval_stop = interval_start + nr
        interval_matrix: jax.Array = jnp.zeros(
            (matrix_size, matrix_size),
            dtype=dtype,
        )
        interval_positions = (interval_index + local_nodes) * interval_width
        if interval_index == 0:
            boundary_block = blo0 / ((interval_index + 1) * interval_width)
        else:
            boundary_block = blo2 / ((interval_index + 1) * interval_width) - blo1 / (
                interval_index * interval_width
            )
        for channel_index, channel in enumerate(channels):
            row = channel_index * nr
            row_slice = slice(row, row + nr)
            diagonal_block = kinetic[interval_index]
            diagonal_block = diagonal_block + (
                channel.l * (channel.l + 1) / (interval_positions**2)
            ) * jnp.eye(nr, dtype=dtype)
            sign = jnp.where(is_open[channel_index], -1.0, 1.0).astype(dtype)
            diagonal_block = diagonal_block + sign * qk_sq[channel_index] * jnp.eye(
                nr,
                dtype=dtype,
            )
            diagonal_block = diagonal_block - closed_ratio[channel_index] * boundary_block

            for coupled_index in range(channel_count):
                column = coupled_index * nr
                column_slice = slice(column, column + nr)
                block = jnp.zeros((nr, nr), dtype=dtype)
                if channel_index == coupled_index:
                    block = block + diagonal_block
                block = block + jnp.diag(
                    potential[channel_index, coupled_index, interval_start:interval_stop]
                    / mass_factor
                )
                interval_matrix = interval_matrix.at[row_slice, column_slice].set(block)

        q1_matrix = _surface_projector(channel_count, nr, local_q1[interval_index], dtype)
        q2_matrix = _surface_projector(channel_count, nr, local_q2[interval_index], dtype)
        solved_q1 = cast(
            jax.Array,
            jnp.linalg.solve(interval_matrix, q1_matrix),
        )
        solved_q2 = cast(
            jax.Array,
            jnp.linalg.solve(interval_matrix, q2_matrix),
        )
        crma_11 = q2_matrix.T @ solved_q2
        crma_12 = q1_matrix.T @ solved_q2
        crma_22 = q1_matrix.T @ solved_q1
        if interval_index == 0:
            crma0 = crma_11 / interval_width
        else:
            boundary_matrix = crma_22 + crma0 * (interval_index * interval_width)
            correction = cast(
                jax.Array,
                jnp.linalg.solve(boundary_matrix, crma_12),
            )
            crma0 = (crma_11 - crma_12.T @ correction) / ((interval_index + 1) * interval_width)

    return crma0


def _surface_projector(
    channel_count: int,
    basis_size_per_interval: int,
    values: jax.Array,
    dtype: jax.typing.DTypeLike,
) -> jax.Array:
    """Return the per-interval surface projector used by the propagation recursion."""

    projector: jax.Array = jnp.zeros(
        (channel_count * basis_size_per_interval, channel_count),
        dtype=dtype,
    )
    for channel_index in range(channel_count):
        start = channel_index * basis_size_per_interval
        stop = start + basis_size_per_interval
        projector = projector.at[start:stop, channel_index].set(values)
    return projector


_RMATRIX_DIRECT_JIT = jax.jit(
    _rmatrix_direct,
    static_argnames=("channels", "matrix_size"),
)
_RMATRIX_DIRECT_GRID_JIT = jax.jit(
    _rmatrix_direct_grid,
    static_argnames=("channels", "matrix_size"),
)


def _uniform_mass_factor(channels: tuple[ChannelSpec, ...]) -> float:
    """Return the shared mass factor expected by the MVP direct solver path."""

    mass_factor = channels[0].mass_factor
    for channel in channels[1:]:
        if channel.mass_factor != mass_factor:
            msg = "The MVP direct solver path requires a uniform mass_factor across channels."
            raise ValueError(msg)
    return mass_factor


__all__ = ["make_rmatrix_direct_grid_observable", "make_rmatrix_direct_kernel"]

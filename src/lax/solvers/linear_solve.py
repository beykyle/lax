"""Per-energy direct R-matrix solver."""

from __future__ import annotations

from typing import cast

import jax
import jax.numpy as jnp

from lax.boundary._types import DirectRMatrixKernel, Mesh, OperatorMatrices
from lax.types import ChannelSpec

from .assembly import assemble_block_hamiltonian, build_Q


def make_rmatrix_direct_kernel(
    mesh: Mesh,
    operators: OperatorMatrices,
    channels: tuple[ChannelSpec, ...],
    energies: jax.Array,
) -> DirectRMatrixKernel:
    """Build `rmatrix_direct(V) -> R(E)` on the compile-time energy grid. [DESIGN.md §11.3]"""

    q = build_Q(mesh, channels)
    channel_radius = mesh.scale
    matrix_size = mesh.n * len(channels)
    mass_factor = _uniform_mass_factor(channels)

    def rmatrix_direct(potential: jax.Array) -> jax.Array:
        hamiltonian = assemble_block_hamiltonian(mesh, operators, channels, potential)

        def one_energy(energy: jax.Array) -> jax.Array:
            energy_dimless = energy / mass_factor
            matrix = hamiltonian - energy_dimless * jnp.eye(  # pyright: ignore[reportUnknownMemberType] -- JAX array constructors have imprecise stubs.
                matrix_size,
                dtype=hamiltonian.dtype,
            )
            solved = cast(
                jax.Array,
                jnp.linalg.solve(  # pyright: ignore[reportUnknownMemberType] -- JAX linalg solve stubs lose the result type here.
                    matrix,
                    q,
                ),
            )
            values: jax.Array = (q.T @ solved) / channel_radius
            return values

        result: jax.Array = jax.vmap(one_energy)(  # pyright: ignore[reportUnknownMemberType] -- JAX vmap wrappers are not precisely typed.
            energies
        )
        return result

    return cast(
        DirectRMatrixKernel,
        jax.jit(rmatrix_direct),  # pyright: ignore[reportUnknownMemberType] -- JAX jit wrappers are not precisely typed.
    )


def _uniform_mass_factor(channels: tuple[ChannelSpec, ...]) -> float:
    """Return the shared mass factor expected by the MVP direct solver path."""

    mass_factor = channels[0].mass_factor
    for channel in channels[1:]:
        if channel.mass_factor != mass_factor:
            msg = "The MVP direct solver path requires a uniform mass_factor across channels."
            raise ValueError(msg)
    return mass_factor


__all__ = ["make_rmatrix_direct_kernel"]

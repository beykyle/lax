"""Hamiltonian assembly helpers for compiled solver kernels."""

from __future__ import annotations

from typing import cast

import jax
import jax.numpy as jnp

from lax.types import ChannelSpec, Mesh, OperatorMatrices


def channel_arrays(
    channels: tuple[ChannelSpec, ...],
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return per-channel ``(centrifugal, thresholds, mass_factors)`` arrays.

    Converts the static :class:`ChannelSpec` data into ``(N_c,)`` arrays —
    ``centrifugal[c] = ℓ_c(ℓ_c + 1)`` — so the assembly can be expressed as a
    pure array program and vmapped over a leading symmetry-block axis
    (DESIGN.md §15.5).
    """

    centrifugal = jnp.array([ch.l * (ch.l + 1) for ch in channels], dtype=jnp.float64)
    thresholds = jnp.array([ch.threshold for ch in channels], dtype=jnp.float64)
    mass_factors = jnp.array([ch.mass_factor for ch in channels], dtype=jnp.float64)
    return centrifugal, thresholds, mass_factors


def block_group_arrays(
    block_groups: tuple[tuple[ChannelSpec, ...], ...],
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return stacked ``(N_b, N_c)`` channel arrays for a set of symmetry blocks.

    The stacked form feeds the block-batched kernels: each row is one block's
    ``channel_arrays`` output.  [DESIGN.md §15.5]
    """

    rows = [channel_arrays(group) for group in block_groups]
    centrifugal = jnp.stack([row[0] for row in rows])
    thresholds = jnp.stack([row[1] for row in rows])
    mass_factors = jnp.stack([row[2] for row in rows])
    return centrifugal, thresholds, mass_factors


def lift_to_blocks(block: jax.Array, has_block_axis: bool, n_blocks: int) -> jax.Array:
    """Broadcast a block-independent array onto the leading ``(N_b,)`` axis.

    Free under jit — each block's Hamiltonian is materialized regardless,
    since the per-block centrifugal differs (DESIGN.md §15.5).
    """

    if has_block_axis:
        return block
    return jnp.broadcast_to(block, (n_blocks, *block.shape))


def assemble_hamiltonian_arrays(
    mesh: Mesh,
    operators: OperatorMatrices,
    centrifugal: jax.Array,
    thresholds: jax.Array,
    mass_factors: jax.Array,
    potential: jax.Array,
) -> jax.Array:
    """Assemble the Bloch-augmented block Hamiltonian from per-channel arrays.

    Array-parameterized core of :func:`assemble_block_hamiltonian`: the
    per-channel centrifugal ``ℓ_c(ℓ_c+1)``, threshold, and mass-factor values
    arrive as traced ``(N_c,)`` arrays instead of static :class:`ChannelSpec`
    fields, so the assembly composes with ``jax.vmap`` over a leading
    symmetry-block axis (DESIGN.md §15.5).  ``N_c`` itself stays static.

    Parameters
    ----------
    mesh
        Compiled mesh supplying the basis size.
    operators
        Precomputed operator matrices; ``TpL`` and ``inv_r2`` must be present.
    centrifugal
        Per-channel ``ℓ_c(ℓ_c + 1)`` values, shape ``(N_c,)``.
    thresholds
        Per-channel thresholds in MeV, shape ``(N_c,)``.
    mass_factors
        Per-channel ℏ²/2μ in MeV·fm², shape ``(N_c,)``.
    potential
        Assembled potential in MeV.  Shape ``(N_c, N_c, N)`` for local,
        ``(N_c, N_c, N, N)`` for non-local, or ``(M, M)`` pre-assembled.

    Returns
    -------
    jax.Array
        Block Hamiltonian, shape ``(N_c·N, N_c·N)``, in MeV.
    """

    channel_count = centrifugal.shape[0]
    basis_size = mesh.n
    t_plus_l = _require_operator(operators.TpL, "TpL")
    inv_r2 = _require_operator(operators.inv_r2, "inv_r2")
    # Match the channel scalars to the baked-operator precision so a float32
    # compile (§14.1) stays float32 instead of promoting back to float64.
    centrifugal = centrifugal.astype(t_plus_l.dtype)
    thresholds = thresholds.astype(t_plus_l.dtype)
    mass_factors = mass_factors.astype(t_plus_l.dtype)

    blocks: list[jax.Array] = []
    for channel_index in range(channel_count):
        row_blocks: list[jax.Array] = []
        m_c = mass_factors[channel_index]
        angular_term = centrifugal[channel_index]
        threshold = thresholds[channel_index]
        for coupled_index in range(channel_count):
            block: jax.Array = jnp.zeros(
                (basis_size, basis_size),
                dtype=potential.dtype,
            )
            if channel_index == coupled_index:
                block = (
                    block
                    + m_c * (t_plus_l + angular_term * inv_r2)
                    + threshold
                    * jnp.eye(
                        basis_size,
                        dtype=potential.dtype,
                    )
                )
            if potential.ndim == 3:
                block = block + _diagonal_from_vector(potential[channel_index, coupled_index])
            elif potential.ndim == 2:
                # Pre-assembled (M, M) block (e.g. from Interaction.block): extract sub-block
                # by slicing rather than channel indexing.
                rs = channel_index * basis_size
                cs = coupled_index * basis_size
                block = block + potential[rs : rs + basis_size, cs : cs + basis_size]
            else:
                block = block + potential[channel_index, coupled_index]
            row_blocks.append(block)
        blocks.append(jnp.concatenate(row_blocks, axis=1))
    matrix: jax.Array = jnp.concatenate(blocks, axis=0)
    return matrix


def assemble_block_hamiltonian(
    mesh: Mesh,
    operators: OperatorMatrices,
    channels: tuple[ChannelSpec, ...],
    potential: jax.Array,
) -> jax.Array:
    """Assemble the Bloch-augmented block Hamiltonian in MeV units.

    Builds the ``(N_c·N, N_c·N)`` matrix whose eigendecomposition drives
    all spectral observables.  Diagonal blocks contain
    ``m_c·(T+L) + E_c·I``; off-diagonal blocks contain the coupling
    potential.  [DESIGN.md §11.5]

    Each channel's kinetic block is scaled by ``m_c`` (ℏ²/2μ in MeV·fm²)
    so that the assembled matrix is in MeV throughout.  The potential ``V``
    is added without rescaling — it must already be in MeV.

    Parameters
    ----------
    mesh
        Compiled mesh supplying ``TpL`` and ``inv_r2``.
    operators
        Precomputed operator matrices; ``TpL`` must be present.
    channels
        Channel definitions (``l``, ``threshold``, ``mass_factor``).
        For energy-dependent μ, pass channels whose ``mass_factor`` fields
        carry the per-energy values (e.g. a vmapped slice of
        ``mass_factor_grid``).
    potential
        Assembled potential in MeV.  Shape ``(N_c, N_c, N)`` for local
        or ``(N_c, N_c, N, N)`` for non-local.

    Returns
    -------
    jax.Array
        Block Hamiltonian, shape ``(N_c·N, N_c·N)``, in MeV.
    """

    centrifugal, thresholds, mass_factors = channel_arrays(channels)
    return assemble_hamiltonian_arrays(
        mesh, operators, centrifugal, thresholds, mass_factors, potential
    )


def build_Q(mesh: Mesh, channels: tuple[ChannelSpec, ...]) -> jax.Array:
    """Return the surface projector matrix Q.

    ``Q[c·N + j, c'] = δ_{cc'} φ_j(a)`` — a block-diagonal matrix that
    picks out the boundary-surface component of each channel's eigenvectors.
    The surface amplitudes are then ``γ = U^T Q``.  [DESIGN.md §11.5]

    Parameters
    ----------
    mesh
        Compiled mesh supplying ``basis_at_boundary`` (shape ``(N,)``).
    channels
        Channel definitions; only the count is used.

    Returns
    -------
    jax.Array
        Surface projector, shape ``(N_c·N, N_c)``.
    """

    channel_count = len(channels)
    basis_size = mesh.n
    boundary = mesh.basis_at_boundary

    q: jax.Array = jnp.zeros(
        (channel_count * basis_size, channel_count),
        dtype=boundary.dtype,
    )
    for channel_index in range(channel_count):
        q = q.at[channel_index * basis_size : (channel_index + 1) * basis_size, channel_index].set(
            boundary
        )
    return q


def _diagonal_from_vector(values: jax.Array) -> jax.Array:
    """Construct a diagonal matrix from one vector."""

    matrix: jax.Array = jnp.diag(values)
    return matrix


def _require_operator(operator: jax.Array | None, name: str) -> jax.Array:
    """Require a precomputed operator matrix."""

    if operator is None:
        msg = f"OperatorMatrices.{name} is required for solver assembly."
        raise ValueError(msg)
    return operator


def uniform_mass_factor(channels: tuple[ChannelSpec, ...], context: str = "solver path") -> float:
    """Return the single mass factor shared by all channels, or raise.

    Several kernels (the spectral eigensolve and the propagated direct solve)
    fold ``ℏ²/2μ`` out of the Hamiltonian using one scalar.  They are only
    correct when every channel shares that mass factor, so this validates the
    assumption at compile time rather than silently producing wrong physics.

    Parameters
    ----------
    channels
        Channel definitions.
    context
        Short description of the caller, interpolated into the error message.

    Returns
    -------
    float
        The shared ``channels[0].mass_factor``.

    Raises
    ------
    ValueError
        If the channels do not all share one mass factor.
    """

    mass_factor = channels[0].mass_factor
    for channel in channels[1:]:
        if channel.mass_factor != mass_factor:
            msg = (
                f"The {context} requires a uniform mass_factor across channels; "
                "use the per-energy/per-channel direct grid path for multi-μ systems."
            )
            raise ValueError(msg)
    return cast(float, mass_factor)


def uniform_block_mass_factor(
    block_groups: tuple[tuple[ChannelSpec, ...], ...],
    context: str = "solver path",
) -> float:
    """Return the single mass factor shared by all channels of all blocks, or raise.

    The spectral path folds one ℏ²/2μ out of the Hamiltonian (§15.5), so a
    blocks-compiled spectral kernel requires one uniform mass factor across
    the entire block set; per-block μ remains a direct-path feature.
    """

    return uniform_mass_factor(
        tuple(channel for group in block_groups for channel in group),
        context=context,
    )


def add_block_axis[T](tree: T) -> T:
    """Insert a leading length-1 block axis on every leaf of a pytree.

    Used in ``channels=`` mode to feed single-block data through the batched
    (blocks-always) kernels; the inverse of :func:`take_block0`.
    """

    def lift(leaf: jax.Array) -> jax.Array:
        return leaf[None]

    return cast("T", jax.tree.map(lift, tree))


def take_block0[T](tree: T) -> T:
    """Squeeze the leading ``N_b == 1`` block axis off every leaf of a pytree.

    Used in ``channels=`` mode to restore the unbatched public shapes after a
    batched kernel ran the single block; the inverse of :func:`add_block_axis`.
    """

    def take(leaf: jax.Array) -> jax.Array:
        return leaf[0]

    return cast("T", jax.tree.map(take, tree))


def reject_block_dependent(potential: object, entry_point: str) -> None:
    """Raise when a block-dependent Interaction reaches a channels-mode kernel.

    Single chokepoint for the guard shared by every kernel compiled with
    ``channels=`` (DESIGN.md §15.5): block-dependent Interactions are only
    meaningful on a solver compiled with ``blocks=``.
    """

    if getattr(potential, "block_dependent", False):
        raise TypeError(
            f"{entry_point}: this solver was compiled with channels=; re-compile "
            "with blocks= to use block-dependent Interactions."
        )


__all__ = [
    "add_block_axis",
    "assemble_block_hamiltonian",
    "assemble_hamiltonian_arrays",
    "block_group_arrays",
    "build_Q",
    "channel_arrays",
    "lift_to_blocks",
    "reject_block_dependent",
    "take_block0",
    "uniform_block_mass_factor",
    "uniform_mass_factor",
]

"""Per-energy direct R-matrix solver.

There is one direct-kernel family: it is always batched over the leading
symmetry-block axis (DESIGN.md §15.5).  A solver compiled with ``channels=``
is simply the ``N_b == 1`` case — the kernels run the same batched code and
squeeze the block axis off the outputs so the single-block public contract is
unchanged.  The only mode-specific kernel is the subinterval-propagated one,
which is defined for single-block local problems only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import jax
import jax.core
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import numpy as np

from lax.spectral.matching import open_channel_smatrix_from_R, phases_from_S
from lax.types import (
    BoundaryValues,
    ChannelSpec,
    DirectRMatrixKernel,
    Interaction,
    Mesh,
    OperatorMatrices,
    PropagationMatrices,
)

from .assembly import (
    assemble_hamiltonian_arrays,
    block_group_arrays,
    build_Q,
    lift_to_blocks,
    reject_block_dependent,
    uniform_mass_factor,
)


def _build_q_prime_blocks(
    q: jax.Array,
    mass_rows: jax.Array,
    basis_size: int,
) -> jax.Array:
    """Build the per-block surface projectors ``Q'_b``, shape ``(N_b, N_c·N, N_c)``.

    ``Q' = diag(repeat(√m_c, N)) · Q`` per block: each block's channel mass
    factors ``mass_rows[b]`` (shape ``(N_b, N_c)``) scale the shared ``Q`` so
    that ``R = Q'^T C_MeV^{-1} Q' / a`` equals the fm⁻² result for uniform μ
    and generalises to per-channel μ.  [DESIGN.md §11.5]
    """

    scale = jnp.repeat(jnp.sqrt(mass_rows), basis_size, axis=1)  # (N_b, N_c·N)
    q_prime_blocks: jax.Array = scale[:, :, None].astype(q.dtype) * q[None]
    return q_prime_blocks


@dataclass(frozen=True)
class _DirectRMatrixKernel:
    """Pickle-safe direct R-matrix kernel, batched over the symmetry-block axis.

    ``block_mode`` distinguishes the two compile modes: ``True`` for
    ``blocks=`` (outputs keep the leading ``(N_b,)`` axis and block-dependent
    Interactions are accepted), ``False`` for ``channels=`` (the ``N_b == 1``
    special case — the block axis is squeezed off the output).
    """

    mesh: Mesh
    operators: OperatorMatrices
    energies: jax.Array
    centrifugal: jax.Array  # (N_b, N_c)
    thresholds: jax.Array  # (N_b, N_c)
    mass_rows: jax.Array  # (N_b, N_c) — per-block channel μ on the fast path
    q: jax.Array  # (N_c·N, N_c), shared across blocks
    q_prime_blocks: jax.Array  # (N_b, N_c·N, N_c)
    mass_factor_grid_blocks: jax.Array  # (N_b, N_E, N_c)
    channel_radius: float
    matrix_size: int
    n_blocks: int
    energy_uniform_mu: bool
    block_mode: bool

    def __call__(self, potential: jax.Array | Interaction) -> jax.Array:
        """Evaluate the direct R-matrix on the compile-time energy grid.

        Returns shape ``(N_E, N_c, N_c)``, with a leading ``(N_b,)`` axis in
        blocks mode.  Block-independent Interactions broadcast across the
        block axis.

        Notes
        -----
        ``mass_factor_grid`` is honoured on every path.  When μ is uniform
        across the energy grid (the common case) the fast path is used with a
        single Hamiltonian assembly per block and the per-channel-μ surface
        projector ``Q'``.  When μ varies with energy, an energy-independent
        block is broadcast over the grid and solved per energy with the mass
        grid in both ``C(E_i)`` and ``Q'`` — matching §11.3.
        """

        if not isinstance(potential, Interaction):
            raise TypeError(
                "rmatrix_direct() accepts only Interaction objects. "
                "Use solver.local_potential()/solver.nonlocal_potential() or solver.interaction_from_block/array/funcs to build one."
            )
        if not self.block_mode:
            reject_block_dependent(potential, "rmatrix_direct()")
        block = lift_to_blocks(potential.block, potential.block_dependent, self.n_blocks)
        if potential.energy_dependent:
            result = cast(
                jax.Array,
                _RMATRIX_DIRECT_GRID_BLOCKS_JIT(
                    block,
                    self.mesh,
                    self.operators,
                    self.centrifugal,
                    self.thresholds,
                    self.energies,
                    self.q,
                    self.channel_radius,
                    self.matrix_size,
                    self.mass_factor_grid_blocks,
                ),
            )
        elif self.energy_uniform_mu:
            # μ is constant across energies: one assembly per block, per-channel-μ Q'.
            result = cast(
                jax.Array,
                _RMATRIX_DIRECT_BLOCKS_JIT(
                    block,
                    self.mesh,
                    self.operators,
                    self.centrifugal,
                    self.thresholds,
                    self.mass_rows,
                    self.energies,
                    self.q_prime_blocks,
                    self.channel_radius,
                    self.matrix_size,
                ),
            )
        else:
            # Energy-independent block but μ(E): broadcast over the grid per
            # block so each C(E_i)/Q' carries the correct mass-grid row.
            n_e = self.energies.shape[0]
            block_grid = jnp.broadcast_to(block[:, None], (self.n_blocks, n_e, *block.shape[1:]))
            result = cast(
                jax.Array,
                _RMATRIX_DIRECT_GRID_BLOCKS_JIT(
                    block_grid,
                    self.mesh,
                    self.operators,
                    self.centrifugal,
                    self.thresholds,
                    self.energies,
                    self.q,
                    self.channel_radius,
                    self.matrix_size,
                    self.mass_factor_grid_blocks,
                ),
            )
        return result if self.block_mode else result[0]


@dataclass(frozen=True)
class _PropagatedDirectRMatrixKernel:
    """Pickle-safe direct R-matrix kernel for subinterval-propagated meshes.

    Selected at compile time by :func:`make_rmatrix_direct_kernel` when
    ``mesh.propagation`` is set, so the batched kernel carries no propagation
    special cases.  Single-block (``channels=``) only — compile() rejects
    ``blocks=`` on propagated meshes.  Supports local energy-independent
    Interactions only: the per-interval ``(N_c, N_c, N)`` array is extracted
    from the block diagonals.
    """

    mesh: Mesh
    channels: tuple[ChannelSpec, ...]
    energies: jax.Array
    mass_factor: float | jax.Array
    boundary: BoundaryValues | None

    def __call__(self, potential: jax.Array | Interaction) -> jax.Array:
        """Evaluate the propagated direct R-matrix on the compile-time energy grid."""

        if not isinstance(potential, Interaction):
            raise TypeError(
                "rmatrix_direct() accepts only Interaction objects. "
                "Use solver.local_potential(fn)/solver.nonlocal_potential(fn) or solver.interaction_from_block(block)."
            )
        reject_block_dependent(potential, "rmatrix_direct()")
        if potential.energy_dependent:
            raise TypeError(
                "rmatrix_direct() does not support energy-dependent Interactions "
                "on propagated meshes."
            )
        n_c = len(self.channels)
        n = self.mesh.n
        # Propagated path supports only local Interactions: sub-blocks must be
        # diagonal.  Validate only outside jax transformations — inside vmap/jit
        # the block is a tracer and host inspection would crash; the diagonality
        # contract is a property of the user's input, checked once when concrete.
        if not isinstance(potential.block, jax.core.Tracer):
            host_block = np.asarray(potential.block)  # one device→host transfer
            for c in range(n_c):
                for cp in range(n_c):
                    sub = host_block[c * n : (c + 1) * n, cp * n : (cp + 1) * n]
                    if not np.allclose(sub, np.diag(np.diag(sub))):
                        raise ValueError(
                            "rmatrix_direct() on propagated meshes supports only local "
                            "Interactions. Non-local propagated direct solves are not supported."
                        )
        local_potential = jnp.stack(
            [
                jnp.stack(
                    [
                        jnp.diag(potential.block[c * n : (c + 1) * n, cp * n : (cp + 1) * n])
                        for cp in range(n_c)
                    ]
                )
                for c in range(n_c)
            ]
        )
        return cast(
            jax.Array,
            _RMATRIX_DIRECT_PROPAGATED_JIT(
                local_potential,
                self.mesh,
                self.channels,
                self.energies,
                self.mass_factor,
                self.boundary,
            ),
        )


@dataclass(frozen=True)
class _SMatrixDirectObservable:
    """Pickle-safe direct S-matrix observable derived from rmatrix_direct.

    ``boundary`` is mode-appropriate: ``(N_E, N_c)`` fields for a
    ``channels=`` solver (including propagated meshes), stacked
    ``(N_b, N_E, N_c)`` fields for a ``blocks=`` solver.
    """

    rmatrix_direct: DirectRMatrixKernel
    boundary: BoundaryValues
    block_mode: bool

    def __call__(self, potential: jax.Array | Interaction) -> jax.Array:
        """Evaluate the S-matrix on the compile-time energy grid."""

        r = self.rmatrix_direct(potential)
        jit_fn = _DIRECT_SMATRIX_BLOCKS_JIT if self.block_mode else _DIRECT_SMATRIX_JIT
        return cast(jax.Array, jit_fn(r, self.boundary))


@dataclass(frozen=True)
class _PhasesDirectObservable:
    """Pickle-safe direct phase-shift observable derived from smatrix_direct."""

    smatrix_direct: _SMatrixDirectObservable

    def __call__(self, potential: jax.Array | Interaction) -> jax.Array:
        """Evaluate phase shifts on the compile-time energy grid."""

        s = self.smatrix_direct(potential)
        jit_fn = _DIRECT_PHASES_BLOCKS_JIT if self.smatrix_direct.block_mode else _DIRECT_PHASES_JIT
        return cast(jax.Array, jit_fn(s))


@dataclass(frozen=True)
class _WavefunctionDirectKernel:
    """Pickle-safe wavefunction kernel on the direct (linear-solve) path,
    batched over the symmetry-block axis (``N_b == 1`` for ``channels=``)."""

    mesh: Mesh
    operators: OperatorMatrices
    energies: jax.Array
    centrifugal: jax.Array  # (N_b, N_c)
    thresholds: jax.Array  # (N_b, N_c)
    mass_factor_grid_blocks: jax.Array  # (N_b, N_E, N_c)
    matrix_size: int
    n_blocks: int
    block_mode: bool

    def __call__(
        self,
        potential: jax.Array | Interaction,
        source: jax.Array,
        energy_index: int,
    ) -> jax.Array:
        """Solve ``C(E_i) ψ = source`` for the internal wavefunction per block.

        Parameters
        ----------
        potential
            An :class:`~lax.Interaction`; block- and/or energy-dependent
            interactions are sliced/broadcast automatically.
        source
            Mesh-space driving term, shape ``(N_c·N,)``.  In blocks mode also
            ``(N_b, N_c·N)`` for per-block sources (the output of
            :func:`lax.make_wavefunction_source` on a blocks-compiled solver).
        energy_index
            Index into the compile-time energy grid (compile-time constant).

        Returns
        -------
        jax.Array
            Wavefunction coefficient vector, shape ``(N_c·N,)`` —
            ``(N_b, N_c·N)`` in blocks mode.
        """

        if not isinstance(potential, Interaction):
            raise TypeError(
                "wavefunction_direct() accepts only Interaction objects. "
                "Use solver.local_potential()/solver.nonlocal_potential() or solver.interaction_from_block/array/funcs to build one."
            )
        if not self.block_mode:
            reject_block_dependent(potential, "wavefunction_direct()")
        block = potential.block
        if potential.energy_dependent:
            block = block[:, energy_index] if potential.block_dependent else block[energy_index]
        block = lift_to_blocks(block, potential.block_dependent, self.n_blocks)

        if source.shape == (self.matrix_size,):
            sources = jnp.broadcast_to(source, (self.n_blocks, *source.shape))
        elif self.block_mode and source.shape == (self.n_blocks, self.matrix_size):
            sources = source
        else:
            expected = (
                f"({self.matrix_size},) or ({self.n_blocks}, {self.matrix_size})"
                if self.block_mode
                else f"({self.matrix_size},)"
            )
            raise ValueError(f"source must have shape {expected}; got {source.shape}.")

        result = cast(
            jax.Array,
            _WAVEFUNCTION_DIRECT_BLOCKS_JIT(
                block,
                sources,
                self.energies[energy_index],
                self.mass_factor_grid_blocks[:, energy_index],
                self.mesh,
                self.operators,
                self.centrifugal,
                self.thresholds,
                self.matrix_size,
            ),
        )
        return result if self.block_mode else result[0]


def _block_mass_data(
    blocks: tuple[tuple[ChannelSpec, ...], ...],
    channel_mass_rows: jax.Array,
    energies: jax.Array,
    mass_factor_grid: jax.Array | None,
) -> tuple[jax.Array, jax.Array, bool]:
    """Resolve the per-block mass data for the direct kernels.

    Returns ``(mass_rows, mass_factor_grid_blocks, energy_uniform_mu)``:
    the ``(N_b, N_c)`` per-block channel μ used on the single-assembly fast
    path, the dense ``(N_b, N_E, N_c)`` grid used on the per-energy path, and
    whether μ is constant across the energy grid.  When ``mass_factor_grid``
    is given it overrides every block's ``ChannelSpec.mass_factor`` (it is
    shared across blocks, like the energy grid).
    """

    n_b = len(blocks)
    n_e = len(energies)
    n_c = len(blocks[0])
    if mass_factor_grid is None:
        mass_rows = channel_mass_rows
        mfg_blocks = jnp.broadcast_to(mass_rows[:, None, :], (n_b, n_e, n_c))
        return mass_rows, mfg_blocks, True
    mfg = jnp.asarray(mass_factor_grid)
    if mfg.shape != (n_e, n_c):
        msg = (
            f"mass_factor_grid must be pre-broadcast to (N_E, N_c)=({n_e}, {n_c}); "
            f"got {mfg.shape}.  Build the solver via lax.compile()."
        )
        raise ValueError(msg)
    mfg_np = np.asarray(mfg)
    energy_uniform_mu = bool(np.allclose(mfg_np, mfg_np[0:1, :]))
    mass_rows = jnp.broadcast_to(mfg[0], (n_b, n_c)) if energy_uniform_mu else channel_mass_rows
    mfg_blocks = jnp.broadcast_to(mfg[None], (n_b, n_e, n_c))
    return mass_rows, mfg_blocks, energy_uniform_mu


def make_rmatrix_direct_kernel(
    mesh: Mesh,
    operators: OperatorMatrices,
    blocks: tuple[tuple[ChannelSpec, ...], ...],
    energies: jax.Array,
    boundary: BoundaryValues | None,
    mass_factor_grid: jax.Array | None = None,
    block_mode: bool = False,
) -> DirectRMatrixKernel:
    """Build a JIT-compiled ``rmatrix_direct(V) → R`` kernel for the compile-time grid.

    The returned kernel solves ``C(E) X = Q'`` per symmetry block for each
    compile-time energy via LU factorization, bypassing the eigendecomposition.
    Supports real and complex potentials, local and non-local, propagated and
    non-propagated meshes.  [DESIGN.md §11.3, §15.5]

    Parameters
    ----------
    mesh
        Compiled mesh; if ``mesh.propagation`` is not ``None``, a subinterval
        propagation recursion is used instead of a global solve (single-block
        only).
    operators
        Precomputed operator matrices (``TpL`` required).
    blocks
        ``N_b`` symmetry blocks of ``N_c`` channels each.  A ``channels=``
        compile passes the single block ``(channels,)``.
    energies
        Compile-time energy grid in MeV, shape ``(N_E,)``.
    boundary
        Compile-time boundary values, used only by the propagated path
        (mode-appropriate shape), or ``None``.
    mass_factor_grid
        Optional per-energy (and optionally per-channel) ℏ²/2μ values in
        MeV·fm², pre-broadcast to ``(N_E, N_c)``; shared across blocks.
    block_mode
        ``True`` for a ``blocks=`` compile: outputs keep the leading
        ``(N_b,)`` axis and block-dependent Interactions are accepted.

    Returns
    -------
    DirectRMatrixKernel
        JIT-compiled callable: ``kernel(V) → R`` with shape
        ``(N_E, N_c, N_c)`` (``(N_b, N_E, N_c, N_c)`` in blocks mode).
    """

    if mesh.propagation is not None:
        # The propagation recursion converts the potential to fm⁻² with a single
        # scalar mass factor, so it is only valid for a uniform-μ channel set.
        propagated_mass_factor = uniform_mass_factor(
            blocks[0], context="propagated direct R-matrix path"
        )
        return cast(
            DirectRMatrixKernel,
            _PropagatedDirectRMatrixKernel(
                mesh=mesh,
                channels=blocks[0],
                energies=energies,
                mass_factor=propagated_mass_factor,
                boundary=boundary,
            ),
        )

    centrifugal, thresholds, channel_mass_rows = block_group_arrays(blocks)
    mass_rows, mfg_blocks, energy_uniform_mu = _block_mass_data(
        blocks, channel_mass_rows, energies, mass_factor_grid
    )
    q = build_Q(mesh, blocks[0])
    return cast(
        DirectRMatrixKernel,
        _DirectRMatrixKernel(
            mesh=mesh,
            operators=operators,
            energies=energies,
            centrifugal=centrifugal,
            thresholds=thresholds,
            mass_rows=mass_rows,
            q=q,
            q_prime_blocks=_build_q_prime_blocks(q, mass_rows, mesh.n),
            mass_factor_grid_blocks=mfg_blocks,
            channel_radius=mesh.scale,
            matrix_size=mesh.n * len(blocks[0]),
            n_blocks=len(blocks),
            energy_uniform_mu=energy_uniform_mu,
            block_mode=block_mode,
        ),
    )


def make_smatrix_direct_observable(
    rmatrix_kernel: DirectRMatrixKernel,
    boundary: BoundaryValues | None,
    block_mode: bool = False,
) -> _SMatrixDirectObservable | None:
    """Build a direct S-matrix observable from a direct R-matrix kernel.

    ``boundary`` is mode-appropriate: ``(N_E, N_c)`` fields for ``channels=``
    solvers, stacked ``(N_b, N_E, N_c)`` fields for ``blocks=`` solvers.
    """

    if boundary is None:
        return None
    return _SMatrixDirectObservable(
        rmatrix_direct=rmatrix_kernel, boundary=boundary, block_mode=block_mode
    )


def make_phases_direct_observable(
    smatrix_observable: _SMatrixDirectObservable | None,
) -> _PhasesDirectObservable | None:
    """Build a direct phase-shift observable from a direct S-matrix observable."""

    if smatrix_observable is None:
        return None
    return _PhasesDirectObservable(smatrix_direct=smatrix_observable)


def make_direct_wavefunction_kernel(
    mesh: Mesh,
    operators: OperatorMatrices,
    blocks: tuple[tuple[ChannelSpec, ...], ...],
    energies: jax.Array,
    mass_factor_grid: jax.Array | None = None,
    block_mode: bool = False,
) -> _WavefunctionDirectKernel:
    """Build a direct wavefunction kernel ``(V, source, i) → ψ``.

    Solves ``C(E_i) ψ = source`` per symmetry block, where ``C = H_MeV − E_i·I``
    is assembled with the per-energy per-channel mass factor and the solve is
    scaled per channel by μ to match the fm⁻² spectral Green's convention.
    [DESIGN.md §11.3, §15.5]
    """

    centrifugal, thresholds, channel_mass_rows = block_group_arrays(blocks)
    _, mfg_blocks, _ = _block_mass_data(blocks, channel_mass_rows, energies, mass_factor_grid)
    return _WavefunctionDirectKernel(
        mesh=mesh,
        operators=operators,
        energies=energies,
        centrifugal=centrifugal,
        thresholds=thresholds,
        mass_factor_grid_blocks=mfg_blocks,
        matrix_size=mesh.n * len(blocks[0]),
        n_blocks=len(blocks),
        block_mode=block_mode,
    )


def _rmatrix_direct_core(
    potential: jax.Array,
    mesh: Mesh,
    operators: OperatorMatrices,
    centrifugal: jax.Array,
    thresholds: jax.Array,
    mass_factors: jax.Array,
    energies: jax.Array,
    q_prime: jax.Array,
    channel_radius: float,
    matrix_size: int,
) -> jax.Array:
    """Return the direct R-matrix across the compile-time energy grid.

    Array-parameterized per-block core (DESIGN.md §15.5): the per-channel
    centrifugal/threshold/mass data arrive as traced ``(N_c,)`` arrays so the
    body composes with ``jax.vmap`` over a leading symmetry-block axis.

    The Hamiltonian is assembled in MeV (symmetric form), and the C matrix
    is ``H_MeV − E·I``.  The surface projector ``Q'`` carries the per-channel
    sqrt(m_c) factor so that ``R = Q'^T C^{-1} Q' / a`` equals the fm⁻² result
    for uniform μ and generalises to per-channel μ.  [DESIGN.md §11.5]

    Parameters
    ----------
    potential
        Assembled potential in MeV.  Local: ``(N_c, N_c, N)``; non-local:
        ``(N_c, N_c, N, N)``; pre-assembled block: ``(M, M)``.
    mesh, operators, energies, q_prime, channel_radius, matrix_size
        Compile-time cached data forwarded from the kernel dataclass.
    centrifugal, thresholds, mass_factors
        Per-channel ``(N_c,)`` arrays (see :func:`~lax.solvers.assembly.channel_arrays`).

    Returns
    -------
    jax.Array
        R-matrix on the compile-time energy grid, shape ``(N_E, N_c, N_c)``.
    """

    hamiltonian = assemble_hamiltonian_arrays(
        mesh,
        operators,
        centrifugal,
        thresholds,
        mass_factors,
        potential,
    )

    def one_energy(energy: jax.Array) -> jax.Array:
        # Hamiltonian is in MeV; C = H_MeV − E·I.  Factor C once (§11.3: one
        # factorization of C(E_i) serves the surface solve here and the
        # wavefunction solve in _wavefunction_direct_core) and reuse it for
        # every column of Q'.
        matrix = hamiltonian - energy * jnp.eye(
            matrix_size,
            dtype=hamiltonian.dtype,
        )
        lu_piv = cast(tuple[jax.Array, jax.Array], jsl.lu_factor(matrix))
        solved = cast(jax.Array, jsl.lu_solve(lu_piv, q_prime))
        values: jax.Array = (q_prime.T @ solved) / channel_radius
        return values

    result: jax.Array = jax.vmap(one_energy)(energies)
    return result


def _rmatrix_direct_propagated(
    potential: jax.Array,
    mesh: Mesh,
    channels: tuple[ChannelSpec, ...],
    energies: jax.Array,
    mass_factor: float | jax.Array,
    boundary: BoundaryValues | None,
) -> jax.Array:
    """Return the propagated direct R-matrix across the compile-time energy grid.

    Parameters
    ----------
    potential
        Local potential in MeV, shape ``(N_c, N_c, N)``.
    mesh, channels, energies, mass_factor, boundary
        Compile-time cached data forwarded from the propagated kernel dataclass.
        ``mass_factor`` converts the potential to fm⁻² inside the recursion.

    Returns
    -------
    jax.Array
        R-matrix on the compile-time energy grid, shape ``(N_E, N_c, N_c)``.
    """

    if boundary is None:
        msg = "Boundary values are required for propagated direct R-matrix solves."
        raise ValueError(msg)
    propagation = mesh.propagation
    if propagation is None:
        msg = "Propagated direct solves require a mesh with propagation data."
        raise ValueError(msg)

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


def _rmatrix_direct_grid_core(
    potentials: jax.Array,
    mesh: Mesh,
    operators: OperatorMatrices,
    centrifugal: jax.Array,
    thresholds: jax.Array,
    energies: jax.Array,
    q: jax.Array,
    channel_radius: float,
    matrix_size: int,
    mass_factor_grid: jax.Array,
) -> jax.Array:
    """Return aligned-grid ``R(E_i; V_i)`` samples for energy-dependent potentials.

    Array-parameterized per-block core (DESIGN.md §15.5).  Evaluates each
    ``(V_i, E_i)`` pair independently — the diagonal of the ``(N_E, N_E)``
    Cartesian product — so the result is physically correct for potentials
    that depend on energy.

    Parameters
    ----------
    potentials
        Per-energy potentials in MeV.  Local: ``(N_E, N_c, N_c, N)``; non-local:
        ``(N_E, N_c, N_c, N, N)``; pre-assembled blocks: ``(N_E, M, M)``.
    mesh, operators, energies, q, channel_radius, matrix_size
        Compile-time cached data.  ``q`` is the unscaled surface projector.
    centrifugal, thresholds
        Per-channel ``(N_c,)`` arrays (see :func:`~lax.solvers.assembly.channel_arrays`).
    mass_factor_grid
        Dense ``(N_E, N_c)`` mass-factor array, always present (promoted at
        compile time from scalar / ``(N_E,)`` / per-channel inputs).

    Returns
    -------
    jax.Array
        R-matrix samples, shape ``(N_E, N_c, N_c)``.
    """

    def one_energy(
        potential: jax.Array,
        energy: jax.Array,
        mu_row: jax.Array,  # (N_c,) per-channel mass factors, vmapped from mass_factor_grid
    ) -> jax.Array:
        hamiltonian = assemble_hamiltonian_arrays(
            mesh, operators, centrifugal, thresholds, mu_row, potential
        )
        # Hamiltonian in MeV; C = H_MeV − E·I.
        # Q' = diag(repeat(sqrt(mu_row), N))·Q — per-channel reduced-width scaling.
        matrix = hamiltonian - energy * jnp.eye(matrix_size, dtype=hamiltonian.dtype)
        n = mesh.n
        scale = jnp.repeat(jnp.sqrt(mu_row), n).astype(q.dtype)  # (N_c·N,)
        q_prime_mu: jax.Array = scale[:, None] * q
        lu_piv = cast(tuple[jax.Array, jax.Array], jsl.lu_factor(matrix))
        solved = cast(jax.Array, jsl.lu_solve(lu_piv, q_prime_mu))
        return (q_prime_mu.T @ solved) / channel_radius

    # mass_factor_grid is (N_E, N_c); vmap slices to (N_c,) per energy step.
    return jax.vmap(one_energy)(potentials, energies, mass_factor_grid)


def _wavefunction_direct_core(
    potential: jax.Array,
    source: jax.Array,
    energy: jax.Array,
    mu_row: jax.Array,
    mesh: Mesh,
    operators: OperatorMatrices,
    centrifugal: jax.Array,
    thresholds: jax.Array,
    matrix_size: int,
) -> jax.Array:
    """Solve the internal wavefunction on the MeV direct path.

    Array-parameterized per-block core (DESIGN.md §15.5).  Assembles
    ``C = H_MeV(μ) − E·I`` using the per-channel mass factors in ``mu_row``
    (shape ``(N_c,)``), solves ``C ψ̃ = source``, then scales channel block
    ``c`` by ``μ_c`` so the result equals the fm⁻² spectral Green's function
    ``G_spectral = μ · (H_MeV − E·I)⁻¹`` channel-by-channel.  For a uniform μ
    this reduces to the previous ``m₀ · solve`` behaviour.
    """

    hamiltonian = assemble_hamiltonian_arrays(
        mesh, operators, centrifugal, thresholds, mu_row, potential
    )
    matrix = hamiltonian - energy * jnp.eye(matrix_size, dtype=hamiltonian.dtype)
    # Same C(E_i) factorization formulation as the direct R-matrix solve (§11.3).
    lu_piv = cast(tuple[jax.Array, jax.Array], jsl.lu_factor(matrix))
    solved = cast(jax.Array, jsl.lu_solve(lu_piv, source))
    scale = jnp.repeat(mu_row, mesh.n)  # (N_c·N,) — per-channel μ on the diagonal
    result: jax.Array = scale.astype(solved.dtype) * solved
    return result


def _direct_smatrix_grid(
    r_grid: jax.Array,
    boundary: BoundaryValues,
) -> jax.Array:
    """Match an (N_E, N_c, N_c) R-matrix grid to the S-matrix grid.

    Uses :func:`open_channel_smatrix_from_R` so closed channels are decoupled
    and projected out exactly as on the spectral path — keeping
    ``smatrix_direct``/``phases_direct`` consistent with ``smatrix``/``phases``.
    For an all-open channel set this is identical to bare ``smatrix_from_R``.
    """

    return jax.vmap(open_channel_smatrix_from_R)(r_grid, boundary)


def _direct_phases_grid(s_grid: jax.Array) -> jax.Array:
    """Extract phase shifts from an (N_E, N_c, N_c) S-matrix grid."""

    return jax.vmap(phases_from_S)(s_grid)


# --------------------------------------------------------------------------
# Symmetry-block batched layers (DESIGN.md §15.5): thin jax.vmap wrappers over
# the array-parameterized per-block cores above.


def _rmatrix_direct_blocks(
    blocks: jax.Array,
    mesh: Mesh,
    operators: OperatorMatrices,
    centrifugal: jax.Array,
    thresholds: jax.Array,
    mass_rows: jax.Array,
    energies: jax.Array,
    q_prime_blocks: jax.Array,
    channel_radius: float,
    matrix_size: int,
) -> jax.Array:
    """Return the block-batched direct R-matrix, shape ``(N_b, N_E, N_c, N_c)``.

    ``jax.vmap`` of :func:`_rmatrix_direct_core` over the leading ``(N_b,)``
    axis of the per-block data (``blocks``, centrifugal/threshold/mass rows,
    and the per-block surface projector ``Q'``); the per-energy vmap lives
    inside the core.
    """

    def one_block(
        block: jax.Array,
        centrifugal_row: jax.Array,
        threshold_row: jax.Array,
        mass_row: jax.Array,
        q_prime: jax.Array,
    ) -> jax.Array:
        return _rmatrix_direct_core(
            block,
            mesh,
            operators,
            centrifugal_row,
            threshold_row,
            mass_row,
            energies,
            q_prime,
            channel_radius,
            matrix_size,
        )

    return jax.vmap(one_block)(blocks, centrifugal, thresholds, mass_rows, q_prime_blocks)


def _rmatrix_direct_grid_blocks(
    blocks: jax.Array,
    mesh: Mesh,
    operators: OperatorMatrices,
    centrifugal: jax.Array,
    thresholds: jax.Array,
    energies: jax.Array,
    q: jax.Array,
    channel_radius: float,
    matrix_size: int,
    mass_factor_grid_blocks: jax.Array,
) -> jax.Array:
    """Return block-batched aligned-grid R samples, shape ``(N_b, N_E, N_c, N_c)``.

    Outer ``jax.vmap`` of :func:`_rmatrix_direct_grid_core` over the block axis
    of ``blocks`` ``(N_b, N_E, M, M)``, the per-block channel rows, and the
    per-block ``(N_E, N_c)`` mass grid; ``energies`` and ``q`` are shared.
    """

    def one_block(
        block_grid: jax.Array,
        centrifugal_row: jax.Array,
        threshold_row: jax.Array,
        mass_factor_grid: jax.Array,
    ) -> jax.Array:
        return _rmatrix_direct_grid_core(
            block_grid,
            mesh,
            operators,
            centrifugal_row,
            threshold_row,
            energies,
            q,
            channel_radius,
            matrix_size,
            mass_factor_grid,
        )

    return jax.vmap(one_block)(blocks, centrifugal, thresholds, mass_factor_grid_blocks)


def _direct_smatrix_blocks(
    r_blocks: jax.Array,
    boundary: BoundaryValues,
) -> jax.Array:
    """Match an (N_b, N_E, N_c, N_c) R-matrix to the S-matrix per block.

    ``boundary`` carries the stacked ``(N_b, N_E, N_c)`` fields; being a
    registered pytree it vmaps jointly with the R-matrix block axis.
    """

    return jax.vmap(_direct_smatrix_grid)(r_blocks, boundary)


def _direct_phases_blocks(s_blocks: jax.Array) -> jax.Array:
    """Extract phase shifts from an (N_b, N_E, N_c, N_c) S-matrix."""

    return jax.vmap(_direct_phases_grid)(s_blocks)


def _wavefunction_direct_blocks(
    blocks: jax.Array,
    sources: jax.Array,
    energy: jax.Array,
    mu_rows: jax.Array,
    mesh: Mesh,
    operators: OperatorMatrices,
    centrifugal: jax.Array,
    thresholds: jax.Array,
    matrix_size: int,
) -> jax.Array:
    """Return block-batched internal wavefunctions, shape ``(N_b, N_c·N)``.

    ``jax.vmap`` of :func:`_wavefunction_direct_core` over the block axis of
    the assembled blocks, sources, mass rows, and channel rows at one energy.
    """

    def one_block(
        block: jax.Array,
        source: jax.Array,
        mu_row: jax.Array,
        centrifugal_row: jax.Array,
        threshold_row: jax.Array,
    ) -> jax.Array:
        return _wavefunction_direct_core(
            block,
            source,
            energy,
            mu_row,
            mesh,
            operators,
            centrifugal_row,
            threshold_row,
            matrix_size,
        )

    return jax.vmap(one_block)(blocks, sources, mu_rows, centrifugal, thresholds)


_RMATRIX_DIRECT_PROPAGATED_JIT = jax.jit(
    _rmatrix_direct_propagated,
    static_argnames=("channels",),
)
_DIRECT_SMATRIX_JIT = jax.jit(_direct_smatrix_grid)
_DIRECT_PHASES_JIT = jax.jit(_direct_phases_grid)
_RMATRIX_DIRECT_BLOCKS_JIT = jax.jit(
    _rmatrix_direct_blocks,
    static_argnames=("matrix_size",),
)
_RMATRIX_DIRECT_GRID_BLOCKS_JIT = jax.jit(
    _rmatrix_direct_grid_blocks,
    static_argnames=("matrix_size",),
)
_WAVEFUNCTION_DIRECT_BLOCKS_JIT = jax.jit(
    _wavefunction_direct_blocks,
    static_argnames=("matrix_size",),
)
_DIRECT_SMATRIX_BLOCKS_JIT = jax.jit(_direct_smatrix_blocks)
_DIRECT_PHASES_BLOCKS_JIT = jax.jit(_direct_phases_blocks)


def _propagated_rmatrix_at_energy(
    potential: jax.Array,
    propagation: PropagationMatrices,
    channels: tuple[ChannelSpec, ...],
    energy: jax.Array,
    h_plus: jax.Array,
    h_plus_p: jax.Array,
    is_open: jax.Array,
    mass_factor: float | jax.Array,
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


__all__ = [
    "make_direct_wavefunction_kernel",
    "make_phases_direct_observable",
    "make_rmatrix_direct_kernel",
    "make_smatrix_direct_observable",
]

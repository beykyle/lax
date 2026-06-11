"""Per-energy direct R-matrix solver."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

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

from .assembly import assemble_block_hamiltonian, build_Q, uniform_mass_factor

if TYPE_CHECKING:
    pass


def _build_q_prime(
    q: jax.Array,
    channels: tuple[ChannelSpec, ...],
    basis_size: int,
) -> jax.Array:
    """Build Q' = diag(repeat(sqrt(m_c), N)) @ Q.

    Q has shape (N_c·N, N_c).  Each channel block c is scaled by sqrt(m_c)
    so that R = Q'^T C_MeV^{-1} Q' / a equals the old fm⁻² result for
    uniform μ and generalises to per-channel μ.  [DESIGN.md §11.5]
    """
    m_c = np.asarray([c.mass_factor for c in channels], dtype=np.float64)
    scale = np.repeat(np.sqrt(m_c), basis_size)  # (N_c·N,) NumPy array
    q_prime: jax.Array = jnp.asarray(scale[:, None], dtype=q.dtype) * q
    return q_prime


@dataclass(frozen=True)
class _DirectRMatrixKernel:
    """Pickle-safe direct R-matrix kernel backed by a module-level JIT function."""

    mesh: Mesh
    operators: OperatorMatrices
    channels: tuple[ChannelSpec, ...]
    static_channels: tuple[ChannelSpec, ...]
    energies: jax.Array
    q: jax.Array
    q_prime: jax.Array
    channel_radius: float
    matrix_size: int
    mass_factor_grid: jax.Array
    energy_uniform_mu: bool

    def __call__(self, potential: jax.Array | Interaction) -> jax.Array:
        """Evaluate the direct R-matrix on the compile-time energy grid.

        Parameters
        ----------
        potential
            :class:`~lax.Interaction` object built by ``solver.local_potential()``/``solver.nonlocal_potential()`` or
            ``solver.interaction_from_{block,array,funcs}()``.  Energy-dependent
            interactions (``energy_dependent=True``) use the per-energy block path.

        Notes
        -----
        ``mass_factor_grid`` is honoured on every path.  When μ is uniform across
        the energy grid (the common case) the fast static path is used with a
        single Hamiltonian assembly and the per-channel-μ surface projector
        ``Q'``.  When μ varies with energy, an energy-independent block is
        broadcast over the grid and solved per energy with ``mass_factor_grid``
        in both ``C(E_i)`` and ``Q'`` — matching §11.3.
        """

        if not isinstance(potential, Interaction):
            raise TypeError(
                "rmatrix_direct() accepts only Interaction objects. "
                "Use solver.local_potential()/solver.nonlocal_potential() or solver.interaction_from_block/array/funcs to build one."
            )

        if potential.energy_dependent:
            return cast(
                jax.Array,
                _RMATRIX_DIRECT_GRID_JIT(
                    potential.block,
                    self.mesh,
                    self.operators,
                    self.channels,
                    self.energies,
                    self.q,
                    self.channel_radius,
                    self.matrix_size,
                    self.mass_factor_grid,
                ),
            )
        if self.energy_uniform_mu:
            # μ is constant across energies: one assembly, per-channel-μ Q'.
            return cast(
                jax.Array,
                _RMATRIX_DIRECT_JIT(
                    potential.block,
                    self.mesh,
                    self.operators,
                    self.static_channels,
                    self.energies,
                    self.q_prime,
                    self.channel_radius,
                    self.matrix_size,
                ),
            )
        # Energy-independent block but energy-dependent μ(E): broadcast the block
        # over the grid and reuse the per-energy grid solver so each C(E_i)/Q'
        # carries the correct mass_factor_grid row.
        n_e = self.energies.shape[0]
        block_grid = jnp.broadcast_to(potential.block, (n_e, *potential.block.shape))
        return cast(
            jax.Array,
            _RMATRIX_DIRECT_GRID_JIT(
                block_grid,
                self.mesh,
                self.operators,
                self.channels,
                self.energies,
                self.q,
                self.channel_radius,
                self.matrix_size,
                self.mass_factor_grid,
            ),
        )


@dataclass(frozen=True)
class _PropagatedDirectRMatrixKernel:
    """Pickle-safe direct R-matrix kernel for subinterval-propagated meshes.

    Selected at compile time by :func:`make_rmatrix_direct_kernel` when
    ``mesh.propagation`` is set, so the non-propagated kernel carries no
    propagation special cases.  Supports local energy-independent
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
    """Pickle-safe direct S-matrix observable derived from rmatrix_direct."""

    rmatrix_direct: DirectRMatrixKernel
    boundary: BoundaryValues

    def __call__(self, potential: jax.Array | Interaction) -> jax.Array:
        """Evaluate the S-matrix on the compile-time energy grid."""

        r = self.rmatrix_direct(potential)
        return cast(jax.Array, _DIRECT_SMATRIX_JIT(r, self.boundary))


@dataclass(frozen=True)
class _PhasesDirectObservable:
    """Pickle-safe direct phase-shift observable derived from smatrix_direct."""

    smatrix_direct: _SMatrixDirectObservable

    def __call__(self, potential: jax.Array | Interaction) -> jax.Array:
        """Evaluate phase shifts on the compile-time energy grid."""

        s = self.smatrix_direct(potential)
        return cast(jax.Array, _DIRECT_PHASES_JIT(s))


@dataclass(frozen=True)
class _WavefunctionDirectKernel:
    """Pickle-safe wavefunction kernel on the direct (linear-solve) path."""

    mesh: Mesh
    operators: OperatorMatrices
    channels: tuple[ChannelSpec, ...]
    energies: jax.Array
    matrix_size: int
    mass_factor_grid: jax.Array

    def __call__(
        self,
        potential: jax.Array | Interaction,
        source: jax.Array,
        energy_index: int,
    ) -> jax.Array:
        """Solve ``C(E_i) x = source`` for the internal wavefunction.

        Parameters
        ----------
        potential
            :class:`~lax.Interaction` object.  For energy-dependent interactions
            the block at ``energy_index`` is extracted automatically.
        source
            Mesh-space driving term, shape ``(N_c·N,)``.
        energy_index
            Index into the compile-time energy grid (compile-time constant).

        Returns
        -------
        jax.Array
            Wavefunction coefficient vector, shape ``(N_c·N,)``.
        """
        if not isinstance(potential, Interaction):
            raise TypeError(
                "wavefunction_direct() accepts only Interaction objects. "
                "Use solver.local_potential()/solver.nonlocal_potential() or solver.interaction_from_block/array/funcs to build one."
            )
        block = potential.block[energy_index] if potential.energy_dependent else potential.block

        return cast(
            jax.Array,
            _WAVEFUNCTION_DIRECT_JIT(
                block,
                source,
                self.energies[energy_index],
                self.mass_factor_grid[energy_index],
                self.mesh,
                self.operators,
                self.channels,
                self.matrix_size,
            ),
        )


def make_rmatrix_direct_kernel(
    mesh: Mesh,
    operators: OperatorMatrices,
    channels: tuple[ChannelSpec, ...],
    energies: jax.Array,
    boundary: BoundaryValues | None,
    mass_factor_grid: jax.Array | None = None,
) -> DirectRMatrixKernel:
    """Build a JIT-compiled ``rmatrix_direct(V) → R`` kernel for the compile-time grid.

    The returned kernel solves ``C(E) X = Q'`` for each compile-time energy via
    ``jnp.linalg.solve``, bypassing the eigendecomposition.  Supports real and
    complex potentials, local and non-local, propagated and non-propagated meshes.
    Also accepts :class:`~lax.Interaction` objects directly.
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
    mass_factor_grid
        Optional per-energy (and optionally per-channel) ℏ²/2μ values in MeV·fm²,
        shape ``(N_E,)`` or ``(N_E, N_c)``.  Applied in the energy-dependent
        Interaction path only (``energy_dependent=True``).

    Returns
    -------
    DirectRMatrixKernel
        JIT-compiled callable: ``kernel(V) → R`` with shape ``(N_E, N_c, N_c)``.
    """

    if mesh.propagation is not None:
        # The propagation recursion converts the potential to fm⁻² with a single
        # scalar mass factor, so it is only valid for a uniform-μ channel set.
        propagated_mass_factor = uniform_mass_factor(
            channels, context="propagated direct R-matrix path"
        )
        return cast(
            DirectRMatrixKernel,
            _PropagatedDirectRMatrixKernel(
                mesh=mesh,
                channels=channels,
                energies=energies,
                mass_factor=propagated_mass_factor,
                boundary=boundary,
            ),
        )

    q = build_Q(mesh, channels)
    n_e = len(energies)
    n_c = len(channels)
    # compile() pre-broadcasts every supported input to the canonical (N_E, N_c)
    # form via compile._broadcast_mass_factor_grid, so the only cases left here
    # are "no grid" (use each channel's static mass_factor) and a ready (N_E, N_c)
    # array.  Validate rather than re-implement the broadcast.
    if mass_factor_grid is None:
        _mfg: jax.Array = jnp.broadcast_to(
            jnp.array([c.mass_factor for c in channels], dtype=float), (n_e, n_c)
        )
    else:
        _mfg = jnp.asarray(mass_factor_grid)
        if _mfg.shape != (n_e, n_c):
            msg = (
                f"mass_factor_grid must be pre-broadcast to (N_E, N_c)=({n_e}, {n_c}); "
                f"got {_mfg.shape}.  Build the solver via lax.compile()."
            )
            raise ValueError(msg)
    # When μ is the same at every energy, the energy-independent path can assemble
    # the Hamiltonian once and use a single per-channel-μ Q'.  The "static" channels
    # carry that uniform μ so both H and Q' honour mass_factor_grid even when it
    # differs from each ChannelSpec.mass_factor.  When μ varies with energy the
    # static path is bypassed (see _DirectRMatrixKernel.__call__).
    mfg_np = np.asarray(_mfg)
    energy_uniform_mu = bool(np.allclose(mfg_np, mfg_np[0:1, :]))
    if energy_uniform_mu:
        uniform_mu = mfg_np[0]
        static_channels = tuple(
            ChannelSpec(l=ch.l, threshold=ch.threshold, mass_factor=float(uniform_mu[i]))
            for i, ch in enumerate(channels)
        )
    else:
        static_channels = channels
    q_prime = _build_q_prime(q, static_channels, mesh.n)
    return cast(
        DirectRMatrixKernel,
        _DirectRMatrixKernel(
            mesh=mesh,
            operators=operators,
            channels=channels,
            static_channels=static_channels,
            energies=energies,
            q=q,
            q_prime=q_prime,
            channel_radius=mesh.scale,
            matrix_size=mesh.n * len(channels),
            mass_factor_grid=_mfg,
            energy_uniform_mu=energy_uniform_mu,
        ),
    )


def make_smatrix_direct_observable(
    rmatrix_kernel: DirectRMatrixKernel,
    boundary: BoundaryValues | None,
) -> _SMatrixDirectObservable | None:
    """Build a direct S-matrix observable from a direct R-matrix kernel."""

    if boundary is None:
        return None
    return _SMatrixDirectObservable(rmatrix_direct=rmatrix_kernel, boundary=boundary)


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
    channels: tuple[ChannelSpec, ...],
    energies: jax.Array,
    mass_factor_grid: jax.Array | None = None,
) -> _WavefunctionDirectKernel:
    """Build a direct wavefunction kernel ``(V, source, i) → ψ``.

    Solves ``C(E_i) ψ = source`` where ``C = H_MeV − E_i · I`` is assembled with
    the per-energy per-channel mass factor ``mass_factor_grid[i]`` and the solve
    is scaled per channel by μ to match the fm⁻² spectral Green's convention.
    ``mass_factor_grid`` follows the canonical ``(N_E, N_c)`` form; when ``None``
    each channel's static ``mass_factor`` is used.  [DESIGN.md §11.3]
    """

    matrix_size = mesh.n * len(channels)
    n_e = len(energies)
    n_c = len(channels)
    if mass_factor_grid is None:
        _mfg: jax.Array = jnp.broadcast_to(
            jnp.array([c.mass_factor for c in channels], dtype=float), (n_e, n_c)
        )
    else:
        _mfg = jnp.asarray(mass_factor_grid)
        if _mfg.shape != (n_e, n_c):
            msg = (
                f"mass_factor_grid must be pre-broadcast to (N_E, N_c)=({n_e}, {n_c}); "
                f"got {_mfg.shape}.  Build the solver via lax.compile()."
            )
            raise ValueError(msg)
    return _WavefunctionDirectKernel(
        mesh=mesh,
        operators=operators,
        channels=channels,
        energies=energies,
        matrix_size=matrix_size,
        mass_factor_grid=_mfg,
    )


def _rmatrix_direct(
    potential: jax.Array,
    mesh: Mesh,
    operators: OperatorMatrices,
    channels: tuple[ChannelSpec, ...],
    energies: jax.Array,
    q_prime: jax.Array,
    channel_radius: float,
    matrix_size: int,
) -> jax.Array:
    """Return the direct R-matrix across the compile-time energy grid.

    The Hamiltonian is assembled in MeV (symmetric form), and the C matrix
    is ``H_MeV − E·I``.  The surface projector ``Q'`` carries the per-channel
    sqrt(m_c) factor so that ``R = Q'^T C^{-1} Q' / a`` equals the fm⁻² result
    for uniform μ and generalises to per-channel μ.  [DESIGN.md §11.5]

    Parameters
    ----------
    potential
        Assembled potential in MeV.  Local: ``(N_c, N_c, N)``; non-local:
        ``(N_c, N_c, N, N)``; pre-assembled block: ``(M, M)``.
    mesh, operators, channels, energies, q_prime, channel_radius, matrix_size
        Compile-time cached data forwarded from the kernel dataclass.

    Returns
    -------
    jax.Array
        R-matrix on the compile-time energy grid, shape ``(N_E, N_c, N_c)``.
    """

    hamiltonian = assemble_block_hamiltonian(
        mesh,
        operators,
        channels,
        potential,
    )

    def one_energy(energy: jax.Array) -> jax.Array:
        # Hamiltonian is in MeV; C = H_MeV − E·I.  Factor C once (§11.3: one
        # factorization of C(E_i) serves the surface solve here and the
        # wavefunction solve in _wavefunction_direct) and reuse it for every
        # column of Q'.
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


def _rmatrix_direct_grid(
    potentials: jax.Array,
    mesh: Mesh,
    operators: OperatorMatrices,
    channels: tuple[ChannelSpec, ...],
    energies: jax.Array,
    q: jax.Array,
    channel_radius: float,
    matrix_size: int,
    mass_factor_grid: jax.Array,
) -> jax.Array:
    """Return aligned-grid ``R(E_i; V_i)`` samples for energy-dependent potentials.

    Evaluates each ``(V_i, E_i)`` pair independently — the diagonal of the
    ``(N_E, N_E)`` Cartesian product — so the result is physically correct for
    potentials that depend on energy.

    Parameters
    ----------
    potentials
        Per-energy potentials in MeV.  Local: ``(N_E, N_c, N_c, N)``; non-local:
        ``(N_E, N_c, N_c, N, N)``; pre-assembled blocks: ``(N_E, M, M)``.
    mesh, operators, channels, energies, q, channel_radius, matrix_size
        Compile-time cached data.  ``q`` is the unscaled surface projector.
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
        updated = tuple(
            ChannelSpec(l=ch.l, threshold=ch.threshold, mass_factor=mu_row[i])
            for i, ch in enumerate(channels)
        )
        hamiltonian = assemble_block_hamiltonian(mesh, operators, updated, potential)
        # Hamiltonian in MeV; C = H_MeV − E·I.
        # Q' = diag(repeat(sqrt(mu_row), N))·Q — per-channel reduced-width scaling.
        matrix = hamiltonian - energy * jnp.eye(matrix_size, dtype=hamiltonian.dtype)
        n = mesh.n
        scale = jnp.repeat(jnp.sqrt(mu_row), n)  # (N_c·N,)
        q_prime_mu: jax.Array = scale[:, None] * q
        lu_piv = cast(tuple[jax.Array, jax.Array], jsl.lu_factor(matrix))
        solved = cast(jax.Array, jsl.lu_solve(lu_piv, q_prime_mu))
        return (q_prime_mu.T @ solved) / channel_radius

    # mass_factor_grid is (N_E, N_c); vmap slices to (N_c,) per energy step.
    return jax.vmap(one_energy)(potentials, energies, mass_factor_grid)


def _wavefunction_direct(
    potential: jax.Array,
    source: jax.Array,
    energy: jax.Array,
    mu_row: jax.Array,
    mesh: Mesh,
    operators: OperatorMatrices,
    channels: tuple[ChannelSpec, ...],
    matrix_size: int,
) -> jax.Array:
    """Solve the internal wavefunction on the MeV direct path.

    Assembles ``C = H_MeV(μ) − E·I`` using the per-channel mass factors in
    ``mu_row`` (shape ``(N_c,)``), solves ``C ψ̃ = source``, then scales channel
    block ``c`` by ``μ_c`` so the result equals the fm⁻² spectral Green's
    function ``G_spectral = μ · (H_MeV − E·I)⁻¹`` channel-by-channel.  For a
    uniform μ this reduces to the previous ``m₀ · solve`` behaviour.
    """

    updated = tuple(
        ChannelSpec(l=ch.l, threshold=ch.threshold, mass_factor=mu_row[i])
        for i, ch in enumerate(channels)
    )
    hamiltonian = assemble_block_hamiltonian(mesh, operators, updated, potential)
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


_RMATRIX_DIRECT_JIT = jax.jit(
    _rmatrix_direct,
    static_argnames=("channels", "matrix_size"),
)
_RMATRIX_DIRECT_PROPAGATED_JIT = jax.jit(
    _rmatrix_direct_propagated,
    static_argnames=("channels",),
)
_RMATRIX_DIRECT_GRID_JIT = jax.jit(
    _rmatrix_direct_grid,
    static_argnames=("channels", "matrix_size"),
)
_WAVEFUNCTION_DIRECT_JIT = jax.jit(
    _wavefunction_direct,
    static_argnames=("channels", "matrix_size"),
)
_DIRECT_SMATRIX_JIT = jax.jit(_direct_smatrix_grid)
_DIRECT_PHASES_JIT = jax.jit(_direct_phases_grid)


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

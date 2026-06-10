"""Per-energy direct R-matrix solver."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

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
from lax.spectral.matching import phases_from_S, smatrix_from_R
from lax.types import ChannelSpec

from .assembly import assemble_block_hamiltonian, build_Q

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
    energies: jax.Array
    q: jax.Array
    q_prime: jax.Array
    channel_radius: float
    matrix_size: int
    mass_factor: float
    boundary: BoundaryValues | None
    mass_factor_grid: jax.Array | None = None

    def __call__(self, potential: jax.Array) -> jax.Array:
        """Evaluate the direct R-matrix on the compile-time energy grid.

        Parameters
        ----------
        potential
            :class:`~lax.Interaction` object built by ``solver.potential()`` or
            ``solver.interaction_from_{block,array,funcs}()``.  Energy-dependent
            interactions (``energy_dependent=True``) use the per-energy block path.

            :class:`~lax.Interaction` object built by ``solver.potential()`` or
            ``solver.interaction_from_{block,array,funcs}()``.  For propagated meshes,
            local energy-independent Interactions are supported: the per-interval
            ``(N_c, N_c, N)`` array is extracted from the block diagonals.
        """
        from lax.types import Interaction  # noqa: PLC0415

        # Propagated meshes use per-interval raw (N_c, N_c, N) arrays.
        # Extract from Interaction.block by taking the diagonal of each sub-block.
        if self.mesh.propagation is not None:
            if not isinstance(potential, Interaction):
                raise TypeError(
                    "rmatrix_direct() accepts only Interaction objects. "
                    "Use solver.potential(fn) or solver.interaction_from_block(block)."
                )
            if potential.energy_dependent:
                raise TypeError(
                    "rmatrix_direct() does not support energy-dependent Interactions "
                    "on propagated meshes."
                )
            N_c = len(self.channels)
            N = self.mesh.n
            # Propagated path supports only local Interactions: sub-blocks must be diagonal.
            for c in range(N_c):
                for cp in range(N_c):
                    sub = np.asarray(potential.block[c * N : (c + 1) * N, cp * N : (cp + 1) * N])
                    if np.any(sub != np.diag(np.diag(sub))):
                        raise ValueError(
                            "rmatrix_direct() on propagated meshes supports only local "
                            "Interactions. Non-local propagated direct solves are not supported."
                        )
            potential = jnp.stack(
                [
                    jnp.stack(
                        [
                            jnp.diag(potential.block[c * N : (c + 1) * N, cp * N : (cp + 1) * N])
                            for cp in range(N_c)
                        ]
                    )
                    for c in range(N_c)
                ]
            )
            return cast(
                jax.Array,
                _RMATRIX_DIRECT_JIT(
                    potential,
                    self.mesh,
                    self.operators,
                    self.channels,
                    self.energies,
                    self.q_prime,
                    self.channel_radius,
                    self.matrix_size,
                    self.mass_factor,
                    self.boundary,
                ),
            )

        if not isinstance(potential, Interaction):
            raise TypeError(
                "rmatrix_direct() accepts only Interaction objects. "
                "Use solver.potential() or solver.interaction_from_block/array/funcs to build one."
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
                    self.q_prime,
                    self.channel_radius,
                    self.matrix_size,
                    self.mass_factor,
                    self.boundary,
                    self.mass_factor_grid,
                ),
            )
        return cast(
            jax.Array,
            _RMATRIX_DIRECT_JIT(
                potential.block,
                self.mesh,
                self.operators,
                self.channels,
                self.energies,
                self.q_prime,
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
    q_prime: jax.Array
    channel_radius: float
    matrix_size: int
    mass_factor: float
    boundary: BoundaryValues | None
    mass_factor_grid: jax.Array | None = None

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
                self.q_prime,
                self.channel_radius,
                self.matrix_size,
                self.mass_factor,
                self.boundary,
                self.mass_factor_grid,
            ),
        )


@dataclass(frozen=True)
class _SMatrixDirectObservable:
    """Pickle-safe direct S-matrix observable derived from rmatrix_direct."""

    rmatrix_direct: _DirectRMatrixKernel
    boundary: BoundaryValues

    def __call__(self, potential: jax.Array) -> jax.Array:
        """Evaluate the S-matrix on the compile-time energy grid."""

        r = self.rmatrix_direct(potential)
        return cast(jax.Array, _DIRECT_SMATRIX_JIT(r, self.boundary))


@dataclass(frozen=True)
class _PhasesDirectObservable:
    """Pickle-safe direct phase-shift observable derived from smatrix_direct."""

    smatrix_direct: _SMatrixDirectObservable

    def __call__(self, potential: jax.Array) -> jax.Array:
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

    def __call__(
        self,
        potential: jax.Array,
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
        from lax.types import Interaction  # noqa: PLC0415

        if not isinstance(potential, Interaction):
            raise TypeError(
                "wavefunction_direct() accepts only Interaction objects. "
                "Use solver.potential() or solver.interaction_from_block/array/funcs to build one."
            )
        block = potential.block[energy_index] if potential.energy_dependent else potential.block

        return cast(
            jax.Array,
            _WAVEFUNCTION_DIRECT_JIT(
                block,
                source,
                self.energies[energy_index],
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

    q = build_Q(mesh, channels)
    q_prime = _build_q_prime(q, channels, mesh.n)
    channel_radius = mesh.scale
    matrix_size = mesh.n * len(channels)
    mass_factor = channels[0].mass_factor  # used by propagated path only
    return cast(
        DirectRMatrixKernel,
        _DirectRMatrixKernel(
            mesh=mesh,
            operators=operators,
            channels=channels,
            energies=energies,
            q=q,
            q_prime=q_prime,
            channel_radius=channel_radius,
            matrix_size=matrix_size,
            mass_factor=mass_factor,
            boundary=boundary,
            mass_factor_grid=mass_factor_grid,
        ),
    )


def make_rmatrix_direct_grid_observable(
    mesh: Mesh,
    operators: OperatorMatrices,
    channels: tuple[ChannelSpec, ...],
    energies: jax.Array,
    boundary: BoundaryValues | None,
    mass_factor_grid: jax.Array | None = None,
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
    mass_factor_grid
        Per-energy ℏ²/2μ values in MeV·fm², shape ``(N_E,)``, or ``None``
        for constant mass factor.

    Returns
    -------
    DirectGridObservable
        JIT-compiled callable: ``kernel(V_grid) → R`` where ``V_grid`` has a
        leading ``N_E`` axis and ``R`` has shape ``(N_E, N_c, N_c)``.
    """

    q = build_Q(mesh, channels)
    q_prime = _build_q_prime(q, channels, mesh.n)
    channel_radius = mesh.scale
    matrix_size = mesh.n * len(channels)
    mass_factor = channels[0].mass_factor  # used by propagated path only
    return cast(
        DirectGridObservable,
        _DirectRMatrixGridObservable(
            mesh=mesh,
            operators=operators,
            channels=channels,
            energies=energies,
            q=q,
            q_prime=q_prime,
            channel_radius=channel_radius,
            matrix_size=matrix_size,
            mass_factor=mass_factor,
            boundary=boundary,
            mass_factor_grid=mass_factor_grid,
        ),
    )


def make_smatrix_direct_observable(
    rmatrix_kernel: _DirectRMatrixKernel,
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
) -> _WavefunctionDirectKernel:
    """Build a direct wavefunction kernel ``(V, source, i) → ψ``.

    Solves ``C(E_i) ψ = source`` where ``C = H_MeV − E_i · I`` using
    ``jnp.linalg.solve``.  [DESIGN.md §11.3]
    """

    matrix_size = mesh.n * len(channels)
    return _WavefunctionDirectKernel(
        mesh=mesh,
        operators=operators,
        channels=channels,
        energies=energies,
        matrix_size=matrix_size,
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
    mass_factor: float,
    boundary: BoundaryValues | None,
) -> jax.Array:
    """Return the direct R-matrix across the compile-time energy grid.

    Dispatches to the propagated or non-propagated path depending on
    ``mesh.propagation``, and to the local or non-local potential path
    depending on ``potential.ndim``.

    The Hamiltonian is assembled in MeV (symmetric form), and the C matrix
    is ``H_MeV − E·I``.  The surface projector ``Q'`` carries the per-channel
    sqrt(m_c) factor so that ``R = Q'^T C^{-1} Q' / a`` equals the fm⁻² result
    for uniform μ and generalises to per-channel μ.  [DESIGN.md §11.5]

    Parameters
    ----------
    potential
        Assembled potential in MeV.  Local: ``(N_c, N_c, N)``; non-local:
        ``(N_c, N_c, N, N)``.
    mesh, operators, channels, energies, q_prime, channel_radius, matrix_size, mass_factor, boundary
        Compile-time cached data forwarded from the kernel dataclass.
        ``mass_factor`` is used only on the propagated path (fm⁻² units).

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
        # Hamiltonian is in MeV; C = H_MeV − E·I.
        matrix = hamiltonian - energy * jnp.eye(
            matrix_size,
            dtype=hamiltonian.dtype,
        )
        solved = cast(
            jax.Array,
            jnp.linalg.solve(
                matrix,
                q_prime,
            ),
        )
        values: jax.Array = (q_prime.T @ solved) / channel_radius
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
    q_prime: jax.Array,
    channel_radius: float,
    matrix_size: int,
    mass_factor: float,
    boundary: BoundaryValues | None,
    mass_factor_grid: jax.Array | None = None,
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
    mesh, operators, channels, energies, q, q_prime, channel_radius, matrix_size, mass_factor, boundary
        Compile-time cached data.  ``q`` is the unscaled surface projector used
        when ``mass_factor_grid`` overrides the per-channel values at JIT time.
        ``mass_factor`` is used only on the propagated path.

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
        # Hamiltonian is in MeV; C = H_MeV − E·I.
        matrix = hamiltonian - energy * jnp.eye(
            matrix_size,
            dtype=hamiltonian.dtype,
        )
        solved = cast(
            jax.Array,
            jnp.linalg.solve(
                matrix,
                q_prime,
            ),
        )
        return (q_prime.T @ solved) / channel_radius

    def one_energy_with_mu(
        potential: jax.Array,
        energy: jax.Array,
        mu_row: jax.Array,  # (N_c,) per-channel mass factors
    ) -> jax.Array:
        hamiltonian = assemble_block_hamiltonian(
            mesh,
            operators,
            channels,
            potential,
            mass_factor_override=mu_row,
        )
        # Hamiltonian assembled with override μ is in MeV; C = H_MeV − E·I.
        # Q' = diag(repeat(sqrt(mu_row), N))·Q — per-channel scaling.
        matrix = hamiltonian - energy * jnp.eye(
            matrix_size,
            dtype=hamiltonian.dtype,
        )
        n = mesh.n
        scale = jnp.repeat(jnp.sqrt(mu_row), n)  # (N_c·N,)
        q_prime_mu: jax.Array = scale[:, None] * q
        solved = cast(
            jax.Array,
            jnp.linalg.solve(
                matrix,
                q_prime_mu,
            ),
        )
        return (q_prime_mu.T @ solved) / channel_radius

    if mass_factor_grid is not None:
        # mass_factor_grid is (N_E, N_c); vmap slices to (N_c,) per energy step.
        return jax.vmap(one_energy_with_mu)(
            potentials,
            energies,
            mass_factor_grid,
        )
    return jax.vmap(one_energy)(
        potentials,
        energies,
    )


def _wavefunction_direct(
    potential: jax.Array,
    source: jax.Array,
    energy: jax.Array,
    mesh: Mesh,
    operators: OperatorMatrices,
    channels: tuple[ChannelSpec, ...],
    matrix_size: int,
) -> jax.Array:
    """Solve the internal wavefunction on the MeV direct path.

    Computes ``m₀ · (H_MeV − E·I)⁻¹ source`` where ``m₀ = channels[0].mass_factor``.
    The ``m₀`` factor makes the result equal to the fm⁻² spectral Green's function:
    ``G_spectral = (H_fm2 − E/m)⁻¹ = m · (H_MeV − E·I)⁻¹``.
    """

    hamiltonian = assemble_block_hamiltonian(mesh, operators, channels, potential)
    matrix = hamiltonian - energy * jnp.eye(matrix_size, dtype=hamiltonian.dtype)
    m0 = channels[0].mass_factor  # evaluated at JIT-trace time (static)
    result: jax.Array = cast(jax.Array, m0 * jnp.linalg.solve(matrix, source))
    return result


def _direct_smatrix_grid(
    r_grid: jax.Array,
    boundary: BoundaryValues,
) -> jax.Array:
    """Match an (N_E, N_c, N_c) R-matrix grid to the S-matrix grid."""

    return jax.vmap(smatrix_from_R)(r_grid, boundary)


def _direct_phases_grid(s_grid: jax.Array) -> jax.Array:
    """Extract phase shifts from an (N_E, N_c, N_c) S-matrix grid."""

    return jax.vmap(phases_from_S)(s_grid)


_RMATRIX_DIRECT_JIT = jax.jit(
    _rmatrix_direct,
    static_argnames=("channels", "matrix_size"),
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


__all__ = [
    "make_direct_wavefunction_kernel",
    "make_phases_direct_observable",
    "make_rmatrix_direct_grid_observable",
    "make_rmatrix_direct_kernel",
    "make_smatrix_direct_observable",
]

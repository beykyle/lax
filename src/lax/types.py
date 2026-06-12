"""Core types: user-facing specs, solver-bundle pytrees, and runtime protocols. See ``DESIGN.md §6``."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol

import jax

from lax.spectral.types import BoundaryValues

if TYPE_CHECKING:
    from lax.spectral.types import Spectrum

type MeshFamily = Literal["legendre", "laguerre"]
type Regularization = Literal[
    "x",
    "x^3/2",
    "x(1-x)",
    "modified_x^2",
]
type Method = Literal["eigh", "eig", "linear_solve"]


def _empty_extras() -> dict[str, object]:
    """Return a typed empty mapping for mesh-specific extra options."""

    return {}


@dataclass(frozen=True)
class MeshSpec:
    """User-facing mesh specification passed to :func:`lax.compile`.

    Attributes
    ----------
    family
        Mesh family registered in :mod:`lax.meshes`. The current public API
        supports ``"legendre"`` and ``"laguerre"``.
    regularization
        Endpoint regularization used by the chosen family. The currently
        supported combinations are:

        - Legendre: ``"x"``, ``"x(1-x)"``, ``"x^3/2"``
        - Laguerre: ``"x"``, ``"modified_x^2"``
    n
        Number of mesh basis functions.
    scale
        Physical length scale for the mesh. For finite-interval meshes this is
        the channel radius; for semi-infinite meshes it is the radial scaling
        factor described in ``DESIGN.md``.
    extras
        Mesh-specific compile-time options forwarded to the registered mesh
        builder.
    """

    family: MeshFamily
    regularization: Regularization
    n: int
    scale: float
    extras: dict[str, object] = field(default_factory=_empty_extras)


@dataclass(frozen=True)
class ChannelSpec:
    """One scattering channel baked into the compiled solver structure.

    Attributes
    ----------
    l
        Orbital angular momentum for the channel.
    threshold
        Channel threshold in MeV. Assembly code converts it to fm^-2 using
        ``mass_factor``.
    mass_factor
        Conversion factor ``ℏ² / 2μ`` in MeV·fm².  Required — there is no
        default, since any fixed value would be physically meaningless for
        an arbitrary nucleus.  Use :func:`lax.constants.hbar2_over_2mu` to
        compute it from particle masses in AMU, e.g.
        ``lax.constants.hbar2_over_2mu(1.008665, 1.008665)`` ≈ 41.47 MeV·fm²
        for nucleon–nucleon systems.
    """

    l: int
    threshold: float
    mass_factor: float | jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Interaction:
    """Assembled coupled-channel potential block in MeV.

    block : (M, M), (N_E, M, M), (N_b, M, M), or (N_b, N_E, M, M)  where M = N_c·N
        Local terms on the per-channel diagonal, non-local terms as full
        Gauss-scaled blocks. Symmetric. Mass-independent — per-channel mass
        factors are applied by the solver, never folded into this block.
        Excludes kinetic, centrifugal, threshold, and energy terms.
        Canonical axis order is block × energy (DESIGN.md §15.5): the
        symmetry-block axis, when present, always leads.
    energy_dependent : bool (static)
        True iff ``block`` has an (N_E,) axis aligned with the compile-time
        energy grid (leading, or directly after the block axis).
    block_dependent : bool (static)
        True iff ``block`` has a leading (N_b,) symmetry-block axis aligned
        with the compile-time ``blocks=`` set (DESIGN.md §15.5).  Parallel to
        ``energy_dependent``; composes with it as ``(N_b, N_E, M, M)``.
    """

    block: jax.Array
    energy_dependent: bool = field(metadata={"static": True})
    block_dependent: bool = field(default=False, metadata={"static": True})

    def __add__(self, other: object) -> Interaction:
        """Combine two Interaction blocks by summing their potential contributions.

        Each static axis flag propagates by OR: any operand carrying the
        energy (block) axis makes the sum energy- (block-) dependent, and the
        other operand is broadcast across the missing axis.  Axes follow the
        canonical block × energy order, so ``(N_E, M, M) + (N_b, M, M)``
        yields ``(N_b, N_E, M, M)``.
        """
        if not isinstance(other, Interaction):
            return NotImplemented
        energy_dependent = self.energy_dependent or other.energy_dependent
        block_dependent = self.block_dependent or other.block_dependent
        b1 = _lift_block(
            self.block,
            self.energy_dependent,
            self.block_dependent,
            energy_dependent,
            block_dependent,
        )
        b2 = _lift_block(
            other.block,
            other.energy_dependent,
            other.block_dependent,
            energy_dependent,
            block_dependent,
        )
        return Interaction(
            block=b1 + b2,
            energy_dependent=energy_dependent,
            block_dependent=block_dependent,
        )

    def __radd__(self, other: object) -> Interaction:
        if other == 0:
            return self
        return NotImplemented


def _lift_block(
    block: jax.Array,
    has_energy: bool,
    has_block: bool,
    want_energy: bool,
    want_block: bool,
) -> jax.Array:
    """Insert size-1 axes so ``block`` broadcasts to the target axis layout.

    Target layout is the canonical ``[N_b,][N_E,] M, M`` order: a missing
    block axis is inserted at position 0, a missing energy axis after it.
    """

    if want_block and not has_block:
        block = block[None]
    if want_energy and not has_energy:
        block = block[:, None] if want_block else block[None]
    return block


type EnergyLike = float | jax.Array


class SpectrumKernel(Protocol):
    """Callable that maps a potential to its spectral decomposition."""

    def __call__(
        self,
        potential: jax.Array | Interaction,
    ) -> Spectrum:
        """Return the spectral decomposition for one potential.

        Parameters
        ----------
        potential
            An :class:`~lax.Interaction`.  Energy-independent interactions yield
            one :class:`Spectrum`; energy-dependent interactions are dispatched
            internally over the energy axis and yield a batched :class:`Spectrum`.
            Raw potential arrays are rejected at runtime — build an Interaction
            first via ``solver.local_potential()`` / ``solver.nonlocal_potential()``
            or ``solver.interaction_from_*``.

        Returns
        -------
        Spectrum
            Eigendecomposition of the Bloch-augmented Hamiltonian.  For a
            solver compiled with ``blocks=`` (§15.5) the leaves carry a
            leading ``(N_b,)`` axis.
        """
        ...


class RMatrixObservable(Protocol):
    """Callable that evaluates the spectral R-matrix at an arbitrary energy."""

    def __call__(self, spectrum: Spectrum, energy: EnergyLike) -> jax.Array:
        """Evaluate the R-matrix at one energy.

        Parameters
        ----------
        spectrum
            Spectral decomposition produced by ``solver.spectrum(V)``.
        energy
            Energy in MeV (scalar).

        Returns
        -------
        jax.Array
            R-matrix, shape ``(N_c, N_c)`` — ``(N_b, N_c, N_c)`` for a solver
            compiled with ``blocks=`` (§15.5).
        """
        ...


class SpectrumObservable(Protocol):
    """Callable that evaluates one observable on the compile-time energy grid."""

    def __call__(self, spectrum: Spectrum) -> jax.Array:
        """Evaluate an observable on the compile-time energy grid.

        Parameters
        ----------
        spectrum
            Spectral decomposition produced by ``solver.spectrum(V)``.

        Returns
        -------
        jax.Array
            Observable values on the compile-time grid, shape ``(N_E, ...)``
            — with a leading ``(N_b,)`` axis for a solver compiled with
            ``blocks=`` (§15.5).
        """
        ...


class SpectrumGridObservable(Protocol):
    """Callable for aligned-grid observables from a batched Spectrum."""

    def __call__(self, spectra: Spectrum) -> jax.Array:
        """Evaluate one observable per compile-time energy / Spectrum pair.

        Parameters
        ----------
        spectra
            Batched ``Spectrum`` produced by calling ``solver.spectrum`` on an
            energy-dependent :class:`~lax.Interaction` (internal dispatch), with a
            leading batch axis of size ``N_E``.

        Returns
        -------
        jax.Array
            Observable values, shape ``(N_E, ...)`` — with a leading
            ``(N_b,)`` axis for a solver compiled with ``blocks=`` (§15.5).
        """
        ...


class GreenFunctionObservable(Protocol):
    """Callable that evaluates the Green's function at an arbitrary energy."""

    def __call__(self, spectrum: Spectrum, energy: EnergyLike) -> jax.Array:
        """Evaluate the Green's function at one energy.

        Parameters
        ----------
        spectrum
            Spectral decomposition; must have been produced with
            ``'greens'`` in ``solvers=``.
        energy
            Energy in MeV (scalar).

        Returns
        -------
        jax.Array
            Resolvent ``(H - E/μ)⁻¹``, shape ``(M, M)`` — ``(N_b, M, M)``
            for a solver compiled with ``blocks=`` (§15.5).
        """
        ...


class WavefunctionObservable(Protocol):
    """Callable that reconstructs the internal scattering wavefunction."""

    def __call__(self, spectrum: Spectrum, energy: EnergyLike, source: jax.Array) -> jax.Array:
        """Evaluate the internal wavefunction at one energy.

        Parameters
        ----------
        spectrum
            Spectral decomposition; must have been produced with
            ``'wavefunction'`` in ``solvers=``.
        energy
            Energy in MeV (scalar).
        source
            Mesh-space driving term, shape ``(N_c · N,)``.  Use
            :func:`lax.make_wavefunction_source` to build it.

        Returns
        -------
        jax.Array
            Internal wavefunction coefficients in the mesh basis, shape
            ``(M,)`` — ``(N_b, M)`` for a solver compiled with ``blocks=``
            (§15.5), where ``source`` is ``(N_b, M)`` or a shared ``(M,)``.
        """
        ...


class EigenpairAccessor(Protocol):
    """Callable for raw access to eigenvalues and eigenvectors.

    Raises ``RuntimeError`` if eigenvectors were not retained at compile time
    (i.e. neither ``'greens'`` nor ``'wavefunction'`` was in ``solvers=``).
    """

    def __call__(self, spectrum: Spectrum) -> tuple[jax.Array, jax.Array]:
        """Return the stored eigenvalues and eigenvectors.

        Parameters
        ----------
        spectrum
            Spectral decomposition with retained eigenvectors.

        Returns
        -------
        tuple[jax.Array, jax.Array]
            ``(eigenvalues, eigenvectors)`` — shapes ``(M,)`` and ``(M, M)``.

        Raises
        ------
        RuntimeError
            If eigenvectors were not retained at compile time.
        """
        ...


class DirectRMatrixKernel(Protocol):
    """Callable that computes the direct R-matrix via per-energy linear solves."""

    def __call__(self, potential: jax.Array | Interaction) -> jax.Array:
        """Evaluate the direct R-matrix on the compile-time energy grid.

        Parameters
        ----------
        potential
            An :class:`~lax.Interaction` object (energy-independent or
            energy-dependent; dispatch is handled transparently).  Raw potential
            arrays are rejected at runtime.

        Returns
        -------
        jax.Array
            R-matrix on the compile-time grid, shape ``(N_E, N_c, N_c)`` —
            ``(N_b, N_E, N_c, N_c)`` for a solver compiled with ``blocks=``
            (§15.5).
        """
        ...


class SMatrixDirectObservable(Protocol):
    """Callable that computes the direct S-matrix via per-energy linear solves."""

    def __call__(self, potential: jax.Array | Interaction) -> jax.Array:
        """Evaluate the S-matrix on the compile-time energy grid.

        Parameters
        ----------
        potential
            An :class:`~lax.Interaction`; raw arrays are rejected at runtime.

        Returns
        -------
        jax.Array
            S-matrix on the compile-time grid, shape ``(N_E, N_c, N_c)``,
            complex — ``(N_b, N_E, N_c, N_c)`` for a solver compiled with
            ``blocks=`` (§15.5).

        Notes
        -----
        ``potential`` must be an :class:`~lax.Interaction`; raw arrays are
        rejected at runtime.
        """
        ...


class PhasesDirectObservable(Protocol):
    """Callable that computes direct phase shifts via per-energy linear solves."""

    def __call__(self, potential: jax.Array | Interaction) -> jax.Array:
        """Evaluate phase shifts on the compile-time energy grid.

        Parameters
        ----------
        potential
            An :class:`~lax.Interaction`; raw arrays are rejected at runtime.

        Returns
        -------
        jax.Array
            Phase shifts, shape ``(N_E, N_c)``, in radians —
            ``(N_b, N_E, N_c)`` for a solver compiled with ``blocks=`` (§15.5).
        """
        ...


class WavefunctionDirectObservable(Protocol):
    """Callable that reconstructs the wavefunction via a direct linear solve."""

    def __call__(
        self,
        potential: jax.Array | Interaction,
        source: jax.Array,
        energy_index: int,
    ) -> jax.Array:
        """Solve ``C(E_i) ψ = source`` for the internal wavefunction.

        Parameters
        ----------
        potential
            An :class:`~lax.Interaction`; raw arrays are rejected at runtime.
        source
            Mesh-space driving term, shape ``(N_c·N,)``.
        energy_index
            Index into the compile-time energy grid (Python int).

        Returns
        -------
        jax.Array
            Internal wavefunction coefficient vector, shape ``(N_c·N,)`` —
            ``(N_b, N_c·N)`` for a solver compiled with ``blocks=`` (§15.5),
            where ``source`` is ``(N_b, N_c·N)`` or a shared ``(N_c·N,)``.
        """
        ...


class GridVectorTransform(Protocol):
    """Callable that projects mesh coefficients onto a fine radial grid."""

    def __call__(self, values: jax.Array) -> jax.Array:
        """Project mesh coefficients onto a radial grid.

        Parameters
        ----------
        values
            Mesh coefficient vector, shape ``(N,)``.

        Returns
        -------
        jax.Array
            Radial function values, shape ``(M_r,)``.
        """
        ...


class FromGridVectorTransform(Protocol):
    """Callable that projects fine-grid values back onto the mesh basis."""

    def __call__(
        self,
        values: jax.Array | Callable[[jax.Array], jax.Array],
    ) -> jax.Array:
        """Project sampled radial-grid values or a callable profile onto the mesh basis.

        Parameters
        ----------
        values
            Either a ``(M_r,)`` array of grid samples or a callable
            ``f(r_grid) → (M_r,)`` that is evaluated on the compile-time grid.

        Returns
        -------
        jax.Array
            Mesh coefficient vector, shape ``(N,)``.
        """
        ...


class GridMatrixTransform(Protocol):
    """Callable that projects a mesh-space kernel onto a fine radial grid."""

    def __call__(self, values: jax.Array) -> jax.Array:
        """Project a mesh-space kernel onto a radial grid.

        Parameters
        ----------
        values
            Mesh kernel matrix, shape ``(N, N)``.

        Returns
        -------
        jax.Array
            Kernel on the fine grid, shape ``(M_r, M_r)``.
        """
        ...


class FourierTransform(Protocol):
    """Callable that maps mesh coefficients or kernels to momentum space."""

    def __call__(self, values: jax.Array, channel_index: int = 0) -> jax.Array:
        """Project mesh coefficients or a kernel onto a momentum grid.

        Parameters
        ----------
        values
            Mesh vector ``(N,)`` or kernel ``(N, N)``.
        channel_index
            Which channel's angular momentum to use for the spherical Bessel
            transform.

        Returns
        -------
        jax.Array
            Momentum-space array, shape ``(M_k,)`` or ``(M_k, M_k)``.
        """
        ...


class DoubleFourierTransform(Protocol):
    """Callable for the double Bessel transform of a mesh-space kernel."""

    def __call__(
        self,
        values: jax.Array,
        left_channel_index: int = 0,
        right_channel_index: int | None = None,
    ) -> jax.Array:
        """Project a mesh-space kernel onto left/right momentum grids.

        Parameters
        ----------
        values
            Mesh kernel, shape ``(N, N)``.
        left_channel_index
            Channel angular momentum for the left (row) transform.
        right_channel_index
            Channel angular momentum for the right (column) transform.
            Defaults to ``left_channel_index``.

        Returns
        -------
        jax.Array
            Double-transformed kernel, shape ``(M_k, M_k)``.
        """
        ...


class Integrator(Protocol):
    """Callable for norms and expectation values in the mesh basis."""

    def __call__(self, values: jax.Array, operator: jax.Array | None = None) -> jax.Array:
        """Integrate mesh coefficients with an optional operator insertion.

        Parameters
        ----------
        values
            Mesh coefficient vector, shape ``(N,)``.
        operator
            Optional ``(N, N)`` operator matrix.  When ``None``, computes
            the norm ``⟨ψ|ψ⟩``.

        Returns
        -------
        jax.Array
            Scalar expectation value ``⟨ψ|O|ψ⟩``.
        """
        ...


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class PropagationMatrices:
    """Precomputed subinterval-propagation matrices for Legendre-x meshes.

    Produced once by :func:`lax.propagate.build_legendre_x_propagation` and
    stored inside the :class:`Mesh`.  All matrices are in fm⁻² units.

    Attributes
    ----------
    n_intervals
        Number of subintervals (static).
    basis_size_per_interval
        Number of Legendre basis functions per subinterval (static).
    interval_width
        Width of each subinterval in fm (static).
    local_nodes
        Legendre quadrature nodes on ``(0, 1)``, shape ``(basis_size,)``.
    local_weights
        Legendre quadrature weights, shape ``(basis_size,)``.
    kinetic
        Per-interval kinetic matrices, shape ``(n_intervals, basis_size, basis_size)``.
    blo0
        Bloch surface-overlap matrix for the left boundary of the first interval.
    blo1
        Bloch surface-overlap matrix at the left boundary of subsequent intervals.
    blo2
        Bloch surface-overlap matrix at the right boundary of interior intervals.
    q1
        Left surface-projector vectors, shape ``(n_intervals, basis_size)``.
    q2
        Right surface-projector vectors, shape ``(n_intervals, basis_size)``.
    """

    n_intervals: int = field(metadata={"static": True})
    basis_size_per_interval: int = field(metadata={"static": True})
    interval_width: float = field(metadata={"static": True})
    local_nodes: jax.Array
    local_weights: jax.Array
    kinetic: jax.Array
    blo0: jax.Array
    blo1: jax.Array
    blo2: jax.Array
    q1: jax.Array
    q2: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Mesh:
    """Concrete mesh data cached inside a compiled solver.

    Produced by the mesh registry and embedded in the :class:`Solver` at
    compile time.  Static fields are baked into the JAX JIT cache key;
    changing them requires recompilation.

    Attributes
    ----------
    family
        Mesh family, e.g. ``"legendre"`` or ``"laguerre"`` (static).
    regularization
        Endpoint regularization, e.g. ``"x"`` or ``"x(1-x)"`` (static).
    n
        Number of basis functions per channel (static).
    scale
        Physical scale: channel radius ``a`` in fm for finite-interval
        meshes, or the Laguerre scaling factor ``h`` in fm (static).
    n_intervals
        Number of subintervals for propagated meshes; ``1`` otherwise (static).
    basis_size_per_interval
        Basis functions per subinterval; equals ``n`` when ``n_intervals == 1``
        (static).
    nodes
        Canonical mesh nodes on ``(0, 1)`` or ``(0, ∞)``, shape ``(n,)``.
    weights
        Gauss quadrature weights λ_i, shape ``(n,)``.
    radii
        Physical radial mesh points r_i = scale · x_i, shape ``(n,)``.
    basis_at_boundary
        Lagrange basis values φ_j(a) at the channel surface, shape ``(n,)``.
        All zeros for semi-infinite (Laguerre) meshes.
    propagation
        Subinterval propagation matrices, or ``None`` for single-interval meshes.
    """

    family: MeshFamily = field(metadata={"static": True})
    regularization: Regularization = field(metadata={"static": True})
    n: int = field(metadata={"static": True})
    scale: float = field(metadata={"static": True})
    n_intervals: int = field(metadata={"static": True})
    basis_size_per_interval: int = field(metadata={"static": True})
    nodes: jax.Array
    weights: jax.Array
    radii: jax.Array
    basis_at_boundary: jax.Array
    propagation: PropagationMatrices | None = None


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class OperatorMatrices:
    """Precomputed single-channel operator matrices in fm⁻² units.

    All populated fields are ``(N, N)`` symmetric real matrices in the
    Lagrange-mesh basis.  Unrequested operators are ``None``.

    Attributes
    ----------
    T
        Kinetic-energy matrix ``-d²/dr²`` (Laguerre meshes, no Bloch term).
    TpL
        Bloch-augmented kinetic matrix ``T + L(B=0)``.  This is the standard
        operator for R-matrix calculations on finite-interval (Legendre) meshes.
    T_alpha
        Hyperradial kinetic matrix for α-type coordinates (Laguerre meshes with
        three-body regularization).
    D
        First-derivative matrix ``d/dr``.
    inv_r
        Diagonal ``1/r`` matrix.
    inv_r2
        Diagonal ``1/r²`` matrix.
    """

    T: jax.Array | None = None
    TpL: jax.Array | None = None
    T_alpha: jax.Array | None = None
    D: jax.Array | None = None
    inv_r: jax.Array | None = None
    inv_r2: jax.Array | None = None


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class TransformMatrices:
    """Precomputed matrices for radial-grid and momentum-space transforms.

    All fields default to ``None``; only those corresponding to the ``grid``
    and ``momenta`` arguments supplied to :func:`lax.compile` are populated.

    Attributes
    ----------
    B_grid
        Basis-evaluation matrix ``B[k, j] = f_j(r_k)``,
        shape ``(M_r, N)``.  Used by ``to_grid_vector`` and ``to_grid_matrix``.
    grid_r
        Fine radial grid passed to :func:`lax.compile`, shape ``(M_r,)`` in fm.
        Also accessible as ``solver.grid_r``.
    F_momentum
        Fourier-Bessel transform matrices, one per channel,
        shape ``(N_c, M_k, N)``.  Used by the ``fourier`` callable.
    momenta
        Momentum grid passed to :func:`lax.compile`, shape ``(M_k,)`` in fm⁻¹.
        Also accessible as ``solver.momenta``.
    """

    B_grid: jax.Array | None = None
    grid_r: jax.Array | None = None
    F_momentum: jax.Array | None = None
    momenta: jax.Array | None = None


@dataclass(frozen=True)
class Solver:
    """Compiled solver bundle produced by :func:`lax.compile`.

    Holds all compile-time caches (mesh, operators, boundary values, transform
    matrices) alongside JIT-compiled, pickle-safe runtime callables.  Call
    ``print(solver)`` to see which observables were compiled.

    Attributes
    ----------
    mesh
        Compiled mesh data (nodes, weights, radii, boundary values).
    operators
        Precomputed single-channel operator matrices in fm⁻².
    channels
        Channel definitions baked into the solver at compile time.
    energies
        Compile-time energy grid in MeV, shape ``(N_E,)``.
    boundary
        Coulomb/Whittaker boundary values at ``r = a``, or ``None`` if no
        energy grid was supplied.
    transforms
        Precomputed radial-grid and momentum-space transform matrices.
    method
        Linear-algebra backend: ``"eigh"``, ``"eig"``, or ``"linear_solve"``.
    mass_factor_grid
        Per-energy ℏ²/2μ values in MeV·fm², shape ``(N_E,)``, or ``None``
        when a constant mass factor is used.  Stored here so the aligned-grid
        observables can use the correct μ(E) at each energy point.
    blocks
        The symmetry-block set passed to ``lax.compile(blocks=…)``, or ``None``
        for a channels-compiled solver (DESIGN.md §15.5).  When set, ``channels``
        holds the template block ``blocks[0]``, ``boundary`` carries a leading
        ``(N_b,)`` axis, and every observable output gains a leading block axis.

    **Spectral-path observables** (present when ``method`` is ``"eigh"``/``"eig"``):

    spectrum
        ``(V) → Spectrum`` — one eigendecomposition per potential.
    rmatrix
        ``(spectrum, E) → R(E)`` — R-matrix at any scalar energy.
    smatrix
        ``(spectrum) → S`` — S-matrix on the compile-time energy grid.
    phases
        ``(spectrum) → δ`` — phase shifts ``(N_E, N_c)`` in radians.
    greens
        ``(spectrum, E) → G(E)`` — Green's function; requires ``'greens'``
        in ``solvers=``.
    wavefunction
        ``(spectrum, E, source) → ψ_int`` — internal wavefunction; requires
        ``'wavefunction'`` in ``solvers=``.
    eigh
        ``(spectrum) → (ε, U)`` — raw eigenpairs; raises if eigenvectors
        were not retained.
    rmatrix_grid
        ``(spectra) → R`` — aligned-grid R for energy-dependent workflows.
    smatrix_grid
        ``(spectra) → S`` — aligned-grid S.
    phases_grid
        ``(spectra) → δ`` — aligned-grid phases.

    **Direct-path observables** (present when ``"rmatrix_direct"`` in ``solvers=``):

    rmatrix_direct
        ``(V) → R`` — per-energy linear-solve R-matrix on the compile-time grid.

    **Transform helpers**:

    to_grid_vector
        ``(c) → ψ(r)`` — mesh coefficients to fine radial grid.
    from_grid_vector
        ``(ψ_or_fn) → c`` — fine grid values back to mesh coefficients.
    to_grid_matrix
        ``(V) → V(r, r')`` — mesh kernel to fine radial grid.
    fourier
        ``(c, channel_index=0) → ũ(k)`` — momentum-space transform.
    double_fourier_transform
        ``(V, ...) → V(p, p')`` — double Bessel transform for kernels.
    integrate
        ``(c, operator=None) → ⟨ψ|O|ψ⟩`` — norms and expectation values.
    """

    mesh: Mesh
    operators: OperatorMatrices
    channels: tuple[ChannelSpec, ...]
    energies: jax.Array
    boundary: BoundaryValues | None
    transforms: TransformMatrices
    method: Method
    mass_factor_grid: jax.Array | None = None
    blocks: tuple[tuple[ChannelSpec, ...], ...] | None = None
    spectrum: SpectrumKernel | None = None
    rmatrix: RMatrixObservable | None = None
    smatrix: SpectrumObservable | None = None
    phases: SpectrumObservable | None = None
    greens: GreenFunctionObservable | None = None
    wavefunction: WavefunctionObservable | None = None
    eigh: EigenpairAccessor | None = None
    rmatrix_grid: SpectrumGridObservable | None = None
    smatrix_grid: SpectrumGridObservable | None = None
    phases_grid: SpectrumGridObservable | None = None
    rmatrix_direct: DirectRMatrixKernel | None = None
    smatrix_direct: SMatrixDirectObservable | None = None
    phases_direct: PhasesDirectObservable | None = None
    wavefunction_direct: WavefunctionDirectObservable | None = None
    interaction_from_block: Callable[..., Any] | None = None
    interaction_from_array: Callable[..., Any] | None = None
    interaction_from_funcs: Callable[..., Any] | None = None
    local_potential: Callable[..., Any] | None = None
    nonlocal_potential: Callable[..., Any] | None = None
    to_grid_vector: GridVectorTransform | None = None
    from_grid_vector: FromGridVectorTransform | None = None
    to_grid_matrix: GridMatrixTransform | None = None
    fourier: FourierTransform | None = None
    double_fourier_transform: DoubleFourierTransform | None = None
    integrate: Integrator | None = None

    # ------------------------------------------------------------------
    # Convenience properties

    @property
    def grid_r(self) -> jax.Array | None:
        """Radial grid passed to :func:`lax.compile`, or ``None``."""
        return self.transforms.grid_r

    @property
    def momenta(self) -> jax.Array | None:
        """Momentum grid passed to :func:`lax.compile`, or ``None``."""
        return self.transforms.momenta

    # ------------------------------------------------------------------
    # Human-readable repr

    def __repr__(self) -> str:
        _observable_names = (
            "spectrum",
            "rmatrix",
            "smatrix",
            "phases",
            "greens",
            "wavefunction",
            "eigh",
            "rmatrix_grid",
            "smatrix_grid",
            "phases_grid",
            "rmatrix_direct",
            "smatrix_direct",
            "phases_direct",
            "wavefunction_direct",
        )
        _transform_names = (
            "to_grid_vector",
            "to_grid_matrix",
            "fourier",
            "integrate",
        )
        live = [n for n in _observable_names if getattr(self, n) is not None]
        transforms = [n for n in _transform_names if getattr(self, n) is not None]
        n_e = len(self.energies)
        block_info = f", {len(self.blocks)} blocks" if self.blocks is not None else ""
        return (
            f"Solver({self.mesh.family}/{self.mesh.regularization} "
            f"n={self.mesh.n} scale={self.mesh.scale}fm, "
            f"method={self.method}, {n_e} {'energy' if n_e == 1 else 'energies'}"
            f"{block_info})\n"
            f"  observables: {' '.join(live) or 'none'}\n"
            f"  transforms:  {' '.join(transforms) or 'none'}"
        )


__all__ = [
    "BoundaryValues",
    "ChannelSpec",
    "DirectRMatrixKernel",
    "DoubleFourierTransform",
    "EigenpairAccessor",
    "EnergyLike",
    "FourierTransform",
    "FromGridVectorTransform",
    "GreenFunctionObservable",
    "GridMatrixTransform",
    "GridVectorTransform",
    "Integrator",
    "Interaction",
    "Mesh",
    "MeshFamily",
    "MeshSpec",
    "Method",
    "OperatorMatrices",
    "PhasesDirectObservable",
    "PropagationMatrices",
    "Regularization",
    "RMatrixObservable",
    "SMatrixDirectObservable",
    "Solver",
    "SpectrumGridObservable",
    "SpectrumKernel",
    "SpectrumObservable",
    "TransformMatrices",
    "WavefunctionDirectObservable",
    "WavefunctionObservable",
]

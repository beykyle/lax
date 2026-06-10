"""Internal pytrees and callable protocols. See DESIGN.md §6."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

import jax

from lax.types import ChannelSpec, MeshFamily, Method, Regularization

if TYPE_CHECKING:
    from lax.spectral.types import Spectrum

type EnergyLike = float | jax.Array


class SpectrumKernel(Protocol):
    """Callable that maps a potential to its spectral decomposition."""

    def __call__(
        self,
        potential: jax.Array,
        mass_factor: float | jax.Array | None = None,
    ) -> Spectrum:
        """Return the spectral decomposition for one assembled potential.

        Parameters
        ----------
        potential
            Assembled potential array, shape ``(N_c, N_c, N)`` for local or
            ``(N_c, N_c, N, N)`` for non-local.
        mass_factor
            Optional per-energy ℏ²/2μ in MeV·fm².  When provided, overrides
            ``ChannelSpec.mass_factor`` so the Hamiltonian uses ``V/μ(E)`` and
            ``threshold/μ(E)`` at each energy.  Typical usage::

                spectra = jax.vmap(
                    lambda V, mu: solver.spectrum(V, mass_factor=mu)
                )(V_grid, mu_grid)

        Returns
        -------
        Spectrum
            Eigendecomposition of the Bloch-augmented Hamiltonian.
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
            R-matrix, shape ``(N_c, N_c)``.
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
            Observable values on the compile-time grid, shape ``(N_E, ...)``.
        """
        ...


class SpectrumGridObservable(Protocol):
    """Callable for aligned-grid observables from a batched Spectrum."""

    def __call__(self, spectra: Spectrum) -> jax.Array:
        """Evaluate one observable per compile-time energy / Spectrum pair.

        Parameters
        ----------
        spectra
            Batched ``Spectrum`` produced by ``jax.vmap(solver.spectrum)(V_grid)``,
            with a leading batch axis of size ``N_E``.

        Returns
        -------
        jax.Array
            Observable values, shape ``(N_E, ...)``.
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
            Resolvent ``(H - E/μ)⁻¹``, shape ``(M, M)``.
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
            Internal wavefunction coefficients in the mesh basis, shape ``(M,)``.
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

    def __call__(self, potential: jax.Array) -> jax.Array:
        """Evaluate the direct R-matrix on the compile-time energy grid.

        Parameters
        ----------
        potential
            Assembled potential array, shape ``(N_c, N_c, N)`` / ``(N_c, N_c, N, N)``
            for energy-independent V, or an :class:`~lax.Interaction` object
            (dispatch is handled transparently at the Python level).

        Returns
        -------
        jax.Array
            R-matrix on the compile-time grid, shape ``(N_E, N_c, N_c)``.
        """
        ...


class SMatrixDirectObservable(Protocol):
    """Callable that computes the direct S-matrix via per-energy linear solves."""

    def __call__(self, potential: jax.Array) -> jax.Array:
        """Evaluate the S-matrix on the compile-time energy grid.

        Parameters
        ----------
        potential
            Assembled potential or :class:`~lax.Interaction`.

        Returns
        -------
        jax.Array
            S-matrix on the compile-time grid, shape ``(N_E, N_c, N_c)``, complex.
        """
        ...


class PhasesDirectObservable(Protocol):
    """Callable that computes direct phase shifts via per-energy linear solves."""

    def __call__(self, potential: jax.Array) -> jax.Array:
        """Evaluate phase shifts on the compile-time energy grid.

        Parameters
        ----------
        potential
            Assembled potential or :class:`~lax.Interaction`.

        Returns
        -------
        jax.Array
            Phase shifts, shape ``(N_E, N_c)``, in radians.
        """
        ...


class WavefunctionDirectObservable(Protocol):
    """Callable that reconstructs the wavefunction via a direct linear solve."""

    def __call__(
        self,
        potential: jax.Array,
        source: jax.Array,
        energy_index: int,
    ) -> jax.Array:
        """Solve ``C(E_i) ψ = source`` for the internal wavefunction.

        Parameters
        ----------
        potential
            Assembled potential or :class:`~lax.Interaction`.
        source
            Mesh-space driving term, shape ``(N_c·N,)``.
        energy_index
            Index into the compile-time energy grid (Python int).

        Returns
        -------
        jax.Array
            Internal wavefunction coefficient vector, shape ``(N_c·N,)``.
        """
        ...


class DirectGridObservable(Protocol):
    """Callable for aligned-grid observables from per-energy potential batches."""

    def __call__(self, potentials: jax.Array) -> jax.Array:
        """Evaluate one observable per compile-time energy / potential pair.

        Parameters
        ----------
        potentials
            Per-energy potentials, shape ``(N_E, N_c, N_c, N)``.

        Returns
        -------
        jax.Array
            Observable values, shape ``(N_E, ...)``.
        """
        ...


class InterpolatorBuilder(Protocol):
    """Callable that builds a Padé interpolator over the compile-time grid."""

    def __call__(
        self,
        values: jax.Array,
        order: tuple[int, int] | None = None,
    ) -> Callable[[EnergyLike], jax.Array]:
        """Build a Padé interpolator over the solver's compile-time energy grid.

        Parameters
        ----------
        values
            Observable samples, shape ``(N_E, ...)``.
        order
            Padé numerator/denominator degrees ``(p, q)`` with ``p + q + 1 == N_E``.
            Defaults to the diagonal approximant.

        Returns
        -------
        Callable[[EnergyLike], jax.Array]
            JIT-compiled interpolant; call it at any energy to evaluate.
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
class BoundaryValues:
    """Coulomb and Whittaker boundary values at the channel radius.

    Precomputed at compile time by ``mpmath`` for every ``(energy, channel)``
    pair.  Open channels use Coulomb Hankel functions; closed channels use
    Whittaker functions that decay exponentially into the barrier.

    Attributes
    ----------
    H_plus
        Outgoing Coulomb Hankel function ``H⁺ = G + iF`` at ``r = a``,
        shape ``(N_E, N_c)``, complex.
    H_minus
        Incoming Coulomb Hankel function ``H⁻ = G - iF`` at ``r = a``,
        shape ``(N_E, N_c)``, complex.
    H_plus_p
        ``ρ · d/dρ H⁺`` evaluated at ``ρ = ka``,
        shape ``(N_E, N_c)``, complex.
    H_minus_p
        ``ρ · d/dρ H⁻`` evaluated at ``ρ = ka``,
        shape ``(N_E, N_c)``, complex.
    is_open
        Boolean mask: ``True`` for open channels (``E > E_threshold``),
        shape ``(N_E, N_c)``.
    k
        Channel wave numbers ``k_c(E)`` in fm⁻¹, shape ``(N_E, N_c)``,
        or ``None`` when not needed for matching.
    """

    H_plus: jax.Array
    H_minus: jax.Array
    H_plus_p: jax.Array
    H_minus_p: jax.Array
    is_open: jax.Array
    k: jax.Array | None = None


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
    rmatrix_direct_grid
        ``(V_grid) → R`` — aligned-grid direct R for energy-dependent V.
    smatrix_direct_grid
        ``(V_grid) → S`` — aligned-grid direct S.
    phases_direct_grid
        ``(V_grid) → δ`` — aligned-grid direct phases.

    **Padé interpolation builders** (present whenever ``energies`` was supplied):

    interpolate_rmatrix, interpolate_smatrix, interpolate_phases
        ``(samples, order=None) → callable`` — build a Padé interpolant over
        the compile-time grid.

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
    rmatrix_direct_grid: DirectGridObservable | None = None
    smatrix_direct_grid: DirectGridObservable | None = None
    phases_direct_grid: DirectGridObservable | None = None
    smatrix_direct: SMatrixDirectObservable | None = None
    phases_direct: PhasesDirectObservable | None = None
    wavefunction_direct: WavefunctionDirectObservable | None = None
    interaction_from_block: Callable | None = None
    interaction_from_array: Callable | None = None
    interaction_from_funcs: Callable | None = None
    potential: Callable | None = None
    interpolate_rmatrix: InterpolatorBuilder | None = None
    interpolate_smatrix: InterpolatorBuilder | None = None
    interpolate_phases: InterpolatorBuilder | None = None
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
        return (
            f"Solver({self.mesh.family}/{self.mesh.regularization} "
            f"n={self.mesh.n} scale={self.mesh.scale}fm, "
            f"method={self.method}, {n_e} {'energy' if n_e == 1 else 'energies'})\n"
            f"  observables: {' '.join(live) or 'none'}\n"
            f"  transforms:  {' '.join(transforms) or 'none'}"
        )


__all__ = [
    "BoundaryValues",
    "DoubleFourierTransform",
    "DirectGridObservable",
    "DirectRMatrixKernel",
    "EigenpairAccessor",
    "EnergyLike",
    "FourierTransform",
    "FromGridVectorTransform",
    "GreenFunctionObservable",
    "GridMatrixTransform",
    "GridVectorTransform",
    "InterpolatorBuilder",
    "Integrator",
    "Mesh",
    "OperatorMatrices",
    "PropagationMatrices",
    "RMatrixObservable",
    "Solver",
    "SpectrumGridObservable",
    "SpectrumKernel",
    "SpectrumObservable",
    "TransformMatrices",
    "PhasesDirectObservable",
    "SMatrixDirectObservable",
    "WavefunctionDirectObservable",
    "WavefunctionObservable",
]

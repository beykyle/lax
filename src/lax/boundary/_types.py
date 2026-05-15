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
    """Callable interface for the compiled spectrum kernel."""

    def __call__(self, potential: jax.Array) -> Spectrum:
        """Return the spectral decomposition for one assembled potential."""
        ...


class RMatrixObservable(Protocol):
    """Callable interface for the spectral R-matrix observable."""

    def __call__(self, spectrum: Spectrum, energy: EnergyLike) -> jax.Array:
        """Evaluate the R-matrix at one energy."""
        ...


class SpectrumObservable(Protocol):
    """Callable interface for observables evaluated from a Spectrum only."""

    def __call__(self, spectrum: Spectrum) -> jax.Array:
        """Evaluate an observable on the compile-time energy grid."""
        ...


class GreenFunctionObservable(Protocol):
    """Callable interface for Green's-function evaluation."""

    def __call__(self, spectrum: Spectrum, energy: EnergyLike) -> jax.Array:
        """Evaluate the Green's function at one energy."""
        ...


class WavefunctionObservable(Protocol):
    """Callable interface for internal wavefunction reconstruction."""

    def __call__(self, spectrum: Spectrum, energy: EnergyLike, source: jax.Array) -> jax.Array:
        """Evaluate the internal wavefunction at one energy."""
        ...


class EigenpairAccessor(Protocol):
    """Callable interface for raw eigensystem access."""

    def __call__(self, spectrum: Spectrum) -> tuple[jax.Array, jax.Array | None]:
        """Return the stored eigenvalues and optional eigenvectors."""
        ...


class DirectRMatrixKernel(Protocol):
    """Callable interface for the linear-solve fallback kernel."""

    def __call__(self, potential: jax.Array) -> jax.Array:
        """Evaluate the direct R-matrix on the compile-time energy grid."""
        ...


class GridVectorTransform(Protocol):
    """Callable interface for mesh-vector to radial-grid transforms."""

    def __call__(self, values: jax.Array) -> jax.Array:
        """Project mesh coefficients onto a radial grid."""
        ...


class FromGridVectorTransform(Protocol):
    """Callable interface for radial-grid to mesh-vector transforms."""

    def __call__(
        self,
        values: jax.Array | Callable[[jax.Array], jax.Array],
    ) -> jax.Array:
        """Project sampled radial-grid values or a callable profile onto the mesh basis."""
        ...


class GridMatrixTransform(Protocol):
    """Callable interface for mesh-matrix to radial-grid transforms."""

    def __call__(self, values: jax.Array) -> jax.Array:
        """Project a mesh-space kernel onto a radial grid."""
        ...


class FourierTransform(Protocol):
    """Callable interface for mesh-to-momentum transforms."""

    def __call__(self, values: jax.Array, channel_index: int = 0) -> jax.Array:
        """Project mesh coefficients or kernels onto a momentum grid."""
        ...


class Integrator(Protocol):
    """Callable interface for norms and expectation values."""

    def __call__(self, values: jax.Array, operator: jax.Array | None = None) -> jax.Array:
        """Integrate mesh coefficients with an optional operator insertion."""
        ...


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Mesh:
    """Concrete mesh data cached inside a compiled solver."""

    family: MeshFamily = field(metadata={"static": True})
    regularization: Regularization = field(metadata={"static": True})
    n: int = field(metadata={"static": True})
    scale: float = field(metadata={"static": True})
    nodes: jax.Array
    weights: jax.Array
    radii: jax.Array
    basis_at_boundary: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class OperatorMatrices:
    """Precomputed single-channel matrices in fm^-2 units."""

    T: jax.Array | None = None
    TpL: jax.Array | None = None
    T_alpha: jax.Array | None = None
    D: jax.Array | None = None
    inv_r: jax.Array | None = None
    inv_r2: jax.Array | None = None


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class BoundaryValues:
    """Boundary values at the channel radius for each compile-time energy."""

    H_plus: jax.Array
    H_minus: jax.Array
    H_plus_p: jax.Array
    H_minus_p: jax.Array
    is_open: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class TransformMatrices:
    """Optional precomputed matrices for grid and momentum transforms."""

    B_grid: jax.Array | None = None
    grid_r: jax.Array | None = None
    F_momentum: jax.Array | None = None
    momenta: jax.Array | None = None


@dataclass(frozen=True)
class Solver:
    """Compiled solver bundle containing caches and bound runtime callables."""

    mesh: Mesh
    operators: OperatorMatrices
    channels: tuple[ChannelSpec, ...]
    energies: jax.Array
    boundary: BoundaryValues | None
    transforms: TransformMatrices
    method: Method
    spectrum: SpectrumKernel | None = None
    rmatrix: RMatrixObservable | None = None
    smatrix: SpectrumObservable | None = None
    phases: SpectrumObservable | None = None
    greens: GreenFunctionObservable | None = None
    wavefunction: WavefunctionObservable | None = None
    eigh: EigenpairAccessor | None = None
    rmatrix_direct: DirectRMatrixKernel | None = None
    to_grid_vector: GridVectorTransform | None = None
    from_grid_vector: FromGridVectorTransform | None = None
    to_grid_matrix: GridMatrixTransform | None = None
    fourier: FourierTransform | None = None
    integrate: Integrator | None = None


__all__ = [
    "BoundaryValues",
    "DirectRMatrixKernel",
    "EigenpairAccessor",
    "EnergyLike",
    "FourierTransform",
    "FromGridVectorTransform",
    "GreenFunctionObservable",
    "GridMatrixTransform",
    "GridVectorTransform",
    "Integrator",
    "Mesh",
    "OperatorMatrices",
    "RMatrixObservable",
    "Solver",
    "SpectrumKernel",
    "SpectrumObservable",
    "TransformMatrices",
    "WavefunctionObservable",
]

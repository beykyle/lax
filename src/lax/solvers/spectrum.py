"""Spectrum-kernel construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np

from lax.boundary._types import Mesh, OperatorMatrices, SpectrumKernel
from lax.spectral.types import Spectrum
from lax.types import ChannelSpec, Method

from .assembly import assemble_block_hamiltonian, build_Q


@dataclass(frozen=True)
class _SpectrumKernel:
    """Pickle-safe bound spectrum kernel backed by module-level JIT functions."""

    mesh: Mesh
    operators: OperatorMatrices
    channels: tuple[ChannelSpec, ...]
    q: jax.Array
    method: Method
    keep_eigenvectors: bool

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
            Optional energy-dependent ℏ²/2μ value in MeV·fm².  When provided,
            overrides ``ChannelSpec.mass_factor`` in the Hamiltonian assembly so
            that ``threshold/μ(E)`` and ``V/μ(E)`` use the correct per-energy
            value.  Typical usage::

                spectra = jax.vmap(
                    lambda V, mu: solver.spectrum(V, mass_factor=mu)
                )(V_grid, mu_grid)

        Returns
        -------
        Spectrum
            Eigendecomposition of the Bloch-augmented Hamiltonian.
        """

        if self.method == "eigh":
            return cast(
                Spectrum,
                _SPECTRUM_EIGH_JIT(
                    potential,
                    self.mesh,
                    self.operators,
                    self.channels,
                    self.q,
                    self.keep_eigenvectors,
                    mass_factor,
                ),
            )
        if self.method == "eig":
            return cast(
                Spectrum,
                _SPECTRUM_EIG_JIT(
                    potential,
                    self.mesh,
                    self.operators,
                    self.channels,
                    self.q,
                    self.keep_eigenvectors,
                    mass_factor,
                ),
            )
        msg = f"Method {self.method!r} is not implemented in the MVP spectrum kernel."
        raise ValueError(msg)


def make_spectrum_kernel(
    mesh: Mesh,
    operators: OperatorMatrices,
    channels: tuple[ChannelSpec, ...],
    method: Method = "eigh",
    keep_eigenvectors: bool = False,
) -> SpectrumKernel:
    """Build a JIT-compiled ``spectrum(V) → Spectrum`` kernel.

    The returned kernel assembles the Bloch-augmented Hamiltonian from the
    supplied potential, eigendecomposes it once, and returns a :class:`Spectrum`
    holding eigenvalues, surface amplitudes, and (optionally) eigenvectors.
    One eigendecomposition supports all downstream observables at all energies.
    [DESIGN.md §11.1]

    Parameters
    ----------
    mesh
        Compiled mesh supplying operator matrices and boundary values.
    operators
        Precomputed single-channel operator matrices (``TpL`` required).
    channels
        Channel definitions baked into the solver.
    method
        Eigendecomposition backend: ``"eigh"`` for real/Hermitian potentials
        (GPU-ready), ``"eig"`` for complex potentials (CPU callback).
    keep_eigenvectors
        Whether to retain the full eigenvector matrix U in the returned
        :class:`Spectrum`.  Required for Green's functions and wavefunctions;
        saves memory and computation when only R/S/phases are needed.

    Returns
    -------
    SpectrumKernel
        JIT-compiled callable: ``kernel(V) → Spectrum``.
    """

    q = build_Q(mesh, channels)
    return _SpectrumKernel(
        mesh=mesh,
        operators=operators,
        channels=channels,
        q=q,
        method=method,
        keep_eigenvectors=keep_eigenvectors,
    )


def _spectrum_eigh(
    potential: jax.Array,
    mesh: Mesh,
    operators: OperatorMatrices,
    channels: tuple[ChannelSpec, ...],
    q: jax.Array,
    keep_eigenvectors: bool,
    mass_factor_override: float | jax.Array | None,
) -> Spectrum:
    """Return the Hermitian spectrum for one potential."""

    hamiltonian = assemble_block_hamiltonian(
        mesh, operators, channels, potential, mass_factor_override
    )
    eigensystem = cast(
        tuple[jax.Array, jax.Array],
        jnp.linalg.eigh(hamiltonian),
    )
    eigenvalues, eigenvectors = eigensystem
    surface_amplitudes: jax.Array = eigenvectors.T @ q
    return Spectrum(
        eigenvalues=eigenvalues,
        surface_amplitudes=surface_amplitudes,
        eigenvectors=eigenvectors if keep_eigenvectors else None,
        is_hermitian=True,
    )


def _spectrum_eig(
    potential: jax.Array,
    mesh: Mesh,
    operators: OperatorMatrices,
    channels: tuple[ChannelSpec, ...],
    q: jax.Array,
    keep_eigenvectors: bool,
    mass_factor_override: float | jax.Array | None,
) -> Spectrum:
    """Return the complex-symmetric spectrum for one potential."""

    hamiltonian = assemble_block_hamiltonian(
        mesh, operators, channels, potential, mass_factor_override
    )
    eigenvalues, eigenvectors = _eig_via_callback(hamiltonian)
    bilinear_norm = jnp.sqrt(jnp.diag(eigenvectors.T @ eigenvectors))
    eigenvectors_normalized = eigenvectors / bilinear_norm[None, :]
    surface_amplitudes: jax.Array = eigenvectors_normalized.T @ q
    return Spectrum(
        eigenvalues=eigenvalues,
        surface_amplitudes=surface_amplitudes,
        eigenvectors=eigenvectors_normalized if keep_eigenvectors else None,
        is_hermitian=False,
    )


_SPECTRUM_EIGH_JIT = jax.jit(
    _spectrum_eigh,
    static_argnames=("channels", "keep_eigenvectors"),
)
_SPECTRUM_EIG_JIT = jax.jit(
    _spectrum_eig,
    static_argnames=("channels", "keep_eigenvectors"),
)


def _eig_via_callback(hamiltonian: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Return a complex eigensystem via host NumPy for complex-symmetric problems."""

    matrix_size = hamiltonian.shape[0]
    result_shape = (
        jax.ShapeDtypeStruct((matrix_size,), jnp.complex128),
        jax.ShapeDtypeStruct((matrix_size, matrix_size), jnp.complex128),
    )

    def numpy_eig(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values, vectors = np.linalg.eig(matrix.astype(np.complex128))
        return values.astype(np.complex128), vectors.astype(np.complex128)

    complex_hamiltonian = hamiltonian.astype(jnp.complex128)
    eigensystem = jax.pure_callback(
        numpy_eig,
        result_shape,
        complex_hamiltonian,
    )
    return cast(tuple[jax.Array, jax.Array], eigensystem)


__all__ = ["make_spectrum_kernel"]

"""Spectrum-kernel construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np

from lax.spectral.types import Spectrum
from lax.types import ChannelSpec, Interaction, Mesh, Method, OperatorMatrices, SpectrumKernel

from .assembly import assemble_block_hamiltonian, build_Q, uniform_mass_factor


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
        potential: jax.Array | Interaction,
    ) -> Spectrum:
        """Return the spectral decomposition for one potential.

        Parameters
        ----------
        potential
            :class:`~lax.Interaction` object built by ``solver.local_potential()``/``solver.nonlocal_potential()`` or
            ``solver.interaction_from_{block,array,funcs}()``.  An
            energy-independent interaction yields one :class:`Spectrum`; an
            energy-dependent interaction (``energy_dependent=True``) is dispatched
            internally over its leading ``(N_E,)`` block axis and yields a batched
            :class:`Spectrum` — the caller need not ``jax.vmap`` by hand
            (§4.3/§11.1).

        Returns
        -------
        Spectrum
            Eigendecomposition of the Bloch-augmented Hamiltonian (batched over
            energy when ``potential.energy_dependent``).
        """
        if not isinstance(potential, Interaction):
            raise TypeError(
                "spectrum() accepts only Interaction objects. "
                "Use solver.local_potential()/solver.nonlocal_potential() or solver.interaction_from_block/array/funcs to build one."
            )
        if potential.energy_dependent:
            return jax.vmap(self._spectrum_one)(potential.block)
        return self._spectrum_one(potential.block)

    def _spectrum_one(self, block: jax.Array) -> Spectrum:
        """Eigendecompose one ``(M, M)`` assembled block via the chosen backend."""

        if self.method == "eigh":
            return cast(
                Spectrum,
                _SPECTRUM_EIGH_JIT(
                    block,
                    self.mesh,
                    self.operators,
                    self.channels,
                    self.q,
                    self.keep_eigenvectors,
                ),
            )
        if self.method == "eig":
            return cast(
                Spectrum,
                _SPECTRUM_EIG_JIT(
                    block,
                    self.mesh,
                    self.operators,
                    self.channels,
                    self.q,
                    self.keep_eigenvectors,
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

    # The eigh/eig kernels fold a single ℏ²/2μ out of the Hamiltonian
    # (H_MeV / m0); validate that assumption here so the kernel cannot be
    # built for a multi-μ channel set and silently produce wrong physics,
    # even when bound without the observable layer.
    uniform_mass_factor(channels, context="spectral eigensolve path")
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
) -> Spectrum:
    """Return the Hermitian spectrum for one potential."""

    H_MeV = assemble_block_hamiltonian(mesh, operators, channels, potential)
    m0 = channels[0].mass_factor
    hamiltonian = H_MeV / m0
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
) -> Spectrum:
    """Return the complex-symmetric spectrum for one potential."""

    H_MeV = assemble_block_hamiltonian(mesh, operators, channels, potential)
    m0 = channels[0].mass_factor
    hamiltonian = H_MeV / m0
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
        vmap_method="sequential",
    )
    return cast(tuple[jax.Array, jax.Array], eigensystem)


__all__ = ["make_spectrum_kernel"]

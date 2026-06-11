"""Spectrum-kernel construction.

There is one spectrum kernel: it is always batched over the leading symmetry-block
axis (DESIGN.md §15.5).  A solver compiled with ``channels=`` is simply the
``N_b == 1`` case — the kernel runs the same batched code and squeezes the block
axis off the returned :class:`Spectrum` so the single-block public contract is
unchanged.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np

from lax.spectral.types import Spectrum
from lax.types import (
    ChannelSpec,
    Interaction,
    Mesh,
    Method,
    OperatorMatrices,
    SpectrumKernel,
)

from .assembly import (
    assemble_hamiltonian_arrays,
    block_group_arrays,
    build_Q,
    lift_to_blocks,
    reject_block_dependent,
    take_block0,
    uniform_block_mass_factor,
)


@dataclass(frozen=True)
class _SpectrumKernel:
    """Pickle-safe spectrum kernel, batched over the symmetry-block axis.

    ``block_mode`` distinguishes the two compile modes: ``True`` for
    ``blocks=`` (outputs keep the leading ``(N_b,)`` axis and block-dependent
    Interactions are accepted), ``False`` for ``channels=`` (the ``N_b == 1``
    special case — the block axis is squeezed off the returned Spectrum).
    """

    mesh: Mesh
    operators: OperatorMatrices
    centrifugal: jax.Array  # (N_b, N_c)
    thresholds: jax.Array  # (N_b, N_c)
    q: jax.Array  # (N_c·N, N_c), shared across blocks
    mass_factor: float  # single ℏ²/2μ shared by all channels of all blocks
    method: Method
    keep_eigenvectors: bool
    n_blocks: int
    block_mode: bool

    def __call__(self, potential: jax.Array | Interaction) -> Spectrum:
        """Return the spectral decomposition(s) for one potential.

        Energy-dependent interactions add an ``(N_E,)`` axis after the block
        axis (block × energy order); block-independent interactions broadcast
        across the ``(N_b,)`` axis.
        """

        if not isinstance(potential, Interaction):
            raise TypeError(
                "spectrum() accepts only Interaction objects. "
                "Use solver.local_potential()/solver.nonlocal_potential() or solver.interaction_from_block/array/funcs to build one."
            )
        if not self.block_mode:
            reject_block_dependent(potential, "spectrum()")
        if self.method == "eigh":
            blocks_jit, grid_jit = (
                _SPECTRUM_EIGH_BLOCKS_JIT,
                _SPECTRUM_EIGH_BLOCKS_GRID_JIT,
            )
        elif self.method == "eig":
            blocks_jit, grid_jit = (
                _SPECTRUM_EIG_BLOCKS_JIT,
                _SPECTRUM_EIG_BLOCKS_GRID_JIT,
            )
        else:
            msg = f"Method {self.method!r} is not implemented in the MVP spectrum kernel."
            raise ValueError(msg)
        block = lift_to_blocks(potential.block, potential.block_dependent, self.n_blocks)
        jit_fn = grid_jit if potential.energy_dependent else blocks_jit
        spectrum = cast(
            Spectrum,
            jit_fn(
                block,
                self.mesh,
                self.operators,
                self.centrifugal,
                self.thresholds,
                self.q,
                self.mass_factor,
                self.keep_eigenvectors,
            ),
        )
        return spectrum if self.block_mode else take_block0(spectrum)


def make_spectrum_kernel(
    mesh: Mesh,
    operators: OperatorMatrices,
    blocks: tuple[tuple[ChannelSpec, ...], ...],
    method: Method = "eigh",
    keep_eigenvectors: bool = False,
    block_mode: bool = False,
) -> SpectrumKernel:
    """Build a JIT-compiled ``spectrum(V) → Spectrum`` kernel.

    The returned kernel assembles the Bloch-augmented Hamiltonian per symmetry
    block, eigendecomposes each block once, and returns a :class:`Spectrum`.
    One eigendecomposition supports all downstream observables at all energies.
    [DESIGN.md §11.1, §15.5]

    Parameters
    ----------
    mesh
        Compiled mesh supplying operator matrices and boundary values.
    operators
        Precomputed single-channel operator matrices (``TpL`` required).
    blocks
        ``N_b`` symmetry blocks of ``N_c`` channels each.  A ``channels=``
        compile passes the single block ``(channels,)``.
    method
        Eigendecomposition backend: ``"eigh"`` for real/Hermitian potentials
        (GPU-ready), ``"eig"`` for complex potentials (CPU callback).
    keep_eigenvectors
        Whether to retain the full eigenvector matrix U in the returned
        :class:`Spectrum`.  Required for Green's functions and wavefunctions;
        saves memory and computation when only R/S/phases are needed.
    block_mode
        ``True`` for a ``blocks=`` compile: outputs keep the leading
        ``(N_b,)`` axis and block-dependent Interactions are accepted.

    Returns
    -------
    SpectrumKernel
        JIT-compiled callable: ``kernel(V) → Spectrum``.

    Notes
    -----
    The spectral path folds a single ℏ²/2μ out of the Hamiltonian, so it
    requires one uniform mass factor across all channels of all blocks;
    per-block/per-channel μ remains a direct-path feature.
    """

    mass_factor = uniform_block_mass_factor(blocks, context="spectral eigensolve path")
    centrifugal, thresholds, _ = block_group_arrays(blocks)
    q = build_Q(mesh, blocks[0])
    return _SpectrumKernel(
        mesh=mesh,
        operators=operators,
        centrifugal=centrifugal,
        thresholds=thresholds,
        q=q,
        mass_factor=mass_factor,
        method=method,
        keep_eigenvectors=keep_eigenvectors,
        n_blocks=len(blocks),
        block_mode=block_mode,
    )


def _spectrum_eigh_core(
    potential: jax.Array,
    mesh: Mesh,
    operators: OperatorMatrices,
    centrifugal: jax.Array,
    thresholds: jax.Array,
    q: jax.Array,
    mass_factor: jax.Array | float,
    keep_eigenvectors: bool,
) -> Spectrum:
    """Return the Hermitian spectrum for one potential.

    Array-parameterized per-block core (DESIGN.md §15.5): the per-channel
    centrifugal/threshold data arrive as traced ``(N_c,)`` arrays so the body
    composes with ``jax.vmap`` over a leading symmetry-block axis.
    ``mass_factor`` is the single ℏ²/2μ shared by all channels.
    """

    mass_factors = jnp.broadcast_to(jnp.asarray(mass_factor), centrifugal.shape)
    H_MeV = assemble_hamiltonian_arrays(
        mesh, operators, centrifugal, thresholds, mass_factors, potential
    )
    hamiltonian = H_MeV / jnp.asarray(mass_factor, dtype=H_MeV.dtype)
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


def _spectrum_eig_core(
    potential: jax.Array,
    mesh: Mesh,
    operators: OperatorMatrices,
    centrifugal: jax.Array,
    thresholds: jax.Array,
    q: jax.Array,
    mass_factor: jax.Array | float,
    keep_eigenvectors: bool,
) -> Spectrum:
    """Return the complex-symmetric spectrum for one potential.

    Array-parameterized per-block core; see :func:`_spectrum_eigh_core`.
    """

    mass_factors = jnp.broadcast_to(jnp.asarray(mass_factor), centrifugal.shape)
    H_MeV = assemble_hamiltonian_arrays(
        mesh, operators, centrifugal, thresholds, mass_factors, potential
    )
    hamiltonian = H_MeV / jnp.asarray(mass_factor, dtype=H_MeV.dtype)
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


def _spectrum_blocks(
    core: Callable[..., Spectrum],
    blocks: jax.Array,
    mesh: Mesh,
    operators: OperatorMatrices,
    centrifugal: jax.Array,
    thresholds: jax.Array,
    q: jax.Array,
    mass_factor: float,
    keep_eigenvectors: bool,
) -> Spectrum:
    """``jax.vmap`` one spectrum core over the leading ``(N_b,)`` block axis."""

    def one_block(
        block: jax.Array,
        centrifugal_row: jax.Array,
        threshold_row: jax.Array,
    ) -> Spectrum:
        return core(
            block,
            mesh,
            operators,
            centrifugal_row,
            threshold_row,
            q,
            mass_factor,
            keep_eigenvectors,
        )

    return jax.vmap(one_block)(blocks, centrifugal, thresholds)


def _spectrum_blocks_grid(
    core: Callable[..., Spectrum],
    blocks: jax.Array,
    mesh: Mesh,
    operators: OperatorMatrices,
    centrifugal: jax.Array,
    thresholds: jax.Array,
    q: jax.Array,
    mass_factor: float,
    keep_eigenvectors: bool,
) -> Spectrum:
    """Nested block × energy vmap of one spectrum core.

    ``blocks`` has shape ``(N_b, N_E, M, M)``; the returned :class:`Spectrum`
    leaves carry ``(N_b, N_E, …)`` axes.  Under ``method="eig"`` the host
    callback (``vmap_method="sequential"``) runs one host ``eig`` per
    ``(b, i)`` pair — correct but sequential.
    """

    def one_block(
        block_grid: jax.Array,
        centrifugal_row: jax.Array,
        threshold_row: jax.Array,
    ) -> Spectrum:
        def one_energy(block: jax.Array) -> Spectrum:
            return core(
                block,
                mesh,
                operators,
                centrifugal_row,
                threshold_row,
                q,
                mass_factor,
                keep_eigenvectors,
            )

        return jax.vmap(one_energy)(block_grid)

    return jax.vmap(one_block)(blocks, centrifugal, thresholds)


_SPECTRUM_EIGH_BLOCKS_JIT = jax.jit(
    partial(_spectrum_blocks, _spectrum_eigh_core),
    static_argnames=("keep_eigenvectors",),
)
_SPECTRUM_EIGH_BLOCKS_GRID_JIT = jax.jit(
    partial(_spectrum_blocks_grid, _spectrum_eigh_core),
    static_argnames=("keep_eigenvectors",),
)
_SPECTRUM_EIG_BLOCKS_JIT = jax.jit(
    partial(_spectrum_blocks, _spectrum_eig_core),
    static_argnames=("keep_eigenvectors",),
)
_SPECTRUM_EIG_BLOCKS_GRID_JIT = jax.jit(
    partial(_spectrum_blocks_grid, _spectrum_eig_core),
    static_argnames=("keep_eigenvectors",),
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

"""Factories for building Interaction blocks from potential terms."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import jax
import jax.core
import jax.numpy as jnp
import numpy as np

from lax.types import ChannelSpec, Interaction, Mesh


def _is_concrete(value: jax.Array) -> bool:
    """Return False when ``value`` is a JAX tracer (inside jit/vmap/grad)."""

    return not isinstance(value, jax.core.Tracer)


@dataclass(frozen=True)
class _InteractionFromBlock:
    """Pickle-safe callable that wraps a pre-assembled block as an Interaction."""

    N: int
    N_c: int
    N_E: int

    def __call__(
        self,
        block: jax.Array,
        *,
        energy_dependent: bool = False,
    ) -> Interaction:
        """Wrap a pre-assembled ``(M, M)`` or ``(N_E, M, M)`` block."""

        M = self.N_c * self.N
        expected_shape = (self.N_E, M, M) if energy_dependent else (M, M)
        if block.shape != expected_shape:
            raise ValueError(f"Expected block shape {expected_shape}, got {block.shape}.")
        return Interaction(block=block, energy_dependent=energy_dependent)


@dataclass(frozen=True)
class _InteractionFromArray:
    """Pickle-safe callable that assembles an Interaction from (form-factor, coupling) terms."""

    N: int
    N_c: int
    N_E: int
    gauss_scale: jax.Array  # (N, N) = sqrt(λ_i λ_j) * a

    def _validate_A(self, A: jax.Array, label: str) -> None:
        if A.shape != (self.N_c, self.N_c):
            raise ValueError(
                f"{label}: coupling matrix A must be ({self.N_c},{self.N_c}), got {A.shape}."
            )
        if _is_concrete(A) and not jnp.allclose(A, A.T, atol=1e-12):
            raise ValueError(f"{label}: coupling matrix A must be symmetric.")

    def __call__(
        self,
        local: Sequence[tuple[jax.Array, jax.Array]] = (),
        nonlocal_: Sequence[tuple[jax.Array, jax.Array]] = (),
        energy_dependent: bool = False,
    ) -> Interaction:
        """Build Interaction from (form_factor, coupling_matrix) term lists.

        Each local term: (g, A) where g has shape (N,) or (N_E, N).
        Each nonlocal term: (g, A) where g has shape (N, N) or (N_E, N, N).
        A has shape (N_c, N_c) and must be symmetric.

        Local term contributes ``A[c,cp] * g_n`` to the (c,cp) diagonal block.
        Nonlocal term contributes ``A[c,cp] * sqrt(λ_i λ_j) * a * g[i,j]``
        to the full (c,cp) block.
        """
        N = self.N
        N_E = self.N_E
        M = self.N_c * N

        terms: list[jax.Array] = []
        for term_idx, (g_raw, a_raw) in enumerate(local):
            g = jnp.asarray(g_raw)
            a_mat = jnp.asarray(a_raw)
            self._validate_A(a_mat, f"local term {term_idx}")
            expected = (N_E, N) if energy_dependent else (N,)
            if g.shape != expected:
                raise ValueError(
                    f"local term {term_idx}: energy_dependent={energy_dependent} requires "
                    f"g shape {expected}, got {g.shape}."
                )
            kernel = jax.vmap(jnp.diag)(g) if energy_dependent else jnp.diag(g)
            terms.append(_coupling_kron(a_mat, kernel))

        for term_idx, (g_raw, a_raw) in enumerate(nonlocal_):
            g = jnp.asarray(g_raw)
            a_mat = jnp.asarray(a_raw)
            self._validate_A(a_mat, f"nonlocal term {term_idx}")
            expected = (N_E, N, N) if energy_dependent else (N, N)
            if g.shape != expected:
                raise ValueError(
                    f"nonlocal term {term_idx}: energy_dependent={energy_dependent} requires "
                    f"g shape {expected}, got {g.shape}."
                )
            terms.append(_coupling_kron(a_mat, g * self.gauss_scale))

        shape = (N_E, M, M) if energy_dependent else (M, M)
        dtype = jnp.result_type(*terms) if terms else jnp.float64
        block: jax.Array = jnp.zeros(shape, dtype=dtype)
        for term in terms:
            block = block + term

        # Value checks need concrete arrays: inside jit/vmap the block is a tracer
        # and boolean conversion would crash, so symmetry is validated only when the
        # Interaction is built outside jax transformations.
        first_block = block[0] if energy_dependent else block
        if _is_concrete(first_block) and not jnp.allclose(first_block, first_block.T, atol=1e-10):
            raise ValueError("Assembled Interaction block is not symmetric.")

        return Interaction(block=block, energy_dependent=energy_dependent)


def _coupling_kron(a_mat: jax.Array, kernel: jax.Array) -> jax.Array:
    """Return the coupled-channel block ``A ⊗ kernel``.

    ``kernel`` is ``(N, N)`` or, with a leading energy axis, ``(N_E, N, N)``;
    the result is ``(M, M)`` or ``(N_E, M, M)`` with ``M = N_c·N`` and
    block ``(c, cp)`` equal to ``A[c, cp] · kernel``.
    """

    n_c = a_mat.shape[0]
    n = kernel.shape[-1]
    if kernel.ndim == 3:
        batched: jax.Array = jnp.einsum("cd,eij->ecidj", a_mat, kernel)
        return batched.reshape(kernel.shape[0], n_c * n, n_c * n)
    flat: jax.Array = jnp.einsum("cd,ij->cidj", a_mat, kernel)
    return flat.reshape(n_c * n, n_c * n)


@dataclass(frozen=True)
class _InteractionFromFuncs:
    """Pickle-safe callable that evaluates potential functions then delegates to array builder."""

    N: int
    N_E: int
    radii: jax.Array  # (N,)
    energies: jax.Array  # (N_E,)
    array_builder: _InteractionFromArray

    def __call__(
        self,
        local: Sequence[tuple[Any, Any]] = (),
        nonlocal_: Sequence[tuple[Any, Any]] = (),
        energy_dependent: bool = False,
    ) -> Interaction:
        """Build Interaction from callable (form_factor_fn, coupling_matrix) terms.

        Local fn signatures:
            g(r)        — energy-independent; r shape (N,), returns (N,)
            g(r, E)     — energy-dependent;  r shape (N,), E scalar, returns (N,)
        Nonlocal fn signatures:
            g(r, r')    — energy-independent; r,r' shape (N,N), returns (N,N)
            g(r, r', E) — energy-dependent;  r,r' shape (N,N), E scalar, returns (N,N)
        """
        r = self.radii
        ri, rj = jnp.meshgrid(r, r, indexing="ij")  # (N, N)
        # One device→host transfer up front instead of one per energy sample.
        energy_values = [float(energy) for energy in np.asarray(self.energies)]

        local_arrays: list[Any] = []
        for g_fn, a_mat in local:
            if energy_dependent:
                g_arr = jnp.stack([g_fn(r, energy) for energy in energy_values])  # (N_E, N)
            else:
                g_arr = g_fn(r)  # (N,)
            local_arrays.append((g_arr, a_mat))

        nonlocal_arrays: list[Any] = []
        for g_fn, a_mat in nonlocal_:
            if energy_dependent:
                g_arr = jnp.stack([g_fn(ri, rj, energy) for energy in energy_values])  # (N_E, N, N)
            else:
                g_arr = g_fn(ri, rj)  # (N, N)
            nonlocal_arrays.append((g_arr, a_mat))

        return self.array_builder(
            local=local_arrays,
            nonlocal_=nonlocal_arrays,
            energy_dependent=energy_dependent,
        )


def make_interaction_from_block(
    mesh: Mesh,
    channels: tuple[ChannelSpec, ...],
    energies: jax.Array,
) -> _InteractionFromBlock:
    """Return a pickle-safe callable that wraps a pre-assembled block as an Interaction."""

    N = mesh.n
    N_c = len(channels)
    N_E = len(energies)
    return _InteractionFromBlock(N=N, N_c=N_c, N_E=N_E)


def make_interaction_from_array(
    mesh: Mesh,
    channels: tuple[ChannelSpec, ...],
    energies: jax.Array,
) -> _InteractionFromArray:
    """Return a pickle-safe callable that assembles an Interaction from (g, A) term lists."""

    N = mesh.n
    N_c = len(channels)
    N_E = len(energies)
    lam = mesh.weights  # (N,)
    a = float(mesh.scale)
    lami, lamj = jnp.meshgrid(lam, lam, indexing="ij")
    gauss_scale = jnp.sqrt(lami * lamj) * a  # (N, N)
    return _InteractionFromArray(N=N, N_c=N_c, N_E=N_E, gauss_scale=gauss_scale)


def make_interaction_from_funcs(
    mesh: Mesh,
    channels: tuple[ChannelSpec, ...],
    energies: jax.Array,
) -> _InteractionFromFuncs:
    """Return a pickle-safe callable that evaluates functions then builds an Interaction."""

    array_builder = make_interaction_from_array(mesh, channels, energies)
    return _InteractionFromFuncs(
        N=mesh.n,
        N_E=len(energies),
        radii=mesh.radii,
        energies=energies,
        array_builder=array_builder,
    )


@dataclass(frozen=True)
class _PotentialBuilder:
    """Pickle-safe single-kind ``solver.{local,nonlocal}_potential`` callable.

    Wraps ``_InteractionFromFuncs`` for one fixed ``kind`` (``"local"`` or
    ``"nonlocal"``), with a sensible ``coupling`` default for single-channel
    problems.  The local/nonlocal choice is made explicit by the entry point the
    caller picks, so there is no arity inference.
    """

    n_c: int
    funcs_builder: _InteractionFromFuncs
    kind: Literal["local", "nonlocal"]

    def __call__(
        self,
        fn: object,
        *,
        coupling: np.ndarray | None = None,
        energy_dependent: bool = False,
    ) -> Interaction:
        """Build an :class:`~lax.Interaction` from a potential function.

        Parameters
        ----------
        fn
            Potential function for this builder's fixed ``kind``:

            * local  — ``fn(r)`` or, with ``energy_dependent=True``, ``fn(r, E)``
            * nonlocal — ``fn(r, r')`` or ``fn(r, r', E)``

        coupling
            ``(N_c, N_c)`` symmetric coupling matrix.  Defaults to ``[[1.0]]``
            when ``N_c == 1``.  Required for multi-channel solvers.
        energy_dependent
            Whether ``fn`` takes an energy argument and the result should carry
            a leading ``(N_E,)`` axis.
        """
        if coupling is None:
            if self.n_c != 1:
                raise ValueError(
                    f"coupling is required for {self.n_c}-channel solvers; "
                    "pass coupling=np.array([[...]])."
                )
            coupling = np.ones((1, 1), dtype=np.float64)

        if energy_dependent and self.funcs_builder.N_E == 0:
            raise ValueError(
                "energy_dependent=True requires an energy grid; "
                "re-compile with energies=... to use energy-dependent potentials."
            )

        if self.kind == "local":
            return self.funcs_builder(local=[(fn, coupling)], energy_dependent=energy_dependent)  # type: ignore[list-item]
        return self.funcs_builder(nonlocal_=[(fn, coupling)], energy_dependent=energy_dependent)  # type: ignore[list-item]


def make_potential_builders(
    mesh: Mesh,
    channels: tuple[ChannelSpec, ...],
    energies: jax.Array,
) -> tuple[_PotentialBuilder, _PotentialBuilder]:
    """Return ``(local_potential, nonlocal_potential)`` builders for one solver.

    Each builds an :class:`~lax.Interaction` from a function of the corresponding
    kind, defaulting ``coupling`` to ``[[1.0]]`` for single-channel problems.  Both
    share one underlying ``_InteractionFromFuncs``.  See :class:`_PotentialBuilder`.
    """
    funcs_builder = make_interaction_from_funcs(mesh, channels, energies)
    n_c = len(channels)
    local = _PotentialBuilder(n_c=n_c, funcs_builder=funcs_builder, kind="local")
    nonlocal_ = _PotentialBuilder(n_c=n_c, funcs_builder=funcs_builder, kind="nonlocal")
    return local, nonlocal_


__all__ = [
    "make_interaction_from_array",
    "make_interaction_from_block",
    "make_interaction_from_funcs",
    "make_potential_builders",
]

"""Factories for building Interaction blocks from potential terms."""

from __future__ import annotations

import inspect
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from lax.boundary._types import Mesh
from lax.types import ChannelSpec, Interaction


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
        if not jnp.allclose(A, A.T, atol=1e-12):
            raise ValueError(f"{label}: coupling matrix A must be symmetric.")

    def __call__(
        self,
        local: list[tuple[jax.Array, jax.Array]] = (),
        nonlocal_: list[tuple[jax.Array, jax.Array]] = (),
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
        N_c = self.N_c
        N_E = self.N_E
        M = N_c * N
        dtype = jnp.float64
        all_terms = list(local) + list(nonlocal_)
        if all_terms:
            g0, _ = all_terms[0]
            dtype = jnp.asarray(g0).dtype

        if energy_dependent:
            block = jnp.zeros((N_E, M, M), dtype=dtype)
        else:
            block = jnp.zeros((M, M), dtype=dtype)

        for term_idx, (g, A) in enumerate(local):
            g = jnp.asarray(g)
            A = jnp.asarray(A)
            self._validate_A(A, f"local term {term_idx}")
            if energy_dependent:
                if g.ndim != 2 or g.shape != (N_E, N):
                    raise ValueError(
                        f"local term {term_idx}: energy_dependent=True requires g shape "
                        f"({N_E}, {N}), got {g.shape}."
                    )
                for c in range(N_c):
                    for cp in range(N_c):
                        if A[c, cp] == 0:
                            continue
                        row_start = c * N
                        col_start = cp * N
                        diag_blocks = jax.vmap(jnp.diag)(A[c, cp] * g)  # (N_E, N, N)
                        block = block.at[
                            :, row_start : row_start + N, col_start : col_start + N
                        ].add(diag_blocks)
            else:
                if g.ndim != 1 or g.shape != (N,):
                    raise ValueError(
                        f"local term {term_idx}: energy_dependent=False requires g shape "
                        f"({N},), got {g.shape}."
                    )
                for c in range(N_c):
                    for cp in range(N_c):
                        if A[c, cp] == 0:
                            continue
                        row_start = c * N
                        col_start = cp * N
                        block = block.at[row_start : row_start + N, col_start : col_start + N].add(
                            jnp.diag(A[c, cp] * g)
                        )

        for term_idx, (g, A) in enumerate(nonlocal_):
            g = jnp.asarray(g)
            A = jnp.asarray(A)
            self._validate_A(A, f"nonlocal term {term_idx}")
            if energy_dependent:
                if g.ndim != 3 or g.shape != (N_E, N, N):
                    raise ValueError(
                        f"nonlocal term {term_idx}: energy_dependent=True requires g shape "
                        f"({N_E}, {N}, {N}), got {g.shape}."
                    )
                scaled = g * self.gauss_scale[None, :, :]  # (N_E, N, N)
                for c in range(N_c):
                    for cp in range(N_c):
                        if A[c, cp] == 0:
                            continue
                        row_start = c * N
                        col_start = cp * N
                        block = block.at[
                            :, row_start : row_start + N, col_start : col_start + N
                        ].add(A[c, cp] * scaled)
            else:
                if g.ndim != 2 or g.shape != (N, N):
                    raise ValueError(
                        f"nonlocal term {term_idx}: energy_dependent=False requires g shape "
                        f"({N}, {N}), got {g.shape}."
                    )
                scaled = g * self.gauss_scale  # (N, N)
                for c in range(N_c):
                    for cp in range(N_c):
                        if A[c, cp] == 0:
                            continue
                        row_start = c * N
                        col_start = cp * N
                        block = block.at[row_start : row_start + N, col_start : col_start + N].add(
                            A[c, cp] * scaled
                        )

        first_block = block[0] if energy_dependent else block
        if not jnp.allclose(first_block, first_block.T, atol=1e-10):
            raise ValueError("Assembled Interaction block is not symmetric.")

        return Interaction(block=block, energy_dependent=energy_dependent)


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
        local: list = (),
        nonlocal_: list = (),
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
        N_E = self.N_E
        ri, rj = jnp.meshgrid(r, r, indexing="ij")  # (N, N)

        local_arrays = []
        for g_fn, A in local:
            if energy_dependent:
                g_arr = jnp.stack(
                    [g_fn(r, float(self.energies[ie])) for ie in range(N_E)]
                )  # (N_E, N)
            else:
                g_arr = g_fn(r)  # (N,)
            local_arrays.append((g_arr, A))

        nonlocal_arrays = []
        for g_fn, A in nonlocal_:
            if energy_dependent:
                g_arr = jnp.stack(
                    [g_fn(ri, rj, float(self.energies[ie])) for ie in range(N_E)]
                )  # (N_E, N, N)
            else:
                g_arr = g_fn(ri, rj)  # (N, N)
            nonlocal_arrays.append((g_arr, A))

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
    """Pickle-safe ``solver.potential(fn, coupling, energy_dependent)`` callable.

    Wraps ``_InteractionFromFuncs`` with arity-based local/nonlocal dispatch and
    a sensible ``coupling`` default for single-channel problems.
    """

    n_c: int
    funcs_builder: _InteractionFromFuncs

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
            Potential function.  Arity determines local vs nonlocal:

            * ``energy_dependent=False``: ``fn(r)`` → local, ``fn(r, r')`` → nonlocal
            * ``energy_dependent=True``:  ``fn(r, E)`` → local, ``fn(r, r', E)`` → nonlocal

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

        n_args = len(inspect.signature(fn).parameters)  # type: ignore[arg-type]
        if energy_dependent:
            if n_args == 2:
                return self.funcs_builder(local=[(fn, coupling)], energy_dependent=True)  # type: ignore[list-item]
            elif n_args == 3:
                return self.funcs_builder(nonlocal_=[(fn, coupling)], energy_dependent=True)  # type: ignore[list-item]
            else:
                raise ValueError(
                    f"For energy_dependent=True, fn must take 2 args fn(r, E) "
                    f"(local) or 3 args fn(r, r', E) (nonlocal); got {n_args}."
                )
        else:
            if n_args == 1:
                return self.funcs_builder(local=[(fn, coupling)], energy_dependent=False)  # type: ignore[list-item]
            elif n_args == 2:
                return self.funcs_builder(nonlocal_=[(fn, coupling)], energy_dependent=False)  # type: ignore[list-item]
            else:
                raise ValueError(
                    f"fn must take 1 arg fn(r) (local) or 2 args fn(r, r') "
                    f"(nonlocal); got {n_args}."
                )


def make_potential_builder(
    mesh: Mesh,
    channels: tuple[ChannelSpec, ...],
    energies: jax.Array,
) -> _PotentialBuilder:
    """Return ``solver.potential`` — a simple callable that builds an Interaction from a function.

    The returned callable auto-detects local vs nonlocal from function arity and
    defaults ``coupling`` to ``[[1.0]]`` for single-channel problems.  See
    :class:`_PotentialBuilder` for the full signature.
    """
    funcs_builder = make_interaction_from_funcs(mesh, channels, energies)
    return _PotentialBuilder(n_c=len(channels), funcs_builder=funcs_builder)


__all__ = [
    "make_interaction_from_array",
    "make_interaction_from_block",
    "make_interaction_from_funcs",
    "make_potential_builder",
]

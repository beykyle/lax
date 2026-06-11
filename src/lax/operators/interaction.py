"""Factories for building Interaction blocks from potential terms."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

import jax
import jax.core
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from lax.types import ChannelSpec, Interaction, Mesh


def _is_concrete(value: jax.Array) -> bool:
    """Return False when ``value`` is a JAX tracer (inside jit/vmap/grad)."""

    return not isinstance(value, jax.core.Tracer)


def _default_coupling(coupling: Any, n_c: int, context: str) -> Any:
    """Return the coupling matrix, defaulting to ``[[1.0]]`` for single-channel builds.

    Omitting the coupling is single-channel sugar: for ``n_c > 1`` an explicit
    symmetric ``(N_c, N_c)`` matrix is required (no silent ``eye(N_c)``).
    """

    if coupling is not None:
        return coupling
    if n_c != 1:
        raise ValueError(
            f"{context}: coupling is required for {n_c}-channel solvers; "
            "pass coupling=np.array([[...]])."
        )
    return np.ones((1, 1), dtype=np.float64)


def _normalize_term(term: Any, n_c: int, label: str) -> tuple[Any, Any]:
    """Normalize one term to the canonical ``(form_factor, coupling)`` pair.

    A term is either a bare form factor (single-channel sugar — the coupling
    defaults to ``[[1.0]]``) or an explicit ``(g, A)`` tuple.  Tuples are
    reserved for the explicit pair, so a bare block-dependent funcs term must
    be a *non-tuple* sequence of callables (e.g. a list).
    """

    if isinstance(term, tuple):
        pair = cast("tuple[Any, ...]", term)
        if len(pair) != 2:
            raise ValueError(
                f"{label}: a term tuple must be (form_factor, coupling); got a "
                f"tuple of length {len(pair)}."
            )
        return pair[0], pair[1]
    return term, _default_coupling(None, n_c, label)


def _require_blocks_mode(n_blocks: int | None, builder_name: str) -> int:
    """Return N_b, or raise when the solver was not compiled with ``blocks=``."""

    if n_blocks is None:
        raise TypeError(
            f"{builder_name}: block_dependent=True requires a solver compiled with "
            "blocks=[[ChannelSpec, ...], ...]; this solver was compiled with channels=."
        )
    return n_blocks


def _leading_axes(
    n_blocks: int | None,
    n_energies: int,
    energy_dependent: bool,
    block_dependent: bool,
    builder_name: str,
) -> tuple[int, ...]:
    """Return the expected leading axes ``([N_b,][N_E,])`` for one build."""

    axes: tuple[int, ...] = ()
    if block_dependent:
        axes = (_require_blocks_mode(n_blocks, builder_name),)
    if energy_dependent:
        axes = (*axes, n_energies)
    return axes


@dataclass(frozen=True)
class _InteractionFromBlock:
    """Pickle-safe callable that wraps a pre-assembled block as an Interaction."""

    N: int
    N_c: int
    N_E: int
    N_b: int | None = None

    def __call__(
        self,
        block: jax.Array,
        *,
        energy_dependent: bool = False,
        block_dependent: bool = False,
    ) -> Interaction:
        """Wrap a pre-assembled ``([N_b,][N_E,] M, M)`` block."""

        M = self.N_c * self.N
        leading = _leading_axes(
            self.N_b, self.N_E, energy_dependent, block_dependent, "interaction_from_block"
        )
        expected_shape = (*leading, M, M)
        if block.shape != expected_shape:
            raise ValueError(f"Expected block shape {expected_shape}, got {block.shape}.")
        return Interaction(
            block=block,
            energy_dependent=energy_dependent,
            block_dependent=block_dependent,
        )


@dataclass(frozen=True)
class _InteractionFromArray:
    """Pickle-safe callable that assembles an Interaction from (form-factor, coupling) terms."""

    N: int
    N_c: int
    N_E: int
    gauss_scale: jax.Array  # (N, N) = sqrt(λ_i λ_j) * a
    N_b: int | None = None

    def _validate_A(self, A: jax.Array, label: str) -> None:
        if A.shape != (self.N_c, self.N_c):
            raise ValueError(
                f"{label}: coupling matrix A must be ({self.N_c},{self.N_c}), got {A.shape}."
            )
        if _is_concrete(A) and not jnp.allclose(A, A.T, atol=1e-12):
            raise ValueError(f"{label}: coupling matrix A must be symmetric.")

    def __call__(
        self,
        local: Sequence[ArrayLike | tuple[ArrayLike, ArrayLike]] = (),
        nonlocal_: Sequence[ArrayLike | tuple[ArrayLike, ArrayLike]] = (),
        energy_dependent: bool = False,
        block_dependent: bool = False,
    ) -> Interaction:
        """Build Interaction from (form_factor, coupling_matrix) term lists.

        Each term is either a bare form factor ``g`` (single-channel sugar —
        coupling defaults to ``[[1.0]]``; raises for ``N_c > 1``) or an
        explicit ``(g, A)`` tuple.
        Each local ``g`` has shape ([N_b,][N_E,] N).
        Each nonlocal ``g`` has shape ([N_b,][N_E,] N, N).
        A has shape (N_c, N_c) and must be symmetric.  The leading axes are
        selected by ``block_dependent`` / ``energy_dependent`` in the canonical
        block × energy order (DESIGN.md §15.5).

        Local term contributes ``A[c,cp] * g_n`` to the (c,cp) diagonal block.
        Nonlocal term contributes ``A[c,cp] * sqrt(λ_i λ_j) * a * g[i,j]``
        to the full (c,cp) block.
        """
        N = self.N
        M = self.N_c * N
        leading = _leading_axes(
            self.N_b, self.N_E, energy_dependent, block_dependent, "interaction_from_array"
        )
        flags = f"block_dependent={block_dependent}, energy_dependent={energy_dependent}"

        terms: list[jax.Array] = []
        for term_idx, term in enumerate(local):
            g_raw, a_raw = _normalize_term(term, self.N_c, f"local term {term_idx}")
            g = jnp.asarray(g_raw)
            a_mat = jnp.asarray(a_raw)
            self._validate_A(a_mat, f"local term {term_idx}")
            expected = (*leading, N)
            if g.shape != expected:
                raise ValueError(
                    f"local term {term_idx}: {flags} requires g shape {expected}, got {g.shape}."
                )
            # Diagonal kernel with arbitrary leading axes: (..., N) -> (..., N, N).
            kernel = jnp.einsum("...i,ij->...ij", g, jnp.eye(N, dtype=g.dtype))
            terms.append(_coupling_kron(a_mat, kernel))

        for term_idx, term in enumerate(nonlocal_):
            g_raw, a_raw = _normalize_term(term, self.N_c, f"nonlocal term {term_idx}")
            g = jnp.asarray(g_raw)
            a_mat = jnp.asarray(a_raw)
            self._validate_A(a_mat, f"nonlocal term {term_idx}")
            expected = (*leading, N, N)
            if g.shape != expected:
                raise ValueError(
                    f"nonlocal term {term_idx}: {flags} requires g shape {expected}, got {g.shape}."
                )
            terms.append(_coupling_kron(a_mat, g * self.gauss_scale))

        shape = (*leading, M, M)
        dtype = jnp.result_type(*terms) if terms else jnp.float64
        block: jax.Array = jnp.zeros(shape, dtype=dtype)
        for term in terms:
            block = block + term

        # Value checks need concrete arrays: inside jit/vmap the block is a tracer
        # and boolean conversion would crash, so symmetry is validated only when the
        # Interaction is built outside jax transformations.
        first_block = block[(0,) * len(leading)]
        if _is_concrete(first_block) and not jnp.allclose(first_block, first_block.T, atol=1e-10):
            raise ValueError("Assembled Interaction block is not symmetric.")

        return Interaction(
            block=block,
            energy_dependent=energy_dependent,
            block_dependent=block_dependent,
        )


def _coupling_kron(a_mat: jax.Array, kernel: jax.Array) -> jax.Array:
    """Return the coupled-channel block ``A ⊗ kernel``.

    ``kernel`` is ``(N, N)`` with any leading batch axes (energy and/or
    symmetry block); the result is ``(..., M, M)`` with ``M = N_c·N`` and
    block ``(c, cp)`` equal to ``A[c, cp] · kernel``.
    """

    n_c = a_mat.shape[0]
    n = kernel.shape[-1]
    if jnp.issubdtype(kernel.dtype, jnp.inexact):
        # The structural coupling follows the form-factor precision so a
        # float32 compile (§14.1) stays float32; real-valued, so safe for
        # complex kernels too.
        a_mat = a_mat.astype(cast("jnp.dtype[Any]", jnp.finfo(kernel.dtype).dtype))
    batched: jax.Array = jnp.einsum("cd,...ij->...cidj", a_mat, kernel)
    return batched.reshape(*kernel.shape[:-2], n_c * n, n_c * n)


@dataclass(frozen=True)
class _InteractionFromFuncs:
    """Pickle-safe callable that evaluates potential functions then delegates to array builder."""

    N: int
    N_E: int
    radii: jax.Array  # (N,)
    energies: jax.Array  # (N_E,)
    array_builder: _InteractionFromArray
    N_b: int | None = None

    def __call__(
        self,
        local: Sequence[Any] = (),
        nonlocal_: Sequence[Any] = (),
        energy_dependent: bool = False,
        block_dependent: bool = False,
    ) -> Interaction:
        """Build Interaction from callable (form_factor_fn, coupling_matrix) terms.

        Each term is either a bare form-factor callable (single-channel sugar —
        coupling defaults to ``[[1.0]]``; raises for ``N_c > 1``) or an explicit
        ``(fn, coupling)`` tuple.

        Local fn signatures:
            g(r)        — energy-independent; r shape (N,), returns (N,)
            g(r, E)     — energy-dependent;  r shape (N,), E scalar, returns (N,)
        Nonlocal fn signatures:
            g(r, r')    — energy-independent; r,r' shape (N,N), returns (N,N)
            g(r, r', E) — energy-dependent;  r,r' shape (N,N), E scalar, returns (N,N)

        With ``block_dependent=True`` each term supplies a *sequence of N_b
        callables* — one per symmetry block, each with the signature above —
        whose evaluations are stacked on the leading (N_b,) axis (§15.5).
        A bare block-dependent term must be a non-tuple sequence (e.g. a list);
        tuples are reserved for the explicit ``(fns, coupling)`` pair.
        """
        # One device→host transfer up front instead of one per term/block/sample.
        energy_values = (
            [float(energy) for energy in np.asarray(self.energies)] if energy_dependent else []
        )

        local_arrays: list[Any] = []
        for term_idx, term in enumerate(local):
            label = f"local term {term_idx}"
            g_fn, a_mat = _normalize_term(term, self.array_builder.N_c, label)
            g_arr = self._evaluate_term(
                g_fn,
                kind="local",
                block_dependent=block_dependent,
                energy_values=energy_values,
                energy_dependent=energy_dependent,
                label=label,
            )
            local_arrays.append((g_arr, a_mat))

        nonlocal_arrays: list[Any] = []
        for term_idx, term in enumerate(nonlocal_):
            label = f"nonlocal term {term_idx}"
            g_fn, a_mat = _normalize_term(term, self.array_builder.N_c, label)
            g_arr = self._evaluate_term(
                g_fn,
                kind="nonlocal",
                block_dependent=block_dependent,
                energy_values=energy_values,
                energy_dependent=energy_dependent,
                label=label,
            )
            nonlocal_arrays.append((g_arr, a_mat))

        return self.array_builder(
            local=local_arrays,
            nonlocal_=nonlocal_arrays,
            energy_dependent=energy_dependent,
            block_dependent=block_dependent,
        )

    def _evaluate_term(
        self,
        g_fn: Any,
        *,
        kind: str,
        block_dependent: bool,
        energy_values: list[float],
        energy_dependent: bool,
        label: str,
    ) -> jax.Array:
        """Evaluate one term's callable(s) on the mesh (and energy/block axes)."""

        if block_dependent:
            n_b = _require_blocks_mode(self.N_b, "interaction_from_funcs")
            if callable(g_fn):
                raise TypeError(
                    f"{label}: block_dependent=True requires a sequence of {n_b} "
                    "callables (one per symmetry block), got a single callable."
                )
            fns = tuple(g_fn)
            if len(fns) != n_b:
                raise ValueError(
                    f"{label}: expected one callable per symmetry block "
                    f"(N_b={n_b}), got {len(fns)}."
                )
            return jnp.stack(
                [self._evaluate_one(fn, kind, energy_values, energy_dependent) for fn in fns]
            )
        if not callable(g_fn):
            raise TypeError(
                f"{label}: expected a callable form factor; sequences of "
                "callables require block_dependent=True."
            )
        return self._evaluate_one(g_fn, kind, energy_values, energy_dependent)

    def _evaluate_one(
        self,
        g_fn: Any,
        kind: str,
        energy_values: list[float],
        energy_dependent: bool,
    ) -> jax.Array:
        """Evaluate one callable over the mesh (and the energy grid if requested)."""

        r = self.radii
        if kind == "local":
            if energy_dependent:
                return jnp.stack([g_fn(r, energy) for energy in energy_values])  # (N_E, N)
            return cast(jax.Array, g_fn(r))  # (N,)
        ri, rj = jnp.meshgrid(r, r, indexing="ij")  # (N, N)
        if energy_dependent:
            return jnp.stack([g_fn(ri, rj, energy) for energy in energy_values])  # (N_E, N, N)
        return cast(jax.Array, g_fn(ri, rj))  # (N, N)


def make_interaction_from_block(
    mesh: Mesh,
    channels: tuple[ChannelSpec, ...],
    energies: jax.Array,
    n_blocks: int | None = None,
) -> _InteractionFromBlock:
    """Return a pickle-safe callable that wraps a pre-assembled block as an Interaction."""

    N = mesh.n
    N_c = len(channels)
    N_E = len(energies)
    return _InteractionFromBlock(N=N, N_c=N_c, N_E=N_E, N_b=n_blocks)


def make_interaction_from_array(
    mesh: Mesh,
    channels: tuple[ChannelSpec, ...],
    energies: jax.Array,
    n_blocks: int | None = None,
) -> _InteractionFromArray:
    """Return a pickle-safe callable that assembles an Interaction from (g, A) term lists."""

    N = mesh.n
    N_c = len(channels)
    N_E = len(energies)
    lam = mesh.weights  # (N,)
    a = float(mesh.scale)
    lami, lamj = jnp.meshgrid(lam, lam, indexing="ij")
    gauss_scale = jnp.sqrt(lami * lamj) * a  # (N, N)
    return _InteractionFromArray(N=N, N_c=N_c, N_E=N_E, gauss_scale=gauss_scale, N_b=n_blocks)


def make_interaction_from_funcs(
    mesh: Mesh,
    channels: tuple[ChannelSpec, ...],
    energies: jax.Array,
    n_blocks: int | None = None,
) -> _InteractionFromFuncs:
    """Return a pickle-safe callable that evaluates functions then builds an Interaction."""

    array_builder = make_interaction_from_array(mesh, channels, energies, n_blocks=n_blocks)
    return _InteractionFromFuncs(
        N=mesh.n,
        N_E=len(energies),
        radii=mesh.radii,
        energies=energies,
        array_builder=array_builder,
        N_b=n_blocks,
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
        block_dependent: bool = False,
    ) -> Interaction:
        """Build an :class:`~lax.Interaction` from a potential function.

        Parameters
        ----------
        fn
            Potential function for this builder's fixed ``kind``:

            * local  — ``fn(r)`` or, with ``energy_dependent=True``, ``fn(r, E)``
            * nonlocal — ``fn(r, r')`` or ``fn(r, r', E)``

            With ``block_dependent=True``, a *sequence of N_b such callables*,
            one per symmetry block (§15.5).
        coupling
            ``(N_c, N_c)`` symmetric coupling matrix.  Defaults to ``[[1.0]]``
            when ``N_c == 1``.  Required for multi-channel solvers.
        energy_dependent
            Whether ``fn`` takes an energy argument and the result should carry
            a leading ``(N_E,)`` axis.
        block_dependent
            Whether the term varies per symmetry block and the result should
            carry a leading ``(N_b,)`` axis.  Requires a solver compiled with
            ``blocks=``.
        """
        coupling = _default_coupling(coupling, self.n_c, f"{self.kind}_potential")

        if energy_dependent and self.funcs_builder.N_E == 0:
            raise ValueError(
                "energy_dependent=True requires an energy grid; "
                "re-compile with energies=... to use energy-dependent potentials."
            )

        if self.kind == "local":
            return self.funcs_builder(
                local=[(fn, coupling)],
                energy_dependent=energy_dependent,
                block_dependent=block_dependent,
            )
        return self.funcs_builder(
            nonlocal_=[(fn, coupling)],
            energy_dependent=energy_dependent,
            block_dependent=block_dependent,
        )


def make_potential_builders(
    mesh: Mesh,
    channels: tuple[ChannelSpec, ...],
    energies: jax.Array,
    n_blocks: int | None = None,
) -> tuple[_PotentialBuilder, _PotentialBuilder]:
    """Return ``(local_potential, nonlocal_potential)`` builders for one solver.

    Each builds an :class:`~lax.Interaction` from a function of the corresponding
    kind, defaulting ``coupling`` to ``[[1.0]]`` for single-channel problems.  Both
    share one underlying ``_InteractionFromFuncs``.  See :class:`_PotentialBuilder`.
    """
    funcs_builder = make_interaction_from_funcs(mesh, channels, energies, n_blocks=n_blocks)
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

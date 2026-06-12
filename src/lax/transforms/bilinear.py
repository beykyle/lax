"""Two-state bilinear matrix elements in the mesh basis.

``matrix_element`` evaluates ⟨bra|O|ket⟩-style contractions between two
*different* mesh-coefficient vectors, optionally without complex conjugation
(the bilinear form natural to complex-symmetric spectra), batched over the
symmetry-block and energy axes.  It complements :mod:`lax.transforms.integration`,
which remains the single-state, always-conjugating "norms and expectation
values" helper.

**Operator scaling contract** (the most important fact in this module): in the
Lagrange-mesh Gauss approximation the mesh coefficients already absorb
``√(λᵢ a)``, so

* a *local* operator enters as the **unscaled** node values ``V(rᵢ)`` —
  shape ``(M,)`` — and the element is the plain node sum
  ``Σᵢ braᵢ · V(rᵢ) · ketᵢ``;
* a *non-local* kernel supplied as a bare ``(M, M)`` array must be
  **Gauss-scaled** by the caller, ``K̃ᵢⱼ = √(λᵢ λⱼ)·a·K(rᵢ, rⱼ)`` — never the
  raw kernel values;
* an :class:`~lax.Interaction` is the recommended operator form: its assembled
  ``block`` already carries exactly this scaling for both local-diagonal and
  non-local content, so no caller-side weighting can go wrong.

``matrix_element`` adds **no** factors of ``a``, ``k``, or quadrature weights
beyond what is inside the operator; the result is in ``MeV·[ψ]²`` where
``[ψ]`` is the normalization of the supplied coefficient vectors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import jax
import jax.numpy as jnp

from lax.types import Interaction, MatrixElementHelper, lift_block


def _me_overlap(bra: jax.Array, ket: jax.Array, conjugate: bool) -> jax.Array:
    """Overlap ``Σᵢ braᵢ ketᵢ`` batched over broadcast leading axes."""

    b = jnp.conj(bra) if conjugate else bra
    result: jax.Array = jnp.einsum("...m,...m->...", b, ket)
    return result


def _me_diagonal(bra: jax.Array, operator: jax.Array, ket: jax.Array, conjugate: bool) -> jax.Array:
    """Diagonal/local element ``Σᵢ braᵢ V(rᵢ) ketᵢ`` over broadcast leading axes."""

    b = jnp.conj(bra) if conjugate else bra
    result: jax.Array = jnp.einsum("...m,...m,...m->...", b, operator, ket)
    return result


def _me_matrix(bra: jax.Array, operator: jax.Array, ket: jax.Array, conjugate: bool) -> jax.Array:
    """Full-matrix element ``braᵀ O ket`` over broadcast leading axes."""

    b = jnp.conj(bra) if conjugate else bra
    result: jax.Array = jnp.einsum("...m,...mn,...n->...", b, operator, ket)
    return result


_ME_OVERLAP_JIT = jax.jit(_me_overlap, static_argnames=("conjugate",))
_ME_DIAGONAL_JIT = jax.jit(_me_diagonal, static_argnames=("conjugate",))
_ME_MATRIX_JIT = jax.jit(_me_matrix, static_argnames=("conjugate",))


def matrix_element(
    bra: jax.Array,
    ket: jax.Array,
    operator: jax.Array | Interaction | None = None,
    *,
    conjugate: bool,
) -> jax.Array:
    """Bilinear matrix element on raw arrays with NumPy-style broadcasting.

    The standalone form: no solver-aware axis interpretation or validation is
    performed — callers align their own leading batch axes (size-1 axes
    broadcast; the contraction is over the trailing ``M`` axes only).  The
    solver-bound ``solver.matrix_element`` adds the canonical block × energy
    interpretation, the deterministic rank-2 rule, and shape validation.

    Parameters
    ----------
    bra, ket
        Mesh-coefficient arrays, shape ``(..., M)``.
    operator
        ``None`` (overlap), unscaled local node values ``(M,)``, a
        Gauss-scaled kernel matrix ``(..., M, M)`` (the trailing two axes are
        contracted), or an :class:`~lax.Interaction` (its ``block`` is used
        directly — already correctly scaled; see the module docstring).
        A *batched diagonal* operator is not expressible as a bare array here
        — pass it as diagonal-embedded matrices or an ``Interaction``.
    conjugate
        Keyword-required.  ``False`` computes the non-conjugated bilinear
        ``braᵀ·O·ket``; ``True`` computes ``bra†·O·ket``.

    Returns
    -------
    jax.Array
        The broadcast batch shape of the inputs' leading axes — a scalar for
        unbatched inputs.
    """

    if isinstance(operator, Interaction):
        operator = operator.block
    if operator is None:
        return cast(jax.Array, _ME_OVERLAP_JIT(bra, ket, conjugate=conjugate))
    operator = jnp.asarray(operator)
    if operator.ndim == 1:
        return cast(jax.Array, _ME_DIAGONAL_JIT(bra, operator, ket, conjugate=conjugate))
    return cast(jax.Array, _ME_MATRIX_JIT(bra, operator, ket, conjugate=conjugate))


@dataclass(frozen=True)
class _MatrixElementHelper:
    """Pickle-safe solver-bound bilinear matrix-element helper.

    Interprets bare-array inputs in the canonical **block × energy** axis
    order with a deterministic, mode-based rank-2 rule (never shape-sniffing):
    in blocks mode rank-2 is always block-leading ``(N_b, M)``; in channels
    mode it is always energy-leading ``(N_E, M)``.  Internally every state is
    lifted to rank-3 ``(N_b|1, N_E|1, M)`` so block-leading inputs broadcast
    on the block axis rather than right-aligning onto the energy axis.
    """

    block_mode: bool
    n_blocks: int
    n_energies: int | None
    matrix_size: int

    def __call__(
        self,
        bra: jax.Array,
        ket: jax.Array,
        operator: jax.Array | Interaction | None = None,
        *,
        conjugate: bool,
    ) -> jax.Array:
        """Evaluate the bilinear element per the F1 contract.

        See :func:`matrix_element` for the operator forms and the scaling
        contract.  Output shape is the broadcast of the batch axes the inputs
        actually carry: scalar for unbatched inputs, ``(N_b,)`` /
        ``(N_E,)`` / ``(N_b, N_E)`` otherwise.
        """

        bra3, bra_has_block, bra_has_energy = self._normalize_state(bra, "bra")
        ket3, ket_has_block, ket_has_energy = self._normalize_state(ket, "ket")
        has_block = bra_has_block or ket_has_block
        has_energy = bra_has_energy or ket_has_energy

        if operator is None:
            result = cast(jax.Array, _ME_OVERLAP_JIT(bra3, ket3, conjugate=conjugate))
        elif isinstance(operator, Interaction):
            has_block = has_block or operator.block_dependent
            has_energy = has_energy or operator.energy_dependent
            block = lift_block(
                operator.block,
                operator.energy_dependent,
                operator.block_dependent,
                True,
                True,
            )
            if block.shape[-2:] != (self.matrix_size, self.matrix_size):
                msg = (
                    f"operator Interaction block has trailing shape {block.shape[-2:]}, "
                    f"expected (M, M) = ({self.matrix_size}, {self.matrix_size}); "
                    "the Interaction must be assembled by this solver's builders."
                )
                raise ValueError(msg)
            result = cast(jax.Array, _ME_MATRIX_JIT(bra3, block, ket3, conjugate=conjugate))
        else:
            operator = jnp.asarray(operator)
            if operator.ndim == 1:
                if operator.shape != (self.matrix_size,):
                    msg = (
                        f"diagonal operator must have shape (M,) = ({self.matrix_size},), "
                        f"got {operator.shape}; values are the unscaled V(r_i) node samples."
                    )
                    raise ValueError(msg)
                result = cast(
                    jax.Array, _ME_DIAGONAL_JIT(bra3, operator, ket3, conjugate=conjugate)
                )
            elif operator.ndim == 2:
                if operator.shape != (self.matrix_size, self.matrix_size):
                    msg = (
                        f"matrix operator must have shape (M, M) = "
                        f"({self.matrix_size}, {self.matrix_size}), got {operator.shape}."
                    )
                    raise ValueError(msg)
                result = cast(jax.Array, _ME_MATRIX_JIT(bra3, operator, ket3, conjugate=conjugate))
            else:
                msg = (
                    f"bare-array operators must be (M,) or (M, M), got rank {operator.ndim}; "
                    "batched (block-/energy-dependent) operators are passed as an Interaction."
                )
                raise ValueError(msg)

        # Squeeze the axes no input contributed: result is (N_b|1, N_E|1).
        if not has_energy:
            result = jnp.squeeze(result, axis=-1)
        if not has_block:
            result = jnp.squeeze(result, axis=0)
        return result

    def _normalize_state(self, values: jax.Array, name: str) -> tuple[jax.Array, bool, bool]:
        """Lift a state array to rank-3 ``(N_b|1, N_E|1, M)`` per the §3.3 rule."""

        values = jnp.asarray(values)
        if values.ndim == 0 or values.shape[-1] != self.matrix_size:
            msg = (
                f"{name} must have trailing dimension M = {self.matrix_size}, "
                f"got shape {values.shape}."
            )
            raise ValueError(msg)
        if values.ndim == 1:
            return values[None, None, :], False, False
        if values.ndim == 2:
            if self.block_mode:
                if values.shape[0] != self.n_blocks:
                    msg = (
                        f"rank-2 {name} in blocks mode is block-leading: expected "
                        f"(N_b, M) = ({self.n_blocks}, {self.matrix_size}), got "
                        f"{values.shape}; an energy-batched, block-broadcast input "
                        "must be written (1, N_E, M)."
                    )
                    raise ValueError(msg)
                return values[:, None, :], True, False
            if self.n_energies is None:
                msg = (
                    f"rank-2 {name} in channels mode is energy-leading (N_E, M), but "
                    "this solver was compiled without an energy grid; pass (M,) or "
                    "recompile with energies=."
                )
                raise ValueError(msg)
            if values.shape[0] != self.n_energies:
                msg = (
                    f"rank-2 {name} in channels mode is energy-leading: expected "
                    f"(N_E, M) = ({self.n_energies}, {self.matrix_size}), got {values.shape}."
                )
                raise ValueError(msg)
            return values[None, :, :], False, True
        if values.ndim == 3:
            n_b, n_e = values.shape[0], values.shape[1]
            if n_b not in (1, self.n_blocks):
                msg = (
                    f"rank-3 {name} leading block axis must be 1 or N_b = {self.n_blocks}, "
                    f"got shape {values.shape}."
                )
                raise ValueError(msg)
            allowed_e = (1,) if self.n_energies is None else (1, self.n_energies)
            if n_e not in allowed_e:
                msg = (
                    f"rank-3 {name} energy axis must be one of {allowed_e}, "
                    f"got shape {values.shape}."
                )
                raise ValueError(msg)
            return values, True, True
        msg = (
            f"{name} must be rank 1-3 with trailing dimension M = {self.matrix_size}, "
            f"got shape {values.shape}."
        )
        raise ValueError(msg)


def make_matrix_element(
    *,
    matrix_size: int,
    n_blocks: int,
    n_energies: int | None,
    block_mode: bool,
) -> MatrixElementHelper:
    """Return the solver-bound batched bilinear matrix-element helper.

    Depends only on compile-time shapes (never on mesh data, boundary values,
    energies, or method), so it is bound unconditionally in every compile mode.

    Parameters
    ----------
    matrix_size
        ``M = N_c · N``, the per-block coefficient-vector length.
    n_blocks
        ``N_b`` (1 in channels mode).
    n_energies
        ``N_E``, or ``None`` when no energy grid was compiled.
    block_mode
        Whether the solver was compiled with ``blocks=``.
    """

    return _MatrixElementHelper(
        block_mode=block_mode,
        n_blocks=n_blocks,
        n_energies=n_energies,
        matrix_size=matrix_size,
    )


__all__ = ["make_matrix_element", "matrix_element"]

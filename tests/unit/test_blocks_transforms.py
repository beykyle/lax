"""F3 acceptance tests: block-batched Fourier/grid transforms (spec v0.1.5.1, T10)."""

from __future__ import annotations

import importlib
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import lax
from tests.unit._blocks_helpers import HBAR2_2MU, partial_wave_groups

N = 16
RADIUS = 10.0
ENERGIES = jnp.linspace(2.0, 30.0, 5)
MOMENTA = jnp.linspace(0.1, 2.0, 9)
GRID = jnp.linspace(0.4, RADIUS - 0.4, 7)
MESH = lax.MeshSpec("legendre", "x", n=N, scale=RADIUS)
BLOCK_GROUPS = partial_wave_groups()
N_B = len(BLOCK_GROUPS)

TIGHT = dict(rtol=1e-10, atol=1e-12)


def _blocked_solver() -> lax.Solver:
    return lax.compile(
        mesh=MESH,
        blocks=BLOCK_GROUPS,
        solvers=("spectrum",),
        energies=ENERGIES,
        momenta=MOMENTA,
        grid=GRID,
    )


def _single_solver(block: tuple[lax.ChannelSpec, ...]) -> lax.Solver:
    return lax.compile(
        mesh=MESH,
        channels=block,
        solvers=("spectrum",),
        energies=ENERGIES,
        momenta=MOMENTA,
        grid=GRID,
    )


def test_compile_accepts_momenta_with_blocks() -> None:
    """The former momenta × blocks rejection is lifted (old C1)."""

    solver = _blocked_solver()
    assert solver.fourier is not None
    assert solver.double_fourier_transform is not None
    assert solver.transforms.F_momentum is not None
    assert solver.transforms.F_momentum.shape == (N_B, 1, len(MOMENTA), N)


def test_blocked_fourier_matches_per_block_solvers() -> None:
    """Blocked fourier/double_fourier ≡ per-block channels-mode solvers (§15.5)."""

    solver = _blocked_solver()
    rng = np.random.default_rng(20)
    vectors = jnp.asarray(rng.normal(size=(N_B, N)))
    kernels = jnp.asarray(rng.normal(size=(N_B, N, N)))

    assert solver.fourier is not None
    assert solver.double_fourier_transform is not None
    batched_vec = solver.fourier(vectors)
    batched_ker = solver.fourier(kernels)
    batched_double = solver.double_fourier_transform(kernels)
    assert batched_vec.shape == (N_B, len(MOMENTA))
    assert batched_ker.shape == (N_B, len(MOMENTA), len(MOMENTA))

    for b, group in enumerate(BLOCK_GROUPS):
        single = _single_solver(group)
        assert single.fourier is not None
        assert single.double_fourier_transform is not None
        np.testing.assert_allclose(
            np.asarray(batched_vec[b]),
            np.asarray(single.fourier(vectors[b])),
            err_msg=f"fourier vector block {b}",
            **TIGHT,
        )
        np.testing.assert_allclose(
            np.asarray(batched_ker[b]),
            np.asarray(single.fourier(kernels[b])),
            err_msg=f"fourier kernel block {b}",
            **TIGHT,
        )
        np.testing.assert_allclose(
            np.asarray(batched_double[b]),
            np.asarray(single.double_fourier_transform(kernels[b])),
            err_msg=f"double fourier block {b}",
            **TIGHT,
        )


def test_unbatched_inputs_broadcast_across_blocks() -> None:
    """Unbatched ``(N,)`` / ``(N, N)`` inputs broadcast over the block axis."""

    solver = _blocked_solver()
    rng = np.random.default_rng(21)
    vector = jnp.asarray(rng.normal(size=N))
    kernel = jnp.asarray(rng.normal(size=(N, N)))

    assert solver.fourier is not None
    assert solver.double_fourier_transform is not None
    broadcast_vec = solver.fourier(vector)
    broadcast_ker = solver.fourier(kernel)
    broadcast_double = solver.double_fourier_transform(kernel)
    assert broadcast_vec.shape == (N_B, len(MOMENTA))

    stacked_vec = solver.fourier(jnp.broadcast_to(vector, (N_B, N)))
    stacked_ker = solver.fourier(jnp.broadcast_to(kernel, (N_B, N, N)))
    stacked_double = solver.double_fourier_transform(jnp.broadcast_to(kernel, (N_B, N, N)))
    np.testing.assert_allclose(np.asarray(broadcast_vec), np.asarray(stacked_vec), **TIGHT)
    np.testing.assert_allclose(np.asarray(broadcast_ker), np.asarray(stacked_ker), **TIGHT)
    np.testing.assert_allclose(np.asarray(broadcast_double), np.asarray(stacked_double), **TIGHT)


def test_blocks_mode_shape_violations_raise() -> None:
    solver = _blocked_solver()
    rng = np.random.default_rng(22)
    assert solver.fourier is not None
    assert solver.double_fourier_transform is not None
    with pytest.raises(ValueError, match="blocks mode"):
        solver.fourier(jnp.asarray(rng.normal(size=(N_B + 1, N))))
    with pytest.raises(ValueError, match="blocks mode"):
        solver.double_fourier_transform(jnp.asarray(rng.normal(size=(N_B + 1, N, N))))


def test_grid_transforms_pass_leading_batch_axes() -> None:
    """``grid=`` projections accept arbitrary leading batch axes in both modes."""

    solver = _blocked_solver()
    rng = np.random.default_rng(23)
    coefficients = jnp.asarray(rng.normal(size=(N_B, len(ENERGIES), N)))
    kernels = jnp.asarray(rng.normal(size=(N_B, N, N)))

    assert solver.to_grid_vector is not None
    assert solver.from_grid_vector is not None
    assert solver.to_grid_matrix is not None
    batched = solver.to_grid_vector(coefficients)
    assert batched.shape == (N_B, len(ENERGIES), len(GRID))
    for b in range(N_B):
        for e in range(len(ENERGIES)):
            np.testing.assert_allclose(
                np.asarray(batched[b, e]),
                np.asarray(solver.to_grid_vector(coefficients[b, e])),
                **TIGHT,
            )

    batched_kernels = solver.to_grid_matrix(kernels)
    assert batched_kernels.shape == (N_B, len(GRID), len(GRID))
    for b in range(N_B):
        np.testing.assert_allclose(
            np.asarray(batched_kernels[b]),
            np.asarray(solver.to_grid_matrix(kernels[b])),
            **TIGHT,
        )

    round_trip = solver.from_grid_vector(batched)
    assert round_trip.shape == coefficients.shape


def test_f_momentum_deduplicates_per_unique_ell(monkeypatch: pytest.MonkeyPatch) -> None:
    """``compute_F_momentum`` runs once per unique ℓ across the whole block set."""

    compile_module = importlib.import_module("lax.compile")
    real_compute = compile_module.compute_F_momentum
    calls: list[int] = []

    def counting_compute(mesh: Any, momenta: Any, angular_momentum: int, *args: Any) -> Any:
        calls.append(angular_momentum)
        return real_compute(mesh, momenta, angular_momentum, *args)

    monkeypatch.setattr(compile_module, "compute_F_momentum", counting_compute)
    shared_ell_blocks = (
        (lax.ChannelSpec(l=0, threshold=0.0, mass_factor=HBAR2_2MU),),
        (lax.ChannelSpec(l=1, threshold=0.0, mass_factor=HBAR2_2MU),),
        (lax.ChannelSpec(l=1, threshold=0.0, mass_factor=HBAR2_2MU),),
    )
    lax.compile(
        mesh=MESH,
        blocks=shared_ell_blocks,
        solvers=("spectrum",),
        energies=ENERGIES,
        momenta=MOMENTA,
    )
    assert sorted(calls) == [0, 1]


def test_channels_mode_fourier_is_unchanged() -> None:
    """Channels-mode fourier keeps its rank-1/rank-2 contract and rejects batches."""

    single = _single_solver(BLOCK_GROUPS[0])
    rng = np.random.default_rng(24)
    assert single.fourier is not None
    assert single.transforms.F_momentum is not None
    assert single.transforms.F_momentum.shape == (1, len(MOMENTA), N)
    with pytest.raises(ValueError, match="vector or"):
        single.fourier(jnp.asarray(rng.normal(size=(2, 3, N))))


def test_blocked_transforms_round_trip_through_pickle() -> None:
    import pickle

    solver = _blocked_solver()
    restored = pickle.loads(pickle.dumps(solver))
    rng = np.random.default_rng(25)
    vectors = jnp.asarray(rng.normal(size=(N_B, N)))
    assert restored.fourier is not None and solver.fourier is not None
    np.testing.assert_allclose(
        np.asarray(restored.fourier(vectors)),
        np.asarray(solver.fourier(vectors)),
        **TIGHT,
    )


def test_jit_compatible_dispatch() -> None:
    """The block dispatch is shape-static, so the transforms compose with jax.jit."""

    solver = _blocked_solver()
    rng = np.random.default_rng(26)
    vectors = jnp.asarray(rng.normal(size=(N_B, N)))
    assert solver.fourier is not None
    fourier = solver.fourier

    @jax.jit
    def pipeline(values: jax.Array) -> jax.Array:
        return fourier(values)

    np.testing.assert_allclose(np.asarray(pipeline(vectors)), np.asarray(fourier(vectors)), **TIGHT)

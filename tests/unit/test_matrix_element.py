"""F1 acceptance tests: ``solver.matrix_element`` bilinear forms (spec v0.1.5.1, T1-T3)."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import lax
from lax.transforms import matrix_element as standalone_matrix_element
from tests.unit._blocks_helpers import HBAR2_2MU, gaussian_kernel, partial_wave_groups

N = 8
RADIUS = 6.0
ENERGIES = jnp.linspace(1.0, 9.0, 3)
N_E = len(ENERGIES)
MESH = lax.MeshSpec("legendre", "x", n=N, scale=RADIUS)

TIGHT = dict(rtol=1e-12, atol=1e-13)


def _channels_solver(energies: jnp.ndarray | None = ENERGIES) -> lax.Solver:
    return lax.compile(
        mesh=MESH,
        channels=(lax.ChannelSpec(l=0, threshold=0.0, mass_factor=HBAR2_2MU),),
        solvers=("spectrum",),
        energies=energies,
    )


def _blocks_solver():
    return lax.compile(
        mesh=MESH,
        blocks=partial_wave_groups(),
        solvers=("spectrum",),
        energies=ENERGIES,
    )


def _random_complex(rng: np.random.Generator, shape: tuple[int, ...]) -> jnp.ndarray:
    return jnp.asarray(rng.normal(size=shape) + 1j * rng.normal(size=shape))


# ---------------------------------------------------------------------------
# T1 — matrix_element ↔ integrate consistency


def test_conjugated_self_element_matches_integrate() -> None:
    """``matrix_element(x, x, O, conjugate=True)`` equals ``integrate(x, O)``."""

    solver = _channels_solver()
    rng = np.random.default_rng(1)
    x = _random_complex(rng, (N,))
    hermitian = _random_complex(rng, (N, N))
    hermitian = hermitian + hermitian.conj().T
    diagonal = jnp.asarray(rng.normal(size=N))

    assert solver.matrix_element is not None
    assert solver.integrate is not None
    np.testing.assert_allclose(
        np.asarray(solver.matrix_element(x, x, conjugate=True)),
        np.asarray(solver.integrate(x)),
        **TIGHT,
    )
    np.testing.assert_allclose(
        np.asarray(solver.matrix_element(x, x, diagonal, conjugate=True)),
        np.asarray(solver.integrate(x, diagonal)),
        **TIGHT,
    )
    np.testing.assert_allclose(
        np.asarray(solver.matrix_element(x, x, hermitian, conjugate=True)),
        np.asarray(solver.integrate(x, hermitian)),
        **TIGHT,
    )


# ---------------------------------------------------------------------------
# T2 — analytic separable bilinear + batched broadcasting


def test_separable_operator_is_exact() -> None:
    """For rank-1 ``O = u vᵀ``: ``matrix_element(b, k, O) == (bᵀu)(vᵀk)`` exactly."""

    solver = _channels_solver()
    rng = np.random.default_rng(2)
    b = _random_complex(rng, (N,))
    k = _random_complex(rng, (N,))
    u = _random_complex(rng, (N,))
    v = _random_complex(rng, (N,))
    separable = jnp.outer(u, v)

    np.testing.assert_allclose(
        np.asarray(solver.matrix_element(b, k, separable, conjugate=False)),
        np.asarray((b @ u) * (v @ k)),
        **TIGHT,
    )
    np.testing.assert_allclose(
        np.asarray(solver.matrix_element(b, k, separable, conjugate=True)),
        np.asarray((b.conj() @ u) * (v @ k)),
        **TIGHT,
    )


def test_batched_states_match_python_loop() -> None:
    """Full-rank ``(N_b, N_E, M)`` states reproduce the explicit double loop."""

    solver = _blocks_solver()
    n_b = len(partial_wave_groups())
    rng = np.random.default_rng(3)
    bra = _random_complex(rng, (n_b, N_E, N))
    ket = _random_complex(rng, (n_b, N_E, N))
    operator = _random_complex(rng, (N, N))

    result = solver.matrix_element(bra, ket, operator, conjugate=False)
    assert result.shape == (n_b, N_E)
    expected = np.array(
        [
            [
                np.asarray(bra[b, e]) @ np.asarray(operator) @ np.asarray(ket[b, e])
                for e in range(N_E)
            ]
            for b in range(n_b)
        ]
    )
    np.testing.assert_allclose(np.asarray(result), expected, **TIGHT)


def test_blocks_mode_rank2_is_block_leading() -> None:
    """Rank-2 in blocks mode lifts to ``(N_b, 1, M)`` — never onto the energy axis."""

    solver = _blocks_solver()
    n_b = len(partial_wave_groups())
    assert n_b == N_E  # the ambiguous case the deterministic rule resolves
    rng = np.random.default_rng(4)
    bra2 = _random_complex(rng, (n_b, N))
    ket3 = _random_complex(rng, (1, N_E, N))

    result = solver.matrix_element(bra2, ket3, conjugate=False)
    assert result.shape == (n_b, N_E)
    expected = np.array(
        [[np.asarray(bra2[b]) @ np.asarray(ket3[0, e]) for e in range(N_E)] for b in range(n_b)]
    )
    np.testing.assert_allclose(np.asarray(result), expected, **TIGHT)


def test_channels_mode_rank2_is_energy_leading() -> None:
    """Rank-2 in channels mode lifts to ``(1, N_E, M)``."""

    solver = _channels_solver()
    rng = np.random.default_rng(5)
    bra2 = _random_complex(rng, (N_E, N))
    ket1 = _random_complex(rng, (N,))

    result = solver.matrix_element(bra2, ket1, conjugate=False)
    assert result.shape == (N_E,)
    expected = np.array([np.asarray(bra2[e]) @ np.asarray(ket1) for e in range(N_E)])
    np.testing.assert_allclose(np.asarray(result), expected, **TIGHT)


def test_unbatched_inputs_return_a_scalar() -> None:
    solver = _channels_solver()
    rng = np.random.default_rng(6)
    b = _random_complex(rng, (N,))
    k = _random_complex(rng, (N,))
    assert solver.matrix_element(b, k, conjugate=False).shape == ()


def test_shape_violations_raise() -> None:
    blocks = _blocks_solver()
    channels_no_grid = _channels_solver(energies=None)
    rng = np.random.default_rng(7)
    n_b = len(partial_wave_groups())

    with pytest.raises(ValueError, match="block-leading"):
        blocks.matrix_element(
            _random_complex(rng, (n_b + 1, N)), _random_complex(rng, (N,)), conjugate=False
        )
    with pytest.raises(ValueError, match="without an energy grid"):
        channels_no_grid.matrix_element(
            _random_complex(rng, (2, N)), _random_complex(rng, (N,)), conjugate=False
        )
    with pytest.raises(ValueError, match="Interaction"):
        blocks.matrix_element(
            _random_complex(rng, (N,)),
            _random_complex(rng, (N,)),
            _random_complex(rng, (n_b, N, N)),
            conjugate=False,
        )
    with pytest.raises(ValueError, match="trailing dimension"):
        blocks.matrix_element(
            _random_complex(rng, (N + 1,)), _random_complex(rng, (N,)), conjugate=False
        )


# ---------------------------------------------------------------------------
# T3 — Interaction scaling contract


def test_nonlocal_interaction_matches_explicit_gauss_sum() -> None:
    """``matrix_element(b, k, interaction)`` equals the explicit double Gauss sum."""

    solver = _channels_solver()
    rng = np.random.default_rng(8)
    b = _random_complex(rng, (N,))
    k = _random_complex(rng, (N,))
    kernel = gaussian_kernel(10.0)
    assert solver.nonlocal_potential is not None
    interaction = solver.nonlocal_potential(kernel)

    radii = np.asarray(solver.mesh.radii)
    weights = np.asarray(solver.mesh.weights)
    scale = solver.mesh.scale
    raw_kernel = np.asarray(kernel(jnp.asarray(radii[:, None]), jnp.asarray(radii[None, :])))
    gauss_scaled = np.sqrt(np.outer(weights, weights)) * scale * raw_kernel

    np.testing.assert_allclose(
        np.asarray(solver.matrix_element(b, k, interaction, conjugate=False)),
        np.asarray(b) @ gauss_scaled @ np.asarray(k),
        rtol=1e-12,
        atol=1e-12,
    )


def test_local_interaction_matches_plain_node_sum() -> None:
    """A local Interaction term reduces to the unscaled node sum ``Σᵢ bᵢ V(rᵢ) kᵢ``."""

    solver = _channels_solver()
    rng = np.random.default_rng(9)
    b = _random_complex(rng, (N,))
    k = _random_complex(rng, (N,))

    def profile(r: jnp.ndarray) -> jnp.ndarray:
        return -3.0 * jnp.exp(-0.1 * r**2)

    assert solver.local_potential is not None
    interaction = solver.local_potential(profile)
    node_values = np.asarray(profile(solver.mesh.radii))

    np.testing.assert_allclose(
        np.asarray(solver.matrix_element(b, k, interaction, conjugate=False)),
        np.sum(np.asarray(b) * node_values * np.asarray(k)),
        rtol=1e-12,
        atol=1e-12,
    )
    # The bare-array diagonal form takes the same unscaled node values.
    np.testing.assert_allclose(
        np.asarray(solver.matrix_element(b, k, jnp.asarray(node_values), conjugate=False)),
        np.asarray(solver.matrix_element(b, k, interaction, conjugate=False)),
        **TIGHT,
    )


def test_block_dependent_interaction_broadcasts_per_block() -> None:
    """A block-dependent Interaction aligns on the block axis of rank-2 states."""

    solver = _blocks_solver()
    n_b = len(partial_wave_groups())
    rng = np.random.default_rng(10)
    bra = _random_complex(rng, (n_b, N))
    ket = _random_complex(rng, (n_b, N))
    kernels = [gaussian_kernel(5.0 * (b + 1)) for b in range(n_b)]
    assert solver.nonlocal_potential is not None
    interaction = solver.nonlocal_potential(kernels, block_dependent=True)

    result = solver.matrix_element(bra, ket, interaction, conjugate=False)
    assert result.shape == (n_b,)
    expected = np.array(
        [
            np.asarray(bra[b]) @ np.asarray(interaction.block[b]) @ np.asarray(ket[b])
            for b in range(n_b)
        ]
    )
    np.testing.assert_allclose(np.asarray(result), expected, **TIGHT)


# ---------------------------------------------------------------------------
# Standalone form


def test_standalone_matches_solver_bound_helper() -> None:
    solver = _channels_solver()
    rng = np.random.default_rng(11)
    b = _random_complex(rng, (N,))
    k = _random_complex(rng, (N,))
    operator = _random_complex(rng, (N, N))

    np.testing.assert_allclose(
        np.asarray(standalone_matrix_element(b, k, operator, conjugate=False)),
        np.asarray(solver.matrix_element(b, k, operator, conjugate=False)),
        **TIGHT,
    )
    np.testing.assert_allclose(
        np.asarray(standalone_matrix_element(b, k, conjugate=True)),
        np.asarray(solver.matrix_element(b, k, conjugate=True)),
        **TIGHT,
    )


def test_conjugate_is_keyword_required() -> None:
    solver = _channels_solver()
    rng = np.random.default_rng(12)
    b = _random_complex(rng, (N,))
    with pytest.raises(TypeError):
        solver.matrix_element(b, b)  # type: ignore[call-arg]

# Copilot instructions for `lax`

## What this repository is

A JAX-compiled Python library implementing the Lagrange-mesh method (LMM) for solving
the radial Schrödinger equation. It supports bound-state (eigenvalue) and continuum
(R-matrix, S-matrix, Green's function) calculations via a unified spectral-decomposition
kernel. The library is designed as a low-level numerical engine for reaction codes,
potential-fitting pipelines, and uncertainty-quantification workflows.

**Read `DESIGN.md` before writing any code.** It is the authoritative reference for every
design decision, formula, unit convention, and module contract. All section and equation
references in code comments use the notation from that document (e.g. `[Baye eq. 3.120]`,
`[Desc. eq. 22]`, `DESIGN.md §11.5`).

The two reference papers are:
- **[1] Baye (2015)** — *Physics Reports 565*, 1–107. The Lagrange-mesh method review.
- **[2] Descouvemont (2016)** — *CPC 200*, 199–219. The R-matrix Fortran package.

---

## Git workflow — mandatory

1. **Never commit directly to `main` or any integration branch.**
2. **Create a branch per issue** using the naming convention:
   `<issue-number>-<short-slug>`, e.g. `3-legendre-x-builder` or `14-yamaguchi-keystone`.
3. **Open a pull request and request review** before merging, even for trivial changes.
   Describe what changed, what tests now pass, and (for numerical code) what reference
   value was verified.
4. **Each commit must leave the test suite in a passing state** for all tests that were
   passing before the commit. A commit that breaks an existing test must not be pushed.
5. **Do not squash benchmark or property test additions** — they are permanent record of
   what has been validated.

Branch protection on `main`:
- At least one approving review required.
- All status checks must pass before merge.
- Force-push disabled.

---

## Repository layout

```
src/lax/          # installed package (src layout)
│
├── __init__.py             # sets jax_enable_x64=True; re-exports public API
├── types.py                # MeshSpec, ChannelSpec (user-facing specs)
├── compile.py              # compile() factory — the public entry point
├── propagate.py            # R-matrix subinterval propagation (Phase 9)
│
├── meshes/                 # one file per (family, regularization) pair
│   ├── _registry.py        # @register decorator and build_mesh() dispatch
│   ├── _basis_eval.py      # f_j(r) evaluation for to_grid / fourier
│   ├── _quadrature.py      # node/weight tables
│   ├── legendre.py         # shifted Legendre: "x", "x(1-x)", "x^3/2"
│   └── laguerre.py         # Laguerre: "x", "x^3/2", "modified_x^2"
│
├── operators/              # compile-time matrix builders
│   ├── kinetic.py          # T, T+L, T_alpha
│   ├── position.py         # 1/r, 1/r², x, x²
│   ├── derivative.py       # d/dr
│   └── potential.py        # assemble_local, assemble_nonlocal
│
├── spectral/               # mesh-independent; pure linear algebra on eigenpairs
│   ├── types.py            # Spectrum pytree
│   ├── observables.py      # rmatrix_from_spectrum, greens_from_spectrum, ...
│   ├── matching.py         # smatrix_from_R, phases_from_S
│   └── interpolation.py   # pade_interpolate
│
├── solvers/                # wires mesh + operators + spectral into JIT kernels
│   ├── assembly.py         # assemble_block_hamiltonian, build_Q
│   ├── spectrum.py         # make_spectrum_kernel (eigh / eig)
│   ├── observables.py      # _bind_solvers (rmatrix, smatrix, phases, greens)
│   └── linear_solve.py     # make_rmatrix_direct_kernel (linear-solve fallback)
│
├── boundary/               # mpmath boundary-value computation (compile-time only)
│   ├── _types.py           # BoundaryValues, OperatorMatrices, Mesh, Solver pytrees
│   └── coulomb.py          # compute_boundary_values
│
└── transforms/             # compile-time matrix builders; runtime is pure matmul
    ├── grid.py             # compute_B_grid, make_to_grid_vector/matrix
    ├── fourier.py          # compute_F_momentum
    └── integration.py      # norm, expect_local, expect_operator

tests/
├── conftest.py             # shared fixtures (mesh instances, reference potentials)
├── unit/                   # fast, no JAX tracing required
├── benchmarks/             # known-analytic comparisons (slower, marked @benchmark)
└── property/               # hypothesis-based (marked @property)

theory/
├── cc_rmatrix.pdf          # Descouvemont's R-matrix review (2016 CPC 200)
└── lagrange_mesh.pdf       # Baye's Lagrange-mesh review (2015 Physics Reports 565)
```

---

## Typing — strict

Every function, method, and module-level variable must carry complete type annotations.

### Rules

```python
# Required at the top of every .py file:
from __future__ import annotations
```

- **All function signatures must be fully annotated** — parameters and return type,
  including `-> None` for procedures.
- **No `Any`** from `typing`. If a type is genuinely unknown, use `object` or an
  appropriate `TypeVar`. If JAX's type stubs are insufficient, use
  `jax.Array` (not `jnp.ndarray`, which is an alias for `Any` in practice).
- **Use `jax.Array`** for all runtime array arguments and return values.
  Use `np.ndarray` only for compile-time NumPy arrays (mesh builders, quadrature).
- **Use `Literal` and `TypeAlias`** for constrained string arguments.
  `MeshFamily`, `Regularization`, and `Method` are already defined in `types.py`.
- **Dataclasses must be `frozen=True`** and annotated field by field.
- **`Callable` signatures should be spelled out** where the callable is part of a
  public API. Use `Callable[[ArgType, ...], ReturnType]`, not bare `Callable`.
- **`tuple[X, ...]` not `Tuple[X, ...]`** (Python 3.12+ builtin generics throughout).

### Pyright

The project uses Pyright in `basic` mode (`pyproject.toml` `[tool.pyright]`). All code
must pass `pyright src/` with zero errors before a PR is opened. If Pyright cannot
infer a type due to JAX's own stub limitations, add a narrow inline cast:
```python
eigvals: jax.Array = jnp.linalg.eigh(H)[0]  # Pyright loses type through tuple unpack
```
Do not use `# type: ignore` without a comment explaining why.

### Example of correct annotation style

```python
from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from lax.types import ChannelSpec, MeshSpec
from lax.spectral.types import Spectrum


def rmatrix_from_spectrum(
    spectrum: Spectrum,
    E: float,
    channel_radius: float,
    mass_factor: float,
) -> jax.Array:
    """R_cc'(E) = (1/a) Σ_k γ_kc γ_kc' / (ε_k - E/μ). Shape: (N_c, N_c)."""
    gamma: jax.Array = spectrum.surface_amplitudes
    E_dimless: jax.Array = jnp.asarray(E / mass_factor)
    denom: jax.Array = 1.0 / (spectrum.eigenvalues - E_dimless)
    return jnp.einsum("m,mc,md->cd", denom, gamma, gamma) / channel_radius
```

---

## Formatting — ruff

All code must pass `ruff check src/ tests/` and `ruff format --check src/ tests/` with
zero violations before a PR is opened.

Current ruff config (`pyproject.toml`):
- Line length: 100
- Rules: `E`, `F`, `W`, `I` (isort), `UP` (pyupgrade)
- `E501` (line too long) is suppressed because ruff's formatter handles it
- `E741` (ambiguous variable name) is suppressed repo-wide because single-letter
  physics variables like `l` are standard and intentional here

Auto-format before committing:
```bash
uv run ruff format src/ tests/
uv run ruff check --fix src/ tests/
```

### Additional style rules not enforced by ruff

- **Import order in every file:** stdlib → third-party (`jax`, `numpy`, `scipy`,
  `mpmath`) → local (`lax.*`). One blank line between groups.
- **No wildcard imports** (`from jax.numpy import *`).
- **Single-letter physics names like `l` are allowed.** Ruff E741 is ignored
  repo-wide, so do not add per-use `noqa` comments just for orbital angular momentum
  or similar conventional notation.
- **No mutable default arguments.** Use `field(default_factory=...)` in dataclasses,
  `None` + guard in functions.
- **String quotes:** double quotes throughout (ruff formatter enforces this).
- **Docstrings:** NumPy style. Every public function has a one-line summary, a
  `Parameters` section if there are non-obvious arguments, and a `Returns` section.
  Reference the design doc equation inline in the summary line where applicable:
  ```python
  def build_Q(mesh: Mesh, channels: tuple[ChannelSpec, ...]) -> jax.Array:
      """Surface projector Q[c·N+j, c'] = δ_cc' φ_j(a). [DESIGN.md §11.5]"""
  ```

---

## JAX patterns — required and forbidden

### Required

- **`from __future__ import annotations` before any JAX import** in every file that
  uses JAX types in annotations. This prevents forward-reference evaluation errors.
- **`jax.config.update("jax_enable_x64", True)` runs at package import** (already in
  `src/lax/__init__.py`). Never call it again elsewhere; never call
  `jax.config.update("jax_enable_x64", False)` anywhere in the codebase.
- **All arrays passed to `@jax.jit` functions must be `jax.Array`**, not `np.ndarray`.
  Convert with `jnp.asarray(x)` at the boundary between compile-time (numpy) and
  runtime (JAX) code. The canonical boundary is the end of `compile()`.
- **Static fields in pytree dataclasses** must be declared with
  `field(metadata={"static": True})`. Changing a static field invalidates the JIT
  cache. The fields `n`, `family`, `regularization`, `scale`, `is_hermitian`, and
  `method` are static.
- **Python `for` loops over `range(N_c)` in assemblers are correct and intentional.**
  `N_c` (number of channels) is a compile-time static integer. JAX unrolls these loops
  at trace time. Do not replace them with `jnp.stack` or `lax.scan` unless N_c is
  genuinely dynamic (it never is in this library).
- **Use `jnp.linalg.solve(A, B)` rather than `jnp.linalg.inv(A) @ B`** everywhere.
  More numerically stable and better supported by JAX's VJP rules.
- **Use `jax.tree.map`** (not `jax.tree_util.tree_map`) for all pytree operations in
  library code. `jax.tree_util.tree_map` still works but is the older API; `jax.tree`
  is the canonical namespace introduced in JAX 0.4.25 and preferred from then on.
- **`@jax.tree_util.register_dataclass` requires JAX >= 0.4.36** for the decorator
  form with optional `data_fields`/`meta_fields`. Always mark static fields with
  `field(metadata={"static": True})`; all other fields default to pytree leaves.
  Never call the older `partial(jax.tree_util.register_dataclass, data_fields=[...],
  meta_fields=[...])` form — the `>= 0.4.36` floor ensures the simpler form is
  always available.

### Forbidden inside `@jax.jit` functions

- **`if array_value:`** — use `jnp.where` instead. Python `if` on a traced value
  raises a `ConcretizationTypeError` at trace time.
- **`.item()` or `float(array)`** — forces device-to-host transfer, breaks JIT.
- **`np.ndarray` operations** — any `numpy` call inside a JIT'd function will
  silently operate on the concrete value at trace time, not at call time.
- **`mpmath` calls** — `mpmath` is not JAX-traceable. All `mpmath` usage is confined
  to `boundary/coulomb.py` → `compute_boundary_values()`, which is called only from
  `compile()`, never from a JIT'd function.
- **`print()`** — use `jax.debug.print("{x}", x=array)` if runtime printing is needed
  for debugging. Remove before committing.
- **`scipy` calls** — all `scipy` usage is compile-time only (mesh builders,
  quadrature). Never call `scipy` inside `@jax.jit`.
- **`ndim` or `shape` conditionals** — use two separate JIT'd functions rather than
  `if x.ndim == 1` inside a single JIT'd function. See `to_grid_vector` vs
  `to_grid_matrix` in `transforms/grid.py`.

---

## Unit convention — mandatory

All quantities inside the library use the **fm⁻² convention** (dividing the Schrödinger
equation through by ℏ²/2μ). See `DESIGN.md §15.4` for the full table. Summary:

| Quantity | User provides | Library stores / computes |
|---|---|---|
| Energy `E`, threshold `E_c` | MeV | Divided by `mass_factor` → fm⁻² inside assembler |
| Potential `V` | MeV | Divided by `mass_factor` → fm⁻² inside assembler |
| `mass_factor` = ℏ²/2μ | MeV·fm² | Conversion factor |
| Kinetic matrices `TpL`, `T` | — | fm⁻² (mesh builders divide by `scale²`) |
| `Spectrum.eigenvalues` | — | fm⁻² |
| `rmatrix_from_spectrum(E=...)` | E in MeV | Divides by `mass_factor` internally |

**The `threshold / mass_factor` division in `assemble_block_hamiltonian` is not
optional.** Omitting it introduces an energy-scale error that produces plausible-looking
but wrong eigenvalues. This is `DESIGN.md Appendix C.1`.

Standard nuclear value: ℏ²/2mₙ = 20.736 MeV·fm² (`DESIGN.md §15.4`, `[2] eq. 46`).

---

## Validation matrix

Every PR that touches any of the listed modules must pass the corresponding tests.
"Must pass" means: `uv run pytest <path>` exits 0 and the assertion tolerance is met.

### Unit tests — always run

| Test file | Covers | Key assertion |
|---|---|---|
| `tests/unit/test_mesh_legendre.py` | `meshes/legendre.py`, `meshes/_registry.py` | TpL symmetric; boundary values match Desc. eq. 24; nodes ∈ (0,1) |
| `tests/unit/test_mesh_laguerre.py` | `meshes/laguerre.py` | TpL symmetric; nodes ∈ (0,∞); `basis_at_boundary` all zero |
| `tests/unit/test_spectral.py` | `spectral/` | Spectral R-matrix matches direct inverse; `smatrix_from_R` gives `\|S\|=1` for real R |
| `tests/unit/test_boundary.py` | `boundary/coulomb.py` | Wronskian `FG' - GF' = 1`; `is_open` correct; `H_+ = conj(H_-)` for real η |

### Benchmark tests — run before merging any solver change

Invoked with `uv run pytest -m benchmark`. Reference values from Baye [1] and
Descouvemont [2]; tolerances are fixed in the test parametrization and must not be
loosened.

| Test | Mesh | N | Reference | Tolerance |
|---|---|---|---|---|
| Yamaguchi δ(E=0.1 MeV) | Legendre-x, a=8 | 15 | −15.078689° | 1e-4° |
| Yamaguchi δ(E=10 MeV) | Legendre-x, a=8 | 15 | 85.634560° | 1e-4° |
| Yamaguchi δ(E=0.1 MeV) | Legendre-x, a=12 | 15 | −15.078689° | 1e-5° |
| Yamaguchi δ(E=10 MeV) | Legendre-x, a=12 | 15 | 85.634560° | 1e-5° |
| Hydrogen E₁ₛ (a.u.) | Laguerre-x, h=2 | 30 | −0.5 | 1e-10 |
| Confined H E₁ₛ, R=2 | Legendre-x(1-x), a=2 | 8 | −0.125 | 1e-13 |
| 3D HO E₀ | Mod. Laguerre, h=1 | 20 | 1.5 | 1e-12 |
| Pöschl-Teller δ(E) | Legendre-x, a=15 | 40 | analytic formula | 1e-8° |
| α+²⁰⁸Pb |U₂₀₊| (real V) | Legendre-x, a=14 | 60 | Desc. App. A | 1e-4 |
| α+²⁰⁸Pb |U₂₀₊| (complex V, `eig`) | Legendre-x, a=14 | 60 | Desc. App. A | 1e-4 |
| α+²⁰⁸Pb |U₂₀₊| (complex V, `linear_solve`) | Legendre-x, a=14 | 60 | Desc. App. A | 1e-4 |

### Property tests — run before merging any solver change

Invoked with `uv run pytest -m property`. These use Hypothesis and must pass with the
default profile (200 examples). **Tolerances are fixed and must not be loosened.**

| Test | What it checks | Tolerance |
|---|---|---|
| `test_hermiticity.py::test_TpL_symmetric` | `‖TpL - TpLᵀ‖_F / ‖TpL‖_F` | 1e-12 |
| `test_unitarity.py::test_S_unitary_real_V` | `‖S†S - I‖_F` for random real V | 1e-10 |
| `test_unitarity.py::test_S_symmetric_real_V` | `‖S - Sᵀ‖_F` for random real symmetric V | 1e-10 |
| `test_unitarity.py::test_spectral_vs_direct` | `‖R_spectral - R_direct‖_F` at each E | 1e-10 |
| `test_autograd.py::test_grad_loss_vs_fd` | `‖∂loss/∂params - FD‖ / ‖FD‖` | 1e-5 |
| `test_autograd.py::test_vmap_vs_loop` | `‖vmap(rmatrix)(spec, Es) - for-loop‖` | 1e-12 |

### Full suite command

```bash
uv run pytest tests/ -n auto
```

To run only the fast tests during development:
```bash
uv run pytest tests/unit/ -n auto
```

To run a specific benchmark:
```bash
uv run pytest tests/benchmarks/test_yamaguchi.py -v
```

---

## Before opening a pull request — checklist

Run all of the following from the repo root. A PR must not be opened if any step fails.

```bash
# 1. Format
uv run ruff format src/ tests/

# 2. Lint
uv run ruff check src/ tests/

# 3. Type-check
uv run pyright src/

# 4. Unit tests
uv run pytest tests/unit/ -n auto

# 5. Property tests (if solver code was touched)
uv run pytest -m property -n auto

# 6. Benchmark tests (if solver or mesh code was touched)
uv run pytest -m benchmark
```

If `pyright` is not available in the venv, it is missing from the `dev` group in
`pyproject.toml` — add it there rather than with `uv add --group dev pyright` directly,
so the lock file and pyproject stay in sync.

### Environment setup — backend selection

The library declares `jax>=0.4.36` (pure Python) as a runtime dependency. The compiled
backend (`jaxlib`) is installed separately via an optional extra. All backends listed
below are available from **PyPI** — no custom index URL is required. Install exactly one;
mixing backends in the same environment produces undefined behaviour.

| Extra | Hardware | Command |
|---|---|---|
| `cpu` | Any platform (default for dev) | `uv sync --extra cpu --group dev` |
| `cuda13` | NVIDIA, SM 7.5+, driver ≥ 580 | `uv sync --extra cuda13 --group dev` |
| `cuda12` | NVIDIA, SM 5.2+, driver ≥ 525 | `uv sync --extra cuda12 --group dev` |
| `cuda13-local` | NVIDIA + local CUDA 13 toolkit | `uv sync --extra cuda13-local --group dev` |
| `cuda12-local` | NVIDIA + local CUDA 12 toolkit | `uv sync --extra cuda12-local --group dev` |
| `rocm` | AMD GPU, ROCm 7, Linux | `uv sync --extra rocm --group dev` |
| `tpu` | Google Cloud TPU | `uv sync --extra tpu --group dev` |

**CUDA 13 is the recommended NVIDIA target.** CUDA 12 is still supported but JAX has
indicated it will be dropped in a future release. Prefer `cuda13` for new setups.

**AMD ROCm** requires local ROCm 7 installation before running the sync; see
[AMD's instructions](https://github.com/jax-ml/jax/blob/main/build/rocm/README.md).

**Apple Silicon GPU** is not yet supported by JAX. Use `cpu` on macOS.

**Intel GPU** support is experimental via a third-party plugin and is not listed as
an official extra. See [intel-extension-for-openxla](https://github.com/intel/intel-extension-for-openxla).

The `dev` dependency group pins `jax[cpu]` so `uv sync --group dev` (no extra flag)
always produces a working environment. GPU/TPU/ROCm developers run this first, then
add their backend extra to swap `jaxlib` out.

**Do not add a backend extra to `[project.dependencies]`.** That section must stay
backend-agnostic so downstream users can install their preferred backend without
a conflict.

---

## Numerical debugging protocol

When a benchmark test fails, use this sequence before changing implementation code:

1. **Check the unit convention first.** Is `E` being divided by `mass_factor` before
   being subtracted from eigenvalues? Are eigenvalues in fm⁻², not MeV?
   See `DESIGN.md §15.4` and `Appendix C.1`.

2. **Check signs.** Run the Yamaguchi test at `E = 0.1 MeV`. The correct answer is
   **negative** (−15.08°). A positive value indicates a sign error in the boundary
   values, the spectral sum denominator, or the Q-matrix construction.
   See `DESIGN.md Appendix C.1`.

3. **Cross-check against the prototype.** `rmatrix.py` (in the repo root or attached
   to the issue) is a known-good reference. Print `TpL`, `Q`, and `γ` side by side
   for `N=5`. Element-by-element agreement to machine precision means the mesh is
   correct; a systematic sign pattern means a boundary-value sign error.

4. **Bisect with `spectral.rmatrix_from_spectrum` vs `rmatrix_direct`.** If both give
   the same wrong answer, the error is upstream (Hamiltonian assembly or boundary
   values). If only one is wrong, the error is in the spectral sum or the linear solve
   respectively.

5. **Increase `mpmath` precision.** For boundary-value failures, temporarily set
   `dps=80` in `compute_boundary_values` and check if the error changes. A change
   indicates a cancellation problem.

---

## What agents should NOT do

- **Do not change test tolerances** to make a failing test pass. Tolerances are
  physics-determined (they match the published reference values in [1] and [2]).
  If a test fails, fix the implementation.
- **Do not add `# noqa` or `# type: ignore` comments** without a precise explanation
  in the same line comment of why the suppression is necessary.
- **Do not add dependencies** to `[project.dependencies]` in `pyproject.toml` without
  explicit approval. The dependency surface is intentionally minimal. In particular:
  never add a JAX backend extra (`jax[cpu]`, `jax[cuda13]`, `jax[rocm7-local]`, etc.) to
  `[project.dependencies]` — backends belong in `[project.optional-dependencies]` or
  the `dev` dependency group only.
- **Do not use `jnp.linalg.inv`** anywhere. Always use `jnp.linalg.solve`.
- **Do not call `mpmath` inside any function decorated with `@jax.jit`** or any
  function called from one. All `mpmath` usage is confined to `boundary/coulomb.py`
  and must stay there.
- **Do not modify `DESIGN.md`** without raising it as a design discussion first.
  The design doc is the source of truth; code is the implementation of it.

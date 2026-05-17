# `lax`: A JAX-Compiled Lagrange-Mesh Library for Quantum Scattering and Bound-State Problems

**Design Document v1.2**

## Revision history

- **v1.2** — Final pre-implementation review. Fixed unit convention bug (`threshold / mass_factor` in `assemble_block_hamiltonian`); fixed `rmatrix_from_spectrum` and `greens_from_spectrum` to convert E from MeV to fm⁻²; replaced undefined `_project_open`/`_pad_back` with full implementation; rewrote `rmatrix_direct` to vmap over compile-time energies; fixed Example 16.7 (energy-dependent V) which was semantically incorrect; split `to_grid` into two explicit functions; added §15.4 unit convention table; added Appendix C (10 implementation sharp edges).
- **v1.1** — Unified all continuum solvers around a single spectral decomposition kernel. R-matrix and Green's function are computed as spectral sums over the eigenpairs of the Bloch-augmented Hamiltonian. Introduced a mesh-independent `spectral` submodule. Added energy-dependent V(E) compile mode with Padé interpolation. Added method-dispatch (`eigh` / `eig` / `linear_solve`) to handle real, complex, and GPU constraints. Linear-solve fallback moved to its own `rmatrix_direct` namespace.
- **v1.0** — Initial design (per-energy linear solves throughout).

---

## Table of contents

1. [Overview and scope](#1-overview-and-scope)
2. [Background](#2-background)
3. [Design goals and constraints](#3-design-goals-and-constraints)
4. [Architecture](#4-architecture)
5. [Module layout](#5-module-layout)
6. [Core types](#6-core-types)
7. [Mesh builders and the registry pattern](#7-mesh-builders-and-the-registry-pattern)
8. [Operators](#8-operators)
9. [Boundary values: Coulomb, Hankel, and Whittaker functions](#9-boundary-values-coulomb-hankel-and-whittaker-functions)
10. [The spectral submodule](#10-the-spectral-submodule)
11. [Solvers and method dispatch](#11-solvers-and-method-dispatch)
12. [Padé interpolation for energy-dependent potentials](#12-padé-interpolation-for-energy-dependent-potentials)
13. [Transforms: grid, Fourier, integration](#13-transforms-grid-fourier-integration)
14. [The `compile()` factory](#14-the-compile-factory)
15. [Coupled-channel structure](#15-coupled-channel-structure)
16. [Public API and usage examples](#16-public-api-and-usage-examples)
17. [JAX considerations](#17-jax-considerations)
18. [Testing strategy and benchmarks](#18-testing-strategy-and-benchmarks)
19. [Build order](#19-build-order)
20. [References](#20-references)
21. [Appendix A: Mesh formula tables](#appendix-a-mesh-formula-tables)
22. [Appendix B: Glossary of symbols](#appendix-b-glossary-of-symbols)
23. [Appendix C: Implementation sharp edges](#appendix-c-implementation-sharp-edges)

---

## 1. Overview and scope

`lax` is a JAX-based library implementing the Lagrange-mesh method (LMM) for numerically solving the radial Schrödinger equation. It supports both bound states (eigenvalue problems) and continuum states (R-matrix, S-matrix, Green's function) through a single unified spectral-decomposition kernel. The library is designed as a low-level numerical engine that plugs into reaction codes, fitting pipelines, and uncertainty-quantification workflows.

The library targets three classes of users:

1. **Reaction-code authors** who need a fast, JIT-compiled scattering kernel they can call from inside larger calculations (CDCC, optical model, DWBA, transfer reactions).
2. **Researchers fitting potentials to data** who need gradients through the entire scattering calculation so optimizers and HMC samplers work natively.
3. **Practitioners running parameter scans** who want to push a batch of potentials or energy grids through the same precompiled mesh on a GPU in one call.

### Core technical strategy

The library separates a problem into two phases:

- **Compile time** (Python with NumPy, scipy, mpmath; runs once per problem definition): construct the mesh, precompute kinetic and operator matrices, evaluate Coulomb and Whittaker boundary values at the user's energy grid, JIT-compile the requested solver kernels.
- **Runtime** (pure JAX; runs many times): push potentials through the precompiled kernels. The kernels return either a `Spectrum` object (one eigendecomposition per V) or, in the linear-solve fallback, R-matrices directly. Everything is JIT-compatible, vmap-compatible, and grad-compatible.

The **spectral kernel is the runtime currency.** A `Spectrum` from `solver.spectrum(V)` is the input to all observables: R-matrix and Green's function are spectral sums; the S-matrix follows from the R-matrix and the precomputed boundary values; phase shifts follow from the S-matrix. One eigendecomposition supports every observable at every energy — a dramatic speedup over per-energy linear solves and an architectural simplification.

### What this document covers

The remainder of the document specifies the architecture, module layout, core data types, all numerical formulas needed to implement each mesh and solver, the public API, and a test plan. It is intended to be sufficient for offline implementation. It is written assuming familiarity with the Lagrange-mesh method as reviewed by Baye [1] and the R-matrix formalism on Lagrange meshes as implemented by Descouvemont [2].

---

## 2. Background

### 2.1 The Lagrange-mesh method in one page

A Lagrange mesh consists of three things: (i) a set of $N$ mesh points $x_i$, typically the zeros of an orthogonal polynomial; (ii) Gauss-quadrature weights $\lambda_i$ associated with these points; (iii) a set of $N$ basis functions $f_j(x)$ — the *Lagrange functions* — satisfying the Lagrange condition

$$f_i(x_j) = \lambda_i^{-1/2}\, \delta_{ij} \qquad \text{[Baye eq. 2.10]} \tag{1}$$

orthonormal at the consistent Gauss quadrature [1, §2.2]. A wave function is expanded as

$$\psi(x) = \sum_{j=1}^{N} c_j\, f_j(x), \qquad c_j = \lambda_j^{1/2}\, \psi(x_j) \qquad \text{[Baye eqs. 2.25, 2.26]} \tag{2}$$

so expansion coefficients are weight-scaled wave-function samples.

The miracle of the LMM is that with the consistent Gauss approximation the potential matrix becomes **diagonal** [1, eq. 2.28]:

$$V_{ij} \approx V(x_i)\, \delta_{ij} \tag{3}$$

For a non-local potential with kernel $W(x, x')$ [1, eq. 2.30]:

$$W_{ij} \approx (\lambda_i \lambda_j)^{1/2}\, W(x_i, x_j) \tag{4}$$

No quadrature, no analytical integrals. Convergence is exponential in $N$ for sufficiently smooth potentials.

### 2.2 Mesh families and regularization

Within each orthogonal-polynomial family, *regularizations* address singularities at endpoints by multiplying basis functions by a small smooth factor. Each (family, regularization) pair has its own explicit formulas in [1]. The library implements:

| Family | Regularization | Interval | Primary use |
|---|---|---|---|
| Legendre, shifted | by $x$ | $(0, a)$ | R-matrix on finite interval ([1, §3.4.5], [2, eq. 18]) |
| Legendre, shifted | by $x(1-x)$ | $(0, a)$ | Confined systems ([1, §3.4.7]) |
| Legendre, shifted | by $x^{3/2}$ | $(0, a)$ | Hyperspherical coordinates ([1, §3.4.6]) |
| Laguerre | by $x$ | $(0, \infty)$ | Radial Schrödinger; hydrogen ([1, §3.3.4]) |
| Laguerre | by $x^{3/2}$ | $(0, \infty)$ | Three-body hyperradial ([1, §3.3.5]) |
| Laguerre, modified ($t = x^2$) | by $x$ | $(0, \infty)$ | Harmonic-oscillator-like ([1, §3.3.7]) |

A regularization replaces $f_j(x)$ with $\hat f_j(x) = (R(x)/R(x_j))\, f_j(x)$ [1, eq. 2.58]. The regularized basis is no longer exactly orthonormal, but is orthonormal at the Gauss approximation, and the LMM accuracy survives essentially intact ([1, §2.7]).

### 2.3 R-matrix theory on Lagrange meshes

The R-matrix method [2, §2] divides configuration space into an internal region $r \in [0, a]$ and an external region $r > a$. The kinetic-energy operator is not Hermitian on a finite interval; the Bloch surface operator [2, eq. 8]

$$L(B_i) = \frac{\hbar^2}{2\mu}\, \delta(r - a)\!\left(\frac{d}{dr} - \frac{B_i}{r}\right) \tag{5}$$

restores Hermiticity. For an $N_c$-channel system one assembles [2, eq. 14]

$$C_{in, jm}(E) = \big\langle \varphi_n \big| (T_i + L(B_i) + E_i - E)\delta_{ij} + V_{ij} \big| \varphi_m \big\rangle_{\text{int}} \tag{6}$$

and the R-matrix is the surface projection of $C^{-1}$ [2, eq. 15]:

$$R_{ij}(E) = \frac{\hbar^2}{2\mu a} \sum_{n,m} \varphi_n(a)\, [C(E)^{-1}]_{in,jm}\, \varphi_m(a) \tag{7}$$

The collision (S-)matrix follows from the matching condition [2, eqs. 16–17]:

$$U(E) = Z_O(E)^{-1} Z_I(E), \qquad (Z_O)_{ij} = (k_j a)^{-1/2}\big[ O_{L_i}(k_i a)\delta_{ij} - k_j a\, R_{ij}\, O'_{L_j}(k_j a) \big] \tag{8}$$

For the shifted Legendre mesh regularized by $x$ with $\nu = 1$, the basis functions take the explicit form [2, eq. 18]

$$\varphi_n(r) = (-1)^{N+n} \left(\frac{r}{ax_n}\right) \sqrt{a x_n(1-x_n)}\, \frac{P_N(2r/a - 1)}{r - ax_n} \tag{9}$$

with boundary values [2, eq. 24]

$$\varphi_n(a) = (-1)^{N+n} \sqrt{\frac{1}{a\, x_n(1 - x_n)}} \tag{10}$$

and the $T + L(B)$ matrix elements [2, eqs. 22–23] given in Appendix A.

### 2.4 The spectral form of the R-matrix

The key identity that drives this library's architecture is the spectral decomposition of $C(E)^{-1}$. Write the Bloch-augmented Hamiltonian (in matrix form, including channel structure):

$$H = T + L + V_{\text{coupling}} + (\text{thresholds}), \qquad C(E) = H - E\,\mathbb{I}$$

Eigendecompose $H$ once:

$$H = U \operatorname{diag}(\varepsilon) U^T \qquad (\text{Hermitian: } U^T \to U^\dagger) \tag{11}$$

Then for *every* energy:

$$C(E)^{-1} = \sum_k \frac{|u_k\rangle \langle u_k|}{\varepsilon_k - E} \tag{12}$$

Define the **surface amplitudes** (the Lagrange-mesh analog of reduced-width amplitudes):

$$\gamma_{kc} = \sum_n \varphi_n(a)\, u_{k,(c,n)} = (U^T Q)_{kc} \tag{13}$$

where $Q$ is the $(N_c N) \times N_c$ block-diagonal "surface picker" matrix with $\varphi_n(a)$ on the diagonal of each block. The R-matrix is then

$$\boxed{R_{cc'}(E) = \frac{1}{a}\sum_k \frac{\gamma_{kc}\, \gamma_{kc'}}{\varepsilon_k - E}} \tag{14}$$

This is the classical Wigner–Eisenbud expansion: $\varepsilon_k$ are R-matrix poles, $\gamma_{kc}$ are reduced-width amplitudes. The Green's function is

$$G_{nm}(E) = \sum_k \frac{u_{kn} u_{km}}{\varepsilon_k - E} \tag{15}$$

(with appropriate conjugation in the Hermitian case). Bound states are eigenpairs $(\varepsilon_k, u_k)$ with $\varepsilon_k$ below all thresholds.

**Architectural consequence.** All scattering and bound-state observables derive from one eigendecomposition. The energy axis is a tensor contraction, not a linear-algebra loop.

---

## 3. Design goals and constraints

### Goals

1. **Spectral kernel is the default runtime primitive.** `solver.spectrum(V)` returns a `Spectrum` object; all observables (R, S, G, phases, bound states, wavefunctions) are pure functions of that object plus (for matching-dependent quantities) precomputed boundary values.
2. **Generality across mesh families.** Both Legendre and Laguerre families with their main regularizations supported via a single registry. Adding a new family or regularization requires writing one function.
3. **Multiple solver modes from one mesh.** A single compiled solver bundle supports eigenvalue calculations, R-matrix calculations, S-matrix evaluation, scattering wavefunctions, and Green's-function evaluation.
4. **Two energy modes.** Energy-independent V(E) (compile a spectrum-producing kernel; observables evaluable at any E for spectrum-derived quantities, at the compile-time grid for boundary-value-dependent quantities). Energy-dependent V(E) (user supplies V at each grid point; library produces observables at the grid and offers Padé interpolation between grid points).
5. **Arbitrary user-supplied potentials.** Local $V(r)$ and non-local $W(r, r')$, real or complex, single or coupled channel.
6. **Mesh-independent spectral submodule.** Spectral storage, sums, and interpolation live in `lax.spectral` and depend on nothing in the rest of the package.
7. **Fine-grid / momentum-space / integration helpers.** Conversion of mesh vectors and matrices to finer radial grids or momentum space is precomputed matrix multiplication; integration is trivial in the Lagrange-mesh basis.
8. **Full JAX integration.** Everything inside the runtime hot path is `jit`-, `vmap`-, and `grad`-compatible. Pytree registration explicit and minimal. No `equinox`/`flax` dependency.
9. **Method dispatch for the complex / GPU case.** Real V uses `eigh` (GPU-ready). Complex V uses `eig` (CPU host callback) or a `linear_solve` fallback (GPU+vmap-ready but only computes R-matrix-derived quantities). Complex-symmetric Lanczos in JAX is a future enhancement.
10. **Extensive benchmark coverage.** Yamaguchi non-local, hydrogen atom, 3D harmonic oscillator, confined hydrogen, Pöschl-Teller, Coulomb scattering, α + ²⁰⁸Pb optical, multi-channel n-p — all reproducing published reference values.

### Constraints

**Coulomb and Whittaker functions are not JIT-able.** The Coulomb regular and irregular functions $F_L(\eta, \rho)$, $G_L(\eta, \rho)$ and the Whittaker function $W_{-\eta, \ell+1/2}(\rho)$ are needed at $r = a$ for every channel and every energy. We use `mpmath` for arbitrary-precision evaluation. Since `mpmath` is pure Python, we evaluate boundary values once at compile time over a user-specified energy grid, stack them into JAX arrays of shape `(N_E, N_c)`, and embed them in the `Solver` bundle.

**Consequences.** The energy grid for boundary-value-dependent quantities (S-matrix, scattering wavefunctions, phase shifts) is fixed at compile time. R-matrix and Green's function — which are pure functions of the spectrum — can be evaluated at any runtime energy. Recomputing for a different energy grid means rebuilding the solver; this is cheap relative to the JIT trace time and `mpmath` calls take milliseconds each.

**`jnp.linalg.eig` is CPU-only in current JAX**, which constrains the complex-potential path. See §11 for the method-dispatch policy.

**Non-Hermitian Lanczos in JAX would benefit large complex problems but is non-trivial.** Listed as future work; the v1 fallback for the GPU+complex case is per-energy linear solves (R-matrix only).

**Compiled solvers must be round-trip serializable.** `compile()` is expensive enough
that users must be able to cache a `Solver` across Python processes or sessions. The
preferred contract is stdlib `pickle`; `dill` is an acceptable fallback only if a
specific JAX runtime object proves impossible to serialize cleanly with stdlib tools.
This constrains the implementation: the `Solver` bundle may not store local closures or
other non-importable callables. Bound runtime entry points must be represented by
module-level callable objects and/or explicit reconstruction logic so the solver's
cached mesh/operator/boundary state survives a serialization round trip.

### Non-goals (for v1)

- Cross-section calculation (user computes from S).
- Wave-function propagation across subintervals [2, §2.4]. Future enhancement.
- Three-body hyperspherical solvers [1, §7]. Architecture does not preclude them but they are not implemented.
- Hermite, Jacobi, Fourier, sinc meshes [1, §3.2, §3.5, §3.7]. The registry accommodates them; only Legendre and Laguerre are in v1.
- GPU-ready complex eigendecomposition. v1 uses host callbacks for `eig`; future work may add complex-symmetric Lanczos in pure JAX.

---

## 4. Architecture

### 4.1 Compile time vs runtime

The library has two distinct phases.

**Compile time** runs in plain Python with NumPy and `mpmath`. The user calls `lax.compile(...)` once, specifying:

- Mesh family, regularization, size, scale.
- Channel structure (per-channel $\ell$, threshold, mass factor).
- Energy grid (required if any boundary-value-dependent observable is requested).
- Energy mode (independent vs dependent).
- Operators to precompute (`T+L`, `1/r`, `1/r²`, ...).
- Solvers to bake (`spectrum`, `smatrix`, `phases`, `greens`, `rmatrix_direct`).
- Optional fine radial grid (for `to_grid`) and momentum grid (for `fourier`).
- Method (`"eigh"`, `"eig"`, `"linear_solve"`) and target device.

The compile step:

1. Builds the mesh via the registry.
2. Precomputes operator matrices (kinetic, position, derivative).
3. Calls `mpmath` to evaluate Coulomb $F, G, F', G'$ on open channels and Whittaker $W, W'$ on closed channels, at every $(E, c)$ pair.
4. Optionally precomputes basis-evaluation matrices for grid and momentum transforms.
5. Constructs the requested solver kernels as closures over the cached data, JIT-compiles them, and places them on the target device.
6. Returns a `Solver` pytree.

**Runtime** runs inside JAX. The user calls solver methods with potential data:

- `spectrum = solver.spectrum(V)` runs one eigendecomposition. Returns a `Spectrum` pytree.
- `R = solver.rmatrix(spectrum, E)` spectral sum at any scalar E.
- `S = solver.smatrix(spectrum)` returns shape `(N_E, N_c, N_c)` using cached boundary values.
- `G = solver.greens(spectrum, E)` spectral sum.
- `e, u = solver.eigh(spectrum)` raw access for bound states.
- `R = solver.rmatrix_direct(V, E)` (alternative kernel; per-E linear solve; bypasses Spectrum).

Energy-dependent V is the same flow with a `vmap` on the user side over the grid, plus Padé interpolation utilities.

### 4.2 The spectral form drives everything

The runtime data flow is:

```
V  ──spectrum──▶  Spectrum  ──┬──▶  rmatrix(E)         ──▶  R(E)
                              ├──▶  greens(E)          ──▶  G(E)
                              ├──▶  wavefunction(E)    ──▶  ψ_int(E)
                              ├──▶  eigh()             ──▶  (ε, u)
                              └──▶  smatrix()  + boundary ──▶  S
                                                              │
                                                              └──▶ phases  ──▶  δ
```

`Spectrum` is the central pytree. All downstream observables are pure functions in `lax.spectral` of `Spectrum` plus (for boundary-dependent ones) `BoundaryValues`. The library's other modules build, decorate, and combine these.

### 4.3 The `Solver` bundle

`Solver` is a plain frozen dataclass (not a pytree, because it holds callables). It carries:

```
Solver
├── mesh: Mesh                       # nodes, weights, radii, basis_at_boundary
├── operators: OperatorMatrices      # cached single-channel matrices
├── channels: tuple[ChannelSpec]     # static, baked into JIT
├── energies: jnp.ndarray            # (N_E,) compile-time grid
├── boundary: BoundaryValues         # (N_E, N_c) cached Coulomb/Whittaker
├── transforms: TransformMatrices    # optional B_grid, F_momentum
├── method: str                      # "eigh" | "eig" | "linear_solve"
└── (callables bound at compile time):
    ├── spectrum(V)        -> Spectrum                # spectrum-method only
    ├── rmatrix(spec, E)   -> R(E)
    ├── smatrix(spec)      -> S at compile-time E
    ├── phases(spec)       -> δ at compile-time E
    ├── greens(spec, E)    -> G(E)
    ├── wavefunction(spec, E, source) -> ψ_int(E)
    ├── eigh(spec)         -> (ε, u) accessor
    ├── rmatrix_direct(V, E)  -> R          # linear_solve namespace
    ├── to_grid(...)
    ├── fourier(...)
    └── integrate(...)
```

The bound runtime callables are implemented as **module-level callable objects**, not
local closures, so a compiled solver can be serialized and restored. For
energy-independent V, `solver.spectrum(V)` is called once per V. For energy-dependent
V, the user wraps it in `jax.vmap` over their per-grid-point V batch.

---

## 5. Module layout

```
lax/
├── __init__.py            # Public API: compile, MeshSpec, ChannelSpec, ...
├── compile.py             # The compile() factory; main entry point
├── types.py               # Pytree dataclasses: Mesh, OperatorMatrices, Solver
│
├── meshes/
│   ├── __init__.py
│   ├── _registry.py       # (family, regularization) -> builder dispatch
│   ├── legendre.py        # Shifted Legendre: x, x(1-x), x^{3/2}
│   ├── laguerre.py        # Laguerre: x, x^{3/2}, modified-x^2
│   ├── _basis_eval.py     # f_j(x) evaluation for grid/Fourier transforms
│   └── _quadrature.py     # NumPy node/weight tables
│
├── operators/
│   ├── __init__.py
│   ├── kinetic.py         # T, T+L, T_alpha matrix builders
│   ├── derivative.py      # D = d/dr
│   ├── position.py        # x, x², 1/x, 1/x²
│   └── potential.py       # local/nonlocal V assemblers
│
├── boundary/
│   ├── __init__.py
│   ├── coulomb.py         # mpmath F, G, F', G' (open channels)
│   ├── whittaker.py       # mpmath W, W' (closed channels)
│   └── _types.py          # BoundaryValues
│
├── spectral/              # ── MESH-INDEPENDENT submodule ──
│   ├── __init__.py
│   ├── types.py           # Spectrum dataclass (pytree)
│   ├── observables.py     # rmatrix_from_spectrum, greens_from_spectrum, ...
│   ├── matching.py        # smatrix_from_R, phases_from_S
│   ├── interpolation.py   # pade_interpolate
│   └── tests/             # spectral submodule has its own tests
│
├── solvers/
│   ├── __init__.py
│   ├── spectrum.py        # The spectrum kernel: eigh/eig dispatch
│   ├── linear_solve.py    # Per-energy linear-solve fallback (rmatrix_direct)
│   ├── assembly.py        # Block-Hamiltonian assembly from operators + V
│   └── wavefunction.py    # Scattering wavefunctions (internal + external)
│
├── transforms/
│   ├── __init__.py
│   ├── grid.py            # mesh <-> radial grid
│   ├── fourier.py         # mesh <-> momentum grid (Bessel transforms)
│   └── integration.py     # norms, expectation values
│
├── propagate.py           # (future) R-matrix subinterval propagation
│
└── tests/
    ├── benchmarks/        # Yamaguchi, hydrogen, HO, confined H, α+²⁰⁸Pb, ...
    ├── unit/              # Per-builder mesh/operator unit tests
    └── property/          # Hermiticity, unitarity, autograd, vmap parity
```

### Dependencies

- **Required**: `jax`, `jaxlib`, `numpy`, `scipy`, `mpmath`.
- **Test**: `pytest`, `hypothesis`, `chex` (optional, for shape/dtype assertions).

No `equinox`, no `flax`. Pytree registration uses `jax.tree_util.register_dataclass`.

The library requires `jax>=0.4.36`, which introduced optional `data_fields`/`meta_fields` arguments to `register_dataclass` for `@dataclass` inputs. Fields default to pytree leaves (data fields) unless annotated with `field(metadata={"static": True})`, which marks them as static metadata baked into the JIT cache key. This is the form used throughout the library.

---

## 6. Core types

All public dataclasses are frozen. Numerical-data dataclasses are JAX pytrees. `Solver` is plain (holds callables).

```python
# types.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Literal
import jax
import jax.numpy as jnp


# ---------------------------------------------------------------- Mesh kinds

MeshFamily = Literal["legendre", "laguerre", "hermite", "jacobi"]
Regularization = Literal[
    "none", "x", "x^3/2", "x(1-x)", "sqrt(1-x^2)", "modified_x^2",
]
Method = Literal["eigh", "eig", "linear_solve", "lanczos"]


# ----------------------------------------------------------- Mesh specification

@dataclass(frozen=True)
class MeshSpec:
    """User-facing spec for a mesh. Passed to compile()."""
    family: MeshFamily
    regularization: Regularization
    n: int
    scale: float                # `a` for finite, `h` for Laguerre
    extras: dict = field(default_factory=dict)


# ----------------------------------------------------------- Mesh data (pytree)

@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Mesh:
    family: str            = field(metadata={"static": True})
    regularization: str    = field(metadata={"static": True})
    n: int                 = field(metadata={"static": True})
    scale: float           = field(metadata={"static": True})

    nodes:               jnp.ndarray   # (N,) on canonical interval
    weights:             jnp.ndarray   # (N,) λ_i
    radii:               jnp.ndarray   # (N,) physical r_i
    basis_at_boundary:   jnp.ndarray   # (N,) φ_j(a); zeros for unbounded


# ------------------------------------------------------------- Operator cache

@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class OperatorMatrices:
    """Precomputed single-channel (N, N) matrices in units of ℏ²/2μ.
    Unrequested operators are None."""
    T:        jnp.ndarray | None = None
    TpL:      jnp.ndarray | None = None
    T_alpha:  jnp.ndarray | None = None
    D:        jnp.ndarray | None = None
    inv_r:    jnp.ndarray | None = None
    inv_r2:   jnp.ndarray | None = None


# ----------------------------------------------------------- Channel structure

@dataclass(frozen=True)
class ChannelSpec:
    l: int
    threshold: float
    mass_factor: float = 1.0    # ℏ²/2μ in user units (e.g. MeV·fm²)


# ----------------------------------------------------------- Boundary values

@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class BoundaryValues:
    """Coulomb / Whittaker boundary values at r = a, for every (E, c).
    
    For open channels (E > E_c):
        H_± = G(ka) ± iF(ka)             (outgoing/incoming Coulomb)
        H'_± = (ρ d/dρ)(G ± iF) at ρ=ka
    For closed channels (E < E_c):
        H_+ = H_- = W_{-η, ℓ+1/2}(2|k|a)  (Whittaker function value)
        H'_± = (ρ d/dρ) W at ρ=2|k|a       (used to construct B_c)
    
    The is_open mask routes downstream solvers to the right matching path.
    """
    H_plus:    jnp.ndarray   # (N_E, N_c) complex
    H_minus:   jnp.ndarray   # (N_E, N_c) complex
    H_plus_p:  jnp.ndarray   # (N_E, N_c) complex
    H_minus_p: jnp.ndarray   # (N_E, N_c) complex
    is_open:   jnp.ndarray   # (N_E, N_c) bool


# ------------------------------------------------------ Transform matrices

@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class TransformMatrices:
    B_grid:     jnp.ndarray | None = None  # (M_r, N)
    grid_r:     jnp.ndarray | None = None  # (M_r,)
    F_momentum: jnp.ndarray | None = None  # (N_c, M_k, N) — per-l
    momenta:    jnp.ndarray | None = None  # (M_k,)


# ----------------------------------------------------------- Solver bundle

@dataclass(frozen=True)
class Solver:
    """Output of compile(). Holds cached data and JIT'd callables."""
    mesh: Mesh
    operators: OperatorMatrices
    channels: tuple[ChannelSpec, ...]
    energies: jnp.ndarray
    boundary: BoundaryValues | None
    transforms: TransformMatrices
    method: str

    # Callables (filled in by compile()); None if not requested.
    spectrum:        Callable | None = None
    rmatrix:         Callable | None = None
    smatrix:         Callable | None = None
    phases:          Callable | None = None
    greens:          Callable | None = None
    wavefunction:    Callable | None = None
    eigh:            Callable | None = None
    rmatrix_direct:  Callable | None = None   # linear-solve namespace
    to_grid:         Callable | None = None
    fourier:         Callable | None = None
    integrate:       Callable | None = None
```

### `Spectrum` lives in the spectral submodule

```python
# spectral/types.py

@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Spectrum:
    """Spectral decomposition of the Bloch-augmented Hamiltonian.
    
    For Hermitian H (real V):       H = U diag(ε) U†,   U†U = I
    For complex-symmetric H (cplx): H = U diag(ε) U^T,  U^T U = I  (bilinear)
    
    Surface amplitudes γ_kc = (U^T Q)_kc — the projection onto the
    boundary "Q vector". Sufficient on its own for R-matrix and S-matrix.
    Full eigenvectors U needed only for Green's functions and wavefunctions.
    """
    eigenvalues:        jnp.ndarray              # (M,)
    surface_amplitudes: jnp.ndarray              # (M, N_c) — γ_kc
    eigenvectors:       jnp.ndarray | None       # (M, M) — optional
    is_hermitian:       bool = field(metadata={"static": True})
```

At compile time, the solver knows whether Green's or wavefunctions were requested and bakes that into a `keep_eigenvectors: bool` flag inside the JIT'd kernel. This avoids wasting memory and time on `U` when only R/S/phases are wanted.

---

## 7. Mesh builders and the registry pattern

### 7.1 The registry

```python
# meshes/_registry.py
from typing import Callable
from ..types import Mesh, OperatorMatrices

Builder = Callable[..., tuple[Mesh, OperatorMatrices]]
_BUILDERS: dict[tuple[str, str], Builder] = {}


def register(family: str, regularization: str):
    """Decorator to register a mesh+regularization builder."""
    def deco(fn: Builder) -> Builder:
        key = (family, regularization)
        if key in _BUILDERS:
            raise ValueError(f"Builder already registered for {key}")
        _BUILDERS[key] = fn
        return fn
    return deco


def build_mesh(
    family: str,
    regularization: str,
    n: int,
    scale: float,
    operators: set[str],
    **extras,
) -> tuple[Mesh, OperatorMatrices]:
    """Dispatch to the appropriate builder."""
    key = (family, regularization)
    if key not in _BUILDERS:
        raise ValueError(f"No builder for {key}. Available: {sorted(_BUILDERS)}")
    return _BUILDERS[key](n=n, scale=scale, operators=operators, **extras)
```

### 7.2 Shifted Legendre regularized by $x$ — the R-matrix workhorse

Formulas from [2, §2.2] and [1, §3.4.5].

```python
# meshes/legendre.py
import numpy as np
import jax.numpy as jnp
from scipy.special import roots_legendre

from ._registry import register
from ..types import Mesh, OperatorMatrices


@register("legendre", "x")
def build_legendre_x(
    *,
    n: int,
    scale: float,
    operators: set[str],
    **extras,
) -> tuple[Mesh, OperatorMatrices]:
    """Shifted Legendre on (0, a) regularized by x.
    
    Nodes:    P_N(2x_i - 1) = 0 on (0, 1); r_i = a x_i      [Baye eq. 3.120]
    Weights:  λ̂_i = 1/[4 x_i(1-x_i) P'_N(2x_i-1)²]          [Baye eq. 3.121]
    Basis:    φ_j(r) from Descouvemont eq. 18, ν=1
    Boundary: φ_n(a) = (-1)^{N+n} sqrt(1/[a x_n(1-x_n)])    [Desc. eq. 24]
    T+L(0):   diagonal Desc. eq. 22; off-diag Desc. eq. 23
    """
    a = float(scale)

    # Nodes/weights on (0, 1)
    x_raw, w_raw = roots_legendre(n)
    x = 0.5 * (x_raw + 1.0)
    lam = 0.5 * w_raw
    r = a * x

    # Boundary basis values φ_n(a)  [Desc. eq. 24]
    sign = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    if n % 2 == 1:
        sign = -sign
    boundary = sign / np.sqrt(a * x * (1 - x))

    # T + L(B=0)  [Desc. eqs. 22, 23]
    TpL = np.zeros((n, n))
    np.fill_diagonal(
        TpL,
        ((4 * n * (n + 1) + 3) * x * (1 - x) - 6 * x + 1) / (3 * x * (1 - x)),
    )
    i, j = np.triu_indices(n, k=1)
    val = (
        n * (n + 1) + 1
        + (x[i] + x[j] - 2 * x[i] * x[j]) / (x[i] - x[j]) ** 2
        - 1.0 / (1 - x[i]) - 1.0 / (1 - x[j])
    )
    val /= np.sqrt(x[i] * (1 - x[i]) * x[j] * (1 - x[j]))
    val *= np.where((i + j) % 2 == 1, -1.0, 1.0)
    TpL[i, j] = val
    TpL[j, i] = val
    TpL = TpL / (a ** 2)

    mesh = Mesh(
        family="legendre", regularization="x",
        n=n, scale=a,
        nodes=jnp.asarray(x),
        weights=jnp.asarray(lam),
        radii=jnp.asarray(r),
        basis_at_boundary=jnp.asarray(boundary),
    )

    ops_kwargs = {"TpL": jnp.asarray(TpL)}
    if "1/r" in operators:
        ops_kwargs["inv_r"] = jnp.diag(jnp.asarray(1.0 / r))
    if "1/r^2" in operators:
        ops_kwargs["inv_r2"] = jnp.diag(jnp.asarray(1.0 / r ** 2))
    if "D" in operators:
        ops_kwargs["D"] = jnp.asarray(_legendre_x_derivative(x, a))

    return mesh, OperatorMatrices(**ops_kwargs)


def _legendre_x_derivative(x: np.ndarray, a: float) -> np.ndarray:
    """d/dr matrix [Baye eqs. 3.123, 3.124], scaled to physical units."""
    n = len(x)
    D = np.zeros((n, n))
    np.fill_diagonal(D, 1.0 / (2.0 * x * (1.0 - x)))
    for i in range(n):
        for j in range(n):
            if i == j: continue
            sign = (-1) ** (i - j)
            D[i, j] = sign * np.sqrt(
                x[i] * (1 - x[j]) / (x[j] * (1 - x[i]))
            ) / (x[i] - x[j])
    return D / a
```

### 7.3 Laguerre regularized by $x$

Formulas from [1, §3.3.4].

```python
# meshes/laguerre.py
import numpy as np
import jax.numpy as jnp
from scipy.special import roots_laguerre

from ._registry import register
from ..types import Mesh, OperatorMatrices


@register("laguerre", "x")
def build_laguerre_x(
    *,
    n: int,
    scale: float,
    operators: set[str],
    alpha: float = 0.0,
    **extras,
) -> tuple[Mesh, OperatorMatrices]:
    """Laguerre on (0,∞) regularized by x, scale r = h x.
    
    Nodes:    L_N^α(x_i) = 0                                [Baye eq. 3.50]
    Weights:  Baye eq. 3.51
    T̂ Gauss: diag eq. 3.76; off-diag eq. 3.75 (α=0)
    
    For α=0 the regularized Laguerre mesh gives Gauss-exact 1/r matrix
    elements [Baye eq. 3.61], making it ideal for Coulomb-type problems.
    """
    if alpha != 0.0:
        raise NotImplementedError("Only α=0 supported in v1.")
    h = float(scale)

    x, w = roots_laguerre(n)
    lam = w * np.exp(x)           # Baye λ̂_i
    r = h * x

    TpL = np.zeros((n, n))
    np.fill_diagonal(
        TpL,
        -(x ** 2 - 2 * (2 * n + 1) * x - 4) / (12 * x ** 2),
    )
    i, j = np.triu_indices(n, k=1)
    val = (x[i] + x[j]) / (np.sqrt(x[i] * x[j]) * (x[i] - x[j]) ** 2)
    val *= np.where((i - j) % 2 == 1, -1.0, 1.0)
    TpL[i, j] = val
    TpL[j, i] = val
    TpL = TpL / (h ** 2)

    mesh = Mesh(
        family="laguerre", regularization="x",
        n=n, scale=h,
        nodes=jnp.asarray(x),
        weights=jnp.asarray(lam),
        radii=jnp.asarray(r),
        basis_at_boundary=jnp.zeros(n),   # no finite boundary
    )

    ops_kwargs = {"TpL": jnp.asarray(TpL), "T": jnp.asarray(TpL)}
    if "1/r" in operators:
        ops_kwargs["inv_r"] = jnp.diag(jnp.asarray(1.0 / r))
    if "1/r^2" in operators:
        ops_kwargs["inv_r2"] = jnp.diag(jnp.asarray(1.0 / r ** 2))

    return mesh, OperatorMatrices(**ops_kwargs)
```

### 7.4 Adding a new mesh

To add Hermite, Jacobi, or another regularization, write a builder and decorate it:

```python
@register("hermite", "none")
def build_hermite(*, n, scale, operators, **extras):
    # ... compute T, basis, weights per Baye §3.2 ...
    return mesh, ops
```

The new builder appears automatically in `lax.compile(MeshSpec(family="hermite", ...))`. No other file changes.

Basis-function evaluation for `to_grid` and `fourier` is also family-specific and registered in a parallel registry in `meshes/_basis_eval.py`.

---

## 8. Operators

Operators are pre-built matrices stored in `OperatorMatrices`. Each (mesh, regularization) builder fills in the matrices it supports; the formulas are in [1, §3] and Appendix A here.

**Kinetic and Bloch-augmented kinetic.** For Legendre-$x$, `operators.TpL` is the matrix [Desc. eq. 22–23] with $B = 0$. For non-zero $B$, the library updates per channel as $T + L(B) = T + L(0) - B\, b b^T / a$ where $b = $ `basis_at_boundary`. Closed channels use $B_c = 2 k_c a\, W'/W$ [Desc. eq. 9] computed from the cached Whittaker values.

For Laguerre, basis functions vanish at both endpoints, so $T$ is Hermitian on its own; `TpL` and `T` coincide.

**Position operators.** For Legendre-$x$ and Laguerre-$x$ ($\alpha=0$), $1/r$ is Gauss-exact diagonal [Baye eqs. 3.61, 3.140 analog]. $1/r^2$ likewise diagonal [Baye eq. 3.62].

**Centrifugal term.** $\ell(\ell+1)/r^2$ is injected by the Hamiltonian assembler, per channel. The user does not add it manually.

**Derivative.** $D_{ij} = \langle f_i | d/dr | f_j \rangle$ in closed form per family. Useful for momentum-like observables and Bloch off-diagonals if needed.

**Local potential injection.** A local $V_c(r)$ in channel $c$ enters the Hamiltonian as a diagonal block with entries $V_c(r_i)/\mu_c$ (where $\mu_c$ is the channel's $\hbar^2/2\mu_c$ factor).

**Non-local potential injection.** A non-local $W_{cc'}(r, r')$ enters as a full block with entries $(\lambda_i \lambda_j)^{1/2}\, a\, W_{cc'}(r_i, r_j) / \mu_c$ [Desc. eq. 26].

A helper handles the scaling:

```python
# operators/potential.py
import jax.numpy as jnp

def assemble_local(mesh, V_callable, n_channels=1):
    """V[c, c', i] = V_cc'(r_i). Returns (N_c, N_c, N)."""
    r = mesh.radii
    if n_channels == 1:
        return V_callable(r)[None, None, :]
    return jnp.stack([
        jnp.stack([V_callable(r, c, cp) for cp in range(n_channels)])
        for c in range(n_channels)
    ])

def assemble_nonlocal(mesh, W_callable, n_channels=1):
    """W[c, c', i, j] = (λ_i λ_j)^{1/2} a W_cc'(r_i, r_j). Returns (N_c, N_c, N, N)."""
    r, lam = mesh.radii, mesh.weights
    ri, rj = jnp.meshgrid(r, r, indexing="ij")
    wi, wj = jnp.meshgrid(lam, lam, indexing="ij")
    scale = jnp.sqrt(wi * wj) * mesh.scale
    if n_channels == 1:
        return (W_callable(ri, rj) * scale)[None, None, :, :]
    blocks = []
    for c in range(n_channels):
        row = [W_callable(ri, rj, c, cp) * scale for cp in range(n_channels)]
        blocks.append(jnp.stack(row))
    return jnp.stack(blocks)
```

---

## 9. Boundary values: Coulomb, Hankel, and Whittaker functions

The Coulomb regular $F_L(\eta, \rho)$, irregular $G_L(\eta, \rho)$, and Whittaker $W_{-\eta, \ell+1/2}(\rho)$ functions are needed at $r = a$ for every channel and every energy in the compile-time grid. We use `mpmath` with `dps = 40` by default; this is overkill for double precision but cheap insurance against cancellation when subtracting $H_-$ from $R H_-'$ near sharp resonances.

### Implementation

```python
# boundary/coulomb.py
import mpmath as mp
import numpy as np
import jax.numpy as jnp

from ..types import BoundaryValues, ChannelSpec


def compute_boundary_values(
    channels: tuple[ChannelSpec, ...],
    energies: np.ndarray,        # (N_E,) MeV
    channel_radius: float,        # fm
    z1z2: tuple[int, int] | None = None,
    dps: int = 40,
) -> BoundaryValues:
    """Compute boundary values at r = a for every (E, c).
    
    Open channels (E_rel = E - E_c > 0):
        H_+ = G + iF              (outgoing)
        H_- = G - iF              (incoming)
        H'_± = (ρ d/dρ)(G ± iF) at ρ = ka
    
    Closed channels (E_rel < 0): we store the Whittaker function and its
    log-derivative; downstream solvers convert these to the Bloch parameter
    B_c [Desc. eq. 9] and use the matching scheme of Desc. §2.1.
        H_+ = H_- = W_{-η, ℓ+1/2}(2|k|a)
        H'_± = ρ W'/W  · W = ρ W'       (the dimensionless ρ-derivative)
    """
    mp.mp.dps = dps
    n_e, n_c = len(energies), len(channels)
    Hp  = np.zeros((n_e, n_c), dtype=complex)
    Hm  = np.zeros((n_e, n_c), dtype=complex)
    Hpp = np.zeros((n_e, n_c), dtype=complex)
    Hmp = np.zeros((n_e, n_c), dtype=complex)
    is_open = np.zeros((n_e, n_c), dtype=bool)

    for ie, E in enumerate(energies):
        for ic, ch in enumerate(channels):
            E_rel = E - ch.threshold
            l = ch.l
            if E_rel > 0:
                k = np.sqrt(E_rel / ch.mass_factor)
                rho = k * channel_radius
                eta = _sommerfeld(z1z2, k, ch.mass_factor) if z1z2 else 0.0
                F  = float(mp.coulombf(l, eta, rho))
                G  = float(mp.coulombg(l, eta, rho))
                dF = float(mp.diff(lambda r: mp.coulombf(l, eta, r), rho))
                dG = float(mp.diff(lambda r: mp.coulombg(l, eta, r), rho))
                Hp[ie, ic]  = G + 1j * F
                Hm[ie, ic]  = G - 1j * F
                Hpp[ie, ic] = (dG + 1j * dF) * rho
                Hmp[ie, ic] = (dG - 1j * dF) * rho
                is_open[ie, ic] = True
            else:
                k = np.sqrt(-E_rel / ch.mass_factor)
                rho = 2 * k * channel_radius
                eta = _sommerfeld(z1z2, k, ch.mass_factor) if z1z2 else 0.0
                # Whittaker W_{-η, ℓ+1/2}(ρ)
                W  = float(mp.whitw(-eta, l + 0.5, rho))
                dW = float(mp.diff(lambda r: mp.whitw(-eta, l + 0.5, r), rho))
                Hp[ie, ic] = Hm[ie, ic] = W
                Hpp[ie, ic] = Hmp[ie, ic] = rho * dW
                is_open[ie, ic] = False

    return BoundaryValues(
        H_plus=jnp.asarray(Hp),
        H_minus=jnp.asarray(Hm),
        H_plus_p=jnp.asarray(Hpp),
        H_minus_p=jnp.asarray(Hmp),
        is_open=jnp.asarray(is_open),
    )


def _sommerfeld(z1z2, k, mass_factor):
    """η = Z1 Z2 e²/(ℏv) in natural nuclear units (e² ≈ 1.44 MeV·fm)."""
    z1, z2 = z1z2
    return z1 * z2 * 1.44 / (2.0 * mass_factor * k)
```

### Why Whittaker for closed channels

For $E < E_c$ the channel is classically closed. The asymptotic solution is $u_c(r) \propto W_{-\eta_c, \ell_c+1/2}(2|k_c| r)$ which decays exponentially. The R-matrix matching parameter $B_c = 2 k_c a\, W'/W$ [Desc. eq. 9] is constructed such that the surface term vanishes for closed channels, eliminating the source of numerical instability that plagues finite-difference methods near thresholds.

The library transparently routes the right matching condition based on `BoundaryValues.is_open[ie, ic]`. The S-matrix is constructed only on the open subspace, with closed-channel amplitudes extracted separately if requested.

---

## 10. The spectral submodule

This is the heart of the library and is intentionally **mesh-independent**. It knows nothing about Lagrange functions, kinetic operators, or potentials. It deals only with eigenpairs and surface amplitudes — pure linear algebra on a Bloch-augmented Hamiltonian that has already been built somewhere else.

### 10.1 `Spectrum`

```python
# spectral/types.py
import jax
import jax.numpy as jnp
from dataclasses import dataclass, field


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Spectrum:
    """Spectral decomposition of an N×N Bloch-augmented Hamiltonian.
    
    For Hermitian H (real V):
        H = U diag(ε) U†,   U†U = I,   ε real
    For complex-symmetric H (complex/optical V):
        H = U diag(ε) U^T,  U^T U = I  (bilinear form, not sesquilinear),
        ε complex.
    
    Fields
    ------
    eigenvalues : (M,) array
        ε_k, real if Hermitian else complex.
    surface_amplitudes : (M, N_c) array
        γ_kc = (U^T Q)_kc. Sufficient on its own for R-matrix and S-matrix.
    eigenvectors : (M, M) array or None
        Full U; None if not needed (R/S/phases only).
    is_hermitian : bool (static)
        Routes downstream conjugation; baked into JIT signatures.
    """
    eigenvalues:        jnp.ndarray
    surface_amplitudes: jnp.ndarray
    eigenvectors:       jnp.ndarray | None
    is_hermitian:       bool = field(metadata={"static": True})
```

### 10.2 Observable spectral sums

All take a `Spectrum` and return JAX arrays. No mesh, no channels, no boundary — just contractions over eigenmodes.

```python
# spectral/observables.py
import jax
import jax.numpy as jnp
from .types import Spectrum


def rmatrix_from_spectrum(
    spectrum: Spectrum, E: float, channel_radius: float, mass_factor: float
) -> jnp.ndarray:
    """R_cc'(E) = (1/a) Σ_k γ_kc γ_kc' / (ε_k - E/μ)
    
    Wigner-Eisenbud form. ε_k are in fm⁻², E is in MeV, mass_factor
    (ℏ²/2μ) is in MeV·fm². The ratio E/mass_factor converts E to fm⁻²
    to match the eigenvalue units.
    
    For the single-mass-factor case (v1), mass_factor = channels[0].mass_factor.
    Vmappable in E, differentiable in V (through spectrum) and in E.
    """
    γ = spectrum.surface_amplitudes
    E_dimless = E / mass_factor                          # convert to fm⁻²
    denom = 1.0 / (spectrum.eigenvalues - E_dimless)    # (M,) in fm²
    return jnp.einsum("m,mc,md->cd", denom, γ, γ) / channel_radius


def greens_from_spectrum(spectrum: Spectrum, E: float, mass_factor: float) -> jnp.ndarray:
    """G(E)_nm = Σ_k u_kn u_km / (ε_k - E/μ)
    
    This is the resolvent of H (in fm⁻² units), evaluated at E in MeV.
    The sign convention matches Descouvemont: C = H - E/μ, so
    C⁻¹ = Σ_k |u_k><u_k| / (ε_k - E/μ), i.e. (H - E/μ)⁻¹.
    
    For Hermitian H: G = U diag(1/(ε-E/μ)) U†
    For complex-symmetric H: G = U diag(1/(ε-E/μ)) U^T  (bilinear resolvent)
    
    Requires spectrum.eigenvectors. Returns (M, M).
    """
    U = spectrum.eigenvectors
    E_dimless = E / mass_factor
    denom = 1.0 / (spectrum.eigenvalues - E_dimless)
    UT = U.conj().T if spectrum.is_hermitian else U.T
    return (U * denom[None, :]) @ UT


def wavefunction_internal_from_spectrum(
    spectrum: Spectrum, E: float, source: jnp.ndarray
) -> jnp.ndarray:
    """ψ_int = G(E) · source, with source the L(B) projected external
    function from Descouvemont eq. 27. Returns (M,)."""
    G = greens_from_spectrum(spectrum, E)
    return G @ source
```

### 10.3 Matching: from R to S

```python
# spectral/matching.py
import jax
import jax.numpy as jnp


def smatrix_from_R(R: jnp.ndarray, boundary_at_E) -> jnp.ndarray:
    """S = (H_- - R H_-')(H_+ - R H_+')^{-1}     (open subspace).
    
    boundary_at_E carries length-N_c arrays for one energy E:
        H_plus, H_minus, H_plus_p, H_minus_p, is_open
    """
    Hp, Hm = jnp.diag(boundary_at_E.H_plus), jnp.diag(boundary_at_E.H_minus)
    Hpp, Hmp = jnp.diag(boundary_at_E.H_plus_p), jnp.diag(boundary_at_E.H_minus_p)
    A = Hm - R @ Hmp
    B = Hp - R @ Hpp
    return jnp.linalg.solve(B.T, A.T).T


def phases_from_S(S: jnp.ndarray) -> jnp.ndarray:
    """Phase shifts δ = (1/2) arg(eigvals(S)). Returns (N_c,) in radians."""
    eigvals = jnp.linalg.eigvals(S)
    return 0.5 * jnp.angle(eigvals)
```

For closed channels, the `is_open` mask drives a projection that's implemented inside the compile-time-bound solver methods (so it stays static, no dynamic shapes inside JIT).

### 10.4 The submodule's public surface

```python
# spectral/__init__.py
from .types import Spectrum
from .observables import (
    rmatrix_from_spectrum,
    greens_from_spectrum,
    wavefunction_internal_from_spectrum,
)
from .matching import smatrix_from_R, phases_from_S
from .interpolation import pade_interpolate

__all__ = [
    "Spectrum",
    "rmatrix_from_spectrum",
    "greens_from_spectrum",
    "wavefunction_internal_from_spectrum",
    "smatrix_from_R",
    "phases_from_S",
    "pade_interpolate",
]
```

This submodule has its own test suite (`spectral/tests/`) using synthetic Hamiltonians — random Hermitian or complex-symmetric matrices with known spectral structure — independent of any mesh logic. Property tests verify the spectral identities exactly: $\sum_k \gamma_{kc} \gamma_{kc'} / (\varepsilon_k - E) = (Q^T (H - E I)^{-1} Q)_{cc'}$ to round-off.

---

## 11. Solvers and method dispatch

### 11.1 The spectrum kernel

```python
# solvers/spectrum.py
import jax
import jax.numpy as jnp
from functools import partial

from ..spectral.types import Spectrum
from .assembly import assemble_block_hamiltonian, build_Q


def make_spectrum_kernel(
    mesh, operators, channels,
    method: str = "eigh",
    keep_eigenvectors: bool = False,
):
    """Build a JIT'd spectrum(V) -> Spectrum kernel.
    
    method:
        "eigh"  — Hermitian eigendecomposition (real V). GPU-ready.
        "eig"   — general eigendecomposition (complex V). Uses host
                  callback on GPU; native on CPU.
    
    keep_eigenvectors: bake into the kernel whether to return U.
        True if Green's functions or wavefunctions were requested.
    """
    Q = build_Q(mesh, channels)            # (M, N_c)
    is_hermitian = (method == "eigh")
    
    if method == "eigh":
        @jax.jit
        def spectrum(V):
            H = assemble_block_hamiltonian(mesh, operators, channels, V)
            eigvals, U = jnp.linalg.eigh(H)
            γ = U.T @ Q                    # (M, N_c)
            U_out = U if keep_eigenvectors else None
            return Spectrum(
                eigenvalues=eigvals,
                surface_amplitudes=γ,
                eigenvectors=U_out,
                is_hermitian=True,
            )
        return spectrum
    
    elif method == "eig":
        # Complex symmetric: U^T U = I (bilinear), not U^† U = I
        # JAX's jnp.linalg.eig is CPU; on GPU we use host_callback
        @jax.jit
        def spectrum(V):
            H = assemble_block_hamiltonian(mesh, operators, channels, V)
            eigvals, U = _eig_via_callback(H)
            # Normalize via bilinear form (not sesquilinear)
            U_norm = U / jnp.sqrt(jnp.diag(U.T @ U))[None, :]
            γ = U_norm.T @ Q
            U_out = U_norm if keep_eigenvectors else None
            return Spectrum(
                eigenvalues=eigvals,
                surface_amplitudes=γ,
                eigenvectors=U_out,
                is_hermitian=False,
            )
        return spectrum
    
    raise ValueError(f"Unknown method: {method}")


def _eig_via_callback(H: jax.Array) -> tuple[jax.Array, jax.Array]:
    """jnp.linalg.eig on host via jax.pure_callback — CPU only in current JAX.

    jax.pure_callback signature (JAX >= 0.4.1):
        jax.pure_callback(callback, result_shape_dtypes, *args, vectorized=False)

    result_shape_dtypes must be a pytree of ShapeDtypeStruct matching the
    return structure of callback. The callback receives NumPy arrays and must
    return NumPy-compatible arrays matching the declared shapes and dtypes.
    """
    result_shapes = (
        jax.ShapeDtypeStruct((H.shape[0],), jnp.complex128),   # eigenvalues
        jax.ShapeDtypeStruct(H.shape, jnp.complex128),          # eigenvectors
    )

    def _numpy_eig(h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        vals, vecs = np.linalg.eig(h)
        return vals.astype(np.complex128), vecs.astype(np.complex128)

    return jax.pure_callback(_numpy_eig, result_shapes, H, vectorized=False)
```

### 11.2 Built-in solver functions (bound to `Solver` at compile time)

These are thin wrappers that close over the cached boundary and energies. They are what the user actually calls.

```python
# Inside compile.py (sketch)

def _project_open(R, boundary_tuple, is_open):
    """Project R-matrix and boundary values to the open-channel subspace.
    
    is_open: (N_c,) bool
    R:       (N_c, N_c) full R-matrix
    
    Returns R_open (N_open, N_open) and BoundarySlice for open channels.
    
    Implementation note: JAX requires static shapes inside jit. We handle
    this by always returning the full (N_c, N_c) matrix but zeroing out
    closed-channel rows/columns. The S-matrix is then interpreted only on
    the open subspace by the caller.
    
    In v1 we use the simpler mask approach (valid only when closed-channel
    contributions to the open-channel R-matrix are negligible, i.e. when
    B_c is set correctly so that L(B_c) u_ext = 0 for closed channels per
    [2, eq. 9]). Full decoupling is handled in Phase 9 of the build order.
    """
    mask = is_open.astype(R.dtype)           # (N_c,) 0 or 1
    R_masked = R * mask[:, None] * mask[None, :]
    H_plus, H_minus, H_plus_p, H_minus_p = boundary_tuple
    return R_masked, (
        H_plus  * mask,
        H_minus * mask,
        H_plus_p  * mask,
        H_minus_p * mask,
    )


def _boundary_tuple_to_struct(b_tuple, N_c):
    """Re-wrap a boundary tuple as a simple namespace for smatrix_from_R."""
    H_p, H_m, H_pp, H_mp = b_tuple
    class _B:
        H_plus = H_p;  H_minus = H_m
        H_plus_p = H_pp; H_minus_p = H_mp
    return _B()


def _bind_solvers(spectrum_fn, mesh, channels, energies, boundary, mass_factor):
    """Bind the runtime methods. All returned functions close over
    cached data; none takes boundary values as arguments — those are baked in."""
    a = mesh.scale

    @jax.jit
    def rmatrix(spectrum, E):
        from ..spectral.observables import rmatrix_from_spectrum
        return rmatrix_from_spectrum(spectrum, E, a, mass_factor)

    @jax.jit
    def smatrix(spectrum):
        """Vmap rmatrix → smatrix over the compile-time energy grid.
        Returns (N_E, N_c, N_c). Closed-channel rows/columns are masked
        to zero in v1 (full B_c decoupling is Phase 9).
        """
        from ..spectral.observables import rmatrix_from_spectrum
        from ..spectral.matching import smatrix_from_R

        def _one(E, H_p, H_m, H_pp, H_mp, is_open):
            R = rmatrix_from_spectrum(spectrum, E, a, mass_factor)
            R_m, b_tuple = _project_open(R, (H_p, H_m, H_pp, H_mp), is_open)
            b = _boundary_tuple_to_struct(b_tuple, len(channels))
            return smatrix_from_R(R_m, b)

        return jax.vmap(_one)(
            energies,
            boundary.H_plus, boundary.H_minus,
            boundary.H_plus_p, boundary.H_minus_p,
            boundary.is_open,
        )                                   # (N_E, N_c, N_c)

    @jax.jit
    def phases(spectrum):
        from ..spectral.matching import phases_from_S
        S = smatrix(spectrum)
        return jax.vmap(phases_from_S)(S)   # (N_E, N_c)

    @jax.jit
    def greens(spectrum, E):
        from ..spectral.observables import greens_from_spectrum
        return greens_from_spectrum(spectrum, E, mass_factor)

    @jax.jit
    def eigh(spectrum):
        """Return (eigenvalues, eigenvectors). eigenvectors is None if
        'greens' and 'wavefunction' were not in the requested solvers —
        check spectrum.eigenvectors is not None before use."""
        return spectrum.eigenvalues, spectrum.eigenvectors

    return rmatrix, smatrix, phases, greens, eigh
```

### 11.3 The linear-solve fallback: `rmatrix_direct`

For complex V on GPU where `eig` is unavailable, the user can opt into a per-energy linear solve. This sits in its own namespace because it does not produce a `Spectrum` and cannot serve Green's functions.

```python
# solvers/linear_solve.py
import jax
import jax.numpy as jnp

from .assembly import assemble_block_hamiltonian, build_Q


def make_rmatrix_direct_kernel(mesh, operators, channels, energies, mass_factor):
    """Build a JIT'd rmatrix_direct(V) -> R kernel for all compile-time energies.
    
    Solves C(E) X = Q for X at each energy, then R = Q^T X / a.
    C(E) = H - (E/mass_factor)·I, all in fm⁻².
    No Spectrum is produced. Real or complex V; fully GPU+vmap-ready.
    
    Returns R of shape (N_E, N_c, N_c).
    """
    Q = build_Q(mesh, channels)
    a = mesh.scale
    n_c = len(channels)
    M = mesh.n * n_c

    @jax.jit
    def rmatrix_direct(V):
        H = assemble_block_hamiltonian(mesh, operators, channels, V)  # (M, M) fm⁻²

        def _one_energy(E):
            E_dimless = E / mass_factor
            C = H - E_dimless * jnp.eye(M, dtype=H.dtype)
            Cinv_Q = jnp.linalg.solve(C, Q)
            return (Q.T @ Cinv_Q) / a              # (N_c, N_c)

        return jax.vmap(_one_energy)(energies)      # (N_E, N_c, N_c)

    return rmatrix_direct
```

For a single off-grid energy, the user calls:
```python
# Not a solver method — use the spectral submodule directly
R_single = lax.spectral.rmatrix_from_spectrum(spec, E=7.5, a=solver.mesh.scale,
                                              mass_factor=solver.channels[0].mass_factor)
S_single = lax.spectral.smatrix_from_R(R_single, boundary_at_7p5)
```

### 11.4 Method dispatch policy

When `method=None` (the default), `compile()` picks based on dtype and backend:

```python
def _select_method(method, V_is_complex, backend):
    if method is not None:
        return method
    if not V_is_complex:
        return "eigh"                   # real V → eigh
    if backend == "cpu":
        return "eig"                    # complex V on CPU → eig
    return "linear_solve"               # complex V on GPU → linear solve
```

The user passes `V_is_complex=True` to `compile()` to flag the intent (since the compile step doesn't see V values yet). The default is `False`. Override the auto-selection by setting `method=` explicitly.

| `method`     | When to use                                | Returns Spectrum? | GPU?   | vmap/grad? |
|--------------|--------------------------------------------|-------------------|--------|------------|
| `eigh`       | Real (Hermitian) V — default               | yes               | ✅     | ✅         |
| `eig`        | Complex V, small problems                  | yes (complex)     | callback | grad ✅, vmap awkward |
| `linear_solve` | Complex V on GPU, or want only R-matrix  | no                | ✅     | ✅         |
| `lanczos`    | (future) very large coupled, complex V     | partial           | ✅     | ✅         |

### 11.5 Assembling the block Hamiltonian

#### Unit convention — read this first

The library works throughout in **fm⁻²** (i.e. energy divided by ℏ²/2μ). This matches Descouvemont's convention where `CPOT` is described as "local potentials divided by ℏ²/2μ" [2, §3]. The consequences are:

- `TpL` and all kinetic matrices are stored in fm⁻² (from the mesh builders, which divide by `scale²`).
- `V` supplied by the user should be in **MeV** and is divided by `mass_factor` (= ℏ²/2μ, in MeV·fm²) inside the assembler → result in fm⁻².
- Channel thresholds `E_c` are in MeV and are divided by `mass_factor` → fm⁻².
- Energies `E` passed to `rmatrix_from_spectrum(spec, E, a)` are in **MeV** and the function internally divides by the mass factor of channel 0 (assumed uniform across channels in v1). For multi-mass problems, see §15.4.
- The eigenvalues stored in `Spectrum.eigenvalues` are in fm⁻².
- `rmatrix_from_spectrum` computes `denom = 1 / (eigenvalues - E/mass_factor)` — all in fm⁻².

For a single-mass-factor problem (all channels share the same μ) the R-matrix formula simplifies cleanly. The multi-mass extension is noted in §15 but not fully implemented in v1.

```python
# solvers/assembly.py
import jax.numpy as jnp


def assemble_block_hamiltonian(mesh, operators, channels, V):
    """Build (N_c·N, N_c·N) Bloch-augmented Hamiltonian in fm⁻² units.
    
    All terms are in units of fm⁻² (equivalent to dividing the Schrödinger
    equation through by ℏ²/2μ, following Descouvemont [2] §3 convention).
    
    Diagonal blocks (c == c'):
        TpL + ℓ_c(ℓ_c+1)/r² + E_c/μ_c · I + V_cc/μ_c
    Off-diagonal blocks (c ≠ c'):
        V_cc'/μ_c   (using channel c's mass factor; see §15.4 for multi-mass)
    
    V is supplied in MeV with shape:
        (N_c, N_c, N)      — local  (diagonal in r)
        (N_c, N_c, N, N)   — non-local kernel
    
    The Python for-loop over c, cp is unrolled at JIT trace time (N_c is
    static), so no dynamic Python branching occurs at runtime.
    """
    n_c, N = len(channels), mesh.n
    TpL = operators.TpL
    inv_r2 = operators.inv_r2 if operators.inv_r2 is not None else \
             jnp.diag(1.0 / mesh.radii ** 2)

    blocks = []
    for c in range(n_c):
        mu_c = channels[c].mass_factor        # ℏ²/2μ_c in MeV·fm²
        row = []
        for cp in range(n_c):
            block = jnp.zeros((N, N), dtype=V.dtype)
            if c == cp:
                # Kinetic (already in fm⁻²) + centrifugal (fm⁻²)
                block = block + TpL + channels[c].l * (channels[c].l + 1) * inv_r2
                # Threshold energy: MeV / (MeV·fm²) = fm⁻²
                block = block + (channels[c].threshold / mu_c) * jnp.eye(N)
            # Potential: MeV / (MeV·fm²) = fm⁻²
            if V.ndim == 3:
                block = block + jnp.diag(V[c, cp]) / mu_c
            else:
                block = block + V[c, cp] / mu_c
            row.append(block)
        blocks.append(jnp.concatenate(row, axis=1))
    return jnp.concatenate(blocks, axis=0)    # (N_c·N, N_c·N) in fm⁻²


def build_Q(mesh, channels):
    """Q[c·N + j, c'] = δ_{cc'} φ_j(a). Returns (N_c·N, N_c)."""
    N = mesh.n
    n_c = len(channels)
    b = mesh.basis_at_boundary
    Q = jnp.zeros((n_c * N, n_c))
    for c in range(n_c):
        Q = Q.at[c * N:(c + 1) * N, c].set(b)
    return Q
```

---

## 12. Padé interpolation for energy-dependent potentials

For energy-dependent V(E), the user computes the spectrum at every grid energy and the library evaluates observables at the grid. To get observables at off-grid energies, the library provides Padé interpolation. Padé is the natural choice because observables inherit the rational structure of the R-matrix (poles at the eigenvalues, smooth modulation through the boundary values).

### 12.1 Interface

```python
# spectral/interpolation.py
from typing import Callable
import jax
import jax.numpy as jnp


def pade_interpolate(
    values: jnp.ndarray,            # (N_E, ...) sampled observable
    knots: jnp.ndarray,             # (N_E,) energy grid
    order: tuple[int, int] | None = None,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Return a JIT'd callable f(E_query) → interpolated values.
    
    For each leading-axis sample, fit a rational (p,q)-Padé approximant
    in E about the grid center. Trailing axes are interpolated element-wise.
    
    Default order: (N_E//2 - 1, N_E//2), giving p + q + 1 = N_E.
    """
    n_e = len(knots)
    if order is None:
        order = (n_e // 2 - 1, n_e // 2)
    p, q = order
    assert p + q + 1 == n_e, (
        f"order p+q+1 must equal N_E={n_e}, got {p}+{q}+1={p+q+1}"
    )
    
    # Flatten trailing dims for vectorized fitting
    trail_shape = values.shape[1:]
    flat = values.reshape(n_e, -1)            # (N_E, K)
    
    # Center for numerical stability
    E0 = jnp.mean(knots)
    s = knots - E0
    
    # Build the Padé linear system per K-column
    a_coeffs, b_coeffs = jax.vmap(_solve_pade_lsq, in_axes=(1, None, None, None))(
        flat, s, p, q
    )
    # a_coeffs: (K, p+1), b_coeffs: (K, q+1) with b[0] = 1
    
    a_coeffs = a_coeffs.reshape(*trail_shape, p + 1)
    b_coeffs = b_coeffs.reshape(*trail_shape, q + 1)
    
    @jax.jit
    def evaluate(E):
        """Evaluate at scalar or batched E."""
        E_scalar = jnp.asarray(E)
        ds = E_scalar - E0
        num = sum(a_coeffs[..., k] * ds ** k for k in range(p + 1))
        den = sum(b_coeffs[..., k] * ds ** k for k in range(q + 1))
        return num / den
    
    return evaluate


def _solve_pade_lsq(samples, s, p, q):
    """Solve the Padé linear system for one sequence of samples.
    
    samples = a(s) / b(s) at s_k, with deg(a) = p, deg(b) = q, b_0 = 1.
    
    Standard linearization:
        samples_k · b(s_k) - a(s_k) = 0
        samples_k · (1 + b_1 s_k + ... + b_q s_k^q) - (a_0 + a_1 s_k + ... + a_p s_k^p) = 0
    
    Stack into an N_E × (p+q+1) linear system.
    """
    n_e = len(s)
    A = jnp.zeros((n_e, p + q + 1))
    # Columns 0..p: -s^k coefficient of a_k
    for k in range(p + 1):
        A = A.at[:, k].set(-(s ** k))
    # Columns p+1..p+q: samples * s^k coefficient of b_k (k >= 1)
    for k in range(1, q + 1):
        A = A.at[:, p + k].set(samples * (s ** k))
    rhs = -samples                                # = -samples · b_0 = -samples
    coeffs = jnp.linalg.solve(A, rhs)
    a_coeffs = coeffs[:p + 1]
    b_coeffs = jnp.concatenate([jnp.array([1.0]), coeffs[p + 1:]])
    return a_coeffs, b_coeffs
```

### 12.2 Notes

- **Matrix observables.** The flattening handles any trailing shape, so `pade_interpolate(S_grid, energies_grid)` returns a function that maps E to `(N_c, N_c)` matrices.
- **Complex values.** The Padé construction works element-wise; for complex `values` the coefficients are complex.
- **Order auto-selection.** The default `(N_E//2 - 1, N_E//2)` gives a diagonal Padé (`p ≈ q`), which is typically well-conditioned. Users with prior knowledge of the analytic structure can pass an explicit `order`.
- **Spurious poles.** Standard Padé construction can produce zeros of the denominator inside the interpolation interval (Froissart doublets). For resonance fitting where the pole structure is physically meaningful, downstream tools can analyze `b_coeffs` and warn or project. v1 does not enforce real poles automatically; future work could add an `enforce_real_poles=True` flag using Stoer-Bulirsch or barycentric variants.
- **Differentiability.** The interpolator is differentiable in `values` (so you can fit potential parameters against finely-sampled experimental data while only paying for N_E spectra) and in `E` (so resonance positions can be tracked via `jax.grad(evaluate)(E_resonance)`).

---

## 13. Transforms: grid, Fourier, integration

The defining property of the LMM is that observables are simple sums over mesh points [1, §2.9], and conversion between the mesh basis and finer grids is a single matrix multiplication. The library precomputes these conversion matrices at compile time so they are essentially free at runtime.

### 13.1 Mesh to fine radial grid

For a vector $c$ of mesh coefficients (a wave function, an eigenvector, etc.), the value on a fine grid is

$$\psi(r_k) = \sum_{j} c_j\, f_j(r_k) = \sum_j B_{kj}\, c_j$$

where $B_{kj} = f_j(r_k)$ depends only on the mesh and the chosen grid. For a matrix $V_{nm}$ in basis representation (e.g. a non-local potential, or the Green's function $G_{nm}(E)$), the kernel on the grid is

$$V(r_k, r_l) = (B\, V\, B^T)_{kl}$$

```python
# transforms/grid.py
import jax
import jax.numpy as jnp

from ..meshes._basis_eval import basis_at


def compute_B_grid(mesh, r_grid):
    """B[k, j] = f_j(r_grid[k]).  Computed once at compile time."""
    return basis_at(mesh, r_grid)              # (M_r, N)


def make_to_grid(B_grid):
    """Return two JIT'd callables closing over B_grid.
    
    Two separate functions are used instead of a single conditional because
    JAX retraces on each distinct input shape, and an `if ndim==1` inside
    `@jax.jit` would silently produce two compiled kernels. Explicit names
    make the contract clear and avoid surprising retraces.
    """
    @jax.jit
    def to_grid_vector(c):
        """Mesh coefficient vector (N,) → radial grid (M_r,)."""
        return B_grid @ c

    @jax.jit
    def to_grid_matrix(V):
        """Mesh matrix (N, N) → radial kernel (M_r, M_r)."""
        return B_grid @ V @ B_grid.T

    return to_grid_vector, to_grid_matrix
```

The `Solver` exposes both as `solver.to_grid_vector` and `solver.to_grid_matrix`. For a coupled-channel eigenvector of shape `(N_c, N)`, the user vmaps over the channel axis:

```python
# Wavefunctions on the fine grid for each channel:
psi_on_grid = jax.vmap(solver.to_grid_vector)(eigvec.reshape(n_c, n))
```

The basis-evaluation function `basis_at(mesh, r)` is dispatched by `(family, regularization)` via a parallel registry to the mesh builders. For Legendre-$x$ on $(0, a)$ with $\nu = 1$, this is the Lagrange function of [2, eq. 18]:

```python
# meshes/_basis_eval.py
from scipy.special import eval_legendre
import jax.numpy as jnp

def _legendre_x_basis_at(mesh, r):
    """f_j(r) for shifted Legendre regularized by x. Shape: (len(r), N)."""
    N = mesh.n
    a = mesh.scale
    x_j = mesh.nodes                # (N,)
    
    u = r / a                       # (M_r,)
    P_N = eval_legendre(N, 2*u - 1) # (M_r,)
    
    # Outer construction: numerator (-1)^{N-j} * (r/a) * P_N(2r/a-1)
    sign = jnp.where(jnp.arange(N) % 2 == ((N - 1) % 2), 1.0, -1.0)  # (N,)
    norm = jnp.sqrt((1 - x_j) / x_j) / jnp.sqrt(a)                   # (N,)
    
    # f_j(r) = (-1)^{N-j} sqrt((1-x_j)/x_j) * u P_N(2u-1) / (u - x_j) / sqrt(a)
    # Handle u → x_j limit via L'Hopital or small epsilon (rare in practice)
    num = sign[None, :] * norm[None, :] * u[:, None] * P_N[:, None]   # (M_r, N)
    denom = u[:, None] - x_j[None, :]                                 # (M_r, N)
    return num / denom
```

(A careful implementation handles `u → x_j` via the derivative of $P_N$; for a grid not coincident with mesh nodes this never triggers.)

The same `B_grid` matrix is reused by:

- `solver.to_grid(c)` for a wave-function vector.
- `lax.spectral.wavefunction_internal_on_grid(spec, E, channel, B_grid)` which projects the internal wavefunction onto the grid for visualization.
- The Fourier-transform builder (next section), which uses `B_grid` evaluated at a fine internal quadrature.

### 13.2 Momentum-space (Fourier) transforms

The momentum-space partial-wave amplitude is

$$\tilde u_\ell(k) = \sqrt{\frac{2}{\pi}} \int_0^a j_\ell(kr)\, u_\ell(r)\, dr$$

where $j_\ell$ is the spherical Bessel function and $u_\ell(r) = \sum_j c_j f_j(r)$ is the radial wave function. Substituting:

$$\tilde u_\ell(k) = \sum_j F^{(\ell)}_{kj}\, c_j, \qquad F^{(\ell)}_{kj} = \sqrt{\frac{2}{\pi}} \int_0^a j_\ell(k r)\, f_j(r)\, dr$$

The integral is done once at compile time using a fine internal Gauss-Legendre quadrature; the result is a `(M_k, N)` matrix per partial wave $\ell$. Different channels with the same $\ell$ share the same Fourier matrix; the library deduplicates them.

```python
# transforms/fourier.py
import numpy as np
import jax.numpy as jnp
from scipy.special import spherical_jn


def compute_F_momentum(mesh, momenta, l, n_quad=200):
    """F[k, j] = sqrt(2/π) ∫₀^a j_l(k r) f_j(r) dr.
    
    Internal Gauss-Legendre quadrature with n_quad points for accuracy.
    """
    from ..meshes._basis_eval import basis_at
    
    # Fine quadrature on (0, a)
    x_q, w_q = np.polynomial.legendre.leggauss(n_quad)
    r_q = 0.5 * mesh.scale * (x_q + 1)
    w_q = 0.5 * mesh.scale * w_q
    
    f_at_q = np.asarray(basis_at(mesh, jnp.asarray(r_q)))      # (n_quad, N)
    
    F = np.zeros((len(momenta), mesh.n))
    for ik, k in enumerate(momenta):
        j_l_kr = spherical_jn(l, k * r_q)                       # (n_quad,)
        integrand = j_l_kr[:, None] * f_at_q                    # (n_quad, N)
        F[ik] = np.sqrt(2.0 / np.pi) * (w_q @ integrand)
    
    return jnp.asarray(F)
```

For matrix transforms, $\tilde V(p, p') = F V F^T$ as before. For coupled channels, the user passes the channel index to select the right $\ell$:

```python
@jax.jit
def fourier(c_or_V, channel_idx=0):
    F = transform_matrices.F_momentum[channel_idx]    # (M_k, N) for this ℓ
    if c_or_V.ndim == 1:
        return F @ c_or_V
    return F @ c_or_V @ F.T
```

### 13.3 Integration

Norms and expectation values of operators in the mesh basis are simple sums [1, §2.9, eq. 2.82]:

```python
# transforms/integration.py
import jax
import jax.numpy as jnp


def make_integration(mesh):
    @jax.jit
    def norm(c):
        """⟨ψ|ψ⟩ ≈ Σ c_j²  (exact at Gauss approximation when basis is orthonormal)."""
        return jnp.sum(c ** 2)
    
    @jax.jit
    def expect_local(c, V_at_mesh):
        """⟨ψ|V(r)|ψ⟩ ≈ Σ c_j² V(r_j)         [Baye eq. 2.82]."""
        return jnp.sum(c ** 2 * V_at_mesh)
    
    @jax.jit
    def expect_operator(c, O):
        """⟨ψ|O|ψ⟩ = c† O c for any precomputed operator matrix."""
        return c.conj() @ O @ c
    
    return {"norm": norm, "expect_local": expect_local, "expect_op": expect_operator}
```

For a wave function in a regularized basis, the orthonormality of the *Gauss approximation* [1, eq. 2.12] guarantees that $\langle\psi|\psi\rangle \approx \sum_j c_j^2$ to the same accuracy as the LMM itself, even when the regularized basis is not exactly orthonormal in the full $L^2$ sense [1, §2.7, eq. 2.69].

---

## 14. The `compile()` factory

This is the user's entry point. `compile()` builds the mesh and operators, evaluates Coulomb / Whittaker boundary values via `mpmath`, selects the linear-algebra method, builds the requested transform matrices, and constructs the JIT'd solver kernels closing over all of the above.

### 14.1 Signature

```python
# compile.py
from typing import Iterable, Literal
import jax
import jax.numpy as jnp
import numpy as np

from .types import MeshSpec, ChannelSpec, Solver, TransformMatrices
from .meshes._registry import build_mesh
from .boundary.coulomb import compute_boundary_values
from .transforms.grid import compute_B_grid, make_to_grid
from .transforms.fourier import compute_F_momentum
from .transforms.integration import make_integration
from .solvers.spectrum import make_spectrum_kernel
from .solvers.observables import (
    make_rmatrix, make_smatrix, make_phases, make_greens,
    make_wavefunction_internal,
)
from .solvers.direct import make_rmatrix_direct


Method = Literal["eigh", "eig", "linear_solve"]


def compile(
    *,
    mesh: MeshSpec,
    channels: Iterable[ChannelSpec],
    operators: Iterable[str] = ("T+L",),
    solvers: Iterable[str] = ("spectrum", "rmatrix", "smatrix", "phases"),
    energies: jnp.ndarray | None = None,
    energy_dependent: bool = False,
    method: Method | None = None,
    V_is_complex: bool = False,
    grid: jnp.ndarray | None = None,
    momenta: jnp.ndarray | None = None,
    z1z2: tuple[int, int] | None = None,
    dtype: jnp.dtype = jnp.float64,
    device: jax.Device | str | None = None,
    dps: int = 40,
) -> Solver:
    """Build a Solver specialized to the given mesh, channels, and energy grid.

    Parameters
    ----------
    mesh : MeshSpec
        Mesh family, regularization, size, and scale.
    channels : iterable of ChannelSpec
        Channel structure (l, threshold, mass_factor).
    operators : iterable of str
        Which operator matrices to precompute. Default ["T+L"]. Options:
        "T", "T+L", "1/r", "1/r^2", "D".
    solvers : iterable of str
        Which kernels to build and attach to the returned Solver. Options:
        "spectrum"     — eigendecomposition of H (always built unless `rmatrix_direct`)
        "rmatrix"      — R(E) from spectrum, free in E
        "smatrix"      — S at compile-time energies (needs `energies`)
        "phases"       — δ at compile-time energies
        "greens"       — G(E) from spectrum, free in E
        "wavefunction" — internal wavefunction on the mesh
        "rmatrix_direct" — per-energy linear solve fallback (no spectrum)
    energies : jnp.ndarray, optional
        Energy grid (MeV). Required if any of {smatrix, phases, rmatrix_direct}
        is requested, or if `energy_dependent=True`. Used to precompute
        boundary values via mpmath.
    energy_dependent : bool
        If True, indicates that V will be supplied per-energy at runtime
        (one V per compile-time energy point). The user is expected to
        call `jax.vmap(solver.spectrum)` over the energy axis themselves;
        `solver.smatrix` and friends consume the resulting batched Spectrum.
        Padé interpolation across grid points is available via
        `lax.spectral.pade_interpolate`.
    method : "eigh" | "eig" | "linear_solve" | None
        Linear-algebra backend. None invokes the default policy (see §11.4).
    V_is_complex : bool
        Whether the user will supply complex potentials. Drives default
        method selection if `method=None`.
    grid : jnp.ndarray, optional
        Fine radial grid (fm) for `to_grid` and `wavefunction_on_grid`.
    momenta : jnp.ndarray, optional
        Momentum grid (fm⁻¹) for `fourier`.
    z1z2 : tuple of int, optional
        (Z₁, Z₂) for Coulomb scattering. Default neutral (η=0).
    dtype : jnp.dtype
        Floating-point precision. Default float64.
    device : str or jax.Device, optional
        Where to place the compiled solver.
    dps : int
        mpmath decimal precision for Coulomb / Whittaker evaluation.

    Returns
    -------
    Solver
        Bundle with cached data and JIT'd callables.
    """
    channels = tuple(channels)
    operators_set = set(operators)
    solvers_set = set(solvers)

    # --- Validate ---
    needs_boundary = bool(solvers_set & {"smatrix", "phases", "rmatrix_direct"})
    if (needs_boundary or energy_dependent) and energies is None:
        raise ValueError("`energies` required for continuum solvers or energy-dependent V")
    if "T+L" not in operators_set and "spectrum" in solvers_set:
        operators_set.add("T+L")

    # --- Method selection ---
    if method is None:
        method = _select_method(V_is_complex)

    # --- Mesh and operators ---
    mesh_data, operator_matrices = build_mesh(
        family=mesh.family,
        regularization=mesh.regularization,
        n=mesh.n,
        scale=mesh.scale,
        operators=operators_set,
        **mesh.extras,
    )

    # --- Boundary values (mpmath, plain Python) ---
    if energies is not None:
        energies_np = np.asarray(energies)
        boundary = compute_boundary_values(
            channels=channels,
            energies=energies_np,
            channel_radius=mesh.scale,
            z1z2=z1z2,
            dps=dps,
        )
        energies_arr = jnp.asarray(energies_np, dtype=dtype)
    else:
        boundary = None
        energies_arr = jnp.zeros((0,), dtype=dtype)

    # --- Transform matrices ---
    transforms = TransformMatrices()
    if grid is not None:
        transforms = transforms._replace(
            B_grid=compute_B_grid(mesh_data, jnp.asarray(grid)),
            grid_r=jnp.asarray(grid),
        )
    if momenta is not None:
        unique_ls = sorted({ch.l for ch in channels})
        F_per_l = {l: compute_F_momentum(mesh_data, jnp.asarray(momenta), l)
                   for l in unique_ls}
        F_stack = jnp.stack([F_per_l[ch.l] for ch in channels])
        transforms = transforms._replace(
            F_momentum=F_stack,
            momenta=jnp.asarray(momenta),
        )

    # --- Build the spectrum kernel (the central primitive) ---
    if "spectrum" in solvers_set or solvers_set & {"rmatrix", "smatrix", "phases", "greens", "wavefunction"}:
        spectrum_fn = make_spectrum_kernel(
            mesh_data, operator_matrices, channels, method=method,
        )
    else:
        spectrum_fn = None

    # --- Bind observables derived from spectrum ---
    rmatrix_fn = make_rmatrix(mesh_data, channels) if "rmatrix" in solvers_set else None
    smatrix_fn = make_smatrix(mesh_data, channels, energies_arr, boundary) if "smatrix" in solvers_set else None
    phases_fn  = make_phases(smatrix_fn) if "phases" in solvers_set else None
    greens_fn  = make_greens() if "greens" in solvers_set else None
    wf_int_fn  = make_wavefunction_internal(mesh_data, channels) if "wavefunction" in solvers_set else None

    # --- Linear-solve fallback ---
    if "rmatrix_direct" in solvers_set:
        rmatrix_direct_fn = make_rmatrix_direct(
            mesh_data, operator_matrices, channels, energies_arr, boundary,
        )
    else:
        rmatrix_direct_fn = None

    # --- Transforms ---
    to_grid_fn = make_to_grid(transforms.B_grid) if transforms.B_grid is not None else None
    integ = make_integration(mesh_data)

    # --- Assemble Solver ---
    solver = Solver(
        mesh=mesh_data,
        operators=operator_matrices,
        channels=channels,
        energies=energies_arr,
        boundary=boundary,
        transforms=transforms,
        method=method,
        spectrum=spectrum_fn,
        rmatrix=rmatrix_fn,
        smatrix=smatrix_fn,
        phases=phases_fn,
        greens=greens_fn,
        wavefunction_internal=wf_int_fn,
        rmatrix_direct=rmatrix_direct_fn,
        to_grid=to_grid_fn,
        integrate=integ,
    )

    if device is not None:
        solver = _to_device(solver, device)
    return solver


def _select_method(V_is_complex: bool) -> Method:
    """Default method-selection policy. See §11.4."""
    if not V_is_complex:
        return "eigh"
    if jax.default_backend() == "cpu":
        return "eig"
    return "linear_solve"
```

### 14.2 What changes vs v1.0

Three structural changes from the previous design:

1. **`spectrum` is the central kernel.** Everything else (`rmatrix`, `smatrix`, `phases`, `greens`, `wavefunction_internal`) is a thin closure over it. The factory builds the spectrum kernel first and then attaches lightweight observables.

2. **`method` is a new parameter** that controls the linear-algebra backend. The default policy is real → `eigh`, complex on CPU → `eig`, complex on GPU → `linear_solve`. The user can always override.

3. **`rmatrix_direct` is a separate namespace.** When the user explicitly requests it (typically because they need complex V on GPU and have chosen `method="linear_solve"`), it is built as a per-energy direct kernel. It does *not* produce a Spectrum and cannot be used with `wavefunction_internal` or `greens`.

### 14.3 What the factory does *not* do

- It does not accept potentials. The solver is potential-agnostic.
- It does not perform per-call setup. Everything not depending on `V` is done here, once.
- It does not implicitly broadcast over potential parameters. The user uses `jax.vmap` over their parametric `V` builder.
- It does not handle off-grid energies internally for S-matrix or phases. Those require Coulomb boundary values (mpmath, not JIT-able) and so are pinned to the compile-time grid. For off-grid evaluation the user calls `lax.spectral.pade_interpolate`.

---

## 15. Coupled-channel structure

A coupled-channel calculation has $N_c$ channels (each characterized by `(l, threshold, mass_factor)` per [2, §2.1, eq. 2]). The Hamiltonian is block-structured:

$$\mathcal{H} = \begin{pmatrix} T_{1} + V_{11} & V_{12} & \cdots \\ V_{21} & T_{2} + V_{22} & \cdots \\ \vdots & & \ddots \end{pmatrix}$$

where each block is $(N, N)$ and the full matrix is $(N_c N, N_c N)$. The diagonal blocks include the kinetic operator augmented with the centrifugal term and the channel's threshold; off-diagonal blocks are channel-coupling potentials.

### 15.1 Input format for V

The library accepts user-supplied potentials as JAX arrays:

- **Local potential** (diagonal in $r$): shape `(N_c, N_c, N)`. Entry `V[c, c', i]` is $V_{cc'}(r_i)$ in MeV. For local potentials in the LMM, off-diagonal channel coupling at different mesh points is zero by construction; only the diagonal-in-$r$ structure is needed.
- **Non-local potential** (kernel in $(r, r')$): shape `(N_c, N_c, N, N)`. Entry `V[c, c', i, j]` is the (already Gauss-scaled) matrix element $(\lambda_i \lambda_j)^{1/2}\, W_{cc'}(r_i, r_j) \cdot a$ per [2, eq. 26].

A helper does the Gauss scaling automatically:

```python
def assemble_nonlocal(mesh, kernel_fn, n_c=1):
    """Build V[c, c', i, j] from a user kernel W(r, r').
    
    For uncoupled (N_c=1): kernel_fn(r1, r2) -> scalar.
    For coupled: kernel_fn(r1, r2) returns (N_c, N_c) — vectorize internally.
    """
    r = mesh.radii
    lam = mesh.weights
    a = mesh.scale
    
    if n_c == 1:
        ri, rj = jnp.meshgrid(r, r, indexing="ij")
        wi, wj = jnp.meshgrid(lam, lam, indexing="ij")
        W = kernel_fn(ri, rj) * jnp.sqrt(wi * wj) * a
        return W[None, None, :, :]                  # (1, 1, N, N)
    # Coupled case: kernel returns (..., N_c, N_c) at each (r, r') pair
    ri, rj = jnp.meshgrid(r, r, indexing="ij")
    wi, wj = jnp.meshgrid(lam, lam, indexing="ij")
    W = kernel_fn(ri, rj) * jnp.sqrt(wi * wj)[..., None, None] * a
    return jnp.einsum('ijcd->cdij', W)              # (N_c, N_c, N, N)
```

### 15.2 Surface amplitudes carry channel structure

The `Spectrum.surface_amplitudes` array has shape `(M, N_c)` where $M = N_c N$. Element `γ[k, c]` is the surface amplitude of eigenmode $k$ in channel $c$:

$$\gamma_{kc} = \sum_n \varphi_n(a)\, u^{(k)}_{(c, n)}$$

where $u^{(k)}$ is the $k$-th eigenvector of the block Hamiltonian and the second index runs over $(c, n)$ with $n$ the mesh point and $c$ the channel. The R-matrix and S-matrix spectral sums then naturally produce $N_c \times N_c$ matrices.

### 15.3 Mass factor per channel

Different channels can have different reduced masses (e.g. nucleon-nucleus vs nucleus-nucleus). The `ChannelSpec.mass_factor` field is $\hbar^2/2\mu_c$ in user units (typically MeV·fm²). For the radial Schrödinger equation, the diagonal block becomes (after rescaling so all entries are in MeV):

$$\mu_c\, \hat T_c + \frac{\ell_c(\ell_c+1)\, \mu_c}{r^2} + V_{cc}(r) + E_c$$

The library handles the $\mu_c$ scaling internally; the user supplies $V$ in MeV.

### 15.4 Convention summary — units and Hamiltonian scaling

**The library works throughout in fm⁻² (nuclear units, dividing by ℏ²/2μ).** This is Descouvemont's convention [2, §3] where input potentials are described as "divided by ℏ²/2μ". The specific rules:

| Quantity | User provides | Stored / computed as |
|---|---|---|
| Energies `E`, `E_c` | MeV | MeV (divided by `mass_factor` inside assemblers) |
| Lengths `a`, `h` | fm | fm |
| Potential values `V` | MeV | MeV → divided by `mass_factor` fm⁻² in assembler |
| `mass_factor` (= ℏ²/2μ) | MeV·fm² | used as conversion factor |
| `TpL`, `T`, `D`, `inv_r`, `inv_r2` | — | fm⁻² (mesh builders divide by `scale²`) |
| `Spectrum.eigenvalues` | — | fm⁻² |
| `rmatrix_from_spectrum(E=...)` | E in MeV | converts via `E_dimless = E / mass_factor` |

Standard nuclear value: ℏ²/2mₙ = 20.736 MeV·fm² [2, eq. 46].

**Multi-mass channels (v1 limitation).** When `N_c > 1` with different mass factors per channel, the diagonal blocks of H have different effective scales. In v1, `rmatrix_from_spectrum` uses `channels[0].mass_factor` as the global conversion. For problems where all channels share the same μ (elastic scattering, single-nucleon reactions), this is exact. For genuinely different masses (e.g. proton+nucleus and neutron+nucleus coupled), the user should normalize all channels to a common μ by absorbing the ratio into V before passing to the solver. A proper per-channel mass implementation is future work.

---

## 16. Public API and usage examples

### 16.1 Public namespace

```python
# lax/__init__.py
from .types import MeshSpec, ChannelSpec, Solver
from .compile import compile
from .operators.potential import assemble_local, assemble_nonlocal
from . import spectral                              # mesh-independent submodule

__all__ = [
    "MeshSpec", "ChannelSpec", "Solver",
    "compile",
    "assemble_local", "assemble_nonlocal",
    "spectral",
]
```

The `lax.spectral` submodule is exposed as a first-class peer because its functions (`rmatrix_from_spectrum`, `smatrix_from_R`, `pade_interpolate`, etc.) are useful standalone — for postprocessing, for stitching different solvers together, or for implementing custom observables.

### 16.2 Example 1: Yamaguchi non-local potential

This reproduces the test in the user's prototype and corresponds to Example 5 of Descouvemont [2, §5.8]. It is the canonical end-to-end test.

```python
import jax.numpy as jnp
import lax

HBAR2_2MU = 41.472   # MeV·fm² for N–N (Descouvemont eq. 46)
ALPHA = 0.2316053    # fm⁻¹
BETA  = 1.3918324    # fm⁻¹

def yamaguchi(r1, r2):
    """Yamaguchi non-local kernel [Descouvemont eq. 53], in MeV·fm⁻¹."""
    return -2.0 * BETA * (ALPHA + BETA)**2 * jnp.exp(-BETA * (r1 + r2)) * HBAR2_2MU

energies = jnp.array([0.1, 10.0])

solver = lax.compile(
    mesh     = lax.MeshSpec("legendre", "x", n=20, scale=8.0),
    channels = (lax.ChannelSpec(l=0, threshold=0.0, mass_factor=HBAR2_2MU),),
    operators = ("T+L",),
    solvers  = ("spectrum", "rmatrix", "smatrix", "phases"),
    energies = energies,
)

V = lax.assemble_nonlocal(solver.mesh, yamaguchi)  # (1, 1, 20, 20)

# One eigendecomposition, multiple observables:
spec    = solver.spectrum(V)
S_grid  = solver.smatrix(spec)                     # (2, 1, 1) at compile-time E
δ_grid  = solver.phases(spec) * (180/jnp.pi)       # (2, 1) degrees
R_off   = solver.rmatrix(spec, E=5.0)              # off-grid: R(5 MeV)

# Reference values from Baye/Descouvemont:
#   E=0.1   → δ = -15.078689°
#   E=10.0  → δ =  85.634560°
```

### 16.3 Example 2: hydrogen atom (bound states)

Tests the regularized-Laguerre family. The ground state of hydrogen is reproduced to machine precision at $h = 1/2$ with $N \geq 2$ [1, §5.4.2, eq. 5.29]. Since no continuum solver is requested, `energies` is omitted.

```python
solver = lax.compile(
    mesh      = lax.MeshSpec("laguerre", "x", n=30, scale=2.0),  # h = n/2 = 2
    channels  = (lax.ChannelSpec(l=0, threshold=0.0, mass_factor=0.5),),
    operators = ("T", "1/r"),
    solvers   = ("spectrum",),
)

V = -1.0 / solver.mesh.radii        # hydrogen: V(r) = -1/r
V = V[None, None, :]                 # (1, 1, 30) — local, coupled-channel shape

spec = solver.spectrum(V)
# spec.eigenvalues[:7] should be {-1/2, -1/8, -1/18, -1/32, -1/50, -1/72, -1/98}
#                              = E_n = -1/(2n²)
```

For bound states only, the `spectrum` kernel produces eigenvalues and eigenvectors directly — no observables construction needed.

### 16.4 Example 3: energy scan with gradients

Fit a parametric non-local potential to target phase shifts on a 500-energy grid. The fit reuses a single Spectrum per loss evaluation.

```python
import jax
import optax

energies = jnp.linspace(0.1, 100.0, 500)
target_δ = ...                            # (500,) experimental degrees

solver = lax.compile(
    mesh     = lax.MeshSpec("legendre", "x", n=40, scale=10.0),
    channels = (lax.ChannelSpec(l=0, threshold=0.0, mass_factor=HBAR2_2MU),),
    solvers  = ("spectrum", "phases"),
    energies = energies,
)

def loss(params):
    α, β = params["alpha"], params["beta"]
    kernel = lambda r1, r2: -2*β * (α+β)**2 * jnp.exp(-β*(r1+r2)) * HBAR2_2MU
    V = lax.assemble_nonlocal(solver.mesh, kernel)
    spec = solver.spectrum(V)                          # ONE eigendecomp
    δ = solver.phases(spec)[:, 0] * (180/jnp.pi)       # 500 phase shifts
    return jnp.mean((δ - target_δ)**2)

opt = optax.adam(1e-3)
params = {"alpha": 0.2, "beta": 1.4}
opt_state = opt.init(params)

@jax.jit
def step(params, opt_state):
    val, grads = jax.value_and_grad(loss)(params)
    updates, opt_state = opt.update(grads, opt_state)
    return optax.apply_updates(params, updates), opt_state, val

for it in range(1000):
    params, opt_state, l = step(params, opt_state)
```

### 16.5 Example 4: batch over potentials on GPU

```python
solver = lax.compile(
    mesh     = lax.MeshSpec("legendre", "x", n=40, scale=10.0),
    channels = (lax.ChannelSpec(l=0, threshold=0.0, mass_factor=HBAR2_2MU),),
    solvers  = ("spectrum", "phases"),
    energies = jnp.linspace(0.1, 50.0, 100),
    device   = "gpu:0",
)

# 1000 different (α, β) combinations
αs = jnp.linspace(0.1, 0.5, 1000)
βs = jnp.linspace(1.0, 2.0, 1000)

def make_V(α, β):
    kernel = lambda r1, r2: -2*β * (α+β)**2 * jnp.exp(-β*(r1+r2)) * HBAR2_2MU
    return lax.assemble_nonlocal(solver.mesh, kernel)

V_batch = jax.vmap(make_V)(αs, βs)                  # (1000, 1, 1, 40, 40)
spec_batch = jax.vmap(solver.spectrum)(V_batch)      # batched Spectrum
δ_batch    = jax.vmap(solver.phases)(spec_batch)     # (1000, 100, 1) on GPU
```

### 16.6 Example 5: Green's function for response calculations

```python
solver = lax.compile(
    mesh      = lax.MeshSpec("laguerre", "x", n=40, scale=0.3),
    channels  = (lax.ChannelSpec(l=1, threshold=0.0, mass_factor=0.5),),
    operators = ("T", "1/r", "1/r^2"),
    solvers   = ("spectrum", "greens"),
)

def V_test(r):
    """Test potential from Baye §6.5 (eq. 6.40)."""
    return -8.0 * jnp.exp(-0.16 * r**2) + 4.0 * jnp.exp(-0.04 * r**2)

V = V_test(solver.mesh.radii)[None, None, :]        # (1, 1, 40)
spec = solver.spectrum(V)

# Evaluate G(E) at arbitrary energies without recomputing the spectrum:
G_at_E   = solver.greens(spec, E=0.5)               # (M, M)
G_scan   = jax.vmap(lambda E: solver.greens(spec, E))(jnp.linspace(-2, 5, 200))
# G_scan: (200, M, M) — one eigendecomposition, 200 resolvents
```

### 16.7 Example 6: energy-dependent V(E) with Padé interpolation

For energy-dependent potentials the spectrum varies with energy. At each compile-time energy `E_i` the user computes `spec_i = solver.spectrum(V(E_i))` and wants `S(E_i)`. **This requires evaluating the R-matrix at `E_i` from `spec_i` specifically** — using `solver.smatrix(spec_i)` would give S at all N_E compile-time energies using `spec_i` as if V were energy-independent, which is wrong.

The correct pattern uses `lax.spectral` functions directly, exploiting the fact that `BoundaryValues` is a pytree and vmap slices its leading axis automatically:

```python
energies_grid = jnp.linspace(0.1, 50.0, 21)            # sparse compile-time grid

solver = lax.compile(
    mesh             = lax.MeshSpec("legendre", "x", n=40, scale=10.0),
    channels         = (lax.ChannelSpec(l=0, threshold=0.0, mass_factor=HBAR2_2MU),),
    solvers          = ("spectrum",),          # no "smatrix" — we call spectral directly
    energies         = energies_grid,
)

def make_V_at(E, params):
    """Energy-dependent potential (dispersive optical model style)."""
    α, β, γ = params["alpha"], params["beta"], params["gamma"]
    kernel = lambda r1, r2: (-2*β*(α+β)**2 * jnp.exp(-β*(r1+r2)) + γ*E) * HBAR2_2MU
    return lax.assemble_nonlocal(solver.mesh, kernel)

params = {"alpha": 0.23, "beta": 1.39, "gamma": 0.01}

# Build V at each grid energy and take the spectrum:
V_grid    = jax.vmap(lambda E: make_V_at(E, params))(energies_grid)  # (21, 1, 1, 40, 40)
spec_grid = jax.vmap(solver.spectrum)(V_grid)                         # batched Spectrum

# Evaluate R(E_i) from spec_i, then S(E_i) from boundary(E_i).
# solver.boundary is a BoundaryValues pytree with field shapes (N_E, N_c).
# jax.vmap slices the leading axis of every field automatically.

a          = solver.mesh.scale
mu         = solver.channels[0].mass_factor

def S_at_own_energy(spec_i, E_i, bdy_i):
    """Compute S(E_i) using the spectrum computed at E_i."""
    R_i = lax.spectral.rmatrix_from_spectrum(spec_i, E_i, a, mu)
    return lax.spectral.smatrix_from_R(R_i, bdy_i)

S_grid = jax.vmap(S_at_own_energy)(spec_grid, energies_grid, solver.boundary)
# S_grid: (21, N_c, N_c) — S at each compile-time energy using V(E_i)

# Padé-interpolate to any energy (S is smooth here):
S_of_E = lax.spectral.pade_interpolate(S_grid, energies_grid)
S_fine = jax.vmap(S_of_E)(jnp.linspace(0.1, 50.0, 2000))             # (2000, N_c, N_c)
```

Key points:
- `solver.boundary` is vmapped over in `S_at_own_energy`: each call receives `bdy_i` with field shapes `(N_c,)` (the per-energy slice).
- `solver.smatrix(spec)` is intentionally NOT used here — that function always evaluates S at all N_E compile-time energies from a single energy-independent spectrum.
- `energy_dependent=True` in `compile()` is informational metadata only; the runtime API is unchanged. The user signals intent through how they call `vmap`.
- Differentiating through `S_grid` with respect to `params` works because the entire chain (`make_V_at` → `solver.spectrum` → `S_at_own_energy`) is JAX-traceable.

### 16.8 Example 7: complex V on GPU via `rmatrix_direct`

For an optical-model potential (complex V) on GPU, the `eig` fallback is not GPU-ready. The user picks the linear-solve method explicitly. `rmatrix_direct(V)` returns the R-matrix at all compile-time energies in one vectorized call.

```python
solver = lax.compile(
    mesh         = lax.MeshSpec("legendre", "x", n=60, scale=14.0),
    channels     = (lax.ChannelSpec(l=20, threshold=0.0, mass_factor=20.736/4),),
    operators    = ("T+L",),
    solvers      = ("rmatrix_direct",),
    energies     = jnp.array([10.0, 20.0, 30.0, 40.0, 50.0]),   # MeV
    V_is_complex = True,
    method       = "linear_solve",
    device       = "gpu:0",
)

def V_optical(r):
    """α + 208Pb optical potential [Descouvemont §5.4, eq. 47], in MeV."""
    V0, W0 = 100.0, 10.0
    R_nuc  = 1.1132 * (208**(1/3) + 4**(1/3))
    a_nuc  = 0.5803
    f      = 1.0 / (1.0 + jnp.exp((r - R_nuc) / a_nuc))
    # Point-Coulomb: Z1=2, Z2=82, e²=1.44 MeV·fm
    R_c = R_nuc
    V_coul = jnp.where(r > R_c,
                       2 * 82 * 1.44 / r,
                       2 * 82 * 1.44 / (2*R_c) * (3 - (r/R_c)**2))
    return -V0 * f - 1j * W0 * f + V_coul

V = V_optical(solver.mesh.radii)[None, None, :]        # (1, 1, 60) complex

# Per-energy linear solve, no spectrum:
R_direct = solver.rmatrix_direct(V)                    # (5, 1, 1) complex

# S-matrix using the spectral submodule's matching directly:
S_direct = jax.vmap(lax.spectral.smatrix_from_R)(
    R_direct,
    solver.boundary,           # BoundaryValues pytree — vmap slices (N_E, N_c) → (N_c,)
)                              # (5, 1, 1)
# Reproduces Appendix A of Descouvemont [2].
```

`rmatrix_direct` returns the R-matrix at all compile-time energies; there is no `Spectrum` and `solver.greens` / `solver.wavefunction` are unavailable. To also get Green's functions for complex V, recompile with `method="eig"` accepting the CPU fallback.

---

## 17. JAX considerations

### 17.1 Pytree registration

All numerical-data dataclasses (`Mesh`, `OperatorMatrices`, `BoundaryValues`, `TransformMatrices`, `Spectrum`) are registered using the `@jax.tree_util.register_dataclass` decorator (JAX >= 0.4.36). Fields default to pytree leaves; structural fields are marked with `field(metadata={"static": True})`. Use `jax.tree.map` (not the deprecated `jax.tree_util.tree_map`) when walking pytrees in library code. This registration ensures:

- Tracing-time fields (`n`, `family`, `regularization`, `is_hermitian`) are baked into the JIT cache.
- Numerical leaves flow through `jit`, `vmap`, `grad` transparently.
- A `Spectrum` returned from one function can be fed into another as a single argument — JAX traces through it without manual unpacking.

`Solver` is **not** a pytree (it holds Python callables). It is a plain frozen dataclass used as a namespace. Those callables must remain importable / reconstructible so a compiled solver is pickleable.

### 17.2 Method dispatch and tracing

`method` is a compile-time choice baked into the JIT'd `spectrum` kernel. Changing it requires rebuilding the `Solver`. The three methods have different traceability properties:

| Method | Internally calls | Differentiable? | `vmap` over V? | GPU? |
|---|---|---|---|---|
| `eigh` | `jnp.linalg.eigh` | Yes (closed-form JVP) | Yes | Yes |
| `eig`  | `jnp.linalg.eig` (via host callback) | Yes (custom JVP in JAX) | Awkward | No |
| `linear_solve` | `jnp.linalg.solve` per energy | Yes | Yes | Yes |

The `eigh` derivative rule has a known degeneracy issue at eigenvalue crossings — gradients become large or ill-defined when two eigenvalues collide [see `jax.experimental.linalg.eigh` notes]. For potential-fitting workflows this is rarely problematic in practice (level crossings as a function of potential parameters are measure-zero), but it's worth documenting.

### 17.3 Dtype and precision

Default `float64` for everything except where promoted to complex. JAX disables `float64` by default; the library calls `jax.config.update("jax_enable_x64", True)` at import time.

For the Yamaguchi benchmark, achieving the published accuracy ($\delta = 85.634560°$ to 6 digits) requires `float64`. `float32` is offered as a fast path for fits where 4-digit accuracy in the loss is sufficient — set `dtype=jnp.float32` in `compile()`.

### 17.4 Devices and sharding

The `device` argument to `compile()` places all cached arrays on the requested device. Subsequent solver calls receive `V` and produce outputs on the same device. For multi-device sharding the user wraps `solver.spectrum` in `jax.shard_map` or uses `jax.pmap` on the leading batch axis of a vmap'd call.

### 17.5 Gradient support

The hot path is `eigh` (default) or `solve` (linear-solve fallback). Both have JAX-defined custom JVP/VJP rules and are fully differentiable. The only non-JAX components (mpmath, scipy.special) are confined to compile time, so they do not enter any backward pass.

For second derivatives, `jax.hessian(loss)` works through both `eigh` and `solve`. Be aware that the second derivative of `eigh` involves squared denominators in eigenvalue differences and can be numerically unstable near degeneracies.

### 17.6 JIT caching and recompilation

Recompilation triggers on changes to:

- Mesh structure: `(family, regularization, n)`.
- Channel structure: number and `l` values of channels.
- Number of energies $N_E$.
- Set of requested operators and solvers.
- `method`, `V_is_complex`, `energy_dependent`.

It does *not* trigger on:

- Channel radius `a` or Laguerre scale `h`.
- Energy values (leaves of a fixed-shape array).
- Thresholds `E_c`.
- Potential values, including their dtype within real/complex parity.

The user can rebuild a `Solver` with a different `a` cheaply (same JIT cache) but changing `n` is a full retrace.

---

## 18. Testing strategy and benchmarks

### 18.1 Analytic benchmark table

Each row tests a different combination of (mesh family, regularization, solver kernel, method).

| Benchmark | Mesh | Solver path | Reference | Tolerance |
|---|---|---|---|---|
| 1D harmonic oscillator | Hermite, $N=20$ | spectrum (eigenvalues only) | Baye Table 4 | $10^{-13}$ |
| 3D harmonic oscillator, $\ell=0$ | modified Laguerre $x^2$, $N=20$ | spectrum | exact $E_n = 2n + \ell + 3/2$ | $10^{-12}$ |
| Hydrogen atom, $\ell=0$..$4$ | regularized Laguerre $x$, $N=30$, $h=n/2$ | spectrum | Baye Table 7 | $10^{-10}$ |
| Confined hydrogen $R=2$ | shifted Legendre $x(1-x)$, $N=8$ | spectrum | Baye Table 9, $E_{1s} = -1/8$ | $10^{-13}$ |
| Pöschl-Teller, $\ell=0$ | Legendre $x$, $N=40$, $a=15$ | spectrum → rmatrix → smatrix → phases | analytic $\delta(E)$ | $10^{-8}$ |
| Yamaguchi non-local | Legendre $x$, $N=20$, $a=8$ | spectrum → smatrix → phases | Desc. §5.8: $\delta(10 \text{ MeV}) = 85.6346°$ | $10^{-5}$ |
| α + ²⁰⁸Pb optical (real, GPU) | Legendre $x$, $N=60$, $a=14$ | spectrum → smatrix (`eigh`) | Desc. Appendix A (real part) | $10^{-4}$ |
| α + ²⁰⁸Pb optical (complex, CPU) | Legendre $x$, $N=60$, $a=14$ | spectrum → smatrix (`eig`) | Desc. Appendix A | $10^{-4}$ |
| α + ²⁰⁸Pb optical (complex, GPU) | Legendre $x$, $N=60$, $a=14$ | rmatrix_direct → smatrix | Desc. Appendix A | $10^{-4}$ |
| Coulomb scattering, pure | Legendre $x$, $\eta \neq 0$ | spectrum → phases | pure Coulomb: $\delta = 0$ | $10^{-8}$ |
| E1 strength function | Legendre $x$, $N=30$, $a=12$ | spectrum → greens | Baye Fig. 20 | qualitative |
| Energy-dependent V(E) + Padé | Legendre $x$, $N=40$, sparse grid | spectrum_batch → smatrix → pade | round-trip with dense grid | $10^{-6}$ |

The Yamaguchi test is the keystone end-to-end test: it exercises non-local potential assembly, the spectrum kernel, the spectral R-matrix sum, the Coulomb boundary path, the S-matrix matching, and phase-shift extraction. It must pass before anything else is merged.

The three α + ²⁰⁸Pb rows verify that `eigh` (real), `eig` (complex CPU), and `linear_solve` (complex GPU) all produce the same physical answer.

### 18.2 Property tests

These run on every PR via `pytest` with `hypothesis` for fuzzed inputs:

1. **Hermiticity of $T+L$.** $\|T+L - (T+L)^T\|_F / \|T+L\|_F < 10^{-12}$ for any (mesh, regularization).
2. **Spectrum-vs-direct cross-check.** For any real V, `solver.rmatrix(spec, E)` from the spectral sum agrees with `solver.rmatrix_direct(V)` evaluated at the same E to $10^{-10}$. This is the most valuable consistency test in the suite — catches both spectral-sum bugs and linear-solve bugs simultaneously.
3. **Unitarity of $S$ for real V.** $\|S^\dagger S - I\|_F < 10^{-10}$ for any real potential and any energy.
4. **Symmetry of $S$.** $\|S - S^T\| < 10^{-10}$ for any real symmetric V.
5. **Wronskian.** $FG' - GF' = 1$ at every boundary computation (catches mpmath setup errors).
6. **Pole structure.** For known resonances, the eigenvalues $\varepsilon_k$ from `spectrum(V)` lie near the resonance energy with width related to the imaginary part of the surface amplitudes (as $V$ approaches the resonance condition).
7. **vmap parity.** A Python `for` loop over 100 energies and `vmap(solver.rmatrix)(spec, energies)` produce identical outputs.
8. **JIT cache stability.** Calling `solver.spectrum(V)` twice with different V of the same shape does not recompile.
9. **Autograd correctness.** `jax.test_util.check_grads(loss, ...)` passes for `loss(params) = ||S(params) - S_target||²` with finite-difference vs autograd agreement at 1e-5.
10. **Padé round-trip.** Sampling a smooth observable on a dense grid, fitting on every-third-point, evaluating on the dense grid, and comparing against the original gives $<10^{-6}$ error for analytic functions.
11. **Round-trip mesh ↔ grid.** $\sum_j c_j^2 \approx \int |\psi(r)|^2 dr$ on the fine grid (to LMM accuracy).

### 18.3 Regression tests

A frozen reference dataset (`tests/benchmarks/data/`) contains exact-value comparisons for:

- All entries of Table 9 of Baye [1] (confined hydrogen).
- The first 8 eigenvalues of the 1D HO from Baye Table 4.
- The full output of Descouvemont [2] Appendix A (α + ²⁰⁸Pb collision matrix at 5 energies).
- The Yamaguchi phase shifts at the energy grid from Descouvemont [2] Example 5.

These are compared bit-exact (up to round-off) on every CI run.

### 18.4 Performance regression

Benchmark suite measures, for fixed problem sizes:

- Compile time (mesh + operators + boundary + JIT trace), CPU only.
- Single-shot `solver.spectrum(V)` time, CPU and GPU.
- Energy-scan throughput (problems per second) at various $N$, $N_c$, $N_E$.
- Speedup of `spectrum → smatrix` over `rmatrix_direct → smatrix` as a function of $N_E$.

The expected scaling: `spectrum`-path cost is dominated by one $O((N_c N)^3)$ eigendecomposition; `rmatrix_direct` cost is $N_E$ solves of the same complexity. For $N_E \gtrsim 5$ the spectrum path should always win.

---

## 19. Build order

The recommended sequence for offline development. Each step ends with at least one passing test.

**Phase 1 — Foundation**

1. `types.py` (excluding Spectrum): all dataclasses, pytree registration, dtype handling.
2. `meshes/_registry.py`: registry pattern and dispatch.
3. `meshes/legendre.py`: implement `build_legendre_x` only. **Test:** ports of the kinetic-matrix construction from the user's prototype; verify nodes against scipy roots, weights are positive, $T+L$ is symmetric.

**Phase 2 — The mesh-independent spectral submodule**

4. `spectral/types.py`: `Spectrum` dataclass with pytree registration.
5. `spectral/observables.py`: `rmatrix_from_spectrum`, `smatrix_from_R`, `greens_from_spectrum`, `phases_from_S`. Pure functions, no JAX magic.
6. `spectral/interpolation.py`: `pade_interpolate`. **Test:** round-trip a known smooth function (e.g. sigmoid) sampled on a coarse grid; verify reconstruction error.

**Phase 3 — Spectrum kernel and the keystone test**

7. `boundary/coulomb.py`: `compute_boundary_values` for open channels. **Test:** Wronskian $FG' - GF' = 1$ within mpmath precision.
8. `solvers/spectrum.py`: `make_spectrum_kernel` with `method="eigh"`. Single-channel only first.
9. `solvers/observables.py`: thin closures producing `solver.rmatrix`, `solver.smatrix`, `solver.phases`. **Test:** port `test_yamaguchi.py` from the prototype. *This is the milestone.*

**Phase 4 — The compile factory**

10. `compile.py`: implement `compile()` for the spectrum path. Wire steps 1–9. **Test:** the user-facing examples in §16.2 and §16.4 run as written.

**Phase 5 — Coupled channels, Green's, second mesh family**

11. Extend `make_spectrum_kernel` and observables to coupled channels (`N_c > 1`). **Test:** two-channel n-p $J=1^+$ example from Descouvemont [2, §5.5].
12. `solvers/observables.py` add `make_greens` and `make_wavefunction_internal`. **Test:** Baye §6.5 (E1 strength reproducing Fig. 20).
13. `meshes/laguerre.py`: implement `build_laguerre_x`. **Test:** hydrogen atom eigenvalues to machine precision at $h = n/2$.

**Phase 6 — Transforms and Padé**

14. `transforms/grid.py` + `meshes/_basis_eval.py` for Legendre-$x$. **Test:** $\|c\|^2 \approx \int |\psi|^2$ for an eigenstate.
15. Wire `pade_interpolate` into `compile()` flow. **Test:** energy-dependent-V example from §16.7.
16. `transforms/fourier.py`. **Test:** momentum-space Yamaguchi reproducing the position-space result.

**Phase 7 — Method dispatch: complex V and the linear-solve fallback**

17. Add `method="eig"` path to `make_spectrum_kernel`. **Test:** complex Yamaguchi (artificial imaginary part) reproduces real result in the real-part limit.
18. `solvers/direct.py`: implement `make_rmatrix_direct`. **Test:** real-V cross-check against the spectrum path (property test #2 in §18.2).
19. **Test:** α + ²⁰⁸Pb optical model reproducing Descouvemont Appendix A — all three method paths (`eigh`, `eig`, `linear_solve`) agree.

**Phase 8 — Additional regularizations**

20. Legendre $x(1-x)$ for confined problems. **Test:** Baye Table 9 row by row.
21. Legendre $x^{3/2}$ for hyperspherical applications.
22. Modified Laguerre $x^2$ regularized by $x$ for 3D HO.

**Phase 9 — Production**

23. Closed-channel boundary handling (Whittaker functions); verify against Descouvemont's closed-channel examples.
24. R-matrix propagation [2, §2.4].
25. Performance optimization: complex-symmetric Lanczos in JAX (long-term).

By the end of Phase 4 the library has feature parity with the user's prototype but with the spectral approach and clean architecture. Phases 5–6 add the major value-adds (coupled channels, Green's, Padé). Phase 7 handles complex V. Phases 8–9 are completion and production hardening.

Estimated effort, one person working part-time:

- Phases 1–3: 2 weeks.
- Phase 4: 1 week.
- Phases 5–6: 2 weeks.
- Phase 7: 2 weeks.
- Phases 8–9: open-ended.

---

## 20. References

[1] **D. Baye**, *The Lagrange-mesh method*, **Physics Reports 565**, 1–107 (2015). DOI: [10.1016/j.physrep.2014.11.006](https://doi.org/10.1016/j.physrep.2014.11.006).

The definitive review of the LMM. Section 2 is the theoretical foundation; Section 3 provides explicit formulas for every mesh family and regularization; Sections 5–6 cover bound-state and continuum applications. This document references specific equations as "[Baye eq. X.Y]" or sections as "[Baye §X]".

[2] **P. Descouvemont**, *An R-matrix package for coupled-channel problems in nuclear physics*, **Computer Physics Communications 200**, 199–219 (2016). DOI: [10.1016/j.cpc.2015.10.015](https://doi.org/10.1016/j.cpc.2015.10.015). arXiv: [1510.03540](https://arxiv.org/abs/1510.03540).

The Fortran R-matrix package whose architecture and Lagrange-Legendre formulas this library mirrors. Section 2 gives the R-matrix-on-Lagrange-mesh formalism; eqs. 14–17 are the central solver equations; eqs. 18–24 give the explicit shifted Legendre-$x$ formulas; §2.4 covers R-matrix propagation; §5 gives validation examples (with Example 5 being the Yamaguchi non-local potential used in this library's primary benchmark).

[3] **A. M. Lane, R. G. Thomas**, *R-matrix theory of nuclear reactions*, **Reviews of Modern Physics 30**, 257 (1958). The original R-matrix formalism with the Wigner-Eisenbud spectral decomposition that this library's `spectral` submodule reflects.

[4] **M. Hesse, J.-M. Sparenberg, F. Van Raemdonck, D. Baye**, *Coupled-channel R-matrix method on a Lagrange mesh*, **Nuclear Physics A 640**, 37–51 (1998). The original coupled-channel R-matrix-on-Lagrange-mesh paper.

[5] **M. Hesse, J. Roland, D. Baye**, *Solution of the Yamaguchi nonlocal problem on a Lagrange mesh*, **Nuclear Physics A 709**, 184–195 (2002). Original non-local potential application; reference values for the Yamaguchi benchmark.

[6] **P. Descouvemont, D. Baye**, *The R-matrix theory*, **Reports on Progress in Physics 73**, 036301 (2010). General review of R-matrix theory in nuclear physics; discusses phenomenological R-matrix fitting in terms of poles and reduced widths, which are directly accessible from the `Spectrum` object.

[7] **G. A. Baker Jr., P. Graves-Morris**, *Padé Approximants*, 2nd ed., Cambridge University Press (1996). Reference for the Padé interpolation construction used in `spectral.interpolation`.

---

## Appendix A: Mesh formula tables

This appendix collects the formulas needed to implement each mesh + regularization builder. All formulas are referenced to Baye [1] or Descouvemont [2].

### A.1 Shifted Legendre on $(0, a)$, regularized by $x$  [Baye §3.4.5, Descouvemont eqs. 18–24]

| Quantity | Formula | Reference |
|---|---|---|
| Mesh points | $P_N(2x_i - 1) = 0$ on $(0, 1)$; $r_i = a x_i$ | Baye 3.120 |
| Weights | $\hat\lambda_i = \frac{1}{4 x_i (1-x_i) [P'_N(2x_i-1)]^2}$ | Baye 3.121 |
| Basis | $\hat f_j(r) = \frac{(-1)^{N-j}}{\sqrt{a x_j(1-x_j)}} \cdot \frac{r/a \cdot P_N(2r/a - 1)}{r/a - x_j}$ (with $\nu=1$) | Desc. 18 |
| Boundary | $\varphi_n(a) = (-1)^{N+n}\sqrt{1/[a x_n(1-x_n)]}$ | Desc. 24 |
| $T+L(B)$ diag | $\frac{1}{a^2 x_n(1-x_n)} \cdot \frac{(4N^2+4N+3)x_n(1-x_n) - 6x_n + 1}{3 x_n(1-x_n)} - \frac{B}{a^2 x_n(1-x_n)}$ | Desc. 22 |
| $T+L(B)$ off | $\frac{1}{a^2}\frac{(-1)^{n-m}}{\sqrt{x_n(1-x_n) x_m(1-x_m)}}\left[ N^2+N+1 + \frac{x_n+x_m-2x_n x_m}{(x_n-x_m)^2} - \frac{1}{1-x_n} - \frac{1}{1-x_m} - B\right]$ | Desc. 23 |
| $1/r$ exact | $\langle \hat f_n | 1/r | \hat f_m \rangle$ Gauss-exact: $\delta_{nm}/r_n$ | Baye 3.140 (analog) |
| $d/dr$ exact diag | $\hat D_{ii} = \frac{1}{2a x_i(1-x_i)}$ | Baye 3.124 |
| $d/dr$ exact off | $\hat D_{i \neq j} = \frac{(-1)^{i-j}}{a}\sqrt{\frac{x_i(1-x_j)}{x_j(1-x_i)}} \cdot \frac{1}{x_i - x_j}$ | Baye 3.123 |

### A.2 Laguerre on $(0, \infty)$, regularized by $x$, $\alpha = 0$  [Baye §3.3.4]

| Quantity | Formula | Reference |
|---|---|---|
| Mesh points | $L^0_N(x_i) = 0$; $r_i = h x_i$ | Baye 3.50 |
| Weights | $\hat\lambda_i = \frac{e^{x_i}}{N [L^0_{N-1}(x_i)]^2 x_i}$ | Baye 3.51 (regularized form) |
| Basis | $\hat f_j(r) = (-1)^j \sqrt{h/x_j} \cdot \frac{L^0_N(r/h)}{r/h - x_j} \cdot (r/h) e^{-r/2h}$ | Baye 3.70 |
| Boundary | $0$ (interval is open) | – |
| $\hat T$ Gauss diag | $-\frac{1}{12 h^2 x_i^2}\left[x_i^2 - 2(2N+1)x_i - 4\right]$ | Baye 3.76, $\alpha=0$ |
| $\hat T$ Gauss off | $\frac{(-1)^{i-j}}{h^2} \cdot \frac{x_i + x_j}{\sqrt{x_i x_j}(x_i - x_j)^2}$ | Baye 3.75 |
| Exact $T$ correction | $\hat T_{ij} = \hat T^G_{ij} - \frac{(-1)^{i-j}}{4 h^2 \sqrt{x_i x_j}}$ | Baye 3.77 |
| $1/r$ Gauss diag | $1/r_i$ | – |
| $1/r$ Gauss off | $0$ (exact for $\alpha = 0$) | Baye 3.61 |

### A.3 Shifted Legendre on $(0, a)$, regularized by $x(1-x)$  [Baye §3.4.7]

For confined systems where the wave function vanishes at both endpoints.

| Quantity | Formula | Reference |
|---|---|---|
| Mesh points | $P_N(2x_i - 1) = 0$; $r_i = a x_i$ | – |
| Weights | $\hat\lambda_i = 1 / [4 x_i (1-x_i) (P'_N(2x_i-1))^2]$ | – |
| Basis | $\hat f_j(r) = (-1)^{N-j} \frac{x(1-x)}{\sqrt{x_j(1-x_j)}} \frac{P_N(2x-1)}{x - x_j}$ with $x = r/a$ | Baye 3.138 |
| $T$ Gauss diag | $\frac{1}{a^2 \cdot 3 x_i(1-x_i)}[N(N+1) + 1/(x_i(1-x_i))]$ | Baye 3.143 |
| $T$ Gauss off | $\frac{1}{a^2} \cdot \frac{(-1)^{i-j}(x_i + x_j - 2 x_i x_j)}{R_{ij}(x_i - x_j)^2}$ where $R_{ij} = \sqrt{x_i(1-x_i)x_j(1-x_j)}$ | Baye 3.142 |
| Exact $T$ correction | $T_{ij} = T^G_{ij} - \frac{(-1)^{i-j} N(N+1)}{a^2(2N+1) R_{ij}}$ | Baye 3.144 |
| Overlap (off-diag) | nonzero, see Baye 3.139 — but treat as orthonormal for LMM | – |

### A.4 Modified Laguerre $t = x^2$, regularized by $x$  [Baye §3.3.7]

For 3D harmonic-oscillator-like problems.

| Quantity | Formula | Reference |
|---|---|---|
| Mesh points | $L^\alpha_N(x_i^2) = 0$ | Baye 3.82 |
| Basis | $\hat f_j(r) = \frac{r}{x_j} \cdot \text{(modified Laguerre eq. 3.83)}$ | Baye 3.92 |
| $\hat T$ Gauss diag | $\frac{1}{3 h^2}[-x_i^2 + 2(2N+\alpha+1) - (\alpha^2 - 3/4)/x_i^2]$ | Baye 3.94 |
| $\hat T$ Gauss off | $\frac{(-1)^{i-j} \cdot 4(x_i^2 + x_j^2)}{h^2 (x_i^2 - x_j^2)^2}$ | Baye 3.93 |

---

## Appendix B: Glossary of symbols

| Symbol | Meaning |
|---|---|
| $N$ | Number of mesh points per channel |
| $N_c$ | Number of coupled channels |
| $M = N_c \cdot N$ | Total dimension of the block Hamiltonian |
| $N_E$ | Number of energies in compile-time grid |
| $a$ | Channel radius (for Legendre meshes on finite interval) |
| $h$ | Scale parameter (for Laguerre meshes on $(0, \infty)$) |
| $r_i$ | Physical radial mesh point ($r_i = a x_i$ or $r_i = h x_i$) |
| $x_i$ | Canonical mesh point in canonical interval |
| $\lambda_i$ | Gauss weight (Baye eq. 2.6) |
| $f_j(x)$ | Lagrange function (unregularized) |
| $\hat f_j(x)$ | Regularized Lagrange function |
| $\varphi_n(r)$ | Descouvemont's basis function for the Legendre-$x$ mesh, eq. 18 |
| $T$ | Kinetic energy operator $-\hbar^2/2\mu \cdot d^2/dr^2$ |
| $L(B)$ | Bloch surface operator, $L(B) = \hbar^2/2\mu \cdot \delta(r-a)(d/dr - B/r)$ [Desc. eq. 8] |
| $T+L$ | Bloch-augmented kinetic operator (Hermitian on $[0,a]$ for real V) |
| $\mathcal{H}$ | Block-channel Bloch-augmented Hamiltonian, dimension $M \times M$ |
| $V_{ij}(r)$ | Coupling potential between channels $i$ and $j$ |
| $W(r, r')$ | Non-local potential kernel |
| $\varepsilon_k$ | $k$-th eigenvalue of $\mathcal{H}$ (R-matrix pole / Wigner-Eisenbud level) |
| $u^{(k)}$ | $k$-th eigenvector of $\mathcal{H}$ |
| $\gamma_{kc}$ | Surface amplitude of mode $k$ in channel $c$ (reduced width) |
| $Q$ | Surface projector matrix, $(M, N_c)$ |
| $R$ | R-matrix [Desc. eq. 15] |
| $S = U$ | Collision (scattering) matrix |
| $G(E)$ | Green's function (resolvent) of $\mathcal{H}$ |
| $F_L, G_L$ | Regular and irregular Coulomb functions |
| $H_\pm = G \pm iF$ | Outgoing/incoming Coulomb (Hankel) functions |
| $H_\pm'$ | $(d/d\rho) H_\pm$ evaluated at $\rho = ka$, multiplied by $\rho$ |
| $W_{-\eta, \ell+1/2}$ | Whittaker function (for closed channels) |
| $B_c$ | Bloch boundary parameter for channel $c$ [Desc. eq. 9] |
| $\delta$ | Phase shift |
| $\eta$ | Sommerfeld parameter $Z_1 Z_2 e^2 / (\hbar v)$ |
| $\hbar^2 / 2\mu$ | Mass factor in MeV·fm² (e.g. 20.736 for nucleon mass) |
| $\ell_c$ | Orbital angular momentum of channel $c$ |
| $E_c$ | Threshold energy of channel $c$ |
| $k_c$ | Wave number in channel $c$, $k_c^2 = (E - E_c)/(\hbar^2/2\mu_c)$ |
| $(p, q)$ | Padé numerator/denominator orders |

---

## Appendix C: Implementation sharp edges

This appendix collects gotchas that are easy to miss and would require re-reading the paper or the JAX documentation to resolve. Read this before starting each phase.

### C.1 Sign conventions for surface amplitudes and boundary values

Three sign conventions must be consistent or the R-matrix will have the wrong sign or complex phase:

1. **Boundary values `φ_n(a)` sign** [Desc. eq. 24]: $\varphi_n(a) = (-1)^{N+n} / \sqrt{a x_n(1-x_n)}$. With 0-based indexing (n = 0..N-1), this becomes $(-1)^{N+(n+1)}$. The formula in `build_legendre_x` produces the correct alternating pattern — do not change it without re-verifying against the Yamaguchi test.

2. **Q matrix signs carry through to γ**: `γ = U.T @ Q` inherits the signs of Q. `rmatrix_from_spectrum` computes `einsum("m,mc,md->cd", 1/(ε-E), γ, γ)`. Since R enters the S-matrix as `Hm - R @ Hmp`, any global sign error in R will produce phases shifted by π and appear as `δ → -δ` or `δ → π - δ` in the benchmark.

3. **Denominator sign in spectral sums**: `1/(ε_k - E)` (not `1/(E - ε_k)`). This gives `C^{-1} = (H-E)^{-1}` matching Descouvemont eq. 14 where `C = H - E`. A sign flip here also produces `δ → -δ`.

The Yamaguchi test at `E = 0.1 MeV` produces `δ = -15.08°` (negative!) — this makes it a good sign discriminator. A positive value indicates a sign error somewhere in the chain.

### C.2 Hermitian vs complex-symmetric normalization

For `method="eig"` (complex symmetric H), the eigenvectors satisfy the bilinear orthogonality condition $U^T U = I$, not $U^\dagger U = I$. `jnp.linalg.eig` returns eigenvectors normalized under the usual $L^2$ inner product ($U^\dagger U = I$), not the bilinear one. The normalization step in the spectrum kernel:

```python
U_norm = U / jnp.sqrt(jnp.diag(U.T @ U))[None, :]
```

divides each column by $\sqrt{(U^T U)_{kk}}$, converting to bilinear normalization. This is safe as long as $(U^T U)_{kk} \neq 0$, which holds generically but can fail for nearly-degenerate eigenvalues. The `_eig_via_callback` wrapper also promotes the real Hamiltonian to complex before calling `eig`; ensure `H.astype(jnp.complex128)` is called before the callback.

### C.3 JAX static vs dynamic fields — the `scale` trap

`Mesh.scale` (the channel radius `a` or Laguerre scale `h`) is marked as a **static field**. This means changing `a` invalidates the JIT cache. The compile factory builds meshes with a specific `a`; if the user calls `compile()` twice with different `a` values, they get two separate `Solver` objects with two separate JIT caches — this is correct behavior, but the compile cost (including `mpmath` for boundary values) is paid again. Do not try to make `a` dynamic to avoid recompilation; the operator matrices and basis-evaluation matrices fundamentally depend on `a`.

### C.4 `assemble_nonlocal` vs raw array input

The Yamaguchi kernel:
```python
def yamaguchi(r1, r2):
    return -2 * BETA * (ALPHA + BETA)**2 * exp(-BETA * (r1 + r2)) * HBAR2_2MU
```
returns values in **MeV** (because of the `* HBAR2_2MU` factor). When passed to `assemble_nonlocal`, the result is `V[0,0,i,j] = kernel(r_i, r_j) * sqrt(λ_i λ_j) * a` in MeV. Inside the Hamiltonian assembler, this gets divided by `mass_factor = HBAR2_2MU`, giving the correct fm⁻² matrix element. This matches the `test_yamaguchi.py` prototype exactly.

If the user provides a kernel already divided by ℏ²/2μ (i.e. in fm⁻¹), they should set `mass_factor=1.0` in `ChannelSpec`, which is an unusual but valid convention.

### C.5 Padé conditioning and the choice of `N_E`

The default order `(N_E//2 - 1, N_E//2)` uses all available information but produces a Padé matrix whose condition number grows like `Vandermonde(s)^2`. For energy grids spanning more than a decade, Vandermonde matrices become severely ill-conditioned. Mitigations:

- Use Chebyshev-spaced knots (`jnp.cos(jnp.linspace(0, π, N_E))` mapped to the energy range) rather than uniform spacing. This keeps the effective Lebesgue constant small.
- For `N_E > 20` or wide ranges, prefer lower-order Padé `(p, q)` with `p + q + 1 < N_E` and least-squares fitting (`jnp.linalg.lstsq` instead of `solve`).
- Always check that `abs(b_coeffs)` has no roots near the interpolation interval (Froissart doublets).

### C.6 `eigh` derivative at near-degenerate eigenvalues

The VJP of `jnp.linalg.eigh` involves terms of the form $1/(\varepsilon_i - \varepsilon_j)$. When two eigenvalues are nearly equal (which can happen for deep bound states in a large basis), the gradient spikes. For fitting workflows, regularize by:

```python
# Soft degenerate eigenvalue regularizer:
eps_reg = 1e-6   # in fm⁻²
# This is NOT built into the library — add to the user's loss function if needed.
```

Alternatively, use `jax.experimental.linalg.eigh_generalized` or add a small random perturbation to H before differentiation (breaks exact symmetry but stabilizes gradients).

### C.7 Complex Coulomb and Sommerfeld parameter

For charged particles with $\eta \neq 0$, `compute_boundary_values` calls `mpmath.coulombf(l, eta, rho)` and `mpmath.coulombg(l, eta, rho)`. With `dps=40` these are reliable for all $l$ and moderate $\eta$. Two known edge cases:

- **Very large $\eta$ (heavy-ion Coulomb)**: `mpmath` may be slow (> 1 s per evaluation) for $\eta \gg 1$ at low energy. Increase `dps` or use a dedicated asymptotic expansion.
- **Very small $\rho = ka$ (sub-barrier)**: When $ka \ll 1$ the Coulomb functions are dominated by the centrifugal barrier. `mpmath` handles this correctly but returns very large $G_L$ and very small $F_L$. The R-matrix then involves differences of large numbers; the `dps=40` setting provides sufficient guard digits.

### C.8 The `is_open` mask and closed channels in v1

In v1, closed-channel rows/columns of the S-matrix are masked to zero in `_project_open` (not decoupled via the Whittaker boundary condition method of [2, eq. 9]). This is exact when the Bloch boundary parameter $B_c$ is set to eliminate the $L(B_c) u_{ext}$ term for closed channels. Until Phase 9 implements the Whittaker path:

- For energies where all channels are open, results are exact.
- For energies where some channels are closed but far from threshold, the masking approximation is good.
- For energies very close to a channel threshold, small systematic errors may appear (the eigenvectors of H are not aware of the closed-channel matching condition). Flag these energies by checking `solver.boundary.is_open`.

### C.9 `Spectrum` pytree vmap behavior

When `jax.vmap(f)(batched_spectrum)` is called, JAX treats `Spectrum` as a pytree and maps over the leading axis of each array field:
- `eigenvalues`: `(B, M)` → each call sees `(M,)`
- `surface_amplitudes`: `(B, M, N_c)` → each call sees `(M, N_c)`
- `eigenvectors`: `(B, M, M)` or `None`
- `is_hermitian`: static, not batched (same bool for all)

This means `jax.vmap(solver.spectrum)(V_batch)` where `V_batch` has shape `(B, N_c, N_c, N, N)` produces a `Spectrum` with leading batch axis `B` — directly usable in `jax.vmap(S_at_own_energy)(spec_grid, ...)` as in Example 16.7.

`BoundaryValues` vmaps the same way: a `BoundaryValues` with `H_plus` of shape `(N_E, N_c)` when sliced under `jax.vmap` gives per-energy slices of shape `(N_c,)`.

### C.10 `jax.config.update("jax_enable_x64", True)`

This must be called **before any JAX operation**. The library calls it at the top of `lax/__init__.py`:

```python
import jax
jax.config.update("jax_enable_x64", True)
```

If the user imports JAX before importing `lax`, the call may not take effect (JAX freezes the config on first array creation). The fix is to import `lax` first, or to put the config call at the top of the user's script before any `import jax.numpy as jnp` usage. Document this prominently in the README.
---

*End of design document. Version 1.2, intended for offline reference during library development.*

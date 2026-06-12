# `lax`: A JAX-Compiled Lagrange-Mesh Library for Quantum Scattering and Bound-State Problems

**Design Document v1.6**

## Revision history

- **v1.6** — Batched two-state evaluation and grid wavefunctions (spec v0.1.5.1, F1–F3). **F1:** `solver.matrix_element(bra, ket, operator=None, *, conjugate)` — a two-state, optionally non-conjugated bilinear form batched over the block/energy axes, bound unconditionally on every solver, plus a standalone `lax.transforms.matrix_element` (§13.4). **F2:** `solver.wavefunction_grid(spectrum, channel_index=0)` (spectral, both evaluation regimes via internal rank detection) and `solver.wavefunction_direct_grid(V, channel_index=0)` (linear-solve companion), with the Descouvemont eq.-27 source stack baked at compile time on `Solver.wavefunction_sources` and `lax.make_wavefunction_source_grid` exposing it (§11.2–11.3). **F3:** the `momenta=` × `blocks=` rejection is lifted — `F_momentum` gains a leading `(N_b,)` axis and the `grid=` projection kernels generalize to arbitrary leading batch axes (§13, §15.5). **C4 fix and guard:** the spectrum kernel now assembles each per-energy Hamiltonian with its own μ(E) when a non-uniform `mass_factor_grid` is compiled (previously the energy-batched path silently used a single uniform μ), `spectrum(V)` auto-routes to the energy-batched path on such solvers (`V.energy_dependent or mass_factor_nonuniform`), and the five static-regime spectral observables are bound to raising stubs (Appendix C.10). **C8:** the `eig` path's non-differentiability (host `pure_callback`) is documented; gradient/UQ pipelines use `linear_solve` + `wavefunction_direct_grid` (Appendix C.11).
- **v1.5** — Generalized the (previously design-only) partial-wave axis into a **symmetry-block batch axis** (§15.5). Any set of *symmetry blocks* — `(J, π)` coupled-channel groups, individual partial waves, or any other independent solves that share a channel shape `N_c` — is declared through a new `compile(blocks=…)` argument, stacked on a leading `(N_b,)` axis, and `vmap`-ped at runtime. This is the energy-axis mechanism (§4.2) applied along a second batch axis; partial waves are the `N_c = 1` special case. The `Interaction` gains a static `block_dependent` flag parallel to `energy_dependent`, with block shapes `(N_b, M, M)` / `(N_b, N_E, M, M)`. **Status: implemented and shipped** — both the direct path (`rmatrix_direct`/`smatrix_direct`/`phases_direct`/`wavefunction_direct`) and the full spectral path (`spectrum`, `rmatrix`, `smatrix`, `phases`, `greens`, `wavefunction`, `eigh`, the `*_grid` observables) vmap over the block axis; the spectral path requires one uniform mass factor across all blocks, per-block μ remains a direct-path feature (see §15.5). This revision also reconciles the document with the shipped code: core types moved to `lax/types.py` and `BoundaryValues` to `lax/spectral/types.py` (`boundary/_types.py` deleted); the explicit `solver.local_potential` / `solver.nonlocal_potential` builders (no arity inference, no `solver.potential`); `interaction_from_funcs(nonlocal_=…)` (the keyword `nonlocal` is invalid Python); `smatrix_from_R`'s √k normalization (`R̃ = K R K⁻¹`, `K = diag(√k_c)`) and the extra `BoundaryValues.k` field; per-channel μ scaling in `wavefunction_direct`; single-channel coupling sugar for the list builders; `dtype`/`device` compile parameters; R-matrix propagation (`propagate.py`) promoted from a non-goal to a documented module; and several Appendix A formula fixes. A post-implementation review (2026-06-11) verified the batched paths against per-block compiled solvers and closed the remaining open items (single-channel coupling sugar for the list builders; `dtype`/`device` compile parameters — both now implemented and tested). The phased build order that guided the implementation (former §19) was retired at the v1.5 close-out, with every phase through 11 shipped. The Padé interpolation utilities (`spectral.pade_interpolate` and the solver-bound `interpolate_*` builders, including the planned phase-12 derivative-enhanced variant) were subsequently **removed**: interpolation is observable-specific — global rational fits are defeated by the mod-π branch structure of phase shifts and by thresholds — so off-grid evaluation is left to the user (§12).
- **v1.4** — Designed the **partial-wave (ℓ) batch axis** (§15.5, build-order Phase 11): a compile-time `partial_waves` set with baked per-wave centrifugal and boundary, a leading partial-wave axis on `Interaction` (parallel to `energy_dependent`), and partial-wave-vmapped direct observables. Distinct from the coupled-channel axis (independent solves, not coupling). Motivated by ℓ-dependent non-local kernels in downstream optical-model workflows. **This was a design only — it was never implemented; v1.5 supersedes it with the more general symmetry-block axis.**
- **v1.3** — Unified interaction interface and direct-path wavefunctions. The canonical solver input is now an assembled `Interaction` block `(N_E, M, M)` built by `interaction_from_{block,array,funcs}`; raw `(N_c,N_c,N[,N])` arrays are no longer accepted by solvers. Added internal wavefunctions on the linear-solve path (`wavefunction_direct`). Reworked the block assembler to the **symmetric MeV form** (mass baked into the kinetic block, coupling potential left untouched), which fixes the multi-mass asymmetry and lifts the single-μ limitation: **per-channel and energy-dependent reduced mass** (`ChannelSpec.mass_factor` per channel, `mass_factor_grid` shape `(N_E, N_c)`) are now first-class on the direct path. Updated §8, §10.2, §11.2–11.3, §11.5, §15; added Phase 10 to the build order (former Phase 10 → 11).
- **v1.2** — Final pre-implementation review. Fixed unit convention bug (`threshold / mass_factor` in `assemble_block_hamiltonian`); fixed `rmatrix_from_spectrum` and `greens_from_spectrum` to convert E from MeV to fm⁻²; replaced undefined `_project_open`/`_pad_back` with full implementation; rewrote `rmatrix_direct` to vmap over compile-time energies; fixed Example 16.7 (energy-dependent V) which was semantically incorrect; split `to_grid` into two explicit functions; added §15.4 unit convention table; added Appendix C (10 implementation sharp edges).
- **v1.1** — Unified all continuum solvers around a single spectral decomposition kernel. R-matrix and Green's function are computed as spectral sums over the eigenpairs of the Bloch-augmented Hamiltonian. Introduced a mesh-independent `spectral` submodule. Added energy-dependent V(E) compile mode with Padé interpolation. Added method-dispatch (`eigh` / `eig` / `linear_solve`) to handle real, complex, and GPU constraints. Linear-solve R-matrix-direct path moved to its own `rmatrix_direct` namespace.
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
12. [Off-grid energies: interpolation is out of scope](#12-off-grid-energies-interpolation-is-out-of-scope)
13. [Transforms: grid, Fourier, integration](#13-transforms-grid-fourier-integration)
14. [The `compile()` factory](#14-the-compile-factory)
15. [Coupled-channel structure](#15-coupled-channel-structure)
16. [Public API and usage examples](#16-public-api-and-usage-examples)
17. [JAX considerations](#17-jax-considerations)
18. [Testing strategy and benchmarks](#18-testing-strategy-and-benchmarks)
19. [References](#19-references)
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
- **Runtime** (pure JAX; runs many times): push potentials through the precompiled kernels. The kernels return either a `Spectrum` object (one eigendecomposition per V) or, on the R-matrix-direct path, R/S/phases (and internal wavefunctions) directly. Everything is JIT-compatible, vmap-compatible, and grad-compatible.

The **spectral kernel is the runtime currency.** A `Spectrum` from `solver.spectrum(interaction)` is the input to all observables: R-matrix and Green's function are spectral sums; the S-matrix follows from the R-matrix and the precomputed boundary values; phase shifts follow from the S-matrix. One eigendecomposition supports every observable at every energy — a dramatic speedup over per-energy linear solves and an architectural simplification.

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

1. **Spectral kernel is the default runtime primitive.** `solver.spectrum(interaction)` returns a `Spectrum` object; all observables (R, S, G, phases, bound states, wavefunctions) are pure functions of that object plus (for matching-dependent quantities) precomputed boundary values.
2. **Generality across mesh families.** Both Legendre and Laguerre families with their main regularizations supported via a single registry. Adding a new family or regularization requires writing one function.
3. **Multiple solver modes from one mesh.** A single compiled solver bundle supports eigenvalue calculations, R-matrix calculations, S-matrix evaluation, scattering wavefunctions, and Green's-function evaluation.
4. **Two energy modes.** Energy-independent V(E) (compile a spectrum-producing kernel; observables evaluable at any E for spectrum-derived quantities, at the compile-time grid for boundary-value-dependent quantities). Energy-dependent V(E) (user supplies V at each grid point; the library produces observables at the grid — off-grid interpolation is deliberately left to the user, §12).
5. **Arbitrary user-supplied potentials.** Local $V(r)$ and non-local $W(r, r')$, real or complex, single or coupled channel.
6. **Mesh-independent spectral submodule.** Spectral storage, sums, and matching live in `lax.spectral` and depend on nothing in the rest of the package.
7. **Fine-grid / momentum-space / integration helpers.** Conversion of mesh vectors and matrices to finer radial grids or momentum space is precomputed matrix multiplication; integration is trivial in the Lagrange-mesh basis.
8. **Full JAX integration.** Everything inside the runtime hot path is `jit`-, `vmap`-, and `grad`-compatible. Pytree registration explicit and minimal. No `equinox`/`flax` dependency.
9. **Method dispatch for the complex / GPU case.** Real V uses `eigh` (GPU-ready). Complex V uses `eig` (CPU host callback) or the `linear_solve` R-matrix-direct path (GPU+vmap-ready; produces R/S/phases and internal wavefunctions, but no `Spectrum`/Green's function). Complex-symmetric Lanczos in JAX is a future enhancement.
10. **Extensive benchmark coverage.** Yamaguchi non-local, hydrogen atom, 3D harmonic oscillator, confined hydrogen, Pöschl-Teller, Coulomb scattering, α + ²⁰⁸Pb optical, multi-channel n-p — all reproducing published reference values.

### Constraints

**Coulomb and Whittaker functions are not JIT-able.** The Coulomb regular and irregular functions $F_L(\eta, \rho)$, $G_L(\eta, \rho)$ and the Whittaker function $W_{-\eta, \ell+1/2}(\rho)$ are needed at $r = a$ for every channel and every energy. We use `mpmath` for arbitrary-precision evaluation. Since `mpmath` is pure Python, we evaluate boundary values once at compile time over a user-specified energy grid, stack them into JAX arrays of shape `(N_E, N_c)`, and embed them in the `Solver` bundle.

**Consequences.** The energy grid for boundary-value-dependent quantities (S-matrix, scattering wavefunctions, phase shifts) is fixed at compile time. R-matrix and Green's function — which are pure functions of the spectrum — can be evaluated at any runtime energy. Recomputing for a different energy grid means rebuilding the solver; this is cheap relative to the JIT trace time and `mpmath` calls take milliseconds each.

**`jnp.linalg.eig` is CPU-only in current JAX**, which constrains the complex-potential path. See §11 for the method-dispatch policy.

**Non-Hermitian Lanczos in JAX would benefit large complex problems but is non-trivial.** Listed as future work; the R-matrix-direct path (per-energy linear solves) handles the GPU+complex case, producing R/S/phases and wavefunctions but no `Spectrum`.

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
- Three-body hyperspherical solvers [1, §7]. Architecture does not preclude them but they are not implemented.
- Hermite, Jacobi, Fourier, sinc meshes [1, §3.2, §3.5, §3.7]. The registry accommodates them; only Legendre and Laguerre are in v1.
- GPU-ready complex eigendecomposition. v1 uses host callbacks for `eig`; future work may add complex-symmetric Lanczos in pure JAX.

---

## 4. Architecture

### 4.1 Compile time vs runtime

The library has two distinct phases.

**Compile time** runs in plain Python with NumPy and `mpmath`. The user calls `lax.compile(...)` once, specifying:

- Channel structure (per-channel $\ell$, threshold, mass factor). A single coupled-channel group is passed as `channels=[…]`; a **batch of same-shaped symmetry blocks** (independent `(J, π)` groups, partial waves, …) is passed as `blocks=[[…], […], …]` — see §15.5.
- Mesh family, regularization, size, scale.
- Energy grid (required if any boundary-value-dependent observable is requested).
- Energy mode (independent vs dependent) and, optionally, an energy-/channel-dependent reduced mass `mass_factor_grid` (shape `(N_E, N_c)`) for semi-relativistic or multi-mass problems.
- Operators to precompute (`T+L`, `1/r`, `1/r²`, ...).
- Solvers to bake (`spectrum`, `smatrix`, `phases`, `greens`, `rmatrix_direct`, `wavefunction_direct`).
- Optional fine radial grid (for `to_grid_vector`/`to_grid_matrix`) and momentum grid (for `fourier`).
- Numeric `dtype`/`device` and the method (`"eigh"`, `"eig"`, `"linear_solve"`). x64 is enabled globally via `jax.config`; `dtype`/`device` select the precision and placement of the baked arrays (§14.1).

**All kinematics are resolved here, once.** The energy grid, charges `z1z2` (Sommerfeld parameters), per-channel thresholds, and the reduced-mass factors — *including any energy dependence via `mass_factor_grid`* — are baked into the cached boundary values (`k_c, η_c, B_c` per `(E, c)`) and the kinetic-block scaling. Nothing kinematic is supplied at runtime.

The compile step:

1. Builds the mesh via the registry.
2. Precomputes operator matrices (kinetic, position, derivative).
3. Calls `mpmath` to evaluate Coulomb $F, G, F', G'$ on open channels and Whittaker $W, W'$ on closed channels, at every $(E, c)$ pair, using each channel's (and energy's) reduced mass.
4. Optionally precomputes basis-evaluation matrices for grid and momentum transforms.
5. Constructs the requested solver kernels as closures over the cached data, JIT-compiles them, and places them on the target device.
6. Returns a `Solver` pytree.

**Runtime** runs inside JAX. Because kinematics are baked in, **the solver is a pure map from an `Interaction` (the assembled potential block, §15.1) to one or more outputs.** Raw potential arrays are not accepted; the user builds an `Interaction` via `solver.interaction_from_{block,array,funcs}` and passes it to a solve method:

- `spectrum = solver.spectrum(interaction)` runs the eigendecomposition(s); returns a `Spectrum` pytree (one decomposition if the interaction is energy-independent; one per energy if energy-dependent).
- `R = solver.rmatrix(spectrum, E)` — spectral sum at any scalar E.
- `S = solver.smatrix(spectrum)` — shape `(N_E, N_c, N_c)` using the cached boundary values.
- `G = solver.greens(spectrum, E)`, `e, u = solver.eigh(spectrum)`.
- `R = solver.rmatrix_direct(interaction)` — R-matrix-direct path; per-energy linear solve; no `Spectrum`. `smatrix_direct`/`phases_direct`/`wavefunction_direct` likewise.

The `Interaction` carries an `energy_dependent` flag, so a single solve entry point covers both the energy-independent and energy-dependent cases; off-grid energies are handled by recompiling on a new grid or by user-side interpolation of the sampled observables (§12).

### 4.2 The spectral form drives everything

The runtime data flow is:

```
Interaction  ──spectrum──▶  Spectrum  ──┬──▶  rmatrix(E)         ──▶  R(E)
                                        ├──▶  greens(E)          ──▶  G(E)
                                        ├──▶  wavefunction(E)    ──▶  ψ_int(E)
                                        ├──▶  eigh()             ──▶  (ε, u)
                                        └──▶  smatrix()  + boundary ──▶  S
                                                                        │
                                                                        └──▶ phases  ──▶  δ

Interaction  ──rmatrix_direct / smatrix_direct / phases_direct──▶  R / S / δ   (R-matrix-direct path)
Interaction  ──wavefunction_direct(source, energy_index)──▶  ψ_int             (no Spectrum needed)
```

`Spectrum` is the central pytree of the spectrum path. All downstream observables are pure functions in `lax.spectral` of `Spectrum` plus (for boundary-dependent ones) `BoundaryValues`. The R-matrix-direct path reaches R/S/phases/ψ without a `Spectrum`, via per-energy linear solves. Either way the runtime input is an `Interaction` and the kinematics are already baked in.

**Three batch axes, one mechanism.** The same `vmap`-over-a-leading-axis pattern carries the data flow along each of three independent axes, gated by static flags on the `Interaction`: the **energy** axis (`energy_dependent`, §12), the **symmetry-block** axis (`block_dependent`, §15.5), and — within a single block — the **coupled-channel** axis that lives inside the assembled `(M, M)` matrix itself (§15.1). The canonical shape order is **block × channel × energy**; the block and energy axes are `vmap`-ped, the channel axis is solved. No observable takes an `ℓ`/block/energy argument — all three dependencies are carried by the `Interaction`, preserving the "solver = `Interaction` → outputs" invariant.

### 4.3 The `Solver` bundle

`Solver` is a plain frozen dataclass (not a pytree, because it holds callables). It carries:

```
Solver
├── mesh: Mesh                       # nodes, weights, radii, basis_at_boundary
├── operators: OperatorMatrices      # cached single-channel matrices
├── channels: tuple[ChannelSpec]     # static, baked into JIT (ℓ, threshold, mass_factor)
│                                    # a batch of symmetry blocks (§15.5) is stored on
│                                    # solver.blocks, with a stacked (N_b,) axis on boundary
├── energies: jnp.ndarray            # (N_E,) compile-time grid
├── mass_factor_grid: jnp.ndarray|None  # (N_E,) or (N_E, N_c); per-energy/channel μ
├── boundary: BoundaryValues         # (N_E, N_c) cached Coulomb/Whittaker (k_c, η_c, k)
├── transforms: TransformMatrices    # optional B_grid, F_momentum
├── method: str                      # "eigh" | "eig" | "linear_solve"
└── (callables bound at compile time; input is an Interaction):
    ├── spectrum(interaction)  -> Spectrum            # spectrum-path methods
    ├── rmatrix(spec, E)       -> R(E)
    ├── smatrix(spec)          -> S at compile-time E
    ├── phases(spec)           -> δ at compile-time E
    ├── greens(spec, E)        -> G(E)
    ├── wavefunction(spec, E, source) -> ψ_int(E)
    ├── eigh(spec)             -> (ε, u) accessor
    ├── rmatrix_direct(interaction)    -> R           # R-matrix-direct path
    ├── smatrix_direct(interaction)    -> S
    ├── phases_direct(interaction)     -> δ
    ├── wavefunction_direct(interaction, source, energy_index) -> ψ_int
    ├── interaction_from_block / interaction_from_array / interaction_from_funcs
    ├── local_potential / nonlocal_potential   # single-kind function builders
    ├── to_grid_vector / to_grid_matrix / from_grid_vector
    ├── fourier / double_fourier_transform
    └── integrate(...)
```

The bound runtime callables are implemented as **module-level callable objects**, not
local closures, so a compiled solver can be serialized and restored. Every solve method
takes an `Interaction`; kinematics (energies, charges, thresholds, per-channel/energy μ)
are already baked into `boundary` and the kinetic scaling. For an energy-independent
`Interaction`, `solver.spectrum(interaction)` does one eigendecomposition; for an
energy-dependent one it does N_E (the user need not vmap by hand).

---

## 5. Module layout

```
lax/
├── __init__.py            # Public API: compile, MeshSpec, ChannelSpec, Solver, Interaction, ...
├── compile.py             # The compile() factory; main entry point
├── types.py               # Core pytree dataclasses + protocols: Mesh, OperatorMatrices,
│                          #   TransformMatrices, PropagationMatrices, Solver, ChannelSpec,
│                          #   Interaction, and the solver/observable Protocols
├── constants.py           # Physical constants (ℏ²/2m, etc.)
├── _angular.py            # Angular-momentum / spin algebra helpers
├── wavefunction.py        # make_wavefunction_source and scattering-wavefunction helpers
├── propagate.py           # R-matrix subinterval propagation for Legendre-x meshes
│                          #   (build_legendre_x_propagation; PropagationMatrices)
│
├── meshes/
│   ├── __init__.py
│   ├── _registry.py       # (family, regularization) -> builder dispatch
│   ├── legendre.py        # Shifted Legendre: x, x(1-x), x^{3/2}
│   ├── laguerre.py        # Laguerre: x, x^{3/2}, modified-x^2
│   ├── _basis_eval.py     # f_j(x) evaluation for grid/Fourier transforms
│   └── _utils.py          # node/weight helpers
│
├── operators/
│   ├── __init__.py
│   └── interaction.py     # interaction_from_{block,array,funcs} + local/nonlocal_potential
│                          #   builders (the Interaction type itself lives in types.py).
│                          #   There is no separate potential.py / assemble_local/nonlocal;
│                          #   Gauss scaling is folded into the array builder.
│
├── boundary/
│   ├── __init__.py
│   └── coulomb.py         # mpmath Coulomb F, G, F', G' (open) AND Whittaker W, W'
│                          #   (closed) — both live here; there is no whittaker.py.
│                          #   (BoundaryValues now lives in spectral/types.py.)
│
├── models/                # Convenience physics models (optional, not core)
│   ├── __init__.py
│   ├── optical.py         # Rotor / optical-model form factors
│   ├── reid.py            # Reid interaction
│   └── presets.py         # Named compile presets
│
├── spectral/              # ── MESH-INDEPENDENT submodule (depends only on JAX) ──
│   ├── __init__.py
│   ├── types.py           # Spectrum AND BoundaryValues dataclasses (pytrees)
│   ├── observables.py     # rmatrix_from_spectrum, greens_from_spectrum, ...
│   └── matching.py        # smatrix_from_R (√k normalization), open_channel_smatrix_from_R,
│                          #   coupled_channel_parameters_from_S, phases_from_S
│
├── solvers/
│   ├── __init__.py
│   ├── spectrum.py        # The spectrum kernel: eigh/eig dispatch
│   ├── linear_solve.py    # R-matrix-direct path (rmatrix_direct, wavefunction_direct,
│   │                      #   propagated direct solve)
│   ├── assembly.py        # Block-Hamiltonian assembly from operators + V
│   └── observables.py     # Bound observable objects + aligned-grid {rmatrix,smatrix,phases}_grid
│                          #   (in place of solvers/wavefunction.py)
│
└── transforms/
    ├── __init__.py
    ├── grid.py            # mesh <-> radial grid (to_grid_vector/matrix, from_grid_vector)
    ├── fourier.py         # mesh <-> momentum grid (fourier, double_fourier_transform)
    └── integration.py     # norms, expectation values (integrate)
```

Tests live in a top-level `tests/` directory (not inside the package): `tests/benchmarks/`
(Yamaguchi, hydrogen, HO, confined H, α+²⁰⁸Pb, Descouvemont np/¹⁶O–⁴⁴Ca, …), `tests/unit/`
(per-builder unit tests; the spectral tests are `tests/unit/test_spectral*.py`, **not** a
`spectral/tests/` subdirectory), and `tests/property/` (hermiticity, unitarity, autograd,
vmap parity).

### Dependencies

- **Required**: `jax`, `jaxlib`, `numpy`, `scipy`, `mpmath`.
- **Test**: `pytest`, `hypothesis`, `chex` (optional, for shape/dtype assertions).

No `equinox`, no `flax`. Pytree registration uses `jax.tree_util.register_dataclass`.

The library requires `jax>=0.4.36`, which introduced optional `data_fields`/`meta_fields` arguments to `register_dataclass` for `@dataclass` inputs. Fields default to pytree leaves (data fields) unless annotated with `field(metadata={"static": True})`, which marks them as static metadata baked into the JIT cache key. This is the form used throughout the library.

---

## 6. Core types

All public dataclasses are frozen. Numerical-data dataclasses are JAX pytrees. `Solver` is plain (holds callables). `types.py` is the single home for the core types **and** the solver/observable `Protocol`s, plus `Mesh`, `OperatorMatrices`, `TransformMatrices`, `PropagationMatrices`, `Solver`, `ChannelSpec`, and `Interaction`. `BoundaryValues` and `Spectrum` live in `spectral/types.py` so that `lax.spectral` depends only on JAX (the old `boundary/_types.py` is deleted). The sketch below shows the data-bearing types; the `Protocol`s are omitted for brevity.

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
    # Subinterval structure for propagated (Legendre-x) meshes; 1 / n otherwise.
    n_intervals:             int = field(metadata={"static": True})
    basis_size_per_interval: int = field(metadata={"static": True})

    nodes:               jnp.ndarray   # (N,) on canonical interval
    weights:             jnp.ndarray   # (N,) λ_i
    radii:               jnp.ndarray   # (N,) physical r_i
    basis_at_boundary:   jnp.ndarray   # (N,) φ_j(a); zeros for unbounded
    propagation:         "PropagationMatrices | None" = None  # subinterval propagation, or None


# --------------------------------------------------- Propagation matrices (pytree)

@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class PropagationMatrices:
    """Per-subinterval R-matrix propagation matrices for Legendre-x meshes.

    Produced by lax.propagate.build_legendre_x_propagation and cached in the
    Mesh. All matrices are in fm⁻². Carries n_intervals/basis_size_per_interval
    (static), the local nodes/weights, per-interval kinetic blocks, the Bloch
    surface-overlap matrices (blo0/blo1/blo2) and surface projectors (q1/q2).
    """
    n_intervals: int = field(metadata={"static": True})
    basis_size_per_interval: int = field(metadata={"static": True})
    interval_width: float = field(metadata={"static": True})
    local_nodes: jnp.ndarray
    local_weights: jnp.ndarray
    kinetic: jnp.ndarray
    blo0: jnp.ndarray
    blo1: jnp.ndarray
    blo2: jnp.ndarray
    q1: jnp.ndarray
    q2: jnp.ndarray


# ------------------------------------------------------------- Operator cache

@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class OperatorMatrices:
    """Precomputed single-channel (N, N) matrices in fm⁻² units.
    Unrequested operators are None — except `inv_r2`, which the block
    assembler always needs for the centrifugal term and so is always built."""
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
    mass_factor: float | jnp.ndarray = 1.0  # ℏ²/2μ in user units (e.g. MeV·fm²); may be an array


# ----------------------------------------------------------- Boundary values
#
# NOTE: BoundaryValues lives in spectral/types.py (not here), so that the
# mesh-independent lax.spectral submodule depends only on JAX. Reproduced here
# for reference.

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
    The k field carries the channel wave numbers and is consumed by the
    √k S-matrix normalization (§10.3).
    """
    H_plus:    jnp.ndarray   # (N_E, N_c) complex
    H_minus:   jnp.ndarray   # (N_E, N_c) complex
    H_plus_p:  jnp.ndarray   # (N_E, N_c) complex
    H_minus_p: jnp.ndarray   # (N_E, N_c) complex
    is_open:   jnp.ndarray   # (N_E, N_c) bool
    k:         jnp.ndarray   # (N_E, N_c) channel wave numbers k_c(E) in fm⁻¹

# For the symmetry-block axis (§15.5), every BoundaryValues field gains a leading
# (N_b,) axis — shape (N_b, N_E, N_c) — stacked per block at compile time.


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
    mass_factor_grid: jnp.ndarray | None = None   # (N_E,) or (N_E, N_c); per-energy/channel μ
    blocks: tuple[tuple[ChannelSpec, ...], ...] | None = None  # §15.5 block set; None for channels=

    # Callables (filled in by compile()); None if not requested.
    spectrum:        Callable | None = None
    rmatrix:         Callable | None = None
    smatrix:         Callable | None = None
    phases:          Callable | None = None
    greens:          Callable | None = None
    wavefunction:    Callable | None = None
    eigh:            Callable | None = None   # bound whenever the spectrum path is active
    rmatrix_grid:    Callable | None = None   # aligned-grid observables (energy-dependent V)
    smatrix_grid:    Callable | None = None
    phases_grid:     Callable | None = None
    rmatrix_direct:        Callable | None = None   # linear-solve namespace
    smatrix_direct:        Callable | None = None
    phases_direct:         Callable | None = None
    wavefunction_direct:   Callable | None = None   # C⁻¹ s on the direct path (§11.3)
    # Transforms:
    to_grid_vector:    Callable | None = None
    to_grid_matrix:    Callable | None = None
    from_grid_vector:  Callable | None = None
    fourier:           Callable | None = None
    double_fourier_transform: Callable | None = None
    integrate:         Callable | None = None

    # Interaction builders (close over mesh/channels/energies; see §15.1).
    interaction_from_block: Callable | None = None
    interaction_from_array: Callable | None = None
    interaction_from_funcs: Callable | None = None
    local_potential:        Callable | None = None   # fn(r[,E])      -> single-kind Interaction
    nonlocal_potential:     Callable | None = None   # fn(r,r'[,E])   -> single-kind Interaction

    # Convenience properties (not fields): grid_r, momenta; plus __repr__ listing
    # which observables were compiled.
```

> **v1.3 note.** The single `rmatrix_direct`/`smatrix_direct`/`phases_direct` entry points dispatch on whether the supplied `Interaction` is energy-dependent (the former `rmatrix_direct` vs `rmatrix_direct_grid` split is internal). Mass enters per channel/energy through `channels[c].mass_factor` and the optional `mass_factor_grid`; see §11.5 and §15.3.
>
> **v1.5 note.** When a batch of symmetry blocks is compiled (`blocks=`, §15.5), the block set is stored on a new `Solver.blocks` field (`channels` holds the template block `blocks[0]`), `boundary` carries a leading `(N_b,)` axis, and every observable above gains a corresponding leading block axis on its output. `solver.grid_r` and `solver.momenta` are read-only properties proxying `transforms.grid_r`/`transforms.momenta`.

### `Interaction` — the canonical solver input (§15.1)

```python
@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Interaction:
    """Assembled coupled-channel interaction (potential) block, in MeV.

    block : (M, M) or (N_E, M, M),  M = N_c·N
        (and, with the symmetry-block axis, (N_b, M, M) or (N_b, N_E, M, M))
        Local terms on the per-channel diagonal, non-local terms as full
        blocks (Gauss-scaled per Desc. eq. 26). Symmetric and mass-independent:
        per-channel mass is applied to the kinetic/boundary/surface terms inside
        the solver (§11.5), never folded into V. Excludes kinetic, centrifugal,
        threshold, and the energy term — those are the solver's cached job.
    energy_dependent : bool (static)
        True iff `block` carries a leading (N_E,) axis aligned with `energies`.
    block_dependent : bool (static)   [§15.5]
        True iff `block` carries a leading (N_b,) symmetry-block axis. Parallel
        to energy_dependent; composes with it as (N_b, N_E, M, M).

    Two Interactions add term-wise via `+` (`__add__`/`__radd__`): mixed and
    multi-term interactions are composed by summing single-kind builds, with
    energy-dependent promotion when one operand carries the (N_E,) axis. This is
    how the explicit single-kind `local_potential`/`nonlocal_potential` builds
    are combined (§15.1).
    """
    block: jnp.ndarray
    energy_dependent: bool = field(metadata={"static": True})
    block_dependent: bool = field(metadata={"static": True}, default=False)  # §15.5
```

### `Spectrum` lives in the spectral submodule

`spectral/types.py` is the home of both pure-data pytrees the submodule operates on —
`Spectrum` (below) and `BoundaryValues` (§6) — so that `lax.spectral` depends only on JAX.

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
        # Desc. eq. 22 / Appendix A.1: denominator is 3·x²·(1-x)² (then ÷a² below).
        ((4 * n * (n + 1) + 3) * x * (1 - x) - 6 * x + 1) / (3 * x ** 2 * (1 - x) ** 2),
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

Basis-function evaluation for `to_grid_vector`/`to_grid_matrix` and `fourier` is also family-specific and registered in a parallel registry in `meshes/_basis_eval.py`.

---

## 8. Operators

Operators are pre-built matrices stored in `OperatorMatrices`. Each (mesh, regularization) builder fills in the matrices it supports; the formulas are in [1, §3] and Appendix A here.

**Kinetic and Bloch-augmented kinetic.** For Legendre-$x$, `operators.TpL` is the matrix [Desc. eq. 22–23] with $B = 0$. For non-zero $B$, the library updates per channel as $T + L(B) = T + L(0) - B\, b b^T / a$ where $b = $ `basis_at_boundary`. Closed channels use $B_c = 2 k_c a\, W'/W$ [Desc. eq. 9] computed from the cached Whittaker values.

For Laguerre, basis functions vanish at both endpoints, so $T$ is Hermitian on its own; `TpL` and `T` coincide.

**Position operators.** For Legendre-$x$ and Laguerre-$x$ ($\alpha=0$), $1/r$ is Gauss-exact diagonal [Baye eqs. 3.61, 3.140 analog]. $1/r^2$ likewise diagonal [Baye eq. 3.62].

**Centrifugal term.** $\ell(\ell+1)/r^2$ is injected by the Hamiltonian assembler, per channel. The user does not add it manually.

**Derivative.** $D_{ij} = \langle f_i | d/dr | f_j \rangle$ in closed form per family. Useful for momentum-like observables and Bloch off-diagonals if needed.

**Local potential injection.** A local $V_{cc'}(r)$ enters the Hamiltonian as a (diagonal-in-$r$) block with entries $V_{cc'}(r_i)$ in **MeV**, added without any per-channel division. The channel mass factor is applied to the *kinetic* block instead (§11.5, symmetric MeV form), so the assembled block stays symmetric.

**Non-local potential injection.** A non-local $W_{cc'}(r, r')$ enters as a full block with entries $(\lambda_i \lambda_j)^{1/2}\, a\, W_{cc'}(r_i, r_j)$ in **MeV** [Desc. eq. 26] — again with no per-channel division.

There are **no** public `assemble_local`/`assemble_nonlocal` primitives (earlier drafts
exported them from a `operators/potential.py`; that module does not exist). The Gauss
scaling above — `V_cc'(r_i)` on the diagonal and `(λ_i λ_j)^{1/2} a W_cc'(r_i, r_j)` for
non-local kernels — is folded directly into the array builder. The single public way to
build a coupled-channel potential is the `Interaction` interface (§15.1), all of which
lives in `operators/interaction.py`:

- `solver.local_potential(fn)` / `solver.nonlocal_potential(fn)` — single-kind builders
  from a callable (`fn(r[,E])` and `fn(r,r'[,E])`); no arity inference, no `kind=` argument.
- `solver.interaction_from_{block,array,funcs}` — sum any number of
  *(form factor ⊗ channel-coupling-matrix)* terms (local and non-local together) into the
  canonical block. `interaction_from_array` applies the Gauss scaling to each non-local
  term internally.
- Compose mixed/multi-term interactions by adding single-kind builds with `+` (§6).

---

## 9. Boundary values: Coulomb, Hankel, and Whittaker functions

The Coulomb regular $F_L(\eta, \rho)$, irregular $G_L(\eta, \rho)$, and Whittaker $W_{-\eta, \ell+1/2}(\rho)$ functions are needed at $r = a$ for every channel and every energy in the compile-time grid. We use `mpmath` with `dps = 40` by default; this is overkill for double precision but cheap insurance against cancellation when subtracting $H_-$ from $R H_-'$ near sharp resonances.

**Neutral fast path (η = 0).** When there is no Coulomb interaction (`z1z2` is `None`/neutral so $\eta = 0$), the boundary functions reduce to Riccati–Bessel functions. The implementation takes a deliberate fast path here: it evaluates them with double-precision **scipy spherical Bessel** functions and obtains the $\rho$-derivatives from the **Bessel recurrence relations** rather than `mp.diff`. This is exact to double precision (no cancellation guard needed without the Coulomb tail) and far cheaper than the `mpmath` `dps = 40` path, which is reserved for the charged ($\eta \neq 0$) case. Both paths populate the same `BoundaryValues`.

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
    # NOTE: the mpmath path shown here is taken only for charged channels (η ≠ 0).
    # When z1z2 is None/neutral the implementation instead uses the η = 0 fast path
    # (scipy spherical Bessels + recurrence-relation derivatives); see above.
    mp.mp.dps = dps
    n_e, n_c = len(energies), len(channels)
    Hp  = np.zeros((n_e, n_c), dtype=complex)
    Hm  = np.zeros((n_e, n_c), dtype=complex)
    Hpp = np.zeros((n_e, n_c), dtype=complex)
    Hmp = np.zeros((n_e, n_c), dtype=complex)
    is_open = np.zeros((n_e, n_c), dtype=bool)
    k_vals  = np.zeros((n_e, n_c), dtype=float)   # channel wave numbers in fm⁻¹

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
                k_vals[ie, ic] = k
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
                k_vals[ie, ic] = k

    return BoundaryValues(
        H_plus=jnp.asarray(Hp),
        H_minus=jnp.asarray(Hm),
        H_plus_p=jnp.asarray(Hpp),
        H_minus_p=jnp.asarray(Hmp),
        is_open=jnp.asarray(is_open),
        k=jnp.asarray(k_vals),
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

> **Direct-path counterpart (v1.3).** The same internal wavefunction is available without a `Spectrum` on the linear-solve path as `solver.wavefunction_direct` (§11.3): it solves `C(E_i) f = source` for `f = C⁻¹ source`, which is *the same* linear system as the R-matrix solve (different RHS), so it reuses the `C(E_i)` factorization and needs no eigenvectors. The `source` comes from `make_wavefunction_source(solver, channel_index, energy_index)` and the result has the same `(M,)` mesh-coefficient convention as the spectral path, so the two are interchangeable (and `to_grid_vector` reconstructs `u(r)` from either via Desc. eq. 28). This is the GPU-/vmap-native route for complex (optical) potentials, where the spectral `eig` is a CPU callback.

### 10.3 Matching: from R to S

```python
# spectral/matching.py
import jax
import jax.numpy as jnp


def smatrix_from_R(R: jnp.ndarray, boundary_at_E) -> jnp.ndarray:
    """S-matrix from the R-matrix, with the √k normalization of Desc. eq. 17.

    The physical R is first rescaled to the dimensionless R̃ = K R K⁻¹ with
    K = diag(√k_c) (boundary_at_E.k), then matched:
        S = (H_- - R̃ H_-')(H_+ - R̃ H_+')^{-1}    (open subspace).

    boundary_at_E carries length-N_c arrays for one energy E:
        H_plus, H_minus, H_plus_p, H_minus_p, is_open, k
    """
    sqrt_k = jnp.sqrt(boundary_at_E.k)
    K, Kinv = jnp.diag(sqrt_k), jnp.diag(1.0 / sqrt_k)
    R_tilde = K @ R @ Kinv                                  # Desc. eq. 17
    Hp, Hm = jnp.diag(boundary_at_E.H_plus), jnp.diag(boundary_at_E.H_minus)
    Hpp, Hmp = jnp.diag(boundary_at_E.H_plus_p), jnp.diag(boundary_at_E.H_minus_p)
    A = Hm - R_tilde @ Hmp
    B = Hp - R_tilde @ Hpp
    return jnp.linalg.solve(B.T, A.T).T


def phases_from_S(S: jnp.ndarray) -> jnp.ndarray:
    """Phase shifts δ = (1/2) arg(eigvals(S)). Returns (N_c,) in radians."""
    eigvals = jnp.linalg.eigvals(S)
    return 0.5 * jnp.angle(eigvals)
```

The √k rescaling (`K = diag(√k_c)`) is why `BoundaryValues` carries the extra `k` field
(§6). For closed channels, the `is_open` mask drives a projection that's implemented inside
the compile-time-bound solver methods (so it stays static, no dynamic shapes inside JIT);
`open_channel_smatrix_from_R` performs the open-subspace restriction explicitly, and
`coupled_channel_parameters_from_S` extracts Blatt–Biedenharn mixing parameters from a
coupled S-matrix.

### 10.4 The submodule's public surface

```python
# spectral/__init__.py
from .types import Spectrum, BoundaryValues
from .observables import (
    rmatrix_from_spectrum,
    greens_from_spectrum,
    wavefunction_internal_from_spectrum,
)
from .matching import (
    smatrix_from_R,
    open_channel_smatrix_from_R,
    coupled_channel_parameters_from_S,
    phases_from_S,
)

__all__ = [
    "Spectrum",
    "BoundaryValues",
    "rmatrix_from_spectrum",
    "greens_from_spectrum",
    "wavefunction_internal_from_spectrum",
    "smatrix_from_R",
    "open_channel_smatrix_from_R",
    "coupled_channel_parameters_from_S",
    "phases_from_S",
]
```

This submodule is exercised by synthetic-Hamiltonian tests — random Hermitian or
complex-symmetric matrices with known spectral structure, independent of any mesh logic.
The tests live alongside the rest of the suite in `tests/unit/test_spectral*.py` (there is
**no** `spectral/tests/` subdirectory). Property tests verify the spectral identities
exactly: $\sum_k \gamma_{kc} \gamma_{kc'} / (\varepsilon_k - E) = (Q^T (H - E I)^{-1} Q)_{cc'}$ to round-off.

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
    """Build a JIT'd spectrum(interaction) -> Spectrum kernel.

    method:
        "eigh"  — Hermitian eigendecomposition (real V). GPU-ready.
        "eig"   — general eigendecomposition (complex V). Uses host
                  callback on GPU; native on CPU.

    keep_eigenvectors: bake into the kernel whether to return U.
        True if Green's functions or wavefunctions were requested.

    The assembler produces the symmetric MeV block (§11.5). The spectrum path
    is single-μ in scope, so we divide by the uniform mass factor m0 to recover
    the fm⁻² Bloch-augmented Hamiltonian — eigenvalues stay in fm⁻² and the
    fm⁻² spectral observables (§10.2) are unchanged. (Per-channel μ here is a
    generalized eigenproblem, deferred — §15.3.) Q is the plain surface
    projector; the per-channel √m_c reduced-width factor lives only on the
    R-matrix-direct path (§11.3).
    """
    Q = build_Q(mesh, channels)            # plain (M, N_c)
    if method not in ("eigh", "eig"):
        raise ValueError(f"spectrum kernel supports 'eigh'/'eig', not {method!r} "
                         "(use the R-matrix-direct path for 'linear_solve')")
    m0 = _channel_masses(channels)[0]      # spectrum path: single μ (all channels equal)
    is_hermitian = (method == "eigh")
    _eig = jnp.linalg.eigh if is_hermitian else _eig_via_callback  # eig: CPU/host-callback

    def _one(block):                       # block: (M, M) MeV
        H = assemble_block_hamiltonian(mesh, operators, channels, block) / m0   # → fm⁻²
        eigvals, U = _eig(H)
        if not is_hermitian:               # complex-symmetric: bilinear (U^T U = I) normalization
            U = U / jnp.sqrt(jnp.diag(U.T @ U))[None, :]
        γ = (U.conj().T if is_hermitian else U.T) @ Q          # (M, N_c)
        return Spectrum(
            eigenvalues=eigvals,
            surface_amplitudes=γ,
            eigenvectors=(U if keep_eigenvectors else None),
            is_hermitian=is_hermitian,
        )

    @jax.jit
    def spectrum(interaction):
        # energy_dependent is static → branch at trace time.
        if interaction.energy_dependent:    # (N_E, M, M): one Spectrum per energy
            return jax.vmap(_one)(interaction.block)
        return _one(interaction.block)      # (M, M): single Spectrum
    return spectrum


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
    [2, eq. 9]). Full decoupling is handled by `open_channel_smatrix_from_R`.
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

**`wavefunction_grid` (v1.6).** When `'wavefunction'` is requested with an energy grid, the
solver additionally binds `wavefunction_grid(spectrum, channel_index=0)`: internal
wavefunctions for **every** `(block, grid-energy)` pair against the compile-time-baked
source stack (`Solver.wavefunction_sources`, Descouvemont eq. 27 — built from
`mesh.basis_at_boundary` and the `H⁻` boundary cache, with **no** `is_open` masking;
sub-threshold entries are defined but are not scattering wavefunctions).  One entry point
serves both evaluation regimes, detected from the `Spectrum` rank
(`spectrum_is_energy_batched`): a static-V spectrum runs the fused diagonalize-once einsum
``ψ_e = U · diag(1/(ε_k − E_e/μ)) · (Uᵀ_metric · source_e)`` over all energies without
materializing Green's matrices (§10.2), while an energy-batched spectrum vmaps the
per-energy Green's contraction over the aligned `(spec_e, E_e, source_e, μ_e)` axes — the
`smatrix_grid` pattern.  `channel_index=None` returns all incoming channels on an extra
`N_c` axis.  **Regime rule:** the energy-batched path is selected by
`V.energy_dependent or mass_factor_nonuniform` — a non-uniform μ(E) makes diagonalize-once
invalid physics even for a static V (Appendix C.10).

### 11.3 The direct (linear-solve) path: `rmatrix_direct` and `wavefunction_direct`

The direct path solves the Bloch–Schrödinger system per energy without an eigendecomposition. It is the GPU-/vmap-native route for complex V (where `eig` is a CPU callback) and the only path for non-local kernels that avoids forming a dense `M×M` eigenvector matrix. It consumes an `Interaction` (§15.1) and, as of v1.3, also produces internal wavefunctions.

**`wavefunction_direct_grid` (v1.6).** Whenever the direct path is active the solver also
binds `wavefunction_direct_grid(V, channel_index=0)`: one LU factorization per
`(block, energy)` solved against every incoming-channel column of the baked source stack —
the same shapes and closed-channel contract as the spectral `wavefunction_grid`.  It
inherits the direct path's per-channel μ(E) support and, unlike the `eig` spectral path, is
**fully differentiable** (Appendix C.11) — the route for gradient/UQ work with complex
optical potentials.  Propagated multi-interval meshes are rejected
(`NotImplementedError`): the boundary basis behind the source stack differs per interval.

```python
# solvers/linear_solve.py
import jax
import jax.numpy as jnp

from .assembly import assemble_block_hamiltonian, build_Q   # symmetric MeV assembler; build_Q is plain (√m_c folded per-energy below)


def make_direct_kernels(mesh, operators, channels, energies, boundary, mass_factor_grid=None):
    """Build rmatrix_direct / smatrix_direct / phases_direct / wavefunction_direct.

    Per energy E_i solves the complex-symmetric system  C(E_i) X = RHS  with
        C(E_i) = H_MeV(interaction) − E_i·I        (MeV; see §11.5)
    where H_MeV bakes each channel's mass into its kinetic block and leaves the
    coupling potential untouched. R = Q'ᵀ C⁻¹ Q' / a with Q' = diag(√m_c)·Q
    (the per-channel reduced-width factor). One LU factorization of C(E_i) serves
    both the R-matrix (RHS = Q') and the wavefunction (RHS = source).
    """
    Q  = build_Q(mesh, channels)                 # plain surface projector (M, N_c)
    a  = mesh.scale
    M  = mesh.n * len(channels)
    N  = mesh.n
    # Per-(energy, channel) mass factors m_c(E_i), shape (N_E, N_c): broadcasts a
    # scalar, a (N_c,) per-channel value, a (N_E,) per-energy value, or (N_E, N_c).
    mu_grid = _channel_mass_grid(channels, mass_factor_grid, energies)   # (N_E, N_c)

    def _C(block_i, E_i, mu_row):                 # mu_row: (N_c,)
        H = assemble_block_hamiltonian(mesh, operators, channels,
                                       block_i, mass_factor_grid=mu_row)   # MeV
        return H - E_i * jnp.eye(M, dtype=H.dtype)

    def _Qp(mu_row):                              # Q' = diag(√m_c)·Q, per energy
        return jnp.repeat(jnp.sqrt(mu_row), N)[:, None] * Q   # (M, N_c)

    @jax.jit
    def rmatrix_direct(interaction):
        blocks = interaction.block if interaction.energy_dependent \
                 else jnp.broadcast_to(interaction.block, (energies.shape[0], M, M))
        def _one(blk, E_i, mu_row):
            lu = jax.scipy.linalg.lu_factor(_C(blk, E_i, mu_row))
            Qp = _Qp(mu_row)
            return (Qp.T @ jax.scipy.linalg.lu_solve(lu, Qp)) / a          # (N_c, N_c)
        return jax.vmap(_one)(blocks, energies, mu_grid)                   # (N_E, N_c, N_c)

    @jax.jit
    def wavefunction_direct(interaction, source, energy_index):
        i = energy_index
        blk = interaction.block[i] if interaction.energy_dependent else interaction.block
        mu_row = mu_grid[i]                                          # (N_c,)
        f = jnp.linalg.solve(_C(blk, energies[i], mu_row), source)  # MeV⁻¹·source
        # Per-channel μ scaling: C(E_i) is in MeV, so C⁻¹·source = G_MeV·source,
        # whereas the spectral wavefunction (§10.2) uses the fm⁻² Green's function.
        # Multiplying channel block c by μ_c converts between them:
        #   scale[c·N + j] = μ_c,   ψ = scale · f.
        # For a uniform μ this reduces to the old m0·solve(...); per channel it
        # preserves the fm⁻² spectral-Green's equivalence block-by-block.
        scale = jnp.repeat(mu_row, N)                               # (M,)
        return scale * f                                            # (M,) ≡ spectral wavefunction

    # smatrix_direct / phases_direct: rmatrix_direct → spectral matching, vmapped over the grid.
    return rmatrix_direct, smatrix_direct, phases_direct, wavefunction_direct
```

The mass helpers are shared with the assembler: `_channel_masses(channels, override=None) → (N_c,)` returns the per-channel `m_c` (an `override` row wins), and `_channel_mass_grid(channels, mass_factor_grid, energies) → (N_E, N_c)` broadcasts a scalar / `(N_c,)` / `(N_E,)` / `(N_E, N_c)` `mass_factor_grid` to the full grid (§15.3).

Notes:

- **One entry point per observable.** `Interaction.energy_dependent` selects broadcast vs. per-energy indexing internally; there is no separate `*_direct_grid` surface.
- **Per-channel / energy-dependent μ.** `mass_factor_grid` (shape `(N_E, N_c)`, broadcasting from `(N_E,)` or scalar — §15.3) overrides `channels[c].mass_factor`. Because the assembly is symmetric (mass on the kinetic block, not dividing V), per-channel μ needs no metric and no generalized solve here. When μ is energy-dependent, both `C(E_i)` and the `√m_c` reduced-width factor in `Q'` use `m_c(E_i)` (so `Qp` is formed per energy inside `_one`).
- **Shared factorization.** When both R and ψ are wanted at the same energy, factor `C(E_i)` once and reuse for the `Q'` and `source` solves.
- **`source`** is `make_wavefunction_source(solver, channel_index, energy_index)` (§10.2); the returned `f` matches the spectral `wavefunction` convention.

For a single off-grid energy, use the spectral submodule directly (`lax.spectral.rmatrix_from_spectrum` + `smatrix_from_R`); the direct path evaluates only on the compile-time energy grid.

> **Symmetry-block axis (§15.5).** The body above — `_C`, `_Qp`, and the per-energy `_one` — depends on the block's channels only through their centrifugal `ℓ_c(ℓ_c+1)·inv_r2`, thresholds, masses, and the cached `boundary`. As shipped, these are factored into array-parameterized per-block cores (`_rmatrix_direct_core`, `_rmatrix_direct_grid_core`, `_wavefunction_direct_core` in `solvers/linear_solve.py`, taking traced `(N_c,)` centrifugal/threshold/μ rows from `solvers.assembly.channel_arrays` rather than reading `ChannelSpec.l`); the symmetry-block layer is a thin `jax.vmap` of each core over a leading `(N_b,)` axis (stacked centrifugal/threshold/μ rows from `block_group_arrays` + per-block `boundary`), composing with the per-energy vmap already present. The spectrum kernel (§11.1) factors the same way (`_spectrum_{eigh,eig}_core` vmapped in `_spectrum_blocks[_grid]`). No new linear algebra is introduced.

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
| `linear_solve` | Complex V on GPU, or R/S/phases + wavefunctions without a `Spectrum` | no (R-matrix-direct path) | ✅     | ✅         |
| `lanczos`    | (future) very large coupled, complex V     | partial           | ✅     | ✅         |

### 11.5 Assembling the block Hamiltonian

#### Unit convention — read this first

The assembler builds the block Hamiltonian in the **symmetric MeV form** (v1.3). Each diagonal block's kinetic and centrifugal operators are scaled by that channel's mass factor `m_c = ℏ²/2μ_c` (MeV·fm²), the threshold `E_c` is added in MeV, and the coupling potential `V` is added **as supplied, in MeV, with no per-channel division**:

- `TpL` and kinetic matrices are stored mass-free in fm⁻² (mesh builders divide by `scale²`); the assembler multiplies each diagonal block by `m_c` to put it in MeV.
- `V` (the `Interaction` block) is in MeV and enters untouched → the assembled block is symmetric (this is why off-diagonal coupling must *not* be divided by a per-row mass).
- The **assembled block is in MeV**. The **R-matrix-direct path** uses it directly: `C(E) = H_MeV − E·I` (MeV), with `Q' = diag(√m_c)·Q` carrying the per-channel reduced-width factor. The **spectrum path** divides the block by the uniform mass factor `m` to recover the fm⁻² Bloch-augmented Hamiltonian, so `Spectrum.eigenvalues` stay in **fm⁻²** and the fm⁻² spectral observables (§10.2) are unchanged.

**Relation across paths.** The two are exactly equivalent for a single mass factor: dividing `H_MeV` by `m` gives the fm⁻² eigenproblem (`denom = 1/(ε − E/m)`, plain `Q`), while the direct path keeps MeV (the mass absorbed into `C` and the `√m`-folded `Q'`). Observables (R, S, phases) are invariant under the rescaling. **Per-channel μ** is supported on the **direct path** via this symmetric form; on the spectral path it is a generalized eigenproblem (`H_MeV u = E·diag(1/μ_c)·u`) deferred with the rest of that path (§15.3).

Standard nuclear value: ℏ²/2mₙ = 20.736 MeV·fm² [2, eq. 46].

```python
# solvers/assembly.py
import jax.numpy as jnp


def assemble_block_hamiltonian(mesh, operators, channels, interaction, mass_factor_grid=None):
    """Build the (N_c·N, N_c·N) Bloch-augmented block Hamiltonian in MeV.

    Symmetric MeV form (v1.3): each diagonal block's kinetic + centrifugal
    operators are scaled by the channel mass factor m_c; the threshold E_c is
    added in MeV; the coupling potential is added in MeV with NO per-channel
    division, so the block is symmetric for symmetric V.

    Diagonal block c == c':
        m_c·(TpL + ℓ_c(ℓ_c+1)·inv_r2) + E_c·I + V_cc
    Off-diagonal c ≠ c':
        V_cc'

    `interaction` is the potential contribution and may be:
        (N_c, N_c, N)      — local  (diagonal in r),     MeV
        (N_c, N_c, N, N)   — non-local kernel,           MeV
        (M, M)             — pre-assembled Interaction.block (MeV); kinetic/
                             centrifugal/threshold are still added by this fn.
    `mass_factor_grid`, when given, supplies per-channel (and, upstream, per-
    energy) m_c overriding channels[c].mass_factor.
    """
    n_c, N = len(channels), mesh.n
    TpL = operators.TpL
    inv_r2 = operators.inv_r2 if operators.inv_r2 is not None else \
             jnp.diag(1.0 / mesh.radii ** 2)
    m = _channel_masses(channels, mass_factor_grid)      # (N_c,)

    # Kinetic + centrifugal + threshold (block-diagonal, MeV) — interaction-free.
    kin_blocks = []
    for c in range(n_c):
        diag = m[c] * (TpL + channels[c].l * (channels[c].l + 1) * inv_r2) \
               + channels[c].threshold * jnp.eye(N)
        kin_blocks.append(diag)
    H_kin = _block_diag(kin_blocks)                      # (M, M) MeV

    V = _interaction_to_block(interaction, n_c, N)       # (M, M) MeV (local→diag, nl→full)
    return H_kin + V                                     # symmetric MeV


def build_Q(mesh, channels):
    """Q[c·N + j, c'] = δ_{cc'} φ_j(a). Returns (N_c·N, N_c) — the plain surface
    projector. The spectrum path (fm⁻²) uses it directly; the R-matrix-direct
    path (MeV) wraps it with the per-channel reduced-width factor as
    Q' = diag(√m_c)·Q (§11.3).
    """
    N, n_c = mesh.n, len(channels)
    b = mesh.basis_at_boundary
    Q = jnp.zeros((n_c * N, n_c), dtype=b.dtype)
    for c in range(n_c):
        Q = Q.at[c * N:(c + 1) * N, c].set(b)
    return Q
```

---

## 12. Off-grid energies: interpolation is out of scope

> **Removed at the v1.5 close-out.** Earlier revisions shipped Padé interpolation
> (`spectral.pade_interpolate` plus solver-bound `interpolate_{rmatrix,smatrix,phases}`
> builders, v1.1–v1.5). The feature was removed deliberately rather than refined.

Boundary-value-dependent observables (S, phases) are produced on the compile-time energy
grid (§3); for energy-dependent V(E) the aligned-grid helpers (§11) sample every
observable at its own grid energy. Getting values *between* grid points is an
interpolation problem, and the right interpolant depends on the observable:

- **R-matrix** — meromorphic in E at fixed V (`R(E) = Σ_k γ_k²/(ε_k − E)`), so rational
  (Padé-like) fits are principled. But on the spectrum path R is already **exact** at any
  runtime energy from one `Spectrum`, so there is nothing to interpolate; and for
  energy-dependent V the sampled `R(E_i; V(E_i))` is no longer rational in E.
- **S-matrix** — smooth in |S| away from thresholds, but its energy dependence includes
  Coulomb functions (not rational), and channel openings introduce branch points.
- **Phase shifts** — defined only modulo π. Principal-branch samples acquire artificial
  ±180° steps wherever the physical curve crosses the branch cut, and a global rational
  approximant responds by parking a real pole at the step (a Froissart doublet) and
  ringing across the whole interval.

A single library-blessed interpolator therefore either silently misbehaves on the most
commonly requested observables or grows per-observable special cases. The library instead
guarantees cheap resampling: recompiling on a new energy grid costs milliseconds of
`mpmath` boundary evaluation per (E, c) pair, and users who want true off-grid evaluation
can interpolate the sampled arrays themselves with the appropriate tool (spline of |S|,
unwrap-then-spline for δ, rational/K-matrix fits for resonance work).

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

`solver.from_grid_vector(ψ_or_fn)` is the inverse: it projects fine-grid values (or a
callable sampled on the grid) back onto mesh coefficients, the entry point used by
`interaction_from_funcs` and grid→mesh form-factor construction.

The same `B_grid` matrix is reused by:

- `solver.to_grid_vector(c)` for a wave-function vector and `solver.to_grid_matrix(V)` for a kernel.
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

For coupled channels, the user passes the channel index to select the right $\ell$:

```python
@jax.jit
def fourier(c, channel_index=0):
    F = transform_matrices.F_momentum[channel_index]    # (M_k, N) for this ℓ
    return F @ c                                        # (M_k,)
```

For a non-local kernel, `solver.double_fourier_transform(V, …)` performs the double Bessel
transform $\tilde V(p, p') = F\, V\, F^T$ (the analogue of `to_grid_matrix` in momentum
space).

### 13.3 Integration

Norms and expectation values of operators in the mesh basis are simple sums [1, §2.9, eq. 2.82]:

`make_integration` returns a **single** `integrate(c, operator=None)` callable rather than a
dict of separate functions: with no operator it is the norm ⟨ψ|ψ⟩; with a precomputed
operator matrix it is the expectation value ⟨ψ|O|ψ⟩.

```python
# transforms/integration.py
import jax
import jax.numpy as jnp


def make_integration(mesh):
    @jax.jit
    def integrate(c, operator=None):
        """Norm or expectation value in the mesh basis.

        operator is None  → ⟨ψ|ψ⟩ ≈ Σ c_j²            [Baye eq. 2.82]
        operator is (N,N) → ⟨ψ|O|ψ⟩ = c† O c          (any precomputed operator)
        """
        if operator is None:
            return jnp.sum(c ** 2)
        return c.conj() @ operator @ c

    return integrate
```

(For a local `V(r)`, build the diagonal operator `jnp.diag(V_at_mesh)` and pass it as
`operator`, recovering the Gauss-sum form ⟨ψ|V(r)|ψ⟩ ≈ Σ c_j² V(r_j).)

For a wave function in a regularized basis, the orthonormality of the *Gauss approximation* [1, eq. 2.12] guarantees that $\langle\psi|\psi\rangle \approx \sum_j c_j^2$ to the same accuracy as the LMM itself, even when the regularized basis is not exactly orthonormal in the full $L^2$ sense [1, §2.7, eq. 2.69].

### 13.4 Two-state bilinear matrix elements (v1.6)

`integrate` is deliberately single-state and always-conjugating.  The batched two-state
form lives in `transforms/bilinear.py` and is bound **unconditionally** on every solver
(it depends only on compile-time shapes):

```python
solver.matrix_element(bra, ket, operator=None, *, conjugate)  # keyword-required
#   conjugate=False :  braᵀ · O · ket     (the DWBA bilinear; matches the
#                                          complex-symmetric U diag Uᵀ metric)
#   conjugate=True  :  bra† · O · ket     (the Hermitian inner product)
```

Operator forms and the **scaling contract** (the silent-wrongness trap): ``None`` is the
overlap; a ``(M,)`` array is a *local* operator as **unscaled** node values ``V(rᵢ)``
(the coefficients absorb ``√(λᵢ a)``, so the element is the plain node sum); a ``(M, M)``
array is a caller-**Gauss-scaled** kernel ``K̃ᵢⱼ = √(λᵢλⱼ)·a·K(rᵢ,rⱼ)``; an
:class:`Interaction` is the recommended form — its assembled block already carries exactly
this scaling for both local and non-local content, and its static flags drive the
block/energy axis alignment (via `lift_block`).  `matrix_element` adds no factors of
``a``, ``k``, or quadrature weights beyond what is inside the operator.

Batching: `bra`/`ket` accept ``(M,)``, rank-2, or ``(N_b, N_E, M)`` in the canonical
block × energy order.  Rank-2 follows a **deterministic mode-based rule** (never
shape-sniffing): blocks mode → block-leading ``(N_b, M)``; channels mode → energy-leading
``(N_E, M)``; mismatches raise naming the expected axis.  Internally states are lifted to
``(N_b|1, N_E|1, M)`` and the three jitted einsum kernels
(`'...m,...m->...'` / `'...m,...m,...m->...'` / `'...m,...mn,...n->...'`) broadcast; the
output squeezes axes no input contributed (scalar for unbatched inputs).  A standalone
`lax.transforms.matrix_element` does the pure broadcast einsum with no solver-aware
interpretation or validation.

> **v1.6 note on §13.1–13.2 batching.** The `grid=` projection kernels are trailing-axis
> einsums (``'rn,...n->...r'`` and kin), so arbitrary leading batch axes (block, energy)
> pass through in both compile modes.  In blocks mode `F_momentum` is
> ``(N_b, N_c, M_k, N)`` and `fourier`/`double_fourier_transform` accept ``(N_b, N)`` /
> ``(N_b, N, N)`` inputs, broadcasting unbatched ones across blocks; rank-2 inputs with
> leading dimension ``N_b`` are **always** block-batched vectors (in the ``N_b == N``
> corner a broadcast kernel must be written ``jnp.broadcast_to(K, (N_b, N, N))``).

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
from .solvers.linear_solve import make_direct_kernels   # R-matrix-direct path (§11.3)
from .operators.interaction import (
    make_interaction_from_block, make_interaction_from_array, make_interaction_from_funcs,
)


Method = Literal["eigh", "eig", "linear_solve"]


def compile(
    *,
    mesh: MeshSpec,
    channels: Iterable[ChannelSpec] | None = None,
    blocks: Iterable[Iterable[ChannelSpec]] | None = None,   # §15.5; mutually exclusive with channels
    operators: Iterable[str] = ("T+L",),
    solvers: Iterable[str] = ("spectrum", "rmatrix", "smatrix", "phases"),
    energies: jnp.ndarray | None = None,
    energy_dependent: bool = False,
    method: Method | None = None,
    V_is_complex: bool = False,
    grid: jnp.ndarray | None = None,
    momenta: jnp.ndarray | None = None,
    z1z2: tuple[int, int] | None = None,
    mass_factor_grid: jnp.ndarray | None = None,
    dtype: jnp.dtype = jnp.float64,
    device: jax.Device | str | None = None,
    dps: int = 40,
) -> Solver:
    """Build a Solver specialized to the given mesh, channels, and energy grid.

    Parameters
    ----------
    mesh : MeshSpec
        Mesh family, regularization, size, and scale.
    channels : iterable of ChannelSpec, optional
        Channel structure (l, threshold, mass_factor) for a single coupled-channel
        block. Mutually exclusive with `blocks`; exactly one must be given.
    blocks : iterable of iterable of ChannelSpec, optional
        A batch of same-shaped symmetry blocks (independent (J, π) groups, partial
        waves, …); see §15.5. Each inner group must have the same length N_c. The
        compiled solver carries a leading (N_b,) axis on `channels`/`boundary`, and
        every observable gains a corresponding leading block axis. Partial-wave
        batching is the N_c == 1 case: blocks=[[ChannelSpec(l=0,…)], …]. Mutually
        exclusive with `channels`.
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
        "rmatrix_direct" — per-energy R-matrix-direct path (linear solve; no Spectrum)
    energies : jnp.ndarray, optional
        Energy grid (MeV). Required if any of {smatrix, phases, rmatrix_direct}
        is requested, or if `energy_dependent=True`. Used to precompute
        boundary values via mpmath.
    energy_dependent : bool
        If True, indicates that V will be supplied per-energy at runtime
        (one V per compile-time energy point). The user is expected to
        call `jax.vmap(solver.spectrum)` over the energy axis themselves;
        `solver.smatrix` and friends consume the resulting batched Spectrum.
    method : "eigh" | "eig" | "linear_solve" | None
        Linear-algebra backend. None invokes the default policy (see §11.4).
    V_is_complex : bool
        Whether the user will supply complex potentials. Drives default
        method selection if `method=None`.
    grid : jnp.ndarray, optional
        Fine radial grid (fm) for `to_grid_vector`/`to_grid_matrix`/`from_grid_vector`.
    momenta : jnp.ndarray, optional
        Momentum grid (fm⁻¹) for `fourier`.
    z1z2 : tuple of int, optional
        (Z₁, Z₂) for Coulomb scattering. Default neutral (η=0).
    mass_factor_grid : jnp.ndarray, optional
        Per-energy (and optionally per-channel) reduced-mass factor
        m_c = ℏ²/2μ_c in MeV·fm². Shape (N_E,) broadcasts across channels;
        (N_E, N_c) is fully general; a scalar / omission falls back to
        `ChannelSpec.mass_factor`. Feeds both the symmetric assembly (§11.5)
        and the per-energy boundary values. Supports semi-relativistic μ(E)
        and per-channel masses on the direct path (§15.3).
    dtype : jnp.dtype
        Floating-point precision for the baked arrays. Default float64. (x64 itself
        is enabled globally via `jax.config.update("jax_enable_x64", True)`; `dtype`
        selects the precision of the cached mesh/operator/boundary arrays.)
    device : str or jax.Device, optional
        Where to place the compiled solver's arrays.
    dps : int
        mpmath decimal precision for Coulomb / Whittaker evaluation.

    Returns
    -------
    Solver
        Bundle with cached data and JIT'd callables.
    """
    # --- Resolve the (possibly batched) block structure (§15.5) ---
    if (channels is None) == (blocks is None):
        raise ValueError("pass exactly one of `channels` or `blocks`")
    if channels is not None:
        block_groups = (tuple(channels),)               # single block, N_b = 1
    else:
        block_groups = tuple(tuple(b) for b in blocks)   # N_b stacked blocks
        n_c = len(block_groups[0])
        if any(len(b) != n_c for b in block_groups):
            raise ValueError("all `blocks` must share the same channel shape N_c")
    channels = block_groups[0]   # template block; per-block centrifugal/boundary are
                                 # stacked on a leading (N_b,) axis below and vmapped
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

    # --- R-matrix-direct path (linear solve): consumes an Interaction ---
    direct_requested = (method == "linear_solve") or ("rmatrix_direct" in solvers_set)
    if direct_requested:
        rmatrix_direct_fn, smatrix_direct_fn, phases_direct_fn, wavefunction_direct_fn = \
            make_direct_kernels(mesh_data, operator_matrices, channels,
                                energies_arr, boundary, mass_factor_grid)
    else:
        rmatrix_direct_fn = smatrix_direct_fn = phases_direct_fn = wavefunction_direct_fn = None

    # "wavefunction" binds the direct-path solver under linear_solve, else the spectrum-path one.
    if "wavefunction" in solvers_set:
        if method == "linear_solve":
            wavefunction_fn, wf_direct_fn = None, wavefunction_direct_fn
        else:
            wavefunction_fn = make_wavefunction_internal(mesh_data, channels)
            wf_direct_fn = wavefunction_direct_fn
    else:
        wavefunction_fn, wf_direct_fn = None, wavefunction_direct_fn

    # --- Interaction builders (close over mesh, channels, energies) ---
    iface_block = make_interaction_from_block(mesh_data, channels, energies_arr)
    iface_array = make_interaction_from_array(mesh_data, channels, energies_arr)
    iface_funcs = make_interaction_from_funcs(mesh_data, channels, energies_arr)
    local_pot, nonlocal_pot = make_potential_builders(mesh_data, channels, energies_arr)

    # --- Transforms ---
    to_grid_vec, to_grid_mat = make_to_grid(transforms.B_grid) if transforms.B_grid is not None else (None, None)
    from_grid_vec = make_from_grid(transforms.B_grid) if transforms.B_grid is not None else None
    integ = make_integration(mesh_data)

    # --- Assemble Solver ---
    solver = Solver(
        mesh=mesh_data,
        operators=operator_matrices,
        channels=channels,
        energies=energies_arr,
        mass_factor_grid=mass_factor_grid,
        boundary=boundary,
        transforms=transforms,
        method=method,
        spectrum=spectrum_fn,
        rmatrix=rmatrix_fn,
        smatrix=smatrix_fn,
        phases=phases_fn,
        greens=greens_fn,
        wavefunction=wavefunction_fn,
        eigh=eigh_fn,
        rmatrix_grid=rmatrix_grid_fn, smatrix_grid=smatrix_grid_fn, phases_grid=phases_grid_fn,
        rmatrix_direct=rmatrix_direct_fn,
        smatrix_direct=smatrix_direct_fn,
        phases_direct=phases_direct_fn,
        wavefunction_direct=wf_direct_fn,
        interaction_from_block=iface_block,
        interaction_from_array=iface_array,
        interaction_from_funcs=iface_funcs,
        local_potential=local_pot,
        nonlocal_potential=nonlocal_pot,
        to_grid_vector=to_grid_vec,
        to_grid_matrix=to_grid_mat,
        from_grid_vector=from_grid_vec,
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

Four structural changes from the previous design:

1. **`spectrum` is the central kernel.** Everything else (`rmatrix`, `smatrix`, `phases`, `greens`, `wavefunction_internal`) is a thin closure over it. The factory builds the spectrum kernel first and then attaches lightweight observables.

2. **`method` is a new parameter** that controls the linear-algebra backend. The default policy is real → `eigh`, complex on CPU → `eig`, complex on GPU → `linear_solve`. The user can always override.

3. **The R-matrix-direct path is a separate namespace.** When the user explicitly requests it (typically complex V on GPU with `method="linear_solve"`), it is built as a per-energy direct kernel consuming an `Interaction`. It produces R, S, phases, and internal wavefunctions (`wavefunction_direct`), but does *not* produce a `Spectrum`, so `greens` and the spectrum-path `wavefunction` are unavailable on it.

4. **All solve methods consume an `Interaction`** (v1.3). The factory bakes every kinematic quantity — energies, charges, thresholds, and per-channel/energy reduced mass — so the runtime is a pure map from an `Interaction` to outputs. Raw potential arrays are not accepted.

### 14.3 What the factory does *not* do

- It does not accept potentials. The solver is potential-agnostic.
- It does not perform per-call setup. Everything not depending on `V` is done here, once.
- It does not implicitly broadcast over potential parameters. The user uses `jax.vmap` over their parametric `V` builder.
- It does not change the meaning of the existing energy-independent observable methods. `solver.smatrix(spec)` and `solver.phases(spec)` always evaluate one fixed `Spectrum` against the full compile-time boundary grid; they must not be repurposed for energy-dependent `V(E)`. The aligned-grid helpers `rmatrix_grid`/`smatrix_grid`/`phases_grid` cover the energy-dependent case separately rather than overloading these methods.

---

## 15. Coupled-channel structure

A coupled-channel calculation has $N_c$ channels (each characterized by `(l, threshold, mass_factor)` per [2, §2.1, eq. 2]). The Hamiltonian is block-structured:

$$\mathcal{H} = \begin{pmatrix} T_{1} + V_{11} & V_{12} & \cdots \\ V_{21} & T_{2} + V_{22} & \cdots \\ \vdots & & \ddots \end{pmatrix}$$

where each block is $(N, N)$ and the full matrix is $(N_c N, N_c N)$. The diagonal blocks include the kinetic operator augmented with the centrifugal term and the channel's threshold; off-diagonal blocks are channel-coupling potentials.

### 15.1 Input format for V: the `Interaction` interface

The canonical solver input is an assembled `Interaction` (§6) whose `block` is the coupled-channel potential in the Lagrange-mesh basis, shape `(M, M)` or `(N_E, M, M)` with `M = N_c·N`, in **MeV** — local terms on the per-channel diagonal, non-local terms as full Gauss-scaled blocks, all summed. This single object is what every solver consumes; the former raw `(N_c, N_c, N[,N])` array inputs are no longer accepted.

**Single-kind function builders.** The common case — a potential from a callable — uses two explicit entry points (no arity inference, no `kind=` argument; the caller chooses):

```python
solver.local_potential(fn,    *, coupling=None, energy_dependent=False)   # fn(r)      or fn(r, E)
solver.nonlocal_potential(fn, *, coupling=None, energy_dependent=False)   # fn(r, r')  or fn(r, r', E)
```

Each builds one single-kind `Interaction`. `coupling` defaults to `[[1.0]]` when `N_c == 1`
and **raises** for `N_c > 1` (no silent `eye(N_c)`). Mixed / multi-term interactions compose
by summing single-kind builds with `+` (`Interaction.__add__`, §6) — e.g.
`solver.local_potential(central) + solver.nonlocal_potential(exchange)`.

**List builders.** Three solver-bound builders assemble multi-term interactions directly (they close over `mesh`, `channels`, `energies`):

```python
# Raw escape hatch — for potentials that do not factor (e.g. microscopic RGM kernels)
solver.interaction_from_block(block, *, energy_dependent=False)

# From node arrays — the array path (e.g. a downstream that samples its own potentials)
solver.interaction_from_array(
    local     = [(g_i,  A), ...],   # g_i : (N,) or (N_E, N)        diagonal-in-r term
    nonlocal_ = [(g_ij, A), ...],   # g_ij: (N,N) or (N_E,N,N)      kernel term
)

# From callables — a thin wrapper around interaction_from_array
solver.interaction_from_funcs(
    local     = [(g,  A), ...],     # g(r)      or g(r, E)
    nonlocal_ = [(g,  A), ...],     # g(r, r')  or g(r, r', E)
    energy_dependent = False,
)
```

Note `nonlocal_` (trailing underscore): `nonlocal` is a Python keyword and cannot be a
parameter name.

A potential is a **sum of terms**. Each term is *either* a bare form factor `g` *or* a
`(g, A)` tuple of a form factor and a channel-coupling matrix `A` (shape `(N_c, N_c)`),
disambiguated by `isinstance(term, tuple)` (form-factor arrays are not tuples):

- `interaction_from_array(nonlocal_=[g])`      → coupling defaults to `[[1.0]]`  (single-channel sugar)
- `interaction_from_array(nonlocal_=[(g, A)])` → explicit coupling

Omitting the coupling is single-channel sugar: when `N_c > 1` a bare term **raises**,
requiring an explicit `(N_c, N_c)` matrix — matching `solver.local_potential`/`nonlocal_potential`.
The default is factored into a shared `_default_coupling(coupling, n_c)` helper reused by all
three function builders. Because tuples are reserved for the explicit `(g, A)` pair, a *bare*
block-dependent funcs term (§15.5) must be a non-tuple sequence of per-block callables (e.g. a
list). With explicit couplings, `V_cc'(r) = Σ_t A^t_cc' g_t(r)` (local),
`W_cc'(r,r') = Σ_t A^t_cc' g_t(r,r')` (non-local) — the natural *form-factor ⊗ coupling-matrix*
physics structure (rotor, Reid, etc.; §16/§models). Assembly:

```
local    block[c·N+n, c'·N+n] += A^t_cc' · g_t(r_n)                          # diagonal in n
nonlocal block[c·N+i, c'·N+j] += A^t_cc' · √(λ_i λ_j) · a · g_t(r_i, r_j)     # Desc. eq. 26
```

The `local=`/`nonlocal_=` split (rather than one mixed list) removes the rank ambiguity between a non-local energy-independent `(N,N)` term and a local energy-dependent `(N_E,N)` term; `energy_dependent` is an explicit flag, and any leading axis is validated against `len(energies)`. Builders **validate symmetry** of each `A` and of the assembled block (they do not silently symmetrize). `interaction_from_block` takes a pre-assembled block and so has no coupling argument. The Gauss scaling is folded into `interaction_from_array` directly; there are no public `assemble_local`/`assemble_nonlocal` primitives (§8).

### 15.2 Surface amplitudes carry channel structure

The `Spectrum.surface_amplitudes` array has shape `(M, N_c)` where $M = N_c N$. Element `γ[k, c]` is the surface amplitude of eigenmode $k$ in channel $c$:

$$\gamma_{kc} = \sum_n \varphi_n(a)\, u^{(k)}_{(c, n)}$$

where $u^{(k)}$ is the $k$-th eigenvector of the block Hamiltonian and the second index runs over $(c, n)$ with $n$ the mesh point and $c$ the channel. The R-matrix and S-matrix spectral sums then naturally produce $N_c \times N_c$ matrices.

### 15.3 Mass factor per channel

Channels may have different reduced masses (e.g. nucleon-nucleus vs nucleus-nucleus). `ChannelSpec.mass_factor` is `m_c = ℏ²/2μ_c` in MeV·fm². In the **symmetric MeV form** (§11.5) the diagonal block is

$$m_c\,(\hat T_c + L(B_c)) + \frac{\ell_c(\ell_c+1)\, m_c}{r^2} + E_c + V_{cc}(r),$$

with the off-diagonal coupling $V_{cc'}$ added as-is. Per-channel mass thus enters in three channel-diagonal places only — the kinetic-block scaling $m_c$, the reduced-width factor $\sqrt{m_c}$ folded into the surface projector ($Q' = \mathrm{diag}(\sqrt{m_c})\,Q$, so $R = Q'^{T} C^{-1} Q'/a$), and the per-channel boundary $k_c, \eta_c, B_c$ ($k_c^2 = 2\mu_c(E-E_c)/\hbar^2$) — and the coupling potential is never divided, so the block stays symmetric.

**Energy-dependent μ.** For semi-relativistic kinematics μ depends on energy; supply `mass_factor_grid` of shape `(N_E, N_c)`, broadcasting from `(N_E,)` (per-energy, uniform over channels) or a scalar (the common uniform case). It overrides `ChannelSpec.mass_factor` and feeds both the assembly and the per-energy boundary.

**Path support.** Fully supported on the **direct path** (a per-energy symmetric linear solve; no metric, no generalized solver). On the **spectral path** per-channel μ makes `H u = E·diag(1/μ_c)·u` a generalized eigenproblem; the clean resolution is the standard symmetric MeV eigenproblem (`H_MeV u = ε u`, ε in MeV), which subsumes the single-μ fm⁻² form — deferred with the rest of the spectral path.

### 15.4 Convention summary — units and Hamiltonian scaling

**The assembler builds the block Hamiltonian in the symmetric MeV form (v1.3).** Mass is applied to the kinetic blocks; the coupling potential is left untouched and the block is symmetric. The specific rules:

| Quantity | User provides | Stored / computed as |
|---|---|---|
| Energies `E`, `E_c` | MeV | MeV |
| Lengths `a`, `h` | fm | fm |
| `Interaction.block` (V) | MeV | MeV (added untouched; never divided per channel) |
| `mass_factor` `m_c` (= ℏ²/2μ_c) | MeV·fm² | scales the diagonal kinetic block; `√m_c` in `Q'`; sets `k_c` |
| `mass_factor_grid` | MeV·fm², `(N_E, N_c)` | per-energy/channel override of `m_c` |
| `TpL`, `T`, `D`, `inv_r`, `inv_r2` | — | fm⁻² (mesh builders divide by `scale²`); `× m_c` → MeV in the block |
| assembled block / `C(E)` (direct path) | — | MeV (`C = H_MeV − E·I`) |
| `Spectrum.eigenvalues` (spectrum path) | — | fm⁻² (block ÷ uniform `m`); spectral observables unchanged |

Standard nuclear value: ℏ²/2mₙ = 20.736 MeV·fm² [2, eq. 46].

**Single-μ equivalence.** When all `m_c` are equal, dividing the block by that scalar `m` recovers the older fm⁻² convention (eigenvalues in fm⁻², `rmatrix_from_spectrum` denom `ε − E/m`), which the spectral path uses for the single-μ case. Observables are invariant under the rescaling.

**Multi-mass channels.** Genuinely different per-channel μ is supported on the **direct path** via the symmetric form above (this corrects the v1.2 behavior, which divided off-diagonal coupling by the row channel's mass and was therefore non-symmetric for multi-μ). The spectral-path generalized-eigenproblem version is deferred (§15.3).

### 15.5 Symmetry-block batching (the block axis)

> **Status: implemented and shipped.** This is the headline v1.5 feature. Both the direct path and the full spectral path vmap over the block axis;
> the equivalence tests (`tests/unit/test_blocks_{direct,spectral}.py`,
> `tests/benchmarks/test_blocks_partial_waves.py`) pin the batched run to per-block
> compiled solvers. Partial-wave batching is the `N_c = 1` special case of this feature.

A scattering calculation block-diagonalizes by conserved quantum numbers: each `(J, π)`
sector is an **independent** coupled-channel solve, and within a single-channel treatment
each partial wave `(ℓ, j)` is its own one-channel solve. These *symmetry blocks* are **not
coupled** to one another — they differ only in their per-channel `ℓ`/threshold/mass (hence
their centrifugal `ℓ(ℓ+1)/r²` and their boundary `F_ℓ, G_ℓ`) while sharing the mesh and the
mass-free kinetic. So the set of blocks is a **batch axis**, distinct from both the
coupled-channel axis *inside* a block (§15.1) and the energy axis (§12).

Stacking `N_b` blocks that share a channel shape `N_c` and `vmap`-ping a per-block solve is
the right structure: it is `N_b` independent `N_c·N`-sized solves, not one dense
`(N_b·N_c·N)³` solve (an `O(N_b²)`→`O(N_b³)` FLOP blowup). This is the **energy-axis vmap
mechanism (§4.2) applied along a second batch axis** — no new linear algebra.

**Compile-time set.** Like the energy grid, the block set is fixed at compile, because the
ℓ-dependent boundary `F_ℓ, G_ℓ` is baked per block via `mpmath`. A batch is declared with
`blocks=` (mutually exclusive with `channels=`):

```python
# Partial waves (N_c = 1 per block): the common single-channel case
solver = lax.compile(blocks=[[ChannelSpec(l=0, ...)],
                             [ChannelSpec(l=1, ...)],
                             [ChannelSpec(l=2, ...)]], ...)

# Coupled (J, π) groups (N_c > 1 per block), all sharing the same N_c
solver = lax.compile(blocks=[block_Jπ1, block_Jπ2, ...], ...)
```

All inner groups must have the same length `N_c`. Compile bakes the per-block centrifugal
and boundary as arrays stacked on a leading `(N_b,)` axis; the shared `T`/`inv_r2` are stored
once. Because the boundary depends only on `ℓ` (not `j`), it is deduped across blocks that
share an `ℓ` (e.g. `j = ℓ ± ½`).

**Interaction axis.** The `Interaction` gains an optional **leading block axis**, gated by a
static `block_dependent` flag exactly parallel to `energy_dependent` (§6):

```
block shape:  (N_b, M, M)   or   (N_b, N_E, M, M)
```

`interaction_from_array`/`interaction_from_funcs` accept a leading block axis on each term —
e.g. an ℓ-dependent non-local kernel `W[b, i, j]` (the case the local centrifugal/spin-orbit
operators cannot represent: microscopic exchange, Perey–Buck, or SCGF-derived optical
kernels). Local terms may also carry the axis (per-`(ℓ, j)` spin-orbit). A block-independent
interaction broadcasts across the `(N_b,)` axis.

**BoundaryValues axis.** Every `BoundaryValues` field gains a leading `(N_b,)` axis →
`(N_b, N_E, N_c)` — produced at compile by calling the existing per-`ℓ` Coulomb/Whittaker
routine (§9) once per distinct `ℓ` and stacking per block.

**Outputs.** `spectrum`, `rmatrix_direct`, `smatrix_direct`, `phases_direct`,
`wavefunction_direct` `vmap` over the block axis, returning e.g. `(N_b, N_E, N_c, N_c)`. **No**
`ℓ`/block argument is added to any observable — the block dependence is carried by the
`Interaction`, preserving the "solver = `Interaction` → outputs" invariant.

**Implementation sketch.** The mass-free kinetic `T` and the position operators are shared
across blocks; the *only* per-block operator change is the centrifugal `ℓ(ℓ+1)·inv_r2` added
to each channel's kinetic block, and the *only* per-block cached data is the boundary. So the
batch layer factors the §11.3 (and §11.1) per-block core out and `vmap`s it:

```python
# Batch the §11.3 direct solve over a compile-time set of symmetry blocks.

def make_block_batched_direct(mesh, operators, block_groups, energies,
                              boundary_blocks, mass_factor_grid=None):
    """block_groups     : tuple[tuple[ChannelSpec, ...], ...]  # N_b groups, same N_c (static)
       boundary_blocks  : BoundaryValues stacked on a leading (N_b,) axis
                          (F_ℓ, G_ℓ, k, η, is_open per block; deduped across shared ℓ)
    """
    # Per-block, per-channel centrifugal L_c(c) = ℓ_c(ℓ_c+1), shape (N_b, N_c)
    Lc = jnp.array([[ch.l * (ch.l + 1) for ch in grp] for grp in block_groups])

    @jax.jit
    def rmatrix_direct(interaction):
        blocks = interaction.block          # (N_b, M, M) or (N_b, N_E, M, M)
        def _one_block(Lc_row, boundary_b, blk):
            # _direct_solve is the §11.3 per-energy core, parameterized on the
            # per-channel centrifugal (added to the shared kinetic) and one block's
            # boundary, rather than reading ChannelSpec.l for a single fixed block.
            return _direct_solve(mesh, operators, Lc_row, energies,
                                 boundary_b, blk, mass_factor_grid)   # (N_E, N_c, N_c)
        return jax.vmap(_one_block)(Lc, boundary_blocks, blocks)      # (N_b, N_E, N_c, N_c)

    # smatrix_direct / phases_direct: rmatrix_direct → per-block boundary matching, vmapped.
    # wavefunction_direct: jax.vmap of the per-block C⁻¹·source over the (N_b,) axis.
    return rmatrix_direct, smatrix_direct, phases_direct, wavefunction_direct
```

The bulk of the work is refactoring §11.3 to expose
`_direct_solve(mesh, operators, centrifugal, energies, boundary, block, mass_factor_grid)`
(the existing body, with the per-channel centrifugal passed in rather than rebuilt from
`ChannelSpec.l`); the batch layer above is then a thin `vmap`. The spectrum kernel (§11.1)
factors the same way (`_one` parameterized on the per-block centrifugal/boundary and
`vmap`-ped over `(N_b,)`). Both compose with the existing per-energy vmap.

**One kernel family.** As shipped there are no separate single-block kernels: every
kernel and observable runs the batched code path, and a `channels=` compile is the
`N_b = 1` special case — its inputs are lifted onto a length-1 block axis and the axis
is squeezed off at the public boundary, so the single-block contract (no leading axis)
is unchanged.

**Composition of axes.** The canonical shape order is **block (batch) × channel (coupled
within a `(J, π)` block) × energy (vmap)**. Single-channel elastic is one channel per block
(partial waves); coupled inelastic/DWBA is `N_c > 1` per block, batched over `(J, π)`.

**Implementation map.** `compile(blocks=…)` baking (`compile.py`); static
`Interaction.block_dependent` (`types.py`); `(N_b,)`-stacked `BoundaryValues`
(`boundary/coulomb.py:compute_boundary_values_blocks`, deduped per distinct channel); the
array-parameterized per-block cores + `jax.vmap(block axis)` in
`solvers/{assembly,spectrum,linear_solve}.py` and the block-batched spectral observables in
`solvers/observables.py`; and a block axis on the
`interaction_from_array`/`interaction_from_funcs` terms (`operators/interaction.py` — a
block-dependent funcs term is a *sequence of N_b callables*). Constraints as shipped: the
spectral path requires one uniform mass factor across all blocks (per-block/per-channel μ is
a direct-path feature); `mass_factor_grid` is shared across blocks; `blocks=` excludes
propagated meshes.  As of v1.6 the `momenta=`/`grid=` transforms are block-batched —
`F_momentum` is `(N_b, N_c, M_k, N)` (deduped per unique ℓ across the whole block set),
`fourier`/`double_fourier_transform` accept `(N_b, …)` inputs and broadcast unbatched ones,
and the `grid=` projections pass arbitrary leading batch axes through (§13).

---

## 16. Public API and usage examples

### 16.1 Public namespace

```python
# lax/__init__.py
from .types import MeshSpec, ChannelSpec, Solver, Interaction
from .compile import compile
from .wavefunction import make_wavefunction_source
from . import spectral                              # mesh-independent submodule
from . import models                                # convenience physics models (optical, reid, presets)

__all__ = [
    "MeshSpec", "ChannelSpec", "Solver", "Interaction",
    "compile",
    "make_wavefunction_source",
    "spectral",
    "models",
]
```

`Interaction` objects are built via the solver methods — the single-kind
`solver.local_potential`/`solver.nonlocal_potential` (from a callable), or the list builders
`solver.interaction_from_{block,array,funcs}` (which close over the mesh, channels, and
energies). The type is exported for annotations and for the raw-block escape hatch. There are
no `assemble_local`/`assemble_nonlocal` exports (those primitives do not exist; the Gauss
scaling is internal to `interaction_from_array`, §8).

The `lax.spectral` submodule is exposed as a first-class peer because its functions (`rmatrix_from_spectrum`, `smatrix_from_R`, `coupled_channel_parameters_from_S`, etc.) are useful standalone — for postprocessing, for stitching different solvers together, or for implementing custom observables.

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

interaction = solver.nonlocal_potential(yamaguchi)   # single channel: coupling defaults to [[1.0]]

# One eigendecomposition, multiple observables:
spec    = solver.spectrum(interaction)
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

interaction = solver.local_potential(lambda r: -1.0 / r)   # single channel: coupling → [[1.0]]

spec = solver.spectrum(interaction)
# spec.eigenvalues[:7] should be {-1/2, -1/8, -1/18, -1/32, -1/50, -1/72, -1/98}
#                              = E_n = -1/(2n²)   (fm⁻²; block ÷ uniform m=0.5)
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
    interaction = solver.nonlocal_potential(kernel)    # single channel
    spec = solver.spectrum(interaction)                # ONE eigendecomp
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

def make_interaction(α, β):
    kernel = lambda r1, r2: -2*β * (α+β)**2 * jnp.exp(-β*(r1+r2)) * HBAR2_2MU
    return solver.nonlocal_potential(kernel)

interactions = jax.vmap(make_interaction)(αs, βs)    # Interaction; block (1000, 40, 40), energy_dependent=False
spec_batch   = jax.vmap(solver.spectrum)(interactions)   # batched Spectrum (one eigendecomp each)
δ_batch      = jax.vmap(solver.phases)(spec_batch)       # (1000, 100, 1) on GPU
```

Here the batch axis is over *potentials* (1000 different `(α, β)`), distinct from the energy axis (100 compile-time energies handled inside each `Spectrum`). `energy_dependent` stays `False`; vmap maps over the leading axis of `interactions.block`.

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

interaction = solver.local_potential(V_test)        # single channel
spec = solver.spectrum(interaction)

# Evaluate G(E) at arbitrary energies without recomputing the spectrum:
G_at_E   = solver.greens(spec, E=0.5)               # (M, M)
G_scan   = jax.vmap(lambda E: solver.greens(spec, E))(jnp.linspace(-2, 5, 200))
# G_scan: (200, M, M) — one eigendecomposition, 200 resolvents
```

### 16.7 Example 6: energy-dependent V(E)

For an energy-dependent potential the user builds an **energy-dependent `Interaction`** (`energy_dependent=True`; callables take a trailing `E`), whose block is `(N_E, M, M)`. `solver.spectrum(interaction)` then returns one `Spectrum` per energy. The S-matrix at each grid point must pair `spec_i` with *its own* energy `E_i` — using `solver.smatrix(spec_i)` would evaluate S at all N_E compile-time energies from a single spectrum, as if V were energy-independent, which is wrong.

The correct pattern uses `lax.spectral` functions directly, exploiting the fact that `BoundaryValues` is a pytree and vmap slices its leading axis automatically:

```python
energies_grid = jnp.linspace(0.1, 50.0, 21)            # sparse compile-time grid

solver = lax.compile(
    mesh             = lax.MeshSpec("legendre", "x", n=40, scale=10.0),
    channels         = (lax.ChannelSpec(l=0, threshold=0.0, mass_factor=HBAR2_2MU),),
    solvers          = ("spectrum",),          # no "smatrix" — we call spectral directly
    energies         = energies_grid,
)

params = {"alpha": 0.23, "beta": 1.39, "gamma": 0.01}

def kernel(r1, r2, E):
    """Energy-dependent non-local kernel (dispersive optical-model style)."""
    α, β, γ = params["alpha"], params["beta"], params["gamma"]
    return (-2*β*(α+β)**2 * jnp.exp(-β*(r1+r2)) + γ*E) * HBAR2_2MU

# Energy-dependent Interaction: block is (21, M, M); spectrum() decomposes per energy.
interaction = solver.nonlocal_potential(kernel, energy_dependent=True)
spec_grid = solver.spectrum(interaction)              # batched Spectrum (one per energy)

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

# Off-grid energies: recompile on a finer grid, or interpolate S_grid yourself
# with an observable-appropriate scheme (§12).
```

Key points:
- `solver.boundary` is vmapped over in `S_at_own_energy`: each call receives `bdy_i` with field shapes `(N_c,)` (the per-energy slice).
- `solver.smatrix(spec)` is intentionally NOT used here — that function always evaluates S at all N_E compile-time energies from a single energy-independent spectrum.
- `energy_dependent=True` in `compile()` is informational metadata only; the runtime API is unchanged. The user signals intent through how they call `vmap`.
- Differentiating through `S_grid` with respect to `params` works because the entire chain (`make_V_at` → `solver.spectrum` → `S_at_own_energy`) is JAX-traceable.

### 16.8 Example 7: complex V on GPU via the direct path

For an optical-model potential (complex V) on GPU, the `eig` method is not GPU-ready. The user picks the linear-solve method explicitly and builds an `Interaction`; `rmatrix_direct` returns the R-matrix at all compile-time energies in one vectorized call, and `wavefunction_direct` returns the internal distorted wave.

```python
solver = lax.compile(
    mesh         = lax.MeshSpec("legendre", "x", n=60, scale=14.0),
    channels     = (lax.ChannelSpec(l=20, threshold=0.0, mass_factor=20.736/4),),
    operators    = ("T+L",),
    solvers      = ("rmatrix_direct", "wavefunction"),   # wavefunction → direct-path solve
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
    R_c = R_nuc
    V_coul = jnp.where(r > R_c, 2*82*1.44 / r,
                       2*82*1.44 / (2*R_c) * (3 - (r/R_c)**2))
    return -V0 * f - 1j * W0 * f + V_coul

# Build the Interaction (single channel → default [[1]] coupling):
interaction = solver.local_potential(V_optical)

# Per-energy linear solve, no spectrum:
R_direct = solver.rmatrix_direct(interaction)          # (5, 1, 1) complex
S_direct = solver.smatrix_direct(interaction)          # (5, 1, 1) — matched internally

# Internal distorted wave at the 50 MeV grid point (energy_index=4):
src = lax.make_wavefunction_source(solver, channel_index=0, energy_index=4)
psi = solver.wavefunction_direct(interaction, src, energy_index=4)   # (M,)
u_r = solver.to_grid_vector(psi)                       # u(r) on the fine grid
# Reproduces Appendix A (collision matrix) and Fig. 2 (wavefunction) of Descouvemont [2].
```

Green's functions still require a `Spectrum` (recompile with `method="eig"`); the internal wavefunction, however, is now available directly via the linear solve.

### 16.9 Example 8: partial-wave / symmetry-block batching (§15.5)

A neutron elastic-scattering phase-shift calculation needs many partial waves. Each `ℓ` is an
independent single-channel solve sharing the mesh and kinetic — the `N_c = 1` case of
symmetry-block batching. Declaring the waves as `blocks` batches them along one leading axis:

```python
ells = range(0, 9)                                  # ℓ = 0 .. 8
solver = lax.compile(
    mesh     = lax.MeshSpec("legendre", "x", n=40, scale=12.0),
    blocks   = [[lax.ChannelSpec(l=ell, threshold=0.0, mass_factor=HBAR2_2MU)]
                for ell in ells],                   # N_b = 9 single-channel blocks
    solvers  = ("rmatrix_direct",),                 # binds {rmatrix,smatrix,phases}_direct
    energies = jnp.linspace(0.1, 50.0, 100),
)

# One kernel per block (ℓ-dependent non-local kernel): a block-dependent funcs
# term is a sequence of N_b callables, one per block, stacked at build time.
interaction = solver.interaction_from_funcs(
    nonlocal_=[(W_per_ell, coupling)],              # W_per_ell = [W_l0, W_l1, …], each W(r, r')
    block_dependent=True,
)                                                   # block shape (9, M, M)

δ = solver.phases_direct(interaction)               # (9, 100, 1): block × energy × N_c
#  — one vectorized call over all 9 partial waves, no Python loop.
```

For coupled `(J, π)` groups (`N_c > 1`), each inner list holds that block's channels (all
blocks sharing the same `N_c`), and the outputs carry the same leading `(N_b,)` axis.

### 16.10 Example 9: batched (p,n) DWBA transition elements (v1.6)

The driving case for `wavefunction_grid` + `matrix_element`: distorted waves from two
compiled solvers (proton entrance with Coulomb, neutron exit without), contracted with an
isovector transition operator — **non-conjugated**, batched over every `(block, energy)`
pair in one call:

```python
mesh = lax.MeshSpec("legendre", "x", n=40, scale=12.0)
blocks = [[lax.ChannelSpec(l=ell, threshold=0.0, mass_factor=mf_p)] for ell in ells]
proton = lax.compile(mesh=mesh, blocks=blocks, solvers=("spectrum", "wavefunction"),
                     energies=E_p, z1z2=(1, Z), V_is_complex=True, method="eig")
neutron = lax.compile(mesh=mesh, blocks=[[c._replace(mass_factor=mf_n)] for [c] in blocks],
                      solvers=("spectrum", "wavefunction"),
                      energies=E_p - Q_pn, V_is_complex=True, method="eig")

chi_p = proton.wavefunction_grid(proton.spectrum(V_p))     # (N_b, N_E, M)
chi_n = neutron.wavefunction_grid(neutron.spectrum(V_n))   # (N_b, N_E, M)
U1 = proton.interaction_from_array(local=[(U1_stack, A)], block_dependent=True)

k_p = proton.boundary.k[:, :, 0]                           # (N_b, N_E)
k_n = neutron.boundary.k[:, :, 0]
T_pn = proton.matrix_element(chi_p, chi_n, U1, conjugate=False) / (mesh.scale * k_p * k_n)
#       (N_b, N_E) — the whole partial-wave × energy T-matrix in one bilinear call.
```

Both solvers must share the identical mesh (the Gauss scaling inside `U1.block` belongs to
it) and index-aligned block sets and energy grids — invariants the caller owns.  For
gradient/UQ work replace the `eig` spectral path with `method="linear_solve"` and
`wavefunction_direct_grid(V)` (Appendix C.11).

---

## 17. JAX considerations

### 17.1 Pytree registration

All numerical-data dataclasses (`Mesh`, `PropagationMatrices`, `OperatorMatrices`, `BoundaryValues`, `TransformMatrices`, `Interaction`, `Spectrum`) are registered using the `@jax.tree_util.register_dataclass` decorator (JAX >= 0.4.36). Fields default to pytree leaves; structural fields are marked with `field(metadata={"static": True})`. Use `jax.tree.map` (not the deprecated `jax.tree_util.tree_map`) when walking pytrees in library code. This registration ensures:

- Tracing-time fields (`n`, `family`, `regularization`, `is_hermitian`) are baked into the JIT cache.
- Numerical leaves flow through `jit`, `vmap`, `grad` transparently.
- A `Spectrum` returned from one function can be fed into another as a single argument — JAX traces through it without manual unpacking.

`Solver` is **not** a pytree (it holds Python callables). It is a plain frozen dataclass used as a namespace. Those callables must remain importable / reconstructible so a compiled solver is pickleable.

### 17.2 Method dispatch and tracing

`method` is a compile-time choice baked into the JIT'd `spectrum` kernel. Changing it requires rebuilding the `Solver`. The three methods have different traceability properties:

| Method | Internally calls | Differentiable? | `vmap` over V? | GPU? |
|---|---|---|---|---|
| `eigh` | `jnp.linalg.eigh` | Yes (closed-form JVP) | Yes | Yes |
| `eig`  | host `np.linalg.eig` via `jax.pure_callback` | **No** (the callback has no JVP/VJP rule — Appendix C.11) | Sequential | No |
| `linear_solve` | `jnp.linalg.solve` per energy | Yes | Yes | Yes |

The `eigh` derivative rule has a known degeneracy issue at eigenvalue crossings — gradients become large or ill-defined when two eigenvalues collide [see `jax.experimental.linalg.eigh` notes]. For potential-fitting workflows this is rarely problematic in practice (level crossings as a function of potential parameters are measure-zero), but it's worth documenting.

### 17.3 Dtype and precision

Default `float64` for everything except where promoted to complex. JAX disables `float64` by default; the library calls `jax.config.update("jax_enable_x64", True)` at import time.

For the Yamaguchi benchmark, achieving the published accuracy ($\delta = 85.634560°$ to 6 digits) requires `float64`. `float32` is offered as a fast path for fits where 4-digit accuracy in the loss is sufficient — set `dtype=jnp.float32` in `compile()`.

### 17.4 Devices and sharding

The `device` argument to `compile()` places all cached arrays on the requested device. Subsequent solver calls receive `V` and produce outputs on the same device. For multi-device sharding the user wraps `solver.spectrum` in `jax.shard_map` or uses `jax.pmap` on the leading batch axis of a vmap'd call.

### 17.5 Gradient support

The hot path is `eigh` (default) or `solve` (R-matrix-direct path). Both have JAX-defined custom JVP/VJP rules and are fully differentiable. The compile-time non-JAX components (mpmath, scipy.special) never enter a backward pass.  The **`eig` path is the exception**: its spectra flow through a runtime host callback (`jax.pure_callback` → `np.linalg.eig`) with no JVP/VJP rule, so `jax.grad` through any `eig`-path spectrum raises — gradient/UQ pipelines with complex optical potentials use `method="linear_solve"` (+ `wavefunction_direct_grid`) or restrict to real-V `eigh` problems (Appendix C.11).

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
| Energy-dependent V(E) aligned grid | Legendre $x$, $N=40$, sparse grid | spectrum_batch → smatrix_grid | matches per-energy recompiles | $10^{-10}$ |

The Yamaguchi test is the keystone end-to-end test: it exercises non-local potential assembly, the spectrum kernel, the spectral R-matrix sum, the Coulomb boundary path, the S-matrix matching, and phase-shift extraction. It must pass before anything else is merged.

The three α + ²⁰⁸Pb rows verify that `eigh` (real), `eig` (complex CPU), and `linear_solve` (complex GPU) all produce the same physical answer.

### 18.2 Property tests

These run on every PR via `pytest` with `hypothesis` for fuzzed inputs:

1. **Hermiticity of $T+L$.** $\|T+L - (T+L)^T\|_F / \|T+L\|_F < 10^{-12}$ for any (mesh, regularization).
2. **Spectrum-vs-direct cross-check.** For any real V, `solver.rmatrix(spec, E)` from the spectral sum agrees with `solver.rmatrix_direct(interaction)` (same V, built via `interaction_from_*`) evaluated at the same E to $10^{-10}$. This is the most valuable consistency test in the suite — catches both spectral-sum bugs and linear-solve bugs simultaneously. (For multi-μ, the spectral side is single-μ in scope, so the cross-check runs with a uniform mass factor.)
3. **Unitarity of $S$ for real V.** $\|S^\dagger S - I\|_F < 10^{-10}$ for any real potential and any energy.
4. **Symmetry of $S$.** $\|S - S^T\| < 10^{-10}$ for any real symmetric V.
5. **Wronskian.** $FG' - GF' = 1$ at every boundary computation (catches mpmath setup errors).
6. **Pole structure.** For known resonances, the eigenvalues $\varepsilon_k$ from `spectrum(interaction)` lie near the resonance energy with width related to the imaginary part of the surface amplitudes (as $V$ approaches the resonance condition).
7. **vmap parity.** A Python `for` loop over 100 energies and `vmap(solver.rmatrix)(spec, energies)` produce identical outputs.
8. **JIT cache stability.** Calling `solver.spectrum(interaction)` twice with different interaction blocks of the same shape does not recompile.
9. **Autograd correctness.** `jax.test_util.check_grads(loss, ...)` passes for `loss(params) = ||S(params) - S_target||²` with finite-difference vs autograd agreement at 1e-5.
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
- Single-shot `solver.spectrum(interaction)` time, CPU and GPU.
- Energy-scan throughput (problems per second) at various $N$, $N_c$, $N_E$.
- Speedup of `spectrum → smatrix` over `rmatrix_direct → smatrix` as a function of $N_E$.

The expected scaling: `spectrum`-path cost is dominated by one $O((N_c N)^3)$ eigendecomposition; `rmatrix_direct` cost is $N_E$ solves of the same complexity. For $N_E \gtrsim 5$ the spectrum path should always win.

---

## 19. References

[1] **D. Baye**, *The Lagrange-mesh method*, **Physics Reports 565**, 1–107 (2015). DOI: [10.1016/j.physrep.2014.11.006](https://doi.org/10.1016/j.physrep.2014.11.006).

The definitive review of the LMM. Section 2 is the theoretical foundation; Section 3 provides explicit formulas for every mesh family and regularization; Sections 5–6 cover bound-state and continuum applications. This document references specific equations as "[Baye eq. X.Y]" or sections as "[Baye §X]".

[2] **P. Descouvemont**, *An R-matrix package for coupled-channel problems in nuclear physics*, **Computer Physics Communications 200**, 199–219 (2016). DOI: [10.1016/j.cpc.2015.10.015](https://doi.org/10.1016/j.cpc.2015.10.015). arXiv: [1510.03540](https://arxiv.org/abs/1510.03540).

The Fortran R-matrix package whose architecture and Lagrange-Legendre formulas this library mirrors. Section 2 gives the R-matrix-on-Lagrange-mesh formalism; eqs. 14–17 are the central solver equations; eqs. 18–24 give the explicit shifted Legendre-$x$ formulas; §2.4 covers R-matrix propagation; §5 gives validation examples (with Example 5 being the Yamaguchi non-local potential used in this library's primary benchmark).

[3] **A. M. Lane, R. G. Thomas**, *R-matrix theory of nuclear reactions*, **Reviews of Modern Physics 30**, 257 (1958). The original R-matrix formalism with the Wigner-Eisenbud spectral decomposition that this library's `spectral` submodule reflects.

[4] **M. Hesse, J.-M. Sparenberg, F. Van Raemdonck, D. Baye**, *Coupled-channel R-matrix method on a Lagrange mesh*, **Nuclear Physics A 640**, 37–51 (1998). The original coupled-channel R-matrix-on-Lagrange-mesh paper.

[5] **M. Hesse, J. Roland, D. Baye**, *Solution of the Yamaguchi nonlocal problem on a Lagrange mesh*, **Nuclear Physics A 709**, 184–195 (2002). Original non-local potential application; reference values for the Yamaguchi benchmark.

[6] **P. Descouvemont, D. Baye**, *The R-matrix theory*, **Reports on Progress in Physics 73**, 036301 (2010). General review of R-matrix theory in nuclear physics; discusses phenomenological R-matrix fitting in terms of poles and reduced widths, which are directly accessible from the `Spectrum` object.
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
| Boundary | $\varphi_n(a) = 0$ (basis $\propto x(1-x)$ vanishes at both endpoints — confined) | – |
| $T$ Gauss diag | $\frac{1}{a^2 \cdot 3 x_i(1-x_i)}[N(N+1) + 1/(x_i(1-x_i))]$ | Baye 3.143 |
| $T$ Gauss off | $\frac{1}{a^2} \cdot \frac{(-1)^{i-j}(x_i + x_j - 2 x_i x_j)}{R_{ij}(x_i - x_j)^2}$ where $R_{ij} = \sqrt{x_i(1-x_i)x_j(1-x_j)}$ | Baye 3.142 |
| Exact $T$ correction | $T_{ij} = T^G_{ij} - \frac{(-1)^{i-j} N(N+1)}{a^2(2N+1) R_{ij}}$ | Baye 3.144 |
| Overlap (off-diag) | nonzero, see Baye 3.139 — but treat as orthonormal for LMM | – |

**Shifted Legendre regularized by $x^{3/2}$ (hyperspherical, Baye §3.4.6).** This shares the
A.1 nodes/weights but the regularization factor is $R(x) = x^{3/2}$ rather than $x$, so the
**boundary value picks up an extra $x_n^{-1/2}$** relative to the $\nu = 1$ form:
$\varphi_n(a) \propto x_n^{-1/2}\,\varphi_n^{(\nu=1)}(a)$. ⚠️ *Verify against Baye §3.4.6
before relying on this:* the current `meshes/legendre.py::_legendre_boundary_values` reuses
the $x$-regularization boundary formula, which omits the extra factor — reconcile the code
and this row together.

### A.4 Modified Laguerre $t = x^2$, regularized by $x$  [Baye §3.3.7]

For 3D harmonic-oscillator-like problems.

| Quantity | Formula | Reference |
|---|---|---|
| Mesh points | $L^\alpha_N(x_i^2) = 0$ | Baye 3.82 |
| Basis | $\hat f_j(r) = \frac{r}{x_j} \cdot \text{(modified Laguerre eq. 3.83)}$ | Baye 3.92 |
| $\hat T$ Gauss diag | $\frac{1}{3 h^2}\left[-x_i^2 + 2(2N+\alpha+1) + (2\alpha^2 - 2)/x_i^2\right]$, with $\alpha = \tfrac12$ for $\ell = 0$ | Baye 3.88–3.91 |
| $\hat T$ Gauss off | $\frac{(-1)^{i-j} \cdot 4(x_i^2 + x_j^2)}{h^2 (x_i^2 - x_j^2)^2}$ | Baye 3.93 |

> **A.4 reconciliation (verify).** This diagonal matches the implementation
> (`meshes/laguerre.py`), which uses the unregularized Baye 3.88–3.91 form
> $(2\alpha^2-2)/x_i^2$ with $\alpha = \tfrac12$ at $\ell = 0$, and the 3D harmonic-oscillator
> benchmark passes against it. An earlier draft cited the *regularized* Baye 3.94 form
> $-(\alpha^2 - 3/4)/x_i^2$, which differs (e.g. $-3/2$ vs $+1/2$ at $\alpha = \tfrac12$).
> Confirm the intended Baye equation and the $\alpha\leftrightarrow\ell$ convention before
> changing either side; the code is treated as the source of truth here.

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

### C.4 Non-local kernels through the `Interaction` interface

The Yamaguchi kernel:
```python
def yamaguchi(r1, r2):
    return -2 * BETA * (ALPHA + BETA)**2 * exp(-BETA * (r1 + r2)) * HBAR2_2MU
```
returns values in **MeV** (the `* HBAR2_2MU` factor). Pass it as a non-local potential:
`solver.nonlocal_potential(yamaguchi)` (single channel; or `interaction_from_funcs(nonlocal_=[yamaguchi])`). The builder Gauss-scales it to `block[i,j] += √(λ_i λ_j)·a·yamaguchi(r_i, r_j)` (MeV) and the assembler adds it to the block **untouched** (symmetric MeV form, §11.5) — there is no longer a per-channel division to reconcile, so the result matches the `test_yamaguchi.py` prototype directly. The channel's `mass_factor = HBAR2_2MU` enters only through the kinetic block and the boundary `k`. (A kernel pre-divided by ℏ²/2μ is no longer a meaningful special case, since V is not divided.)

### C.5 `eigh` derivative at near-degenerate eigenvalues

The VJP of `jnp.linalg.eigh` involves terms of the form $1/(\varepsilon_i - \varepsilon_j)$. When two eigenvalues are nearly equal (which can happen for deep bound states in a large basis), the gradient spikes. For fitting workflows, regularize by:

```python
# Soft degenerate eigenvalue regularizer:
eps_reg = 1e-6   # in fm⁻²
# This is NOT built into the library — add to the user's loss function if needed.
```

Alternatively, use `jax.experimental.linalg.eigh_generalized` or add a small random perturbation to H before differentiation (breaks exact symmetry but stabilizes gradients).

### C.6 Complex Coulomb and Sommerfeld parameter

For charged particles with $\eta \neq 0$, `compute_boundary_values` calls `mpmath.coulombf(l, eta, rho)` and `mpmath.coulombg(l, eta, rho)`. With `dps=40` these are reliable for all $l$ and moderate $\eta$. Two known edge cases:

- **Very large $\eta$ (heavy-ion Coulomb)**: `mpmath` may be slow (> 1 s per evaluation) for $\eta \gg 1$ at low energy. Increase `dps` or use a dedicated asymptotic expansion.
- **Very small $\rho = ka$ (sub-barrier)**: When $ka \ll 1$ the Coulomb functions are dominated by the centrifugal barrier. `mpmath` handles this correctly but returns very large $G_L$ and very small $F_L$. The R-matrix then involves differences of large numbers; the `dps=40` setting provides sufficient guard digits.

### C.7 The `is_open` mask and closed channels in v1

In v1, closed-channel rows/columns of the S-matrix are masked to zero in `_project_open` (not decoupled via the Whittaker boundary condition method of [2, eq. 9]). This is exact when the Bloch boundary parameter $B_c$ is set to eliminate the $L(B_c) u_{ext}$ term for closed channels. Until Phase 9 implements the Whittaker path:

- For energies where all channels are open, results are exact.
- For energies where some channels are closed but far from threshold, the masking approximation is good.
- For energies very close to a channel threshold, small systematic errors may appear (the eigenvectors of H are not aware of the closed-channel matching condition). Flag these energies by checking `solver.boundary.is_open`.

### C.8 `Spectrum` pytree vmap behavior

When `jax.vmap(f)(batched_spectrum)` is called, JAX treats `Spectrum` as a pytree and maps over the leading axis of each array field:
- `eigenvalues`: `(B, M)` → each call sees `(M,)`
- `surface_amplitudes`: `(B, M, N_c)` → each call sees `(M, N_c)`
- `eigenvectors`: `(B, M, M)` or `None`
- `is_hermitian`: static, not batched (same bool for all)

This means `jax.vmap(solver.spectrum)(V_batch)` where `V_batch` has shape `(B, N_c, N_c, N, N)` produces a `Spectrum` with leading batch axis `B` — directly usable in `jax.vmap(S_at_own_energy)(spec_grid, ...)` as in Example 16.7.

`BoundaryValues` vmaps the same way: a `BoundaryValues` with `H_plus` of shape `(N_E, N_c)` when sliced under `jax.vmap` gives per-energy slices of shape `(N_c,)`.

### C.9 `jax.config.update("jax_enable_x64", True)`

This must be called **before any JAX operation**. The library calls it at the top of `lax/__init__.py`:

```python
import jax
jax.config.update("jax_enable_x64", True)
```

If the user imports JAX before importing `lax`, the call may not take effect (JAX freezes the config on first array creation). The fix is to import `lax` first, or to put the config call at the top of the user's script before any `import jax.numpy as jnp` usage. Document this prominently in the README.

### C.10 Non-uniform `mass_factor_grid` forces the energy-batched spectral regime (v1.6)

With a non-uniform μ(E) the dimensionless Hamiltonian `H/μ(E)` itself changes with energy,
so "diagonalize once, evaluate many energies" is invalid **physics**, not just missing
plumbing: a static-regime evaluation would mix per-energy boundary values with a single-μ
spectral denominator — silently wrong matching.  As shipped in v1.6: (1) the spectrum
kernel assembles each per-energy Hamiltonian with its own μ_e (the energy-batched path
takes a `(N_E,)` `mass_factor_grid`, validated per-channel-uniform for the spectral path);
(2) `spectrum(V)` auto-routes to the energy-batched path when
`V.energy_dependent or mass_factor_nonuniform` — never on `V.energy_dependent` alone;
(3) the five static-regime observables (`rmatrix`, `smatrix`, `phases`, `greens`,
`wavefunction`) are bound to pickle-safe **raising stubs** (`_NonuniformMassFactorStub`)
that point at the `*_grid` observables and the direct path; `wavefunction_grid`
additionally raises in its static branch if handed a foreign static spectrum.  Per-channel
non-uniform grids remain a direct-path-only feature and are rejected at compile for the
spectral path.

### C.11 `method="eig"` spectra are not differentiable (v1.6)

Complex-symmetric spectra are computed through `jax.pure_callback` → host
`np.linalg.eig` (`solvers/spectrum.py::_eig_via_callback`).  The callback carries no
JVP/VJP rule, so `jax.grad` through **any** `eig`-path spectrum raises ("Pure callbacks do
not support JVP").  DWBA distorted waves from complex optical potentials are exactly the
`eig` case: gradient/UQ pipelines must use `method="linear_solve"` +
`wavefunction_direct_grid` (fully differentiable through `jnp.linalg.solve`) or restrict
to real-V `eigh` problems.  A test asserts the raise so a future custom-JVP upgrade is
noticed.
---

*End of design document. Version 1.6, intended for offline reference during library development.*

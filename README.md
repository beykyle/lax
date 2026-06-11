# lax

[![CI](https://github.com/beykyle/lax/actions/workflows/ci.yml/badge.svg)](https://github.com/beykyle/lax/actions/workflows/ci.yml)
[![Docs](https://github.com/beykyle/lax/actions/workflows/docs.yml/badge.svg)](https://beykyle.github.io/lax/)

JAX-compiled Lagrange-mesh solvers for quantum scattering and bound-state problems.

## Documentation

Full API reference and rendered examples: **https://beykyle.github.io/lax/**

## Installation

`lax` requires JAX with a compiled backend (`jaxlib`). Install the
backend that matches your hardware. All backends are available from PyPI unless
noted otherwise.

### CPU (any platform)

Works on Linux x86_64, Linux aarch64, macOS Apple Silicon, and Windows x86_64.

```bash
uv sync --extra cpu --group dev
```

### NVIDIA GPU — CUDA 13 (recommended)

Requires Linux, SM 7.5+ GPU, driver ≥ 580.

```bash
uv sync --extra cuda13 --group dev
```

### NVIDIA GPU — CUDA 12

For older drivers (SM 5.2+, driver ≥ 525). Supports Windows WSL2 experimentally.

```bash
uv sync --extra cuda12 --group dev
```

### NVIDIA GPU — local CUDA toolkit

If CUDA is already installed on the host machine rather than via pip:

```bash
# CUDA 13
uv sync --extra cuda13-local --group dev

# CUDA 12
uv sync --extra cuda12-local --group dev
```

### AMD GPU — ROCm (Linux)

Requires ROCm 7 installed locally. See
[AMD's instructions](https://github.com/jax-ml/jax/blob/main/build/rocm/README.md)
for prerequisites. ROCm support on Windows WSL2 is experimental.

```bash
uv sync --extra rocm --group dev
```

### Google Cloud TPU

```bash
uv sync --extra tpu --group dev
```

### Notes

- **Install exactly one backend.** Having both `jax[cpu]` and `jax[cuda13]` in the
  same environment produces undefined behaviour.
- **GPU developers:** `uv sync --group dev` installs the CPU backend by default
  (so the environment works everywhere). Run your GPU backend sync afterward to
  swap it out.
- **Apple Silicon GPU** is not yet supported by JAX; use the CPU backend on macOS.
- **Intel GPU** support is experimental via a third-party plugin; see
  [intel-extension-for-openxla](https://github.com/intel/intel-extension-for-openxla)
  for installation instructions.

## Quick start

```python
import lax   # must come before jax.numpy (sets x64 mode)
import lax.constants as C
import jax.numpy as jnp

HBAR2_2MU = C.hbar2_over_2mu(1.008665, 1.008665)  # ≈ 41.47 MeV·fm² for n-n

solver = lax.compile(
    mesh     = lax.MeshSpec("legendre", "x", n=20, scale=8.0),
    channels = (lax.ChannelSpec(l=0, threshold=0.0, mass_factor=HBAR2_2MU),),
    solvers  = ("spectrum", "phases"),
    energies = jnp.array([0.1, 10.0]),
)
```

> **Note:** `lax.compile` shadows Python's built-in `compile`.
> Avoid `from lax import compile` in modules that also use the built-in.

## Currently supported meshes and methods

- **Mesh families:** `legendre`, `laguerre`
- **Legendre regularizations:** `x`, `x(1-x)`, `x^3/2`
- **Laguerre regularizations:** `x`, `modified_x^2`
- **Methods:** `eigh`, `eig`, `linear_solve`

## Running tests

```bash
# Fast unit tests only
uv run pytest tests/unit/ -n auto

# Full suite including benchmarks
uv run pytest tests/ -n auto

# Specific benchmark
uv run pytest tests/benchmarks/test_yamaguchi.py -v
```

## Examples and notebooks

```bash
uv sync --group dev --group jupyter
uv run jupyter lab
```

## Development

See `DESIGN.md` for the full architecture documentation and
`.github/copilot-instructions.md` for coding conventions.

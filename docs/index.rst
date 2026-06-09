lax
===

**JAX-compiled Lagrange-mesh solvers for quantum scattering and bound-state problems.**

``lax`` is a low-level numerical engine implementing the Lagrange-mesh method (LMM)
for solving the radial Schrödinger equation.  It supports bound-state (eigenvalue) and
continuum (R-matrix, S-matrix, Green's function) calculations through a unified
spectral-decomposition kernel, and is designed for use inside reaction codes,
potential-fitting pipelines, and uncertainty-quantification workflows.

Key features:

- One eigendecomposition supports R-matrix, S-matrix, phase shifts, and Green's
  functions at arbitrary energies — no per-energy linear solves.
- Full ``jit``, ``vmap``, and ``grad`` compatibility: push batches of potentials
  through a precompiled solver on CPU or GPU.
- Coupled-channel, non-local potentials, optical (absorptive) potentials, and
  R-matrix subinterval propagation.
- Legendre (finite-interval) and Laguerre (semi-infinite) mesh families, each with
  multiple endpoint regularizations.

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api
   examples

Links
-----

- `Source code <https://github.com/beykyle/lax>`_
- `Architecture reference (DESIGN.md) <https://github.com/beykyle/lax/blob/main/DESIGN.md>`_
- `Issue tracker <https://github.com/beykyle/lax/issues>`_

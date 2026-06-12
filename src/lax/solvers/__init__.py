"""Solver assembly and runtime-kernel construction."""

from lax.solvers.assembly import assemble_block_hamiltonian, build_Q
from lax.solvers.linear_solve import (
    make_direct_wavefunction_kernel,
    make_phases_direct_observable,
    make_rmatrix_direct_kernel,
    make_smatrix_direct_observable,
)
from lax.solvers.observables import (
    bind_grid_observables,
    bind_observables,
)
from lax.solvers.spectrum import make_spectrum_kernel

__all__ = [
    "assemble_block_hamiltonian",
    "bind_grid_observables",
    "bind_observables",
    "build_Q",
    "make_direct_wavefunction_kernel",
    "make_phases_direct_observable",
    "make_rmatrix_direct_kernel",
    "make_smatrix_direct_observable",
    "make_spectrum_kernel",
]

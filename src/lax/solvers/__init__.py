"""Solver assembly and runtime-kernel construction."""

from lax.solvers.assembly import assemble_block_hamiltonian, build_Q
from lax.solvers.linear_solve import make_rmatrix_direct_kernel
from lax.solvers.observables import bind_observables
from lax.solvers.spectrum import make_spectrum_kernel

__all__ = [
    "assemble_block_hamiltonian",
    "bind_observables",
    "build_Q",
    "make_rmatrix_direct_kernel",
    "make_spectrum_kernel",
]

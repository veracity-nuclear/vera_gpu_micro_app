#pragma once

#include <petscpc.h>

// Registers the "kokkosbgs" PETSc preconditioner type: a block red-black Gauss-Seidel / SOR sweep
// implemented in Kokkos for block-structured AIJ matrices (block size = rows per cell, i.e. nGroups).
//
// Each block's diagonal sub-matrix is inverted once in PCSetUp; one sweep updates all blocks of one color
// at a time (colors come from a greedy coloring of the block graph, two colors on a Cartesian mesh), each
// block being an independent dense (bs x bs) solve. Relaxation follows MPACT's RBSOR (Stimpson & Collins,
// "Implementation of a Red-Black SOR CMFD Solver in MPACT", Trans. ANS 115, 2016): cyclic Chebyshev
// semi-iterative (CCSI) factors, different for the red and black half-sweeps, from a Jacobi spectral radius
// estimated with unrelaxed sweeps at setup. Options:
//   -pc_type kokkosbgs -pc_kokkosbgs_sweeps <n> (default 1)
//   -pc_kokkosbgs_adaptive <bool> (default true; falls back to the fixed omega when the graph needs > 2 colors)
//   -pc_kokkosbgs_omega <w> (fixed factor, default 1.0) -pc_kokkosbgs_rho_sweeps <n> (default 10)
// With -ksp_type richardson -ksp_max_it 1 -pc_kokkosbgs_sweeps N it is MPACT's inner iteration: N RBSOR
// sweeps per eigenvalue update.
PetscErrorCode PCRegisterKokkosBlockGS();

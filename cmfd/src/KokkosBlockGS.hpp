#pragma once

#include <petscpc.h>

// Registers the "kokkosbgs" PETSc preconditioner type: a block red-black Gauss-Seidel / SOR sweep
// implemented in Kokkos for block-structured AIJ matrices (block size = rows per cell, i.e. nGroups).
//
// Each block's diagonal sub-matrix is inverted once in PCSetUp; one sweep updates all blocks of one color
// at a time (colors come from a greedy coloring of the block graph, two colors on a Cartesian mesh), each
// block being an independent dense (bs x bs) solve. Options:
//   -pc_type kokkosbgs -pc_kokkosbgs_sweeps <n> (default 1) -pc_kokkosbgs_omega <w> (default 1.0)
// With -ksp_type richardson it becomes the plain stationary block-SOR iteration.
PetscErrorCode PCRegisterKokkosBlockGS();

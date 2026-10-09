#pragma once

#include "petscvec_kokkos.hpp"
#include "petscmat_kokkos.hpp"
#include "petscksp.h"

#include <memory>
#include <string>
#include <vector>

#include "PetscMatrixAssembler.hpp"

// Generalized Davidson eigenvalue solver for the CMFD problem  M phi = (1/keff) F phi.
//
// Follows the approach used for MPACT in CASL-X-2016-1106-000 (Collins, Hamilton, Stimpson, Yee, Larsen,
// Kochunas, 2016): instead of a (shifted) power iteration whose every step needs a converged linear solve
// with M, a subspace V is grown one vector per iteration. Each iteration does a Rayleigh-Ritz solve of the
// projected problem (V^T M V) y = theta (V^T F V) y for the largest keff = 1/theta, forms the residual
// r = M phi - theta F phi of the Ritz pair, and adds the preconditioned correction t = P^-1 r to V, where
// P approximates M (not the near-singular shifted operator). No linear system is solved; P here is the
// Kokkos block red-black Gauss-Seidel preconditioner from KokkosBlockGS.hpp (a few sweeps per
// application), which is the GPU-friendly analogue of the AMG/ILU preconditioners used in the report.
//
// Cost per iteration: one M matvec, one fission-operator application, one preconditioner application and
// O(m) dot products / axpys for the subspace of size m. The subspace is restarted from the current Ritz
// vector when it reaches maxSubspace.
//
// Options (PETSc, prefix davidson_):
//   -davidson_max_subspace <m>      subspace size before restart (default 20)
//   -davidson_pc_type <type>        preconditioner (default kokkosbgs)
//   -davidson_pc_kokkosbgs_sweeps   block RB-GS sweeps per correction (default 6)
//   -davidson_rtol <r>              relative residual ||r|| / ||M phi|| required in addition to |dk| < tol (default 1e-8)
struct DavidsonEigenSolver
{
    using AssemblerPtr = std::unique_ptr<MatrixAssemblerInterface>;

    static constexpr double tol = 1.e-7;   // convergence tolerance on keff (same as PetscEigenSolver)
    // Relative eigen-residual tolerance. The eigenvalue error of a non-symmetric pencil is first order in
    // the eigenvector error, so this must be ~tol/10 to guarantee |keff - exact| < tol (1e-6 left the
    // weakly preconditioned Simple-assembler case 1.7e-7 off while satisfying |dk| < tol).
    double rtol = 1.e-8;
    PetscInt maxSubspace = 20;

    double keff = 1.0;
    std::vector<double> keffHistory;
    std::vector<double> residualHistory;

    Vec currentFlux = nullptr;
    PC pc = nullptr;
    AssemblerPtr assemblerPtr;

    DavidsonEigenSolver(AssemblerPtr&& assemblerPtr, PetscScalar initialGuess = 1.0);

    DavidsonEigenSolver(const std::string& filePath)
        : DavidsonEigenSolver([&filePath]() {
            HighFive::File file(filePath, HighFive::File::ReadOnly);
            HighFive::Group group = file.getGroup("CMFD_CoarseMesh");
            return std::make_unique<CSRMatrixAssembler>(group);
        }()) {}

    ~DavidsonEigenSolver();

    PetscErrorCode solve(size_t maxIterations = 100);
};

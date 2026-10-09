#pragma once

#include "petscvec_kokkos.hpp"
#include "petscmat_kokkos.hpp"
#include "petscksp.h"

#include "PetscMatrixAssembler.hpp"

struct PetscEigenSolver
{
    using AssemblerPtr = std::unique_ptr<MatrixAssemblerInterface>;

    KSP ksp; // Linear solver context
    PC pc;   // Preconditioner context

    // tol is the convergence tolerance for the eigenvalue problem (|delta keff| < tol).
    static constexpr double tol = 1.e-7;
    // Relative tolerance of the KSP (linear) solver. Because each inner solve starts from the previous
    // flux (nonzero initial guess), an inner tolerance equal to tol lets the inner solve stop after
    // ~0 iterations once the flux stops changing by more than tol, which stalls the power iteration
    // short of the true keff (7x7 case: |keff - gold| = 1.2e-7). Solving 10x tighter than the outer
    // tolerance avoids that and still needs fewer total inner iterations than zero-guess solves at tol.
    static constexpr double kspRtol = tol / 10;
    double keff;
    std::vector<double> keffHistory;

    Vec pastFission, currentFlux;

    AssemblerPtr assemblerPtr;

    PetscEigenSolver() = default;

    // The assemblerPtr must be moved with std::move
    PetscEigenSolver(AssemblerPtr&& assemblerPtr, PetscScalar initialGuess = 1.0);

    // Constructor that initializes the solver with a file path and assumes the CSRMatrixAssembler type.
    PetscEigenSolver(const std::string& filePath)
        : PetscEigenSolver([&filePath]() {
            HighFive::File file(filePath, HighFive::File::ReadOnly);
            HighFive::Group group = file.getGroup("CMFD_CoarseMesh");
            return std::make_unique<CSRMatrixAssembler>(group);
        }()) {}

    ~PetscEigenSolver();

    PetscErrorCode solve(size_t maxIterations = 100);

    // Relative residual ||M phi - F phi / keff|| / ||M phi|| of the current eigenpair
    PetscErrorCode eigenResidual(PetscReal &relres);
};
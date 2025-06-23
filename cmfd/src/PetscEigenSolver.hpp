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
    const double tol = 1.e-7;
    double keff = 1.0;
    std::vector<double> keffHistory;

    Vec pastFission, currentFission, currentFlux;

    AssemblerPtr assemblerPtr;

    PetscEigenSolver() = default;

    // The assemblerPtr must be moved with std::move
    PetscEigenSolver(AssemblerPtr&& assemblerPtr, PCType pcType = PCJACOBI);

    ~PetscEigenSolver();

    PetscErrorCode solve(size_t maxIterations = 100);
};
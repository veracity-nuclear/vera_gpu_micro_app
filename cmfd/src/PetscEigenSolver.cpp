#include "PetscEigenSolver.hpp"

PetscEigenSolver::PetscEigenSolver(AssemblerPtr&& _assemblerPtr, PetscScalar initialGuess)
    : assemblerPtr(std::move(_assemblerPtr))
{
    const Mat& A = assemblerPtr->getM();

    PetscFunctionBeginUser;

    PetscCallCXXAbort(PETSC_COMM_SELF, KSPCreate(PETSC_COMM_WORLD, &ksp));
    PetscCallCXXAbort(PETSC_COMM_SELF, KSPSetOperators(ksp, A, A));
    PetscCallCXXAbort(PETSC_COMM_SELF, KSPSetTolerances(ksp, tol, PETSC_CURRENT, PETSC_CURRENT, PETSC_CURRENT));

    // These defaults can be override by command line options.
    PetscCallCXXAbort(PETSC_COMM_SELF, KSPGetPC(ksp, &pc));
    PetscCallCXXAbort(PETSC_COMM_SELF, PCSetType(pc, PCJACOBI));

    PetscCallCXXAbort(PETSC_COMM_SELF, KSPSetFromOptions(ksp));

    PetscCallCXXAbort(PETSC_COMM_SELF, assemblerPtr->instantiateVec(pastFission));
    PetscCallCXXAbort(PETSC_COMM_SELF, assemblerPtr->instantiateVec(currentFlux));

    // Initialize flux vector to the initial guess at all values
    VecSet(currentFlux, initialGuess);

}

PetscEigenSolver::~PetscEigenSolver() {
    if (ksp) {
        PetscCallCXXAbort(PETSC_COMM_SELF, KSPDestroy(&ksp));
    }
    PetscCallCXXAbort(PETSC_COMM_SELF, VecDestroy(&pastFission));
    PetscCallCXXAbort(PETSC_COMM_SELF, VecDestroy(&currentFlux));
    // Do not destroy currentFission since it is managed by assemblerPtr

}

PetscErrorCode PetscEigenSolver::solve(size_t maxIterations) {
    PetscFunctionBeginUser;

    PetscScalar currentCurrentFissionDot, pastCurrentFissionDot;
    keff = 1.0; // Initial guess for keff
    keffHistory.clear();
    keffHistory.reserve(maxIterations);

    // typename Vec is a pointer. The assembler retains ownership and data are not copied.
    Vec currentFission = assemblerPtr->getFissionSource(currentFlux);

    for (size_t iter = 0; iter < maxIterations; ++iter)
    {
        if constexpr(true) // future logging condition
        {
            PetscPrintf(PETSC_COMM_WORLD, "Iteration %zu: keff = %g\n", iter, keff);
        }

        // Update pastFission with the current fission source (pastFission = currentFission)
        PetscCall(VecCopy(currentFission, pastFission));
        keffHistory.push_back(keff);

        // Fission vector becomes the RHS b vector when we divide by keff.
        PetscCall(VecScale(currentFission, 1/keff));

        // Solves Ax=b with KSPSolve(ksp, b, x) where A is set in the constructor.
        // Updates currentFlux with the solution x.
        // TODO: We could actually store currentFlux in currentFission to save memory,
        // but currentFlux is likely used later, so we need to be careful.
        PetscCall(KSPSolve(ksp, currentFission, currentFlux));

        // This updates currentFission since it is a pointer.
        assemblerPtr->getFissionSource(currentFlux);

        // TODO: Use VecMDot to do these dot products at the same time
        PetscCall(VecDot(currentFission, currentFission, &currentCurrentFissionDot));
        PetscCall(VecDot(pastFission, currentFission, &pastCurrentFissionDot));

        keff *= currentCurrentFissionDot / pastCurrentFissionDot;
        assert(keff > 0.0 && "keff should always be positive");

        if (iter > 0 && std::abs(keff - keffHistory[iter-1]) < tol) {
            // Convergence achieved
            keffHistory.push_back(keff);
            return PETSC_SUCCESS;
        }
    }

    keffHistory.push_back(keff);
    std::cerr << "Warning: Maximum iterations reached without convergence." << std::endl;

    PetscFunctionReturn(PETSC_SUCCESS);
}
#include "PetscEigenSolver.hpp"

PetscEigenSolver::PetscEigenSolver(AssemblerPtr&& _assemblerPtr, PCType pcType)
    : assemblerPtr(std::move(_assemblerPtr))
{
    const Mat& A = assemblerPtr->getM();

    PetscFunctionBeginUser;
    KSPCreate(PETSC_COMM_WORLD, &ksp);
    KSPSetOperators(ksp, A, A);
    KSPSetTolerances(ksp, tol, PETSC_CURRENT, PETSC_CURRENT, PETSC_CURRENT);
    KSPGetPC(ksp, &pc);
    PCSetType(pc, pcType);

    assemblerPtr->instantiateVec(pastFission);
    assemblerPtr->instantiateVec(currentFlux);
    VecSet(currentFlux, 1.0); // Initialize flux vector to zero
}

PetscEigenSolver::~PetscEigenSolver() {
    if (ksp) {
        KSPDestroy(&ksp);
    }
    VecDestroy(&pastFission);
    VecDestroy(&currentFlux);
    // Do not destroy currentFission since it is managed by assemblerPtr

}

PetscErrorCode PetscEigenSolver::solve(size_t maxIterations) {
    PetscFunctionBeginUser;
    PetscScalar currentCurrentFissionDot, pastCurrentFissionDot;

    currentFission = assemblerPtr->getFissionSource(currentFlux);

    {
        // Update pastFission with the current fission source (pastFission = currentFission)
        PetscCall(VecCopy(currentFission, pastFission));

        PetscCall(VecScale(currentFission, 1/keff));

        // Solves Ax=b with KSPSolve(ksp, b, x) where A is set in the constructor.
        // Updates currentFlux with the solution x.
        // TODO: We could actually store currentFlux in currentFission to save memory,
        // but currentFlux is likely used later.
        PetscCall(KSPSolve(ksp, currentFission, currentFlux));

        // This updates currentFission since it is a pointer.
        assemblerPtr->getFissionSource(currentFlux);

        // TODO: Use VecMDot to do these dot products at the same time
        PetscCall(VecDot(currentFission, currentFission, &currentCurrentFissionDot));
        PetscCall(VecDot(pastFission, currentFission, &pastCurrentFissionDot));

        keff *= currentCurrentFissionDot / pastCurrentFissionDot;
    }

    return PETSC_SUCCESS;
}
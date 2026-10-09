#include "PetscEigenSolver.hpp"
#include "KokkosBlockGS.hpp"

PetscEigenSolver::PetscEigenSolver(AssemblerPtr&& _assemblerPtr, PetscScalar initialGuess)
    : assemblerPtr(std::move(_assemblerPtr))
{
    const Mat& A = assemblerPtr->getM();

    PetscFunctionBeginUser;

    PetscCallCXXAbort(PETSC_COMM_SELF, KSPCreate(PETSC_COMM_WORLD, &ksp));
    PetscCallCXXAbort(PETSC_COMM_SELF, KSPSetOperators(ksp, A, A));
    PetscCallCXXAbort(PETSC_COMM_SELF, KSPSetTolerances(ksp, kspRtol, PETSC_CURRENT, PETSC_CURRENT, PETSC_CURRENT));

    // Each inner solve starts from the flux of the previous power iteration, which is
    // already close to the solution, instead of from zero (~30-50% fewer KSP iterations).
    // This is why the KSP uses the tighter kspRtol rather than tol (see the header).
    PetscCallCXXAbort(PETSC_COMM_SELF, KSPSetInitialGuessNonzero(ksp, PETSC_TRUE));

    // Default preconditioner: one Kokkos block red-black Gauss-Seidel sweep (see KokkosBlockGS.hpp).
    // Measured on mini-core against point Jacobi: GMRES matvecs 1122 -> 516, KSPSolve 2.3x faster on
    // the host, and the sweep is two kernel launches with no reductions, so it maps well to GPUs.
    // Point-block Jacobi (-pc_type pbjacobi) gave no reduction at all: the convergence limiter is the
    // spatial leakage coupling, not the within-cell group coupling. All of this can be overridden on
    // the command line (-pc_type jacobi, -ksp_type richardson for a pure block-SOR inner iteration, ...).
    PetscCallCXXAbort(PETSC_COMM_SELF, PCRegisterKokkosBlockGS());
    PetscCallCXXAbort(PETSC_COMM_SELF, KSPGetPC(ksp, &pc));
    PetscCallCXXAbort(PETSC_COMM_SELF, PCSetType(pc, "kokkosbgs"));
    // Two CCSI-relaxed sweeps per application: same total sweeps as one sweep per GMRES iteration but
    // half the Krylov iterations (mini-core: 241 vs 516 matvecs), i.e. half the reductions on a GPU.
    PetscCallCXXAbort(PETSC_COMM_SELF, PetscOptionsSetValue(nullptr, "-pc_kokkosbgs_sweeps", "2"));

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

    // dots[0] = currentFission . currentFission, dots[1] = pastFission . currentFission
    PetscScalar dots[2];
    keff = 1.0; // Initial guess for keff
    keffHistory.clear();
    keffHistory.reserve(maxIterations);

    // typename Vec is a pointer. The assembler retains ownership and data are not copied.
    Vec currentFission = assemblerPtr->getFissionSource(currentFlux);
    const Vec dotVecs[2] = {currentFission, pastFission};

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

        // Both dot products in one pass over currentFission (and one reduction).
        PetscCall(VecMDot(currentFission, 2, dotVecs, dots));

        keff *= dots[0] / dots[1];
        assert(keff > 0.0 && "keff should always be positive");

        if (iter > 0 && std::abs(keff - keffHistory[iter-1]) < tol) {
            // Convergence achieved
            keffHistory.push_back(keff);
            PetscFunctionReturn(PETSC_SUCCESS);
        }
    }

    keffHistory.push_back(keff);
    std::cerr << "Warning: Maximum iterations reached without convergence." << std::endl;

    PetscFunctionReturn(PETSC_SUCCESS);
}
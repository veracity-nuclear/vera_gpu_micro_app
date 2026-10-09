#include "DavidsonEigenSolver.hpp"
#include "KokkosBlockGS.hpp"

#include <petscblaslapack.h>
#include <cmath>
#include <algorithm>

DavidsonEigenSolver::DavidsonEigenSolver(AssemblerPtr&& _assemblerPtr, PetscScalar initialGuess)
    : assemblerPtr(std::move(_assemblerPtr))
{
    const Mat& M = assemblerPtr->getM();

    PetscFunctionBeginUser;
    PetscCallCXXAbort(PETSC_COMM_SELF, PetscOptionsGetInt(nullptr, nullptr, "-davidson_max_subspace", &maxSubspace, nullptr));
    PetscCallCXXAbort(PETSC_COMM_SELF, PetscOptionsGetReal(nullptr, nullptr, "-davidson_rtol", &rtol, nullptr));

    // Preconditioner approximating M (never the shifted operator, which approaches singularity)
    PetscCallCXXAbort(PETSC_COMM_SELF, PCRegisterKokkosBlockGS());
    PetscCallCXXAbort(PETSC_COMM_SELF, PCCreate(PETSC_COMM_SELF, &pc));
    PetscCallCXXAbort(PETSC_COMM_SELF, PCSetOptionsPrefix(pc, "davidson_"));
    PetscCallCXXAbort(PETSC_COMM_SELF, PCSetType(pc, "kokkosbgs"));
    PetscBool set;
    PetscCallCXXAbort(PETSC_COMM_SELF, PetscOptionsHasName(nullptr, "davidson_", "-pc_kokkosbgs_sweeps", &set));
    // 6 block RB-GS sweeps per correction: mini-core converges in 16 Davidson iterations (3 sweeps: 26,
    // 10 sweeps: 11); more sweeps trade cheap, reduction-free sweep kernels for fewer iterations.
    if (!set) PetscCallCXXAbort(PETSC_COMM_SELF, PetscOptionsSetValue(nullptr, "-davidson_pc_kokkosbgs_sweeps", "6"));
    PetscCallCXXAbort(PETSC_COMM_SELF, PCSetOperators(pc, M, M));
    PetscCallCXXAbort(PETSC_COMM_SELF, PCSetFromOptions(pc));
    PetscCallCXXAbort(PETSC_COMM_SELF, PCSetUp(pc));

    PetscCallCXXAbort(PETSC_COMM_SELF, assemblerPtr->instantiateVec(currentFlux));
    PetscCallCXXAbort(PETSC_COMM_SELF, VecSet(currentFlux, initialGuess));
}

DavidsonEigenSolver::~DavidsonEigenSolver()
{
    if (pc) PetscCallCXXAbort(PETSC_COMM_SELF, PCDestroy(&pc));
    if (currentFlux) PetscCallCXXAbort(PETSC_COMM_SELF, VecDestroy(&currentFlux));
}

PetscErrorCode DavidsonEigenSolver::solve(size_t maxIterations)
{
    PetscFunctionBeginUser;
    const Mat M = assemblerPtr->getM();
    const PetscInt mmax = maxSubspace;

    // Subspace basis V, and the images MV = M V and FV = F V kept alongside so that Ritz vectors and
    // residuals are linear combinations (no extra operator applications).
    Vec *V, *MV, *FV, r, t, phi, Mphi, Fphi;
    PetscCall(VecDuplicateVecs(currentFlux, mmax, &V));
    PetscCall(VecDuplicateVecs(currentFlux, mmax, &MV));
    PetscCall(VecDuplicateVecs(currentFlux, mmax, &FV));
    PetscCall(VecDuplicate(currentFlux, &r));
    PetscCall(VecDuplicate(currentFlux, &t));
    PetscCall(VecDuplicate(currentFlux, &phi));
    PetscCall(VecDuplicate(currentFlux, &Mphi));
    PetscCall(VecDuplicate(currentFlux, &Fphi));

    // Projected matrices A = V^T M V and B = V^T F V, column-major, leading dimension mmax
    std::vector<PetscScalar> A(mmax * mmax, 0.0), B(mmax * mmax, 0.0);
    std::vector<PetscScalar> Awork(mmax * mmax), Cwork(mmax * mmax), wr(mmax), wi(mmax), vr(mmax * mmax), y(mmax);
    std::vector<PetscBLASInt> ipiv(mmax);
    std::vector<PetscScalar> dots(mmax), coeffs(mmax);

    auto appendVector = [&](Vec v, PetscInt m) -> PetscErrorCode {
        // v is already orthonormal to V[0..m). Store it and its images, extend A and B.
        PetscFunctionBeginUser;
        PetscCall(VecCopy(v, V[m]));
        PetscCall(MatMult(M, V[m], MV[m]));
        Vec Fv = assemblerPtr->getFissionSource(V[m]);
        PetscCall(VecCopy(Fv, FV[m]));
        // column m: A(i,m) = V_i . (M v_m);  row m: A(m,i) = v_m . (M V_i)
        PetscCall(VecMDot(MV[m], m + 1, V, dots.data()));
        for (PetscInt i = 0; i <= m; ++i) A[i + m * mmax] = dots[i];
        PetscCall(VecMDot(V[m], m + 1, MV, dots.data()));
        for (PetscInt i = 0; i <= m; ++i) A[m + i * mmax] = dots[i];
        PetscCall(VecMDot(FV[m], m + 1, V, dots.data()));
        for (PetscInt i = 0; i <= m; ++i) B[i + m * mmax] = dots[i];
        PetscCall(VecMDot(V[m], m + 1, FV, dots.data()));
        for (PetscInt i = 0; i <= m; ++i) B[m + i * mmax] = dots[i];
        PetscFunctionReturn(PETSC_SUCCESS);
    };

    // Start from the current flux
    PetscCall(VecCopy(currentFlux, t));
    PetscCall(VecNormalize(t, nullptr));
    PetscInt m = 0;
    PetscCall(appendVector(t, m)); m = 1;

    keffHistory.clear();
    residualHistory.clear();
    keffHistory.reserve(maxIterations);
    double keffPrev = keff;

    PetscBool verbose = PETSC_FALSE;
    PetscCall(PetscOptionsGetBool(nullptr, nullptr, "-davidson_verbose", &verbose, nullptr));
    PetscPrintf(PETSC_COMM_WORLD, "Generalized Davidson: subspace %d, preconditioner from M\n", (int)mmax);
    for (size_t iter = 0; iter < maxIterations; ++iter)
    {
        // ---- Rayleigh-Ritz: largest eigenvalue k of A^-1 B (k = keff, theta = 1/k) ----
        {
            PetscBLASInt bm = m, lda = mmax, info = 0, lwork = 8 * mmax;
            std::vector<PetscScalar> work(lwork);
            std::copy(A.begin(), A.end(), Awork.begin());
            std::copy(B.begin(), B.end(), Cwork.begin()); // C <- A^-1 B (solve A X = B)
            PetscCallBLAS("LAPACKgesv", LAPACKgesv_(&bm, &bm, Awork.data(), &lda, ipiv.data(), Cwork.data(), &lda, &info));
            PetscCheck(info == 0, PETSC_COMM_SELF, PETSC_ERR_LIB, "LAPACK gesv failed (info %d) in Rayleigh-Ritz", (int)info);
            PetscCallBLAS("LAPACKgeev", LAPACKgeev_("N", "V", &bm, Cwork.data(), &lda, wr.data(), wi.data(), nullptr, &lda, vr.data(), &lda, work.data(), &lwork, &info));
            PetscCheck(info == 0, PETSC_COMM_SELF, PETSC_ERR_LIB, "LAPACK geev failed (info %d) in Rayleigh-Ritz", (int)info);
            PetscInt best = 0;
            for (PetscInt i = 1; i < m; ++i) if (wr[i] > wr[best]) best = i;
            keff = wr[best];
            for (PetscInt i = 0; i < m; ++i) y[i] = vr[i + best * mmax];
            if (verbose) {
                std::vector<PetscInt> order(m);
                for (PetscInt i = 0; i < m; ++i) order[i] = i;
                std::sort(order.begin(), order.end(), [&](PetscInt a, PetscInt b) { return wr[a] > wr[b]; });
                PetscPrintf(PETSC_COMM_WORLD, "   Ritz values:");
                for (PetscInt i = 0; i < std::min<PetscInt>(m, 6); ++i)
                    PetscPrintf(PETSC_COMM_WORLD, " %.9f%+.1ei", wr[order[i]], wi[order[i]]);
                PetscPrintf(PETSC_COMM_WORLD, "\n");
            }
        }
        const double theta = 1.0 / keff;

        // Ritz vector and its images: phi = V y, M phi = (MV) y, F phi = (FV) y
        PetscCall(VecSet(phi, 0.0));  PetscCall(VecMAXPY(phi, m, y.data(), V));
        PetscCall(VecSet(Mphi, 0.0)); PetscCall(VecMAXPY(Mphi, m, y.data(), MV));
        PetscCall(VecSet(Fphi, 0.0)); PetscCall(VecMAXPY(Fphi, m, y.data(), FV));

        // Residual r = M phi - theta F phi
        PetscCall(VecWAXPY(r, -theta, Fphi, Mphi));
        PetscReal rnorm, mnorm;
        PetscCall(VecNorm(r, NORM_2, &rnorm));
        PetscCall(VecNorm(Mphi, NORM_2, &mnorm));
        const double relres = rnorm / mnorm;

        keffHistory.push_back(keff);
        residualHistory.push_back(relres);
        PetscPrintf(PETSC_COMM_WORLD, "Iteration %zu: keff = %.9f  |dk| = %.2e  ||r||/||M phi|| = %.2e  (m = %d)\n",
                    iter, keff, std::abs(keff - keffPrev), relres, (int)m);

        if (verbose) {
            PetscReal fmin, fmax;
            PetscCall(VecMin(phi, nullptr, &fmin));
            PetscCall(VecMax(phi, nullptr, &fmax));
            PetscPrintf(PETSC_COMM_WORLD, "   Ritz vector range: [%.3e, %.3e]\n", fmin, fmax);
        }
        if (iter > 0 && std::abs(keff - keffPrev) < tol && relres < rtol) {
            PetscCall(VecCopy(phi, currentFlux));
            break;
        }
        keffPrev = keff;

        // ---- Correction t = P^-1 r, orthonormalized against V (two Gram-Schmidt passes) ----
        PetscCall(PCApply(pc, r, t));
        for (int pass = 0; pass < 2; ++pass) {
            PetscCall(VecMDot(t, m, V, dots.data()));
            for (PetscInt i = 0; i < m; ++i) coeffs[i] = -dots[i];
            PetscCall(VecMAXPY(t, m, coeffs.data(), V));
        }
        PetscReal tnorm;
        PetscCall(VecNormalize(t, &tnorm));

        // ---- Restart from the Ritz vector when the subspace is full ----
        if (m == mmax) {
            PetscCall(VecCopy(phi, V[0]));
            PetscReal pnorm;
            PetscCall(VecNormalize(V[0], &pnorm));
            PetscCall(VecCopy(Mphi, MV[0])); PetscCall(VecScale(MV[0], 1.0 / pnorm));
            PetscCall(VecCopy(Fphi, FV[0])); PetscCall(VecScale(FV[0], 1.0 / pnorm));
            PetscScalar a00, b00;
            PetscCall(VecDot(MV[0], V[0], &a00));
            PetscCall(VecDot(FV[0], V[0], &b00));
            std::fill(A.begin(), A.end(), 0.0); std::fill(B.begin(), B.end(), 0.0);
            A[0] = a00; B[0] = b00;
            m = 1;
            // re-orthogonalize t against the new basis
            PetscCall(VecMDot(t, m, V, dots.data()));
            coeffs[0] = -dots[0];
            PetscCall(VecMAXPY(t, m, coeffs.data(), V));
            PetscCall(VecNormalize(t, &tnorm));
        }
        if (tnorm == 0.0) {
            PetscPrintf(PETSC_COMM_WORLD, "Davidson: correction vanished, stopping\n");
            PetscCall(VecCopy(phi, currentFlux));
            break;
        }
        PetscCall(appendVector(t, m)); ++m;

        if (iter + 1 == maxIterations) {
            PetscCall(VecCopy(phi, currentFlux));
            std::cerr << "Warning: Maximum iterations reached without convergence." << std::endl;
        }
    }

    PetscCall(VecDestroyVecs(mmax, &V));
    PetscCall(VecDestroyVecs(mmax, &MV));
    PetscCall(VecDestroyVecs(mmax, &FV));
    PetscCall(VecDestroy(&r)); PetscCall(VecDestroy(&t)); PetscCall(VecDestroy(&phi));
    PetscCall(VecDestroy(&Mphi)); PetscCall(VecDestroy(&Fphi));
    PetscFunctionReturn(PETSC_SUCCESS);
}

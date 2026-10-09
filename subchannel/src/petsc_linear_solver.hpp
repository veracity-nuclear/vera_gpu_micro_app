#pragma once

#include <Kokkos_Core.hpp>
#include "petscvec_kokkos.hpp"
#include "petscmat_kokkos.hpp"
#include "petscksp.h"

/**
 * @brief PETSc-Kokkos linear solver for subchannel thermal-hydraulics
 *
 * This solver takes Kokkos Views (dfdg matrix and f0 vector) from the ANTS functor
 * and solves the linear system using PETSc's Kokkos-integrated solvers.
 *
 * @tparam ExecutionSpace The Kokkos execution space (Serial, OpenMP, CUDA, etc.)
 */
template <typename ExecutionSpace = Kokkos::DefaultExecutionSpace>
class PetscLinearSolver {
public:
    using MemorySpace = typename ExecutionSpace::memory_space;
    using View1D = Kokkos::View<double*, MemorySpace>;
    using View2D = Kokkos::View<double**, MemorySpace>;

    /**
     * @brief Construct a new PETSc Linear Solver
     *
     * @param size Size of the linear system (number of equations/unknowns)
     * @param tolerance Convergence tolerance for the iterative solver
     */
    PetscLinearSolver(PetscInt size, PetscReal tolerance = 1.0e-6)
        : n(size), tol(tolerance), initialized(false), ksp(nullptr), pc(nullptr), A(nullptr), x(nullptr), b(nullptr)
    {
        // Defer initialization until first solve() call to avoid MPI_INIT issues
    }

    /**
     * @brief Destroy the PETSc Linear Solver and clean up resources
     */
    ~PetscLinearSolver() {
        if (initialized) {
            PetscCallCXXAbort(PETSC_COMM_SELF, cleanup());
        }
    }

    /**
     * @brief Solve the linear system Ax = b using PETSc-Kokkos integration
     *
     * This method takes the Jacobian matrix (dfdg) and residual vector (f0) from
     * the ANTS functor, converts them to PETSc format, solves the system, and
     * writes the solution back to f0.
     *
     * @param dfdg The Jacobian matrix (A in Ax=b) - modified during solve
     * @param f0 The residual vector (b in Ax=b) - overwritten with solution x
     * @return PetscErrorCode Success or error code
     */
    template<typename ViewType2D, typename ViewType1D>
    PetscErrorCode solve(ViewType2D& dfdg, ViewType1D& f0);

    /**
     * @brief Solve Ax = b where A is given directly in CSR form (no dense matrix is ever formed)
     *
     * @param row_ptr  CSR row pointer (size n+1), int-valued view
     * @param col_idx  CSR column indices (size nnz), int-valued view
     * @param values   CSR values (size nnz)
     * @param f0       Right-hand side; overwritten with the solution x
     */
    template<typename RowView, typename ColView, typename ValView, typename ViewType1D>
    PetscErrorCode solve_csr(const RowView& row_ptr, const ColView& col_idx, const ValView& values, ViewType1D& f0);

    /**
     * @brief Convert Kokkos 2D View to PETSc Mat using Kokkos integration
     *
     * @param kokkos_mat Input Kokkos matrix view
     * @return PetscErrorCode
     */
    template<typename ViewType>
    PetscErrorCode kokkosMatToPetsc(const ViewType& kokkos_mat);

    /**
     * @brief Convert Kokkos 1D View to PETSc Vec using Kokkos integration
     *
     * @param kokkos_vec Input Kokkos vector view
     * @param petsc_vec Output PETSc vector
     * @return PetscErrorCode
     */
    template<typename ViewType>
    PetscErrorCode kokkosVecToPetsc(const ViewType& kokkos_vec, Vec& petsc_vec);

    /**
     * @brief Convert PETSc Vec to Kokkos 1D View using Kokkos integration
     *
     * @param petsc_vec Input PETSc vector
     * @param kokkos_vec Output Kokkos vector view
     * @return PetscErrorCode
     */
    template<typename ViewType>
    PetscErrorCode petscVecToKokkos(Vec& petsc_vec, ViewType& kokkos_vec);

private:
    PetscInt n;              // Size of the system
    PetscReal tol;           // Convergence tolerance
    bool initialized;        // Initialization flag

    KSP ksp;                 // PETSc Krylov Subspace (linear) solver context
    PC pc;                   // Preconditioner context
    Mat A;                   // PETSc matrix
    Vec x, b;                // PETSc solution and RHS vectors

    // CSR storage handed to PETSc by solve_csr; kept alive for the lifetime of A
    Kokkos::View<PetscInt*> csr_row_ptr, csr_col_idx;
    Kokkos::View<PetscScalar*> csr_values;

    /**
     * @brief Initialize PETSc objects
     * @return PetscErrorCode
     */
    PetscErrorCode initialize();

    /**
     * @brief Clean up PETSc objects
     * @return PetscErrorCode
     */
    PetscErrorCode cleanup();
};

// solve_csr is a member template over arbitrary (compatible) view types, so it is defined here.
template <typename ExecutionSpace>
template <typename RowView, typename ColView, typename ValView, typename ViewType1D>
PetscErrorCode PetscLinearSolver<ExecutionSpace>::solve_csr(const RowView& row_ptr, const ColView& col_idx, const ValView& values, ViewType1D& f0) {
    static_assert(sizeof(typename RowView::non_const_value_type) == sizeof(PetscInt), "CSR index views must match PetscInt");
    PetscFunctionBeginUser;

    if (!initialized) {
        PetscCall(initialize());
    }

    const PetscInt nnz = static_cast<PetscInt>(col_idx.extent(0));

    // PETSc's Kokkos matrix constructors want views in the default memory space; (re)allocate only on size change
    if (csr_row_ptr.extent(0) != row_ptr.extent(0)) {
        csr_row_ptr = Kokkos::View<PetscInt*>("csr_row_ptr", row_ptr.extent(0));
        csr_col_idx = Kokkos::View<PetscInt*>("csr_col_idx", nnz);
        csr_values = Kokkos::View<PetscScalar*>("csr_values", nnz);
    }
    Kokkos::deep_copy(csr_row_ptr, row_ptr);
    Kokkos::deep_copy(csr_col_idx, col_idx);
    Kokkos::deep_copy(csr_values, values);

    if (A) {
        PetscCall(MatDestroy(&A));
    }
    if constexpr (Kokkos::SpaceAccessibility<Kokkos::HostSpace, MemorySpace>::accessible) {
        // On host execution spaces a plain SeqAIJ matrix gives PETSc's native sparse LU, which is far
        // cheaper than the Kokkos-Kernels triangular solves for systems of this size.
        PetscCall(MatCreateSeqAIJWithArrays(PETSC_COMM_SELF, n, n, csr_row_ptr.data(), csr_col_idx.data(), csr_values.data(), &A));
    } else {
        PetscCall(MatCreateSeqAIJKokkosWithKokkosViews(PETSC_COMM_SELF, n, n, csr_row_ptr, csr_col_idx, csr_values, &A));
    }
    PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
    PetscCall(KSPSetOperators(ksp, A, A));

    PetscCall(kokkosVecToPetsc(f0, b));
    PetscCall(KSPSolve(ksp, b, x));

    KSPConvergedReason reason;
    PetscCall(KSPGetConvergedReason(ksp, &reason));
    if (reason < 0) {
        PetscInt its;
        PetscReal rnorm;
        PetscCall(KSPGetIterationNumber(ksp, &its));
        PetscCall(KSPGetResidualNorm(ksp, &rnorm));
        PetscPrintf(PETSC_COMM_SELF, "WARNING: Linear solver diverged. Reason: %d, Iterations: %d, Residual: %e\n",
                    reason, its, rnorm);
    }

    PetscCall(petscVecToKokkos(x, f0));
    PetscFunctionReturn(PETSC_SUCCESS);
}


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

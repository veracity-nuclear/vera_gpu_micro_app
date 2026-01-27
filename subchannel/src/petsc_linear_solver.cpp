#include "petsc_linear_solver.hpp"

template <typename ExecutionSpace>
PetscErrorCode PetscLinearSolver<ExecutionSpace>::initialize() {
    PetscFunctionBeginUser;

    // Check if PETSc is already initialized
    PetscBool petsc_initialized;
    PetscInitialized(&petsc_initialized);

    if (!petsc_initialized) {
        // Initialize PETSc if not already done
        // Note: This should ideally be done by the application, but we do it here as a fallback
        int argc = 0;
        char **argv = nullptr;
        PetscCall(PetscInitialize(&argc, &argv, nullptr, nullptr));
    }

    // Note: Matrix A is now created in kokkosMatToPetsc with the CSR structure
    // This avoids pre-allocating a dense matrix structure

    // Create vectors with Kokkos type (use COMM_SELF for sequential)
    PetscCall(VecCreate(PETSC_COMM_SELF, &x));
    PetscCall(VecSetSizes(x, PETSC_DECIDE, n));
    PetscCall(VecSetType(x, VECKOKKOS));
    PetscCall(VecSetFromOptions(x));

    PetscCall(VecCreate(PETSC_COMM_SELF, &b));
    PetscCall(VecSetSizes(b, PETSC_DECIDE, n));
    PetscCall(VecSetType(b, VECKOKKOS));
    PetscCall(VecSetFromOptions(b));

    // Create KSP solver context
    PetscCall(KSPCreate(PETSC_COMM_SELF, &ksp));
    PetscCall(KSPSetOperators(ksp, A, A));

    // Use BiCGStab - more robust for non-symmetric thermal-hydraulics matrices than GMRES
    PetscCall(KSPSetType(ksp, KSPBCGS));

    PetscCall(KSPGetPC(ksp, &pc));
    // Use simple Jacobi preconditioner (diagonal scaling) - always works, no fill
    PetscCall(PCSetType(pc, PCLU));

    // Relaxed tolerances for nonlinear iteration: rtol=1e-2, atol=1e-4, max_iter=1000
    // These are sufficient since we're inside an outer Newton loop
    PetscCall(KSPSetTolerances(ksp, 1e-8, 1e-12, PETSC_DEFAULT, 10000));

    // Allow command line options to override defaults
    PetscCall(KSPSetFromOptions(ksp));

    initialized = true;
    PetscFunctionReturn(PETSC_SUCCESS);
}

template <typename ExecutionSpace>
PetscErrorCode PetscLinearSolver<ExecutionSpace>::cleanup() {
    PetscFunctionBeginUser;

    if (ksp) {
        PetscCall(KSPDestroy(&ksp));
    }
    if (A) {
        PetscCall(MatDestroy(&A));
    }
    if (x) {
        PetscCall(VecDestroy(&x));
    }
    if (b) {
        PetscCall(VecDestroy(&b));
    }

    initialized = false;
    PetscFunctionReturn(PETSC_SUCCESS);
}

template <typename ExecutionSpace>
template <typename ViewType>
PetscErrorCode PetscLinearSolver<ExecutionSpace>::kokkosMatToPetsc(const ViewType& kokkos_mat) {
    PetscFunctionBeginUser;

    const PetscInt mat_size = n;  // Capture n as local variable for lambda

    // Analyze sparsity and convert to CSR format ON HOST (serial) to avoid race conditions
    // Copy matrix to host for deterministic serial processing
    auto h_kokkos_mat = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), kokkos_mat);

    // Count non-zeros per row (serial on host for deterministic results)
    std::vector<PetscInt> nnz_per_row(mat_size);
    PetscInt total_nnz = 0;

    for (PetscInt i = 0; i < mat_size; ++i) {
        PetscInt count = 0;
        for (PetscInt j = 0; j < mat_size; ++j) {
            if (std::abs(h_kokkos_mat(i, j)) > PETSC_MACHINE_EPSILON) {
                count++;
            }
        }
        nnz_per_row[i] = count;
        total_nnz += count;
    }

    // Build CSR structure on host (serial)
    std::vector<PetscInt> h_row_offsets(mat_size + 1);
    std::vector<PetscInt> h_col_indices(total_nnz);
    std::vector<PetscScalar> h_values(total_nnz);

    h_row_offsets[0] = 0;
    for (PetscInt i = 0; i < mat_size; ++i) {
        h_row_offsets[i + 1] = h_row_offsets[i] + nnz_per_row[i];

        PetscInt offset = h_row_offsets[i];
        PetscInt count = 0;
        for (PetscInt j = 0; j < mat_size; ++j) {
            PetscScalar val = h_kokkos_mat(i, j);
            if (std::abs(val) > PETSC_MACHINE_EPSILON) {
                h_col_indices[offset + count] = j;
                h_values[offset + count] = val;
                count++;
            }
        }
    }

    // Create device views and copy CSR data
    Kokkos::View<PetscInt*, MemorySpace> row_offsets_temp("row_offsets_temp", mat_size + 1);
    Kokkos::View<PetscInt*, MemorySpace> col_indices_temp("col_indices_temp", total_nnz);
    Kokkos::View<PetscScalar*, MemorySpace> values_temp("values_temp", total_nnz);

    auto h_row_offsets_view = Kokkos::create_mirror_view(row_offsets_temp);
    auto h_col_indices_view = Kokkos::create_mirror_view(col_indices_temp);
    auto h_values_view = Kokkos::create_mirror_view(values_temp);

    for (PetscInt i = 0; i <= mat_size; ++i) h_row_offsets_view(i) = h_row_offsets[i];
    for (PetscInt i = 0; i < total_nnz; ++i) {
        h_col_indices_view(i) = h_col_indices[i];
        h_values_view(i) = h_values[i];
    }

    Kokkos::deep_copy(row_offsets_temp, h_row_offsets_view);
    Kokkos::deep_copy(col_indices_temp, h_col_indices_view);
    Kokkos::deep_copy(values_temp, h_values_view);

    // Create PETSc-compatible views (without explicit memory space)
    // NOTE: PETSc requires views WITHOUT explicit memory space in template
    Kokkos::View<PetscInt*> row_offsets("row_offsets", mat_size + 1);
    Kokkos::View<PetscInt*> col_indices("col_indices", total_nnz);
    Kokkos::View<PetscScalar*> values("values", total_nnz);

    // Copy data to PETSc-compatible views
    Kokkos::deep_copy(row_offsets, row_offsets_temp);
    Kokkos::deep_copy(col_indices, col_indices_temp);
    Kokkos::deep_copy(values, values_temp);

    // Destroy old matrix if it exists
    if (A) {
        PetscCall(MatDestroy(&A));
    }

    // Check for empty rows (would indicate singular matrix)
    auto h_row_offs_check = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), row_offsets);
    for (PetscInt i = 0; i < mat_size; ++i) {
        if (h_row_offs_check(i+1) == h_row_offs_check(i)) {
            PetscPrintf(PETSC_COMM_SELF, "WARNING: Row %d has no non-zero entries!\n", i);
        }
    }

    // Create matrix using CSR format with Kokkos views
    // Pass views directly like CMFD does - the API accepts views with MemorySpace
    PetscCall(MatCreateSeqAIJKokkosWithKokkosViews(PETSC_COMM_SELF, mat_size, mat_size,
                                                     row_offsets, col_indices, values, &A));

    // Assemble the matrix
    PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

    PetscFunctionReturn(PETSC_SUCCESS);
}

template <typename ExecutionSpace>
template <typename ViewType>
PetscErrorCode PetscLinearSolver<ExecutionSpace>::kokkosVecToPetsc(const ViewType& kokkos_vec, Vec& petsc_vec) {
    PetscFunctionBeginUser;

    // Get Kokkos view from PETSc Vec
    View1D petsc_view;
    PetscCall(VecGetKokkosView(petsc_vec, &petsc_view));

    // Deep copy from input Kokkos view to PETSc's Kokkos view
    Kokkos::deep_copy(petsc_view, kokkos_vec);

    // Restore the view
    PetscCall(VecRestoreKokkosView(petsc_vec, &petsc_view));

    PetscFunctionReturn(PETSC_SUCCESS);
}

template <typename ExecutionSpace>
template <typename ViewType>
PetscErrorCode PetscLinearSolver<ExecutionSpace>::petscVecToKokkos(Vec& petsc_vec, ViewType& kokkos_vec) {
    PetscFunctionBeginUser;

    // Get Kokkos view from PETSc Vec (const view for reading)
    Kokkos::View<const PetscScalar*, MemorySpace> petsc_view;
    PetscCall(VecGetKokkosView(petsc_vec, &petsc_view));

    // Deep copy from PETSc's Kokkos view to output Kokkos view
    Kokkos::deep_copy(kokkos_vec, petsc_view);

    // Restore the view
    PetscCall(VecRestoreKokkosView(petsc_vec, &petsc_view));

    PetscFunctionReturn(PETSC_SUCCESS);
}

template <typename ExecutionSpace>
template <typename ViewType2D, typename ViewType1D>
PetscErrorCode PetscLinearSolver<ExecutionSpace>::solve(ViewType2D& dfdg, ViewType1D& f0) {
    PetscFunctionBeginUser;

    // Lazy initialization: only initialize PETSc objects on first solve() call
    if (!initialized) {
        PetscCall(initialize());
    }

    // Convert Kokkos matrix to PETSc format
    PetscCall(kokkosMatToPetsc(dfdg));

    // Update KSP operators if matrix was just created
    if (ksp) {
        PetscCall(KSPSetOperators(ksp, A, A));
    }

    // Convert Kokkos RHS vector to PETSc format
    PetscCall(kokkosVecToPetsc(f0, b));

    // Solve the linear system Ax = b
    PetscCall(KSPSolve(ksp, b, x));

    // Check convergence
    KSPConvergedReason reason;
    PetscCall(KSPGetConvergedReason(ksp, &reason));

    PetscInt its;
    PetscReal rnorm;
    PetscCall(KSPGetIterationNumber(ksp, &its));
    PetscCall(KSPGetResidualNorm(ksp, &rnorm));

    if (reason < 0) {
        PetscPrintf(PETSC_COMM_SELF, "WARNING: Linear solver diverged. Reason: %d, Iterations: %d, Residual: %e\n",
                    reason, its, rnorm);
    }
    // Removed verbose convergence logging - only warn on failure

    // Convert PETSc solution back to Kokkos format (overwrites f0)
    PetscCall(petscVecToKokkos(x, f0));

    PetscFunctionReturn(PETSC_SUCCESS);
}

// Explicit template instantiations for common execution spaces
template class PetscLinearSolver<Kokkos::Serial>;

// Explicit instantiation of template methods for Serial execution space
template PetscErrorCode PetscLinearSolver<Kokkos::Serial>::solve<
    Kokkos::View<double**, Kokkos::Serial>,
    Kokkos::View<double*, Kokkos::Serial>
>(Kokkos::View<double**, Kokkos::Serial>&, Kokkos::View<double*, Kokkos::Serial>&);

template PetscErrorCode PetscLinearSolver<Kokkos::Serial>::kokkosMatToPetsc<
    Kokkos::View<double**, Kokkos::Serial>
>(const Kokkos::View<double**, Kokkos::Serial>&);

template PetscErrorCode PetscLinearSolver<Kokkos::Serial>::kokkosVecToPetsc<
    Kokkos::View<double*, Kokkos::Serial>
>(const Kokkos::View<double*, Kokkos::Serial>&, Vec&);

template PetscErrorCode PetscLinearSolver<Kokkos::Serial>::petscVecToKokkos<
    Kokkos::View<double*, Kokkos::Serial>
>(Vec&, Kokkos::View<double*, Kokkos::Serial>&);

#ifdef KOKKOS_ENABLE_OPENMP
template class PetscLinearSolver<Kokkos::OpenMP>;

// Explicit instantiation of template methods for OpenMP execution space
template PetscErrorCode PetscLinearSolver<Kokkos::OpenMP>::solve<
    Kokkos::View<double**, Kokkos::OpenMP>,
    Kokkos::View<double*, Kokkos::OpenMP>
>(Kokkos::View<double**, Kokkos::OpenMP>&, Kokkos::View<double*, Kokkos::OpenMP>&);

template PetscErrorCode PetscLinearSolver<Kokkos::OpenMP>::kokkosMatToPetsc<
    Kokkos::View<double**, Kokkos::OpenMP>
>(const Kokkos::View<double**, Kokkos::OpenMP>&);

template PetscErrorCode PetscLinearSolver<Kokkos::OpenMP>::kokkosVecToPetsc<
    Kokkos::View<double*, Kokkos::OpenMP>
>(const Kokkos::View<double*, Kokkos::OpenMP>&, Vec&);

template PetscErrorCode PetscLinearSolver<Kokkos::OpenMP>::petscVecToKokkos<
    Kokkos::View<double*, Kokkos::OpenMP>
>(Vec&, Kokkos::View<double*, Kokkos::OpenMP>&);
#endif

#ifdef KOKKOS_ENABLE_CUDA
template class PetscLinearSolver<Kokkos::Cuda>;

// Explicit instantiation of template methods for Cuda execution space
template PetscErrorCode PetscLinearSolver<Kokkos::Cuda>::solve<
    Kokkos::View<double**, Kokkos::Cuda>,
    Kokkos::View<double*, Kokkos::Cuda>
>(Kokkos::View<double**, Kokkos::Cuda>&, Kokkos::View<double*, Kokkos::Cuda>&);

template PetscErrorCode PetscLinearSolver<Kokkos::Cuda>::kokkosMatToPetsc<
    Kokkos::View<double**, Kokkos::Cuda>
>(const Kokkos::View<double**, Kokkos::Cuda>&);

template PetscErrorCode PetscLinearSolver<Kokkos::Cuda>::kokkosVecToPetsc<
    Kokkos::View<double*, Kokkos::Cuda>
>(const Kokkos::View<double*, Kokkos::Cuda>&, Vec&);

template PetscErrorCode PetscLinearSolver<Kokkos::Cuda>::petscVecToKokkos<
    Kokkos::View<double*, Kokkos::Cuda>
>(Vec&, Kokkos::View<double*, Kokkos::Cuda>&);
#endif

#pragma once

#include <Kokkos_Core.hpp>
#include "petscvec_kokkos.hpp"
#include "petscmat_kokkos.hpp"
#include "petscksp.h"
#include <iostream>

/**
 * @brief Channel-space (Schur complement) solver for the crossflow Newton step.
 *
 * The analytic crossflow Jacobian has the form J = F + B C (+ a small two-phase donor-quality term):
 *   F  diagonal over surfaces (secant-regularized gap friction slope),
 *   B  surface-to-channel incidence (nsurf x nchan, +1 on from_node, -1 on to_node),
 *   C  channel pressure response to each surface's crossflow (nchan x nsurf, two entries per column).
 * B^T-type systems like this have the cycle space of the surface graph as a near-null space (dimension
 * nsurf - nchan + 1), which makes every incomplete preconditioner on J fail. The Woodbury identity turns
 * the solve with P = F + B C into a well-posed channel-space system
 *     N y = C F^-1 r,   N = I + C F^-1 B   (nchan x nchan, 5-point stencil, no structural null space)
 *     x   = F^-1 (r - B y)
 * which is used here as the preconditioner of a flexible GMRES iteration on the exact J, so the term the
 * preconditioner neglects is still accounted for.
 *
 * N = I - L with L a weighted graph Laplacian over the channels (weights = pressure coupling / friction
 * slope, up to ~1e8), so N has one eigenvalue +1 (uniform pressure) and all others large and negative.
 * Since B 1 = 0, the uniform component of y does not change x, so N may be replaced by N + 1 a^T for any a,
 * and the uniform component never has to be solved for. The inner Krylov iteration therefore runs on
 *     O = -N + (1/n) 1 1^T = L - I + (1/n) 1 1^T,   which is exactly singular on constants,
 * with the constant null space attached (PETSc projects it out of the right-hand side and the iterates) and
 * the definite preconditioning matrix L + I = -N + 2 I, which differs from O on non-uniform modes by a
 * relative 2/lambda only. Without the projection every approximate preconditioner has to resolve the
 * uniform mode, whose eigenvalue is 1e6-1e8 times smaller than the rest, and Jacobi, ILU and AMG all stall.
 *
 * Two inner modes:
 *   direct    (-crossflow_inner_direct, default on host memory): N itself is factored once per Newton step
 *             with PETSc's sparse LU (a 5-point channel matrix, far cheaper than the LU of J); the outer
 *             FGMRES then needs 1-2 iterations.
 *   iterative (default on device memory): BCGS on the projected O preconditioned from L + I
 *             (-crossflow_inner_pc_type jacobi|sor|gamg ...). Note that x = F^-1 (r - B y) amplifies inner
 *             errors by 1/F, so the inner tolerance trades off against outer FGMRES iterations. Every piece is a Kokkos kernel or a PETSc Kokkos
 * operation, so the same path serves host and device builds. The inner solve on N takes PETSc options with
 * the prefix -crossflow_inner_ (default BCGS + Jacobi), the outer one with -crossflow_.
 */
template <typename ExecutionSpace = Kokkos::DefaultExecutionSpace>
class SchurCrossflowSolver {
public:
    using MemorySpace = typename ExecutionSpace::memory_space;
    using DeviceSpace = typename Kokkos::DefaultExecutionSpace::memory_space; // what PETSc's Kokkos objects live in
    using ScalarView  = Kokkos::View<PetscScalar*>;
    using ScalarView2 = Kokkos::View<PetscScalar**>;
    using IntView     = Kokkos::View<int*>;
    using IntView2    = Kokkos::View<int**>;
    using PIntView    = Kokkos::View<PetscInt*>;
    // PETSc's Vec accessors deduce the memory space from the view type, so it must be spelled out
    using VecView      = Kokkos::View<PetscScalar*, DeviceSpace>;
    using ConstVecView = Kokkos::View<const PetscScalar*, DeviceSpace>;

    struct Stats {
        PetscInt solves = 0, outer_its = 0, inner_its = 0, pc_applies = 0;
    };

    SchurCrossflowSolver(PetscInt nsurf, PetscInt nchan, PetscReal rtol = 1.0e-8, PetscReal abstol = 1.0e-10)
        : nsurf(nsurf), nchan(nchan), rtol(rtol), abstol(abstol) {}

    ~SchurCrossflowSolver() {
        if (initialized) PetscCallCXXAbort(PETSC_COMM_SELF, cleanup());
    }
    SchurCrossflowSolver(const SchurCrossflowSolver&) = delete;
    SchurCrossflowSolver& operator=(const SchurCrossflowSolver&) = delete;

    const Stats& stats() const { return ctx.stats; }

    /**
     * Solve J x = f0 and overwrite f0 with x.
     * @param jac_row_ptr/jac_col_idx/jac_values  CSR of the exact Jacobian J (surface space)
     * @param schur_row_ptr/schur_col_idx          CSR structure of N (channel space, from Geometry)
     * @param schur_pos (ns,0..3), schur_diag (ch) positions in N's CSR (from Geometry)
     * @param surfaces  Surface view (from_node, to_node)
     * @param F, C      per-surface factors of J = F + B C (schur_terms kernel)
     */
    template <typename RowView, typename ColView, typename ValView, typename PosView, typename SurfView,
              typename FView, typename CView, typename RhsView>
    PetscErrorCode solve(const RowView& jac_row_ptr, const ColView& jac_col_idx, const ValView& jac_values,
                         const RowView& schur_row_ptr, const ColView& schur_col_idx, const PosView& schur_pos,
                         const ColView& schur_diag, const SurfView& surfaces,
                         const FView& F, const CView& C, RhsView& f0) {
        PetscFunctionBeginUser;
        if (!initialized) PetscCall(initialize(jac_row_ptr, jac_col_idx, schur_row_ptr, schur_col_idx, schur_pos, schur_diag, surfaces));

        // --- bring the per-step data into PETSc's memory space
        Kokkos::deep_copy(J_v, jac_values);
        Kokkos::deep_copy(ctx.F, F);
        Kokkos::deep_copy(ctx.C, C);

        // --- assemble N = I + C F^-1 B on the fixed channel pattern
        {
            auto Nv = N_v; auto diag = N_diag; auto pos = N_pos; auto Fv = ctx.F; auto Cv = ctx.C;
            Kokkos::deep_copy(Nv, 0.0);
            Kokkos::parallel_for("Schur::N identity", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, nchan),
                KOKKOS_LAMBDA(const PetscInt ch) { Nv(diag(ch)) = 1.0; });
            // direct mode stores N = I + C F^-1 B; iterative mode stores Npc = -N + 2 I = I - C F^-1 B
            const double sgn = direct ? 1.0 : -1.0;
            Kokkos::parallel_for("Schur::N assemble", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, nsurf),
                KOKKOS_LAMBDA(const PetscInt ns) {
                    const double wi = sgn * Cv(ns, 0) / Fv(ns); // C(from, ns) / F(ns)
                    const double wj = sgn * Cv(ns, 1) / Fv(ns); // C(to,   ns) / F(ns)
                    // (C F^-1 B)(ch, ch') += C(ch, ns) / F(ns) * B(ns, ch'),  B(ns, from) = +1, B(ns, to) = -1
                    Kokkos::atomic_add(&Nv(pos(ns, 0)),  wi);
                    Kokkos::atomic_add(&Nv(pos(ns, 1)), -wi);
                    Kokkos::atomic_add(&Nv(pos(ns, 2)),  wj);
                    Kokkos::atomic_add(&Nv(pos(ns, 3)), -wj);
                });
            Kokkos::fence();
        }

        // --- (re)create the PETSc operators on the updated value arrays
        if (J) PetscCall(MatDestroy(&J));
        if (ctx.Npc) PetscCall(MatDestroy(&ctx.Npc));
        PetscCall(MatCreateSeqAIJKokkosWithKokkosViews(PETSC_COMM_SELF, nsurf, nsurf, J_rp, J_ci, J_v, &J));
        if constexpr (Kokkos::SpaceAccessibility<Kokkos::HostSpace, DeviceSpace>::accessible) {
            if (direct) // plain SeqAIJ: PETSc's native sparse LU is much faster than the Kokkos-Kernels triangular solves
                PetscCall(MatCreateSeqAIJWithArrays(PETSC_COMM_SELF, nchan, nchan, N_rp.data(), N_ci.data(), N_v.data(), &ctx.Npc));
            else
                PetscCall(MatCreateSeqAIJKokkosWithKokkosViews(PETSC_COMM_SELF, nchan, nchan, N_rp, N_ci, N_v, &ctx.Npc));
        } else {
            PetscCall(MatCreateSeqAIJKokkosWithKokkosViews(PETSC_COMM_SELF, nchan, nchan, N_rp, N_ci, N_v, &ctx.Npc));
        }
        PetscCall(MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY)); PetscCall(MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY));
        PetscCall(MatAssemblyBegin(ctx.Npc, MAT_FINAL_ASSEMBLY)); PetscCall(MatAssemblyEnd(ctx.Npc, MAT_FINAL_ASSEMBLY));
        if (direct) PetscCall(KSPSetOperators(ctx.inner, ctx.Npc, ctx.Npc));
        else        PetscCall(KSPSetOperators(ctx.inner, O, ctx.Npc));
        PetscCall(KSPSetOperators(outer, J, J));

        // --- rhs in, solution out
        {
            VecView bv; PetscCall(VecGetKokkosView(b, &bv)); Kokkos::deep_copy(bv, f0); PetscCall(VecRestoreKokkosView(b, &bv));
        }
        PetscCall(KSPSolve(outer, b, x));
        KSPConvergedReason reason; PetscCall(KSPGetConvergedReason(outer, &reason));
        PetscInt its; PetscCall(KSPGetIterationNumber(outer, &its));
        ctx.stats.solves++; ctx.stats.outer_its += its;
        if (reason < 0) {
            PetscReal rnorm; PetscCall(KSPGetResidualNorm(outer, &rnorm));
            PetscPrintf(PETSC_COMM_SELF, "WARNING: Schur crossflow solver diverged. Reason: %d, Iterations: %d, Residual: %e\n", (int)reason, (int)its, (double)rnorm);
        }
        {
            ConstVecView xv; PetscCall(VecGetKokkosView(x, &xv)); Kokkos::deep_copy(f0, xv); PetscCall(VecRestoreKokkosView(x, &xv));
        }
        PetscFunctionReturn(PETSC_SUCCESS);
    }

private:
    // Everything the shell preconditioner needs, in PETSc's memory space
    struct Context {
        PetscInt nsurf = 0, nchan = 0;
        ScalarView  F;      // nsurf
        ScalarView2 C;      // nsurf x 2
        IntView from, to;   // nsurf
        Vec g = nullptr, y = nullptr;
        KSP inner = nullptr;
        Mat Npc = nullptr;   // direct: N; iterative: L + I (definite preconditioning matrix of the inner solve)
        bool direct = false;
        Stats stats;
    };

    // O x = (L - I + (1/n) 1 1^T) x = Npc x - 2 x + (1/n) (1^T x) 1
    static PetscErrorCode shell_mult(Mat O, Vec x, Vec y) {
        PetscFunctionBeginUser;
        Context* c; PetscCall(MatShellGetContext(O, &c));
        PetscCall(MatMult(c->Npc, x, y));
        PetscCall(VecAXPY(y, -2.0, x));
        PetscScalar s; PetscCall(VecSum(x, &s));
        PetscCall(VecShift(y, s / (PetscScalar)c->nchan));
        PetscFunctionReturn(PETSC_SUCCESS);
    }
    // diag(O) = diag(Npc) - 2 + 1/n  (needed by smoothers / eigenvalue estimates that look at the operator)
    static PetscErrorCode shell_get_diagonal(Mat O, Vec d) {
        PetscFunctionBeginUser;
        Context* c; PetscCall(MatShellGetContext(O, &c));
        PetscCall(MatGetDiagonal(c->Npc, d));
        PetscCall(VecShift(d, -2.0 + 1.0 / (PetscScalar)c->nchan));
        PetscFunctionReturn(PETSC_SUCCESS);
    }

    static PetscErrorCode pc_apply(PC pc, Vec r, Vec xo) {
        PetscFunctionBeginUser;
        Context* c; PetscCall(PCShellGetContext(pc, &c));
        const PetscInt nsurf = c->nsurf;
        const double sgn = c->direct ? 1.0 : -1.0;
        auto F = c->F; auto C = c->C; auto from = c->from; auto to = c->to;
        ConstVecView rv; PetscCall(VecGetKokkosView(r, &rv));
        // g = C F^-1 r  (scatter over surfaces)
        {
            VecView gv; PetscCall(VecGetKokkosViewWrite(c->g, &gv));
            Kokkos::deep_copy(gv, 0.0);
            Kokkos::parallel_for("Schur::PC gather", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, nsurf),
                KOKKOS_LAMBDA(const PetscInt ns) {
                    const double u = sgn * rv(ns) / F(ns);   // rhs: N y = C F^-1 r (direct) or O y = -(C F^-1 r)
                    Kokkos::atomic_add(&gv(from(ns)), C(ns, 0) * u);
                    Kokkos::atomic_add(&gv(to(ns)),   C(ns, 1) * u);
                });
            Kokkos::fence();
            PetscCall(VecRestoreKokkosViewWrite(c->g, &gv));
        }
        // O y = -g  (y defined up to a uniform shift, which x does not see)
        PetscCall(KSPSolve(c->inner, c->g, c->y));
        {
            PetscInt its; PetscCall(KSPGetIterationNumber(c->inner, &its));
            c->stats.inner_its += its; c->stats.pc_applies++;
            KSPConvergedReason reason; PetscCall(KSPGetConvergedReason(c->inner, &reason));
            if (reason < 0) PetscPrintf(PETSC_COMM_SELF, "WARNING: Schur inner (channel-space) solve diverged. Reason: %d, Iterations: %d\n", (int)reason, (int)its);
        }
        // x = F^-1 (r - B y)
        {
            ConstVecView yv; PetscCall(VecGetKokkosView(c->y, &yv));
            VecView xv; PetscCall(VecGetKokkosViewWrite(xo, &xv));
            Kokkos::parallel_for("Schur::PC scatter", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, nsurf),
                KOKKOS_LAMBDA(const PetscInt ns) {
                    xv(ns) = (rv(ns) - (yv(from(ns)) - yv(to(ns)))) / F(ns);
                });
            Kokkos::fence();
            PetscCall(VecRestoreKokkosViewWrite(xo, &xv));
            PetscCall(VecRestoreKokkosView(c->y, &yv));
        }
        PetscCall(VecRestoreKokkosView(r, &rv));
        PetscFunctionReturn(PETSC_SUCCESS);
    }

    template <typename RowView, typename ColView, typename PosView, typename SurfView>
    PetscErrorCode initialize(const RowView& jac_row_ptr, const ColView& jac_col_idx,
                              const RowView& schur_row_ptr, const ColView& schur_col_idx, const PosView& schur_pos,
                              const ColView& schur_diag, const SurfView& surfaces) {
        static_assert(sizeof(typename RowView::non_const_value_type) == sizeof(PetscInt), "CSR index views must match PetscInt");
        PetscFunctionBeginUser;
        PetscBool petsc_initialized; PetscCall(PetscInitialized(&petsc_initialized));
        if (!petsc_initialized) { int argc = 0; char** argv = nullptr; PetscCall(PetscInitialize(&argc, &argv, nullptr, nullptr)); }

        // structure (fixed by the geometry)
        J_rp = PIntView("schur_J_rp", jac_row_ptr.extent(0)); J_ci = PIntView("schur_J_ci", jac_col_idx.extent(0)); J_v = ScalarView("schur_J_v", jac_col_idx.extent(0));
        N_rp = PIntView("schur_N_rp", schur_row_ptr.extent(0)); N_ci = PIntView("schur_N_ci", schur_col_idx.extent(0)); N_v = ScalarView("schur_N_v", schur_col_idx.extent(0));
        N_pos = IntView2("schur_N_pos", nsurf, 4); N_diag = IntView("schur_N_diag", nchan);
        Kokkos::deep_copy(J_rp, jac_row_ptr); Kokkos::deep_copy(J_ci, jac_col_idx);
        Kokkos::deep_copy(N_rp, schur_row_ptr); Kokkos::deep_copy(N_ci, schur_col_idx);
        Kokkos::deep_copy(N_pos, schur_pos); Kokkos::deep_copy(N_diag, schur_diag);
        ctx.nsurf = nsurf; ctx.nchan = nchan;
        ctx.F = ScalarView("schur_F", nsurf); ctx.C = ScalarView2("schur_C", nsurf, 2);
        ctx.from = IntView("schur_from", nsurf); ctx.to = IntView("schur_to", nsurf);
        {
            auto h_surf = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), surfaces);
            auto h_from = Kokkos::create_mirror_view(ctx.from); auto h_to = Kokkos::create_mirror_view(ctx.to);
            for (PetscInt ns = 0; ns < nsurf; ++ns) { h_from(ns) = static_cast<int>(h_surf(ns).from_node); h_to(ns) = static_cast<int>(h_surf(ns).to_node); }
            Kokkos::deep_copy(ctx.from, h_from); Kokkos::deep_copy(ctx.to, h_to);
        }

        // vectors
        auto make_vec = [&](Vec& v, PetscInt n) -> PetscErrorCode {
            PetscFunctionBeginUser;
            PetscCall(VecCreate(PETSC_COMM_SELF, &v)); PetscCall(VecSetSizes(v, PETSC_DECIDE, n));
            PetscCall(VecSetType(v, VECKOKKOS)); PetscCall(VecSetFromOptions(v));
            PetscFunctionReturn(PETSC_SUCCESS);
        };
        PetscCall(make_vec(b, nsurf)); PetscCall(make_vec(x, nsurf)); PetscCall(make_vec(ctx.g, nchan)); PetscCall(make_vec(ctx.y, nchan));

        // inner mode: direct LU on N (host default) or Krylov on the projected O (device default)
        direct = Kokkos::SpaceAccessibility<Kokkos::HostSpace, DeviceSpace>::accessible;
        { PetscBool flg = direct ? PETSC_TRUE : PETSC_FALSE, set;
          PetscCall(PetscOptionsGetBool(nullptr, nullptr, "-crossflow_inner_direct", &flg, &set));
          direct = flg; }
        ctx.direct = direct;

        // shell operator O = L - I + (1/n) 1 1^T applied through Npc (see class comment)
        PetscCall(MatCreateShell(PETSC_COMM_SELF, nchan, nchan, nchan, nchan, &ctx, &O));
        PetscCall(MatShellSetOperation(O, MATOP_MULT, (void (*)(void))shell_mult));
        PetscCall(MatShellSetOperation(O, MATOP_GET_DIAGONAL, (void (*)(void))shell_get_diagonal));
        PetscCall(MatShellSetVecType(O, VECKOKKOS));
        { MatNullSpace nsp; PetscCall(MatNullSpaceCreate(PETSC_COMM_SELF, PETSC_TRUE, 0, nullptr, &nsp));
          PetscCall(MatSetNullSpace(O, nsp)); PetscCall(MatNullSpaceDestroy(&nsp)); }

        // inner solver: preonly+LU on N (direct) or BCGS + Jacobi on O / Npc (iterative) (both run on the device); -crossflow_inner_* overrides
        PetscCall(KSPCreate(PETSC_COMM_SELF, &ctx.inner));
        PetscCall(KSPSetOptionsPrefix(ctx.inner, "crossflow_inner_"));
        if (direct) {
            PetscCall(KSPSetType(ctx.inner, KSPPREONLY));
            { PC ipc; PetscCall(KSPGetPC(ctx.inner, &ipc)); PetscCall(PCSetType(ipc, PCLU)); }
        } else {
            PetscCall(KSPSetType(ctx.inner, KSPBCGS));
            { PC ipc; PetscCall(KSPGetPC(ctx.inner, &ipc)); PetscCall(PCSetType(ipc, PCJACOBI)); }
            PetscCall(KSPSetTolerances(ctx.inner, 1.0e-8, 1.0e-30, PETSC_DEFAULT, 5000));
        }
        PetscCall(KSPSetFromOptions(ctx.inner));

        // outer: flexible GMRES on the exact J with the Woodbury/Schur shell preconditioner
        PetscCall(KSPCreate(PETSC_COMM_SELF, &outer));
        PetscCall(KSPSetOptionsPrefix(outer, "crossflow_"));
        PetscCall(KSPSetType(outer, KSPFGMRES));
        PetscCall(KSPSetTolerances(outer, rtol, abstol, PETSC_DEFAULT, 200));
        { PC opc; PetscCall(KSPGetPC(outer, &opc)); PetscCall(PCSetType(opc, PCSHELL)); PetscCall(PCShellSetContext(opc, &ctx));
          PetscCall(PCShellSetApply(opc, pc_apply)); PetscCall(PCShellSetName(opc, "channel-space Schur complement")); }
        PetscCall(KSPSetFromOptions(outer));

        initialized = true;
        PetscFunctionReturn(PETSC_SUCCESS);
    }

    PetscErrorCode cleanup() {
        PetscFunctionBeginUser;
        if (outer) PetscCall(KSPDestroy(&outer));
        if (ctx.inner) PetscCall(KSPDestroy(&ctx.inner));
        if (J) PetscCall(MatDestroy(&J));
        if (ctx.Npc) PetscCall(MatDestroy(&ctx.Npc));
        if (O) PetscCall(MatDestroy(&O));
        if (b) PetscCall(VecDestroy(&b));
        if (x) PetscCall(VecDestroy(&x));
        if (ctx.g) PetscCall(VecDestroy(&ctx.g));
        if (ctx.y) PetscCall(VecDestroy(&ctx.y));
        initialized = false;
        PetscFunctionReturn(PETSC_SUCCESS);
    }

    PetscInt nsurf, nchan;
    PetscReal rtol, abstol;
    bool initialized = false;
    bool direct = false;
    Mat J = nullptr, O = nullptr;
    KSP outer = nullptr;
    Vec b = nullptr, x = nullptr;
    Context ctx;
    PIntView J_rp, J_ci, N_rp, N_ci;
    ScalarView J_v, N_v;
    IntView2 N_pos;
    IntView N_diag;
};

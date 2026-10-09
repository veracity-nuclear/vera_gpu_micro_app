// petscvec_kokkos.hpp must precede every other PETSc header in a C++ translation unit
#include "petscvec_kokkos.hpp"
#include "KokkosBlockGS.hpp"

#include <petsc/private/pcimpl.h>
#include <petscmat.h>
#include <Kokkos_Core.hpp>
#include <vector>
#include <cmath>
#include <queue>
#include <algorithm>

namespace {

constexpr int MAX_BS = 16; // largest block (group count) supported by the in-register block inverse

using MemSpace = Kokkos::DefaultExecutionSpace::memory_space;
using ExecSpace = Kokkos::DefaultExecutionSpace;
using IntView = Kokkos::View<PetscInt *, MemSpace>;
using ScalarView = Kokkos::View<PetscScalar *, MemSpace>;
using Scalar2DView = Kokkos::View<PetscScalar **, Kokkos::LayoutRight, MemSpace>;

struct PC_KokkosBGS {
    PetscInt sweeps = 1;
    PetscReal omega = 1.0;       // fixed relaxation factor (used when adaptive is off or ncolors != 2)
    PetscBool adaptive = PETSC_TRUE; // CCSI red/black relaxation factors (Stimpson & Collins, ANS Trans. 115, 2016)
    PetscInt rhoSweeps = 10;     // unrelaxed GS sweeps used to estimate the spectral radius
    PetscReal rhoJ = 0.0;        // estimated Jacobi spectral radius
    std::vector<PetscReal> omegaRed, omegaBlack; // CCSI schedule, one pair per sweep

    PetscInt n = 0, bs = 0, nblocks = 0, ncolors = 0;
    IntView rowPtr, colIdx;      // CSR structure (device copy)
    ScalarView vals;             // CSR values (device copy)
    Scalar2DView dinv;           // (nblocks, bs*bs) inverse of each diagonal block, row-major
    IntView colorBlocks;         // block indices grouped by color
    std::vector<PetscInt> colorPtr; // host offsets into colorBlocks, size ncolors + 1
};

// One relaxed block Gauss-Seidel update of all blocks of one color: y_block <- (1-w) y_block + w D^-1 (x - offblock(A) y)
template <class XView, class YView>
void color_sweep(const PC_KokkosBGS &ctx, PetscInt color, PetscReal omega, const XView &xv, const YView &yv)
{
    const PetscInt bs = ctx.bs;
    auto rowPtr = ctx.rowPtr; auto colIdx = ctx.colIdx; auto vals = ctx.vals; auto dinv = ctx.dinv;
    auto colorBlocks = ctx.colorBlocks;
    const PetscInt begin = ctx.colorPtr[color], end = ctx.colorPtr[color + 1];
    Kokkos::parallel_for("kokkosbgs: color sweep", Kokkos::RangePolicy<ExecSpace>(begin, end),
        KOKKOS_LAMBDA(const PetscInt idx)
    {
        const PetscInt b = colorBlocks(idx);
        PetscScalar rhs[MAX_BS];
        for (int r = 0; r < bs; ++r) {
            const PetscInt row = b * bs + r;
            PetscScalar acc = xv(row);
            for (PetscInt k = rowPtr(row); k < rowPtr(row + 1); ++k) {
                const PetscInt col = colIdx(k);
                if (col / bs != b) acc -= vals(k) * yv(col);
            }
            rhs[r] = acc;
        }
        for (int r = 0; r < bs; ++r) {
            PetscScalar ynew = 0.0;
            for (int cc = 0; cc < bs; ++cc) ynew += dinv(b, r * bs + cc) * rhs[cc];
            const PetscInt row = b * bs + r;
            yv(row) = (1.0 - omega) * yv(row) + omega * ynew;
        }
    });
}

// Relaxation factors for sweep s (0-based) of an application, per color. With CCSI (two colors) the red
// and black half-sweeps get the cyclic Chebyshev factors of Eqs. 2-3 in Stimpson & Collins (2016):
//   w_R^1 = 1, w_B^1 = 1 / (1 - rho_J^2 / 2),
//   w_R^{i+1} = 1 / (1 - rho_J^2 w_B^i / 4), w_B^{i+1} = 1 / (1 - rho_J^2 w_R^{i+1} / 4),
// which converge to the optimal SOR factor 2 / (1 + sqrt(1 - rho_J^2)).
void ensure_omega_schedule(PC_KokkosBGS &ctx, PetscInt sweeps)
{
    if ((PetscInt)ctx.omegaRed.size() >= sweeps) return;
    const PetscReal r2 = ctx.rhoJ * ctx.rhoJ;
    if (ctx.omegaRed.empty()) {
        ctx.omegaRed.push_back(1.0);
        ctx.omegaBlack.push_back(1.0 / (1.0 - 0.5 * r2));
    }
    while ((PetscInt)ctx.omegaRed.size() < sweeps) {
        const PetscReal wR = 1.0 / (1.0 - 0.25 * r2 * ctx.omegaBlack.back());
        const PetscReal wB = 1.0 / (1.0 - 0.25 * r2 * wR);
        ctx.omegaRed.push_back(wR);
        ctx.omegaBlack.push_back(wB);
    }
}

inline bool use_ccsi(const PC_KokkosBGS &ctx) { return ctx.adaptive && ctx.ncolors == 2; }

PetscErrorCode PCSetUp_KokkosBGS(PC pc)
{
    auto *ctx = static_cast<PC_KokkosBGS *>(pc->data);
    Mat A = pc->pmat;

    PetscFunctionBeginUser;
    PetscCall(MatGetSize(A, &ctx->n, nullptr));
    PetscCall(MatGetBlockSize(A, &ctx->bs));
    PetscCheck(ctx->bs >= 1 && ctx->bs <= MAX_BS, PetscObjectComm((PetscObject)A), PETSC_ERR_SUP,
               "kokkosbgs needs a matrix block size in [1, %d] (MatSetBlockSize); got %d", MAX_BS, (int)ctx->bs);
    PetscCheck(ctx->n % ctx->bs == 0, PetscObjectComm((PetscObject)A), PETSC_ERR_ARG_INCOMP, "matrix size not a multiple of the block size");
    ctx->nblocks = ctx->n / ctx->bs;

    // CSR arrays of the (sequential) AIJ matrix; device pointers for Kokkos matrices, host for plain AIJ.
    const PetscInt *i_ptr, *j_ptr;
    PetscScalar *a_ptr;
    PetscMemType mtype;
    PetscCall(MatSeqAIJGetCSRAndMemType(A, &i_ptr, &j_ptr, &a_ptr, &mtype));
    const PetscInt nnz = 0; (void)nnz;
    PetscInt nnzA;
    {
        // row pointer lives wherever the matrix does; read its last entry through a device-aware copy
        Kokkos::View<const PetscInt *, MemSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> ui(i_ptr, ctx->n + 1);
        Kokkos::deep_copy(nnzA, Kokkos::subview(ui, ctx->n));
        ctx->rowPtr = IntView("kokkosbgs rowPtr", ctx->n + 1);
        Kokkos::deep_copy(ctx->rowPtr, ui);
        Kokkos::View<const PetscInt *, MemSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> uj(j_ptr, nnzA);
        Kokkos::View<const PetscScalar *, MemSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> ua(a_ptr, nnzA);
        ctx->colIdx = IntView("kokkosbgs colIdx", nnzA);
        ctx->vals = ScalarView("kokkosbgs vals", nnzA);
        Kokkos::deep_copy(ctx->colIdx, uj);
        Kokkos::deep_copy(ctx->vals, ua);
    }

    // Greedy coloring of the block graph in BFS order (2 colors when the graph is bipartite, e.g. a
    // Cartesian mesh with face couplings only). Done once on the host; the structure never changes.
    {
        auto h_rowPtr = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ctx->rowPtr);
        auto h_colIdx = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ctx->colIdx);
        const PetscInt bs = ctx->bs, nb = ctx->nblocks;
        std::vector<std::vector<PetscInt>> adj(nb);
        for (PetscInt b = 0; b < nb; ++b) {
            for (PetscInt r = b * bs; r < (b + 1) * bs; ++r) {
                for (PetscInt k = h_rowPtr(r); k < h_rowPtr(r + 1); ++k) {
                    const PetscInt nbk = h_colIdx(k) / bs;
                    if (nbk != b) adj[b].push_back(nbk);
                }
            }
            std::sort(adj[b].begin(), adj[b].end());
            adj[b].erase(std::unique(adj[b].begin(), adj[b].end()), adj[b].end());
        }
        std::vector<int> color(nb, -1);
        int ncolors = 0;
        for (PetscInt start = 0; start < nb; ++start) {
            if (color[start] >= 0) continue;
            std::queue<PetscInt> q;
            q.push(start);
            color[start] = 0;
            ncolors = std::max(ncolors, 1);
            while (!q.empty()) {
                const PetscInt b = q.front(); q.pop();
                for (PetscInt nbk : adj[b]) {
                    if (color[nbk] >= 0) continue;
                    // smallest color not used by an already-colored neighbor
                    std::vector<char> used(ncolors + 1, 0);
                    for (PetscInt m : adj[nbk]) if (color[m] >= 0) used[color[m]] = 1;
                    int c = 0;
                    while (c < ncolors && used[c]) ++c;
                    color[nbk] = c;
                    ncolors = std::max(ncolors, c + 1);
                    q.push(nbk);
                }
            }
        }
        ctx->ncolors = ncolors;
        ctx->colorPtr.assign(ncolors + 1, 0);
        for (PetscInt b = 0; b < nb; ++b) ctx->colorPtr[color[b] + 1]++;
        for (int c = 0; c < ncolors; ++c) ctx->colorPtr[c + 1] += ctx->colorPtr[c];
        std::vector<PetscInt> fill(ctx->colorPtr.begin(), ctx->colorPtr.end() - 1);
        auto h_colorBlocks = Kokkos::View<PetscInt *, Kokkos::HostSpace>("kokkosbgs colorBlocks host", nb);
        for (PetscInt b = 0; b < nb; ++b) h_colorBlocks(fill[color[b]]++) = b;
        ctx->colorBlocks = IntView("kokkosbgs colorBlocks", nb);
        Kokkos::deep_copy(ctx->colorBlocks, h_colorBlocks);
        PetscCall(PetscInfo(pc, "kokkosbgs: %d blocks of size %d, %d colors\n", (int)nb, (int)bs, ncolors));
    }

    // Invert every diagonal block (Gauss-Jordan with partial pivoting, in registers/local memory)
    {
        const PetscInt bs = ctx->bs;
        ctx->dinv = Scalar2DView("kokkosbgs dinv", ctx->nblocks, bs * bs);
        auto rowPtr = ctx->rowPtr; auto colIdx = ctx->colIdx; auto vals = ctx->vals; auto dinv = ctx->dinv;
        Kokkos::parallel_for("kokkosbgs: invert diagonal blocks", Kokkos::RangePolicy<ExecSpace>(0, ctx->nblocks),
            KOKKOS_LAMBDA(const PetscInt b)
        {
            PetscScalar M[MAX_BS][MAX_BS], Inv[MAX_BS][MAX_BS];
            for (int r = 0; r < bs; ++r)
                for (int c = 0; c < bs; ++c) { M[r][c] = 0.0; Inv[r][c] = (r == c) ? 1.0 : 0.0; }
            for (int r = 0; r < bs; ++r) {
                const PetscInt row = b * bs + r;
                for (PetscInt k = rowPtr(row); k < rowPtr(row + 1); ++k) {
                    const PetscInt col = colIdx(k);
                    if (col / bs == b) M[r][col - b * bs] += vals(k);
                }
            }
            for (int p = 0; p < bs; ++p) {
                int piv = p;
                for (int r = p + 1; r < bs; ++r) if (Kokkos::abs(M[r][p]) > Kokkos::abs(M[piv][p])) piv = r;
                if (piv != p) for (int c = 0; c < bs; ++c) { PetscScalar t = M[p][c]; M[p][c] = M[piv][c]; M[piv][c] = t; t = Inv[p][c]; Inv[p][c] = Inv[piv][c]; Inv[piv][c] = t; }
                const PetscScalar d = 1.0 / M[p][p];
                for (int c = 0; c < bs; ++c) { M[p][c] *= d; Inv[p][c] *= d; }
                for (int r = 0; r < bs; ++r) {
                    if (r == p) continue;
                    const PetscScalar f = M[r][p];
                    if (f == 0.0) continue;
                    for (int c = 0; c < bs; ++c) { M[r][c] -= f * M[p][c]; Inv[r][c] -= f * Inv[p][c]; }
                }
            }
            for (int r = 0; r < bs; ++r)
                for (int c = 0; c < bs; ++c) dinv(b, r * bs + c) = Inv[r][c];
        });
        Kokkos::fence();
    }

    // Estimate the Jacobi spectral radius for the CCSI relaxation factors. Unrelaxed red-black block GS on
    // the homogeneous system A e = 0 from a constant start decays like rho_GS^k, and for a two-color
    // (consistently ordered) system rho_J = sqrt(rho_GS). As in MPACT the estimate uses the ratio of the
    // last two correction norms after `rhoSweeps` sweeps. Done once per matrix; only setup needs reductions.
    ctx->omegaRed.clear();
    ctx->omegaBlack.clear();
    ctx->rhoJ = 0.0;
    if (use_ccsi(*ctx)) {
        ScalarView e("kokkosbgs rho estimate", ctx->n), zero("kokkosbgs zero rhs", ctx->n);
        Kokkos::deep_copy(e, 1.0);
        Kokkos::deep_copy(zero, 0.0);
        auto norm2 = [&](const ScalarView &v) {
            PetscReal sum = 0.0;
            Kokkos::parallel_reduce("kokkosbgs: norm", Kokkos::RangePolicy<ExecSpace>(0, ctx->n),
                KOKKOS_LAMBDA(const PetscInt i, PetscReal &acc) { acc += v(i) * v(i); }, sum);
            return std::sqrt(sum);
        };
        PetscReal prev = norm2(e), rhoGS = 0.0;
        for (PetscInt s = 0; s < ctx->rhoSweeps; ++s) {
            for (PetscInt c = 0; c < ctx->ncolors; ++c) color_sweep(*ctx, c, 1.0, zero, e);
            const PetscReal cur = norm2(e);
            if (prev > 0.0) rhoGS = cur / prev;
            prev = cur;
        }
        rhoGS = std::min(std::max(rhoGS, 0.0), 0.999999);
        ctx->rhoJ = std::sqrt(rhoGS);
        PetscCall(PetscInfo(pc, "kokkosbgs: estimated rho_GS = %g, rho_J = %g, asymptotic omega = %g\n",
                            (double)rhoGS, (double)ctx->rhoJ, (double)(2.0 / (1.0 + std::sqrt(1.0 - rhoGS)))));
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PCApply_KokkosBGS(PC pc, Vec x, Vec y)
{
    auto *ctx = static_cast<PC_KokkosBGS *>(pc->data);
    PetscFunctionBeginUser;

    Kokkos::View<const PetscScalar *, MemSpace> xv;
    Kokkos::View<PetscScalar *, MemSpace> yv;
    PetscCall(VecGetKokkosView(x, &xv));
    PetscCall(VecGetKokkosViewWrite(y, &yv));

    // y = 0; then `sweeps` block-SOR sweeps on A y = x. A fixed number of sweeps from zero (with a
    // predetermined relaxation schedule) is a fixed linear operator, so this is a valid (non-symmetric)
    // preconditioner for GMRES/BiCGStab. With -ksp_type richardson -ksp_max_it 1 it is the whole inner
    // solve, and the CCSI schedule then runs over all `sweeps` half-sweep pairs as in MPACT.
    Kokkos::deep_copy(yv, 0.0);
    const bool ccsi = use_ccsi(*ctx);
    if (ccsi) ensure_omega_schedule(*ctx, ctx->sweeps);
    for (PetscInt s = 0; s < ctx->sweeps; ++s) {
        for (PetscInt c = 0; c < ctx->ncolors; ++c) {
            const PetscReal omega = ccsi ? (c == 0 ? ctx->omegaRed[s] : ctx->omegaBlack[s]) : ctx->omega;
            color_sweep(*ctx, c, omega, xv, yv);
        }
    }
    Kokkos::fence();

    PetscCall(VecRestoreKokkosViewWrite(y, &yv));
    PetscCall(VecRestoreKokkosView(x, &xv));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PCSetFromOptions_KokkosBGS(PC pc, PetscOptionItems PetscOptionsObject)
{
    auto *ctx = static_cast<PC_KokkosBGS *>(pc->data);
    PetscFunctionBeginUser;
    PetscOptionsHeadBegin(PetscOptionsObject, "Kokkos block red-black Gauss-Seidel options");
    PetscCall(PetscOptionsInt("-pc_kokkosbgs_sweeps", "Number of block SOR sweeps per application", "", ctx->sweeps, &ctx->sweeps, nullptr));
    PetscCall(PetscOptionsBool("-pc_kokkosbgs_adaptive", "Use CCSI red/black relaxation factors from an estimated spectral radius", "", ctx->adaptive, &ctx->adaptive, nullptr));
    PetscCall(PetscOptionsReal("-pc_kokkosbgs_omega", "Fixed SOR relaxation factor (when adaptive is off)", "", ctx->omega, &ctx->omega, nullptr));
    PetscCall(PetscOptionsInt("-pc_kokkosbgs_rho_sweeps", "Unrelaxed sweeps used to estimate the spectral radius", "", ctx->rhoSweeps, &ctx->rhoSweeps, nullptr));
    PetscOptionsHeadEnd();
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PCView_KokkosBGS(PC pc, PetscViewer viewer)
{
    auto *ctx = static_cast<PC_KokkosBGS *>(pc->data);
    PetscBool isascii;
    PetscFunctionBeginUser;
    PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
    if (isascii) {
        if (use_ccsi(*ctx)) {
            ensure_omega_schedule(*ctx, ctx->sweeps);
            PetscCall(PetscViewerASCIIPrintf(viewer, "  Kokkos block red-black Gauss-Seidel: block size %d, %d colors, %d sweeps, CCSI relaxation (rho_J = %g, omega_B^1 = %g, omega_R^%d = %g)\n",
                                             (int)ctx->bs, (int)ctx->ncolors, (int)ctx->sweeps, (double)ctx->rhoJ, (double)ctx->omegaBlack[0], (int)ctx->sweeps, (double)ctx->omegaRed.back()));
        } else {
            PetscCall(PetscViewerASCIIPrintf(viewer, "  Kokkos block red-black Gauss-Seidel: block size %d, %d colors, %d sweeps, fixed omega %g\n",
                                             (int)ctx->bs, (int)ctx->ncolors, (int)ctx->sweeps, (double)ctx->omega));
        }
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PCDestroy_KokkosBGS(PC pc)
{
    PetscFunctionBeginUser;
    delete static_cast<PC_KokkosBGS *>(pc->data);
    pc->data = nullptr;
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PCCreate_KokkosBGS(PC pc)
{
    PetscFunctionBeginUser;
    pc->data = new PC_KokkosBGS();
    pc->ops->apply = PCApply_KokkosBGS;
    pc->ops->setup = PCSetUp_KokkosBGS;
    pc->ops->destroy = PCDestroy_KokkosBGS;
    pc->ops->setfromoptions = PCSetFromOptions_KokkosBGS;
    pc->ops->view = PCView_KokkosBGS;
    PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace

PetscErrorCode PCRegisterKokkosBlockGS()
{
    static bool registered = false;
    PetscFunctionBeginUser;
    if (!registered) {
        PetscCall(PCRegister("kokkosbgs", PCCreate_KokkosBGS));
        registered = true;
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

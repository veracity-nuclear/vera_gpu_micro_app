# CMFD inner solver study: block red-black Gauss-Seidel in Kokkos

Branch `perf/cmfd-kokkos-block-gs` (on top of `perf/benchmark-and-optimize`). Question asked: is it worth
replacing PETSc's inner solve for the CMFD eigenvalue problem with a hand-rolled Kokkos block RB-SOR, with
each cell's group-by-group block solved directly?

## What was built

1. **Well-formed CSR** (`cmfd/src/PetscMatrixAssembler.cpp`, `CSRMatrixAssembler::_assembleM`): rows are
   compacted (the unused leakage slots of boundary cells are dropped) and sorted by column, and the matrix
   block size is set to nGroups. Before this, `-pc_type ilu` aborted and `-pc_type pbjacobi` returned NaN.
2. **`kokkosbgs` PETSc preconditioner** (`cmfd/src/KokkosBlockGS.{hpp,cpp}`, registered with `PCRegister`):
   - setup: copies the CSR to Kokkos views, colors the cell graph greedily in BFS order (2 colors on the
     Cartesian coarse mesh), and inverts every nGroups x nGroups diagonal block once on the device;
   - apply: `sweeps` block-SOR sweeps on A y = x from y = 0, one kernel per color, one thread per cell doing
     a gather of neighbor fluxes and a dense (bs x bs) multiply by the stored inverse. No reductions, no
     host traffic. Options `-pc_kokkosbgs_sweeps`, `-pc_kokkosbgs_omega`.
   - With `-ksp_type richardson` it is the plain stationary block-SOR iteration, and with
     `-ksp_norm_type none -ksp_max_it N` it is "N sweeps per outer iteration with no residual check",
     i.e. exactly what a standalone Kokkos inner solver would do. That is how the PETSc-free variant was
     measured without writing a second driver.
3. **New default**: GMRES + one `kokkosbgs` sweep (`PetscEigenSolver.cpp`). Everything else is unchanged
   and every variant below is selectable from the command line (`PETSC_OPTIONS=...`).

## Measurements

mini-core (722 MB case, 13 power iterations to |dk| < 1e-7, inner rtol 1e-8), 8 threads, PETSc `-log_view`.
"matvec" counts MatMult calls, which for the fixed-sweep variants equals the number of sweeps.

| Inner solver | matvecs | KSPSolve (s) | PETSc total (s) | notes |
|---|---|---|---|---|
| GMRES + point Jacobi (previous default) | 1122 | 0.167 | 0.214 | |
| GMRES + point-block Jacobi (bs 7) | 1117 | 0.226 | 0.275 | no gain: group coupling is not the limiter |
| GMRES + ILU(0) | 372 | 0.356 | 0.458 | host only; slow setup/apply |
| **GMRES + block RB-GS, 1 sweep (new default)** | **516** | **0.072** | **0.175** | |
| GMRES + block RB-GS, 2 sweeps | 324 | 0.133 | 0.217 | fewer matvecs, more time on CPU |
| BiCGStab + block RB-GS, 1 sweep | 584 | 0.080 | 0.119 | |
| Richardson + block SOR, omega 1.0 (pure block GS) | 5053 | 0.799 | 0.806 | |
| Richardson + block SOR, omega 1.5 | 1699 | 0.306 | 0.344 | |
| Richardson + block SOR, omega 1.7 | 745 | 0.173 | 0.189 | |
| Richardson + block SOR, omega 1.8 | 940 | 0.120 | 0.129 | optimum between 1.7 and 1.8 |
| Richardson + block SOR, omega 1.9 | 1986 | 0.229 | 0.301 | |

Inexact power iteration (fixed sweeps per outer, no residual checks; outer still stops at |dk| < 1e-7):

| Inner | sweeps/outer | outer its | total sweeps | PETSc total (s) | gold test (1e-7) |
|---|---|---|---|---|---|
| block SOR omega 1.8 | 5 | 51 | 260 | 0.061 | not checked |
| block SOR omega 1.8 | 10 | 18 | 190 | 0.068 | **fails** for one assembler by 1.9e-7 (false convergence) |
| block SOR omega 1.7 | 10 | ~18 | ~200 | ~0.07 | passes |
| block SOR omega 1.8 | 20 | 12 | 260 | 0.108 | passes |
| GMRES + block RB-GS | 5 iterations | 25 | 156 | 0.074 | passes |
| GMRES + block RB-GS | 10 iterations | 15 | 176 | 0.070 | passes |

7x7 (4 outer iterations) shows the same ordering: Jacobi 282 matvecs, pbjacobi 273, block RB-GS 93 (1 sweep)
and 68 (2 sweeps), Richardson block SOR 247 at omega 1.7 / 279 at 1.8, 10 fixed sweeps per outer 130.

CMFD test suite with the new default: 10/10 pass in 17 s (37 s with Jacobi, 498 s originally).

## Conclusions

- **The block structure alone is not the lever.** Point-block Jacobi, i.e. exactly inverting each cell's 7x7
  block, changes nothing (1117 vs 1122 matvecs). What helps is the spatial Gauss-Seidel coupling between
  cells; the block inverse is just the natural way to do that coupling cell by cell.
- **Block RB-GS is a good preconditioner and a reasonable standalone solver.** As a GMRES preconditioner it
  halves the matvecs and cuts the host KSP time 2.3x. As a bare SOR it needs a tuned omega (1.7-1.8 here,
  problem dependent) and then costs roughly the same number of sweeps as GMRES+Jacobi matvecs, but each
  sweep is two kernel launches with no dot products, which is the GPU-relevant property.
- **Inexact inner solves cut total work another 2-3x** (190-260 sweeps for the whole eigenvalue solve vs 516
  matvecs plus GMRES overhead), but the outer convergence test must then be stronger than |dk| between
  consecutive iterates: a fission-source norm, a dominance-ratio estimate, or simply more sweeps near
  convergence. The 10-sweep/omega-1.8 run stopping 1.9e-7 early is exactly that failure mode.
- **On this CPU none of this matters for wall time**: the whole CMFD solve is 0.07-0.2 s and MPI startup is
  1.5 s. The payoff is on a GPU, where GMRES's per-iteration reductions and PETSc's per-op launches dominate
  at this problem size and a fused two-kernel sweep does not. That needs measuring on a GPU.

## Recommendation for a PETSc-free path

The kernels in `KokkosBlockGS.cpp` (coloring, block inversion, color sweep) are already plain Kokkos; a
standalone solver would wrap them with (a) the CSR views the assembler already builds, (b) a power
iteration using the existing fission-source kernel, (c) N block-SOR sweeps per outer with omega estimated
from a few Jacobi iterations or fixed at ~1.7, and (d) a fission-source-norm convergence test (or Wielandt
shift, which also cuts the outer count). Expect ~200 sweeps for mini-core. Keep the PETSc path behind the
same interface for the multi-rank case; the assembler and fission-source kernels are shared either way.

## Figure

`cmfd_solver_comparison.png` compares original master, the first optimization branch (GMRES + Jacobi with the
fixed fission kernel and nonzero initial guess) and this branch, best of 3 runs each. Sub-0.1 s timings on this
WSL machine jitter by 2-10x between runs (OpenMP wake-up latency), so treat the matvec counts as the reliable
column and the small-case times as indicative only.

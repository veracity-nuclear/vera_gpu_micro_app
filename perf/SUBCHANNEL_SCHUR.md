# Subchannel crossflow solve in channel space (Schur complement)

Branch `perf/subchannel-schur-crossflow`, stacked on `perf/subchannel-analytic-jacobian`. Replaces the sparse LU
of the surface-space crossflow Jacobian, which was 96% of the Watts Bar solve time and single-threaded, with a
solve in channel space that is exact, threaded/device-capable in every kernel, and ~4x faster on the CPU.

## Why the surface-space system cannot be solved iteratively

The crossflow Newton system has one unknown per gap (surface) and one momentum balance per gap. Its Jacobian is

    J = F + B C    (+ a small two-phase donor-quality term)

with F the diagonal of (secant-regularized) gap friction slopes, B the surface-to-channel incidence matrix
(+1 on from_node, -1 on to_node) and C the channel pressure response to each surface's crossflow (two entries
per surface). B C has the cycle space of the surface graph as its null space (dimension nsurf - nchan + 1, i.e.
741 of 1,540 unknowns on mini-core, ~15,900 of 32,096 on Watts Bar): a circulation around a loop of gaps changes
no channel's net source. Only F resists it, and F is 1e6-1e8 times smaller than the pressure coupling. Measured
condition numbers are 1e6-1e8 at every Newton step; ILU(0/1), Jacobi, SOR, ICC, GAMG and even a lagged exact LU
all fail as preconditioners (`SUBCHANNEL_JACOBIAN.md`, "Linear solver study").

## What the branch does

Woodbury identity on P = F + B C:

    P^-1 r = F^-1 r - F^-1 B y,     (I + C F^-1 B) y = C F^-1 r

so one solve with P costs two surface-local kernels and one channel-space solve with

    N = I + C F^-1 B    (nchan x nchan, 5-point stencil, M-matrix structure, no structural null space).

P is used as the preconditioner of a flexible GMRES iteration on the exact J (so the neglected two-phase term is
still accounted for); in single phase P = J and the outer iteration converges in one step. Implementation:

- `Geometry::build_surface_connectivity` also builds the channel-adjacency CSR of N and, per surface, the four
  positions (from,from), (from,to), (to,from), (to,to) it contributes to.
- `ANTSFunctor::operator()(schur_terms, ns)` evaluates F(ns), C(from,ns), C(to,ns) from the same per-channel
  partials (`channel_partials`) the analytic Jacobian uses.
- `SchurCrossflowSolver` (`subchannel/src/schur_crossflow_solver.hpp`): assembles N with atomic scatters,
  implements the Woodbury pre/post kernels as a PETSc `PCSHELL`, FGMRES outer (`-crossflow_*` options) and an
  inner KSP on N (`-crossflow_inner_*`). All vectors/matrices are PETSc Kokkos objects.
- `--crossflow_solver schur|lu` (default `schur`; env `SUBCHANNEL_CROSSFLOW_SOLVER` for the tests). The LU path
  of branch 4 is kept.

Two inner modes:

- **direct** (`-crossflow_inner_direct 1`, default on host memory): PETSc's sparse LU of N once per Newton step.
  N is half the size of J with a 5-point stencil, so its factorization is far cheaper than J's, and the outer
  FGMRES needs 1-2 iterations.
- **iterative** (default on device memory, `-crossflow_inner_direct 0`): BCGS on N preconditioned by Jacobi,
  SOR or GAMG. Two details make this work. (1) N = I - L with L a weighted graph Laplacian, so N has one
  eigenvalue +1 (uniform pressure) and all others negative and up to 1e8 in size. Since B 1 = 0 the uniform
  component of y never enters x, so the inner solve runs on the exactly singular O = -N + (1/n) 1 1^T with the
  constant null space attached (PETSc projects it out) and the definite matrix L + I as the preconditioning
  matrix; without that projection every approximate preconditioner stalls on the uniform mode. (2) Restarted
  GMRES stagnates on this Laplacian-like system; BCGS (or CG-like methods) do not. Also note that
  x = F^-1 (r - B y) amplifies inner errors by 1/F (up to 1e8 at zero crossflow), so the inner tolerance must
  stay tight (1e-8); looser tolerances blow up the outer iteration count.

## Measurements

Crossflow solve summed over the 10 planes (MPI/PETSc startup excluded), best of 3, Release, 4 OpenMP threads
(3x3: serial). "outer" = FGMRES iterations, "inner" = channel-space iterations, both summed over all solves.
Charts: `subchannel_schur_comparison.png` (solve time and iterations) and `subchannel_schur_total_runtime.png`
(total wall clock of the test executable; same session, 4 threads: branch 1 29.4 s / branch 4 9.4 s / direct 3.8 s /
BCGS+GAMG 10.6 s on Watts Bar, mini-core 2.48 / 1.72 / 1.62 / 2.26 s).

| case | branch 4: LU(J) | channel-space, direct LU(N) | channel-space, BCGS + GAMG |
|---|---|---|---|
| 3x3 (12 surf.)             | 0.001 s / 45 its | 0.001 s / 45 its (outer 58)        | 0.048 s / 45 its (outer 93, inner 358) |
| mini-core (1,540 surf.)    | 0.153 s / 37 its | 0.092 s / 37 its (outer 38)        | 0.714 s / 37 its (outer 83, inner 1,044) |
| Watts Bar (32,096 surf.)   | 7.35 s / 63 its  | **1.68 s** / 64 its (outer 84)     | 8.47 s / 63 its (outer 287, inner 7,315) |

Converged exit void / pressure tables are identical in all cases (Watts Bar plane 10 takes one more Newton step
with the Schur solver because the FGMRES stop at 1e-8 lands the residual on the other side of the 1e-8 Newton
tolerance). All 11 subchannel tests pass with `schur` as the default.

Observations:

- **Direct channel-space solve: 4.4x on Watts Bar.** Per Newton step the LU of N (16k unknowns, 5-point) costs
  ~25 ms against ~115 ms for the LU of J (32k unknowns, 7 per row, 17.8x fill), and the outer FGMRES needs 1.6
  iterations. This is the CPU default. It is still a serial factorization, so it does not scale with threads
  (and, like branch 4, slows down when 8 spinning OpenMP threads share 8 cores with it; 4 threads is the sweet
  spot on this laptop).
- **Iterative channel-space solve works but is not yet cheaper on the CPU.** BCGS + GAMG needs ~25 inner
  iterations per preconditioner apply and ~5 outer iterations per Newton step on Watts Bar (the 1/F error
  amplification in the back-substitution forces the tight inner tolerance and the extra outer iterations), i.e.
  ~125 V-cycles per Newton step, plus a GAMG setup per step. On the CPU that matches the old LU; on a GPU each
  V-cycle on 16k unknowns is sub-millisecond and every kernel in the path (N assembly, the two Woodbury kernels,
  BCGS, Jacobi/GAMG apply) runs on the device, which is the point of the branch. Jacobi alone needs ~800 inner
  iterations per apply on Watts Bar (condition ~1e4 from the 128-channel-wide core), SOR ~350.
- **Next levers for the device path.** (1) Raise the zero-crossflow friction floor (`gtol/2 = 5e-4` in F) to a
  physically motivated fraction of the typical crossflow: it bounds 1/F, so the inner solve could be looser and
  the outer iteration count would fall; it only changes the Newton path, not the converged answer. (2) Reuse
  the GAMG hierarchy across Newton steps (the pattern is fixed, the weights change) or replace it with a
  Kokkos-native smoother such as the colored block GS from the CMFD branch. (3) Loosen the outer/Newton
  coupling (inexact Newton) once the inner solve is cheap.


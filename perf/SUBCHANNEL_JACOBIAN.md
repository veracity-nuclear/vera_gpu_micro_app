# Subchannel crossflow solve: analytic Jacobian and Newton void-fraction root

Branch `perf/subchannel-analytic-jacobian`. Builds on the branch-1 rewrite (sparse single-kernel finite-difference
Jacobian) and replaces the two remaining hot spots of the crossflow Newton iteration.

## What was built

1. **Safeguarded Newton root for the drift-flux void fraction** (`solve_void_fraction_impl` in
   `subchannel/src/th.hpp`). f(alpha) and its closed-form derivative are evaluated together; Newton steps
   are taken inside a shrinking bracket with bisection fallback, so it converges in a handful of
   iterations instead of ~50 bisections to the same |f| < 1e-8 / bracket < 1e-8 criteria. The original
   bisection is kept behind `--void_solver bisection`. The planar kernel uses it too.
2. **Analytic Jacobian** (`channel_partials` + `jacobian_columns_analytic`). The residual of surface ns
   depends on gk(ns1) only through the four source terms of ns1's two channels, so the derivative factors
   as (per-channel partials of P and X with respect to the channel's sources) x (trivial, closed-form
   derivatives of the surface source terms with respect to gk). The per-channel partials come from the
   chain rule through flow rates (with their clamps), void fraction (implicit function theorem at the
   converged root, including the B1(Re) dependence of the drift-flux model), quality, and every term of
   the pressure-drop expression. One differentiated evaluation per channel replaces the 14 full channel
   re-evaluations per Jacobian column of the finite-difference scheme (800 vs ~21,000 on mini-core).
3. **Modes**: `--jacobian analytic` (default), `fd` (the branch-1 finite-difference columns), `check`
   (both; prints max |J_an - J_fd|). Environment variables `SUBCHANNEL_JACOBIAN`, `SUBCHANNEL_VOID_SOLVER`
   select the same for the test executables, `SUBCHANNEL_FD_CONSISTENT_X=1` makes the FD columns recompute
   the quality from the perturbed flow rates, and `SUBCHANNEL_DEBUG_CHANNEL=<ij>` prints an FD check of the
   per-channel pressure partials.
4. PETSc is now initialized before the plane loop, so per-plane timings no longer absorb the ~1 s MPI
   startup; the NaN-safe residual maximum no longer reports a NaN state as converged.

## Two things the exact derivative exposed

- **Singular Jacobian at zero crossflow.** The friction term Fns = 0.5 K gk |gk| / rho_m has zero slope
  at gk = 0, so at the first Newton step of every plane (gk starts from the previous plane's value, which
  is zero at the first) the exact Jacobian reduces to the incidence form B^T D B, whose null space is the
  cycle space of the surface graph (rank 8 of 12 on the 3x3 case). PETSc's LU fails on it. The finite-
  difference column never saw this because its secant slope is ~K (|gk| + dG/2) / rho_m, i.e. it was
  regularized by its own step. The analytic diagonal now uses that same secant slope, which is the only
  place it deliberately departs from the exact derivative.
- **The FD Jacobian was not the derivative of the residual.** Its channel re-evaluation read the quality
  from the stored state instead of the perturbed flow rates, so dX/dgk was missing. Against an FD with the
  quality made consistent, the analytic Jacobian agrees to ~2e-6 relative (one Newton step at 5e-3,
  where the FD step crosses a clamp); against the original FD it differs by ~10%. Converged results are
  identical either way (residual definition unchanged); the Newton path is shorter with the exact one.

Validation: per-channel partials match a central finite difference to 7 digits in single- and two-phase
conditions; exit void / pressure-drop tables identical to the baseline on 3x3, 7x7 and mini-core; all 11
subchannel tests pass.

## Measurements

Sum of the per-plane crossflow Newton solve times over the 10 axial planes (MPI/PETSc startup excluded),
best of 3 runs, Release build, 8 OpenMP threads (3x3: serial). "its" = total crossflow Newton iterations.
Chart: `subchannel_jacobian_comparison.png` (script: `tools/subchannel_jacobian_comparison_chart.py`).

| variant                                   | 3x3 (12 surf.)  | mini-core (1,540 surf.) | Watts Bar (32,096 surf.) |
|-------------------------------------------|-----------------|-------------------------|--------------------------|
| original (master)                         | ~0.07 s / 90    | ~20.4 s / 96 (*)        | not run (**)             |
| branch 1: FD Jacobian + bisection         | 0.072 s / 90    | 1.063 s / 96            | 29.2 s / 99              |
| FD Jacobian + Newton void root            | 0.032 s / 102   | 0.462 s / 96            | 16.1 s / 77              |
| analytic Jacobian + bisection             | 0.017 s / 62    | 0.305 s / 78            | 18.4 s / 93              |
| **branch 4: analytic Jacobian + Newton**  | **0.003 s / 45**| **0.126 s / 37**        | **11.6 s / 63**          |

(*) measured as 21.4 s with the ~1.0 s MPI/UCX startup inside the first solved plane; subtracted.
(**) the original finite-difference Jacobian was killed after 86 CPU-minutes without finishing Watts Bar.

Relative to branch 1 the full solve is 8.4x faster on mini-core and 2.5x faster on Watts Bar; relative to the
original it is ~160x faster on mini-core. Converged exit void and pressure tables are identical for every
variant and case.

Where the gains come from:

- **Newton iteration count.** The exact Jacobian shortens the Newton path (mini-core 96 -> 37, Watts Bar
  99 -> 63) because the FD Jacobian was missing the dX/dgk coupling (see above). The void-root change
  alone does not change the iteration count on mini-core; on 3x3 it even adds a few, since the two root
  finders stop at slightly different points inside the 1e-8 bracket.
- **Per-iteration cost.** Per Newton iteration mini-core goes from 11 ms (FD + bisection) to 3.4 ms
  (analytic + Newton); the FD scheme evaluated every channel 14 times per Jacobian build, the analytic
  one once.
- **Watts Bar is linear-solve bound.** Per Newton iteration: FD + bisection 0.295 s, FD + Newton root
  0.209 s, analytic + bisection 0.198 s, analytic + Newton 0.184 s. Jacobian assembly and void-root cost
  together were therefore at most ~0.11 s/iteration; the remaining ~0.18 s is the residual evaluation
  plus PETSc's sparse direct solve of the 32k x 32k crossflow system, which is why the Jacobian work
  buys 2.5x there and 8.4x on mini-core. The next step for large cores is the linear solver (an
  iterative solver with a cheap preconditioner, or reusing the factorization pattern across Newton
  iterations), not the Jacobian.

Not pursued: the semi-analytic table-lookup variant (tabulated derivatives of the closures with
interpolation). With the closed-form chain rule in place the per-channel derivative costs about the
same as one channel evaluation, so a table would save little and add interpolation error.

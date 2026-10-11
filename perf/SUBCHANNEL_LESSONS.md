# Speeding up a subchannel (ANTS/COBRA-style) crossflow solver: what worked, what did not, and why

Portable notes from optimizing the VERA subchannel micro-app (Kokkos + PETSc, but nothing below depends on
Kokkos). The app marches axial planes; on each plane it solves, by Newton iteration, for the lateral
crossflow on every gap between neighbouring channels so that the two channels' pressures balance the gap
friction. Per Newton step: accumulate gap sources into channels, solve each channel's planar
flow/enthalpy/void/quality/pressure, evaluate one momentum residual per gap, build a Jacobian, solve a sparse
linear system, update the crossflows. Results below are for a 16k-channel / 32k-gap full core (Watts Bar)
and an 800-channel / 1,540-gap mini-core on an 8-core laptop.

Overall: mini-core 41 s -> 1.6 s wall; Watts Bar did not finish in 86 CPU-minutes -> 3.8 s wall. Converged
results unchanged at every step. The order below is the order the work was done, and roughly the order of
payoff per effort.

---------------------------------------------------------------------------------------------------------

## 0. Measure before touching anything

- Time the phases of one Newton step separately (sources, planar solve, residual, Jacobian, linear solve,
  update). In the original the Jacobian was 80% of the time; after fixing it the linear solve was 96%. Each
  step of the work changed which phase dominated, and the right fix each time was different.
- Exclude one-time startup from the per-plane numbers. MPI_Init alone was 1.5-1.9 s per process (UCX probing)
  and was hiding inside the first plane's time. Either initialize before the plane loop or subtract it.
- Measure with the thread count that makes sense for the serial parts. Serial sections (sparse LU, AMG setup)
  ran 2x slower with 8 spinning OpenMP threads on 8 cores than with 4. Set the wait policy to passive or use
  fewer threads before concluding anything about scaling.
- Keep an unmodified build around and diff its printed result tables against every change.

## 1. The Jacobian: from O(n^2) dense finite differences to O(n) sparse (16x on mini-core)

The original formed a dense nsurf x nsurf Jacobian by finite differences: for each gap, perturb its crossflow,
re-run the *entire* plane (all channels), difference *all* residuals, then scan the dense matrix on the host
to find nonzeros for the sparse solver.

What to do instead, in increasing order of effort:

1. **Exploit locality.** Perturbing gap ns1 changes only the sources of its two channels, hence only the
   residuals of the gaps touching those two channels (<= 7 on a lattice). A finite-difference column therefore
   needs two channel re-evaluations and <= 7 residuals, not a whole plane. This alone is O(nsurf) per Jacobian.
2. **Fix the sparsity pattern once, from the geometry.** Build the CSR row pointer, column indices and a
   (column, neighbour-slot) -> value-position map at setup. The Jacobian kernel then writes straight into the
   CSR value array; no dense matrix, no host-side nonzero scan, memory 8 n^2 bytes -> ~56 n bytes.
3. **Check that the FD Jacobian is the derivative of the residual.** Ours was not: the channel re-evaluation
   read the quality from the stored state instead of recomputing it from the perturbed flow rates, so dX/dgk
   was missing and the Jacobian was ~10% off. Newton still converged (the residual was right) but along a
   longer path. Always compare an FD Jacobian against a *consistent* re-evaluation of the full map.

## 2. Analytic Jacobian + Newton root for the void fraction (8x on mini-core)

Hand-derived, no autodiff library. The structure that makes it tractable:

    residual(ns) = -dz*aspect * ( P(from) - P(to) - Fns(gk(ns)) )
    dResidual(ns)/dgk(ns1) = sum over channels ch shared by ns and ns1 of
                             B(ns,ch) * B(ns1,ch) * [ dP_ch/dsources . dsources/dgk(ns1) ]  -  dFns/dgk

with B the gap-to-channel incidence (+1 from, -1 to). So the derivative factors into
- **per-gap** closed-form derivatives of the three source terms (liquid, vapour, momentum) with respect to the
  gap's crossflow (a few lines; they depend on the donor channel's quality and mixture velocity), and
- **per-channel** partials of pressure and quality with respect to the channel's accumulated sources, obtained
  by the chain rule through: sources -> flow rates (including the clamps) -> void fraction and quality ->
  every term of the pressure-drop expression (acceleration, wall friction with its Reynolds-dependent friction
  factor and two-phase multiplier).

The void fraction is defined implicitly as the root of a drift-flux relation f(alpha, Wl, Wv) = 0, so use the
implicit function theorem at the converged root: d alpha/dW = -f_W / f_alpha. Since the root finder already
evaluates f_alpha, this is nearly free, and it is exact where differentiating through 50 bisection iterations
would be noisy. Include every parameter that depends on the flow (for us the drift-flux B1(Re) coefficient;
leaving it out gave a 0.15% mismatch that cost Newton iterations).

Replace bisection on the void fraction with a safeguarded Newton (Newton step inside a shrinking bracket,
bisection fallback): same tolerance, ~5 iterations instead of ~50, and the derivative comes out for free.

Validate each stage against a central finite difference of the *same* function (we got 7 digits on the channel
partials, 2e-6 relative on the full Jacobian against a consistent FD). Keep a `--jacobian check` mode.

Result: one differentiated evaluation per channel per Newton step instead of 14 re-evaluations per column, and
the exact derivative cut Newton iterations from 96 to 37 on mini-core (99 -> 63 on Watts Bar).

### The zero-crossflow singularity (you will hit this with an exact Jacobian)

Gap friction is quadratic, Fns = 0.5 K gk|gk|/rho, so its slope K|gk|/rho is zero at zero crossflow. At the
first Newton step of a plane (crossflows start at zero, or at the previous plane's values) the exact Jacobian
reduces to B^T D B, whose null space is the cycle space of the gap graph (see section 3): sparse LU fails with a
zero pivot. A finite-difference Jacobian never shows this because its secant slope K(|gk| + dG/2)/rho is
regularized by its own step. Use the same secant slope on the diagonal of the analytic Jacobian: the residual is
unchanged, so the converged answer is unchanged; only the first steps are a (mildly) modified Newton.

## 3. Why the linear system is hard, and why no off-the-shelf preconditioner works

This is a property of the *formulation* (gap crossflows as unknowns, one momentum balance per gap), not of the
indexing, and not something any solver option fixes. Measured on the mini-core Jacobians:

| property | value |
|---|---|
| condition number, first / converged Newton step | 1.6e8 / 7e6 |
| singular values below 1e-3 of the largest | 741 = nsurf - nchan + 1, at every step |
| |diag| / sum|offdiag| | 1/3 in every row |
| relative asymmetry | <= 4e-3 (donor-cell upwinding; zero in uniform single phase) |

The Jacobian is F + B C: F the diagonal of friction slopes, B the incidence matrix, C the channel pressure
response. B C is an edge Laplacian. Any circulation around a closed loop of gaps changes no channel's net source,
so the pressure coupling cannot see it; only friction can, and friction is 1e6-1e8 times weaker. About half the
unknowns are such circulations. Consequences we measured on Watts Bar (32k unknowns):

| solver | result |
|---|---|
| BiCGStab/GMRES + ILU(0), ILU(1), Jacobi, SOR, ICC | diverge on nearly every solve |
| GMRES + algebraic multigrid (GAMG) | killed after 850 s, diverging |
| exact LU factored once per plane and reused as preconditioner | converges, but 180-230 Krylov its per solve (on the circulation modes the preconditioned operator is the ratio of new to old friction slopes, which changes by O(1) per Newton step) |
| fresh sparse LU every Newton step | the only thing that works; 0.15-0.3 s per factorization, fill 17.8x, single-threaded |

A threshold ILU that does converge keeps 75% of the LU fill. Threaded direct solvers (Pardiso, MUMPS) would buy
2-4x on this size and nothing on a GPU. Do not spend time tuning preconditioners on this system.

## 4. Solve in channel space instead (4.4x on Watts Bar; the GPU-capable path)

The same physics written in channel pressures is a node Laplacian with no structural null space (the "node"
vs "loop" method of pipe-network analysis). You do not have to rewrite the residual to get it: the Woodbury
identity on P = F + B C gives

    P^-1 r = F^-1 r - F^-1 B y,     N y = C F^-1 r,     N = I + C F^-1 B

with N of size nchan (half the unknowns), 5-point stencil, M-matrix structure. Implementation:

- Build the channel-adjacency CSR of N at setup, plus for each gap the four positions (from,from), (from,to),
  (to,from), (to,to) it contributes to. N is assembled with a scatter over gaps (atomics if threaded).
- Per gap, store F(ns) and the two entries C(from,ns), C(to,ns); they come from the same per-channel partials
  and per-gap source derivatives as the analytic Jacobian.
- Apply P^-1 as: u = r/F; g = scatter of C u into channels; solve N y = g; x = (r - (y(from) - y(to)))/F.
- Use P as the **preconditioner of a flexible GMRES on the exact Jacobian** rather than as the solver. The
  exact Jacobian has one more small term (two-phase donor-quality derivative) that P drops; in single phase
  P = J and the outer iteration converges in one step, in two-phase in 1-2 with an exact inner solve.

Two inner-solve modes:

- **Direct LU of N** (CPU default): 1.68 s vs 7.35 s for the LU of J on Watts Bar at 4 threads, identical
  results. The factorization of a 16k 5-point matrix is ~25 ms vs ~115 ms for J.
- **Iterative on N** (the device path: every kernel is a sparse mat-vec, a scatter, or a vector operation).
  Three things that cost us most of a day, in order:
  1. N = I - L with L a weighted channel Laplacian whose weights are pressure coupling / friction slope, i.e.
     up to 1e8. N has one eigenvalue +1 (uniform pressure) and all others large and negative. Every approximate
     preconditioner (Jacobi, ILU, AMG) stalls trying to resolve that uniform mode. But B 1 = 0, so the uniform
     component of y never enters x. Solve instead the exactly singular O = -N + (1/n) 1 1^T with the constant
     null space attached (project it out of the right-hand side and iterates) and use the definite L + I as
     the preconditioning matrix; it differs from O by a relative 2/lambda on the modes that matter.
  2. Restarted GMRES stagnates on this Laplacian-like system even when the matrix is a diagonally dominant
     M-matrix; BiCGStab (or CG-type methods) converge. Check with an offline solve of a dumped matrix before
     blaming the preconditioner.
  3. The back-substitution x = (r - B y)/F amplifies inner errors by 1/F, up to 1e8 at zero crossflow. The
     inner tolerance must stay tight (1e-8); at 1e-5 the outer iteration count explodes. On Watts Bar that is
     ~25 AMG-preconditioned BiCGStab iterations per apply and ~5 outer iterations per Newton step, which only
     ties the LU of J on the CPU but is all device-side work.

Results are identical to the LU path (one plane needed one extra Newton step because the linear solve stops
at a relative 1e-8 instead of machine precision).

## 5. Robustness details that bit us

- A NaN residual must not look converged: `max(|r|)` with NaNs returned -inf and the plane "converged" in one
  step. Map NaN to +inf in the reduction.
- Degenerate inputs (channels with zero flow area) give all-zero Jacobian rows; the original produced NaN
  pressures after hundreds of zero-pivot errors. Detect and report, do not optimize around it.
- Per-plane timings that include first-call initialization (MPI, library setup) will mislead every comparison.
- Setup accessors that copy a whole table on each call, used inside setup loops over all channels, made the
  geometry setup O(n^2); cache them. The gap-neighbour search was also O(nsurf^2) (a billion comparisons on
  Watts Bar); a channel-to-gap map makes it linear.

## 6. Not done yet, in the order we would try them

1. Raise the zero-crossflow friction floor from 5e-4 to a physical fraction of typical crossflow: bounds 1/F,
   lets the iterative inner solve run looser, cuts outer iterations. Changes the Newton path only.
2. Keep the matrix objects between Newton steps so the symbolic factorization / AMG hierarchy is reused (the
   pattern is fixed, only values change).
3. Inexact Newton (Eisenstat-Walker): the residual falls from 1e4 to 1e-8 over 8-10 steps per plane; early
   steps do not need 1e-8 linear solves.
4. Better per-plane initial guess (extrapolate from the two previous planes instead of copying one).
5. Question the absolute Newton tolerance (1e-8 on a residual that starts at 1e4) and the non-smooth donor-cell
   switch, which keeps Newton from converging quadratically. Both are modeling decisions.
6. Startup: MPI_Init is 1.5 s per run for a single-process app; build the dependencies without MPI or set the
   MPI transport to shared-memory only.

## 7. Checklist for the other app

- [ ] Phase timings per Newton step; startup excluded; thread count sanity-checked.
- [ ] Jacobian O(n): locality (two channels, <= 7 residuals per column), fixed CSR pattern, consistent FD.
- [ ] Analytic Jacobian: per-gap source derivatives x per-channel partials; implicit derivative at the void
      root; secant-regularized friction diagonal; `check` mode against consistent FD.
- [ ] Newton root for the void fraction with derivative output.
- [ ] Do not tune preconditioners on the gap-space system; it has a cycle-space near-null space by construction.
- [ ] Channel-space (Woodbury / node-formulation) solve: direct LU of N on CPU; projected singular operator +
      BiCGStab + AMG for the device; tight inner tolerance; FGMRES on the exact Jacobian outside.
- [ ] NaN-safe convergence test; detect degenerate channels.

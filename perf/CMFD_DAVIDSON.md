# CMFD eigenvalue solve: Generalized Davidson

Branch `perf/cmfd-generalized-davidson` (on top of `perf/cmfd-kokkos-block-gs`). Follows the approach MPACT
adopted in CASL-X-2016-1106-000 (Collins, Hamilton, Stimpson, Yee, Larsen, Kochunas, 2016): replace the
(shifted) power iteration, which needs a converged linear solve with M every outer iteration, by a
Generalized Davidson subspace iteration that needs only one preconditioner application per iteration.

## What was built

`cmfd/src/DavidsonEigenSolver.{hpp,cpp}`, selected by `-eigensolver davidson` (now the `cmfd_exec` default;
`-eigensolver power` gives the previous solver, which `PetscEigenSolver` still implements for the tests).

Per iteration, for the subspace V (orthonormal, with M V and F V stored alongside):
1. Rayleigh-Ritz: (V^T M V) y = theta (V^T F V) y, solved as the small dense standard problem
   (V^T M V)^-1 (V^T F V) y = k y with LAPACK `dgesv` + `dgeev`; take the largest real k (= keff).
2. Ritz vector phi = V y and its images M phi, F phi as linear combinations (no operator applications).
3. Residual r = M phi - F phi / k; converged when |dk| < 1e-7 and ||r|| / ||M phi|| < 1e-8.
4. Correction t = P^-1 r with P built on M (never the shifted operator): here P is the Kokkos block
   red-black Gauss-Seidel from branch 2 (`kokkosbgs`, CCSI relaxation, 6 sweeps by default), the
   GPU-friendly stand-in for the AMG/ILU preconditioners of the report.
5. t is orthonormalized against V (two Gram-Schmidt passes) and appended, with one M matvec and one
   fission-operator application; at subspace size 20 the basis restarts from the current Ritz vector.

Options: `-davidson_max_subspace`, `-davidson_rtol`, `-davidson_pc_type`, `-davidson_pc_kokkosbgs_sweeps`,
`-davidson_verbose` (prints the Ritz spectrum and Ritz-vector range each iteration).

`cmfd_exec` now also prints the a-posteriori eigen-residual ||M phi - F phi/k|| / ||M phi|| for either
solver, so convergence can be judged independently of each iteration's own stopping test.

## Measurements (8 threads, best of 3, identical keff to 1e-8 for all methods)

| | original | branch 1 (GMRES+Jacobi, fixed kernels) | branch 2 (GMRES + 2 CCSI RB-GS sweeps) | **branch 3 (Davidson, 6 RB-GS sweeps)** |
|---|---|---|---|---|
| mini-core: outer / Davidson iterations | 13 | 13 | 13 | 18 |
| mini-core: M matvecs | 1585 | 1122 | 241 | **19** |
| mini-core: preconditioner applications (sweeps for 2, 3) | 1599 | 1136 | 510 | **108** |
| mini-core: PETSc-logged time (s) | 1.754 | 0.115 | 0.085 | **0.054** |
| 7x7: M matvecs | 549 | 282 | 62 | **13** |
| 7x7: preconditioner applications | 554 | 287 | 134 | **72** |
| 7x7: PETSc-logged time (s) | 0.082 | 0.057 | 0.027 | **0.011** |

Davidson parameter sensitivity on mini-core (iterations / sweeps): 1 sweep per correction 67 / 67,
3 sweeps 26 / 78, 6 sweeps 16-18 / 96-108, 10 sweeps 11 / 100; subspace 10 vs 20 vs 40 makes no difference
(25 / 26 / 24 iterations); point-Jacobi as P needs 100+ iterations, so the block RB-GS preconditioner is
doing the work. Each Davidson iteration also costs O(m) dot products (155 VecMDot calls for the 3-sweep run
vs 241 for the GMRES-based power iteration).

`cmfd_solver_comparison_4way.png` plots the table.

## Correctness notes

- All 12 CMFD tests pass, including new `CMFD-test_DavidsonEigenSolverLight/Heavy` that compare keff and the
  flux against the gold values for all three assemblers. Suite time 19 s.
- The eigenvalue of a non-symmetric pencil is first order in the eigenvector error, so a residual tolerance
  of 1e-6 was not enough: the weakly preconditioned Simple-assembler case satisfied |dk| < 1e-7 while being
  1.7e-7 off. The default is now 1e-8 (26 -> 32 iterations with 3 sweeps on mini-core).
- Found and fixed via Davidson: `SimpleMatrixAssembler::_assembleFission` skipped entries below machine
  epsilon without clearing them, so values from the previous call survived and F was not a linear operator
  when applied to correction vectors with tiny components (the power iteration only ever applies F to
  positive fluxes and never noticed). It now zeroes the vector first. The Simple and COO assemblers also set
  the matrix block size so block preconditioners work on them.
- Davidson's Ritz spectrum on 7x7 shows the fundamental at 1.3408666 with the next eigenvalue near 0.05
  (dominance ratio ~0.04), which is why the power iteration also needs only 4-5 outer iterations there;
  mini-core (vacuum boundaries) has a much larger dominance ratio and is where Davidson's advantage shows.

## Relation to the report

The report measured a ~13x reduction in CMFD solve time (shifted power + GMRES/block-Jacobi -> Davidson +
AMG) on 2D and 3D quarter cores at 15-4000 cores. Here, on one node, the chain of changes gives 1585 -> 19
matvecs and 1.75 -> 0.054 s of solver time on mini-core (33x), of which the Davidson step itself is 241 -> 19
matvecs over branch 2. The report's caveat that Davidson "struggled for large systems" with weak PETSc
preconditioners applies: with point Jacobi it needs 100+ iterations here too. A multigrid-style P (block
RB-GS as smoother plus an assembly-level coarse correction) is the natural next step for full-core sizes.

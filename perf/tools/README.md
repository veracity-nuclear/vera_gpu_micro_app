# perf tools

- `bench.sh <build_dir> <out_log> [quick] [name_regex]` – runs the benchmark set used in `../REPORT.md`
  (MOC with fixed sweep counts, CMFD, subchannel and conduction tests) from a build directory and appends
  `BENCH <name> wall=... rss=...` lines to the log. Copy the large LFS data files
  (`moc/data/mini-core_*.h5`, `cmfd/data/mini-core_*.h5`) into `<build_dir>/{moc,cmfd}/data` first;
  CMake only copies the LFS pointer files.
- `kptimer.cpp` – a ~60-line Kokkos Tools profiler: `g++ -O2 -fPIC -shared -o libkptimer.so kptimer.cpp`,
  then `KOKKOS_TOOLS_LIBS=$PWD/libkptimer.so ./micro_moc ...` prints per-kernel, per-region and deep-copy
  totals at exit. Requires a Kokkos built with tools support (the default).
- `baseline_breakdown_chart.py`, `before_after_chart.py`, `cmfd_solver_comparison_chart.py`,
  `cmfd_solver_comparison_4way_chart.py`, `subchannel_jacobian_comparison_chart.py`,
  `subchannel_total_runtime_chart.py`, `subchannel_schur_comparison_chart.py`, `subchannel_schur_total_runtime_chart.py` – matplotlib scripts for the
  figures in `..`. The numbers are embedded; edit them after a new benchmark run.

PETSc options must be passed via `PETSC_OPTIONS="-log_view"` because the apps' argument parser rejects unknown
single-dash options other than `-ksp_*`, `-pc_*`, `-mat_*`.

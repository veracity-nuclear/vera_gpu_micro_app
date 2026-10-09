#!/bin/bash
# Usage: bench.sh <build_dir> <out_log> [quick]
# Runs a fixed benchmark set and appends "BENCH <name> wall=<s> rss=<KB> <extra>" lines to <out_log>.
B=$(readlink -f "$1"); OUT=$(readlink -f "$2"); QUICK=$3; FILTER=${4:-.}
# Set up PETSC_DIR/PETSC_ARCH/LD_LIBRARY_PATH etc. before running (or source your own env script here)
export OMP_NUM_THREADS=8
TMO=1800
run() { # name, cwd, cmd...
  local name=$1 cwd=$2; shift 2
  [[ "$name" =~ $FILTER ]] || return 0
  local t0=$(date +%s.%N)
  local tf=$(mktemp)
  ( cd "$cwd" && timeout $TMO /usr/bin/time -f "%e %M" -o "$tf" "$@" > "$OUT.$name.txt" 2>&1 )
  local rc=$?
  local wall rss; read wall rss < "$tf"; rm -f "$tf"
  local extra=""
  extra+=$(grep -E 'Converged after|Hit maximum|keff = |Final keff|Solved .* pins|Completed axial plane' "$OUT.$name.txt" | tail -1 | tr -s ' ' | cut -c1-80)
  echo "BENCH $name rc=$rc wall=${wall}s rss=${rss}KB | $extra" | tee -a "$OUT"
}
MOC="$B/moc/micro_moc"
XS=data/c5g7.xsl
# ---- MOC: fixed iteration counts so per-iteration cost is comparable ----
run moc_7x7_serialsweeper_dbl  "$B/moc" $MOC data/7x7_7g_16a_3p_serial.h5 $XS --sweeper serial --device serial --precision double --max_iter 40
run moc_7x7_kokkos_serial_dbl  "$B/moc" $MOC data/7x7_7g_16a_3p_serial.h5 $XS --sweeper kokkos --device serial --precision double --max_iter 40
run moc_7x7_kokkos_omp8_dbl    "$B/moc" $MOC data/7x7_7g_16a_3p_serial.h5 $XS --sweeper kokkos --device openmp --precision double --max_iter 40 --kokkos-num-threads=8
run moc_7x7_kokkos_omp8_sgl    "$B/moc" $MOC data/7x7_7g_16a_3p_serial.h5 $XS --sweeper kokkos --device openmp --precision single --max_iter 40 --kokkos-num-threads=8
run moc_pin_kokkos_omp8_dbl_full "$B/moc" $MOC data/pin_7g_16a_3p_serial.h5 $XS --sweeper kokkos --device openmp --precision double --kokkos-num-threads=8
if [ -z "$QUICK" ] && [ $(stat -c %s "$B/moc/data/mini-core_7g_16a_3p_serial.h5") -gt 1000000 ]; then
run moc_minicore_kokkos_omp8_dbl "$B/moc" $MOC data/mini-core_7g_16a_3p_serial.h5 $XS --sweeper kokkos --device openmp --precision double --max_iter 5 --kokkos-num-threads=8
fi
# ---- CMFD ----
CM="$B/cmfd/cmfd_exec"
run cmfd_pin       "$B/cmfd" $CM data/pin_7g_16a_3p_serial.h5
run cmfd_7x7       "$B/cmfd" $CM data/7x7_7g_16a_3p_serial.h5
run cmfd_smallpar  "$B/cmfd" $CM data/small_parallel.h5
[ -z "$QUICK" ] && run cmfd_minicore "$B/cmfd" $CM data/mini-core_7g_16a_3p_serial.h5
# ---- Conduction ----
run conduction_smr_serial "$B/conduction" $B/conduction/test_smr_serial
[ -x "$B/conduction/test_smr_omp" ] && run conduction_smr_omp "$B/conduction" $B/conduction/test_smr_omp --kokkos-num-threads=8
# ---- Subchannel ----
run subch_3x3_serial  "$B/subchannel" $B/subchannel/test_ants_3x3_serial
run subch_7x7_serial  "$B/subchannel" $B/subchannel/test_ants_7x7_serial
run subch_7x7_omp     "$B/subchannel" $B/subchannel/test_ants_7x7_omp --kokkos-num-threads=8
run subch_smr_omp     "$B/subchannel" $B/subchannel/test_ants_smr_omp
if [ -z "$QUICK" ]; then
run subch_minicore_omp "$B/subchannel" $B/subchannel/test_ants_minicore_omp --kokkos-num-threads=8
fi
echo "BENCH DONE" | tee -a "$OUT"

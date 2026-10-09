import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
INK, INK2, SURF, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"
COLS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
versions = ["original\n(master)", "branch 1\nGMRES+Jacobi\nfixed kernels", "branch 2\nGMRES+2 CCSI\nRB-GS sweeps", "branch 3\nGen. Davidson\n6 RB-GS sweeps"]
# best of 3 runs, 8 threads, same machine/session
data = {
 "mini-core": {
   "inner matvecs":                        [1585, 1122, 241, 19],
   "preconditioner applications\n(Jacobi applies; block RB-GS sweeps for 2, 3)": [1599, 1136, 510, 108],
   "PETSc-logged time (s)\nsolve + fission source + assembly": [1.754, 0.115, 0.085, 0.054],
   "wall time (s)\nincl. ~1.5 s MPI startup":  [3.25, 1.63, 1.58, 1.55],
 },
 "7x7": {
   "inner matvecs":                        [549, 282, 62, 13],
   "preconditioner applications\n(Jacobi applies; block RB-GS sweeps for 2, 3)": [554, 287, 134, 72],
   "PETSc-logged time (s)\nsolve + fission source + assembly": [0.082, 0.057, 0.027, 0.0107],
   "wall time (s)\nincl. ~1.5 s MPI startup":  [1.57, 1.58, 1.52, 1.53],
 },
}
outer = {"mini-core": [13, 13, 13, 18], "7x7": [4, 4, 4, 12]}
fig, axes = plt.subplots(2, 4, figsize=(17, 8.8), facecolor=SURF)
fig.suptitle("CMFD eigenvalue solve: original, fixed kernels, Kokkos block RB-SOR inner solver, Generalized Davidson",
             fontsize=14, color=INK, fontweight="bold", x=0.02, ha="left", y=0.99)
for r, (case, metrics) in enumerate(data.items()):
    for c, (metric, vals) in enumerate(metrics.items()):
        ax = axes[r, c]; ax.set_facecolor(SURF)
        x = np.arange(4)
        bars = ax.bar(x, vals, color=COLS, width=0.66, edgecolor=SURF, linewidth=1)
        top = max(vals); ax.set_ylim(0, top * 1.3)
        for b, v in zip(bars, vals):
            txt = f"{v:,.0f}" if v >= 20 else (f"{v:.3f}" if v < 0.1 else f"{v:.2f}")
            ax.text(b.get_x() + b.get_width()/2, v + top*0.02, txt, ha="center", va="bottom", fontsize=9.5, color=INK)
        for i in (1, 2, 3):
            ax.text(x[i], top*1.19, f"×{vals[0]/vals[i]:.1f}", ha="center", fontsize=9, color=INK2, style="italic")
        ax.set_xticks(x); ax.set_xticklabels(versions if r == 1 else [""]*4, fontsize=8.2, color=INK2)
        ax.set_title(metric, fontsize=9.5, color=INK, loc="left")
        ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
        for s in ["top", "right"]: ax.spines[s].set_visible(False)
        ax.spines["left"].set_color(GRID); ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=INK2, length=0, labelsize=8.5)
    its = outer[case]
    axes[r, 0].set_ylabel(f"{case}\n(outer its: {', '.join(map(str, its))})", fontsize=10.5, color=INK, fontweight="bold", labelpad=10)
fig.text(0.02, 0.005, "Best of 3 runs, 8 OpenMP threads, Release build, identical keff (to 1e-8) across all four. Italic ×N = improvement over original. "
         "Davidson 'outer its' are Davidson iterations (one matvec, one fission apply, one preconditioner apply each); the others are power iterations.",
         fontsize=8.3, color=INK2)
plt.tight_layout(rect=(0, 0.025, 1, 0.96))
out = "perf/cmfd_solver_comparison_4way.png"
plt.savefig(out, dpi=160, facecolor=SURF, bbox_inches="tight", pad_inches=0.25)
print(out)

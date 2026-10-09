import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
INK, INK2, SURF, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"
COLS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a5cd6"]
versions = ["original", "branch 1\nFD Jacobian\n+ bisection", "FD Jacobian\n+ Newton\nvoid root",
            "analytic\nJacobian\n+ bisection", "branch 4\nanalytic\n+ Newton"]
# total wall-clock time of the test executable (/usr/bin/time %e), best of 3 runs, 8 OpenMP threads
# (3x3: serial). Includes ~1.5 s of MPI/UCX startup, geometry setup and the test harness.
cases = ["3x3 (12 surfaces)", "mini-core (1,540 surfaces)", "Watts Bar (32,096 surfaces)"]
wall = {cases[0]: [1.59, 1.57, 1.52, 1.51, 1.50],
        cases[1]: [21.86, 2.31, 1.92, 1.82, 1.65],
        cases[2]: [None, 31.15, 18.11, 20.42, 13.60]}
fig, axes = plt.subplots(1, 3, figsize=(16, 5.6), facecolor=SURF)
fig.suptitle("Subchannel total runtime (wall clock) by case and crossflow Jacobian / void-root method",
             fontsize=14, color=INK, fontweight="bold", x=0.02, ha="left", y=0.99)
x = np.arange(5)
for c, case in enumerate(cases):
    ax = axes[c]; ax.set_facecolor(SURF)
    vals = wall[case]
    shown = [v for v in vals[1:] if v is not None]
    top = max(shown) * 1.5
    for i, v in enumerate(vals):
        if v is None:
            ax.bar(x[i], top * 0.55, color="none", edgecolor=GRID, hatch="///", width=0.66, linewidth=1)
            ax.text(x[i], top * 0.58, "not run\n(>86 CPU-min,\nkilled)", ha="center", va="bottom",
                    fontsize=8, color=INK2, style="italic")
            continue
        h = min(v, top * 0.74)
        ax.bar(x[i], h, color=COLS[i], width=0.66, edgecolor=SURF, linewidth=1)
        txt = f"{v:.2f} s" if v < 10 else f"{v:.1f} s"
        if v > h:
            ax.plot([x[i] - 0.4, x[i] + 0.4], [h * 0.84, h * 0.88], color=SURF, lw=5, zorder=3)
            ax.plot([x[i] - 0.4, x[i] + 0.4], [h * 0.84, h * 0.88], color=INK2, lw=1, zorder=4)
            txt += "\n(off scale)"
        ax.text(x[i], h + top * 0.015, txt, ha="center", va="bottom", fontsize=9.5, color=INK)
    ref = vals[1]
    for i in (2, 3, 4):
        ax.text(x[i], top * 0.90, f"×{ref / vals[i]:.2f}", ha="center", va="bottom", fontsize=9, color=INK2, style="italic")
    ax.set_ylim(0, top)
    ax.set_xticks(x); ax.set_xticklabels(versions, fontsize=8.2, color=INK2)
    ax.set_title(case, fontsize=11, color=INK, loc="left", fontweight="bold")
    ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(GRID); ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, length=0, labelsize=8.5)
axes[0].set_ylabel("total wall-clock time (s)", fontsize=10.5, color=INK, fontweight="bold", labelpad=10)
fig.text(0.02, 0.005,
         "Wall-clock time of the whole test executable (/usr/bin/time), best of 3 runs, 8 OpenMP threads (3x3: serial), Release build. "
         "Every run carries ~1.5 s of MPI/UCX startup plus geometry setup, which is why 3x3 is flat and\nmini-core bottoms out near 1.6 s. "
         "Italic ×N = speedup over branch 1 (FD Jacobian + bisection). The original was not run on Watts Bar (killed after 86 CPU-minutes).",
         fontsize=8.3, color=INK2)
plt.tight_layout(rect=(0, 0.06, 1, 0.94))
out = "perf/subchannel_total_runtime.png"
plt.savefig(out, dpi=160, facecolor=SURF, bbox_inches="tight", pad_inches=0.25)
print(out)

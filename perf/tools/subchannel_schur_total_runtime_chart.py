import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
INK, INK2, SURF, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"
COLS = ["#2a78d6", "#eb6834", "#8a5cd6", "#1baf7a", "#eda100"]
versions = ["original\n(master)", "branch 1\nFD Jacobian\n+ bisection\n+ LU(J)", "branch 4\nanalytic\n+ Newton\n+ LU(J)",
            "branch 5\nchannel-space\ndirect LU(N)", "branch 5\nchannel-space\nBCGS + GAMG"]
cases = ["3x3 (12 surfaces)", "mini-core (1,540 surfaces)", "Watts Bar (32,096 surfaces)"]
# total wall-clock of the test executable (/usr/bin/time %e), best of 3, 4 OpenMP threads (3x3: serial),
# same session; includes ~1.5 s MPI/UCX startup, geometry setup and the test harness
wall = {cases[0]: [2.24, 1.59, 1.55, 1.59, 1.63], cases[1]: [12.06, 2.48, 1.72, 1.62, 2.26], cases[2]: [None, 29.41, 9.37, 3.78, 10.59]}
fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.8), facecolor=SURF)
fig.suptitle("Subchannel total runtime (wall clock): surface-space LU vs channel-space crossflow solver",
             fontsize=14, color=INK, fontweight="bold", x=0.02, ha="left", y=0.99)
x = np.arange(5)
for c, case in enumerate(cases):
    ax = axes[c]; ax.set_facecolor(SURF)
    vals = wall[case]; top = max(v for v in vals[2:]) * 1.5
    for i, v in enumerate(vals):
        if v is None:
            ax.bar(x[i], top * 0.55, color="none", edgecolor=GRID, hatch="///", width=0.66, linewidth=1)
            ax.text(x[i], top * 0.58, "not run\n(>86 CPU-min,\nkilled)", ha="center", va="bottom", fontsize=8, color=INK2, style="italic")
            continue
        h = min(v, top * 0.74)
        ax.bar(x[i], h, color=COLS[i], width=0.66, edgecolor=SURF, linewidth=1)
        txt = f"{v:.2f} s" if v < 10 else f"{v:.1f} s"
        if v > h:
            ax.plot([x[i] - 0.4, x[i] + 0.4], [h * 0.84, h * 0.88], color=SURF, lw=5, zorder=3)
            ax.plot([x[i] - 0.4, x[i] + 0.4], [h * 0.84, h * 0.88], color=INK2, lw=1, zorder=4)
            txt += "\n(off scale)"
        ax.text(x[i], h + top * 0.015, txt, ha="center", va="bottom", fontsize=9.5, color=INK)
    for i in (3, 4):
        r = vals[2] / vals[i]
        ax.text(x[i], top * 0.90, f"×{r:.2f}", ha="center", va="bottom", fontsize=9, color=INK2, style="italic")
    ax.set_ylim(0, top)
    ax.set_xticks(x); ax.set_xticklabels(versions, fontsize=7.8, color=INK2)
    ax.set_title(case, fontsize=11, color=INK, loc="left", fontweight="bold")
    ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(GRID); ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, length=0, labelsize=8.5)
axes[0].set_ylabel("total wall-clock time (s)", fontsize=10.5, color=INK, fontweight="bold", labelpad=10)
fig.text(0.02, 0.005,
         "Wall-clock time of the whole test executable (/usr/bin/time), best of 3 runs, 4 OpenMP threads (3x3: serial), Release build; original from the master build, "
         "the other four from the branch-5 binary, same session. The original was not run on Watts Bar.\nEvery run carries ~1.5 s of MPI/UCX startup plus geometry setup, which is why 3x3 is flat. "
         "Italic ×N = speedup over branch 4. Converged results are identical across variants.",
         fontsize=8.3, color=INK2)
plt.tight_layout(rect=(0, 0.06, 1, 0.94))
out = "perf/subchannel_schur_total_runtime.png"
plt.savefig(out, dpi=160, facecolor=SURF, bbox_inches="tight", pad_inches=0.25)
print(out)

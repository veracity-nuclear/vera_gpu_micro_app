import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
INK, INK2, SURF, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"
COLS = ["#eb6834", "#8a5cd6", "#1baf7a", "#eda100"]
versions = ["branch 1\nFD Jacobian\n+ bisection\n+ LU(J)", "branch 4\nanalytic\n+ Newton\n+ LU(J)",
            "branch 5\nchannel-space\ndirect LU(N)", "branch 5\nchannel-space\nBCGS + GAMG"]
cases = ["3x3 (12 surfaces)", "mini-core (1,540 surfaces)", "Watts Bar (32,096 surfaces)"]
# best of 3 runs, 4 OpenMP threads (3x3: serial). Branch 1 column from the branch-4 study (8 threads).
time = {cases[0]: [0.072, 0.001, 0.001, 0.048], cases[1]: [1.063, 0.153, 0.092, 0.714], cases[2]: [29.2, 7.35, 1.68, 8.47]}
its  = {cases[0]: [90, 45, 45, 45], cases[1]: [96, 37, 37, 37], cases[2]: [99, 63, 64, 63]}
rows = [("crossflow solve time, all planes (s)", time, True), ("crossflow Newton iterations, all planes", its, False)]
fig, axes = plt.subplots(2, 3, figsize=(15, 8.4), facecolor=SURF)
fig.suptitle("Subchannel crossflow solve: surface-space LU vs channel-space (Schur complement) solver",
             fontsize=14, color=INK, fontweight="bold", x=0.02, ha="left", y=0.99)
x = np.arange(4)
for r, (metric, data, is_time) in enumerate(rows):
    for c, case in enumerate(cases):
        ax = axes[r, c]; ax.set_facecolor(SURF)
        vals = data[case]; top = max(vals[1:]) * 1.5 if is_time else max(vals) * 1.4
        for i, v in enumerate(vals):
            h = min(v, top * 0.74)
            ax.bar(x[i], h, color=COLS[i], width=0.66, edgecolor=SURF, linewidth=1)
            txt = (f"{v:.3f}" if v < 0.1 else (f"{v:.2f}" if v < 10 else f"{v:.1f}")) if is_time else f"{v:d}"
            if v > h:
                ax.plot([x[i] - 0.4, x[i] + 0.4], [h * 0.84, h * 0.88], color=SURF, lw=5, zorder=3)
                ax.plot([x[i] - 0.4, x[i] + 0.4], [h * 0.84, h * 0.88], color=INK2, lw=1, zorder=4)
                txt += "\n(off scale)"
            ax.text(x[i], h + top * 0.015, txt, ha="center", va="bottom", fontsize=9.5, color=INK)
        if is_time:
            for i in (2, 3):
                ax.text(x[i], top * 0.90, (f"×{vals[1] / vals[i]:.1f}" if vals[1] / vals[i] >= 1 else f"×{vals[1] / vals[i]:.2f}"), ha="center", va="bottom", fontsize=9, color=INK2, style="italic")
        ax.set_ylim(0, top)
        ax.set_xticks(x); ax.set_xticklabels(versions if r == 1 else [""] * 4, fontsize=8.2, color=INK2)
        if r == 0: ax.set_title(case, fontsize=11, color=INK, loc="left", fontweight="bold")
        ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
        for s in ["top", "right"]: ax.spines[s].set_visible(False)
        ax.spines["left"].set_color(GRID); ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=INK2, length=0, labelsize=8.5)
    axes[r, 0].set_ylabel(metric, fontsize=10.5, color=INK, fontweight="bold", labelpad=10)
fig.text(0.02, 0.005,
         "Sum of per-plane crossflow Newton solve times over the 10 axial planes, MPI/PETSc startup excluded; best of 3 runs, Release build, "
         "4 OpenMP threads (3x3: serial). Italic ×N = speedup over branch 4.\n"
         "Branch 5 'direct' factors the channel-space matrix N (half the unknowns of J, 5-point stencil); 'BCGS + GAMG' solves N iteratively "
         "with every kernel device-capable. Converged exit void and pressure tables are identical across all variants.",
         fontsize=8.3, color=INK2)
plt.tight_layout(rect=(0, 0.04, 1, 0.96))
out = "perf/subchannel_schur_comparison.png"
plt.savefig(out, dpi=160, facecolor=SURF, bbox_inches="tight", pad_inches=0.25)
print(out)

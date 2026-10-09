import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
INK, INK2, SURF, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"
COLS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a5cd6"]
versions = ["original", "branch 1\nFD Jacobian\n+ bisection", "FD Jacobian\n+ Newton\nvoid root",
            "analytic\nJacobian\n+ bisection", "branch 4\nanalytic\n+ Newton"]
# best of 3 runs, 8 OpenMP threads (3x3 serial), same machine/session; sum of per-plane crossflow solve
# times over the 10 axial planes, MPI/PETSc startup excluded. Original mini-core: 21.4 s measured with the
# startup inside plane 2, ~1.0 s subtracted. Original on Watts Bar not run (killed after 86 CPU-minutes).
cases = ["3x3 (12 surfaces)", "mini-core (1,540 surfaces)", "Watts Bar (32,096 surfaces)"]
time = {cases[0]: [0.07, 0.072, 0.032, 0.017, 0.003],
        cases[1]: [20.4, 1.063, 0.462, 0.305, 0.126],
        cases[2]: [None, 29.20, 16.11, 18.37, 11.59]}
its  = {cases[0]: [90, 90, 102, 62, 45],
        cases[1]: [96, 96, 96, 78, 37],
        cases[2]: [None, 99, 77, 93, 63]}
rows = [("crossflow Newton solve time, all planes (s)", time, True),
        ("crossflow Newton iterations, all planes", its, False)]
fig, axes = plt.subplots(2, 3, figsize=(16, 8.6), facecolor=SURF)
fig.suptitle("Subchannel crossflow solve: finite-difference vs analytic Jacobian, bisection vs Newton void-fraction root",
             fontsize=14, color=INK, fontweight="bold", x=0.02, ha="left", y=0.99)
x = np.arange(5)
for r, (metric, data, is_time) in enumerate(rows):
    for c, case in enumerate(cases):
        ax = axes[r, c]; ax.set_facecolor(SURF)
        vals = data[case]
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
            if is_time:
                txt = f"{v:.3f}" if v < 0.1 else (f"{v:.2f}" if v < 10 else f"{v:.1f}")
            else:
                txt = f"{v:d}"
            if v > h:  # off-scale bar: break marks and label
                ax.plot([x[i] - 0.4, x[i] + 0.4], [h * 0.84, h * 0.88], color=SURF, lw=5, zorder=3)
                ax.plot([x[i] - 0.4, x[i] + 0.4], [h * 0.84, h * 0.88], color=INK2, lw=1, zorder=4)
                txt += "\n(off scale)"
            ax.text(x[i], h + top * 0.015, txt, ha="center", va="bottom", fontsize=9.5, color=INK)
        ref = vals[1]
        for i in (2, 3, 4):
            ax.text(x[i], top * 0.90, f"×{ref / vals[i]:.1f}" if ref / vals[i] < 100 else f"×{ref / vals[i]:.0f}",
                    ha="center", va="bottom", fontsize=9, color=INK2, style="italic")
        ax.set_ylim(0, top)
        ax.set_xticks(x); ax.set_xticklabels(versions if r == 1 else [""] * 5, fontsize=8.2, color=INK2)
        if r == 0:
            ax.set_title(case, fontsize=11, color=INK, loc="left", fontweight="bold")
        ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
        for s in ["top", "right"]: ax.spines[s].set_visible(False)
        ax.spines["left"].set_color(GRID); ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=INK2, length=0, labelsize=8.5)
    axes[r, 0].set_ylabel(metric, fontsize=10.5, color=INK, fontweight="bold", labelpad=10)
fig.text(0.02, 0.005,
         "Sum of per-plane crossflow Newton solve times over the 10 axial planes, MPI/PETSc startup excluded; best of 3 runs, "
         "8 OpenMP threads (3x3: serial), Release build. Italic ×N = improvement over branch 1 (FD Jacobian + bisection).\n"
         "Original mini-core time: 21.4 s measured with ~1.0 s startup inside the first solved plane, subtracted here. "
         "The original was not run on Watts Bar (killed after 86 CPU-minutes without finishing). Exit void and pressure "
         "tables are identical across all variants.",
         fontsize=8.3, color=INK2)
plt.tight_layout(rect=(0, 0.04, 1, 0.96))
out = "perf/subchannel_jacobian_comparison.png"
plt.savefig(out, dpi=160, facecolor=SURF, bbox_inches="tight", pad_inches=0.25)
print(out)

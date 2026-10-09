import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INK, INK2, SURF, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"
BEFORE, AFTER = "#2a78d6", "#eb6834"   # categorical slots 1 and 2

# (label, before_s, after_s, note)
rows = [
 ("MOC  7x7 · Kokkos Serial · 40 sweeps",            18.43, 6.51, ""),
 ("MOC  7x7 · Kokkos OpenMP×8 · 40 sweeps (double)",  4.53, 2.53, ""),
 ("MOC  7x7 · Kokkos OpenMP×8 · 40 sweeps (single)",  5.56, 2.51, ""),
 ("MOC  pin · OpenMP×8 · full solve",                 2.52, 1.11, ""),
 ("MOC  mini-core (62k rays) · OpenMP×8 · 5 sweeps", 19.53, 9.65, "sweeps 7.9→1.8 s; rest is HDF5 read"),
 ("CMFD mini-core · 14 power iterations",             3.27, 1.61, "PETSc time 1.76→0.10 s; rest is MPI init"),
 ("Subchannel  mini-core · OpenMP×8 · 10 planes",    40.97, 2.48, ""),
 ("Subchannel  7x7 · OpenMP×8",                        3.63, 1.67, ""),
 ("Subchannel  SMR · OpenMP×4",                        3.13, 2.42, "ill-posed input (NaN Jacobian) in both"),
 ("Conduction  SMR 184,730 pins · serial",             0.83, 0.64, "solve 319→190 ms; was 8 threads before"),
 ("Conduction  SMR 184,730 pins · OpenMP×8",           0.83, 0.70, "solve 42 ms; new OpenMP path"),
]
labels = [r[0] for r in rows]; before = np.array([r[1] for r in rows]); after = np.array([r[2] for r in rows])
y = np.arange(len(rows))[::-1]
h = 0.36
fig, ax = plt.subplots(figsize=(13, 8.2), facecolor=SURF)
ax.set_facecolor(SURF)
ax.barh(y + h/2, before, height=h, color=BEFORE, label="before", edgecolor=SURF, linewidth=1)
ax.barh(y - h/2, after,  height=h, color=AFTER,  label="after",  edgecolor=SURF, linewidth=1)
ax.set_xscale("log")
ax.set_xlim(0.4, 80)
ax.set_xlabel("wall-clock seconds (log scale, lower is better)", color=INK2)
ax.set_yticks(y); ax.set_yticklabels(labels, color=INK, fontsize=10)
ax.xaxis.grid(True, color=GRID, linewidth=1); ax.set_axisbelow(True)
for s in ["top", "right", "left"]: ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(GRID); ax.tick_params(axis="x", colors=INK2); ax.tick_params(axis="y", length=0)
for yi, b, a, note in zip(y, before, after, [r[3] for r in rows]):
    ax.text(b * 1.08, yi + h/2, f"{b:.2f} s", va="center", fontsize=9, color=INK2)
    ax.text(a * 1.08, yi - h/2, f"{a:.2f} s   ×{b/a:.1f}", va="center", fontsize=9, color=INK, fontweight="bold")
    if note:
        ax.text(78, yi - h*1.02, note, va="center", ha="right", fontsize=8.5, color=INK2, style="italic")
ax.legend(loc="upper right", frameon=False, labelcolor=INK2, fontsize=10)
ax.set_title("Micro-app wall time before vs after optimization\n8-core laptop, Release build, same inputs and iteration counts; identical results",
             color=INK, fontsize=13, loc="left", pad=12)
plt.tight_layout()
out = "perf/before_after.png"
plt.savefig(out, dpi=160, facecolor=SURF, bbox_inches="tight", pad_inches=0.25)
print(out)

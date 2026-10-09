import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]
INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"

panels = [
 ("MOC  ·  7x7 assembly, Kokkos OpenMP ×8, 40 sweeps", 5.07, [
   ("Transport sweep kernel", 2.303),
   ("Ray data read (HDF5)", 1.593),
   ("Startup / teardown", 0.640),
   ("Setup: XS, mirrors, exp table", 0.289),
   ("Sweep overhead: copies, zeroing, reduction", 0.175),
   ("Source + fission + k-eff update (host)", 0.066)]),
 ("CMFD  ·  mini-core, 14 power iterations", 3.57, [
   ("Startup / teardown (MPI, PETSc, Kokkos)", 1.593),
   ("Fission-source kernel (TeamPolicy on host)", 1.295),
   ("KSPSolve: GMRES + Jacobi", 0.459),
   ("HDF5 read + matrix assembly", 0.223)]),
 ("Subchannel  ·  mini-core, OpenMP ×8, 10 planes", 26.47, [
   ("Jacobian: re-accumulate all surface sources", 15.13),
   ("Jacobian: perturbed residuals", 6.46),
   ("Startup, geometry setup, host loops", 2.93),
   ("Sparse LU solve", 1.20),
   ("Copies, zeroing, view allocation", 0.68),
   ("Physics kernels (planar, mixing, evaporation)", 0.07)]),
 ("Conduction  ·  SMR, 184,730 pins, serial", 0.96, [
   ("HDF5 load, init, teardown", 0.55),
   ("Per-pin solver + vector allocation", 0.21),
   ("Conduction solve kernel", 0.17),
   ("Result view setup", 0.03)]),
]

fig, axes = plt.subplots(2, 2, figsize=(14, 13), facecolor=SURF)
fig.suptitle("Where the time goes: baseline wall-clock breakdown per micro-app",
             fontsize=16, color=INK, fontweight="bold", y=0.985)
for ax, (title, total, parts) in zip(axes.flat, panels):
    ax.set_facecolor(SURF)
    labels = [p[0] for p in parts]; vals = [p[1] for p in parts]
    cols = PAL[:len(parts)]
    wedges, _ = ax.pie(vals, colors=cols, startangle=90, counterclock=False,
                       wedgeprops=dict(width=0.42, edgecolor=SURF, linewidth=2))
    ax.set_title(f"{title}\ntotal {total:.2f} s", fontsize=12, color=INK, pad=8)
    # percentage labels, only for slices >= 4%
    for w, v in zip(wedges, vals):
        pct = 100 * v / total
        if pct < 4: continue
        ang = (w.theta2 + w.theta1) / 2.0
        import math
        x, y = 0.79 * math.cos(math.radians(ang)), 0.79 * math.sin(math.radians(ang))
        ax.text(x, y, f"{pct:.0f}%", ha="center", va="center", fontsize=10,
                color="#ffffff" if pct > 8 else INK, fontweight="bold")
    leg_labels = [f"{l}  —  {v:.2f} s" for l, v in zip(labels, vals)]
    ax.legend(wedges, leg_labels, loc="upper center", bbox_to_anchor=(0.5, -0.02),
              fontsize=9, frameon=False, labelcolor=INK2, ncol=1)
    ax.set_aspect("equal")
plt.subplots_adjust(left=0.03, right=0.97, top=0.92, bottom=0.08, hspace=0.6, wspace=0.1)
out = "perf/baseline_timing_breakdown.png"
plt.savefig(out, dpi=160, facecolor=SURF, bbox_inches="tight", pad_inches=0.3)
print(out)

"""
run_autonomous.py
=================
Autonomous (no stimulus) dynamics study.

Produces:
  output/autonomous/sweep_vmem.png   — Vmem at T=1000 for 9 W/B/R combos (Figure 3 style)
  output/autonomous/timeseries.png   — std, mean, Gp over 20000 steps (stigmergic params)
  output/autonomous/cell_traj.png    — individual cell trajectories
  output/autonomous/snapshots.png    — Vmem snapshots at key moments (stigmergic params)

"""

import os
import numpy as np
import matplotlib.pyplot as plt

from dynamics_t_sparse_gh import BioelectricLattice, G_r, E_p, E_d

OUT = "output/autonomous"
os.makedirs(OUT, exist_ok=True)

# Scala dei plot presa dalle costanti del modello (E_p/E_d), non hardcodata.
VMIN_MV, VMAX_MV = E_p * 1000, E_d * 1000

# -----------------------------------------------------------------------
# 1.  Parameter sweep — Figure 3 style  (W × R combinations)
#     Weak sensitivity:   W=10,   B=0.02
#     Strong sensitivity: W=1000, B=0.0005
# -----------------------------------------------------------------------
SWEEP = [
    # label,          W,     B,       R
    ("weak / short",   10,    0.02,   1),
    ("weak / mid",     10,    0.02,   4),
    ("weak / long",    10,    0.02,  21),
    ("mid / short",   500,   0.005,   1),
    ("mid / mid",     500,   0.005,   4),
    ("mid / long",    500,   0.005,  21),
    ("strong / short", 1000, 0.0005,  1),
    ("strong / mid",   1000, 0.0005,  4),
    ("strong / long",  1000, 0.0005, 21),
]

T_SWEEP = 1000
G       = 0.05

print("Running parameter sweep (T=1000)...")
fig, axes = plt.subplots(3, 3, figsize=(10, 10))
fig.suptitle(
    "Vmem at T=1000 — autonomous dynamics\n"
    "Rows: weak / mid / strong field sensitivity   "
    "Cols: R=1 / R=4 / R=21",
    fontsize=11
)

for ax, (label, W, B, R) in zip(axes.ravel(), SWEEP):
    net = BioelectricLattice(G=G, R=R, W=W, B=B)
    v_hist, _, _ = net.run(T=T_SWEEP, save_every=T_SWEEP)
    v_final = v_hist[-1].numpy().reshape(11, 11) * 1000
    im = ax.imshow(v_final, cmap="gray", origin="upper")
    ax.set_title(f"{label}\nv=[{v_final.min():.0f},{v_final.max():.0f}]mV", fontsize=7)
    ax.set_xticks([]); ax.set_yticks([])

plt.savefig(f"{OUT}/sweep_vmem.png", dpi=150, bbox_inches="tight")
plt.close()
print("  saved sweep_vmem.png")

# -----------------------------------------------------------------------
# 2.  Long run — stigmergic parameters (W=1000, B=5e-4, R=4)
# -----------------------------------------------------------------------
PARAMS = dict(G=0.05, R=4, W=1000.0, B=5e-4)
T_LONG = 20000
SAVE   = 20    # save every 20 steps → 1000 frames

print(f"Running long autonomous run (T={T_LONG}, stigmergic params)...")
net = BioelectricLattice(**PARAMS)
v_hist, gp_hist, em_hist = net.run(T=T_LONG, save_every=SAVE)

v  = v_hist.numpy()
gp = gp_hist.numpy()
em = em_hist.numpy()
steps = np.arange(1, len(v) + 1) * SAVE

# ── 2a. Timeseries ──────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
fig.suptitle(
    f"Autonomous dynamics timeseries — W={PARAMS['W']}, B={PARAMS['B']}, "
    f"R={PARAMS['R']}, G={PARAMS['G']}\n(T={T_LONG} steps)",
    fontsize=11
)

ax = axes[0]
ax.plot(steps, v.mean(1)*1000, color="steelblue", lw=1.5, label="mean")
ax.fill_between(steps, v.min(1)*1000, v.max(1)*1000,
                alpha=0.2, color="steelblue", label="min / max")
ax.plot(steps, v.std(1)*1000, color="tomato", lw=1.2, ls="--", label="std")
ax.axhline(VMIN_MV, color="gray", ls=":", lw=0.8, label=f"E_p = {VMIN_MV:.0f} mV")
ax.axhline(VMAX_MV, color="gray", ls="-.", lw=0.8, label=f"E_d = {VMAX_MV:.0f} mV")
ax.set_ylabel("mV"); ax.set_title("Vmem statistics")
ax.legend(fontsize=8, ncol=5); ax.grid(alpha=0.3)

ax = axes[1]
ax.plot(steps, gp.mean(1)/G_r, color="darkorange", lw=1.5, label="mean G_p / G_r")
ax.set_ylim(0, 2.1); ax.set_ylabel("G_p / G_r")
ax.set_title("Mean polarizing channel conductance")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[2]
ax.plot(steps, v.std(1)*1000, color="mediumpurple", lw=1.5)
ax.set_ylabel("std(Vmem) [mV]"); ax.set_xlabel("step")
ax.set_title("Pattern heterogeneity (std)")
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUT}/timeseries.png", dpi=150, bbox_inches="tight")
plt.close()
print("  saved timeseries.png")

# ── 2b. Vmem snapshots at key moments ──────────────────────────────────
SNAP_STEPS  = [100, 500, 1000, 2000, 5000, 10000, 15000, 20000]
SNAP_FRAMES = [s // SAVE - 1 for s in SNAP_STEPS]

fig, axes = plt.subplots(2, 4, figsize=(13, 7))
fig.suptitle("Vmem snapshots — autonomous, stigmergic parameters", fontsize=11)

for ax, (fi, st) in zip(axes.ravel(), zip(SNAP_FRAMES, SNAP_STEPS)):
    im = ax.imshow(
        v[fi].reshape(11, 11) * 1000,
        vmin=VMIN_MV, vmax=VMAX_MV, cmap="gray", origin="upper"
    )
    ax.set_title(f"t = {st}", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])

fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7, label="Vmem (mV)")
plt.savefig(f"{OUT}/snapshots.png", dpi=150, bbox_inches="tight")
plt.close()
print("  saved snapshots.png")

# ── 2c. Individual cell trajectories ───────────────────────────────────
CELLS = {
    "corner (0,0)":    0,
    "edge (0,5)":      5,
    "near-edge (1,1)": 1*11+1,
    "interior (5,5)":  5*11+5,
}

fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
fig.suptitle("Individual cell trajectories — stigmergic parameters", fontsize=11)

for name, idx in CELLS.items():
    axes[0].plot(steps, v[:, idx]*1000, label=name, lw=1.3)
    axes[1].plot(steps, gp[:, idx]/G_r, label=name, lw=1.3)

axes[0].axhline(VMIN_MV, color="gray", ls=":", lw=0.8)
axes[0].axhline(VMAX_MV, color="gray", ls="-.", lw=0.8)
axes[0].set_ylabel("Vmem (mV)"); axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)
axes[1].set_ylim(0, 2.1)
axes[1].set_ylabel("G_p / G_r"); axes[1].set_xlabel("step")
axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUT}/cell_traj.png", dpi=150, bbox_inches="tight")
plt.close()
print("  saved cell_traj.png")

print("\nDone. All autonomous plots in", OUT)



"""
run_stimulated.py
=================
Stimulated dynamics study.

During the first T_STIM steps the 22 left boundary field-grid-points receive
an oscillatory signal (mirrored to the right half inside step())

Signal formula:
    signal_left[t, k] = amp[k] * cos(freq[k] * t + phase[k])
    where t = linspace(0, 0.5, T_STIM)   (no 2*pi factor)

After T_STIM steps the boundary stimulus is removed and the system evolves freely.
Step 2 (_update_Gp_clamp) is active only during stimulation, exactly as in training.

Produces:
  output/stimulated/vmem_snapshots.png — Vmem at key moments (stim + free)
  output/stimulated/timeseries.png     — timeseries comparison stim vs auto
  output/stimulated/final_compare.png  — autonomous vs stimulated final pattern
"""

import os
import math
import numpy as np
import torch
import matplotlib.pyplot as plt

from dynamics_t_sparse_gh import (BioelectricLattice, G_r, DT, E_p, E_d,
                               _boundary_indices)

OUT = "output/stimulated"
os.makedirs(OUT, exist_ok=True)

# Scala dei plot presa dalle costanti del modello (E_p/E_d).
VMIN_MV, VMAX_MV = E_p * 1000, E_d * 1000

# -----------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------
PARAMS   = dict(G=0.05, R=4, W=1000.0, B=5e-4)
T_TOTAL  = 20000
T_STIM   = int(0.1 * T_TOTAL)   # first 10% = 2000 steps
SAVE     = 20
ROWS, COLS = 11, 11

# Stimulation signal: stessi range dell'inizializzazione stigmergic in
# train_cfn_mod.py
# amp   [-0.5, 0.5] V   (centrata su zero)
# freq  [100, 1000]
# phase [0, 2π]
np.random.seed(42)
torch.manual_seed(42)

_, left_idx, _ = _boundary_indices(ROWS, COLS)
n_left = len(left_idx)   # 22

amp_np   = np.random.uniform(0.1, 0.5, n_left)
freq_np  = np.random.uniform(100.0, 1000.0, n_left)
phase_np = np.random.uniform(0.0, 2 * np.pi, n_left)

amp   = torch.tensor(amp_np,   dtype=torch.float64)
freq  = torch.tensor(freq_np,  dtype=torch.float64)
phase = torch.tensor(phase_np, dtype=torch.float64)

# Pre-compute signal: [T_STIM, n_left]
# t normalizzato su [0, 0.5], consistent with forward() in train_cfn_mod.py
t_phys = torch.linspace(0.0, 0.5, T_STIM, dtype=torch.float64)

    # [T_stim, n_left]
signal_left = amp.unsqueeze(0) * torch.cos(
        freq.unsqueeze(0) * t_phys.unsqueeze(1) + phase.unsqueeze(0)
    )


def run_stimulated(net, T, T_stim, sig_left, save_every=1):
    """
    T        : total steps
    T_stim   : steps with active boundary stimulus
    sig_left : [T_stim, n_left] precomputed signal tensor
    """
    net.reset()
    vs, gps, ems = [], [], []

    for step in range(T):
        v_left = sig_left[step] if step < T_stim else None
        ex, ey = net.step(v_left_22=v_left)

        if (step + 1) % save_every == 0:
            vs.append(net.v.clone())
            gps.append(net.Gp.clone())
            ems.append(torch.sqrt(ex**2 + ey**2).clone())

    return (
        torch.stack(vs,  dim=0),
        torch.stack(gps, dim=0),
        torch.stack(ems, dim=0),
    )


# -----------------------------------------------------------------------
# Run stimulated
# -----------------------------------------------------------------------
print(f"Running stimulated dynamics (T={T_TOTAL}, T_stim={T_STIM})...")
net_stim = BioelectricLattice(rows=ROWS, cols=COLS, **PARAMS)
v_stim, gp_stim, em_stim = run_stimulated(
    net_stim, T=T_TOTAL, T_stim=T_STIM,
    sig_left=signal_left, save_every=SAVE
)

# -----------------------------------------------------------------------
# Run autonomous for comparison (step() without v_left_22 — no step 2)
# -----------------------------------------------------------------------
print("Running autonomous for comparison...")
net_auto = BioelectricLattice(rows=ROWS, cols=COLS, **PARAMS)
v_auto, gp_auto, em_auto = net_auto.run(T=T_TOTAL, save_every=SAVE)

v_s  = v_stim.numpy();  gp_s = gp_stim.numpy()
v_a  = v_auto.numpy();  gp_a = gp_auto.numpy()
steps = np.arange(1, len(v_s) + 1) * SAVE

# -----------------------------------------------------------------------
# Plot 1: Vmem snapshots at key moments (stimulated)
# -----------------------------------------------------------------------
SNAP       = [100, 500, 1000, 2000, 5000, 10000, 15000, 20000]
SNAP_F     = [s // SAVE - 1 for s in SNAP]

fig, axes = plt.subplots(2, 4, figsize=(13, 7))
fig.suptitle(
    f"Vmem snapshots — stimulated (T_stim={T_STIM})\n"
    "Stigmergic params: W=1000, B=5e-4, R=4",
    fontsize=10
)
for ax, (fi, st) in zip(axes.ravel(), zip(SNAP_F, SNAP)):
    im = ax.imshow(
        v_s[fi].reshape(11, 11) * 1000,
        vmin=VMIN_MV, vmax=VMAX_MV, cmap="gray", origin="upper"
    )
    title = f"t = {st}"
    if st <= T_STIM:
        title += " ◀ stim"
    ax.set_title(title, fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7, label="Vmem (mV)")
plt.savefig(f"{OUT}/vmem_snapshots.png", dpi=150, bbox_inches="tight")
plt.close()
print("  saved vmem_snapshots.png")

# -----------------------------------------------------------------------
# Plot 2: Timeseries comparison — autonomous vs stimulated
# -----------------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
fig.suptitle(
    "std(Vmem) over time: autonomous vs stimulated\n"
    "Stigmergic params: W=1000, B=5e-4, R=4, G=0.05",
    fontsize=11
)

for ax, (v_arr, label, color) in zip(
        axes,
        [(v_a, "Autonomous", "steelblue"),
         (v_s, "Stimulated", "tomato")]):
    ax.plot(steps, v_arr.mean(1)*1000, color=color, lw=1.3, label="mean")
    ax.fill_between(steps, v_arr.min(1)*1000, v_arr.max(1)*1000,
                    alpha=0.15, color=color, label="min/max")
    ax.plot(steps, v_arr.std(1)*1000, color=color, lw=1.5, ls="--", label="std")
    if "Stim" in label:
        ax.axvspan(0, T_STIM, alpha=0.08, color="orange",
                   label=f"stim window (0–{T_STIM})")
    ax.axhline(VMIN_MV, color="gray", ls=":", lw=0.8)
    ax.axhline(VMAX_MV, color="gray", ls="-.", lw=0.8)
    ax.set_ylabel("mV"); ax.set_title(label)
    ax.legend(fontsize=8, ncol=5); ax.grid(alpha=0.3)

axes[-1].set_xlabel("step")
plt.tight_layout()
plt.savefig(f"{OUT}/timeseries.png", dpi=150, bbox_inches="tight")
plt.close()
print("  saved timeseries.png")

# -----------------------------------------------------------------------
# Plot 3: Final pattern comparison — autonomous vs stimulated
# -----------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(9, 5))
fig.suptitle(f"Final Vmem at t={T_TOTAL}\n"
             "Left: autonomous   Right: stimulated", fontsize=11)

for ax, v_arr, label in zip(axes, [v_a, v_s], ["Autonomous", "Stimulated"]):
    im = ax.imshow(v_arr[-1].reshape(11, 11)*1000,
                   vmin=VMIN_MV, vmax=VMAX_MV, cmap="gray", origin="upper")
    ax.set_title(label, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
fig.colorbar(im, ax=axes.tolist(), shrink=0.8, label="Vmem (mV)")
plt.savefig(f"{OUT}/final_compare.png", dpi=150, bbox_inches="tight")
plt.close()
print("  saved final_compare.png")

# -----------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------
print(f"\nSummary at t={T_TOTAL}:")
print(f"  Autonomous:  std={v_a[-1].std()*1000:.2f} mV  "
      f"v=[{v_a[-1].min()*1000:.1f}, {v_a[-1].max()*1000:.1f}] mV")
print(f"  Stimulated:  std={v_s[-1].std()*1000:.2f} mV  "
      f"v=[{v_s[-1].min()*1000:.1f}, {v_s[-1].max()*1000:.1f}] mV")
print(f"\nDone. All stimulated plots in {OUT}/")




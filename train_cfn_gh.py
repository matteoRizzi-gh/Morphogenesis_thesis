import os
import math
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dynamics_t_sparse_gh import (BioelectricLattice, _boundary_indices,
                               build_drive, E_p, E_d, DT)

# ═══════════════════════════════════════════════════════════════════════════════
#  Output
# ═══════════════════════════════════════════════════════════════════════════════
OUTPUT_DIR = 'output'

def out_path(name):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return os.path.join(OUTPUT_DIR, name)


# ═══════════════════════════════════════════════════════════════════════════════
ROWS, COLS   = 11, 11
T_SIM        = 1000
T_STIM_FRAC  = 0.10
T_DRIVE_END  = 0.5

V_HYP_TH = -40e-3
V_DEP_TH = -20e-3

# Target del paper
# NON coincidono con E_p / E_d: il valore iperpolarizzato (-60 mV) sta oltre
# l'equilibrio bistabile del modello (~-50 mV), quindi il target e'
# irraggiungibile per costruzione e la loss ha un pavimento non nullo.
TARGET_HYP = -60.0e-3
TARGET_DEP =  -9.2e-3
V_MID      = 0.5 * (TARGET_HYP + TARGET_DEP)

VARIANTS = {
    'stigmergic': dict(W=1000.0, B=5e-4,
                       amp_range=(0.1, 0.5),
                       freq_range=(100.0, 1000.0),
                       step_sizes=None),
    'mosaic':     dict(W=10.0, B=0.02,
                       amp_range=(0.1, 0.5),
                       freq_range=(10.0, 100.0),
                       step_sizes=(1e-6, 0.1)),
}


# ═══════════════════════════════════════════════════════════════════════════════
#  Regioni e target
# ═══════════════════════════════════════════════════════════════════════════════

def _outer_ring_indices(rows, cols):
    return [r * cols + c
            for r in range(rows) for c in range(cols)
            if r == 0 or r == rows-1 or c == 0 or c == cols-1]

def _face_region_indices(rows=ROWS, cols=COLS):
    skin  = _outer_ring_indices(rows, cols)
    eyes  = [24, 25, 35, 36, 29, 30, 40, 41]
    nose  = [49, 60, 71]
    mouth = [92, 93, 94]
    all_hyp      = set(skin + eyes + nose + mouth)
    interior_idx = [i for i in range(rows * cols) if i not in all_hyp]
    return dict(skin=skin, eyes=eyes, nose=nose, mouth=mouth,
                interior=interior_idx)

def build_target(rows=ROWS, cols=COLS):
    """Stessi 54 indici del target originale (verificati identici a quelli
    degli autori): -60 mV / -9.2 mV."""
    target  = torch.full((rows * cols,), TARGET_DEP, dtype=torch.float64)
    regions = _face_region_indices(rows, cols)
    for name in ['skin', 'eyes', 'nose', 'mouth']:
        for i in regions[name]:
            target[i] = TARGET_HYP
    return target

def drive_cycles(freq):
    return freq.abs() * T_DRIVE_END / (2.0 * math.pi)


# ═══════════════════════════════════════════════════════════════════════════════
#  Loss
# ═══════════════════════════════════════════════════════════════════════════════

REGION_LOSSES = ('partitioned', 'LocalRMSE', 'LocalRMSEweighted',
                 'GeneralLocalRMSE')

def compute_loss(v_eval, target, method='globalmean',
                 regions=None, dGp_eval=None, dv_eval=None):
    if method in REGION_LOSSES and regions is None:
        raise ValueError(f"'{method}' richiede regions=dict")

    if method == 'globalsum':
        return ((target.unsqueeze(0) - v_eval) ** 2).sum().sqrt()

    elif method == 'globalmean':
        return ((target.unsqueeze(0) - v_eval) ** 2).mean().sqrt()

    elif method == 'partitioned':
        loss = 0.0
        for name in ['skin', 'eyes', 'nose', 'mouth']:
            idx  = torch.tensor(regions[name], dtype=torch.long)
            diff = target[idx].unsqueeze(0) - v_eval[:, idx]
            loss = loss + (diff ** 2).sum().clamp(min=1e-12).sqrt() / len(regions[name])
        return loss

    elif method == 'LocalRMSE':
        loss = 0.0
        for name in ['skin', 'eyes', 'nose', 'mouth']:
            idx  = torch.tensor(regions[name], dtype=torch.long)
            diff = target[idx].unsqueeze(0) - v_eval[:, idx]
            loss = loss + ((diff ** 2).sum().clamp(min=1e-12) / len(regions[name])).sqrt()
        return loss

    elif method == 'LocalRMSEweighted':
        weights = {'skin': 1.0, 'eyes': 1.0, 'nose': 1.0, 'mouth': 2.0}
        loss = 0.0
        for name in ['skin', 'eyes', 'nose', 'mouth']:
            idx  = torch.tensor(regions[name], dtype=torch.long)
            diff = target[idx].unsqueeze(0) - v_eval[:, idx]
            loss = loss + weights[name] * ((diff**2).sum().clamp(min=1e-12) / len(regions[name])).sqrt()
        return loss

    elif method == 'GeneralLocalRMSE':
        w_interior = 2.0
        loss = 0.0
        for name in ['skin', 'eyes', 'nose', 'mouth']:
            idx  = torch.tensor(regions[name], dtype=torch.long)
            diff = target[idx].unsqueeze(0) - v_eval[:, idx]
            loss = loss + (diff**2).mean().clamp(min=1e-12).sqrt()
        int_idx  = torch.tensor(regions['interior'], dtype=torch.long)
        diff_int = target[int_idx].unsqueeze(0) - v_eval[:, int_idx]
        loss = loss + w_interior * (diff_int**2).mean().clamp(min=1e-12).sqrt()
        return loss

    elif method == 'globalsumWithdGpol':
        if dGp_eval is None:
            raise ValueError("'globalsumWithdGpol' richiede dGp_eval")
        loss1 = ((target.unsqueeze(0) - v_eval) ** 2).sum().sqrt()
        observed_max  = dGp_eval.abs().max()
        normalization = torch.clamp(observed_max, max=0.05)
        dGp_scaled    = dGp_eval * (normalization / (observed_max + 1e-12))
        loss2 = (dGp_scaled ** 2).sum().sqrt()
        return (loss1 + loss2) / 2.0

    elif method == 'globalsumWithdVmem':
        if dv_eval is None:
            raise ValueError("'globalsumWithdVmem' richiede dv_eval")
        loss1 = ((target.unsqueeze(0) - v_eval) ** 2).sum().sqrt()
        observed_max  = dv_eval.abs().max()
        normalization = torch.clamp(observed_max, max=0.05)
        dv_scaled     = dv_eval * (normalization / (observed_max + 1e-12))
        loss2 = (dv_scaled ** 2).sum().sqrt()
        return (loss1 + loss2) / 2.0

    else:
        raise ValueError(f"Loss sconosciuta: '{method}'")


def region_rmse(v_eval, target, regions):
    v = v_eval.detach()
    out = {}
    for name in ['skin', 'eyes', 'nose', 'mouth', 'interior']:
        idx  = torch.tensor(regions[name], dtype=torch.long)
        diff = target[idx].unsqueeze(0) - v[:, idx]
        out[name] = (diff ** 2).mean().sqrt().item()
    return out


def saturation_fraction(dGp_eval):
    """Proxy della frazione di coppie (cella, passo) con Gp incollata al clamp."""
    return (dGp_eval.detach() == 0).double().mean().item()


# ═══════════════════════════════════════════════════════════════════════════════
#  Forward differenziabile — delega a net.step()
# ═══════════════════════════════════════════════════════════════════════════════

def forward(net, amp, freq, phase, T_sim, T_stim):
    """
    (v_eval, dGp_eval, dv_eval) sugli ultimi 10% dei timestep.
    """
    net.reset()
    drive = build_drive(amp, freq, phase, T_stim)

    eval_steps = max(1, T_sim // 10)
    v_eval, dGp_eval, dv_eval = [], [], []

    for step in range(T_sim):
        v_prev  = net.v.clone()
        Gp_prev = net.Gp.clone()

        net.step(v_left_22=drive[step] if step < T_stim else None)

        if step >= T_sim - eval_steps:
            v_eval.append(net.v.clone())
            dGp_eval.append((net.Gp - Gp_prev).clone())
            dv_eval.append((net.v - v_prev).clone())

    return (torch.stack(v_eval,   dim=0),
            torch.stack(dGp_eval, dim=0),
            torch.stack(dv_eval,  dim=0))


# ═══════════════════════════════════════════════════════════════════════════════
#  Training
# ═══════════════════════════════════════════════════════════════════════════════

def run_training(
        model_variant = 'stigmergic',
        num_iters     = 1000,
        T_sim         = T_SIM,
        lr            = 0.01,
        seed          = 42,
        loss_method   = 'globalsum',
        verbose       = True,
        save_prefix   = None,
) -> dict:

    torch.manual_seed(seed)
    np.random.seed(seed)

    if model_variant not in VARIANTS:
        raise ValueError(f"Variante sconosciuta: '{model_variant}'")
    p = VARIANTS[model_variant]

    net     = BioelectricLattice(rows=ROWS, cols=COLS, G=0.05, R=4,
                                 W=p['W'], B=p['B'])
    T_stim  = max(1, int(T_sim * T_STIM_FRAC))
    target  = build_target(ROWS, COLS)
    regions = _face_region_indices(ROWS, COLS)

    _, left_idx, _ = _boundary_indices(ROWS, COLS)
    n_left = len(left_idx)

    lo_a, hi_a = p['amp_range']
    lo_f, hi_f = p['freq_range']

    if verbose:
        print(f"Modello: {model_variant}  (W={p['W']}, B={p['B']})")
        print(f"T_sim={T_sim}, T_stim={T_stim}, loss={loss_method}")
        n_hyp = (target < V_MID).sum().item()
        print(f"Target: {n_hyp} celle iperpolarizzate a {TARGET_HYP*1000:.1f} mV, "
              f"{ROWS*COLS - n_hyp} depolarizzate a {TARGET_DEP*1000:.1f} mV")
        print(f"        E_p={E_p*1000:.0f} mV  E_d={E_d*1000:.0f} mV  "
              f"(il target iperpolarizzato e' oltre l'equilibrio: irraggiungibile)")
        print(f"Init:  amp [{lo_a}, {hi_a}] V   freq [{lo_f}, {hi_f}]   phase [0, 2pi]")
        f_lim = 2.0 * math.pi * T_stim * min(1.0, DT * p['W'])
        print(f"       cicli sulla finestra: [{lo_f*T_DRIVE_END/(2*math.pi):.2f}, "
              f"{hi_f*T_DRIVE_END/(2*math.pi):.2f}]   limite indicativo freq ~ {f_lim:.0f}")
        print(f"       step_sizes Rprop: "
              f"{p['step_sizes'] if p['step_sizes'] else 'default (1e-6, 50)'}")
        print(f"       output in: {os.path.abspath(OUTPUT_DIR)}\n")

    amp   = nn.Parameter(torch.rand(n_left, dtype=torch.float64) * (hi_a - lo_a) + lo_a)
    freq  = nn.Parameter(torch.rand(n_left, dtype=torch.float64) * (hi_f - lo_f) + lo_f)
    phase = nn.Parameter(torch.rand(n_left, dtype=torch.float64) * 2.0 * math.pi)

    freq_init = freq.detach().clone()
    amp_init  = amp.detach().clone()

    if p['step_sizes'] is not None:
        optimizer = torch.optim.Rprop([amp, freq, phase], lr=lr,
                                      step_sizes=p['step_sizes'])
    else:
        optimizer = torch.optim.Rprop([amp, freq, phase], lr=lr)

    loss_history   = []
    grad_history   = {'amp': [], 'freq': [], 'phase': []}
    amp_history    = []
    freq_drift_history = []
    sat_history    = []
    region_history = []
    best_loss      = float('inf')
    best_iter      = -1
    best_params    = None

    # ── helpers print ────────────────────────────────────────────────────────

    def _print_grid_float(data_1d, label, fmt='{:6.1f}'):
        print(f"  {label}:")
        print("      " + "".join(f"{c:7d}" for c in range(COLS)))
        print("      " + "-" * (COLS * 7))
        for row in range(ROWS):
            row_str = f"  {row:2d} |"
            for col in range(COLS):
                val = data_1d[row * COLS + col]
                if isinstance(val, torch.Tensor):
                    val = val.item()
                row_str += (" " + fmt.format(val))
            print(row_str)
        print()

    def _print_grid_symbol(v_mv_1d):
        hyp_mv, dep_mv = V_HYP_TH * 1000, V_DEP_TH * 1000
        print(f"  Pattern Vmem  (# <{hyp_mv:.0f}mV  ~ intermedio  . >{dep_mv:.0f}mV):")
        for row in range(ROWS):
            row_str = "    "
            for col in range(COLS):
                val = v_mv_1d[row * COLS + col]
                if isinstance(val, torch.Tensor):
                    val = val.item()
                row_str += "#" if val < hyp_mv else ("." if val > dep_mv else "~")
            print(row_str)
        print()

    def _print_param_grid(param_data, label, fmt='{:7.4f}'):
        print(f"  {label} ({n_left} valori bordo sx, dall'alto):")
        for i, v in enumerate(param_data):
            val = v.item() if isinstance(v, torch.Tensor) else v
            print(f"    [{i:2d}]  " + fmt.format(val))
        print()

    def _print_snapshot(it, loss_val, v_eval, dGp_eval,
                        amp, freq, phase, gn_amp, gn_freq, gn_phase, reg, sat):
        print(f"\n{'═'*65}")
        print(f"  ITERAZIONE {it:4d}/{num_iters}   "
              f"loss={loss_val:.6f}   best={best_loss:.6f} (iter {best_iter})")
        print(f"{'═'*65}")

        v_last = v_eval[-1].detach()
        v_mv   = v_last * 1000
        _print_grid_float(v_mv, "Vmem (mV)", fmt='{:6.1f}')
        _print_grid_symbol(v_mv)

        N = ROWS * COLS
        n_hyp = (v_last < V_HYP_TH).sum().item()
        n_dep = (v_last > V_DEP_TH).sum().item()
        print(f"  celle iper (<{V_HYP_TH*1000:.0f}mV): {n_hyp:3d}/{N}   "
              f"depo (>{V_DEP_TH*1000:.0f}mV): {n_dep:3d}/{N}   "
              f"intermedie: {N-n_hyp-n_dep:3d}/{N}")
        print(f"  Vmem  min={v_mv.min():.1f}mV  max={v_mv.max():.1f}mV  "
              f"mean={v_mv.mean():.1f}mV  std={v_mv.std():.1f}mV\n")

        print("  RMSE per regione (mV):  " + "   ".join(
            f"{k}={v*1000:6.2f}" for k, v in reg.items()))
        print(f"  dGp esattamente nulli (proxy clamp saturo): {100*sat:5.1f}%\n")

        dgp = dGp_eval[-1].detach().abs()
        _print_grid_float(dgp * 1e12, "|dGp| (pS)", fmt='{:6.3f}')

        _print_param_grid(amp.data,   "amp   (V)",        fmt='{:7.4f}')
        _print_param_grid(freq.data,  "freq  (adimens.)", fmt='{:8.2f}')
        _print_param_grid(phase.data, "phase (rad)",      fmt='{:7.4f}')

        cyc   = drive_cycles(freq.data)
        drift = (freq.data - freq_init).abs()
        print(f"  cicli sulla finestra:  min={cyc.min():.2f}  max={cyc.max():.2f}  "
              f"media={cyc.mean():.2f}")
        print(f"  drift di freq da init: media={drift.mean():.3f}  max={drift.max():.3f}  "
              f"({100*drift.mean()/(hi_f-lo_f):.3f}% del range)")
        print(f"  media |amp|: {amp.data.abs().mean():.4f} V   "
              f"(init {amp_init.abs().mean():.4f} V, "
              f"x{amp.data.abs().mean()/amp_init.abs().mean():.1f})")
        print(f"  gradienti:  |amp|={gn_amp:.3e}  "
              f"|freq|={gn_freq:.3e}  |phase|={gn_phase:.3e}")

        if len(loss_history) >= 2:
            recent = loss_history[-min(100, len(loss_history)):]
            delta  = recent[-1] - recent[0]
            trend  = "↓" if delta < -1e-6 else ("↑" if delta > 1e-6 else "→")
            print(f"  loss trend (ultimi {len(recent)}): "
                  f"{recent[0]:.6f} → {recent[-1]:.6f}  {trend}  Δ={delta:+.6f}")
        print(f"{'═'*65}\n")

    # ── Loop principale ──────────────────────────────────────────────────────
    for it in range(num_iters):
        optimizer.zero_grad()

        v_eval, dGp_eval, dv_eval = forward(net, amp, freq, phase, T_sim, T_stim)

        loss = compute_loss(
            v_eval, target,
            method   = loss_method,
            regions  = regions  if loss_method in REGION_LOSSES else None,
            dGp_eval = dGp_eval if 'dGpol' in loss_method        else None,
            dv_eval  = dv_eval  if 'dVmem' in loss_method        else None,
        )
        loss.backward()

        gn_amp   = amp.grad.norm().item()   if amp.grad   is not None else 0.0
        gn_freq  = freq.grad.norm().item()  if freq.grad  is not None else 0.0
        gn_phase = phase.grad.norm().item() if phase.grad is not None else 0.0
        grad_history['amp'].append(gn_amp)
        grad_history['freq'].append(gn_freq)
        grad_history['phase'].append(gn_phase)

        reg = region_rmse(v_eval, target, regions)
        sat = saturation_fraction(dGp_eval)
        region_history.append(reg)
        sat_history.append(sat)
        amp_history.append(amp.detach().abs().mean().item())
        freq_drift_history.append((freq.detach() - freq_init).abs().mean().item())

        optimizer.step()

        with torch.no_grad():
            phase.data %= (2.0 * math.pi)

        loss_val = loss.item()
        loss_history.append(loss_val)

        if loss_val < best_loss:
            best_loss   = loss_val
            best_iter   = it
            best_params = dict(amp=amp.detach().clone(),
                               freq=freq.detach().clone(),
                               phase=phase.detach().clone())

        if verbose and (it % 100 == 0 or it == num_iters - 1):
            _print_snapshot(it, loss_val, v_eval, dGp_eval,
                            amp, freq, phase, gn_amp, gn_freq, gn_phase, reg, sat)

    if save_prefix is None:
        save_prefix = f'face_{model_variant}_seed{seed}'

    result = dict(
        variant            = model_variant,
        seed               = seed,
        mechanism          = 'aligned_official',
        loss_method        = loss_method,
        best_loss          = best_loss,
        best_iter          = best_iter,
        best_params        = best_params,
        loss_history       = loss_history,
        grad_history       = grad_history,
        amp_history        = amp_history,
        freq_drift_history = freq_drift_history,
        sat_history        = sat_history,
        region_history     = region_history,
        amp_init           = amp_init,
        freq_init          = freq_init,
        target             = target,
        net_params         = dict(W=p['W'], B=p['B'], G=0.05, R=4,
                                  T_sim=T_sim, T_stim=T_stim),
        init_ranges        = dict(amp=p['amp_range'], freq=p['freq_range'],
                                  step_sizes=p['step_sizes']),
    )
    torch.save(result, out_path(f'{save_prefix}.pt'))

    fig, axes = plt.subplots(2, 1, figsize=(7, 6), tight_layout=True, sharex=True)
    axes[0].plot(loss_history)
    axes[0].set_ylabel(f'Loss ({loss_method})')
    axes[0].set_title(f'Training [allineato] — {model_variant}  (seed={seed})')
    axes[1].semilogy(grad_history['amp'],   label='|grad_amp|',   alpha=0.8)
    axes[1].semilogy(grad_history['freq'],  label='|grad_freq|',  alpha=0.8)
    axes[1].semilogy(grad_history['phase'], label='|grad_phase|', alpha=0.8)
    axes[1].set_ylabel('Gradient norm (log)')
    axes[1].set_xlabel('Iterazione')
    axes[1].legend()
    fig.savefig(out_path(f'{save_prefix}_loss.png'), dpi=120)
    plt.close(fig)

    if verbose:
        fd = freq_drift_history[-1]
        print(f"Best loss: {best_loss:.6f}  raggiunta all'iterazione {best_iter}/{num_iters}")
        print(f"  media |amp|: {amp_init.abs().mean():.4f} -> "
              f"{amp_history[-1]:.4f} V  (x{amp_history[-1]/amp_init.abs().mean():.1f})")
        print(f"  drift medio di freq: {fd:.3f} su un range di {hi_f-lo_f:.0f} "
              f"({100*fd/(hi_f-lo_f):.3f}%)")
        print(f"  saturazione del clamp a fine run: {100*sat_history[-1]:.1f}%")
        print(f"Salvato in {OUTPUT_DIR}/: {save_prefix}.pt  e  {save_prefix}_loss.png")

    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  Visualizzazione
# ═══════════════════════════════════════════════════════════════════════════════

def visualize_result(result, save_prefix='face_result', n_snaps=8):
    p      = result['net_params']
    bp     = result['best_params']
    target = result['target']
    T_sim  = p['T_sim']
    T_stim = p['T_stim']

    net = BioelectricLattice(rows=ROWS, cols=COLS,
                             G=p['G'], R=p['R'], W=p['W'], B=p['B'])
    drive = build_drive(bp['amp'], bp['freq'], bp['phase'], T_stim)

    snap_times = [T_stim] + [
        int(T_stim + (T_sim - T_stim) * i / (n_snaps - 1))
        for i in range(1, n_snaps)]
    snap_times[-1] = T_sim
    snap_steps = {t - 1 for t in snap_times}
    snapshots  = {}

    net.reset()
    with torch.no_grad():
        for step in range(T_sim):
            net.step(v_left_22=drive[step] if step < T_stim else None)
            if step in snap_steps:
                snapshots[step] = net.v.clone()

    keys   = sorted(snap_steps)
    labels = [f't={k+1}' for k in keys]
    n_cols = len(keys) + 1
    fig, axes = plt.subplots(1, n_cols, figsize=(3 * n_cols, 3.5))
    vmin, vmax = E_p - 5e-3, E_d + 5e-3

    for ax, k, lbl in zip(axes[:-1], keys, labels):
        im = ax.imshow(snapshots[k].reshape(ROWS, COLS).numpy(),
                       cmap='gray', vmin=vmin, vmax=vmax, origin='upper')
        ax.set_title(lbl, fontsize=8); ax.axis('off')
    axes[-1].imshow(target.reshape(ROWS, COLS).numpy(),
                    cmap='gray', vmin=vmin, vmax=vmax, origin='upper')
    axes[-1].set_title('Target', fontsize=8); axes[-1].axis('off')
    plt.colorbar(im, ax=axes, shrink=0.6, label='Vmem (V)')
    lm = result.get('loss_method', 'globalmean')
    fig.suptitle(f"Variante: {result['variant']}  |  loss={lm}  |  "
                 f"best={result['best_loss']:.6f}" f"  |  (seed={result['seed']})")
    fig.savefig(out_path(f'{save_prefix}_pattern.png'), dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"Pattern salvato: {OUTPUT_DIR}/{save_prefix}_pattern.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  Multi-run parallelo
# ═══════════════════════════════════════════════════════════════════════════════

def _run_worker(args):
    model_variant, num_iters, T_sim, lr, loss_method, run_idx, save_prefix = args
    return run_training(
        model_variant = model_variant,
        num_iters     = num_iters,
        T_sim         = T_sim,
        lr            = lr,
        loss_method   = loss_method,
        seed          = run_idx,
        save_prefix   = f'{save_prefix}_run{run_idx:03d}',
        verbose       = False,
    )


def run_multi_parallel(
        model_variant = 'stigmergic',
        n_runs        = 100,
        num_iters     = 1000,
        T_sim         = T_SIM,
        lr            = 0.01,
        loss_method   = 'globalsum',
        n_workers     = 5,
        save_prefix   = None,
) -> dict:
    from concurrent.futures import ProcessPoolExecutor, as_completed

    if save_prefix is None:
        save_prefix = f'face_{model_variant}'

    args_list = [(model_variant, num_iters, T_sim, lr, loss_method, i, save_prefix)
                 for i in range(n_runs)]

    best_result, best_loss, completed = None, float('inf'), 0

    print(f"Avvio {n_runs} run su {n_workers} processi  (loss={loss_method})...")
    print(f"Output in: {os.path.abspath(OUTPUT_DIR)}")
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_run_worker, a): a[5] for a in args_list}
        for future in as_completed(futures):
            run_idx = futures[future]
            try:
                result = future.result()
            except Exception as e:
                print(f"  Run {run_idx} ERRORE: {e}")
                continue
            completed += 1
            loss = result['best_loss']
            print(f"  Run {run_idx:3d} completata  loss={loss:.6f}  "
                  f"(best @ iter {result['best_iter']})  ({completed}/{n_runs})")
            if loss < best_loss:
                best_loss, best_result = loss, result
                print(f"  Nuovo best: loss={best_loss:.6f}  (run {run_idx})")

    torch.save(best_result, out_path(f'{save_prefix}_BEST.pt'))
    print(f"\nCompletato  Best loss: {best_loss:.6f}  -> {OUTPUT_DIR}/{save_prefix}_BEST.pt")
    return best_result


# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':

    MODE    = 'single'
    VARIANT = 'mosaic'
    LOSS    = 'globalsum'

    NUM_ITERS = 1000
    T_SIM_RUN = 1000
    LR        = 0.01
    SEED      = 71
    N_RUNS    = 100
    N_WORKERS = 8

    if MODE == 'single':
        result = run_training(
            model_variant = VARIANT,
            num_iters     = NUM_ITERS,
            T_sim         = T_SIM_RUN,
            lr            = LR,
            seed          = SEED,
            loss_method   = LOSS,
            save_prefix   = f'face_{VARIANT}_seed{SEED}',
        )
        visualize_result(result, save_prefix=f'face_{VARIANT}_seed{SEED}')

    elif MODE == 'multi_parallel':
        best = run_multi_parallel(
            model_variant = VARIANT,
            n_runs        = N_RUNS,
            num_iters     = NUM_ITERS,
            T_sim         = T_SIM_RUN,
            lr            = LR,
            loss_method   = LOSS,
            n_workers     = N_WORKERS,
            save_prefix   = f'face_{VARIANT}',
        )
        visualize_result(best, save_prefix=f'face_{VARIANT}_BEST')

    else:
        raise ValueError(f"MODE non riconosciuto: '{MODE}'")

    

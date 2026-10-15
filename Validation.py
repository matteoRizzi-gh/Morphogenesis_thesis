"""
Validation.py
"""

import numpy as np
import torch

from dynamics_t_sparse_gh import (BioelectricLattice, _boundary_indices,
                               _mirror_perm, E_p, E_d)

ROWS, COLS = 11, 11

# Il confronto con il pattern degli autori NON viene eseguito di default.
# Mettere True per lanciarlo da main().
RUN_VALIDATION = True


# ═══════════════════════════════════════════════════════════════════════════
#  Caricamento
# ═══════════════════════════════════════════════════════════════════════════

def load(path):
    """I .dat sono pickle di torch: serve weights_only=False."""
    return torch.load(path, weights_only=False, map_location='cpu')


def hyperparams(d):
    """W, B, G, R e le altre impostazioni del modello."""
    fp, gj = d['fieldParameters'], d['GJParameters']
    f = lambda x: float(x) if torch.is_tensor(x) else x
    return dict(
        W               = f(fp['fieldTransductionWeight']),
        B               = f(fp['fieldTransductionBias']),
        G               = f(gj['GJStrength']),
        R               = fp['fieldScreenSize'],
        gain            = f(fp['fieldTransductionGain']),
        time_constant   = f(fp['fieldTransductionTimeConstant']),
        field_vector    = fp['fieldVector'],
        field_aggregation = fp['fieldAggregation'],
    )


def train_info(d):
    tp = d['trainParameters']
    h  = tp['bestLossHistory']
    return dict(
        loss_method  = tp['lossMethod'],
        best_loss    = tp['bestLoss'],
        best_iter    = h[-1][0],          # iterazione in cui e' arrivato il best
        num_iters    = tp['numLearnIters'],
        lr           = tp['lr'],
        eval_prop    = tp['evalDurationProp'],
        loss_history = h,
    )


def target_vmem(d):
    """Target [121] in Volt, come lo usano gli autori (-0.060 / -0.0092)."""
    return d['trainParameters']['targetVmem'].detach().view(-1).double()


def final_vmem(d):
    """Pattern finale [121] ottenuto dagli autori: e' il termine di paragone."""
    return d['trainParameters']['actualVmem'].detach().view(-1).double()


# ═══════════════════════════════════════════════════════════════════════════
#  Estrazione del drive nel formato di dynamics_t_sparse
# ═══════════════════════════════════════════════════════════════════════════

def boundary_drive_left(d, rows=ROWS, cols=COLS, check=True):
    """
    Ritorna il drive [T_stim, 22] pronto per step(v_left_22=...).
    """
    all_bnd, left_bnd, right_bnd = _boundary_indices(rows, cols)
    cp = d['clampParameters']
    their_idx = np.asarray(cp['clampIndices'][1])
    if not np.array_equal(their_idx, all_bnd):
        raise ValueError("Gli indici di bordo degli autori non coincidono con "
                         "_boundary_indices: il drive non e' trasferibile cosi'.")

    cv = cp['clampValues'].detach().double()          # [T_stim, 44]
    order   = {b: k for k, b in enumerate(all_bnd)}
    pos_left = torch.tensor([order[b] for b in left_bnd], dtype=torch.long)
    v_left = cv[:, pos_left]                          # [T_stim, 22]

    if check:
        mirror = torch.tensor(_mirror_perm(rows, cols), dtype=torch.long)
        lmap = {v: k for k, v in enumerate(left_bnd)}
        rmap = {v: k + len(left_bnd) for k, v in enumerate(right_bnd)}
        lr_to_all = torch.tensor([lmap[a] if a in lmap else rmap[a]
                                  for a in all_bnd], dtype=torch.long)
        rebuilt = torch.cat([v_left, v_left[:, mirror]], dim=1)[:, lr_to_all]
        err = (rebuilt - cv).abs().max().item()
        if err > 1e-12:
            raise ValueError(f"Il drive non e' mirror-simmetrico secondo la "
                             f"regola del modello (errore {err:.3e}): andrebbe "
                             f"iniettato sui 44 punti direttamente.")
    return v_left


# ═══════════════════════════════════════════════════════════════════════════
#  Stampa
# ═══════════════════════════════════════════════════════════════════════════

def _grid(v121, hyp=-0.04, dep=-0.02):
    g = v121.view(ROWS, COLS)
    return "\n".join("    " + "".join(
        "#" if g[r, c] < hyp else ("." if g[r, c] > dep else "~")
        for c in range(COLS)) for r in range(ROWS))


def summarize(d, name):
    hp, ti = hyperparams(d), train_info(d)
    tgt, act = target_vmem(d), final_vmem(d)
    drive = boundary_drive_left(d)

    print(f"\n{'='*64}\n  {name}\n{'='*64}")
    print(f"  W={hp['W']}  B={hp['B']:.6f}  G={hp['G']}  R={hp['R']}")
    print(f"  gain={hp['gain']}  tau={hp['time_constant']}  "
          f"fieldVector={hp['field_vector']}  aggregation={hp['field_aggregation']}")
    print(f"  -> prefattore 10/tau = {10.0/hp['time_constant']:.3f}"
          f"   (se 1.0, dp = -Gp + (2*sigmoid(gain*e + B) - 1)*W)")

    print(f"\n  training: loss={ti['loss_method']}  best={ti['best_loss']:.6f}")
    print(f"            best raggiunto a iter {ti['best_iter']} "
          f"su un budget di {ti['num_iters']}   lr={ti['lr']}")
    h = ti['loss_history']; span = h[0][1] - h[-1][1]
    for frac in (0.5, 0.9, 0.99):
        it = next(i for i, l in h if l <= h[0][1] - frac*span)
        print(f"            {int(frac*100):3d}% del guadagno totale a iter {it}")

    print(f"\n  drive di bordo: {tuple(drive.shape)}  (T_stim={drive.shape[0]} step)")
    print(f"            |amp| max={drive.abs().max():.3f} V   "
          f"mediana dei massimi per punto={drive.abs().max(0).values.median():.4f} V")

    vals = sorted(set(np.round(tgt.numpy(), 5)))
    n_hyp = int((tgt < (tgt.max()+tgt.min())/2).sum())
    print(f"\n  target: valori {vals} V   ({n_hyp} celle iperpolarizzate)")
    print(f"          confronto con dynamics_t_sparse: E_p={E_p} V  E_d={E_d} V")
    print("  target (loro):"); print(_grid(tgt, hyp=-0.03, dep=-0.03))
    print(f"\n  pattern finale ottenuto dagli autori "
          f"(min={act.min():.4f}  max={act.max():.4f} V):")
    print(_grid(act))
    print(f"  RMSE(finale, target) = "
          f"{torch.sqrt(((act-tgt.double())**2).mean()):.6f}")


# ═══════════════════════════════════════════════════════════════════════════
#  Confronto col modello locale (non eseguito di default)
# ═══════════════════════════════════════════════════════════════════════════

def validate(d, name, T_sim=1000):
    """
    Rigira il drive degli autori dentro dynamics_t_sparse e confronta il
    pattern risultante con il loro actualVmem.
    """
    hp = hyperparams(d)
    drive = boundary_drive_left(d)
    T_stim = drive.shape[0]

    net = BioelectricLattice(rows=ROWS, cols=COLS,
                             G=hp['G'], R=hp['R'], W=hp['W'], B=hp['B'])
    net.reset()
    eval_n, vs = max(1, T_sim // 10), []
    with torch.no_grad():
        for s in range(T_sim):
            net.step(v_left_22=drive[s] if s < T_stim else None)
            if s >= T_sim - eval_n:
                vs.append(net.v.clone())
    V = torch.stack(vs)

    act = final_vmem(d)
    for label, mine in (("ultimo step", V[-1]), ("media finestra", V.mean(0))):
        diff = (mine - act).abs()
        print(f"  {name} [{label}]:  E_p={E_p*1000:.0f} mV"
              f"  diff media={diff.mean()*1000:7.3f} mV"
              f"  max={diff.max()*1000:7.3f} mV")
        print(f"  {name} [{label}]: diff media={diff.mean():.3e} V"
              f"  max={diff.max():.3e} V")
    print(f"  pattern ottenuto localmente:"); print(_grid(V[-1]))

    return V


# ═══════════════════════════════════════════════════════════════════════════

FILES = [("MosaicModelParameters.dat",     "MOSAIC  (Fig. 7)"),
         ("StigmergicModelParameters.dat", "STIGMERGIC  (Fig. 8)")]


def main():
    models = {}
    for path, name in FILES:
        d = load(path)
        models[name] = d
        summarize(d, name)

    print(f"\n{'='*64}\n  DRIVE ESTRATTI\n{'='*64}")
    for path, name in FILES:
        drive = boundary_drive_left(models[name])
        out = path.replace('ModelParameters.dat', '_drive_left.pt')
        torch.save(drive, out)
        print(f"  {name:22} -> {out}   shape={tuple(drive.shape)}")
    print("\n  Si passano a step(v_left_22=drive[t]) un passo alla volta.")

    if RUN_VALIDATION:
        print(f"\n{'='*64}\n  CONFRONTO COL MODELLO LOCALE\n{'='*64}")
        for path, name in FILES:
            validate(models[name], name)
    else:
        print("\n  RUN_VALIDATION=False: il confronto col modello locale non e' "
              "stato eseguito.\n  Metti True in cima al file per lanciarlo.")
    figura_replica(models)
    return models


# ═══════════════════════════════════════════════════════════════════════════
#  Figura della slide 5
# ═══════════════════════════════════════════════════════════════════════════

def figura_replica(models, outdir='figure', larghezza_cm=6.5, altezza_cm=4.5):
    """Pattern degli autori e replica con i loro segnali, accanto al target.
    Scrive figure/fig_replica_pattern.pdf, 6.5 x 4.5 cm."""
    import os
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    righe = []
    for path, name in FILES:
        d = models[name]
        V = validate(d, name)                    # stesso replay della validazione
        mio, loro = V[-1].numpy(), final_vmem(d).numpy()
        nome = 'mosaico' if 'MOSAIC' in name else 'stigmergico'
        print(f"  {nome}: differenza massima {np.abs(mio - loro).max():.1e} V")
        righe.append((nome, loro, mio))
    target = target_vmem(models[FILES[0][1]]).numpy()

    plt.rcParams.update({'font.size': 7, 'axes.titlesize': 7, 'axes.labelsize': 7})
    vmin, vmax = E_p - 5e-3, E_d + 5e-3
    fig = plt.figure(figsize=(larghezza_cm / 2.54, altezza_cm / 2.54))
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 0.85, 0.07], wspace=0.12, hspace=0.12,
                          left=0.09, right=0.84, top=0.90, bottom=0.04)
    for r, (nome, loro, mio) in enumerate(righe):
        for c, (x, tit) in enumerate(((loro, 'autori'), (mio, 'replica'))):
            a = fig.add_subplot(gs[r, c])
            im = a.imshow(x.reshape(ROWS, COLS), cmap='gray', vmin=vmin, vmax=vmax)
            a.set_xticks([]); a.set_yticks([])
            if r == 0:
                a.set_title(tit, pad=2)
            if c == 0:
                a.set_ylabel(nome, labelpad=2)
    a = fig.add_subplot(gs[:, 2])
    a.imshow(target.reshape(ROWS, COLS), cmap='gray', vmin=vmin, vmax=vmax)
    a.set_xticks([]); a.set_yticks([]); a.set_title('target', pad=2)
    cb = fig.colorbar(im, cax=fig.add_subplot(gs[:, 3]), ticks=[E_p, E_d])
    cb.ax.set_yticklabels([f'${E_p * 1e3:.0f}$', f'${E_d * 1e3:.0f}$'])
    cb.ax.tick_params(length=2, pad=1)
    cb.set_label(r'$V_\mathrm{mem}$ (mV)', labelpad=1)
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, 'fig_replica_pattern.pdf')
    fig.savefig(out)
    plt.close(fig)
    print("  figura salvata in", out)

if __name__ == '__main__':
    main()



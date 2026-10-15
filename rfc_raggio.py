"""
rfc_raggio.py
=============
Separa, nella dipendenza dell'RFC dal raggio d'azione, il contributo della
mappa di misura da quello della dinamica (Oss. "Un confondimento che l'RFC non
separa").

  (A) dinamica fissa, mappa variabile: le serie di Vmem sono simulate a R=4;
      il campo osservabile e' ricalcolato con raggio R' per ogni R'.
  (B) dinamica e mappa variabili: le serie sono simulate a R, e il campo
      osservabile e' calcolato con lo stesso R, come nell'articolo.

"""

import os
import sys
import numpy as np
import torch

import Rfc_and_ID as RF
from dynamics_t_sparse_gh import (BioelectricLattice, C, k_e, e_r, EPSILON,
                                  FIELD_STRENGTH)

RADII     = list(range(1, 22))          # tutti i raggi; la Fig. 5 usa 1, 4, 10, 15, 21
N_WORKERS = max(1, (os.cpu_count() or 2) // 2)
R_DYN     = 4                           # raggio della dinamica fissa in (A)
N_SAMPLES = 32
T         = 5000
G         = 0.05
SEED      = 0
N_BOOT    = 10_000
MODELS    = {'stigmergico': (1000.0, 5e-4), 'mosaic': (10.0, 0.02)}
OUT_DIR   = 'output_rfc_raggio'


# ---------------------------------------------------------------------------
#  Mappa di misura Phi_R = A_R o E_R, in forma matriciale
# ---------------------------------------------------------------------------

def phi_matrices(R_obs):
    """Matrici della mappa al raggio R_obs, dalla stessa geometria del modello.

    Mx, My [N_f, N]: E_x = fc * Mx @ q,  E_y = fc * My @ q,  q = C v
                     
    Agg    [N, N_f]: A_c = sum_{f: c in A_f(R)} E_f / 4
                    
    """
    net = BioelectricLattice(rows=11, cols=11, G=G, R=R_obs, W=1.0, B=0.0)
    f = net.endo_f_idx.numpy()
    c = net.endo_c_idx.numpy()
    Mx = np.zeros((net.N_f, net.N))
    My = np.zeros((net.N_f, net.N))
    np.add.at(Mx, (f, c), net.endo_coeff_x.numpy())
    np.add.at(My, (f, c), net.endo_coeff_y.numpy())
    Agg = np.zeros((net.N, net.N_f))
    np.add.at(Agg, (c, f), 1.0 / RF.NUM_FIELD_NEIGHBORS)
    return Mx, My, Agg


def phi(V, mats):
    """Campo osservabile Phi_R(v) per ogni riga di V [T, N]."""
    Mx, My, Agg = mats
    fc = FIELD_STRENGTH * (k_e / e_r)
    q = C * V
    ex = fc * q @ Mx.T
    ey = fc * q @ My.T
    eV = np.sqrt(ex**2 + ey**2 + EPSILON)
    return eV @ Agg.T


def rfc_of(A, V):
    """RFC di una traiettoria, in punti percentuali, e le due quote."""
    pa = RF.pca_explained(A).sum()
    pv = RF.pca_explained(V).sum()
    return 100.0 * (pa - pv), 100.0 * pa, 100.0 * pv


def boot_ci(x, n_boot=N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=np.float64)
    means = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


# ---------------------------------------------------------------------------
#  Calcolo
# ---------------------------------------------------------------------------

def _path(model, R_sim):
    os.makedirs(OUT_DIR, exist_ok=True)
    return os.path.join(OUT_DIR, f'{model}_Rsim{R_sim}.npz')


def mask_key(R_val):
    """Firma della maschera d'azione: raggi con la stessa firma danno la stessa
    mappa e la stessa dinamica."""
    net = BioelectricLattice(rows=11, cols=11, G=G, R=R_val, W=1.0, B=0.0)
    return (tuple(net.endo_f_idx.tolist()), tuple(net.endo_c_idx.tolist()))


def _work(args):
    torch.set_num_threads(1)
    model, R_sim, i, R_obs_list = args
    W, B = MODELS[model]
    ts = RF.run_timeseries(W, B, R_sim, G, seed=SEED * 100000 + i, T=T,
                           homogeneous=(i == 0))
    V = ts['vmem']
    return i, {R_obs: rfc_of(phi(V, phi_matrices(R_obs)), V)
               for R_obs in R_obs_list}


def compute_all(models=None, radii=None, n_workers=N_WORKERS):
    """(A) serie a R=4 con tutte le mappe; (B) serie a R con la mappa a R."""
    models = models or list(MODELS)
    radii = radii or RADII
    keys = {R_val: mask_key(R_val) for R_val in radii}
    jobs, todo = [], []
    for model in models:
        for R_sim in radii:
            if os.path.exists(_path(model, R_sim)):
                if R_sim != R_DYN:
                    continue
                have = set(np.load(_path(model, R_sim)).files)
                if all(f'Robs{r}' in have for r in radii):
                    continue                     # (A) gia' completo
                # file della versione con cinque raggi: si ricalcola
            first = next(r for r in radii if keys[r] == keys[R_sim])
            if first != R_sim and R_sim != R_DYN:
                continue                         # copiato dopo da 'first'
            R_obs_list = radii if R_sim == R_DYN else [R_sim]
            todo.append((model, R_sim, R_obs_list))
            jobs += [(model, R_sim, i, R_obs_list) for i in range(N_SAMPLES)]
    results = {}
    if jobs:
        print(f'  {len(jobs)} traiettorie su {n_workers} processi', flush=True)
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            for n, (args, (i, res)) in enumerate(
                    zip(jobs, ex.map(_work, jobs, chunksize=4)), 1):
                results[(args[0], args[1], i)] = res
                if n % 32 == 0:
                    print(f'    {n}/{len(jobs)}', flush=True)
    for model, R_sim, R_obs_list in todo:
        out = {R_obs: [results[(model, R_sim, i)][R_obs]
                       for i in range(N_SAMPLES)] for R_obs in R_obs_list}
        np.savez(_path(model, R_sim),
                 **{f'Robs{R_obs}': np.array(v) for R_obs, v in out.items()})
    # raggi con maschera gia' vista: stessi risultati del primo raggio equivalente
    for model in models:
        for R_sim in radii:
            first = next(r for r in radii if keys[r] == keys[R_sim])
            if first != R_sim and not os.path.exists(_path(model, R_sim)):
                d = np.load(_path(model, first))
                np.savez(_path(model, R_sim), **{f'Robs{R_sim}': d[f'Robs{first}']})


def load(model, R_sim, R_obs):
    return np.load(_path(model, R_sim))[f'Robs{R_obs}']   # [n, 3]


def report():
    for model in MODELS:
        print(f'\n{model}   (G={G}, {N_SAMPLES} traiettorie di {T} passi)')
        print(f'{"R":>4} | {"(A) dinamica a R=4, mappa a R":>34} |'
              f' {"(B) dinamica e mappa a R":>34} | {"prime 3 comp. in (B)":>22}')
        print(f'{"":>4} | {"RFC media [IC 95%]":>34} | {"RFC media [IC 95%]":>34} |'
              f' {"campo":>10} {"Vmem":>10}')
        for R_obs in RADII:
            a = load(model, R_DYN, R_obs)
            b = load(model, R_obs, R_obs)
            ca, cb = boot_ci(a[:, 0]), boot_ci(b[:, 0])
            print(f'{R_obs:4d} | {a[:,0].mean():+8.2f}%  [{ca[0]:+7.2f}, {ca[1]:+7.2f}]'
                  f'{"":6s} | {b[:,0].mean():+8.2f}%  [{cb[0]:+7.2f}, {cb[1]:+7.2f}]'
                  f'{"":6s} | {b[:,1].mean():9.2f}% {b[:,2].mean():9.2f}%')


def figure(path='figures/rfc_raggio.pdf'):

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4), sharex=True)
    for ax, model in zip(axes, MODELS):

        for R_sim_of, lab, st in ((lambda R_obs: R_DYN, 'dinamica a $R=4$', 'o-'),
                                  (lambda R_obs: R_obs, 'dinamica a $R$', 's-')):
            m, lo, hi = [], [], []
            for R_obs in RADII:
                x = load(model, R_sim_of(R_obs), R_obs)[:, 0]
                ci = boot_ci(x)
                m.append(x.mean()); lo.append(ci[0]); hi.append(ci[1])
            ax.plot(RADII, m, st, ms=3.5, lw=1.2, label=lab)
            ax.fill_between(RADII, lo, hi, alpha=0.25)
        ax.axvline(R_DYN, color='0.6', lw=0.8, ls=':')
        ax.axhline(0, color='0.8', lw=0.6)
        ax.set_title(model)
        ax.set_xlabel('raggio d\'azione $R$')
        ax.set_xticks([1, 4, 10, 15, 21])
    axes[0].set_ylabel('RFC (\\%)')
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    print(f'  figura salvata in {path}')

 
def figure_slide(outdir='figure', larghezza_cm=13.0, altezza_cm=3.9):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8,
                         'axes.labelsize': 8, 'xtick.labelsize': 7,
                         'ytick.labelsize': 7, 'legend.fontsize': 7})
    titoli = {'stigmergico': 'regime forte', 'mosaic': 'regime debole'}
    curve = ((lambda R_obs: R_DYN, 'dinamica a $R=4$', 'o-', 'C0'),
             (lambda R_obs: R_obs, 'dinamica a $R$',   's-', 'C1'))
 
    # medie e intervalli bootstrap, calcolati una volta per entrambi i passi
    dati = {}
    for model in MODELS:
        for k, (R_sim_of, _, _, _) in enumerate(curve):
            m, lo, hi = [], [], []
            for R_obs in RADII:
                x = load(model, R_sim_of(R_obs), R_obs)[:, 0]
                ci = boot_ci(x)
                m.append(x.mean()); lo.append(ci[0]); hi.append(ci[1])
            dati[(model, k)] = (np.array(m), np.array(lo), np.array(hi))
 
    # limiti comuni ai due passi, calcolati su entrambe le curve
    lim = {}
    for model in MODELS:
        basso = min(dati[(model, k)][1].min() for k in range(2))
        alto = max(dati[(model, k)][2].max() for k in range(2))
        pad = 0.08 * (alto - basso)
        lim[model] = (min(basso, 0.0) - pad, alto + pad)
 
    os.makedirs(outdir, exist_ok=True)
    for passo in (1, 2):
        fig, axes = plt.subplots(1, 2, figsize=(larghezza_cm / 2.54,
                                                altezza_cm / 2.54))
        for ax, model in zip(axes, MODELS):
            for k in range(passo):
                _, lab, st, col = curve[k]
                m, lo, hi = dati[(model, k)]
                ax.plot(RADII, m, st, ms=2.2, lw=1.0, color=col, label=lab)
                ax.fill_between(RADII, lo, hi, color=col, alpha=0.25, lw=0)
            ax.axvline(R_DYN, color='0.6', lw=0.7, ls=':')
            ax.axhline(0, color='0.8', lw=0.5)
            ax.set_xlim(0.5, 21.5)
            ax.set_ylim(*lim[model])
            ax.set_xticks([1, 4, 10, 15, 21])
            ax.set_title(titoli[model])
            ax.set_xlabel("raggio d'azione $R$")
        axes[0].set_ylabel('RFC (%)')
        axes[0].legend(frameon=False, loc='center right')
        fig.tight_layout(pad=0.3, w_pad=1.0)
        fig.savefig(os.path.join(outdir, f'fig_rfc_raggio_{passo}.pdf'))
        plt.close(fig)
    print(f'  figure per le slide salvate in {outdir}/')

if __name__ == '__main__':
    torch.set_num_threads(1)
    compute_all()
    report()
    figure()
    figure_slide()
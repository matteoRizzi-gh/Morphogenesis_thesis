"""
timing_map.py
=============
Mappa di fase con il TEMPO DI CONVERGENZA al posto del conteggio degli stati.

"""

import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.stats import spearmanr

import Rfc_and_ID as R
from dynamics_t_sparse_gh import BioelectricLattice, G_r, V_INIT
from fixed_point import step_pure, stability, SAT_EDGE

FIGDIR = 'figures'
TOL    = 1e-14
T_MAX  = 100000        # oltre i 50000 del run precedente: 22634 era vicino


def figpath(name):
    os.makedirs(FIGDIR, exist_ok=True)
    return os.path.join(FIGDIR, name)


def _sat_frac(Gp):
    """Frazione di celle troncate dal clamp."""
    return float(((Gp <= 0.0) | (Gp >= 2.0 * G_r)).double().mean())


# ═══════════════════════════════════════════════════════════════════════
#  Traccia di una singola convergenza
# ═══════════════════════════════════════════════════════════════════════

def trace_convergence(net, v0=None, Gp0=None, T_max=T_MAX, tol=TOL,
                      sat_target=1.0):
    """
    Itera registrando DUE tempi:
      t_sat  = primo passo in cui la frazione di Gp saturate raggiunge
               sat_target (default: tutte le celle)
      t_conv = passo in cui lo spostamento scende sotto tol
    e d_sat, lo spostamento nell'istante della saturazione
    """
    v  = (torch.full((net.N,), V_INIT, dtype=torch.float64)
          if v0 is None else v0.clone())
    Gp = (torch.full((net.N,), 1.0*G_r, dtype=torch.float64)
          if Gp0 is None else Gp0.clone())
    t_sat, d_sat, d = None, None, float('inf')
    with torch.no_grad():
        for t in range(1, T_max + 1):
            v_new, Gp_new = step_pure(v, Gp, net)
            d = float(torch.max((v_new - v).abs().max(),
                                (Gp_new - Gp).abs().max() / G_r))
            v, Gp = v_new, Gp_new
            if t_sat is None and _sat_frac(Gp) >= sat_target - 1e-12:
                t_sat, d_sat = t, d
            if d < tol:
                return dict(converged=True, t_conv=t, t_sat=t_sat,
                            d_sat=d_sat, last_step=d, v=v, Gp=Gp,
                            frac_sat=_sat_frac(Gp))
    return dict(converged=False, t_conv=None, t_sat=t_sat, d_sat=d_sat,
                last_step=d, v=v, Gp=Gp, frac_sat=_sat_frac(Gp))


# ═══════════════════════════════════════════════════════════════════════
#  Una cella della griglia
# ═══════════════════════════════════════════════════════════════════════

def analyze_cell_timing(W, B, Rr=4, G=0.05, n_starts=5, T_max=T_MAX,
                        tol=TOL, seed=0, verbose=True):
    net = BioelectricLattice(rows=11, cols=11, G=G, R=Rr, W=W, B=B)
    runs = []
    for i in range(n_starts):
        if i == 0:
            v0, Gp0 = None, None                       # omogenea
        else:
            g = torch.Generator().manual_seed(seed*1000 + i)
            v0  = torch.full((net.N,), V_INIT, dtype=torch.float64)
            Gp0 = torch.rand(net.N, generator=g, dtype=torch.float64)*2.0*G_r
        runs.append(trace_convergence(net, v0, Gp0, T_max, tol))

    conv = [r for r in runs if r['converged']]
    tc   = np.array([r['t_conv'] for r in conv], dtype=float)
    ts   = np.array([r['t_sat'] for r in runs
                     if r['t_sat'] is not None], dtype=float)

    out = dict(W=W, B=B, R=Rr, G=G, n_starts=n_starts, T_max=T_max,
               n_converged=len(conv),
               t_conv_min=float(tc.min()) if len(tc) else np.nan,
               t_conv_med=float(np.median(tc)) if len(tc) else np.nan,
               t_conv_max=float(tc.max()) if len(tc) else np.nan,
               t_sat_min=float(ts.min()) if len(ts) else np.nan,
               t_sat_med=float(np.median(ts)) if len(ts) else np.nan,
               t_sat_max=float(ts.max()) if len(ts) else np.nan,
               n_sat=len(ts),
               frac_sat_final=float(np.mean([r['frac_sat'] for r in runs])))

    if conv:
        st = stability(net, conv[0]['v'], conv[0]['Gp'])
        out['rho'] = st['spectral_radius']
        out['n_gp_frozen'] = st['n_gp_frozen']
        ds = [r['d_sat'] for r in conv if r['d_sat'] is not None]
        if ds and 0 < out['rho'] < 1:
            out['d_sat_med'] = float(np.median(ds))
            out['relax_pred_from_sat'] = float(np.log(tol / out['d_sat_med']) /
                                               np.log(out['rho']))
        else:
            out['d_sat_med'] = np.nan
            out['relax_pred_from_sat'] = np.nan
        out['relax_obs_from_sat'] = (out['t_conv_med'] - out['t_sat_med']
                                     if np.isfinite(out['t_sat_med'])
                                     else np.nan)
    else:
        out.update(rho=np.nan, n_gp_frozen=-1, d_sat_med=np.nan,
                   relax_pred_from_sat=np.nan, relax_obs_from_sat=np.nan)

    if verbose:
        tcs = ("mai" if not conv else
               f"{out['t_conv_med']:.0f} "
               f"[{out['t_conv_min']:.0f}-{out['t_conv_max']:.0f}]")
        tsv = out['t_sat_med'] if out['n_sat'] else float('nan')
        print(f"  W={W:>6.0f} B={B:.4f}  converge {out['n_converged']}/{n_starts}  "
              f"t_conv={tcs:>22}  t_sat={tsv:>8.0f}  "
              f"rilass. da t_sat: oss={out['relax_obs_from_sat']:>7.1f} "
              f"pred={out['relax_pred_from_sat']:>7.1f}  "
              f"rho={out['rho']:.6f}")
    return out



# ═══════════════════════════════════════════════════════════════════════
#  Griglia
# ═══════════════════════════════════════════════════════════════════════

def _init_worker():
    torch.set_num_threads(1)


def _work(a):
    torch.set_num_threads(1)
    W, B, Rr, G, n_starts, T_max, tol, seed = a
    return analyze_cell_timing(W, B, Rr, G, n_starts, T_max, tol, seed,
                               verbose=False)


def grid_timing(Ws=R.PAPER_W, Bs=R.PAPER_B, Rr=4, G=0.05, n_starts=5,
                T_max=T_MAX, tol=TOL, seed=0, n_workers=None,
                save_as='timing_map.pt'):
    jobs = [(W, B, Rr, G, n_starts, T_max, tol, seed) for W in Ws for B in Bs]
    print(f"Griglia dei tempi: {len(jobs)} celle x {n_starts} partenze "
          f"(T_max={T_max})")
    res = []
    if n_workers and n_workers > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=n_workers,
                                 initializer=_init_worker) as ex:
            futs = [ex.submit(_work, j) for j in jobs]
            for c, f in enumerate(as_completed(futs), 1):
                res.append(f.result())
                print(f"  [{c}/{len(jobs)}] fatta")
                torch.save(res, R.out_path(save_as))
    else:
        for c, j in enumerate(jobs, 1):
            res.append(_work(j)); print(f"  [{c}/{len(jobs)}] fatta")
            torch.save(res, R.out_path(save_as))
    res.sort(key=lambda r: (r['W'], r['B']))
    print_timing(res)
    check_decomposition(res)
    torch.save(res, R.out_path(save_as))
    return res


def print_timing(res):
    print(f"\n{'='*128}\n  MAPPA DEI TEMPI  (R={res[0]['R']}, G={res[0]['G']}, "
          f"{res[0]['n_starts']} partenze)\n{'='*128}")
    print(f"  {'W':>7}{'B':>9}{'conv.':>8}{'t_conv med':>12}"
          f"{'min-max':>16}{'t_sat med':>11}{'min-max':>16}"
          f"{'rilass.':>9}{'rho':>11}{'congelate':>11}")
    for r in res:
        tc = "mai" if not r['n_converged'] else f"{r['t_conv_med']:.0f}"
        mm = ("n/a" if not r['n_converged']
              else f"{r['t_conv_min']:.0f}-{r['t_conv_max']:.0f}")
        ts = "n/a" if not r['n_sat'] else f"{r['t_sat_med']:.0f}"
        sm = ("n/a" if not r['n_sat']
              else f"{r['t_sat_min']:.0f}-{r['t_sat_max']:.0f}")
        rl = ("n/a" if not np.isfinite(r['relax_obs_from_sat'])
              else f"{r['relax_obs_from_sat']:.0f}")
        print(f"  {r['W']:7.0f}{r['B']:9.4f}"
              f"{r['n_converged']:>4}/{r['n_starts']:<3}{tc:>12}{mm:>16}"
              f"{ts:>11}{sm:>16}{rl:>9}{r['rho']:>11.6f}"
              f"{r['n_gp_frozen']:>11d}")
    print("\n  'rilass.' = t_conv - t_sat, cioe' il tempo che resta DOPO che")
    print("  Gp e' satura ovunque. ")



def check_decomposition(res):
    """t_conv = t_sat + rilassamento, con rilassamento costante?"""
    ok = [r for r in res if np.isfinite(r['relax_obs_from_sat'])
          and np.isfinite(r['relax_pred_from_sat'])]
    if not ok:
        print("\n  (nessuna cella con entrambi i tempi definiti)")
        return None

    obs  = np.array([r['relax_obs_from_sat'] for r in ok])
    pred = np.array([r['relax_pred_from_sat'] for r in ok])
    tsat = np.array([r['t_sat_med'] for r in ok])

    print(f"\n{'='*84}\n  DECOMPOSIZIONE  t_conv = t_sat + rilassamento\n{'='*84}")
    print(f"  celle con entrambi i tempi: {len(ok)}")
    print(f"  t_sat varia da {tsat.min():.0f} a {tsat.max():.0f} passi "
          f"(fattore {tsat.max()/tsat.min():.0f})")
    print(f"  rilassamento osservato: mediana {np.median(obs):.1f}  "
          f"intervallo [{obs.min():.0f}, {obs.max():.0f}]  "
          f"dev.st. {obs.std():.1f}")
    print(f"  rilassamento predetto da rho: mediana {np.median(pred):.1f}  "
          f"intervallo [{pred.min():.0f}, {pred.max():.0f}]")
    err = np.abs(obs - pred)
    print(f"  scarto |osservato - predetto|: mediana {np.median(err):.1f}  "
          f"max {err.max():.1f}")

    cv = obs.std() / obs.mean() if obs.mean() else np.nan
    rho_s = float(spearmanr(tsat, obs).statistic) if len(obs) >= 3 else np.nan
    n_slow = min(4, len(obs))
    order = np.argsort(tsat)
    slow, fast = obs[order][-n_slow:], obs[order][:n_slow]

    print(f"\n  coefficiente di variazione del rilassamento: {100*cv:.1f}%")
    print(f"  Spearman(t_sat, rilassamento) = {rho_s:+.3f}")
    print(f"  residuo sulle {n_slow} celle piu' VELOCI: "
          f"[{fast.min():.0f}, {fast.max():.0f}]")
    print(f"  residuo sulle {n_slow} celle piu' LENTE:  "
          f"[{slow.min():.0f}, {slow.max():.0f}]  "
          f"(CV {100*slow.std()/slow.mean():.1f}%)")

    if cv < 0.20:
        verdict = 'costante'
        print("\n  -> COSTANTE entro il 20%: la decomposizione regge, il tempo")
        print("     totale e' governato da quanto ci mette Gp a saturare.")
    elif np.isfinite(rho_s) and rho_s < -0.7:
        verdict = 'monotono'
        print("\n  -> NON costante, ma DECRESCE monotonamente in t_sat.")
    else:
        verdict = 'senza struttura'
        print("\n  -> NON costante e SENZA struttura in t_sat: il meccanismo e'")
        print("     incompleto, dopo la saturazione succede qualcosa che")
        print("     dipende ancora da W o da B.")


    pairs = [(a, b) for i, a in enumerate(ok) for b in ok[i+1:]
             if (a['t_sat_med'] > 0 and b['t_sat_med'] > 0
                 and abs(np.log10(a['t_sat_med'] / b['t_sat_med'])) < 0.1)]
    if pairs:
        dres = np.array([abs(a['relax_obs_from_sat'] - b['relax_obs_from_sat'])
                         for a, b in pairs])
        dW = np.array([max(a['W'], b['W']) / min(a['W'], b['W'])
                       for a, b in pairs])
        print(f"\n  coppie con t_sat entro il 25%: {len(pairs)}")
        print(f"  differenza di residuo: mediana {np.median(dres):.1f} passi, "
              f"max {dres.max():.1f}")
        print(f"  a fronte di rapporti W fino a {dW.max():.0f}x")
        if dres.max() < 0.15 * np.median(obs):
            print(f"  -> il residuo e' funzione del SOLO t_sat: celle con W")
            print(f"     diversi fino a {dW.max():.0f}x e t_sat simile danno lo")
            print(f"     stesso residuo. Dopo la saturazione non resta nessuna dipendenza dai parametri")
        else:
            print(f"  -> il residuo NON e' funzione del solo t_sat: celle con")
            print(f"     lo stesso t_sat danno valori diversi, quindi dopo la")
            print(f"     saturazione resta una dipendenza dai parametri.")
    else:
        print(f"\n  (nessuna coppia con t_sat simile: controllo non eseguibile)")


    return dict(n=len(ok), cv=float(cv), spearman_tsat=rho_s,
                relax_median=float(np.median(obs)),
                relax_slow=[float(slow.min()), float(slow.max())],
                relax_fast=[float(fast.min()), float(fast.max())],
                n_pairs=len(pairs),
                pair_max_diff=(float(dres.max()) if pairs else float('nan')),
                verdict=verdict)

# ═══════════════════════════════════════════════════════════════════════
#  Grafici
# ═══════════════════════════════════════════════════════════════════════

def _matrix(res, key):
    Ws = sorted({r['W'] for r in res}); Bs = sorted({r['B'] for r in res})
    M = np.full((len(Ws), len(Bs)), np.nan)
    for r in res:
        M[Ws.index(r['W']), Bs.index(r['B'])] = r[key]
    return np.array(Ws), np.array(Bs), M


def plot_timing_map(res, fname='07_mappa_tempi.png'):
    """Tempo di convergenza e tempo di saturazione sulla griglia."""
    Ws, Bs, TC = _matrix(res, 't_conv_med')
    _,  _,  TS = _matrix(res, 't_sat_med')
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    for ax, M, title in ((axes[0], TC, 'Tempo di convergenza (passi)'),
                         (axes[1], TS, 'Tempo di saturazione di Gp (passi)')):
        fin = M[np.isfinite(M)]
        norm = LogNorm(vmin=max(1, fin.min()), vmax=fin.max()) if len(fin) else None
        im = ax.imshow(M, origin='lower', aspect='auto', cmap='viridis',
                       norm=norm)
        ax.set_xticks(range(len(Bs))); ax.set_xticklabels([f'{b:g}' for b in Bs])
        ax.set_yticks(range(len(Ws))); ax.set_yticklabels([f'{w:g}' for w in Ws])
        ax.set_xlabel('B (soglia di trasduzione)')
        ax.set_ylabel('W (peso di trasduzione)')
        ax.set_title(title, fontsize=10)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                ax.text(j, i, 'mai' if not np.isfinite(v) else f'{v:.0f}',
                        ha='center', va='center', fontsize=8, color='white')
        fig.colorbar(im, ax=ax, shrink=0.85)

    fin_tc = TC[np.isfinite(TC)]
    fig.savefig(figpath(fname), dpi=150); plt.close(fig)
    print(f"  {FIGDIR}/{fname}")


def plot_decomposition(res, fname='08_decomposizione_tempi.png'):
    ok = [r for r in res if np.isfinite(r['relax_obs_from_sat'])]
    if not ok:
        print(f"  (salto {fname}: nessuna cella utilizzabile)"); return
    ts = np.array([r['t_sat_med'] for r in ok])
    tc = np.array([r['t_conv_med'] for r in ok])
    rl = tc - ts
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)

    axes[0].scatter(ts, tc, s=100, c=[r['B'] for r in ok],
                    cmap='plasma', edgecolor='k', linewidth=0.6, zorder=3)
    lim = [min(ts.min(), tc.min())*0.8, max(ts.max(), tc.max())*1.2]
    axes[0].plot(lim, lim, '--', color='0.5', lw=1.2, label='t_conv = t_sat')
    axes[0].plot(lim, np.array(lim) + np.median(rl), '-', color='0.3', lw=1.4,
                 label=f't_sat + {np.median(rl):.0f}')
    axes[0].set_xscale('log'); axes[0].set_yscale('log')
    axes[0].set_xlabel('t_sat: passi perché Gp saturi ovunque')
    axes[0].set_ylabel('t_conv: passi fino al punto fisso')
    axes[0].set_title('Il tempo totale è saturazione più una costante',
                      fontsize=10)
    axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3, which='both')

    axes[1].scatter(ts, rl, s=100, c=[r['B'] for r in ok], cmap='plasma',
                    edgecolor='k', linewidth=0.6, zorder=3)
    axes[1].axhline(np.median(rl), color='0.3', lw=1.4,
                    label=f'mediana {np.median(rl):.0f}')
    pr = np.array([r['relax_pred_from_sat'] for r in ok])
    if np.isfinite(pr).any():
        axes[1].axhline(np.nanmedian(pr), color='C3', ls='--', lw=1.4,
                        label=f'da $\\rho$, da $t_{{sat}}$: '
                              f'{np.nanmedian(pr):.0f}')
    axes[1].set_xscale('log')
    axes[1].set_xlabel('t_sat (passi)')
    axes[1].set_ylabel('t_conv - t_sat (passi)')
    axes[1].set_title('Il residuo decresce in $t_{sat}$: è il tratto\n'
                      'non geometrico, che pesa in proporzione inversa',
                      fontsize=10)
    axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)
    fig.savefig(figpath(fname), dpi=150); plt.close(fig)
    print(f"  {FIGDIR}/{fname}")



# ═══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':

    N_STARTS = 30
    WORKERS  = max(1, (os.cpu_count() or 2)//2)

    res = grid_timing(n_starts=N_STARTS, n_workers=WORKERS)
    print(f"\n  grafici in {os.path.abspath(FIGDIR)}/")
    plot_timing_map(res)
    plot_decomposition(res)






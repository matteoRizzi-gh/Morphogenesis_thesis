"""
grid_and_gp.py
==============
1. Mappa di fase su (W, B): stati distinti, ID, saturazione di Gp, e
   condizione di consistenza sul bilancio dei termini.
2. Scansione fine della tolleranza sulla cella di frontiera, con H0.

"""

import os
import numpy as np
import torch
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

import Rfc_and_ID as R
from dynamics_t_sparse_gh import BioelectricLattice, G_r

TOLS = np.logspace(-9, -3, 13)
# GP_EDGE = 1e-12
GP_EDGE = 0.0

def _drive(e, B, W):
    """drive(e) = (2*sigma(-e + B) - 1) * W.

    """
    return (2.0 / (1.0 + np.exp(-(-np.asarray(e, dtype=float) + B))) - 1.0) * W


def frozen_field_prediction(e, B, W, g0=1.0, dt=None):
    """Predizione a campo congelato per il modello implementato.

    Trattando e come costante, il termine -Gp e' inerte e g = Gp/G_r integra:
        g_{t+1} = g_t + dt * drive(e)

    """
    from dynamics_t_sparse_gh import DT as _DT
    dt = _DT if dt is None else dt

    d  = np.atleast_1d(_drive(e, B, W))
    g0 = np.broadcast_to(np.atleast_1d(np.asarray(g0, dtype=float)), d.shape)

    branch = np.where(d > 0.0, 'hi', np.where(d < 0.0, 'lo', 'none'))
    dist   = np.where(d > 0.0, 2.0 - g0, g0)
    with np.errstate(divide='ignore', invalid='ignore'):
        t_sat = np.where(d != 0.0, dist / (dt * np.abs(d)), np.inf)
    t_sat = np.maximum(t_sat, 0.0)

    return dict(drive=d, branch=branch, t_sat=t_sat,
                t_sat_all=float(np.max(t_sat)),      # quando saturano TUTTE
                t_sat_median=float(np.median(t_sat)),
                frac_hi=float((d > 0.0).mean()),
                frac_lo=float((d < 0.0).mean()))

# ═══════════════════════════════════════════════════════════════════════
#  Simulazione: stato finale + e_bar accumulato sulla traiettoria
# ═══════════════════════════════════════════════════════════════════════

def final_state(W, B, Rr=4, G=0.05, T=5000, seed=0):
    """
    Ritorna:
        v      [121]  Vmem finale
        Gp     [121]  conduttanza finale
        field  [121]  campo aggregato con la maschera d'AZIONE (raggio R)
                      -- e' l'osservabile su cui e' definito l'RFC
        e_bar  [121]  campo PERCEPITO all'ultimo passo (maschera di percezione,
                      R=1): e' cio' che guida Gp, non l'osservabile
        e_traj [121]  campo percepito mediato sulla traiettoria
        e_init [121]  campo percepito al passo 0, PRIMA di ogni aggiornamento
                      -- serve alla predizione a campo congelato, che valutata
                      su e_traj sarebbe circolare
    """
    net = BioelectricLattice(rows=11, cols=11, G=G, R=Rr, W=W, B=B)
    net.reset()
    g = torch.Generator().manual_seed(int(seed))
    net.Gp = torch.rand(net.N, generator=g, dtype=torch.float64) * 2.0 * G_r
    Gp_init = net.Gp.clone()

    def _perceived(eV):
        return torch.zeros(net.N, dtype=torch.float64).scatter_add_(
            0, net.perc_cell_idx,
            eV.index_select(0, net.perc_field_idx)) / net.perc_n

    with torch.no_grad():
        # campo percepito allo stato iniziale, prima del primo passo
        e_init = _perceived(net._eV_from_cells())

        e_sum = torch.zeros(net.N, dtype=torch.float64)
        for _ in range(T):
            eV, _ = net.step(v_left_22=None)
            e_sum += _perceived(eV)

        field = torch.zeros(net.N, dtype=torch.float64).scatter_add_(
            0, net.endo_c_idx,
            eV.index_select(0, net.endo_f_idx)) / R.NUM_FIELD_NEIGHBORS
        e_bar = _perceived(eV)

    return (net.v.clone().numpy(), net.Gp.clone().numpy(), field.numpy(),
            e_bar.numpy(), (e_sum / T).numpy(), e_init.numpy(),
            (Gp_init / G_r).numpy())


def make_cloud(W, B, Rr=4, G=0.05, n_points=200, T=5000, seed=0):
    V, P, F, E, Etr, E0, G0 = [], [], [], [], [], [], []
    for i in range(n_points):
        v, gp, f, e, etr, e0, g0 = final_state(W, B, Rr, G, T,
                                               seed=seed * 100000 + i)
        V.append(v); P.append(gp); F.append(f)
        E.append(e); Etr.append(etr); E0.append(e0); G0.append(g0)
    return (np.array(V), np.array(P), np.array(F), np.array(E),
            np.array(Etr), np.array(E0), np.array(G0))


# ═══════════════════════════════════════════════════════════════════════
#  Saturazione di Gp
# ═══════════════════════════════════════════════════════════════════════

def gp_saturation(P):
    """P: [n_points, 121] di Gp finali, gia' ritagliate in [0, 2*G_r].

    Test esatto sui bordi (GP_EDGE = 0): il clamp restituisce esattamente 0 o
    2*G_r quando taglia. Con una tolleranza si contano come sature celle
    strettamente interne, che sono ancora dentro l'anello di retroazione.

    """
    P = np.asarray(P)
    lo = float((P <= GP_EDGE).mean())
    hi = float((P >= 2.0 * G_r - GP_EDGE).mean())
    return dict(frac_zero=lo, frac_max=hi, frac_interior=1.0 - lo - hi,
                mean_over_Gr=float(P.mean() / G_r),
                std_over_Gr=float(P.std() / G_r),
                gp_fully_saturated=(lo + hi > 0.9999))


def drive_stats(E, Etr, B, W, E0=None, G0=None):
    """Statistiche del drive nei tre stati.

    E   : campo percepito allo stato FINALE  -> misura a posteriori, circolare
    Etr : campo percepito mediato sulla traiettoria -> meno circolare ma
          ancora dominato dal regime post-saturazione per le celle veloci
    E0  : campo percepito allo stato INIZIALE -> l'unico valutabile come
          PREDIZIONE, ed e' quello da usare per il confronto con t_sat

    G0  : conduttanza adimensionale iniziale per cella; serve alla predizione
          del tempo di saturazione (distanza dal bordo da percorrere).
    """
    d_tr  = _drive(Etr, B, W)
    d_fin = _drive(E,   B, W)
    out = dict(
        e_final_mean=float(np.mean(E)),
        e_traj_mean=float(np.mean(Etr)),
        e_traj_std=float(np.std(Etr)),
        drive_final=float(_drive(np.mean(E),   B, W)),
        drive_traj=float(_drive(np.mean(Etr), B, W)),
        drive_traj_percell_mean=float(d_tr.mean()),
        drive_final_percell_mean=float(d_fin.mean()),
        frac_hi_traj=float((d_tr > 0.0).mean()),
        frac_lo_traj=float((d_tr < 0.0).mean()),
    )
    if E0 is not None:
        g0 = 1.0 if G0 is None else np.asarray(G0, dtype=float)
        pred = frozen_field_prediction(np.asarray(E0, dtype=float).ravel(),
                                       B, W,
                                       g0=(g0.ravel() if G0 is not None else 1.0))
        out.update(
            e_init_mean=float(np.mean(E0)),
            drive_init=float(_drive(np.mean(E0), B, W)),
            drive_init_percell_mean=float(pred['drive'].mean()),
            pred_branch_hi=pred['frac_hi'],
            pred_branch_lo=pred['frac_lo'],
            pred_t_sat_all=pred['t_sat_all'],
            pred_t_sat_median=pred['t_sat_median'],
        )
    return out

# ═══════════════════════════════════════════════════════════════════════
#  H0: componenti connesse al variare della scala
# ═══════════════════════════════════════════════════════════════════════

def h0_profile(X, tols=TOLS, min_height=1e-15):
    """
    Distanze RMS per cella (scala in volt, confrontabile fra celle della
    griglia), single-linkage, conteggio a ogni tolleranza.

    """
    X = np.asarray(X, dtype=np.float64)
    n, dim = X.shape
    d = pdist(X) / np.sqrt(dim)
    spread = float(d.max())
    if spread == 0.0:
        return dict(counts={float(t): 1 for t in tols},
                    merge_heights=np.zeros(n-1), largest_gap=0.0,
                    gap_at=0.0, spread=0.0, n_points=n,
                    n_below_min=int(n-1), degenerate=True)
    Z = linkage(d, method='single')
    counts = {float(t): int(len(np.unique(fcluster(Z, t, criterion='distance'))))
              for t in tols}
    h = np.sort(Z[:, 2])
    hp = h[h > min_height]
    n_below = int(len(h) - len(hp))
    if len(hp) > 1:
        ratios = hp[1:] / hp[:-1]
        j = int(np.argmax(ratios))
        gap, gap_at = float(ratios[j]), float(hp[j])
        # componenti che sopravvivono al salto: i punti meno le fusioni gia'
        # avvenute, comprese quelle sotto min_height
        n_comp = int(n - (n_below + j + 1))
    else:
        gap, gap_at = 1.0, (float(hp[0]) if len(hp) else 0.0)
        n_comp = 1
    return dict(counts=counts, merge_heights=h, largest_gap=gap,
                gap_at=gap_at, n_comp_at_gap=n_comp, spread=spread,
                n_points=n, n_below_min=n_below, degenerate=False)



def print_h0(prof, label=''):
    """Profilo di H0 """
    print(f"\n  H0 (componenti connesse) al variare della scala {label}")
    print(f"  {'eps (V)':>12}{'componenti':>13}")
    for t, c in sorted(prof['counts'].items()):
        print(f"  {t:12.2e}{c:13d}")

    if prof.get('degenerate'):
        print("  nuvola degenere: tutti gli stati coincidono, "
              "nessuna fusione da analizzare")
        return

    spread = prof['spread']
    gap, gap_at = prof['largest_gap'], prof['gap_at']
    n_dup = prof.get('n_below_min', 0)
    n_comp = prof.get('n_comp_at_gap')

    print(f"\n  spread della nuvola: {spread:.2e} V "
          f"(distanza RMS per cella)")
    if n_dup:
        print(f"  fusioni al livello dell'arrotondamento: "
              f"{n_dup}/{prof['n_points']-1}")
    print(f"  salto relativo massimo fra fusioni consecutive: {gap:.2f}x  "
          f"a eps={gap_at:.2e}", end='')
    if n_comp is not None:
        print(f"   ({n_comp} componenti sopravvivono)")
    else:
        print()

    if gap_at > 0 and spread > 0:
        decadi = np.log10(spread / gap_at)
        print(f"  il salto sta {decadi:.0f} ordini di grandezza sotto "
              f"lo spread")
    else:
        decadi = float('inf')

    if R.is_collapsed(spread):
        print(f"\n  Questa nuvola e' COLLASSATA secondo il criterio uniforme")
        print(f"  (spread < {R.SPREAD_COLLAPSED:g} V, R.is_collapsed).")
        print(f"  Le componenti che sopravvivono al salto NON sono attrattori")
        print(f"  distinti: sono struttura residua a una scala molti ordini")
        print(f"  sotto la dispersione della nuvola, cioe' al limite della")
        print(f"  risoluzione numerica. E' una cella lenta colta prima del")
        print(f"  collasso, e a T maggiore lo spread va a zero.")
    elif gap >= 10 and decadi <= 3:
        print(f"\n  Salto marcato a una scala confrontabile con lo spread:")
        print(f"  esiste una scala privilegiata e il numero di attrattori e'")
        print(f"  ben definito a quella scala.")
    elif gap >= 10:
        print(f"\n  Salto marcato ma a una scala {decadi:.0f} ordini sotto lo")
        print(f"  spread: la struttura risolta e' molto piu' fine della")
        print(f"  dispersione della nuvola, e il conteggio va interpretato con")
        print(f"  cautela anche se la nuvola non e' formalmente collassata.")
    else:
        print(f"\n  Nessun salto marcato: le fusioni sono distribuite con")
        print(f"  continuita', nessun conteggio e' stabile e 'quanti")
        print(f"  attrattori' non ha risposta indipendente dalla scala.")


# ═══════════════════════════════════════════════════════════════════════
#  Griglia
# ═══════════════════════════════════════════════════════════════════════

def _init_worker():
    torch.set_num_threads(1)


def _work_cell(a):
    torch.set_num_threads(1)
    W, B, Rr, G, n_points, T, seed, tols, k = a
    V, P, F, E, Etr, E0, G0 = make_cloud(W, B, Rr, G, n_points, T, seed)
    hv = h0_profile(V, tols)
    return dict(
        W=W, B=B, R=Rr, G=G, n_points=n_points, T=T,
        gp=gp_saturation(P),
        drv=drive_stats(E, Etr, B, W, E0=E0, G0=G0),
        h0=hv,
        id_vmem=R.mle_id(V, k=k) if not hv['degenerate'] else 0.0,
        id_field=R.mle_id(F, k=k) if not hv['degenerate'] else 0.0,
        vmem_spread=float(pdist(V).max() / np.sqrt(V.shape[1])),
    )


def grid(Ws=R.PAPER_W, Bs=R.PAPER_B, Rr=4, G=0.05, n_points=200, T=5000,
         seed=0, tols=TOLS, k=10, n_workers=None, save_as='grid_gp.pt'):
    jobs = [(W, B, Rr, G, n_points, T, seed, tols, k) for W in Ws for B in Bs]
    print(f"Griglia: {len(jobs)} celle x {n_points} punti (T={T})")
    print("  (e_bar accumulato sulla traiettoria: circa il doppio del tempo)")
    res = []
    if n_workers and n_workers > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=n_workers,
                                 initializer=_init_worker) as ex:
            futs = [ex.submit(_work_cell, j) for j in jobs]
            for c, f in enumerate(as_completed(futs), 1):
                res.append(f.result())
                print(f"  [{c}/{len(jobs)}] fatta")
                torch.save(res, R.out_path(save_as))
    else:
        for c, j in enumerate(jobs, 1):
            res.append(_work_cell(j)); print(f"  [{c}/{len(jobs)}] fatta")
            torch.save(res, R.out_path(save_as))
    res.sort(key=lambda r: (r['W'], r['B']))
    print_grid(res, tols)
    check_condition(res)
    torch.save(res, R.out_path(save_as))
    return res


def print_grid(res, tols=TOLS):
    show = [t for t in (1e-8, 1e-6, 1e-4) if any(
        abs(t - x) < 1e-12 for x in list(res[0]['h0']['counts']))]
    if not show:
        show = sorted(res[0]['h0']['counts'])[::max(1, len(tols)//3)][:3]
    print(f"\n{'='*126}\n  MAPPA DI FASE  (R={res[0]['R']}, G={res[0]['G']}, "
          f"{res[0]['n_points']} punti, T={res[0]['T']})\n{'='*126}")
    hdr = f"  {'W':>7}{'B':>9}" + "".join(f"{f'H0@{t:.0e}':>11}" for t in show)
    hdr += (f"{'IDVmem':>9}{'spread':>11}{'Gp=0':>7}{'Gp=max':>8}"
            f"{'e_init':>11}{'e_traj':>11}{'drive_0':>10}{'t_sat pred':>12}")
    print(hdr)
    for r in res:
        g, d = r['gp'], r['drv']
        row = f"  {r['W']:7.0f}{r['B']:9.4f}"
        row += "".join(f"{r['h0']['counts'][float(t)]:11d}" for t in show)
        row += (f"{r['id_vmem']:9.2f}{r['vmem_spread']:11.2e}"
                f"{100*g['frac_zero']:6.1f}%{100*g['frac_max']:7.1f}%"
                f"{d.get('e_init_mean', float('nan')):11.3e}"
                f"{d['e_traj_mean']:11.3e}"
                f"{d.get('drive_init', float('nan')):10.4f}"
                f"{d.get('pred_t_sat_all', float('nan')):12.0f}")
        # Due marcatori distinti: la saturazione completa di Gp e' la CAUSA
        # del collasso, il collasso e' giudicato dallo spread. Sulla cella di
        # transizione i due non coincidono, ed e' esattamente il caso in cui
        # serve vederli separati.
        if R.is_collapsed(r['vmem_spread']):
            row += "   COLLASSATA"
        if g['gp_fully_saturated']:
            row += "   Gp SATURA"
        print(row)
    print("\n  'drive_0' = (2*sigma(-e_0 + B) - 1)*W allo stato INIZIALE.")




def check_condition(res, T=None):
    """Il modello a campo congelato"""
    print(f"\n{'='*96}\n  MODELLO A CAMPO CONGELATO: RAMO E TEMPO\n{'='*96}")
    print(f"  {'W':>7}{'B':>9}{'Gp=0':>8}{'Gp=max':>9}"
          f"{'ramo pred.':>12}{'ramo oss.':>11}{'ok':>5}"
          f"{'t_sat pred':>12}")

    ok_branch = tot = 0
    for r in res:
        g, d = r['gp'], r['drv']
        if 'pred_branch_hi' not in d:
            print("  (drive_stats chiamata senza E0: rilancia la griglia)")
            return None
        pred = 'hi' if d['pred_branch_hi'] > 0.5 else 'lo'
        # ramo osservato: dove sta la saturazione effettiva
        if g['frac_max'] > g['frac_zero']:
            obs = 'hi'
        elif g['frac_zero'] > g['frac_max']:
            obs = 'lo'
        else:
            obs = 'none'
        good = (pred == obs)
        ok_branch += good; tot += 1
        print(f"  {r['W']:7.0f}{r['B']:9.4f}{100*g['frac_zero']:7.1f}%"
              f"{100*g['frac_max']:8.1f}%{pred:>12}{obs:>11}"
              f"{('si' if good else 'NO'):>5}{d['pred_t_sat_all']:12.0f}")

    print(f"\n  ramo predetto correttamente: {ok_branch}/{tot}")
    return dict(n=tot, ok_branch=ok_branch)


# ═══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':

    MODE = 'grid'     # 'frontier' | 'grid'
    WORKERS = max(1, (os.cpu_count() or 2)//2)

    if MODE == 'frontier':
        W, B, Rr, G, NP, T = 500.0, 0.002, 4, 0.05, 300, 5000
        print(f"Cella di frontiera W={W} B={B} R={Rr} ({NP} punti, T={T})")
        V, P, F, E, Etr, E0, G0 = make_cloud(W, B, Rr, G,
                                             n_points=NP, T=T)
        prof = h0_profile(V, np.logspace(-13, -2, 111))
        print_h0(prof, f"(W={W}, B={B})")
        g = gp_saturation(P)
        d = drive_stats(E, Etr, B, W, E0=E0, G0=G0)
        print(f"\n  Gp finale: {100*g['frac_zero']:.1f}% a 0, "
              f"{100*g['frac_max']:.1f}% a 2*G_r, "
              f"{100*g['frac_interior']:.1f}% all'interno")
        print(f"  drive iniziale {d['drive_init']:+.4f}  "
              f"-> ramo '{'hi' if d['pred_branch_hi'] > 0.5 else 'lo'}', "
              f"t_sat predetto {d['pred_t_sat_all']:.0f} passi")
        print(f"  drive sulla traiettoria {d['drive_traj']:+.4f} "
              f"(a posteriori, non predittivo)")

        torch.save(dict(W=W, B=B, R=Rr, G=G, n_points=NP, T=T,
                        h0=prof, gp=g, drv=d),
                   R.out_path('frontier_W500_B0002.pt'))
        print(f"\n  salvato in {R.out_path('frontier_W500_B0002.pt')}")

    elif MODE == 'grid':
        grid(n_points=300, T=5000, n_workers=WORKERS)


"""
run_parallel_rfc_and_id.py
==========================
Esegue in parallelo il lavoro di Rfc_and_ID.py.
"""

import os
import numpy as np
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.spatial.distance import pdist

import Rfc_and_ID as R


for _name in ('sample_cloud', 'analyze_cloud', 'id_of_cloud',
              'cloud_diagnostics', 'run_snapshots'):
    if not hasattr(R, _name):
        raise ImportError(
            f"rfc_and_id.py non espone '{_name}': stai usando una versione vecchia del file.")

N_WORKERS = max(1, (os.cpu_count() or 2) // 2)   


def _init_worker():
    torch.set_num_threads(1)


# ═══════════════════════════════════════════════════════════════════════════
#  Nuvola di stati in parallelo
# ═══════════════════════════════════════════════════════════════════════════

def _work_point(a):
    """Una traiettoria
    """
    torch.set_num_threads(1)
    idx, W, B, Rr, G, seed_i, snapshots = a
    snap = R.run_snapshots(W=W, B=B, R=Rr, G=G, seed=seed_i,
                           snapshots=snapshots)
    return idx, {t: (snap[t]['vmem'], snap[t]['field']) for t in snapshots}


def parallel_cloud(W, B, Rr=4, G=0.05, n_points=300,
                   snapshots=(1000, 3000, 5000), seed=0,
                   n_workers=N_WORKERS, verbose=True):
    jobs = [(i, W, B, Rr, G, seed * 100000 + i, snapshots)
            for i in range(n_points)]
    if verbose:
        print(f"  nuvola: {n_points} traiettorie su {n_workers} processi "
              f"(fino a {max(snapshots)} passi)")
    parts = []
    with ProcessPoolExecutor(max_workers=n_workers,
                             initializer=_init_worker) as ex:
        futs = [ex.submit(_work_point, j) for j in jobs]
        for c, f in enumerate(as_completed(futs), 1):
            parts.append(f.result())
            if verbose and c % 50 == 0:
                print(f"    {c}/{n_points}")

    parts.sort(key=lambda x: x[0])
    parts = [p[1] for p in parts]
    return {t: dict(vmem =np.array([p[t][0] for p in parts]),
                    field=np.array([p[t][1] for p in parts]))
            for t in snapshots}


def parallel_analyze_cloud(W, B, Rr=4, G=0.05, n_points=300,
                           snapshots=(1000, 3000, 5000), seed=0, k=10,
                           fractions=(0.25, 0.5, 1.0),
                           n_workers=N_WORKERS, verbose=True):
    if verbose:
        print(f"\n{'='*78}\n  W={W}  B={B}  R={Rr}  G={G}   "
              f"nuvola di {n_points} stati, istantanei a {snapshots}\n{'='*78}")
    cloud = parallel_cloud(W, B, Rr, G, n_points, snapshots, seed,
                           n_workers, verbose)
    res = {}
    for t in snapshots:
        res[t] = {}
        for var in ('field', 'vmem'):
            X = cloud[t][var]
            t2, ml, diag = R.id_of_cloud(X, k=k)
            res[t][var] = dict(twonn=t2, mle=ml, diag=diag,
                               pca=R.pca_profile(X),
                               id_vs_n=R.id_vs_n(X, fractions, k, seed=seed))

        Xv = np.asarray(cloud[t]['vmem'])
        spread = float(pdist(Xv).max() / np.sqrt(Xv.shape[1]))
        res[t]['vmem_spread'] = spread
        res[t]['collapsed'] = R.is_collapsed(spread)
    if verbose:
        R._print_cloud(res, snapshots, fractions)
        print(f"\n  criterio di collasso: spread < {R.SPREAD_COLLAPSED:g} V")
        for t in snapshots:
            tag = 'collassata' if res[t]['collapsed'] else 'viva'
            print(f"    t={t:>6}: spread {res[t]['vmem_spread']:.2e} V  {tag}")
    return dict(W=W, B=B, R=Rr, G=G, n_points=n_points, snapshots=snapshots,
                seed=seed, cloud=res)

# ═══════════════════════════════════════════════════════════════════════════
#  RFC in parallelo
# ═══════════════════════════════════════════════════════════════════════════

def _work_rfc(a):
    torch.set_num_threads(1)
    W, B, Rr, G, idx, T, seed = a
    ts = R.run_timeseries(W=W, B=B, R=Rr, G=G, seed=seed * 100000 + idx, T=T,
                          homogeneous=(idx == 0))
    return (idx,
            float(R.pca_explained(ts['ev_agg']).sum()),
            float(R.pca_explained(ts['ev_raw']).sum()),
            float(R.pca_explained(ts['vmem']).sum()))


def parallel_rfc(W, B, Rr=4, G=0.05, n_samples=101, T=5000, seed=0,
                 n_workers=N_WORKERS, verbose=True):
    jobs = [(W, B, Rr, G, i, T, seed) for i in range(n_samples)]
    parts = []
    with ProcessPoolExecutor(max_workers=n_workers,
                             initializer=_init_worker) as ex:
        futs = [ex.submit(_work_rfc, j) for j in jobs]
        for c, f in enumerate(as_completed(futs), 1):
            parts.append(f.result())
            if verbose and c % 25 == 0:
                print(f"    RFC {c}/{n_samples}")
    parts.sort(key=lambda x: x[0])
    agg = np.array([p[1] for p in parts]); raw = np.array([p[2] for p in parts])
    vm  = np.array([p[3] for p in parts])
    return dict(ev_agg_mean=100*agg.mean(), ev_raw_mean=100*raw.mean(),
                vmem_mean=100*vm.mean(), RFC_mean=100*(agg-vm).mean(),
                RFC_sample0=100*(agg[0]-vm[0]), RFC_raw_mean=100*(raw-vm).mean(),
                n_samples=n_samples, T=T)


def parallel_replicate(W, B, Rr=4, Gs=R.PAPER_G, n_samples=101, T=5000,
                       seed=0, n_workers=N_WORKERS):
    print(f"\n{'='*78}\n  REPLICA RFC   W={W}  B={B}  R={Rr}   G in {Gs}\n"
          f"  {n_samples} campioni x {T} passi\n{'='*78}")
    per_G = []
    for G in Gs:
        r = parallel_rfc(W, B, Rr, G, n_samples, T, seed, n_workers,
                         verbose=False)
        per_G.append(dict(G=G, **{k: r[k] for k in
                                  ('RFC_mean', 'RFC_sample0', 'RFC_raw_mean')}))
        print(f"  G={G:5.2f}   RFC media {r['RFC_mean']:+7.2f}%   "
              f"campione 0 {r['RFC_sample0']:+7.2f}%   "
              f"campo grezzo {r['RFC_raw_mean']:+7.2f}%")
    m  = float(np.mean([x['RFC_mean'] for x in per_G]))
    s0 = float(np.mean([x['RFC_sample0'] for x in per_G]))
    print(f"\n  mediato su G:  RFC media {m:+.2f}%   campione 0 {s0:+.2f}%")
    print(f"  paper: ~+35% (forte) / -5..+5% (debole)")
    return dict(W=W, B=B, R=Rr, per_G=per_G, RFC_mean=m, RFC_sample0=s0)


# ═══════════════════════════════════════════════════════════════════════════
#  Sweep sulla griglia del paper, con la nuvola
# ═══════════════════════════════════════════════════════════════════════════

def _work_combo_cloud(a):
    torch.set_num_threads(1)
    W, B, Rr, G, n_points, snapshots, seed, k, fractions = a
    cloud = {t: dict(vmem=[], field=[]) for t in snapshots}
    for i in range(n_points):
        snap = R.run_snapshots(W=W, B=B, R=Rr, G=G, seed=seed*100000+i,
                               snapshots=snapshots)
        for t in snapshots:
            cloud[t]['vmem'].append(snap[t]['vmem'])
            cloud[t]['field'].append(snap[t]['field'])
    res = {}
    for t in snapshots:
        res[t] = {}
        for var in ('field', 'vmem'):
            X = np.asarray(cloud[t][var])
            t2, ml, diag = R.id_of_cloud(X, k=k)
            res[t][var] = dict(twonn=t2, mle=ml, diag=diag,
                               pca=R.pca_profile(X),
                               id_vs_n=R.id_vs_n(X, fractions, k, seed=seed))

        Xv = np.asarray(cloud[t]['vmem'])
        spread = float(pdist(Xv).max() / np.sqrt(Xv.shape[1]))
        res[t]['vmem_spread'] = spread
        res[t]['collapsed'] = R.is_collapsed(spread)
    return dict(W=W, B=B, R=Rr, G=G, n_points=n_points, snapshots=snapshots,
                seed=seed, cloud=res)


def parallel_sweep(Ws=R.PAPER_W, Bs=R.PAPER_B, Rr=4, G=0.05, n_points=200,
                   snapshots=(5000,), seed=0, k=10, fractions=(0.5, 1.0),
                   n_workers=N_WORKERS, save_as='sweep_cloud.pt'):
    """Parallelo sulle COMBINAZIONI: ogni worker costruisce la sua nuvola"""
    jobs = [(W, B, Rr, G, n_points, snapshots, seed, k, fractions)
            for W in Ws for B in Bs]
    print(f"Sweep: {len(jobs)} combinazioni su {n_workers} processi "
          f"({n_points} punti di nuvola ciascuna, T={max(snapshots)})")
    res, done = [], 0
    with ProcessPoolExecutor(max_workers=n_workers,
                             initializer=_init_worker) as ex:
        futs = [ex.submit(_work_combo_cloud, j) for j in jobs]
        for f in as_completed(futs):
            r = f.result(); res.append(r); done += 1
            t = max(snapshots)
            fd, vd = r['cloud'][t]['field'], r['cloud'][t]['vmem']
            tag = 'collassata' if r['cloud'][t]['collapsed'] else ''
            print(f"  [{done}/{len(jobs)}] W={r['W']:>6.0f} B={r['B']:.4f}   "
                  f"ID campo {fd['mle']:6.2f}   ID Vmem {vd['mle']:6.2f}   "
                  f"H0 {vd['diag']['n_distinct']}/{n_points}   "
                  f"spread {r['cloud'][t]['vmem_spread']:.2e} {tag}")
            torch.save(res, R.out_path(save_as))
    res.sort(key=lambda r: (r['W'], r['B']))
    _print_sweep(res, max(snapshots), Rr, G, n_points)
    torch.save(res, R.out_path(save_as))
    return res


def _print_sweep(res, t, Rr, G, n_points):
    print(f"\n{'='*100}\n  RIEPILOGO  (R={Rr}, G={G}, t={t}, "
          f"{n_points} punti)\n{'='*100}")
    print(f"  {'W':>8}{'B':>9}{'IDcampo':>9}{'IDVmem':>9}{'diff':>8}"
          f"{'PC90 campo':>12}{'PC90 Vmem':>11}{'H0':>7}{'spread':>12}{'':>13}")
    for r in res:
        c = r['cloud'][t]
        f_, v_ = c['field'], c['vmem']
        sp = c.get('vmem_spread', float('nan'))
        col = c.get('collapsed')
        tag = ('   collassata' if col else '' if col is not None
               else '   (spread n/d)')
        print(f"  {r['W']:8.0f}{r['B']:9.4f}{f_['mle']:9.2f}{v_['mle']:9.2f}"
              f"{f_['mle']-v_['mle']:+8.2f}{f_['pca']['pc_90']:12d}"
              f"{v_['pca']['pc_90']:11d}{v_['diag']['n_distinct']:7d}"
              f"{sp:12.2e}{tag:>13}")

    print(f"\n  'H0' = componenti connesse della nuvola di Vmem alla scala")
    print(f"  eps = {R.COLLAPSE_TOL:g} V (RMS per cella), con single-linkage.")
    print(f"  'spread' = massima distanza fra due stati finali, stessa unita'.")
    print(f"\n  Il criterio di COLLASSO e' lo spread (< {R.SPREAD_COLLAPSED:g} V, R.is_collapsed), non H0. ")

    have = [r for r in res if 'collapsed' in r['cloud'][t]]
    if have:
        dead = [r for r in have if r['cloud'][t]['collapsed']]
        print(f"\n  collassate: {len(dead)}/{len(have)}")
        odd = [(r['W'], r['B'], r['cloud'][t]['vmem']['diag']['n_distinct'])
               for r in dead if r['cloud'][t]['vmem']['diag']['n_distinct'] > 1]
        if odd:
            print(f"  celle collassate con H0 > 1 (transizione): {odd}")

# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':

    MODE = 'cloud'          # 'cloud' | 'sweep' | 'replicate'
    WORKERS = N_WORKERS

    print(f"core rilevati: {os.cpu_count()}   worker: {WORKERS}   "
          f"torch.get_num_threads(): {torch.get_num_threads()}")

    if MODE == 'cloud':
        torch.save(parallel_analyze_cloud(1000.0, 5e-4, n_points=1000,
                                          snapshots=(1000, 3000, 5000),
                                          n_workers=WORKERS),
                   R.out_path('cloud_stigmergic.pt'))
        torch.save(parallel_analyze_cloud(10.0, 0.02, n_points=1000,
                                          snapshots=(1000, 3000, 5000),
                                          n_workers=WORKERS),
                   R.out_path('cloud_mosaic.pt'))

    elif MODE == 'sweep':
        parallel_sweep(n_points=300, snapshots=(5000,), n_workers=WORKERS)

    elif MODE == 'replicate':
        torch.save([parallel_replicate(1000.0, 5e-4, n_workers=WORKERS),
                    parallel_replicate(  10.0, 0.02, n_workers=WORKERS)],
                   R.out_path('replicate_rfc.pt'))

    else:
        raise ValueError(f"MODE non riconosciuto: '{MODE}'")




    
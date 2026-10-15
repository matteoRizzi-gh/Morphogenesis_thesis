"""
Rfc_and_ID.py
=============
Due misure a confronto sul modello allineato di Manicka & Levin:

  (1) RFC, la "dimensionalita'" del paper: varianza spiegata dalle prime 3 PC
      della SERIE TEMPORALE, campo meno Vmem.
  (2) Dimensione intrinseca della NUVOLA DI STATI raggiunti da molte
      condizioni iniziali indipendenti.


DUE CAUTELE SULLE STIME DI ID, ENTRAMBE GESTITE
---------------------------------------------------
A) NUMEROSITA'. Il numero di punti necessario cresce esponenzialmente con la
   dimensione: una stima intorno a 25 con 300 punti e' quasi certamente
   SOTTOSTIMATA. id_vs_n() ripete la stima su frazioni crescenti della nuvola:
   se il valore sale ancora con N, non e' saturo e non va riportato come tale.
B) ISTANTE DI CAMPIONAMENTO. Il modello stigmergico non converge entro 5000
   passi (misurato: ||dV|| ancora sopra l'1% del picco all'ultimo passo), quindi
   la nuvola dipende da QUANDO la si campiona. Per questo si campiona a piu'
   istanti e si riportano separatamente.

"""

import os
import math
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

from dynamics_t_sparse_gh import BioelectricLattice, G_r

OUTPUT_DIR = 'output_rfc'

def out_path(name):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return os.path.join(OUTPUT_DIR, name)

PAPER_W = [10.0, 100.0, 200.0, 500.0, 1000.0]
PAPER_B = [0.0005, 0.002, 0.005, 0.01, 0.02]
PAPER_R = [1, 4, 10, 15, 21]
PAPER_G = [0.0, 0.05, 0.2, 0.5, 1.0]

NUM_FIELD_NEIGHBORS = 4.0
COLLAPSE_TOL = 1e-6   # 1e-6 | 1e-8 | 1e-4   # V, RMS per cella: sotto questa soglia due stati sono lo stesso

SPREAD_COLLAPSED = 1e-3        # V, distanza RMS per cella


def is_collapsed(vmem_spread, tol=SPREAD_COLLAPSED):
    """Criterio unico di collasso:
    vmem_spread: massima distanza fra due stati finali della nuvola,
    normalizzata a RMS per cella (volt), cioe' pdist(V).max()/sqrt(n_celle).
    """
    s = float(vmem_spread)
    return bool(np.isfinite(s) and s < tol)




# ═══════════════════════════════════════════════════════════════════════════
#  Simulazione
# ═══════════════════════════════════════════════════════════════════════════

def _aggregate_field(net, eV):
    """Campo per cella con la maschera del raggio d'azione R"""
    return torch.zeros(net.N, dtype=torch.float64).scatter_add_(
        0, net.endo_c_idx, eV.index_select(0, net.endo_f_idx)) / NUM_FIELD_NEIGHBORS


def _make_net(W, B, R, G, seed, homogeneous=False):
    net = BioelectricLattice(rows=11, cols=11, G=G, R=R, W=W, B=B)
    net.reset()
    if not homogeneous:
        g = torch.Generator().manual_seed(int(seed))
        net.Gp = torch.rand(net.N, generator=g, dtype=torch.float64) * 2.0 * G_r
    return net


def run_snapshots(W, B, R, G, seed, snapshots=(1000, 3000, 5000),
                  homogeneous=False):
    """
    Una traiettoria; ritorna solo gli istantanei richiesti.
        {t: dict(vmem=[121], field=[121])}
    """
    net = _make_net(W, B, R, G, seed, homogeneous)
    want, out, T = set(snapshots), {}, max(snapshots)
    with torch.no_grad():
        for t in range(1, T + 1):
            eV, _ = net.step(v_left_22=None)
            if t in want:
                out[t] = dict(vmem=net.v.clone().numpy(),
                              field=_aggregate_field(net, eV).numpy())
    return out


def run_timeseries(W, B, R, G, seed, T=5000, homogeneous=False):
    """Serie temporale completa: serve solo all'RFC."""
    net = _make_net(W, B, R, G, seed, homogeneous)
    vm, er, ea = [], [], []
    with torch.no_grad():
        for _ in range(T):
            eV, _ = net.step(v_left_22=None)
            vm.append(net.v.clone()); er.append(eV.clone())
            ea.append(_aggregate_field(net, eV))
    return dict(vmem=torch.stack(vm).numpy(),
                ev_raw=torch.stack(er).numpy(),
                ev_agg=torch.stack(ea).numpy())


def sample_cloud(W, B, R=4, G=0.05, n_points=300, snapshots=(1000, 3000, 5000),
                 seed=0, verbose=False):
    """
    Nuvola di stati: una condizione iniziale indipendente per punto.
    Ritorna {t: dict(vmem=[n,121], field=[n,121])}.
    """
    acc = {t: dict(vmem=[], field=[]) for t in snapshots}
    for i in range(n_points):
        snap = run_snapshots(W, B, R, G, seed=seed * 100000 + i,
                             snapshots=snapshots)
        for t in snapshots:
            acc[t]['vmem'].append(snap[t]['vmem'])
            acc[t]['field'].append(snap[t]['field'])
        if verbose and (i + 1) % 50 == 0:
            print(f"    {i+1}/{n_points}")
    return {t: {k: np.asarray(v) for k, v in d.items()} for t, d in acc.items()}


# ═══════════════════════════════════════════════════════════════════════════
#  (1) Proxy del paper, sulla serie temporale
# ═══════════════════════════════════════════════════════════════════════════

def rfc_timeseries(W, B, R=4, G=0.05, n_samples=8, T=5000, seed=0, verbose=True):
    agg, raw, vm = [], [], []
    for i in range(n_samples):
        ts = run_timeseries(W, B, R, G, seed=seed * 100000 + i, T=T,
                            homogeneous=(i == 0))
        agg.append(pca_explained(ts['ev_agg']).sum())
        raw.append(pca_explained(ts['ev_raw']).sum())
        vm .append(pca_explained(ts['vmem']).sum())
        if verbose and (i + 1) % 10 == 0:
            print(f"    RFC {i+1}/{n_samples}")
    agg, raw, vm = np.array(agg), np.array(raw), np.array(vm)
    return dict(ev_agg_mean=100*agg.mean(), ev_raw_mean=100*raw.mean(),
                vmem_mean=100*vm.mean(), RFC_mean=100*(agg-vm).mean(),
                RFC_sample0=100*(agg[0]-vm[0]), RFC_raw_mean=100*(raw-vm).mean(),
                n_samples=n_samples, T=T)


def replicate_rfc(W, B, R=4, Gs=PAPER_G, n_samples=101, T=5000, seed=0):
    """RFC mediato su G"""
    print(f"\n{'='*78}\n  REPLICA RFC   W={W}  B={B}  R={R}   G in {Gs}\n"
          f"  {n_samples} campioni x {T} passi\n{'='*78}")
    per_G = []
    for G in Gs:
        r = rfc_timeseries(W, B, R, G, n_samples, T, seed, verbose=False)
        per_G.append(dict(G=G, **{k: r[k] for k in
                                  ('RFC_mean', 'RFC_sample0', 'RFC_raw_mean')}))
        print(f"  G={G:5.2f}   RFC media {r['RFC_mean']:+7.2f}%   "
              f"campione 0 {r['RFC_sample0']:+7.2f}%   "
              f"campo grezzo {r['RFC_raw_mean']:+7.2f}%")
    m  = float(np.mean([x['RFC_mean'] for x in per_G]))
    s0 = float(np.mean([x['RFC_sample0'] for x in per_G]))
    print(f"\n  mediato su G:  RFC media {m:+.2f}%   campione 0 {s0:+.2f}%")
    return dict(W=W, B=B, R=R, per_G=per_G, RFC_mean=m, RFC_sample0=s0)


# ═══════════════════════════════════════════════════════════════════════════
#  (2) Dimensione intrinseca
# ═══════════════════════════════════════════════════════════════════════════

def twonn(X, discard=None):
    """TwoNN nella forma MLE esatta: d_hat = n / sum(log mu) """
    X = np.asarray(X, dtype=np.float64)
    if len(X) < 10: return float('nan')
    d, _ = NearestNeighbors(n_neighbors=3).fit(X).kneighbors(X)
    r1, r2 = d[:, 1], d[:, 2]
    ok = (r1 > 0) & (r2 > r1)
    if ok.sum() < 10: return float('nan')
    lm = np.log(r2[ok] / r1[ok])
    s = lm.sum()
    return float(len(lm) / s) if s > 0 else float('nan')


def mle_id(X, k=10):
    """Levina & Bickel 2004, correzione MacKay-Ghahramani."""
    X = np.asarray(X, dtype=np.float64)
    if len(X) < k + 2: return float('nan')
    d, _ = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X)
    d = d[:, 1:]
    d = d[d[:, 0] > 0]
    if len(d) == 0: return float('nan')
    logd = np.log(d)
    inv = (logd[:, k-1:k] - logd[:, :k-1]).sum(1) / (k - 1)
    inv = inv[inv > 0]
    return float(1.0 / inv.mean()) if len(inv) else float('nan')


def subsample(X, n, mode, rng):
    """Ritorna (Y, subsampled). subsampled=False quando n >= len(X)."""
    T = len(X)
    if n >= T:
        return X, False
    if mode == 'stride':
        return X[:: max(1, T // n)][:n], True
    if mode == 'random':
        return X[np.sort(rng.choice(T, size=n, replace=False))], True
    raise ValueError(mode)


def cloud_diagnostics(X, tol=COLLAPSE_TOL):
    X = np.asarray(X, dtype=np.float64)
    n, dim = X.shape
    var = float(X.var(axis=0).sum())

    if n < 2:
        return dict(n_points=n, n_distinct=n, total_variance=var,
                    max_rms_dist=0.0, median_nn_dist=0.0,
                    collapsed=(n <= 1))

    # distanze condensate, normalizzate a RMS per cella
    cond = pdist(X) / np.sqrt(dim)

    n_distinct = int(len(np.unique(
        fcluster(linkage(cond, method='single'), t=tol,
                 criterion='distance'))))

    # distanza al vicino piu' prossimo, per punto, dalla forma condensata
    sq = np.zeros((n, n), dtype=np.float64)
    iu = np.triu_indices(n, k=1)
    sq[iu] = cond
    sq += sq.T
    np.fill_diagonal(sq, np.inf)
    nn = sq.min(axis=1)

    return dict(n_points=n, n_distinct=n_distinct, total_variance=var,
                max_rms_dist=float(cond.max()),
                median_nn_dist=float(np.median(nn)),
                collapsed=(n_distinct == 1))


def id_of_cloud(X, k=10, standardize=False):
    """(TwoNN, MLE). Se la nuvola e' degenere ritorna (0.0, 0.0)."""
    diag = cloud_diagnostics(X)
    if diag['collapsed']:
        return 0.0, 0.0, diag
    Z = np.asarray(X, dtype=np.float64)
    if standardize:
        Z = StandardScaler().fit_transform(Z)
        Z = Z[:, np.isfinite(Z).all(0)]
        if Z.shape[1] == 0:
            return 0.0, 0.0, diag
    return twonn(Z), mle_id(Z, k=k), diag


def id_vs_n(X, fractions=(0.25, 0.5, 1.0), k=10, standardize=False, seed=0):
    """
    Stessa stima su frazioni crescenti della nuvola. Se il valore CRESCE ancora
    passando da 0.5 a 1.0, la stima non e' satura: il numero di punti non basta
    per quella dimensione.
    """
    rng = np.random.default_rng(seed)
    n = len(X); out = {}
    for f in fractions:
        m = max(20, int(n * f))
        idx = rng.choice(n, size=min(m, n), replace=False)
        t2, ml, _ = id_of_cloud(X[idx], k=k, standardize=standardize)
        out[f] = (t2, ml, min(m, n))
    return out


def pca_profile(X, thresholds=(0.90, 0.99)):
    Z = StandardScaler().fit_transform(np.asarray(X, dtype=np.float64))
    nan_out = dict(first3=float('nan'),
                   **{f'pc_{int(100*t)}': 0 for t in thresholds})
    if not np.isfinite(Z).all():
        return nan_out
    tot = float(Z.var(axis=0).sum())
    if not np.isfinite(tot) or tot <= 1e-300:
        return nan_out
    ev = PCA().fit(Z).explained_variance_ratio_
    if not np.isfinite(ev).all():
        return nan_out
    cum = np.cumsum(ev)
    out = dict(first3=100 * float(cum[2]) if len(cum) >= 3 else float('nan'))
    for t in thresholds:
        out[f'pc_{int(100*t)}'] = int(np.searchsorted(cum, t)) + 1
    return out


def pca_explained(data, ndims=3, drop_degenerate=False):
    """Varianza spiegata dalle prime ndims PC, su dati standardizzati
    """
    X = StandardScaler().fit_transform(np.asarray(data, dtype=np.float64))
    if drop_degenerate:
        X = X[:, np.isfinite(X).all(0)]
        if X.shape[1] < ndims:
            return np.full(ndims, np.nan)
    return PCA(n_components=ndims).fit(X).explained_variance_ratio_

# ═══════════════════════════════════════════════════════════════════════════
#  Analisi di una combinazione
# ═══════════════════════════════════════════════════════════════════════════

def analyze_cloud(W, B, R=4, G=0.05, n_points=300, snapshots=(1000, 3000, 5000),
                  seed=0, k=10, fractions=(0.25, 0.5, 1.0), verbose=True):
    if verbose:
        print(f"\n{'='*78}\n  W={W}  B={B}  R={R}  G={G}   "
              f"nuvola di {n_points} stati, istantanei a {snapshots}\n{'='*78}")
    cloud = sample_cloud(W, B, R, G, n_points, snapshots, seed, verbose)

    res = {}
    for t in snapshots:
        res[t] = {}
        for var in ('field', 'vmem'):
            X = cloud[t][var]
            t2, ml, diag = id_of_cloud(X, k=k)
            res[t][var] = dict(twonn=t2, mle=ml, diag=diag,
                               pca=pca_profile(X),
                               id_vs_n=id_vs_n(X, fractions, k, seed=seed))
    if verbose:
        _print_cloud(res, snapshots, fractions)
    return dict(W=W, B=B, R=R, G=G, n_points=n_points, snapshots=snapshots,
                seed=seed, cloud=res)


def _print_cloud(res, snapshots, fractions):
    print(f"\n  --- struttura della nuvola ---")
    print(f"  {'t':>6}{'var':>7}{'stati distinti':>16}{'dist max':>11}"
          f"{'dist mediana ai vicini':>24}")
    for t in snapshots:
        for var in ('field', 'vmem'):
            d = res[t][var]['diag']
            print(f"  {t:6d}{var:>7}{d['n_distinct']:>10}/{d['n_points']:<5}"
                  f"{d['max_rms_dist']:11.3e}{d['median_nn_dist']:24.3e}"
                  + ("   COLLASSATA -> dimensione 0" if d['collapsed'] else ""))
    print(f"\n  --- dimensione intrinseca e profilo PCA ---")
    print(f"  {'t':>6}{'var':>7}{'TwoNN':>8}{'MLE':>8}"
          f"{'PC 90%':>8}{'PC 99%':>8}{'3 PC':>8}")
    for t in snapshots:
        for var in ('field', 'vmem'):
            r = res[t][var]; p = r['pca']
            print(f"  {t:6d}{var:>7}{r['twonn']:8.2f}{r['mle']:8.2f}"
                  f"{p['pc_90']:8d}{p['pc_99']:8d}{p['first3']:7.1f}%")
    print(f"\n  --- saturazione: ID contro numero di punti (MLE) ---")
    print(f"  {'t':>6}{'var':>7}" + "".join(f"{f'{int(100*f)}%':>12}" for f in fractions))
    for t in snapshots:
        for var in ('field', 'vmem'):
            row = res[t][var]['id_vs_n']
            print(f"  {t:6d}{var:>7}" +
                  "".join(f"{row[f][1]:8.2f}({row[f][2]:>3})" for f in fractions))
    print("  se il valore cresce ancora dal 50% al 100%, la stima NON e' satura:")

# ═══════════════════════════════════════════════════════════════════════════




# ═══════════════════════════════════════════════════════════════════════════


if __name__ == '__main__':

    MODE = 'rfc'          # 'cloud' | 'rfc' | 'replicate'

    N_POINTS  = 1000
    SNAPSHOTS = (1000, 3000, 5000)

    if MODE == 'cloud':
        torch.save(analyze_cloud(1000.0, 5e-4, n_points=N_POINTS,
                                 snapshots=SNAPSHOTS),
                   out_path('cloud_stigmergic.pt'))
        torch.save(analyze_cloud(10.0, 0.02, n_points=N_POINTS,
                                 snapshots=SNAPSHOTS),
                   out_path('cloud_mosaic.pt'))

    elif MODE == 'rfc':
        for W, B in ((1000.0, 5e-4), (10.0, 0.02)):
            r = rfc_timeseries(W, B, n_samples=32, T=5000)
            print(f"W={W} B={B}:  campo agg {r['ev_agg_mean']:.2f}%  "
                  f"Vmem {r['vmem_mean']:.2f}%  "
                  f"RFC media {r['RFC_mean']:+.2f}%  "
                  f"campione 0 {r['RFC_sample0']:+.2f}%")

    elif MODE == 'replicate':
        torch.save([replicate_rfc(1000.0, 5e-4, n_samples=101, T=5000),
                    replicate_rfc(  10.0, 0.02, n_samples=101, T=5000)],
                   out_path('replicate_rfc.pt'))

    else:
        raise ValueError(f"MODE non riconosciuto: '{MODE}'")
    
    
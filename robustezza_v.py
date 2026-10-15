"""
robustezza_v.py
===============
Test di robustezza rispetto alla condizione iniziale del potenziale.
"""

import argparse
import os
import time

import numpy as np
import torch
from scipy.spatial.distance import pdist

import Rfc_and_ID as R
import grid_and_gp as gg
from dynamics_t_sparse_gh import BioelectricLattice, G_r, E_p, E_d

INSIEMI = ('paper', 'v_g', 'v_solo')
V_SEED_OFFSET = 10**9            # semi di v disgiunti da quelli di g
TOLS_H0 = (1e-8, 1e-6, 1e-4)

CELLE_DEFAULT = [
    (1000.0, 0.0005),   # stigmergico nominale
    (10.0,   0.02),     # mosaic nominale
    (500.0,  0.002),    # cella di frontiera
    (10.0,   0.005),    # non collassata a T=5000, collassa asintoticamente
]


RIF = {
    (10.0, 0.0005): (37.27, 3.38e-2), (10.0, 0.002): (44.72, 3.54e-2),
    (10.0, 0.005): (33.77, 2.75e-2), (100.0, 0.0005): (36.45, 2.98e-2),
    (100.0, 0.002): (14.85, 1.84e-2), (200.0, 0.0005): (29.88, 2.57e-2),
    (200.0, 0.002): (4.52, 1.33e-2), (500.0, 0.0005): (27.11, 2.41e-2),
    (500.0, 0.002): (1.76, 6.15e-5), (1000.0, 0.0005): (25.66, 2.41e-2),
}

COLLASSATE = {(10.0, 0.01), (10.0, 0.02), (100.0, 0.005),
                        (100.0, 0.01), (100.0, 0.02), (200.0, 0.005),
                        (200.0, 0.01), (200.0, 0.02), (500.0, 0.005),
                        (500.0, 0.01), (500.0, 0.02), (1000.0, 0.002),
                        (1000.0, 0.005), (1000.0, 0.01), (1000.0, 0.02)}

RIF_RFC = {(1000.0, 0.0005): (19.63, 29.68), (10.0, 0.02): (6.30, 0.00)}


# ═══════════════════════════════════════════════════════════════════════════
#  Condizioni iniziali
# ═══════════════════════════════════════════════════════════════════════════

def _rete(W, B, Rr, G, i, seed, insieme, v_range, g_omogenea=False):
    """Rete inizializzata secondo l'insieme. Semi identici a grid_and_gp."""
    if insieme not in INSIEMI:
        raise ValueError(insieme)
    net = BioelectricLattice(rows=11, cols=11, G=G, R=Rr, W=W, B=B)
    net.reset()                              # v = V_INIT, Gp = G_r
    s = int(seed) * 100000 + int(i)
    if insieme in ('paper', 'v_g') and not g_omogenea:
        gen = torch.Generator().manual_seed(s)
        net.Gp = torch.rand(net.N, generator=gen, dtype=torch.float64) * 2.0 * G_r
    if insieme in ('v_g', 'v_solo'):
        gen = torch.Generator().manual_seed(s + V_SEED_OFFSET)
        lo, hi = v_range
        net.v = lo + (hi - lo) * torch.rand(net.N, generator=gen,
                                            dtype=torch.float64)
    return net


# ═══════════════════════════════════════════════════════════════════════════
#  Simulazione
# ═══════════════════════════════════════════════════════════════════════════

def _nuvola(args):
    """Una nuvola: n_punti traiettorie, stati a ciascun istante"""
    torch.set_num_threads(1)
    W, B, Rr, G, n_punti, snapshots, seed, insieme, v_range = args
    want, T = set(snapshots), max(snapshots)
    V = {t: [] for t in snapshots}
    F = {t: [] for t in snapshots}
    P = []
    with torch.no_grad():
        for i in range(n_punti):
            net = _rete(W, B, Rr, G, i, seed, insieme, v_range)
            for t in range(1, T + 1):
                eV, _ = net.step(v_left_22=None)
                if t in want:
                    V[t].append(net.v.clone().numpy())
                    F[t].append(R._aggregate_field(net, eV).numpy())
            P.append(net.Gp.clone().numpy())
    return dict(W=W, B=B, insieme=insieme,
                V={t: np.asarray(x) for t, x in V.items()},
                F={t: np.asarray(x) for t, x in F.items()},
                P=np.asarray(P))


def _rfc(args):
    """RFC con il protocollo di Rfc_and_ID.rfc_timeseries."""
    torch.set_num_threads(1)
    W, B, Rr, G, n_camp, T, seed, insieme, v_range = args
    agg, vm = [], []
    with torch.no_grad():
        for i in range(n_camp):
            net = _rete(W, B, Rr, G, i, seed, insieme, v_range,
                        g_omogenea=(i == 0))          # campione 0 omogeneo
            sv, sf = [], []
            for _ in range(T):
                eV, _ = net.step(v_left_22=None)
                sv.append(net.v.clone())
                sf.append(R._aggregate_field(net, eV))
            agg.append(R.pca_explained(torch.stack(sf).numpy()).sum())
            vm.append(R.pca_explained(torch.stack(sv).numpy()).sum())
    camp = 100 * (np.array(agg) - np.array(vm))       # RFC per campione, in %
    return dict(W=W, B=B, insieme=insieme, rfc_camp=camp,
                rfc_media=float(camp.mean()), rfc_camp0=float(camp[0]),
                n=n_camp)


# ═══════════════════════════════════════════════════════════════════════════
#  Misure
# ═══════════════════════════════════════════════════════════════════════════

def _misure_nuvola(V, F, k=10):
    h = gg.h0_profile(V, TOLS_H0)
    tv, mv, _ = R.id_of_cloud(V, k=k)
    tf, mf, _ = R.id_of_cloud(F, k=k)
    return dict(spread=h['spread'], collassata=R.is_collapsed(h['spread']),
                h0={t: h['counts'][float(t)] for t in TOLS_H0},
                id_v=(tv, mv), id_f=(tf, mf))


def _memoria_v0(Va, Vb):
    """Distanza mediana a parita' di g0, relativa a quella fra g0 diversi."""
    dim = Va.shape[1]
    appaiate = np.linalg.norm(Va - Vb, axis=1) / np.sqrt(dim)
    fra_g0 = pdist(Va) / np.sqrt(dim)
    num, den = float(np.median(appaiate)), float(np.median(fra_g0))
    return dict(appaiata=num, riferimento=den,
                indice=(num / den) if den > 0 else float('nan'))


def _ic95(x, n_boot=10000, seed=0):
    """Intervallo bootstrap al 95% per la media."""
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    m = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(1)
    return (float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)))


def analizza(nuvole, rfc=None, k=10):
    """nuvole: {(W,B): {insieme: risultato di _nuvola}}."""
    out = {}
    celle = list(nuvole) + [(w, b) for (w, b, _) in (rfc or {})
                            if (w, b) not in nuvole]
    for (W, B) in dict.fromkeys(celle):
        per_ins = nuvole.get((W, B), {})
        snaps = sorted(next(iter(per_ins.values()))['V']) if per_ins else []
        r = dict(W=W, B=B, misure={}, memoria={}, gp={})
        for ins, c in per_ins.items():
            r['misure'][ins] = {t: _misure_nuvola(c['V'][t], c['F'][t], k)
                                for t in snaps}
            r['gp'][ins] = gg.gp_saturation(c['P'])
        if 'paper' in per_ins and 'v_g' in per_ins:
            r['memoria'] = {t: _memoria_v0(per_ins['paper']['V'][t],
                                           per_ins['v_g']['V'][t])
                            for t in snaps}
        if rfc:
            r['rfc'] = {ins: x for (w, b, ins), x in rfc.items()
                        if (w, b) == (W, B)}
            if 'paper' in r['rfc'] and 'v_g' in r['rfc']:
                # campioni appaiati
                d = r['rfc']['v_g']['rfc_camp'] - r['rfc']['paper']['rfc_camp']
                r['rfc_diff'] = dict(media=float(d.mean()),
                                     ic95=_ic95(d), n=len(d))
        out[(W, B)] = r
    return out


# ═══════════════════════════════════════════════════════════════════════════
#  Stampa
# ═══════════════════════════════════════════════════════════════════════════

def _f(x, fmt):
    return '—' if (x is None or not np.isfinite(x)) else format(x, fmt)


def stampa(ris, n_punti, T):
    for (W, B), r in ris.items():
        print(f"\n{'='*100}\n  W={W:g}  B={B:g}\n{'='*100}")
        print(f"  {'insieme':<8}{'t':>6}{'spread (V)':>12}{'coll.':>7}"
              + ''.join(f"{f'H0@{t:.0e}':>10}" for t in TOLS_H0)
              + f"{'ID Vmem TwoNN/MLE':>20}{'ID campo TwoNN/MLE':>21}")
        for ins, per_t in r['misure'].items():
            for t, m in per_t.items():
                print(f"  {ins:<8}{t:>6}{m['spread']:>12.2e}"
                      f"{('sì' if m['collassata'] else 'no'):>7}"
                      + ''.join(f"{m['h0'][x]:>10d}" for x in TOLS_H0)
                      + f"{_f(m['id_v'][0], '.2f'):>11}/{_f(m['id_v'][1], '<8.2f')}"
                      + f"{_f(m['id_f'][0], '.2f'):>12}/{_f(m['id_f'][1], '<8.2f')}")
        if r['memoria']:
            print("\n  memoria di v0 ('paper' contro 'v_g', stessi g0):")
            for t, m in r['memoria'].items():
                print(f"    t={t:>5}: distanza appaiata {m['appaiata']:.2e} V, "
                      f"fra g0 diversi {m['riferimento']:.2e} V, "
                      f"indice {_f(m['indice'], '.3f')}")
        if r['gp']:
            print("\n  Gp all'ultimo istante:")
        for ins, g in r['gp'].items():
            print(f"    {ins:<8}{100*g['frac_zero']:6.1f}% a 0  "
                  f"{100*g['frac_max']:6.1f}% a 2G_r  "
                  f"{100*g['frac_interior']:6.1f}% interne")
        if r.get('rfc'):
            print("\n  RFC (protocollo di Rfc_and_ID, campione 0 a g omogenea):")
            for ins, x in r['rfc'].items():
                lo, hi = _ic95(x['rfc_camp'])
                print(f"    {ins:<8} media {x['rfc_media']:+7.2f}%  "
                      f"[IC95 {lo:+.2f}, {hi:+.2f}]   "
                      f"campione 0 {x['rfc_camp0']:+7.2f}%   (n={x['n']})")
            if 'rfc_diff' in r:
                d = r['rfc_diff']
                print(f"    differenza appaiata v_g - paper: {d['media']:+.2f} "
                      f"punti [IC95 {d['ic95'][0]:+.2f}, {d['ic95'][1]:+.2f}]")
        _controllo(W, B, r, n_punti, T)


def _controllo(W, B, r, n_punti, T):
    """L'insieme 'paper' deve riprodurre l'appendice."""
    m = r['misure'].get('paper', {}).get(5000)
    righe = []
    if m is not None and n_punti == 300 and T == 5000:
        if (W, B) in RIF:
            id_ref, sp_ref = RIF[(W, B)]
            ok = (abs(m['id_v'][1] - id_ref) < 0.01
                  and abs(m['spread'] - sp_ref) <= 0.006 * sp_ref)
            righe.append(f"ID MLE {m['id_v'][1]:.2f} (appendice {id_ref}), "
                         f"spread {m['spread']:.2e} (appendice {sp_ref:.2e}) "
                         f"-> {'OK' if ok else 'DIVERSO'}")
        elif (W, B) in COLLASSATE:
            ok = m['spread'] == 0.0
            righe.append(f"spread {m['spread']:.2e} (appendice 0) "
                         f"-> {'OK' if ok else 'DIVERSO'}")
    x = r.get('rfc', {}).get('paper')
    if x is not None and x['n'] == 32 and (W, B) in RIF_RFC:
        mr, s0 = RIF_RFC[(W, B)]
        ok = abs(x['rfc_media'] - mr) < 0.01 and abs(x['rfc_camp0'] - s0) < 0.01
        righe.append(f"RFC media {x['rfc_media']:+.2f}% (appendice {mr:+.2f}), "
                     f"campione 0 {x['rfc_camp0']:+.2f}% (appendice {s0:+.2f}) "
                     f"-> {'OK' if ok else 'DIVERSO'}")
    if righe:
        print("\n  CONTROLLO contro l'appendice (insieme 'paper'):")
        for s in righe:
            print("    " + s)
    elif m is not None:
        print("\n  (controllo contro l'appendice non eseguito: servono "
              "--punti 300 e T=5000, e una cella di cui sono noti i valori)")


# ═══════════════════════════════════════════════════════════════════════════

def _esegui(fun, jobs, workers):
    out = []
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers,
                                 initializer=gg._init_worker) as ex:
            for n, x in enumerate(ex.map(fun, jobs), 1):
                out.append(x); print(f"  [{n}/{len(jobs)}]", flush=True)
    else:
        for n, j in enumerate(jobs, 1):
            t0 = time.time(); out.append(fun(j))
            print(f"  [{n}/{len(jobs)}] {time.time()-t0:.0f} s", flush=True)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    p.add_argument('--celle', default=None,
                   help='"W,B;W,B;..." (default: le 4 celle di CELLE_DEFAULT)')
    p.add_argument('--griglia', action='store_true',
                   help='tutte le 25 celle PAPER_W x PAPER_B')
    p.add_argument('--insiemi', default=','.join(INSIEMI))
    p.add_argument('--punti', type=int, default=300)
    p.add_argument('--T', type=int, default=5000)
    p.add_argument('--snapshot', default='1000,3000,5000',
                   help='istanti di campionamento; il massimo e\' la durata')
    p.add_argument('--v-min', type=float, default=E_p * 1e3, help='mV')
    p.add_argument('--v-max', type=float, default=E_d * 1e3, help='mV')
    p.add_argument('--rfc', type=int, default=32,
                   help='campioni per l\'RFC (0 = non calcolarlo; 32 = appendice)')
    p.add_argument('--R', type=int, default=4)
    p.add_argument('--G', type=float, default=0.05)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--workers', type=int,
                   default=max(1, (os.cpu_count() or 2) // 2))
    p.add_argument('--salva', default='robustezza_v.pt')
    a = p.parse_args()

    if a.griglia:
        celle = [(W, B) for W in R.PAPER_W for B in R.PAPER_B]
    elif a.celle:
        celle = [tuple(float(x) for x in s.split(',')) for s in a.celle.split(';')]
    else:
        celle = CELLE_DEFAULT
    insiemi = [s.strip() for s in a.insiemi.split(',')]
    snaps = sorted({int(x) for x in a.snapshot.split(',')} | {a.T})
    snaps = [t for t in snaps if t <= a.T]
    v_range = (a.v_min * 1e-3, a.v_max * 1e-3)

    print(f"Celle {celle}\ninsiemi {insiemi}, {a.punti} punti, istanti {snaps}, "
          f"v0 ~ U({a.v_min:g}, {a.v_max:g}) mV, R={a.R}, G={a.G}")

    jobs = [(W, B, a.R, a.G, a.punti, tuple(snaps), a.seed, ins, v_range)
            for (W, B) in celle for ins in insiemi]
    nuvole = {}
    if a.punti > 0:
        for c in _esegui(_nuvola, jobs, a.workers):
            nuvole.setdefault((c['W'], c['B']), {})[c['insieme']] = c

    rfc = None
    if a.rfc > 0:
        jobs = [(W, B, a.R, a.G, a.rfc, a.T, a.seed, ins, v_range)
                for (W, B) in celle for ins in insiemi]
        rfc = {(x['W'], x['B'], x['insieme']): x
               for x in _esegui(_rfc, jobs, a.workers)}

    ris = analizza(nuvole, rfc)
    stampa(ris, a.punti, a.T)
    torch.save(dict(argomenti=vars(a), risultati=ris), R.out_path(a.salva))
    print(f"\nsalvato in {R.out_path(a.salva)}")


if __name__ == '__main__':
    main()

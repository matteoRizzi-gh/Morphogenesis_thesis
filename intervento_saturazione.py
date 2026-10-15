"""
intervento_saturazione.py
=========================
Test per INTERVENTO dell'ipotesi della Sez. 42.1: la dimensione della nuvola
segue la frazione di conduttanze interne (gradi di liberta' attivi).

"""

import argparse
import os
import time

import numpy as np
import torch
from scipy.stats import spearmanr

import Rfc_and_ID as R
import grid_and_gp as gg
from dynamics_t_sparse_gh import BioelectricLattice, G_r


class RetiBloccata(BioelectricLattice):
    """Rete in cui la conduttanza di alcune cellule e' fissata a un valore. """
    def __init__(self, idx, valore, **kw):
        super().__init__(**kw)
        self.idx = torch.as_tensor(idx, dtype=torch.long)
        self.valore = float(valore)

    def _blocca(self):
        if len(self.idx):
            self.Gp[self.idx] = self.valore

    def _update_Gp(self, eV):
        super()._update_Gp(eV)
        self._blocca()


def _nuvola(args):
    torch.set_num_threads(1)
    W, B, Rr, G, n_punti, T, seed, idx, lato = args
    valore = 2.0 * G_r if lato == 'max' else 0.0
    V, P = [], []
    with torch.no_grad():
        for i in range(n_punti):
            net = RetiBloccata(idx, valore, rows=11, cols=11,
                               G=G, R=Rr, W=W, B=B)
            net.reset()
            g = torch.Generator().manual_seed(int(seed) * 100000 + i)
            net.Gp = torch.rand(net.N, generator=g,
                                dtype=torch.float64) * 2.0 * G_r
            net._blocca()
            for _ in range(T):
                net.step(v_left_22=None)
            V.append(net.v.clone().numpy())
            P.append(net.Gp.clone().numpy())
    return np.asarray(V), np.asarray(P)


def _misure(V, P, idx):
    N = V.shape[1]
    liberi = np.setdiff1d(np.arange(N), idx)
    interne = (P > gg.GP_EDGE) & (P < 2.0 * G_r - gg.GP_EDGE)
    h = gg.h0_profile(V, (1e-6,))
    tw, ml, _ = R.id_of_cloud(V, k=10)
    return dict(
        attiva=float(interne.mean()),                  # su tutte le celle
        attiva_libere=float(interne[:, liberi].mean()),
        spread=h['spread'], collassata=R.is_collapsed(h['spread']),
        id_twonn=tw, id_mle=ml)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--W', type=float, default=10.0)
    p.add_argument('--B', type=float, default=0.002)
    p.add_argument('--livelli', default='0,15,30,60,90',
                   help='numeri di cellule bloccate')
    p.add_argument('--lati', default='max,zero')
    p.add_argument('--insiemi', type=int, default=3,
                   help='insiemi casuali di cellule per ogni livello')
    p.add_argument('--punti', type=int, default=300)
    p.add_argument('--T', type=int, default=5000)
    p.add_argument('--R', type=int, default=4)
    p.add_argument('--G', type=float, default=0.05)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--workers', type=int,
                   default=max(1, (os.cpu_count() or 2) // 2))
    p.add_argument('--salva', default='intervento_saturazione.pt')
    a = p.parse_args()

    livelli = [int(x) for x in a.livelli.split(',')]
    lati = [s.strip() for s in a.lati.split(',')]
    N = 121

    # Configurazioni. k = 0 e' la stessa per ogni lato e insieme: una sola.
    conf = []
    for k in livelli:
        if k == 0:
            conf.append((0, 'nessuno', 0, np.array([], dtype=int)))
            continue
        for lato in lati:
            for s in range(a.insiemi):
                rng = np.random.default_rng(1000 * k + s)
                conf.append((k, lato, s, np.sort(rng.choice(N, k, replace=False))))

    jobs = [(a.W, a.B, a.R, a.G, a.punti, a.T, a.seed, idx,
             lato if lato != 'nessuno' else 'max')
            for (k, lato, s, idx) in conf]
    print(f"W={a.W:g}  B={a.B:g}  R={a.R}  G={a.G}   {len(jobs)} nuvole da "
          f"{a.punti} punti, T={a.T}")

    t0 = time.time()
    if a.workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=a.workers,
                                 initializer=gg._init_worker) as ex:
            out = list(ex.map(_nuvola, jobs))
    else:
        out = []
        for n, j in enumerate(jobs, 1):
            out.append(_nuvola(j))
            print(f"  [{n}/{len(jobs)}] {time.time()-t0:.0f} s", flush=True)

    righe = []
    for (k, lato, s, idx), (V, P) in zip(conf, out):
        m = _misure(V, P, idx)
        righe.append(dict(k=k, lato=lato, insieme=s, idx=idx, **m))

    print(f"\n  {'k':>4}{'lato':>9}{'ins.':>6}{'attiva':>9}{'att. libere':>13}"
          f"{'spread (V)':>12}{'coll.':>7}{'TwoNN':>8}{'MLE':>8}")
    for r in righe:
        print(f"  {r['k']:>4}{r['lato']:>9}{r['insieme']:>6}"
              f"{100*r['attiva']:>8.1f}%{100*r['attiva_libere']:>12.1f}%"
              f"{r['spread']:>12.2e}{('sì' if r['collassata'] else 'no'):>7}"
              f"{r['id_twonn']:>8.2f}{r['id_mle']:>8.2f}")

    # Relazione fra frazione attiva e dimensione, a (W, B) fissati.
    vive = [r for r in righe if not r['collassata'] and np.isfinite(r['id_mle'])]
    print("\n  Spearman fra frazione attiva misurata e ID MLE (nuvole non collassate):")
    for nome, sel in [('tutte', vive)] + [(l, [r for r in vive if r['lato'] in (l, 'nessuno')])
                                          for l in lati]:
        if len(sel) >= 4:
            rs = spearmanr([r['attiva'] for r in sel], [r['id_mle'] for r in sel])
            print(f"    {nome:>6}: n={len(sel):2d}  rho_S={rs.statistic:+.3f}  "
                  f"p={rs.pvalue:.3g} (asintotico, indicativo)")

    base = [r for r in righe if r['k'] == 0][0]
    if a.punti == 300 and a.T == 5000 and (a.W, a.B) == (10.0, 0.002):
        ok = abs(base['id_mle'] - 44.72) < 0.01 and abs(base['spread'] - 3.54e-2) < 3e-4
        print(f"\n  controllo k=0: ID MLE {base['id_mle']:.2f} (appendice 44.72), "
              f"spread {base['spread']:.2e} (appendice 3.54e-02) "
              f"-> {'OK' if ok else 'DIVERSO'}")

    torch.save(dict(argomenti=vars(a), righe=righe), R.out_path(a.salva))
    print(f"\n  salvato in {R.out_path(a.salva)}")


if __name__ == '__main__':
    main()

    
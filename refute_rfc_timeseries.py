"""
refute_rfc_timeseries.py
========================

refute_rfc esegue i controlli (colonne mescolate / gaussiana / uniforme) su una
NUVOLA spaziale di condizioni iniziali a t fisso, e vi calcola un "RFC analogo".
L'RFC del paper (e la sua replica in Rfc_and_ID.rfc_timeseries) e' invece
calcolato sulla SERIE TEMPORALE di una singola traiettoria: prime 3 PC della
serie del campo meno prime 3 PC della serie di Vmem.

"""

import numpy as np
from itertools import product
import torch


import Rfc_and_ID as R
from refute_rfc import phi_batch, control_clouds, control_clouds_iaaft
from dynamics_t_sparse_gh import BioelectricLattice


def _first3(X):
    """Somma della varianza spiegata dalle prime 3 PC, scartando le colonne
    degeneri che i controlli possono produrre."""
    ev = R.pca_explained(X, ndims=3, drop_degenerate=True)
    return float(np.sum(ev))


def rfc_of_series(V, net):
    """RFC del paper su UNA serie temporale di Vmem [T,121]."""
    F = phi_batch(V, net)                      # [T,121] campo aggregato
    f_field = 100.0 * _first3(F)
    f_vmem  = 100.0 * _first3(V)
    return f_field - f_vmem, f_field, f_vmem


def _paired_test(a, b, max_exact=20, n_mc=100000, seed=0):
    """Test appaiato sulle differenze a - b, senza assumere normalita'. """
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 2:
        return dict(n=n, mean=float('nan'), sem=float('nan'),
                    p=float('nan'), exact=False)

    obs = abs(float(d.mean()))
    sem = float(d.std(ddof=1) / np.sqrt(n))

    if n <= max_exact:
        hits = sum(1 for s in product((-1.0, 1.0), repeat=n)
                   if abs(float(np.dot(s, d) / n)) >= obs - 1e-15)
        p = hits / (2 ** n)
        exact = True
    else:
        rng = np.random.default_rng(seed)
        s = rng.choice((-1.0, 1.0), size=(n_mc, n))
        p = float((np.abs(s @ d / n) >= obs - 1e-15).mean())
        exact = False

    return dict(n=n, mean=float(d.mean()), sem=sem, p=p, exact=exact)


def refute_rfc_timeseries(W=1000.0, B=5e-4, Rr=4, G=0.05, T=5000,
                          n_samples=8, seed=0, verbose=True,
                          controls='iaaft'):
    """
    Applica i controlli alla serie temporale e calcola l'RFC del paper su
    ciascuno. Media su n_samples traiettorie reali (campione 0 omogeneo, come
    rfc_timeseries); per ogni traiettoria i controlli sono generati dalle sue
    stesse colonne

    controls='iaaft'   surrogati che preservano l'autocorrelazione
                       (control_clouds_iaaft). E' il nullo corretto per una
                       serie temporale.
    controls='shuffle' i controlli originali (control_clouds), che distruggono
                       anche l'autocorrelazione. Tenuto per confronto.
    """
    net = BioelectricLattice(rows=11, cols=11, G=G, R=Rr, W=W, B=B)
    maker = control_clouds_iaaft if controls == 'iaaft' else control_clouds

    acc = None
    names = None
    paper_native = []

    for i in range(n_samples):
        ts = R.run_timeseries(W, B, Rr, G, seed=seed*100000 + i, T=T,
                              homogeneous=(i == 0))
        V_real = ts['vmem']

        paper_native.append(
            100.0 * (_first3(ts['ev_agg']) - _first3(V_real)))

        ctrl = maker(V_real, seed=seed*100000 + i)
        if acc is None:
            names = list(ctrl.keys())
            acc = {nm: dict(rfc=[], field=[], vmem=[]) for nm in names}

        for nm, Vc in ctrl.items():
            rfc, ff, fv = rfc_of_series(Vc, net)
            acc[nm]['rfc'].append(rfc)
            acc[nm]['field'].append(ff)
            acc[nm]['vmem'].append(fv)

        if verbose:
            print(f"    {i+1}/{n_samples}")

    m = lambda a: float(np.nanmean(a)); s = lambda a: float(np.nanstd(a))

    print(f"\n{'='*92}")
    print(f"  RFC DEL PAPER SU CONTROLLI SENZA DINAMICA (serie temporale)")
    print(f"  W={W} B={B} R={Rr} G={G}   {n_samples} traiettorie x T={T}"
          f"   controlli: {controls}")
    print(f"{'='*92}")
    print(f"  ancora  -- RFC del paper, campo nativo ev_agg vs Vmem: "
          f"{m(paper_native):+.2f}%  (+/-{s(paper_native):.2f})")
    print(f"  (la riga 'dinamica vera' usa Phi(Vmem) invece di ev_agg: se")
    print(f"   coincide con l'ancora, Phi riproduce il campo del paper)\n")
    print(f"  {'variante':>22}{'first3 campo':>15}{'first3 Vmem':>14}{'RFC':>12}")
    print("  " + "-"*78)
    for nm in names:
        d = acc[nm]
        print(f"  {nm:>22}{m(d['field']):11.2f}%   {m(d['vmem']):10.2f}%  "
              f"{m(d['rfc']):+9.2f}%")

    ref = 'dinamica vera'
    paired = {}
    print(f"\n  --- differenza appaiata contro '{ref}' ---")
    print(f"  {'variante':>22}{'d(RFC)':>12}{'SEM':>9}{'p':>9}{'':>6}"
          f"{'d(first3 campo)':>18}{'d(first3 Vmem)':>17}")
    for nm in names:
        if nm == ref:
            continue
        pr = _paired_test(acc[ref]['rfc'], acc[nm]['rfc'], seed=seed)
        pf = _paired_test(acc[ref]['field'], acc[nm]['field'], seed=seed)
        pv = _paired_test(acc[ref]['vmem'], acc[nm]['vmem'], seed=seed)
        paired[nm] = dict(rfc=pr, field=pf, vmem=pv)
        tag = 'es.' if pr['exact'] else 'MC'
        print(f"  {nm:>22}{pr['mean']:+12.2f}{pr['sem']:9.2f}{pr['p']:9.4f}"
              f"{tag:>6}{pf['mean']:+18.2f}{pv['mean']:+17.2f}")

    p_floor = 2.0 / 2**n_samples
    if n_samples <= 20:
        print(f"\n  p = sign-flip appaiato a due code, enumerazione esatta "
              f"su 2^{n_samples} segni.")
        print(f"  p minimo ottenibile: {p_floor:.2e}")
    else:
        print(f"\n  p = sign-flip appaiato a due code, Monte Carlo su "
              f"{100000} estrazioni (n={n_samples}).")
        print(f"  p=0.0000 significa 'sotto la risoluzione del Monte Carlo', "
              f"cioe' < 1e-5.")

    if controls == 'iaaft':
        print("\n  'fasi comuni' preserva la covarianza a precisione di macchina,")
        print("  quindi d(first3 Vmem) deve essere ~0 per costruzione: e' un")
        print("  controllo di correttezza, non un risultato. Il contenuto del")
        print("  test sta in d(first3 campo), che NON e' determinato dalla")
        print("  covarianza perche' Phi e' non lineare.")

    return dict(paper_native=paper_native, acc=acc, names=names,
                paired=paired, controls=controls, W=W, B=B, R=Rr, G=G, T=T,
                n_samples=n_samples)



if __name__ == '__main__':

    N_SAMPLES = 32   
    
    print("\n##### STIGMERGICO (forte) #####")
    r_stig = refute_rfc_timeseries(W=1000.0, B=5e-4, T=5000,
                                   n_samples=N_SAMPLES, verbose=False,
                                   controls='iaaft')

    print("\n##### MOSAIC (debole) #####")
    r_mos = refute_rfc_timeseries(W=10.0, B=0.02, T=5000,
                                  n_samples=N_SAMPLES, verbose=False,
                                  controls='iaaft')

    print("\n##### STIGMERGICO, controlli precedenti #####")
    r_shuf_stig = refute_rfc_timeseries(W=1000.0, B=5e-4, T=5000,
                                   n_samples=N_SAMPLES, verbose=False,
                                   controls='shuffle')

    
    print("\n##### MOSAIC (debole) #####")
    r_shuf_mos = refute_rfc_timeseries(W=10.0, B=0.02, T=5000,
                                  n_samples=N_SAMPLES, verbose=False,
                                  controls='shuffle')
    

    torch.save(dict(stigmergico=r_stig, mosaic=r_mos,
                    stigmergico_shuffle=r_shuf_stig, 
                    mosaic_shuffle = r_shuf_mos, ),
               R.out_path('refute_rfc_timeseries.pt'))
    



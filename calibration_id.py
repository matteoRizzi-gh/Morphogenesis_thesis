"""
calibration_id.py
=================
Calibra TwoNN e MLE su varieta' di dimensione NOTA, alle stesse condizioni
delle misure reali (N punti in R^121). Serve a convertire una stima in una
dimensione vera, e a stabilire fino a che d l'inversione e' affidabile.

"""

import os
import numpy as np
from sklearn.preprocessing import StandardScaler

import Rfc_and_ID as R

AMBIENT = 121
N_SEEDS = 10        # realizzazioni indipendenti mediate per ogni (d, N)


# ═══════════════════════════════════════════════════════════════════════════
#  Famiglie di varieta' sintetiche a dimensione nota
# ═══════════════════════════════════════════════════════════════════════════

def synth_linear(d, n, ambient=AMBIENT, rng=None):
    """Piano d-dimensionale immerso a caso: il caso piu' facile."""
    rng = rng or np.random.default_rng(0)
    Z = rng.normal(size=(n, d))
    A = rng.normal(size=(d, ambient))
    return Z @ A


def synth_curved(d, n, ambient=AMBIENT, rng=None):
    """Varieta' d-dimensionale CURVA
    """
    rng = rng or np.random.default_rng(0)
    Z = rng.uniform(-1, 1, size=(n, d))
    F = np.hstack([Z, np.sin(3*Z), np.cos(2*Z), Z**2])
    return F @ rng.normal(size=(F.shape[1], ambient))


def synth_curved_strong(d, n, ambient=AMBIENT, rng=None):
    """Curvatura piu' forte: frequenze piu' alte e piu' armoniche"""
    rng = rng or np.random.default_rng(0)
    Z = rng.uniform(-1, 1, size=(n, d))
    F = np.hstack([Z, np.sin(5*Z), np.cos(4*Z), np.sin(9*Z), Z**2, Z**3])
    return F @ rng.normal(size=(F.shape[1], ambient))


def synth_sphere(d, n, ambient=AMBIENT, rng=None):
    """Sfera d-dimensionale S^d."""
    rng = rng or np.random.default_rng(0)
    Z = rng.normal(size=(n, d + 1))
    Z /= np.linalg.norm(Z, axis=1, keepdims=True)      # punti su S^d
    A = rng.normal(size=(d + 1, ambient))
    return Z @ A


def synth_torus(d, n, ambient=AMBIENT, rng=None):
    """Toro d-dimensionale T^d = prodotto di d cerchi"""
    rng = rng or np.random.default_rng(0)
    theta = rng.uniform(0.0, 2*np.pi, size=(n, d))
    Z = np.concatenate([np.cos(theta), np.sin(theta)], axis=1)   # [n, 2d]
    A = rng.normal(size=(2 * d, ambient))
    return Z @ A


FAMILIES = (synth_linear, synth_curved, synth_curved_strong,
            synth_sphere, synth_torus)


# ═══════════════════════════════════════════════════════════════════════════
#  Calibrazione
# ═══════════════════════════════════════════════════════════════════════════

def calibrate(ds=(2, 5, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100, 121),
              ns=(250, 500, 1000, 2000), k=10, seed=0,
              generator=synth_curved, standardize=False,
              n_seeds=N_SEEDS, verbose=True):
    """Curva stima-contro-dimensione-vera, mediata su n_seeds realizzazioni.

    table[(d, n)] = (twonn_mean, mle_mean, twonn_sd, mle_sd)

    """
    if verbose:
        print(f"\n{'='*86}\n  CALIBRAZIONE  ({generator.__name__}, R^{AMBIENT}, "
              f"{n_seeds} semi)\n{'='*86}")
        print(f"  {'d vero':>7}" + "".join(f"{f'N={n}':>26}" for n in ns))
        print(f"  {'':>7}" + "".join(f"{'TwoNN':>13}{'MLE':>13}" for _ in ns))

    table = {}
    for d in ds:
        row = f"  {d:7d}"
        for n in ns:
            t2s, mls = [], []
            for s in range(n_seeds):
                rng = np.random.default_rng(seed * 100003 + s)
                X = generator(d, n, AMBIENT, rng)
                if standardize:
                    X = StandardScaler().fit_transform(X)
                t2s.append(R.twonn(X))
                mls.append(R.mle_id(X, k=k))
            t2s, mls = np.asarray(t2s, float), np.asarray(mls, float)
            table[(d, n)] = (float(np.nanmean(t2s)), float(np.nanmean(mls)),
                             float(np.nanstd(t2s, ddof=1)) if n_seeds > 1 else 0.0,
                             float(np.nanstd(mls, ddof=1)) if n_seeds > 1 else 0.0)
            row += (f"{table[(d, n)][0]:8.2f}+-{table[(d, n)][2]:<3.2f}"
                    f"{table[(d, n)][1]:8.2f}+-{table[(d, n)][3]:<3.2f}")
        if verbose:
            print(row)
    return table


def _curve(table, n, estimator='mle'):
    """Estrae (ds, stime, sd) per una data numerosita', gia' filtrata dai
    valori non finiti e ordinata per d crescente."""
    j = 1 if estimator == 'mle' else 0
    js = 3 if estimator == 'mle' else 2
    pts = sorted((d, v[j], v[js]) for (d, nn), v in table.items() if nn == n)
    ds = np.array([p[0] for p in pts], dtype=float)
    est = np.array([p[1] for p in pts], dtype=float)
    sd = np.array([p[2] for p in pts], dtype=float)
    ok = np.isfinite(est)
    return ds[ok], est[ok], sd[ok]


def monotone_prefix(ds, est):
    """Piu' lungo tratto INIZIALE strettamente crescente. """
    bad = np.where(np.diff(est) <= 0)[0]
    cut = int(bad[0]) + 1 if len(bad) else len(est)
    return ds[:cut], est[:cut], (int(ds[bad[0]]) if len(bad) else None)


# ═══════════════════════════════════════════════════════════════════════════
#  Inversione e diagnostica
# ═══════════════════════════════════════════════════════════════════════════

def invert(table, measured, n, estimator='mle', strict=False):
    """Dato un valore misurato a un dato N, quale d vero lo produce?
    Interpolazione lineare sulla curva di calibrazione"""
    ds, est, _ = _curve(table, n, estimator)
    if len(est) < 2:
        return float('nan')

    if strict:
        bad = np.where(np.diff(est) <= 0)[0]
        if len(bad):
            raise ValueError(
                f"curva non monotona a N={n} ({estimator}): "
                f"{[(float(ds[i]), float(est[i]), float(est[i+1])) for i in bad]}")

    ds, est, _ = monotone_prefix(ds, est)
    if len(est) < 2:
        return float('nan')
    if measured <= est[0]:
        return float(ds[0])
    if measured >= est[-1]:
        return float('inf')
    return float(np.interp(measured, est, ds))


def inversion_uncertainty(table, measured, n, estimator='mle'):
    """Incertezza su d indotta dal rumore campionario della curva. """
    ds_all, est_all, sd_all = _curve(table, n, estimator)
    ds, est, _ = monotone_prefix(ds_all, est_all)
    sd = sd_all[:len(est)]
    d_hat = invert(table, measured, n, estimator)
    if not np.isfinite(d_hat) or len(est) < 2:
        return dict(d=d_hat, slope=float('nan'), amplification=float('nan'),
                    sd_est=float('nan'), d_err=float('nan'))
    i = int(np.clip(np.searchsorted(est, measured) - 1, 0, len(est) - 2))
    s = (est[i+1] - est[i]) / (ds[i+1] - ds[i])
    sd_loc = float(np.mean(sd[i:i+2])) if len(sd) > i + 1 else float('nan')
    return dict(d=d_hat, slope=float(s),
                amplification=float(1.0 / s) if s > 0 else float('inf'),
                sd_est=sd_loc,
                d_err=float(sd_loc / s) if s > 0 else float('inf'))


def usable_range(table, n, tol=0.15, estimator='mle'):
    """Fino a che d la stima GREZZA e' entro tol di errore relativo. """
    j = 1 if estimator == 'mle' else 0
    last = None
    for d in sorted({d for d, nn in table if nn == n}):
        v = table[(d, n)][j]
        if np.isfinite(v) and abs(v - d)/d <= tol:
            last = d
    return last


# ═══════════════════════════════════════════════════════════════════════════
#  Misure dal disco
# ═══════════════════════════════════════════════════════════════════════════

def load_measurements():

    import torch
    out = []
    for fname, label, entries in (
            ('cloud_stigmergic.pt', 'stigmergico',
             [(3000, 'vmem'), (5000, 'vmem'), (5000, 'field')]),
            ('cloud_mosaic.pt', 'mosaic',
             [(1000, 'vmem'), (1000, 'field')]),
    ):
        path = R.out_path(fname)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} non trovato. Lancia run_parallel_rfc_and_id "
                f"con MODE='cloud'.")
        d = torch.load(path, weights_only=False)
        for t, var in entries:
            if t not in d['cloud']:
                raise KeyError(f"{fname}: istantaneo t={t} assente "
                               f"(presenti: {sorted(d['cloud'])})")
            snap = d['cloud'][t]
            rec = snap[var]

            spread = snap.get('vmem_spread')
            if spread is None:
                collapsed = bool(rec['diag']['collapsed'])
                print(f"  [{fname} t={t}] 'vmem_spread' assente: uso il "
                      f"criterio H0 di cloud_diagnostics. Rilancia "
                      f"run_parallel_rfc_and_id con MODE='cloud' per avere "
                      f"il criterio uniforme.")
            else:
                collapsed = R.is_collapsed(spread)

            out.append(dict(
                label=f"{label} {var} t={t}",
                value=float(rec['mle']),
                n_points=int(d['n_points']),
                spread=(float(spread) if spread is not None else float('nan')),
                collapsed=collapsed))
    return out

# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':

    N_REF = 1000        # deve coincidere con n_points delle nuvole misurate
    DS = (2, 5, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100, 121)
    W = max(15, max(len(g.__name__) for g in FAMILIES) + 2)
    names = [g.__name__ for g in FAMILIES]

    # ------------------------------------------------------------------
    # Misure: si scartano qui, una volta sola, quelle non invertibili.
    # Nuvole collassate: mle = 0, non c'e' niente da invertire.
    # Numerosita' diversa da N_REF: il bias dipende da n, invertire con la
    # curva sbagliata da' un risultato falso senza segnalarlo.
    # ------------------------------------------------------------------
    measurements = load_measurements()
    usable, skipped = [], []
    for m in measurements:
        if m['collapsed']:
            sp = ('' if not np.isfinite(m['spread'])
                  else f", spread {m['spread']:.2e} V < {R.SPREAD_COLLAPSED:g}")
            skipped.append((m['label'], f"nuvola collassata{sp}"))
        elif m['n_points'] != N_REF:
            skipped.append((m['label'], f"nuvola di {m['n_points']} punti, "
                                        f"taratura a N={N_REF}"))
        else:
            usable.append(m)

    if skipped:
        print(f"\n{'='*78}\n  MISURE ESCLUSE\n{'='*78}")
        for label, why in skipped:
            print(f"  {label:>28}: {why}")
    if not usable:
        raise SystemExit("nessuna misura invertibile: controlla i .pt "
                         "delle nuvole")

    # ------------------------------------------------------------------
    # Una tabella di taratura per famiglia, alla sola numerosita' N_REF.
    # ------------------------------------------------------------------
    tables, inversions, uncert, cuts = {}, {}, {}, {}
    for gen in FAMILIES:
        nm = gen.__name__
        tables[nm] = calibrate(ds=DS, ns=(N_REF,), generator=gen)
        ds_a, est_a, _ = _curve(tables[nm], N_REF, 'mle')
        _, _, cut = monotone_prefix(ds_a, est_a)
        cuts[nm] = cut
        inversions[nm] = {m['label']: invert(tables[nm], m['value'], N_REF)
                          for m in usable}
        uncert[nm] = {m['label']: inversion_uncertainty(
                          tables[nm], m['value'], N_REF) for m in usable}

    # ------------------------------------------------------------------
    # Dominio di validita' di ciascuna curva.
    # ------------------------------------------------------------------
    print(f"\n{'='*100}\n  DOMINIO DELLE CURVE  (N={N_REF}, MLE, {N_SEEDS} semi)"
          f"\n{'='*100}")
    print(f"  {'famiglia':>22}{'monotona fino a':>18}{'tetto stima':>14}"
          f"{'d entro 15%':>14}")
    for nm in names:
        ds_a, est_a, _ = _curve(tables[nm], N_REF, 'mle')
        ds_m, est_m, _ = monotone_prefix(ds_a, est_a)
        print(f"  {nm:>22}{(str(cuts[nm]) if cuts[nm] else 'tutta'):>18}"
              f"{est_m[-1]:14.2f}{str(usable_range(tables[nm], N_REF)):>14}")
    print("\n  'tetto stima' = massimo invertibile: oltre, la curva ha smesso")
    print("  di salire e la misura e' fuori dal potere risolutivo.")


    # ------------------------------------------------------------------
    # Inversione cross-famiglia.
    # ------------------------------------------------------------------
    print(f"\n{'='*100}\n  INVERSIONE CROSS-FAMIGLIA  (N={N_REF}, MLE)\n{'='*100}")
    print(f"  {'misura':>28}{'stima':>9}" + "".join(f"{n:>{W}}" for n in names))
    for m in usable:
        row = f"  {m['label']:>28}{m['value']:9.2f}"
        for nm in names:
            d = inversions[nm][m['label']]
            row += f"{'fuori':>{W}}" if d == float('inf') else f"{d:{W}.1f}"
        print(row)

    print(f"\n  incertezza indotta dal rumore della curva (+/- d):")
    print(f"  {'misura':>28}{'':>9}" + "".join(f"{n:>{W}}" for n in names))
    for m in usable:
        row = f"  {m['label']:>28}{'':>9}"
        for nm in names:
            u = uncert[nm][m['label']]
            row += (f"{'—':>{W}}" if not np.isfinite(u['d_err'])
                    else f"{'+-' + format(u['d_err'], '.1f'):>{W}}")
        print(row)

    print(f"\n  amplificazione 1/pendenza nel punto di inversione:")
    print(f"  {'misura':>28}{'':>9}" + "".join(f"{n:>{W}}" for n in names))
    for m in usable:
        row = f"  {m['label']:>28}{'':>9}"
        for nm in names:
            u = uncert[nm][m['label']]
            row += (f"{'—':>{W}}" if not np.isfinite(u['amplification'])
                    else f"{u['amplification']:{W}.1f}")
        print(row)
    print("\n  Amplificazione alta = curva piatta = inversione poco informativa,")
    print("  indipendentemente da quanto sia grande il bias.")

    # ------------------------------------------------------------------
    # Rapporti fra misure. 
    # ------------------------------------------------------------------
    order = sorted(usable, key=lambda m: -m['value'])
    pairs = [(order[i], order[j])
             for i in range(len(order)) for j in range(i + 1, len(order))]

    if pairs:
        print(f"\n{'='*118}\n  RAPPORTI FRA MISURE, PER FAMIGLIA\n{'='*118}")
        print(f"  {'coppia':>52}" + "".join(f"{n:>{W}}" for n in names)
              + f"{'escursione':>16}")
        for a, b in pairs:
            lab = f"{a['label']} / {b['label']}"
            row, vals = f"  {lab:>52}", []
            for nm in names:
                da, db = inversions[nm][a['label']], inversions[nm][b['label']]
                if da == float('inf') or db == float('inf') or db <= 0:
                    row += f"{'—':>{W}}"
                else:
                    vals.append(da / db)
                    row += f"{da/db:{W}.2f}"
            row += (f"{f'{min(vals):.1f}-{max(vals):.1f}':>16}" if vals
                    else f"{'—':>16}")
            print(row)




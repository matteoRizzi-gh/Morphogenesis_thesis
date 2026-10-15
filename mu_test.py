"""
mu_test.py
==========

"""

import os
import numpy as np
from sklearn.neighbors import NearestNeighbors

AMBIENT = 121
EST_THRESH = 3.0    # soglia sul rapporto TwoNN/Levina-Bickel: tipo B se >=
KS_THRESH  = 1.0    # KS/critico: sopra 1 la Pareto e' rifiutata al 5%

# ═══════════════════════════════════════════════════════════════════════
#  mu e le stime di TwoNN
# ═══════════════════════════════════════════════════════════════════════

def mu_ratios(X, drop_zero=True, drop_ties=True):
    """mu_i = r2_i/r1_i sui due vicini piu' prossimi, ordinati.

    drop_zero  scarta i punti con r1 = 0 (duplicati esatti): il rapporto non
               e' definito.
    drop_ties  scarta i punti con r2 = r1 (mu = 1 esatto): log(mu) = 0, e la
               MLE non e' definita su quei termini.

    """
    X = np.asarray(X, dtype=np.float64)
    d, _ = NearestNeighbors(n_neighbors=3).fit(X).kneighbors(X)
    r1, r2 = d[:, 1], d[:, 2]
    ok = np.ones(len(r1), dtype=bool)
    if drop_zero:
        ok &= (r1 > 0)
    if drop_ties:
        ok &= (r2 > r1)
    return np.sort(r2[ok] / r1[ok])


def twonn_mle(mu):
    """
    MLE esatta di d sotto P(mu) = d*mu^(-d-1):  d_hat = n / sum(log mu).

    """
    mu = np.asarray(mu, dtype=np.float64)
    lm = np.log(mu)
    s = lm.sum()
    return float(len(lm) / s) if len(lm) and s > 0 else np.nan


def twonn_regression(mu, discard=0.10):
    """Pendenza di log(mu) contro -log(1-F)."""
    m = len(mu)
    if m < 10:
        return np.nan
    keep = int(m * (1.0 - discard))
    F = np.arange(1, m + 1) / m
    x, y = np.log(mu[:keep]), -np.log1p(-F[:keep])
    den = (x * x).sum()
    return float((x * y).sum() / den) if den > 0 else np.nan


# ═══════════════════════════════════════════════════════════════════════
#  Il test
# ═══════════════════════════════════════════════════════════════════════

def ks_statistic(mu, d):
    """Distanza KS fra la CDF empirica dei mu e F(mu) = 1 - mu^(-d)."""
    mu = np.sort(np.asarray(mu, dtype=np.float64))
    n = len(mu)
    F = 1.0 - mu ** (-d)
    i = np.arange(1, n + 1)
    return float(np.max(np.maximum(i/n - F, F - (i-1)/n)))


def pareto_test(mu, n_boot=500, seed=0, n_raw=None):
    """
    Aderenza alla Pareto CON parametro stimato dai dati.

    """
    rng = np.random.default_rng(seed)
    mu = np.asarray(mu, dtype=np.float64)
    n = len(mu)
    d_hat = twonn_mle(mu)
    if not np.isfinite(d_hat) or d_hat <= 0:
        return dict(d_hat=np.nan, ks=np.nan, p_value=np.nan, n=n,
                    ks_crit_05=np.nan, frac_mu_near_1=np.nan,
                    mu_median=np.nan, frac_dropped=np.nan)
    ks_obs = ks_statistic(mu, d_hat)

    null = np.empty(n_boot)
    for b in range(n_boot):
        u = rng.random(n)                       # inversione: mu = (1-u)^(-1/d)
        sim = np.sort((1.0 - u) ** (-1.0 / d_hat))
        d_sim = twonn_mle(sim)
        null[b] = ks_statistic(sim, d_sim) if np.isfinite(d_sim) else np.inf

    return dict(d_hat=float(d_hat), ks=float(ks_obs),
                p_value=float((null >= ks_obs).mean()), n=n,
                ks_crit_05=float(np.quantile(null, 0.95)),
                frac_mu_near_1=float((mu < 1.05).mean()),
                mu_median=float(np.median(mu)),
                frac_dropped=(float(1.0 - n / n_raw)
                              if n_raw else float('nan')))


def diagnose(X, n_boot=500, seed=0, k_mle=10,
             est_thresh=EST_THRESH, ks_thresh=KS_THRESH):
    """
    Verdetto sulla dimensione dell'effetto, con DUE discriminanti.

    """
    from Rfc_and_ID import mle_id
    mu = mu_ratios(X)
    t = pareto_test(mu, n_boot=n_boot, seed=seed, n_raw=len(X))
    t['twonn_regression'] = twonn_regression(mu)
    t['levina_bickel'] = mle_id(X, k=k_mle)
    t['k_mle'] = k_mle

    t['ks_ratio'] = (t['ks'] / t['ks_crit_05']
                     if np.isfinite(t['ks_crit_05']) and t['ks_crit_05'] > 0
                     else np.nan)
    lb = t['levina_bickel']
    t['estimator_ratio'] = (t['d_hat'] / lb
                            if np.isfinite(lb) and lb > 0 else np.nan)

    r_est, r_ks = t['estimator_ratio'], t['ks_ratio']
    if not (np.isfinite(r_est) and np.isfinite(r_ks)):
        t['violation'] = 'indeterminato'
    elif r_est >= est_thresh:
        t['violation'] = 'tipo B'
    elif r_ks < ks_thresh:
        t['violation'] = 'nessuna'
    else:
        t['violation'] = 'tipo A'

    t['twonn_usable'] = t['violation'] in ('nessuna', 'tipo A')
    t['note'] = {
        'nessuna': 'Pareto compatibile: TwoNN valido',
        'tipo A':  'localita persa: TwoNN distorto ma monotono, usare nei confronti',
        'tipo B':  'campionamento deterministico: TwoNN inutilizzabile',
        'indeterminato': 'stima non definita',
    }[t['violation']]
    return t

# ═══════════════════════════════════════════════════════════════════════
#  Casi di prova a struttura nota
# ═══════════════════════════════════════════════════════════════════════

def case_linear(d, n, rng, ambient=AMBIENT):
    """Piano d-dimensionale, punti indipendenti."""
    return rng.normal(size=(n, d)) @ rng.normal(size=(d, ambient))


def case_curve_regular(n, rng, ambient=AMBIENT):
    """Curva 1D a passo COSTANTE: r1 ~ r2, tipo B."""
    t = np.linspace(0, 10, n)
    A = rng.normal(size=(3, ambient))
    return np.stack([np.sin(t), np.cos(1.7*t), t], 1) @ A


def case_curve_random(n, rng, ambient=AMBIENT):
    """Stessa curva, tempi casuali: la regolarita' e' rotta, d ~ 1."""
    t = np.sort(rng.uniform(0, 10, n))
    A = rng.normal(size=(3, ambient))
    return np.stack([np.sin(t), np.cos(1.7*t), t], 1) @ A


def case_two_clusters(d, n, rng, ambient=AMBIENT, gap=20.0):
    """
    Due gruppi separati. 
    """
    A = rng.normal(size=(d, ambient))
    Z = rng.normal(size=(n, d))
    Z[n//2:] += gap
    return Z @ A


def case_grid(d, n_side, rng, ambient=AMBIENT):
    """Reticolo regolare: r1 ~ r2 senza struttura di traiettoria, tipo B."""
    axes = [np.linspace(0, 1, n_side)] * d
    Z = np.stack(np.meshgrid(*axes), -1).reshape(-1, d)
    return Z @ rng.normal(size=(d, ambient))


def case_curve_jittered(n, rng, ambient=AMBIENT, jitter=1e-3):
    """
    Curva a passo costante con rumore: caso intermedio fra A e B. Serve a
    vedere DOVE la soglia cade: con jitter crescente il tipo B deve
    degradare verso il tipo A.
    """
    X = case_curve_regular(n, rng, ambient)
    return X + rng.normal(scale=jitter*np.abs(X).mean(), size=X.shape)


def calibrate_on_synthetic(n=1000, n_boot=300, seed=0,
                           est_thresh=EST_THRESH):
    """Il test riconosce i casi di cui si conosce la risposta?"""
    rng = np.random.default_rng(seed)
    cases = [
        ('piano d=2',             case_linear(2,  n, rng),        'nessuna', 2),
        ('piano d=10',            case_linear(10, n, rng),        'tipo A',  10),
        ('piano d=30',            case_linear(30, n, rng),        'tipo A',  30),
        ('due gruppi d=5',        case_two_clusters(5, n, rng),   'nessuna', 5),
        ('curva, tempi casuali',  case_curve_random(n, rng),      'nessuna', 1),
        ('curva, jitter 1e-3',    case_curve_jittered(n, rng),    'tipo B',  1),
        ('curva, passo costante', case_curve_regular(n, rng),     'tipo B',  1),
        ('reticolo d=3',          case_grid(3, 10, rng),          'tipo B',  3),
    ]
    print(f"\n{'='*120}\n  IL TEST RICONOSCE I CASI NOTI?  "
          f"(n={n}, soglia sul rapporto = {est_thresh})\n{'='*120}")
    print(f"  {'caso':>22}{'d vero':>8}{'TwoNN':>9}{'Lev-Bick':>10}"
          f"{'rapporto':>10}{'KS/crit':>9}{'mu~1':>7}"
          f"{'atteso':>10}{'trovato':>10}{'':>10}")
    out = []
    for name, X, expected, d_true in cases:
        r = diagnose(X, n_boot=n_boot, seed=seed, est_thresh=est_thresh)
        agree = (r['violation'] == expected)
        out.append(dict(name=name, d_true=d_true, expected=expected,
                        agree=agree, **r))
        print(f"  {name:>22}{d_true:8d}{r['d_hat']:9.2f}"
              f"{r['levina_bickel']:10.2f}{r['estimator_ratio']:10.1f}"
              f"{r['ks_ratio']:9.2f}{100*r['frac_mu_near_1']:6.0f}%"
              f"{expected:>10}{r['violation']:>10}"
              f"{'  OK' if agree else '  SBAGLIA':>10}")
    n_ok = sum(o['agree'] for o in out)
    print(f"\n  {n_ok}/{len(out)} casi riconosciuti correttamente")
    print("  Il KS da solo dava 7/8: mancava 'curva, jitter 1e-3' (KS/crit")
    print("  2.03, in mezzo ai piani buoni, ma TwoNN 24.13 su d vero 1).")
    return out

# ═══════════════════════════════════════════════════════════════════════
#  Calibrazione della soglia: dove separa davvero?
# ═══════════════════════════════════════════════════════════════════════

def scan_threshold(ds=(2, 5, 10, 20, 30, 50), ns=(250, 500, 1000, 2000),
                   n_boot=200, seed=0):
    """
    Fin dove arriva il rapporto TwoNN/Levina-Bickel nei casi in cui TwoNN
    FUNZIONA (piani gaussiani, al variare di d e n), e da dove parte nei
    casi patologici. 
    """
    rng = np.random.default_rng(seed)
    print(f"\n{'='*104}\n  DOVE SEPARA LA SOGLIA?  "
          f"rapporto TwoNN/Levina-Bickel su casi di tipo noto\n{'='*104}")
    print(f"  {'':>18}" + "".join(f"{f'n={n}':>18}" for n in ns))
    print(f"  {'d vero':>18}" + "".join(f"{'rapp.':>9}{'TwoNN':>9}"
                                        for _ in ns))
    tipoA = []
    for d in ds:
        row = f"  {d:>18}"
        for n in ns:
            r = diagnose(case_linear(d, n, rng), n_boot=n_boot, seed=seed)
            tipoA.append(r['estimator_ratio'])
            row += f"{r['estimator_ratio']:9.2f}{r['d_hat']:9.2f}"
        print(row)

    print(f"\n  {'caso tipo B':>18}" + "".join(f"{'rapp.':>9}{'TwoNN':>9}"
                                               for _ in ns))
    tipoB = []
    for name, fn in (
            ('curva regolare', lambda n, r: case_curve_regular(n, r)),
            ('curva jitter',   lambda n, r: case_curve_jittered(n, r)),
            ('reticolo d=3',   lambda n, r: case_grid(
                3, max(2, int(round(n ** (1/3)))), r))):
        row = f"  {name:>18}"
        for n in ns:
            r = diagnose(fn(n, rng), n_boot=n_boot, seed=seed)
            tipoB.append(r['estimator_ratio'])
            row += f"{r['estimator_ratio']:9.2f}{r['d_hat']:9.2f}"
        print(row)

    A  = np.array([x for x in tipoA if np.isfinite(x)])
    Bv = np.array([x for x in tipoB if np.isfinite(x)])
    print(f"\n  tipo A (TwoNN funziona): rapporto in "
          f"[{A.min():.2f}, {A.max():.2f}]")
    print(f"  tipo B (TwoNN esplode):  rapporto in "
          f"[{Bv.min():.2f}, {Bv.max():.2f}]")
    if A.max() < Bv.min():
        lo, hi = A.max(), Bv.min()
        print(f"  -> DISGIUNTI: qualunque soglia in ({lo:.2f}, {hi:.2f}) separa.")
        print(f"     media geometrica: {np.sqrt(lo*hi):.2f}   "
              f"margine: {hi/lo:.1f}x")
    else:
        print(f"  -> SOVRAPPOSTI fra {Bv.min():.2f} e {A.max():.2f}: nessuna")
        print("     soglia separa perfettamente, il limite va dichiarato.")
    return dict(tipoA=A, tipoB=Bv)


def scan_jitter(jitters=np.logspace(-7, -1, 13), n=1000, n_boot=200, seed=0,
                d_expected=1.0):
    """
    Quanta irregolarita' serve perche' TwoNN torni affidabile?

    """
    rng = np.random.default_rng(seed)
    print(f"\n{'='*104}\n  QUANTA IRREGOLARITA' SERVE?  "
          f"curva 1D in R^{AMBIENT}, n={n}, d atteso = {d_expected:g}"
          f"\n{'='*104}")
    print(f"  {'jitter':>12}{'TwoNN':>10}{'Lev-Bick':>10}{'rapporto':>10}"
          f"{'KS/crit':>9}{'mu~1':>7}{'tipo':>10}{'struttura':>12}")
    rows = []
    for j in jitters:
        X = case_curve_jittered(n, rng, jitter=float(j))
        r = diagnose(X, n_boot=n_boot, seed=seed)
        lb = r['levina_bickel']
        alive = bool(np.isfinite(lb) and
                     abs(lb - d_expected) < 0.5 * d_expected + 1.0)
        rows.append(dict(jitter=float(j), ratio=r['estimator_ratio'],
                         lb=lb, alive=alive))
        print(f"  {j:12.2e}{r['d_hat']:10.2f}{lb:10.2f}"
              f"{r['estimator_ratio']:10.1f}{r['ks_ratio']:9.2f}"
              f"{100*r['frac_mu_near_1']:6.0f}%{r['violation']:>10}"
              f"{'viva' if alive else 'PERSA':>12}")

    alive = [x for x in rows if x['alive'] and np.isfinite(x['ratio'])]
    print(f"\n  configurazioni con la struttura ancora leggibile: "
          f"{len(alive)}/{len(rows)}")
    if alive:
        rmin = min(x['ratio'] for x in alive)
        jmin = [x['jitter'] for x in alive if x['ratio'] == rmin][0]
        print(f"  rapporto minimo a struttura viva: {rmin:.1f} "
              f"(jitter {jmin:.2e})")
        if rmin >= EST_THRESH:
            print(f"  -> il rapporto NON scende mai sotto {EST_THRESH:g} "
                  f"finche' la struttura sopravvive:")
            print("     TwoNN non e' recuperabile aggiungendo irregolarita'.")
        else:
            print(f"  -> esiste una finestra con struttura viva e rapporto "
                  f"sotto soglia: attraversamento genuino.")
    lost = [x for x in rows if not x['alive']]
    if lost:
        print(f"  attenzione: {len(lost)} configurazioni con struttura persa "
              f"(Levina-Bickel fuori da {d_expected:g} +/- "
              f"{0.5*d_expected+1.0:g}).")
        print("     lì il rapporto scende perche' salgono ENTRAMBI gli")
        print("     stimatori, non perche' TwoNN si corregga: non contano.")
    return rows



def scan_k(ks=(3, 5, 10, 20, 40), n=1000, n_boot=200, seed=0):
    """
    Test della spiegazione meccanicistica.

    """
    rng = np.random.default_rng(seed)
    cases = [
        ('piano d=10',            case_linear(10, n, rng),     'A'),
        ('piano d=30',            case_linear(30, n, rng),     'A'),
        ('curva, jitter 1e-3',    case_curve_jittered(n, rng), 'B'),
        ('curva, passo costante', case_curve_regular(n, rng),  'B'),
        ('reticolo d=3',          case_grid(3, 10, rng),       'B'),
    ]
    print(f"\n{'='*92}\n  IL RAPPORTO CRESCE CON k?  (n={n})\n{'='*92}")
    print(f"  {'caso':>22}{'tipo':>6}" +
          "".join(f"{f'k={k}':>13}" for k in ks) + f"{'k=40/k=3':>11}")
    out = {}
    for name, X, tp in cases:
        vals = []
        for k in ks:
            r = diagnose(X, n_boot=n_boot, seed=seed, k_mle=k)
            vals.append(r['estimator_ratio'])
        out[name] = vals
        growth = (vals[-1]/vals[0] if np.isfinite(vals[0]) and vals[0] > 0
                  else np.nan)
        print(f"  {name:>22}{tp:>6}" +
              "".join(f"{v:13.1f}" for v in vals) + f"{growth:11.1f}")
    print("\n  Previsione: crescita marcata nei casi B, piatta nei casi A.")
    print("  Se i B non crescono, il rapporto funziona per un altro motivo")
    return out



# ═══════════════════════════════════════════════════════════════════════
#  Applicazione ai dati del modello
# ═══════════════════════════════════════════════════════════════════════

def diagnose_model_clouds(W=1000.0, B=5e-4, Rr=4, G=0.05, n_points=500,
                          snapshots=(1000, 5000), n_boot=300, seed=0):

    import Rfc_and_ID as R
    print(f"\n{'='*104}\n  NUVOLE DEL MODELLO  W={W} B={B} R={Rr} G={G}, "
          f"{n_points} punti\n{'='*104}")
    cloud = R.sample_cloud(W, B, Rr, G, n_points, snapshots, seed)
    print(f"  {'t':>7}{'var':>7}{'TwoNN':>9}{'Lev-Bick':>10}{'rapp.':>8}"
          f"{'KS/crit':>9}{'mu~1':>7}{'verdetto':>52}")
    out = {}
    for t in snapshots:
        for var in ('field', 'vmem'):
            X = cloud[t][var]
            if np.ptp(X) == 0:
                print(f"  {t:7d}{var:>7}   nuvola collassata: dimensione 0")
                continue
            r = diagnose(X, n_boot=n_boot, seed=seed)
            out[(t, var)] = r
            er = r['estimator_ratio']
            print(f"  {t:7d}{var:>7}{r['d_hat']:9.2f}{r['levina_bickel']:10.2f}"
                  f"{er if np.isfinite(er) else float('nan'):8.1f}"
                  f"{r['ks_ratio']:9.2f}{100*r['frac_mu_near_1']:6.0f}%"
                  f"{r['note']:>52}")
    return out


# ═══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':

    MODE = 'model'   # 'calibrate' | 'threshold' | 'jitter' | 'k' | 'model'

    if MODE == 'calibrate':
        calibrate_on_synthetic(n=1000, n_boot=300)
    elif MODE == 'threshold':
        scan_threshold(n_boot=300)
    elif MODE == 'jitter':
        scan_jitter(n_boot=300)
    elif MODE == 'k':
        scan_k(n_boot=300)
    elif MODE == 'model':
        diagnose_model_clouds(W=1000.0, B=5e-4, n_points=300)
        diagnose_model_clouds(W=1000.0, B=5e-4, n_points=500)
        diagnose_model_clouds(W=1000.0, B=5e-4, n_points=1000)
    else:
        raise ValueError(f"MODE non riconosciuto: '{MODE}'")




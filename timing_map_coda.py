"""
timing_map_coda.py
=============

Mappa dei tempi di convergenza con diagnostica del regime lineare
post-saturazione.

Idea:
    Gp satura
        ->
    il feedback Gp <- E può diventare localmente nullo
        ->
    dinamica residua di Vmem
        ->
    ingresso nel regime lineare locale
        ->
    d_t ~ C * rho^t


"""

import os
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import Rfc_and_ID as R
from dynamics_t_sparse_gh import BioelectricLattice, G_r, V_INIT
from fixed_point import step_pure, stability


FIGDIR = "figures"

TOL = 1e-14
T_MAX = 100000

# Parametri per identificare il regime lineare.

RHO_TOL = 0.01          # |q_t - rho| / rho < RHO_TOL
LINEAR_WINDOW = 50      # passi consecutivi richiesti
MIN_LINEAR_STEPS = 20   # minimo numero di punti per un fit log-lineare

# Rilevatore indipendente da rho: si richiede che q_t sia COSTANTE sulla finestra
CV_TOL = 0.02           # coefficiente di variazione di q sulla finestra

def figpath(name):
    os.makedirs(FIGDIR, exist_ok=True)
    return os.path.join(FIGDIR, name)


def _sat_frac(Gp):
    """Frazione di celle troncate dal clamp. """
    return float(((Gp <= 0.0) | (Gp >= 2.0 * G_r)).double().mean())


# ======================================================================
# Traccia della convergenza
# ======================================================================

def trace_convergence(
    net,
    v0=None,
    Gp0=None,
    T_max=T_MAX,
    tol=TOL,
    sat_target=1.0,
):
    """
    Registra lo spostamento per passo:

        d_t = max(||v_{t+1} - v_t||_inf,  ||Gp_{t+1} - Gp_t||_inf / G_r)

    e, dopo la saturazione:

        q_t = d_{t+1} / d_t.
    """

    v = (torch.full((net.N,), V_INIT, dtype=torch.float64)
        if v0 is None else v0.clone())

    Gp = (torch.full((net.N,), 1.0 * G_r, dtype=torch.float64)
        if Gp0 is None else Gp0.clone())

    t_sat = None
    d_sat = None

    d_hist = []
    sat_hist = []

    with torch.no_grad():
        for t in range(1, T_max + 1):
            v_new, Gp_new = step_pure(v, Gp, net)
            d = float(
                torch.max((v_new - v).abs().max(),
                    (Gp_new - Gp).abs().max() / G_r)
                    )
            v, Gp = v_new, Gp_new
            sat = _sat_frac(Gp)
            d_hist.append(d)
            sat_hist.append(sat)
            if (t_sat is None and sat >= sat_target - 1e-12):
                t_sat = t
                d_sat = d
            if d < tol:
                return dict(
                    converged=True,
                    t_conv=t,
                    t_sat=t_sat,
                    d_sat=d_sat,
                    last_step=d,
                    v=v,
                    Gp=Gp,
                    frac_sat=sat,
                    d_hist=np.asarray(d_hist, dtype=float),
                    sat_hist=np.asarray(sat_hist, dtype=float),
                )

    return dict(
        converged=False,
        t_conv=None,
        t_sat=t_sat,
        d_sat=d_sat,
        last_step=d_hist[-1] if d_hist else np.nan,
        v=v,
        Gp=Gp,
        frac_sat=_sat_frac(Gp),
        d_hist=np.asarray(d_hist, dtype=float),
        sat_hist=np.asarray(sat_hist, dtype=float),
    )


# ======================================================================
# Rapporto q_t = d_{t+1}/d_t
# ======================================================================

def compute_q(d_hist, eps=1e-300):
    """
    Restituisce

        q_t = d_{t+1} / d_t

    indicizzato sul passo t.

    """
    d = np.asarray(d_hist, dtype=float)
    if len(d) < 2:
        return np.empty(0, dtype=float)
    den = np.maximum(d[:-1], eps)
    q = d[1:] / den
    q[~np.isfinite(q)] = np.nan
    return q


def find_linear_regime(
    d_hist,
    t_sat,
    rho,
    rho_tol=RHO_TOL,
    window=LINEAR_WINDOW,
    min_steps=MIN_LINEAR_STEPS,
):
    """
    Cerca il primo tratto in cui q_t = d_{t+1}/d_t rimane vicino a rho.

    Criterio:

        |q_t - rho| / rho < rho_tol

    per 'window' passi consecutivi. Restituisce None se il regime non viene identificato.
    """

    if t_sat is None:
        return None
    
    if not np.isfinite(rho) or rho <= 0.0 or rho >= 1.0:
        return None

    d = np.asarray(d_hist, dtype=float)
    if len(d) < 2:
        return None

    q = compute_q(d)
    start_idx = max(int(t_sat - 1), 0)
    close = np.zeros(len(q), dtype=bool)
    good = np.isfinite(q)

    close[good] = (np.abs(q[good] - rho) / rho < rho_tol)

    # Cerca una finestra completamente contenuta dopo t_sat.
    for i in range(start_idx, len(q) - window + 1):
        if not np.all(close[i:i + window]):
            continue

        t_lin = i + 1
        if t_lin < t_sat + min_steps:
            continue

        d_lin = d[t_lin - 1]
        if not np.isfinite(d_lin) or d_lin <= 0.0:
            continue

        return dict(
            t_lin=int(t_lin),
            d_lin=float(d_lin),
            q_window=float(np.mean(q[i:i + window])),
            q_std=float(np.std(q[i:i + window])),
            window=int(window),
        )

    return None



def find_linear_regime_rho_free(
    d_hist,
    t_sat,
    window=LINEAR_WINDOW,
    cv_tol=CV_TOL,
):
    """Identifica l'ingresso nel regime lineare SENZA usare rho. """
    if t_sat is None:
        return None

    d = np.asarray(d_hist, dtype=float)
    if len(d) < window + 2:
        return None

    q = compute_q(d)
    start_idx = max(int(t_sat) - 1, 0)

    for i in range(start_idx, len(q) - window + 1):
        w = q[i:i + window]
        if not np.all(np.isfinite(w)) or np.any(w <= 0.0):
            continue
        m = float(np.mean(w))
        if m <= 0.0:
            continue
        cv = float(np.std(w) / m)
        if cv >= cv_tol:
            continue

        t_lin = i + 1
        d_lin = d[t_lin - 1]
        if not np.isfinite(d_lin) or d_lin <= 0.0:
            continue

        return dict(
            t_lin=int(t_lin),
            d_lin=float(d_lin),
            q_window=m,
            q_std=float(np.std(w)),
            q_cv=cv,
            window=int(window),
        )

    return None


# ======================================================================
# Fit del regime lineare
# ======================================================================

def fit_linear_tail(
    d_hist,
    t_start,
    t_end=None,
    min_steps=MIN_LINEAR_STEPS,
):
    """
    Fit:

        log d_t = a + b t

    sul tratto finale indicato.

    Restituisce:

        rho_eff = exp(b)

    che costituisce una stima empirica del fattore di contrazione.
    """

    d = np.asarray(d_hist, dtype=float)

    if t_start is None:
        return None

    if t_end is None:
        t_end = len(d)

    t_start = max(int(t_start), 1)
    t_end = min(int(t_end), len(d))

    if t_end - t_start < min_steps:
        return None

    # d_hist[k] corrisponde al passo k+1.
    steps = np.arange(t_start, t_end + 1, dtype=float)
    vals = d[t_start - 1:t_end]

    good = (np.isfinite(vals) & (vals > 0.0))

    if good.sum() < min_steps:
        return None

    x = steps[good]
    y = np.log(vals[good])
    slope, intercept = np.polyfit(x, y, 1)
    pred = intercept + slope * x
    ss_res = float(np.sum((y - pred) ** 2))
    ybar = float(np.mean(y))
    ss_tot = float(np.sum((y - ybar) ** 2))

    r2 = (1.0 - ss_res / ss_tot 
          if ss_tot > 0.0 
          else np.nan)

    rho_eff = float(np.exp(slope))

    return dict(
        rho_eff=rho_eff,
        slope=float(slope),
        intercept=float(intercept),
        r2=float(r2),
        n_points=int(good.sum()),
        t_start=int(x.min()),
        t_end=int(x.max()),
    )


# ======================================================================
# Una cella della griglia
# ======================================================================

def analyze_cell_timing(
    W,
    B,
    Rr=4,
    G=0.05,
    n_starts=5,
    T_max=T_MAX,
    tol=TOL,
    seed=0,
    verbose=True,
):
    net = BioelectricLattice(rows=11, cols=11, G=G, R=Rr, W=W, B=B, )
    runs = []
    for i in range(n_starts):
        if i == 0:
            v0, Gp0 = None, None
        else:
            g = torch.Generator().manual_seed(seed * 1000 + i)
            v0 = torch.full((net.N,), V_INIT, dtype=torch.float64, )
            Gp0 = ( torch.rand( net.N, generator=g, dtype=torch.float64, ) * 2.0 * G_r )
        run = trace_convergence(net, v0=v0, Gp0=Gp0, T_max=T_max, tol=tol, )

        runs.append(run)

    conv = [r for r in runs if r["converged"]]
    # Per il rho basta calcolare la Jacobiana su un punto fisso convergente.
    if conv:
        st = stability( net, conv[0]["v"], conv[0]["Gp"], )
        rho = st["spectral_radius"]
        n_gp_frozen = st["n_gp_frozen"]

    else:
        rho = np.nan
        n_gp_frozen = -1

    run_analysis = []
    for run in runs:
        item = dict(
            converged=bool(run["converged"]),
            t_conv=run["t_conv"],
            t_sat=run["t_sat"],
            d_sat=run["d_sat"],
            t_lin=None,
            t_lin_legacy=None,      
            d_lin=None,
            q_lin=None,
            q_lin_std=None,
            q_lin_cv=None,          
            relax_sat=None,
            relax_lin=None,
            relax_lin_pred=None,
            rho_eff=None,
            rho_eff_r2=None,
        )

        if not run["converged"]:
            run_analysis.append(item)
            continue

        if run["t_sat"] is not None:
            item["relax_sat"] = (run["t_conv"] - run["t_sat"])

        lin_free = find_linear_regime_rho_free(run["d_hist"], run["t_sat"])
        lin_leg  = find_linear_regime(run["d_hist"], run["t_sat"], rho)

        item["t_lin_legacy"] = lin_leg["t_lin"] if lin_leg else None
        item["q_lin_cv"] = lin_free["q_cv"] if lin_free else None

        lin = lin_free
        if lin is not None:

            item["t_lin"] = lin["t_lin"]
            item["d_lin"] = lin["d_lin"]
            item["q_lin"] = lin["q_window"]
            item["q_lin_std"] = lin["q_std"]

            item["relax_lin"] = run["t_conv"] - lin["t_lin"]

            # Previsione dalla linearizzazione.
            if (np.isfinite(rho) and 0.0 < rho < 1.0
                    and lin["d_lin"] > tol):
                item["relax_lin_pred"] = max(
                    0.0,
                    float(np.ceil(np.log(tol / lin["d_lin"]) / np.log(rho)))
                )

            # ----------------------------
            # Fit empirico del tratto lineare
            # ----------------------------

            span = run["t_conv"] - lin["t_lin"]
            head_margin = min(30, max(0, (span - MIN_LINEAR_STEPS) // 3))
            tail_margin = min(5,  max(0, (span - MIN_LINEAR_STEPS) // 10))

            fit = fit_linear_tail(run["d_hist"],
                t_start=lin["t_lin"] + head_margin,
                t_end=run["t_conv"] - tail_margin, )
            if fit is not None:
                item["rho_eff"] = fit["rho_eff"]
                item["rho_eff_r2"] = fit["r2"]

        run_analysis.append(item)

    # ------------------------------------------------------------------
    # Aggregazione di cella
    # ------------------------------------------------------------------

    def finite_array(key):
        vals = [x[key] for x in run_analysis
                if x.get(key) is not None and np.isfinite(x[key])]
        return np.asarray(vals, dtype=float)

    tc = finite_array("t_conv")
    ts = finite_array("t_sat")
    tl = finite_array("t_lin")
    tl_leg = finite_array("t_lin_legacy")

    relax_sat = finite_array("relax_sat")
    relax_lin = finite_array("relax_lin")
    relax_pred = finite_array("relax_lin_pred")
    rho_eff = finite_array("rho_eff")
    r2_eff = finite_array("rho_eff_r2")

    out = dict(
        W=W,
        B=B,
        R=Rr,
        G=G,
        n_starts=n_starts,
        T_max=T_max,
        tol=tol,
        n_converged=len(conv),
        t_conv_min=(float(tc.min()) if len(tc) else np.nan),
        t_conv_med=(float(np.median(tc)) if len(tc) else np.nan),
        t_conv_max=(float(tc.max()) if len(tc) else np.nan),
        t_sat_med=(float(np.median(ts)) if len(ts) else np.nan),
        t_lin_med=(float(np.median(tl)) if len(tl) else np.nan),
        t_lin_legacy_med=(float(np.median(tl_leg)) if len(tl_leg) else np.nan),
        n_linear_legacy=len(tl_leg),
        relax_sat_med=(float(np.median(relax_sat))
                       if len(relax_sat) else np.nan),
        relax_lin_med=(float(np.median(relax_lin))
                       if len(relax_lin) else np.nan),
        relax_lin_pred_med=(float(np.median(relax_pred))
                            if len(relax_pred) else np.nan),
        rho=float(rho),
        n_gp_frozen=n_gp_frozen,
        rho_eff_med=(float(np.median(rho_eff))
                     if len(rho_eff) else np.nan),
        rho_eff_min=(float(rho_eff.min())
                     if len(rho_eff) else np.nan),
        rho_eff_max=(float(rho_eff.max())
                     if len(rho_eff) else np.nan),
        rho_eff_r2_med=(float(np.median(r2_eff))
                        if len(r2_eff) else np.nan),
        n_linear=len(tl),
        runs=run_analysis,
    )

    if verbose:
        tc_txt = (f"{out['t_conv_med']:.0f} "
                  f"[{out['t_conv_min']:.0f}-{out['t_conv_max']:.0f}]"
                  if len(tc) else "mai")
        ts_txt = f"{out['t_sat_med']:.0f}" if len(ts) else "n/a"
        tl_txt = f"{out['t_lin_med']:.0f}" if len(tl) else "n/a"
        tlg_txt = f"{out['t_lin_legacy_med']:.0f}" if len(tl_leg) else "n/a"
        rl_txt = (f"{out['relax_lin_med']:.1f}"
                  if np.isfinite(out['relax_lin_med']) else "n/a")
        re_txt = (f"{out['rho_eff_med']:.6f}"
                  if np.isfinite(out['rho_eff_med']) else "n/a")

        print(f"  W={W:>6.0f} B={B:.4f}"
              f"  conv={len(conv):>2}/{n_starts}"
              f"  t_conv={tc_txt:>22}"
              f"  t_sat={ts_txt:>8}"
              f"  t_lin={tl_txt:>8}"
              f"  (leg {tlg_txt:>8})"
              f"  relax_lin={rl_txt:>7}"
              f"  rho={rho:.6f}"
              f"  rho_eff={re_txt:>8}")

    return out


# ======================================================================
# Griglia
# ======================================================================

def _init_worker():
    torch.set_num_threads(1)


def _work(a):
    torch.set_num_threads(1)
    (W, B, Rr, G, n_starts, T_max, tol, seed, ) = a

    return analyze_cell_timing(W, B, Rr, G, n_starts, 
                               T_max, tol, seed, verbose=False, )


def grid_timing(
    Ws=R.PAPER_W,
    Bs=R.PAPER_B,
    Rr=4,
    G=0.05,
    n_starts=5,
    T_max=T_MAX,
    tol=TOL,
    seed=0,
    n_workers=None,
    save_as="timing_map.pt",
):
    jobs = [
        ( W, B, Rr, G, n_starts, T_max, tol, seed, )
        for W in Ws
        for B in Bs
    ]

    print( f"Griglia dei tempi: {len(jobs)} celle "
          f"x {n_starts} partenze (T_max={T_max})")

    res = []

    if n_workers and n_workers > 1:

        from concurrent.futures import (ProcessPoolExecutor, as_completed, )
        with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker, ) as ex:
            futs = [ex.submit(_work, j) for j in jobs]

            for c, f in enumerate(as_completed(futs), 1):
                res.append(f.result())
                print(f"  [{c}/{len(jobs)}] fatta" )
                torch.save( res, R.out_path(save_as), )

    else:
        for c, j in enumerate(jobs, 1):
            res.append(_work(j))
            print(f"  [{c}/{len(jobs)}] fatta")
            torch.save(res, R.out_path(save_as), )

    res.sort(key=lambda r: ( r["W"], r["B"], ) )
    print_timing(res)
    check_linear_regime(res)
    torch.save(res, R.out_path(save_as), )
    return res


# ======================================================================
# Tabella
# ======================================================================

def print_timing(res):
    """Tabella dei tempi e del regime lineare. """
    print(
        f"\n{'='*156}\n"
        f"  MAPPA DEI TEMPI E DEL REGIME LINEARE"
        f"  (R={res[0]['R']}, G={res[0]['G']}, "
        f"{res[0]['n_starts']} partenze)\n"
        f"{'='*156}"
    )

    print(
        f"  {'W':>7}{'B':>9}"
        f"{'conv.':>8}"
        f"{'t_conv':>10}"
        f"{'t_sat':>9}"
        f"{'t_lin':>9}"
        f"{'t_lin_leg':>11}"
        f"{'n_leg':>8}"
        f"{'relax':>9}"
        f"{'relax*':>10}"
        f"{'rho':>11}"
        f"{'rho_eff':>11}"
        f"{'R2':>8}"
    )

    for r in res:
        conv = f"{r['n_converged']}/{r['n_starts']}"
        tc = (f"{r['t_conv_med']:.0f}"
              if np.isfinite(r["t_conv_med"]) else "mai")
        ts = (f"{r['t_sat_med']:.0f}"
              if np.isfinite(r["t_sat_med"]) else "n/a")
        tl = (f"{r['t_lin_med']:.0f}"
              if np.isfinite(r["t_lin_med"]) else "n/a")
        tlg = (f"{r['t_lin_legacy_med']:.0f}"
               if np.isfinite(r.get("t_lin_legacy_med", np.nan)) else "n/a")
        nlg = (f"{r.get('n_linear_legacy', 0)}/{r['n_converged']}"
               if r['n_converged'] else "n/a")
        rl = (f"{r['relax_lin_med']:.0f}"
              if np.isfinite(r["relax_lin_med"]) else "n/a")
        rlp = (f"{r['relax_lin_pred_med']:.0f}"
               if np.isfinite(r["relax_lin_pred_med"]) else "n/a")
        rho = (f"{r['rho']:.6f}" if np.isfinite(r["rho"]) else "nan")
        rho_eff = (f"{r['rho_eff_med']:.6f}"
                   if np.isfinite(r["rho_eff_med"]) else "n/a")
        r2 = (f"{r['rho_eff_r2_med']:.3f}"
              if np.isfinite(r["rho_eff_r2_med"]) else "n/a")

        print(
            f"  {r['W']:7.0f}"
            f"{r['B']:9.4f}"
            f"{conv:>8}"
            f"{tc:>10}"
            f"{ts:>9}"
            f"{tl:>9}"
            f"{tlg:>11}"
            f"{nlg:>8}"
            f"{rl:>9}"
            f"{rlp:>10}"
            f"{rho:>11}"
            f"{rho_eff:>11}"
            f"{r2:>8}"
        )



# ======================================================================
# Controlli globali
# ======================================================================

def check_linear_regime(res):
    """Il tasso di contrazione osservato coincide con l'autovalore?"""
    ok = [r for r in res
          if (r["n_linear"] > 0
              and np.isfinite(r["rho"])
              and np.isfinite(r["rho_eff_med"]))
          ]

    print(f"\n{'='*86}\n"
          f"  VERIFICA DEL REGIME LINEARE POST-SATURAZIONE\n"
          f"{'='*86}")

    if not ok:
        print("\n  Nessuna cella ha fornito un tratto lineare identificabile.")
        return None

    rho = np.array([r["rho"] for r in ok], dtype=float)
    rho_eff = np.array([r["rho_eff_med"] for r in ok], dtype=float)
    err_rel = np.abs(rho_eff - rho) / rho
    signed = (rho_eff - rho) / rho

    relax_obs = np.array([r["relax_lin_med"] for r in ok], dtype=float)
    relax_pred = np.array([r["relax_lin_pred_med"] for r in ok], dtype=float)
    valid_relax = np.isfinite(relax_obs) & np.isfinite(relax_pred)

    print(f"  celle con regime lineare identificato: {len(ok)}")
    print(f"  rho Jacobiana: mediana {np.median(rho):.6f}")
    print(f"  rho_eff da fit log-lineare: "
          f"mediana {np.median(rho_eff):.6f} "
          f"[{rho_eff.min():.6f}, {rho_eff.max():.6f}]")
    print(f"  errore relativo |rho_eff-rho|/rho: "
          f"mediana {100*np.median(err_rel):.2f}% "
          f"max {100*err_rel.max():.2f}%")

    n_below = int((signed < 0).sum())
    print(f"  rho_eff < rho in {n_below}/{len(ok)} celle "
          f"(scarto con segno: mediana {100*np.median(signed):+.2f}%)")

    if valid_relax.any():
        err_relax = np.abs(relax_obs[valid_relax] - relax_pred[valid_relax])
        rel_relax = err_relax / relax_obs[valid_relax]

        print(f"\n  rilassamento osservato dal regime lineare:"
              f" mediana {np.median(relax_obs[valid_relax]):.1f}"
              f"  intervallo [{relax_obs[valid_relax].min():.0f},"
              f" {relax_obs[valid_relax].max():.0f}]")
        print(f"  rilassamento predetto:"
              f" mediana {np.median(relax_pred[valid_relax]):.1f}"
              f"  intervallo [{relax_pred[valid_relax].min():.0f},"
              f" {relax_pred[valid_relax].max():.0f}]")
        print(f"  scarto assoluto:"
              f" mediana {np.median(err_relax):.1f},"
              f" max {np.max(err_relax):.1f}")
        print(f"  scarto relativo:"
              f" mediana {100*np.median(rel_relax):.1f}%,"
              f" max {100*np.max(rel_relax):.1f}%")


    n_free = sum(r.get("n_linear", 0) for r in res)
    n_leg = sum(r.get("n_linear_legacy", 0) for r in res)
    n_conv = sum(r.get("n_converged", 0) for r in res)
    print(f"\n  partenze in cui il rilevatore trova una finestra:")
    print(f"    rho-free (q costante):  {n_free}/{n_conv}")
    print(f"    legacy   (q ~ rho):     {n_leg}/{n_conv}")
    if n_conv and n_leg < n_free:
        print(f"  Il legacy riesce {n_free - n_leg} volte in meno")

    return dict(n=len(ok),
                rho_eff_median=float(np.median(rho_eff)),
                err_rel_median=float(np.median(err_rel)),
                err_rel_max=float(err_rel.max()),
                n_below=n_below,
                relax_err_rel_median=(float(np.median(rel_relax))
                                      if valid_relax.any() else float('nan')))

# ======================================================================
# Grafico: q_t e convergenza logaritmica
# ======================================================================

def plot_run_diagnostics(run, rho, fname="09_regime_lineare.png"):
    """
    Grafico diagnostico di una singola traiettoria.
    """
    d = np.asarray(run["d_hist"], dtype=float)
    if len(d) < 2:
        print("  run troppo corto per il grafico.")
        return

    q = compute_q(d)
    t_d = np.arange(1, len(d) + 1)
    t_q = np.arange(1, len(q) + 1)

    lin = find_linear_regime_rho_free(d, run["t_sat"])
    lin_leg = find_linear_regime(d, run["t_sat"], rho)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)

    axes[0].semilogy(t_d, d, lw=1.2, color="C0", label=r"$d_t$")
    if run["t_sat"] is not None:
        axes[0].axvline(run["t_sat"], ls="--", lw=1.2, color="0.4",
                        label=r"$t_{\rm sat}$")
    if lin is not None:
        axes[0].axvline(lin["t_lin"], ls="-.", lw=1.4, color="C2",
                        label=r"$t_{\rm lin}$ ($q$ costante)")
    if lin_leg is not None:
        axes[0].axvline(lin_leg["t_lin"], ls=":", lw=1.4, color="C3",
                        label=r"$t_{\rm lin}$ (vicino a $\rho$)")
    axes[0].axhline(TOL, ls=":", lw=1.0, color="0.6", label="tolleranza")
    axes[0].set_xlabel("passo")
    axes[0].set_ylabel(r"$d_t$")
    axes[0].set_title("Convergenza: transitorio e regime asintotico")
    axes[0].grid(alpha=0.3, which="both")
    axes[0].legend(fontsize=8)

    axes[1].plot(t_q, q, lw=1.0, color="C0", label=r"$q_t=d_{t+1}/d_t$")
    if np.isfinite(rho):
        axes[1].axhline(rho, ls="--", lw=1.2, color="0.2",
                        label=fr"$\rho={rho:.6f}$")
    if run["t_sat"] is not None:
        axes[1].axvline(run["t_sat"], ls="--", lw=1.2, color="0.4",
                        label=r"$t_{\rm sat}$")
    if lin is not None:
        axes[1].axvline(lin["t_lin"], ls="-.", lw=1.4, color="C2",
                        label=r"$t_{\rm lin}$ ($q$ costante)")
    if lin_leg is not None:
        axes[1].axvline(lin_leg["t_lin"], ls=":", lw=1.4, color="C3",
                        label=r"$t_{\rm lin}$ (vicino a $\rho$)")
    axes[1].set_xlabel("passo")
    axes[1].set_ylabel(r"$d_{t+1}/d_t$")
    axes[1].set_title("Ingresso nel regime lineare: due criteri a confronto")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)

    fig.savefig(figpath(fname), dpi=150)
    plt.close(fig)
    print(f"  {FIGDIR}/{fname}")

    

# ======================================================================
# Grafico globale rho_eff vs rho
# ======================================================================

def plot_rho_comparison(res, fname="10_rho_eff_vs_rho.png", ):

    ok = [r for r in res if (np.isfinite(r["rho"])
                             and np.isfinite(r["rho_eff_med"]))
                             ]

    if not ok:
        print(f"  (salto {fname}: nessun regime lineare identificato)")
        return

    rho = np.array([r["rho"] for r in ok], dtype=float, )

    rho_eff = np.array( [r["rho_eff_med"] for r in ok], dtype=float, )

    fig, ax = plt.subplots(figsize=(6.5, 6), constrained_layout=True, )

    ax.scatter(rho, rho_eff, s=80, edgecolor="k", linewidth=0.6, )

    lo = min(rho.min(), rho_eff.min(), )

    hi = max(rho.max(), rho_eff.max(), )

    ax.plot([lo, hi], [lo, hi], "--", lw=1.2, label=r"$\rho_{\rm eff}=\rho$", )

    ax.set_xlabel(r"$\rho$ dalla Jacobiana")
    ax.set_ylabel(r"$\rho_{\rm eff}$ dal fit")
    ax.set_title("Fattore di contrazione locale: teoria vs traiettoria")

    ax.grid(alpha=0.3, )
    ax.legend()

    fig.savefig(figpath(fname), dpi=150, )

    plt.close(fig)

    print(f"  {FIGDIR}/{fname}" )


# ======================================================================
# Esecuzione
# ======================================================================

if __name__ == "__main__":

    N_STARTS = 30
    WORKERS = max(1, (os.cpu_count() or 2) // 2, )

    res = grid_timing(n_starts=N_STARTS, n_workers=WORKERS,)

    print(f"\n  grafici in {os.path.abspath(FIGDIR)}/" )

    plot_timing_map = None

    plot_rho_comparison(res)

    diagnostic_done = False

    for r in res:
        if (r["n_linear"] > 0 and r["n_converged"] > 0):

            net = BioelectricLattice(rows=11, cols=11, G=r["G"], 
                                     R=r["R"], W=r["W"], B=r["B"], )
            run = trace_convergence(net, T_max=r["T_max"], tol=r["tol"], )

            b_tag = f"{r['B']:g}".replace('.', 'p')
            plot_run_diagnostics(run, r["rho"],
                                 fname=(f"09_regime_lineare_"
                                        f"W{int(r['W'])}_B{b_tag}.png"))
            diagnostic_done = True
            break

    if not diagnostic_done:
        print("  Nessuna traiettoria adatta al grafico diagnostico.")


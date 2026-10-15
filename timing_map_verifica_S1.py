"""
Verifica della condizione (S1): separazione di scale nello spettro della dinamica linearizzata
"""

import os
import numpy as np
import scipy.linalg as sl
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import Rfc_and_ID as R
from dynamics_t_sparse_gh import BioelectricLattice, G_r, V_INIT
from fixed_point import step_pure


# ═══════════════════════════════════════════════════════════════════════
#  CRITERI 
# ═══════════════════════════════════════════════════════════════════════

TAU_RATIO_MIN = 10.0      # (S1a) separazione fra i tempi ai lati del gap
SLOW_FRAC_MAX = 0.10      # (S1b) frazione massima di modi lenti

ZERO_TOL = 1e-12          # sotto questo un modulo e' uno zero esatto
SPECTRUM_DRIFT_TOL = 0.01
TOP_K = 30
LYAP_STEPS = 500          # passi della finestra per gli esponenti di Lyapunov
LYAP_BURN = 100           # passi iniziali in cui Q si allinea, non accumulati
STATIONARY_TOL = 1e-12    # spostamento all'ultimo passo sotto cui lo stato e' un punto fisso

FIGDIR = 'figures'


def figpath(name):
    os.makedirs(FIGDIR, exist_ok=True)
    return os.path.join(FIGDIR, name)


# ═══════════════════════════════════════════════════════════════════════
#  Stato e Jacobiana
# ═══════════════════════════════════════════════════════════════════════

def _sat_frac(Gp):
    return float(((Gp <= 0.0) | (Gp >= 2.0 * G_r)).double().mean())


def state_after_steps(W, B, Rr=4, G=0.05, T=5000, seed=None):
    """Stato (v, Gp) dopo T passi, piu' lo spostamento all'ultimo passo."""
    net = BioelectricLattice(rows=11, cols=11, G=G, R=Rr, W=W, B=B)
    net.reset()

    v = torch.full((net.N,), V_INIT, dtype=torch.float64)
    if seed is None:
        Gp = torch.full((net.N,), G_r, dtype=torch.float64)
    else:
        g = torch.Generator().manual_seed(int(seed))
        Gp = torch.rand(net.N, generator=g, dtype=torch.float64) * 2.0 * G_r

    last_step = float('nan')
    sat_hist = []
    with torch.no_grad():
        for t in range(T):
            v_new, Gp_new = step_pure(v, Gp, net)
            last_step = float(torch.max((v_new - v).abs().max(),
                                        (Gp_new - Gp).abs().max() / G_r))
            v, Gp = v_new, Gp_new
            if (t + 1) % max(1, T // 20) == 0:
                sat_hist.append((t + 1, _sat_frac(Gp)))

    return net, v.detach().clone(), Gp.detach().clone(), last_step, sat_hist


def full_step_map(x, net):
    N = net.N
    return torch.cat(step_pure(x[:N], x[N:], net))


def full_dynamics_jacobian(net, v, Gp, vectorize=True):
    x0 = torch.cat((v, Gp)).detach().clone().requires_grad_(True)
    J = torch.autograd.functional.jacobian(
        lambda x: full_step_map(x, net), x0,
        create_graph=False, vectorize=vectorize)
    return J.detach().cpu().numpy()


def _scaled_step(x, net):
    """Mappa a un passo nelle coordinate (v, g), con g = Gp/G_r."""
    N = net.N
    v_new, Gp_new = step_pure(x[:N], x[N:] * G_r, net)
    return torch.cat((v_new, Gp_new / G_r))


def lyapunov_spectrum(net, v, Gp, M=LYAP_STEPS, seed=0, burn=LYAP_BURN):
    """Esponenti di Lyapunov a tempo finito (per passo) su M passi da (v, Gp). 
    Metodo QR di Benettin et al. (1980) """
    N = net.N
    x = torch.cat((v, Gp / G_r)).detach().clone()
    xs = []
    always_sat = torch.ones(N, dtype=torch.bool)
    ever_sat = torch.zeros(N, dtype=torch.bool)
    with torch.no_grad():
        for _ in range(burn + M):
            xs.append(x.clone())
            x_new = _scaled_step(x, net)
            g, g_new = x[N:], x_new[N:]
            stuck = (((g <= 0.0) & (g_new <= 0.0)) |
                     ((g >= 2.0) & (g_new >= 2.0)))
            always_sat &= stuck
            ever_sat |= stuck
            x = x_new
    n_dead = int(always_sat.sum())
    K = 2 * N - n_dead
    rng = np.random.default_rng(seed)
    Q = np.linalg.qr(rng.standard_normal((2 * N, K)))[0]
    acc = np.zeros(K)
    for t, xt in enumerate(xs):
        J = torch.autograd.functional.jacobian(
            lambda y: _scaled_step(y, net), xt, vectorize=True).numpy()
        Q, Rm = np.linalg.qr(J @ Q)
        if t >= burn:
            acc += np.log(np.maximum(np.abs(np.diag(Rm)), 1e-300))
    lam = np.sort(acc / M)[::-1]
    return dict(lyap=lam, n_dead=n_dead, K=K, M=M, burn=burn, n_ever_sat=int(ever_sat.sum()))


# ═══════════════════════════════════════════════════════════════════════
#  Spettro
# ═══════════════════════════════════════════════════════════════════════

def _tau_steps(mods):
    """tau = -1/ln|z| in passi (eq:tau_mu). nan se |z|>=1, 0 se |z|=0."""
    tau = np.full(len(mods), np.nan, dtype=float)
    ok = (mods > 0.0) & (mods < 1.0)
    tau[ok] = -1.0 / np.log(mods[ok])
    tau[mods == 0.0] = 0.0
    return tau


def _tau_ratio(mb, ma):
    """Separazione di scale attraverso il gap, con i casi degeneri espliciti."""
    if mb >= 1.0 and ma >= 1.0:
        return np.nan                    # nessuno dei due rilassa
    if mb >= 1.0 and ma < 1.0:
        return np.inf                    # separazione infinita
    if ma <= 0.0:
        return np.inf                    # il veloce e' esattamente nullo
    return float(np.log(ma) / np.log(mb))


def spectral_summary(eigvals, eigvecs=None, N=None, left_eigvecs=None):
    """Moduli ordinati, gap su tutto lo spettro non nullo, partecipazione. """
    eigvals = np.asarray(eigvals, dtype=np.complex128)
    order = np.argsort(np.abs(eigvals))[::-1]
    eigvals = eigvals[order]
    mods = np.abs(eigvals)
    tau = _tau_steps(mods)

    out = dict(
        eigvals=eigvals, abs_eigs=mods, tau=tau,
        rho=float(mods[0]) if len(mods) else 0.0,
        n_total=int(len(mods)),
        n_noncontracting=int(np.sum(mods >= 1.0)),
        n_above_099=int(np.sum(mods > 0.99)),
        n_above_090=int(np.sum(mods > 0.90)),
        n_zero=int(np.sum(mods <= ZERO_TOL)),
        deciles=(np.percentile(mods, np.arange(0, 101, 10))
                 if len(mods) else np.array([])),
    )

    # Fattori di partecipazione: indipendenti dalle unita' delle coordinate.
    if eigvecs is not None and left_eigvecs is not None and N is not None:
        r = eigvecs[:, order]
        l = np.conj(left_eigvecs[:, order])     # scipy: vl^H J = w vl^H
        p = np.abs(l * r)
        tot = np.maximum(p.sum(axis=0), np.finfo(float).tiny)
        out['part_v'] = p[:N, :].sum(axis=0) / tot

    # Gap: si escludono gli zeri esatti
    nz = int(np.sum(mods > ZERO_TOL))
    if nz < 2:
        out.update(gap_after=None, gap_ratio_max=np.nan, tau_ratio=np.nan,
                   mod_before_gap=np.nan, mod_after_gap=np.nan,
                   gap_salience=np.nan)
        return out

    m = mods[:nz]
    ratio = m[:-1] / m[1:]
    log_gap = np.log(ratio)
    k = int(np.argmax(log_gap))

    med = float(np.median(log_gap))
    out.update(
        gap_after=k + 1,
        gap_ratio=ratio, log_gap=log_gap,
        gap_ratio_max=float(ratio[k]),
        gap_salience=(float(log_gap[k] / med) if med > 0 else np.inf),
        mod_before_gap=float(m[k]), mod_after_gap=float(m[k + 1]),
        tau_before_gap=float(tau[k]), tau_after_gap=float(tau[k + 1]),
        tau_ratio=_tau_ratio(float(m[k]), float(m[k + 1])),
        n_nonzero=nz,
    )
    return out


def gp_block_diagnosis(J, N):
    """Il blocco dGp_new/dGp e' diagonale."""
    JGG = J[N:, N:]
    diag = np.diag(JGG)
    off = float(np.abs(JGG - np.diag(diag)).max()) if N > 1 else 0.0
    nz = diag[np.abs(diag) > 1e-14]
    val = float(np.median(nz)) if len(nz) else np.nan
    return dict(gp_diag_value=val,
                relax_inferred=(1.0 - val) if np.isfinite(val) else np.nan,
                n_gp_active=int(len(nz)), n_gp_frozen=int(N - len(nz)),
                gp_block_offdiag_max=off)


def verdict_S1(st, gp, ):
    k = st['gap_after']
    if k is None:
        return dict(passes=False, reason='spettro degenere',
                    n_slow=np.nan, n_slow_net=np.nan,
                    n_gp_cluster=np.nan, crit_tau=False, crit_few=False)

    mods = st['abs_eigs']
    val = gp['gp_diag_value']
    n_gp_cluster = int(np.sum(mods[:k] > 0.99))
    n_slow_net = k - n_gp_cluster

    tr = st['tau_ratio']
    a = bool(np.isfinite(tr) and tr >= TAU_RATIO_MIN) or (tr == np.inf)
    b = bool(1 <= n_slow_net <= SLOW_FRAC_MAX * st['n_total'])

    if a and b:
        reason = 'entrambi i criteri soddisfatti'
    else:
        parti = []
        if not a:
            parti.append('separazione insufficiente o non definita '
                         f'(tau_ratio = {tr})')
        if not b:
            parti.append(f'gruppo lento non ristretto '
                         f'({n_slow_net}/{st["n_total"]} netti, '
                         f'soglia {SLOW_FRAC_MAX:.0%})')
        reason = ' ; '.join(parti)

    return dict(passes=bool(a and b), reason=reason, n_slow=k,
                n_slow_net=n_slow_net, n_gp_cluster=n_gp_cluster,
                crit_tau=a, crit_few=b)


# ═══════════════════════════════════════════════════════════════════════
#  Una configurazione
# ═══════════════════════════════════════════════════════════════════════

def separation_excluding_cluster(mods, thr=0.99, zero=ZERO_TOL, min_mod=np.exp(-2.0)):
    """Massima separazione fra modi consecutivi, esclusi il cluster delle
    conduttanze (|z| > thr) e gli zeri esatti.
    """
    mods = np.asarray(mods, dtype=float)
    m = np.sort(mods[(mods >= min_mod) & (mods <= thr)])[::-1]
    if len(m) < 2:
        return dict(ratio=np.nan, after=None, n=int(len(m)),
                    tau_slow=np.nan, tau_fast=np.nan)
    rat = np.log(m[1:]) / np.log(m[:-1])
    k = int(np.argmax(rat))
    return dict(ratio=float(rat[k]), after=k + 1, n=int(len(m)),
                tau_slow=float(-1.0 / np.log(m[k])),
                tau_fast=float(-1.0 / np.log(m[k + 1])))


# Errore degli esponenti a tempo finito, stimato nel punto fisso del mosaic (M = 500, burn = 100)
LYAP_ABS_ERR = 2e-3


def analyze_S1(W, B, Rr=4, G=0.05, T=5000, seed=None, verbose=True,
               lyap_steps=LYAP_STEPS):
    net, v, Gp, last_step, sat_hist = state_after_steps(W, B, Rr, G, T, seed)
    N = net.N
    frac_sat = _sat_frac(Gp)
    stationary = bool(last_step <= STATIONARY_TOL)

    J = full_dynamics_jacobian(net, v, Gp)
    w, VL, VR = sl.eig(J, left=True, right=True)
    st = spectral_summary(w, eigvecs=VR, N=N, left_eigvecs=VL)
    st_v = spectral_summary(np.linalg.eigvals(J[:N, :N]))
    gp = gp_block_diagnosis(J, N)
    vd = verdict_S1(st, gp)

    # Quanti dei modi non contraenti sono il cluster Gp perturbato.
    val = gp['gp_diag_value']
    ge1 = st['abs_eigs'] >= 1.0
    n_ge1_gp = (int(np.sum(ge1 & (np.abs(st['abs_eigs'] - val) < 1e-3)))
                if np.isfinite(val) else 0)

    # Fuori dal punto fisso il verdetto si basa sugli esponenti di Lyapunov.
    ly = st_l = vd_l = None
    if not stationary and lyap_steps:
        ly = lyapunov_spectrum(net, v, Gp, M=lyap_steps)
        mods_l = np.r_[np.exp(ly['lyap']), np.zeros(ly['n_dead'])]
        st_l = spectral_summary(mods_l)
        vd_l = verdict_S1(st_l, gp)

    mods_verdict = (st['abs_eigs'] if (stationary or st_l is None)
                    else st_l['abs_eigs'])
    sep = separation_excluding_cluster(mods_verdict)

    out = dict(W=W, B=B, R=Rr, G=G, T=T, seed=seed, N=N, separation=sep,
               frac_sat=frac_sat, last_step=last_step, sat_hist=sat_hist,
               stationary=stationary,
               n_active=int(N - gp['n_gp_frozen']),
               full=st, vblock=st_v, gp=gp, verdict=vd, n_ge1_gp=n_ge1_gp,
               lyap=ly, lyap_summary=st_l, verdict_lyap=vd_l,
               verdict_final=(vd if stationary or vd_l is None else vd_l))

    if verbose:
        print("\n" + "=" * 100)
        print(f"(S1)  W={W:g}  B={B:g}  T={T}  seed={seed}")
        print("=" * 100)
        print(f"  stato: Gp satura {100*frac_sat:.1f}%   "
              f"attive {out['n_active']}/{N}   "
              f"spostamento ultimo passo {last_step:.3e}")
        if last_step > 1e-12:
            print("  NB: stato NON stazionario: guadagno locale, non stabilita'.")
        if sat_hist:
            fr = [f"{100*f:.0f}" for _, f in sat_hist]
            print(f"  saturazione lungo la traiettoria (%): {' '.join(fr)}")
            nonmono = any(b < a - 1e-12 for (_, a), (_, b)
                          in zip(sat_hist[:-1], sat_hist[1:]))
            if nonmono:
                print("  ATTENZIONE: la saturazione NON e' monotona: celle")
                print("  saturano e desaturano, quindi l'insieme dei rami")
                print("  attivi non e' invariante (cfr. oss:nonsmooth).")

        print(f"\n  blocco Gp: diagonale = {gp['gp_diag_value']:.14f}")
        print(f"      1 - diag = {gp['relax_inferred']:.6g}   "
              f"tau corrispondente = "
              f"{1.0/gp['relax_inferred'] if gp['relax_inferred'] > 0 else float('inf'):.4g} passi")
        print(f"      {gp['n_gp_active']} attive, {gp['n_gp_frozen']} nulle   "
              f"(fuori diagonale max {gp['gp_block_offdiag_max']:.2e})")

        print(f"\n  spettro completo (2N = {st['n_total']}):")
        print(f"      rho = {st['rho']:.10f}   |z|>=1: {st['n_noncontracting']} "
              f"(di cui {n_ge1_gp} del cluster Gp)   "
              f"|z|>0.99: {st['n_above_099']}   zeri: {st['n_zero']}")
        print("      decili: " + "  ".join(f"{x:.4f}" for x in st['deciles']))
        tv = st_v['tau']
        fin = tv[np.isfinite(tv) & (tv > 0)]
        print(f"      blocco Vmem: rho = {st_v['rho']:.10f}   "
              f"|z|>=1: {st_v['n_noncontracting']}   "
              + (f"tau in [{fin.min():.2f}, {fin.max():.2f}] passi"
                 if len(fin) else "nessun modo contraente"))

        k = st['gap_after']
        if k is not None:
            print(f"\n  gap dopo il modo {k} (su {st['n_nonzero']} modi non nulli):")
            print(f"      |z_k| = {st['mod_before_gap']:.12g}   "
                  f"|z_k+1| = {st['mod_after_gap']:.12g}")
            print(f"      rapporto moduli = {st['gap_ratio_max']:.6g}   "
                  f"tau_ratio = {st['tau_ratio']}")
            print(f"      (salienza {st['gap_salience']:.3g} — diagnostica, "
                  f"non entra nel verdetto)")

        if 'part_v' in st and k is not None:
            pv = st['part_v']
            print(f"\n  partecipazione di Vmem (fattori di partecipazione, "
                  f"indipendenti dalle unita'):")
            print(f"      modi lenti (1..{k}):  mediana {np.median(pv[:k]):.4f}")
            print(f"      modi veloci ({k+1}..): mediana {np.median(pv[k:]):.4f}")
            print("      vicino a 0 = il modo coinvolge le conduttanze;")
            print("      vicino a 1 = coinvolge i potenziali.")

        def _esito(v_, titolo):
            print(f"\n  ESITO {titolo}: "
                  f"{'(S1) SODDISFATTA' if v_['passes'] else '(S1) NON soddisfatta'}")
            print(f"      {v_['reason']}")
            print(f"      modi lenti {v_['n_slow']}, di cui {v_['n_gp_cluster']} "
                  f"del cluster Gp => {v_['n_slow_net']} netti")

        if stationary:
            _esito(vd, "(punto fisso: spettro della jacobiana)")
        else:
            _esito(vd, "LOCALE (jacobiana a un passo, solo indicativo)")
            if vd_l is not None:
                lam = ly['lyap']
                print(f"\n  Lyapunov a tempo finito su {ly['M']} passi "
                      f"(dopo {ly['burn']} di allineamento): "
                      f"{ly['K']} esponenti, {ly['n_dead']} direzioni sature "
                      f"per tutta la finestra (-inf)")
                print(f"      lambda_max = {lam[0]:+.4e}   "
                      f"esponenti >= 0: {int(np.sum(lam >= 0))}")
                if abs(lam[0]) < LYAP_ABS_ERR:
                    print(f"      NB: |lambda_max| < {LYAP_ABS_ERR:g}, "
                          f"errore stimato a {ly['M']} passi: segno non "
                          f"determinato")
                _esito(vd_l, "(esponenti di Lyapunov)")

        if sep['after'] is not None:
            print(f"\n  separazione fra i soli modi non strutturali "
                  f"(esclusi |z| > 0.99 e zeri, {sep['n']} modi):")
            print(f"      massimo ln-rapporto = {sep['ratio']:.3f} "
                  f"dopo {sep['after']} modi   "
                  f"(tau {sep['tau_slow']:.3g} -> {sep['tau_fast']:.3g} passi)")
            lenti_pochi = sep['after'] <= SLOW_FRAC_MAX * 2 * N
            print(f"      separazione >= {TAU_RATIO_MIN:g}: "
                  f"{sep['ratio'] >= TAU_RATIO_MIN}   "
                  f"lenti prima del salto pochi (<= {SLOW_FRAC_MAX:.0%}): "
                  f"{lenti_pochi}")
    return out


# ═══════════════════════════════════════════════════════════════════════
#  Stazionarieta': sul blocco Vmem
# ═══════════════════════════════════════════════════════════════════════

def scan_over_T(W, B, Ts=(1000, 5000, 20000, 100000), Rr=4, G=0.05,
                seed=None, top_k=TOP_K):
    print("\n" + "#" * 100)
    print(f"#  STAZIONARIETA' DELLO SPETTRO IN T   W={W:g}  B={B:g}  seed={seed}")
    print("#" * 100)

    res = [analyze_S1(W, B, Rr, G, T, seed) for T in Ts]

    print("\n" + "-" * 100)
    print(f"  Deriva dei primi {top_k} moduli del BLOCCO Vmem")
    print("-" * 100)
    print(f"  {'T_a':>8}{'T_b':>10}{'deriva rel. max':>20}"
          f"{'rho_v(a)':>12}{'rho_v(b)':>12}{'staz.':>9}")
    drifts = []
    for a, b in zip(res[:-1], res[1:]):
        ma = a['vblock']['abs_eigs'][:top_k]
        mb = b['vblock']['abs_eigs'][:top_k]
        den = np.maximum(np.abs(ma), np.finfo(float).tiny)
        d = float(np.max(np.abs(mb - ma) / den))
        drifts.append(d)
        print(f"  {a['T']:>8}{b['T']:>10}{d:>20.4e}"
              f"{a['vblock']['rho']:>12.6f}{b['vblock']['rho']:>12.6f}"
              f"{str(d < SPECTRUM_DRIFT_TOL):>9}")

    stationary = bool(drifts) and all(d < SPECTRUM_DRIFT_TOL for d in drifts)
    print(f"\n  -> blocco Vmem {'STAZIONARIO' if stationary else 'NON stazionario'}")
    rv = [r['vblock']['rho'] for r in res]
    if min(rv) < 1.0 <= max(rv):
        print("  ATTENZIONE: rho del blocco Vmem attraversa l'unita' lungo T")
        print(f"  ({' -> '.join(f'{x:.4f}' for x in rv)}): da verificare se sia")
        print("  un effetto della saturazione che cambia nel frattempo.")
    return res, drifts, stationary


def plot_spectra(res, fname='spettro_S1.png'):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for r in res:
        m = np.maximum(r['full']['abs_eigs'], 1e-18)
        ax.semilogy(np.arange(1, len(m) + 1), m, marker='.', lw=1,
                    label=f"T = {r['T']}")
        k = r['full']['gap_after']
        if k is not None:
            ax.axvline(k + 0.5, ls=':', lw=0.8, alpha=0.5)
    ax.axhline(1.0, color='k', lw=0.8)
    ax.set_xlabel('modo (ordinato per modulo decrescente)')
    ax.set_ylabel(r'$|z|$')
    ax.set_title(f"Spettro della Jacobiana 2N "
                 f"(W={res[0]['W']:g}, B={res[0]['B']:g})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figpath(fname), dpi=150)
    plt.close(fig)
    print(f"  {FIGDIR}/{fname}")


# ═══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':

    print("CRITERI DICHIARATI (invariati rispetto alla v1)")
    print(f"  (S1a) tau_ratio al gap >= {TAU_RATIO_MIN}")
    print(f"  (S1b) modi lenti netti <= {SLOW_FRAC_MAX:.0%} di 2N")
    print("  gap cercato su tutto lo spettro non nullo")

    tutto = {}

    res_T, drifts, stat = scan_over_T(
        W=1000.0, B=5e-4, Ts=(1000, 5000, 20000, 100000), seed=None)
    plot_spectra(res_T)
    tutto['stigmergico_T'] = dict(res=res_T, drifts=drifts, stationary=stat)

    tutto['stigmergico_seed'] = [
        analyze_S1(1000.0, 5e-4, T=5000, seed=s) for s in (1, 2)]

    tutto['colonna_B5e-4'] = [
        analyze_S1(W, 5e-4, T=5000, seed=None)
        for W in (100.0, 200.0, 500.0, 1000.0)]

    tutto['mosaic'] = analyze_S1(10.0, 0.02, T=5000, seed=None)

    torch.save(tutto, R.out_path('spettro_S1_v2.pt'))
    print(f"\nSalvato: {R.out_path('spettro_S1_v2.pt')}")



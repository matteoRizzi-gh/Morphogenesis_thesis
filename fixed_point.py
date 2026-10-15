"""
fixed_point.py
==============
Analisi di stabilita' del punto fisso del sistema accoppiato (Vmem, Gp),
da condizioni iniziali omogenee e da condizioni eterogenee.

"""

import os
import numpy as np
import torch
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

import Rfc_and_ID as R
from dynamics_t_sparse_gh import (BioelectricLattice, C, E_p, E_d, G_r, G_d,
                                  k_e, e_r, z, v_th, v_T, v_0, DT,
                                  EPSILON, GAIN, FIELD_STRENGTH, V_INIT)

# SAT_EDGE = 1e-12
SAT_EDGE = 0.0
# Il clamp torch.clamp(x, 0, 2*G_r) restituisce ESATTAMENTE 0.0 o 2*G_r quando
# taglia, quindi il test esatto e' il test corretto per "questa cella e' stata
# troncata".

# ═══════════════════════════════════════════════════════════════════════
#  Passo puro (senza stato interno)
# ═══════════════════════════════════════════════════════════════════════

def step_pure(v, Gp, net):
    """
    Un passo della dinamica autonoma come funzione pura di (v, Gp).
    Deve coincidere con net.step(v_left_22=None) a precisione di macchina:
    verify_step_pure() lo controlla. 
    """
    q  = C * v
    qg = q.index_select(0, net.endo_c_idx)
    fc = FIELD_STRENGTH * (k_e / e_r)
    ex = torch.zeros(net.N_f, dtype=torch.float64).scatter_add_(
        0, net.endo_f_idx, net.endo_coeff_x * qg) * fc
    ey = torch.zeros(net.N_f, dtype=torch.float64).scatter_add_(
        0, net.endo_f_idx, net.endo_coeff_y * qg) * fc
    eV = torch.sqrt(ex**2 + ey**2 + EPSILON)

    e_g = eV.index_select(0, net.perc_field_idx)
    s   = torch.zeros(net.N, dtype=torch.float64).scatter_add_(
        0, net.perc_cell_idx, e_g)
    avg_e = s / net.perc_n
    sigma = torch.sigmoid(net.gain * avg_e + net.B)
    dp    = -Gp + (2.0 * sigma - 1.0) * net.W
    Gp_new = torch.clamp(Gp + DT * dp * G_r, 0.0, 2.0 * G_r)

    arg   = z * (v - v_th) / v_T
    I_pol = -Gp_new * (v - E_p) / (1.0 + torch.exp( arg))
    I_dep = -G_d    * (v - E_d) / (1.0 + torch.exp(-arg))
    dv_e  = v[net.ei] - v[net.ej]
    G_ij  = 2.0 * net.G * G_r / (1.0 + torch.cosh(dv_e / v_0))
    I_gj  = torch.zeros(net.N, dtype=torch.float64)
    I_gj  = I_gj.scatter_add(0, net.ei, G_ij * (v[net.ej] - v[net.ei]))
    I_gj  = I_gj.scatter_add(0, net.ej, G_ij * (v[net.ei] - v[net.ej]))
    v_new = v + DT * (I_pol + I_dep + I_gj) / C
    return v_new, Gp_new


def verify_step_pure(W=1000.0, B=5e-4, Rr=4, G=0.05, n_steps=50, seed=0):
    """Se questo fallisce, tutto il resto del file e' privo di significato."""
    net = BioelectricLattice(rows=11, cols=11, G=G, R=Rr, W=W, B=B)
    net.reset()
    g = torch.Generator().manual_seed(seed)
    net.Gp = torch.rand(net.N, generator=g, dtype=torch.float64) * 2.0 * G_r
    v, Gp = net.v.clone(), net.Gp.clone()
    worst = 0.0
    with torch.no_grad():
        for _ in range(n_steps):
            net.step(v_left_22=None)
            v, Gp = step_pure(v, Gp, net)
            worst = max(worst, float((v - net.v).abs().max()),
                        float((Gp - net.Gp).abs().max()))
    ok = worst < 1e-15
    print(f"  verify_step_pure: differenza massima su {n_steps} passi = "
          f"{worst:.2e}   {'OK' if ok else 'DISCREPANZA'}")
    return ok


# ═══════════════════════════════════════════════════════════════════════
#  Punto fisso
# ═══════════════════════════════════════════════════════════════════════

def find_fixed_point(net, 
                     # T_max=50000, 
                     T_max = 200000, 
                     tol=1e-14, v0=None, Gp0=None):
    """
    Itera la mappa finche' lo spostamento per passo non scende sotto tol.

    """
    v  = (torch.full((net.N,), V_INIT, dtype=torch.float64)
          if v0 is None else v0.clone())
    Gp = (torch.full((net.N,), 1.0*G_r, dtype=torch.float64)
          if Gp0 is None else Gp0.clone())
    hist, d = [], float('inf')
    with torch.no_grad():
        for t in range(T_max):
            v_new, Gp_new = step_pure(v, Gp, net)
            d = float(torch.max((v_new - v).abs().max(),
                                (Gp_new - Gp).abs().max() / G_r))
            v, Gp = v_new, Gp_new
            if t % 100 == 0:
                hist.append((t, d))
            if d < tol:
                return dict(v=v, Gp=Gp, converged=True, iters=t + 1,
                            last_step=d, hist=hist)
    return dict(v=v, Gp=Gp, converged=False, iters=T_max, last_step=d,
                hist=hist)


def _frac_saturated(Gp):
    return float(((Gp <= 0.0) | (Gp >= 2.0*G_r)).double().mean())

# ═══════════════════════════════════════════════════════════════════════
#  Stabilita'
# ═══════════════════════════════════════════════════════════════════════

def jacobian_at(net, v, Gp):
    """
    Jacobiana 2N x 2N della mappa a un passo, in (v, Gp).
    Dove il clamp e' attivo la derivata rispetto a Gp e' nulla: quelle celle
    escono dall'anello
    """
    N = net.N
    def f(state):
        vv, gg = state[:N], state[N:]
        v2, g2 = step_pure(vv, gg, net)
        return torch.cat([v2, g2])
    st = torch.cat([v.detach().clone(), Gp.detach().clone()])
    return torch.autograd.functional.jacobian(f, st)


def stability(net, v, Gp):
    J = jacobian_at(net, v, Gp)
    mag = np.abs(torch.linalg.eigvals(J).numpy())
    N = net.N
    Jg = J[N:, :].abs().sum(1).numpy()      # righe di Gp identicamente nulle
    return dict(
        spectral_radius=float(mag.max()),
        n_unstable=int((mag > 1.0 + 1e-12).sum()),
        n_marginal=int((np.abs(mag - 1.0) <= 1e-12).sum()),
        eig_mag_top=np.sort(mag)[::-1][:10],
        frac_gp_saturated=_frac_saturated(Gp),
        n_gp_frozen=int((Jg < 1e-30).sum()),
        stable=bool(mag.max() < 1.0 - 1e-12),
    )


# ═══════════════════════════════════════════════════════════════════════
#  Campo medio scalare: controllo indipendente
# ═══════════════════════════════════════════════════════════════════════

def kappa(net):
    """In uno stato omogeneo v_j = v il campo percepito e' e = kappa*|v|."""
    fc = FIELD_STRENGTH * (k_e / e_r) * C
    ex = torch.zeros(net.N_f, dtype=torch.float64).scatter_add_(
        0, net.endo_f_idx, net.endo_coeff_x)
    ey = torch.zeros(net.N_f, dtype=torch.float64).scatter_add_(
        0, net.endo_f_idx, net.endo_coeff_y)
    eV_unit = fc * torch.sqrt(ex**2 + ey**2)
    g = eV_unit.index_select(0, net.perc_field_idx)
    s = torch.zeros(net.N, dtype=torch.float64).scatter_add_(
        0, net.perc_cell_idx, g)
    return (s / net.perc_n).numpy()


def mean_field_scan(net, n=4001, mode='implemented'):
    """
    Punti fissi omogenei in approssimazione di campo medio.
    Su uno stato omogeneo le gap junction si annullano, quindi
        F(v) = I_pol(v, Gp*(v)) + I_dep(v)
    e gli zeri di F sono i punti fissi omogenei.

    Il set-point Gp*(v) dipende da QUALE modello si sta analizzando:

    mode='implemented'
        La mappa come e' scritta nel codice:
            Gp <- clamp(Gp + DT*(-Gp + drive)*G_r, 0, 2*G_r)

    mode='intended'
        La variante dimensionalmente coerente, in cui G_ref scala il solo
        set-point:
            Gp <- clamp(Gp + DT*(-Gp + drive*G_r), 0, 2*G_r)
        Qui g* = drive e il clamp morde solo se drive esce da (0, 2).
    """
    if mode not in ('implemented', 'intended'):
        raise ValueError(f"mode non riconosciuto: {mode!r}")

    k  = float(np.mean(kappa(net)))
    vs = np.linspace(E_p, E_d, n)

    def _drive_of(x):
        e = k * np.abs(x)
        return (2.0 / (1.0 + np.exp(-(GAIN * e + net.B))) - 1.0) * net.W

    def _gp_star(x):
        """Set-point NON troncato, in siemens."""
        d = _drive_of(x)
        return d * G_r if mode == 'intended' else d

    def _F(x):
        gp = np.clip(_gp_star(x), 0.0, 2.0 * G_r)
        a  = z * (x - v_th) / v_T
        return (-gp * (x - E_p) / (1 + np.exp(a))) + \
               (-G_d * (x - E_d) / (1 + np.exp(-a)))

    F = _F(vs)
    roots = []
    for i in range(n - 1):
        if F[i] == 0.0 or F[i] * F[i + 1] < 0:
            lo, hi = vs[i], vs[i + 1]
            f_lo = F[i]
            for _ in range(80):
                mid = 0.5 * (lo + hi)
                f_mid = _F(mid)
                if f_lo * f_mid <= 0:
                    hi = mid
                else:
                    lo, f_lo = mid, f_mid
            r  = 0.5 * (lo + hi)
            gs = float(_gp_star(r))
            roots.append(dict(v=float(r),
                              Gp_unclamped_over_Gr=gs / G_r,
                              saturated=bool(gs <= 0.0 or gs >= 2.0 * G_r)))
    return dict(kappa=k, mode=mode, roots=roots, v=vs, F=F)

# ═══════════════════════════════════════════════════════════════════════
#  Condizioni OMOGENEE
# ═══════════════════════════════════════════════════════════════════════

def analyze_cell(W, B, Rr=4, G=0.05, 
                 # T_max=50000, 
                 T_max=300000, 
                 verbose=True):
    net = BioelectricLattice(rows=11, cols=11, G=G, R=Rr, W=W, B=B)
    fp  = find_fixed_point(net, T_max=T_max)
    out = dict(W=W, B=B, R=Rr, G=G, converged=fp['converged'],
               iters=fp['iters'], last_step=fp['last_step'])
    if fp['converged']:
        out.update(stability(net, fp['v'], fp['Gp']))
        out['v_mean'] = float(fp['v'].mean())
        out['v_std']  = float(fp['v'].std())
        out['gp_mean_over_Gr'] = float(fp['Gp'].mean() / G_r)
        out['predict_collapse'] = out['stable']
    else:
        out.update(spectral_radius=float('nan'), stable=False,
                   n_gp_frozen=-1,
                   frac_gp_saturated=_frac_saturated(fp['Gp']),
                   predict_collapse=False)

    # Campo medio nei DUE modelli. 'implemented' e' quello simulato dal codice;
    # 'intended' e' la variante dimensionalmente coerente, riportata per confronto
    mf_i = mean_field_scan(net, mode='implemented')
    mf_c = mean_field_scan(net, mode='intended')
    out['mf_kappa'] = mf_i['kappa']
    out['mf_implemented_n_roots'] = len(mf_i['roots'])
    out['mf_implemented_roots']   = mf_i['roots']
    out['mf_implemented_all_saturated'] = all(r['saturated']
                                              for r in mf_i['roots']) \
                                          if mf_i['roots'] else True
    out['mf_intended_n_roots'] = len(mf_c['roots'])
    out['mf_intended_roots']   = mf_c['roots']
    out['mf_intended_n_interior'] = sum(1 for r in mf_c['roots']
                                        if not r['saturated'])

    if verbose:
        print(f"  W={W:>6.0f} B={B:.4f}  "
              f"{'converge' if out['converged'] else 'NON converge':>12} "
              f"({out['iters']} iter)  rho={out['spectral_radius']:.6f}  "
              f"Gp satura {100*out['frac_gp_saturated']:5.1f}%  "
              f"congelate {out['n_gp_frozen']:>4}  "
              f"radici c.m. impl/int "
              f"{out['mf_implemented_n_roots']}/{out['mf_intended_n_roots']}"
              f" (int. interne: {out['mf_intended_n_interior']})  "
              f"-> {'COLLASSO' if out['predict_collapse'] else 'molti stati'}")
    return out


def grid_prediction(Ws=R.PAPER_W, Bs=R.PAPER_B, Rr=4, G=0.05,
                    save_as='fixed_point.pt'):
    print(f"\n{'='*112}\n  PUNTO FISSO E STABILITA', CONDIZIONI OMOGENEE  "
          f"(R={Rr}, G={G})\n{'='*112}")
    res = [analyze_cell(W, B, Rr, G) for W in Ws for B in Bs]
    torch.save(res, R.out_path(save_as))
    return res


# ═══════════════════════════════════════════════════════════════════════
#  Condizioni ETEROGENEE
# ═══════════════════════════════════════════════════════════════════════

def fixed_points_heterogeneous(net, n_starts=30,
                               T_max=300000,
                               tol=1e-14,
                               seed=0, cluster_tol=1e-6):
    """
    Punti fissi raggiunti da G_pol casuale in [0, 2*G_r], cioe' lo stesso
    ensemble di condizioni iniziali usato per costruire le nuvole.
    Il conteggio dei punti fissi distinti usa il single-linkage, come
    h0_profile: canonico e indipendente dall'ordine dei punti.

    """
    recs, states = [], []
    for i in range(n_starts):
        g   = torch.Generator().manual_seed(seed * 1000 + i)
        Gp0 = torch.rand(net.N, generator=g, dtype=torch.float64) * 2.0 * G_r
        v0  = torch.full((net.N,), V_INIT, dtype=torch.float64)
        fp  = find_fixed_point(net, T_max=T_max, tol=tol, v0=v0, Gp0=Gp0)
        rec = dict(start=i, converged=fp['converged'], iters=fp['iters'],
                   last_step=fp['last_step'],
                   frac_sat=_frac_saturated(fp['Gp']))
        if fp['converged']:
            rec.update(stability(net, fp['v'], fp['Gp']))
            states.append(fp['v'].numpy())
        else:
            rec.update(spectral_radius=float('nan'), stable=False,
                       n_gp_frozen=-1)
        recs.append(rec)

    n_distinct = 0
    if len(states) > 1:
        X = np.asarray(states)
        d = pdist(X) / np.sqrt(X.shape[1])
        n_distinct = (1 if d.max() == 0.0 else
                      int(len(np.unique(fcluster(
                          linkage(d, method='single'),
                          cluster_tol, criterion='distance')))))
    elif len(states) == 1:
        n_distinct = 1

    conv = [r for r in recs if r['converged']]
    rho  = np.array([r['spectral_radius'] for r in conv]) if conv else np.array([])
    sats = np.array([r['frac_sat'] for r in recs])
    its  = np.array([r['iters'] for r in conv]) if conv else np.array([])
    return dict(
        n_starts=n_starts, n_converged=len(conv), n_distinct=n_distinct,
        n_stable=int(sum(1 for r in conv if r['stable'])),
        rho_min=float(rho.min()) if len(rho) else float('nan'),
        rho_max=float(rho.max()) if len(rho) else float('nan'),
        rho_unique=int(len(np.unique(np.round(rho, 6)))) if len(rho) else 0,
        frac_sat_min=float(sats.min()), frac_sat_max=float(sats.max()),
        frac_sat_mean=float(sats.mean()),
        iters_median=int(np.median(its)) if len(its) else None,
        iters_max=int(its.max()) if len(its) else None,
        iters_min=int(its.min()) if len(its) else None,
        records=recs,
    )

def analyze_cell_heterogeneous(W, B, Rr=4, G=0.05, n_starts=30,
                               # T_max=50000,
                               T_max=300000, 
                               verbose=True):
    net = BioelectricLattice(rows=11, cols=11, G=G, R=Rr, W=W, B=B)
    het = fixed_points_heterogeneous(net, n_starts=n_starts, T_max=T_max)
    # collasso previsto se TUTTE le partenze convergono a UN solo stato
    predict = (het['n_converged'] == n_starts) and (het['n_distinct'] == 1)
    out = dict(W=W, B=B, R=Rr, G=G, predict_collapse=predict, **het)
    if verbose:
        print(f"  W={W:>6.0f} B={B:.4f}  "
              f"convergono {het['n_converged']:>3}/{n_starts}  "
              f"distinti {het['n_distinct']:>3}  stabili {het['n_stable']:>3}  "
              f"rho in [{het['rho_min']:.6f}, {het['rho_max']:.6f}] "
              f"({het['rho_unique']} valori)  "
              f"Gp sat {100*het['frac_sat_min']:.0f}-"
              f"{100*het['frac_sat_max']:.0f}%  "
              f"-> {'COLLASSO' if predict else 'molti stati'}")
    return out


def grid_heterogeneous(cells=None, Rr=4, G=0.05, n_starts=30, 
                       # T_max=50000,
                        T_max=300000,
                       save_as='fixed_point_het.pt'):
    """
    Default: le 4 celle dove la previsione omogenea sbagliava, piu' due
    controlli (una collassata certa, una viva certa) per verificare che il
    metodo eterogeneo non rompa i casi che gia' funzionavano.
    """
    if cells is None:
        cells = [(10.0, 0.005), (100.0, 0.002), (200.0, 0.002), (500.0, 0.002),
                 (1000.0, 0.002), (1000.0, 0.0005)]
    print(f"\n{'='*118}\n  PUNTI FISSI DA CONDIZIONI ETEROGENEE  "
          f"(R={Rr}, G={G}, {n_starts} partenze)\n{'='*118}")
    res = [analyze_cell_heterogeneous(W, B, Rr, G, n_starts, T_max)
           for W, B in cells]
    torch.save(res, R.out_path(save_as))
    return res


# ═══════════════════════════════════════════════════════════════════════
#  Confronti con la griglia osservata
# ═══════════════════════════════════════════════════════════════════════

def _load_observed(observed_pt='grid_gp.pt'):
    path = R.out_path(observed_pt)
    if not os.path.exists(path):
        print(f"\n  (salto il confronto: manca {path})")
        return None
    return {(round(r['W'], 6), round(r['B'], 6)): r
            for r in torch.load(path, weights_only=False)}


def check_collapse_state(pred, v_tol=1e-9):
    """
    Le celle collassate hanno tutte rho = 0.844814 identico.
    """
    coll = [p for p in pred
            if p['converged'] and np.isfinite(p['spectral_radius'])
            and p['frac_gp_saturated'] >= 1.0 - 1e-9]
    if not coll:
        print("\n  (nessuna cella collassata da confrontare)")
        return
    print(f"\n{'='*92}\n  STATO DELLE CELLE COLLASSATE  "
          f"(perche' rho e' identico?)\n{'='*92}")
    print(f"  {'W':>7}{'B':>9}{'rho':>11}{'v_mean (mV)':>14}"
          f"{'v_std (mV)':>13}{'Gp/G_r':>10}")
    for p in sorted(coll, key=lambda r: (r['W'], r['B'])):
        print(f"  {p['W']:7.0f}{p['B']:9.4f}{p['spectral_radius']:11.6f}"
              f"{1e3*p['v_mean']:14.6f}{1e3*p['v_std']:13.2e}"
              f"{p['gp_mean_over_Gr']:10.4f}")

    vm = np.array([p['v_mean'] for p in coll])
    vs = np.array([p['v_std']  for p in coll])
    gp = np.array([p['gp_mean_over_Gr'] for p in coll])
    spread = float(vm.max() - vm.min())
    print(f"\n  v_mean: spread fra le celle {1e3*spread:.2e} mV  "
          f"(intervallo [{1e3*vm.min():.6f}, {1e3*vm.max():.6f}] mV)")
    print(f"  v_std massimo fra le celle: {1e3*vs.max():.2e} mV")
    print(f"  Gp/G_r: intervallo [{gp.min():.4f}, {gp.max():.4f}]")

    same = (spread < v_tol) and (vs.max() < v_tol)
    if same:
        print("\n  -> STESSO stato uniforme in tutte le celle collassate:")
        print("     rho identico e' spiegato. Gp pinnata -> W, B escono dal")
        print("     punto fisso -> lo stato e' lo stesso ovunque.")
    else:
        print("\n  -> stato NON identico (o non uniforme) ma rho identico:")
        print("     l'universalita' di rho non viene dallo stato e va")
        print("     spiegata a parte.")
    return dict(spread_vmean=spread, v_std_max=float(vs.max()), same_state=bool(same))


def compare_with_observed(pred, observed_pt='grid_gp.pt'):
    obs = _load_observed(observed_pt)
    if obs is None:
        return
    print(f"\n{'='*92}\n  PREVISIONE OMOGENEA CONTRO OSSERVAZIONE\n{'='*92}")
    print(f"  {'W':>7}{'B':>9}{'rho':>11}{'previsto':>12}"
          f"{'osservato':>12}{'Gp sat prev':>13}{'Gp sat oss':>12}{'':>6}")
    ok = tot = 0
    missing = []
    for p in sorted(pred, key=lambda r: (r['W'], r['B'])):
        o = obs.get((round(p['W'], 6), round(p['B'], 6)))
        if o is None:
            missing.append((p['W'], p['B']))
            continue
        obs_collapse = R.is_collapsed(o['vmem_spread'])
        agree = (p['predict_collapse'] == obs_collapse)
        ok += agree
        tot += 1
        print(f"  {p['W']:7.0f}{p['B']:9.4f}{p['spectral_radius']:11.6f}"
              f"{'collasso' if p['predict_collapse'] else 'molti':>12}"
              f"{'collasso' if obs_collapse else 'molti':>12}"
              f"{100*p['frac_gp_saturated']:12.1f}%"
              f"{100*(o['gp']['frac_zero']+o['gp']['frac_max']):11.1f}%"
              f"{'  si' if agree else '  NO':>6}")
    print(f"\n  accordo: {ok}/{len(pred)} celle")
    if missing:
        raise RuntimeError(
            f"grid_gp.pt contiene {tot} celle su {len(pred)}. "
            f"Mancano: {missing}. Rilancia grid_and_gp.grid() prima di "
            f"leggere questo numero.")
    return ok


def compare_heterogeneous(pred_het, observed_pt='grid_gp.pt'):
    """Previsione da condizioni eterogenee contro la nuvola osservata. """
    obs = _load_observed(observed_pt)
    if obs is None:
        return
    print(f"\n{'='*100}\n  PREVISIONE ETEROGENEA CONTRO OSSERVAZIONE\n{'='*100}")
    print(f"  {'W':>7}{'B':>9}{'distinti':>10}{'converg.':>10}"
          f"{'previsto':>12}{'osservato':>12}{'H0 oss':>9}"
          f"{'t_conv max':>12}{'':>6}")
    ok = tot = 0
    missing = []
    for p in pred_het:
        o = obs.get((round(p['W'], 6), round(p['B'], 6)))
        if o is None:
            missing.append((p['W'], p['B']))
            continue
        obs_collapse = R.is_collapsed(o['vmem_spread'])
        h0 = o['h0']['counts'][min(o['h0']['counts'],
                                   key=lambda t: abs(t - 1e-6))]
        agree = (p['predict_collapse'] == obs_collapse)
        ok += agree
        tot += 1
        tc = p.get('iters_max')
        print(f"  {p['W']:7.0f}{p['B']:9.4f}{p['n_distinct']:10d}"
              f"{p['n_converged']:>6}/{p['n_starts']:<3}"
              f"{'collasso' if p['predict_collapse'] else 'molti':>12}"
              f"{'collasso' if obs_collapse else 'molti':>12}"
              f"{h0:9d}{(str(tc) if tc else '>T_max'):>12}"
              f"{'  si' if agree else '  NO':>6}")

    print(f"\n  gia' collassate all'osservazione: {ok}/{tot} celle confrontabili")
    if missing:
        raise RuntimeError(
            f"{observed_pt} contiene {tot} celle su {len(pred_het)} predette. "
            f"Mancano: {missing}. Rilancia grid_and_gp.grid(): le celle "
            f"assenti non contano come disaccordo ma restano nel denominatore.")
    return dict(ok=ok, tot=tot)

# ═══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':

    N_STARTS = 30      

    print("Verifica preliminare")
    if not verify_step_pure():
        raise SystemExit("step_pure non riproduce net.step: fermarsi qui.")
    
    
    pred = grid_prediction()
    compare_with_observed(pred)
    check_collapse_state(pred)

    
    het = grid_heterogeneous(n_starts=N_STARTS)
    compare_heterogeneous(het)


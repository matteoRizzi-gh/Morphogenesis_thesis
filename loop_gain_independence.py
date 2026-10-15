"""
loop_gain_independence.py
============
Spettro del TERMINE D'ANELLO della retroazione, e confronto con le mappe di
campo del modello.

"""

import numpy as np
import torch

import Rfc_and_ID as R
from dynamics_t_sparse_gh import (BioelectricLattice, C, E_p, G_r,
                                  k_e, e_r, z, v_th, v_T, DT,
                                  EPSILON, FIELD_STRENGTH)

# SAT_EDGE = 1e-12
SAT_EDGE = 0.0

# ═══════════════════════════════════════════════════════════════════════
#  Le tre mappe di campo
# ═══════════════════════════════════════════════════════════════════════

def field_raw(v, net):
    """(a) v -> eV sui punti di campo. Nessuna aggregazione. [N_f]"""
    q   = C * v
    q_g = q.index_select(0, net.endo_c_idx)
    fc  = FIELD_STRENGTH * (k_e / e_r)
    ex = torch.zeros(net.N_f, dtype=torch.float64).scatter_add_(
        0, net.endo_f_idx, net.endo_coeff_x * q_g) * fc
    ey = torch.zeros(net.N_f, dtype=torch.float64).scatter_add_(
        0, net.endo_f_idx, net.endo_coeff_y * q_g) * fc
    return torch.sqrt(ex**2 + ey**2 + EPSILON)


def field_observable(v, net):
    """(b) Campo OSSERVABILE: aggregazione con la maschera d'azione (raggio R)."""
    eV = field_raw(v, net)
    return torch.zeros(net.N, dtype=torch.float64).scatter_add_(
        0, net.endo_c_idx,
        eV.index_select(0, net.endo_f_idx)) / R.NUM_FIELD_NEIGHBORS


def field_perceived(v, net):
    """(c) Campo PERCEPITO: media sui punti adiacenti (maschera di percezione, R=1 fisso)"""
    eV = field_raw(v, net)
    e_g = eV.index_select(0, net.perc_field_idx)
    s = torch.zeros(net.N, dtype=torch.float64).scatter_add_(
        0, net.perc_cell_idx, e_g)
    return s / net.perc_n


# ═══════════════════════════════════════════════════════════════════════
#  I due fattori del termine d'anello
# ═══════════════════════════════════════════════════════════════════════

def gp_new_from_v(v, Gp, net):
    """v -> (c) -> sigmoide -> clamp -> Gp_new. Replica _update_Gp. Il clamp ha derivata nulla al bordo."""
    avg_e = field_perceived(v, net)
    sigma = torch.sigmoid(net.gain * avg_e + net.B)
    dp = -Gp + (2.0 * sigma - 1.0) * net.W
    return torch.clamp(Gp + DT * dp * G_r, 0.0, 2.0 * G_r)


def dv_dgp_diagonal(v, net):
    """dv_new/dGp_new, diagonale. Da v_new = v + DT*(I_pol+...)/C con
    I_pol = -Gp*(v-E_p)/(1+exp(arg))  ->  d/dGp = -(v-E_p)/(1+exp(arg))."""
    arg = z * (v - v_th) / v_T
    return (DT / C) * (-(v - E_p) / (1.0 + torch.exp(arg)))


# ═══════════════════════════════════════════════════════════════════════
#  Spettro
# ═══════════════════════════════════════════════════════════════════════

def spectrum_stats(M, levels=(0.90, 0.99), row_scale=None):
    """SVD e concentrazione spettrale. """
    M = np.asarray(M, dtype=np.float64)
    if row_scale is not None:
        s = np.asarray(row_scale, dtype=np.float64).ravel()
        if s.shape[0] != M.shape[0]:
            raise ValueError(
                f"row_scale ha {s.shape[0]} elementi, M ha {M.shape[0]} righe")
        M = M / np.where(s > 0, s, 1.0)[:, None]

    sv = np.linalg.svd(M, compute_uv=False)
    energy = sv**2
    tot = float(energy.sum())
    out = dict(sv=sv, n=min(M.shape), s_max=float(sv[0]) if sv.size else 0.0,
               rank_num=int((sv > 1e-12 * sv[0]).sum()) if sv.size and sv[0] > 0
                        else 0,
               degenerate=(tot <= 0), scaled=(row_scale is not None))
    if tot <= 0:                       # matrice nulla: anello aperto
        for L in levels:
            out[f'dir{int(100*L)}'] = 0
        out['participation'] = 0.0
        return out
    cum = np.cumsum(energy) / tot
    for L in levels:
        out[f'dir{int(100*L)}'] = int(np.searchsorted(cum, L) + 1)
    out['participation'] = float(sv.sum()**2 / tot)
    return out



# ═══════════════════════════════════════════════════════════════════════
#  Stato e calcolo
# ═══════════════════════════════════════════════════════════════════════

def state_at(W, B, Rr=4, G=0.05, T=5000, seed=0):
    """Stato (v, Gp) dopo T passi da Gp iniziale casuale, come nelle nuvole. """
    net = BioelectricLattice(rows=11, cols=11, G=G, R=Rr, W=W, B=B)
    net.reset()
    g = torch.Generator().manual_seed(int(seed))
    net.Gp = torch.rand(net.N, generator=g, dtype=torch.float64) * 2.0 * G_r
    with torch.no_grad():
        for _ in range(T):
            net.step(v_left_22=None)
    return net, net.v.detach().clone(), net.Gp.detach().clone()


def jacobians_at(net, v, Gp):
    """Termine d'anello e le tre mappe di campo, tutte nello STESSO stato."""
    J_gp_v = torch.autograd.functional.jacobian(
        lambda x: gp_new_from_v(x, Gp, net), v)
    J_loop = torch.diag(dv_dgp_diagonal(v, net)) @ J_gp_v

    J_obs = torch.autograd.functional.jacobian(
        lambda x: field_observable(x, net), v)
    J_per = torch.autograd.functional.jacobian(
        lambda x: field_perceived(x, net), v)

    frac_sat = float(((Gp <= 0.0) | (Gp >= 2.0 * G_r)).double().mean())
    return dict(loop=J_loop.numpy(), obs=J_obs.numpy(), per=J_per.numpy(),
                frac_sat=frac_sat)



def cloud_row_scale(W, B, Rr=4, G=0.05, T=5000, n_points=100, seed=0):
    """Deviazione standard per cella di Vmem sulla nuvola di stati a (W,B,T). """
    X = np.array([R.run_snapshots(W=W, B=B, R=Rr, G=G,
                                  seed=seed * 100000 + i,
                                  snapshots=(T,))[T]['vmem']
                  for i in range(n_points)])
    return X.std(axis=0)


# ═══════════════════════════════════════════════════════════════════════

def loop_closed_mask(net, v, Gp):
    """Maschera delle celle per cui il valore pre-clamp e' interno."""
    with torch.no_grad():
        avg_e = field_perceived(v, net)
        sigma = torch.sigmoid(net.gain * avg_e + net.B)
        pre = Gp + DT * (-Gp + (2.0*sigma - 1.0) * net.W) * G_r
        return (pre > 0.0) & (pre < 2.0*G_r)


def n_loop_closed(net, v, Gp):
    """Numero di celle con anello chiuso, cioe' pre-clamp interno."""
    return int(loop_closed_mask(net, v, Gp).sum())


def active_rank_stats(J_per, active_mask):
    """Rango numerico delle sole righe attive di J_per.

    Con i     fattori diagonali attivi non nulli, rank(J_loop)=|A| 
    richiede che le righe attive di J_per siano linearmente indipendenti.
    """
    A = torch.nonzero(active_mask, as_tuple=False).flatten().cpu().numpy()
    n_active = int(A.size)

    if n_active == 0:
        return dict(n_active=0, rank_active=0, sigma_min=0.0,
                    sigma_max=0.0, conditioning=0.0)

    J_A = np.asarray(J_per, dtype=np.float64)[A, :]
    sv = np.linalg.svd(J_A, compute_uv=False)
    sigma_max = float(sv[0])
    sigma_min = float(sv[-1])
    rank_active = int((sv > 1e-12 * sigma_max).sum()) if sigma_max > 0 else 0
    conditioning = sigma_min / sigma_max if sigma_max > 0 else 0.0

    return dict(n_active=n_active, rank_active=rank_active,
                sigma_min=sigma_min, sigma_max=sigma_max,
                conditioning=conditioning)


def verify_rank_identity(cells=None, Rr=4, G=0.05, T=5000, seeds=(0, 1, 2)):
    """Verifica numerica del rango del termine d'anello. """
    if cells is None:
        cells = [(W, B) for W in R.PAPER_W for B in R.PAPER_B]
    print(f"\n{'='*122}")
    print(f"  RANGO DEL TERMINE D'ANELLO   (R={Rr}, G={G}, T={T})")
    print(f"{'='*122}")
    print("  A = {i : Gp_pre-clamp(i) in (0, 2 G_r)}   (maschera esatta)")
    print("  'stato@0'    = celle non sature nello stato corrente, soglia esatta")
    print("  'stato@1e-12'= stesso conteggio con la vecchia tolleranza SAT_EDGE")
    print("  'rimbalzi'   = celle con Gp ESATTAMENTE al bordo e pre-clamp interno")
    print("                 (il fenomeno che giustificherebbe una distinzione")
    print("                  concettuale fra le due formulazioni)\n")
    print(f"  {'W':>7}{'B':>9}{'seed':>6}{'|A|':>7}{'rank loop':>11}"
          f"{'rank A-per':>12}{'stato@0':>10}{'stato@1e-12':>13}"
          f"{'rimbalzi':>10}{'cond A-per':>13}")

    ok_exact = ok_active = ok_state0 = ok_state_tol = tot = 0
    n_bounce_tot = 0
    min_cond = np.inf

    for (W, B) in cells:
        for sd in seeds:
            net, v, Gp = state_at(W, B, Rr, G, T, seed=sd)
            J = jacobians_at(net, v, Gp)

            active = loop_closed_mask(net, v, Gp)
            n_closed = int(active.sum())

            rk_loop = spectrum_stats(J['loop'])['rank_num']
            ar = active_rank_stats(J['per'], active)
            rk_active = ar['rank_active']
            cond = ar['conditioning']
            if n_closed > 0:
                min_cond = min(min_cond, cond)

            with torch.no_grad():
                interior_0 = (Gp > 0.0) & (Gp < 2.0 * G_r)
                interior_t = (Gp > 1e-12) & (Gp < 2.0 * G_r - 1e-12)
                at_edge = (Gp <= 0.0) | (Gp >= 2.0 * G_r)
                n_bounce = int((at_edge & active).sum())
            n_state0 = int(interior_0.sum())
            n_statet = int(interior_t.sum())
            n_bounce_tot += n_bounce

            ok_exact     += (n_closed == rk_loop)
            ok_active    += (n_closed == rk_active)
            ok_state0    += (n_state0 == rk_loop)
            ok_state_tol += (n_statet == rk_loop)
            tot += 1

            print(f"  {W:7.0f}{B:9.4f}{sd:6d}{n_closed:7d}{rk_loop:11d}"
                  f"{rk_active:12d}{n_state0:10d}{n_statet:13d}"
                  f"{n_bounce:10d}{cond:13.3e}")

    print(f"\n  rank(J_loop) = |A|:                          {ok_exact}/{tot}")
    print(f"  righe attive di J_per a rango pieno:         {ok_active}/{tot}")
    print(f"  conteggio sullo stato corrente, soglia 0:    {ok_state0}/{tot}")
    print(f"  conteggio sullo stato corrente, soglia 1e-12:{ok_state_tol}/{tot}")
    print(f"  celle di 'rimbalzo' trovate in totale:       {n_bounce_tot}")
    if np.isfinite(min_cond):
        print(f"  minimo sigma_min/sigma_max sulle A non vuote: {min_cond:.3e}")
    if n_bounce_tot == 0:
        print("\n  Nessun rimbalzo osservato: su questo modello la formulazione")
        print("  basata sullo stato corrente, se valutata con soglia esatta, coincide con |A|.")
    return dict(n=tot, ok_exact=ok_exact, ok_active=ok_active,
                ok_state0=ok_state0, ok_state_tol=ok_state_tol,
                n_bounce=n_bounce_tot, min_active_condition=min_cond)



def run(cells=None, Rr=4, G=0.05, T=5000, seeds=(0, 1, 2),
        row_scale_cells=None, row_scale_n=100):
    """Capacita' del canale di retroazione. """
    if cells is None:
        cells = [(W, B) for W in R.PAPER_W for B in R.PAPER_B]
    row_scale_cells = set(row_scale_cells or [])

    print(f"\n{'='*118}")
    print(f"  CAPACITA' DEL CANALE DI RETROAZIONE   (R={Rr}, G={G}, T={T})")
    print(f"{'='*118}")
    print("  dir90 = direzioni che portano il 90% dell'energia dello spettro.")
    print("  (b) osservabile = maschera d'azione: e' il campo di cui si misura")
    print("      la dimensione e su cui e' definito l'RFC.")
    print("  (c) percepito   = maschera di percezione: e' il campo che guida Gp.")
    print("      Sono lo stesso campo letto a due raggi diversi, e il canale")
    print("      di retroazione passa per (c), non per (b).")
    print("  anello = diag(dv/dGp) . (dGp/dv), il guadagno di retroazione vero.")
    print("  dir99 = direzioni che portano il 99% dell'energia.")
    print("  dir90*  = stessa quantita' con le righe riscalate per la std per")
    print("      cella della nuvola: controllo di convenzione. Le stime di")
    print("      dimensione sono sui dati grezzi, quindi il confronto")
    print("      principale usa dir90 non riscalato.\n")
    print(f"  {'W':>7}{'B':>9}{'Gp sat':>9}"
          f"{'dir90 (b)':>11}{'dir90 (c)':>11}{'dir90 anello':>14}{'dir99 anello':>14}"
          f"{'dir90* anello':>15}{'rango anello':>14}{'|J| anello':>13}")

    rows = []
    for (W, B) in cells:
        rs = None
        if (W, B) in row_scale_cells:
            rs = cloud_row_scale(W, B, Rr, G, T, n_points=row_scale_n)

        d90b, d90c, d90l, d99l, d90ls, rkl, smx, fss = [], [], [], [], [], [], [], []
        for sd in seeds:
            net, v, Gp = state_at(W, B, Rr, G, T, seed=sd)
            J = jacobians_at(net, v, Gp)
            sb = spectrum_stats(J['obs'])
            sc = spectrum_stats(J['per'])
            sl = spectrum_stats(J['loop'])
            d90b.append(sb['dir90']); d90c.append(sc['dir90'])
            d90l.append(sl['dir90']); d99l.append(sl['dir99']); rkl.append(sl['rank_num'])
            smx.append(sl['s_max']); fss.append(J['frac_sat'])
            if rs is not None:
                d90ls.append(spectrum_stats(J['loop'], row_scale=rs)['dir90'])

        rec = dict(W=W, B=B, frac_sat=float(np.median(fss)),
                   dir90_obs=float(np.median(d90b)),
                   dir90_per=float(np.median(d90c)),
                   dir90_loop=float(np.median(d90l)),
                   dir99_loop=float(np.median(d99l)),
                   dir90_loop_scaled=(float(np.median(d90ls)) if d90ls
                                      else float('nan')),
                   rank_loop=float(np.median(rkl)),
                   s_max_loop=float(np.median(smx)))
        rows.append(rec)
        sc_txt = (f"{rec['dir90_loop_scaled']:15.0f}"
                  if d90ls else f"{'—':>15}")
        print(f"  {W:7.0f}{B:9.4f}{100*rec['frac_sat']:8.1f}%"
              f"{rec['dir90_obs']:11.0f}{rec['dir90_per']:11.0f}"
              f"{rec['dir90_loop']:14.0f}{rec['dir99_loop']:14.0f}{sc_txt}"
              f"{rec['rank_loop']:14.0f}{rec['s_max_loop']:13.3e}")

    return rows


'''
if __name__ == '__main__':
    # row_scale solo sulle celle citate nel testo: costa 100 traiettorie ognuna
    rows = run(row_scale_cells=[(1000.0, 5e-4), (10.0, 5e-4), (200.0, 5e-4)])
    verify_rank_identity()
'''
if __name__ == '__main__':
    rows = run(row_scale_cells=[(W, 5e-4) for W in R.PAPER_W])
    verify_rank_identity()
    
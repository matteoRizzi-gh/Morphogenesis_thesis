"""
campo_max.py
Maggiorazione del campo percepito valida su TUTTO lo spazio degli stati.

|E_f| <= kappa * C * max|v| * sum_{c in A_f} 1/rho_fc   (disuguaglianza
triangolare sulla somma vettoriale, con |v| <= max(|E_p|, |E_d|)), poi media
sui quattro vertici di ciascuna cellula. Se B supera la maggiorazione, il
drive W*tanh((B - e)/2) e' positivo in ogni stato: ogni conduttanza cresce
di almeno dt*D_min per passo e satura in al piu' 2/(dt*D_min) passi.
"""
import numpy as np, torch
from dynamics_t_sparse_gh import (BioelectricLattice, C, E_p, E_d, k_e, e_r,
                                  FIELD_STRENGTH, EPSILON, DT)
import loop_gain_independence as LG

net = BioelectricLattice(rows=11, cols=11, G=0.05, R=4, W=10.0, B=0.02)
fc = FIELD_STRENGTH * (k_e / e_r)
coef = torch.sqrt(net.endo_coeff_x**2 + net.endo_coeff_y**2)
vmax = max(abs(E_p), abs(E_d))
Eb = torch.zeros(net.N_f, dtype=torch.float64).scatter_add_(
    0, net.endo_f_idx, coef) * fc * C * vmax
Eb = torch.sqrt(Eb**2 + EPSILON)
eb = torch.zeros(net.N, dtype=torch.float64).scatter_add_(
    0, net.perc_cell_idx, Eb.index_select(0, net.perc_field_idx)) / net.perc_n
emax = float(eb.max())
print(f"maggiorazione del campo percepito: {emax:.3e} V")
for B in (0.002, 0.005, 0.01, 0.02):
    if B <= emax:
        print(f"  B={B}: non garantito"); continue
    for W in (10, 100, 200, 500, 1000):
        Dmin = W * np.tanh((B - emax) / 2)
        print(f"  B={B} W={W}: drive minimo {Dmin:.4f}, "
              f"saturazione entro {2/(DT*Dmin):.0f} passi da qualunque stato")

# -----------------------------------------------------------------------------
# Esistenza dello stato collassato: con v uniforme a v* e g = 2 ovunque, lo
# stato e' un punto fisso se e solo se il drive e' positivo in ogni cellula,
# cioe' se B supera il massimo del campo percepito in quello stato.
# -----------------------------------------------------------------------------

v_star = -0.052996376801
e_star = LG.field_perceived(torch.full((net.N,), v_star, dtype=torch.float64), net)
print(f"\ncampo percepito nello stato collassato: max {float(e_star.max()):.4e} V, "
      f"medio {float(e_star.mean()):.4e} V")
for B in (5e-4, 2e-3, 5e-3, 0.01, 0.02):
    n_neg = int((e_star > B).sum())
    print(f"  B={B}: celle con drive negativo {n_neg:3d} -> "
          f"{'NON e' if n_neg else 'e'}' un punto fisso")

"""
verifica_criterio_collasso.py
=============================
Due verifiche del criterio di collasso adottato in Rfc_and_ID.is_collapsed.

(1) La classificazione non dipende dalla soglia.
(2) Le celle di transizione sono lente, non un regime intermedio.
"""

import numpy as np
import torch
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist

import Rfc_and_ID as R
from grid_and_gp import make_cloud

EPS_NUM = 1e-15          # distanza RMS per cella sotto cui due stati sono
                         # coincidenti a meno dell'arrotondamento in doppia
                         # precisione (Vmem ~ 5e-2 V, eps macchina ~1e-18)
TRANSIZIONE = [(500.0, 0.002), (700.0, 0.002)]
TEMPI = (5000, 10000, 20000)

'''
from fixed_point import analyze_cell


r = analyze_cell(1000.0, 5e-4, T_max=1_000_000, verbose=True)
print(f"\nconverge: {r['converged']}   iterazioni: {r['iters']}")
print(f"Gp satura: {100*r['frac_gp_saturated']:.1f}%")
print(f"ultimo spostamento per passo: {r['last_step']:.3e}  (tol 1e-14)")
'''

import torch, Rfc_and_ID as R
tsat = {(10,0.005):11552, (10,0.01):4628, (10,0.02):2122,
        (100,0.002):27299, (100,0.005):1130, (100,0.01):454, (100,0.02):210,
        (200,0.002):13448, (200,0.005):555, (200,0.01):224, (200,0.02):104,
        (500,0.002):5144, (500,0.005):213, (500,0.01):87, (500,0.02):41,
        (1000,0.002):2342, (1000,0.005):99, (1000,0.01):42, (1000,0.02):21}
def trova(d, k):
    if isinstance(d, dict):
        if k in d: return d[k]
        for v in d.values():
            x = trova(v, k)
            if x is not None: return x
for r in torch.load(R.out_path('grid_gp.pt'), weights_only=False):
    key = (int(r['W']), round(float(r['B']), 4))
    if key in tsat:
        print(key, f"{tsat[key] / trova(r, 'pred_t_sat_all'):.3f}")


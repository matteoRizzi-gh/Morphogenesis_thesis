import numpy as np, torch
from dynamics_t_sparse_gh import BioelectricLattice, G_r

def passo_inversione(W, B=5e-4, seed=None, T=1000):
    net = BioelectricLattice(rows=11, cols=11, G=0.05, R=4, W=W, B=B)
    net.reset()
    if seed is None:
        net.Gp = torch.full((net.N,), G_r, dtype=torch.float64)
    else:
        g = torch.Generator().manual_seed(seed)
        net.Gp = torch.rand(net.N, generator=g, dtype=torch.float64) * 2.0 * G_r
    with torch.no_grad():
        for t in range(1, T + 1):
            eV, _ = net.step(v_left_22=None)
            e = torch.zeros(net.N, dtype=torch.float64).scatter_add_(
                0, net.perc_cell_idx, eV.index_select(0, net.perc_field_idx)) / net.perc_n
            if float(e.mean()) > B:
                return t
    return -1

for W in (10, 100, 200, 500, 1000):
    het = np.array([passo_inversione(W, seed=i) for i in range(30)])
    print(f"W={W:5d}  omogenea {passo_inversione(W):4d}   "
          f"eterogenee: mediana {np.median(het):.0f}, min {het.min()}, max {het.max()}")



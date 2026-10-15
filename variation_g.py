import numpy as np
from dynamics_t_sparse_gh import BioelectricLattice
from fixed_point import find_fixed_point, stability
import Rfc_and_ID as R

Gs = (0.0, 0.05, 0.2, 0.5, 1.0)
combinazioni = [(W, B) for W in R.PAPER_W for B in R.PAPER_B if B >= 0.002]

for G in Gs:
    rhos, vs, stds, gmin, gmax = [], [], [], [], []
    for W, B in combinazioni:
        net = BioelectricLattice(rows=11, cols=11, G=G, R=4, W=W, B=B)
        fp = find_fixed_point(net, T_max=300000)
        if fp['converged']:
            rhos.append(stability(net, fp['v'], fp['Gp'])['spectral_radius'])
            vs.append(float(fp['v'].mean()))
            stds.append(float(fp['v'].std()))
            g = fp['Gp'] / 1e-9          # G_r = 1e-9 S; se Gp è già normalizzato, togli la divisione
            gmin.append(float(g.min()))
            gmax.append(float(g.max()))
    print(f"G={G}: convergono {len(rhos)}/{len(combinazioni)}, "
          f"rho in [{min(rhos):.9f}, {max(rhos):.9f}], "
          f"v medio in [{1e3*min(vs):.6f}, {1e3*max(vs):.6f}] mV, "
          f"std spaziale max {1e3*max(stds):.2e} mV, "
          f"g in [{min(gmin):.6f}, {max(gmax):.6f}]")

    
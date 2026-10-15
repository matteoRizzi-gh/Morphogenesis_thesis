"""Diagramma di biforcazione della singola cellula: equilibri di v al variare di g = Gp/Gr.
Costanti identiche a dynamics_t_sparse_gh.py. Nessuna dipendenza dal resto del codice."""
import numpy as np
from scipy.optimize import brentq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

Ep, Ed, Gr, Gd = -55e-3, -5e-3, 1e-9, 1.5e-9
z, v_th, v_T = 3.0, -27e-3, 27e-3
C, DT = 0.1e-9, 0.01

def I_ion(v, g):
    a = z * (v - v_th) / v_T
    return -g * Gr * (v - Ep) / (1 + np.exp(a)) - Gd * (v - Ed) / (1 + np.exp(-a))

def dI(v, g, h=1e-7):
    return (I_ion(v + h, g) - I_ion(v - h, g)) / (2 * h)

def equilibri(g, vmin=-0.07, vmax=0.0, n=70001):
    vs = np.linspace(vmin, vmax, n); f = I_ion(vs, g)
    idx = np.where(f[:-1] * f[1:] < 0)[0]
    return [brentq(I_ion, vs[i], vs[i + 1], args=(g,)) for i in idx]

gs = np.linspace(0, 2, 2001)
stab_g, stab_v, inst_g, inst_v = [], [], [], []
for g in gs:
    for v in equilibri(g):
        (stab_g if dI(v, g) < 0 else inst_g).append(g)
        (stab_v if dI(v, g) < 0 else inst_v).append(v)

if __name__ == "__main__":
    # verifiche numeriche riportate nel testo
    for g in (0.0, 1.0, 2.0):
        print(f"g = {g}: " + ", ".join(
            f"{v*1e3:.4f} mV ({'stabile' if dI(v, g) < 0 else 'instabile'}, "
            f"moltiplicatore Eulero {1 + DT / C * dI(v, g):.3f})" for v in equilibri(g)))
    bist = [g for g in gs if len(equilibri(g)) == 3]
    print(f"finestra bistabile: g in [{min(bist):.3f}, {max(bist):.3f}]")
    for gb in (min(bist), max(bist)):
        ins = [v for v in equilibri(gb) if dI(v, gb) > 0]
        print(f"  ai bordi della finestra (g={gb:.3f}) il ramo instabile sta a {ins[0]*1e3:.1f} mV")

    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.plot(stab_g, np.array(stab_v) * 1e3, ".", ms=1.5, color="k", label="stabile")
    ax.plot(inst_g, np.array(inst_v) * 1e3, ".", ms=1.5, color="0.6", label="instabile")
    ax.axhline(-9.2, ls=":", lw=0.8, color="C3")
    ax.plot([1.0], [-9.2], "o", color="C3", ms=5, label="condizione iniziale")
    ax.set_xlabel(r"$g = G_p/G_r$"); ax.set_ylabel(r"$v^\ast$ (mV)")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], color="k", lw=2, label="stabile"),
                       Line2D([], [], color="0.6", lw=2, label="instabile"),
                       Line2D([], [], color="C3", marker="o", ls="", label="condizione iniziale")],
              frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig("biforcazione.pdf")

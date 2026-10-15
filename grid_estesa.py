"""
grid_estesa.py
==============
Esperimento SUPPLEMENTARE, separato dalla replica.

"""

import os
import math
import numpy as np
import torch

import Rfc_and_ID as R
from grid_and_gp import grid

# B fisso: l'unico valore in cui la frazione attiva copre un intervallo ampio.
B_FIXED = 0.002

# W aggiunti, per infittire in modo approssimativamente logaritmico fra i
# valori gia' presenti in PAPER_W (10, 100, 200, 500, 1000).
W_EXTRA = (30.0, 50.0, 150.0, 300.0, 700.0)

BASE_PT = 'grid_gp.pt'
EXTRA_PT = 'grid_gp_extraW.pt'
MERGED_PT = 'grid_gp_estesoW.pt'


# ═══════════════════════════════════════════════════════════════════════════
#  Calcolo e merge
# ═══════════════════════════════════════════════════════════════════════════

def compute_extra(n_points=300, T=5000, Rr=4, G=0.05, n_workers=None):
    """Calcola le sole celle nuove: len(W_EXTRA) celle a B = B_FIXED.

    n_points e T devono coincidere con quelli di grid_gp.pt, altrimenti il
    merge fallisce: e' un controllo voluto, non un ostacolo.
    """
    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 2) // 2)
    print(f"Celle supplementari: {len(W_EXTRA)} W a B={B_FIXED:g} "
          f"= {len(W_EXTRA)} celle ({n_points} punti, T={T})")
    print(f"  W nuovi: {[f'{w:g}' for w in W_EXTRA]}")
    return grid(Ws=W_EXTRA, Bs=(B_FIXED,), Rr=Rr, G=G,
                n_points=n_points, T=T, n_workers=n_workers,
                save_as=EXTRA_PT)


def merge(base_pt=BASE_PT, extra_pt=EXTRA_PT, out_pt=MERGED_PT):
    """Fonde la griglia del paper con le celle supplementari in W. """
    a = torch.load(R.out_path(base_pt), weights_only=False)
    b = torch.load(R.out_path(extra_pt), weights_only=False)

    for key in ('n_points', 'T', 'R', 'G'):
        va = {r.get(key) for r in a}
        vb = {r.get(key) for r in b}
        if va != vb:
            raise ValueError(
                f"parametro '{key}' incoerente fra i due file: "
                f"{base_pt} ha {va}, {extra_pt} ha {vb}. "
                f"Le celle non sono confrontabili.")


    for nome, res in ((base_pt, a), (extra_pt, b)):
        manca = [k for k in ('vmem_spread', 'gp')
                 if any(k not in r for r in res)]
        if manca:
            raise ValueError(
                f"{nome}: chiavi mancanti {manca}.")

    merged = sorted(a + b, key=lambda r: (r['W'], r['B']))
    cells = [(r['W'], r['B']) for r in merged]
    if len(set(cells)) != len(cells):
        dup = sorted({c for c in cells if cells.count(c) > 1})
        raise ValueError(f"celle duplicate nel merge: {dup}")

    torch.save(merged, R.out_path(out_pt))

    at_B = [r for r in merged if abs(r['B'] - B_FIXED) < 1e-12]
    dead_B = [r for r in at_B if R.is_collapsed(r['vmem_spread'])]
    print(f"  {len(a)} + {len(b)} = {len(merged)} celle "
          f"-> {R.out_path(out_pt)}")
    print(f"  celle a B={B_FIXED:g}: {len(at_B)} "
          f"(erano {sum(1 for r in a if abs(r['B'] - B_FIXED) < 1e-12)}), "
          f"di cui {len(dead_B)} collassate")
    if dead_B:
        det = ", ".join(f"W={r['W']:g} (spread {r['vmem_spread']:.1e})"
                        for r in dead_B)
        print(f"    collassate: {det}")
    n_live_B = len(at_B) - len(dead_B)
    if 1 <= n_live_B <= 20:
        print(f"  -> il gruppo del test avra' n = {n_live_B}, "
              f"p minimo {2.0 / math.factorial(n_live_B):.1e}")
    else:
        print(f"  -> il gruppo del test avra' n = {n_live_B}")
    return merged


# ═══════════════════════════════════════════════════════════════════════════
#  Test
# ═══════════════════════════════════════════════════════════════════════════

def report(merged, tol=1e-6):
    """Il test di correlazione a B fisso, con la numerosita' nuova."""
    import analyze_grid as AG

    rows = AG.to_table(merged, tol=tol)
    live, _ = AG.print_table(rows)

    print(f"\n{'='*92}")
    print(f"  TEST DI CORRELAZIONE, GRIGLIA INFITTITA IN W A B={B_FIXED:g}")
    print(f"{'='*92}")
    print(f"  ATTENZIONE: questa griglia NON e' quella del paper. I valori")
    print(f"  W = {[f'{w:g}' for w in W_EXTRA]} sono stati aggiunti apposta")
    print(f"  per portare il gruppo B={B_FIXED:g} a una numerosita' in cui il test possa scendere sotto 0.05.")
    stats = AG.correlation(live)

    print(f"\n{'='*92}\n  CONFRONTO: GRIGLIA DEL PAPER CONTRO GRIGLIA INFITTITA"
          f"\n{'='*92}")
    print(f"  Entrambi i lati usano lo stesso criterio di collasso")
    print(f"  (R.is_collapsed, spread < {R.SPREAD_COLLAPSED:g} V): cambia solo")

    base = torch.load(R.out_path(BASE_PT), weights_only=False)
    live0, _ = AG.print_table(AG.to_table(base, tol=tol))
    stats0 = AG.correlation(live0)

    def fmt(s, key):
        if s is None:
            return '-'
        if key == 'p_exact':
            if s.get('degenerate'):
                return 'degenere'
            if not np.isfinite(s['p_exact']):
                return '-'
            return f"{s['p_exact']:.5f}"
        if key == 'spearman':
            return f"{s['spearman']:+.3f}"
        return str(s[key])

    print(f"\n  {'gruppo':>16}{'n paper':>10}{'p paper':>11}"
          f"{'n esteso':>11}{'p esteso':>11}{'rho esteso':>13}")
    for k in sorted(set(stats) | set(stats0), key=str):
        s0, s1 = stats0.get(k), stats.get(k)
        print(f"  {str(k):>16}{fmt(s0,'n'):>10}{fmt(s0,'p_exact'):>11}"
              f"{fmt(s1,'n'):>11}{fmt(s1,'p_exact'):>11}"
              f"{fmt(s1,'spearman'):>13}")

    # ------------------------------------------------------------------
    # Verdetto sul unico gruppo
    # ------------------------------------------------------------------
    s = stats.get(B_FIXED)
    print()
    if s is None:
        print(f"  Il gruppo B={B_FIXED:g} non compare fra i risultati:")
        print(f"  controlla che le celle nuove non siano tutte collassate.")
    elif s.get('degenerate'):
        print(f"  Il gruppo B={B_FIXED:g} da' rho esattamente 0: il test e'")
        print(f"  degenere e la relazione non c'e'.")
    elif np.isfinite(s['p_exact']) and s['p_exact'] < 0.05:
        print(f"  Il gruppo B={B_FIXED:g} ha n={s['n']}, "
              f"rho_S = {s['spearman']:+.3f}, p = {s['p_exact']:.5f} < 0.05:")
        print(f"  il test e' concludente. Va riportato dichiarando che la")
        print(f"  griglia in W e' supplementare e che il disegno e' a B fisso.")
        if s['n'] < (len(R.PAPER_W) + len(W_EXTRA)):
            print(f"  Nota: {(len(R.PAPER_W) + len(W_EXTRA)) - s['n']} celle a")
            print(f"  questo B sono classificate collassate e non entrano nel test")
    else:
        print(f"  Il gruppo B={B_FIXED:g} ha n={s['n']}, "
              f"rho_S = {s['spearman']:+.3f}, p = {s['p_exact']:.5f}:")
        print(f"  il test NON e' concludente nemmeno con la numerosità aumentata")
    return stats, stats0


# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    compute_extra(n_points=300, T=5000)
    merged = merge()
    report(merged)

    
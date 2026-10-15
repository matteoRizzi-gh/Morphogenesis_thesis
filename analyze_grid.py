"""
analyze_grid.py
===============
Legge output_rfc/grid_gp.pt e produce (a) la relazione fra gradi di liberta'
attivi in Gp e dimensione della nuvola, (b) i grafici per mostrarla.


L'IPOTESI DA TESTARE
--------------------
Gp evolve come dp = -Gp + (2*sigma(-e + B) - 1)*W ed e' ritagliata in
[0, 2*G_r]. Una cella la cui Gp resta incollata al bordo non e' piu' una
variabile: e' una costante. Le condizioni iniziali sono proprio su Gp,
quindi dove Gp satura l'informazione iniziale viene cancellata.

L'ipotesi piu' forte, che questo file testa, e' quantitativa: la dimensione
della nuvola dovrebbe seguire la FRAZIONE di gradi di liberta' attivi, non
solo il fatto che ce ne siano. Attenzione: la frazione non satura e' un
conteggio esatto, l'ID e' una stima con bias; quindi si testa una relazione
fra una quantita' certa e una incerta, e vanno riportate come tali.

"""

import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import math
from itertools import permutations
from scipy.stats import spearmanr

import Rfc_and_ID as R

FIGDIR = 'figures'
GRID_PT = os.path.join(R.OUTPUT_DIR, 'grid_gp.pt')

# Tetto di riferimento se non e' misurabile dallo spettro della jacobiana.
CEILING_FALLBACK = 10.0


def figpath(name):
    os.makedirs(FIGDIR, exist_ok=True)
    return os.path.join(FIGDIR, name)


def measured_ceiling(Rr=4, verbose=True):
    """dir90 della jacobiana di Phi_obs, misurato """
    try:
        from refute_rfc import jacobian_spectrum
        c = float(jacobian_spectrum(Rr=Rr)['n90'])
        if verbose:
            print(f"  tetto misurato da jacobian_spectrum (R={Rr}): "
                  f"dir90 = {c:.0f}")
        return c
    except Exception as e:
        if verbose:
            print(f"  tetto non misurabile ({type(e).__name__}: {e}), "
                  f"uso il valore di ripiego {CEILING_FALLBACK:g}")
        return CEILING_FALLBACK


# ═══════════════════════════════════════════════════════════════════════
#  Lettura e tabella
# ═══════════════════════════════════════════════════════════════════════

def load(path=GRID_PT):
    res = torch.load(path, weights_only=False)
    res.sort(key=lambda r: (r['W'], r['B']))
    return res


def to_table(res, tol=1e-6):
    """Una riga per cella. 'active' = frazione di coppie (punto, cella) in
    cui Gp NON e' al bordo del clamp"""
    rows = []
    for r in res:
        g = r['gp']
        counts = r['h0']['counts']
        key = min(counts, key=lambda t: abs(t - tol))
        rows.append(dict(
            W=r['W'], B=r['B'],
            active=g['frac_interior'],
            sat_hi=g['frac_max'], sat_lo=g['frac_zero'],
            gp_mean=g['mean_over_Gr'],
            h0=counts[key],
            id_vmem=r['id_vmem'], id_field=r['id_field'],
            spread=r['vmem_spread'],
            n_points=r.get('n_points'), T=r.get('T'),
            collapsed=R.is_collapsed(r['vmem_spread']),
        ))
    return rows


def print_table(rows):
    """Tabella per cella e verifica della corrispondenza saturazione/collasso. """
    n_pts = {r['n_points'] for r in rows if r['n_points']}
    Ts = {r['T'] for r in rows if r['T']}
    hdr = "  GRADI DI LIBERTA' ATTIVI E DIMENSIONE"
    if n_pts and Ts:
        hdr += f"   ({max(n_pts)} punti, T={max(Ts)})"
    print(f"\n{'='*88}\n{hdr}\n{'='*88}")
    print(f"  {'W':>7}{'B':>9}{'Gp attiva':>11}{'Gp media':>10}"
          f"{'H0':>6}{'ID Vmem':>9}{'ID campo':>10}{'spread':>11}")
    for r in rows:
        mark = "   collassata" if r['collapsed'] else ""
        print(f"  {r['W']:7.0f}{r['B']:9.4f}{100*r['active']:10.1f}%"
              f"{r['gp_mean']:10.3f}{r['h0']:6d}{r['id_vmem']:9.2f}"
              f"{r['id_field']:10.2f}{r['spread']:11.2e}{mark}")

    live = [r for r in rows if not r['collapsed']]
    dead = [r for r in rows if r['collapsed']]

    print(f"\n  criterio di collasso: spread < {R.SPREAD_COLLAPSED:g} V "
          f"(distanza RMS per cella)")
    print(f"  collassate: {len(dead)}   non collassate: {len(live)}")

    if not (dead and live):
        return live, dead

    sp_d = np.array([r['spread'] for r in dead])
    sp_l = np.array([r['spread'] for r in live])
    act_d = np.array([r['active'] for r in dead])
    act_l = np.array([r['active'] for r in live])

    print(f"\n  spread:  collassate [{sp_d.min():.2e}, {sp_d.max():.2e}]   "
          f"non collassate [{sp_l.min():.2e}, {sp_l.max():.2e}]")
    if sp_d.max() > 0:
        print(f"           margine fra i due gruppi: "
              f"{sp_l.min()/sp_d.max():.0f}x")

    print(f"\n  Gp attiva:  collassate [{100*act_d.min():.1f}%, "
          f"{100*act_d.max():.1f}%]   "
          f"non collassate [{100*act_l.min():.1f}%, {100*act_l.max():.1f}%]")

    if act_d.max() < act_l.min():
        print(f"  -> intervalli DISGIUNTI (margine "
              f"{act_l.min() - act_d.max():.3f} in frazione): la")
        print(f"     corrispondenza fra saturazione di Gp e collasso della")
        print(f"     nuvola regge su tutte e {len(rows)} le celle.")
    else:
        over = [(r['W'], r['B'], round(100*r['active'], 1))
                for r in dead if r['active'] >= act_l.min()]
        print(f"  -> intervalli SOVRAPPOSTI: {len(over)} cella/e collassata/e")
        print(f"     ha piu' gradi di liberta' attivi della cella viva piu'")
        print(f"     contratta: {over}")
        print(f"     La corrispondenza non regge come enunciato: va rivista.")

    n_full = sum(1 for r in dead if r['active'] < 1e-9)
    if n_full < len(dead):
        part = [(r['W'], r['B'], round(100*r['active'], 1))
                for r in dead if r['active'] >= 1e-9]
        print(f"\n  {len(dead)-n_full} cella/e collassata/e non ha Gp satura al")
        print(f"  100%: {part}")
        print(f"  Sono celle di transizione, colte a T finito mentre l'ultima")
        print(f"  cella sta ancora saturando. La corrispondenza saturazione")
        print(f"  completa <=> collasso e' ASINTOTICA, non a T fissato.")
    else:
        print(f"\n  tutte le {len(dead)} celle collassate hanno Gp satura al 100%")

    return live, dead

# ═══════════════════════════════════════════════════════════════════════
#  Correlazione
# ═══════════════════════════════════════════════════════════════════════

def correlation(live, n_mc=20000, seed=0):
    """ID della nuvola contro frazione di gradi di liberta' attivi, per regime. """
    def _stats(sel, max_exact=8):
        n = len(sel)
        if n < 3:
            return None
        a = np.array([r['active'] for r in sel], dtype=float)
        d = np.array([r['id_vmem'] for r in sel], dtype=float)

        rho_s = float(spearmanr(a, d).statistic)
        p, exact, degenerate = float('nan'), False, False
        if np.isfinite(rho_s):
            obs = abs(rho_s)
            if obs <= 1e-12:
                p, degenerate = 1.0, True
            elif n <= max_exact:
                hits = sum(1 for pm in permutations(range(n))
                           if abs(spearmanr(a, d[list(pm)]).statistic)
                           >= obs - 1e-12)
                p, exact = hits / math.factorial(n), True
            else:
                rng = np.random.default_rng(seed)
                hits = sum(1 for _ in range(n_mc)
                           if abs(spearmanr(a, rng.permutation(d)).statistic)
                           >= obs - 1e-12)
                p = hits / n_mc

        sl, ic = np.polyfit(a, d, 1)
        p_floor = 2.0 / math.factorial(n) if n <= max_exact else 1.0 / n_mc
        return dict(n=n,
                    pearson=float(np.corrcoef(a, d)[0, 1]),
                    spearman=rho_s,
                    p_exact=p, exact=exact, degenerate=degenerate,
                    p_floor=float(p_floor),
                    slope=float(sl), intercept=float(ic))

    def _pstr(s):
        if s['degenerate']:
            return 'degenere'
        if not np.isfinite(s['p_exact']):
            return '—'
        if s['p_exact'] <= s['p_floor'] * 1.001:
            return f"{s['p_exact']:.2e} (min)"
        return f"{s['p_exact']:.4f}" + ('' if s['exact'] else ' MC')

    print(f"\n{'='*92}\n  ID CONTRO FRAZIONE ATTIVA, PER REGIME\n{'='*92}")
    print(f"  {'gruppo':>22}{'n':>4}{'Pearson':>10}{'Spearman':>11}"
          f"{'p':>13}{'p minimo':>11}{'pendenza':>11}")
    out = {}
    glob = _stats(live)
    if glob:
        out['tutte'] = glob
        print(f"  {'tutte insieme':>22}{glob['n']:>4}{glob['pearson']:>+10.3f}"
              f"{glob['spearman']:>+11.3f}{_pstr(glob):>13}"
              f"{glob['p_floor']:>11.4f}{glob['slope']:>11.1f}")
    for B in sorted({r['B'] for r in live}):
        sel = [r for r in live if r['B'] == B]
        s = _stats(sel)
        if s:
            out[B] = s
            print(f"  {f'B = {B:g}':>22}{s['n']:>4}{s['pearson']:>+10.3f}"
                  f"{s['spearman']:>+11.3f}{_pstr(s):>13}"
                  f"{s['p_floor']:>11.4f}{s['slope']:>11.1f}")
        else:
            print(f"  {f'B = {B:g}':>22}{len(sel):>4}"
                  f"{'—':>10}{'—':>11}{'—':>13}{'—':>11}{'—':>11}"
                  "   (n < 3)")

    print("\n  'p minimo' = il piu' piccolo p ottenibile a quella numerosita':")
    print("  un p uguale al minimo significa 'il piu' significativo possibile")
    ns = {k: s['n'] for k, s in out.items() if k != 'tutte'}
    print(f"\n  Lettura. Dove il clamp NON morde la frazione attiva non ordina")
    print(f"  l'ID: quella e' contrazione dinamica di altra origine. Dove il")
    print(f"  clamp stringe, l'ordinamento e' monotono. Numerosita' dei gruppi:")
    print(f"  {', '.join(f'B={k:g} n={v}' for k, v in sorted(ns.items()))}.")
    print(f"  La frazione attiva e' un conteggio esatto, l'ID una stima con bias.")
    return out


def field_ceiling(live, ceiling=None):
    """ID(campo) contro il tetto imposto dalla mappa di misura. """
    if ceiling is None:
        ceiling = measured_ceiling()

    print(f"\n{'='*78}\n  IL TETTO DEL CAMPO  (tetto = {ceiling:g})\n{'='*78}")
    print(f"  {'ID Vmem':>10}{'ID campo':>10}{'min(tetto,Vmem)':>18}"
          f"{'scarto':>9}{'':>4}")
    viol = []
    for r in sorted(live, key=lambda x: -x['id_vmem']):
        pred = min(ceiling, r['id_vmem'])
        gap = r['id_field'] - pred
        if gap > 0:
            viol.append((r, gap))
        print(f"  {r['id_vmem']:10.2f}{r['id_field']:10.2f}{pred:18.2f}"
              f"{gap:+9.2f}{('  >' if gap > 0 else ''):>4}")

    above = [r for r in live if r['id_vmem'] > ceiling]
    below = [r for r in live if r['id_vmem'] <= ceiling]
    if above:
        v = [r['id_field'] for r in above]
        print(f"\n  con ID(Vmem) > {ceiling:g}: campo fra {min(v):.2f} e "
              f"{max(v):.2f}  (banda imposta da Phi_obs)")
    if below:
        print(f"  con ID(Vmem) <= {ceiling:g}: il campo SEGUE Vmem verso il basso")
        for r in below:
            print(f"      Vmem {r['id_vmem']:5.2f} -> campo {r['id_field']:5.2f}")
    if viol:
        worst = max(viol, key=lambda t: t[1])
        print(f"\n  il tetto e' superato in {len(viol)}/{len(live)} celle, "
              f"al massimo di {worst[1]:+.2f} "
              f"({100*worst[1]/ceiling:+.0f}% del tetto).")
        print("  Coerente con il fatto che (ii) e' un'euristica e non un")
        print("  teorema: e' una banda, non un limite superiore dimostrato.")
    else:
        print(f"\n  nessuna cella supera il tetto.")
    return dict(ceiling=ceiling, n_violations=len(viol), n_live=len(live))


# ═══════════════════════════════════════════════════════════════════════
#  Grafici
# ═══════════════════════════════════════════════════════════════════════

def _grid_matrix(rows, key):
    Ws = sorted({r['W'] for r in rows})
    Bs = sorted({r['B'] for r in rows})
    M = np.full((len(Ws), len(Bs)), np.nan)
    for r in rows:
        M[Ws.index(r['W']), Bs.index(r['B'])] = r[key]
    return np.array(Ws), np.array(Bs), M



def plot_phase_map_slide(res, eps=1e-4, outdir='figure', fname='fig_mappa_fase.pdf',
                         larghezza_cm=13.0, altezza_cm=4.0):
    """Mappa di fase """
    rows = []
    for r in res:
        counts = r['h0']['counts']
        key = min(counts, key=lambda t: abs(np.log10(t) - np.log10(eps)))
        if abs(np.log10(key) - np.log10(eps)) > 0.01:
            print(f"  ATTENZIONE: scala {eps:g} assente in W={r['W']:g}, "
                  f"B={r['B']:g}, uso {key:g}")
        rows.append(dict(W=r['W'], B=r['B'],
                         active=r['gp']['frac_interior'], h0=counts[key]))
    Ws, Bs, SAT = _grid_matrix(rows, 'active')
    _, _, H0 = _grid_matrix(rows, 'h0')
    esp = int(round(np.log10(eps)))

    stile = {'font.size': 7, 'axes.titlesize': 7, 'axes.labelsize': 7,
             'xtick.labelsize': 6, 'ytick.labelsize': 6}
    with plt.rc_context(stile):          # non cambia lo stile delle altre figure
        fig, axes = plt.subplots(1, 2, figsize=(larghezza_cm / 2.54, altezza_cm / 2.54),
                                 constrained_layout=True)
        pannelli = (
            (axes[0], 100 * SAT, 'conduttanze interne (%)', 'viridis', None),
            (axes[1], H0, rf'stati distinti ($H_0$ a $\varepsilon=10^{{{esp}}}$ V)',
             'magma', LogNorm(vmin=1, vmax=max(2, np.nanmax(H0)))))
        for ax, M, title, cmap, norm in pannelli:
            im = ax.imshow(M, origin='lower', aspect='auto', cmap=cmap, norm=norm)
            ax.set_xticks(range(len(Bs))); ax.set_xticklabels([f'{b:g}' for b in Bs])
            ax.set_yticks(range(len(Ws))); ax.set_yticklabels([f'{w:g}' for w in Ws])
            ax.set_xlabel('$B$', labelpad=1); ax.set_ylabel('$W$', labelpad=1)
            ax.set_title(title, pad=3)
            ax.tick_params(length=2, pad=1.5)
            for i in range(M.shape[0]):
                for j in range(M.shape[1]):
                    v = M[i, j]
                    if np.isfinite(v):
                        txt = f'{v:.1f}' if 0 < v < 1 else f'{v:.0f}'
                        ax.text(j, i, txt, ha='center', va='center', fontsize=6,
                                color='white' if v < np.nanmax(M) * 0.6 else 'black')
            cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.06)
            cb.ax.tick_params(labelsize=6, length=2, pad=1)
        os.makedirs(outdir, exist_ok=True)
        out = os.path.join(outdir, fname)
        fig.savefig(out)
        plt.close(fig)
    print(f"  {out}")

def plot_active_vs_id(rows, stats=None, title_suffix='',
                      fname='02_gradi_liberta_vs_dimensione.png'):
    """calcolo gradi di libertà"""
    live = [r for r in rows if not r['collapsed']]
    dead = [r for r in rows if r['collapsed']]
    Bs = sorted({r['B'] for r in live})

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5), constrained_layout=True,
                             sharey=True)
    groups = [
        (axes[0], [b for b in Bs if b <= 0.0005],
         'Clamp non vincolante (B = 0.0005)\ncontrazione dinamica pura'),
        (axes[1], [b for b in Bs if b > 0.0005],
         'Clamp che stringe (B $\\geq$ 0.002)\nl\'ID traccia i gradi di libertà'),
    ]
    colors = dict(zip(Bs, plt.cm.plasma(np.linspace(0.15, 0.85, len(Bs)))))

    for ax, bs, title in groups:
        sel = [r for r in live if r['B'] in bs]
        for B in bs:
            s = [r for r in sel if r['B'] == B]
            if not s:
                continue
            in_fit = (stats and B in stats and stats[B].get('n', 0) >= 3
                      and len([b for b in bs
                               if b in stats and stats[b].get('n', 0) >= 3]) == 1)
            lab_b = f'B = {B:g}' if in_fit else f'B = {B:g} (fuori dal test)'
            ax.scatter([100*r['active'] for r in s], [r['id_vmem'] for r in s],
                       s=110, color=colors[B], edgecolor='k', linewidth=0.6,
                       label=lab_b, zorder=3)
            for r in s:
                ax.annotate(f"W={r['W']:g}", (100*r['active'], r['id_vmem']),
                            textcoords='offset points', xytext=(8, 5), fontsize=8)

        if len(sel) >= 3:

            key = None
            if stats:
                cand = [b for b in bs
                        if b in stats and stats[b].get('n', 0) >= 3]
                if len(cand) == 1:
                    key = cand[0]

            if key is not None:
                st = stats[key]
                fit = [r for r in sel if r['B'] == key]
                a = np.array([r['active'] for r in fit])
                d = np.array([r['id_vmem'] for r in fit])
                rho = st['spearman']
                ntxt = f"n={st['n']}, B={key:g}"
                if st.get('degenerate'):
                    ptxt = 'degenere'
                elif np.isfinite(st.get('p_exact', np.nan)):
                    ptxt = (f"p={st['p_exact']:.4f}"
                            + ('' if st.get('exact', True) else ' MC'))
                else:
                    ptxt = ''
            else:
                a = np.array([r['active'] for r in sel])
                d = np.array([r['id_vmem'] for r in sel])
                rho = float(spearmanr(a, d).statistic)
                ntxt, ptxt = f"n={len(sel)}, pool", ''

            lab = f'Spearman $\\rho$ = {rho:+.2f}  ({ntxt}'
            lab += f', {ptxt})' if ptxt else ')'
            if ptxt == 'degenere':

                ax.plot([], [], ' ', label=lab)
            else:
                sl, ic = np.polyfit(a, d, 1)
                xs = np.linspace(min(a), max(a), 50)
                ax.plot(100*xs, sl*xs + ic, '--', color='0.45', lw=1.3,
                        zorder=1, label=lab)

        if ax is axes[1] and dead:
            ax.scatter([0]*len(dead), [0]*len(dead), s=110, marker='x',
                       color='0.4', linewidth=2,
                       label=f'collassate ({len(dead)})', zorder=3)
        ax.set_xlabel('Gp non satura: gradi di libertà attivi (%)')
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(alpha=0.3)
        ax.set_xlim(-4, 104)

    axes[0].set_ylabel('Dimensione intrinseca della nuvola (Vmem, MLE)')
    fig.savefig(figpath(fname), dpi=150)
    plt.close(fig)
    print(f"  {FIGDIR}/{fname}")




def plot_h0_curve(path=os.path.join(R.OUTPUT_DIR, 'frontier_W500_B0002.pt'),
                  fname='04_h0_senza_scala.png', grid_rows=None):
    """H0 contro la scala nella cella di frontiera."""
    if not os.path.exists(path):
        print(f"  (salto {fname}: manca {path} — lancia grid_and_gp "
              f"con MODE='frontier')")
        return
    d = torch.load(path, weights_only=False)
    W, B = d.get('W'), d.get('B')
    Rr, npts, T = d.get('R'), d.get('n_points'), d.get('T')
    print(f"  frontier: W={W} B={B} R={Rr} n={npts} T={T}")

    if Rr is not None and Rr != 4:
        print(f"  ATTENZIONE: il file e' a R={Rr}, la griglia e' a R=4.")

    c = d['h0']['counts']
    eps = np.array(sorted(c)); n = np.array([c[e] for e in eps])


    if grid_rows is not None and W is not None:
        match = [r for r in grid_rows
                 if abs(r['W'] - W) < 1e-9 and abs(r['B'] - B) < 1e-12]
        if match and match[0]['h0'] != int(np.interp(1e-6, eps, n)):
            print(f"  ATTENZIONE: H0 a eps=1e-6 vale {match[0]['h0']} in "
                  f"grid_gp.pt e ~{int(np.interp(1e-6, eps, n))} qui: "
                  f"i due file vengono da run diversi.")

    sat = d.get('gp', {}).get('frac_max')
    sub = (f"(Gp satura al {100*sat:.1f}%)" if sat is not None else "")


    gap = d['h0'].get('largest_gap')
    gap_eps = d['h0'].get('gap_at')
    spread = d['h0'].get('spread')
    if spread is None:
        spread = d.get('drv', {}).get('vmem_spread')

    if spread is not None and R.is_collapsed(spread):
        dec = (np.log10(spread / gap_eps)
               if (gap_eps and gap_eps > 0 and spread > 0) else float('nan'))
        plateau = (f'nuvola COLLASSATA (spread {spread:.1e} V)')
        if gap is not None and np.isfinite(dec):
            sub = (f'salto {gap:.1f}x a $\\varepsilon$={gap_eps:.0e}, '
                   f'{dec:.0f} ordini sotto lo spread\n' + sub)
    elif gap is not None and gap >= 10:
        plateau = (f'scala privilegiata a $\\varepsilon\\approx$'
                   f'{gap_eps:.0e} (salto {gap:.1f}x)')
    elif gap is not None:
        plateau = f'nessun salto marcato (max {gap:.1f}x)'
    else:
        plateau = 'salto non disponibile nel file'

    fig, ax = plt.subplots(figsize=(6.5, 4.6), constrained_layout=True)
    ax.step(eps, n, where='post', lw=1.8)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('scala $\\varepsilon$ (V, distanza RMS per cella)')
    ax.set_ylabel('componenti connesse $H_0$')
    ax.set_title(f'W={W:g}, B={B:g}: {plateau}\n{sub}', fontsize=11)
    if gap_eps and gap_eps > 0:
        if eps.min() <= gap_eps <= eps.max():
            ax.axvline(gap_eps, ls=':', lw=1.4, color='C3',
                       label=f'salto {gap:.1f}x')
            ax.legend(fontsize=8)
        else:
            ax.text(0.02, 0.06,
                    f'salto {gap:.1f}x a $\\varepsilon$={gap_eps:.0e}, '
                    f'fuori dal grafico',
                    transform=ax.transAxes, fontsize=8, color='C3')
    if spread is not None and eps.min() <= spread <= eps.max():
        ax.axvline(spread, ls='--', lw=1.2, color='0.4',
                   label=f'spread {spread:.1e} V')
        ax.legend(fontsize=8)

    ax.grid(alpha=0.3, which='both')
    fig.savefig(figpath(fname), dpi=150)
    plt.close(fig)
    print(f"  {FIGDIR}/{fname}")


def plot_rfc_refutation(rows_dict=None, fname='05_confutazione_rfc.png'):
    if rows_dict is None:
        path = os.path.join(R.OUTPUT_DIR, 'refute_rfc.pt')
        if not os.path.exists(path):
            print(f"  (salto {fname}: manca {path}, rilancia refute_rfc.py)")
            return
        rr = torch.load(path, weights_only=False)
        data = {k: (v['id_vmem'], v['id_field'], v['rfc']) for k, v in rr.items()}
    else:
        data = rows_dict

    names = list(data)
    idv = [data[k][0] for k in names]
    idf = [data[k][1] for k in names]
    rfc = [data[k][2] for k in names]
    x = np.arange(len(names)); w = 0.35

    ref = 0            # la prima riga e' 'dinamica vera'
    n_other = len(rfc) - 1
    n_higher = sum(1 for i in range(len(rfc)) if i != ref and rfc[i] > rfc[ref])
    if n_other == 0:
        t1 = 'RFC'
    elif n_higher == n_other:
        t1 = ('Tutti i controlli senza dinamica:\n'
              'RFC piu\' alto della dinamica vera')
    elif n_higher > 0:
        t1 = (f'{n_higher} controlli su {n_other} senza dinamica:\n'
              f'RFC piu\' alto della dinamica vera')
    else:
        t1 = ('Controlli senza dinamica:\n'
              'RFC non superiore a quello della dinamica vera')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
    axes[0].bar(x - w/2, idv, w, label='ID Vmem')
    axes[0].bar(x + w/2, idf, w, label='ID campo')
    axes[0].set_xticks(x); axes[0].set_xticklabels(names, fontsize=8, rotation=15)
    axes[0].set_ylabel('Dimensione intrinseca (MLE)')
    axes[0].set_title(f'$\\Phi$ porta tutto a ~{np.mean(idf):.0f}, '
                      'qualunque sia l\'ingresso', fontsize=10)
    axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3, axis='y')

    cols = ['0.6'] * len(names); cols[ref] = 'C0'
    bars = axes[1].bar(x, rfc, color=cols)
    bars[ref].set_edgecolor('k'); bars[ref].set_linewidth(1.5)
    axes[1].set_xticks(x); axes[1].set_xticklabels(names, fontsize=8, rotation=15)
    axes[1].set_ylabel('RFC (%)')
    axes[1].set_title(t1, fontsize=10)
    axes[1].grid(alpha=0.3, axis='y')

    fig.savefig(figpath(fname), dpi=150)
    plt.close(fig)
    print(f"  {FIGDIR}/{fname}")


def plot_frontier_collapse(rows, fname='06_frontiera_BW.png'):
    """La frontiera in funzione del prodotto B*W. """
    bw = np.array([r['B'] * r['W'] for r in rows])
    col = np.array([r['collapsed'] for r in rows])

    # pareggi esatti in B*W con esito opposto: falsificano la separazione
    ties = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            if abs(bw[i] - bw[j]) <= 1e-12 * max(bw[i], bw[j]) \
                    and col[i] != col[j]:
                ties.append((rows[i], rows[j]))

    # separabilita': esiste una soglia che divide i due gruppi?
    if col.any() and (~col).any():
        sep = bw[col].min() > bw[~col].max()
    else:
        sep = True

    tie_cells = {(id(a), id(b)) for a, b in ties}
    tie_ids = {i for pair in tie_cells for i in pair}

    fig, ax = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
    for r in rows:
        c = 'C3' if r['collapsed'] else 'C0'
        ax.scatter(r['B']*r['W'], 100*r['active'], s=110, color=c,
                   edgecolor='k', linewidth=0.6, zorder=3)
        if id(r) in tie_ids:
            ax.annotate(f"{r['W']:g}/{r['B']:g}",
                        (r['B']*r['W'], 100*r['active']),
                        textcoords='offset points', xytext=(6, 4), fontsize=8,
                        fontweight='bold')
    ax.set_xscale('log')
    ax.set_xlabel('$B \\cdot W$')
    ax.set_ylabel('Gp non satura (%)')
    title = ('Il prodotto $B \\cdot W$ separa i due regimi' if sep
             else 'Il prodotto $B \\cdot W$ NON separa i due regimi')
    ax.set_title(f'{title}\n(rosso = nuvola collassata)', fontsize=11)
    ax.grid(alpha=0.3, which='both')
    fig.savefig(figpath(fname), dpi=150)
    plt.close(fig)
    print(f"  {FIGDIR}/{fname}")

    if ties:
        print(f"\n  ATTENZIONE: {len(ties)} pareggi esatti in B*W con esito "
              f"opposto:")
        for a, b in ties:
            ca = 'collassata' if a['collapsed'] else 'viva'
            cb = 'collassata' if b['collapsed'] else 'viva'
            print(f"    B*W = {a['B']*a['W']:g}:  "
                  f"W={a['W']:g}/B={a['B']:g} {ca}  vs  "
                  f"W={b['W']:g}/B={b['B']:g} {cb}")
        print("  Il prodotto non e' la variabile che governa la frontiera:")
        print("  l'enunciato del testo va rivisto.")
    elif not sep:
        print(f"\n  ATTENZIONE: i due gruppi si sovrappongono in B*W "
              f"(collassate da {bw[col].min():g}, vive fino a {bw[~col].max():g}):")
        print("  nessuna soglia su B*W separa le due popolazioni.")
    return dict(separable=bool(sep), n_ties=len(ties))


# ═══════════════════════════════════════════════════════════════════════
if __name__ == '__main__':

    # ------------------------------------------------------------------
    # Griglia del paper: e' quella su cui poggiano la mappa di fase, la
    # replica dell'RFC e ogni confronto con gli autori.
    # ------------------------------------------------------------------
    res  = load()
    rows = to_table(res)
    live, dead = print_table(rows)
    stats = correlation(live)
    field_ceiling(live)

    print(f"\n  grafici in {os.path.abspath(FIGDIR)}/")
    plot_phase_map_slide(res)
    plot_active_vs_id(rows, stats=stats,
                      title_suffix='griglia del paper')
    plot_frontier_collapse(rows)
    plot_h0_curve(grid_rows=rows)
    plot_rfc_refutation()

    # ------------------------------------------------------------------
    # Griglia infittita in W a B fisso, prodotta da grid_estesa.py.
    # ------------------------------------------------------------------
    ext_pt = os.path.join(R.OUTPUT_DIR, 'grid_gp_estesoW.pt')
    if os.path.exists(ext_pt):
        print(f"\n{'='*88}")
        print(f"  GRIGLIA INFITTITA IN W (supplementare, non e' quella del paper)")
        print(f"{'='*88}")
        rows_e = to_table(torch.load(ext_pt, weights_only=False))
        live_e, _ = print_table(rows_e)
        stats_e = correlation(live_e)
        plot_active_vs_id(rows_e, stats=stats_e,
                          title_suffix='griglia infittita in W a B fisso',
                          fname='02b_gradi_liberta_vs_dimensione_estesa.png')
    else:
        print(f"\n  ({ext_pt} assente: lancia grid_estesa.py se vuoi anche la")
        print(f"   figura sulla griglia infittita)")


        
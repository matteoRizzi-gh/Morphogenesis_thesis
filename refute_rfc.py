"""
refute_rfc.py
=============
Genera i numeri che mostrano perche' l'RFC non e' evidenza di asservimento.
"""

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist

import Rfc_and_ID as R
from dynamics_t_sparse_gh import BioelectricLattice, G_r



# ═══════════════════════════════════════════════════════════════════════
#  Phi: Vmem -> campo aggregato per cella
# ═══════════════════════════════════════════════════════════════════════

def field_from_vmem(v, net):
    """Riusa _eV_from_cells della dinamica verificata, cosi' Phi non puo'
    divergere dal modello. Differenziabile: serve per la jacobiana."""
    saved = net.v
    net.v = v
    try:
        eV = net._eV_from_cells()
    finally:
        net.v = saved
    return torch.zeros(net.N, dtype=torch.float64).scatter_add_(
        0, net.endo_c_idx,
        eV.index_select(0, net.endo_f_idx)) / R.NUM_FIELD_NEIGHBORS


def phi_batch(V, net):
    with torch.no_grad():
        return torch.stack([
            field_from_vmem(torch.as_tensor(x, dtype=torch.float64), net)
            for x in np.asarray(V)]).numpy()


def _settled_net(W, B, Rr, G, T, seed):
    net = BioelectricLattice(rows=11, cols=11, G=G, R=Rr, W=W, B=B)
    net.reset()
    g = torch.Generator().manual_seed(int(seed))
    net.Gp = torch.rand(net.N, generator=g, dtype=torch.float64) * 2.0 * G_r
    with torch.no_grad():
        for _ in range(T):
            net.step(v_left_22=None)
    return net


# ═══════════════════════════════════════════════════════════════════════
#  Passo 2: lo spettro di Phi
# ═══════════════════════════════════════════════════════════════════════

def jacobian_spectrum(W=1000.0, B=5e-4, Rr=4, G=0.05, T=5000, seed=0,
                      row_scale=None):
    """Spettro della jacobiana di Phi nello stato raggiunto dopo T passi. """
    net = _settled_net(W, B, Rr, G, T, seed)
    v0 = net.v.detach().clone()
    J = torch.autograd.functional.jacobian(
        lambda v: field_from_vmem(v, net), v0).numpy()

    if row_scale is not None:
        sc = np.asarray(row_scale, dtype=np.float64).ravel()
        if sc.shape[0] != J.shape[0]:
            raise ValueError(
                f"row_scale ha {sc.shape[0]} elementi, J ha {J.shape[0]} righe")
        J = J / np.where(sc > 0, sc, 1.0)[:, None]

    s = np.linalg.svd(J, compute_uv=False)
    energy = np.cumsum(s**2) / (s**2).sum()
    return dict(s=s, s_max=float(s[0]), s_min=float(s[-1]),
                cond=float(s[0]/s[-1]) if s[-1] > 0 else float('inf'),
                rank=int((s > 1e-12*s[0]).sum()),
                participation=float(s.sum()**2 / (s**2).sum()),
                n90=int(np.searchsorted(energy, 0.90)) + 1,
                n99=int(np.searchsorted(energy, 0.99)) + 1,
                scaled=(row_scale is not None))


def spectrum_vs_R(Rs=(1, 4, 10, 21), W=1000.0, B=5e-4, G=0.05, T=5000, seed=0,
                  row_scale=None):
    print(f"\n{'='*72}\n  SPETTRO DELLA JACOBIANA DI Phi (Vmem -> campo)\n{'='*72}")
    if row_scale is not None:
        print("  righe riscalate per la std per cella della nuvola del campo")
        print("  (controllo di robustezza: la versione riportata e' quella grezza)")
    print(f"  {'R':>4}{'rango':>8}{'condiz.':>12}{'partecipaz.':>13}"
          f"{'dir. 90%':>10}{'dir. 99%':>10}")
    out = {}
    for Rr in Rs:
        d = jacobian_spectrum(W, B, Rr, G, T, seed, row_scale=row_scale)
        out[Rr] = d
        print(f"  {Rr:4d}{d['rank']:>5}/121{d['cond']:12.2e}"
              f"{d['participation']:13.2f}{d['n90']:10d}{d['n99']:10d}")
    print("  'dir. 90%' = direzioni che portano il 90% dell'energia.")
    return out

# ═══════════════════════════════════════════════════════════════════════
#  Passo 3: Phi applicata a nuvole senza dinamica
# ═══════════════════════════════════════════════════════════════════════

def make_cloud(W=1000.0, B=5e-4, Rr=4, G=0.05, n_points=200, T=5000,
               seed=0, verbose=True):
    """Nuvola di Vmem. In parallelo con run_parallel_rfc_and_id.py ."""
    V = []
    for i in range(n_points):
        snap = R.run_snapshots(W=W, B=B, R=Rr, G=G,
                               seed=seed*100000 + i, snapshots=(T,))
        V.append(snap[T]['vmem'])
        if verbose and (i + 1) % 50 == 0:
            print(f"    {i+1}/{n_points}")
    return np.asarray(V)


def control_clouds(V, seed=0):
    """Nuvole con la stessa scala ma NESSUNA struttura dinamica."""
    rng = np.random.default_rng(seed)
    n, d = V.shape
    return {
        'dinamica vera':      V,
        'colonne mescolate':  np.column_stack(
            [rng.permutation(V[:, j]) for j in range(d)]),
        'gaussiana marginali': rng.normal(V.mean(0), V.std(0) + 1e-15,
                                          size=V.shape),
        'uniforme nel range':  rng.uniform(V.min(0), V.max(0) + 1e-15,
                                           size=V.shape),
    }


def _phase_randomize_common(X, rng):
    """Randomizzazione di fase con fasi COMUNI a tutte le colonne."""
    X = np.asarray(X, dtype=np.float64)
    T = X.shape[0]
    mu = X.mean(axis=0)
    F = np.fft.rfft(X - mu, axis=0)
    ph = rng.uniform(0.0, 2.0 * np.pi, size=F.shape[0])
    ph[0] = 0.0                      # media reale
    if T % 2 == 0:
        ph[-1] = 0.0                 # Nyquist reale
    return np.fft.irfft(F * np.exp(1j * ph)[:, None], n=T, axis=0) + mu

def _iaaft_1d(x, rng, n_iter=200, tol=1e-10):
    """IAAFT su una singola serie: alterna imposizione dello spettro di
    ampiezza e riordinamento sui valori originali, fino a punto fisso."""
    x = np.asarray(x, dtype=np.float64)
    T = len(x)
    amp = np.abs(np.fft.rfft(x))
    sorted_x = np.sort(x)
    y = rng.permutation(x)
    prev = None
    for _ in range(n_iter):
        Fy = np.fft.rfft(y)
        y = np.fft.irfft(amp * np.exp(1j * np.angle(Fy)), n=T)
        y = sorted_x[np.argsort(np.argsort(y))]
        if prev is not None and np.max(np.abs(y - prev)) < tol:
            break
        prev = y.copy()
    return y


def _iaaft(X, rng, n_iter=200):
    """IAAFT colonna per colonna, con fasi indipendenti.
    """
    X = np.asarray(X, dtype=np.float64)
    return np.column_stack([_iaaft_1d(X[:, j], rng, n_iter)
                            for j in range(X.shape[1])])


def control_clouds_iaaft(V, seed=0, n_iter=200):
    """Nuvole di controllo che preservano l'autocorrelazione.
    """
    rng = np.random.default_rng(seed)
    V = np.asarray(V, dtype=np.float64)
    return {
        'dinamica vera':     V,
        'fasi comuni':       _phase_randomize_common(V, rng),
        'IAAFT per cella':   _iaaft(V, rng, n_iter),
        'colonne mescolate': np.column_stack(
            [rng.permutation(V[:, j]) for j in range(V.shape[1])]),
    }


def _std(X):
    Z = StandardScaler().fit_transform(np.asarray(X, dtype=np.float64))
    return Z[:, np.isfinite(Z).all(0)]



def refute_rfc(W=1000.0, B=5e-4, Rr=4, G=0.05, n_points=200, T=5000,
               seed=0, V=None, k=10, verbose=True):
    # Phi dipende dallo stato solo attraverso net.v, che phi_batch sovrascrive
    # riga per riga: qui basta la geometria, non serve portare la rete a regime.
    net = BioelectricLattice(rows=11, cols=11, G=G, R=Rr, W=W, B=B)

    if V is None:
        if verbose:
            print(f"  nuvola di {n_points} stati (T={T})...")
        V = make_cloud(W, B, Rr, G, n_points, T, seed, verbose)
    spread = float(pdist(V).max() / np.sqrt(V.shape[1]))
    diag = R.cloud_diagnostics(V)
    if R.is_collapsed(spread):
        print(f"  ATTENZIONE: nuvola collassata (spread {spread:.2e} V < "
              f"{R.SPREAD_COLLAPSED:g}).")
        print(f"  Con questi parametri il test non ha oggetto: usa un t "
              f"precedente alla convergenza.")
        print(f"  (H0 a eps={R.COLLAPSE_TOL:g}: {diag['n_distinct']} componenti "
              f"su {len(V)} — H0 risolve fino al round-off e non e' il criterio)")
        return None

    print(f"\n{'='*76}\n  Phi APPLICATA A NUVOLE CON E SENZA DINAMICA\n"
          f"  W={W} B={B} R={Rr} G={G}  n={len(V)}  T={T}\n"
          f"  spread {spread:.2e} V   H0 {diag['n_distinct']}/{len(V)} "
          f"a eps={R.COLLAPSE_TOL:g}\n{'='*76}")
    print(f"  ID: Levina-Bickel k={k} su dati GREZZI, stessa convenzione di")
    print(f"  Rfc_and_ID.id_of_cloud e di mu_test.diagnose")
    print(f"  {'nuvola':>22}{'ID Vmem':>10}{'ID campo':>10}{'calo':>9}"
          f"{'RFC analogo':>13}")
    rows = {}
    for name, X in control_clouds(V, seed).items():
        F = phi_batch(X, net)
        id_v, id_f = R.mle_id(X, k=k), R.mle_id(F, k=k)
        # La PCA resta standardizzata: replica StandardScaler -> PCA del
        # codice degli autori, ed e' parte della quantita' da riprodurre.
        rfc = 100*(R.pca_explained(F).sum() - R.pca_explained(X).sum())
        rows[name] = dict(id_vmem=id_v, id_field=id_f, rfc=rfc,
                          twonn_vmem=R.twonn(X), twonn_field=R.twonn(F))
        print(f"  {name:>22}{id_v:10.2f}{id_f:10.2f}{id_f-id_v:+9.2f}"
              f"{rfc:+12.2f}%")
    torch.save(rows, R.out_path('refute_rfc.pt'))
    return rows



# ═══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    # versione grezza
    spectrum_vs_R(Rs=(1, 4, 10, 21))

    # versione nella convenzione delle stime di ID: serve la std per cella
    # della nuvola del campo alla stessa (W, B, R, T)
    net0 = BioelectricLattice(rows=11, cols=11, G=0.05, R=4, W=1000.0, B=5e-4)
    V0 = make_cloud(1000.0, 5e-4, 4, 0.05, n_points=300, T=5000, verbose=False)
    F0 = phi_batch(V0, net0)
    spectrum_vs_R(Rs=(1, 4, 10, 21), row_scale=F0.std(axis=0))

    refute_rfc(W=1000.0, B=5e-4, Rr=4, n_points=300, T=5000, V=V0)


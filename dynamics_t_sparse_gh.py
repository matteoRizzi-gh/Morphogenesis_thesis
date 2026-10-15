"""
dynamics_t_sparse_gh.py 
================================================================
Sequenza di un passo:
    eV = |E| dalle sole celle             (sovrascrive eV)
    Gp  <- eV
    Vmem
    --- se clampato ---
    eV[bordo] = valori di clamp           (imposti)
    eV[liberi] += |E| dalle cariche C*eV(bordo)   (additivo, 1/r^2)
    Gp  <- eV
    Vmem

"""

import math
import numpy as np
import torch

C   = 0.1e-9
E_p = -55e-3          
E_d = -5e-3           
G_r = 1.0e-9
G_d = 1.5e-9
k_e = 8.987e9
e_r = 1.0e7
r   = 5.0e-6

z    = 3.0
v_th = -27e-3
v_T  =  27e-3
v_0  =  12e-3

DT   = 0.01

V_INIT  = -9.2e-3     # condizione iniziale, come nel paper
EPSILON = 1e-15       
GAIN    = -1.0        # fieldTransductionGain 
FIELD_STRENGTH = 1.0  # fieldStrength

T_PHYS_END = 0.5


def build_drive(amp, freq, phase, T_stim):
    """signal[t,k] = amp[k]*cos(freq[k]*t + phase[k]), t = linspace(0, 0.5, T_stim)."""
    t = torch.linspace(0.0, T_PHYS_END, T_stim, dtype=torch.float64)
    return amp.unsqueeze(0) * torch.cos(
        freq.unsqueeze(0) * t.unsqueeze(1) + phase.unsqueeze(0))


# ═══════════════════════════════════════════════════════════════════════════
#  Setup geometrico (invariato)
# ═══════════════════════════════════════════════════════════════════════════

def _cell_positions(rows, cols):
    step = 2.0 * r
    js = (np.arange(cols) + 0.5) * step
    is_ = (np.arange(rows) + 0.5) * step
    xs, ys = np.meshgrid(js, is_)
    return np.stack([xs.ravel(), ys.ravel()], axis=1)

def _field_positions(rows, cols):
    step = 2.0 * r
    js = np.arange(cols + 1) * step
    is_ = np.arange(rows + 1) * step
    xs, ys = np.meshgrid(js, is_)
    return np.stack([xs.ravel(), ys.ravel()], axis=1)

def _action_mask(cells_or_sources, field_pts, R_val):
    diff   = field_pts[:, None, :] - cells_or_sources[None, :, :]
    dist   = np.sqrt((diff**2).sum(-1))
    cutoff = math.sqrt(2.0) * r * R_val
    tol    = r * 1e-6
    return (dist <= cutoff + tol), dist

def _perception_mask(cells, field_pts):
    mask, _ = _action_mask(cells, field_pts, R_val=1)
    return mask.T

def _adjacency_edges(rows, cols):
    ei, ej = [], []
    for row in range(rows):
        for col in range(cols):
            k = row * cols + col
            if col < cols - 1:
                ei.append(k); ej.append(k + 1)
            if row < rows - 1:
                ei.append(k); ej.append(k + cols)
    return np.array(ei, dtype=np.int64), np.array(ej, dtype=np.int64)

def _boundary_indices(rows, cols):
    fc = cols + 1
    fr = rows + 1
    all_bnd, left_bnd, right_bnd = [], [], []
    for row in range(fr):
        for col in range(fc):
            on_bnd = (row == 0 or row == fr - 1 or col == 0 or col == fc - 1)
            if not on_bnd:
                continue
            idx = row * fc + col
            all_bnd.append(idx)
            if col < fc // 2:
                left_bnd.append(idx)
            else:
                right_bnd.append(idx)
    return (np.array(all_bnd, dtype=np.int64),
            np.array(left_bnd, dtype=np.int64),
            np.array(right_bnd, dtype=np.int64))

def _mirror_perm(rows, cols):
    half   = (cols + 1) // 2
    n_side = rows - 1
    perm = (list(range(half-1, -1, -1))
            + list(range(half, half + n_side))
            + list(range(half + n_side + half - 1, half + n_side - 1, -1)))
    return perm


class BioelectricLattice:
    def __init__(self, rows=11, cols=11, G=0.05, R=4, W=1000.0, B=5e-4,
                 gain=GAIN, device=None, drive_form="codice"):
        
        if drive_form not in ("codice", "stampata"):
            raise ValueError("drive_form: 'codice' o 'stampata'")
        self.drive_form = drive_form


        self.last_avg_e = self.last_drive = None
        self.rows = rows; self.cols = cols; self.N = rows * cols
        self.G = G; self.R = R; self.W = W; self.B = B; self.gain = gain
        self.device = torch.device('cpu')

        cells_np     = _cell_positions(rows, cols)
        field_pts_np = _field_positions(rows, cols)
        self.N_f     = len(field_pts_np)
        self.field_pts_np = field_pts_np

        # ── Campo endogeno: COO sparso (field_point, cell), screening R ─────
        r_ij_np = field_pts_np[:, None, :] - cells_np[None, :, :]
        mask_np, dist_np = _action_mask(cells_np, field_pts_np, R)
        safe_dist = np.where(mask_np, dist_np, 1.0)

        f_idx, c_idx = np.nonzero(mask_np)
        inv_r2_vals = 1.0 / (safe_dist[f_idx, c_idx] ** 2)
        coeff_x = inv_r2_vals * r_ij_np[f_idx, c_idx, 0]
        coeff_y = inv_r2_vals * r_ij_np[f_idx, c_idx, 1]

        self.endo_f_idx   = torch.tensor(f_idx, dtype=torch.long)
        self.endo_c_idx   = torch.tensor(c_idx, dtype=torch.long)
        self.endo_coeff_x = torch.tensor(coeff_x, dtype=torch.float64)
        self.endo_coeff_y = torch.tensor(coeff_y, dtype=torch.float64)
        self.endo_nnz     = len(f_idx)

        # ── Percezione (R=1 fisso): COO su (cell, field_point) ─────────────
        perc_mask_np = _perception_mask(cells_np, field_pts_np)
        p_cell_idx, p_field_idx = np.nonzero(perc_mask_np)
        self.perc_cell_idx  = torch.tensor(p_cell_idx, dtype=torch.long)
        self.perc_field_idx = torch.tensor(p_field_idx, dtype=torch.long)
        perc_n_np = np.bincount(p_cell_idx, minlength=self.N).astype(np.float64)
        self.perc_n = torch.tensor(np.clip(perc_n_np, 1.0, None), dtype=torch.float64)

        ei_np, ej_np = _adjacency_edges(rows, cols)
        self.ei = torch.tensor(ei_np, dtype=torch.long)
        self.ej = torch.tensor(ej_np, dtype=torch.long)

        self._setup_boundary(field_pts_np, rows, cols)
        self.mirror_perm_t = torch.tensor(_mirror_perm(rows, cols), dtype=torch.long)
        self.reset()

    def _setup_boundary(self, field_pts_np, rows, cols):
        """
        Geometria per il contributo del clamp: dai 44 punti di bordo verso i
        soli punti LIBERI, con 1/r^2 vettoriale e senza screening R
        """
        all_bnd, left_bnd, right_bnd = _boundary_indices(rows, cols)
        self.bnd_all_idx  = all_bnd
        self.bnd_left_idx = left_bnd
        self.n_left       = len(left_bnd)
        self.bnd_idx_t    = torch.tensor(all_bnd, dtype=torch.long)

        free = np.setdiff1d(np.arange(len(field_pts_np)), all_bnd)
        self.free_idx_t = torch.tensor(free, dtype=torch.long)
        self.n_free     = len(free)

        # delta = posizione_libero - posizione_bordo   [n_free, n_bnd]
        dx = field_pts_np[free][:, None, 0] - field_pts_np[all_bnd][None, :, 0]
        dy = field_pts_np[free][:, None, 1] - field_pts_np[all_bnd][None, :, 1]
        r2 = dx**2 + dy**2
        self.clamp_coeff_x = torch.tensor(dx / r2, dtype=torch.float64)
        self.clamp_coeff_y = torch.tensor(dy / r2, dtype=torch.float64)

        left_map  = {v: k for k, v in enumerate(left_bnd)}
        right_map = {v: k + len(left_bnd) for k, v in enumerate(right_bnd)}
        lr_to_all = [left_map[a] if a in left_map else right_map[a] for a in all_bnd]
        self.lr_to_all_perm_t = torch.tensor(lr_to_all, dtype=torch.long)

    def reset(self):
        self.v  = torch.full((self.N,), V_INIT,  dtype=torch.float64)
        self.Gp = torch.full((self.N,), 1.0*G_r, dtype=torch.float64)

    # ── Campo endogeno: eV = |E| dalle sole celle ────────────────────────────
    def _field_endo(self):
        """Componenti (ex, ey) del campo generato dalle cariche delle celle."""
        q   = C * self.v
        q_g = q.index_select(0, self.endo_c_idx)
        fc  = FIELD_STRENGTH * (k_e / e_r)
        ex = torch.zeros(self.N_f, dtype=torch.float64).scatter_add_(
            0, self.endo_f_idx, self.endo_coeff_x * q_g) * fc
        ey = torch.zeros(self.N_f, dtype=torch.float64).scatter_add_(
            0, self.endo_f_idx, self.endo_coeff_y * q_g) * fc
        return ex, ey

    def _eV_from_cells(self):
        """updateExtracellularVoltage(source='Vmem') con fieldVector=True."""
        ex, ey = self._field_endo()
        return torch.sqrt(ex**2 + ey**2 + EPSILON)

    # ── Contributo additivo del clamp ai punti liberi ────────────────────────
    def _eV_clamp_contribution(self, v_bnd_44):
        """
        Le cariche vengono dai valori di eV imposti al bordo: Q = C * eV(bordo).
        Ritorna deV [n_free], da SOMMARE a eV nei punti liberi.
        """
        q_bnd = C * v_bnd_44
        fc    = FIELD_STRENGTH * (k_e / e_r)
        eVx = fc * torch.matmul(self.clamp_coeff_x, q_bnd)
        eVy = fc * torch.matmul(self.clamp_coeff_y, q_bnd)
        return torch.sqrt(eVx**2 + eVy**2)

    def _mirror_to_44(self, v_left_22):
        v_right_22 = v_left_22[self.mirror_perm_t]
        v_lr       = torch.cat([v_left_22, v_right_22])
        return v_lr[self.lr_to_all_perm_t]

    # ── Aggiornamento di Gp: unica funzione, usata in entrambi i punti ──────
    def _update_Gp(self, eV):
        """ dp = -Gp + (2*sigmoid(gain*mean(eV) + B) - 1) * W ;  Gp += DT*dp*G_r"""
        e_g = eV.index_select(0, self.perc_field_idx)
        sum_per_cell = torch.zeros(self.N, dtype=torch.float64).scatter_add_(
            0, self.perc_cell_idx, e_g)
        avg_e = sum_per_cell / self.perc_n
        sigma = torch.sigmoid(self.gain * avg_e + self.B)
        if self.drive_form == "codice":
            drive = (2.0 * sigma - 1.0) * self.W      # repository
        else:
            drive = 2.0 * self.W * sigma - 1.0        # equazione del paper
        self.last_avg_e, self.last_drive = avg_e.detach(), drive.detach()
        dp = -self.Gp + drive
        self.Gp = torch.clamp(self.Gp + DT * dp * G_r, 0.0, 2.0 * G_r)

    # ── Aggiornamento di Vmem (invariato) ───────────────────────────────────
    def _update_v(self):
        arg   = z * (self.v - v_th) / v_T
        I_pol = -self.Gp * (self.v - E_p) / (1.0 + torch.exp( arg))
        I_dep = -G_d     * (self.v - E_d) / (1.0 + torch.exp(-arg))
        dv_edge = self.v[self.ei] - self.v[self.ej]
        G_ij    = 2.0 * self.G * G_r / (1.0 + torch.cosh(dv_edge / v_0))
        I_gj = torch.zeros(self.N, dtype=torch.float64)
        I_gj.scatter_add_(0, self.ei,  G_ij * (self.v[self.ej] - self.v[self.ei]))
        I_gj.scatter_add_(0, self.ej,  G_ij * (self.v[self.ei] - self.v[self.ej]))
        self.v = self.v + DT * (I_pol + I_dep + I_gj) / C

    # ── Passo, nella sequenza del codice ufficiale ──────────────────────────
    def step(self, v_left_22=None):
        """
        Ritorna (eV, zeros): il primo elemento e' GIA' il modulo del campo,
        il secondo e' un segnaposto nullo per compatibilita' con il codice
        a valle che calcola sqrt(ex**2 + ey**2) — che cosi' riottiene eV.
        """
        eV = self._eV_from_cells()          # sovrascrive: solo celle
        self._update_Gp(eV)
        self._update_v()

        if v_left_22 is not None:
            v_bnd_44 = self._mirror_to_44(v_left_22)
            eV = eV.clone()
            eV[self.bnd_idx_t] = v_bnd_44                       # imposto
            deV = self._eV_clamp_contribution(v_bnd_44)
            eV[self.free_idx_t] = eV[self.free_idx_t] + deV     # additivo
            self._update_Gp(eV)
            self._update_v()

        return eV, torch.zeros_like(eV)

    def run(self, T=1000, save_every=1,
            amplitudes=None, phases=None, frequencies=None, T_stim=None):
        drive = None
        if amplitudes is not None:
            if T_stim is None:
                T_stim = max(1, T // 10)
            amp  = torch.as_tensor(amplitudes,  dtype=torch.float64)
            ph   = torch.as_tensor(phases,      dtype=torch.float64)
            freq = torch.as_tensor(frequencies, dtype=torch.float64)
            drive = build_drive(amp, freq, ph, T_stim)

        vs, gps, ems = [], [], []
        for t in range(T):
            v_left = drive[t] if (drive is not None and t < T_stim) else None
            ex, ey = self.step(v_left_22=v_left)
            if (t + 1) % save_every == 0:
                vs.append(self.v.clone())
                gps.append(self.Gp.clone())
                ems.append(torch.sqrt(ex**2 + ey**2).clone())
        return (torch.stack(vs, dim=0), torch.stack(gps, dim=0),
                torch.stack(ems, dim=0))

    
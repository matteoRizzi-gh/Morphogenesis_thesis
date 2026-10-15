"""
figure_training.py
Figure della validazione dell'ottimizzazione e del test di persistenza.

Legge face_mosaic_BEST.pt e face_stigmergic_BEST.pt (output di
train_cfn_gh.run_multi_parallel), rigira i drive migliori nel modello per
T_LONG passi e produce:
  training_validazione.pdf : curve di loss + pattern a t=1000 accanto al target
  persistenza_faccia.pdf   : RMSE dal target e celle iperpolarizzate nel tempo
Stampa anche la loss ricalcolata dai parametri salvati.
"""
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dynamics_t_sparse_gh import BioelectricLattice, build_drive, E_p, E_d
import train_cfn_gh as tc

T_LONG = 5000
AUTORI = {'mosaic': 1.0255, 'stigmergic': 2.0842}     # best dai file .dat
PAVIMENTO = 0.5147
NOMI = {'mosaic': 'mosaic', 'stigmergic': 'stigmergico'}


def replay(res, T=T_LONG):
    p, bp = res['net_params'], res['best_params']
    net = BioelectricLattice(rows=11, cols=11, G=p['G'], R=p['R'],
                             W=p['W'], B=p['B'])
    drive = build_drive(bp['amp'], bp['freq'], bp['phase'], p['T_stim'])
    net.reset(); V = []
    with torch.no_grad():
        for s in range(T):
            net.step(v_left_22=drive[s] if s < p['T_stim'] else None)
            V.append(net.v.clone())
    return torch.stack(V)


def main():
    target = tc.build_target()
    dati = {}
    for v in ('mosaic', 'stigmergic'):
        res = torch.load(f'face_{v}_BEST.pt', weights_only=False,
                         map_location='cpu')
        V = replay(res)
        loss = ((target.unsqueeze(0) - V[900:1000]) ** 2).sum().sqrt().item()
        print(f"{v}: loss salvata {res['best_loss']:.4f}, "
              f"ricalcolata dai parametri salvati {loss:.4f}, "
              f"best all'iterazione {res['best_iter']}")
        dati[v] = dict(res=res, V=V)

    # --- figura 1: loss e pattern -----------------------------------------
    fig = plt.figure(figsize=(9, 5.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.1], hspace=0.55)
    ax = fig.add_subplot(gs[0, :])
    for v, c in (('mosaic', 'C0'), ('stigmergic', 'C1')):
        ax.plot(dati[v]['res']['loss_history'], color=c, label=NOMI[v])
        ax.axhline(AUTORI[v], color=c, ls='--', lw=0.8)
    ax.axhline(PAVIMENTO, color='k', ls=':', lw=0.8, label='pavimento')
    ax.set_xlabel('iterazione'); ax.set_ylabel('loss')
    ax.legend(frameon=False, fontsize=8)
    vmin, vmax = E_p - 5e-3, E_d + 5e-3
    pannelli = [(dati['mosaic']['V'][999], 'mosaic, t=1000'),
                (dati['stigmergic']['V'][999], 'stigmergico, t=1000'),
                (target, 'target')]
    for k, (x, tit) in enumerate(pannelli):
        a = fig.add_subplot(gs[1, k])
        im = a.imshow(x.view(11, 11).numpy(), cmap='gray', vmin=vmin, vmax=vmax)
        a.set_title(tit, fontsize=9); a.axis('off')
    fig.colorbar(im, ax=fig.axes[1:], shrink=0.8, label=r'$V_\mathrm{mem}$ (V)')
    fig.savefig('training_validazione.pdf', bbox_inches='tight')

    # --- figura 2: persistenza --------------------------------------------
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.2))
    t = np.arange(1, T_LONG + 1)
    for v, c in (('mosaic', 'C0'), ('stigmergic', 'C1')):
        V = dati[v]['V']
        rmse = ((V - target) ** 2).mean(1).sqrt().numpy() * 1e3
        axs[0].plot(t, rmse, color=c, label=NOMI[v])
        axs[1].plot(t, (V < -0.04).sum(1).numpy(), color=c, label=NOMI[v])
    for a in axs:
        a.axvspan(900, 1000, color='0.85', lw=0)
        a.set_xlabel('passo')
    axs[0].set_ylabel('RMSE dal target (mV)')
    axs[1].set_ylabel('celle con $v<-40$ mV')
    axs[1].axhline(54, color='k', ls=':', lw=0.8)
    axs[0].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig('persistenza_faccia.pdf')
    print("salvate training_validazione.pdf e persistenza_faccia.pdf")


if __name__ == '__main__':
    main()

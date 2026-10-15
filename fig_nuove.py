
import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
import os


# ---------- 1. Test per intervento (dati dall'output di intervento_saturazione.py)
rows = [  # k, lato, attiva(%), MLE
 (0,'nessuno',85.3,44.72),
 (15,'max',75.2,41.62),(15,'max',75.2,41.80),(15,'max',75.6,41.04),
 (15,'zero',74.5,41.65),(15,'zero',74.6,42.31),(15,'zero',74.4,40.98),
 (30,'max',65.1,36.01),(30,'max',65.0,37.65),(30,'max',64.8,35.88),
 (30,'zero',63.8,36.73),(30,'zero',63.7,37.62),(30,'zero',63.6,36.98),
 (60,'max',44.2,29.50),(60,'max',43.5,28.40),(60,'max',44.3,29.29),
 (60,'zero',42.3,29.46),(60,'zero',42.2,28.37),(60,'zero',42.3,29.80),
 (90,'max',22.3,17.45),(90,'max',21.9,17.05),(90,'max',22.9,17.38),
 (90,'zero',21.3,17.85),(90,'zero',21.2,16.85),(90,'zero',21.3,16.73)]
fig, ax = plt.subplots(figsize=(6.2,4.4))
cols = {0:'k',15:'C0',30:'C1',60:'C2',90:'C3'}
for lato, mk, lab in (('nessuno','*','k = 0 (nessun blocco)'),
                      ('max','o','bloccate a g = 2'),('zero','s','bloccate a g = 0')):
    sel=[r for r in rows if r[1]==lato]
    ax.scatter([r[2] for r in sel],[r[3] for r in sel],marker=mk,
               c=[cols[r[0]] for r in sel],s=110 if mk=='*' else 45,
               edgecolors='k',linewidths=0.5,label=lab,zorder=3)
for k in (15,30,60,90):
    sel=[r for r in rows if r[0]==k]
    ax.annotate(f"k = {k}",(np.mean([r[2] for r in sel]),np.mean([r[3] for r in sel])),
                xytext=(8,-14),textcoords='offset points',fontsize=8,color=cols[k])
ax.set_xlabel('frazione di conduttanze interne misurata (%)')
ax.set_ylabel('dimensione intrinseca di $V_\\mathrm{mem}$ (MLE)')
ax.set_xlim(0,100); ax.set_ylim(0,50); ax.grid(alpha=0.3)
from matplotlib.lines import Line2D
h=[Line2D([],[],marker='*',ls='',color='k',ms=11,label='k = 0 (nessun blocco)'),
   Line2D([],[],marker='o',ls='',mfc='0.7',mec='k',ms=7,label='bloccate a g = 2'),
   Line2D([],[],marker='s',ls='',mfc='0.7',mec='k',ms=7,label='bloccate a g = 0')]
ax.legend(handles=h,frameon=False,fontsize=8,loc='upper left')
fig.tight_layout(); fig.savefig('11_intervento.png',dpi=200); fig.savefig('11_intervento.pdf')

# ---------- 2. RFC sulla serie temporale: dinamica vera e controlli
var=['dinamica\nvera','fasi\ncomuni','IAAFT\nper cella','colonne\nmescolate']
dati={'stigmergico':[(79.88,60.27),(75.51,60.27),(71.19,53.90),(56.79,3.24)],
      'mosaic':[(98.22,91.93),(97.33,91.93),(62.58,38.46),(43.22,3.24)]}
fig, axs = plt.subplots(1,2,figsize=(9,3.8),sharey=True)
x=np.arange(4); w=0.38
for ax,(nome,d) in zip(axs,dati.items()):
    c=[a for a,_ in d]; v=[b for _,b in d]
    ax.bar(x-w/2,c,w,label='campo',color='C1'); ax.bar(x+w/2,v,w,label='$V_\\mathrm{mem}$',color='C0')
    for i,(a,b) in enumerate(d):
        ax.text(i,max(a,b)+3,f"RFC\n{a-b:+.1f}",ha='center',fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(var,fontsize=8); ax.set_title(nome,fontsize=10)
    ax.set_ylim(0,118); ax.grid(axis='y',alpha=0.3)
axs[0].set_ylabel('varianza nelle prime tre PC (%)')
axs[0].legend(frameon=False,fontsize=8,loc='upper right')
fig.tight_layout(); fig.savefig('12_rfc_serie_temporale.png',dpi=200); fig.savefig('12_rfc_serie_temporale.pdf')
print('ok')


###########################################################################################################################

LARGHEZZA_CM, ALTEZZA_CM = 13.0, 4.2
SOGLIA_DENTRO = 15   # sotto questa altezza (%) il numero va sopra la barra
plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 7,
                     'xtick.labelsize': 6.5, 'ytick.labelsize': 7,
                     'legend.fontsize': 7})
 
var = ['dinamica\nvera', 'fasi\ncomuni', 'IAAFT\nper cellula', 'colonne\nmescolate']
# (campo, Vmem) in percentuale, RFC come riportata nella tesi
dati = {'regime forte':  [(79.88, 60.27), (75.51, 60.27), (71.19, 53.90), (56.79, 3.24)],
        'regime debole': [(98.22, 91.93), (97.33, 91.93), (62.58, 38.46), (43.22, 3.24)]}
rfc = {'regime forte':  [19.60, 15.24, 17.29, 53.56],
       'regime debole': [6.29, 5.40, 24.12, 39.98]}
 
fig, axs = plt.subplots(1, 2, figsize=(LARGHEZZA_CM / 2.54, ALTEZZA_CM / 2.54),
                        sharey=True)
x = np.arange(4); w = 0.38
for ax, (nome, d) in zip(axs, dati.items()):
    c = [a for a, _ in d]; v = [b for _, b in d]
    ax.bar(x - w/2, c, w, label='campo', color='C1')
    ax.bar(x + w/2, v, w, label='$V_\\mathrm{mem}$', color='C0')
    # valori dentro le barre, alla base; sopra la barra se e' troppo bassa
    for xs, hs in ((x - w/2, c), (x + w/2, v)):
        for xi, h in zip(xs, hs):
            if h >= SOGLIA_DENTRO:
                ax.text(xi, 2, f'{h:.1f}', ha='center', va='bottom',
                        fontsize=6, color='0.1')
            else:
                ax.text(xi, h + 1.5, f'{h:.1f}', ha='center', va='bottom',
                        fontsize=6, color='0.1')
    for i, (a, b) in enumerate(d):
        ax.text(i, max(a, b) + 3, f'{rfc[nome][i]:+.1f}', ha='center', fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(var)
    ax.set_title(nome)
    ax.set_ylim(0, 118); ax.grid(axis='y', alpha=0.3)
axs[0].set_ylabel('prime tre PC (%)')
axs[0].legend(frameon=False, loc='upper right', ncol=2)
fig.tight_layout(pad=0.3, w_pad=0.8)
os.makedirs('figure', exist_ok=True)
fig.savefig('figure/fig_surrogati.pdf')
print('figura salvata in figure/fig_surrogati.pdf')


################################################################################################################################

celle=[("W=10\nB=0.002",44.72,10.35),("W=10\nB=0.0005",37.27,9.83),("W=100\nB=0.0005",36.45,11.21),
       ("W=10\nB=0.005",33.77,11.43),("W=200\nB=0.0005",29.88,9.74),("W=500\nB=0.0005",27.11,9.66),
       ("W=1000\nB=0.0005",25.66,9.08),("W=100\nB=0.002",14.85,7.59),("W=200\nB=0.002",4.52,3.45)]
x=np.arange(len(celle)); v=[c[1] for c in celle]; e=[c[2] for c in celle]
fig,ax=plt.subplots(figsize=(8.5,4.2))
ax.plot(x,v,'o-',color='C0',label='$V_\\mathrm{mem}$')
ax.plot(x,e,'s-',color='C1',label='campo osservabile ($R=4$)')
ax.axhline(8,color='k',ls='--',lw=0.9,label='$\\mathrm{dir}_{90}(\\Phi)=8$')
ax.axhline(29,color='0.5',ls=':',lw=0.9,label='$\\mathrm{dir}_{99}(\\Phi)=29$')
ax.set_xticks(x); ax.set_xticklabels([c[0] for c in celle],fontsize=7)
ax.set_ylabel('dimensione intrinseca (MLE)'); ax.set_ylim(0,48); ax.grid(alpha=0.3)
ax.legend(frameon=False,fontsize=8,loc='upper right')
fig.tight_layout(); fig.savefig('figures/03_campo_vs_vmem.png',dpi=200); fig.savefig('figures/03_campo_vs_vmem.pdf')



#############################################################################################################################


# (W, B): (t_conv mediano, t_sat mediano, residuo mediano)  -- output di timing_map
d={(10,.005):(11645,11552,92),(10,.01):(4725,4628,98),(10,.02):(2225,2122,103),
   (100,.002):(27384,27299,85),(100,.005):(1235,1130,106),(100,.01):(565,454,110),(100,.02):(347,210,137),
   (200,.002):(13535,13448,86),(200,.005):(664,555,110),(200,.01):(357,224,132),(200,.02):(263,104,159),
   (500,.002):(5238,5144,94),(500,.005):(342,213,128),(500,.01):(249,87,162),(500,.02):(217,41,176),
   (1000,.002):(2439,2342,98),(1000,.005):(256,99,157),(1000,.01):(218,42,176),(1000,.02):(206,21,185)}
col={.002:'C3',.005:'C1',.01:'C2',.02:'C0'}
fig,axs=plt.subplots(1,2,figsize=(10,4))
x=np.logspace(1,4.6,200)
axs[0].plot(x,x,'k--',lw=0.8,label='$t_\\mathrm{conv}=t_\\mathrm{sat}$')
axs[0].plot(x,x+110,'k-',lw=0.8,label='$t_\\mathrm{conv}=t_\\mathrm{sat}+110$')
for B in col:
    pts=[v for k,v in d.items() if k[1]==B]
    axs[0].scatter([p[1] for p in pts],[p[0] for p in pts],color=col[B],edgecolors='k',lw=0.4,s=35,label=f'$B={B:g}$',zorder=3)
    axs[1].scatter([p[1] for p in pts],[p[2] for p in pts],color=col[B],edgecolors='k',lw=0.4,s=35,label=f'$B={B:g}$',zorder=3)
axs[0].set_xscale('log'); axs[0].set_yscale('log')
axs[0].set_xlabel('tempo di saturazione (passi)'); axs[0].set_ylabel('tempo di convergenza (passi)')
axs[1].set_xscale('log'); axs[1].axhline(110,color='0.5',lw=0.8,ls=':')
axs[1].set_xlabel('tempo di saturazione (passi)'); axs[1].set_ylabel('coda $t_\\mathrm{conv}-t_\\mathrm{sat}$ (passi)')
for a in axs: a.grid(alpha=0.3)
axs[0].legend(frameon=False,fontsize=7,loc='upper left'); axs[1].legend(frameon=False,fontsize=7,loc='upper right')
fig.tight_layout(); fig.savefig('figures/08_decomposizione_tempi.png',dpi=200); fig.savefig('figures/08_decomposizione_tempi.pdf')


#################################################################################################################################

import numpy as np, torch, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
import timing_map_verifica_S1 as S
from dynamics_t_sparse_gh import G_r


d=torch.load('output_rfc/spettro_S1_v2.pt',weights_only=False)
full=[r for r in d['stigmergico_T']['res'] if r['T']==5000][0]['lyap']['lyap']
net,v,Gp,_,_=S.state_after_steps(1000.0,5e-4,T=5000,seed=None); N=net.N
x=torch.cat((v,Gp/G_r)); Q=np.linalg.qr(np.random.default_rng(0).standard_normal((N,N)))[0]; acc=np.zeros(N)
for t in range(600):
    J=torch.autograd.functional.jacobian(lambda y:S._scaled_step(y,net),x,vectorize=True).numpy()[:N,:N]
    Q,Rm=np.linalg.qr(J@Q)
    if t>=100: acc+=np.log(np.abs(np.diag(Rm)))
    with torch.no_grad(): x=S._scaled_step(x,net)
vb=np.sort(acc/500)[::-1]
np.save('vblock_T5000.npy',vb)
fig,ax=plt.subplots(figsize=(7,3.8))
ax.plot(np.arange(1,len(full)+1),full,'.',ms=4,color='C1',label='sistema completo (242)')
ax.plot(np.arange(1,len(vb)+1),vb,'.',ms=4,color='C0',label='solo potenziali (121)')
ax.axhline(0,color='k',lw=0.6)
ax.axhspan(-2e-3,2e-3,color='0.85',lw=0,label='errore stimato ($\\pm2\\times10^{-3}$)')
ax.set_ylim(-0.16,0.012); ax.set_xlabel('indice (esponenti in ordine decrescente)')
ax.set_ylabel('esponente di Lyapunov per passo')
ax.legend(frameon=False,fontsize=8,loc='lower left'); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig('figures/13_lyapunov_T5000.png',dpi=200); fig.savefig('figures/13_lyapunov_T5000.pdf')
print('fuori scala (sotto -0.25):',int(np.sum(full<-0.25)))





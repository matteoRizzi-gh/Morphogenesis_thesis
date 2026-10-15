import numpy as np, torch
import timing_map_verifica_S1 as S
from dynamics_t_sparse_gh import G_r
import Rfc_and_ID as R

def vblock(W,B,T,seed,M=500,burn=100):
    net,v,Gp,_,_=S.state_after_steps(W,B,T=T,seed=seed); N=net.N
    x=torch.cat((v,Gp/G_r)); Q=np.linalg.qr(np.random.default_rng(0).standard_normal((N,N)))[0]; acc=np.zeros(N)
    for t in range(burn+M):
        J=torch.autograd.functional.jacobian(lambda y:S._scaled_step(y,net),x,vectorize=True).numpy()[:N,:N]
        Q,Rm=np.linalg.qr(J@Q)
        if t>=burn: acc+=np.log(np.abs(np.diag(Rm)))
        with torch.no_grad(): x=S._scaled_step(x,net)
    lam=np.sort(acc/M)[::-1]; c=lam[lam<0]
    rat=np.max(c[1:]/c[:-1]) if len(c)>1 else np.nan
    return lam[0], -1/c.max(), -1/c.min(), rat

stati=[(1000.,5e-4,1000,None),(1000.,5e-4,5000,None),(1000.,5e-4,20000,None),(1000.,5e-4,100000,None),
       (1000.,5e-4,5000,1),(1000.,5e-4,5000,2),(100.,5e-4,5000,None),(200.,5e-4,5000,None),(500.,5e-4,5000,None)]

with open(R.out_path('lyap_vblock_all.txt'),'w') as f:
    for W,B,T,sd in stati:
        l0,tmax,tmin,rat=vblock(W,B,T,sd)
        s=f"W={W:g} T={T} seme={sd}: lambda_max_v {l0:+.4f}  tau_v in [{tmin:.1f}, {tmax:.1f}]  max rapporto consecutivo {rat:.2f}"
        print(s,flush=True); f.write(s+"\n"); f.flush()

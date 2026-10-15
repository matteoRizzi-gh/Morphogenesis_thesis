import numpy as np, torch, time
from concurrent.futures import ProcessPoolExecutor
from dynamics_t_sparse_gh import BioelectricLattice, G_r
import Rfc_and_ID as R
import loop_gain_independence as LG

def traj(i):
    torch.set_num_threads(1)
    net=BioelectricLattice(rows=11,cols=11,G=0.05,R=4,W=1000.0,B=5e-4); net.reset()
    g=torch.Generator().manual_seed(i)            # stessi semi di grid(): seed*100000+i con seed=0
    net.Gp=torch.rand(net.N,generator=g,dtype=torch.float64)*2*G_r
    with torch.no_grad():
        for _ in range(5000): eV,_=net.step(v_left_22=None)
        v=net.v.clone()
        A=torch.zeros(net.N,dtype=torch.float64).scatter_add_(
            0,net.endo_c_idx,eV.index_select(0,net.endo_f_idx))/R.NUM_FIELD_NEIGHBORS
        P=torch.zeros(net.N,dtype=torch.float64).scatter_add_(
            0,net.perc_cell_idx,eV.index_select(0,net.perc_field_idx))/net.perc_n
        return v.numpy(), A.numpy(), P.numpy(), eV.clone().numpy()

if __name__=='__main__':
    t=time.time()
    with ProcessPoolExecutor() as ex: out=list(ex.map(traj,range(300)))
    V,A,P,E=(np.array([o[k] for o in out]) for k in range(4))
    print("tempo %.0f s"%(time.time()-t))
    for nome,X in (("Vmem",V),("osservabile A_R",A),("percepito A_1",P),("grezzo eV (144)",E)):
        tw,ml,_=R.id_of_cloud(X,k=10)
        sat=R.id_vs_n(X)
        print(f"{nome:>18}: TwoNN {tw:6.2f}  MLE {ml:6.2f}   MLE a 25/50/100%: "
              + " / ".join(f"{sat[f][1]:.2f}" for f in (0.25,0.5,1.0)))
    np.savez(R.out_path('nuvola_stig_campi.npz'),V=V,A=A,P=P,E=E)

import sys, os, warnings; sys.path.insert(0,'scripts'); sys.path.insert(0,'.')
import numpy as np, torch
import tensorqtl.hapmixqtl as HM
from compare_mixqtl_replication import load_inputs, gene_variant_index
I=load_inputs(); genes,keep=I['genes'],I['keep']
A,T,Va,Vt,_=HM.compute_summaries_from_gibbs(I['YL'],I['YR'],yT=I['YT'],count_noise=True)
A,T,Va,Vt=A[:,keep],T[:,keep],Va[:,keep],Vt[:,keep]
dos=I['dos'][I['idx']][:,keep]; sgn=(I['xL']-I['xR'])[I['idx']][:,keep]
dev=torch.device('cpu'); Tt=lambda x: torch.tensor(np.asarray(x),dtype=torch.float64,device=dev)
cov_t=Tt(I['cov_df'].values)
ra_, rt_, ta_, tt_ = [],[],[],[]
for j,g in enumerate(genes):
    vsel=gene_variant_index(I,g)
    if vsel.size==0: continue
    gt=np.nan_to_num(dos[vsel].astype(float),nan=1.0); sg=sgn[vsel].astype(float)
    v=gt.var(1)>0
    if not v.any(): continue
    gt,sg=gt[v],sg[v]
    a_t,t_t,va_t,vt_t=Tt(A[j]),Tt(T[j]),Tt(Va[j]),Tt(Vt[j])
    out={}
    for name,tm in (('shipped','estimate'),('zero','zero')):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            wa,wt,rA,rT,info=HM._prepare_channels(a_t,t_t,va_t,vt_t,cov_t,tm,dev,
                                                  ase_covariates_t=None,return_info=True)
            o=HM.calculate_hapmixqtl_nominal(Tt(gt),Tt(sg),a_t,t_t,wa,wt,rA,rT)
        out[name]=(o[4].cpu().numpy(), o[6].cpu().numpy(), info)  # se_a, se_t
    ok=np.isfinite(out['shipped'][0])&np.isfinite(out['zero'][0])&(out['shipped'][0]>0)
    if ok.sum()>20:
        ra_.append(np.median(out['zero'][0][ok]/out['shipped'][0][ok]))
    ok2=np.isfinite(out['shipped'][1])&np.isfinite(out['zero'][1])&(out['shipped'][1]>0)
    if ok2.sum()>20:
        rt_.append(np.median(out['zero'][1][ok2]/out['shipped'][1][ok2]))
    ta_.append(out['shipped'][2]['tau_a']); tt_.append(out['shipped'][2]['tau_t'])
ta_=np.array([x for x in ta_ if x is not None]); tt_=np.array([x for x in tt_ if x is not None])
print('median tau_a %.5f   median median(Va) %.5f   ratio tau_a/Va %.2f'%(np.median(ta_),np.median(np.median(Va,1)),np.median(ta_)/np.median(np.median(Va,1))))
print('median tau_t %.5f   median median(Vt) %.5f   ratio tau_t/Vt %.1f'%(np.median(tt_),np.median(np.median(Vt,1)),np.median(tt_)/np.median(np.median(Vt,1))))
print()
print('reported SE, tau_mode=zero / tau_mode=estimate (both known-variance):')
print('  ALLELIC channel  median ratio %.4f  (%d genes)'%(np.median(ra_),len(ra_)))
print('  TOTAL   channel  median ratio %.4f  (%d genes)'%(np.median(rt_),len(rt_)))

"""Is the flat slope of Salmon's half-depth allelic ratio on its full-depth ratio (salmon_half_depth_20260927) shrinkage
or unshared noise? Shrinkage narrows the half-depth spread and puts the reverse regression (full on half) above 1;
noise in both runs puts both regressions below 1. Per band, over records two-sided at both depths (thinned columns:
two-sided at full depth and after the benchmark's thinning), from that record's per_gene.tsv.

  python3 scripts/salmon_half_depth_dilution.py
"""
import numpy as np
import pandas as pd

PER_GENE = '/mnt/ssd/lalli/brainvar_hapmix_deploy/salmon_half_depth_20260927/per_gene.tsv'
KAPPA, MIN = 0.5, 0.5
BANDS = ('1-29', '30-99', '100-999', '1000+')


def ratio(L, R):
    return np.log2((L + KAPPA) / (R + KAPPA))


def slope(y, x):
    return np.polyfit(x, y, 1)[0]


d = pd.read_csv(PER_GENE, sep='\t')
full, half, thin = ratio(d.pL_full, d.pR_full), ratio(d.pL_half, d.pR_half), ratio(d.pL_thin, d.pR_thin)
two = (d.pL_full >= MIN) & (d.pR_full >= MIN) & (d.pL_half >= MIN) & (d.pR_half >= MIN)
two_t = (d.pL_full >= MIN) & (d.pR_full >= MIN) & (d.pL_thin >= MIN) & (d.pR_thin >= MIN)
print('band     records | sd full  sd half | half on full  full on half  corr | thinned: sd  on full  full on it')
for b in BANDS:
    s, t = (d.band == b) & two, (d.band == b) & two_t
    print(f'{b:8s} {int(s.sum()):7d} | {full[s].std():.3f}    {half[s].std():.3f}  | {slope(half[s], full[s]):.3f}         '
          f'{slope(full[s], half[s]):.3f}         {np.corrcoef(full[s], half[s])[0, 1]:.3f} | {thin[t].std():.3f}  '
          f'{slope(thin[t], full[t]):.3f}    {slope(full[t], thin[t]):.3f}')

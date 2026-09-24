"""Does hapmixQTL's miscalibration track how heterogeneous its weights are?

One pre-specified hypothesis, not a fishing expedition.

RASQUAL estimates its dispersion structure per gene from the data and re-fits it
on every run: theta (beta-binomial overdispersion), phi (reference bias), delta
(error rate). Under permutation it re-fits them, so whatever extra variance is
present gets absorbed and the likelihood-ratio statistic keeps its reference
distribution.

hapmixQTL does something different. It takes the Gibbs across-draw variance v as
a FIXED per-donor SHAPE and fits only one scalar, sigma^2, per variant. A single
scale can repair a uniform error in v; it cannot repair a SHAPE error, where
some donors are mis-weighted relative to others. CLAUDE.md already records that
limit for the simulation benchmark -- robustness was demonstrated for a uniform
scale error only.

PREDICTION. If the miscalibration is a shape error, it should be worse in genes
whose v varies most across donors, because that is where a wrong shape has the
most room to act. If it is uniform mis-scaling, per-gene heterogeneity of v
should not predict it, since sigma^2 absorbs scale by construction.

The per-gene null rate over 30 draws is coarse, so this is a correlation across
46 genes and is reported with that limit attached.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

D = Path('/mnt/ssd/lalli/brainvar_hapmix_deploy')
CACHE = D / 'cache/gibbs_56b63c3b37ed5df8'

lg = pd.read_csv(D / 'realized_variance_20260924/null_long.tsv', sep='\t')
strata = pd.read_csv(D / 'genes_59_strata_20260923.tsv', sep='\t').set_index('gene')

# per-gene miscalibration: excess of small p relative to uniform
per = []
for arm in ('hapmixQTL', 'mixQTL', 'RASQUAL'):
    sub = lg[lg.arm == arm]
    for g, s in sub.groupby('gene'):
        p = s.pval.dropna().values
        if len(p) >= 10:
            per.append(dict(arm=arm, gene=g, n=len(p),
                            frac_lt_10=float((p < 0.10).mean()),
                            mean_neglog=float(np.mean(-np.log10(np.clip(p, 1e-12, 1))))))
per = pd.DataFrame(per)

# per-gene heterogeneity of the Gibbs variance across donors
genes_all = np.array(open(CACHE / 'genes.txt').read().split())
gi = {g: i for i, g in enumerate(genes_all)}
YL = np.load(CACHE / 'YL.npy', mmap_mode='r')
YR = np.load(CACHE / 'YR.npy', mmap_mode='r')
YT = np.load(CACHE / 'YT.npy', mmap_mode='r')

rows = []
for g in per[per.arm == 'hapmixQTL'].gene.unique():
    if g not in gi:
        continue
    i = gi[g]
    mL, mR = np.asarray(YL[i]).mean(1), np.asarray(YR[i]).mean(1)
    # the allelic log-ratio's across-draw variance, the quantity used as the
    # per-donor shape; restricted to donors carrying allelic information
    with np.errstate(divide='ignore', invalid='ignore'):
        lr = np.log((np.asarray(YL[i]) + 0.5) / (np.asarray(YR[i]) + 0.5))
    v = lr.var(axis=1, ddof=0)
    inf = (mL + mR) > 0
    vv = v[inf & np.isfinite(v) & (v > 0)]
    tt = np.asarray(YT[i]).mean(1)
    rows.append(dict(gene=g, n_inf=int(inf.sum()),
                     sd_logv=float(np.std(np.log(vv))) if len(vv) > 5 else np.nan,
                     med_reads=float(np.median((mL + mR)[inf])) if inf.any() else np.nan,
                     med_total=float(np.median(tt))))
het = pd.DataFrame(rows)

t = per[per.arm == 'hapmixQTL'].merge(het, on='gene')
t['stratum'] = [strata.loc[g, 'stratum'] for g in t.gene]
print(f'{len(t)} genes\n')
print('spread of log v across donors, within gene:')
print(f'  median {t.sd_logv.median():.3f}   range {t.sd_logv.min():.3f} - {t.sd_logv.max():.3f}')
print(f'\nper-gene null rate p<0.10 (expected 0.10): median {t.frac_lt_10.median():.3f}')

print('\nPRE-SPECIFIED TEST -- does weight heterogeneity predict miscalibration?')
for y in ('frac_lt_10', 'mean_neglog'):
    m = t[y].notna() & t.sd_logv.notna()
    r, p = sps.spearmanr(t.loc[m, 'sd_logv'], t.loc[m, y])
    print(f'  spearman(sd_logv, {y:12s}) = {r:+.3f}   p = {p:.3g}   n={m.sum()}')

print('\ncontrol -- the same against read depth, which sigma^2 CAN absorb:')
for y in ('frac_lt_10', 'mean_neglog'):
    m = t[y].notna() & t.med_reads.notna()
    r, p = sps.spearmanr(np.log10(t.loc[m, 'med_reads'] + 1), t.loc[m, y])
    print(f'  spearman(log reads, {y:12s}) = {r:+.3f}   p = {p:.3g}   n={m.sum()}')

print('\nby tertile of weight heterogeneity:')
t['het3'] = pd.qcut(t.sd_logv, 3, labels=['low', 'mid', 'high'])
print(t.groupby('het3', observed=True)[['sd_logv', 'frac_lt_10']].median().round(3).to_string())

print('\nsame per-gene rate, by arm (median over genes):')
print(per.groupby('arm').frac_lt_10.median().round(3).to_string())

t.to_csv(D / 'realized_variance_20260924/miscalibration_vs_heterogeneity.tsv',
         sep='\t', index=False)
print('\nwrote miscalibration_vs_heterogeneity.tsv')

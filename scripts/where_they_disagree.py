"""Where and how the two arms disagree, on the observed pass only.

Read-only over observed_matched_variants.parquet. No permutation, no null,
no model fitting: every number is a descriptive summary of estimates the two
arms already produced. Stratifies the disagreement so it can be located
rather than only quantified.
"""

import json

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

D = '/mnt/ssd/lalli/brainvar_hapmix_deploy/mixqtl_replication_20260919'


def r_of(d):
    if len(d) < 5 or d.hm_beta.std() == 0 or d.mx_beta.std() == 0:
        return np.nan
    return pearsonr(d.hm_beta, d.mx_beta)[0]


def main():
    m = pd.read_parquet(f'{D}/observed_matched_variants.parquet')
    out = {}

    print(f'{len(m):,} matched variants, {m.gene.nunique()} genes\n')

    print('=== HOW: magnitude and spread ===')
    for lbl, a, b in [('|beta|', m.hm_beta.abs(), m.mx_beta.abs()),
                      ('SE', m.hm_se, m.mx_se)]:
        print(f'  median {lbl:6s}  hapmixQTL {a.median():.5f}   '
              f'mixQTL {b.median():.5f}   ratio {b.median() / a.median():.3f}')
    sr = m.mx_se / m.hm_se
    print(f'  per-variant SE ratio mixQTL/hapmixQTL: median {sr.median():.3f}, '
          f'IQR {sr.quantile(.25):.3f}-{sr.quantile(.75):.3f}, '
          f'{100 * (sr > 1).mean():.1f}% above 1')
    out['se_ratio_median'] = float(sr.median())
    out['se_ratio_frac_above_1'] = float((sr > 1).mean())

    print('\n=== HOW: agreement overall ===')
    r_all = pearsonr(m.hm_beta, m.mx_beta)[0]
    slope = float(np.polyfit(m.hm_beta, m.mx_beta, 1)[0])
    sign = float((np.sign(m.hm_beta) == np.sign(m.mx_beta)).mean())
    sign_pred = 1 - np.arccos(np.clip(r_all, -1, 1)) / np.pi
    print(f'  Pearson r on beta            {r_all:.4f}')
    print(f'  regression slope mx on hm    {slope:.4f}')
    print(f'  sign concordance             {sign:.4f}')
    print(f'  ... predicted from r alone   {sign_pred:.4f}   '
          f'(bivariate normal centred at 0)')
    out.update(beta_r=float(r_all), slope=slope, sign_conc=sign,
               sign_conc_predicted_from_r=float(sign_pred))

    print('\n=== WHERE: by which channel mixQTL used ===')
    for meth, d in m.groupby('mx_method'):
        print(f'  {meth:5s}  n={len(d):7,}  r={r_of(d):.4f}  '
              f'median SE ratio {(d.mx_se / d.hm_se).median():.3f}')
        out[f'r_method_{meth}'] = None if np.isnan(r_of(d)) else float(r_of(d))

    print('\n=== WHERE: by allelic donor count mixQTL admitted ===')
    m['asc_bin'] = pd.cut(m.mx_n_asc, [0, 40, 55, 65, 75, 92],
                          include_lowest=True)
    for b, d in m.groupby('asc_bin', observed=True):
        print(f'  n_asc {str(b):14s} n={len(d):7,}  r={r_of(d):.4f}  '
              f'median SE ratio {(d.mx_se / d.hm_se).median():.3f}')

    print('\n=== WHERE: per gene, best and worst agreement ===')
    pg = m.groupby('gene').apply(
        lambda d: pd.Series({'n': len(d), 'r': r_of(d),
                             'se_ratio': (d.mx_se / d.hm_se).median(),
                             'n_asc': d.mx_n_asc.iloc[0]}),
        include_groups=False).reset_index().sort_values('r')
    print(pg.head(5).to_string(index=False))
    print('  ...')
    print(pg.tail(5).to_string(index=False))
    print(f'\n  per-gene r: median {pg.r.median():.3f}, '
          f'range {pg.r.min():.3f} to {pg.r.max():.3f}')
    out['per_gene_r_median'] = float(pg.r.median())
    out['per_gene_r_min'] = float(pg.r.min())
    out['per_gene_r_max'] = float(pg.r.max())
    print(f'  corr(per-gene r, allelic donor count) = '
          f'{pg.r.corr(pg.n_asc):.3f}')
    out['corr_gene_r_with_n_asc'] = float(pg.r.corr(pg.n_asc))

    print('\n=== WHERE: does agreement depend on effect size? ===')
    m['mag_bin'] = pd.qcut(m.hm_beta.abs(), 5, duplicates='drop')
    for b, d in m.groupby('mag_bin', observed=True):
        print(f'  |hm beta| {str(b):22s} n={len(d):7,}  r={r_of(d):.4f}')

    print('\n=== WHERE: the top-ranked variants, which is what a scan returns ===')
    top = m.loc[m.groupby('gene').hm_stat.idxmax()]
    print(f'  at hapmixQTL leads (29): beta r = {r_of(top):.4f}, '
          f'sign concordance {(np.sign(top.hm_beta) == np.sign(top.mx_beta)).mean():.3f}')
    q = m.hm_stat.quantile(0.99)
    d99 = m[m.hm_stat >= q]
    print(f'  top 1% by hapmixQTL stat (n={len(d99):,}): r = {r_of(d99):.4f}')
    out['r_at_hm_leads'] = float(r_of(top))
    out['r_top1pct'] = float(r_of(d99))

    json.dump(out, open(f'{D}/where_they_disagree.json', 'w'), indent=1)


if __name__ == '__main__':
    main()

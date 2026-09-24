"""Scale-free precision: observed effect over null spread, at the same variant.

The raw null sd of beta_hat is NOT comparable across these arms, and reporting
it as precision would be wrong. mixQTL's response is log2 with a kappa=0.5
pseudocount and hapmixQTL's is natural log; the pseudocount COMPRESSES the
response, and a compressed response shrinks beta and its null spread by the same
factor. mixQTL's null sd came in 0.804x hapmixQTL's while its observed betas
came in 0.424/0.611 = 0.69x hapmixQTL's against the same RASQUAL reference --
the same compression showing up twice. A smaller spread on a compressed scale is
not less error.

The ratio of observed effect to null spread is invariant to that compression: if
an arm scales beta and sd by the same k, the ratio is unchanged. It is the
operational meaning of precision -- how far the signal sits from the noise the
method itself generates under a null.

Read at RASQUAL's lead, which RASQUAL selected, so RASQUAL's own numerator
carries a winner's curse the other two do not. hapmixQTL vs mixQTL is the clean
pair here; RASQUAL is shown with that caveat attached.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

D = Path('/mnt/ssd/lalli/brainvar_hapmix_deploy')
RUN = D / 'rasqual_default_mode_20260923'

sd = pd.read_csv(D / 'realized_variance_20260924/null_beta_sd.tsv', sep='\t')
me = pd.read_csv(RUN / 'matched_effects.tsv', sep='\t')
mx = pd.read_csv(D / 'mixqtl_rasqual_control_20260924/permissive/three_way_matched_59.tsv',
                 sep='\t')[['gene', 'mx_at_r']]
t = sd.merge(me[['gene', 'afc_r', 'afc_h_at_r']], on='gene').merge(mx, on='gene')

t['snr_hm'] = np.abs(t.afc_h_at_r) / t.sd_hm
t['snr_mx'] = np.abs(t.mx_at_r) / t.sd_mx
t['snr_rq'] = np.abs(t.afc_r) / t.sd_rq

print(f'{len(t)} genes at RASQUAL\'s lead\n')
print('median |observed beta| / null sd  (higher = signal further from the '
      'method\'s own noise):')
for k, lab in (('hm', 'hapmixQTL'), ('mx', 'mixQTL'), ('rq', 'RASQUAL (curse-inflated)')):
    v = t[f'snr_{k}'].dropna()
    print(f'  {lab:26s} n={len(v):3d}  median {v.median():5.3f}   '
          f'IQR {v.quantile(.25):.3f}-{v.quantile(.75):.3f}')

res = {}
print('\npaired, same gene and same variant:')
for a, b, la, lb in (('hm', 'mx', 'hapmixQTL', 'mixQTL'),
                     ('hm', 'rq', 'hapmixQTL', 'RASQUAL')):
    m = t[f'snr_{a}'].notna() & t[f'snr_{b}'].notna()
    r = (t.loc[m, f'snr_{a}'] / t.loc[m, f'snr_{b}']).values
    wins = int((r > 1).sum())
    p = sps.binomtest(wins, len(r), 0.5).pvalue
    print(f'  {la} / {lb}: n={len(r):3d}  median {np.median(r):5.3f}   '
          f'{la} higher on {wins}/{len(r)}   sign p={p:.3g}')
    res[f'snr_{a}_over_{b}'] = dict(n=len(r), median=float(np.median(r)),
                                    n_first_higher=wins, sign_p=float(p))

print('\nby stratum, hapmixQTL / mixQTL:')
for s in ['HIGH', 'MID', 'LOW']:
    sub = t[(t.stratum == s) & t.snr_hm.notna() & t.snr_mx.notna()]
    if len(sub) >= 4:
        r = (sub.snr_hm / sub.snr_mx).values
        print(f'  {s:4s} n={len(r):2d}  median {np.median(r):5.3f}  '
              f'hapmixQTL higher on {int((r > 1).sum())}/{len(r)}')

print('\ncompression check -- the thing that makes raw sd uncomparable:')
for k, lab in (('hm', 'hapmixQTL'), ('mx', 'mixQTL')):
    b = t[f'afc_h_at_r'] if k == 'hm' else t['mx_at_r']
    print(f'  {lab:10s} median |observed beta| {np.nanmedian(np.abs(b)):.4f}   '
          f'median null sd {t[f"sd_{k}"].median():.4f}')

out = D / 'realized_variance_20260924'
t.to_csv(out / 'null_snr.tsv', sep='\t', index=False)
(out / 'snr_summary.json').write_text(json.dumps(res, indent=2, default=float))
print(f'\nwrote {out}/null_snr.tsv')

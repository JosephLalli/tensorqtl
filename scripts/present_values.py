"""Actual values, with the observed runs and the null runs kept apart.

No ratios. Each arm's numbers stand on their own so they can be read directly.

OBSERVED and NULL are different experiments and are never mixed here. The
observed section is one estimate per gene at a named variant. The null section
is 20 permutations per gene at the SAME variant, where the true slope is zero,
and is summarised as the distribution of the nominal p-value -- which should be
uniform if an arm's nominal scale is calibrated at a fixed variant. A count of
"hits" is not reported: it collapses a distribution to one threshold and is what
made the earlier detection tables move around under more permutations.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

D = Path('/mnt/ssd/lalli/brainvar_hapmix_deploy')
RUN = D / 'rasqual_default_mode_20260923'
OUT = D / 'values_report_20260924'
OUT.mkdir(exist_ok=True)


def q(v):
    v = np.asarray(pd.to_numeric(v, errors='coerce'), float)
    v = v[np.isfinite(v)]
    if not len(v):
        return None
    return dict(n=int(len(v)), median=float(np.median(v)),
                q25=float(np.quantile(v, .25)), q75=float(np.quantile(v, .75)),
                mean=float(v.mean()))


def wald_se(beta, stat):
    b, s = np.asarray(beta, float), np.asarray(stat, float)
    with np.errstate(invalid='ignore', divide='ignore'):
        se = np.abs(b) / np.sqrt(s)
    se[~np.isfinite(se) | (s <= 0)] = np.nan
    return se


def row(name, d, unit=''):
    if d is None:
        print(f'  {name:34s}  --')
        return
    print(f'  {name:34s}  n={d["n"]:3d}   median {d["median"]:9.4f}   '
          f'IQR {d["q25"]:8.4f} .. {d["q75"]:8.4f}{unit}')


me = pd.read_csv(RUN / 'matched_effects.tsv', sep='\t')
mx = pd.read_csv(D / 'mixqtl_rasqual_control_20260924/permissive/three_way_matched_59.tsv',
                 sep='\t')
t = me.merge(mx[['gene', 'mx_at_h', 'mx_se_at_h', 'mx_at_r', 'mx_se_at_r',
                 'mx_method_at_r']], on='gene')
t['se_h_at_h'] = wald_se(t.afc_h, t.stat_h)
t['se_r_at_h'] = wald_se(t.afc_r_at_h, t.stat_r_at_h)
t['se_r_at_r'] = wald_se(t.afc_r, t.stat_r)
t['se_h_at_r'] = wald_se(t.afc_h_at_r, t.stat_h_at_r)

res = {}
print('=' * 74)
print('OBSERVED RUN -- one estimate per gene, natural-log allelic fold change')
print('=' * 74)
print('\nAt RASQUAL\'s lead variant (RASQUAL selected it; the other two did not):')
obs_r = {
    'hapmixQTL |beta|': q(np.abs(t.afc_h_at_r)), 'hapmixQTL se': q(t.se_h_at_r),
    'mixQTL |beta|': q(np.abs(t.mx_at_r)), 'mixQTL se': q(t.mx_se_at_r),
    'RASQUAL |beta|': q(np.abs(t.afc_r)), 'RASQUAL se': q(t.se_r_at_r)}
for k, v in obs_r.items():
    row(k, v)
res['observed_at_rasqual_lead'] = obs_r

print('\nAt hapmixQTL\'s lead variant (hapmixQTL selected it):')
obs_h = {
    'hapmixQTL |beta|': q(np.abs(t.afc_h)), 'hapmixQTL se': q(t.se_h_at_h),
    'mixQTL |beta|': q(np.abs(t.mx_at_h)), 'mixQTL se': q(t.mx_se_at_h),
    'RASQUAL |beta|': q(np.abs(t.afc_r_at_h)), 'RASQUAL se': q(t.se_r_at_h)}
for k, v in obs_h.items():
    row(k, v)
res['observed_at_hapmixqtl_lead'] = obs_h

print('\n' + '=' * 74)
print('NULL RUN -- 20 permutations per gene at the SAME fixed variant')
print('=' * 74)
lg = pd.read_csv(D / 'realized_variance_20260924/null_long.tsv', sep='\t')
sd = pd.read_csv(D / 'realized_variance_20260924/null_beta_sd.tsv', sep='\t')

print('\nspread of beta_hat across permutations (per-gene sd, then summarised):')
for k, lab in (('sd_hm', 'hapmixQTL'), ('sd_mx', 'mixQTL'), ('sd_rq', 'RASQUAL')):
    row(f'{lab} null sd of beta', q(sd[k]))
res['null_beta_sd'] = {lab: q(sd[k]) for k, lab in
                       (('sd_hm', 'hapmixQTL'), ('sd_mx', 'mixQTL'), ('sd_rq', 'RASQUAL'))}

print('\nnominal p-value at the fixed variant, decile histogram.')
print('Under the null a calibrated nominal scale is UNIFORM: ~10% per decile.')
edges = np.arange(0, 1.01, 0.1)
hist = {}
for arm, sub in lg.groupby('arm'):
    p = sub.pval.dropna().values
    cnt, _ = np.histogram(p, bins=edges)
    frac = cnt / cnt.sum()
    ks = sps.kstest(p, 'uniform')
    hist[arm] = dict(n=int(len(p)), counts=cnt.tolist(), frac=frac.round(4).tolist(),
                     frac_below_05=float((p < 0.05).mean()),
                     frac_below_01=float((p < 0.01).mean()),
                     median_p=float(np.median(p)),
                     ks_stat=float(ks.statistic), ks_p=float(ks.pvalue))
    print(f'\n  {arm}  (n={len(p)})')
    print('    decile:  ' + ' '.join(f'{x:5.1f}%' for x in frac * 100))
    print(f'    median p {np.median(p):.3f}   p<0.05 {(p < 0.05).mean():.3f}   '
          f'p<0.01 {(p < 0.01).mean():.3f}   KS vs uniform D={ks.statistic:.3f}, '
          f'p={ks.pvalue:.2e}')
res['null_pvalue_histogram'] = hist

(OUT / 'values.json').write_text(json.dumps(res, indent=2, default=float))
t.to_csv(OUT / 'observed_matched_values.tsv', sep='\t', index=False)
print(f'\nwrote {OUT}')

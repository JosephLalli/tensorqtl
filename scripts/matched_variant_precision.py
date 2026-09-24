"""Precision head-to-head at matched variants: whose standard error is smaller?

This is the measure that bears on hapmixQTL's claim. The claim is that carrying
the quantifier's inferential uncertainty into the standard error buys PRECISION
-- not that hapmixQTL agrees with RASQUAL about effect size, and not that it
calls more genes. Effect-size agreement and detection counts are reported
elsewhere and are not evidence for or against the claim.

WHY STANDARD ERRORS AND NOT EFFECT SIZES. At the variant an arm selected, that
arm's beta is inflated by the winner's curse: the lead is the maximum over a
~1 Mb window of its own noisy statistic. The standard error is far less
sensitive, because a standard error is not selected on directly -- and to first
order the inflation cancels, since stat ~ (beta/se)^2, so a beta inflated by c
arrives with a statistic inflated by about c^2 and leaves se unchanged. The
comparison is still reported at both leads so that any residual asymmetry is
visible rather than assumed away.

SE IS RECOVERED BY THE WALD IDENTITY, se = |beta| / sqrt(stat), because neither
hapmixQTL's nor RASQUAL's gene-level output carries an explicit standard error.
mixQTL reports beta, se AND stat, so the identity is CHECKED here against its
own three columns rather than taken on trust.

All three arms are on the natural-log allelic-fold-change scale: RASQUAL through
log(pi/(1-pi)), hapmixQTL natively, mixQTL by x ln2 from its log2 response.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

D = Path('/mnt/ssd/lalli/brainvar_hapmix_deploy')
RUN = D / 'rasqual_default_mode_20260923'


def wald_se(beta, stat):
    beta, stat = np.asarray(beta, float), np.asarray(stat, float)
    with np.errstate(invalid='ignore', divide='ignore'):
        se = np.abs(beta) / np.sqrt(stat)
    se[~np.isfinite(se) | (stat <= 0)] = np.nan
    return se


def ratio_block(num, den, num_name, den_name, label):
    """Paired ratio of standard errors at the SAME variant.

    Reported as a median with a sign test on log ratio, because SE ratios are
    right-skewed and a mean would be led by a few genes. The sign test asks the
    only question that matters here: on how many genes is one arm tighter.
    """
    num, den = np.asarray(num, float), np.asarray(den, float)
    m = np.isfinite(num) & np.isfinite(den) & (num > 0) & (den > 0)
    if m.sum() < 5:
        print(f'  {label:44s} n={m.sum()} -- too few')
        return None
    r = num[m] / den[m]
    wins = int((r > 1).sum())
    from scipy import stats as sps
    p = sps.binomtest(wins, int(m.sum()), 0.5).pvalue
    print(f'  {label:44s} n={m.sum():3d}  median {num_name}/{den_name} '
          f'{np.median(r):5.3f}   {den_name} tighter on {wins}/{m.sum()}  '
          f'sign p={p:.3g}')
    return dict(n=int(m.sum()), median_ratio=float(np.median(r)),
                iqr=[float(np.quantile(r, .25)), float(np.quantile(r, .75))],
                n_denominator_tighter=wins, sign_p=float(p))


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else 'permissive'
    me = pd.read_csv(RUN / 'matched_effects.tsv', sep='\t')
    mx = pd.read_csv(D / f'mixqtl_rasqual_control_20260924/{which}/three_way_matched_59.tsv',
                     sep='\t')
    st = pd.read_csv(D / 'genes_59_strata_20260923.tsv', sep='\t').set_index('gene')
    t = me.merge(mx[['gene', 'mx_at_h', 'mx_se_at_h', 'mx_stat_at_h', 'mx_method_at_h',
                     'mx_at_r', 'mx_se_at_r', 'mx_stat_at_r', 'mx_method_at_r']],
                 on='gene')
    t['stratum'] = [st.loc[g, 'stratum'] if g in st.index else '?' for g in t.gene]
    print(f'cutoffs: {which};  {len(t)} genes\n')

    # --- the identity, on mixQTL's own three columns, in the RIGHT form -------
    # The arms do not report the same kind of statistic and the difference is
    # easy to miss, because the wrong form still yields plausible numbers:
    #   mixQTL     stat is a SIGNED Z  -> se = |beta| / |stat|
    #   hapmixQTL  stat is T^2         -> se = |beta| / sqrt(stat), exact
    #   RASQUAL    stat is 2 log LR    -> se = |beta| / sqrt(stat), the Wald
    #              APPROXIMATION to a likelihood ratio, not an identity
    # Applying sqrt to mixQTL's Z gave a median relative error of 0.55 and is
    # what exposed this. mixQTL's reported se is used directly below; only
    # hapmixQTL and RASQUAL are derived.
    m = (np.isfinite(t.mx_stat_at_h) & np.isfinite(t.mx_se_at_h) & (t.mx_se_at_h > 0))
    pred = np.abs(t.mx_at_h[m]) / np.abs(t.mx_stat_at_h[m])
    rel = np.abs(pred - t.mx_se_at_h[m]) / t.mx_se_at_h[m]
    print(f'identity check on mixQTL (|beta|/|Z| vs its reported se): n={int(m.sum())}, '
          f'max relative error {np.nanmax(rel):.3e}')
    assert np.nanmax(rel) < 1e-8, 'mixQTL se is not |beta|/|stat|; re-derive before trusting'
    print('  -> hapmixQTL se = |beta|/sqrt(T^2) is exact by construction; '
          'RASQUAL se = |beta|/sqrt(chi2) is a Wald approximation to an LRT')

    t['se_h_at_h'] = wald_se(t.afc_h, t.stat_h)
    t['se_r_at_h'] = wald_se(t.afc_r_at_h, t.stat_r_at_h)
    t['se_r_at_r'] = wald_se(t.afc_r, t.stat_r)
    t['se_h_at_r'] = wald_se(t.afc_h_at_r, t.stat_h_at_r)

    res = {'cutoffs': which, 'n_genes': int(len(t))}
    print('\n=== PRIMARY: at RASQUAL\'s lead (hapmixQTL and mixQTL did not choose it) ===')
    res['at_rasqual_lead'] = {
        'RASQUAL_over_hapmixQTL': ratio_block(t.se_r_at_r, t.se_h_at_r, 'RASQUAL', 'hapmixQTL',
                                              'RASQUAL vs hapmixQTL'),
        'mixQTL_over_hapmixQTL': ratio_block(t.mx_se_at_r, t.se_h_at_r, 'mixQTL', 'hapmixQTL',
                                             'mixQTL vs hapmixQTL'),
        'RASQUAL_over_mixQTL': ratio_block(t.se_r_at_r, t.mx_se_at_r, 'RASQUAL', 'mixQTL',
                                           'RASQUAL vs mixQTL')}

    print('\n=== SECONDARY: at hapmixQTL\'s lead (hapmixQTL chose it) ===')
    res['at_hapmixqtl_lead'] = {
        'RASQUAL_over_hapmixQTL': ratio_block(t.se_r_at_h, t.se_h_at_h, 'RASQUAL', 'hapmixQTL',
                                              'RASQUAL vs hapmixQTL'),
        'mixQTL_over_hapmixQTL': ratio_block(t.mx_se_at_h, t.se_h_at_h, 'mixQTL', 'hapmixQTL',
                                             'mixQTL vs hapmixQTL')}

    print('\n=== by stratum, at RASQUAL\'s lead ===')
    res['by_stratum'] = {}
    for s in ['HIGH', 'MID', 'LOW']:
        sub = t[t.stratum == s]
        if len(sub) >= 5:
            print(f'  -- {s} (n={len(sub)}) --')
            res['by_stratum'][s] = {
                'RASQUAL_over_hapmixQTL': ratio_block(sub.se_r_at_r, sub.se_h_at_r,
                                                      'RASQUAL', 'hapmixQTL', f'{s}: RASQUAL vs hapmixQTL'),
                'mixQTL_over_hapmixQTL': ratio_block(sub.mx_se_at_r, sub.se_h_at_r,
                                                     'mixQTL', 'hapmixQTL', f'{s}: mixQTL vs hapmixQTL')}

    out = D / f'matched_variant_precision_20260924/{which}'
    out.mkdir(parents=True, exist_ok=True)
    t.to_csv(out / 'matched_variant_se.tsv', sep='\t', index=False)
    (out / 'precision_summary.json').write_text(json.dumps(res, indent=2, default=float))
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()

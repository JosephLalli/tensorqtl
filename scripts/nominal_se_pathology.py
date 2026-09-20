"""For pval_nominal specifically: how many per-variant SEs are pathological?

pval_nominal takes no maximum, so the selection effect measured elsewhere
does not apply. The only question is whether each variant's standard error
is right.

The SE collapse at low-carrier variants still occurs without selection; it
is simply not amplified. This counts how often it occurs and how far the
affected nominal p-values move, so the exposure can be judged rather than
assumed.

Carrier count is the number of donors with s = xL - xR nonzero, i.e.
heterozygous. The sandwich sums residuals only over those, so it is the
variable that governs how many terms the variance estimate has.

Observed pass; no permutation, no selection.
"""

import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, 'scripts')
sys.path.insert(0, '.')

OUT = '/mnt/ssd/lalli/brainvar_hapmix_deploy/mixqtl_replication_20260919'


def main():
    import tensorqtl.hapmixqtl as HM
    from compare_mixqtl_replication import load_inputs, gene_variant_index
    from se_forms_calibration import fit_all_se

    I = load_inputs()
    genes, keep = I['genes'], I['keep']
    A, _T, Va, _Vt, _C = HM.compute_summaries_from_gibbs(
        I['YL'], I['YR'], yT=I['YT'], count_noise=True)
    A, Va = A[:, keep], Va[:, keep]
    s_all = (I['xL'] - I['xR'])[I['idx']][:, keep]

    parts = []
    for j, g in enumerate(genes):
        vsel = gene_variant_index(I, g)
        if vsel.size == 0:
            continue
        inf = Va[j] > 1e-12
        if int(inf.sum()) <= 2:
            continue
        a = A[j][inf]
        S = s_all[vsel][:, inf]
        S = S[S.var(1) > 0]
        if S.shape[0] == 0:
            continue
        X = S.T
        w = 1.0 / np.maximum(Va[j][inf], 1e-12)
        beta, se2_known, se2_fit, se2_sand, se2_hc3 = fit_all_se(a, X, w)
        nz = (X != 0).sum(0)
        ok = np.isfinite(beta) & (se2_fit > 0) & (se2_sand > 0)
        parts.append(pd.DataFrame(dict(
            gene=g, carriers=nz[ok],
            ratio=np.sqrt(se2_sand[ok] / se2_fit[ok]),
            ratio3=np.sqrt(se2_hc3[ok] / se2_fit[ok]))))
    d = pd.concat(parts, ignore_index=True)
    d.to_parquet(f'{OUT}/nominal_se_pathology.parquet')

    print(f'{len(d):,} variants across {d.gene.nunique()} genes\n')
    print('sandwich se / fitted-sigma se, all variants, no selection:')
    for lab, col in [('HC0', 'ratio'), ('HC3', 'ratio3')]:
        r = d[col]
        print(f'  {lab}  median {r.median():.3f}   '
              f'1st pct {r.quantile(.01):.3f}   '
              f'below 0.5: {100 * (r < 0.5).mean():5.2f}%   '
              f'below 0.2: {100 * (r < 0.2).mean():5.2f}%')

    print('\nby carrier count (donors heterozygous for the variant):')
    d['bin'] = pd.cut(d.carriers, [-0.5, 4.5, 9.5, 14.5, 24.5, 1e9],
                      labels=['<=4', '5-9', '10-14', '15-24', '>=25'])
    for b, g in d.groupby('bin', observed=True):
        print(f'  {b:>6s} carriers  n={len(g):7,} ({100*len(g)/len(d):5.1f}%)'
              f'   median ratio {g.ratio.median():.3f}'
              f'   below 0.5: {100 * (g.ratio < 0.5).mean():5.2f}%')

    # what a collapsed SE does to a nominal p, holding the statistic fixed
    print('\nEffect on pval_nominal, for variants with ratio below 0.5:')
    bad = d[d.ratio < 0.5]
    print(f'  {len(bad):,} variants ({100 * len(bad) / len(d):.2f}%)')
    if len(bad):
        print(f'  median ratio {bad.ratio.median():.3f} -> t is inflated '
              f'{1 / bad.ratio.median():.1f}x, t^2 by '
              f'{1 / bad.ratio.median() ** 2:.1f}x')
        print(f'  median carrier count among them: {bad.carriers.median():.0f}')
    thresh = 15
    print(f'\n  a carrier-count filter at {thresh} would remove '
          f'{100 * (d.carriers < thresh).mean():.1f}% of variants and '
          f'{100 * (bad.carriers < thresh).mean():.1f}% of the affected ones')

    json.dump(dict(
        n_variants=int(len(d)),
        hc0_median=float(d.ratio.median()),
        hc0_pct_below_0p5=float(100 * (d.ratio < 0.5).mean()),
        hc0_pct_below_0p2=float(100 * (d.ratio < 0.2).mean()),
        hc3_median=float(d.ratio3.median()),
        hc3_pct_below_0p5=float(100 * (d.ratio3 < 0.5).mean()),
        median_carriers_affected=float(bad.carriers.median()) if len(bad) else None,
        pct_removed_by_carrier_filter_15=float(100 * (d.carriers < 15).mean()),
        pct_of_affected_removed=float(100 * (bad.carriers < 15).mean())
        if len(bad) else None,
    ), open(f'{OUT}/nominal_se_pathology.json', 'w'), indent=1)


if __name__ == '__main__':
    main()

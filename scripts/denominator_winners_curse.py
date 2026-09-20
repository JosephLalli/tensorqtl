"""Does selecting on max t^2 with a sandwich SE pick variants where that SE
happened to come out small? A Winner's Curse on the denominator.

Huang et al. 2018 establish selection bias in the NUMERATOR: take the
window maximum and its effect size is biased high. A residual-driven
standard error invites the same pathology in the denominator, because
max t^2 = beta^2 / se^2 rewards a small se just as much as a large beta,
and the sandwich se is an estimate from n residuals rather than a smooth
function of the design.

Test: within each gene, compare the ratio (sandwich se / model-based se) at
the selected lead against the distribution of that ratio over all variants
in the same gene. If selection is exploiting sandwich noise, the lead's
ratio sits systematically below its gene's median.

The model-based se here is the fitted-sigma form, which is smooth in the
design and so is not itself selectable-on-noise in the same way.

Observed pass only; no permutation.
"""

import json
import os
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
    if os.environ.get('HWE', '0') == '1':
        from make_hwe_filtered_variants import apply_hwe_filter
        I = apply_hwe_filter(I)
        print(f"[HWE-filtered variant set: dropped "
              f"{I['n_dropped_by_hwe']:,} variants]\n")
    else:
        print('[unfiltered variant set]\n')
    genes, keep = I['genes'], I['keep']
    A, _T, Va, _Vt, _C = HM.compute_summaries_from_gibbs(
        I['YL'], I['YR'], yT=I['YT'], count_noise=True)
    A, Va = A[:, keep], Va[:, keep]
    s_all = (I['xL'] - I['xR'])[I['idx']][:, keep]

    rows = []
    for j, g in enumerate(genes):
        vsel = gene_variant_index(I, g)
        if vsel.size == 0:
            continue
        inf = Va[j] > 1e-12
        n = int(inf.sum())
        if n <= 2:
            continue
        a = A[j][inf]
        S = s_all[vsel][:, inf]
        S = S[S.var(1) > 0]
        if S.shape[0] < 50:
            continue
        X = S.T
        w = 1.0 / np.maximum(Va[j][inf], 1e-12)

        beta, se2_known, se2_fit, se2_sand, se2_hc3 = fit_all_se(a, X, w)
        ok = np.isfinite(beta) & (se2_fit > 0) & (se2_sand > 0)
        if ok.sum() < 50:
            continue
        ratio = np.sqrt(se2_sand[ok] / se2_fit[ok])     # sandwich se / model se
        ratio3 = np.sqrt(se2_hc3[ok] / se2_fit[ok])
        t2_sand = beta[ok] ** 2 / se2_sand[ok]
        t2_hc3 = beta[ok] ** 2 / se2_hc3[ok]
        t2_fit = beta[ok] ** 2 / se2_fit[ok]

        lead_s = int(np.argmax(t2_sand))
        lead_3 = int(np.argmax(t2_hc3))
        lead_f = int(np.argmax(t2_fit))
        # how many donors actually carry this variant (s != 0)? The
        # sandwich meat sums only over those, so a variant carried by few
        # donors has few terms and high leverage.
        nz = (X != 0).sum(0)[ok]
        rows.append(dict(
            gene=g, n_var=int(ok.sum()),
            ratio_at_sandwich_lead=float(ratio[lead_s]),
            ratio_at_model_lead=float(ratio[lead_f]),
            gene_median_ratio=float(np.median(ratio)),
            gene_p05_ratio=float(np.percentile(ratio, 5)),
            pct_rank_of_lead=float(100 * (ratio < ratio[lead_s]).mean()),
            same_lead=bool(lead_s == lead_f),
            t2_ratio_lead=float(t2_sand[lead_s] / t2_fit[lead_s]),
            ratio3_at_hc3_lead=float(ratio3[lead_3]),
            gene_median_ratio3=float(np.median(ratio3)),
            pct_rank_lead3=float(100 * (ratio3 < ratio3[lead_3]).mean()),
            same_lead3=bool(lead_3 == lead_f),
            t2_ratio_lead3=float(t2_hc3[lead_3] / t2_fit[lead_3]),
            nz_at_sandwich_lead=int(nz[lead_s]),
            nz_median=float(np.median(nz)),
        ))

    d = pd.DataFrame(rows)
    d.to_csv(f'{OUT}/denominator_winners_curse.tsv', sep='\t', index=False)

    print(f'{len(d)} genes\n')
    print('sandwich se / model se, at the sandwich-selected lead vs the gene:')
    print(f'  median ratio at the lead        {d.ratio_at_sandwich_lead.median():.4f}')
    print(f'  median ratio over all variants  {d.gene_median_ratio.median():.4f}')
    print(f'  lead ratio as a percentile within its gene: '
          f'median {d.pct_rank_of_lead.median():.1f}th')
    print(f'  genes where the lead ratio is below the gene median: '
          f'{int((d.ratio_at_sandwich_lead < d.gene_median_ratio).sum())}/{len(d)}')
    print(f'\n  lead changes between model and sandwich SE: '
          f'{int((~d.same_lead).sum())}/{len(d)} genes')
    print(f'  t^2 inflation at the sandwich lead (sandwich/model): '
          f'median {d.t2_ratio_lead.median():.3f}')
    print(f'\n  donors carrying the sandwich lead variant (s != 0): '
          f'median {d.nz_at_sandwich_lead.median():.0f}')
    print(f'  donors carrying a typical variant: '
          f'median {d.nz_median.median():.0f}')
    print('\n--- HC3, which divides each squared residual by (1-h)^2 ---')
    print(f'  median ratio at the HC3 lead    {d.ratio3_at_hc3_lead.median():.4f}')
    print(f'  median ratio over all variants  {d.gene_median_ratio3.median():.4f}')
    print(f'  lead percentile within gene: median {d.pct_rank_lead3.median():.1f}th')
    print(f'  lead changes vs model SE: {int((~d.same_lead3).sum())}/{len(d)}')
    print(f'  t^2 inflation at the HC3 lead: median '
          f'{d.t2_ratio_lead3.median():.3f}')
    print('\n  a percentile well below 50 means selection is exploiting '
          'noise;\n  near 50 means it is not.')

    json.dump(dict(
        n_genes=int(len(d)),
        median_lead_ratio=float(d.ratio_at_sandwich_lead.median()),
        median_gene_ratio=float(d.gene_median_ratio.median()),
        median_lead_percentile=float(d.pct_rank_of_lead.median()),
        n_lead_below_median=int(
            (d.ratio_at_sandwich_lead < d.gene_median_ratio).sum()),
        n_lead_changed=int((~d.same_lead).sum()),
        median_t2_inflation=float(d.t2_ratio_lead.median()),
        median_carriers_at_lead=float(d.nz_at_sandwich_lead.median()),
        median_carriers_typical=float(d.nz_median.median()),
        hc3_median_lead_ratio=float(d.ratio3_at_hc3_lead.median()),
        hc3_median_lead_percentile=float(d.pct_rank_lead3.median()),
        hc3_median_t2_inflation=float(d.t2_ratio_lead3.median()),
        hc3_n_lead_changed=int((~d.same_lead3).sum()),
    ), open(f'{OUT}/denominator_winners_curse.json', 'w'), indent=1)


if __name__ == '__main__':
    main()

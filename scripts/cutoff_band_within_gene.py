"""Two controls the first band pass needed.

1. WITHIN-GENE band comparison. The pooled comparison confounds band with
   gene: the above-ceiling band is mostly high-expression genes and the
   in-band one mostly mid-expression, so a difference in excess dispersion
   between them could be a property of the genes rather than of the band.
   Restricting to genes that carry pairs in BOTH bands removes that.

2. ALLELIC-SLOPE comparison restricted to genes whose allelic channel is ON
   in both runs. Under the published cutoffs most genes fall below
   hapmixQTL's own sparse-channel rule and contribute slope_a = 0, which
   makes a pooled regression slope meaningless -- it measures how often the
   channel switched off, not whether the estimate shifted. Reads the
   parquets the first pass already wrote; nothing is re-run.

Writes cutoff_band_within_gene.json.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
D = '/mnt/ssd/lalli/brainvar_hapmix_deploy'
OUT = f'{D}/mixqtl_replication_20260919'

sys.path.insert(0, HERE)
sys.path.insert(0, REPO)


def read_slopes(tmp):
    d = pd.concat([pd.read_parquet(f'{tmp}/{f}') for f in os.listdir(tmp)
                   if f.endswith('.parquet')], ignore_index=True)
    d = d.rename(columns={'phenotype_id': 'gene'})
    d['variant_id'] = d.variant_id.astype(str)
    return d[['gene', 'variant_id', 'slope_a', 'slope_a_se']]


def main():
    import tensorqtl.hapmixqtl as HM
    from tensorqtl import mixqtl_replication as MX
    from compare_mixqtl_replication import load_inputs

    I = load_inputs()
    genes, keep = I['genes'], I['keep']
    A, T, Va, Vt, _ = HM.compute_summaries_from_gibbs(
        I['YL'], I['YR'], yT=I['YT'], count_noise=True)
    A, Va = A[:, keep], Va[:, keep]
    y1, y2, yt = MX.summaries_from_gibbs_posterior_mean(I['YL'], I['YR'], I['YT'])
    y1, y2 = y1[:, keep], y2[:, keep]

    informative = (y1 + y2) > 0
    fails_floor = (y1 < MX.ASC_CUTOFF) | (y2 < MX.ASC_CUTOFF)
    fails_ceil = (y1 > MX.ASC_CAP) | (y2 > MX.ASC_CAP)
    in_band = informative & ~fails_floor & ~fails_ceil
    above = informative & ~fails_floor & fails_ceil

    # ---- 1. within-gene: excess dispersion, in-band vs above-ceiling ----
    rows = []
    for j, g in enumerate(genes):
        mi, ma = in_band[j], above[j]
        if mi.sum() < 15 or ma.sum() < 15:
            continue
        vi, vm = float(np.var(A[j][mi], ddof=1)), float(np.mean(Va[j][mi]))
        va_, vam = float(np.var(A[j][ma], ddof=1)), float(np.mean(Va[j][ma]))
        rows.append(dict(gene=g, n_in=int(mi.sum()), n_above=int(ma.sum()),
                         excess_in=vi - vm, excess_above=va_ - vam,
                         absa_in=float(np.median(np.abs(A[j][mi]))),
                         absa_above=float(np.median(np.abs(A[j][ma]))),
                         meanVa_in=vm, meanVa_above=vam))
    wg = pd.DataFrame(rows)
    within = dict(n_genes_with_both_bands=int(len(wg)))
    if len(wg) >= 5:
        d = wg.excess_above - wg.excess_in
        # Wilcoxon signed-rank: a paired test on the ranks of the absolute
        # differences, so it needs no normality and one gene cannot dominate
        within.update(
            median_excess_in=float(wg.excess_in.median()),
            median_excess_above=float(wg.excess_above.median()),
            median_paired_diff=float(d.median()),
            genes_above_higher=int((d > 0).sum()),
            wilcoxon_p=float(wilcoxon(wg.excess_above, wg.excess_in).pvalue),
            median_absa_in=float(wg.absa_in.median()),
            median_absa_above=float(wg.absa_above.median()),
            median_meanVa_in=float(wg.meanVa_in.median()),
            median_meanVa_above=float(wg.meanVa_above.median()),
        )
    wg.to_csv(f'{OUT}/cutoff_band_within_gene.tsv', sep='\t', index=False)

    # ---- 2. allelic slope, only where the channel is ON in both runs ----
    full = read_slopes(f'{OUT}/_band_tmp_full')
    out = {}
    for label in ('cut', 'rnd'):
        r = read_slopes(f'{OUT}/_band_tmp_{label}')
        m = full.merge(r, on=['gene', 'variant_id'], suffixes=('_full', '_x'))
        on = (np.isfinite(m.slope_a_full) & np.isfinite(m.slope_a_x)
              & (m.slope_a_se_full > 0) & (m.slope_a_se_x > 0)
              & (m.slope_a_full != 0) & (m.slope_a_x != 0))
        m = m[on]
        per_gene, mag = [], []
        for g, dd in m.groupby('gene'):
            if len(dd) < 20:
                continue
            x, yv = dd.slope_a_full.values, dd.slope_a_x.values
            per_gene.append(float((x @ yv) / (x @ x)))
            # A through-origin slope confounds SCALE with CORRELATION, so a
            # value below 1 is not by itself attenuation. This separates
            # them: the ratio of median |slope_a| is pure scale. If the
            # regression slope falls but this does not, the mask has
            # displaced the estimate rather than shrunk it.
            mag.append(float(np.median(np.abs(yv)) / np.median(np.abs(x))))
        per_gene, mag = np.array(per_gene), np.array(mag)
        out[label] = dict(
            n_variants=int(len(m)), n_genes_channel_on=int(len(per_gene)),
            slope_a_pearson_r=float(pearsonr(m.slope_a_full, m.slope_a_x)[0])
            if len(m) > 10 else float('nan'),
            per_gene_slope_median=float(np.median(per_gene)) if len(per_gene) else float('nan'),
            per_gene_slope_iqr=[float(np.percentile(per_gene, 25)),
                                float(np.percentile(per_gene, 75))] if len(per_gene) else None,
            per_gene_abs_magnitude_ratio_median=float(np.median(mag)) if len(mag) else float('nan'),
            per_gene_abs_magnitude_ratio_iqr=[float(np.percentile(mag, 25)),
                                              float(np.percentile(mag, 75))] if len(mag) else None,
            median_se_ratio=float(np.median(m.slope_a_se_x / m.slope_a_se_full))
            if len(m) else float('nan'),
        )

    res = dict(within_gene_bands=within, allelic_slope_channel_on=out)
    json.dump(res, open(f'{OUT}/cutoff_band_within_gene.json', 'w'), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()

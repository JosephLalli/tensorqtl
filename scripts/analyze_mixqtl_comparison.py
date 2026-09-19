"""Join the hapmixQTL and mixQTL-replication arms and report the comparison.

Reads the hapmixQTL arm from the existing estimator ablation (config
`origin_q`, which is the shipped configuration: allelic channel through the
origin, count_noise on) rather than re-running it, and the mixQTL arm from
compare_mixqtl_replication.py.

Calibration is measured the same way for both arms so the numbers are
comparable. Each arm has 40 null genotype permutations and, per gene, the
maximum statistic over the cis window in each. For null draw d the
leave-one-out empirical p is

    p_d = (#{d' != d : stat_d' >= stat_d} + 1) / 40

which is uniform on (0, 1] if the arm is calibrated. Type-I at 5% is the
fraction of (gene, draw) pairs with p_d <= 0.05. This uses each arm as its
own reference, so it does not require the two arms to be on a common scale.

hapmixQTL's own within-gene permutation p (pval_perm, 1000 permutations) is
reported alongside as the reference measure, since it is what the pipeline
actually calls on.
"""

import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

D = '/mnt/ssd/lalli/brainvar_hapmix_deploy'
ABL = f'{D}/estimator_ablation_20260916'
OUT = f'{D}/mixqtl_replication_20260919'
NDRAW = 40


def loo_pvals(wide):
    """Leave-one-out empirical p per (gene, draw) from a genes x draws frame."""
    out = {}
    for g, row in wide.iterrows():
        s = row.values.astype(float)
        s = s[np.isfinite(s)]
        if s.size < 5:
            continue
        p = [(np.sum(np.delete(s, d) >= s[d]) + 1) / s.size for d in range(s.size)]
        out[g] = np.array(p)
    return out


def main():
    # ---- hapmixQTL arm, shipped configuration
    hm_obs = pd.read_csv(f'{ABL}/observed.origin_q.tsv', sep='\t')
    hm_obs = hm_obs[hm_obs.refit].set_index('gene')
    hm_nulls = pd.concat(
        [pd.read_csv(f'{ABL}/null.origin_q.{p:03d}.tsv', sep='\t').assign(draw=p)
         for p in range(NDRAW)], ignore_index=True)
    hm_wide = hm_nulls.pivot(index='gene', columns='draw', values='stat')

    # ---- mixQTL replication arm
    mx_obs = pd.read_csv(f'{OUT}/endtoend_mixqtl_observed.tsv', sep='\t').set_index('gene')
    mx_nulls = pd.read_csv(f'{OUT}/endtoend_mixqtl_nulls.tsv', sep='\t')
    mx_wide = mx_nulls.pivot(index='gene', columns='draw', values='stat')

    genes = sorted(set(hm_obs.index) & set(mx_obs.index))

    # ---- agreement on the observed pass
    lead_same = [hm_obs.loc[g, 'variant_id'] == mx_obs.loc[g, 'variant_id']
                 for g in genes]
    rho, prho = spearmanr(hm_obs.loc[genes, 'stat'], mx_obs.loc[genes, 'stat'])

    # ---- calibration, identically measured
    hm_p = loo_pvals(hm_wide.loc[genes])
    mx_p = loo_pvals(mx_wide.loc[genes])
    hm_all = np.concatenate([hm_p[g] for g in genes if g in hm_p])
    mx_all = np.concatenate([mx_p[g] for g in genes if g in mx_p])

    # hapmixQTL's own within-gene permutation p under the null
    hm_perm_typeI = float((hm_nulls['pval_perm'] <= 0.05).mean())

    res = dict(
        n_genes=len(genes),
        lead_agreement=float(np.mean(lead_same)),
        n_lead_same=int(np.sum(lead_same)),
        spearman_observed_stat=float(rho),
        spearman_p=float(prho),
        hapmixqtl_median_observed_stat=float(hm_obs.loc[genes, 'stat'].median()),
        mixqtl_median_observed_stat=float(mx_obs.loc[genes, 'stat'].median()),
        hapmixqtl_null_p95_stat=float(np.nanquantile(hm_wide.loc[genes].values, 0.95)),
        mixqtl_null_p95_stat=float(np.nanquantile(mx_wide.loc[genes].values, 0.95)),
        hapmixqtl_loo_typeI_at_5pct=float((hm_all <= 0.05).mean()),
        mixqtl_loo_typeI_at_5pct=float((mx_all <= 0.05).mean()),
        hapmixqtl_within_gene_perm_typeI=hm_perm_typeI,
        mixqtl_channel_used=(mx_obs.loc[genes, 'method'].value_counts().to_dict()
                             if 'method' in mx_obs else {}),
        mixqtl_median_n_asc=float(mx_obs.loc[genes, 'n_asc'].median()),
        mixqtl_median_n_cov_selected=float(mx_obs.loc[genes, 'n_cov_selected'].median()),
    )

    per_gene = pd.DataFrame({
        'gene': genes,
        'hapmix_stat': hm_obs.loc[genes, 'stat'].values,
        'mixqtl_stat': mx_obs.loc[genes, 'stat'].values,
        'hapmix_lead': hm_obs.loc[genes, 'variant_id'].values,
        'mixqtl_lead': mx_obs.loc[genes, 'variant_id'].values,
        'lead_same': lead_same,
        'hapmix_pval_perm': hm_obs.loc[genes, 'pval_perm'].values,
        'mixqtl_n_asc': mx_obs.loc[genes, 'n_asc'].values,
        'mixqtl_method': mx_obs.loc[genes, 'method'].values,
    })
    per_gene.to_csv(f'{OUT}/endtoend_per_gene.tsv', sep='\t', index=False)
    json.dump(res, open(f'{OUT}/endtoend_summary.json', 'w'), indent=1)
    for k, v in res.items():
        print(f'{k:42s} {v}')


if __name__ == '__main__':
    main()

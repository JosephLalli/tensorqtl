"""Join the hapmixQTL and mixQTL-replication arms and report the comparison.

Reads the hapmixQTL arm from the existing estimator ablation (config
`origin_q`: allelic channel through the origin, count_noise on -- the shipped
configuration) rather than re-running it, and the mixQTL arm from
compare_mixqtl_replication.py.

TWO MEASUREMENT CHOICES WORTH STATING

Lead agreement is computed on SIGNAL genes only. The 29 genes are the
null-calibration set; on most of them neither arm has a real association, so
the "lead" is the argmax of noise over ~4,000 correlated variants and
disagreement carries no information about the methods. Agreement is therefore
reported among genes with hapmixQTL pval_perm < 0.05, and by LD (r^2 between
the two leads) rather than exact identity, since two variants in tight LD are
the same signal.

Calibration is NOT measured by a leave-one-out rank p over the null draws.
That statistic is uniform by construction whenever the draws are
exchangeable, so it returns 0.05 for any arm and tests nothing. What is
reported instead is each arm's own gene-level p under the null where one
exists, and the null statistic distributions, which are comparable in shape
though not in scale.
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, binomtest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
D = '/mnt/ssd/lalli/brainvar_hapmix_deploy'
ABL = f'{D}/estimator_ablation_20260916'
OUT = f'{D}/mixqtl_replication_20260919'
NDRAW = 40

sys.path.insert(0, HERE)


def lead_ld(I, v1, v2):
    """r^2 between two variant ids, from the cohort dosages."""
    vdf, dos = I['vdf'], I['dos']
    try:
        i1 = vdf.index.get_loc(v1)
        i2 = vdf.index.get_loc(v2)
    except KeyError:
        return np.nan
    a, b = dos[i1].astype(float), dos[i2].astype(float)
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1] ** 2)


def main():
    from compare_mixqtl_replication import load_inputs
    I = load_inputs()
    vdf = I['vdf']

    hm_obs = pd.read_csv(f'{ABL}/observed.origin_q.tsv', sep='\t')
    hm_obs = hm_obs[hm_obs.refit].set_index('gene')
    hm_nulls = pd.concat(
        [pd.read_csv(f'{ABL}/null.origin_q.{p:03d}.tsv', sep='\t').assign(draw=p)
         for p in range(NDRAW)], ignore_index=True)
    hm_wide = hm_nulls.pivot(index='gene', columns='draw', values='stat')

    mx_obs = pd.read_csv(f'{OUT}/endtoend_mixqtl_observed.tsv', sep='\t').set_index('gene')
    mx_nulls = pd.read_csv(f'{OUT}/endtoend_mixqtl_nulls.tsv', sep='\t')
    mx_wide = mx_nulls.pivot(index='gene', columns='draw', values='stat')

    genes = sorted(set(hm_obs.index) & set(mx_obs.index))

    rows = []
    for g in genes:
        l1, l2 = hm_obs.loc[g, 'variant_id'], mx_obs.loc[g, 'variant_id']
        p1 = int(l1.split('_')[1]) if '_' in l1 else np.nan
        p2 = int(l2.split('_')[1]) if '_' in l2 else np.nan
        rows.append(dict(
            gene=g,
            hapmix_stat=float(hm_obs.loc[g, 'stat']),
            mixqtl_stat=float(mx_obs.loc[g, 'stat']),
            hapmix_pval_perm=float(hm_obs.loc[g, 'pval_perm']),
            hapmix_lead=l1, mixqtl_lead=l2,
            lead_same=l1 == l2,
            lead_r2=lead_ld(I, l1, l2),
            lead_bp=abs(p1 - p2) if np.isfinite(p1) and np.isfinite(p2) else np.nan,
            mixqtl_n_asc=int(mx_obs.loc[g, 'n_asc']),
            mixqtl_method=str(mx_obs.loc[g, 'method']),
        ))
    pg = pd.DataFrame(rows)
    pg.to_csv(f'{OUT}/endtoend_per_gene.tsv', sep='\t', index=False)

    sig = pg[pg.hapmix_pval_perm < 0.05]
    rho, prho = spearmanr(pg.hapmix_stat, pg.mixqtl_stat)

    # paired sign test on the weighting ablation: same 40 permutations for
    # every arm, so a per-gene win/loss is distribution-free.
    ab = pd.read_csv(f'{OUT}/weighting_ablation.tsv', sep='\t')
    piv = ab.pivot(index='gene', columns='arm', values='median_var_beta')
    signs = {}
    for a, b in [('gibbs_1_over_v', 'equal_ols'),
                 ('gibbs_1_over_v', 'harmonic_uncapped'),
                 ('gibbs_1_over_v', 'harmonic_poisson_capped'),
                 ('gibbs_1_over_v', 'gibbs_capped'),
                 ('harmonic_uncapped', 'equal_ols'),
                 ('gibbs_1_over_v', 'gibbs_draws_only')]:
        if a in piv and b in piv:
            wins = int((piv[a] < piv[b]).sum())
            n = int(piv[[a, b]].notna().all(axis=1).sum())
            signs[f'{a}_beats_{b}'] = dict(
                wins=wins, n=n,
                binom_p=float(binomtest(wins, n, 0.5).pvalue),
                median_ratio=float((piv[a] / piv[b]).median()))

    res = dict(
        n_genes=len(genes),
        n_signal_genes=int(len(sig)),
        lead_exact_agreement_all=float(pg.lead_same.mean()),
        lead_exact_agreement_signal=float(sig.lead_same.mean()) if len(sig) else None,
        lead_r2_median_signal=float(sig.lead_r2.median()) if len(sig) else None,
        lead_r2_ge_0p8_signal=int((sig.lead_r2 >= 0.8).sum()) if len(sig) else None,
        lead_bp_median_signal=float(sig.lead_bp.median()) if len(sig) else None,
        spearman_observed_stat=float(rho), spearman_p=float(prho),
        hapmixqtl_median_observed_stat=float(pg.hapmix_stat.median()),
        mixqtl_median_observed_stat=float(pg.mixqtl_stat.median()),
        hapmixqtl_null_median_stat=float(np.nanmedian(hm_wide.loc[genes].values)),
        mixqtl_null_median_stat=float(np.nanmedian(mx_wide.loc[genes].values)),
        hapmixqtl_null_p95_stat=float(np.nanquantile(hm_wide.loc[genes].values, 0.95)),
        mixqtl_null_p95_stat=float(np.nanquantile(mx_wide.loc[genes].values, 0.95)),
        hapmixqtl_within_gene_perm_typeI=float((hm_nulls['pval_perm'] <= 0.05).mean()),
        mixqtl_within_gene_perm_typeI=None,   # see REPORT: not run
        mixqtl_channel_used=mx_obs.loc[genes, 'method'].value_counts().to_dict(),
        mixqtl_median_n_asc=float(mx_obs.loc[genes, 'n_asc'].median()),
        mixqtl_genes_trc_only=int((mx_obs.loc[genes, 'method'] == 'trc').sum()),
        mixqtl_median_n_cov_selected=float(mx_obs.loc[genes, 'n_cov_selected'].median()),
        ablation_sign_tests=signs,
    )
    json.dump(res, open(f'{OUT}/endtoend_summary.json', 'w'), indent=1)
    for k, v in res.items():
        if k == 'ablation_sign_tests':
            print('ablation_sign_tests:')
            for kk, vv in v.items():
                print(f"   {kk:48s} {vv['wins']}/{vv['n']}  "
                      f"p={vv['binom_p']:.2e}  median ratio {vv['median_ratio']:.3f}")
        else:
            print(f'{k:40s} {v}')


if __name__ == '__main__':
    main()

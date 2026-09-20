"""Gene-level type-I error of the mixQTL replication arm under the null.

hapmixQTL reports a within-gene permutation p (pval_perm) whose type-I under
null genotype permutations is measured; the mixQTL arm had no counterpart, so
the two could not be compared on calibration. This supplies one.

For each of NULL_DRAWS null genotype permutations, and each gene, the arm's
gene-level p is the fraction of NPERM within-gene phenotype-bundle
permutations whose maximum |meta statistic| over the cis window reaches the
observed maximum. Under the null that p should be uniform, so the fraction
below 0.05 estimates type-I at the 5% level.

IMPORTANT SCOPE: this uses the CORRECTED permutation path
(strict_reference_cap=False). The published path cannot produce this number
at all -- its weight cap zeroes every allelic weight whenever any sample
fails the ASE cutoff, which happens for every gene here. So this is the
calibration of a repaired port, not of mixQTL as distributed. It is also not
the published pipeline's Beta approximation, which is a separate layer.
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
D = '/mnt/ssd/lalli/brainvar_hapmix_deploy'
OUT = f'{D}/mixqtl_replication_20260919'

sys.path.insert(0, HERE)
sys.path.insert(0, REPO)

NPERM = int(os.environ.get('NPERM', '200'))
NULL_DRAWS = int(os.environ.get('NULL_DRAWS', '10'))
SEED = 42          # master seed; child streams derived as SEED + offset + index

log = lambda *a: print(time.strftime('%H:%M:%S'), *a, flush=True)


def main():
    from compare_mixqtl_replication import load_inputs, gene_variant_index, WIN
    from tensorqtl import mixqtl_replication as MX

    I = load_inputs()
    genes, order, keep = I['genes'], I['order'], I['keep']
    y1, y2, yt = MX.summaries_from_gibbs_posterior_mean(I['YL'], I['YR'], I['YT'])
    y1, y2, yt = y1[:, keep], y2[:, keep], yt[:, keep]
    n = len(order)
    cov = I['cov_df'].values
    lib = I['lib_size']

    rows = []
    for d in range(NULL_DRAWS):
        gprm = np.random.RandomState(SEED + 10007 + d).permutation(n)
        for j, g in enumerate(genes):
            vsel = gene_variant_index(I, g)
            if vsel.size == 0:
                continue
            h1 = I['xL'][I['idx']][:, keep][vsel].T.astype(float)
            h2 = I['xR'][I['idx']][:, keep][vsel].T.astype(float)
            a1, a2, at = y1[j][gprm], y2[j][gprm], yt[j][gprm]
            cv, lb = cov[gprm], lib[gprm]

            obs = MX.mixqtl_scan(a1, a2, at, lb, h1, h2, covariates=cv)
            s_obs = np.nanmax(np.abs(obs['meta']['stat']))
            if not np.isfinite(s_obs):
                continue
            perm_idx = np.array([
                np.random.RandomState(SEED + 7919 + d * 1000 + k).permutation(n)
                for k in range(NPERM)])
            s_perm = MX.mixqtl_permutation_scan(a1, a2, at, lb, h1, h2, perm_idx,
                                                covariates=cv)
            ok = np.isfinite(s_perm)
            if ok.sum() < NPERM // 2:
                continue
            p = (np.sum(s_perm[ok] >= s_obs) + 1) / (ok.sum() + 1)
            rows.append(dict(draw=d, gene=g, stat=float(s_obs ** 2),
                             pval_perm=float(p), n_perm_ok=int(ok.sum())))
        log(f'null draw {d + 1}/{NULL_DRAWS} done ({len(rows)} rows)')

    df = pd.DataFrame(rows)
    df.to_csv(f'{OUT}/mixqtl_gene_level_null.tsv', sep='\t', index=False)
    res = dict(
        nperm=NPERM, null_draws=NULL_DRAWS, n_rows=len(df),
        mixqtl_gene_level_typeI_5pct=float((df.pval_perm <= 0.05).mean()),
        mixqtl_gene_level_typeI_10pct=float((df.pval_perm <= 0.10).mean()),
        mixqtl_median_pval_perm=float(df.pval_perm.median()),
        note='corrected permutation path (strict_reference_cap=False); '
             'the published path cannot produce this number',
    )
    json.dump(res, open(f'{OUT}/mixqtl_gene_level_typeI.json', 'w'), indent=1)
    for k, v in res.items():
        print(f'{k:36s} {v}')


if __name__ == '__main__':
    main()

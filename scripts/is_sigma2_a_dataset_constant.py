"""Is mixQTL's sigma^2 a dataset property, or does it need fitting per gene?

mixQTL writes var(a_i) = sigma^2 * q_i with q_i = 1/y1 + 1/y2. If the draws
are taken as the truth, then sigma^2 = v_i / q_i, so the draws give a
NON-CIRCULAR estimate of sigma^2 -- one that never touches association
residuals. That makes the question empirical rather than philosophical.

Three possibilities, distinguished by a variance decomposition of
log(v/q) into between-gene and within-gene parts:

  all variation WITHIN gene, none between   -> sigma^2 is one dataset
                                               constant; fitting it per gene
                                               fits noise
  substantial BETWEEN-gene variation        -> sigma^2 is a per-gene
                                               parameter and must be
                                               estimated per gene
  large WITHIN-gene variation               -> sigma^2 is not a constant at
                                               all, the shape q is wrong,
                                               and no per-gene scalar fixes
                                               it

These are not exclusive. Deterministic; no permutation, no null, no seed.
"""

import json
import sys

import numpy as np

sys.path.insert(0, 'scripts')
sys.path.insert(0, '.')

OUT = '/mnt/ssd/lalli/brainvar_hapmix_deploy/mixqtl_replication_20260919'
KAPPA = 0.5


def main():
    import tensorqtl.hapmixqtl as HM
    from compare_mixqtl_replication import load_inputs

    I = load_inputs()
    keep, genes = I['keep'], I['genes']
    _A, _T, Va_noq, _Vt, _C = HM.compute_summaries_from_gibbs(
        I['YL'], I['YR'], yT=I['YT'], count_noise=False)
    v = Va_noq[:, keep]
    mL, mR = I['YL'][:, keep].mean(2), I['YR'][:, keep].mean(2)
    q = 1.0 / (mL + KAPPA) + 1.0 / (mR + KAPPA)
    inf = v > 1e-12

    per_gene_median, within_sd, ns = [], [], []
    all_centered = []
    for j, g in enumerate(genes):
        m = inf[j]
        if m.sum() < 10:
            continue
        lr = np.log(v[j][m] / q[j][m])
        per_gene_median.append(np.median(lr))
        within_sd.append(lr.std())
        all_centered.append(lr - lr.mean())
        ns.append(int(m.sum()))
    per_gene_median = np.array(per_gene_median)
    within_sd = np.array(within_sd)
    within = np.concatenate(all_centered)

    between_var = per_gene_median.var(ddof=1)
    within_var = within.var(ddof=1)
    total = between_var + within_var

    print(f'{len(per_gene_median)} genes, '
          f'{sum(ns)} informative donor-gene pairs\n')

    print('Per-gene sigma^2 estimated from the draws, exp(median log v/q):')
    pg = np.exp(per_gene_median)
    print(f'  median across genes {np.median(pg):.2f}, '
          f'range {pg.min():.2f} to {pg.max():.2f}  '
          f'({pg.max() / pg.min():.1f}-fold)\n')

    print('Variance decomposition of log(v/q):')
    print(f'  BETWEEN-gene variance {between_var:.4f}  '
          f'({100 * between_var / total:.1f}% of total)')
    print(f'  WITHIN-gene  variance {within_var:.4f}  '
          f'({100 * within_var / total:.1f}% of total)')
    print(f'  between-gene sd {np.sqrt(between_var):.3f} '
          f'({np.exp(np.sqrt(between_var)):.2f}-fold)')
    print(f'  within-gene  sd {np.sqrt(within_var):.3f} '
          f'({np.exp(np.sqrt(within_var)):.2f}-fold)\n')

    # How much would a single dataset-wide constant cost, versus per-gene?
    grand = np.median(np.concatenate(
        [np.log(v[j][inf[j]] / q[j][inf[j]]) for j in range(len(genes))
         if inf[j].sum() >= 10]))
    resid_global = np.concatenate(
        [np.log(v[j][inf[j]] / q[j][inf[j]]) - grand for j in range(len(genes))
         if inf[j].sum() >= 10])
    print('Cost of using ONE dataset-wide constant instead of per-gene:')
    print(f'  rms of log(v/q) about the global constant  '
          f'{np.sqrt((resid_global ** 2).mean()):.4f}')
    print(f'  rms about each gene\'s own constant         '
          f'{np.sqrt((within ** 2).mean()):.4f}')
    print('  the gap is what per-gene fitting buys; the second number is '
          'what\n  no per-gene scalar can remove.')

    json.dump(dict(
        n_genes=len(per_gene_median), n_pairs=int(sum(ns)),
        per_gene_sigma2_median=float(np.median(pg)),
        per_gene_sigma2_min=float(pg.min()), per_gene_sigma2_max=float(pg.max()),
        between_gene_var=float(between_var), within_gene_var=float(within_var),
        pct_between=float(100 * between_var / total),
        rms_about_global=float(np.sqrt((resid_global ** 2).mean())),
        rms_about_per_gene=float(np.sqrt((within ** 2).mean())),
    ), open(f'{OUT}/sigma2_decomposition.json', 'w'), indent=1)


if __name__ == '__main__':
    main()

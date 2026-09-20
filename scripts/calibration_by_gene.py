"""Does the 3.3x calibration gap vary by gene, and is any spread real?

The headline 3.3x is a median over 29 genes of
var(beta_hat) / mean(se^2) under the known-variance standard error. This
reports the per-gene distribution.

A per-gene spread is only interesting if it exceeds Monte Carlo noise. Each
gene's var(beta_hat) comes from 40 permutations, so it carries roughly
sqrt(2/39) = 23% relative noise on its own, and the ratio inherits that.
Rather than argue from that figure, the spread is checked against an
empirical floor: the same quantity computed under an independent master
seed. If per-gene calibration is real, the two seeds should agree per gene.

The gibbs_1_over_v arm uses uncapped 1/v weights, so it is untouched by the
weight-cap change that accompanied the cutoff-preset switch, and the seed-0
and seed-42 runs are directly comparable for it.

Deterministic given the stored runs; no new permutation.
"""

import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, 'scripts')
sys.path.insert(0, '.')

OUT = '/mnt/ssd/lalli/brainvar_hapmix_deploy/mixqtl_replication_20260919'
ARM = 'gibbs_1_over_v'


def load(path):
    d = pd.read_csv(path, sep='\t')
    d = d[d.arm == ARM].set_index('gene')
    return d


def main():
    new = load(f'{OUT}/weighting_ablation.tsv')
    old = load(f'{OUT}/seed0_superseded/weighting_ablation.tsv')
    col = 'median_calibration_known_var'

    c = new[col]
    print(f'Per-gene calibration var(beta_hat)/mean(se^2), '
          f'known-variance SE, {len(c)} genes\n')
    print(f'  median {c.median():.2f}')
    print(f'  IQR    {c.quantile(.25):.2f} - {c.quantile(.75):.2f}')
    print(f'  range  {c.min():.2f} - {c.max():.2f}   '
          f'({c.max() / c.min():.1f}-fold)')
    print(f'  genes below 1 (conservative): {int((c < 1).sum())}/{len(c)}\n')

    print('Least and most inflated genes:')
    s = c.sort_values()
    for g, val in list(s.items())[:4]:
        print(f'  {g:10s} {val:7.2f}   n_inf={int(new.loc[g, "n_inf"])}')
    print('  ...')
    for g, val in list(s.items())[-4:]:
        print(f'  {g:10s} {val:7.2f}   n_inf={int(new.loc[g, "n_inf"])}')

    # empirical noise floor: same gene, independent seed
    common = sorted(set(new.index) & set(old.index))
    a, b = old.loc[common, col], new.loc[common, col]
    r = pearsonr(a, b)[0]
    rel = np.abs(b - a) / ((a + b) / 2)
    print(f'\nEmpirical noise floor, seed 0 against seed 42, same genes:')
    print(f'  Pearson r = {r:.4f} over {len(common)} genes')
    print(f'  median relative difference between seeds {100 * np.median(rel):.1f}%')
    print(f'  between-gene sd of log calibration '
          f'{np.log(b).std():.3f} '
          f'({np.exp(np.log(b).std()):.2f}-fold)')
    print(f'  within-gene  sd across seeds (log) '
          f'{np.std(np.log(b) - np.log(a)) / np.sqrt(2):.3f}')

    between = np.log(b).var(ddof=1)
    withins = np.var(np.log(b) - np.log(a), ddof=1) / 2
    print(f'\n  between-gene variance {between:.4f}, '
          f'seed-noise variance {withins:.4f}')
    print(f'  -> {100 * max(0, between - withins) / between:.1f}% of the '
          f'between-gene spread survives the noise floor')

    # does it track anything obvious?
    print('\nDoes per-gene calibration track gene properties?')
    for name in ['n_inf', 'eff_n', 'weight_fold_spread', 'median_var_beta']:
        if name in new.columns:
            rho = spearmanr(new[name], c)[0]
            print(f'  Spearman with {name:22s} {rho:+.3f}')

    json.dump(dict(
        median=float(c.median()), iqr=[float(c.quantile(.25)),
                                       float(c.quantile(.75))],
        min=float(c.min()), max=float(c.max()),
        fold=float(c.max() / c.min()),
        cross_seed_r=float(r),
        median_rel_diff_between_seeds=float(np.median(rel)),
        between_gene_var_log=float(between),
        seed_noise_var_log=float(withins),
        pct_spread_surviving_noise=float(
            100 * max(0, between - withins) / between),
    ), open(f'{OUT}/calibration_by_gene.json', 'w'), indent=1)


if __name__ == '__main__':
    main()

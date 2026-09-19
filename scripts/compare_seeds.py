"""Seed replication: put the seed-0 and seed-42 ablation results side by side.

The master seed moved from 0 to 42 on 2026-09-19. Re-running at a different
seed is not only bookkeeping -- it is a free replication check. The reported
efficiency ratios are medians over 29 genes of a variance estimated from 40
permutations, so each carries roughly 23%/sqrt(29) of Monte Carlo noise. Two
independent seeds agreeing to within that is evidence the differences between
weighting arms are real rather than an artifact of one permutation draw.

The seed-0 outputs are preserved under seed0_superseded/.
"""

import os

import numpy as np
import pandas as pd

D = '/mnt/ssd/lalli/brainvar_hapmix_deploy/mixqtl_replication_20260919'


def ratios(path):
    ab = pd.read_csv(path, sep='\t')
    piv = ab.pivot(index='gene', columns='arm', values='median_var_beta')
    return (piv.div(piv['equal_ols'], axis=0)).median(), piv


def main():
    old_path = f'{D}/seed0_superseded/weighting_ablation.tsv'
    new_path = f'{D}/weighting_ablation.tsv'
    if not os.path.exists(old_path):
        raise SystemExit(f'no seed-0 baseline at {old_path}')
    r0, p0 = ratios(old_path)
    r42, p42 = ratios(new_path)

    print('median var(beta_hat) ratio vs unweighted OLS, by master seed\n')
    print(f'{"arm":28s} {"seed 0":>9s} {"seed 42":>9s} {"abs diff":>9s}')
    arms = [a for a in r42.index if a != 'equal_ols']
    for a in arms:
        if a in r0:
            print(f'{a:28s} {r0[a]:9.4f} {r42[a]:9.4f} {abs(r42[a] - r0[a]):9.4f}')
        else:
            print(f'{a:28s} {"--":>9s} {r42[a]:9.4f} {"--":>9s}')

    common = [a for a in arms if a in r0]
    d = np.array([abs(r42[a] - r0[a]) for a in common])
    print(f'\nlargest absolute shift across arms: {d.max():.4f}')
    print('separation between adjacent arms at seed 42: '
          + ', '.join(f'{x:.3f}' for x in
                      np.diff(sorted(r42[a] for a in arms))))

    # per-gene correlation of the two seeds' Gibbs efficiency
    if 'gibbs_1_over_v' in p0 and 'gibbs_1_over_v' in p42:
        g = sorted(set(p0.index) & set(p42.index))
        a = (p0.loc[g, 'gibbs_1_over_v'] / p0.loc[g, 'equal_ols'])
        b = (p42.loc[g, 'gibbs_1_over_v'] / p42.loc[g, 'equal_ols'])
        print(f'\nper-gene Gibbs/OLS ratio, seed 0 vs seed 42: '
              f'Pearson r = {np.corrcoef(a, b)[0, 1]:.3f} over {len(g)} genes')


if __name__ == '__main__':
    main()

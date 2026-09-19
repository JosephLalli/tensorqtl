"""Does the weights-only baseline also reproduce the magnitude and SE gaps?

Correlation is a compressed summary. mixQTL also reports systematically
LARGER effects (median |beta| 1.24x) and larger standard errors (1.30x),
and those are directional facts that a correlation cannot speak to. If
changing the weight vector alone reproduces them, they need no further
explanation either. If it does not, they need a separate mechanism, and the
pseudocount is the obvious candidate since it shrinks the contrast toward
zero.
"""

import pandas as pd
import numpy as np

D = '/mnt/ssd/lalli/brainvar_hapmix_deploy/mixqtl_replication_20260919'


def main():
    j = pd.read_parquet(f'{D}/partition_and_baseline.parquet')
    b = j.dropna(subset=['base_w_gibbs', 'base_w_harmonic'])

    print('MAGNITUDE: does changing only the weights reproduce the 1.24x gap?\n')
    obs = j.mx_beta.abs().median() / j.slope.abs().median()
    base = b.base_w_harmonic.abs().median() / b.base_w_gibbs.abs().median()
    print(f'  observed, full arms      median |mixQTL| / |hapmixQTL| = {obs:.3f}')
    print(f'  baseline, weights only   median |harmonic| / |1/v|     = {base:.3f}')

    pg = b.groupby('gene').apply(lambda d: pd.Series({
        'obs': d.mx_beta.abs().median() / d.slope.abs().median(),
        'base': d.base_w_harmonic.abs().median() / d.base_w_gibbs.abs().median(),
    }), include_groups=False)
    print(f'\n  per gene (median over genes): observed {pg.obs.median():.3f}, '
          f'baseline {pg.base.median():.3f}')

    print('\n  allelic channel alone: median |mx_beta_asc| / |slope_a| = '
          f'{j.mx_beta_asc.abs().median() / j.slope_a.abs().median():.3f}')

    print('\nWhat the weights cannot explain is the residual:')
    print(f'  magnitude gap left over = {obs:.3f} observed against '
          f'{base:.3f} from weights alone')


if __name__ == '__main__':
    main()

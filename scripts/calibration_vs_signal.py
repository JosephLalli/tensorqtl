"""Is the per-gene calibration inflation driven by genes carrying real signal?

A stated caveat on the 3.3x figure was that permutation turns any real cis
effect into apparent spread, so genes with signal should look worse
calibrated than they are. That is testable rather than assertable: split the
29 genes by whether hapmixQTL calls them and compare.

If the inflation concentrates in called genes, the figure is a signal
artifact and the null-gene median is the better estimate. If it does not,
the variation is a property of the genes and the caveat, while true in
principle, is not what is driving the spread.
"""

import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu

D = '/mnt/ssd/lalli/brainvar_hapmix_deploy/mixqtl_replication_20260919'


def main():
    ab = pd.read_csv(f'{D}/weighting_ablation.tsv', sep='\t')
    ab = ab[ab.arm == 'gibbs_1_over_v'].set_index('gene')
    pg = pd.read_csv(f'{D}/endtoend_per_gene.tsv', sep='\t').set_index('gene')
    j = pd.DataFrame({'calib': ab['median_calibration_known_var'],
                      'pperm': pg['hapmix_pval_perm']}).dropna()
    sig = j.pperm < 0.05

    print(f'{len(j)} genes; {int(sig.sum())} called by hapmixQTL '
          f'(pval_perm < 0.05)\n')
    print(f'  calibration, CALLED genes : median {j.calib[sig].median():6.2f}'
          f'   range {j.calib[sig].min():.2f}-{j.calib[sig].max():.2f}')
    print(f'  calibration, OTHER genes  : median {j.calib[~sig].median():6.2f}'
          f'   range {j.calib[~sig].min():.2f}-{j.calib[~sig].max():.2f}')
    u = mannwhitneyu(j.calib[sig], j.calib[~sig])
    print(f'  Mann-Whitney p = {u.pvalue:.3f}')
    rho = spearmanr(j.calib, -np.log10(j.pperm))[0]
    print(f'  Spearman(calibration, -log10 pval_perm) = {rho:+.3f}\n')

    print('  called genes, by calibration:')
    for g, r in j[sig].sort_values('calib').iterrows():
        print(f'    {g:10s} calib {r.calib:6.2f}   pval_perm {r.pperm:.4f}')

    json.dump(dict(
        n_genes=int(len(j)), n_called=int(sig.sum()),
        median_called=float(j.calib[sig].median()),
        median_other=float(j.calib[~sig].median()),
        mannwhitney_p=float(u.pvalue),
        spearman_calib_vs_signal=float(rho),
    ), open(f'{D}/calibration_vs_signal.json', 'w'), indent=1)


if __name__ == '__main__':
    main()

"""Where does this cohort sit relative to Huang et al. 2018 (NAR e133)?

Two findings in that paper bear on our comparison rather than on the
definition of gene-level testing:

  1. "for studies with 100 samples, a MAF threshold of 10% is necessary to
     control FDR of hierarchical multiple testing procedure". We tested at
     MAF >= 0.05 with n = 92, below their smallest simulated sample size.

  2. "for studies with 80% power to detect a given eQTL of MAF <= 25%, the
     top eSNP was the true causal eSNP 90% of the time", falling toward
     70% for common variants in strong LD. Lead identity is therefore an
     intrinsically noisy readout even for a correctly behaving method.

This reports the MAF composition of what we tested and of the leads we
called, so the exposure to (1) is quantified rather than assumed.
"""

import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, 'scripts')
sys.path.insert(0, '.')

OUT = '/mnt/ssd/lalli/brainvar_hapmix_deploy/mixqtl_replication_20260919'


def main():
    from compare_mixqtl_replication import load_inputs
    I = load_inputs()
    dos = I['dos'][I['idx']][:, I['keep']]
    af = dos.mean(1) / 2.0
    maf = np.minimum(af, 1 - af)
    n = dos.shape[1]

    print(f'cohort n = {n} donors '
          f'(Huang et al. simulate 100, 200, 500, 1000, 2000, 5000; '
          f'our n is below their smallest)\n')
    print(f'{len(maf):,} tested variants, MAF threshold applied: 0.05')
    for lo, hi in [(0.05, 0.10), (0.10, 0.25), (0.25, 0.50)]:
        k = ((maf >= lo) & (maf < hi)).sum()
        print(f'  MAF {lo:.2f}-{hi:.2f}: {k:7,} ({100 * k / len(maf):5.1f}%)')
    below10 = (maf < 0.10).sum()
    print(f'\n  variants below the MAF 0.10 the paper recommends at n=100: '
          f'{below10:,} ({100 * below10 / len(maf):.1f}%)')

    # MAF of the leads we actually called
    pg = pd.read_csv(f'{OUT}/endtoend_per_gene.tsv', sep='\t')
    vdf = I['vdf'].iloc[I['idx']]
    lut = pd.Series(maf, index=vdf.index.astype(str))
    pg['lead_maf'] = pg.hapmix_lead.astype(str).map(lut)
    called = pg[pg.hapmix_pval_perm < 0.05]
    print(f'\nleads of the {len(called)} called genes:')
    for _, r in called.sort_values('lead_maf').iterrows():
        flag = '  <- below 0.10' if r.lead_maf < 0.10 else ''
        print(f'  {r.gene:10s} MAF {r.lead_maf:.3f}   '
              f'pval_perm {r.hapmix_pval_perm:.4f}{flag}')
    print(f'\n  called leads below MAF 0.10: '
          f'{int((called.lead_maf < 0.10).sum())}/{len(called)}')
    print(f'  all 29 leads below MAF 0.10: '
          f'{int((pg.lead_maf < 0.10).sum())}/{len(pg)}')

    json.dump(dict(
        n_donors=int(n), n_variants=int(len(maf)),
        frac_below_maf10=float(below10 / len(maf)),
        called_leads_below_maf10=int((called.lead_maf < 0.10).sum()),
        n_called=int(len(called)),
        all_leads_below_maf10=int((pg.lead_maf < 0.10).sum()),
        median_lead_maf=float(pg.lead_maf.median()),
    ), open(f'{OUT}/maf_power_regime.json', 'w'), indent=1)


if __name__ == '__main__':
    main()

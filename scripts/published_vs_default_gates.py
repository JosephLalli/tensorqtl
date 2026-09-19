"""mixQTL's package defaults are not the settings its authors published with.

    function defaults   trc_cutoff=20   asc_cutoff=5    weight_cap=100  asc_cap=5000
    roxygen examples    trc_cutoff=100  asc_cutoff=50   weight_cap=100  asc_cap=1000
    GTEx v8 driver      trc_cutoff=100  asc_cutoff=50   weight_cap=10   asc_cap=1000

The replication arm was built on the function defaults. The examples and the
authors' own production driver agree with each other and disagree with the
defaults, so the defaults are the outlier. This counts what each setting does
to the allelic channel on these 29 high-coverage genes.
"""

import json
import sys

import numpy as np

sys.path.insert(0, 'scripts')
sys.path.insert(0, '.')

D = '/mnt/ssd/lalli/brainvar_hapmix_deploy/mixqtl_replication_20260919'


def main():
    import tensorqtl.hapmixqtl as HM
    from compare_mixqtl_replication import load_inputs

    I = load_inputs()
    keep = I['keep']
    _A, _T, Va, _Vt, _C = HM.compute_summaries_from_gibbs(
        I['YL'], I['YR'], yT=I['YT'], count_noise=True)
    Va = Va[:, keep]
    mL, mR = I['YL'].mean(2)[:, keep], I['YR'].mean(2)[:, keep]
    inf = Va > 1e-12
    n_pairs = inf.size

    print(f'{n_pairs} donor-gene pairs; {int(inf.sum())} carry allelic '
          f'information (hapmixQTL keeps all of these)\n')
    out = {'n_pairs': int(n_pairs), 'n_informative': int(inf.sum())}

    for label, lo, hi in [('package defaults', 5, 5000),
                          ('published / GTEx v8', 50, 1000)]:
        gate = (mL >= lo) & (mR >= lo) & (mL <= hi) & (mR <= hi)
        lost_lo = int((inf & ~((mL >= lo) & (mR >= lo))).sum())
        lost_hi = int((inf & ((mL >= lo) & (mR >= lo)) & ~((mL <= hi) & (mR <= hi))).sum())
        kept = int((inf & gate).sum())
        print(f'{label}  (asc_cutoff={lo}, asc_cap={hi})')
        print(f'   kept {kept:5d} of {int(inf.sum())} informative '
              f'({100 * kept / inf.sum():.1f}%)')
        print(f'   lost to the LOWER cutoff: {lost_lo:5d}')
        print(f'   lost to the UPPER cap   : {lost_hi:5d}')
        per_gene = (inf & gate).sum(1)
        print(f'   donors per gene: median {int(np.median(per_gene))}, '
              f'min {int(per_gene.min())}, '
              f'genes under 15 donors: {int((per_gene < 15).sum())}/{len(per_gene)}')
        print(f'   genes with 0 usable allelic donors: '
              f'{int((per_gene == 0).sum())}\n')
        out[label] = dict(kept=kept, lost_lower=lost_lo, lost_upper=lost_hi,
                          median_per_gene=int(np.median(per_gene)),
                          genes_under_15=int((per_gene < 15).sum()),
                          genes_zero=int((per_gene == 0).sum()))

    print('mixQTL falls back to total-counts-only when a channel has fewer '
          'than 15 samples\n(meta_analyze n_cutoff), so "genes under 15 '
          'donors" is where the allelic\nchannel stops contributing at all.')
    json.dump(out, open(f'{D}/published_vs_default_gates.json', 'w'), indent=1)


if __name__ == '__main__':
    main()

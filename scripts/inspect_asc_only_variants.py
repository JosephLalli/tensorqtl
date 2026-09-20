"""What are the 7 variants where mixQTL's meta fell through to allelic-only?

Not a hypothesis test -- a triage. The r = -0.39 seen on those 7 has no
standing at n = 7, so the question is not "is it real" but "is the CATEGORY
they belong to worth pursuing".

In the port, meta_analyze labels a variant 'asc' when the total-channel beta
is NaN while the allelic one is not. The total beta is NaN when the variant
is monomorphic AFTER the total cutoff, i.e. dosage is constant among the
donors that passed. A variant can have constant dosage -- every donor
heterozygous -- while the phased contrast h1 - h2 still varies, because
1|0 and 0|1 donors differ. That is one of the catalogued divergences: the
global constant-dosage filter discards variants whose allelic test is
perfectly well posed.

So this counts how large that category is, rather than interpreting r.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
D = '/mnt/ssd/lalli/brainvar_hapmix_deploy/mixqtl_replication_20260919'

sys.path.insert(0, HERE)
sys.path.insert(0, REPO)


def fisher_ci(r, n, alpha=0.05):
    from scipy.stats import norm
    if n < 4:
        return (np.nan, np.nan)
    z = np.arctanh(r)
    se = 1 / np.sqrt(n - 3)
    lo, hi = z - norm.ppf(1 - alpha / 2) * se, z + norm.ppf(1 - alpha / 2) * se
    return float(np.tanh(lo)), float(np.tanh(hi))


def main():
    from compare_mixqtl_replication import load_inputs

    m = pd.read_parquet(f'{D}/observed_matched_variants.parquet')
    asc = m[m.mx_method == 'asc']
    print(f'{len(asc)} variants labelled asc-only, of {len(m):,} matched\n')
    if len(asc):
        from scipy.stats import pearsonr
        r = pearsonr(asc.hm_beta, asc.mx_beta)[0]
        lo, hi = fisher_ci(r, len(asc))
        print(f'  r = {r:.3f}   95% CI ({lo:.3f}, {hi:.3f})   n = {len(asc)}')
        print(f'  CI width {hi - lo:.3f}; contains 0: {lo < 0 < hi}\n')
        print(asc[['gene', 'variant_id', 'hm_beta', 'mx_beta',
                   'hm_se', 'mx_se', 'mx_n_asc']].to_string(index=False))

    # how big is the category in the tested universe, not just the matched set?
    I = load_inputs()
    dos = I['dos'][I['idx']][:, I['keep']]
    xL = I['xL'][I['idx']][:, I['keep']]
    xR = I['xR'][I['idx']][:, I['keep']]
    const_dos = dos.std(axis=1) == 0
    phase_varies = (xL - xR).std(axis=1) > 0
    both = const_dos & phase_varies
    print(f'\nacross all {dos.shape[0]:,} tested variants in the window:')
    print(f'  constant dosage across donors        {int(const_dos.sum()):,}')
    print(f'  ... of which the phased contrast varies {int(both.sum()):,} '
          f'({100 * both.mean():.4f}%)')
    print('  (these are the ones a constant-dosage filter would discard '
          'while their allelic test remains well posed)')

    json.dump(dict(
        n_asc_only_matched=int(len(asc)),
        r=float(r) if len(asc) else None,
        r_ci=[lo, hi] if len(asc) else None,
        n_tested_variants=int(dos.shape[0]),
        n_constant_dosage=int(const_dos.sum()),
        n_constant_dosage_phase_varies=int(both.sum()),
        frac_constant_dosage_phase_varies=float(both.mean()),
    ), open(f'{D}/asc_only_triage.json', 'w'), indent=1)


if __name__ == '__main__':
    main()

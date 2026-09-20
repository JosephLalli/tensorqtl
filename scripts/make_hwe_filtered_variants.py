"""Produce an HWE-filtered variant set as a standalone dataset.

This deliberately does NOT modify the pipeline. It writes a variant-level
table and a pass list to disk; analysis scripts opt in by calling
apply_hwe_filter, so the production path is unchanged and every figure can
state which variant set it used.

Why: the tested set is filtered at MAF >= 0.05, but MAF counts alleles
while the allelic channel consumes heterozygotes. Variants whose minor
alleles sit in homozygotes pass MAF and supply almost no heterozygotes. The
123 variants with two or fewer heterozygotes have a median of 1 observed
against 12.9 expected, and 6 alternate homozygotes against 0.53 expected;
all 123 fail Hardy-Weinberg at p < 1e-6.

The Hardy-Weinberg test is a one-degree-of-freedom chi-square on the
genotype counts against the proportions implied by the allele frequency.
An exact test is preferable at small counts; at 92 donors and MAF >= 0.05
the chi-square is adequate for a screen, and the threshold is deliberately
permissive so it removes artifacts rather than real variants.
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import chi2

sys.path.insert(0, 'scripts')
sys.path.insert(0, '.')

OUT = '/mnt/ssd/lalli/brainvar_hapmix_deploy/hwe_filtered_variants_20260920'
HWE_P = 1e-6


def hwe_pvalues(dos):
    """One-df chi-square HWE p per variant from a [V, N] dosage matrix."""
    n_aa = (dos == 0).sum(1)
    n_ab = (dos == 1).sum(1)
    n_bb = (dos == 2).sum(1)
    tot = n_aa + n_ab + n_bb
    p = (2 * n_aa + n_ab) / (2 * tot)
    q = 1 - p
    e_aa, e_ab, e_bb = p ** 2 * tot, 2 * p * q * tot, q ** 2 * tot
    with np.errstate(divide='ignore', invalid='ignore'):
        x2 = ((n_aa - e_aa) ** 2 / np.maximum(e_aa, 1e-9)
              + (n_ab - e_ab) ** 2 / np.maximum(e_ab, 1e-9)
              + (n_bb - e_bb) ** 2 / np.maximum(e_bb, 1e-9))
    return chi2.sf(x2, 1), n_aa, n_ab, n_bb, e_ab


def apply_hwe_filter(I, path=None):
    """Opt-in: restrict a loaded input dict to HWE-passing variants.

    Returns a shallow copy with idx narrowed. Analysis scripts call this
    explicitly so the variant set in use is always visible at the call site.
    """
    path = path or f'{OUT}/pass_variant_ids.txt'
    keep_ids = set(open(path).read().split())
    vdf = I['vdf'].iloc[I['idx']]
    mask = np.array([str(v) in keep_ids for v in vdf.index])
    J = dict(I)
    J['idx'] = I['idx'][mask]
    J['hwe_filtered'] = True
    J['n_dropped_by_hwe'] = int((~mask).sum())
    return J


def main():
    from compare_mixqtl_replication import load_inputs
    os.makedirs(OUT, exist_ok=True)

    I = load_inputs()
    keep = I['keep']
    idx = I['idx']
    dos = I['dos'][idx][:, keep]
    xL, xR = I['xL'][idx][:, keep], I['xR'][idx][:, keep]
    vdf = I['vdf'].iloc[idx]

    hwe_p, n_aa, n_ab, n_bb, e_ab = hwe_pvalues(dos)
    af = dos.mean(1) / 2.0
    maf = np.minimum(af, 1 - af)
    het_phase = (xL != xR).sum(1)

    d = pd.DataFrame(dict(
        variant_id=vdf.index.astype(str), chrom=vdf['chrom'].values,
        pos=vdf['pos'].values, maf=maf,
        n_ref_hom=n_aa, n_het=n_ab, n_alt_hom=n_bb,
        n_het_expected=e_ab, n_het_phased=het_phase,
        hwe_p=hwe_p, pass_hwe=hwe_p >= HWE_P))
    d.to_csv(f'{OUT}/variant_hwe_stats.tsv', sep='\t', index=False)
    with open(f'{OUT}/pass_variant_ids.txt', 'w') as fh:
        fh.write('\n'.join(d.variant_id[d.pass_hwe]) + '\n')

    n, npass = len(d), int(d.pass_hwe.sum())
    print(f'{n:,} tested variants, {npass:,} pass HWE at p >= {HWE_P:g} '
          f'({100 * npass / n:.2f}%)')
    print(f'  dropped: {n - npass:,} ({100 * (n - npass) / n:.2f}%)')
    print(f'  MAF range retained: {d.maf[d.pass_hwe].min():.4f} to '
          f'{d.maf[d.pass_hwe].max():.4f}')
    print(f'\nheterozygote counts before and after:')
    print(f'  all tested : min {d.n_het.min()}, '
          f'median {d.n_het.median():.0f}')
    print(f'  HWE-passing: min {d.n_het[d.pass_hwe].min()}, '
          f'median {d.n_het[d.pass_hwe].median():.0f}')
    print(f'\n  variants with <=2 heterozygotes: '
          f'{int((d.n_het <= 2).sum())} before, '
          f'{int(((d.n_het <= 2) & d.pass_hwe).sum())} after')
    print(f'  variants with <=9 heterozygotes: '
          f'{int((d.n_het <= 9).sum())} before, '
          f'{int(((d.n_het <= 9) & d.pass_hwe).sum())} after')

    readme = f"""# HWE-filtered variant set, {pd.Timestamp.today():%Y-%m-%d}

Produced by `scripts/make_hwe_filtered_variants.py` on the same tested
variant set the comparison uses: the 1 Mb cis windows of the 29
null-calibration genes, outside gene bodies, MAF >= 0.05 over 92 donors.

NOT part of the pipeline. The production path is unchanged. Analysis
scripts opt in with `apply_hwe_filter(I)`.

Rationale. The MAF filter counts alleles; the allelic channel consumes
heterozygotes. A variant whose minor alleles sit in homozygotes passes MAF
and supplies almost no heterozygotes. The 123 variants with two or fewer
heterozygotes had a median of 1 observed against 12.9 expected, and 6
alternate homozygotes against 0.53 expected, and all 123 failed
Hardy-Weinberg at p < 1e-6.

Filter: one-degree-of-freedom chi-square HWE test on genotype counts,
retained at p >= {HWE_P:g}.

  tested   {n:,}
  retained {npass:,} ({100 * npass / n:.2f}%)
  dropped  {n - npass:,} ({100 * (n - npass) / n:.2f}%)

An exact test is preferable at small counts. At 92 donors with MAF >= 0.05
the chi-square is adequate for a screen, and the threshold is permissive by
design, to remove artifacts rather than real variants.

Files:
  variant_hwe_stats.tsv   every tested variant with its genotype counts,
                          expected heterozygotes, HWE p and pass flag
  pass_variant_ids.txt    the retained variant IDs, one per line
"""
    open(f'{OUT}/README.md', 'w').write(readme)
    json.dump(dict(n_tested=n, n_pass=npass, n_dropped=n - npass,
                   hwe_threshold=HWE_P,
                   min_het_before=int(d.n_het.min()),
                   min_het_after=int(d.n_het[d.pass_hwe].min())),
              open(f'{OUT}/summary.json', 'w'), indent=1)
    print(f'\nwrote {OUT}/')


if __name__ == '__main__':
    main()

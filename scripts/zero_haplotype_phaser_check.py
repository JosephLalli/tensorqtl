"""Are Salmon's zero-haplotype point estimates monoallelic expression?

Salmon's point estimate puts one haplotype copy at zero reads in 36% of
donor-gene pairs with haplotype-informative reads (docs/pipeline_rules.md).
One explanation is genuine monoallelic expression, imprinting above all. The
Gibbs draw means cannot test it: where the two copies share sequence the
sampler spreads ambiguous reads over both by construction, so a draw mean
above zero does not show that both alleles are expressed.

This uses the independent measurement: phASER's alignment-based counts of
reads at heterozygous SNPs, which a read either carries or does not, per
(gene, donor), gw_phased only, at least 20 such reads. For each pair it takes
the minor-allele fraction (smaller count + 1/2) / (both + 1). Monoallelic
expression puts it near 0; balanced expression near 0.5.

Two tests:
  1. per pair: minor-allele fraction of zero-haplotype pairs against pairs
     whose point estimate has reads on both copies, at matched read depth
  2. per gene: imprinting silences the same parental copy in nearly every
     heterozygous donor, so an imprinted gene should be one-copy in most of
     its donors; genes with >= 10 donors at >= 100 haplotype-informative
     reads and >= 80% of them one-copy are labelled imprinting-like

Output: <OUT>/summary.json and printed tables. No randomness.
"""
import json
from pathlib import Path

import numpy as np

import alignment_discordance_coupling as ADC

D = Path('/mnt/ssd/lalli/brainvar_hapmix_deploy')
CACHE = D / 'cache' / 'gibbs_56b63c3b37ed5df8'
OUT = D / 'zero_haplotype_phaser_check_20260925'
MIN_PHASER, GENE_MIN_DONORS, GENE_MIN_READS, GENE_ONE_COPY = 20, 10, 100, 0.8
BANDS = ((10, 100), (100, 1000), (1000, np.inf))


def main():
    OUT.mkdir(exist_ok=True)
    genes = (CACHE / 'genes.txt').read_text().split()
    samples = (CACHE / 'samples.txt').read_text().split()
    pL = np.load(CACHE / 'point_estimates' / 'pL.npy')
    pR = np.load(CACHE / 'point_estimates' / 'pR.npy')
    P, _ = ADC.load_phaser_all()
    ap, _, k, comp = ADC.phaser_arrays(P, genes, samples)   # ap = log((a+.5)/(b+.5))
    r = np.exp(-np.abs(ap))
    minor = r / (1 + r)
    n = pL + pR
    zero = (n > 0) & ((pL < 0.5) ^ (pR < 0.5))
    both = (n > 0) & (pL >= 0.5) & (pR >= 0.5)
    cov = comp & (k >= MIN_PHASER)

    hi = n >= GENE_MIN_READS
    nh, nz = hi.sum(1), (zero & hi).sum(1)
    ok = nh >= GENE_MIN_DONORS
    like = ok & (nz / np.maximum(nh, 1) >= GENE_ONE_COPY)
    order = np.argsort(-np.where(ok, nz / np.maximum(nh, 1), -1))
    summary = dict(
        n_zero_pairs_ge100=int((zero & hi).sum()),
        n_genes_evaluable=int(ok.sum()), n_imprinting_like_genes=int(like.sum()),
        share_of_zero_pairs_ge100_in_imprinting_like_genes=float((zero & hi)[like].sum() / (zero & hi).sum()),
        imprinting_like_genes=[f'{genes[i]} {int(nz[i])}/{int(nh[i])}' for i in order[:int(like.sum())]],
        n_zero_pairs_ge100_with_phaser=int((zero & hi & cov).sum()),
        bands={})
    print(f"imprinting-like genes: {summary['n_imprinting_like_genes']} of "
          f"{summary['n_genes_evaluable']}, holding "
          f"{summary['share_of_zero_pairs_ge100_in_imprinting_like_genes']:.1%} of zero pairs at >= 100 reads")
    for lo, up in BANDS:
        band = (n >= lo) & (n < up) & cov
        for lab, m in (('zero, imprinting-like gene', zero & band & like[:, None]),
                       ('zero, other genes', zero & band & ~like[:, None]),
                       ('control, reads on both copies', both & band)):
            x = minor[m]
            row = dict(n=int(x.size), median=float(np.median(x)) if x.size else None,
                       share_below_0_05=float((x < 0.05).mean()) if x.size else None,
                       share_above_0_15=float((x > 0.15).mean()) if x.size else None)
            summary['bands'][f'[{lo},{up}) {lab}'] = row
            if x.size:
                print(f'  [{lo},{up}) {lab:30s} n={x.size:>7,}  median {row["median"]:.3f}  '
                      f'< 0.05: {row["share_below_0_05"]:.1%}  > 0.15: {row["share_above_0_15"]:.1%}')
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()

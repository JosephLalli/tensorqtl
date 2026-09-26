"""What does dropping zero-haplotype pairs from the allelic channel cost?

Proposal (user, 2026-09-25): when Salmon's point estimate puts one haplotype of
a donor-gene pair at zero, drop that pair from the allelic channel; the total
channel keeps it. Measured on the calibration gene set that has Gibbs draws:

  - how many informative pairs the rule removes, exactly-zero and below 0.5
    reads, by haplotype-informative read band
  - the share of the allelic channel's weight they carry, with the shipped
    weight 1 / (Gibbs variance + counting term at the point estimate)
  - how many genes fall below 20 and 10 informative donors
No randomness.
"""
import json
from pathlib import Path

import numpy as np

D = Path('/mnt/ssd/lalli/brainvar_hapmix_deploy')
CACHE = D / 'cache' / 'gibbs_56b63c3b37ed5df8'
OUT = D / 'drop_zero_haplotype_cost_20260925'
KAPPA, LN2 = 0.5, np.log(2.0)


def main():
    OUT.mkdir(exist_ok=True)
    genes = (CACHE / 'genes.txt').read_text().split()
    cal = set((CACHE / 'point_estimates' / 'edger' / 'calibration_genes.txt').read_text().split())
    rows = np.array([i for i, g in enumerate(genes) if g in cal])
    pL = np.load(CACHE / 'point_estimates' / 'pL.npy')[rows]
    pR = np.load(CACHE / 'point_estimates' / 'pR.npy')[rows]
    YL = np.load(CACHE / 'YL.npy', mmap_mode='r'); YR = np.load(CACHE / 'YR.npy', mmap_mode='r')
    gv = np.empty(pL.shape)
    for s in range(0, len(rows), 1000):
        r = rows[s:s + 1000]
        gv[s:s + 1000] = np.log2((np.asarray(YL[r]) + KAPPA) / (np.asarray(YR[r]) + KAPPA)).var(2)
    n = pL + pR
    inf = n > 0                                    # the no_cov guard's informative set
    w = np.where(inf, 1 / (gv + (1 / (pL + KAPPA) + 1 / (pR + KAPPA)) / LN2 ** 2), 0.0)
    zero_exact = inf & ((pL == 0) ^ (pR == 0))
    zero = inf & ((pL < 0.5) ^ (pR < 0.5))
    out = dict(n_genes=int(len(rows)), n_informative_pairs=int(inf.sum()),
               n_zero_exact=int(zero_exact.sum()), n_zero_below_half=int(zero.sum()),
               share_pairs_removed=float(zero.sum() / inf.sum()),
               share_weight_removed=float(w[zero].sum() / w.sum()),
               median_gene_share_weight_removed=float(np.median(
                   w[inf.any(1)].__mul__(zero[inf.any(1)]).sum(1) / w[inf.any(1)].sum(1))),
               bands={})
    print(f"{len(rows):,} calibration genes with draws; informative pairs {inf.sum():,}; "
          f"one haplotype exactly 0: {zero_exact.sum():,}; below 0.5: {zero.sum():,} "
          f"({out['share_pairs_removed']:.1%}); allelic weight removed {out['share_weight_removed']:.2%}; "
          f"median per-gene weight removed {out['median_gene_share_weight_removed']:.2%}")
    for lab, lo, hi in (('1-9', 0, 10), ('10-99', 10, 100), ('100-999', 100, 1000), ('1000+', 1000, np.inf)):
        m = inf & (n >= lo) & (n < hi)
        out['bands'][lab] = dict(pairs=int(m.sum()), share_removed=float((zero & m).sum() / m.sum()),
                                 share_band_weight_removed=float(w[zero & m].sum() / w[m].sum()))
        print(f"  {lab:8s} pairs {m.sum():>9,}  removed {out['bands'][lab]['share_removed']:6.1%}  "
              f"of that band's weight {out['bands'][lab]['share_band_weight_removed']:6.2%}")
    na0, na1 = inf.sum(1), (inf & ~zero).sum(1)
    for t in (20, 10):
        out[f'genes_ge_{t}_informative_before'] = int((na0 >= t).sum())
        out[f'genes_ge_{t}_informative_after'] = int((na1 >= t).sum())
        print(f"  genes with >= {t} informative donors: {int((na0 >= t).sum()):,} -> {int((na1 >= t).sum()):,}")
    (OUT / 'summary.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()

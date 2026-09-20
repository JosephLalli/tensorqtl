"""How can a MAF >= 0.05 variant have one heterozygous carrier?

The tested set is filtered at MAF >= 0.05 over all 92 donors, which implies
roughly 2*f*(1-f)*92 = 8.7 heterozygotes. Carrier counts of one were
nonetheless observed. Two candidate explanations:

  filtering error     the MAF threshold is not doing what it appears to
  subset attrition    carriers are counted within each gene's INFORMATIVE
                      donor set (Va > 0), not over all 92, and that subset
                      is both smaller and not independent of genotype

This separates them by reporting, for the low-carrier variants, the MAF and
the heterozygote count over all 92 donors alongside the carrier count in
the informative subset.
"""

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, 'scripts')
sys.path.insert(0, '.')

OUT = '/mnt/ssd/lalli/brainvar_hapmix_deploy/mixqtl_replication_20260919'


def main():
    import tensorqtl.hapmixqtl as HM
    from compare_mixqtl_replication import load_inputs, gene_variant_index

    I = load_inputs()
    genes, keep = I['genes'], I['keep']
    _A, _T, Va, _Vt, _C = HM.compute_summaries_from_gibbs(
        I['YL'], I['YR'], yT=I['YT'], count_noise=True)
    Va = Va[:, keep]

    dos = I['dos'][I['idx']][:, keep]          # [V, 92]
    xL = I['xL'][I['idx']][:, keep]
    xR = I['xR'][I['idx']][:, keep]
    het_all = (xL != xR)                        # het over all 92
    af = dos.mean(1) / 2.0
    maf = np.minimum(af, 1 - af)

    print(f'tested variants: {dos.shape[0]:,}, donors: {dos.shape[1]}')
    print(f'MAF threshold applied at load time: 0.05')
    print(f'  min MAF actually present: {maf.min():.5f}')
    print(f'  variants below 0.05: {int((maf < 0.05).sum())}')
    print(f'  heterozygotes over all 92, by MAF: '
          f'min {het_all.sum(1).min()}, median {np.median(het_all.sum(1)):.0f}\n')

    rows = []
    for j, g in enumerate(genes):
        vsel = gene_variant_index(I, g)
        if vsel.size == 0:
            continue
        inf = Va[j] > 1e-12
        n_inf = int(inf.sum())
        if n_inf <= 2:
            continue
        sub = (xL[vsel] != xR[vsel])
        ncar = sub[:, inf].sum(1)               # carriers within informative
        rows.append(pd.DataFrame(dict(
            gene=g, n_inf=n_inf, carriers=ncar,
            het_all92=sub.sum(1), maf=maf[vsel])))
    d = pd.concat(rows, ignore_index=True)

    print('for variants with few carriers in the informative subset:')
    print(f'{"carriers":>9s} {"n":>7s} {"median MAF":>11s} '
          f'{"median het/92":>14s} {"median n_inf":>13s}')
    for k in [1, 2, 3, 4]:
        gg = d[d.carriers == k]
        if len(gg):
            print(f'{k:>9d} {len(gg):7,} {gg.maf.median():11.4f} '
                  f'{gg.het_all92.median():14.0f} {gg.n_inf.median():13.0f}')
    gg = d[d.carriers >= 25]
    print(f'{">=25":>9s} {len(gg):7,} {gg.maf.median():11.4f} '
          f'{gg.het_all92.median():14.0f} {gg.n_inf.median():13.0f}')

    one = d[d.carriers == 1]
    print(f'\nthe {len(one)} single-carrier cases:')
    print(f'  MAF range {one.maf.min():.4f} to {one.maf.max():.4f} '
          f'(all >= 0.05: {bool((one.maf >= 0.05).all())})')
    print(f'  heterozygotes over all 92: median {one.het_all92.median():.0f}, '
          f'range {one.het_all92.min()} to {one.het_all92.max()}')
    print(f'  informative donors in their gene: median {one.n_inf.median():.0f}')
    print(f'  so the attrition is {one.het_all92.median():.0f} hets over 92 '
          f'-> 1 within the informative subset')
    print(f'\n  genes contributing them: '
          f'{one.gene.value_counts().head(5).to_dict()}')


if __name__ == '__main__':
    main()

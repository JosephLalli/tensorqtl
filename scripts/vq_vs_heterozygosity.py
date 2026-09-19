"""Does v/q fall toward 1 as the two haplotypes become more distinguishable?

The proposed mechanism: the Gibbs posterior spreads wider than the Poisson
prediction because most reads cannot be assigned to a haplotype. Where the
two haplotypes differ at MANY sites, reads are assignable, ambiguity is
small, and v should approach q. Where they differ at few sites, v >> q.

That is a testable prediction and it is directional: log(v/q) should fall
as the number of heterozygous sites in the gene rises.

Number of heterozygous variants inside the gene body is used as the proxy
for distinguishability. It is a proxy, not the quantity itself -- what
matters for read assignment is het sites in TRANSCRIBED and covered
sequence, and a variant in an intron or an unexpressed exon distinguishes
nothing. So a weak relationship would be ambiguous between "the mechanism
is wrong" and "the proxy is poor", while a strong one is informative.

Deterministic; no permutation, no null, no seed.
"""

import json
import sys

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, 'scripts')
sys.path.insert(0, '.')

OUT = '/mnt/ssd/lalli/brainvar_hapmix_deploy/mixqtl_replication_20260919'
KAPPA = 0.5


def main():
    import tensorqtl.hapmixqtl as HM
    from compare_mixqtl_replication import load_inputs

    I = load_inputs()
    keep, genes, gp, vdf = I['keep'], I['genes'], I['gp'], I['vdf']
    _A, _T, v_all, _Vt, _C = HM.compute_summaries_from_gibbs(
        I['YL'], I['YR'], yT=I['YT'], count_noise=False)
    v = v_all[:, keep]
    mL, mR = I['YL'][:, keep].mean(2), I['YR'][:, keep].mean(2)
    q = 1.0 / (mL + KAPPA) + 1.0 / (mR + KAPPA)

    # heterozygous variants inside each gene body, per donor
    xL = I['xL'][:, keep]
    xR = I['xR'][:, keep]
    het = (xL != xR)
    pos = vdf['pos'].values
    ch = vdf['chrom'].values.astype(str)

    rows = []
    for j, g in enumerate(genes):
        r = gp.loc[g]
        inbody = ((ch == str(r['chr'])) & (pos >= int(r['start']))
                  & (pos <= int(r['end'])))
        nhet = het[inbody].sum(0) if inbody.any() else np.zeros(v.shape[1], int)
        inf = v[j] > 1e-12
        for i in np.where(inf)[0]:
            rows.append((g, int(nhet[i]), float(v[j, i] / q[j, i]),
                         float(mL[j, i] + mR[j, i])))

    import pandas as pd
    d = pd.DataFrame(rows, columns=['gene', 'n_het', 'vq', 'total_count'])
    d['log_vq'] = np.log(d.vq)
    d.to_parquet(f'{OUT}/vq_vs_heterozygosity.parquet')

    print(f'{len(d)} informative donor-gene pairs, '
          f'{d.gene.nunique()} genes\n')
    print(f'heterozygous variants in gene body per donor: '
          f'median {d.n_het.median():.0f}, '
          f'range {d.n_het.min()}-{d.n_het.max()}\n')

    print('v/q by number of heterozygous sites in the gene body:')
    d['bin'] = pd.cut(d.n_het, [-0.5, 0.5, 2.5, 5.5, 10.5, 20.5, 1e9],
                      labels=['0', '1-2', '3-5', '6-10', '11-20', '>20'])
    for b, gdf in d.groupby('bin', observed=True):
        print(f'  {b:>6s} sites  n={len(gdf):5d}  median v/q '
              f'{gdf.vq.median():8.2f}   median total count '
              f'{gdf.total_count.median():8.0f}')

    rho_all = spearmanr(d.n_het, d.log_vq)
    print(f'\npooled Spearman(n_het, log v/q) = {rho_all[0]:+.4f} '
          f'(p={rho_all[1]:.2e})')

    # within gene, so gene-level expression and length are held fixed
    within = []
    for g, gdf in d.groupby('gene'):
        if gdf.n_het.nunique() > 2 and len(gdf) > 10:
            within.append(spearmanr(gdf.n_het, gdf.log_vq)[0])
    within = np.array(within)
    print(f'within-gene Spearman: median {np.median(within):+.4f}, '
          f'{int((within < 0).sum())}/{len(within)} genes negative')
    print('  (within gene, expression and transcript structure are held '
          'fixed,\n   so this isolates donor-level distinguishability)')

    json.dump(dict(
        n=len(d), pooled_spearman=float(rho_all[0]),
        pooled_p=float(rho_all[1]),
        within_gene_median_spearman=float(np.median(within)),
        within_gene_n_negative=int((within < 0).sum()),
        within_gene_n=len(within),
        median_vq_by_bin={str(b): float(gdf.vq.median())
                          for b, gdf in d.groupby('bin', observed=True)},
    ), open(f'{OUT}/vq_vs_heterozygosity.json', 'w'), indent=1)


if __name__ == '__main__':
    main()

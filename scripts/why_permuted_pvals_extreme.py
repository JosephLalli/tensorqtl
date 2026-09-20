"""Why do permuted replicates so often produce a minimum-p of exactly zero?

For the weighted through-origin fit with s in {-1, 0, +1}, write z_i = s_i
a_i over the CARRIERS only (donors with s != 0). Then

    beta_hat = sum(w z) / sum(w)          a weighted mean of z
    s_i e_i  = z_i - beta_hat
    meat     = sum(w^2 (z - beta_hat)^2)
    se^2     = meat / (sum w)^2

so the sandwich variance is the weighted spread of z among carriers. It
approaches zero when those values coincide, which happens by two routes:

  few carriers        with one carrier the fit is exact and se^2 is
                      identically zero; with two or three it is near zero
                      whenever the values happen to agree

  weight concentration if one carrier's weight dominates, beta_hat sits on
                      top of that carrier, its residual vanishes, and the
                      remaining terms are multiplied by small weights

The Gibbs weights span roughly 323-fold within a gene, so the second route
is available even at moderate carrier counts. This measures which one
operates, by recording the carrier count and the weight concentration of
the variant achieving the minimum p in each permutation.

Weight concentration is the largest carrier weight over the sum of carrier
weights: 1/k if all equal, approaching 1 if one dominates.
"""

import sys

import numpy as np
import pandas as pd
from scipy.stats import t as tdist

sys.path.insert(0, 'scripts')
sys.path.insert(0, '.')

SEED = 42
NPERM = 100


def main():
    import tensorqtl.hapmixqtl as HM
    from compare_mixqtl_replication import load_inputs, gene_variant_index
    from se_forms_calibration import fit_all_se

    I = load_inputs()
    genes, keep = I['genes'], I['keep']
    A, _T, Va, _Vt, _C = HM.compute_summaries_from_gibbs(
        I['YL'], I['YR'], yT=I['YT'], count_noise=True)
    A, Va = A[:, keep], Va[:, keep]
    s_all = (I['xL'] - I['xR'])[I['idx']][:, keep]

    recs = []
    for j, g in enumerate(genes[:8]):
        vsel = gene_variant_index(I, g)
        if vsel.size == 0:
            continue
        inf = Va[j] > 1e-12
        n = int(inf.sum())
        if n <= 2:
            continue
        a = A[j][inf]
        S = s_all[vsel][:, inf]
        S = S[S.var(1) > 0]
        if S.shape[0] < 50:
            continue
        X = S.T
        w = 1.0 / np.maximum(Va[j][inf], 1e-12)
        dof = max(n - 1, 1)

        carriers = (X != 0)
        ncar = carriers.sum(0)
        # weight concentration per variant: max carrier weight / total
        wc = np.array([w[carriers[:, v]].max() / w[carriers[:, v]].sum()
                       if ncar[v] > 0 else np.nan
                       for v in range(X.shape[1])])

        for k in range(NPERM):
            prm = np.random.RandomState(SEED + 90001 + k).permutation(n)
            b, _k, _f2, s2, _h = fit_all_se(a[prm], X, w[prm])
            ok = np.isfinite(b) & (s2 > 0)
            if ok.sum() == 0:
                continue
            t2 = np.full(X.shape[1], np.nan)
            t2[ok] = b[ok] ** 2 / s2[ok]
            v = int(np.nanargmax(t2))
            p = float(2 * tdist.sf(np.sqrt(t2[v]), dof))
            # concentration must be recomputed on the PERMUTED weights
            cw = w[prm][carriers[:, v]]
            recs.append(dict(gene=g, perm=k, p=p, is_zero=(p == 0.0),
                             carriers=int(ncar[v]),
                             conc=float(cw.max() / cw.sum())))
    d = pd.DataFrame(recs)

    print(f'{len(d)} permutations across {d.gene.nunique()} genes\n')
    print(f'permutations whose minimum p is exactly 0: '
          f'{int(d.is_zero.sum())}/{len(d)} ({100 * d.is_zero.mean():.1f}%)\n')
    print('the variant achieving the minimum p, by whether p underflowed:')
    for z, gg in d.groupby('is_zero'):
        lab = 'p == 0' if z else 'p > 0'
        print(f'  {lab:8s} n={len(gg):5d}   median carriers '
              f'{gg.carriers.median():5.0f}   '
              f'median weight concentration {gg.conc.median():.3f}')
    print()
    print('carrier-count composition of the underflowing minima:')
    z = d[d.is_zero]
    if len(z):
        for lo, hi, lab in [(0, 1, '1'), (2, 2, '2'), (3, 4, '3-4'),
                            (5, 9, '5-9'), (10, 10 ** 9, '>=10')]:
            k = ((z.carriers >= lo) & (z.carriers <= hi)).sum()
            print(f'  {lab:>5s} carriers: {k:5d} ({100 * k / len(z):5.1f}%)')
        print(f'\n  weight concentration among them: median '
              f'{z.conc.median():.3f}, '
              f'90th pct {z.conc.quantile(.9):.3f}')
        print(f'  concentration if carriers were equally weighted would be '
              f'1/k = {1 / max(z.carriers.median(), 1):.3f}')


if __name__ == '__main__':
    main()

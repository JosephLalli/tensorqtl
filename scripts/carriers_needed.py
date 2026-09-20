"""How many heterozygous carriers does the allelic channel need?

The arithmetic minima are k >= 1 for beta_hat to exist and k >= 2 for a
residual-based variance to be nonzero. Neither says when the estimate is
trustworthy. That is measured here by calibration stratified on carrier
count: var(beta_hat) across 40 null permutations against mean(se^2), for
both standard-error forms.

Under permutation the true slope is zero, so the spread of beta_hat IS the
estimation error. If a carrier count is adequate, the reported se^2 should
track that spread at whatever level the form achieves overall; if it is
inadequate, the ratio departs from that level.

Also reported: the realized precision itself, var(beta_hat) by carrier
count, which says how much the allelic channel can contribute at each k
regardless of whether its se is honest.

Deterministic given the seed; master seed 42.
"""

import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, 'scripts')
sys.path.insert(0, '.')

OUT = '/mnt/ssd/lalli/brainvar_hapmix_deploy/mixqtl_replication_20260919'
NP_NULL = 40
SEED = 42
BINS = [0, 1, 2, 4, 9, 14, 24, 10 ** 9]
LABELS = ['1', '2', '3-4', '5-9', '10-14', '15-24', '>=25']


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

    parts = []
    for j, g in enumerate(genes):
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
        if S.shape[0] < 20:
            continue
        X = S.T
        w = 1.0 / np.maximum(Va[j][inf], 1e-12)
        ncar = (X != 0).sum(0)

        P = X.shape[1]
        b = np.empty((NP_NULL, P))
        k_acc = np.empty((NP_NULL, P))
        f_acc = np.empty((NP_NULL, P))
        for pi in range(NP_NULL):
            prm = np.random.RandomState(SEED + 10007 + pi).permutation(n)
            bb, k1, k2, _s, _h = fit_all_se(a[prm], X, w[prm])
            b[pi] = bb
            k_acc[pi] = k1
            f_acc[pi] = k2
        vb = np.var(b, axis=0, ddof=1)
        mk = np.nanmean(k_acc, axis=0)
        mf = np.nanmean(f_acc, axis=0)
        ok = np.isfinite(vb) & (mk > 0) & (mf > 0)
        parts.append(pd.DataFrame(dict(
            gene=g, carriers=ncar[ok], var_beta=vb[ok],
            calib_known=vb[ok] / mk[ok], calib_fitted=vb[ok] / mf[ok])))

    d = pd.concat(parts, ignore_index=True)
    d['bin'] = pd.cut(d.carriers, BINS, labels=LABELS)
    d.to_parquet(f'{OUT}/carriers_needed.parquet')

    print(f'{len(d):,} variants across {d.gene.nunique()} genes\n')
    print(f'{"carriers":>9s} {"n":>8s} {"share":>7s} '
          f'{"var(beta)":>11s} {"calib known":>12s} {"calib fitted":>13s}')
    rows = []
    for lab, gg in d.groupby('bin', observed=True):
        print(f'{lab:>9s} {len(gg):8,} {100*len(gg)/len(d):6.1f}% '
              f'{gg.var_beta.median():11.5f} '
              f'{gg.calib_known.median():12.2f} '
              f'{gg.calib_fitted.median():13.2f}')
        rows.append(dict(bin=lab, n=int(len(gg)),
                         median_var_beta=float(gg.var_beta.median()),
                         calib_known=float(gg.calib_known.median()),
                         calib_fitted=float(gg.calib_fitted.median())))

    ref = d[d.carriers >= 25]
    print(f'\nreference level from the >=25 bin: '
          f'known {ref.calib_known.median():.2f}, '
          f'fitted {ref.calib_fitted.median():.2f}')
    print('a bin whose calibration departs from that level is one where the')
    print('reported uncertainty stops tracking the realized spread.\n')

    print('precision relative to the >=25 bin (median var(beta) ratio):')
    base = ref.var_beta.median()
    for lab, gg in d.groupby('bin', observed=True):
        print(f'  {lab:>7s}  {gg.var_beta.median() / base:6.2f}x')

    json.dump(dict(n=int(len(d)), bins=rows,
                   ref_calib_known=float(ref.calib_known.median()),
                   ref_calib_fitted=float(ref.calib_fitted.median())),
              open(f'{OUT}/carriers_needed.json', 'w'), indent=1)


if __name__ == '__main__':
    main()

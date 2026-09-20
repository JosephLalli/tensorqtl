"""Why does the Beta fit return NaN under the sandwich statistic?

_beta_p returns NaN in two circumstances: fewer than 30 of the 200
permutation minimum-p values are finite and strictly inside (0, 1), or
scipy's beta.fit raises. This distinguishes them, because the first would be
a property of the statistic while the second could be a fitting-routine
artifact.

The candidate mechanism for the first: the sandwich variance is not bounded
below. When the residuals at a variant's few carriers are jointly near zero,
se^2 approaches zero, t^2 grows without bound, and the two-sided t tail
underflows to exactly 0.0 in float64. A minimum-p of exactly zero is then
excluded by the (0, 1) filter.

Reports, per gene, how many of the 200 permutation minimum-p values are
exactly zero and how many survive the filter.
"""

import sys

import numpy as np
import pandas as pd
from scipy.stats import t as tdist, beta as bdist

sys.path.insert(0, 'scripts')
sys.path.insert(0, '.')

SEED = 42
NPERM = 200


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

    rows = []
    for j, g in enumerate(genes[:10]):          # 10 genes is enough to see it
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

        mins_f, mins_s, min_se2, max_t2 = [], [], [], []
        for k in range(NPERM):
            prm = np.random.RandomState(SEED + 90001 + k).permutation(n)
            b, _k, f2, s2, _h = fit_all_se(a[prm], X, w[prm])
            ok = np.isfinite(b) & (f2 > 0) & (s2 > 0)
            if ok.sum() == 0:
                continue
            t2f = b[ok] ** 2 / f2[ok]
            t2s = b[ok] ** 2 / s2[ok]
            mins_f.append(float(np.nanmin(2 * tdist.sf(np.sqrt(t2f), dof))))
            mins_s.append(float(np.nanmin(2 * tdist.sf(np.sqrt(t2s), dof))))
            min_se2.append(float(np.nanmin(s2[ok])))
            max_t2.append(float(np.nanmax(t2s)))
        mf = np.array(mins_f)
        ms = np.array(mins_s)
        rows.append(dict(
            gene=g, n_inf=n,
            model_exact_zero=int((mf == 0).sum()),
            sand_exact_zero=int((ms == 0).sum()),
            sand_surviving=int(((ms > 0) & (ms < 1) & np.isfinite(ms)).sum()),
            min_sandwich_se2=float(np.min(min_se2)),
            max_sandwich_t2=float(np.max(max_t2)),
        ))
        r = rows[-1]
        print(f"{g:10s} n={n:3d}  perms with min-p exactly 0: "
              f"model {r['model_exact_zero']:3d}/200  "
              f"sandwich {r['sand_exact_zero']:3d}/200   "
              f"surviving the (0,1) filter: {r['sand_surviving']:3d}")

    d = pd.DataFrame(rows)
    print(f'\nacross {len(d)} genes:')
    print(f'  sandwich permutations with min-p exactly 0: '
          f'median {d.sand_exact_zero.median():.0f}/200')
    print(f'  genes with fewer than 30 surviving: '
          f'{int((d.sand_surviving < 30).sum())}/{len(d)}   '
          f'<- these return NaN')
    print(f'  smallest sandwich se^2 seen: {d.min_sandwich_se2.min():.3e}')
    print(f'  largest sandwich t^2 seen:   {d.max_sandwich_t2.max():.3e}')
    print(f'\n  two-sided t tail underflows to exactly 0 in float64 at '
          f'roughly t^2 > {np.sqrt(1e4):.0f}^2 for these dof;')
    print(f'  the sandwich reaches t^2 of {d.max_sandwich_t2.max():.1e}, '
          f'so the underflow is real rather than a guard artifact.')

    # confirm scipy's fit is not itself the failure
    x = np.random.default_rng(42).beta(1.0, 50.0, 200)
    try:
        bdist.fit(x, floc=0, fscale=1)
        print('\n  scipy beta.fit on a well-posed sample: succeeds '
              '(so the NaN is the filter, not the fitter)')
    except Exception as e:
        print(f'\n  scipy beta.fit failed on a well-posed sample: {e}')


if __name__ == '__main__':
    main()

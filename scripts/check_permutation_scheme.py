"""Two checks on the null permutation used by the weighting ablation.

CHECK 1 -- bias. The allelic design is fitted through the origin and is not
centred, so the permutation mean of beta_hat need not be exactly zero. The
ablation compares var(beta_hat) across arms; if the arms differed in their
permutation BIAS, comparing variance alone would be incomplete. This measures
|mean(beta_hat)| against sd(beta_hat) per arm.

CHECK 2 -- seed sharing. The ablation draws its permutation as
RandomState(10007 + i).permutation(n_inf), so two genes with the same
informative-donor count receive the SAME donor permutation. Their data are
otherwise unrelated, so the induced dependence should be negligible, but the
sign test across 29 genes assumes independence. This re-runs the ablation
with a gene-specific seed and compares.
"""

import collections
import sys

import numpy as np

sys.path.insert(0, 'scripts')
sys.path.insert(0, '.')

from compare_mixqtl_replication import load_inputs, gene_variant_index  # noqa: E402
import tensorqtl.hapmixqtl as HM                                        # noqa: E402
from tensorqtl import mixqtl_replication as MX                          # noqa: E402

NP_NULL = 40


def main():
    I = load_inputs()
    keep, genes = I['keep'], I['genes']
    A, _T, Va, _Vt, _C = HM.compute_summaries_from_gibbs(
        I['YL'], I['YR'], yT=I['YT'], count_noise=True)
    _a, _t, Vnq, _v, _c = HM.compute_summaries_from_gibbs(
        I['YL'], I['YR'], yT=I['YT'], count_noise=False)
    A, Va, Vnq = A[:, keep], Va[:, keep], Vnq[:, keep]
    mL, mR = I['YL'].mean(2)[:, keep], I['YR'].mean(2)[:, keep]
    s_all = (I['xL'] - I['xR'])[I['idx']][:, keep]

    n_infs = []
    res = {}
    for shared in (True, False):
        tot = {}
        for j, g in enumerate(genes):
            vsel = gene_variant_index(I, g)
            inf = (Va[j] > 1e-12) & (Vnq[j] > 1e-12)
            n = int(inf.sum())
            if vsel.size == 0 or n <= 2:
                continue
            if shared:
                n_infs.append(n)
            a = A[j][inf]
            S = s_all[vsel][:, inf]
            S = S[S.var(1) > 0]
            if S.shape[0] == 0:
                continue
            X = S.T
            wg = 1.0 / np.maximum(Va[j][inf], 1e-12)
            wh, _, _ = MX.apply_weight_cap(
                MX.harmonic_weights(np.maximum(mL[j][inf], 1e-12),
                                    np.maximum(mR[j][inf], 1e-12)),
                n, MX.WEIGHT_CAP)
            arms = {'gibbs': wg, 'harm_cap': wh, 'equal': np.ones(n)}
            bb = {k: np.empty((NP_NULL, X.shape[1])) for k in arms}
            for pi in range(NP_NULL):
                seed = 10007 + pi if shared else 10007 + pi + 9973 * j
                prm = np.random.RandomState(seed).permutation(n)
                for k, w in arms.items():
                    b, _ = MX._simple_regression_through_origin(a[prm], X, w[prm])
                    bb[k][pi] = b
            for k in arms:
                mb = np.nanmedian(np.abs(np.nanmean(bb[k], 0)))
                vr = np.nanmedian(np.nanvar(bb[k], 0, ddof=1))
                tot.setdefault(k, []).append((mb, np.sqrt(vr), vr))
        res[shared] = tot

    for shared in (True, False):
        tag = 'SHARED seeds (as the ablation ran)' if shared else 'PER-GENE seeds'
        print(f'--- {tag} ---')
        base = np.median([x[2] for x in res[shared]['equal']])
        for k, v in res[shared].items():
            mb = np.median([x[0] for x in v])
            sd = np.median([x[1] for x in v])
            vr = np.median([x[2] for x in v])
            print(f'  {k:9s} |perm mean beta| {mb:.5f}   sd(beta) {sd:.5f}   '
                  f'|mean|/sd {mb / sd:.3f}   var ratio vs OLS {vr / base:.3f}')

    c = collections.Counter(n_infs)
    print(f'\ngenes sharing an n_inf with another gene: '
          f'{sum(v for v in c.values() if v > 1)}/{sum(c.values())}')


if __name__ == '__main__':
    main()

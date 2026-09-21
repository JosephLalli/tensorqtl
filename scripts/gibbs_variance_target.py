"""Which Gibbs variance should weight the allelic channel?

Three candidates, and they are not all estimating the same thing:

  1/v_ratio        v_ratio = Var_draws(log(yL+k) - log(yR+k)), the DIRECT
                   across-draw variance of the quantity actually regressed.
                   What compute_summaries_from_gibbs computes.
  1/(1/y1 + 1/y2)  mixQTL's harmonic weight. A different ROUTE to the same
                   target: for a binomial split at fixed total, and for
                   independent Poisson counts alike, the delta-method
                   variance of the log ratio is 1/y1 + 1/y2.
  1/(vL + vR)      the sum of the two marginal per-haplotype draw variances.
                   Equals Var(log ratio) ONLY if Cov(log yL, log yR) = 0.

The third stands or falls on that covariance, so this measures it. It also
separates the LEVEL difference from the WITHIN-GENE SHAPE difference,
because the two have completely different consequences: a gene-constant
factor cancels from a freely fitted sigma and from a within-gene permutation
p-value, whereas a factor that varies across donors within a gene does not
cancel from anything and changes the estimate.
"""
import sys
import os

import numpy as np
from scipy.stats import pearsonr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from compare_mixqtl_replication import load_inputs

KAPPA = 0.5


def main():
    I = load_inputs()
    keep, genes = I['keep'], I['genes']
    YL, YR = I['YL'][:, keep, :], I['YR'][:, keep, :]
    lL, lR = np.log(YL + KAPPA), np.log(YR + KAPPA)

    vL, vR = lL.var(2), lR.var(2)
    cov = ((lL - lL.mean(2, keepdims=True))
           * (lR - lR.mean(2, keepdims=True))).mean(2)
    v_ratio = (lL - lR).var(2)
    mL, mR = YL.mean(2), YR.mean(2)
    inf = (mL + mR) > 0

    print(f'informative donor-gene pairs: {inf.sum()} of {inf.size}')
    print('identity check  max|v_ratio - (vL + vR - 2cov)| = '
          f'{np.abs(v_ratio - (vL + vR - 2 * cov))[inf].max():.3e}')

    r = cov[inf] / np.sqrt(np.maximum(vL[inf] * vR[inf], 1e-300))
    print('\nCov(log yL, log yR) across draws:')
    print(f'  median covariance   {np.median(cov[inf]):+.5f}')
    print(f'  fraction NEGATIVE   {np.mean(cov[inf] < 0):.1%}')
    print(f'  median correlation  {np.median(r):+.3f}  '
          f'(10th {np.percentile(r, 10):+.3f}, 90th {np.percentile(r, 90):+.3f})')

    ratio = v_ratio[inf] / np.maximum(vL[inf] + vR[inf], 1e-300)
    print('\nLEVEL: v_ratio vs vL + vR')
    print(f'  median ratio  {np.median(ratio):.3f}   quartiles '
          f'{np.percentile(ratio, 25):.3f} / {np.percentile(ratio, 75):.3f}')
    print(f'  fraction where summing UNDERSTATES  {np.mean(ratio > 1):.1%}')

    R = (mL + mR)[inf]
    print('\n  by allele-resolved reads:')
    for lo, hi, lab in [(0, 10, '<10'), (10, 100, '10-100'),
                        (100, 1000, '100-1k'), (1000, 1e18, '1k+')]:
        m = (R >= lo) & (R < hi)
        if m.sum() > 20:
            print(f'    {lab:8s} n={m.sum():6d}  ratio {np.median(ratio[m]):.3f}'
                  f'  corr {np.median(r[m]):+.3f}')

    print('\nSHAPE within gene (a gene-constant factor cancels; this does not)')
    sd_log, fold, r_w, dev = [], [], [], []
    for j in range(len(genes)):
        m = inf[j]
        if m.sum() < 20:
            continue
        lr = np.log(v_ratio[j][m] / (vL[j][m] + vR[j][m]))
        sd_log.append(lr.std(ddof=1))
        fold.append(np.exp(np.percentile(lr, 90) - np.percentile(lr, 10)))
        a = -np.log(np.maximum(v_ratio[j][m], 1e-12))
        b = -np.log(np.maximum(vL[j][m] + vR[j][m], 1e-12))
        r_w.append(pearsonr(a, b)[0])
        dev.append(np.std((a - a.mean()) - (b - b.mean()), ddof=1))
    sd_log, fold = np.array(sd_log), np.array(fold)
    r_w, dev = np.array(r_w), np.array(dev)
    print(f'  genes used: {len(sd_log)}')
    print(f'  within-gene sd of log(v_ratio/(vL+vR))  median {np.median(sd_log):.4f}')
    print(f'  within-gene 10-90 fold spread of it     median {np.median(fold):.3f}x')
    print(f'  corr of the two log-weight vectors      median {np.median(r_w):.4f}'
          f'  (min {r_w.min():.4f})')
    print(f'  centred log-weight difference (shape)   median sd {np.median(dev):.4f}'
          f'  -> {np.exp(np.median(dev)):.3f}x typical donor disagreement')


if __name__ == '__main__':
    main()

"""Worked example: what v, q and v/q actually are, on one real donor-gene pair.

v and q are two estimates of the SAME quantity -- the variance of one
donor's allelic log ratio for one gene. v measures it from Salmon's
posterior draws; q predicts it from the posterior-mean counts under a
Poisson assumption. Their ratio is therefore unitless: observed spread over
predicted spread.
"""

import sys

import numpy as np

sys.path.insert(0, 'scripts')
sys.path.insert(0, '.')

KAPPA = 0.5


def main():
    from compare_mixqtl_replication import load_inputs
    I = load_inputs()
    keep, genes = I['keep'], I['genes']
    YL = I['YL'][:, keep]
    YR = I['YR'][:, keep]

    # pick a well-covered, informative donor-gene pair to illustrate
    mL, mR = YL.mean(2), YR.mean(2)
    ok = (mL > 200) & (mR > 200)
    j, i = [int(x[0]) for x in np.where(ok)]

    yl, yr = YL[j, i], YR[j, i]          # 200 draws each
    print(f'gene {genes[j]}, donor index {i}\n')
    print(f'Salmon gives {len(yl)} Gibbs draws. Each draw is a pair of '
          f'haplotype counts.')
    print(f'  first 6 draws, YL: {np.round(yl[:6], 1)}')
    print(f'  first 6 draws, YR: {np.round(yr[:6], 1)}')
    print(f'  posterior-mean counts:  YL_bar = {yl.mean():.1f}   '
          f'YR_bar = {yr.mean():.1f}\n')

    a = np.log(yl + KAPPA) - np.log(yr + KAPPA)
    print('Turn each draw into an allelic log ratio '
          'a_d = log(YL_d + 0.5) - log(YR_d + 0.5):')
    print(f'  first 6: {np.round(a[:6], 4)}')
    print(f'  mean over draws  = {a.mean():+.4f}   <- this is "a", the '
          f'value hapmixQTL regresses on genotype')
    print(f'  variance over draws = {a.var():.6f}   <- this is v\n')

    q = 1.0 / (yl.mean() + KAPPA) + 1.0 / (yr.mean() + KAPPA)
    print('Now predict that same variance from the counts alone, assuming '
          'Poisson.')
    print('  If y ~ Poisson(mu) then var(y) = mu, and by the delta method')
    print('  var(log y) ~= var(y)/mu^2 = 1/mu. For a difference of two logs,')
    print('  var(log y1 - log y2) ~= 1/mu1 + 1/mu2.')
    print(f'  q = 1/({yl.mean():.1f} + 0.5) + 1/({yr.mean():.1f} + 0.5) '
          f'= {q:.6f}\n')

    print('Both numbers are estimates of the SAME thing: the variance of this')
    print('donor\'s allelic log ratio. One measured, one predicted.')
    print(f'  v (measured from the draws) = {a.var():.6f}')
    print(f'  q (Poisson prediction)      = {q:.6f}')
    print(f'  v / q                       = {a.var() / q:.2f}\n')
    print(f'So for this donor the posterior spreads {a.var() / q:.1f}x wider '
          f'in variance,\n  {np.sqrt(a.var() / q):.1f}x in standard '
          f'deviation, than Poisson counting alone predicts.\n')

    print('Why the ratio is the quantity of interest: mixQTL models the')
    print('variance as sigma^2 * q with sigma^2 free. If the draws are right,')
    print('then sigma^2 = v/q. So v/q IS mixQTL\'s sigma^2, estimated per')
    print('donor-gene from the draws, never touching a regression residual.')


if __name__ == '__main__':
    main()

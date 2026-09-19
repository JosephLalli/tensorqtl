"""Two cheap pre-checks that shape the test designs. Deterministic; no nulls.

(a) Is the library-size offset redundant with hapmixQTL's covariates?
    hapmixQTL has no offset and relies on the 17 covariates to absorb depth.
    If log(lib_size) is already nearly spanned by those covariates, mechanism
    5 is weak a priori and should be demoted.

(b) Are the two combination rules actually different?
    hapmixQTL pools as (xy_a + xy_t)/(xx_a + xx_t). mixQTL does inverse
    variance on (beta, se). If hapmixQTL's SE is the known-variance form
    1/sqrt(xx), then w = 1/se^2 = xx, and inverse variance gives
    (xx_a*(xy_a/xx_a) + xx_t*(xy_t/xx_t))/(xx_a + xx_t), which is the pooled
    score exactly. Verified numerically here, because if it holds then
    mechanism 4 is not about the rule at all -- it is about mixQTL's SEs
    carrying a fitted residual scale, which reweights the channels.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
D = '/mnt/ssd/lalli/brainvar_hapmix_deploy/mixqtl_replication_20260919'

sys.path.insert(0, HERE)
sys.path.insert(0, REPO)


def main():
    from compare_mixqtl_replication import load_inputs

    out = {}
    I = load_inputs()
    C = I['cov_df'].values
    lib = I['lib_size']

    # (a) how much of log(lib_size) do the covariates already explain?
    y = np.log(lib)
    X = np.column_stack([np.ones(len(y)), C])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    r2 = 1 - resid.var() / y.var()
    print('(a) library size against the 17 covariates')
    print(f'    log(lib_size) spread: {lib.max() / lib.min():.2f}-fold across '
          f'{len(lib)} donors, sd(log) = {y.std():.4f}')
    print(f'    R^2 of log(lib_size) on the 17 covariates = {r2:.4f}')
    print(f'    residual sd after adjustment = {resid.std():.4f} '
          f'({np.exp(resid.std()):.3f}-fold)')
    print(f'    -> the offset can only matter through this residual part')
    out['lib_r2_on_covariates'] = float(r2)
    out['lib_fold_range'] = float(lib.max() / lib.min())
    out['lib_resid_sd_log'] = float(resid.std())

    # (b) are the two combination rules identical under known-variance SEs?
    rng = np.random.default_rng(42)
    n = 100000
    xy_a, xx_a = rng.normal(size=n), rng.gamma(5, 2, n)
    xy_t, xx_t = rng.normal(size=n), rng.gamma(5, 2, n)
    b_a, b_t = xy_a / xx_a, xy_t / xx_t
    se_a, se_t = 1 / np.sqrt(xx_a), 1 / np.sqrt(xx_t)      # known-variance form
    pooled = (xy_a + xy_t) / (xx_a + xx_t)                  # hapmixQTL
    w_a, w_t = 1 / se_a ** 2, 1 / se_t ** 2
    ivw = (w_a * b_a + w_t * b_t) / (w_a + w_t)             # mixQTL
    dev = np.abs(pooled - ivw).max()
    print('\n(b) combination rules on identical inputs, known-variance SEs')
    print(f'    max |pooled score - inverse variance| over {n:,} draws = {dev:.3e}')
    print(f'    identical: {dev < 1e-12}')
    out['combination_rules_identical_under_known_var'] = bool(dev < 1e-12)
    out['combination_max_abs_deviation'] = float(dev)

    # and with a fitted residual scale on one channel, they separate
    s = rng.gamma(4, 0.25, n)                               # fitted sigma
    se_a2, se_t2 = s / np.sqrt(xx_a), s / np.sqrt(xx_t)
    w_a2, w_t2 = 1 / se_a2 ** 2, 1 / se_t2 ** 2
    ivw2 = (w_a2 * b_a + w_t2 * b_t) / (w_a2 + w_t2)
    print(f'    with a COMMON fitted sigma on both channels, still identical: '
          f'{np.abs(pooled - ivw2).max() < 1e-12}')
    s_a = rng.gamma(4, 0.25, n)
    s_t = rng.gamma(4, 0.25, n)                             # per-channel sigma
    w_a3, w_t3 = xx_a / s_a ** 2, xx_t / s_t ** 2
    ivw3 = (w_a3 * b_a + w_t3 * b_t) / (w_a3 + w_t3)
    print(f'    with SEPARATE per-channel sigmas, they diverge: '
          f'corr = {np.corrcoef(pooled, ivw3)[0, 1]:.4f}')
    out['rules_diverge_only_with_per_channel_sigma'] = True
    out['corr_pooled_vs_ivw_per_channel_sigma'] = float(
        np.corrcoef(pooled, ivw3)[0, 1])

    json.dump(out, open(f'{D}/prechecks.json', 'w'), indent=1)


if __name__ == '__main__':
    main()

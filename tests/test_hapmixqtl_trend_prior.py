"""The smooth ('trend') empirical-Bayes prior of estimate_variance_priors:
curves of log c and log tau on log10 expression with a between-gene spread
about them, fitted by local marginal likelihood over all genes (a
normal-lognormal deconvolution of the raw estimates), limma's trend=TRUE
idea, in place of ten expression bins.

Pins: the curves recover a simulated smooth trend and its scatter; every gene
gets a finite prior (genes whose raw estimate is not positive included) and
the trend agrees with the decile prior when the truth is flat; the frame
threads through map_cis with its attrs; an unknown method raises.
"""
import numpy as np
import pandas as pd
import pytest
import torch

import tensorqtl.hapmixqtl as hapmixqtl


def _simulate(seed, G=600, N=60, slope_c=0.6, slope_tau=-0.5, sd_between=0.3, flat=False):
    rng = np.random.RandomState(seed)
    reads = 10 ** rng.uniform(1.0, 3.5, G)
    xl = np.log10(reads) - 2.0
    logc = (-1.0 if flat else -1.0 + slope_c * xl) + rng.normal(0, sd_between, G)
    logt = (-3.0 if flat else -3.0 + slope_tau * xl) + rng.normal(0, sd_between, G)
    v = 10 ** rng.uniform(-2.0, 0.0, (G, N))
    a = rng.normal(0, np.sqrt(np.exp(logc)[:, None] * v + np.exp(logt)[:, None]))
    samples = [f'S{i:03d}' for i in range(N)]
    A = pd.DataFrame(a, index=[f'g{i}' for i in range(G)], columns=samples)
    Va = pd.DataFrame(v, index=A.index, columns=samples)
    return A, Va, pd.Series(reads, index=A.index), logc, logt, xl


def test_trend_prior_recovers_a_smooth_trend_and_its_spread():
    A, Va, reads, logc, logt, xl = _simulate(41)
    pri = hapmixqtl.estimate_variance_priors(A, Va, expression=reads, prior_method='trend', min_informative=30)
    assert pri.attrs['method'] == 'trend' and 'curve' in pri.attrs and pri.attrs['span'] == 0.15
    mid = (xl > np.quantile(xl, 0.1)) & (xl < np.quantile(xl, 0.9))
    true_c = -1.0 + 0.6 * xl
    true_t = -3.0 - 0.5 * xl
    err_c = np.abs(pri['prior_logc_m'].values - true_c)[mid]
    err_t = np.abs(pri['prior_logtau_m'].values - true_t)[mid]
    # the curve tracks the truth over the middle 80% of expression; the fitted
    # value is a mean of a log estimate, so a small positive bias from the
    # positive-only selection is allowed for
    assert np.median(err_c) < 0.15 and err_c.max() < 0.4, (np.median(err_c), err_c.max())
    assert np.median(err_t) < 0.15 and err_t.max() < 0.4, (np.median(err_t), err_t.max())
    # the between-gene spread about the curve, 0.3 in truth, floored at 0.2
    s_c = pri['prior_logc_s'].values[mid]; s_t = pri['prior_logtau_s'].values[mid]
    assert 0.2 <= np.median(s_c) <= 0.45, np.median(s_c)
    assert 0.2 <= np.median(s_t) <= 0.45, np.median(s_t)
    # the curve's slope on the log scale is the simulated one to within a third
    lo, hi = np.quantile(xl, [0.2, 0.8])
    fit_slope = (np.interp(hi, xl[np.argsort(xl)], pri['prior_logc_m'].values[np.argsort(xl)]) -
                 np.interp(lo, xl[np.argsort(xl)], pri['prior_logc_m'].values[np.argsort(xl)])) / (hi - lo)
    assert abs(fit_slope - 0.6) < 0.2, fit_slope
    bins = pri.attrs['bins']
    for col in ('trend_logc_m', 'trend_logtau_m', 'c_raw_nonpositive', 'x_median', 'prior_c'):
        assert col in bins.columns, col


def test_every_gene_gets_a_finite_prior_and_a_flat_truth_matches_the_deciles():
    A, Va, reads, logc, logt, xl = _simulate(42, flat=True, sd_between=0.25)
    tr = hapmixqtl.estimate_variance_priors(A, Va, expression=reads, prior_method='trend', min_informative=30)
    de = hapmixqtl.estimate_variance_priors(A, Va, expression=reads, prior_method='deciles', min_informative=30)
    for col in ('prior_logc_m', 'prior_logc_s', 'prior_logtau_m', 'prior_logtau_s', 'prior_c', 'prior_tau'):
        assert np.isfinite(tr[col].values).all(), col
        assert (tr[col].values > 0).all() if col.startswith('prior_c') or col.startswith('prior_tau') else True
    # genes with a non-positive raw estimate still get the curve's value
    neg = tr['c_raw'].values <= 0
    if neg.any():
        assert np.isfinite(tr.loc[neg, 'prior_logc_m'].values).all()
    # flat truth: the two methods agree on the prior mean of log c where the deciles are well determined
    d = np.abs(tr['prior_logc_m'].values - de['prior_logc_m'].values)
    assert np.median(d) < 0.25, np.median(d)
    assert abs(np.median(tr['prior_logc_m'].values) - (-1.0)) < 0.2, np.median(tr['prior_logc_m'].values)


def test_trend_prior_threads_through_map_cis_and_unknown_method_raises():
    from tests.test_hapmixqtl import _make_dataset, _make_gaussian_seed
    d = _make_dataset(seed=126)
    rng = _make_gaussian_seed(4005)
    samples = list(d['A_df'].columns)
    extra = 120
    reads = 10 ** rng.uniform(1.5, 3.5, extra)
    v = 10 ** rng.uniform(-2, -0.5, (extra, len(samples)))
    a = rng.normal(0, np.sqrt(1.5 * v + 0.02))
    ix = [f'bg{i}' for i in range(extra)]
    A_big = pd.concat([d['A_df'], pd.DataFrame(a, index=ix, columns=samples)])
    Va_big = pd.concat([d['Va_df'], pd.DataFrame(v, index=ix, columns=samples)])
    expr = pd.Series(np.concatenate([np.full(len(d['A_df']), 300.0), reads]), index=A_big.index)
    with pytest.raises(ValueError):
        hapmixqtl.estimate_variance_priors(A_big, Va_big, expression=expr, prior_method='spline', min_informative=30)
    pri = hapmixqtl.estimate_variance_priors(A_big, Va_big, expression=expr, prior_method='trend', min_informative=30)
    res = hapmixqtl.map_cis(d['genotype_df'], d['variant_df'], d['A_df'], d['T_df'], d['Va_df'], d['Vt_df'], d['pos_df'],
                            xL_df=d['xL_df'], xR_df=d['xR_df'], nperm=200, window=1000000, seed=6, verbose=False,
                            variance_model='two_component', variance_prior=pri)
    assert res['variance_prior'].astype(bool).all()
    assert np.isfinite(res['c_a'].astype(float)).all() and (res['c_a'].astype(float) > 0).all()
    assert (res['tau_a'].astype(float) > 0).all()

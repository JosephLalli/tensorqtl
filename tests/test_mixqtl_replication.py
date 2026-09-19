"""Tests for the mixQTL replication arm.

Two kinds of test, because the R reference cannot be executed on this host
(libR.so double-links libblas.so.3 and libopenblas.so.0; lm(), %*% and
crossprod all segfault; tensorA and glmnet are absent; no sudo):

  ALGEBRA  -- the vectorized closed forms are checked against
              numpy.linalg.lstsq fitted one variant at a time. This catches
              errors in the tensor-contraction algebra, which is where a
              vectorized port actually goes wrong.

  SEMANTICS -- each gate, cap, degrees-of-freedom choice and fallback rule is
              pinned by a test that cites the R source line it encodes. These
              are transcription checks, not derivations.

What is NOT covered: agreement with a running mixQTL. No cross-language
execution check exists. Any claim of exact numerical reproduction of the
published implementation would be unsupported.
"""

import numpy as np
import pytest

from tensorqtl.mixqtl_replication import (
    harmonic_weights, apply_weight_cap, covariate_offset,
    trc_channel, asc_channel, meta_analyze, mixqtl_scan,
    mixqtl_permutation_scan, summaries_from_gibbs_posterior_mean,
    _simple_regression_through_origin, _simple_regression_with_intercept,
)


# ---------------------------------------------------------------------------
#  ALGEBRA: vectorized closed forms vs lstsq, per variant
# ---------------------------------------------------------------------------

def test_through_origin_matches_lstsq_per_variant():
    """Weighted through-origin fit == lstsq on the sqrt(w)-whitened design."""
    rng = np.random.default_rng(0)
    n, P = 60, 7
    X = rng.normal(size=(n, P))
    y = rng.normal(size=n)
    w = rng.gamma(3.0, 2.0, size=n)

    beta, se = _simple_regression_through_origin(y, X, w)
    sw = np.sqrt(w)
    for j in range(P):
        Xj = (X[:, j] * sw)[:, None]
        yj = y * sw
        b_ref, *_ = np.linalg.lstsq(Xj, yj, rcond=None)
        resid = yj - Xj @ b_ref
        sigma = np.sqrt(resid @ resid / (n - 1))          # dof n-1, one_dim
        se_ref = sigma / np.sqrt(float(Xj.T @ Xj))
        assert beta[j] == pytest.approx(float(b_ref[0]), rel=1e-10, abs=1e-12)
        assert se[j] == pytest.approx(float(se_ref), rel=1e-10, abs=1e-12)


def test_intercept_ols_matches_lstsq_per_variant():
    """Intercept OLS == lstsq on [x, 1], with two_dim's n-2 dof."""
    rng = np.random.default_rng(1)
    n, P = 55, 6
    X = rng.normal(size=(n, P))
    y = rng.normal(size=n)

    beta, se = _simple_regression_with_intercept(y, X)
    for j in range(P):
        D = np.column_stack([X[:, j], np.ones(n)])
        b_ref, *_ = np.linalg.lstsq(D, y, rcond=None)
        resid = y - D @ b_ref
        sigma2 = float(resid @ resid) / (n - 2)
        cov = sigma2 * np.linalg.inv(D.T @ D)
        assert beta[j] == pytest.approx(float(b_ref[0]), rel=1e-10, abs=1e-12)
        assert se[j] == pytest.approx(float(np.sqrt(cov[0, 0])), rel=1e-10, abs=1e-12)


def test_through_origin_recovers_a_planted_slope():
    """Sanity: with tiny noise the through-origin fit returns the truth."""
    rng = np.random.default_rng(2)
    n = 400
    x = rng.normal(size=(n, 1))
    y = 0.7 * x[:, 0] + rng.normal(scale=1e-6, size=n)
    beta, _ = _simple_regression_through_origin(y, x, np.ones(n))
    assert beta[0] == pytest.approx(0.7, abs=1e-6)


# ---------------------------------------------------------------------------
#  SEMANTICS: gates, weights, dof, fallbacks
# ---------------------------------------------------------------------------

def test_harmonic_weights_are_inverse_poisson_variance():
    """harmonic_sum_(x, y) = 1/(1/x + 1/y)   [rlib_util.R:27-29]"""
    assert harmonic_weights(np.array([10.0]), np.array([40.0]))[0] == pytest.approx(8.0)
    # equals the reciprocal of the delta-method variance of log(y1/y2)
    y1, y2 = np.array([25.0]), np.array([100.0])
    assert harmonic_weights(y1, y2)[0] == pytest.approx(1.0 / (1 / 25 + 1 / 100))


def test_weight_cap_is_a_fold_limit_on_the_minimum():
    """cap = min(weight_cap, floor(n/10)); cutoff = min(w)*cap  [matrix_ls.R:88-90]"""
    w = np.array([1.0, 2.0, 5.0, 100.0, 1000.0])
    capped, cap, cutoff = apply_weight_cap(w, sample_size=92, weight_cap=100.0)
    assert cap == 9.0                       # floor(92/10) binds, not 100
    assert cutoff == 9.0                    # min(w)=1 * 9
    assert capped.max() == 9.0
    assert capped[0] == 1.0                 # below the cutoff, untouched
    # at n=40 the cap tightens to 4-fold
    _, cap40, _ = apply_weight_cap(w, sample_size=40, weight_cap=100.0)
    assert cap40 == 4.0


def test_asc_gate_excludes_both_tails_on_both_haplotypes():
    """5 <= y1,y2 <= 5000, on BOTH haplotypes  [matrix_ls.R:80]"""
    rng = np.random.default_rng(3)
    n = 60
    y1 = rng.uniform(50, 200, n); y2 = rng.uniform(50, 200, n)
    y1[0] = 4.0        # below cutoff
    y2[1] = 4.9        # below cutoff, other haplotype
    y1[2] = 5001.0     # above cap
    X = rng.normal(size=(n, 3))
    out = asc_channel(y1, y2, X)
    assert out['sample_size'] == n - 3


def test_trc_gate_is_on_the_raw_count_not_the_response():
    """na_ind = is.na(trc) | trc_in < trc_cutoff  [matrix_ls.R:33]"""
    rng = np.random.default_rng(4)
    n = 50
    ytotal = rng.uniform(100, 500, n)
    ytotal[:4] = 19.0                      # below the raw-count cutoff
    lib = np.full(n, 1e6)
    X = rng.normal(size=(n, 3))
    out = trc_channel(ytotal, lib, X)
    assert out['sample_size'] == n - 4


def test_zero_count_donors_are_excluded_by_the_asc_gate():
    """Salmon emits YL=YR=0 for donors with no allele-informative reads.

    mixQTL's >=5 gate removes them without any extra sentinel, which is why
    this arm needs no equivalent of hapmixQTL's _zero_degenerate_ase_weights.
    """
    rng = np.random.default_rng(5)
    n = 40
    y1 = rng.uniform(40, 90, n); y2 = rng.uniform(40, 90, n)
    y1[:12] = 0.0; y2[:12] = 0.0
    X = rng.normal(size=(n, 2))
    out = asc_channel(y1, y2, X)
    assert out['sample_size'] == n - 12
    assert np.isfinite(out['beta']).all()


def test_monomorphic_variants_dropped_after_sample_filtering():
    """mono_ind is computed on the FILTERED x  [matrix_ls.R:35-36]

    A variant can be polymorphic overall yet constant among gated-in donors.
    """
    n = 40
    y1 = np.full(n, 60.0); y2 = np.full(n, 60.0)
    y1[:20] = 1.0                       # first 20 donors fail the gate
    x = np.zeros((n, 1))
    x[:20, 0] = 1.0                     # varies only among the FAILING donors
    out = asc_channel(y1, y2, x)
    assert out['mono'][0]               # constant post-filter -> dropped
    assert np.isnan(out['beta'][0])


def test_asc_uses_natural_log_with_no_pseudocount():
    """asc = log(asc1/asc2)  [matrix_ls.R:83] -- no kappa anywhere.

    Build a design where the response IS log(2) * x, so the through-origin
    slope must come back as exactly log(2) if and only if the transform is
    the natural log of the bare ratio. Any pseudocount shrinks it.
    """
    n = 40
    x = np.ones((n, 1)); x[::2] = -1.0
    y1 = np.where(x[:, 0] > 0, 100.0, 50.0)
    y2 = np.where(x[:, 0] > 0, 50.0, 100.0)     # ratio 2 at x=+1, 1/2 at x=-1
    out = asc_channel(y1, y2, x)
    assert out['beta'][0] == pytest.approx(np.log(2.0), rel=1e-12)

    # a kappa=0.5 pseudocount would give log(100.5/50.5) != log(2)
    shrunk = np.log(100.5 / 50.5)
    assert abs(out['beta'][0] - shrunk) > 1e-4


def test_trc_response_carries_the_library_size_offset():
    """trc = log(ytotal/2/lib_size) - cov  [matrix_ls.R:28]"""
    n = 40
    rng = np.random.default_rng(6)
    ytotal = np.full(n, 1000.0)
    lib = np.full(n, 1e6)
    x = rng.normal(size=(n, 1))
    out = trc_channel(ytotal, lib, x)
    # constant response -> zero slope, and the intercept absorbs the offset
    assert out['beta'][0] == pytest.approx(0.0, abs=1e-10)
    # doubling every library size shifts the response by -log(2), slope unchanged
    out2 = trc_channel(ytotal, lib * 2, x)
    assert out2['beta'][0] == pytest.approx(0.0, abs=1e-10)


def test_covariate_offset_selects_on_t_over_two_and_excludes_intercept():
    """two-step: fit all, keep |t|>2, refit, offset excludes the intercept
    [rlib_covariate.R:27-40]"""
    rng = np.random.default_rng(7)
    n = 200
    c_strong = rng.normal(size=n)
    c_null = rng.normal(size=n)
    lib = np.full(n, 1e6)
    logmu = -7.0 + 0.8 * c_strong + rng.normal(scale=0.02, size=n)
    ytotal = 2.0 * lib * np.exp(logmu)
    off, sel = covariate_offset(ytotal, lib, np.column_stack([c_strong, c_null]))
    assert sel[0] and not sel[1]
    # the offset is exactly (fitted coefficient) * c_strong -- no intercept term
    coef = off[0] / c_strong[0]
    assert np.allclose(off, coef * c_strong, rtol=1e-12, atol=1e-12)
    # and that coefficient recovers the planted 0.8 up to the simulated noise
    assert coef == pytest.approx(0.8, rel=1e-2)


def test_meta_is_inverse_variance_and_falls_back_when_a_channel_is_small():
    """both n>=15 -> IVW; else the larger-n channel  [rlib_meta.R:60-103]"""
    trc = dict(beta=np.array([1.0]), se=np.array([1.0]), sample_size=80)
    asc = dict(beta=np.array([3.0]), se=np.array([1.0]), sample_size=40)
    out = meta_analyze(trc, asc)
    assert out['meta']['beta'][0] == pytest.approx(2.0)          # equal weights
    assert out['meta']['se'][0] == pytest.approx(np.sqrt(0.5))
    assert out['meta']['method'][0] == 'meta'

    asc_small = dict(beta=np.array([3.0]), se=np.array([1.0]), sample_size=10)
    out2 = meta_analyze(trc, asc_small)
    assert out2['meta']['beta'][0] == pytest.approx(1.0)         # trc only
    assert out2['meta']['method'][0] == 'trc'


def test_meta_fills_nan_from_the_other_channel():
    """na_1_notna_2 fill  [rlib_meta.R:68-73]"""
    trc = dict(beta=np.array([np.nan]), se=np.array([np.nan]), sample_size=80)
    asc = dict(beta=np.array([2.5]), se=np.array([0.5]), sample_size=40)
    out = meta_analyze(trc, asc)
    assert out['meta']['beta'][0] == pytest.approx(2.5)
    assert out['meta']['method'][0] == 'asc'


def test_na_genotypes_impute_to_half():
    """h1[is.na(h1)] = 0.5  [mixqtl.R:50-53]"""
    rng = np.random.default_rng(8)
    n = 40
    y1 = rng.uniform(40, 90, n); y2 = rng.uniform(40, 90, n)
    yt = y1 + y2; lib = np.full(n, 1e6)
    h1 = rng.integers(0, 2, (n, 2)).astype(float)
    h2 = rng.integers(0, 2, (n, 2)).astype(float)
    h1[0, 0] = np.nan
    out = mixqtl_scan(y1, y2, yt, lib, h1, h2)
    assert np.isfinite(out['meta']['beta']).any()


def test_posterior_mean_adapter_discards_the_draw_variance():
    """This arm's only use of the draws is their mean."""
    rng = np.random.default_rng(9)
    yL = rng.gamma(20, 2, (3, 5, 200))
    yR = rng.gamma(20, 2, (3, 5, 200))
    yT = yL + yR
    a, b, t = summaries_from_gibbs_posterior_mean(yL, yR, yT)
    assert np.allclose(a, yL.mean(2))
    assert np.allclose(b, yR.mean(2))
    assert np.allclose(t, yT.mean(2))


# ---------------------------------------------------------------------------
#  the reference permutation defect
# ---------------------------------------------------------------------------

def test_strict_reference_cap_reproduces_the_all_weights_zeroed_defect():
    """weights[!passed]=0 before min() zeroes every weight.
    [rlib_matrix_ls_with_mask.R:93-97]

    Demonstrates the defect rather than hiding it: with the reference rule and
    at least one gate-failing sample, every permuted statistic is undefined.
    """
    # n = 40 passing, so the cap itself is a healthy 4-fold; the only thing
    # that breaks the strict path is the zero from the one failing sample.
    rng = np.random.default_rng(20)
    w = np.concatenate([[0.0], rng.uniform(10.0, 300.0, 40)])
    passed = np.concatenate([[False], np.ones(40, bool)])

    capped, cap, cutoff = apply_weight_cap(w, 40, 100.0, passed=passed,
                                           strict_reference_cap=True)
    assert cap == 4.0
    assert cutoff == 0.0                        # min() saw the zero
    assert capped.sum() == 0.0                  # everything zeroed

    fixed, cap2, cutoff_fixed = apply_weight_cap(w, 40, 100.0, passed=passed,
                                                 strict_reference_cap=False)
    assert cap2 == 4.0
    assert cutoff_fixed == pytest.approx(w[passed].min() * 4.0)
    assert fixed[passed].sum() > 0.0


def test_cap_is_zero_below_ten_passing_samples():
    """floor(n/10) = 0 for n <= 9, so every weight is zeroed even with no
    gate failures. The reference guard is only `sample_size > 2`.
    [rlib_matrix_ls.R:88-90]"""
    w = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    capped, cap, cutoff = apply_weight_cap(w, sample_size=9, weight_cap=100.0)
    assert cap == 0.0 and cutoff == 0.0
    assert capped.sum() == 0.0


def test_cap_of_one_makes_the_channel_unweighted():
    """For 10 <= n < 20 the cap is 1, forcing every weight to min(w).
    The harmonic weights then have no effect: the fit is plain OLS."""
    w = np.array([10.0, 20.0, 30.0, 40.0])
    capped, cap, cutoff = apply_weight_cap(w, sample_size=15, weight_cap=100.0)
    assert cap == 1.0
    assert np.allclose(capped, 10.0)            # all equal -> unweighted

    rng = np.random.default_rng(21)
    n, P = 15, 3
    X = rng.normal(size=(n, P))
    y = rng.normal(size=n)
    w_uniform = np.full(n, 10.0)
    b_capped, _ = _simple_regression_through_origin(y, X, w_uniform)
    b_ols, _ = _simple_regression_through_origin(y, X, np.ones(n))
    assert np.allclose(b_capped, b_ols, rtol=1e-12)


def test_permutation_scan_is_usable_by_default_and_degenerate_under_strict():
    rng = np.random.default_rng(10)
    n, P = 60, 4
    y1 = rng.uniform(40, 200, n); y2 = rng.uniform(40, 200, n)
    y1[:6] = 0.0; y2[:6] = 0.0                  # realistic gate failures
    yt = y1 + y2 + 500.0
    lib = np.full(n, 1e6)
    h1 = rng.integers(0, 2, (n, P)).astype(float)
    h2 = rng.integers(0, 2, (n, P)).astype(float)
    perm = np.array([rng.permutation(n) for _ in range(5)])

    ok = mixqtl_permutation_scan(y1, y2, yt, lib, h1, h2, perm)
    assert np.isfinite(ok).all()

    bad = mixqtl_permutation_scan(y1, y2, yt, lib, h1, h2, perm,
                                  strict_reference_cap=True)
    # the defect kills the ALLELIC channel; meta then silently falls back to
    # the total channel, so the statistic stays finite but is trc-only.
    no_asc = mixqtl_permutation_scan(np.zeros(n), np.zeros(n), yt, lib,
                                     h1, h2, perm)
    assert np.allclose(bad, no_asc, rtol=1e-12, equal_nan=True)
    assert not np.allclose(bad, ok)


def test_permutation_moves_response_weights_and_mask_together():
    """The parent permutes the phenotype bundle, not residuals.

    Under the identity permutation the permuted statistic must equal the
    observed one; that only holds if y, w and the mask are indexed alike.
    """
    rng = np.random.default_rng(11)
    n, P = 50, 3
    y1 = rng.uniform(40, 200, n); y2 = rng.uniform(40, 200, n)
    yt = y1 + y2 + 400.0
    lib = np.full(n, 1e6)
    h1 = rng.integers(0, 2, (n, P)).astype(float)
    h2 = rng.integers(0, 2, (n, P)).astype(float)

    obs = mixqtl_scan(y1, y2, yt, lib, h1, h2)
    identity = np.arange(n)[None, :]
    perm = mixqtl_permutation_scan(y1, y2, yt, lib, h1, h2, identity)
    assert perm[0] == pytest.approx(np.nanmax(np.abs(obs['meta']['stat'])),
                                    rel=1e-9)

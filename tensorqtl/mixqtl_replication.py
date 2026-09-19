"""mixQTL replication arm: hapmixQTL's estimator with every divergence removed.

WHAT THIS IS
------------
A faithful NumPy port of the published mixQTL estimator (Liang et al., Nat
Commun 2021; ``hakyimlab/mixqtl`` @ 624ae4421e43fe152f3851f795b6326901dfeb93).
It exists so the hapmixQTL estimator can be compared against its parent on
identical inputs, rather than against a description of its parent.

It is also, by construction, **the no-draws comparator**. mixQTL consumes
observed counts; this arm feeds it the Salmon posterior-mean counts
``mean_draws(YL), mean_draws(YR), mean_draws(YT)`` and then never looks at the
draws again. Where hapmixQTL weights donor i by ``1/(c_g v_gi + tau_g)`` with
``v_gi`` the across-draw variance of the log ratio, this arm weights by the
harmonic sum ``1/(1/y1 + 1/y2)`` -- the Poisson-predicted precision of
``log(y1/y2)``. So the pair (hapmixQTL, mixqtl_replication) isolates exactly
one thing: whether propagating the Gibbs draws buys anything over the Poisson
approximation that the counts alone imply.

Nothing here should be read as an endorsement of either weighting. Per the
standing instruction recorded in CLAUDE.md, the allelic error model is
undetermined and no parameterization is incumbent.

THE DIVERGENCES THIS ARM REMOVES
--------------------------------
Relative to ``hapmixqtl.py``, and keyed to the divergence table in
``mixqtl_algorithm_review_20260914/REPORT.md`` section 4:

  1. Gibbs variance propagation      -> posterior-mean counts only
  2. weights 1/(c*v + tau)           -> harmonic sum of counts, capped
  3. per-gene null-model tau         -> no tau; dispersion is the residual scale
  4. known-variance SE 1/sqrt(xx)    -> fitted sigma_hat from the residuals
  5. pseudocount kappa=0.5, and the
     mean over draws of the log    -> no pseudocount, and the log of the
                                        mean (mixQTL gates instead). NOTE
                                        both arms are in NATURAL log:
                                        hapmixqtl.py:406 uses np.log and
                                        log2 appears nowhere in it, the
                                        migration being pending. An earlier
                                        version of this list said log2 and
                                        was wrong; the betas are directly
                                        comparable.
  6. no library-size offset          -> log(YT / 2 / lib_size)
  7. joint WLS covariate adjustment  -> two-step selected-covariate offset
  8. combined Z^2 statistic          -> inverse-variance meta of (beta, se)
  9. common t reference              -> normal reference (n > 15)
 10. no count gates                  -> trc >= 100; 50 <= asc <= 1000, the
                                        PUBLISHED gates. These are NOT the R
                                        signature's defaults; see the Gate
                                        presets block below.
 11. permuted whitened residuals     -> permuted phenotype bundle (y, w, mask)

Items already aligned before this module (do not re-remove): the allelic
channel is fitted through the origin in both, and both are in natural log
units (the log2 migration in hapmixQTL is still pending, so no unit
conversion happens anywhere in this comparison).

VALIDATION BOUNDARY
-------------------
The R reference CANNOT be executed on this host: ``libR.so`` links both
``libblas.so.3`` and ``libopenblas.so.0``, and every linear-algebra entry
point tested (``lm``, ``%*%``, ``crossprod``) segfaults; ``tensorA`` and
``glmnet`` are not installed and sudo is unavailable. So there is no
cross-language execution check behind this port. Its algebra is validated
against ``numpy.linalg.lstsq`` per variant, and each filtering/weighting rule
is pinned by a unit test that cites the R source line it encodes. See
``tests/test_mixqtl_replication.py``.

A DEFECT IN THE REFERENCE PERMUTATION PATH
------------------------------------------
``matrix_ls_asc_permutation`` (rlib_matrix_ls_with_mask.R:87-117) sets the
weights of gate-failing samples to zero *before* computing the cap:

    weights[!passed_ind] = 0
    weight_cap    = min(weight_cap, floor(sample_size / 10))
    weight_cutoff = min(weights) * weight_cap        # <- min() is now 0
    weights[weights > weight_cutoff] = weight_cutoff # <- zeroes everything

Whenever at least one sample fails the ASE gate, ``min(weights)`` is 0, so the
cutoff is 0 and every surviving weight is set to 0. ``XtX`` is then 0 and every
permuted beta is 0/0. The non-permutation ``matrix_ls_asc`` escapes this only
because it subsets the vectors before taking the min.

This is reproduced verbatim under ``strict_reference_cap=True`` so the defect
can be demonstrated. The default is ``False``, which takes the min over
gate-passing samples only -- the behaviour the non-permutation path already
has, and the minimal change that makes a gene-level p-value obtainable.
"""

import numpy as np

__all__ = [
    'harmonic_weights', 'apply_weight_cap', 'covariate_offset',
    'trc_channel', 'asc_channel', 'meta_analyze', 'mixqtl_scan',
    'mixqtl_permutation_scan', 'summaries_from_gibbs_posterior_mean',
    'PUBLISHED_GATES', 'PACKAGE_DEFAULT_GATES',
]

# ---------------------------------------------------------------------------
#  Gate presets
# ---------------------------------------------------------------------------
# mixQTL's R function signature and the settings its authors actually
# published with are different, and the signature is the outlier: both the
# roxygen examples and the GTEx v8 production driver
# (mixqtl-pipeline/code/gtex_v8_mixqtl.R:77) use the stricter values. This
# module defaults to the PUBLISHED settings, because the point of the arm is
# to reproduce what mixQTL did rather than an unused default.
#
#                    trc_cutoff  asc_cutoff  weight_cap  asc_cap
#   R signature              20           5         100     5000
#   roxygen examples        100          50         100     1000
#   GTEx v8 driver          100          50          10     1000   <- default
#
# No rationale for any of these appears in the source, the update notes, the
# supplement or the driver; the supplement states the lower thresholds as
# gene-inclusion criteria and never mentions an upper bound.
#
# CONSEQUENCE, measured on the 29 high-coverage calibration genes: the
# published gates keep 499 of 2,193 informative donor-gene pairs against
# 1,558 under the R signature, with 1,656 lost to the upper cap. Median
# allelic donors per gene falls from 60 to 7, and 17 of 29 genes drop below
# META_N_CUTOFF so the allelic channel stops contributing at all. On
# Salmon posterior-mean abundances the upper cap behaves as a coverage
# ceiling rather than the outlier guard it presumably was for observed read
# counts. Pass PACKAGE_DEFAULT_GATES explicitly to get the other behaviour.

PUBLISHED_GATES = dict(trc_cutoff=100.0, asc_cutoff=50.0,
                       weight_cap=10.0, asc_cap=1000.0)
PACKAGE_DEFAULT_GATES = dict(trc_cutoff=20.0, asc_cutoff=5.0,
                             weight_cap=100.0, asc_cap=5000.0)

TRC_CUTOFF = PUBLISHED_GATES['trc_cutoff']
ASC_CUTOFF = PUBLISHED_GATES['asc_cutoff']
ASC_CAP = PUBLISHED_GATES['asc_cap']
WEIGHT_CAP = PUBLISHED_GATES['weight_cap']
META_N_CUTOFF = 15


# ---------------------------------------------------------------------------
#  input adapter
# ---------------------------------------------------------------------------

def summaries_from_gibbs_posterior_mean(yL, yR, yT=None):
    """Collapse Gibbs draws to the posterior-mean counts mixQTL would consume.

    This is the whole of this arm's use of the draws: their mean. The
    across-draw variance is deliberately discarded -- that discarding is the
    comparison this module exists to support.

    Args:
        yL, yR: haplotype counts [features, samples, draws]
        yT:     gene total counts [features, samples, draws]; defaults to
                yL + yR, matching hapmixqtl.compute_summaries_from_gibbs.

    Returns (y1, y2, ytotal), each [features, samples].
    """
    tot = (yL + yR) if yT is None else np.asarray(yT)
    return yL.mean(axis=2), yR.mean(axis=2), np.asarray(tot).mean(axis=2)


# ---------------------------------------------------------------------------
#  weights
# ---------------------------------------------------------------------------

def harmonic_weights(y1, y2):
    """mixQTL's ``harmonic_sum_``: 1 / (1/y1 + 1/y2)   [rlib_util.R:27-29]

    This is the reciprocal of the delta-method Poisson variance of
    log(y1/y2), whose leading term is 1/y1 + 1/y2. It is the Poisson
    approximation that the Gibbs draw variance would replace.
    """
    y1 = np.asarray(y1, dtype=float)
    y2 = np.asarray(y2, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        return 1.0 / (1.0 / y1 + 1.0 / y2)


def apply_weight_cap(weights, sample_size, weight_cap=WEIGHT_CAP, passed=None,
                     strict_reference_cap=False):
    """mixQTL's weight cap  [rlib_matrix_ls.R:88-90].

        cap    = min(weight_cap, floor(sample_size / 10))
        cutoff = min(weights) * cap
        weights[weights > cutoff] = cutoff

    Note the cap is a *fold* limit relative to the smallest weight, and it
    tightens as the sample shrinks: at n = 92 it is 9-fold, at n = 40 it is
    4-fold. That is the mechanism which keeps mixQTL's allelic weights from
    the unbounded-weight failure that 1/v weighting shows on this cohort.

    Two consequences of ``floor(n / 10)`` worth stating explicitly, because
    both are silent:

      * 3 <= n <= 9  ->  cap = 0, so the cutoff is 0 and EVERY weight is
        zeroed. The reference's own guard is ``sample_size > 2``, so such
        genes pass the guard and then return NaN from 0/0.
      * 10 <= n < 20 ->  cap = 1, so the cutoff is min(w) and every weight is
        forced down to the minimum. The channel is then exactly unweighted
        OLS; the harmonic weights have no effect at all.

    So mixQTL's allelic weighting phases in with sample size rather than
    applying uniformly. At this cohort's n = 92 the cap is 9-fold.

    Args:
        passed: boolean mask of gate-passing samples. When given, the min is
            taken over passing samples only. Required whenever `weights`
            contains zeroed-out failing samples.
        strict_reference_cap: reproduce the reference permutation path's
            defect verbatim (min over ALL entries, including the zeros). See
            the module docstring.
    """
    w = np.array(weights, dtype=float, copy=True)
    cap = min(float(weight_cap), float(np.floor(sample_size / 10)))
    if strict_reference_cap or passed is None:
        pool = w
    else:
        pool = w[passed]
    if pool.size == 0:
        return w, cap, 0.0
    cutoff = float(np.min(pool)) * cap
    w[w > cutoff] = cutoff
    return w, cap, cutoff


# ---------------------------------------------------------------------------
#  covariates
# ---------------------------------------------------------------------------

def covariate_offset(ytotal, lib_size, covariates, t_threshold=2.0):
    """mixQTL's two-step covariate offset  [rlib_covariate.R:27-40].

    1. regress log(ytotal / lib_size / 2) on ALL covariates with an intercept
    2. keep those with |t| > 2
    3. refit on the kept set; the offset is their fitted contribution,
       EXCLUDING the intercept (the R code drops row 1 via ``out[-1, 1]``)

    This is mixQTL's response pre-adjustment. It differs from hapmixQTL's
    joint WLS adjustment of response and genotype together; that difference is
    divergence 7 and is removed here by using mixQTL's procedure.

    Args:
        covariates: [samples, n_cov]
    Returns:
        offset [samples], plus the boolean mask of selected covariates.
    """
    ytotal = np.asarray(ytotal, dtype=float)
    lib_size = np.asarray(lib_size, dtype=float)
    C = np.asarray(covariates, dtype=float)
    n = ytotal.shape[0]
    with np.errstate(divide='ignore', invalid='ignore'):
        lhs = np.log(ytotal / lib_size / 2.0)
    ok = np.isfinite(lhs)
    if ok.sum() <= C.shape[1] + 1:
        return np.zeros(n), np.zeros(C.shape[1], dtype=bool)

    def _fit(design, y):
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        resid = y - design @ beta
        dof = design.shape[0] - design.shape[1]
        if dof <= 0:
            return beta, np.full(beta.shape, np.inf)
        s2 = float(resid @ resid) / dof
        xtx_inv = np.linalg.pinv(design.T @ design)
        se = np.sqrt(np.maximum(s2 * np.diag(xtx_inv), 0.0))
        return beta, se

    X_all = np.column_stack([np.ones(ok.sum()), C[ok]])
    beta, se = _fit(X_all, lhs[ok])
    with np.errstate(divide='ignore', invalid='ignore'):
        tvals = np.where(se > 0, beta / se, 0.0)
    selected = np.abs(tvals[1:]) > t_threshold          # drop the intercept
    if not selected.any():
        return np.zeros(n), selected

    X_sel = np.column_stack([np.ones(ok.sum()), C[ok][:, selected]])
    beta_sel, _ = _fit(X_sel, lhs[ok])
    # offset = selected covariates' contribution only; intercept excluded,
    # mirroring `t(covariates[selected, -1]) %*% out[-1, 1]`.
    return C[:, selected] @ beta_sel[1:], selected


# ---------------------------------------------------------------------------
#  channel estimators
# ---------------------------------------------------------------------------

def _simple_regression_through_origin(y, X, w):
    """Per-column weighted through-origin fit; vectorized matrixQTL_one_dim.

    [rlib_matrix_ls.R:129-160].  sigma_hat uses (n - 1) degrees of freedom.
    Returns (beta [P], se [P]).
    """
    sw = np.sqrt(w)
    yw = y * sw
    Xw = X * sw[:, None]
    XtX = (Xw * Xw).sum(0)
    XtY = Xw.T @ yw
    with np.errstate(divide='ignore', invalid='ignore'):
        beta = XtY / XtX
    YtY = float(yw @ yw)
    rss = YtY - 2.0 * beta * XtY + beta ** 2 * XtX
    n = int(np.sum(w > 0))
    dof = max(n - 1, 1)
    sigma = np.sqrt(np.maximum(rss, 0.0) / dof)
    with np.errstate(divide='ignore', invalid='ignore'):
        se = sigma / np.sqrt(XtX)
    bad = ~np.isfinite(XtX) | (XtX <= 0)
    beta = np.where(bad, np.nan, beta)
    se = np.where(bad, np.nan, se)
    return beta, se


def _simple_regression_with_intercept(y, X):
    """Per-column OLS with an intercept; vectorized matrixQTL_two_dim.

    [rlib_matrix_ls.R:186-240].  sigma_hat uses (n - 2) degrees of freedom and
    Delta is taken in absolute value, exactly as the reference does.
    Returns (beta_x [P], se_x [P]).
    """
    n = y.shape[0]
    ones = np.ones(n)
    S11 = (X * X).sum(0)
    S12 = X.sum(0)                       # X' 1
    S22 = float(ones @ ones)
    T1 = X.T @ y
    T2 = float(ones @ y)
    Delta = np.abs(S11 * S22 - S12 * S12)
    with np.errstate(divide='ignore', invalid='ignore'):
        b1 = (S22 * T1 - S12 * T2) / Delta
        b2 = (S11 * T2 - S12 * T1) / Delta
    YtY = float(y @ y)
    rss = (YtY - 2.0 * b1 * T1 - 2.0 * b2 * T2
           + 2.0 * b1 * b2 * S12 + b1 ** 2 * S11 + b2 ** 2 * S22)
    dof = max(n - 2, 1)
    sigma = np.sqrt(np.maximum(rss, 0.0) / dof)
    with np.errstate(divide='ignore', invalid='ignore'):
        se = sigma * np.sqrt(S22 / Delta)
    bad = ~np.isfinite(Delta) | (Delta <= 0)
    return np.where(bad, np.nan, b1), np.where(bad, np.nan, se)


def trc_channel(ytotal, lib_size, X, cov_offset=None, trc_cutoff=TRC_CUTOFF):
    """mixQTL's trcQTL  [rlib_matrix_ls.R:26-46].

    response  log(ytotal / 2 / lib_size) - cov_offset      (natural log)
    gate      ytotal >= trc_cutoff, and finite response
    design    x = (h1 + h2)/2 plus an intercept, UNWEIGHTED OLS
    Variants with zero genotype variance AFTER sample filtering are dropped.

    Args:
        X: [samples, P] already formed as (h1 + h2) / 2
    Returns dict with beta, se, sample_size, mono (dropped-variant mask).
    """
    ytotal = np.asarray(ytotal, dtype=float)
    lib_size = np.asarray(lib_size, dtype=float)
    X = np.asarray(X, dtype=float)
    P = X.shape[1]
    off = np.zeros_like(ytotal) if cov_offset is None else np.asarray(cov_offset, float)
    with np.errstate(divide='ignore', invalid='ignore'):
        resp = np.log(ytotal / 2.0 / lib_size) - off
    keep = np.isfinite(resp) & (ytotal >= trc_cutoff)

    beta = np.full(P, np.nan)
    se = np.full(P, np.nan)
    Xk, yk = X[keep], resp[keep]
    n = int(keep.sum())
    if n <= 2 or Xk.shape[0] == 0:
        return dict(beta=beta, se=se, sample_size=n, mono=np.ones(P, bool))
    mono = Xk.var(axis=0) == 0
    if (~mono).any():
        b, s = _simple_regression_with_intercept(yk, Xk[:, ~mono])
        beta[~mono] = b
        se[~mono] = s
    return dict(beta=beta, se=se, sample_size=n, mono=mono)


def asc_channel(y1, y2, X, asc_cutoff=ASC_CUTOFF, weight_cap=WEIGHT_CAP,
                asc_cap=ASC_CAP):
    """mixQTL's ascQTL  [rlib_matrix_ls.R:77-104].

    gate      asc_cutoff <= y1, y2 <= asc_cap   (BOTH haplotypes)
    response  log(y1 / y2)                      (natural log, NO pseudocount)
    weights   harmonic sum, fold-capped
    design    x = h1 - h2, fitted THROUGH THE ORIGIN, no covariates
    SE        fitted residual sigma / sqrt(XtX), dof = n - 1

    Args:
        X: [samples, P] already formed as h1 - h2
    Returns dict with beta, se, sample_size, mono, weights, cap, cutoff.
    """
    y1 = np.asarray(y1, dtype=float)
    y2 = np.asarray(y2, dtype=float)
    X = np.asarray(X, dtype=float)
    P = X.shape[1]
    passed = (y1 >= asc_cutoff) & (y2 >= asc_cutoff) & (y1 <= asc_cap) & (y2 <= asc_cap)

    beta = np.full(P, np.nan)
    se = np.full(P, np.nan)
    n = int(passed.sum())
    if n <= 2:
        return dict(beta=beta, se=se, sample_size=n, mono=np.ones(P, bool),
                    weights=np.zeros(n), cap=0.0, cutoff=0.0)

    Xk = X[passed]
    with np.errstate(divide='ignore', invalid='ignore'):
        resp = np.log(y1[passed] / y2[passed])
    w = harmonic_weights(y1[passed], y2[passed])
    w, cap, cutoff = apply_weight_cap(w, n, weight_cap)

    mono = Xk.var(axis=0) == 0
    if (~mono).any():
        b, s = _simple_regression_through_origin(resp, Xk[:, ~mono], w)
        beta[~mono] = b
        se[~mono] = s
    return dict(beta=beta, se=se, sample_size=n, mono=mono,
                weights=w, cap=cap, cutoff=cutoff)


# ---------------------------------------------------------------------------
#  meta-analysis
# ---------------------------------------------------------------------------

def _z2p(z):
    """2 * upper normal tail; mixQTL's ``z2p_``  [rlib_meta.R:137-139]."""
    from scipy.stats import norm
    return 2.0 * norm.sf(np.abs(z))


def _t2p(t, dof):
    """mixQTL's ``t2p_``  [rlib_meta.R:141-143]."""
    from scipy.stats import t as tdist
    return 2.0 * tdist.sf(np.abs(t), df=dof)


def _pval(b, s, n, n_cutoff=META_N_CUTOFF):
    """mixQTL's ``get_pval_fast_``  [rlib_meta.R:119-128]: z if n > 15 else t."""
    with np.errstate(divide='ignore', invalid='ignore'):
        stat = b / s
    if n > n_cutoff:
        return _z2p(stat), stat, 'z-val'
    return _t2p(stat, n), stat, 't-val'


def meta_analyze(trc, asc, n_cutoff=META_N_CUTOFF):
    """mixQTL's ``meta_analyze`` / ``my_meta_fast_``  [rlib_meta.R:34-110].

    If both channels have at least n_cutoff samples, combine by inverse
    variance:  w = 1/se^2,  b = sum(w b)/sum(w),  se = sqrt(1/sum(w)),
    and take the p-value against n_trc + n_asc. Otherwise fall back to the
    channel with the larger sample size, filling that channel's NAs from the
    other. Note this is a scalar meta-analysis of two summary statistics, not
    hapmixQTL's pooled Z^2 = (xy_a + xy_t)^2 / (xx_a + xx_t).
    """
    b1, s1, n1 = trc['beta'], trc['se'], trc['sample_size']
    b2, s2, n2 = asc['beta'], asc['se'], asc['sample_size']
    p1, stat1, _ = _pval(b1, s1, n1, n_cutoff)
    p2, stat2, _ = _pval(b2, s2, n2, n_cutoff)

    if n1 >= n2:
        bm, sm, pm, sm_stat = b1.copy(), s1.copy(), p1.copy(), stat1.copy()
        method = np.full(b1.shape, 'trc', dtype=object)
        fill = np.isnan(b1) & ~np.isnan(b2)
        bm[fill], sm[fill], pm[fill], sm_stat[fill] = b2[fill], s2[fill], p2[fill], stat2[fill]
        method[fill] = 'asc'
    else:
        bm, sm, pm, sm_stat = b2.copy(), s2.copy(), p2.copy(), stat2.copy()
        method = np.full(b2.shape, 'asc', dtype=object)
        fill = np.isnan(b2) & ~np.isnan(b1)
        bm[fill], sm[fill], pm[fill], sm_stat[fill] = b1[fill], s1[fill], p1[fill], stat1[fill]
        method[fill] = 'trc'

    if n1 >= n_cutoff and n2 >= n_cutoff:
        with np.errstate(divide='ignore', invalid='ignore'):
            w1, w2 = 1.0 / s1 ** 2, 1.0 / s2 ** 2
            b0 = (w1 * b1 + w2 * b2) / (w1 + w2)
            s0 = np.sqrt(1.0 / (w1 + w2))
        p0, stat0, _ = _pval(b0, s0, n1 + n2, n_cutoff)
        ok = ~np.isnan(p0)
        bm[ok], sm[ok], pm[ok], sm_stat[ok] = b0[ok], s0[ok], p0[ok], stat0[ok]
        method[ok] = 'meta'

    return dict(trc=dict(beta=b1, se=s1, pval=p1, stat=stat1, sample_size=n1),
                asc=dict(beta=b2, se=s2, pval=p2, stat=stat2, sample_size=n2),
                meta=dict(beta=bm, se=sm, pval=pm, stat=sm_stat, method=method))


# ---------------------------------------------------------------------------
#  scan
# ---------------------------------------------------------------------------

def mixqtl_scan(y1, y2, ytotal, lib_size, h1, h2, covariates=None,
                trc_cutoff=TRC_CUTOFF, asc_cutoff=ASC_CUTOFF,
                asc_cap=ASC_CAP, weight_cap=WEIGHT_CAP):
    """One gene's nominal pass; mixQTL's ``mixqtl``  [mixqtl.R:44-68].

    NA genotypes are imputed to 0.5, as the reference does, before forming
    Xasc = h1 - h2 and Xtrc = (h1 + h2)/2.

    Args:
        y1, y2, ytotal: [samples] posterior-mean counts
        lib_size:       [samples]
        h1, h2:         [samples, P] phased haplotype dosages in {0, 1}
        covariates:     [samples, n_cov] or None
    """
    h1 = np.array(h1, dtype=float, copy=True)
    h2 = np.array(h2, dtype=float, copy=True)
    h1[np.isnan(h1)] = 0.5
    h2[np.isnan(h2)] = 0.5
    Xasc = h1 - h2
    Xtrc = (h1 + h2) / 2.0

    if covariates is None:
        off = np.zeros(len(ytotal))
        selected = None
    else:
        off, selected = covariate_offset(ytotal, lib_size, covariates)

    trc = trc_channel(ytotal, lib_size, Xtrc, off, trc_cutoff)
    asc = asc_channel(y1, y2, Xasc, asc_cutoff, weight_cap, asc_cap)
    out = meta_analyze(trc, asc)
    out['cov_selected'] = selected
    return out


def mixqtl_permutation_scan(y1, y2, ytotal, lib_size, h1, h2, perm_idx,
                            covariates=None, trc_cutoff=TRC_CUTOFF,
                            asc_cutoff=ASC_CUTOFF, asc_cap=ASC_CAP,
                            weight_cap=WEIGHT_CAP, strict_reference_cap=False):
    """Permutation pass; mixQTL's ``*_permutation``  [rlib_matrix_ls_with_mask.R].

    The reference permutes the *phenotype bundle*: response, weights and mask
    all move together under the same index, while the genotype design stays
    put. That is the same conclusion hapmixQTL reached independently on
    2026-09-17 (``perm_scheme='records'``), and it is divergence 11.

    Args:
        perm_idx: [n_perm, samples] integer permutations of range(samples)
        strict_reference_cap: reproduce the reference's all-weights-zeroed
            defect. See the module docstring. Default False.
    Returns [n_perm] array of the maximum meta |stat| over variants.
    """
    h1 = np.array(h1, dtype=float, copy=True)
    h2 = np.array(h2, dtype=float, copy=True)
    h1[np.isnan(h1)] = 0.5
    h2[np.isnan(h2)] = 0.5
    Xasc, Xtrc = h1 - h2, (h1 + h2) / 2.0

    y1 = np.asarray(y1, float); y2 = np.asarray(y2, float)
    ytotal = np.asarray(ytotal, float); lib_size = np.asarray(lib_size, float)
    off = (np.zeros(len(ytotal)) if covariates is None
           else covariate_offset(ytotal, lib_size, covariates)[0])

    passed_a = (y1 >= asc_cutoff) & (y2 >= asc_cutoff) & (y1 <= asc_cap) & (y2 <= asc_cap)
    with np.errstate(divide='ignore', invalid='ignore'):
        resp_a = np.where(passed_a, np.log(np.where(passed_a, y1, 1.0) /
                                           np.where(passed_a, y2, 1.0)), 0.0)
    w_a = harmonic_weights(np.where(passed_a, y1, 1.0), np.where(passed_a, y2, 1.0))
    w_a = np.where(passed_a, w_a, 0.0)
    n_a = int(passed_a.sum())
    w_a, _, _ = apply_weight_cap(w_a, n_a, weight_cap, passed=passed_a,
                                 strict_reference_cap=strict_reference_cap)
    w_a = np.where(passed_a, w_a, 0.0)

    with np.errstate(divide='ignore', invalid='ignore'):
        resp_t_full = np.log(ytotal / 2.0 / lib_size) - off
    passed_t = np.isfinite(resp_t_full) & (ytotal >= trc_cutoff)
    resp_t = np.where(passed_t, resp_t_full, 0.0)
    n_t = int(passed_t.sum())

    mono_a = Xasc.var(axis=0) == 0
    mono_t = Xtrc.var(axis=0) == 0

    out = np.full(len(perm_idx), np.nan)
    for k, idx in enumerate(perm_idx):
        wk = w_a[idx]
        ak = resp_a[idx]
        b_a = np.full(Xasc.shape[1], np.nan); s_a = np.full(Xasc.shape[1], np.nan)
        if (~mono_a).any() and wk.sum() > 0:
            b_a[~mono_a], s_a[~mono_a] = _simple_regression_through_origin(
                ak, Xasc[:, ~mono_a], wk)

        mk = passed_t[idx].astype(float)
        tk = resp_t[idx]
        b_t = np.full(Xtrc.shape[1], np.nan); s_t = np.full(Xtrc.shape[1], np.nan)
        if (~mono_t).any() and mk.sum() > 2:
            sel = mk > 0
            b_t[~mono_t], s_t[~mono_t] = _simple_regression_with_intercept(
                tk[sel], Xtrc[sel][:, ~mono_t])

        m = meta_analyze(dict(beta=b_t, se=s_t, sample_size=n_t),
                         dict(beta=b_a, se=s_a, sample_size=n_a))
        stat = np.abs(m['meta']['stat'])
        out[k] = np.nanmax(stat) if np.isfinite(stat).any() else np.nan
    return out

"""DEPRECATED fitted-variance machinery, quarantined 2026-09-23.

This module is NOT part of either shipped mode. It exists only to reproduce
results recorded before 2026-09-23 and will be removed.

NAMING WARNING: "fitted variance" here means a variance FUNCTION fitted per
gene from that gene's own squared residuals -- `tau_g` alone under the
`additive` model, or `(c_g, tau_g)` jointly under `two_component` /
`library_scaled`, optionally shrunk by `estimate_variance_priors`. It does NOT
mean `se_mode='fitted'`, which is the SHIPPED default and stays in
`hapmixqtl.py`. The two are opposites: `se_mode='fitted'` fits one residual
SCALE given weights it did not choose, whereas everything in this file fits
the weight function itself from the residuals it then weights.

Why these are deprecated rather than merely old (user decision, 2026-09-23):

  1. Circularity. Every estimator here fits its per-observation variance FROM
     a gene's own squared residuals and then weights those same residuals by
     the fit. No comparator method does this -- limma, edgeR, sleuth and swish
     all fix the per-observation variance before a gene's residuals are seen --
     and it breaks the premise that makes the resulting statistic exact.
  2. The free-`c` models discard the draws' calibration. With `(c_g, tau_g)`
     both free, rescaling every `v_ig` in a gene by `k` returns `c_g/k` with
     `tau_g` unchanged, so the weights are invariant to the absolute scale of
     the Gibbs draws and only their within-gene shape survives. Propagating
     the quantifier's uncertainty is the point of the method, so a model that
     cannot feel that scale is answering a different question.

Efficiency arguments do not reopen this. A model whose weights are fitted
from the residuals they weight can win on realized variance and still be
unsound, because the quantity it optimises is not the quantity it reports.

The two shipped modes never import this module, which a test pins.

The live module owns the sparse-channel rule (`_min_informative`) and the
weighted residualizer, and this module borrows them. The dependency is
one-way: `hapmixqtl` imports this module only from inside the deprecated
branches, so it is always fully initialised before this import runs and there
is no cycle in either direction.
"""

import numpy as np
import pandas as pd
import torch

try:                                                    # package import
    from .hapmixqtl import WeightedResidualizer, _min_informative
except ImportError:                                     # flat import
    from hapmixqtl import WeightedResidualizer, _min_informative


def _estimate_tau(y_t, v_inf_t, covariates_t, device, intercept=True):
    """
    Estimate overdispersion parameter tau using moment estimator.

    Under Var(error_i) = v_inf_i + tau, whitening by w_i = 1/v_inf_i gives
    Var(y*_i) = 1 + tau*w_i, so the residual sum of squares after projecting
    out the null design P = QQ' has expectation

        E[RSS] = tr((I - P) diag(1 + tau*w)) = (n - q) + tau * sum_i w_i(1-h_i)

    with h_i the leverage (the diagonal of P) and q the design's rank. Solving
    gives the estimator below. The denominator is sum_i w_i(1-h_i), NOT
    (n - q) * mean(w): the two agree only when every sample has the same
    leverage, and for an intercept-only design (h_i = w_i / sum_j w_j) the
    correct form reduces exactly to DerSimonian and Laird's
    sum_i w_i - sum_i w_i^2 / sum_j w_j. Using the mean weight understated tau
    by a median 0.8% in the allelic channel and 3.0% in the total channel of
    the BrainVar genes, whose whitened 18-column design reaches a leverage of
    0.74.
    """
    w = 1.0 / v_inf_t.clamp(min=1e-8)
    sqrt_w = torch.sqrt(w)
    res = WeightedResidualizer(covariates_t, sqrt_w, intercept=intercept)
    y_star = (y_t * sqrt_w).unsqueeze(0)
    y_res = res.transform(y_star).squeeze()

    rss = (y_res * y_res).sum()
    dof_null = y_t.shape[0] - res.Q_t.shape[1]
    h = (res.Q_t * res.Q_t).sum(1)
    denom = (w * (1.0 - h)).sum()
    tau = torch.clamp((rss - dof_null) / denom.clamp(min=1e-30), min=0.0)
    return tau

def _estimate_tau_informative(y_t, v_inf_t, covariates_t, device, eps=1e-12,
                              intercept=True):
    """_estimate_tau over the samples with v_inf > eps (see _prepare_channels).

    A sample with v_inf = 0 carries no information and must never enter the
    moment estimator: it would dominate mean(1/v) and collapse tau. There is
    deliberately no fallback to every sample when few informative ones remain
    (an earlier version had one, and re-admitted exactly those rows for
    sparse genes). Raises instead; _prepare_channels switches such a channel
    off before getting here, so reaching the error means the sparse-channel
    rule was bypassed.
    """
    keep = v_inf_t > eps
    n_keep = int(keep.sum())
    if n_keep < _min_informative(covariates_t, intercept=intercept):
        n_cols = int(intercept) + (0 if covariates_t is None else covariates_t.shape[1])
        raise ValueError(
            f'tau cannot be estimated from {n_keep} informative samples against '
            f'a design of {n_cols} columns; the channel must be switched off')
    c = None if covariates_t is None else covariates_t[keep]
    return _estimate_tau(y_t[keep], v_inf_t[keep], c, device, intercept=intercept)

# DEPRECATED (2026-09-23), all three. The shipped error model is
#     Var(eps_i) = sigma^2 * v_i        (TIMES, no additive floor)
# reached by tau_mode='zero' + se_mode='fitted'. These three, the
# variance_prior shrinkage and the tau_mode='estimate' they require are kept
# ONLY to reproduce historical results and WILL BE REMOVED. They are not
# alternatives and not a fallback. They are inferior for a structural reason:
# each fits its layer-1 variance from a gene's own squared residuals and then
# weights those same residuals, which no comparator method does, and the
# free-c forms are additionally invariant to the absolute scale of the Gibbs
# draws so the quantifier's calibration never reaches the answer. Per-gene
# efficiency comparisons do not rehabilitate them; efficiency was never the
# objection. See CLAUDE.md, "The variance models are deprecated".
VARIANCE_MODELS = ('additive', 'two_component', 'library_scaled')

def _check_variance_model(variance_model, tau_mode, library_factor_t, variance_prior=None):
    """Argument validation shared by the mapping functions."""
    if variance_prior is not None:
        if variance_model == 'additive':
            raise ValueError("variance_prior applies to the two-component models only; c is fixed at 1 under 'additive'")
        if not isinstance(variance_prior, pd.DataFrame) or 'prior_c' not in variance_prior.columns:
            raise ValueError('variance_prior must be the DataFrame returned by estimate_variance_priors')
        if 'library_scaled' not in variance_prior.attrs or 'kappa' not in variance_prior.attrs:
            raise ValueError('variance_prior has lost its attrs (kappa, library_scaled): pass the frame '
                             'estimate_variance_priors returned, not one reloaded from a file')
        if bool(variance_prior.attrs['library_scaled']) != (variance_model == 'library_scaled'):
            raise ValueError("variance_prior was estimated for a different model: pass library_factor to "
                             "estimate_variance_priors exactly when the scan uses 'library_scaled'")
    if variance_model not in VARIANCE_MODELS:
        raise ValueError(f'variance_model must be one of {VARIANCE_MODELS}, got {variance_model!r}')
    if variance_model != 'additive' and tau_mode != 'estimate':
        raise ValueError(f"variance_model={variance_model!r} requires tau_mode='estimate'")
    if variance_model == 'library_scaled' and library_factor_t is None:
        raise ValueError(
            "variance_model='library_scaled' needs library_factor: one positive value per "
            "sample, estimated across genes with estimate_library_factors(A_df, Va_df, ...)")
    if variance_model != 'library_scaled' and library_factor_t is not None:
        raise ValueError("library_factor is only used under variance_model='library_scaled'")

def _library_factor_tensor(library_factor, samples, device):
    """library_factor as a float32 tensor aligned to the phenotype samples, or
    None. Accepts a Series indexed by sample (reindexed, every sample
    required) or an array in phenotype column order."""
    if library_factor is None:
        return None
    if isinstance(library_factor, pd.Series):
        d = library_factor.reindex(samples).values.astype(float)
    else:
        d = np.asarray(library_factor, dtype=float)
        assert d.shape == (len(samples),), 'library_factor must have one value per sample'
    if not np.all(np.isfinite(d)) or np.any(d <= 0):
        raise ValueError('library_factor must be finite and positive for every sample')
    return torch.tensor(d, dtype=torch.float32).to(device)

def _estimate_c_tau(y_t, v_t, covariates_t, device, intercept=True, d_t=None,
                    max_iter=500, tol=1e-7, prior=None, _theta0=None):
    """
    Two-component variance fit for one channel of one gene,

        Var(e_i) = d_i (c v_i + tau),   c >= 0, tau >= 0,

    on the informative samples (the caller drops v <= eps). d_t is the
    per-library factor (None means 1 for every sample).

    Under weights w_i = 1/(d_i (c v_i + tau)) the leverage-corrected squared
    null residual e2_i = r_i^2 / (w_i (1 - h_i)) has expectation
    d_i (c v_i + tau), so e2_i / d_i regressed on [v_i, 1] gives the slope c
    and intercept tau. A squared Gaussian residual has variance proportional
    to its variance squared, so the line is fitted with weights
    1/(c v_i + tau)^2; the weights depend on (c, tau), so the fit iterates,
    damped by averaging each update with the previous value (the undamped
    update can cycle between the tau = 0 clamp and an interior point). When
    a clamp hits, the other parameter is refitted alone. Starts at c = 1 and
    the DerSimonian-Laird tau of y/sqrt(d). Runs in float64.

    This is the estimator whose calibration was measured on BrainVar (300
    genes across expression tiers, 40 genotype permutations each: type-I
    0.029-0.041 at nominal 0.05; deprecated_models/estimator_ablation_tiers_20260917/
    tiered_calibration.py, fit_cvt). With d = 1 it is that prototype exactly;
    with d != 1 and an empty design it equals fitting y/sqrt(d) against v,
    which is what the prototype's library-scaled configuration did.

    ``prior`` replaces the clamp with empirical-Bayes shrinkage: a tuple
    (m_logc, s_logc, m_logtau, s_logtau, kappa) from estimate_variance_priors,
    independent normal priors on log c and log tau. The fit is then the
    posterior mode in (log c, log tau) of the Gaussian likelihood of the
    leverage-corrected squared residuals, tempered by 2/kappa (kappa =
    Var(z^2) of the standardized residuals; 2 under Gaussian errors, larger
    with heavy tails), found by Fisher scoring with a backtracking line
    search on the penalized objective. The unpenalized
    stationary point of that likelihood is the same weighted regression of
    e^2 on [v, 1] as the clamped fit (Fisher scoring for a Gaussian variance
    model is that iteration), so the two estimators agree away from the
    boundary; on the log scale positivity is automatic and nothing is
    clamped. Each scoring step is backtracked (halved until the penalized
    objective, the gamma quasi-log-likelihood plus the log prior, does not
    decrease): without that, the step overshoots along the direction the
    gene's data do not identify and the iteration cycles between two points
    at the step cap. The ascent is run from two starts, the prior mean and
    the unpenalized clamped fit, and the higher objective is kept: under a
    prior centred far below a gene's identified c the objective is bimodal
    and the ascent from the prior mean alone stops in the spurious mode near
    the prior. On the prior path ``floored`` is always False (nothing is
    clamped); on the clamped path it reports whether a clamp branch was taken
    on the final iteration.

    Returns a dict: c, tau (the values the weights use), converged,
    c_raw, tau_raw (the unpenalized, unclamped solution at the final
    weights), floored. A fit that has not met the tolerance after
    ``max_iter`` iterations returns its last iterate with converged False.
    """
    y = y_t.to(torch.float64)
    v = v_t.to(torch.float64)
    cov = None if covariates_t is None else covariates_t.to(torch.float64)
    d = torch.ones_like(v) if d_t is None else d_t.to(torch.float64)
    c = 1.0
    tau = float(_estimate_tau(y / torch.sqrt(d), v, cov, device, intercept=intercept)) \
        if y.shape[0] > 3 else 0.0
    ones = torch.ones_like(v)
    converged = False
    hit = False
    c_raw, tau_raw = float('nan'), float('nan')

    def _moments(c_, tau_):
        base = (c_ * v + tau_).clamp(min=1e-10)
        w = 1.0 / (d * base)
        sw = torch.sqrt(w)
        res = WeightedResidualizer(cov, sw, intercept=intercept)
        r = res.transform((y * sw).unsqueeze(0))[0]
        h = (res.Q_t * res.Q_t).sum(1)
        e2 = (r * r) / (w * (1.0 - h).clamp(min=1e-3)) / d
        om = 1.0 / (base * base)
        X = torch.stack([v, ones], 1)
        return base, e2, om, X

    if prior is None:
        for _ in range(max_iter):
            base, e2, om, X = _moments(c, tau)
            Am = X.T @ (om[:, None] * X)
            b = X.T @ (om * e2)
            try:
                sol = torch.linalg.solve(Am, b)
                c_raw, tau_raw = float(sol[0]), float(sol[1])
            except Exception:
                c_raw, tau_raw = c, tau
            cn, tn = c_raw, tau_raw
            hit = False
            if cn < 0:
                cn = 0.0
                tn = float((om * e2).sum() / om.sum())
                hit = True
            if tn < 0:
                tn = 0.0
                cn = float((om * e2 * v).sum() / (om * v * v).sum())
                hit = True
            if cn <= 0 and tn <= 0:
                cn, tn = 1e-6, 1e-6
            cn, tn = 0.5 * (c + cn), 0.5 * (tau + tn)
            done = abs(cn - c) < tol * (1 + c) and abs(tn - tau) < tol * (1 + tau)
            c, tau = cn, tn
            if done:
                converged = True
                break
        return dict(c=c, tau=tau, converged=converged, c_raw=c_raw, tau_raw=tau_raw, floored=hit)

    m_logc, s_logc, m_logtau, s_logtau, kappa = [float(x) for x in prior]
    dev64 = dict(dtype=torch.float64, device=y.device)
    m = torch.tensor([m_logc, m_logtau], **dev64)
    P = torch.tensor([[1.0 / s_logc ** 2, 0.0], [0.0, 1.0 / s_logtau ** 2]], **dev64)

    def _merit(th):
        # The objective whose gradient is the tempered score below: Wedderburn's
        # quasi-log-likelihood for mean c v + tau and variance function 2 mu^2,
        # Q = sum(-e2/base - log base), times 2/kappa, plus the log prior. The
        # factor must match the score's 1/kappa exactly: with a 0.5 here the
        # line search accepts only moves toward an objective in which the prior
        # weighs twice as much, and the iteration stalls between the two modes
        # (measured: 190 of 200 identified genes moved, up to a factor 9).
        base, e2, om, X = _moments(float(torch.exp(th[0])), float(torch.exp(th[1])))
        ll = -float((torch.log(base) + e2 / base).sum()) / kappa
        dth = th - m
        return ll - 0.5 * float(dth @ (P @ dth)), (base, e2, om, X)

    if _theta0 is None:
        # The penalized objective is bimodal under a low-centred prior (the
        # two lowest expression bins on BrainVar have a prior median c of
        # 0.0018): an ascent from the prior mean alone settles in the spurious
        # low-c mode for a gene whose data identify c near 2, and reports it
        # converged, 12 log-posterior units below the mode. Ascend from both
        # the prior mean and the unpenalized clamped fit and keep the higher
        # objective; the tau seed is floored because the clamped fit often
        # returns tau = 0 exactly.
        cl = _estimate_c_tau(y_t, v_t, covariates_t, device, intercept, d_t=d_t,
                             max_iter=max_iter, tol=tol, prior=None)
        alt = [float(np.log(max(cl['c'], 1e-8))),
               float(np.log(max(cl['tau'], float(np.exp(m_logtau)) * 1e-3)))]
        a = _estimate_c_tau(y_t, v_t, covariates_t, device, intercept, d_t=d_t,
                            max_iter=max_iter, tol=tol, prior=prior,
                            _theta0=[m_logc, m_logtau])
        b = _estimate_c_tau(y_t, v_t, covariates_t, device, intercept, d_t=d_t,
                            max_iter=max_iter, tol=tol, prior=prior, _theta0=alt)
        ma = _merit(torch.tensor([np.log(a['c']), np.log(a['tau'])], **dev64))[0]
        mb = _merit(torch.tensor([np.log(b['c']), np.log(b['tau'])], **dev64))[0]
        return b if mb > ma else a
    theta = torch.tensor([float(_theta0[0]), float(_theta0[1])], **dev64)
    merit, (base, e2, om, X) = _merit(theta)
    for _ in range(max_iter):
        Am = X.T @ (om[:, None] * X)
        b = X.T @ (om * e2)
        try:
            sol = torch.linalg.solve(Am, b)
            c_raw, tau_raw = float(sol[0]), float(sol[1])
        except Exception:
            c_raw, tau_raw = float(torch.exp(theta[0])), float(torch.exp(theta[1]))
        # tempered Gaussian score and Fisher information in (c, tau), then in (log c, log tau)
        g = (X.T @ (om * (e2 - base))) / kappa
        J = torch.diag(torch.exp(theta))
        g_th = J @ g
        I_th = J @ (Am / kappa) @ J
        try:
            step = torch.linalg.solve(I_th + P, g_th - P @ (theta - m))
        except Exception:
            break
        # Cap the step at one unit on the log scale, scaling the whole vector
        # so its direction is kept: clamping each component separately turns
        # the ascent direction into one that need not ascend, and the search
        # below then stalls at a point that is not a maximum (measured with a
        # near-flat prior: 5 of 22 identified genes stopped up to a factor 67
        # from the clamped fit; scaled, all agree to within 0.7%).
        step_max = float(step.abs().max())
        if step_max > 1.0:
            step = step / step_max
        # Backtracking line search. The expected information is nearly zero
        # along the direction the gene's data do not identify (log c when c v
        # is negligible against tau, log tau in the opposite case), so the full
        # step overshoots there and, undamped, the iteration settles into a
        # two-cycle at the step cap (12.7% of BrainVar genes did). Halving the
        # step until the penalized objective does not decrease makes every
        # iteration an ascent and leaves a step that already ascends untouched.
        # Thirty halvings are needed: with twelve, 58 of 60 such genes still
        # failed to converge. Along that flat direction the accepted step
        # shrinks slowly, so convergence is also declared when the objective's
        # gain is negligible; 1e-10 relative reproduces the step-rule endpoint
        # to 0.1% and converges every gene (median 20 iterations, at most 350).
        accepted = False
        for _k in range(30):
            cand = theta + step
            merit_c, mom_c = _merit(cand)
            if merit_c >= merit:
                accepted = True
                break
            step = 0.5 * step
        if not accepted:
            converged = True  # no ascent at any scale: the gradient is numerically zero
            break
        gain = merit_c - merit
        theta, merit, (base, e2, om, X) = cand, merit_c, mom_c
        if float(step.abs().max()) < tol or gain < 1e-10 * (1.0 + abs(merit)):
            converged = True
            break
    c, tau = float(torch.exp(theta[0])), float(torch.exp(theta[1]))
    return dict(c=c, tau=tau, converged=converged, c_raw=c_raw, tau_raw=tau_raw, floored=False)

def _fit_c_tau_vectorized(a2, v, M, max_iter=400, tol=1e-7):
    """_estimate_c_tau for many genes at once, through-origin and without
    covariates (h = 0, so e2 = a^2), in numpy: rows are genes, M marks the
    informative samples. Same start, weights, clamps, damping and tolerance
    as the per-gene fit. Returns (c, tau) arrays."""
    a2 = np.where(M, a2, 0.0)
    v = np.where(M, v, 1.0)
    w0 = np.where(M, 1.0 / v, 0.0)
    n = M.sum(1)
    tau = np.clip(((a2 * w0).sum(1) - n) / np.maximum(w0.sum(1), 1e-300), 0.0, None)
    c = np.ones(a2.shape[0])
    active = np.ones(a2.shape[0], bool)
    for _ in range(max_iter):
        base = np.maximum(c[:, None] * v + tau[:, None], 1e-10)
        om = np.where(M, 1.0 / (base * base), 0.0)
        Svv, Sv1, S11 = (om * v * v).sum(1), (om * v).sum(1), om.sum(1)
        Sve, S1e = (om * v * a2).sum(1), (om * a2).sum(1)
        det = Svv * S11 - Sv1 ** 2
        ok = det > 1e-300
        safe = np.where(ok, det, 1.0)
        cn = np.where(ok, (Sve * S11 - S1e * Sv1) / safe, c)
        tn = np.where(ok, (Svv * S1e - Sv1 * Sve) / safe, tau)
        neg_c = cn < 0
        cn = np.where(neg_c, 0.0, cn)
        tn = np.where(neg_c, S1e / np.maximum(S11, 1e-300), tn)
        neg_t = tn < 0
        tn = np.where(neg_t, 0.0, tn)
        cn = np.where(neg_t, Sve / np.maximum(Svv, 1e-300), cn)
        both = (cn <= 0) & (tn <= 0)
        cn = np.where(both, 1e-6, cn)
        tn = np.where(both, 1e-6, tn)
        cn, tn = 0.5 * (c + cn), 0.5 * (tau + tn)
        done = (np.abs(cn - c) < tol * (1 + c)) & (np.abs(tn - tau) < tol * (1 + tau))
        c = np.where(active, cn, c)
        tau = np.where(active, tn, tau)
        active &= ~done
        if not active.any():
            break
    return c, tau

def _raw_c_tau_and_cov(a2, v, M, c, tau, kappa=2.0, robust=False):
    """At the weights implied by (c, tau) per gene, the unclamped
    weighted-least-squares solution of a^2 on [v, 1] and its sampling
    covariance. With ``robust=False`` the model-based kappa (X' Omega X)^-1,
    kappa = Var(z^2) of the standardized residuals, which is right when (c,
    tau) are the gene's own converged fit. With ``robust=True`` the sandwich
    (X' Omega X)^-1 X' Omega diag(r^2) Omega X (X' Omega X)^-1 with r the
    residuals of a^2 about the unclamped line, which stays honest when the
    weights come from elsewhere (the trend prior's second pass): a gene far
    from the curve then keeps a large sampling variance instead of the
    spuriously small model-based one. Returns c_raw, tau_raw, var_c, var_tau
    (arrays over genes)."""
    a2 = np.where(M, a2, 0.0)
    v = np.where(M, v, 1.0)
    base = np.maximum(c[:, None] * v + tau[:, None], 1e-10)
    om = np.where(M, 1.0 / (base * base), 0.0)
    Svv, Sv1, S11 = (om * v * v).sum(1), (om * v).sum(1), om.sum(1)
    Sve, S1e = (om * v * a2).sum(1), (om * a2).sum(1)
    det = np.maximum(Svv * S11 - Sv1 ** 2, 1e-300)
    c_raw = (Sve * S11 - S1e * Sv1) / det
    tau_raw = (Svv * S1e - Sv1 * Sve) / det
    if not robust:
        return c_raw, tau_raw, kappa * S11 / det, kappa * Svv / det
    r = np.where(M, a2 - (c_raw[:, None] * v + tau_raw[:, None]), 0.0)
    w2r2 = om * om * r * r
    Bvv, Bv1, B11 = (w2r2 * v * v).sum(1), (w2r2 * v).sum(1), w2r2.sum(1)
    var_c = (S11 * S11 * Bvv - 2 * S11 * Sv1 * Bv1 + Sv1 * Sv1 * B11) / (det * det)
    var_tau = (Sv1 * Sv1 * Bvv - 2 * Sv1 * Svv * Bv1 + Svv * Svv * B11) / (det * det)
    return c_raw, tau_raw, np.maximum(var_c, 1e-300), np.maximum(var_tau, 1e-300)

PRIOR_METHODS = ('deciles', 'trend')

def _trend_prior(x, raw, var, span=0.15, min_width=0.25, floor_abs=0.2, n_grid=40, n_nodes=24):
    """A smooth empirical-Bayes prior on the log scale for a positive parameter,
    fitted by local marginal likelihood from the raw, unbiased, possibly
    negative per-gene estimates: raw_g ~ N(p_g, var_g) and log p_g ~ N(m(x_g),
    s(x_g)^2) with x the log10 expression. At each of ``n_grid`` grid points
    (quantiles of x) the kernel-weighted log marginal likelihood, the integral
    over log p done by Gauss-Hermite quadrature with ``n_nodes`` nodes, is
    maximized over a local line for the mean and a local constant for the
    spread (limma's trend=TRUE idea for a variance prior, with the
    normal-lognormal deconvolution in place of a moment match). Every gene
    enters, a non-positive raw estimate included: it says the parameter is
    small relative to its sampling error and pulls the curve down where such
    genes are common, which a fit restricted to positive estimates would
    miss (measured on BrainVar: that restriction put the prior for tau above
    1,000 reads at 0.029 where the clamped fits' median is 0.003).

    Windows are tricube kernels whose half-width at each grid point is the
    distance to the ``span``-fraction nearest neighbour, never below
    ``min_width``. The spread is bounded below by ``floor_abs`` (0.2 on the log
    scale: a prior tighter than about 20% would over-shrink identified
    genes). Raw estimates are winsorized at the 0.5th and 99.5th percentiles.
    Returns (grid, m, s), to be read by interpolation, flat beyond the grid.
    """
    from scipy.optimize import minimize
    from scipy.special import logsumexp
    x = np.asarray(x, float); raw = np.asarray(raw, float); var = np.asarray(var, float)
    n = len(x)
    lo, hi = np.percentile(raw, [0.5, 99.5]); raw = np.clip(raw, lo, hi)
    sig = np.sqrt(np.maximum(var, 1e-12))
    k = max(int(round(span * n)), 10)
    grid = np.quantile(x, np.linspace(0.01, 0.99, n_grid))
    grid = np.unique(np.concatenate([[x.min()], grid, [x.max()]]))
    xs = np.sort(x)
    t, wq = np.polynomial.hermite.hermgauss(n_nodes)
    logw = np.log(wq) - 0.5 * np.log(np.pi)
    sqrt2 = np.sqrt(2.0)
    pos = raw > 0
    m = np.empty(len(grid)); sd = np.empty(len(grid))
    theta = None
    for i, x0 in enumerate(grid):
        d = np.abs(xs - x0)
        h = max(float(np.partition(d, min(k, n - 1))[min(k, n - 1)]), min_width)
        u = (x - x0) / h
        K = np.where(np.abs(u) < 1, (1 - np.abs(u) ** 3) ** 3, 0.0)
        idx = np.nonzero(K > 0)[0]
        Kw, xw, rw, sw = K[idx], x[idx] - x0, raw[idx], sig[idx]
        if theta is None:
            pw = idx[pos[idx]]
            m_init = float(np.median(np.log(raw[pw]))) if len(pw) >= 5 else float(np.log(max(np.mean(np.abs(rw)), 1e-6)))
            theta = np.array([m_init, 0.0, np.log(0.5)])

        def nll(th):
            m0, b, ls = th
            s_ = np.exp(ls)
            mu = m0 + b * xw
            P = np.exp(mu[:, None] + sqrt2 * s_ * t[None, :])            # [n_w, nodes]
            z = (rw[:, None] - P) / sw[:, None]
            lp = -0.5 * z * z - np.log(sw)[:, None] - 0.5 * np.log(2 * np.pi) + logw[None, :]
            return -float((Kw * logsumexp(lp, axis=1)).sum())

        # two starts per window: the previous window's optimum and the best
        # of a coarse grid over the mean (the surface is rough where the
        # sampling variances are small, and a single warm start carried up
        # the expression range stalled at the top decile on BrainVar)
        bounds = [(-30, 30), (-20, 20), (np.log(floor_abs), np.log(5.0))]
        mg = np.linspace(theta[0] - 8, theta[0] + 8, 33)
        g_best = mg[int(np.argmin([nll([mm, 0.0, theta[2]]) for mm in mg]))]
        best = None
        for start in (theta, np.array([g_best, 0.0, theta[2]])):
            res = minimize(nll, start, method='L-BFGS-B', bounds=bounds)
            if best is None or res.fun < best.fun:
                best = res
        theta = best.x
        m[i] = theta[0]
        sd[i] = float(np.exp(theta[2]))
    return grid, m, sd

def estimate_variance_priors(A_df, Va_df, genes=None, n_bins=10, expression=None,
                             min_informative=40, library_factor=None, floor_frac=0.1,
                             eps=1e-12, max_iter=400, tol=1e-7, prior_method='deciles', span=0.15, pass2_variance='model'):
    """
    Empirical-Bayes priors for the allelic (c, tau) of the two-component
    models, one prior per expression bin, to be passed as ``variance_prior``
    to the mapping functions in place of the zero clamp.

    Every gene with at least ``min_informative`` informative samples is fitted
    through the origin (_fit_c_tau_vectorized, clamped, as the scan without a
    prior would). At each gene's converged weights the unclamped solution
    (c_raw, tau_raw) and its sampling covariance are taken from the
    squared-residual regression, with kappa = Var(z^2) of the standardized
    residuals estimated as the median over genes of E[z^4] - 1 (2 under
    Gaussian errors; larger with the heavy tails these residuals have).
    Genes are binned by expression, ``expression`` if supplied (a Series by
    gene, e.g. median allele-resolved reads) and otherwise the median over
    informative samples of log Va, which falls as 1/reads. In each bin the
    natural-scale mean of each parameter is the mean of the raw estimates
    (unbiased even when some are negative; winsorized at the 1st and 99th
    percentiles) and its between-gene variance is the variance of the raw
    estimates minus the median sampling variance, floored at ``floor_frac``
    of the variance so the prior never collapses to a point:
    DerSimonian-Laird's step applied across genes. Because c and tau are
    positive and right-skewed, the prior is the log-normal with those two
    moments (a normal prior on the natural scale is nearly uninformative
    about the sign of tau where its mean is small against its spread, and
    reintroduces the clamp), so the per-gene fit is penalized on the log
    scale and needs no clamp. A bin whose raw mean is not positive (on
    BrainVar the two lowest expression bins, for c) has it replaced by a
    twentieth of the raw spread so the log-normal exists; the per-bin table
    reports the raw mean (mean_raw_c, mean_raw_tau) and whether the
    replacement happened (c_mean_floored, tau_mean_floored). A bin left with
    fewer than 10 genes by tied proxies takes the pooled prior (pooled).

    ``prior_method='trend'`` replaces the ten bins by two smooth curves, one
    per parameter, fitted on the log scale by local marginal likelihood
    (_trend_prior): each gene's raw unbiased estimate is normal around the
    true value with its sampling variance, the log of the true value is
    normal around a locally linear curve in log10 expression with a locally
    constant spread, and the curve and spread are the kernel-weighted
    maximum-likelihood fit with the integral over the log value done by
    Gauss-Hermite quadrature. Every gene enters, negative raw estimates
    included, so the curve is not biased by dropping the genes whose true
    value is near zero; the spread is floored at 0.2 on the log scale.
    Every gene's prior is the curve at its expression, flat beyond the
    fitted range. The fit is made twice: the raw estimates computed with each
    gene's own clamped weights are biased low (those weights are correlated
    with the gene's noise), so the curves from that pass supply weights for a
    second set of raw estimates, independent of each gene's residuals, from
    which the final curves are fitted (measured on a simulated smooth truth:
    curve error median 0.05 on the log scale, against 0.19 in one pass; the
    between-gene spread is over-estimated by about a quarter because the
    model-based sampling variances are slightly understated under external
    weights, which errs toward shrinking less). The decile prior's moment
    match on the natural scale is what put the prior median of c at 0.0018
    in the two lowest bins and made the posterior bimodal; the trend prior
    works on the log scale from the start. The bins table reports the trend
    at each decile's median expression beside the decile prior, and the
    fraction of genes whose (second-pass) raw estimate is not positive.
    The per-gene c_raw and tau_raw columns stay the first-pass values, with
    their model-based sampling variances in c_raw_var and tau_raw_var; the
    trend adds the second-pass estimates and variances as c_raw_pass2,
    tau_raw_pass2, c_raw_pass2_var and tau_raw_pass2_var, so the prior can
    be checked against the raw estimates it was fitted to (for instance the
    fraction of non-positive raw estimates it predicts in a decile against
    the fraction observed).

    The prior is for the model the scan will use: pass ``library_factor``
    when the scan is 'library_scaled' (the raw fits are then on a/sqrt(d));
    map_cis checks the pairing.

    Returns a DataFrame indexed by every gene of A_df with columns bin,
    prior_c, prior_tau, prior_sd_c, prior_sd_tau (natural-scale mean and
    between-gene sd), prior_logc_m, prior_logc_s, prior_logtau_m,
    prior_logtau_s (the log-normal prior the fit uses), c_raw, tau_raw,
    c_raw_var, tau_raw_var (NaN for genes outside the estimation set),
    expression_proxy; attrs
    carry 'bins' (the per-bin table; under 'trend' the decile prior's columns
    stay as a reference and the trend's values at each decile's median
    expression are added as trend_*), 'kappa', 'n_genes', 'library_scaled',
    'method', and under 'trend' also 'span' and 'curve' (a DataFrame of the
    fitted curves on a fine grid of log10 expression).
    """
    if prior_method not in PRIOR_METHODS:
        raise ValueError(f'prior_method must be one of {PRIOR_METHODS}, got {prior_method!r}')
    A = np.asarray(A_df.values, dtype=float)
    V = np.asarray(Va_df.values, dtype=float)
    if library_factor is not None:
        dvec = _library_factor_tensor(library_factor, A_df.columns, 'cpu').numpy().astype(float)
        A = A / np.sqrt(dvec)[None, :]
    M = V > eps
    n_inf = M.sum(1)
    if expression is not None:
        proxy = pd.Series(expression).reindex(A_df.index).values.astype(float)
    else:
        with np.errstate(divide='ignore', invalid='ignore'):
            lv = np.where(M, np.log(np.where(M, V, 1.0)), np.nan)
            proxy = -np.nanmedian(lv, axis=1)   # larger = more expressed
    est = (n_inf >= min_informative) & np.isfinite(proxy)
    if genes is not None:
        est &= A_df.index.isin(pd.Index(genes))
    if est.sum() < 10 * n_bins:
        raise ValueError(f'only {int(est.sum())} genes are eligible for {n_bins} bins; lower n_bins or min_informative')
    Ae, Ve, Me = A[est], V[est], M[est]
    c, tau = _fit_c_tau_vectorized(Ae ** 2, Ve, Me, max_iter=max_iter, tol=tol)
    z2 = np.where(Me, Ae ** 2 / np.maximum(c[:, None] * Ve + tau[:, None], 1e-300), np.nan)
    kappa = max(float(np.nanmedian(np.nanmean(z2 * z2, axis=1) - 1.0)), 2.0)
    c_raw, tau_raw, var_c, var_tau = _raw_c_tau_and_cov(Ae ** 2, Ve, Me, c, tau, kappa)
    edges = np.quantile(proxy[est], np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    b_est = np.clip(np.searchsorted(edges, proxy[est], side='right') - 1, 0, n_bins - 1)

    def _prior(x, var_x):
        """Natural-scale mean and between-gene variance of a positive parameter
        from its unbiased (possibly negative) raw estimates, then the
        log-normal prior with those moments."""
        lo, hi = np.percentile(x, [1, 99])
        xw = np.clip(x, lo, hi)
        mu = float(np.mean(xw))
        spread = float(np.var(xw))
        noise = float(np.median(var_x))
        var_true = max(spread - noise, floor_frac * spread, 1e-24)
        # The log-normal needs a positive mean. Where the bin's raw mean is at
        # or below zero (the two lowest expression bins on BrainVar, whose raw
        # mean of c is negative) it is replaced by a twentieth of the raw
        # spread; the raw mean and the fact of the replacement are reported.
        mu_floor = 0.05 * np.sqrt(spread)
        mu_pos = max(mu, mu_floor, 1e-12)
        s2 = float(np.log1p(var_true / mu_pos ** 2))
        return (mu_pos, float(np.sqrt(var_true)), float(np.log(mu_pos) - 0.5 * s2), float(np.sqrt(s2)),
                mu, bool(mu_pos > mu))
    rows = []
    for k in range(n_bins):
        sel = b_est == k
        pooled = int(sel.sum()) < 10   # tied proxies can leave a bin empty: fall back to all genes
        use = np.ones_like(sel) if pooled else sel
        mu_c, sd_c, lm_c, ls_c, raw_c, fl_c = _prior(c_raw[use], var_c[use])
        mu_t, sd_t, lm_t, ls_t, raw_t, fl_t = _prior(tau_raw[use], var_tau[use])
        rows.append(dict(bin=k, genes=int(sel.sum()), proxy_lo=edges[k], proxy_hi=edges[k + 1],
                         prior_c=mu_c, prior_sd_c=sd_c, prior_tau=mu_t, prior_sd_tau=sd_t,
                         prior_logc_m=lm_c, prior_logc_s=ls_c, prior_logtau_m=lm_t, prior_logtau_s=ls_t,
                         mean_raw_c=raw_c, c_mean_floored=fl_c, mean_raw_tau=raw_t, tau_mean_floored=fl_t, pooled=pooled,
                         median_c_clamped=float(np.median(c[use])), tau_zero_clamped=float(np.mean(tau[use] < 1e-6))))
    bins = pd.DataFrame(rows)
    proxy_all = np.where(np.isfinite(proxy), proxy, -np.inf)   # genes with no informative sample: lowest bin
    b_all = np.clip(np.searchsorted(edges, proxy_all, side='right') - 1, 0, n_bins - 1)
    out = pd.DataFrame({'bin': b_all, 'expression_proxy': proxy}, index=A_df.index)
    for col in ('prior_c', 'prior_tau', 'prior_sd_c', 'prior_sd_tau',
                'prior_logc_m', 'prior_logc_s', 'prior_logtau_m', 'prior_logtau_s'):
        out[col] = bins[col].values[b_all]
    out['c_raw'] = np.nan
    out['tau_raw'] = np.nan
    out.loc[A_df.index[est], 'c_raw'] = c_raw
    out.loc[A_df.index[est], 'tau_raw'] = tau_raw
    out['c_raw_var'] = np.nan
    out['tau_raw_var'] = np.nan
    out.loc[A_df.index[est], 'c_raw_var'] = var_c
    out.loc[A_df.index[est], 'tau_raw_var'] = var_tau
    out.attrs['method'] = prior_method
    if prior_method == 'trend':
        # log10 expression for the curve; the proxy from the draws is already a log scale
        x_est = np.log10(np.maximum(proxy[est], 1e-6)) if expression is not None else proxy[est]
        x_all = np.log10(np.maximum(np.where(np.isfinite(proxy), proxy, 1e-6), 1e-6)) if expression is not None else np.where(np.isfinite(proxy), proxy, np.nanmin(proxy))
        if est.sum() < 20:
            raise ValueError(f'only {int(est.sum())} genes are eligible; the trend prior needs at least 20 to run and hundreds to mean anything')
        # Two passes. The raw estimates above use each gene's own clamped fit
        # for the weights, and those weights are correlated with the gene's
        # noise (small residuals give a small fitted variance, large weights
        # and a low slope): on a simulated smooth truth the raw estimates ran
        # 0.2 to 0.3 standard errors low and the curve inherited it. The
        # second pass recomputes every raw estimate with weights taken from
        # the first-pass curves at the gene's expression, which do not depend
        # on the gene's own residuals (bias 0.08 / -0.05 s.e. in the same
        # simulation; the true weights give 0.03 / -0.02). Their sampling
        # variance is model-based by default (pass2_variance='model'): it
        # depends on the curve weights and the design, not on the gene's own
        # residuals. The sandwich alternative is small for a gene whose
        # residuals happen to hug its line and brings the noise correlation
        # back (curve bias 0.15 to 0.19 in the simulation); the larger of the
        # two (pass2_variance='max') drove the fitted spread to its floor.
        # The stall this once masked (the top decile's tau prior at 0.5 to
        # 1.9 with a single warm start) is handled in _trend_prior by the
        # second start from a grid over the mean.
        curves = {}
        for name, raw, var in (('c', c_raw, var_c), ('tau', tau_raw, var_tau)):
            grid, m, sd = _trend_prior(x_est, raw, var, span=span)
            curves[name] = (grid, m, sd)
        c0 = np.exp(np.interp(x_est, *curves['c'][:2])); t0 = np.exp(np.interp(x_est, *curves['tau'][:2]))
        c_raw2, tau_raw2, var_c2, var_tau2 = _raw_c_tau_and_cov(Ae ** 2, Ve, Me, c0, t0, kappa)
        if pass2_variance == 'max':
            _, _, var_c2s, var_tau2s = _raw_c_tau_and_cov(Ae ** 2, Ve, Me, c0, t0, kappa, robust=True)
            var_c2, var_tau2 = np.maximum(var_c2, var_c2s), np.maximum(var_tau2, var_tau2s)
        for col, val in (('c_raw_pass2', c_raw2), ('tau_raw_pass2', tau_raw2), ('c_raw_pass2_var', var_c2), ('tau_raw_pass2_var', var_tau2)):
            out[col] = np.nan
            out.loc[A_df.index[est], col] = val
        curves = {}
        for name, raw, var in (('c', c_raw2, var_c2), ('tau', tau_raw2, var_tau2)):
            pos = raw > 0
            grid, m, sd = _trend_prior(x_est, raw, var, span=span)
            curves[name] = (grid, m, sd, pos)
        gc, mc, sc, posc = curves['c']; gt, mt, st, post = curves['tau']
        out['prior_logc_m'] = np.interp(x_all, gc, mc); out['prior_logc_s'] = np.interp(x_all, gc, sc)
        out['prior_logtau_m'] = np.interp(x_all, gt, mt); out['prior_logtau_s'] = np.interp(x_all, gt, st)
        for name in ('c', 'tau'):
            m_, s_ = out[f'prior_log{name}_m'].values, out[f'prior_log{name}_s'].values
            out[f'prior_{name}'] = np.exp(m_ + 0.5 * s_ ** 2)
            out[f'prior_sd_{name}'] = np.sqrt((np.exp(s_ ** 2) - 1.0) * np.exp(2 * m_ + s_ ** 2))
        # the decile table keeps the decile prior as a reference and gains the trend at each decile's median
        xmed = np.array([float(np.median(x_est[b_est == k])) if (b_est == k).any() else np.nan for k in range(n_bins)])
        bins['x_median'] = xmed
        bins['trend_logc_m'] = np.interp(xmed, gc, mc); bins['trend_logc_s'] = np.interp(xmed, gc, sc)
        bins['trend_logtau_m'] = np.interp(xmed, gt, mt); bins['trend_logtau_s'] = np.interp(xmed, gt, st)
        bins['trend_c'] = np.exp(bins['trend_logc_m'] + 0.5 * bins['trend_logc_s'] ** 2)
        bins['trend_tau'] = np.exp(bins['trend_logtau_m'] + 0.5 * bins['trend_logtau_s'] ** 2)
        bins['trend_c_median'] = np.exp(bins['trend_logc_m']); bins['trend_tau_median'] = np.exp(bins['trend_logtau_m'])
        bins['c_raw_nonpositive'] = [float(np.mean(~posc[b_est == k])) if (b_est == k).any() else np.nan for k in range(n_bins)]
        bins['tau_raw_nonpositive'] = [float(np.mean(~post[b_est == k])) if (b_est == k).any() else np.nan for k in range(n_bins)]
        out.attrs['span'] = float(span)
        out.attrs['curve'] = pd.DataFrame({'x_log10': gc, 'logc_m': mc, 'logc_s': sc,
                                           'logtau_m': np.interp(gc, gt, mt), 'logtau_s': np.interp(gc, gt, st)})
    out.attrs['bins'] = bins
    out.attrs['kappa'] = kappa
    out.attrs['n_genes'] = int(est.sum())
    out.attrs['library_scaled'] = library_factor is not None
    return out

def _prior_tuple(variance_prior, gene):
    """The (m_logc, s_logc, m_logtau, s_logtau, kappa) prior of one gene from
    the frame estimate_variance_priors returns, or None when no prior is in use."""
    if variance_prior is None:
        return None
    if gene not in variance_prior.index:
        raise KeyError(f'no variance prior for {gene}: estimate_variance_priors must be run on the same A_df')
    if 'kappa' not in variance_prior.attrs:
        raise ValueError('variance_prior has lost its attrs (kappa, library_scaled): pass the frame '
                         'estimate_variance_priors returned, not one reloaded from a file')
    r = variance_prior.loc[gene]
    return (float(r['prior_logc_m']), float(r['prior_logc_s']), float(r['prior_logtau_m']),
            float(r['prior_logtau_s']), float(variance_prior.attrs['kappa']))

def estimate_library_factors(A_df, Va_df, genes=None, min_informative=40,
                             n_iter=2, eps=1e-12, max_iter=400, tol=1e-7,
                             min_genes=20):
    """
    The per-library variance factor d_i of variance_model='library_scaled',
    estimated across genes from the allelic channel.

    For every gene with at least ``min_informative`` informative samples
    (Va > eps), fit (c_g, tau_g) through the origin with _fit_c_tau_vectorized;
    keep the genes whose fit is interior (c_g > 0 and tau_g > 0), because a
    clamped gene puts its whole floor into the other parameter and its
    residuals are not on a common scale; then d_i is the mean over those genes
    of a_gi^2 / (c_g v_gi + tau_g), whose expectation is d_i, normalized to
    mean 1 over samples (d and c share a scale). ``n_iter`` = 2 refits
    (c_g, tau_g) with the first-pass d in place and recomputes d once.

    ``genes`` restricts the estimation set. Pass the well-expressed genes
    (at least 100 allele-resolved reads on BrainVar): below that c is not
    identifiable. On BrainVar the choice moves any one library by at most
    11% (correlation 0.986 between the >= 100-read set and all interior genes,
    0.996 against >= 30 reads). Genes to be scanned may stay in the set: each
    is one of thousands.

    If fewer than ``min_genes`` genes have an interior fit (a data set with
    no between-sample floor, or a tiny one) every gene whose fit is not
    degenerate (c > 0 or tau > 0) is used instead, with a warning: the mean
    standardized squared residual still estimates d_i, on a scale set by the
    clamped parameter.

    Returns a Series indexed by sample, with attrs['n_genes'] the number of
    genes that entered the mean and attrs['interior_only'] whether only
    interior fits did.
    """
    import warnings
    A = np.asarray(A_df.values, dtype=float)
    V = np.asarray(Va_df.values, dtype=float)
    if genes is not None:
        rows = A_df.index.get_indexer(pd.Index(genes))
        if (rows < 0).any():
            raise ValueError(f'{int((rows < 0).sum())} of the requested genes are not in A_df')
        A, V = A[rows], V[rows]
    M = V > eps
    ok = M.sum(1) >= min_informative
    A, V, M = A[ok], V[ok], M[ok]
    if A.shape[0] == 0:
        raise ValueError('no gene has enough informative samples to estimate library factors')
    N = A.shape[1]
    d = np.ones(N)
    n_genes = 0
    for _ in range(max(1, int(n_iter))):
        c, tau = _fit_c_tau_vectorized(A ** 2 / d, V, M, max_iter=max_iter, tol=tol)
        interior = (c > 1e-6) & (tau > 1e-6)
        interior_only = True
        if interior.sum() < min_genes:
            interior_only = False
            interior = (c > 1e-6) | (tau > 1e-6)
            warnings.warn(
                f'estimate_library_factors: only {int(((c > 1e-6) & (tau > 1e-6)).sum())} genes have '
                f'an interior (c > 0, tau > 0) fit; using the {int(interior.sum())} non-degenerate '
                f'fits instead', RuntimeWarning, stacklevel=2)
        if not interior.any():
            raise ValueError('no gene has a non-degenerate fit; cannot estimate library factors')
        Mi = M[interior]
        e2 = np.where(Mi, A[interior] ** 2 / (c[interior, None] * V[interior] + tau[interior, None]), 0.0)
        n = Mi.sum(0)
        d_new = np.where(n > 0, e2.sum(0) / np.maximum(n, 1), np.nan)
        if np.isnan(d_new).any():
            raise ValueError('some samples are informative for none of the estimation genes')
        d = d_new / d_new.mean()
        n_genes = int(interior.sum())
    out = pd.Series(d, index=A_df.columns, name='library_factor')
    out.attrs['n_genes'] = n_genes
    out.attrs['interior_only'] = interior_only
    return out


def _channel_weights_estimated(y_t, v_t, covariates_t, device, eps, tau_extra_t,
                               intercept, variance_model, d_t, prior, n_inf):
    """The DEPRECATED `tau_mode='estimate'` tail of `hapmixqtl._channel_weights`.

    Fits the variance function from this gene's own squared residuals and then
    weights those same residuals by the fit -- `tau_g` alone under 'additive',
    `(c_g, tau_g)` jointly otherwise. Moved here verbatim on 2026-09-23; the
    live module has already applied the sparse-channel rule and handled
    `tau_mode='zero'` before delegating, and passes `n_inf` so the refit
    admission test is not recomputed.

    Returns the same 6-tuple as `_channel_weights`.
    """
    design, refit = covariates_t, False
    if tau_extra_t is not None:
        cand = tau_extra_t if covariates_t is None else torch.cat([covariates_t, tau_extra_t], dim=1)
        if n_inf >= _min_informative(cand, intercept=intercept):
            design, refit = cand, True
    if variance_model == 'additive':
        tau = _estimate_tau_informative(y_t, v_t, design, device, eps,
                                        intercept=intercept)
        return torch.sqrt(1.0 / (v_t.clamp(min=1e-8) + tau)), float(tau), refit, 1.0, True, None
    keep = v_t > eps
    d_keep = None if d_t is None else d_t[keep]
    fit = _estimate_c_tau(y_t[keep], v_t[keep],
                          None if design is None else design[keep],
                          device, intercept=intercept, d_t=d_keep, prior=prior)
    c, tau = fit['c'], fit['tau']
    d_all = torch.ones_like(v_t) if d_t is None else d_t
    var = d_all * (c * v_t.clamp(min=1e-8) + tau)
    return torch.sqrt(1.0 / var.clamp(min=1e-30)), float(tau), refit, float(c), fit['converged'], fit

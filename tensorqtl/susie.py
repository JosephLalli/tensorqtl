# SuSiE (sum of single effects) model
#
# References:
# [1] Wang et al., J. Royal Stat. Soc. B, 2020
#     https://rss.onlinelibrary.wiley.com/doi/full/10.1111/rssb.12388
#
# This implementation is largely based on the original R version at
# https://github.com/stephenslab/susieR

import torch
import numpy as np
import pandas as pd
from collections import defaultdict
import sys
import os
import time
from scipy.optimize import minimize_scalar

sys.path.insert(1, os.path.dirname(__file__))
import genotypeio
import susieash
from core import *
from susieslot import (
    finish_slot_sweep,
    initialize_slot_state,
    slot_prior_betabinom,
    slot_prior_elbo,
    slot_prior_poisson,
    update_slot_weight,
)


def get_x_attributes(X_t, center=True, scale=True):
    """Compute column means and SDs"""
    cm_t = X_t.mean(0)
    csd_t = X_t.std(0, unbiased=True)
    # set sd = 1 when the column has variance 0
    csd_t[csd_t == 0] = 1

    if not center:
        cm_t = torch.zeros(X_t.shape[1], dtype=X_t.dtype, device=X_t.device)
    if not scale:
        csd_t = torch.ones(X_t.shape[1], dtype=X_t.dtype, device=X_t.device)

    x_std_t = (X_t - cm_t) / csd_t
    xattr = {
        'd': (x_std_t * x_std_t).sum(0),
        'scaled_center': cm_t,
        'scaled_scale': csd_t,
    }
    return xattr


def init_setup(n, p, L, scaled_prior_variance, varY, residual_variance=None,
               prior_weights=None, null_weight=None):  # , standardize
    """
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if scaled_prior_variance < 0:
        raise ValueError('Scaled prior variance must be positive.')
    # if standardize and scaled_prior_variance > 1:
    #    raise ValueError('Scaled prior variance must be no greater than 1 when standardize = True.')
    if residual_variance is None:
        residual_variance = varY
    if prior_weights is None:
        prior_weights = torch.full([p], 1/p, dtype=torch.float32).to(device)
    else:
        # accept tensor/numpy/list and land on `device` so s['pi'] matches the
        # rest of the internal state (which is allocated on `device` below).
        prior_weights = torch.as_tensor(prior_weights, dtype=torch.float32, device=device)
        prior_weights = prior_weights / prior_weights.sum()
    if len(prior_weights) != p:
        raise ValueError('Prior weights must have length p.')
    if (p < L):
        L = p

    s = {
        'alpha': torch.full((L,p), 1/p).to(device),
        'mu': torch.zeros((L,p)).to(device),
        'mu2': torch.zeros((L,p)).to(device),
        'Xr': torch.zeros(n).to(device),
        'KL': torch.full([L], np.nan).to(device),
        'lbf': torch.full([L], np.nan).to(device),
        'lbf_variable': torch.full([L, p], np.nan).to(device),
        'sigma2': residual_variance,
        'V': scaled_prior_variance * varY,
        'pi': prior_weights,
    }
    if null_weight is None:
        s['null_index'] = 0
    else:
        s['null_index'] = p

    return s


def init_finalize(s, X_t=None, Xr_t=None):
    """
    Update a susie fit object in order to initialize susie model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if s['V'].ndim == 0:
        # s['V'] = np.tile(s['V'], s['alpha'].shape[0])
        s['V'] = torch.full([s['alpha'].shape[0]], s['V']).to(device)

    if s['sigma2'] <= 0:
        raise ValueError("residual variance 'sigma2' must be positive (is var(Y) zero?)")

    if not (s['V'] >= 0).all():
        raise ValueError("prior variance must be non-negative")

    if Xr_t is not None:
        s['Xr'] = Xr_t
    if X_t is not None:
        raise NotImplementedError()
        # s['Xr'] = compute_Xb(X_t, colSums(s$mu*s$alpha))

    # reset KL and lbf
    s['KL'] =  torch.full([s['alpha'].shape[0]], np.nan).to(device)
    s['lbf'] = torch.full([s['alpha'].shape[0]], np.nan).to(device)

    return s


def compute_Xb(X_t, b_t, cm_t, csd_t):
    """Compute Xb with column standardized X"""
    # scale Xb
    scaled_Xb_t = torch.mm(X_t, (b_t/csd_t).reshape(-1,1)).squeeze()
    # center Xb
    Xb_t = scaled_Xb_t - (cm_t*b_t/csd_t).sum()
    return Xb_t


def compute_Xty(X_t, y_t, cm_t, csd_t):
    """
    cm: column means of X
    csd: column SDs of X
    """
    ytX_t = torch.mm(y_t.T, X_t)
    # scale Xty
    scaled_Xty_t = ytX_t.T / csd_t.reshape(-1,1)
    # center Xty
    centered_scaled_Xty_t = scaled_Xty_t - cm_t.reshape(-1,1)/csd_t.reshape(-1,1) * y_t.sum()
    return centered_scaled_Xty_t.squeeze()


def compute_MXt(M_t, X_t, xattr):
    """
    Compute M * cstd(X).T, where cstd() means col-standardized
    M: L x p matrix
    X: n x p matrix
    """
    return torch.mm(M_t, (X_t / xattr['scaled_scale']).T) - torch.mm(M_t, (xattr['scaled_center']/xattr['scaled_scale']).reshape(-1,1))


def loglik(V, betahat, shat2, prior_weights):

    # log(bf) on each SNP
    lbf = torch.distributions.Normal(0, torch.sqrt(V+shat2)).log_prob(betahat) - torch.distributions.Normal(0, torch.sqrt(shat2)).log_prob(betahat)
    lbf[torch.isinf(shat2)] = 0 # deal with special case of infinite shat2 (eg happens if X does not vary)

    maxlbf = lbf.max()
    # w = np.exp(lbf - maxlbf)  # w =BF/BFmax
    # w_weighted = w * prior_weights
    # weighted_sum_w = np.sum(w_weighted)
    # return np.log(weighted_sum_w) + maxlbf
    return torch.log((torch.exp(lbf - maxlbf) * prior_weights).sum()) + maxlbf


def neg_loglik_logscale(lV, betahat, shat2, prior_weights):
    return -loglik(torch.exp(lV), betahat, shat2, prior_weights)


def optimize_prior_variance_brent(V_init, betahat, shat2, prior_weights,
                                  check_null_threshold=0):
    """Optimize scalar SER prior variance on susieR's log-variance bounds."""
    device = betahat.device
    dtype = betahat.dtype
    betahat64 = betahat.detach().to(torch.float64)
    shat264 = shat2.detach().to(torch.float64)
    weights64 = prior_weights.detach().to(torch.float64)

    def objective(log_variance):
        value = neg_loglik_logscale(
            torch.as_tensor(log_variance, dtype=torch.float64, device=device),
            betahat64, shat264, weights64,
        )
        return float(value.detach().cpu())

    result = minimize_scalar(
        objective,
        bounds=(-30.0, 15.0),
        method='bounded',
        options={'xatol': 1e-8},
    )
    current = float(V_init)
    candidate = float(np.exp(result.x)) if result.success else current
    current_log = -np.inf if current == 0 else float(np.log(current))
    if objective(result.x) > objective(current_log):
        candidate = current

    V = torch.as_tensor(candidate, dtype=dtype, device=device)
    if (
        float(loglik(0, betahat, shat2, prior_weights))
        + check_null_threshold
        >= float(loglik(V, betahat, shat2, prior_weights))
    ):
        V = torch.zeros((), dtype=dtype, device=device)
    return V


def optimize_prior_variance(optimize_V, betahat, shat2, prior_weights,
                            alpha=None, post_mean2=None, V_init=None,
                            check_null_threshold=0):
    """"""
    # EM solution
    V = (alpha * post_mean2).sum()

    # set V exactly 0 if that beats the numerical value
    # by check_null_threshold in loglik.
    # check_null_threshold = 0.1 is exp(0.1) = 1.1 on likelihood scale;
    # it means that for parsimony reasons we set estimate of V to zero, if its
    # numerical estimate is only "negligibly" different from zero. We use a likelihood
    # ratio of exp(check_null_threshold) to define "negligible" in this context.
    # This is fairly modest condition compared to, say, a formal LRT with p-value 0.05.
    # But the idea is to be lenient to non-zeros estimates unless they are indeed small enough
    # to be neglible.
    # See more intuition at https://stephens999.github.io/fiveMinuteStats/LR_and_BF.html
    if loglik(0, betahat, shat2, prior_weights) + check_null_threshold >= loglik(V, betahat, shat2, prior_weights):
        V = 0
    return V


def SER_posterior_e_loglik(X_t, xattr, Y_t, s2, Eb, Eb2):
    n = X_t.shape[0]
    return -0.5*n*torch.log(2*np.pi*s2) - (0.5/s2) * ((Y_t*Y_t).sum() - 2*(Y_t.squeeze()*compute_Xb(X_t, Eb, xattr['scaled_center'], xattr['scaled_scale'])).sum() + (xattr['d']*Eb2).sum())


def single_effect_regression(Y_t, X_t, xattr, V, residual_variance=1, prior_weights=None,
                             optimize_V='EM', check_null_threshold=0):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if optimize_V not in {'none', 'optim', 'EM', 'simple'}:
        raise ValueError(
            "estimate_prior_method must be 'none', 'optim', 'EM', or 'simple'."
        )

    Xty = compute_Xty(X_t, Y_t, xattr['scaled_center'], xattr['scaled_scale'])
    betahat = (1/xattr['d']) * Xty

    shat2 = residual_variance / xattr['d']
    if prior_weights is None:
        prior_weights = torch.full(
            [X_t.shape[1]], 1 / X_t.shape[1],
            dtype=X_t.dtype, device=X_t.device,
        )

    if optimize_V == 'optim':
        V = optimize_prior_variance_brent(
            V, betahat, shat2, prior_weights,
            check_null_threshold=check_null_threshold,
        )
    elif optimize_V == 'simple':
        if (
            float(loglik(0, betahat, shat2, prior_weights))
            + check_null_threshold
            >= float(loglik(V, betahat, shat2, prior_weights))
        ):
            V = torch.zeros((), dtype=betahat.dtype, device=betahat.device)

    # lbf = stats.norm.logpdf(betahat, 0, np.sqrt(V+shat2)) - stats.norm.logpdf(betahat, 0, np.sqrt(shat2))
    lbf = torch.distributions.Normal(0, torch.sqrt(V+shat2)).log_prob(betahat) - torch.distributions.Normal(0, torch.sqrt(shat2)).log_prob(betahat)

    # log(bf) on each SNP
    lbf[torch.isinf(shat2)] = 0  # deal with special case of infinite shat2 (eg happens if X does not vary)
    maxlbf = lbf.max()
    w = torch.exp(lbf - maxlbf)  # w is proportional to BF, but subtract max for numerical stability
    # posterior prob on each SNP
    w_weighted = w * prior_weights
    weighted_sum_w = w_weighted.sum()
    alpha = w_weighted / weighted_sum_w
    if V == 0:
        post_var = torch.zeros(xattr['d'].shape).to(device)
    else:
        post_var = (1/V + xattr['d']/residual_variance)**(-1)  # posterior variance
    # print("V: {}  {}".format(V, post_var[0]))   ############
    try:
        post_mean = (1/residual_variance) * post_var * Xty
    except:
        print(residual_variance.device)
        print(post_var.device)
        print(post_var)
        print(Xty.device)

    post_mean2 = post_var + post_mean**2  # second moment
    # BF for single effect model
    lbf_model = maxlbf + torch.log(weighted_sum_w)
    # loglik = lbf_model + np.sum(stats.norm.logpdf(Y_t, 0, np.sqrt(residual_variance)))
    loglik = lbf_model + torch.distributions.Normal(0, torch.sqrt(residual_variance)).log_prob(Y_t).sum()

    if optimize_V == 'EM':
        V = optimize_prior_variance(optimize_V, betahat, shat2, prior_weights, alpha,
                                    post_mean2, check_null_threshold=check_null_threshold)

    return {
        'alpha': alpha,
        'mu': post_mean,
        'mu2': post_mean2,
        'lbf': lbf,
        'lbf_model': lbf_model,
        'V': V,
        'loglik': loglik,
    }


def update_each_effect(X_t, xattr, Y_t, s, estimate_prior_variance=False,
                       estimate_prior_method='EM', check_null_threshold=0):
    """

    """
    if not estimate_prior_variance:
        estimate_prior_method = 'none'

    # Repeat for each effect to update
    L = s['alpha'].shape[0]
    use_c_hat = 'c_hat_state' in s

    for l in range(L):
        if (
            use_c_hat
            and float(s['slot_weights'][l]) < s['c_hat_state']['skip_threshold']
        ):
            continue

        slot_weight = float(s['slot_weights'][l]) if use_c_hat else 1.0

        # remove lth effect from fitted values
        s['Xr'] = s['Xr'] - slot_weight * compute_Xb(
            X_t, s['alpha'][l, :] * s['mu'][l, :],
            xattr['scaled_center'], xattr['scaled_scale']
        )

        # compute residuals
        R_t = Y_t - s['Xr'].reshape(-1,1)

        res = single_effect_regression(R_t, X_t, xattr, s['V'][l],
                                       residual_variance=s['sigma2'], prior_weights=s['pi'],
                                       optimize_V=estimate_prior_method,
                                       check_null_threshold=check_null_threshold)

        # update the variational estimate of the posterior mean
        s['mu'][l] = res['mu']
        s['alpha'][l] = res['alpha']
        s['mu2'][l] = res['mu2']
        s['V'][l] = res['V']
        s['lbf'][l] = res['lbf_model']
        s['lbf_variable'][l] = res['lbf']
        s['KL'][l] = -res['loglik'] + SER_posterior_e_loglik(X_t, xattr, R_t, s['sigma2'], res['alpha']*res['mu'], res['alpha']*res['mu2'])
        effect_fitted = compute_Xb(
            X_t, s['alpha'][l, :] * s['mu'][l, :],
            xattr['scaled_center'], xattr['scaled_scale']
        )
        s['Xr'] = s['Xr'] + slot_weight * effect_fitted

        if use_c_hat:
            old_c, new_c = update_slot_weight(s, l)
            if abs(new_c - old_c) > 1e-15:
                s['Xr'] = s['Xr'] + (new_c - old_c) * effect_fitted

    if use_c_hat:
        finish_slot_sweep(s)
    return(s)


def get_objective(X_t, xattr, Y_t, s, er2=None):
    """Get objective function from data and susie fit object."""
    objective = eloglik(X_t, xattr, Y_t, s, er2=er2) - (s['KL']).sum()
    if 'c_hat_state' in s:
        objective = objective + slot_prior_elbo(s)
    return objective


def eloglik(X_t, xattr, Y_t, s, er2=None):
    """Expected log-likelihood for a SuSiE fit."""
    n = X_t.shape[0]
    if er2 is None:
        er2 = get_ER2(X_t, xattr, Y_t, s)
    return -(n/2) * torch.log(2*np.pi*s['sigma2']) - (1/(2*s['sigma2'])) * er2


def get_ER2(X_t, xattr, Y_t, s):
    """expected squared residuals
      Xr_L is L by N matrix
      s['Xr'] is column sum of Xr_L
    """
    Xr_L = compute_MXt(s['alpha']*s['mu'], X_t, xattr)
    postb2 = s['alpha'] * s['mu2']  # posterior second moment
    if 'slot_weights' not in s:
        return ((Y_t.squeeze()-s['Xr'])**2).sum() - (Xr_L**2).sum() + (xattr['d'].reshape(-1,1) * postb2.T).sum()

    slot_weights = s['slot_weights']
    per_slot_Eb2 = torch.matmul(postb2, xattr['d'])
    per_slot_Xb2 = (Xr_L**2).sum(1)
    return (
        ((Y_t.squeeze() - s['Xr'])**2).sum()
        + (slot_weights * per_slot_Eb2
           - slot_weights**2 * per_slot_Xb2).sum()
    )


def estimate_residual_variance_fct(X_t, xattr, Y_t, s, er2=None):
    n = X_t.shape[0]
    if er2 is None:
        er2 = get_ER2(X_t, xattr, Y_t, s)
    return (1/n) * er2


def susie_get_pip(res, prune_by_cs=False, prior_tol=1e-9):
    """
    Compute posterior inclusion probability (PIP) for all variables

      res:  a susie fit, the output of susie(), or simply the posterior inclusion probability matrix alpha
      prune_by_cs:  whether or not to ignore single effects not in reported CS when calculating PIP
      prior_tol:  filter out effects having estimated prior variance smaller than this threshold

    Returns:
      array of posterior inclusion probabilities
    """
    alpha = res['alpha']

    # drop null weight columns
    if res['null_index'] > 0:
        keep = torch.arange(alpha.shape[1], device=alpha.device) != res['null_index']
        alpha = alpha[:, keep]

    # drop the single effect with estimated prior zero
    include_idx = torch.where(res['V'] > prior_tol)[0]

    # only consider variables in reported CS
    # this is not what we do in the SuSiE paper
    # so by default prune_by_cs = FALSE means we do not run the following code
    if prune_by_cs:  # TODO: not tested
        raise NotImplementedError()
        # if 'sets' in res and 'cs_index' in res['sets']:
        #     include_idx = np.intersect1d(include_idx, res['sets']['cs_index'])
        # else:
        #     include_idx = np.array([0])

    # now extract relevant rows from alpha matrix
    if len(include_idx) > 0:
        alpha = alpha[include_idx]
        slot_weights = res.get('slot_weights')
        if slot_weights is not None:
            alpha = alpha * slot_weights[include_idx, None]
    else:
        alpha = torch.zeros(
            [1, alpha.shape[1]], dtype=alpha.dtype, device=alpha.device
        )

    return 1 - (1 - alpha).prod(0)


def _weighted_sparse_effect(s):
    """Posterior mean sparse effect on the standardized-X scale."""
    weights = s.get('slot_weights')
    effects = s['alpha'] * s['mu']
    if weights is not None:
        effects = effects * weights[:, None]
    return effects.sum(0)


def _recompute_weighted_fitted(X_t, xattr, s):
    s['Xr'] = compute_Xb(
        X_t, _weighted_sparse_effect(s),
        xattr['scaled_center'], xattr['scaled_scale']
    )


def _pip_state_converged(s, history, tol, cycle_window):
    """Check the upstream alpha/PIP fixed point and short-cycle criterion."""
    current_alpha = s['alpha'].detach().clone()
    current_pip = susie_get_pip(s).detach().clone()
    max_lag = min(max(1, int(cycle_window)), len(history))

    for lag in range(1, max_lag + 1):
        old_alpha, old_pip = history[-lag]
        state_diff = max(
            float((current_alpha - old_alpha).abs().max()),
            float((current_pip - old_pip).abs().max()),
        )
        if state_diff < tol:
            return True, state_diff, lag
    return False, state_diff, None


def in_CS(res, coverage=0.9):
    """
    returns an l by p binary matrix
    indicating which variables are in susie credible sets
    """
    o = torch.flip(res['alpha'].argsort(), [1])  # sorts each row
    n = (torch.cumsum(torch.gather(res['alpha'], 1, o), 1) < coverage).sum(1) + 1
    result = torch.zeros(res['alpha'].shape, dtype=torch.bool)
    for i in range(result.shape[0]):
        result[i, o[i][:n[i]]] = True
    return result


def cov(X_t):
    X0_t = X_t - X_t.mean(1, keepdim=True)
    return torch.mm(X0_t, X0_t.T) / (X_t.shape[1] - 1)


def corrcoef(X_t):
    c = cov(X_t)
    sd = torch.sqrt(torch.diag(c))
    c /= sd[:, None]
    c /= sd[None, :]
    return torch.clamp(c, -1, 1, out=c)


def get_purity(pos, X, Xcorr, squared=False, n=100):
    """Deterministically subsample and compute min, mean and median correlation."""
    if len(pos) == 1:
        return np.ones(3)
    else:
        if len(pos) > n:
            if torch.is_tensor(pos):
                generator = torch.Generator(device=pos.device)
                generator.manual_seed(1)
                pos = pos[torch.randperm(len(pos), generator=generator, device=pos.device)[:n]]
            else:
                pos = np.random.default_rng(1).choice(pos, n, replace=False)
        if Xcorr is None:
            X_sub = X[:, pos]
            if len(pos) > n:  # remove columns with identical values
                pos_rm = (X_sub - X_sub.mean(0) < torch.finfo(torch.float64).eps**0.5).abs().all(0)
                if any(pos_rm):
                    X_sub = X_sub[:, ~pos_rm]
            value = corrcoef(X_sub.T).abs()
        else:
            value = (Xcorr[pos][:, pos]).abs()
        if squared:
            value = value**2
        # return np.nanmin(value), np.nanmean(value), np.nanmedian(value)
        return float(value.min()), float(value.mean()), float(value.median())


def _abs_corr_members_to_all(members, X=None, Xcorr=None):
    """Absolute correlation of each CS member against all p variants -> (len(members), p).

    With a precomputed Xcorr, index directly. From individual-level X (n x p) we
    standardize columns (Pearson correlation is shift/scale invariant, so the raw
    genotype columns give the same answer) and take z[:,members].T @ z / (n-1).
    """
    if Xcorr is not None:
        return Xcorr[members].abs().clamp(max=1.0)
    n = X.shape[0]
    Xc = X - X.mean(0)
    sd = torch.sqrt((Xc*Xc).sum(0) / (n - 1))
    sd[sd == 0] = 1
    z = Xc / sd
    corr = (z[:, members].T @ z) / (n - 1)
    return corr.abs().clamp(max=1.0)


def extend_cs_by_correlation(cs, threshold, null_index, X=None, Xcorr=None):
    """susieR-2.0 `cs_extension_corr`: absorb into each CS every variant whose
    |corr| to ANY current member exceeds `threshold` (recommended 0.99). Runs
    before purity, so it changes CS membership and the reported purity numbers.
    Off by default upstream; only called when cs_extension_corr is set."""
    if len(cs) == 0:
        return cs
    device = cs[0].device
    extended = []
    for members in cs:
        corr_rows = _abs_corr_members_to_all(members, X=X, Xcorr=Xcorr)  # (m, p)
        in_tight = torch.where((corr_rows > threshold).any(0))[0].to(device)
        if null_index > 0:
            in_tight = in_tight[in_tight != null_index]
        extended.append(torch.unique(torch.cat([members, in_tight])))  # sorted, unique
    return extended


def susie_get_cs(res, X=None, Xcorr=None, coverage=0.95, min_abs_corr=0.5,
                 median_abs_corr=None, cs_extension_corr=None,
                 dedup=True, squared=False):
    """Extract credible sets.

    susieR-2.0 additions (both default-off, so the default call is unchanged):
      median_abs_corr:   keep a CS if min|corr| >= min_abs_corr OR
                         median|corr| >= median_abs_corr (OR-linked, so it can
                         only ADMIT extra CSs whose bulk is tight but whose
                         minimum is dragged down by one weak member).
      cs_extension_corr: before purity, absorb near-perfect proxies (|corr| >
                         threshold to a member) into each CS.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if X is not None and Xcorr is not None:
        raise ValueError('Only one of X or Xcorr should be specified.')
    for name, value in [('min_abs_corr', min_abs_corr),
                        ('median_abs_corr', median_abs_corr),
                        ('cs_extension_corr', cs_extension_corr)]:
        if value is not None and not 0 <= value <= 1:
            raise ValueError(f'{name} must be between 0 and 1.')
    # if Xcorr is not None and not is_symmetric_matrix(Xcorr):
    #     raise ValueError('Xcorr matrix must be symmetric.')

    null_index = res['null_index']
    include_mask = res['V'] > 1e-9

    # L by P bool matrix
    status = in_CS(res, coverage=coverage)

    # an L list of CS positions
    cs = [torch.where(i)[0] for i in status]
    include_mask = include_mask & torch.BoolTensor([len(i) > 0 for i in cs]).to(device)
    # FIXME: see issue 21
    # https://github.com/stephenslab/susieR/issues/21
    if dedup:
        duplicated = torch.ones(status.shape[0], dtype=bool).to(device)
        _,ix = status.unique(dim=0, return_inverse=True)
        duplicated[ix.unique()] = False
        include_mask = include_mask & ~duplicated

    if not any(include_mask):
        return {'cs':None, 'coverage':coverage}

    # compute and filter by "purity"
    if Xcorr is None and X is None:
        cs_dict = {f'L{k+1}':cs[k] for k,i in enumerate(include_mask) if i}
        return {'cs':cs_dict, 'coverage':coverage}
    else:
        cs = [cs[k] for k,i in enumerate(include_mask) if i]

        # susieR-2.0 cs_extension_corr: absorb near-perfect proxies before purity
        if cs_extension_corr is not None:
            cs = extend_cs_by_correlation(cs, cs_extension_corr, null_index, X=X, Xcorr=Xcorr)

        purity = []
        for i in range(len(cs)):
            if null_index > 0 and null_index in cs[i]:
                purity.append([-9, -9, -9])
            else:
                purity.append(get_purity(cs[i], X, Xcorr, squared=squared))
        if squared:
            cols = ['min_sq_corr', 'mean_sq_corr', 'median_sq_corr']
        else:
            cols = ['min_abs_corr', 'mean_abs_corr', 'median_abs_corr']
        purity = pd.DataFrame(purity, columns=cols)

        # susieR-2.0: keep a CS if it passes the min OR the median criterion.
        # Default (min_abs_corr=0.5, median_abs_corr=None) reduces to the pre-2.0
        # min-only filter. Both None -> keep every non-null CS (null CS has -9).
        if min_abs_corr is None and median_abs_corr is None:
            keep = purity.values[:, 0] > -1
        else:
            keep = np.zeros(len(purity), dtype=bool)
            if min_abs_corr is not None:
                thr = min_abs_corr**2 if squared else min_abs_corr
                keep = keep | (purity.values[:, 0] >= thr)
            if median_abs_corr is not None:
                thr = median_abs_corr**2 if squared else median_abs_corr
                keep = keep | (purity.values[:, 2] >= thr)
        is_pure = np.where(keep)[0]
        if len(is_pure) > 0:
            include_idx = torch.where(include_mask)[0]
            cs = [cs[k] for k in is_pure]

            # subset by purity
            purity = purity.iloc[is_pure]
            rownames = [f'L{i+1}' for i in include_idx[is_pure]]
            purity.index = rownames

            # re-order CS list and purity rows based on purity
            ordering = purity.values[:,0].argsort()[::-1]
            return {'cs': {rownames[i]:cs[i].numpy() for i in ordering},
                    'purity': purity.iloc[ordering],
                    'cs_index': include_idx[is_pure[ordering]].cpu().numpy(),
                    'coverage': coverage}
        else:
            return {'cs':None, 'coverage':coverage}


def susie(X_t, y_t, L=10, scaled_prior_variance=0.2,
          residual_variance=None, prior_weights=None, null_weight=None,
          standardize=True, intercept=True,
          estimate_residual_variance=True, estimate_prior_variance=True,
          estimate_prior_method=None,
          check_null_threshold=0, prior_tol=1e-9,
          residual_variance_upperbound=np.inf,
          slot_prior=None,
          unmappable_effects=None,
          coverage=0.95, min_abs_corr=0.5,
          median_abs_corr=None, cs_extension_corr=None,
          compute_univariate_zscore=False,
          na_rm=False, max_iter=100, tol=0.001,
          convergence_method='elbo', pip_stall_window=5,
          verbose=False, track_fit=False):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Normalize caller-provided inputs onto the compute device. susie() allocates
    # all of its internal state (alpha/mu/Xr/...) on `device`, so inputs built on a
    # different device (e.g. CPU tensors handed in directly) must be moved here or
    # they collide with that state in the first matmul (compute_Xb). map_* callers
    # already build on `device`, making this a no-op for them.
    X_t = X_t.to(device)
    y_t = y_t.to(device)

    n, p = X_t.shape
    mean_y = y_t.mean()

    if convergence_method not in {'elbo', 'pip'}:
        raise ValueError("convergence_method must be 'elbo' or 'pip'.")
    if not isinstance(pip_stall_window, (int, np.integer)) or pip_stall_window < 1:
        raise ValueError('pip_stall_window must be a positive integer.')
    if unmappable_effects not in {None, 'ash'}:
        raise ValueError("unmappable_effects must be None or 'ash'.")

    # SuSiE-ash (Mr.ASH polygenic background). Between IBSS iterations a Mr.ASH
    # fit absorbs a diffuse polygenic background theta on residuals with confident
    # credible-set variants masked; the SER then sees residuals net of X@theta.
    # Convergence is PIP-based (susieR forces convergence_method="pip" for
    # unmappable-effects models, which have no well-defined ELBO).
    use_ash = unmappable_effects == 'ash'
    if use_ash and not estimate_residual_variance:
        # susieR gates the in-loop refit on estimate_residual_variance=TRUE
        # (update_model_variance early-returns otherwise); the shared sigma2 is
        # driven by Mr.ASH. Enforce it so theta is actually fit inside the loop.
        raise ValueError("unmappable_effects='ash' requires estimate_residual_variance=True.")
    if estimate_prior_method is None:
        # Keep TensorQTL's historical EM default on the ordinary path, while
        # matching susieR's optimizer default for SuSiE-ash.
        estimate_prior_method = 'optim' if use_ash else 'EM'
    if estimate_prior_method not in {'none', 'optim', 'EM', 'simple'}:
        raise ValueError(
            "estimate_prior_method must be 'none', 'optim', 'EM', or 'simple'."
        )
    if use_ash and estimate_prior_method == 'EM':
        raise ValueError(
            "unmappable_effects='ash' does not support "
            "estimate_prior_method='EM'; use 'optim'."
        )
    if use_ash and not intercept:
        raise ValueError("unmappable_effects='ash' requires intercept=True.")
    if use_ash and not standardize:
        raise ValueError("unmappable_effects='ash' requires standardize=True.")
    if use_ash and slot_prior is None:
        # The activity prior is required to identify sparse slots separately
        # from the dense Mr.ASH background.
        slot_prior = slot_prior_betabinom()
    if slot_prior is not None or use_ash:
        # The ordinary SuSiE ELBO does not define a valid stopping rule once
        # fitted effects are marginalized by c_hat or theta changes residuals.
        convergence_method = 'pip'

    if intercept:
        y_t = y_t - mean_y

    xattr = get_x_attributes(X_t, center=intercept, scale=standardize)

    if use_ash:
        # Materialize the standardized design x_std = (X - cm)/csd; the SER effects
        # (mu/alpha) live in these units and compute_Xb(X_t, b) == x_std @ b, so
        # running Mr.ASH here (intercept=False) puts theta in the same units and
        # X_theta subtracts consistently. mr.ash is CPU/numpy, so keep a float64
        # host copy for refits. w = colSums(x_std^2) = xattr['d']; the default sa2
        # grid matches susieR (median(w), n).
        x_std_t = (X_t - xattr['scaled_center']) / xattr['scaled_scale']
        x_std_np = x_std_t.detach().cpu().numpy().astype(np.float64)
        d_t = xattr['d']
        sa2_np = susieash.default_sa2_grid(
            d_t.detach().cpu().numpy().astype(np.float64), n
        )
        Xcorr_t = susieash.variant_corr(x_std_t, d_t)
        theta_t = torch.zeros(p, dtype=X_t.dtype, device=device)
        X_theta_t = torch.zeros(n, dtype=X_t.dtype, device=device)
        ash_pi = None
        ash_tau2 = 0.0
        ash_state = susieash.initialize_state(
            min(L, p), p, device
        )
        ash_policy_history = []

    # initialize susie fit
    s = init_setup(n, p, L, scaled_prior_variance, y_t.var(unbiased=True),
                   residual_variance=residual_variance,
                   prior_weights=prior_weights, null_weight=null_weight)
    s = init_finalize(s)
    if slot_prior is not None:
        s['slot_weights'], s['c_hat_state'] = initialize_slot_state(
            slot_prior, s['alpha'].shape[0], s['alpha'].dtype, s['alpha'].device
        )
        _recompute_weighted_fitted(X_t, xattr, s)

    # initialize elbo to NA
    elbo = torch.full([max_iter + 1], np.nan).to(device)
    elbo[0] = -np.inf;
    tracking = []
    state_history = [(
        s['alpha'].detach().clone(),
        susie_get_pip(s, prior_tol=prior_tol).detach().clone(),
    )]
    for i in range(1, max_iter+1):

        # SuSiE-ash: the SER regresses on residuals net of the polygenic
        # background X@theta (theta is 0 on iter 1). Xr stays sparse-only, so
        # subtracting X_theta from the target is the only ash-specific change.
        y_eff = (y_t - X_theta_t.reshape(-1, 1)) if use_ash else y_t

        s = update_each_effect(X_t, xattr, y_eff, s,
                               estimate_prior_variance=estimate_prior_variance,
                               estimate_prior_method=estimate_prior_method,
                               check_null_threshold=(
                                   0 if use_ash else check_null_threshold
                               ))
        # Both calculations use the same expected residual sum of squares.
        objective_y = y_eff if use_ash else y_t
        er2 = get_ER2(X_t, xattr, objective_y, s)
        elbo[i] = get_objective(
            X_t, xattr, objective_y, s, er2=er2
        )
        if verbose:
            print(f'Objective (iter {i}): {elbo[i]}')
        if convergence_method == 'pip':
            converged, state_diff, cycle_lag = _pip_state_converged(
                s, state_history, tol, pip_stall_window
            )
            if verbose:
                print(f'max|d(alpha,PIP)| (iter {i}): {state_diff}')
            if i > 1 and converged:
                s['converged'] = True
                s['convergence_reason'] = (
                    'alpha_pip_fixed_point' if cycle_lag == 1
                    else f'alpha_pip_cycle_{cycle_lag}'
                )
                break
            state_history.append((
                s['alpha'].detach().clone(),
                susie_get_pip(s, prior_tol=prior_tol).detach().clone(),
            ))
            if len(state_history) > pip_stall_window:
                state_history.pop(0)
        else:
            elbo_diff = elbo[i] - elbo[i-1]
            if elbo_diff >= 0 and elbo_diff < tol:
                s['converged'] = True
                break

        if use_ash:
            # Pinned susieR updates ash state only after the sparse sweep has
            # failed its convergence check. The resulting theta is consumed by
            # the next sparse sweep.
            policy = susieash.update_policy(
                s['alpha'], s['mu'], Xcorr_t, s['slot_weights'], ash_state
            )
            mask_t = policy['mask']
            b_conf_fit = compute_Xb(
                X_t, policy['b_confident'],
                xattr['scaled_center'], xattr['scaled_scale'],
            )
            target_np = (
                y_t.squeeze() - b_conf_fit
            ).detach().cpu().numpy().astype(np.float64)
            mask_np = mask_t.detach().cpu().numpy()
            beta_init_np = (
                theta_t.detach().cpu().numpy().astype(np.float64)
            )
            beta_init_np[mask_np] = 0.0
            convtol_ash = (
                1e-3 if ash_state['ash_iter'] < 2 else 1e-4
            )
            theta_np, sig, ash_pi, ash_tau2 = susieash.refit(
                x_std_np, target_np, float(s['sigma2']),
                beta_init_np, ash_pi, sa2_np, convtol_ash, True,
            )
            theta_np[mask_np] = 0.0
            s['sigma2'] = torch.as_tensor(
                min(sig, float(residual_variance_upperbound)),
                dtype=X_t.dtype, device=device,
            )
            theta_t = torch.as_tensor(
                theta_np, dtype=X_t.dtype, device=device
            )
            X_theta_t = compute_Xb(
                X_t, theta_t,
                xattr['scaled_center'], xattr['scaled_scale'],
            )
            if track_fit:
                ash_policy_history.append({
                    key: (
                        value.detach().cpu().clone()
                        if torch.is_tensor(value) else value
                    )
                    for key, value in policy.items()
                    if key != 'state'
                })
            continue

        if estimate_residual_variance:
            s['sigma2'] = estimate_residual_variance_fct(
                X_t, xattr, y_t, s, er2=er2
            )
            if s['sigma2'] > residual_variance_upperbound:
                s['sigma2'] = residual_variance_upperbound
            if verbose:
                print(f'Objective (iter {i}): {get_objective(X_t, xattr, y_t, s)}')

    s['elbo'] = elbo[1:i+1].cpu().numpy()  # Remove first (infinite) entry, and trailing NAs.
    s['niter'] = i

    if 'converged' not in s:
        print(f"\n    WARNING: IBSS algorithm did not converge in {max_iter} iterations!")
        s['converged'] = False

    sparse_effect_std = _weighted_sparse_effect(s)
    sparse_effect_raw = sparse_effect_std / xattr['scaled_scale']
    s['sparse_effects'] = sparse_effect_raw
    ash_final_pass = False
    if use_ash and s['converged']:
        # Final unmasked pass: unlike the pinned source's unweighted subtraction,
        # use the fitted model's c_hat-weighted sparse mean. This preserves the
        # identity between the residual fitted here and the reported predictor.
        b_all_fit = compute_Xb(
            X_t, sparse_effect_std,
            xattr['scaled_center'], xattr['scaled_scale'],
        )
        target_np = (y_t.squeeze() - b_all_fit).detach().cpu().numpy().astype(np.float64)
        beta_init_np = theta_t.detach().cpu().numpy().astype(np.float64)
        theta_np, sig, ash_pi, ash_tau2 = susieash.refit(
            x_std_np, target_np, float(s['sigma2']), beta_init_np, ash_pi,
            sa2_np, 1e-4, estimate_residual_variance)
        s['sigma2'] = torch.as_tensor(min(sig, float(residual_variance_upperbound)),
                                      dtype=X_t.dtype, device=device)
        theta_t = torch.as_tensor(theta_np, dtype=X_t.dtype, device=device)
        X_theta_t = compute_Xb(X_t, theta_t, xattr['scaled_center'], xattr['scaled_scale'])
        ash_final_pass = True

    if use_ash:
        s['theta'] = theta_t.detach().cpu().numpy()
        s['ash_pi'] = ash_pi
        s['tau2'] = ash_tau2
        s['ash_final_pass'] = ash_final_pass
        for key, value in ash_state.items():
            s[key] = (
                value.detach().cpu().numpy()
                if torch.is_tensor(value) else value
            )
        if track_fit:
            s['ash_policy_history'] = ash_policy_history

    total_effect_raw = sparse_effect_raw
    if use_ash:
        total_effect_raw = total_effect_raw + theta_t / xattr['scaled_scale']
        s['theta_raw'] = theta_t / xattr['scaled_scale']

    if intercept:
        s['intercept'] = mean_y - (
            xattr['scaled_center'] * total_effect_raw
        ).sum()
        s['fitted'] = s['Xr'] + mean_y
    else:
        s['intercept'] = 0
        s['fitted'] = s['Xr']

    if use_ash:
        # fitted = sparse Xr + polygenic X_theta + intercept (get_fitted, IDM:480-488)
        s['fitted'] = s['fitted'] + X_theta_t

    s['fitted'] = s['fitted'].squeeze()
    if track_fit:
        s['trace'] = tracking

    s['lbf_variable'] = s['lbf_variable'].cpu().numpy()

    if slot_prior is not None:
        s['c_hat'] = s['slot_weights'].detach().cpu().numpy()
        s['C_hat'] = float(s['slot_weights'].sum())
        if s['c_hat_state']['prior_type'] == 'betabinom':
            s['a_beta'] = s['c_hat_state']['a_beta']
            s['b_beta'] = s['c_hat_state']['b_beta']
        else:
            s['a_g'] = s['c_hat_state']['a_g']
            s['b_g'] = s['c_hat_state']['b_g']

    # SuSiE CS and PIP
    if coverage is not None:
        s['sets'] = susie_get_cs(s, coverage=coverage, X=X_t, min_abs_corr=min_abs_corr,
                                 median_abs_corr=median_abs_corr, cs_extension_corr=cs_extension_corr)
    s['pip'] = susie_get_pip(
        s, prune_by_cs=False, prior_tol=prior_tol
    ).cpu().numpy()

    return s


def map_loci(locus_df, genotype_df, variant_df, phenotype_df, covariates_df, **kwargs):
    """
    Run fine-mapping on phenotype-locus pairs defined in locus_df.

    Parameters
    ----------
    locus_df : pd.DataFrame
        DataFrame with columns ['phenotype_id', 'chr', 'start', 'end'] or
        ['phenotype_id', 'chr', 'position'] where chr and pos define the
        center of each locus to fine-map (±window)
    genotype_df : pd.DataFrame
        Genotypes (variants x samples)
    variant_df : pd.DataFrame
        Mapping of variant_id (index) to ['chrom', 'pos']
    phenotype_df : pd.DataFrame
        Phenotypes (phenotypes x samples)
    covariates_df : pd.DataFrame
        Covariates (samples x covariates)

    See map() for optional parameters.

    Returns
    -------
    summary_df : pd.DataFrame
        Summary table of all credible sets
    susie_outputs : dict
        Full output, including Bayes factors
    """
    if 'window' in kwargs:
        window = kwargs['window']
    else:
        window = 1000000

    locus_df = locus_df.rename(columns={'position':'pos'}).copy()

    # number of loci and index for each phenotype
    num_loci = defaultdict(int)
    locus_ix = []
    for phenotype_id in locus_df['phenotype_id']:
        num_loci[phenotype_id] += 1
        locus_ix.append(num_loci[phenotype_id])
    locus_df['locus'] = locus_ix

    if 'start' in locus_df and 'end' in locus_df:
        locus_df['locus_id'] = locus_df.apply(lambda x: f"{x['chr']}:{np.maximum(x['start'], 1)}-{x['end']}")
        pos_df = locus_df[['phenotype_id', 'chr', 'start', 'end']]
    else:
        locus_df['locus_id'] = locus_df.apply(lambda x: f"{x['chr']}:{np.maximum(x['pos']-window, 1)}-{x['pos']+window}", axis=1)
        pos_df = locus_df[['phenotype_id', 'chr', 'pos']]

    # fine-map each locus (iterate over chunks, since phenotype can only be present in input once)
    summary_df = []
    res = {}
    nmax = locus_df['locus'].max()
    for i in np.arange(1, nmax + 1):
        print(f"Processing locus group {i}/{nmax}")
        m = locus_df['locus'] == i
        chunk_summary_df, chunk_res = map(genotype_df, variant_df,
                                          phenotype_df.loc[locus_df.loc[m, 'phenotype_id']], pos_df[m].set_index('phenotype_id'),
                                          covariates_df, summary_only=False, **kwargs)
        if len(chunk_summary_df) > 0:
            chunk_summary_df.insert(1, 'locus', i)
            merge_cols = ['phenotype_id', 'locus']
            locus_coords_s = chunk_summary_df.merge(locus_df.loc[m, merge_cols + ['locus_id']],
                                                    left_on=merge_cols, right_on=merge_cols)['locus_id']
            # chunk_summary_df.insert(2, 'locus_id', chunk_summary_df['phenotype_id'] + '_' + locus_coords_s)
            chunk_summary_df.insert(2, 'locus_id', chunk_summary_df['phenotype_id'] + '_' + chunk_summary_df['locus'].astype(str))
            id_dict = chunk_summary_df.set_index('phenotype_id')['locus_id'].to_dict()
            chunk_res = {id_dict[k]:v for k,v in chunk_res.items()}

            summary_df.append(chunk_summary_df)
            res |= chunk_res

    summary_df = pd.concat(summary_df).reset_index(drop=True)

    return summary_df, res


def map(genotype_df, variant_df, phenotype_df, phenotype_pos_df, covariates_df,
        paired_covariate_df=None, L=10, scaled_prior_variance=0.2, estimate_residual_variance=True,
        estimate_prior_variance=True, tol=1e-3, coverage=0.95, min_abs_corr=0.5,
        summary_only=True, maf_threshold=0, max_iter=200, window=1000000,
        logger=None, verbose=True, warn_monomorphic=False):
    """
    SuSiE fine-mapping: computes SuSiE model for all phenotypes
    """
    assert phenotype_df.columns.equals(covariates_df.index)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if logger is None:
        logger = SimpleLogger()

    logger.write('SuSiE fine-mapping')
    logger.write(f'  * {phenotype_df.shape[1]} samples')
    logger.write(f'  * {phenotype_df.shape[0]} phenotypes')
    logger.write(f'  * {covariates_df.shape[1]} covariates')
    if paired_covariate_df is not None:
        assert covariates_df is not None
        assert paired_covariate_df.columns.equals(phenotype_df.columns), f"Paired covariate samples must match samples in phenotype matrix."
        paired_covariate_df = paired_covariate_df.T  # samples x phenotypes
        logger.write(f'  * including phenotype-specific covariate')
    logger.write(f'  * {variant_df.shape[0]} variants')
    logger.write(f'  * cis-window: ±{window:,}')
    if maf_threshold > 0:
        logger.write(f'  * applying in-sample MAF >= {maf_threshold} filter')

    residualizer = Residualizer(torch.tensor(covariates_df.values, dtype=torch.float32).to(device))

    genotype_ix = np.array([genotype_df.columns.tolist().index(i) for i in phenotype_df.columns])
    genotype_ix_t = torch.from_numpy(genotype_ix).to(device)

    igc = genotypeio.InputGeneratorCis(genotype_df, variant_df, phenotype_df, phenotype_pos_df, window=window)
    if igc.n_phenotypes == 0:
        raise ValueError('No valid phenotypes found.')

    start_time = time.time()
    logger.write('  * fine-mapping')
    copy_keys = ['pip', 'sets', 'converged', 'elbo', 'niter', 'lbf_variable']
    susie_summary = []
    if not summary_only:
        susie_res = {}
    for k, (phenotype, genotypes, genotype_range, phenotype_id) in enumerate(igc.generate_data(verbose=verbose), 1):
        # copy genotypes to GPU
        genotypes_t = torch.tensor(genotypes, dtype=torch.float).to(device)
        genotypes_t = genotypes_t[:,genotype_ix_t]
        impute_mean(genotypes_t)

        variant_ids = variant_df.index[genotype_range[0]:genotype_range[-1]+1].rename('variant_id')

        # filter monomorphic variants
        mask_t = ~(genotypes_t == genotypes_t[:, [0]]).all(1)
        if warn_monomorphic:
            logger.write(f'    * WARNING: excluding {~mask_t.sum()} monomorphic variants')
        if maf_threshold > 0:
            maf_t = calculate_maf(genotypes_t)
            mask_t &= maf_t >= maf_threshold
        if mask_t.any():
            genotypes_t = genotypes_t[mask_t]
            mask = mask_t.cpu().numpy().astype(bool)
            variant_ids = variant_ids[mask]
            genotype_range = genotype_range[mask]

        if genotypes_t.shape[0] == 0:
            logger.write(f'WARNING: skipping {phenotype_id} (no valid variants)')
            continue

        if paired_covariate_df is None or phenotype_id not in paired_covariate_df:
            iresidualizer = residualizer
        else:
            iresidualizer = Residualizer(torch.tensor(np.c_[covariates_df, paired_covariate_df[phenotype_id]],
                                                      dtype=torch.float32).to(device))

        phenotype_t = torch.tensor(phenotype, dtype=torch.float).to(device)
        genotypes_res_t = iresidualizer.transform(genotypes_t)  # variants x samples
        phenotype_res_t = iresidualizer.transform(phenotype_t.reshape(1,-1))  # phenotypes x samples

        res = susie(genotypes_res_t.T, phenotype_res_t.T, L=L,
                    scaled_prior_variance=scaled_prior_variance,
                    coverage=coverage, min_abs_corr=min_abs_corr,
                    estimate_residual_variance=estimate_residual_variance,
                    estimate_prior_variance=estimate_prior_variance,
                    tol=tol, max_iter=max_iter)

        af_t = genotypes_t.sum(1) / (2 * genotypes_t.shape[1])
        res['pip'] = pd.DataFrame({'pip':res['pip'], 'af':af_t.cpu().numpy()}, index=variant_ids)
        if res['sets']['cs'] is not None:
            if res['converged'] == True:
                for c in sorted(res['sets']['cs'], key=lambda x: int(x.replace('L',''))):
                    cs = res['sets']['cs'][c]  # indexes
                    p = res['pip'].iloc[cs].copy().reset_index()
                    p['cs_id'] = c.replace('L','')
                    p.insert(0, 'phenotype_id', phenotype_id)
                    susie_summary.append(p)
                res['lbf_variable'] = res['lbf_variable'][res['sets']['cs_index']]  # drop zero entries
            else:
                print(f'    * phenotype ID: {phenotype_id}')

        if not summary_only:  # keep full results
            susie_res[phenotype_id] = {k:res[k] for k in copy_keys}

    logger.write(f'  Time elapsed: {(time.time()-start_time)/60:.2f} min')
    logger.write('done.')
    if susie_summary:
        susie_summary = pd.concat(susie_summary, axis=0).rename(columns={'snp': 'variant_id'}).reset_index(drop=True)
    if summary_only:
        return susie_summary
    else:
        drop_ids = [k for k in susie_res if susie_res[k]['sets']['cs'] is None]
        for k in drop_ids:
            del susie_res[k]
        return susie_summary, susie_res


def get_summary(res_dict, verbose=True):
    """

      res_dict: gene_id -> SuSiE results
    """
    summary_df = []
    for n,k in enumerate(res_dict, 1):
        if verbose:
            print(f'\rMaking summary {n}/{len(res_dict)}', end='' if n < len(res_dict) else None)
        if res_dict[k]['sets']['cs'] is not None:
            assert res_dict[k]['converged'] == True
            for c in sorted(res_dict[k]['sets']['cs'], key=lambda x: int(x.replace('L',''))):
                cs = res_dict[k]['sets']['cs'][c]  # indexes
                p = res_dict[k]['pip'].iloc[cs].copy().reset_index()
                p['cs_id'] = c.replace('L','')
                p.insert(0, 'phenotype_id', k)
                summary_df.append(p)
    summary_df = pd.concat(summary_df, axis=0).rename(columns={'snp':'variant_id'}).reset_index(drop=True)
    return summary_df

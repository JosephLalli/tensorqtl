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

sys.path.insert(1, os.path.dirname(__file__))
import genotypeio
from core import *


def get_x_attributes(X_t, center=True, scale=True):
    """Compute column means and SDs"""
    cm_t = X_t.mean(0)
    csd_t = X_t.std(0, unbiased=True)
    # set sd = 1 when the column has variance 0
    csd_t[csd_t == 0] = 1

    # honor center/scale flags (matches susieR compute_colstats with
    # center=FALSE -> cm=0, scale=FALSE -> csd=1). NOTE: the previous code
    # assigned to unused locals cm/csd instead of cm_t/csd_t, so center=False
    # and scale=False were silently ignored. This only affects callers that
    # pass intercept=False or standardize=False (e.g. hapmixqtl.map_susie);
    # the default intercept=True/standardize=True path is unchanged.
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
        prior_weights = prior_weights / sum(prior_weights)
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
    # assert optimize_V in ["none", "optim", "uniroot", "EM", "simple"]

    Xty = compute_Xty(X_t, Y_t, xattr['scaled_center'], xattr['scaled_scale'])
    betahat = (1/xattr['d']) * Xty

    shat2 = residual_variance / xattr['d']
    if prior_weights is None:
        prior_weights = torch.full([X.shape[1]], 1/X.shape[1])

    # if optimize_V != 'EM' and optimize_V != 'none':
    #     V = optimize_prior_variance(optimize_V, betahat, shat2, prior_weights,
    #                                 alpha=None, post_mean2=None, V_init=V,
    #                                 check_null_threshold=check_null_threshold)

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

    # if optimize_V == 'EM':
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

    for l in range(L):
        # remove lth effect from fitted values
        s['Xr'] = s['Xr'] - compute_Xb(X_t, (s['alpha'][l,:] * s['mu'][l,:]), xattr['scaled_center'], xattr['scaled_scale'])

        # compute residuals
        R_t = Y_t - s['Xr'].reshape(-1,1)

        res = single_effect_regression(R_t, X_t, xattr, s['V'][l],
                                       residual_variance=s['sigma2'], prior_weights=s['pi'],
                                       optimize_V=estimate_prior_method)

        # update the variational estimate of the posterior mean
        s['mu'][l] = res['mu']
        s['alpha'][l] = res['alpha']
        s['mu2'][l] = res['mu2']
        s['V'][l] = res['V']
        s['lbf'][l] = res['lbf_model']
        s['lbf_variable'][l] = res['lbf']
        s['KL'][l] = -res['loglik'] + SER_posterior_e_loglik(X_t, xattr, R_t, s['sigma2'], res['alpha']*res['mu'], res['alpha']*res['mu2'])
        s['Xr'] = s['Xr'] + compute_Xb(X_t, (s['alpha'][l,:] * s['mu'][l,:]), xattr['scaled_center'], xattr['scaled_scale'])
    return(s)


def get_objective(X_t, xattr, Y_t, s):
    """Get objective function from data and susie fit object"""
    return eloglik(X_t, xattr, Y_t, s) - (s['KL']).sum()


def eloglik(X_t, xattr, Y_t, s):
    """expected loglikelihood for a susie fit"""
    n = X_t.shape[0]
    return -(n/2) * torch.log(2*np.pi*s['sigma2']) - (1/(2*s['sigma2'])) * get_ER2(X_t, xattr, Y_t, s)


def get_ER2(X_t, xattr, Y_t, s):
    """expected squared residuals
      Xr_L is L by N matrix
      s['Xr'] is column sum of Xr_L
    """
    Xr_L = compute_MXt(s['alpha']*s['mu'], X_t, xattr)
    postb2 = s['alpha'] * s['mu2']  # posterior second moment
    return ((Y_t.squeeze()-s['Xr'])**2).sum() - (Xr_L**2).sum() + (xattr['d'].reshape(-1,1) * postb2.T).sum()


def estimate_residual_variance_fct(X_t, xattr, Y_t, s):
    n = X_t.shape[0]
    return (1/n) * get_ER2(X_t, xattr, Y_t, s)


def susie_get_pip(res, prune_by_cs=False, prior_tol=1e-9):
    """
    Compute posterior inclusion probability (PIP) for all variables

      res:  a susie fit, the output of susie(), or simply the posterior inclusion probability matrix alpha
      prune_by_cs:  whether or not to ignore single effects not in reported CS when calculating PIP
      prior_tol:  filter out effects having estimated prior variance smaller than this threshold

    Returns:
      array of posterior inclusion probabilities
    """
    # drop null weight columns
    if res['null_index'] > 0:
        res['alpha'] = res['alpha'][:, -res['null_index']]

    # drop the single effect with estimated prior zero
    include_idx = torch.where(res['V'] > 1e-9)[0]

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
        res = res['alpha'][include_idx]  # TODO: check dims
    else:
        res = torch.zeros([1, res['alpha'].shape[1]])

    return 1 - (1 - res).prod(0)


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
    """subsample and compute min, mean, median and max abs corr"""
    if len(pos) == 1:
        return np.ones(3)
    else:
        if len(pos) > n:
            pos = np.random.choice(pos, n, replace=False)
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


def susie_get_cs(res, X=None, Xcorr=None, coverage=0.95, min_abs_corr=0.5,
                 dedup=True, squared=False):
    """"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if X is not None and Xcorr is not None:
        raise ValueError('Only one of X or Xcorr should be specified.')
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

        threshold = min_abs_corr**2 if squared else min_abs_corr
        is_pure = np.where(purity.values[:,0] >= threshold)[0]
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
          estimate_prior_method='EM',
          check_null_threshold=0, prior_tol=1e-9,
          residual_variance_upperbound=np.inf,
          # s_init=None,
          coverage=0.95, min_abs_corr=0.5,
          compute_univariate_zscore=False,
          na_rm=False, max_iter=100, tol=0.001,
          verbose=False, track_fit=False):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n, p = X_t.shape
    mean_y = y_t.mean()

    if intercept:
        y_t = y_t - mean_y

    xattr = get_x_attributes(X_t, center=intercept, scale=standardize)

    # initialize susie fit
    s = init_setup(n, p, L, scaled_prior_variance, y_t.var(unbiased=True),
                   residual_variance=residual_variance,
                   prior_weights=prior_weights, null_weight=null_weight)
    s = init_finalize(s)

    # initialize elbo to NA
    elbo = torch.full([max_iter + 1], np.nan).to(device)
    elbo[0] = -np.inf;
    tracking = []
    for i in range(1, max_iter+1):

        s = update_each_effect(X_t, xattr, y_t, s,
                               estimate_prior_variance=estimate_prior_variance,
                               estimate_prior_method=estimate_prior_method,
                               check_null_threshold=0)
        elbo[i] = get_objective(X_t, xattr, y_t, s)
        if verbose:
            print(f'Objective (iter {i}): {elbo[i]}')
        if (elbo[i] - elbo[i-1]) < tol:
            s['converged'] = True
            break

        if estimate_residual_variance:
            s['sigma2'] = estimate_residual_variance_fct(X_t, xattr, y_t, s)
            if s['sigma2'] > residual_variance_upperbound:
                s['sigma2'] = residual_variance_upperbound
            if verbose:
                print(f'Objective (iter {i}): {get_objective(X_t, xattr, y_t, s)}')

    s['elbo'] = elbo[1:i+1].cpu().numpy()  # Remove first (infinite) entry, and trailing NAs.
    s['niter'] = i

    if 'converged' not in s:
        print(f"\n    WARNING: IBSS algorithm did not converge in {max_iter} iterations!")
        s['converged'] = False

    if intercept:
        s['intercept'] = mean_y - (xattr['scaled_center'] * ((s['alpha']*s['mu']).sum(0)/xattr['scaled_scale'])).sum()
        s['fitted'] = s['Xr'] + mean_y
    else:
        s['intercept'] = 0
        s['fitted'] = s['Xr']

    s['fitted'] = s['fitted'].squeeze()
    # if track_fit:
    #     s['trace'] = tracking

    s['lbf_variable'] = s['lbf_variable'].cpu().numpy()

    # SuSiE CS and PIP
    if coverage is not None and min_abs_corr is not None:
        s['sets'] = susie_get_cs(s, coverage=coverage, X=X_t, min_abs_corr=min_abs_corr)
        s['pip'] = susie_get_pip(s, prune_by_cs=False, prior_tol=prior_tol).cpu().numpy()

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


def map_knockoffs(genotype_df, variant_df, phenotype_df, phenotype_pos_df, covariates_df,
                  fdr=0.05, n_knockoffs=5, knockoff='gaussian', shrink=0.05,
                  w_stat='pip', knockoff_offset=1, seed=0, permute_null=False,
                  paired_covariate_df=None, L=10, scaled_prior_variance=0.2,
                  estimate_residual_variance=True, estimate_prior_variance=True,
                  tol=1e-3, coverage=0.95, min_abs_corr=0.5,
                  maf_threshold=0, max_iter=200, window=1000000,
                  logger=None, verbose=True, warn_monomorphic=False):
    """
    EXPERIMENTAL -- knockoff-INFORMED calibration score, NOT FDR-controlled.

    This computes a credible-set-level statistic from the ORIGINAL columns of an
    augmented [X, X_ko] SuSiE fit and pools it genome-wide. External review
    established (and tests/test_knockoffs.py::TestSwapEquivariance confirms) that
    this CS-level statistic is NOT swap-antisymmetric -- under the model-X swap a
    real signal's credible set disappears rather than negating -- so the pooled
    negatives are not valid negative controls and the resulting 'knockoff_qval'
    does NOT control FDR. It is retained only as an exploratory calibration
    score. For a valid, FDR-controlled procedure use map_egenes_knockoffs
    (eGene-level FDR, Path A).

    Additional known issues vs. a rigorous procedure (see design doc):
      - uses L (not 2L) now, but still fits an augmented design;
      - averages W across draws (invalid derandomization; e-values would be
        required);
      - member-set matching of dynamically-formed CSs is unstable in real LD.

    Args (knockoff-specific; the rest mirror susie.map):
        fdr: target genome-wide CS-level FDR (the sensitivity dial).
        n_knockoffs: M knockoff draws per gene; each CS's W is averaged over
            draws (matched by original-member set) to reduce selection variance.
        knockoff: 'gaussian' (default). 'hmm' reserved for a later phase.
        shrink: covariance shrinkage for the Gaussian generator (mandatory > 0
            at small N).
        w_stat: 'pip' or 'max_alpha' importance.
        knockoff_offset: 1 for knockoff+ (recommended).
        permute_null: if True, permute each phenotype across samples before
            fitting (destroys all association) -- for the calibration harness:
            every selected CS is then a known false discovery.

    Returns:
        (summary_df, diagnostics): summary_df has one row per credible-set member
        (as susie.map's summary) plus columns cs_W, knockoff_qval, selected.
        diagnostics is a dict with the pooled W distribution and per-gene counts.
    """
    import knockoffs as ko

    assert phenotype_df.columns.equals(covariates_df.index)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if logger is None:
        logger = SimpleLogger()

    logger.write('SuSiE fine-mapping with knockoff FDR calibration')
    logger.write(f'  * {phenotype_df.shape[1]} samples')
    logger.write(f'  * {phenotype_df.shape[0]} phenotypes')
    logger.write(f'  * {covariates_df.shape[1]} covariates')
    logger.write(f'  * target FDR: {fdr}')
    logger.write(f'  * knockoffs: {knockoff}, {n_knockoffs} draw(s), shrink={shrink}')
    if permute_null:
        logger.write('  * PERMUTE-NULL mode (calibration): phenotypes shuffled')
    logger.write(f'  * cis-window: ±{window:,}')

    rng = np.random.RandomState(seed)

    residualizer = Residualizer(torch.tensor(covariates_df.values, dtype=torch.float32).to(device))

    genotype_ix = np.array([genotype_df.columns.tolist().index(i) for i in phenotype_df.columns])
    genotype_ix_t = torch.from_numpy(genotype_ix).to(device)

    igc = genotypeio.InputGeneratorCis(genotype_df, variant_df, phenotype_df, phenotype_pos_df, window=window)
    if igc.n_phenotypes == 0:
        raise ValueError('No valid phenotypes found.')

    L_aug = 2 * L  # augmented design has 2p columns

    start_time = time.time()
    logger.write('  * PASS 1: per-gene knockoff fits')
    # each element: dict(phenotype_id, cs_id, W, member_df) -- member_df carries
    # the original-variant rows (variant_id, pip, af) for the summary output.
    cs_records = []
    n_genes_used = 0
    for k, (phenotype, genotypes, genotype_range, phenotype_id) in enumerate(igc.generate_data(verbose=verbose), 1):
        genotypes_t = torch.tensor(genotypes, dtype=torch.float).to(device)
        genotypes_t = genotypes_t[:, genotype_ix_t]
        impute_mean(genotypes_t)

        variant_ids = variant_df.index[genotype_range[0]:genotype_range[-1]+1].rename('variant_id')

        mask_t = ~(genotypes_t == genotypes_t[:, [0]]).all(1)
        if maf_threshold > 0:
            maf_t = calculate_maf(genotypes_t)
            mask_t &= maf_t >= maf_threshold
        if mask_t.any():
            genotypes_t = genotypes_t[mask_t]
            mask = mask_t.cpu().numpy().astype(bool)
            variant_ids = variant_ids[mask]
        if genotypes_t.shape[0] == 0:
            logger.write(f'WARNING: skipping {phenotype_id} (no valid variants)')
            continue

        if paired_covariate_df is None or phenotype_id not in paired_covariate_df:
            iresidualizer = residualizer
        else:
            iresidualizer = Residualizer(torch.tensor(np.c_[covariates_df, paired_covariate_df[phenotype_id]],
                                                      dtype=torch.float32).to(device))

        phenotype = np.asarray(phenotype, dtype=np.float64)
        if permute_null:
            phenotype = phenotype[rng.permutation(phenotype.shape[0])]
        phenotype_t = torch.tensor(phenotype, dtype=torch.float).to(device)
        genotypes_res_t = iresidualizer.transform(genotypes_t)          # variants x samples
        phenotype_res_t = iresidualizer.transform(phenotype_t.reshape(1, -1))  # 1 x samples

        X_t = genotypes_res_t.T                                          # samples x variants
        y_t = phenotype_res_t.T                                         # samples x 1
        af_t = genotypes_t.sum(1) / (2 * genotypes_t.shape[1])
        af_np = af_t.cpu().numpy()
        p = X_t.shape[1]

        # M knockoff draws; accumulate each CS's W keyed by its original-member set
        from collections import defaultdict as _dd
        acc_W = _dd(list)
        acc_members = {}
        for r in range(n_knockoffs):
            gen = torch.Generator(device='cpu').manual_seed(seed * 100003 + k * 101 + r)
            if knockoff == 'gaussian':
                Xk_t = ko.gaussian_knockoff(X_t.cpu(), shrink=shrink, generator=gen).to(device)
            else:
                raise NotImplementedError(f"knockoff='{knockoff}' not yet implemented")
            res, _ = ko.augmented_susie_fit(
                susie, X_t, y_t, Xk_t, L_aug,
                scaled_prior_variance=scaled_prior_variance,
                coverage=coverage, min_abs_corr=min_abs_corr,
                estimate_residual_variance=estimate_residual_variance,
                estimate_prior_variance=estimate_prior_variance,
                tol=tol, max_iter=max_iter)
            if not res.get('converged', False):
                continue
            for c in ko.cs_level_W(res, p, stat=w_stat):
                key = frozenset(c['orig_idx'].tolist())
                acc_W[key].append(c['W'])
                acc_members[key] = c['orig_idx']

        if not acc_W:
            continue
        n_genes_used += 1
        for key, ws in acc_W.items():
            orig_idx = acc_members[key]
            member_df = pd.DataFrame({
                'variant_id': [variant_ids[i] for i in orig_idx],
                'af': af_np[orig_idx],
            })
            cs_records.append({
                'phenotype_id': phenotype_id,
                'W': float(np.mean(ws)),          # mean-W aggregation over draws
                'n_draws': len(ws),
                'members': member_df,
            })

    logger.write(f'  * {n_genes_used} genes with credible sets; {len(cs_records)} CSs total')
    logger.write(f'  * PASS 2: pooling W genome-wide and assigning knockoff q-values')

    W_all = np.array([r['W'] for r in cs_records], dtype=np.float64)
    qvals = ko.pooled_cs_qvalues(W_all, offset=knockoff_offset) if len(W_all) else np.array([])

    rows = []
    for rec, q in zip(cs_records, qvals):
        m = rec['members'].copy()
        m.insert(0, 'phenotype_id', rec['phenotype_id'])
        m['cs_W'] = rec['W']
        m['knockoff_qval'] = q
        m['selected'] = bool(q <= fdr)
        rows.append(m)
    summary_df = pd.concat(rows, axis=0).reset_index(drop=True) if rows else \
        pd.DataFrame(columns=['phenotype_id', 'variant_id', 'af', 'cs_W', 'knockoff_qval', 'selected'])

    n_sel_cs = sum(1 for r, q in zip(cs_records, qvals) if q <= fdr)
    logger.write(f'  * {n_sel_cs} credible sets selected at FDR <= {fdr}')
    logger.write(f'  Time elapsed: {(time.time()-start_time)/60:.2f} min')
    logger.write('done.')

    diagnostics = {
        'W_all': W_all,
        'qvals': np.asarray(qvals),
        'n_genes_used': n_genes_used,
        'n_cs_total': len(cs_records),
        'n_cs_selected': n_sel_cs,
        'fdr': fdr,
    }
    return summary_df, diagnostics


def map_egenes_knockoffs(genotype_df, variant_df, phenotype_df, phenotype_pos_df, covariates_df,
                         fdr=0.1, n_knockoffs=1, knockoff='gaussian', shrink=0.05,
                         gene_stat='max', knockoff_offset=0, selection='qvalue',
                         aggregate='median', seed=0, permute_null=False,
                         paired_covariate_df=None, L=10, scaled_prior_variance=0.2,
                         estimate_residual_variance=True, estimate_prior_variance=True,
                         tol=1e-3, coverage=0.95, min_abs_corr=0.5,
                         maf_threshold=0, max_iter=200, window=1000000,
                         hmm_K=8, hmm_em_iter=25, hmm_params=None,
                         hmm_method='genotype', phased_haplotypes=None,
                         localize=True, logger=None, verbose=True):
    """
    eGene discovery with knockoff FDR control (Path A), then SuSiE localization.

    This is the VALID knockoff procedure (external review, Path A). For each gene
    it fits SuSiE on the augmented [X, X_knockoff] design and computes the
    gene-level statistic

        W_g = max PIP(original block) - max PIP(knockoff block)   (gene_stat='max')

    which tests the FIXED hypothesis H_g: gene g has no cis signal. Because H_g
    does not depend on which credible sets SuSiE forms, W_g is swap-antisymmetric
    (verified in tests) and is a valid model-X knockoff statistic.

    Selection (validated defaults, docs/knockoff_susie_design.md phase 3):
    genes are selected by a genome-wide-pooled, CALIBRATED knockoff q-value
    (selection='qvalue', knockoff_offset=0). This tracks realized FDR ~ nominal q
    at operating q -- the project's goal (calibration, not merely conservative
    control). The default n_knockoffs=1 is a single draw; n_knockoffs>1 with
    aggregate='median' gently stabilizes seed-dependence by taking the median of
    the CALIBRATED per-gene q-value across draws (NOT e-values). The alternative
    selection='ebh' uses Ren-Barber e-value derandomization (knockoff+ control),
    which is provably FDR<=q but empirically over-conservative and unstable near
    the detection floor -- retained as an option, not the default.

    The reported unit is the GENE: "these genes have cis signal at FDR <= q." It
    does NOT claim per-credible-set FDR. If localize=True, ordinary SuSiE is run
    on the original genotypes of each selected gene to produce credible sets that
    localize the signal within the selected genes.

    IMPORTANT: L is the intended number of biological effects, not 2L; original
    and knockoff variables compete for the same L slots.

    Args (knockoff-specific; rest mirror susie.map):
        fdr: target genome-wide eGene FDR.
        n_knockoffs: number of knockoff draws (1 = single draw, default). For
            knockoff='hmm' this is the number of chromosome-coherent draws.
        knockoff: 'gaussian' (second-order, residualized, per-gene; O(p^3)) or
            'hmm' (fastPHASE-style HMM fit per chromosome, then chromosome-
            COHERENT knockoff draws sliced per gene). The HMM generator draws one
            knockoff copy of each whole chromosome, so genes with overlapping
            cis-windows share the same knockoff on shared variants -- the
            coherence the per-gene Gaussian generator cannot give. See hmm_method.
        gene_stat: 'max' or 'sum' for gene_level_W.
        knockoff_offset: 0 (calibrated FDP estimate, default) or 1 (knockoff+
            control, conservative).
        selection: 'qvalue' (calibrated pooled q-value, default), 'ebh'
            (Ren-Barber e-value derandomization / knockoff+ control), 'pvalue'
            (per-gene knockoff p-value + BH; needs many draws, see
            ko.select_egenes_pvalue), or 'calibrated' (step-3 known-null
            Binomial(M,1/2) Storey q-value with a pi0-free mirror cross-check and
            an interval-valued local fdr; ko.select_egenes_calibrated). With
            'calibrated', n_knockoffs is the number of draws M that define the
            per-gene null resolution -- use M>=20-50.
        aggregate: for selection='qvalue' with n_knockoffs>1: 'median' (default)
            | 'mean' | 'none' (draw 0 only).
        hmm_K: number of latent haplotype clusters for the per-chromosome HMM fit
            (knockoff='hmm' only).
        hmm_em_iter: Baum-Welch EM iterations for the HMM fit (knockoff='hmm').
        hmm_params: optional dict chrom -> pre-fit HMM parameters (format depends
            on hmm_method); skips EM for those chromosomes (knockoff='hmm' only).
        hmm_method: HMM knockoff construction (knockoff='hmm' only):
            'genotype'     : Route 1 -- the EXACT diploid pair-state genotype HMM
                             fit from unphased dosages (default). O(N p K^4).
            'haplotype'    : Route 2 -- phased haplotype knockoffs summed to a
                             dosage; requires phased_haplotypes. Exact, O(N p K^2),
                             the cheaper path when phase is available.
            'single_chain' : cheap single K-state chain with a free 3-way emission
                             (approximate; not the exact diploid law).
        phased_haplotypes: (xL_df, xR_df) tuple of phased-allele DataFrames with
            the SAME index/columns as genotype_df (0/1 alleles). Required for
            hmm_method='haplotype'.
        permute_null: Freedman-Lane residual permutation (calibration harness).
        localize: if True, fine-map selected genes with ordinary SuSiE.

    Returns:
        (egene_df, localize_summary_df, diagnostics)
        egene_df: one row per gene: phenotype_id, qvalue (or evalue for ebh),
            selected.
        localize_summary_df: SuSiE credible-set members for selected genes
            (as susie.map's summary), or None if localize=False.
        diagnostics: dict (W_per_draw matrix, selected genes, etc.).
    """
    import knockoffs as ko

    assert phenotype_df.columns.equals(covariates_df.index)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if logger is None:
        logger = SimpleLogger()

    logger.write('Knockoff eGene discovery (Path A: gene-level FDR)')
    logger.write(f'  * {phenotype_df.shape[1]} samples, {phenotype_df.shape[0]} phenotypes')
    logger.write(f'  * target eGene FDR: {fdr}')
    logger.write(f'  * knockoffs: {knockoff}, {n_knockoffs} draw(s), shrink={shrink}; '
                 f'selection={selection} (offset={knockoff_offset})')
    if permute_null:
        logger.write('  * PERMUTE-NULL mode (calibration)')
    logger.write(f'  * cis-window: ±{window:,}')

    rng = np.random.RandomState(seed)
    residualizer = Residualizer(torch.tensor(covariates_df.values, dtype=torch.float32).to(device))
    genotype_ix = np.array([genotype_df.columns.tolist().index(i) for i in phenotype_df.columns])
    genotype_ix_t = torch.from_numpy(genotype_ix).to(device)

    igc = genotypeio.InputGeneratorCis(genotype_df, variant_df, phenotype_df, phenotype_pos_df, window=window)
    if igc.n_phenotypes == 0:
        raise ValueError('No valid phenotypes found.')

    # Calibration null via Freedman-Lane residual permutation. ONE shared sample
    # permutation is applied to EVERY gene's covariate-residual, so cross-gene
    # residual covariance Cov(r_g, r_h) -- the very structure the overlapping-
    # gene joint-sign concern is about -- is preserved. (Permuting raw phenotype,
    # or permuting each gene independently, would destroy that structure and make
    # the calibration artificially easy.) Applied in residual space, which is
    # where the downstream augmented SuSiE fit operates. Note: this assumes
    # unrestricted sample exchangeability after covariate adjustment; restricted
    # exchangeability blocks (relatedness, batch, strata) are not yet supported.
    n_samples = phenotype_df.shape[1]
    fl_perm = rng.permutation(n_samples) if permute_null else None

    # -----------------------------------------------------------------------
    # HMM knockoffs: fit one genotype HMM per chromosome and draw M coherent
    # knockoff copies of the whole chromosome UP FRONT. Each gene later slices
    # its cis-window out of these draws (see the per-gene loop). This is what
    # makes overlapping genes share a knockoff on shared variants; a per-gene
    # generator (like Gaussian below) cannot. Memory/compute scale with the
    # whole chromosome, which is inherent to coherent construction.
    # -----------------------------------------------------------------------
    hmm_draws_by_chrom = None   # chrom -> [M, N, p_chrom] int knockoff dosages
    chrom_row_offset = None     # chrom -> first genotype_df row index for chrom
    if knockoff == 'hmm':
        logger.write(f'  * PASS 0: per-chromosome HMM ({hmm_method}) fit + '
                     f'{n_knockoffs} coherent knockoff draw(s) (K={hmm_K})')
        if hmm_method == 'haplotype' and phased_haplotypes is None:
            raise ValueError("hmm_method='haplotype' requires phased_haplotypes=(xL_df, xR_df)")

        def _states_in_pheno_order(df_values, lo, hi):
            """Slice variant rows [lo:hi], reorder to phenotype samples, impute,
            round to integer states."""
            V = df_values[lo:hi][:, genotype_ix].astype(np.float64)   # [p_c, N]
            if np.isnan(V).any():
                rm = np.nanmean(V, axis=1, keepdims=True)
                rm = np.where(np.isnan(rm), 0.0, rm)
                V = np.where(np.isnan(V), rm, V)
            return np.rint(V).T                                        # [N, p_c] float

        # Row blocks per chromosome. cis-ranges are contiguous row slices, so we
        # require each chromosome to occupy a contiguous, ordered row block (the
        # standard tensorQTL genotype layout, sorted by chrom then position).
        chrom_arr = variant_df['chrom'].values
        hmm_draws_by_chrom, chrom_row_offset = {}, {}
        row0 = 0
        for c in pd.unique(chrom_arr):
            rows = np.where(chrom_arr == c)[0]
            if not (rows[-1] - rows[0] + 1 == rows.size and (np.diff(rows) == 1).all()):
                raise ValueError(f"chromosome {c} variants are not a contiguous "
                                 "ordered block in genotype_df; sort by chrom,pos.")
            lo, hi = rows[0], rows[-1] + 1
            params = None if hmm_params is None else hmm_params.get(c, None)
            cseed = seed * 100003 + int(row0)
            if hmm_method == 'haplotype':
                xL_df, xR_df = phased_haplotypes
                xL = _states_in_pheno_order(xL_df.values, lo, hi).clip(0, 1).astype(np.int64)
                xR = _states_in_pheno_order(xR_df.values, lo, hi).clip(0, 1).astype(np.int64)
                draws = ko.chromosome_hmm_knockoffs(
                    K=hmm_K, M=n_knockoffs, n_em_iter=hmm_em_iter, seed=cseed,
                    method='haplotype', xL=xL, xR=xR, params=params)
            else:  # 'genotype' (exact) or 'single_chain' (approx)
                G = _states_in_pheno_order(genotype_df.values, lo, hi).clip(0, 2).astype(np.int64)
                draws = ko.chromosome_hmm_knockoffs(
                    G, K=hmm_K, M=n_knockoffs, E=3, n_em_iter=hmm_em_iter,
                    seed=cseed, method=hmm_method, params=params)
            hmm_draws_by_chrom[c] = draws
            chrom_row_offset[c] = rows[0]
            row0 += rows.size
        logger.write(f'  * PASS 0 done: {len(hmm_draws_by_chrom)} chromosome(s)')

    start_time = time.time()
    logger.write('  * PASS 1: per-gene augmented fits, gene-level W per knockoff draw')

    gene_ids = []
    W_rows = []  # one row per gene: [W_draw_0, ..., W_draw_{M-1}]
    for k, (phenotype, genotypes, genotype_range, phenotype_id) in enumerate(igc.generate_data(verbose=verbose), 1):
        genotypes_t = torch.tensor(genotypes, dtype=torch.float).to(device)
        genotypes_t = genotypes_t[:, genotype_ix_t]
        impute_mean(genotypes_t)

        mask_t = ~(genotypes_t == genotypes_t[:, [0]]).all(1)
        if maf_threshold > 0:
            maf_t = calculate_maf(genotypes_t)
            mask_t &= maf_t >= maf_threshold
        # remember the variant mask so the HMM knockoff slice is masked identically
        mask_applied = mask_t.cpu().numpy() if mask_t.any() else None
        if mask_t.any():
            genotypes_t = genotypes_t[mask_t]
        if genotypes_t.shape[0] == 0:
            continue

        if paired_covariate_df is None or phenotype_id not in paired_covariate_df:
            iresidualizer = residualizer
        else:
            iresidualizer = Residualizer(torch.tensor(np.c_[covariates_df, paired_covariate_df[phenotype_id]],
                                                      dtype=torch.float32).to(device))

        phenotype = np.asarray(phenotype, dtype=np.float64)
        phenotype_t = torch.tensor(phenotype, dtype=torch.float).to(device)
        X_t = iresidualizer.transform(genotypes_t).T
        y_t = iresidualizer.transform(phenotype_t.reshape(1, -1)).T   # residual space
        if fl_perm is not None:
            # Freedman-Lane: permute the covariate-residual with the SHARED
            # permutation (rows = samples). Preserves cross-gene residual
            # covariance; disrupts sample-wise alignment with genotype.
            y_t = y_t[fl_perm]
        p = X_t.shape[1]

        draw_W = []
        for r in range(n_knockoffs):
            if knockoff == 'gaussian':
                gen = torch.Generator(device='cpu').manual_seed(seed * 100003 + k * 101 + r)
                Xk_t = ko.gaussian_knockoff(X_t.cpu(), shrink=shrink, generator=gen).to(device)
            elif knockoff == 'hmm':
                # slice this gene's cis-window out of the chromosome-coherent
                # knockoff draw r, mask identically, then residualize like X.
                c = variant_df['chrom'].values[genotype_range[0]]
                local = np.asarray(genotype_range) - chrom_row_offset[c]
                Xk_win = hmm_draws_by_chrom[c][r][:, local]          # [N, p_window]
                if mask_applied is not None:
                    Xk_win = Xk_win[:, mask_applied]                 # [N, p]
                Xk_raw = torch.tensor(Xk_win.T, dtype=torch.float).to(device)  # [p, N]
                Xk_t = iresidualizer.transform(Xk_raw).T             # [N, p]
            else:
                raise NotImplementedError(f"knockoff='{knockoff}' not yet implemented")
            res, _ = ko.augmented_susie_fit(
                susie, X_t, y_t, Xk_t, L,
                scaled_prior_variance=scaled_prior_variance,
                coverage=coverage, min_abs_corr=min_abs_corr,
                estimate_residual_variance=estimate_residual_variance,
                estimate_prior_variance=estimate_prior_variance,
                tol=tol, max_iter=max_iter)
            draw_W.append(ko.gene_level_W(res['pip'], p, kind=gene_stat))
        gene_ids.append(phenotype_id)
        W_rows.append(draw_W)

    if not gene_ids:
        raise ValueError('No genes produced statistics.')
    W_per_draw = np.array(W_rows, dtype=np.float64).T   # [n_draws, n_genes]

    if selection == 'ebh':
        logger.write(f'  * PASS 2: e-value (knockoff+) eGene selection at FDR <= {fdr}')
        sel = ko.select_egenes(gene_ids, W_per_draw, q=fdr, offset=(knockoff_offset or 1))
        selected_genes = set(sel['selected'])
        score_col, score_vals = 'evalue', sel['evalues']
    elif selection == 'pvalue':
        logger.write(f'  * PASS 2: per-gene knockoff p-value + BH selection at FDR <= {fdr} '
                     f'(M={W_per_draw.shape[0]} draws)')
        sel = ko.select_egenes_pvalue(gene_ids, W_per_draw, q=fdr,
                                      offset=(knockoff_offset or 1))
        selected_genes = set(sel['selected'])
        score_col, score_vals = 'pvalue', sel['pvalues']
    elif selection == 'calibrated':
        logger.write(f'  * PASS 2: step-3 known-null (Binomial(M,1/2)) calibrated '
                     f'q-value selection at FDR <= {fdr} (M={W_per_draw.shape[0]} draws)')
        sel = ko.select_egenes_calibrated(gene_ids, W_per_draw, q=fdr,
                                          offset=(knockoff_offset or 1))
        selected_genes = set(sel['selected'])
        score_col, score_vals = 'qvalue', sel['qvalues']
        logger.write(f'    - pi0={sel["pi0"]:.3f}; mirror cross-check selected '
                     f'{sel["mirror"]["n_selected"]} (agreement={sel["agreement"]:.2f}); '
                     f'pi0 interval=[{sel["lfdr"]["pi0_lo"]:.3f}, {sel["lfdr"]["pi0_hi"]:.3f}]')
        if sel.get('mirror_informative') and sel['agreement'] < 0.5 and selected_genes:
            logger.write('    ! WARNING: q-value and pi0-free mirror selections '
                         'disagree (agreement<0.5, above the mirror detection '
                         'floor) -- possible knockoff misspecification; treat FDR '
                         'numbers with caution.')
    else:  # 'qvalue' (default, calibrated)
        logger.write(f'  * PASS 2: calibrated q-value eGene selection at FDR <= {fdr} '
                     f'(offset={knockoff_offset}, aggregate={aggregate})')
        sel = ko.select_egenes_qvalue(gene_ids, W_per_draw, q=fdr,
                                      offset=knockoff_offset, aggregate=aggregate)
        selected_genes = set(sel['selected'])
        score_col, score_vals = 'qvalue', sel['qvalues']
    logger.write(f'  * {len(selected_genes)}/{len(gene_ids)} genes selected as eGenes')

    egene_df = pd.DataFrame({
        'phenotype_id': gene_ids,
        score_col: score_vals,
        'selected': [g in selected_genes for g in gene_ids],
    })

    localize_summary_df = None
    if localize and selected_genes:
        logger.write('  * PASS 3: SuSiE localization within selected eGenes')
        sub_pheno = phenotype_df.loc[[g for g in phenotype_df.index if g in selected_genes]]
        sub_pos = phenotype_pos_df.loc[sub_pheno.index]
        localize_summary_df = map(
            genotype_df, variant_df, sub_pheno, sub_pos, covariates_df,
            paired_covariate_df=paired_covariate_df, L=L,
            scaled_prior_variance=scaled_prior_variance,
            estimate_residual_variance=estimate_residual_variance,
            estimate_prior_variance=estimate_prior_variance,
            tol=tol, coverage=coverage, min_abs_corr=min_abs_corr,
            summary_only=True, maf_threshold=maf_threshold, max_iter=max_iter,
            window=window, logger=logger, verbose=verbose)

    logger.write(f'  Time elapsed: {(time.time()-start_time)/60:.2f} min')
    logger.write('done.')

    diagnostics = {
        'gene_ids': gene_ids,
        'W_per_draw': W_per_draw,
        'scores': score_vals,
        'selection': selection,
        'n_genes': len(gene_ids),
        'n_selected': len(selected_genes),
        'fdr': fdr,
    }
    return egene_df, localize_summary_df, diagnostics


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

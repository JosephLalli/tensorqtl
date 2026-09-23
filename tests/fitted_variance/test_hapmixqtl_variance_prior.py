"""Regression tests for the empirical-Bayes prior fit of (c, tau) in
hapmixqtl._estimate_c_tau (variance_model='two_component' / 'library_scaled'
with variance_prior).

Three defects were found while landing the prior on BrainVar (2026-09-17) and
each test below fails on the code that had it:

1. Undamped Fisher scoring in (log c, log tau) cycles between two points at
   the step cap when the gene's data do not identify one component (c v
   negligible against tau, or the reverse): 12.7% of 16,674 transcriptome fits
   returned converged=False after 500 iterations, and the returned point was
   one end of the cycle. Fixed by a backtracking line search on the penalized
   objective, a step cap that keeps the step's direction, and a stop rule on
   the objective's gain.
2. A line search whose merit function scales the likelihood by 0.5/kappa while
   the score is scaled by 1/kappa converges quickly to a point that is not the
   posterior mode (190 of 200 identified BrainVar genes moved, up to a factor
   9, every one toward the prior). The merit must be the quasi-likelihood
   whose gradient is the score: sum(-e2/base - log base) / kappa + log prior.
3. Under the deployed low-expression priors (prior median c 0.0018, log-sd
   2.4) the penalized objective is bimodal, and an ascent from the prior mean
   alone stops in the spurious low-c mode for a gene whose data identify c
   near 2, reporting convergence 12 log-posterior units below the mode (an
   estimated 70% of identifiable genes in those two bins, a fifth of the
   transcriptome). Fixed by ascending from both the prior mean and the
   unpenalized clamped fit and keeping the higher objective.

The checks are (a) convergence is reported, (b) the returned point solves the
penalized score equation the fit claims to solve, and (c) with a near-flat
prior the fit agrees with the unpenalized clamped fit on identified genes and
never reaches a lower objective than it.
"""
import numpy as np
import pytest
import torch

import tensorqtl.hapmixqtl as hapmixqtl


def _penalized_score_residual(y, v, c, tau, prior):
    """max over (log c, log tau) of |score - prior gradient| / sqrt(information):
    zero at the posterior mode the fit is defined to return."""
    m_logc, s_logc, m_logtau, s_logtau, kappa = prior
    base = c * v + tau
    om = 1.0 / (base * base)
    X = torch.stack([v, torch.ones_like(v)], 1)
    g = ((X.T @ (om * (y * y - base))) / kappa).cpu().numpy()
    scale_ct = np.array([c, tau])
    g_th = scale_ct * g
    pen = np.array([(np.log(c) - m_logc) / s_logc ** 2, (np.log(tau) - m_logtau) / s_logtau ** 2])
    info = scale_ct[:, None] * (X.T @ (om[:, None] * X) / kappa).cpu().numpy() * scale_ct[None, :]
    denom = np.sqrt(np.diag(info) + np.array([1.0 / s_logc ** 2, 1.0 / s_logtau ** 2]))
    return float(np.max(np.abs(g_th - pen) / denom))


def _quasi_loglik(y, v, c, tau, kappa):
    base = c * v + tau
    return -float((y * y / base + torch.log(base)).sum()) / kappa


def _tensors(y, v, device):
    T = lambda x: torch.tensor(x, dtype=torch.float64, device=device)
    return T(y), T(v)


def test_prior_fit_converges_and_is_stationary_where_c_is_not_identified(device):
    """c v is negligible against tau on every donor, so the objective is flat
    in log c and the expected information there is nearly zero. Undamped
    scoring cycled here (defect 1); the fit must converge and sit at the
    posterior mode."""
    rng = np.random.RandomState(11)
    G, N = 40, 60
    v = 10 ** rng.uniform(-2, 0, (G, N))
    c_true, tau_true = 1e-3, 0.05
    a = rng.normal(0, np.sqrt(c_true * v + tau_true))
    prior = (np.log(0.5), 1.5, np.log(0.05), 1.0, 2.0)   # prior mean far from the truth in log c
    n_conv, worst = 0, 0.0
    for g in range(G):
        y, vv = _tensors(a[g], v[g], device)
        fit = hapmixqtl._estimate_c_tau(y, vv, None, device, intercept=False, prior=prior)
        n_conv += fit['converged']
        assert fit['c'] > 0 and fit['tau'] > 0 and np.isfinite(fit['c']) and np.isfinite(fit['tau'])
        worst = max(worst, _penalized_score_residual(y, vv, fit['c'], fit['tau'], prior))
    assert n_conv == G, f'{G - n_conv} of {G} fits did not converge'
    assert worst < 1e-2, f'returned point is not the posterior mode: score residual {worst:.2e}'


def test_prior_fit_solves_the_score_equation_and_reduces_to_the_clamped_fit(device):
    """Well-identified genes (v spans three decades, both components matter)
    under a near-flat prior centred far from the truth. The fit must reach the
    same optimum as the unpenalized clamped fit; a merit function inconsistent
    with the score (defect 2) stops a factor 60 away."""
    rng = np.random.RandomState(12)
    G, N = 40, 60
    v = 10 ** rng.uniform(-3, 0, (G, N))
    c_true, tau_true = 1.5, 0.02
    a = rng.normal(0, np.sqrt(c_true * v + tau_true))
    prior = (np.log(0.01), 1e3, np.log(1.0), 1e3, 2.0)
    worst_score, worst_dlog, n_interior = 0.0, 0.0, 0
    for g in range(G):
        y, vv = _tensors(a[g], v[g], device)
        f0 = hapmixqtl._estimate_c_tau(y, vv, None, device, intercept=False)
        f1 = hapmixqtl._estimate_c_tau(y, vv, None, device, intercept=False, prior=prior)
        assert f1['converged']
        worst_score = max(worst_score, _penalized_score_residual(y, vv, f1['c'], f1['tau'], prior))
        if f0['c'] > 1e-3 and f0['tau'] > 1e-4:
            n_interior += 1
            worst_dlog = max(worst_dlog, abs(np.log(f1['c'] / f0['c'])), abs(np.log(f1['tau'] / f0['tau'])))
    assert n_interior >= 0.8 * G
    assert worst_score < 1e-3, f'score residual {worst_score:.2e}'
    assert worst_dlog < 1e-3, f'near-flat prior fit differs from the clamped fit by {worst_dlog:.2e} in log'


def test_near_flat_prior_never_reaches_a_lower_objective_than_the_clamped_fit(device):
    """On a mixture of identified and unidentified genes the prior fit under a
    near-flat prior is at least as good an optimum of the quasi-likelihood as
    the clamped iteration, which can stop short along a flat direction."""
    rng = np.random.RandomState(13)
    G, N = 60, 50
    v = 10 ** rng.uniform(-2.5, 0, (G, N))
    c_true = np.where(rng.uniform(size=G) < 0.5, 1e-3, 1.5)
    tau_true = np.where(c_true < 0.01, 0.05, 0.002)
    a = rng.normal(0, np.sqrt(c_true[:, None] * v + tau_true[:, None]))
    kappa = 2.0
    prior = (np.log(0.3), 1e3, np.log(0.01), 1e3, kappa)
    for g in range(G):
        y, vv = _tensors(a[g], v[g], device)
        f0 = hapmixqtl._estimate_c_tau(y, vv, None, device, intercept=False)
        f1 = hapmixqtl._estimate_c_tau(y, vv, None, device, intercept=False, prior=prior)
        assert f1['converged']
        q0 = _quasi_loglik(y, vv, max(f0['c'], 1e-12), max(f0['tau'], 1e-12), kappa)
        q1 = _quasi_loglik(y, vv, f1['c'], f1['tau'], kappa)
        assert q1 >= q0 - 1e-6 * (1.0 + abs(q0)), (g, q0, q1)


# bins 1 and 2 of the deployed BrainVar prior (variance_prior_bins_cvt.tsv): the two that trap
DEPLOYED_LOW_BINS = [(1, -6.321148, 2.439234, -1.514128, 1.688169),
                     (2, -6.308367, 2.439854, -1.836938, 1.723303)]
DEPLOYED_KAPPA = 2.3806


def _penalized_objective(c, tau, prior, y, v):
    m_lc, s_lc, m_lt, s_lt, kappa = prior
    base = c * v + tau
    ll = -np.sum(np.log(base) + y ** 2 / base) / kappa
    d = np.array([np.log(c) - m_lc, np.log(tau) - m_lt])
    return ll - 0.5 * np.sum(d * d * np.array([1 / s_lc ** 2, 1 / s_lt ** 2]))


@pytest.mark.parametrize('bin_id,m_lc,s_lc,m_lt,s_lt', DEPLOYED_LOW_BINS)
def test_prior_fit_is_not_trapped_below_the_clamped_fit(device, bin_id, m_lc, s_lc, m_lt, s_lt):
    """A gene whose data identify c near 2.4 under the deployed low-expression
    prior: the returned posterior mode must score at least as well on the
    penalized objective as the unpenalized clamped fit does. A single ascent
    from the prior mean returns c near 0.002 and fails this by 7 to 12
    log-posterior units."""
    rng = np.random.default_rng(5)
    v = 10 ** rng.uniform(-2, -0.5, 80)
    y = rng.normal(0, np.sqrt(2.5 * v + 0.02))
    prior = (m_lc, s_lc, m_lt, s_lt, DEPLOYED_KAPPA)
    yt, vt = _tensors(y, v, device)
    clamped = hapmixqtl._estimate_c_tau(yt, vt, None, device, intercept=False)
    fit = hapmixqtl._estimate_c_tau(yt, vt, None, device, intercept=False, prior=prior)
    assert fit['converged']
    tau_ref = max(clamped['tau'], np.exp(m_lt) * 1e-3)
    o_fit = _penalized_objective(fit['c'], fit['tau'], prior, y, v)
    o_ref = _penalized_objective(clamped['c'], tau_ref, prior, y, v)
    assert o_fit >= o_ref - 1e-6, (
        f"bin {bin_id}: prior fit c={fit['c']:.6g} is {o_ref - o_fit:.3f} log-posterior units "
        f"below the clamped fit c={clamped['c']:.6g}, yet converged=True")
    assert fit['c'] > 0.5 * clamped['c'], (fit['c'], clamped['c'])

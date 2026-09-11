"""
Model-X knockoffs for calibrating SuSiE credible-set false-discovery rate.

Motivation
----------
SuSiE reports credible sets (CSs) with a nominal coverage guarantee, but that
guarantee is conditional on the model being correctly specified (sparse effects,
correct residual variance, in-sample LD). In real data -- especially at modest
sample size and under a polygenic background -- SuSiE's CSs can be miscalibrated
and over-reported (Cui et al. 2023, Nat Genet; McCreight et al. 2025). Model
fixes such as SuSiE-inf / SuSiE-ash reduce this but do not *certify* a target
FDR: "1.5-3x fewer false CSs" is not "5% of my calls are false."

Knockoffs provide the certification. We synthesize per-gene "knockoff" variants
that share the LD structure of the real variants but are, by construction,
conditionally independent of the phenotype. Running SuSiE on the augmented
design [X, X_knockoff] and comparing each real variant's importance to its
knockoff's yields an *empirical* false-discovery estimate -- a filter that
controls FDR at a user-chosen level q regardless of whether SuSiE's model is
correct.

This module is a general SuSiE calibration facility: its core operates on a
(variants x samples) genotype matrix and a SuSiE result, with no dependency on
the two-channel hapmixQTL specifics. It is used by both the standard
``susie.map`` path and ``hapmixqtl.map_susie``.

Method
------
This module provides TWO generators:

* **HMM knockoffs (recommended default for realistic genotypes)** -- a
  fastPHASE-style hidden-Markov model of the discrete diploid genotypes, matched
  to real LD structure and rare variants, drawn chromosome-coherently. Validated
  swap-exchangeable, O(N*p*K^2..K^4), the scalable path to whole-chromosome
  knockoffs. See ``chromosome_hmm_knockoffs`` / ``genotype_hmm_knockoffs`` /
  ``haplotype_hmm_knockoffs``. The single- and two-channel pipelines default to
  this generator (``knockoff='hmm'``).

* **Gaussian knockoffs (fast second-order approximation)** -- ``gaussian_knockoff``,
  on the covariate-residualized dosage matrix. Given the variant covariance
  ``Sigma``, sample::

    X_knockoff = X (I - Sigma^{-1} D) + E chol(2D - D Sigma^{-1} D)

  where ``D = diag(s)`` and ``E ~ N(0, I)``. Fast and dependency-free, but a
  SECOND-ORDER approximation that is **misspecified on non-Gaussian, HMM-
  structured genotypes**: on strong-LD (r^2>0.5) / rare-variant data it can
  inflate the original-favored false-positive tail (Barber, Candes & Samworth
  2020; empirically confirmed in docs/calibration_findings.md). Use only for
  small p / mild LD, or as a fast comparison arm -- not as the default on real
  eQTL genotypes.

At small N (e.g. ~200 eQTL samples) the empirical covariance is noisy and
rank-deficient, so **shrinkage is mandatory** -- ``Sigma`` is regularized toward
its diagonal (Barber, Candes & Samworth 2020 show the FDR-control error is
governed by how well the knockoff distribution is estimated). Without it, the
knockoffs become too easy to distinguish and the realized FDR is anti-
conservative. Whether the guarantee actually holds at a given N is an empirical
question; see ``calibration_report`` for the null-permutation check that must be
run before trusting the target FDR.

References
----------
- Candes, Fan, Janson & Lv (2018) "Panning for gold: model-X knockoffs", JRSS-B.
- Barber, Candes & Samworth (2020) "Robust inference with knockoffs", Ann.
  Statist. -- FDR robustness under an estimated feature distribution.
- Ren & Barber (2024) "Derandomised knockoffs", JRSS-B -- e-value aggregation.
"""

import torch
import numpy as np


# ---------------------------------------------------------------------------
#  Knockoff generation (second-order / Gaussian)
# ---------------------------------------------------------------------------

def _shrink_covariance(X_t, shrink):
    """
    Regularized correlation-scale covariance of the columns of X.

    Args:
        X_t: [N, p] design (samples x variants), assumed ~zero-mean columns
        shrink: shrinkage intensity gamma in [0, 1]; Sigma is pulled toward its
                diagonal as (1-gamma)*Sigma_hat + gamma*diag(Sigma_hat). This
                both conditions Sigma at small N and guarantees it is PD.

    Returns:
        Sigma_t: [p, p] shrunk covariance (PD for shrink > 0)
    """
    N = X_t.shape[0]
    Xc = X_t - X_t.mean(0, keepdim=True)
    Sigma = (Xc.t() @ Xc) / max(N - 1, 1)
    d = torch.diag(Sigma).clamp(min=1e-8)
    Sigma = (1.0 - shrink) * Sigma + shrink * torch.diag(d)
    return Sigma


def _solve_s_equicorrelated(Sigma_t):
    """
    Equicorrelated s-vector: s_j = min(1, 2*lambda_min(Sigma_corr)) on the
    correlation scale, mapped back to the covariance scale.

    This is the cheap, closed-form choice. It is less powerful than MVR/maxent
    but has no optimization loop and is numerically bulletproof, which matters
    for a default that runs per gene. s is returned on the *covariance* scale
    (i.e. multiplied by the per-variant variances).
    """
    d = torch.diag(Sigma_t).clamp(min=1e-12)
    dinv_sqrt = 1.0 / torch.sqrt(d)
    # correlation matrix
    R = Sigma_t * dinv_sqrt.unsqueeze(0) * dinv_sqrt.unsqueeze(1)
    # smallest eigenvalue of the correlation matrix (symmetric -> eigvalsh)
    evals = torch.linalg.eigvalsh(R)
    lam_min = evals[0].clamp(min=0.0)
    s_corr = torch.clamp(2.0 * lam_min, max=1.0)
    # back to covariance scale
    s = s_corr * d
    return s


def gaussian_knockoff(X_t, shrink=0.05, s_method='equicorrelated',
                      generator=None, jitter=1e-6):
    """
    Second-order (Gaussian) model-X knockoffs for a residualized design.

    Args:
        X_t: [N, p] samples x variants, covariate-residualized (columns ~mean 0)
        shrink: covariance shrinkage intensity (see _shrink_covariance).
            Mandatory > 0 at small N; the default 0.05 is a reasonable start
            but should be tuned against the calibration report.
        s_method: 'equicorrelated' (closed form; only choice implemented here)
        generator: optional torch.Generator for reproducible sampling
        jitter: added to the diagonal of the conditional covariance before
            Cholesky, for numerical PSD safety

    Returns:
        Xk_t: [N, p] knockoff matrix on the same device/dtype as X_t
    """
    N, p = X_t.shape
    device, dtype = X_t.device, X_t.dtype

    mu = X_t.mean(0, keepdim=True)
    Xc = X_t - mu

    Sigma = _shrink_covariance(X_t, shrink)
    if s_method == 'equicorrelated':
        s = _solve_s_equicorrelated(Sigma)
    else:
        raise ValueError(f"unknown s_method: {s_method}")

    D = torch.diag(s)
    # Sigma^{-1} via a solve against D (more stable than explicit inverse)
    Sigma_inv = torch.linalg.inv(Sigma)
    # conditional mean shift: Xc @ (I - Sigma^{-1} D)  == Xc - Xc @ Sigma^{-1} D
    mean_shift = Xc @ (torch.eye(p, device=device, dtype=dtype) - Sigma_inv @ D)

    # conditional covariance V = 2D - D Sigma^{-1} D  (must be PSD)
    V = 2.0 * D - D @ Sigma_inv @ D
    V = 0.5 * (V + V.t())  # symmetrize
    V = V + jitter * torch.eye(p, device=device, dtype=dtype)
    # PSD-safe factor: try Cholesky, fall back to eigendecomposition clamp
    try:
        L = torch.linalg.cholesky(V)
    except Exception:
        evals, evecs = torch.linalg.eigh(V)
        evals = evals.clamp(min=0.0)
        L = evecs @ torch.diag(torch.sqrt(evals))

    if generator is not None:
        E = torch.randn(N, p, device=device, dtype=dtype, generator=generator)
    else:
        E = torch.randn(N, p, device=device, dtype=dtype)

    Xk = mu + mean_shift + E @ L.t()
    return Xk


# ---------------------------------------------------------------------------
#  HMM knockoffs (fastPHASE-style), for realistic genotype distributions and
#  coherent chromosome-wide construction. Independent reimplementation of the
#  algorithm in Sesia, Sabatti & Candes (2019, Biometrika) "Gene hunting with
#  hidden Markov model knockoffs" (cf. the SNPknock package): the HMM knockoff
#  reduces to a discrete-Markov-chain (DMC) knockoff on the latent state path
#  plus independent re-emission. The DMC sampler uses a forward partition
#  function Z and backward messages QN. Unlike the Gaussian generator, this
#  models discrete haplotypic genotypes with rare variants, and is O(N p K^2) --
#  linear in p -- so it scales to chromosome-wide designs where the O(p^3)
#  Gaussian construction is infeasible.
#
#  Implements Algorithm 1 (DMC knockoffs, eqs 4-5 of the paper) and Algorithm 2
#  (HMM knockoffs: forward-backward sample the latent path, DMC-knockoff it,
#  re-emit). VALIDITY VERIFIED: swap-exchangeability holds to Monte-Carlo
#  tolerance at chromosome-window scale (pairwise-swap-TV ~0.005 and FLAT in p up
#  to p=20; see tests/test_hmm_knockoffs.py). NOTE on testing: a naive
#  full-joint-distribution swap-TV over (X, X_knockoff) has a noise floor that
#  GROWS with p because the number of joint cells (|X|^{2p}) vastly exceeds the
#  sample size -- an earlier apparent "compounding bug" was entirely this test
#  artifact, not the algorithm. The correct check compares against that noise
#  floor or uses low-order (pairwise) swap statistics.
# ---------------------------------------------------------------------------

def dmc_knockoffs(X, init_p, Q, seed=0):
    """
    Discrete Markov chain knockoffs (Sesia et al. 2019, Algorithm 1; eqs 4-5).

    Sequentially samples X_knockoff_j from the conditional (eq 4) using the
    forward partition function N_j (eq 5): N_j(k) = sum_l Q1(l) Q_{j+1}(k|l),
    where Q1(l) = Q_j(l|x_{j-1}) Q_j(l|x~_{j-1}) / N_{j-1}(l). Swap-exchangeable
    to Monte-Carlo tolerance at all p (verified in tests).

    Args:
        X: [N, p] integer state matrix (values in 0..K-1).
        init_p: [K] initial state distribution P(state_0).
        Q: [p-1, K, K] transition matrices; Q[j-1][a, b] = P(state_j=b | state_{j-1}=a).
        seed: RNG seed.

    Returns:
        Xt: [N, p] integer knockoff state matrix, swap-exchangeable with X.
    """
    X = np.asarray(X)
    N, p = X.shape
    K = init_p.shape[0]
    rng = np.random.default_rng(seed)
    Xt = np.empty((N, p), dtype=np.int64)

    # Vectorized over individuals. Z is the forward partition function [N, K].
    Z = np.ones((N, K), dtype=np.float64)
    U = rng.random((N, p))  # uniforms for inverse-CDF sampling
    for j in range(p):
        if j > 0:
            Qj = Q[j - 1]                                   # [K, K]
            # Q1[i,k] = Qj[X[i,j-1], k] * Qj[Xt[i,j-1], k] / Z[i,k]
            Q1 = Qj[X[:, j - 1], :] * Qj[Xt[:, j - 1], :] / Z
        else:
            Q1 = np.tile(init_p, (N, 1))                    # [N, K]

        # forward partition update: Znew = Q1 (L=1); if j<p-1: Znew = Q[j].T @ Q1
        if j < p - 1:
            Znew = Q1 @ Q[j]                                # [N,K] @ [K,K] = Q[j].T applied row-wise
        else:
            Znew = Q1.copy()
        Znew = np.maximum(Znew, 1e-50)

        # backward message QN[i,k] = P(X[i,j+1] | state=k) = Q[j][k, X[i,j+1]]
        if j < p - 1:
            QN = Q[j][:, X[:, j + 1]].T                     # [N, K]
        else:
            QN = np.full((N, K), 1.0 / K)

        w = Q1 * QN                                         # [N, K]
        w = w / w.sum(1, keepdims=True)
        # inverse-CDF sample per row
        cdf = np.cumsum(w, axis=1)
        Xt[:, j] = (U[:, j:j + 1] > cdf).sum(1).clip(max=K - 1)
        Z = Znew
    return Xt


def _backward_hmm(X, init_p, Q, emission_p):
    """
    Backward messages beta[j][i,k] = P(X[i,j+1:p] | H_j=k), normalized per (i,j).
    X: [N, p] observed states; emission_p: [p, E, K] with P(X_j=e | H_j=k).
    Returns beta: [p, N, K].
    """
    N, p = X.shape
    K = init_p.shape[0]
    beta = np.ones((p, N, K), dtype=np.float64)
    for j in range(p - 2, -1, -1):
        # fBeta[i,l] = emission_p[j+1][X[i,j+1], l] * beta[j+1][i,l]
        emit_next = emission_p[j + 1][X[:, j + 1], :]       # [N, K]
        fBeta = emit_next * beta[j + 1]                     # [N, K]
        # beta[j][i,k] = sum_l Q[j][k,l] * fBeta[i,l]
        b = fBeta @ Q[j].T                                 # [N, K]
        b = b / np.maximum(b.sum(1, keepdims=True), 1e-300)
        beta[j] = b
    return beta


def _sample_hidden_states(X, init_p, Q, emission_p, beta, rng):
    """Forward-sample H ~ P(H | X) using the backward messages beta."""
    N, p = X.shape
    K = init_p.shape[0]
    H = np.empty((N, p), dtype=np.int64)
    U = rng.random((N, p))
    # j = 0
    w = init_p[None, :] * emission_p[0][X[:, 0], :] * beta[0]
    w = w / w.sum(1, keepdims=True)
    H[:, 0] = (U[:, 0:1] > np.cumsum(w, 1)).sum(1).clip(max=K - 1)
    for j in range(1, p):
        w = Q[j - 1][H[:, j - 1], :] * emission_p[j][X[:, j], :] * beta[j]
        w = w / w.sum(1, keepdims=True)
        H[:, j] = (U[:, j:j + 1] > np.cumsum(w, 1)).sum(1).clip(max=K - 1)
    return H


def hmm_knockoffs(X, init_p, Q, emission_p, seed=0):
    """
    HMM knockoffs for observed genotype sequences (Sesia et al. 2019, Alg. 2).

    Verified swap-exchangeable to Monte-Carlo tolerance at chromosome-window
    scale. (An earlier "small-p only" caveat was a test artifact -- naive
    full-joint swap-TV, whose noise floor grows with p -- not a real bug.)

    Three steps: (1) backward pass for beta = P(future | H_j); (2) forward-sample
    the hidden states H ~ P(H | X); (3) DMC knockoff on H -> Ht; (4) re-emit
    Xt_j ~ P(X_j | H_j = Ht_j). The composition is swap-exchangeable because the
    DMC step is, and the emission is independent given the state.

    Args:
        X: [N, p] observed genotype states (integers, e.g. dosage 0/1/2 -> E=3).
        init_p: [K] hidden initial distribution.
        Q: [p-1, K, K] hidden transition matrices.
        emission_p: [p, E, K] emission P(X_j = e | H_j = k).
        seed: RNG seed.

    Returns:
        Xt: [N, p] observed knockoff genotype states.
    """
    X = np.asarray(X)
    N, p = X.shape
    K = init_p.shape[0]
    E = emission_p.shape[1]
    rng = np.random.default_rng(seed)

    beta = _backward_hmm(X, init_p, Q, emission_p)
    H = _sample_hidden_states(X, init_p, Q, emission_p, beta, rng)
    Ht = dmc_knockoffs(H, init_p, Q, seed=seed + 100000)

    # re-emit observations from knockoff hidden states
    Xt = np.empty((N, p), dtype=np.int64)
    U = rng.random((N, p))
    for j in range(p):
        # w[i,e] = emission_p[j][e, Ht[i,j]]
        w = emission_p[j][:, Ht[:, j]].T                    # [N, E]
        w = w / w.sum(1, keepdims=True)
        Xt[:, j] = (U[:, j:j + 1] > np.cumsum(w, 1)).sum(1).clip(max=E - 1)
    return Xt


# ---------------------------------------------------------------------------
#  Fitting the HMM from data (fastPHASE-style Baum-Welch EM)
#
#  hmm_knockoffs above assumes the HMM parameters (init_p, Q, emission_p) are
#  known. On simulated data they are; on real genotypes they must be estimated.
#  fit_hmm below is a position-inhomogeneous single-chain HMM fitter by
#  Baum-Welch (EM), producing parameters in EXACTLY the layout hmm_knockoffs
#  consumes. This mirrors the fastPHASE model (K latent haplotype/ancestry
#  clusters, position-specific transitions and categorical emissions) but is a
#  self-contained, dependency-free reimplementation.
#
#  Model note. tensorQTL genotypes are unphased dosages in {0,1,2}. We fit a
#  single K-state chain with a 3-category emission (E=3) directly to the dosage
#  sequence. This is the genotype-HMM view: the knockoff is EXACT for the fitted
#  distribution (Candes-Fan-Janson-Lv 2018), and robustness to the fit error is
#  the empirical-calibration question (Barber-Candes-Samworth 2020), exactly as
#  for the Gaussian generator's covariance estimate. Phased haplotypes (E=2, fed
#  one row per haplotype) fit the same way and enable the two-channel work.
# ---------------------------------------------------------------------------

def _forward_backward_scaled(X, init_p, Q, emission_p):
    """
    Scaled forward-backward for a position-inhomogeneous single-chain HMM,
    vectorized over the N sequences.

    Args:
        X: [N, p] integer observations in 0..E-1.
        init_p: [K] initial state distribution.
        Q: [p-1, K, K] transitions, Q[j][a, b] = P(H_{j+1}=b | H_j=a).
        emission_p: [p, E, K], P(X_j = e | H_j = k).

    Returns:
        gamma: [p, N, K] posterior state marginals P(H_j=k | X).
        xi_sum: [p-1, K, K] transition posteriors summed over sequences,
            sum_i P(H_j=a, H_{j+1}=b | X_i).
        loglik: total log-likelihood sum_i log P(X_i).
    """
    N, p = X.shape
    K = init_p.shape[0]
    alpha = np.empty((p, N, K), dtype=np.float64)
    beta = np.empty((p, N, K), dtype=np.float64)
    c = np.empty((p, N), dtype=np.float64)               # scaling factors

    # emission likelihood B[j][i,k] = P(X[i,j] | H_j=k)
    def emit(j):
        return emission_p[j][X[:, j], :]                 # [N, K]

    # forward
    a = init_p[None, :] * emit(0)                        # [N, K]
    c[0] = a.sum(1)
    c[0] = np.maximum(c[0], 1e-300)
    alpha[0] = a / c[0][:, None]
    for j in range(1, p):
        a = (alpha[j - 1] @ Q[j - 1]) * emit(j)          # [N,K]@[K,K] -> [N,K]
        c[j] = np.maximum(a.sum(1), 1e-300)
        alpha[j] = a / c[j][:, None]

    # backward (scaled with the same c)
    beta[p - 1] = 1.0
    for j in range(p - 2, -1, -1):
        b_next = emission_p[j + 1][X[:, j + 1], :] * beta[j + 1]   # [N,K]
        beta[j] = (b_next @ Q[j].T) / c[j + 1][:, None]

    gamma = alpha * beta                                 # [p, N, K]
    gamma /= np.maximum(gamma.sum(2, keepdims=True), 1e-300)

    # transition posteriors, summed over sequences
    xi_sum = np.zeros((p - 1, K, K), dtype=np.float64)
    for j in range(p - 1):
        b_next = emission_p[j + 1][X[:, j + 1], :] * beta[j + 1]   # [N,K]
        # xi[i,a,b] = alpha[j,i,a] Q[j][a,b] b_next[i,b] / c[j+1,i]
        xi = (alpha[j][:, :, None] * Q[j][None, :, :]
              * b_next[:, None, :]) / c[j + 1][:, None, None]
        xi_sum[j] = xi.sum(0)

    loglik = float(np.log(c).sum())
    return gamma, xi_sum, loglik


def fit_hmm(X, K, E=None, n_iter=25, tol=1e-4, pseudocount=1.0, seed=0,
            init_p=None, Q=None, emission_p=None, verbose=False):
    """
    Fit a position-inhomogeneous single-chain HMM by Baum-Welch EM.

    Produces parameters in the exact (init_p, Q, emission_p) layout that
    ``hmm_knockoffs`` / ``dmc_knockoffs`` consume, so the fit output can be fed
    straight into knockoff generation.

    Args:
        X: [N, p] integer observations in 0..E-1 (e.g. dosage 0/1/2 -> E=3, or
           phased alleles 0/1 -> E=2). One row per sequence (individual or
           haplotype).
        K: number of latent states (ancestry/haplotype clusters).
        E: number of emission categories; inferred as X.max()+1 if None.
        n_iter: maximum EM iterations.
        tol: stop when the per-iteration log-likelihood gain per sequence falls
            below this.
        pseudocount: Dirichlet smoothing added to expected counts in the M-step.
            Keeps transitions/emissions strictly positive (required for the
            knockoff sampler's divisions) and regularizes at small N.
        seed: RNG seed for parameter initialization.
        init_p, Q, emission_p: optional warm-start parameters; random init if None.
        verbose: print the log-likelihood trace.

    Returns:
        dict with 'init_p' [K], 'Q' [p-1,K,K], 'emission_p' [p,E,K],
        'loglik' (list of per-iteration total log-likelihoods), 'n_iter', 'E'.
    """
    X = np.asarray(X)
    N, p = X.shape
    if E is None:
        E = int(X.max()) + 1
    rng = np.random.default_rng(seed)

    # --- initialization ---
    if init_p is None:
        init_p = rng.dirichlet(np.ones(K))
    else:
        init_p = np.asarray(init_p, dtype=np.float64).copy()
    if Q is None:
        # bias toward self-transitions (LD persistence) so states are identifiable
        Q = np.empty((p - 1, K, K), dtype=np.float64)
        base = np.full((K, K), 1.0) + (K * 2.0) * np.eye(K)
        for j in range(p - 1):
            for a in range(K):
                Q[j, a] = rng.dirichlet(base[a])
    else:
        Q = np.asarray(Q, dtype=np.float64).copy()
    if emission_p is None:
        emission_p = np.empty((p, E, K), dtype=np.float64)
        for j in range(p):
            for k in range(K):
                emission_p[j][:, k] = rng.dirichlet(np.ones(E))
    else:
        emission_p = np.asarray(emission_p, dtype=np.float64).copy()

    loglik_trace = []
    prev_ll = -np.inf
    for it in range(n_iter):
        gamma, xi_sum, ll = _forward_backward_scaled(X, init_p, Q, emission_p)
        loglik_trace.append(ll)
        if verbose:
            print(f"  [fit_hmm] iter {it:3d}  loglik={ll:.4f}")

        # --- M-step ---
        # initial distribution from first-position marginals
        g0 = gamma[0].sum(0) + pseudocount
        init_p = g0 / g0.sum()

        # transitions: row-normalize summed xi
        Qn = xi_sum + pseudocount
        Q = Qn / Qn.sum(2, keepdims=True)

        # emissions: for each position and category, weight gamma by 1{X==e}
        emission_p = np.empty((p, E, K), dtype=np.float64)
        for e in range(E):
            mask = (X == e).astype(np.float64)           # [N, p]
            # num[j,k] = sum_i mask[i,j] * gamma[j,i,k]; gamma is [p, N, K]
            num = np.einsum('ij,jik->jk', mask, gamma)   # [p, K]
            emission_p[:, e, :] = num
        emission_p += pseudocount
        emission_p /= emission_p.sum(1, keepdims=True)   # normalize over E

        if ll - prev_ll < tol * N and it > 0:
            break
        prev_ll = ll

    return {'init_p': init_p, 'Q': Q, 'emission_p': emission_p,
            'loglik': loglik_trace, 'n_iter': len(loglik_trace), 'E': E}


# ---------------------------------------------------------------------------
#  Diploid genotype HMM (Route 1): the TRUE unphased-dosage law
#
#  An unphased dosage G_j = xL_j + xR_j is the sum of two independent haplotype
#  chains, each a fastPHASE HMM (K clusters, transition Q_hap, allele-1 prob
#  theta[j,k]). The marginal law of the dosage sequence is therefore ITSELF an
#  HMM, but its hidden state is the (ordered) PAIR of clusters (a, b):
#     * initial:     init_pair(a,b) = init_hap(a) init_hap(b)
#     * transition:  Q_pair = Q_hap (Kronecker) Q_hap   (two chains recombine
#                    independently)
#     * emission:    P(G_j=g | a,b) = convolution of Bernoulli(theta[j,a]) and
#                    Bernoulli(theta[j,b]):
#                      P(0)=(1-ta)(1-tb), P(1)=ta(1-tb)+(1-ta)tb, P(2)=ta tb.
#  Feeding this K^2-state pair HMM to hmm_knockoffs (with the OBSERVED dosages as
#  E=3 observations) yields a knockoff dosage that is exact for the true diploid
#  law -- WITHOUT phasing (Sesia, Sabatti & Candes 2019, genotype knockoffs).
#  Cost is O(N p K^4): the price of not phasing. Route 2 below (phased) is
#  O(N p K^2) but needs haplotypes.
# ---------------------------------------------------------------------------

def _pair_indices(K):
    """Ordered-pair index maps: state s = a*K + b -> (A[s]=a, B[s]=b)."""
    A = np.repeat(np.arange(K), K)      # L-haplotype cluster per pair-state
    B = np.tile(np.arange(K), K)        # R-haplotype cluster per pair-state
    return A, B


def build_genotype_pair_hmm(init_p_hap, Q_hap, theta):
    """
    Assemble the K^2-state genotype (pair) HMM from haplotype-level parameters.

    Args:
        init_p_hap: [K] haplotype initial cluster distribution.
        Q_hap: [p-1, K, K] haplotype transition matrices.
        theta: [p, K] allele-1 emission probabilities per position and cluster.

    Returns:
        (init_pair [K^2], Q_pair [p-1, K^2, K^2], emission_pair [p, 3, K^2]) in
        the layout hmm_knockoffs consumes (E=3 dosage emissions).
    """
    init_p_hap = np.asarray(init_p_hap, dtype=np.float64)
    Q_hap = np.asarray(Q_hap, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    K = init_p_hap.shape[0]
    p = theta.shape[0]
    A, B = _pair_indices(K)

    init_pair = init_p_hap[A] * init_p_hap[B]
    init_pair = init_pair / init_pair.sum()
    # Q_pair[j] = kron(Q_hap[j], Q_hap[j]); kron indexes (a*K+b, c*K+d)=Q[a,c]Q[b,d]
    Q_pair = np.stack([np.kron(Q_hap[j], Q_hap[j]) for j in range(p - 1)])
    ta = theta[:, A]                                  # [p, K^2]
    tb = theta[:, B]                                  # [p, K^2]
    emission_pair = np.stack([(1 - ta) * (1 - tb),
                              ta * (1 - tb) + (1 - ta) * tb,
                              ta * tb], axis=1)        # [p, 3, K^2]
    return init_pair, Q_pair, emission_pair


def fit_genotype_hmm(G, K, n_iter=25, tol=1e-4, pseudocount=1.0, theta_pseudo=1.0,
                     seed=0, init_p_hap=None, Q_hap=None, theta=None, verbose=False):
    """
    Fit the diploid genotype HMM from UNPHASED dosages by constrained Baum-Welch
    (fastPHASE-style genotype EM).

    The E-step is an ordinary forward-backward on the K^2-state pair HMM built
    from the current haplotype parameters. The M-step maps the pair-state
    posteriors back to HAPLOTYPE-level parameters (the constraint that makes this
    the diploid model rather than a generic K^2-state HMM):
      * init/transition: pool the L- and R-haplotype marginals of the pair
        posteriors (haplotypes are exchangeable);
      * emission theta[j,k]: expected number of allele-1 emissions attributed to
        cluster k, where a heterozygous dosage (g=1) is split between the two
        clusters in proportion to their current allele probabilities.

    Args:
        G: [N, p] unphased dosage matrix in {0,1,2}.
        K: number of haplotype clusters.
        n_iter, tol: EM stopping controls.
        pseudocount: smoothing for init/transition counts.
        theta_pseudo: smoothing for the emission (keeps theta in (0,1)).
        seed: RNG seed for initialization.
        init_p_hap, Q_hap, theta: optional warm-start haplotype parameters.
        verbose: print the log-likelihood trace.

    Returns:
        dict: 'init_p_hap' [K], 'Q_hap' [p-1,K,K], 'theta' [p,K],
        'loglik' (per-iteration), 'n_iter', 'K'.
    """
    G = np.asarray(G).astype(np.int64)
    N, p = G.shape
    rng = np.random.default_rng(seed)
    A, B = _pair_indices(K)

    if init_p_hap is None:
        init_p_hap = rng.dirichlet(np.ones(K))
    else:
        init_p_hap = np.asarray(init_p_hap, dtype=np.float64).copy()
    if Q_hap is None:
        Q_hap = np.empty((p - 1, K, K), dtype=np.float64)
        base = np.ones((K, K)) + (K * 2.0) * np.eye(K)   # self-transition bias
        for j in range(p - 1):
            for a in range(K):
                Q_hap[j, a] = rng.dirichlet(base[a])
    else:
        Q_hap = np.asarray(Q_hap, dtype=np.float64).copy()
    if theta is None:
        theta = rng.uniform(0.05, 0.95, size=(p, K))
    else:
        theta = np.asarray(theta, dtype=np.float64).copy()

    loglik_trace = []
    prev_ll = -np.inf
    for it in range(n_iter):
        init_pair, Q_pair, emission_pair = build_genotype_pair_hmm(init_p_hap, Q_hap, theta)
        gamma, xi_sum, ll = _forward_backward_scaled(G, init_pair, Q_pair, emission_pair)
        loglik_trace.append(ll)
        if verbose:
            print(f"  [fit_genotype_hmm] iter {it:3d}  loglik={ll:.4f}")

        # --- M-step: initial haplotype distribution (pool L and R marginals) ---
        g0 = gamma[0].sum(0)                              # [K^2]
        occ = 0.5 * (np.bincount(A, g0, minlength=K)
                     + np.bincount(B, g0, minlength=K)) + pseudocount
        init_p_hap = occ / occ.sum()

        # --- M-step: haplotype transitions from pair-transition posteriors ---
        Qn = np.empty((p - 1, K, K), dtype=np.float64)
        for j in range(p - 1):
            xij = xi_sum[j].reshape(K, K, K, K)          # [a, b, c, d]
            L = xij.sum(axis=(1, 3))                      # L: a -> c
            R = xij.sum(axis=(0, 2))                      # R: b -> d
            tc = L + R + pseudocount
            Qn[j] = tc / tc.sum(1, keepdims=True)
        Q_hap = Qn

        # --- M-step: emission theta (allele-1 responsibilities per cluster) ---
        theta_new = np.empty((p, K), dtype=np.float64)
        for j in range(p):
            ta = theta[j][A]
            tb = theta[j][B]
            het = ta * (1 - tb) + (1 - ta) * tb          # P(g=1 | a,b) numerator
            # E[xL=1 | g=1, a, b] = ta(1-tb) / het ; xR gets the complement
            exL1 = np.where(het > 1e-12, ta * (1 - tb) / np.maximum(het, 1e-12), 0.5)
            g = G[:, j]                                   # [N]
            gam = gamma[j]                                # [N, K^2]
            ExL = (g[:, None] == 1) * exL1[None, :] + (g[:, None] == 2) * 1.0
            ExR = g[:, None] - ExL                        # since xL + xR = g
            wLs = (gam * ExL).sum(0)                      # [K^2] allele-1 (L)
            wRs = (gam * ExR).sum(0)                      # [K^2] allele-1 (R)
            gs = gam.sum(0)                               # [K^2] total mass
            num = np.bincount(A, wLs, minlength=K) + np.bincount(B, wRs, minlength=K)
            den = np.bincount(A, gs, minlength=K) + np.bincount(B, gs, minlength=K)
            theta_new[j] = (num + theta_pseudo) / (den + 2 * theta_pseudo)
        theta = np.clip(theta_new, 1e-4, 1 - 1e-4)

        if it > 0 and ll - prev_ll < tol * N:
            break
        prev_ll = ll

    return {'init_p_hap': init_p_hap, 'Q_hap': Q_hap, 'theta': theta,
            'loglik': loglik_trace, 'n_iter': len(loglik_trace), 'K': K}


def genotype_hmm_knockoffs(G, K=8, M=1, n_em_iter=25, seed=0, params=None,
                           return_params=False, verbose=False):
    """
    Route 1 (exact, unphased): knockoff dosages from the diploid genotype HMM.

    Fits (or accepts) haplotype-level parameters, assembles the K^2-state pair
    HMM, and draws M knockoff dosage copies. Exact for the true diploid law.

    Args:
        G: [N, p] unphased dosage matrix in {0,1,2}.
        K: haplotype clusters (cost is O(N p K^4) -- keep modest).
        M: number of knockoff draws.
        n_em_iter: EM iterations (ignored if params given).
        seed: RNG seed.
        params: optional {'init_p_hap','Q_hap','theta'} to skip fitting.
        return_params: also return the (fitted/supplied) haplotype params.
        verbose: forwarded to fit_genotype_hmm.

    Returns:
        draws [M, N, p] int knockoff dosages (+ params if requested).
    """
    G = np.asarray(G).astype(np.int64)
    N, p = G.shape
    if params is None:
        params = fit_genotype_hmm(G, K=K, n_iter=n_em_iter, seed=seed, verbose=verbose)
    init_pair, Q_pair, emission_pair = build_genotype_pair_hmm(
        params['init_p_hap'], params['Q_hap'], params['theta'])
    draws = np.empty((M, N, p), dtype=np.int64)
    for m in range(M):
        draws[m] = hmm_knockoffs(G, init_pair, Q_pair, emission_pair, seed=seed + 1000 + m)
    if return_params:
        return draws, params
    return draws


# ---------------------------------------------------------------------------
#  Phased haplotype knockoffs (Route 2): generate knockoff diploid genomes
#
#  When phase is available, the cheaper and equally exact construction is to
#  knock off the two haplotypes directly. Fit a single haplotype HMM (E=2) to the
#  pooled 2N haplotypes, draw x~L conditional on xL and x~R conditional on xR
#  (same fitted model, independent randomness), and set G~ = x~L + x~R. This is a
#  valid knockoff of G because the simultaneous swap of BOTH haplotype systems --
#  under which each is individually exchangeable and the two are independent --
#  induces exactly the genotype swap G_j <-> G~_j and preserves the joint law.
#  Cost O(N p K^2). The phased knockoffs x~L, x~R are also what the two-channel
#  hapmixQTL ASE model needs (signed indicator s = xL - xR).
# ---------------------------------------------------------------------------

def haplotype_hmm_knockoffs(xL, xR, K=8, M=1, n_em_iter=25, seed=0, params=None,
                            return_phased=False, return_params=False, verbose=False):
    """
    Route 2 (exact, phased): knockoff dosages from phased haplotype knockoffs.

    Args:
        xL, xR: [N, p] phased haplotype allele matrices in {0,1} (the two
            haplotypes per individual). xL + xR is the dosage.
        K: haplotype clusters.
        M: number of knockoff draws.
        n_em_iter: EM iterations for the haplotype HMM (ignored if params given).
        seed: RNG seed.
        params: optional {'init_p','Q','emission_p'} haplotype HMM (E=2) to skip
            fitting.
        return_phased: also return the phased knockoffs (x~L, x~R) per draw.
        return_params: also return the haplotype HMM params.
        verbose: forwarded to fit_hmm.

    Returns:
        draws [M, N, p] int knockoff dosages. If return_phased, also
        (xkL [M,N,p], xkR [M,N,p]); if return_params, also the params dict.
    """
    xL = np.asarray(xL).astype(np.int64)
    xR = np.asarray(xR).astype(np.int64)
    N, p = xL.shape
    if params is None:
        H = np.concatenate([xL, xR], axis=0)             # [2N, p] pooled haplotypes
        params = fit_hmm(H, K=K, E=2, n_iter=n_em_iter, seed=seed, verbose=verbose)
    ip, Q, em = params['init_p'], params['Q'], params['emission_p']

    draws = np.empty((M, N, p), dtype=np.int64)
    xkL_all = np.empty((M, N, p), dtype=np.int64) if return_phased else None
    xkR_all = np.empty((M, N, p), dtype=np.int64) if return_phased else None
    for m in range(M):
        xkL = hmm_knockoffs(xL, ip, Q, em, seed=seed + 2000 + 2 * m)
        xkR = hmm_knockoffs(xR, ip, Q, em, seed=seed + 2000 + 2 * m + 1)
        draws[m] = xkL + xkR
        if return_phased:
            xkL_all[m], xkR_all[m] = xkL, xkR

    out = [draws]
    if return_phased:
        out.append((xkL_all, xkR_all))
    if return_params:
        out.append(params)
    return out[0] if len(out) == 1 else tuple(out)


# ---------------------------------------------------------------------------
#  Chromosome-coherent HMM knockoff generation (dispatcher)
#
#  The knockoff of a whole-chromosome genotype vector is, by the model-X swap
#  property, also a valid knockoff for any subset of variants (marginalizing the
#  others preserves exchangeability of the retained coordinates). So we fit ONE
#  HMM per chromosome, draw M knockoff copies of the entire chromosome, and slice
#  each gene's cis-window out of them. Two genes with overlapping windows then
#  share the SAME knockoff values on the shared variants -- the "coherent"
#  property a per-gene generator cannot provide, and the prerequisite for the
#  per-gene knockoff p-values and the overlapping-gene analysis.
# ---------------------------------------------------------------------------

def chromosome_hmm_knockoffs(G=None, K=8, M=1, E=3, n_em_iter=25, seed=0,
                             params=None, method='genotype', xL=None, xR=None,
                             return_params=False, return_phased=False,
                             verbose=False):
    """
    Fit an HMM on one chromosome and draw M coherent knockoff dosage copies.

    Args:
        G: [N, p] dosage-state matrix (needed for method 'genotype'/'single_chain').
        K: number of latent haplotype clusters.
        M: number of knockoff draws.
        E: emission categories for method='single_chain' (3 for dosages).
        n_em_iter: EM iterations for the fit (ignored if params given).
        seed: RNG seed (fit init + per-draw seeds derived from it).
        params: optional pre-fit parameters (format depends on method).
        method: which generator --
            'genotype'     : Route 1, exact diploid pair-state HMM (default);
                             params = {'init_p_hap','Q_hap','theta'}.
            'haplotype'    : Route 2, phased haplotype knockoffs; requires xL, xR;
                             params = {'init_p','Q','emission_p'} (E=2).
            'single_chain' : approximate single K-state chain with a free E=3
                             emission (cheapest, O(N p K^2), NOT the exact diploid
                             law); params = {'init_p','Q','emission_p'}.
        xL, xR: [N, p] phased haplotypes for method='haplotype'.
        return_params: also return the fitted/supplied parameters.
        return_phased: (method='haplotype' only) also return the phased knockoffs
            (x~L, x~R) as [M, N, p] each -- needed by the two-channel hapmixQTL
            path, which reconstructs both the ASE (x~L - x~R) and total
            (x~L + x~R) channels from the SAME coherent knockoff haplotypes.
        verbose: forwarded to the fitter.

    Returns:
        draws: [M, N, p] integer knockoff dosages, coherent across the whole
        chromosome. If return_phased, also (x~L, x~R). If return_params, also the
        params dict (order: draws, [phased], [params]).
    """
    if method == 'genotype':
        if return_phased:
            raise ValueError("return_phased is only supported for method='haplotype'")
        return genotype_hmm_knockoffs(G, K=K, M=M, n_em_iter=n_em_iter, seed=seed,
                                      params=params, return_params=return_params,
                                      verbose=verbose)
    elif method == 'haplotype':
        assert xL is not None and xR is not None, \
            "method='haplotype' requires phased xL and xR"
        return haplotype_hmm_knockoffs(xL, xR, K=K, M=M, n_em_iter=n_em_iter,
                                       seed=seed, params=params,
                                       return_params=return_params,
                                       return_phased=return_phased, verbose=verbose)
    elif method == 'single_chain':
        G = np.asarray(G)
        N, p = G.shape
        if params is None:
            params = fit_hmm(G, K=K, E=E, n_iter=n_em_iter, seed=seed, verbose=verbose)
        ip, Q, em = params['init_p'], params['Q'], params['emission_p']
        draws = np.empty((M, N, p), dtype=np.int64)
        for m in range(M):
            draws[m] = hmm_knockoffs(G, ip, Q, em, seed=seed + 1000 + m)
        return (draws, params) if return_params else draws
    else:
        raise ValueError(f"unknown method: {method}")


# ---------------------------------------------------------------------------
#  Importance statistics and the knockoff filter
# ---------------------------------------------------------------------------

def pip_importance(pip_t, p):
    """
    Split a length-2p PIP vector from a SuSiE fit on [X, X_knockoff] into the
    per-variant importance contrast W_j = PIP_j - PIP_{knockoff(j)}.

    Args:
        pip_t: length-2p array/tensor of PIPs; columns 0:p are originals,
               columns p:2p are the matching knockoffs (same order).
        p: number of original variants.

    Returns:
        W: numpy array [p], antisymmetric knockoff statistic.
    """
    pip = np.asarray(pip_t, dtype=np.float64)
    assert pip.shape[0] == 2 * p, f"expected 2p={2*p} PIPs, got {pip.shape[0]}"
    return pip[:p] - pip[p:]


def gene_level_W(pip_t, p, kind='max'):
    """
    Gene-level knockoff statistic from a SuSiE fit on [X, X_knockoff].

    This tests a FIXED hypothesis -- H_g: gene g has no cis signal (Y_g is
    conditionally independent of every cis variant) -- which does NOT depend on
    which credible sets SuSiE happens to form. That fixedness is what makes the
    statistic a valid model-X knockoff statistic: swapping every original with
    its knockoff exchanges the two blocks, so W_g -> -W_g exactly. (This is the
    property the CS-level statistic fails; see the swap test in the test suite.)

        W_g = f(PIP over original block) - f(PIP over knockoff block)

    with f = max (default; the strongest single cis signal) or sum.

    Args:
        pip_t: length-2p PIPs (originals 0:p, knockoffs p:2p, same order).
        p: number of original variants.
        kind: 'max' or 'sum'.

    Returns:
        W_g: float, antisymmetric gene-level statistic.
    """
    pip = np.asarray(pip_t, dtype=np.float64)
    assert pip.shape[0] == 2 * p, f"expected 2p={2*p} PIPs, got {pip.shape[0]}"
    orig, knock = pip[:p], pip[p:]
    if kind == 'max':
        return float(orig.max() - knock.max())
    elif kind == 'sum':
        return float(orig.sum() - knock.sum())
    else:
        raise ValueError(f"unknown kind: {kind}")


def knockoff_threshold(W, q=0.1, offset=1):
    """
    Knockoff / knockoff+ threshold for target FDR q.

    Args:
        W: array [p] of antisymmetric statistics
        q: target FDR level
        offset: 1 for knockoff+ (adds 1 to the negative count; controls FDR
            exactly), 0 for the basic knockoff filter (controls a modified FDR).

    Returns:
        tau: smallest threshold t > 0 meeting the FDP-estimate constraint, or
             +inf if none exists (=> select nothing).
    """
    W = np.asarray(W, dtype=np.float64)
    ts = np.sort(np.abs(W[W != 0]))
    if ts.size == 0:
        return np.inf
    for t in ts:
        num = offset + np.sum(W <= -t)
        den = max(1, np.sum(W >= t))
        if num / den <= q:
            return t
    return np.inf


def selected_variants(W, q=0.1, offset=1):
    """Return boolean mask [p] of originals passing the knockoff+ filter."""
    tau = knockoff_threshold(W, q=q, offset=offset)
    W = np.asarray(W, dtype=np.float64)
    return W >= tau


# ---------------------------------------------------------------------------
#  Credible-set level filtering
# ---------------------------------------------------------------------------

def cs_level_W(res, p, stat='pip'):
    """
    EXPERIMENTAL -- NOT a valid FDR-controlled knockoff statistic. Do not use
    for FDR claims. Retained as a calibration/exploration score only.

    Why it is invalid: this contrasts a credible set's *original* members
    against their knockoff counterparts, but credible sets are constructed from
    the observed data and the specific knockoff draw and the original/knockoff
    blocks are treated asymmetrically (original-only extraction). Under the
    model-X swap (exchange every X_j with its knockoff), a valid statistic must
    map to its negation via a deterministic hypothesis correspondence; this one
    does not -- the credible set for a real signal *disappears* rather than
    negating (see tests/test_knockoffs.py::TestSwapEquivariance). Hence the
    pooled negatives are not valid negative controls and any FDR derived from
    them is unjustified. Use gene_level_W (Path A) for a valid statistic.

    Compute a per-credible-set contrast from a SuSiE fit on the augmented
    [X, X_knockoff] design: for each credible set, contrast its original members'
    importance against that of their knockoff counterparts.

    Args:
        res: SuSiE result dict (from susie.susie on the augmented design). Must
             contain res['sets']['cs'] (dict Lk -> index tensor/array) and
             res['pip'] (length 2p) or res['alpha'] (L x 2p).
        p: number of original variants.
        stat: 'pip' (default) or 'max_alpha' importance per column.

    Returns:
        list of dicts, one per credible set that has >=1 original member:
          {'cs_id': str, 'orig_idx': np.array of original-variant indices (0..p-1),
           'W': float}
        where W = sum_{j in CS orig} Z_j - sum_{j in CS orig} Z_{knockoff(j)}.
    """
    sets = res.get('sets', None)
    if sets is None or sets.get('cs', None) is None:
        return []

    if stat == 'pip':
        Z = np.asarray(res['pip'], dtype=np.float64)
    elif stat == 'max_alpha':
        alpha = np.asarray(res['alpha'], dtype=np.float64)  # [L, 2p]
        Z = alpha.max(axis=0)
    else:
        raise ValueError(f"unknown stat: {stat}")

    out = []
    for cs_id, members in sets['cs'].items():
        members = np.asarray(members).ravel().astype(int)
        orig = members[members < p]
        if orig.size == 0:
            continue  # CS made only of knockoff columns -> not a real signal
        W = float(Z[orig].sum() - Z[orig + p].sum())
        out.append({'cs_id': str(cs_id), 'orig_idx': orig, 'W': W})
    return out


def filter_credible_sets(res, p, q=0.1, stat='pip', offset=1):
    """
    Apply the knockoff+ filter at the credible-set level.

    Args:
        res: SuSiE fit on the augmented design.
        p: number of original variants.
        q: target per-gene CS-level FDR.
        stat: importance statistic for cs_level_W.
        offset: 1 for knockoff+ (recommended).

    Returns:
        dict with:
          'kept': list of CS dicts (from cs_level_W) with W >= tau, sorted by
                  decreasing W (calibrated confidence order),
          'all': the full list of CS dicts with their W,
          'tau': the threshold,
          'estimated_fdp': the knockoff FDP estimate at tau (diagnostic).
    """
    cs = cs_level_W(res, p, stat=stat)
    if not cs:
        return {'kept': [], 'all': [], 'tau': np.inf, 'estimated_fdp': np.nan}

    W = np.array([c['W'] for c in cs], dtype=np.float64)
    tau = knockoff_threshold(W, q=q, offset=offset)
    kept = sorted([c for c in cs if c['W'] >= tau], key=lambda c: -c['W'])

    if np.isfinite(tau):
        num = offset + np.sum(W <= -tau)
        den = max(1, np.sum(W >= tau))
        est_fdp = num / den
    else:
        est_fdp = np.nan
    return {'kept': kept, 'all': cs, 'tau': tau, 'estimated_fdp': est_fdp}


def pooled_cs_qvalues(W_all, offset=1):
    """
    Genome-wide knockoff q-value for each credible set from the POOLED W.

    Per-gene knockoffs are generated separately, but FDR is controlled by
    pooling the CS-level statistics across all genes and thresholding once (see
    docs/knockoff_susie_design.md). This is valid despite wildly different W
    magnitudes across genes because the knockoff guarantee depends only on the
    sign-symmetry of null statistics, not on their scale: each gene's null CSs
    are independently sign-symmetric, so the pooled null set is too.

    The q-value for a CS with statistic W_k is the smallest achievable knockoff+
    FDP estimate over all thresholds t <= W_k:

        qval(W_k) = min_{t <= W_k} (offset + #{W_all <= -t}) / max(1, #{W_all >= t})

    monotonized so q is non-increasing in W. Selecting CSs with qval <= q yields
    a set with genome-wide FDR <= q. Non-positive W (knockoff beats original)
    gets qval = 1.

    Args:
        W_all: array [m] of CS-level statistics pooled across ALL genes.
        offset: 1 for knockoff+ (recommended), 0 for basic knockoff.

    Returns:
        qval: array [m], the knockoff q-value aligned to W_all.
    """
    W = np.asarray(W_all, dtype=np.float64)
    m = W.shape[0]
    fdp_at = np.ones(m, dtype=np.float64)
    if m == 0:
        return fdp_at

    # qval(W_k) = min over thresholds t <= W_k of FDP(t). As W_k grows, the set
    # {t <= W_k} grows, so the achievable min FDP can only decrease -> q is
    # non-increasing in W. Sweep positive thresholds from SMALLEST to LARGEST,
    # keeping a running min of FDP, and assign each CS the running min at t=W_k.
    pos_idx = np.where(W > 0)[0]
    order = pos_idx[np.argsort(W[pos_idx])]  # ascending W among positives
    best = np.inf
    for idx in order:
        t = W[idx]
        num = offset + np.sum(W <= -t)
        den = max(1, np.sum(W >= t))
        fdp = num / den
        best = min(best, fdp)
        fdp_at[idx] = min(best, 1.0)
    # non-positive W (knockoff beats original) keep qval = 1
    return fdp_at


def augmented_susie_fit(susie_fn, X_t, y_t, Xk_t, L, **susie_kwargs):
    """
    Fit SuSiE on the augmented design [X, X_knockoff] and return (res, p).

    Originals occupy columns 0:p, knockoffs p:2p (same variant order), the layout
    gene_level_W / pip_importance expect.

    NOTE on L: pass the intended number of biological effects, NOT 2*L. Adding
    knockoff columns doubles the *candidate* variables, not the true number of
    effects; original and knockoff variables are meant to compete for the same L
    single-effect slots. (An earlier draft doubled L; external review correctly
    flagged that this is not implied by knockoff theory and may encourage the
    model to spend components duplicating a signal across the two blocks.)

    Args:
        susie_fn: the susie.susie callable (injected to avoid an import cycle).
        X_t:  [N, p] original design (residualized).
        y_t:  [N, 1] response.
        Xk_t: [N, p] knockoff design (same rows/samples as X_t).
        L: number of single effects for the augmented fit.
        susie_kwargs: forwarded to susie_fn.

    Returns:
        (res, p): the SuSiE result dict and the original-variant count p.
    """
    assert X_t.shape == Xk_t.shape, "knockoff must match original shape"
    p = X_t.shape[1]
    X_aug = torch.cat([X_t, Xk_t], dim=1)
    res = susie_fn(X_aug, y_t, L=L, **susie_kwargs)
    return res, p


# ---------------------------------------------------------------------------
#  eGene-level FDR (Path A): valid knockoff selection of genes
# ---------------------------------------------------------------------------

def select_egenes(gene_ids, W_per_draw, q=0.1, offset=1):
    """
    Select eGenes at target FDR q from gene-level knockoff statistics, with
    Ren-Barber (2024) e-value derandomization across knockoff draws.

    Each gene has a FIXED hypothesis H_g (no cis signal), so its statistic W_g
    (from gene_level_W) is a valid antisymmetric knockoff statistic. With a
    single draw the knockoff+ filter controls FDR directly; with multiple draws
    we must NOT average the W's (that can break the null sign distribution and
    the threshold estimator -- external review's point 7). Instead, per
    Ren-Barber, convert each draw's knockoff+ selection into a per-gene e-value
    and average the e-values, then apply e-BH. This provably controls FDR while
    removing run-to-run selection variance.

    Args:
        gene_ids: list of m gene identifiers.
        W_per_draw: array [n_draws, m] of gene-level statistics, one row per
            knockoff draw (columns aligned to gene_ids).
        q: target FDR.
        offset: knockoff+ offset (1 recommended).

    Returns:
        dict with:
          'selected': list of selected gene_ids,
          'evalues': array [m] averaged e-values (aligned to gene_ids),
          'n_draws': number of draws.
    """
    W_per_draw = np.atleast_2d(np.asarray(W_per_draw, dtype=np.float64))
    n_draws, m = W_per_draw.shape
    assert m == len(gene_ids), "W columns must align to gene_ids"

    # Per-draw e-values (knockoff-as-eBH construction, Ren & Barber 2024):
    #   for draw with knockoff+ threshold tau, a gene's e-value is
    #   m * 1{W_g >= tau} / (offset + #{W <= -tau}); 0 if no finite tau.
    E = np.zeros((n_draws, m), dtype=np.float64)
    for d in range(n_draws):
        W = W_per_draw[d]
        tau = knockoff_threshold(W, q=q, offset=offset)
        if not np.isfinite(tau):
            continue
        n_neg = offset + np.sum(W <= -tau)
        sel = W >= tau
        E[d, sel] = m / n_neg
    ebar = E.mean(axis=0)  # average e-values across draws

    # e-BH at level q: sort e-values descending, find largest k with
    # e_(k) >= m / (q k), select the top k.
    order = np.argsort(-ebar)
    selected_idx = []
    for k, idx in enumerate(order, start=1):
        if ebar[idx] >= m / (q * k):
            selected_idx = order[:k]
    selected = [gene_ids[i] for i in selected_idx]
    return {'selected': selected, 'evalues': ebar, 'n_draws': n_draws}


def select_egenes_qvalue(gene_ids, W_per_draw, q=0.1, offset=0, aggregate='median'):
    """
    Select eGenes by a CALIBRATED per-gene knockoff q-value (default path).

    Validation phase 3 (docs/knockoff_susie_design.md) showed that the e-value
    (e-BH) derandomization in select_egenes is over-conservative and unstable
    near the detection floor -- it collapses power to zero when individual draws
    select nothing. The single-draw knockoff q-value, in contrast, is roughly
    CALIBRATED (realized FDR ~ nominal q at operating q), which is the project's
    goal (calibration, not merely control). This function is that calibrated
    path, with an optional gentle multi-draw stabilizer that aggregates the
    CALIBRATED quantity (the per-gene q-value) across draws -- NOT e-values --
    so seed-stability does not cost calibration.

    offset defaults to 0 (plain knockoff): FDP_hat = #{W<=-t}/#{W>=t} is an
    approximately unbiased FDP estimate. offset=1 (knockoff+) instead guarantees
    FDR<=q at the cost of conservatism.

    Per draw d, the genome-wide-pooled q-value for gene g is
        q_g^(d) = pooled_cs_qvalues(W^(d))[g]
    (the same pooled monotone knockoff q-value used elsewhere, here over the
    per-gene W vector). Across draws these are combined by:
        aggregate='median' : q_g = median_d q_g^(d)   (default; robust stabilizer)
        aggregate='mean'   : q_g = mean_d q_g^(d)
        aggregate='none'   : use draw 0 only (pure single-draw)
    A gene is selected iff its aggregated q-value <= q.

    Args:
        gene_ids: list of m gene identifiers.
        W_per_draw: array [n_draws, m] of gene-level statistics.
        q: target FDR.
        offset: 0 (calibrated, default) or 1 (knockoff+ control).
        aggregate: 'median' | 'mean' | 'none'.

    Returns:
        dict: 'selected' (list of gene_ids), 'qvalues' (array [m], aligned),
              'n_draws'.
    """
    W_per_draw = np.atleast_2d(np.asarray(W_per_draw, dtype=np.float64))
    n_draws, m = W_per_draw.shape
    assert m == len(gene_ids), "W columns must align to gene_ids"

    if aggregate == 'none':
        qmat = pooled_cs_qvalues(W_per_draw[0], offset=offset)[None, :]
    else:
        qmat = np.vstack([pooled_cs_qvalues(W_per_draw[d], offset=offset)
                          for d in range(n_draws)])
    if aggregate == 'mean':
        qagg = qmat.mean(axis=0)
    else:  # 'median' or 'none' (single row)
        qagg = np.median(qmat, axis=0)

    selected = [gene_ids[i] for i in range(m) if qagg[i] <= q]
    return {'selected': selected, 'qvalues': qagg, 'n_draws': n_draws}


# ---------------------------------------------------------------------------
#  Per-gene knockoff p-values (step 2) -- the primitive for empirical-Bayes
#  interval-pi0 calibration (step 3). With M coherent knockoff draws, the
#  per-gene "how often does the knockoff beat the real gene" statistic is an
#  (approximately) uniform-null p-value. It has a KNOWN null (uniform), which is
#  exactly what the two-groups / local-FDR calibration needs, and -- because it
#  is computed within each gene from that gene's own knockoffs -- it is robust to
#  the pooled-f_0 contamination that a genome-wide knockoff-score density suffers
#  (alt-gene knockoffs are suppressed by the winning real signal).
# ---------------------------------------------------------------------------

def per_gene_pvalues(W_per_draw, offset=1):
    """
    Per-gene knockoff p-value from M draws of the gene-level statistic.

    W_per_draw[m, g] = W_g^(m) = R_g^(m) - K_g^(m) (real max PIP minus knockoff
    max PIP in draw m). Under the null H_g (gene has no cis signal) gene_level_W
    is swap-antisymmetric, so each W_g^(m) is symmetric about 0 (its sign is a
    coin flip). The knockoff p-value counts how often the knockoff wins:

        p_g = (offset + #{m: W_g^(m) <= 0}) / (offset + M)

    VALIDITY, not marginal uniformity. This is a valid (super-uniform /
    CONSERVATIVE) p-value: P(p_g <= a | H_g) <= a. It is NOT marginally uniform.
    Under the model-X null each draw m uses an INDEPENDENTLY generated knockoff
    copy, so the swap X_j <-> X~_j^(m) can be applied per draw independently; that
    flips the sign of W_g^(m) alone and leaves the joint law invariant. Invariance
    under flipping any subset of the M signs forces the sign vector to be uniform
    on {+-1}^M, hence #{W_g^(m) <= 0} ~ Binomial(M, 1/2) EXACTLY (the shared real
    genotypes do NOT induce dependence in the signs). The count concentrates near
    M/2, so the p-value piles up near 0.5 and its LEFT TAIL is much lighter than
    Uniform's -- exactly the property that keeps FDR control safe (and makes it
    low-power; see the resolution note on select_egenes_pvalue). Because the null
    is the EXACT Binomial(M,1/2), the step-3 calibration uses that known null CDF
    directly (calibrated_qvalues) rather than the uniform-null Storey formula.
    Resolution is 1/(M+1), so use M >= 10-20 at a minimum; small-q BH selection
    needs far more (see select_egenes_pvalue).

    Args:
        W_per_draw: array [n_draws, n_genes] of gene-level statistics.
        offset: 1 (knockoff+, recommended) or 0.

    Returns:
        pvals: array [n_genes] in (0, 1].
    """
    W = np.atleast_2d(np.asarray(W_per_draw, dtype=np.float64))
    M, n = W.shape
    b = (W <= 0).sum(axis=0)                 # knockoff wins/ties per gene
    return (offset + b) / (offset + M)


def bh_select(pvals, q, dependence='prds'):
    """
    Benjamini-Hochberg selection at FDR q. Returns a boolean mask [n].

    ================  THE GENOME-WIDE DEPENDENCE QUESTION  ==================
    This is where the "overlapping-gene joint-sign" problem is actually resolved.
    Once each gene has a p-value with a KNOWN marginal null (per_gene_pvalues:
    super-uniform, exact Binomial), controlling eGene FDR across the genome is NO
    LONGER a novel knockoff-joint-sign theorem -- it is the classical problem of
    BH under DEPENDENCE, which is solved:

      * dependence='ind'  -- Benjamini-Hochberg 1995. Threshold q*i/n. Valid FDR
        control when the null p-values are independent.
      * dependence='prds' (default) -- Benjamini & Yekutieli 2001 proved plain BH
        (same q*i/n threshold) also controls FDR under POSITIVE REGRESSION
        DEPENDENCY ON THE NULL SET (PRDS). Overlapping eQTL genes share genotypes
        (and, with the coherent HMM generator, share knockoffs on shared
        variants), which induces POSITIVE association between their p-values --
        the PRDS regime. This is the pragmatic default and what the overlap-
        stress tests validate empirically.
      * dependence='arbitrary' -- Benjamini & Yekutieli 2001's distribution-free
        variant: threshold q*i/(n*c(n)) with c(n)=sum_{j=1}^n 1/j (the harmonic
        number). Controls FDR under ANY dependence, at the cost of the log-factor
        c(n) ~ ln n + 0.577 (here ~5-6x more conservative genome-wide). Use when
        you are unwilling to assume PRDS.

    'ind' and 'prds' share the identical BH threshold; the only reason to
    distinguish them is documentary honesty about which theorem you are invoking.

    Args:
        pvals: array [n] of (super-)uniform-null p-values.
        q: target FDR.
        dependence: 'prds' (default) / 'ind' (same threshold, BH); or 'arbitrary'
            (Benjamini-Yekutieli harmonic correction, valid under any dependence).

    Returns:
        boolean mask [n].
    """
    p = np.asarray(pvals, dtype=np.float64)
    n = p.shape[0]
    if n == 0:
        return np.zeros(0, dtype=bool)
    if dependence == 'arbitrary':
        cn = np.sum(1.0 / np.arange(1, n + 1))     # harmonic number
    elif dependence in ('prds', 'ind'):
        cn = 1.0
    else:
        raise ValueError(f"unknown dependence: {dependence}")
    order = np.argsort(p)
    thresh = q * np.arange(1, n + 1) / (n * cn)
    passed = p[order] <= thresh
    if not passed.any():
        return np.zeros(n, dtype=bool)
    kmax = np.max(np.where(passed)[0])
    cutoff = p[order][kmax]
    return p <= cutoff


def select_egenes_pvalue(gene_ids, W_per_draw, q=0.1, offset=1, dependence='prds'):
    """
    eGene selection via per-gene knockoff p-values + Benjamini-Hochberg.

    This is the step-2 selection mode: compute a per-gene p-value with a KNOWN
    (approximately uniform) null from the M coherent draws (per_gene_pvalues),
    then select at FDR q with BH. Its output (`pvalues`) is also the input the
    step-3 interval-pi0 empirical-Bayes calibration consumes. `dependence`
    ('prds' default | 'ind' | 'arbitrary') selects the cross-gene dependence
    assumption of the BH step (see bh_select).

    RESOLUTION LIMIT (important for choosing M). The smallest attainable p-value
    is offset/(offset+M) = 1/(M+1) at offset=1. For BH at level q to make R
    discoveries, the R-th ordered p-value must clear q*R/n; a maximally-signalled
    gene (p_g = 1/(M+1)) can therefore only enter a rejection set of size R when
        1/(M+1) <= q * R / n   =>   M >= n / (q * R) - 1.
    The bound is governed by R, the number of JOINTLY discoverable genes:
      * Many true genes (R ~ n): the wall is cheap. n=100 all-signal, q=0.1
        needs only M >= n/(q*n) - 1 = 1/q - 1 = 9, so M=15 already selects them.
      * Few true genes (small R): the wall is expensive. n=100 with only R=5 true
        needs M >= 100/(0.1*5) - 1 = 199 draws before those 5 can be selected;
        M=15/30/60 select nothing at all.
    So direct BH is only practical when discoveries are plentiful. The per-gene
    p-value's primary role is as the primitive for the step-3 interval-pi0
    empirical-Bayes calibration, which pools the p-value DISTRIBUTION across genes
    and does not hit this hard per-gene 1/(M+1) BH wall.

    Returns:
        dict: 'selected' (gene_ids), 'pvalues' (array [n], aligned), 'n_draws'.
    """
    W = np.atleast_2d(np.asarray(W_per_draw, dtype=np.float64))
    n_draws, m = W.shape
    assert m == len(gene_ids), "W columns must align to gene_ids"
    pvals = per_gene_pvalues(W, offset=offset)
    mask = bh_select(pvals, q, dependence=dependence)
    selected = [gene_ids[i] for i in range(m) if mask[i]]
    return {'selected': selected, 'pvalues': pvals, 'n_draws': n_draws}


# ===========================================================================
#  STEP 3 -- Empirical-Bayes eGene calibration from the per-gene knockoff
#  statistics.  Three estimators, three literatures, one shared input.
#  ==========================================================================
#
#  INPUT.  For each gene g we have the counts of how often, across M coherent
#  knockoff draws, the KNOCKOFF beat (or tied) the real gene:
#         b_g = #{m : W_g^(m) <= 0},        b_g in {0, ..., M}.
#  Large b_g -> the knockoff usually wins -> the gene looks null. Small b_g ->
#  the real gene usually wins -> the gene looks like a true eGene.
#
#  THE ONE FACT EVERYTHING RESTS ON.  Under the null H_g ("gene g has no cis
#  signal") the sign of each draw's W is a fair, independent coin (per-draw
#  model-X swap; see per_gene_pvalues), so
#         b_g ~ Binomial(M, 1/2)   EXACTLY under H_g.
#  We therefore have a *known, discrete, non-uniform* null. Every estimator
#  below is just a different, well-studied way to turn "known symmetric null +
#  observed left-skew" into a calibrated false-discovery statement. They are
#  redundant ON PURPOSE: if they disagree, the knockoffs are misspecified and
#  none of the numbers should be trusted (that disagreement is the alarm).
#
#  WHICH ONE TO USE (short version).
#    * calibrated_qvalues  -> the SHIPPED q-value you select eGenes with. Most
#                             stable; needs a pi0 estimate (handled for you).
#    * mirror_fdp          -> a pi0-FREE cross-check that uses only the null's
#                             symmetry. If it disagrees with the q-values, stop.
#    * local_fdr_interval  -> per-gene posterior "prob this gene is null", with
#                             an explicit interval reflecting that pi0 is only
#                             partially identified. For interpretation, not the
#                             primary selector.
#  Full statistical tradeoffs are in each function's docstring.
# ===========================================================================


def _binom_half_pmf(M):
    """PMF of Binomial(M, 1/2) on b = 0..M, computed with stdlib lgamma (no
    scipy dependency). g0[b] = C(M,b) / 2^M."""
    from math import lgamma, log
    b = np.arange(M + 1)
    logc = (lgamma(M + 1)
            - np.array([lgamma(k + 1) for k in b])
            - np.array([lgamma(M - k + 1) for k in b]))
    return np.exp(logc - M * log(2.0))


def _counts_from_W(W_per_draw):
    """b_g = #{m: W_g^(m) <= 0} per gene, and M. Accepts [M, n] or [n]."""
    W = np.atleast_2d(np.asarray(W_per_draw, dtype=np.float64))
    M = W.shape[0]
    b = (W <= 0).sum(axis=0).astype(int)
    return b, M


def estimate_pi0_known_null(b, M, lam=None, grid=None):
    """
    Storey-type null-proportion estimate for the KNOWN discrete Binomial(M,1/2)
    null (Storey 2003; Storey, Taylor & Siegmund 2004), generalized off the
    uniform null.

    SUMMARY OF THE CHOICE.  pi0 is "what fraction of genes are truly null."
    Standard Storey assumes null p-values are Uniform(0,1); ours are not (they
    live on the Binomial grid), so the uniform formula would MIS-estimate pi0 and
    silently break calibration. Here we substitute the true null tail mass.

    HOW.  Pick a cut lam on the count axis in the "null-looking" region (large b,
    where true eGenes essentially never land). Everything above lam is treated as
    (almost) all null, so
         pi0_hat = [#genes with b > lam] / [n * P(Binomial(M,1/2) > lam)].
    The denominator is the EXACT null probability of landing above lam -- that is
    the only change from textbook Storey, and it is what makes the estimate
    honest for this non-uniform null.

    TRADEOFF IN lam.  Small lam (cut nearer the center M/2) uses more genes ->
    lower variance, but risks including true-signal mass -> DOWNWARD bias in pi0
    -> anti-conservative (too few nulls -> FDR under-stated). Large lam (deep in
    the right tail) is cleaner of signal -> less bias, but uses few genes ->
    high variance -- and PAST a point the null mass P(B>lam) itself is ~0, where
    the ratio 0/0 is pure noise and, worse, drops to 0 whenever no gene happens
    to exceed lam, biasing an averaged estimate DOWNWARD. This is the standard
    Storey pitfall. We therefore restrict the default cut grid to the region
    where the null still carries appreciable mass, P(B>lam) in [0.10, 0.50] --
    the analog of Storey's "lambda up to ~0.95, not to 1" discipline. Under a
    decreasing alternative true eGenes essentially never land at b > M/2, so this
    band is both signal-poor and mass-rich: the sweet spot. We average the
    estimate over that band (the smoother-of-Storey idea) and clip to [1/n, 1].

    Args:
        b: array [n] of per-gene knockoff-win counts.
        M: number of draws.
        lam: single cut (integer count). If None, uses `grid`.
        grid: iterable of cuts to average over. Default = the mass-rich right
            band {c : 0.10 <= P(Binom(M,1/2) > c) <= 0.50}.

    Returns:
        pi0_hat in (0, 1].
    """
    b = np.asarray(b, dtype=int)
    n = b.shape[0]
    if n == 0:
        return 1.0
    g0 = _binom_half_pmf(M)
    ccdf = np.array([g0[k + 1:].sum() for k in range(M + 1)])   # P(B > k)
    if lam is not None:
        cuts = [int(lam)]
    elif grid is not None:
        cuts = [int(c) for c in grid]
    else:
        # mass-rich, signal-poor band: 0.10 <= P(B>c) <= 0.50.
        cuts = [c for c in range(M) if 0.10 <= ccdf[c] <= 0.50]
        if not cuts:
            # tiny M: fall back to the single cut nearest the median.
            cuts = [int(np.clip(round(M / 2.0), 0, M - 1))]
    ests = []
    for c in cuts:
        denom = ccdf[c] * n
        if denom <= 0:
            continue
        ests.append((b > c).sum() / denom)
    if not ests:
        return 1.0
    return float(min(1.0, max(np.mean(ests), 1.0 / n)))


def null_cdf(b, M):
    """
    Left CDF of the knockoff count under the KNOWN null: F0(b) = P(Binomial(M,
    1/2) <= b). This is the object that replaces the uniform "p" everywhere the
    calibration touches Storey/BH machinery.

    SUMMARY OF THE CHOICE.  A p-value's whole job is "probability the null would
    look at least this extreme." Because our null is the exact Binomial(M,1/2)
    -- NOT Uniform(0,1) -- the correct such probability is the Binomial CDF, and
    plugging it in is what turns a raw knockoff vote into a *calibrated* tail
    probability. Using uniform-p here (the naive default) would be the single
    most common way to get the FDR wrong for this statistic.

    WHY IT'S EXACT, NOT APPROXIMATE.  b_g ~ Binomial(M,1/2) under H_g is exact
    (per-draw model-X swap), so F0 is exact; there is nothing to estimate. This
    is the payoff for having built a statistic with a known null.

    Args:
        b: array-like of counts (0..M).
        M: number of draws.

    Returns:
        F0: array of P(B <= b), same shape as b.
    """
    g0 = _binom_half_pmf(M)
    cdf = np.cumsum(g0)                       # cdf[k] = P(B <= k)
    b = np.asarray(b, dtype=int)
    return cdf[np.clip(b, 0, M)]


def calibrated_qvalues(W_per_draw, pi0='auto', offset=1, dependence='prds'):
    """
    SHIPPED eGene q-values: Storey q-values computed against the EXACT discrete
    Binomial(M,1/2) null instead of the uniform null.

    ===================  WHAT THIS BUYS YOU (plain language)  ================
    A q-value of 0.1 on a gene means: "if you accept every gene at least this
    knockoff-convincing, about 10% of the accepted genes are expected to be
    false eGenes." That is the calibrated FDR statement you actually want -- not
    "1.5-3x fewer false calls," but "5% of my calls are false, no more no less."
    Select eGenes by thresholding these q-values at your target FDR.

    ===================  HOW IT WORKS (the statistics)  =====================
    Two-groups / Storey FDR (Storey 2003; Storey, Taylor & Siegmund 2004) says
    the false-discovery proportion when you accept all genes with statistic at
    least as extreme as threshold t is estimated by

            FDP_hat(t) = pi0 * n * F0(t) / R(t),

      * F0(t) = expected FRACTION of NULL genes that reach t  -- here the EXACT
        Binomial CDF null_cdf (this is the only departure from textbook Storey,
        and the load-bearing one: textbook uses F0(t)=t, valid only for a
        uniform null, which we do NOT have);
      * R(t)  = OBSERVED number of genes reaching t;
      * pi0   = fraction of truly null genes (estimate_pi0_known_null).
    The q-value of a gene is the smallest FDP_hat over all thresholds at least as
    extreme as that gene (the usual monotone "min from the tail"), clipped to 1.

    ===================  THE TRADEOFFS YOU'RE ACCEPTING  ====================
    + Stability. It uses the CDF (a cumulative, well-estimated quantity), so it
      is far less noisy than a per-gene density ratio (contrast local_fdr_
      interval). This is why it is the SHIPPED selector.
    + Exactness of the null. No null to estimate; F0 is known.
    - It leans on pi0, which is only PARTIALLY identified (Genovese & Wasserman
      2004): only an upper bound on pi0 is ever learnable from the data. We
      default pi0 to a Storey estimate that is mildly conservative (tends to
      OVER-state pi0 -> OVER-state FDP -> slightly too few discoveries). Set
      pi0=1 for the most conservative, assumption-light version (valid FDR
      control, less power); set pi0='auto' (default) for the calibrated estimate;
      pass a float to pin it.
    - Discreteness. With M draws the statistic lives on M+1 grid points, so many
      genes tie and share a q-value, and the smallest reachable q is bounded
      (the 1/(M+1) resolution wall documented on select_egenes_pvalue). Larger M
      -> finer q-values. This is the discrete-null regime of Doehler, Durand &
      Roquain 2018 (their DiscreteFDR bounds are the exact-FDR refinement of the
      Storey estimate used here; we use the simpler Storey plug-in and validate
      calibration empirically in the tests).

    CROSS-GENE DEPENDENCE.  These q-values are the pooled genome-wide statement,
    so their FDR guarantee inherits the BH-under-dependence theory (see
    bh_select): the Storey/BH q-value controls FDR under independence or PRDS
    (Benjamini & Yekutieli 2001) -- the regime overlapping eQTL genes plausibly
    occupy (shared genotypes -> positively associated statistics). Set
    dependence='arbitrary' to multiply the q-values by the harmonic factor
    c(n)=sum 1/j for a distribution-free guarantee under ANY dependence (~5-6x
    more conservative genome-wide). 'prds' (default) and 'ind' leave q unscaled.

    Args:
        W_per_draw: [M, n] gene-level statistics (or [n] for a single draw).
        pi0: 'auto' (estimate), 1.0 / a float (pin), or 'one' (=1.0).
        offset: passed through for the count convention (kept for symmetry with
            per_gene_pvalues; does not change the ranking).
        dependence: 'prds' (default) / 'ind' (unscaled) or 'arbitrary'
            (Benjamini-Yekutieli harmonic inflation for any dependence).

    Returns:
        dict:
          'qvalues'  array [n] aligned to genes,
          'pi0'      the pi0 used,
          'counts'   b_g per gene,
          'F0'       null tail-mass F0(b_g) per gene,
          'M', 'dependence'.
    """
    b, M = _counts_from_W(W_per_draw)
    n = b.shape[0]
    if pi0 == 'auto':
        pi0_val = estimate_pi0_known_null(b, M)
    elif pi0 in ('one', 1, 1.0):
        pi0_val = 1.0
    else:
        pi0_val = float(pi0)

    if dependence == 'arbitrary':
        cn = float(np.sum(1.0 / np.arange(1, n + 1))) if n > 0 else 1.0
    elif dependence in ('prds', 'ind'):
        cn = 1.0
    else:
        raise ValueError(f"unknown dependence: {dependence}")

    F0 = null_cdf(b, M)                        # per-gene null tail mass (left)
    # For each distinct count value v, R(v) = #{genes with b <= v}.
    uniq, inv, counts_v = np.unique(b, return_inverse=True, return_counts=True)
    cum = np.cumsum(counts_v)                  # cum[k] = #genes with b <= uniq[k]
    F0_uniq = null_cdf(uniq, M)
    fdp_uniq = cn * pi0_val * n * F0_uniq / np.maximum(cum, 1)
    # Monotone (q-value) step: min over thresholds at least as extreme (b >= v),
    # i.e. take running min from the LARGE-b end back to small-b, because a more
    # extreme discovery threshold is SMALLER b.
    q_uniq = np.minimum.accumulate(fdp_uniq[::-1])[::-1]
    q_uniq = np.clip(q_uniq, 0.0, 1.0)
    qvals = q_uniq[inv]
    return {'qvalues': qvals, 'pi0': pi0_val, 'counts': b, 'F0': F0, 'M': M,
            'dependence': dependence}


def mirror_fdp(W_per_draw, q=0.1, offset=1):
    """
    pi0-FREE cross-check selector: a mirror / symmetry FDP estimator on the
    per-gene vote margin. Uses ONLY the null's symmetry -- no pi0, no CDF.

    ===================  WHAT THIS BUYS YOU (plain language)  ================
    An independent second opinion on which genes are eGenes, built from a
    DIFFERENT assumption than the q-values. If mirror_fdp and calibrated_qvalues
    pick nearly the same genes, you can trust the call. If they diverge, your
    knockoffs are misspecified (the "known null" isn't holding) and NEITHER
    number is safe -- that divergence is the whole point of computing both.

    ===================  HOW IT WORKS (the statistics)  =====================
    Define the per-gene vote margin
            T_g = #{m: W_g^(m) > 0} - #{m: W_g^(m) <= 0} = M - 2 b_g.
    Under H_g, b_g ~ Binomial(M,1/2), so T_g is SYMMETRIC about 0: a null gene is
    exactly as likely to land at +t as at -t. True eGenes push T_g positive.
    The mirror principle (Barber & Candes 2015's knockoff+ threshold; the mirror-
    statistic / data-splitting FDR line, Dai, Lin, Xing & Liu 2023; Xing, Zhao &
    Liu 2023) estimates the number of FALSE positives among {T_g >= t} by the
    number of nulls that leaked to the mirror image {T_g <= -t}:

            FDP_hat(t) = (offset + #{T_g <= -t}) / max(1, #{T_g >= t}).

    Choose the smallest t with FDP_hat(t) <= q and select {T_g >= t}. With
    offset=1 this is the finite-sample-valid knockoff+ form.

    ===================  THE TRADEOFFS YOU'RE ACCEPTING  ====================
    + No pi0. It sidesteps the partial-identification problem entirely (Genovese
      & Wasserman 2004): symmetry does the job pi0 does elsewhere. This is its
      main virtue as a check on the pi0-dependent q-values.
    + Assumption-light and finite-sample valid (offset=1).
    - Coarser / lower power. It throws away the exact null SHAPE (it uses only
      "left tail counts the right tail's false positives"), so at fixed M it
      typically selects a bit less than the q-values. Good for a check, weaker
      as the sole selector.
    - Detection floor. Like every knockoff+ filter it cannot certify an FDR
      below 1/(#selected) and needs enough right-tail mass to cross q at all;
      with few discoveries it returns "select nothing" (correct, not a bug).

    Args:
        W_per_draw: [M, n] (or [n]).
        q: target FDR.
        offset: 1 (knockoff+, recommended) or 0.

    Returns:
        dict: 'selected_mask' [n] bool, 'T' margins [n], 'tau' threshold (or inf
              if nothing selected), 'n_selected'.
    """
    b, M = _counts_from_W(W_per_draw)
    T = (M - 2 * b).astype(np.float64)
    n = T.shape[0]
    ts = np.unique(np.abs(T[T != 0]))
    ts = np.sort(ts)
    tau = np.inf
    for t in ts:
        num = offset + np.sum(T <= -t)
        den = max(1, np.sum(T >= t))
        if num / den <= q:
            tau = t
            break
    mask = T >= tau if np.isfinite(tau) else np.zeros(n, dtype=bool)
    return {'selected_mask': mask, 'T': T, 'tau': tau,
            'n_selected': int(mask.sum())}


def local_fdr_interval(W_per_draw, offset=1, smooth=0.5):
    """
    Per-gene LOCAL false-discovery rate (posterior "prob this gene is null"),
    reported as an INTERVAL that makes the pi0 identifiability limit explicit.

    ===================  WHAT THIS BUYS YOU (plain language)  ================
    For each gene, a number near 0 means "almost surely a real eGene" and near 1
    means "almost surely null" -- a per-gene confidence, not a set-level FDR.
    Because the fraction of null genes (pi0) can only be BOUNDED, not pinned
    down, each gene gets a RANGE [lfdr_lo, lfdr_hi] rather than a single value.
    Use lfdr_hi (the conservative end) if you ever gate on it; read the width of
    the interval as "how much the answer depends on the unknowable pi0."

    ===================  HOW IT WORKS (the statistics)  =====================
    Two-groups model (Efron 2004; Efron et al. 2001): the observed count
    distribution is a mixture
            f(b) = pi0 * g0(b) + (1 - pi0) * g1(b),
    with g0 = Binomial(M,1/2) (KNOWN) and g1 the unknown alternative (mass at
    small b). The local fdr at count b is
            lfdr(b) = pi0 * g0(b) / f(b),
    i.e. of the genes sitting at vote-count b, the fraction that are null.
    We estimate f(b) by the smoothed empirical fraction of genes at each count.

    THE INTERVAL comes from pi0 being only partially identified (Genovese &
    Wasserman 2004): the data identify an UPPER bound pi0_hi (Storey estimate on
    the known null) and, via the null's symmetry, a lower reference pi0_lo (the
    right tail b>M/2 is pure null and, since g0 is symmetric about M/2, it fixes
    how much of the LEFT tail must also be null; the residual left-tail EXCESS is
    signal). Then
            lfdr_lo(b) = pi0_lo * g0(b) / f(b),
            lfdr_hi(b) = pi0_hi * g0(b) / f(b)      (both clipped to [0,1]).

    ===================  THE TRADEOFFS YOU'RE ACCEPTING  ====================
    + Interpretability. A per-gene posterior is often what a biologist actually
      wants ("how sure are we about THIS gene?").
    + Honesty about pi0. The interval width surfaces the identifiability limit
      instead of hiding it behind one number.
    - Noise. lfdr is a DENSITY RATIO; densities are hard to estimate, so with few
      genes and/or large M (empty grid cells) lfdr is jumpy. `smooth` adds a
      pseudocount to stabilize it, at the cost of a small bias toward 1. This is
      exactly why the SHIPPED selector is the CDF-based calibrated_qvalues, not
      lfdr: cumulatives are stable, densities are not. Treat lfdr as an
      interpretive overlay, not the primary FDR gate.
    - Same discreteness / detection limits as the other two.

    Args:
        W_per_draw: [M, n] (or [n]).
        offset: kept for API symmetry (unused in the density ratio).
        smooth: Laplace pseudocount per grid cell for f(b) (>=0).

    Returns:
        dict:
          'lfdr_lo','lfdr_hi' arrays [n] aligned to genes,
          'pi0_lo','pi0_hi'   the interval endpoints,
          'counts'            b_g per gene, 'M'.
    """
    b, M = _counts_from_W(W_per_draw)
    n = b.shape[0]
    g0 = _binom_half_pmf(M)

    # Smoothed empirical f(b) over the full 0..M grid.
    hist = np.bincount(b, minlength=M + 1).astype(np.float64)
    f = (hist + smooth) / (n + smooth * (M + 1))

    # pi0_hi: conservative Storey upper bound on the known null.
    pi0_hi = estimate_pi0_known_null(b, M)

    # pi0_lo: symmetry / excess-mass reference. The right tail (b > M/2) is pure
    # null; by symmetry g0(b)=g0(M-b), so its mirror fixes the null mass on the
    # left. The left-tail EXCESS over the mirrored right tail is signal, hence
    #   1 - pi0_lo = sum_{b < M/2} [ f_hat(b) - f_hat(M-b) ]_+   (a valid signal
    # lower bound under "no alternative mass on the right"), giving pi0_lo.
    left = np.arange(0, int(np.floor(M / 2.0)))
    excess = np.clip(f[left] - f[M - left], 0.0, None).sum()
    pi0_lo = float(min(pi0_hi, max(0.0, 1.0 - excess)))

    fb = f[b]
    lfdr_hi = np.clip(pi0_hi * g0[b] / fb, 0.0, 1.0)
    lfdr_lo = np.clip(pi0_lo * g0[b] / fb, 0.0, 1.0)
    # Ensure lo <= hi elementwise (pi0_lo <= pi0_hi guarantees this, but clip
    # can reorder at the [0,1] boundary; enforce for safety).
    lfdr_lo = np.minimum(lfdr_lo, lfdr_hi)
    return {'lfdr_lo': lfdr_lo, 'lfdr_hi': lfdr_hi,
            'pi0_lo': pi0_lo, 'pi0_hi': pi0_hi, 'counts': b, 'M': M}


def select_egenes_calibrated(gene_ids, W_per_draw, q=0.1, pi0='auto', offset=1,
                             dependence='prds'):
    """
    Calibrated eGene selection (the shipped step-3 selector) plus its two
    cross-checks, in one call.

    Selection is by calibrated_qvalues (known-null Storey q-values) at level q.
    The return also carries the pi0-free mirror_fdp selection and the
    local_fdr_interval, so a caller can (and the tests do) assert that the three
    agree -- the built-in misspecification alarm. `dependence` ('prds' default |
    'ind' | 'arbitrary') sets the cross-gene dependence assumption of the
    genome-wide q-values (see bh_select / calibrated_qvalues).

    Returns:
        dict:
          'selected'      list of gene_ids with q <= q,
          'qvalues'       array [n] aligned to gene_ids,
          'pi0'           pi0 used by the q-values,
          'mirror'        result of mirror_fdp (selected_mask, tau, ...),
          'lfdr'          result of local_fdr_interval,
          'n_draws'       M,
          'agreement'     Jaccard overlap between q-value and mirror selections.
    """
    W = np.atleast_2d(np.asarray(W_per_draw, dtype=np.float64))
    M, m = W.shape
    assert m == len(gene_ids), "W columns must align to gene_ids"
    qres = calibrated_qvalues(W, pi0=pi0, offset=offset, dependence=dependence)
    qvals = qres['qvalues']
    q_mask = qvals <= q
    selected = [gene_ids[i] for i in range(m) if q_mask[i]]

    mir = mirror_fdp(W, q=q, offset=offset)
    lf = local_fdr_interval(W, offset=offset)

    a = set(np.where(q_mask)[0].tolist())
    b_ = set(np.where(mir['selected_mask'])[0].tolist())
    union = a | b_
    agreement = 1.0 if not union else len(a & b_) / len(union)

    # The mirror (knockoff+ at this offset) has a DETECTION FLOOR: it cannot
    # certify FDR<=q with fewer than ~offset/q discoveries, because its estimate
    # is (offset + #neg)/#pos and needs #pos >= offset/q even with zero negatives.
    # So when the total discovery count is below that floor the mirror is EXPECTED
    # to select little/nothing regardless of correctness, and low agreement is
    # uninformative -- NOT evidence of misspecification. `mirror_informative`
    # flags whether the agreement check is meaningful; callers should gate any
    # misspecification warning on it.
    floor = int(np.ceil((offset or 1) / q)) if q > 0 else m
    mirror_informative = bool(len(selected) >= floor)

    return {'selected': selected, 'qvalues': qvals, 'pi0': qres['pi0'],
            'mirror': mir, 'lfdr': lf, 'n_draws': M, 'agreement': agreement,
            'mirror_informative': mirror_informative}


# ===========================================================================
#  KFc-style continuous statistic + empirical mirror-null FDR (fixes problem B)
#  ==========================================================================
#
#  WHY.  The gene statistic W_g = maxPIP(orig) - maxPIP(knockoff) is DEGENERATE
#  under a null gene: SuSiE collapses the prior variance to exactly 0, PIPs become
#  identically 0, and W_g is a point mass at 0 (an ATOM). The per-gene count
#  b = #{W <= 0} is then pinned at M and the assumed Binomial(M,1/2) null is false
#  (see docs/calibration_findings.md). No amount of pi0 / discrete-FDR tuning
#  fixes a statistic whose null is an atom.
#
#  THE FIX (KFc; Wang et al., NAR Genomics & Bioinformatics 2023,
#  github.com/QingboWang/KFc; and the GhostKnockoff / Barber-Candes lineage):
#    1. Use a CONTINUOUS per-gene importance -- the strongest marginal cis
#       association, -log10(min nominal p) over the gene's variants -- computed on
#       the REAL genotypes and on a KNOCKOFF copy. This has NO atom and NO ties
#       (min-p over t-tests is continuous), so
#           W_g = imp(real) - imp(knockoff)
#       is a genuinely continuous, swap-antisymmetric statistic whose null sign is
#       a real coin flip.
#    2. Select genome-wide with the EMPIRICAL MIRROR-NULL knockoff+ threshold
#       (Barber & Candes 2015): FDP_hat(t) = (offset + #{W_g <= -t}) / #{W_g >= t}.
#       No distributional assumption; and W_g <= 0 can NEVER be selected (a tie or
#       a knockoff-win "provides no evidence against the null", Candes et al.
#       2018), so the 2^-M-tail landmine and the atom both disappear.
#  SuSiE is then used only to LOCALIZE within the selected eGenes, not for the FDR
#  statistic. This is faster too (a marginal scan, no per-gene SuSiE fit for the
#  filter).
# ===========================================================================


def marginal_importance(X_t, y_t, dof_adjust=2):
    """
    Continuous per-gene importance = -log10(min two-sided nominal p) over the
    gene's cis variants, from per-variant marginal t-tests.

    No SuSiE, no atom, no ties: min-p over continuous t-statistics is continuous.

    Args:
        X_t: [N, p] genotype/dosage matrix for the gene's cis window. Should be
            covariate-RESIDUALIZED (and the same residualization applied to y_t);
            columns need not be standardized (correlation is scale-free).
        y_t: [N] or [N,1] residualized phenotype.
        dof_adjust: residual dof = N - dof_adjust (2 for an intercept + slope; use
            2 + n_covariates if covariates were projected out beforehand -- the
            exact value barely moves -log10 p at eQTL N).

    Returns:
        importance: float, -log10(min_j p_j). Larger = stronger cis signal.
    """
    from scipy import stats as _stats
    X = np.asarray(X_t, dtype=np.float64)
    y = np.asarray(y_t, dtype=np.float64).ravel()
    N, p = X.shape
    Xc = X - X.mean(0)
    yc = y - y.mean()
    sx = np.sqrt((Xc * Xc).sum(0))
    sy = np.sqrt((yc * yc).sum())
    denom = sx * sy
    good = denom > 0
    r = np.zeros(p)
    r[good] = (Xc[:, good] * yc[:, None]).sum(0) / denom[good]
    r = np.clip(r, -0.999999, 0.999999)
    dof = max(1, N - dof_adjust)
    t = r * np.sqrt(dof / (1.0 - r * r))
    # two-sided p; -log10 via the log-sf for numerical range
    logsf = _stats.t.logsf(np.abs(t), dof)           # log P(T > |t|)
    neglog10p = -(np.log(2.0) + logsf) / np.log(10.0)  # -log10(2*sf)
    neglog10p[~good] = 0.0
    return float(neglog10p.max()) if p > 0 else 0.0


def gene_W_marginal(X_t, Xk_t, y_t, dof_adjust=2):
    """
    Continuous KFc gene statistic: W_g = imp(real) - imp(knockoff), where imp is
    marginal_importance (-log10 min-p). Swap-antisymmetric and continuous (no
    atom). X_t and Xk_t must be the SAME gene's real and knockoff windows, both
    covariate-residualized like y_t.
    """
    return marginal_importance(X_t, y_t, dof_adjust) - \
        marginal_importance(Xk_t, y_t, dof_adjust)


def mirror_select_egenes(gene_ids, W, q=0.1, offset=1):
    """
    Genome-wide eGene selection from continuous per-gene statistics via the
    empirical mirror-null knockoff+ threshold (Barber & Candes 2015).

    FDP_hat(t) = (offset + #{W_g <= -t}) / max(1, #{W_g >= t}); select the genes
    with W_g >= tau, where tau is the smallest t>0 with FDP_hat(t) <= q. Genes
    with W_g <= 0 are never selected. Valid for ANY continuous swap-antisymmetric
    W (no Binomial / no known-null assumption) -- the correct selector for the
    continuous KFc statistic.

    Args:
        gene_ids: list of n gene ids.
        W: array [n] of continuous per-gene statistics (gene_W_marginal), aligned.
        q: target FDR.
        offset: 1 (knockoff+, finite-sample valid) or 0.

    Returns:
        dict: 'selected' (gene_ids), 'W' (array), 'tau', 'qvalues' (per-gene
              knockoff q-value = min over t<=W_g of FDP_hat), 'n_selected'.
    """
    W = np.asarray(W, dtype=np.float64)
    n = W.shape[0]
    tau = knockoff_threshold(W, q=q, offset=offset)
    mask = W >= tau if np.isfinite(tau) else np.zeros(n, dtype=bool)
    # per-gene knockoff q-value: for each positive W_g, the min achievable FDP if
    # the threshold were set at that gene (monotone from the top).
    qvals = np.ones(n)
    order = np.argsort(-W)                    # descending
    pos = W[order] > 0
    for rank, idx in enumerate(order):
        t = W[idx]
        if t <= 0:
            continue
        num = offset + np.sum(W <= -t)
        den = max(1, np.sum(W >= t))
        qvals[idx] = min(1.0, num / den)
    # enforce monotonicity in W (larger W -> smaller q)
    qsorted = np.minimum.accumulate(qvals[order])
    qvals[order] = qsorted
    selected = [gene_ids[i] for i in range(n) if mask[i]]
    return {'selected': selected, 'W': W, 'tau': tau, 'qvalues': qvals,
            'n_selected': int(mask.sum())}


# ---------------------------------------------------------------------------
#  Derandomization (Ren & Barber 2024, e-value aggregation)
# ---------------------------------------------------------------------------

def derandomize_cs(per_draw_cs, q, offset=1):
    """
    Aggregate CS selections across multiple knockoff draws via e-values + e-BH.

    A single knockoff draw makes CS selection random (different seed -> different
    calls), which is unacceptable for a pipeline. Ren & Barber (2024) convert
    each draw's knockoff selection into e-values and average them, then apply
    e-BH, controlling FDR while removing run-to-run variability.

    Args:
        per_draw_cs: list (one per draw) of the 'all' CS lists returned by
            filter_credible_sets, where CSs are matched across draws by a stable
            key (here the frozenset of original indices). Each element is the
            list of {'cs_id','orig_idx','W'} for that draw plus the draw's tau.
            We recompute per-draw selection at the given q.
        q: target FDR.
        offset: knockoff+ offset.

    Returns:
        list of stable CS keys (frozensets of original indices) selected after
        e-BH aggregation.
    """
    # Collect the universe of CS keys and per-draw e-values.
    # e-value for a selected CS in a draw: p_cs / (1 + #negatives at tau),
    # following the knockoff-as-e-BH construction; 0 if not selected.
    from collections import defaultdict
    evalues = defaultdict(list)
    all_keys = set()

    for draw in per_draw_cs:
        cs_list = draw['all']
        W = np.array([c['W'] for c in cs_list], dtype=np.float64)
        tau = knockoff_threshold(W, q=q, offset=offset)
        n_neg = offset + np.sum(W <= -tau) if np.isfinite(tau) else np.inf
        m = len(cs_list)
        for c in cs_list:
            key = frozenset(c['orig_idx'].tolist())
            all_keys.add(key)
            if np.isfinite(tau) and c['W'] >= tau:
                evalues[key].append(m / n_neg)
            else:
                evalues[key].append(0.0)

    keys = sorted(all_keys, key=lambda k: min(k) if k else -1)
    # average e-value per CS (missing draws contribute 0)
    n_draws = len(per_draw_cs)
    ebar = np.array([np.sum(evalues[k]) / n_draws for k in keys], dtype=np.float64)

    # e-BH at level q
    order = np.argsort(-ebar)  # descending
    m = len(keys)
    selected = []
    for rank, idx in enumerate(order, start=1):
        if ebar[idx] >= m / (q * rank):
            selected = order[:rank]
        # (e-BH: largest rank where e_(k) >= m/(q k))
    return [keys[i] for i in selected]


# ---------------------------------------------------------------------------
#  Calibration harness (the empirical FDR check that must gate any use)
# ---------------------------------------------------------------------------

def calibration_report(V_per_replicate, R_per_replicate, q):
    """
    Estimate empirical FDR correctly from many independent simulation replicates.

    FDR is the EXPECTATION of the realized false-discovery proportion, so it must
    be estimated as the mean per-replicate FDP (external review, point 8):

        FDR_hat = (1/B) sum_b  V_b / max(R_b, 1)

    NOT as the discovery-weighted pooled ratio sum_b V_b / sum_b R_b, which
    estimates a different quantity. Each element of the inputs is ONE replicate
    (e.g. one whole null-permutation of the dataset), not one gene.

    For a COMPLETE null (every discovery false, V_b == R_b), FDP_b is 1 whenever
    R_b > 0 and 0 otherwise, so FDR = P(R > 0); this is also reported as
    'prob_any_discovery' and requires many replicates to estimate -- a single
    null run cannot.

    Args:
        V_per_replicate: array [B] of false-discovery counts, one per replicate.
        R_per_replicate: array [B] of total-discovery counts, one per replicate.
        q: target FDR.

    Returns:
        dict with the mean-FDP estimate, its standard error across replicates,
        P(any discovery), and a 'calibrated' flag (mean FDP <= q within 2 SE).
    """
    V = np.asarray(V_per_replicate, dtype=np.float64)
    R = np.asarray(R_per_replicate, dtype=np.float64)
    assert V.shape == R.shape, "V and R must have one entry per replicate"
    B = V.shape[0]
    if B == 0:
        return {'target_fdr': q, 'empirical_fdr': 0.0, 'se': 0.0,
                'prob_any_discovery': 0.0, 'n_replicates': 0, 'calibrated': True}

    fdp = V / np.maximum(R, 1.0)               # per-replicate realized FDP
    emp_fdr = float(fdp.mean())
    se = float(fdp.std(ddof=1) / np.sqrt(B)) if B > 1 else float('nan')
    return {
        'target_fdr': q,
        'empirical_fdr': emp_fdr,               # mean per-replicate FDP
        'se': se,
        'prob_any_discovery': float((R > 0).mean()),
        'n_replicates': B,
        'mean_discoveries': float(R.mean()),
        'calibrated': bool(emp_fdr <= q + 2 * (se if np.isfinite(se) else 0.0)),
    }

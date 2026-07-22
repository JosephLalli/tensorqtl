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
Default construction: second-order (Gaussian) model-X knockoffs on the
covariate-residualized dosage matrix. Given the variant covariance ``Sigma``,
sample::

    X_knockoff = X (I - Sigma^{-1} D) + E chol(2D - D Sigma^{-1} D)

where ``D = diag(s)`` and ``E ~ N(0, I)``. The vector ``s`` controls how
distinguishable each knockoff is from its original; ``2D - D Sigma^{-1} D`` must
stay PSD, which bounds ``s``.

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
#  linear in p -- so it would scale to chromosome-wide designs where the O(p^3)
#  Gaussian construction is infeasible.
#
#  *** WORK IN PROGRESS -- NOT YET CORRECT. DO NOT USE FOR REAL KNOCKOFFS. ***
#  This reimplementation is swap-exchangeable ONLY for small p: the swap-test
#  total-variation distance grows monotonically with p (empirically ~0.008 at
#  p=2, ~0.026 at p=4, ~0.066 at p=5, ~0.11 at p=6), indicating a per-step error
#  in the forward partition-function (Z) recursion that compounds along the
#  chain. At cis-window scale (p = hundreds) this would be badly invalid. The
#  algorithm STRUCTURE matches SNPknock's dmc.cpp/hmm.cpp (the earlier
#  from-memory attempts were fundamentally wrong -- missing Z entirely -- and
#  failed at TV~0.2 for all p); the remaining bug is a subtle normalization
#  error in the Z carry. Blocked on a correct reference (working snpknock build
#  needs Armadillo; R/Python reference not retrievable). Gated by
#  tests/test_hmm_knockoffs.py, which asserts validity at p<=4 and DOCUMENTS the
#  large-p failure (xfail) so this is not mistaken for a finished generator.
# ---------------------------------------------------------------------------

def dmc_knockoffs(X, init_p, Q, seed=0):
    """
    Discrete Markov chain knockoffs for a state sequence.

    *** WORK IN PROGRESS -- valid only for small p (see module comment). The Z
    partition recursion has a compounding bug at larger p; do not use for
    real (p >> 10) knockoffs until the swap-test passes at p=hundreds. ***

    Args:
        X: [N, p] integer state matrix (values in 0..K-1).
        init_p: [K] initial state distribution P(state_0).
        Q: [p-1, K, K] transition matrices; Q[j-1][a, b] = P(state_j=b | state_{j-1}=a).
        seed: RNG seed.

    Returns:
        Xt: [N, p] integer knockoff state matrix (swap-exchangeable only at small p).
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
    HMM knockoffs for observed genotype sequences (Sesia et al. 2019).

    *** WORK IN PROGRESS -- valid only for small p; inherits the compounding Z
    bug from dmc_knockoffs (see module comment). Do not use for real
    p=hundreds knockoffs yet. ***

    Three steps: (1) backward pass for beta = P(future | H_j); (2) forward-sample
    the hidden states H ~ P(H | X); (3) DMC knockoff on H -> Ht; (4) re-emit
    Xt_j ~ P(X_j | H_j = Ht_j). The composition is swap-exchangeable (at small p)
    because the DMC step is, and the emission is independent given the state.

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

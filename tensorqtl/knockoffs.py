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
    Compute a per-credible-set knockoff statistic from a SuSiE fit on the
    augmented [X, X_knockoff] design.

    For each SuSiE credible set (whose members are indices into 0..2p-1), we
    consider only its *original* members (index < p) and contrast their summed
    importance against that of their knockoff counterparts. Treating the CS as
    the unit of control (rather than the variant) is both what an eQTL analyst
    reports and more powerful under tight cis-window LD, where a single variant
    and its knockoff are hard to separate but the CS as a whole is not.

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

def calibration_report(realized_false, realized_total, q):
    """
    Summarize a null-permutation calibration run.

    Under a null (phenotype permuted so no variant is causal), every reported CS
    is by definition false, so the realized false-discovery *count* per gene,
    averaged over genes and permutations, estimates the achieved FDR. This is
    the only evidence that the target q holds at a given N; the model-X theory
    guarantees it only with a correctly estimated knockoff distribution, which
    at small N is not automatic.

    Args:
        realized_false: array of per-gene false-CS counts across null runs
        realized_total: array of per-gene reported-CS counts across null runs
        q: the target FDR that was requested

    Returns:
        dict with the empirical FDR (mean false / mean reported), the target,
        and a boolean 'calibrated' (empirical <= q within Poisson noise).
    """
    realized_false = np.asarray(realized_false, dtype=np.float64)
    realized_total = np.asarray(realized_total, dtype=np.float64)
    tot = realized_total.sum()
    emp_fdr = realized_false.sum() / tot if tot > 0 else 0.0
    # crude Poisson-ish tolerance on the false count
    se = np.sqrt(max(realized_false.sum(), 1.0)) / tot if tot > 0 else 0.0
    return {
        'target_fdr': q,
        'empirical_fdr': emp_fdr,
        'reported_cs_total': float(tot),
        'false_cs_total': float(realized_false.sum()),
        'se': se,
        'calibrated': bool(emp_fdr <= q + 2 * se),
    }

# Mr.ASH (Multiple Regression with Adaptive SHrinkage) coordinate-ascent solver.
#
# This is the polygenic-background fitter used by SuSiE-ash (susie(...,
# unmappable_effects="ash") in susieR 2.0): a mixture-of-normals prior over all
# variant effects, fit by coordinate ascent. SuSiE-ash refits this background
# between IBSS iterations on residuals net of the sparse credible-set effects.
#
# Faithful numpy port of susieR's compiled core (src/caisa.cpp updatebetaj /
# caisa_cpp) and its R wrapper (R/mr.ash.R). The coordinate sweep is inherently
# sequential (Gauss-Seidel: each beta_j update mutates the residual seen by the
# next coordinate), so it stays a CPU loop; K (mixture components) and the
# per-coordinate work are vectorized. Verified numerically identical to
# susieR:::caisa_cpp (see tests / verification harness).
import numpy as np


def default_sa2_grid(w, n):
    """susieR's default mixture-variance grid: (2^(k/25)-1)^2 for k=0..24, scaled
    by n/median(w). Component 0 is the spike (sa2=0)."""
    sa2 = (2.0 ** (np.arange(25) / 25.0) - 1.0) ** 2
    return sa2 / np.median(w) * n


def _caisa(X, w, sa2, pi, beta, r, sigma2, order, max_iter, min_iter,
           convtol, epstol, method_q, update_pi, update_sigma):
    """Coordinate-ascent Mr.ASH in sufficient-statistic form (port of caisa_cpp).

    X (n,p) centered design; w=colSums(X^2); sa2 (K,) mixture variances (sa2[0]=0);
    pi (K,) mixture weights; beta (p,) effects (mutated copy); r (n,) residual
    y-X@beta (mutated copy); sigma2 residual variance. order: 1D int array of length
    >= p*max_iter giving the coordinate visited at each step (None -> cyclic 0..p-1).
    Returns dict(beta, sigma2, pi, iter, varobj)."""
    n, p = X.shape
    K = sa2.shape[0]
    beta = beta.astype(np.float64).copy()
    r = r.astype(np.float64).copy()
    pi = pi.astype(np.float64).copy()
    sa2 = sa2.astype(np.float64)
    w = w.astype(np.float64)

    # S2inv[k,j] = 1/(1/sa2[k] + w[j]); spike row (k=0, sa2=0 -> 1/sa2=inf) -> epstol
    with np.errstate(divide='ignore'):
        inv_sa2 = np.where(sa2 > 0, 1.0 / sa2, np.inf)
    S2inv = 1.0 / (inv_sa2[:, None] + w[None, :])   # (K, p)
    S2inv[0, :] = epstol

    varobj = np.zeros(max_iter)
    it = 0
    while it < max_iter:
        a1 = 0.0
        a2 = 0.0
        piold = pi.copy()
        betaold = beta.copy()
        pi = np.zeros(K)

        for jj in range(p):
            j = jj if order is None else int(order[it * p + jj])
            xj = X[:, j]
            wj = w[j]
            s2inv_j = S2inv[:, j]

            bjwj = r @ xj + beta[j] * wj
            r += xj * beta[j]                        # add current effect back

            muj = bjwj * s2inv_j                     # (K,) posterior means
            muj[0] = 0.0
            phij = np.log(piold + epstol) - np.log(1.0 + sa2 * wj) / 2.0 + muj * (bjwj / 2.0 / sigma2)
            phij = np.exp(phij - phij.max())
            phij = phij / phij.sum()

            pi += phij / p
            beta[j] = phij @ muj
            r += -xj * beta[j]                       # subtract updated effect

            a1 += bjwj * beta[j]
            a2 += phij @ np.log(phij + epstol)
            phij0 = phij.copy()
            phij0[0] = 0.0
            a2 += -(phij0 @ np.log(s2inv_j)) / 2.0

        varobj[it] = r @ r - (beta ** 2) @ w + a1
        if update_sigma:
            if method_q == "sigma_indep_q":
                sigma2 = (varobj[it] + p * (1.0 - pi[0]) * sigma2) / (n + p * (1.0 - pi[0]))
            elif method_q == "sigma_dep_q":
                sigma2 = varobj[it] / n
        if update_pi:
            piold = pi
        varobj[it] = (varobj[it] / sigma2 / 2.0
                      + np.log(2.0 * np.pi * sigma2) / 2.0 * n
                      - (pi @ np.log(piold + epstol)) * p + a2)
        for k in range(1, K):
            varobj[it] += pi[k] * np.log(sa2[k]) * p / 2.0
        if not update_pi:
            pi = piold

        if it >= min_iter - 1:
            beta_norm = np.linalg.norm(beta)
            if np.linalg.norm(betaold - beta) < convtol * max(1.0, beta_norm):
                it += 1
                break
            if it > 0 and varobj[it] > varobj[it - 1]:
                break
        it += 1

    return {'beta': beta, 'sigma2': float(sigma2), 'pi': pi, 'iter': it, 'varobj': varobj[:it]}


def mr_ash(X, y, sa2=None, sigma2=None, pi=None, beta_init=None,
           update_pi=True, update_sigma=True, method_q="sigma_dep_q",
           intercept=True, max_iter=1000, min_iter=1, convtol=1e-8, epstol=1e-12,
           order=None):
    """Mr.ASH wrapper (port of R/mr.ash.R): centers X/y (intercept), builds the
    default sa2 grid and uniform pi, and runs the coordinate ascent. Returns
    dict(beta, sigma2, pi, iter, intercept, varobj). standardize is not supported
    (SuSiE-ash calls with standardize=FALSE)."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("X must be a two-dimensional array.")
    if y.ndim == 2 and y.shape[1] == 1:
        y = y[:, 0]
    elif y.ndim != 1:
        raise ValueError("y must be a one-dimensional array or a single column.")
    n, p = X.shape
    if n < 2 or p < 1:
        raise ValueError("X must contain at least two samples and one predictor.")
    if y.shape[0] != n:
        raise ValueError("X and y must contain the same number of samples.")
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("X and y must contain only finite values.")
    if method_q not in {"sigma_dep_q", "sigma_indep_q"}:
        raise ValueError("method_q must be 'sigma_dep_q' or 'sigma_indep_q'.")
    if not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer.")
    if not isinstance(min_iter, (int, np.integer)) or not 1 <= min_iter <= max_iter:
        raise ValueError("min_iter must be between 1 and max_iter.")
    if not np.isfinite(convtol) or convtol <= 0:
        raise ValueError("convtol must be positive and finite.")
    if not np.isfinite(epstol) or epstol <= 0:
        raise ValueError("epstol must be positive and finite.")

    x_mean = X.mean(0) if intercept else np.zeros(p)
    y_mean = float(y.mean()) if intercept else 0.0
    if intercept:
        X = X - x_mean
        y = y - y_mean
    w = (X * X).sum(0)
    if not np.isfinite(w).all() or np.any(w <= 0):
        raise ValueError("X must not contain constant predictors.")
    if sa2 is None:
        sa2 = default_sa2_grid(w, n)
    else:
        sa2 = np.asarray(sa2, dtype=np.float64)
    if (
        sa2.ndim != 1
        or sa2.size < 2
        or not np.isfinite(sa2).all()
        or sa2[0] != 0
        or np.any(sa2[1:] <= 0)
    ):
        raise ValueError(
            "sa2 must start with 0 and contain positive, finite slab variances."
        )
    K = sa2.shape[0]
    has_beta_init = beta_init is not None
    beta = (
        np.zeros(p)
        if beta_init is None
        else np.asarray(beta_init, dtype=np.float64).copy()
    )
    if beta.ndim != 1 or beta.shape[0] != p or not np.isfinite(beta).all():
        raise ValueError("beta_init must contain one finite value per predictor.")
    r = y - X @ beta
    if sigma2 is None:
        sigma2 = float(((r - r.mean()) ** 2).mean())   # var.n
    else:
        sigma2 = float(sigma2)
    if not np.isfinite(sigma2) or sigma2 <= 0:
        raise ValueError("sigma2 must be positive and finite.")
    if pi is None:
        # port of R/mr.ash.R pi-init: uniform when beta.init is absent, but
        # data-driven when an explicit beta.init is supplied (even zeros). The
        # data-driven branch is what SuSiE-ash's first refit relies on
        # (pi=None, beta.init=theta=0) -- a uniform init there diverges from
        # susieR. beta.init absent here means beta_init was None above.
        if not has_beta_init:
            pi = np.full(K, 1.0 / K)
        else:
            with np.errstate(divide='ignore'):
                S = (np.where(w > 0, 1.0 / w, np.inf)[:, None] + sa2[None, :]) * sigma2
            Phi = -beta[:, None] ** 2 / S / 2.0 - np.log(S) / 2.0   # (p, K)
            Phi = np.exp(Phi - Phi.max(axis=1, keepdims=True))
            Phi = Phi / Phi.sum(axis=1, keepdims=True)
            pi = Phi.mean(axis=0)
    else:
        pi = np.asarray(pi, dtype=np.float64)
    if (
        pi.ndim != 1
        or pi.shape[0] != K
        or not np.isfinite(pi).all()
        or np.any(pi < 0)
        or not np.isclose(pi.sum(), 1.0)
    ):
        raise ValueError("pi must be a probability vector with one value per sa2.")

    if order is not None:
        order = np.asarray(order)
        if (
            order.ndim != 1
            or not np.issubdtype(order.dtype, np.integer)
            or order.size < p * max_iter
            or np.any(order < 0)
            or np.any(order >= p)
        ):
            raise ValueError(
                "order must contain at least p * max_iter valid predictor indices."
            )
    out = _caisa(X, w, sa2, pi, beta, r, float(sigma2), order,
                 max_iter, min_iter, convtol, epstol, method_q, update_pi, update_sigma)
    out['intercept'] = float(y_mean - x_mean @ out['beta']) if intercept else 0.0
    return out

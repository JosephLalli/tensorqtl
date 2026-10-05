# Mr.ASH (Multiple Regression with Adaptive SHrinkage) coordinate-ascent solver.
#
# This is the polygenic-background fitter used by SuSiE-ash (susie(...,
# unmappable_effects="ash") in susieR 2.0): a mixture-of-normals prior over all
# variant effects, fit by coordinate ascent. SuSiE-ash refits this background
# between IBSS iterations (on residuals net of the sparse credible-set effects,
# with confident variants masked) so a diffuse polygenic background is absorbed
# adaptively instead of distorting the sparse fit -- the paper's claimed 1.5-3x
# FDR reduction vs SuSiE-inf's single Gaussian background.
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
    y = np.asarray(y, dtype=np.float64).ravel()
    n, p = X.shape
    if intercept:
        X = X - X.mean(0)
        y = y - y.mean()
    w = (X * X).sum(0)
    if sa2 is None:
        sa2 = default_sa2_grid(w, n)
    else:
        sa2 = np.asarray(sa2, dtype=np.float64)
    if sa2[0] != 0:
        raise ValueError("sa2[0] must be 0 (spike component).")
    K = sa2.shape[0]
    beta = np.zeros(p) if beta_init is None else np.asarray(beta_init, dtype=np.float64).copy()
    r = y - X @ beta
    if sigma2 is None:
        sigma2 = float(((r - r.mean()) ** 2).mean())   # var.n
    if pi is None:
        pi = np.full(K, 1.0 / K)
    out = _caisa(X, w, sa2, pi, beta, r, float(sigma2), order,
                 max_iter, min_iter, convtol, epstol, method_q, update_pi, update_sigma)
    out['intercept'] = float(y.mean()) if intercept else 0.0
    return out

"""
SuSiE-inf: SuSiE fine-mapping with an infinitesimal (polygenic) random-effect
component. Models

    y = X b_sparse + X u_infinitesimal + e,
    u ~ N(0, tau^2 I),  e ~ N(0, sigma^2 I),

so a diffuse polygenic background is absorbed into `tau^2` instead of being
forced onto the sparse credible sets. This improves PIP calibration when the
cis architecture is not truly sparse (Cui, Kanai, ... Finucane, Nat Genet 2024,
"Improving fine-mapping by modeling infinitesimal effects").

This is a faithful port of the reference implementation
(github.com/FinucaneLab/fine-mapping-inf, `susieinf/susieinf.py`) -- the same
algorithm that susieR (>=2.0 / GitHub master) exposes as
`susie(..., unmappable_effects = "inf")`. It works on summary statistics
(z-scores + LD) via the eigendecomposition of X'X, which is what makes the
infinitesimal term tractable (var = tau^2 * D^2 + sigma^2 is diagonal in the
eigenbasis of X'X). `susie_inf_from_data` is a convenience wrapper that builds
the summary statistics from individual-level (residualized) genotypes and
phenotype, matching the interface of `tensorqtl.susie.susie`.

Dependency-light on purpose (numpy + scipy only): importable and testable
without the full tensorqtl import chain, like `tensorqtl.knockoffs`.
"""
import numpy as np
from scipy.optimize import minimize_scalar, minimize
from scipy import linalg as sla
from scipy import special as ssp


def _mom_variances(PIP, mu, omega, sigmasq, tausq, n, V, Dsq, VtXty, Xty, yty,
                   est_sigmasq, est_tausq):
    """Method-of-moments update of (sigma^2, tau^2). Recommended as the more
    stable estimator for the infinitesimal model."""
    p, L = mu.shape
    A = np.array([[n, Dsq.sum()], [Dsq.sum(), (Dsq ** 2).sum()]], dtype=np.float64)
    # diag(V' M V), where M is the posterior second-moment matrix of Xb_sparse
    b = np.sum(mu * PIP, axis=1)
    Vtb = V.T.dot(b)
    diagVtMV = Vtb ** 2
    tmpD = np.zeros(p)
    for l in range(L):
        bl = mu[:, l] * PIP[:, l]
        diagVtMV -= V.T.dot(bl) ** 2
        tmpD += PIP[:, l] * (mu[:, l] ** 2 + 1.0 / omega[:, l])
    diagVtMV += np.sum((V.T) ** 2 * tmpD, axis=1)
    x = np.zeros(2)
    x[0] = yty - 2 * np.sum(b * Xty) + np.sum(Dsq * diagVtMV)
    x[1] = np.sum(Xty ** 2) - 2 * np.sum(Vtb * VtXty * Dsq) + np.sum(Dsq ** 2 * diagVtMV)
    if est_tausq:
        sol = sla.solve(A, x)
        if sol[0] > 0 and sol[1] > 0:
            sigmasq, tausq = sol
        else:  # degenerate -> fall back to plain SuSiE (tau^2 = 0)
            sigmasq, tausq = x[0] / n, 0.0
    elif est_sigmasq:
        sigmasq = (x[0] - A[0, 1] * tausq) / n
    return float(sigmasq), float(tausq)


def _mle_variances(PIP, mu, omega, sigmasq, tausq, n, V, Dsq, VtXty, yty,
                   est_sigmasq, est_tausq, sigmasq_range, tausq_range):
    """Maximum-likelihood (negative-ELBO, L-BFGS-B) update of (sigma^2, tau^2)."""
    p, L = mu.shape
    if sigmasq_range is None:
        sigmasq_range = (0.2 * yty / n, 1.2 * yty / n)
    if tausq_range is None:
        tausq_range = (1e-12, 1.2 * yty / (n * p))
    b = np.sum(mu * PIP, axis=1)
    Vtb = V.T.dot(b)
    diagVtMV = Vtb ** 2
    tmpD = np.zeros(p)
    for l in range(L):
        bl = mu[:, l] * PIP[:, l]
        diagVtMV -= V.T.dot(bl) ** 2
        tmpD += PIP[:, l] * (mu[:, l] ** 2 + 1.0 / omega[:, l])
    diagVtMV += np.sum((V.T) ** 2 * tmpD, axis=1)

    def negelbo(x):  # x = (sigma^2, tau^2)
        vr = x[1] * Dsq + x[0]
        return (0.5 * (n - p) * np.log(x[0]) + 0.5 / x[0] * yty
                + np.sum(0.5 * np.log(vr)
                         - 0.5 * x[1] / x[0] * VtXty ** 2 / vr
                         - Vtb * VtXty / vr
                         + 0.5 * Dsq / vr * diagVtMV))

    if est_tausq:
        res = minimize(negelbo, (sigmasq, tausq), method='L-BFGS-B',
                       bounds=(sigmasq_range, tausq_range))
        if res.success:
            sigmasq, tausq = res.x
    elif est_sigmasq:
        res = minimize(lambda s: negelbo((s, tausq)), sigmasq, method='L-BFGS-B',
                       bounds=(sigmasq_range,))
        if res.success:
            sigmasq = float(res.x)
    return float(sigmasq), float(tausq)


def susie_inf(z, meansq, n, L=10, LD=None, V=None, Dsq=None,
              est_ssq=True, ssq=None, ssq_range=(0, 1), pi0=None,
              est_sigmasq=True, est_tausq=True, sigmasq=1.0, tausq=0.0,
              method='moments', sigmasq_range=None, tausq_range=None,
              PIP=None, mu=None, maxiter=100, PIP_tol=1e-3, verbose=False):
    """SuSiE with an infinitesimal random effect, from summary statistics.

    Args:
        z: length-p z-scores, z = X'y / sqrt(n)  (X, y column-standardized).
        meansq: ||y||^2 / n.
        n: sample size.
        L: number of modeled sparse causal effects.
        LD: p x p LD matrix X'X / n. Provide this OR (V, Dsq).
        V, Dsq: precomputed eigenvectors and eigenvalues of X'X (Dsq = n * eig(LD)).
        method: 'moments' (recommended, closed-form) or 'MLE' (L-BFGS-B on -ELBO).
        est_tausq: estimate the infinitesimal variance tau^2 (set False, tausq=0
            to recover ordinary SuSiE in this eigenbasis parameterization).
        (remaining args mirror the reference implementation.)

    Returns:
        dict with PIP [p,L], mu, omega, lbf, lbf_variable, ssq, sigmasq, tausq,
        alpha (posterior mean of the infinitesimal effect vector), converged.
    """
    z = np.asarray(z, dtype=np.float64)
    p = len(z)
    if (V is None or Dsq is None) and LD is None:
        raise ValueError("Provide either LD or the pair (V, Dsq).")
    if V is None or Dsq is None:
        eigvals, V = sla.eigh(np.asarray(LD, dtype=np.float64))
        Dsq = np.maximum(n * eigvals, 0.0)
    else:
        V = np.asarray(V, dtype=np.float64)
        Dsq = np.maximum(np.asarray(Dsq, dtype=np.float64), 0.0)

    Xty = np.sqrt(n) * z
    VtXty = V.T.dot(Xty)
    yty = n * meansq

    var = tausq * Dsq + sigmasq
    diagXtOmegaX = np.sum(V ** 2 * (Dsq / var), axis=1)
    XtOmegay = V.dot(VtXty / var)

    if ssq is None:
        ssq = np.ones(L) * 0.2
    else:
        ssq = np.array(ssq, dtype=np.float64)
    if PIP is None:
        PIP = np.ones((p, L)) / p
    else:
        PIP = np.array(PIP, dtype=np.float64)
    if mu is None:
        mu = np.zeros((p, L))
    else:
        mu = np.array(mu, dtype=np.float64)
    lbf_variable = np.zeros((p, L))
    lbf = np.zeros(L)
    omega = diagXtOmegaX[:, None] + 1.0 / ssq

    if pi0 is None:
        logpi0 = np.full(p, np.log(1.0 / p))
    else:
        logpi0 = np.full(p, -np.inf)
        nz = np.nonzero(pi0 > 0)[0]
        logpi0[nz] = np.log(pi0[nz])

    converged = False
    PIP_diff = np.inf
    for it in range(maxiter):
        PIP_prev = PIP.copy()
        for l in range(L):
            b = np.sum(mu * PIP, axis=1) - mu[:, l] * PIP[:, l]
            XtOmegaXb = V.dot(V.T.dot(b) * Dsq / var)
            XtOmegar = XtOmegay - XtOmegaXb
            if est_ssq:
                def negloglik(x):  # -log marginal likelihood over the SER prior variance
                    return -ssp.logsumexp(-0.5 * np.log(1 + x * diagXtOmegaX)
                                          + x * XtOmegar ** 2 / (2 * (1 + x * diagXtOmegaX))
                                          + logpi0)
                res = minimize_scalar(negloglik, bounds=ssq_range, method='bounded')
                if res.success:
                    ssq[l] = res.x
            omega[:, l] = diagXtOmegaX + 1.0 / ssq[l]
            mu[:, l] = XtOmegar / omega[:, l]
            lbf_variable[:, l] = XtOmegar ** 2 / (2 * omega[:, l]) - 0.5 * np.log(omega[:, l] * ssq[l])
            logPIP = lbf_variable[:, l] + logpi0
            lbf[l] = ssp.logsumexp(logPIP)
            PIP[:, l] = np.exp(logPIP - lbf[l])
        if est_sigmasq or est_tausq:
            if method == 'moments':
                sigmasq, tausq = _mom_variances(PIP, mu, omega, sigmasq, tausq, n, V,
                                                Dsq, VtXty, Xty, yty, est_sigmasq, est_tausq)
            elif method == 'MLE':
                sigmasq, tausq = _mle_variances(PIP, mu, omega, sigmasq, tausq, n, V,
                                                Dsq, VtXty, yty, est_sigmasq, est_tausq,
                                                sigmasq_range, tausq_range)
            else:
                raise ValueError("method must be 'moments' or 'MLE'")
            var = tausq * Dsq + sigmasq
            diagXtOmegaX = np.sum(V ** 2 * (Dsq / var), axis=1)
            XtOmegay = V.dot(VtXty / var)
        PIP_diff = np.max(np.abs(PIP_prev - PIP))
        if verbose:
            print(f"  susie_inf iter {it}: max|dPIP|={PIP_diff:.3e} sigmasq={sigmasq:.4g} tausq={tausq:.3e}")
        if PIP_diff < PIP_tol:
            converged = True
            break

    b = np.sum(mu * PIP, axis=1)
    XtOmegaXb = V.dot(V.T.dot(b) * Dsq / var)
    XtOmegar = XtOmegay - XtOmegaXb
    alpha = tausq * XtOmegar
    return {'PIP': PIP, 'mu': mu, 'omega': omega, 'lbf': lbf,
            'lbf_variable': lbf_variable, 'ssq': ssq, 'sigmasq': float(sigmasq),
            'tausq': float(tausq), 'alpha': alpha, 'converged': converged}


def credible_sets(PIP, coverage=0.9, purity=0.5, LD=None, V=None, Dsq=None, n=None, dedup=True):
    """Credible sets from the per-effect PIP matrix, purity-filtered. Mirrors the
    reference `cred()`. Returns a list of variable-index lists."""
    PIP = np.asarray(PIP, dtype=np.float64)
    if (V is None or Dsq is None or n is None) and LD is None:
        raise ValueError("Provide LD or (V, Dsq, n) for purity filtering.")
    out = []
    for l in range(PIP.shape[1]):
        order = np.argsort(PIP[:, l])[::-1]
        ind = int(np.nonzero(np.cumsum(PIP[order, l]) >= coverage)[0].min())
        credset = order[:ind + 1]
        if len(credset) == 1:
            out.append(list(credset))
            continue
        if len(credset) < 100:
            rows = credset
        else:
            rows = np.random.RandomState(123).choice(credset, size=100, replace=False)
        if LD is not None:
            LDloc = LD[np.ix_(rows, rows)]
        else:
            LDloc = (V[rows, :] * Dsq).dot(V[rows, :].T) / n
        if np.min(np.abs(LDloc)) > purity:
            out.append(sorted(list(credset)))
    if dedup:
        seen, deduped = set(), []
        for c in out:
            key = tuple(c)
            if key not in seen:
                seen.add(key); deduped.append(list(c))
        out = deduped
    return out


def susie_inf_from_data(X, y, L=10, covariates=None, method='moments',
                        est_tausq=True, coverage=0.95, min_abs_corr=0.5, **kwargs):
    """Convenience wrapper: run SuSiE-inf from individual-level data, matching the
    (X, y) interface of `tensorqtl.susie.susie`. Residualizes on covariates,
    column-standardizes, builds z / meansq / LD, fits, and returns the fit dict
    augmented with `cs` (purity-filtered credible sets) and `pip` (max PIP per
    variant across effects).

    Args:
        X: [N, p] genotype/dosage matrix.
        y: [N] phenotype.
        covariates: optional [N, k] covariates to project out of both X and y.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    N, p = X.shape
    if covariates is not None:
        C = np.asarray(covariates, dtype=np.float64)
        C = np.column_stack([np.ones(N), C])
        Q, _ = np.linalg.qr(C)
        X = X - Q @ (Q.T @ X)
        y = y - Q @ (Q.T @ y)
    # column-standardize genotypes; center/scale y to unit variance
    Xc = X - X.mean(0); sx = Xc.std(0); sx[sx == 0] = 1.0
    Xs = Xc / sx
    yc = y - y.mean(); sy = yc.std(); yc = yc / (sy if sy > 0 else 1.0)
    z = Xs.T.dot(yc) / np.sqrt(N)
    meansq = float(yc.dot(yc) / N)
    LD = (Xs.T.dot(Xs)) / N
    fit = susie_inf(z, meansq, N, L=L, LD=LD, method=method, est_tausq=est_tausq, **kwargs)
    fit['cs'] = credible_sets(fit['PIP'], coverage=coverage, purity=min_abs_corr, LD=LD, n=N)
    fit['pip'] = fit['PIP'].max(axis=1)
    return fit

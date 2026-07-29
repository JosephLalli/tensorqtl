"""
Regression tests for tensorqtl.mrash (Mr.ASH coordinate-ascent solver), the
polygenic-background fitter used by SuSiE-ash.

The checked-in pinned end-to-end SuSiE-ash fixture exercises `_caisa` through
the individual-data integration. The checks below are dependency-light
arithmetic and solver property tests (no R needed); they are not by themselves
a multi-case direct oracle for every Mr.ASH option.
"""
import sys
import numpy as np
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tensorqtl'))
import mrash


def _design(n, p, seed):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, p)
    return X - X.mean(0)


def _caisa_reference(X, w, sa2, pi, beta, r, sigma2, order, max_iter,
                     min_iter, convtol, epstol, method_q, update_pi, update_sigma):
    """Pre-optimization arithmetic for exact-regression comparison."""
    n, p = X.shape
    K = sa2.shape[0]
    beta = beta.astype(np.float64).copy()
    r = r.astype(np.float64).copy()
    pi = pi.astype(np.float64).copy()
    sa2 = sa2.astype(np.float64)
    w = w.astype(np.float64)
    with np.errstate(divide="ignore"):
        inv_sa2 = np.where(sa2 > 0, 1.0 / sa2, np.inf)
    S2inv = 1.0 / (inv_sa2[:, None] + w[None, :])
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
            r += xj * beta[j]
            muj = bjwj * s2inv_j
            muj[0] = 0.0
            phij = (np.log(piold + epstol) - np.log(1.0 + sa2 * wj) / 2.0
                    + muj * (bjwj / 2.0 / sigma2))
            phij = np.exp(phij - phij.max())
            phij = phij / phij.sum()
            pi += phij / p
            beta[j] = phij @ muj
            r += -xj * beta[j]
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
    return {"beta": beta, "sigma2": float(sigma2), "pi": pi,
            "iter": it, "varobj": varobj[:it]}


def test_default_grid():
    """Default sa2 grid: 25 points, spike at 0, scaled by n/median(w)."""
    w = np.full(30, 80.0)
    sa2 = mrash.default_sa2_grid(w, 100)
    assert sa2.shape == (25,)
    assert sa2[0] == 0.0
    assert np.all(sa2[1:] > 0) and np.all(np.diff(sa2) >= 0)


def test_recovers_single_signal():
    """Mr.ASH assigns a substantial effect to a strong causal variant and shrinks
    the rest toward 0."""
    rng = np.random.RandomState(0)
    n, p, causal = 200, 30, 7
    X = _design(n, p, 0)
    Xs = X / X.std(0)
    y = 1.5 * Xs[:, causal] + 0.5 * rng.randn(n)
    out = mrash.mr_ash(X, y, update_sigma=True)
    b = out['beta']
    assert abs(b[causal]) == np.max(np.abs(b))          # causal is the top effect
    assert abs(b[causal]) > 0.5
    others = np.delete(np.abs(b), causal)
    assert others.max() < abs(b[causal]) / 2            # rest much smaller


def test_null_shrinks_to_near_zero():
    """With no signal, Mr.ASH shrinks all effects toward 0 (pi mass on the spike)."""
    rng = np.random.RandomState(4)
    n, p = 150, 40
    X = _design(n, p, 4)
    y = rng.randn(n)
    out = mrash.mr_ash(X, y, update_sigma=True)
    assert np.max(np.abs(out['beta'])) < 0.2
    assert out['pi'][0] > 0.5                            # most mass on the spike component


def test_outputs_well_formed():
    """pi is a valid distribution, sigma2 positive, residual reduced vs null."""
    rng = np.random.RandomState(2)
    n, p = 120, 25
    X = _design(n, p, 2)
    y = 1.0 * (X[:, 3] / X[:, 3].std()) + 0.6 * rng.randn(n)
    out = mrash.mr_ash(X, y, update_sigma=True)
    assert abs(out['pi'].sum() - 1.0) < 1e-10 and np.all(out['pi'] >= 0)
    assert out['sigma2'] > 0
    yc = y - y.mean()
    rss_fit = float(((yc - (X - X.mean(0)) @ out['beta']) ** 2).sum())
    assert rss_fit < float((yc ** 2).sum())             # explains some variance


def test_fixed_pi_no_update():
    """update_pi=False leaves pi at its initial value."""
    rng = np.random.RandomState(6)
    n, p = 90, 20
    X = _design(n, p, 6)
    y = (X[:, 2] / X[:, 2].std()) + 0.5 * rng.randn(n)
    K = 25
    pi0 = np.full(K, 1.0 / K)
    out = mrash.mr_ash(X, y, pi=pi0.copy(), update_pi=False, update_sigma=False, sigma2=1.0)
    assert np.allclose(out['pi'], pi0)


def test_intercept_reconstructs_predictions_on_raw_design():
    """The returned intercept and coefficients operate on the uncentered X."""
    rng = np.random.RandomState(8)
    X = rng.randn(100, 8) + np.arange(8)
    y = 4.0 + X @ rng.randn(8) + 0.5 * rng.randn(100)
    out = mrash.mr_ash(X, y)
    fitted_raw = out['intercept'] + X @ out['beta']
    fitted_centered = y.mean() + (X - X.mean(0)) @ out['beta']
    assert np.allclose(fitted_raw, fitted_centered)


@pytest.mark.parametrize(
    "X, y, match",
    [
        (np.ones((20, 2)), np.arange(20.0), "constant predictors"),
        (np.ones(20), np.arange(20.0), "two-dimensional"),
        (np.ones((20, 2)), np.arange(19.0), "same number of samples"),
    ],
)
def test_rejects_invalid_designs(X, y, match):
    with pytest.raises(ValueError, match=match):
        mrash.mr_ash(X, y)


def test_matches_manual_single_variant():
    """One coordinate update reduces to the closed-form mixture posterior mean."""
    # single variant, single non-spike component -> hand-computable
    n = 40
    rng = np.random.RandomState(11)
    x = rng.randn(n); x -= x.mean()
    y = 0.8 * x + 0.3 * rng.randn(n); y -= y.mean()
    w = float((x * x).sum())
    sa2 = np.array([0.0, 1.0])
    out = mrash._caisa(x[:, None], np.array([w]), sa2, np.array([0.5, 0.5]),
                       np.array([0.0]), y.copy(), 1.0, None, 1, 1, 1e-8, 1e-12, "sigma_dep_q", True, False)
    # manual updatebetaj
    bjwj = float(x @ y)
    s2inv = np.array([1e-12, 1.0 / (1.0 + w)])
    muj = bjwj * s2inv; muj[0] = 0.0
    phij = np.log(np.array([0.5, 0.5]) + 1e-12) - np.log(1 + sa2 * w) / 2 + muj * (bjwj / 2.0)
    phij = np.exp(phij - phij.max()); phij /= phij.sum()
    beta_manual = float(phij @ muj)
    assert abs(out['beta'][0] - beta_manual) < 1e-12


@pytest.mark.parametrize("seed, ordered", [(31, False), (32, True)])
def test_caisa_optimization_preserves_reference_arithmetic(seed, ordered):
    """Log hoisting and in-place spike removal retain every solver output exactly."""
    rng = np.random.RandomState(seed)
    n, p, K, max_iter = 37, 9, 5, 6
    X = _design(n, p, seed)
    w = (X * X).sum(0)
    sa2 = np.r_[0.0, np.exp(rng.uniform(-2.0, 1.0, K - 1))]
    pi = rng.dirichlet(np.ones(K))
    beta = rng.randn(p) / 10.0
    r = rng.randn(n) - X @ beta
    order = (rng.permutation(p * max_iter) % p) if ordered else None
    args = (X, w, sa2, pi, beta, r, 1.3, order, max_iter, max_iter,
            1e-20, 1e-12, "sigma_indep_q", True, True)
    reference = _caisa_reference(*args)
    optimized = mrash._caisa(*args)
    assert optimized["iter"] == reference["iter"]
    for key in ("beta", "pi", "varobj"):
        assert np.array_equal(optimized[key], reference[key])
    assert optimized["sigma2"] == reference["sigma2"]

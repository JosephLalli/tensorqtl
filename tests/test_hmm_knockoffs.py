"""
Tests / validity gate for the HMM (fastPHASE-style) knockoff generator.

STATUS: WORK IN PROGRESS. The DMC/HMM knockoff reimplementation is swap-
exchangeable only for small p -- the swap-test total-variation distance grows
with p, indicating a compounding bug in the forward partition-function (Z)
recursion. These tests:
  - assert validity at small p (where it IS correct), and
  - DOCUMENT the large-p failure (xfail), so the generator cannot be mistaken
    for finished. When the Z bug is fixed, the xfail test should start passing
    and can be promoted to a hard assertion.

The swap-exchangeability test is the ground truth for a valid knockoff: swap each
column with its knockoff and check the joint law is unchanged (TV -> 0 up to
Monte-Carlo noise, which shrinks like 1/sqrt(n)).
"""

import numpy as np
import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import tensorqtl.knockoffs as ko


def _random_chain(p, K, seed):
    rng = np.random.default_rng(seed)
    init_p = rng.dirichlet(np.ones(K))
    Q = np.array([rng.dirichlet(np.ones(K), size=K) for _ in range(p - 1)])
    return init_p, Q


def _sample_chain(n, init_p, Q, seed):
    rng = np.random.default_rng(seed)
    p = len(Q) + 1
    K = len(init_p)
    X = np.empty((n, p), dtype=int)
    X[:, 0] = rng.choice(K, n, p=init_p)
    for j in range(1, p):
        for x in range(K):
            m = X[:, j - 1] == x
            if m.any():
                X[m, j] = rng.choice(K, m.sum(), p=Q[j - 1][x])
    return X


def _mean_swap_tv(X, Xt, base):
    """Mean over columns of TV(joint, joint-with-column-c-swapped)."""
    n, p = X.shape
    tvs = []
    for c in range(p):
        Xs = X.copy(); Xts = Xt.copy()
        Xs[:, c], Xts[:, c] = Xt[:, c].copy(), X[:, c].copy()
        key = (Xs * (base ** np.arange(p))).sum(1) * (base ** p) + \
              (Xts * (base ** np.arange(p))).sum(1)
        sw = np.bincount(key, minlength=(base ** p) ** 2) / n
        tvs.append(0.5 * np.abs(_joint(X, Xt, base) - sw).sum())
    return float(np.mean(tvs))


def _joint(A, B, base):
    p = A.shape[1]
    key = (A * (base ** np.arange(p))).sum(1) * (base ** p) + \
          (B * (base ** np.arange(p))).sum(1)
    return np.bincount(key, minlength=(base ** p) ** 2) / len(A)


class TestDMCKnockoff:

    def test_marginals_match(self):
        """Knockoff has the correct per-position marginal law (necessary cond.)."""
        p, K = 4, 3
        init_p, Q = _random_chain(p, K, seed=1)
        n = 60000
        X = _sample_chain(n, init_p, Q, seed=2)
        Xt = ko.dmc_knockoffs(X, init_p, Q, seed=3)
        marg = [init_p]
        for j in range(p - 1):
            marg.append(marg[-1] @ Q[j])
        for j in range(p):
            obs = np.bincount(Xt[:, j], minlength=K) / n
            assert np.allclose(obs, marg[j], atol=0.02), f"j={j}"

    @pytest.mark.parametrize("p", [2, 3, 4])
    def test_swap_exchangeable_small_p(self, p):
        """Valid knockoff at small p: swap-test TV ~ 0 (Monte-Carlo noise)."""
        K = 3
        init_p, Q = _random_chain(p, K, seed=5)
        n = 120000
        X = _sample_chain(n, init_p, Q, seed=6)
        Xt = ko.dmc_knockoffs(X, init_p, Q, seed=7)
        tv = _mean_swap_tv(X, Xt, base=K)
        # threshold scales with #cells / n; generous but well below the ~0.2 of
        # a fundamentally broken sampler.
        assert tv < 0.04, f"p={p}: swap-TV={tv:.4f} (should be ~0)"

    @pytest.mark.xfail(reason="known compounding Z-recursion bug at large p; "
                              "generator is WIP and invalid for p>~4",
                       strict=True)
    def test_swap_exchangeable_large_p(self):
        """DOCUMENTS the failure: at p=8 the swap-TV is large (invalid)."""
        p, K = 8, 3
        init_p, Q = _random_chain(p, K, seed=5)
        n = 120000
        X = _sample_chain(n, init_p, Q, seed=6)
        Xt = ko.dmc_knockoffs(X, init_p, Q, seed=7)
        tv = _mean_swap_tv(X, Xt, base=K)
        assert tv < 0.04, f"p={p}: swap-TV={tv:.4f}"


class TestHMMKnockoff:

    def test_hmm_swap_exchangeable_small_p(self):
        """HMM knockoff valid at small p (p=2 exhaustive-ish check)."""
        p, K, E = 2, 2, 2
        init_p = np.array([0.6, 0.4])
        Q = np.array([[[0.7, 0.3], [0.2, 0.8]]])
        emission_p = np.array([[[0.8, 0.3], [0.2, 0.7]],
                               [[0.6, 0.35], [0.4, 0.65]]])  # [p,E,K]
        rng = np.random.default_rng(1)
        n = 200000
        # sample observed X from the HMM
        H = np.empty((n, p), int)
        H[:, 0] = rng.choice(K, n, p=init_p)
        H[:, 1] = [rng.choice(K, p=Q[0][h]) for h in H[:, 0]]
        X = np.empty((n, p), int)
        for j in range(p):
            X[:, j] = [rng.choice(E, p=emission_p[j][:, h]) for h in H[:, j]]
        Xt = ko.hmm_knockoffs(X, init_p, Q, emission_p, seed=3)
        tv = _mean_swap_tv(X, Xt, base=E)
        assert tv < 0.02, f"HMM p=2 swap-TV={tv:.4f}"


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))

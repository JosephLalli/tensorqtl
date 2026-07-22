"""
Tests for the HMM / discrete-Markov-chain (DMC) knockoff generators.

These implement Sesia, Sabatti & Candes (2019, Biometrika) Algorithms 1 & 2.
Validity is swap-exchangeability of (X, X_knockoff): swapping any variable with
its knockoff must leave the joint law unchanged.

IMPORTANT testing note: a naive check that computes the total-variation distance
between the FULL joint of (X, X_knockoff) and its column-swapped version has a
noise floor that grows with p, because the number of joint cells (|X|^{2p}) grows
exponentially while the sample size does not. That artifact was once mistaken for
a "compounding bug." The correct, noise-robust check is either (a) compare the
swap-TV against the split-half noise floor of the same joint, or (b) use
LOW-ORDER (pairwise) swap statistics, whose cell counts are small and which are
well estimated at moderate n. We use the pairwise check here -- it is sensitive,
fast, and stays flat in p for a valid knockoff.
"""

import numpy as np
import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import tensorqtl.knockoffs as ko


def _vec_chain(n, p, K, initP, Q, rng):
    X = np.empty((n, p), dtype=int)
    X[:, 0] = rng.choice(K, n, p=initP)
    for j in range(1, p):
        cdf = np.cumsum(Q[j - 1], 1)
        u = rng.random(n)
        X[:, j] = (u[:, None] > cdf[X[:, j - 1]]).sum(1).clip(max=K - 1)
    return X


def _pairwise_swap_maxtv(X, Xt, K):
    """
    Max over all column pairs (a,b) of the TV between the (col_a, col_b) 2-way
    table and its version with column a swapped real<->knockoff. For a valid
    knockoff this is ~ Monte-Carlo noise and does NOT grow with p.
    """
    N, p = X.shape
    aug = np.concatenate([X, Xt], 1)   # 0..p-1 real, p..2p-1 knockoff

    def tab(i, j):
        c = aug[:, i] * K + aug[:, j]
        return np.bincount(c, minlength=K * K) / N

    worst = 0.0
    for a in range(p):
        for b in range(p):
            if a == b:
                continue
            worst = max(worst, 0.5 * np.abs(tab(a, b) - tab(a + p, b)).sum())
            worst = max(worst, 0.5 * np.abs(tab(a, b + p) - tab(a + p, b)).sum())
    return worst


class TestDMCKnockoff:

    def test_marginals_match(self):
        rng = np.random.default_rng(0)
        p, K = 8, 3
        initP = rng.dirichlet(np.ones(K))
        Q = np.array([rng.dirichlet(np.ones(K), size=K) for _ in range(p - 1)])
        n = 100000
        X = _vec_chain(n, p, K, initP, Q, np.random.default_rng(1))
        Xt = ko.dmc_knockoffs(X, initP, Q, seed=2)
        marg = [initP]
        for j in range(p - 1):
            marg.append(marg[-1] @ Q[j])
        for j in range(p):
            obs = np.bincount(Xt[:, j], minlength=K) / n
            assert np.allclose(obs, marg[j], atol=0.02), f"j={j} marginal off"

    @pytest.mark.parametrize("p", [6, 12, 20])
    def test_swap_exchangeable_scales_with_p(self, p):
        """Pairwise-swap-TV stays ~ Monte-Carlo noise and FLAT as p grows."""
        rng = np.random.default_rng(5)
        K = 3
        initP = rng.dirichlet(np.ones(K))
        Q = np.array([rng.dirichlet(np.ones(K), size=K) for _ in range(p - 1)])
        n = 150000
        X = _vec_chain(n, p, K, initP, Q, np.random.default_rng(1))
        Xt = ko.dmc_knockoffs(X, initP, Q, seed=2)
        tv = _pairwise_swap_maxtv(X, Xt, K)
        # noise scale ~ few / sqrt(n); assert well below a generous bound and,
        # crucially, that it does NOT blow up with p.
        assert tv < 0.02, f"p={p}: pairwise-swap-TV={tv:.4f}"

    def test_swap_tv_below_full_joint_noise_floor(self):
        """
        Sanity that the classic full-joint swap-TV, while nonzero, tracks the
        split-half noise floor (i.e. it is sampling noise, not bias). K=2 keeps
        the joint small enough to bincount.
        """
        rng = np.random.default_rng(5)
        p, K = 8, 2
        initP = rng.dirichlet(np.ones(K))
        Q = np.array([rng.dirichlet(np.ones(K), size=K) for _ in range(p - 1)])
        n = 400000
        X = _vec_chain(n, p, K, initP, Q, np.random.default_rng(1))
        Xt = ko.dmc_knockoffs(X, initP, Q, seed=2)
        pw = K ** np.arange(p)
        M = (K ** p) ** 2

        def dist(A, B):
            c = (A @ pw).astype(np.int64) * (K ** p) + (B @ pw).astype(np.int64)
            return np.bincount(c, minlength=M) / len(A)

        base = dist(X, Xt)
        floor = 0.5 * np.abs(dist(X[:n // 2], Xt[:n // 2]) -
                             dist(X[n // 2:], Xt[n // 2:])).sum()
        sw = []
        for c in range(p):
            Xs, Xts = X.copy(), Xt.copy()
            Xs[:, c], Xts[:, c] = Xt[:, c].copy(), X[:, c].copy()
            sw.append(0.5 * np.abs(base - dist(Xs, Xts)).sum())
        # swap-TV must not exceed the pure-noise split-half floor by much
        assert np.mean(sw) <= 1.5 * floor, \
            f"swap-TV={np.mean(sw):.4f} exceeds noise floor={floor:.4f} -> bias"


class TestHMMKnockoff:

    def _model(self, p, K, E, seed):
        rng = np.random.default_rng(seed)
        initP = rng.dirichlet(np.ones(K))
        Q = np.array([rng.dirichlet(np.ones(K), size=K) for _ in range(p - 1)])
        emit = np.zeros((p, E, K))
        for j in range(p):
            for k in range(K):
                emit[j][:, k] = rng.dirichlet(np.ones(E))
        return initP, Q, emit

    def _sample(self, n, p, K, E, initP, Q, emit, rng):
        H = _vec_chain(n, p, K, initP, Q, rng)
        X = np.empty((n, p), dtype=int)
        for j in range(p):
            cdf = np.cumsum(emit[j].T, 1)
            X[:, j] = (rng.random(n)[:, None] > cdf[H[:, j]]).sum(1).clip(max=E - 1)
        return X

    @pytest.mark.parametrize("p", [8, 16])
    def test_hmm_swap_exchangeable(self, p):
        K, E = 3, 3
        initP, Q, emit = self._model(p, K, E, seed=5)
        n = 150000
        X = self._sample(n, p, K, E, initP, Q, emit, np.random.default_rng(1))
        Xt = ko.hmm_knockoffs(X, initP, Q, emit, seed=7)
        tv = _pairwise_swap_maxtv(X, Xt, E)
        assert tv < 0.02, f"HMM p={p}: pairwise-swap-TV={tv:.4f}"

    def test_hmm_marginals_match(self):
        p, K, E = 10, 3, 3
        initP, Q, emit = self._model(p, K, E, seed=2)
        n = 120000
        X = self._sample(n, p, K, E, initP, Q, emit, np.random.default_rng(1))
        Xt = ko.hmm_knockoffs(X, initP, Q, emit, seed=3)
        pH = [initP]
        for j in range(p - 1):
            pH.append(pH[-1] @ Q[j])
        for j in range(p):
            obs = np.bincount(Xt[:, j], minlength=E) / n
            true = emit[j] @ pH[j]
            assert np.allclose(obs, true, atol=0.02), f"j={j} obs marginal off"


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))

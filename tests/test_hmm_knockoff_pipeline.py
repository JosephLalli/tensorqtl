"""
Tests for the HMM-knockoff *fitting* and *pipeline wiring* (step 1 of the
knockoff-calibrated SuSiE plan): fit an HMM from genotypes (Baum-Welch EM),
generate chromosome-COHERENT knockoff draws, and slice them per gene inside
susie.map_egenes_knockoffs.

Separation of concerns in the assertions below:
  * fit_hmm quality is tested against GROUND TRUTH -- a fitted-model knockoff
    must be as swap-valid as one built from the true parameters (the fitter is
    sound), and the EM log-likelihood must be monotone.
  * Model-X knockoffs are exact for the FITTED distribution; validity against the
    real data-generating law depends on fit quality (Barber-Candes-Samworth).
    For the single-chain genotype HMM applied to *diploid* dosages this is a
    misspecified model, so we check swap-validity empirically against a
    Monte-Carlo noise bound, not an exact zero.

The pairwise-swap-TV helper is reused from test_hmm_knockoffs (it is the
noise-robust validity statistic; a naive full-joint swap-TV has a floor that
grows with p -- see that module's docstring).
"""

import numpy as np
import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import tensorqtl.knockoffs as ko
import tensorqtl.susie as susie
from tests.test_hmm_knockoffs import _pairwise_swap_maxtv, _vec_chain
from tests.hmm_genotype_simulator import simulate_hmm_genotypes
from tests.test_knockoffs_calibration import _simulate_cis_dataset


def _sample_single_chain(N, p, K, E, initP, Q, emit, seed):
    """Sample observations from a true single-chain HMM (hidden chain + emit)."""
    rng = np.random.default_rng(seed)
    H = _vec_chain(N, p, K, initP, Q, rng)
    X = np.empty((N, p), dtype=int)
    for j in range(p):
        cdf = np.cumsum(emit[j].T, 1)
        u = rng.random(N)
        X[:, j] = (u[:, None] > cdf[H[:, j]]).sum(1).clip(max=E - 1)
    return X


def _random_hmm(p, K, E, seed):
    rng = np.random.default_rng(seed)
    initP = rng.dirichlet(np.ones(K))
    # self-transition bias so states persist (real LD) and are identifiable
    Q = np.stack([np.stack([rng.dirichlet(np.ones(K) * 0.5 + 3 * np.eye(K)[a])
                            for a in range(K)]) for _ in range(p - 1)])
    emit = np.stack([np.stack([rng.dirichlet(np.ones(E)) for k in range(K)], 1)
                     for j in range(p)])
    return initP, Q, emit


class TestFitHMM:

    def test_loglik_monotone(self):
        """Baum-Welch log-likelihood must be non-decreasing every iteration."""
        initP, Q, emit = _random_hmm(p=20, K=3, E=3, seed=0)
        X = _sample_single_chain(1500, 20, 3, 3, initP, Q, emit, seed=1)
        fit = ko.fit_hmm(X, K=3, E=3, n_iter=25, seed=2)
        ll = fit['loglik']
        assert all(ll[i + 1] >= ll[i] - 1e-6 for i in range(len(ll) - 1)), \
            f"loglik not monotone: {ll}"
        assert ll[-1] > ll[0]

    def test_params_strictly_positive(self):
        """Pseudocount must keep Q and emissions > 0 (sampler divides by them)."""
        initP, Q, emit = _random_hmm(p=12, K=4, E=3, seed=3)
        X = _sample_single_chain(600, 12, 4, 3, initP, Q, emit, seed=4)
        fit = ko.fit_hmm(X, K=4, E=3, n_iter=10, seed=5)
        assert (fit['Q'] > 0).all()
        assert (fit['emission_p'] > 0).all()
        assert np.allclose(fit['Q'].sum(2), 1.0)
        assert np.allclose(fit['emission_p'].sum(1), 1.0)
        assert np.isclose(fit['init_p'].sum(), 1.0)

    def test_recovers_marginals(self):
        """A knockoff from the fitted model matches per-site data marginals."""
        initP, Q, emit = _random_hmm(p=18, K=3, E=3, seed=6)
        X = _sample_single_chain(3000, 18, 3, 3, initP, Q, emit, seed=7)
        fit = ko.fit_hmm(X, K=3, E=3, n_iter=30, seed=8)
        Xt = ko.hmm_knockoffs(X, fit['init_p'], fit['Q'], fit['emission_p'], seed=9)
        for j in [0, 9, 17]:
            a = np.bincount(X[:, j], minlength=3) / X.shape[0]
            b = np.bincount(Xt[:, j], minlength=3) / X.shape[0]
            assert np.allclose(a, b, atol=0.03), f"j={j}: {a} vs {b}"

    def test_fitted_knockoff_as_valid_as_truth(self):
        """
        The core fitter-quality claim: a knockoff built from the ESTIMATED
        parameters is (to Monte-Carlo tolerance) as swap-exchangeable as one
        built from the TRUE parameters. This isolates fitter quality from the
        sampler and from any model misspecification.
        """
        p, K, E, N = 20, 3, 3, 4000
        initP, Q, emit = _random_hmm(p, K, E, seed=10)
        X = _sample_single_chain(N, p, K, E, initP, Q, emit, seed=11)
        fit = ko.fit_hmm(X, K=K, E=E, n_iter=40, seed=12)
        tv_true = _pairwise_swap_maxtv(
            X, ko.hmm_knockoffs(X, initP, Q, emit, seed=13), E)
        tv_fit = _pairwise_swap_maxtv(
            X, ko.hmm_knockoffs(X, fit['init_p'], fit['Q'], fit['emission_p'],
                                seed=13), E)
        # fitted must not be meaningfully worse than ground truth
        assert tv_fit <= tv_true + 0.02, \
            f"fitted swap-TV {tv_fit:.4f} >> true-param {tv_true:.4f}"


class TestChromosomeKnockoffs:

    def test_shape_and_state_range(self):
        geno, _, _ = simulate_hmm_genotypes(n_snps=20, N=300, seed=1, K=5)
        G = geno.T.astype(int)
        draws = ko.chromosome_hmm_knockoffs(G, K=6, M=3, E=3, n_em_iter=10, seed=2)
        assert draws.shape == (3, 300, 20)
        assert draws.min() >= 0 and draws.max() <= 2

    def test_coherent_across_overlapping_windows(self):
        """
        The property the whole design turns on: a shared variant gets the SAME
        knockoff value in every gene whose window contains it, because the draw
        is generated once for the chromosome and sliced. Two overlapping window
        slices must agree exactly on their shared columns.
        """
        geno, _, _ = simulate_hmm_genotypes(n_snps=30, N=250, seed=3, K=5)
        G = geno.T.astype(int)
        draws = ko.chromosome_hmm_knockoffs(G, K=6, M=1, E=3, n_em_iter=10, seed=4)
        win_a = draws[0][:, 5:20]     # variants 5..19
        win_b = draws[0][:, 12:30]    # variants 12..29 (overlap 12..19)
        assert np.array_equal(win_a[:, 7:], win_b[:, :8]), \
            "overlapping windows disagree on shared variants -> not coherent"

    def test_swap_valid_on_hmm_genotypes(self):
        """
        End-to-end validity on realistic diploid HMM genotypes: the fitted
        single-chain genotype-HMM knockoff is swap-exchangeable within a
        Monte-Carlo noise bound (~E/sqrt(N) for the 2-way tables).
        """
        N = 1200
        geno, _, _ = simulate_hmm_genotypes(n_snps=22, N=N, seed=5, K=5)
        G = geno.T.astype(int)
        draws = ko.chromosome_hmm_knockoffs(G, K=8, M=1, E=3, n_em_iter=30, seed=6)
        tv = _pairwise_swap_maxtv(G, draws[0], 3)
        noise = 3.0 / np.sqrt(N)   # generous 2-way-table Monte-Carlo scale
        assert tv < noise, f"swap-TV {tv:.4f} exceeds noise bound {noise:.4f}"

    def test_prefit_params_passthrough(self):
        """Supplying params skips EM and the draw reflects those params."""
        initP, Q, emit = _random_hmm(p=15, K=3, E=3, seed=14)
        X = _sample_single_chain(1000, 15, 3, 3, initP, Q, emit, seed=15)
        params = {'init_p': initP, 'Q': Q, 'emission_p': emit}
        draws = ko.chromosome_hmm_knockoffs(X, M=1, E=3, params=params, seed=16)
        # marginals should match the supplied-emission-implied marginals
        pH = [initP]
        for j in range(14):
            pH.append(pH[-1] @ Q[j])
        for j in [0, 7, 14]:
            obs = np.bincount(draws[0][:, j], minlength=3) / X.shape[0]
            true = emit[j] @ pH[j]
            assert np.allclose(obs, true, atol=0.04), f"j={j}: {obs} vs {true}"


class TestPipelineHMM:

    def test_runs_and_recovers_causal(self):
        """map_egenes_knockoffs(knockoff='hmm') runs and finds the causal genes
        without false positives on a small clean dataset."""
        d = _simulate_cis_dataset(n_genes=8, N=200, p_per_gene=12, seed=1,
                                  n_causal_genes=3, causal_effect=2.0)
        causal = {f"G{g}" for g in range(3)}
        eg, loc, diag = susie.map_egenes_knockoffs(
            d['genotype_df'], d['variant_df'], d['phenotype_df'], d['pos_df'],
            d['cov_df'], fdr=0.1, n_knockoffs=2, knockoff='hmm', hmm_K=6,
            hmm_em_iter=12, window=1_000_000, L=5, max_iter=100, verbose=False,
            seed=2, localize=False)
        sel = set(eg[eg['selected']]['phenotype_id'])
        assert diag['W_per_draw'].shape == (2, 8)
        assert causal.issubset(sel), f"missed causal genes: {causal - sel}"
        assert not (sel - causal), f"false discoveries: {sel - causal}"

    def test_hmm_params_passthrough(self):
        """The pipeline accepts pre-fit per-chromosome HMM params (chrom -> dict)
        and uses them instead of running EM."""
        d = _simulate_cis_dataset(n_genes=6, N=180, p_per_gene=10, seed=9,
                                  n_causal_genes=2, causal_effect=2.5)
        # fit params ourselves for the single chromosome and pass them in
        rows = np.where(d['variant_df']['chrom'].values == 'chr1')[0]
        gix = np.array([d['genotype_df'].columns.tolist().index(i)
                        for i in d['phenotype_df'].columns])
        G = d['genotype_df'].values[rows[0]:rows[-1] + 1][:, gix]
        G = np.rint(G).clip(0, 2).astype(int).T
        params = ko.fit_hmm(G, K=5, E=3, n_iter=10, seed=0)
        eg, _, diag = susie.map_egenes_knockoffs(
            d['genotype_df'], d['variant_df'], d['phenotype_df'], d['pos_df'],
            d['cov_df'], fdr=0.1, n_knockoffs=1, knockoff='hmm',
            hmm_params={'chr1': params}, window=1_000_000, L=5, max_iter=100,
            verbose=False, seed=3, localize=False)
        assert diag['W_per_draw'].shape == (1, 6)
        # causal genes should have the strongest (positive) W
        assert set(f"G{g}" for g in range(2)).issubset(
            set(eg[eg['selected']]['phenotype_id']))


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))

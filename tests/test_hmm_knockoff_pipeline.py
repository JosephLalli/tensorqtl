"""
Tests for the HMM-knockoff *fitting* and *pipeline wiring* (step 1 of the
knockoff-calibrated SuSiE plan): fit an HMM from genotypes, generate
chromosome-COHERENT knockoff draws, and slice them per gene inside
susie.map_egenes_knockoffs.

Two exact constructions of the true diploid law are tested, plus a cheap
approximation:
  * Route 1 (method='genotype'): the pair-state genotype HMM fit from UNPHASED
    dosages (Kronecker transition + Bernoulli-convolution emission). Exact for
    the true diploid law; O(N p K^4).
  * Route 2 (method='haplotype'): fit a haplotype HMM (E=2), knock off each
    haplotype, sum to a knockoff dosage. Exact; O(N p K^2); needs phase; also
    yields the phased knockoffs the two-channel hapmixQTL model needs.
  * single-chain (method='single_chain'): one K-state chain with a free E=3
    emission -- cheapest, approximate, NOT the exact diploid law.

Separation of concerns in the assertions:
  * fitter quality is tested against GROUND TRUTH -- a fitted-model knockoff must
    be as swap-valid as one from the true parameters, and EM log-likelihood must
    be monotone.
  * model-X knockoffs are exact for the FITTED distribution; validity against the
    real data-generating law depends on fit quality (Barber-Candes-Samworth), so
    end-to-end validity on the diploid simulator is checked against a Monte-Carlo
    noise bound, not an exact zero.

The pairwise-swap-TV helper is reused from test_hmm_knockoffs (it is the
noise-robust validity statistic; a naive full-joint swap-TV has a floor that
grows with p -- see that module's docstring).
"""

import numpy as np
import pandas as pd
import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import tensorqtl.knockoffs as ko
import tensorqtl.susie as susie
from tests.test_hmm_knockoffs import _pairwise_swap_maxtv, _vec_chain
from tests.hmm_genotype_simulator import simulate_hmm_genotypes
from tests.test_knockoffs_calibration import _simulate_cis_dataset


def _phased_cis_dataset(n_genes, N, p_per_gene, seed, n_causal_genes=0,
                        causal_effect=2.0, K=5):
    """
    A cis dataset with genuine PHASED haplotypes (for method='haplotype'): each
    gene's variants are simulated from the fastPHASE HMM, dosage = xL + xR.
    Returns the usual frames plus (xL_df, xR_df).
    """
    rng = np.random.RandomState(seed)
    samples = [f"S{i:04d}" for i in range(N)]
    vids, chroms, poss = [], [], []
    xL_blocks, xR_blocks, geno_blocks = [], [], []
    pheno, pids, pchr, ppos = [], [], [], []
    for g in range(n_genes):
        base = 1_000_000 * g + 10_000
        _, _, info = simulate_hmm_genotypes(n_snps=p_per_gene, N=N, seed=seed * 97 + g,
                                            K=K, return_phased=True)
        xL, xR = info['xL'], info['xR']          # [p, N] in {0,1}
        xL_blocks.append(xL); xR_blocks.append(xR); geno_blocks.append(xL + xR)
        for j in range(p_per_gene):
            vids.append(f"g{g}_v{j}"); chroms.append("chr1"); poss.append(base + j * 100)
        y = rng.randn(N)
        if g < n_causal_genes:
            y = y + causal_effect * (xL + xR)[5]
        pheno.append(y); pids.append(f"G{g}"); pchr.append("chr1")
        ppos.append(base + (p_per_gene // 2) * 100)
    geno = np.vstack(geno_blocks).astype(float)
    genotype_df = pd.DataFrame(geno, index=vids, columns=samples)
    xL_df = pd.DataFrame(np.vstack(xL_blocks).astype(float), index=vids, columns=samples)
    xR_df = pd.DataFrame(np.vstack(xR_blocks).astype(float), index=vids, columns=samples)
    variant_df = pd.DataFrame({'chrom': chroms, 'pos': poss}, index=vids)
    phenotype_df = pd.DataFrame(np.array(pheno), index=pids, columns=samples)
    pos_df = pd.DataFrame({'chr': pchr, 'pos': ppos}, index=pids)
    cov_df = pd.DataFrame(rng.randn(N, 2), index=samples, columns=['PC1', 'PC2'])
    return dict(genotype_df=genotype_df, variant_df=variant_df,
                phenotype_df=phenotype_df, pos_df=pos_df, cov_df=cov_df,
                xL_df=xL_df, xR_df=xR_df)


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


class TestGenotypePairHMM:
    """Route 1: the exact diploid pair-state genotype HMM (unphased)."""

    def test_pair_hmm_construction(self):
        """Kronecker transition + convolution emission are well-formed."""
        K, p = 4, 6
        rng = np.random.default_rng(0)
        ih = rng.dirichlet(np.ones(K))
        Q = np.stack([0.7 * np.eye(K) + 0.3 * rng.dirichlet(np.ones(K), size=K)
                      for _ in range(p - 1)])
        theta = rng.uniform(0.05, 0.95, size=(p, K))
        init_pair, Q_pair, emit_pair = ko.build_genotype_pair_hmm(ih, Q, theta)
        assert init_pair.shape == (K * K,) and Q_pair.shape == (p - 1, K * K, K * K)
        assert emit_pair.shape == (p, 3, K * K)
        assert np.isclose(init_pair.sum(), 1.0)
        assert np.allclose(Q_pair.sum(2), 1.0)          # rows of transition
        assert np.allclose(emit_pair.sum(1), 1.0)       # dosage emission over {0,1,2}

    def test_genotype_em_monotone_and_recovers_marginals(self):
        geno, _, _ = simulate_hmm_genotypes(n_snps=18, N=1200, seed=3, K=5)
        G = geno.T.astype(int)
        fit = ko.fit_genotype_hmm(G, K=5, n_iter=25, seed=1)
        ll = fit['loglik']
        assert all(ll[i + 1] >= ll[i] - 1e-6 for i in range(len(ll) - 1))
        Xt = ko.genotype_hmm_knockoffs(G, params=fit)[0]
        for j in [0, 9, 17]:
            a = np.bincount(G[:, j], minlength=3) / G.shape[0]
            b = np.bincount(Xt[:, j], minlength=3) / G.shape[0]
            assert np.allclose(a, b, atol=0.04), f"j={j}: {a} vs {b}"

    def test_swap_valid_on_diploid_genotypes(self):
        """The exact route is swap-valid within Monte-Carlo noise on realistic
        diploid HMM genotypes."""
        N = 1500
        geno, _, _ = simulate_hmm_genotypes(n_snps=20, N=N, seed=5, K=5)
        G = geno.T.astype(int)
        draws = ko.genotype_hmm_knockoffs(G, K=5, M=1, n_em_iter=30, seed=6)
        tv = _pairwise_swap_maxtv(G, draws[0], 3)
        noise = 3.0 / np.sqrt(N)
        assert tv < noise, f"Route-1 swap-TV {tv:.4f} exceeds noise {noise:.4f}"


class TestHaplotypeKnockoffs:
    """Route 2: phased haplotype knockoffs summed to a dosage."""

    def test_phased_outputs_consistent(self):
        geno, _, info = simulate_hmm_genotypes(n_snps=15, N=400, seed=7, K=5,
                                               return_phased=True)
        xL, xR = info['xL'].T.astype(int), info['xR'].T.astype(int)
        dos, (xkL, xkR) = ko.haplotype_hmm_knockoffs(
            xL, xR, K=5, M=2, n_em_iter=15, seed=1, return_phased=True)
        assert dos.shape == (2, 400, 15)
        assert set(np.unique(xkL)).issubset({0, 1})
        assert set(np.unique(xkR)).issubset({0, 1})
        assert np.array_equal(xkL + xkR, dos)           # dosage = sum of haplotypes

    def test_swap_valid_on_diploid_genotypes(self):
        N = 1500
        geno, _, info = simulate_hmm_genotypes(n_snps=20, N=N, seed=5, K=5,
                                               return_phased=True)
        G = geno.T.astype(int)
        xL, xR = info['xL'].T.astype(int), info['xR'].T.astype(int)
        draws = ko.haplotype_hmm_knockoffs(xL, xR, K=6, M=1, n_em_iter=30, seed=6)
        tv = _pairwise_swap_maxtv(G, draws[0], 3)
        noise = 3.0 / np.sqrt(N)
        assert tv < noise, f"Route-2 swap-TV {tv:.4f} exceeds noise {noise:.4f}"


class TestChromosomeCoherence:
    """The coherence property (shared by all methods): a chromosome draw sliced
    into overlapping windows agrees on the shared variants."""

    def test_coherent_across_overlapping_windows(self):
        geno, _, _ = simulate_hmm_genotypes(n_snps=30, N=250, seed=3, K=5)
        G = geno.T.astype(int)
        draws = ko.chromosome_hmm_knockoffs(G, K=5, M=1, n_em_iter=8, seed=4,
                                            method='genotype')
        win_a = draws[0][:, 5:20]     # variants 5..19
        win_b = draws[0][:, 12:30]    # variants 12..29 (overlap 12..19)
        assert np.array_equal(win_a[:, 7:], win_b[:, :8]), \
            "overlapping windows disagree on shared variants -> not coherent"

    def test_single_chain_shape_and_range(self):
        geno, _, _ = simulate_hmm_genotypes(n_snps=20, N=300, seed=1, K=5)
        G = geno.T.astype(int)
        draws = ko.chromosome_hmm_knockoffs(G, K=6, M=3, n_em_iter=8, seed=2,
                                            method='single_chain')
        assert draws.shape == (3, 300, 20)
        assert draws.min() >= 0 and draws.max() <= 2


class TestPipelineHMM:

    @pytest.mark.parametrize("method", ['genotype', 'single_chain'])
    def test_runs_and_recovers_causal(self, method):
        """map_egenes_knockoffs(knockoff='hmm') runs for each unphased method and
        finds the causal genes without false positives."""
        d = _simulate_cis_dataset(n_genes=8, N=200, p_per_gene=12, seed=1,
                                  n_causal_genes=3, causal_effect=2.0)
        causal = {f"G{g}" for g in range(3)}
        eg, loc, diag = susie.map_egenes_knockoffs(
            d['genotype_df'], d['variant_df'], d['phenotype_df'], d['pos_df'],
            d['cov_df'], fdr=0.1, n_knockoffs=1, knockoff='hmm',
            hmm_method=method, hmm_K=5, hmm_em_iter=10, window=1_000_000, L=5,
            max_iter=100, verbose=False, seed=2, localize=False)
        sel = set(eg[eg['selected']]['phenotype_id'])
        assert diag['W_per_draw'].shape == (1, 8)
        assert causal.issubset(sel), f"[{method}] missed causal: {causal - sel}"
        assert not (sel - causal), f"[{method}] false discoveries: {sel - causal}"

    def test_haplotype_method_recovers_causal(self):
        """method='haplotype' with phased inputs recovers causal genes."""
        d = _phased_cis_dataset(n_genes=6, N=200, p_per_gene=12, seed=1,
                                n_causal_genes=2, causal_effect=2.0)
        causal = {f"G{g}" for g in range(2)}
        eg, _, diag = susie.map_egenes_knockoffs(
            d['genotype_df'], d['variant_df'], d['phenotype_df'], d['pos_df'],
            d['cov_df'], fdr=0.1, n_knockoffs=1, knockoff='hmm',
            hmm_method='haplotype', phased_haplotypes=(d['xL_df'], d['xR_df']),
            hmm_K=5, hmm_em_iter=10, window=1_000_000, L=5, max_iter=100,
            verbose=False, seed=2, localize=False)
        sel = set(eg[eg['selected']]['phenotype_id'])
        assert diag['W_per_draw'].shape == (1, 6)
        assert causal.issubset(sel), f"missed causal: {causal - sel}"

    def test_haplotype_method_requires_phased(self):
        d = _simulate_cis_dataset(n_genes=4, N=150, p_per_gene=10, seed=9,
                                  n_causal_genes=1)
        with pytest.raises(ValueError, match="phased_haplotypes"):
            susie.map_egenes_knockoffs(
                d['genotype_df'], d['variant_df'], d['phenotype_df'], d['pos_df'],
                d['cov_df'], knockoff='hmm', hmm_method='haplotype',
                window=1_000_000, L=5, verbose=False, localize=False)

    def test_hmm_params_passthrough(self):
        """The pipeline accepts pre-fit per-chromosome haplotype params (Route 1
        format) and uses them instead of running EM."""
        d = _simulate_cis_dataset(n_genes=6, N=180, p_per_gene=10, seed=9,
                                  n_causal_genes=2, causal_effect=2.5)
        rows = np.where(d['variant_df']['chrom'].values == 'chr1')[0]
        gix = np.array([d['genotype_df'].columns.tolist().index(i)
                        for i in d['phenotype_df'].columns])
        G = d['genotype_df'].values[rows[0]:rows[-1] + 1][:, gix]
        G = np.rint(G).clip(0, 2).astype(int).T
        params = ko.fit_genotype_hmm(G, K=5, n_iter=8, seed=0)
        eg, _, diag = susie.map_egenes_knockoffs(
            d['genotype_df'], d['variant_df'], d['phenotype_df'], d['pos_df'],
            d['cov_df'], fdr=0.1, n_knockoffs=1, knockoff='hmm',
            hmm_method='genotype', hmm_params={'chr1': params}, window=1_000_000,
            L=5, max_iter=100, verbose=False, seed=3, localize=False)
        assert diag['W_per_draw'].shape == (1, 6)
        assert set(f"G{g}" for g in range(2)).issubset(
            set(eg[eg['selected']]['phenotype_id']))


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))

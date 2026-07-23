"""
Tests for the KFc-style continuous statistic + empirical mirror-null eGene FDR
(knockoffs.marginal_importance / gene_W_marginal / mirror_select_egenes).

This is the redesign that fixes problem B (docs/calibration_findings.md): the
maxPIP-difference statistic is a point mass at 0 under a null gene (an ATOM) that
breaks its Binomial null. The KFc statistic W_g = -log10(min p_real) -
-log10(min p_knockoff) is CONTINUOUS (no atom, no ties), and the empirical
mirror-null knockoff+ threshold (Barber & Candes 2015) selects with no
distributional assumption and never selects W_g <= 0.

Fast tests: no-atom continuity, marginal_importance sanity, mirror_select_egenes
unit behaviour on synthetic continuous W. A slow test runs the full statistic on
realistic HMM genotypes with HMM knockoffs and checks FDR control + power.
"""

import numpy as np
import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import tensorqtl.knockoffs as ko


def _block(N, p, seed):
    rng = np.random.RandomState(seed)
    Z = rng.randn(N, 5); load = np.zeros((p, 5))
    for j in range(p):
        load[j, j % 5] = 1.0
    X = Z @ load.T + 0.5 * rng.randn(N, p)
    return X, rng


class TestMarginalImportance:

    def test_signal_higher_than_null(self):
        X, rng = _block(200, 30, 0)
        y_null = rng.randn(200)
        c = 5
        xc = (X[:, c] - X[:, c].mean()) / X[:, c].std()
        y_sig = np.sqrt(0.1 / 0.9) * xc + rng.randn(200)
        imp_null = ko.marginal_importance(X, y_null)
        imp_sig = ko.marginal_importance(X, y_sig)
        assert imp_sig > imp_null
        assert imp_null >= 0 and imp_sig >= 0

    def test_gene_W_continuous_no_atom(self):
        """Null W over many genes has NO atom at 0 (contrast with maxPIP)."""
        rng = np.random.RandomState(1)
        W = []
        for g in range(120):
            X, _ = _block(150, 25, 10 + g)
            Xk, _ = _block(150, 25, 5000 + g)   # independent block ~ a crude knockoff
            y = rng.randn(150)
            W.append(ko.gene_W_marginal(X, Xk, y))
        W = np.array(W)
        assert np.mean(np.abs(W) < 1e-9) == 0.0, "continuous statistic must have no atom"
        assert np.unique(np.round(W, 6)).size >= 100, "W should be continuous"


class TestMirrorSelect:

    def test_wle0_never_selected(self):
        """Genes with W <= 0 are never selected."""
        rng = np.random.RandomState(2)
        W = np.concatenate([np.full(10, 5.0), rng.randn(200)])  # 10 strong + noise
        gid = [f"g{i}" for i in range(len(W))]
        out = ko.mirror_select_egenes(gid, W, q=0.1)
        sel_idx = [int(g[1:]) for g in out['selected']]
        assert all(W[i] > 0 for i in sel_idx)

    def test_null_symmetric_controls(self):
        """Pure-null symmetric continuous W: mirror knockoff+ selects ~nothing,
        realized false-selection rate <= q on average."""
        rng = np.random.RandomState(3)
        n, q = 300, 0.1
        gid = [f"g{i}" for i in range(n)]
        rates = []
        for _ in range(40):
            W = rng.randn(n)                    # symmetric about 0 -> pure null
            out = ko.mirror_select_egenes(gid, W, q=q)
            rates.append(out['n_selected'] / n)  # all null -> any selection false
        assert np.mean(rates) <= q + 0.02

    def test_power_and_fdr_synthetic(self):
        """Signal genes W>0 shifted; realized FDR controlled with power."""
        rng = np.random.RandomState(4)
        n_sig, n_null, q = 60, 240, 0.1
        gid = [f"S{i}" for i in range(n_sig)] + [f"N{i}" for i in range(n_null)]
        fdps, powers = [], []
        for _ in range(30):
            W = np.concatenate([rng.randn(n_sig) + 3.0, rng.randn(n_null)])
            out = ko.mirror_select_egenes(gid, W, q=q)
            sel = set(out['selected'])
            R = len(sel)
            fp = sum(1 for g in sel if g.startswith('N'))
            tp = sum(1 for g in sel if g.startswith('S'))
            fdps.append(fp / max(R, 1)); powers.append(tp / n_sig)
        assert np.mean(fdps) <= q + 0.03
        assert np.mean(powers) >= 0.5


@pytest.mark.slow
class TestEndToEndHMM:

    def test_fdr_controlled_hmm_genotypes(self):
        """Full statistic on realistic HMM genotypes with matched HMM knockoffs:
        no atom, realized FDR controlled, non-trivial power."""
        from hmm_genotype_simulator import simulate_hmm_genotypes
        q = 0.1
        fdps, powers, atoms = [], [], []
        for rep in range(4):
            rng = np.random.RandomState(200 + rep)
            n_genes, n_sig = 100, 40
            gid = [f"G{g}" for g in range(n_genes)]
            truth = np.zeros(n_genes, bool); truth[:n_sig] = True
            W = []
            for g in range(n_genes):
                geno, _, info = simulate_hmm_genotypes(30, 200, seed=(200 + rep) * 1000 + g)
                G = geno.T.astype(np.float64)
                poly = ~(G == G[0]).all(0); G = G[:, poly]
                y = rng.randn(200)
                if g < n_sig:
                    maf = info['maf'][poly]; elig = np.where(maf > 0.05)[0]
                    if elig.size:
                        c = elig[rng.randint(elig.size)]
                        xc = (G[:, c] - G[:, c].mean()) / (G[:, c].std() + 1e-9)
                        y = y + np.sqrt(0.10 / 0.90) * xc
                Xk = ko.genotype_hmm_knockoffs(np.rint(G).astype(np.int64), K=5, M=1,
                                               n_em_iter=6, seed=7000 + g)[0].astype(np.float64)
                W.append(ko.gene_W_marginal(G, Xk, y))
            W = np.array(W)
            atoms.append(np.mean(np.abs(W) < 1e-9))
            out = ko.mirror_select_egenes(gid, W, q=q)
            smask = np.array([g in set(out['selected']) for g in gid])
            R = smask.sum()
            fdps.append((smask & ~truth).sum() / max(R, 1))
            powers.append((smask & truth).sum() / n_sig)
        assert np.mean(atoms) == 0.0, "no atom on the continuous statistic"
        assert np.mean(fdps) <= q + 0.05, f"realized FDR {np.mean(fdps):.3f}"
        assert np.mean(powers) >= 0.4, f"power {np.mean(powers):.2f}"


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v', '-s']))

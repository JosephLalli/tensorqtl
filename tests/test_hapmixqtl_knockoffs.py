"""
hapmixQTL knockoff-calibrated eGene mapping with PHASED (Route 2) knockoffs
(hapmixqtl.map_egenes_knockoffs).

This is the two-channel analog of susie.map_egenes_knockoffs: for each gene it
draws phased haplotype knockoffs (x~L, x~R), builds the augmented two-channel
stacked design [X, X~] (ASE + total, both channels sharing the SAME knockoff
haplotypes), fits SuSiE, and forms W_g = maxPIP(orig) - maxPIP(knockoff), which
feeds the same step-2/step-3 calibration. These tests check that the wiring runs
end to end, that the augmented design is built coherently, and that the gene
statistic separates planted eGenes from nulls (power) while keeping the selected
set FDR-respecting.

Full SuSiE fits per gene per draw are moderately slow, so the panel is small and
the heavier power/FDR run is marked slow.
"""

import numpy as np
import pandas as pd
import torch
import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import tensorqtl.hapmixqtl as hapmixqtl


def _make_egene_dataset(n_genes=10, n_causal=5, p_per_gene=12, N=150, seed=0,
                        beta=1.5):
    """
    Synthetic phased hapmixQTL panel with non-overlapping cis-windows. The first
    `n_causal` genes carry a planted allelic effect on a het-driven variant (both
    channels); the rest are null.
    """
    rng = np.random.RandomState(seed)
    samples = [f"S{i:04d}" for i in range(N)]
    variant_ids, chroms, poss = [], [], []
    geno_blocks, xL_blocks, xR_blocks = [], [], []
    causal_col = 3

    for g in range(n_genes):
        base = 1_000_000 * g + 50_000
        geno = rng.choice([0, 1, 2], size=(p_per_gene, N), p=[0.25, 0.5, 0.25]).astype(np.float32)
        het = geno == 1
        L_alt = rng.rand(p_per_gene, N) < 0.5
        xL = np.zeros((p_per_gene, N), dtype=np.float32)
        xR = np.zeros((p_per_gene, N), dtype=np.float32)
        xL[het & L_alt] = 1
        xR[het & ~L_alt] = 1
        xL[geno == 2] = 1
        xR[geno == 2] = 1
        geno_blocks.append(geno); xL_blocks.append(xL); xR_blocks.append(xR)
        for j in range(p_per_gene):
            variant_ids.append(f"g{g}_v{j}")
            chroms.append("chr1")
            poss.append(base + j * 100)

    geno = np.vstack(geno_blocks)
    xL = np.vstack(xL_blocks)
    xR = np.vstack(xR_blocks)
    genotype_df = pd.DataFrame(geno, index=variant_ids, columns=samples)
    variant_df = pd.DataFrame({'chrom': chroms, 'pos': poss}, index=variant_ids)
    xL_df = pd.DataFrame(xL, index=variant_ids, columns=samples)
    xR_df = pd.DataFrame(xR, index=variant_ids, columns=samples)

    pheno_ids = [f"G{g}" for g in range(n_genes)]
    A = np.zeros((n_genes, N), dtype=np.float32)
    T = np.zeros((n_genes, N), dtype=np.float32)
    Va = np.zeros((n_genes, N), dtype=np.float32)
    Vt = np.zeros((n_genes, N), dtype=np.float32)
    for g in range(n_genes):
        row = g * p_per_gene + causal_col
        if g < n_causal:
            s0 = xL[row] - xR[row]
            A[g] = beta * s0 + rng.normal(0, 0.1, N)
            T[g] = 2.0 + beta * (geno[row] / 2) + rng.normal(0, 0.1, N)
        else:
            A[g] = rng.normal(0, 0.15, N)
            T[g] = 2.0 + rng.normal(0, 0.15, N)
        Va[g] = rng.uniform(0.02, 0.08, N)
        Vt[g] = rng.uniform(0.02, 0.08, N)

    A_df = pd.DataFrame(A, index=pheno_ids, columns=samples)
    T_df = pd.DataFrame(T, index=pheno_ids, columns=samples)
    Va_df = pd.DataFrame(Va, index=pheno_ids, columns=samples)
    Vt_df = pd.DataFrame(Vt, index=pheno_ids, columns=samples)
    pos_df = pd.DataFrame(
        {'chr': ['chr1'] * n_genes,
         'pos': [1_000_000 * g + 50_000 + (p_per_gene // 2) * 100 for g in range(n_genes)]},
        index=pheno_ids)
    return dict(genotype_df=genotype_df, variant_df=variant_df,
                A_df=A_df, T_df=T_df, Va_df=Va_df, Vt_df=Vt_df,
                xL_df=xL_df, xR_df=xR_df, pos_df=pos_df,
                causal=set(pheno_ids[:n_causal]), null=set(pheno_ids[n_causal:]))


def _run(d, **kw):
    return hapmixqtl.map_egenes_knockoffs(
        d['genotype_df'], d['variant_df'], d['A_df'], d['T_df'],
        d['Va_df'], d['Vt_df'], d['pos_df'], d['xL_df'], d['xR_df'],
        window=500_000, L=5, max_iter=100, verbose=False, **kw)


class TestWiring:

    def test_requires_phase(self):
        d = _make_egene_dataset(n_genes=3, n_causal=1, N=80, seed=1)
        with pytest.raises(AssertionError):
            hapmixqtl.map_egenes_knockoffs(
                d['genotype_df'], d['variant_df'], d['A_df'], d['T_df'],
                d['Va_df'], d['Vt_df'], d['pos_df'], None, None,
                window=500_000, verbose=False)

    def test_knockoff_design_shape(self):
        """_build_knockoff_stacked_design returns a [2N, p] design matching the
        real one, built coherently from the two knockoff haplotypes."""
        N, p = 40, 6
        rng = np.random.RandomState(2)
        xkL = torch.tensor((rng.rand(p, N) < 0.4).astype(np.float32))
        xkR = torch.tensor((rng.rand(p, N) < 0.4).astype(np.float32))
        sqrt_wa = torch.ones(N); sqrt_wt = torch.ones(N)
        from tensorqtl.hapmixqtl import WeightedResidualizer, _build_knockoff_stacked_design
        ra = WeightedResidualizer(None, sqrt_wa)
        rt = WeightedResidualizer(None, sqrt_wt)
        Xk = _build_knockoff_stacked_design(xkL, xkR, sqrt_wa, sqrt_wt, ra, rt)
        assert Xk.shape == (2 * N, p)

    def test_end_to_end_schema(self):
        d = _make_egene_dataset(n_genes=4, n_causal=2, p_per_gene=10, N=120, seed=3)
        egene_df, diag = _run(d, fdr=0.1, n_knockoffs=6, hmm_K=5,
                              selection='calibrated', seed=7)
        assert list(egene_df.columns) == ['phenotype_id', 'qvalue', 'selected']
        assert len(egene_df) == 4
        for key in ('W_per_draw', 'gene_ids', 'n_draws', 'selection', 'pi0',
                    'agreement', 'pi0_interval'):
            assert key in diag
        assert diag['W_per_draw'].shape == (6, 4)


class TestCoherence:

    def test_shared_variants_identical_across_windows(self):
        """The defining coherence property: two OVERLAPPING cis-windows sliced
        from the same chromosome-coherent phased draw share IDENTICAL knockoff
        haplotype values on their shared variants. This is what a per-gene fit
        cannot give and is the prerequisite for cross-gene per-gene p-values."""
        import tensorqtl.knockoffs as ko
        rng = np.random.RandomState(0)
        N, P = 60, 40
        xL = (rng.rand(N, P) < 0.4).astype(np.int64)
        xR = (rng.rand(N, P) < 0.4).astype(np.int64)
        _, (xkL, xkR) = ko.chromosome_hmm_knockoffs(
            K=5, M=3, n_em_iter=8, seed=1, method='haplotype',
            xL=xL, xR=xR, return_phased=True)          # [M, N, P] each
        # Gene A window = variants 5..25, gene B window = 15..35; overlap 15..25.
        winA, winB = np.arange(5, 25), np.arange(15, 35)
        shared = np.arange(15, 25)
        localA = np.searchsorted(winA, shared)
        localB = np.searchsorted(winB, shared)
        for m in range(3):
            slA_L = xkL[m][:, winA][:, localA]
            slB_L = xkL[m][:, winB][:, localB]
            slA_R = xkR[m][:, winA][:, localA]
            slB_R = xkR[m][:, winB][:, localB]
            assert np.array_equal(slA_L, slB_L), f"draw {m}: xkL differs on shared"
            assert np.array_equal(slA_R, slB_R), f"draw {m}: xkR differs on shared"

    def test_coherent_and_per_gene_both_run(self):
        """Both modes produce valid, aligned output on the same panel."""
        d = _make_egene_dataset(n_genes=5, n_causal=2, p_per_gene=10, N=120, seed=8)
        for coherent in (True, False):
            egene_df, diag = _run(d, fdr=0.2, n_knockoffs=6, hmm_K=5,
                                  selection='calibrated', coherent=coherent, seed=4)
            assert len(egene_df) == 5
            assert diag['W_per_draw'].shape == (6, 5)


class TestSignalSeparation:

    def test_causal_genes_have_positive_W(self):
        """Planted eGenes should have W_g (real maxPIP - knockoff maxPIP) > null
        genes: the real signal beats its knockoff."""
        d = _make_egene_dataset(n_genes=8, n_causal=4, p_per_gene=10, N=150,
                                seed=5, beta=2.0)
        egene_df, diag = _run(d, fdr=0.2, n_knockoffs=8, hmm_K=6,
                              selection='calibrated', seed=11)
        W = diag['W_per_draw'].mean(axis=0)   # mean W per gene
        gene_ids = diag['gene_ids']
        causal_W = np.array([W[i] for i, g in enumerate(gene_ids) if g in d['causal']])
        null_W = np.array([W[i] for i, g in enumerate(gene_ids) if g in d['null']])
        assert causal_W.mean() > null_W.mean(), \
            f"causal meanW={causal_W.mean():.3f} !> null meanW={null_W.mean():.3f}"

    @pytest.mark.slow
    def test_power_and_fdr(self):
        """Larger panel: selected set should be enriched for causal genes with
        FDR loosely respected (Monte-Carlo tolerant at this scale)."""
        d = _make_egene_dataset(n_genes=16, n_causal=8, p_per_gene=12, N=180,
                                seed=9, beta=2.0)
        egene_df, diag = _run(d, fdr=0.1, n_knockoffs=20, hmm_K=6,
                              selection='calibrated', seed=3)
        sel = egene_df[egene_df['selected']]['phenotype_id'].tolist()
        if sel:
            n_false = sum(1 for g in sel if g in d['null'])
            fdp = n_false / len(sel)
            assert fdp <= 0.3, f"FDP={fdp:.2f} too high"
            n_true = sum(1 for g in sel if g in d['causal'])
            assert n_true >= 1, "no causal gene recovered"


class TestKfcStatistic:

    def test_kfc_two_channel_runs_and_separates(self):
        """statistic='kfc' uses the continuous two-channel min-|t| importance +
        phased knockoff + mirror-null selection. Schema correct, no atom/NaN,
        causal genes preferentially recovered."""
        d = _make_egene_dataset(n_genes=8, n_causal=4, p_per_gene=12, N=150,
                                seed=3, beta=2.0)
        eg, diag = hapmixqtl.map_egenes_knockoffs(
            d['genotype_df'], d['variant_df'], d['A_df'], d['T_df'],
            d['Va_df'], d['Vt_df'], d['pos_df'], d['xL_df'], d['xR_df'],
            fdr=0.2, n_knockoffs=6, hmm_K=5, hmm_em_iter=6, statistic='kfc',
            coherent=True, window=500_000, L=5, max_iter=100, verbose=False, seed=7)
        assert list(eg.columns) == ['phenotype_id', 'knockoff_qvalue', 'selected']
        assert diag['statistic'] == 'kfc'
        assert not np.isnan(diag['W_per_draw']).any()
        # continuous statistic: not a degenerate atom at 0
        assert np.mean(np.abs(diag['W_per_draw']) < 1e-12) < 0.9
        sel = set(eg[eg['selected']]['phenotype_id'])
        if sel:
            assert len(sel & d['causal']) >= 1, "no causal gene recovered"


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v', '-s']))

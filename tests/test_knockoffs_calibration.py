"""
Calibration gate for the EXPERIMENTAL credible-set-level path (map_knockoffs).

NOTE: this file exercises `susie.map_knockoffs` -- the CS-level knockoff path,
which is EXPERIMENTAL / not FDR-controlled (its statistic is not swap-antisymmetric;
see docs/knockoff_susie_design.md STATUS and test_knockoffs.py::TestSwapEquivariance).
The SHIPPED, valid path is eGene-level FDR (`susie.map_egenes_knockoffs`), whose
calibration gate is tests/test_knockoffs_egenes.py. These tests remain as the
empirical characterization of the experimental CS-level path.

Model-X knockoffs guarantee FDR only with a correctly estimated knockoff
distribution. At the modest N of eQTL studies that is not automatic, so the
target FDR must be validated empirically before it is trusted. These tests are
the gate: if the null-permutation FDR does not track q here, the feature is not
ready and the fix is more shrinkage / the HMM generator / more knockoff draws.

Because a full end-to-end SuSiE fit per gene is slow, the null-FDR test uses a
moderate number of genes but is still an honest end-to-end run of map_knockoffs
(real knockoff generation + real SuSiE fits + real pooled q-values). It is
marked slow; run with `-m slow` or directly.

These tests are intentionally tolerant on exact FDR (Monte-Carlo noise at this
scale is large); they assert the *direction* and *order of magnitude* that would
distinguish a working filter from a broken one.
"""

import numpy as np
import pandas as pd
import torch
import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import tensorqtl.susie as susie


def _simulate_cis_dataset(n_genes, N, p_per_gene, seed, n_causal_genes=0,
                          causal_effect=1.5, n_factors=3, geno_noise=0.4,
                          polygenic=False, polygenic_sd=0.15):
    """
    Build a synthetic cis eQTL dataset with block-LD genotypes.

    If n_causal_genes > 0, the first that many genes get one causal variant
    (index 5 in their window); the rest are pure null (no genotype effect).

    If polygenic=True, every gene's expression is instead driven by a dense sum
    of many tiny effects across its whole window (no sparse causal variant).
    This is the regime where SuSiE miscalibrates and reports false credible sets
    (Cui et al. 2023, replication-failure-rate) -- the case a knockoff filter is
    supposed to catch. Under polygenic=True there is no "causal" variant to
    recover; every reported CS is effectively a false positive.
    """
    rng = np.random.RandomState(seed)
    samples = [f"S{i:04d}" for i in range(N)]

    variant_ids, chroms, poss, geno_blocks = [], [], [], []
    causal_variant_by_gene = {}
    for g in range(n_genes):
        base = 1_000_000 * g + 10_000
        Z = rng.randn(N, n_factors)
        load = np.zeros((p_per_gene, n_factors))
        for j in range(p_per_gene):
            load[j, j % n_factors] = 1.0
        X = Z @ load.T + geno_noise * rng.randn(N, p_per_gene)
        Xd = np.clip(np.round(X - X.min(0)), 0, 2)  # crude dosage 0/1/2
        geno_blocks.append(Xd.T)
        for j in range(p_per_gene):
            variant_ids.append(f"g{g}_v{j}")
            chroms.append("chr1")
            poss.append(base + j * 100)
        if g < n_causal_genes:
            causal_variant_by_gene[f"G{g}"] = f"g{g}_v5"
    geno = np.vstack(geno_blocks)
    genotype_df = pd.DataFrame(geno, index=variant_ids, columns=samples)
    variant_df = pd.DataFrame({'chrom': chroms, 'pos': poss}, index=variant_ids)

    pheno, pids, pchr, ppos = [], [], [], []
    for g in range(n_genes):
        base = 1_000_000 * g + 10_000
        y = rng.randn(N)
        if polygenic:
            block = geno[g * p_per_gene:(g + 1) * p_per_gene]  # p x N
            beta = rng.randn(p_per_gene) * polygenic_sd
            y = y + block.T @ beta
        elif g < n_causal_genes:
            causal_row = g * p_per_gene + 5
            y = y + causal_effect * geno[causal_row]
        pheno.append(y)
        pids.append(f"G{g}")
        pchr.append("chr1")
        ppos.append(base + (p_per_gene // 2) * 100)
    phenotype_df = pd.DataFrame(np.array(pheno), index=pids, columns=samples)
    pos_df = pd.DataFrame({'chr': pchr, 'pos': ppos}, index=pids)
    cov_df = pd.DataFrame(rng.randn(N, 2), index=samples, columns=['PC1', 'PC2'])

    return dict(genotype_df=genotype_df, variant_df=variant_df,
                phenotype_df=phenotype_df, pos_df=pos_df, cov_df=cov_df,
                causal=causal_variant_by_gene)


class TestKnockoffCalibration:

    @pytest.mark.slow
    def test_null_fdr_controlled(self):
        """
        NULL: every phenotype permuted -> no gene has signal -> every selected
        credible set is a false discovery. The realized false-selection rate at
        target FDR q must not greatly exceed q.

        This is the load-bearing gate. At this N and gene count Monte-Carlo
        noise is large, so the assertion is deliberately loose: we require that
        the filter does not grossly over-select under the null (a broken or
        non-exchangeable filter would select a large fraction of the pooled
        null CSs). Exact calibration at production scale is a separate, larger
        empirical study.
        """
        d = _simulate_cis_dataset(n_genes=40, N=250, p_per_gene=25, seed=1)
        q = 0.1
        summary_df, diag = susie.map_knockoffs(
            d['genotype_df'], d['variant_df'], d['phenotype_df'], d['pos_df'],
            d['cov_df'], fdr=q, n_knockoffs=2, permute_null=True,
            window=1_000_000, L=5, max_iter=100, verbose=False, seed=7)

        n_total = diag['n_cs_total']
        n_selected = diag['n_cs_selected']
        # Under a pure null, selections are false discoveries. With few pooled
        # CSs the knockoff+ floor often selects zero (which is perfect FDR
        # control). The failure mode we guard against is gross over-selection.
        if n_total > 0:
            frac = n_selected / n_total
            assert frac <= 0.5, \
                f"null over-selection: {n_selected}/{n_total} = {frac:.2f}"
        # And in absolute terms we should not be emitting many false CSs.
        assert n_selected <= max(2, int(0.25 * max(n_total, 1))), \
            f"too many null selections: {n_selected} of {n_total}"

    @pytest.mark.slow
    def test_polygenic_background_fdr_controlled(self):
        """
        The meaningful gate: a POLYGENIC background (dense small effects, no
        sparse causal) is the regime where standard SuSiE hallucinates credible
        sets (Cui et al. RFR). The augmented SuSiE here DOES form many CSs, so
        the knockoff filter is genuinely exercised (unlike a pure null, where
        SuSiE forms nothing and the filter has an easy job). The filter must
        keep the false-selection rate near q.
        """
        d = _simulate_cis_dataset(n_genes=50, N=250, p_per_gene=40, seed=17,
                                  polygenic=True, polygenic_sd=0.15)
        q = 0.1
        summary_df, diag = susie.map_knockoffs(
            d['genotype_df'], d['variant_df'], d['phenotype_df'], d['pos_df'],
            d['cov_df'], fdr=q, n_knockoffs=3, window=1_000_000, L=5,
            max_iter=100, verbose=False, seed=5)

        # SuSiE should hallucinate several CSs under the polygenic background
        # (if it forms none, the filter isn't being tested and the assertion is
        # vacuous -- we note that rather than silently pass).
        if diag['n_cs_total'] < 5:
            pytest.skip(f"only {diag['n_cs_total']} CSs formed; filter not stressed")
        # Every polygenic CS is effectively a false positive. Realized false
        # selections must respect the target FDR loosely (Monte-Carlo tolerance).
        frac = diag['n_cs_selected'] / diag['n_cs_total']
        assert frac <= 0.3, \
            f"polygenic over-selection: {diag['n_cs_selected']}/{diag['n_cs_total']} = {frac:.2f}"

    @pytest.mark.slow
    def test_power_on_spike_in(self):
        """
        POWER: with clear causal variants in many genes, the pooled filter
        should recover a good fraction of them in selected credible sets while
        the q-values of the causal CSs are small.
        """
        d = _simulate_cis_dataset(n_genes=30, N=300, p_per_gene=20, seed=2,
                                  n_causal_genes=20, causal_effect=2.0)
        q = 0.1
        summary_df, diag = susie.map_knockoffs(
            d['genotype_df'], d['variant_df'], d['phenotype_df'], d['pos_df'],
            d['cov_df'], fdr=q, n_knockoffs=2, window=1_000_000, L=5,
            max_iter=100, verbose=False, seed=3)

        assert diag['n_cs_total'] >= 10, \
            f"expected many CSs from 20 causal genes, got {diag['n_cs_total']}"
        # Enough discoveries that q=0.1 is reachable (detection floor 1/k).
        assert diag['n_cs_selected'] >= 5, \
            f"low power: only {diag['n_cs_selected']} selected"

        # The selected CSs should preferentially contain the true causal variants.
        if len(summary_df):
            sel = summary_df[summary_df['selected']]
            causal_vids = set(d['causal'].values())
            hit = sel['variant_id'].isin(causal_vids).any()
            assert hit, "no causal variant recovered in any selected CS"

    def test_map_knockoffs_output_schema(self):
        """Fast structural check (not marked slow): columns and dtypes."""
        d = _simulate_cis_dataset(n_genes=4, N=150, p_per_gene=15, seed=4,
                                  n_causal_genes=2, causal_effect=2.0)
        summary_df, diag = susie.map_knockoffs(
            d['genotype_df'], d['variant_df'], d['phenotype_df'], d['pos_df'],
            d['cov_df'], fdr=0.2, n_knockoffs=1, window=1_000_000, L=5,
            max_iter=100, verbose=False, seed=5)
        for col in ['phenotype_id', 'variant_id', 'af', 'cs_W',
                    'knockoff_qval', 'selected']:
            assert col in summary_df.columns
        for key in ['W_all', 'qvals', 'n_genes_used', 'n_cs_total',
                    'n_cs_selected', 'fdr']:
            assert key in diag
        if len(summary_df):
            assert summary_df['knockoff_qval'].between(0, 1).all()


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v', '-s']))

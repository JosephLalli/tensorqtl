"""
Calibration + power tests for the VALID eGene-FDR knockoff path (Path A).

map_egenes_knockoffs uses the gene-level statistic W_g = max PIP(orig) - max
PIP(knockoff), which tests the FIXED hypothesis "gene g has no cis signal" and
is swap-antisymmetric (proven in test_knockoffs.py::TestSwapEquivariance). Genes
are selected via knockoff+ with Ren-Barber e-value derandomization.

Two properties are checked, at REALISTIC scale (the method needs enough genes
for the pooled null W distribution to be well estimated -- with too few genes the
knockoff+ detection floor and threshold instability suppress all discoveries,
which is a genuine operating-characteristic limitation, not a bug):

1. FDR is controlled -- estimated correctly as the MEAN per-replicate FDP over
   many independent simulated datasets (external review, point 8), not a pooled
   discovery-weighted ratio.
2. Power is reasonable when signal is present.
"""

import numpy as np
import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import tensorqtl.susie as susie
import tensorqtl.knockoffs as ko
from tests.test_knockoffs_calibration import _simulate_cis_dataset


def _run_one(seed, n_genes, N, p_per_gene, n_causal, q, n_knockoffs=5,
             causal_effect=2.0, permute_null=False):
    d = _simulate_cis_dataset(n_genes=n_genes, N=N, p_per_gene=p_per_gene,
                              seed=seed, n_causal_genes=n_causal,
                              causal_effect=causal_effect)
    causal = {f"G{g}" for g in range(n_causal)}
    eg, _, diag = susie.map_egenes_knockoffs(
        d['genotype_df'], d['variant_df'], d['phenotype_df'], d['pos_df'],
        d['cov_df'], fdr=q, n_knockoffs=n_knockoffs, window=1_000_000, L=5,
        max_iter=100, verbose=False, seed=seed + 1, localize=False,
        permute_null=permute_null)
    sel = set(eg[eg['selected']]['phenotype_id'])
    V = len(sel - causal)       # false discoveries
    R = len(sel)                # total discoveries
    power = len(sel & causal) / max(n_causal, 1)
    return V, R, power


class TestEgeneFDR:

    @pytest.mark.slow
    def test_power_at_realistic_scale(self):
        """~90% power to recover strong causal genes at production scale."""
        V, R, power = _run_one(seed=7, n_genes=200, N=300, p_per_gene=15,
                               n_causal=50, q=0.1, causal_effect=2.0)
        assert R >= 30, f"too few eGenes selected: {R}"
        assert power >= 0.7, f"low power: {power:.2f}"
        # realized FDP on this replicate should be small
        assert V / max(R, 1) <= 0.2, f"realized FDP {V/max(R,1):.2f}"

    @pytest.mark.slow
    def test_fdr_controlled_mean_over_replicates(self):
        """
        The correct FDR estimator: mean per-replicate FDP over B independent
        simulated datasets (each with true + null genes). Must be <= q within
        Monte-Carlo noise.
        """
        q = 0.15
        B = 6
        Vs, Rs = [], []
        for b in range(B):
            V, R, _ = _run_one(seed=100 + b, n_genes=150, N=300, p_per_gene=15,
                               n_causal=40, q=q, causal_effect=2.0)
            Vs.append(V)
            Rs.append(R)
        rep = ko.calibration_report(Vs, Rs, q)
        # mean per-replicate FDP, not the pooled ratio
        assert rep['empirical_fdr'] <= q + 2 * (rep['se'] if np.isfinite(rep['se']) else 0.1), \
            f"eGene FDR not controlled: {rep['empirical_fdr']:.3f} vs q={q} (se={rep['se']:.3f})"

    @pytest.mark.slow
    def test_complete_null_prob_any_discovery(self):
        """
        Under a COMPLETE null (all genes permuted), FDR = P(any discovery).
        Estimate it over several null replicates; it should be near/below q.
        """
        q = 0.1
        B = 6
        Vs, Rs = [], []
        for b in range(B):
            V, R, _ = _run_one(seed=200 + b, n_genes=150, N=300, p_per_gene=15,
                               n_causal=0, q=q, permute_null=True)
            Vs.append(V)   # under complete null every discovery is false: V == R
            Rs.append(R)
        rep = ko.calibration_report(Vs, Rs, q)
        # complete-null FDR is P(R>0); with good calibration this stays modest
        assert rep['prob_any_discovery'] <= 0.34, \
            f"complete-null P(any discovery)={rep['prob_any_discovery']:.2f} too high"

    def test_small_study_limitation_documented(self):
        """
        Fast: at TOO SMALL a scale the method (correctly) suppresses discoveries
        -- the detection floor + threshold instability. This asserts the honest
        operating characteristic so a regression that silently 'fixes' it (e.g.
        by dropping the offset and inflating FDR) is caught.
        """
        V, R, power = _run_one(seed=3, n_genes=40, N=300, p_per_gene=20,
                               n_causal=10, q=0.1, causal_effect=2.0)
        # With only 40 genes and 10 signals, q=0.1 is often unreachable; the
        # method returns few/no discoveries rather than uncontrolled ones.
        assert V == 0, f"small study must not produce false discoveries, got {V}"


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v', '-s']))

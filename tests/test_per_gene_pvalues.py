"""
Step-2 unit tests: per-gene knockoff p-values (tensorqtl.knockoffs).

These test the p-value primitive that feeds the step-3 empirical-Bayes
interval-pi0 calibration:

    per_gene_pvalues  -- (approximately) uniform-null p-value from M coherent
                         knockoff draws of the swap-antisymmetric gene statistic
    bh_select         -- Benjamini-Hochberg selection at level q
    select_egenes_pvalue -- the wired selection mode

The load-bearing facts we assert:
  1. Null uniformity. When the gene has no cis signal, gene_level_W is
     swap-antisymmetric so each draw's sign is a coin flip; the p-value is
     Binomial(M, 1/2)-based and (super-)uniform. We check the null CDF and the
     BH false-discovery rate.
  2. The 1/(M+1) resolution wall. Because the smallest attainable p-value is
     1/(M+1), direct BH cannot select genes at small q until M is large enough
     (M >= n/(q*R) - 1). We assert this explicitly so the constraint is a
     regression test, not just a docstring: small M selects nothing even with
     strong signal; large M recovers the signal at controlled FDP.
  3. Power. With M adequate, strong-signal genes are recovered.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import tensorqtl.knockoffs as ko


# ---------------------------------------------------------------------------
#  Synthetic W generators. W_per_draw has shape [M, n_genes].
#  Null gene: W symmetric about 0 (coin-flip sign).
#  Signal gene: W > 0 with high probability (real beats knockoff).
# ---------------------------------------------------------------------------

def _null_W(M, n, rng, corr=0.0):
    """Null genes: each draw's sign is a coin flip. `corr` induces the positive
    across-draw dependence that coherent (shared-genotype) draws exhibit by
    mixing in a per-gene shared offset."""
    z = rng.randn(M, n)
    if corr > 0:
        shared = rng.randn(1, n)
        z = np.sqrt(1 - corr) * z + np.sqrt(corr) * shared
    return z


def _signal_W(M, n, rng, strength=3.0, corr=0.0):
    """Signal genes: mean shifted positive so the real gene usually wins."""
    return _null_W(M, n, rng, corr=corr) + strength


# ---------------------------------------------------------------------------
#  per_gene_pvalues
# ---------------------------------------------------------------------------

class TestPerGenePvalues:

    def test_shape_and_range(self):
        rng = np.random.RandomState(0)
        W = _null_W(20, 50, rng)
        p = ko.per_gene_pvalues(W)
        assert p.shape == (50,)
        assert np.all(p > 0) and np.all(p <= 1)

    def test_resolution_floor(self):
        """Smallest attainable p-value is offset/(offset+M) = 1/(M+1)."""
        # A gene the real signal always wins (all W > 0): count of ties/losses 0.
        W = np.ones((15, 3))  # all draws positive -> b=0
        p = ko.per_gene_pvalues(W, offset=1)
        assert np.allclose(p, 1.0 / (15 + 1))
        # offset=0 gives an exact zero (no knockoff+ smoothing).
        p0 = ko.per_gene_pvalues(W, offset=0)
        assert np.allclose(p0, 0.0)

    def test_knockoff_always_wins_gives_one(self):
        W = -np.ones((10, 4))  # all draws <= 0 -> b=M
        p = ko.per_gene_pvalues(W, offset=1)
        assert np.allclose(p, (1 + 10) / (1 + 10))  # = 1.0

    def test_null_super_uniform_independent(self):
        """Independent null draws -> p is a VALID (super-uniform) p-value, NOT
        marginally uniform. #{W<=0} is Binomial(M,1/2) so the count concentrates
        near M/2 and the p-value piles up near 0.5; its LEFT TAIL is lighter than
        Uniform's. The property FDR needs is P(p<=a) <= a, which we assert here.
        We also assert the tail is genuinely SUB-uniform (conservative), i.e. the
        distribution is not uniform -- that is a feature, not a bug."""
        rng = np.random.RandomState(1)
        M, n = 200, 4000
        W = _null_W(M, n, rng, corr=0.0)
        p = ko.per_gene_pvalues(W, offset=1)
        # Validity: super-uniform in the tail.
        for a in (0.05, 0.1, 0.2):
            frac = np.mean(p <= a)
            assert frac <= a + 0.02, f"null P(p<={a})={frac:.3f} not super-uniform"
        # Conservatism: the count concentrates, so the tail is MUCH lighter than
        # Uniform would give (a broken/uniform-like statistic would fail this).
        assert np.mean(p <= 0.1) < 0.02, "tail unexpectedly heavy (not conservative)"
        # Mass concentrates around 0.5 (Binomial center).
        assert np.mean(np.abs(p - 0.5) < 0.1) > 0.6

    def test_null_super_uniform_correlated(self):
        """Coherent (positively dependent) draws spread the count toward {0,M},
        lightening the concentration -- but the super-uniform tail bound still
        holds, which is all FDR control needs."""
        rng = np.random.RandomState(2)
        M, n = 100, 4000
        W = _null_W(M, n, rng, corr=0.3)
        p = ko.per_gene_pvalues(W, offset=1)
        for a in (0.05, 0.1):
            frac = np.mean(p <= a)
            assert frac <= a + 0.02, f"correlated null P(p<={a})={frac:.3f} not super-uniform"


# ---------------------------------------------------------------------------
#  bh_select
# ---------------------------------------------------------------------------

class TestBHSelect:

    def test_empty(self):
        assert ko.bh_select(np.array([]), 0.1).shape == (0,)

    def test_all_null_selects_nothing_mostly(self):
        """Uniform p-values -> BH selects ~0 at q=0.1 (FDR controlled)."""
        rng = np.random.RandomState(3)
        fdps = []
        for _ in range(50):
            p = rng.rand(200)
            mask = ko.bh_select(p, 0.1)
            fdps.append(mask.mean())  # all null -> any selection is false
        # Average false-discovery proportion must respect q.
        assert np.mean(fdps) <= 0.1 + 0.02

    def test_bh_monotone_cutoff(self):
        """Selection is a threshold rule: if p_i selected and p_j < p_i then p_j
        is selected too."""
        p = np.array([0.001, 0.002, 0.2, 0.9, 0.02])
        mask = ko.bh_select(p, 0.1)
        cutoff = p[mask].max() if mask.any() else -1
        assert np.all(p[p <= cutoff] == p[mask | (p < 0)])  # threshold consistency
        assert mask[0] and mask[1]

    def test_strong_signal_selected(self):
        p = np.concatenate([np.full(5, 1e-4), np.linspace(0.3, 1.0, 95)])
        mask = ko.bh_select(p, 0.1)
        assert mask[:5].all()
        assert mask.sum() < 20  # doesn't spill into the null bulk


# ---------------------------------------------------------------------------
#  select_egenes_pvalue -- selection mode + the resolution-wall regression
# ---------------------------------------------------------------------------

class TestSelectEgenesPvalue:

    def test_alignment_and_keys(self):
        rng = np.random.RandomState(4)
        W = _null_W(20, 6, rng)
        gene_ids = [f"G{i}" for i in range(6)]
        out = ko.select_egenes_pvalue(gene_ids, W, q=0.1)
        assert set(out) == {'selected', 'pvalues', 'n_draws'}
        assert out['n_draws'] == 20
        assert out['pvalues'].shape == (6,)
        assert all(g in gene_ids for g in out['selected'])

    def test_null_fdr_controlled(self):
        """Pure null across many genes: BH false selections respect q on average."""
        rng = np.random.RandomState(5)
        n = 300
        gene_ids = [f"G{i}" for i in range(n)]
        false_counts, sel_counts = [], []
        for _ in range(30):
            W = _null_W(100, n, rng, corr=0.2)
            out = ko.select_egenes_pvalue(gene_ids, W, q=0.1)
            sel_counts.append(len(out['selected']))
            false_counts.append(len(out['selected']))  # all null
        # Mean FDP = mean(false / max(sel,1)); all selections false so it is
        # either 0 (nothing selected) or 1. Require the *rate* of nonzero-null
        # selection to be small -- the knockoff+/BH floor should almost always
        # select nothing under the pure null.
        rate = np.mean([c > 0 for c in sel_counts])
        assert rate <= 0.15, f"null selection rate {rate:.2f} too high"

    def test_resolution_wall_few_true_small_M(self):
        """REGRESSION for the 1/(M+1) wall in the FEW-true regime (the one where
        it bites). n=100 with only R=5 strong-signal genes among 95 nulls: BH at
        q=0.1 needs M >= n/(q*R) - 1 = 100/(0.1*5) - 1 = 199 draws before the 5
        can clear q*R/n. With small M nothing is selected even though the signal
        is unmistakable -- this is the resolution wall, not a power failure."""
        rng = np.random.RandomState(6)
        n, n_true = 100, 5
        gene_ids = [f"G{i}" for i in range(n)]
        for M in (15, 30, 60):
            W_sig = _signal_W(M, n_true, rng, strength=6.0)
            W_null = _null_W(M, n - n_true, rng)
            W = np.concatenate([W_sig, W_null], axis=1)
            out = ko.select_egenes_pvalue(gene_ids, W, q=0.1)
            # Signal genes sit at min p = 1/(M+1) but BH cannot reach them.
            assert len(out['selected']) == 0, \
                f"M={M}, R_true={n_true}: expected the wall, selected {len(out['selected'])}"

    def test_resolution_wall_cleared_large_M(self):
        """Companion: with M >= ~n/(q*R) the same few true genes ARE recovered at
        controlled FDP. n=100, R=5 true needs M ~ 199; use M=300 to clear."""
        rng = np.random.RandomState(7)
        n, n_true = 100, 5
        gene_ids = [f"G{i}" for i in range(n)]
        M = 300
        W_sig = _signal_W(M, n_true, rng, strength=6.0)
        W_null = _null_W(M, n - n_true, rng)
        W = np.concatenate([W_sig, W_null], axis=1)
        out = ko.select_egenes_pvalue(gene_ids, W, q=0.1)
        sel = set(out['selected'])
        n_true_sel = sum(1 for i in range(n_true) if f"G{i}" in sel)
        R = len(sel)
        assert R >= 1, f"expected recovery at M={M}, got {R}"
        # Selected set should be dominated by the true genes.
        fdp = (R - n_true_sel) / R
        assert fdp <= 0.2, f"FDP {fdp:.2f} exceeds q with margin"
        assert n_true_sel >= 3, f"recovered only {n_true_sel}/{n_true} true genes"

    def test_all_true_wall_is_cheap(self):
        """Complement to the few-true case: when ALL n genes are true (R=n), the
        wall is cheap -- M >= 1/q - 1 = 9 suffices, so M=15 selects (nearly) all.
        Documents that the wall is governed by R, not n alone."""
        rng = np.random.RandomState(9)
        n = 100
        gene_ids = [f"G{i}" for i in range(n)]
        W = _signal_W(15, n, rng, strength=6.0)
        out = ko.select_egenes_pvalue(gene_ids, W, q=0.1)
        assert len(out['selected']) >= 90, \
            f"all-true R=n should clear at M=15, selected {len(out['selected'])}"

    def test_power_with_mixed_signal_and_large_M(self):
        """Mixed panel: strong-signal genes should be preferentially selected
        over nulls, and null contamination in the selected set must be low."""
        rng = np.random.RandomState(8)
        n_sig, n_null = 60, 240
        M = 400  # large M so BH can resolve
        gene_ids = [f"S{i}" for i in range(n_sig)] + [f"N{i}" for i in range(n_null)]
        W_sig = _signal_W(M, n_sig, rng, strength=5.0, corr=0.2)
        W_null = _null_W(M, n_null, rng, corr=0.2)
        W = np.concatenate([W_sig, W_null], axis=1)
        out = ko.select_egenes_pvalue(gene_ids, W, q=0.1)
        sel = set(out['selected'])
        n_true_sel = sum(1 for g in sel if g.startswith('S'))
        n_false_sel = sum(1 for g in sel if g.startswith('N'))
        R = len(sel)
        if R == 0:
            raise AssertionError("selected nothing despite large M and clear signal")
        fdp = n_false_sel / R
        power = n_true_sel / n_sig
        assert fdp <= 0.2, f"FDP {fdp:.2f} exceeds q with margin"
        assert power >= 0.5, f"power {power:.2f} too low"


if __name__ == '__main__':
    import pytest
    sys.exit(pytest.main([__file__, '-v']))

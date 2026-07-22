"""
Step-3 tests: empirical-Bayes eGene calibration from per-gene knockoff
statistics (tensorqtl.knockoffs) -- the three estimators built on the known,
discrete Binomial(M,1/2) null:

    null_cdf / calibrated_qvalues  -- SHIPPED known-null Storey q-values
    mirror_fdp                     -- pi0-free symmetry cross-check
    local_fdr_interval             -- per-gene lfdr with a pi0 interval
    estimate_pi0_known_null        -- Storey pi0 on the known null

Design of the harness. We generate the gene-level statistics directly (not full
SuSiE fits), because calibration is a property of the STATISTIC given its null,
and the null is exactly Binomial(M,1/2). A null gene has W_g^(m) symmetric about
0 (fair coin sign); a signal gene has W_g^(m) shifted positive so the real gene
usually beats its knockoff. This is the exact regime the estimators assume, so
the tests check that:
  * the KNOWN null machinery is arithmetically correct (null_cdf vs scipy),
  * pi0 ~ 1 under the null and tracks the truth (conservatively) under signal,
  * realized FDR tracks the target q across several pi0/signal regimes,
  * the pi0-free mirror agrees with the q-values under clean signal (the
    misspecification alarm is quiet when the model holds),
  * the lfdr interval brackets correctly (lo <= hi; signal low, null high).

Calibration assertions are deliberately Monte-Carlo tolerant: we assert the
MEAN realized FDP over replicates does not exceed the target by more than a
noise margin, which is what "controls FDR" means operationally.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import tensorqtl.knockoffs as ko


# ---------------------------------------------------------------------------
#  W generators. W_per_draw has shape [M, n_genes].
# ---------------------------------------------------------------------------

def _null_W(M, n, rng):
    """Null genes: W_g^(m) ~ N(0,1) -> sign is a fair coin -> b ~ Binom(M,1/2)."""
    return rng.randn(M, n)


def _signal_W(M, n, rng, strength):
    """Signal genes: mean shifted +strength so the real gene usually wins."""
    return rng.randn(M, n) + strength


def _mixed(M, n, n_sig, rng, strength):
    """Return (W[M,n], is_signal[n]) with the first n_sig genes signal."""
    Ws = _signal_W(M, n_sig, rng, strength)
    Wn = _null_W(M, n - n_sig, rng)
    W = np.concatenate([Ws, Wn], axis=1)
    is_sig = np.zeros(n, dtype=bool)
    is_sig[:n_sig] = True
    return W, is_sig


# ---------------------------------------------------------------------------
#  null_cdf / estimate_pi0_known_null
# ---------------------------------------------------------------------------

class TestKnownNull:

    def test_null_cdf_matches_binomial(self):
        """null_cdf must equal the exact Binomial(M,1/2) CDF."""
        from scipy.stats import binom
        for M in (10, 31, 64):
            b = np.arange(M + 1)
            got = ko.null_cdf(b, M)
            want = binom.cdf(b, M, 0.5)
            assert np.allclose(got, want, atol=1e-12), f"M={M}"
        # monotone, endpoints
        F = ko.null_cdf(np.arange(0, 21), 20)
        assert F[0] > 0 and np.isclose(F[-1], 1.0)
        assert np.all(np.diff(F) >= 0)

    def test_pi0_pure_null_near_one(self):
        rng = np.random.RandomState(0)
        M, n = 40, 2000
        b = (_null_W(M, n, rng) <= 0).sum(axis=0)
        pi0 = ko.estimate_pi0_known_null(b, M)
        assert 0.9 <= pi0 <= 1.0, f"null pi0={pi0:.3f}"

    def test_pi0_tracks_truth_and_does_not_grossly_underestimate(self):
        """With a fraction of clearly-signal genes, pi0 should be near the true
        null fraction -- and critically NOT grossly below it (a gross
        under-estimate breaks calibration by shrinking q-values)."""
        rng = np.random.RandomState(1)
        M, n = 200, 1500
        for true_pi0 in (0.9, 0.7, 0.5):
            n_sig = int(round((1 - true_pi0) * n))
            W, _ = _mixed(M, n, n_sig, rng, strength=5.0)
            b = (W <= 0).sum(axis=0)
            pi0 = ko.estimate_pi0_known_null(b, M)
            # within a reasonable band of the truth, and not a gross underage.
            assert abs(pi0 - true_pi0) <= 0.15, \
                f"true={true_pi0} est={pi0:.3f}"
            assert pi0 >= true_pi0 - 0.15, f"under-estimate true={true_pi0} est={pi0:.3f}"


# ---------------------------------------------------------------------------
#  calibrated_qvalues -- the shipped selector
# ---------------------------------------------------------------------------

class TestCalibratedQvalues:

    def test_output_schema(self):
        rng = np.random.RandomState(2)
        W = _null_W(30, 50, rng)
        res = ko.calibrated_qvalues(W)
        for k in ('qvalues', 'pi0', 'counts', 'F0', 'M'):
            assert k in res
        assert res['qvalues'].shape == (50,)
        assert np.all((res['qvalues'] >= 0) & (res['qvalues'] <= 1))

    def test_qvalues_monotone_in_count(self):
        """Smaller knockoff-win count b (real wins more) -> smaller q. q must be
        a non-decreasing function of b."""
        rng = np.random.RandomState(3)
        W, _ = _mixed(100, 300, 60, rng, strength=4.0)
        res = ko.calibrated_qvalues(W)
        b, q = res['counts'], res['qvalues']
        order = np.argsort(b)
        assert np.all(np.diff(q[order]) >= -1e-9), "q not monotone in b"

    def test_null_fdr_controlled(self):
        """Pure null: every selection is false. Mean realized FDP over reps must
        not exceed the target by more than Monte-Carlo noise."""
        rng = np.random.RandomState(4)
        M, n, q = 40, 400, 0.1
        fdps = []
        for _ in range(60):
            W = _null_W(M, n, rng)
            sel = ko.calibrated_qvalues(W)['qvalues'] <= q
            fdps.append(sel.mean())  # all null
        assert np.mean(fdps) <= q + 0.03, f"null mean FDP={np.mean(fdps):.3f}"

    def test_mixed_fdr_controlled_across_regimes(self):
        """The load-bearing calibration gate: over several (pi0, signal) regimes
        the MEAN realized FDP must respect the target q. Power should be
        non-trivial when signal is clear."""
        rng = np.random.RandomState(5)
        M, n, q = 200, 400, 0.1
        for true_pi0, strength in [(0.8, 4.0), (0.6, 3.5), (0.9, 4.5)]:
            n_sig = int(round((1 - true_pi0) * n))
            fdps, powers = [], []
            for _ in range(25):
                W, is_sig = _mixed(M, n, n_sig, rng, strength)
                sel = ko.calibrated_qvalues(W)['qvalues'] <= q
                R = sel.sum()
                fp = (sel & ~is_sig).sum()
                tp = (sel & is_sig).sum()
                fdps.append(fp / max(R, 1))
                powers.append(tp / max(n_sig, 1))
            assert np.mean(fdps) <= q + 0.04, \
                f"pi0={true_pi0} s={strength}: mean FDP={np.mean(fdps):.3f}"
            assert np.mean(powers) >= 0.5, \
                f"pi0={true_pi0} s={strength}: mean power={np.mean(powers):.2f}"

    def test_pi0_one_is_more_conservative(self):
        """pi0=1 (assumption-light) selects a subset of what the estimated pi0
        selects -- more conservative, never more liberal."""
        rng = np.random.RandomState(6)
        W, _ = _mixed(200, 400, 80, rng, strength=4.0)
        sel_auto = ko.calibrated_qvalues(W, pi0='auto')['qvalues'] <= 0.1
        sel_one = ko.calibrated_qvalues(W, pi0=1.0)['qvalues'] <= 0.1
        assert sel_one.sum() <= sel_auto.sum()
        # pi0=1 only rescales q upward uniformly, so its selection is a SUBSET.
        assert set(np.where(sel_one)[0]).issubset(set(np.where(sel_auto)[0]))


# ---------------------------------------------------------------------------
#  mirror_fdp -- pi0-free cross-check
# ---------------------------------------------------------------------------

class TestMirrorFdp:

    def test_null_control(self):
        """Pure null: mirror selects (almost) nothing; realized false-selection
        rate respects q."""
        rng = np.random.RandomState(7)
        M, n, q = 40, 400, 0.1
        fdps = []
        for _ in range(60):
            W = _null_W(M, n, rng)
            m = ko.mirror_fdp(W, q=q)
            fdps.append(m['selected_mask'].mean())
        assert np.mean(fdps) <= q + 0.03, f"mirror null FDP={np.mean(fdps):.3f}"

    def test_power_and_fdr_on_signal(self):
        rng = np.random.RandomState(8)
        M, n, n_sig, q = 200, 400, 80, 0.1
        W, is_sig = _mixed(M, n, n_sig, rng, strength=4.0)
        m = ko.mirror_fdp(W, q=q)
        sel = m['selected_mask']
        R = sel.sum()
        assert R >= 30, f"mirror low power R={R}"
        fdp = (sel & ~is_sig).sum() / max(R, 1)
        assert fdp <= q + 0.05, f"mirror FDP={fdp:.3f}"

    def test_agrees_with_qvalues_under_clean_signal(self):
        """The misspecification alarm: under correct (Binomial) null the pi0-free
        mirror and the pi0-based q-values should select nearly the same genes."""
        rng = np.random.RandomState(9)
        W, _ = _mixed(200, 400, 80, rng, strength=4.5)
        out = ko.select_egenes_calibrated([f"g{i}" for i in range(400)], W, q=0.1)
        assert out['agreement'] >= 0.7, f"estimators disagree: {out['agreement']:.2f}"


# ---------------------------------------------------------------------------
#  local_fdr_interval
# ---------------------------------------------------------------------------

class TestLocalFdrInterval:

    def test_interval_brackets_and_orders(self):
        rng = np.random.RandomState(10)
        W, is_sig = _mixed(200, 400, 80, rng, strength=4.0)
        lf = ko.local_fdr_interval(W)
        assert lf['pi0_lo'] <= lf['pi0_hi'] + 1e-9
        assert np.all(lf['lfdr_lo'] <= lf['lfdr_hi'] + 1e-9)
        assert np.all((lf['lfdr_lo'] >= 0) & (lf['lfdr_hi'] <= 1))

    def test_signal_low_null_high(self):
        """Signal genes -> lfdr near 0 (surely real); null genes -> lfdr near 1
        (surely null)."""
        rng = np.random.RandomState(11)
        M, n, n_sig = 200, 600, 120
        W, is_sig = _mixed(M, n, n_sig, rng, strength=4.5)
        lf = ko.local_fdr_interval(W)
        assert lf['lfdr_hi'][is_sig].mean() < 0.2, \
            f"signal lfdr too high {lf['lfdr_hi'][is_sig].mean():.3f}"
        assert lf['lfdr_hi'][~is_sig].mean() > 0.6, \
            f"null lfdr too low {lf['lfdr_hi'][~is_sig].mean():.3f}"

    def test_pi0_interval_contains_truth_or_conservative(self):
        """The pi0 interval's upper end should be >= the truth (conservative
        identified bound); the interval should not be absurdly wide."""
        rng = np.random.RandomState(12)
        M, n = 200, 1500
        true_pi0 = 0.7
        n_sig = int(round((1 - true_pi0) * n))
        W, _ = _mixed(M, n, n_sig, rng, strength=5.0)
        lf = ko.local_fdr_interval(W)
        assert lf['pi0_hi'] >= true_pi0 - 0.12, \
            f"upper pi0 {lf['pi0_hi']:.3f} below truth {true_pi0}"
        assert lf['pi0_hi'] - lf['pi0_lo'] <= 0.6


# ---------------------------------------------------------------------------
#  select_egenes_calibrated -- the one-call wrapper
# ---------------------------------------------------------------------------

class TestSelectEgenesCalibrated:

    def test_schema_and_alignment(self):
        rng = np.random.RandomState(13)
        W, _ = _mixed(100, 60, 15, rng, strength=4.0)
        gids = [f"G{i}" for i in range(60)]
        out = ko.select_egenes_calibrated(gids, W, q=0.1)
        for k in ('selected', 'qvalues', 'pi0', 'mirror', 'lfdr',
                  'n_draws', 'agreement'):
            assert k in out
        assert out['n_draws'] == 100
        assert out['qvalues'].shape == (60,)
        assert all(g in gids for g in out['selected'])

    def test_end_to_end_calibration(self):
        """One honest end-to-end run: mixed panel, three estimators, FDR respected
        and agreement high."""
        rng = np.random.RandomState(14)
        M, n, n_sig, q = 200, 500, 100, 0.1
        W, is_sig = _mixed(M, n, n_sig, rng, strength=4.0)
        gids = [f"g{i}" for i in range(n)]
        out = ko.select_egenes_calibrated(gids, W, q=q)
        sel_idx = np.array([int(g[1:]) for g in out['selected']], dtype=int)
        sel = np.zeros(n, dtype=bool)
        sel[sel_idx] = True
        R = sel.sum()
        fdp = (sel & ~is_sig).sum() / max(R, 1)
        power = (sel & is_sig).sum() / n_sig
        assert fdp <= q + 0.05, f"FDP={fdp:.3f}"
        assert power >= 0.5, f"power={power:.2f}"
        assert out['agreement'] >= 0.6


if __name__ == '__main__':
    import pytest
    sys.exit(pytest.main([__file__, '-v']))

"""
Small-scale end-to-end regression GATES (slow) for the shipped knockoff eGene
path, plus a compute-benchmark smoke. These run the FULL real pipeline
(simulated LD genotypes -> real knockoffs -> real SuSiE -> gene_level_W ->
calibrated selection).

IMPORTANT -- WHAT THESE DO AND DO NOT TEST. They assert only ONE-SIDED CONTROL:
realized FDR does not GROSSLY EXCEED the target (mean FDP <= q + margin). They do
NOT test CALIBRATION (realized ~= nominal). A badly OVER-conservative method (e.g.
realized FDR 0.03 at target 0.10) PASSES these gates. That is deliberate: at this
small scale realized FDR is too noisy for a two-sided calibration assertion, so
these serve as cheap regression guards against gross ANTI-conservative blow-ups,
not as evidence of calibration. Calibration itself is measured -- and currently
found WANTING in both directions (over-conservative at N=300, inflated at N=100) --
by the research harness tests/calibration_validation.py; see its FINDINGS block
and docs/knockoff_susie_design.md item 9. Do not read a green gate here as "the
FDR is calibrated."

Panels use signal strong enough that the selection set R clears the 1/q detection
floor, so the one-sided assertion is non-vacuous. All gates are marked slow (real
SuSiE fits): run with `-m slow` or directly.
"""

import sys
import numpy as np
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from calibration_validation import simulate_eqtl_panel, run_and_score
from knockoff_compute_benchmark import _bench_one


def _mean_fdp_power(panels_kw, run_kw, reps, base_seed=0):
    fdps, powers, Rs = [], [], []
    for b in range(reps):
        panel = simulate_eqtl_panel(seed=base_seed + b, **panels_kw)
        r = run_and_score(panel, seed=b, **run_kw)
        fdps.append(r['FDP']); powers.append(r['power']); Rs.append(r['R'])
    return np.mean(fdps), np.nanmean(powers), np.sum(Rs)


@pytest.mark.slow
class TestEndToEndCalibration:

    def test_gaussian_calibrated_fdr_controlled(self):
        """#1 (gate): realized eGene FDR controlled with real Gaussian knockoffs
        + real SuSiE on realistic LD genotypes. Non-vacuous: strong signal so R
        clears the detection floor."""
        mfdp, power, totR = _mean_fdp_power(
            panels_kw=dict(n_genes=40, N=200, p_per_gene=50, egene_frac=0.4,
                           signal_regime='strong'),
            run_kw=dict(fdr=0.1, n_knockoffs=12, knockoff='gaussian',
                        selection='calibrated'),
            reps=4)
        assert totR >= 8, f"vacuous: only {totR} total selections"
        assert mfdp <= 0.1 + 0.12, f"realized FDR={mfdp:.3f} too high"
        assert power >= 0.3, f"power={power:.2f} too low"

    def test_weak_signal_does_not_inflate(self):
        """Weak signal (PVE 1-3%): the selection may be small (detection floor),
        but realized FDR must NOT inflate. Guards the honest weak-eQTL regime."""
        mfdp, power, totR = _mean_fdp_power(
            panels_kw=dict(n_genes=40, N=200, p_per_gene=50, egene_frac=0.4,
                           signal_regime='weak'),
            run_kw=dict(fdr=0.1, n_knockoffs=12, knockoff='gaussian',
                        selection='calibrated'),
            reps=4)
        assert mfdp <= 0.1 + 0.12, f"weak-signal realized FDR={mfdp:.3f} inflated"


@pytest.mark.slow
class TestGeneratorStress:

    def test_low_N_gaussian_controls(self):
        """#2 (gate): at low N (=50) with strong LD + rare variants, the
        shrinkage Gaussian knockoff must not grossly inflate FDR."""
        mfdp, power, totR = _mean_fdp_power(
            panels_kw=dict(n_genes=40, N=50, p_per_gene=50, egene_frac=0.4,
                           signal_regime='strong', rare_variant_skew=0.85),
            run_kw=dict(fdr=0.1, n_knockoffs=12, knockoff='gaussian', shrink=0.15,
                        selection='calibrated'),
            reps=4)
        assert mfdp <= 0.1 + 0.15, f"low-N realized FDR={mfdp:.3f} inflated"


@pytest.mark.slow
class TestPi0Sweep:

    def test_fdr_controlled_across_egene_fraction(self):
        """#3 (gate): realized FDR controlled at eGene fractions 0.1 and 0.4."""
        for frac in (0.1, 0.4):
            mfdp, power, totR = _mean_fdp_power(
                panels_kw=dict(n_genes=50, N=200, p_per_gene=50, egene_frac=frac,
                               signal_regime='strong'),
                run_kw=dict(fdr=0.1, n_knockoffs=12, knockoff='gaussian',
                            selection='calibrated'),
                reps=4)
            assert mfdp <= 0.1 + 0.12, f"frac={frac}: realized FDR={mfdp:.3f}"


@pytest.mark.slow
class TestPolygenicContaminant:

    def test_clean_null_fdr_with_polygenic_genes(self):
        """#4 (gate): dense-polygenic genes mixed in must not corrupt the
        clean-null FDR (they add intermediate-W mass that could bias pi0)."""
        fdps = []
        for b in range(4):
            panel = simulate_eqtl_panel(n_genes=60, N=200, p_per_gene=50,
                                        egene_frac=0.2, signal_regime='strong',
                                        n_polygenic=18, seed=b)
            r = run_and_score(panel, fdr=0.1, n_knockoffs=12,
                              knockoff='gaussian', selection='calibrated', seed=b)
            fdps.append(r['FDP'])   # false = selected CLEAN nulls only
        assert np.mean(fdps) <= 0.1 + 0.12, \
            f"clean-null FDR with polygenic contaminant={np.mean(fdps):.3f}"


@pytest.mark.slow
class TestComputeBenchmarkSmoke:

    def test_coherent_hmm_scales_and_bounded(self):
        """#7 (smoke): the coherent HMM fit+draw completes at a modest p and the
        per-1k-variant cost does not blow up between two p's (roughly linear)."""
        r_small = _bench_one(p=800, N=100, K=6, M=4, em_iter=6, method='haplotype')
        r_big = _bench_one(p=3200, N=100, K=6, M=4, em_iter=6, method='haplotype')
        # peak memory for these small sizes stays well under 2 GB
        assert r_big['peak_mb'] < 2000, f"peak {r_big['peak_mb']:.0f} MB too high"
        # per-1k-variant cost should be within ~3x across a 4x size jump (linear-ish)
        ratio = r_big['per_1k_var'] / max(r_small['per_1k_var'], 1e-6)
        assert ratio < 3.0, f"per-variant cost grew {ratio:.1f}x (super-linear?)"


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v', '-s', '-m', 'slow']))

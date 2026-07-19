"""
Tests for the fastPHASE-style HMM genotype simulator.

Confirms the simulator produces the features that make it a meaningful test bed
for knockoff validation on realistic (non-Gaussian) genotypes: discrete dosages,
a rare-variant tail, LD that decays with distance, recombination hotspots that
break LD, and phased haplotypes consistent with the dosage.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.hmm_genotype_simulator import (
    simulate_hmm_genotypes, make_recombination_map, ld_decay)


class TestHMMSimulator:

    def test_dosage_range_and_phasing(self):
        g, pos, info = simulate_hmm_genotypes(200, 250, seed=1, return_phased=True)
        assert g.shape == (200, 250)
        assert g.min() >= 0 and g.max() <= 2
        # phased haplotype alleles are 0/1 and sum to the dosage
        assert set(np.unique(info['xL'])).issubset({0, 1})
        assert np.array_equal(info['xL'] + info['xR'], g.astype(np.int8))

    def test_rare_variant_tail(self):
        _, _, info = simulate_hmm_genotypes(400, 300, seed=2)
        maf = info['maf']
        # a meaningful fraction of low-frequency variants (the regime Gaussian
        # knockoffs handle worst) -- the whole point of using this simulator
        assert (maf < 0.10).mean() > 0.1

    def test_ld_decays_with_distance(self):
        g, pos, _ = simulate_hmm_genotypes(300, 400, seed=4, base_rate=0.02,
                                           n_hotspots=0)
        c, r2 = ld_decay(g, pos, max_dist=150_000)
        assert len(r2) >= 4
        # short-range LD exceeds long-range LD (decay)
        assert r2[0] > r2[-1]
        assert r2[0] > 0.02  # non-trivial LD exists

    def test_hotspot_breaks_ld(self):
        rng = np.random.default_rng(3)
        rho = make_recombination_map(200, rng, base_rate=0.01, n_hotspots=1,
                                     hotspot_strength=30, hotspot_width=2)
        hot = int(np.argmax(rho))
        # keep the hotspot away from the ends so both flanks exist
        if hot < 10 or hot > 190:
            rho[:] = 0.01
            rho[100] = 0.30
            hot = 100
        g, pos, _ = simulate_hmm_genotypes(200, 400, seed=3, rho=rho)
        gs = (g - g.mean(1, keepdims=True)) / (g.std(1, keepdims=True) + 1e-9)

        def r2(i, j):
            return float(np.corrcoef(gs[i], gs[j])[0, 1] ** 2)
        across = r2(hot - 2, hot + 2)
        within = np.mean([r2(k, k + 4) for k in range(20, 40)])
        assert across <= within + 0.02  # LD is not stronger across the hotspot

    def test_recombination_map_hotspots(self):
        rng = np.random.default_rng(0)
        rho = make_recombination_map(100, rng, base_rate=0.05, n_hotspots=3,
                                     hotspot_strength=8.0)
        assert rho[0] == 0.0
        assert rho.max() > rho[rho > 0].min() * 5  # hotspots are much larger


if __name__ == '__main__':
    import pytest
    sys.exit(pytest.main([__file__, '-v']))

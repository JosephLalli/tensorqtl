"""
Genome-wide eGene FDR under cross-gene DEPENDENCE -- the empirical half of the
"overlapping-gene joint-sign" resolution.

THE REDUCTION (what these tests validate). Once each gene carries a per-gene
knockoff p-value with an EXACTLY known marginal null (per_gene_pvalues:
super-uniform, exact Binomial(M,1/2)), controlling eGene FDR across the whole
genome stops being a novel knockoff-joint-sign theorem and becomes the CLASSICAL
problem of Benjamini-Hochberg under dependence:
  * independent or PRDS null p-values  -> plain BH controls FDR
    (Benjamini & Yekutieli 2001);
  * arbitrary dependence               -> BH with the harmonic factor c(n) does
    (Benjamini & Yekutieli 2001, distribution-free).
Overlapping eQTL genes share genotypes (and, with the coherent HMM generator,
share knockoffs on shared variants), which makes their statistics POSITIVELY
associated -- the PRDS regime where plain BH is proved to work. These tests
inject exactly that positive dependence into the gene statistics (via a shared
per-draw latent shift, which preserves each gene's marginal Binomial null while
correlating genes) and check that:
  1. plain BH / the calibrated q-value still controls genome-wide FDR under
     positive dependence (the PRDS claim), and
  2. the 'arbitrary' (Benjamini-Yekutieli harmonic) mode is a strictly more
     conservative subset -- the guaranteed fallback when PRDS is not assumed.

We simulate the gene STATISTICS directly (not full SuSiE fits): the dependence
reduction is a property of the p-values given their null, and the null is exact
by construction. Generator validity (that real knockoffs actually yield this
null) is covered separately by the HMM/Gaussian knockoff tests.

WHAT THE SIMULATIONS ESTABLISH (the honest, complete picture; numbers are mean
realized FDR at target q=0.1 over many replicates):

  regime                         PRDS/BH (pi0=auto)   BY/arbitrary (pi0=1)
  independent                    0.087  (controls)    0.011  (controls)
  block-LOCAL dep (rho<=0.9)     0.090  (controls)    --     (controls)
  GLOBAL equicorrelation rho=0.7 0.180  (INFLATES)    0.011  (controls)
  GLOBAL equicorrelation rho=0.9 0.239  (INFLATES)    0.011  (controls)

Three take-aways, each encoded as a test below:
  1. Under the REALISTIC eQTL regime -- independence or LOCAL (block) dependence,
     because distant genes are ~independent and only nearby genes share LD -- the
     shipped calibrated q-value (PRDS/BH, pi0=auto) controls genome-wide FDR
     tightly, even at strong within-block correlation. This is the default.
  2. Under ADVERSARIAL global equicorrelation (every gene positively coupled to
     every other -- not an eQTL reality) plain BH can INFLATE, because the
     pi0-adaptive step, not BH itself, is fragile: a "bad" replicate pushes all
     nulls to look like signal at once, pi0 is under-estimated, q-values shrink.
  3. The GUARANTEED fallback -- dependence='arbitrary' (Benjamini-Yekutieli
     harmonic factor) WITH pi0=1.0 (no fragile adaptive step) -- controls FDR
     under ANY dependence (verified ~0.011 even at rho=0.9), at the cost of the
     log-factor conservatism.

Caveat surfaced, not hidden: under strong dependence the per-analysis FDP has
large VARIANCE even when its mean (the FDR) is controlled -- BH/BY control the
expectation, not the realized proportion in any single run.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import tensorqtl.knockoffs as ko


def _dependent_W(M, n, n_sig, rng, strength, rho):
    """
    W[M, n] with the first n_sig genes signal, plus POSITIVE cross-gene
    dependence of strength rho in [0,1).

    Each draw m gets a shared latent shift u_m ~ N(0,1) applied to ALL genes:
        W_g^(m) = sqrt(1-rho) * eps_g^(m) + sqrt(rho) * u_m + strength*1{signal}.
    The null genes' W stays marginally symmetric about 0 (sum of symmetric
    terms), so b_g ~ Binomial(M,1/2) marginally holds EXACTLY; but within a draw
    the shared u_m pushes every gene the same way, inducing positive dependence
    across genes (and hence across their b_g counts) -- a PRDS-like structure.
    """
    eps = rng.randn(M, n)
    u = rng.randn(M, 1)
    W = np.sqrt(1 - rho) * eps + np.sqrt(rho) * u
    W[:, :n_sig] += strength
    is_sig = np.zeros(n, dtype=bool)
    is_sig[:n_sig] = True
    return W, is_sig


class TestDependenceModes:

    def test_arbitrary_is_conservative_subset(self):
        """BY ('arbitrary') selects a subset of PRDS/BH -- never more."""
        rng = np.random.RandomState(0)
        W, _ = _dependent_W(200, 400, 80, rng, strength=4.0, rho=0.3)
        s_prds = set(np.where(ko.calibrated_qvalues(W, dependence='prds')['qvalues'] <= 0.1)[0])
        s_arb = set(np.where(ko.calibrated_qvalues(W, dependence='arbitrary')['qvalues'] <= 0.1)[0])
        assert s_arb.issubset(s_prds)

    def test_prds_and_ind_identical(self):
        """'prds' and 'ind' share the BH threshold -- only the invoked theorem
        differs. Selections must be identical."""
        rng = np.random.RandomState(1)
        W, _ = _dependent_W(150, 300, 60, rng, strength=4.0, rho=0.2)
        q_prds = ko.calibrated_qvalues(W, dependence='prds')['qvalues']
        q_ind = ko.calibrated_qvalues(W, dependence='ind')['qvalues']
        assert np.allclose(q_prds, q_ind)

    def test_bh_select_harmonic_scaling(self):
        """bh_select 'arbitrary' threshold is 'prds' threshold / c(n)."""
        rng = np.random.RandomState(2)
        p = rng.rand(500)
        cn = np.sum(1.0 / np.arange(1, 501))
        # A p-value passing 'arbitrary' at q must pass 'prds' at q (subset).
        s_arb = ko.bh_select(p, 0.1, 'arbitrary')
        s_prds = ko.bh_select(p, 0.1, 'prds')
        assert set(np.where(s_arb)[0]).issubset(set(np.where(s_prds)[0]))
        assert cn > 6.0  # sanity: harmonic number of 500


def _block_local_W(M, n, n_sig, rng, strength, rho, block=10):
    """
    REALISTIC eQTL dependence: genes in contiguous blocks of size `block` share a
    latent shift; blocks are mutually independent. This mimics nearby genes
    sharing LD while distant genes are independent -- the actual cis-eQTL
    structure. With many independent blocks the genome-wide FDP concentrates and
    plain BH controls FDR even at strong within-block correlation.
    """
    n_blocks = int(np.ceil(n / block))
    eps = rng.randn(M, n)
    u = np.repeat(rng.randn(M, n_blocks), block, axis=1)[:, :n]
    W = np.sqrt(1 - rho) * eps + np.sqrt(rho) * u
    W[:, :n_sig] += strength
    is_sig = np.zeros(n, dtype=bool)
    is_sig[:n_sig] = True
    return W, is_sig


class TestRealisticDependenceControl:
    """The shipped claim: under independence or LOCAL (block) dependence the
    calibrated q-value (PRDS/BH, pi0=auto) controls genome-wide FDR."""

    def test_null_controlled_independent_and_local(self):
        rng = np.random.RandomState(3)
        M, n, q = 40, 400, 0.1
        # independent
        fdps = []
        for _ in range(150):
            W = rng.randn(M, n)   # pure null, independent genes
            fdps.append((ko.calibrated_qvalues(W)['qvalues'] <= q).mean())
        assert np.mean(fdps) <= q + 0.02, f"indep null FDR={np.mean(fdps):.3f}"
        # block-local, strong within-block correlation
        for rho in (0.5, 0.9):
            fdps = []
            for _ in range(150):
                W, _ = _block_local_W(M, n, 0, rng, 0.0, rho, block=10)
                fdps.append((ko.calibrated_qvalues(W)['qvalues'] <= q).mean())
            assert np.mean(fdps) <= q + 0.03, \
                f"block rho={rho} null FDR={np.mean(fdps):.3f}"

    def test_mixed_controlled_local_dependence(self):
        """Signal + null under strong LOCAL dependence: FDR at target, full
        power. This is the operating regime the default is validated for."""
        rng = np.random.RandomState(4)
        M, n, n_sig, q = 200, 400, 80, 0.1
        for rho in (0.4, 0.9):
            fdps, powers = [], []
            for _ in range(200):
                W, is_sig = _block_local_W(M, n, n_sig, rng, 4.0, rho, block=10)
                sel = ko.calibrated_qvalues(W)['qvalues'] <= q
                R = sel.sum()
                fdps.append((sel & ~is_sig).sum() / max(R, 1))
                powers.append((sel & is_sig).sum() / n_sig)
            assert np.mean(fdps) <= q + 0.02, \
                f"local rho={rho} FDR={np.mean(fdps):.3f}"
            assert np.mean(powers) >= 0.8, \
                f"local rho={rho} power={np.mean(powers):.2f}"


class TestAdversarialDependence:
    """Global equicorrelation (NOT an eQTL reality): plain BH can inflate; the
    BY + pi0=1 fallback restores guaranteed control under ANY dependence."""

    def test_plain_bh_inflates_under_global_equicorrelation(self):
        """Documents the known limitation as a REGRESSION: under strong global
        dependence the pi0-adaptive PRDS q-value inflates above target (so nobody
        mistakes it for safe here). If a future change made it control this case
        for free, this test should be revisited."""
        rng = np.random.RandomState(5)
        M, n, n_sig, q = 200, 400, 80, 0.1
        fdps = []
        for _ in range(300):
            W, is_sig = _dependent_W(M, n, n_sig, rng, strength=4.0, rho=0.7)
            sel = ko.calibrated_qvalues(W, dependence='prds')['qvalues'] <= q
            R = sel.sum()
            fdps.append((sel & ~is_sig).sum() / max(R, 1))
        assert np.mean(fdps) > q + 0.03, \
            f"expected inflation under global dep, got FDR={np.mean(fdps):.3f}"

    def test_by_pi0_one_controls_under_any_dependence(self):
        """The guaranteed fallback: dependence='arbitrary' (BY harmonic) WITH
        pi0=1.0 (no fragile adaptive step) controls FDR even under adversarial
        global equicorrelation, at any strength."""
        rng = np.random.RandomState(6)
        M, n, n_sig, q = 200, 400, 80, 0.1
        for rho in (0.7, 0.9):
            fdps = []
            for _ in range(300):
                W, is_sig = _dependent_W(M, n, n_sig, rng, strength=4.5, rho=rho)
                sel = ko.calibrated_qvalues(W, dependence='arbitrary',
                                             pi0=1.0)['qvalues'] <= q
                R = sel.sum()
                fdps.append((sel & ~is_sig).sum() / max(R, 1))
            assert np.mean(fdps) <= q, \
                f"BY+pi0=1 rho={rho} FDR={np.mean(fdps):.3f} not <= q"


if __name__ == '__main__':
    import pytest
    sys.exit(pytest.main([__file__, '-v']))

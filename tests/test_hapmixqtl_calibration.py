"""
CI calibration gate for hapmixQTL (docs/ase_validation.md sec 8).

The unit tests in test_hapmixqtl.py check that the code computes the formula
it was written to compute. None of them can see an invalid error rate: the
tau_mode='zero' defect (docs sec 2, 6, 7d) passed every one of them while
making every null test on real GTEx data significant. This module is the cheap
version of Tiers 0 / 0b / 1 of tests/ase_validation.py that would have caught
it, run on every push:

  * control cell (no unmodelled variance): nominal for BOTH tau modes; if
    this fails the harness itself is broken, not the method;
  * inflated cell (biological SD 0.6): 'estimate' stays calibrated AND 'zero'
    is detectably inflated -- the second half keeps the gate honest, so a
    simulator change that hides the defect fails too;
  * 95% CI coverage of a planted log aFC: 'estimate' covers, 'zero' does not;
  * the two channel estimators stay uncorrelated when the a and t noise is
    correlated at 0.9 -- the measured basis for ignoring Cat (docs sec 3).

Deterministic seeds; about 10 s on CPU.
"""
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from ase_validation import simulate_channels, run_association, pvals_from_t  # noqa: E402

N, V = 200, 20


def _null_pvals(reps, sigma_bio, rho, tau_mode, seed):
    rng = np.random.RandomState(seed)
    out = []
    for _ in range(reps):
        g, s, a, t, va, vt = simulate_channels(N, V, rng, beta=0.0,
                                               sigma_bio=sigma_bio, rho_at=rho)
        tstat = run_association(g, s, a, t, va, vt, tau_mode=tau_mode)[0]
        out.append(pvals_from_t(tstat, N))
    return np.concatenate(out)


def _type1(p, alpha):
    return float(np.mean(p < alpha))


def _lambda_gc(p):
    return float(np.median(stats.chi2.isf(p, 1)) / stats.chi2.ppf(0.5, 1))


def test_control_cell_is_nominal_for_both_tau_modes():
    """With no unmodelled variance the known-variance model is exactly true,
    so both settings must be nominal. Failure here means the harness, not
    the method, is broken."""
    for tau_mode in ('zero', 'estimate'):
        p = _null_pvals(150, 0.0, 0.0, tau_mode, seed=11)
        t1 = _type1(p, 0.05)
        assert 0.03 <= t1 <= 0.07, \
            f'{tau_mode}: type-I at 0.05 = {t1:.4f} on {p.size} null tests'
        lam = _lambda_gc(p)
        assert 0.8 <= lam <= 1.2, f'{tau_mode}: lambda_GC = {lam:.3f}'


def test_uninformative_samples_do_not_collapse_tau():
    """A sample with no allele-specific reads has v_inf = 0 exactly and a = 0.

    On BrainVar the mapping functions clamped v to 1e-8 before estimating
    tau, so such samples entered the moment estimator at weight 1e8, drove
    tau_a to ~1e-6 for the gene, and left the informative samples weighted by
    v_inf alone: the known-variance SE was too small by the unmodelled
    variance (2-25x) and the allelic channel's permutation null reached chi2
    40-120 where ~12-16 is calibrated. With 10% such samples and unmodelled
    variance, 'estimate' must stay calibrated; and feeding the clamped
    variances that hid them must be detectably miscalibrated, so the gate
    stays honest. Which way it breaks depends on the design: with BrainVar's
    17 covariates the hidden samples are fit exactly and the rest inflate;
    with an intercept alone their 1e8 weights dominate xx and every
    statistic collapses toward zero.
    """
    def pvals(clamped):
        rng = np.random.RandomState(7)
        out = []
        for _ in range(30):
            g, s, a, t, va, vt = simulate_channels(N, V, rng, beta=0.0, sigma_bio=0.6)
            k = rng.choice(N, N // 10, replace=False)
            a[k] = 0.0
            va[k] = 1e-8 if clamped else 0.0
            tstat = run_association(g, s, a, t, va, vt, tau_mode='estimate')[0]
            out.append(pvals_from_t(tstat, N))
        return np.concatenate(out)

    p = pvals(clamped=False)
    assert _type1(p, 0.05) < 0.08, _type1(p, 0.05)
    assert _lambda_gc(p) < 1.2, _lambda_gc(p)
    lam_hidden = _lambda_gc(pvals(clamped=True))
    assert lam_hidden < 0.8 or lam_hidden > 1.25, lam_hidden


def test_estimate_stays_calibrated_with_unmodelled_variance():
    """The cell that exposed the defect: biological SD 0.6 on both channels."""
    p = _null_pvals(150, 0.6, 0.0, 'estimate', seed=22)
    assert _type1(p, 0.05) <= 0.07, f'type-I at 0.05 = {_type1(p, 0.05):.4f}'
    assert _type1(p, 0.01) <= 0.02, f'type-I at 0.01 = {_type1(p, 0.01):.4f}'
    assert _lambda_gc(p) <= 1.2, f'lambda_GC = {_lambda_gc(p):.3f}'


def test_gate_detects_the_tau_zero_defect():
    """Sensitivity check: the same cell under tau_mode='zero' must be visibly
    inflated (6.8x at alpha 0.05 in the full harness). If this ever passes
    calibration, the simulator has stopped exercising the defect and the
    test above is no longer evidence of anything."""
    p = _null_pvals(150, 0.6, 0.0, 'zero', seed=22)
    assert _type1(p, 0.05) >= 0.15, \
        f'type-I at 0.05 = {_type1(p, 0.05):.4f}; the gate has lost its sensitivity'
    assert _lambda_gc(p) >= 2.0, f'lambda_GC = {_lambda_gc(p):.3f}'


def test_ci_coverage_with_unmodelled_variance():
    """Nominal 95% CI for a planted log aFC = 0.5 (Tier 1): 'estimate' must
    cover, 'zero' must not (0.959 vs 0.645 in the full harness)."""
    cov = {}
    for tau_mode in ('estimate', 'zero'):
        rng = np.random.RandomState(33)
        hit = []
        for _ in range(200):
            g, s, a, t, va, vt = simulate_channels(N, V, rng, beta=0.5, sigma_bio=0.6)
            _, slope, se, *_ = run_association(g, s, a, t, va, vt, tau_mode=tau_mode)
            hit.append(abs(slope[0] - 0.5) <= 1.96 * se[0])
        cov[tau_mode] = float(np.mean(hit))
    assert cov['estimate'] >= 0.90, f"coverage under 'estimate' = {cov['estimate']:.3f}"
    assert cov['zero'] <= 0.80, \
        f"coverage under 'zero' = {cov['zero']:.3f}; the gate has lost its sensitivity"


def test_channel_estimators_uncorrelated_under_correlated_noise():
    """The basis for ignoring Cat: with the a and t inferential noise
    correlated at 0.9, the ASE and total slope estimators are still
    uncorrelated, because s = xL - xR is orthogonal to g/2 under random
    phase (docs sec 3)."""
    rng = np.random.RandomState(44)
    ba, bt = [], []
    for _ in range(400):
        g, s, a, t, va, vt = simulate_channels(N, V, rng, beta=0.0, sigma_bio=0.3, rho_at=0.9)
        _, _, _, slope_t, _, slope_a, _ = run_association(g, s, a, t, va, vt, tau_mode='estimate')
        ba.append(slope_a[0]); bt.append(slope_t[0])
    r = float(np.corrcoef(ba, bt)[0, 1])
    assert abs(r) <= 0.15, f'corr(beta_a, beta_t) = {r:.3f} at rho = 0.9'


def test_per_channel_covariates_stay_calibrated():
    """BrainVar's layout: covariates that move total expression are projected
    out of the total channel, the allelic channel gets an intercept only, the
    variances are heteroskedastic with unmodelled biological variance, and 10%
    of samples carry no allele-specific coverage. The null must stay nominal
    on the shared t reference (N - 2 - n_cov), and an allelic channel fitted
    without the covariates must not be less calibrated than one fitted with
    them."""
    n_cov = 8

    def pvals(ase, seed):
        rng = np.random.RandomState(seed)
        out = []
        for _ in range(30):
            g, s, a, t, va, vt = simulate_channels(N, V, rng, beta=0.0, sigma_bio=0.6)
            C = rng.normal(size=(N, n_cov))
            t = t + C @ rng.normal(0, 0.3, n_cov)
            k = rng.choice(N, N // 10, replace=False)
            a[k] = 0.0
            va[k] = 0.0
            tstat = run_association(g, s, a, t, va, vt, tau_mode='estimate',
                                    covariates=C, ase_covariates=ase)[0]
            out.append(pvals_from_t(tstat, N, n_cov=n_cov))
        return np.concatenate(out)

    p = pvals(None, seed=41)
    assert _type1(p, 0.05) < 0.08, _type1(p, 0.05)
    assert 0.8 < _lambda_gc(p) < 1.2, _lambda_gc(p)
    p_shared = pvals('same', seed=41)
    assert _type1(p_shared, 0.05) < 0.08, _type1(p_shared, 0.05)


def _map_cis_null_pvals(reps, nperm=200, seed=3, n_samples=80, n_variants=20):
    """map_cis on null genes with heteroskedastic v_inf (0.01-2), covariates
    on the total channel, an intercept-only allelic channel and 10% samples
    without allele-specific coverage; returns (pval_perm, pval_beta)."""
    import io, contextlib
    import pandas as pd
    from tensorqtl.hapmixqtl import map_cis
    N, V = n_samples, n_variants
    rng = np.random.RandomState(seed)
    pp, pb = [], []
    samples = [f'S{i}' for i in range(N)]
    vids = [f'chr1_{1000 + i * 100}_A_G' for i in range(V)]
    vdf = pd.DataFrame({'chrom': ['chr1'] * V, 'pos': [1000 + i * 100 for i in range(V)]}, index=vids)
    pos = pd.DataFrame({'chr': ['chr1'], 'pos': [1000]}, index=['G1'])
    for rep in range(reps):
        g, s, a, t, va, vt = simulate_channels(N, V, rng, beta=0.0, v_inf_lo=0.01,
                                               v_inf_hi=2.0, sigma_bio=0.3)
        C = rng.normal(size=(N, 4))
        t = t + C @ rng.normal(0, 0.3, 4)
        k = rng.choice(N, N // 10, replace=False)
        a[k] = 0.0
        va[k] = 0.0
        xL = ((g == 2) | ((g == 1) & (s > 0))).astype(float)
        xR = ((g == 2) | ((g == 1) & (s < 0))).astype(float)
        mk = lambda x: pd.DataFrame(x[None, :], index=['G1'], columns=samples)
        with contextlib.redirect_stdout(io.StringIO()):
            res = map_cis(pd.DataFrame(g, index=vids, columns=samples), vdf,
                          mk(a), mk(t), mk(va), mk(vt), pos,
                          xL_df=pd.DataFrame(xL, index=vids, columns=samples),
                          xR_df=pd.DataFrame(xR, index=vids, columns=samples),
                          nperm=nperm, covariates_df=pd.DataFrame(C, index=samples),
                          ase_covariates_df=None, seed=rep, verbose=False)
        pp.append(float(res['pval_perm'].iloc[0]))
        pb.append(float(res['pval_beta'].iloc[0]))
    return np.array(pp), np.array(pb)


def test_pval_perm_is_calibrated_under_heteroskedasticity():
    """The empirical p-value must be uniform on null genes whose samples have
    very different inferential precision. Permuting the RAW phenotype values
    at fixed weights (the scheme before the whitened-residual permutation)
    hands each sample another sample's value at its own precision and
    mis-scales the null: on this design pval_perm averaged 0.94 with no
    rejection at 0.05 in 100 genes. A valid scheme gives mean 0.5 and the
    nominal rejection rate."""
    pp, pb = _map_cis_null_pvals(60)
    assert 0.40 <= pp.mean() <= 0.60, pp.mean()
    assert (pp < 0.05).mean() <= 0.13, (pp < 0.05).mean()
    assert 0.35 <= (pp < 0.5).mean() <= 0.65, (pp < 0.5).mean()
    ok = np.isfinite(pb)
    assert ok.mean() > 0.9
    assert 0.40 <= pb[ok].mean() <= 0.60, pb[ok].mean()

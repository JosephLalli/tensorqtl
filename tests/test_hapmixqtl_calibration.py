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
    correlated at 0.9 -- the measured basis for ignoring Cat (docs sec 3);
  * the COMBINED SE stays calibrated when donors differ two orders of
    magnitude in v_inf (BrainVar's allelic range) with that same correlated
    noise, against a homogeneous-v control.

Deterministic seeds; about 50 s on CPU.
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


def test_combined_se_calibrated_under_heterogeneous_gibbs_variance():
    """The test above shows corr(beta_a, beta_t) stays at zero under rho = 0.9
    when every donor has the same v. On BrainVar the allelic v_inf spans two
    orders of magnitude within a gene, so a few donors dominate both channels'
    weighted sums and the per-gene covariance the combination ignores is no
    longer a sum of many small random-sign terms. This checks the COMBINED
    SE in that regime: 92 donors, het fraction 0.4 with random phase, v_a
    log-uniform on [1e-3, 0.5] and v_t on [1e-4, 0.05], a-t inferential noise
    correlated at 0.9, biological tau_a = 0.03 and tau_t = 0.01, through the
    production path (_prepare_channels with tau_mode='estimate' and a
    through-origin allelic channel, then calculate_hapmixqtl_nominal); the
    control is the same design with every donor at the geometric-mean v.

    Measured on 20000 null genes per arm (MC SE 0.010 on mean t^2, 0.0015 on
    the 0.05 tail, 0.0007 on the 0.01 tail): heterogeneous v gives mean t^2
    1.023, |t| > 1.96 in 0.0512 and > 2.576 in 0.0106, corr(beta_a, beta_t)
    -0.009 (SE 0.007), median allelic share of the combined weight 0.51;
    homogeneous v gives 1.000, 0.0479, 0.0094, -0.006. The heterogeneous
    design with rho = 0 gives mean t^2 1.023 as well, so the 2% residual
    inflation is not the ignored covariance: it is the tau plug-in. The
    moment estimate of tau is noisier when 1/v spans three decades (SD 0.0084
    against 0.0077 on tau_a, 0.0027 against 0.0018 on tau_t) and the
    known-variance SE does not propagate that noise. Both arms sit far inside
    the 0.1 tolerance on mean t^2 and inside 3 binomial SD on both tails.

    Coverage of a planted beta = 0.3 is checked on both tau paths. The
    nominal scan estimates tau under the null design, so the planted effect
    inflates tau (0.066 against 0.030 on the allelic channel, 0.019 against
    0.010 on the total) and the interval over-covers, 0.981 heterogeneous and
    0.987 homogeneous: that path is only required not to under-cover. The
    lead refit puts the tested predictor into tau's design, as map_cis does
    for the reported lead, and must cover at 0.95 two-sided: 0.946
    heterogeneous and 0.949 homogeneous with the production t_{N-2}
    multiplier (0.943 and 0.946 with 1.96; SE 0.0024 on 8000 genes), with
    var(slope) / mean(se^2) = 1.05 and 1.01, i.e. the SE is understated by
    at most 1-3% under heterogeneity, and the per-channel intervals under-
    cover by the same amount, so the combination adds nothing. Type-I is
    taken from pvals_from_t (the production t_{N-2} reference; the
    1.96 / 2.576 normal fractions are reported alongside). Tolerances: 3
    binomial SD on every proportion, 0.1 on mean t^2 (5.5 SD at 6000 genes).
    About 40 s on CPU.
    """
    import torch
    from tensorqtl.hapmixqtl import _prepare_channels, calculate_hapmixqtl_nominal

    n, het_frac = 92, 0.4
    maf = 0.5 * (1.0 - np.sqrt(1.0 - 2.0 * het_frac))   # HWE: 2p(1-p) = het_frac
    rho, tau_a, tau_t, beta_alt = 0.9, 0.03, 0.01, 0.3
    va_lo, va_hi, vt_lo, vt_hi = 1e-3, 5e-1, 1e-4, 5e-2
    n_null, n_cov = 6000, 4000
    q95 = float(stats.t.ppf(0.975, n - 2))   # the multiplier pvals_from_t implies

    def simulate(rng, beta, homogeneous, rho_at):
        g = rng.binomial(2, maf, n).astype(np.float64)
        s = np.zeros(n)
        het = g == 1
        s[het] = rng.choice([-1.0, 1.0], size=int(het.sum()))
        if homogeneous:
            va = np.full(n, np.sqrt(va_lo * va_hi))
            vt = np.full(n, np.sqrt(vt_lo * vt_hi))
        else:
            va = np.exp(rng.uniform(np.log(va_lo), np.log(va_hi), n))
            vt = np.exp(rng.uniform(np.log(vt_lo), np.log(vt_hi), n))
        z = rng.multivariate_normal([0.0, 0.0], [[1.0, rho_at], [rho_at, 1.0]], size=n)
        a = np.sqrt(va) * z[:, 0] + rng.normal(0.0, np.sqrt(tau_a), n) + beta * s
        t = 2.0 + np.sqrt(vt) * z[:, 1] + rng.normal(0.0, np.sqrt(tau_t), n) + beta * g / 2.0
        return g, s, a, t, va, vt

    def fit(g, s, a, t, va, vt, refit):
        """tstat, slope, se, slope_a, se_a, slope_t, se_t of the one variant;
        refit=True adds the tested predictor to tau's design (the lead refit)."""
        g_t, s_t, a_t, t_t, va_t, vt_t = (torch.tensor(x, dtype=torch.float64)
                                          for x in (g, s, a, t, va, vt))
        wa, wt, res_a, res_t = _prepare_channels(
            a_t, t_t, va_t, vt_t, None, 'estimate', 'cpu', ase_covariates_t=None,
            tau_extra_a_t=s_t.unsqueeze(1) if refit else None,
            tau_extra_t_t=(g_t / 2.0).unsqueeze(1) if refit else None)
        out = calculate_hapmixqtl_nominal(g_t[None, :], s_t[None, :], a_t, t_t,
                                          wa, wt, res_a, res_t)
        return [float(x[0]) for x in out]

    def null_arm(homogeneous, rho_at, seed):
        rng = np.random.RandomState(seed)
        r = np.array([fit(*simulate(rng, 0.0, homogeneous, rho_at), False)
                      for _ in range(n_null)])
        tstat, p = r[:, 0], pvals_from_t(r[:, 0], n)
        share_a = r[:, 6] ** 2 / (r[:, 4] ** 2 + r[:, 6] ** 2)   # inv_var_a / total
        return dict(t2=float(np.mean(tstat ** 2)),
                    t1_05=_type1(p, 0.05), t1_01=_type1(p, 0.01),
                    z_05=float(np.mean(np.abs(tstat) > 1.96)),
                    z_01=float(np.mean(np.abs(tstat) > 2.576)),
                    corr=float(np.corrcoef(r[:, 3], r[:, 5])[0, 1]),
                    share_a=float(np.median(share_a)),
                    ratio=float(np.var(r[:, 1]) / np.mean(r[:, 2] ** 2)))

    def coverage_arm(homogeneous, seed):
        rng = np.random.RandomState(seed)
        nominal, refit = [], []
        for _ in range(n_cov):
            d = simulate(rng, beta_alt, homogeneous, rho)
            nominal.append(fit(*d, False))
            refit.append(fit(*d, True))
        out = {}
        for name, r in (('nominal', np.array(nominal)), ('refit', np.array(refit))):
            err = np.abs(r[:, 1] - beta_alt) / r[:, 2]
            out[name] = dict(cov=float(np.mean(err <= q95)),
                             cov196=float(np.mean(err <= 1.96)),
                             ratio=float(np.var(r[:, 1]) / np.mean(r[:, 2] ** 2)))
        return out

    def tol(p0, m):
        return 3.0 * np.sqrt(p0 * (1.0 - p0) / m)

    arms = {}
    for name, homogeneous in (('heterogeneous', False), ('homogeneous', True)):
        arms[name] = (null_arm(homogeneous, rho, seed=55), coverage_arm(homogeneous, seed=66))
    control = null_arm(False, 0.0, seed=55)   # heterogeneous v, independent a-t noise

    lines = []
    for name, (nul, cov) in arms.items():
        lines.append(
            f"{name} v: null mean t^2 {nul['t2']:.3f} (+-{np.sqrt(2.0 / n_null):.3f}), "
            f"type-I {nul['t1_05']:.4f} at 0.05 and {nul['t1_01']:.4f} at 0.01 "
            f"(|t| > 1.96: {nul['z_05']:.4f}, > 2.576: {nul['z_01']:.4f}), "
            f"corr(beta_a, beta_t) {nul['corr']:.3f}, median allelic share {nul['share_a']:.3f}, "
            f"var(slope)/mean(se^2) {nul['ratio']:.3f}; beta = {beta_alt} coverage "
            f"{cov['nominal']['cov']:.4f} nominal / {cov['refit']['cov']:.4f} lead-refit "
            f"(1.96 multiplier: {cov['nominal']['cov196']:.4f} / {cov['refit']['cov196']:.4f}; "
            f"var(slope)/mean(se^2) {cov['nominal']['ratio']:.3f} / {cov['refit']['ratio']:.3f})")
    lines.append(f"heterogeneous v, rho = 0 control: null mean t^2 {control['t2']:.3f}, "
                 f"type-I {control['t1_05']:.4f} / {control['t1_01']:.4f}, "
                 f"corr(beta_a, beta_t) {control['corr']:.3f}")
    summary = '\n'.join(lines)
    print('\n' + summary)

    for name, (nul, cov) in arms.items():
        assert abs(nul['t2'] - 1.0) <= 0.1, f'{name}: mean t^2 = {nul["t2"]:.3f}\n{summary}'
        assert abs(nul['t1_05'] - 0.05) <= tol(0.05, n_null), \
            f'{name}: type-I at 0.05 = {nul["t1_05"]:.4f}, tolerance {tol(0.05, n_null):.4f}\n{summary}'
        assert abs(nul['t1_01'] - 0.01) <= tol(0.01, n_null), \
            f'{name}: type-I at 0.01 = {nul["t1_01"]:.4f}, tolerance {tol(0.01, n_null):.4f}\n{summary}'
        assert abs(cov['refit']['cov'] - 0.95) <= tol(0.95, n_cov), \
            f"{name}: lead-refit coverage = {cov['refit']['cov']:.4f}, tolerance {tol(0.95, n_cov):.4f}\n{summary}"
        assert cov['nominal']['cov'] >= 0.95 - tol(0.95, n_cov), \
            f"{name}: nominal-path coverage = {cov['nominal']['cov']:.4f} under-covers\n{summary}"


def test_per_channel_covariates_stay_calibrated():
    """BrainVar's layout: covariates that move total expression are projected
    out of the total channel, the allelic channel is through-origin, the
    variances are heteroskedastic with unmodelled biological variance, and 10%
    of samples carry no allele-specific coverage. The null must stay nominal
    on the shared t reference (N - 2 - n_cov), and a through-origin allelic channel
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

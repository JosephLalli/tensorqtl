"""
Tests for tensorqtl.hapmixqtl module.

hapmixQTL performs cis-QTL mapping using haplotype-resolved expression
posteriors (Salmon Gibbs draws), incorporating inferential uncertainty into
beta and beta_se via inverse-variance weighted regression.

These tests validate the core mathematical properties of Method A:
  - WLS via the sqrt-weight transform reproduces statsmodels WLS
  - The inverse-variance meta-analysis down-weights uncertain channels
  - When Va -> infinity the combined estimate collapses to total-only
  - When there is no phase (s=0) the ASE channel drops out and the combined
    estimate collapses to total-only
  - A true allelic fold change is recovered from heterozygotes
  - Sample-ordering, dtype and device consistency
"""

import pytest
import numpy as np
import pandas as pd
import torch
import sys
from pathlib import Path

# Add tensorqtl to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import tensorqtl.hapmixqtl as hapmixqtl
from tensorqtl.hapmixqtl import (
    WeightedResidualizer,
    _wls_regression,
    _estimate_tau,
    calculate_hapmixqtl_nominal,
    calculate_hapmixqtl_permutations,
    compute_summaries_from_gibbs,
    read_hapmixqtl_inputs,
    map_nominal,
    map_cis,
    map_susie,
)


# ---------------------------------------------------------------------------
#  Reference implementations (numpy) for cross-checking
# ---------------------------------------------------------------------------

def _wls_reference(y, X, w):
    """
    Reference known-variance GLS via normal equations.

    The weights are treated as absolute precisions (w_i = 1/Var(error_i) with
    Var known), so the coefficient covariance is (X'WX)^-1 with NO estimated
    dispersion factor. This mirrors hapmixqtl._wls_regression, where the Gibbs
    inferential variances are known measurement variances.

    Args:
        y: [N] response
        X: [N, p] design (including intercept column)
        w: [N] weights (absolute precisions)
    Returns:
        beta: [p] coefficients
        se:   [p] known-variance standard errors
    """
    W = np.diag(w)
    XtWX = X.T @ W @ X
    XtWX_inv = np.linalg.inv(XtWX)
    beta = XtWX_inv @ (X.T @ W @ y)
    # Known-variance GLS: Cov(beta_hat) = (X'WX)^-1 (no sigma^2 factor).
    se = np.sqrt(np.diag(XtWX_inv))
    return beta, se


def _make_gaussian_seed(seed):
    rng = np.random.RandomState(seed)
    return rng


# ---------------------------------------------------------------------------
#  WeightedResidualizer
# ---------------------------------------------------------------------------

class TestWeightedResidualizer:

    def test_intercept_only_projection(self, device):
        """With no covariates, transform removes the weighted-intercept component."""
        N = 50
        rng = _make_gaussian_seed(0)
        w = rng.uniform(0.1, 5.0, N)
        sqrt_w = torch.tensor(np.sqrt(w), dtype=torch.float32, device=device)

        res = WeightedResidualizer(None, sqrt_w)

        # dof = N - 1 - 1 (one intercept column)
        assert res.dof == N - 2

        M = torch.tensor(rng.normal(0, 1, (3, N)), dtype=torch.float32, device=device)
        M_res = res.transform(M)

        # Residual must be orthogonal to the weighted-intercept column sqrt_w
        dotp = (M_res * sqrt_w.unsqueeze(0)).sum(1)
        assert torch.allclose(dotp, torch.zeros_like(dotp), atol=1e-4)

    def test_covariate_projection(self, device):
        """Transform removes both weighted intercept and weighted covariates."""
        N = 60
        rng = _make_gaussian_seed(1)
        w = rng.uniform(0.2, 3.0, N)
        sqrt_w = torch.tensor(np.sqrt(w), dtype=torch.float32, device=device)
        C = torch.tensor(rng.normal(0, 1, (N, 3)), dtype=torch.float32, device=device)

        res = WeightedResidualizer(C, sqrt_w)
        assert res.dof == N - 1 - 4  # intercept + 3 covariates

        M = torch.tensor(rng.normal(0, 1, (5, N)), dtype=torch.float32, device=device)
        M_res = res.transform(M)

        # Orthogonal to each weighted design column
        design = torch.cat([sqrt_w.unsqueeze(1), sqrt_w.unsqueeze(1) * C], dim=1)
        proj = torch.mm(M_res, design)
        assert torch.allclose(proj, torch.zeros_like(proj), atol=1e-3)


# ---------------------------------------------------------------------------
#  _wls_regression vs. reference WLS
# ---------------------------------------------------------------------------

class TestWLSRegression:

    def test_matches_reference_no_cov(self, device):
        """Single-predictor WLS matches numpy normal-equation reference."""
        N = 80
        rng = _make_gaussian_seed(2)
        x = rng.normal(0, 1, N)
        w = rng.uniform(0.5, 4.0, N)
        beta_true = 1.7
        y = 0.3 + beta_true * x + rng.normal(0, 0.5, N)

        # Reference: design with intercept
        X = np.column_stack([np.ones(N), x])
        beta_ref, se_ref = _wls_reference(y, X, w)

        sqrt_w = torch.tensor(np.sqrt(w), dtype=torch.float64, device=device)
        res = WeightedResidualizer(None, sqrt_w)
        y_star = torch.tensor(y * np.sqrt(w), dtype=torch.float64, device=device).unsqueeze(0)
        x_star = torch.tensor(x * np.sqrt(w), dtype=torch.float64, device=device).unsqueeze(0)

        slope, slope_se = _wls_regression(y_star, x_star, res)

        assert np.isclose(slope.item(), beta_ref[1], atol=1e-6)
        assert np.isclose(slope_se.item(), se_ref[1], atol=1e-6)

    def test_matches_reference_with_cov(self, device):
        """WLS with covariates matches numpy reference for the predictor slope."""
        N = 100
        rng = _make_gaussian_seed(3)
        x = rng.normal(0, 1, N)
        C = rng.normal(0, 1, (N, 2))
        w = rng.uniform(0.3, 2.5, N)
        y = 0.5 + 1.1 * x + 0.7 * C[:, 0] - 0.4 * C[:, 1] + rng.normal(0, 0.4, N)

        X = np.column_stack([np.ones(N), x, C])
        beta_ref, se_ref = _wls_reference(y, X, w)

        sqrt_w_np = np.sqrt(w)
        sqrt_w = torch.tensor(sqrt_w_np, dtype=torch.float64, device=device)
        C_t = torch.tensor(C, dtype=torch.float64, device=device)
        res = WeightedResidualizer(C_t, sqrt_w)
        y_star = torch.tensor(y * sqrt_w_np, dtype=torch.float64, device=device).unsqueeze(0)
        x_star = torch.tensor(x * sqrt_w_np, dtype=torch.float64, device=device).unsqueeze(0)

        slope, slope_se = _wls_regression(y_star, x_star, res)

        # beta_ref[1] is the x slope (index 0 is intercept)
        assert np.isclose(slope.item(), beta_ref[1], atol=1e-6)
        assert np.isclose(slope_se.item(), se_ref[1], atol=1e-6)

    def test_zero_variance_predictor(self, device):
        """A constant predictor yields slope 0 and infinite SE (skipped safely)."""
        N = 40
        rng = _make_gaussian_seed(4)
        y = rng.normal(0, 1, N)
        w = np.ones(N)
        sqrt_w = torch.tensor(np.sqrt(w), dtype=torch.float32, device=device)
        res = WeightedResidualizer(None, sqrt_w)

        y_star = torch.tensor(y * np.sqrt(w), dtype=torch.float32, device=device).unsqueeze(0)
        # Constant predictor -> zero variance after intercept removal
        x_star = torch.ones(1, N, dtype=torch.float32, device=device) * sqrt_w.unsqueeze(0)

        slope, slope_se = _wls_regression(y_star, x_star, res)
        assert slope.item() == 0.0
        assert not torch.isfinite(slope_se).item()

    def test_batched_predictors(self, device):
        """Multiple predictors regressed simultaneously match per-predictor WLS."""
        N = 70
        V = 6
        rng = _make_gaussian_seed(5)
        w = rng.uniform(0.5, 2.0, N)
        sqrt_w_np = np.sqrt(w)
        y = rng.normal(0, 1, N)

        X_all = rng.normal(0, 1, (V, N))
        sqrt_w = torch.tensor(sqrt_w_np, dtype=torch.float64, device=device)
        res = WeightedResidualizer(None, sqrt_w)
        y_star = torch.tensor(y * sqrt_w_np, dtype=torch.float64, device=device).unsqueeze(0)
        x_star = torch.tensor(X_all * sqrt_w_np, dtype=torch.float64, device=device)

        slopes, ses = _wls_regression(y_star, x_star, res)

        for v in range(V):
            Xv = np.column_stack([np.ones(N), X_all[v]])
            beta_ref, se_ref = _wls_reference(y, Xv, w)
            assert np.isclose(slopes[v].item(), beta_ref[1], atol=1e-6)
            assert np.isclose(ses[v].item(), se_ref[1], atol=1e-6)


# ---------------------------------------------------------------------------
#  compute_summaries_from_gibbs
# ---------------------------------------------------------------------------

class TestGibbsSummaries:

    def test_shapes_and_values(self):
        """Summaries have the right shapes and match a manual computation."""
        rng = _make_gaussian_seed(6)
        F, S, D = 3, 10, 25
        yL = rng.gamma(2.0, 2.0, (F, S, D))
        yR = rng.gamma(2.0, 2.0, (F, S, D))
        kappa = 0.5

        A, T, Va, Vt, Cat = compute_summaries_from_gibbs(yL, yR, kappa=kappa)

        assert A.shape == (F, S)
        assert T.shape == (F, S)
        assert Va.shape == (F, S)
        assert Vt.shape == (F, S)
        assert Cat.shape == (F, S)

        # Manual check for one entry
        a_draws = np.log(yL[0, 0] + kappa) - np.log(yR[0, 0] + kappa)
        t_draws = np.log((yL[0, 0] + yR[0, 0]) / 2 + kappa)
        assert np.isclose(A[0, 0], a_draws.mean())
        assert np.isclose(T[0, 0], t_draws.mean())
        assert np.isclose(Va[0, 0], a_draws.var())
        assert np.isclose(Vt[0, 0], t_draws.var())
        cov = np.mean((a_draws - a_draws.mean()) * (t_draws - t_draws.mean()))
        assert np.isclose(Cat[0, 0], cov)

    def test_variance_nonnegative(self):
        """Inferential variances are always non-negative."""
        rng = _make_gaussian_seed(7)
        yL = rng.gamma(1.0, 1.0, (2, 5, 30))
        yR = rng.gamma(1.0, 1.0, (2, 5, 30))
        _, _, Va, Vt, _ = compute_summaries_from_gibbs(yL, yR)
        assert (Va >= 0).all()
        assert (Vt >= 0).all()


# ---------------------------------------------------------------------------
#  calculate_hapmixqtl_nominal: core statistical behaviour
# ---------------------------------------------------------------------------

def _build_channel_inputs(genotypes, sign, a, t, va, vt, device, dtype=torch.float64):
    """Helper to build tensors for calculate_hapmixqtl_nominal."""
    genotypes_t = torch.tensor(genotypes, dtype=dtype, device=device)
    sign_t = torch.tensor(sign, dtype=dtype, device=device)
    a_t = torch.tensor(a, dtype=dtype, device=device)
    t_t = torch.tensor(t, dtype=dtype, device=device)
    va_t = torch.tensor(va, dtype=dtype, device=device).clamp(min=1e-8)
    vt_t = torch.tensor(vt, dtype=dtype, device=device).clamp(min=1e-8)
    sqrt_wa_t = torch.sqrt(1.0 / va_t)
    sqrt_wt_t = torch.sqrt(1.0 / vt_t)
    res_a = WeightedResidualizer(None, sqrt_wa_t)
    res_t = WeightedResidualizer(None, sqrt_wt_t)
    return (genotypes_t, sign_t, a_t, t_t, sqrt_wa_t, sqrt_wt_t, res_a, res_t)


class TestNominalAssociation:

    def test_no_phase_matches_total_only(self, device):
        """When s=0 for all samples, combined == total-channel regression."""
        N = 120
        V = 4
        rng = _make_gaussian_seed(10)

        genotypes = rng.choice([0, 1, 2], size=(V, N)).astype(float)
        sign = np.zeros((V, N))  # no phase
        # total expression driven by genotype
        beta_true = 0.8
        t = 1.0 + beta_true * (genotypes[0] / 2) + rng.normal(0, 0.3, N)
        a = rng.normal(0, 0.2, N)  # ASE noise, unrelated
        va = rng.uniform(0.1, 0.5, N)
        vt = rng.uniform(0.1, 0.5, N)

        inputs = _build_channel_inputs(genotypes, sign, a, t, va, vt, device)
        tstat, slope, se, slope_a, se_a, slope_t, se_t = \
            calculate_hapmixqtl_nominal(*inputs)

        # ASE channel has zero-variance predictor (s=0) -> se_a = inf, dropped
        assert not torch.isfinite(se_a).any()
        # Combined equals total channel exactly
        assert torch.allclose(slope, slope_t, atol=1e-6)
        assert torch.allclose(se, se_t, atol=1e-6)

    def test_huge_va_collapses_to_total(self, device):
        """Va -> infinity kills ASE weight; combined -> total-only."""
        N = 100
        V = 3
        rng = _make_gaussian_seed(11)

        genotypes = rng.choice([0, 1, 2], size=(V, N)).astype(float)
        # Give hets a real signed indicator so ASE predictor is non-degenerate
        sign = np.where(genotypes == 1, rng.choice([-1, 1], size=(V, N)), 0).astype(float)
        t = 1.0 + 0.6 * (genotypes[0] / 2) + rng.normal(0, 0.3, N)
        a = 0.5 * sign[0] + rng.normal(0, 0.2, N)
        vt = rng.uniform(0.1, 0.5, N)
        va = np.full(N, 1e12)  # enormous ASE uncertainty

        inputs = _build_channel_inputs(genotypes, sign, a, t, va, vt, device)
        tstat, slope, se, slope_a, se_a, slope_t, se_t = \
            calculate_hapmixqtl_nominal(*inputs)

        # ASE SE is astronomically large -> negligible weight
        assert torch.allclose(slope, slope_t, atol=1e-4)
        assert torch.allclose(se, se_t, atol=1e-4)

    def test_recovers_true_afc_from_hets(self, device):
        """
        When both channels agree on a true log aFC, the combined slope
        recovers it (both ASE and total centered on the same beta).
        """
        N = 400
        rng = _make_gaussian_seed(12)
        beta = 0.9  # true log aFC

        # Genotypes: force a good number of heterozygotes
        genotypes = rng.choice([0, 1, 2], size=(1, N), p=[0.25, 0.5, 0.25]).astype(float)
        het = genotypes[0] == 1
        # Phase: random assignment of ALT to L/R for hets
        s = np.zeros(N)
        s[het] = rng.choice([-1, 1], size=het.sum())

        # ASE channel: a = beta * s + small noise
        a = beta * s + rng.normal(0, 0.05, N)
        # Total channel: t = beta * (g/2) + intercept + small noise
        t = 2.0 + beta * (genotypes[0] / 2) + rng.normal(0, 0.05, N)

        va = np.full(N, 0.01)
        vt = np.full(N, 0.01)

        inputs = _build_channel_inputs(genotypes, s.reshape(1, N), a, t, va, vt, device)
        tstat, slope, se, slope_a, se_t_slope, slope_t, se_t = \
            calculate_hapmixqtl_nominal(*inputs)

        assert np.isclose(slope_a.item(), beta, atol=0.05), f"ASE slope {slope_a.item()}"
        assert np.isclose(slope_t.item(), beta, atol=0.1), f"total slope {slope_t.item()}"
        assert np.isclose(slope.item(), beta, atol=0.05), f"combined slope {slope.item()}"

    def test_inverse_variance_combine_formula(self, device):
        """
        Combined slope/SE follow the standard inverse-variance meta formula:
        beta_c = (b_a/se_a^2 + b_t/se_t^2)/(1/se_a^2 + 1/se_t^2)
        se_c   = sqrt(1/(1/se_a^2 + 1/se_t^2))
        """
        N = 200
        rng = _make_gaussian_seed(13)
        genotypes = rng.choice([0, 1, 2], size=(1, N), p=[0.25, 0.5, 0.25]).astype(float)
        het = genotypes[0] == 1
        s = np.zeros(N)
        s[het] = rng.choice([-1, 1], size=het.sum())
        a = 0.7 * s + rng.normal(0, 0.2, N)
        t = 1.0 + 0.7 * (genotypes[0] / 2) + rng.normal(0, 0.2, N)
        va = rng.uniform(0.1, 0.4, N)
        vt = rng.uniform(0.1, 0.4, N)

        inputs = _build_channel_inputs(genotypes, s.reshape(1, N), a, t, va, vt, device)
        tstat, slope, se, slope_a, se_a, slope_t, se_t = \
            calculate_hapmixqtl_nominal(*inputs)

        iva = 1.0 / se_a.item()**2
        ivt = 1.0 / se_t.item()**2
        beta_expected = (slope_a.item() * iva + slope_t.item() * ivt) / (iva + ivt)
        se_expected = np.sqrt(1.0 / (iva + ivt))

        assert np.isclose(slope.item(), beta_expected, atol=1e-6)
        assert np.isclose(se.item(), se_expected, atol=1e-6)
        assert np.isclose(tstat.item(), slope.item() / se.item(), atol=1e-6)

    def test_total_channel_uses_half_dosage(self, device):
        """
        The total-channel slope estimates the FULL aFC (g/2 predictor), so a
        phenotype simulated as beta*(g/2) recovers beta, not beta/2.
        """
        N = 300
        rng = _make_gaussian_seed(14)
        genotypes = rng.choice([0, 1, 2], size=(1, N)).astype(float)
        sign = np.zeros((1, N))
        beta = 1.2
        t = beta * (genotypes[0] / 2) + rng.normal(0, 0.05, N)
        a = rng.normal(0, 0.1, N)
        va = np.full(N, 1.0)
        vt = np.full(N, 0.01)

        inputs = _build_channel_inputs(genotypes, sign, a, t, va, vt, device)
        _, _, _, _, _, slope_t, _ = calculate_hapmixqtl_nominal(*inputs)
        assert np.isclose(slope_t.item(), beta, atol=0.05)

    def test_se_shrinks_with_n(self, device):
        """
        Spec requirement: with a true allelic effect and small Va, the SE
        shrinks as N increases (roughly like 1/sqrt(N)).
        """
        beta = 0.8

        def _se_for_n(N, seed):
            rng = _make_gaussian_seed(seed)
            genotypes = rng.choice([0, 1, 2], size=(1, N), p=[0.25, 0.5, 0.25]).astype(float)
            het = genotypes[0] == 1
            s = np.zeros(N)
            s[het] = rng.choice([-1, 1], size=het.sum())
            a = beta * s + rng.normal(0, 0.1, N)
            t = 1.0 + beta * (genotypes[0] / 2) + rng.normal(0, 0.1, N)
            va = np.full(N, 0.05)
            vt = np.full(N, 0.05)
            inputs = _build_channel_inputs(genotypes, s.reshape(1, N), a, t, va, vt, device)
            _, _, se, _, _, _, _ = calculate_hapmixqtl_nominal(*inputs)
            return se.item()

        se_small = _se_for_n(100, 40)
        se_large = _se_for_n(400, 41)
        # Larger N -> smaller SE
        assert se_large < se_small
        # Roughly 1/sqrt(N): 4x samples -> ~2x smaller SE (loose bounds)
        ratio = se_small / se_large
        assert 1.5 < ratio < 2.7, f"SE ratio {ratio}"

    def test_known_variance_se_scales_with_uncertainty(self, device):
        """
        Known-variance behaviour: uniformly inflating the inferential variance
        of a channel inflates its SE by the same sqrt-factor (an
        estimated-dispersion SE would be invariant to this scaling).
        """
        N = 200
        rng = _make_gaussian_seed(42)
        genotypes = rng.choice([0, 1, 2], size=(1, N)).astype(float)
        sign = np.zeros((1, N))
        t = 1.0 + 0.5 * (genotypes[0] / 2) + rng.normal(0, 0.3, N)
        a = rng.normal(0, 0.1, N)
        va = np.full(N, 1.0)
        vt_base = rng.uniform(0.1, 0.5, N)

        inp1 = _build_channel_inputs(genotypes, sign, a, t, va, vt_base, device)
        _, _, _, _, _, slope1, se1 = calculate_hapmixqtl_nominal(*inp1)

        # Inflate total-channel inferential variance by 4x -> SE up by ~2x,
        # slope unchanged (weights scaled uniformly).
        inp4 = _build_channel_inputs(genotypes, sign, a, t, va, vt_base * 4.0, device)
        _, _, _, _, _, slope4, se4 = calculate_hapmixqtl_nominal(*inp4)

        assert np.isclose(slope1.item(), slope4.item(), atol=1e-5)
        assert np.isclose(se4.item() / se1.item(), 2.0, atol=1e-3)

    def test_robust_se_differs_from_model(self, device):
        """Robust (HC1) SEs are computed and differ under heteroskedasticity."""
        N = 200
        rng = _make_gaussian_seed(15)
        genotypes = rng.choice([0, 1, 2], size=(1, N)).astype(float)
        sign = np.zeros((1, N))
        # Heteroskedastic noise scaling with genotype
        noise = rng.normal(0, 1, N) * (0.1 + 0.5 * genotypes[0])
        t = 0.5 * (genotypes[0] / 2) + noise
        a = rng.normal(0, 0.1, N)
        va = np.full(N, 1.0)
        vt = np.full(N, 0.2)

        inputs = _build_channel_inputs(genotypes, sign, a, t, va, vt, device)
        _, _, _, _, _, slope_t_m, se_t_model = calculate_hapmixqtl_nominal(*inputs, robust=False)
        _, _, _, _, _, slope_t_r, se_t_robust = calculate_hapmixqtl_nominal(*inputs, robust=True)

        # Slopes identical; SEs differ under heteroskedasticity
        assert np.isclose(slope_t_m.item(), slope_t_r.item(), atol=1e-8)
        assert not np.isclose(se_t_model.item(), se_t_robust.item(), atol=1e-4)


# ---------------------------------------------------------------------------
#  tau estimation
# ---------------------------------------------------------------------------

class TestTauEstimation:

    def test_tau_nonnegative(self, device):
        """Estimated tau is clamped to be non-negative."""
        N = 100
        rng = _make_gaussian_seed(20)
        # Small inferential variance, extra biological dispersion present
        v_inf = np.full(N, 0.1)
        y = rng.normal(0, 1.0, N)  # variance >> v_inf -> positive tau
        y_t = torch.tensor(y, dtype=torch.float64, device=device)
        v_t = torch.tensor(v_inf, dtype=torch.float64, device=device)
        tau = _estimate_tau(y_t, v_t, None, device)
        assert tau.item() >= 0.0

    def test_tau_zero_when_overweighted(self, device):
        """If residual variance is below the weighting scale, tau clamps to 0."""
        N = 100
        rng = _make_gaussian_seed(21)
        # Huge v_inf -> weighted residual variance tiny -> tau=0
        v_inf = np.full(N, 100.0)
        y = rng.normal(0, 0.01, N)
        y_t = torch.tensor(y, dtype=torch.float64, device=device)
        v_t = torch.tensor(v_inf, dtype=torch.float64, device=device)
        tau = _estimate_tau(y_t, v_t, None, device)
        assert tau.item() == 0.0


# ---------------------------------------------------------------------------
#  Sample ordering / dtype / device
# ---------------------------------------------------------------------------

class TestConsistency:

    def test_dtype_consistency(self, device):
        """float32 and float64 give close results."""
        N = 100
        rng = _make_gaussian_seed(30)
        genotypes = rng.choice([0, 1, 2], size=(2, N)).astype(float)
        het = genotypes[0] == 1
        sign = np.where(genotypes == 1, rng.choice([-1, 1], size=(2, N)), 0).astype(float)
        a = 0.5 * sign[0] + rng.normal(0, 0.2, N)
        t = 1.0 + 0.5 * (genotypes[0] / 2) + rng.normal(0, 0.2, N)
        va = rng.uniform(0.1, 0.5, N)
        vt = rng.uniform(0.1, 0.5, N)

        inp64 = _build_channel_inputs(genotypes, sign, a, t, va, vt, device, dtype=torch.float64)
        inp32 = _build_channel_inputs(genotypes, sign, a, t, va, vt, device, dtype=torch.float32)
        r64 = calculate_hapmixqtl_nominal(*inp64)
        r32 = calculate_hapmixqtl_nominal(*inp32)

        for x64, x32 in zip(r64, r32):
            finite = torch.isfinite(x64) & torch.isfinite(x32)
            if finite.any():
                assert torch.allclose(
                    x64[finite].to(torch.float32), x32[finite], atol=1e-3, rtol=1e-3
                )

    def test_device_matches_input(self, device):
        """Output tensors live on the same device as inputs."""
        N = 50
        rng = _make_gaussian_seed(31)
        genotypes = rng.choice([0, 1, 2], size=(1, N)).astype(float)
        sign = np.zeros((1, N))
        a = rng.normal(0, 1, N)
        t = rng.normal(0, 1, N)
        va = np.full(N, 0.3)
        vt = np.full(N, 0.3)
        inputs = _build_channel_inputs(genotypes, sign, a, t, va, vt, device)
        results = calculate_hapmixqtl_nominal(*inputs)
        for r in results:
            assert r.device.type == device.type

    def test_permutation_row_alignment(self, device):
        """
        Reordering samples consistently across genotypes, phase, expression and
        weights leaves the nominal statistics unchanged.
        """
        N = 150
        rng = _make_gaussian_seed(32)
        genotypes = rng.choice([0, 1, 2], size=(3, N), p=[0.25, 0.5, 0.25]).astype(float)
        sign = np.where(genotypes == 1, rng.choice([-1, 1], size=(3, N)), 0).astype(float)
        a = 0.6 * sign[0] + rng.normal(0, 0.2, N)
        t = 1.0 + 0.6 * (genotypes[0] / 2) + rng.normal(0, 0.2, N)
        va = rng.uniform(0.1, 0.5, N)
        vt = rng.uniform(0.1, 0.5, N)

        inputs = _build_channel_inputs(genotypes, sign, a, t, va, vt, device)
        base = calculate_hapmixqtl_nominal(*inputs)

        perm = rng.permutation(N)
        inputs_p = _build_channel_inputs(
            genotypes[:, perm], sign[:, perm], a[perm], t[perm], va[perm], vt[perm], device
        )
        permuted = calculate_hapmixqtl_nominal(*inputs_p)

        for b, p in zip(base, permuted):
            finite = torch.isfinite(b) & torch.isfinite(p)
            if finite.any():
                assert torch.allclose(b[finite], p[finite], atol=1e-6)


# ---------------------------------------------------------------------------
#  End-to-end: map_nominal and map_cis with synthetic BED-like inputs
# ---------------------------------------------------------------------------

def _make_dataset(seed=100, n_samples=80, n_variants=30, n_phenotypes=3):
    """Build a small synthetic hapmixQTL dataset with a planted association."""
    rng = np.random.RandomState(seed)
    samples = [f"S{i:03d}" for i in range(n_samples)]
    variant_ids = [f"chr1_{10000 + i * 1000}_A_G" for i in range(n_variants)]
    pheno_ids = [f"ENSG{i:08d}.1" for i in range(n_phenotypes)]

    # Genotypes (variants x samples)
    genotypes = rng.choice([0, 1, 2], size=(n_variants, n_samples),
                           p=[0.25, 0.5, 0.25]).astype(np.float32)
    genotype_df = pd.DataFrame(genotypes, index=variant_ids, columns=samples)
    variant_df = pd.DataFrame({
        'chrom': ['chr1'] * n_variants,
        'pos': [10000 + i * 1000 for i in range(n_variants)],
    }, index=variant_ids)

    # Phase: signed het indicator built from a random ALT-on-L assignment
    xL = np.zeros((n_variants, n_samples), dtype=np.float32)
    xR = np.zeros((n_variants, n_samples), dtype=np.float32)
    het = genotypes == 1
    L_alt = rng.rand(n_variants, n_samples) < 0.5
    # het & L_alt -> ALT on L; het & ~L_alt -> ALT on R
    xL[het & L_alt] = 1
    xR[het & ~L_alt] = 1
    # homozygous ALT: both haplotypes ALT
    homalt = genotypes == 2
    xL[homalt] = 1
    xR[homalt] = 1
    xL_df = pd.DataFrame(xL, index=variant_ids, columns=samples)
    xR_df = pd.DataFrame(xR, index=variant_ids, columns=samples)

    # Build expression: phenotype 0 driven by variant 0 with aFC beta
    beta = 1.0
    A = np.zeros((n_phenotypes, n_samples), dtype=np.float32)
    T = np.zeros((n_phenotypes, n_samples), dtype=np.float32)
    Va = np.zeros((n_phenotypes, n_samples), dtype=np.float32)
    Vt = np.zeros((n_phenotypes, n_samples), dtype=np.float32)
    for p in range(n_phenotypes):
        if p == 0:
            s0 = xL[0] - xR[0]
            A[p] = beta * s0 + rng.normal(0, 0.1, n_samples)
            T[p] = 2.0 + beta * (genotypes[0] / 2) + rng.normal(0, 0.1, n_samples)
        else:
            A[p] = rng.normal(0, 0.3, n_samples)
            T[p] = 2.0 + rng.normal(0, 0.3, n_samples)
        Va[p] = rng.uniform(0.02, 0.1, n_samples)
        Vt[p] = rng.uniform(0.02, 0.1, n_samples)

    A_df = pd.DataFrame(A, index=pheno_ids, columns=samples)
    T_df = pd.DataFrame(T, index=pheno_ids, columns=samples)
    Va_df = pd.DataFrame(Va, index=pheno_ids, columns=samples)
    Vt_df = pd.DataFrame(Vt, index=pheno_ids, columns=samples)
    pos_df = pd.DataFrame({
        'chr': ['chr1'] * n_phenotypes,
        'pos': [10000, 15000, 20000][:n_phenotypes],
    }, index=pheno_ids)

    return dict(
        genotype_df=genotype_df, variant_df=variant_df,
        A_df=A_df, T_df=T_df, Va_df=Va_df, Vt_df=Vt_df,
        xL_df=xL_df, xR_df=xR_df, pos_df=pos_df,
        beta=beta, causal_variant=variant_ids[0], causal_pheno=pheno_ids[0],
    )


class TestMapNominal:

    def test_map_nominal_writes_output(self, temp_dir):
        """map_nominal runs end-to-end and writes a valid parquet file."""
        d = _make_dataset(seed=100)
        map_nominal(
            d['genotype_df'], d['variant_df'],
            d['A_df'], d['T_df'], d['Va_df'], d['Vt_df'],
            d['pos_df'], xL_df=d['xL_df'], xR_df=d['xR_df'],
            prefix='test', window=1000000, output_dir=temp_dir, verbose=False,
        )
        out = Path(temp_dir) / 'test.hapmixqtl_pairs.chr1.parquet'
        assert out.exists()
        res = pd.read_parquet(out)

        # Required columns present
        for col in ['phenotype_id', 'variant_id', 'pval_nominal', 'slope', 'slope_se',
                    'pval_a', 'slope_a', 'slope_a_se', 'pval_t', 'slope_t', 'slope_t_se']:
            assert col in res.columns, f"missing {col}"

        # p-values in [0, 1]
        pv = res['pval_nominal'].dropna()
        assert (pv >= 0).all() and (pv <= 1).all()

    def test_map_nominal_recovers_causal(self, temp_dir):
        """The causal variant is the most significant pair for the causal gene."""
        d = _make_dataset(seed=101)
        map_nominal(
            d['genotype_df'], d['variant_df'],
            d['A_df'], d['T_df'], d['Va_df'], d['Vt_df'],
            d['pos_df'], xL_df=d['xL_df'], xR_df=d['xR_df'],
            prefix='test', window=1000000, output_dir=temp_dir, verbose=False,
        )
        res = pd.read_parquet(Path(temp_dir) / 'test.hapmixqtl_pairs.chr1.parquet')
        sub = res[res['phenotype_id'] == d['causal_pheno']].copy()
        top = sub.loc[sub['pval_nominal'].idxmin()]
        assert top['variant_id'] == d['causal_variant']
        # Recovered slope near the true aFC
        assert abs(top['slope'] - d['beta']) < 0.2

    def test_map_nominal_no_phase_equals_total(self, temp_dir):
        """
        Without phase inputs, map_nominal falls back to the total channel: the
        combined slope equals the total-channel slope for every pair.
        """
        d = _make_dataset(seed=102)
        map_nominal(
            d['genotype_df'], d['variant_df'],
            d['A_df'], d['T_df'], d['Va_df'], d['Vt_df'],
            d['pos_df'], xL_df=None, xR_df=None,
            prefix='nophase', window=1000000, output_dir=temp_dir, verbose=False,
        )
        res = pd.read_parquet(Path(temp_dir) / 'nophase.hapmixqtl_pairs.chr1.parquet')
        # Combined == total channel
        np.testing.assert_allclose(
            res['slope'].values, res['slope_t'].values, atol=1e-4
        )
        np.testing.assert_allclose(
            res['slope_se'].values, res['slope_t_se'].values, atol=1e-4
        )


class TestMapCis:

    def test_map_cis_runs_and_recovers_causal(self):
        """map_cis produces empirical p-values and finds the causal variant."""
        d = _make_dataset(seed=103)
        res_df = map_cis(
            d['genotype_df'], d['variant_df'],
            d['A_df'], d['T_df'], d['Va_df'], d['Vt_df'],
            d['pos_df'], xL_df=d['xL_df'], xR_df=d['xR_df'],
            nperm=1000, window=1000000, seed=42, verbose=False,
        )
        assert d['causal_pheno'] in res_df.index
        row = res_df.loc[d['causal_pheno']]
        assert row['variant_id'] == d['causal_variant']
        # Empirical p-value valid
        assert 0 < row['pval_perm'] <= 1
        # Strong association -> small permutation p-value
        assert row['pval_perm'] < 0.05

    def test_map_cis_beta_approx_columns(self):
        """Beta-approximation populates pval_beta and shape parameters."""
        d = _make_dataset(seed=104)
        res_df = map_cis(
            d['genotype_df'], d['variant_df'],
            d['A_df'], d['T_df'], d['Va_df'], d['Vt_df'],
            d['pos_df'], xL_df=d['xL_df'], xR_df=d['xR_df'],
            nperm=1000, beta_approx=True, window=1000000, seed=7, verbose=False,
        )
        row = res_df.loc[d['causal_pheno']]
        assert np.isfinite(row['pval_beta'])
        assert np.isfinite(row['beta_shape1'])
        assert np.isfinite(row['beta_shape2'])

    def test_map_cis_with_maf_filter_keeps_phase_aligned(self):
        """
        With a MAF filter (and monomorphic exclusion) the phase-derived sign
        matrix must stay row-aligned with the filtered genotypes. Inject a
        rare and a monomorphic variant, then confirm the causal variant is
        still recovered without shape/index errors.
        """
        d = _make_dataset(seed=110, n_samples=100, n_variants=25)
        g = d['genotype_df']
        # Force variant 5 monomorphic and variant 7 very rare (single het).
        g.iloc[5, :] = 0
        g.iloc[7, :] = 0
        g.iloc[7, 0] = 1
        d['xL_df'].iloc[5, :] = 0
        d['xR_df'].iloc[5, :] = 0
        d['xL_df'].iloc[7, :] = 0
        d['xR_df'].iloc[7, :] = 0
        d['xL_df'].iloc[7, 0] = 1  # ALT on L for the single het

        res_df = map_cis(
            g, d['variant_df'],
            d['A_df'], d['T_df'], d['Va_df'], d['Vt_df'],
            d['pos_df'], xL_df=d['xL_df'], xR_df=d['xR_df'],
            nperm=500, maf_threshold=0.05, window=1000000, seed=11,
            verbose=False, warn_monomorphic=False,
        )
        row = res_df.loc[d['causal_pheno']]
        assert row['variant_id'] == d['causal_variant']
        # Filtered-out variants must never be selected as the top hit.
        assert row['variant_id'] not in {g.index[5], g.index[7]}


class TestMapSusie:

    def test_map_susie_recovers_causal_in_cs(self):
        """SuSiE fine-mapping places the causal variant in a credible set."""
        d = _make_dataset(seed=120, n_samples=120, n_variants=20)
        summary_df, res = map_susie(
            d['genotype_df'], d['variant_df'],
            d['A_df'], d['T_df'], d['Va_df'], d['Vt_df'],
            d['pos_df'], xL_df=d['xL_df'], xR_df=d['xR_df'],
            L=5, window=1000000, max_iter=200,
            summary_only=False, verbose=False,
        )
        # Causal phenotype should have at least one credible set containing
        # the causal variant.
        assert d['causal_pheno'] in res
        sub = summary_df[summary_df['phenotype_id'] == d['causal_pheno']]
        assert d['causal_variant'] in set(sub['variant_id'])
        # The causal variant should carry high PIP.
        pip = sub.loc[sub['variant_id'] == d['causal_variant'], 'pip'].max()
        assert pip > 0.5, f"causal PIP {pip}"

    def test_map_susie_summary_only_columns(self):
        """summary_only=True returns a tidy credible-set table."""
        d = _make_dataset(seed=121, n_samples=100, n_variants=15)
        summary_df = map_susie(
            d['genotype_df'], d['variant_df'],
            d['A_df'], d['T_df'], d['Va_df'], d['Vt_df'],
            d['pos_df'], xL_df=d['xL_df'], xR_df=d['xR_df'],
            L=5, window=1000000, max_iter=200,
            summary_only=True, verbose=False,
        )
        for col in ['phenotype_id', 'variant_id', 'pip', 'af', 'cs_id']:
            assert col in summary_df.columns
        # PIPs are valid probabilities.
        assert (summary_df['pip'] >= 0).all() and (summary_df['pip'] <= 1).all()

    def test_map_susie_no_phase_runs(self):
        """Without phase, SuSiE fine-maps the total channel alone."""
        d = _make_dataset(seed=122, n_samples=100, n_variants=15)
        summary_df = map_susie(
            d['genotype_df'], d['variant_df'],
            d['A_df'], d['T_df'], d['Va_df'], d['Vt_df'],
            d['pos_df'], xL_df=None, xR_df=None,
            L=5, window=1000000, max_iter=200,
            summary_only=True, verbose=False,
        )
        # Still recovers the causal variant from the total channel.
        sub = summary_df[summary_df['phenotype_id'] == d['causal_pheno']]
        assert d['causal_variant'] in set(sub['variant_id'])


# ---------------------------------------------------------------------------
#  I/O round-trip
# ---------------------------------------------------------------------------

class TestIO:

    def test_read_hapmixqtl_inputs_roundtrip(self, temp_dir):
        """Writing BED files and reading them back reproduces the matrices."""
        d = _make_dataset(seed=105)

        def _write_bed(df, pos_df, path):
            bed = pos_df.copy()
            bed.columns = ['#chr', 'end']  # pos_df has chr, pos
            bed.insert(1, 'start', bed['end'] - 1)
            bed.insert(3, 'pid', df.index)
            out = pd.concat([bed.reset_index(drop=True),
                             df.reset_index(drop=True)], axis=1)
            out = out.rename(columns={'pid': 'gene_id'})
            out.to_csv(path, sep='\t', index=False)

        paths = {}
        for name, mat in [('A', d['A_df']), ('T', d['T_df']),
                          ('Va', d['Va_df']), ('Vt', d['Vt_df'])]:
            p = Path(temp_dir) / f'{name}.bed'
            _write_bed(mat, d['pos_df'], p)
            paths[name] = str(p)

        A_df, T_df, Va_df, Vt_df, Cat_df, pos_df = read_hapmixqtl_inputs(
            paths['A'], paths['T'], paths['Va'], paths['Vt']
        )
        assert A_df.index.equals(d['A_df'].index)
        assert A_df.columns.equals(d['A_df'].columns)
        np.testing.assert_allclose(A_df.values, d['A_df'].values, atol=1e-5)
        np.testing.assert_allclose(Vt_df.values, d['Vt_df'].values, atol=1e-5)
        assert Cat_df is None


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))

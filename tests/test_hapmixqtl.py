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
    fine_mapping_provenance,
    cis_trans_diagnostic,
    _prepare_channels,
    _estimate_tau_informative,
    _permute_within_informative,
    _leverage_standardized,
    SAME_COVARIATES,
    orient_haplotypes,
    reference_bias_diagnostic,
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

    def test_through_origin_without_covariates_is_identity(self, device):
        """An ASE through-origin design has no automatic projection column."""
        N = 23
        rng = _make_gaussian_seed(101)
        sqrt_w = torch.tensor(rng.uniform(0.2, 2.0, N), dtype=torch.float64,
                              device=device)
        M = torch.tensor(rng.normal(size=(4, N)), dtype=torch.float64, device=device)
        res = WeightedResidualizer(None, sqrt_w, intercept=False)
        assert res.Q_t.shape == (N, 0)
        assert res.dof == N - 1
        assert torch.equal(res.transform(M), M)

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

    def test_through_origin_matches_raw_weighted_formula(self, device):
        """The ASE no-covariate path is exactly sum(w*s*a)/sum(w*s^2)."""
        rng = _make_gaussian_seed(102)
        N = 41
        s = rng.choice([-1.0, 0.0, 1.0], N)
        a = 0.8 + 1.4 * s + rng.normal(0, 0.2, N)
        w = rng.uniform(0.2, 3.0, N)
        sw = np.sqrt(w)
        res = WeightedResidualizer(None,
                                   torch.tensor(sw, dtype=torch.float64, device=device),
                                   intercept=False)
        slope, slope_se = _wls_regression(
            torch.tensor((a * sw)[None, :], dtype=torch.float64, device=device),
            torch.tensor((s * sw)[None, :], dtype=torch.float64, device=device), res)
        xx = np.sum(w * s * s)
        assert np.isclose(slope.item(), np.sum(w * s * a) / xx, atol=1e-10)
        assert np.isclose(slope_se.item(), 1 / np.sqrt(xx), atol=1e-10)

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

        # count_noise defaults True; this test checks the raw draw summaries,
        # so it asks for them explicitly. The counting term is covered below.
        A, T, Va, Vt, Cat = compute_summaries_from_gibbs(yL, yR, kappa=kappa,
                                                         count_noise=False)

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

    def test_counting_noise_is_on_by_default(self):
        """The default gained the Poisson counting term on 2026-09-13: without
        it a sample whose draws are unanimous has zero variance and the largest
        weight in the gene, which is what drove the total channel's type-I error
        to 52% at alpha = 0.05."""
        yL = np.full((1, 3, 5), 8.0)            # unanimous draws: raw v_inf == 0
        yR = np.full((1, 3, 5), 8.0)
        yT = np.full((1, 3, 5), 40.0)
        _, _, Va_d, Vt_d, _ = compute_summaries_from_gibbs(yL, yR, yT=yT)
        _, _, Va_raw, Vt_raw, _ = compute_summaries_from_gibbs(yL, yR, yT=yT,
                                                               count_noise=False)
        assert (Va_raw == 0).all() and (Vt_raw == 0).all()
        assert (Va_d > 0).all() and (Vt_d > 0).all()
        assert np.allclose(Va_d, 2 / (8 + 0.5))
        assert np.allclose(Vt_d, 1 / (40 + 1.0))
        # a sample with NO allele-specific reads still gets Va = 0, so the
        # degenerate-ASE guard keeps excluding it
        z = np.zeros((1, 1, 5))
        _, _, Va_z, _, _ = compute_summaries_from_gibbs(z, z, yT=np.full((1, 1, 5), 12.0))
        assert (Va_z == 0).all()

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
    res_a = WeightedResidualizer(None, sqrt_wa_t, intercept=False)
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

    def test_map_susie_purity_uses_genotype_ld(self):
        """
        Credible-set purity must be measured on genotype LD, not on the
        stacked/whitened design SuSiE is fit on. Build two variants in near
        perfect genotype LD (one causal); a single credible set covering both
        must be reported as pure (they are genuinely indistinguishable), which
        only holds if purity is computed from the dosage correlation.
        """
        d = _make_dataset(seed=130, n_samples=150, n_variants=12)
        g = d['genotype_df']
        # Make variant 1 an almost-perfect copy of the causal variant 0.
        g.iloc[1, :] = g.iloc[0, :]
        g.iloc[1, 0] = 2 if g.iloc[0, 0] != 2 else 0  # break perfect identity slightly
        # Mirror the phase so the sign indicator is consistent for variant 1.
        d['xL_df'].iloc[1, :] = d['xL_df'].iloc[0, :]
        d['xR_df'].iloc[1, :] = d['xR_df'].iloc[0, :]

        summary_df = map_susie(
            g, d['variant_df'],
            d['A_df'], d['T_df'], d['Va_df'], d['Vt_df'],
            d['pos_df'], xL_df=d['xL_df'], xR_df=d['xR_df'],
            L=5, min_abs_corr=0.5, window=1000000, max_iter=200,
            summary_only=True, verbose=False,
        )
        sub = summary_df[summary_df['phenotype_id'] == d['causal_pheno']]
        # The causal variant is in a reported (pure) credible set.
        assert d['causal_variant'] in set(sub['variant_id'])


    def test_map_susie_records_tau_mode_provenance(self):
        """Fine-mapping output records the tau_mode it was produced under, and
        fine_mapping_provenance flags results from the old default (sec 7g)."""
        d = _make_dataset(seed=123, n_samples=100, n_variants=15)
        summary_df, res = map_susie(
            d['genotype_df'], d['variant_df'],
            d['A_df'], d['T_df'], d['Va_df'], d['Vt_df'],
            d['pos_df'], xL_df=d['xL_df'], xR_df=d['xR_df'],
            L=5, window=1000000, max_iter=200,
            summary_only=False, verbose=False,
        )
        assert 'tau_mode' in summary_df.columns
        assert (summary_df['tau_mode'] == 'estimate').all()
        assert all(v['tau_mode'] == 'estimate' for v in res.values())
        assert fine_mapping_provenance(summary_df)['status'] == 'ok'
        assert fine_mapping_provenance(summary_df.drop(columns='tau_mode'))['status'] == 'unknown'
        stale = summary_df.copy()
        stale['tau_mode'] = 'zero'
        assert fine_mapping_provenance(stale)['status'] == 'stale'


# ---------------------------------------------------------------------------
#  cis/trans diagnostic (sec 7c)
# ---------------------------------------------------------------------------

class TestCisTransDiagnostic:

    def test_wald_formula_and_nan_rules(self):
        a, p = cis_trans_diagnostic([1.0, 1.0, 1.0], [0.1, np.inf, 0.1],
                                    [1.0, 1.0, 0.0], [0.1, 0.1, 0.1], dof=100)
        assert np.isclose(a[0], 1.0) and np.isclose(p[0], 1.0)      # identical channels
        assert np.isnan(a[1]) and np.isnan(p[1])                      # no ASE channel
        assert np.isnan(a[2]) and p[2] < 1e-8                         # slope_t = 0: alpha undefined, test fires

    def test_pure_cis_passes_and_trans_only_is_flagged(self):
        """A planted cis effect gives alpha ~ 1 and no flag; a planted effect on
        total expression only (a trans-like effect) gives alpha ~ 0 and a flag,
        with the combined slope attenuated exactly as sec 7c predicts."""
        n = 150
        d = _make_dataset(seed=140, n_samples=n, n_variants=20)
        rng = np.random.RandomState(7)
        trans_pheno, trans_var = d['A_df'].index[1], d['genotype_df'].index[5]
        g5 = d['genotype_df'].loc[trans_var].values
        d['T_df'].loc[trans_pheno] = (2.0 + 1.0 * (g5 / 2) + rng.normal(0, 0.1, n)).astype(np.float32)
        d['A_df'].loc[trans_pheno] = rng.normal(0, 0.1, n).astype(np.float32)
        res = map_cis(
            d['genotype_df'], d['variant_df'],
            d['A_df'], d['T_df'], d['Va_df'], d['Vt_df'],
            d['pos_df'], xL_df=d['xL_df'], xR_df=d['xR_df'],
            window=1000000, nperm=100, verbose=False,
        )
        for col in ('slope_a', 'slope_a_se', 'slope_t', 'slope_t_se', 'alpha_cis', 'pval_cis_trans'):
            assert col in res.columns
        cis = res.loc[d['causal_pheno']]
        assert cis['variant_id'] == d['causal_variant']
        assert abs(cis['alpha_cis'] - 1.0) < 0.25, cis['alpha_cis']
        assert cis['pval_cis_trans'] > 0.01, cis['pval_cis_trans']
        tr = res.loc[trans_pheno]
        assert tr['variant_id'] == trans_var
        assert abs(tr['alpha_cis']) < 0.25, tr['alpha_cis']
        assert tr['pval_cis_trans'] < 1e-4, tr['pval_cis_trans']
        # the combined slope is attenuated relative to the true total effect of 1.0
        assert tr['slope'] < 0.5 * tr['slope_t']

    def test_no_phase_gives_nan_diagnostic(self):
        d = _make_dataset(seed=141, n_samples=100, n_variants=15)
        res = map_cis(
            d['genotype_df'], d['variant_df'],
            d['A_df'], d['T_df'], d['Va_df'], d['Vt_df'],
            d['pos_df'], xL_df=None, xR_df=None,
            window=1000000, nperm=50, verbose=False,
        )
        assert res['pval_cis_trans'].isna().all() and res['alpha_cis'].isna().all()

    def test_map_nominal_carries_pval_cis_trans(self, temp_dir):
        d = _make_dataset(seed=142, n_samples=100, n_variants=15)
        map_nominal(
            d['genotype_df'], d['variant_df'],
            d['A_df'], d['T_df'], d['Va_df'], d['Vt_df'],
            d['pos_df'], xL_df=d['xL_df'], xR_df=d['xR_df'],
            prefix='ct', output_dir=temp_dir, verbose=False,
        )
        df = pd.read_parquet(Path(temp_dir) / 'ct.hapmixqtl_pairs.chr1.parquet')
        assert 'pval_cis_trans' in df.columns
        row = df[(df['phenotype_id'] == d['causal_pheno']) & (df['variant_id'] == d['causal_variant'])].iloc[0]
        assert row['pval_cis_trans'] > 0.01
        assert df['pval_cis_trans'].between(0, 1).all()


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


# ---------------------------------------------------------------------------
#  Per-channel covariates and the sparse-channel rule
# ---------------------------------------------------------------------------

def _gene_with_covariates(seed, N=60, V=8, n_cov=5, n_off=0, sigma_bio=0.5):
    """One null gene: heteroskedastic v, covariates that act on both haplotypes
    alike (they enter t, not the within-sample contrast a), and n_off samples
    with no allele-specific coverage (a = 0, va = 0 exactly)."""
    rng = np.random.RandomState(seed)
    g = rng.binomial(2, 0.4, size=(V, N)).astype(float)
    sign = np.zeros((V, N))
    het = g == 1
    sign[het] = rng.choice([-1.0, 1.0], size=int(het.sum()))
    va = rng.uniform(0.05, 0.3, N)
    vt = rng.uniform(0.05, 0.3, N)
    C = rng.normal(size=(N, n_cov))
    a = np.sqrt(va) * rng.normal(size=N) + rng.normal(0, sigma_bio, N)
    t = (2.0 + C @ rng.normal(size=n_cov) + np.sqrt(vt) * rng.normal(size=N)
         + rng.normal(0, sigma_bio, N))
    if n_off:
        k = rng.choice(N, n_off, replace=False)
        a[k] = 0.0
        va[k] = 0.0
    return g, sign, a, t, va, vt, C


def _nominal(g, s, a, t, va, vt, C, device, ase='same', tau_mode='estimate'):
    T = lambda x: torch.tensor(x, dtype=torch.float64, device=device)
    C_t = None if C is None else T(C)
    ase_t = ase if isinstance(ase, str) else (None if ase is None else T(ase))
    wa, wt, ra, rt = _prepare_channels(T(a), T(t), T(va), T(vt), C_t, tau_mode,
                                       device, ase_covariates_t=ase_t)
    res = calculate_hapmixqtl_nominal(T(g), T(s), T(a), T(t), wa, wt, ra, rt)
    return dict(zip(['tstat', 'slope', 'se', 'slope_a', 'se_a', 'slope_t', 'se_t'], res),
                wa=wa, wt=wt, ra=ra, rt=rt)


class TestPerChannelCovariates:

    def test_total_channel_retains_automatic_intercept(self, device):
        """Changing ASE to through-origin does not change the total WLS fit."""
        rng = _make_gaussian_seed(103)
        N = 47
        g = rng.choice([0.0, 1.0, 2.0], N)
        t = 1.3 + 0.7 * (g / 2) + rng.normal(0, 0.15, N)
        a = rng.normal(0, 0.2, N)
        s = rng.choice([-1.0, 0.0, 1.0], N)
        va = rng.uniform(0.1, 0.3, N)
        vt = rng.uniform(0.1, 0.3, N)
        T = lambda x: torch.tensor(x, dtype=torch.float64, device=device)
        wa, wt, ra, rt = _prepare_channels(T(a), T(t), T(va), T(vt), None,
                                           'zero', device, ase_covariates_t=None)
        out = calculate_hapmixqtl_nominal(T(g[None, :]), T(s[None, :]), T(a), T(t),
                                           wa, wt, ra, rt)
        beta, se = _wls_reference(t, np.column_stack([np.ones(N), g / 2]), 1 / vt)
        assert rt.Q_t.shape[1] == 1
        assert np.isclose(float(out[5][0]), beta[1], atol=1e-10)
        assert np.isclose(float(out[6][0]), se[1], atol=1e-10)

    def test_donor_label_swaps_preserve_ase_tau_and_observed_fits(self, device):
        """Swapping an arbitrary donor subset leaves through-origin ASE fits invariant.

        This checks observed null and lead-refit tau plus the ASE, total, and
        combined nominal fits. It intentionally makes no assertion about a
        finite-seed permutation p-value.
        """
        rng = _make_gaussian_seed(104)
        N = 37
        g = rng.choice([0.0, 1.0, 2.0], N).astype(float)
        s = np.zeros(N)
        het = g == 1.0
        s[het] = rng.choice([-1.0, 1.0], int(het.sum()))
        a = 0.9 * s + rng.normal(0, 0.25, N)
        t = 1.1 + 0.9 * (g / 2) + rng.normal(0, 0.2, N)
        va = rng.uniform(0.05, 0.25, N)
        vt = rng.uniform(0.05, 0.25, N)
        flip = rng.rand(N) < 0.45
        T = lambda x: torch.tensor(x, dtype=torch.float64, device=device)

        def fit(a0, s0, lead):
            args = dict(ase_covariates_t=None, return_info=True)
            if lead:
                args.update(tau_extra_a_t=T(s0[:, None]),
                            tau_extra_t_t=T((g / 2)[:, None]))
            wa, wt, ra, rt, info = _prepare_channels(T(a0), T(t), T(va), T(vt),
                                                      None, 'estimate', device, **args)
            nominal = calculate_hapmixqtl_nominal(
                T(g[None, :]), T(s0[None, :]), T(a0), T(t), wa, wt, ra, rt)
            return info, nominal

        a_swap, s_swap = a.copy(), s.copy()
        a_swap[flip] *= -1
        s_swap[flip] *= -1
        for lead in (False, True):
            info, nominal = fit(a, s, lead)
            info_swap, nominal_swap = fit(a_swap, s_swap, lead)
            assert np.isclose(info['tau_a'], info_swap['tau_a'], rtol=1e-10, atol=1e-12)
            assert np.isclose(info['tau_t'], info_swap['tau_t'], rtol=1e-10, atol=1e-12)
            for got, expected in zip(nominal, nominal_swap):
                assert torch.allclose(got, expected, rtol=1e-10, atol=1e-10)

    def test_through_origin_allelic_channel_ignores_total_covariates(self, device):
        """ase_covariates_t=None: the allelic slope and SE are those of the
        no-covariate fit, the total channel's are those of the covariate fit,
        and the allelic residualizer is through-origin."""
        g, s, a, t, va, vt, C = _gene_with_covariates(1)
        split = _nominal(g, s, a, t, va, vt, C, device, ase=None)
        none = _nominal(g, s, a, t, va, vt, None, device)
        shared = _nominal(g, s, a, t, va, vt, C, device)
        assert torch.allclose(split['slope_a'], none['slope_a'])
        assert torch.allclose(split['se_a'], none['se_a'])
        assert torch.allclose(split['slope_t'], shared['slope_t'])
        assert torch.allclose(split['se_t'], shared['se_t'])
        assert split['ra'].Q_t.shape[1] == 0
        assert split['rt'].Q_t.shape[1] == 1 + C.shape[1]
        assert split['ra'].dof == len(a) - 1
        assert split['rt'].dof == len(a) - 2 - C.shape[1]
        # Explicit shared covariates retain the same ASE design in both calls.
        default = _nominal(g, s, a, t, va, vt, C, device, ase=SAME_COVARIATES)
        assert torch.allclose(default['se_a'], shared['se_a'])

    def test_projecting_covariates_out_of_the_allelic_channel_only_loses_precision(self, device):
        """At fixed weights the through-origin design is nested in the shared
        one, so residualizing the covariates as well can only shrink the
        predictor's residual norm: the known-variance SE of the allelic slope
        is never smaller with them than without (CCNI: 17 -> 9 with 17
        covariates on ~50 informative samples)."""
        g, s, a, t, va, vt, C = _gene_with_covariates(2, N=50, n_cov=12, n_off=5)
        split = _nominal(g, s, a, t, va, vt, C, device, ase=None)
        wa = split['wa']
        T = lambda x: torch.tensor(x, dtype=torch.float64, device=device)
        a_star = (T(a) * wa).unsqueeze(0)
        s_star = T(s) * wa.unsqueeze(0)
        _, se_int = _wls_regression(a_star, s_star, WeightedResidualizer(None, wa, intercept=False))
        _, se_cov = _wls_regression(a_star, s_star,
                                    WeightedResidualizer(T(C), wa, intercept=False))
        ok = torch.isfinite(se_int) & torch.isfinite(se_cov)
        assert ok.any()
        assert (se_cov[ok] >= se_int[ok] * (1 - 1e-9)).all()
        assert (se_cov[ok] > se_int[ok]).any()

    def test_map_cis_and_map_nominal_share_the_dof_rule(self, tmp_path):
        """With a through-origin allelic channel the two residualizers have
        different dof; the nominal p-value must use one rule in both mapping
        functions (N - 2 - max(n_cov, n_cov_a)), so map_cis's pval_nominal at
        the lead equals map_nominal's for that pair."""
        d = _make_dataset(seed=105, n_samples=60)
        rng = np.random.RandomState(9)
        cov_df = pd.DataFrame(rng.normal(size=(60, 4)), index=d['A_df'].columns,
                              columns=[f'c{i}' for i in range(4)])
        common = dict(covariates_df=cov_df, ase_covariates_df=None,
                      window=1000000, verbose=False)
        cis = map_cis(d['genotype_df'], d['variant_df'], d['A_df'], d['T_df'],
                      d['Va_df'], d['Vt_df'], d['pos_df'], xL_df=d['xL_df'],
                      xR_df=d['xR_df'], nperm=200, seed=1, **common)
        map_nominal(d['genotype_df'], d['variant_df'], d['A_df'], d['T_df'],
                    d['Va_df'], d['Vt_df'], d['pos_df'], xL_df=d['xL_df'],
                    xR_df=d['xR_df'], prefix='t', output_dir=str(tmp_path), **common)
        pairs = pd.read_parquet(tmp_path / 't.hapmixqtl_pairs.chr1.parquet')
        for pid, row in cis.iterrows():
            pair = pairs[(pairs['phenotype_id'] == pid) & (pairs['variant_id'] == row['variant_id'])]
            assert len(pair) == 1
            pn, pc = float(pair['pval_nominal'].iloc[0]), float(row['pval_nominal'])
            assert np.isclose(pn, pc, rtol=1e-4, atol=0), (pid, pn, pc)
            assert np.isclose(float(pair['slope'].iloc[0]), float(row['slope']), rtol=1e-4, atol=0)


class TestSparseChannel:

    def test_tau_estimator_refuses_too_few_informative_samples(self, device):
        """Zero-variance rows never reach the estimator, and there is no
        fallback that re-admits them: below design columns + 2 informative
        samples it raises."""
        N, n_cov = 40, 5
        g, s, a, t, va, vt, C = _gene_with_covariates(3, N=N, n_cov=n_cov)
        T = lambda x: torch.tensor(x, dtype=torch.float64, device=device)
        for n_inf, ok in ((7, False), (8, True)):
            v = va.copy(); v[n_inf:] = 0.0
            if ok:
                tau = _estimate_tau_informative(T(a), T(v), T(C), device)
                assert torch.isfinite(tau) and tau >= 0
            else:
                with pytest.raises(ValueError, match='informative samples'):
                    _estimate_tau_informative(T(a), T(v), T(C), device)

    def test_switched_off_channel_yields_total_only(self, device):
        """4 informative allelic samples against 5 covariates: the allelic
        channel is off (every weight zero, nothing projected, SE infinite)
        and the combined statistic is the total channel's. Through-origin,
        the same 4 samples keep the channel on."""
        N = 60
        g, s, a, t, va, vt, C = _gene_with_covariates(4, N=N, n_cov=5, n_off=N - 4)
        off = _nominal(g, s, a, t, va, vt, C, device)
        assert (off['wa'] == 0).all()
        assert off['ra'].Q_t.shape == (N, 0)
        assert torch.isinf(off['se_a']).all()
        total_only = _nominal(g, np.zeros_like(s), a, t, va, vt, C, device)
        assert torch.allclose(off['tstat'], total_only['tstat'])
        assert torch.allclose(off['slope'], total_only['slope_t'])
        assert torch.allclose(off['se'], total_only['se_t'])
        on = _nominal(g, s, a, t, va, vt, C, device, ase=None)
        assert int((on['wa'] > 0).sum()) == 4
        assert torch.isfinite(on['se_a']).any()

    def test_zero_weight_residualizer_is_the_identity(self, device):
        N = 12
        w = torch.zeros(N, dtype=torch.float64, device=device)
        C = torch.randn(N, 3, dtype=torch.float64, device=device)
        M = torch.randn(4, N, dtype=torch.float64, device=device)
        res = WeightedResidualizer(C, w)
        assert res.Q_t.shape == (N, 0)
        assert torch.equal(res.transform(M), M)
        assert res.dof == N - 1 - 4

    def test_map_cis_runs_with_a_switched_off_allelic_channel(self):
        """A phenotype with one allele-specific sample goes through map_cis
        on its total channel alone: finite p-values, infinite allelic SE, and
        the planted association in the other phenotype is still found."""
        d = _make_dataset(seed=106, n_samples=60)
        sparse = d['A_df'].index[1]
        A, Va = d['A_df'].copy(), d['Va_df'].copy()
        A.loc[sparse, A.columns[1:]] = 0.0
        Va.loc[sparse, Va.columns[1:]] = 0.0
        rng = np.random.RandomState(8)
        cov_df = pd.DataFrame(rng.normal(size=(60, 3)), index=A.columns, columns=list('xyz'))
        res = map_cis(d['genotype_df'], d['variant_df'], A, d['T_df'], Va, d['Vt_df'],
                      d['pos_df'], xL_df=d['xL_df'], xR_df=d['xR_df'], nperm=200,
                      seed=3, covariates_df=cov_df, ase_covariates_df=None, verbose=False)
        row = res.loc[sparse]
        assert 0 < row['pval_nominal'] <= 1 and 0 < row['pval_perm'] <= 1
        assert np.isinf(row['slope_a_se'])
        assert np.isfinite(row['slope_t_se'])
        assert res.loc[d['causal_pheno'], 'variant_id'] == d['causal_variant']


class TestWhitenedResidualPermutation:

    def test_permutes_only_within_informative_samples(self, device):
        N, nperm = 10, 200
        rng = np.random.RandomState(0)
        r = torch.tensor(rng.normal(size=N), dtype=torch.float64, device=device)
        inf = torch.tensor([1, 1, 0, 1, 0, 1, 1, 0, 1, 1], dtype=torch.bool, device=device)
        perm = torch.tensor(np.array([rng.permutation(N) for _ in range(nperm)]),
                            dtype=torch.long, device=device)
        out = _permute_within_informative(r, inf, perm)
        assert out.shape == (nperm, N)
        # non-informative entries untouched
        assert torch.equal(out[:, ~inf], r[~inf].unsqueeze(0).expand(nperm, -1))
        # each row is a permutation of the informative entries
        ref = torch.sort(r[inf]).values
        for row in out:
            assert torch.allclose(torch.sort(row[inf]).values, ref)
        # most informative entries move
        moved = (out[:, inf] != r[inf].unsqueeze(0)).float().mean().item()
        assert moved > 0.7
        # with every sample informative this is the plain permutation
        all_inf = torch.ones(N, dtype=torch.bool, device=device)
        assert torch.equal(_permute_within_informative(r, all_inf, perm), r[perm])

    def test_map_cis_rejects_robust_se(self):
        d = _make_dataset(seed=107)
        with pytest.raises(ValueError, match='robust'):
            map_cis(d['genotype_df'], d['variant_df'], d['A_df'], d['T_df'],
                    d['Va_df'], d['Vt_df'], d['pos_df'], xL_df=d['xL_df'],
                    xR_df=d['xR_df'], nperm=10, se_mode='robust', verbose=False)


class TestOrientHaplotypes:

    def test_depth_weighted_sign_and_uniform_fallback(self):
        # sites x samples. Sample 0 is het at two sites with opposite phase;
        # site 0 carries 30 of its reads, site 1 five.
        sign = np.array([[+1, -1, 0], [-1, -1, 0]])
        depth = np.array([[30, 10, 0], [5, 10, 0]])
        assert orient_haplotypes(sign, depth).tolist() == [1, -1, 0]
        assert orient_haplotypes(sign).tolist() == [0, -1, 0]       # uniform: sample 0 cancels
        assert orient_haplotypes(np.zeros((0, 3))).tolist() == [0, 0, 0]
        assert orient_haplotypes(np.array([1, 0, -1])).tolist() == [1, 0, -1]

    def test_diagnostic_sees_planted_bias_only_through_the_right_orientation(self):
        """Reference bias planted at every site with reads: the depth-weighted
        orientation exposes it (pooled reference fraction above 0.5, flagged);
        orienting every gene by one unrelated site, as a row-index bug would,
        shows nothing."""
        rng = np.random.RandomState(0)
        G, N, S = 40, 60, 3
        yL = np.zeros((G, N)); yR = np.zeros((G, N))
        sign = np.zeros((G, S, N)); depth = np.zeros((G, S, N))
        for g in range(G):
            reads = np.array([60, 15, 4])          # per-site depth, site 0 deepest
            for v in range(S):
                het = rng.rand(N) < 0.6
                s = np.where(het, rng.choice([-1, 1], N), 0)
                n = rng.poisson(reads[v], N) * het
                ref = rng.binomial(n, 0.62)          # 62% of reads to REF: mapping bias
                alt = n - ref
                # ALT on L when s > 0
                yL[g] += np.where(s > 0, alt, ref) * het
                yR[g] += np.where(s > 0, ref, alt) * het
                sign[g, v] = s; depth[g, v] = n
        right = np.stack([orient_haplotypes(sign[g], depth[g]) for g in range(G)])
        d = reference_bias_diagnostic(yL, yR, right)
        assert d['flag'] and d['ref_fraction'] > 0.55, d['message']
        # one fixed, unrelated site per gene (the shallowest) -- the bias mass sits elsewhere
        wrong = sign[:, 2, :]
        d2 = reference_bias_diagnostic(yL, yR, wrong)
        assert d2['ref_fraction'] < d['ref_fraction'] - 0.05, (d2['ref_fraction'], d['ref_fraction'])


class TestPermutedNullScale:

    def test_permuted_null_matches_the_known_variance(self, device):
        """The observed xy = s_res . e has variance xx (s_res is orthogonal to
        the null design and e is whitened). Permuting the raw null residuals,
        whose variance is 1 - h_ii, gives a null short by about (N - p)/N;
        leverage-standardized residuals restore Var(xy_perm) = xx. With 18
        columns on 60 samples the deficit is 0.70, far outside permutation
        noise at 20,000 draws."""
        N, p, nperm = 60, 17, 20000
        rng = np.random.RandomState(4)
        C = torch.tensor(rng.normal(size=(N, p)), dtype=torch.float64, device=device)
        w = torch.tensor(rng.uniform(0.5, 2.0, N), dtype=torch.float64, device=device)
        res = WeightedResidualizer(C, w)
        # a null whitened response and one predictor
        e = res.transform(torch.tensor(rng.normal(size=(1, N)), dtype=torch.float64, device=device))[0]
        s = torch.tensor(rng.choice([-1.0, 0.0, 1.0], N), dtype=torch.float64, device=device) * w
        s_res = res.transform(s.unsqueeze(0))[0]
        xx = float((s_res * s_res).sum())
        perm = torch.tensor(np.array([rng.permutation(N) for _ in range(nperm)]),
                            dtype=torch.long, device=device)
        inf = torch.ones(N, dtype=torch.bool, device=device)
        # average over several null draws of e so the ratio is not one residual's luck
        raw, std = [], []
        for _ in range(8):
            e = res.transform(torch.tensor(rng.normal(size=(1, N)), dtype=torch.float64, device=device))[0]
            xy_raw = _permute_within_informative(e, inf, perm) @ s_res
            xy_std = _permute_within_informative(_leverage_standardized(e, res), inf, perm) @ s_res
            raw.append(float(xy_raw.var()) / xx)
            std.append(float(xy_std.var()) / xx)
        raw, std = float(np.mean(raw)), float(np.mean(std))
        expected_deficit = (N - (p + 1)) / N                    # 0.70
        assert abs(raw - expected_deficit) < 0.08, (raw, expected_deficit)
        assert abs(std - 1.0) < 0.08, std


def _heteroskedastic_dataset(seed, n_samples=80, beta=0.6, tau=0.05):
    """_make_dataset with a planted effect on phenotype 0, heteroskedastic
    inferential variances and between-sample variance tau, so tau estimated
    under the null absorbs the planted signal."""
    d = _make_dataset(seed=seed, n_samples=n_samples)
    rng = np.random.RandomState(seed + 1)
    N = n_samples
    xL, xR, g = d['xL_df'].values, d['xR_df'].values, d['genotype_df'].values
    for k, pid in enumerate(d['A_df'].index):
        va = rng.uniform(0.01, 0.2, N); vt = rng.uniform(0.01, 0.2, N)
        a = np.sqrt(va + tau) * rng.normal(size=N); t = 2.0 + np.sqrt(vt + tau) * rng.normal(size=N)
        if k == 0:
            a = a + beta * (xL[0] - xR[0]); t = t + beta * g[0] / 2
        d['A_df'].loc[pid] = a.astype(np.float32); d['T_df'].loc[pid] = t.astype(np.float32)
        d['Va_df'].loc[pid] = va.astype(np.float32); d['Vt_df'].loc[pid] = vt.astype(np.float32)
    return d


class TestLeadRefit:

    def _run(self, d, refit, cov_df=None):
        return map_cis(d['genotype_df'], d['variant_df'], d['A_df'], d['T_df'], d['Va_df'],
                       d['Vt_df'], d['pos_df'], xL_df=d['xL_df'], xR_df=d['xR_df'], nperm=300,
                       seed=5, covariates_df=cov_df, ase_covariates_df=None, verbose=False,
                       tau_refit=refit)

    def test_refit_changes_only_the_reported_scale(self):
        """Same lead, same pval_perm and pval_beta to the bit (the scan is
        untouched); on the planted gene tau falls and pval_nominal shrinks;
        the tau columns record both estimates."""
        d = _heteroskedastic_dataset(seed=201)
        rng = np.random.RandomState(3)
        cov_df = pd.DataFrame(rng.normal(size=(80, 4)), index=d['A_df'].columns, columns=list('wxyz'))
        off = self._run(d, False, cov_df); on = self._run(d, True, cov_df)
        assert (off['variant_id'] == on['variant_id']).all()
        assert np.array_equal(off['pval_perm'].values.astype(float), on['pval_perm'].values.astype(float))
        assert np.array_equal(off['pval_beta'].values.astype(float), on['pval_beta'].values.astype(float))
        assert (~off['tau_refit'].astype(bool)).all() and on['tau_refit'].astype(bool).all()
        assert np.allclose(off['tau_a'].astype(float), off['tau_a_null'].astype(float))
        assert np.allclose(on['tau_a_null'].astype(float), off['tau_a_null'].astype(float))
        pid = d['causal_pheno']
        assert on.loc[pid, 'tau_a'] < on.loc[pid, 'tau_a_null']
        assert on.loc[pid, 'tau_t'] < on.loc[pid, 'tau_t_null']
        assert on.loc[pid, 'pval_nominal'] < off.loc[pid, 'pval_nominal'] / 10
        assert on.loc[pid, 'slope_se'] < off.loc[pid, 'slope_se']
        # the null phenotypes move little: tau within 30% and p within a factor of 3
        for q in d['A_df'].index[1:]:
            assert abs(on.loc[q, 'tau_a'] - off.loc[q, 'tau_a']) <= 0.3 * off.loc[q, 'tau_a'] + 1e-3
            assert on.loc[q, 'pval_nominal'] > off.loc[q, 'pval_nominal'] / 3

    def test_refit_matches_a_manual_refit(self, device):
        """_estimate_tau with [covariates, lead column] -> weights -> the null
        residualizer -> _wls_regression reproduces map_cis's refit slope and
        SE at the lead in both channels."""
        d = _heteroskedastic_dataset(seed=202)
        rng = np.random.RandomState(4)
        cov_df = pd.DataFrame(rng.normal(size=(80, 3)), index=d['A_df'].columns, columns=list('xyz'))
        on = self._run(d, True, cov_df)
        pid = d['causal_pheno']; lead = on.loc[pid, 'variant_id']
        T = lambda x: torch.tensor(np.asarray(x, dtype=np.float32), dtype=torch.float32)
        a, t = T(d['A_df'].loc[pid]), T(d['T_df'].loc[pid])
        va, vt = T(d['Va_df'].loc[pid]), T(d['Vt_df'].loc[pid])
        s = T(d['xL_df'].loc[lead].values - d['xR_df'].loc[lead].values)
        g2 = T(d['genotype_df'].loc[lead].values / 2); C = T(cov_df.values)
        ka = va > 1e-12
        tau_a = _estimate_tau(a[ka], va[ka], s[ka].unsqueeze(1), 'cpu', intercept=False)
        tau_t = _estimate_tau(t, vt, torch.cat([C, g2.unsqueeze(1)], 1), 'cpu')
        wa = torch.sqrt(1.0 / (va.clamp(min=1e-8) + tau_a)); wt = torch.sqrt(1.0 / (vt.clamp(min=1e-8) + tau_t))
        sl_a, se_a = _wls_regression((a * wa).unsqueeze(0), (s * wa).unsqueeze(0),
                                      WeightedResidualizer(None, wa, intercept=False))
        sl_t, se_t = _wls_regression((t * wt).unsqueeze(0), (g2 * wt).unsqueeze(0), WeightedResidualizer(C, wt))
        assert np.isclose(float(tau_a), on.loc[pid, 'tau_a'], rtol=1e-4)
        assert np.isclose(float(tau_t), on.loc[pid, 'tau_t'], rtol=1e-4)
        assert np.isclose(float(sl_a[0]), on.loc[pid, 'slope_a'], rtol=1e-4)
        assert np.isclose(float(se_a[0]), on.loc[pid, 'slope_a_se'], rtol=1e-4)
        assert np.isclose(float(sl_t[0]), on.loc[pid, 'slope_t'], rtol=1e-4)
        assert np.isclose(float(se_t[0]), on.loc[pid, 'slope_t_se'], rtol=1e-4)
        ia, it = 1 / float(se_a[0]) ** 2, 1 / float(se_t[0]) ** 2
        comb = (float(sl_a[0]) * ia + float(sl_t[0]) * it) / (ia + it)
        assert np.isclose(comb, on.loc[pid, 'slope'], rtol=1e-4)
        assert np.isclose(1 / np.sqrt(ia + it), on.loc[pid, 'slope_se'], rtol=1e-4)

    def test_refit_keeps_a_barely_identifiable_channel_on_the_null_tau(self):
        """An allelic channel with exactly design columns + 2 informative
        samples is on for the scan; the refit's extra column would need one
        more, so that channel keeps its null tau and the flag is False."""
        d = _heteroskedastic_dataset(seed=203)
        A, Va = d['A_df'].copy(), d['Va_df'].copy()
        sparse = A.index[1]
        A.loc[sparse, A.columns[2:]] = 0.0; Va.loc[sparse, Va.columns[2:]] = 0.0    # 2 informative = through-origin + 2
        d['A_df'], d['Va_df'] = A, Va
        on = self._run(d, True)
        row = on.loc[sparse]
        assert np.isclose(row['tau_a'], row['tau_a_null'])      # allelic: null tau kept
        assert row['tau_refit']                                  # the total channel was refit
        assert row['tau_t'] != row['tau_t_null']
        assert np.isfinite(row['pval_nominal']) and 0 < row['pval_perm'] <= 1


class TestTauDenominator:

    def test_intercept_only_tau_is_exactly_dersimonian_laird(self, device):
        """With an intercept alone the leverage is h_i = w_i / sum_j w_j, so the
        moment estimator's denominator sum_i w_i(1-h_i) equals DerSimonian and
        Laird's sum w - sum w^2 / sum w. The estimator must match that closed
        form exactly. Dividing by (n-q)*mean(w) instead, as an earlier version
        did, agrees only under equal weights."""
        rng = np.random.RandomState(5)
        n = 60
        v = rng.uniform(0.02, 2.0, n)                       # 100x spread in precision
        y = rng.normal(0, np.sqrt(v + 0.4), n)              # true tau = 0.4
        y_t = torch.tensor(y, dtype=torch.float64, device=device)
        v_t = torch.tensor(v, dtype=torch.float64, device=device)
        tau = float(_estimate_tau(y_t, v_t, None, device))

        w = 1.0 / v
        sw = np.sqrt(w); q = sw / np.linalg.norm(sw)
        rss = float(((y * sw) - q * (q @ (y * sw))) @ ((y * sw) - q * (q @ (y * sw))))
        dl_denom = w.sum() - (w ** 2).sum() / w.sum()
        assert np.isclose(dl_denom, (w * (1 - q ** 2)).sum())      # the identity itself
        assert np.isclose(tau, max(0.0, (rss - (n - 1)) / dl_denom), rtol=1e-6)

        old = max(0.0, (rss / (n - 1) - 1.0) / w.mean())           # the superseded form
        assert not np.isclose(tau, old, rtol=1e-3), (tau, old)

    def test_equal_weights_make_the_two_denominators_agree(self, device):
        """The superseded form was not wrong everywhere: under equal weights the
        two denominators coincide, which is why the error stayed small."""
        rng = np.random.RandomState(6)
        n = 60
        v = np.full(n, 0.25)
        y = rng.normal(0, np.sqrt(v[0] + 0.4), n)
        tau = float(_estimate_tau(torch.tensor(y, dtype=torch.float64, device=device),
                                  torch.tensor(v, dtype=torch.float64, device=device), None, device))
        w = 1.0 / v
        sw = np.sqrt(w); q = sw / np.linalg.norm(sw)
        r = (y * sw) - q * (q @ (y * sw)); rss = float(r @ r)
        old = max(0.0, (rss / (n - 1) - 1.0) / w.mean())
        assert np.isclose(tau, old, rtol=1e-6)


# ---------------------------------------------------------------------------
#  Variance models: additive, two-component, library-scaled
# ---------------------------------------------------------------------------

class TestVarianceModels:

    @staticmethod
    def _simulate_channel(rng, N, c, tau, d=None):
        v = 10 ** rng.uniform(-3, -1, N)
        d = np.ones(N) if d is None else d
        y = rng.normal(0, np.sqrt(d * (c * v + tau)), N)
        return y, v

    def test_c_tau_recovers_simulated_values(self, device):
        """The joint (c, tau) fit recovers a planted two-component variance
        on a through-origin channel, with and without a per-sample factor."""
        rng = _make_gaussian_seed(300)
        N, c, tau = 4000, 3.0, 0.01
        y, v = self._simulate_channel(rng, N, c, tau)
        T = lambda x: torch.tensor(x, dtype=torch.float64, device=device)
        fit = hapmixqtl._estimate_c_tau(T(y), T(v), None, device, intercept=False)
        c_hat, tau_hat, conv = fit['c'], fit['tau'], fit['converged']
        assert conv
        assert abs(c_hat - c) < 0.4, c_hat
        assert abs(tau_hat - tau) < 0.003, tau_hat
        d = np.exp(rng.normal(0, 0.3, N))
        y2, v2 = self._simulate_channel(rng, N, c, tau, d)
        fit2 = hapmixqtl._estimate_c_tau(T(y2), T(v2), None, device, intercept=False, d_t=T(d))
        c2, tau2, conv2 = fit2['c'], fit2['tau'], fit2['converged']
        assert conv2
        assert abs(c2 - c) < 0.4, c2
        assert abs(tau2 - tau) < 0.003, tau2
        # with the factor in the weights, the standardized squared residuals
        # carry no per-sample structure; without it they track d
        base = c2 * v2 + tau2
        e2_with = y2 ** 2 / (d * base)
        fit3 = hapmixqtl._estimate_c_tau(T(y2), T(v2), None, device, intercept=False)
        c3, tau3 = fit3['c'], fit3['tau']
        e2_without = y2 ** 2 / (c3 * v2 + tau3)
        lo, hi = d <= np.quantile(d, 0.2), d >= np.quantile(d, 0.8)
        ratio_with = e2_with[hi].mean() / e2_with[lo].mean()
        ratio_without = e2_without[hi].mean() / e2_without[lo].mean()
        assert abs(ratio_with - 1) < 0.2, ratio_with
        assert ratio_without > 1.5, ratio_without

    def test_additive_is_the_default_and_unchanged(self):
        """variance_model='additive' reproduces the pre-existing map_cis output
        exactly and reports c_a = 1."""
        d = _make_dataset(seed=120)
        kw = dict(xL_df=d['xL_df'], xR_df=d['xR_df'], nperm=500, window=1000000,
                  seed=3, verbose=False)
        base = map_cis(d['genotype_df'], d['variant_df'], d['A_df'], d['T_df'],
                       d['Va_df'], d['Vt_df'], d['pos_df'], **kw)
        add = map_cis(d['genotype_df'], d['variant_df'], d['A_df'], d['T_df'],
                      d['Va_df'], d['Vt_df'], d['pos_df'], variance_model='additive', **kw)
        pd.testing.assert_frame_equal(base, add)
        assert (base['variance_model'] == 'additive').all()
        assert (base['c_a'] == 1.0).all() and (base['c_a_null'] == 1.0).all()

    def test_library_scaled_with_unit_factor_equals_two_component(self):
        d = _make_dataset(seed=121)
        kw = dict(xL_df=d['xL_df'], xR_df=d['xR_df'], nperm=500, window=1000000,
                  seed=5, verbose=False)
        two = map_cis(d['genotype_df'], d['variant_df'], d['A_df'], d['T_df'],
                      d['Va_df'], d['Vt_df'], d['pos_df'], variance_model='two_component', **kw)
        ones = pd.Series(1.0, index=d['A_df'].columns)
        lib = map_cis(d['genotype_df'], d['variant_df'], d['A_df'], d['T_df'],
                      d['Va_df'], d['Vt_df'], d['pos_df'], variance_model='library_scaled',
                      library_factor=ones, **kw)
        for col in ('variant_id',):
            assert (two[col] == lib[col]).all()
        for col in ('slope', 'slope_se', 'pval_perm', 'c_a', 'tau_a'):
            np.testing.assert_allclose(two[col].astype(float), lib[col].astype(float), rtol=1e-5)
        assert (two['variance_model'] == 'two_component').all()
        assert (lib['variance_model'] == 'library_scaled').all()
        assert (two['c_a'] > 0).all()

    def test_all_models_recover_the_causal_variant(self):
        d = _make_dataset(seed=122)
        lf = pd.Series(np.exp(_make_gaussian_seed(9).normal(0, 0.2, len(d['A_df'].columns))),
                       index=d['A_df'].columns)
        lf /= lf.mean()
        for model, factor in (('additive', None), ('two_component', None), ('library_scaled', lf)):
            res = map_cis(d['genotype_df'], d['variant_df'], d['A_df'], d['T_df'],
                          d['Va_df'], d['Vt_df'], d['pos_df'], xL_df=d['xL_df'], xR_df=d['xR_df'],
                          nperm=500, window=1000000, seed=8, verbose=False,
                          variance_model=model, library_factor=factor, tau_refit=True)
            row = res.loc[d['causal_pheno']]
            assert row['variant_id'] == d['causal_variant'], model
            assert row['pval_perm'] < 0.05, model
            assert set(['c_a', 'c_a_null', 'variance_model']) <= set(res.columns)
            assert np.isfinite(row['c_a']) and row['c_a'] >= 0

    def test_estimate_library_factors_recovers_planted_pattern(self):
        rng = _make_gaussian_seed(301)
        G, N = 400, 60
        samples = [f'S{i:03d}' for i in range(N)]
        d_true = np.exp(rng.normal(0, 0.3, N)); d_true /= d_true.mean()
        c = rng.uniform(0.5, 3.0, G); tau = rng.uniform(0.002, 0.02, G)
        v = 10 ** rng.uniform(-3, -1, (G, N))
        a = rng.normal(0, np.sqrt(d_true[None, :] * (c[:, None] * v + tau[:, None])))
        # a few samples uninformative for some genes
        v[rng.random((G, N)) < 0.1] = 0.0
        A_df = pd.DataFrame(a, index=[f'g{i}' for i in range(G)], columns=samples)
        Va_df = pd.DataFrame(v, index=A_df.index, columns=samples)
        d_hat = hapmixqtl.estimate_library_factors(A_df, Va_df, min_informative=30)
        assert list(d_hat.index) == samples
        assert abs(d_hat.mean() - 1.0) < 1e-9
        assert d_hat.attrs['n_genes'] > 100
        rho = pd.Series(d_hat.values).corr(pd.Series(d_true), method='spearman')
        assert rho > 0.9, rho
        assert np.abs(np.log(d_hat.values / d_true)).max() < 0.4
        sub = hapmixqtl.estimate_library_factors(A_df, Va_df, genes=A_df.index[:200], min_informative=30)
        assert pd.Series(sub.values).corr(pd.Series(d_true), method='spearman') > 0.85

    def test_estimate_library_factors_falls_back_without_a_floor(self):
        """Data with no between-sample floor has every tau clamped at zero;
        the estimator then uses the non-degenerate fits and says so."""
        rng = _make_gaussian_seed(302)
        G, N = 60, 50
        d_true = np.exp(rng.normal(0, 0.3, N)); d_true /= d_true.mean()
        v = 10 ** rng.uniform(-3, -1, (G, N))
        a = rng.normal(0, np.sqrt(d_true[None, :] * 2.0 * v))
        A_df = pd.DataFrame(a, columns=[f'S{i}' for i in range(N)]); Va_df = pd.DataFrame(v, columns=A_df.columns)
        # noise makes about half of these fits interior; require more than
        # the data set can supply so the fallback is exercised
        with pytest.warns(RuntimeWarning, match='interior'):
            d_hat = hapmixqtl.estimate_library_factors(A_df, Va_df, min_informative=30, min_genes=G + 1)
        assert not d_hat.attrs['interior_only']
        assert d_hat.attrs['n_genes'] >= 0.9 * G
        assert pd.Series(d_hat.values).corr(pd.Series(d_true), method='spearman') > 0.8

    def test_variance_prior_reduces_error_and_removes_the_clamp(self, device):
        """Empirical-Bayes priors toward expression bins: lower error on (c, tau)
        than the clamped fit, and the tau = 0 clamp all but disappears."""
        rng = _make_gaussian_seed(303)
        G, N = 200, 60
        samples = [f'S{i:03d}' for i in range(N)]
        # a floor small relative to the measurement term, as on well-expressed
        # BrainVar genes, so that the per-gene intercept is often unsupported
        c_true = np.exp(rng.normal(np.log(1.5), 0.4, G))
        tau_true = np.exp(rng.normal(np.log(0.002), 0.5, G))
        v = 10 ** rng.uniform(-2.5, -0.5, (G, N))
        a = rng.normal(0, np.sqrt(c_true[:, None] * v + tau_true[:, None]))
        A_df = pd.DataFrame(a, index=[f'g{i}' for i in range(G)], columns=samples)
        Va_df = pd.DataFrame(v, index=A_df.index, columns=samples)
        priors = hapmixqtl.estimate_variance_priors(A_df, Va_df, n_bins=4, min_informative=30)
        assert set(['prior_c', 'prior_tau', 'prior_sd_c', 'prior_sd_tau', 'bin']) <= set(priors.columns)
        assert list(priors.index) == list(A_df.index)
        assert priors.attrs['kappa'] >= 2.0 and priors.attrs['n_genes'] == G
        assert (priors['prior_tau'] > 0).all() and (priors['prior_sd_tau'] > 0).all()
        T = lambda x: torch.tensor(x, dtype=torch.float64, device=device)
        e_clamp, e_prior, tau0_clamp, floored = [], [], 0, 0
        for g in range(G):
            f0 = hapmixqtl._estimate_c_tau(T(a[g]), T(v[g]), None, device, intercept=False)
            f1 = hapmixqtl._estimate_c_tau(T(a[g]), T(v[g]), None, device, intercept=False,
                                           prior=hapmixqtl._prior_tuple(priors, A_df.index[g]))
            e_clamp.append(((f0['c'] - c_true[g]) / c_true[g]) ** 2 + ((f0['tau'] - tau_true[g]) / tau_true[g]) ** 2)
            e_prior.append(((f1['c'] - c_true[g]) / c_true[g]) ** 2 + ((f1['tau'] - tau_true[g]) / tau_true[g]) ** 2)
            tau0_clamp += f0['tau'] < 1e-6
            floored += f1['floored']
            assert f1['c'] > 0 and f1['tau'] > 0                # log-scale posterior: positivity is automatic
            assert np.isfinite(f1['c_raw']) and np.isfinite(f1['tau_raw'])
        assert np.mean(e_prior) < 0.7 * np.mean(e_clamp), (np.mean(e_prior), np.mean(e_clamp))
        assert tau0_clamp >= 0.05 * G, tau0_clamp          # the clamp does bite on this design
        assert floored <= 0.05 * G and floored <= tau0_clamp / 5, (floored, tau0_clamp)   # the prior all but removes it

    def test_map_cis_with_variance_prior(self):
        d = _make_dataset(seed=124)
        rng = _make_gaussian_seed(304)
        samples = list(d['A_df'].columns)
        extra = 60
        v = 10 ** rng.uniform(-2, -0.5, (extra, len(samples)))
        a = rng.normal(0, np.sqrt(1.5 * v + 0.02))
        A_big = pd.concat([d['A_df'], pd.DataFrame(a, index=[f'x{i}' for i in range(extra)], columns=samples)])
        Va_big = pd.concat([d['Va_df'], pd.DataFrame(v, index=[f'x{i}' for i in range(extra)], columns=samples)])
        kw = dict(xL_df=d['xL_df'], xR_df=d['xR_df'], nperm=500, window=1000000, seed=9, verbose=False)
        args = (d['genotype_df'], d['variant_df'], d['A_df'], d['T_df'], d['Va_df'], d['Vt_df'], d['pos_df'])
        priors = hapmixqtl.estimate_variance_priors(A_big, Va_big, n_bins=2, min_informative=30)
        res = map_cis(*args, variance_model='two_component', variance_prior=priors, **kw)
        row = res.loc[d['causal_pheno']]
        assert row['variant_id'] == d['causal_variant'] and row['pval_perm'] < 0.05
        assert res['variance_prior'].astype(bool).all()
        assert set(['c_a_raw', 'tau_a_raw', 'c_a_floored']) <= set(res.columns)
        assert (res['tau_a'].astype(float) > 0).all()
        lf = pd.Series(np.exp(rng.normal(0, 0.2, len(samples))), index=samples); lf /= lf.mean()
        with pytest.raises(ValueError, match='different model'):
            map_cis(*args, variance_model='library_scaled', library_factor=lf, variance_prior=priors, **kw)
        priors_lf = hapmixqtl.estimate_variance_priors(A_big, Va_big, n_bins=2, min_informative=30, library_factor=lf)
        res2 = map_cis(*args, variance_model='library_scaled', library_factor=lf, variance_prior=priors_lf, **kw)
        assert res2.loc[d['causal_pheno'], 'variant_id'] == d['causal_variant']
        with pytest.raises(ValueError, match='two-component'):
            map_cis(*args, variance_model='additive', variance_prior=priors, **kw)

    def test_variance_model_argument_errors(self):
        d = _make_dataset(seed=123)
        kw = dict(xL_df=d['xL_df'], xR_df=d['xR_df'], nperm=100, window=1000000, verbose=False)
        args = (d['genotype_df'], d['variant_df'], d['A_df'], d['T_df'], d['Va_df'], d['Vt_df'], d['pos_df'])
        ones = pd.Series(1.0, index=d['A_df'].columns)
        with pytest.raises(ValueError, match='library_factor'):
            map_cis(*args, variance_model='library_scaled', **kw)
        with pytest.raises(ValueError, match='library_factor'):
            map_cis(*args, variance_model='additive', library_factor=ones, **kw)
        with pytest.raises(ValueError, match='variance_model'):
            map_cis(*args, variance_model='multiplicative', **kw)
        with pytest.raises(ValueError, match="tau_mode='estimate'"):
            map_cis(*args, variance_model='two_component', tau_mode='zero', **kw)
        with pytest.raises(ValueError, match='positive'):
            map_cis(*args, variance_model='library_scaled', library_factor=ones * 0, **kw)

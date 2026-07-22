"""
hapmixQTL: cis-QTL mapping using haplotype-resolved expression posteriors.

Combines two information channels via inverse-variance meta-analysis:
  1. Allelic contrast (ASE): log(yL + kappa) - log(yR + kappa), weighted by
     inferential uncertainty from Gibbs draws
  2. Total expression: log((yL + yR)/2 + kappa), similarly weighted

Method A performs separate WLS regressions per channel (ASE on signed het
indicator s, total on g/2) and combines via inverse-variance meta-analysis.
The slope estimates log allelic fold change (log aFC).

WLS is implemented via the sqrt-weight transform: multiplying both response
and predictors by sqrt(w_i) converts WLS into OLS, enabling efficient
GPU-vectorized computation across all cis variants simultaneously.

Inferential variances from Gibbs draws propagate into weights as
w_i = 1 / (v_inf_i + tau), where v_inf_i is the across-draw variance of
the transformed expression for sample i, and tau is an optional
overdispersion parameter.

Phase determines the signed heterozygote indicator s_i = xL_i - xR_i:
  s = +1 if ALT allele is on haplotype L
  s = -1 if ALT allele is on haplotype R
  s =  0 if homozygous (or phase unknown)
When phase is unavailable (s=0 for all samples), the ASE channel contributes
nothing and results match total-channel-only regression.
"""

import torch
import numpy as np
import pandas as pd
import scipy.stats as stats
import sys
import os
import time
from collections import OrderedDict

sys.path.insert(1, os.path.dirname(__file__))
import genotypeio
import susie
import knockoffs as ko
from core import *


# ---------------------------------------------------------------------------
#  I/O utilities
# ---------------------------------------------------------------------------

def read_hapmixqtl_inputs(a_bed, t_bed, va_bed, vt_bed, cat_bed=None):
    """
    Load precomputed hapmixQTL summary matrices in BED-like format.

    Each file follows the tensorQTL phenotype BED convention:
    chr, start, end, phenotype_id, sample1, sample2, ...

    Returns:
        A_df:   allelic contrast a_i [phenotypes x samples]
        T_df:   log total t_i [phenotypes x samples]
        Va_df:  inferential variance of a [phenotypes x samples]
        Vt_df:  inferential variance of t [phenotypes x samples]
        Cat_df: inferential covariance a,t [phenotypes x samples] (or None)
        pos_df: phenotype positions [phenotypes x (chr, pos|start,end)]
    """
    A_df, pos_df = read_phenotype_bed(a_bed)
    T_df, _ = read_phenotype_bed(t_bed)
    Va_df, _ = read_phenotype_bed(va_bed)
    Vt_df, _ = read_phenotype_bed(vt_bed)

    assert A_df.index.equals(T_df.index), "Phenotype IDs must match across A and T"
    assert A_df.index.equals(Va_df.index), "Phenotype IDs must match across A and Va"
    assert A_df.index.equals(Vt_df.index), "Phenotype IDs must match across A and Vt"
    assert A_df.columns.equals(T_df.columns), "Sample IDs must match across A and T"
    assert A_df.columns.equals(Va_df.columns), "Sample IDs must match across A and Va"
    assert A_df.columns.equals(Vt_df.columns), "Sample IDs must match across A and Vt"

    Cat_df = None
    if cat_bed is not None:
        Cat_df, _ = read_phenotype_bed(cat_bed)
        assert A_df.index.equals(Cat_df.index)
        assert A_df.columns.equals(Cat_df.columns)

    return A_df, T_df, Va_df, Vt_df, Cat_df, pos_df


def compute_summaries_from_gibbs(yL, yR, kappa=0.5):
    """
    Compute hapmixQTL summary statistics from Gibbs draws.

    Args:
        yL: haplotype L expression [features, samples, draws]
        yR: haplotype R expression [features, samples, draws]
        kappa: pseudocount (default 0.5)

    Returns:
        A:   allelic contrast mean [features, samples]
        T:   log total mean [features, samples]
        Va:  inferential variance of a [features, samples]
        Vt:  inferential variance of t [features, samples]
        Cat: inferential covariance of a,t [features, samples]
    """
    a_draws = np.log(yL + kappa) - np.log(yR + kappa)
    t_draws = np.log((yL + yR) / 2 + kappa)

    A = a_draws.mean(axis=2)
    T = t_draws.mean(axis=2)
    Va = a_draws.var(axis=2, ddof=0)
    Vt = t_draws.var(axis=2, ddof=0)
    Cat = ((a_draws - a_draws.mean(axis=2, keepdims=True)) *
           (t_draws - t_draws.mean(axis=2, keepdims=True))).mean(axis=2)

    return A, T, Va, Vt, Cat


# ---------------------------------------------------------------------------
#  WeightedResidualizer
# ---------------------------------------------------------------------------

class WeightedResidualizer:
    """
    Residualizer for weighted least squares via sqrt-weight transform.

    In standard OLS the intercept is handled by centering. In WLS after the
    sqrt-weight transform, a constant intercept alpha becomes alpha*sqrt(w_i),
    which varies across samples. This class includes sqrt(w) as an explicit
    column in the design matrix so the QR projection removes it correctly.
    """

    def __init__(self, C_t, sqrt_w_t):
        """
        Args:
            C_t: covariates [N, n_cov] (without intercept), or None
            sqrt_w_t: sqrt per-sample weights [N]
        """
        N = sqrt_w_t.shape[0]
        intercept = sqrt_w_t.unsqueeze(1)
        if C_t is not None and C_t.numel() > 0 and C_t.shape[1] > 0:
            C_star = sqrt_w_t.unsqueeze(1) * C_t
            design = torch.cat([intercept, C_star], dim=1)
        else:
            design = intercept
        self.Q_t, _ = torch.linalg.qr(design)
        self.dof = N - 1 - design.shape[1]

    def transform(self, M_t):
        """Project out weighted covariates from rows of M_t [features, N]."""
        return M_t - torch.mm(torch.mm(M_t, self.Q_t), self.Q_t.t())


# ---------------------------------------------------------------------------
#  Core regression
# ---------------------------------------------------------------------------

def _wls_regression(y_star_t, x_star_t, residualizer, robust=False):
    """
    Known-variance GLS on sqrt-weight-transformed data.

    The Gibbs inferential variances are treated as *known* measurement
    variances: Var(error_i) = v_inf_i + tau. Under the sqrt-weight transform
    (y* = sqrt(w) y, x* = sqrt(w) x, w_i = 1/(v_inf_i + tau)), the estimator
    reduces to ordinary dot products, but the standard error is the
    known-variance GLS SE

        Var(beta_hat) = (x*' x*)^-1 = 1 / xx

    rather than the estimated-dispersion WLS SE sqrt(sigma2_hat / xx). This is
    the key difference from standard WLS and is what lets the inferential
    uncertainty propagate into beta_se in absolute terms: uniformly inflating
    all v_inf shrinks the weights, shrinks xx, and inflates the SE (an
    estimated-dispersion SE would instead absorb the scale into sigma2_hat and
    be invariant to it, so a channel with huge inferential variance could never
    be down-weighted -- see the huge-Va test).

    Args:
        y_star_t: [1, N] sqrt(w) * phenotype
        x_star_t: [V, N] sqrt(w) * predictors
        residualizer: WeightedResidualizer
        robust: if True use sandwich (HC1) standard errors instead, which are
            robust to misspecification of the known variance scale

    Returns:
        slope_t: [V] estimated slopes
        slope_se_t: [V] standard errors (inf where predictor has zero variance)
    """
    y_res = residualizer.transform(y_star_t)
    x_res = residualizer.transform(x_star_t)

    xy = (x_res * y_res).sum(1)
    xx = (x_res * x_res).sum(1)

    # A predictor lying (nearly) in the design span leaves only rounding noise
    # after residualization. Gate on the residual variance relative to the
    # original predictor scale so degenerate predictors (e.g. s=0, or a column
    # collinear with the covariates) are treated as zero-variance in float32.
    xx_pre = (x_star_t * x_star_t).sum(1)
    valid = xx > 1e-12 * xx_pre.clamp(min=1e-30)
    slope = torch.zeros_like(xy)
    slope_se = torch.full_like(xy, float('inf'))

    if valid.any():
        slope[valid] = xy[valid] / xx[valid]
        if not robust:
            # Known-variance GLS: Var(beta_hat) = 1 / xx (weights are absolute
            # precisions, so no residual-based dispersion is estimated).
            slope_se[valid] = torch.sqrt(1.0 / xx[valid])
        else:
            # Heteroskedasticity-robust (HC1) sandwich SE, valid even if the
            # supplied known variances are only correct up to an unknown scale.
            e = y_res - slope.unsqueeze(1) * x_res
            N = x_res.shape[1]
            meat = (x_res * x_res * e * e).sum(1)
            correction = float(N) / max(residualizer.dof, 1)
            var_robust = meat / (xx * xx) * correction
            slope_se[valid] = torch.sqrt(var_robust[valid])

    return slope, slope_se


def _estimate_tau(y_t, v_inf_t, covariates_t, device):
    """
    Estimate overdispersion parameter tau using moment estimator.

    Under the model Var(error_i) = v_inf_i + tau, the weighted
    residuals (with w_i = 1/v_inf_i) have expected variance
    1 + tau * mean(1/v_inf). This function solves for tau from
    the observed residual variance.
    """
    sqrt_w = torch.sqrt(1.0 / v_inf_t.clamp(min=1e-8))
    res = WeightedResidualizer(covariates_t, sqrt_w)
    y_star = (y_t * sqrt_w).unsqueeze(0)
    y_res = res.transform(y_star).squeeze()

    rss = (y_res * y_res).sum()
    dof_null = y_t.shape[0] - res.Q_t.shape[1]
    sigma2_hat = rss / max(dof_null, 1)

    mean_inv_v = (1.0 / v_inf_t.clamp(min=1e-8)).mean()
    tau = torch.clamp((sigma2_hat - 1.0) / mean_inv_v, min=0.0)
    return tau


# ---------------------------------------------------------------------------
#  Association tests
# ---------------------------------------------------------------------------

def calculate_hapmixqtl_nominal(genotypes_t, sign_t, a_t, t_t,
                                 sqrt_wa_t, sqrt_wt_t,
                                 residualizer_a, residualizer_t,
                                 robust=False):
    """
    hapmixQTL association test for all variants in a cis window (Method A).

    Runs two separate WLS regressions (ASE on s, total on g/2) and
    combines via inverse-variance meta-analysis. Using g/2 as the total
    channel predictor makes its slope estimate the same quantity as the
    ASE channel slope: the full log allelic fold change (log aFC).

    Args:
        genotypes_t: [V, N] dosage (0/1/2)
        sign_t:      [V, N] signed het indicator s = xL - xR
        a_t:         [N]    allelic contrast
        t_t:         [N]    log total expression
        sqrt_wa_t:   [N]    sqrt ASE weights
        sqrt_wt_t:   [N]    sqrt total weights
        residualizer_a: WeightedResidualizer for ASE channel
        residualizer_t: WeightedResidualizer for total channel
        robust: if True use sandwich SEs

    Returns:
        tstat_t:      [V] combined t-statistic
        slope_t:      [V] combined slope (log aFC)
        slope_se_t:   [V] combined SE
        slope_a_t:    [V] ASE channel slope
        slope_a_se_t: [V] ASE channel SE
        slope_tc_t:   [V] total channel slope
        slope_tc_se_t:[V] total channel SE
    """
    # ASE channel: a = beta * s + covariates + error
    a_star = (a_t * sqrt_wa_t).unsqueeze(0)
    s_star = sign_t * sqrt_wa_t.unsqueeze(0)
    slope_a, se_a = _wls_regression(a_star, s_star, residualizer_a, robust=robust)

    # Total channel: t = beta * (g/2) + covariates + error
    t_star = (t_t * sqrt_wt_t).unsqueeze(0)
    g_half_star = (genotypes_t / 2) * sqrt_wt_t.unsqueeze(0)
    slope_tc, se_tc = _wls_regression(t_star, g_half_star, residualizer_t, robust=robust)

    # Inverse-variance meta-analysis
    inv_var_a = torch.where(
        torch.isfinite(se_a) & (se_a > 0),
        1.0 / (se_a * se_a),
        torch.zeros_like(se_a),
    )
    inv_var_t = torch.where(
        torch.isfinite(se_tc) & (se_tc > 0),
        1.0 / (se_tc * se_tc),
        torch.zeros_like(se_tc),
    )

    total_inv_var = inv_var_a + inv_var_t
    slope_combined = torch.where(
        total_inv_var > 0,
        (slope_a * inv_var_a + slope_tc * inv_var_t) / total_inv_var,
        torch.zeros_like(slope_a),
    )
    se_combined = torch.where(
        total_inv_var > 0,
        torch.sqrt(1.0 / total_inv_var),
        torch.full_like(slope_a, float('inf')),
    )
    tstat_combined = slope_combined / se_combined

    return tstat_combined, slope_combined, se_combined, slope_a, se_a, slope_tc, se_tc


def calculate_hapmixqtl_permutations(genotypes_t, sign_t, a_t, t_t,
                                      sqrt_wa_t, sqrt_wt_t,
                                      residualizer_a, residualizer_t,
                                      permutation_ix_t):
    """
    Compute nominal and permutation statistics for hapmixQTL.

    Uses fixed weights across permutations (approximate permutation):
    only phenotype values (a, t) are shuffled while weights remain in
    original sample order. This enables efficient batch computation via
    matrix multiplication. The approximation is very good when inferential
    variances are similar across samples.

    Returns:
        r_nominal:  signed correlation for best variant (scalar)
        std_ratio:  sqrt(pheno_var/geno_var) for best variant (scalar)
        best_ix:    index of best variant (scalar)
        r2_perm_t:  max r^2 per permutation [nperm]
        g_best:     genotype vector for best variant [N]
    """
    dof = residualizer_a.dof

    # --- Pre-transform and residualize fixed predictors ---
    # ASE
    s_star = sign_t * sqrt_wa_t.unsqueeze(0)
    s_star_res = residualizer_a.transform(s_star)
    xx_a = (s_star_res * s_star_res).sum(1)

    # Total
    g_half_star = (genotypes_t / 2) * sqrt_wt_t.unsqueeze(0)
    g_half_star_res = residualizer_t.transform(g_half_star)
    xx_t = (g_half_star_res * g_half_star_res).sum(1)

    # --- Nominal statistics ---
    a_star = (a_t * sqrt_wa_t).unsqueeze(0)
    a_star_res = residualizer_a.transform(a_star)
    t_star = (t_t * sqrt_wt_t).unsqueeze(0)
    t_star_res = residualizer_t.transform(t_star)

    xy_a_nom = (s_star_res * a_star_res).sum(1)
    yy_a_nom = (a_star_res * a_star_res).sum()
    xy_t_nom = (g_half_star_res * t_star_res).sum(1)
    yy_t_nom = (t_star_res * t_star_res).sum()

    tstat2_nom = _combined_tstat2(xy_a_nom, xx_a, yy_a_nom,
                                  xy_t_nom, xx_t, yy_t_nom, dof)

    tstat2_nom_clean = tstat2_nom.clone()
    tstat2_nom_clean[torch.isnan(tstat2_nom_clean)] = -1
    best_ix = tstat2_nom_clean.argmax()

    # Known-variance combine for the best variant (inverse variances are xx).
    iva = torch.where(xx_a[best_ix] > 1e-30, xx_a[best_ix], torch.zeros_like(xx_a[best_ix]))
    ivt = torch.where(xx_t[best_ix] > 1e-30, xx_t[best_ix], torch.zeros_like(xx_t[best_ix]))
    xy_a_b = torch.where(xx_a[best_ix] > 1e-30, xy_a_nom[best_ix], torch.zeros_like(xy_a_nom[best_ix]))
    xy_t_b = torch.where(xx_t[best_ix] > 1e-30, xy_t_nom[best_ix], torch.zeros_like(xy_t_nom[best_ix]))
    slope_nom = (xy_a_b + xy_t_b) / (iva + ivt + 1e-30)

    # Map the combined statistic to a correlation-like r for the empirical
    # p-value. tstat2 already equals slope_nom^2 * total_inv; convert with the
    # usual r^2 = t^2/(t^2+dof) so nominal and permutation values are directly
    # comparable (the mapping is monotonic, so ranks -- and thus the empirical
    # p-value -- are preserved regardless of the exact scaling).
    tstat2_best = tstat2_nom[best_ix]
    r2_nominal = tstat2_best / (tstat2_best + dof)
    r_nominal = torch.sign(slope_nom) * torch.sqrt(r2_nominal.clamp(min=0))

    pheno_var = yy_a_nom / a_star_res.shape[1] + yy_t_nom / t_star_res.shape[1]
    geno_var = xx_a[best_ix] / s_star_res.shape[1] + xx_t[best_ix] / g_half_star_res.shape[1]
    std_ratio = torch.sqrt(pheno_var / (geno_var + 1e-30))

    # --- Permutation statistics ---
    nperm = permutation_ix_t.shape[0]

    a_perms = a_t[permutation_ix_t]
    t_perms = t_t[permutation_ix_t]

    a_star_perms = a_perms * sqrt_wa_t.unsqueeze(0)
    t_star_perms = t_perms * sqrt_wt_t.unsqueeze(0)

    a_star_res_perms = residualizer_a.transform(a_star_perms)
    t_star_res_perms = residualizer_t.transform(t_star_perms)

    xy_a_perm = torch.mm(s_star_res, a_star_res_perms.t())
    yy_a_perm = (a_star_res_perms * a_star_res_perms).sum(1)
    xy_t_perm = torch.mm(g_half_star_res, t_star_res_perms.t())
    yy_t_perm = (t_star_res_perms * t_star_res_perms).sum(1)

    tstat2_perm = _combined_tstat2(xy_a_perm, xx_a, yy_a_perm,
                                    xy_t_perm, xx_t, yy_t_perm, dof)

    tstat2_perm[torch.isnan(tstat2_perm)] = 0
    r2_perm = tstat2_perm / (tstat2_perm + dof)
    max_r2_perm, _ = r2_perm.max(0)

    return r_nominal, std_ratio, best_ix, max_r2_perm, genotypes_t[best_ix]


def _combined_tstat2(xy_a, xx_a, yy_a, xy_t, xx_t, yy_t, dof):
    """
    Compute combined (known-variance) statistic squared from dot-product
    summaries, matching the inverse-variance meta-analysis in
    ``calculate_hapmixqtl_nominal``.

    Under known-variance GLS the per-channel inverse variance of the slope is
    just ``xx`` (since se^2 = 1/xx), so the combined statistic simplifies to

        beta_c   = (xy_a + xy_t) / (xx_a + xx_t)          [both channels valid]
        stat^2   = beta_c^2 * (xx_a + xx_t)

    ``yy_a``/``yy_t`` are unused for the SE here (kept in the signature for
    symmetry with an estimated-dispersion variant and for callers that also
    want residual sums of squares). Works for both scalar (nominal) and 2D
    (permutation) ``xy`` by broadcasting.
    """
    is_perm = xy_a.dim() == 2

    if is_perm:
        xx_a_e = xx_a.unsqueeze(1)
        xx_t_e = xx_t.unsqueeze(1)
    else:
        xx_a_e = xx_a
        xx_t_e = xx_t

    # Known-variance inverse variances of the per-channel slopes are xx itself;
    # a degenerate predictor (xx ~ 0) contributes zero weight.
    inv_var_a = torch.where(xx_a_e > 1e-30, xx_a_e, torch.zeros_like(xx_a_e))
    inv_var_t = torch.where(xx_t_e > 1e-30, xx_t_e, torch.zeros_like(xx_t_e))

    xy_a_eff = torch.where(xx_a_e > 1e-30, xy_a, torch.zeros_like(xy_a))
    xy_t_eff = torch.where(xx_t_e > 1e-30, xy_t, torch.zeros_like(xy_t))

    total_inv = inv_var_a + inv_var_t
    # beta_c * total_inv = slope_a*inv_var_a + slope_t*inv_var_t
    #                    = xy_a + xy_t (since slope = xy/xx and inv_var = xx)
    numer = xy_a_eff + xy_t_eff
    slope_comb = torch.where(
        total_inv > 0,
        numer / (total_inv + 1e-30),
        torch.zeros_like(numer),
    )
    tstat2 = slope_comb * slope_comb * total_inv
    return tstat2


# ---------------------------------------------------------------------------
#  Nominal mapping
# ---------------------------------------------------------------------------

def map_nominal(genotype_df, variant_df, A_df, T_df, Va_df, Vt_df,
                phenotype_pos_df, xL_df=None, xR_df=None, prefix='',
                covariates_df=None, maf_threshold=0, window=1000000,
                tau_mode='zero', se_mode='model',
                output_dir='.', logger=None, verbose=True):
    """
    hapmixQTL cis-QTL mapping: nominal associations for all variant-phenotype pairs.

    Writes per-chromosome parquet files in the format:
        <output_dir>/<prefix>.hapmixqtl_pairs.<chr>.parquet

    Args:
        genotype_df:      genotypes [variants x samples]
        variant_df:       variant positions (chrom, pos)
        A_df:             allelic contrast [phenotypes x samples]
        T_df:             log total expression [phenotypes x samples]
        Va_df:            inferential variance for a [phenotypes x samples]
        Vt_df:            inferential variance for t [phenotypes x samples]
        phenotype_pos_df: phenotype positions [phenotypes x (chr, pos)]
        xL_df:            haplotype L ALT allele (0/1) [variants x samples] or None
        xR_df:            haplotype R ALT allele (0/1) [variants x samples] or None
        prefix:           output file prefix
        covariates_df:    covariates [samples x covariates] or None
        maf_threshold:    minimum minor allele frequency
        window:           cis-window size in bases
        tau_mode:         'zero' (default) or 'estimate'
        se_mode:          'model' (default) or 'robust' (sandwich)
        output_dir:       output directory
        logger:           SimpleLogger instance
        verbose:          print progress
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if logger is None:
        logger = SimpleLogger()

    samples = A_df.columns
    N = len(samples)
    assert A_df.columns.equals(T_df.columns), "Sample mismatch between A and T"
    assert A_df.columns.equals(Va_df.columns), "Sample mismatch between A and Va"
    assert A_df.columns.equals(Vt_df.columns), "Sample mismatch between A and Vt"
    assert A_df.index.equals(T_df.index), "Phenotype mismatch between A and T"
    assert A_df.index.equals(Va_df.index), "Phenotype mismatch between A and Va"
    assert A_df.index.equals(Vt_df.index), "Phenotype mismatch between A and Vt"

    logger.write('hapmixQTL mapping: nominal associations for all variant-phenotype pairs')
    logger.write(f'  * {N} samples')
    logger.write(f'  * {A_df.shape[0]} phenotypes')

    robust = se_mode == 'robust'

    if covariates_df is not None:
        assert np.all(samples == covariates_df.index), \
            "Covariate samples must match phenotype samples"
        logger.write(f'  * {covariates_df.shape[1]} covariates')
        covariates_t = torch.tensor(covariates_df.values, dtype=torch.float32).to(device)
        n_cov = covariates_df.shape[1]
    else:
        covariates_t = None
        n_cov = 0

    has_phase = xL_df is not None and xR_df is not None
    if has_phase:
        logger.write('  * phase genotypes available (ASE + total channels)')
        assert (xL_df.index == genotype_df.index).all(), \
            "xL variant IDs must match genotype variant IDs"
        assert (xR_df.index == genotype_df.index).all(), \
            "xR variant IDs must match genotype variant IDs"
    else:
        logger.write('  * no phase genotypes (total channel only)')

    logger.write(f'  * {variant_df.shape[0]} variants')
    logger.write(f'  * tau mode: {tau_mode}')
    logger.write(f'  * SE mode: {se_mode}')
    if maf_threshold > 0:
        logger.write(f'  * applying in-sample {maf_threshold} MAF filter')
    logger.write(f'  * cis-window: ±{window:,}')

    dof = N - 2 - n_cov

    genotype_ix = np.array([genotype_df.columns.tolist().index(i) for i in samples])
    genotype_ix_t = torch.from_numpy(genotype_ix).to(device)

    # Use T_df as phenotype for InputGeneratorCis (less likely to be constant)
    igc = genotypeio.InputGeneratorCis(
        genotype_df, variant_df, T_df, phenotype_pos_df, window=window,
    )
    pheno_ix = {pid: i for i, pid in enumerate(A_df.index)}

    start_time = time.time()
    k = 0
    logger.write('  * Computing associations')
    for chrom in igc.chrs:
        logger.write(f'    Mapping chromosome {chrom}')

        n = 0
        for pid in igc.phenotype_pos_df[igc.phenotype_pos_df['chr'] == chrom].index:
            if pid in igc.cis_ranges:
                j = igc.cis_ranges[pid]
                n += j[1] - j[0] + 1

        chr_res = OrderedDict()
        chr_res['phenotype_id'] = []
        chr_res['variant_id'] = []
        chr_res['start_distance'] = np.empty(n, dtype=np.int32)
        if 'pos' not in phenotype_pos_df:
            chr_res['end_distance'] = np.empty(n, dtype=np.int32)
        chr_res['af'] = np.empty(n, dtype=np.float32)
        chr_res['ma_samples'] = np.empty(n, dtype=np.int32)
        chr_res['ma_count'] = np.empty(n, dtype=np.int32)
        chr_res['pval_nominal'] = np.empty(n, dtype=np.float64)
        chr_res['slope'] = np.empty(n, dtype=np.float32)
        chr_res['slope_se'] = np.empty(n, dtype=np.float32)
        chr_res['pval_a'] = np.empty(n, dtype=np.float64)
        chr_res['slope_a'] = np.empty(n, dtype=np.float32)
        chr_res['slope_a_se'] = np.empty(n, dtype=np.float32)
        chr_res['pval_t'] = np.empty(n, dtype=np.float64)
        chr_res['slope_t'] = np.empty(n, dtype=np.float32)
        chr_res['slope_t_se'] = np.empty(n, dtype=np.float32)

        start = 0
        for k, (_, genotypes, genotype_range, phenotype_id) in enumerate(
            igc.generate_data(chrom=chrom, verbose=verbose), k + 1
        ):
            if phenotype_id not in pheno_ix:
                continue

            pidx = pheno_ix[phenotype_id]
            a_t = torch.tensor(A_df.values[pidx], dtype=torch.float32).to(device)
            t_t = torch.tensor(T_df.values[pidx], dtype=torch.float32).to(device)
            va_t = torch.tensor(Va_df.values[pidx], dtype=torch.float32).to(device).clamp(min=1e-8)
            vt_t = torch.tensor(Vt_df.values[pidx], dtype=torch.float32).to(device).clamp(min=1e-8)

            if tau_mode == 'estimate':
                tau_a = _estimate_tau(a_t, va_t, covariates_t, device)
                tau_t_val = _estimate_tau(t_t, vt_t, covariates_t, device)
                sqrt_wa_t = torch.sqrt(1.0 / (va_t + tau_a))
                sqrt_wt_t = torch.sqrt(1.0 / (vt_t + tau_t_val))
            else:
                sqrt_wa_t = torch.sqrt(1.0 / va_t)
                sqrt_wt_t = torch.sqrt(1.0 / vt_t)

            residualizer_a = WeightedResidualizer(covariates_t, sqrt_wa_t)
            residualizer_tc = WeightedResidualizer(covariates_t, sqrt_wt_t)

            genotypes_t = torch.tensor(genotypes, dtype=torch.float32).to(device)
            genotypes_t = genotypes_t[:, genotype_ix_t]
            impute_mean(genotypes_t)

            if has_phase:
                xL_vals = xL_df.values[genotype_range[0]:genotype_range[-1] + 1]
                xR_vals = xR_df.values[genotype_range[0]:genotype_range[-1] + 1]
                xL_t = torch.tensor(xL_vals, dtype=torch.float32).to(device)[:, genotype_ix_t]
                xR_t = torch.tensor(xR_vals, dtype=torch.float32).to(device)[:, genotype_ix_t]
                sign_t = xL_t - xR_t
            else:
                sign_t = torch.zeros_like(genotypes_t)

            variant_ids = variant_df.index[genotype_range[0]:genotype_range[-1] + 1]
            start_distance = np.int32(
                variant_df['pos'].values[genotype_range[0]:genotype_range[-1] + 1]
                - igc.phenotype_start[phenotype_id]
            )
            if 'pos' not in phenotype_pos_df:
                end_distance = np.int32(
                    variant_df['pos'].values[genotype_range[0]:genotype_range[-1] + 1]
                    - igc.phenotype_end[phenotype_id]
                )

            if maf_threshold > 0:
                maf_t = calculate_maf(genotypes_t)
                mask_t = maf_t >= maf_threshold
                genotypes_t = genotypes_t[mask_t]
                sign_t = sign_t[mask_t]
                mask = mask_t.cpu().numpy().astype(bool)
                variant_ids = variant_ids[mask]
                start_distance = start_distance[mask]
                if 'pos' not in phenotype_pos_df:
                    end_distance = end_distance[mask]

            if genotypes_t.shape[0] == 0:
                continue

            res = calculate_hapmixqtl_nominal(
                genotypes_t, sign_t, a_t, t_t,
                sqrt_wa_t, sqrt_wt_t,
                residualizer_a, residualizer_tc,
                robust=robust,
            )
            (tstat, slope, slope_se, slope_a, se_a,
             slope_tc, se_tc) = [r.cpu().numpy() for r in res]

            tstat_a = np.where(np.isfinite(se_a) & (se_a > 0),
                               slope_a / se_a, 0.0)
            tstat_tc = np.where(np.isfinite(se_tc) & (se_tc > 0),
                                slope_tc / se_tc, 0.0)

            af_t, ma_samples_t, ma_count_t = get_allele_stats(genotypes_t)
            af, ma_samples, ma_count = [
                x.cpu().numpy() for x in [af_t, ma_samples_t, ma_count_t]
            ]

            nv = len(variant_ids)
            chr_res['phenotype_id'].extend([phenotype_id] * nv)
            chr_res['variant_id'].extend(variant_ids)
            chr_res['start_distance'][start:start + nv] = start_distance
            if 'pos' not in phenotype_pos_df:
                chr_res['end_distance'][start:start + nv] = end_distance
            chr_res['af'][start:start + nv] = af
            chr_res['ma_samples'][start:start + nv] = ma_samples
            chr_res['ma_count'][start:start + nv] = ma_count
            chr_res['pval_nominal'][start:start + nv] = tstat
            chr_res['slope'][start:start + nv] = slope
            chr_res['slope_se'][start:start + nv] = slope_se
            chr_res['pval_a'][start:start + nv] = tstat_a
            chr_res['slope_a'][start:start + nv] = slope_a
            chr_res['slope_a_se'][start:start + nv] = se_a
            chr_res['pval_t'][start:start + nv] = tstat_tc
            chr_res['slope_t'][start:start + nv] = slope_tc
            chr_res['slope_t_se'][start:start + nv] = se_tc
            start += nv

        logger.write(f'    time elapsed: {(time.time() - start_time) / 60:.2f} min')

        if start < n:
            for x in chr_res:
                chr_res[x] = chr_res[x][:start]

        if start == 0:
            continue

        chr_res_df = pd.DataFrame(chr_res)
        m = chr_res_df['pval_nominal'].notnull()
        m = m[m].index
        chr_res_df.loc[m, 'pval_nominal'] = get_t_pval(
            chr_res_df.loc[m, 'pval_nominal'], dof
        )
        chr_res_df.loc[m, 'pval_a'] = get_t_pval(
            chr_res_df.loc[m, 'pval_a'], dof
        )
        chr_res_df.loc[m, 'pval_t'] = get_t_pval(
            chr_res_df.loc[m, 'pval_t'], dof
        )
        print('    * writing output')
        chr_res_df.to_parquet(
            os.path.join(output_dir, f'{prefix}.hapmixqtl_pairs.{chrom}.parquet')
        )

    logger.write('done.')


# ---------------------------------------------------------------------------
#  Permutation-based mapping (empirical p-values)
# ---------------------------------------------------------------------------

def map_cis(genotype_df, variant_df, A_df, T_df, Va_df, Vt_df,
            phenotype_pos_df, xL_df=None, xR_df=None,
            covariates_df=None, maf_threshold=0, beta_approx=True,
            nperm=10000, window=1000000, tau_mode='zero', se_mode='model',
            logger=None, seed=None, verbose=True, warn_monomorphic=True):
    """
    hapmixQTL cis-QTL mapping with permutation-based empirical p-values.

    For each phenotype, finds the best cis variant and computes empirical
    p-values by permuting sample labels. Uses approximate permutation with
    fixed weights for computational efficiency.

    Returns:
        DataFrame with one row per phenotype, analogous to cis.map_cis output.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if logger is None:
        logger = SimpleLogger()

    samples = A_df.columns
    N = len(samples)
    assert A_df.columns.equals(T_df.columns)
    assert A_df.index.equals(T_df.index)

    logger.write('hapmixQTL mapping: empirical p-values for phenotypes')
    logger.write(f'  * {N} samples')
    logger.write(f'  * {A_df.shape[0]} phenotypes')

    robust = se_mode == 'robust'

    if covariates_df is not None:
        assert covariates_df.index.equals(A_df.columns), \
            'Sample names in phenotype columns and covariate rows must match'
        logger.write(f'  * {covariates_df.shape[1]} covariates')
        covariates_t = torch.tensor(covariates_df.values, dtype=torch.float32).to(device)
        n_cov = covariates_df.shape[1]
    else:
        covariates_t = None
        n_cov = 0

    has_phase = xL_df is not None and xR_df is not None
    if has_phase:
        logger.write('  * phase genotypes available (ASE + total channels)')
    else:
        logger.write('  * no phase genotypes (total channel only)')

    logger.write(f'  * {variant_df.shape[0]} variants')
    if maf_threshold > 0:
        logger.write(f'  * applying in-sample {maf_threshold} MAF filter')
    logger.write(f'  * cis-window: ±{window:,}')

    dof = N - 2 - n_cov

    genotype_ix = np.array([genotype_df.columns.tolist().index(i) for i in samples])
    genotype_ix_t = torch.from_numpy(genotype_ix).to(device)

    n_samples = N
    ix = np.arange(n_samples)
    if seed is not None:
        logger.write(f'  * using seed {seed}')
        np.random.seed(seed)
    permutation_ix_t = torch.LongTensor(
        np.array([np.random.permutation(ix) for _ in range(nperm)])
    ).to(device)

    igc = genotypeio.InputGeneratorCis(
        genotype_df, variant_df, T_df, phenotype_pos_df, window=window,
    )
    if igc.n_phenotypes == 0:
        raise ValueError('No valid phenotypes found.')
    pheno_ix = {pid: i for i, pid in enumerate(A_df.index)}

    res_df = []
    start_time = time.time()
    logger.write('  * computing permutations')
    for k, (_, genotypes, genotype_range, phenotype_id) in enumerate(
        igc.generate_data(verbose=verbose), 1
    ):
        if phenotype_id not in pheno_ix:
            continue

        pidx = pheno_ix[phenotype_id]
        a_t = torch.tensor(A_df.values[pidx], dtype=torch.float32).to(device)
        t_t = torch.tensor(T_df.values[pidx], dtype=torch.float32).to(device)
        va_t = torch.tensor(Va_df.values[pidx], dtype=torch.float32).to(device).clamp(min=1e-8)
        vt_t = torch.tensor(Vt_df.values[pidx], dtype=torch.float32).to(device).clamp(min=1e-8)

        if tau_mode == 'estimate':
            tau_a = _estimate_tau(a_t, va_t, covariates_t, device)
            tau_t_val = _estimate_tau(t_t, vt_t, covariates_t, device)
            sqrt_wa_t = torch.sqrt(1.0 / (va_t + tau_a))
            sqrt_wt_t = torch.sqrt(1.0 / (vt_t + tau_t_val))
        else:
            sqrt_wa_t = torch.sqrt(1.0 / va_t)
            sqrt_wt_t = torch.sqrt(1.0 / vt_t)

        residualizer_a = WeightedResidualizer(covariates_t, sqrt_wa_t)
        residualizer_tc = WeightedResidualizer(covariates_t, sqrt_wt_t)

        genotypes_t = torch.tensor(genotypes, dtype=torch.float32).to(device)
        genotypes_t = genotypes_t[:, genotype_ix_t]
        impute_mean(genotypes_t)

        # Build the signed het indicator for the full (contiguous) cis window
        # BEFORE any filtering, so that every subsequent mask applies to
        # genotypes and phase identically (genotype_range is contiguous here).
        if has_phase:
            xL_vals = xL_df.values[genotype_range[0]:genotype_range[-1] + 1]
            xR_vals = xR_df.values[genotype_range[0]:genotype_range[-1] + 1]
            xL_t = torch.tensor(xL_vals, dtype=torch.float32).to(device)[:, genotype_ix_t]
            xR_t = torch.tensor(xR_vals, dtype=torch.float32).to(device)[:, genotype_ix_t]
            sign_t = xL_t - xR_t
        else:
            sign_t = torch.zeros_like(genotypes_t)

        if maf_threshold > 0:
            maf_t = calculate_maf(genotypes_t)
            mask_t = maf_t >= maf_threshold
            genotypes_t = genotypes_t[mask_t]
            sign_t = sign_t[mask_t]
            genotype_range = genotype_range[mask_t.cpu().numpy().astype(bool)]

        mono_t = (genotypes_t == genotypes_t[:, [0]]).all(1)
        if mono_t.any():
            genotypes_t = genotypes_t[~mono_t]
            sign_t = sign_t[~mono_t]
            genotype_range = genotype_range[~mono_t.cpu().numpy().astype(bool)]
            if warn_monomorphic:
                logger.write(
                    f'    * WARNING: excluding {mono_t.sum()} monomorphic variants'
                )

        if genotypes_t.shape[0] == 0:
            logger.write(f'WARNING: skipping {phenotype_id} (no valid variants)')
            continue

        res = calculate_hapmixqtl_permutations(
            genotypes_t, sign_t, a_t, t_t,
            sqrt_wa_t, sqrt_wt_t,
            residualizer_a, residualizer_tc,
            permutation_ix_t,
        )
        r_nominal, std_ratio, var_ix, r2_perm, g = [i.cpu().numpy() for i in res]
        var_ix = genotype_range[var_ix]

        variant_id = variant_df.index[var_ix]
        start_distance = variant_df['pos'].values[var_ix] - igc.phenotype_start[phenotype_id]
        end_distance = variant_df['pos'].values[var_ix] - igc.phenotype_end[phenotype_id]

        r2_nominal = r_nominal * r_nominal
        pval_perm = (np.sum(r2_perm >= r2_nominal) + 1) / (nperm + 1)

        slope = r_nominal * std_ratio
        tstat2 = dof * r2_nominal / (1 - r2_nominal) if r2_nominal < 1 else np.inf
        slope_se = np.abs(slope) / np.sqrt(tstat2) if tstat2 > 0 else np.inf

        n2 = 2 * len(g)
        af = np.sum(g) / n2
        if af <= 0.5:
            ma_samples = np.sum(g > 0.5)
            ma_count = np.sum(g[g > 0.5])
        else:
            ma_samples = np.sum(g < 1.5)
            ma_count = n2 - np.sum(g[g > 0.5])

        res_s = pd.Series(OrderedDict([
            ('num_var', genotypes_t.shape[0]),
            ('beta_shape1', np.nan),
            ('beta_shape2', np.nan),
            ('true_df', np.nan),
            ('pval_true_df', np.nan),
            ('variant_id', variant_id),
            ('start_distance', start_distance),
            ('end_distance', end_distance),
            ('ma_samples', ma_samples),
            ('ma_count', ma_count),
            ('af', af),
            ('pval_nominal', pval_from_corr(r2_nominal, dof)),
            ('slope', slope),
            ('slope_se', slope_se),
            ('pval_perm', pval_perm),
            ('pval_beta', np.nan),
        ]), name=phenotype_id)

        if beta_approx:
            try:
                res_s[['pval_beta', 'beta_shape1', 'beta_shape2',
                       'true_df', 'pval_true_df']] = \
                    calculate_beta_approx_pval(r2_perm, r2_nominal, dof)
            except Exception:
                pass

        res_df.append(res_s)

    res_df = pd.concat(res_df, axis=1, sort=False).T
    res_df.index.name = 'phenotype_id'
    logger.write(f'  Time elapsed: {(time.time() - start_time) / 60:.2f} min')
    logger.write('done.')
    return res_df.astype(output_dtype_dict).infer_objects()


# ---------------------------------------------------------------------------
#  SuSiE fine-mapping
# ---------------------------------------------------------------------------

def _build_stacked_design(genotypes_t, sign_t, a_t, t_t,
                          sqrt_wa_t, sqrt_wt_t,
                          residualizer_a, residualizer_t):
    """
    Build the stacked, whitened, covariate-residualized design for SuSiE.

    hapmixQTL's Method A shares a single effect ``beta`` (the log allelic fold
    change) across two channels: the ASE channel regresses ``a`` on the signed
    het indicator ``s``, and the total channel regresses ``t`` on the half
    dosage ``g/2``. The sqrt-weight transform (multiply response and predictors
    by ``sqrt(w_i)`` with ``w_i = 1/(v_inf_i + tau)``) whitens each channel to
    unit-variance, homoskedastic noise -- exactly the model SuSiE assumes
    (``estimate_residual_variance=False, residual_variance=1``).

    Because the two channels estimate the *same* per-variant effect, we can
    stack them into one regression with 2N pseudo-samples and a single design
    matrix. ``WeightedResidualizer`` has already projected out the weighted
    intercept and covariates from each channel, so the stacked responses and
    predictors are covariate-free (SuSiE is then called with
    ``intercept=False``).

    Returns:
        X_aug_t: [2N, p] stacked predictors (variants as columns)
        y_aug_t: [2N, 1] stacked response
    """
    # ASE channel (sqrt-weighted + residualized)
    s_star_res = residualizer_a.transform(sign_t * sqrt_wa_t.unsqueeze(0))      # p x N
    a_star_res = residualizer_a.transform((a_t * sqrt_wa_t).unsqueeze(0))       # 1 x N
    # Total channel (sqrt-weighted + residualized)
    g_half_star_res = residualizer_t.transform((genotypes_t / 2) * sqrt_wt_t.unsqueeze(0))  # p x N
    t_star_res = residualizer_t.transform((t_t * sqrt_wt_t).unsqueeze(0))       # 1 x N

    # Stack the two channels along the sample axis -> 2N pseudo-samples.
    X_aug_t = torch.cat([s_star_res, g_half_star_res], dim=1).T                 # (2N) x p
    y_aug_t = torch.cat([a_star_res, t_star_res], dim=1).reshape(-1, 1)         # (2N) x 1
    return X_aug_t, y_aug_t


def _build_knockoff_stacked_design(xkL_t, xkR_t, sqrt_wa_t, sqrt_wt_t,
                                   residualizer_a, residualizer_t):
    """
    Build the [2N, p] stacked knockoff design for a phased (Route 2) knockoff.

    The knockoff enters BOTH channels COHERENTLY: a single pair of knockoff
    haplotypes (x~L, x~R) produces the knockoff ASE predictor s~ = x~L - x~R and
    the knockoff total predictor g~/2 = (x~L + x~R)/2, whitened and residualized
    exactly like the real channels in _build_stacked_design. This coherence --
    the same knockoff haplotypes driving both the allelic-contrast and total
    channels -- is why phased (Route 2) knockoffs are the right construction for
    the two-channel model: an independent knockoff per channel would not respect
    the shared per-variant effect the model estimates.

    Args:
        xkL_t, xkR_t: [p, N] knockoff haplotype allele matrices (variant x
            sample), same window/order as the real design.
        sqrt_wa_t, sqrt_wt_t, residualizer_a, residualizer_t: the SAME weights
            and covariate residualizers used for the real design.

    Returns:
        Xk_aug_t: [2N, p] stacked knockoff predictors.
    """
    sign_k_t = xkL_t - xkR_t                                          # p x N
    geno_k_t = xkL_t + xkR_t                                          # p x N
    s_k_res = residualizer_a.transform(sign_k_t * sqrt_wa_t.unsqueeze(0))       # p x N
    g_k_half_res = residualizer_t.transform((geno_k_t / 2) * sqrt_wt_t.unsqueeze(0))  # p x N
    Xk_aug_t = torch.cat([s_k_res, g_k_half_res], dim=1).T                      # (2N) x p
    return Xk_aug_t


def map_susie(genotype_df, variant_df, A_df, T_df, Va_df, Vt_df,
              phenotype_pos_df, xL_df=None, xR_df=None,
              covariates_df=None, L=10, scaled_prior_variance=0.2,
              estimate_residual_variance=False, estimate_prior_variance=True,
              coverage=0.95, min_abs_corr=0.5, maf_threshold=0,
              tau_mode='zero', max_iter=500, window=1000000, tol=1e-3,
              summary_only=True, logger=None, verbose=True,
              warn_monomorphic=False):
    """
    hapmixQTL SuSiE fine-mapping.

    For each phenotype, fine-maps the shared log-aFC effect using the combined
    ASE + total evidence. The two sqrt-weighted, covariate-residualized
    channels are stacked into a single whitened design and passed to
    ``tensorqtl.susie.susie`` unchanged, so any improvement to the core SuSiE
    implementation is inherited automatically.

    ``estimate_residual_variance`` defaults to ``False`` (with an implied
    residual variance of 1): the sqrt-weight transform already whitens the
    noise to unit variance using the *known* Gibbs inferential variances, which
    is consistent with the known-variance GLS standard errors used elsewhere in
    this module. Set it to ``True`` to let SuSiE re-estimate a scalar
    dispersion instead (matching the default individual-level ``susie.map``).

    Args mirror ``susie.map``; hapmixQTL-specific inputs (A/T/Va/Vt and the
    optional phase matrices xL/xR) match ``map_cis``.

    Returns:
        summary_df (if summary_only) or (summary_df, susie_res dict), analogous
        to ``susie.map``.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if logger is None:
        logger = SimpleLogger()

    samples = A_df.columns
    N = len(samples)
    assert A_df.columns.equals(T_df.columns)
    assert A_df.index.equals(T_df.index)

    logger.write('hapmixQTL SuSiE fine-mapping')
    logger.write(f'  * {N} samples')
    logger.write(f'  * {A_df.shape[0]} phenotypes')

    if covariates_df is not None:
        assert covariates_df.index.equals(A_df.columns), \
            'Sample names in phenotype columns and covariate rows must match'
        logger.write(f'  * {covariates_df.shape[1]} covariates')
        covariates_t = torch.tensor(covariates_df.values, dtype=torch.float32).to(device)
        # Unweighted residualizer for genotype LD used in credible-set purity
        # (see below): purity should reflect genotype correlation, not the
        # sqrt-weighted stacked design that SuSiE is fit on.
        ld_residualizer = Residualizer(covariates_t)
    else:
        covariates_t = None
        ld_residualizer = None

    has_phase = xL_df is not None and xR_df is not None
    if has_phase:
        logger.write('  * phase genotypes available (ASE + total channels)')
    else:
        logger.write('  * no phase genotypes (total channel only)')

    logger.write(f'  * {variant_df.shape[0]} variants')
    if maf_threshold > 0:
        logger.write(f'  * applying in-sample MAF >= {maf_threshold} filter')
    logger.write(f'  * cis-window: ±{window:,}')
    logger.write(f'  * max effects (L): {L}')

    genotype_ix = np.array([genotype_df.columns.tolist().index(i) for i in samples])
    genotype_ix_t = torch.from_numpy(genotype_ix).to(device)

    igc = genotypeio.InputGeneratorCis(
        genotype_df, variant_df, T_df, phenotype_pos_df, window=window,
    )
    if igc.n_phenotypes == 0:
        raise ValueError('No valid phenotypes found.')
    pheno_ix = {pid: i for i, pid in enumerate(A_df.index)}

    copy_keys = ['pip', 'sets', 'converged', 'elbo', 'niter', 'lbf_variable']
    susie_summary = []
    susie_res = {} if not summary_only else None

    start_time = time.time()
    logger.write('  * fine-mapping')
    for k, (_, genotypes, genotype_range, phenotype_id) in enumerate(
        igc.generate_data(verbose=verbose), 1
    ):
        if phenotype_id not in pheno_ix:
            continue

        pidx = pheno_ix[phenotype_id]
        a_t = torch.tensor(A_df.values[pidx], dtype=torch.float32).to(device)
        t_t = torch.tensor(T_df.values[pidx], dtype=torch.float32).to(device)
        va_t = torch.tensor(Va_df.values[pidx], dtype=torch.float32).to(device).clamp(min=1e-8)
        vt_t = torch.tensor(Vt_df.values[pidx], dtype=torch.float32).to(device).clamp(min=1e-8)

        if tau_mode == 'estimate':
            tau_a = _estimate_tau(a_t, va_t, covariates_t, device)
            tau_t_val = _estimate_tau(t_t, vt_t, covariates_t, device)
            sqrt_wa_t = torch.sqrt(1.0 / (va_t + tau_a))
            sqrt_wt_t = torch.sqrt(1.0 / (vt_t + tau_t_val))
        else:
            sqrt_wa_t = torch.sqrt(1.0 / va_t)
            sqrt_wt_t = torch.sqrt(1.0 / vt_t)

        residualizer_a = WeightedResidualizer(covariates_t, sqrt_wa_t)
        residualizer_tc = WeightedResidualizer(covariates_t, sqrt_wt_t)

        genotypes_t = torch.tensor(genotypes, dtype=torch.float32).to(device)
        genotypes_t = genotypes_t[:, genotype_ix_t]
        impute_mean(genotypes_t)

        variant_ids = variant_df.index[genotype_range[0]:genotype_range[-1] + 1].rename('variant_id')

        # Build phase-derived sign over the contiguous window before filtering,
        # so masks apply identically to genotypes and phase (see map_cis).
        if has_phase:
            xL_vals = xL_df.values[genotype_range[0]:genotype_range[-1] + 1]
            xR_vals = xR_df.values[genotype_range[0]:genotype_range[-1] + 1]
            xL_t = torch.tensor(xL_vals, dtype=torch.float32).to(device)[:, genotype_ix_t]
            xR_t = torch.tensor(xR_vals, dtype=torch.float32).to(device)[:, genotype_ix_t]
            sign_t = xL_t - xR_t
        else:
            sign_t = torch.zeros_like(genotypes_t)

        # filter monomorphic (and optionally low-MAF) variants
        mask_t = ~(genotypes_t == genotypes_t[:, [0]]).all(1)
        if warn_monomorphic and (~mask_t).any():
            logger.write(f'    * WARNING: excluding {int((~mask_t).sum())} monomorphic variants')
        if maf_threshold > 0:
            maf_t = calculate_maf(genotypes_t)
            mask_t &= maf_t >= maf_threshold
        if not mask_t.all():
            genotypes_t = genotypes_t[mask_t]
            sign_t = sign_t[mask_t]
            mask = mask_t.cpu().numpy().astype(bool)
            variant_ids = variant_ids[mask]

        if genotypes_t.shape[0] == 0:
            logger.write(f'WARNING: skipping {phenotype_id} (no valid variants)')
            continue

        X_aug_t, y_aug_t = _build_stacked_design(
            genotypes_t, sign_t, a_t, t_t,
            sqrt_wa_t, sqrt_wt_t, residualizer_a, residualizer_tc,
        )

        # susie() applies torch operations to residual_variance, so it must be
        # a tensor. When estimating it, pass None (susie initializes it to
        # var(y)); otherwise fix it to 1 (the channels are whitened to unit
        # variance by the sqrt-weight transform).
        if estimate_residual_variance:
            resvar = None
        else:
            resvar = torch.tensor(1.0, dtype=torch.float32, device=device)

        res = susie.susie(
            X_aug_t, y_aug_t, L=L,
            scaled_prior_variance=scaled_prior_variance,
            intercept=False,  # channels already covariate-residualized
            estimate_residual_variance=estimate_residual_variance,
            estimate_prior_variance=estimate_prior_variance,
            residual_variance=resvar,
            coverage=coverage, min_abs_corr=min_abs_corr,
            tol=tol, max_iter=max_iter,
        )

        # Recompute credible sets using genotype LD for the purity filter.
        # susie() measured purity as correlation across the stacked, whitened,
        # 2N-row design (X_aug_t) -- that is not genotype LD, and min_abs_corr
        # is conventionally a genotype-correlation threshold (in the no-phase
        # case the ASE half is all zeros, which especially distorts it). Pass
        # the (covariate-residualized) dosage correlation as Xcorr instead so
        # the purity threshold means what users expect.
        if ld_residualizer is not None:
            geno_ld_t = ld_residualizer.transform(genotypes_t)
        else:
            geno_ld_t = genotypes_t
        Xcorr_t = susie.corrcoef(geno_ld_t)
        res['sets'] = susie.susie_get_cs(
            res, Xcorr=Xcorr_t, coverage=coverage, min_abs_corr=min_abs_corr,
        )

        af_t = genotypes_t.sum(1) / (2 * genotypes_t.shape[1])
        res['pip'] = pd.DataFrame(
            {'pip': res['pip'], 'af': af_t.cpu().numpy()}, index=variant_ids
        )
        if res['sets']['cs'] is not None:
            if res['converged']:
                for c in sorted(res['sets']['cs'], key=lambda x: int(x.replace('L', ''))):
                    cs = res['sets']['cs'][c]
                    p = res['pip'].iloc[cs].copy().reset_index()
                    p['cs_id'] = c.replace('L', '')
                    p.insert(0, 'phenotype_id', phenotype_id)
                    susie_summary.append(p)
                res['lbf_variable'] = res['lbf_variable'][res['sets']['cs_index']]
            else:
                logger.write(f'    * WARNING: {phenotype_id} did not converge')

        if not summary_only:
            susie_res[phenotype_id] = {key: res[key] for key in copy_keys}

    logger.write(f'  Time elapsed: {(time.time() - start_time) / 60:.2f} min')
    logger.write('done.')

    if susie_summary:
        susie_summary = pd.concat(susie_summary, axis=0).rename(
            columns={'snp': 'variant_id'}
        ).reset_index(drop=True)
    else:
        susie_summary = pd.DataFrame(
            columns=['phenotype_id', 'variant_id', 'pip', 'af', 'cs_id']
        )

    if summary_only:
        return susie_summary
    else:
        drop_ids = [key for key in susie_res if susie_res[key]['sets']['cs'] is None]
        for key in drop_ids:
            del susie_res[key]
        return susie_summary, susie_res


def map_egenes_knockoffs(genotype_df, variant_df, A_df, T_df, Va_df, Vt_df,
                         phenotype_pos_df, xL_df, xR_df, covariates_df=None,
                         fdr=0.1, n_knockoffs=20, hmm_K=8, hmm_em_iter=25,
                         coherent=True, hmm_params=None,
                         gene_stat='max', selection='calibrated',
                         knockoff_offset=1, dependence='prds',
                         L=10, scaled_prior_variance=0.2,
                         estimate_residual_variance=False,
                         estimate_prior_variance=True,
                         coverage=0.95, min_abs_corr=0.5, maf_threshold=0,
                         tau_mode='zero', max_iter=500, window=1000000, tol=1e-3,
                         seed=0, logger=None, verbose=True):
    """
    hapmixQTL knockoff-calibrated eGene FDR using PHASED (Route 2) knockoffs.

    This is the two-channel analog of ``susie.map_egenes_knockoffs``. For each
    gene it draws ``n_knockoffs`` phased haplotype knockoffs (x~L, x~R) under a
    per-gene haplotype HMM (``knockoffs.haplotype_hmm_knockoffs``), builds the
    augmented two-channel stacked design ``[X, X~]`` (each of the ASE and total
    channels gets its knockoff columns from the SAME knockoff haplotypes -- the
    coherence phased knockoffs provide), fits SuSiE on it, and forms the
    swap-antisymmetric gene statistic ``W_g = maxPIP(orig) - maxPIP(knockoff)``.
    The per-gene W across the M draws then feeds the same step-2/step-3
    calibration used for standard SuSiE, so eGene FDR is calibrated identically.

    Phase (``xL_df``, ``xR_df``) is REQUIRED: the knockoff must respect the
    allelic-contrast channel, which only exists with phase.

    ``coherent`` (default True) fits ONE haplotype HMM per chromosome and draws M
    knockoff copies of the whole chromosome up front (PASS 0), then slices each
    gene's cis-window out of them -- so two genes with overlapping windows share
    the SAME knockoff haplotypes on their shared variants. That cross-gene
    coherence is what makes the per-gene knockoff p-values (step 2) mutually
    comparable and is the prerequisite for the genome-wide dependence analysis;
    it also amortizes one HMM fit over all genes on a chromosome. ``coherent=False``
    falls back to an independent per-gene HMM fit (fine when windows do not
    overlap, e.g. one gene per locus). Coherent mode requires each chromosome's
    variants to be a contiguous, ordered block in ``genotype_df`` (sort by
    chrom, pos) -- the standard tensorQTL layout.

    Args (knockoff-specific; the rest mirror ``map_susie``):
        fdr: target genome-wide eGene FDR.
        n_knockoffs: number of phased knockoff draws M (>= ~20 for 'calibrated').
        hmm_K: haplotype clusters for the haplotype HMM.
        hmm_em_iter: Baum-Welch iterations for the HMM fit.
        coherent: True (chromosome-coherent draws, default) or False (per-gene).
        hmm_params: optional dict chrom -> pre-fit haplotype HMM params
            {'init_p','Q','emission_p'} to skip EM (coherent mode).
        gene_stat: 'max' or 'sum' for gene_level_W.
        selection: 'calibrated' (default; known-null Storey q-value + mirror
            cross-check, ko.select_egenes_calibrated), 'qvalue', 'pvalue', or
            'ebh'. See susie.map_egenes_knockoffs.
        knockoff_offset: offset for the count/threshold conventions.
        dependence: cross-gene dependence assumption for the genome-wide
            selection ('prds' default | 'ind' | 'arbitrary'; see
            knockoffs.bh_select).

    Returns:
        (egene_df, diagnostics):
          egene_df: DataFrame [phenotype_id, qvalue/score, selected].
          diagnostics: dict with 'W_per_draw', 'gene_ids', 'n_draws', 'selection',
            and (for 'calibrated') 'pi0', 'agreement', 'pi0_interval'.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if logger is None:
        logger = SimpleLogger()

    samples = A_df.columns
    N = len(samples)
    assert xL_df is not None and xR_df is not None, \
        "map_egenes_knockoffs requires phased haplotypes (xL_df, xR_df)"
    assert A_df.columns.equals(T_df.columns) and A_df.index.equals(T_df.index)

    logger.write('hapmixQTL knockoff-calibrated eGene mapping (Route 2, phased)')
    logger.write(f'  * {N} samples, {A_df.shape[0]} phenotypes')
    logger.write(f'  * {n_knockoffs} phased knockoff draws '
                 f'({"chromosome-coherent" if coherent else "per-gene"}); HMM K={hmm_K}; '
                 f'selection={selection}, FDR<={fdr}, dependence={dependence}')

    if covariates_df is not None:
        assert covariates_df.index.equals(A_df.columns)
        covariates_t = torch.tensor(covariates_df.values, dtype=torch.float32).to(device)
    else:
        covariates_t = None

    genotype_ix = np.array([genotype_df.columns.tolist().index(i) for i in samples])
    genotype_ix_t = torch.from_numpy(genotype_ix).to(device)

    igc = genotypeio.InputGeneratorCis(
        genotype_df, variant_df, T_df, phenotype_pos_df, window=window)
    if igc.n_phenotypes == 0:
        raise ValueError('No valid phenotypes found.')
    pheno_ix = {pid: i for i, pid in enumerate(A_df.index)}

    resvar = None if estimate_residual_variance else \
        torch.tensor(1.0, dtype=torch.float32, device=device)

    # PASS 0 (coherent): fit one haplotype HMM per chromosome and draw M coherent
    # phased knockoff copies of the whole chromosome. Each gene later slices its
    # cis-window out of these, so overlapping genes share the SAME knockoff
    # haplotypes on shared variants (the coherence the per-gene fit cannot give).
    xkL_by_chrom = xkR_by_chrom = chrom_row_offset = None
    if coherent:
        logger.write(f'  * PASS 0: per-chromosome haplotype HMM fit + '
                     f'{n_knockoffs} coherent phased knockoff draw(s) (K={hmm_K})')

        def _hap_in_pheno_order(df_values, lo, hi):
            V = df_values[lo:hi][:, genotype_ix].astype(np.float64)   # [p_c, N]
            if np.isnan(V).any():
                rm = np.nanmean(V, axis=1, keepdims=True)
                rm = np.where(np.isnan(rm), 0.0, rm)
                V = np.where(np.isnan(V), rm, V)
            return np.rint(V).T.clip(0, 1).astype(np.int64)           # [N, p_c]

        chrom_arr = variant_df['chrom'].values
        xkL_by_chrom, xkR_by_chrom, chrom_row_offset = {}, {}, {}
        for c in pd.unique(chrom_arr):
            rows = np.where(chrom_arr == c)[0]
            if not (rows[-1] - rows[0] + 1 == rows.size and (np.diff(rows) == 1).all()):
                raise ValueError(f"chromosome {c} variants are not a contiguous "
                                 "ordered block in genotype_df; sort by chrom,pos "
                                 "or use coherent=False.")
            lo, hi = rows[0], rows[-1] + 1
            xL_c = _hap_in_pheno_order(xL_df.values, lo, hi)
            xR_c = _hap_in_pheno_order(xR_df.values, lo, hi)
            cparams = None if hmm_params is None else hmm_params.get(c, None)
            _, (xkL_c, xkR_c) = ko.chromosome_hmm_knockoffs(
                K=hmm_K, M=n_knockoffs, n_em_iter=hmm_em_iter,
                seed=seed * 100003 + int(lo), method='haplotype',
                xL=xL_c, xR=xR_c, params=cparams, return_phased=True)
            xkL_by_chrom[c], xkR_by_chrom[c] = xkL_c, xkR_c   # [M, N, p_c] each
            chrom_row_offset[c] = rows[0]
        logger.write(f'  * PASS 0 done: {len(xkL_by_chrom)} chromosome(s)')

    gene_ids, W_rows = [], []
    start_time = time.time()
    for k, (_, genotypes, genotype_range, phenotype_id) in enumerate(
        igc.generate_data(verbose=verbose), 1
    ):
        if phenotype_id not in pheno_ix:
            continue
        pidx = pheno_ix[phenotype_id]
        a_t = torch.tensor(A_df.values[pidx], dtype=torch.float32).to(device)
        t_t = torch.tensor(T_df.values[pidx], dtype=torch.float32).to(device)
        va_t = torch.tensor(Va_df.values[pidx], dtype=torch.float32).to(device).clamp(min=1e-8)
        vt_t = torch.tensor(Vt_df.values[pidx], dtype=torch.float32).to(device).clamp(min=1e-8)

        if tau_mode == 'estimate':
            tau_a = _estimate_tau(a_t, va_t, covariates_t, device)
            tau_tv = _estimate_tau(t_t, vt_t, covariates_t, device)
            sqrt_wa_t = torch.sqrt(1.0 / (va_t + tau_a))
            sqrt_wt_t = torch.sqrt(1.0 / (vt_t + tau_tv))
        else:
            sqrt_wa_t = torch.sqrt(1.0 / va_t)
            sqrt_wt_t = torch.sqrt(1.0 / vt_t)

        residualizer_a = WeightedResidualizer(covariates_t, sqrt_wa_t)
        residualizer_tc = WeightedResidualizer(covariates_t, sqrt_wt_t)

        genotypes_t = torch.tensor(genotypes, dtype=torch.float32).to(device)
        genotypes_t = genotypes_t[:, genotype_ix_t]
        impute_mean(genotypes_t)

        # Phased haplotype window (contiguous), before filtering.
        xL_vals = xL_df.values[genotype_range[0]:genotype_range[-1] + 1]
        xR_vals = xR_df.values[genotype_range[0]:genotype_range[-1] + 1]
        xL_t = torch.tensor(xL_vals, dtype=torch.float32).to(device)[:, genotype_ix_t]
        xR_t = torch.tensor(xR_vals, dtype=torch.float32).to(device)[:, genotype_ix_t]
        sign_t = xL_t - xR_t

        # filter monomorphic (and optionally low-MAF) variants, masking phase too
        mask_t = ~(genotypes_t == genotypes_t[:, [0]]).all(1)
        if maf_threshold > 0:
            mask_t &= calculate_maf(genotypes_t) >= maf_threshold
        mask_applied = None
        if not mask_t.all():
            genotypes_t = genotypes_t[mask_t]
            sign_t = sign_t[mask_t]
            xL_t = xL_t[mask_t]
            xR_t = xR_t[mask_t]
            mask_applied = mask_t.cpu().numpy().astype(bool)
        if genotypes_t.shape[0] == 0:
            continue

        X_aug_t, y_aug_t = _build_stacked_design(
            genotypes_t, sign_t, a_t, t_t,
            sqrt_wa_t, sqrt_wt_t, residualizer_a, residualizer_tc)

        if coherent:
            # Slice this gene's cis-window out of the chromosome-coherent phased
            # draws, then apply the SAME variant mask used on the real design.
            c = variant_df['chrom'].values[genotype_range[0]]
            local = np.asarray(genotype_range) - chrom_row_offset[c]
            xkL_win = xkL_by_chrom[c][:, :, local]        # [M, N, p_window]
            xkR_win = xkR_by_chrom[c][:, :, local]
            if mask_applied is not None:
                xkL_win = xkL_win[:, :, mask_applied]
                xkR_win = xkR_win[:, :, mask_applied]
            xkL_all, xkR_all = xkL_win, xkR_win           # [M, N, p]
        else:
            # Fit the per-gene haplotype HMM once, draw M phased knockoffs.
            xL_np = xL_t.T.cpu().numpy().astype(np.int64)     # [N, p] in {0,1}
            xR_np = xR_t.T.cpu().numpy().astype(np.int64)
            _, (xkL_all, xkR_all) = ko.haplotype_hmm_knockoffs(
                xL_np, xR_np, K=hmm_K, M=n_knockoffs, n_em_iter=hmm_em_iter,
                seed=seed * 100003 + k, return_phased=True)

        draw_W = []
        for r in range(n_knockoffs):
            xkL_t = torch.tensor(xkL_all[r].T, dtype=torch.float32).to(device)  # [p, N]
            xkR_t = torch.tensor(xkR_all[r].T, dtype=torch.float32).to(device)
            Xk_aug_t = _build_knockoff_stacked_design(
                xkL_t, xkR_t, sqrt_wa_t, sqrt_wt_t,
                residualizer_a, residualizer_tc)
            res, p = ko.augmented_susie_fit(
                susie.susie, X_aug_t, y_aug_t, Xk_aug_t, L,
                scaled_prior_variance=scaled_prior_variance, intercept=False,
                estimate_residual_variance=estimate_residual_variance,
                estimate_prior_variance=estimate_prior_variance,
                residual_variance=resvar, coverage=coverage,
                min_abs_corr=min_abs_corr, tol=tol, max_iter=max_iter)
            draw_W.append(ko.gene_level_W(res['pip'], p, kind=gene_stat))
        gene_ids.append(phenotype_id)
        W_rows.append(draw_W)

    if not gene_ids:
        raise ValueError('No genes produced statistics.')
    W_per_draw = np.array(W_rows, dtype=np.float64).T          # [M, n_genes]

    diagnostics = {'W_per_draw': W_per_draw, 'gene_ids': gene_ids,
                   'n_draws': W_per_draw.shape[0], 'selection': selection}
    if selection == 'calibrated':
        sel = ko.select_egenes_calibrated(gene_ids, W_per_draw, q=fdr,
                                          offset=(knockoff_offset or 1),
                                          dependence=dependence)
        selected = set(sel['selected'])
        score_col, score_vals = 'qvalue', sel['qvalues']
        diagnostics.update(pi0=sel['pi0'], agreement=sel['agreement'],
                           mirror_informative=sel.get('mirror_informative'),
                           pi0_interval=(sel['lfdr']['pi0_lo'], sel['lfdr']['pi0_hi']))
        logger.write(f'  * pi0={sel["pi0"]:.3f}; mirror cross-check selected '
                     f'{sel["mirror"]["n_selected"]} (agreement={sel["agreement"]:.2f}'
                     f'{"" if sel.get("mirror_informative") else ", below mirror detection floor"})')
        if sel.get('mirror_informative') and sel['agreement'] < 0.5 and selected:
            logger.write('  ! WARNING: q-value and pi0-free mirror disagree '
                         '(agreement<0.5, above the mirror detection floor) -- '
                         'possible knockoff misspecification.')
    elif selection == 'pvalue':
        sel = ko.select_egenes_pvalue(gene_ids, W_per_draw, q=fdr,
                                      offset=(knockoff_offset or 1),
                                      dependence=dependence)
        selected = set(sel['selected'])
        score_col, score_vals = 'pvalue', sel['pvalues']
    elif selection == 'ebh':
        sel = ko.select_egenes(gene_ids, W_per_draw, q=fdr,
                               offset=(knockoff_offset or 1))
        selected = set(sel['selected'])
        score_col, score_vals = 'evalue', sel['evalues']
    else:  # 'qvalue'
        sel = ko.select_egenes_qvalue(gene_ids, W_per_draw, q=fdr,
                                      offset=knockoff_offset)
        selected = set(sel['selected'])
        score_col, score_vals = 'qvalue', sel['qvalues']

    logger.write(f'  * {len(selected)}/{len(gene_ids)} genes selected as eGenes')
    logger.write(f'  Time elapsed: {(time.time()-start_time)/60:.2f} min')
    egene_df = pd.DataFrame({
        'phenotype_id': gene_ids,
        score_col: score_vals,
        'selected': [g in selected for g in gene_ids],
    })
    return egene_df, diagnostics

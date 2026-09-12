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

Cat, the a-t inferential covariance, is INTENTIONALLY UNUSED.
compute_summaries_from_gibbs returns it and read_hapmixqtl_inputs / --hap_Cat
accept it, for completeness and so it can be inspected, but no mapping function
consumes it. The scalar inverse-variance meta-analysis assumes the two channel
estimators are independent, and they are, even when the a and t noise is
strongly correlated: the ASE predictor s = xL - xR is orthogonal to the total
predictor g/2 under random phase (E[s | g=1] = 0), so the two regressions
project any shared noise onto orthogonal directions. Measured:
corr(beta_a, beta_t) = +0.01 / -0.03 / -0.02 at a-t noise correlation
rho = 0 / 0.5 / 0.9 (95% CIs all cover zero), unchanged under 50% phasing error
(docs/ase_validation.md sec 3 and 7f; tests/test_hapmixqtl_calibration.py
asserts it on every push). Adding a 2*w_a*w_t*Cat term to the combined SE would
change a statistic that is correct as written -- do not "fix" this.
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
        Cat_df: inferential covariance a,t [phenotypes x samples] (or None).
                Loaded for inspection only: it is INTENTIONALLY UNUSED by every
                mapping function (see the module docstring for why).
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


def _zero_degenerate_ase_weights(sqrt_wa_t, va_t, eps=1e-12):
    """Zero the ASE weight of samples carrying NO allele-specific information.

    A sample with no reads over any heterozygous feature SNP has a degenerate
    Gibbs posterior: every draw is identical, so v_inf is EXACTLY 0 and its
    allelic contrast a = log(0+kappa) - log(0+kappa) is exactly 0. Weighting by
    1/v_inf then hands those samples the LARGEST weight in the dataset (1/eps)
    while they carry zero information, and their a = 0 drags the ASE slope
    toward the null.

    Measured: with 24 of 80 samples lacking allele-specific coverage, they
    received weights ~1e8 against a median ~6.7 for informative samples -- about
    1e7 times more -- and hapmixQTL's power fell BELOW total-counts-only.
    tau_mode='estimate' softens but does not fix this: the weight becomes 1/tau,
    still the largest in the dataset.

    The correct weight for a sample with no information is zero.
    """
    keep = va_t > eps
    return torch.where(keep, sqrt_wa_t, torch.zeros_like(sqrt_wa_t))


def reference_bias_diagnostic(yL, yR, sign, min_total=10, min_sites=20):
    """
    Detect reference mapping bias from haplotype counts (RASQUAL's phi).

    WHY THIS MATTERS
    ----------------
    hapmixQTL does not model reference mapping bias. It has no analogue of
    RASQUAL's phi, and it degrades catastrophically rather than gracefully when
    bias is present: at phi = 0.60 the nominal type-I error rises to 0.635 (a
    13x inflation) and power at matched FPR collapses to ~0, while a phi-fitting
    model is completely unaffected (docs/ase_validation.md sec 7h). hapmixQTL's
    validity is therefore contingent on mapping-bias-filtered input (WASP,
    phASER-style site filtering, or a variant-aware aligner). This function
    checks that precondition instead of assuming it.

    HOW IT SEPARATES BIAS FROM REAL SIGNAL
    --------------------------------------
    A single gene's allelic ratio confounds mapping bias with genuine
    allele-specific expression. Pooling across genes separates them: whether a
    real cis-eQTL's increasing allele happens to be the REFERENCE or the
    ALTERNATE allele is arbitrary, so true ASE contributes random-sign deviations
    that cancel in the pool. Reference mapping bias always favours the reference
    allele, so it accumulates. A pooled reference fraction significantly above
    0.5 is therefore evidence of bias, not of biology.

    Args:
        yL, yR:   [genes, samples] haplotype-L and haplotype-R counts.
        sign:     [genes, samples] signed het indicator s = xL - xR (+1 when the
                  ALTERNATE allele is on haplotype L, -1 when on R, 0 for
                  homozygotes, which carry no allelic information and are
                  ignored).
        min_total: minimum allele-specific depth for a gene-sample to count.
        min_sites: minimum usable gene-samples before a verdict is issued.

    Returns:
        dict with
          ref_fraction   pooled reference-allele fraction (0.5 = unbiased)
          implied_phi    the same quantity on RASQUAL's phi scale
          z, pvalue      test of ref_fraction against 0.5
          n_obs, n_reads counts entering the pool
          per_gene       [genes] per-gene reference fraction (NaN if unusable),
                         for flagging individual genes
          flag           True when bias is detected at p < 1e-3
          message        a human-readable verdict
    """
    yL = np.asarray(yL, dtype=float)
    yR = np.asarray(yR, dtype=float)
    sign = np.asarray(sign, dtype=float)
    if yL.ndim == 1:
        yL, yR, sign = yL[None, :], yR[None, :], sign[None, :]

    # ALT allele sits on L when s > 0, on R when s < 0; REF is the other one.
    alt = np.where(sign > 0, yL, yR)
    ref = np.where(sign > 0, yR, yL)
    tot = alt + ref
    usable = (np.abs(sign) > 0) & (tot >= min_total)

    n_obs = int(usable.sum())
    if n_obs < min_sites:
        return dict(ref_fraction=float('nan'), implied_phi=float('nan'),
                    z=float('nan'), pvalue=float('nan'), n_obs=n_obs,
                    n_reads=0, per_gene=np.full(yL.shape[0], np.nan),
                    flag=False,
                    message=f'insufficient data ({n_obs} usable gene-samples)')

    ref_tot = float(ref[usable].sum())
    all_tot = float(tot[usable].sum())
    frac = ref_tot / max(all_tot, 1.0)

    # The test must be clustered at the GENE level. Whether a real cis effect
    # raises the reference or the alternate allele is decided once per gene, so
    # the gene is the unit of randomization; treating gene-samples as
    # independent omits the between-gene variance and false-positives on
    # genuine ASE (measured at 37.5% before this was fixed). Reads within a
    # gene-sample are also overdispersed, so a binomial test on pooled counts
    # would be worse still.
    per_gene = np.full(yL.shape[0], np.nan)
    for g in range(yL.shape[0]):
        u = usable[g]
        if u.sum() >= 3:
            per_gene[g] = float(ref[g][u].sum() / max(tot[g][u].sum(), 1.0))

    gene_frac = per_gene[np.isfinite(per_gene)]
    if gene_frac.size < 5:
        return dict(ref_fraction=float(ref[usable].sum() / max(all_tot, 1.0)),
                    implied_phi=float('nan'), z=float('nan'), pvalue=float('nan'),
                    n_obs=n_obs, n_reads=int(all_tot), per_gene=per_gene,
                    flag=False,
                    message=f'insufficient genes ({gene_frac.size}) for a '
                            f'gene-clustered test')
    m = float(gene_frac.mean())
    sd = float(gene_frac.std(ddof=1))
    z = (m - 0.5) / (sd / np.sqrt(gene_frac.size)) if sd > 0 else 0.0
    pval = float(2 * stats.norm.sf(abs(z)))
    n_genes = int(gene_frac.size)

    flag = bool(pval < 1e-3)
    if flag:
        direction = 'reference' if m > 0.5 else 'alternate'
        message = (f'REFERENCE BIAS DETECTED: pooled allelic fraction {m:.4f} '
                   f'favours the {direction} allele (z={z:.1f}, p={pval:.2e}). '
                   f'hapmixQTL does not model mapping bias and is severely '
                   f'anticonservative in its presence -- filter with WASP or a '
                   f'variant-aware aligner before trusting these results. '
                   f'See docs/ase_validation.md sec 7h.')
    else:
        message = (f'no significant reference bias (gene-mean fraction '
                   f'{m:.4f}, p={pval:.2g}, {n_genes} genes)')

    return dict(ref_fraction=float(m), implied_phi=float(m), z=float(z),
                pvalue=pval, n_obs=n_obs, n_genes=n_genes,
                n_reads=int(all_tot), per_gene=per_gene, flag=flag,
                message=message)


def orient_haplotypes(sign_sites, depth_sites=None):
    """Per-sample haplotype orientation of a gene for reference_bias_diagnostic.

    The diagnostic needs, for every gene-sample, which haplotype carries the
    REFERENCE allele where that sample's reads land. A gene has many
    heterozygous sites and the reference allele sits on L at some and on R at
    others, so no single site's phase describes the gene. What mapping bias
    adds to the haplotype totals is sum_v reads_v * s_v: bias favours REF at
    every site carrying reads, and s_v = xL - xR says which haplotype is ALT
    there. The orientation that exposes it is therefore the sign of that
    depth-weighted sum; without per-site depths every het site counts alike.

    Args:
        sign_sites:  [n_sites, N] s = xL - xR at the gene's feature sites
                     (exonic hets, or gene-body hets without an exon table);
                     0 where the sample is homozygous
        depth_sites: [n_sites, N] allele-specific reads per site and sample,
                     or None for uniform weights

    Returns [N] in {-1, 0, +1}; 0 when the sample has no usable site or its
    weighted phases cancel.
    """
    s = np.asarray(sign_sites, dtype=float)
    if s.ndim == 1:
        s = s[None, :]
    if s.shape[0] == 0:
        return np.zeros(s.shape[1])
    w = np.ones_like(s) if depth_sites is None else np.asarray(depth_sites, dtype=float)
    return np.sign((w * s).sum(0))


def compute_summaries_from_gibbs(yL, yR, kappa=0.5, yT=None, count_noise=False):
    """
    Compute hapmixQTL summary statistics from Gibbs draws.

    Args:
        yL: haplotype L expression [features, samples, draws]
        yR: haplotype R expression [features, samples, draws]
        kappa: pseudocount (default 0.5)
        yT: optional gene TOTAL expression [features, samples, draws]. The
            allelic contrast a can only be formed where both haplotypes were
            quantified separately, but the total t must not be restricted that
            way. Against a personalized diploid transcriptome the second
            haplotype copy of a transcript only exists where the sample is
            heterozygous, so yL + yR is a heterozygous-transcript subtotal and
            using it as the total makes t a two-point mixture whose low cluster
            is determined by local heterozygosity -- which is in LD with the
            cis variants being tested. Pass the total summed over ALL
            transcripts. Defaults to yL + yR for backwards compatibility.
        count_noise: add per-sample Poisson counting noise to Va and Vt. This
            is for Salmon GIBBS draws, which reassign one fixed set of reads:
            their across-draw variance is read-ASSIGNMENT uncertainty only.
            (Bootstrap draws resample the reads and already carry counting
            noise; the term would double-count it.) A sample
            whose count is the same in every draw -- zero reads, or reads
            compatible with nothing else -- has v_inf exactly 0 and, under
            w = 1/(v_inf + tau), the largest weight in the gene, while on the
            log scale it is the least informative observation there is. On
            BrainVar LOC124902138 (median 9 reads per sample) three zero-count
            samples had Vt = 1e-32, the tau moment estimator collapsed to
            4e-6, and the three carried 99.8% of the total channel's weight:
            chi2 141 at a variant where an unweighted regression of t gives
            31, a Poisson GLM 34 and RASQUAL 3.0. The seven pilot genes with
            no zero-count sample had their three heaviest samples at 3-4% of
            the weight, i.e. uniform. The term is the plug-in Poisson
            variance of a log count -- 1/(tot + 2 kappa) for t = log(tot/2 +
            kappa), 1/(yL + kappa) + 1/(yR + kappa) for a -- so for a
            well-covered gene it sits far below v_inf + tau and changes
            nothing. Samples with no allele-specific reads at all keep Va = 0
            so _zero_degenerate_ase_weights still excludes them.

    Returns:
        A:   allelic contrast mean [features, samples]
        T:   log total mean [features, samples]
        Va:  inferential variance of a [features, samples]
        Vt:  inferential variance of t [features, samples]
        Cat: inferential covariance of a,t [features, samples]. Returned for
             inspection; INTENTIONALLY UNUSED by the mapping functions (see the
             module docstring).
    """
    a_draws = np.log(yL + kappa) - np.log(yR + kappa)
    tot = (yL + yR) if yT is None else np.asarray(yT)
    t_draws = np.log(tot / 2 + kappa)

    A = a_draws.mean(axis=2)
    T = t_draws.mean(axis=2)
    Va = a_draws.var(axis=2, ddof=0)
    Vt = t_draws.var(axis=2, ddof=0)
    Cat = ((a_draws - a_draws.mean(axis=2, keepdims=True)) *
           (t_draws - t_draws.mean(axis=2, keepdims=True))).mean(axis=2)

    if count_noise:
        mL, mR, mT = yL.mean(axis=2), yR.mean(axis=2), tot.mean(axis=2)
        no_cov = (mL + mR) <= 0
        Va = np.where(no_cov, 0.0, Va + 1.0 / (mL + kappa) + 1.0 / (mR + kappa))
        Vt = Vt + 1.0 / (mT + 2.0 * kappa)

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
        if bool((sqrt_w_t != 0).any()):
            self.Q_t, _ = torch.linalg.qr(design)
        else:
            # A switched-off channel (every weight zero; see _prepare_channels)
            # has nothing to project. The QR of an all-zero design returns
            # unit vectors, which would "residualize" the first design.shape[1]
            # samples of whatever is transformed.
            self.Q_t = design.new_zeros((N, 0))
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

def _estimate_tau_informative(y_t, v_inf_t, covariates_t, device, eps=1e-12):
    """_estimate_tau over the samples with v_inf > eps (see _prepare_channels).

    A sample with v_inf = 0 carries no information and must never enter the
    moment estimator: it would dominate mean(1/v) and collapse tau. There is
    deliberately no fallback to every sample when few informative ones remain
    (an earlier version had one, and re-admitted exactly those rows for
    sparse genes). Raises instead; _prepare_channels switches such a channel
    off before getting here, so reaching the error means the sparse-channel
    rule was bypassed.
    """
    keep = v_inf_t > eps
    n_keep = int(keep.sum())
    if n_keep < _min_informative(covariates_t):
        n_cols = 1 + (0 if covariates_t is None else covariates_t.shape[1])
        raise ValueError(
            f'tau cannot be estimated from {n_keep} informative samples against '
            f'a design of {n_cols} columns; the channel must be switched off')
    c = None if covariates_t is None else covariates_t[keep]
    return _estimate_tau(y_t[keep], v_inf_t[keep], c, device)


def _min_informative(covariates_t, extra=2):
    """Informative samples a channel needs to stay on: one per design column
    (intercept + covariates) plus `extra` residual degrees of freedom."""
    n_cov = 0 if covariates_t is None else covariates_t.shape[1]
    return 1 + n_cov + extra


def _channel_weights(y_t, v_t, covariates_t, tau_mode, device, eps=1e-12):
    """sqrt weights of one channel, or all zeros when the channel is off.

    The sparse-channel rule: with fewer informative samples (v > eps) than
    the channel's design has columns plus two, neither the regression nor
    tau is identifiable from that channel, so it contributes nothing (every
    weight zero -> xx = 0 -> the meta-analysis takes the other channel alone).
    """
    n_inf = int((v_t > eps).sum())
    if n_inf < _min_informative(covariates_t):
        return torch.zeros_like(v_t)
    if tau_mode == 'estimate':
        tau = _estimate_tau_informative(y_t, v_t, covariates_t, device, eps)
        return torch.sqrt(1.0 / (v_t.clamp(min=1e-8) + tau))
    return torch.sqrt(1.0 / v_t.clamp(min=1e-8))


SAME_COVARIATES = 'same'


def _resolve_ase_covariates(ase_covariates_df, covariates_df, samples, device, logger):
    """The allelic channel's covariate design as _prepare_channels wants it:
    SAME_COVARIATES (the total channel's), None (intercept only) or a tensor
    built from its own DataFrame. Also returns its column count for the dof
    rule."""
    if isinstance(ase_covariates_df, str) and ase_covariates_df == SAME_COVARIATES:
        n = 0 if covariates_df is None else covariates_df.shape[1]
        logger.write('  * allelic channel covariates: same as the total channel')
        return SAME_COVARIATES, n
    if ase_covariates_df is None:
        logger.write('  * allelic channel covariates: none (intercept only)')
        return None, 0
    assert np.all(np.asarray(samples) == np.asarray(ase_covariates_df.index)), \
        'Allelic-channel covariate samples must match phenotype samples'
    logger.write(f'  * allelic channel covariates: {ase_covariates_df.shape[1]}')
    t = torch.tensor(ase_covariates_df.values, dtype=torch.float32).to(device)
    return t, ase_covariates_df.shape[1]


def _warn_tau_zero(tau_mode):
    """tau_mode='zero' is anticonservative on real data (see docs/ase_validation.md)."""
    if tau_mode == 'zero':
        import warnings
        warnings.warn(
            "hapmixQTL: tau_mode='zero' asserts that Gibbs inferential variance is the "
            "ENTIRE error variance, which real quantifier posteriors never satisfy. "
            "Measured consequences: up to 107x nominal type-I error at alpha=1e-3, ~100% "
            "false positives on count-level simulations, 64.5% coverage of nominal 95% "
            "CIs, and -- at matched empirical type-I error -- no more power than using "
            "total counts alone. Use tau_mode='estimate' unless reproducing prior "
            "results. See docs/ase_validation.md.",
            RuntimeWarning, stacklevel=3)


def _prepare_channels(a_t, t_t, va_t, vt_t, covariates_t, tau_mode, device,
                      ase_covariates_t=SAME_COVARIATES, eps=1e-12):
    """
    Per-phenotype whitening shared by every mapping function.

    Estimates tau per channel (unless tau_mode='zero'), builds the sqrt
    weights w_i = 1/(v_inf_i + tau) with zero-coverage ASE samples zeroed
    out, and the weighted residualizers that project out the intercept and
    covariates. Any second-pass regression that reuses this is whitened
    exactly like the lead scan.

    Takes the RAW inferential variances. The mapping functions used to
    clamp them to 1e-8 first, which hid every zero-coverage sample from
    both the tau estimator and the degenerate-ASE guard (whose threshold is
    1e-12); the floor is applied here, inside the weight, where it is
    harmless.

    tau is estimated on the samples that carry information (v > eps). A
    sample with v_inf = 0 (no allele-specific reads; a zero total) would
    enter the moment estimator with weight 1/1e-8 and dominate mean(1/v), so
    a handful of them drove tau to ~1e-6 for the whole gene and left the
    informative samples weighted by v_inf alone, which understates the
    between-sample variance of a by 2-25x on well-covered BrainVar genes:
    the allelic channel's permutation null reached chi2 40-120 (CRMP1 97.7
    against a calibrated ~12-16). Estimated on informative samples, the same
    genes give tau_a 0.007-0.13 and near-uniform weights.

    Sparse-channel rule: a channel with fewer informative samples than its
    design has columns plus two is switched off (all weights zero), so the
    meta-analysis falls back to the other channel. Zero-variance rows never
    reach the tau estimator under any branch.

    The two channels take separate covariate designs. ``covariates_t`` is
    the total channel's; ``ase_covariates_t`` is the allelic channel's:
    SAME_COVARIATES (default) reuses the total channel's, None fits an
    intercept only. The allelic contrast a = log(yL/yR) is a within-sample
    difference in which anything acting on both haplotypes alike (library
    size, expression PCs, sex, age) cancels, so the total channel's
    covariates are normally not wanted there: each column costs one of the
    informative samples and can only remove signal (10 expression PCs
    against ~50 informative samples halved CCNI's allelic statistic).

    Returns:
        sqrt_wa_t, sqrt_wt_t, residualizer_a, residualizer_t
    """
    _warn_tau_zero(tau_mode)
    if isinstance(ase_covariates_t, str) and ase_covariates_t == SAME_COVARIATES:
        ase_covariates_t = covariates_t
    sqrt_wa_t = _zero_degenerate_ase_weights(
        _channel_weights(a_t, va_t, ase_covariates_t, tau_mode, device, eps), va_t, eps)
    sqrt_wt_t = _channel_weights(t_t, vt_t, covariates_t, tau_mode, device, eps)
    residualizer_a = WeightedResidualizer(ase_covariates_t, sqrt_wa_t)
    residualizer_t = WeightedResidualizer(covariates_t, sqrt_wt_t)
    return sqrt_wa_t, sqrt_wt_t, residualizer_a, residualizer_t


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

    The scalar meta-analysis treats the two channel estimators as
    independent and deliberately ignores Cat (the a-t inferential
    covariance). That is correct, not an omission: s = xL - xR is orthogonal
    to g/2 under random phase (E[s | g=1] = 0), so the two regressions
    project any shared noise onto orthogonal directions and corr(beta_a,
    beta_t) stays at zero even when a and t are correlated at rho = 0.9
    (measured, docs/ase_validation.md sec 3; asserted by
    tests/test_hapmixqtl_calibration.py).

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


def _permute_within_informative(r_t, informative_t, permutation_ix_t):
    """Permute the entries of r_t [N] among the samples flagged informative,
    once per row of permutation_ix_t [nperm, N] (each row a permutation of
    0..N-1). Returns [nperm, N].

    A row's permutation is restricted to the informative subset by keeping
    the informative samples in the order the row visits them, which is a
    uniformly random permutation of that subset and lets both channels share
    one draw. Entries that are not informative keep their value (zero for a
    sample the channel does not use), so no information lands on a zero
    weight and no zero lands on an informative sample.
    """
    nperm, N = permutation_ix_t.shape
    out = r_t.unsqueeze(0).expand(nperm, N).clone()
    inf_ix = torch.nonzero(informative_t, as_tuple=False).flatten()
    n_inf = inf_ix.numel()
    if n_inf <= 1:
        return out
    visits = informative_t[permutation_ix_t]
    src = permutation_ix_t[visits].view(nperm, n_inf)
    out[:, inf_ix] = r_t[src]
    return out


def calculate_hapmixqtl_permutations(genotypes_t, sign_t, a_t, t_t,
                                      sqrt_wa_t, sqrt_wt_t,
                                      residualizer_a, residualizer_t,
                                      permutation_ix_t, dof=None):
    """
    Compute nominal and permutation statistics for hapmixQTL.

    ``dof`` is the t-reference used to map the known-variance statistic to a
    correlation scale; the mapping is monotonic, so the empirical p-value does
    not depend on it, but the nominal p-value the caller derives from
    r_nominal does. Default: the smaller of the two channels' residualizer
    dof, which with per-channel covariates is the total channel's (the
    larger design). map_cis passes its own N - 2 - n_cov so both agree.

    The permutation null is Freedman-Lane in whitened space. Under the null
    the whitened residuals e*_i = sqrt(w_i) (y_i - C b) have unit variance
    whatever the sample's v_inf, so they are exchangeable and are what a
    permutation may move between samples; the predictors and the weights
    stay in sample order, so every permuted statistic is one matrix product.
    The earlier scheme permuted the RAW phenotype values at fixed weights,
    handing sample i another sample's value at its own precision; under
    heteroskedastic v_inf that mis-scales the null (v_inf spanning 0.01-2:
    a null gene's empirical p averaged 0.94, type-I 0.000 at alpha 0.05).
    Each channel permutes among its informative samples only (weight > 0),
    and the two channels share the draw (see _permute_within_informative).
    Because the residualized predictors are orthogonal to the null design,
    re-residualizing the permuted residuals would leave xy unchanged, and
    _combined_tstat2 does not use yy, so that step is skipped.

    Returns:
        r_nominal:  signed correlation-scale statistic for best variant (scalar)
        std_ratio:  factor with r_nominal * std_ratio == the known-variance
                    GLS slope of the best variant (scalar; see below)
        best_ix:    index of best variant (scalar)
        r2_perm_t:  max r^2 per permutation [nperm]
        g_best:     genotype vector for best variant [N]
    """
    if dof is None:
        dof = min(residualizer_a.dof, residualizer_t.dof)

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

    # std_ratio is defined so that the caller's tensorQTL-style reconstruction
    #     slope    = r_nominal * std_ratio
    #     slope_se = |slope| / sqrt(dof * r2 / (1 - r2)) = |slope| / sqrt(tstat2)
    # returns EXACTLY the known-variance GLS slope (xy_a + xy_t)/(xx_a + xx_t)
    # and its SE 1/sqrt(xx_a + xx_t), i.e. the same estimator map_nominal
    # reports. (The OLS identity slope = r * sd_y/sd_x does not hold for the
    # known-variance GLS statistic, so sqrt(pheno_var/geno_var) would give an
    # approximate slope that disagrees with map_nominal by several percent.)
    std_ratio = torch.where(r_nominal.abs() > 0, slope_nom / r_nominal,
                            torch.zeros_like(slope_nom))

    # --- Permutation statistics: whitened residuals permuted within each
    # channel's informative samples (see the docstring) ---
    a_res_perms = _permute_within_informative(a_star_res[0], sqrt_wa_t > 0, permutation_ix_t)
    t_res_perms = _permute_within_informative(t_star_res[0], sqrt_wt_t > 0, permutation_ix_t)

    xy_a_perm = torch.mm(s_star_res, a_res_perms.t())
    yy_a_perm = (a_res_perms * a_res_perms).sum(1)
    xy_t_perm = torch.mm(g_half_star_res, t_res_perms.t())
    yy_t_perm = (t_res_perms * t_res_perms).sum(1)

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


def cis_trans_diagnostic(slope_a, se_a, slope_t, se_t, dof):
    """
    Per-variant test of the assumption the meta-analysis rests on: that the
    ASE and total channels estimate the SAME effect, i.e. a pure cis effect.

    In CSeQTL's notation eta_A = alpha * eta_T, and the effect is cis exactly
    when alpha = 1. hapmixQTL assumes alpha = 1 without checking. When it is
    violated -- a trans component acting on total expression only, reference
    mapping bias attenuating the ASE channel, systematic phasing error,
    feature-level misquantification -- the combined slope averages two
    different quantities and is silently attenuated: at alpha = 0 it reports
    0.08 for a true 0.40, and at alpha = -0.5 it flips sign
    (docs/ase_validation.md sec 7c).

    Because the two channel estimators are uncorrelated (sec 3), the
    difference has variance se_a^2 + se_t^2 with no covariance term and a
    Wald test is exact:  z = (slope_a - slope_t) / sqrt(se_a^2 + se_t^2).

    This is a DIAGNOSTIC, not a correction or a filter: it flags genes whose
    reported effect should not be read as a cis log aFC. It is a screen for
    gross violations (detection 92% at alpha = 0, 12% at alpha = 0.75).

    Returns:
        alpha_cis:      slope_a / slope_t (NaN when slope_t ~ 0 or a channel
                        has no finite SE)
        pval_cis_trans: two-sided p on the same t reference (dof) as the other
                        p-values; NaN when a channel is unavailable, e.g. no
                        phase -> no ASE channel -> nothing to compare
    """
    slope_a = np.atleast_1d(np.asarray(slope_a, float)); se_a = np.atleast_1d(np.asarray(se_a, float))
    slope_t = np.atleast_1d(np.asarray(slope_t, float)); se_t = np.atleast_1d(np.asarray(se_t, float))
    ok = np.isfinite(se_a) & (se_a > 0) & np.isfinite(se_t) & (se_t > 0)
    z = np.full(slope_a.shape, np.nan)
    z[ok] = (slope_a[ok] - slope_t[ok]) / np.sqrt(se_a[ok] ** 2 + se_t[ok] ** 2)
    pval = np.full(slope_a.shape, np.nan)
    pval[ok] = 2 * stats.t.sf(np.abs(z[ok]), dof)
    with np.errstate(divide='ignore', invalid='ignore'):
        alpha = np.where(ok & (np.abs(slope_t) > 1e-12), slope_a / slope_t, np.nan)
    return alpha, pval


# ---------------------------------------------------------------------------
#  Nominal mapping
# ---------------------------------------------------------------------------

def map_nominal(genotype_df, variant_df, A_df, T_df, Va_df, Vt_df,
                phenotype_pos_df, xL_df=None, xR_df=None, prefix='',
                covariates_df=None, maf_threshold=0, window=1000000,
                tau_mode='estimate', se_mode='model',
                output_dir='.', logger=None, verbose=True,
                ase_covariates_df=SAME_COVARIATES):
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
        covariates_df:    covariates [samples x covariates] or None, applied
                          to the TOTAL channel (and, by default, the allelic one)
        ase_covariates_df: covariates for the ALLELIC channel: SAME_COVARIATES
                          (default; the total channel's), None (intercept only)
                          or a DataFrame [samples x k]. The allelic contrast is
                          a within-sample difference in which covariates that act
                          on both haplotypes alike cancel, so None is the usual
                          choice; each column projected out of the allelic
                          channel costs one informative sample and can only
                          remove signal (see _prepare_channels)
        maf_threshold:    minimum minor allele frequency
        window:           cis-window size in bases
        tau_mode:         'estimate' (default) or 'zero'.

            'estimate' adds a moment-estimated overdispersion term tau to the
            per-sample variance, so the weights are w_i = 1/(v_inf_i + tau).

            'zero' uses w_i = 1/v_inf_i, i.e. it asserts the Gibbs inferential
            variance is the ENTIRE error variance. That is essentially never
            true of real data -- a quantifier's posterior captures only
            allelic-assignment uncertainty conditional on the observed total,
            not the counts' sampling variance and not biological variance -- so
            the weights come out uniformly too large, the known-variance GLS SE
            (Var(beta) = 1/xx) collapses, and p-values are severely
            anticonservative. Measured: up to 107x the nominal type-I error at
            alpha=1e-3 on Gaussian simulations and ~100% false positives on
            count-level simulations (lambda_GC -> inf); nominal 95% CIs cover
            64.5%. At matched empirical type-I error it also loses all the power
            the allele-specific channel provides, performing no better than
            total-count-only. See docs/ase_validation.md.

            This mirrors the parent method: mixQTL (Liang et al. 2021) writes the
            ASE error as N(0, sigma^2 * (1/Y1 + 1/Y2)) where the counts set only
            the SHAPE of the weights and sigma^2 is a free scale parameter it
            infers from the data (Supplementary Notes 5.2). 'zero' is what you
            get by dropping that free scale; 'estimate' restores it.

            'zero' is retained only for reproducing prior results and emits a
            warning.
        se_mode:          'model' (default) or 'robust' (sandwich)
        output_dir:       output directory
        logger:           SimpleLogger instance
        verbose:          print progress
    
    Every pair also carries ``pval_cis_trans``, a Wald test that the ASE and
    total channels estimate the same effect (``cis_trans_diagnostic``). A
    small value flags a pair whose combined slope should not be read as a
    cis log aFC; it is a diagnostic column, not a filter (docs sec 7c).
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

    ase_covariates_t, n_cov_a = _resolve_ase_covariates(
        ase_covariates_df, covariates_df, samples, device, logger)
    # one t reference for the combined statistic: the larger design's
    dof = N - 2 - max(n_cov, n_cov_a)

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
        chr_res['pval_cis_trans'] = np.empty(n, dtype=np.float64)

        start = 0
        for k, (_, genotypes, genotype_range, phenotype_id) in enumerate(
            igc.generate_data(chrom=chrom, verbose=verbose), k + 1
        ):
            if phenotype_id not in pheno_ix:
                continue

            pidx = pheno_ix[phenotype_id]
            a_t = torch.tensor(A_df.values[pidx], dtype=torch.float32).to(device)
            t_t = torch.tensor(T_df.values[pidx], dtype=torch.float32).to(device)
            va_t = torch.tensor(Va_df.values[pidx], dtype=torch.float32).to(device)
            vt_t = torch.tensor(Vt_df.values[pidx], dtype=torch.float32).to(device)

            sqrt_wa_t, sqrt_wt_t, residualizer_a, residualizer_tc = _prepare_channels(
                a_t, t_t, va_t, vt_t, covariates_t, tau_mode, device,
                ase_covariates_t=ase_covariates_t)

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
            chr_res['pval_cis_trans'][start:start + nv] = cis_trans_diagnostic(
                slope_a, se_a, slope_tc, se_tc, dof)[1]
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
            nperm=10000, window=1000000, tau_mode='estimate', se_mode='model',
            logger=None, seed=None, verbose=True, warn_monomorphic=True,
            ase_covariates_df=SAME_COVARIATES):
    """
    hapmixQTL cis-QTL mapping with permutation-based empirical p-values.

    ``ase_covariates_df`` is the allelic channel's covariate design (see
    map_nominal): SAME_COVARIATES, None for an intercept only, or its own
    DataFrame. The nominal p-value uses one t reference for both channels,
    dof = N - 2 - max(n_cov, n_cov_a).

    For each phenotype, finds the best cis variant and computes empirical
    p-values by permuting the whitened null residuals of each channel among
    its informative samples (Freedman-Lane in whitened space; see
    calculate_hapmixqtl_permutations). ``se_mode`` must be 'model': the
    permutation statistic is the known-variance GLS statistic, which has no
    sandwich counterpart here; use map_nominal for robust standard errors.

    Returns:
        DataFrame with one row per phenotype, analogous to cis.map_cis output.
    
    The lead variant's per-channel slopes (``slope_a``/``slope_t`` with SEs),
    ``alpha_cis = slope_a / slope_t`` and ``pval_cis_trans`` (Wald test of
    ``slope_a = slope_t``, see ``cis_trans_diagnostic``) are reported per gene.
    A small ``pval_cis_trans`` means the two channels disagree, so the
    combined slope should not be read as a cis log aFC (a trans component,
    mapping bias or phasing error attenuate it; docs sec 7c). It is a
    diagnostic column, not a filter.
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

    if se_mode != 'model':
        raise ValueError(
            f"map_cis: se_mode={se_mode!r} is not available. The permutation "
            "statistic is the known-variance GLS statistic, and sandwich "
            "standard errors have no permutation counterpart here; use "
            "map_nominal for se_mode='robust'")

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

    ase_covariates_t, n_cov_a = _resolve_ase_covariates(
        ase_covariates_df, covariates_df, samples, device, logger)
    # one t reference for the combined statistic: the larger design's
    dof = N - 2 - max(n_cov, n_cov_a)

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
        va_t = torch.tensor(Va_df.values[pidx], dtype=torch.float32).to(device)
        vt_t = torch.tensor(Vt_df.values[pidx], dtype=torch.float32).to(device)

        sqrt_wa_t, sqrt_wt_t, residualizer_a, residualizer_tc = _prepare_channels(
            a_t, t_t, va_t, vt_t, covariates_t, tau_mode, device,
            ase_covariates_t=ase_covariates_t)

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
            permutation_ix_t, dof=dof,
        )
        r_nominal, std_ratio, var_ix, r2_perm, g = [i.cpu().numpy() for i in res]
        best_local = int(var_ix)
        var_ix = genotype_range[var_ix]

        # per-channel slopes at the lead variant, for the cis/trans diagnostic
        _, _, _, lead_a, lead_a_se, lead_t, lead_t_se = [
            float(x.cpu().numpy()[0]) for x in calculate_hapmixqtl_nominal(
                genotypes_t[best_local:best_local + 1], sign_t[best_local:best_local + 1],
                a_t, t_t, sqrt_wa_t, sqrt_wt_t, residualizer_a, residualizer_tc)]
        alpha_cis, pval_cis_trans = cis_trans_diagnostic(
            lead_a, lead_a_se, lead_t, lead_t_se, dof)

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
            ('slope_a', lead_a),
            ('slope_a_se', lead_a_se),
            ('slope_t', lead_t),
            ('slope_t_se', lead_t_se),
            ('alpha_cis', float(alpha_cis[0])),
            ('pval_cis_trans', float(pval_cis_trans[0])),
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


def map_susie(genotype_df, variant_df, A_df, T_df, Va_df, Vt_df,
              phenotype_pos_df, xL_df=None, xR_df=None,
              covariates_df=None, L=10, scaled_prior_variance=0.2,
              estimate_residual_variance=False, estimate_prior_variance=True,
              coverage=0.95, min_abs_corr=0.5, maf_threshold=0,
              tau_mode='estimate', max_iter=500, window=1000000, tol=1e-3,
              summary_only=True, logger=None, verbose=True,
              warn_monomorphic=False, ase_covariates_df=SAME_COVARIATES):
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
        to ``susie.map``. The summary carries a ``tau_mode`` column and every
        ``susie_res`` entry a ``tau_mode`` key, so fine-mapping produced under
        the invalid ``'zero'`` setting (docs/ase_validation.md sec 7g) can be
        identified later; see ``fine_mapping_provenance``.
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
    ase_covariates_t, _ = _resolve_ase_covariates(
        ase_covariates_df, covariates_df, samples, device, logger)

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
        va_t = torch.tensor(Va_df.values[pidx], dtype=torch.float32).to(device)
        vt_t = torch.tensor(Vt_df.values[pidx], dtype=torch.float32).to(device)

        sqrt_wa_t, sqrt_wt_t, residualizer_a, residualizer_tc = _prepare_channels(
            a_t, t_t, va_t, vt_t, covariates_t, tau_mode, device,
            ase_covariates_t=ase_covariates_t)

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
            susie_res[phenotype_id]['tau_mode'] = tau_mode

    logger.write(f'  Time elapsed: {(time.time() - start_time) / 60:.2f} min')
    logger.write('done.')

    if susie_summary:
        susie_summary = pd.concat(susie_summary, axis=0).rename(
            columns={'snp': 'variant_id'}
        ).reset_index(drop=True)
        susie_summary['tau_mode'] = tau_mode
    else:
        susie_summary = pd.DataFrame(
            columns=['phenotype_id', 'variant_id', 'pip', 'af', 'cs_id', 'tau_mode']
        )

    if summary_only:
        return susie_summary
    else:
        drop_ids = [key for key in susie_res if susie_res[key]['sets']['cs'] is None]
        for key in drop_ids:
            del susie_res[key]
        return susie_summary, susie_res


def fine_mapping_provenance(summary):
    """
    Classify a ``map_susie`` summary (a DataFrame, or the path of the parquet
    or tab-delimited file the CLI writes) by the tau_mode it was produced
    under.

    Fine-mapping run under ``tau_mode='zero'`` is invalid, not merely
    miscalibrated: nominal 95% credible sets covered the causal variant 36.8%
    of the time and PIP-0.98 variants were causal 34% of the time
    (docs/ase_validation.md sec 7g). Such results should be redone, not
    re-thresholded. Summaries written before outputs recorded ``tau_mode``
    carry no provenance and, unless ``tau_mode='estimate'`` was passed
    explicitly, were produced under the old default ``'zero'``.

    Returns:
        dict(status, tau_modes, message) with status 'ok', 'stale' or 'unknown'
    """
    if not isinstance(summary, pd.DataFrame):
        path = str(summary)
        summary = (pd.read_parquet(path) if path.endswith('.parquet')
                   else pd.read_csv(path, sep='\t'))
    if 'tau_mode' not in summary.columns:
        return dict(status='unknown', tau_modes=[], message=(
            'no tau_mode column: produced before map_susie recorded provenance. '
            "Unless tau_mode='estimate' was passed explicitly this was run under the "
            "old default 'zero' and should be redone (docs/ase_validation.md sec 7g)."))
    modes = sorted(set(summary['tau_mode'].dropna().astype(str)))
    if 'zero' in modes:
        return dict(status='stale', tau_modes=modes, message=(
            "produced under tau_mode='zero': credible sets and PIPs are invalid "
            '(docs/ase_validation.md sec 7g); redo with the default tau_mode.'))
    return dict(status='ok', tau_modes=modes,
                message='produced under tau_mode=' + '/'.join(modes))


# ---------------------------------------------------------------------------
#  Second-pass regressions: multi-column joint GLS for one locus
# ---------------------------------------------------------------------------
#
# The lead scan tests one column per variant. Two kinds of loci carry more
# than one column's worth of information and get a SECOND regression after
# the scan, whitened identically (same tau, weights, residualizers):
#
#   * multiallelic non-repeat sites (multi-ALT SNVs, indels): a CATEGORICAL
#     model with one indicator per non-reference allele, i.e. the K-1
#     split-biallelic rows fitted JOINTLY. Each beta_k is the log aFC of
#     allele k against a clean reference allele (the marginal split-row fit
#     lumps the other ALT alleles into "not k"); a 1/2 heterozygote
#     estimates beta_1 - beta_2 directly through the ASE channel; and the
#     joint K-1 df test asks whether allele identity matters at all, with no
#     ordering assumed. See map_multiallelic.
#
#   * STRs: a LINEAR + CURVATURE model in repeat length, per haplotype
#     f(L) = b1 L + b2 L^2. Because hapmixQTL shares one per-haplotype effect
#     across both channels the square goes on the HAPLOTYPE: the total row is
#     (f(L_A) + f(L_B))/2, so the squared column is (L_A^2 + L_B^2)/2 and NOT
#     ((L_A+L_B)/2)^2 (the two differ by (L_A-L_B)^2/4, a heterozygosity term
#     the ASE channel cannot share); the ASE row is f(L_A) - f(L_B). b2 is a
#     1-df curvature test. See map_str_curvature.
#
# Neither touches lead selection: the scan stays 1 df per variant (the
# linear-in-length row for an STR, the split rows for a multiallelic site).
#
# For one shared coefficient vector beta [p] the stacked whitened design is
#     y = [a*; t*]    X = [Xa*; Xt*]    (2N pseudo-samples, block-diag cov)
# and known-variance GLS gives beta = (X'X)^-1 X'y, Var = (X'X)^-1, which for
# p = 1 is exactly the inverse-variance meta-analysis of the two channel fits
# that the lead scan uses.


def _gls_solve(X, y, xx_pre, robust=False, dof_robust=None):
    """
    Known-variance GLS on whitened, residualized data.

    Args:
        X:      [p, M] predictors (float64 numpy)
        y:      [M] response
        xx_pre: [p] predictor norms BEFORE residualization (estimability gate,
                same criterion as _wls_regression)
        robust: HC1 sandwich covariance instead of (X'X)^-1

    Returns:
        beta [p] (NaN where not estimable), cov [p, p] (NaN likewise),
        estimable [p] bool, rank_ok bool (False -> everything NaN: the
        estimable columns are collinear, e.g. two alleles carried by the
        same handful of samples)
    """
    p = X.shape[0]
    beta = np.full(p, np.nan)
    cov = np.full((p, p), np.nan)
    xx = (X * X).sum(1)
    est = xx > 1e-12 * np.maximum(xx_pre, 1e-30)
    if not est.any():
        return beta, cov, est, True
    Xe = X[est]
    XtX = Xe @ Xe.T
    ev = np.linalg.eigvalsh(XtX)
    if ev[0] <= 1e-10 * ev[-1]:
        return beta, cov, est, False
    XtX_inv = np.linalg.inv(XtX)
    b = XtX_inv @ (Xe @ y)
    if robust:
        e = y - b @ Xe
        meat = (Xe * e) @ (Xe * e).T
        M = Xe.shape[1]
        corr = M / max(dof_robust if dof_robust else M - Xe.shape[0], 1)
        c = XtX_inv @ meat @ XtX_inv * corr
    else:
        c = XtX_inv
    ix = np.where(est)[0]
    beta[ix] = b
    cov[np.ix_(ix, ix)] = c
    return beta, cov, est, True


def _joint_gls(Xa_t, Xt_t, a_t, t_t, sqrt_wa_t, sqrt_wt_t,
               residualizer_a, residualizer_t, robust=False, n_cov=0):
    """
    Joint known-variance GLS of a shared coefficient vector across channels,
    plus the same design fitted within each channel alone.

    Args:
        Xa_t: [p, N] ASE-channel predictors (per-haplotype contrasts)
        Xt_t: [p, N] total-channel predictors (per-haplotype means)
        a_t, t_t, sqrt_wa_t, sqrt_wt_t, residualizers: as in
            calculate_hapmixqtl_nominal

    Returns dict with beta, cov, chi2 (Wald on the estimable columns),
    estimable, rank_ok, and per-channel beta_a, cov_a, beta_t, cov_t.
    """
    a_star = (a_t * sqrt_wa_t).unsqueeze(0)
    t_star = (t_t * sqrt_wt_t).unsqueeze(0)
    Xa_star = Xa_t * sqrt_wa_t.unsqueeze(0)
    Xt_star = Xt_t * sqrt_wt_t.unsqueeze(0)
    pre_a = (Xa_star * Xa_star).sum(1)
    pre_t = (Xt_star * Xt_star).sum(1)
    Xa = residualizer_a.transform(Xa_star).double().cpu().numpy()
    Xt = residualizer_t.transform(Xt_star).double().cpu().numpy()
    ya = residualizer_a.transform(a_star).double().cpu().numpy()[0]
    yt = residualizer_t.transform(t_star).double().cpu().numpy()[0]
    pre_a = pre_a.double().cpu().numpy(); pre_t = pre_t.double().cpu().numpy()
    N = Xa.shape[1]; p = Xa.shape[0]
    X = np.concatenate([Xa, Xt], axis=1)
    y = np.concatenate([ya, yt])
    dof_rob = 2 * N - 2 * (1 + n_cov) - p
    beta, cov, est, ok = _gls_solve(X, y, pre_a + pre_t, robust, dof_rob)
    chi2 = np.nan
    if ok and est.any():
        ix = np.where(est)[0]
        b = beta[ix]
        chi2 = float(b @ np.linalg.solve(cov[np.ix_(ix, ix)], b))
    beta_a, cov_a, _, _ = _gls_solve(Xa, ya, pre_a, robust, N - (1 + n_cov) - p)
    beta_t, cov_t, _, _ = _gls_solve(Xt, yt, pre_t, robust, N - (1 + n_cov) - p)
    return dict(beta=beta, cov=cov, chi2=chi2, estimable=est, rank_ok=ok,
                beta_a=beta_a, cov_a=cov_a, beta_t=beta_t, cov_t=cov_t)


def _categorical_design(hapA, hapB, phased, min_hap):
    """
    Per-haplotype allele indicators for one multiallelic site.

    Args:
        hapA, hapB: [N] int allele index per haplotype, -1 = missing
        phased:     [N] bool; unphased heterozygotes contribute no ASE contrast
        min_hap:    alleles carried by fewer haplotypes are pooled into an
                    'other' column; if even the pool is below min_hap those
                    haplotypes are treated as missing

    Returns None if nothing is testable, else dict with Xa [p,N], Xt [p,N],
    labels (allele index as str, or 'other'), ref, n_hap [p] (called carrier
    haplotypes per column), n_alleles (observed), n_missing_hap, pooled.
    Missing haplotypes get the mean-imputed indicator (allele frequency) in
    the total channel and a zero ASE contrast, mirroring mean-imputed dosage.
    """
    hapA = hapA.astype(int).copy(); hapB = hapB.astype(int).copy()
    haps = np.concatenate([hapA[hapA >= 0], hapB[hapB >= 0]])
    if haps.size == 0:
        return None
    alleles, counts = np.unique(haps, return_counts=True)
    ref = int(alleles[np.argmax(counts)])
    others = [(int(al), int(c)) for al, c in zip(alleles, counts) if al != ref]
    cols = [(str(al), [al]) for al, c in others if c >= min_hap]
    rare = [al for al, c in others if c < min_hap]
    pooled = False
    if rare:
        if sum(c for al, c in others if c < min_hap) >= min_hap:
            cols.append(('other', rare)); pooled = True
        else:
            hapA[np.isin(hapA, rare)] = -1
            hapB[np.isin(hapB, rare)] = -1
    if not cols:
        return None
    N = hapA.shape[0]; p = len(cols)
    mA = hapA < 0; mB = hapB < 0
    eA = np.zeros((p, N)); eB = np.zeros((p, N))
    for j, (_, members) in enumerate(cols):
        eA[j] = np.isin(hapA, members); eB[j] = np.isin(hapB, members)
    n_hap = eA[:, ~mA].sum(1) + eB[:, ~mB].sum(1)
    n_called = (~mA).sum() + (~mB).sum()
    freq = n_hap / max(n_called, 1)
    eA[:, mA] = freq[:, None]; eB[:, mB] = freq[:, None]
    Xt = (eA + eB) / 2.0
    Xa = (eA - eB) * (phased & ~mA & ~mB)[None, :]
    return dict(Xa=Xa, Xt=Xt, labels=[c[0] for c in cols], ref=ref,
                n_hap=n_hap.astype(int), n_alleles=int(len(alleles)),
                n_missing_hap=int(mA.sum() + mB.sum()), pooled=pooled)


def _str_design(LA, LB, phased, winsor=(0.01, 0.99)):
    """
    Per-haplotype [L, L^2] basis for one STR, centered on the cohort mean
    haplotype length (winsorized first so a few long alleles do not own the
    squared column).

    Args:
        LA, LB: [N] float repeat lengths in repeat units, NaN = missing
        phased: [N] bool

    Returns None if there is no length variation, else dict with Xa [2,N],
    Xt [2,N], center, lo, hi, n_called, n_phased. Missing samples get the
    mean-imputed basis in the total channel and a zero ASE contrast.
    """
    called = ~(np.isnan(LA) | np.isnan(LB))
    if called.sum() < 3:
        return None
    haps = np.concatenate([LA[called], LB[called]])
    if winsor:
        lo, hi = np.quantile(haps, winsor)
    else:
        lo, hi = haps.min(), haps.max()
    la = np.clip(LA, lo, hi); lb = np.clip(LB, lo, hi)
    c = float(np.clip(haps, lo, hi).mean())
    la = la - c; lb = lb - c
    fA = np.stack([la, la ** 2]); fB = np.stack([lb, lb ** 2])
    basis_mean = np.concatenate([fA[:, called], fB[:, called]], axis=1).mean(1)
    fA[:, ~called] = basis_mean[:, None]; fB[:, ~called] = basis_mean[:, None]
    if np.std(fA[0, called] + fB[0, called]) < 1e-9:
        return None
    Xt = (fA + fB) / 2.0
    Xa = (fA - fB) * (phased & called)[None, :]
    return dict(Xa=Xa, Xt=Xt, center=c, lo=float(lo), hi=float(hi),
                n_called=int(called.sum()), n_phased=int((phased & called).sum()))


def _cis_sites(site_chrom, site_pos, phenotype_pos_df, window):
    """Per phenotype, indices of sites within the cis window (and the
    phenotype start used for distances)."""
    import bisect
    site_chrom = np.asarray(site_chrom).astype(str)
    site_pos = np.asarray(site_pos).astype(int)
    by_chrom = {}
    for c in np.unique(site_chrom):
        ix = np.where(site_chrom == c)[0]
        o = np.argsort(site_pos[ix], kind='stable')
        by_chrom[c] = (site_pos[ix][o], ix[o])
    if 'pos' in phenotype_pos_df:
        starts = phenotype_pos_df['pos'].astype(int); ends = starts
    else:
        starts = phenotype_pos_df['start'].astype(int)
        ends = phenotype_pos_df['end'].astype(int)
    chrs = phenotype_pos_df['chr'].astype(str)
    out = {}
    for pid in phenotype_pos_df.index:
        c = chrs[pid]
        if c not in by_chrom:
            continue
        pos_sorted, ix_sorted = by_chrom[c]
        lb = bisect.bisect_left(pos_sorted, starts[pid] - window)
        ub = bisect.bisect_right(pos_sorted, ends[pid] + window)
        if ub > lb:
            out[pid] = (ix_sorted[lb:ub], int(starts[pid]))
    return out


def _second_pass(kind, n_sites, site_chrom, site_pos, site_samples,
                 A_df, T_df, Va_df, Vt_df, phenotype_pos_df, fit_site,
                 covariates_df=None, window=1000000, tau_mode='estimate',
                 se_mode='model', logger=None, verbose=True,
                 ase_covariates_df=SAME_COVARIATES):
    """Shared per-phenotype driver: whiten once per gene, call fit_site for
    every site in the cis window, collect its row dicts."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if logger is None:
        logger = SimpleLogger()
    samples = A_df.columns
    N = len(samples)
    for df, name in ((T_df, 'T'), (Va_df, 'Va'), (Vt_df, 'Vt')):
        assert A_df.columns.equals(df.columns), f"Sample mismatch between A and {name}"
        assert A_df.index.equals(df.index), f"Phenotype mismatch between A and {name}"
    missing = [s for s in samples if s not in set(site_samples)]
    assert not missing, f"{len(missing)} phenotype samples absent from the site data, e.g. {missing[:3]}"
    site_ix = np.array([list(site_samples).index(s) for s in samples])
    if covariates_df is not None:
        assert np.all(samples == covariates_df.index), \
            "Covariate samples must match phenotype samples"
        covariates_t = torch.tensor(covariates_df.values, dtype=torch.float32).to(device)
        n_cov = covariates_df.shape[1]
    else:
        covariates_t = None; n_cov = 0
    ase_covariates_t, n_cov_a = _resolve_ase_covariates(
        ase_covariates_df, covariates_df, samples, device, logger)
    n_cov = max(n_cov, n_cov_a)          # one dof rule for both channels
    robust = se_mode == 'robust'
    pos_df = phenotype_pos_df.loc[phenotype_pos_df.index.isin(A_df.index)]
    cis = _cis_sites(site_chrom, site_pos, pos_df, window)
    logger.write(f'hapmixQTL second pass ({kind})')
    logger.write(f'  * {N} samples, {len(A_df)} phenotypes, {n_sites} sites, '
                 f'{len(cis)} phenotypes with a site in the cis-window (±{window:,})')
    logger.write(f'  * tau mode: {tau_mode}; SE mode: {se_mode}')
    pheno_ix = {pid: i for i, pid in enumerate(A_df.index)}
    rows = []
    t0 = time.time()
    for n, (pid, (sites, start)) in enumerate(cis.items(), 1):
        pidx = pheno_ix[pid]
        a_t = torch.tensor(A_df.values[pidx], dtype=torch.float32).to(device)
        t_t = torch.tensor(T_df.values[pidx], dtype=torch.float32).to(device)
        va_t = torch.tensor(Va_df.values[pidx], dtype=torch.float32).to(device)
        vt_t = torch.tensor(Vt_df.values[pidx], dtype=torch.float32).to(device)
        sqrt_wa_t, sqrt_wt_t, res_a, res_t = _prepare_channels(
            a_t, t_t, va_t, vt_t, covariates_t, tau_mode, device,
            ase_covariates_t=ase_covariates_t)
        ctx = dict(a_t=a_t, t_t=t_t, sqrt_wa_t=sqrt_wa_t, sqrt_wt_t=sqrt_wt_t,
                   res_a=res_a, res_t=res_t, robust=robust, n_cov=n_cov, N=N,
                   device=device, site_ix=site_ix, pid=pid, start=start)
        for j in sites:
            rows.extend(fit_site(int(j), ctx))
        if verbose and n % 500 == 0:
            logger.write(f'    {n}/{len(cis)} phenotypes, {(time.time() - t0) / 60:.1f} min')
    logger.write(f'  * done: {len(rows)} rows in {(time.time() - t0) / 60:.2f} min')
    return rows


def _fit_design(design, ctx):
    """Run _joint_gls on a numpy design dict built for the site's samples."""
    Xa = torch.tensor(design['Xa'], dtype=torch.float32).to(ctx['device'])
    Xt = torch.tensor(design['Xt'], dtype=torch.float32).to(ctx['device'])
    return _joint_gls(Xa, Xt, ctx['a_t'], ctx['t_t'], ctx['sqrt_wa_t'], ctx['sqrt_wt_t'],
                      ctx['res_a'], ctx['res_t'], robust=ctx['robust'], n_cov=ctx['n_cov'])


def _pvals(fit, N, n_cov):
    """Joint F p-value on the estimable columns and per-coefficient t p-values,
    with dof = N - 1 - n_cov - p (p = 1 reproduces the lead scan's dof)."""
    p = int(fit['estimable'].sum())
    dof = max(N - 1 - n_cov - p, 1)
    se = np.sqrt(np.diag(fit['cov']))
    with np.errstate(invalid='ignore', divide='ignore'):
        t = fit['beta'] / se
    p_coef = np.where(np.isfinite(t), 2 * stats.t.sf(np.abs(t), dof), np.nan)
    p_joint = stats.f.sf(fit['chi2'] / p, p, dof) if (fit['rank_ok'] and p > 0
                                                       and np.isfinite(fit['chi2'])) else np.nan
    return p_joint, p, dof, se, p_coef


def map_multiallelic(hap_alleles, site_df, site_samples, A_df, T_df, Va_df, Vt_df,
                     phenotype_pos_df, hap_phased=None, covariates_df=None,
                     window=1000000, min_hap=10, tau_mode='estimate',
                     se_mode='model', logger=None, verbose=True,
                     ase_covariates_df=SAME_COVARIATES):
    """
    Categorical (per-allele) cis-QTL test for multiallelic non-repeat sites:
    the K-1 split-biallelic rows of a site fitted jointly.

    Args:
        hap_alleles: [n_sites, n_samples, 2] int allele index per haplotype
                     (0 = REF, k = k-th ALT), -1 = missing
        site_df:     DataFrame indexed by site id with columns chrom, pos
        site_samples: sample order of hap_alleles' second axis
        hap_phased:  [n_sites, n_samples] bool (default all phased); an
                     unphased heterozygote keeps its total-channel row and
                     contributes no ASE contrast
        min_hap:     alleles carried by fewer haplotypes are pooled into an
                     'other' column (or, if the pool is still too small,
                     treated as missing)
        window, covariates_df, tau_mode, se_mode: as in map_nominal

    Returns:
        site_res_df: one row per (phenotype, site): n_alleles, n_tested,
            ref_allele, pooled_other, pval_joint (F on n_tested df), chi2, dof,
            rank_deficient
        allele_res_df: one row per (phenotype, site, allele): n_hap, slope
            (log aFC of the allele vs the reference allele), slope_se, pval,
            and the same fit within each channel alone (slope_a/slope_t)
    """
    hap_alleles = np.asarray(hap_alleles)
    if hap_phased is None:
        hap_phased = np.ones(hap_alleles.shape[:2], bool)
    hap_phased = np.asarray(hap_phased, bool)
    site_ids = np.asarray(site_df.index)
    site_res, allele_res = [], []

    def fit_site(j, ctx):
        ix = ctx['site_ix']
        d = _categorical_design(hap_alleles[j, ix, 0], hap_alleles[j, ix, 1],
                                hap_phased[j, ix], min_hap)
        if d is None:
            return []
        fit = _fit_design(d, ctx)
        p_joint, p, dof, se, p_coef = _pvals(fit, ctx['N'], ctx['n_cov'])
        se_a = np.sqrt(np.diag(fit['cov_a'])); se_t = np.sqrt(np.diag(fit['cov_t']))
        site_res.append(dict(
            phenotype_id=ctx['pid'], site_id=site_ids[j],
            start_distance=int(site_df['pos'].iloc[j]) - ctx['start'],
            n_alleles=d['n_alleles'], n_tested=p, ref_allele=d['ref'],
            pooled_other=d['pooled'], n_missing_hap=d['n_missing_hap'],
            pval_joint=p_joint, chi2=fit['chi2'], dof=dof,
            rank_deficient=not fit['rank_ok']))
        for k, lab in enumerate(d['labels']):
            allele_res.append(dict(
                phenotype_id=ctx['pid'], site_id=site_ids[j], allele=lab,
                n_hap=int(d['n_hap'][k]), slope=fit['beta'][k], slope_se=se[k],
                pval=p_coef[k], slope_a=fit['beta_a'][k], slope_a_se=se_a[k],
                slope_t=fit['beta_t'][k], slope_t_se=se_t[k]))
        return [1]

    _second_pass('categorical, multiallelic sites', len(site_df),
                 site_df['chrom'].values, site_df['pos'].values, site_samples,
                 A_df, T_df, Va_df, Vt_df, phenotype_pos_df, fit_site,
                 covariates_df, window, tau_mode, se_mode, logger, verbose,
                 ase_covariates_df=ase_covariates_df)
    site_cols = ['phenotype_id', 'site_id', 'start_distance', 'n_alleles', 'n_tested',
                 'ref_allele', 'pooled_other', 'n_missing_hap', 'pval_joint', 'chi2',
                 'dof', 'rank_deficient']
    allele_cols = ['phenotype_id', 'site_id', 'allele', 'n_hap', 'slope', 'slope_se',
                   'pval', 'slope_a', 'slope_a_se', 'slope_t', 'slope_t_se']
    return (pd.DataFrame(site_res, columns=site_cols),
            pd.DataFrame(allele_res, columns=allele_cols))


def map_str_curvature(str_len, str_phased, str_df, site_samples, A_df, T_df, Va_df, Vt_df,
                      phenotype_pos_df, covariates_df=None, window=1000000,
                      winsor=(0.01, 0.99), tau_mode='estimate', se_mode='model',
                      logger=None, verbose=True, ase_covariates_df=SAME_COVARIATES):
    """
    Linear + curvature cis-QTL model for STRs: per haplotype
    f(L) = b1 (L - c) + b2 (L - c)^2 in repeat units, c = cohort mean length.

    The linear-only fit (slope_lin) is the same model the lead scan applies
    to the STR's linear-in-length row and should reproduce its slope (up to
    winsorization). b2 (slope_sq) tests curvature on 1 df: same sign as b1
    means the per-unit effect accelerates with length, opposite sign means it
    saturates. pval_joint2 is the 2-df test of the whole quadratic model.

    Lengths are reference-relative repeat units (0 = the reference allele, as
    the encoder stores them). The quadratic basis is centred on the cohort
    mean c only for numerical conditioning, so slope_l is the slope at L = c;
    slope_at_ref = b1 - 2*c*b2 re-expresses the same fitted curve at the
    reference length (L = 0), with a delta-method SE, so effects are read on
    the reference origin. b2 does not depend on the centring.

    Args:
        str_len:    [n_str, n_samples, 2] float repeat lengths in repeat units
                    per haplotype, NaN = missing
        str_phased: [n_str, n_samples] bool
        str_df:     DataFrame indexed by STR id with columns chrom, pos
        site_samples: sample order of str_len's second axis
        winsor:     quantiles at which haplotype lengths are clipped before
                    building the basis (None = no clipping)
        window, covariates_df, tau_mode, se_mode: as in map_nominal

    Returns one row per (phenotype, STR).
    """
    str_len = np.asarray(str_len, float)
    str_phased = np.asarray(str_phased, bool)
    str_ids = np.asarray(str_df.index)
    out = []

    def fit_site(j, ctx):
        ix = ctx['site_ix']
        d = _str_design(str_len[j, ix, 0], str_len[j, ix, 1], str_phased[j, ix], winsor)
        if d is None:
            return []
        lin = _fit_design(dict(Xa=d['Xa'][:1], Xt=d['Xt'][:1]), ctx)
        p_lin, _, _, se_lin, _ = _pvals(lin, ctx['N'], ctx['n_cov'])
        quad = _fit_design(d, ctx)
        p_joint, p, dof, se, p_coef = _pvals(quad, ctx['N'], ctx['n_cov'])
        se_a = np.sqrt(np.diag(quad['cov_a'])); se_t = np.sqrt(np.diag(quad['cov_t']))
        # The basis is centred on the cohort mean c (conditioning), so slope_l
        # is df/dL at L = c. Re-express the same curve at the encoder's origin,
        # the REFERENCE length (L = 0): f'(0) = b1 - 2 c b2, delta-method SE.
        c = d['center']; b1, b2 = quad['beta'][0], quad['beta'][1]; cv = quad['cov']
        s_ref = b1 - 2.0 * c * b2
        v_ref = cv[0, 0] + 4.0 * c * c * cv[1, 1] - 4.0 * c * cv[0, 1]
        se_ref = float(np.sqrt(v_ref)) if np.isfinite(v_ref) and v_ref >= 0 else np.nan
        out.append(dict(
            phenotype_id=ctx['pid'], str_id=str_ids[j],
            start_distance=int(str_df['pos'].iloc[j]) - ctx['start'],
            n_called=d['n_called'], n_phased=d['n_phased'],
            len_center=d['center'], len_lo=d['lo'], len_hi=d['hi'],
            slope_lin=lin['beta'][0], slope_lin_se=se_lin[0], pval_lin=p_lin,
            slope_l=quad['beta'][0], slope_l_se=se[0],
            slope_sq=quad['beta'][1], slope_sq_se=se[1], pval_curv=p_coef[1],
            slope_at_ref=s_ref, slope_at_ref_se=se_ref,
            slope_sq_a=quad['beta_a'][1], slope_sq_a_se=se_a[1],
            slope_sq_t=quad['beta_t'][1], slope_sq_t_se=se_t[1],
            pval_joint2=p_joint, rank_deficient=not quad['rank_ok']))
        return [1]

    _second_pass('linear + curvature, STRs', len(str_df),
                 str_df['chrom'].values, str_df['pos'].values, site_samples,
                 A_df, T_df, Va_df, Vt_df, phenotype_pos_df, fit_site,
                 covariates_df, window, tau_mode, se_mode, logger, verbose,
                 ase_covariates_df=ase_covariates_df)
    cols = ['phenotype_id', 'str_id', 'start_distance', 'n_called', 'n_phased',
            'len_center', 'len_lo', 'len_hi', 'slope_lin', 'slope_lin_se', 'pval_lin',
            'slope_l', 'slope_l_se', 'slope_sq', 'slope_sq_se', 'pval_curv',
            'slope_at_ref', 'slope_at_ref_se',
            'slope_sq_a', 'slope_sq_a_se', 'slope_sq_t', 'slope_sq_t_se',
            'pval_joint2', 'rank_deficient']
    return pd.DataFrame(out, columns=cols)

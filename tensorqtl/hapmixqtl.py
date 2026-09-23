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
w_i = 1 / Var(e_i), where v_inf_i is the across-draw variance of the
transformed expression for sample i. Three variance models are selectable
(``variance_model`` on every mapping function; VARIANCE_MODELS):

  additive        Var(e_i) = v_inf_i + tau                  (the default)
  two_component   Var(e_i) = c v_inf_i + tau
  library_scaled  Var(e_i) = d_i (c v_inf_i + tau)

tau is a per-phenotype, per-channel between-sample variance, c a per-gene
factor by which the draws understate the measurement error, and d_i a
per-library factor shared by every gene (estimate_library_factors). Under
'additive' tau is the moment estimate of _estimate_tau, fitted under the
channel's null model (the total automatic intercept and covariates, or the
ASE through-origin/covariate design, no genotype term) on the samples that
carry information (v_inf > 0), with the exact leverage denominator
sum_i w_i(1 - h_i) -- DerSimonian-Laird's form for an intercept-only design.
Under the other two models (c, tau) are fitted jointly by _estimate_c_tau,
a damped iterated weighted least squares of the leverage-corrected squared
null residuals on [v, 1], clamped at zero by default or, with
``variance_prior`` (estimate_variance_priors), shrunk toward the gene's
expression bin by an empirical-Bayes prior in place of the clamp. The models
govern the ALLELIC channel; the total
channel keeps v_t + tau_t under every model, because v_t is nearly constant
across samples so c_t is not identifiable, and d_i was measured on the
allelic channel. tau_mode='estimate' is the DEFAULT. tau_mode='zero' asserts
the Gibbs variance is the entire error variance, which is severely
anticonservative on real data; it is retained only to reproduce earlier
results, warns when used, and is only accepted with the additive model.
Measured on BrainVar (estimator_ablation_20260916, estimator_ablation_tiers_20260917):
the additive model is anticonservative at low expression (type-I 0.068 at
nominal 0.05) and conservative at high (0.022); the two other models are
0.029-0.041 in every tier with the same null width and calls; only
library_scaled leaves whitened residuals with no per-library spread.

Four further things shape what the mapping functions do:

  * Per-channel covariate designs. covariates_df is the TOTAL channel's;
    ase_covariates_df is the ALLELIC channel's -- SAME_COVARIATES reuses the
    total channel's, while its public default None is through-origin. The
    allelic contrast is a within-sample difference in which covariates
    acting on both haplotypes alike cancel, so None is the usual choice on
    real data: each column projected out costs one informative sample.
    scripts/compare_pipelines.py defaults its allelic channel to a
    through-origin (--ase-covariates none).
  * A sparse-channel rule. A channel with fewer informative samples than its
    design has columns plus two is switched off (all weights zero) and the
    meta-analysis falls back to the other channel.
  * map_cis's permutation null is Freedman-Lane in whitened space: each
    channel's null residuals are leverage-standardized and permuted among
    that channel's own informative samples. map_cis therefore REQUIRES
    se_mode='model' and raises otherwise -- the permutation statistic is the
    known-variance GLS statistic and has no sandwich counterpart; robust
    standard errors are available in map_nominal only.
  * map_cis(tau_refit=True) re-estimates each channel's tau with the lead's
    predictor in the design and reports the lead's slope, SE and nominal p
    on that scale ALONE. pval_perm and pval_beta stay on the scan scale,
    where they are calibrated, and map_nominal stays on the null-model scale.

docs/hapmixqtl_methods.md specifies all of this for reproduction.

Phase determines the signed heterozygote indicator s_i = xL_i - xR_i:
  s = +1 if ALT allele is on haplotype L
  s = -1 if ALT allele is on haplotype R
  s =  0 if homozygous (or phase unknown)
When phase is unavailable (s=0 for all samples), the ASE channel contributes
nothing and results match total-channel-only regression. The phase frames are
indexed POSITIONALLY by the genotype frame's column order; _assert_phase_columns
guards that at every entry point, because the same samples in a different order
silently corrupts the allelic channel while leaving the total channel correct.

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
asserts it in the test suite). Adding a 2*w_a*w_t*Cat term to the combined SE would
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


def _assert_phase_columns(xL_df, xR_df, genotype_df):
    """The phase frames are indexed POSITIONALLY by the genotype frame's column
    order (genotype_ix is built from genotype_df.columns, then applied to
    xL_df.values). If the frames carry the same samples in a different order,
    the allelic channel is silently computed from the wrong samples while the
    total channel stays correct: on a planted effect of 0.9 the combined slope
    came back 0.47 with the total-channel slope still reading 0.95, and nothing
    raised. Checked once per call."""
    for name, df in (('xL_df', xL_df), ('xR_df', xR_df)):
        if not genotype_df.columns.equals(df.columns):
            if set(df.columns) == set(genotype_df.columns):
                raise ValueError(
                    f'{name} has the same samples as genotype_df in a different order. '
                    'Phase is indexed positionally by the genotype column order, so this '
                    f'would corrupt the allelic channel silently. Reindex with '
                    f'{name} = {name}[genotype_df.columns].')
            raise ValueError(
                f'{name} columns do not match genotype_df columns '
                f'({len(df.columns)} vs {len(genotype_df.columns)} samples).')


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


def compute_summaries_from_gibbs(yL, yR, kappa=0.5, yT=None, count_noise=True):
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
        count_noise: add per-sample Poisson counting noise to Va and Vt.
            DEFAULT TRUE since 2026-09-13: three separate failures close when it
            is on and are open when it is off. A sample with zero counts in every
            draw drives the total channel's type-I error to 52% at alpha = 0.05;
            a sample whose draws are unanimous because its reads are unambiguous
            rather than absent is discarded as uninformative, throwing away the
            most informative allele-specific observation in the channel; and the
            tau moment estimator collapses on either. Pass False only to
            reproduce results from before that date. This
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


def count_cutoff_masks(yL, yR, yT=None, asc_cutoff=None, asc_cap=None,
                       trc_cutoff=None):
    """Per-channel admission masks from mixQTL-style count cutoffs.

    hapmixQTL has no count cutoffs of its own: every donor with allelic
    information enters the allelic channel and every donor enters the total
    channel. This builds the masks that reproduce mixQTL's count thresholds,
    so the two estimators can be run on a MATCHED donor set. It is the caller's
    job to pass the masks on; nothing here is applied by default.

    The cutoffs (mixQTL's names, and its GTEx v8 published values):

        asc_cutoff  50    BOTH haplotype counts must be >= this
        asc_cap   1000    BOTH haplotype counts must be <= this
        trc_cutoff 100    total counts must be >= this

    Each is optional and skipped when None, so a caller can apply the floor
    without the ceiling.

    WHICH COUNT EACH CUTOFF READS matters and is easy to get wrong.
    ``trc_cutoff`` reads the TOTAL count ``yT``, summed over every transcript,
    exactly as mixQTL's trcQTL does (``mixqtl_replication.trc_channel``
    thresholds ``ytotal``). It must NOT be applied to ``yL + yR``, which is a
    paired-transcript subtotal: where a donor is homozygous across a gene,
    Salmon's deduplicated index leaves no ``_R`` row, the ingest credits
    neither haplotype, and ``yL + yR`` is 0 while ``yT`` is whatever the gene
    actually expressed. On the 29 calibration genes, thresholding ``yL + yR``
    at 100 excludes 502 donor-gene pairs that ``yT >= 100`` admits -- 18.8% of
    the cohort, silently, and exactly the homozygous-but-expressed pairs.
    ``yT`` defaults to ``yL + yR`` for signature consistency with
    ``compute_summaries_from_gibbs``, and that default is a HAZARD rather
    than a convenience here: it is the wrong quantity for ``trc_cutoff``.
    Always pass the real total when using a total cutoff. The runner does;
    the default is only reachable by calling this directly.

    THESE CUTOFFS ARE NOT RECOMMENDED AS A DEFAULT on Salmon posterior means.
    mixQTL's Methods justify the ``asc_cap`` as an alignment-artifact guard
    ("very large allele-specific counts to be likely alignment artifacts"),
    which does not transfer: a posterior-mean abundance above 1000 is a
    well-expressed gene, not a pileup. Measured on the 29 calibration genes,
    the published band keeps 499 of 2,193 informative donor-gene pairs, losing
    1,656 to the ceiling against 38 to the floor, while ``trc_cutoff=100``
    excludes nobody. See
    ``/mnt/ssd/lalli/brainvar_hapmix_deploy/mixqtl_replication_20260919/REPORT.md``.

    Args:
        yL, yR: haplotype counts, either [features, samples] posterior means
                or [features, samples, draws] Gibbs draws (averaged here)
        yT:     total counts, same shape; defaults to yL + yR
        asc_cutoff, asc_cap, trc_cutoff: thresholds, or None to skip

    Returns:
        (keep_a, keep_t), each a boolean [features, samples] array. A False
        entry means that donor contributes nothing to that channel for that
        gene; the weight is set to zero and the donor is excluded from the
        channel's tau and variance fits, exactly as a zero-coverage donor is.
    """
    yL = np.asarray(yL, dtype=float)
    yR = np.asarray(yR, dtype=float)
    mL = yL.mean(axis=2) if yL.ndim == 3 else yL
    mR = yR.mean(axis=2) if yR.ndim == 3 else yR
    if yT is None:
        mT = mL + mR
    else:
        yT = np.asarray(yT, dtype=float)
        mT = yT.mean(axis=2) if yT.ndim == 3 else yT

    keep_a = np.ones(mL.shape, dtype=bool)
    if asc_cutoff is not None:
        keep_a &= (mL >= asc_cutoff) & (mR >= asc_cutoff)
    if asc_cap is not None:
        keep_a &= (mL <= asc_cap) & (mR <= asc_cap)

    keep_t = np.ones(mT.shape, dtype=bool)
    if trc_cutoff is not None:
        keep_t &= mT >= trc_cutoff
    return keep_a, keep_t


def _keep_row(keep_df, pidx, device):
    """One phenotype's admission mask as a bool tensor, or None if unmasked."""
    if keep_df is None:
        return None
    return torch.tensor(np.asarray(keep_df.values[pidx], dtype=bool)).to(device)


def _log_keep_frames(keep_a_df, keep_t_df, logger):
    """Say what the cutoffs admit, so a filtered run is never silent."""
    if logger is None:
        return
    for label, df in (('allelic', keep_a_df), ('total', keep_t_df)):
        if df is None:
            continue
        n_keep, n = int(df.values.sum()), int(df.values.size)
        logger.write(f'  * count cutoffs on the {label} channel: '
                     f'{n_keep:,}/{n:,} donor-gene pairs admitted '
                     f'({n_keep / n:.1%})')


def _assert_keep_frames(keep_a_df, keep_t_df, A_df):
    """Admission masks must be indexed exactly like the phenotype frames.

    Positional misalignment here is the same defect class as the phase-column
    bug ``_assert_phase_columns`` guards: it silently excludes the wrong
    donors and the run still completes, so it is checked rather than trusted.
    """
    for name, df in (('keep_a_df', keep_a_df), ('keep_t_df', keep_t_df)):
        if df is None:
            continue
        if df.shape != A_df.shape:
            raise ValueError(
                f'{name} has shape {df.shape}, expected {A_df.shape} to match '
                'the phenotype frames')
        if not df.index.equals(A_df.index):
            raise ValueError(f'{name} rows are not the phenotype index of A_df')
        if not df.columns.equals(A_df.columns):
            raise ValueError(f'{name} columns are not the samples of A_df')


# ---------------------------------------------------------------------------
#  WeightedResidualizer
# ---------------------------------------------------------------------------

class WeightedResidualizer:
    """
    Residualizer for weighted least squares via sqrt-weight transform.

    When ``intercept=True``, a constant intercept alpha becomes
    alpha*sqrt(w_i) after the sqrt-weight transform. This class includes
    sqrt(w) as an explicit design column so the QR projection removes it
    correctly. ``intercept=False`` retains a through-origin fit; with no
    covariates it has an empty design and is the identity transform.
    """

    def __init__(self, C_t, sqrt_w_t, intercept=True):
        """
        Args:
            C_t: covariates [N, n_cov] (without automatic intercept), or None
            sqrt_w_t: sqrt per-sample weights [N]
            intercept: include the automatic weighted intercept
        """
        N = sqrt_w_t.shape[0]
        cols = []
        if intercept:
            cols.append(sqrt_w_t.unsqueeze(1))
        if C_t is not None and C_t.numel() > 0 and C_t.shape[1] > 0:
            C_star = sqrt_w_t.unsqueeze(1) * C_t
            cols.append(C_star)
        if cols:
            design = torch.cat(cols, dim=1)
        else:
            design = sqrt_w_t.new_zeros((N, 0))
        # kept for the donor-record permutation, which rebuilds the whitened
        # design from permuted weights and covariate rows
        self.C_t = C_t if (C_t is not None and C_t.numel() > 0 and C_t.shape[1] > 0) else None
        self.intercept = bool(intercept)
        self.sqrt_w_t = sqrt_w_t
        if design.shape[1] > 0 and bool((sqrt_w_t != 0).any()):
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

def _wls_regression(y_star_t, x_star_t, residualizer, robust=False, fitted=False):
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
        fitted: if True use the estimated-dispersion SE sigma_hat/sqrt(xx)
            instead, which makes the weights a shape only. Takes precedence
            over ``robust``. This is mixQTL's Eq 11 treatment; it gives up the
            absolute propagation of inferential variance described above in
            exchange for immunity to getting that absolute scale wrong.

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
        if fitted:
            # Estimated-dispersion WLS: Var(beta_hat) = sigma2_hat / xx, with
            # sigma2_hat the weighted residual mean square. The weights are
            # then only a SHAPE -- rescaling them all by any constant leaves
            # beta_hat and this SE exactly unchanged, because sigma2_hat
            # absorbs the factor. That is mixQTL's Eq 11 model, and it is the
            # one configuration in which the draws' absolute scale does not
            # have to be right; see docs and the mixQTL comparison reports.
            #
            # dof counts INFORMATIVE donors, not rows. Zero-weight samples
            # contribute nothing to rss (y* and x* are both 0 there), so
            # charging them degrees of freedom would shrink sigma2_hat and
            # understate the SE. residualizer.dof is N-1-ncol over all rows,
            # which is wrong here for exactly that reason.
            e = y_res - slope.unsqueeze(1) * x_res
            rss = (e * e).sum(1)
            n_eff = int((residualizer.sqrt_w_t != 0).sum())
            dof_f = max(n_eff - 1 - residualizer.Q_t.shape[1], 1)
            slope_se[valid] = torch.sqrt(rss[valid] / dof_f / xx[valid])
        elif not robust:
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


def _estimate_tau(y_t, v_inf_t, covariates_t, device, intercept=True):
    """
    Estimate overdispersion parameter tau using moment estimator.

    Under Var(error_i) = v_inf_i + tau, whitening by w_i = 1/v_inf_i gives
    Var(y*_i) = 1 + tau*w_i, so the residual sum of squares after projecting
    out the null design P = QQ' has expectation

        E[RSS] = tr((I - P) diag(1 + tau*w)) = (n - q) + tau * sum_i w_i(1-h_i)

    with h_i the leverage (the diagonal of P) and q the design's rank. Solving
    gives the estimator below. The denominator is sum_i w_i(1-h_i), NOT
    (n - q) * mean(w): the two agree only when every sample has the same
    leverage, and for an intercept-only design (h_i = w_i / sum_j w_j) the
    correct form reduces exactly to DerSimonian and Laird's
    sum_i w_i - sum_i w_i^2 / sum_j w_j. Using the mean weight understated tau
    by a median 0.8% in the allelic channel and 3.0% in the total channel of
    the BrainVar genes, whose whitened 18-column design reaches a leverage of
    0.74.
    """
    w = 1.0 / v_inf_t.clamp(min=1e-8)
    sqrt_w = torch.sqrt(w)
    res = WeightedResidualizer(covariates_t, sqrt_w, intercept=intercept)
    y_star = (y_t * sqrt_w).unsqueeze(0)
    y_res = res.transform(y_star).squeeze()

    rss = (y_res * y_res).sum()
    dof_null = y_t.shape[0] - res.Q_t.shape[1]
    h = (res.Q_t * res.Q_t).sum(1)
    denom = (w * (1.0 - h)).sum()
    tau = torch.clamp((rss - dof_null) / denom.clamp(min=1e-30), min=0.0)
    return tau


# ---------------------------------------------------------------------------
#  Association tests
# ---------------------------------------------------------------------------

def _estimate_tau_informative(y_t, v_inf_t, covariates_t, device, eps=1e-12,
                              intercept=True):
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
    if n_keep < _min_informative(covariates_t, intercept=intercept):
        n_cols = int(intercept) + (0 if covariates_t is None else covariates_t.shape[1])
        raise ValueError(
            f'tau cannot be estimated from {n_keep} informative samples against '
            f'a design of {n_cols} columns; the channel must be switched off')
    c = None if covariates_t is None else covariates_t[keep]
    return _estimate_tau(y_t[keep], v_inf_t[keep], c, device, intercept=intercept)


def _min_informative(covariates_t, extra=2, intercept=True):
    """Informative samples a channel needs to stay on: one per design column
    (automatic intercept, if present, plus covariates) plus `extra` residual
    degrees of freedom."""
    n_cov = 0 if covariates_t is None else covariates_t.shape[1]
    return int(intercept) + n_cov + extra


# DEPRECATED (2026-09-23), all three. The shipped error model is
#     Var(eps_i) = sigma^2 * v_i        (TIMES, no additive floor)
# reached by tau_mode='zero' + se_mode='fitted'. These three, the
# variance_prior shrinkage and the tau_mode='estimate' they require are kept
# ONLY to reproduce historical results and WILL BE REMOVED. They are not
# alternatives and not a fallback. They are inferior for a structural reason:
# each fits its layer-1 variance from a gene's own squared residuals and then
# weights those same residuals, which no comparator method does, and the
# free-c forms are additionally invariant to the absolute scale of the Gibbs
# draws so the quantifier's calibration never reaches the answer. Per-gene
# efficiency comparisons do not rehabilitate them; efficiency was never the
# objection. See CLAUDE.md, "The variance models are deprecated".
VARIANCE_MODELS = ('additive', 'two_component', 'library_scaled')


def _check_variance_model(variance_model, tau_mode, library_factor_t, variance_prior=None):
    """Argument validation shared by the mapping functions."""
    if variance_prior is not None:
        if variance_model == 'additive':
            raise ValueError("variance_prior applies to the two-component models only; c is fixed at 1 under 'additive'")
        if not isinstance(variance_prior, pd.DataFrame) or 'prior_c' not in variance_prior.columns:
            raise ValueError('variance_prior must be the DataFrame returned by estimate_variance_priors')
        if 'library_scaled' not in variance_prior.attrs or 'kappa' not in variance_prior.attrs:
            raise ValueError('variance_prior has lost its attrs (kappa, library_scaled): pass the frame '
                             'estimate_variance_priors returned, not one reloaded from a file')
        if bool(variance_prior.attrs['library_scaled']) != (variance_model == 'library_scaled'):
            raise ValueError("variance_prior was estimated for a different model: pass library_factor to "
                             "estimate_variance_priors exactly when the scan uses 'library_scaled'")
    if variance_model not in VARIANCE_MODELS:
        raise ValueError(f'variance_model must be one of {VARIANCE_MODELS}, got {variance_model!r}')
    if variance_model != 'additive' and tau_mode != 'estimate':
        raise ValueError(f"variance_model={variance_model!r} requires tau_mode='estimate'")
    if variance_model == 'library_scaled' and library_factor_t is None:
        raise ValueError(
            "variance_model='library_scaled' needs library_factor: one positive value per "
            "sample, estimated across genes with estimate_library_factors(A_df, Va_df, ...)")
    if variance_model != 'library_scaled' and library_factor_t is not None:
        raise ValueError("library_factor is only used under variance_model='library_scaled'")


def _library_factor_tensor(library_factor, samples, device):
    """library_factor as a float32 tensor aligned to the phenotype samples, or
    None. Accepts a Series indexed by sample (reindexed, every sample
    required) or an array in phenotype column order."""
    if library_factor is None:
        return None
    if isinstance(library_factor, pd.Series):
        d = library_factor.reindex(samples).values.astype(float)
    else:
        d = np.asarray(library_factor, dtype=float)
        assert d.shape == (len(samples),), 'library_factor must have one value per sample'
    if not np.all(np.isfinite(d)) or np.any(d <= 0):
        raise ValueError('library_factor must be finite and positive for every sample')
    return torch.tensor(d, dtype=torch.float32).to(device)


def _estimate_c_tau(y_t, v_t, covariates_t, device, intercept=True, d_t=None,
                    max_iter=500, tol=1e-7, prior=None, _theta0=None):
    """
    Two-component variance fit for one channel of one gene,

        Var(e_i) = d_i (c v_i + tau),   c >= 0, tau >= 0,

    on the informative samples (the caller drops v <= eps). d_t is the
    per-library factor (None means 1 for every sample).

    Under weights w_i = 1/(d_i (c v_i + tau)) the leverage-corrected squared
    null residual e2_i = r_i^2 / (w_i (1 - h_i)) has expectation
    d_i (c v_i + tau), so e2_i / d_i regressed on [v_i, 1] gives the slope c
    and intercept tau. A squared Gaussian residual has variance proportional
    to its variance squared, so the line is fitted with weights
    1/(c v_i + tau)^2; the weights depend on (c, tau), so the fit iterates,
    damped by averaging each update with the previous value (the undamped
    update can cycle between the tau = 0 clamp and an interior point). When
    a clamp hits, the other parameter is refitted alone. Starts at c = 1 and
    the DerSimonian-Laird tau of y/sqrt(d). Runs in float64.

    This is the estimator whose calibration was measured on BrainVar (300
    genes across expression tiers, 40 genotype permutations each: type-I
    0.029-0.041 at nominal 0.05; estimator_ablation_tiers_20260917/
    tiered_calibration.py, fit_cvt). With d = 1 it is that prototype exactly;
    with d != 1 and an empty design it equals fitting y/sqrt(d) against v,
    which is what the prototype's library-scaled configuration did.

    ``prior`` replaces the clamp with empirical-Bayes shrinkage: a tuple
    (m_logc, s_logc, m_logtau, s_logtau, kappa) from estimate_variance_priors,
    independent normal priors on log c and log tau. The fit is then the
    posterior mode in (log c, log tau) of the Gaussian likelihood of the
    leverage-corrected squared residuals, tempered by 2/kappa (kappa =
    Var(z^2) of the standardized residuals; 2 under Gaussian errors, larger
    with heavy tails), found by Fisher scoring with a backtracking line
    search on the penalized objective. The unpenalized
    stationary point of that likelihood is the same weighted regression of
    e^2 on [v, 1] as the clamped fit (Fisher scoring for a Gaussian variance
    model is that iteration), so the two estimators agree away from the
    boundary; on the log scale positivity is automatic and nothing is
    clamped. Each scoring step is backtracked (halved until the penalized
    objective, the gamma quasi-log-likelihood plus the log prior, does not
    decrease): without that, the step overshoots along the direction the
    gene's data do not identify and the iteration cycles between two points
    at the step cap. The ascent is run from two starts, the prior mean and
    the unpenalized clamped fit, and the higher objective is kept: under a
    prior centred far below a gene's identified c the objective is bimodal
    and the ascent from the prior mean alone stops in the spurious mode near
    the prior. On the prior path ``floored`` is always False (nothing is
    clamped); on the clamped path it reports whether a clamp branch was taken
    on the final iteration.

    Returns a dict: c, tau (the values the weights use), converged,
    c_raw, tau_raw (the unpenalized, unclamped solution at the final
    weights), floored. A fit that has not met the tolerance after
    ``max_iter`` iterations returns its last iterate with converged False.
    """
    y = y_t.to(torch.float64)
    v = v_t.to(torch.float64)
    cov = None if covariates_t is None else covariates_t.to(torch.float64)
    d = torch.ones_like(v) if d_t is None else d_t.to(torch.float64)
    c = 1.0
    tau = float(_estimate_tau(y / torch.sqrt(d), v, cov, device, intercept=intercept)) \
        if y.shape[0] > 3 else 0.0
    ones = torch.ones_like(v)
    converged = False
    hit = False
    c_raw, tau_raw = float('nan'), float('nan')

    def _moments(c_, tau_):
        base = (c_ * v + tau_).clamp(min=1e-10)
        w = 1.0 / (d * base)
        sw = torch.sqrt(w)
        res = WeightedResidualizer(cov, sw, intercept=intercept)
        r = res.transform((y * sw).unsqueeze(0))[0]
        h = (res.Q_t * res.Q_t).sum(1)
        e2 = (r * r) / (w * (1.0 - h).clamp(min=1e-3)) / d
        om = 1.0 / (base * base)
        X = torch.stack([v, ones], 1)
        return base, e2, om, X

    if prior is None:
        for _ in range(max_iter):
            base, e2, om, X = _moments(c, tau)
            Am = X.T @ (om[:, None] * X)
            b = X.T @ (om * e2)
            try:
                sol = torch.linalg.solve(Am, b)
                c_raw, tau_raw = float(sol[0]), float(sol[1])
            except Exception:
                c_raw, tau_raw = c, tau
            cn, tn = c_raw, tau_raw
            hit = False
            if cn < 0:
                cn = 0.0
                tn = float((om * e2).sum() / om.sum())
                hit = True
            if tn < 0:
                tn = 0.0
                cn = float((om * e2 * v).sum() / (om * v * v).sum())
                hit = True
            if cn <= 0 and tn <= 0:
                cn, tn = 1e-6, 1e-6
            cn, tn = 0.5 * (c + cn), 0.5 * (tau + tn)
            done = abs(cn - c) < tol * (1 + c) and abs(tn - tau) < tol * (1 + tau)
            c, tau = cn, tn
            if done:
                converged = True
                break
        return dict(c=c, tau=tau, converged=converged, c_raw=c_raw, tau_raw=tau_raw, floored=hit)

    m_logc, s_logc, m_logtau, s_logtau, kappa = [float(x) for x in prior]
    dev64 = dict(dtype=torch.float64, device=y.device)
    m = torch.tensor([m_logc, m_logtau], **dev64)
    P = torch.tensor([[1.0 / s_logc ** 2, 0.0], [0.0, 1.0 / s_logtau ** 2]], **dev64)

    def _merit(th):
        # The objective whose gradient is the tempered score below: Wedderburn's
        # quasi-log-likelihood for mean c v + tau and variance function 2 mu^2,
        # Q = sum(-e2/base - log base), times 2/kappa, plus the log prior. The
        # factor must match the score's 1/kappa exactly: with a 0.5 here the
        # line search accepts only moves toward an objective in which the prior
        # weighs twice as much, and the iteration stalls between the two modes
        # (measured: 190 of 200 identified genes moved, up to a factor 9).
        base, e2, om, X = _moments(float(torch.exp(th[0])), float(torch.exp(th[1])))
        ll = -float((torch.log(base) + e2 / base).sum()) / kappa
        dth = th - m
        return ll - 0.5 * float(dth @ (P @ dth)), (base, e2, om, X)

    if _theta0 is None:
        # The penalized objective is bimodal under a low-centred prior (the
        # two lowest expression bins on BrainVar have a prior median c of
        # 0.0018): an ascent from the prior mean alone settles in the spurious
        # low-c mode for a gene whose data identify c near 2, and reports it
        # converged, 12 log-posterior units below the mode. Ascend from both
        # the prior mean and the unpenalized clamped fit and keep the higher
        # objective; the tau seed is floored because the clamped fit often
        # returns tau = 0 exactly.
        cl = _estimate_c_tau(y_t, v_t, covariates_t, device, intercept, d_t=d_t,
                             max_iter=max_iter, tol=tol, prior=None)
        alt = [float(np.log(max(cl['c'], 1e-8))),
               float(np.log(max(cl['tau'], float(np.exp(m_logtau)) * 1e-3)))]
        a = _estimate_c_tau(y_t, v_t, covariates_t, device, intercept, d_t=d_t,
                            max_iter=max_iter, tol=tol, prior=prior,
                            _theta0=[m_logc, m_logtau])
        b = _estimate_c_tau(y_t, v_t, covariates_t, device, intercept, d_t=d_t,
                            max_iter=max_iter, tol=tol, prior=prior, _theta0=alt)
        ma = _merit(torch.tensor([np.log(a['c']), np.log(a['tau'])], **dev64))[0]
        mb = _merit(torch.tensor([np.log(b['c']), np.log(b['tau'])], **dev64))[0]
        return b if mb > ma else a
    theta = torch.tensor([float(_theta0[0]), float(_theta0[1])], **dev64)
    merit, (base, e2, om, X) = _merit(theta)
    for _ in range(max_iter):
        Am = X.T @ (om[:, None] * X)
        b = X.T @ (om * e2)
        try:
            sol = torch.linalg.solve(Am, b)
            c_raw, tau_raw = float(sol[0]), float(sol[1])
        except Exception:
            c_raw, tau_raw = float(torch.exp(theta[0])), float(torch.exp(theta[1]))
        # tempered Gaussian score and Fisher information in (c, tau), then in (log c, log tau)
        g = (X.T @ (om * (e2 - base))) / kappa
        J = torch.diag(torch.exp(theta))
        g_th = J @ g
        I_th = J @ (Am / kappa) @ J
        try:
            step = torch.linalg.solve(I_th + P, g_th - P @ (theta - m))
        except Exception:
            break
        # Cap the step at one unit on the log scale, scaling the whole vector
        # so its direction is kept: clamping each component separately turns
        # the ascent direction into one that need not ascend, and the search
        # below then stalls at a point that is not a maximum (measured with a
        # near-flat prior: 5 of 22 identified genes stopped up to a factor 67
        # from the clamped fit; scaled, all agree to within 0.7%).
        step_max = float(step.abs().max())
        if step_max > 1.0:
            step = step / step_max
        # Backtracking line search. The expected information is nearly zero
        # along the direction the gene's data do not identify (log c when c v
        # is negligible against tau, log tau in the opposite case), so the full
        # step overshoots there and, undamped, the iteration settles into a
        # two-cycle at the step cap (12.7% of BrainVar genes did). Halving the
        # step until the penalized objective does not decrease makes every
        # iteration an ascent and leaves a step that already ascends untouched.
        # Thirty halvings are needed: with twelve, 58 of 60 such genes still
        # failed to converge. Along that flat direction the accepted step
        # shrinks slowly, so convergence is also declared when the objective's
        # gain is negligible; 1e-10 relative reproduces the step-rule endpoint
        # to 0.1% and converges every gene (median 20 iterations, at most 350).
        accepted = False
        for _k in range(30):
            cand = theta + step
            merit_c, mom_c = _merit(cand)
            if merit_c >= merit:
                accepted = True
                break
            step = 0.5 * step
        if not accepted:
            converged = True  # no ascent at any scale: the gradient is numerically zero
            break
        gain = merit_c - merit
        theta, merit, (base, e2, om, X) = cand, merit_c, mom_c
        if float(step.abs().max()) < tol or gain < 1e-10 * (1.0 + abs(merit)):
            converged = True
            break
    c, tau = float(torch.exp(theta[0])), float(torch.exp(theta[1]))
    return dict(c=c, tau=tau, converged=converged, c_raw=c_raw, tau_raw=tau_raw, floored=False)


def _fit_c_tau_vectorized(a2, v, M, max_iter=400, tol=1e-7):
    """_estimate_c_tau for many genes at once, through-origin and without
    covariates (h = 0, so e2 = a^2), in numpy: rows are genes, M marks the
    informative samples. Same start, weights, clamps, damping and tolerance
    as the per-gene fit. Returns (c, tau) arrays."""
    a2 = np.where(M, a2, 0.0)
    v = np.where(M, v, 1.0)
    w0 = np.where(M, 1.0 / v, 0.0)
    n = M.sum(1)
    tau = np.clip(((a2 * w0).sum(1) - n) / np.maximum(w0.sum(1), 1e-300), 0.0, None)
    c = np.ones(a2.shape[0])
    active = np.ones(a2.shape[0], bool)
    for _ in range(max_iter):
        base = np.maximum(c[:, None] * v + tau[:, None], 1e-10)
        om = np.where(M, 1.0 / (base * base), 0.0)
        Svv, Sv1, S11 = (om * v * v).sum(1), (om * v).sum(1), om.sum(1)
        Sve, S1e = (om * v * a2).sum(1), (om * a2).sum(1)
        det = Svv * S11 - Sv1 ** 2
        ok = det > 1e-300
        safe = np.where(ok, det, 1.0)
        cn = np.where(ok, (Sve * S11 - S1e * Sv1) / safe, c)
        tn = np.where(ok, (Svv * S1e - Sv1 * Sve) / safe, tau)
        neg_c = cn < 0
        cn = np.where(neg_c, 0.0, cn)
        tn = np.where(neg_c, S1e / np.maximum(S11, 1e-300), tn)
        neg_t = tn < 0
        tn = np.where(neg_t, 0.0, tn)
        cn = np.where(neg_t, Sve / np.maximum(Svv, 1e-300), cn)
        both = (cn <= 0) & (tn <= 0)
        cn = np.where(both, 1e-6, cn)
        tn = np.where(both, 1e-6, tn)
        cn, tn = 0.5 * (c + cn), 0.5 * (tau + tn)
        done = (np.abs(cn - c) < tol * (1 + c)) & (np.abs(tn - tau) < tol * (1 + tau))
        c = np.where(active, cn, c)
        tau = np.where(active, tn, tau)
        active &= ~done
        if not active.any():
            break
    return c, tau


def _raw_c_tau_and_cov(a2, v, M, c, tau, kappa=2.0, robust=False):
    """At the weights implied by (c, tau) per gene, the unclamped
    weighted-least-squares solution of a^2 on [v, 1] and its sampling
    covariance. With ``robust=False`` the model-based kappa (X' Omega X)^-1,
    kappa = Var(z^2) of the standardized residuals, which is right when (c,
    tau) are the gene's own converged fit. With ``robust=True`` the sandwich
    (X' Omega X)^-1 X' Omega diag(r^2) Omega X (X' Omega X)^-1 with r the
    residuals of a^2 about the unclamped line, which stays honest when the
    weights come from elsewhere (the trend prior's second pass): a gene far
    from the curve then keeps a large sampling variance instead of the
    spuriously small model-based one. Returns c_raw, tau_raw, var_c, var_tau
    (arrays over genes)."""
    a2 = np.where(M, a2, 0.0)
    v = np.where(M, v, 1.0)
    base = np.maximum(c[:, None] * v + tau[:, None], 1e-10)
    om = np.where(M, 1.0 / (base * base), 0.0)
    Svv, Sv1, S11 = (om * v * v).sum(1), (om * v).sum(1), om.sum(1)
    Sve, S1e = (om * v * a2).sum(1), (om * a2).sum(1)
    det = np.maximum(Svv * S11 - Sv1 ** 2, 1e-300)
    c_raw = (Sve * S11 - S1e * Sv1) / det
    tau_raw = (Svv * S1e - Sv1 * Sve) / det
    if not robust:
        return c_raw, tau_raw, kappa * S11 / det, kappa * Svv / det
    r = np.where(M, a2 - (c_raw[:, None] * v + tau_raw[:, None]), 0.0)
    w2r2 = om * om * r * r
    Bvv, Bv1, B11 = (w2r2 * v * v).sum(1), (w2r2 * v).sum(1), w2r2.sum(1)
    var_c = (S11 * S11 * Bvv - 2 * S11 * Sv1 * Bv1 + Sv1 * Sv1 * B11) / (det * det)
    var_tau = (Sv1 * Sv1 * Bvv - 2 * Sv1 * Svv * Bv1 + Svv * Svv * B11) / (det * det)
    return c_raw, tau_raw, np.maximum(var_c, 1e-300), np.maximum(var_tau, 1e-300)


PRIOR_METHODS = ('deciles', 'trend')


def _trend_prior(x, raw, var, span=0.15, min_width=0.25, floor_abs=0.2, n_grid=40, n_nodes=24):
    """A smooth empirical-Bayes prior on the log scale for a positive parameter,
    fitted by local marginal likelihood from the raw, unbiased, possibly
    negative per-gene estimates: raw_g ~ N(p_g, var_g) and log p_g ~ N(m(x_g),
    s(x_g)^2) with x the log10 expression. At each of ``n_grid`` grid points
    (quantiles of x) the kernel-weighted log marginal likelihood, the integral
    over log p done by Gauss-Hermite quadrature with ``n_nodes`` nodes, is
    maximized over a local line for the mean and a local constant for the
    spread (limma's trend=TRUE idea for a variance prior, with the
    normal-lognormal deconvolution in place of a moment match). Every gene
    enters, a non-positive raw estimate included: it says the parameter is
    small relative to its sampling error and pulls the curve down where such
    genes are common, which a fit restricted to positive estimates would
    miss (measured on BrainVar: that restriction put the prior for tau above
    1,000 reads at 0.029 where the clamped fits' median is 0.003).

    Windows are tricube kernels whose half-width at each grid point is the
    distance to the ``span``-fraction nearest neighbour, never below
    ``min_width``. The spread is bounded below by ``floor_abs`` (0.2 on the log
    scale: a prior tighter than about 20% would over-shrink identified
    genes). Raw estimates are winsorized at the 0.5th and 99.5th percentiles.
    Returns (grid, m, s), to be read by interpolation, flat beyond the grid.
    """
    from scipy.optimize import minimize
    from scipy.special import logsumexp
    x = np.asarray(x, float); raw = np.asarray(raw, float); var = np.asarray(var, float)
    n = len(x)
    lo, hi = np.percentile(raw, [0.5, 99.5]); raw = np.clip(raw, lo, hi)
    sig = np.sqrt(np.maximum(var, 1e-12))
    k = max(int(round(span * n)), 10)
    grid = np.quantile(x, np.linspace(0.01, 0.99, n_grid))
    grid = np.unique(np.concatenate([[x.min()], grid, [x.max()]]))
    xs = np.sort(x)
    t, wq = np.polynomial.hermite.hermgauss(n_nodes)
    logw = np.log(wq) - 0.5 * np.log(np.pi)
    sqrt2 = np.sqrt(2.0)
    pos = raw > 0
    m = np.empty(len(grid)); sd = np.empty(len(grid))
    theta = None
    for i, x0 in enumerate(grid):
        d = np.abs(xs - x0)
        h = max(float(np.partition(d, min(k, n - 1))[min(k, n - 1)]), min_width)
        u = (x - x0) / h
        K = np.where(np.abs(u) < 1, (1 - np.abs(u) ** 3) ** 3, 0.0)
        idx = np.nonzero(K > 0)[0]
        Kw, xw, rw, sw = K[idx], x[idx] - x0, raw[idx], sig[idx]
        if theta is None:
            pw = idx[pos[idx]]
            m_init = float(np.median(np.log(raw[pw]))) if len(pw) >= 5 else float(np.log(max(np.mean(np.abs(rw)), 1e-6)))
            theta = np.array([m_init, 0.0, np.log(0.5)])

        def nll(th):
            m0, b, ls = th
            s_ = np.exp(ls)
            mu = m0 + b * xw
            P = np.exp(mu[:, None] + sqrt2 * s_ * t[None, :])            # [n_w, nodes]
            z = (rw[:, None] - P) / sw[:, None]
            lp = -0.5 * z * z - np.log(sw)[:, None] - 0.5 * np.log(2 * np.pi) + logw[None, :]
            return -float((Kw * logsumexp(lp, axis=1)).sum())

        # two starts per window: the previous window's optimum and the best
        # of a coarse grid over the mean (the surface is rough where the
        # sampling variances are small, and a single warm start carried up
        # the expression range stalled at the top decile on BrainVar)
        bounds = [(-30, 30), (-20, 20), (np.log(floor_abs), np.log(5.0))]
        mg = np.linspace(theta[0] - 8, theta[0] + 8, 33)
        g_best = mg[int(np.argmin([nll([mm, 0.0, theta[2]]) for mm in mg]))]
        best = None
        for start in (theta, np.array([g_best, 0.0, theta[2]])):
            res = minimize(nll, start, method='L-BFGS-B', bounds=bounds)
            if best is None or res.fun < best.fun:
                best = res
        theta = best.x
        m[i] = theta[0]
        sd[i] = float(np.exp(theta[2]))
    return grid, m, sd


def estimate_variance_priors(A_df, Va_df, genes=None, n_bins=10, expression=None,
                             min_informative=40, library_factor=None, floor_frac=0.1,
                             eps=1e-12, max_iter=400, tol=1e-7, prior_method='deciles', span=0.15, pass2_variance='model'):
    """
    Empirical-Bayes priors for the allelic (c, tau) of the two-component
    models, one prior per expression bin, to be passed as ``variance_prior``
    to the mapping functions in place of the zero clamp.

    Every gene with at least ``min_informative`` informative samples is fitted
    through the origin (_fit_c_tau_vectorized, clamped, as the scan without a
    prior would). At each gene's converged weights the unclamped solution
    (c_raw, tau_raw) and its sampling covariance are taken from the
    squared-residual regression, with kappa = Var(z^2) of the standardized
    residuals estimated as the median over genes of E[z^4] - 1 (2 under
    Gaussian errors; larger with the heavy tails these residuals have).
    Genes are binned by expression, ``expression`` if supplied (a Series by
    gene, e.g. median allele-resolved reads) and otherwise the median over
    informative samples of log Va, which falls as 1/reads. In each bin the
    natural-scale mean of each parameter is the mean of the raw estimates
    (unbiased even when some are negative; winsorized at the 1st and 99th
    percentiles) and its between-gene variance is the variance of the raw
    estimates minus the median sampling variance, floored at ``floor_frac``
    of the variance so the prior never collapses to a point:
    DerSimonian-Laird's step applied across genes. Because c and tau are
    positive and right-skewed, the prior is the log-normal with those two
    moments (a normal prior on the natural scale is nearly uninformative
    about the sign of tau where its mean is small against its spread, and
    reintroduces the clamp), so the per-gene fit is penalized on the log
    scale and needs no clamp. A bin whose raw mean is not positive (on
    BrainVar the two lowest expression bins, for c) has it replaced by a
    twentieth of the raw spread so the log-normal exists; the per-bin table
    reports the raw mean (mean_raw_c, mean_raw_tau) and whether the
    replacement happened (c_mean_floored, tau_mean_floored). A bin left with
    fewer than 10 genes by tied proxies takes the pooled prior (pooled).

    ``prior_method='trend'`` replaces the ten bins by two smooth curves, one
    per parameter, fitted on the log scale by local marginal likelihood
    (_trend_prior): each gene's raw unbiased estimate is normal around the
    true value with its sampling variance, the log of the true value is
    normal around a locally linear curve in log10 expression with a locally
    constant spread, and the curve and spread are the kernel-weighted
    maximum-likelihood fit with the integral over the log value done by
    Gauss-Hermite quadrature. Every gene enters, negative raw estimates
    included, so the curve is not biased by dropping the genes whose true
    value is near zero; the spread is floored at 0.2 on the log scale.
    Every gene's prior is the curve at its expression, flat beyond the
    fitted range. The fit is made twice: the raw estimates computed with each
    gene's own clamped weights are biased low (those weights are correlated
    with the gene's noise), so the curves from that pass supply weights for a
    second set of raw estimates, independent of each gene's residuals, from
    which the final curves are fitted (measured on a simulated smooth truth:
    curve error median 0.05 on the log scale, against 0.19 in one pass; the
    between-gene spread is over-estimated by about a quarter because the
    model-based sampling variances are slightly understated under external
    weights, which errs toward shrinking less). The decile prior's moment
    match on the natural scale is what put the prior median of c at 0.0018
    in the two lowest bins and made the posterior bimodal; the trend prior
    works on the log scale from the start. The bins table reports the trend
    at each decile's median expression beside the decile prior, and the
    fraction of genes whose (second-pass) raw estimate is not positive.
    The per-gene c_raw and tau_raw columns stay the first-pass values, with
    their model-based sampling variances in c_raw_var and tau_raw_var; the
    trend adds the second-pass estimates and variances as c_raw_pass2,
    tau_raw_pass2, c_raw_pass2_var and tau_raw_pass2_var, so the prior can
    be checked against the raw estimates it was fitted to (for instance the
    fraction of non-positive raw estimates it predicts in a decile against
    the fraction observed).

    The prior is for the model the scan will use: pass ``library_factor``
    when the scan is 'library_scaled' (the raw fits are then on a/sqrt(d));
    map_cis checks the pairing.

    Returns a DataFrame indexed by every gene of A_df with columns bin,
    prior_c, prior_tau, prior_sd_c, prior_sd_tau (natural-scale mean and
    between-gene sd), prior_logc_m, prior_logc_s, prior_logtau_m,
    prior_logtau_s (the log-normal prior the fit uses), c_raw, tau_raw,
    c_raw_var, tau_raw_var (NaN for genes outside the estimation set),
    expression_proxy; attrs
    carry 'bins' (the per-bin table; under 'trend' the decile prior's columns
    stay as a reference and the trend's values at each decile's median
    expression are added as trend_*), 'kappa', 'n_genes', 'library_scaled',
    'method', and under 'trend' also 'span' and 'curve' (a DataFrame of the
    fitted curves on a fine grid of log10 expression).
    """
    if prior_method not in PRIOR_METHODS:
        raise ValueError(f'prior_method must be one of {PRIOR_METHODS}, got {prior_method!r}')
    A = np.asarray(A_df.values, dtype=float)
    V = np.asarray(Va_df.values, dtype=float)
    if library_factor is not None:
        dvec = _library_factor_tensor(library_factor, A_df.columns, 'cpu').numpy().astype(float)
        A = A / np.sqrt(dvec)[None, :]
    M = V > eps
    n_inf = M.sum(1)
    if expression is not None:
        proxy = pd.Series(expression).reindex(A_df.index).values.astype(float)
    else:
        with np.errstate(divide='ignore', invalid='ignore'):
            lv = np.where(M, np.log(np.where(M, V, 1.0)), np.nan)
            proxy = -np.nanmedian(lv, axis=1)   # larger = more expressed
    est = (n_inf >= min_informative) & np.isfinite(proxy)
    if genes is not None:
        est &= A_df.index.isin(pd.Index(genes))
    if est.sum() < 10 * n_bins:
        raise ValueError(f'only {int(est.sum())} genes are eligible for {n_bins} bins; lower n_bins or min_informative')
    Ae, Ve, Me = A[est], V[est], M[est]
    c, tau = _fit_c_tau_vectorized(Ae ** 2, Ve, Me, max_iter=max_iter, tol=tol)
    z2 = np.where(Me, Ae ** 2 / np.maximum(c[:, None] * Ve + tau[:, None], 1e-300), np.nan)
    kappa = max(float(np.nanmedian(np.nanmean(z2 * z2, axis=1) - 1.0)), 2.0)
    c_raw, tau_raw, var_c, var_tau = _raw_c_tau_and_cov(Ae ** 2, Ve, Me, c, tau, kappa)
    edges = np.quantile(proxy[est], np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    b_est = np.clip(np.searchsorted(edges, proxy[est], side='right') - 1, 0, n_bins - 1)

    def _prior(x, var_x):
        """Natural-scale mean and between-gene variance of a positive parameter
        from its unbiased (possibly negative) raw estimates, then the
        log-normal prior with those moments."""
        lo, hi = np.percentile(x, [1, 99])
        xw = np.clip(x, lo, hi)
        mu = float(np.mean(xw))
        spread = float(np.var(xw))
        noise = float(np.median(var_x))
        var_true = max(spread - noise, floor_frac * spread, 1e-24)
        # The log-normal needs a positive mean. Where the bin's raw mean is at
        # or below zero (the two lowest expression bins on BrainVar, whose raw
        # mean of c is negative) it is replaced by a twentieth of the raw
        # spread; the raw mean and the fact of the replacement are reported.
        mu_floor = 0.05 * np.sqrt(spread)
        mu_pos = max(mu, mu_floor, 1e-12)
        s2 = float(np.log1p(var_true / mu_pos ** 2))
        return (mu_pos, float(np.sqrt(var_true)), float(np.log(mu_pos) - 0.5 * s2), float(np.sqrt(s2)),
                mu, bool(mu_pos > mu))
    rows = []
    for k in range(n_bins):
        sel = b_est == k
        pooled = int(sel.sum()) < 10   # tied proxies can leave a bin empty: fall back to all genes
        use = np.ones_like(sel) if pooled else sel
        mu_c, sd_c, lm_c, ls_c, raw_c, fl_c = _prior(c_raw[use], var_c[use])
        mu_t, sd_t, lm_t, ls_t, raw_t, fl_t = _prior(tau_raw[use], var_tau[use])
        rows.append(dict(bin=k, genes=int(sel.sum()), proxy_lo=edges[k], proxy_hi=edges[k + 1],
                         prior_c=mu_c, prior_sd_c=sd_c, prior_tau=mu_t, prior_sd_tau=sd_t,
                         prior_logc_m=lm_c, prior_logc_s=ls_c, prior_logtau_m=lm_t, prior_logtau_s=ls_t,
                         mean_raw_c=raw_c, c_mean_floored=fl_c, mean_raw_tau=raw_t, tau_mean_floored=fl_t, pooled=pooled,
                         median_c_clamped=float(np.median(c[use])), tau_zero_clamped=float(np.mean(tau[use] < 1e-6))))
    bins = pd.DataFrame(rows)
    proxy_all = np.where(np.isfinite(proxy), proxy, -np.inf)   # genes with no informative sample: lowest bin
    b_all = np.clip(np.searchsorted(edges, proxy_all, side='right') - 1, 0, n_bins - 1)
    out = pd.DataFrame({'bin': b_all, 'expression_proxy': proxy}, index=A_df.index)
    for col in ('prior_c', 'prior_tau', 'prior_sd_c', 'prior_sd_tau',
                'prior_logc_m', 'prior_logc_s', 'prior_logtau_m', 'prior_logtau_s'):
        out[col] = bins[col].values[b_all]
    out['c_raw'] = np.nan
    out['tau_raw'] = np.nan
    out.loc[A_df.index[est], 'c_raw'] = c_raw
    out.loc[A_df.index[est], 'tau_raw'] = tau_raw
    out['c_raw_var'] = np.nan
    out['tau_raw_var'] = np.nan
    out.loc[A_df.index[est], 'c_raw_var'] = var_c
    out.loc[A_df.index[est], 'tau_raw_var'] = var_tau
    out.attrs['method'] = prior_method
    if prior_method == 'trend':
        # log10 expression for the curve; the proxy from the draws is already a log scale
        x_est = np.log10(np.maximum(proxy[est], 1e-6)) if expression is not None else proxy[est]
        x_all = np.log10(np.maximum(np.where(np.isfinite(proxy), proxy, 1e-6), 1e-6)) if expression is not None else np.where(np.isfinite(proxy), proxy, np.nanmin(proxy))
        if est.sum() < 20:
            raise ValueError(f'only {int(est.sum())} genes are eligible; the trend prior needs at least 20 to run and hundreds to mean anything')
        # Two passes. The raw estimates above use each gene's own clamped fit
        # for the weights, and those weights are correlated with the gene's
        # noise (small residuals give a small fitted variance, large weights
        # and a low slope): on a simulated smooth truth the raw estimates ran
        # 0.2 to 0.3 standard errors low and the curve inherited it. The
        # second pass recomputes every raw estimate with weights taken from
        # the first-pass curves at the gene's expression, which do not depend
        # on the gene's own residuals (bias 0.08 / -0.05 s.e. in the same
        # simulation; the true weights give 0.03 / -0.02). Their sampling
        # variance is model-based by default (pass2_variance='model'): it
        # depends on the curve weights and the design, not on the gene's own
        # residuals. The sandwich alternative is small for a gene whose
        # residuals happen to hug its line and brings the noise correlation
        # back (curve bias 0.15 to 0.19 in the simulation); the larger of the
        # two (pass2_variance='max') drove the fitted spread to its floor.
        # The stall this once masked (the top decile's tau prior at 0.5 to
        # 1.9 with a single warm start) is handled in _trend_prior by the
        # second start from a grid over the mean.
        curves = {}
        for name, raw, var in (('c', c_raw, var_c), ('tau', tau_raw, var_tau)):
            grid, m, sd = _trend_prior(x_est, raw, var, span=span)
            curves[name] = (grid, m, sd)
        c0 = np.exp(np.interp(x_est, *curves['c'][:2])); t0 = np.exp(np.interp(x_est, *curves['tau'][:2]))
        c_raw2, tau_raw2, var_c2, var_tau2 = _raw_c_tau_and_cov(Ae ** 2, Ve, Me, c0, t0, kappa)
        if pass2_variance == 'max':
            _, _, var_c2s, var_tau2s = _raw_c_tau_and_cov(Ae ** 2, Ve, Me, c0, t0, kappa, robust=True)
            var_c2, var_tau2 = np.maximum(var_c2, var_c2s), np.maximum(var_tau2, var_tau2s)
        for col, val in (('c_raw_pass2', c_raw2), ('tau_raw_pass2', tau_raw2), ('c_raw_pass2_var', var_c2), ('tau_raw_pass2_var', var_tau2)):
            out[col] = np.nan
            out.loc[A_df.index[est], col] = val
        curves = {}
        for name, raw, var in (('c', c_raw2, var_c2), ('tau', tau_raw2, var_tau2)):
            pos = raw > 0
            grid, m, sd = _trend_prior(x_est, raw, var, span=span)
            curves[name] = (grid, m, sd, pos)
        gc, mc, sc, posc = curves['c']; gt, mt, st, post = curves['tau']
        out['prior_logc_m'] = np.interp(x_all, gc, mc); out['prior_logc_s'] = np.interp(x_all, gc, sc)
        out['prior_logtau_m'] = np.interp(x_all, gt, mt); out['prior_logtau_s'] = np.interp(x_all, gt, st)
        for name in ('c', 'tau'):
            m_, s_ = out[f'prior_log{name}_m'].values, out[f'prior_log{name}_s'].values
            out[f'prior_{name}'] = np.exp(m_ + 0.5 * s_ ** 2)
            out[f'prior_sd_{name}'] = np.sqrt((np.exp(s_ ** 2) - 1.0) * np.exp(2 * m_ + s_ ** 2))
        # the decile table keeps the decile prior as a reference and gains the trend at each decile's median
        xmed = np.array([float(np.median(x_est[b_est == k])) if (b_est == k).any() else np.nan for k in range(n_bins)])
        bins['x_median'] = xmed
        bins['trend_logc_m'] = np.interp(xmed, gc, mc); bins['trend_logc_s'] = np.interp(xmed, gc, sc)
        bins['trend_logtau_m'] = np.interp(xmed, gt, mt); bins['trend_logtau_s'] = np.interp(xmed, gt, st)
        bins['trend_c'] = np.exp(bins['trend_logc_m'] + 0.5 * bins['trend_logc_s'] ** 2)
        bins['trend_tau'] = np.exp(bins['trend_logtau_m'] + 0.5 * bins['trend_logtau_s'] ** 2)
        bins['trend_c_median'] = np.exp(bins['trend_logc_m']); bins['trend_tau_median'] = np.exp(bins['trend_logtau_m'])
        bins['c_raw_nonpositive'] = [float(np.mean(~posc[b_est == k])) if (b_est == k).any() else np.nan for k in range(n_bins)]
        bins['tau_raw_nonpositive'] = [float(np.mean(~post[b_est == k])) if (b_est == k).any() else np.nan for k in range(n_bins)]
        out.attrs['span'] = float(span)
        out.attrs['curve'] = pd.DataFrame({'x_log10': gc, 'logc_m': mc, 'logc_s': sc,
                                           'logtau_m': np.interp(gc, gt, mt), 'logtau_s': np.interp(gc, gt, st)})
    out.attrs['bins'] = bins
    out.attrs['kappa'] = kappa
    out.attrs['n_genes'] = int(est.sum())
    out.attrs['library_scaled'] = library_factor is not None
    return out


def _prior_tuple(variance_prior, gene):
    """The (m_logc, s_logc, m_logtau, s_logtau, kappa) prior of one gene from
    the frame estimate_variance_priors returns, or None when no prior is in use."""
    if variance_prior is None:
        return None
    if gene not in variance_prior.index:
        raise KeyError(f'no variance prior for {gene}: estimate_variance_priors must be run on the same A_df')
    if 'kappa' not in variance_prior.attrs:
        raise ValueError('variance_prior has lost its attrs (kappa, library_scaled): pass the frame '
                         'estimate_variance_priors returned, not one reloaded from a file')
    r = variance_prior.loc[gene]
    return (float(r['prior_logc_m']), float(r['prior_logc_s']), float(r['prior_logtau_m']),
            float(r['prior_logtau_s']), float(variance_prior.attrs['kappa']))


def estimate_library_factors(A_df, Va_df, genes=None, min_informative=40,
                             n_iter=2, eps=1e-12, max_iter=400, tol=1e-7,
                             min_genes=20):
    """
    The per-library variance factor d_i of variance_model='library_scaled',
    estimated across genes from the allelic channel.

    For every gene with at least ``min_informative`` informative samples
    (Va > eps), fit (c_g, tau_g) through the origin with _fit_c_tau_vectorized;
    keep the genes whose fit is interior (c_g > 0 and tau_g > 0), because a
    clamped gene puts its whole floor into the other parameter and its
    residuals are not on a common scale; then d_i is the mean over those genes
    of a_gi^2 / (c_g v_gi + tau_g), whose expectation is d_i, normalized to
    mean 1 over samples (d and c share a scale). ``n_iter`` = 2 refits
    (c_g, tau_g) with the first-pass d in place and recomputes d once.

    ``genes`` restricts the estimation set. Pass the well-expressed genes
    (at least 100 allele-resolved reads on BrainVar): below that c is not
    identifiable. On BrainVar the choice moves any one library by at most
    11% (correlation 0.986 between the >= 100-read set and all interior genes,
    0.996 against >= 30 reads). Genes to be scanned may stay in the set: each
    is one of thousands.

    If fewer than ``min_genes`` genes have an interior fit (a data set with
    no between-sample floor, or a tiny one) every gene whose fit is not
    degenerate (c > 0 or tau > 0) is used instead, with a warning: the mean
    standardized squared residual still estimates d_i, on a scale set by the
    clamped parameter.

    Returns a Series indexed by sample, with attrs['n_genes'] the number of
    genes that entered the mean and attrs['interior_only'] whether only
    interior fits did.
    """
    import warnings
    A = np.asarray(A_df.values, dtype=float)
    V = np.asarray(Va_df.values, dtype=float)
    if genes is not None:
        rows = A_df.index.get_indexer(pd.Index(genes))
        if (rows < 0).any():
            raise ValueError(f'{int((rows < 0).sum())} of the requested genes are not in A_df')
        A, V = A[rows], V[rows]
    M = V > eps
    ok = M.sum(1) >= min_informative
    A, V, M = A[ok], V[ok], M[ok]
    if A.shape[0] == 0:
        raise ValueError('no gene has enough informative samples to estimate library factors')
    N = A.shape[1]
    d = np.ones(N)
    n_genes = 0
    for _ in range(max(1, int(n_iter))):
        c, tau = _fit_c_tau_vectorized(A ** 2 / d, V, M, max_iter=max_iter, tol=tol)
        interior = (c > 1e-6) & (tau > 1e-6)
        interior_only = True
        if interior.sum() < min_genes:
            interior_only = False
            interior = (c > 1e-6) | (tau > 1e-6)
            warnings.warn(
                f'estimate_library_factors: only {int(((c > 1e-6) & (tau > 1e-6)).sum())} genes have '
                f'an interior (c > 0, tau > 0) fit; using the {int(interior.sum())} non-degenerate '
                f'fits instead', RuntimeWarning, stacklevel=2)
        if not interior.any():
            raise ValueError('no gene has a non-degenerate fit; cannot estimate library factors')
        Mi = M[interior]
        e2 = np.where(Mi, A[interior] ** 2 / (c[interior, None] * V[interior] + tau[interior, None]), 0.0)
        n = Mi.sum(0)
        d_new = np.where(n > 0, e2.sum(0) / np.maximum(n, 1), np.nan)
        if np.isnan(d_new).any():
            raise ValueError('some samples are informative for none of the estimation genes')
        d = d_new / d_new.mean()
        n_genes = int(interior.sum())
    out = pd.Series(d, index=A_df.columns, name='library_factor')
    out.attrs['n_genes'] = n_genes
    out.attrs['interior_only'] = interior_only
    return out


def _channel_weights(y_t, v_t, covariates_t, tau_mode, device, eps=1e-12,
                     tau_extra_t=None, intercept=True, variance_model='additive',
                     d_t=None, prior=None):
    """sqrt weights of one channel, or all zeros when the channel is off, with
    the tau used (None under tau_mode='zero'), whether tau was estimated
    with the extra column(s) in its design, the c used (1 under the
    additive model, None under tau_mode='zero') and whether the variance fit
    converged (always True under the additive model).

    variance_model selects the variance function (VARIANCE_MODELS); d_t is
    the per-sample library factor under 'library_scaled'; prior is the
    gene's (m_logc, s_logc, m_logtau, s_logtau, kappa) for the shrinkage fit, or None
    for the clamped fit. The last return value is the fit dict of
    _estimate_c_tau (None under the additive model and tau_mode='zero').

    The sparse-channel rule: with fewer informative samples (v > eps) than
    the channel's design has columns plus two, neither the regression nor
    tau is identifiable from that channel, so it contributes nothing (every
    weight zero -> xx = 0 -> the meta-analysis takes the other channel alone).

    tau_extra_t ([N, k] or None) adds columns to the design tau is estimated
    under, without changing what the residualizer projects out: the lead
    refit passes the lead's predictor so tau stops absorbing the tested
    effect (see map_cis). When the informative samples cannot support the
    larger design the null-design tau is kept and the flag says so; the refit
    never switches a channel off that the scan had on.
    """
    n_inf = int((v_t > eps).sum())
    if n_inf < _min_informative(covariates_t, intercept=intercept):
        return torch.zeros_like(v_t), None, False, None, True, None
    if tau_mode != 'estimate':
        return torch.sqrt(1.0 / v_t.clamp(min=1e-8)), None, False, None, True, None
    design, refit = covariates_t, False
    if tau_extra_t is not None:
        cand = tau_extra_t if covariates_t is None else torch.cat([covariates_t, tau_extra_t], dim=1)
        if n_inf >= _min_informative(cand, intercept=intercept):
            design, refit = cand, True
    if variance_model == 'additive':
        tau = _estimate_tau_informative(y_t, v_t, design, device, eps,
                                        intercept=intercept)
        return torch.sqrt(1.0 / (v_t.clamp(min=1e-8) + tau)), float(tau), refit, 1.0, True, None
    keep = v_t > eps
    d_keep = None if d_t is None else d_t[keep]
    fit = _estimate_c_tau(y_t[keep], v_t[keep],
                          None if design is None else design[keep],
                          device, intercept=intercept, d_t=d_keep, prior=prior)
    c, tau = fit['c'], fit['tau']
    d_all = torch.ones_like(v_t) if d_t is None else d_t
    var = d_all * (c * v_t.clamp(min=1e-8) + tau)
    return torch.sqrt(1.0 / var.clamp(min=1e-30)), float(tau), refit, float(c), fit['converged'], fit


SAME_COVARIATES = 'same'


def _resolve_ase_covariates(ase_covariates_df, covariates_df, samples, device, logger):
    """The allelic channel's covariate design as _prepare_channels wants it:
    SAME_COVARIATES (the total channel's), None (through-origin) or a tensor
    built from its own DataFrame. Also returns its column count for the dof
    rule."""
    if isinstance(ase_covariates_df, str) and ase_covariates_df == SAME_COVARIATES:
        n = 0 if covariates_df is None else covariates_df.shape[1]
        logger.write('  * allelic channel covariates: same as the total channel')
        return SAME_COVARIATES, n
    if ase_covariates_df is None:
        logger.write('  * allelic channel covariates: none (through origin)')
        return None, 0
    assert np.all(np.asarray(samples) == np.asarray(ase_covariates_df.index)), \
        'Allelic-channel covariate samples must match phenotype samples'
    logger.write(f'  * allelic channel covariates: {ase_covariates_df.shape[1]}')
    t = torch.tensor(ase_covariates_df.values, dtype=torch.float32).to(device)
    return t, ase_covariates_df.shape[1]


def _warn_deprecated_variance_path(tau_mode, variance_model):
    """tau_mode='estimate' is the gateway to the deprecated variance models.

    The shipped model is Var(eps) = sigma^2 * v, which needs tau_mode='zero'.
    Anything reaching the (c_g, tau_g) family is historical; see
    VARIANCE_MODELS above for why they are inferior rather than merely old.
    """
    if tau_mode == 'estimate':
        import warnings
        warnings.warn(
            "hapmixQTL: tau_mode='estimate'"
            + (f" with variance_model={variance_model!r}" if variance_model != 'additive' else "")
            + " is DEPRECATED and will be removed. The shipped error model is "
              "Var(eps_i) = sigma^2 * v_i (tau_mode='zero', se_mode='fitted'). The "
              "(c_g, tau_g) family fits its layer-1 variance from the residuals it "
              "then weights, which is why it is inferior and not merely older; "
              "efficiency comparisons do not rehabilitate it. Use it only to "
              "reproduce historical results. See CLAUDE.md, 'The variance models "
              "are deprecated'.",
            DeprecationWarning, stacklevel=3)


def _warn_tau_zero(tau_mode, fitted_scale=False):
    """tau_mode='zero' is anticonservative ONLY under a known-variance SE.

    The evidence behind this warning -- 107x nominal type-I at alpha=1e-3,
    ~100% false positives on count-level simulations, 64.5% coverage -- was
    all measured with the known-variance standard error, where the weights
    are absolute precisions and tau=0 asserts that the Gibbs variance IS the
    entire error variance. That assertion is what fails.

    With a fitted residual scale (se_mode='fitted') no such assertion is
    made: sigma^2 absorbs whatever the draws got wrong about the absolute
    scale, and the weights are only a shape. Measured 2026-09-20 on the
    combined statistic over 40 null permutations: tau_mode='zero' with the
    known-variance SE calibrates at 23.3, and with a fitted scale at 1.068.
    So the warning is scoped, and firing it for the fitted pairing would be
    a warning about a different configuration than the one being run.
    """
    if tau_mode == 'zero' and not fitted_scale:
        import warnings
        warnings.warn(
            "hapmixQTL: tau_mode='zero' with a known-variance standard error asserts "
            "that Gibbs inferential variance is the ENTIRE error variance, which real "
            "quantifier posteriors never satisfy. Measured consequences: up to 107x "
            "nominal type-I error at alpha=1e-3, ~100% false positives on count-level "
            "simulations, 64.5% coverage of nominal 95% CIs, and a combined-statistic "
            "calibration of 23.3. Either use tau_mode='estimate', or pair tau_mode='zero' "
            "with se_mode='fitted' (calibration 1.068), which is the Var(eps)=sigma^2*v "
            "model and the shipped default. See docs/ase_validation.md.",
            RuntimeWarning, stacklevel=3)


def _prepare_channels(a_t, t_t, va_t, vt_t, covariates_t, tau_mode, device,
                      ase_covariates_t=SAME_COVARIATES, eps=1e-12,
                      tau_extra_a_t=None, tau_extra_t_t=None, return_info=False,
                      variance_model='additive', library_factor_t=None, prior=None,
                      keep_a_t=None, keep_t_t=None, total_variance_model='additive',
                      fitted_scale=False):
    """
    Per-phenotype whitening shared by every mapping function.

    ``variance_model`` (VARIANCE_MODELS) selects the allelic channel's
    variance function: 'additive' v + tau, 'two_component' c v + tau, or
    'library_scaled' d_i (c v + tau) with ``library_factor_t`` the per-sample
    d_i. ``prior`` is the gene's (m_logc, s_logc, m_logtau, s_logtau, kappa) from
    estimate_variance_priors (shrinkage in place of the clamp) or None. The
    total channel is v_t + tau_t under every model (module docstring).

    Estimates tau per channel (unless tau_mode='zero'), builds the sqrt
    weights w_i = 1/(v_inf_i + tau) with zero-coverage ASE samples zeroed
    out. The total residualizer projects out its automatic intercept and
    covariates; the ASE residualizer is through-origin and projects only
    explicitly supplied ASE covariates. Any second-pass regression that
    reuses this is whitened exactly like the lead scan.

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
    SAME_COVARIATES (default) reuses the total channel's, None is
    through-origin. The allelic contrast a = log(yL/yR) is a within-sample
    difference in which anything acting on both haplotypes alike (library
    size, expression PCs, sex, age) cancels, so the total channel's
    covariates are normally not wanted there: each column costs one of the
    informative samples and can only remove signal (10 expression PCs
    against ~50 informative samples halved CCNI's allelic statistic).

    tau is estimated under the design the residualizer projects out (the
    null model). ``tau_extra_a_t`` / ``tau_extra_t_t`` add columns to that
    design for the tau estimate only -- the lead refit passes the lead's
    predictor of each channel -- so a strong cis effect stops inflating tau
    and shrinking the reported scale (_channel_weights).

    Returns:
        sqrt_wa_t, sqrt_wt_t, residualizer_a, residualizer_t
        (+ info dict with tau_a, tau_t, c_a, converged_a, refit_a, refit_t,
        variance_model when return_info)
    """
    _warn_tau_zero(tau_mode, fitted_scale=fitted_scale)
    _warn_deprecated_variance_path(tau_mode, variance_model)
    _check_variance_model(variance_model, tau_mode, library_factor_t)
    if isinstance(ase_covariates_t, str) and ase_covariates_t == SAME_COVARIATES:
        ase_covariates_t = covariates_t

    # Count cutoffs (count_cutoff_masks), when the caller supplies them. A
    # sample the cutoffs exclude carries nothing usable for that channel,
    # which is the situation a zero-coverage sample is already in, so it is
    # put in exactly that state: its WORKING inferential variance is zeroed.
    # Every informative-set test downstream is `v > eps` -- the
    # sparse-channel rule, _estimate_tau_informative, and _estimate_c_tau --
    # so one assignment keeps all three consistent with the weights, which
    # is the property that matters: tau must never be estimated on samples
    # the fit then excludes. Only the local tensors change; the caller's
    # Va_df and Vt_df are untouched and still report what the draws measured.
    if keep_a_t is not None:
        va_t = torch.where(keep_a_t, va_t, torch.zeros_like(va_t))
    if keep_t_t is not None:
        vt_t = torch.where(keep_t_t, vt_t, torch.zeros_like(vt_t))

    wa, tau_a, refit_a, c_a, conv_a, fit_a = _channel_weights(
        a_t, va_t, ase_covariates_t, tau_mode, device, eps, tau_extra_a_t,
        intercept=False, variance_model=variance_model, d_t=library_factor_t, prior=prior)
    sqrt_wa_t = _zero_degenerate_ase_weights(wa, va_t, eps)
    # The total channel's variance function. It was v_t + tau_t under every
    # allelic variance_model until 2026-09-20, and still is by default, so
    # total_variance_model='additive' reproduces every prior result exactly.
    #
    # Why the option exists. The draws DO carry per-donor information about
    # total expression -- Vt spans 2 to 3.5-fold across donors within every
    # one of the 29 calibration genes -- but with c fixed at 1, tau_t swamps
    # it about 19 to 1, so the weights come out nearly equal (10-90 spread
    # 1.04, Kish 92 of 92 donors). Fitting c_t per gene is the lever that
    # changes that: measured median c_t 8.7, which lifts the draws' share of
    # the weight denominator from 0.04 to 0.38 and the weight spread to 1.51.
    # 'library_scaled' is deliberately NOT offered here: d_i is estimated
    # from the ALLELIC channel's residuals by estimate_library_factors and
    # has no meaning for this one.
    if total_variance_model not in ('additive', 'two_component'):
        raise ValueError("total_variance_model must be 'additive' or "
                         f"'two_component', got {total_variance_model!r}")
    if total_variance_model != 'additive' and tau_mode != 'estimate':
        raise ValueError("total_variance_model='two_component' requires "
                         "tau_mode='estimate'")
    sqrt_wt_t, tau_t, refit_t, c_t, conv_t, _ = _channel_weights(
        t_t, vt_t, covariates_t, tau_mode, device, eps, tau_extra_t_t,
        intercept=True, variance_model=total_variance_model)
    # The allelic channel's zeroing is already done by
    # _zero_degenerate_ase_weights above, since va_t is now 0 there. The
    # total channel has no such guard (CLAUDE.md, "The total channel has no
    # zero-count guard"), so an excluded sample would otherwise keep the
    # finite weight 1/(1e-8 + tau_t). Zero it explicitly. This is scoped to
    # samples the caller's cutoffs excluded and introduces no zero-count rule
    # of its own.
    if keep_t_t is not None:
        sqrt_wt_t = torch.where(keep_t_t, sqrt_wt_t, torch.zeros_like(sqrt_wt_t))
    residualizer_a = WeightedResidualizer(ase_covariates_t, sqrt_wa_t, intercept=False)
    residualizer_t = WeightedResidualizer(covariates_t, sqrt_wt_t, intercept=True)
    if return_info:
        return sqrt_wa_t, sqrt_wt_t, residualizer_a, residualizer_t, dict(
            tau_a=tau_a, tau_t=tau_t, c_a=c_a, converged_a=conv_a, refit_a=refit_a,
            refit_t=refit_t, variance_model=variance_model,
            c_t=c_t, converged_t=conv_t, total_variance_model=total_variance_model,
            c_a_raw=(fit_a['c_raw'] if fit_a else c_a), tau_a_raw=(fit_a['tau_raw'] if fit_a else tau_a),
            floored_a=(fit_a['floored'] if fit_a else False), prior_used=prior is not None)
    return sqrt_wa_t, sqrt_wt_t, residualizer_a, residualizer_t


def calculate_hapmixqtl_nominal(genotypes_t, sign_t, a_t, t_t,
                                 sqrt_wa_t, sqrt_wt_t,
                                 residualizer_a, residualizer_t,
                                 robust=False, fitted=False):
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
        fitted: if True use estimated-dispersion SEs (takes precedence)

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
    slope_a, se_a = _wls_regression(a_star, s_star, residualizer_a, robust=robust,
                                    fitted=fitted)

    # Total channel: t = beta * (g/2) + covariates + error
    t_star = (t_t * sqrt_wt_t).unsqueeze(0)
    g_half_star = (genotypes_t / 2) * sqrt_wt_t.unsqueeze(0)
    slope_tc, se_tc = _wls_regression(t_star, g_half_star, residualizer_t, robust=robust,
                                      fitted=fitted)

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


def _leverage_standardized(res_t, residualizer):
    """Whitened null residuals divided by sqrt(1 - h_ii), h_ii the leverage of
    the null design (the diagonal of Q Q').

    (I - Q Q') e has per-sample variance 1 - h_ii, not 1, so a permutation of
    the raw residuals is under-dispersed by about (N - p)/N. That would cancel
    for a pivotal statistic refitted per permutation, but the known-variance
    statistic here is built from xy and xx alone, so the residual scale enters
    the null directly: with 18 design columns on 92 samples the permuted null
    would be short by 20% in chi2 and the empirical p anticonservative.
    Standardizing restores unit variance. A switched-off channel (Q of width
    0) and a zero-weight sample (a zero design row, residual 0) are unchanged.
    """
    h = (residualizer.Q_t * residualizer.Q_t).sum(1)
    return res_t / torch.sqrt((1.0 - h).clamp(min=1e-3))


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


PERM_SCHEMES = ('records', 'residuals')


def _record_permutation_channel(x_t, y_t, sqrt_w_t, residualizer, permutation_ix_t, chunk=None):
    """One channel's permutation statistics under the donor-record permutation.

    For each row s of ``permutation_ix_t`` [nperm, N], donor i receives donor
    s[i]'s record: its whitened phenotype value, its weight and its covariate
    row move together; the predictors ``x_t`` [V, N] (the allelic sign
    contrast, or dosage/2) stay in place. The whitened design is rebuilt from
    the permuted weights and covariate rows, the phenotype and the weighted
    predictors are residualized on it, and the known-variance summaries
    follow. Relabeling the donors shows this equals the statistic under the
    permutation of the genotype columns by the inverse of s, so the
    empirical p is the one FastQTL and tensorQTL compute, with per-donor
    weights carried along.

    Returns xy [V, nperm], xx [V, nperm] (the denominator changes with the
    permutation) and yy [nperm]. A through-origin channel with no nuisance
    columns (the default allelic design) needs no re-residualization and
    costs two matrix products; the general case is chunked over permutations
    with a batched QR. A switched-off channel (every weight zero) returns
    zeros.
    """
    nperm, N = permutation_ix_t.shape
    V = x_t.shape[0]
    if not bool((sqrt_w_t != 0).any()):
        z = x_t.new_zeros((V, nperm))
        return z, z.clone(), x_t.new_zeros(nperm)
    w = sqrt_w_t * sqrt_w_t
    y_star = y_t * sqrt_w_t                       # moves as a whole with the record
    C_t = getattr(residualizer, 'C_t', None)
    intercept = bool(getattr(residualizer, 'intercept', False))
    p = (1 if intercept else 0) + (int(C_t.shape[1]) if C_t is not None else 0)
    if p == 0:
        wy = w * y_t
        xy = torch.mm(x_t, wy[permutation_ix_t].t())
        xx = torch.mm(x_t * x_t, w[permutation_ix_t].t())
        yy = (y_star * y_star)[permutation_ix_t].sum(1)
        return xy, xx, yy
    if chunk is None:
        chunk = int(max(8, min(256, 2e7 // max(V * N, 1))))
    xy = x_t.new_empty((V, nperm))
    xx = x_t.new_empty((V, nperm))
    yy = x_t.new_empty(nperm)
    for s0 in range(0, nperm, chunk):
        ix = permutation_ix_t[s0:s0 + chunk]
        K = ix.shape[0]
        sw = sqrt_w_t[ix]                                              # [K, N]
        cols = []
        if intercept:
            cols.append(sw.unsqueeze(2))
        if C_t is not None:
            cols.append(sw.unsqueeze(2) * C_t[ix])                     # [K, N, c]
        design = torch.cat(cols, 2)                                    # [K, N, p]
        Q, _ = torch.linalg.qr(design)                                 # [K, N, p]
        ys = y_star[ix]                                                # [K, N]
        e = ys - torch.bmm(Q, torch.bmm(Q.transpose(1, 2), ys.unsqueeze(2))).squeeze(2)
        Xs = x_t.unsqueeze(0) * sw.unsqueeze(1)                        # [K, V, N]
        Xs = Xs - torch.bmm(torch.bmm(Xs, Q), Q.transpose(1, 2))       # residualized, no cancellation in xx
        xy[:, s0:s0 + K] = torch.bmm(Xs, e.unsqueeze(2)).squeeze(2).t()
        xx[:, s0:s0 + K] = (Xs * Xs).sum(2).t()
        yy[s0:s0 + K] = (e * e).sum(1)
    return xy, xx, yy


def calculate_hapmixqtl_permutations(genotypes_t, sign_t, a_t, t_t,
                                      sqrt_wa_t, sqrt_wt_t,
                                      residualizer_a, residualizer_t,
                                      permutation_ix_t, dof=None, perm_scheme='records',
                                      fitted=False):
    """
    Compute nominal and permutation statistics for hapmixQTL.

    ``dof`` is the t-reference used to map the known-variance statistic to a
    correlation scale; the mapping is monotonic, so the empirical p-value does
    not depend on it, but the nominal p-value the caller derives from
    r_nominal does. Default: the smaller of the two channels' residualizer
    dof, which with per-channel covariates is the total channel's (the
    larger design). map_cis passes its own N - 2 - n_cov so both agree.

    The permutation null. ``perm_scheme='records'`` (the default) permutes
    donor records: for each permutation every donor receives another donor's
    whitened phenotype value, weight and covariate row together, the
    genotypes stay in place, and the statistic is recomputed with the
    permuted weights and design (the denominator xx changes per permutation:
    a second matrix product on the allelic channel, a chunked batched
    re-residualization on the total channel; _record_permutation_channel).
    Relabeling the donors shows this is exactly the distribution of the
    statistic under a permutation of the genotype columns, the null FastQTL
    and tensorQTL use, with per-donor weights carried along.

    ``perm_scheme='residuals'`` is the earlier scheme, Freedman-Lane in
    whitened space: the leverage-standardized whitened residuals are permuted
    among each channel's informative donors (the two channels share the
    draw; _permute_within_informative, _leverage_standardized) while the
    predictors, weights and covariates stay in place, so every permuted
    statistic is one matrix product. It is exact only if the standardized
    residuals are exchangeable across donors, and on BrainVar they are not.
    On the allelic channel the null it builds has the scale of the
    unweighted mean of the squared standardized residuals, whereas a
    genotype permutation, or a real null gene, has the scale of their
    weight-weighted mean, which the (c, tau) fit pins at 1; where the weights
    span two decades (well-expressed genes) the unweighted mean sits about 3%
    higher, and the maximum over about 4,500 correlated variants turns that
    into a third fewer rejections (allelic channel alone, 100 genes x 40
    genotype permutations: type-I 0.032 at nominal 0.05 on the high tier
    against 0.048 with records; 0.048 against 0.049 on the low tier, where
    tau makes the weights nearly equal). On the total channel, whose weights
    are nearly equal, the residual scheme is short in every tier (0.029 to
    0.035) because with 18 nuisance columns on 92 donors the residuals have
    covariance proportional to I - H and the leverage standardization
    corrects only its diagonal (without it, 0.23; with the whole record
    permuted, 0.05 to 0.07 on 30 genes). Neither scheme is the one this
    docstring used to warn against, permuting the raw phenotype values at
    fixed weights, which hands a donor another donor's value at its own
    precision and mis-scales the null the other way (type-I 0.000): the
    record scheme moves the weight with the value. Measured 2026-09-17,
    estimator_ablation_tiers_20260917/permutation_scheme_experiment.py and
    total_channel_schemes.py.

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
    # Per-channel degrees of freedom for the fitted residual scale. Counts
    # INFORMATIVE donors, not rows: a zero-weight donor contributes nothing
    # to the residual sum of squares, so charging it dof would shrink the
    # scale and inflate the statistic (same rule as _wls_regression).
    dof_a = max(int((residualizer_a.sqrt_w_t != 0).sum())
                - 1 - residualizer_a.Q_t.shape[1], 1)
    dof_t = max(int((residualizer_t.sqrt_w_t != 0).sum())
                - 1 - residualizer_t.Q_t.shape[1], 1)
    _fit = dict(fitted=fitted, dof_a=dof_a, dof_t=dof_t)

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
                                  xy_t_nom, xx_t, yy_t_nom, dof, **_fit)

    tstat2_nom_clean = tstat2_nom.clone()
    tstat2_nom_clean[torch.isnan(tstat2_nom_clean)] = -1
    best_ix = tstat2_nom_clean.argmax()

    # Combined slope for the best variant, taken from the SAME routine that
    # produced the statistic. This was previously a second, hand-inlined
    # known-variance copy, which silently disagreed with map_nominal the
    # moment a fitted residual scale was allowed; a test caught it.
    _, slope_all = _combined_tstat2(xy_a_nom, xx_a, yy_a_nom,
                                    xy_t_nom, xx_t, yy_t_nom, dof,
                                    return_slope=True, **_fit)
    slope_nom = slope_all[best_ix]

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

    # --- Permutation statistics (see the docstring) ---
    if perm_scheme == 'records':
        xy_a_perm, xx_a_perm, yy_a_perm = _record_permutation_channel(
            sign_t, a_t, sqrt_wa_t, residualizer_a, permutation_ix_t)
        xy_t_perm, xx_t_perm, yy_t_perm = _record_permutation_channel(
            genotypes_t / 2, t_t, sqrt_wt_t, residualizer_t, permutation_ix_t)
        tstat2_perm = _combined_tstat2(xy_a_perm, xx_a_perm, yy_a_perm,
                                        xy_t_perm, xx_t_perm, yy_t_perm, dof,
                                        **_fit)
    elif perm_scheme == 'residuals':
        a_res_perms = _permute_within_informative(
            _leverage_standardized(a_star_res[0], residualizer_a), sqrt_wa_t > 0, permutation_ix_t)
        t_res_perms = _permute_within_informative(
            _leverage_standardized(t_star_res[0], residualizer_t), sqrt_wt_t > 0, permutation_ix_t)

        xy_a_perm = torch.mm(s_star_res, a_res_perms.t())
        yy_a_perm = (a_res_perms * a_res_perms).sum(1)
        xy_t_perm = torch.mm(g_half_star_res, t_res_perms.t())
        yy_t_perm = (t_res_perms * t_res_perms).sum(1)

        tstat2_perm = _combined_tstat2(xy_a_perm, xx_a, yy_a_perm,
                                        xy_t_perm, xx_t, yy_t_perm, dof,
                                        **_fit)
    else:
        raise ValueError(f'perm_scheme must be one of {PERM_SCHEMES}, got {perm_scheme!r}')

    tstat2_perm[torch.isnan(tstat2_perm)] = 0
    r2_perm = tstat2_perm / (tstat2_perm + dof)
    max_r2_perm, _ = r2_perm.max(0)

    return r_nominal, std_ratio, best_ix, max_r2_perm, genotypes_t[best_ix]


def _combined_tstat2(xy_a, xx_a, yy_a, xy_t, xx_t, yy_t, dof,
                     fitted=False, dof_a=None, dof_t=None,
                     return_slope=False):
    """
    Compute combined (known-variance) statistic squared from dot-product
    summaries, matching the inverse-variance meta-analysis in
    ``calculate_hapmixqtl_nominal``.

    Under known-variance GLS the per-channel inverse variance of the slope is
    just ``xx`` (since se^2 = 1/xx), so the combined statistic simplifies to

        beta_c   = (xy_a + xy_t) / (xx_a + xx_t)          [both channels valid]
        stat^2   = beta_c^2 * (xx_a + xx_t)

    With ``fitted=True`` that estimated-dispersion variant is taken instead,
    which is what ``yy_a``/``yy_t`` were always carried for. Each channel gets
    its own residual scale, refit at every permutation exactly as mixQTL's
    ``mixqtl_permutation_scan`` does:

        rss     = yy - xy^2 / xx          (through-origin residual SS)
        s2      = rss / dof_channel
        inv_var = xx / s2                 (instead of xx)

    The two forms then share one algebra, since known-variance is s2 = 1:
    ``inv_var = xx/s2`` and the meta numerator is ``xy/s2``, which collapses
    to ``xy_a + xy_t`` when both scales are 1. ``dof_a``/``dof_t`` must count
    INFORMATIVE donors per channel, for the reason given in _wls_regression.

    Works for both scalar (nominal) and 2D (permutation) ``xy`` by
    broadcasting.
    """
    is_perm = xy_a.dim() == 2

    if is_perm:
        # xx is [V] when the predictors are fixed across permutations and
        # [V, nperm] under the donor-record permutation
        xx_a_e = xx_a.unsqueeze(1) if xx_a.dim() == 1 else xx_a
        xx_t_e = xx_t.unsqueeze(1) if xx_t.dim() == 1 else xx_t
    else:
        xx_a_e = xx_a
        xx_t_e = xx_t

    # Known-variance inverse variances of the per-channel slopes are xx itself;
    # a degenerate predictor (xx ~ 0) contributes zero weight.
    inv_var_a = torch.where(xx_a_e > 1e-30, xx_a_e, torch.zeros_like(xx_a_e))
    inv_var_t = torch.where(xx_t_e > 1e-30, xx_t_e, torch.zeros_like(xx_t_e))

    xy_a_eff = torch.where(xx_a_e > 1e-30, xy_a, torch.zeros_like(xy_a))
    xy_t_eff = torch.where(xx_t_e > 1e-30, xy_t, torch.zeros_like(xy_t))

    if fitted:
        # Per-channel fitted residual scale, refit here (and so, under
        # permutation, at every permutation) as mixQTL does.
        # rss = yy - xy^2/xx is catastrophic cancellation when the fit is
        # good (rss << yy), which is exactly the regime a strong cis effect
        # puts us in, so the subtraction is done in float64 and cast back.
        # The summary form is kept rather than forming residuals explicitly
        # because the permutation path would otherwise materialise a
        # [variants x permutations x samples] residual array.
        _d = torch.float64
        rss_a = (yy_a.to(_d) - xy_a_eff.to(_d) ** 2
                 / (inv_var_a.to(_d) + 1e-300)).clamp(min=0)
        rss_t = (yy_t.to(_d) - xy_t_eff.to(_d) ** 2
                 / (inv_var_t.to(_d) + 1e-300)).clamp(min=0)
        s2_a = (rss_a / max(int(dof_a) if dof_a else 1, 1)).to(xy_a.dtype)
        s2_t = (rss_t / max(int(dof_t) if dof_t else 1, 1)).to(xy_t.dtype)
        good_a = (inv_var_a > 1e-30) & (s2_a > 0)
        good_t = (inv_var_t > 1e-30) & (s2_t > 0)
        inv_var_a = torch.where(good_a, inv_var_a / s2_a.clamp(min=1e-300),
                                torch.zeros_like(inv_var_a))
        inv_var_t = torch.where(good_t, inv_var_t / s2_t.clamp(min=1e-300),
                                torch.zeros_like(inv_var_t))
        xy_a_eff = torch.where(good_a, xy_a_eff / s2_a.clamp(min=1e-300),
                               torch.zeros_like(xy_a_eff))
        xy_t_eff = torch.where(good_t, xy_t_eff / s2_t.clamp(min=1e-300),
                               torch.zeros_like(xy_t_eff))

    total_inv = inv_var_a + inv_var_t
    # beta_c * total_inv = slope_a*inv_var_a + slope_t*inv_var_t
    #                    = xy_a/s2_a + xy_t/s2_t, which is xy_a + xy_t in the
    #                    known-variance case where both scales are 1
    numer = xy_a_eff + xy_t_eff
    slope_comb = torch.where(
        total_inv > 0,
        numer / (total_inv + 1e-30),
        torch.zeros_like(numer),
    )
    tstat2 = slope_comb * slope_comb * total_inv
    if return_slope:
        return tstat2, slope_comb
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
                tau_mode='zero', se_mode='fitted',
                output_dir='.', logger=None, verbose=True,
                ase_covariates_df=None, variance_model='additive',
                library_factor=None, variance_prior=None,
                keep_a_df=None, keep_t_df=None,
                total_variance_model='additive'):
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
                          to the TOTAL channel; pass SAME_COVARIATES to apply
                          them to the allelic channel too
        ase_covariates_df: covariates for the ALLELIC channel: None (the
                          default, through-origin),
                          SAME_COVARIATES (the total channel's set) or a
                          DataFrame [samples x k]. The allelic contrast is a
                          within-sample difference in which covariates that act
                          on both haplotypes alike cancel. Measured on 30
                          BrainVar genes: the 17-covariate set explains 96% of
                          the total channel's whitened residual variance but
                          only 24% of the allelic channel's against 22% expected
                          by chance, while each column costs one informative
                          sample. Pass SAME_COVARIATES to reproduce earlier
                          results (see _prepare_channels)
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

            The parent method makes the same point differently: mixQTL (Liang
            et al. 2021) writes the ASE error as N(0, sigma^2 * (1/Y1 + 1/Y2)),
            where the counts set only the SHAPE of the weights and sigma^2 is a
            free scale parameter inferred from the data (Supplementary Notes
            5.2). 'zero' drops that free parameter; 'estimate' restores a free
            parameter, but ADDITIVELY (v_inf + tau) rather than multiplicatively
            (sigma^2 * v_inf), so it is not mixQTL's variance model. A
            multiplicative scale suits quantification noise of the right shape
            and the wrong size; an additive term suits biological variance,
            which does not shrink with read depth. The two have NOT been
            compared: docs/ase_validation.md sec 7b runs no-tau, a weight cap,
            additive tau and the nested sigma^2 * v_inf + tau, and has no
            multiplicative arm. This contrast applies to the ALLELE-SPECIFIC
            channel only; mixQTL's total-count channel carries a single flat
            variance that hapmixQTL's v_t + tau_t decomposes rather than
            departs from (docs/hapmixqtl_methods.md sec 3.2).

            'zero' is retained only for reproducing prior results and emits a
            warning.
        se_mode:          'model' (default; known-variance 1/sqrt(xx)),
                          'robust' (HC1 sandwich), or 'fitted'
                          (estimated dispersion sigma_hat/sqrt(xx), which
                          makes the weights a shape only -- Var(eps) =
                          sigma^2 * v under tau_mode='zero')
                          Statistics are on the null-model tau scale (tau
                          estimated once per gene without a genotype term);
                          map_cis(tau_refit=True) reports its lead with tau
                          re-estimated under the alternative, so a strong
                          gene's lead pair is larger there than here.
        total_variance_model: the TOTAL channel's variance function,
                          'additive' (default, v_t + tau_t -- what every
                          earlier result used) or 'two_component'
                          (c_t v_t + tau_t, c_t fitted per gene). Independent
                          of ``variance_model``, which governs the allelic
                          channel only.
        variance_model:   'additive' (default; v + tau), 'two_component'
                          (c v + tau) or 'library_scaled' (d_i (c v + tau)),
                          the allelic channel's error variance; see the module
                          docstring. The total channel is v_t + tau_t always.
        library_factor:   per-sample d_i for 'library_scaled' (Series indexed
                          by sample, or array in phenotype column order), from
                          estimate_library_factors; required by that model and
                          rejected by the others.
        variance_prior:   frame from estimate_variance_priors, or None: with it
                          the per-gene (c, tau) fit shrinks toward the gene's
                          expression bin instead of clamping at zero.
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
    fitted = se_mode == 'fitted'

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
        _assert_phase_columns(xL_df, xR_df, genotype_df)
    else:
        logger.write('  * no phase genotypes (total channel only)')

    _assert_keep_frames(keep_a_df, keep_t_df, A_df)
    _log_keep_frames(keep_a_df, keep_t_df, logger)

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
    library_factor_t = _library_factor_tensor(library_factor, samples, device)
    _check_variance_model(variance_model, tau_mode, library_factor_t, variance_prior)
    logger.write(f'  * variance model: {variance_model}' + (' with empirical-Bayes prior' if variance_prior is not None else ''))

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
                ase_covariates_t=ase_covariates_t, variance_model=variance_model,
                library_factor_t=library_factor_t, prior=_prior_tuple(variance_prior, phenotype_id),
                keep_a_t=_keep_row(keep_a_df, pidx, device),
                keep_t_t=_keep_row(keep_t_df, pidx, device),
                total_variance_model=total_variance_model,
                fitted_scale=fitted)

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
                robust=robust, fitted=fitted,
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
            nperm=10000, window=1000000, tau_mode='zero', se_mode='fitted',
            logger=None, seed=None, verbose=True, warn_monomorphic=True,
            ase_covariates_df=None, tau_refit=False, variance_model='additive',
            library_factor=None, variance_prior=None,
            perm_scheme='records', keep_a_df=None, keep_t_df=None):
    """
    hapmixQTL cis-QTL mapping with permutation-based empirical p-values.

    ``variance_model`` selects the allelic channel's error variance:
    'additive' (default) v + tau, 'two_component' c v + tau, or
    'library_scaled' d_i (c v + tau) with ``library_factor`` the per-sample
    d_i from estimate_library_factors (required by that model, rejected by
    the others). The total channel is v_t + tau_t under every model. The
    output carries ``c_a`` (the c of the reported statistic; 1 under
    'additive'), ``c_a_null`` (the scan's) and ``variance_model``. With
    ``tau_refit`` the two-component models refit (c, tau) together with the
    lead in the design, as the additive model refits tau. ``variance_prior``
    (the frame from estimate_variance_priors) replaces the zero clamp of the
    per-gene (c, tau) fit with empirical-Bayes shrinkage toward the gene's
    expression bin; the output then also carries ``c_a_raw``/``tau_a_raw``
    (the unshrunk solution at the final weights) and ``c_a_floored`` (a clamp
    branch was taken on the final iteration of the clamped fit; always False
    under a prior, where nothing is clamped).

    ``tau_refit``: tau is estimated once per gene under the null model (no
    genotype term) for the scan, so a strong cis effect inflates it and
    shrinks every statistic in the window by a common factor (30-110% on
    BrainVar genes with real signal). That factor cancels in ``pval_perm``
    and ``pval_beta``, which compare the scan statistic with permutations
    carrying the same tau, but not in the nominal scale. With
    ``tau_refit=True`` each channel's tau is re-estimated with the lead's
    predictor in the model and the lead's ``slope``, ``slope_se``,
    ``pval_nominal`` and per-channel diagnostics are reported on that scale
    (the like-for-like with a model that fits its dispersion under the
    alternative, as RASQUAL does); ``pval_perm`` and ``pval_beta`` stay on
    the scan scale, where they are calibrated. ``tau_a``/``tau_t`` are the
    tau of the reported statistic, ``tau_a_null``/``tau_t_null`` the scan's.
    A lead's ``pval_nominal`` is never a gene-level p (it is the best of the
    window), and the refit makes it more selective; ``pval_beta`` is the
    gene-level p. map_nominal stays on the null-model scale.

    ``ase_covariates_df`` is the allelic channel's covariate design (see
    map_nominal): None for through-origin (the default), SAME_COVARIATES for
    the total channel's set, or its own DataFrame. The nominal p-value uses one t reference for both channels,
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

    if se_mode not in ('model', 'fitted'):
        raise ValueError(
            f"map_cis: se_mode={se_mode!r} is not available. 'model' and "
            "'fitted' both have permutation counterparts -- 'fitted' refits a "
            "per-channel residual scale at every permutation, exactly as "
            "mixQTL's mixqtl_permutation_scan does. The HC1 sandwich does "
            "not; use map_nominal for se_mode='robust'.")

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
        _assert_phase_columns(xL_df, xR_df, genotype_df)
    else:
        logger.write('  * no phase genotypes (total channel only)')

    _assert_keep_frames(keep_a_df, keep_t_df, A_df)
    _log_keep_frames(keep_a_df, keep_t_df, logger)

    logger.write(f'  * {variant_df.shape[0]} variants')
    if maf_threshold > 0:
        logger.write(f'  * applying in-sample {maf_threshold} MAF filter')
    logger.write(f'  * cis-window: ±{window:,}')

    ase_covariates_t, n_cov_a = _resolve_ase_covariates(
        ase_covariates_df, covariates_df, samples, device, logger)
    # one t reference for the combined statistic: the larger design's
    dof = N - 2 - max(n_cov, n_cov_a)
    library_factor_t = _library_factor_tensor(library_factor, samples, device)
    if perm_scheme not in PERM_SCHEMES:
        raise ValueError(f'perm_scheme must be one of {PERM_SCHEMES}, got {perm_scheme!r}')
    _check_variance_model(variance_model, tau_mode, library_factor_t, variance_prior)
    logger.write(f'  * variance model: {variance_model}' + (' with empirical-Bayes prior' if variance_prior is not None else ''))

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

        prior_g = _prior_tuple(variance_prior, phenotype_id)
        keep_a_t = _keep_row(keep_a_df, pidx, device)
        keep_t_t = _keep_row(keep_t_df, pidx, device)
        sqrt_wa_t, sqrt_wt_t, residualizer_a, residualizer_tc, tau_info = _prepare_channels(
            a_t, t_t, va_t, vt_t, covariates_t, tau_mode, device,
            ase_covariates_t=ase_covariates_t, return_info=True,
            variance_model=variance_model, library_factor_t=library_factor_t, prior=prior_g,
            keep_a_t=keep_a_t, keep_t_t=keep_t_t,
            fitted_scale=(se_mode == 'fitted'))

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
            permutation_ix_t, dof=dof, perm_scheme=perm_scheme,
            fitted=(se_mode == 'fitted'),
        )
        r_nominal, std_ratio, var_ix, r2_perm, g = [i.cpu().numpy() for i in res]
        best_local = int(var_ix)
        var_ix = genotype_range[var_ix]

        # the lead on the scan scale: per-channel slopes for the cis/trans diagnostic
        g_lead = genotypes_t[best_local:best_local + 1]
        s_lead = sign_t[best_local:best_local + 1]
        lead_stat = calculate_hapmixqtl_nominal(
            g_lead, s_lead, a_t, t_t, sqrt_wa_t, sqrt_wt_t, residualizer_a, residualizer_tc)
        tau_used = dict(tau_a=tau_info['tau_a'], tau_t=tau_info['tau_t'], refit=False,
                        c_a=tau_info['c_a'])
        if tau_refit and tau_mode == 'estimate':
            # tau (and c) with the lead's predictor in the model; the
            # residualizers still project out the null design only
            wa_r, wt_r, res_a_r, res_t_r, info_r = _prepare_channels(
                a_t, t_t, va_t, vt_t, covariates_t, tau_mode, device,
                ase_covariates_t=ase_covariates_t, return_info=True,
                tau_extra_a_t=s_lead.t(), tau_extra_t_t=(g_lead / 2).t(),
                variance_model=variance_model, library_factor_t=library_factor_t, prior=prior_g,
                keep_a_t=keep_a_t, keep_t_t=keep_t_t)
            lead_stat = calculate_hapmixqtl_nominal(
                g_lead, s_lead, a_t, t_t, wa_r, wt_r, res_a_r, res_t_r)
            tau_used = dict(tau_a=info_r['tau_a'], tau_t=info_r['tau_t'],
                            refit=bool(info_r['refit_a'] or info_r['refit_t']),
                            c_a=info_r['c_a'])
        (lead_tstat, lead_slope, lead_slope_se, lead_a, lead_a_se, lead_t, lead_t_se) = [
            float(x.cpu().numpy()[0]) for x in lead_stat]
        alpha_cis, pval_cis_trans = cis_trans_diagnostic(
            lead_a, lead_a_se, lead_t, lead_t_se, dof)

        variant_id = variant_df.index[var_ix]
        start_distance = variant_df['pos'].values[var_ix] - igc.phenotype_start[phenotype_id]
        end_distance = variant_df['pos'].values[var_ix] - igc.phenotype_end[phenotype_id]

        # empirical p and the beta approximation stay on the scan scale
        r2_nominal = r_nominal * r_nominal
        pval_perm = (np.sum(r2_perm >= r2_nominal) + 1) / (nperm + 1)

        if tau_used['refit']:
            slope, slope_se = lead_slope, lead_slope_se
            pval_nominal = float(get_t_pval(lead_tstat, dof)) if np.isfinite(lead_tstat) else np.nan
        else:
            slope = r_nominal * std_ratio
            tstat2 = dof * r2_nominal / (1 - r2_nominal) if r2_nominal < 1 else np.inf
            slope_se = np.abs(slope) / np.sqrt(tstat2) if tstat2 > 0 else np.inf
            pval_nominal = pval_from_corr(r2_nominal, dof)

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
            ('pval_nominal', pval_nominal),
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
            ('tau_a', tau_used['tau_a']),
            ('tau_t', tau_used['tau_t']),
            ('tau_a_null', tau_info['tau_a']),
            ('tau_t_null', tau_info['tau_t']),
            ('tau_refit', tau_used['refit']),
            ('c_a', tau_used['c_a']),
            ('c_a_null', tau_info['c_a']),
            ('c_a_converged', bool(tau_info['converged_a'])),
            ('c_a_raw', tau_info['c_a_raw']),
            ('tau_a_raw', tau_info['tau_a_raw']),
            ('c_a_floored', bool(tau_info['floored_a'])),
            ('variance_prior', bool(tau_info['prior_used'])),
            ('variance_model', variance_model),
            ('perm_scheme', perm_scheme),
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
    n_floored = int(res_df['c_a_floored'].astype(bool).sum())
    if n_floored:
        logger.write(f'  * {n_floored} of {len(res_df)} phenotypes had c or tau clamped at zero in the allelic fit (c_a_floored)')
    n_unconverged = int((~res_df['c_a_converged'].astype(bool)).sum())
    if n_unconverged:
        logger.write(f'  * WARNING: the allelic (c, tau) fit did not converge on '
                     f'{n_unconverged} of {len(res_df)} phenotypes (c_a_converged False); '
                     f'their last iterate was used')
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
    total intercept/covariates and explicit ASE covariates from each channel,
    so the stacked responses and predictors are covariate-free (SuSiE is then called with
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
              warn_monomorphic=False, ase_covariates_df=None,
              variance_model='additive', library_factor=None, variance_prior=None,
              keep_a_df=None, keep_t_df=None):
    """
    hapmixQTL SuSiE fine-mapping. ``variance_model`` and ``library_factor``
    select the allelic channel's error variance as in map_cis.

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
        _assert_phase_columns(xL_df, xR_df, genotype_df)
    else:
        logger.write('  * no phase genotypes (total channel only)')

    _assert_keep_frames(keep_a_df, keep_t_df, A_df)
    _log_keep_frames(keep_a_df, keep_t_df, logger)

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

    library_factor_t = _library_factor_tensor(library_factor, A_df.columns, device)
    _check_variance_model(variance_model, tau_mode, library_factor_t, variance_prior)
    logger.write(f'  * variance model: {variance_model}' + (' with empirical-Bayes prior' if variance_prior is not None else ''))

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
            ase_covariates_t=ase_covariates_t, variance_model=variance_model,
            library_factor_t=library_factor_t, prior=_prior_tuple(variance_prior, phenotype_id),
            keep_a_t=_keep_row(keep_a_df, pidx, device),
            keep_t_t=_keep_row(keep_t_df, pidx, device))

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
                 covariates_df=None, window=1000000, tau_mode='zero',
                 se_mode='fitted', logger=None, verbose=True,
                 ase_covariates_df=None, variance_model='additive',
                 library_factor=None, variance_prior=None):
    """Shared per-phenotype driver: whiten once per gene, call fit_site for
    every site in the cis window, collect its row dicts. ``variance_model``
    and ``library_factor`` as in map_cis."""
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
    library_factor_t = _library_factor_tensor(library_factor, samples, device)
    _check_variance_model(variance_model, tau_mode, library_factor_t, variance_prior)
    logger.write(f'  * variance model: {variance_model}' + (' with empirical-Bayes prior' if variance_prior is not None else ''))
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
            ase_covariates_t=ase_covariates_t, variance_model=variance_model,
            library_factor_t=library_factor_t, prior=_prior_tuple(variance_prior, pid))
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
                     window=1000000, min_hap=10, tau_mode='zero',
                     se_mode='fitted', logger=None, verbose=True,
                     ase_covariates_df=None, variance_model='additive',
                     library_factor=None, variance_prior=None):
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
                 ase_covariates_df=ase_covariates_df,
                 variance_model=variance_model, library_factor=library_factor,
                 variance_prior=variance_prior)
    site_cols = ['phenotype_id', 'site_id', 'start_distance', 'n_alleles', 'n_tested',
                 'ref_allele', 'pooled_other', 'n_missing_hap', 'pval_joint', 'chi2',
                 'dof', 'rank_deficient']
    allele_cols = ['phenotype_id', 'site_id', 'allele', 'n_hap', 'slope', 'slope_se',
                   'pval', 'slope_a', 'slope_a_se', 'slope_t', 'slope_t_se']
    return (pd.DataFrame(site_res, columns=site_cols),
            pd.DataFrame(allele_res, columns=allele_cols))


def map_str_curvature(str_len, str_phased, str_df, site_samples, A_df, T_df, Va_df, Vt_df,
                      phenotype_pos_df, covariates_df=None, window=1000000,
                      winsor=(0.01, 0.99), tau_mode='zero', se_mode='fitted',
                      logger=None, verbose=True, ase_covariates_df=None,
                      variance_model='additive', library_factor=None, variance_prior=None):
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
                 ase_covariates_df=ase_covariates_df,
                 variance_model=variance_model, library_factor=library_factor,
                 variance_prior=variance_prior)
    cols = ['phenotype_id', 'str_id', 'start_distance', 'n_called', 'n_phased',
            'len_center', 'len_lo', 'len_hi', 'slope_lin', 'slope_lin_se', 'pval_lin',
            'slope_l', 'slope_l_se', 'slope_sq', 'slope_sq_se', 'pval_curv',
            'slope_at_ref', 'slope_at_ref_se',
            'slope_sq_a', 'slope_sq_a_se', 'slope_sq_t', 'slope_sq_t_se',
            'pval_joint2', 'rank_deficient']
    return pd.DataFrame(out, columns=cols)

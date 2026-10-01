"""
hapmixQTL: cis-QTL mapping using haplotype-resolved expression.

The production default (2026-09-29) is half-read split preprocessing:
prepare_default_inputs constructs ASE log2((pL+0.5)/(pR+0.5)), its admitted
Gibbs/count-noise variances, total log2((pT+0.5)/(Leff+1)*1e6), and unit total
working variances. Salmon point estimates supply the expression values.
Expression PCs are built on the same half-read log-CPM
(scripts/build_covariates.py, since 2026-09-30). The older
summaries_from_point_estimates and compute_summaries_from_gibbs functions
remain available for reproducing earlier analyses.

The mapping APIs consume explicitly supplied phenotypes and variances; they
do not transform counts or replace caller-provided weights. Nominal and permutation mapping default
to tau_mode='zero' and se_mode='fitted': relative inverse-variance weights
are accompanied by empirical per-channel residual scales. Unit Vt therefore
gives unweighted total-expression regression with a fitted residual SE, not
an assertion that total expression has known error variance one.

Separate regressions fit ASE on signed heterozygosity s=xL-xR and total on
g/2, then combine the effects with the existing inverse-variance weighting,
Meier correction and Welch-Satterthwaite reference. The sqrt-weight transform
and matrix multiplication evaluate all cis variants on the GPU. The half-read
change does not alter these regression or permutation kernels.

Total expression uses covariates_df plus an intercept. ASE's default
ase_covariates_df=None is through-origin; SAME_COVARIATES reuses total
covariates. Insufficiently informative channels are disabled; fitted combined
mapping also applies min_allelic_donors. Genotype-tied covariates remain tied
to genotypes during permutation; RNA records and their weights move together
under the fitted record-permutation null. See map_nominal/map_cis for exact
reference distributions and admission rules.

Phase determines s: +1 for ALT on L, -1 for ALT on R, zero for homozygous or
unphased samples. Phase frames must have exactly the genotype sample order;
_assert_phase_columns checks this at each mapping entry point. Without phase,
the ASE channel contributes nothing and mapping uses total expression alone.

Cat is inspection-only and is not consumed by the channel combination. The
new default helper does not calculate it. Historical summarizers and the BED
reader retain optional Cat support for compatibility. Nondefault variance
models and known-variance/robust modes remain available for historical work;
see docs/hapmixqtl_methods.md and the individual function documentation.
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

def read_hapmixqtl_inputs(a_bed, t_bed, va_bed, vt_bed=None, cat_bed=None):
    """
    Load precomputed hapmixQTL summary matrices in BED-like format.

    Each file follows the tensorQTL phenotype BED convention:
    chr, start, end, phenotype_id, sample1, sample2, ...

    The default split-weight method expects half-read log-CPM in T and the
    admitted ASE variances from prepare_default_inputs in Va. Omitting vt_bed
    supplies unit total-expression working variances. An explicit vt_bed is
    preserved as a custom-weight override. BED inputs have no raw counts or
    library sizes, so this reader cannot construct or verify the transform.

    Returns:
        A_df:   allelic contrast a_i [phenotypes x samples]
        T_df:   log total t_i [phenotypes x samples]
        Va_df:  inferential variance of a [phenotypes x samples]
        Vt_df:  total working variance [phenotypes x samples]; ones by default
        Cat_df: inferential covariance a,t [phenotypes x samples] (or None).
                Loaded for inspection only: it is INTENTIONALLY UNUSED by every
                mapping function (see the module docstring for why).
        pos_df: phenotype positions [phenotypes x (chr, pos|start,end)]
    """
    A_df, pos_df = read_phenotype_bed(a_bed)
    T_df, _ = read_phenotype_bed(t_bed)
    Va_df, _ = read_phenotype_bed(va_bed)
    if vt_bed is None:
        Vt_df = pd.DataFrame(1.0, index=T_df.index, columns=T_df.columns)
    else:
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


def _validate_inputs(genotype_df, A_df, T_df, Va_df, Vt_df, xL_df=None, xR_df=None):
    """The input contract of map_nominal and map_cis, which read every frame
    positionally after this point. Without it (implementation_audit_20260914,
    sec 1, re-run on the default mode 2026-10-01) a NaN in A or T silently
    dropped that channel and moved the lead, a negative or NaN Va was read as
    "no information", Va columns or phase rows in another order were accepted
    and changed the result, and one phase frame alone ran a total-only
    analysis. prepare_default_inputs never produces any of these: A and T are
    finite by construction and an excluded donor-gene pair has Va = 0."""
    for name, df in (('A_df', A_df), ('T_df', T_df), ('Va_df', Va_df), ('Vt_df', Vt_df)):
        if not df.columns.equals(A_df.columns):
            raise ValueError(f'{name} columns (samples) differ from A_df columns in identity or order.')
        if not df.index.equals(A_df.index):
            raise ValueError(f'{name} rows (phenotypes) differ from A_df rows in identity or order.')
        values = df.to_numpy(dtype=float)
        bad = ~np.isfinite(values)
        if name in ('Va_df', 'Vt_df'):
            bad |= values < 0
        if bad.any():
            i, j = np.argwhere(bad)[0]
            raise ValueError(
                f'{name} has {int(bad.sum())} missing, non-finite'
                f'{" or negative" if name in ("Va_df", "Vt_df") else ""} values '
                f'(first: phenotype {df.index[i]}, sample {df.columns[j]}).'
                + (' An excluded donor-gene pair is Va = 0 with a finite A, as '
                   'prepare_default_inputs writes it.' if name in ('A_df', 'Va_df') else ''))
    for what, ids in (('phenotype', A_df.index), ('sample', A_df.columns), ('variant', genotype_df.index)):
        if ids.has_duplicates:
            raise ValueError(f'duplicate {what} ids, e.g. {ids[ids.duplicated()][0]}')
    if (xL_df is None) != (xR_df is None):
        raise ValueError('pass both phase frames (xL_df and xR_df) or neither; one alone '
                         'would silently run a total-only analysis.')
    if xL_df is None:
        return
    _assert_phase_columns(xL_df, xR_df, genotype_df)
    for name, df in (('xL_df', xL_df), ('xR_df', xR_df)):
        if not df.index.equals(genotype_df.index):
            raise ValueError(f'{name} rows (variants) differ from genotype_df rows in identity or '
                             'order; phase is read by genotype row position.')
        if not np.isfinite(df.to_numpy(dtype=float)).all():
            raise ValueError(f'{name} has missing or non-finite phase values.')


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
    Historical natural-log Gibbs-mean summaries, retained for reproduction.
    For the current point-estimate half-read split route use
    prepare_default_inputs. The behavior described below is this helper's
    historical contract, not the production association default.

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


LN2 = float(np.log(2.0))


def half_read_log_cpm(counts, eff_lib_size):
    """Transform point counts [features, samples] with a half-read offset.

    The offset is in read units, before normalization, rather than one CPM:
    ``log2((counts + 0.5) / (eff_lib_size + 1) * 1e6)``. Zero counts remain
    finite. Effective library sizes are the existing edgeR/TMM values.
    """
    counts = np.asarray(counts, dtype=float)
    L = np.asarray(eff_lib_size, dtype=float)
    if counts.ndim != 2:
        raise ValueError('counts must be [features, samples]')
    if not np.all(np.isfinite(counts)) or np.any(counts < 0):
        raise ValueError('counts must be finite and nonnegative')
    if L.shape != (counts.shape[1],) or not np.all(np.isfinite(L)) or not np.all(L > 0):
        raise ValueError('eff_lib_size must be one finite positive value per sample')
    return np.log2((counts + 0.5) / (L[None, :] + 1.0) * 1e6)


def prepare_default_inputs(pL, pR, pT, eff_lib_size, yL, yR,
                           kappa=0.5, count_noise=True):
    """Prepare the half-read split default adopted on 2026-09-29.

    Values come from Salmon point estimates. A and its Gibbs/count-noise
    variance retain the historical point-estimate definition; T uses
    half_read_log_cpm and Vt is a unit *working* variance, not an estimate of
    total measurement uncertainty. With tau_mode='zero', se_mode='fitted',
    the mapper fits empirical residual scales for both channels.

    pL/pR are paired-transcript counts; pT includes ALL transcripts and must
    not be substituted with pL+pR. Counts have shape [features, samples];
    yL/yR are the corresponding [features, samples, draws] Gibbs arrays.
    No total Gibbs transform or unused ASE-total covariance is computed.

    ASE admission matches the evaluated split arm: variance must exceed
    1e-12, no-coverage donor-gene pairs are excluded, and exactly one haplotype below
    0.5 reads excludes that donor-gene pair. An excluded pair has Va=0.
    Total expression retains every donor, including zero-count donors.
    Expression PCs are built on the same half-read log-CPM
    (scripts/build_covariates.py, since 2026-09-30).

    Returns A, T, Va, Vt. The historical summaries_from_point_estimates
    utility and mapping APIs accepting explicit variances remain unchanged.
    """
    pL, pR, pT = (np.asarray(x, dtype=float) for x in (pL, pR, pT))
    yL, yR = (np.asarray(x, dtype=float) for x in (yL, yR))
    if any(x.ndim != 2 for x in (pL, pR, pT)) or not (pL.shape == pR.shape == pT.shape):
        raise ValueError('point counts must have matching [features, samples] shapes')
    if (yL.ndim != 3 or yR.shape != yL.shape or
            yL.shape[:2] != pL.shape or yL.shape[2] == 0):
        raise ValueError('Gibbs draws must be nonempty [features, samples, draws] matching point counts')
    for name, values in (('pL', pL), ('pR', pR), ('pT', pT), ('yL', yL), ('yR', yR)):
        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError(f'{name} counts must be finite and nonnegative')
    if not np.isscalar(kappa) or not np.isfinite(kappa) or kappa <= 0:
        raise ValueError('kappa must be finite and positive')
    T = half_read_log_cpm(pT, eff_lib_size)
    A = np.log2((pL + kappa) / (pR + kappa))
    a_d = np.log2((yL + kappa) / (yR + kappa))
    Va = a_d.var(axis=2, ddof=0)
    if count_noise:
        Va += (1.0 / (pL + kappa) + 1.0 / (pR + kappa)) / LN2 ** 2
    keep_a = ((Va > 1e-12) & ((pL + pR) > 0) &
              ~((pL < 0.5) ^ (pR < 0.5)))
    Va = np.where(keep_a, Va, 0.0)
    return A, T, Va, np.ones_like(T)


def summaries_from_point_estimates(pL, pR, pT, eff_lib_size, yL, yR, yT,
                                   kappa=0.5, count_noise=True):
    """Historical 2026-09-25 summaries, retained for reproduction.

    For the current half-read split default use prepare_default_inputs. This
    older helper uses log2(CPM+1) totals and Gibbs variances in both channels.
    VALUES come from Salmon point estimates; VARIANCE from the Gibbs draws.

    User rules (2026-09-25): every value is computed from the point estimates
    (quant.sf NumReads); the Gibbs draws are used ONLY for the measurement
    variance of that same value, through the identical transform. The unit is
    log2(CPM + 1), with CPM from edgeR's effective library size (lib.size x
    TMM norm.factors, computed by edgeR on the calibration gene set). The one
    exception is the allelic ratio, a within-sample contrast in which library
    size cancels, which stays a pseudocounted log ratio of haplotype counts:

        a  = log2((pL + kappa) / (pR + kappa))
        t  = log2(pT / Leff * 1e6 + 1)
        Va = var over draws of log2((yL + kappa) / (yR + kappa))  [+ q_a]
        Vt = var over draws of log2(yT / Leff * 1e6 + 1)           [+ q_t]

    The counting terms are the delta-method Poisson variance of the same
    transform at the point estimate: q_a = (1/(pL+kappa) + 1/(pR+kappa)) /
    ln2^2, and q_t = k^2 (pT + 1/2) / ((k (pT + 1/2) + 1)^2 ln2^2) with
    k = 1e6 / Leff, evaluated at count + 1/2 so a zero-read donor keeps a
    positive variance (user decision 2026-09-25) -- on this scale a zero-read
    donor is a precise observation of no expression. They are the floor the
    total channel otherwise lacks, as count_noise is in
    compute_summaries_from_gibbs. A donor with no haplotype-informative reads
    (pL + pR = 0) keeps Va = 0, so the allelic channel excludes it.

    Args:
        pL, pR: point-estimate haplotype counts [features, samples], summed
                over paired transcripts only (as the Gibbs ingest does)
        pT:     point-estimate total counts [features, samples], summed over
                ALL transcripts (never pL + pR; see compute_summaries_from_gibbs)
        eff_lib_size: edgeR effective library sizes [samples]
        yL, yR, yT: the Gibbs draws [features, samples, draws], same summing
    Returns:
        A, T, Va, Vt, Cat -- Cat is the across-draw covariance of a and t,
        returned for inspection and unused by the mapping functions.
    """
    pL, pR, pT = (np.asarray(x, dtype=float) for x in (pL, pR, pT))
    yL, yR, yT = (np.asarray(x, dtype=float) for x in (yL, yR, yT))
    L = np.asarray(eff_lib_size, dtype=float)
    if pL.shape != pR.shape or pL.shape != pT.shape:
        raise ValueError(f'point-estimate shapes differ: {pL.shape}, {pR.shape}, {pT.shape}')
    if yL.shape[:2] != pL.shape or yR.shape != yL.shape or yT.shape != yL.shape:
        raise ValueError('Gibbs draw arrays must be [features, samples, draws] matching the point estimates')
    if L.shape != (pL.shape[1],) or not np.all(np.isfinite(L)) or not np.all(L > 0):
        raise ValueError('eff_lib_size must be one positive value per sample')
    k = 1e6 / L                                                  # CPM per count
    A = np.log2((pL + kappa) / (pR + kappa))
    T = np.log2(pT * k[None, :] + 1.0)
    a_d = np.log2((yL + kappa) / (yR + kappa))
    t_d = np.log2(yT * k[None, :, None] + 1.0)
    Va = a_d.var(axis=2, ddof=0)
    Vt = t_d.var(axis=2, ddof=0)
    Cat = ((a_d - a_d.mean(axis=2, keepdims=True)) *
           (t_d - t_d.mean(axis=2, keepdims=True))).mean(axis=2)
    no_cov = (pL + pR) <= 0
    if count_noise:
        q_a = (1.0 / (pL + kappa) + 1.0 / (pR + kappa)) / LN2 ** 2
        y = pT + 0.5
        q_t = (k[None, :] ** 2) * y / ((k[None, :] * y + 1.0) ** 2 * LN2 ** 2)
        Va = Va + q_a
        Vt = Vt + q_t
    Va = np.where(no_cov, 0.0, Va)
    return A, T, Va, Vt, Cat


def _combine_covariates(covariates_df, genotype_covariates_df, samples, logger=None):
    """One covariate design with the genotype-tied columns LAST.

    ``covariates_df`` holds the covariates tied to the RNA record (metadata,
    expression PCs); ``genotype_covariates_df`` those tied to the genotypes
    (genotype PCs). Both enter the regression identically; they differ only in
    the permutation, where the genotype-tied columns stay with the genotypes
    (user rule, 2026-09-25). Returns the combined frame (or None) and the
    number of genotype-tied columns.

    Rejects a design that is not finite or not of full column rank together
    with the total channel's intercept: a QR of a deficient design returns a
    basis direction the covariates do not span, which changes the projection
    (L2 1.18 on a duplicated covariate; implementation_audit_20260914 sec 2)
    and charges a degree of freedom for it.
    """
    n_genotype = 0
    combined = covariates_df
    if genotype_covariates_df is not None:
        if not np.all(np.asarray(samples) == np.asarray(genotype_covariates_df.index)):
            raise ValueError('genotype-covariate samples must match phenotype samples, in order')
        if covariates_df is not None:
            if not np.all(np.asarray(samples) == np.asarray(covariates_df.index)):
                raise ValueError('covariate samples must match phenotype samples, in order')
            clash = set(covariates_df.columns) & set(genotype_covariates_df.columns)
            if clash:
                raise ValueError(f'columns in both covariate frames: {sorted(clash)}')
            combined = pd.concat([covariates_df, genotype_covariates_df], axis=1)
        else:
            combined = genotype_covariates_df.copy()
        n_genotype = int(genotype_covariates_df.shape[1])
        if logger is not None:
            logger.write(f'  * {n_genotype} covariates tied to the genotypes '
                         f'(stay with them under permutation): {list(genotype_covariates_df.columns)}')
    if combined is not None:
        design = np.column_stack([np.ones(len(combined)), combined.to_numpy(dtype=float)])
        if not np.isfinite(design).all():
            raise ValueError('covariates contain missing or non-finite values')
        rank = int(np.linalg.matrix_rank(design))
        if rank < design.shape[1]:
            raise ValueError(
                f'the covariate design (intercept plus {combined.shape[1]} covariates) has rank '
                f'{rank} of {design.shape[1]}: drop a duplicated, constant or collinear column')
    return combined, n_genotype


def count_cutoff_masks(yL, yR, yT=None, asc_cutoff=None, asc_cap=None,
                       trc_cutoff=None):
    """Per-channel admission masks from mixQTL-style count cutoffs.

    These optional masks are additional to prepare_default_inputs' fixed
    ASE admission rule (no coverage, negligible Va, or one-sided counts below
    0.5 reads are excluded). Total expression otherwise keeps every donor.
    This builds the masks that reproduce mixQTL's count thresholds,
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
        # the LAST n_fixed_cov columns of C_t are tied to the genotypes (genotype
        # PCs): the donor-record permutation leaves them at their position while
        # the other covariate columns move with the RNA record. Set by map_cis.
        self.n_fixed_cov = 0
        self.sqrt_w_t = sqrt_w_t
        if design.shape[1] > 0 and bool((sqrt_w_t != 0).any()):
            # _combine_covariates rejects a deficient covariate matrix; zero
            # weights can still make one gene's weighted design deficient (a
            # categorical covariate constant over the donors that carry weight,
            # under ase_covariates or count cutoffs). A QR would add a direction
            # the design does not span, here and in every permutation. A column
            # in the span of the ones before it leaves a diagonal entry of R
            # at rounding level relative to its own norm (scale-free per column).
            self.Q_t, R = torch.linalg.qr(design)
            tol = max(design.shape) * torch.finfo(design.dtype).eps
            dependent = R.diagonal().abs() <= tol * torch.linalg.vector_norm(design, dim=0)
            if bool(dependent.any()):
                raise ValueError(
                    f'weighted design has {int(dependent.sum())} of {design.shape[1]} columns '
                    f'dependent on the others over the {int((sqrt_w_t != 0).sum())} donors with '
                    'nonzero weight; drop the covariate that is constant or collinear over them')
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
    Regression on sqrt-weight-transformed data, evaluated by dot products.

    The production association mappers call this with fitted=True. They use
    relative weights (Gibbs/count-noise ASE variances or unit total working
    variances) and estimate Var(beta_hat) = sigma2_hat / xx from residuals.
    Rescaling all weights leaves that fitted SE and the slope unchanged.

    The low-level default fitted=False retains known-variance GLS for
    historical callers: Var(beta_hat) = 1 / xx assumes the supplied weights
    are absolute precisions. robust=True instead selects HC1 unless fitted
    takes precedence. These branches are not the default association model.

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
            # which is wrong here for exactly that reason (_channel_dof).
            e = y_res - slope.unsqueeze(1) * x_res
            rss = (e * e).sum(1)
            dof_f = _channel_dof(residualizer)
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




# ---------------------------------------------------------------------------
#  Association tests
# ---------------------------------------------------------------------------



def _min_informative(covariates_t, extra=2, intercept=True):
    """Informative samples a channel needs to stay on: one per design column
    (automatic intercept, if present, plus covariates) plus `extra` residual
    degrees of freedom."""
    n_cov = 0 if covariates_t is None else covariates_t.shape[1]
    return int(intercept) + n_cov + extra


# Default mode's allelic admission floor: the allelic channel enters the
# combined statistic only for genes with at least this many informative
# allelic donors. It is mixQTL's META_N_CUTOFF (meta_analyze combines only when
# both channels have n >= 15; rlib_meta.R:34-110 via mixqtl_replication.py).
MIN_ALLELIC_DONORS = 15


def _residual_dof(residualizer):
    """Residual degrees of freedom of one channel's fitted scale
    (se_mode='fitted'), and so the t reference of that channel's statistic.

    Counts INFORMATIVE donors, not rows: a zero-weight donor contributes
    nothing to the residual sum of squares, so charging it a degree of
    freedom would shrink the scale and inflate the statistic. One for the
    slope and one per column the residualizer projects out: n_a - 1 - n_cov_a
    for the allelic channel (through-origin by default, so n_a - 1), and
    n_t - 2 - n_cov for the total channel. At least 1 for every channel that
    is on (the sparse-channel rule of _channel_weights switches off a channel
    with fewer informative donors than design columns plus two), so it is
    below 1 exactly when the channel is off.
    """
    n_eff = int((residualizer.sqrt_w_t != 0).sum())
    return n_eff - 1 - residualizer.Q_t.shape[1]


def _channel_dof(residualizer):
    """_residual_dof clamped at 1, so a channel without residual degrees of
    freedom never divides by zero. The single definition used by the standard
    error (_wls_regression), the permutation scan
    (calculate_hapmixqtl_permutations) and the combination's reference
    (_satterthwaite_dof)."""
    return max(_residual_dof(residualizer), 1)


def _reported_dof(residualizer):
    """_residual_dof for the output columns: NaN, not the clamp, when the
    channel is off, so its p-value is NaN rather than a zero statistic's 1."""
    dof = _residual_dof(residualizer)
    return float(dof) if dof >= 1 else float('nan')


def _allelic_admitted(residualizer_a, residualizer_t, min_allelic_donors, fitted):
    """Whether the allelic channel enters the combined statistic: the gene's
    count of informative allelic donors against MIN_ALLELIC_DONORS (a
    gene-level count; it does not check that a given variant has heterozygous
    informative donors, or that phase was supplied).

    The floor applies in default mode only, and only when the total channel
    is on. When it is off, e.g. in an allelic-only run (keep_t_df all False),
    there is no combination to protect and the allelic channel is the
    statistic whenever it is on, as mixQTL's meta_analyze falls back to the
    channel it has. Under the known-variance and HC1 standard errors any
    informative allelic donor admits it, as before."""
    n_a = int((residualizer_a.sqrt_w_t != 0).sum())
    if not fitted:
        return n_a >= 1
    if _residual_dof(residualizer_t) < 1:
        return _residual_dof(residualizer_a) >= 1
    return n_a >= min_allelic_donors


def _satterthwaite_dof(w_a, w_t, dof_a, dof_t):
    """Welch-Satterthwaite degrees of freedom of the combined statistic.

    The combination is sum_k c_k beta_k with c_k = w_k / sum w and
    w_k = 1/se_k^2, its variance estimate is sum_k c_k^2 se_k^2 = 1/sum w,
    and for a FIXED linear combination of independent variance estimates
    the Welch-Satterthwaite degrees of freedom are

        (sum_k c_k^2 se_k^2)^2 / sum_k (c_k^2 se_k^2)^2 / nu_k
            = (w_a + w_t)^2 / (w_a^2 / nu_a + w_t^2 / nu_t)

    with nu_k the channel's _channel_dof. The value lies between
    min(nu_a, nu_t) and nu_a + nu_t, varies per variant because the weights
    do, is exactly nu_a where the total channel carries no weight (e.g. an
    allelic-only run) and exactly nu_t where the allelic channel carries none
    (below the admission floor, or no heterozygous informative donor), and is
    NaN where neither does (no statistic, so no p-value). Computed in float64
    on the weight shares, so large weights cannot overflow.

    The weights are treated as fixed, although they are estimated from the
    same residuals, and that costs calibration where both channels carry
    weight: a chance-small se_a both inflates |t_a| and raises the allelic
    share (the Graybill-Deal effect). Measured 2026-09-27 under the exact
    model (normal errors at known weights, N = 75, median allelic share 0.50,
    20,000 null genes per arm, map_nominal): at n_a = 15 the combined p
    rejects at 0.0592 / 0.01355 / 0.00155 at 0.05 / 0.01 / 0.001, about
    1.2x / 1.35x / 1.5x nominal, while each channel alone is nominal on its
    own dof (the old shared N - 2 reference gave 0.0627 / 0.0163 / 0.0024);
    0.0598 / 0.01245 / 0.00115 at n_a = 20 and 0.0547 / 0.0104 / 0.00155 at
    n_a = 40; an independent seed gives 0.0587 / 0.01205 / 0.0014 at
    n_a = 15. Below the floor it is worse (n_a = 3: 0.116 at 0.05), which is
    what MIN_ALLELIC_DONORS excludes. Since 2026-09-27 (user decision) the
    combined standard error carries Meier's first-order correction for that
    effect (_meier_factor); the dof here is unchanged.

    Args:
        w_a, w_t: [V] per-variant channel weights 1/se^2 (0 where absent)
        dof_a, dof_t: the channels' residual degrees of freedom
    Returns:
        [V] float64 tensor
    """
    w_a = w_a.to(torch.float64)
    w_t = w_t.to(torch.float64)
    total = w_a + w_t
    f_a = torch.where(total > 0, w_a / total.clamp(min=1e-300), torch.zeros_like(total))
    f_t = 1.0 - f_a
    nu = 1.0 / (f_a * f_a / dof_a + f_t * f_t / dof_t)
    nu = torch.where(w_t > 0, nu, torch.full_like(nu, float(dof_a)))
    nu = torch.where(w_a > 0, nu, torch.full_like(nu, float(dof_t)))
    return torch.where(total > 0, nu, torch.full_like(nu, float('nan')))


def _meier_factor(w_a, w_t, dof_a, dof_t):
    """Meier's first-order inflation of the combined statistic's variance for
    channel weights estimated from the same residuals (default mode; user
    decision 2026-09-27).

    The combination is the inverse-variance weighted mean of the two channel
    slopes with weights w_k = 1/se_k^2 ESTIMATED from the residuals it
    combines: a Graybill-Deal mean (Graybill and Deal 1959), whose plug-in
    variance 1/W, W = w_a + w_t, is too small because a chance-small se_k is
    at once a chance-large weight and a chance-large |t_k|. Meier (1953,
    Biometrics 9:59-73, "Variance of a weighted mean") gives, to first order
    in 1/nu_k with f_k = w_k/W the weight shares and nu_k the channels'
    residual degrees of freedom, a true variance (1/W)[1 + 2 S] against an
    expected reported variance (1/W)[1 - 2 S], S = sum_k f_k (1 - f_k)/nu_k;
    the ratio is 1 + 4 S, and with two channels f_a (1 - f_a) = f_t (1 - f_t)
    = f_a f_t, so

        M = 1 + 4 f_a f_t (1/nu_a + 1/nu_t)

    per variant, from the fitted channel weights and the same _channel_dof
    as _satterthwaite_dof. The combined SE becomes se_c sqrt(M) and the
    combined t falls by sqrt(M), referred to the unchanged Welch-Satterthwaite
    dof. M is exactly 1 wherever fewer than two channels carry weight: with
    one channel absent, below the admission floor or without a heterozygous
    informative donor the combination IS the other channel, whose t is exact
    on its own dof, and with neither the zero statistic stays zero (the
    info dict reports NaN there, like dof_nominal). Evaluated at the
    estimated shares, in float64 like _satterthwaite_dof. Its measured
    effect is in docs/hapmixqtl_methods.md, Section 4.5.

    Args:
        w_a, w_t: [V] or [V, nperm] channel weights 1/se^2 (0 where absent)
        dof_a, dof_t: the channels' residual degrees of freedom
    Returns:
        float64 tensor of w_a's shape
    """
    w_a = w_a.to(torch.float64)
    w_t = w_t.to(torch.float64)
    f_a = w_a / (w_a + w_t).clamp(min=1e-300)
    m = 1.0 + 4.0 * (1.0 / dof_a + 1.0 / dof_t) * f_a * (1.0 - f_a)
    return torch.where((w_a > 0) & (w_t > 0), m, torch.ones_like(m))


def _fitted_variance():
    """Import the DEPRECATED fitted-variance machinery on demand.

    Neither shipped mode reaches this. Importing lazily, from inside the
    deprecated branches only, is what keeps `tensorqtl.fitted_variance` out of
    a default-mode run entirely (a test pins that it is never imported), and
    keeps the dependency one-way so that module can borrow from this one.
    """
    try:                                                # package import
        from . import fitted_variance
    except ImportError:                                 # flat import
        import fitted_variance
    return fitted_variance


def _check_variance_model(variance_model, tau_mode, library_factor_t, variance_prior=None):
    """Live gate on the variance path. The two shipped modes need no
    fitted-variance machinery, so they return here without importing it;
    anything else is deprecated and is validated by the quarantine."""
    if (tau_mode == 'zero' and variance_model == 'additive'
            and library_factor_t is None and variance_prior is None):
        return
    _fitted_variance()._check_variance_model(
        variance_model, tau_mode, library_factor_t, variance_prior)


def _library_factor_tensor(library_factor, samples, device):
    """Live gate: no per-library factor is the shipped case. A supplied one
    belongs to the deprecated `library_scaled` model."""
    if library_factor is None:
        return None
    return _fitted_variance()._library_factor_tensor(library_factor, samples, device)


def _prior_tuple(variance_prior, gene):
    """Live gate: no variance prior is the shipped case. A supplied frame
    belongs to the deprecated shrinkage path."""
    if variance_prior is None:
        return None
    return _fitted_variance()._prior_tuple(variance_prior, gene)
























def _channel_weights(y_t, v_t, covariates_t, tau_mode, device, eps=1e-12,
                     tau_extra_t=None, intercept=True, variance_model='additive',
                     d_t=None, prior=None):
    """sqrt weights of one channel, or all zeros when the channel is off.

    In the shipped default mode (`tau_mode='zero'`) the weight is `1/v_ig`,
    the reciprocal Gibbs across-draw variance, used as a SHAPE only: the
    absolute scale is carried by the fitted residual scale of
    `se_mode='fitted'`, which together give `Var(eps_i) = sigma^2 v_i`. No
    variance function is fitted here and nothing is imported from the
    quarantine.

    Returns (sqrt weights, tau, whether tau used the extra design column(s),
    c, whether the variance fit converged, the fit dict). Under the shipped
    mode tau and c are None and the fit dict is None, because no such
    parameters exist in the model -- reporting 0.0 would wrongly imply an
    additive component that was estimated and came out at zero.

    `variance_model`, `d_t` and `prior` belong to the DEPRECATED
    fitted-variance path reached only by `tau_mode='estimate'`; see
    tensorqtl/fitted_variance.py.

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
    # DEPRECATED from here: tau_mode='estimate' fits the variance function from
    # this gene's own residuals and then weights those residuals by the fit.
    # Quarantined in tensorqtl/fitted_variance.py; no shipped mode reaches it.
    return _fitted_variance()._channel_weights_estimated(
        y_t, v_t, covariates_t, device, eps, tau_extra_t, intercept,
        variance_model, d_t, prior, n_inf)


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
            "calibration of 23.3. Use DEFAULT MODE: pair tau_mode='zero' with "
            "se_mode='fitted' (calibration 1.068), which is the Var(eps)=sigma^2*v model "
            "and the shipped default. tau_mode='estimate' is NOT the remedy -- it is "
            "deprecated as of 2026-09-23 along with the known-variance SE that makes this "
            "warning fire (tensorqtl/fitted_variance.py). See docs/ase_validation.md.",
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
                                 robust=False, fitted=False,
                                 min_allelic_donors=MIN_ALLELIC_DONORS,
                                 return_info=False):
    """
    hapmixQTL association test for all variants in a cis window (Method A).

    Runs two separate WLS regressions (ASE on s, total on g/2) and
    combines via inverse-variance meta-analysis. Using g/2 as the total
    channel predictor makes its slope estimate the same quantity as the
    ASE channel slope: the full log allelic fold change (log aFC).

    With ``fitted=True`` (default mode) the allelic channel enters the
    combination only when the gene has at least ``min_allelic_donors``
    informative allelic donors (MIN_ALLELIC_DONORS; _allelic_admitted, which
    waives it when there is no total channel); below that the combined slope,
    SE and statistic ARE the total channel's, while the allelic slope and SE
    are still returned. Under the known-variance and HC1 standard errors
    there is no fitted scale, so the info dict's dof entries are None and the
    caller keeps its single reference.

    With ``fitted=True`` the combined SE is se_c sqrt(M), M the Meier factor
    of _meier_factor for channel weights estimated from the same residuals,
    wherever both channels carry weight
    (exactly se_c elsewhere); the combined t falls by sqrt(M) and is referred
    to the unchanged ``dof_nominal``. ``slope_se_t`` carries the corrected
    value and ``meier_factor`` in the info dict is M (1.0 where it does not
    apply, NaN where no channel carries weight).

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
        min_allelic_donors: the allelic admission floor (fitted only)
        return_info: if True also return the info dict below

    Returns:
        tstat_t:      [V] combined t-statistic
        slope_t:      [V] combined slope (log aFC)
        slope_se_t:   [V] combined SE
        slope_a_t:    [V] ASE channel slope
        slope_a_se_t: [V] ASE channel SE
        slope_tc_t:   [V] total channel slope
        slope_tc_se_t:[V] total channel SE
        info:         (only with return_info) dict of each statistic's t
                      reference: ``dof_a``/``dof_t`` (_reported_dof, NaN for
                      a channel that is off), ``dof_nominal`` [V]
                      (_satterthwaite_dof), ``meier_factor`` [V] float64
                      (_meier_factor; None without a fitted scale) and
                      ``allelic_admitted``
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

    admitted = _allelic_admitted(residualizer_a, residualizer_t, min_allelic_donors, fitted)
    if admitted or not fitted:
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
    else:
        # below the allelic admission floor the combination IS the total
        # channel, taken verbatim so no rounding separates the two
        inv_var_a = torch.zeros_like(inv_var_a)
        slope_combined, se_combined = slope_tc.clone(), se_tc.clone()
    if fitted:
        # the weights are estimated from the same residuals: Meier's
        # first-order inflation of the combined SE (_meier_factor), exactly
        # 1 wherever fewer than two channels carry weight
        meier = _meier_factor(inv_var_a, inv_var_t,
                              _channel_dof(residualizer_a), _channel_dof(residualizer_t))
        se_combined = se_combined * torch.sqrt(meier).to(se_combined.dtype)
    tstat_combined = slope_combined / se_combined

    out = (tstat_combined, slope_combined, se_combined, slope_a, se_a, slope_tc, se_tc)
    if not return_info:
        return out
    info = dict(dof_a=None, dof_t=None, dof_nominal=None, meier_factor=None,
                allelic_admitted=admitted)
    if fitted:
        dof_nominal = _satterthwaite_dof(inv_var_a, inv_var_t,
                                         _channel_dof(residualizer_a),
                                         _channel_dof(residualizer_t))
        info.update(dof_a=_reported_dof(residualizer_a), dof_t=_reported_dof(residualizer_t),
                    dof_nominal=dof_nominal,
                    # NaN exactly where dof_nominal is: no channel carries weight
                    meier_factor=torch.where(torch.isnan(dof_nominal), dof_nominal, meier))
    return out + (info,)


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


PERM_SCHEMES = ('records_signflip', 'records', 'residuals')
DEFAULT_PERM_SCHEME = 'records_signflip'


def _record_permutation_channel(x_t, y_t, sqrt_w_t, residualizer, permutation_ix_t, chunk=None,
                                flip_t=None, mask_t=None):
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

    ``flip_t`` [nperm, N] of +-1, if given, multiplies the whitened phenotype
    value placed at each position before residualization: the record's
    haplotype labels L and R are swapped with that sign, which negates its
    allelic log ratio and leaves its weight and covariate row unchanged. It is
    passed for the allelic channel only (``perm_scheme='records_signflip'``);
    a swap leaves total expression unchanged.

    Covariates tied to the genotypes (the genotype PCs; the LAST
    ``residualizer.n_fixed_cov`` columns of its covariate matrix, set by
    map_cis from ``genotype_covariates_df``) stay at their position with the
    genotypes; every other covariate column moves with the RNA record
    (user rule, 2026-09-25). By relabeling, this equals permuting the genotype
    columns and the genotype-covariate rows together by the inverse
    permutation, with the records and their covariates fixed.

    ``mask_t`` [nperm, N] of 0/1 multiplies the weight placed at each
    position (each row a resample); _lead_influence passes the identity order
    with one donor zeroed per row. None for the permutations.

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
    if flip_t is not None and tuple(flip_t.shape) != (nperm, N):
        raise ValueError(f'flip_t must have shape {(nperm, N)}, got {tuple(flip_t.shape)}')
    w = sqrt_w_t * sqrt_w_t
    y_star = y_t * sqrt_w_t                       # moves as a whole with the record
    C_t = getattr(residualizer, 'C_t', None)
    intercept = bool(getattr(residualizer, 'intercept', False))
    p = (1 if intercept else 0) + (int(C_t.shape[1]) if C_t is not None else 0)
    if p == 0:
        wy = w * y_t
        wy_perm = wy[permutation_ix_t]
        w_perm = w[permutation_ix_t]
        yy_perm = (y_star * y_star)[permutation_ix_t]
        if flip_t is not None:
            wy_perm = wy_perm * flip_t
        if mask_t is not None:
            wy_perm, w_perm, yy_perm = wy_perm * mask_t, w_perm * mask_t, yy_perm * mask_t
        xy = torch.mm(x_t, wy_perm.t())
        xx = torch.mm(x_t * x_t, w_perm.t())
        yy = yy_perm.sum(1)                       # a sign swap leaves squares alone
        return xy, xx, yy
    if chunk is None:
        chunk = int(max(8, min(256, 2e7 // max(V * N, 1))))
    n_fixed = int(getattr(residualizer, 'n_fixed_cov', 0) or 0)
    C_move, C_fix = C_t, None
    if C_t is not None and n_fixed > 0:
        if n_fixed > C_t.shape[1]:
            raise ValueError(f'n_fixed_cov {n_fixed} exceeds the {C_t.shape[1]} covariate columns')
        C_move = C_t[:, :C_t.shape[1] - n_fixed] if C_t.shape[1] > n_fixed else None
        C_fix = C_t[:, C_t.shape[1] - n_fixed:]                        # stays with the genotypes
    xy = x_t.new_empty((V, nperm))
    xx = x_t.new_empty((V, nperm))
    yy = x_t.new_empty(nperm)
    for s0 in range(0, nperm, chunk):
        ix = permutation_ix_t[s0:s0 + chunk]
        K = ix.shape[0]
        sw = sqrt_w_t[ix]                                              # [K, N]
        if mask_t is not None:
            sw = sw * mask_t[s0:s0 + K]
        cols = []
        if intercept:
            cols.append(sw.unsqueeze(2))
        if C_move is not None:
            cols.append(sw.unsqueeze(2) * C_move[ix])                  # [K, N, c] moves with the record
        if C_fix is not None:
            cols.append(sw.unsqueeze(2) * C_fix.unsqueeze(0))          # [K, N, g] stays in place
        design = torch.cat(cols, 2)                                    # [K, N, p]
        Q, _ = torch.linalg.qr(design)                                 # [K, N, p]
        ys = y_star[ix]                                                # [K, N]
        if mask_t is not None:
            ys = ys * mask_t[s0:s0 + K]
        if flip_t is not None:
            ys = ys * flip_t[s0:s0 + K]                                # swap before projecting

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
                                      permutation_ix_t, dof=None, perm_scheme=DEFAULT_PERM_SCHEME,
                                      fitted=False, flip_t=None,
                                      min_allelic_donors=MIN_ALLELIC_DONORS, return_info=False):
    """
    Compute nominal and permutation statistics for hapmixQTL.

    ``dof`` is the t-reference used to map the known-variance statistic to a
    correlation scale; the mapping is monotonic, so the empirical p-value does
    not depend on it, but the nominal p-value the caller derives from
    r_nominal does. Default: the smaller of the two channels' residualizer
    dof, which with per-channel covariates is the total channel's (the
    larger design). map_cis passes its own N - 2 - n_cov so both agree.

    With ``fitted=True`` the mapping constant is no longer the lead's t
    reference: that is the Welch-Satterthwaite dof of its combination
    (_satterthwaite_dof), returned with ``return_info=True`` as a sixth item,
    a dict with ``dof_nominal`` (None when not fitted; the caller keeps
    ``dof``) and ``allelic_admitted``. The allelic admission floor
    (``min_allelic_donors``, MIN_ALLELIC_DONORS) is applied to the observed
    scan and to every permutation alike: the informative allelic donors are
    the same set under every permutation, because each donor's weight moves
    with its record, so the gene-level null is built from the statistic that
    was observed.

    Meier's correction (_meier_factor) is part of that
    statistic in default mode: every combined tstat2, observed and permuted,
    is divided by M computed from the channel weights refitted for that
    variant and that permutation (_combined_tstat2). The observed scan and
    every permutation are therefore ONE statistic, the records null keeps its
    exchangeability, the empirical p remains the rank of the observed maximum
    among permuted maxima of the same statistic, and the Beta approximation
    is fitted to those permuted maxima, so both stay valid; a correction
    applied to the observed scan alone would bias pval_perm. Because M
    varies per variant and per permutation it is NOT a monotone per-gene map
    and can re-rank the lead against the uncorrected scan.

    The permutation null. ``perm_scheme='records'`` permutes donor records:
    for each permutation every donor receives another donor's whitened
    phenotype value, weight and covariate row together, the genotypes stay in
    place, and the statistic is recomputed with the permuted weights and
    design (the denominator xx changes per permutation: a second matrix
    product on the allelic channel, a chunked batched re-residualization on
    the total channel; _record_permutation_channel). Relabeling the donors
    shows this is exactly the distribution of the statistic under a
    permutation of the genotype columns, the null FastQTL and tensorQTL use,
    with per-donor weights carried along.

    ``perm_scheme='records_signflip'`` (the default since 2026-09-25) does the
    same and, in addition, swaps each permuted record's haplotype labels L and
    R with probability one half: ``flip_t`` [nperm, N] of +-1 multiplies the
    allelic log ratio placed at each position, and the caller must supply it
    (map_cis draws it right after the permutation indices, from the same
    generator, so the indices and therefore the total channel are identical to
    'records'). The labels L and R are arbitrary phase order, so under H0 a
    record's allelic ratio is as likely to carry either sign, and the swap is a
    symmetry of the null the record permutation alone does not use. Its effect:
    the allelic numerator's permutation distribution becomes symmetric about
    zero for every gene and every phase pattern. Under 'records' alone the
    through-origin allelic slope has a permutation mean equal to the gene's
    net allelic imbalance times the lopsidedness of the variant's phase
    (sum of s over the heterozygotes); on 46 BrainVar genes at a fixed variant
    that shifted 13 genes' permuted slopes by up to 0.21 standard errors,
    matching the algebra at Pearson r = 0.90, while the phase orientation was
    itself balanced (628 ALT-on-L against 642 ALT-on-R heterozygotes) and the
    genes' net imbalances were those of chance (z sd 0.98). Pooled rejection
    rates were unchanged (0.0690 against 0.0692 at 0.05). The swap changes the
    null only: the observed slope keeps whatever chance offset its own records
    carry, which with arbitrary phase labels is exchangeable with the swapped
    draws. The total channel is not flipped. Measured 2026-09-25,
    brainvar_hapmix_deploy/nominal_p_null_instrument_20260925 and
    lead_signal_share_corrected_20260925.

    ``perm_scheme='residuals'`` is the earlier scheme (retained, unflipped),
    Freedman-Lane in whitened space: the leverage-standardized whitened
    residuals are permuted
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
    deprecated_models/estimator_ablation_tiers_20260917/permutation_scheme_experiment.py and
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
    # Per-channel degrees of freedom for the fitted residual scale, counting
    # INFORMATIVE donors (_channel_dof, the rule _wls_regression uses), and
    # the allelic admission floor. Both are per gene: a permutation moves
    # weights with their records and so changes neither.
    dof_a = _channel_dof(residualizer_a)
    dof_t = _channel_dof(residualizer_t)
    admitted = _allelic_admitted(residualizer_a, residualizer_t, min_allelic_donors, fitted)
    _fit = dict(fitted=fitted, dof_a=dof_a, dof_t=dof_t, allelic=admitted)

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

    # The statistic, the combined slope and its t reference from ONE call.
    # The slope was previously a second, hand-inlined known-variance copy,
    # which silently disagreed with map_nominal the moment a fitted residual
    # scale was allowed; a test caught it.
    tstat2_nom, slope_all, dof_all = _combined_tstat2(
        xy_a_nom, xx_a, yy_a_nom, xy_t_nom, xx_t, yy_t_nom, dof,
        return_slope=True, return_dof=True, **_fit)

    tstat2_nom_clean = tstat2_nom.clone()
    tstat2_nom_clean[torch.isnan(tstat2_nom_clean)] = -1
    best_ix = tstat2_nom_clean.argmax()
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
    if perm_scheme in ('records', 'records_signflip'):
        if perm_scheme == 'records_signflip':
            if flip_t is None:
                raise ValueError(
                    "perm_scheme='records_signflip' needs flip_t, the [nperm, N] "
                    "+-1 haplotype-label swaps; map_cis draws them after the "
                    "permutation indices. Pass perm_scheme='records' for the "
                    "unswapped record permutation.")
        else:
            flip_t = None
        xy_a_perm, xx_a_perm, yy_a_perm = _record_permutation_channel(
            sign_t, a_t, sqrt_wa_t, residualizer_a, permutation_ix_t, flip_t=flip_t)
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

    out = (r_nominal, std_ratio, best_ix, max_r2_perm, genotypes_t[best_ix])
    if not return_info:
        return out
    # the lead's own t reference (None without a fitted scale, as in
    # calculate_hapmixqtl_nominal; the caller then keeps `dof`)
    info = dict(dof_nominal=float(dof_all[best_ix]) if fitted else None,
                allelic_admitted=admitted)
    return out + (info,)


def _combined_tstat2(xy_a, xx_a, yy_a, xy_t, xx_t, yy_t, dof,
                     fitted=False, dof_a=None, dof_t=None,
                     return_slope=False, allelic=True, return_dof=False):
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

    ``allelic=False`` (fitted form only) leaves the allelic channel out of the
    combination: the gene is below the allelic admission floor
    (MIN_ALLELIC_DONORS). Returns ``tstat2``, then the combined slope if
    ``return_slope``, then if ``return_dof`` the combination's
    Welch-Satterthwaite dof (_satterthwaite_dof), which exists only in the
    fitted form and is None otherwise. The fitted form divides ``tstat2`` by
    Meier's factor (_meier_factor), under permutation recomputed from each
    permutation's refitted weights, so the permuted statistic is the
    observed one (calculate_hapmixqtl_permutations).

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
        if not allelic:
            # below the allelic admission floor: the total channel alone
            inv_var_a = torch.zeros_like(inv_var_a)
            xy_a_eff = torch.zeros_like(xy_a_eff)

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
    if fitted:
        # Meier's inflation of the combined variance for weights estimated
        # from the same residuals (_meier_factor), from the scales refit
        # above; exactly 1 wherever fewer than two channels carry weight
        meier = _meier_factor(inv_var_a, inv_var_t, dof_a, dof_t)
        tstat2 = tstat2 / meier.to(tstat2.dtype)
    out = (tstat2,)
    if return_slope:
        out += (slope_comb,)
    if return_dof:
        # inv_var_k is xx_k / s2_k = 1/se_k^2, the weight of the combination
        out += (_satterthwaite_dof(inv_var_a, inv_var_t, dof_a, dof_t) if fitted else None,)
    return out if len(out) > 1 else tstat2


def _lead_influence(g_lead, s_lead, a_t, t_t, sqrt_wa_t, sqrt_wt_t, residualizer_a,
                    residualizer_t, min_allelic_donors):
    """Leave-one-donor-out at a fixed lead in default mode, vectorized like
    the record permutations: resample k is the identity order with donor k's
    weight zeroed in both channels, so one _record_permutation_channel call
    per channel gives every refit's dot products. Under tau_mode='zero' an
    exclusion changes no other donor's weight, so this is the exact refit
    (map_cis with that donor masked out). The per-channel dof, the
    sparse-channel rule (_channel_weights) and the allelic admission floor
    (_allelic_admitted) depend only on the channels the donor was informative
    in, so _combined_tstat2 is applied once per such class. A donor with
    leverage 1 in a channel's null design (the only donor of a covariate
    level) leaves that design rank-deficient when excluded and is skipped.

    Returns (index of the donor whose exclusion moves the combined |t|
    furthest toward zero, that |t|, its Welch-Satterthwaite dof), or
    (None, nan, nan) when no donor can be evaluated.
    """
    N = sqrt_wa_t.shape[0]
    on_a, on_t = sqrt_wa_t != 0, sqrt_wt_t != 0
    donors = (on_a | on_t).nonzero().flatten()
    K = donors.numel()
    order = torch.arange(N, device=a_t.device).expand(K, N)
    mask = torch.ones((K, N), dtype=a_t.dtype, device=a_t.device)
    mask[torch.arange(K, device=a_t.device), donors] = 0
    xy_a, xx_a, yy_a = _record_permutation_channel(s_lead, a_t, sqrt_wa_t, residualizer_a,
                                                   order, mask_t=mask)
    xy_t, xx_t, yy_t = _record_permutation_channel(g_lead / 2, t_t, sqrt_wt_t, residualizer_t,
                                                   order, mask_t=mask)
    evaluable = torch.ones(K, dtype=torch.bool, device=a_t.device)
    for res in (residualizer_a, residualizer_t):
        if res.Q_t.shape[1]:
            leverage = (res.Q_t * res.Q_t).sum(1)[donors]
            evaluable &= leverage < 1 - N * torch.finfo(res.Q_t.dtype).eps
    n_a, n_t = int(on_a.sum()), int(on_t.sum())
    best = (None, float('nan'), float('nan'))
    for in_a in (True, False):
        for in_t in (True, False):
            cls = ((on_a[donors] == in_a) & (on_t[donors] == in_t) & evaluable).nonzero().flatten()
            if not cls.numel():
                continue
            # the refit's channels: the sparse-channel rule switches one off
            # below its design's columns plus two, leaving no weight and dof < 1
            na, nt = n_a - in_a, n_t - in_t
            live_a = na >= _min_informative(residualizer_a.C_t, intercept=residualizer_a.intercept)
            live_t = nt >= _min_informative(residualizer_t.C_t, intercept=residualizer_t.intercept)
            dof_a = na - 1 - residualizer_a.Q_t.shape[1] if live_a else -1
            dof_t = nt - 1 - residualizer_t.Q_t.shape[1] if live_t else -1
            admitted = dof_a >= 1 if dof_t < 1 else (na if live_a else 0) >= min_allelic_donors
            ch = [xy_a[:, cls], xx_a[:, cls], yy_a[cls], xy_t[:, cls], xx_t[:, cls], yy_t[cls]]
            if not live_a:
                ch[:3] = [torch.zeros_like(x) for x in ch[:3]]
            if not live_t:
                ch[3:] = [torch.zeros_like(x) for x in ch[3:]]
            tstat2, dof_nom = _combined_tstat2(*ch, None, fitted=True, dof_a=max(dof_a, 1),
                                               dof_t=max(dof_t, 1), allelic=admitted,
                                               return_dof=True)
            t_abs = torch.sqrt(tstat2[0].clamp(min=0))
            j = int(t_abs.argmin())
            if best[0] is None or float(t_abs[j]) < best[1]:
                best = (int(donors[cls[j]]), float(t_abs[j]), float(dof_nom[0, j]))
    return best


def cis_trans_diagnostic(slope_a, se_a, slope_t, se_t, dof, dof_a=None, dof_t=None):
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

    ``dof`` is the shared t reference of the known-variance and HC1 standard
    errors. In default mode the caller passes each channel's residual dof as
    ``dof_a``/``dof_t`` instead, and z is referred to the Welch-Satterthwaite
    dof of the difference, (se_a^2 + se_t^2)^2 / (se_a^4/dof_a + se_t^4/dof_t),
    the construction of _satterthwaite_dof with coefficients +1 and -1.

    Returns:
        alpha_cis:      slope_a / slope_t (NaN when slope_t ~ 0 or a channel
                        has no finite SE)
        pval_cis_trans: two-sided p; NaN when a channel is unavailable, e.g.
                        no phase -> no ASE channel -> nothing to compare
    """
    slope_a = np.atleast_1d(np.asarray(slope_a, float)); se_a = np.atleast_1d(np.asarray(se_a, float))
    slope_t = np.atleast_1d(np.asarray(slope_t, float)); se_t = np.atleast_1d(np.asarray(se_t, float))
    ok = np.isfinite(se_a) & (se_a > 0) & np.isfinite(se_t) & (se_t > 0)
    z = np.full(slope_a.shape, np.nan)
    z[ok] = (slope_a[ok] - slope_t[ok]) / np.sqrt(se_a[ok] ** 2 + se_t[ok] ** 2)
    if dof_a is not None and dof_t is not None:
        va, vt = se_a[ok] ** 2, se_t[ok] ** 2
        dof = (va + vt) ** 2 / (va ** 2 / dof_a + vt ** 2 / dof_t)
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
                total_variance_model='additive', genotype_covariates_df=None,
                min_allelic_donors=MIN_ALLELIC_DONORS):
    """
    hapmixQTL cis-QTL mapping: nominal associations for all variant-phenotype pairs.

    Inputs are already prepared. Use prepare_default_inputs for half-read
    split (Gibbs-informed ASE weights, unit total weights). Explicit caller
    phenotypes and variances are preserved; Vt_df remains required here.

    ``genotype_covariates_df`` (the genotype PCs) enters the design exactly as
    ``covariates_df`` does; the split exists for map_cis's permutation, where
    genotype-tied covariates stay with the genotypes. Pass it here too so the
    nominal and permutation runs use the same design.

    t references (default mode, se_mode='fitted'). Each p-value is referred
    to the degrees of freedom of the scale its standard error was fitted
    with: ``pval_a`` to ``dof_a`` = n_a - 1 - n_cov_a and ``pval_t`` to
    ``dof_t`` = n_t - 2 - n_cov, counting informative donors (_residual_dof;
    NaN, and so a NaN p, for a channel that is switched off), and
    ``pval_nominal`` to ``dof_nominal``, the per-pair Welch-Satterthwaite dof
    of the combination (_satterthwaite_dof; NaN where neither channel carries
    weight, e.g. a monomorphic variant). ``pval_cis_trans`` is referred to the
    Welch-Satterthwaite dof of the difference (cis_trans_diagnostic). The
    allelic channel enters the combination only for genes with at least
    ``min_allelic_donors`` informative allelic donors (MIN_ALLELIC_DONORS,
    mixQTL's own cutoff; waived when the gene has no total channel, as in an
    allelic-only run); ``allelic_admitted`` records that gene-level count
    test, and below the floor the combined slope, SE and p are the total
    channel's while ``pval_a`` is still reported. Where both channels carry
    weight, ``slope_se`` is the combined SE times sqrt(M), M Meier's
    first-order factor for channel weights estimated from the same residuals
    (_meier_factor), and ``pval_nominal`` is the correspondingly smaller t on
    the unchanged ``dof_nominal``. The known-variance and HC1 standard
    errors keep the single reference N - 2 - max(n_cov, n_cov_a), which the
    dof columns then carry, no floor and no correction.

    Writes per-chromosome parquet files in the format:
        <output_dir>/<prefix>.hapmixqtl_pairs.<chr>.parquet

    Args:
        genotype_df:      genotypes [variants x samples]
        variant_df:       variant positions (chrom, pos)
        A_df:             allelic contrast [phenotypes x samples]
        T_df:             log total expression [phenotypes x samples]
        Va_df:            inferential variance for a [phenotypes x samples]
        Vt_df:            total working variance [phenotypes x samples];
                          supply ones for half-read split
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
        tau_mode:         'zero' (default), no additive floor. Together with
                          se_mode='fitted', working variances set relative
                          weights and each channel estimates a residual scale.
                          'estimate' is a deprecated compatibility path in
                          fitted_variance.py, not the production default.
        se_mode:          'fitted' (default), estimated-dispersion SE;
                          'robust', HC1 sandwich for nominal mapping only;
                          or deprecated 'model', known-variance 1/sqrt(xx).
                          The historical invalid zero-tau/known-variance
                          pairing is not the zero-tau/fitted default. See
                          docs/ase_validation.md for those dated results.
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
    _validate_inputs(genotype_df, A_df, T_df, Va_df, Vt_df, xL_df, xR_df)

    logger.write('hapmixQTL mapping: nominal associations for all variant-phenotype pairs')
    logger.write(f'  * {N} samples')
    logger.write(f'  * {A_df.shape[0]} phenotypes')

    robust = se_mode == 'robust'
    fitted = se_mode == 'fitted'

    covariates_df, _ = _combine_covariates(covariates_df, genotype_covariates_df, samples, logger)
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
    # One t reference, the larger design's, for the known-variance and HC1
    # standard errors. Default mode refers each p-value to its own channel's
    # dof instead (see the docstring).
    dof = N - 2 - max(n_cov, n_cov_a)
    library_factor_t = _library_factor_tensor(library_factor, samples, device)
    _check_variance_model(variance_model, tau_mode, library_factor_t, variance_prior)
    logger.write(f'  * variance model: {variance_model}' + (' with empirical-Bayes prior' if variance_prior is not None else ''))
    if fitted:
        logger.write(f'  * allelic channel enters the combination at >= {min_allelic_donors} informative donors')

    genotype_ix = np.array([genotype_df.columns.tolist().index(i) for i in samples])
    genotype_ix_t = torch.from_numpy(genotype_ix).to(device)

    # Use T_df as phenotype for InputGeneratorCis (less likely to be constant)
    igc = genotypeio.InputGeneratorCis(
        genotype_df, variant_df, T_df, phenotype_pos_df, window=window,
    )
    pheno_ix = {pid: i for i, pid in enumerate(A_df.index)}

    start_time = time.time()
    k = 0
    n_below_floor = 0
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
        chr_res['dof_nominal'] = np.empty(n, dtype=np.float64)
        chr_res['dof_a'] = np.empty(n, dtype=np.float64)
        chr_res['dof_t'] = np.empty(n, dtype=np.float64)
        chr_res['allelic_admitted'] = np.empty(n, dtype=bool)

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
                min_allelic_donors=min_allelic_donors, return_info=True,
            )
            (tstat, slope, slope_se, slope_a, se_a,
             slope_tc, se_tc) = [r.cpu().numpy() for r in res[:7]]
            ref = res[7]
            if fitted:
                dof_nominal = ref['dof_nominal'].cpu().numpy()
                dof_a, dof_t = ref['dof_a'], ref['dof_t']
                ct_dof = dict(dof_a=dof_a, dof_t=dof_t)
                n_below_floor += (not ref['allelic_admitted']
                                  and bool((residualizer_a.sqrt_w_t != 0).any()))
            else:
                dof_nominal, dof_a, dof_t = np.float64(dof), dof, dof
                ct_dof = {}

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
                slope_a, se_a, slope_tc, se_tc, dof, **ct_dof)[1]
            chr_res['dof_nominal'][start:start + nv] = dof_nominal
            chr_res['dof_a'][start:start + nv] = dof_a
            chr_res['dof_t'][start:start + nv] = dof_t
            chr_res['allelic_admitted'][start:start + nv] = ref['allelic_admitted']
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
        # each statistic on its own reference (the dof columns; see docstring)
        chr_res_df.loc[m, 'pval_nominal'] = get_t_pval(
            chr_res_df.loc[m, 'pval_nominal'], chr_res_df.loc[m, 'dof_nominal']
        )
        chr_res_df.loc[m, 'pval_a'] = get_t_pval(
            chr_res_df.loc[m, 'pval_a'], chr_res_df.loc[m, 'dof_a']
        )
        chr_res_df.loc[m, 'pval_t'] = get_t_pval(
            chr_res_df.loc[m, 'pval_t'], chr_res_df.loc[m, 'dof_t']
        )
        print('    * writing output')
        chr_res_df.to_parquet(
            os.path.join(output_dir, f'{prefix}.hapmixqtl_pairs.{chrom}.parquet')
        )

    if n_below_floor:
        logger.write(f'  * {n_below_floor} phenotypes had informative allelic donors but fewer than '
                     f'{min_allelic_donors}: allelic channel left out of the combined statistic '
                     f'(allelic_admitted False; pval_a still reported)')
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
            perm_scheme=DEFAULT_PERM_SCHEME, keep_a_df=None, keep_t_df=None,
            genotype_covariates_df=None, min_allelic_donors=MIN_ALLELIC_DONORS):
    """
    hapmixQTL cis-QTL mapping with permutation-based empirical p-values.

    Current association defaults are tau_mode='zero', se_mode='fitted' and
    perm_scheme='records_signflip'. Prepare half-read split inputs with
    prepare_default_inputs; explicit caller inputs and weights are preserved.
    Both channels fit residual scales. tau_refit has no effect at zero tau.

    The variance-model/tau-refit descriptions in the next two paragraphs
    apply to deprecated tau_mode='estimate' compatibility paths only.

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
    the total channel's set, or its own DataFrame.

    t references. In default mode (se_mode='fitted') the lead's
    ``pval_nominal`` is referred to ``dof_nominal``, the Welch-Satterthwaite
    dof of its combination, exactly as map_nominal refers that pair; the
    allelic channel enters the scanned statistic only for genes with at least
    ``min_allelic_donors`` informative allelic donors (``allelic_admitted``;
    MIN_ALLELIC_DONORS, waived without a total channel), in the observed scan
    and in every permutation alike. The lead is the variant with the largest
    combined |t|; with a per-variant dof its ``pval_nominal`` need not be the
    gene's smallest pval_nominal in map_nominal. Its per-channel slopes and
    SEs, ``alpha_cis`` and ``pval_cis_trans`` are on the scan's fitted scale,
    as in map_nominal. A gene in which neither channel carries weight has
    NaN ``pval_nominal``, ``pval_perm`` and ``dof_nominal``: it was not
    tested. The constant dof = N - 2 - max(n_cov, n_cov_a) still maps the
    scanned statistic to the correlation scale shared by the observed lead
    and the permutations, and seeds the Beta approximation's fit of
    ``true_df``; being one monotone map per gene it leaves the lead,
    ``pval_perm`` and ``pval_beta`` exactly as the statistic determines them.
    Under se_mode='model' it is also the t reference of ``pval_nominal``, and
    ``dof_nominal`` reports it. In default mode the scanned statistic carries
    Meier's correction for estimated channel weights (_meier_factor) in the
    observed scan and in every permutation alike, so ``pval_perm`` and
    ``pval_beta`` are built from the corrected statistic
    (calculate_hapmixqtl_permutations); the lead's ``slope_se`` is the
    combined SE times sqrt(M), M the factor at the lead (1 where it does not
    apply), and its ``pval_nominal`` is the corrected t on ``dof_nominal``,
    as map_nominal reports that pair.

    For each phenotype, finds the best cis variant and computes empirical
    p-values from ``nperm`` permutations shared across genes. The default
    ``perm_scheme='records_signflip'`` permutes donor records (each donor's
    whitened phenotype value, weight and covariate row move together against
    fixed genotypes) and swaps each permuted record's haplotype labels with
    probability one half, which negates its allelic log ratio; the signs are
    drawn right after the permutation indices from the same seeded generator,
    so 'records' (no swap) sees the same indices and the same total channel.
    'residuals' is the pre-2026-09-17 Freedman-Lane scheme. See
    calculate_hapmixqtl_permutations. ``se_mode`` must be 'model' or
    'fitted': the HC1 sandwich has no permutation counterpart here; use
    map_nominal for robust standard errors.

    Returns:
        DataFrame with one row per phenotype, analogous to cis.map_cis output.
    
    The lead variant's per-channel slopes (``slope_a``/``slope_t`` with SEs),
    ``alpha_cis = slope_a / slope_t`` and ``pval_cis_trans`` (Wald test of
    ``slope_a = slope_t``, see ``cis_trans_diagnostic``) are reported per gene.
    A small ``pval_cis_trans`` means the two channels disagree, so the
    combined slope should not be read as a cis log aFC (a trans component,
    mapping bias or phasing error attenuate it; docs sec 7c). It is a
    diagnostic column, not a filter.

    In default mode ``loo_donor`` and ``loo_pval_nominal`` are a
    leave-one-donor-out check at the lead: the donor whose exclusion from
    both channels moves the lead's combined |t| furthest toward zero, and the
    lead's nominal p without it. Diagnostic only: the lead is held fixed
    (excluding the donor can move it) and pval_perm is not recomputed. All
    refits run as one batch through the permutation machinery
    (_lead_influence; about 8 ms per gene at 92 donors). A donor that alone
    identifies a covariate level is not evaluated.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if logger is None:
        logger = SimpleLogger()

    samples = A_df.columns
    N = len(samples)
    _validate_inputs(genotype_df, A_df, T_df, Va_df, Vt_df, xL_df, xR_df)

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

    covariates_df, n_genotype_cov = _combine_covariates(
        covariates_df, genotype_covariates_df, samples, logger)
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

    _assert_keep_frames(keep_a_df, keep_t_df, A_df)
    _log_keep_frames(keep_a_df, keep_t_df, logger)

    logger.write(f'  * {variant_df.shape[0]} variants')
    if maf_threshold > 0:
        logger.write(f'  * applying in-sample {maf_threshold} MAF filter')
    logger.write(f'  * cis-window: ±{window:,}')

    ase_covariates_t, n_cov_a = _resolve_ase_covariates(
        ase_covariates_df, covariates_df, samples, device, logger)
    # the larger design's dof: the correlation-scale map of the scan and the
    # Beta approximation's starting true_df in every mode, and the lead's t
    # reference only under se_mode='model' (see the docstring)
    dof = N - 2 - max(n_cov, n_cov_a)
    library_factor_t = _library_factor_tensor(library_factor, samples, device)
    if perm_scheme not in PERM_SCHEMES:
        raise ValueError(f'perm_scheme must be one of {PERM_SCHEMES}, got {perm_scheme!r}')
    _check_variance_model(variance_model, tau_mode, library_factor_t, variance_prior)
    logger.write(f'  * variance model: {variance_model}' + (' with empirical-Bayes prior' if variance_prior is not None else ''))
    if se_mode == 'fitted':
        logger.write(f'  * allelic channel enters the combination at >= {min_allelic_donors} informative donors')

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
    # haplotype-label swaps, drawn AFTER the indices from the same stream so the
    # indices (and the total channel) are identical to perm_scheme='records'
    flip_t = None
    if perm_scheme == 'records_signflip':
        logger.write('  * permutation null: donor records, haplotype labels swapped at random')
        flip_t = torch.tensor(
            np.random.randint(0, 2, size=(nperm, n_samples)) * 2 - 1,
            dtype=torch.float32).to(device)
    else:
        logger.write(f'  * permutation null: {perm_scheme}')

    igc = genotypeio.InputGeneratorCis(
        genotype_df, variant_df, T_df, phenotype_pos_df, window=window,
    )
    if igc.n_phenotypes == 0:
        raise ValueError('No valid phenotypes found.')
    pheno_ix = {pid: i for i, pid in enumerate(A_df.index)}

    res_df = []
    n_below_floor = 0
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
        # genotype-tied covariates (the last n_genotype_cov columns) stay with
        # the genotypes in the record permutation; the rest move with the record
        residualizer_tc.n_fixed_cov = n_genotype_cov
        if isinstance(ase_covariates_t, str) and ase_covariates_t == SAME_COVARIATES:
            residualizer_a.n_fixed_cov = n_genotype_cov

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
            fitted=(se_mode == 'fitted'), flip_t=flip_t,
            min_allelic_donors=min_allelic_donors, return_info=True,
        )
        lead_ref = res[5]
        r_nominal, std_ratio, var_ix, r2_perm, g = [i.cpu().numpy() for i in res[:5]]
        best_local = int(var_ix)
        var_ix = genotype_range[var_ix]

        # the lead on the scan scale (fitted SEs and the floor in default
        # mode): per-channel slopes for the cis/trans diagnostic
        g_lead = genotypes_t[best_local:best_local + 1]
        s_lead = sign_t[best_local:best_local + 1]
        *lead_stat, lead_info = calculate_hapmixqtl_nominal(
            g_lead, s_lead, a_t, t_t, sqrt_wa_t, sqrt_wt_t, residualizer_a, residualizer_tc,
            fitted=(se_mode == 'fitted'), min_allelic_donors=min_allelic_donors,
            return_info=True)
        ct_dof = (dict(dof_a=lead_info['dof_a'], dof_t=lead_info['dof_t'])
                  if se_mode == 'fitted' else {})
        tau_used = dict(tau_a=tau_info['tau_a'], tau_t=tau_info['tau_t'], refit=False,
                        c_a=tau_info['c_a'])
        if tau_refit and tau_mode == 'estimate':
            # tau (and c) with the lead's predictor in the model; the
            # residualizers still project out the null design only. The
            # refit (deprecated tau_mode='estimate' only) is known-variance,
            # so it has no per-channel dof and the allelic floor does not
            # reach it; allelic_admitted still describes the scan.
            wa_r, wt_r, res_a_r, res_t_r, info_r = _prepare_channels(
                a_t, t_t, va_t, vt_t, covariates_t, tau_mode, device,
                ase_covariates_t=ase_covariates_t, return_info=True,
                tau_extra_a_t=s_lead.t(), tau_extra_t_t=(g_lead / 2).t(),
                variance_model=variance_model, library_factor_t=library_factor_t, prior=prior_g,
                keep_a_t=keep_a_t, keep_t_t=keep_t_t)
            lead_stat = calculate_hapmixqtl_nominal(
                g_lead, s_lead, a_t, t_t, wa_r, wt_r, res_a_r, res_t_r)
            ct_dof = {}
            tau_used = dict(tau_a=info_r['tau_a'], tau_t=info_r['tau_t'],
                            refit=bool(info_r['refit_a'] or info_r['refit_t']),
                            c_a=info_r['c_a'])
        (lead_tstat, lead_slope, lead_slope_se, lead_a, lead_a_se, lead_t, lead_t_se) = [
            float(x.cpu().numpy()[0]) for x in lead_stat]
        alpha_cis, pval_cis_trans = cis_trans_diagnostic(
            lead_a, lead_a_se, lead_t, lead_t_se, dof, **ct_dof)

        # Leave-one-donor-out at the fixed lead (default mode; _lead_influence).
        # A diagnostic, not a filter: the lead is held fixed and pval_perm is
        # not recomputed. One donor's allelic record carried CALM2's
        # gene-level call (2026-09-25).
        loo_donor, loo_pval = None, np.nan
        if se_mode == 'fitted' and tau_mode == 'zero' and np.isfinite(lead_tstat):
            i, loo_t, loo_dof = _lead_influence(
                g_lead, s_lead, a_t, t_t, sqrt_wa_t, sqrt_wt_t, residualizer_a,
                residualizer_tc, min_allelic_donors)
            if i is not None:
                loo_donor = samples[i]
                if np.isfinite(loo_dof):
                    loo_pval = float(get_t_pval(loo_t, loo_dof))

        variant_id = variant_df.index[var_ix]
        start_distance = variant_df['pos'].values[var_ix] - igc.phenotype_start[phenotype_id]
        end_distance = variant_df['pos'].values[var_ix] - igc.phenotype_end[phenotype_id]

        # empirical p and the beta approximation stay on the scan scale
        r2_nominal = r_nominal * r_nominal
        pval_perm = (np.sum(r2_perm >= r2_nominal) + 1) / (nperm + 1)

        if tau_used['refit']:
            # the refit's statistic is known-variance: the shared reference
            dof_nominal = dof
            slope, slope_se = lead_slope, lead_slope_se
            pval_nominal = float(get_t_pval(lead_tstat, dof)) if np.isfinite(lead_tstat) else np.nan
        else:
            # the lead's t, recovered from the correlation scale, referred to
            # the lead's own reference (dof itself unless se_mode='fitted');
            # NaN when neither channel carries weight, so the gene was not
            # tested and its pval_perm (every statistic 0) is not a p-value.
            # In default mode tstat2 already carries the Meier factor of the
            # scan, so slope_se below is the combined SE times sqrt(M).
            dof_nominal = dof if lead_ref['dof_nominal'] is None else lead_ref['dof_nominal']
            slope = r_nominal * std_ratio
            tstat2 = dof * r2_nominal / (1 - r2_nominal) if r2_nominal < 1 else np.inf
            slope_se = np.abs(slope) / np.sqrt(tstat2) if tstat2 > 0 else np.inf
            pval_nominal = get_t_pval(np.sqrt(tstat2), dof_nominal)
            if not np.isfinite(dof_nominal):
                pval_perm = np.nan
        n_below_floor += (not lead_ref['allelic_admitted']
                          and bool((residualizer_a.sqrt_w_t != 0).any()))

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
            ('n_genotype_covariates', n_genotype_cov),
            ('dof_nominal', float(dof_nominal)),
            ('allelic_admitted', bool(lead_ref['allelic_admitted'])),
            ('loo_donor', loo_donor),
            ('loo_pval_nominal', loo_pval),
        ]), name=phenotype_id)

        if beta_approx and np.isfinite(pval_perm):
            try:
                res_s[['pval_beta', 'beta_shape1', 'beta_shape2',
                       'true_df', 'pval_true_df']] = \
                    calculate_beta_approx_pval(r2_perm, r2_nominal, dof)
            except Exception:
                pass

        res_df.append(res_s)

    res_df = pd.concat(res_df, axis=1, sort=False).T
    res_df.index.name = 'phenotype_id'
    if n_below_floor:
        logger.write(f'  * {n_below_floor} of {len(res_df)} phenotypes had informative allelic donors but '
                     f'fewer than {min_allelic_donors}: allelic channel left out of the scan and '
                     f'every permutation (allelic_admitted False)')
    n_untested = int(res_df['pval_perm'].isna().sum())
    if n_untested:
        logger.write(f'  * WARNING: {n_untested} of {len(res_df)} phenotypes have no channel carrying '
                     f'weight and were not tested (pval_perm NaN)')
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

    NOT SUPPORTED IN DEFAULT MODE (2026-10-01). This is a legacy
    stacked-design path with no per-channel residual scale and no se_mode:
    by default it treats the working variances as the entire error variance,
    and estimate_residual_variance=True fits one scale shared by both
    channels, whose fitted scales differ by a median factor of five (the
    stacked arm of 2026-09-24 rejected at 0.128 at nominal 0.05). With
    tau_mode='zero' either is uncalibrated, so tau_mode='zero' is refused; the
    deprecated tau_mode='estimate' default remains for reproducing earlier
    results. Credible sets and PIPs have not been validated under any mode.

    Args mirror ``susie.map``; hapmixQTL-specific inputs (A/T/Va/Vt and the
    optional phase matrices xL/xR) match ``map_cis``.

    Returns:
        summary_df (if summary_only) or (summary_df, susie_res dict), analogous
        to ``susie.map``. The summary carries a ``tau_mode`` column and every
        ``susie_res`` entry a ``tau_mode`` key, so fine-mapping produced under
        the invalid ``'zero'`` setting (docs/ase_validation.md sec 7g) can be
        identified later; see ``fine_mapping_provenance``.
    """
    if tau_mode == 'zero':
        raise ValueError(
            "map_susie is not supported in default mode: with tau_mode='zero' its "
            'stacked design has no per-channel residual scale and is not calibrated.')
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
    under. This legacy classifier marks any recorded 'zero' as 'stale',
    missing tau_mode as 'unknown', and other recorded modes as 'ok'. It does
    not inspect the phenotype transform or certify half-read calibration.

    Historical calibration motivating this classifier: fine-mapping run under ``tau_mode='zero'`` is invalid, not merely
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
            '(docs/ase_validation.md sec 7g). map_susie now refuses that setting and is '
            'not supported in default mode, so there is no default-mode fine-mapping to redo it with.'))
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
    and ``library_factor`` as in map_cis.

    Not available in default mode: _joint_gls has no fitted residual scale,
    so with tau_mode='zero' its standard errors take the Gibbs variance and
    the unit total working variance as the entire error variance, the
    withdrawn pairing _warn_tau_zero describes."""
    if tau_mode == 'zero':
        raise ValueError(
            f'the {kind} second pass has known-variance standard errors only, which with '
            "tau_mode='zero' (default mode) are not calibrated; it is not supported in "
            'default mode. STR and multi-allelic rows can still enter map_cis as scan rows.')
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
    the K-1 split-biallelic rows of a site fitted jointly. Known-variance
    standard errors only, so not supported in default mode (_second_pass).

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
    Known-variance standard errors only, so not supported in default mode
    (_second_pass).

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


# ---------------------------------------------------------------------------
# Deprecated names, resolved from the quarantine.
#
# The fitted-variance estimators moved to tensorqtl/fitted_variance.py on
# 2026-09-23 (see that module's docstring for why they are inferior rather than
# merely old). They are resolved here, with a warning, so historical scripts and
# the quarantined tests keep working unchanged.
#
# PEP 562: this hook runs only for a name that is NOT already a module global,
# so it never shadows the live gates (_check_variance_model,
# _library_factor_tensor, _prior_tuple), and a default-mode run -- which asks
# for none of these -- never imports the quarantine at all.
_QUARANTINED_NAMES = frozenset({
    '_estimate_tau', '_estimate_tau_informative', '_estimate_c_tau',
    '_fit_c_tau_vectorized', '_raw_c_tau_and_cov', '_trend_prior',
    'estimate_variance_priors', 'estimate_library_factors',
    'VARIANCE_MODELS', 'PRIOR_METHODS',
})


def __getattr__(name):
    if name in _QUARANTINED_NAMES:
        import warnings
        warnings.warn(
            f'hapmixqtl.{name} is DEPRECATED and now lives in '
            f'tensorqtl/fitted_variance.py. It fits a variance function from a '
            f"gene's own residuals and then weights those residuals by the fit, "
            f'which neither shipped mode does; it is retained only to reproduce '
            f'results recorded before 2026-09-23 and will be removed. The '
            f'shipped modes are mixQTL mode (the NumPy port in '
            f'tensorqtl/mixqtl_replication.py, no draws) and default mode '
            f"(tau_mode='zero' + se_mode='fitted', i.e. Var(eps_i) = sigma^2 "
            f'v_i on the Gibbs variance).',
            DeprecationWarning, stacklevel=2)
        return getattr(_fitted_variance(), name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')

# Implementation status — 2026-09-16

This inventory distinguishes settled directions, current source, bounded
prototype evidence, and deferred extension work. The current gate is a
from-beginning weighted-OLS recommendation: empirical per-variant residual
variance plus Gibbs-informed precision, without assuming absolute known-variance
SEs. Exact preservation of unweighted OLS coefficients is not a constraint.
Measured on 2026-09-16: see `brainvar_hapmix_deploy/deprecated_models/estimator_ablation_20260916/REPORT.md`;
the fitted-scale rule coincides with the shipped estimator at a self-consistent
`tau` away from the `tau = 0` boundary and discards the Gibbs variance floor at it.

The current specification question is covariance in haplotype-derived
phenotypes and joint ASE/total effects. Technical Gibbs covariance is not
automatically total residual covariance; WLS with per-variant empirical
residual scale remains the working direction.

## Covariance walkthrough outcome (not adoption)

Fit ASE and total WLS separately, with per-variant empirical scales. For their
fixed coefficient maps `kA,kT`, the technical Gibbs contribution is
`cG = kA' M_AT kT`, distinct from total slope covariance c. Combine slopes
only through coherent `S=[[VA,c],[c,VT]]`:
`beta = 1'S^-1 b/(1'S^-1 1)` and `var(beta)=(1'S^-1 1)^-1`. `c=0` recovers old
inverse-variance weighting; this has no per-variant optimizer and remains
GPU-batchable.

Choose either a model-based total residual covariance with separate channel
scales, or a donor-paired residual sandwich (HC-style) that estimates all three
entries together. Do not combine Gibbs-only c with unrelated diagonal entries
without checking coherence. L/R covariance enters ASE phenotype construction
through Var(ASE), which is distinct from ASE/total covariance. The two-stage
combination is not generally identical to the audit's donor-row joint GLS.
For illustration only, equal 0.2 slope SEs with error correlation 0.8 give
combined SE 0.1897 versus 0.1414 under independence; this is algebra, not data.

Precision-weighted residual-scale models in Smyth's voom/limma tradition must
be distinguished from the custom local RNA-workflow catchSalmon divided-count
method; they are not interchangeable precedents.

## Recommended weighted architecture (not method adoption)

Fix a gene-level W, fit each variant by `beta = (X'WX)^-1 X'Wy`, then use its
empirical `s^2 = e'We/df` and coefficient covariance
`s^2 (X'WX)^-1`. Here s² is the total residual scale in weighted units, not
biological tau or an absolute known-variance SE. Uniform rescaling of W
cancels from coefficients and SEs; this uses relative precision and does not
automatically enforce an absolute Salmon measurement-variance floor. Any such
floor is a separate explicit decision. Validate the OLS limit, weight rescaling,
and a dense oracle. Paired channels require separate residual scales and their
covariance.

A previously proposed candidate is a voom-style prediction of total variance
with Gibbs uncertainty as a precision covariate. No evidence yet establishes it
preferable to adapting existing overdispersion estimates. A concrete working
family is stabilized `1/(m+c_gene)`, where c is a shape regularizer, not
automatically biology; pure `1/m` makes a stronger proportionality assumption.
Do not multiply a voom variance by Gibbs variance or use Fano blindly. The
proper residual-scale path extends `_wls_regression`/`_combined_tstat2` to
return per-variant `yy` and RSS, with the same computation under the matching
null; a new tau estimator is not a prerequisite.

**Clarification:** nothing establishes that the existing overdispersion
framework is wrong or must be replaced by a voom-style Gibbs-covariate trend.
Gibbs/Fano quantification-dispersion estimators remain legitimate candidate
inputs, and multiplicative fitted-residual-scale WLS is itself a dispersion
framework. Any adaptation must handle count-to-log2/ASE covariance, separately
fit residual dispersion, omit unconditional extra q, and preserve the aFC mean
through an allele-count scaling/offset contract.

[voom](https://genomebiology.biomedcentral.com/articles/10.1186/gb-2014-15-2-r29)
and [limma lmFit](https://raw.githubusercontent.com/bioc/limma/master/R/lmfit.R)
are the primary weighted-linear-model precedents. The 2024
[catchSalmon/edgeR count-QL article](https://academic.oup.com/nargab/article/6/4/lqae151/7874835)
is a distinct divided-count approach.

## Historical OLS-first alternative (pending validation)

This is not an adopted confirmatory default. Keep fixed OLS coefficients and
the usual per-variant RSS. From normalized, matched Gibbs draws, estimate M;
its use as repeated-sampling measurement covariance is a working assumption
requiring calibration. Assuming constant biological residual variance across
retained donors, retain the raw `V = s^2 a'a + a'Ma - [tr(RM)/df] a'a` and its
boundary, then use `V = max(0, s^2-tr(RM)/df)*a'a + a'Ma`. Thus it remains an
additive covariance model that changes SEs, not OLS coefficients, and does not
reject the broader M+tau family. Under `M=mI`, the raw correction cancels; the
clipped boundary can break cancellation, so no automatic SE reduction follows.

Classical OLS and HC3 are baselines. The direct Gibbs slope-sensitivity
`Var_b(a'y_b)=a'Ma` is conditional fixed-cohort quantification uncertainty,
not a total population SE and is not divided by B. Calibration must bridge
posterior M to repeated-sampling covariance and establish mean-zero error with
respect to genotype; it does not automatically repair bias. Do not add M
blindly or use naive Rubin pooling. The user said on 2026-09-17 that weighted OLS makes sense here ("in line with
what Smyth's group was proposing"); this older fixed-OLS alternative does not constrain it.
The existing tau target remains biological residual variance; its empirical
estimator can absorb technical or mean-model error.

The motivating precedent is [sleuth Supplementary Note 2, sections 5–6,
PDF pages 42–43](https://media.springernature.com/original/springer-static/esm/art:10.1038%2Fnmeth.4324/MediaObjects/41592_2017_BFnmeth4324_MOESM1_ESM.pdf), with [operational code](https://github.com/pachterlab/sleuth/blob/master/R/measurement_error.R#L608-L626): OLS coefficients with raw residual variance less mean bootstrap variance,
then shrinkage. The donor heterogeneity/leverage extension here is neither a
sleuth implementation nor a validated method.

| Area | Status | Authority / boundary |
|---|---|---|
| Log2 target, no automatic ASE intercept, GPU support, half-count 0.5, existing filters, local-linear `s` and `g/2` | Settled direction; current dirty source | [hapmixQTL source](/mnt/ssd/lalli/tensorqtl/.claude/worktrees/hapmix-runbook/tensorqtl/hapmixqtl.py). |
| Biological residual tau and separate ASE/total components | Existing source/model direction under reassessment | It is not the current accepted objective; assess weighted-linear per-variant residual estimation and Gibbs uncertainty without double counting. |
| Additive measurement-plus-residual weighting and leverage-corrected weighted residual-minus-measurement tau | Implemented in source | `_estimate_tau`, [hapmixQTL source](/mnt/ssd/lalli/tensorqtl/.claude/worktrees/hapmix-runbook/tensorqtl/hapmixqtl.py:520); fixed-gene GPU scan and lead refit exist. Per-variant tau is not implemented. |
| Full-M paired log2 GLS, no extra q, CPU/GPU oracle, phase-label swap | Completed bounded prototype | [audit implementation](/mnt/ssd/lalli/brainvar_hapmix_deploy/gibbs_influence_audit_20260915/analyze.py), [report](/mnt/ssd/lalli/brainvar_hapmix_deploy/gibbs_influence_audit_20260915/REPORT.md), and [validation receipt](/mnt/ssd/lalli/brainvar_hapmix_deploy/gibbs_influence_audit_20260915/validation.json) record max effect error 2.833e-13 and zero swap error. Audit null REML weights and depth offsets are audit-specific, not a production helper. |
| Counting interpretation | Bounded validated evidence | The closed counting fixture supports no unconditional extra q for its tested Salmon configuration. Production `count_noise=True` still adds q. |
| Historical calibration | Closed historical receipt | `deprecated_models/null_calibration_29b` exited 0 with 145 old-null statistics; it is not calibration of the changed model. |

## Deferred M+tau extension work (if later selected)

1. Port full-M, normalized-log2/no-q wiring with unit metadata, retained raw
   `YL/YR/YT` cache, and explicit support/availability semantics; retain `YT`
   for unpaired transcripts.
2. Batch the existing weighted moment across variants and propagate updated
   residual estimates into both observed and null-statistic SEs.
3. Add the offline haplotype length-aware TMM adapter and metadata under the
   existing normalization direction; workflow integration remains deferred.
4. Run targeted extension tests and calibration for the changed statistic.

## Detailed integration map (deferred specification only)

This maps proposed edits; it is neither implemented nor accepted new science.

1. Produce log2/no-q summaries; thread `Cat` into 2×2 joint whitening. Derive
   masks from coverage/support rather than `v > eps` alone, retain raw
   `YL/YR/YT`, and preserve `YT` under a fixed-factor, draw-wise offline
   normalization interface.
2. Extend the weighted RSS/leverage moment per variant with rank-one updates:
   `tau_raw_cv = [RSS_cv - sum_i (1-h_icv) w_ic m_ic] /
   sum_i (1-h_icv) w_ic`, where `w_ic = 1/max(m_ic,floor)` and `h_icv` is the
   leverage of weighted covariates plus the tested predictor. Retain supported
   `m=0` with a finite working weight; use raw m in the trace and clipping only
   for covariance. Fixed-coefficient observed and null statistic SEs use full
   M plus the variant residual estimate; both use the same `beta^2/V`
   statistic and recompute residual quantities per null replicate. The
   null-generator method remains unresolved; a parametric bootstrap is not
   mandated.
3. Update `cis_trans_diagnostic` to carry slope covariance or mark that path
   unsupported for full M. The haplotype length-aware TMM adapter stays offline
   and workflow integration deferred. Full-M SuSiE compatibility/guard work is
   future integration only if SuSiE is ported.

Unweighted tau moments and a model-conditional parametric bootstrap are optional
proposals, neither accepted nor required. Do not rerun sealed counting, shape,
or influence work merely to reset this inventory; no automatic new-gene pilot is
planned. New-model SuSiE compatibility/guard work remains deferred until
covariance mapping is consistent.

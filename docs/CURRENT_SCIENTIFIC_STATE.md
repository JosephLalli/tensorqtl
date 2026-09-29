# Current scientific state: hapmixQTL

Reconciled 2026-09-16, updated 2026-09-18; dated sections for 2026-09-25,
2026-09-27 and 2026-09-28/29 added since (the latest state is the last dated
section). A router
to the current state: what is implemented, what was measured, and what is
open.

> **SUPERSEDED IN PART, 2026-09-23.** hapmixQTL now ships exactly **two modes**:
> **mixQTL mode** (the published estimator on posterior-mean counts, no draws,
> `tensorqtl/mixqtl_replication.py`) and **default mode**
> (`Var(eps_i) = sigma^2 v_i` — the Gibbs across-draw variance as a shape with
> the residual scale fitted, no additive floor; `tau_mode='zero'` +
> `se_mode='fitted'`).
>
> Wherever this document discusses choosing among `additive`, `two_component`
> or `library_scaled`, the `variance_prior` shrinkage, `tau_mode='estimate'`, or
> the known-variance standard error, it is describing **DEPRECATED** work. Those
> measurements were correctly made and are not withdrawn as measurements; only
> their status as live options is. The estimators are quarantined in
> `tensorqtl/fitted_variance.py` and the reports in
> `brainvar_hapmix_deploy/deprecated_models/` (see its README). The structural
> reasons, in brief: they fit a variance function from a gene's own squared
> residuals and then weight those residuals by the fit, which no comparator
> method does; and with `(c_g, tau_g)` both free the weights are provably
> invariant to the absolute scale of the Gibbs draws, so the quantifier's
> calibration never reaches the answer.
>
> The sections on what `v_ig` IS — that it is a donor-by-gene interaction, the
> RTA comparison, the draw-count adequacy, the `squeezeVar` diagnostics — remain
> live: they measure the Gibbs variance itself, not a choice among models.

## State at a glance

- **Implemented, committed (bea450c, 2026-09-16):** ASE regression, null tau,
  and lead-refit tau use no automatic intercept; total expression retains its
  intercept.
  [ASE_IMPLEMENTATION.md](../../../../../brainvar_hapmix_deploy/mixqtl_algorithm_review_20260914/salmon_variance_theory_20260915/ASE_IMPLEMENTATION.md)
  records the scope and targeted validation. Measured on the 29 calibration
  genes (`/mnt/ssd/lalli/brainvar_hapmix_deploy/deprecated_models/estimator_ablation_20260916/REPORT.md`):
  median |change| in the lead statistic 0.42, one borderline call (TCF4) added.
- **Current source behavior:** Gibbs summaries use natural logs and add an
  extra Poisson q term when `count_noise=True`; runtime migration to log2 is
  pending. (SUPERSEDED 2026-09-25 for the default-mode runner, whose phenotype
  is `summaries_from_point_estimates` in log2 units; see "Pipeline
  correction, 2026-09-25" below.) It computes cross-channel Gibbs covariance `Cat` but the scan does
  not use it. See `tensorqtl/hapmixqtl.py:349` (`compute_summaries_from_gibbs`;
  CORRECTED 2026-09-18 from a stale `:325`, which is inside
  `orient_haplotypes`) and the method map in `docs/hapmixqtl_methods.md`.
- **Why 57.8% of donor-gene pairs carry zero allele-specific information
  (established 2026-09-18):** a Salmon-indexing and pipeline-ingest fact,
  not a low-expression one. `salmon index` runs without `--keepDuplicates`
  against the personalized diploid transcriptome, so a homozygous donor's
  duplicate `_L`/`_R` transcript collapses to one row; the ingest script
  then credits reads to the allelic channel only from PAIRED `_L`/`_R`
  transcripts, so an unpaired (homozygous) transcript's reads reach the
  total channel but not the allelic one. 22.7% of these pairs (416,207,
  13.1% of all 3,170,044 donor-gene pairs) are genuinely expressed with zero
  allelic information, not merely low-expression. The code's `no_cov` guard
  (`tensorqtl/hapmixqtl.py:417-420`) already handles this correctly; see
  CLAUDE.md's "Why 57.8%..." bullet for the full mechanism, code citations,
  and two open items.
- **Validated, bounded evidence:** the closed controlled Salmon experiment
  supports counting default Gibbs uncertainty once for its singleton,
  fixed-depth ASE configuration: Gibbs-only predicted/observed variance was
  0.9441 (95% bootstrap interval 0.7888–1.1551); Gibbs plus q was 1.9059.
  It does not validate total-expression variance, eQTL p-values, interval
  calibration, residual tau, or multimodal posteriors. Gibbs-only nominal 95%
  normal-interval coverage was 91.5%, so average variance agreement is not
  interval calibration. Read
  `/mnt/ssd/lalli/brainvar_hapmix_deploy/salmon_gibbs_counting_sim_20260915/REPORT.md`.
- **Structural relationship to limma/edgeR/sleuth/swish clarified
  (2026-09-18):** read fresh upstream source (git clones made on 2026-09-18, not the
  older installed R packages) for edgeR's quasi-likelihood pipeline, limma's
  `squeezeVar` (the empirical-Bayes posterior these packages moderate a
  per-gene scale through) and `vooma` (limma's way to build a per-observation
  weight matrix from a continuous covariate), sleuth, and swish. See
  "Structural relationship to limma, edgeR, sleuth, swish" below for the
  three-layer framing and the two cases for where a per-observation variance
  can live; `CLAUDE.md` has the checkable component-by-component reference
  table and the numeric findings. Confirms two things already suspected
  here: the spike at `tau_g <= 0` (40-47% of well-expressed genes, "Three
  variance models" bullet in CLAUDE.md) is why `squeezeVar` cannot be
  applied to `tau_g` directly, and `v_ig` is a donor-by-gene interaction
  (median within-gene across-donor sd of log `v` 0.77, allele-resolved-reads
  R^2 only 0.32 — RELABELED 2026-09-18 from "depth"; see CLAUDE.md),
  which is why neither `catchSalmon` (edgeR's per-transcript overdispersion
  from Salmon Gibbs/bootstrap draws, pooled across samples)/sleuth-style
  per-gene pooling nor limma-style per-sample array weights can substitute
  for the `(c_g, tau_g)` weight matrix. New 2026-09-18: the empirical-Bayes
  moderation family (`squeezeVar`, `fitFDist`, `fitFDistRobustly`,
  `fitFDistUnequalDF1`) runs fine here, since it is closed-form and never
  fits a linear model — unlike `lmFit`/`vooma`/`voomaLmFit`/
  `voomWithQualityWeights`/`arrayWeights`, which all crash on this machine's
  mixed BLAS/LAPACK install (see the environment note below; every one of
  these was tested individually on 2026-09-18). `squeezeVar` was run
  diagnostically on the allelic channel's naive per-gene scale, confirming
  `tau` is real (the scale's quartiles 0.72/1.34/2.10 exceed the 1.00 the
  Gibbs draws alone would give) while showing classic empirical-Bayes
  moderation is nearly inert at BrainVar's depth (median 2.7% of a gene's
  moderated variance from the prior). Practical consequence: of the two
  limma-native routes identified here, the pooled-trend route (`vooma` with
  a Gibbs-derived predictor) cannot be attempted until the BLAS is fixed;
  the moderation route (`squeezeVar`, and `fitFDistRobustly` for the
  hypervariable tail) can be run today. Open: whether a limma-`vooma`-style
  cross-gene trend (fit once, between genes, then applied within every gene)
  should replace or supplement the current per-gene fit (fit separately,
  within each gene) — not measured, not settled by the moderation result
  above, and not currently runnable here regardless (see below). Sharper
  yet (Joseph's observation, checked 2026-09-18): hapmixQTL's layer 1 is the
  only one in the whole comparison that is NOT fixed before a gene's own
  residuals are seen — every other method's per-observation variance is set
  in advance, and so, in a different way, is the shipped `additive`
  default's `c=1` — and under `two_component`/`library_scaled`'s clamped
  fit, where `(c_g, tau_g)` are both free per gene, the weights are provably
  invariant to the absolute scale of the Gibbs draws, only their within-gene
  shape survives (NOT true of `additive`, which feels the draws' scale the
  same way sleuth's fixed `c=1` does). Both are read in full below
  ("Structural relationship..."), with the measurement of
  where in allele-resolved-read space the weights actually track the draws
  at all (57.5% of informative donor-gene datapoints have `c_g v_ig > tau_g`
  overall, falling to 18-36% under 100 allele-resolved reads/donor).

- **Completed bounded shape audit:** three randomly selected libraries
  (566_R1→566_D1, 591_R1→593_D1, 618_R1→618_D1), 9,000 transcript and 4,500
  diagnostic-filtered gene distributions. Gaussian was competitive within
  0.05 bits/draw for 98.51% of gene ASE and 99.93% of total summaries. Three
  ASE candidates were screened; ZNF529 and RNF175 reproduced two fitted modes
  across both block groups, and none occurred for total expression. TRAPPC5
  shows transcript modes disappearing after aggregation. ZNF529 occupancy
  ranged 12–76%; RNF175's eight restart blocks were 44,0,0,0,0,72,36,0%, so
  reliable pooled mode probabilities remain unestablished, especially for
  RNF175. Read
  `/mnt/ssd/lalli/brainvar_hapmix_deploy/gibbs_shape_pilot_20260915/REPORT.md`.
  Fractional exported counts meant discrete NB/BB PMFs were not scored.

## Current decision order

**State on 2026-09-16, after measurement.** The three estimator questions
reopened on 09-14 to 09-16 were run on the 29 calibration genes
(`/mnt/ssd/lalli/brainvar_hapmix_deploy/deprecated_models/estimator_ablation_20260916/REPORT.md`).
The ASE intercept and the counting term `q` are inert there (median |change| in
the lead statistic 0.42 and 0.19 chi2). The residual scale is not: with the
DerSimonian-Laird `tau` the whitened residual mean square on the allelic channel
is 1.22 (0.93-2.58 per gene), and a Paule-Mandel `tau` that restores 1.00
lowers the reported statistics by a median 11%, the null 95th percentile from
21.6 to 18.9 and halves the between-gene spread of the null. That is the
measured content of the "fitted residual scale" recommendation in
[IMPLEMENTATION_STATUS_20260916.md](IMPLEMENTATION_STATUS_20260916.md); the
code still uses DL by default. The residual shape remains wrong after PM
(standardized squared residual rises with `log v`, pooled slope +0.21):
`c*v + tau` is no longer just a candidate, it is implemented and measured as
the `two_component`/`library_scaled` variance models with an empirical-Bayes
prior on `(c_g, tau_g)` (2026-09-17 — see CLAUDE.md's "Three variance models
and an empirical-Bayes prior" bullet for the current numbers), just not the
default (`additive` is). Settled directions: log2 units, no automatic ASE
intercept, GPU matrix scan; `count_noise` stays True until zero-read samples
in the total channel get a coverage-based rule.

**Decision record.**

1. Settled by measurement (`deprecated_models/estimator_ablation_20260916`): the allelic channel
   is through-origin (bea450c); the counting term `q` is inert for genes with
   reads and stays on only as a floor for zero-count total samples until a
   coverage-based floor replaces it; a fitted per-variant residual scale is the
   shipped known-variance estimator with a self-consistent `tau`, so it is not
   a new method; the cross-channel Gibbs covariance `Cat` stays out of the
   statistic (measured corr(beta_a, beta_t) within 0.03 of zero at noise
   correlation 0.9, live test); the moment/GPU representation of the draws is
   retained (shape pilot and two-gene influence audit, reports below).
2. Awaiting the user's decision: DerSimonian-Laird vs Paule-Mandel `tau`. The
   numbers are in the ablation report; nothing further needs measuring for the
   level. The residual shape (`c*v + tau`) is implemented and measured
   (`two_component`/`library_scaled`, 2026-09-17, CLAUDE.md); what remains
   open is whether to make one of them the default in place of `additive`.
3. Open and unmeasured: the `tau = 0` boundary and zero-read total samples, both
   invisible on well-expressed genes; a depth-stratified arm of the ablation
   from the existing cache is the experiment. Whether the Gibbs posterior
   covariance stands in for repeated-library measurement error cannot be
   settled on BrainVar (one library per donor).
4. `tau_A`/`tau_T` name the biological residual variance after the tested cis
   effect and covariates (the user's definition of the target); the estimator
   is and will remain an aggregate residual moment, since the design cannot
   separate biological from technical residual.

The Codex-period experiments remain the record for what they measured: the
[mixQTL algorithm review](/mnt/ssd/lalli/brainvar_hapmix_deploy/mixqtl_algorithm_review_20260914/REPORT.md),
the [counting simulation](/mnt/ssd/lalli/brainvar_hapmix_deploy/salmon_gibbs_counting_sim_20260915/REPORT.md),
the [Gibbs shape pilot](/mnt/ssd/lalli/brainvar_hapmix_deploy/gibbs_shape_pilot_20260915/REPORT.md),
the [two-gene influence audit](/mnt/ssd/lalli/brainvar_hapmix_deploy/gibbs_influence_audit_20260915/REPORT.md)
(a self-contained GLS, not the production mapper), and the equations of the
proposed extensions in [IMPLEMENTATION_STATUS_20260916.md](IMPLEMENTATION_STATUS_20260916.md).
The influence audit's restart-block occupancy instability (RNF175 blocks
44,0,0,0,0,72,36,0%) is carried forward as an unresolved fact about the draws.

## Structural relationship to limma, edgeR, sleuth, swish

Read 2026-09-18 from fresh upstream source (git clones made that day, not the
older R packages installed on this machine — see below). The full
component-by-component reference table, with file paths and installed-vs-
devel-only status, is in `CLAUDE.md`. This section is the framing a new
reader needs before the table makes sense.

**Three layers, shared across all of these methods.** Every one of limma,
edgeR's quasi-likelihood pipeline (which literally calls limma's moderation:
`glmQLFTest` -> `squeezeVar`, edgeR R/glmQLFTest.R:152), sleuth, and hapmixQTL
factors the same problem into the same three layers:

1. **The per-observation variance treated as known — except by hapmixQTL.**
   Every published method here fixes layer 1 before looking at a gene's own
   residuals: limma takes prior weights (or none, `w=1`) as given; `vooma`
   fits a trend across ALL genes and applies it to this one; edgeR's
   quasi-likelihood pipeline takes a quasi-dispersion from the fitted mean;
   `catchSalmon` pools RTA overdispersion from Gibbs draws across samples;
   sleuth pools bootstrap variance to one number per transcript. hapmixQTL
   fits `(c_g, tau_g)` FROM the gene's own squared residuals (the regression
   of squared residuals on `[v, 1]`) and then uses that fit to weight those
   same residuals. This is a categorical difference, not one of degree
   (Joseph's observation, checked 2026-09-18):

   | Method | Layer-1 source | Fixed before seeing this gene's residuals? |
   |---|---|---|
   | limma | prior weights, or none (`w=1`) | Yes |
   | `vooma`/`voomaLmFit` | trend fitted across all genes | Yes — a between-gene trend, not this gene's own residuals |
   | edgeR quasi-likelihood | quasi-dispersion from the fitted mean | Yes |
   | `catchSalmon` | RTA overdispersion from Gibbs draws, pooled across samples | Yes |
   | sleuth | bootstrap variance, pooled across samples | Yes |
   | hapmixQTL | `(c_g, tau_g)` fit from this gene's own squared residuals | **No** |

2. **One positive per-gene scale, estimated from layer 1's residuals.**
   limma/edgeR fit `sigma_g^2` (or a quasi-dispersion); hapmixQTL fits
   `(c_g, tau_g)`.
3. **Cross-gene moderation of that scale**, toward a fitted prior or trend.
   limma's `squeezeVar` empirical-Bayes posterior is the shared mechanism
   (`(df*var + df.prior*var.prior)/(df+df.prior)`, limma R/squeezeVar.R);
   hapmixQTL's analog is the `(c_g, tau_g)` prior in
   `estimate_variance_priors`/the continuous trend prior (`_trend_prior`).

RESOLVED 2026-09-23 by deprecating the whole family: the shipped model is
`Var(eps_i) = sigma^2 * v_i` (fitted scale times the Gibbs variance, no
floor; `tau_mode='zero'` + `se_mode='fitted'`), and `additive`,
`two_component`, `library_scaled` and `variance_prior` are deprecated,
historical-only, and to be removed. The paragraph below states the defect
that motivated it and is retained for that reason.

hapmixQTL's departure is at layer 1: under every `variance_model` it fits
its per-observation variance FROM the gene's own squared residuals, then
uses that fit to weight those same residuals (the table above) — `tau_g`
alone (`c` fixed at 1) under the shipped `additive` default, or the
two-parameter `c_g * v_ig + tau_g` (rather than `v_ig` alone) under
`two_component`/`library_scaled`. Every method it is being compared to
depends on the premise that the layer-1 variance is fixed before the
residuals are seen; under that premise, a layer-2 weighted residual scale
that comes out close to 1 is informative — it says the fixed weights were
right. Once the weights themselves are fit from the residuals, a scale near
1 is close to guaranteed by construction (the `squeezeVar` diagnostic
below, and CLAUDE.md's fuller account of it, measure how close) and says
comparatively little on its own: it is better read as a symptom of the
circularity than as independent confirmation the weights are correct. This
is the same objection an adversarial review raised on 2026-09-18, and
rejected, against a simpler one-parameter reparametrization
(`sigma_g^2 = tau_g`, `w = 1/(1 + v/tau)`; this note is that review's
record, there is no separate report file): making the weight depend on the
scale the model is meant to estimate breaks the premise that makes a
moderated t-statistic exact. Precisely: `1/(1+v/tau) = tau/(tau+v)`, the shipped `additive`
weight `1/(v+tau)` up to a gene-constant factor — the SAME one-parameter
circularity already shipped. `two_component`/`library_scaled` go further,
with `c_g` also free and also fit from the residuals it then weights — two
circular parameters where the rejected proposal, and `additive`, have one.

Two measured consequences, both 2026-09-18 and detailed in CLAUDE.md's
"Relationship to limma..." section, and here `additive` and the free-`c`
models diverge. Under `two_component`/`library_scaled`'s clamped fit the
weights are exactly invariant to the absolute scale of the Gibbs draws
(rescale every `v_ig` in a gene by a constant and `c_g` rescales inversely,
leaving `c_g v_ig + tau_g` unchanged — only the within-gene SHAPE of `v`
across donors survives). This does NOT hold under `additive`, which fixes
`c=1` rather than fitting it — exactly sleuth's choice — so `additive`'s
weights DO feel the draws' absolute scale; it is the scale-respecting
member of this model family, and it is `two_component`/`library_scaled`
that discard that scale. That is not an endorsement of `additive`: trusting
the scale at a fixed `c=1` means trusting a scale the data say is off
(measured `c` is 2.6 median on the 29 calibration genes, 1.8
transcriptome-wide, both in CLAUDE.md), which is why the shared fix below
is a global or trend `c`, not reverting to `c=1`. Even for those, the invariance is only
approximate under the production `variance_prior` shrinkage, whose
bin-level prior mean is estimated from OTHER genes and so does not itself
rescale. `estimate_variance_priors`/`_trend_prior` already push layer 3
onto `(c_g, tau_g)` directly, arrived at independently before this
comparison was made on 2026-09-18.

**Where in allele-resolved-read space the fitted weights actually track the
draws.** (RELABELED 2026-09-18 from "read-depth space"/"reads per donor":
the tier variable is allele-resolved reads `mL+mR` per donor, not total
expression or sequencing depth — see CLAUDE.md's 57.8% bullet.) Reported by
Joseph 2026-09-18 as a session computation not yet folded into
`variance_layer_mapping_20260918/` (full table and the read-tier breakdown
in CLAUDE.md's "Relationship to limma..." section): over the production
`variance_prior`-shrunk fits' informative set (1,174,211 donor-gene
datapoints with `v_ig > eps`, 16,674 genes — not the full 34,457 x 92 grid
the other measurements above use), 57.5% have `c_g v_ig > tau_g` overall,
rising by expression tier from 18-36% under 100 allele-resolved reads/donor
to 89% above 1,000. Below 100 allele-resolved reads/donor (45% of the
transcriptome, 7,455 genes) the Gibbs-derived signal is almost entirely
flattened by the floor and donors are weighed nearly alike regardless of
what the draws say; above 300 allele-resolved reads/donor the weights track
the draws closely. This is how much of the
draws' WITHIN-GENE SHAPE survives the floor into the weights, a separate
quantity from the approximate-invariance gap above (that gap is about the
draws' ABSOLUTE scale reaching the weights through the shrinkage prior;
this is about how much of their within-gene shape gets through at all).

`squeezeVar` itself is not used in production, and cannot be applied to
`tau_g` unmodified: it requires a positive variance with known degrees of
freedom under a scaled chi-square sampling distribution, and `tau_g` is a
signed difference of moments, non-positive for 40-47% of well-expressed
genes (the spike-at-zero problem already in CLAUDE.md's "Three variance
models" bullet). `squeezeVar` was nonetheless run diagnostically on the
naive per-gene scale under pure `1/v` weights (it is unaffected by the
linear-model-fitting crash below, since it is closed-form and never fits a
model): see CLAUDE.md's "Relationship to limma, edgeR, sleuth, swish" section
for the
numbers. Two things came of it. First, the per-gene scale's quartiles
(0.72/1.34/2.10, against 1.00 if the Gibbs draws explained all the scatter)
confirm that the excess over 1 — `tau` — is demanded by the data, not an
invented term. Second, moderation itself is nearly inert at BrainVar's
depth (median 2.7% of a gene's moderated variance from the prior, because
the median informative-donor count per gene, 72, leaves little for
cross-gene borrowing to add), which weighs against the case for heavier
layer-3 borrowing in general, though it does not by itself settle the
narrower `vooma`-style layer-1 trend question below.

**Two cases for where a per-observation variance can live**, and which one
we are in:

1. **One number per gene, pooled across samples.** `catchSalmon` (edgeR) and
   sleuth's `sigma_q_sq` both do this. It is the well-supported, well-trodden
   case upstream, but it is the wrong shape for us: `v_ig` is a
   donor-by-gene interaction, not a gene property. Measured 2026-09-18 on
   the cached per-draw arrays, 34,457 genes x 92 donors x 200 draws, by
   `/mnt/ssd/lalli/brainvar_hapmix_deploy/variance_layer_mapping_20260918/variance_layer_measurements.py`
   (not under git, cite by path): within a gene, across donors, log
   `v` has median sd 0.77 (residual sd at matched allele-resolved read count
   0.557) while allele-resolved read count (RELABELED 2026-09-18 from
   "sequencing depth"; see CLAUDE.md) explains only R^2 = 0.32 of it; at
   matched read count `v` still spans 1.7-fold between donors of the same
   gene — what remains once read count is held fixed is heterozygosity.
   Pooling across samples into one number per gene, as `catchSalmon`
   and sleuth do, would average this donor-specific signal away.
2. **One number per gene-per-sample, a full weight matrix.** limma's `lmFit`
   accepts a weights matrix directly (`Var = sigma_g^2 / w_ig`), and `vooma`/
   `voomaLmFit` is limma's supported way to *build* one from a continuous
   per-observation covariate — a role `v_ig` can fill. This is the case
   hapmixQTL is actually in, and it already has upstream precedent (`CAVEAT`:
   `vooma`'s trend coefficients are identified BETWEEN genes, via
   `rowMeans(predictor)`, then applied WITHIN a gene between donors; those
   slopes need not agree, and hapmixQTL's own residual-shape diagnostic —
   the standardized squared residual vs `log v` slope discussed under the
   `two_component`/`library_scaled` bullet in CLAUDE.md — can test whether
   they do here. Not tested as of 2026-09-18).

The per-donor-gene shape is also why `v_ig` cannot be a donor property
either, ruling out a limma-style per-sample array-weight factor on its own:
the per-donor mean of read-count-adjusted log `v` has sd only 0.073 across
the 92 donors, far tighter than the within-gene, across-donor spread above. Only a
full gene-by-sample matrix — what hapmixQTL already fits — holds this
structure. The allelic channel carries about three times as much of its
structure at the gene-sample level as the total channel: within-gene/
between-gene median sd of log `v` is 0.41/2.12 (ratio 0.19) for total,
0.72/1.29 (ratio 0.55) for allelic (same measurement,
`variance_layer_mapping_20260918/`).

**Open, not measured, though the argument for one side is now stronger.**
The layer-1 circularity and the (approximately) discarded absolute scale of
the draws point toward the same fix, though closing the circularity takes
more than a global coefficient: a global or smooth-in-expression `c` alone
restores the draws' absolute scale (closing the scale-invariance gap) but
not the circularity, since a per-gene `tau_g` fit from the residuals it
then weights keeps `c*v + tau_g` dependent on this gene's own residuals.
Closing the circularity needs BOTH the coefficient and the floor fixed in
advance, from a between-gene trend — the full `vooma` form, with whatever
per-gene freedom remains moved to a layer-2 scale estimated given those
fixed weights (CLAUDE.md's "Both defects..." paragraph has the mechanism
and the existing, currently unused, `_trend_prior` curves that could
supply it). That is exactly hapmixQTL's own `vooma`-style pooled-trend
route. This is an argument for prioritizing this measurement, not a
substitute for it: it upgrades the case for the trend route from "more
stable" to "restores the premise every layer-1 method in the table above
depends on," but whether hapmixQTL's per-gene fit of `(c_g, tau_g)` should
instead (or in
addition) borrow a cross-gene trend the way `vooma` does remains
unmeasured — i.e., whether the `vooma` caveat above (between-gene trend
coefficients applied within-gene) actually holds on BrainVar, and whether
it would help or hurt relative to the current per-gene fit. This is a
genuinely open empirical question, not a known direction. The `squeezeVar`
diagnostic above (layer-3 moderation of a naive per-gene scale, median 2.7%
prior contribution) bears on it without closing it: it says classic
single-scale cross-gene borrowing has little left to add once a gene has
~72 informative donors, but it does not test `vooma`'s different
mechanism — a between-gene TREND in the scale, applied within each gene —
which could still help even where moderation of the scale's level does not.

## Pipeline correction, 2026-09-25 (point estimates, log2 units, genotype-tied permutation)

Six user rules, stated in full with a code map in `docs/pipeline_rules.md`.
The "SUPERSEDED IN PART, 2026-09-23" box above describes mixQTL mode as
running "on posterior-mean counts, no draws"; a posterior mean is computed
from the draws, and mixQTL mode now takes point estimates.

- **Implemented and committed:** the library, the default-mode runner and the
  mixQTL port and driver. The first corrected mixQTL-driver run is in
  `brainvar_hapmix_deploy/mixqtl_replication_point_estimates_20260925/`.
- **Not yet switched:** `scripts/compare_pipelines.py`; no corrected stored
  null and no before/after calibration comparison.
  **UPDATE 2026-09-26:** the corrected stored null and the before/after
  comparison this bullet says were missing have both been run since
  (`corrected_null_store_20260925/`, `total_channel_decomposition_20260926/`);
  `docs/pipeline_rules.md`'s "Nominal-p calibration on the corrected
  pipeline" and "What made the total channel worse" sections are the record.
  `compare_pipelines.py` is still not switched.
- **Open, needs a user decision:** 1,208 filtered genes have no Gibbs draws;
  Salmon's point estimates put one haplotype at exactly zero in 79 of 2,188
  informative pairs on the 29 calibration genes, which dominates the
  corrected weighting ablation.
  **UPDATE 2026-09-26:** two more open decisions were added on the corrected
  pipeline, each its own section in `docs/pipeline_rules.md`: which
  weighting configuration ships (the shipped Gibbs-both-channels default is
  anticonservative in the corrected null; unit/1/v split weighting and
  `1/(v+1)` both calibrate it, at different costs to precision), and why
  tying the genotype PCs to the genotypes under permutation interacts with
  the weights to move the total channel's calibration.
- **Pre-correction:** every calibration number recorded on or before
  2026-09-25, and the stored null in `protein_coding_null_store_20260925/`.

## Reference degrees of freedom, Meier's correction and the known-effect benchmark (2026-09-27)

Commits 8a06803 to cf488c2 on branch `simulation-benchmark`. This section
routes; the numbers live in the documents and pages it names, all result
paths under `/mnt/ssd/lalli/brainvar_hapmix_deploy/`.

- **Implemented and committed, library.** Commit 8a06803: in default mode
  `pval_a` and `pval_t` are referred to t on each channel's own residual
  degrees of freedom (informative donors minus fitted columns),
  `pval_nominal` to the Welch-Satterthwaite degrees of freedom of the
  inverse-variance combination (the value that matches the first two moments
  of the combined variance estimate to a scaled chi-square, with the channel
  weights treated as fixed), and the allelic channel enters the combined
  statistic only for a gene with at least 15 informative allelic donors
  (`MIN_ALLELIC_DONORS`, mixQTL's own cutoff for combining its channels;
  waived in an allelic-only run). New columns `dof_nominal`, `dof_a`,
  `dof_t`, `allelic_admitted`; an off channel's p is NaN rather than 1.
  Commit a1b2ef4 (user decision): Meier's first-order correction, which
  multiplies the combined standard error by
  `sqrt(1 + 4 f_a f_t (1/dof_a + 1/dof_t))`, `f` the channels' weight shares,
  because weights estimated from the same residuals they combine make the
  plug-in variance too small; applied in `map_nominal`, in `map_cis`'s
  observed scan and every permutation, and at the lead (`_meier_factor`).
  Rule and derivation: `docs/hapmixqtl_methods.md` Section 4.5; columns:
  `docs/outputs.md`. Tests `tests/test_hapmixqtl_allelic_df.py` and
  `tests/test_hapmixqtl_meier.py`; the surface is 215 tests (`CLAUDE.md`,
  "Self-tests").
- **Implemented and committed, benchmark and checks.** `scripts/plasmode/` is
  the numbered benchmark pipeline (`README.md` there; run order `run_all.sh`;
  acceptance `99_acceptance.py`, which writes only into its own acceptance
  roots); the previous twelve scripts were removed in commit fc238df. Two gene
  sets: the deep set (the 100 genes of `corrected_null_store_20260925`) and
  the low-coverage set (`stratum30_100`: 100 genes whose median
  haplotype-informative reads over admitted allelic donors lie in [30, 100),
  each with at least 15 admitted allelic donors, drawn once by
  `scripts/plasmode/select_stratum_genes.py`). One-off scripts beside it:
  `scripts/allelic_df_null_check.py`, `scripts/combined_reference_exact_model.py`,
  `scripts/salmon_half_depth_check.py`, `scripts/rasqual_read_level.py`.
- **Validated results.**
  - The per-channel references on the stored 100-gene null
    (`allelic_df_fix_20260927/`): `docs/pipeline_rules.md`, "After the
    per-channel t references", and `docs/hapmixqtl_methods.md` Section 7.
    Measured before Meier's correction.
  - The combined reference under the exact model, with and without Meier's
    correction (`combined_reference_exact_model_20260927/`): Section 4.5. The
    correction removes most of the reference's excess; a residual remains,
    largest at 0.001 with 15 to 30 allelic donors under split weighting.
  - The known-effect benchmark on the current library, nine arms: four
    hapmixQTL weightings (gibbs, split, unit, plus_one), mixQTL mode at its
    published and permissive cutoffs, each with its own permutation p,
    RASQUAL, asSeq TReCASE and total-only tensorQTL. Gene-level power is
    scored by each arm's permutation p where it has one and by eigenMT's p for
    every arm (eigenMT: the gene's smallest nominal p times an effective
    number of independent tests counted from the eigenvalues of the tested
    variants' genotype correlation matrix; at 92 donors that count is set by
    the matrix's shrinkage rather than by linkage disequilibrium, which both
    pages state). Deep set `plasmode_meier_20260927/report/plasmode_report.html`;
    low-coverage set `plasmode_lowcov_meier_20260927/report/plasmode_report.html`.
    Their RASQUAL and TReCASE results are staged from the earlier runs, which
    the correction does not touch.
  - The thinning rule against Salmon itself
    (`salmon_half_depth_20260927/salmon_half_depth.html`; donor 100
    re-quantified at half depth): the rule over-predicts the allelic Gibbs
    variance below 1,000 haplotype reads (measured over predicted, median 0.80
    at 30-99 reads; pre-registered verdict FAIL), and Salmon at half depth
    makes far more one-sided records than thinning and attenuates the allelic
    ratio. The low-coverage page states this as its limit.
  - RASQUAL on native per-SNP allele counts against the benchmark's pseudo
    feature SNP, 30 observed genes (`rasqual_read_level_20260927/report.html`):
    native's excess of p < 0.05 at random variants of null genes persists under
    the records permutation and under records plus haplotype swap, so it is an
    offset of its statistic, not association; the pseudo construction's share
    has 95% intervals that include 0.05 under records, records plus swap and
    RASQUAL's own `-r`.
- **Run state.** Nothing of this project was running when this section was
  written (2026-09-27). Current benchmark roots: `plasmode_meier_20260927`,
  `plasmode_lowcov_meier_20260927`. Earlier runs kept as records:
  `plasmode_20260926` (deep set, rerun under 8a06803, before Meier's
  correction) and `plasmode_stratum30_100_20260927` (low-coverage set, before
  Meier's correction). `plasmode2_acceptance_20260927` and
  `plasmode2_stratum_acceptance_20260927` are acceptance roots, not results.
- **Open.** Which weighting ships (`docs/pipeline_rules.md`, "Open decision:
  which weighting configuration ships"); the two benchmark pages are now its
  known-effect evidence. The residual of the corrected reference in the tail,
  and the unit-weighted total channel's own excess at 0.001, which the
  correction does not touch (`docs/hapmixqtl_methods.md` Sections 4.5 and 7;
  `CLAUDE.md`, "Known and unfixed").
  Deferred, not done: re-running the stored nulls under Meier's correction;
  every stored-null rate, including the bands in section 3.7 of the deep-set
  page, predates it.

## Native counts, held-out replication and the effect-size question (2026-09-28/29)

Commits 9369bb1 to 5d9123f on branch `simulation-benchmark`. No change to
the library (`tensorqtl/` is as at a1b2ef4). This section routes; result
paths are under `/mnt/ssd/lalli/brainvar_hapmix_deploy/`.

- **Implemented and committed.**
  - The external benchmark harness (`tests/ase_external_benchmark.py`, commit
    9369bb1) now builds the total channel from Poisson draws of the simulated
    totals, fits the allelic channel through the origin and runs default mode
    at gibbs, split and plus_one; `scripts/external_benchmark_mirror*` run it
    beside the real asSeq TReCASE. The three harnesses that called its
    removed `tau_mode='estimate'` arm (`tests/ase_rasqual_comparison.py`,
    `ase_rasqual_real.py`, `ase_reference_bias.py`) stop with that reason
    (user decision); their recorded results stand, as `docs/ase_validation.md`
    and `docs/LOCAL_HANDOFF.md` say.
  - Alignment-based ("native") counts for TReCASE and a control arm:
    `scripts/native_counts.py` (featureCounts totals, phASER haplotype
    counts), with the phASER inputs rebuilt to phASER's assumptions
    (`scripts/phaser_input_vcf.py`, `phaser_features.py`,
    `phaser_stranded.py`) and WASP filtering ahead of phASER
    (`scripts/wasp_star_index.sh`, `phaser_wasp.py`). Current counts
    `native_counts_wasp_20260928`; the two earlier builds
    (`native_counts_20260928`, `native_counts_stranded_20260928`) are kept.
    The benchmark's `scripts/plasmode/05b_native_arms.py` runs
    `split_native` and `trecase_native` on them; 06 and 08 score and report
    them (`scripts/plasmode/README.md`, "Native-input arms").
  - The held-out replication referee: `scripts/referee_replication.py`,
    `referee_trecase.py`, `referee_score.py`.
  - The input diagnosis of TReCASE and RASQUAL:
    `scripts/trecase_input_diagnosis.py`, `rasqual_input_diagnosis.py`.
  - Summary pages: `scripts/benchmark_summary.py` and
    `scripts/hapmix_vs_trecase.py`, each with its `_template.html`.
  - chr14, chr15 and chr22 excluded short term (user decision; `CLAUDE.md`,
    "Known and unfixed").
- **Validated results** (each record states its own numbers and limits).
  - The fixed external benchmark at N = 200 and N = 92:
    `external_benchmark_current_20260928/report.html`. It supersedes the
    2026-09-23 record for default mode.
  - Why TReCASE and RASQUAL rank below total-only tensorQTL on Salmon inputs
    (deep set, one input changed at a time):
    `input_diagnosis_20260928/trecase_integer/report.html` and
    `input_diagnosis_20260928/rasqual_total_only/report.html`.
  - The native counts, stage by stage, with their effect on the native arms:
    `phaser_stranded_20260928/README.md` and `wasp_20260928/README.md` (the
    latter tabulates power for all three builds and the referee's TReCASE
    difference). With WASP the per-donor reference-allele share is
    0.500-0.516; the referee's own `facts.json` still carries the pre-WASP
    share and is not the figure to quote.
  - Both benchmark pages (`plasmode_meier_20260927`,
    `plasmode_lowcov_meier_20260927`) were rescored on 2026-09-29 with the
    native arms on the WASP counts; each root keeps the earlier native stages
    as `native_unstranded_20260928/` and `native_stranded_nowasp_20260928/`.
  - Held-out replication on real data, 11,740 genes, 92 discovery and 135
    held-out donors, every arm scored at matched list depth (TReCASE, on
    native counts only, on a seeded 1,500-gene subset with every arm
    restricted to it): `referee_replication_20260928/report.html` (numbers
    in its `score/score.json`). The referee measures total expression only.
  - One-page summaries: `benchmark_summary_20260929/summary.html` and
    `benchmark_summary_20260929/hapmix_vs_trecase.html`.
- **Run state.** Nothing of this project was running when this section was
  written, on 2026-09-29 before the effect-size work below began (the WASP
  rerun chain, `wasp_rerun_chain_20260928.log`, ends "done" at 03:02).
- **Open.**
  - Which weighting ships (`docs/pipeline_rules.md`); the referee is now its
    real-data evidence beside the two benchmark pages.
  - **The effect-size shortfall.** Measured 2026-09-28/29, explained
    2026-09-29 (below); the decisions it feeds are open. The statistic is 06's `bias_count`: per non-null gene
    unit, the slope at the planted causal variant divided by the count-scale
    truth (`allelic_truth`, the planted effect, for a combined or joint
    slope; `total_truth` for tensorQTL, whose one slope is a total-channel
    slope), averaged over a scenario's finite units (at most 150: 50 genes x
    3 datasets), 95% interval from resampling genes (`summary.json`,
    `recovery/beta0.4/<arm>/combined/bias_count`; Table 3.3 and Figure 2 of
    each page). For split weighting's combined slope at |beta| 0.4 it is
    0.915 [0.866, 0.966] on the deep set and 0.721 [0.667, 0.777] on the
    low-coverage set. On the deep set it is 0.778 [0.649, 0.905] in the 42
    units of genes under 100 median allele-resolved reads, against 0.961
    and 0.978 at 100-999 and 1,000+.
    Total-only tensorQTL on the same Salmon totals gives 0.898 and 0.706,
    `split_native` on alignment counts 0.911 and 0.723, and TReCASE on
    alignment counts 0.991 and 0.941. Cause (validated 2026-09-29, plasmode
    only): mostly the total channel's log2(CPM + 1) pseudocount, the rest the
    allelic channel's Gibbs-variance weights, which are computed from the same
    counts as the allelic ratio. Record, with the budget, the counterfactual
    refits through `map_nominal`, a voom log-CPM arm and the limits:
    `brainvar_hapmix_deploy/beta_shortfall_20260929/beta_shortfall.html`
    (scripts `beta_shortfall_budget.py`, `beta_shortfall_refits.py`,
    `beta_shortfall_report.py`). Proposed, not run: a real-data allelic
    weighting that avoids the coupling; voom with decoupled allelic weights
    together.

## Routing and run state

Start with `docs/hapmixqtl_methods.md` for implementation and
`docs/ase_validation.md` for historical calibration claims, then use
`docs/brainvar_deploy_runbook.md` for the BrainVar comparison state. The
cross-project review at
`/mnt/ssd/lalli/brainvar_hapmix_deploy/mixqtl_algorithm_review_20260914/REPORT.md`
is the current methods/review record. The mixQTL replication arm's own report,
including the weighting ablation that answers what the Gibbs draws buy, is
`/mnt/ssd/lalli/brainvar_hapmix_deploy/mixqtl_replication_20260919/REPORT.md`
(regenerated 2026-09-20 under mixQTL's published cutoffs; the permissive
configuration is preserved beside it as a sensitivity arm).

Work that is **proposed and not yet run** is kept out of this document and out
of `CLAUDE.md`, both of which record what is established. It lives in
[OPEN_INVESTIGATIONS_20260920.md](OPEN_INVESTIGATIONS_20260920.md): currently
the per-channel residual sigma test (the last untested of the six mixQTL
disagreement mechanisms) and the pending log2 unit migration. (The residual-
sigma test was itself closed 2026-09-20, refuted; that file keeps it only as
a pointer and no longer lists it as open work — this sentence was not
updated when that happened.)

Worktree as of 2026-09-16: `hapmix-runbook-local`; the through-origin change
is commit `bea450c` on top of `f11d586`. Current work (the last two dated
sections above) is on branch `simulation-benchmark`. The estimator ablation
(`deprecated_models/estimator_ablation_20260916`) is complete and reproducible from its scripts. Neither completed pilot nor audit implemented
final TMM normalization, production association mapping, or biological-residual
calibration. Applying phASER error correction before constructing quantification
references remains a future option, not an implemented workflow change.

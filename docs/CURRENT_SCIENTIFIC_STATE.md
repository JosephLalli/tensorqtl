# Current scientific state: hapmixQTL

State at the release documentation pass of 2026-10-01. This document says what
ships, what has been validated and on which configuration, what is proposed or
on hold, what is open, and what was running. Superseded ideas appear only in the
last section, as pointers; evidence measured on an earlier configuration appears
above it only with that configuration named.

Detail lives elsewhere: [pipeline_rules.md](pipeline_rules.md) is the input and
estimator contract with its dated decision records;
[hapmixqtl_methods.md](hapmixqtl_methods.md) defines the statistic;
[outputs.md](outputs.md) defines the output columns;
[brainvar_deploy_runbook.md](brainvar_deploy_runbook.md) runs the BrainVar
deployment; [ase_validation.md](ase_validation.md) holds the historical
calibration record, including withdrawn claims. Result folders named below are
under `/mnt/ssd/lalli/brainvar_hapmix_deploy/` unless a full path is given.

## At a glance

- **Ships.** Two modes only: mixQTL mode (the published estimator on Salmon
  point estimates, no Gibbs draws) and default mode (half-read split: Gibbs
  variance as the shape of the allelic channel's error, half-read total
  expression with unit weights, fitted residual scales). The half-read default
  was adopted on 2026-09-29 (merge `86b947f`), its expression PCs moved to the
  same unit on 2026-09-30 (`a2f4314`), and the default path was closed on
  2026-10-01 (`8e347fd`: input contract, rank checks, refusal of the
  known-variance paths, leave-one-donor-out influence at each lead).
- **Validated on exactly the shipped default** (the simulated-effects benchmark with
  half-read PCs, plus an exact-model check of the combined reference): on the
  benchmark's no-effect dataset the combined nominal rejection rate
  at 0.05 / 0.01 / 0.001 is 0.0444 / 0.0078 / 0.0008 on the deep gene set and
  0.0486 / 0.0095 / 0.0015 on the low-coverage set; the combined slope recovers
  0.983 [0.937, 1.033] and 0.986 [0.960, 1.013] of the simulated effect at
  beta 0.4 and 0.8 on the deep set, 0.925 [0.849, 0.997] and 0.943 [0.907,
  0.981] on the low-coverage set.
- **Most comparative evidence predates the shipped default.** Power against
  competitors, the held-out replication referee, the TReCASE and RASQUAL
  head-to-heads and the external mirror benchmark were measured on the
  predecessor configuration (split weighting on a `log2(CPM + 1)` total with
  `log2(CPM + 1)` expression PCs) and are labelled so below.
- **On hold or deferred.** Re-quantifying the 92 donors with the
  `--gibbsPriorGroups` Salmon fork (user decision 2026-10-01); every current
  result uses stock Salmon draws, whose `Va` is too small for donor-gene pairs
  with few haplotype-informative reads. Re-running the stored nulls under
  Meier's correction.
- **Open.** chr14, chr15 and chr22 excluded; the nominal p's anticonservatism
  under Gibbs weighting (mechanism identified, source not); single-donor
  influence (now visible, not prevented); the Beta approximation's tail
  conservatism; no per-sample allelic read floor; 1,208 calibration genes
  untested; the comparative evidence not re-run under the shipped default and
  no transcriptome-wide run under it.
- **Run state.** Nothing of this project was running on 2026-10-01.

## What ships

### Two modes

**mixQTL mode** (`tensorqtl/mixqtl_replication.py`) is a NumPy port of
`hakyimlab/mixqtl` at `624ae44` with the eleven divergences found in the
2026-09-14 review removed. It consumes Salmon point estimates and never the
Gibbs draws, so it is the no-draws comparator: the measure of what the draws
buy. Published cutoffs `100/50/10/1000` (the GTEx v8 driver that produced the
paper), `weight_cap` 10, natural-log response by design. Driver
`scripts/compare_mixqtl_replication.py`.

**Default mode** is the shipped hapmixQTL configuration. Its inputs are built
by `prepare_default_inputs` in `tensorqtl/hapmixqtl.py` from Salmon point
estimates (`quant.sf` NumReads) and the 200 Gibbs draws:

- The allelic contrast is `A = log2((pL + 0.5)/(pR + 0.5))` of the
  point-estimate haplotype counts, summed over paired `_L`/`_R` transcripts.
  Its variance `Va` is the across-draw variance (ddof 0) of
  `log2((yL + 0.5)/(yR + 0.5))` plus the delta-method counting term
  `q = (1/(pL + 0.5) + 1/(pR + 0.5)) / ln(2)^2` (`count_noise=True`). A
  donor-gene pair enters the allelic channel only if `Va > 1e-12`,
  `pL + pR > 0`, and not exactly one haplotype is below 0.5 reads; an excluded
  pair carries `Va = 0`.
- The total phenotype is the half-read log-CPM
  `T = log2((pT + 0.5)/(effective_library_size + 1) * 1e6)`, where `pT` sums
  every transcript of the gene (never `pL + pR`) and the effective library
  size is edgeR's library size times its TMM factor (trimmed mean of
  M-values: a per-library scale chosen so that most genes' log ratios to a
  reference library centre on zero). The total working variance is
  `Vt = 1` and every donor is kept, including zero-count donors.
- Weights are `1/Va` in the allelic channel, used as a shape, and 1 in the
  total channel. Each channel's error is `Var(eps_i) = sigma^2 v_i` with
  `sigma^2` fitted per channel and variant (`tau_mode='zero'`,
  `se_mode='fitted'`). The allelic channel is fitted through the origin with
  no covariates (`ase_covariates_df=None`); the total channel has an
  intercept and the covariates.
- Each channel's p is referred to t on that channel's own residual degrees of
  freedom. The combined statistic is the inverse-variance combination of the
  two channel slopes, referred to t on Welch-Satterthwaite degrees of freedom
  (the value that matches the first two moments of the combined variance
  estimate to a scaled chi-square, with the channel weights treated as fixed).
  Its standard error carries Meier's first-order correction for estimated
  weights, a factor `sqrt(1 + 4 f_a f_t (1/dof_a + 1/dof_t))` with `f` the
  channels' weight shares (`_meier_factor`), applied in `map_nominal`, in
  `map_cis`'s scan and every permutation, and at the lead. The allelic
  channel enters the combined statistic only for a gene with at least 15
  informative allelic donors (`MIN_ALLELIC_DONORS`). Derivations:
  `docs/hapmixqtl_methods.md` Section 4.5.
- The permutation null is `records_signflip`: donor records are permuted
  (phenotype value, weight and the covariate row tied to the RNA record move
  together, genotypes stay), and each permuted record's haplotype labels are
  swapped with probability one half, which negates its allelic log ratio.
  Genotype PCs stay with the genotypes.
- Covariates are age, age squared, sex, RIN and ten expression PCs (tied to the
  RNA record) plus three genotype PCs (tied to the genotypes). The expression
  PCs are built on the half-read log-CPM since 2026-09-30, and their gene
  filter equals the eQTL gene filter (`filterByExpr`, protein-coding,
  autosomal; 12,955 genes). Current build
  `cov/half_read_point_calibration_20260930/`; the Salmon runner refuses
  covariates whose recorded PC unit, gene set or library sizes differ.
- The detection call is the empirical permutation p (`pval_perm`, and
  `pval_beta`, its approximation by a Beta distribution fitted to the permuted
  minimum p values). The lead's `pval_nominal` is never a gene-level p.

The rules behind each choice, with their dates and code map, are in
[pipeline_rules.md](pipeline_rules.md). The driver from Salmon output is
`scripts/run_hapmixqtl_from_salmon.py` (`--selftest` exercises it end to end).
The raw BED CLI (`tensorqtl --mode hapmixqtl` or `hapmixqtl_nominal`) supplies
`Vt = 1` when `--hap_Vt` is omitted but cannot establish that the supplied `T`
is half-read or that the allelic admission rule was applied.

### The default path as closed on 2026-10-01

Commit `8e347fd`.

- **Input contract.** `map_nominal` and `map_cis` call `_validate_inputs`: `A`,
  `T`, `Va` and `Vt` must share rows and columns in the same order and be
  finite, `Va` and `Vt` nonnegative, phenotype, sample and variant ids unique,
  and phase given as both frames or neither with rows equal to the genotype
  rows and finite values. Before, each violation was accepted silently; under
  the default the reproduced symptom was that a NaN in `A` dropped the allelic
  channel and moved the lead.
- **Rank checks.** The covariate design with its intercept must be finite and
  of full column rank (`_combine_covariates`), and `WeightedResidualizer`
  refuses a weighted design that zero weights leave rank-deficient.
- **Leave-one-donor-out influence at each lead.** `map_cis` reports
  `loo_donor`, the donor whose exclusion from both channels moves the lead's
  combined |t| furthest toward zero, and `loo_pval_nominal`, the lead's
  nominal p without that donor. Under the default an exclusion changes no other
  donor's weight, so this is the exact refit; it runs as one batch through the
  record-permutation machinery (`_lead_influence`, about 8 ms per gene at 92
  donors) and is pinned against `map_nominal` with the donor masked
  (`tests/test_hapmixqtl.py`). At closure it was also checked against a
  per-donor refit loop on 40 genes (same donor in 40 of 40; |t| and degrees of
  freedom within 7.8e-5 relative); logs and scripts in `brainvar_hapmix_deploy/release_closure_20261001/`. Limits:
  the lead is held fixed although excluding the donor can move it, `pval_perm`
  is not recomputed, and a donor that alone identifies a covariate level is
  not evaluated. It is a diagnostic, never a filter.
- **No lead refit.** With no additive variance there is nothing to refit; the
  runner no longer passes `tau_refit` and its evaluation bundle records
  `tau_refit` False. The lead's `slope`, `slope_se` and `pval_nominal` are on
  the scan's fitted scale.
- **Fine-mapping is not supported in default mode.** `map_susie` has no
  per-channel fitted residual scale and refuses `tau_mode='zero'`;
  `--mode hapmixqtl_susie` was removed from the CLI. Its deprecated
  `tau_mode='estimate'` default remains only to reproduce earlier results.
  Credible sets and PIPs were never validated.
- **The STR-curvature and multi-allelic categorical second pass is not
  supported in default mode.** `_second_pass` refuses `tau_mode='zero'` (it
  has known-variance standard errors only) and the Salmon runner no longer
  runs it; `--str-vcf` and `--multiallelic` still add STR and multi-ALT rows
  to the `map_cis` scan. This was the fourth entry point found defaulting to
  the withdrawn known-variance pairing, after `map_susie` and two in the
  2026-09-23 CLI trim.
- **`scripts/compare_pipelines.py` is retired.** It ran the pre-correction
  pipeline (natural-log Gibbs-mean phenotype, known-variance second pass). It
  is archived with its SHA256 in `retired_scripts_20261001/` (README there),
  whose README states that its RASQUAL output field list now lives in the
  simulated-effects benchmark's `common.py` and in `scripts/rasqual_read_level.py`;
  `scripts/realized_variance.py`, which called it, stops with that reason.

### Engineering cleanup of 2026-09-30

The default Salmon runner no longer allocates or aggregates total-channel
Gibbs arrays (`YT`); two historical callers keep them through
`load_counts(include_total=True)`. Benchmark lead extraction rejects malformed
p values and records untestable pairs; analysis outputs are written
atomically; cache reuse verifies inputs, sources and outputs. 39 targeted tests
and the runner self-test passed and 16 regenerated numerical tables matched the
saved results exactly; no estimator, mapping kernel or scientific conclusion
changed. Receipts: `half_read_cleanup_20260930/final/run_manifest.json`;
entry point and check scope: [half_read_analysis.md](half_read_analysis.md).

### Deprecated and quarantined (2026-09-23)

`variance_model` (`additive`, `two_component`, `library_scaled`),
`variance_prior` (`deciles`, `trend`), `tau_mode='estimate'`, which they
require, and the known-variance standard error `se_mode='model'`. Code
`tensorqtl/fitted_variance.py`, tests `tests/fitted_variance/`, records
`deprecated_models/` (its README states the reasons). The two reasons are
structural: each model fits a gene's per-observation variance from that gene's
own squared residuals and then weights those residuals by the fit, which no
comparator method (limma, edgeR, sleuth, swish) does; and with both `c_g` and
`tau_g` free the weights are invariant to the absolute scale of the Gibbs
draws. `tests/test_fitted_variance_quarantine.py` pins that a default-mode run
never imports the quarantined module. Their measurements are not withdrawn;
only their status as options is.

## Validated results

Each entry states the configuration it was measured on. Three configurations
carry evidence: the shipped default exactly; the shipped half-read total with
the earlier `log2(CPM + 1)` expression PCs (the evidence the 2026-09-29
adoption rested on); and the predecessor, split weighting on a
`log2(CPM + 1)` total. The simulated-effects benchmark builds datasets with simulated
effects (beta = 0.2 / 0.4 / 0.8, log2 allelic fold change) from the cohort's
own Salmon records; its deep gene set is the 100 genes of
`corrected_null_store_20260925` and its low-coverage set 100 genes whose median
haplotype-informative reads over admitted allelic donors lie in [30, 100)
(`benchmark/simulated_effects/README.md`).

### On the shipped default

- **Null rates and sensitivity to the PC unit**
  (`scripts/expression_pc_unit_impact.py`; log
  `cov/half_read_point_calibration_20260930/expression_pc_unit_impact.log`).
  Refitting the default on the benchmark's no-effect dataset (487,454 tested
  gene-variant pairs deep, 517,376 low coverage) with the earlier and the
  half-read PCs, the combined `pval_nominal` rejection rate at 0.05 / 0.01 /
  0.001 moved from 0.0443 / 0.0079 / 0.0009 to 0.0444 / 0.0078 / 0.0008 (deep)
  and from 0.0484 / 0.0098 / 0.0014 to 0.0486 / 0.0095 / 0.0015 (low
  coverage). On three beta = 0.8 datasets per set the median slope change was
  0.04 to 0.05 standard errors (largest 0.51 to 0.67), the correlation of
  -log10 p 0.993 to 0.9996, and the causal slope over the simulated effect
  rose by +0.001 to +0.004. The two 10-PC spaces have canonical correlations
  0.974 to 0.9999. These rates come from one no-effect dataset per set, whose
  tested pairs are not independent.
- **Effect-size recovery** (`scripts/beta_recovery_current.py`,
  `beta_recovery_current_20260930/`). Combined slope over the simulated
  effect, 150 causal gene-dataset units per entry, 95% interval from
  resampling genes: deep 0.983 [0.937, 1.033] at beta 0.4 and 0.986 [0.960,
  1.013] at 0.8; low coverage 0.925 [0.849, 0.997] and 0.943 [0.907, 0.981].
  TReCASE on alignment-based counts (same record) gives 0.991 / 0.984 deep and
  0.941 / 0.958 low. The low-coverage shortfall is
  almost all allelic (allelic slope 0.751 / 0.806 of the simulated effect;
  total channel 0.980 / 0.988). No paired hapmixQTL-TReCASE interval was
  computed and beta 0.2 was not measured.
- **The combined reference under the exact model**
  (`scripts/combined_reference_exact_model.py`,
  `combined_reference_exact_model_20260927/`). Phenotypes replaced by normal
  noise at each gene's real scales, with the shipped weighting structure
  (allelic Gibbs variance, unit total), so the result does not depend on the
  total phenotype's transform. Pooled rejection over nominal at 0.05 / 0.01 /
  0.001 with Meier's correction: 1.017x / 1.047x / 1.123x at 15 allelic donors
  and 1.005x / 1.022x / 1.031x with all donors (uncorrected 1.087x / 1.164x /
  1.331x and 1.050x / 1.093x / 1.149x). Each channel alone is nominal on its
  own degrees of freedom. Full table: `docs/hapmixqtl_methods.md` Section 4.5.

### On the shipped half-read total with the earlier expression PCs

The 2026-09-29 adoption evidence. The PC change was measured to move null
rates and effect recovery by the small amounts stated above; standard errors,
power and mean squared error were not re-measured with the half-read PCs.

- **Half-read trial** (`half_read_trial_20260929/CONCLUSION.md`, `REPORT.md`,
  `repeated_summary.json`). Against the predecessor, combined recovery at
  beta 0.4 rose from 0.915 to 0.980 (deep) and 0.721 to 0.918 (low coverage);
  mean squared error against the simulated effect (bias squared plus
  variance) has ratio half-read / predecessor 1.019 [0.823, 1.256] deep and
  1.257 [1.064, 1.470] low at beta 0.4, and 0.691 deep and 0.673 low at
  beta 0.8. On a records null of 2,000 permutations at one variant per gene
  (100 genes, 92 donors), the half-read total channel rejects at 0.05 in
  0.05083 (deep) and 0.05065 (low) and at 0.001 in 0.001840 and 0.001585, so
  its tail is not shown nominal at every threshold. In an
  independent negative-binomial count model the total-only half-read
  transform recovers 99.6-100.6% of the effect with 94.8-95.2% coverage of
  nominal 95% intervals, but at dispersion 0.2 on the low-coverage set its
  variance, normalized by the squared noise-free response, rose 13-17%
  (1.159 [1.138, 1.179] at beta +0.4). The trial's conclusion predates the
  adoption and its no-adoption sentence is superseded by it.
- **Reported standard errors, nominal p, power and precision-recall**
  (`half_read_se_comparison_20260929/`, `half_read_pvalue_comparison_20260929/`,
  `half_read_unit_power_pr_20260929/index.html`). On the methods' common finite
  support at beta 0.4 (142 deep and 118 low-coverage causal units; 959 common
  finite units over all betas), mean reported slope SE is deep 0.1009
  (half-read), 0.0918 (split predecessor), 0.0994 (unit weights) and low
  0.1682 / 0.1311 / 0.1345. Oracle power at beta 0.8 (the share of non-null
  genes called at the threshold that holds the realized false-discovery
  proportion at 5%) is deep 0.853 / 0.853 / 0.820 and low 0.553 / 0.527 /
  0.560. Intervals from
  resampling genes are wide and establish no universal winner. These are the
  measurements the user weighed in accepting the beta/precision tradeoff;
  adoption records in `half_read_default_adoption_20260929/`
  (`verification.json`, `integration.json`).

### On the predecessor configuration (split weighting, `log2(CPM + 1)` total)

- **Stored records null** (`hybrid_weights_null_20260926/`,
  `allelic_df_fix_20260927/`): 100 genes, 200 permutations. Split weighting's
  combined rate is 0.0512 / 0.0117 / 0.0027 at 0.05 / 0.01 / 0.001 under the
  earlier shared 73-df reference and 0.0505 at 0.05 and 0.0012 [0.0010,
  0.0014] at 0.001 under the per-channel references, before Meier's
  correction; Gibbs weights in both channels read 0.0719 / 0.0206 / 0.0054
  under the 73-df reference.
  Tables and decomposition: [pipeline_rules.md](pipeline_rules.md).
- **Known-effect benchmark pages**, nine arms (four hapmixQTL weightings,
  mixQTL mode at both cutoff settings, RASQUAL, asSeq TReCASE, total-only
  tensorQTL), gene-level power scored by each arm's permutation p and by eigenMT
  (the gene's smallest nominal p times an effective number of independent
  tests counted from the eigenvalues of the tested variants' genotype
  correlation matrix): `plasmode_meier_20260927/report/plasmode_report.html`
  (deep) and `plasmode_lowcov_meier_20260927/report/plasmode_report.html` (low
  coverage), made with the `log2(CPM + 1)` PCs and the four-arm configuration
  of 2026-09-27, native arms rescored on WASP-filtered counts on 2026-09-29.
  They are records, not a run of the shipped default. One-page summaries:
  `benchmark_summary_20260929/summary.html` and `hapmix_vs_trecase.html` beside
  it.
- **Held-out replication on real data**
  (`referee_replication_20260928/report.html`, numbers in
  `score/score.json`). 11,740 genes, 92 discovery and 135 held-out donors; the
  referee measures total expression only. Among each arm's top 200 genes by
  eigenMT p, 0.900 of total-only tensorQTL's leads replicate against 0.820 of
  split weighting's (difference -0.080 [-0.130, -0.030]); at the top 1,565
  genes split replicates more, +0.077 [+0.056, +0.095].
- **External mirror benchmark on TReCASE's own generative model**
  (`external_benchmark_current_20260928/report.html`, `summary.json`; 500
  replicates). The split arm's total phenotype is
  `summaries_from_point_estimates`'s `log2(T/lib + 1)`. At N = 200 split
  rejects at 0.052 / 0.014 at 0.05 / 0.01 against TReCASE's 0.054 / 0.012, with
  matched power 0.274 / 0.758 / 0.998 against 0.274 / 0.760 / 0.998 at allelic
  fold change 1.05 / 1.10 / 1.20; at N = 92, 0.058 / 0.012 against 0.050 /
  0.014 and power 0.138 / 0.392 / 0.924 against 0.148 / 0.436 / 0.942 (paired
  differences -0.010 / -0.044 / -0.018, paired se 0.010 / 0.015 / 0.009).
  TReCASE is the likelihood of the generating model, so parity here is parity
  with a ceiling.
- **Why the predecessor's slopes in the simulated-effects benchmark fell short of the simulated effect**
  (`beta_shortfall_20260929/beta_shortfall.html`): mostly the total channel's
  `log2(CPM + 1)` pseudocount, the rest the allelic channel's Gibbs-variance
  weights, which are computed from the same counts as the allelic ratio. Split
  weighting's combined slope at beta 0.4 was 0.915 [0.866, 0.966] deep and
  0.721 [0.667, 0.777] low. This motivated the half-read total. Literature
  context: population-expression allelic fold-change estimates run 6.35%
  smaller than allele-specific ones overall in
  [Mohammadi et al. (2017)](https://genome.cshlp.org/content/27/11/1872),
  inverse-normalized coefficients are not molecular fold changes
  ([ACME, 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC5920774/)), and
  pseudocount-induced fold-change distortion is general
  ([Erhard, 2018](https://doi.org/10.1093/bioinformatics/bty471)).
- **Why TReCASE and RASQUAL rank below total-only tensorQTL on Salmon inputs**
  (deep set, one input changed at a time):
  `input_diagnosis_20260928/trecase_integer/report.html` and
  `input_diagnosis_20260928/rasqual_total_only/report.html`.
- **Alignment-based ("native") counts** for TReCASE and a control arm
  (`scripts/native_counts.py`; featureCounts totals, phASER haplotype counts on
  a strand-split exonic model after WASP filtering): current build
  `native_counts_wasp_20260928/`; stage-by-stage records
  `phaser_stranded_20260928/README.md` and `wasp_20260928/README.md` (with WASP
  the per-donor reference-allele share is 0.500-0.516).
- **RASQUAL on native per-SNP allele counts**
  (`rasqual_read_level_20260927/report.html`, 30 observed genes): the native
  construction's excess of p < 0.05 at random variants of null genes persists
  under records permutation with and without the haplotype swap, so it is an
  offset of its statistic, not association.

### The Gibbs variance itself (stock Salmon 1.10.3 draws)

These measure `v`, not a choice among models, and hold for the shipped default.
All were made on draws sampled under Salmon's stock Gibbs prior of 1 per active
transcript (next section).

- **It carries counting noise.** In a controlled Salmon experiment on singleton
  equivalence classes at about 100 reads per haplotype, Gibbs-only predicted
  over observed variance was 0.9441 (95% interval 0.7888-1.1551) and Gibbs plus
  the counting term 1.9059; nominal 95% normal-interval coverage was 91.5%
  (`salmon_gibbs_counting_sim_20260915/REPORT.md`). The counting term `q`
  therefore double-counts sampling noise where reads exist; it is inert for
  weighting on well-covered genes (efficiency 0.3402 against 0.3410 without
  it, `mixqtl_replication_20260919/REPORT.md`) and the default keeps it by
  rule 6 of [pipeline_rules.md](pipeline_rules.md).
- **Two hundred draws suffice to treat `v` as known**: lag-1 autocorrelation
  median 0.081, median effective draws about 170, relative sd of `v` median
  0.109 with 3.2% of datapoints above 0.25
  (`variance_layer_mapping_20260918/variance_layer_measurements.json`).
- **`v` is a donor-by-gene interaction.** Within a gene, log `v` has median sd
  0.77 across donors; allele-resolved read count explains R^2 = 0.32 of it,
  leaving residual sd 0.557 at matched read count (heterozygosity, not depth);
  the per-donor mean of read-count-adjusted log `v` has sd 0.073 across 92
  donors. Only a gene-by-sample weight matrix holds it, not a per-gene pooled
  value (as in edgeR's `catchSalmon` or sleuth) nor a per-sample factor. The
  within-gene / between-gene median sd of log `v` is 0.72 / 1.29 allelic and
  0.41 / 2.12 total (same record).
- **At gene level, read-to-transcript ambiguity adds almost nothing.** edgeR's
  read-to-transcript-ambiguity (RTA) overdispersion estimator on 3,000 genes:
  quartiles 1.00 / 1.00 / 1.01, 61.4% at the floor of 1; the across-draw
  variance of the log total is 0.98x (IQR 0.91-1.05) an RTA-inflated Poisson's
  (same record, `rta_vs_c.py`).
- **Cross-gene moderation would change little.** limma's `squeezeVar`
  (empirical-Bayes shrinkage of per-gene variances toward a fitted prior) on the
  allelic channel's per-gene scale under pure `1/v` weights gives prior degrees
  of freedom 2.0, so at the median 72 informative donors 2.7% of a gene's
  moderated variance comes from the prior; the scale's quartiles are 0.72 /
  1.34 / 2.10 (same record, `run.log`; 2026-09-18, pre-correction natural-log
  pipeline). Under the shipped model the per-donor
  shape `v` is fixed before a gene's residuals are seen, as in limma, edgeR,
  sleuth and swish; only the per-channel scale is fitted.
- **What the weights buy.** In the allelic channel `1/v` weighting cut the
  null-permutation variance of the slope to 0.340 of unweighted on the 29
  calibration genes (25 of 29 genes; pre-correction natural-log pipeline,
  `mixqtl_replication_20260919/REPORT.md`), and on the corrected pipeline the
  per-gene ratio of realized null-slope sd, unit over Gibbs weights, has median
  1.38 (10th-90th percentile 1.11-2.19). In the total channel the same ratio has
  median 0.93 and Gibbs weights made the stated se about 7% too small
  (`gibbs_weight_benefit_by_gene_20260926/per_gene.tsv`,
  `total_channel_decomposition_20260926/`), which is why the total channel
  carries unit weights.
- **Why 57.8% of donor-gene pairs carry no allelic information.** `salmon
  index` ran without `--keepDuplicates` (`aux_info/meta_info.json`
  `"keep_duplicates": false`), so a homozygous donor's byte-identical `_L` and
  `_R` copies collapse to one row, and the runner's `pair_haplotypes` credits
  allelic reads only from transcripts with both rows. 13.1% of all 3,170,044
  donor-gene pairs (416,207) are expressed with no allelic information; the
  total channel keeps them. Under point estimates the share without allelic
  information is 59.2% ([pipeline_rules.md](pipeline_rules.md), "Built
  inputs").
- **Shape and influence of the draws**, bounded audits on three libraries and
  two genes: a Gaussian is competitive within 0.05 bits per draw for 98.51% of
  4,500 gene allelic and 99.93% of total summaries, with two reproducible
  two-mode allelic exceptions (ZNF529, RNF175)
  (`gibbs_shape_pilot_20260915/REPORT.md`); the largest one-block
  mean-plus-covariance shift was 0.0736 and 0.1396 working SE
  (`gibbs_influence_audit_20260915/REPORT.md`). Not association calibration.

### Salmon's Gibbs prior and the split-half calibration of `Va` (2026-10-01)

Record `salmon_informative_reads_20260930/README.md` (sections "The
--gibbsPriorGroups option", "Split-half calibration of Va", "Why the
point-estimate and Gibbs priors differ") and page
`salmon_informative_reads_20260930/split_half/split_half_calibration.html`.

- **The production draws' prior.** Salmon 1.10.3 with default flags fits its
  point estimate by variational Bayes (an optimizer that adds a prior
  pseudocount, `--vbPrior`, of 0.01 per transcript) but samples its Gibbs draws
  under a prior of max(1, `--vbPrior`) = 1 per active transcript
  (`CollapsedGibbsSampler.cpp`; the floor dates from Salmon v1.2.0). The
  record's re-implementation of the sampler reproduces Salmon's draws on four
  genes only with that prior. For a gene without allelic information, the
  stock prior makes the haplotype split a Beta(k, k) with k the gene's active
  paired isoforms on a haplotype, so its Gibbs variance depends on isoform
  count.
- **Split-half calibration, donor 100 only, random error only.** Donor 100's
  42,449,536 read pairs were split at random into two disjoint halves, each
  quantified under five prior configurations, and per gene
  `z = (A1 - A2) / sqrt(Va1 + Va2)` was formed with the pipeline's `Va`
  including `q` (8,198 genes admitted in both halves); mean z^2 is 1 when `Va`
  matches the between-half error. With stock draws, mean z^2 by full-depth
  haplotype-informative reads 3-10 / 10-30 / 30-100 / >=100 is 2.70 / 3.17 /
  1.81 / 1.20: `Va` understates the error 1.5- to 3.2-fold at 3-30 reads,
  concentrated in a tail of over-confident genes.
- **The fork.** `--gibbsPriorGroups <file>` divides each transcript's Gibbs
  prior by the number of its group's transcripts in an equivalence class; with
  gene x haplotype groups the prior is 1/k. Point estimates and the
  `--numBootstraps` path are unchanged. Source
  `/mnt/ssd/lalli/usr/local/src/salmon-gibbs-prior` (uncommitted there when
  last recorded),
  patch `/mnt/ssd/lalli/usr/local/src/salmon-gibbs-prior-groups.patch`, binary
  `/mnt/ssd/lalli/usr/local/salmon-1.10.3-gibbspriorgroups`. On donor 100 it
  gives mean z^2 1.04 / 1.56 / 1.23 / 1.06, the closest to calibrated of the
  five configurations, with full-depth point estimates within 0.1 log2 of stock
  in 99.5% of genes (stock against itself 99.4%). Every configuration keeps an
  excess at 10-30 reads (mean z^2 1.35-1.58), source untested; bands below 3
  reads were not judged.
- **Not established:** other donors, full depth, errors shared by both halves
  (reference bias, phasing), a prior of 1 for both point estimate and draws
  (`--vbPrior 1`, not run), and anything about association calibration.
- **Reading the rest of this document.** Default mode uses `Va` as a shape
  under a fitted scale, which absorbs a uniform scale error by construction; an
  error that varies with read count, as this one does, was not tested. Where
  any document calls the Gibbs variance "the quantifier's uncertainty", read it
  as the stock sampler's posterior variance under a prior of 1 per active
  transcript.

### Allelic Gibbs variance and count imbalance (2026-09-30)

`scripts/allelic_weight_imbalance.py`; results
`allelic_weight_imbalance_20260930/imbalance_corrected_null_store_20260925.json`
(deep) and `imbalance_stratum30_100.json` (low coverage). The folder has no
README; this summary is read from the JSON files and the script's docstring.

The question was whether the allelic Gibbs variance grows as a donor's allele
counts become lopsided, so that donors showing an effect more strongly get less
weight. On real records and real draws (no simulation), over admitted
heterozygous donor-gene pairs with both haplotypes at 0.5 reads or more (6,029
pairs in 100 deep genes, 4,497 in 100 low-coverage genes), the within-gene
least-squares slope of log Gibbs variance (without `q`) on log p(1 - p), with
p the pseudocounted allele fraction and log(n + 1) as a second regressor, is
-0.82 [-1.02, -0.69] deep and -0.30 [-0.36, -0.24] low coverage, against -1
under the Poisson delta-method law `Var = 1/(n p (1 - p) ln(2)^2)`; with `q`
included (the shipped `Va`) the slopes are -0.91 and -0.58. Intervals resample
genes (2,000 draws). The Gibbs variance therefore does grow with imbalance,
close to the counting law on the deep set and more weakly on the low-coverage
set; at an even split its median, scaled by (n + 1) ln(2)^2, exceeds the law's
4.0 by more as read count rises (deep: 7.5 at 1-30 reads to 31.9 at 300 or
more).

On the simulated-effects datasets, where each thinned record's variance is set by the
benchmark's formula, the mean within-gene Spearman rank correlation between the
weight `1/Va` and the allelic ratio in the direction of a simulated beta = 0.8
effect is -0.052 [-0.106, +0.002] deep (125 gene units) and -0.171 [-0.220,
-0.118] low coverage (133); with weights from each record before the effect was
simulated it is +0.041 and +0.021, so the paired difference, the coupling the
effect itself creates, is -0.093 [-0.114, -0.074] deep and -0.192 [-0.219,
-0.164] low. On the no-effect dataset the correlation is +0.051 [-0.008,
+0.113] deep and +0.0065 [-0.062, +0.078] low. So under a true effect the Gibbs
weights tilt away from the donors that show it, more on the low-coverage set:
one measured route by which `1/Va` weighting pulls the allelic slope toward
zero, consistent with the low-coverage allelic recovery of 0.751 above. Limit:
the second part measures coupling under the benchmark's variance model, not
under re-quantified Salmon draws; no estimator change follows from it.

### Salmon half-depth check and the dilution of its allelic ratio (2026-09-27, 2026-09-30)

`salmon_half_depth_20260927/salmon_half_depth.html` (with `check.log`,
`summary.json`): donor 100 re-quantified at full and at half depth to test the
benchmark's thinning rule against Salmon itself. The rule over-predicts the
allelic Gibbs variance below 1,000 haplotype reads (measured over predicted,
median 0.798 [0.783, 0.811] at 30-99 reads; pre-registered verdict FAIL), and
Salmon at half depth makes more one-sided records than thinning does (at 30-99
reads 43.34% against 27.48%). The ordinary-least-squares slope of the
half-depth allelic ratio on the full-depth ratio is 0.597 / 0.713 / 0.899 /
0.974 at 1-29 / 30-99 / 100-999 / 1,000+ reads.

`scripts/salmon_half_depth_dilution.py` (log
`allelic_weight_imbalance_20260930/salmon_half_depth_dilution.log`, read from
the half-depth record's `per_gene.tsv`) asked whether that flat slope is
shrinkage of the half-depth ratio toward zero or noise unshared between the two
runs. Shrinkage would narrow the half-depth spread and put the reverse
regression (full on half) above 1; noise in both runs puts both below 1. Over
records two-sided at both depths (537 / 1,105 / 4,876 / 2,418), the reverse
slope is 0.751 / 0.643 / 0.614 / 0.681, below 1 in every band, and the
half-depth sd exceeds the full-depth sd in three of four bands (1.382 against
1.312, 1.220 against 1.009, 0.898 against 0.750; 1.127 against 1.264 at 1-29);
correlations are 0.670 to 0.814. The attenuation is therefore regression
dilution from unshared noise, not shrinkage. The benchmark's thinned ratio is
far closer to its full-depth source (reverse slope 0.908 to 0.995). The
low-coverage benchmark page, written before this dilution check, states the
half-depth check as its limit at that depth.

## Proposed or on hold

- **Cohort re-quantification with `--gibbsPriorGroups`: prepared, on hold
  (user decision 2026-10-01).** Manifest and driver in
  `salmon_gibbspriorgroups_20261001/` (README there): per donor, rebuild the
  personalized index with production's recipe, write the gene x haplotype
  group file, run Salmon in mapping mode with production flags, and accept
  only if index hashes and fragment counts reproduce production; about 19
  hours at 5 donors and 24 threads each. No association, benchmark or
  calibration result exists under the new draws.
- **Stored nulls under Meier's correction: deferred.** Every stored-null rate,
  including the bands in section 3.7 of the deep-set benchmark page, was
  measured without it.
- **A depth-independent variance term, explored as measurement only** (user
  authorization 2026-09-25; nothing ships and no third mode appears without a
  further decision). Candidates must pin the technical coefficient (`Var = v +
  tau`, never `c_g v + tau_g`) and must not let a record's own residual set its
  own weight. Measured so far (pre-correction pipeline,
  `count_scale_weights_20260925/`): a flat additive floor makes the allelic
  weights nearly uniform and is out on efficiency; unit weights for the total
  channel cost no precision, which the shipped default adopted. Unmeasured: a
  closed-form permutation-variance reference and a single global shape
  exponent on `v`. Proposal record: `nominal_p_hypotheses_20260925/`.
- **An allelic weighting that avoids the effect-weight coupling**, on real
  data: proposed in the effect-shortfall record, not run. The count-imbalance
  measurement above bears on it.
- **phASER error correction before building the quantification references**:
  a future option, not implemented.

## Open

- **chr14, chr15 and chr22 are excluded** (user decision 2026-09-28): every copy
  of the phased genotypes stops within the first 1.5-3.1 Mb of these contigs,
  so analyses cover 19 autosomes. The unphased joint calls are complete.
  Record `phased_vcf_inventory_20260928/README.md`. 1,188 of the 1,208
  filtered genes the runner reports but does not test (no Gibbs draws) are all
  the filtered genes on these three contigs; the other 20 lie on seven other
  chromosomes (`release_closure_20261001/untested_genes_check.log`).
- **The analysis uses the statistical phase, not phASER's read-backed phase.**
  phASER ran without `--gw_phase_vcf`, so its read-backed phase was never
  written into GT; on donor 100_D1, chr1:1-6 Mb, the analysis VCF's GT equals
  the phased BCF's at all 41,474 shared biallelic SNPs (7,465 heterozygous, 0
  phase flips; `release_closure_20261001/phase_check.log`). Documents that
  described a read-backed overlay were corrected on 2026-10-01; whether a
  read-backed overlay would change results has not been measured.
- **The nominal p is anticonservative under Gibbs weighting; the mechanism is
  identified, the generative source is not.** Within a gene, records with high
  Gibbs weight have larger whitened squared residuals on average than
  low-weight records, so the fitted scale (an unweighted mean) understates the
  slope's variance (residual variance grows about as `v^0.65`, not `v^1`).
  Measured on the pre-correction pipeline (`nominal_p_hypotheses_20260925/`,
  report `nominal_p_report.html`); the decomposition on the corrected pipeline
  is in [pipeline_rules.md](pipeline_rules.md). The shipped default removes
  Gibbs weighting from the total channel; the allelic channel keeps it. The
  detection call is the empirical permutation p.
- **A single donor record can carry a gene-level call.** On the pre-correction
  pipeline CALM2's `pval_perm` was 0.028 with donor 657_D1's allelic record and
  0.684 without it. Such cases are now visible through `loo_donor` and
  `loo_pval_nominal`, not prevented.
- **The Beta approximation is conservative in the tail**, costing power at
  transcriptome-scale thresholds. This runs opposite to the nominal-p
  inflation; the two must not be netted.
- **No per-donor allelic read floor by default.** mixQTL's published driver required at least 50 reads on each haplotype; the Salmon runner's `--asc-cutoff` supplies one for a matched comparison.
- **Low-information `Va` under stock draws** (previous section); the excess at
  10-30 informative reads that every prior configuration keeps has no
  identified source.
- **Evidence lag.** Power against competitors, the held-out referee and the
  TReCASE and external benchmarks have not been re-run under the shipped
  default, and no transcriptome-wide association run under it exists. Nothing
  below nominal 0.001 has been tested on real genes.
- **1,208 of the 12,955 calibration genes are reported and not tested** by the
  Salmon runner: they have no haplotype-paired transcript in any donor and so
  no Gibbs draws. Under the default their total channel would need no draws;
  the runner still skips them ([pipeline_rules.md](pipeline_rules.md)).
- **Cross-donor correlation** (relatedness, population structure, batch not
  absorbed by the covariates) is untested as a source of miscalibration on
  observed data. A records permutation cannot create excess from it.
- **Historical code paths keep their old units.** `compute_summaries_from_gibbs`
  and the dated scripts that import it remain on the natural log;
  `summaries_from_point_estimates` remains on `log2(CPM + 1)`. Both are kept
  only to reproduce dated results.
- **`tests/ase_gtex_real_data.py` still fabricates the total channel's
  inferential variance** (emulated draws conserve `yL + yR`; no `yT`), so its
  total-channel results measure nothing; its allelic channel is real GTEx
  structure. The external benchmark harness had the same defect and was fixed
  on 2026-09-28.

### No longer open under the default

- **The total channel's zero-count floor.** The total channel has `Vt = 1`, so
  a zero-count total sample cannot take an extreme weight; the problem, and the
  role of `count_noise` as a floor, apply only to the historical
  `compute_summaries_from_gibbs` path.
- **Selection bias of the lead refit.** No refit occurs in default mode.
- **The robust second pass's degrees of freedom** (`_joint_gls`/`_pvals`
  charging the allelic channel an intercept). The second pass is not supported
  in default mode.
- **The three input-validation defects of the 2026-09-14 audit** (a NaN at a
  zero-weight sample, unchecked variant-row identity of the phase frames, no
  rank check on the design): closed by `_validate_inputs` and the rank checks
  on 2026-10-01.

## Run state, 2026-10-01

No hapmixQTL, Salmon, RASQUAL or TReCASE computation of this project was
running (process list checked on 2026-10-01). The prepared re-quantification
in `salmon_gibbspriorgroups_20261001/` was not started. The last commit before
this documentation pass is `8e347fd` on branch `simulation-benchmark`, which
tracks `origin/simulation-benchmark` and is not merged into `master`; verify
the current Git state before acting.

Current result roots: covariates `cov/half_read_point_calibration_20260930/`;
shipped-default effect recovery `beta_recovery_current_20260930/`; benchmark
pages `plasmode_meier_20260927/` and `plasmode_lowcov_meier_20260927/`
(predecessor configuration); referee `referee_replication_20260928/`; external
benchmark `external_benchmark_current_20260928/`. Earlier benchmark runs kept as
records: `plasmode_20260926/` and `plasmode_stratum30_100_20260927/` (before
Meier's correction); `plasmode2_acceptance_20260927/` and
`plasmode2_stratum_acceptance_20260927/` are acceptance roots, not results.
The benchmark code is `benchmark/simulated_effects/` (run order `run_all.sh`,
acceptance `99_acceptance.py`; `SIMULATED_EFFECTS_ROOT` sends a run's outputs to a
fresh directory).

## Superseded: pointers only

Each item below was current once and is not now. The full text this document
carried before the release pass is `git show 8e347fd:docs/CURRENT_SCIENTIFIC_STATE.md`.

- **The 2026-09-16 decision order** (an additive `tau` estimated by
  DerSimonian-Laird, a method-of-moments estimate, against Paule-Mandel, the
  value that makes the weighted residual mean square equal one; the `additive`
  default; the `c*v + tau` models with an empirical-Bayes prior; `count_noise`
  kept as a floor):
  superseded by the 2026-09-23 deprecation. Records
  `deprecated_models/estimator_ablation_20260916/REPORT.md`,
  `deprecated_models/README.md`, [IMPLEMENTATION_STATUS_20260916.md](IMPLEMENTATION_STATUS_20260916.md).
  The through-origin allelic channel it settled (`bea450c`) ships.
- **The 2026-09-18 structural comparison with limma, edgeR, sleuth and swish**
  (three layers: per-observation variance, per-gene scale, cross-gene
  moderation; the circularity of a per-gene fitted variance; the free-`c`
  scale invariance): the argument behind the deprecation. Its measurements of
  `v` are carried above; the rest is in the git text named above.
- **The 2026-09-25 pipeline-correction interim state** ("not yet switched",
  "open, needs a user decision"): resolved in [pipeline_rules.md](pipeline_rules.md).
- **The weighting decision as open** (2026-09-26 to 2026-09-29; candidates
  Gibbs in both channels, split, `1/(v+1)`): settled on 2026-09-29 by adopting
  half-read split. The pre-adoption evidence is in
  [pipeline_rules.md](pipeline_rules.md).
- **The half-read trial's "rejected as a blanket default"**
  (`half_read_trial_20260929/CONCLUSION.md`): superseded the same day by the
  adoption.
- **Fixed donor-gene full balance** of the allelic variance, tested and
  rejected 2026-09-29 (allelic mean squared error ratios 1.882 deep and 2.097
  low at beta 0.4; type-I at 0.05 rose to 0.06967 and 0.09791):
  `beta_balance_trial_20260929/CONCLUSION.md`; its live source was deleted
  after the archive copies matched SHA256.
- **Post-fit adjustment of estimated slopes**: removed with its outputs by user
  decision 2026-09-30; adjustments belong in the transform, weights or design.
- **The 2026-09-23 external benchmark harness** (fabricated total-channel
  variance): superseded by the fixed harness, `external_benchmark_current_20260928/`.
- **The Salmon fork options `--gibbsMinPrior` and `--priorGroups`**: removed on
  2026-10-01; patches in `salmon_informative_reads_20260930/superseded_salmon_patches/`.
- **`scripts/compare_pipelines.py` and its results** (for example
  `rasqual_default_mode_20260923/` and `deprecated_models/null_calibration_29b/`;
  summary page `calibration_summary_20260924/calibration_summary.html`):
  pre-correction records; to rerun them, check out commit `8e347fd~1`.
- **The Salmon-emulator benchmark design**: [simulation_benchmark_spec.md](simulation_benchmark_spec.md),
  superseded by the simulated-effects benchmark (its real-data calibration appendices
  still hold).
- **Dated handoffs and proposals**: [LOCAL_HANDOFF.md](LOCAL_HANDOFF.md) and
  [OPEN_INVESTIGATIONS_20260920.md](OPEN_INVESTIGATIONS_20260920.md) are
  historical; the latter's log2 migration proposal does not describe
  `prepare_default_inputs`.
- **Early methods records that remain valid measurements** but predate every
  current rule: the [mixQTL algorithm review](/mnt/ssd/lalli/brainvar_hapmix_deploy/mixqtl_algorithm_review_20260914/REPORT.md)
  and the mixQTL replication arm (`mixqtl_replication_20260919/REPORT.md`,
  regenerated 2026-09-20 under the published cutoffs).

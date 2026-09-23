# tensorQTL, hapmixQTL branch

A fork of tensorQTL adding **hapmixQTL**: cis-eQTL mapping from haplotype-resolved
expression posteriors (Salmon Gibbs draws against a personalized diploid
transcriptome), with the quantifier's inferential uncertainty carried into the
standard error. hapmixQTL is an **extension of mixQTL**.

## THE TWO MODES. There are only two.

Everything else has been deprecated and quarantined (2026-09-23, user
decision). Do not add a third, do not reintroduce a removed one, and do not
report a comparison against one.

### (a) mixQTL mode — the published estimator, no draws

`tensorqtl/mixqtl_replication.py`, a NumPy port of `hakyimlab/mixqtl` @
`624ae44` with all eleven divergences from the 2026-09-14 review removed. It
consumes Salmon **posterior-mean counts** and never touches the draws, so it
doubles as the **no-draws comparator**: it is what hapmixQTL has to beat, and
the only honest measure of what the draws buy. Published cutoffs
(`100/50/10/1000`, the GTEx v8 driver that produced the paper) are the primary
setting; `weight_cap` is 10. Driver `scripts/compare_mixqtl_replication.py`,
analysis `scripts/analyze_mixqtl_comparison.py`.

### (b) default mode — `Var(eps_i) = sigma^2 * v_i`

TIMES, not plus. `v_i` is the **Gibbs across-draw variance**, used as a SHAPE
only; `sigma^2` is the residual scale fitted per variant. There is no additive
floor. Reached by `tau_mode='zero'` + `se_mode='fitted'`, which are the
defaults everywhere and are not worth overriding. Driver
`scripts/run_hapmixqtl_from_salmon.py`, which offers no other hapmixQTL
configuration.

Say **"Gibbs variance"** and **"Gibbs draws"**. Never "bootstrap" — that word
belongs to sleuth and to Salmon's `--numBootstraps`, which this cohort does not
use (200 Gibbs draws).

### What was deprecated, and why it does not come back

Quarantined: `variance_model` (`additive`, `two_component`, `library_scaled`),
`variance_prior` (`deciles` and `trend`), `tau_mode='estimate'` which they
require, and the known-variance standard error `se_mode='model'`. Code in
`tensorqtl/fitted_variance.py`, tests in `tests/fitted_variance/`, reports and
result files in `/mnt/ssd/lalli/brainvar_hapmix_deploy/fitted_variance/`
(which has a README; note that folder name means a fitted variance FUNCTION,
the opposite of `se_mode='fitted'`).

Two structural reasons, neither of them empirical:

1. **Circularity.** Each fits its per-observation variance from a gene's own
   squared residuals and then weights those residuals by the fit. No
   comparator does this — limma, edgeR, sleuth and swish all fix the
   per-observation variance before a gene's residuals are seen — and it breaks
   the premise that makes the statistic exact.
2. **The free-`c` models discard the draws' calibration.** With `(c_g, tau_g)`
   both free, rescaling every `v_ig` in a gene by `k` returns `c_g/k` with
   `tau_g` unchanged: the weights are invariant to the draws' absolute scale,
   so only their within-gene shape survives. Propagating the quantifier's
   uncertainty is the entire point, so a model that cannot feel that scale is
   answering a different question.

**Efficiency does not reopen it.** Per-gene cases where a deprecated model
estimates more precisely exist and are recorded in the quarantine. A model
whose weights are fitted from the residuals they weight can win on realized
variance and still be unsound, because the quantity it optimises is not the
quantity it reports. Efficiency was never the objection.

Anything in `docs/` dated before 2026-09-23 that calls one of these
"production", "default" or "the shipped model" is historical. Those numbers
were correctly measured and are not withdrawn as measurements; only their
status as current practice is.

## Which document answers which question

| Question | Document |
|---|---|
| What is the statistic, exactly, and how do I reproduce it? | `docs/hapmixqtl_methods.md` |
| How do I run the BrainVar deployment end to end? | `docs/brainvar_deploy_runbook.md` |
| What was measured, and how do I know it is calibrated? | `docs/ase_validation.md` — includes withdrawn claims |
| What do the output columns mean? | `docs/outputs.md` |
| What does a new session need to pick this up? | `docs/LOCAL_HANDOFF.md` |
| What is implemented, proposed, validated, running? | `docs/CURRENT_SCIENTIFIC_STATE.md` |
| What was deprecated on 2026-09-23 and why? | `brainvar_hapmix_deploy/fitted_variance/README.md` |

## Scientific phase transitions

Before a new substantive scientific phase, run a documentation agent to
validate and reconcile the actual local state into concise authoritative
pointers to the intellectual state, generated files, performed experiments and
results, decisions, existing code, and how to find them. After that agent
completes, resume already-authorized work. State boundaries explicitly:
distinguish implemented behavior, proposed work, validated results, and current
run state. The pass records existing state only; it does not start an
experiment.

## Writing conventions (the user's)

- Define every named statistical method on first use, in terms of what it
  computes (DerSimonian-Laird, Paule-Mandel, Freedman-Lane, Kish, and so on).
- Never write "cell" for a table entry or a (donor, gene) datapoint; in this
  work "cell" means a biological cell. Say datapoint, donor-gene pair, or
  zero-read sample.
- Reports for the user are HTML pages with figures, not long markdown.
- Say "Gibbs variance", never "bootstrap".

## Facts that are easy to get wrong

- **Use log2 for expression, ASE ratios, aFC, and their uncertainty.** Project
  convention since 2026-09-15. beta=1 means a twofold effect. Runtime
  conversion is PENDING: current outputs are still natural logs. See the
  [unit convention record](/mnt/ssd/lalli/brainvar_hapmix_deploy/mixqtl_algorithm_review_20260914/salmon_variance_theory_20260915/LOG2_CONVENTION.md).

- **ASE has no automatic intercept since 2026-09-15.** `ase_covariates_df=None`
  means through-origin with no nuisance columns; the total intercept remains.
  Results recorded before that used the old design and are historical.

- **Sample order is positional.** Phase frames are indexed by the genotype
  frame's column order. `_assert_phase_columns` guards it; before that,
  reordered columns corrupted the allelic channel while the total channel
  stayed correct. `_assert_keep_frames` does the same job for the cutoff masks.

- **Two scales are reported.** `pval_perm` and `pval_beta` are gene-level and
  come from the scan; `slope`, `slope_se` and `pval_nominal` come from the lead
  refit when `tau_refit=True`. A lead's nominal p is never a gene-level p.

- **The detection call is the empirical p**, not the statistic. Statistic
  magnitude has been corrected several times; the called genes did not move.

- **`tau_a`/`tau_t`/`c_a` come back as `None` in default mode, and that is
  correct.** Under `Var(eps) = sigma^2 v` no such parameters exist in the
  model. Reporting `0.0` would wrongly imply an additive component that was
  estimated and came out at zero. Compare with `.equals()`, not `==`: `None`
  is not equal to itself in a pandas object column, which broke a test.

- **Default Salmon Gibbs includes counting noise, and `count_noise` is inert
  on well-expressed genes.** The controlled experiment at
  `/mnt/ssd/lalli/brainvar_hapmix_deploy/salmon_gibbs_counting_sim_20260915/REPORT.md`
  (Gibbs-only variance / MSE 0.944, with `q` 1.906) shows `q` double-counts the
  sampling noise of donors that have reads, and a census of the production
  draws shows the docstring's other case — reads with unanimous draws — never
  occurs (0 of 76,286 datapoints). The flag is nevertheless load-bearing, for a
  reason neither side originally stated: **the total channel has no
  degenerate-sample guard** (`_zero_degenerate_ase_weights` is ASE-only), so a
  zero-count total sample would keep weight `1/(1e-8 + tau_t)`; `q` prevents
  that only because `1/(0 + 2*kappa) = 1` acts as a floor. 39.95% of donor-gene
  datapoints in 2,000 random genes have no reads (none in the 29 calibration
  genes); without `q` such samples hold a median 89.5% of the total channel's
  weight in the 281 of 1,079 sampled genes that have them, against 17.4% with
  `q`. The fix is a coverage-based floor for zero-count total samples that is
  independent of the flag, then dropping `q` for samples with reads. Excluding
  zero-count total samples is wrong: `t = log kappa` is real low-expression
  information, and exclusion conditions on the outcome. Default stays True
  until that is done.

  Quantified 2026-09-18 for the allelic channel's
  `q_a = 1/(mL+kappa) + 1/(mR+kappa)`, from the cached per-draw arrays (34,457
  genes x 92 donors x 200 draws,
  `/mnt/ssd/lalli/brainvar_hapmix_deploy/variance_layer_mapping_20260918/variance_layer_measurements.py`,
  not under git): median `q_a`/`v` is 0.90 at 1-9 haplotype-informative reads,
  0.30 at 10-99, 0.12 at 100-999, 0.07 at 1,000+. So `q_a` matters least
  exactly where reads are plentiful and `v` is trustworthy alone. Its `+kappa`
  pseudocount is the Haldane-Anscombe correction to the empirical log odds (the
  standard small-sample fix for a zero-count donor-gene datapoint in a two-way
  allele split), whose delta-method variance on the natural-log scale is exactly
  `1/(mL+1/2)+1/(mR+1/2)`; edgeR's binomial pipeline adds the identical half
  (`y <- y + ceiling(prior.count)/2`, `R/binQLFtest.R`). The total channel's
  counting term is a different, single-count Poisson delta-method correction,
  not a log-odds one.

  Where both haplotypes are empty, `compute_summaries_from_gibbs`'s `no_cov`
  guard (`tensorqtl/hapmixqtl.py`, `no_cov = (mL+mR)<=0`) forces `Va` to exactly
  `0.0` rather than `Va + q_a`, and `_zero_degenerate_ase_weights`
  (`keep = va_t > eps`) then zeroes the allelic weight entirely — so no
  fabricated observation of perfect allelic balance enters the allelic channel.
  The guard exists BECAUSE `q_a` would otherwise apply there, and it keeps those
  pairs OUT of the fit rather than in it at a floored weight.

- **Why 57.8% of donor-gene pairs have no allele-specific information: a
  Salmon-indexing and pipeline-ingest fact, not a low-expression one.**
  Established and re-verified 2026-09-18 from pipeline code, run metadata and
  the cached summaries. g2gtools emits TWO haplotype copies of every transcript,
  suffixed `_L`/`_R`; on sex chromosomes only the PAR regions get the suffixes
  (`bin/append_suffixes_to_sex_chroms.py` in `/mnt/ssd/lalli/nf_stage/rnaseq_JLL`,
  keyed to the X/Y `seqid`, because PAR alone is genuinely diploid).
  `salmon index` was run WITHOUT `--keepDuplicates`, so where a donor is
  homozygous across a transcript its `_L` and `_R` sequences are byte-identical,
  the indexer keeps one, and the `_R` row never appears in `quant.sf` at all —
  not a zero row, no row. Every `aux_info/meta_info.json` checked records
  `"keep_duplicates": false`; `_L` counts 176,395/176,222/176,226 against `_R`
  counts 111,472/92,348/107,377 (spread 28,674 across 92 donors, the signature
  of collapsed duplicates rather than lost reads), TPM summing to 1e6 within
  floating-point rounding. CAVEAT: `conf/modules.config` conditionally sets
  `--keepDuplicates` when `use_personalized_references` is true, and this run's
  captured params record it true, so the config text alone suggests dedup
  should have been OFF. It was not. `meta_info.json` is the runtime ground truth
  and is what this relies on; why the conditional did not fire is unresolved and
  is an `rnaseq_JLL` question, not a hapmixQTL one.

  Losing the `_R` row does not by itself zero a pair's allelic information.
  What makes `mL` ALSO exactly 0 is the ingest step:
  `scripts/run_hapmixqtl_from_salmon.py`'s `pair_haplotypes` pairs a base
  transcript id only when BOTH suffix rows exist, and `load_counts` accumulates
  `YL`/`YR` ONLY over paired transcripts, so an unpaired (homozygous,
  deduplicated) transcript contributes to NEITHER however many reads its
  surviving row carries. `YT`, by contrast, sums EVERY transcript
  unconditionally. This is exactly why `compute_summaries_from_gibbs` insists on
  a `yT` summed over ALL transcripts rather than `yL+yR`: `yL+yR` IS that
  paired-only subtotal, so using it as the total would silently zero these same
  pairs there too. A pair has `mL=mR=0` exactly when NONE of that gene's
  transcripts have a surviving heterozygous pair in that donor.

  Measured against the cache: `R` equals `mL+mR` to max|diff| 2.3e-10 over all
  3,170,044 datapoints; `R==0` for 1,831,718/3,170,044 = **57.8%**; `Va==0.0`
  exactly in 100% of those and in 0 of the remaining 1,338,326, an exact match
  to the `no_cov` guard. Of the zero-informative pairs, 68.0% have zero total
  expression too, but 22.7% — 416,207 pairs, **13.1% of ALL pairs** — are
  expressed with no allelic information whatsoever, including 89,639 pairs at
  1,000+ total reads. Independently reproduced against the separate `YT`
  per-draw array (67.9% / 22.7%), which pins "expressed" at >=10 total reads and
  leaves a third, unlabeled band: ~9.3% have trace expression (0-10 reads).
  These are homozygous-but-expressed pairs: real expression, zero allelic
  information, by construction, not a depth artifact. The total channel
  correctly retains them, so the 13.1% figure is total-channel-only BY
  CONSTRUCTION and is easy to misread as loss.

  DESIGN NOTE: Salmon's Gibbs resampling was meant to be self-downweighting — if
  two haplotypes were indistinguishable the draws should disagree, `v` should
  blow up and the weight should collapse on its own. That cannot fire here,
  because deduplication plus the pairing rule mean there is no second copy to be
  uncertain between; every draw has `yL=yR=0` by construction, not by chance.

  An expression-based gene filter and the allelic channel's informativeness
  filter are NOT nested: 2,799 genes pass a standard expression filter but fail
  a >=40-informative-donors rule, because a gene can be well expressed in every
  donor and heterozygous in few. Expression governs admission to the TOTAL
  channel; heterozygosity governs the ALLELIC channel; passing one says nothing
  about passing the other.

- **The permutation null permutes donor records, not residuals.**
  `perm_scheme='records'` is the default and the one to use: each donor's
  whitened phenotype value, weight and covariate row move together, genotypes
  stay, and the denominator is recomputed per permutation
  (`_record_permutation_channel`). By relabeling this equals the
  genotype-permutation null of FastQTL and tensorQTL with per-donor weights,
  pinned to 1e-9 by `tests/test_hapmixqtl_perm_scheme.py`. mixQTL permutes the
  phenotype bundle the same way. `perm_scheme='residuals'` is the earlier
  Freedman-Lane scheme (leverage-standardized whitened residuals permuted at
  fixed weights), retained but conservative where weights vary — most so at
  high expression. Cost 0.34 s vs 0.10 s per 1,000-permutation scan.

- **Gene-level Gibbs shape has bounded real-data evidence.** The three-library
  pilot found Gaussian competitive within 0.05 bits/draw for 98.51% of 4,500 ASE
  and 99.93% of total summaries, retaining two reproducible ASE candidates and
  no total candidates. ZNF529 and RNF175 are exceptions to interpret; RNF175
  restart-block occupancy is unreliable. Not association or calibration. See
  `/mnt/ssd/lalli/brainvar_hapmix_deploy/gibbs_shape_pilot_20260915/REPORT.md`.

- **The two-gene influence audit retained the moment/GPU baseline.** Across
  three genotype-selected common SNPs per gene and 92 donors (87 ASE), maximum
  one-block mean-plus-covariance shifts were 0.0736 working SE (ZNF529) and
  0.1396 (RNF175); leave-one-block maxima 0.00901 and 0.01028. Bounded
  sensitivity, not calibration. See
  `/mnt/ssd/lalli/brainvar_hapmix_deploy/gibbs_influence_audit_20260915/REPORT.md`.

- **Gibbs draw count is adequate to treat `v` as known.** 200 draws; lag-1
  autocorrelation median 0.081; median effective draws ~170; the relative sd of
  `v` as an estimate is median 0.109, with only 3.2% of datapoints above 0.25.
  Treating `v` as known rather than itself uncertain is a good approximation and
  errors-in-variables attenuation from doing so is small. Measured by
  `variance_layer_mapping_20260918/variance_layer_measurements.py`.

- **`v_ig` is a donor-by-gene interaction, so only a gene-by-sample weight
  matrix can hold it.** Within a gene across donors, log `v` has median sd 0.77
  (residual sd 0.557 at matched allele-resolved read count). Allele-resolved
  read count explains only R^2 = 0.32 of it — at matched read count `v` still
  spans 1.7-fold between donors of the same gene — so what remains once read
  COUNT is fixed is how INFORMATIVE those reads are, i.e. heterozygosity, not
  depth. The per-donor mean of read-count-adjusted log `v` has sd only 0.073
  across the 92 donors. So `v_ig` is neither a gene property (pooling across
  samples, as `catchSalmon` and sleuth do, cannot hold it) nor a donor property
  (a per-sample array-weight factor cannot either). The allelic channel carries
  about three times as much of its structure at the gene-sample level as the
  total channel: within-gene/between-gene median sd of log `v` is 0.41/2.12
  (ratio 0.19) for total, 0.72/1.29 (ratio 0.55) for allelic.

- **RTA overdispersion is at its floor at gene level, which independently
  confirms the shot-noise finding.** edgeR's read-to-transcript-ambiguity
  estimator on our own gene-level draws (3,000 genes): overdispersion quartiles
  1.00/1.00/1.01, 61.4% exactly at the floor of 1, 95th percentile 1.13, max
  35.6 — at gene level RTA has almost nothing to do, consistent with
  `catchSalmonGene` having arrived separately from transcript-level
  `catchSalmon`. Separately the across-draw variance of our log-total statistic
  is 0.98x (IQR 0.91-1.05) what an RTA-inflated Poisson predicts: an
  independent, differently-derived confirmation that Salmon's default Gamma draw
  carries shot noise. `variance_layer_mapping_20260918/rta_vs_c.py`.

- **A gene-by-sample quantification-uncertainty correction already exists in
  this project's RNA pipeline**, predating `catchSalmonGene`:
  `calc_expression_stats.R` (2025-05-02), run as a Nextflow process. Its
  `getGeneOverdispersion` adapts `catchSalmon` for personalized transcriptomes
  where the transcript set differs between samples: sums per-draw transcript
  counts to gene level within each sample, supports `merge_alleles`, keeps
  `OverDisp` as a gene-by-sample matrix moderated per sample (`colMedians`,
  `DFPrior=3`), then divides counts by it element-wise. hapmixQTL reaches the
  same information independently from the raw per-draw arrays, and the
  RTA-vs-Poisson finding says these are the same quantity on different scales,
  so the two are checkable against each other — not yet done.

- **RASQUAL's beta-binomial overdispersion rho is an available external check.**
  RASQUAL models the allelic count as beta-binomial (a binomial whose success
  probability is itself Beta-distributed across donors, giving extra-binomial
  variance `rho*p(1-p)`); `rho` is that extra-binomial fraction.
  `best_rasqual_row` in `scripts/compare_pipelines.py` currently reads
  1-indexed fields 3-6 (chrom/pos/ref/alt), 11 (chi2), 12 (pi), 14 (phi) and 23
  (convergence), plus field 2 to detect `SKIPPED`. RASQUAL's vendored
  `rasqual_src/README.md:64` lists field 15 as "Overdispersion"
  (repo-verified 2026-09-18) but that has not been cross-checked against parsed
  stdout in a run here; retaining that one field would enable the check.

- **The mixed-BLAS crash blocks any limma path that fits a linear model.** The
  BLAS is Debian's openblas-pthread `libblas.so.3` while the LAPACK is a
  Homebrew openblas. Tested function by function, each in its own R process, so
  one crash could not hide another. CRASHES: `limma::lmFit` (segfaults even
  UNWEIGHTED), `limma::arrayWeights`, and therefore `vooma`, `voomaLmFit` and
  `voomWithQualityWeights`. RUNS: the whole closed-form empirical-Bayes
  moderation family — `squeezeVar`, `fitFDist`, `fitFDistRobustly`,
  `fitFDistUnequalDF1`. Also `libR.so` links `libblas.so.3` and
  `libopenblas.so.0` together, so `lm()`, `%*%` and `crossprod()` segfault,
  `tensorA`/`glmnet` are absent, and there is no sudo — which is why **no
  cross-language check of the mixQTL port exists and none is possible here.**
  The port's algebra is validated per variant against `numpy.linalg.lstsq` to
  1e-10 and every cutoff/cap/dof rule is pinned to the R source line it
  encodes (23 tests). Exact reproduction of the published code is NOT claimed.
  Installed: limma 3.64.3, edgeR 4.6.3 (a release behind upstream devel).
  DEVEL ONLY, not installed: `catchSalmonGene`, `binQLFit`, `PCList`,
  `sampleWeights`.

## What the draws buy, measured (2026-09-19, mixQTL mode as the baseline)

Report `brainvar_hapmix_deploy/mixqtl_replication_20260919/REPORT.md`. Before
this, mixQTL had never actually been run: `compare_pipelines.py` is
hapmixQTL-vs-RASQUAL.

- **THE DRAWS DO IMPROVE THE POINT ESTIMATE.** Holding response, donor set,
  variants and design fixed and varying ONLY the weights, `1/v` weighting cuts
  `var(beta_hat)` across 40 null permutations to **0.340** of unweighted (25/29
  genes, sign p=1.0e-4) — 1.71x in SE, 2.9x in variance. It beats mixQTL's
  published capped harmonic weights (0.499, 23/29) and uncapped harmonic
  (0.763, 21/29). Kish effective n falls 77 -> 45 while variance drops 2.9x,
  which is why this is not a concentration artifact.
- **It replicates across seeds.** An independent master seed moves every arm by
  <= 0.024 in the ratio and the per-gene Gibbs/OLS ratio correlates at r=0.997
  over 29 genes; the gaps carrying the conclusion (0.18, 0.26) are 7-11x that.
  The one pair that does not separate is capped Gibbs 0.780 vs capped harmonic
  0.826 (gap 0.046, ~2x the shift): read those as close, not ordered.
  `compare_seeds.py`.
- **mixQTL's fold cap costs two thirds of the gain** (0.340 -> 0.780). It is a
  workaround for a known-variance SE, which is why hapmixQTL is NOT given one.
- **`count_noise`'s q is inert for weighting**: q-on vs q-off efficiency 0.3402
  vs 0.3410, a 0.2% difference. CAUTION on the sign test there: 21/29
  (p=0.024) at seed 42 and 19/29 (p=0.14) at the other. A sign test responds to
  DIRECTION regardless of magnitude, so a p crossing 0.05 between seeds while
  the effect stays at 0.2% means inert, not real. Second, independent
  confirmation of the 2026-09-15 double-count finding.
- **Gene-level calibration.** The mixQTL port through the CORRECTED permutation
  path rejects at 5% in 0.0483 of 290 gene-draw pairs (se 0.0126, median p
  0.4925), indistinguishable from uniform and replicating across seeds (0.0448,
  median 0.4950). The published path yields no number at all here.
- Two defects in the distributed reference, both reproduced under
  `strict_reference_cap=True` and documented in the module docstring:
  `matrix_ls_asc_permutation` zeroes cutoff-failing weights before taking the
  min for the fold cap, so ANY cutoff failure zeroes every weight and all
  permuted betas are 0/0 (the non-permutation path subsets first and escapes);
  and `floor(n/10)` makes the cap 0 for 3-9 passing samples (past the
  `sample_size > 2` guard) and 1 for 10-19, where the channel becomes plain OLS.
- **End-to-end on the 29 genes is the weak half, and how weak depends on the
  cutoffs.** Under the PUBLISHED cutoffs: lead agreement among the 6 called
  genes 0/6, median lead LD r^2 0.160, Spearman of the per-gene statistic 0.183
  (p=0.34), beta Pearson r 0.562. Under permissive R-signature cutoffs
  (`20/5/100/5000`), kept as a sensitivity arm: 1/6, r^2 0.744, Spearman 0.517
  (p=0.004), r 0.697. The collapse has an identified cause: mixQTL's upper cap
  `y <= 1000`, stated in its Methods as an alignment-artifact guard, removes
  1,656 of 2,193 informative donor-gene pairs when applied to Salmon posterior
  means (the lower `y >= 50` removes only 38), leaving 18 of 29 genes
  total-counts-only and 7 with no allelic donors at all. At those cutoffs the
  arms are largely not measuring the same thing, so the permissive column is
  the better comparison of the two METHODS and the published column the better
  comparison against mixQTL AS RUN. A signal-bearing gene set is needed to
  sharpen either. Scope: these are high-coverage genes, 17.8% uninformative
  donor-gene pairs against 57.8% transcriptome-wide, so the 3x is not shown to
  transfer to low-count genes where the Gibbs and Poisson weights converge.
- The weighting ablation and the SE calibration are BYTE-IDENTICAL under the
  two cutoff settings, verified by re-running and diffing 2026-09-20. They run
  on hapmixQTL's informative-donor set (40-92 donors/gene), which mixQTL's count
  cutoffs never reach, and the only cutoff parameter they touch is `weight_cap`
  through `cap = min(weight_cap, floor(n/10))`; at <= 92 donors
  `floor(n/10) <= 9`, strictly below both candidate values, so realized caps are
  4-9 and `weight_cap` can never bind below 100 donors.

## hapmixQTL can apply mixQTL's count cutoffs (2026-09-20)

`count_cutoff_masks(yL, yR, yT, asc_cutoff, asc_cap, trc_cutoff)` builds
per-channel boolean admission masks from posterior-mean counts; `map_nominal`,
`map_cis` and `map_susie` take them as `keep_a_df`/`keep_t_df`, and the runner
exposes `--asc-cutoff --asc-cap --trc-cutoff` plus `--mixqtl-cutoffs`. The point
is a MATCHED-DONOR comparison between the two modes: the non-weighting ladder
found the donor set — not the response, not the weights — to be the dominant
non-weighting difference between the estimators.

- **Default is off and must stay off.** All four parameters default to None and
  an all-True mask reproduces the unmasked run bit for bit
  (`TestCountCutoffsEndToEnd`). The published cutoffs discard 1,656 of 2,193
  informative donor-gene pairs on the calibration genes, so this is a comparison
  instrument, not a production setting.
- **The mask is applied by zeroing the WORKING inferential variance** in
  `_prepare_channels`, putting an excluded donor in exactly the state a
  zero-coverage donor is already in. Every informative-set test downstream is
  `v > eps`, so one assignment keeps everything consistent with the weights.
  `Va_df`/`Vt_df` as the caller passed them are untouched.
- **`trc_cutoff` reads `yT`, never `yL + yR`, and this is a real trap.**
  `yL + yR` is the paired-transcript subtotal, exactly 0 for a
  homozygous-but-expressed donor. Thresholding `yL + yR` at 100 excludes 502
  donor-gene pairs that `yT >= 100` admits, 18.8% of the cohort, all of them
  good total-channel data. A test pins it.
- Measured retention on the 29 calibration genes (2,668 pairs, 2,193
  informative): the published allelic band `[50, 1000]` on both haplotypes keeps
  499; `trc_cutoff = 100` on `yT` excludes nobody, so the total-channel mask is
  inert on this high-coverage set.
- Not wired: `weight_cap`. hapmixQTL is not given mixQTL's fold cap, because
  capping costs two thirds of the efficiency the Gibbs weights buy.

## Default mode holds on non-circular ground truth (2026-09-23)

`tests/ase_external_benchmark.py` against the RASQUAL/TReCASE generative model,
which hapmixQTL does not assume. 500 replicates, N=200, mu=200, NB dispersion
0.2, BB overdispersion 0.01, allele-specific fraction 0.25. Report:
`brainvar_hapmix_deploy/external_benchmark_fitted_defaults_20260923/`.

- **Calibrated at nominal 0.05 and at parity with the generating model's own
  joint likelihood.** Type-I 0.0640 (lambda_GC 1.24), 1.4 Monte Carlo standard
  errors above nominal (se 0.0097). Matched power — each arm held to its own
  empirical 95th percentile — is 0.248/0.744/1.000 at allelic fold change
  1.05/1.10/1.20 against TReCASE's 0.274/0.760/0.998: differences
  -0.026/-0.016/+0.002 against paired se 0.028/0.027/0.002, so |z| <= 1.0
  throughout. TReCASE is the joint likelihood OF THE GENERATING MODEL, so this
  is parity with a ceiling, not a peer. Both dominate the single-channel arms
  (TReC-only 0.104/0.204/0.486; ASE-only 0.160/0.558/0.994, the latter
  genuinely anticonservative at type-I 0.158).
- **The bound is in the tail.** At nominal 0.01 default mode reads 0.0200 — 2.0x
  nominal, 2.2 Monte Carlo se above it (se 0.0045) — against TReCASE's 0.0120.
  Three runs put nominal-0.05 type-I near 1.3x (0.064 at 500 reps, 0.073 and
  0.067 at 150), so this is a pattern, not noise. NOTHING below nominal 0.01
  was tested and transcriptome-scale thresholds are far below it. It runs
  OPPOSITE to the Beta approximation's known tail conservatism, which concerns
  the gene-level permutation p rather than the nominal p, so the two must not be
  netted against each other.
- **It answers the circularity objection at this design point, and only there.**
  Fitting the residual scale from the residuals it scales did NOT produce
  anticonservatism at this N and depth. The objection is structural, so one
  design point is not a general refutation.
- **The harness fabricates the total channel's inferential variance, and the
  consequence for default mode was measured rather than argued.**
  `hapmix_pval` emulates draws as `yL ~ Binomial(n, frac)` with `yR = n - yL`,
  so `yL+yR` is EXACTLY constant across draws and the total channel's
  across-draw variance is zero by construction; `compute_summaries_from_gibbs`
  is called without `yT`. Only the Poisson term `1/(m_T + 2*kappa)` survives,
  computed on the ALLELE-SPECIFIC total (median 48 reads) while the phenotype
  comes from true totals (median 195.5): measured median `Vt` 0.0204082 against
  median `1/(m_T+2*kappa)` 0.0204082, identical to every printed digit. The
  delta-method variance of the harness's own phenotype, `T/(T+lib)^2`, has
  median 0.00507, so the harness **overstates the total channel's inferential
  variance by 4.0x**. Substituting it over 150 replicates moves default mode
  0.073 -> 0.067 in type-I and 0.733 -> 0.760 in matched power at 1.10, both
  inside the 150-rep Monte Carlo floor (~0.018, ~0.038), because a fitted
  `sigma^2` absorbs a wrong absolute scale on `v` by construction. TReCASE is
  unchanged to three decimals, the required internal control since it never
  sees `v`. SCOPE: this demonstrates robustness to a UNIFORM scale error only.
  It does NOT test a SHAPE error — a few samples mis-weighted against the rest —
  which is what the open total-channel zero-count defect produces and which no
  global scale can repair. Do not generalise to "robust to errors in `v`".
- **The harness reseeds per replicate**, data from `RandomState(seed0+r)` and
  each arm its own `RandomState(seed0+900000+r)`, so arm count and order cannot
  perturb any result. That is what makes reporting a subset of arms exact with
  nothing re-run. Checked before being relied on.
- **`lambda_GC` from this harness is CENSORED at 3019.92**, because `calib()`
  clips p to 1e-300 and `chi2.isf(1e-300,1)/chi2.ppf(0.5,1) = 3019.92`. Default
  mode is nowhere near it, but any lambda from this harness near that value is a
  floor, not an estimate. A figure of "3020" reported from it twice, on
  different data, was one ceiling reached twice rather than two agreeing
  measurements — see `docs/ase_validation.md` sec 7d, whose real-data finding is
  withdrawn for the separate reason below.

## Claims withdrawn — do not re-assert

**2026-09-13** (an eight-angle review retired these; they may survive in older text):

- The additive, multiplicative and nested variance forms were **never** compared:
  no multiplicative arm existed in `tests/`.
- `docs/ase_validation.md` sec 7b's weight-cap and nested conclusions are void;
  both arms were arithmetically incapable of differing from their comparators.
- The additive `tau` is **not** "the one place hapmixQTL departs from mixQTL".

Corrected facts that replaced them (these are true; do not negate them):
mixQTL's total channel already carries a flat additive variance, so `v_t + tau_t`
decomposed the parent rather than departing from it, and the multiplicative
scale is the allele-specific channel only. mixQTL's scan refits its dispersion
at EVERY variant. mixQTL's allelic regression is also through the origin
(`y ~ -1 + x`).

**2026-09-16:** the stated reasons for `count_noise=True` — "Gibbs across-draw
variance is read-assignment uncertainty only" (false: Salmon's default Gamma
draw carries shot noise, `CollapsedGibbsSampler.cpp:122`) and "a sample with
unambiguous reads has unanimous draws and would be discarded" (impossible under
default flags: identical draws occur only with zero reads). The flag stays on
for the missing-floor reason above. Also withdrawn: that `count_noise=False`
drops zero-read samples from the total channel (it does not; the guard is
ASE-only).

**2026-09-23:** `docs/ase_validation.md` sec 7d's finding 1, "the defect is
confirmed on real data, at full severity", is withdrawn as INDEPENDENT
real-data corroboration. `tests/ase_gtex_real_data.py` has the same fabricated
total channel as the external benchmark: line 79 sets `YR = n_i - YL` so
`YL+YR` is exactly constant across draws, and line 124 calls
`compute_summaries_from_gibbs` WITHOUT `yT`. Its phenotype is real total
expression but the variance attached to it measures nothing. That is the
benchmark's mechanism reproduced on real allele counts, not corroborated by
them. What sec 7d DOES still establish: its ALLELIC channel carries genuine
GTEx overdispersion, depth and zero-inflation structure.

## Known and unfixed

- The lead refit is biased by selection: appending the window maximum to the
  design removes far more residual sum of squares than the one degree of freedom
  it is charged. Gene-level p-values are unaffected.
- The Beta approximation is conservative in the tail, costing power at
  transcriptome-scale thresholds. Note this is the OPPOSITE direction to the
  nominal-p inflation measured above; do not net them.
- No per-sample allele-specific read floor, where mixQTL used 15 reads.
- The total channel has no zero-count guard; `count_noise` stands in for a
  floor. A coverage-based floor for zero-count total samples, then no `q` for
  samples with reads, is the fix — not the flag.
- `tests/ase_external_benchmark.py` and `tests/ase_gtex_real_data.py` both
  fabricate the total channel's inferential variance (emulated draws conserve
  `yL+yR` exactly; `compute_summaries_from_gibbs` called without `yT`). Fix is
  to pass a `yT` built from the real/simulated totals. Both also keep an
  intercept on the allelic residualizer, where production has been
  through-origin since 2026-09-15.
- `calib()` in those harnesses reports a censored `lambda_GC` (saturates at
  3019.92). A censoring flag alongside it would stop the number being read as a
  magnitude, as it was until 2026-09-23.
- Three input-validation defects from the 2026-09-14 audit: a NaN at a
  zero-weight sample drives `pval_perm` to the `1/(nperm+1)` floor; variant-row
  identity between the genotype and phase frames is not checked beyond column
  order; the QR is taken on a rank-deficient design without a rank check.
- After the through-origin change, `_joint_gls`/`_pvals` (the robust
  second-pass path) still charge the ASE channel `1 + n_cov` columns, so the
  robust SE there is ~6% conservative. `map_cis`/`map_nominal` are unaffected.
- The log2 migration is pending; outputs are still natural logs.
- `run_second_pass` and `map_susie` have not been re-verified under default mode
  as carefully as `map_cis`/`map_nominal`.
- RASQUAL agreement has NOT been re-measured under default mode. RASQUAL is now
  built from source and self-validated against the authors' bundled example, so
  the old blocker is cleared. The historical baseline is in the quarantine and
  was measured under a deprecated configuration, so it is not a valid
  comparison; a fresh run is needed.

## Self-tests

```bash
# the hapmixQTL/mixQTL surface: 175 tests, all passing as of 2026-09-23
pytest tests/test_hapmixqtl.py tests/test_hapmixqtl_calibration.py \
       tests/test_hapmixqtl_perm_scheme.py tests/test_fitted_variance_quarantine.py \
       tests/fitted_variance/ tests/test_cli.py tests/test_mixqtl_replication.py -q

python3 scripts/run_hapmixqtl_from_salmon.py --selftest
RASQUAL_BIN=rasqual_src/src/rasqual python3 scripts/compare_pipelines.py --selftest
```

**Four test files carry 46 pre-existing failures from upstream fork drift and
are not ours:** `tests/test_post.py` and `tests/test_trans.py` (22 between them)
and `tests/test_genotypeio.py` and `tests/test_integration.py` (24). **None of
the four contains a single reference to hapmixqtl** — verified, not assumed —
and their failures are BED sort order, a missing `chr` column and a missing
`pval_beta` column. A bare `pytest tests/` therefore reads 46 failed / 223
passed and that is the expected state; do not read it as a hapmixQTL
regression. Run the command above to see the surface this fork owns.

`tests/test_fitted_variance_quarantine.py` pins that a default-mode run never
imports the deprecated module, including a full `map_cis` checked in a
subprocess.

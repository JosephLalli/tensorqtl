# Simulation benchmark for hapmixQTL: implementation specification

> **SUPERSEDED 2026-09-26 (user decision): the Salmon emulator described
> below will not be built.** Rerunning or emulating Salmon per dataset was
> judged too slow. The benchmark that exists instead builds datasets from the
> cohort's own Salmon point estimates and Gibbs draws: donor records are
> permuted against fixed genotypes with a random L/R swap, and known cis
> effects are injected by binomial thinning of the haplotype carrying the
> lower-expressed allele. Code, design and checks: `scripts/plasmode/README.md`
> and the numbered scripts it describes (run order in `run_all.sh`); the
> scripts this box first named were replaced by that numbered pipeline on
> 2026-09-27 (commit fc238df). Results, `<root>/report/plasmode_report.html` under
> `/mnt/ssd/lalli/brainvar_hapmix_deploy/`: `plasmode_meier_20260927` (deep
> set) and `plasmode_lowcov_meier_20260927` (low-coverage set) on the current
> library; `plasmode_20260926` and `plasmode_stratum30_100_20260927` are the
> earlier runs, made before Meier's correction of the combined standard error
> (its first-order inflation for channel weights estimated from the same
> residuals they combine; commit a1b2ef4).
> The calibration measurements of the real data in the appendices below remain
> valid as measurements; the emulator layers and the build plan do not apply.
> In particular the external-benchmark tier's plan to freeze `hapmix_pval` was
> not followed: on 2026-09-28 (commit 9369bb1) `hapmix_pval` itself was fixed
> and now runs default mode, and the three harnesses that used its removed
> arm stop (record `external_benchmark_current_20260928/`).
>
> **2026-09-29 historical-label boundary.** `split` remains the recorded
> `log2(CPM+1)` arm and `gibbs (shipped)` means the implementation shipped at
> that run's date. Neither is the current half-read association default; this
> benchmark does not establish uniform superiority or SuSiE calibration.

Date: 2026-09-26, revised the same day after two review passes. Status:
specification only; nothing in it is implemented.
Audience: the engineer who builds the simulator and the benchmark driver.
Pinned pipeline: commit `ac11b79f6c146f9a316d6b409ed29ac00d5e129b` on
`mixqtl-replication` (section 2.1), retrievable from origin as branch
`simulation-benchmark`.

This document turns three positions into one buildable design:

- **A, the user's design document** "Simulating benchmarks for total + ASE
  cis-QTL methods" (2026-09-26). It defines Layers 0-6 (real-cohort
  scaffold; architecture families; haplotype expression
  `log mu = log theta0 + z_i + b_ih + eps_ih`; reads with an emergent
  allele-informative fraction; a Gibbs-sampler emulator; injected failure
  modes; truth and evaluation) and describes `tests/hapmix_simulator.py` with
  14 tests in `tests/test_hapmix_simulator.py`. The document and both files
  were delivered in chat and are not on this machine: a filesystem-wide search
  found neither file, and no commit on any branch carries either path.
  Section 3 therefore specifies every interface in full; the attachments, if
  they arrive, are reconciled to it (section 3, "Status of the interfaces").
- **B, the critique** from the analysis session that verified the design
  against the current pipeline.
- **C, the user's reply to that critique.** It is accepted except where a
  verified fact contradicts it; every such point is named in Appendix A.

Where the three disagree, Appendix A records the disagreement, the resolution
and the measurement or code that settled it. Section 3 carries every
parameter with its default and its source; Appendix B lists every parameter
whose default is not yet calibrated, with the measurement that would set it.

**Conventions.** Every expression value, allelic ratio, effect size and truth
quantity is in log2 units (project convention since 2026-09-15; beta = 1 is a
twofold effect). Two scales carry log2 units and are kept apart: the model's
log2 CPM, and the pipeline's log2(CPM + 1) (section 3.4, "Scale"). The one
exception to log2 is donor variance multipliers, which were measured as
effects on the natural log of squared residuals and are stated that way. A
"record" is one donor's data for one gene in one channel; a "donor-gene pair"
is one donor and one gene. The across-draw variance of Salmon's posterior
samples is the "Gibbs variance" and the samples are "Gibbs draws". All
randomness comes from one master `SEED = 42` (section 3.1). Metrics (section
5.4), acceptance tests (section 7) and structural requirements (section 3.0)
are named, not numbered, and are cited by name; decisions are cited by name
with their section 9 number.

---

## 1. Purpose and the decisions it must inform

### 1.1 Why a simulator is needed

The real-data instruments built so far (the stored permutation nulls, the
weighting nulls on 100 genes, the phASER comparisons) measure one thing well:
whether a nominal p-value is uniform when donor records are permuted. Four
questions stay open because real data carry no truth:

1. **Efficiency against a known optimum.** The best weight for a record is
   the inverse of its true error variance, which is unknown. The 2026-09-26
   variance-family test (`scripts/variance_family_test.py`,
   `brainvar_hapmix_deploy/variance_family_test_20260926/`) rejected every
   candidate variance function on the bin-level criterion (chi-square at
   least 43 on 8 degrees of freedom; at least 53 among the main
   observed-`v` and fitted-value-`v` analyses, `bin_gls.tsv`), so no
   real-data fit can stand in for the truth.
2. **Which permutation null is the right reference.** Holding the genotype
   principal components (PCs) with the genotypes moved the total channel's
   rejection rate at 0.05 from 0.0662 to 0.0835 on the corrected pipeline
   (`docs/pipeline_rules.md`, "What made the total channel worse"). Real data
   cannot say which of the two nulls matches the sampling distribution of the
   observed statistic.
3. **Power and false discovery at a known truth.**
4. **The cost of excluding records whose truth is unknown.** 8.1% of the
   zero-haplotype pairs that phASER can score are clearly imbalanced by
   phASER, so dropping them discards some real signal; how much it matters
   for power and bias cannot be measured without truth.

A simulator whose truth is known, whose quantification step reproduces
Salmon's behaviour on this cohort, and whose output passes through the
unmodified pinned pipeline can measure all four, provided it also reproduces
the real-data failures that motivate them. Section 7 therefore separates
correctness tests (the simulator does what this document says) from realism
tests (the simulator reproduces the real null and the real variance profile);
a result informs a decision only when both kinds pass.

### 1.2 The decisions it must inform

| Decision | Where it is recorded as open | What the benchmark measures for it | What it cannot settle |
|---|---|---|---|
| Which weighting configuration ships (section 9.4) | `docs/pipeline_rules.md`, "Open decision: which weighting configuration ships" | Efficiency relative to oracle weights, standard-error accuracy, nominal-p calibration per channel, weight-residual coupling, power at empirical FDR, for every weighting arm of section 5.2 | Whether the simulator's variance generator is BrainVar's. Every result is conditional on the correctness tests of section 7, and this decision is informed only if the realism tests *Real-null reproduction* and *Variance-profile reproduction* also pass. On real data the shipped weights fail through a mechanism whose generative source is not identified (the within-gene coupling of Gibbs weight and whitened residual, whose across-gene spread is 1.75 times its model null; `brainvar_hapmix_deploy/weight_residual_coupling_20260925/`). A simulator built only from Salmon's own likelihood and a Gaussian per-gene biological term may not contain it; if the realism tests fail, the benchmark says so and does not inform this decision until the mechanism is added |
| Zero-haplotype handling (section 9.3; keep or drop pairs whose point estimate puts one haplotype at exactly zero) | `docs/pipeline_rules.md`, "Open decision: Salmon point estimates put one haplotype at exactly zero" | Bias, calibration and efficiency with the pairs kept and dropped, including simulated monoallelic genes where dropping discards true signal | Re-quantification with Salmon's `--useEM`, which needs real Salmon runs (tier 3, deferred). Conclusions hold for the calibration-gene universe only (section 3.2) |
| The genotype-PC permutation rule (section 9.6; genotype PCs held with the genotypes, or moving with the RNA record) | `docs/pipeline_rules.md`, "Open decision: the genotype-PC permutation rule interacts with the weights" | Permutation-null rejection rates under each rule against the sampling-null rate, which is the truth, crossed with the weightings `gibbs_both`, `split`, `unit_both` and `oracle` and with an ancestry-correlated expression term of share 0 and 0.04 (section 5.2, genotype-PC grid). The cross is required because the real tie hurts only through the Gibbs total weights (unit total weights are nominal at every level, `docs/pipeline_rules.md`) | The real size of the ancestry term, which is bracketed at 0 to 0.04 of RNA-tied residual variance (section 3.4) |
| The counting term for donors with reads (section 9.5; `count_noise`, the counting term added to the Gibbs variance) | `docs/pipeline_rules.md`, rule 6; `brainvar_hapmix_deploy/salmon_gibbs_counting_sim_20260915/REPORT.md` | Standard-error accuracy and efficiency with the total counting term on every donor, or only on zero-read total donors, under the two weightings whose total weights use `Vt` (`gibbs_both`, `plus_one`); separately, the allelic counting term removed (section 5.2, counting-term grid) | The floor for zero-read total donors is exercised only in the zero-read stratum of the gene sample (section 3.2), which is expected to hold about 100 such pairs (30 of the 121 genes that carry the 404 real ones), so its power there is limited |
| Fine-mapping (SuSiE) and knockoff eGene false-discovery rate, eventually (section 9.7) | `tensorqtl/hapmixqtl.py:2452` (`map_susie`) and `:2690` (`fine_mapping_provenance`); section 2.3 | Nothing yet: `map_susie` cannot reach default mode (section 2.3), so the evaluation is blocked on a design decision, not deferred by choice | Everything, until that decision is made |

SuSiE ("sum of single effects") is the fine-mapping regression in
`tensorqtl/susie.py`; it models a gene's cis signal as a small number of
single-variant effects and reports posterior inclusion probabilities and
credible sets. Knockoffs (`tensorqtl/knockoffs.py`) are synthetic copies of
the variants that preserve their correlation structure but are independent of
the phenotype by construction; comparing each variant's importance with its
copy's gives an empirical false-discovery estimate.

The 2026-09-25 user authorization to explore a biological variance term set
four pre-registered criteria for any candidate
(`brainvar_hapmix_deploy/nominal_p_hypotheses_20260925/`): pooled rejection
rates within the gene-clustered interval of nominal on both the records null
and a sampling null; per-gene coupling `R_g` in band for all but about 1 of
the 46 real instrument genes; at least 70% of the `1/v` efficiency gain over
unit weights retained; and parity with TReCASE (the joint
total-plus-allele-specific likelihood of Sun 2012) on the external benchmark
once its total channel is repaired. This benchmark supplies the sampling null
(metric *Nominal-p calibration*), the per-gene coupling on simulated genes
(metric *Weight-residual coupling*, whose tested-set analogue of the 46-gene
count is defined there), the efficiency share (metric *Efficiency against the
oracle*) and the repaired external benchmark (section 6.2).

### 1.3 What the benchmark is not

It is not a comparison of RASQUAL or TReCASE on real data, not a substitute
for observed-data calibration, and not valid for any pipeline commit other
than the one it records (section 2.1). Out of scope, each for a stated
reason: design A's single-nucleus ATAC extension and its open question on
pangenome alignment (neither touches the pinned pipeline); dynamic (age- or
sex-interaction) eQTLs, because the pipeline has no interaction test
(`tensorqtl/hapmixqtl.py` fits no genotype-by-covariate term); structural
variants, because the personalized index was built from a HaplotypeCaller
call set, which carries none.

---

## 2. The target pipeline

### 2.1 The pin

The benchmark targets commit `ac11b79f6c146f9a316d6b409ed29ac00d5e129b` on
branch `mixqtl-replication` ("Record the weighting null on
reference-dependent and high-effect high-se genes", 2026-09-26). That commit
is retrievable from origin as branch `simulation-benchmark`, pushed
2026-09-26 14:31:15 (the remote-tracking reflog records "update by push").
At the time of writing, `mixqtl-replication` was 30 commits ahead of
`origin/mixqtl-replication` and 0 behind; `summaries_from_point_estimates`
was introduced by `89bed4e`, one of those local-only commits (the oldest is
`d3248f0`). Simulator work goes on `simulation-benchmark` (section 9.1).

The three calibration scripts whose measurements set this document's defaults,
`scripts/simulator_layer2_calibration.py`,
`scripts/simulator_layer4_calibration.py` and `scripts/variance_family_test.py`,
were committed with this document in `d9a9f16`, which descends from the pin
and changes no pipeline file. The implementer builds against the tip that
contains them and:

- records `git rev-parse HEAD`, and the blob hashes of
  `tensorqtl/hapmixqtl.py` and `tensorqtl/mixqtl_replication.py`, in a
  `provenance.json` beside every output;
- makes the driver refuse to run when either blob differs from its hash at the
  pin, unless `--allow-unpinned` is passed, in which case `provenance.json`
  says so.

### 2.2 Functions the simulator must call

The simulator produces inputs; the pipeline computes everything the pipeline
computes. No pipeline arithmetic is re-implemented in the simulator.

| Step | Function (file:line at the pin) | How it is called |
|---|---|---|
| Values and variances | `tensorqtl/hapmixqtl.py:429` `summaries_from_point_estimates(pL, pR, pT, eff_lib_size, yL, yR, yT, kappa=0.5, count_noise=True)` | Always with the simulated `yT`; `pL`/`pR` summed over paired transcripts only, `pT`/`yT` over every transcript of the gene (section 4). Returns `A, T, Va, Vt, Cat`, all `[genes, donors]`. The counting terms are at lines 485-488: `q_a = (1/(pL+kappa) + 1/(pR+kappa)) / ln2^2`, `q_t = k^2 (pT+1/2) / ((k (pT+1/2) + 1)^2 ln2^2)` with `k = 1e6 / eff_lib` |
| Reference-bias gate | `tensorqtl/hapmixqtl.py:203` `reference_bias_diagnostic(yL, yR, sign, min_total=10, min_sites=20)`, called by the runner at `scripts/run_hapmixqtl_from_salmon.py:1358` on the point estimates `pL`, `pR` with the per-gene orientation from `gene_orientation` (`:486`) | Run on every simulated dataset exactly as the runner does (without per-site allelic counts, as the runner does when `--allelic-counts` is absent). The runner refuses to proceed when `flag` is set unless `--force` is given; the driver always proceeds (the `--force` behaviour), records `ref_fraction`, `implied_phi`, `pvalue` and `flag`, and reports the gate's sensitivity and specificity (section 5.4, *Nominal-p calibration*, and the reference-bias process of section 3.7) |
| Nominal scan | `tensorqtl/hapmixqtl.py:1695` `map_nominal(...)` | `tau_mode='zero'`, `se_mode='fitted'`, `ase_covariates_df=None` (through-origin allelic channel), `covariates_df` = RNA-tied covariates, `genotype_covariates_df` = genotype PCs (or `None` in the "moving" arm, section 5.2), `window=1_000_000`, `keep_a_df`/`keep_t_df` as the arm requires. It returns `None` and writes `<output_dir>/<prefix>.hapmixqtl_pairs.<chr>.parquet` per chromosome. Read back as `scripts/corrected_null_store.py` `draw()` does: a scratch directory per call, emptied first; concatenate the parquet files; cast `variant_id` to `str`; inner-merge onto the tested-pair list |
| Gene-level scan | `tensorqtl/hapmixqtl.py:2053` `map_cis(...)` | As the runner calls it (`scripts/run_hapmixqtl_from_salmon.py:1400`): `tau_refit=True`, `perm_scheme='records_signflip'`, `tau_mode='zero'`, `se_mode='fitted'`, plus `seed=42` passed explicitly (the runner passes none). `nperm=1000` for the benchmark (production uses 10,000; section 9.15). Stream sharing with the stored permutations holds only at `nperm == 1000` (section 2.4, item 1) |
| Count cutoffs (matched-donor arm only) | `tensorqtl/hapmixqtl.py:526` `count_cutoff_masks(yL, yR, yT, asc_cutoff, asc_cap, trc_cutoff)` | With the point estimates `pL, pR, pT` (it accepts `[genes, donors]` arrays), always passing the total; its default total is `yL + yR`, which is wrong for homozygous-but-expressed donors |
| Library normalization | `scripts/run_hapmixqtl_from_salmon.py:975` `edger_normalize(totals_all, restrict, out_dir)` and `:992` `read_edger_dir(edger_dir, samples)`, which run `scripts/edger_library_normalization.R` | `restrict` = the real `cache/gibbs_56b63c3b37ed5df8/point_estimates/edger/calibration_genes.txt` (12,955 genes). `totals_all` = the simulated point-estimate totals of every calibration gene (tested plus background, including the 1,208 without draws) plus one rest-of-library row per donor (section 3.1). edgeR is installed (4.6.3) and this path fits no linear model, so the mixed-BLAS crash does not reach it |
| Expression PCs | `scripts/build_covariates.py:220` `expression_pcs_log2cpm(totals_all, eff_lib, genes, base, n_pc=10)` | On simulated totals, with `genes` = the fixed real calibration list (not edgeR's per-replicate kept set) and `base` = metadata plus genotype PCs, so the rebuilt PCs are orthogonal to the genotype PCs exactly as the pipeline's are |
| mixQTL mode | `tensorqtl/mixqtl_replication.py:190` `inputs_from_point_estimates(pL, pR, pT)`, `:577` `mixqtl_scan(...)`, `:618` `mixqtl_permutation_scan(..., genotype_covariates=...)` | On simulated point estimates only; never on draws (pipeline rule 5). Both functions work per gene (`y1`, `y2`, `ytotal` of shape `[samples]`; `h1`, `h2` of shape `[samples, P]`), so the driver loops over genes. `mixqtl_permutation_scan` returns only the per-permutation maximum `|stat|`, so the driver computes mixQTL's empirical gene-level p as `(1 + #{perm >= obs}) / (1 + nperm)`, with an optional Beta fit, under mixQTL's own permutation (no haplotype-label swap). mixQTL's slopes and standard errors are on the natural-log scale (pipeline rule 2); the driver divides both by ln 2 before any comparison |
| Haplotype pairing rule (reference behaviour, not called on files) | `scripts/run_hapmixqtl_from_salmon.py:248` `pair_haplotypes`, `:1039` `load_counts`, `:920` `load_point_estimates` | The in-memory simulator reproduces their summing rules exactly (section 4); if it ever writes Salmon-shaped files, it must round-trip through these readers |

TMM ("trimmed mean of M-values") is edgeR's library normalization factor: it
scales each library so that most genes' log ratios to a reference library
centre on zero. `filterByExpr` is edgeR's expression filter (CPM at least
10 / median library size in millions in at least `10 + 0.7 (n - 10)` samples,
and a total count of at least 15). Both are applied exactly as
`docs/pipeline_rules.md` rule 2 describes: `filterByExpr` computes CPM on the
library sizes of the whole `DGEList`, which in the real run held every gene;
the rest-of-library row reproduces that denominator and is then removed by
the restriction. The driver reports, per replicate, how many calibration
genes `filterByExpr` dropped.

### 2.3 Functions the simulator must not call, and why

- `compute_summaries_from_gibbs` (`hapmixqtl.py:349`): the pre-2026-09-25
  natural-log path, kept only so dated scripts reproduce.
- `map_susie` (`hapmixqtl.py:2452`): it takes no `se_mode`, its call to
  `_prepare_channels` omits `fitted_scale`, and its defaults are the
  deprecated `tau_mode='estimate'` and `variance_model='additive'`. It cannot
  reach default mode, so any fine-mapping or knockoff result from it would
  describe a quarantined configuration.
- Anything in `tensorqtl/fitted_variance.py`, and `hapmix_pval` in
  `tests/ase_external_benchmark.py` (section 6.2 lists its defects).

### 2.4 Pipeline properties the simulator must respect

1. **`map_cis` reseeds and consumes the global NumPy generator**
   (`np.random.seed(seed)`, then `nperm` calls to `np.random.permutation`,
   then `np.random.randint` for the haplotype-label flips;
   `hapmixqtl.py:2190-2204`). The simulator must never touch `np.random.*`;
   it uses local generators only (section 3.1). Because the flips are drawn
   only after all `nperm` permutations, the stream equals the stored one in
   `brainvar_hapmix_deploy/protein_coding_null_store_20260925/permutations.npz`
   (1,000 permutations, then a 1,000 x 92 flip matrix, as
   `scripts/corrected_null_store.py` regenerates and checks) only when
   `nperm == 1000` exactly: at any other `nperm` the permutations agree row
   for row up to `min(nperm, 1000)` but the flips do not (this follows from the draw
   order at `hapmixqtl.py:2190-2204`, and a review-time run at `nperm = 200`
   confirmed it). The driver refuses any other `nperm` in a result that
   claims to share the stored stream. The external permutation null of the
   metric *Nominal-p calibration* takes `perms[:200]` and `flips[:200]` from
   `permutations.npz`, as `corrected_null_store.py` does.
2. **Sample order is positional.** Every frame is indexed by the same donor
   order; `_assert_phase_columns` and `_assert_keep_frames` enforce it.
3. **`Va == 0` excludes a record from the allelic channel.**
   `_zero_degenerate_ase_weights` (`hapmixqtl.py:181`) zeroes the allelic
   weight where `Va <= 1e-12`; `summaries_from_point_estimates` sets `Va = 0`
   where `pL + pR <= 0`.
4. **The total channel has no zero guard.** The total weight is
   `1 / clamp(Vt, 1e-8)`, so `Vt == 0` gives a weight of 1e8. `Vt` from the
   draws alone is exactly 0 wherever every draw of the total is identical:
   among the 404 calibration donor-gene pairs with `pT < 0.5`, 19% have
   draws-only `Vt` of exactly 0 (10th percentile 0). Any arm that removes the
   counting term must keep a floor wherever the draws-only `Vt` is zero
   (section 5.2, arm `readless_floor_t`), and total-channel exclusion must go
   through `keep_t_df`, never through `Vt`.
5. **Sparse-channel rule.** A channel switches off when its informative donors
   (`v > 1e-12`) number fewer than intercept + covariates + 2
   (`_min_informative`, `hapmixqtl.py:803`).
6. **A switched-off channel reports p = 1, not a missing value.** Where a
   channel's standard error is infinite or zero (the channel is off by the
   sparse rule, or the variant has no heterozygous informative donor),
   `map_nominal` sets that channel's t to 0 and writes a finite p = 1
   (`hapmixqtl.py:1988-1991`); the combined standard error is infinite only
   when both channels are off (`hapmixqtl.py:1188-1209`). The metric
   *Nominal-p calibration* accounts for this explicitly.
   SUPERSEDED IN PART 2026-09-27 (commit 8a06803): in default mode a
   switched-off channel's p is NaN (its `dof_a` or `dof_t` is NaN), and the
   combined p is NaN where neither channel carries weight (`docs/outputs.md`).
7. **One reference distribution for all three p-values.** `pval_a`, `pval_t`
   and `pval_nominal` are all referred to `t` with
   `dof = N - 2 - max(n_cov, n_cov_a)` (`hapmixqtl.py:1873`), 73 at N = 92
   with 17 covariates, while the allelic scale is fitted on its own informative
   count. The simulator reports allelic calibration by informative-donor count
   so that this property is visible (section 5.4, *Nominal-p calibration*).
   SUPERSEDED 2026-09-27 (commits 8a06803, a1b2ef4): in default mode each
   channel's p is on its own residual degrees of freedom and the combined p on
   the Welch-Satterthwaite degrees of freedom (matched to the first two
   moments of the combined variance estimate, weights treated as fixed), with
   Meier's correction and a 15-donor allelic floor;
   `docs/hapmixqtl_methods.md` Section 4.5.
8. **Genotype-tied covariates go last** (`_combine_covariates`) and stay fixed
   under permutation (`WeightedResidualizer.n_fixed_cov`).
9. **The gene universe** of the real cache is the set of genes with at least
   one haplotype-paired transcript in some donor (`load_counts`); 1,208 of the
   12,955 calibration genes fall outside it (section 9.8). The simulator
   reproduces that rule for the tested genes by default and exposes a flag to
   include pair-less genes in the total channel; all 12,955 genes enter
   `totals_all` either way.

---

## 3. The layers as implementable interfaces

**Status of the interfaces.** Design A's attachments are not on this machine
(section 9.13). Every class, signature, field and array shape below is
therefore specified by this document and is the contract. If the attachments
are provided before build step 3 (section 8.2) completes, they are committed
verbatim in their own commit and then amended, in later commits, until they
satisfy this contract; every difference found is recorded in Appendix A. If
they are not provided by then, `tests/hapmix_simulator.py` is written from
this document alone, and the acceptance test *Layer unit tests* is the
document's own list rather than the attachment's 14 tests.

### 3.0 Structural requirements

These apply to every layer and each has an acceptance test in section 7.

- **Variance from the simulated counts.** The simulator specifies biological
  and technical variance at each record's expected count, generates counts,
  runs the quantification emulator, and obtains `Va`/`Vt` only from
  `summaries_from_point_estimates` on its output. It never draws `v` first and
  residuals given `v`. The variance-family test showed that a record's `v` is
  computed from its own point estimate, which couples value and variance:
  within a gene the median correlation of `log v` with the total residual is
  -0.258 with observed `v` and -0.005 with the fitted-value `v`, and a
  homoscedastic simulation with `v` recomputed from the simulated value
  reproduces a U-shaped profile of local slopes (-0.89, -0.86, -0.54, -0.54,
  -0.25, 0.17, 0.38, 0.69, 0.91 across within-gene deciles) with no variance
  function at all. A simulator that draws `v` first misses this coupling
  (acceptance test *Coupling mechanism*).
- **Values from the point estimate.** Layer 4 emits a Salmon-like point
  estimate and Gibbs draws; the values the pipeline sees come from the point
  estimate (pipeline rule 1).
- **Log2 throughout.** Truth, arms and metrics are in log2; mixQTL outputs
  are converted (section 2.2).
- **Local generators only**, addressed absolutely from `SEED = 42` (section
  3.1).
- **Reads from the TRUE haplotypes; the index from the OBSERVED haplotypes**
  (section 4).
- **Real data only as structure and nuisance.** Genotypes, exons, library
  sizes and library composition, covariates, per-gene variance parameters and
  per-donor multipliers enter from the real cohort. No real expression value
  of a gene is ever used as a simulated value.
- **Simulator imports.** The simulator modules import NumPy, pandas,
  `scipy.special` (digamma for the VB update) and `scipy.stats` (the
  Poisson-binomial interval of the acceptance test *No-failure-mode
  identity*), and never torch. Pipeline calls (which import torch) live in
  the arms and driver modules (section 8.1).

### 3.1 Randomness and containers

One master `SEED = 42`. Every stream is addressed absolutely as
`np.random.default_rng(np.random.SeedSequence(42, spawn_key=key))`; `spawn()`
is never called on a shared `SeedSequence`, because each call advances its
`n_children_spawned` and a second call would silently yield different
children. Keys:

| Child | Key | Stream |
|---|---|---|
| 0 | `(0,)` | Synthetic scaffold (`hwe` kind only; the `brainvar` scaffold is deterministic) |
| 1 | `(1, d)` | Architecture of power dataset `d` |
| 2, 3, 4 | `(k, r)` | Expression, reads, quantification of replicate `r`. The key carries no scenario, so replicate `r` of every scenario uses the same underlying random numbers (common random numbers), which pairs scenarios that differ in one parameter |
| 5 | `(5, s)` | Layer 5 failure-mode realization, once per scenario `s` |
| 6 | `(6, s, j)`; `(6, s, d, j)` in power scenarios | Oracle Monte Carlo re-draw `j` (section 3.8) |
| 7 | `(7, s)` | Evaluation resampling (gene-clustered intervals) |
| 8 | `(8,)` | Per-gene nuisance draws fixed across replicates and scenarios: `tau2_g` quantiles, signs of `a_g`, background-gene lognormal draws |
| 9 | `(9, s, r, j)` | Technical Monte Carlo of the metric *Gibbs-variance fidelity* |

Scenario ids `s` come from an append-only registry in the driver
(`scenarios.tsv`), never reordered, so adding a replicate, an arm or a
scenario never changes another's numbers. The only integer seed handed to
pipeline code is `seed=42` for `map_cis`; datasets are independent through
their data, not through the permutation stream.

A simulated dataset is one `SimDataset` (in `tests/hapmix_simulator.py`),
with G tested genes, G_bg background genes, N = 92 donors, D = 200 draws:

| Field | Type and shape | Meaning |
|---|---|---|
| `donors` | list of str, N | DNA-library ids (`100_D1`, ...) in cache sample order (`cache/gibbs_56b63c3b37ed5df8/samples.txt`) |
| `genes`, `genes_bg` | list of str, G and G_bg | Tested genes (full stack) and background genes (totals only); together every one of the 12,955 calibration genes |
| `sites` | DataFrame, V rows: `chrom`, `pos`, `ref`, `alt`, `kind` (`snv`/`indel`), `maf`, `role` (`index`, `tested`, `both`), `variant_id` | Every variant the simulator uses. `index` rows are exonic variants of the tested genes from the build call set (SNVs and indels of any frequency), used only to build the index and score fragments; `tested` rows are the SNVs the pipeline tests (section 3.2); sites in both files appear once with `role = both` |
| `X_true_L`, `X_true_R` | int8 `[V, N]` | True phased alleles, rows aligned to `sites` |
| `X_obs_L`, `X_obs_R` | int8 `[V, N]` | Observed (called) phased alleles; what the index and the pipeline see |
| `genotype_df` | DataFrame `[V_tested, N]`, float dosage `X_obs_L + X_obs_R`, indexed by `variant_id` | Tested rows only; what `map_nominal`/`map_cis` receive |
| `variant_df` | DataFrame `[V_tested]`: `chrom`, `pos`, indexed by `variant_id` | Tested rows only |
| `xL_df`, `xR_df` | DataFrames `[V_tested, N]` | Observed phased alleles of the tested rows |
| `phenotype_pos_df` | DataFrame `[G]`: `chr`, `pos` | `pos` is the fifth column (TSS) of `annot/genes.tsv`, which has no header; gene ids that occur more than once are dropped, as `corrected_null_store.select_genes` does |
| `pL`, `pR`, `pT` | float64 `[G, N]` | Emulated point estimates |
| `yL`, `yR`, `yT` | float32 `[G, N, D]` | Emulated Gibbs draws |
| `totals_all` | DataFrame `[G + G_bg + 1, N]` | Point-estimate totals for edgeR and expression PCs; the last row, `__rest_of_library__`, is each donor's real total of the genes outside the calibration list (from `point_estimates/totals_all.tsv.gz`, 41,552 genes), redrawn as Poisson each replicate, so `filterByExpr` sees the real library sizes |
| `eff_lib` | float64 `[N]` | edgeR effective library size recomputed on `totals_all` |
| `covariates_df`, `genotype_covariates_df` | DataFrames `[N, 14]`, `[N, 3]` | Metadata plus rebuilt expression PCs; real genotype PCs |
| `truth` | `Truth` | Section 3.8 |
| `quant_diag` | dict of arrays `[G, N]` | Per-pair `d_L`, `d_R`, `n_amb`, `n_U`, copy counts; used only by tests and diagnostics |

### 3.2 Layer 0: the scaffold is the real cohort

**Module and class.** `tests/brainvar_scaffold.py`, class `BrainVarScaffold`,
exposed through `GenotypeScaffold(kind, ...)` in `tests/hapmix_simulator.py`.
Two kinds are part of the contract:

- `'brainvar'`: this layer.
- `'hwe'`: a synthetic scaffold for the acceptance test *Permutation
  exactness* and for unit tests. N = 92; for each tested gene, tested SNVs at
  the real window's positions with alleles drawn on 2N independent haplotypes
  at each site's real in-sample MAF (Hardy-Weinberg, no linkage
  disequilibrium), and exonic index sites drawn the same way and
  independently of the tested sites; library sizes, metadata and factor
  scores taken from the real cohort but assigned to donors by a random
  permutation from child 0, so that no genotype is related to any record
  property.

An `'hmm'` kind wrapping `simulate_hmm_genotypes` (section 8.1) is optional,
for scale experiments only. Any other kind the attachments carry
(`balanced`, `phased`, `pair_resample`) is outside the contract and no
reported benchmark uses it (Appendix A, row 7).

**Interface.**
`BrainVarScaffold.load(genes, deploy_dir, window=1_000_000, exon_model='gene_union', seed_key=(0,))`
returns an object with:

| Attribute | Shape | Source (all verified present) |
|---|---|---|
| `donors` | N = 92 | `cache/gibbs_56b63c3b37ed5df8/samples.txt` (DNA-library ids) |
| `eff_lib_real` | `[N]` | `cache/gibbs_56b63c3b37ed5df8/point_estimates/edger/edger_samples.tsv` (`lib_size` x `norm_factor`); median 17,172,092 |
| `meta` | `[N, 4]`: `age_days`, `age_days_sq`, `rin`, `sex` | `cov/log2cpm1_point_calibration_20260925/covariates.tsv` |
| `geno_pcs` | `[N, 3]`: `geno_pc1-3` | same file; genotype-tied per `genotype_covariates.txt` |
| `factor_scores` | `[N, 10]`: the real `expr_pc1-10` | same file; used only as latent factor scores in Layer 2, never passed to the pipeline |
| `gene_table` | G rows: `gene`, `chr`, `start`, `end`, `tss` | `annot/genes.tsv` (no header; columns gene, chr, start, end, TSS); duplicated gene ids dropped |
| `sites`, `X_obs_L`, `X_obs_R` | V rows; int8 `[V, N]` | `index` rows: exonic variants (SNVs and indels) of the tested genes from the build call set `/mnt/ssd/lalli/nf_stage/brainvar2/gatk_t2t_haplotypecaller.joint_called.phased.all_variants.multiallelic.all.nostar.bcf` (271 samples, subset to the 92 donors in cache sample order); `tested` rows: SNVs with in-sample MAF >= 0.01 inside each tested gene's window from `prepped/analysis.snps.maf01.vcf.gz` via `read_phased_vcf` |
| `exons` | per gene (per transcript when available) | `annot/exons.tsv` (gene-merged exons) for `exon_model='gene_union'`; per-transcript exons from the annotation that built the personalized transcriptomes for `exon_model='transcript'` (not yet located, section 3.5) |
| `fld` | per donor fragment-length probabilities | Salmon's `aux_info/fld.gz` in each donor's quantification directory; donor 100_D1 (RNA library 100_R1) records mean 189.1 and sd 78.2 in `meta_info.json`, which the parser must reproduce |
| `library_type` | `ISR` | `meta_info.json` and `cmd_info.json` (paired-end, stranded) |

**Identifiers and contigs.** The donor key is the DNA-library id (`100_D1`):
the cache, the covariate files and both genotype files use it. Salmon
directories are named by RNA-library id (`100_R1`) and are reached only
through `cohort/salmon.tsv` (no header; DNA-library id, directory) or
`cohort/pairing.tsv` (header `dna_library rna_library bam`), never by string
edits of the id. The analysis VCF, the build BCF and `annot/genes.tsv` use
`chrN` contig names; `prepped/rephased.vcf.gz` uses RefSeq accessions
(`NC_060925.1`, ...), and wherever it is read the loader joins through
`annot/genes.NC.tsv`. The loader asserts that exon, site and gene contig
names join.

**Tested and index variants.** Only `tested` and `both` rows reach the
pipeline, which therefore tests exactly what production tests (SNVs, in-sample
MAF >= 0.01, the runner's `maf_threshold=0` on the MAF >= 0.01 call set).
Sites present in both files are deduplicated on `(chrom, pos, ref, alt)`, and
the loader asserts that their genotype and phase agree in every donor; a
review-time spot check over chr1:1-3 Mb in three donors found 5,485
heterozygous genotypes identical in both files.

The build call set and `prepped/rephased.vcf.gz` gave identical heterozygous
exonic counts over 18,400 sampled donor-gene pairs (layer-4 calibration,
`vcf_summary.json`, the same 14,909 records); the build call set is used
because it is the one the personalized transcriptomes were built from, and
the loader must confirm it carries the indels that decide whether two
haplotype copies differ. The layer-4 check found 3 of 18,400 pairs where the
pairing indicator and the heterozygous exonic count disagree, which that
analysis attributes to transcripts absent from `annot/exons.tsv`.

**Which genes** (defaults; the choice is the user's, section 9.15). G = 200
tested genes:

- the 100 genes of `brainvar_hapmix_deploy/corrected_null_store_20260925/`,
  so simulated and real null results are at the same loci (acceptance test
  *Real-null reproduction*);
- 70 calibration genes with draws, drawn from
  `edger/calibration_genes.txt` stratified by the median of `pL + pR` over
  informative donors (`pL + pR > 0`), the definition of
  `scripts/coupling_reach.py`: strata <30, 30-100, 100-300, 300-1,000,
  >=1,000 hold 815 / 1,653 / 2,336 / 3,671 / 3,272 of the 11,747 genes. The
  30-100 stratum, worst transcriptome-wide (0.075 at 0.05, 2026-09-25), is
  oversampled;
- 30 genes of a zero-read stratum, drawn from the 121 calibration genes with
  at least one donor at `pT < 0.5` (404 such donor-gene pairs in all,
  0.037% of calibration pairs). This stratum exists because zero-read total
  donors are otherwise nearly absent: the 100 corrected-null-store genes have
  none, and the 30-100 stratum holds 208 in 42 of its 1,653 genes (81 in 19
  of 1,412 genes if the median is taken over all donors instead). Whether the
  simulator reproduces them is checked (section 3.7); if it does not, the
  counting-term grid is also run with the injection described there.

Background genes: every other calibration gene, totals only, including the
1,208 without draws. Conclusions about boundary zeros and zero-haplotype
handling apply to the calibration-gene universe only: its informative pairs
are more deeply covered than the transcriptome's (Appendix A, row 2).

**Deferred.** A different N (from the HMM scaffold, section 8.1); the
annotation-track tilt of Layer 1.

### 3.3 Layer 1: architecture

**Module and class.** `Architecture` and `ScenarioGrid` in
`tests/hapmix_simulator.py`. Effects act on the TRUE haplotypes.

**Interface.** `Architecture(family, effect, alpha=-0.5, k=1, pi=None, causal_pool='instrument', null_fraction=0.5).draw(X_true_L, X_true_R, sites, gene, rng)`
returns `beta [M_g]` over the gene's tested cis variants and the haplotype
burdens `b [N, 2]`, with
`b[i, h] = sum_j beta_j (X_true_h[j, i] - xbar_j)`, where `xbar_j` is the
allele frequency of variant j over all 2N true haplotypes. Burdens are
therefore centred over the 2N haplotypes, so a non-zero mean burden never
moves the gene's baseline away from `c_g`. `effect` means:

- `sparse`: `|beta|` of each causal variant, in log2 aFC per allele copy
  (beta = 1 is a twofold difference between the haplotype carrying the
  allele and one that does not); the sign is random per gene. The grid is on
  beta directly because scaling a single variant's burden to a fixed sd gives
  `beta = sd / sqrt(p (1 - p))`, which depends strongly on MAF (at MAF 0.05
  and sd 0.4, beta = 1.84, a 3.6-fold effect). Power and aFC recovery are
  stratified by causal MAF.
- `infinitesimal`: `burden_sd`, the standard deviation of the centred `b`
  across all 2N haplotypes; weights `w_j = [2 p_j (1 - p_j)]^alpha` with
  effects `beta_j ~ N(0, w_j)` rescaled to that sd.

| Parameter | Default | Source and status |
|---|---|---|
| Families in the first benchmark | `null`, `sparse` with k = 1, `infinitesimal` | Reply C, pushback 1: both non-null families from day one |
| Families available, not in the first grid | `sparse` k = 2, 3; `point_normal` with pi in {0.01, 0.1, 0.5}; annotation tilt; age interaction (no pipeline interaction test; section 1.3) | Design A; grid values uncalibrated; whether `point_normal` pi = 0.1 joins the first grid is section 9.11 |
| `sparse` effect grid (log2 aFC per allele) | {0.1, 0.2, 0.4, 0.8} | Uncalibrated (section 9.11) |
| `infinitesimal` `burden_sd` grid (log2, per haplotype) | {0.05, 0.1, 0.2, 0.4} | Uncalibrated; design A's burden-SD grid, empirical cis-heritability pool still open |
| `alpha` (infinitesimal weight exponent) | -0.5, with -1 and 0 as sensitivity | Uncalibrated; design A's demo value |
| `causal_pool` for sparse | `'instrument'`: the null-instrument set, tested SNVs with MAF >= 0.05 outside every tested gene's body (the set the metric *Nominal-p calibration* reports separately); option `'all'`: every tested SNV | Default is a design choice so lead-variant recovery is defined; `'instrument'` excludes every gene-body variant, including exonic ones that also change distinguishability, which design A's annotation tilt targets. The choice is the user's (section 9.15) |
| Infinitesimal pool | every tested SNV (MAF >= 0.01) | Design choice |
| `null_fraction` in power datasets | 0.5; calibration datasets are 100% null | Design choice, so empirical FDR is estimable |
| Causal-MAF strata for reporting | [0.05, 0.2), [0.2, 0.5]; with `causal_pool='all'` also [0.01, 0.05) | Reporting choice |

**Ancestry.** No ancestry term is added in this layer. An infinitesimal burden
over real variants whose frequencies differ by ancestry is itself
ancestry-correlated expression (reply C); the benchmark measures it on output
as the correlation of `(b_iL + b_iR)/2` with `geno_pc1` and does not add a
term on top. The explicit ancestry term lives in Layer 2.

**Truth emitted.** `beta`, `b`, the per-individual log2 allelic fold change
`d_i = b_iL - b_iR`, the observed-label truth `d_obs` (section 4), and the
realized cis variances and cis heritability (section 5.4, *Cis-variance
recovery*).

### 3.4 Layer 2: haplotype expression

**Module and class.** `ExpressionModel` in `tests/hapmix_simulator.py`.

**Interface.** `ExpressionModel(params).draw(b, scaffold, gene_params, rng) -> log2_mu`,
with `b` the Layer 1 burdens `[G, N, 2]` and `log2_mu` the haplotype
expression `[G, N, 2]` in log2 CPM units;
`ExpressionModel(params).draw_background(scaffold, gene_params_bg, rng) -> log2_mu_bg [G_bg, N]`
gives the totals-only genes (`log2 mu = log2 c_g + z_ig`, no haplotype
terms). `gene_params` is a per-gene table (`c_g`, `gamma_g [4]`,
`lambda_g [10]`, `s2_u_g`, `tau2_g`, `a_g`), all on the model's log2 CPM
scale; `scripts/calibrate_simulator_nuisance.py` writes it (section 8.1).

**Model.** For gene g, donor i, haplotype h in {L, R}:

```
log2 mu_igh = log2 c_g - 1 + z_ig + b_igh + eps_igh
z_ig        = M_i . gamma_g + a_g PC1_i + F_i . lambda_g + u_ig
u_ig        ~ N(0, s2_u_g * m_i)
eps_igh     ~ N(0, (tau2_g / 2) * m_a_i)          independent across h
```

`mu_igh` is the haplotype's expected expression in CPM units, so the expected
total is `c_g * 2^z` at `b = eps = 0`. `M_i` is the donor's metadata row,
`PC1_i` the real genotype PC1, `F_i` the real expression-PC scores used as
latent factor scores (metadata, PC1 and factor columns are centred across
donors, so `c_g` stays the baseline), and `m_i`, `m_a_i` the donor variance
multipliers. With these definitions the allelic biological variance is
`Var(eps_L - eps_R) = tau2_g m_a_i`, and the total channel receives about
`tau2_g m_a_i / 4` from the haplotype terms.

**Scale.** The model is written on log2 CPM (log2 mu); every Layer 2
calibration was measured on the pipeline's T = log2(CPM + 1)
(`layer2/summary.json` labels them "log2(CPM+1) squared units"). The
derivative of log2(CPM + 1) with respect to log2 CPM is `c / (c + 1)`, so a
variance measured on the +1 scale understates the log2 CPM variance by
`(c / (c + 1))^2`: 0.25 at 1 CPM, 0.44 at 2, 0.79 at 8, 0.89 at 16. 4,096 of
the 11,747 calibration genes lie below 16 CPM. Every per-gene variance is
therefore converted before use: the log2 CPM variance is the value whose
image on the log2(CPM + 1) scale at the gene's `c_g`, computed by
Gauss-Hermite quadrature of `Var(log2(c_g 2^x + 1))` with `x ~ N(0, s)` and
`s` the log2 CPM variance sought, equals the measured variance; the first-order factor `((c_g + 1) / c_g)^2`
is its starting value (median by CPM bin 2.99 / 1.82 / 1.37 / 1.18 / 1.09 /
1.04 / 1.02 / 1.01 / 1.01 / 1.00 / 1.00 in the bins of the table below).
`gamma_g`, `lambda_g` and `a_g` are converted by the first-order factor
`(c_g + 1) / c_g`. The acceptance test *Expression-layer recovery* checks
the result on the pipeline's own scale, bin by bin, including the <2 and 2-4
CPM bins.

| Parameter | Default | Source and status |
|---|---|---|
| `c_g`, baseline CPM | The real gene's median CPM | `layer2/per_gene_total.tsv`, `median_cpm`; calibrated. For the 1,208 genes without draws, from `point_estimates/totals_all.tsv.gz` and the real effective library sizes |
| `gamma_g`, metadata coefficients | The real gene's OLS coefficients on age, age squared, RIN, sex in the 17-covariate fit, times `(c_g + 1)/c_g` | To compute at build time (`scripts/calibrate_simulator_nuisance.py`, section 8.1); not yet run |
| `lambda_g`, factor loadings on the 10 real expression-PC scores | The real gene's OLS coefficients on `expr_pc1-10` in the same fit, times `(c_g + 1)/c_g` | To compute at build time; not yet run |
| `B_g`, the gene's non-technical residual variance (log2(CPM+1) scale) | Empirical-Bayes shrinkage of `bio_draw_only_full` (below) | `layer2/per_gene_total.tsv`; calibrated |
| `s2_u_g`, idiosyncratic (post-PC) biological variance | `S(B_g) - tau2_g/4 - V_cis,g`, where `S()` is the scale conversion above and `V_cis,g` is the architecture's expected total cis variance, `Var_i((b_iL + b_iR)/2)` (for independent haplotypes `burden_sd^2 / 2` in the infinitesimal family, `beta^2 p (1 - p) / 2` in the sparse family); floored at `0.25 S(B_g)` | The floor is a design choice whose only remaining role is to keep the subtraction positive; the driver reports the share of genes where it binds. `bio_draw_only_full` contains the gene's own cis-genetic variance, so `B_g` is an upper bound on the non-genetic term (Appendix B, item 10) |
| Background genes | `s2_u_g = S(B_g)`, from their own per-gene rows (no haplotype or cis terms). The 1,208 genes without draws: biological variance `s2 - sum_i (1 - h_i) 0.975 q_t,i / 74` from their real totals (the median Gibbs-to-Poisson ratio of the total channel, 0.975, standing in for the missing draws), shrunk the same way; where that is not positive, a lognormal whose MEAN equals the bin mean of the table below, so its median is the mean times `exp(-sd^2/2)`, with log-sd 0.9 below 256 CPM and 0.6 above | `layer2/summary.json` across-gene spread; calibrated (moment-based below 16 CPM) |
| `m_i`, total-channel donor multiplier | Per donor and per CPM tercile (cuts 14.97 and 53.4 CPM), calibrated by inversion (below) | `layer2/donors.tsv`, `layer2/summary.json` `donor_component`; to calibrate at build time |
| `m_a_i`, allelic donor multiplier | Per donor, calibrated by inversion against `delta_allelic_depth_spline` (sd 0.161, permutation floor 0.029) | `layer2/donors.tsv`; to calibrate at build time |
| `tau2_g`, allelic biological variance | Lognormal, median 0.10, mean 0.30 (log-sd 1.48), independent of depth; one standard-normal quantile per gene from child 8 | Bracketed, not pinned. At >= 1,000 haplotype-informative reads, on draw-mean values with the draws-only Gibbs variance subtracted (`layer2/summary.json`, `point_estimate_error_sensitivity`, `tau2_draw_mean_draw_only`), the gene means are 0.20 (1,000-3,000 reads) and 0.49 (>= 3,000) with medians 0.09 and 0.11; with the shipped variance subtracted they are 0.19 / 0.48 and 0.08 / 0.11, so the counting term changes nothing at depth. On point values the means are 0.65 / 0.84 and the medians 0.33 / 0.28. Below 300 reads it is not measurable. The default leads with the draw-mean end because the point-value excess is 60-95% point-estimator error, which Layer 4 regenerates itself; for the same reason the per-gene `tau2_mom_draw_only` of `per_gene_allelic.tsv`, which is computed on point values, is not used |
| `eps` distribution | Gaussian | Uncalibrated tail: the allelic mean variance is dominated by a heavy-tailed imbalance that a Gaussian term matches in shape but not in tail (variance-family test); a Student-t option with its degrees of freedom is a TODO |
| Per-gene allelic offset | none | Calibrated: gene net imbalance across genes has sd 0.164 against a sampling expectation of 0.164 (`layer2/summary.json`) |
| `a_g`, ancestry coefficient | `a_g = sign_g sqrt(share x V_g / Var_i(PC1_i))` with `V_g = s2_u_g + tau2_g/4` (log2 CPM scale), `sign_g` random per gene from child 8, and `s2_u_g` reduced by `share x V_g` so the non-genetic residual variance is unchanged; share 0 (default) and 0.04 (arm) | Bracketed: genotype PC1's excess partial R^2 is 0.016-0.041 over a rebuilt null, 0.8-2.1 null standard deviations; after unrestricted expression PCs 0.0066. The observed 0.23 for three PCs is 0.133 construction (expression PCs built orthogonal to the genotype PCs) plus two outlier donors, and must not be used |

**Biological variance: shipped versus draws-only technical term.**
`scripts/simulator_layer2_calibration.py` (line 614) computes
`bio_full = s2 - sum_i (1 - h_i) Vt_i / df` with the SHIPPED `Vt` (Gibbs
variance plus the counting term `q_t`). The Gibbs draws already carry
Poisson shot noise (Gibbs `Vt` is 0.975 of the Poisson delta-method value at
the median gene), so the shipped `Vt` counts it twice and `bio_full`
subtracts it twice. The same table carries `bio_draw_only_full`, which
subtracts the draws-only Gibbs variance: that is the right quantity for a
simulator whose Layer 3 is Poisson and whose Layer 4 draws reproduce the
Gibbs variance. Medians are 0.0201 (`bio_full`) against 0.0268
(`bio_draw_only_full`); the per-gene ratio has median 1.19, the gap is 13.8%
of `s2` at the median gene, and `bio_full` is negative in 1.2% of genes
against 0.03% for `bio_draw_only_full`. The simulator uses
`bio_draw_only_full`.

**Shrinkage.** `x_g = bio_draw_only_full_g` has sampling variance
`V_g = s2_g^2 (2/74 + (kurt_g - 3)/92)` (the kurtosis-based delta method of
`sampling_var_s2` in `scripts/simulator_layer2_calibration.py`, df 74 from
`layer2/summary.json` `total.full.df`; `2 s2_g^2 / 74` for Gaussian
residuals). Within each CPM bin b, on the log scale: `y_g = ln x_g` with
sampling variance `V_g / x_g^2`, prior mean `m_b` = the bin's mean of `y_g`
(the log of the bin's median under a lognormal), prior variance
`A_b = max(0, var_b(y_g) - mean_b(V_g / x_g^2))`; the posterior mean is
`ln B_g = y_g + (V_g/x_g^2) / (V_g/x_g^2 + A_b) (m_b - y_g)`. Genes with
`x_g <= 0` (0.03%) take the bin median. A noisy small estimate is therefore
shrunk almost entirely to its bin's median, which replaces the earlier rule
`max(bio_full, bin value)` (that rule used bin means labelled as medians and
biased every gene upward, since `E[max(X, m)] > E[X]`).

**Biological variance by median CPM** (log2(CPM+1)^2 units; per-gene columns
of `layer2/per_gene_total.tsv` binned on `median_cpm`; mean / median).
"Draws-only" rows are what the simulator uses; "shipped" rows are the values
the earlier text quoted, kept for traceability. The `s2` rows are the
recovery targets of the acceptance test *Expression-layer recovery*
(`bins[].s2_q10_q25_q50_q75_q90` in `layer2/summary.json`).

| median CPM | <2 | 2-4 | 4-8 | 8-16 | 16-32 | 32-64 | 64-128 | 128-256 | 256-512 | 512-1024 | >=1024 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| genes | 769 | 833 | 1,026 | 1,468 | 2,093 | 2,196 | 1,701 | 1,013 | 422 | 163 | 63 |
| all 17 covariates, draws-only, mean / median | 0.084 / 0.060 | 0.098 / 0.061 | 0.091 / 0.052 | 0.063 / 0.032 | 0.039 / 0.024 | 0.035 / 0.019 | 0.028 / 0.017 | 0.028 / 0.019 | 0.026 / 0.019 | 0.028 / 0.023 | 0.039 / 0.022 |
| all 17 covariates, shipped, mean / median | 0.056 / 0.031 | 0.074 / 0.038 | 0.074 / 0.035 | 0.053 / 0.023 | 0.034 / 0.019 | 0.032 / 0.016 | 0.027 / 0.016 | 0.027 / 0.018 | 0.025 / 0.019 | 0.028 / 0.023 | 0.039 / 0.022 |
| metadata only, draws-only, mean / median | 0.267 / 0.187 | 0.397 / 0.260 | 0.340 / 0.208 | 0.256 / 0.139 | 0.184 / 0.105 | 0.157 / 0.091 | 0.144 / 0.092 | 0.154 / 0.107 | 0.143 / 0.109 | 0.149 / 0.111 | 0.153 / 0.130 |
| metadata only, shipped, mean / median | 0.238 / 0.159 | 0.374 / 0.236 | 0.324 / 0.192 | 0.246 / 0.130 | 0.178 / 0.100 | 0.155 / 0.088 | 0.143 / 0.091 | 0.153 / 0.106 | 0.142 / 0.108 | 0.149 / 0.111 | 0.153 / 0.130 |
| `s2` median, all 17 covariates (target) | 0.086 | 0.082 | 0.067 | 0.042 | 0.029 | 0.022 | 0.019 | 0.019 | 0.020 | 0.023 | 0.022 |
| `s2` median, metadata only (target) | 0.215 | 0.284 | 0.224 | 0.149 | 0.111 | 0.094 | 0.093 | 0.108 | 0.109 | 0.111 | 0.130 |

**Donor multipliers by inversion.** The recorded donor effects were estimated
from `log(e^2 / (1 - h))` on gene fixed effects, a spline in `log v` and
donor effects, where `e` carries both biological and technical residual
(technical share of `s2` at the median gene 0.27, larger at low CPM). A
multiplier applied to `u` alone is diluted by the biological share, and the
real effect grows with expression: its sd across donors is 0.25 / 0.29 /
0.40 in the low / middle / high CPM tercile (`donor_component.
total_full_by_cpm_tercile`), 0.30 pooled. So `log m_i,t` (donor i, tercile
t) starts at the real `delta_total_full_tercile{t}` (`layer2/donors.tsv`)
and is iterated as `log m <- log m + (delta_real - delta_sim)`, where
`delta_sim` re-estimates the same effect with the layer-2 donor fit
(`donor_fit` in `scripts/simulator_layer2_calibration.py`) on a simulated
null calibration dataset, until the per-donor root-mean-square difference is
below 0.02 (the permutation floor) in every tercile. `m_a_i` is calibrated
the same way against `delta_allelic_depth_spline`. The calibrated values are
written under `simulator_calibration_<date>/nuisance/` and recorded in
`provenance.json`.

**Why the real expression-PC scores are the latent factors.** The ten
expression PCs remove about 76% of the metadata-only biological variance
(draws-only means 0.212 to 0.050). Using the real donors' PC scores as factor
scores carries the real donor-level structure, including the outlier donors
(618_D1, 623_D1, 440_D1, 416_D1 at 3-10x before PCs), into the simulation;
the pipeline then rebuilds its own PCs from simulated totals, so its
construction (residualized on genotype PCs, rebuilt null share 0.133 for
three genotype PCs) is inherited rather than imposed. The genotype-PC outlier
donors 265_D1 and 416_D1 are represented through their donor multipliers and
factor scores, not through an ancestry term.

**Deferred.** Sex and age interactions with the cis effect, because the
pipeline has no interaction test to evaluate them (section 1.3); cross-donor
correlation beyond the covariates (relatedness, batch), which remains
untested on real data.

### 3.5 Layer 3: reads and equivalence classes

**Module and classes.** `tests/salmon_quant_emulator.py`:
`TranscriptModel` (exon structure per gene), `ObservedIndex` (copies per
transcript per donor, section 4), `FragmentClassBuilder` (fragments to
weighted equivalence classes). `CountModel` in `tests/hapmix_simulator.py`
delegates to these.

**Interface.** `ObservedIndex.build(X_obs_L, X_obs_R, sites, transcript_model) -> index`,
giving per (gene, donor) the copy table `[K]`;
`FragmentClassBuilder(fld, read_length, score_params).build(log2_mu, X_true_L, X_true_R, index, eff_lib_real, rng) -> classes`,
giving per (gene, donor) the copy table `[K]`, the weight matrix `[C, K]`, the
class counts `[C]` and the diagnostics below; background genes get
`Poisson` totals `[G_bg, N]` only.

**Counts.** Fragments per haplotype are Poisson,
`n_igh ~ Poisson(mu_igh * eff_lib_real_i / 1e6)`, split across the gene's
transcripts by isoform proportions. Layer 3 adds no negative-binomial
overdispersion: Layer 2 already carries every measured extra-Poisson term,
and the Gibbs variance of the total is 0.975 of the Poisson delta-method
variance at the median gene (IQR 0.95-0.99), so a negative binomial on top
would count the biological variance twice (Appendix A, row 11).

**Fragment placement by coverage pattern.** Simulating fragments one by one is
too slow (up to 10^5 per donor-gene pair). For each transcript, precompute the
distribution over coverage patterns: for every start position and fragment
length (from `fld`), the set of variant sites covered by the two mates'
aligned bases, with a site covered by both mates counted twice. Fragments per
pattern are then `Multinomial(n_igt, pattern probabilities)`. The number of
distinct patterns per transcript is small, so this is exact and fast.
Placement is uniform along the transcript: the production runs used
`--gcBias --seqBias`, and bias correction reaches the estimates only through
the per-copy effective lengths in `quant.sf`, which the copy table carries;
fragment start and GC bias in the simulated placement itself are deferred.

**Compatibility and weights.** For a fragment drawn from true haplotype h,
compute its score difference to each index copy of the transcript:
`Delta = 6 x (covered SNV sites where the copy's allele differs from the
fragment's true allele) + indel cost` (Salmon 1.10.3 defaults as read in the
layer-4 calibration: match +2, mismatch -4, so 6 score units per mismatch; gap
open 6, extend 2 per base; `scoreExp` 1; `minAlnProb` 1e-5; `hardFilter`
false). The best copy gets weight 1, another copy `exp(-Delta)`, and a copy
with `Delta > 11.5` (that is, `exp(-Delta) < 1e-5`) is dropped. So a fragment
is strictly distinguishable when it carries at least two SNV differences or a
costly indel, softly distinguishable (weight `e^-6`, likelihood ratio 403) at
one SNV difference, and ambiguous at none.

**Production Salmon flags** (`cmd_info.json` of every donor checked, e.g.
100_R1): Salmon 1.10.3, `--libType ISR`, `--validateMappings`, `--gcBias`,
`--seqBias`, `--rangeFactorizationBins 4`, `--dumpEq`,
`--numGibbsSamples 200`, no `--useEM`, no `--keepDuplicates` (`meta_info.json` records
`"keep_duplicates": false`).

| Parameter | Default | Source and status |
|---|---|---|
| Fragment-length distribution | Per donor from `aux_info/fld.gz` | Calibrated (on disk) |
| Read length | Unknown | Uncalibrated: not checked in the layer-4 calibration; read it from the donors' trimmed FASTQ (`HSB100.R1_val_1.fq.gz` and so on) or the trimming logs. The strict-distinguishability geometry depends on it |
| Score constants | As above | Read from Salmon 1.10.3 source in the layer-4 calibration, not verified against the binary; 1.10.3 is not installed locally (a 1.8.0 binary and a 2.1.2 build are) |
| Soft-weight component (`e^-6` for one-difference fragments) | On | Structure from source defaults; magnitude unverified because the dumped equivalence classes carry no weights. The target it must remove is the strict emulator's 1.23x excess Gibbs variance in one-SNV genes (1.16 at 2-3 SNVs, 1.16 at 4+, 1.10 with any indel). Settled directly by requantifying two or three donors with `--dumpEqWeights` |
| Isoform proportions | Cohort-level shares per transcript: each donor's per-transcript Gibbs draws (the draw file under `aux_info/`, in the subdirectory Salmon uses for resampled estimates) summed over the transcript's L and R copies, averaged over draws, then over donors | To compute (`calibrate_simulator_nuisance.py`). Per-donor point-estimate shares from `quant.sf` are not used: Salmon's VB estimate sets some isoforms and copies to exactly 0 in a donor, and an isoform the estimator zeroed would then never generate reads, turning an estimator artifact into truth |
| Exon model | `gene_union` (interim) | See below |
| Cross-gene shared fragments | Off by default; the paralog arm of Layer 5 turns it on | 3.0-7.3% of a gene's reads fall in classes shared with another gene (layer 4, by band); which genes and how is uncalibrated |

**Exon model, interim and target.** The target is per-transcript exons from
the annotation that built the personalized transcriptomes (the Salmon
directories are named `personalized_T2T_NCBI110_pseudoalignment`, pointing to
the NCBI release-110 annotation on T2T); that file has not been located. The
interim `gene_union` model treats each gene as one transcript on its merged
exons. It reproduces the pairing rule (a gene has distinct copies in a donor
if and only if the donor carries at least one heterozygous exonic variant,
18,397 of 18,400 pairs) and the homozygous-but-expressed pairs
(`pL = pR = 0`, `pT > 0`), but it cannot produce reads that are compatible
only with homozygous transcripts of an otherwise informative gene, which are
27% / 21% / 4.6% / 0.5% of reads in the 1-9 / 10-99 / 100-999 / 1,000+ bands.
Until the per-transcript model lands, the interim model adds a homozygous-only
segment carrying a share `u` of each informative gene's fragments, fitted to
that band table (not yet fitted). The per-transcript target departs from
design A's decision to keep a single haplotype mixture per gene; that
override is Appendix A, row 20, and needs the user's confirmation (section
9.16).

**Output.** Per donor-gene pair: a copy table `[K]` (transcript, label
`L`/`R`/unpaired, effective length taken from the donor's `quant.sf`), a
compatibility-weight matrix `[C, K]` and class counts `[C]`, plus the
diagnostic counts `d_L`, `d_R` (strictly distinguishable), `n_amb` and `n_U`.

### 3.6 Layer 4: the quantification emulator

**Module and class.** `tests/salmon_quant_emulator.py`, class
`SalmonEmulator(mode='chain')`. It is shared with the competitor-model tier
(section 6.2), which is why it is its own module. `QuantificationModel` in
`tests/hapmix_simulator.py` is a thin wrapper that calls it and then calls
`summaries_from_point_estimates`; it computes no summary of its own.

**Interface.**

- `point_estimate(classes) -> counts [K]`: Salmon's default variational Bayes
  (VB) optimum. VB here means the variational approximation to the posterior
  over transcript abundances that Salmon maximizes by default: responsibilities
  `r_ck` proportional to `w_ck exp(E[log eta_k]) / efflen_k`, with
  `E[log eta_k] = digamma(alpha_k) - digamma(sum alpha)` and
  `alpha_k = prior + sum_c n_c r_ck`. The prior is 0.01 per transcript copy
  in the documented defaults of the local Salmon 1.8.0 binary (`--vbPrior`
  0.01, per transcript unless `--perNucleotidePrior`); the production runs
  passed no prior flag and no `--useEM`. The prior at 1.10.3, the
  convergence rule and the rule that sets small abundances to exactly zero
  must be read from `src/CollapsedEMOptimizer.cpp` at tag `v1.10.3` of the
  Salmon repository (not on local disk) and pinned in a test (Appendix B,
  item 6). The real data show exact zeros and never a value in (0, 0.5).
- `gibbs_draws(classes, point, n_draws=200, n_chains=8, draws_per_chain=25,
  rounds_per_draw=16) -> draws [D, K]`: the chain as Salmon runs it, per the
  layer-4 reading of `CollapsedGibbsSampler.cpp`: 8 chains of 25 recorded
  draws each, each chain started from the VB point estimate, 16 rounds per
  recorded draw (`--thinningFactor` 16 in the 1.8.0 help). Draw index `j`
  belongs to chain `j // 25` at position `j mod 25`, which a test pins. One
  round allocates each class's fragments multinomially to its copies in
  proportion to `w_ck eta_k / efflen_k`, then draws each copy's abundance
  from a Gamma distribution whose shape is the allocated count plus the
  Gibbs prior. The prior's value, the Gamma rate parameterization (effective
  length, with or without an added constant) and whether a recorded draw is
  the allocated count or a rescaled abundance are NOT settled: the source is
  not on disk, and the calibration data argue against a prior of 1 per copy
  (a prior of 1 would let a zeroed copy escape within a few rounds, while the
  zeroed side's draw mean in fact keeps rising across each 25-draw chain, by
  a last-over-first-block ratio of 1.46 at 100-999 reads and 3.0 at 1,000+).
  All three are Appendix B, item 6a, settled from Salmon v1.10.3 source and
  pinned by a unit test before the acceptance test *Emulator on real
  equivalence classes*, which carries the block-ratio target that
  discriminates the prior directly.
- Summation: `pL`, `pR` over paired copies only; `pT` over every copy of the
  gene; the same for each draw into `yL`, `yR`, `yT` (section 4).
- `mode='beta'`: design A's stationary emulator
  (`p ~ Beta(d_L + k_L, d_R + k_R)`, paired total `~ Gamma(N + k_L + k_R)`),
  kept for fast unit tests only.
- `mode='parametric'`: a fallback point-estimate rule from the measured zero
  pattern (below), used only if the VB mode fails the emulator tests.

**Why the chain and not the stationary Beta.** On the three donors' real
equivalence classes the stationary Beta reproduces the median Gibbs variance
to 2-4% at 10 or more reads, but its per-pair error (sd of log10 simulated
over observed 0.195) is 4.2 times its Monte Carlo floor (0.046). Two measured
causes: chains started from the sparse VB estimate do not mix at depth (the
zeroed haplotype's draw mean rises across the 25-draw chain, last over first
block median 1.46 at 100-999 reads and 3.0 at 1,000+; the stationary Beta
over-predicts the zeroed side's mean, observed over predicted 0.81
[0.34, 0.97]), and soft-weighted one-difference fragments (section 3.5).
Running the chain on the simulator's own weighted classes absorbs both.

**Targets the emulator must reproduce**, from
`brainvar_hapmix_deploy/simulator_calibration_20260926/layer4/cache_targets.json`
(11,747 calibration genes, 786,919 informative pairs), except the tie row,
which comes from the equivalence classes of three donors
(`eq_class_summary.json`), and the block-ratio row, from `mixing_summary.json`;
banded by haplotype-informative reads `pL + pR`. These are calibration-wide
values; the acceptance test *Emulator band targets* recomputes them on the
tested genes.

| Target | 1-9 | 10-99 | 100-999 | 1,000+ |
|---|---|---|---|---|
| Point estimate zeroes one haplotype | 89.5% | 42.4% | 7.8% | 2.2% |
| Gibbs `Va` median (draws only, log2^2) | 1.57 | 0.73 | 0.206 | 0.052 |
| Gibbs `Va` / counting term `q_a`, both haplotypes positive | 0.77 | 2.52 | 7.23 | 13.2 |
| Gibbs `Va` / `q_a`, one haplotype zero (median) | 0.36 | 0.33 | 0.41 | 0.39 |
| Gibbs `Vt` / `q_t` | 0.965 | 0.917 | 0.963 | 0.986 |
| Fano factor of `yT` draws (across-draw variance / mean) | 0.986 | 0.987 | 0.987 | 0.986 |
| `(pL + pR)` / draw mean | 0.39 | 0.84 | 0.986 | 1.004 |
| Zeroed side's share of the draw total (median) | 0.49 | 0.45 | 0.34 | 0.12 |
| Zeroed side's draw mean, last over first 5-draw block of a 25-draw chain (median) | 1.00 | 1.03 | 1.46 | 3.0 |
| Tie zeroed (no strict-distinguishable read on either side) | 96% | 87% | 69% | 48% |

Further targets: a side with any strict-distinguishable read is never zeroed
(0 exceptions in 42,356 pair-sides); a side with none, when the other side has
some, is zeroed in 72.9% of pairs (1,764 of 2,420); in ties, L and R are
zeroed equally often (881 against 837); the maximum-likelihood L fraction
`d_L / (d_L + d_R)` agrees with the point L fraction at Pearson 0.82 and
median absolute difference 0.021; the zeroed haplotype's draws are almost
never zero (in every band the median share of its draws at exactly zero is
0); about 1.0% of calibration pairs (10,913) have `pL = pR = 0` with
positive draws, a rate reproduced by rate only because its rule was not
derived. From the layer-2 calibration: allelic Gibbs variance is
`r(n) x 4 / ((n + 1) ln2^2)` with `r` = 1.7 / 3.4 / 5.9 / 9.3 / 13.0 / 17.1 at
<30 / 30-100 / 100-300 / 300-1,000 / 1,000-3,000 / >=3,000 reads; and the
allelic biological variance estimated on point values exceeds the same
estimate on draw-mean values by 1.88 / 1.91 / 1.29 / 0.76 / 0.45 / 0.35
(log2^2) in the same bands, which is the point estimator's own error
variance and must emerge from the emulator, not be added. The
per-pair distinguishable fraction `phi_pair = (d_L + d_R) / (d_L + d_R + n_amb)`
has median 0.065 [IQR 0.027, 0.126] across pairs (its per-gene median over
informative donors is the `Truth` field `phi_g`, section 3.8), and at 100 or more reads rises with heterozygous
variants per exonic kb: 0.022 / 0.062 / 0.117 / 0.205 / 0.335 at
<0.5 / 0.5-1 / 1-2 / 2-4 / 4+ per kb.

**Then** `summaries_from_point_estimates(pL, pR, pT, eff_lib, yL, yR, yT, kappa=0.5, count_noise=...)`
on the emulator output, with `eff_lib` from edgeR on the simulated
`totals_all`.

**Deferred.** Tier 3 (section 6.3): real Salmon on simulated reads.

### 3.7 Layer 5: failure modes, applied to what the method sees

**Module and class.** `Artifacts` in `tests/hapmix_simulator.py`, defined as
processes that make the TRUE haplotypes differ from the OBSERVED ones
(section 4), plus explicit injections. Every rate defaults to zero; each
process is an arm.

**Interface.** `Artifacts(rates).apply(X_obs_L, X_obs_R, sites, gene_table, rng) -> (X_true_L, X_true_R, masks)`,
with the haplotype arrays int8 `[V, N]` and `masks` a dict of boolean arrays
(`[V, N]` for altered sites, `[G, N]` for altered records, `[G]` for
imprinted and paralog genes) that `Truth` keeps. It runs before Layer 1, so
effects and reads both see the TRUE haplotypes. **Its realization is drawn
once per scenario** (key `(5, s)`) and held fixed across that scenario's
replicates, because it describes the cohort's call set, not sampling noise;
the sampling null and the oracle both hold it fixed (sections 3.8, 5.4). The
reference-bias process acts on fragments, so its thinning draws are part of
Layer 3 and redrawn per replicate at the scenario's fixed `phi_ref`.

| Process | What it does | Default rate in its arm | Status |
|---|---|---|---|
| Phase switch error between the gene and its cis variants | For each donor, switch points along the chromosome form a Poisson process of rate `r_switch` per Mb over the gene's window; the true phase of a heterozygous site is the opposite of the observed phase when an odd number of switches lies between it and the gene's TSS. The phase of tested variants relative to the gene's exonic heterozygotes then changes with distance (design A's odd-number-of-switches probability, `(1 - exp(-2 r_switch d)) / 2` at distance `d` Mb) | swept over {0.1, 0.5, 2} switches per Mb | Uncalibrated: no switch-error measurement exists for this call set (Appendix B, item 17). Design A calls this the allelic channel's main exposure; Hu 2015 found the joint model loses to total counts alone below imputation Rsq 0.4 and that imputed phase inflates the cis/trans test |
| Per-site phase flips at exonic heterozygotes (index errors) | True phase opposite to the called phase at single exonic sites, by minor-allele-count class; this mixes allocation between the two copies of a multi-SNP transcript | Singleton 0.191, common 0.020 | Upper bounds only: these are the rates at which called phase disagrees with read-backed phase (`brainvar_hapmix_deploy/nominal_p_hypotheses_20260925/reconciliation.md`), not error rates. `nf_stage/brainvar2/trio_benchmark_eval/` holds HG002 call sets and coverage reports, not a phase comparison |
| Genotype miscall | Called heterozygote truly homozygous (index has copies the reads cannot separate by truth), or called homozygote truly heterozygous (a true allele neither copy carries) | TODO | Uncalibrated; candidate measurement: the HG002 call sets under `trio_benchmark_eval/` against the GIAB truth set, for SNVs and indels separately |
| Singleton and indel errors | The above, at elevated rates for singletons and indels | TODO | Uncalibrated; this is the mechanism reply C proposes for confident-but-wrong records |
| Confident-but-wrong record (explicit) | For a chosen fraction of informative pairs, reverse the true phase of every exonic heterozygote of one donor-gene pair | TODO | Uncalibrated: 0.82% of records are Salmon-versus-alignment discordant beyond 3 sd, a rate of symptoms, not of causes. The CALM2 donor 657_D1 record is the model case |
| Reference bias (alternate-fragment loss) | A fragment that carries an alternate allele at a covered heterozygous exonic site is lost with probability `1 - (1 - phi_ref)/phi_ref`, so that at allelic balance the expected reference fraction of distinguishable fragments is `phi_ref` (RASQUAL's phi; named `phi_ref` here to keep it apart from the distinguishable fraction `phi_g` and the harness's dispersion `phi`). This emulates alternate-allele mapping loss; with a personalized diploid index it stands in for uncalled or reference-like index sites | swept over `phi_ref` in {0.52, 0.55, 0.60} | Uncalibrated (Appendix B, item 26). `docs/ase_validation.md` section 7h measured nominal type-I 0.635 at phi = 0.60, a failure it calls catastrophic rather than gradual, but on the since-deprecated `tau='estimate'` configuration; the size under default mode is unmeasured. The production runner gates on `reference_bias_diagnostic` and refuses to proceed when it fires (section 2.2) |
| Paralog or segmental-duplication genes | A fraction of a gene's fragments also compatible with a partner gene's copies | 5% | Uncalibrated structure; the measured shared-class share is 3.0-7.3% by band |
| Imprinted genes | One haplotype silenced (random L or R per donor, since L/R is not parental) | 0.3% of genes | Rough: 35 of 11,500 evaluable genes are one-copy in at least 80% of their deep donors |
| Random monoallelic expression | One haplotype silenced in a donor-gene pair | TODO | Uncalibrated: at 1,000+ reads 21.6% of zero-haplotype pairs have phASER minor fraction below 0.05, but no per-pair rate for the simulator follows from that |
| Boundary zeros | Not injected | | They emerge from Layer 4 (section 4) |
| Zero-read total donors | Not injected by default | | Expected to emerge from low expression in the zero-read stratum (section 3.2), whose 30 genes are drawn from the 121 calibration genes carrying the 404 real zero-read pairs. The driver reports the simulated count against the real one. If the simulated count is below half the real count for the same genes, the counting-term grid (section 5.2) is also run with an injection that silences the gene (`mu = 0` on both haplotypes) in exactly the real zero-read donor-gene pairs of those genes, labelled as injected. Without either, the benchmark does not test the zero-read floor |

Deferred: within-sample overdispersion and double counting, which matter only
for per-SNP count writers (section 6.2); trans-only genes, which in this
pipeline's scope are null genes unless the trans signal correlates with
genotype through ancestry, which the ancestry arm covers. Design A wants
trans-only genes included so the method is scored on not calling them; the
deferral is the user's to confirm (section 9.17).

### 3.8 Layer 6: truth and evaluation

**Module and class.** `Truth` in `tests/hapmix_simulator.py`, plus
`tests/simulation_benchmark_metrics.py` for the metrics of section 5.4.

`Truth` holds, per dataset: `beta`, `b [G, N, 2]`, `d [G, N]`
(`b_iL - b_iR`, log2), the observed-label truth `d_obs [G, N]` (section 4),
realized cis variances on both scales and the realized cis heritability
(section 5.4, *Cis-variance recovery*), the family and effect size per gene,
the causal variant ids and their MAF, the donor multipliers, `tau2_g`,
`s2_u_g`, the per-gene distinguishable fraction
`phi_g = median over informative donors of (d_L + d_R) / (d_L + d_R + n_amb)`,
the failure-mode masks (which sites and records were altered, and the
imprinted, monoallelic, paralog and confident-but-wrong classes), and the
oracle variances `vstar_a`, `vstar_t [G, N]`.

**Oracle variance.** Defined over exactly the components that vary between
sampling replicates (section 5.4, *Nominal-p calibration*, sampling null):
`u`, `eps`, the Poisson fragments, their placement, and the equivalence
classes and VB point estimate that follow from them. Held fixed, as in the
sampling null: the scaffold, the Layer 5 realization, the architecture
(null in calibration scenarios), and every nuisance parameter (factor
scores, loadings, `c_g`, `tau2_g`, `s2_u_g`, `a_g`, donor multipliers).

- `vstar_t(g, i) = Var(T_gi)` over re-draws, with `eff_lib` held at the
  scenario's reference value (the edgeR effective library sizes of that
  scenario's replicate 0); edgeR is not re-run per re-draw.
- `vstar_a(g, i) = Var(A_gi | the record is included under the arm's
  inclusion rule)`, over the re-draws in which it is included: `pL + pR > 0`
  for `zeros_keep`, and additionally not a one-haplotype-zero pair for
  `zeros_drop`. In a dataset, the oracle arm uses `vstar_a` where that
  dataset's own record is included and 0 where the dataset's `Va == 0` or the
  record is dropped.
- A record needs at least 50 included re-draws out of 200; otherwise it is
  excluded from the oracle arm (its `f` set to 0) and the count is reported.
  This is the one place the oracle arm's allelic record set can differ from
  the other arms'.
- Computed ONCE per calibration scenario from child 6, with 200 re-draws
  (relative standard error of each variance about 10%), because it depends
  only on quantities held fixed there. In power scenarios the architecture
  differs per dataset, so the oracle is recomputed per dataset with 100
  re-draws (minimum 25 included), on the first 10 power datasets of each
  scenario only; the oracle arm's power is reported on those.
- Cross-checked against the delta-method value at the expected count in the
  acceptance test *Oracle honesty*.

---

## 4. The true-versus-observed haplotype split

**Construction.** The OBSERVED haplotypes are the real called, phased
genotypes of the 92 donors, exactly as the index and the pipeline saw them.
The TRUE haplotypes are the observed ones with the Layer 5 error processes
applied. With every rate at zero, true and observed are identical, which is
the no-failure-mode baseline and the acceptance test *No-failure-mode
identity*. Building it this way round, rather than simulating truth and then
corrupting a copy, makes the index's structure (which donor-gene pairs have
distinct copies) exactly the real one.

**Steps, in order.**

1. Expression truth (Layers 1-2) acts through the TRUE haplotypes.
2. Fragments are generated from the TRUE haplotype sequences (Layer 3).
3. The index is built from the OBSERVED haplotypes. For each donor and
   transcript, the two copies are distinct if and only if the observed
   haplotypes differ at one or more exonic variants (SNV or indel) of that
   transcript. Identical copies collapse to one row, the lone `_L`, with no
   `_R` row at all (not a zero row), as Salmon's indexer did without
   `--keepDuplicates` (every `meta_info.json` records
   `"keep_duplicates": false`).
4. Each fragment is scored against the OBSERVED copies (section 3.5) to form
   weighted equivalence classes.
5. Layer 4 gives the point estimate and draws per copy.
6. Summation follows `load_counts` exactly: `pL`, `pR`, `yL`, `yR` over paired
   transcripts only (both copies present in that donor's index); `pT`, `yT`
   over every copy of every transcript of the gene, paired or not. So
   `pT != pL + pR` in general, and a homozygous-but-expressed pair has
   `pL = pR = 0` with `pT > 0`.
7. The pipeline receives the tested rows of the OBSERVED haplotypes as
   `xL_df`, `xR_df` and dosage `genotype_df`, the emulated point estimates
   and draws, library sizes from edgeR on the simulated totals, the metadata
   and real genotype PCs, and expression PCs rebuilt from the simulated
   totals.

**What emerges without separate injection**, and how each was checked
against the reply's claims:

- **Uninformative pairs.** Reply C says the 58-59% of donor-gene pairs with no
  allelic information emerge from whether any indexed transcript differs. The
  structural part does: over all 34,457 cache genes, 38.2% of pairs have no
  paired transcript. But the 59.2% total is a SUM: 38.2% structural, plus
  19.6% paired with zero reads in every draw, plus 1.4% paired with positive
  draws that the VB estimate puts at zero on both copies. On the 11,747
  calibration genes the shares are 26.2% structural and 27.2% in total. All
  three components emerge in the simulator, the second and third only if
  expression and the VB step are realistic, which the acceptance test
  *Uninformative-pair structure* checks.
- **Boundary zeros.** Reply C says the maximum-likelihood split
  `total x d_L / (d_L + d_R)` is exactly zero when `d_R = 0` while the draws
  stay positive. The layer-4 measurement refines this in two ways the
  emulator must follow. When one side has no strictly distinguishable reads
  and the other has some, Salmon's point estimate is zero on the empty side in
  72.9% of pairs, not always; in the other 27.1% the empty side keeps a median
  share of 0.40, mostly where the other side has only one to three
  distinguishable reads. And when neither side has any distinguishable read
  (a tie), the maximum-likelihood split is undefined, yet Salmon zeroes one
  side in 81.6% of ties, falling with depth from 96% to 48%. The VB estimator
  produces both behaviours; a maximum-likelihood split produces neither.
  Most real zero-haplotype pairs are biallelic by phASER (median minor-allele
  fraction 0.41-0.44 against 0.44-0.47 in controls), so zeros are mostly an
  estimator effect, which is why they appear even when true and observed
  haplotypes agree.
- **Confident-but-wrong records.** An observed heterozygote that is truly
  homozygous, or a phase or indel error in the index, sends distinguishable
  fragments to the wrong copy with a small Gibbs variance. This matches the
  higher read-backed phase disagreement of singletons (at most 19.1%) than of
  common exonic SNVs (2.0%). It is not established as the cause of such
  records on real data: Salmon-versus-alignment discordant records account
  for 0.03 / 1.5 / 4.7 / 10% of the weight-residual coupling at
  0.05 / 0.01 / 0.001 / 1e-4, and singleton mis-phasing reaches only 137 of
  9,328 discordant records, which could not be distinguished from chance.
  So the simulator produces them through index errors and also offers the
  explicit injection of section 3.7.

**Orientation and observed-label truth.** The allelic value is
`log2((pL + 0.5) / (pR + 0.5))` against `s = xL - xR` on the observed
haplotypes. The truth the pipeline could at best recover is defined from the
noise-free class model, not by flipping `d_i`:
`d_obs,i = log2(E[fragments whose highest-weight observed copy is L] /
E[fragments whose highest-weight observed copy is R])`, with ties excluded,
computed exactly from the coverage-pattern probabilities (section 3.5) at the
true `mu`. Without failure modes `d_obs = d`; a whole-haplotype swap gives
`-d`; a phase error at one site of a multi-SNP transcript gives a value in
between, which a sign flip cannot express. The metric code scores against
`d_obs` and also reports `d` for the failure-mode arms.

---

## 5. Comparator arms and evaluation metrics

### 5.1 One mechanism for every weighting arm

There is no weight argument in the pipeline. Default mode fits
`Var(eps) = sigma^2 f`, where `f` is whatever arrives as `Va` or `Vt`, and uses
`f` only as a shape. So every weighting arm is reached by replacing `Va` and
`Vt` with a function `f(Va)`, `f(Vt)` before calling `map_nominal` or
`map_cis`, exactly as `scripts/gene_set_weighting_null.py` `variances` does.
Constant factors cancel against the fitted scale. Unless a row says
otherwise, `f` acts on the pipeline's OBSERVED `Va` and `Vt` (Gibbs variance
plus the counting term, as `summaries_from_point_estimates` returns them).
The rules:

- `f` must keep `Va == 0` where the pipeline had `Va == 0`, so the allelic
  record set is the same in every arm except the zero-haplotype drop arm, the
  `oracle` records with too few included re-draws (section 3.8), and the
  `no_q_a` records whose draws are unanimous (section 5.2); each exception
  is counted and reported.
- `f(Vt)` must be positive for every donor (section 2.4, item 4).
- Total-channel exclusion uses `keep_t_df`; allelic exclusion may use either
  `Va = 0` or `keep_a_df`.
- Arms are compared on identical datasets: every arm of one replicate reads
  the same `SimDataset`.

### 5.2 The arms

**Weighting arms** (the weight is `1 / f`):

| Arm | Allelic `f` | Total `f` | Why it is here | Evidence behind it |
|---|---|---|---|---|
| `gibbs_both` (shipped) | `Va` | `Vt` | The shipped default | Anticonservative on the corrected real null: combined 0.0719 at 0.05 with zeros dropped |
| `split` | `Va` | 1 | The calibrated candidate on real data | Combined 0.0512 / 0.0117 / 0.0027 at 0.05 / 0.01 / 0.001 on the 100-gene real null |
| `unit_both` | 1 where `Va > 0` | 1 | The no-draws weighting | Combined 0.0529 / 0.0122 / 0.0028 |
| `plus_one` | `Va + 1` | `Vt + 1` | Tested on real data | Combined 0.0498; widens the allelic true spread by 22% |
| `power(ga, gt)` | `Va^ga` | `Vt^gt` | The global-exponent candidate the 2026-09-25 authorization lists | Variance-family test, pseudo-likelihood fits: on OBSERVED `v`, total gamma 0.35 [0.33, 0.37] and allelic 1.2 (bin level: allelic 1.3 within-gene, 1.14 absolute); on FITTED-VALUE `v`, total 0.37 [0.35, 0.38] (bin level 0.16-0.20) and allelic 0.56 (bin level about 0). The arm applies the exponent to observed `v`, so the grid is set from the observed-`v` fits and brackets the fitted-value ones: allelic `ga` in {0.56, 1.0, 1.2, 1.5} with total at `split` (`gt = 0`), and total `gt` in {0.2, 0.35, 0.6} with allelic at `split` (`ga = 1`). `ga = 1, gt = 0` is `split` itself and `ga = gt = 1` is `gibbs_both`. The `v^0.65` residual-variance growth recorded on 2026-09-25 (and cited in reply C) was measured on the pre-correction pipeline (natural-log draw-mean values, no library normalization) and is superseded by these fits, which were made on the pipeline this benchmark pins |
| `relative_floor` | `Va + r_a x median_g(Va)` | `Vt + r_t x median_g(Vt)` | The additive form, carried as a measurement (section 9.10) | Gene-relative floor. Total: `r_t = 1.45` from the observed-`v` fit, log10 r 0.16 [0.14, 0.18], where it beats the power law by 7,740 deviance units (fitted-value `v`: 0.18 [0.16, 0.20]). Allelic: `r_a = 1.0` is the FITTED-VALUE-`v` fit (log10 r 0.00, where it beats the power law); on observed `v` the allelic fit runs to the grid edge (log10 r = -4, that is no floor) and loses to the power law by +7,851 [7,221, 8,466] deviance, so the allelic setting is a fitted-value-`v` parameter applied to observed `v`, labelled as such. `median_g` is taken over the gene's records with `Va > 0` (all records for `Vt`), matching `medg = median(v[g][mask[g]])` in `variance_family_test.py`. See the caution below |
| `oracle` | `vstar_a` | `vstar_t` | The ceiling every scheme is scored against | Section 3.8 |
| `donor_quality` | `Va x d_i` | `Vt x d_i` | A per-donor variance multiplier independent of `v` | The per-donor mean of read-count-adjusted log Gibbs variance varies little across donors (sd 0.073 recorded; 0.063-0.102 on the corrected pipeline), while the donor residual component is larger (sd 0.30 after PCs) and does not track it (Spearman 0.077) |
| `donor_true_multipliers` | `Va x m_a_i` | `Vt x m_i` | True biological multipliers applied to `v` | Multipliers from `Truth`. This is not the best a per-donor multiplier can do: `m` scales only the biological term in Layer 2, while this arm scales the whole Gibbs variance |
| `donor_multiplier_ceiling` | `Va x dstar_a,i` | `Vt x dstar_t,i` | The best a per-donor multiplier can do | Per donor and channel, `dstar_i = exp(mean over the donor's records of log(vstar / v))`, the multiplier closest to the oracle in log-squared distance; needs the oracle, so it runs where the oracle does |

Two weighting forms are not run. A pooled additive floor `1/(v + tau)` with
one `tau` for all genes lost to the power law in both channels under both
observed and fitted-value `v` (for example +1,229 [+820, +1,658] deviance in
the total channel with observed `v`, +1,474 [+1,126, +1,827] with
fitted-value `v`), so it is excluded by measurement. A per-gene `tau` fitted
from the gene's own residuals is excluded by the 2026-09-23 deprecation
(circularity). A fitted-value-`v` version of the `power` and `relative_floor`
arms (building fitted-value `v` in the arms module with the recipe in
`variance_family_test.py`'s docstring, "OBSERVED v AND FITTED-VALUE v") is
not in the first grid: the fitted value carries the record's own `h_i e_i`,
so whether such an arm stays clear of the circularity objection needs the
user's view first (section 9.10).

**Caution on `relative_floor`.** `f = v + r x median_g(v)` scales with `v`:
doubling every `v` in a gene doubles `f` and leaves the weights unchanged, so
the weights cannot feel the draws' absolute scale. That is the property of the
second deprecation objection (free-`c` models discard the draws'
calibration). It is not circular (it uses no residuals), but reporting it as
a candidate rather than a measurement needs a user decision (section 9.10).
The `power` family shares scale invariance with the shipped default itself,
whose fitted `sigma^2` also absorbs any constant.

**`donor_quality` estimator.** In the spirit of limma's `arrayWeights`, which
estimates one variance multiplier per sample from residuals pooled across all
genes, `d_i` is estimated per channel with the layer-2 donor-component design
(item 2 of the docstring of `scripts/simulator_layer2_calibration.py`,
function `donor_fit`): `log(e_ig^2 / (1 - h_ig))` on gene fixed effects, a
natural cubic regression spline with 6 degrees of freedom in `log v` (in the
allelic channel, in log haplotype-informative depth, the "depth spline"
variant, because `Va` grows with the donor's own imbalance) and donor effects
summing to zero. Total: `e` is the OLS residual of `T` on the pipeline's
covariates without any variant and `h` its leverage. Allelic: `e` is `A` minus
the gene's `1/f`-weighted mean and `h_i = w_i / sum w`. Records with
`e^2 <= 1e-12` (for example `A` equal to the weighted mean exactly) are
excluded and counted, since their log is minus infinity. It is cross-fitted:
`d_i` for genes on one chromosome is estimated from genes on all other
chromosomes, so no record's own residual sets its own weight. Under
permutation, `d_i` travels with the donor's record.

**Counting-term arms.**

| Arm | What changes | How it is implemented |
|---|---|---|
| `count_noise_on` | Counting term on every record (shipped) | `summaries_from_point_estimates(count_noise=True)` |
| `readless_floor_t` | Total counting term kept only where the draws give no variance; allelic term unchanged | Call `summaries_from_point_estimates(count_noise=False)` for the draws-only `Va_d`, `Vt_d`, and compute `q_a`, `q_t` with the formulas at `hapmixqtl.py:485-488`. Then `Vt = Vt_d + q_t x 1[pT == 0 or Vt_d <= 1e-12]` and `Va = Va_d + q_a` where `pL + pR > 0` (0 otherwise), the shipped allelic value. The floor keys on `Vt_d <= 1e-12` as well as `pT == 0` because the 1e8 weight arises wherever the draws-only `Vt` is 0 (section 2.4, item 4). This is the fix the counting simulation of 2026-09-15 and `docs/pipeline_rules.md` rule 6 leave open: a floor for zero-count total donors and no counting term for donors with reads |
| `no_q_a` | Allelic counting term removed; total unchanged | `Va = Va_d` where `pL + pR > 0`, `Vt` as shipped. A record whose draws are unanimous (`Va_d <= 1e-12` with `pL + pR > 0`) leaves the allelic channel; the count is reported. For one-zero pairs the draws-only `Va` is about 0.33-0.41 of `q_a` (table, section 3.6), so their weight rises about 3.5-4 times; this arm isolates that change from the total-channel question |

A plain `count_noise=False` arm is invalid (weight 1e8 for zero-variance total
donors) and is not run.

**Genotype-PC arms.** `pcs_tied`: pass the genotype PCs as
`genotype_covariates_df`. `pcs_moving`: append them to `covariates_df` and
pass `genotype_covariates_df=None`.

**Other arms.**

| Arm | What changes | How it is implemented |
|---|---|---|
| `zeros_keep` / `zeros_drop` | Zero-haplotype pairs kept, or excluded from the allelic channel | Drop: `Va := 0` where `(pL < 0.5) XOR (pR < 0.5)`, the rule of `scripts/corrected_null_store.py`; the total channel keeps them |
| `total_only` | Allelic channel off | `keep_a_df` all False |
| `mixqtl_published` | mixQTL mode at `PUBLISHED_CUTOFFS`: `trc_cutoff` 100, `asc_cutoff` 50, `weight_cap` 10, `asc_cap` 1000 | `inputs_from_point_estimates`, `mixqtl_scan`, `mixqtl_permutation_scan(genotype_covariates=...)` per gene on simulated point estimates and `eff_lib`; slopes and standard errors divided by ln 2 (section 2.2) |
| `mixqtl_permissive` | mixQTL mode at `PACKAGE_DEFAULT_CUTOFFS`: 20, 5, 100, 5000 in the same order | As above. Kept because the published upper cap removed 1,656 of 2,193 informative pairs on the 29 calibration genes |
| `hapmix_mixqtl_cutoffs` | hapmixQTL restricted to mixQTL's donors, under `gibbs_both` and `split` | `count_cutoff_masks` into `keep_a_df`/`keep_t_df`. It reads its counts from whatever arrays it is given; pass the point estimates `pL`, `pR`, `pT` (pipeline rule 1: count cutoffs come from point estimates), with `pT` as the total, never `pL + pR` |

**Grids.** An arm that changes one factor while the total channel carries unit
weights cannot see a factor that acts only through the total weights. The
counting term changes `Vt`, which `split` and `unit_both` never use, and the
genotype-PC tie hurts only through the Gibbs total weights on real data. The
first benchmark therefore runs three factorial grids and one set of
auxiliary arms:

| Grid | Factors | Held at | Scenarios |
|---|---|---|---|
| Weighting grid | {`gibbs_both`, `split`, `unit_both`, `plus_one`, `oracle`} x {`zeros_keep`, `zeros_drop`} | `count_noise_on`, `pcs_tied` | Every scenario |
| Genotype-PC grid | {`pcs_tied`, `pcs_moving`} x {`gibbs_both`, `split`, `unit_both`, `oracle`} x ancestry share {0, 0.04} | `zeros_drop`, `count_noise_on` | The default calibration scenario and its ancestry-0.04 counterpart |
| Counting-term grid | {`count_noise_on`, `readless_floor_t`} x {`gibbs_both`, `plus_one`}; plus `no_q_a` x {`gibbs_both`, `split`} at `zeros_keep` (where one-zero pairs are in the allelic channel) | `zeros_drop` (except `no_q_a`), `pcs_tied` | Calibration scenarios; the zero-read stratum (section 3.2) carries the floor |
| Auxiliary arms | `power`, `relative_floor`, `donor_quality`, `donor_true_multipliers`, `donor_multiplier_ceiling`, `total_only`, `mixqtl_published`, `mixqtl_permissive`, `hapmix_mixqtl_cutoffs` | `zeros_drop`, `count_noise_on`, `pcs_tied` by default, the setting of the real-data weighting nulls, so their results line up with `docs/pipeline_rules.md`'s tables | Every scenario |

**Reference configurations.** Two are reported side by side in every table:
`ship` = `gibbs_both`, `zeros_keep`, `count_noise_on`, `pcs_tied`, the state
of the pinned code; and `candidate` = `split`, `zeros_drop`,
`count_noise_on`, `pcs_tied`, the configuration the real-data nulls found
calibrated. Naming it does not choose it: which configuration ships is the
user's (section 9.4), and so is the setting at which the auxiliary arms are
held (section 9.14).

### 5.3 Scenarios

| Axis | Levels in the first benchmark |
|---|---|
| Architecture | `null` (calibration); `sparse` k = 1 with log2 aFC per allele in {0.1, 0.2, 0.4, 0.8}; `infinitesimal` (alpha -0.5) with `burden_sd` in {0.05, 0.1, 0.2, 0.4} |
| Ancestry share | 0; 0.04 |
| Allelic biological variance | Default lognormal (median 0.10, mean 0.30, log-sd 1.48); high bracket (mean 0.8, log-sd 1.48, so median 0.27, near the point-value medians at depth) |
| Donor multipliers | Empirical (calibrated by inversion, section 3.4); none (all 1) |
| Failure modes | None first; then each Layer 5 process alone |

Calibration scenarios vary one nuisance axis at a time from the default
(null architecture, ancestry 0, default `tau2_g`, empirical multipliers, no
failure modes): four calibration scenarios. Power scenarios cross the two
non-null families with their four effect sizes at the default nuisance
setting: eight power scenarios, each mixing null and non-null genes at
`null_fraction` 0.5. Failure-mode scenarios (build step 10) add one process
at a time to the default calibration scenario and to one power scenario per
family.

### 5.4 Metrics, with exact definitions

A **gene-clustered interval** is computed by resampling genes with
replacement 2,000 times from child 7, carrying every pair (and every
replicate) of a resampled gene together, and taking the 2.5 and 97.5
percentiles of the recomputed statistic. Power intervals resample datasets
instead.

**Replicates.** A *sampling replicate* redraws Layers 2-4 (the random parts
of expression `u` and `eps`, background totals, fragments and their
placement, equivalence classes, VB point estimate and Gibbs draws) and
reruns the pipeline (edgeR, expression PCs, scans). The scaffold, the Layer 5
realization, the architecture and every nuisance parameter stay fixed within
a scenario; this is the same set the oracle holds fixed (section 3.8).
Replicate counts are in section 5.5. Every per-(gene, variant) statistic
over replicates is accumulated as sufficient statistics (count of finite
values, sum and sum of squares of the slope, sum and sum of squares of the
standard error, counts below each alpha), so no per-replicate table is kept.

**Nominal-p calibration.** For every tested gene-variant pair of a null gene
and channel c in {allelic `pval_a`, total `pval_t`, combined
`pval_nominal`}, the rejection rate
`R(alpha) = #{pairs with finite, positive channel se and p < alpha} /
#{pairs with finite, positive channel se}` at alpha 0.05, 0.01 and 0.001,
with gene-clustered intervals. The excluded pairs (channel off by the sparse
rule, or no heterozygous informative donor at the variant; section 2.4,
item 6) are counted and reported, and the rate with them included at p = 1 is
reported beside it for comparability with the stored real nulls
(`corrected_null_store.py` and `gene_set_weighting_null.py` count every pair
with a finite p). Two nulls, reported side by side:

- *Sampling null (the truth).* R_null sampling replicates of the calibration
  scenario, as defined above. This is the distribution the nominal p-value
  claims to follow.
- *Permutation null.* On each of K_perm sampling replicates, the
  `records_signflip` stream (N = 92, `perms[:200]` and `flips[:200]` of the
  stored `permutations.npz`) applied outside the pipeline as
  `scripts/corrected_null_store.py` does: permute together the donor columns
  of `A` (times the flips), `T`, the arm's `f(Va)` and `f(Vt)`, `keep_a_df`
  and `keep_t_df`, the RNA-tied covariates and, in the `donor_quality` arm,
  `d_i`; apply the flips to `A` only; keep the genotypes and the genotype PCs
  in place (or move the PCs, per arm); call `map_nominal` per permutation.
  The rate is averaged over the K_perm replicates and compared with the
  sampling rate. Because of its cost it runs for the arms named in section
  5.5 (the genotype-PC grid and the realism test's arms), and for others on
  request.

A configuration is calibrated when nominal lies inside the interval at all
three levels. The difference between the two nulls, paired on genes, is the
measurement for the genotype-PC permutation rule (section 9.6). Allelic rates
are also reported by informative-donor count in the half-open bins [2, 20),
[20, 40), [40, 60), [60, 92] (the allelic channel stays on at 2 or more,
section 2.4, items 5 and 7). Variant sets: all variants the pipeline tests,
and separately the null-instrument set (MAF at least 0.05, outside every
tested gene's body) so results line up with `docs/pipeline_rules.md`'s
tables. In failure-mode scenarios the rates are also reported by the
`Truth` gene and record classes (imprinted, monoallelic, paralog,
confident-but-wrong, altered-phase records), so a method is scored on not
calling them. The reference-bias gate's verdict (section 2.2) is reported per
dataset: its flag rate without reference bias is its false-alarm rate, its
flag rate at each `phi_ref` its sensitivity, and the rates above are also given
with the gate bypassed (the driver always proceeds).

**Standard-error accuracy (stated over true).** For gene g, variant j and
channel c, over the sampling replicates: `stated = sqrt(mean of se^2)`
(`slope_a_se`, `slope_t_se`, `slope_se`); `true` = standard deviation
(ddof = 1) of the slope over the same replicates; ratio = `stated / true`.
Pairs enter when their values are finite in more than half the replicates
and `true > 0`, the inclusion rule of `scripts/gene_set_weighting_null.py`
`summarize()`, which produced the real-data tables this metric is compared
with. Summarized as the median over (gene, variant) pairs of null genes, with
a gene-clustered interval. Under a correct model the root-mean-square ratio
is unbiased, while the mean of `se` carries the downward bias of a
square-rooted estimated scale (about 0.988 at 20 degrees of freedom, typical
of the allelic channel's informative counts); the mean-based ratio
(`mean se / true`) is therefore reported only as a secondary column. A
permutation version (the same over the 200 permutations of each K_perm
replicate, for the arms that carry a permutation null) is reported beside it,
and every arm's ratio is also reported
divided by the `oracle` arm's ratio on the same data.

**Efficiency against the oracle.** For each (gene, variant):
`E = Var_rep(slope_oracle) / Var_rep(slope_arm)`, variances over sampling
replicates; reported as the median over variants within a gene, then the
median and deciles over genes, per channel and combined. The share of the
unit-to-`1/v` gain retained is computed per channel from sums pooled over
(gene, variant) pairs,
`(sum Var_unit - sum Var_arm) / (sum Var_unit - sum Var_gibbs)`, with
`unit_both` and `gibbs_both` at the arm's zero handling, and a gene-clustered
interval. It is reported only where the interval of its denominator,
`sum Var_unit - sum Var_gibbs`, excludes 0; otherwise the raw ratios
`sum Var_arm / sum Var_unit` and `sum Var_arm / sum Var_gibbs` are reported
instead. This matters in the total channel, where on the real null the Gibbs
weights' realized spread is about 1.07 times the unit weights' (unit over
Gibbs 0.936), so the denominator is near zero or negative. The pre-registered
0.70 bar (section 1.2) applies to the allelic and combined slopes, for arms
proposed as candidates.

**Power at empirical FDR.** On power datasets, eGenes are called from
`map_cis` `pval_beta` (the gene-level p-value from a Beta distribution fitted
to the permutation minimum p-values) and, separately, `pval_perm`. The
realized false-discovery proportion is `#null genes called / #genes called`,
using the truth. Two readings:

- *Calibrated procedure.* Storey q-values (Benjamini-Hochberg adjusted
  p-values scaled by `pi0`, the estimated share of null genes,
  `#{p > 0.5} / (0.5 m)`) at 0.05; report power (`#non-null called /
  #non-null`) and the realized FDR averaged over datasets.
- *Ranking only.* Pool genes across datasets, rank by `pval_beta`, and report
  power at the threshold where the realized FDR is 0.05. This separates
  ranking quality from calibration.

Power is stratified by causal MAF (section 3.3) and by the gene's
distinguishable fraction `phi_g` in the bands <0.03, 0.03-0.10, >=0.10
(design A asks that the allele-informative fraction be reported beside
power); false positives are also stratified by the `Truth` classes. mixQTL
arms use their own empirical p-values (section 2.2).

**Allelic fold-change recovery** (headline for the sparse family). Truth
`d_obs` (section 4). Estimate `d_hat_i = slope_lead x s_i,lead` from
`map_cis`'s lead variant (with `tau_refit=True`, the refit slope) on the
observed haplotypes. Over heterozygous-informative donors of non-null genes:
squared Pearson correlation, and the calibration slope from regressing
`d_obs` on `d_hat_i`. Also reported: lead-variant concordance (lead equals the
causal variant; lead in LD r^2 >= 0.8 with it), and the raw allelic value
`A_i` against `d_obs` as a no-model reference, stratified by causal MAF.
Under the infinitesimal family the pipeline has no per-individual estimator
beyond the single lead, so the same numbers are reported and labelled as the
single-lead share. Design A argues the infinitesimal family is where
hapmixQTL should be judged; the evidence the first benchmark offers on that
claim is power at empirical FDR and raw `A_i` against `d_obs` under the
infinitesimal family, and held-out polygenic prediction waits for
fine-mapping (section 9.7). The reordering relative to design A is Appendix
A, row 22.

**Cis-variance recovery** (headline for the sparse family, as a single-lead
share otherwise). Truth on two scales. Model scale:
`V_G_t = Var_i(log2(2^b_iL + 2^b_iR))` and allelic `V_G_a = Var_i(d_obs,i)`
over heterozygous-informative donors. Pipeline scale, because `slope_t` is
estimated on log2(CPM + 1):
`V_G_t,pipe = Var_i(log2(c_g (2^b_iL + 2^b_iR) / 2 + 1))`, the noise-free
change in `T` implied by `b` at the gene's baseline CPM. Estimate from the
lead: `(slope^2 - se^2) x Var_i(x_i)`, with `x = g/2` (total) or `s`
(allelic). Report estimate over the pipeline-scale truth by family and effect
size, with the headline restricted to genes at `c_g >= 16` (where the two
scales differ by at most about 12% in variance, `(16/17)^2` = 0.886); below 16 CPM the ratio
`V_G_t,pipe / V_G_t` is reported separately as a property of the pipeline's
unit, not as estimator failure. Beside `V_G`, the realized cis heritability
`h2_cis,g = V_G_t / Var_i(log2 of the donor's true total expression)`, over
all Layer 2 terms, is reported rather than assumed. The pipeline has no
multi-variant cis-variance estimator; say so wherever this metric is shown.

**Gibbs-variance fidelity.** Per band, the rank correlation of `Va` with the
true technical variance of `A`, and the mean of `(A - E[A | truth])^2 / Va`.
The technical variance is a separate Monte Carlo (child 9): on each of the
K_perm replicates, 100 re-draws of Layers 3-4 at that replicate's fixed Layer
2 realization. This tests design A's requirement that `Va` tracks the true
ambiguity.

**Weight-residual coupling.** Per null gene, per dataset and per arm, in the
allelic channel (through the origin with no nuisance columns, so the
residual is `A` itself): `w = 1/f`, `z^2 = A^2 / f` over the admitted records,
and `R_g = mean(w z^2) / (mean(w) mean(z^2))`, as
`scripts/weight_residual_coupling.py` defines it (`rcoup`). `R_g` is 1 under
the model and is, to first order, the realized-over-reported variance of the
permuted slope. Its band is the per-gene 2.5 and 97.5 percentiles of `R_g`
under model records (`z ~ N(0, 1)` at the realized weights, 10,000 draws per
gene), as that script computes it. Reported: the number of genes outside the
band per dataset, against its model distribution (Binomial(G_null, 0.05) for
a two-sided 95% band), and the across-gene standard deviation of log `R_g`
against its model distribution (on real data 1.75 times the model null). The
pre-registered criterion "in band for all but about 1 of 46 genes" refers to
the 46 real instrument genes; its tested-set analogue here is the
out-of-band count among the null tested genes compared with that model
distribution. The total channel's analogue (`R_t` on leverage-corrected
residuals) is reported as secondary.

**Cis/trans test calibration.** `pval_cis_trans` (`cis_trans_diagnostic`,
`hapmixqtl.py:1650`), which `map_nominal` writes for every pair: rejection
rates on null genes and on the non-null genes of power datasets, where the
simulator's truth is purely cis so the test's null holds, under no failure
modes and under each phase-error process of section 3.7 (Hu 2015 found that
imputed phase inflates this test; `tests/ase_robustness.py`, part B, is the
existing phasing-error harness).

**Robustness curves** (after the first benchmark): nominal-p calibration,
standard-error accuracy, efficiency and power against the switch-error rate,
the per-site flip rates, a multiplier on the distinguishable fraction
`phi_g`, the paralog shared fraction and the reference-bias `phi_ref`.

### 5.5 Replicate counts, precision and compute budget

| Quantity | Default | Precision it buys |
|---|---|---|
| R_null, sampling replicates per calibration scenario | 200 | Relative standard error of a per-pair standard deviation about `1/sqrt(2 (R_null - 1))` = 5.0%. Clustered by gene over 200 genes, the median stated-over-true ratio should carry a gene-clustered interval of roughly +/-0.01, so the acceptance test *Oracle honesty* can resolve a 0.03 departure. Rejection-rate intervals at 0.05 should be about 1/sqrt(2) the width of the real 100-gene, 200-permutation store's ([0.0481, 0.0558] for `split`) |
| R_power, power datasets per power scenario | 50 | 100 non-null genes per dataset, 5,000 per scenario; the standard error of power across datasets is about 0.007 at power 0.5 |
| K_perm, sampling replicates carrying a permutation null | 5 | 200 permutations each; the mean permutation rate over five realized nuisances, so one dataset's `eff_lib`, rebuilt PCs and draw of `u` do not stand in for the whole null |
| Oracle re-draws | 200 per calibration scenario (minimum 50 included); 100 per power dataset on 10 datasets (minimum 25) | Relative standard error of each oracle variance about 10% (14% at 100) |
| Technical Monte Carlo (*Gibbs-variance fidelity*) | 100 re-draws on each of the K_perm replicates | |
| `nperm` for `map_cis` | 1,000 | `pval_perm` resolution 1/1,001; stream sharing (section 2.4, item 1) |

**Cost per replicate** (single process, to be replaced by measured times at
build step 8 and recorded):

```
t_rep = t_emul(G x 92 pairs: VB + 200 draws x 16 rounds)
      + t_bg(Poisson totals for G_bg genes) + t_edgeR + t_pcs
      + n_arms x t_nominal(G) + n_mixqtl_arms x t_mixqtl(G, nperm)
      + (power datasets) t_mapcis(G, nperm) x n_arms_mapcis
```

The one measured anchor is `t_nominal`: the corrected null store ran 400
`map_nominal` calls on 100 genes (about 4,900 tested variants per gene) in 33
minutes, about 5 s per call, so about 10 s per arm at G = 200 and about
250 s per replicate for the roughly 25 arms of section 5.2. `t_emul` is not
measured; it is timed on the three donors' real equivalence classes at build
step 4 and recorded. Per calibration scenario the budget is
`R_null x t_rep + 200 x t_emul` (oracle) `+ K_perm x 200 x n_perm_arms x
t_nominal(G)` (permutation null). The permutation null is the dominant term,
so it runs only where a question needs it: the eight arms of the genotype-PC
grid on its two scenarios, and the five arms of the acceptance test
*Real-null reproduction*, 17 arm-scenario combinations after removing
overlaps; other arms get it only on request. That is about
17 x 5 x 200 x 10 s, roughly 47 hours single-process (less where the
realism test's 100-gene subset applies). Replicates are independent, and the
driver runs them in parallel processes. It computes the full first-benchmark
estimate from measured times and presents it for the user's approval before
launching (section 9.9). R_null, R_power,
K_perm, the oracle and technical Monte Carlo counts, `nperm`, and the
measured per-step times are recorded in `provenance.json`.

---

## 6. Tiering

### 6.1 The simulator tier (new, primary)

Sections 3-5: the real cohort's genotypes, exons, library sizes and
covariates, the full layer stack, and the pinned pipeline. This tier answers
the decisions of section 1.2.

### 6.2 The competitor-model tier: `tests/ase_external_benchmark.py`, kept

Its generator (`simulate_locus`: one HWE variant, negative-binomial totals,
beta-binomial allele-specific counts) and its three likelihood-ratio
comparators (`trec_lrt`, the negative-binomial total-count test; `ase_lrt`,
the beta-binomial allele-specific test; `trecase_lrt`, the joint test with a
shared fold change) draw data from the TReCASE/RASQUAL generative model,
which hapmixQTL does not assume. That is the reason to keep it (reply C,
pushback 2). `trecase_lrt` is the joint likelihood of the generating model
itself, so in this tier TReCASE is a ceiling, not a peer: parity with it is
parity with the best a method can do on these data (the 2026-09-23 record
says "parity with a ceiling"). Changes:

1. **Supersede `hapmix_pval` in the reported methods; keep it frozen.** Its
   defects, verified: it clips `Va` to 1e-8 at line 309 before the
   `va_t > 1e-12` guard at line 328, so the degenerate-allelic guard never
   fires (at `mu = 20`, six donors with no allele-specific reads took
   0.99999976 of the allelic weight); it emulates draws that conserve
   `yL + yR` and calls `compute_summaries_from_gibbs` without `yT`, which
   overstates the total channel's Gibbs variance about 4.0x; its phenotype is
   natural-log `log(T/lib + 1)`; it keeps an intercept on the allelic
   residualizer (production is through-origin); it bypasses `map_nominal` and
   uses `N - 2` degrees of freedom; it imports the quarantined
   `_estimate_tau`. Other harnesses import it, so it stays byte-identical as
   a frozen legacy function and is no longer reported; the new arm is a new
   function. Dispositions:

   | Dependent | What it imports from the harness | Disposition |
   |---|---|---|
   | `tests/ase_reference_bias.py` | `trec_lrt`, `trecase_lrt`, `hapmix_pval`, `trec_multiplier` | Pinned to the frozen `hapmix_pval` (its hapmixQTL arm is `tau='estimate'`, quarantined, so its numbers are historical); its reference-bias sweep is migrated to the Layer 5 reference-bias process when that lands |
   | `tests/ase_rasqual_comparison.py` | `trec_lrt`, `trecase_lrt`, `hapmix_pval`, `trec_multiplier`, the parameter bounds, `DTYPE` | Pinned to the frozen function; migration deferred |
   | `tests/ase_rasqual_real.py` | `hapmix_pval`, `trecase_lrt`, `trec_lrt` | Pinned to the frozen function; migration deferred |
   | `tests/ase_published_designs.py` | `simulate_locus`, `trec_lrt`, `ase_lrt`, `trecase_lrt`, `DTYPE` (and `compute_summaries_from_gibbs`, `_estimate_tau` from `hapmixqtl` directly) | Its design grids are adopted below (it states that it supersedes this harness's parameters); its own hapmixQTL arm uses the pre-2026-09-25 path and is retired from reporting |
   | `tests/ase_compute_benchmark.py` | `simulate_locus`, `trec_lrt`, `trecase_lrt`, `DTYPE` | Unaffected: the unflagged generator path is unchanged |
   | `scripts/allelic_overdispersion_floor.py` | the module (`B.compute_summaries_from_gibbs`, re-exported) | Unaffected as long as the harness keeps its import of `compute_summaries_from_gibbs` |

2. **The new arm.** Build equivalence classes from the simulated counts:
   allele-specific reads are strictly distinguishable (`d_L`, `d_R` from the
   beta-binomial split), the remaining `T - n_as` fragments are ambiguous
   between the two copies, and a donor homozygous at the variant has one
   collapsed copy (no allelic information). Run `SalmonEmulator` for the point
   estimate and draws (so `yT` carries Gamma shot noise), take library sizes
   as the simulated offsets times the BrainVar median effective library size
   (17,172,092; the constant matters because of the `+1` in log2(CPM + 1)),
   call `summaries_from_point_estimates` with `yT`, and run `map_nominal` on a
   one-variant locus with the default-mode arguments of section 2.2. The
   weighting arms of section 5.2 apply unchanged.
3. **Take the deprecated arms out of the reported `METHODS`** (`tau='zero'`
   with the known-variance standard error, `tau='estimate'`, `tau+fitted`):
   the 2026-09-23 deprecation record
   (`brainvar_hapmix_deploy/deprecated_models/README.md`) forbids reporting a
   comparison against a quarantined configuration. The file keeps them as
   documented controls; they may stay behind a flag that is off by default.
4. **Flag censoring in `calib()`**: report a boolean beside `lambda_GC` when it
   sits at the 3019.92 ceiling set by clipping p at 1e-300.
5. **Parameter settings.** The harness's current setting (N 200, `mu` 200,
   `phi` 0.2, `rho` 0.01, `as_frac` 0.25) is its own and unanchored (its
   docstring says the published values were unreachable when it was
   written). Keep it as the regression setting. Every generator extension
   the published settings need sits behind a flag that is off by default, so
   the regression setting runs the unflagged path byte-identically
   (acceptance test *Competitor-tier regression*):

| Setting | N | Total model | Allele-specific fraction | Allelic overdispersion | Other | Generator change, behind a flag |
|---|---|---|---|---|---|---|
| Harness regression | 200 | NB, mean 200, `phi` 0.2 | 0.25 | 0.01 | MAF 0.3 | None |
| Sun 2012 (TReCASE) | 65 (also 130, 100) | NB, overdispersion 1.0; mean 1,000 (500, 650 at the larger N) | 0.005 | 0.1 | MAF 0.05 and 0.2 | None, after mapping each paper's overdispersion to the harness's parameterization |
| Hu 2015 | 79 and 500 | NB, mean `900 exp(0.1 Z + effect)`, dispersion 0.2 | 0.034 | 0.05 | One covariate Z | `--covariate`: a standard-normal donor covariate with coefficient 0.1 on the log mean, passed to every arm |
| van de Geijn 2015 (WASP) | 10, 20, 50, 100 | Beta-negative-binomial (Omega 0.01, phi 100) | 20 reads per heterozygote, fixed | 0.2 | 1% mis-called heterozygotes at p = 0.99; MAF 0.2 | `--bnb` (beta-negative-binomial totals) and `--fixed-as-count` |
| Zhabotynsky 2022 | 64 | NB, mean 100 | `round(0.1 x total)` | Between-sample Beta around logistic(+/-b0) | Balanced haplotype classes, not HWE | `--balanced-genotypes` |
| Liang 2021 (mixQTL), via `tests/ase_published_designs.py` | its settings | Read-position generator in the paper; the harness uses the NB/beta-binomial model with mixQTL's grid | Emergent | none | aFC grid 1, 1.01, 1.05, 1.1, 1.25, 1.5, 2, 3; 200 replicates per setting | None beyond the aFC grid; read-position generation is the simulator tier's |
| Kumasaka 2016 (RASQUAL), via `tests/ase_published_designs.py` | 5, 10, 25, 50, 100 | Per-feature empirical parameters in the paper | Empirical | Empirical | Power at empirical FPR 10% | None beyond the N grid; empirical parameter draws are not reproduced |

Each paper defines "overdispersion" in its own parameterization; the
implementer records the mapping to the harness's `phi` (`Var = mu + phi mu^2`)
and `rho` for every row, and marks any row whose definition cannot be
confirmed from the paper or its code.

### 6.3 Tier 3, deferred: real Salmon on simulated reads

Write the diploid transcriptome from the observed haplotypes, simulate FASTQ
from the true haplotypes, and run Salmon 1.10.3 with the production flags on
a few hundred genes, to check the emulator's `Va` distribution and to test
`--useEM` for zero-haplotype handling (section 9.3). Blockers: Salmon 1.10.3
is not installed (a 1.8.0 binary and a 2.1.2 build are), and the budget is
open (section 9.9).

---

## 7. Acceptance tests before any benchmark is reported

Two kinds. **Correctness tests** check that the simulator and driver do what
this document says; no benchmark number is reported until every one passes on
the pinned commit. **Realism tests** check that the simulator reproduces the
real-data behaviour that motivates the weighting decision; their results are
always reported, and the benchmark informs the weighting decision (section
9.4) only if they pass (section 1.2). Tolerances marked "proposed" are this
document's judgement, not derived from a measurement; the user may tighten
them (section 9.12).

**Correctness tests.**

| Test | Pass condition | Where it runs |
|---|---|---|
| *Pin check* | `provenance.json` records the commit and the two blob hashes; the driver refuses a mismatch without `--allow-unpinned` | pytest |
| *Randomness isolation* | A dataset simulated before and after a `map_cis` call is byte-identical; the `map_cis` stream at N = 92, `seed=42` and the benchmark's `nperm` equals `permutations.npz` (at `nperm = 1000`, both the permutations and the flips; the driver refuses any other `nperm` in a result that claims the shared stream); addressing a stream by key gives the same draws regardless of which other streams were created first | pytest |
| *No-failure-mode identity* | With every Layer 5 rate at zero: `X_true == X_obs` at every site; no zeroed side has a strictly distinguishable read (exact, as on real data: 0 of 42,356); and the number of pair-sides with no strictly distinguishable read lies inside the 99% interval of its expectation computed from each side's expected distinguishable count (Poisson-binomial) | pytest (small) and `scripts/validate_salmon_emulator.py` |
| *Coupling mechanism* | Configuration: null architecture, `m_i = m_a_i = 1`, `lambda_g = gamma_g = 0`, `a_g = 0` (so each gene's biological variance is constant across donors and only the technical term varies). Statistic, reusing `scripts/variance_family_test.py`: within-gene decile bins of `v` (`bin_index`), the raked bin effects (`rake`: quasi-Poisson with gene fixed effects) of the deconvolved squared residual `y` (`deconvolve`, through the design's mixing matrix), and local slopes (`local_slopes`: change in bin effect over change in bin-mean `log(v / median_g v)`). Against observed `Vt`: U-shaped, first-decile slope below -0.4 and last above +0.4 (reference -0.89 ... +0.91, `simulation/coupled_total_homoscedastic/against_recomputed_v`; raw `e^2` version -0.72 ... +0.69 reported beside it). Against expected `v`, the shipped `Vt` formula evaluated at each record's expected count: every decile slope within 3 of the simulator's own replicate-to-replicate standard deviations of zero (reference -0.13 ... +0.09, `against_exogenous_v`, where the variance-family test used the fitted-value `v`). Allelic, against observed `Va` with `tau2_g = 0` (a binomial allelic truth): every decile slope inside the reference range 0.61-1.02 (`coupled_allelic_binomial`) widened by 3 replicate standard deviations | `scripts/validate_simulator_layers.py` |
| *Oracle honesty* | No failure modes, sampling null: the root-mean-square stated-over-true ratio of the `oracle` arm has a gene-clustered 95% interval covering 1, per channel and combined. If the interval's half-width exceeds 0.03, R_null is raised before the test is read. Cross-check: for records with expected `pL, pR >= 30` (allelic) or expected `pT >= 30` (total), the median of `vstar` over the delta-method variance at the expected count lies in [0.9, 1.1] (proposed) | benchmark driver, calibration run |
| *Emulator band targets* | `scripts/validate_salmon_emulator.py` recomputes the banded targets of section 3.6 on the tested genes directly from the cache (the `part_cache` code of `scripts/simulator_layer4_calibration.py`, restricted to those genes), writes them beside its results, and applies these tolerances to them: zeroing share within 5 / 6 / 2.5 / 1.0 points in the four bands; Gibbs `Va` median within a factor 1.25; `Va / q_a` within a factor 1.3; Fano of `yT` in [0.97, 1.00]; zeroed-side draw share within +/-0.08; `(pL + pR)` / draw mean within +/-0.05 at 100+ reads (all proposed). Tie-zeroing is recomputed from the three donors' equivalence classes restricted to the tested genes and tested within +/-10 points only if at least 200 ties fall in those genes; otherwise it is left to the next test | `scripts/validate_salmon_emulator.py` |
| *Emulator on real equivalence classes* | Run the emulator's VB and chain on the dumped classes of the three calibration donors (563_D1, 591_D1, 657_D1) and compare with their real `quant.sf` and draws: no zeroed side with a strictly distinguishable read (at most 0.1% exceptions, allowing for merged range-factorized bins); empty-side zeroing 72.9% +/- 5 points; tie zeroing 96 / 87 / 69 / 48% by band within +/-10 points; median simulated over observed Gibbs `Va` in [0.90, 1.15] at 10+ reads (the weightless dump cannot carry soft weights, so one-SNV pairs are expected near 1.23); zeroed side's last-over-first-block draw-mean ratio 1.00 / 1.03 / 1.46 / 3.0 by band within +/-0.15 below 100 reads and +/-25% at 100+ (all proposed). The block ratio discriminates the Gibbs prior directly (section 3.6) | `scripts/validate_salmon_emulator.py` |
| *Permutation exactness* | On the `hwe` synthetic scaffold (section 3.2), where tested and exonic genotypes are drawn independently of every record property: ancestry share 0, null architecture, 500 or more genes, metadata covariates plus expression PCs built from the simulated totals, no genotype PCs. `pval_perm` must reject at 0.05 inside the binomial 99% interval and pass a Kolmogorov-Smirnov test (the largest distance between the empirical and uniform distribution functions) at p > 0.01, for every weighting arm; records are exchangeable given the genotypes there, so the permutation p-value is exact for any weights and a failure is a plumbing error. The same run on the BrainVar scaffold is a reported measurement, not a pass condition: there the allelic informative set and `Va` are fixed by each donor's own exonic heterozygosity, which is in LD with the tested variants, and fixed per-donor offsets (`F_i . lambda_g`, `M_i . gamma_g`) give the sampling null a fixed per-(gene, variant) component that permutation spreads at random, so records are not exchangeable given the genotypes; that difference is the genotype-PC permutation rule's question (section 9.6) | benchmark driver |
| *Pipeline input equivalence* | Simulated arrays pass every production guard; `summaries_from_point_estimates` on them equals an independent NumPy computation to 1e-12; homozygous-but-expressed pairs exist with `pL = pR = 0`, `pT > 0` | pytest |
| *Expression-layer recovery* | After the pipeline's own covariates, on the pipeline's log2(CPM + 1) scale: per CPM bin (all eleven, including <2 and 2-4), the median `s2` over every simulated gene (tested and background) within +/-10% of the target row of section 3.4, or within the real bin's gene-resampling 95% half-width where that is larger, for all 17 covariates and for metadata only; checked twice, against those real targets and against the values the simulator's own parameter table implies, so a generator error and a recovery error are told apart. On the tested genes: median `s2 / median_i Vt` 3.9 (all 17 covariates) and 15.7 (metadata only) within +/-25%. The statistic of `layer2/summary.json` `log_ratio_trend_linear`: across genes, `ln(s2_g / median_i Vt_gi)` (natural log; `Vt` as shipped, `count_noise=True`) regressed on log2 median CPM, slope 0.33 (all 17 covariates) and 0.43 (metadata only) within +/-0.1. Expression PCs remove 76% +/- 10 points of the metadata-only biological variance. Donor effect re-estimated with `donor_fit`: sd 0.59 +/- 0.1 before PCs, and after PCs 0.25 / 0.29 / 0.40 by CPM tercile, each within +/-0.05, with the per-donor values correlating with the real ones. Three-genotype-PC partial R^2 at ancestry share 0 equal to the rebuilt-null 0.133 +/- 0.041 (tolerances proposed) | `scripts/validate_simulator_layers.py` |
| *Uninformative-pair structure* | On the calibration-gene sample: no-paired-transcript share equal to the real share for the same genes (26.2% on all calibration genes) to within 0.05% of pairs, or with every disagreement being one of the pairs flagged in `vcf_summary.json` (3 of 18,400 there); total `pL + pR = 0` share 27.2% +/- 2 points (proposed) | `scripts/validate_simulator_layers.py` |
| *Layer unit tests* | This document's own unit tests for Layers 1, 2, 5 and 6: centred burdens have mean 0 over the 2N haplotypes and the grid's sd; a sparse causal variant's effect equals the grid beta; Layer 2 variances and the ancestry share are recovered by moments on a large synthetic cohort; Layer 5 at zero rates is the identity; the switch process flips relative phase with probability `(1 - exp(-2 r d)) / 2` at distance `d`; reference-bias thinning gives reference fraction `phi_ref` at balance; `d_obs == d` without failure modes; the chain-order rule (draw `j` in chain `j // 25` at position `j mod 25`). If the attachments arrive, their 14 tests, adapted to log2 and to the new Layer 4, are added | pytest |
| *Competitor-tier regression* | On the unflagged generator path, the TReCASE arm at the harness regression setting, 500 replicates, `seed0` 1000 for the null and 5000 for the alternatives, fold changes 1.05 / 1.10 / 1.20, reproduces its recorded matched power 0.274 / 0.760 / 0.998 exactly to the printed digits (`brainvar_hapmix_deploy/external_benchmark_fitted_defaults_20260923/`). The harness reseeds every replicate (data `RandomState(seed0 + r)`, arms `RandomState(seed0 + 900000 + r)`), so with the generator and the TReCASE arm untouched the numbers must reproduce exactly; the test records `seed0`. The new hapmixQTL arm is expected to change | `tests/ase_external_benchmark.py` |

**Realism tests.**

| Test | Pass condition | Where it runs |
|---|---|---|
| *Real-null reproduction* | On the simulated 100 corrected-null-store genes (default calibration scenario, no failure modes), the permutation null of *Nominal-p calibration* with `perms[:200]` and `flips[:200]` on each of the K_perm replicates, for `gibbs_both` (zeros dropped), `split`, `unit_both`, `plus_one`, and `gibbs_both` with `pcs_moving`. Pass when, for every arm, the per-channel rates at 0.05 / 0.01 / 0.001, averaged over the replicates, lie inside the real gene-clustered intervals (table below, read from the source files), and when the per-gene unit-over-Gibbs realized-sd ratio distribution (median over a gene's tested variants, as `scripts/gibbs_weight_benefit_by_gene.py` computes it) has its 10th, 50th and 90th percentiles inside the gene-resampling 95% intervals of the real ones (real: total 0.70 / 0.93 / 1.08; allelic 1.11 / 1.38 / 2.19; `gibbs_weight_benefit_by_gene_20260926/per_gene.tsv`). On failure, the report states that the simulator lacks the mechanism behind the real weighting failure and cannot inform the weighting decision until it is added; the candidates are the per-gene weight-residual coupling with the real across-gene spread of log `R_g` (1.75 times its model null), a heavy-tailed `eps`, and a per-donor variance component of the kind seen in donor 221_D1 | benchmark driver |
| *Variance-profile reproduction* | At the default calibration scenario, the within-gene-decile local-slope profiles (the functions of *Coupling mechanism*) against observed `v` and against fitted-value `v`, in both channels, computed on the simulated tested genes, each lie within the real profile's 95% gene-resampling interval computed on the same genes with the variance-family test's own code, widened by the simulated slope's Monte Carlo standard error over replicates. Full-cohort references (`variance_family_test_20260926/summary.json`, `bins.within_gene_decile.y.local_slope`): total against observed `v` -1.41 ... +2.28; total against fitted-value `v` -0.37 ... +1.45, not flat, with the top three deciles 0.62, 0.66 and 1.45 and intervals excluding 0; allelic against observed `v` 0.85 ... 1.34, outside the 0.4-1.1 band of the homoscedastic mechanism; allelic against fitted-value `v` -1.16 ... +0.20. Failure has the same consequence as above | `scripts/validate_simulator_layers.py` |

Real targets for *Real-null reproduction* (rates at 0.05 / 0.01 / 0.001; the
intervals not printed here are read from the source files by the test):

| Arm | Allelic | Total | Combined | Source |
|---|---|---|---|---|
| `gibbs_both`, zeros dropped, genotype PCs tied | 0.0455 [0.0405, 0.0506] / 0.0103 / 0.0026 | 0.0835 [0.0752, 0.0923] / 0.0261 / 0.0066 | 0.0719 [0.0656, 0.0793] / 0.0206 / 0.0054 | `corrected_null_store_20260925/summary.json` |
| `split` | as `gibbs_both` | 0.0503 / 0.0104 / 0.0011 | 0.0512 [0.0481, 0.0558] / 0.0117 / 0.0027 | `hybrid_weights_null_20260926/` |
| `unit_both` | 0.0546 / 0.0136 / 0.0034 | 0.0503 / 0.0104 / 0.0011 | 0.0529 / 0.0122 / 0.0028 | `hybrid_weights_null_20260926/` (`--config=unit`) |
| `plus_one` | 0.0419 / 0.0095 / 0.0027 | 0.0503 / 0.0104 / 0.0012 | 0.0498 [0.0475, 0.0538] / 0.0112 / 0.0026 | `hybrid_weights_null_20260926/summary_plus_one.json` |
| `gibbs_both`, genotype PCs moving | | 0.0662 / 0.0165 / 0.0025 | | `total_channel_decomposition_20260926/` |

---

## 8. File layout and build order

### 8.1 Files

| Path | Status | Contents |
|---|---|---|
| `tests/hmm_genotype_simulator.py` | Bring from master by path | fastPHASE-style hidden-Markov genotype simulator (`simulate_hmm_genotypes`, `make_recombination_map`, `ld_decay`), used for unit tests and scale experiments. Bring it with `git show master:tests/hmm_genotype_simulator.py > tests/hmm_genotype_simulator.py` and confirm `git hash-object` of the result equals `git rev-parse master:tests/hmm_genotype_simulator.py` (blob `05844e3`, introduced by `061ed7c`). It uses a local `default_rng(seed)`; pass a child seed |
| `tests/hapmix_simulator.py`, `tests/test_hapmix_simulator.py` | New, written to section 3; if the attachments are provided in time (section 3, "Status of the interfaces"), they are committed verbatim in their own commit first and amended in later commits so the diff shows each change this document requires | `GenotypeScaffold`, `Architecture`, `ScenarioGrid`, `ExpressionModel`, `CountModel`, `QuantificationModel`, `Artifacts`, `Truth`, `SimDataset`, and the *Layer unit tests* |
| `tests/brainvar_scaffold.py` | New | `BrainVarScaffold` (Layer 0) and the `hwe` synthetic scaffold |
| `tests/salmon_quant_emulator.py` | New | `TranscriptModel`, `ObservedIndex`, `FragmentClassBuilder`, `SalmonEmulator` (Layers 3-4); shared with the competitor tier |
| `tests/simulation_benchmark_arms.py` | New | Arm definitions (`Va`/`Vt` transforms, masks, covariate splits), the `donor_quality` estimator, and the calls into the pipeline |
| `tests/simulation_benchmark_metrics.py` | New | The metrics of section 5.4 and the gene-clustered interval |
| `tests/test_salmon_quant_emulator.py`, `tests/test_simulation_benchmark.py` | New | The fast correctness tests (*Pin check*, *Randomness isolation*, *No-failure-mode identity* small, *Pipeline input equivalence*, *Layer unit tests*) and unit tests of the Salmon-source rules |
| `tests/ase_external_benchmark.py` | Modified | Section 6.2 |
| `scripts/calibrate_simulator_nuisance.py` | New | Per-gene metadata coefficients, expression-PC loadings, shrunk biological variances (section 3.4), isoform proportions, and the donor multipliers by inversion, from the real point estimates and draws, written under `brainvar_hapmix_deploy/simulator_calibration_<date>/nuisance/` |
| `scripts/validate_salmon_emulator.py` | New | *No-failure-mode identity* (full), *Emulator band targets*, *Emulator on real equivalence classes* |
| `scripts/validate_simulator_layers.py` | New | *Coupling mechanism*, *Expression-layer recovery*, *Uninformative-pair structure*, *Variance-profile reproduction* |
| `scripts/run_simulation_benchmark.py` | New | Driver: scenarios by arms, outputs under `brainvar_hapmix_deploy/simulation_benchmark_<date>/` with `provenance.json` |
| `scripts/analyze_simulation_benchmark.py` | New | Metrics tables and an HTML report with figures (the user's reporting convention) |
| `scripts/simulator_layer2_calibration.py`, `scripts/simulator_layer4_calibration.py`, `scripts/variance_family_test.py` | Committed (`d9a9f16`) | The provenance of the defaults |

The simulator modules in `tests/` import NumPy, pandas, `scipy.special` and
`scipy.stats` only (section 3.0); the arms and driver import the pipeline.
The repository's `.gitignore` ignores `*.md` (only `README.md` is excepted),
so this document and any new Markdown file under `docs/` must be added with
`git add -f`, as the existing tracked docs were.

### 8.2 Build order

Each step is gated by the acceptance tests named in it.

1. **Prerequisites.** Bring the HMM simulator from master (the three
   calibration scripts are already committed). If the attachments have been provided, commit them
   verbatim in their own commit; this is the only step that may reference
   them, and nothing later waits for them.
2. **Pin and randomness** (*Pin check*, *Randomness isolation*).
3. **Layer 0 scaffold**, both kinds, and `calibrate_simulator_nuisance.py`
   (except the donor multipliers, which need step 6). The attachments'
   fallback point: if they have not arrived by the end of this step, Layers
   1, 2, 5 and 6 are written from this document alone.
4. **Layer 4 on real equivalence classes** (*Emulator on real equivalence
   classes*), before any simulated read, because it can be checked against
   real Salmon output independently of the rest of the simulator. Read the
   VB and Gibbs rules (Appendix B, items 6 and 6a) from the Salmon 1.10.3
   source first. Time the emulator here (section 5.5).
5. **Layer 3 read and class construction**; then *No-failure-mode identity*,
   *Emulator band targets*, *Uninformative-pair structure*.
6. **Layer 2 expression**, then the donor multipliers by inversion;
   *Coupling mechanism*, *Expression-layer recovery*.
7. **Layer 1** with `null`, `sparse` k = 1 and `infinitesimal` (reply C,
   pushback 1).
8. **Arms and metrics**; *Oracle honesty*, *Permutation exactness*, *Pipeline
   input equivalence*, *Layer unit tests*; then the realism tests
   *Real-null reproduction* and *Variance-profile reproduction*, whose
   outcome decides whether the first benchmark can inform the weighting
   decision. Measure per-step times and present the budget (section 5.5).
9. **First benchmark**: the grids of section 5.2 on the scenarios of section
   5.3 without failure modes, after the user approves the budget.
10. **Layer 5 failure-mode arms**, one at a time.
11. **Competitor tier revision** (section 6.2; *Competitor-tier regression*).
    It needs only the emulator, so it can proceed in parallel from step 5.
12. **Deferred**: fine-mapping and knockoffs once `map_susie` reaches default
    mode; tier 3; per-SNP count writers for TReCASE and RASQUAL; other N from
    the HMM scaffold.

This merges critique B's order (Layers 0 and 2-4 first) with reply C's
insistence on both non-null families from the first benchmark, and moves the
emulator's real-data check ahead of simulated reads.

---

## 9. Open decisions that belong to the user

Each is stated with its options; this document takes no position on them.

**9.1 Where the benchmark lives (settled 2026-09-26).** The user chose the
branch `simulation-benchmark`, pushed to origin at the pinned commit.
Simulator work is committed there, in its own worktree, so the session that
works on `mixqtl-replication` is not disturbed.

**9.2 Tool names in file names, paths and branch names (settled 2026-09-26).**
The user decided these are not to be changed.

**9.3 Zero-haplotype handling in the shipped pipeline.** Keep point
estimates and let the weights handle them; drop them from the allelic channel;
re-quantify with `--useEM`; or use a different point summary for the allelic
split only. The simulator measures keep and drop; `--useEM` needs tier 3.

**9.4 Which weighting configuration ships.** The options of
`docs/pipeline_rules.md`, with the benchmark's calibration, standard-error,
efficiency, coupling and power metrics as new evidence, on the condition of
section 1.2 (both realism tests passing).

**9.5 The counting term for donors with reads.** Keep `count_noise=True`;
adopt the floor only where the draws give no total variance (arm
`readless_floor_t`); and, separately, whether the allelic counting term stays
(arm `no_q_a`).

**9.6 The genotype-PC permutation rule.** Tied (current); moving; or tied with
a correction if the benchmark shows the tied null departs from the sampling
null only when an ancestry term is present.

**9.7 `map_susie` and fine-mapping provenance.** `map_susie` needs an
`se_mode`, and `fine_mapping_provenance` must be redefined now that
`tau_mode='zero'` is the shipped mode; the change flips
`TestMapSusie::test_map_susie_records_tau_mode_provenance`. Until this is
decided, fine-mapping, held-out prediction and knockoff eGene FDR are out of
the benchmark.

**9.8 The 1,208 calibration genes without Gibbs draws.** Extend `load_counts`
to collect total-channel draws for pair-less genes (a cache rebuild of about
45 minutes), or accept 11,747 genes. The simulator follows the current rule by
default for tested genes and can include them behind a flag; all 12,955 enter
the totals either way.

**9.9 Tier 3 and the compute budget.** Whether to install Salmon 1.10.3 and
how many genes get real simulated FASTQ and quantification; and approval of
the first benchmark's compute budget, which the driver computes from measured
per-step times (section 5.5), including whether to lower R_null, R_power,
K_perm or the arm set to fit it.

**9.10 The `relative_floor` arm, and fitted-value-`v` arms.** Report
`relative_floor` as a measurement only, or allow it as a candidate despite
its scale invariance (section 5.2 caution); and whether `power` and
`relative_floor` should also run on fitted-value `v`, whose fitted value
carries the record's own `h_i e_i`.

**9.11 Architecture grid and the crossover question.** The sparse beta,
`burden_sd`, `alpha` and `pi` grids and a source for an empirical
cis-heritability pool (design A's open item). Design A's stated purpose is to
show the crossover point between sparse and polygenic architectures; with
only k = 1 and the infinitesimal family the first benchmark cannot. Options:
add `point_normal` with pi = 0.1 to the first grid; or declare the crossover
question out of scope for the first benchmark.

**9.12 Acceptance tolerances.** The tolerances marked "proposed" in section 7.

**9.13 The design document and attachments.** Whether design A is committed
beside this document (for example as `docs/simulation_benchmark_design.md`),
and whether the two attachment files will be provided before the fallback
point (end of build step 3), after which the modules are written from this
document alone.

**9.14 The pivot configuration.** At which setting the auxiliary arms of
section 5.2 are held: the default here is `zeros_drop`, `count_noise_on`,
`pcs_tied`, the setting of the real-data weighting nulls; the alternatives
are the `ship` setting (`zeros_keep`), or both. Choosing `candidate`'s
setting as the pivot is not a choice of what ships.

**9.15 Causal pool, gene sample and `nperm`.** `causal_pool` `'instrument'`
(default) or `'all'`, in which case lead-variant recovery is scored by LD
r^2 rather than identity, or both as arms; the tested-gene sample of section
3.2 (sizes of the null-store, stratified and zero-read strata); and
`nperm` 1,000 (benchmark default, shares the stored stream) against the
production 10,000, which changes `pval_beta` and the FDR results. All three
are recorded in `provenance.json`.

**9.16 The per-transcript-copy emulator.** Design A decided that the
quantification target is gene-level `yL`/`yR` and that the Gibbs emulator
stays a single haplotype mixture per gene. This document emulates Salmon per
transcript copy with isoform proportions (Appendix A, row 20). Options:
confirm the override; or keep design A's single mixture and accept the
misses listed in that row.

**9.17 Trans-only genes.** Include a trans-only gene class (large `z`
variance correlated with genotype through the ancestry arm) so the method is
scored on not calling it, as design A wants; or confirm the deferral of
section 3.7.

---

## Appendix A. How the three positions were reconciled

| # | Topic | Design A | Critique B | Reply C | Resolution | What settled it |
|---|---|---|---|---|---|---|
| 1 | The method the benchmark targets | Weights `1/(v_inf + tau)`, fixed-weight permutation, natural log | The current pipeline: point-estimate values in log2 CPM on edgeR library sizes, draws for variance only, `sigma^2 v` with fitted scale, `records_signflip`, genotype PCs tied | | B | The code at the pin; `docs/pipeline_rules.md` |
| 2 | Summaries inside Layer 4 | `compute_summaries_from_gibbs` | Layer 4 must emit the point estimate as well as draws; the point estimate zeroes one haplotype in 36.10% of informative pairs transcriptome-wide (46.06% at 10-99 reads, 92.69% at 1-9), against 1.66% and 0.01% on draw means | | B: `summaries_from_point_estimates` | Pipeline rule 1. B's transcriptome-wide point-estimate shares are reproduced by `cache_targets.json` `all_cache_genes` (34,457 genes, 1,294,098 informative pairs). On the 11,747 calibration genes (786,919 informative pairs, `cache_targets.json`) the share is 14.6% overall and 42.4% at 10-99 reads (89.5% at 1-9). The two sets differ in denominator: the calibration genes are more deeply covered, so the 1-9-read regime is barely represented in the benchmark and its boundary-zero conclusions hold for the calibration-gene universe only (section 3.2) |
| 3 | The Gibbs emulator | Stationary Beta with Binomial ambiguous reads | Draws must carry shot noise on totals | | Chain with Gamma draws started from the VB point estimate; Beta kept for unit tests | Layer-4 calibration: Beta reproduces medians to 2-4% but per-pair spread is 4.2x its floor; restart deficit 0.81; one-SNV excess 1.23; `yT` Fano 0.986-0.989 |
| 4 | Boundary zeros | | Inject as a failure mode | Emerge from the ML split when `d_R = 0` | Emerge, through VB on classes from true reads against the observed index; C's ML statement refined | `eq_class_summary.json`: empty side zeroed in 72.9%, ties zeroed in 81.6%, which ML cannot produce |
| 5 | The 58-59% of pairs with no allelic information | | Transcript-level identity with index deduplication | Emerges from whether any indexed transcript differs | Emerges, but it is a sum: 38.2% structural + 19.6% paired with no reads + 1.4% VB zeros | `names_pairing_summary.json`; `vcf_summary.json` (18,397 of 18,400 pairs follow the heterozygous-exonic rule) |
| 6 | Confident-but-wrong records | | Inject (CALM2 657_D1) | Emerge from singleton or indel errors in the index | Both: emergent through index errors, plus an explicit injection arm | Cohort evidence does not establish index errors as the cause (discordant records carry at most 10% of the coupling; singleton mis-phasing indistinguishable at 137 of 9,328) |
| 7 | Larger N | Pair real haplotypes at random | | Drop pair-resampling; use the 92 donors as they are | C; other N only from the HMM scaffold, labelled synthetic | Layer-2 ancestry analysis: the genotype-PC structure is two outlier donors plus construction, which pair-resampling would destroy |
| 8 | Infinitesimal family timing | Implemented | Build Layers 0 and 2-4 first | Keep sparse-1 and infinitesimal from day one | C | C's argument that a polygenic cis burden is itself ancestry-correlated expression |
| 9 | Ancestry term | | Add a term correlated with genotype PCs | The infinitesimal burden provides it | Both, sized by measurement: explicit share 0 by default and 0.04 as an arm; the burden's ancestry correlation measured on output | Layer 2: PC1 excess 0.016-0.041 at 0.8-2.1 null sd; the observed 0.23 is not ancestry |
| 10 | Weighting family | `1/(v + tau)` | | Test additive against power law in the cache | Pooled `tau` excluded; gene-relative floor and power law both carried as arms, applied to observed `v`; neither is the variance function. The allelic relative floor survived only on fitted-value `v` (on observed `v` its fit is no floor and loses to the power law by +7,851 deviance); the total one survived on both | Variance-family test: pooled `tau` loses in both channels on both `v`; every family rejected at the bin level (chi-square at least 43 on 8 df; at least 53 among the main analyses) |
| 11 | Layer 3 overdispersion | Negative binomial per haplotype | | | Poisson at Layer 3; extra-Poisson variance lives in Layer 2 | Gibbs `Vt` is 0.975 of Poisson; Layer 2's variances are the excess over the draws-only technical variance (section 3.4) |
| 12 | The external benchmark | | | Keep it as the competitor-model tier with the new emulator and published settings | C, with `hapmix_pval` superseded and frozen for its importers, generator extensions behind flags, TReCASE labelled as the ceiling | Verification: generator and likelihood-ratio arms sound; `hapmix_pval` has six defects (section 6.2) |
| 13 | Evaluation | Power, knockoffs, aFC, cis variance, fine-mapping, `Va` fidelity, robustness | Add nominal-p calibration and stated over true standard error per channel; weighting options first-class | Oracle ceiling, efficiency relative to it, per-donor quality weights | Adopted, with fine-mapping and knockoffs blocked, and with design A's ordering and scope changed (row 22) | `map_susie` cannot reach default mode |
| 14 | The null for eGene power | Permute whole haplotype pairs across individuals | `records_signflip` with genotype PCs held | | The shipped `records_signflip`, plus a sampling null as the truth reference | `records` equals genotype permutation by relabeling (pinned to 1e-9 by `tests/test_hapmixqtl_perm_scheme.py`); the sign flip is the default since 2026-09-25 |
| 15 | Units of truth | Natural-log aFC | log2 | | log2, with the model's log2 CPM kept apart from the pipeline's log2(CPM + 1) (section 3.4) | Project convention; `layer2/summary.json` units |
| 16 | Build order | Scaffold, runner, harnesses, writers | Layers 0 and 2-4 first | Both non-null families from day one | Section 8.2, with the emulator's real-data check first | |
| 17 | The counting term | | | Possible double count of Poisson variance through `count_noise` | Confirmed; arms `readless_floor_t` and `no_q_a`, run under the weightings that use `Vt`. The same double count sat in the layer-2 biological variance (`bio_full`) and is removed there (section 3.4) | The verification pass and the layer-2 calibration: Gibbs draws already carry shot noise (Gibbs `Vt` is 0.975 of Poisson), and the counting term adds it again for every donor with reads (shipped `Vt` 1.96x Poisson) |
| 18 | HMM simulator location | Beside the attachments in `tests/` | | Also on master | On master only; bring by path | `git log`: introduced by `061ed7c`, absent at the pin and on origin's branch |
| 19 | Tier 3 | Real Salmon for a subset | | | Deferred | Salmon 1.10.3 not installed |
| 20 | Quantification target | Decided: gene-level `yL`/`yR`, the Gibbs emulator a single haplotype mixture per gene | | | Override, pending the user's confirmation (section 9.16): per-transcript copies with isoform proportions | Homozygous-only reads of informative genes are 27% / 21% / 4.6% / 0.5% by band, which a single mixture per gene cannot produce; 38.2% of all pairs are structurally uninformative (`names_pairing_summary.json`), a transcript-level property |
| 21 | Phase error | A distance-dependent switch process between each donor's gene and its regulatory variants, the allelic channel's main exposure | | | Both processes: the distance-dependent switch process (swept over switches per Mb) and per-site flips at exonic heterozygotes (index errors); an earlier draft kept only the second | Design A; Hu 2015; `tests/ase_robustness.py` part B. No switch-error measurement exists (Appendix B, item 17) |
| 22 | Primary estimand and metric order | The per-individual log aFC is primary under every family; the infinitesimal family is where hapmixQTL should be judged; power, knockoffs, aFC and `V_G` first | Calibration and standard-error accuracy first | | Calibration first; aFC and cis-variance recovery headline the sparse family only; `point_normal` deferred; the crossover question is the user's (section 9.11). Under the infinitesimal family, power and raw `A_i` against `d_obs` are reported as the available evidence on design A's claim | The weighting decision needs calibration first; the pipeline has no per-individual or multi-variant estimator beyond the lead |
| 23 | Reference bias | Kept, "because the aligner still sees a reference" | | | Kept: the production gate runs in the driver, and a Layer 5 alternate-fragment-loss process is swept over `phi_ref`; an earlier draft deferred it on argument alone | `docs/ase_validation.md` section 7h (type-I 0.635 at phi = 0.60, on the since-deprecated `tau='estimate'`); the runner's gate (`run_hapmixqtl_from_salmon.py:1358`) |
| 24 | Variance-function exponent | | | `v^0.65` residual-variance growth (2026-09-25) | Superseded by the corrected-pipeline fits (total 0.35, allelic 1.2 on observed `v`), which set the `power` grid | The 0.65 was measured on the pre-correction pipeline's natural-log draw-mean values without library normalization; the variance-family test re-measured on the pinned pipeline |

## Appendix B. Parameters whose defaults are not yet calibrated

Each entry names the measurement that would set it.

1. **Per-transcript exon structure** (a structural input): locate the NCBI
   release-110 T2T annotation that built the personalized transcriptomes.
   Until then, the `gene_union` model with the homozygous-only share `u`.
2. **Homozygous-only read share `u`** (interim exon model): fit to the band
   table 27 / 21 / 4.6 / 0.5%.
3. **Read length**: from the donors' trimmed FASTQ or trimming logs.
4. **Soft-weight magnitude for one-difference fragments**: requantify two or
   three donors with `--dumpEqWeights` (or `--hardFilter` for unweighted
   classes).
5. **Salmon score constants** (mismatch 6 units, gap open 6, extend 2,
   `minAlnProb` 1e-5): verify against the 1.10.3 source and binary.
6. **VB prior, convergence and zero-truncation rule**: from
   `CollapsedEMOptimizer.cpp` at 1.10.3 (the 1.8.0 help documents
   `--vbPrior` 0.01 per transcript); validated by *Emulator band targets* and
   *Emulator on real equivalence classes*.
   6a. **Gibbs prior value, Gamma rate parameterization, and the recorded
   quantity** (allocated count or rescaled abundance): from
   `CollapsedGibbsSampler.cpp` at 1.10.3, pinned by a unit test; the
   zeroed-side block ratio (1.00 / 1.03 / 1.46 / 3.0 by band) is the
   acceptance target that discriminates the prior.
7. **The rule behind point-estimate zeros on both haplotypes with positive
   draws** (1.0% of calibration pairs): reproduced by rate only.
8. **Isoform proportions per transcript**: cohort-level shares from Gibbs
   draw means (deterministic, not yet computed).
9. **Metadata coefficients `gamma_g`, factor loadings `lambda_g` and the
   shrunk biological variances `B_g`**: deterministic from the real data
   (`calibrate_simulator_nuisance.py`, not yet run), including the prior
   variance `A_b` per CPM bin.
10. **Split of each gene's measured biological variance into cis-genetic and
    non-genetic parts**: `bio_draw_only_full` is an upper bound on the
    non-genetic term; no conditioning was done (multi-variant conditioning is
    excluded by user decision).
11. **Floor on `s2_u_g`** (`0.25 S(B_g)`, its only remaining role to keep the
    subtraction of `tau2_g/4` and the cis variance positive): a design
    choice; the share of genes where it binds is reported.
12. **Allelic biological variance `tau2_g`**: bracketed (gene means 0.20-0.49
    on draw-mean values, 0.65-0.84 on point values, at 1,000+ reads, with the
    draws-only Gibbs variance subtracted); not measurable below 300 reads;
    settled only by an allelic value free of point-estimator error, or by
    tier 3.
13. **Tail of the haplotype term `eps`**: Student-t degrees of freedom from the
    kurtosis of allelic residuals on draw-mean values at depth.
14. **Ancestry share per gene**: the per-gene distribution of genotype PC1's
    excess over the rebuilt null, which the construction inflation currently
    hides.
15. **Architecture grids** (sparse beta, `burden_sd`, `alpha`, `pi`, `k`) and
    an empirical cis-heritability pool (section 9.11).
16. **Age and sex interaction slopes** (deferred families; no pipeline
    interaction test).
17. **Phase error rates**: a switch-error rate per Mb for the distance-
    dependent process, and per-site error rates (trio, long-read or
    read-backed with an error model); the 19.1% and 2.0% figures are upper
    bounds of disagreement, not error rates.
18. **Genotype miscall rates** (heterozygote to homozygote and the reverse;
    singletons and indels separately): concordance of the HG002 call sets
    under `nf_stage/brainvar2/trio_benchmark_eval/` with the GIAB truth set.
19. **Confident-but-wrong injection rate**: no measurement of causes exists;
    0.82% discordant records is a symptom rate.
20. **Paralog shared fraction and partner structure** (default 5%): the
    measured 3.0-7.3% shared-class share gives a range, not a gene-level
    structure.
21. **Random monoallelic expression rate** per donor-gene pair.
22. **Imprinted-gene fraction** (0.3%): rough, from 35 of 11,500 genes.
23. **Cross-donor correlation** beyond the covariates (relatedness, batch): not
    modelled; untested on real data.
24. **Design counts**: R_null, R_power, K_perm, the oracle and technical Monte
    Carlo counts, `nperm`, the gene sample and `null_fraction` (section 5.5;
    the user's choices are in sections 9.9 and 9.15), recorded in
    `provenance.json`.
25. **Acceptance tolerances marked "proposed"** in section 7.
26. **Reference-bias `phi_ref`**: the pooled reference fraction at phASER
    heterozygous sites in BrainVar, and the production gate's verdict on the
    real cohort (the `reference_bias` block of the deployment's
    `eval_bundle.json`), which set the realistic upper end of the sweep.
27. **Donor multipliers `m_i` (per CPM tercile) and `m_a_i`**: calibrated by
    inversion at build step 6 (section 3.4).
28. **Emulator cost per replicate**: timed at build step 4 (section 5.5).

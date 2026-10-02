# tensorQTL, hapmixQTL branch

A fork of tensorQTL adding **hapmixQTL**: cis-eQTL mapping from haplotype-resolved
expression (Salmon against a personalized diploid transcriptome), with the
quantifier's Gibbs variance carried into the allelic channel's weights.
hapmixQTL is an **extension of mixQTL**. This file is an index: the documents
it points to hold the detail, and dated measurements live in
`docs/measurement_record.md`, not here.

## The two modes. There are only two.

Everything else was deprecated and quarantined on 2026-09-23 (user decision).
Do not add a third, do not reintroduce a removed one, and do not report a
comparison against one.

- **mixQTL mode** (`tensorqtl/mixqtl_replication.py`): a NumPy port of
  `hakyimlab/mixqtl` @ `624ae44` with the eleven divergences of the 2026-09-14
  review removed. It consumes Salmon point estimates and never the Gibbs draws,
  so it is the no-draws comparator hapmixQTL has to beat. Published cutoffs
  `100/50/10/1000`, `weight_cap` 10, natural-log response by design. Driver
  `scripts/compare_mixqtl_replication.py`.
- **Default mode** (`prepare_default_inputs`, then `map_nominal` / `map_cis`
  with `tau_mode='zero'`, `se_mode='fitted'`): allelic contrast
  `log2((pL+.5)/(pR+.5))` weighted by `1/Va`, Va the Gibbs variance used as a
  SHAPE, so `Var(eps_i) = sigma^2 v_i` with `sigma^2` fitted per variant; total
  `log2((pT+.5)/(effective_library_size+1)*1e6)` with unit working variance
  (half-read, adopted 2026-09-29). Per-channel t references, a
  Welch-Satterthwaite reference for the combination, Meier's correction of the
  combined SE, a 15-donor allelic admission floor, the `records_signflip`
  permutation null. The Salmon driver `scripts/run_hapmixqtl_from_salmon.py`
  fixes this configuration. Definition: `docs/hapmixqtl_methods.md`.

Quarantined (code `tensorqtl/fitted_variance.py`, tests `tests/fitted_variance/`,
records `brainvar_hapmix_deploy/deprecated_models/README.md`):
`variance_model`, `variance_prior`, `tau_mode='estimate'`, the known-variance
SE `se_mode='model'`. Two structural reasons, neither empirical, so efficiency
does not reopen them: circularity (each fits a record's variance from the
gene's own residuals and then weights those residuals by it) and, for the
free-`c` models, invariance to the draws' absolute scale. A depth-independent
variance term may be explored as measurement only, under the conditions of
`brainvar_hapmix_deploy/nominal_p_hypotheses_20260925/` (record in
`docs/measurement_record.md`); nothing ships without a further decision.

Not supported in default mode (2026-10-01): fine-mapping (`map_susie`; the CLI
mode was removed) and the STR-curvature / multi-allelic second pass. Both have
no per-channel residual scale and refuse `tau_mode='zero'`.

Say **"Gibbs variance"** and **"Gibbs draws"**, never "bootstrap" (the cohort
has 200 Gibbs draws and no `--numBootstraps`).

## Which document answers which question

| Question | Document |
|---|---|
| What is the statistic, exactly, and how do I reproduce it? | `docs/hapmixqtl_methods.md` |
| What do the output columns mean (including `loo_donor`, `loo_pval_nominal`)? | `docs/outputs.md` |
| How do I run the BrainVar deployment end to end? | `docs/brainvar_deploy_runbook.md` |
| What is implemented, validated, proposed, on hold, open, running? | `docs/CURRENT_SCIENTIFIC_STATE.md` |
| What rules govern values, units, gene filter and permutation, and what was decided when? | `docs/pipeline_rules.md` |
| What is tested, and how do I run the self-test? | `tests/README.md` |
| How are benchmark datasets with known effects made and scored? | `benchmark/simulated_effects/README.md` (run order `run_all.sh`; `SIMULATED_EFFECTS_ROOT` for a fresh output root). The run on the shipped configuration (every arm on the half-read total, 2026-10-01) is `brainvar_hapmix_deploy/simulated_effects_half_read_20261001/` (page `report/plasmode_report.html`, stored null `stored_null_half_read_20261001/`), the low-coverage set's `simulated_effects_lowcov_half_read_20261001/` (2026-10-02, no stored null); the 2026-09-27 pages `plasmode_meier_20260927/` and `plasmode_lowcov_meier_20260927/` are records of the log2(CPM+1) total |
| The whole benchmark on one page; hapmixQTL against TReCASE? | `brainvar_hapmix_deploy/benchmark_summary_20260929/summary.html`, `hapmix_vs_trecase.html` beside it |
| Whose top genes replicate in held-out BrainVar donors? | `brainvar_hapmix_deploy/referee_replication_20260928/report.html` |
| How does default mode do on data drawn from TReCASE's own model? | `brainvar_hapmix_deploy/external_benchmark_current_20260928/report.html` |
| Why do effect sizes in the simulated-effects benchmark fall short of the simulated effect? | `brainvar_hapmix_deploy/beta_shortfall_20260929/beta_shortfall.html` |
| Why half-read split ships (trials, comparisons, adoption)? | `brainvar_hapmix_deploy/half_read_trial_20260929/CONCLUSION.md`, `half_read_unit_power_pr_20260929/index.html`, `half_read_default_adoption_20260929/`; the analysis driver `docs/half_read_analysis.md` |
| What Gibbs prior do Salmon's draws carry, how calibrated is `Va`, and what is `--gibbsPriorAggregation`? | `brainvar_hapmix_deploy/salmon_informative_reads_20260930/README.md`; prepared (on hold) re-quantification `salmon_gibbspriorgroups_20261001/`; the C++ 1.10.3 fork `/mnt/ssd/lalli/usr/local/src/salmon-gibbs-prior` and the Rust 2.8.0 clone `salmon-rust-gibbs-prior` beside it (change uncommitted in both as of 2026-10-01; commit and push commands for the fork `github.com/JosephLalli/salmon` in `brainvar_hapmix_deploy/release_closure_20261001/README.md`); the patches `/mnt/ssd/lalli/usr/local/src/salmon-gibbs-prior-aggregation.patch` (C++) and `salmon-rust-gibbs-prior-aggregation.patch` (Rust), and the Rust pull-request drafts `salmon-rust-gibbs-prior-aggregation.{PR.md,fork-PR.md}` (upstream `develop`; the fork's `master`); the `*-groups.*` files beside them are the earlier file-based form |
| How are the alignment-based (native) counts built? | `brainvar_hapmix_deploy/phaser_stranded_20260928/README.md`, `wasp_20260928/README.md` |
| What is the RASQUAL comparison, and what can it settle? | `brainvar_hapmix_deploy/rasqual_comparison_design_20260923/rasqual_comparison.html`, `rasqual_read_level_20260927/report.html` |
| What was deprecated on 2026-09-23 and why? | `brainvar_hapmix_deploy/deprecated_models/README.md` |
| What was retired on 2026-10-01? | `brainvar_hapmix_deploy/retired_scripts_20261001/README.md` (`compare_pipelines.py`; its Gibbs-cache writer is now `scripts/build_gibbs_cache.py`) |
| What verified the 2026-10-01 closure and the benchmark split and rename? | `brainvar_hapmix_deploy/release_closure_20261001/README.md` (input-contract probe, leave-one-donor-out checks, phase check, untested-genes check, the benchmark's before/after comparison, the rename checks, and the `--gibbsPriorAggregation` checks of both Salmon forks) |
| What was measured from 2026-09-13 to 2026-09-25, and which claims were withdrawn? | `docs/measurement_record.md`; the validation record `docs/ase_validation.md` |
| Superseded designs and handoffs | `docs/simulation_benchmark_spec.md`, `docs/LOCAL_HANDOFF.md`, `docs/IMPLEMENTATION_STATUS_20260916.md`, `docs/OPEN_INVESTIGATIONS_20260920.md`, `docs/COVARIATE_VARIANCE_SCREEN_20260925.md` (each marked as a record at its top) |

## Pipeline rules (user decisions, standing)

Every value comes from Salmon point estimates; default mode uses the Gibbs draws
only for the allelic measurement variance. One unit throughout: log2, the
half-read log CPM for total expression and for the expression PCs (on edgeR
effective library sizes). Expression-PC gene filter = eQTL gene filter.
Genotype PCs stay with the genotypes under permutation; every other covariate
moves with the RNA record. mixQTL never touches the draws. Statement, code map
and dated decisions: `docs/pipeline_rules.md`.

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
- Simulated effects are beta = 0.2 / 0.4 / 0.8; never "planted".
- Reports for the user are HTML pages with figures, not long markdown.
- Say "Gibbs variance", never "bootstrap".

## Traps (each one has cost time; detail in the linked document)

- **Units.** log2 everywhere in default mode (beta = 1 is a twofold effect);
  mixQTL mode's response is natural log by design; `compute_summaries_from_gibbs`
  and dated scripts built on it are natural log and historical.
- **The detection call is the empirical permutation p** (`pval_perm`,
  `pval_beta`), never a lead's `pval_nominal`, which is the best of the window.
  The lead is reported on the scan's fitted scale; no refit occurs in default
  mode.
- **Allelic channel is through the origin** (no automatic intercept since
  2026-09-15); the total channel keeps its intercept.
- **Inputs are read positionally after validation.** `_validate_inputs`
  rejects misaligned, non-finite or negative inputs and one-sided phase;
  covariates must be full rank with the intercept. An excluded donor-gene pair
  is `Va = 0` with a finite `A`, never NaN.
- **`tau_a`, `tau_t`, `c_a` are `None` in default mode**, correctly: no such
  parameters exist in the model. Compare with `.equals()`, not `==`.
- **57.8% of donor-gene pairs carry no allelic information** because the
  Salmon index deduplicated homozygous transcript copies and the ingest pairs
  only transcripts with both haplotype rows; the total channel keeps them.
  Total expression must sum every transcript (`pT`), never `pL + pR`.
  Detail: `docs/measurement_record.md`, "Facts that are easy to get wrong".
- **The permutation null permutes donor records and swaps haplotype labels**
  (`records_signflip`); `records` is the FastQTL/tensorQTL-equivalent null;
  `residuals` is the earlier Freedman-Lane scheme, conservative where weights
  vary. Detail: `docs/hapmixqtl_methods.md`.
- **A single donor record can carry a gene-level call** (CALM2, 2026-09-25).
  `loo_donor` / `loo_pval_nominal` make it visible at the fixed lead; they do
  not recompute `pval_perm`.
- **RASQUAL's output field 15 is theta, a precision (10000 = no
  overdispersion), not a fraction.** Two pre-2026-09-24 runs carry it under the
  wrong name `rho`. The field list lives in `benchmark/simulated_effects/common.py`.
- **An upstream `tensorqtl` is installed in the linuxbrew Python**; outside the
  repository `import tensorqtl` gets it, not this fork. Scripts put the
  repository first on `sys.path`.
- **R's BLAS crash is an environment clash.** R segfaults in BLAS unless run
  as `R_LD_LIBRARY_PATH=/usr/lib/R/lib:/usr/lib/x86_64-linux-gnu
  LD_LIBRARY_PATH=/usr/local/cuda/lib64 Rscript ...`; asSeq (TReCASE) and
  MatrixEQTL are in `~/usr/local/lib/R/library`.
- **A stored result is reproduced with the inputs it was made with.** The
  benchmark's reproduction check failed from the day the expression PCs moved
  to the half-read build, because it compared against nulls made with the
  log2(CPM+1) build; it now reads that build explicitly.
- **The analysis VCF carries the statistical phase, not phASER's read-backed
  phase** (phASER ran without `--gw_phase_vcf`; `rephased.vcf.gz` is a
  misnomer). Verified on one donor: `release_closure_20261001/phase_check.log`.
- **Scratch is not owned.** The job `tmp/` vanished mid-run on 2026-10-01.
  Anything to be compared, cited or committed is written to an owned folder
  (a dated record folder, `~/usr/local/src`) from the start.

## Known and unfixed

- **Low-information allelic `Va` is shaped by Salmon's Gibbs prior** (stock
  draws, prior 1 per active transcript); kept as a caveat, re-quantification
  with `--gibbsPriorAggregation` prepared and on hold (user decision 2026-10-01).
- **chr14, chr15 and chr22 are excluded** (phased genotypes truncated; user
  decision 2026-09-28): `brainvar_hapmix_deploy/phased_vcf_inventory_20260928/README.md`.
  They are 1,188 of the 1,208 filtered genes the runner reports but cannot
  test for lack of Gibbs draws.
- **No transcriptome-wide run of the shipped default exists yet.** Both gene sets of the
  simulated-effects benchmark run on the shipped configuration (deep set 2026-10-01,
  low-coverage set 2026-10-02); the referee and the external benchmark were made on the
  predecessor configuration; `docs/CURRENT_SCIENTIFIC_STATE.md` says which result was measured on which.
- **The nominal p is anticonservative** through weight-residual coupling
  (mechanism identified on the pre-correction pipeline, source not); the
  detection call is unaffected. `docs/pipeline_rules.md`,
  `brainvar_hapmix_deploy/nominal_p_hypotheses_20260925/`.
- **The Beta approximation is conservative in the tail** (the opposite
  direction; do not net the two).
- **No per-donor allelic read floor by default** (mixQTL's published driver
  required at least 50 reads on each haplotype; `--asc-cutoff` supplies one
  for a matched comparison).
- **The benchmark's native-input arms and RASQUAL are not yet on the shipped configuration**
  (RASQUAL is out of the scored arms; both gene sets, the deep set's stored null and TReCASE are
  on it since 2026-10-01/02; the low-coverage set has no stored null); what would rerun each:
  `benchmark/simulated_effects/README.md`.
- **`tests/ase_gtex_real_data.py` fabricates the total channel's inferential
  variance** and keeps an allelic intercept (historical harness).
- **Cross-donor correlation** (relatedness, structure, batch beyond the 17
  covariates) is untested as a source of miscalibration on observed data.
- **`import tensorqtl` loads `hapmixqtl.py` twice**: as `tensorqtl.hapmixqtl`
  and, because upstream's `tensorqtl/tensorqtl.py:14-17` puts its own directory
  on `sys.path` and imports its siblings by bare name, as a separate module
  object `hapmixqtl` with its own globals. Upstream's design, left as is.

## Self-tests

```bash
# the method surface (249 tests, all passing 2026-10-01; the Meier known answer
# needs the GPU and /mnt/ssd/lalli/brainvar_hapmix_deploy, else it skips)
pytest tests/test_hapmixqtl.py tests/test_hapmixqtl_calibration.py \
       tests/test_hapmixqtl_perm_scheme.py tests/test_hapmixqtl_point_estimates.py \
       tests/test_hapmixqtl_allelic_df.py tests/test_hapmixqtl_meier.py \
       tests/test_half_read_default.py tests/test_half_read_runner.py \
       tests/test_fitted_variance_quarantine.py tests/fitted_variance/ \
       tests/test_cli.py tests/test_mixqtl_replication.py -q
python3 scripts/run_hapmixqtl_from_salmon.py --selftest
python3 scripts/str_integrate.py --selftest
```

The analysis-script tests (`tests/test_half_read_io.py`,
`tests/test_half_read_analysis_inputs.py`, `tests/test_half_read_trial.py`)
run separately. Four general test files carry 21 pre-existing failures (of 45
tests) from drift against the upstream modules they exercise (`test_post.py`,
`test_trans.py`, `test_genotypeio.py`, `test_integration.py`); none references
hapmixqtl. Detail: `tests/README.md`.

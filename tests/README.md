# Tests

This directory holds three groups of tests: the tests of hapmixQTL and mixQTL mode (the method's surface, run by the self-test below), tests of the dated half-read analysis scripts, and general tensorQTL tests that were added to this fork before hapmixQTL and exercise the upstream modules. Counts below are the number of tests `pytest --collect-only` reports for each file.

## The self-test

One command runs the method's surface:

```bash
pytest tests/test_hapmixqtl.py tests/test_hapmixqtl_calibration.py tests/test_hapmixqtl_perm_scheme.py \
       tests/test_hapmixqtl_point_estimates.py tests/test_hapmixqtl_allelic_df.py tests/test_hapmixqtl_meier.py \
       tests/test_half_read_default.py tests/test_half_read_runner.py tests/test_fitted_variance_quarantine.py \
       tests/fitted_variance/ tests/test_cli.py tests/test_mixqtl_replication.py -q
python3 scripts/run_hapmixqtl_from_salmon.py --selftest
```

The pytest command collects **249 tests**. All of them should pass; on a machine without a GPU, or without the BrainVar deployment directory, 4 of them skip (see [Tests that skip](#tests-that-skip)). `tests/test_hapmixqtl_calibration.py` takes about 50 seconds on a CPU. The second command runs the Salmon runner end to end on fabricated Salmon output and a small phased VCF, and ends with `SELF-TEST OK`; it calls `Rscript` with edgeR, so it needs R with that package installed. Run both from the repository root, so that this checkout's `tensorqtl` is imported rather than an installed upstream copy.

## The method's surface

| File | Tests | What it covers |
| --- | --- | --- |
| `test_hapmixqtl.py` | 91 | The core of `tensorqtl/hapmixqtl.py`: the weighted residualizer and weighted least squares through the square-root-weight transform, checked against reference fits; the per-variant association (a gene without phase or with very large `Va` reduces to the total channel, a simulated allelic fold change is recovered, the inverse-variance combination, the total channel's half dosage); `map_nominal` and `map_cis` end to end; per-channel covariates and the through-origin allelic channel; a switched-off allelic channel; the *cis*/*trans* diagnostic `pval_cis_trans`; the BED input reader; mixQTL-style count cutoff masks; the input contract (mismatched rows or columns, missing or negative values, phase frames, rank-deficient covariate designs); the leave-one-donor-out lead diagnostic (`loo_donor`); the fitted standard error. It also pins that `map_susie` and the known-variance paths refuse default mode, and keeps tests of the lead refit, which acts only under the deprecated `tau_mode='estimate'`. |
| `test_hapmixqtl_calibration.py` | 9 | Simulated-null gates. Eight run deprecated configurations (`tau_mode='estimate'` or the known-variance standard error): they keep the harness able to detect that `tau_mode='zero'` without a fitted scale is anticonservative, and check that the two channel estimators stay uncorrelated when their noise is correlated and that the combined standard error stays calibrated when donors' Gibbs variances differ by two orders of magnitude. One runs `map_cis` at its defaults and checks that `pval_perm` and `pval_beta` are calibrated on null genes with heteroskedastic weights. |
| `test_hapmixqtl_perm_scheme.py` | 9 | The permutation null. `records` equals a permutation of the genotype columns to floating-point precision; `residuals` (a Freedman-Lane scheme: leverage-standardized whitened residuals permuted at fixed weights) falls below nominal when residual size tracks the weights; `records_signflip`, the default, negates exactly the allelic numerator of a swapped record, centres the permuted allelic slope, and equals `records` when there is no phase. |
| `test_hapmixqtl_point_estimates.py` | 6 | The historical `summaries_from_point_estimates` inputs (`log2(CPM + 1)` totals): values come from the point estimates and the variance from the same transform of the draws. Also pins that genotype-tied covariates in `map_cis` leave the observed fit unchanged and change only the null. |
| `test_hapmixqtl_allelic_df.py` | 15 | Per-channel t references (`n_a - 1` degrees of freedom for the allelic channel, `n_t - 2 - n_cov` for the total), the Welch-Satterthwaite degrees of freedom of the combined statistic (matched to the first two moments of the combined variance estimate), and the 15-donor allelic admission floor, including that a gene below the floor gives exactly the total-channel result in `map_cis`'s scan and every permutation. |
| `test_hapmixqtl_meier.py` | 10 | Meier's correction of the combined standard error for channel weights estimated from the same residuals they combine. Four known-answer tests reproduce a stored exact-model run; six check that no correction applies when fewer than two channels carry weight or no residual scale is fitted. |
| `test_half_read_default.py` | 21 | `prepare_default_inputs`: the half-read transform keeps zero counts finite and adds half a read before normalization; the allelic contrast, Gibbs variance and admission boundaries (including exactly one haplotype below 0.5 reads); the draws change only the allelic weights; invalid values and shapes are refused; the BED reader supplies `Vt = 1` unless an explicit file is given; mapping matches the benchmark's manually built arm. |
| `test_half_read_runner.py` | 2 | The Salmon runner reports the transform, the unit total working variance and the allelic admission policy in its provenance; its count loader builds the haplotype counts from the fractional Gibbs draws of paired transcripts only, while the total sums every transcript. |
| `test_fitted_variance_quarantine.py` | 15 | The deprecated variance machinery in `tensorqtl/fitted_variance.py` stays quarantined: a default-mode run, including a full `map_cis` checked in a subprocess, never imports it; the deprecated names still resolve from `hapmixqtl` with a `DeprecationWarning`; the live code paths do not warn. |
| `fitted_variance/` | 34 | The deprecated configurations (`variance_model`, `variance_prior`, `tau_mode='estimate'`, the known-variance standard error), kept passing only so that results recorded with them can be reproduced. Every test names its deprecated configuration explicitly. Six files: `test_hapmixqtl_additive_golden.py` (4), `test_hapmixqtl_prior_threading.py` (4), `test_hapmixqtl_tau_estimator.py` (4), `test_hapmixqtl_trend_prior.py` (3), `test_hapmixqtl_variance_models.py` (14), `test_hapmixqtl_variance_prior.py` (5). |
| `test_cli.py` | 11 | The command line: help, argument validation and mode selection; that the hapmixQTL modes (`hapmixqtl_nominal`, `hapmixqtl`) offer only the default configuration (no deprecated variance options, no known-variance standard error, no fine-mapping mode), that `--help` documents their options, and that the total working variance defaults to one unless `--hap_Vt` is given. |
| `test_mixqtl_replication.py` | 26 | mixQTL mode (`tensorqtl/mixqtl_replication.py`). The vectorized closed forms are checked against `numpy.linalg.lstsq` fitted one variant at a time, and each cutoff, cap, degrees-of-freedom choice and fallback rule is pinned to the line of the R source it transcribes. No test executes the R implementation, so exact numerical agreement with a running mixQTL is not established. |

## Analysis-script tests

These test dated analysis scripts in `scripts/` rather than the method, and are run separately:

```bash
pytest tests/test_half_read_io.py tests/test_half_read_analysis_inputs.py tests/test_half_read_trial.py -q
```

| File | Tests | What it covers |
| --- | --- | --- |
| `test_half_read_io.py` | 5 | `scripts/half_read_io.py`: an interrupted write leaves the previous result in place, and a cache is reused only when its inputs, source code and outputs all match their recorded hashes. |
| `test_half_read_analysis_inputs.py` | 6 | Lead extraction for the half-read analysis: ties in p are broken by the statistic and then the variant, malformed, infinite and out-of-range p-values are refused, and the gene family and untestable pairs are recorded. |
| `test_half_read_trial.py` | 5 | The half-read transform of `scripts/half_read_trial.py`: going from zero to half a read raises the transformed value by exactly one at every library size, and invalid counts or library sizes are refused. |

The [half-read analysis guide](../docs/half_read_analysis.md) describes the driver these scripts belong to.

## General tensorQTL tests

These files test the upstream tensorQTL modules and the `nbqtl` module. They were added to this fork before hapmixQTL and are not part of the method's surface.

| File | Tests | What it covers |
| --- | --- | --- |
| `test_core.py` | 23 | Core utilities (`Residualizer`, statistical functions, data types) |
| `test_cis.py` | 13 | *cis*-QTL mapping |
| `test_edge_cases.py` | 16 | Edge cases, error handling and performance benchmarks |
| `test_nbqtl.py` | 12 | The negative-binomial score-test mode (`nbqtl-score`) |
| `test_genotypeio.py` | 17 | Genotype and phenotype input and output |
| `test_integration.py` | 9 | End-to-end *cis*, *trans* and post-processing workflows |
| `test_post.py` | 9 | Post-processing (q-values, false discovery rate) |
| `test_trans.py` | 10 | *trans*-QTL mapping |

A bare `pytest tests/` collects 374 tests: the 249 of the self-test, the 16 analysis-script tests, and these 109.

### Pre-existing failures

`test_post.py`, `test_trans.py`, `test_genotypeio.py` and `test_integration.py` carry 21 pre-existing failures among their 45 tests (counted 2026-10-01; earlier notes said 46) from drift between these tests and the upstream code they exercise (BED sort order, a missing `chr` column, a missing `pval_beta` column). None of the four files references hapmixQTL, and the upstream modules they test are unchanged from upstream tensorQTL, so their failures are not regressions of this fork's method. Run the self-test command to see the surface this fork owns.

### Test data and markers

The general tests read data under `tests/data/`, created by `prepare_test_data.py` from the GEUVADIS example data plus synthetic data:

```bash
cd tests
python prepare_test_data.py
```

`conftest.py` holds shared fixtures and `utils.py` helper functions. `pytest.ini` defines the markers `slow`, `integration`, `gpu_required` and `benchmark`, so `pytest tests/ -m "not slow and not integration"` deselects the slow and integration tests. The `benchmark` marker is applied in `test_edge_cases.py`.

## Tests that skip

Only the four known-answer tests of `test_hapmixqtl_meier.py` skip. Their fixture skips when CUDA is not available, when the stored run `combined_reference_exact_model_20260927` is not found under `/mnt/ssd/lalli/brainvar_hapmix_deploy`, or when `scripts/combined_reference_exact_model.py` or its inputs cannot be loaded; the reason is printed with `-rs`. Every other test in the self-test runs on a CPU, using the GPU when PyTorch sees one. `conftest.py` also defines a `device_param` fixture that skips its `cuda` case without a GPU; no current test uses it.

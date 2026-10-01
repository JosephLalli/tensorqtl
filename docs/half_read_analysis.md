# Reproducing the half-read analysis

The half-read analysis is the evidence on which the half-read split default was
adopted: the beta, reported-SE, nominal p-value, power, and precision–recall
comparisons of the half-read arm against the split-weighting predecessor, unit
weights, mixQTL mode, and total-only tensorQTL on the plasmode benchmark
datasets. The decision it supports was an accuracy/precision tradeoff on these
benchmarks; it was not a claim that precision improves uniformly. The current
input and estimator contract is [pipeline rules](pipeline_rules.md).

The recorded results were made on the plasmode datasets and arm results of
2026-09-27, with the expression PCs of the time, on `log2(CPM + 1)`; the
shipped default has since moved its expression PCs to the half-read unit. The
driver below regenerates that adoption record exactly; it is not a run of the
shipped default.

## One entry point

From this checkout, in the recorded Python environment:

```bash
python3 scripts/run_half_read_analysis.py \
  --deploy-root /path/to/brainvar_hapmix_deploy \
  --output /path/to/new_half_read_analysis
```

The driver reads the saved benchmark datasets, nominal results, half-read
caches, and trial outputs. It regenerates the tables and figures and compares
16 numerical tables exactly with the original saved tables. It never launches
a regression scan, Salmon run, or new simulation. The output directory must be
new. Individual files are written beside their destination under temporary
names and renamed on success; `run_manifest.json` is written last and marks
the complete run. A failed run retains its logs but has no completion manifest.

The input root must retain the recorded layout:

- `plasmode_meier_20260927/` and `plasmode_lowcov_meier_20260927/`:
  `datasets/`, `results/`, and `eigenmt_m_eff.tsv`.
- `corrected_null_store_20260925/` and
  `plasmode_stratum30_100_20260927/gene_set/`: gene design tables.
- `beta_shortfall_20260929/` and `beta_balance_trial_20260929/`:
  saved refits and comparison evidence used by the trial report.
- `half_read_trial_20260929/`: completed `deep/` and `low/` trial inputs.
- `half_read_se_comparison_20260929/`,
  `half_read_pvalue_comparison_20260929/`, and
  `half_read_unit_power_pr_20260929/`: recorded half-read caches and reference
  tables. The driver copies the small required caches into the new output root.
- `half_read_default_adoption_20260929/verification.json`: historical adoption
  receipt linked by the regenerated report.

No input files are modified. The output includes input and source hashes,
source snapshots, the Git revision and working patch, package versions,
`pip freeze`, per-stage logs, plot data, and exact-table comparison receipts.
Paths in these receipts describe the run location; they should not be edited
to pretend that a historical run used a different environment or method.

## Environment and run order

The driver checks Python **3.11.14** and the exact pins in
[`benchmark/plasmode/requirements.txt`](../benchmark/plasmode/requirements.txt),
and stops before writing anything if either differs. Its analysis stages use
the local source scripts and do not require R or a GPU. The pinned Torch
package is retained for consistency with the benchmark environment; none of
the stages the driver runs imports it, but the pin check still requires that
exact version to be installed. Run in an environment containing those
versions; the driver does not install packages. `runtime.json` and
`environment.txt` record what actually ran. The trial's `REPRODUCE.md`,
archived sources, and container receipt in `half_read_trial_20260929/` remain
the reference for repeating its GPU simulations; its mounted Python/R
libraries are external dependencies, not a self-contained container image.

| Order | Script | Main outputs |
|---|---|---|
| 1 | `half_read_score.py` | Paired beta/error data and causal summary |
| 2 | `half_read_report.py` | Trial report and repeated-sampling summary |
| 3 | `half_read_se_mse_audit.py` | Squared-error arithmetic and example calculations |
| 4 | `half_read_se_plot.py` | Four-method SE tables and figures by beta/coverage |
| 5 | `half_read_pvalue_plot.py` | Four-method mean −log10(p) tables and figures |
| 6 | `half_read_unit_power_pr.py` | Five-method SE/log-p/power figures, PR curves, HTML report |

The power and precision–recall stage uses `unit_power_inputs.py` to extract
saved baseline results. Its baseline cache and the p-value stage's cache are
reused only when every expected output, input hash, and relevant source hash
matches. Reuse is printed. Partial, unverifiable historical, or mismatched
caches stop with an error; use a new output directory to rebuild them.
Existing historical artifacts are retained.

For individual stages, set `HALF_READ_DEPLOY_ROOT` to the recorded input root
and `HALF_READ_OUTPUT_ROOT` to the root holding this run's half-read stage
directories before starting Python. With neither set, the input root defaults
to the checkout's `data/half_read`, and the cross-stage result root equals it.
Each stage's existing `--output`/`--root` still selects its own destination.

The optional scan-producing commands (`half_read_trial.py`,
`half_read_pvalue_cache.py`, `half_read_gene_cache.py`, and SE `--fill-missing`)
remain separate from this saved-input driver. They load inputs through the
plasmode benchmark's loader (`benchmark/plasmode/common.py`), which also
requires the original genotype, covariate, and Salmon-cache paths; setting the
analysis root alone does not relocate those dependencies. That loader now reads
the current covariate build, whose expression PCs are in the half-read unit,
so rerunning these commands would not reproduce the recorded scans, which used
the `log2(CPM + 1)` expression PCs.

## What the checks establish

- The transform tests use a hand-checkable invariant: adding half a read to
  zero doubles the numerator and increases the transformed value by one.
  Salmon expected counts may be fractional; finite, nonnegative counts remain
  valid.
- The production/manual-arm tests check mapping parity. They share the mapping
  kernel and do not independently establish calibration of SEs or p-values.
- The PR check uses explicit expected counts and tie behavior. The saved-table
  comparison checks that the regenerated numbers equal the recorded ones; it
  adds no scientific evidence of calibration or generalization.
- Half-read gene-lead extraction rejects malformed, infinite, or out-of-range
  nominal p-values. NaN pairs are counted and written to an exclusion table.
  Every fixed gene must have a valid lead; a gene without one stops extraction.
  Baseline untestable genes retain their existing no-call treatment and remain
  in the discovery denominator. Exact zero p-values retain the existing
  underflow policy.

Mean reported SE, empirical repeated-sampling SD, and RMSE are different
quantities. The MSE audit computes the arithmetic mean of
`(estimated_beta - signed_truth)**2` on the stated support. Coverage panels use
the recorded gene-level coverage definition, and discovery panels retain the
full prespecified gene families. Current plots do not redefine these quantities.
The tests of these scripts are listed under "Analysis-script tests" in
[tests/README.md](../tests/README.md).

# Reproducing the half-read analysis

The current input and estimator contract is [pipeline rules](pipeline_rules.md).
This guide covers the analysis scripts behind the beta, reported-SE, nominal
p-value, power, and precision–recall comparisons. The accepted decision was an
accuracy/precision tradeoff on these benchmarks; it was not a claim that
precision improves uniformly.

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
`pip freeze`, per-stage logs,
plot data, and exact-table comparison receipts. Paths in these receipts describe
the run location; they should not be edited to pretend that a historical run
used a different environment or method.

## Environment and run order

The driver checks Python **3.11.14** and the existing exact pins in
[`scripts/plasmode/requirements.txt`](../scripts/plasmode/requirements.txt).
Its analysis stages use the local source scripts and do not require R or a GPU.
The pinned Torch package is retained for consistency with the benchmark
environment. Run in an environment containing those versions; the driver does
not install packages. `runtime.json` and `environment.txt` record what actually
ran. The earlier trial's `REPRODUCE.md`, archived sources, and container receipt
remain the reference for repeating its GPU simulations; its mounted Python/R
libraries are external dependencies, not a self-contained container image.

| Order | Script | Main outputs |
|---|---|---|
| 1 | `half_read_score.py` | Paired beta/error data and causal summary |
| 2 | `half_read_report.py` | Trial report and repeated-sampling summary |
| 3 | `half_read_se_mse_audit.py` | Squared-error arithmetic and example calculations |
| 4 | `half_read_se_plot.py` | Four-method SE tables and figures by beta/coverage |
| 5 | `half_read_pvalue_plot.py` | Four-method mean −log10(p) tables and figures |
| 6 | `half_read_unit_power_pr.py` | Five-method SE/log-p/power figures, PR curves, HTML report |

Stage 6 uses `unit_power_inputs.py` to extract saved baseline results. Its
baseline cache and stage 5's cache are reused only when every expected output,
input hash, and relevant source hash matches. Reuse is printed. Partial,
unverifiable historical, or mismatched caches stop with an error; use a new
output directory to rebuild them. Existing historical artifacts are retained.

For individual stages, set `HALF_READ_DEPLOY_ROOT` to the recorded input root
and `HALF_READ_OUTPUT_ROOT` to the root holding this run's half-read stage
directories before starting Python. With neither set, the input root defaults
to the checkout's `data/half_read`, and the cross-stage result root equals it.
Each stage's existing `--output`/`--root` still selects its own destination.

The optional scan-producing commands (`half_read_trial.py`,
`half_read_pvalue_cache.py`, `half_read_gene_cache.py`, and SE `--fill-missing`)
remain separate from this saved-input driver. Their historical plasmode loader
also requires the original genotype, covariate, and Salmon-cache paths. Setting
the analysis root alone does not relocate those older dependencies.

## What the checks establish

- The transform tests use a hand-checkable invariant: adding half a read to
  zero doubles the numerator and increases the transformed value by one.
  Salmon expected counts may be fractional; finite, nonnegative counts remain
  valid.
- The production/manual-arm tests check mapping parity. They share the mapping
  kernel and do not independently establish calibration of SEs or p-values.
- The PR check uses explicit expected counts and tie behavior. The saved-table
  comparison checks that this engineering cleanup preserves the reported
  numbers; it adds no scientific evidence of calibration or generalization.
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

# tensorQTL, hapmixQTL branch

A fork of tensorQTL adding **hapmixQTL**: cis-eQTL mapping from haplotype-resolved
expression posteriors (Salmon Gibbs draws against a personalized diploid
transcriptome), with the quantifier's inferential uncertainty carried into the
standard error. Work lives on `claude/hapmixqtl-gibbs-uncertainty-IQ6Za`.

## Which document answers which question

The operating knowledge is large and split by purpose. Start in the right place:

| Question | Document |
|---|---|
| What is the statistic, exactly, and how do I reproduce it? | `docs/hapmixqtl_methods.md` — the specification: numbered equations, the algorithm, defaults, assumptions, and a symbol-to-code map |
| How do I run the BrainVar deployment end to end? | `docs/brainvar_deploy_runbook.md` — inputs, sample pairing, RASQUAL's invocation, gene selection, the comparison |
| What was measured, and how do I know the method is calibrated? | `docs/ase_validation.md` — the validation record, including withdrawn claims |
| What do the output columns mean? | `docs/outputs.md` |
| What does a new session need to pick this up? | `docs/LOCAL_HANDOFF.md` |
| What is implemented, proposed, validated, or currently running? | `docs/CURRENT_SCIENTIFIC_STATE.md` — concise current-state router and phase gate |

## Scientific phase transitions

The user's phase-transition rule supersedes the earlier policy to offer
`/reload`. Before a new substantive scientific phase, run a documentation agent
to validate and reconcile the actual local state into concise authoritative
pointers to the intellectual state, generated files, performed experiments and
results, decisions, existing code, and how to find them. After that agent
completes, resume already-authorized work.

State boundaries explicitly: distinguish implemented behavior, proposed work,
validated results, and current run state. This documentation pass records the
existing state only; it does not imply or start a new experiment.

## Facts that are easy to get wrong

- **Use log2 for expression, ASE ratios, aFC, and their uncertainty.** The user
  established this project convention on 2026-09-15. beta=1 means a twofold
  effect; tau_A/tau_T and Gibbs covariance use squared log2 units. Runtime
  conversion is pending; current outputs still use natural logs. See the
  [unit convention and conversion record](/mnt/ssd/lalli/brainvar_hapmix_deploy/mixqtl_algorithm_review_20260914/salmon_variance_theory_20260915/LOG2_CONVENTION.md).

- **ASE has no automatic intercept as of 2026-09-15.** This applies to its
  regression, null tau, and lead-refit tau designs. `ase_covariates_df=None`
  means through-origin with no nuisance columns; the total intercept remains.
  Results recorded before this change used the old design and are historical.
- **Default Salmon Gibbs includes counting noise.** The completed controlled
  experiment at `/mnt/ssd/lalli/brainvar_hapmix_deploy/salmon_gibbs_counting_sim_20260915/REPORT.md`
  contradicts the earlier assignment-only explanation below. The approved
  intercept correction does not change counting terms or select a new residual
  variance model; those decisions remain separate.

- **Gene-level Gibbs shape has bounded real-data evidence.** The completed
  three-library pilot found Gaussian competitive within 0.05 bits/draw for
  98.51% of 4,500 ASE and 99.93% of total summaries; it retained two
  reproducible ASE candidates and no total candidates. ZNF529 and RNF175 are
  exceptions to interpret, while RNF175 restart-block occupancy is unreliable.
  This is not association or tau calibration. See
  `/mnt/ssd/lalli/brainvar_hapmix_deploy/gibbs_shape_pilot_20260915/REPORT.md`.

- **The two-gene influence audit retained the moment/GPU baseline.** Across
  three genotype-selected common SNPs per gene and 92 donors (87 ASE), maximum
  one-block mean-plus-covariance shifts were 0.0736 working SE (ZNF529) and
  0.1396 (RNF175); leave-one-block maxima were 0.00901 and 0.01028. These are
  bounded sensitivity results, not association calibration. See
  `/mnt/ssd/lalli/brainvar_hapmix_deploy/gibbs_influence_audit_20260915/REPORT.md`.

- **Parents already estimate aggregate residual variance.** TensorQTL and
  mixQTL estimate a per-variant residual scale after fitting covariates and
  genotype; it combines measurement, biology, and other residual sources.
  They do not identify biological tau separately, and that identification is
  not required for association. Do not add M unchanged to that residual scale.
  The existing weighted leverage/M+tau extension is an alternative under
  reassessment, not the current accepted objective. The current task is a
  from-beginning recommendation for weighted OLS incorporating empirical
  Salmon Gibbs precision and per-variant residual variance. Exact preservation
  of unweighted OLS coefficients is not a constraint. The fixed-OLS candidate
  remains a historical alternative, not the primary direction. The recommended
  weighted architecture, its non-adoption boundary, and the deferred extension
  map are in `docs/IMPLEMENTATION_STATUS_20260916.md`.
  Recommendations are allowed; no new method is adopted or implemented. See
  `docs/IMPLEMENTATION_STATUS_20260916.md` for the preserved source/prototype
  boundary, first proposed OLS candidate, and deferred extension map; no model
  is selected by it.

- **`tau` is not something the Gibbs draws measure.** The per-sample weight is
  `1/(v_inf + tau)`. The current code uses Gibbs measurement variance plus an
  extra Poisson counting term (under review because default Gibbs already
  includes counting noise); `tau` is the residual variance estimated across samples. On BrainVar's
  well-expressed genes `v_inf` is about a third of the allelic channel's error and
  under 1% of the total channel's.
- **Two scales are reported.** `pval_perm` and `pval_beta` are gene-level and come
  from the scan; `slope`, `slope_se` and `pval_nominal` come from the lead refit when
  `tau_refit=True`. A lead's nominal p is never a gene-level p.
- **The detection call is the empirical p**, not the statistic. Statistic magnitude
  has been corrected three times; the called genes did not move.
- **Sample order is positional.** Phase frames are indexed by the genotype frame's
  column order. `_assert_phase_columns` now guards it; before that, reordered columns
  corrupted the allelic channel while the total channel stayed correct.
- **Library and driver defaults agree except on one flag.** `count_noise` is on
  and `ase_covariates_df=None` is through-origin everywhere since 2026-09-15; pass
  `count_noise=False` or `ase_covariates_df=SAME_COVARIATES` to reproduce
  earlier counting/covariate settings (this does not restore the former ASE
  intercept). `tau_refit` is the exception: on in both drivers, off in the
  library and CLI, because it carries a known unfixed selection bias (below) and
  a default should fail conservative. It should flip once the refit is charged
  the selection it performs.

## Claims withdrawn on 2026-09-13 — do not re-assert

An eight-angle review retired these. They may survive in older text.

- The additive, multiplicative and nested variance forms were **never** compared:
  no multiplicative arm exists in `tests/`.
- `docs/ase_validation.md` §7b's weight-cap and nested conclusions are void; both
  arms were arithmetically incapable of differing from their comparators.
- The additive `tau` is **not** "the one place hapmixQTL departs from mixQTL".
  mixQTL's total channel already carries a flat additive variance, so `v_t + tau_t`
  decomposes the parent rather than departing from it; the multiplicative scale is
  the allele-specific channel only.
- mixQTL's scan refits its dispersion at **every** variant, so hapmixQTL's per-gene
  null-model `tau` is the departure and `tau_refit` moves back toward the parent.

## Known and unfixed

- The lead refit is biased by selection: appending the window maximum to the `tau`
  design removes far more residual sum of squares than the one degree of freedom it
  is charged, so `tau` comes back 11-15% low under the null and the reported
  statistic is inflated. Gene-level p-values are unaffected.
- The Beta approximation is conservative in the tail, which costs power at
  transcriptome-scale thresholds.
- No per-sample allele-specific read floor, where mixQTL used 15 reads.

## Self-tests

Every script self-tests on fabricated data and prints `SELF-TEST OK`:

```bash
pytest tests/test_hapmixqtl.py tests/test_hapmixqtl_calibration.py tests/test_cli.py -q
RASQUAL_BIN=rasqual_src/src/rasqual python3 scripts/compare_pipelines.py --selftest
python3 scripts/run_hapmixqtl_from_salmon.py --selftest
```

The comparison driver's self-test covers the RASQUAL arm, the matched-variant effect
comparison, and that a rerun reuses its null-round checkpoints byte-identically.

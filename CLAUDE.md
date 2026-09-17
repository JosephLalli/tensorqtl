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
- **Default Salmon Gibbs includes counting noise, and `count_noise` is inert on
  well-expressed genes.** The controlled experiment at
  `/mnt/ssd/lalli/brainvar_hapmix_deploy/salmon_gibbs_counting_sim_20260915/REPORT.md`
  (Gibbs-only variance / MSE 0.944, with `q` 1.906) shows `q` double-counts
  the sampling noise of cells that have reads, and a census of the production
  draws shows the docstring's other case, reads with unanimous draws, never
  occurs (0 of 76,286 cells). On the 29 calibration genes removing `q` moves
  the lead statistic by a median 0.19 chi2 and changes no call. The flag is
  nevertheless load-bearing, for a reason neither side stated: the total
  channel has no degenerate-sample guard (`_zero_degenerate_ase_weights` exists
  only for ASE), so a zero-count total sample keeps weight `1/(1e-8 + tau_t)`,
  and in low-expressed genes where `tau_t` clamps to zero that is the
  2026-09-13 "52% type-I" weight-domination failure. `q` prevents it only
  because `1/(0 + 2*kappa) = 1` acts as a floor. 39.95% of cells in 2,000
  random genes have no reads (none in the 29 calibration genes); a verifier
  measured that without `q` such samples hold a median 89.5% of the total
  channel's weight in the 281 of 1,079 sampled genes that have them, against
  17.4% with `q`. The fix is a coverage-based floor for zero-count total
  samples that is independent of the flag, then dropping `q` for samples with
  reads; excluding zero-count total samples is wrong (t = log kappa is real
  low-expression information, and exclusion conditions on the outcome). The
  default stays True until that is done; see
  `/mnt/ssd/lalli/brainvar_hapmix_deploy/estimator_ablation_20260916/REPORT.md`.

- **The DerSimonian-Laird `tau` understates the allelic residual scale.** With
  the final weights `1/(v + tau_DL)` the whitened residual mean square at the
  null design is 1.22 on the allelic channel (0.93-2.58 over the 29 genes) and
  1.04 on the total channel, so the known-variance SE is too small on the
  allelic channel by a gene-specific factor. A Paule-Mandel `tau` (iterate the
  moment update with the final weights until RSS_w = df) restores 1.00 by
  construction: reported statistics fall by a median 11% (CCNI 38 -> 10, APC
  25 -> 15, RPL15 22 -> 15), the null 95th percentile from 21.6 to 18.9, and
  the between-gene spread of null medians halves, after which a pooled
  threshold and the per-gene null call the same genes. This is measured, not
  adopted: `_estimate_tau` is still DL. The residual shape is also wrong: after
  PM whitening the standardized squared residual still rises with `log v`
  (pooled slope +0.21, s.e. 0.03), so a two-parameter `c*v + tau` form is the
  next candidate. Same report as above. PM is not the 2026-09-16 "fitted
  residual scale" rule: away from the `tau = 0` boundary the two coincide
  (SE_codex = s_v * SE_lib and s_v = 1 at the PM fixed point), but at the
  boundary the fitted-scale rule discards the Gibbs variance floor and halves
  the SE, and a verifier measured that boundary at 64.5% of allelic and 47.2%
  of total channels in 600 unfiltered cohort genes (41.5% / 13.5% without
  `q`; 0% for total at depth >= 50). Every experiment so far (29-gene null,
  30-gene pilot, shape pilot, influence audit) lives in the high-depth regime
  where the rules agree; the disagreement lives in the low-expression genes
  none of them included. Keep the absolute floor (known-variance SE, PM
  `tau`), and run a depth-stratified arm before any genome-wide scan.

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
  The 2026-09-16 "fitted residual scale" recommendation in
  `docs/IMPLEMENTATION_STATUS_20260916.md` is, within the existing framework,
  a self-consistent `tau` (Paule-Mandel) applied per variant; the measured
  consequence of the level part is in the bullet above. Do not reopen this
  from theory: run `estimator_ablation_20260916/ablation29.py` and measure.

- **`tau` is not something the Gibbs draws measure.** The per-sample weight is
  `1/(v_inf + tau)`. The current code uses Gibbs measurement variance plus the
  Poisson counting term `q` (see above); `tau` is the residual variance
  estimated across samples. On BrainVar's well-expressed genes `v_inf` is
  about a third of the allelic channel's error and under 1% of the total
  channel's.
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

Withdrawn on 2026-09-16 (measured in `estimator_ablation_20260916`):

- The stated reasons for `count_noise=True`: "Gibbs across-draw variance is
  read-assignment uncertainty only" (false: Salmon's default Gamma draw
  carries shot noise, CollapsedGibbsSampler.cpp:122) and "a sample with
  unambiguous reads has unanimous draws and would be discarded" (impossible
  under default flags: identical draws occur only with zero reads). The flag
  stays on for the reason above, which is a missing floor, not a variance
  term.
- `count_noise=False` dropping zero-read cells from the total channel: wrong,
  they stay in at weight `1/(1e-8 + tau_t)` (the guard is ASE-only).
- "The pooled 5% threshold calls 8 genes against 5 by the per-gene null" as a
  property of the data: it was the DL `tau` scale inflation of three genes.

## Known and unfixed

- The lead refit is biased by selection: appending the window maximum to the `tau`
  design removes far more residual sum of squares than the one degree of freedom it
  is charged, so `tau` comes back 11-15% low under the null and the reported
  statistic is inflated. Gene-level p-values are unaffected.
- The Beta approximation is conservative in the tail, which costs power at
  transcriptome-scale thresholds.
- No per-sample allele-specific read floor, where mixQTL used 15 reads.
- `_estimate_tau` is DerSimonian-Laird, whose allelic scale is 22% low at the
  median (above); Paule-Mandel is measured, not adopted, and the residual
  shape (`c*v + tau`) is untested.
- The total channel has no zero-count guard; `count_noise` is standing in for
  a floor (above). A coverage-based floor for zero-count total samples, then
  no `q` for samples with reads, is the fix, not the flag.
- Three input-validation defects found by the 2026-09-14 audit are unfixed
  and unarguable: a NaN at a zero-weight sample drives `pval_perm` to the
  `1/(nperm+1)` floor; variant-row identity between the genotype and phase
  frames is not checked beyond column order; the QR is taken on a
  rank-deficient design without a rank check.
- After the through-origin change, `_joint_gls`/`_pvals` (the robust
  second-pass path) still charge the ASE channel `1 + n_cov` columns, so the
  robust SE there is ~6% conservative; `map_cis`/`map_nominal` are unaffected.

## Estimator debates go through the ablation

Every estimator question since 2026-09-13 has been answerable in minutes on
the 29 calibration genes: `estimator_ablation_20260916/ablation29.py`
reproduces `null_calibration_29b` byte for byte and reruns the hapmixQTL arm
with the choice toggled, with 40 null draws. Before writing a theory document
about the variance model, add a configuration there and report the numbers.

## Self-tests

Every script self-tests on fabricated data and prints `SELF-TEST OK`:

```bash
pytest tests/test_hapmixqtl.py tests/test_hapmixqtl_calibration.py tests/test_cli.py -q
RASQUAL_BIN=rasqual_src/src/rasqual python3 scripts/compare_pipelines.py --selftest
python3 scripts/run_hapmixqtl_from_salmon.py --selftest
```

The comparison driver's self-test covers the RASQUAL arm, the matched-variant effect
comparison, and that a rerun reuses its null-round checkpoints byte-identically.

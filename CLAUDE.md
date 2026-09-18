# tensorQTL, hapmixQTL branch

A fork of tensorQTL adding **hapmixQTL**: cis-eQTL mapping from haplotype-resolved
expression posteriors (Salmon Gibbs draws against a personalized diploid
transcriptome), with the quantifier's inferential uncertainty carried into the
standard error. Work lives on the worktree branch `hapmix-runbook-local`,
pushed as `origin/claude/hapmixqtl-gibbs-uncertainty-IQ6Za`.

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

## Writing conventions (the user's)

- Define every named statistical method on first use, in terms of what it
  computes (DerSimonian-Laird, Paule-Mandel, Freedman-Lane, Kish, and so on).
- Never write "cell" for a table entry or a (donor, gene) datapoint; in this
  work "cell" means a biological cell. Say datapoint, donor-gene pair, or
  zero-read sample.
- Reports for the user are HTML pages with figures, not long markdown.

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
  the sampling noise of donors that have reads, and a census of the production
  draws shows the docstring's other case, reads with unanimous draws, never
  occurs (0 of 76,286 datapoints). On the 29 calibration genes removing `q` moves
  the lead statistic by a median 0.19 chi2 and changes no call. The flag is
  nevertheless load-bearing, for a reason neither side stated: the total
  channel has no degenerate-sample guard (`_zero_degenerate_ase_weights` exists
  only for ASE), so a zero-count total sample keeps weight `1/(1e-8 + tau_t)`,
  and in low-expressed genes where `tau_t` clamps to zero that is the
  2026-09-13 "52% type-I" weight-domination failure. `q` prevents it only
  because `1/(0 + 2*kappa) = 1` acts as a floor. 39.95% of donor-gene datapoints in 2,000
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
  (pooled slope +0.21, s.e. 0.03). Measured 2026-09-17 (report addendum): the
  two-parameter `c*v + tau` model (allelic c median 2.6, tau 0.006) fits the
  shape (slope +0.02), is calibrated (3.9% of true-null gene-draws below 0.05),
  puts the null on RASQUAL's scale (95th percentile 15.7 vs 14.6), and calls 9
  genes vs 6; the one gene among the 29 that is an eGene in the independent
  225-sample total-expression run (CCNI) is caught by it and by RASQUAL and
  missed by both additive configurations. The pure multiplicative model
  (weights 1/v with a fitted scale; mixQTL's form, but mixQTL caps the weight
  dynamic range at 10x and filters counts, which is a floor on v by another
  name) is anticonservative here (15.1% at nominal 5%) because its whitened
  residuals are not exchangeable under a floor and the allelic effective
  sample size collapses (Kish 70.6 -> 43.6); a fitted-scale permutation must
  also re-project permuted residuals onto the design complement, which the
  known-variance construction does not need. Verified independently: the SE
  rule is nearly inert for the gene-level permutation p, the weight rule is
  everything. Same report as above. PM is not the 2026-09-16 "fitted
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

- **Three variance models and an empirical-Bayes prior (2026-09-17).**
  `variance_model` on every mapping function and both CLIs: `additive`
  (default, `v + tau`; its nominal statistics and leads are byte-identical
  to before, max |d stat| 2.3e-6 on the 29-gene outputs; the permutation p
  changed by design when the record permutation became the default, below), `two_component` (`c_g v + tau_g`, fitted per
  gene by damped iterated weighted least squares of leverage-corrected squared
  residuals on [v, 1]) and `library_scaled` (`d_i (c_g v + tau_g)`, `d_i` from
  `estimate_library_factors`, required by and only accepted by that model).
  `variance_prior` from `estimate_variance_priors` replaces the zero clamp on
  (c_g, tau_g) with a log-normal prior per expression decile (means from the
  raw unclamped estimates, variances less the median sampling variance, kappa
  = Var(z^2) = 2.38) and fits the posterior mode in (log c, log tau) by
  damped Fisher scoring; rejected under `additive`. Measured on the 300-gene
  tiered calibration (`estimator_ablation_tiers_20260917/summary_prior.tsv`):
  type-I at 5% 0.040/0.036/0.030 with the prior against
  0.040/0.032/0.030 clamped (s.e. 0.003), no gene at either bound, the clamped fit's 43 plus EXOC6B, LOC124903631
  called, 91% of leads identical. The prior removes the clamp but NOT
  the low-expression non-identifiability: below ~70 allele-resolved reads the
  prior on c is 0.035 with log-sd 2.4 and the weights stay near-equal. A
  normal-scale prior floored 41-54% of genes again and was rejected. The
  `c_a_floored` output column reports a clamp hit on the clamped
  two-component path and is always False under the log-scale prior.
  The posterior-mode fit needs its backtracking line search with the merit
  consistent with the score (no 0.5), a direction-keeping step cap and the
  objective-gain stop rule, and TWO STARTS (prior mean and the clamped fit,
  higher objective kept): without them 12.7% of fits two-cycle, an
  inconsistent merit converges to a point that is not the mode, and under
  the two lowest bins' prior (median c 0.0018) a single start from the prior
  mean returns the spurious low-c mode for most identifiable genes
  (tests/test_hapmixqtl_variance_prior.py pins all of it; a golden test pins
  the additive path at 99921fd). The runner's library-scaled mode crashed on
  any cohort with a quantified gene lacking a position row (fixed; self-test
  covers it). Open: the bin prior estimator floors a negative raw mean in the
  two lowest bins and its moment match manufactures the second mode (the
  continuous prior below removes both, but is not the default); kappa
  is global; run_second_pass ignores the variance model; outputs.md/README
  lack the new columns and flags.
  Production form for BrainVar: `library_scaled` with the prior. Report:
  `estimator_ablation_20260916/REPORT.md`, the shrinkage and two-start
  sections.
  A continuous prior (`prior_method='trend'`, 2026-09-18: curves of log c and
  log tau on log10 reads fitted by local marginal likelihood over all genes,
  two passes with curve-based weights, two starts per window) removes the
  decile edges and the moment-match collapse: c agrees with the deciles
  within 22% from 68 reads up; tau in the top two deciles is
  0.018-0.019 where the deciles gave 0.0038-0.0067, and the predictive
  check settles it against the trend: 40-47% of raw tau estimates there are
  non-positive, the trend's log-normal predicts 7-13%, the deciles' 31%; the
  optimizer is at its optimum and a direct ML of the same log-normal on the
  same genes gives 0.006-0.007 (predicting 19-28%), so the lift to 0.018 is
  the 0.5/99.5% winsorization in _trend_prior choosing the narrow of two
  solutions, and no log-normal holds the near-zero bulk (a spike-and-slab on
  tau is the fix), while for c the trend is the better-behaved prior; mode switching 0.1% vs 1.9% (deciles); record-scheme
  type-I 0.057/0.052/0.054, calls 17/16/17 = 50 vs 16/16/18 = 50 (deciles), paired
  discordance 3+3 of 300. Default stays 'deciles'; per-gene
  c_raw_var/tau_raw_var and the pass-2 columns are in the prior frame for
  such checks. Report: the same file, last section.

- **The permutation null permutes donor records, not residuals (2026-09-17).**
  `perm_scheme='records'` (default): each donor's whitened phenotype value,
  weight and covariate row move together, genotypes stay, the denominator
  is recomputed per permutation (`_record_permutation_channel`); by
  relabeling this equals the genotype-permutation null of FastQTL and
  tensorQTL with per-donor weights, pinned to 1e-9 by
  tests/test_hapmixqtl_perm_scheme.py. `perm_scheme='residuals'` is the
  earlier Freedman-Lane scheme (leverage-standardized whitened residuals
  permuted at fixed weights). Why it changed: on BrainVar the residual
  scheme was conservative and most so at high expression (two-component
  type-I 0.040/0.036/0.030 by tier; shipped 0.068/0.045/0.022). Allelic
  channel: its null has the scale of the UNWEIGHTED mean of z^2, a real null
  has the w-weighted mean (pinned at 1 by the fit); R = weighted/unweighted
  predicts per-gene type-I (bins 0.006 -> 0.072), R = 1.00 where weights are
  equal (low tier) and 0.97 where they span two decades (high tier);
  allelic-only 0.032 -> 0.048 with records. Total channel: short in every
  tier (0.029-0.035) from the covariate handling (18 columns / 92 donors;
  no-leverage 0.23, phenotype-permute 0.00, records ~0.05-0.07). Cost 0.34 s
  vs 0.10 s per 1,000-permutation scan. Report:
  `estimator_ablation_20260916/REPORT.md`, two last sections.

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

Corrected facts that replaced them (these are true; do not negate them):

- mixQTL's total channel already carries a flat additive variance, so `v_t + tau_t`
  decomposes the parent rather than departing from it; the multiplicative scale is
  the allele-specific channel only.
- mixQTL's scan refits its dispersion at **every** variant, so hapmixQTL's per-gene
  null-model `tau` is the departure and `tau_refit` moves back toward the parent.
  mixQTL's allelic regression is also through the origin (`y ~ -1 + x`).

Withdrawn on 2026-09-16 (measured in `estimator_ablation_20260916`):

- The stated reasons for `count_noise=True`: "Gibbs across-draw variance is
  read-assignment uncertainty only" (false: Salmon's default Gamma draw
  carries shot noise, CollapsedGibbsSampler.cpp:122) and "a sample with
  unambiguous reads has unanimous draws and would be discarded" (impossible
  under default flags: identical draws occur only with zero reads). The flag
  stays on for the reason above, which is a missing floor, not a variance
  term.
- `count_noise=False` dropping zero-read samples from the total channel: wrong,
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
- Closed 2026-09-17: `test_combined_se_calibrated_under_heterogeneous_gibbs_variance`
  (tests/test_hapmixqtl_calibration.py) shows the combined SE stays calibrated
  when donors span two orders of magnitude in `v_inf` with a-t noise
  correlated at 0.9: null mean t^2 1.023 with correlation 0.9 and 1.023 with
  0 (20,000 genes each), corr(beta_a, beta_t) -0.009, type-I 0.051 / 0.011.
  Ignoring `Cat` is safe in BrainVar's regime; the 2% residual inflation is
  the tau plug-in, not the covariance. Two side facts: the null-design tau
  over-covers a planted effect (0.98) by construction, which is why the lead
  refit exists, and the refit SE is understated by 1-3% in variance
  (unresolved at 8,000 genes).

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

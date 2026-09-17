# Current scientific state: hapmixQTL

Reconciled 2026-09-16. A router to the current state: what is implemented,
what was measured, and what is open.

## State at a glance

- **Implemented, committed (bea450c, 2026-09-16):** ASE regression, null tau,
  and lead-refit tau use no automatic intercept; total expression retains its
  intercept.
  [ASE_IMPLEMENTATION.md](../../../../../brainvar_hapmix_deploy/mixqtl_algorithm_review_20260914/salmon_variance_theory_20260915/ASE_IMPLEMENTATION.md)
  records the scope and targeted validation. Measured on the 29 calibration
  genes (`/mnt/ssd/lalli/brainvar_hapmix_deploy/estimator_ablation_20260916/REPORT.md`):
  median |change| in the lead statistic 0.42, one borderline call (TCF4) added.
- **Current source behavior:** Gibbs summaries use natural logs and add an
  extra Poisson q term when `count_noise=True`; runtime migration to log2 is
  pending. It computes cross-channel Gibbs covariance `Cat` but the scan does
  not use it. See `tensorqtl/hapmixqtl.py:325` and the method map in
  `docs/hapmixqtl_methods.md`.
- **Validated, bounded evidence:** the closed controlled Salmon experiment
  supports counting default Gibbs uncertainty once for its singleton,
  fixed-depth ASE configuration: Gibbs-only predicted/observed variance was
  0.9441 (95% bootstrap interval 0.7888–1.1551); Gibbs plus q was 1.9059.
  It does not validate total-expression variance, eQTL p-values, interval
  calibration, residual tau, or multimodal posteriors. Gibbs-only nominal 95%
  normal-interval coverage was 91.5%, so average variance agreement is not
  interval calibration. Read
  `/mnt/ssd/lalli/brainvar_hapmix_deploy/salmon_gibbs_counting_sim_20260915/REPORT.md`.
- **Completed bounded shape audit:** three randomly selected libraries
  (566_R1→566_D1, 591_R1→593_D1, 618_R1→618_D1), 9,000 transcript and 4,500
  diagnostic-filtered gene distributions. Gaussian was competitive within
  0.05 bits/draw for 98.51% of gene ASE and 99.93% of total summaries. Three
  ASE candidates were screened; ZNF529 and RNF175 reproduced two fitted modes
  across both block groups, and none occurred for total expression. TRAPPC5
  shows transcript modes disappearing after aggregation. ZNF529 occupancy
  ranged 12–76%; RNF175's eight restart blocks were 44,0,0,0,0,72,36,0%, so
  reliable pooled mode probabilities remain unestablished, especially for
  RNF175. Read
  `/mnt/ssd/lalli/brainvar_hapmix_deploy/gibbs_shape_pilot_20260915/REPORT.md`.
  Fractional exported counts meant discrete NB/BB PMFs were not scored.

## Current decision order

**State on 2026-09-16, after measurement.** The three estimator questions
reopened on 09-14 to 09-16 were run on the 29 calibration genes
(`/mnt/ssd/lalli/brainvar_hapmix_deploy/estimator_ablation_20260916/REPORT.md`).
The ASE intercept and the counting term `q` are inert there (median |change| in
the lead statistic 0.42 and 0.19 chi2). The residual scale is not: with the
DerSimonian-Laird `tau` the whitened residual mean square on the allelic channel
is 1.22 (0.93-2.58 per gene), and a Paule-Mandel `tau` that restores 1.00
lowers the reported statistics by a median 11%, the null 95th percentile from
21.6 to 18.9 and halves the between-gene spread of the null. That is the
measured content of the "fitted residual scale" recommendation in
[IMPLEMENTATION_STATUS_20260916.md](IMPLEMENTATION_STATUS_20260916.md); the
code still uses DL. The residual shape remains wrong after PM (standardized
squared residual rises with `log v`, pooled slope +0.21), so `c*v + tau` is the
next candidate. Settled directions: log2 units, no automatic ASE intercept,
GPU matrix scan; `count_noise` stays True until zero-read cells in the total
channel get a coverage-based rule.

**Decision record.**

1. Settled by measurement (`estimator_ablation_20260916`): the allelic channel
   is through-origin (bea450c); the counting term `q` is inert for genes with
   reads and stays on only as a floor for zero-count total samples until a
   coverage-based floor replaces it; a fitted per-variant residual scale is the
   shipped known-variance estimator with a self-consistent `tau`, so it is not
   a new method; the cross-channel Gibbs covariance `Cat` stays out of the
   statistic (measured corr(beta_a, beta_t) within 0.03 of zero at noise
   correlation 0.9, live test); the moment/GPU representation of the draws is
   retained (shape pilot and two-gene influence audit, reports below).
2. Awaiting the user's decision: DerSimonian-Laird vs Paule-Mandel `tau`. The
   numbers are in the ablation report; nothing further needs measuring for the
   level. The residual shape (`c*v + tau`) needs one more ablation configuration.
3. Open and unmeasured: the `tau = 0` boundary and zero-read total cells, both
   invisible on well-expressed genes; a depth-stratified arm of the ablation
   from the existing cache is the experiment. Whether the Gibbs posterior
   covariance stands in for repeated-library measurement error cannot be
   settled on BrainVar (one library per donor).
4. `tau_A`/`tau_T` name the biological residual variance after the tested cis
   effect and covariates (the user's definition of the target); the estimator
   is and will remain an aggregate residual moment, since the design cannot
   separate biological from technical residual.

The Codex-period experiments remain the record for what they measured: the
[mixQTL algorithm review](/mnt/ssd/lalli/brainvar_hapmix_deploy/mixqtl_algorithm_review_20260914/REPORT.md),
the [counting simulation](/mnt/ssd/lalli/brainvar_hapmix_deploy/salmon_gibbs_counting_sim_20260915/REPORT.md),
the [Gibbs shape pilot](/mnt/ssd/lalli/brainvar_hapmix_deploy/gibbs_shape_pilot_20260915/REPORT.md),
the [two-gene influence audit](/mnt/ssd/lalli/brainvar_hapmix_deploy/gibbs_influence_audit_20260915/REPORT.md)
(a self-contained GLS, not the production mapper), and the equations of the
proposed extensions in [IMPLEMENTATION_STATUS_20260916.md](IMPLEMENTATION_STATUS_20260916.md).
The influence audit's restart-block occupancy instability (RNF175 blocks
44,0,0,0,0,72,36,0%) is carried forward as an unresolved fact about the draws.

## Routing and run state

Start with `docs/hapmixqtl_methods.md` for implementation and
`docs/ase_validation.md` for historical calibration claims, then use
`docs/brainvar_deploy_runbook.md` for the BrainVar comparison state. The
cross-project review at
`/mnt/ssd/lalli/brainvar_hapmix_deploy/mixqtl_algorithm_review_20260914/REPORT.md`
is the current methods/review record.

Worktree: `hapmix-runbook-local`; the through-origin change is commit
`bea450c` on top of `f11d586`. The estimator ablation
(`estimator_ablation_20260916`) is complete and reproducible from its scripts. Neither completed pilot nor audit implemented
final TMM normalization, production association mapping, or biological-residual
calibration. Applying phASER error correction before constructing quantification
references remains a future option, not an implemented workflow change.

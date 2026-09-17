# Current scientific state: hapmixQTL

Reconciled 2026-09-16. This is a router for the next scientific phase, not a
new analysis or an authorization to run one.

## State at a glance

- **Implemented, uncommitted:** ASE regression, null tau, and lead-refit tau
  now use no automatic intercept; total expression retains its intercept.
  `tensorqtl/hapmixqtl.py`, the targeted tests, and the CLI changes are dirty.
  [ASE_IMPLEMENTATION.md](../../../../../brainvar_hapmix_deploy/mixqtl_algorithm_review_20260914/salmon_variance_theory_20260915/ASE_IMPLEMENTATION.md)
  records the exact scope and targeted validation; broad calibration was not
  rerun.
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

**Current gate, 2026-09-16:** from-beginning weighted-OLS theoretical
recommendation only; no production edit, test, experiment, or new method
adoption is authorized. The current direction is empirical per-variant
residual variance with Gibbs-informed precision, without treating SEs as known
absolute variances; exact preservation of unweighted OLS coefficients is not a
constraint. The existing
weighted leverage-corrected tau moment, additive per-channel weights,
fixed-gene GPU scan, lead refit, half-count/filter/local-linear defaults, and
separate residual components remain source facts, not an accepted objective
for this reassessment. Settled directions remain log2, no automatic ASE
intercept, GPU support, no unconditional q for the tested configuration, and
minimal targeted testing. See the authoritative compact matrix in
[IMPLEMENTATION_STATUS_20260916.md](IMPLEMENTATION_STATUS_20260916.md), which
separates the completed full-M log2/no-q audit prototype and deferred
M+tau integration map from the current OLS-first question.

**Historical OLS-first alternative:** the earlier fixed-coefficient/RSS
candidate remains pending validation, not an adopted confirmatory default or a
constraint on the weighted recommendation. See its formula and calibration
boundaries in the status record.

**Recommended weighted architecture, not method adoption:** fix W within a
gene, fit WLS per variant, and estimate its weighted total residual scale from
that variant's residuals. Gibbs uncertainty informs precision; the residual
scale is not biological tau or an absolute known-variance SE. Weight-rule
selection, paired-channel covariance, and calibration remain open; see the
exact equations and required matching-null path in the status record.
No finding requires replacing the existing overdispersion framework; the
voom-style Gibbs trend remains a proposal, not a necessary change.

**Current specification question:** how should covariance enter haplotype-derived
phenotypes and joint ASE/total effects? WLS with an empirical per-variant
residual scale remains the working direction. Technical Gibbs covariance is not
automatically the total residual covariance.

**Covariance walkthrough outcome, not adoption:** fit ASE and total WLS slopes
with separate per-variant empirical scales, then combine only with a coherent
2×2 slope covariance. The prior donor-row joint GLS audit remains a separate
estimator; it is not generally identical to two-stage combined WLS.

The [next-step record](/mnt/ssd/lalli/brainvar_hapmix_deploy/mixqtl_algorithm_review_20260914/salmon_variance_theory_20260915/BIMODALITY_NEXT_STEPS.md)
records the completed existing-output audit and the subsequent interpretation.

The [completed interpretation and proposed decision table](/mnt/ssd/lalli/brainvar_hapmix_deploy/mixqtl_algorithm_review_20260914/salmon_variance_theory_20260915/PILOT_INTERPRETATION_AND_DECISIONS.md)
recommended the gene-level moment/GPU baseline. The completed two-gene audit
used 92 donors (87 ASE) and three genotype-selected common SNPs per gene; all
552 selected-locus VCI L/R calls matched the original GT. Primary maximum
single-block mean-plus-covariance shifts were 0.0736 working SE for ZNF529 and
0.1396 for RNF175; leave-one-block maxima were 0.00901 and 0.01028. With the
zero-ASE-proxy sensitivity, maximum block shifts were 0.0848 and 0.1188 SE.
The working moment/GPU baseline remains appropriate for this bounded result.
Read `/mnt/ssd/lalli/brainvar_hapmix_deploy/gibbs_influence_audit_20260915/REPORT.md`.

1. **Clarified parent baseline:** TensorQTL and mixQTL already estimate a
   per-variant aggregate residual scale after covariates/genotype. It includes
   measurement, biology, and other residual sources; it is not identified as
   biological tau. Accurate biological decomposition is not required for an
   association test, and M must not be added unchanged on top of that scale.
2. **Working direction and extension boundary:** ordinary parent residual-scale
   modeling remains the comparison baseline. The existing diagonal additive
   residual treatment is implemented; full-M wiring and per-variant extension
   are not integrated. M is an absolute measurement covariance when that
   extension is used, rather than a term added unchanged to the parent scale.
   Its channel moment is empirical excess RSS after the tested fit, not a
   demand to model every other residual cause. M removes known modeled
   uncertainty in expectation, not realized noise. In single-channel OLS with
   equal m, `tau_raw = RSS/df - m`; away from the zero boundary, `m+tau_raw`
   equals ordinary OLS residual variance and the fitted coefficients are OLS.
   For diagonal M, the correction is `sum_i (1-h_i)m_i / df`. Tau remains a
   biological target, while its empirical estimate can absorb technical or
   mean-model error. Biological purity is not required for association under
   an adequate residual-covariance model.
3. Retain the posterior-moment working measurement-error baseline unless
   stable shape adds consequential information. If it does, specify one
   GPU-compatible shape-sensitive statistic and matching null calibration.

**Clarification outcome:** parent OLS/WLS residual scales already model what
remains after each variant's fit, without separating biological variance from
measurement error. The residual-moment direction is not inherent to haplotype
testing or GPU scanning, and remains an alternative under reassessment; the
audit introduced no replacement model. The audit remains closed; no rerun, data
analysis, simulation, permutation, production code change, or workflow
integration is authorized.

The hard GPU matrix scan remains a requirement. tau_A and tau_T mean biological
residual variance after the tested cis effect and covariates, in squared log2
units; Salmon M remains separate and includes modeled counting and competition.
The counting fixture supports removing unconditional extra q for its tested
Salmon configuration; it does not validate every configuration. Beta fitting is
established machinery. Workflow integration is deferred and minimal tests are
the only completed validation for the uncommitted ASE-intercept change.

## Routing and run state

Start with `docs/hapmixqtl_methods.md` for implementation and
`docs/ase_validation.md` for historical calibration claims, then use
`docs/brainvar_deploy_runbook.md` for the BrainVar comparison state. The
cross-project review at
`/mnt/ssd/lalli/brainvar_hapmix_deploy/mixqtl_algorithm_review_20260914/REPORT.md`
is the current methods/review record.

Worktree: `hapmix-runbook-local`, base `f11d586`; it is dirty in the source,
tests, runbook, README, methods, and this documentation. No run was started or
resumed by this reconciliation. Neither completed pilot nor audit implemented
final TMM normalization, production association mapping, or biological-residual
calibration. Applying phASER error correction before constructing quantification
references remains a future option, not an implemented workflow change.

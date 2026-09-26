# Covariate variance screen: pre-measurement state (2026-09-25)

## Scope and status

This documentation checkpoint precedes the authorized exploratory screen of
RIN and available donor covariates against residual variability. No screen has
run, no association is claimed, and this record does not change weights,
default mode, or shipped behavior. Experimental methods and thresholds remain
for the parent to fix before execution. Planned outputs are
`/mnt/ssd/lalli/brainvar_hapmix_deploy/covariate_variance_screen_20260925/`.

## Observations already established

- The total-channel null investigation found a donor-level variance component:
  `221_D1` has RIN 3.1 and mean leverage-corrected whitened squared residual
  3.75, versus model-record maximum 1.93; five donors exceeded the model
  95th-percentile maximum. This was not converted to a covariate association.
- On the same 46-gene, 92-donor nominal-p instrument, total-channel residual
  variance follows a per-gene scale and approximately `Var(e) ~ v_t^0.66`.
  That is evidence about a records null, not about a causal RIN, age, sex, PC,
  or unmeasured-batch effect.
- The current estimator has only published posterior-mean-count mixQTL and
  default Gibbs-shape variance with fitted residual scale. Deprecated
  per-gene fitted variance-function modes remain excluded; the count-scale
  result did not authorize a weighting or default-mode change.

## Inputs and limitations

The screen can reuse the exact null-tool inputs:

- Gibbs cache: `/mnt/ssd/lalli/brainvar_hapmix_deploy/cache/gibbs_56b63c3b37ed5df8`
  (`genes.txt`, `samples.txt`, `YT.npy`; mapped by sample ID).
- Nominal instrument:
  `/mnt/ssd/lalli/brainvar_hapmix_deploy/nominal_p_null_instrument_20260925/inputs_at_lead.npz`
  and `null_long.tsv.gz`. The total-null tool recomputes total summaries from
  the cache and gates mapping at relative difference `<= 1e-9`.

The available covariate matrix has 17 columns: RIN, age, age squared, sex,
three genotype PCs, and ten expression PCs. No batch covariate is available.
Expression PCs are outcome-derived, so an association with residual variability
is conditioned descriptive evidence, not an independent causal exposure.

## Authorized question and required safeguards

The exploratory question is whether RIN or another available covariate is
associated with residual variability after accounting for gene scale. ASE and
total are separate outcomes; no pooled-channel conclusion is licensed. The
execution design must retain donor-level uncertainty, assess single-donor
sensitivity, and correct across the tested covariate/outcome family. Before
inspection, it must define gene-scale adjustment, donor clustering or
resampling, deletion rule, and multiple-testing family.

## Hypotheses, not findings

Low RIN may be associated with excess residual variability, motivated by
`221_D1`. It may instead be a donor-specific outlier, confounding by another
measured covariate, unmeasured batch or donor state, or chance. This supplied
data screen can identify associations; it cannot establish cause or repair
nominal-p calibration.

Relevant established context is in `CLAUDE.md` and
[CURRENT_SCIENTIFIC_STATE.md](CURRENT_SCIENTIFIC_STATE.md).

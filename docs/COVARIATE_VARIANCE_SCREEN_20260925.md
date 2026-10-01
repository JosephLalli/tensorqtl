# Covariate variance screen: completed bounded result (2026-09-25)

> **Status, 2026-10-01: completed, bounded exploratory screen; a dated
> record.** On 2026-09-25 this screen tested whether RIN or another available
> covariate is associated with residual variability after accounting for gene
> scale, separately in the allelic and total channels, and found no compelling
> association (numbers below); it changed no weights, no default and no shipped
> behavior, and authorizes none. It ran on the inputs of that date, before the
> half-read default (2026-09-29) and the half-read expression PCs (2026-09-30):
> the 17-column covariate matrix `brainvar_hapmix_deploy/cov/covariates.tsv`,
> the 2026-09-25 nominal-p instrument and the Gibbs cache listed below, so its
> numbers describe that pipeline and are not re-measured on the shipped one.
> Outputs, with the report `report.html` and `RESULTS.md`, are in
> `/mnt/ssd/lalli/brainvar_hapmix_deploy/covariate_variance_screen_20260925/`.
> The current method, and where the nominal-p mechanism and the per-donor
> variance component (donor `221_D1`) now stand, are in
> `docs/CURRENT_SCIENTIFIC_STATE.md` and `docs/pipeline_rules.md`.

## Observations already established

- The total-channel null investigation found a donor-level variance component:
  `221_D1` has RIN 3.1 and mean leverage-corrected whitened squared residual
  3.75, versus model-record maximum 1.93; five donors exceeded the model
  95th-percentile maximum. This was not converted to a covariate association.
- On the same 46-gene, 92-donor nominal-p instrument, total-channel residual
  variance follows a per-gene scale and approximately `Var(e) ~ v_t^0.66`.
  That is evidence about a records null, not about a causal RIN, age, sex, PC,
  or unmeasured-batch effect.
- The current estimator has published point-estimate mixQTL and the accepted
  half-read default: legacy ASE Gibbs variance/admission with fitted residual
  scales and unit total working variance. Deprecated
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

## Completed question and result

The screen tested whether RIN or another available covariate associated with
residual variability after accounting for gene scale, with ASE and total kept
separate. It used 20,281 genes, 92 donors, 17 covariates, 100,000 shared donor
permutations, and 100 fixed-design Gaussian replicates. RIN was not compelling:
ASE rho 0.108945 (p 0.300717; max-p 0.999960) and total rho 0.167760
(p 0.108349; max-p 0.959130). These descriptive results authorize no causal
claim or weighting/default change.

## Historical hypotheses

Low RIN may be associated with excess residual variability, motivated by
`221_D1`. It may instead be a donor-specific outlier, confounding by another
measured covariate, unmeasured batch or donor state, or chance. This supplied
data screen can identify associations; it cannot establish cause or repair
nominal-p calibration.

Relevant established context is in `CLAUDE.md` and
[CURRENT_SCIENTIFIC_STATE.md](CURRENT_SCIENTIFIC_STATE.md).

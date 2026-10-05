# Current implementation state

The public release integrates the default hapmixQTL implementation from the
simulation-benchmark development branch. This is a software release and
onboarding change, not a new estimator or a new calibration experiment.

- Default hapmixQTL combines an allelic channel and a half-read total channel,
  with fitted residual scales, Meier's correction and record/sign permutations.
- Published mixQTL replication is available as a separate point-estimate mode.
- Deprecated fitted-variance models are retained only for reproducibility.
- The portable Salmon preparation tool uses existing normalization and PC
  algorithms and accepts user-supplied numeric covariates.
- The worked example and public tests use fabricated data. Their success proves
  software execution, not unbiasedness, power or empirical FDR in a cohort.
- Fine-mapping and the STR/multi-allelic second pass are unsupported in default
  mode. Low-information quantifier variance, phase quality and mapping bias
  remain practical limitations.

Input contract: [pipeline_rules.md](pipeline_rules.md). Mathematical definition:
[hapmixqtl_methods.md](hapmixqtl_methods.md). Input preparation:
[hapmixqtl_inputs.md](hapmixqtl_inputs.md). Outputs: [outputs.md](outputs.md).

Cohort-specific experiments, manuscripts, operational logs and individual-level
records are excluded from this release tree. No new cohort analysis is authorized
by this documentation or by the release.

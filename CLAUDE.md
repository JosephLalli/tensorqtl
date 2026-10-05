# hapmixQTL development guide

Read README.md for the portable quick start, docs/CURRENT_SCIENTIFIC_STATE.md
for implementation boundaries, and docs/pipeline_rules.md for the input contract.

Two supported modes: default hapmixQTL and published mixQTL replication.
Default: Salmon point estimates define expression; Gibbs draws define allelic
variance shape; fitted per-channel residual scales, unweighted half-read total,
Meier combination correction, and records_signflip permutations.
Deprecated variance models stay quarantined in tensorqtl/fitted_variance.py.
Do not introduce a new scientific mode without an explicit decision.

Use fabricated fixtures for development. Never commit cohort manifests, sample
identifiers, personal workstation paths, count/covariate matrices, genotype
files, individual-level derived results, or local operational records.
The example/hapmixqtl generator is fully artificial. Public release changes
are validated by the synthetic checks in tests/README.md.

Before a scientific phase transition, reconcile current state and supporting
records. Historical research material is separate from this portable release.
When editing paired notebooks, edit the Jupytext source and sync the notebook.

### Output files
#### Mode `cis_nominal`
Column | Description
--- | ---
`phenotype_id` | Phenotype ID
`variant_id` | Variant ID
`start_distance` | Distance between the variant and phenotype start position (e.g., TSS)
`end_distance` | Distance between the variant and phenotype end position (only present if different from start position)
`af` | In-sample ALT allele frequency of the variant
`ma_samples` | Number of samples carrying at least on minor allele
`ma_count` | Number of minor alleles
`pval_nominal` | Nominal p-value of the association between the phenotype and variant
`slope` | Regression slope
`slope_se` | Standard error of the regression slope

#### Mode `cis_nominal`, with interaction term
When an interaction term is included, the output additionally contains the following columns instead of `pval_nominal`, `slope`, `slope_se`:
Column | Description
--- | ---
`pval_g` | Nominal p-value of the genotype term
`b_g` | Slope of the genotype term
`b_g_se` | Standard error of `b_g`
`pval_i` | Nominal p-value of the interaction variable
`b_i` | Slope of the interaction variable
`b_i_se` | Standard error of `b_i`
`pval_gi` | Nominal p-value of the interaction term
`b_gi` | Slope of the interaction term
`b_gi_se` | Standard error of `b_gi`
`tests_emt` | Effective number of independent variants (M<sub>eff</sub>) estimated by eigenMT
`pval_emt` | Bonferroni-adjusted `pval_gi` (i.e., multiplied by M<sub>eff</sub>)
`pval_adj_bh` | Benjamini-Hochberg adjusted `pval_emt`

#### Mode `cis`
Column | Description
--- | ---
`phenotype_id` | Phenotype ID
`num_var` | Number of variants in *cis*-window
`beta_shape1` | Parameter of the fitted Beta distribution
`beta_shape2` | Parameter of the fitted Beta distribution
`true_df` | Degrees of freedom used to compute p-values
`pval_true_df` | Nominal p-value based on `true_df`
`variant_id` | Variant ID
`start_distance` | Distance between the variant and phenotype start position (e.g., TSS)
`end_distance` | Distance between the variant and phenotype end position (only present if different from start position)
`ma_samples` | Number of samples carrying at least on minor allele
`ma_count` | Number of minor alleles
`af` | In-sample ALT allele frequency of the variant
`pval_nominal` | Nominal p-value of the association between the phenotype and variant
`slope` | Regression slope
`slope_se` | Standard error of the regression slope
`pval_perm` | Empirical p-value from permutations
`pval_beta` | Beta-approximated empirical p-value
`qval` | Storey q-value corresponding to `pval_beta`
`pval_nominal_threshold` | Nominal p-value threshold for significant associations with the phenotype

#### Mode `cis_independent`
The columns are the same as for `cis`, excluding `qval` and `pval_nominal_threshold`, and adding:
Column | Description
--- | ---
`rank` | Rank of the variant for the phenotype

#### Mode `trans`
Column | Description
--- | ---
`variant_id` | Variant ID
`phenotype_id` | Phenotype ID
`pval` | Nominal p-value of the association between the phenotype and variant
`b` | Regression slope
`b_se` | Standard error of the regression slope
`r2` | Squared residual genotype-phenotype correlation (only generated if `map_trans(..., return_r2=True)`)
`af` | In-sample ALT allele frequency of the variant

#### Mode `hapmixqtl_nominal`
One parquet per chromosome, `${prefix}.hapmixqtl_pairs.${chr}.parquet`.
Column | Description
--- | ---
`phenotype_id` | Phenotype ID
`variant_id` | Variant ID
`start_distance` | Distance between the variant and phenotype start position (e.g., TSS)
`end_distance` | Distance between the variant and phenotype end position (only present if different from start position)
`af` | In-sample ALT allele frequency of the variant
`ma_samples` | Number of samples carrying at least one minor allele
`ma_count` | Number of minor alleles
`pval_nominal` | Nominal p-value of the combined (ASE + total) association
`slope` | Combined effect size (log allelic fold change per ALT allele)
`slope_se` | Standard error of the combined effect size
`pval_a` | Nominal p-value of the allelic-contrast (ASE) channel
`slope_a` | Effect size from the ASE channel
`slope_a_se` | Standard error of `slope_a`
`pval_t` | Nominal p-value of the total-expression channel
`slope_t` | Effect size from the total-expression channel
`slope_t_se` | Standard error of `slope_t`
`pval_cis_trans` | Wald test that the two channels estimate the same effect (`hapmixqtl.cis_trans_diagnostic`). A small value flags a pair whose combined slope should not be read as a cis log allelic fold change; a diagnostic column, not a filter

All statistics are on the null-model tau scale (tau estimated once per phenotype per channel without a genotype term); `map_cis(tau_refit=True)` is the only entry point that reports a lead on a refit scale.

#### Mode `hapmixqtl`
Top association per phenotype with permutation and Beta-approximated p-values, written to `${prefix}.hapmixqtl.txt.gz`. The columns of `cis` are all present (plus `qval` and `pval_nominal_threshold` when rpy2/qvalue is available), where `slope`/`slope_se`/`pval_nominal` are the combined ASE + total estimates and `slope` is interpretable as the log allelic fold change per ALT allele. `beta_shape1`, `beta_shape2`, `true_df`, `pval_true_df` and `pval_beta` are NaN when `--disable_beta_approx` is set or the Beta fit fails. The following columns are additional to `cis`:

Column | Description
--- | ---
`slope_a` | Lead variant's effect size from the ASE channel
`slope_a_se` | Standard error of `slope_a`
`slope_t` | Lead variant's effect size from the total-expression channel
`slope_t_se` | Standard error of `slope_t`
`alpha_cis` | `slope_a / slope_t` at the lead
`pval_cis_trans` | Wald test that the two channels estimate the same effect at the lead (`hapmixqtl.cis_trans_diagnostic`); a diagnostic column, not a filter
`tau_a` | Overdispersion of the ASE channel used for the reported `slope`/`slope_se`/`pval_nominal`
`tau_t` | Overdispersion of the total channel used for the reported `slope`/`slope_se`/`pval_nominal`
`tau_a_null` | ASE-channel overdispersion of the scan, estimated under the null model (no genotype term)
`tau_t_null` | Total-channel overdispersion of the scan, estimated under the null model
`tau_refit` | Whether either channel's tau was re-estimated with the lead in the design (`--tau_refit`; false otherwise, in which case `tau_a`/`tau_t` equal `tau_a_null`/`tau_t_null`)

Two scales coexist in this table. `pval_perm` and `pval_beta` are always on the scan scale, where the permutations carry the same tau and the empirical p is calibrated; `pval_nominal`, `slope` and `slope_se` move to the refit scale when `tau_refit` is true. `pval_nominal` is the best of the cis-window and is never a gene-level p — `pval_beta` is.

#### Mode `hapmixqtl_susie`
SuSiE fine-mapping of the combined ASE + total signal. Two files are written: a credible-set summary parquet `${prefix}.hapmixqtl_SuSiE_summary.parquet` and a pickle `${prefix}.hapmixqtl_SuSiE.pickle` with the full per-phenotype SuSiE results (PIPs, credible sets, log Bayes factors, ELBO, convergence).
Summary columns:
Column | Description
--- | ---
`phenotype_id` | Phenotype ID
`variant_id` | Variant ID (member of the credible set)
`pip` | Posterior inclusion probability
`af` | In-sample ALT allele frequency of the variant
`cs_id` | Credible-set index (the SuSiE single-effect `L` this variant belongs to)
`tau_mode` | The `tau_mode` the fine-mapping was run under, recorded as provenance. Results produced under `zero` **with** `se_mode='model'` are invalid (`docs/ase_validation.md` §7g); `zero` is the default again since 2026-09-21 but paired with `se_mode='fitted'`, which does not carry that defect, so read the pairing and not the `tau_mode` alone; `hapmixqtl.fine_mapping_provenance()` classifies a file as `ok`, `stale` or `unknown`

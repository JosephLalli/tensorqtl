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
`pval_nominal` | Nominal p-value of the combined (ASE + total) association, referred to t with `dof_nominal` df. NaN where neither channel carries weight at the pair (e.g. a constant-dosage variant), because nothing was tested there
`slope` | Combined effect size (log allelic fold change per ALT allele)
`slope_se` | Standard error of the combined effect size
`pval_a` | Nominal p-value of the allelic-contrast (ASE) channel, referred to t with `dof_a` df. Still reported for a gene below the allelic admission floor, whose combined statistic does not use it
`slope_a` | Effect size from the ASE channel
`slope_a_se` | Standard error of `slope_a`
`pval_t` | Nominal p-value of the total-expression channel, referred to t with `dof_t` df
`slope_t` | Effect size from the total-expression channel
`slope_t_se` | Standard error of `slope_t`
`pval_cis_trans` | Wald test that the two channels estimate the same effect (`hapmixqtl.cis_trans_diagnostic`), referred in default mode to the Welch-Satterthwaite df of the difference, `(se_a^2 + se_t^2)^2 / (se_a^4/dof_a + se_t^4/dof_t)`. A small value flags a pair whose combined slope should not be read as a cis log allelic fold change; a diagnostic column, not a filter
`dof_nominal` | Added 2026-09-27 (commit 8a06803). The t reference of `pval_nominal`: in default mode the Welch-Satterthwaite df of the inverse-variance combination, `(w_a + w_t)^2 / (w_a^2/dof_a + w_t^2/dof_t)` with `w = 1/se^2`, per pair; `dof_a` where the total channel carries no weight, `dof_t` where the allelic channel carries none (including every pair of a gene below the admission floor), NaN where neither does
`dof_a` | Added 2026-09-27. Residual df of the allelic channel's fitted scale, `n_a - 1 - n_cov_a` over the gene's informative allelic donors (`n_a - 1` for the default through-origin design); NaN when the channel is switched off
`dof_t` | Added 2026-09-27. Residual df of the total channel's fitted scale, `n_t - 2 - n_cov` over the gene's informative total-channel donors; NaN when the channel is switched off
`allelic_admitted` | Added 2026-09-27. Whether the allelic channel enters the combined statistic: the gene has at least `MIN_ALLELIC_DONORS` = 15 informative allelic donors (mixQTL's own cutoff). When False, `slope`, `slope_se` and `pval_nominal` ARE the total channel's. Waived when the gene has no total channel (an allelic-only run, `keep_t_df` all False), where the allelic channel is the statistic whenever it is on

**t references (default mode, since 2026-09-27).** Each p-value is referred to the degrees of freedom of the scale its standard error was fitted with; an off channel's `pval_a` or `pval_t` is NaN, not the 1.0 of a zero statistic. Before commit 8a06803 all three were referred to one shared `N - 2 - max(n_cov, n_cov_a)` (73 on BrainVar), so tables written before 2026-09-27 have no dof columns and their `pval_nominal`/`pval_a`/`pval_t` are on that shared reference. The known-variance and HC1 standard errors (deprecated `se_mode='model'`, `robust=True`) keep the shared reference and no floor; the dof columns then carry it. Derivation and the reference's measured cost: `docs/hapmixqtl_methods.md` Section 4.5.

In default mode (`tau_mode='zero'`, `se_mode='fitted'`) no tau exists: every statistic uses each channel's residual scale fitted per variant. Under the deprecated `tau_mode='estimate'` the statistics are on the null-model tau scale (tau estimated once per phenotype per channel without a genotype term), and `map_cis(tau_refit=True)` is the only entry point that reports a lead on a refit scale.

**Units.** From `run_hapmixqtl_from_salmon.py` since 2026-09-25, `slope`/`slope_a`/`slope_t` are log2 effect sizes (beta=1 is a twofold effect): the phenotype is `summaries_from_point_estimates` (`docs/pipeline_rules.md`). Tables written before 2026-09-25, and by `compare_pipelines.py` until it is switched, are natural-log effect sizes from `compute_summaries_from_gibbs`.

#### Mode `hapmixqtl`
Top association per phenotype with permutation and Beta-approximated p-values, written to `${prefix}.hapmixqtl.txt.gz`. The columns of `cis` are all present (plus `qval` and `pval_nominal_threshold` when rpy2/qvalue is available), where `slope`/`slope_se`/`pval_nominal` are the combined ASE + total estimates and `slope` is interpretable as the log allelic fold change per ALT allele. `beta_shape1`, `beta_shape2`, `true_df`, `pval_true_df` and `pval_beta` are NaN when `--disable_beta_approx` is set or the Beta fit fails. The following columns are additional to `cis`:

Column | Description
--- | ---
`slope_a` | Lead variant's effect size from the ASE channel
`slope_a_se` | Standard error of `slope_a`
`slope_t` | Lead variant's effect size from the total-expression channel
`slope_t_se` | Standard error of `slope_t`
`alpha_cis` | `slope_a / slope_t` at the lead
`pval_cis_trans` | Wald test that the two channels estimate the same effect at the lead (`hapmixqtl.cis_trans_diagnostic`), referred in default mode to the Welch-Satterthwaite df of the difference, `(se_a^2 + se_t^2)^2 / (se_a^4/dof_a + se_t^4/dof_t)`; a diagnostic column, not a filter
`tau_a` | Overdispersion of the ASE channel used for the reported `slope`/`slope_se`/`pval_nominal`
`tau_t` | Overdispersion of the total channel used for the reported `slope`/`slope_se`/`pval_nominal`
`tau_a_null` | ASE-channel overdispersion of the scan, estimated under the null model (no genotype term)
`tau_t_null` | Total-channel overdispersion of the scan, estimated under the null model
`tau_refit` | Whether either channel's tau was re-estimated with the lead in the design (`--tau_refit`; false otherwise, in which case `tau_a`/`tau_t` equal `tau_a_null`/`tau_t_null`)
`perm_scheme` | Which permutation null produced `pval_perm`/`pval_beta` (`records_signflip`, the default since 2026-09-25; `records`; or `residuals`) — see CLAUDE.md's permutation bullet
`n_genotype_covariates` | Added 2026-09-25 (user rule). Count of covariate columns tied to the genotypes (genotype PCs) rather than the RNA record, which stay fixed in genotype order under the permutation null (`map_cis(genotype_covariates_df=...)`); 0 when none were passed, as in every run before 2026-09-25
`dof_nominal` | Added 2026-09-27 (commit 8a06803). The t reference of the lead's `pval_nominal`: in default mode the Welch-Satterthwaite df of the lead's combination, exactly as `map_nominal` refers that pair (see its `dof_nominal`); under `se_mode='model'` the shared `N - 2 - max(n_cov, n_cov_a)`. NaN for a gene in which neither channel carries weight, which was not tested: its `pval_nominal`, `pval_perm` and `pval_beta` are NaN too and no Beta fit is made
`allelic_admitted` | Added 2026-09-27. Whether the allelic channel entered the scanned statistic (at least 15 informative allelic donors, waived without a total channel). The floor is applied identically to the observed scan, every permutation and the lead, so `pval_perm` and `pval_beta` are built from the statistic that was observed. When False the lead's `slope`, `slope_se` and `pval_nominal` are the total channel's

Two scales coexist in this table. `pval_perm` and `pval_beta` are always on the scan scale, where the permutations carry the same statistic and the empirical p is calibrated; under the deprecated `tau_mode='estimate'`, `pval_nominal`, `slope` and `slope_se` move to the refit scale when `tau_refit` is true (in default mode there is no tau and the refit never runs). Since 2026-09-27 the lead's `slope_a_se`, `slope_t_se`, `alpha_cis` and `pval_cis_trans` are on the scan's fitted scale in default mode, as in `map_nominal`; before that they were computed with the known-variance standard error even in default mode. The scan still maps its statistic to a correlation with the constant `N - 2 - max(n_cov, n_cov_a)`, which seeds the Beta fit's `true_df`; that is one monotone map per gene, so the lead, `pval_perm` and `pval_beta` are exactly as the statistic determines them. The lead is the variant with the largest combined |t|; with a per-variant `dof_nominal` its `pval_nominal` need not be the gene's smallest `map_nominal` p. `pval_nominal` is the best of the cis-window and is never a gene-level p — `pval_beta` is.

`pval_nominal` (and `pval_a`, `pval_t`) is anticonservative under a donor-record permutation null, measured 2026-09-25 on 46 BrainVar genes at a fixed variant with 2,000 permutations: the combined statistic rejects at 0.068 / 0.0175 / 0.0028 at nominal 0.05 / 0.01 / 0.001, and the allelic channel transcriptome-wide (20,281 genes) at about 1.3x / 1.8x / 3.3x / 8.5x nominal at 0.05 / 0.01 / 0.001 / 1e-4. The cause is a per-gene mismatch between the reported variance and the realized one, set by how each gene's Gibbs weights pair with its residual sizes; the estimator itself is correct when `Var(eps) = sigma^2 v` holds. `pval_perm` and `pval_beta` are the detection calls and are unaffected by that scale error, but a call can rest on a single donor record (CLAUDE.md, "Known and unfixed"). See CLAUDE.md, "What the 2026-09-25 hypothesis round established". Those rates were measured on the pre-correction pipeline under the shared 73-df reference; the rates after the per-channel references of 2026-09-27, on the corrected pipeline's 100-gene null, are in `docs/pipeline_rules.md` ("After the per-channel t references").

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
`tau_mode` | Provenance of the fine-mapping run. In **default mode** this is `'zero'`, paired with `se_mode='fitted'`, i.e. `Var(eps_i) = sigma^2 v_i`. Results produced under `'zero'` **with the deprecated known-variance SE** are invalid (`docs/ase_validation.md` §7g), so read the PAIRING and never the `tau_mode` alone; `hapmixqtl.fine_mapping_provenance()` classifies a file as `ok`, `stale` or `unknown`

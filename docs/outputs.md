### Output files
#### Mode `cis_nominal`
Column | Description
--- | ---
`phenotype_id` | Phenotype ID
`variant_id` | Variant ID
`start_distance` | Distance between the variant and phenotype start position (e.g., TSS)
`end_distance` | Distance between the variant and phenotype end position (only present if different from start position)
`af` | In-sample ALT allele frequency of the variant
`ma_samples` | Number of samples carrying at least one minor allele
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
`ma_samples` | Number of samples carrying at least one minor allele
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

---

## hapmixQTL outputs

This section covers every file the shipped hapmixQTL paths write: the `tensorqtl` command-line modes `hapmixqtl_nominal` (`hapmixqtl.map_nominal`) and `hapmixqtl` (`hapmixqtl.map_cis`), and the Salmon driver `scripts/run_hapmixqtl_from_salmon.py`. The command line fixes `tau_mode='zero'` and defaults to `se_mode='fitted'`, which together are default mode (`docs/hapmixqtl_methods.md`); its one alternative, `--se_mode robust`, is described under `hapmixqtl_nominal`. The Salmon runner runs default mode only. The equations and section numbers cited below are those of `docs/hapmixqtl_methods.md`.

**Conventions.** Effect sizes are log2 allelic fold changes per ALT allele: a `slope` of 1 means the ALT haplotype is expressed at twice the REF haplotype's level, and the allelic fold change is `2^slope`. "The allelic channel" is the regression of the donor's log2 haplotype ratio on signed heterozygosity; "the total channel" is the regression of the donor's log2 half-read total expression on half the dosage; "the combined" statistic is their inverse-variance combination with Meier's correction, a first-order inflation of its standard error for channel weights estimated from the same residuals they combine (Sections 4.4 and 4.5). A channel "carries weight" at a variant when it is switched on and its predictor varies among its informative donors there.

**Input provenance.** The association tables do not record how their inputs were built. The Salmon runner builds them with `prepare_default_inputs` (half-read total, unit total working variance, Gibbs allelic variance with its admission rule) and records that in its `eval_bundle.json`. The command-line route reads precomputed BED files (`--hap_A`, `--hap_T`, `--hap_Va` required; `--hap_Vt` optional, ones when omitted) and cannot verify the total transform or the allelic admission rule, so keep the input-preparation record with any table it produces.

### Mode `hapmixqtl_nominal`

One parquet file per chromosome, `${prefix}.hapmixqtl_pairs.${chr}.parquet`, with one row per (phenotype, cis variant) pair, plus the run log `${prefix}.tensorQTL.hapmixqtl_nominal.log`. Variants are not filtered for constant dosage here (Section 5.1).

Column | Description
--- | ---
`phenotype_id` | Phenotype (gene) ID.
`variant_id` | Variant ID.
`start_distance` | Variant position minus the phenotype's start position (the TSS when one position is given).
`end_distance` | Variant position minus the phenotype's end position; present only when the phenotype positions have separate start and end columns.
`af` | In-sample ALT allele frequency of the variant, from the mean-imputed dosages.
`ma_samples` | Number of donors carrying at least one minor allele.
`ma_count` | Number of minor alleles.
`pval_nominal` | Two-sided p-value of the combined statistic `slope / slope_se` on `dof_nominal` degrees of freedom (equation 16). NaN where neither channel carries weight at the pair, for example a monomorphic variant, because nothing was tested there.
`slope` | Combined effect size, the inverse-variance weighted mean of `slope_a` and `slope_t` (equation 13). Equal to `slope_t` when `allelic_admitted` is False, and to the one channel's slope wherever only one channel carries weight.
`slope_se` | Standard error of `slope`: the plug-in inverse-variance standard error times the square root of Meier's factor `1 + 4 f_a f_t (1/dof_a + 1/dof_t)`, with `f_a`, `f_t` the two channels' weight shares (equation 16). The factor corrects for channel weights estimated from the same residuals they combine and is exactly 1 wherever fewer than two channels carry weight.
`pval_a` | Two-sided p-value of the allelic channel alone, `slope_a / slope_a_se` on `dof_a` (equation 14). Reported even for a gene below the allelic admission floor, whose combined statistic does not use it. 1 where the allelic predictor is invalid at the variant (no heterozygous informative donor); NaN when the allelic channel is switched off.
`slope_a` | Allelic channel's slope: the log2 haplotype ratio regressed through the origin on signed heterozygosity, weighted by the reciprocal Gibbs variance (equations 8 and 11). 0 where the predictor is invalid.
`slope_a_se` | Standard error of `slope_a` on the allelic channel's fitted residual scale (equation 12). Infinite where the predictor is invalid.
`pval_t` | Two-sided p-value of the total channel alone, `slope_t / slope_t_se` on `dof_t`. NaN when the total channel is switched off.
`slope_t` | Total channel's slope: the log2 half-read total regressed on half the dosage with an intercept and the covariates, unweighted.
`slope_t_se` | Standard error of `slope_t` on the total channel's fitted residual scale. Infinite for a constant-dosage variant.
`pval_cis_trans` | Wald test that the two channels estimate the same effect, `(slope_a - slope_t) / sqrt(slope_a_se^2 + slope_t_se^2)`, referred to the Welch-Satterthwaite degrees of freedom of that difference, `(se_a^2 + se_t^2)^2 / (se_a^4/dof_a + se_t^4/dof_t)` (equation 20). A small value flags a pair whose combined slope should not be read as a cis log2 allelic fold change. NaN when either channel has no finite standard error. A diagnostic, not a filter.
`dof_nominal` | Reference degrees of freedom of `pval_nominal`: the Welch-Satterthwaite degrees of freedom of the combination, `(w_a + w_t)^2 / (w_a^2/dof_a + w_t^2/dof_t)` with `w = 1/se^2` per pair (equation 15). Equal to `dof_a` where the total channel carries no weight and to `dof_t` where the allelic channel carries none (including every pair of a gene below the admission floor); NaN where neither does.
`dof_a` | Residual degrees of freedom of the allelic channel's fitted scale, `n_a - 1 - n_cov_a` over the gene's informative allelic donors (`n_a - 1` for the default through-origin design). NaN when the channel is switched off. Constant within a gene.
`dof_t` | Residual degrees of freedom of the total channel's fitted scale, `n_t - 2 - n_cov` over the gene's informative total-channel donors (every donor when the total working variance is 1 and no count cutoffs are applied). NaN when the channel is switched off. Constant within a gene.
`allelic_admitted` | Whether the allelic channel enters the combined statistic: True when the gene has at least 15 informative allelic donors (`MIN_ALLELIC_DONORS`), and whenever the gene has no total channel. When False, `slope`, `slope_se` and `pval_nominal` are the total channel's. Constant within a gene. The count does not check phase: in a run without phase frames the flag can read True while the allelic predictor is zero at every variant, so `slope_a` is 0, `pval_a` is 1 and the combined columns are the total channel's.

**With `--se_mode robust`.** The command line also offers, for this mode only, the HC1 sandwich standard error, which estimates each slope's variance from the donors' squared residuals times their squared predictors, with a small-sample factor, instead of from a fitted residual scale. Every p-value is then referred to one shared `N - 2 - max(n_cov, n_cov_a)`, which the three dof columns carry; there is no admission floor (`allelic_admitted` is True whenever the gene has any informative allelic donor) and no Meier factor.

### Mode `hapmixqtl`

The lead association per phenotype with its gene-level permutation and Beta-approximated p-values, written as a tab-separated table `${prefix}.hapmixqtl.txt.gz` indexed by `phenotype_id`, plus the run log `${prefix}.tensorQTL.hapmixqtl.log`. The columns appear in the order below.

Column | Description
--- | ---
`phenotype_id` | Phenotype (gene) ID; the table index.
`num_var` | Number of variants scanned in the cis window, after the MAF filter and the removal of constant-dosage variants.
`beta_shape1`, `beta_shape2` | Parameters of the Beta distribution fitted to the permutation p-values (equation 19). NaN when the Beta approximation is disabled (`--disable_beta_approx`), fails, or the gene was not tested.
`true_df` | Effective degrees of freedom of the Beta approximation, fitted so that the first Beta shape parameter is 1, starting from the constant `N - 2 - max(n_cov, n_cov_a)` (Section 5.4). Not the lead's t reference, which is `dof_nominal`.
`pval_true_df` | The lead's statistic converted to a p-value on `true_df`, through the constant correlation-scale map; the value the Beta CDF is applied to.
`variant_id` | Lead variant: the variant with the largest combined absolute t statistic in the window.
`start_distance` | Lead position minus the phenotype's start position.
`end_distance` | Lead position minus the phenotype's end position; always present, and equal to `start_distance` when the phenotype has a single position.
`ma_samples`, `ma_count`, `af` | Allele statistics of the lead, as in `hapmixqtl_nominal`.
`pval_nominal` | The lead's combined two-sided p-value on `dof_nominal`, with Meier's correction, exactly as `map_nominal` reports that pair. It is the best of the window and never a gene-level p-value; with a per-variant reference it need not be the gene's smallest `map_nominal` p. NaN for a gene that was not tested.
`slope`, `slope_se` | The lead's combined effect size and its Meier-corrected standard error.
`slope_a`, `slope_a_se` | The lead's allelic-channel slope and fitted-scale standard error.
`slope_t`, `slope_t_se` | The lead's total-channel slope and fitted-scale standard error.
`alpha_cis` | `slope_a / slope_t` at the lead. NaN when `|slope_t|` is below 1e-12 or either channel has no finite standard error.
`pval_cis_trans` | The cis/trans Wald test at the lead, as in `hapmixqtl_nominal`.
`pval_perm` | Empirical gene-level p-value: (number of permutations whose window maximum is at least the observed maximum, plus one) over (number of permutations plus one), with the donor-record permutation null named in `perm_scheme` (equation 18). Meier's factor is part of both the observed and the permuted statistics. NaN for a gene that was not tested.
`pval_beta` | Beta-approximated gene-level p-value (equation 19); the detection call when present, `pval_perm` otherwise. NaN in the cases listed for `beta_shape1`, so check a run for missing values in this column.
`tau_a`, `tau_t` | Deprecated path only: the additive variance of each channel used for the reported statistic under `tau_mode='estimate'`. Empty (None) in default mode, which has no additive variance term; empty is not zero.
`tau_a_null`, `tau_t_null` | Deprecated path only: each channel's additive variance estimated under the null model for the scan. Empty in default mode.
`tau_refit` | Deprecated path only: whether either channel's additive variance was re-estimated with the lead in the design. Always False in default mode, where no refit runs; the command line's `--tau_refit` is accepted and has no effect.
`c_a`, `c_a_null` | Deprecated path only: the allelic multiplicative factor of the two-component variance models, for the reported statistic and for the scan. Empty in default mode.
`c_a_converged` | Deprecated path only: whether the allelic two-component variance fit converged. Always True in default mode, where no such fit runs.
`c_a_raw`, `tau_a_raw` | Deprecated path only: the unshrunk two-component solution under an empirical-Bayes variance prior. Empty in default mode.
`c_a_floored` | Deprecated path only: whether a parameter of the two-component fit was clamped at zero. Always False in default mode.
`variance_prior` | Deprecated path only: whether an empirical-Bayes variance prior was used. Always False in default mode.
`variance_model` | The value of the deprecated `variance_model` argument. Reads `additive` in default mode because that is the argument's default; no variance function is fitted in default mode and no additive variance term is used, so do not read this column as describing the model.
`perm_scheme` | The permutation null that produced `pval_perm` and `pval_beta`: `records_signflip` (the default: donor records permuted with each permuted record's haplotype labels swapped with probability one half), `records` (no swap) or `residuals` (the retained whitened-residual scheme). Section 5.3.
`n_genotype_covariates` | Number of covariate columns tied to the genotypes (genotype principal components, `map_cis(genotype_covariates_df=...)`), which stay in genotype order under the permutation null while every other covariate moves with the donor's RNA record. 0 when none were passed; the command line passes none, so on that route every covariate moves with the RNA record.
`dof_nominal` | The lead's t reference: the Welch-Satterthwaite degrees of freedom of its combination, as in `hapmixqtl_nominal`. NaN for a gene that was not tested.
`allelic_admitted` | Whether the allelic channel entered the scanned statistic (at least 15 informative allelic donors, waived without a total channel). The floor is applied identically to the observed scan, every permutation and the lead, so `pval_perm` and `pval_beta` are built from the statistic that was observed. When False, the lead's `slope`, `slope_se` and `pval_nominal` are the total channel's. As in `hapmixqtl_nominal`, the count does not check phase, so in a run without phase frames the flag can read True although the allelic channel contributes nothing.
`loo_donor` | Leave-one-donor-out influence at the lead (Section 5.6): the donor ID whose exclusion from both channels moves the lead's combined absolute t furthest toward zero. Every donor informative in either channel is tried, except a donor with leverage 1 in a channel's covariate design (for example the only donor at one level of a categorical covariate), whose exclusion would leave that design rank-deficient and which is skipped. Empty when no donor can be evaluated, and when the run is not in default mode.
`loo_pval_nominal` | The lead's combined nominal p-value with `loo_donor` excluded: an exact refit of the lead (equal to `map_nominal` at that variant with the donor masked out of both channels), with per-channel degrees of freedom, the sparse-channel rule, the admission floor and Meier's factor all recomputed, referred to that refit's own Welch-Satterthwaite degrees of freedom. Two limits: the lead is held fixed, although excluding the donor can move the lead to another variant; and `pval_perm` and `pval_beta` are not recomputed, so the column measures how much the lead's nominal statistic rests on one donor, not whether the gene-level call survives without it. Because it is the minimum over single exclusions it can be smaller than `pval_nominal` when every exclusion strengthens the lead. NaN when no donor can be evaluated or when the selected exclusion leaves no channel carrying weight. A diagnostic, never a filter.
`qval` | Command line only, when rpy2 and the R package qvalue are available: Storey's q-value of `pval_beta` (of `pval_perm` when no Beta fit exists), the smallest false discovery rate at which the gene would be called, estimated with the proportion of true nulls taken from the p-value distribution.
`pval_nominal_threshold` | Command line only, with `qval`: the per-gene nominal threshold, the inverse Beta CDF (`beta_shape1`, `beta_shape2`) at the gene-level p-value cutoff that corresponds to the requested false discovery rate. It is on the scale of `pval_true_df` (the statistic mapped through the constant correlation-scale reference and then `true_df`), not on the scale of `map_nominal`'s `pval_nominal`, which uses the per-pair Welch-Satterthwaite reference. To apply it to a pair, convert the pair's `(slope / slope_se)^2` the same way before comparing.

**Genes that were not tested.** A gene in which neither channel carries weight has NaN `pval_nominal`, `pval_perm` and `dof_nominal` and no Beta fit; the log reports how many there were. A gene with no variant left after filtering is skipped and has no row.

**Two scales in one row.** `pval_perm` and `pval_beta` are gene-level and come from the permutation scan; `slope`, `slope_se` and `pval_nominal` describe the lead. The nominal p-values (`pval_nominal`, `pval_a`, `pval_t`) are not the detection call. Before the point-estimate rules of 2026-09-25 they were measured to be anticonservative under record permutation of observed data (0.068 at nominal 0.05 for the combined statistic), through a coupling of the Gibbs weights with residual size that a single fitted scale does not capture; on the corrected pipeline's stored null, whose total phenotype predates the half-read default, the shipped weighting rejected at 0.0505 at 0.05 and 0.0012 at 0.001 before Meier's correction. A stored null of the exact shipped configuration is pending. The empirical p-values are unaffected by that scale error, but a gene-level call can rest on a single donor's record, which `loo_donor` and `loo_pval_nominal` make visible. Measurements and their scope: `docs/hapmixqtl_methods.md` Sections 9 and 10, and `docs/pipeline_rules.md`.

### Salmon runner: `scripts/run_hapmixqtl_from_salmon.py`

The runner reads Salmon point estimates and Gibbs draws, builds the default-mode inputs, gates on reference mapping bias, runs `map_cis` in default mode and writes into `--out`:

File | Contents
--- | ---
`hapmixqtl_cis.tsv.gz` | The `map_cis` table of mode `hapmixqtl`, with `phenotype_id` as the first column rather than the index, and without `qval` or `pval_nominal_threshold` (the runner computes no q-values). With `--str-vcf` or `--multiallelic` it has one more column, `variant_type`, the lead's row type: `snp`, `str` (a short tandem repeat entered as per-haplotype repeat length, so its slope is per repeat unit) or `ma_allele` (one split row of a multi-ALT site). The table carries donor identifiers in `loo_donor`; keep it local.
`eval_bundle.json` | Aggregate statistics only, with no per-donor values, designed to be shared (keys below).
`edger/` | Written when `--edger-dir` is not given: `totals_all.tsv.gz` (point-estimate totals of every gene, the edgeR input), `restrict.txt` (the gene restriction), `edger_samples.tsv` (per donor: `lib_size`, `norm_factor`, `eff_lib_size`), `calibration_genes.txt` (the eQTL gene set) and `filter_by_expr_all_biotypes.txt` (genes passing edgeR's `filterByExpr` before the restriction).
`rasqual_cis.tsv.gz` | Only with `--rasqual`: one row per gene from the RASQUAL comparison run on the biallelic SNPs.

When the reference-bias gate fires and `--force` is not given, the runner stops before mapping: it writes `eval_bundle.json` with `meta` (`n_samples`, `n_genes`), `reference_bias` and `note: "no cis results produced"` (the `edger/` folder, written earlier, is left in place) and no `hapmixqtl_cis.tsv.gz`.

**`eval_bundle.json` keys.**

Key | Contents
--- | ---
`meta` | `n_samples`, `n_genes_tested`, `n_variants`, `n_gibbs_draws`, `n_covariates_rna`, `n_covariates_genotype`; `phenotype`, a description of the input transforms; `default_input_provenance` (`total_transform`, `total_working_variance` `"unit"`, `ase_count_noise`, `ase_one_sided_threshold` 0.5, and `n_ase_one_sided_excluded` out of `n_ase_donor_gene_pairs`, the donor-gene pairs with exactly one haplotype below 0.5 reads); `median_Va` (over every donor-gene pair, including the zeros of pairs excluded from the allelic channel) and `median_Vt` (1); `mode` `"default_half_read_split"`, `tau_mode` `"zero"`, `se_mode` `"fitted"`, `total_working_variance` `"unit"`; `tau_refit` `false`, because default mode has no lead refit; and `seed`, the seed of the permutation stream (42), which makes `pval_perm` and `pval_beta` reproduce from run to run.
`reference_bias` | The output of `reference_bias_diagnostic` without its per-gene vector: `ref_fraction` (mean over genes of the pooled reference-allele fraction; 0.5 is unbiased), `implied_phi` (the same on RASQUAL's scale), `z`, `pvalue`, `n_obs`, `n_genes` (absent when too few genes are usable for a test), `n_reads`, `flag` (True when bias is detected at p < 1e-3) and `message`. Section 6.3 of the methods.
`pvalues` | `n`, `lambda_gc`, `frac_lt_0.05`, `frac_lt_1e-5` and a 200-point QQ curve (`qq_observed_-log10`, `qq_expected_-log10`), computed over the genes' lead `pval_nominal`. Each is the best of its window, so these values are inflated by selection under the null and do not measure calibration; the calibrated gene-level quantities are `pval_perm` and `pval_beta`.
`channel_concordance` | Least-squares regression across genes of the lead's `slope_a` on `slope_t` (when more than 20 genes have both): `slope`, `slope_se`, `intercept`, `r`. Both channels estimate the same quantity under a purely cis effect, so the expected slope is 1; a departure localizes a bias to one channel.
`cis_trans` | `n_genes`; `frac_pval_cis_trans_lt_0.05`; `n_bh_q_lt_0.10`, the number of leads whose `pval_cis_trans` passes the Benjamini-Hochberg procedure at a false discovery rate of 0.10 (the step-up rule that bounds the expected proportion of false discoveries among the rejected); and, when at least 10 genes have `pval_beta` < 0.05, `alpha_cis_quantiles_among_significant` (5th, 25th, 50th, 75th and 95th percentiles of `alpha_cis` among them).
`effect_sizes` | 5th, 25th, 50th, 75th and 95th percentiles of the leads' `slope` and `slope_se`, with their counts.
`nonstandard` | Only with `--str-vcf` or `--multiallelic`: which options were enabled, the number of scan rows by type, the leads by type, and the fraction of leads that are not SNPs.
`rasqual` | `{"note": "not run (pass --rasqual to enable)"}`, or, with `--rasqual`, the number of genes attempted and converged, quantiles of RASQUAL's fitted `phi`, `delta`, `theta` and `chi2`, a comparison of RASQUAL's `phi` with `reference_bias.ref_fraction`, the rank correlation of the two methods' statistics across genes, and, with `--rasqual-input both`, the effect of RASQUAL's input representation.

### Outputs no longer produced

No shipped path writes the following files; tables carrying these names are historical.

- **Fine-mapping** (`${prefix}.hapmixqtl_SuSiE_summary.parquet`, `${prefix}.hapmixqtl_SuSiE.pickle`). The command-line mode `hapmixqtl_susie` was removed on 2026-10-01. `map_susie` has no per-channel residual scale and refuses default mode; its credible sets and posterior inclusion probabilities have not been validated under any mode (`docs/hapmixqtl_methods.md`, Section 8). `fine_mapping_provenance` classifies a stored summary by the `tau_mode` column it recorded.
- **The STR and multi-allelic second pass** (`hapmixqtl_str_curvature.tsv.gz`, `hapmixqtl_multiallelic_sites.tsv.gz`, `hapmixqtl_multiallelic_alleles.tsv.gz`, and the `str_curvature` and `multiallelic_categorical` keys of `eval_bundle.json`). The Salmon runner no longer runs the second pass, because its joint fits have known-variance standard errors only and refuse default mode (Section 8). `--str-vcf` and `--multiallelic` still add STR and multi-ALT rows to the `map_cis` scan.

### Reading tables written by earlier versions

The columns above describe the shipped code. Tables written earlier differ as follows; the dates are those of the code changes.

- Before 2026-09-25, effect sizes were in natural-log units (`compute_summaries_from_gibbs`); tables written by the retired `scripts/compare_pipelines.py` are also natural log. From 2026-09-25 to 2026-09-29 the runner used log2 with a `log2(CPM+1)` total and Gibbs variance in both channels; from 2026-09-29 it uses the half-read total with unit total weights, and its `eval_bundle.json` records `mode: default_half_read_split`.
- Before commit 8a06803 (2026-09-27) the tables have no `dof_nominal`, `dof_a`, `dof_t` or `allelic_admitted` columns, and `pval_nominal`, `pval_a` and `pval_t` were all referred to one shared `N - 2 - max(n_cov, n_cov_a)` (73 on BrainVar). Before the same date the `map_cis` lead's `slope_a_se`, `slope_t_se`, `alpha_cis` and `pval_cis_trans` used the known-variance standard error.
- Tables written between commit 8a06803 and the adoption of Meier's correction later on 2026-09-27 have the dof columns but an uncorrected `slope_se` and `pval_nominal`.
- Before 2026-10-01 `map_cis` tables have no `loo_donor` or `loo_pval_nominal`, and the runner's `eval_bundle.json` recorded `tau_refit: true`, which had no effect in default mode. Runner bundles without a `seed` key come from runs that passed no permutation seed, whose `pval_perm` and `pval_beta` do not reproduce exactly.

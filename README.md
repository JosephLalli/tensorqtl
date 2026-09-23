## tensorQTL

tensorQTL is a GPU-enabled QTL mapper, achieving ~200-300 fold faster *cis*- and *trans*-QTL mapping compared to CPU-based implementations.

If you use tensorQTL in your research, please cite the following paper:
[Taylor-Weiner, Aguet, et al., *Genome Biol.*, 2019](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-019-1836-7).</br>
Empirical beta-approximated p-values are computed as described in [Ongen et al., *Bioinformatics*, 2016](https://academic.oup.com/bioinformatics/article/32/10/1479/1742545).

### Install
You can install tensorQTL using pip:
```
pip3 install tensorqtl
```
or directly from this repository:
```
$ git clone git@github.com:broadinstitute/tensorqtl.git
$ cd tensorqtl
# install into a new virtual environment and load
$ mamba env create -f install/tensorqtl_env.yml
$ conda activate tensorqtl
```
To install the latest version from this repository, run
```
pip install pip@git+https://github.com/broadinstitute/tensorqtl.git
```

To use PLINK 2 binary files ([pgen/pvar/psam](https://www.cog-genomics.org/plink/2.0/input#pgen)), [pgenlib](https://github.com/chrchang/plink-ng/tree/master/2.0/Python) must be installed using either
```
pip install Pgenlib
```
(this is included in `tensorqtl_env.yml` above), or from the source :
```
git clone git@github.com:chrchang/plink-ng.git
cd plink-ng/2.0/Python/
python3 setup.py build_ext
python3 setup.py install
```

### Requirements

tensorQTL requires an environment configured with a GPU for optimal performance, but can also be run on a CPU. Instructions for setting up a virtual machine on Google Cloud Platform are provided [here](install/INSTALL.md).

### Input formats
Three inputs are required for QTL analyses with tensorQTL: genotypes, phenotypes, and covariates. 
* Phenotypes must be provided in BED format, with a single header line starting with `#` and the first four columns corresponding to: `chr`, `start`, `end`, `phenotype_id`, with the remaining columns corresponding to samples (the identifiers must match those in the genotype input). In addition to .bed/.bed.gz, BED input in .parquet is also supported. The BED file can specify the center of the *cis*-window (usually the TSS), with `start == end-1`, or alternatively, start and end positions, in which case the *cis*-window is [start-window, end+window]. A function for generating a BED template from a gene annotation in GTF format is available in [pyqtl](https://github.com/broadinstitute/pyqtl) (`io.gtf_to_tss_bed`).
* Covariates can be provided as a tab-delimited text file (covariates x samples) or dataframe (samples x covariates), with row and column headers.
* Genotypes should preferrably be in [PLINK2](https://www.cog-genomics.org/plink/2.0/) pgen/pvar/psam format, which can be generated from a VCF as follows:
  ```
  plink2 \
      --output-chr chrM \
      --vcf ${plink_prefix_path}.vcf.gz \
      --out ${plink_prefix_path}
  ```
  If using `--make-bed` with PLINK 1.9 or earlier, add the `--keep-allele-order` flag. 
  
  Alternatively, the genotypes can be provided in bed/bim/fam format, or as a parquet dataframe (genotypes x samples). 


The [examples notebook](example/tensorqtl_examples.ipynb) below contains examples of all input files. The input formats for phenotypes and covariates are identical to those used by [FastQTL](https://github.com/francois-a/fastqtl).

### Examples
For examples illustrating *cis*- and *trans*-QTL mapping, please see [tensorqtl_examples.ipynb](example/tensorqtl_examples.ipynb).

### Running tensorQTL
This section describes how to run the different modes of tensorQTL, both from the command line and within Python.
For a full list of options, run
```
python3 -m tensorqtl --help
```

#### Loading input files
This section is only relevant when running tensorQTL in Python.
The following imports are required:
```
import pandas as pd
import tensorqtl
from tensorqtl import genotypeio, cis, trans
```
Phenotypes and covariates can be loaded as follows:
```
phenotype_df, phenotype_pos_df = tensorqtl.read_phenotype_bed(phenotype_bed_file)
covariates_df = pd.read_csv(covariates_file, sep='\t', index_col=0).T  # samples x covariates
```
Genotypes can be loaded as follows, where `plink_prefix_path` is the path to the VCF in PLINK format (excluding `.bed`/`.bim`/`.fam` extensions):
```
pr = genotypeio.PlinkReader(plink_prefix_path)
# load genotypes and variants into data frames
genotype_df = pr.load_genotypes()
variant_df = pr.bim.set_index('snp')[['chrom', 'pos']]
```
To save memory when using genotypes for a subset of samples, a subset of samples can be loaded (this is not strictly necessary, since tensorQTL will select the relevant samples from `genotype_df` otherwise):
```
pr = genotypeio.PlinkReader(plink_prefix_path, select_samples=phenotype_df.columns)
```

#### *cis*-QTL mapping: permutations
This is the main mode for *cis*-QTL mapping. It generates phenotype-level summary statistics with empirical p-values, enabling calculation of genome-wide FDR.
In Python:
```
cis_df = cis.map_cis(genotype_df, variant_df, phenotype_df, phenotype_pos_df, covariates_df)
tensorqtl.calculate_qvalues(cis_df, qvalue_lambda=0.85)
```
Shell command:
```
python3 -m tensorqtl ${plink_prefix_path} ${expression_bed} ${prefix} \
    --covariates ${covariates_file} \
    --mode cis
```
`${prefix}` specifies the output file name.

#### *cis*-QTL mapping: summary statistics for all variant-phenotype pairs
In Python:
```
cis.map_nominal(genotype_df, variant_df, phenotype_df, phenotype_pos_df,
                prefix, covariates_df, output_dir='.')
```
Shell command:
```
python3 -m tensorqtl ${plink_prefix_path} ${expression_bed} ${prefix} \
    --covariates ${covariates_file} \
    --mode cis_nominal
```
The results are written to a [parquet](https://parquet.apache.org/) file for each chromosome. These files can be read using `pandas`:
```
df = pd.read_parquet(file_name)
```
#### *cis*-QTL mapping: conditionally independent QTLs
This mode maps conditionally independent *cis*-QTLs using the stepwise regression procedure described in [GTEx Consortium, 2017](https://www.nature.com/articles/nature24277). The output from the permutation step (see `map_cis` above) is required.
In Python:
```
indep_df = cis.map_independent(genotype_df, variant_df, cis_df,
                               phenotype_df, phenotype_pos_df, covariates_df)
```
Shell command:
```
python3 -m tensorqtl ${plink_prefix_path} ${expression_bed} ${prefix} \
    --covariates ${covariates_file} \
    --cis_output ${prefix}.cis_qtl.txt.gz \
    --mode cis_independent
```

#### *cis*-QTL mapping: interactions
Instead of mapping the standard linear model (p ~ g), this mode includes an interaction term (p ~ g + i + gi) and returns full summary statistics for the model. The interaction term is a tab-delimited text file or dataframe mapping sample ID to interaction value(s) (if multiple interactions are used, the file must include a header with variable names). With the `run_eigenmt=True` option, [eigenMT](https://www.cell.com/ajhg/fulltext/S0002-9297(15)00492-9)-adjusted p-values are computed.
In Python:
```
cis.map_nominal(genotype_df, variant_df, phenotype_df, phenotype_pos_df, prefix,
                covariates_df=covariates_df,
                interaction_df=interaction_df, maf_threshold_interaction=0.05,
                run_eigenmt=True, output_dir='.', write_top=True, write_stats=True)
```
The input options `write_top` and `write_stats` control whether the top association per phenotype and full summary statistics, respectively, are written to file.

Shell command:
```
python3 -m tensorqtl ${plink_prefix_path} ${expression_bed} ${prefix} \
    --covariates ${covariates_file} \
    --interaction ${interactions_file} \
    --best_only \
    --mode cis_nominal
```
The option `--best_only` disables output of full summary statistics.

Full summary statistics are saved as [parquet](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_parquet.html) files for each chromosome, in `${output_dir}/${prefix}.cis_qtl_pairs.${chr}.parquet`, and the top association for each phenotype is saved to `${output_dir}/${prefix}.cis_qtl_top_assoc.txt.gz`. In these files, the columns `b_g`, `b_g_se`, `pval_g` are the effect size, standard error, and p-value of *g* in the model, with matching columns for *i* and *gi*. In the `*.cis_qtl_top_assoc.txt.gz` file, `tests_emt` is the effective number of independent variants in the cis-window estimated with eigenMT, i.e., based on the eigenvalue decomposition of the regularized genotype correlation matrix ([Davis et al., AJHG, 2016](https://www.cell.com/ajhg/fulltext/S0002-9297(15)00492-9)). `pval_emt = pval_gi * tests_emt`, and `pval_adj_bh` are the Benjamini-Hochberg adjusted p-values corresponding to `pval_emt`. 

#### *trans*-QTL mapping
This mode computes nominal associations between all phenotypes and genotypes. tensorQTL generates sparse output by default (associations with p-value < 1e-5). *cis*-associations are filtered out. The output is in parquet format, with four columns: phenotype_id, variant_id, pval, maf.
In Python:
```
trans_df = trans.map_trans(genotype_df, phenotype_df, covariates_df,
                           return_sparse=True, pval_threshold=1e-5, maf_threshold=0.05,
                           batch_size=20000)
# remove cis-associations
trans_df = trans.filter_cis(trans_df, phenotype_pos_df.T.to_dict(), variant_df, window=5000000)
```
Shell command:
```
python3 -m tensorqtl ${plink_prefix_path} ${expression_bed} ${prefix} \
    --covariates ${covariates_file} \
    --mode trans
```

#### hapmixQTL: *cis*-QTL mapping with haplotype-resolved expression and inferential uncertainty
hapmixQTL is a generalization of [mixQTL](https://www.nature.com/articles/s41467-022-29123-9) that maps *cis*-QTLs using haplotype-resolved expression posteriors — e.g. [Salmon](https://combine-lab.github.io/salmon/) Gibbs draws obtained by quantifying reads against a personalized diploid transcriptome. The inferential (measurement) uncertainty captured by the Gibbs draws is propagated directly into the effect size and its standard error.

For each sample *i* and feature *f*, two information channels are combined:

1. **Allelic contrast (ASE)** channel, using the posterior-mean expression of each haplotype (`L`, `R`) with pseudocount `κ`:
   `a_i = log(yL_i + κ) − log(yR_i + κ)`, regressed **through the origin** on the **signed heterozygote indicator** `s_i = xL_i − xR_i ∈ {−1, 0, +1}` (from phased genotypes; `s_i = 0` if unphased/homozygous). The ASE regression and its tau estimator add no intercept, preserving the fit under paired expression/genotype H1/H2 relabeling within donors.
2. **Total expression** channel:
   `t_i = log((yL_i + yR_i)/2 + κ)`, regressed on the **half dosage** `g_i/2` so that both channels estimate the same quantity — the log allelic fold change (log aFC).

Per-sample inferential variances are computed across the `B` Gibbs draws:
`v_a_i = Var_b(a_i^(b))`, `v_t_i = Var_b(t_i^(b))`, the **Gibbs across-draw variances**. In **default mode** they enter each channel's weighted regression as a SHAPE, `w = 1/v`, with no additive floor, and the residual scale is fitted per variant: `Var(eps_i) = sigma^2 v_i`. Weighting is implemented via the sqrt-weight transform (`y* = sqrt(w) y`, `x* = sqrt(w) x`), which turns weighted least squares into ordinary dot products that vectorize across all *cis* variants on the GPU. Because the scale is fitted, rescaling every weight in a gene leaves beta and its SE unchanged, so only the within-gene shape of the Gibbs variances reaches the answer -- which is also why default mode is insensitive to a uniform error in the absolute scale of `v`.

The two channel estimates are merged by inverse-variance meta-analysis:
```
beta = (beta_a/se_a² + beta_t/se_t²) / (1/se_a² + 1/se_t²)
se   = sqrt(1 / (1/se_a² + 1/se_t²))
```
`beta` is interpretable as the log allelic fold change per ALT allele. When phase is unavailable (`s_i = 0` for all samples) the ASE channel is uninformative and the result reduces to the total-expression channel alone.

**Inputs.** hapmixQTL consumes five phenotype-like matrices (phenotypes × samples, in the same BED format as `read_phenotype_bed`), plus phased haplotype genotypes:

| Argument | Description |
| --- | --- |
| `--hap_A` | Allelic contrast `a_i` (BED) |
| `--hap_T` | Log total expression `t_i` (BED) |
| `--hap_Va` | Inferential variance of `a_i` (BED) |
| `--hap_Vt` | Inferential variance of `t_i` (BED) |
| `--hap_Cat` | Inferential covariance of `a_i,t_i` (optional; loaded for inspection only and **intentionally unused**: the two channel estimators are orthogonal under random phase, so the scalar meta-analysis is exact without it — see the `hapmixqtl` module docstring and `docs/ase_validation.md` §3) |
| `--phase_xL` | ALT allele on haplotype L (0/1), variants × samples, tab-delimited (optional) |
| `--phase_xR` | ALT allele on haplotype R (0/1), variants × samples, tab-delimited (optional) |
| `--ase_covariates` | What `--covariates` are projected out of the **allelic** channel: `none` (default, through-origin with no nuisance columns) or `shared` (the supplied total-channel covariates, without an automatic ASE intercept). Custom allelic nuisance predictors require an explicit biological interpretation and consistent sign under H1/H2 relabeling. The total channel retains its intercept. |
| `--tau_refit` | `hapmixqtl` mode only. τ is estimated once per gene under the null model, so a strong cis effect inflates it and shrinks every nominal statistic in the window by a common factor. With this flag each channel's τ is re-estimated with the lead's predictor in the model and the lead's `slope`, `slope_se`, `pval_nominal` and per-channel diagnostics are reported on that scale; `pval_perm` and `pval_beta` stay on the scan scale, where they are calibrated |
| `--se_mode` | `fitted` (**default**): the estimated-dispersion SE `sigma_hat/sqrt(xx)`. Together with the fixed `tau_mode='zero'` weighting this is **default mode**, `Var(eps_i) = sigma^2 v_i` -- mixQTL's Eq 11 form, in which the Gibbs variances are a SHAPE and their absolute scale cancels. `robust` is the HC1 sandwich, `map_nominal` only. `map_cis` accepts `fitted`, which refits a per-channel residual scale at every permutation exactly as mixQTL's permutation path does, and rejects `robust`, which has no permutation counterpart. The known-variance form is DEPRECATED and no longer selectable (`tensorqtl/fitted_variance.py`) |

The summary matrices can be precomputed from Gibbs draws with `hapmixqtl.compute_summaries_from_gibbs(yL, yR, kappa=0.5)`, where `yL`/`yR` are `[features, samples, draws]` arrays. Two optional arguments matter on real data: `yT` supplies the gene total summed over **all** transcripts (against a personalized diploid transcriptome `yL + yR` is a heterozygous-transcript subtotal, and using it makes `t` a two-point mixture determined by local heterozygosity, which is in LD with the variants being tested), and `count_noise=True` adds the plug-in Poisson variance of a log count to `Va`/`Vt`, without which a zero-count sample has `v_inf = 0` and so the largest weight in the gene. Both drivers under `scripts/` turn counting noise on by default; the library default is off, and it should stay off for bootstrap draws, which resample the reads and already carry counting noise. The positional `${expression_bed}` argument is still required by the CLI but ignored in hapmixQTL modes (all phenotype inputs come from the `--hap_*` flags).

**Nominal mapping** (all *cis* variant–phenotype pairs) writes one parquet per chromosome, `${output_dir}/${prefix}.hapmixqtl_pairs.${chr}.parquet`, with the combined `slope`/`slope_se`/`pval_nominal` plus the per-channel `slope_a`/`slope_a_se`/`pval_a` and `slope_t`/`slope_t_se`/`pval_t`, and `pval_cis_trans`, a Wald test that the two channels estimate the same effect (a diagnostic for effects that are not purely cis — see `hapmixqtl.cis_trans_diagnostic`):
```
python3 -m tensorqtl ${plink_prefix_path} ${expression_bed} ${prefix} \
    --mode hapmixqtl_nominal \
    --hap_A ${A_bed} --hap_T ${T_bed} --hap_Va ${Va_bed} --hap_Vt ${Vt_bed} \
    --phase_xL ${xL_file} --phase_xR ${xR_file} \
    --covariates ${covariates_file}
```
In Python:
```
from tensorqtl import hapmixqtl
hapmixqtl.map_nominal(genotype_df, variant_df, A_df, T_df, Va_df, Vt_df,
                      phenotype_pos_df, xL_df=xL_df, xR_df=xR_df,
                      prefix=prefix, covariates_df=covariates_df, output_dir='.')
```

**Permutation mapping** (top association per phenotype with empirical/beta-approximated p-values), analogous to `cis`, writes `${output_dir}/${prefix}.hapmixqtl.txt.gz`. Each gene's row also carries the lead variant's per-channel slopes, `alpha_cis = slope_a/slope_t` and `pval_cis_trans`: a small `pval_cis_trans` means the ASE and total channels disagree, so the combined slope should not be read as a cis log aFC (a trans component, reference mapping bias or phasing error attenuate it — `docs/ase_validation.md` §7c). It is a diagnostic column, not a filter.

Each row also reports the τ actually used, `tau_a`/`tau_t`, alongside the scan's null-model τ, `tau_a_null`/`tau_t_null`, and the flag `tau_refit`. Without `--tau_refit` these pairs are equal and everything is on the scan scale; with it, `slope`, `slope_se` and `pval_nominal` are on the refit scale while `pval_perm` and `pval_beta` remain on the scan scale. A lead's `pval_nominal` is never a gene-level p (it is the best of the window) — `pval_beta` is:
```
python3 -m tensorqtl ${plink_prefix_path} ${expression_bed} ${prefix} \
    --mode hapmixqtl \
    --hap_A ${A_bed} --hap_T ${T_bed} --hap_Va ${Va_bed} --hap_Vt ${Vt_bed} \
    --phase_xL ${xL_file} --phase_xR ${xR_file} \
    --covariates ${covariates_file}
```
In Python:
```
res_df = hapmixqtl.map_cis(genotype_df, variant_df, A_df, T_df, Va_df, Vt_df,
                           phenotype_pos_df, xL_df=xL_df, xR_df=xR_df,
                           covariates_df=covariates_df)
```

**SuSiE fine-mapping** identifies credible sets of candidate causal variants from the combined ASE + total evidence. Because both channels estimate the same shared effect (log aFC), the two sqrt-weighted, covariate-residualized channels are stacked into a single whitened design and passed to tensorQTL's existing [SuSiE](https://rss.onlinelibrary.wiley.com/doi/full/10.1111/rssb.12388) implementation (`tensorqtl.susie.susie`) unchanged, so any improvements to the core SuSiE code are inherited automatically. The sqrt-weight transform whitens each channel using the Gibbs inferential variances, so `estimate_residual_variance` defaults to `False`; set it to `True` to let SuSiE re-estimate a scalar dispersion instead. Outputs mirror `cis_susie`: a credible-set summary parquet (`${prefix}.hapmixqtl_SuSiE_summary.parquet`) and a pickle of the full per-phenotype results. The summary carries a `tau_mode` column and each pickle entry a `tau_mode` key, as provenance. Fine-mapping produced under `tau_mode='zero'` WITH the deprecated known-variance SE is invalid (`docs/ase_validation.md` sec 7g) and should be redone; note the scope, because default mode also uses `'zero'` but pairs it with `se_mode='fitted'`, which does not carry that defect -- read the PAIRING, never the `tau_mode` alone. `hapmixqtl.fine_mapping_provenance(summary_or_path)` classifies a results file as `ok`, `stale`, or `unknown` (no provenance column, i.e. produced before it was recorded).
```
python3 -m tensorqtl ${plink_prefix_path} ${expression_bed} ${prefix} \
    --mode hapmixqtl_susie \
    --hap_A ${A_bed} --hap_T ${T_bed} --hap_Va ${Va_bed} --hap_Vt ${Vt_bed} \
    --phase_xL ${xL_file} --phase_xR ${xR_file} \
    --covariates ${covariates_file} --max_effects 10
```
In Python:
```
summary_df, susie_res = hapmixqtl.map_susie(
    genotype_df, variant_df, A_df, T_df, Va_df, Vt_df,
    phenotype_pos_df, xL_df=xL_df, xR_df=xR_df,
    covariates_df=covariates_df, L=10, summary_only=False)
```

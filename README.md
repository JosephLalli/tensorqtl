## tensorQTL

tensorQTL is a GPU-enabled QTL mapper, achieving ~200-300 fold faster *cis*- and *trans*-QTL mapping compared to CPU-based implementations.

If you use tensorQTL in your research, please cite the following paper:
[Taylor-Weiner, Aguet, et al., *Genome Biol.*, 2019](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-019-1836-7).</br>
Empirical beta-approximated p-values are computed as described in [Ongen et al., *Bioinformatics*, 2016](https://academic.oup.com/bioinformatics/article/32/10/1479/1742545).

### Current work

This branch develops hapmixQTL. Start with the [current scientific state](docs/CURRENT_SCIENTIFIC_STATE.md), then the [compact decision and evidence index](CLAUDE.md#which-document-answers-which-question). The [hapmixQTL guide below](#hapmixqtl-cis-qtl-mapping-with-haplotype-resolved-expression), [pipeline rules](docs/pipeline_rules.md), and [Salmon deployment runbook](docs/brainvar_deploy_runbook.md) describe the current half-read split default. Dated benchmark reports retain the method configuration used for each measurement.

The [half-read analysis guide](docs/half_read_analysis.md) provides one command
to regenerate the beta/SE/p-value/power/PR reports from recorded benchmark
inputs, with pinned versions and exact comparisons to the saved tables.

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

#### hapmixQTL: cis-QTL mapping with haplotype-resolved expression

hapmixQTL extends [mixQTL](https://www.nature.com/articles/s41467-022-29123-9) with donor-specific Gibbs-informed ASE weights. The **default since 2026-09-29 is half-read split**: Salmon point estimates supply expression values, Gibbs draws inform ASE weights, and total expression uses unit weights. These instructions refer to this branch's source checkout. The published mixQTL implementation remains a separate, unchanged comparator.

For each gene and donor, let `pL` and `pR` be point-estimate counts from paired haplotype transcripts, `pT` the total over **all** transcripts, and `Leff` the edgeR effective library size (`lib.size × TMM factor`). Do not substitute `pL + pR` for `pT`.

| Channel | Phenotype | Working weights | Regression |
| --- | --- | --- | --- |
| ASE | `A = log2((pL + 0.5)/(pR + 0.5))` | `1/Va` on admitted donors | Signed heterozygosity `xL − xR`, through the origin |
| Total | `T = log2((pT + 0.5)/(Leff + 1) × 1e6)` | One for every donor | Half dosage `g/2`, with an intercept and total covariates |

`Va` is the across-draw variance of the same allelic log2 ratio, plus the existing Poisson counting term when `count_noise=True` (the preparation helper's default). ASE admission requires `Va > 1e-12`, positive haplotype-informative coverage, and exclusion when **exactly one** of `pL`, `pR` is below 0.5 reads. Excluded donor-gene pairs have `Va=0`. Both counts below 0.5 do not by themselves trigger the one-sided exclusion. Total-expression zeros remain finite and keep unit weight.

The half-read offset is added **before** library normalization. In `log2(CPM+1)`, the added one CPM corresponds to `Leff/1e6` reads; the new numerator always adds half a read. Existing expression PCs remain based on `log2(CPM+1)` on the same gene set and effective library sizes. The change improves beta recovery in the measured benchmarks, with somewhat larger reported SEs; it was accepted as an accuracy/precision tradeoff, not uniform precision improvement. See the [decision and evidence](docs/CURRENT_SCIENTIFIC_STATE.md).

**Fitting and uncertainty.** `map_nominal` and `map_cis` default to `tau_mode='zero', se_mode='fitted'`: both channels estimate a residual scale for their SEs. Unit total working variance does **not** mean the total noise variance or SE is fixed at one. The sqrt-weight transform retains the existing GPU matrix multiplication across cis variants. The combined slope uses inverse squared channel SEs. When both channels are admitted, its reported SE includes Meier's correction:

```text
wa = 1 / se_a²; wt = 1 / se_t²
beta = (wa * beta_a + wt * beta_t) / (wa + wt)
fa = wa / (wa + wt); ft = wt / (wa + wt)
M = 1 + 4 * fa * ft * (1 / dof_a + 1 / dof_t)
se = sqrt(M / (wa + wt))
```

Nominal p-values use the existing per-channel and Welch–Satterthwaite references. The combined scan requires at least 15 informative ASE donors when total expression is available; otherwise it uses total expression alone. Missing phase also removes ASE information. Slopes are on a log2 allelic fold-change scale, subject to the documented transform and estimation limitations. [Methods](docs/hapmixqtl_methods.md) and [output definitions](docs/outputs.md) give the exact rules.

**Prepare inputs in Python.** Counts are arrays with shape `[features, samples]`; haplotype Gibbs draws have shape `[features, samples, draws]`:

```python
import pandas as pd
from tensorqtl import hapmixqtl

A, T, Va, Vt = hapmixqtl.prepare_default_inputs(
    pL, pR, pT, effective_library_sizes, yL, yR, count_noise=True)
A_df, T_df, Va_df, Vt_df = [
    pd.DataFrame(x, index=phenotype_ids, columns=sample_ids)
    for x in (A, T, Va, Vt)
]
```

The direct mapping APIs require all four matrices and preserve supplied values and weights. They do not normalize counts automatically. `summaries_from_point_estimates` retains historical `log2(CPM+1)` totals and Gibbs total variance; `compute_summaries_from_gibbs` retains the older natural-log draw-mean summaries. Neither is the current default preparation helper. The [Salmon runner](docs/brainvar_deploy_runbook.md) constructs current inputs directly from quantification files.

**BED CLI inputs.** Matrices use the usual phenotype BED layout and identical gene/sample ordering. Phase columns must match genotype sample order exactly.

| Argument | Description |
| --- | --- |
| `--hap_A` | Required point-estimate ASE contrast |
| `--hap_T` | Required precomputed half-read total expression |
| `--hap_Va` | Required ASE working variance, with excluded donor-gene pairs set to zero |
| `--hap_Vt` | Optional total working-variance override; omission supplies ones, a supplied file is preserved |
| `--hap_Cat` | Optional covariance matrix, loaded for inspection only and unused by mapping |
| `--phase_xL`, `--phase_xR` | Optional paired ALT haplotype genotype matrices, variants × samples |
| `--ase_covariates` | `none` (default, through-origin) or `shared`; total expression retains its intercept |
| `--se_mode` | `fitted` (default) for nominal/permutation mapping; `robust` HC1 is nominal-only; this option is not passed to `map_susie` |
| `--tau_refit` | Legacy compatibility flag; no tau refit occurs in the current `tau_mode='zero'` CLI path |

BED files contain no raw counts or library sizes, so the CLI cannot verify the half-read transform or the ASE admission rule. Supply matrices prepared under the contract above. The positional `${expression_bed}` argument remains required by the parser but is ignored in hapmixQTL modes.

**Nominal mapping** writes `${prefix}.hapmixqtl_pairs.${chr}.parquet` with combined and per-channel effects, SEs, p-values, degrees of freedom, and channel diagnostics:

```bash
python3 -m tensorqtl ${plink_prefix_path} ${expression_bed} ${prefix} \
    --mode hapmixqtl_nominal \
    --hap_A ${A_bed} --hap_T ${T_bed} --hap_Va ${Va_bed} \
    --phase_xL ${xL_file} --phase_xR ${xR_file} \
    --covariates ${covariates_file}
```

```python
hapmixqtl.map_nominal(
    genotype_df, variant_df, A_df, T_df, Va_df, Vt_df,
    phenotype_pos_df, xL_df=xL_df, xR_df=xR_df,
    prefix=prefix, covariates_df=covariates_df, output_dir='.')
```

**Permutation mapping** uses donor-record permutations with random haplotype-label swaps (`records_signflip`). RNA covariates move with donor records; genotype-tied covariates stay with genotypes when supplied separately through the Python API or Salmon runner. It writes the top association per gene to `${prefix}.hapmixqtl.txt.gz`:

```bash
python3 -m tensorqtl ${plink_prefix_path} ${expression_bed} ${prefix} \
    --mode hapmixqtl \
    --hap_A ${A_bed} --hap_T ${T_bed} --hap_Va ${Va_bed} \
    --phase_xL ${xL_file} --phase_xR ${xR_file} \
    --covariates ${covariates_file}
```

```python
res_df = hapmixqtl.map_cis(
    genotype_df, variant_df, A_df, T_df, Va_df, Vt_df,
    phenotype_pos_df, xL_df=xL_df, xR_df=xR_df,
    covariates_df=covariates_df,
    genotype_covariates_df=genotype_covariates_df)
```

The lead's `pval_nominal` is a selected variant-level p-value, not a gene-level p-value; `pval_perm` and `pval_beta` account for the cis scan. `pval_cis_trans` measures disagreement between channels and is a diagnostic, not a filter. Legacy tau output fields remain for compatibility; current zero-tau mapping does not perform a tau refit.

**SuSiE compatibility path.** `map_susie` stacks weighted, covariate-residualized ASE and total inputs and has its own variance settings. It has no `se_mode` argument: the direct API retains `tau_mode='estimate'` and `estimate_residual_variance=False`, whereas the CLI passes `tau_mode='zero'`. Selecting `--se_mode fitted` does not change that fine-mapping path. Half-read nominal/permutation validation does not establish credible-set calibration. See the [fine-mapping limitations](docs/hapmixqtl_methods.md) before interpreting its results. The `fine_mapping_provenance` helper checks recorded `tau_mode`; it does not establish the input transform or validate the half-read method.

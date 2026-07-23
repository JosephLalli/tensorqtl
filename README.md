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
   `a_i = log(yL_i + κ) − log(yR_i + κ)`, regressed on the **signed heterozygote indicator** `s_i = xL_i − xR_i ∈ {−1, 0, +1}` (from phased genotypes; `s_i = 0` if unphased/homozygous).
2. **Total expression** channel:
   `t_i = log((yL_i + yR_i)/2 + κ)`, regressed on the **half dosage** `g_i/2` so that both channels estimate the same quantity — the log allelic fold change (log aFC).

Per-sample inferential variances are computed across the `B` Gibbs draws:
`v_a_i = Var_b(a_i^(b))`, `v_t_i = Var_b(t_i^(b))`. These are treated as **known** measurement variances and enter each channel's weighted regression as absolute precisions `w = 1/(v_inf + τ)` (default `τ = 0`). Weighting is implemented via the sqrt-weight transform (`y* = √w · y`, `x* = √w · x`), which turns weighted least squares into ordinary dot products that vectorize across all *cis* variants on the GPU. Because the variances are known, the standard error is the known-variance GLS SE (`se = √(1/xx)`), so inflating a channel's inferential variance genuinely widens its SE and down-weights it in the combination — an ordinary estimated-dispersion WLS would instead absorb that scale and ignore it.

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
| `--hap_Cat` | Inferential covariance of `a_i,t_i` (optional; unused by the default method) |
| `--phase_xL` | ALT allele on haplotype L (0/1), variants × samples, tab-delimited (optional) |
| `--phase_xR` | ALT allele on haplotype R (0/1), variants × samples, tab-delimited (optional) |
| `--tau_mode` | `zero` (default) or `estimate` (per-phenotype moment estimator of overdispersion) |
| `--se_mode` | `model` (default, known-variance GLS) or `robust` (HC1 sandwich) |

The summary matrices can be precomputed from Gibbs draws with `hapmixqtl.compute_summaries_from_gibbs(yL, yR, kappa=0.5)`, where `yL`/`yR` are `[features, samples, draws]` arrays. The positional `${expression_bed}` argument is still required by the CLI but ignored in hapmixQTL modes (all phenotype inputs come from the `--hap_*` flags).

**Nominal mapping** (all *cis* variant–phenotype pairs) writes one parquet per chromosome, `${output_dir}/${prefix}.hapmixqtl_pairs.${chr}.parquet`, with the combined `slope`/`slope_se`/`pval_nominal` plus the per-channel `slope_a`/`slope_a_se`/`pval_a` and `slope_t`/`slope_t_se`/`pval_t`:
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

**Permutation mapping** (top association per phenotype with empirical/beta-approximated p-values), analogous to `cis`, writes `${output_dir}/${prefix}.hapmixqtl.txt.gz`:
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

**SuSiE fine-mapping** identifies credible sets of candidate causal variants from the combined ASE + total evidence. Because both channels estimate the same shared effect (log aFC), the two sqrt-weighted, covariate-residualized channels are stacked into a single whitened design and passed to tensorQTL's existing [SuSiE](https://rss.onlinelibrary.wiley.com/doi/full/10.1111/rssb.12388) implementation (`tensorqtl.susie.susie`) unchanged, so any improvements to the core SuSiE code are inherited automatically. The sqrt-weight transform whitens each channel to unit-variance noise using the known Gibbs inferential variances, so `estimate_residual_variance` defaults to `False` (consistent with the known-variance standard errors used elsewhere in hapmixQTL); set it to `True` to instead let SuSiE re-estimate a scalar dispersion. Outputs mirror `cis_susie`: a credible-set summary parquet (`${prefix}.hapmixqtl_SuSiE_summary.parquet`) and a pickle of the full per-phenotype SuSiE results.
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

#### Knockoff-calibrated eGene FDR (Python API only)
This calls eGenes (phenotypes with a *cis* signal) at a knockoff-controlled false discovery rate, robust to SuSiE's own PIP miscalibration. For each phenotype, the *cis* genotype design is augmented with knockoff copies, a gene-level statistic is computed from `[X, X_knockoff]`, and genes are selected genome-wide at the target FDR before ordinary SuSiE localizes credible sets within the selected genes. **There is no CLI `--mode` for this yet** (see the `--mode` choices in `tensorqtl/tensorqtl.py`); it is called directly from Python: `tensorqtl.susie.map_egenes_knockoffs` for standard BED-format phenotypes (as for `cis`/`cis_susie`), or `tensorqtl.hapmixqtl.map_egenes_knockoffs` for the two-channel hapmixQTL phenotypes.

Two gene-level statistics are available (`statistic=`):
* `'kfc'` (**recommended**) — a continuous statistic `W_g = -log10(min cis p, real) − -log10(min cis p, knockoff)`, selected genome-wide by the empirical mirror-null knockoff+ threshold. This is the only path validated end-to-end on real genotypes (HPRC v2.0, N=232): realized FDR 0.06–0.07 at target FDR 0.10, power 0.31–0.55 (`docs/calibration_findings.md`, "Real-data (HPRC) validation").
* `'maxpip'` (module default, **legacy, not calibrated**) — `W_g = maxPIP(original) − maxPIP(knockoff)` from an augmented SuSiE fit. Under a null gene, SuSiE's prior variance collapses to exactly 0, producing a point mass at `W=0` that breaks the knockoff-null assumption used for selection; retained only for continuity (`docs/knockoff_susie_design.md`, "STATUS" §3.2). Do not use it for FDR claims unless also setting `susie(..., prior_variance_floor=...)` to remove the atom.

Standard *cis*-QTL phenotypes, using the configuration validated for `'kfc'`:
```python
from tensorqtl import susie
egene_df, localize_summary_df, diagnostics = susie.map_egenes_knockoffs(
    genotype_df, variant_df, phenotype_df, phenotype_pos_df, covariates_df,
    fdr=0.1, statistic='kfc', knockoff='gaussian', shrink=0.1, knockoff_offset=1)
```
`knockoff='gaussian', shrink=0.1` is the generator validated for the `kfc` statistic on real LD. The module default `knockoff='hmm'` (chromosome-coherent HMM knockoffs) instead targets the legacy `maxpip` path, where whole-chromosome coherence is the point. `egene_df` has one row per phenotype: `phenotype_id`, `knockoff_qvalue`, `selected` (boolean). `localize_summary_df` (from the default `localize=True`) holds ordinary SuSiE credible sets for the selected eGenes, in the same format as `cis_susie`'s summary output.

hapmixQTL phenotypes — the two-channel analog. Because the ASE channel needs phase, this path always uses chromosome-coherent phased haplotype knockoffs rather than Gaussian:
```python
from tensorqtl import hapmixqtl
egene_df, diagnostics = hapmixqtl.map_egenes_knockoffs(
    genotype_df, variant_df, A_df, T_df, Va_df, Vt_df, phenotype_pos_df,
    xL_df, xR_df, covariates_df=covariates_df, fdr=0.1, statistic='kfc')
```
Because the phased-HMM knockoff is unbiased but lower-power than Gaussian for this statistic on real LD (`docs/calibration_findings.md`, "Real-data (HPRC) validation"), expect this path to be more conservative than the standard *cis* `'kfc'` path above. Both functions require genotypes sorted by chromosome then position (the standard tensorQTL layout), since each chromosome must occupy a contiguous row block.


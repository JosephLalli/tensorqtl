## tensorQTL with hapmixQTL

This repository is a fork of [tensorQTL](https://github.com/broadinstitute/tensorqtl), the GPU-enabled QTL mapper, that adds **hapmixQTL**. hapmixQTL is an extension of [mixQTL](https://github.com/hakyimlab/mixqtl) ([Liang et al., *Nat. Commun.*, 2021](https://doi.org/10.1038/s41467-021-21592-8)): it maps *cis*-eQTLs from haplotype-resolved expression, quantified by Salmon against a personalized diploid transcriptome, and carries the quantifier's Gibbs variance into the allelic channel as per-donor weights. The upstream tensorQTL modes remain available and are documented in the [last part of this page](#upstream-tensorqtl).

### The two modes

hapmixQTL has exactly two modes. Other variance models that existed during development are deprecated and quarantined (see [Deprecated configurations](#deprecated-configurations)).

**Default mode** is the shipped hapmixQTL configuration. Its inputs are prepared by `prepare_default_inputs` in `tensorqtl/hapmixqtl.py` from Salmon point estimates (`quant.sf` NumReads) and Salmon's Gibbs draws. For each gene and donor, `pL` and `pR` are the point-estimate counts summed over haplotype-paired transcripts, `pT` is the point-estimate count summed over **all** transcripts of the gene (never `pL + pR`, which is zero for a donor with no heterozygous transcript pair in the gene), `yL` and `yR` are the paired counts in each Gibbs draw, and `Leff` is edgeR's effective library size: the library size times its TMM factor (trimmed mean of M-values, a per-library scale chosen so that most genes' log ratios to a reference library centre on zero).

| Channel | Phenotype | Working variance and weight | Regression |
| --- | --- | --- | --- |
| Allelic | `A = log2((pL + 0.5)/(pR + 0.5))` | `Va`, the across-draw variance of `log2((yL + 0.5)/(yR + 0.5))` plus, by default (`count_noise=True`), the counting term `(1/(pL + 0.5) + 1/(pR + 0.5)) / ln(2)^2`; weight `1/Va` | On the signed heterozygosity `xL - xR`, through the origin, no covariates |
| Total | `T = log2((pT + 0.5)/(Leff + 1) × 1e6)` (the half-read transform) | `Vt = 1` for every donor, including zero-count donors | On half dosage `g/2`, with an intercept and the covariates |

A donor-gene pair enters the allelic channel only if `Va > 1e-12`, `pL + pR > 0`, and not exactly one of `pL`, `pR` is below 0.5 reads; an excluded pair carries `Va = 0`. Slopes are on the log2 allelic fold-change scale, so a slope of 1 is a twofold effect.

Each channel's error variance is `Var(eps_i) = sigma^2 v_i`: the working variance `v_i` is a shape only, and the residual scale `sigma^2` is fitted per channel and variant (`tau_mode='zero', se_mode='fitted'`). A unit total working variance therefore does not mean the total channel's standard error is fixed at one. Each channel's p-value is referred to a t distribution on that channel's own residual degrees of freedom. The combined slope is the inverse-variance weighted mean of the two channel slopes, with weights `1/se_a²` and `1/se_t²`. Its p-value is referred to a t distribution on Welch-Satterthwaite degrees of freedom (the value that matches the first two moments of the combined variance estimate to a scaled chi-square, with the channel weights treated as fixed). Its standard error carries Meier's correction for estimated weights (Meier, 1953): because the weights are estimated from the same residuals they combine, the plug-in variance `1/(wa + wt)` is too small, and it is multiplied by the first-order factor below.

```text
wa = 1 / se_a²; wt = 1 / se_t²
beta = (wa * beta_a + wt * beta_t) / (wa + wt)
fa = wa / (wa + wt); ft = wt / (wa + wt)
M = 1 + 4 * fa * ft * (1 / dof_a + 1 / dof_t)
se = sqrt(M / (wa + wt))
```

The allelic channel enters the combined statistic only for a gene with at least 15 informative allelic donors (`MIN_ALLELIC_DONORS`); below that the gene is tested on the total channel alone. The floor is waived when there is no total channel.

The permutation null (`perm_scheme='records_signflip'`) permutes donor records against fixed genotypes: each donor's phenotype value, weight and RNA-tied covariate row move together, and each permuted record's haplotype labels are swapped with probability one half, which negates its allelic log ratio. Covariates tied to the genotypes (the genotype PCs) stay with the genotypes. Expression PCs are built on the same half-read transform, effective library sizes and gene set as the total phenotype, and the expression-PC gene filter equals the eQTL gene filter.

The detection call is the empirical permutation p-value: `pval_perm`, or `pval_beta`, its approximation by a Beta distribution fitted to the permuted minimum p-values ([Ongen et al., *Bioinformatics*, 2016](https://academic.oup.com/bioinformatics/article/32/10/1479/1742545)). The lead variant's `pval_nominal` is a selected variant-level p-value and is never a gene-level p-value.

**mixQTL mode** (`tensorqtl/mixqtl_replication.py`) is a NumPy port of `hakyimlab/mixqtl` at commit `624ae44`: hapmixQTL's estimator with each of the eleven points at which it departs from the published mixQTL code reverted, so the two can be compared on identical inputs. It consumes Salmon point estimates and never reads the Gibbs draws, so it is the no-draws comparator: what hapmixQTL has to beat, and the measure of what the draws buy. It keeps mixQTL's published natural-log response (divide its slopes and standard errors by ln 2 before comparing them with hapmixQTL's log2 values), its published count cutoffs (`PUBLISHED_CUTOFFS`: total count at least 100, each haplotype count between 50 and 1,000) and its allelic weight cap of 10.

#### Deprecated configurations

The fitted variance functions (`variance_model='additive'`, `'two_component'`, `'library_scaled'`), the empirical-Bayes `variance_prior`, `tau_mode='estimate'` which they require, and the known-variance standard error `se_mode='model'` are deprecated. The variance functions fit a donor's error variance from the gene's own residuals and then weight those residuals by the fit, and those with a free scale per gene discard the absolute scale of the Gibbs variance; the known-variance standard error treats the Gibbs variance as the entire error variance. The code is kept only to reproduce earlier results, in `tensorqtl/fitted_variance.py` with tests in `tests/fitted_variance/`, and a default-mode run never imports it. Neither the command line nor the Salmon runner offers these configurations.

### Installation

hapmixQTL is installed only from a checkout of this repository. The `tensorqtl` package on PyPI, and the `tensorqtl` entry in `install/tensorqtl_env.yml`, are upstream tensorQTL, which does not contain `hapmixqtl.py` or `mixqtl_replication.py`. An installed upstream copy can carry the same version number as this fork, so check which copy Python imports.

```bash
# from the root of this repository
mamba env create -f install/tensorqtl_env.yml   # optional: an environment with the dependencies
conda activate tensorqtl
pip install -e .                                # replaces any upstream tensorqtl in the environment
python3 -c "import tensorqtl.hapmixqtl, tensorqtl.mixqtl_replication"
```

The Salmon runner, `scripts/build_covariates.py`, the mixQTL driver and the plasmode benchmark put this repository first on `sys.path`, so they use this checkout even when an upstream copy is installed.

Requirements beyond the Python packages in `pyproject.toml`:

- The mapping functions use a GPU when PyTorch sees one and otherwise run on the CPU, more slowly.
- The Salmon runner (`scripts/run_hapmixqtl_from_salmon.py`), including its self-test, calls `Rscript` with the Bioconductor package edgeR to compute effective library sizes, unless `--edger-dir` supplies a finished normalization.
- Salmon output for hapmixQTL must come from a personalized diploid transcriptome (two copies of every transcript, one per haplotype, built from the phased VCF) and be run with `--numGibbsSamples` (the BrainVar deployment used 200 draws). A standard reference transcriptome carries no allelic information.
- The `hapmixqtl` command-line mode adds q-values only when `rpy2` and the R package `qvalue` are available, as in upstream tensorQTL ([install/INSTALL.md](install/INSTALL.md)).

To read PLINK 2 binary files ([pgen/pvar/psam](https://www.cog-genomics.org/plink/2.0/input#pgen)), [pgenlib](https://github.com/chrchang/plink-ng/tree/master/2.0/Python) must be installed, either with `pip install Pgenlib` (included in `install/tensorqtl_env.yml`) or from source:

```bash
git clone git@github.com:chrchang/plink-ng.git
cd plink-ng/2.0/Python/
python3 setup.py build_ext
python3 setup.py install
```

### Running default mode

#### From Salmon output

`scripts/run_hapmixqtl_from_salmon.py` is the supported route from quantification to results. It pairs the haplotype transcripts, aggregates to genes, runs edgeR on the point-estimate totals for the eQTL gene filter and effective library sizes, builds the default-mode inputs with `prepare_default_inputs`, reads the phased VCF, applies a reference-mapping-bias gate, and runs `hapmixqtl.map_cis` in default mode. It offers no other hapmixQTL configuration.

```bash
python3 scripts/run_hapmixqtl_from_salmon.py --selftest     # fabricated inputs, the whole path

python3 scripts/run_hapmixqtl_from_salmon.py \
    --vcf phased.vcf.gz \
    --manifest samples.tsv \
    --tx2gene tx2gene.tsv \
    --gene-pos genes.tsv \
    --covariates cov/covariates.tsv \
    --out results/
```

`samples.tsv` maps each sample id to its Salmon output directory, `tx2gene.tsv` maps base transcript ids (without the haplotype suffix) to gene ids, and `--hap-suffix` names the two haplotype suffixes (default `_hapA,_hapB`; the BrainVar transcriptome uses `_L,_R`). `--covariates` is required and must come from `scripts/build_covariates.py --point-estimates`, which records the expression PCs' unit, gene set and library sizes; the runner refuses covariates whose build record is missing or does not match this run unless `--covariates-unverified` is passed. Genotype PCs listed in `genotype_covariates.txt` beside the covariate file stay with the genotypes under permutation. The runner refuses to proceed when the reference-bias gate detects mapping bias, because hapmixQTL does not model it (`--force` overrides). It writes `hapmixqtl_cis.tsv.gz`, the per-gene results, and `eval_bundle.json`, aggregate statistics only, designed to be shared without individual-level data. `--str-vcf` and `--multiallelic` add short tandem repeats and multi-allelic sites as extra rows of the scan. The [deployment runbook](docs/brainvar_deploy_runbook.md) covers the full BrainVar run, including covariates and sample pairing.

#### From prepared matrices in Python

Counts are arrays with shape `[features, samples]`; the haplotype Gibbs draws have shape `[features, samples, draws]`.

```python
import pandas as pd
from tensorqtl import hapmixqtl

A, T, Va, Vt = hapmixqtl.prepare_default_inputs(
    pL, pR, pT, effective_library_sizes, yL, yR)
A_df, T_df, Va_df, Vt_df = [
    pd.DataFrame(x, index=phenotype_ids, columns=sample_ids)
    for x in (A, T, Va, Vt)
]

# nominal associations for every variant-gene pair, one parquet file per chromosome
hapmixqtl.map_nominal(
    genotype_df, variant_df, A_df, T_df, Va_df, Vt_df,
    phenotype_pos_df, xL_df=xL_df, xR_df=xR_df,
    prefix=prefix, covariates_df=covariates_df,
    genotype_covariates_df=genotype_covariates_df, output_dir='.')

# one row per gene with the empirical permutation p-values
res_df = hapmixqtl.map_cis(
    genotype_df, variant_df, A_df, T_df, Va_df, Vt_df,
    phenotype_pos_df, xL_df=xL_df, xR_df=xR_df,
    covariates_df=covariates_df,
    genotype_covariates_df=genotype_covariates_df, seed=seed)
```

`map_nominal` and `map_cis` default to the default-mode settings and preserve the matrices they are given; they do not normalize counts. They refuse inputs whose four matrices differ in rows or columns, contain missing or non-finite values, or carry negative variances, phase frames (`xL_df`, `xR_df`: ALT-allele indicators per haplotype, variants × samples) whose rows or columns differ from the genotypes, a single phase frame without the other, and a covariate design that is not of full column rank. `covariates_df` holds the RNA-tied covariates and `genotype_covariates_df` the genotype PCs. Without phase frames the allelic channel carries no information and the analysis is total-only.

`map_cis` also reports `loo_donor` and `loo_pval_nominal`: the donor whose exclusion from both channels moves the lead's combined |t| furthest toward zero, and the lead's nominal p-value without that donor. This is a diagnostic, never a filter: the lead is held fixed and `pval_perm` is not recomputed. `pval_cis_trans` tests whether the two channels' slopes disagree and is also a diagnostic.

`summaries_from_point_estimates` and `compute_summaries_from_gibbs` build historical inputs (`log2(CPM + 1)` totals and natural-log draw means, respectively) and are kept only to reproduce dated results.

#### From BED files on the command line

The `hapmixqtl_nominal` and `hapmixqtl` modes read prepared matrices in the phenotype BED layout, with identical gene and sample order.

| Argument | Description |
| --- | --- |
| `--hap_A` | Allelic contrast `A` (required) |
| `--hap_T` | Half-read total expression `T` (required) |
| `--hap_Va` | Allelic working variance, with excluded donor-gene pairs set to zero (required) |
| `--hap_Vt` | Total working variance; omitted, every donor gets 1 |
| `--hap_Cat` | Covariance matrix; loaded for inspection only and not used by the method |
| `--phase_xL`, `--phase_xR` | ALT-allele indicators per haplotype, variants × samples, in genotype sample order |
| `--ase_covariates` | `none` (default): the allelic channel is fitted through the origin; `shared` applies `--covariates` to both channels |
| `--perm_scheme` | `records_signflip` (default); `records` (no label swap) and `residuals` are retained alternatives |
| `--se_mode` | `fitted` (default); `robust` selects the HC1 sandwich standard error (heteroskedasticity-consistent, scaled by n/(n − p)) and is accepted by `hapmixqtl_nominal` only |
| `--tau_refit` | Accepted for compatibility; has no effect in default mode |

```bash
python3 -m tensorqtl ${plink_prefix_path} ${expression_bed} ${prefix} \
    --mode hapmixqtl_nominal \
    --hap_A ${A_bed} --hap_T ${T_bed} --hap_Va ${Va_bed} \
    --phase_xL ${xL_file} --phase_xR ${xR_file} \
    --covariates ${covariates_file}

python3 -m tensorqtl ${plink_prefix_path} ${expression_bed} ${prefix} \
    --mode hapmixqtl \
    --hap_A ${A_bed} --hap_T ${T_bed} --hap_Va ${Va_bed} \
    --phase_xL ${xL_file} --phase_xR ${xR_file} \
    --covariates ${covariates_file}
```

`hapmixqtl_nominal` writes `${prefix}.hapmixqtl_pairs.${chr}.parquet`; `hapmixqtl` writes `${prefix}.hapmixqtl.txt.gz`. The positional `${expression_bed}` argument is required by the parser and ignored in these modes. BED files carry no raw counts or library sizes, so the command line cannot check that `T` is the half-read transform or that the allelic admission rule was applied: supply matrices made by `prepare_default_inputs`. The command line has no option for genotype-tied covariates, so every `--covariates` column moves with the donor record under permutation; to keep the genotype PCs with the genotypes, use the Python API or the Salmon runner.

### Running mixQTL mode

mixQTL mode is a Python API without a command-line mode. `inputs_from_point_estimates(pL, pR, pT)` checks the point-estimate counts and refuses a draws axis; `mixqtl_scan` runs one gene's nominal pass and `mixqtl_permutation_scan` its permutation pass, both at the published cutoffs unless others are passed (`PACKAGE_DEFAULT_CUTOFFS` holds the R function's own defaults).

```python
from tensorqtl import mixqtl_replication as mx

y1, y2, ytotal = mx.inputs_from_point_estimates(pL, pR, pT)          # one gene: [1, samples] each
out = mx.mixqtl_scan(y1[0], y2[0], ytotal[0], effective_library_sizes,
                     h1, h2, covariates=rna_covariates,
                     genotype_covariates=genotype_pcs)              # h1, h2: [samples, variants]
```

`scripts/compare_mixqtl_replication.py` is the driver that ran mixQTL mode against hapmixQTL on the BrainVar calibration genes. It reads the BrainVar deployment directory at a fixed path and takes its output directory from `MIXQTL_OUT` and its number of null permutations from `NP`; it is a record of that comparison rather than a general tool.

### Self-test

```bash
pytest tests/test_hapmixqtl.py tests/test_hapmixqtl_calibration.py tests/test_hapmixqtl_perm_scheme.py \
       tests/test_hapmixqtl_point_estimates.py tests/test_hapmixqtl_allelic_df.py tests/test_hapmixqtl_meier.py \
       tests/test_half_read_default.py tests/test_half_read_runner.py tests/test_fitted_variance_quarantine.py \
       tests/fitted_variance/ tests/test_cli.py tests/test_mixqtl_replication.py -q
python3 scripts/run_hapmixqtl_from_salmon.py --selftest
```

The first command runs the 249 tests of the method's surface; the second exercises the Salmon runner end to end on fabricated inputs and needs R with edgeR. A bare `pytest tests/` also runs the general tensorQTL tests, four files of which carry pre-existing failures that do not involve hapmixQTL; [tests/README.md](tests/README.md) describes every test file, which tests skip without a GPU, and those failures.

### Known limits

- **Fine-mapping is not supported in default mode.** `map_susie` has no per-channel residual scale and refuses `tau_mode='zero'`; its credible sets and posterior inclusion probabilities were never validated, and there is no fine-mapping command-line mode for hapmixQTL.
- **The second pass for short tandem repeats and multi-allelic sites is not supported in default mode.** `map_str_curvature` and `map_multiallelic` refuse `tau_mode='zero'`; in the Salmon runner `--str-vcf` and `--multiallelic` only add rows to the `map_cis` scan.
- **Low-information `Va` is shaped by Salmon's Gibbs prior.** Results use stock Salmon 1.10.3 draws, which are sampled under a prior of one pseudocount per active transcript. For donor-gene pairs with few haplotype-informative reads (roughly 3 to 30), the haplotype split is then partly set by the prior, and on one donor's split-half check `Va` understated the between-half error 1.5- to 3.2-fold. A Salmon modification that divides the prior over gene × haplotype groups (`--gibbsPriorGroups`) has been built and checked on that donor; re-quantifying the cohort with it is prepared and on hold.
- **chr14, chr15 and chr22 are excluded in the BrainVar deployment**, because every copy of the phased genotypes stops within the first few megabases of those contigs; analyses there cover 19 autosomes.
- **Use the empirical permutation p-value for detection.** The nominal p-value was measured, on an earlier version of the pipeline, to be anticonservative under Gibbs weighting, because within a gene the high-weight records have larger whitened residuals than the model allows; the default keeps Gibbs weighting in the allelic channel ([docs/pipeline_rules.md](docs/pipeline_rules.md)). The Beta approximation behind `pval_beta` is conservative in the far tail, in the opposite direction; the two must not be netted.
- **A single donor record can carry a gene-level call.** `loo_donor` and `loo_pval_nominal` make such cases visible; they do not prevent them.
- **There is no per-donor allelic read floor by default.** mixQTL's deployments applied one (at least 50 reads on each haplotype in the GTEx v8 driver that mixQTL mode reproduces); the Salmon runner's `--asc-cutoff` supplies one for a matched comparison.
- **Reference mapping bias is not modelled.** The Salmon runner's gate refuses biased data rather than correcting it.

The current list of what is validated, open and on hold is in [docs/CURRENT_SCIENTIFIC_STATE.md](docs/CURRENT_SCIENTIFIC_STATE.md).

### Documentation

| Question | Document |
| --- | --- |
| What is the statistic, exactly, and how is it derived? | [docs/hapmixqtl_methods.md](docs/hapmixqtl_methods.md) |
| What do the output columns mean? | [docs/outputs.md](docs/outputs.md) |
| How is the BrainVar deployment run end to end? | [docs/brainvar_deploy_runbook.md](docs/brainvar_deploy_runbook.md) |
| Which rules govern values, units, the gene filter and the permutation? | [docs/pipeline_rules.md](docs/pipeline_rules.md) |
| What is implemented, validated, open and on hold? | [docs/CURRENT_SCIENTIFIC_STATE.md](docs/CURRENT_SCIENTIFIC_STATE.md) |
| What was measured about calibration, including withdrawn claims? | [docs/ase_validation.md](docs/ase_validation.md) |
| How are benchmark datasets with known effects made and scored? | [benchmark/plasmode/README.md](benchmark/plasmode/README.md) |
| How are the half-read benchmark tables regenerated from recorded inputs? | [docs/half_read_analysis.md](docs/half_read_analysis.md) |
| What does each test file cover? | [tests/README.md](tests/README.md) |

The plasmode benchmark in `benchmark/plasmode/` builds datasets with known *cis* effects from the BrainVar cohort's own Salmon quantification and scores hapmixQTL weightings, mixQTL mode, total-only tensorQTL, RASQUAL and TReCASE on them. Its delivered reports were made with earlier expression PCs (`log2(CPM + 1)`) and an earlier set of weighting arms; they are records of that configuration, not a run of the shipped default. Other `scripts/*.py` files are dated analyses kept as records; several are marked not runnable, with the reason. `docs/simulation_benchmark_spec.md` is a superseded design.

---

### Upstream tensorQTL

Everything below is upstream tensorQTL documentation. These modes are present in this fork and run upstream's code, apart from a fix in `susie.py` that affects only calls without centering or scaling. The command line also lists `nbqtl-score`, a negative-binomial score-test mode added to this fork before hapmixQTL; it is not part of hapmixQTL and is not documented here.

tensorQTL is a GPU-enabled QTL mapper, achieving ~200-300 fold faster *cis*- and *trans*-QTL mapping compared to CPU-based implementations.

If you use tensorQTL in your research, please cite the following paper:
[Taylor-Weiner, Aguet, et al., *Genome Biol.*, 2019](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-019-1836-7).</br>
Empirical beta-approximated p-values are computed as described in [Ongen et al., *Bioinformatics*, 2016](https://academic.oup.com/bioinformatics/article/32/10/1479/1742545).

#### Requirements

tensorQTL requires an environment configured with a GPU for optimal performance, but can also be run on a CPU. Instructions for setting up a virtual machine on Google Cloud Platform are provided [here](install/INSTALL.md).

#### Input formats
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

#### Examples
For examples illustrating *cis*- and *trans*-QTL mapping, please see [tensorqtl_examples.ipynb](example/tensorqtl_examples.ipynb).

#### Running tensorQTL
This section describes how to run the different modes of tensorQTL, both from the command line and within Python.
For a full list of options, run
```
python3 -m tensorqtl --help
```

##### Loading input files
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

##### *cis*-QTL mapping: permutations
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

##### *cis*-QTL mapping: summary statistics for all variant-phenotype pairs
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
##### *cis*-QTL mapping: conditionally independent QTLs
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

##### *cis*-QTL mapping: interactions
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

##### *trans*-QTL mapping
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

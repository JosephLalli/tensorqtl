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

#### SuSiE 2.0 credible-set options

The Python `susie.susie` interface supports two optional credible-set
refinements from susieR 2.0. `median_abs_corr` retains a credible set when its
median absolute correlation passes the threshold, even if one weakly correlated
member fails `min_abs_corr`. `cs_extension_corr` adds near-perfect proxies to a
credible set before purity is calculated. Both options default to `None`, so
existing analyses are unchanged.

#### SuSiE 2.0 slot activity

The individual-data `susie.susie` interface can estimate whether each
single-effect slot is active instead of treating all `L` slots as present.
Pass either `susie.slot_prior_betabinom()` for the collapsed Beta-Binomial
prior or `susie.slot_prior_poisson(C, nu=8)` for the Gamma-Poisson/Poisson
prior. The result reports the per-slot posterior activity probabilities in
`c_hat` and their sum in `C_hat`; fitted values, sparse effects, and PIPs all
include these weights. Both constructors accept `c_hat_init` for a warm start,
and the Poisson prior supports `update_schedule="sequential"` or `"batch"`.
Leaving `slot_prior=None` preserves the ordinary SuSiE fit.

#### Optional ordinary-SuSiE CUDA graph

Repeated ordinary-SuSiE fits with the same `(n, p, L, dtype)` and sweep options
can capture the complete ordered IBSS sweep once and replay it with:
```
result = susie.susie(X, y, compile_ibss=True)
```
The opt-in path uses CUDA graph capture rather than Inductor arithmetic fusion,
so it executes the same ordered eager kernels and preserves their numerical
outputs. It currently requires CUDA, inference-only inputs, no slot prior, and
`estimate_prior_method="EM"` or `"none"`. The first fit pays a shape-specific
capture cost; use the default `compile_ibss=False` for one-off or differently
shaped loci. The process retains one captured signature and falls back to eager
execution, with a warning, if a later call requests a different signature.
This bounded cache is why the option is exposed on `susie.susie`, not
heterogeneous `susie.map` runs.

#### Opt-in gene-batched ordinary SuSiE

Independent ordinary-SuSiE fits can share tensor operations across genes with
`susie.susie_batched`:
```
results = susie.susie_batched(
    [X_gene1, X_gene2, X_gene3],
    [y_gene1, y_gene2, y_gene3],
    L=10,
    batch_size=32,
)
```
Each design is `samples x variants`; all genes must have the same samples but
may have different numbers of variants. The solver sorts genes by window size,
packs each bucket internally as contiguous `genes x variants x samples`,
masks padded variants and unavailable effect slots, and restores the original
gene order in the returned list. The variant-major layout matches genotype
row storage and keeps the fixed sample dimension contiguous. Thus hundreds of
distinct cis-window sizes do not require hundreds of compiled graphs.

Loaders that already own a padded variant-major workspace can bypass list
packing and per-gene device transfers:
```
results = susie.susie_batched_packed(
    X_variant_major,  # float32 [genes, variant_capacity, samples]
    y,                # float32 [genes, samples]
    variant_counts,   # number of leading real variants per gene
    L=10,
)
```
`X_variant_major` must be contiguous, reside on the same device as `y`, and
contain zero padding after each gene's declared variant count. The list API
coalesces CPU inputs in pinned memory before one asynchronous CUDA transfer;
the packed API exists so production pipelines can reuse their own staging
buffers and avoid that packing step entirely.

The `L` effect updates remain ordered exactly as in IBSS; only the independent
gene dimension is evaluated in parallel with batched matrix multiplication.
The current opt-in path supports float32 ordinary SuSiE with ELBO convergence
and the `EM` or fixed prior-variance methods. It does not yet support slot
priors, SuSiE-inf, SuSiE-ash, the scalar `optim` prior-variance method, or
`null_weight`. Credible sets are computed independently after each fit is
unpacked.

Gene batching does not replace the existing scalar `susie.susie` or
`susie.map` paths; both remain the defaults. Batched GEMM can use a different
floating-point reduction order, so the new solver is validated for
numerical/model equivalence rather than bitwise equality and is not yet the
default map backend.

#### SuSiE-ash fine-mapping

SuSiE-ash adds a dense Mr.ASH adaptive-shrinkage background to the sparse
SuSiE effects. It is available through the Python API:
```
result = susie.susie(
    X, y, unmappable_effects='ash',
    estimate_residual_variance=True,
)
```
The result additionally contains `theta`, `theta_raw`, `ash_pi`, `tau2`, the
slot-activity estimates `c_hat` and `C_hat`, and the persistent ash masking
state. If no `slot_prior` is supplied, the ash mode uses
`slot_prior_betabinom()` to distinguish sparse slots from the dense
background.

The integration implements the pinned susieR 2.0 diffuse, uncertain, and
confident states, including collision and oscillation handling, delayed
exposure, second chances, c_hat-weighted confident-effect subtraction, and the
final unmasked Mr.ASH pass after convergence. Mr.ASH refits currently run on
the CPU. The returned raw-scale sparse and dense effects, intercept, fitted
values, tau2, and residual variance describe the same final predictor.

##### Model-consistency deviations from the pinned implementation

The state-transition policy is ported from susieR commit
`dd9d9ce4693573e9dcd1ec8b3df94b63bac467d2`, but four individual-data details
intentionally differ from its literal bookkeeping. These changes are designed
to reproduce the intended SuSiE-ash result—a sparse posterior plus a dense
background that describe one predictor—when the pinned behavior would combine
incompatible scales or fit states.

1. **Mr.ASH uses the same standardized design as sparse SuSiE.** The pinned
   individual-data helper passes raw `X` to Mr.ASH, while sparse SER updates use
   the centering and scaling attributes of `X`. Adding coefficients from those
   two designs mixes units and can make the result depend on an arbitrary
   shift or positive rescaling of a raw column. TensorQTL materializes the
   standardized design for Mr.ASH and converts both sparse and dense effects
   back to raw-variable units at finalization. This makes mathematically
   equivalent raw encodings reproduce the same PIPs, `theta`, variance
   components, and fitted values. The invariant is exercised by
   `test_ash_is_invariant_to_raw_column_affine_transform` and the raw-design
   reconstruction tests.

2. **The final residual subtracts the `c_hat`-weighted sparse posterior mean.**
   With slot activity, the reported sparse contribution is
   `sum_l c_hat[l] * alpha[l] * mu[l]`. The pinned final pass instead subtracts
   `sum_l alpha[l] * mu[l]`, as though every slot were certainly active.
   TensorQTL uses the weighted mean so the residual refit by Mr.ASH is the
   residual from the same sparse predictor used in PIPs, fitted values, and
   reported effects. This prevents the dense background from compensating for
   the inactive fraction of a sparse slot. The weighted fitted/residual
   identities and final unmasked pass are covered by the slot-prior and ash
   oracle tests.

3. **The residual variance from the final Mr.ASH pass is retained.** The pinned
   final pass uses its new `sigma2` when computing `tau2`, but leaves the model's
   reported `sigma2` at its pre-pass value. TensorQTL stores the final value so
   `theta`, `ash_pi`, `tau2`, and `sigma2` describe the same optimizer result,
   with `tau2 = sigma2 * sum_k ash_pi[k] * sa2[k]`. The pinned scalar difference
   is recorded explicitly in `susier_ash_state_reference.json`; the remaining
   end-to-end outputs match its standardized individual-data oracle within the
   declared numerical tolerances.

4. **A binding residual-variance upper bound causes a constrained refit.**
   Clipping only `sigma2` after optimization leaves `theta`, `ash_pi`, and
   `tau2` fitted under the unconstrained variance. TensorQTL instead fixes
   `sigma2` at the bound and refits the coefficients and mixture weights. This
   reproduces a coherent constrained SuSiE-ash result rather than relabeling an
   unconstrained fit. `test_finite_sigma2_bound_returns_one_consistent_ash_fit`
   verifies the bound and the resulting variance-component identity.

These corrections support a **model-consistent individual-data port** claim,
not a bit-for-bit reproduction of every value or API exposed by susieR.

##### Scope and implementation limits

- **Masking protects the retained background; it is not hard coordinate
  exclusion.** Protected coefficients are zeroed before and after an in-loop
  Mr.ASH refit, matching the pinned policy, but the coordinate solver still
  visits those variants internally. Therefore they can affect intermediate
  residual, mixture-weight, and variance updates even though their returned
  dense coefficients are zero.
- **The parity claim is individual-data only.** TensorQTL does not yet expose
  corresponding sufficient-statistics/RSS implementations for SuSiE-ash or all
  other SuSiE derivatives. The archived `ash_filter_archived` policy and
  unsupported fixed-variance/option combinations are also outside this claim.
- **Oracle coverage targets the supported integration.** Retained susieR
  fixtures exactly exercise the diffuse, uncertain, and confident states,
  collisions, oscillation reversal, delayed exposure, second chance,
  `c_hat`-weighted subtraction, and a representative standardized
  end-to-end fit. They do not directly enumerate every standalone Mr.ASH
  mixture grid, initialization, or solver option.
- **Mr.ASH refits remain CPU-based.** Sparse SuSiE and the masking policy can
  run on a PyTorch GPU, but the Mr.ASH Gauss-Seidel coordinate sweep is a
  sequential NumPy implementation. CPU/GPU agreement is tested, but
  SuSiE-ash should not be described as fully GPU-accelerated.

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

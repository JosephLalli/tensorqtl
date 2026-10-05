# Preparing inputs for hapmixQTL

Run the commands below from the repository root. All paths are relative example
paths. The identifiers and values in the table examples are invented; the
three-row tables illustrate the format, not a sufficient analysis cohort.

## Before starting

Quantify each sample against its own personalized diploid transcriptome built
from phased genotypes, with Salmon Gibbs draws, for example
`--numGibbsSamples 200`. This guide starts from completed quantifications;
constructing personalized transcriptomes and running Salmon are upstream steps.

Keep the transcript haplotypes in the same order as the phased VCF alleles:
the first suffix corresponds to the first GT allele, the second to the second
GT allele. Matching suffix text alone does not establish correct phase.
Use the same reference assembly and chromosome names for genotypes and gene
positions. Use the same transcript identifiers for Salmon and the gene map.

Each Salmon output directory must contain:

```text
quant.sf
aux_info/meta_info.json
aux_info/bootstrap/names.tsv.gz
aux_info/bootstrap/bootstraps.gz
```

`bootstraps` is Salmon's file name for its inferential draws; this workflow uses
Gibbs draws. Retain `aux_info` when copying quantifications. All samples need
the same number of draws. The default haplotype suffix pair is `_hapA,_hapB`;
pass your actual pair with `--hap-suffix` to both preparation and mapping.

## Input tables

### Salmon manifest: `inputs/samples.tsv`

Two tab-separated columns, **no header**: VCF sample ID, Salmon output directory.
Sample IDs must be unique. Relative directory paths resolve from the working
directory where the commands run, not from the manifest's directory.

```text
S001	inputs/salmon/S001
S002	inputs/salmon/S002
S003	inputs/salmon/S003
```

### Transcript map: `inputs/tx2gene.tsv`

Two tab-separated columns, **no header**: base transcript ID, gene ID.
Remove only the haplotype suffix; preserve transcript versions unless the
Salmon names also omit them. Base transcript IDs must be unique.

```text
TX001	GENE001
TX002	GENE001
TX003	GENE002
```

For this example, Salmon transcript names are `TX001_hapA`, `TX001_hapB`, etc.
The total channel sums all mapped transcripts, including an unpaired transcript
copy; the allelic channel uses transcripts for which both haplotype rows exist.

### Gene positions: `inputs/gene_pos.tsv`

Five tab-separated columns, **no header**, in this exact order:
`gene_id`, `chromosome`, `TSS`, `gene_start`, `gene_end`.
Coordinates are **1-based inclusive**. TSS is strand-aware: gene start on the
positive strand, gene end on the negative strand. Gene IDs must be unique and
match `tx2gene.tsv`. Chromosome names must match the VCF, including any `chr`
prefix. Gene bodies are used by the reference-bias diagnostic.

```text
GENE001	chr1	1000	1000	2000
GENE002	chr2	7000	6000	7000
```

The annotation converter writes `genes.tsv` with TSS in its **fifth** column.
Reorder it before passing it to the mapper:

```bash
python scripts/gtf_to_tables.py --gtf inputs/annotation.gtf.gz --out annotation
mkdir -p inputs
cp annotation/tx2gene.tsv inputs/tx2gene.tsv
awk -v OFS='\t' '{print $1, $2, $5, $3, $4}' annotation/genes.tsv > inputs/gene_pos.tsv
```

Only use the converter's `--strip-version` option if Salmon transcript names
also omit versions. Its optional gene-type filter is separate from edgeR's
expression filter.

### Phased genotypes: `inputs/phased.vcf.gz`

Supply a text VCF or gzip-compressed VCF with a `GT` field and phased diploid
calls such as `0|1`. A binary BCF is not accepted by this recipe. Every manifest
sample must occur in the VCF header; extra VCF samples can remain and are
excluded. The default association scan uses biallelic SNPs. Phase must be
consistent with the sample's personalized transcriptome.

### Optional sample covariates: `inputs/sample_covariates.tsv`

This table **has a header** and is samples by covariates. The first column is
the sample ID; the remaining columns must be finite numeric values. Include
exactly the manifest's sample IDs, each once; row order can differ because the
preparation tool aligns by ID. Encode categorical variables numerically with
one level omitted. Do not supply an intercept or constant columns.

```text
sample_id	covariate_1
S003	0.4
S001	-0.3
S002	0.1
```

Use distinct column names; the preparation tool creates `geno_pc*` and
`expr_pc*` columns. These supplied covariates are RNA-tied under permutation.
Genotype PCs created from the VCF are kept tied to genotypes.

## Prepare normalization and covariates

```bash
python scripts/prepare_hapmixqtl_inputs.py \
    --manifest inputs/samples.tsv \
    --tx2gene inputs/tx2gene.tsv \
    --vcf inputs/phased.vcf.gz \
    --sample-covariates inputs/sample_covariates.tsv \
    --hap-suffix _hapA,_hapB \
    --out prepared
```

Omit `--sample-covariates` if there are no supplied covariates. The tool adds
three genotype PCs and ten expression PCs by default. Set `--n-geno-pc` and
`--n-expr-pc` deliberately for your sample size and design; the tool rejects
infeasible counts and a rank-deficient design. `--gene-restrict` optionally
takes a text file with one eligible gene ID per line; edgeR's expression filter
still applies. There is no cohort-specific coding or autosome restriction.

The tool writes:

| File | Purpose |
| --- | --- |
| `prepared/point_estimates/totals_all.tsv.gz` | All mapped gene totals from point estimates. |
| `prepared/point_estimates/edger/edger_samples.tsv` | edgeR library sizes, TMM factors and effective sizes. |
| `prepared/point_estimates/edger/calibration_genes.txt` | Genes retained after expression filtering and any explicit restriction. |
| `prepared/covariates.tsv` | Supplied covariates and computed genotype/expression PCs. |
| `prepared/genotype_covariates.txt` | Names of genotype-tied PC columns. |
| `prepared/covariate_build.json` | The expression-PC unit, normalization source and covariate split. |

Expression PCs are computed on the half-read log-CPM of the retained genes,
after residualizing against supplied covariates and genotype PCs. They use the
same effective library sizes as mapping. Preserve the preparation directory
and use the same working directory for subsequent mapping; the provenance
records a relative preparation path.

## Map cis associations

```bash
python scripts/run_hapmixqtl_from_salmon.py \
    --manifest inputs/samples.tsv --tx2gene inputs/tx2gene.tsv \
    --vcf inputs/phased.vcf.gz --gene-pos inputs/gene_pos.tsv \
    --hap-suffix _hapA,_hapB \
    --covariates prepared/covariates.tsv \
    --edger-dir prepared/point_estimates/edger \
    --out results
```

The runner checks that the gene set, effective library sizes and expression-PC
unit agree. It also checks reference mapping bias before mapping. Investigate a
failed check and rebuild the affected inputs; the walkthrough does not bypass
these checks. By default it uses a 1 Mb window around each TSS and 10,000 gene
permutations. CPU runs are supported; GPUs accelerate mapping.

Results are `results/hapmixqtl_cis.tsv.gz` and `results/eval_bundle.json`.
Gene-level detection uses `pval_perm` or `pval_beta`; multiple-testing correction
across genes is a separate step. See [outputs.md](outputs.md).

Keep cohort manifests, counts, covariates, genotypes, logs and association
results out of version control. These files can contain sample identifiers and
individual-level information. The toy generator in the repository creates only
fabricated inputs and is suitable for demonstrating this workflow publicly.

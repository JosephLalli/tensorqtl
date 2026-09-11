# Running the hapmixQTL / RASQUAL comparison on BrainVar

The end product is `deploy/deploy_comparison.json` (and its `.md` rendering):
the two methods run head-to-head on the same BrainVar genes, each given the
input it was designed for. This document is the operating procedure. The
method rationale is in `docs/ase_validation.md`; the port of the work onto a
local server is in `docs/LOCAL_HANDOFF.md`.

## Where this runs, and why that is a governance question, not a preference

On the machine that holds BrainVar. The data does not move.

BrainVar is dbGaP controlled-access (phs001900), so the relevant question is
what leaves the machine. The four scripts in the comparison chain --
`gtf_to_tables.py`, `phaser_to_matrix.py`, `make_rasqual_inputs.py`,
`compare_pipelines.py` -- contain no network access at all. Checked by grep
over the whole `scripts/` tree for `urllib`, `requests`, `http://`, `https://`,
`socket`, `curl`, `wget` and `git clone`: the only two hits in the tree are in
scripts that are not part of this chain, and neither transmits data outward.

- `build_rasqual.sh` clones `https://github.com/natsuhiko/rasqual.git`. That
  is a one-time fetch of public source code; nothing is uploaded.
- `extract_gtex_phaser.py` downloads public GTEx v8 haplotype-expression
  matrices. It is the §7d public-data path and is not used here.

So the comparison itself is inert local Python plus the RASQUAL binary.

The artifact intended to leave is `deploy_comparison.json`. Read against the
dict it is built from, it holds only scalars and counts: the design
(`n_genes`, `n_samples`, `n_tested_variants`, `n_perm`, `window`, `seed`),
wall-clock seconds per method, and per method a calibration block
(`lambda_gc_null`, nominal type-I on the null), power at matched empirical
FPR (threshold, discovered fraction and count), a replication block
(`n_discovered` and the fraction of discoveries in the known-eGene list), and
RASQUAL's `phi_hat` median and quantiles. The head-to-head block is a Spearman
correlation, a top-k overlap fraction, and the regression of one method's
log-aFC on the other's. Gene identities are used only inside those
computations -- to take a set length or a mean of membership tests -- and no
gene list, per-gene value or per-sample value is written into the JSON.

**The output directory holds more than that.** `compare_pipelines.py` also
writes `observed_hapmixqtl.tsv` and `observed_rasqual.tsv` into `--out`: full
per-gene tables with lead variants and effect sizes. Those are gene-level
summary statistics rather than individual-level data, but they are not covered
by the aggregate description above, so copy the two named files rather than
archiving the directory.

Confirm all of this against your own DUA before carrying anything off the
machine; `LOCAL_HANDOFF.md` §4 has the fuller discussion, including the
separate question of whether an AI coding assistant may run in a directory
holding individual-level data.

## Setup, once

```bash
git clone <repo> && cd tensorqtl
git checkout claude/hapmixqtl-gibbs-uncertainty-IQ6Za
# dependency list and the pinned versions are in docs/LOCAL_HANDOFF.md
pip install numpy scipy pandas torch pandas_plink h5py qtl pysam pytest
pip install -e . --no-deps
./scripts/build_rasqual.sh              # needs libgsl-dev liblapack-dev
```

`build_rasqual.sh` leaves the binary at **`rasqual_src/src/rasqual`** (it
builds in `$DEST/src`, where `$DEST` defaults to `rasqual_src`). Pass a
different `$DEST` as its first argument if you want it elsewhere.

Run the self-tests before touching real data. They exercise the parsing and
the joins on fabricated inputs, so they catch an environment problem in
seconds rather than after hours of quantification:

```bash
python3 scripts/gtf_to_tables.py --selftest
python3 scripts/diploid_tx2gene.py --selftest
python3 scripts/phaser_to_matrix.py --selftest
python3 scripts/make_rasqual_inputs.py --selftest
RASQUAL_BIN=rasqual_src/src/rasqual python3 scripts/compare_pipelines.py --selftest
```

`compare_pipelines.py --selftest` **exits 1 with `set RASQUAL_BIN to the built
rasqual binary`** if that variable is unset or points at a missing file. That
is the expected message before RASQUAL is built, not a failure of the script.

## The diploid-quantification prerequisite

hapmixQTL's arm needs Salmon run against a **personalized diploid
transcriptome** with `--numGibbsSamples 200`. Each transcript must appear
twice, once per haplotype, distinguished by a suffix pair (`--hap-suffix`,
default `_hapA,_hapB`). A standard reference transcriptome carries no allelic
information, so there is nothing for hapmixQTL to read and the comparison
degenerates to RASQUAL against nothing.

`compare_pipelines.py` refuses rather than producing a meaningless number.
`load_counts` raises `no haplotype-paired transcripts in <dir> using suffixes
(...)` and says the reference was probably not diploid. Treat that message as
this step failing, not as a bug.

Check this before anything else: building the diploid index (g2gtools or
vcf2diploid) and re-quantifying is a separate job on the scale of the analysis
itself.

### State of this prerequisite on the current machine

Surveyed 2026-09-10 on the server at `/mnt/ssd/lalli`. **The personalized
diploid quantifications exist.** They are under

```
/mnt/data/lalli/nf_stage/reference_comparison_results_RNA/
    Personalized_T2T_calls_NCBI110/star_salmon/<sample>/
```

with **37 samples** carrying a bootstrap payload (of 45 sample directories).
Read back through `read_salmon_bootstraps`, each holds roughly 215,000
transcripts of which about 107,500 are haplotype-paired.

Four things about this data differ from the defaults and have to be passed or
accounted for.

**The haplotype suffix is `_L`/`_R`,** not `_hapA`/`_hapB`. Transcripts are
named `NR_109817.1_L` and `NR_109817.1_R`. Pass `--hap-suffix _L,_R` to
`compare_pipelines.py`; with the default, `load_counts` pairs nothing and
raises the standard-reference error at data that is in fact diploid.

**Use `star_salmon/`, not the `salmon/` sibling.** Both directories exist under
that arm. `salmon/` has been stripped -- 1 of 42 sample directories still has
a bootstrap payload and only 1 still has a `quant.sf`. `star_salmon/` is the
live output, which matches the run's own parameters (`aligner = star_salmon`,
`skip_pseudo_alignment = true`).

**The draws are 30 bootstrap samples, not 200 Gibbs samples.**
`meta_info.json` records `samp_type = bootstrap`, `num_bootstraps = 30`. The
code path is the same -- Salmon writes both to the same place and hapmixQTL
propagates either as inferential replicates -- but 30 draws estimate the
per-gene inferential variance less precisely than 200, and that variance is
exactly what hapmixQTL's expression-uncertainty weighting consumes. Whether 30
is enough for the weighting to behave is not established here and should be
checked before quoting a result; re-quantifying with `--numGibbsSamples 200`
is the alternative, and the recipe for it is recorded in
`.../bv2/personalized_T2T_NCBI110_star_alignment/pipeline_info/params_2025-05-22_14-28-55.json`
(`num_gibbs_samples = 200`).

**Transcript sets differ between samples,** because a personalized
transcriptome is built per sample from that sample's own variants. Two real
samples shared 215,047 transcripts with a handful unique to each. This is
handled -- the gene set is the union over all samples -- but it is the reason
the manifest order used to matter, and it means the per-sample matrices are
sparse at sample-specific genes rather than complete.

The reference for this arm is **T2T**, so the GTF and the VCF used downstream
must be T2T as well. Its transcripts are RefSeq accessions
(`NR_109817.1`, `XM_047444567.1`) and, in an NCBI GTF, genes are named by
symbol -- so an eGene list keyed on `ENSG...` will not join to it.

### tx2gene for this data comes from a different file than genes.tsv

The per-sample diploid annotations are in `Personalized_T2T_calls_NCBI110/convert/`
as `<sample>-diploid_specific.gtf`, 42 of them. They carry the haplotype suffix
on **both** identifiers:

```
gene_id "SEPTIN14P6_L"; transcript_id "NR_109817.1_L"; gene "SEPTIN14P6";
```

`load_counts` pairs the two haplotype transcripts first and then looks the gene
up by the **unsuffixed** base, `NR_109817.1`. A tx2gene keyed on
`NR_109817.1_L` therefore matches nothing, and the run dies claiming Salmon was
run against a standard reference transcriptome -- the same misleading message,
from the opposite cause. Build it with:

```bash
python3 scripts/diploid_tx2gene.py     --gtf .../convert/HSB238-diploid_specific.gtf     --hap-suffix _L,_R --out annot/tx2gene.tsv
```

On `HSB238-diploid_specific.gtf` that turns 216,175 transcript rows into
108,088 unique pairs over 41,498 genes, and it resolves **100%** of that
sample's 107,505 haplotype-paired transcripts, collapsing them to 40,917 genes.

Take `genes.tsv` and `genes.bed` from the **reference** T2T annotation with
`gtf_to_tables.py`, not from these files. A personalized annotation has
personalized coordinates -- positions in that sample's own haplotype, not in
the reference the VCF is called against -- so its spans and TSS values do not
belong in a cis-window definition shared across samples.

Two non-personalized arms are quantified the same way and make natural
baselines: `T2T_NCBI110/star_salmon` (94 payloads) and
`GRCh38_p14_NCBI110/star_salmon` (103 payloads). Neither carries allelic
information, so neither can feed hapmixQTL -- they are the standard-reference
comparison, not an input.

## Building the annotation tables

```bash
python3 scripts/gtf_to_tables.py --gtf gencode.vXX.annotation.gtf.gz \
    --gene-type protein_coding --out annot/
```

Produces three files from one GTF:

- `genes.tsv` -- `gene_id chr start end tss`, 1-based inclusive, strand-aware
  TSS (start on `+`, end on `-`). This is `--genes`.
- `tx2gene.tsv` -- `transcript_id gene_id`, for collapsing Salmon to genes.
- `genes.bed` -- `chr start stop gene_id`, 0-based half-open and header-free,
  for phASER's `--features` when running phASER per sample.

`genes.bed` is not `genes.tsv` with the columns moved. `phaser_gene_ae.py`
parses positionally from line 1 with `int(columns[1])` and has no header
handling, so a header line is a crash; and it wants 0-based coordinates where
the GTF is 1-based, so the start is shifted by one and the span length is
preserved. The name column is the `gene_id`, because that is what
`phaser_to_matrix.py` indexes genes by and what `genes.tsv` is keyed on -- a
gene symbol there would join to nothing without erroring.

The command prints the chromosome names it saw. Check them against your VCF
now; see the naming section below.

## Running phASER per sample, keeping the VCF

Every argument below except `--write_vcf` is required by phaser.py; the flags
and their meanings are as documented for it.

```bash
phaser.py --vcf pop.vcf.gz --bam S.bam --sample S --paired_end 1 \
    --mapq 255 --baseq 10 --write_vcf 1 --o out/S
phaser_gene_ae.py --haplotypic_counts out/S.haplotypic_counts.txt \
    --features annot/genes.bed --o out/S.gene_ae.txt
```

Three things to get right.

`--mapq 255` is **STAR's** encoding for a uniquely-mapped read. phASER
documents the flag only as a minimum mapping quality, so the value is yours to
choose: under BWA or HISAT2, 255 means something else or never occurs, and a
255 filter discards nearly everything. Set it to whatever your aligner emits
for a unique alignment.

`--write_vcf` defaults to **1**, so the phased VCF is produced by a default
run and the risk is switching it off, not forgetting to switch it on. It is
not optional here: assembling the matrices reads `<prefix>.vcf[.gz]` to
overlay the read-backed phase and fails with `no phASER VCF for <sample>`
without it. Keep the file; do not clean it up between the two commands.

phASER requires the input VCF **gzipped and indexed**, and requires
chromosome names to match between the BAM and the VCF. That is the same
naming constraint the annotation tables are subject to, arriving one step
earlier -- see the naming section below.

## Assembling the matrices and overlaying read-backed phase

```bash
python3 scripts/phaser_to_matrix.py --manifest phaser.tsv \
    --vcf pop.vcf.gz --out prepped/
```

`phaser.tsv` is two columns, `sample_id <TAB> phASER output prefix`. The
prefix must resolve to `<prefix>.gene_ae.txt`, `<prefix>.allelic_counts.txt`
and `<prefix>.vcf[.gz]`.

Outputs `rephased.vcf.gz`, `allelic_counts_manifest.tsv` and `samples.txt`.
Both methods must read the same `rephased.vcf.gz`, or the comparison is
confounded by phase rather than by method.

Read `summary.json` before continuing. Its `rephase` block records
`gt_rephased` (genotypes overlaid with read evidence), `gt_flipped` (those
where phASER's read-backed phase disagreed with the population phase) and
`flip_rate`, their ratio. That rate is the switch-error rate measured in your
own data, not a literature value.

## Running the comparison

Hours of work; run it detached.

```bash
nohup python3 scripts/compare_pipelines.py \
    --vcf prepped/rephased.vcf.gz --genes annot/genes.tsv \
    --salmon salmon.tsv --tx2gene annot/tx2gene.tsv \
    --allelic-counts prepped/allelic_counts_manifest.tsv \
    --rasqual rasqual_src/src/rasqual \
    --known-egenes brain_egenes.txt \
    --hap-suffix _L,_R \
    --n-genes 300 --n-perm 10 --out deploy/ > deploy.log 2>&1 &
```

`--hap-suffix _L,_R` is required for the quantifications described above; the
default is `_hapA,_hapB` and would pair nothing.

`salmon.tsv` is `sample_id <TAB> Salmon output directory` -- the directory
holding `aux_info/bootstraps/`, not the `quant.sf` file.

`compare_pipelines.py` stages RASQUAL's inputs through `tempfile`, so it
inherits `TMPDIR`. On a host where `/tmp` is a spinning disk this becomes the
bottleneck for a run that is otherwise compute-bound; point `TMPDIR` at fast
local storage before launching.

The script reports how many genes are usable by both methods. A gene qualifies
only if it is present in both inputs and has at least one feature-SNP carrying
allelic counts inside the gene body. Fewer than five and it stops with
`too few genes usable by both methods`. That message means the inputs did not
intersect -- most often a naming mismatch, not a shortage of data.

Bring back `deploy/deploy_comparison.json` and `deploy/deploy_comparison.md`.

## The naming trap that silently zeroes everything

Two conventions have to agree across files that are built separately, and
disagreement produces empty joins rather than errors.

**Ensembl version suffixes.** IDs carry them (`ENSG00000123456.7`). Salmon
transcript names usually keep them; published eGene lists usually drop them.
A mismatch makes `tx2gene` pair zero transcripts and `--known-egenes`
replicate zero genes, with no diagnostic. Pick one convention and apply it in
all three places: either pass `--strip-version` when building the annotation
tables *and* strip the
Salmon names *and* strip the eGene list, or keep versions on all three.
`--strip-version` affects only the tables `gtf_to_tables.py` writes; nothing
downstream strips anything for you.

**Chromosome names.** `chr1` and `1` are different strings.
`gtf_to_tables.py` emits chromosome names exactly as they appear in the GTF
and prints the first few, and `compare_pipelines.py` compares them against the
VCF as strings. If they disagree, no variant falls in any gene window and the
usable-gene count collapses to zero.

There is a third form, and the server has one on disk. An NCBI RefSeq
annotation names chromosomes by accession -- running the builder over
`genome_refs/GRCh38_p14_ncbi110/GCF_000001405.40_GRCh38.p14_genomic.gtf.gz`
prints `NC_000001.11, NC_000002.12, NC_000003.12`. Those match no VCF written
against `chr1` or `1`. That GTF also names genes by symbol (`OR4F5`) rather
than by Ensembl ID, so eGene lists keyed on `ENSG...` will not join to it
either. If you want GENCODE-style identifiers, use a GENCODE GTF; if you use
this one, make the VCF and the eGene list agree with it.

Both failures surface in the comparison as `too few genes usable by both
methods`.
When you see it, check naming before concluding the data are thin.

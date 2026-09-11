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
python3 scripts/brainvar_pairing.py --selftest
python3 scripts/verify_pairing.py --selftest
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

Surveyed 2026-09-10. Every Salmon run on both mounts was enumerated -- 1,158
`meta_info.json` files -- and tabulated by sampling type:

| sampling | draws | runs |
| --- | --- | --- |
| gibbs | 200 | 359 |
| bootstrap | 30 | 371 |
| bootstrap | 100 | 12 |
| none | 0 | 416 |

**Personalized diploid quantifications with 200 Gibbs samples exist.** 229 of
the 359 are one coherent published set; the remainder are unpublished `work/`
copies of the same run.

```
/mnt/ssd/lalli/nf_stage/RNA_reference_comparison_results/reference_comparison_results/
    bv2/personalized_T2T_NCBI110_pseudoalignment/expression_results/salmon_pseudocounts/<sample>/
```

229 samples (228 distinct subjects), `samp_type = gibbs`, `num_bootstraps =
200`, ~272k-292k transcripts each, haplotype-paired `_L`/`_R` at 92k-115k pairs
per sample. Read back through `read_salmon_bootstraps` without complaint.

**But that arm has no aligned reads.** It is the pseudoalignment path: zero
BAM or CRAM files anywhere under it. phASER needs aligned reads, so on its own
this arm supports hapmixQTL and nothing else -- `compare_pipelines.py` requires
`--allelic-counts` and cannot run without RASQUAL's native input.

#### The combination that gives the full head-to-head

Take the allelic counts from a **reference-aligned** arm. That is what phASER
wants anyway: reads aligned to the reference the VCF was called against, not to
a personalized genome.

| role | source | n |
| --- | --- | --- |
| hapmixQTL input | `personalized_T2T_NCBI110_pseudoalignment/expression_results/salmon_pseudocounts` | 228 subjects, 200 Gibbs draws |
| phASER / RASQUAL input | `reference_comparison_results_RNA/T2T_NCBI110/star_salmon/*.bam` | 93 subjects, reference T2T |
| genotypes | `nf_stage/brainvar2/gatk_t2t_haplotypecaller.joint_called.phased...nostar.bcf` | 2.4G, phased, 25 contigs |

The two sample-ID schemes differ but correspond: the BAMs are `HSB<n>` and the
quantifications are `<n>_R1`. Normalising both to `<n>` gives **92 subjects in
common** (93 BAM subjects, one without a quantification). That is the usable N
for the comparison, at 200 Gibbs draws.

#### Pairing the samples: do not join by name

Three identifier systems are in play and they do not agree.

| thing | id | example |
| --- | --- | --- |
| diploid quantification | bulk RNA library | `587_R1` |
| genotypes (VCF sample) | WGS library | `589_D1` |
| RNA alignment for phASER | subject | `HSB589` |

BrainVar carries a known sample relabelling, and the two RNA runs resolved it
differently. The alignment run applied it -- the BAM named `HSB587` was built
from `HSB583_1_val_1.fq.gz`, and the 2022 relabelling map says HSB583 is
HSB587 -- while the quantification did not: `587_R1` was quantified from
`HSB587.R1_val_1.fq.gz`. For five subjects this produces a shift chain, and
**joining the quantification to the BAM by number pairs two different donors**,
silently, because every identifier involved exists.

The DNA library is the only common key, and it is what the VCF is keyed on, so
it is what the manifests must use as `sample_id`. Build the pairing with:

```bash
python3 scripts/brainvar_pairing.py \
    --metadata  <bv2>/draft_brainvar2_library_metadata_v1.4.tsv \
    --vci-dir   <arm>/vcf2vci \
    --salmon-dir <arm>/expression_results/salmon_pseudocounts \
    --bam-dir   reference_comparison_results_RNA/T2T_NCBI110/star_salmon \
    --vcf-samples <(bcftools query -l <phased.bcf>) --out cohort/
```

It writes `salmon.tsv`, `bams.tsv`, `samples.txt` and `pairing.tsv`, all keyed
on the DNA library. On the current data it pairs **92** subjects and reports
the five rows a name-join would get wrong:

```
587_D1: rna=583_R2  bam=HSB587        589_D1: rna=587_R1  bam=HSB589
590_D1: rna=589_R1  bam=HSB590        591_D1: rna=590_R1  bam=HSB591
593_D1: rna=591_R1  bam=HSB593
```

**Use metadata v1.4, not earlier.** The authoritative table is
`draft_brainvar2_library_metadata_v1.4.tsv` (SHA-256
`c70e3599...ec663ba6`, 841 rows by 29 columns), and
`METADATA_V1.4_FREEZE.md` in the brainvar2 repository records the freeze.
Against v1.3.1 it changes `matchingDNALibrary` for two usable bulk-RNA
records, `321_R2` (`321_D1` to `321_D2`) and `513_R2` (`175_D1` to `175_D2`).
Both candidate DNA libraries exist in the VCF, so an older table attaches the
wrong one with no error.

Each edge of the join is evidenced rather than assumed.

**RNA library to DNA library** comes from the run itself. Each per-sample
g2gtools VCI header carries `##STRAIN=<dna_library>`, which IS the genotype the
personalized transcriptome was built from. Across the 229 quantified libraries
the VCI agrees with v1.4 for 227. The two exceptions are exactly the two
records v1.4 changed: the quantification ran in April 2025 against the older
assignment, so `321_R2` and `513_R2` have diploid references built from
`321_D1` and `175_D1`. Their allelic quantifications encode a superseded
pairing; drop them or re-quantify. Neither is in the 92-subject cohort.

**BAM to DNA library** is the numeric rule `HSB<n>` to `<n>_D1`, checked
against the reads rather than trusted. Genotype concordance between an RNA BAM
and each candidate DNA sample, over ~2,200-4,600 informative chr1 sites at
depth 10 or more, is 0.99 for the numeric match and 0.42-0.60 for every other
donor -- a separation wide enough that the assignment is not in doubt. Verified
on the three shift-chain subjects, where the rule is likeliest to fail:
`HSB587`/`587_D1` 0.991, `HSB589`/`589_D1` 0.991, `HSB593`/`593_D1` 0.989.
Re-run it on the rest before publishing; it is the only check that does not
depend on a filename.

#### Verify the pairing against the reads before trusting a result

Every identifier here is a claim someone wrote down; the reads are not. An RNA
alignment carries its donor's genotypes at expressed sites, so the donor can be
identified directly:

```bash
python3 scripts/verify_pairing.py --pairing cohort/pairing.tsv \
    --bam-dir reference_comparison_results_RNA/T2T_NCBI110/star_salmon \
    --vcf <phased.bcf> --chrom chr1 --contig-map rename_chrs.tsv --out verify/
```

It scores each BAM against **every** donor in the pairing, not just the claimed
one, and flags any sample whose best match is not the claimed donor or whose
margin over the runner-up is thin. That is the only check here that does not
depend on a filename, and a swap does not announce itself otherwise: mispaired
allelic counts still run, still converge, and still produce QTLs.

Run it on the **whole** pairing, not a subset: the candidate pool is exactly
the donors listed in `--pairing`, and a smaller pool inflates the margin because
it is less likely to hold a close genotype match by chance. Six donors here gave
margins of 0.38-0.50 where the same samples scored against all 92 gave
0.30-0.34.

What it looks like when the pairing is right: on this data the correct donor
scores **0.986-0.994** and the best competing donor **0.60-0.75**, against a
median across donors near 0.64. The separation is the signal -- two unrelated
people agree at roughly 0.6 by chance given the allele-frequency spectrum, so a
margin above about 0.2 is unambiguous and a margin near zero means the sample
is not identified.

Eight samples were checked while this runbook was written, with no
discrepancies: the three shift-chain BAMs where the numeric rule is likeliest
to fail (`HSB587`, `HSB589`, `HSB593`), four ordinary ones (`HSB629`,
`HSB344`, `HSB429`, `HSB260`), and -- separately -- the one pairing that rests
on the metadata rather than on shared provenance.

That last one is worth understanding. For 90 of the 92 pairs the alignment and
the quantification consumed the **same FASTQ**, which can be read off the BAM's
`@PG readFilesIn` and Salmon's `cmd_info.json`; those pairs are the same donor
by construction, whatever any table says. Two are not, and one of those is
`589_D1`, where the quantification used FASTQ `HSB587` and the BAM used
`HSB473`. No alignment arm ever used `HSB587` -- both dropped it as
`duplicate_HSB589` -- so it was aligned from scratch with minimap2 against the
chr-named T2T reference and genotyped: it matches `589_D1` at **0.990** against
a runner-up of 0.747, confirming the metadata. The other, `587_D1`, is the same
library under two filename conventions (`H583_RNA_020_173_S96_L004` and
`HSB583`).

The conclusion to carry forward is that the metadata is not wrong, it is
counterintuitive: the numbers genuinely do not line up, because the
relabelling is real and v1.4 encodes it correctly. The danger is ignoring it.

#### Two things must be fixed before phASER will run

**Contig names disagree between the BAM and the VCF.** The T2T BAMs are named
by RefSeq accession (`NC_060925.1`); the phased BCF and the reference GTF use
`chr1`. phASER requires them to match and will otherwise find nothing. Both
have exactly 25 contigs and they correspond 1:1 -- matching the BAM header
lengths against `chm13v2.0_maskedY_rCRS.fasta.fai` maps all 25 with none left
over on either side, and the 25 targets are exactly the BCF's contig set.
Derive the map and apply it to whichever file you would rather rewrite:

```bash
samtools view -H <bam> | awk '/^@SQ/{for(i=1;i<=NF;i++){if($i~/^SN:/)n=substr($i,4);if($i~/^LN:/)l=substr($i,4)}print n"\t"l}' \
  > bam_ctg.tsv
# join on length against the chr-named reference index
awk 'NR==FNR{a[$2]=$1;next} ($2 in a){print $1"\t"a[$2]}' \
    chm13v2.0_maskedY_rCRS.fasta.fai bam_ctg.tsv > rename_chrs.tsv
```

Renaming the BAM header (`samtools reheader`) keeps the GTF and VCF, which
already agree on `chr`, as the reference convention.

**`tx2gene` must come from the reference annotation, not the per-sample GTFs.**
The arm's `personalized_references/*.gtf` are 229 **dangling symlinks** -- they
list but do not open. This turns out not to matter: the transcript bases under
the `_L`/`_R` suffixes are ordinary RefSeq accessions, so the reference
annotation resolves them. Running `gtf_to_tables.py` on

```
genome_refs/T2T-CHM13_v2_ncbi110/GCF_009914755.1_RS_2024_08-T2T-CHM13v2.0_genomic.UCSC_chr.with_GRCh38_rCDS_chrM.exon_ids.gtf
```

gives 58,516 genes and 183,140 transcripts on `chr`-style names, and resolves
**100%** of a sample's 115,297 haplotype-paired transcripts into 24,437 genes.
`diploid_tx2gene.py` is not needed for this arm -- it is for the case below,
where the per-sample diploid GTFs are present and the bases are not reference
accessions.

#### The fallback, if you would rather not renumber contigs

`reference_comparison_results_RNA/Personalized_T2T_calls_NCBI110/star_salmon`
has personalized quantifications **and** 34 readable BAMs in the same arm, with
per-sample diploid GTFs present in `convert/` rather than dangling. Everything
is self-consistent there, so no contig rename is needed and
`diploid_tx2gene.py` applies directly. The costs are that it is 34 samples
rather than 92, its draws are 30 bootstrap samples rather than 200 Gibbs, and
its BAMs are aligned to each sample's personalized genome rather than to the
reference -- which is the wrong input for phASER in principle, since the
coordinates are not the VCF's.

Prefer the 92-subject, 200-Gibbs combination. It is better on both axes that
matter, and the contig rename is a header rewrite, not a re-run.

Two non-personalized arms make natural baselines and are quantified the same
way: `T2T_NCBI110/star_salmon` (94 payloads) and `GRCh38_p14_NCBI110/star_salmon`
(103). Neither carries allelic information, so neither can feed hapmixQTL --
they are the standard-reference comparison, not an input.

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

### Optional, non-standard: STRs and multiallelic sites

Off by default. Standard cis-QTL mapping tests biallelic SNPs, and both
`compare_pipelines.py` and `run_hapmixqtl_from_salmon.py` do exactly that
unless you add these flags:

```bash
    --str-vcf hipstr.vcf.gz    # STRs join the tested variants as per-haplotype repeat
                               # length (log aFC per repeat unit) + a curvature second pass
    --multiallelic             # multi-ALT rows of the VCF (normally skipped) join as one
                               # split row per ALT + a categorical per-allele second pass
```

In the comparison they add a third, separately reported arm
(`hapmixQTL_nonstandard`); the RASQUAL and standard hapmixQTL arms are
computed exactly as without the flags. RASQUAL cannot test these variants, so
that arm is not a like-for-like comparison with RASQUAL. It answers "what do
the extra variant classes add to hapmixQTL". The STR VCF must be a HipSTR-style
call set on the same samples (phased against the same scaffold if you want the
ASE channel to see the STRs). See `docs/ase_validation.md` §7j.

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

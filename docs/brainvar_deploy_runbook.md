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
    --vcf prepped/rephased.vcf.gz --genes annot/genes.tsv --exons annot/exons.tsv \
    --salmon salmon.tsv --tx2gene annot/tx2gene.tsv \
    --allelic-counts prepped/allelic_counts_manifest.tsv \
    --rasqual rasqual_src/src/rasqual --rasqual-jobs 8 --rasqual-threads 8 \
    --covariates cov/covariates.tsv --cache-dir cache/ \
    --known-egenes brain_egenes.txt \
    --hap-suffix _L,_R \
    --n-genes 300 --n-perm 10 --out deploy/ > deploy.log 2>&1 &
```

`--hap-suffix _L,_R` is required for the quantifications described above; the
default is `_hapA,_hapB` and would pair nothing.

### How RASQUAL is actually run

RASQUAL is a C program that reads a phased VCF as text on stdin and decides
which records are feature SNPs from their position alone. Every one of the
following is a way the first attempts went wrong on real genes, and each is
now what the driver does by default:

- **`-s/-e` are the union of exons, not the gene span.** The README says so;
  `gtf_to_tables.py` writes `annot/exons.tsv` (merged exon starts and ends per
  gene) and `--exons` passes it. On the pilot genes the gene span classified
  421 records as feature SNPs where the exon union classified 17: introns
  contribute no allele-specific reads, only budget.
- **The cis region is the gene body plus/minus the window**, following
  rasqualTools, and the records RASQUAL sees are a tabix slice of an
  **AS-annotated VCF** piped straight in (`bcftools view -H -r REGION as.vcf.gz
  | rasqual ...`). `scripts/build_asvcf.py` builds that VCF once (`AS` FORMAT
  field, `ref,alt` per sample, `0,0` where phASER counted nothing; bgzip, not
  gzip, or tabix refuses it) and `--asvcf` points at it. Without `--asvcf` the
  driver builds one in `--out` over the regions it is about to test, so an
  ad-hoc run needs nothing prebuilt.
- **`--force`.** RASQUAL refuses any gene where `(fSNPs + 1) x tested SNPs`
  exceeds 30,000 (`main.c:582`, "Estimated computational time is too long
  ... aborted", one `SKIPPED` row) and the check is undocumented. Every pilot
  gene with a 1 Mb window is over it (12-33 fSNPs x 4,700-6,900 tested SNPs).
  rasqualTools batches genes by that product and excludes none, so the
  production practice is to run heavy genes, isolated and threaded, not to
  drop them. `--n-threads` parallelizes the tested-SNP loop within a gene;
  `--rasqual-jobs` runs genes concurrently on top of that.
- **Offsets come from the full expression matrix** (`colSums(counts)/mean`,
  as rasqualTools computes them), not from the genes sampled for the run.
- **Covariates are passed to RASQUAL** (`-x`, covariate-major binary written
  by the driver from `--covariates`), not regressed out of the counts.
- **The permutation null uses RASQUAL's own `-r`**, which permutes total and
  allele-specific counts against genotype inside RASQUAL. It draws its own
  permutation (seeded from time and pid), so the RASQUAL null is not paired
  with the hapmixQTL permutation. What `-r` moves (`nbem.c`, `randomPerm`):
  the total counts with their offsets and weights under one random order,
  and each feature SNP's genotype, allele counts and offset as a block under
  its own order; the tested-SNP genotypes and the `-x` covariates stay where
  they are. Under `-r` the covariates therefore explain nothing about the
  permuted totals, which is a difference from a null that permutes
  genotype alone. **The knockoff null does not reach the
  RASQUAL arm yet**: the pipe feeds RASQUAL the real genotypes, and the arm
  reports `knockoff_null_not_implemented` per gene rather than passing an
  observed run off as a null. Writing knockoff haplotypes into the VCF slice
  is the remaining piece.
- `--dump-rasqual DIR` keeps RASQUAL's raw stdout/stderr for every gene that
  produced no converged row. Column 23 is its convergence flag; the stderr
  says which gate fired. Read the dump before changing anything else.

### Low-count genes: counting noise and the expression floor

The comparison found a hapmixQTL failure that the self-tests, built on
Poisson(30) counts, could not: the Gibbs across-draw variance is
read-assignment uncertainty only, so a sample whose count is identical in
every draw -- zero reads, or reads compatible with nothing else -- has
`v_inf = 0` and, under `w = 1/(v_inf + tau)`, the largest weight in the
gene. On LOC124902138 (median 9 reads per sample) three zero-count samples
had `Vt = 1e-32`, the tau moment estimator collapsed to 4e-6, and the three
carried 99.8% of the total channel's weight: chi2 141 at a variant where an
unweighted regression of the same `t` gives 31, a Poisson GLM on the raw
totals 34, and RASQUAL 3.0. The seven pilot genes with no zero-count sample
had their three heaviest samples at 3-4% of the weight, i.e. uniform. The
allelic channel has had a guard for exactly this since the validation work
(`_zero_degenerate_ase_weights`); the total channel had none.

`compute_summaries_from_gibbs(..., count_noise=True)` adds the plug-in
Poisson variance of a log count to both channels (`1/(tot + 2 kappa)` for
`t`, `1/(yL + kappa) + 1/(yR + kappa)` for `a`), which is far below
`v_inf + tau` for a well-covered gene and dominant for a zero. Both the
driver and `run_hapmixqtl_from_salmon.py` default it on (`--no-count-noise`
to reproduce earlier results; the library default stays off so the
validation scripts under `tests/` are unchanged). With it LOC124902138 is
6.8, VLDLR-AS1 (56 reads/sample) moves from 12.8 to 11.5, SRPK1 (2,300
reads/sample) from 45.2 to 43.3 with the same lead.

The 30 well-expressed genes (`pilot30_hc.txt`) then exposed the same
estimator failing at the other end. Every mapping function clamped the
inferential variances to 1e-8 before `_prepare_channels`, so a sample with no
allele-specific reads (`Va = 0` exactly, `a = 0`) was never seen by the
degenerate-ASE guard (threshold 1e-12) and entered the tau moment estimator
at weight 1e8. A handful of them drove `tau_a` to ~1e-6 for the gene; the
informative samples were then weighted by Gibbs variance alone, which
understates the between-sample variance of `a` 2-25x on these genes, and the
known-variance SE was too small by that factor. Measured with a permutation
null (sample labels of genotype against expression): CRMP1 observed chi2
106.7, null maximum mean 97.7; TCF4 94.1 / 61.0; MATR3 71.1 / 41.1 (max
102.5); a calibrated maximum over ~4,600 tested variants is 12-16, which is
where RASQUAL's numbers sat. `_prepare_channels` now takes the raw variances,
estimates tau on the samples with `v_inf > 1e-12` and applies the floor
inside the weight; the null maxima on the same 12 genes are 10.7-17.4 (mean),
and the observed values follow them down (CRMP1 15.1, TCF4 11.5, FABP7 12.0,
TTC3 13.8; ANKRD36B stays at 39.9, and it is the gene where the phASER counts
independently show the same imbalance). `tests/test_hapmixqtl_calibration.py`
carries the gate, and the validation harness now drives `_prepare_channels`
rather than its own copy of the weight formula, which had the same clamp.

One property of the corrected arm to keep in mind when reading its numbers
against RASQUAL's: tau is estimated under the null model, so a gene's own cis
signal inflates it and the test is conservative where the signal is strong
(CCNI: the 22 heterozygotes at the lead show corr(a, s) = -0.90, a hets-only
regression gives chi2 86, the arm 17). A second one, covariates projected out
of both channels, is resolved in the next section.

### Covariates per channel, the sparse-channel rule and the permutation null

hapmixQTL used to project the same covariate set out of both channels,
whereas RASQUAL applies covariates to its total-count model only. The
allelic contrast `a = log((yL + k)/(yR + k))` is a within-sample difference
in which anything that acts on both haplotypes alike -- library size, the
expression PCs, sex, age, RIN -- cancels, so there is nothing for those
columns to remove from it; each one projected out costs one of the
informative samples (with 10 expression PCs against 46-62 informative
samples the allelic statistic of CCNI and CYP51A1 halved). `map_cis`,
`map_nominal`, `map_susie` and the second-pass functions now take
`ase_covariates_df` for the allelic channel: `SAME_COVARIATES` (the library
default, the previous behaviour), `None` for an intercept only, or the
channel's own DataFrame. The driver's `--ase-covariates` defaults to `none`
(`shared` reproduces the earlier runs); the CLI's `--ase_covariates` keeps
`shared` as its default. `run_hapmixqtl_from_salmon.py` passes no
covariates to either channel, so it is unaffected. The nominal p-value uses
one t reference for both channels, `dof = N - 2 - max(n_cov, n_cov_a)`, and
`map_cis` passes that dof to the permutation code, which used to take the
allelic residualizer's (with an intercept-only allelic channel the two
differ by the covariate count, and `map_cis` and `map_nominal` disagreed on
the same pair by a factor of 3.5 in p).

Two rules travel with it. A channel with fewer informative samples
(`v_inf > 1e-12`) than its design has columns plus two is switched off --
every weight zero, nothing projected, infinite SE -- and the meta-analysis
takes the other channel alone; the tau estimator raises rather than falling
back to every sample, which an earlier version did and which re-admitted
exactly the zero-variance rows for sparse genes. And the permutation null of
`map_cis` is now Freedman-Lane in whitened space: the whitened null
residuals of each channel are permuted among that channel's informative
samples (the two channels share one draw), instead of the raw `a` and `t`
being moved between samples at fixed weights. The old scheme handed a sample
another sample's value at its own precision; with inferential variances
spanning 0.01-2 and 10% samples without allele-specific coverage a null
gene's `pval_perm` averaged 0.94 with no rejection at 0.05 in 100 genes.
With the whitened permutation the same design gives mean 0.44, 6% at 0.05
(`test_pval_perm_is_calibrated_under_heteroskedasticity`). `map_cis` refuses
`se_mode='robust'`: the permutation statistic is the known-variance GLS
statistic and a sandwich SE has no counterpart in it; `map_nominal` still
offers it.

On the 30 well-expressed genes (`pilotL`, RASQUAL rows reused from
`pilotI`) the intercept-only allelic channel raises 19 of the 30 gene
statistics and lowers 11 (median change +1.1), and the number of genes above
15 goes from 6 to 13 against RASQUAL's 11; the Spearman correlation with
RASQUAL's statistics is 0.38 (p = 0.037). A fixed 15 is not a null level: on
the 12-gene external permutation null (genotype columns permuted against
expression, 10 draws) the per-gene null-maximum means run 9.7-17.6 and the
maxima 13.6-23.1 with the intercept-only channel (10.7-17.4 and 14.8-24.0
with the shared set), so the covariate split leaves the null where it was
while the observed values on the genes with signal rise (CYP51A1 14.7 ->
21.8 against RASQUAL's 31.8, TTC3 13.8 -> 16.9, APC 17.0 -> 20.8, CCNI 13.9
-> 16.2; ANKRD36B 39.9 -> 37.5). APC at 20.8 sits below its own null-maximum
mean of 17.6 only by that comparison, which is why the driver now carries
each gene's own empirical p (`pval_perm`, 1000 whitened-residual
permutations, `--hapmix-nperm`): in the final run (`pilotM`) 9 of 30 genes
are below 0.05 against their own null (SLC6A15, MON2, AGPAT5, CYCS, PDZD8,
EXOC2, CYP51A1, ANKRD36B, TCF4; 1.5 expected under a global null). RASQUAL's
arm has no per-gene empirical p (its statistic is a likelihood ratio), so
its 11 genes above 15 are not the same kind of count. RASQUAL's and
hapmixQTL's lead variants coincide on 1 of the 30 genes, which is why the
effect comparison has to be made at matched variants (below) rather than
gene-wise. The sparse-channel rule admits an allelic channel with as few as
three informative samples under an intercept-only design; the calibration
gates use about 70 and the 12 BrainVar genes had 46-85, so the floor is
untested below that.

### The reference-bias gate's orientation, and what the runner feeds RASQUAL

`reference_bias_diagnostic` needs, for every gene-sample, which haplotype
carries the reference allele where that sample's reads land. A gene has
many heterozygous sites, and the reference allele sits on L at some and on
R at others, so no single site's phase describes the gene. What mapping
bias adds to the haplotype totals is `sum_v reads_v * s_v` (bias favours REF
at every site carrying reads; `s_v = xL - xR` says which haplotype is ALT
there), so the orientation that exposes it is the sign of that
depth-weighted sum: `orient_haplotypes` in the library, and
`gene_orientation` in `run_hapmixqtl_from_salmon.py`, which assembles the
sites from the exon union when `--exons` is given, else the gene body, else
the het site nearest the TSS, with per-site depths from the phASER counts
when they are available and uniform weights otherwise. The driver and the
runner share it. Before this the runner oriented gene i by row i of the sign
matrix (an unrelated variant for every gene past the first) and the driver
used the single deepest feature SNP; a planted-bias test now shows the
diagnostic flags bias through the depth-weighted orientation and not through
a fixed unrelated site.

The runner also fed RASQUAL `expm1` of hapmixQTL's log phenotype
`log(tot/2 + kappa)`, i.e. half the count, with unit library sizes. It now
passes the Gibbs-mean totals of the tested genes and the per-sample library
size summed over every quantified gene, as the driver does (offset
`K = gene mean x relative library size`). The runner passes no covariates to
either hapmixQTL channel; the driver is the arm that carries the covariate
set.

### Effects at matched variants

`--rasqual-rows DIR` keeps every per-variant row RASQUAL writes (one file
per gene; `pilotK` re-ran the 30 genes with it, 98 min, and reproduces
`pilotI` exactly: the same lead, chi2, effect and phi on all 30). With the
rows, `matched_effects` reads RASQUAL at hapmixQTL's lead and re-runs
hapmixQTL at RASQUAL's lead through the same `map_cis` path restricted to
that variant (`hapmix_at`), and reports sign agreement, correlation and the
slope of hapmixQTL's effect on RASQUAL's at hapmixQTL's leads, at RASQUAL's
leads and at their union (a shared lead entering once). Both effects are
log(ALT/REF): RASQUAL's pi is the ALT haplotype's share of expression
(`nbem.c:1058`: expected expression 2(1 - pi) for hom-REF, 2 pi for hom-ALT)
and hapmixQTL's slope is per ALT dosage on the same VCF record, so no allele
flip is applied. Records are matched on (chrom, pos, ref, alt): a
multi-allelic site split into biallelic records occupies one position twice
(886 such positions in the 30 windows) and bcftools leaves the same joined
ID on both records, so `read_phased_vcf` now replaces a joined or missing ID
with `chrom_pos_ref_alt` and carries ref/alt columns.

`pilotM` (hapmixQTL arm with the intercept-only allelic channel and
`pval_perm`; RASQUAL rows from `pilotK`), 30 genes, per-gene table in
`matched_effects.tsv`:

| read at | pairs | sign agreement | r | slope (hapmixQTL on RASQUAL) |
|---|---|---|---|---|
| gene-wise leads (the old comparison) | 30 | | 0.24 | 0.26 +/- 0.19 |
| hapmixQTL's lead | 26 | 0.96 | 0.86 | 1.19 +/- 0.15 |
| RASQUAL's lead | 30 | 0.93 | 0.80 | 0.53 +/- 0.08 |
| union of leads | 55 | 0.95 | 0.78 | 0.75 +/- 0.08 |

The gene-wise comparison was measuring different quantities: at the same
variant the two arms agree in sign on 52 of 55 pairs. The slopes are
asymmetric in the direction each arm's lead selection predicts: an arm's own
lead is the maximum over about 4,600 tested variants, so its effect there is
inflated (the winner's curse) and the other arm's estimate at that variant
regresses toward zero, giving 1.19 at hapmixQTL's leads and 0.53 at
RASQUAL's. The union slope, 0.75, blends the two and is not a scale
calibration of either arm. Four of the 30 hapmixQTL leads have no RASQUAL
value: RASQUAL's row at those variants did not converge (negative
likelihood ratio, boundary flags in column 23: SLC6A15, AGPAT5, CAMSAP2,
CRMP1), so the comparison at hapmixQTL's leads is conditioned on variants
where RASQUAL's fit succeeded. Genes both arms put above their respective
levels are PDZD8 (RASQUAL 47.8 at its lead, 42.1 at hapmixQTL's; hapmixQTL
27.5, p = 0.001), CYP51A1 (31.8 / 27.9; 21.8, p = 0.003) and EXOC2 (24.6 /
4.7; 14.5, p = 0.034). FABP7 is the clearest disagreement: RASQUAL 23.6 with
log aFC -0.32 at its lead, where hapmixQTL re-run gives 0.004 with +0.002,
while at hapmixQTL's lead 32 kb away both see the effect (RASQUAL 11.6 /
-0.31, hapmixQTL 15.2 / -0.25).

Separately, a gene RASQUAL can use is not necessarily one Salmon quantifies:
CYP3A7 had allele counts at its feature SNPs from the aligner and a median
of 0 Salmon reads (73/92 zero samples), so hapmixQTL's statistic was exactly
0 against RASQUAL's chi2 12. `--min-count 6 --min-count-frac 0.2` (GTEx's
floor) now applies to the Salmon totals before the probe, and the design
record reports how many candidates it dropped.

### Choosing pilot genes

`scripts/select_pilot_genes.py` draws a gene list both methods can use,
stratified by expression, from three facts per gene: median Salmon reads per
sample (from the cache, written once to `annot/gene_expression_summary.tsv`),
exon sites with allele-specific reads in enough samples (phASER counts; the
allelic channel needs them in both methods), and VCF records inside the exon
union (each is a feature SNP to RASQUAL, whose cost is (fSNPs+1) x tested
SNPs, so `--max-exon-records` bounds the run time). `pilot30_hc.txt` is 10
genes from each of the 1k-3k, 3k-10k and >=10k reads/sample strata with
>= 3 informative exon sites and <= 40 exon records; its summary table sits
beside it.

Both observed tables carry a `lead` column (`chrom_pos_ref_alt`), so lead
agreement between arms can be read off directly.

For iteration, `--gene-list` restricts every input read (VCF, allelic counts,
Gibbs draws) to the windows around those genes, and `--cache-dir` keeps the
Gibbs load as memory-mapped arrays keyed by the input paths. An 8-gene rerun
then costs minutes of setup rather than the hour a whole-cohort load takes.

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
the extra variant classes add to hapmixQTL". The STR VCF can come from HipSTR
(`GB`), GangSTR or ExpansionHunter (`REPCN`, symbolic `<STRn>` alleles) on the
same samples; every source is normalized to reference-relative repeat units
(reference = 0, never absolute copy numbers), and the reference copy number is
kept per locus as `ref_units`. Phase the calls against the same scaffold if you
want the ASE channel to see the STRs; unphased calls still feed the total
channel. See `docs/ase_validation.md` §7j.

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

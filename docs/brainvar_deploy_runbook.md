# Running hapmixQTL on BrainVar

This is the operating procedure for mapping cis-eQTLs in the BrainVar cohort
with hapmixQTL, from Salmon quantifications to a per-gene results table, in
the shipped default mode. It also covers mixQTL mode, the published estimator
that serves as the comparator without Gibbs draws, and the optional
comparison with RASQUAL (a joint likelihood model of total and
allele-specific read counts, Kumasaka et al. 2016) that the Salmon runner can
add.

Every path below is on the server that holds BrainVar. `$DEPLOY` is the
deploy root, `/mnt/ssd/lalli/brainvar_hapmix_deploy`, and `$REPO` is the
repository checkout. Commands run from `$DEPLOY` unless they say otherwise,
and relative paths in them resolve there. Other questions have their own
documents: the statistic and its derivation are in
`docs/hapmixqtl_methods.md`; the input contract (values, units, gene filter,
permutation rule) is in `docs/pipeline_rules.md`; output columns are in
`docs/outputs.md`; what is validated, open or running is in
`docs/CURRENT_SCIENTIFIC_STATE.md`.

## The two modes this runbook runs

**Default mode** is the shipped hapmixQTL configuration and the only one the
Salmon runner, `scripts/run_hapmixqtl_from_salmon.py`, offers; it has no
switch for another. For each donor-gene pair it takes two measurements of one
cis effect, both computed from Salmon point estimates (`quant.sf` NumReads):

- The allelic channel is `A = log2((pL + 0.5)/(pR + 0.5))`, the log ratio of
  the two haplotypes' paired-transcript counts. Its weight is `1/Va`, where
  `Va` is the Gibbs variance of the same ratio across Salmon's 200 draws plus
  a counting term; `Va` sets only the shape of the weights, and the residual
  scale is fitted. A donor-gene pair enters this channel only if `Va > 1e-12`,
  it has haplotype reads, and not exactly one haplotype is below 0.5 reads.
  The channel is fitted through the origin.
- The total channel is `T = log2((pT + 0.5)/(effective_library_size + 1) x
  1e6)`, the half-read log-CPM of the count over all of the gene's
  transcripts. It has unit working variance, so every donor is kept at equal
  weight, and it is fitted with an intercept and the covariates.

Each channel's residual scale is fitted per variant (`Var(eps_i) = sigma^2
v_i`), and the two slopes are combined by inverse-variance weighting. The
combined nominal p is referred to a t distribution with Welch-Satterthwaite
degrees of freedom (chosen to match the first two moments of the combined
variance estimate), and the combined standard error carries Meier's
correction (a first-order inflation that accounts for the channel weights
being estimated rather than known). The allelic channel enters a gene's
statistic only if the gene has at least 15 informative allelic donors.

The gene-level detection call is the empirical permutation p, `pval_perm`,
and `pval_beta`, the same p computed from a Beta distribution fitted to the
permutation minima as in FastQTL. The permutation null is `records_signflip`:
each donor's record (phenotype value, weight and RNA-tied covariates) is
permuted against fixed genotypes, the genotype principal components stay with
the genotypes, and each permuted record's haplotype labels are swapped with
probability one half, which negates its allelic log ratio. A lead variant's
`pval_nominal` is never a gene-level p. The exact rules are in
`docs/hapmixqtl_methods.md` (its "Nominal p-values" section covers the t
references, Meier's correction and the admission floor) and
`docs/pipeline_rules.md`.

**mixQTL mode** is a NumPy port of the published mixQTL (`hakyimlab/mixqtl`
at `624ae44`), `tensorqtl/mixqtl_replication.py`, with the eleven divergences
found in the 2026-09-14 review removed. It reads Salmon point estimates and
never the Gibbs draws, so it is the comparator that measures what the draws
buy. It keeps the published natural-log response and the published GTEx v8
settings (total-count floor 100, allele-specific count floor 50 and ceiling
1,000 per haplotype, weight cap 10). Its driver is described in "Running
mixQTL mode" below.

No other configuration is current. The fitted variance models
(`variance_model`, `variance_prior`), `tau_mode='estimate'` and the
known-variance standard error were deprecated on 2026-09-23
(`$DEPLOY/deprecated_models/README.md`). Fine-mapping (`map_susie`) and the
STR and multi-allelic second pass are not supported in default mode. Where
the older procedures and their results are recorded is listed in "Historical
procedures" at the end.

## Data governance: where this runs and what may leave the machine

BrainVar is dbGaP controlled access (phs001900). Everything here runs on the
machine that holds the data; the question is what leaves it.

None of the scripts in this procedure opens a network connection. That was
checked on 2026-10-01 by searching `run_hapmixqtl_from_salmon.py`,
`build_point_estimate_cache.py`, `edger_library_normalization.R`,
`build_covariates.py`, `gtf_to_tables.py`, `brainvar_pairing.py`,
`verify_pairing.py`, `run_phaser_cohort.py`, `phaser_to_matrix.py`,
`str_integrate.py`, `compare_mixqtl_replication.py`, `tensorqtl/hapmixqtl.py`
and `tensorqtl/mixqtl_replication.py` for `urllib`, `requests`, `http://`,
`https://`, `socket`, `curl`, `wget` and `git clone`, with no hits. Two
scripts elsewhere in `scripts/` do reach the network, and neither sends data
out: `build_rasqual.sh` clones RASQUAL's public source, and
`extract_gtex_phaser.py` downloads public GTEx matrices and is not used here.

The runner writes these files into its `--out` directory:

| file | contents | handling |
| --- | --- | --- |
| `eval_bundle.json` | aggregate statistics only: run metadata, the reference-bias gate's pooled result, a QQ summary of the leads' nominal p, quantiles of slopes and standard errors, the channel-concordance regression, the share of leads failing the cis/trans test, and RASQUAL summaries when RASQUAL was run. No per-donor or per-gene value. | designed to be shared |
| `hapmixqtl_cis.tsv.gz` | one row per gene: lead variant, effects, p-values and diagnostics, including `loo_donor`, which names a donor | keep on the machine |
| `edger/` (written only without `--edger-dir`) | `totals_all.tsv.gz`, the gene-by-donor count matrix, and per-donor library sizes | keep on the machine |
| `rasqual_cis.tsv.gz` (only with `--rasqual`) | RASQUAL's per-gene rows | keep on the machine |
| the runner's console output (`<run_dir>.log` in the command below) | progress lines naming each donor, gene counts and the gate message | keep on the machine |

Every file and key is described in `docs/outputs.md`, section "Salmon
runner". Confirm all of this against the data use agreement before carrying
anything off the machine.

## Environment and self-tests

```bash
git clone https://github.com/JosephLalli/tensorqtl.git && cd tensorqtl
git checkout <hapmixQTL release branch>
pip install numpy scipy pandas torch pandas_plink h5py qtl pysam pytest
pip install -e . --no-deps
```

The procedure also needs `bcftools` and `tabix`, R with the edgeR package
(the runner and `build_point_estimate_cache.py` call
`scripts/edger_library_normalization.R` through `Rscript`), and phASER for
the genotype preparation (a tool that phases heterozygous sites from the
reads spanning them and counts reads per allele; installed here at
`$DEPLOY/tools/phaser`, version 1.2.0). On this server R can crash inside a
BLAS routine because the shell environment loads two OpenBLAS builds; the
per-command fix is recorded in `CLAUDE.md` under "R's BLAS crash". Mapping
runs on a GPU when one is visible to torch and on the CPU otherwise.

Run the self-tests before touching real data. They exercise the code on
fabricated inputs, so they catch an environment problem in minutes rather than
after hours of computation.

```bash
cd $REPO
pytest tests/test_hapmixqtl.py tests/test_hapmixqtl_calibration.py \
       tests/test_hapmixqtl_perm_scheme.py tests/test_hapmixqtl_point_estimates.py \
       tests/test_hapmixqtl_allelic_df.py tests/test_hapmixqtl_meier.py \
       tests/test_half_read_default.py tests/test_half_read_runner.py \
       tests/test_fitted_variance_quarantine.py tests/fitted_variance/ \
       tests/test_cli.py tests/test_mixqtl_replication.py -q
python3 scripts/run_hapmixqtl_from_salmon.py --selftest
```

The expected test count is recorded in `tests/README.md`. The known answer in
`tests/test_hapmixqtl_meier.py` needs a GPU and `$DEPLOY`; without them that
test skips. A bare `pytest tests/` also runs four upstream test files that
carry pre-existing failures unrelated to hapmixQTL (`CLAUDE.md`,
"Self-tests"). With `RASQUAL_BIN` pointing at a RASQUAL binary, the runner's
self-test also exercises the RASQUAL comparison.

The input-building scripts have their own self-tests:

```bash
for s in gtf_to_tables brainvar_pairing verify_pairing run_phaser_cohort \
         phaser_to_matrix build_covariates; do
    python3 scripts/$s.py --selftest
done
```

## Inputs on this server

| input | path | built by |
| --- | --- | --- |
| Salmon quantifications, one directory per RNA library | `/mnt/ssd/lalli/nf_stage/RNA_reference_comparison_results/reference_comparison_results/bv2/personalized_T2T_NCBI110_pseudoalignment/expression_results/salmon_pseudocounts/<rna_library>/` | the `rnaseq_JLL` Nextflow pipeline |
| Salmon manifest, `dna_library <TAB> Salmon directory` (the directory holding `quant.sf` and `aux_info/`), 92 donors | `$DEPLOY/cohort/salmon.tsv` | `brainvar_pairing.py` |
| Donor pairing (DNA library, RNA library, BAM) | `$DEPLOY/cohort/pairing.tsv` | `brainvar_pairing.py` |
| Library metadata v1.4 | `/mnt/ssd/lalli/nf_stage/draft_brainvar2_library_metadata_v1.4.tsv` | frozen upstream |
| Reference-aligned RNA BAMs (for phASER) | `/mnt/data/lalli/nf_stage/reference_comparison_results_RNA/T2T_NCBI110/star_salmon/HSB<n>.markdup.sorted.bam` | STAR, T2T-CHM13 reference |
| Population-phased genotypes | `/mnt/ssd/lalli/nf_stage/brainvar2/gatk_t2t_haplotypecaller.joint_called.phased.all_variants.multiallelic.all.nostar.bcf` | GATK HaplotypeCaller joint calls, phased |
| Annotation tables | `$DEPLOY/annot/genes.tsv`, `tx2gene.tsv`, `genes.bed`, `exons.tsv` | `gtf_to_tables.py` |
| Analysis VCF | `$DEPLOY/prepped/analysis.snps.maf01.vcf.gz` | the phased BCF passed through phASER's output VCFs, then `bcftools` (see the genotype section) |
| phASER allelic counts, `chr`-named | `$DEPLOY/prepped/allelic_counts_manifest.chr.tsv` | phASER, contigs renamed |
| Point estimates and edgeR library sizes | `$DEPLOY/cache/gibbs_56b63c3b37ed5df8/point_estimates/` | `build_point_estimate_cache.py` |
| Covariates (current) | `$DEPLOY/cov/half_read_point_calibration_20260930/` | `build_covariates.py --point-estimates` |

## Salmon quantification: what the draws are and their known limit

hapmixQTL needs haplotype-resolved expression. Salmon must have been run
against a personalized diploid transcriptome, with two copies of every
transcript, one per haplotype, distinguished by a suffix pair. A quantification
against a standard reference transcriptome carries no allelic information, and
the runner refuses it with `no haplotype-paired transcripts in <dir> using
suffixes (...)`.

The BrainVar quantifications were made with stock Salmon 1.10.3 in mapping
mode with `--numGibbsSamples 200` against per-donor diploid transcriptomes
built by g2gtools, whose haplotype copies carry the suffixes `_L` and `_R`
(`aux_info/meta_info.json` records `salmon_version` 1.10.3, `samp_type`
`gibbs` and `num_bootstraps` 200; `cmd_info.json` records the command). The
runner's default suffix pair is `_hapA,_hapB`, so every BrainVar run must pass
`--hap-suffix _L,_R`. The index kept no duplicate sequences (`meta_info.json`
records `keep_duplicates: false`), so where a donor is homozygous across a
transcript its two copies are identical and Salmon keeps only one. The runner
pairs a transcript only when both suffixed rows exist, so a donor-gene pair
with no heterozygous transcript has no allelic information at all, however
well expressed the gene is; its total expression still counts in the total
channel. The consequences for the share of donor-gene pairs with allelic
information are recorded in `docs/measurement_record.md` ("Why 57.8% of donor-gene pairs have
no allele-specific information").

**Known limit of the Gibbs variance (kept as a caveat by user decision,
2026-10-01).** Salmon's Gibbs sampler uses a prior of 1 per active transcript,
while its point estimate uses 0.01. For donor-gene pairs with few
haplotype-informative reads, about 3 to 30, that prior shapes `Va` and makes
it too small: on independent split halves of donor 100's reads, the stock
draws understate the random error of the allelic ratio 1.5- to 3.2-fold in
that range. A Salmon 1.10.3 fork with a `--gibbsPriorGroups` option, which
divides the prior over gene-by-haplotype groups, was built and validated on
that one donor. Re-quantifying the 92 donors with it is prepared and on hold
(user decision, 2026-10-01); every current result uses the stock draws. The
evidence is in `$DEPLOY/salmon_informative_reads_20260930/README.md` (and
`split_half/split_half_calibration.html` beside it), and the prepared run,
which has not been executed, is in `$DEPLOY/salmon_gibbspriorgroups_20261001/`.

These are the only Salmon runs this procedure uses. A survey of every Salmon
run on both disks on 2026-09-10 found this to be the one coherent set of
personalized quantifications with 200 Gibbs draws (229 RNA libraries); the
other arms are standard-reference quantifications, runs with
`--numBootstraps` resampling, or runs without draws.

## Pairing donors across RNA, DNA and alignments

### Never join by name

Three identifier systems are in play, and they do not agree:

| thing | identifier | example |
| --- | --- | --- |
| diploid quantification | bulk RNA library | `587_R1` |
| genotypes (VCF sample) | WGS library | `589_D1` |
| RNA alignment for phASER | subject | `HSB589` |

BrainVar carries a known sample relabelling, and the two RNA runs resolved it
differently. The alignment run applied it (the BAM named `HSB587` was built
from `HSB583_1_val_1.fq.gz`, and the 2022 relabelling map says HSB583 is
HSB587), while the quantification did not (`587_R1` was quantified from
`HSB587.R1_val_1.fq.gz`). For five subjects this produces a shift chain, and
joining the quantification to the BAM by number silently pairs two different
donors, because every identifier involved exists.

The DNA library is the only common key, and it is what the VCF is keyed on,
so every manifest uses it as `sample_id`. Build the pairing with:

```bash
ARM=/mnt/ssd/lalli/nf_stage/RNA_reference_comparison_results/reference_comparison_results/bv2/personalized_T2T_NCBI110_pseudoalignment
python3 $REPO/scripts/brainvar_pairing.py \
    --metadata  /mnt/ssd/lalli/nf_stage/draft_brainvar2_library_metadata_v1.4.tsv \
    --vci-dir   $ARM/vcf2vci \
    --salmon-dir $ARM/expression_results/salmon_pseudocounts \
    --bam-dir   /mnt/data/lalli/nf_stage/reference_comparison_results_RNA/T2T_NCBI110/star_salmon \
    --vcf-samples <(bcftools query -l <phased.bcf>) --out cohort/
```

It writes `salmon.tsv`, `bams.tsv`, `samples.txt` and `pairing.tsv`, all keyed
on the DNA library. On this data it pairs 92 donors and reports the five rows
a name join would get wrong:

```
587_D1: rna=583_R2  bam=HSB587        589_D1: rna=587_R1  bam=HSB589
590_D1: rna=589_R1  bam=HSB590        591_D1: rna=590_R1  bam=HSB591
593_D1: rna=591_R1  bam=HSB593
```

Use metadata v1.4, not an earlier table. `draft_brainvar2_library_metadata_v1.4.tsv`
(SHA-256 prefix `c70e3599`, 841 rows by 29 columns) is the frozen table;
`METADATA_V1.4_FREEZE.md` in the brainvar2 repository records the freeze.
Against v1.3.1 it changes `matchingDNALibrary` for two usable bulk-RNA
records, `321_R2` (`321_D1` to `321_D2`) and `513_R2` (`175_D1` to `175_D2`).
Both candidate DNA libraries exist in the VCF, so an older table attaches the
wrong one with no error.

Each edge of the join rests on evidence rather than on a name. RNA library to
DNA library comes from the run itself: each per-sample g2gtools VCI header
carries `##STRAIN=<dna_library>`, which is the genotype the personalized
transcriptome was built from. Across the 229 quantified libraries the VCI
agrees with v1.4 for 227; the two exceptions are exactly the two records v1.4
changed, whose diploid references were built from the superseded assignment.
Neither is in the 92-donor cohort. BAM to DNA library is the numeric rule
`HSB<n>` to `<n>_D1`, checked against the reads below.

### Verifying the pairing against the reads

Every identifier here is a claim someone wrote down; the reads are not. An
RNA alignment carries its donor's genotypes at expressed sites, so the donor
can be identified directly:

```bash
python3 $REPO/scripts/verify_pairing.py --pairing cohort/pairing.tsv \
    --bam-dir /mnt/data/lalli/nf_stage/reference_comparison_results_RNA/T2T_NCBI110/star_salmon \
    --vcf <phased.bcf> --chrom chr1 --contig-map rename_chrs.tsv --out verify/
```

It scores each BAM against every donor in the pairing, not only the claimed
one, and flags any sample whose best match is not the claimed donor or whose
margin over the runner-up is below `--min-margin` (default 0.15). A swap does
not announce itself otherwise: mispaired allelic data still run, still
converge and still produce QTLs. Run it on the whole pairing, because the
candidate pool is exactly the donors listed and a smaller pool inflates the
margin (six donors gave margins of 0.38-0.50 against a subset, and 0.30-0.34
when scored against all 92).

When the pairing is right, the correct donor scores 0.986-0.994 on this data
and the best competing donor 0.60-0.75, against a median across donors near
0.64; two unrelated people agree at roughly 0.6 by chance given the
allele-frequency spectrum, so a margin above about 0.2 is unambiguous. Eight
samples were checked when this procedure was first written, with no
discrepancies: the three shift-chain BAMs (`HSB587` 0.991, `HSB589` 0.991,
`HSB593` 0.989), four ordinary ones, and `589_D1`, the one pairing that rests
on the metadata rather than on shared input files (its quantification read
FASTQ `HSB587`, which no alignment arm used, so that FASTQ was aligned and
genotyped separately and matched `589_D1` at 0.990 against a runner-up of
0.747). Run it on the rest before publishing.

## Annotation tables

The transcript names under the `_L`/`_R` suffixes are ordinary RefSeq
accessions, so the reference annotation resolves them; the per-sample
personalized GTFs in the quantification arm are dangling symlinks and are not
needed. This deployment's tables were built from the T2T-CHM13 NCBI RefSeq
annotation with UCSC-style `chr` names:

```bash
python3 $REPO/scripts/gtf_to_tables.py \
    --gtf /mnt/ssd/lalli/nf_stage/genome_refs/T2T-CHM13_v2_ncbi110/GCF_009914755.1_RS_2024_08-T2T-CHM13v2.0_genomic.UCSC_chr.with_GRCh38_rCDS_chrM.exon_ids.gtf \
    --out annot/
```

That gives 58,516 genes and 183,140 transcripts and resolves every
haplotype-paired transcript of a sample to a gene. The outputs:

- `genes.tsv`: `gene_id chr start end tss`, 1-based inclusive, with a
  strand-aware TSS (start on `+`, end on `-`).
- `tx2gene.tsv`: `transcript_id gene_id`, for collapsing Salmon to genes
  (`--tx2gene`).
- `genes.bed`: `chr start stop gene_id`, 0-based half-open with no header,
  for phASER's `--features`. It is not `genes.tsv` with the columns moved:
  `phaser_gene_ae.py` parses from line 1 with no header handling and expects
  0-based coordinates, and the name column must be the `gene_id` because
  that is what the downstream joins use.
- `exons.tsv`: merged exon starts and ends per gene, used by the RASQUAL
  comparison.

The command prints the chromosome names it saw. They must match the VCF's
(see "Naming traps" below). `annot/genes.NC.tsv` and `annot/genes.NC.bed`
hold the same rows with contigs renamed to RefSeq accessions, for running
phASER against the accession-named BAMs.

## Phased genotypes and the analysis VCF

The runner reads phased genotypes from `$DEPLOY/prepped/analysis.snps.maf01.vcf.gz`:
biallelic SNPs with cohort minor allele frequency at least 0.01 from the
population-phased joint call set. Its GT field carries that population phase
unchanged; phASER's read-backed phase is present only in extra FORMAT fields,
which the runner does not read (see "Assembling the phASER outputs" below). It
was built as follows; each step's exact command is recorded in the VCF
header's `##bcftools_*Command` lines (`bcftools view -h`), which is how to
check the provenance of any copy.

**Contig names.** The T2T BAMs name contigs by RefSeq accession
(`NC_060925.1`) while the phased BCF and the annotation use `chr1`, and phASER
finds nothing when the BAM and the VCF disagree. This deployment renamed the
VCF to the BAMs' accessions with `vcf/chr2nc.tsv` (restricting it to the 92
donors with `-S vcf/cohort92.txt`, giving `vcf/cohort92.NC.vcf.gz`), ran
phASER, and renamed back to `chr` names afterwards with `rename_chrs.tsv`.
Both maps correspond 1:1 over 25 contigs. For a different reference, derive
the map by matching the BAM header's contig lengths against the `chr`-named
FASTA index:

```bash
samtools view -H <bam> | awk '/^@SQ/{for(i=1;i<=NF;i++){if($i~/^SN:/)n=substr($i,4);if($i~/^LN:/)l=substr($i,4)}print n"\t"l}' > bam_ctg.tsv
awk 'NR==FNR{a[$2]=$1;next} ($2 in a){print $1"\t"a[$2]}' chm13v2.0_maskedY_rCRS.fasta.fai bam_ctg.tsv > rename_chrs.tsv
```

**phASER per donor.** `run_phaser_cohort.py` runs `phaser.py` and
`phaser_gene_ae.py` over the pairing, resumably:

```bash
python3 $REPO/scripts/run_phaser_cohort.py --pairing cohort/pairing.tsv \
    --bam-dir /mnt/data/lalli/nf_stage/reference_comparison_results_RNA/T2T_NCBI110/star_salmon \
    --vcf vcf/cohort92.NC.vcf.gz --features annot/genes.NC.bed \
    --phaser-dir tools/phaser --out phaser_out/ --jobs 12 --threads 5
```

Its defaults are the values this data needs: `--mapq 255` (STAR's encoding of
a uniquely mapped read; another aligner needs its own value), `--pass-only 0`
(the call set's FILTER column is `.`, for example at all 110,057 records in
the first 5 Mb of chr1, and phASER's default keeps only `PASS`),
`--id-separator -` (phASER's default `_` is rejected for contig names that
contain it, as RefSeq accessions do) and `--write_vcf 1`. It
always passes `--python_string`, since phASER otherwise re-invokes a Python 2
path partway through each sample. Each donor took about 13 minutes at five
threads (`$DEPLOY/logs/cohort.log`).

**Assembling the phASER outputs.**

```bash
python3 $REPO/scripts/phaser_to_matrix.py --manifest phaser_manifest.tsv --out prepped/
```

`phaser_manifest.tsv` is `sample_id <TAB> phASER output prefix`. This writes
`prepped/phaser_matrix.gw_phased.txt.gz`, `prepped/allelic_counts_manifest.tsv`
and `prepped/samples.txt`; it keeps only gene-donor pairs whose haplotype
labels phASER anchored to the genome-wide phase (`gw_phased = 1`). The
script can also write a VCF whose GT carries phASER's read-backed phase
(`--vcf`), but the analysis VCF was not built that way. Instead
`merge_shards.sh` in the deploy root runs one `bcftools merge` per contig over
phASER's per-donor output VCFs (`phaser_out/*.vcf.gz`) and concatenates them
into `prepped/rephased.vcf.gz`, still accession-named. phASER ran without
`--gw_phase_vcf` (default 0, "do not replace GT"; each donor's log says "GT
field is not being updated"), so its output VCFs keep the input GT and record
phASER's own phase only in the FORMAT fields it adds (`PG`, `PB`, `PI`, `PM`,
`PW`, `PC`). Despite its name, `rephased.vcf.gz` carries the population phase
in GT. Checked on 2026-10-01 for donor `100_D1` over chr1:1-6 Mb: the analysis
VCF's GT equals the original phased BCF's at all 41,474 shared biallelic SNPs,
7,465 of them heterozygous, with 0 phase flips (`brainvar_hapmix_deploy/release_closure_20261001/phase_check.log`).

**Normalizing and filtering.** `normalize.sh` in the deploy root renames the
contigs back with `rename_chrs.tsv`, left-aligns against
`chm13v2.0_maskedY_rCRS.fasta` and splits multi-allelic sites
(`bcftools norm -m -any`), giving `prepped/rephased.norm.vcf.gz`. The analysis
VCF is then

```bash
bcftools view -m2 -M2 -v snps -c 1:minor -q 0.01:minor -Oz \
    -o prepped/analysis.snps.maf01.vcf.gz prepped/rephased.norm.vcf.gz
tabix -p vcf prepped/analysis.snps.maf01.vcf.gz
```

which holds 15,340,329 records. The runner applies no further allele-frequency
filter.

**phASER allelic counts.** `allelic_counts_chr/` holds each donor's phASER
`allelic_counts.txt` with contigs renamed to `chr` names, listed by
`prepped/allelic_counts_manifest.chr.tsv`. Use that manifest wherever the
runner takes `--allelic-counts`. `prepped/allelic_counts_manifest.tsv` points
at the accession-named originals in `phaser_out/`, so none of its counts
falls in a `chr`-named gene window.

**chr14, chr15 and chr22 are excluded (user decision, 2026-09-28).** Every
copy of the phased genotypes, including the personalized references Salmon
quantified against, stops within the first 1.5-3.1 Mb of these three contigs,
so no gene there has haplotype-paired transcripts and none can be tested
(the analysis VCF holds 30,177, 29,951 and 38,724 records on them, against
632,589 on chr13). Nothing has to be passed to exclude them; the runner
reports and skips their genes (see "Genes the run reports but does not
test"). Analyses cover the other 19 autosomes. The unphased joint calls are
complete, so re-phasing is possible later; the record is
`$DEPLOY/phased_vcf_inventory_20260928/README.md`.

**A separate phASER build feeds only the alignment-based counts.** A second
phASER run on 2026-09-28 used an SNV-only VCF with multi-allelic
heterozygotes masked, per-strand runs over gene-unique exonic segments, the
HLA and CHM13-accessibility blacklists, and then WASP filtering (which
re-maps each read with its alleles swapped and discards reads whose mapping
changes). It feeds the alignment-based (native) counts that the benchmark's
native arms and the held-out referee's TReCASE arm read (TReCASE is a joint
likelihood of total read counts and allele-specific expression;
`scripts/native_counts.py` builds the counts), not the analysis VCF above.
Records: `$DEPLOY/phaser_stranded_20260928/README.md` and
`$DEPLOY/wasp_20260928/README.md`.

## Point estimates, the eQTL gene filter and covariates

### Point estimates and edgeR library sizes

```bash
python3 $REPO/scripts/build_point_estimate_cache.py
```

This takes no flags; its paths (`$DEPLOY`, the Gibbs cache
`cache/gibbs_56b63c3b37ed5df8`, `cohort/salmon.tsv`, `annot/tx2gene.tsv`,
`annot/genes.tsv`) are written into the script. It reads every donor's
`quant.sf`, sums the point estimates to genes exactly as the runner does
(`pL`/`pR` over haplotype-paired transcripts, `pT` over all transcripts), and
runs edgeR through `edger_library_normalization.R`: `filterByExpr` with no
design (a gene is kept when it reaches a minimum count-per-million in enough
samples and a minimum total count), intersected with the calibration-phase
restriction (protein-coding, meaning at least one curated RefSeq `NM_`
transcript, and autosomal), then TMM normalization (trimmed mean of M-values,
edgeR's per-library scaling factor that centres most genes' log ratios to a
reference library on zero). The effective library size is `lib.size` times
the TMM factor. It writes `point_estimates/` beside the cache: `pL.npy`,
`pR.npy`, `pT.npy`, `totals_all.tsv.gz`, `restrict_calibration.txt`,
`edger/edger_samples.tsv`, `edger/calibration_genes.txt` and `summary.json`,
and aborts unless the point-estimate totals equal the all-gene totals,
`pL + pR <= pT`, the point-estimate totals track the Gibbs posterior means
(Pearson correlation of log(count + 1) above 0.99) and edgeR's sample order
matches the cache's.

**The Gibbs cache it reads has no live builder.** `build_point_estimate_cache.py`
takes its gene and donor order from `cache/gibbs_56b63c3b37ed5df8/genes.txt`
and `samples.txt` and checks itself against the cached draws `YL.npy`,
`YR.npy` and `YT.npy`. That cache was written by the retired
`compare_pipelines.py --cache-dir` and is on disk for this cohort, so the
step reruns here unchanged. A new cohort has no supported way to build it;
the retired script is archived in `$DEPLOY/retired_scripts_20261001/` and can
be run from a checkout of commit `8e347fd~1`.

The eQTL gene filter is the 12,955 genes in `edger/calibration_genes.txt`.
This is the calibration-phase filter, by user decision explicitly temporary;
the deployment filter may differ. A different filter means changing the
restriction in `build_point_estimate_cache.py`, rebuilding the covariates on
the new `edger/` folder, and passing that folder to the runner, because the
expression-PC gene filter must equal the eQTL gene filter.

### Covariates

```bash
cd $DEPLOY
python3 $REPO/scripts/build_covariates.py \
    --metadata /mnt/ssd/lalli/nf_stage/draft_brainvar2_library_metadata_v1.4.tsv \
    --pairing pairing.tsv --salmon cohort/salmon.tsv --tx2gene annot/tx2gene.tsv \
    --vcf prepped/rephased.vcf.gz --hap-suffix _L,_R \
    --point-estimates cache/gibbs_56b63c3b37ed5df8/point_estimates \
    --out cov/half_read_point_calibration_20260930
```

This is the build in use (`$DEPLOY/cov/half_read_point_calibration_20260930/`;
`covariate_build.json` there records its inputs). It writes 17 columns: age
in days and its square, RIN, sex, three genotype principal components and ten
expression principal components. The expression PCs are computed on the
half-read log-CPM of the point estimates over the eQTL gene set, each gene
centred (not scaled) and first residualized on the metadata and genotype PCs,
so they are orthogonal to those columns and do not re-encode age, batch or
ancestry. An indicator whose minority level has fewer than two donors is
dropped (`--min-level-n`). Beside `covariates.tsv` it writes
`genotype_covariates.txt` (the genotype-PC column names, which stay with the
genotypes under permutation), `covariate_build.json` (columns, the
genotype-tied and RNA-tied split, the input paths and `expression_pc_unit`),
and `covariates.bin`/`covariates.n` for RASQUAL's `-x`.

Run it from `$DEPLOY`: `covariate_build.json` records the point-estimate
folder as the relative path given here, and the runner resolves it from its
working directory. Without `--point-estimates`, `build_covariates.py` builds
the pre-2026-09-25 expression PCs (`log1p` of raw counts), which the runner
refuses.

Covariates are passed to the mapper, not regressed out first. hapmixQTL
projects them out inside each channel's weighted space, and the default
allelic channel is fitted through the origin with no covariates (anything
acting on both haplotypes alike cancels from the within-donor log ratio), so
the covariates act on the total channel.

## Running default mode

### The command

`--gene-pos` is read as gene, chromosome, TSS, start, end, while
`annot/genes.tsv` is gene, chromosome, start, end, TSS; passed as it is, the
runner would take each gene's start as its TSS without complaint. Reorder it
first. Run from `$DEPLOY` (see "Covariates"):

```bash
cd $DEPLOY
mkdir -p <run_dir>
awk -v OFS='\t' '{print $1, $2, $5, $3, $4}' annot/genes.tsv > <run_dir>/gene_pos.tsv
python3 $REPO/scripts/run_hapmixqtl_from_salmon.py \
    --vcf prepped/analysis.snps.maf01.vcf.gz \
    --manifest cohort/salmon.tsv --tx2gene annot/tx2gene.tsv --hap-suffix _L,_R \
    --gene-pos <run_dir>/gene_pos.tsv \
    --covariates cov/half_read_point_calibration_20260930/covariates.tsv \
    --edger-dir cache/gibbs_56b63c3b37ed5df8/point_estimates/edger \
    --out <run_dir> > <run_dir>.log 2>&1
```

`--vcf`, `--manifest`, `--tx2gene`, `--covariates` and `--gene-pos` are
required. `--edger-dir` reuses the edgeR run the covariates were built on, so
the eQTL gene set and effective library sizes are those of the expression
PCs. Without it the runner runs edgeR itself on every gene with no
restriction (or on `--gene-restrict <gene list>`), and the covariate check
refuses unless the result reproduces the gene set and library sizes recorded
for the covariates.

The two largest costs are the Gibbs draws and the VCF. The runner holds the
allelic draws as two float64 arrays of genes x donors x draws, about 5 GB each
for 34,457 genes, 92 donors and 200 draws, before transient copies; and it
parses the whole analysis VCF, which takes roughly a quarter of an hour.

### What the run does, in order

1. Reads each donor's Gibbs draws (`aux_info/bootstrap/`, the directory name
   Salmon uses for either kind of draw) for haplotype-paired transcripts,
   summed to genes per haplotype.
2. Reads the point estimates the same way and takes the effective library
   sizes from edgeR.
3. Builds `A`, `T`, `Va` and `Vt` (`prepare_default_inputs` in
   `tensorqtl/hapmixqtl.py`), restricts to the eQTL gene filter, and prints
   how many filtered genes have no Gibbs draws (they are reported, not
   tested) and how many donor-gene pairs the one-sided rule excluded from the
   allelic channel.
4. Reads the phased VCF and orders donors as the VCF does.
5. Splits the covariates into RNA-tied and genotype-tied columns and checks
   their provenance (`check_covariate_provenance`): the expression PCs must
   be in the half-read unit, on the eQTL gene set, with the same effective
   library sizes. It refuses otherwise.
6. Runs the reference-bias gate (next section).
7. Runs `map_cis` in default mode: a 1 Mb window around each TSS, 10,000
   permutations under `records_signflip`, the allelic admission floor of 15
   donors, and the leave-one-donor-out check at each lead.
8. Writes `hapmixqtl_cis.tsv.gz` and `eval_bundle.json`.

### The reference-bias gate

hapmixQTL does not model reference mapping bias, and its type-I error rises
steeply rather than gradually when bias is present (`docs/hapmixqtl_methods.md`,
"The reference-bias gate"). Before mapping, the runner pools each gene's
reference-allele fraction over its heterozygous donors, orienting each donor
by the sign of a sum over the heterozygous sites in the gene body (taken from
`--gene-pos` start and end), and tests the mean of the per-gene fractions
against 0.5 with genes as the unit. Real cis effects favour the
reference or the alternate allele at random, so they cancel in that mean;
mapping bias always favours the reference. At p < 1e-3 the runner refuses to
map, still writes an `eval_bundle.json` carrying the diagnostic for triage,
and asks for WASP-corrected or variant-aware quantification. `--force`
proceeds anyway and is not recommended. Supplying `--allelic-counts
prepped/allelic_counts_manifest.chr.tsv` weights each site by its phASER
depth instead of counting sites alike; it changes nothing else in a default
run.

### Options

| flag | default | effect |
| --- | --- | --- |
| `--hap-suffix` | `_hapA,_hapB` | haplotype suffix pair; BrainVar needs `_L,_R` |
| `--window` | 1000000 | cis window around the TSS, in bases |
| `--perm-scheme` | `records_signflip` | the permutation null; `records` omits the haplotype-label swap, and `residuals` is the earlier Freedman-Lane scheme (the null model's whitened, leverage-standardized residuals permuted at fixed weights), retained but conservative where weights vary |
| `--count-noise` / `--no-count-noise` | on | adds the counting term to the allelic `Va` only; the total channel's unit variance is unaffected |
| `--genotype-covariates` | `auto` | which covariate columns stay with the genotypes under permutation; `auto` reads `genotype_covariates.txt` beside `--covariates`, `none` ties every column to the RNA record |
| `--edger-dir` | none | reuse a finished edgeR folder instead of running edgeR |
| `--gene-restrict` | none | gene list intersected with `filterByExpr` when the runner runs edgeR itself |
| `--covariates-unverified` | off | proceed when the covariate provenance check fails; not recommended |
| `--force` | off | proceed despite a reference-bias flag; not recommended |
| `--asc-cutoff`, `--asc-cap`, `--trc-cutoff`, `--mixqtl-cutoffs` | off | mixQTL's count cutoffs, applied to hapmixQTL's donor admission so the two estimators can be compared on a matched donor set; a comparison instrument, not a setting for results. mixQTL's weight cap is deliberately not applied |

The runner passes no random seed, so `pval_perm` and `pval_beta` differ
between reruns by Monte Carlo error at 10,000 permutations; leads, slopes and
nominal p-values do not.

### Opt-in variant classes: STRs and multi-allelic sites

Off by default; the standard analysis tests biallelic SNPs only. Two flags
add rows to the `map_cis` scan and nothing else:

```bash
    --str-vcf <str.vcf.gz>   # STRs as per-haplotype repeat length, in reference-relative repeat units
    --multiallelic           # multi-ALT rows of --vcf (normally skipped), one split row per ALT
```

Either can change which variant is a gene's lead, and the output gains a
`variant_type` column (`snp`, `str`, `ma_allele`). The runner no longer runs
the STR-curvature and multi-allelic categorical second pass, which supports
only the deprecated known-variance standard error. The STR VCF can come from
HipSTR, GangSTR or ExpansionHunter on the same samples; every source is
normalized to repeat units relative to the reference allele
(`scripts/str_integrate.py`). Unphased STR calls still feed the total channel.

## Reading the results

### The per-gene table

`hapmixqtl_cis.tsv.gz` has one row per tested gene (`phenotype_id`); every
column is defined in `docs/outputs.md` under "Mode `hapmixqtl`". The essentials:

- **Detection** is `pval_perm` or `pval_beta`, the gene-level p-values
  computed against the permutation null. Those are the calls.
- **The lead** is the variant with the largest combined |t|. Its `slope`
  (on the log2 allelic fold-change scale, per ALT allele), `slope_se` and
  `pval_nominal` are on the scan's fitted scale with Meier's correction, and
  `pval_nominal` is referred to `dof_nominal`. `pval_nominal` is the best
  of a window, never a gene-level p, and it is anticonservative under a
  donor-record permutation null; the mechanism and rates are in
  `docs/pipeline_rules.md`.
- **Per channel**, `slope_a`/`slope_a_se` and `slope_t`/`slope_t_se` are the
  allelic and total estimates at the lead, and `allelic_admitted` says
  whether the allelic channel entered the statistic (15 or more informative
  allelic donors). `alpha_cis` and `pval_cis_trans` compare the two channels
  at the lead; a small `pval_cis_trans` means they disagree (a trans
  component, mapping bias or phase error), and it is a diagnostic, not a
  filter.
- `tau_a`, `tau_t`, `c_a` and their null-scan counterparts are empty in
  default mode, and that is correct: under `Var(eps) = sigma^2 v` no such
  parameter exists. `tau_refit` is false. These columns, and
  `variance_model`, are carried for compatibility with the deprecated
  configurations and describe nothing in a default run.

### Leave-one-donor-out influence columns

A single donor record can carry a gene-level call: in CALM2, measured on the
pre-correction pipeline on 2026-09-25, `pval_perm` was 0.028 with one donor's
allelic record and 0.684 without it. Every default-mode lead therefore
reports:

- `loo_donor`: the donor whose exclusion from both channels moves the lead's
  combined |t| furthest toward zero.
- `loo_pval_nominal`: the lead's nominal p with that donor excluded.

Each is an exact refit under the default model (checked against
`map_nominal` with the donor masked and against a per-donor loop on 40 genes:
the same donor in 40 of 40, |t| and degrees of freedom within 7.8e-5
relative). Read them as a diagnostic, never a filter, within three limits:
the lead is held fixed, although excluding the donor can move it (as in
CALM2); `pval_perm` is not recomputed; and a donor that alone identifies a
covariate level is not evaluated. A gene whose call rests on one donor shows
a `loo_pval_nominal` far above its `pval_nominal`; whether that donor's record
is an error is a separate question, answered against alignment-based allele
counts.

### Genes the run reports but does not test

The runner prints how many genes pass the eQTL gene filter but have no Gibbs
draws, because no donor has a haplotype-paired transcript for them. On this
deployment that is 1,208 of the 12,955 filtered genes, leaving 11,747
testable (counted against the Gibbs cache, which was built with the runner's
`load_counts`). 1,188 of the 1,208 are all of the filtered genes on chr14,
chr15 and chr22, the three contigs whose phased genotypes are truncated (see
the genotype section); the other 20 lie elsewhere and have no
haplotype-paired transcript in any donor. Whether genes without Gibbs draws
should be tested in the total channel alone is an open decision recorded in
`docs/pipeline_rules.md`. The expression PCs use all 12,955 genes, as the
gene-filter rule requires.

### The evaluation bundle

`eval_bundle.json` holds the run metadata (donor and gene counts, covariate
split, input provenance including the number of donor-gene pairs the
one-sided rule excluded, `mode: default_half_read_split`, `tau_mode`,
`se_mode`, `tau_refit: false`), the reference-bias gate's pooled result, and
summaries of the results. Its `lambda_gc` and QQ curve are computed on each
gene's lead `pval_nominal`, the smallest of a window of correlated tests, so
they are expected to sit far above 1 and are not a calibration measure. Its
`channel_concordance` block regresses `slope_a` on `slope_t` across genes;
both channels estimate the same quantity, so the slope should be near 1, and
a departure localizes bias to one channel.

## The RASQUAL comparison on real data (a comparator, not default mode)

The runner can add RASQUAL, run on the same genes, as a real-data comparator.
These flags do not change the default-mode results, and RASQUAL always sees
the biallelic SNPs only:

```bash
python3 $REPO/scripts/run_hapmixqtl_from_salmon.py <the default-mode flags above> \
    --rasqual /mnt/ssd/lalli/usr/local/rasqual/bin/rasqual \
    --allelic-counts prepped/allelic_counts_manifest.chr.tsv \
    --rasqual-input both --rasqual-genes 200
```

- `--rasqual` is a built RASQUAL binary. `scripts/build_rasqual.sh [DEST]`
  builds one from source (default `rasqual_src/src/rasqual`; it needs GSL,
  LAPACK, BLAS and zlib); the one installed here is the path above.
- `--rasqual-input pseudo` (the default) gives RASQUAL each gene's
  haplotype totals as one pseudo feature SNP, the same information hapmixQTL
  sees; `native` gives it phASER's per-feature-SNP counts and requires
  `--allelic-counts`; `both` runs each, which separates the effect of the
  input from the effect of the method.
- `--rasqual-genes` caps the comparison (default 200 genes), because RASQUAL
  is about a thousand times slower than hapmixQTL.
- RASQUAL's counts are the point-estimate totals, with offsets from the edgeR
  effective library sizes. RASQUAL reports no standard error.

It writes `rasqual_cis.tsv.gz` and adds RASQUAL's fitted phi, delta and theta
distributions and the rank correlation of the two methods' statistics to the
bundle. RASQUAL's phi is an independent estimate of reference mapping bias,
so it cross-checks the gate.

The comparisons the project relies on are made elsewhere: the simulated-effects
benchmark, with simulated effects of known size (`benchmark/simulated_effects/`, run
order `run_all.sh`, described in its `README.md`; pages under
`$DEPLOY/plasmode_meier_20260927/` and `plasmode_lowcov_meier_20260927/`,
which are records of the 2026-09-27 configuration rather than runs of the
current default), and the held-out replication referee
(`scripts/referee_replication.py`, `referee_trecase.py`, `referee_score.py`;
page `$DEPLOY/referee_replication_20260928/report.html`). The one-page
summary is `$DEPLOY/benchmark_summary_20260929/summary.html`.

## Running mixQTL mode, the no-draws comparator

`scripts/compare_mixqtl_replication.py` has no command-line flags. Its paths
are written into the script, and two environment variables control a run:

```bash
MIXQTL_OUT=$DEPLOY/<new output folder> NP=40 python3 $REPO/scripts/compare_mixqtl_replication.py
```

It runs on the 29 calibration genes (`$DEPLOY/pilot29_hc.txt`, variants from
`deprecated_models/null_calibration_29b/regions.bed`, those genes' windows)
with the point estimates and edgeR library sizes above, the current
covariates (`cov/half_read_point_calibration_20260930/`, genotype PCs tied to
the genotypes) and `NP` null permutations (default 40). It writes three
analyses: `weighting_ablation.tsv` (hapmixQTL's response and donor set held
fixed while only the weights vary, so the spread of slopes across null
permutations is each weighting's estimation error), `residual_floor_profile.tsv`,
and the end-to-end mixQTL scan `endtoend_mixqtl_observed.tsv` and
`endtoend_mixqtl_nulls.tsv`, with `summary.json`.

Set `MIXQTL_OUT` to a fresh folder. The default,
`$DEPLOY/mixqtl_replication_point_estimates_20260925/`, holds the stored run,
which used the 2026-09-25 covariate build in `log2(CPM + 1)`; a run into it
would overwrite that record. There is no transcriptome-wide mixQTL driver; a
wider run calls `mixqtl_scan` in `tensorqtl/mixqtl_replication.py` per gene
the way this driver's `mixqtl_gene` does, permuting donor records for the
null while the genotype PCs stay with the genotypes.

## Naming traps that empty a join

Conventions that must agree across separately built files fail as empty joins
rather than errors.

**Ensembl version suffixes.** Ensembl identifiers carry versions
(`ENSG00000123456.7`). Salmon transcript names usually keep them and
published eGene lists usually drop them; a mismatch makes `tx2gene` pair zero
transcripts. Pick one convention and apply it everywhere: either pass
`--strip-version` to `gtf_to_tables.py` and strip the Salmon names and any
eGene list too, or keep versions on all of them. `--strip-version` affects only
the tables `gtf_to_tables.py` writes. This deployment uses RefSeq accessions,
whose version suffixes match between Salmon and the annotation.

**Chromosome names.** `chr1`, `1` and `NC_060925.1` are different strings.
`gtf_to_tables.py` emits names exactly as the GTF has them and prints the
first few. The runner refuses when the VCF and `--gene-pos` share no
chromosome name, but a partial mismatch only drops the unmatched genes, with
a printed count (`dropping N phenotypes on chrs. without genotypes`).
An NCBI RefSeq GTF without UCSC renaming (for example
`genome_refs/GRCh38_p14_ncbi110/GCF_000001405.40_GRCh38.p14_genomic.gtf.gz`)
names chromosomes by accession and genes by symbol, so it joins to neither a
`chr`-named VCF nor an Ensembl-keyed gene list.

**Salmon names against `tx2gene`.** The runner strips the haplotype suffix
before looking a transcript up, so `tx2gene.tsv` must list base transcript
identifiers without `_L`/`_R`. If no haplotype-paired transcript matches, the
runner stops with `no haplotype-paired transcript matched --tx2gene`.

## Historical procedures

These are recorded so their results can be read; none of them is run as part
of this procedure.

- **The RASQUAL head-to-head driver, `compare_pipelines.py`**, removed
  2026-10-01. Its hapmixQTL arm used the pre-correction pipeline (a
  natural-log phenotype from Gibbs posterior means, the estimated-`tau`
  model with a lead refit, and the known-variance second pass). The script is
  archived with its SHA-256 in `$DEPLOY/retired_scripts_20261001/`, whose
  README says how to rerun it. Its runs stay where they are: the pilot series
  (`$DEPLOY/pilot*`), `deprecated_models/null_calibration_29b/`,
  `deprecated_models/final30_matched_scale/`, and
  `rasqual_default_mode_20260923/`. The design write-up is
  `rasqual_comparison_design_20260923/rasqual_comparison.html`. The narratives
  this runbook carried about those pilots until 2026-10-01 (counting noise on
  low-count genes, per-channel covariates, the estimated-`tau` model and its
  lead refit, effects at matched variants, the 29-gene null calibration) are
  in this file's git history; their numbers describe that pipeline, not
  default mode.
- **Deprecated variance models** (`variance_model`, `variance_prior`,
  `tau_mode='estimate'`, the known-variance `se_mode='model'`), quarantined
  2026-09-23: code `tensorqtl/fitted_variance.py`, records
  `$DEPLOY/deprecated_models/README.md`.
- **Fine-mapping.** `map_susie` has no per-channel residual scale and refuses
  `tau_mode='zero'`, and `--mode hapmixqtl_susie` was removed from the
  package CLI. Its `tau_mode='estimate'` default remains only to reproduce
  earlier results; credible sets and PIPs were never validated.
- **The STR-curvature and multi-allelic categorical second pass** refuses
  default mode and supports only the known-variance standard error.
- **The lead refit (`tau_refit`)** has nothing to refit in default mode. The
  runner no longer passes it, and the package CLI's `--tau_refit` is a
  compatibility flag with no effect.
- **Earlier covariate builds**, which the runner refuses:
  `cov/covariates.tsv` (pre-2026-09-25; `log1p` of raw counts, genotype PCs
  from another VCF snapshot) and `cov/log2cpm1_point_calibration_20260925/`
  (expression PCs in `log2(CPM + 1)`, the build behind every result stored
  before 2026-09-30).

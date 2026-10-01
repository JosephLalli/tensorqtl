"""Alignment-based integer counts for the 92 donors: gene totals (featureCounts) and haplotype counts (phASER).

WHY. The competitor arms (TReCASE, RASQUAL) are built for integer read counts, while the benchmark so far fed them
Salmon point estimates. This writes their native inputs from the same STAR alignments, for every gene of the eQTL
gene filter (the pool of benchmark/simulated_effects/select_stratum_genes.py: cache genes in calibration_genes.txt), which
covers the 199 genes of the two benchmark gene sets.

TOTALS. featureCounts on cohort/bams.tsv (STAR 2.7.10a to T2T-CHM13v2.0; contigs NC_060925.1..., chrM) with the
annotation of the personalized Salmon run that produced the Gibbs cache (its params_*.json 'gtf'; annot/genes.tsv
and annot/tx2gene.tsv were built from it, so gene ids are ours), exons as a SAF with chr contigs renamed by
vcf/chr2nc.tsv after checking every contig's length against the BAM header. Fragments (-p --countReadPairs),
reverse stranded (-s 2), primary alignments; multi-mapping fragments (NH > 1) and fragments overlapping exons of
two genes are not counted (featureCounts defaults); duplicates are counted, as Salmon counted them (--ignoreDup
would match phASER instead). One run per BAM, checkpointed. Gates, per donor: the summary reconciles with STAR's
own counts (unmapped = input - unique - multimapped; assigned + no-feature + ambiguity = uniquely mapped, to
RECONCILE_TOL), and assigned fragments are within ASSIGNED_RATIO of the pipeline's own featureCounts on the same
libraries (nf_results, gene-biotype counting).

HAPLOTYPES. scripts/phaser_stranded.py gene counts (aCount, bCount) per donor, rows in annot/genes.NC.bed order
(checked): phASER per transcript strand on strand-split BAMs, each gene counted from the heterozygous SNVs its own
exons hold on its own strand (GTEx-style collapsed model, scripts/phaser_features.py; HLA and CHM13-inaccessible
regions blacklisted; reads WASP-filtered, scripts/phaser_wasp.py, and phASER's alignment-score
cutoff off). A gene with gw_phased = 1 sums the genome-wide phased haplotype blocks, A = the VCF's
first allele; gw_phased = 0 is phASER's single best-covered block whose A/B labels are not anchored to the VCF
(phaser_gene_ae's rule), and is written as a = b = 0 (no VCF-oriented allelic information; its fragments stay in
the total). The orientation is verified on a sample of genes by summing phASER's per-SNP counts, from the gene's
own strand, by the analysis VCF's phase.

REMAINDER. U = total - a - b must be non-negative for thinning. The two counts are close to nested: both take a
gene's fragments on its own strand, phASER at SNVs in exon stretches no other same-strand gene shares,
featureCounts over the gene's exons unless the fragment also overlaps another gene's. phASER's filters (MAPQ 255 =
STAR unique, base quality 10, proper pairs, duplicates dropped) all make a + b smaller. Rule: where a + b > total,
a = b = 0 and the total is kept; how often, and in which kind of gene (span overlapping another gene, listed
variant outside the merged exons), is recorded.

Output (OUT): totals.parquet, hap_a.parquet, hap_b.parquet (pool genes x the 92 donors in the Gibbs cache's
samples.txt order, int64), orientation.json, facts.json; featurecounts/<donor>.txt[.summary]; exons.saf.
"""
import concurrent.futures as cf
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

D = Path('/mnt/ssd/lalli/brainvar_hapmix_deploy')
OUT = D / 'native_counts_wasp_20260928'   # native_counts_stranded_20260928: the same without WASP; native_counts_20260928: unstranded spans
BAMS = D / 'cohort' / 'bams.tsv'                        # DNA library id -> BAM (metadata v1.4 pairing)
PAIRING = D / 'cohort' / 'pairing.tsv'                  # dna_library, rna_library, bam stem (HSBxxx)
CACHE = D / 'cache' / 'gibbs_56b63c3b37ed5df8'          # samples.txt (donor order), genes.txt
CAL = CACHE / 'point_estimates' / 'edger' / 'calibration_genes.txt'   # the eQTL gene filter
GENE_SETS = (D / 'corrected_null_store_20260925' / 'genes.txt', D / 'plasmode_stratum30_100_20260927' / 'gene_set' / 'genes.txt')
GENES_TSV = D / 'annot' / 'genes.tsv'                   # gene, chr, start, end, tss (1-based)
GENES_NC_BED = D / 'annot' / 'genes.NC.bed'             # phaser_gene_ae.py --features
EXONS_TSV = D / 'annot' / 'exons.tsv'                   # gene, merged exon starts, ends (1-based)
GTF = Path('/mnt/ssd/lalli/nf_stage/genome_refs/T2T-CHM13_v2_ncbi110/'
           'GCF_009914755.1_RS_2024_08-T2T-CHM13v2.0_genomic.UCSC_chr.with_GRCh38_rCDS_chrM.exon_ids.gtf')   # personalized_T2T_NCBI110_pseudoalignment/pipeline_info/params_*.json
CHR2NC = D / 'vcf' / 'chr2nc.tsv'                       # chr -> RefSeq accession, as phASER's VCF was renamed
FAI = Path('/mnt/ssd/lalli/nf_stage/genome_refs/T2T-CHM13_v2_ncbi110/chm13v2.0_maskedY_rCRS.fasta.fai')   # chr names, lengths
RNA_RUN = Path('/mnt/data/lalli/nf_stage/reference_comparison_results_RNA/T2T_NCBI110')   # the BAMs' nf-core/rnaseq 3.8.1 run
EARLIER_FC = Path('/mnt/data/lalli/nf_stage/genome_refs/T2T-CHM13_v2_ncbi110/nf_results/multiqc_data')   # the pipeline's own featureCounts and RSeQC
FEATURECOUNTS = '/opt/subread/featureCounts'            # v2.0.8
STRAND = 2                                              # Salmon libType ISR; RSeQC infer_experiment pe_antisense ~0.97; samplesheet 'reverse'
N_JOBS, THREADS = 8, 4                                  # 32 processes (task cap)
RECONCILE_TOL = 0.005        # featureCounts summary vs STAR's own counts, relative (100_D1: unmapped exact, unique +0.054%)
ASSIGNED_RATIO = (0.9, 1.1)  # assigned / the pipeline's own featureCounts assigned, same library: annotation and gene-vs-biotype
                             # ambiguity move it by a few percent; a wrong strand (-s 1) would assign ~pe_sense/pe_antisense ~2%
PHASER = D / 'phaser_stranded_wasp_20260928'            # gene_ae/<donor>.gene_ae.txt, phaser/<donor>.<plus|minus>.allelic_counts.txt
ANALYSIS_VCF = D / 'prepped' / 'analysis.snps.maf01.vcf.gz'   # the phase xL/xR the benchmark uses (compare_mixqtl_replication.load_inputs)
PHASER_VCF = D / 'vcf' / 'cohort92.phaser_input.NC.vcf.gz'   # the VCF phASER ran against (scripts/phaser_input_vcf.py)
PHASER_STRAND = D / 'phaser_inputs_20260928' / 'nesting.tsv'  # gene strand, as phaser_stranded.py assigns genes to runs
SEED, ORIENT_KEY, N_ORIENT = 42, 1, 500                 # orientation sample: the benchmark genes plus N_ORIENT random pool genes
CLEAR = (20, 0.2)                                       # 'clear imbalance' for the sign check: a + b >= 20 and |a - b| >= 0.2 (a + b)
REF_SHARE_MIN_READS = 20                                # reference-allele share at sites with at least 20 reads (task 2026-09-28)
TMP = OUT / 'tmp'                                       # featureCounts --tmpDir, on the SSD


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def write_atomic(path, write, mode='w'):
    tmp = path.with_name(path.name + '.tmp')
    with open(tmp, mode) as fh:
        write(fh)
    os.replace(tmp, path)


def donors():
    """The cache's donor order and each donor's BAM, keyed on the DNA library id."""
    order = (CACHE / 'samples.txt').read_text().split()
    bams = pd.read_csv(BAMS, sep='\t', header=None, names=['dna', 'bam']).set_index('dna').bam
    pair = pd.read_csv(PAIRING, sep='\t').set_index('dna_library')
    if sorted(bams.index) != sorted(order) or sorted(pair.index) != sorted(order):
        raise SystemExit(f'{BAMS} or {PAIRING} donors differ from {CACHE}/samples.txt')
    stem = bams.map(lambda p: Path(p).name.split('.')[0])
    if not (stem == pair.bam.reindex(stem.index)).all():
        raise SystemExit(f'BAM file names in {BAMS} differ from {PAIRING}')
    missing = [b for b in bams if not Path(b).exists()]
    if missing:
        raise SystemExit(f'{len(missing)} BAMs missing, e.g. {missing[:2]}')
    print(f'{len(order)} donors in {CACHE}/samples.txt; {len(bams)} BAMs in {BAMS}, all present, stems match {PAIRING}')
    return order, bams.loc[order], stem.loc[order]


def contig_map(bams):
    """chr -> NC, each checked by length against every BAM header (the runbook's recipe)."""
    c2n = dict(pd.read_csv(CHR2NC, sep='\t', header=None).values)
    fai = dict(pd.read_csv(FAI, sep='\t', header=None, usecols=[0, 1]).values)
    headers = set()
    for b in bams:
        h = subprocess.run(['samtools', 'view', '-H', b], capture_output=True, text=True, check=True).stdout
        headers.add(tuple(tuple(x[3:] for x in l.split('\t')[1:3]) for l in h.splitlines() if l.startswith('@SQ')))   # (SN, LN)
    if len(headers) != 1:
        raise SystemExit(f'{len(headers)} distinct @SQ sets among the {len(bams)} BAMs')
    sq = {n: int(l) for n, l in headers.pop()}
    bad = [c for c in fai if c2n.get(c) not in sq or sq[c2n[c]] != fai[c]]
    if bad or len(sq) != len(fai):
        raise SystemExit(f'contig map fails the BAM header: {bad} ({len(sq)} BAM contigs, {len(fai)} reference)')
    print(f'contigs: {len(sq)} in every BAM header (identical across {len(bams)} BAMs); all {len(fai)} chr names map by '
          f'{CHR2NC} to a BAM contig of the same length')
    return c2n


def build_saf(c2n):
    """Exon lines of GTF as SAF (GeneID Chr Start End Strand), contigs renamed to the BAM's."""
    path = OUT / 'exons.saf'
    if path.exists():
        return path
    rows = []
    with open(GTF) as fh:
        for line in fh:
            if line.startswith('#'):
                continue
            f = line.split('\t', 9)
            if f[2] != 'exon':
                continue
            gid = f[8].split('gene_id "', 1)[1].split('"', 1)[0]
            rows.append((gid, c2n[f[0]], f[3], f[4], f[6]))
    saf = pd.DataFrame(rows, columns=['GeneID', 'Chr', 'Start', 'End', 'Strand'])
    exon_genes = pd.read_csv(EXONS_TSV, sep='\t', header=None, usecols=[0]).iloc[:, 0]
    if set(saf.GeneID) != set(exon_genes):
        raise SystemExit(f'{GTF} exon gene ids differ from {EXONS_TSV}')
    write_atomic(path, lambda fh: saf.to_csv(fh, sep='\t', index=False))
    print(f'{len(saf):,} exon lines of {saf.GeneID.nunique():,} genes from {GTF.name} (= the genes of {EXONS_TSV}; '
          f'the other genes of {GENES_TSV.name} have a gene line and no exon) -> {path}')
    return path


def count_bam(dna, bam, saf):
    """One featureCounts run; skipped when its summary exists (checkpoint)."""
    out = OUT / 'featurecounts' / f'{dna}.txt'
    if Path(f'{out}.summary').exists():
        return dna, 0.0
    tmp = out.with_name(f'{dna}.partial.txt')
    t = time.time()
    subprocess.run([FEATURECOUNTS, '-a', str(saf), '-F', 'SAF', '-o', str(tmp), '-p', '--countReadPairs', '-s', str(STRAND),
                    '--primary', '-T', str(THREADS), '--tmpDir', str(TMP), bam],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    os.replace(tmp, out)
    os.replace(f'{tmp}.summary', f'{out}.summary')
    return dna, time.time() - t


def run_featurecounts(order, bams, saf):
    (OUT / 'featurecounts').mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)
    todo = [d for d in order if not (OUT / 'featurecounts' / f'{d}.txt.summary').exists()]
    log(f'featureCounts: {len(order) - len(todo)} of {len(order)} BAMs done; running {len(todo)} at {N_JOBS} x {THREADS} threads')
    with cf.ThreadPoolExecutor(N_JOBS) as ex:
        for i, (d, s) in enumerate(ex.map(lambda d: count_bam(d, bams[d], saf), todo), 1):
            log(f'  {d} {s:.0f} s ({i}/{len(todo)})')


F = {}   # facts.json: every count this script prints


def pool_genes():
    """The eQTL gene filter's genes with Gibbs draws (select_stratum_genes.load_pool), in cache order."""
    genes = (CACHE / 'genes.txt').read_text().split()
    cal = set(CAL.read_text().split())
    pool = [g for g in genes if g in cal]
    bench = sorted(set().union(*(p.read_text().split() for p in GENE_SETS)))
    off = sorted(set(bench) - set(pool))
    if off:
        raise SystemExit(f'{len(off)} benchmark genes are not in the pool: {off[:5]}')
    F.update(n_cache_genes=len(genes), n_filter_genes=len(cal), n_pool_genes=len(pool), n_benchmark_genes=len(bench))
    print(f'pool: {len(pool):,} of {len(genes):,} cache genes pass the eQTL gene filter ({len(cal):,} genes in {CAL.name}); '
          f'the {len(bench)} genes of the two benchmark gene sets are all in it')
    return pool, bench


def library_facts(stem):
    """Strandedness and pairing as the pipeline recorded them, and its duplicate rate."""
    lib = {json.loads((RNA_RUN / 'star_salmon' / s / 'cmd_info.json').read_text())['libType'] for s in stem}
    ie = pd.read_csv(EARLIER_FC / 'multiqc_rseqc_infer_experiment.txt', sep='\t', index_col=0).reindex(stem.values)
    sheet = pd.read_csv(RNA_RUN / 'QC_reports' / 'nextflow_log' / 'samplesheet.valid.csv')
    sheet = sheet.assign(stem=sheet['sample'].str.replace('_T1', '', regex=False)).set_index('stem').reindex(stem.values)
    gs = pd.read_csv(RNA_RUN / 'multiqc/star_salmon/multiqc_data/multiqc_general_stats.txt', sep='\t', index_col=0)
    dup = gs['Picard_mqc-generalstats-picard-PERCENT_DUPLICATION'].reindex(stem.values)
    if ie.isna().any().any() or sheet[['single_end', 'strandedness']].isna().any().any() or dup.isna().any():
        raise SystemExit('a donor lacks RSeQC infer_experiment, a samplesheet row or a Picard duplication rate')
    F.update(salmon_libtype=sorted(lib), rseqc_pe_antisense_min=float(ie.pe_antisense.min()),
             rseqc_pe_sense_max=float(ie.pe_sense.max()), samplesheet_strandedness=sorted(set(sheet.strandedness)),
             samplesheet_single_end=sorted(int(x) for x in set(sheet.single_end)),
             picard_duplication_median=float(dup.median()), picard_duplication_range=[float(dup.min()), float(dup.max())])
    print(f'library: Salmon libType {sorted(lib)} (as specified, all {len(stem)} runs); RSeQC infer_experiment (earlier run of '
          f'these libraries) pe_antisense min {ie.pe_antisense.min():.4f}, pe_sense max {ie.pe_sense.max():.4f}; samplesheet '
          f'strandedness {sorted(set(sheet.strandedness))}, single_end {sorted(set(sheet.single_end))} -> featureCounts -p '
          f'--countReadPairs -s {STRAND}. Picard duplication median {dup.median():.3f} (range {dup.min():.3f}-{dup.max():.3f}): '
          'counted here, as Salmon counted them; phASER drops them (-F 0x400)')


def read_totals(order, stem, pool):
    """Pool genes x donors fragment counts; the per-donor assignment summary, reconciled and gated."""
    fc = OUT / 'featurecounts'
    partial = sorted(fc.glob('*.partial.txt*'))
    done = [d for d in order if (fc / f'{d}.txt.summary').exists()]
    if partial or len(done) != len(order):
        raise SystemExit(f'featureCounts incomplete: {len(done)} of {len(order)} summaries, {len(partial)} partial files')
    counts = pd.DataFrame({d: pd.read_csv(fc / f'{d}.txt', sep='\t', comment='#', index_col=0).iloc[:, -1] for d in order})
    S = pd.DataFrame({d: pd.read_csv(fc / f'{d}.txt.summary', sep='\t', index_col=0).iloc[:, 0] for d in order}).T
    no_exon = [g for g in pool if g not in counts.index]
    if no_exon:
        raise SystemExit(f'{len(no_exon)} pool genes have no exon in the SAF: {no_exon[:5]}')
    star = pd.read_csv(RNA_RUN / 'multiqc/star_salmon/multiqc_data/multiqc_star.txt', sep='\t', index_col=0).reindex(stem.values)
    star.index = order
    other = [c for c in S.columns if S[c].sum() > 0 and c not in
             ('Assigned', 'Unassigned_Unmapped', 'Unassigned_MultiMapping', 'Unassigned_NoFeatures', 'Unassigned_Ambiguity')]
    unmapped_gap = S.Unassigned_Unmapped - (star.total_reads - star.uniquely_mapped - star.multimapped)
    unique_gap = (S.Assigned + S.Unassigned_NoFeatures + S.Unassigned_Ambiguity + S[other].sum(1) - star.uniquely_mapped) / star.uniquely_mapped
    share = S.Assigned / star.uniquely_mapped
    print(f'featureCounts, {len(order)} donors: assigned {int(S.Assigned.sum()):,} fragments; per donor assigned share of STAR '
          f'uniquely mapped pairs {share.min():.3f} / {share.median():.3f} / {share.max():.3f} (min / median / max); '
          f'no-feature {(S.Unassigned_NoFeatures / star.uniquely_mapped).median():.3f}, ambiguity '
          f'{(S.Unassigned_Ambiguity / star.uniquely_mapped).median():.4f} (medians); other non-zero statuses {other}')
    print(f'  reconciliation with STAR (multiqc_star.txt of the BAMs\' run): unassigned-unmapped minus (input - unique - '
          f'multimapped) max |gap| {int(unmapped_gap.abs().max()):,} fragments; (assigned + no-feature + ambiguity + other) '
          f'over uniquely mapped - 1 in [{unique_gap.min():.5f}, {unique_gap.max():.5f}]')
    if (unmapped_gap.abs() / star.total_reads).max() > RECONCILE_TOL or unique_gap.abs().max() > RECONCILE_TOL:
        raise SystemExit(f'featureCounts summary does not reconcile with STAR to {RECONCILE_TOL:.1%}')
    # the earlier run's sample names are offset for five libraries (HSB587-593, the relabelling metadata v1.4 encodes),
    # so its library is found by its STAR input pair count, which identifies the FASTQ, never by the name
    early = pd.read_csv(EARLIER_FC / 'multiqc_featureCounts.txt', sep='\t', index_col=0)
    early_star = pd.read_csv(EARLIER_FC / 'multiqc_star.txt', sep='\t', index_col=0)
    by_reads = early_star.total_reads[~early_star.total_reads.duplicated(keep=False)]
    by_reads = pd.Series(by_reads.index, index=by_reads.values)
    match = star.total_reads.map(by_reads)
    name_differs = [(d, s, match[d]) for d, s in zip(order, stem) if pd.notna(match[d]) and match[d] != s]
    m = match.notna().values
    uniq_rel = np.abs(early_star.uniquely_mapped.reindex(match[m]).values / star.uniquely_mapped.values[m] - 1)
    ratio = S.Assigned.values[m] / early.Assigned.reindex(match[m]).values
    salmon = S.Assigned.values / np.array([json.loads((RNA_RUN / 'star_salmon' / s / 'aux_info' / 'meta_info.json').read_text())['num_mapped']
                                           for s in stem])
    print(f'  earlier pipeline run (nf_results, gene-biotype featureCounts), matched by STAR input pair count: {int(m.sum())} of '
          f'{len(order)} donors ({len(name_differs)} under another HSB name there: {name_differs}); uniquely mapped differs by at most '
          f'{uniq_rel.max():.3%}; assigned here / assigned there {ratio.min():.3f} / {np.median(ratio):.3f} / {ratio.max():.3f} '
          f'(gate [{ASSIGNED_RATIO[0]}, {ASSIGNED_RATIO[1]}]); assigned / Salmon num_mapped of the BAMs\' run {salmon.min():.3f} / '
          f'{np.median(salmon):.3f} / {salmon.max():.3f}')
    per = pd.DataFrame(dict(bam=stem.values, assigned=S.Assigned.values, share_of_unique=share.values, over_earlier=np.nan,
                            over_salmon=salmon), index=pd.Index(order, name='donor'))
    per.loc[m, 'over_earlier'] = ratio
    print('  per donor:\n' + per.to_string(float_format=lambda v: f'{v:.3f}'))
    if np.isnan(ratio).any() or ratio.min() < ASSIGNED_RATIO[0] or ratio.max() > ASSIGNED_RATIO[1]:
        raise SystemExit('assigned fragments outside the gate against the pipeline\'s own featureCounts')
    F.update(fc_assigned_total=int(S.Assigned.sum()),
             fc_assigned_share_of_unique=[float(share.min()), float(share.median()), float(share.max())],
             fc_other_statuses=other, fc_unmapped_gap_max=int(unmapped_gap.abs().max()),
             fc_unique_gap_range=[float(unique_gap.min()), float(unique_gap.max())], earlier_run_matched_donors=int(m.sum()),
             earlier_run_name_differs=name_differs, earlier_run_unique_rel_max=float(uniq_rel.max()),
             fc_assigned_over_earlier=[float(ratio.min()), float(np.median(ratio)), float(ratio.max())],
             fc_assigned_over_salmon_num_mapped=[float(salmon.min()), float(np.median(salmon)), float(salmon.max())],
             fc_per_donor={d: {k: int(v) for k, v in S.loc[d].items() if v} for d in order})
    return counts.loc[pool, order].astype(np.int64).rename_axis('gene')


def read_phaser(order, pool, c2n):
    """aCount, bCount, gw_phased and the listed variants, pool genes x donors."""
    bed = pd.read_csv(GENES_NC_BED, sep='\t', header=None, names=['contig', 'start', 'stop', 'name'], dtype={'contig': str})
    gt = pd.read_csv(GENES_TSV, sep='\t', header=None, names=['gene', 'chr', 'start', 'end', 'pos'], dtype={'chr': str})
    if not ((bed.name == gt.gene).all() and (bed.contig == gt.chr.map(c2n)).all() and (bed.start == gt.start - 1).all()
            and (bed.stop == gt.end).all()):
        raise SystemExit(f'{GENES_NC_BED} is not {GENES_TSV} with contigs renamed by {CHR2NC} and 0-based starts')
    rows = bed.name.isin(set(pool)).values
    A, B, W, V = {}, {}, {}, {}
    for d in order:
        t = pd.read_csv(PHASER / 'gene_ae' / f'{d}.gene_ae.txt', sep='\t', dtype={'contig': str, 'variants': str},
                        usecols=['contig', 'start', 'stop', 'name', 'aCount', 'bCount', 'gw_phased', 'variants'])
        if len(t) != len(bed) or not (t[['contig', 'start', 'stop', 'name']].values == bed.values).all():
            raise SystemExit(f'{d}.gene_ae.txt rows differ from {GENES_NC_BED}')
        if not t.gw_phased.isin([0, 1]).all():
            raise SystemExit(f'{d}.gene_ae.txt: gw_phased outside {{0, 1}}')
        A[d], B[d], W[d], V[d] = t.aCount.values[rows], t.bCount.values[rows], t.gw_phased.values[rows], t.variants.fillna('').values[rows]
    idx = pd.Index(bed.name.values[rows], name='gene')
    A, B, W, V = (pd.DataFrame(X, index=idx).loc[pool] for X in (A, B, W, V))
    print(f'phASER gene_ae: {len(order)} donors x {len(bed):,} genes, rows identical to {GENES_NC_BED.name} (= {GENES_TSV.name} '
          f'with contigs by {CHR2NC.name}: phASER names are our gene ids, NC contigs map back to chr); gw_phased in {{0, 1}}; '
          f'{len(pool):,} pool genes kept')
    return A.astype(np.int64), B.astype(np.int64), W.astype(int), V


def vcf_gt(vcf, bed, order, rename=None):
    """(variant id 'NC-pos-ref-alt') x donors GT strings from bcftools query over the BED regions."""
    txt = subprocess.run(['bcftools', 'query', '-R', str(bed), '-s', ','.join(order), '-f', '%CHROM\t%POS\t%REF\t%ALT[\t%GT]\n',
                          str(vcf)], capture_output=True, text=True, check=True).stdout
    t = pd.DataFrame([l.split('\t') for l in txt.splitlines()]).drop_duplicates()   # -R repeats records in overlapping regions
    chrom = t[0].map(rename) if rename else t[0]
    t.index = chrom + '-' + t[1] + '-' + t[2] + '-' + t[3]
    if not t.index.is_unique:
        raise SystemExit(f'{vcf}: a variant id appears in two different records')
    return pd.DataFrame(t.iloc[:, 4:].values, index=t.index, columns=order)


def check_orientation(order, pool, bench, A, B, W, V, c2n):
    """Sum phASER's per-SNP counts over each pair's listed variants by the VCF phase; compare with a, b."""
    rng = np.random.default_rng(np.random.SeedSequence(SEED, spawn_key=(ORIENT_KEY,)))
    rest = [g for g in pool if g not in set(bench)]
    sample = sorted(set(bench) | set(np.array(rest)[rng.permutation(len(rest))[:N_ORIENT]]))
    reg = pd.read_csv(GENES_TSV, sep='\t', header=None, names=['gene', 'chr', 'start', 'end', 'pos'], dtype={'chr': str}).set_index('gene').loc[sample]
    chr_bed, nc_bed = TMP / 'orientation.chr.bed', TMP / 'orientation.nc.bed'
    reg.assign(s=reg.start - 1)[['chr', 's', 'end']].to_csv(chr_bed, sep='\t', header=False, index=False)
    reg.assign(c=reg.chr.map(c2n), s=reg.start - 1)[['c', 's', 'end']].to_csv(nc_bed, sep='\t', header=False, index=False)
    g92 = vcf_gt(PHASER_VCF, nc_bed, order)
    gan = vcf_gt(ANALYSIS_VCF, chr_bed, order, rename=c2n)
    rec = [(g, d, int(W.at[g, d]), int(A.at[g, d]), int(B.at[g, d]), v) for g in sample for d in order
           if A.at[g, d] + B.at[g, d] > 0 for v in V.at[g, d].split(',')]
    E = pd.DataFrame(rec, columns=['gene', 'donor', 'gw', 'a', 'b', 'var'])
    E['run'] = E.gene.map(pd.read_csv(PHASER_STRAND, sep='\t', index_col='gene').strand.map({'+': 'plus', '-': 'minus'}))
    ac = []
    for (d, s), e in E.groupby(['donor', 'run']):
        c = pd.read_csv(PHASER / 'phaser' / f'{d}.{s}.allelic_counts.txt', sep='\t', usecols=['variantID', 'refCount', 'altCount'],
                        index_col=0)
        ac.append(c.reindex(e['var'].values).set_axis(e.index))
    E = E.join(pd.concat(ac))
    E['gt92'] = [g92.at[v, d] if v in g92.index else None for v, d in zip(E['var'], E.donor)]
    E['gtan'] = [gan.at[v, d] if v in gan.index else None for v, d in zip(E['var'], E.donor)]
    het = ('0|1', '1|0')
    both = E.gt92.isin(het) & E.gtan.notna()
    same = both & (E.gt92 == E.gtan)
    flip = both & (E.gtan == E.gt92.str[::-1])
    o = dict(n_genes=len(sample), n_benchmark_genes=len(bench), n_random_pool_genes=len(sample) - len(bench),
             n_pairs_with_reads=int(E[['gene', 'donor']].drop_duplicates().shape[0]), n_listed_variant_records=len(E),
             n_records_without_allelic_count=int(E.refCount.isna().sum()),
             n_records_het_phased_in_phaser_vcf=int(E.gt92.isin(het).sum()), n_records_in_analysis_vcf=int(E.gtan.notna().sum()),
             n_records_not_in_phaser_vcf=int(E.gt92.isna().sum()),
             n_records_not_in_phaser_vcf_multiallelic_id=int((E.gt92.isna() & (E['var'].str.count('-') > 3)).sum()),
             vcf_gt_identical=int(same.sum()), vcf_gt_reversed=int(flip.sum()), vcf_gt_other=int((both & ~same & ~flip).sum()))
    print(f'orientation sample: {len(sample)} genes ({len(bench)} benchmark + {len(sample) - len(bench)} random pool, SeedSequence({SEED}, '
          f'spawn_key=({ORIENT_KEY},))); {o["n_pairs_with_reads"]:,} donor-gene pairs with phASER reads list {len(E):,} variant '
          f'records; {o["n_records_without_allelic_count"]:,} without a per-SNP count; {o["n_records_het_phased_in_phaser_vcf"]:,} '
          f'phased het in {PHASER_VCF.name}; {o["n_records_in_analysis_vcf"]:,} present in {ANALYSIS_VCF.name}; '
          f'{o["n_records_not_in_phaser_vcf"]:,} not found in {PHASER_VCF.name}, {o["n_records_not_in_phaser_vcf_multiallelic_id"]:,} '
          'of them multiallelic ids (phASER joins the ALT alleles with "-", e.g. a spanning deletion "*"; such pairs are left out below)')
    print(f'  GT at records present in both VCFs: identical {o["vcf_gt_identical"]:,}, reversed {o["vcf_gt_reversed"]:,}, '
          f'other {o["vcf_gt_other"]:,}')
    if (both & ~same).any():
        raise SystemExit('the analysis VCF phase differs from the VCF phASER ran against; orientation not established')
    for name, col in (('phaser_vcf', 'gt92'), ('analysis_vcf', 'gtan')):
        ok = E[col].isin(het) & E.refCount.notna()
        first_ref = E[col].str[0] == '0'
        E['xl'] = np.where(ok, np.where(first_ref, E.refCount, E.altCount), 0)
        E['xr'] = np.where(ok, np.where(first_ref, E.altCount, E.refCount), 0)
        P = E.assign(ok=ok).groupby(['gene', 'donor']).agg(gw=('gw', 'first'), a=('a', 'first'), b=('b', 'first'), xl=('xl', 'sum'),
                                                          xr=('xr', 'sum'), n=('var', 'size'), n_ok=('ok', 'sum'))
        full = P.n_ok == P.n
        one = P[full & (P.n == 1)]
        multi = P[full & (P.n > 1) & (P.a != P.b) & (P.xl != P.xr)]
        r = {'pairs_all_listed_variants_phased': int(full.sum())}
        for gw in (1, 0):
            s1, sm = one[one.gw == gw], multi[multi.gw == gw]
            same_sign = np.sign(sm.a - sm.b) == np.sign(sm.xl - sm.xr)
            clear = (sm.a + sm.b >= CLEAR[0]) & ((sm.a - sm.b).abs() >= CLEAR[1] * (sm.a + sm.b))
            r[f'gw{gw}'] = dict(
                single_variant_pairs=len(s1), single_variant_exact=int(((s1.a == s1.xl) & (s1.b == s1.xr)).sum()),
                single_variant_exact_swapped=int(((s1.a == s1.xr) & (s1.b == s1.xl) & (s1.a != s1.b)).sum()),
                multi_variant_pairs_imbalanced=len(sm), multi_variant_same_sign=int(same_sign.sum()),
                multi_variant_clear_imbalance=int(clear.sum()), multi_variant_clear_same_sign=int((same_sign & clear).sum()))
        o[name] = r
        g1, g0 = r['gw1'], r['gw0']
        print(f'  by the {name} phase, pairs whose every listed variant is phased there ({r["pairs_all_listed_variants_phased"]:,}): '
              f'gw_phased = 1, one variant: a = xL and b = xR in {g1["single_variant_exact"]:,} of {g1["single_variant_pairs"]:,} '
              f'(swapped {g1["single_variant_exact_swapped"]:,}); several variants, a != b: sign(a - b) = sign(xL - xR) in '
              f'{g1["multi_variant_same_sign"]:,} of {g1["multi_variant_pairs_imbalanced"]:,} ({g1["multi_variant_clear_same_sign"]:,} of '
              f'{g1["multi_variant_clear_imbalance"]:,} at a + b >= {CLEAR[0]} and |a - b| >= {CLEAR[1]} (a + b)). gw_phased = 0: one variant '
              f'{g0["single_variant_exact"]} of {g0["single_variant_pairs"]} exact, several {g0["multi_variant_same_sign"]} of '
              f'{g0["multi_variant_pairs_imbalanced"]} same sign')
    return o


def reference_share(order):
    """Reference-mapping bias in the allele counts: per donor, the median over phASER heterozygous sites (both strand
    runs) with at least REF_SHARE_MIN_READS reads of refCount / totalCount, at the sites whose refAllele is the VCF REF
    (phASER's variantID is CHROM-POS-REF-ALT and its refAllele the donor's first carried allele; any row where they
    differ is dropped and counted)."""
    med, sites, alt_alt = {}, [], 0
    for d in order:
        ac = pd.concat([pd.read_csv(PHASER / 'phaser' / f'{d}.{s}.allelic_counts.txt', sep='\t',
                                    dtype={'contig': str, 'variantID': str, 'refAllele': str},
                                    usecols=['contig', 'position', 'variantID', 'refAllele', 'refCount', 'totalCount'])
                        for s in ('plus', 'minus')])
        ac = ac[ac.totalCount >= REF_SHARE_MIN_READS]
        f = ac.variantID.str.split('-')
        if not ((f.str[0] == ac.contig) & (f.str[1] == ac.position.astype(str))).all():
            raise SystemExit(f'{d}: a variantID does not start with its contig and position')
        is_ref = (f.str[2] == ac.refAllele).values
        alt_alt += int((~is_ref).sum())
        sites.append(int(is_ref.sum()))
        med[d] = float((ac.refCount[is_ref] / ac.totalCount[is_ref]).median())
    v = np.array(list(med.values()))
    r = dict(statistic='per donor, median over heterozygous sites of refCount / totalCount', min_reads=REF_SHARE_MIN_READS,
             donors=len(order), median_min=float(v.min()), median_max=float(v.max()),
             sites_per_donor=[min(sites), int(np.median(sites)), max(sites)], alt_alt_sites_dropped=alt_alt)
    print(f'reference share: per-donor median of refCount / totalCount at sites with >= {REF_SHARE_MIN_READS} reads: '
          f'{v.min():.4f} to {v.max():.4f}; sites per donor {r["sites_per_donor"]}; {alt_alt} rows with refAllele != REF dropped')
    return r


def overlap_classes(pool):
    """Per pool gene: its span overlaps another annotated gene's span (either strand, as phaser_gene_ae counts), and
    an opposite-strand exon-bearing gene's span (strand from the SAF)."""
    g = pd.read_csv(GENES_TSV, sep='\t', header=None, names=['gene', 'chr', 'start', 'end', 'pos'], dtype={'chr': str})
    strand = pd.read_csv(OUT / 'exons.saf', sep='\t', usecols=['GeneID', 'Strand']).drop_duplicates().set_index('GeneID').Strand
    if not strand.index.is_unique:
        raise SystemExit('a gene has exons on both strands')
    g['strand'] = g.gene.map(strand).fillna('')
    want = set(pool)
    any_ov, opp_ov = {}, {}
    for _, t in g.groupby('chr'):
        s, e, st = t.start.values, t.end.values, t.strand.values
        for i, name in enumerate(t.gene.values):
            if name in want:
                ov = (s <= e[i]) & (e >= s[i])
                ov[i] = False
                any_ov[name] = bool(ov.any())
                opp_ov[name] = bool((ov & (st != st[i]) & (st != '')).any())
    return pd.Series(any_ov).reindex(pool).values, pd.Series(opp_ov).reindex(pool).values


def nonexonic(V, pool):
    """Per pair: some listed variant lies outside the gene's merged exons (annot/exons.tsv, 1-based)."""
    ex = pd.read_csv(EXONS_TSV, sep='\t', header=None, names=['gene', 's', 'e']).set_index('gene')
    out = np.zeros(V.shape, bool)
    for i, g in enumerate(pool):
        s, e = (np.array(ex.at[g, k].split(','), int) for k in ('s', 'e'))
        for j, vs in enumerate(V.loc[g].values):
            if vs:
                pos = np.array([int(v.split('-')[1]) for v in vs.split(',')])
                k = np.searchsorted(s, pos, side='right') - 1
                out[i, j] = ((k < 0) | (pos > e[np.maximum(k, 0)])).any()
    return out


def remainder(T, A, B, W, V, pool):
    """a, b after the two rules; U = total - a - b, how often it is negative and in which kind of gene."""
    gw0 = ((W == 0) & (A + B > 0)).values
    print(f'gw_phased = 0 with reads: {int(gw0.sum()):,} of {gw0.size:,} pool donor-gene pairs, {int((A + B).values[gw0].sum()):,} '
          f'of {int((A + B).values.sum()):,} phASER fragments; written a = b = 0 (A/B not anchored to the VCF phase)')
    a, b = A.where(W == 1, 0), B.where(W == 1, 0)
    has, neg = ((a + b) > 0).values, (T - a - b < 0).values
    ov_any, ov_opp = overlap_classes(pool)
    ovA, ovO = np.repeat(ov_any[:, None], T.shape[1], 1), np.repeat(ov_opp[:, None], T.shape[1], 1)
    nx = nonexonic(V, pool)
    cls = {name: dict(pairs_with_reads=int((has & m).sum()), negative=int((neg & m).sum()))
           for name, m in (('span_overlaps_another_gene', ovA), ('span_overlaps_opposite_strand_gene', ovO),
                           ('listed_variant_outside_exons', nx), ('either', ovA | nx), ('neither', ~ovA & ~nx))}
    ab, t = (a + b).values, T.values
    r = dict(gw_phased_0_pairs=int(gw0.sum()), gw_phased_0_fragments=int((A + B).values[gw0].sum()),
             phaser_fragments=int((A + B).values.sum()), pairs=int(t.size), pairs_with_reads=int(has.sum()), negative=int(neg.sum()),
             negative_genes=int(neg.any(1).sum()), negative_with_total_zero=int((neg & (t == 0)).sum()),
             negative_part_fragments=int((ab - t)[neg].sum()), allelic_fragments_in_negative=int(ab[neg].sum()),
             allelic_fragments=int(ab.sum()), classes=cls, genes_total_zero_all_donors=int((t == 0).all(1).sum()),
             genes_total_zero_all_donors_with_phaser_reads=int(((t == 0).all(1) & has.any(1)).sum()),
             genes_median_total_below_10=int((np.median(t, 1) < 10).sum()))
    print(f'totals: {r["genes_total_zero_all_donors"]:,} pool genes have featureCounts total 0 in every donor '
          f'({r["genes_total_zero_all_donors_with_phaser_reads"]:,} of them with phASER reads: every exon shared with another gene, '
          f'so each fragment is ambiguous); {r["genes_median_total_below_10"]:,} have a median total below 10')
    print(f'U = total - a - b < 0 in {r["negative"]:,} of {r["pairs_with_reads"]:,} pairs with phASER reads '
          f'({r["negative"] / r["pairs_with_reads"]:.2%}; {r["pairs"]:,} pairs in all), in {r["negative_genes"]:,} genes; '
          f'{r["negative_with_total_zero"]:,} of them have total 0; a + b - total sums to {r["negative_part_fragments"]:,}; their a + b '
          f'is {r["allelic_fragments_in_negative"]:,} of {r["allelic_fragments"]:,} allelic fragments')
    for k, v in cls.items():
        print(f'  {k}: {v["negative"]:,} negative of {v["pairs_with_reads"]:,} pairs with reads '
              f'({v["negative"] / max(v["pairs_with_reads"], 1):.2%})')
    lost = pd.Series(np.where(neg, ab, 0).sum(1), index=T.index).sort_values(ascending=False)
    cum = lost.cumsum() / lost.sum()
    r['genes_holding_50pct_of_zeroed'], r['genes_holding_90pct_of_zeroed'] = int((cum < 0.5).sum() + 1), int((cum < 0.9).sum() + 1)
    r['top_zeroed_genes'] = {g: int(v) for g, v in lost.head(20).items()}
    print(f'  concentration: {r["genes_holding_50pct_of_zeroed"]:,} genes hold 50% and {r["genes_holding_90pct_of_zeroed"]:,} hold 90% '
          f'of the {int(lost.sum()):,} zeroed allelic fragments; largest: '
          + ', '.join(f'{g} {v:,}' for g, v in lost.head(10).items()))
    r['gene_sets'] = {}
    for p in GENE_SETS:
        rows = T.index.isin(set(p.read_text().split()))
        h, n, x = has[rows], neg[rows], ab[rows]
        tab = pd.DataFrame(dict(negative_donors=n.sum(1), donors_with_reads=h.sum(1), median_total=np.median(t[rows], 1),
                                median_ab=np.median(x, 1), zeroed_fragments=np.where(n, x, 0).sum(1),
                                total_zero_all=(t[rows] == 0).all(1)), index=T.index[rows])
        tab = tab[tab.negative_donors > 0].sort_values('zeroed_fragments', ascending=False)
        low = sorted(T.index[rows][np.median(t[rows], 1) < 10])
        name = p.parent.parent.name if p.parent.name == 'gene_set' else p.parent.name
        r['gene_sets'][name] = s = dict(genes=int(rows.sum()), pairs_with_reads=int(h.sum()), negative=int(n.sum()),
                                        negative_genes=int(n.any(1).sum()), allelic_fragments=int(x.sum()),
                                        allelic_fragments_in_negative=int(x[n].sum()), genes_median_total_below_10=low,
                                        genes_total_zero_all_donors=sorted(tab.index[tab.total_zero_all]),
                                        affected_genes=json.loads(tab.to_json(orient='index')))
        print(f'  gene set {name}: {s["negative"]:,} negative of {s["pairs_with_reads"]:,} pairs with reads in {s["negative_genes"]} of '
              f'{s["genes"]} genes; {s["allelic_fragments_in_negative"]:,} of {s["allelic_fragments"]:,} allelic fragments in them; '
              f'median total below 10: {low}; affected genes:\n' + tab.to_string(float_format=lambda v: f'{v:.1f}'))
    a, b = a.where(~neg, 0), b.where(~neg, 0)
    print(f'rule: where a + b > total, a = b = 0 and the total is kept; written a + b <= total in every pair: '
          f'{bool(((a + b) <= T).values.all())}')
    return a, b, r


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    order, bams, stem = donors()
    c2n = contig_map(bams)
    saf = build_saf(c2n)
    run_featurecounts(order, bams, saf)
    pool, bench = pool_genes()
    library_facts(stem)
    T = read_totals(order, stem, pool)
    A, B, W, V = read_phaser(order, pool, c2n)
    orient = check_orientation(order, pool, bench, A, B, W, V, c2n)
    a, b, rem = remainder(T, A, B, W, V, pool)
    orient.update(a_is="the haplotype of the analysis VCF's first GT allele (xL); b the second (xR)",
                  rule_gw_phased_0='a = b = 0: no VCF-oriented allelic counts; the fragments stay in the total',
                  rule_negative_remainder='a = b = 0 where a + b > total; the total is kept')
    F.update(annotation_counted=str(GTF),
             annotation_star_sjdb='GCF_009914755.1_T2T-CHM13v2.0_genomic-RS_2023_03_with_chrM.gtf (BAM @PG; file not found on disk)',
             featurecounts_options=f'-F SAF -p --countReadPairs -s {STRAND} --primary', remainder=rem, orientation=orient,
             reference_share=reference_share(order))
    for name, X in (('totals', T), ('hap_a', a), ('hap_b', b)):
        write_atomic(OUT / f'{name}.parquet', lambda fh: X.astype(np.int64).to_parquet(fh), 'wb')
    write_atomic(OUT / 'orientation.json', lambda fh: fh.write(json.dumps(orient, indent=1)))
    write_atomic(OUT / 'facts.json', lambda fh: fh.write(json.dumps(F, indent=1)))
    print(f'wrote totals.parquet, hap_a.parquet, hap_b.parquet ({len(pool):,} genes x {len(order)} donors, samples.txt order), '
          f'orientation.json, facts.json to {OUT}')


if __name__ == '__main__':
    main()

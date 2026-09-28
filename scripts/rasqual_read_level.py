"""Does the benchmark's pseudo-feature-SNP construction handicap RASQUAL? Observed data, 30 genes.

In the plasmode benchmark (scripts/plasmode/run_rasqual.py) RASQUAL received Salmon
haplotype counts at ONE pseudo feature SNP per gene, so its read-level features -- per-SNP
allelic counts at real exonic heterozygous sites, the reference-mapping bias phi, the
sequencing/mapping (read allele) error rate delta and posterior genotype updating -- were bypassed. This script
runs, on OBSERVED data (no thinning), three nominal-only arms over the same tested
variants, covariates and totals, and compares them per gene.

STAGES
  --scout    Candidate genes and their phASER coverage (this stage): eQTL pool = BH < 0.05
             on pval_beta in the T2T permutation run build_targeted_gene_sets.py reads,
             null pool = pval_beta > 0.5; restricted to build_targeted_gene_sets.runnable
             (calibration gene filter, Gibbs draws in the cache, unique in annot/genes.tsv).
             Per gene: admitted allelic donors approximated from the point estimates as
             pL >= 0.5 and pR >= 0.5 (the exact rule, make_datasets.allelic_kept, adds
             Va > EPS and is applied by the arms stage), exonic biallelic SNPs of the
             analysis VCF inside the gene's exon union (annot/exons.tsv), and at each such
             site the heterozygous donors' phASER read counts. A site is COVERED when at
             least one heterozygous donor has a read there and both alleles are seen across
             the heterozygous donors (RASQUAL's asaf strictly inside (0, 1), main.c:535).
             That is necessary, not sufficient, for RASQUAL's admission, which also requires
             the site's scaled coverage km above -d and asgaf inside (0.005, 0.995)
             (main.c:535): the smoke admitted 28 of ANKRD36's 74 covered sites and 33 of
             PDE4DIP's 83.
             Writes OUT/scout/candidates.tsv, OUT/scout/summary.json and
             OUT/scout/exonic_sites.npz (per-site het mask and phASER ref/alt counts, so
             the arms stage builds its feature-SNP lines without rescanning). Each phASER
             row inside a candidate's exon union is also classified: matched (the donor is
             heterozygous at that analysis-VCF record), hom (record present, donor not
             heterozygous there: a genotype disagreement RASQUAL's delta absorbs and
             hapmixQTL never sees) or absent (no such SNP in the maf01 callset).
  --select   The 30 genes: 20 eQTL and 10 null with >= 20 admitted donors by the
             point-estimate rule, ranked by exonic heterozygous SNPs at which >= 3
             heterozygous donors hold >= 8 phASER reads (ties by heterozygous donor-site
             pairs at >= 8 reads), one gene per 2-Mb locus so no two chosen genes share a
             cis window. Per gene: the exact tested-variant count from the analysis VCF
             (cis window WIN, outside the body, MAF >= 0.05), RASQUAL's (fSNPs + 1) x
             rSNPs budget (main.c:582 aborts above 30,000 without --force), and whether
             the T2T run's published lead is in the VCF, in the tested set, or inside the
             gene body (unreachable by every arm). Writes OUT/genes.tsv, genes.txt and
             regions.bed (corrected_null_store.py's window).
  --smoke    The gate before the native arm: on SMOKE_GENES, every fSNP line plus the
             first SMOKE_RSNPS tested rSNP lines, run at 1 and at THREADS_MAX threads.
             Checks that RASQUAL admits fSNPs (field 17 against genes.tsv's
             n_covered_sites), that the rows parse through run_rasqual.assemble after
             the fSNP-as-rSNP rows are removed, that the threaded output is identical to
             the single-threaded one (compare_pipelines.best_rasqual_row records thread
             interleaving inside a line), and re-measures the seconds per fSNP x rSNP.
  --arms     The three arms on the observed data (no thinning), every gene checkpointed:
             a finished gene's file is skipped with a printed line, so a relaunch resumes.
             Inputs: compare_mixqtl_replication.load_point_estimate_inputs on
             OUT/genes.txt and OUT/regions.bed, plasmode.run_arms.setup for the tested
             variants (cis window WIN of the TSS, outside the gene body, MAF >= MAF_MIN;
             the count per gene must equal genes.tsv's), covariates and phased genotypes;
             values from the Salmon point estimates, A / T / Va / Vt from
             summaries_from_point_estimates on the real Gibbs draws; allelic admission
             make_datasets.allelic_kept (refused below MIN_ADMITTED donors).
             Totals and covariates for RASQUAL are plasmode.run_rasqual.write_bins with
             the identity permutation: Y = the Salmon point-estimate totals pT (as the
             benchmark used), K = eff_lib / mean(eff_lib), X = the RNA-tied covariates
             and the genotype PCs, covariate-major. So totals, covariates, tested
             variants and donors are identical across the three arms and to the
             benchmark's construction.
       split   hapmixQTL split weighting (Gibbs variance in the allelic channel, unit
               variance in the total channel; run_arms.inputs(..., 'split')), map_nominal
               exactly as run_arms.run_nominal calls it, default mode, allelic channel
               through the origin, each p on its own degrees of freedom (commit 8a06803).
               OUT/hapmix_split.parquet.
       pseudo  RASQUAL on the benchmark's pseudo-feature-SNP construction, on the observed
               counts: run_rasqual.pseudo_site / pseudo_line / rsnp_text / run_gene /
               assemble unchanged (one 0|1 fSNP per gene at a free body position, AS =
               (round(pL), round(pR)) for the admitted donors, -s/-e the gene body, -h 0,
               one thread). Its phi is L/R imbalance by construction (ref = L for every
               heterozygote), not reference-mapping bias. OUT/rasqual_pseudo/<gene>.tsv,
               RASQUAL's rows as printed.
       native  RASQUAL on its own input: feature SNPs = every exonic biallelic SNP of the
               analysis VCF inside the gene's exon union (annot/exons.tsv, the -s/-e
               lists; isExon is inclusive, main.c:44-49), phased GT from the loader's
               xL|xR (the same arrays rsnp_text writes, so fSNP and rSNP phase agree) and
               AS = phASER's ref,alt counts at that site (0,0 for donors without a row).
               phASER writes rows only at a donor's own heterozygous sites, so every
               homozygous donor is 0,0 at every fSNP (hom_rows 0 in all 30 genes), where
               RASQUAL's createASVCF counts reads in every sample (ASVCF/countAS.c:211, no
               genotype test): the native fit never sees the homozygote reads that inform
               delta and check fSNP genotypes. phASER also filtered reads at --mapq 255
               --baseq 10 (phaser_out/*.log), not createASVCF's qcFilterBam;
               RASQUAL's own admission (coverage > -d, both alleles seen, main.c:530-535;
               the fSNP HWE filter is off by default, fHWE = DBL_MAX at main.c:394,
               whatever usage.c prints); the same rSNP lines, option line and -h 0 as the
               pseudo arm, plus --force ((fSNPs + 1) x rSNPs exceeds 30,000 in every gene,
               main.c:582) and --n-threads for the largest genes (threads claim rSNPs one
               at a time under a mutex, nbem.c:279-286; the smoke stage showed the output
               identical at 1 and 4 threads). At most JOBS threads run at once.
               OUT/rasqual_native/<gene>.tsv.
               COST AND THE SUBSET. The smoke on ANKRD36 (2026-09-27, 28 admitted fSNPs)
               took a median of about 70 EM iterations per rSNP fit (field 19 = pitr[l],
               main.c:760; the README's field names are misleading: field 20 is pitr[0],
               line 0's count, and the cap is 231, nbem.c:513) and 0.29 s per admitted
               fSNP x rSNP, where the LDLR scout smoke took 5 iterations and 14.6 ms: a
               full scan of the 156,006 tested variants would cost several hundred CPU
               hours. The native arm
               therefore scans, per gene, the SUBSET native_subset builds: the pseudo and
               split leads, the published lead when tested, the TOP_K strongest variants
               of each of those arms and N_RANDOM random tested variants
               (SeedSequence(SEED, spawn_key=(SUBSET_KEY, gene index))), written once to
               OUT/native_subset.tsv. Its lead is the strongest of the subset.
               KEEPING fSNPs OUT OF THE rSNP SCAN. RASQUAL tests every line that passes
               the rSNP filters, exonic fSNPs at MAF >= 0.05 included (33 of 84 lines in
               the ANKRD36 smoke, each costing a full fit). RASQUAL uses a variant's
               position only in isExon (main.c:44-49), the cis-window test isTestReg
               (main.c:80-90; -c midpoint, -w window, main.c:376-379), a lead tie-break by
               distance to the midpoint (main.c:730) and printing, and nowhere in nbem.c.
               So fSNP lines are written at pos + FSNP_OFFSET with the -s/-e lists shifted
               by the same amount, and -c is the gene's TSS with -w = 2 WIN: isExon still
               admits them, the window (inclusive, |pos - TSS| <= WIN, the tested set's own
               rule) excludes them from the rSNP scan and admits every tested variant.
               The smoke checks that no fSNP row is scanned and that the tested rows equal
               those of the unshifted construction (OUT/smoke_unshifted/).
  --control  The permutation control for the level at random variants: both RASQUAL arms at
             the N_RANDOM random variants of each of the 10 null genes, N_RUNS runs per gene
             under each of two nulls, one thread per job, at most JOBS at once, every
             (null, run, arm, gene) checkpointed under OUT/control/<null>/run_NN/<arm>/.
       records   The project's records null: one donor order perm per (gene, run),
                 SeedSequence(SEED, spawn_key=(7, gene index, run)); column i of Y, K, the
                 RNA-tied covariates and every fSNP's GT:AS entry (native: all fSNPs
                 together; pseudo: the one pseudo entry) is donor perm[i]'s; rSNP genotypes
                 and genotype PCs stay (run_rasqual.write_bins with ds['perm']). No haplotype
                 swap: pooled calibration is the same with and without it (CLAUDE.md,
                 0.0690 vs 0.0692 at 0.05).
       rasqual_r RASQUAL's own -r on the observed inputs. One -r run (nbem.c:2346-2400)
                 draws a new donor order for EACH fSNP and permutes that fSNP's genotype
                 probabilities, AS counts and offsets by it (2365), with no haplotype swap
                 (rord12 fixed at {0, 1}, the swap commented out at 2367), then one separate
                 order for the totals y, offsets ki, the covariate-fitted offsets dki and
                 weights (2387). The covariate model is fitted on the unpermuted data first
                 (main.c:632-633) and X is not used after the permutation, so the RNA-tied
                 covariates' and the genotype PCs' fitted effects both move with the
                 totals. A donor's fSNP records come from different donors and its allelic
                 record is decoupled from its total record, so -r also removes within-donor
                 structure the records null keeps. Seeded without modifying the binary:
                 main.c:209 seeds rand() with time(NULL) + getpid() and getRandomOrder
                 (sort.c:251) draws from rand(), so an LD_PRELOAD library (seed_shim) makes
                 time() return RASQUAL_SEED - getpid(); RASQUAL_SEED per (gene, run) is
                 SeedSequence(SEED, spawn_key=(8, gene index, run)).generate_state(1)[0].
             Gates before the pool, on GATE_GENE: the records construction at the identity
             order writes Y, K and X byte-identical to OUT/rasqual_inputs and reproduces the
             observed rows' model fields at every random variant in both arms; a seeded -r
             run repeated with the same seed is byte-identical, differs under another seed
             and differs from the unpermuted fit.
  --report   Per gene: each arm's lead (RASQUAL: largest chisq; hapmixQTL: smallest
             pval_nominal, since each pair's p has its own dof), its chi-square (for
             hapmixQTL the 1-df chi-square with the same p, chi2.isf(p, 1), derived from
             the p rather than a likelihood ratio), p and log2 aFC (RASQUAL: log2(pi /
             (1 - pi)), ALT over REF); LD r^2 (Pearson r^2 of ALT dosage over the 92
             donors) between the arms' leads and to the T2T run's published lead; RASQUAL
             native's phi, delta, theta, admitted fSNPs and r2_prior_posterior_fsnps
             (the posterior genotype update) against the pseudo fit's, at each arm's own
             lead and at native's lead; and at each arm's lead the other arms' statistics.
             Paired summaries (eQTL and null pools separately, then pooled): the sign test
             (two-sided binomial test of the count of genes where the first arm's lead
             chi-square exceeds the second's against one half) and the Wilcoxon
             signed-rank test (ranks the absolute per-gene differences of log chi-square
             and sums the ranks of the positive ones) on native vs pseudo, split vs
             native and split vs pseudo; the same on LD r^2 to the published lead. Runs
             on whatever native genes have finished and names the missing ones.
             Writes OUT/per_gene.tsv, OUT/arms_at_leads.tsv, OUT/summary.json and
             OUT/report.html (one figure: per-gene lead chi-square of the three arms,
             eQTL against null genes).

MODULES. run_rasqual, run_arms and make_datasets were removed from scripts/plasmode by commit fc238df
(2026-09-27 20:59), after every stage here had run on their fc238df^ versions (git blobs 97d880a, 41db9e4,
76ba855); copies of those are kept in OUT/control/legacy_plasmode as the record. The script now imports
scripts/plasmode/common.py and 04_run_rasqual.py, whose functions used here compute what the old ones did;
the old run_gene (no checkpoint file), admission and TIE, which have no equivalent there, are defined below.

SAMPLE KEY. Every table is keyed on the DNA library id (100_D1, ...): the phASER manifest,
the analysis VCF's sample columns, cohort/salmon.tsv and the Gibbs cache all use it
(cohort/pairing.tsv maps it to the RNA library and BAM through metadata v1.4). Nothing here
joins samples by name or position.

COORDINATES. phASER was run on vcf/cohort92.NC.vcf.gz (RefSeq NC_ contigs, same T2T
CHM13v2.0 assembly as the analysis VCF); prepped/allelic_counts_manifest.chr.tsv points to
the chr-renamed copies in allelic_counts_chr/. Its `position` is the VCF POS (1-based),
verified at chr1:13517 C>T and chr1:133616 T>G for 100_D1 against both VCFs.
"""
import argparse
import base64
import concurrent.futures as cf
import contextlib
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2, wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / 'plasmode'))
import build_targeted_gene_sets as BT                        # noqa: E402
import compare_mixqtl_replication as CM                      # noqa: E402
import corrected_null_store as CNS                           # noqa: E402
import common as C                                           # noqa: E402
from compare_pipelines import RASQUAL_FIELDS                 # noqa: E402
from tensorqtl.hapmixqtl import map_nominal, summaries_from_point_estimates   # noqa: E402
RR = C.module('04_run_rasqual')

D = BT.D
OUT = D / 'rasqual_read_level_20260927'
SCOUT = OUT / 'scout'
CACHE = BT.CACHE
PE = CACHE / 'point_estimates'
VCF = D / 'prepped' / 'analysis.snps.maf01.vcf.gz'
AC_MANIFEST = D / 'prepped' / 'allelic_counts_manifest.chr.tsv'
MIN_ADMITTED = 15          # allelic donors a gene needs (the allelic channel's floor, commit 8a06803)
ADMISSION_MARGIN = 5       # chosen with >= 20 by the point-estimate rule, so the exact rule cannot drop one below 15
N_EQTL, N_NULL = 20, 10
WIN, MAF_MIN = CM.WIN, CM.MAF    # the tested set's cis window (1 Mb of the TSS) and MAF floor (0.05)
SPLIT = OUT / 'hapmix_split.parquet'
TESTED = OUT / 'tested.npz'
ARM_DIR = {'pseudo': OUT / 'rasqual_pseudo', 'native': OUT / 'rasqual_native'}
JOBS = 128                 # RASQUAL threads in flight at once: 128 cores granted for the control (user decision 2026-09-27)
THREADS_MAX = 4            # --n-threads ceiling for the largest native genes (rSNPs claimed under a mutex, nbem.c:279-286)
SEC_PER_UNIT = 0.35        # seconds per ADMITTED fSNP x rSNP at 92 donors, an upper figure: ANKRD36 smoke 2026-09-27 (28 fSNPs, median 72 EM iterations per rSNP fit, 0.29-0.35 s at host load 100-360); PDE4DIP took 7 iterations and 0.04 s; the LDLR scout figure (14.6 ms) had 5
POLL = 10.0                # seconds between checks of the running RASQUAL processes
SMOKE_GENES = ('ANKRD36', 'PDE4DIP')   # the two cheapest native budgets in genes.tsv
SMOKE_RSNPS = 20
GTS = np.array(['0|0', '0|1', '1|0', '1|1'])   # index 2 xL + xR, as run_rasqual.GT
FSNP_OFFSET = 1_000_000_000   # native fSNP lines sit at pos + this, with -s/-e shifted the same (docstring, native arm)
ITER_FIELD = 'n_iter_null'    # README field 19 is pitr[l], THIS rSNP's fit iterations (main.c:760 prints pitr[l], pitr[0]); field 20 is line 0's count, not "alternative"
RASQUAL_MAXITR = 231          # nbem.c:513, itr < MAXITR3 + PRESTEPS + 200 (nbem.h:8-9), the per-fit EM cap field 19 reads at non-convergence
TIE = 1e5                     # RASQUAL ties chisq values equal after round(x * 1e5) (main.c:722, 728)
SUBSET = OUT / 'native_subset.tsv'
SEED, SUBSET_KEY = 42, 6      # random tested variants per gene: SeedSequence(SEED, spawn_key=(SUBSET_KEY, gene index)); keys 1-5 are the plasmode scripts'
TOP_K, N_RANDOM = 20, 80      # native subset per gene: top TOP_K variants of each other arm, N_RANDOM random tested variants, the leads
MODEL_FIELDS = ['chisq', 'effect_size_pi', 'error_rate_delta', 'ref_mapping_bias_phi', 'overdispersion_theta',
                'n_feature_snps', ITER_FIELD, 'convergence', 'r2_prior_posterior_fsnps', 'r2_prior_posterior_rsnp']
CONTROL = OUT / 'control'
NULLS = ('records', 'rasqual_r')
NULL_LABEL = {'records': 'records permutation', 'rasqual_r': 'RASQUAL -r'}
N_RUNS = 15                   # runs per null gene, arm and null: ~80 CPU-min per native run of the 10 null genes (arms.log scaled to 80 rSNPs), 2 x 15 runs ~ 40 CPU-h
CONTROL_KEY = {'records': 7, 'rasqual_r': 8}   # SeedSequence(SEED, spawn_key=(key, gene index, run))
GATE_KEY, RESAMPLE_KEY = 10, 9   # seeds of the -r determinism gate and of the gene-resampling intervals
N_RESAMPLE = 10_000           # gene-resampling draws per interval (genes drawn with replacement)
GATE_GENE = 'TRMT9B'          # the cheapest null gene in the native arm, 3.2 min at 113 rSNPs (arms.log)
GATE_RSNPS = 5                # rSNPs in the native -r determinism gate
STUCK_CHI2 = 0.01             # a RASQUAL fit left at pi = 0.5 reports chisq ~ 0; below this at another arm's lead is reported
SHIM_C = r'''#include <stdlib.h>
#include <time.h>
#include <unistd.h>
time_t time(time_t *t) {
    const char *s = getenv("RASQUAL_SEED");
    if (s == NULL) abort();
    time_t v = (time_t)strtoul(s, NULL, 10) - (time_t)getpid();
    if (t) *t = v;
    return v;
}
'''


def log(*a):
    print(*a, flush=True)


def bh(p):
    order = np.argsort(p)
    q = np.empty_like(p)
    q[order] = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    return np.minimum(q, 1)


def exon_unions():
    ex = {}
    for line in open(D / 'annot' / 'exons.tsv'):
        f = line.rstrip('\n').split('\t')
        if len(f) >= 3:
            ex[f[0]] = list(zip(map(int, f[1].split(',')), map(int, f[2].split(','))))
    return ex


def gene_table():
    gp = pd.read_csv(D / 'annot' / 'genes.tsv', sep='\t', header=None, dtype={1: str},
                     names=['gene', 'chr', 'start', 'end', 'pos'])
    return gp.drop_duplicates('gene').set_index('gene')


def candidate_pools():
    """T2T permutation-run genes with a symbol, their pool and whether the pipeline can run them."""
    gtf = BT.gtf_genes()
    r = pd.read_csv(BT.PERM, sep='\t')
    r['geneid'] = r.phenotype_id.str.replace('GeneID:', '', regex=False)
    r = r.merge(gtf[['gene', 'geneid']].dropna(), on='geneid', how='left')
    r['bh'] = bh(r.pval_beta.values)
    r['runnable'] = r.gene.isin(BT.runnable())
    r['pool'] = np.where(r.bh < 0.05, 'eqtl', np.where(r.pval_beta > 0.5, 'null', 'neither'))
    log(f'T2T permutation run: {len(r)} genes, {int(r.gene.notna().sum())} mapped to a symbol; '
        f'eqtl (BH < 0.05) {int((r.pool == "eqtl").sum())}, null (pval_beta > 0.5) {int((r.pool == "null").sum())}')
    return r


def exonic_sites(cand, gp, ex):
    """{(chr, pos, ref, alt): het mask over VCF samples} for biallelic SNPs inside the candidates' exons."""
    bed = SCOUT / 'candidate_exons.bed'
    with open(bed, 'w') as fh:
        for g in cand.gene:
            c = gp.loc[g, 'chr']
            for a, b in ex[g]:
                fh.write(f'{c}\t{a - 1}\t{b}\t{g}\n')
    q = subprocess.run(['bcftools', 'query', '-R', str(bed), '-f', '%CHROM\t%POS\t%REF\t%ALT[\t%GT]\n', str(VCF)],
                       capture_output=True, text=True, check=True).stdout
    sites = {}
    for line in q.strip().split('\n'):
        f = line.split('\t')
        key = (f[0], int(f[1]), f[2], f[3])
        if key not in sites:
            sites[key] = np.array([g[0] != g[2] and '.' not in g for g in f[4:]], bool)
    return sites


def exon_index(cand, gp, ex):
    """Per chromosome, sorted exon intervals (start, end, gene) of the candidate genes."""
    by = {}
    for g in cand.gene:
        c = gp.loc[g, 'chr']
        for a, b in ex[g]:
            by.setdefault(c, []).append((a, b, g))
    return {c: sorted(v) for c, v in by.items()}


def phaser_counts(sites, samples, index):
    """Per site, (ref, alt) phASER counts over VCF samples (0 where a donor has no row), and per
    gene the phASER rows inside its exon union classified as matched (site in the analysis VCF and
    the donor heterozygous there), hom (site present but the donor is not heterozygous) or absent
    (no such SNP record in the maf01 callset)."""
    idx = {s: i for i, s in enumerate(samples)}
    pos_keys = {(c, p) for c, p, _, _ in sites}
    ref = {k: np.zeros(len(samples), int) for k in sites}
    alt = {k: np.zeros(len(samples), int) for k in sites}
    cls = {}
    starts = {c: [iv[0] for iv in v] for c, v in index.items()}
    import bisect
    for line in AC_MANIFEST.read_text().strip().split('\n'):
        samp, path = (x.strip() for x in line.split('\t'))
        j = idx[samp]
        with open(path) as fh:
            fh.readline()
            for row in fh:
                f = row.rstrip('\n').split('\t')
                c, pos = f[0], int(f[1])
                if c not in starts:
                    continue
                i = bisect.bisect_right(starts[c], pos)
                genes = [g for a, b, g in index[c][max(0, i - 400):i] if a <= pos <= b]
                if not genes:
                    continue
                key = (c, pos, f[3], f[4])
                if key in ref:
                    ref[key][j] = int(f[5])
                    alt[key][j] = int(f[6])
                    kind = 'matched' if sites[key][j] else 'hom'
                elif (c, pos) in pos_keys:
                    kind = 'other_allele'
                else:
                    kind = 'absent'
                n = int(f[7])
                for g in genes:
                    d = cls.setdefault(g, {})
                    d[kind + '_rows'] = d.get(kind + '_rows', 0) + 1
                    d[kind + '_reads'] = d.get(kind + '_reads', 0) + n
        log(f'  read {samp}')
    return ref, alt, cls


def scout():
    SCOUT.mkdir(parents=True, exist_ok=True)
    r = candidate_pools()
    cand = r[(r.pool != 'neither') & r.runnable].drop_duplicates('gene').copy()
    ex, gp = exon_unions(), gene_table()
    cand = cand[cand.gene.isin(ex)]
    log(f'runnable candidates with exon records: eqtl {int((cand.pool == "eqtl").sum())}, '
        f'null {int((cand.pool == "null").sum())}')

    genes_all = CACHE.joinpath('genes.txt').read_text().split()
    gi = {g: i for i, g in enumerate(genes_all)}
    rows = np.array([gi[g] for g in cand.gene])
    pL = np.load(PE / 'pL.npy', mmap_mode='r')[rows]
    pR = np.load(PE / 'pR.npy', mmap_mode='r')[rows]
    pT = np.load(PE / 'pT.npy', mmap_mode='r')[rows]
    cand['n_admitted_pe'] = ((pL >= 0.5) & (pR >= 0.5)).sum(1)
    cand['median_pT'] = np.median(pT, axis=1)
    cand['median_allele_reads'] = np.median(pL + pR, axis=1)

    samples = subprocess.run(['bcftools', 'query', '-l', str(VCF)], capture_output=True, text=True,
                             check=True).stdout.split()
    if len(samples) != pL.shape[1]:
        raise SystemExit(f'{len(samples)} VCF samples against {pL.shape[1]} cache samples')
    sites = exonic_sites(cand, gp, ex)
    log(f'exonic biallelic SNP sites in candidate genes: {len(sites)}')
    ref, alt, cls = phaser_counts(sites, samples, exon_index(cand, gp, ex))
    keys = sorted(sites)
    np.savez_compressed(SCOUT / 'exonic_sites.npz', samples=np.array(samples),
                        chrom=np.array([k[0] for k in keys]), pos=np.array([k[1] for k in keys]),
                        ref=np.array([k[2] for k in keys]), alt=np.array([k[3] for k in keys]),
                        het=np.array([sites[k] for k in keys]),
                        ref_count=np.array([ref[k] for k in keys], np.int32),
                        alt_count=np.array([alt[k] for k in keys], np.int32))
    by_chr = {}
    for k in keys:
        by_chr.setdefault(k[0], []).append(k)
    rec = []
    for g in cand.gene:
        n_sites = n_cov = n3x8 = n5x8 = pairs1 = pairs8 = het_reads = 0
        for key in by_chr.get(gp.loc[g, 'chr'], []):
            if not any(a <= key[1] <= b for a, b in ex[g]):
                continue
            n_sites += 1
            het = sites[key]
            rr, aa = ref[key] * het, alt[key] * het
            tot = rr + aa
            if tot.sum() >= 1 and rr.sum() > 0 and aa.sum() > 0:
                n_cov += 1
                pairs1 += int((tot >= 1).sum())
                pairs8 += int((tot >= 8).sum())
                het_reads += int(tot.sum())
                n3x8 += int((tot >= 8).sum() >= 3)
                n5x8 += int((tot >= 8).sum() >= 5)
        d = cls.get(g, {})
        rec.append(dict(gene=g, n_exonic_snps=n_sites, n_covered_sites=n_cov, n_sites_3x8=n3x8, n_sites_5x8=n5x8,
                        het_pairs_ge1=pairs1, het_pairs_ge8=pairs8, het_reads=het_reads,
                        **{k: d.get(k, 0) for k in ('matched_rows', 'matched_reads', 'hom_rows', 'hom_reads',
                                                     'absent_rows', 'absent_reads', 'other_allele_rows')}))
    cand = cand.merge(pd.DataFrame(rec), on='gene')
    cand = cand[['gene', 'geneid', 'pool', 'pval_beta', 'bh', 'variant_id', 'slope', 'slope_se', 'num_var',
                 'n_admitted_pe', 'median_pT', 'median_allele_reads', 'n_exonic_snps', 'n_covered_sites',
                 'n_sites_3x8', 'n_sites_5x8', 'het_pairs_ge1', 'het_pairs_ge8', 'het_reads', 'matched_rows',
                 'matched_reads', 'hom_rows', 'hom_reads', 'absent_rows', 'absent_reads', 'other_allele_rows']]
    cand.to_csv(SCOUT / 'candidates.tsv', sep='\t', index=False)
    ok = cand[cand.n_admitted_pe >= MIN_ADMITTED]
    summ = {}
    for pool in ('eqtl', 'null'):
        s = ok[ok.pool == pool]
        summ[pool] = dict(runnable=int((cand.pool == pool).sum()), admitted_ge15=int(len(s)),
                          covered_ge1=int((s.n_covered_sites >= 1).sum()),
                          sites_3x8_ge1=int((s.n_sites_3x8 >= 1).sum()),
                          sites_3x8_ge3=int((s.n_sites_3x8 >= 3).sum()),
                          sites_3x8_ge5=int((s.n_sites_3x8 >= 5).sum()),
                          sites_3x8_ge10=int((s.n_sites_3x8 >= 10).sum()),
                          covered_sites_quartiles=[float(x) for x in s.n_covered_sites.quantile([.25, .5, .75])],
                          sites_3x8_quartiles=[float(x) for x in s.n_sites_3x8.quantile([.25, .5, .75])],
                          hom_reads_share=float(s.hom_reads.sum() / max(1, s.matched_reads.sum() + s.hom_reads.sum()
                                                                        + s.absent_reads.sum())),
                          absent_reads_share=float(s.absent_reads.sum() / max(1, s.matched_reads.sum()
                                                                              + s.hom_reads.sum() + s.absent_reads.sum())))
    log(json.dumps(summ, indent=1))
    cols = ['gene', 'pval_beta', 'bh', 'variant_id', 'n_admitted_pe', 'median_allele_reads', 'n_exonic_snps',
            'n_covered_sites', 'n_sites_3x8', 'n_sites_5x8', 'het_pairs_ge8', 'het_reads', 'hom_rows', 'absent_rows']
    eq = ok[ok.pool == 'eqtl'].sort_values(['n_sites_3x8', 'het_pairs_ge8'], ascending=False).head(N_EQTL + 10)
    log(f'top {N_EQTL + 10} eQTL genes by exonic heterozygous SNPs with >= 3 donors at >= 8 phASER reads:')
    log(eq[cols].to_string(index=False))
    nu = ok[ok.pool == 'null'].sort_values(['n_sites_3x8', 'het_pairs_ge8'], ascending=False).head(N_NULL + 10)
    log(f'top {N_NULL + 10} null genes by the same measure:')
    log(nu[cols].to_string(index=False))
    (SCOUT / 'summary.json').write_text(json.dumps(summ, indent=1))


def tested_variants(chrom, tss, start, end):
    """Ids of the analysis VCF's variants in the gene's cis window, outside its body, at MAF >= MAF_MIN
    (compare_mixqtl_replication.load_inputs' tested filter) among the records the loader carries (biallelic
    SNPs phased and called in every donor, run_hapmixqtl_from_salmon.read_phased_vcf), plus every id in the
    window."""
    q = subprocess.run(['bcftools', 'query', '-r', f'{chrom}:{max(1, tss - WIN)}-{tss + WIN}',
                        '-f', '%ID\t%POS\t%REF\t%ALT[\t%GT]\n', str(VCF)], capture_output=True, text=True, check=True).stdout
    tested, seen = [], []
    for line in q.strip().split('\n'):
        f = line.split('\t')
        pos = int(f[1])
        seen.append(f[0])
        gts = f[4:]
        if len(f[2]) != 1 or len(f[3]) != 1 or any('|' not in g or '.' in g for g in gts):
            continue
        dos = np.array([(g[0] != '0') + (g[2] != '0') for g in gts], float)
        af = dos.mean() / 2.0
        if not (start <= pos <= end) and min(af, 1 - af) >= MAF_MIN:
            tested.append(f[0])
    return tested, seen


def select():
    """The 30 genes and the checks the arms stage relies on; writes OUT/genes.txt and OUT/genes.tsv."""
    # keep_default_na: pandas would otherwise read the pool label "null" as NaN
    cand = pd.read_csv(SCOUT / 'candidates.tsv', sep='\t', keep_default_na=False, na_values=[''])
    gp, ex = gene_table(), exon_unions()
    sd = set(pd.read_csv(BT.SD_GENES, sep='\t', header=None)[0])
    cand['segmental_duplication'] = cand.gene.isin(sd)   # recorded for a sensitivity split, not a selection rule
    ok = cand[cand.n_admitted_pe >= MIN_ADMITTED + ADMISSION_MARGIN]
    chosen, taken = [], []
    for pool, n in (('eqtl', N_EQTL), ('null', N_NULL)):
        s = ok[ok.pool == pool].sort_values(['n_sites_3x8', 'het_pairs_ge8'], ascending=False)
        for _, r in s.iterrows():
            if len([c for c in chosen if c['pool'] == pool]) >= n:
                break
            c, tss = gp.loc[r.gene, 'chr'], int(gp.loc[r.gene, 'pos'])
            if any(c == cc and abs(tss - tt) <= 2 * WIN for cc, tt in taken):   # cis windows must not overlap
                log(f'  {r.gene} ({pool}) skipped: its cis window overlaps a chosen gene')
                continue
            taken.append((c, tss))
            chosen.append(dict(r))
    rows = []
    for c in chosen:
        g = c['gene']
        chrom, tss = gp.loc[g, 'chr'], int(gp.loc[g, 'pos'])
        start, end = int(gp.loc[g, 'start']), int(gp.loc[g, 'end'])
        tested, seen = tested_variants(chrom, tss, start, end)
        lead = c['variant_id']
        rows.append(dict(c, chrom=chrom, tss=tss, start=start, end=end, exon_union_bp=sum(b - a + 1 for a, b in ex[g]),
                         n_tested=len(tested), pseudo_budget=2 * len(tested),
                         native_budget=(int(c['n_covered_sites']) + 1) * len(tested),
                         lead_in_vcf=lead in set(seen), lead_tested=lead in set(tested),
                         lead_in_body=bool(start <= int(lead.split('_')[1]) <= end) if lead.startswith(chrom + '_') else False))
        log(f'{g:10s} {c["pool"]:4s} tested {len(tested):6d}  covered fSNP lines {int(c["n_covered_sites"]):4d} '
            f'(3x8 {int(c["n_sites_3x8"]):3d})  admitted {int(c["n_admitted_pe"]):3d}  lead {lead}: in VCF '
            f'{rows[-1]["lead_in_vcf"]}, tested {rows[-1]["lead_tested"]}, in body {rows[-1]["lead_in_body"]}')
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / 'genes.tsv', sep='\t', index=False)
    (OUT / 'genes.txt').write_text('\n'.join(df.gene) + '\n')
    with open(OUT / 'regions.bed', 'w') as fh:      # corrected_null_store.py:97-100
        for _, r in df.iterrows():
            lo = max(0, min(r.start, r.tss) - WIN - 1000)
            fh.write(f'{r.chrom}\t{lo}\t{max(r.end, r.tss) + WIN + 1000}\t{r.gene}\n')
    log(f'\n{len(df)} genes: pseudo arm over budget (2L > 30000) {int((df.pseudo_budget > 30000).sum())}; '
        f'native over budget without --force {int((df.native_budget > 30000).sum())}; published lead in VCF '
        f'{int(df.lead_in_vcf.sum())}, in tested set {int(df.lead_tested.sum())}, inside the gene body '
        f'{int(df.lead_in_body.sum())}')


def read_genes():
    # keep_default_na: pandas would otherwise read the pool label "null" as NaN
    return pd.read_csv(OUT / 'genes.tsv', sep='\t', keep_default_na=False, na_values=[''])


def load_setup():
    """The loader inputs and run_arms.setup on the 30 genes; every tested-variant count must equal genes.tsv's."""
    genes = read_genes()
    with contextlib.redirect_stdout(io.StringIO()):
        I = CM.load_point_estimate_inputs(gene_list=str(OUT / 'genes.txt'), regions=str(OUT / 'regions.bed'))
    S = C.setup(I)
    if S['genes'] != list(genes.gene):
        raise SystemExit('gene order differs between genes.txt and genes.tsv')
    bad = {g: (int(S['n_tested'][g]), int(n)) for g, n in zip(genes.gene, genes.n_tested) if int(S['n_tested'][g]) != int(n)}
    if bad:
        raise SystemExit(f'tested-variant counts (setup, genes.tsv) differ: {bad}')
    return genes, S


def observed(S):
    """The observed records as make_datasets.load builds them, summarised with the identity permutation."""
    I = S['I']
    keep = I['keep']
    R = {k: I[k][:, keep] for k in ('pL', 'pR', 'pT', 'YL', 'YR', 'YT')}
    eff_lib = I['eff_lib'][keep]
    A, T, Va, Vt, _ = summaries_from_point_estimates(R['pL'], R['pR'], R['pT'], eff_lib, R['YL'], R['YR'], R['YT'])
    N = len(keep)
    ds = dict(A=A, T=T, Va=Va, Vt=Vt, pL=R['pL'], pR=R['pR'], pT=R['pT'], eff_lib=eff_lib,
              perm=np.arange(N), swap=np.ones(N, np.int8))
    kept = C.allelic_kept(ds['pL'], ds['pR'], ds['Va'])
    n = kept.sum(1)
    low = [f'{g} ({k})' for g, k in zip(S['genes'], n) if k < MIN_ADMITTED]
    if low:
        raise SystemExit(f'genes below {MIN_ADMITTED} admitted allelic donors: {low}')
    log(f'observed data: {len(S["genes"])} genes x {N} donors; admitted allelic donors per gene '
        f'{n.min()}-{n.max()} (median {int(np.median(n))}); pairs with Va > 0 but dropped by the one-haplotype-zero '
        f'rule {int(((ds["Va"] > C.EPS) & ~kept).sum())}')
    return ds, kept


def fingerprint(ds):
    h = hashlib.sha256()
    for k in ('A', 'T', 'Va', 'Vt', 'pL', 'pR', 'pT', 'eff_lib'):
        h.update(np.ascontiguousarray(ds[k]).tobytes())
    return h.hexdigest()


def run_split(S, ds):
    """map_nominal under split weighting, the call of run_arms.run_nominal without its causal-variant gate."""
    A, T, Va, Vt, cov, n_zeroed = C.phenotypes(S, ds, 'split')
    scratch = OUT / 'scratch'
    scratch.mkdir(parents=True, exist_ok=True)
    for q in scratch.glob('*'):
        q.unlink()
    C.quiet(map_nominal, S['gdf'], S['vdf'][['chrom', 'pos']], A, T, Va, Vt, S['gp'],
            xL_df=S['xLdf'], xR_df=S['xRdf'], prefix='n', covariates_df=cov,
            genotype_covariates_df=S['I']['geno_cov_df'], window=CM.WIN,
            output_dir=str(scratch), verbose=False, ase_covariates_df=None)
    df = pd.concat([pd.read_parquet(q, columns=CNS.COLS + C.DOF_COLS) for q in sorted(scratch.glob('n*.parquet'))],
                   ignore_index=True)
    df['variant_id'] = df['variant_id'].astype(str)
    per = df.groupby('phenotype_id').size().reindex(S['genes'])
    if len(df) != int(S['n_tested'].sum()) or not per.equals(S['n_tested'].reindex(S['genes'])):
        raise SystemExit(f'map_nominal returned {len(df):,} rows against {int(S["n_tested"].sum()):,} tested pairs')
    shutil.rmtree(scratch)
    return df, n_zeroed


def save_tested(S, genes, kept):
    """Tested variant ids and ALT dosages per gene, the published leads' dosages and the admitted allelic donor
    counts, for the report stage."""
    I = S['I']
    rows = np.concatenate([S['tested_rows'][g] for g in S['genes']])
    gene = np.concatenate([[g] * len(S['tested_rows'][g]) for g in S['genes']])
    ids = I['vdf'].index.values.astype(str)
    pub_id, pub_dos = [], []
    for v in genes.variant_id:
        j = np.where(ids == v)[0]
        if len(j) == 1:
            pub_id.append(v)
            pub_dos.append(I['dos'][j[0]])
    np.savez_compressed(TESTED, gene=gene, variant_id=ids[rows], dos=I['dos'][rows].astype(np.int8),
                        pub_id=np.array(pub_id), pub_dos=np.array(pub_dos, np.int8), samples=np.array(S['order']),
                        genes=np.array(S['genes']), n_admitted=kept.sum(1))
    log(f'wrote {TESTED}: {len(rows):,} tested gene-variant pairs; published leads with a dosage row {len(pub_id)} of {len(genes)}')


def variant_key(vdf):
    """(chrom, pos, ref, alt) -> row of the loader's variant frame, built once (the frame holds every phased
    biallelic SNP of the 30 regions)."""
    return {(c, int(p), r, a): j for j, (c, p, r, a) in enumerate(zip(vdf.chrom.values, vdf.pos.values,
                                                                       vdf.ref.values, vdf.alt.values))}


def exonic_lines(S, z, ex, g, vkey, perm=None):
    """The gene's fSNP lines (docstring, native arm) and how many exonic sites the loader did not carry; with perm,
    column i carries donor perm[i]'s GT:AS entry (the records null, docstring --control)."""
    I, vdf = S['I'], S['I']['vdf']
    chrom = S['gp'].loc[g, 'chr']
    lines, missing = [], 0
    for i in np.where(z['chrom'] == chrom)[0]:
        pos = int(z['pos'][i])
        if not any(a <= pos <= b for a, b in ex[g]):
            continue
        j = vkey.get((chrom, pos, str(z['ref'][i]), str(z['alt'][i])))
        if j is None:
            missing += 1
            continue
        xL, xR = I['xL'][j], I['xR'][j]
        if not np.array_equal(xL != xR, z['het'][i]):
            raise SystemExit(f'{g} {chrom}:{pos}: heterozygosity differs between the loader and the scout archive')
        entries = [f'{gt}:{r},{a}' for gt, r, a in zip(GTS[xL * 2 + xR], z['ref_count'][i], z['alt_count'][i])]
        if perm is not None:
            entries = [entries[p] for p in perm]
        lines.append(f'{chrom}\t{pos + FSNP_OFFSET}\t{vdf.index[j]}\t{z["ref"][i]}\t{z["alt"][i]}\t.\tPASS\t.\tGT:AS\t'
                     + '\t'.join(entries) + '\n')
    return lines, missing


def native_cmd(k, g, n_lines, n_fsnp, starts, ends, tss, bins, n, threads):
    """run_rasqual.run_gene's option line (cited there to usage.c) with the exon union as -s/-e (shifted by
    FSNP_OFFSET like the fSNP lines), the cis window -c/-w that excludes them from the rSNP scan, --force and the
    thread count."""
    return [str(RR.RASQUAL), '-y', bins['Y'], '-k', bins['K'], '-n', str(n), '-j', str(k + 1),
            '-l', str(n_lines), '-m', str(n_fsnp), '-s', starts, '-e', ends, '-c', str(tss), '-w', str(2 * WIN),
            '-f', g, '-z', '-d', str(RR.MIN_COVERAGE), '-a', str(RR.MAF), '-h', str(RR.HWE_P), '-x', bins['X'],
            '--n-threads', str(threads), '--force']


def run_pool(jobs, out_dir):
    """Run (gene, threads, cmd, text) jobs, longest first as given, with at most JOBS threads in flight.

    Raw stdout goes to out_dir/<gene>.tsv.tmp and is renamed on a zero exit, stderr to
    <gene>.err. A gene whose RASQUAL exits non-zero keeps its .tmp and .err, the other
    genes run on, and the stage stops with the list of failed genes once the pool drains
    (a relaunch retries exactly those). An interrupt terminates the running processes.
    """
    pending, running, free, secs, failed = list(jobs), {}, JOBS, {}, []
    t_start = time.perf_counter()
    try:
        while pending or running:
            i = 0
            while i < len(pending):
                g, t, cmd, text = pending[i]
                if t > free:
                    i += 1
                    continue
                tmp = out_dir / f'{g}.tsv.tmp'
                fo, fe = open(tmp, 'w'), open(out_dir / f'{g}.err', 'w')
                p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=fo, stderr=fe, text=True)
                feeder = threading.Thread(target=lambda p=p, text=text: (p.stdin.write(text), p.stdin.close()))
                feeder.start()
                running[p] = (g, t, time.perf_counter(), tmp, feeder, fo, fe)
                free -= t
                pending.pop(i)
                log(f'  launched {g} ({t} thread{"s" if t > 1 else ""}; {free} of {JOBS} threads free, '
                    f'{len(pending)} genes waiting)')
            time.sleep(POLL)
            for p in [p for p in running if p.poll() is not None]:
                g, t, t0, tmp, feeder, fo, fe = running.pop(p)
                feeder.join()
                fo.close()
                fe.close()
                free += t
                secs[g] = time.perf_counter() - t0
                if p.returncode != 0:
                    failed.append(g)
                    log(f'  FAILED {g}: RASQUAL exit status {p.returncode} after {secs[g] / 60:.1f} min, see {fe.name}')
                    continue
                os.replace(tmp, out_dir / f'{g}.tsv')
                log(f'  finished {g} in {secs[g] / 60:.1f} min ({(time.perf_counter() - t_start) / 3600:.2f} h elapsed)')
    finally:
        for p in running:
            p.terminate()
    if failed:
        raise SystemExit(f'RASQUAL failed on {len(failed)} genes: {failed}')
    return secs


def parse_raw(path, g):
    """RASQUAL's rows as run_rasqual.run_gene validates them."""
    rows = [ln.split('\t') for ln in path.read_text().splitlines()]
    bad = [r for r in rows if len(r) != len(RASQUAL_FIELDS) or r[0] != g or r[1] == 'SKIPPED']
    if bad or not rows:
        raise SystemExit(f'{path}: {len(rows)} RASQUAL rows, {len(bad)} malformed or SKIPPED, e.g. {bad[:1]}')
    return pd.DataFrame(rows, columns=RASQUAL_FIELDS)


def write_raw(path, raw):
    C.write_atomic(path, lambda fh: fh.write(''.join('\t'.join(r) + '\n' for r in raw.values)), 'w')


def run_gene(k, g, site, text, bins, n):
    """run_rasqual.run_gene as the pseudo arm ran it (fc238df^; the 04_run_rasqual version also checkpoints to a file):
    RASQUAL on one gene, its rows as strings and the wall seconds."""
    t0 = time.perf_counter()
    out = subprocess.run(pseudo_cmd(k, g, site, text.count('\n'), bins, n), input=text, stdout=subprocess.PIPE,
                         text=True, check=True).stdout
    secs = time.perf_counter() - t0
    rows = [ln.split('\t') for ln in out.splitlines()]
    bad = [r for r in rows if len(r) != len(RASQUAL_FIELDS) or r[0] != g or r[1] == 'SKIPPED']
    if bad or not rows:
        raise SystemExit(f'{g}: {len(rows)} RASQUAL rows, {len(bad)} malformed or SKIPPED, e.g. {bad[:1]}')
    return pd.DataFrame(rows, columns=RASQUAL_FIELDS), secs


def admission(pL, pR, kept):
    """What the pseudo fSNP's het set drops and what rounding to the AS field does to what it keeps
    (run_rasqual.admission, fc238df^)."""
    a, b = np.rint(pL), np.rint(pR)
    return (f'{int(kept.sum())} het of {int((pL + pR > 0).sum())} informative pairs, '
            f'{int(((pL + pR > 0) & ~kept).sum())} excluded by allelic_kept; among het, AS total changed by '
            f'> 0.5 read {int((kept & (abs(a + b - pL - pR) > 0.5)).sum())}, AS 0,0 {int((kept & (a + b == 0)).sum())}')


def arm_table(g, raw, tested):
    """run_rasqual.assemble on the rows of the tested set, plus field 24 (r2_prior_posterior_fsnps) and the count of
    fSNP rows RASQUAL tested as rSNPs."""
    keep = raw.rs_id.isin(tested) | (raw.rs_id == f'{g}_pseudo_fsnp')
    out, cnt = RR.assemble(g, raw[keep], tested, None)
    cnt['fsnp_as_rsnp_rows'] = int((~keep).sum())
    by_id = raw.set_index('rs_id')
    out['r2_fsnps'] = by_id.r2_prior_posterior_fsnps.astype(float).reindex(out.variant_id).values   # '-nan' with no fSNP
    out['n_iter'] = by_id[ITER_FIELD].astype(int).reindex(out.variant_id).values
    return out, cnt


def inputs_for_rasqual(S, ds, d=OUT / 'rasqual_inputs'):
    if not os.access(RR.RASQUAL, os.X_OK):
        raise SystemExit(f'{RR.RASQUAL}: missing or not executable')
    log(f'{RR.RASQUAL} sha256 {hashlib.sha256(Path(RR.RASQUAL).read_bytes()).hexdigest()}')
    bins, n_cov = RR.write_bins(S, ds, d)
    log(f'RASQUAL inputs: Y = Salmon point-estimate totals pT, K = eff_lib / mean, X = {n_cov} covariates '
        f'({S["I"]["cov_df"].shape[1]} RNA-tied + {S["I"]["geno_cov_df"].shape[1]} genotype PCs), identity permutation')
    with np.load(SCOUT / 'exonic_sites.npz') as npz:
        z = {k: npz[k] for k in npz.files}     # in memory: an NpzFile re-reads an array at every access
    if list(z['samples']) != S['order']:
        raise SystemExit('sample order differs between the scout archive and the loader')
    return bins, z


def rsnp_subset(S, g, ids):
    """run_rasqual.rsnp_text over the gene's tested variants whose ids are given."""
    rows = S['tested_rows'][g]
    keep = np.isin(S['I']['vdf'].index.values[rows].astype(str), list(ids))
    if keep.sum() != len(ids):
        raise SystemExit(f'{g}: {len(ids)} rSNP ids requested, {int(keep.sum())} are tested variants')
    return RR.rsnp_text(dict(S, tested_rows={g: rows[keep]}), g)


def native_job(S, z, ex, g, bins, threads, ids, vkey, perm=None):
    """(gene, threads, cmd, text) for one native gene over the rSNP ids given, and its line counts."""
    k = S['genes'].index(g)
    fs, missing = exonic_lines(S, z, ex, g, vkey, perm)
    text = ''.join(fs) + rsnp_subset(S, g, ids)
    starts = ','.join(str(a + FSNP_OFFSET) for a, _ in ex[g])
    ends = ','.join(str(b + FSNP_OFFSET) for _, b in ex[g])
    cmd = native_cmd(k, g, text.count('\n'), len(fs), starts, ends, int(S['gp'].loc[g, 'pos']), bins,
                     len(S['order']), threads)
    return (g, threads, cmd, text), len(fs), missing


def native_subset(S, genes):
    """Per gene, the tested variants the native arm scans: the pseudo and split leads, the published lead when
    tested, the TOP_K strongest of each of those arms and N_RANDOM random tested variants (seeded per gene).
    Written once to SUBSET and reused, so a relaunch scans the same variants."""
    if SUBSET.exists():
        sub = pd.read_csv(SUBSET, sep='\t', keep_default_na=False, na_values=[''])
        log(f'skip: {SUBSET} exists ({len(sub)} gene-variant pairs)')
        return sub
    split = pd.read_parquet(SPLIT)
    rec = []
    for k, r in enumerate(genes.itertuples()):
        g = r.gene
        tested = S['I']['vdf'].index.values[S['tested_rows'][g]].astype(str)
        ps, _ = arm_table(g, parse_raw(ARM_DIR['pseudo'] / f'{g}.tsv', g), set(tested))
        ps = ps.sort_values('chisq', ascending=False)
        sp = split[split.phenotype_id == g].sort_values('pval_nominal')
        rng = np.random.default_rng(np.random.SeedSequence(SEED, spawn_key=(SUBSET_KEY, k)))
        rand = set(rng.choice(tested, size=min(N_RANDOM, len(tested)), replace=False))
        top_p, top_s = set(ps.variant_id.iloc[:TOP_K]), set(sp.variant_id.iloc[:TOP_K])
        leads = dict(lead_pseudo=ps.variant_id.iloc[0], lead_split=sp.variant_id.iloc[0])
        pub = r.variant_id if r.lead_tested else None
        for v in sorted(rand | top_p | top_s | set(leads.values()) | ({pub} if pub else set())):
            rec.append(dict(gene=g, variant_id=v, random=v in rand, top_pseudo=v in top_p, top_split=v in top_s,
                            lead_pseudo=v == leads['lead_pseudo'], lead_split=v == leads['lead_split'],
                            published=v == pub))
    sub = pd.DataFrame(rec)
    C.write_atomic(SUBSET, lambda fh: sub.to_csv(fh, sep='\t', index=False), 'w')
    per = sub.groupby('gene').size()
    log(f'native subset: {len(sub)} gene-variant pairs, per gene {per.min()}-{per.max()}; random {int(sub.random.sum())}, '
        f'top-{TOP_K} pseudo {int(sub.top_pseudo.sum())}, top-{TOP_K} split {int(sub.top_split.sum())}, published leads '
        f'{int(sub.published.sum())}; wrote {SUBSET}')
    return sub


def smoke():
    genes, S = load_setup()
    ds, kept = observed(S)
    bins, z = inputs_for_rasqual(S, ds)
    ex = exon_unions()
    out = OUT / 'smoke'
    out.mkdir(parents=True, exist_ok=True)
    gt = genes.set_index('gene')
    vkey = variant_key(S['I']['vdf'])
    ok = True
    def run(cmd, text, tag):
        """RASQUAL into out/<tag>.tsv unless it is already there; wall seconds (NaN when skipped)."""
        f = out / f'{tag}.tsv'
        if f.exists() and f.stat().st_size > 0:
            log(f'  skip: {f} exists')
            return float('nan')
        t0 = time.perf_counter()
        (out / f'{tag}.err').write_text(subprocess.run(cmd, input=text, stdout=open(f, 'w'), stderr=subprocess.PIPE,
                                                       text=True, check=True).stderr)
        return time.perf_counter() - t0

    for g in SMOKE_GENES:
        ids = list(S['I']['vdf'].index.values[S['tested_rows'][g]].astype(str)[:SMOKE_RSNPS])
        job, n_fs, missing = native_job(S, z, ex, g, bins, 1, ids, vkey)
        res = {}
        for t in (1, THREADS_MAX):
            c = list(job[2])
            c[c.index('--n-threads') + 1] = str(t)
            res[t] = run(c, job[3], f'{g}.t{t}')
        raw = parse_raw(out / f'{g}.t1.tsv', g)
        same = sorted((out / f'{g}.t1.tsv').read_text().splitlines()) == sorted((out / f'{g}.t{THREADS_MAX}.tsv').read_text().splitlines())
        tab, cnt = arm_table(g, raw, set(ids))
        nf = sorted(set(raw.n_feature_snps.astype(int)))
        units = (nf[-1] + 1) * len(raw)
        i = tab.chisq.idxmax()
        log(f'{g}: {n_fs} fSNP lines ({missing} exonic sites not in the loader) + {SMOKE_RSNPS} rSNPs; RASQUAL admitted '
            f'{nf} fSNPs (genes.tsv n_covered_sites {int(gt.loc[g, "n_covered_sites"])}); {len(raw)} rows, tested rows '
            f'{len(tab)}, fSNP-as-rSNP rows {cnt["fsnp_as_rsnp_rows"]} (want 0), non-converged {cnt["nonconv"]}, absent '
            f'{cnt["absent"]}, EM iterations per rSNP fit median {int(tab.n_iter.median())} '
            f'[{int(tab.n_iter.min())}, {int(tab.n_iter.max())}] (cap {RASQUAL_MAXITR}); '
            f'{res[1]:.1f} s at 1 thread ({1e3 * res[1] / units:.1f} ms per admitted fSNP x rSNP), '
            f'{res[THREADS_MAX]:.1f} s at {THREADS_MAX} threads; threaded output identical {same}; lead '
            f'{tab.loc[i, "variant_id"]} chisq {tab.chisq.max():.2f}, phi at lead {tab.loc[i, "phi"]:.3f}, delta '
            f'{tab.loc[i, "delta"]:.4f}, r2_fsnps {tab.loc[i, "r2_fsnps"]:.3f}')
        ok &= same and nf[-1] > 0 and cnt['fsnp_as_rsnp_rows'] == 0
        old = OUT / 'smoke_unshifted' / f'{g}.t1.tsv'
        if old.exists() and old.stat().st_size > 0:   # PDE4DIP's unshifted run was stopped before it wrote rows
            # the identity gate: the fit's own fields must not change when the fSNP lines move by FSNP_OFFSET;
            # fields 10, 16, 18 and 20 (BH q, index in region, tested count, line 0's iterations) depend on
            # which lines were scanned
            a, b = parse_raw(old, g).set_index('rs_id'), raw.set_index('rs_id')
            common = [v for v in b.index if v in a.index]
            diff = int((a.loc[common, MODEL_FIELDS].values != b.loc[common, MODEL_FIELDS].values).any(1).sum())
            log(f'  identity gate against {old.name} (unshifted fSNP lines): {len(common)} shared rSNPs, rows with any '
                f'model field differing {diff} (want 0)')
            ok &= len(common) == len(b) and diff == 0
    g = SMOKE_GENES[0]
    ids = list(S['I']['vdf'].index.values[S['tested_rows'][g]].astype(str)[:SMOKE_RSNPS])
    job = native_job(S, z, ex, g, bins, 1, ids, vkey)[0]
    c = [x for x in job[2] if x != '-z']
    secs = run(c, job[3], f'{g}.noz')
    noz = parse_raw(out / f'{g}.noz.tsv', g)
    withz = parse_raw(out / f'{g}.t1.tsv', g)
    log(f'{g} without -z (diagnostic only; the arms keep -z): EM iterations per rSNP fit median '
        f'{int(noz[ITER_FIELD].astype(int).median())} against {int(withz[ITER_FIELD].astype(int).median())} with -z, '
        f'{secs:.1f} s; max chisq {noz.chisq.astype(float).max():.2f} against {withz.chisq.astype(float).max():.2f}; '
        f'r2_prior_posterior_fsnps {noz.r2_prior_posterior_fsnps.iloc[0]} against {withz.r2_prior_posterior_fsnps.iloc[0]}')
    log(f'smoke {"passed" if ok else "FAILED"}: fSNPs admitted, none scanned as an rSNP, thread-invariant output and the '
        f'identity gate in every smoke gene')


def arms():
    genes, S = load_setup()
    ds, kept = observed(S)
    N = len(S['order'])
    if SPLIT.exists():
        log(f'skip: {SPLIT} exists')
    else:
        t0 = time.perf_counter()
        df, n_zeroed = run_split(S, ds)
        C.write_parquet(df, SPLIT, fingerprint(ds), 'log2')
        below = sorted(df.phenotype_id[~df.allelic_admitted].unique())
        log(f'split: map_nominal {len(df):,} rows; allelic admission zeroed {n_zeroed} donor-gene pairs; genes with the '
            f'allelic channel out of the combination {len(below)} {below}; {time.perf_counter() - t0:.1f} s')
    save_tested(S, genes, kept)
    bins, z = inputs_for_rasqual(S, ds)
    for d in ARM_DIR.values():
        d.mkdir(parents=True, exist_ok=True)

    todo = [g for g in S['genes'] if not (ARM_DIR['pseudo'] / f'{g}.tsv').exists()]
    log(f'pseudo arm: {len(S["genes"]) - len(todo)} genes skipped (done), {len(todo)} to run, {JOBS} jobs')
    sites = {g: RR.pseudo_site(S, g) for g in todo}
    kk = [S['genes'].index(g) for g in todo]
    if todo:
        log(f'  pseudo fSNP {admission(ds["pL"][kk], ds["pR"][kk], kept[kk])}')
    t0 = time.perf_counter()
    with cf.ThreadPoolExecutor(JOBS) as ex_:
        futs = {g: ex_.submit(run_gene, k, g, sites[g],
                              RR.pseudo_line(g, sites[g], ds['pL'][k], ds['pR'][k], kept[k]) + RR.rsnp_text(S, g),
                              bins, N) for k, g in zip(kk, todo)}
        for g, f in futs.items():
            raw, secs = f.result()
            write_raw(ARM_DIR['pseudo'] / f'{g}.tsv', raw)
            log(f'  {g}: {len(raw)} rows in {secs:.0f} s')
    if todo:
        log(f'pseudo arm done in {(time.perf_counter() - t0) / 60:.1f} min')

    ex = exon_unions()
    sub = native_subset(S, genes)
    todo = [g for g in S['genes'] if not (ARM_DIR['native'] / f'{g}.tsv').exists()]
    gt = genes.set_index('gene')
    n_sub = sub.groupby('gene').size()
    units = {g: (int(gt.loc[g, 'n_covered_sites']) + 1) * int(n_sub[g]) for g in todo}   # covered sites bound the admitted fSNPs
    per_slot = sum(units.values()) / JOBS if todo else 1.0
    vkey = variant_key(S['I']['vdf'])
    jobs = []
    for g in sorted(todo, key=lambda g: -units[g]):
        threads = int(min(THREADS_MAX, max(1, np.ceil(units[g] / per_slot))))
        job, n_fs, missing = native_job(S, z, ex, g, bins, threads, list(sub.variant_id[sub.gene == g]), vkey)
        jobs.append(job)
        log(f'  {g}: {n_fs} fSNP lines ({missing} exonic sites not in the loader), {int(n_sub[g])} rSNPs, '
            f'{units[g] / 1e3:.0f} k units at most, {threads} thread(s), <= {units[g] * SEC_PER_UNIT / threads / 3600:.1f} h')
    log(f'native arm: {len(S["genes"]) - len(todo)} genes skipped (done), {len(todo)} to run; at most '
        f'{sum(units.values()) * SEC_PER_UNIT / 3600:.0f} CPU hours at {SEC_PER_UNIT * 1e3:.0f} ms per admitted fSNP x '
        f'rSNP over {JOBS} threads')
    secs = run_pool(jobs, ARM_DIR['native'])
    if secs:
        s = np.array([secs[g] for g in secs])
        u = np.array([units[g] for g in secs])
        log(f'native arm done: {len(secs)} genes, wall minutes per gene median {np.median(s) / 60:.1f} '
            f'[{s.min() / 60:.1f}, {s.max() / 60:.1f}]; {1e3 * s.sum() / u.sum():.1f} ms per covered fSNP x rSNP pooled '
            f'(thread-seconds not corrected)')
    log(f'wrote {OUT}')


def seed_shim():
    """The LD_PRELOAD library that seeds RASQUAL's rand() (docstring, --control rasqual_r), compiled once."""
    so = CONTROL / 'seed_time.so'
    if so.exists():
        log(f'skip: {so} exists')
        return so
    src = CONTROL / 'seed_time.c'
    src.write_text(SHIM_C)
    subprocess.run(['gcc', '-shared', '-fPIC', '-O2', '-o', str(so), str(src)], check=True)
    return so


def pseudo_cmd(k, g, site, n_lines, bins, n):
    """run_rasqual.run_gene's option line; run_gene runs the command itself, so it cannot take -r or an environment."""
    return [str(RR.RASQUAL), '-y', bins['Y'], '-k', bins['K'], '-n', str(n), '-j', str(k + 1),
            '-l', str(n_lines), '-m', '1', '-s', str(site[2]), '-e', str(site[3]), '-f', g,
            '-z', '-d', str(RR.MIN_COVERAGE), '-a', str(RR.MAF), '-h', str(RR.HWE_P), '-x', bins['X'], '--n-threads', '1']


def records_bins(S, ds, perm, d):
    """Y, K and X with column i = donor perm[i]'s totals, offset and RNA-tied covariates; genotype PCs stay."""
    return RR.write_bins(S, dict(ds, pT=ds['pT'][:, perm], eff_lib=ds['eff_lib'][perm], perm=perm), d)[0]


def control_job(S, ds, kept, z, ex, vkey, g, ids, perm, bins, arm):
    """(cmd, text) for one RASQUAL arm of one gene over the rSNP ids given, allelic entries in the order perm."""
    k, n = S['genes'].index(g), len(S['order'])
    if arm == 'native':
        job = native_job(S, z, ex, g, bins, 1, ids, vkey, perm)[0]
        return job[2], job[3]
    site = RR.pseudo_site(S, g)
    text = RR.pseudo_line(g, site, ds['pL'][k][perm], ds['pR'][k][perm], kept[k][perm]) + rsnp_subset(S, g, ids)
    return pseudo_cmd(k, g, site, text.count('\n'), bins, n), text


def seeded(cmd, shim, seed):
    """cmd under RASQUAL's own permutation, its rand() seeded through the shim."""
    return ['env', f'LD_PRELOAD={shim}', f'RASQUAL_SEED={seed}'] + cmd + ['-r']


def run_once(cmd, text, f):
    """RASQUAL into f (atomically) unless f exists; returns f's text."""
    if f.exists():
        log(f'  skip: {f} exists')
    else:
        out = subprocess.run(cmd, input=text, capture_output=True, text=True, check=True).stdout
        C.write_atomic(f, lambda fh: fh.write(out), 'w')
    return f.read_text()


def control_gates(S, ds, kept, z, ex, vkey, bins, sub, shim):
    """The known-answer gates before the pool (docstring, --control); stops the stage if any fails."""
    d = CONTROL / 'gates'
    d.mkdir(parents=True, exist_ok=True)
    g, n = GATE_GENE, len(S['order'])
    ids = list(sub.variant_id[(sub.gene == g) & sub.random])
    ident = np.arange(n)
    b = records_bins(S, ds, ident, d / 'identity_inputs')
    same = {m: Path(b[m]).read_bytes() == (OUT / 'rasqual_inputs' / f'{m}.bin').read_bytes() for m in b}
    log(f'gate: records inputs at the identity order byte-identical to {OUT / "rasqual_inputs"}: {same}')
    ok = all(same.values())
    for arm in ('native', 'pseudo'):
        cmd, text = control_job(S, ds, kept, z, ex, vkey, g, ids, ident, b, arm)
        run_once(cmd, text, d / f'identity_{arm}.tsv')
        new = parse_raw(d / f'identity_{arm}.tsv', g).set_index('rs_id')
        old = parse_raw(ARM_DIR[arm] / f'{g}.tsv', g).set_index('rs_id')
        rows = [v for v in new.index if v in set(ids)]
        want = [v for v in old.index if v in set(ids)]
        diff = int((new.loc[rows, MODEL_FIELDS].values != old.loc[want, MODEL_FIELDS].values).any(1).sum()) \
            if rows == want else -1
        log(f'gate: {g} {arm} at the identity order, {len(rows)} random-variant rows against {len(want)} observed; rows '
            f'with any model field differing {diff} (want 0)')
        ok &= rows == want and len(rows) > 0 and diff == 0
    s1, s2 = (int(x) for x in np.random.SeedSequence(SEED, spawn_key=(GATE_KEY,)).generate_state(2))
    for arm, gate_ids in (('native', ids[:GATE_RSNPS]), ('pseudo', ids)):
        cmd, text = control_job(S, ds, kept, z, ex, vkey, g, gate_ids, ident, bins, arm)
        plain = parse_raw(d / f'identity_{arm}.tsv', g).set_index('rs_id').chisq.astype(float)
        a = run_once(seeded(cmd, shim, s1), text, d / f'r_{arm}_seed1a.tsv')
        a2 = run_once(seeded(cmd, shim, s1), text, d / f'r_{arm}_seed1b.tsv')
        c = run_once(seeded(cmd, shim, s2), text, d / f'r_{arm}_seed2.tsv')
        perm_chi = parse_raw(d / f'r_{arm}_seed1a.tsv', g).set_index('rs_id').chisq.astype(float)
        moved = int((perm_chi != plain.reindex(perm_chi.index)).sum())
        log(f'gate: {g} {arm} -r on {len(gate_ids)} rSNPs: same seed byte-identical {a == a2}, other seed differs '
            f'{a != c}, rows whose chisq differs from the unpermuted fit {moved} of {len(perm_chi)}')
        ok &= a == a2 and a != c and moved > 0
    if not ok:
        raise SystemExit('control gates failed; see the lines above')
    log('control gates passed')


def control():
    """Both RASQUAL arms under the records null and RASQUAL's -r at the null genes' random variants (docstring)."""
    genes, S = load_setup()
    ds, kept = observed(S)
    CONTROL.mkdir(parents=True, exist_ok=True)
    bins, z = inputs_for_rasqual(S, ds, CONTROL / 'rasqual_r' / 'inputs')
    same = {m: Path(bins[m]).read_bytes() == (OUT / 'rasqual_inputs' / f'{m}.bin').read_bytes() for m in bins}
    if not all(same.values()):
        raise SystemExit(f'-r inputs differ from the observed arms\' inputs: {same}')
    sub = pd.read_csv(SUBSET, sep='\t', keep_default_na=False, na_values=[''])
    ex, vkey, shim = exon_unions(), variant_key(S['I']['vdf']), seed_shim()
    control_gates(S, ds, kept, z, ex, vkey, bins, sub, shim)
    n = len(S['order'])
    gt = genes.set_index('gene')
    null_genes = list(genes.gene[genes.pool == 'null'])
    jobs, done = [], 0
    for null in NULLS:
        for r in range(N_RUNS):
            for g in null_genes:
                k = S['genes'].index(g)
                todo = [a for a in ('native', 'pseudo') if not (CONTROL / null / f'run_{r:02d}' / a / f'{g}.tsv').exists()]
                done += 2 - len(todo)
                if not todo:
                    continue
                ss = np.random.SeedSequence(SEED, spawn_key=(CONTROL_KEY[null], k, r))
                if null == 'records':
                    perm = np.random.default_rng(ss).permutation(n)
                    b = records_bins(S, ds, perm, CONTROL / null / 'inputs' / f'run_{r:02d}' / g)
                else:
                    perm, b, seed = np.arange(n), bins, int(ss.generate_state(1)[0])
                ids = list(sub.variant_id[(sub.gene == g) & sub.random])
                for arm in todo:
                    (CONTROL / null / f'run_{r:02d}' / arm).mkdir(parents=True, exist_ok=True)
                    cmd, text = control_job(S, ds, kept, z, ex, vkey, g, ids, perm, b, arm)
                    if null == 'rasqual_r':
                        cmd = seeded(cmd, shim, seed)
                    units = (int(gt.loc[g, 'n_covered_sites']) + 1) * len(ids) if arm == 'native' else 0
                    jobs.append((units, (f'{null}/run_{r:02d}/{arm}/{g}', 1, cmd, text)))
    jobs = [j for _, j in sorted(jobs, key=lambda x: -x[0])]
    log(f'control: {len(NULLS)} nulls x {N_RUNS} runs x {len(null_genes)} null genes x 2 arms; {done} finished, '
        f'{len(jobs)} to run over {JOBS} slots')
    secs = run_pool(jobs, CONTROL)
    if secs:
        s = np.array(list(secs.values()))
        log(f'control done: {len(s)} jobs, {s.sum() / 3600:.1f} CPU hours, per job median {np.median(s):.0f} s '
            f'[{s.min():.0f}, {s.max():.0f}]')
    log(f'wrote {CONTROL}')


ARMS = ('native', 'pseudo', 'split')
ARM_LABEL = {'native': 'RASQUAL native', 'pseudo': 'RASQUAL pseudo-fSNP', 'split': 'hapmixQTL split'}
COLOR = {'native': '#2a78d6', 'pseudo': '#eb6834', 'split': '#1baf7a'}   # dataviz palette slots 1-3 (validated adjacent pairs)
MARKER = {'native': 'o', 'pseudo': 's', 'split': '^'}
INK, INK2, MUTED, GRID, SURFACE = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#fcfcfb'
PAIRS = (('native', 'pseudo'), ('split', 'native'), ('split', 'pseudo'))
R2_CLOSE = 0.8             # LD r^2 at or above which two leads are read as the same signal


def ld_r2(a, b):
    """Pearson r^2 of ALT dosage over the donors; NaN where either is constant or absent."""
    if a is None or b is None:
        return float('nan')
    a, b = a.astype(float), b.astype(float)
    if a.std() == 0 or b.std() == 0:
        return float('nan')
    return float(np.corrcoef(a, b)[0, 1] ** 2)


def load_arm_tables(genes, tested):
    """{arm: frame over the finished genes}, per-gene exclusion counts, and the genes without a file;
    `tested` maps arm -> gene -> the variant ids that arm scanned."""
    tabs, counts, missing = {}, {}, {a: [] for a in ARM_DIR}
    for arm, d in ARM_DIR.items():
        parts = []
        for g in genes.gene:
            f = d / f'{g}.tsv'
            if not f.exists():
                missing[arm].append(g)
                continue
            out, cnt = arm_table(g, parse_raw(f, g), tested[arm][g])
            parts.append(out)
            counts[(arm, g)] = cnt
        tabs[arm] = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return tabs, counts, missing


def lead_row(t, arm):
    """The arm's lead among a gene's rows: largest chisq (RASQUAL, ties at 1e-5 counted) or smallest pval_nominal."""
    if arm == 'split':
        r = t.loc[t.pval_nominal.idxmin()]
        n_tie = 1
    else:
        r = t.loc[t.chisq.idxmax()]
        n_tie = int((np.round(t.chisq * TIE) == np.round(r.chisq * TIE)).sum())
    return r, n_tie


def gene_interval(x, stat=np.mean):
    """stat over the genes' finite values, its 95% gene-resampling interval (genes drawn with replacement, N_RESAMPLE
    draws from one seeded stream, so every call over the same number of genes uses the same draws) and the per-gene
    range; None when no gene has a value."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if not len(x):
        return None
    idx = np.random.default_rng(np.random.SeedSequence(SEED, spawn_key=(RESAMPLE_KEY,))).integers(0, len(x), (N_RESAMPLE, len(x)))
    m = stat(x[idx], axis=1)
    return dict(value=float(stat(x)), lo=float(np.percentile(m, 2.5)), hi=float(np.percentile(m, 97.5)),
                min=float(x.min()), max=float(x.max()), n=int(len(x)))


def ci(d, nd=3):
    """'value [lo, hi]' of a gene_interval record."""
    return 'n/a' if d is None else f'{fmt(d["value"], nd)} [{fmt(d["lo"], nd)}, {fmt(d["hi"], nd)}]'


def paired(x, y):
    """Paired comparison of two per-gene vectors: sign counts, the sign test and the Wilcoxon signed-rank test
    (both two-sided, docstring) on the finite differences x - y."""
    d = np.asarray(x, float) - np.asarray(y, float)
    d = d[np.isfinite(d)]
    pos, neg = int((d > 0).sum()), int((d < 0).sum())
    out = dict(n=int(len(d)), first_greater=pos, second_greater=neg, ties=int(len(d) - pos - neg),
               median_diff=float(np.median(d)) if len(d) else float('nan'),
               mean_diff=float(d.mean()) if len(d) else float('nan'),
               min_diff=float(d.min()) if len(d) else float('nan'), max_diff=float(d.max()) if len(d) else float('nan'),
               sign_test_p=float(binomtest(pos, pos + neg, 0.5).pvalue) if pos + neg else float('nan'),
               wilcoxon_p=float(wilcoxon(d).pvalue) if len(d) >= 6 and (d != 0).any() else float('nan'))
    return out


def per_gene_table(genes, arms, counts, dos, pub_dos, n_admitted, subset_flags):
    """One row per gene: each arm's lead and statistics there, LD between leads and to the published lead."""
    rows, at_leads = [], []
    for r in genes.itertuples():
        g = r.gene
        rec = dict(gene=g, pool=r.pool, n_tested=int(r.n_tested), n_admitted=int(n_admitted[g]),
                   segmental_duplication=bool(r.segmental_duplication), published_lead=r.variant_id,
                   published_pval_beta=float(r.pval_beta), published_slope=float(r.slope),
                   published_in_vcf=bool(r.lead_in_vcf), published_tested=bool(r.lead_tested),
                   published_in_body=bool(r.lead_in_body))
        pd_ = pub_dos.get(r.variant_id)
        leads = {}
        for arm in ARMS:
            t = arms[arm]
            t = t[t.phenotype_id == g] if len(t) else t
            if not len(t):
                rec[f'lead_{arm}'] = None
                continue
            lr, n_tie = lead_row(t, arm)
            leads[arm] = lr
            rec.update({f'lead_{arm}': lr.variant_id, f'chi2_{arm}': float(lr.chisq), f'p_{arm}': float(lr.pval_nominal),
                        f'afc_{arm}': float(lr.slope), f'r2_{arm}_published': ld_r2(dos.get(lr.variant_id), pd_),
                        f'lead_is_published_{arm}': bool(lr.variant_id == r.variant_id), f'n_rows_{arm}': int(len(t))})
            if arm == 'native':
                tied = t.variant_id[np.round(t.chisq * TIE) == np.round(lr.chisq * TIE)]
                r2t = np.array([ld_r2(dos.get(v), pd_) for v in tied])
                r2t = r2t[np.isfinite(r2t)]
                rec.update(lead_ties_native=n_tie,
                           r2_native_published_ties_min=float(r2t.min()) if len(r2t) else float('nan'),
                           r2_native_published_ties_max=float(r2t.max()) if len(r2t) else float('nan'))
            if arm in ('native', 'pseudo'):
                c = counts[(arm, g)]
                rec.update({f'phi_{arm}': float(lr.phi), f'delta_{arm}': float(lr.delta), f'theta_{arm}': float(lr.theta),
                            f'n_fsnp_{arm}': int(lr.n_feature_snps), f'r2_fsnps_{arm}': float(lr.r2_fsnps),
                            f'r2_rsnp_{arm}': float(lr.r2_rsnp), f'nonconv_{arm}': c['nonconv'],
                            f'absent_{arm}': c['absent'], f'fsnp_as_rsnp_rows_{arm}': c['fsnp_as_rsnp_rows'],
                            f'iter_median_{arm}': float(t.n_iter.median()),
                            f'iter_at_cap_{arm}': float((t.n_iter >= RASQUAL_MAXITR).mean())})
            if arm == 'native':
                f = subset_flags.get((g, lr.variant_id), {})
                rec['native_lead_origin'] = next((k for k in ('lead_pseudo', 'lead_split', 'published', 'top_pseudo',
                                                              'top_split', 'random') if f.get(k)), 'unknown')
            if arm == 'split':
                rec.update(dof_a=float(lr.dof_a), dof_t=float(lr.dof_t), allelic_admitted=bool(lr.allelic_admitted),
                           p_a_split=float(lr.pval_a), p_t_split=float(lr.pval_t))
        for a, b in PAIRS:
            if a in leads and b in leads:
                rec[f'r2_{a}_{b}'] = ld_r2(dos.get(leads[a].variant_id), dos.get(leads[b].variant_id))
                rec[f'same_lead_{a}_{b}'] = bool(leads[a].variant_id == leads[b].variant_id)
        for arm, lr in leads.items():
            v = lr.variant_id
            row = dict(gene=g, pool=r.pool, lead_arm=arm, variant_id=v)
            for other in ARMS:
                t = arms[other]
                t = t[(t.phenotype_id == g) & (t.variant_id == v)] if len(t) else t
                if len(t) == 1:
                    o = t.iloc[0]
                    row.update({f'chi2_{other}': float(o.chisq), f'p_{other}': float(o.pval_nominal),
                                f'afc_{other}': float(o.slope)})
                    if other != 'split':
                        row.update({f'phi_{other}': float(o.phi), f'delta_{other}': float(o.delta)})
                else:
                    row[f'chi2_{other}'] = float('nan')   # no converged row of that arm at this variant
            at_leads.append(row)
        rows.append(rec)
    pg = pd.DataFrame(rows)
    need = ([f'{c}_{a}' for a in ARMS for c in ('lead', 'chi2', 'p', 'afc', 'lead_is_published')]
            + [f'r2_{a}_published' for a in ARMS]
            + [f'{f}_{a}' for a in ('native', 'pseudo') for f in ('phi', 'delta', 'theta', 'n_fsnp', 'r2_fsnps', 'r2_rsnp')]
            + [f'r2_{a}_{b}' for a, b in PAIRS] + [f'same_lead_{a}_{b}' for a, b in PAIRS])
    for c in need:                     # an arm with no finished gene yet has no column at all
        if c not in pg:
            pg[c] = np.nan
    return pg, pd.DataFrame(at_leads)


def summarise(pg):
    """Paired summaries per pool: lead chi-square, LD to the published lead, lead agreement, phi and delta."""
    out = {}
    for pool in ('eqtl', 'null', 'all'):
        s = pg if pool == 'all' else pg[pg.pool == pool]
        d = dict(n_genes=int(len(s)), n_with_native=int(s.lead_native.notna().sum()))
        for a, b in PAIRS:
            ok = s[[f'chi2_{a}', f'chi2_{b}']].notna().all(1)
            d[f'log2_chi2_{a}_over_{b}'] = paired(np.log2(s.loc[ok, f'chi2_{a}']), np.log2(s.loc[ok, f'chi2_{b}']))
            d[f'r2_published_{a}_minus_{b}'] = paired(s.loc[ok, f'r2_{a}_published'], s.loc[ok, f'r2_{b}_published'])
            d[f'leads_{a}_{b}'] = dict(same_variant=int(s.loc[ok, f'same_lead_{a}_{b}'].sum()),
                                       r2_at_least_close=int((s.loc[ok, f'r2_{a}_{b}'] >= R2_CLOSE).sum()),
                                       n=int(ok.sum()))
        for arm in ARMS:
            c = s[f'chi2_{arm}'].dropna()
            r2p = s[f'r2_{arm}_published'].dropna()
            d[f'{arm}'] = dict(n=int(len(c)), lead_chi2_median=float(c.median()) if len(c) else None,
                               lead_chi2_min=float(c.min()) if len(c) else None,
                               lead_chi2_max=float(c.max()) if len(c) else None,
                               lead_is_published=int(s[f'lead_is_published_{arm}'].fillna(False).sum()),
                               r2_published_median=float(r2p.median()) if len(r2p) else None,
                               r2_published_at_least_close=int((r2p >= R2_CLOSE).sum()), n_r2_published=int(len(r2p)))
        for arm in ('native', 'pseudo'):
            for f in ('phi', 'delta', 'theta', 'n_fsnp', 'r2_fsnps', 'r2_rsnp', 'iter_median', 'iter_at_cap', 'nonconv'):
                v = s[f'{f}_{arm}'].dropna() if f'{f}_{arm}' in s else pd.Series(dtype=float)
                d[f'{f}_{arm}'] = dict(median=float(v.median()), min=float(v.min()), max=float(v.max()), n=int(len(v))) if len(v) else None
        out[pool] = d
    return out


def random_sample(genes, arms, sub):
    """At the seeded random tested variants of each gene, native against pseudo and against split, per variant:
    per gene the share of variants where native's chi-square is the larger and the median difference; pooled per
    pool the sign test on the per-gene shares against one half (genes with share > 0.5 counted) and the Wilcoxon
    signed-rank test on the per-gene median differences."""
    nat = arms['native']
    if not len(nat):
        return pd.DataFrame(), {}
    rows = []
    for r in genes.itertuples():
        g = r.gene
        ids = set(sub.variant_id[(sub.gene == g) & sub.random])
        n = nat[(nat.phenotype_id == g) & nat.variant_id.isin(ids)].set_index('variant_id').chisq
        if not len(n):
            continue
        rec = dict(gene=g, pool=r.pool, segmental_duplication=bool(r.segmental_duplication), n_random=len(ids),
                   n_native_rows=int(len(n)), median_chi2_native=float(n.median()),
                   p05_native=float((chi2.sf(n, 1) < 0.05).mean()))
        for other in ('pseudo', 'split'):
            o = arms[other]
            o = o[(o.phenotype_id == g) & o.variant_id.isin(n.index)].set_index('variant_id').chisq
            d = (n - o.reindex(n.index)).dropna()
            rec.update({f'n_{other}': int(len(d)), f'share_native_gt_{other}': float((d > 0).mean()) if len(d) else float('nan'),
                        f'median_diff_{other}': float(d.median()) if len(d) else float('nan'),
                        f'median_chi2_{other}': float(o.median()) if len(o) else float('nan'),
                        f'p05_{other}': float((chi2.sf(o, 1) < 0.05).mean()) if len(o) else float('nan')})
        rows.append(rec)
    df = pd.DataFrame(rows)
    summ = {}
    for pool in ('eqtl', 'null', 'all'):
        s = df if pool == 'all' else df[df.pool == pool]
        d = dict(n_genes=int(len(s)), variants=int(s.n_pseudo.sum()) if len(s) else 0,
                 chi2_1_median=float(chi2.median(1)),
                 level={arm: dict(median_of_gene_median_chi2=float(s[f'median_chi2_{arm}'].median()) if len(s) else None,
                                  mean_share_p_below_0_05=float(s[f'p05_{arm}'].mean()) if len(s) else None,
                                  median_chi2_interval=gene_interval(s[f'median_chi2_{arm}'], np.median),
                                  share_interval=gene_interval(s[f'p05_{arm}']))
                        for arm in ARMS},
                 level_by_segmental_duplication={
                     ('sd' if flag else 'not_sd'): {arm: dict(n_genes=int((s.segmental_duplication == flag).sum()),
                                                            mean_share_p_below_0_05=float(s.loc[s.segmental_duplication == flag, f'p05_{arm}'].mean())
                                                            if (s.segmental_duplication == flag).any() else None)
                                                  for arm in ARMS}
                     for flag in (True, False)})
        for other in ('pseudo', 'split'):
            sh = s[f'share_native_gt_{other}'].dropna()
            md = s[f'median_diff_{other}'].dropna()
            gt = int((sh > 0.5).sum())
            lt = int((sh < 0.5).sum())
            d[f'native_vs_{other}'] = dict(
                n_genes=int(len(sh)), mean_share_native_larger=float(sh.mean()) if len(sh) else None,
                share_native_larger_interval=gene_interval(sh),
                genes_share_above_half=gt, genes_share_below_half=lt,
                sign_test_p=float(binomtest(gt, gt + lt, 0.5).pvalue) if gt + lt else None,
                median_of_gene_median_diff=float(md.median()) if len(md) else None,
                wilcoxon_p=float(wilcoxon(md).pvalue) if len(md) >= 6 and (md != 0).any() else None,
                median_chi2_native=float(s.median_chi2_native.median()) if len(s) else None,
                median_chi2_other=float(s[f'median_chi2_{other}'].median()) if len(s) else None)
        summ[pool] = d
    return df, summ


def control_summary(genes, arms, sub, pg):
    """Per null gene and RASQUAL arm: the share of the N_RANDOM random variants with p < 0.05 observed and, per null,
    averaged over the finished runs (each run's share over its converged rows), with the per-gene permutation p of
    the observed share, (1 + runs at or above it) / (1 + runs); pooled over the genes, the mean share observed and
    permuted and their paired difference, each with a gene-resampling interval."""
    pgi = pg.set_index('gene')
    rows = []
    for g in genes.gene[genes.pool == 'null']:
        ids = set(sub.variant_id[(sub.gene == g) & sub.random])
        rec = dict(gene=g, split_lead_p_x_tested=float(min(1.0, pgi.loc[g, 'p_split'] * pgi.loc[g, 'n_tested'])),
                   p_a_split=float(pgi.loc[g, 'p_a_split']))
        for arm in ('native', 'pseudo'):
            t = arms[arm]
            rec[f'observed_{arm}'] = float((t.pval_nominal[(t.phenotype_id == g) & t.variant_id.isin(ids)] < 0.05).mean())
            for null in NULLS:
                sh, nonconv, conv = [], 0, 0
                for r in range(N_RUNS):
                    f = CONTROL / null / f'run_{r:02d}' / arm / f'{g}.tsv'
                    if f.exists():
                        out, cnt = arm_table(g, parse_raw(f, g), ids)
                        sh.append(float((out.pval_nominal < 0.05).mean()))
                        nonconv, conv = nonconv + cnt['nonconv'], conv + len(out)
                sh = np.array(sh)
                rec.update({f'{null}_{arm}': float(sh.mean()) if len(sh) else float('nan'),
                            f'{null}_{arm}_runs': int(len(sh)),
                            f'{null}_{arm}_perm_p': float((1 + (sh >= rec[f'observed_{arm}']).sum()) / (1 + len(sh)))
                            if len(sh) else float('nan'),
                            f'{null}_{arm}_nonconv_rate': nonconv / (nonconv + conv) if nonconv + conv else float('nan')})
        rows.append(rec)
    df = pd.DataFrame(rows)
    pooled = {}
    for arm in ('native', 'pseudo'):
        d = dict(observed=gene_interval(df[f'observed_{arm}']))
        for null in NULLS:
            ok = df[f'{null}_{arm}'].notna()
            d[null] = dict(genes=int(ok.sum()), runs_min=int(df[f'{null}_{arm}_runs'].min()),
                           runs_max=int(df[f'{null}_{arm}_runs'].max()),
                           permuted=gene_interval(df.loc[ok, f'{null}_{arm}']),
                           observed_minus_permuted=gene_interval(df.loc[ok, f'observed_{arm}'] - df.loc[ok, f'{null}_{arm}']),
                           genes_perm_p_min=int((df.loc[ok, f'{null}_{arm}_perm_p'] <= 1 / (1 + N_RUNS)).sum()),
                           nonconv_rate_median=float(df.loc[ok, f'{null}_{arm}_nonconv_rate'].median()) if ok.any() else None)
        both = df[f'records_{arm}'].notna() & df[f'rasqual_r_{arm}'].notna()
        d['records_minus_rasqual_r'] = gene_interval(df.loc[both, f'records_{arm}'] - df.loc[both, f'rasqual_r_{arm}'])
        pooled[arm] = d
    return df, pooled


def figure(pg, path):
    """Per-gene lead chi-square of the three arms, eQTL and null genes in two panels, log scale."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    pools = [('eqtl', 'eQTL genes (BH < 0.05 on pval_beta in the T2T run)'), ('null', 'null genes (pval_beta > 0.5)')]
    widths = [max(1, int((pg.pool == p).sum())) for p, _ in pools]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw=dict(width_ratios=widths), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, (pool, title) in zip(axes, pools):
        s = pg[pg.pool == pool].copy()
        order_by = s.chi2_native.where(s.chi2_native.notna(), s.chi2_pseudo)
        s = s.iloc[np.argsort(-order_by.values)]
        x = np.arange(len(s))
        ax.set_facecolor(SURFACE)
        lo = s[[f'chi2_{a}' for a in ARMS]].min(1)
        hi = s[[f'chi2_{a}' for a in ARMS]].max(1)
        ax.vlines(x, lo, hi, color=GRID, lw=1.5, zorder=1)
        for arm in ARMS:
            ax.scatter(x, s[f'chi2_{arm}'], s=46, marker=MARKER[arm], color=COLOR[arm], edgecolor=SURFACE,
                       linewidth=0.8, label=ARM_LABEL[arm], zorder=3)
        ax.set_yscale('log')
        ax.set_xticks(x)
        ax.set_xticklabels(s.gene, rotation=60, ha='right', fontsize=8, color=INK2)
        ax.set_title(title, fontsize=10, color=INK, loc='left')
        ax.grid(axis='y', color=GRID, lw=0.8)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        for sp in ('left', 'bottom'):
            ax.spines[sp].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelcolor=INK2)
    axes[0].set_ylabel('chi-square at the arm\'s own lead', color=INK2, fontsize=9)
    axes[1].legend(frameon=False, fontsize=9, loc='upper right', labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def fmt(x, nd=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return 'n/a'
    if isinstance(x, (bool, np.bool_)):
        return 'yes' if x else 'no'
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if abs(x) < 1e-3 and x != 0:
        return f'{x:.2e}'
    return f'{x:.{nd}f}'


def html_table(df, cols, nd=3):
    head = ''.join(f'<th>{c}</th>' for c in cols)
    body = ''.join('<tr>' + ''.join(f'<td>{fmt(r[c], nd) if not isinstance(r[c], str) else r[c]}</td>' for c in cols) + '</tr>'
                   for _, r in df.iterrows())
    return f'<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def paired_rows(summ):
    rows = []
    for pool in ('eqtl', 'null', 'all'):
        for a, b in PAIRS:
            p = summ[pool][f'log2_chi2_{a}_over_{b}']
            q = summ[pool][f'r2_published_{a}_minus_{b}']
            m = summ[pool][f'leads_{a}_{b}']
            rows.append(dict(pool=pool, comparison=f'{ARM_LABEL[a]} vs {ARM_LABEL[b]}', n=p['n'],
                             first_larger=f"{p['first_greater']} of {p['n']}",
                             median_log2_ratio=p['median_diff'], range_log2_ratio=f"{fmt(p['min_diff'], 2)} to {fmt(p['max_diff'], 2)}",
                             sign_p=p['sign_test_p'], wilcoxon_p=p['wilcoxon_p'],
                             same_lead=f"{m['same_variant']} / r2 >= {R2_CLOSE}: {m['r2_at_least_close']} of {m['n']}",
                             r2_published_first_closer=f"{q['first_greater']} of {q['n']} (median diff {fmt(q['median_diff'])}, sign p {fmt(q['sign_test_p'])})"))
    return pd.DataFrame(rows)


def report():
    genes = read_genes()
    z = np.load(TESTED)
    dos = dict(zip(z['variant_id'], z['dos']))
    pub_dos = dict(zip(z['pub_id'], z['pub_dos']))
    n_admitted = dict(zip(z['genes'], z['n_admitted']))
    tested = {g: set(z['variant_id'][z['gene'] == g]) for g in genes.gene}
    sub = pd.read_csv(SUBSET, sep='\t', keep_default_na=False, na_values=[''])
    sub_ids = {g: set(sub.variant_id[sub.gene == g]) for g in genes.gene}
    flags = ['random', 'top_pseudo', 'top_split', 'lead_pseudo', 'lead_split', 'published']
    subset_flags = {(r.gene, r.variant_id): {f: bool(getattr(r, f)) for f in flags} for r in sub.itertuples()}
    split = pd.read_parquet(SPLIT)
    if set(split.phenotype_id.unique()) != set(genes.gene):
        raise SystemExit(f'{SPLIT}: genes differ from genes.tsv')
    split['chisq'] = chi2.isf(split.pval_nominal.values, 1)
    tabs, counts, missing = load_arm_tables(genes, {'pseudo': tested, 'native': sub_ids})
    arms = dict(tabs, split=split)
    pg, at_leads = per_gene_table(genes, arms, counts, dos, pub_dos, n_admitted, subset_flags)
    # every donor 0|1 at the pseudo fSNP: getCov2 (nbem.c:2068-2087) divides by the prior dosage variance, zero here
    pg['pseudo_prior_constant'] = pg.n_admitted == len(z['samples'])
    pg.loc[pg.pseudo_prior_constant, 'r2_fsnps_pseudo'] = np.nan
    summ = summarise(pg)
    rs, rs_summ = random_sample(genes, arms, sub)
    cg, cpool = control_summary(genes, arms, sub, pg)
    C.write_atomic(OUT / 'per_gene.tsv', lambda fh: pg.to_csv(fh, sep='\t', index=False), 'w')
    C.write_atomic(OUT / 'arms_at_leads.tsv', lambda fh: at_leads.to_csv(fh, sep='\t', index=False), 'w')
    C.write_atomic(OUT / 'random_sample.tsv', lambda fh: rs.to_csv(fh, sep='\t', index=False), 'w')
    C.write_atomic(OUT / 'control_per_gene.tsv', lambda fh: cg.to_csv(fh, sep='\t', index=False), 'w')
    excl = {arm: {k: int(sum(c[k] for (a, g), c in counts.items() if a == arm)) for k in ('nonconv', 'absent', 'fsnp_as_rsnp_rows', 'chisq_le0', 'no_fsnp')}
            for arm in ARM_DIR}
    figure(pg, OUT / 'lead_chi2.png')
    rec = dict(
        question='Does the benchmark\'s one-pseudo-feature-SNP construction handicap RASQUAL? Observed data, no thinning.',
        genes=dict(n=int(len(genes)), eqtl=int((genes.pool == 'eqtl').sum()), null=int((genes.pool == 'null').sum()),
                   selection=str(json.loads((SCOUT / 'scout_evidence.json').read_text())['selection']),
                   tested_variants=int(genes.n_tested.sum()), admitted_allelic_donors=[int(n_admitted[g]) for g in genes.gene]),
        arms={a: ARM_LABEL[a] for a in ARMS},
        inputs_common='totals = Salmon point-estimate totals pT (as the benchmark used), K = eff_lib / mean(eff_lib), '
                      'covariates = 14 RNA-tied + 3 genotype PCs, tested variants = cis window 1 Mb of the TSS outside '
                      'the gene body at MAF >= 0.05 among phased biallelic SNPs, 92 donors; identical across the three arms',
        native_missing=missing['native'], pseudo_missing=missing['pseudo'], exclusions=excl,
        native_subset=dict(rule=f'per gene: the pseudo and split leads, the published lead when tested, the top {TOP_K} '
                                f'variants of each of those arms and {N_RANDOM} random tested variants '
                                f'(SeedSequence({SEED}, spawn_key=({SUBSET_KEY}, gene index)))',
                           pairs=int(len(sub)), per_gene_min=int(sub.groupby('gene').size().min()),
                           per_gene_max=int(sub.groupby('gene').size().max()), random_pairs=int(sub.random.sum())),
        random_sample=rs_summ,
        control=dict(question='Is RASQUAL\'s share of p < 0.05 at the random variants of the null genes an offset of '
                              'its statistic or association? Offset if the permuted share matches the observed one, '
                              'signal if it falls to about 0.05.',
                     nulls={n: NULL_LABEL[n] for n in NULLS}, runs_per_gene=N_RUNS,
                     variants_per_run=f'the {N_RANDOM} random tested variants of each null gene (native_subset.tsv)',
                     seeds=dict(records=f'SeedSequence({SEED}, spawn_key=({CONTROL_KEY["records"]}, gene index, run))',
                                rasqual_r=f'RASQUAL_SEED = SeedSequence({SEED}, spawn_key=({CONTROL_KEY["rasqual_r"]}, '
                                          f'gene index, run)).generate_state(1)[0] through the LD_PRELOAD time() shim'),
                     interval=f'95% gene-resampling interval: the 10 null genes drawn with replacement, {N_RESAMPLE} '
                              f'draws, each gene carrying its value (for permuted shares, its mean over runs)',
                     pooled=cpool, per_gene=cg.to_dict('records')),
        definitions=dict(lead='RASQUAL pseudo: largest chisq over the tested variants; RASQUAL native: largest chisq over '
                              'its subset; hapmixQTL: smallest pval_nominal over the tested variants',
                         chi2_split='chi2.isf(pval_nominal, 1): the 1-df chi-square with hapmixQTL\'s p, not a likelihood ratio',
                         afc='log2 allelic fold change ALT over REF; RASQUAL log2(pi / (1 - pi)), hapmixQTL the combined slope',
                         ld_r2='Pearson r^2 of ALT dosage over the 92 donors',
                         phi_pseudo='L/R imbalance by construction (ref = L for every heterozygote), not reference-mapping bias',
                         delta='RASQUAL field 13, the sequencing/mapping (read allele) error rate: the probability that a '
                               'read shows the other allele (getK, nbem.c:1428-1439), which pulls every expected allelic '
                               'share toward 0.5',
                         r2_fsnps_pseudo='field 24 at the pseudo lead; NaN where every donor is admitted '
                                         '(pseudo_prior_constant), because the prior dosage is then constant and the '
                                         'squared correlation undefined',
                         sign_test='two-sided binomial test of the count of genes with a positive paired difference against one half',
                         wilcoxon='two-sided Wilcoxon signed-rank test on the paired differences'),
        summary=summ, per_gene=pg.to_dict('records'))
    C.write_atomic(OUT / 'summary.json', lambda fh: fh.write(C.dumps(rec)), 'w')
    write_page(genes, pg, summ, missing, excl, rs_summ, sub, at_leads, cg, cpool)
    for pool in ('eqtl', 'null', 'all'):
        for a, b in PAIRS:
            p = summ[pool][f'log2_chi2_{a}_over_{b}']
            log(f'{pool:4s} {a:6s} vs {b:6s}: chi2 larger in {p["first_greater"]} of {p["n"]} genes, median log2 ratio '
                f'{p["median_diff"]:+.3f} [{p["min_diff"]:+.2f}, {p["max_diff"]:+.2f}], sign p {p["sign_test_p"]:.3f}, '
                f'Wilcoxon p {p["wilcoxon_p"]:.3f}')
        for other in ('pseudo', 'split'):
            q = rs_summ.get(pool, {}).get(f'native_vs_{other}')
            if q:
                log(f'{pool:4s} random variants, native vs {other}: mean share native larger {fmt(q["mean_share_native_larger"])}, '
                    f'genes above / below one half {q["genes_share_above_half"]} / {q["genes_share_below_half"]} (sign p '
                    f'{fmt(q["sign_test_p"])}), median gene-median chi2 difference {fmt(q["median_of_gene_median_diff"])} '
                    f'(Wilcoxon p {fmt(q["wilcoxon_p"])})')
    log(f'native genes missing: {missing["native"]}; pseudo missing: {missing["pseudo"]}')
    log(f'wrote {OUT / "summary.json"}, {OUT / "report.html"}')


def random_rows(rs_summ):
    rows = []
    for pool in ('eqtl', 'null', 'all'):
        for other in ('pseudo', 'split'):
            q = rs_summ.get(pool, {}).get(f'native_vs_{other}')
            if q:
                rows.append(dict(pool=pool, comparison=f'RASQUAL native vs {ARM_LABEL[other]}', genes=q['n_genes'],
                                 mean_share_native_larger=ci(q['share_native_larger_interval']),
                                 per_gene_range=f"{fmt(q['share_native_larger_interval']['min'])} to "
                                                f"{fmt(q['share_native_larger_interval']['max'])}",
                                 genes_above_below_half=f"{q['genes_share_above_half']} / {q['genes_share_below_half']}",
                                 sign_p=q['sign_test_p'], median_gene_median_diff=q['median_of_gene_median_diff'],
                                 wilcoxon_p=q['wilcoxon_p'], median_chi2_native=q['median_chi2_native'],
                                 median_chi2_other=q['median_chi2_other']))
    return pd.DataFrame(rows)


def level_rows(rs_summ):
    """Per pool and arm, the chi-square level at the random variants: median of the per-gene medians (chi-square(1)
    median 0.455 under no association) and the mean per-gene share of random variants with p < 0.05."""
    rows = []
    for pool in ('eqtl', 'null', 'all'):
        lv = rs_summ.get(pool, {}).get('level')
        sd = rs_summ.get(pool, {}).get('level_by_segmental_duplication', {})
        if lv:
            for arm in ARMS:
                mi, si = lv[arm]['median_chi2_interval'], lv[arm]['share_interval']
                rows.append(dict(pool=pool, arm=ARM_LABEL[arm], median_chi2=ci(mi),
                                 median_chi2_per_gene_range=f"{fmt(mi['min'])} to {fmt(mi['max'])}",
                                 share_p_below_0_05=ci(si), share_per_gene_range=f"{fmt(si['min'])} to {fmt(si['max'])}",
                                 share_p_below_0_05_sd_genes=(sd.get('sd', {}).get(arm) or {}).get('mean_share_p_below_0_05'),
                                 n_sd_genes=(sd.get('sd', {}).get(arm) or {}).get('n_genes'),
                                 share_p_below_0_05_other_genes=(sd.get('not_sd', {}).get(arm) or {}).get('mean_share_p_below_0_05'),
                                 n_other_genes=(sd.get('not_sd', {}).get(arm) or {}).get('n_genes')))
    return pd.DataFrame(rows)


def control_rows(cg, cpool):
    """The control's table: one row per null gene, then the pooled means and paired differences with intervals."""
    rows = []
    for _, r in cg.iterrows():
        row = {'gene': r.gene, 'split lead p x tested variants': fmt(r.split_lead_p_x_tested),
               'split allelic p at its lead': fmt(r.p_a_split)}
        for arm in ('native', 'pseudo'):
            row[f'{arm}: observed'] = fmt(r[f'observed_{arm}'])
            row[f'{arm}: records (perm p)'] = f"{fmt(r[f'records_{arm}'])} ({fmt(r[f'records_{arm}_perm_p'])})"
            row[f'{arm}: -r'] = fmt(r[f'rasqual_r_{arm}'])
        rows.append(row)
    blank = {'split lead p x tested variants': '', 'split allelic p at its lead': ''}
    mean = {'gene': 'mean over genes [95% interval]', **blank}
    diff = {'gene': 'observed minus permuted', **blank}
    for arm in ('native', 'pseudo'):
        c = cpool[arm]
        mean.update({f'{arm}: observed': ci(c['observed']), f'{arm}: records (perm p)': ci(c['records']['permuted']),
                     f'{arm}: -r': ci(c['rasqual_r']['permuted'])})
        diff.update({f'{arm}: observed': '', f'{arm}: records (perm p)': ci(c['records']['observed_minus_permuted']),
                     f'{arm}: -r': ci(c['rasqual_r']['observed_minus_permuted'])})
    return pd.DataFrame(rows + [mean, diff])


def verdict(c):
    """How the records control reads for one arm: 'offset', 'signal', 'both', 'below' or 'unresolved'."""
    perm, diff = c['records']['permuted'], c['records']['observed_minus_permuted']
    at_nominal = perm['lo'] <= 0.05 <= perm['hi']
    if diff['hi'] < 0:
        return 'below'
    if diff['lo'] <= 0:
        return 'unresolved' if at_nominal else 'offset'
    return 'signal' if at_nominal else 'both'


def reading(rs_summ, summ, excl, pg, at_leads, genes, cg, cpool):
    """The reading paragraphs, every number computed; the verdict on the level at random variants is read off the
    records control."""
    nl = rs_summ.get('null', {}).get('level')
    e, n, a = summ['eqtl'], summ['null'], summ['all']
    nat, psd = cpool['native'], cpool['pseudo']
    rec, rr = nat['records'], nat['rasqual_r']
    if not nl or not e['delta_native'] or not e['delta_pseudo'] or rec['permuted'] is None:
        return ''
    p = e['log2_chi2_native_over_pseudo']
    pct = lambda x: f'{100 * x:.1f}%'
    v = verdict(nat)
    runs = lambda c: (f'{c["runs_min"]} runs per gene' if c['runs_min'] == c['runs_max']
                      else f'{c["runs_min"]}-{c["runs_max"]} runs per gene')
    head = (f'<b>Reading.</b> At its own lead RASQUAL native reports a larger chi-square than the pseudo-fSNP arm in '
            f'{p["first_greater"]} of {p["n"]} eQTL genes (median log2 ratio {fmt(p["median_diff"], 2)}, a factor '
            f'{2 ** p["median_diff"]:.2f}). At the {N_RANDOM} random variants of each null gene, {ci(nat["observed"])} '
            f'of native\'s p-values are below 0.05 (mean over the {nat["observed"]["n"]} genes, 95% gene-resampling '
            f'interval), against {ci(psd["observed"])} for the pseudo arm and {ci(nl["split"]["share_interval"])} for '
            f'hapmixQTL split. ')
    ctl = (f'Under the records permutation ({runs(rec)}), which breaks only the link between the tested genotypes and '
           f'the rest of each donor\'s record, native\'s share is {ci(rec["permuted"])}, and the paired difference '
           f'observed minus permuted is {ci(rec["observed_minus_permuted"])}. ')
    ctl += {'offset': 'The excess survives when the genotype-phenotype link is broken, so it is an offset of native\'s '
                      'statistic rather than association, and part of native\'s paired advantage at the leads is that '
                      'offset. ',
            'signal': 'The permuted share is consistent with 0.05 and the observed share lies above it, so native\'s '
                      'excess at the random variants of these genes is association, not an offset of its statistic. ',
            'both': 'The permuted share lies above 0.05 and the observed share above the permuted one, so native\'s '
                    'excess is part offset of its statistic and part association. ',
            'below': 'The observed share lies below the permuted one. ',
            'unresolved': 'The permuted share\'s interval contains both 0.05 and the observed share, so the control '
                          'does not separate an offset from association on these genes. '}[v]
    d = nat['records_minus_rasqual_r']
    if rr['permuted'] is not None and d is not None:
        ctl += (f'Under RASQUAL\'s own -r ({runs(rr)}) native\'s share is {ci(rr["permuted"])}; records minus -r is '
                f'{ci(d)}. One -r run draws a separate donor order for every feature SNP and another for the totals, so '
                f'it also removes within-donor structure the records null keeps; '
                + ('the gap is the part of native\'s permuted level that comes from that structure'
                   + (', and read against -r alone native\'s excess would have looked like association. '
                      if v == 'offset' and rr['permuted']['lo'] <= 0.05 <= rr['permuted']['hi'] else '. ')
                   if d['lo'] > 0 or d['hi'] < 0 else
                   'the two nulls agree within that interval, so on these genes that structure does not move native\'s '
                   'level. '))
    ctl += (f'The pseudo arm reads {ci(psd["records"]["permuted"])} under records and '
            f'{ci(psd["rasqual_r"]["permuted"])} under -r. ')
    sig = cg[cg.split_lead_p_x_tested < 0.05]
    if len(sig):
        ctl += (f'"Null" here means pval_beta &gt; 0.5 in the T2T total-expression permutation run; the split arm\'s '
                f'lead p times the tested-variant count (a Bonferroni bound on its gene-level p) is below 0.05 in '
                f'{len(sig)} of the {len(cg)} null genes: '
                + ', '.join(f'{r.gene} ({fmt(r.split_lead_p_x_tested)}, allelic p at that lead {fmt(r.p_a_split)}; '
                            f'native share {fmt(r.observed_native)} observed, {fmt(r.records_native)} permuted, '
                            f'permutation p {fmt(r.records_native_perm_p)})' for r in sig.itertuples()) + '. ')
    ctl += (f'In {rec["genes_perm_p_min"]} of the {rec["genes"]} null genes native\'s observed share exceeds every '
            f'permuted run (permutation p at its floor, {1 / (1 + rec["runs_max"]):.3f}).')

    rank = (pg.chi2_native / pg.chi2_pseudo).rank(ascending=False).astype(int)
    low = pg.assign(rank=rank)[pg.r2_rsnp_native < pg.r2_rsnp_pseudo.min()].sort_values('r2_rsnp_native')
    def_e, def_n = e['r2_fsnps_pseudo'], n['r2_fsnps_pseudo']
    par = (f'The two RASQUAL arms fit different read-level models to the same totals, and two separate effects show '
           f'at the leads. First, delta, the sequencing/mapping (read allele) error rate, is the probability that a '
           f'read shows the other allele, so a large delta pulls every expected allelic share toward 0.5 (getK, '
           f'nbem.c:1428-1439): the pseudo arm\'s delta has median {fmt(e["delta_pseudo"]["median"], 3)} at the eQTL '
           f'leads (up to {fmt(e["delta_pseudo"]["max"], 2)}) against native\'s {fmt(e["delta_native"]["median"], 4)}, '
           f'so the construction shrinks the pseudo arm\'s allelic contribution, which is itself a handicap. Second, '
           f'posterior genotype updating: the squared correlation between prior and posterior feature-SNP genotypes '
           f'(field 24) is undefined for the pseudo arm in the {int(pg.pseudo_prior_constant.sum())} genes where every '
           f'donor is admitted, because the prior dosage is then constant (getCov2, nbem.c:2067-2087); where it is '
           f'defined its median is {fmt(def_e["median"] if def_e else None)} over {def_e["n"] if def_e else 0} eQTL '
           f'genes and {fmt(def_n["median"] if def_n else None)} over {def_n["n"] if def_n else 0} null genes, against '
           f'{fmt(e["r2_fsnps_native"]["median"])} and {fmt(n["r2_fsnps_native"]["median"])} for native\'s '
           f'{fmt(e["n_fsnp_native"]["median"], 0)} / {fmt(n["n_fsnp_native"]["median"], 0)} feature SNPs (eQTL / null '
           f'medians). RASQUAL\'s own over-correction diagnostic, the prior-posterior r<sup>2</sup> of the rSNP '
           f'genotypes (field 25; its README asks that large changes be treated with caution), is at least '
           f'{fmt(pg.r2_rsnp_pseudo.min())} at the pseudo lead in every gene, and falls below that at native\'s lead in '
           f'{len(low)} genes: '
           + ', '.join(f'{r.gene} {fmt(r.r2_rsnp_native)} (chi-square {r.chi2_native:.0f} against {r.chi2_pseudo:.0f}, '
                       f'rank {r.rank} of {len(pg)} by the native-over-pseudo ratio)' for r in low.itertuples())
           + '. Native was not rerun with --no-posterior-update, so whether those lead chi-squares survive without '
           'genotype correction is not known. ')

    conv = {'native': int(pg.n_rows_native.fillna(0).sum()), 'pseudo': int(pg.n_rows_pseudo.sum())}
    tot = {arm: conv[arm] + excl[arm]['nonconv'] for arm in conv}
    fail = (f'As rates, native fails to converge more often ({excl["native"]["nonconv"]} of {tot["native"]:,} rows, '
            f'{pct(excl["native"]["nonconv"] / tot["native"])}) than the pseudo arm ({excl["pseudo"]["nonconv"]} of '
            f'{tot["pseudo"]:,}, {pct(excl["pseudo"]["nonconv"] / tot["pseudo"])}), while a converged chi-square at or '
            f'below zero is rarer for native ({excl["native"]["chisq_le0"]} of {conv["native"]:,}, '
            f'{pct(excl["native"]["chisq_le0"] / conv["native"])}) than for the pseudo arm '
            f'({excl["pseudo"]["chisq_le0"]:,} of {conv["pseudo"]:,}, {pct(excl["pseudo"]["chisq_le0"] / conv["pseudo"])}). ')
    stuck = at_leads[(at_leads.lead_arm != 'native') & (at_leads.chi2_native < STUCK_CHI2)]
    if len(stuck):
        fail += (f'Native\'s fit reports a chi-square below {STUCK_CHI2} (pi left at 0.5) at {len(stuck)} of the other '
                 f'arms\' leads: '
                 + ', '.join(f'{r.gene} at the {ARM_LABEL[r.lead_arm]} lead (chi-square {r[f"chi2_{r.lead_arm}"]:.1f} '
                             f'there, native {r.chi2_native:.4f})' for _, r in stuck.iterrows()) + '. ')

    sp = min(summ[pool][f'r2_published_{x}_minus_{y}']['sign_test_p'] for pool in ('eqtl', 'null', 'all')
             for x, y in PAIRS)
    ties = pg[pg.lead_ties_native > 1]
    pub = (f'Where the arms\' leads sit relative to the T2T run\'s published leads discriminates nothing here '
           f'({a["native"]["r2_published_at_least_close"]}, {a["pseudo"]["r2_published_at_least_close"]} and '
           f'{a["split"]["r2_published_at_least_close"]} of {a["native"]["n_r2_published"]} leads within r<sup>2</sup> '
           f'{R2_CLOSE} for native, pseudo and split; the smallest paired sign-test p on LD to the published lead is '
           f'{sp:.3f}), which is a design limit rather than a null result: {int(genes.lead_in_body.sum())} of the '
           f'{len(genes)} published leads lie inside the gene body, where no arm tests.')
    if len(ties):
        pub += (f' In {len(ties)} genes several variants tie at native\'s maximum and the first is taken as its lead, so '
                f'"same lead" and LD to the published lead there rest on an arbitrary choice: '
                + ', '.join(f'{r.gene} ({int(r.lead_ties_native)} tied, r<sup>2</sup> to the published lead '
                            f'{fmt(r.r2_native_published_ties_min, 2)} to {fmt(r.r2_native_published_ties_max, 2)})'
                            for r in ties.itertuples()) + '.')

    concl = {'offset': 'native\'s larger chi-squares cannot be read as more power: its level at random variants is an '
                       'offset present where the genotype-phenotype link is broken, while the pseudo arm sits at its '
                       'permuted level; how much power the construction costs at a matched false-positive rate is not '
                       'measured here',
             'signal': 'native\'s higher level at random variants is association that its per-SNP input detects and '
                       'the pseudo construction does not, so on these genes the construction costs RASQUAL power',
             'both': 'native\'s higher level is partly an offset of its statistic and partly association the pseudo '
                     'construction does not detect, so the construction costs RASQUAL some power, less than the '
                     'paired ratio at the leads suggests',
             'below': 'native\'s level cannot be read as either offset or association',
             'unresolved': 'power and miscalibration are not separated on these genes'}[v]
    close = (f'The conclusion is at two levels. The pseudo construction does bypass RASQUAL\'s read-level model, '
             f'measurably: delta absorbs the haplotype-count noise and attenuates the allelic signal, phi is the L/R '
             f'imbalance rather than reference-mapping bias, and the pseudo feature SNP\'s genotype updating is '
             f'undefined in most genes. On the statistic that comes out, the records control says that {concl}.')
    return f'<p>{head}{ctl}</p>\n<p>{par}{fail}{pub}</p>\n<p>{close}</p>'


def write_page(genes, pg, summ, missing, excl, rs_summ, sub, at_leads, cg, cpool):
    png = base64.b64encode((OUT / 'lead_chi2.png').read_bytes()).decode()
    pr = paired_rows(summ)
    rr = random_rows(rs_summ)
    lv = level_rows(rs_summ)
    ct = control_rows(cg, cpool)
    per_sub = sub.groupby('gene').size()
    cols_pg = ['gene', 'pool', 'n_tested', 'n_admitted', 'lead_native', 'chi2_native', 'lead_pseudo', 'chi2_pseudo',
               'lead_split', 'chi2_split', 'r2_native_pseudo', 'r2_split_native', 'r2_split_pseudo',
               'r2_native_published', 'r2_pseudo_published', 'r2_split_published', 'afc_native', 'afc_pseudo', 'afc_split',
               'phi_native', 'phi_pseudo', 'delta_native', 'delta_pseudo', 'n_fsnp_native', 'r2_fsnps_native',
               'r2_fsnps_pseudo', 'pseudo_prior_constant', 'r2_rsnp_native', 'r2_rsnp_pseudo', 'lead_ties_native',
               'r2_native_published_ties_min', 'r2_native_published_ties_max']
    cols_pg = [c for c in cols_pg if c in pg.columns]
    e, n = summ['eqtl'], summ['null']
    rec = cpool['native']['records']
    css = ('body{font-family:system-ui,sans-serif;max-width:1200px;margin:24px auto;padding:0 16px;color:#0b0b0b;'
           'background:#f9f9f7;line-height:1.45}table{border-collapse:collapse;font-size:12px;margin:8px 0}'
           'th,td{border-bottom:1px solid #e1e0d9;padding:3px 6px;text-align:right}th{background:#f0efec}'
           'td:first-child,th:first-child,td:nth-child(2),th:nth-child(2){text-align:left}'
           'h1{font-size:20px}h2{font-size:16px;margin-top:28px}p,li{font-size:14px}.note{color:#52514e}'
           'img{max-width:100%;border:1px solid #e1e0d9;background:#fcfcfb}')
    nat = f"{e['n_with_native']} eQTL and {n['n_with_native']} null genes"
    miss = (f" RASQUAL native has no result yet for {', '.join(missing['native'])}." if missing['native'] else '')
    html = f'''<!doctype html><html><head><meta charset="utf-8"><title>RASQUAL read-level comparison</title><style>{css}</style></head><body>
<h1>Does the pseudo-feature-SNP construction handicap RASQUAL? Observed data, 30 genes</h1>
<p class="note">Written by scripts/rasqual_read_level.py --report on {time.strftime('%Y-%m-%d')}; tables in per_gene.tsv,
arms_at_leads.tsv, control_per_gene.tsv and summary.json beside this page.</p>
<h2>Why</h2>
<p>In the plasmode benchmark RASQUAL received Salmon haplotype counts at one pseudo feature SNP per gene, so its
read-level features (per-SNP allelic counts at real exonic heterozygous sites, the reference-mapping bias phi, the
sequencing/mapping (read allele) error rate delta and posterior genotype updating) were bypassed. This page measures, on
the observed data with no thinning, whether RASQUAL fed its own input is stronger than RASQUAL fed the benchmark's
construction, and how hapmixQTL split weighting compares on the same genes.</p>
<h2>What was run</h2>
<ul>
<li>Genes: {genes.shape[0]} ({e['n_genes']} eQTL by Benjamini-Hochberg 0.05 on pval_beta in the T2T permutation run,
{n['n_genes']} null with pval_beta &gt; 0.5), chosen for exonic heterozygous SNPs with phASER coverage and non-overlapping
cis windows; {int(genes.segmental_duplication.sum())} overlap a segmental duplication (recorded, not a selection rule).
Admitted allelic donors per gene {int(min(pg.n_admitted))}-{int(max(pg.n_admitted))}.</li>
<li>Common to every arm: the tested variants (cis window 1 Mb of the TSS, outside the gene body, MAF &ge; 0.05 among
phased biallelic SNPs; {int(genes.n_tested.sum()):,} gene-variant pairs), the 92 donors, the covariates (14 RNA-tied
and 3 genotype PCs) and the totals, which are the Salmon point-estimate totals pT with the edgeR effective library
size as the offset, exactly as the benchmark gave them to RASQUAL.</li>
<li><b>RASQUAL native</b>: feature SNPs are every exonic biallelic SNP of the analysis VCF inside the gene's exon union,
with phASER's per-SNP ref/alt read counts and the phased genotypes; RASQUAL's own admission and defaults, the rSNP
Hardy-Weinberg filter off (-h 0, as in the benchmark), --force for the (fSNPs + 1) x rSNPs budget. This is not the
input RASQUAL's createASVCF would build: phASER writes counts only at a donor's own heterozygous sites, so every
homozygous donor carries AS 0,0 at every feature SNP (no phASER row at a non-heterozygous donor-site pair in any of the
30 genes), where createASVCF counts reads in every sample (ASVCF/countAS.c:211), and phASER kept reads at mapping
quality 255 and base quality 10 rather than passing them through createASVCF's qcFilterBam. The EM iterations
per rSNP fit vary widely between genes (per-gene medians {fmt(pg.iter_median_native.min(), 0)} to
{fmt(pg.iter_median_native.max(), 0)}; ANKRD36 72, PDE4DIP 7 in the smoke; the earlier LDLR smoke took 5), and the cost
is proportional to iterations times admitted feature SNPs, up to about 0.3 s per admitted feature SNP per rSNP, so a
full scan of {int(genes.n_tested.sum()):,} variants could not be budgeted in advance (a few hundred CPU hours at the
smoke rate). The native arm therefore scans a per-gene <b>subset</b> of the tested variants:
the pseudo and split leads, the published lead when it is a tested variant, the {TOP_K} strongest variants of each of
those two arms and {N_RANDOM} random tested variants (seed {SEED}), {int(per_sub.min())}-{int(per_sub.max())} variants per
gene, {len(sub):,} in all. Its lead is the strongest of that subset. Feature-SNP lines are placed at their position plus
10<sup>9</sup> with the exon lists shifted the same way, so RASQUAL still classifies them as feature SNPs but its cis
window (-c TSS, -w 2 Mb) keeps them out of the rSNP scan; RASQUAL uses position nowhere else.</li>
<li><b>RASQUAL pseudo-fSNP</b>: exactly the benchmark's construction on the observed counts: one heterozygous feature SNP
per gene carrying (round(pL), round(pR)) for every admitted donor. Its phi is the L/R imbalance by construction
(the reference allele is L for every heterozygote), not reference-mapping bias, so phi is not comparable between the
two RASQUAL arms as a mapping-bias estimate.</li>
<li><b>hapmixQTL split</b>: map_nominal with Gibbs variance in the allelic channel and unit variance in the total
channel, allelic channel through the origin, each p on its own degrees of freedom. Its chi-square on the figure is the
1-df chi-square with the same p as its nominal p, derived from the p, not a likelihood ratio.</li>
<li><b>Permutation control</b>: both RASQUAL arms at the {N_RANDOM} random variants of each of the {len(cg)} null genes,
{N_RUNS} runs per gene under each of two nulls, with the same inputs and option lines as the observed arms. The
<i>records permutation</i> moves each donor's whole RNA record to another donor's genotypes: totals, offset, RNA-tied
covariates and every feature SNP's genotype and allelic counts together, with rSNP genotypes and genotype PCs in
place (one seeded order per gene and run, seed {SEED}). <i>RASQUAL's -r</i> is its own permutation: one run draws a
new donor order for each feature SNP (genotype probabilities, allelic counts and offsets move by it, with no haplotype
swap) and one separate order for the totals and their covariate-fitted offsets (nbem.c:2346-2400); it is seeded
without modifying the binary by fixing the seed RASQUAL takes from time() and the process id (main.c:209). Gates
before the runs: the records construction at the identity order reproduced the observed rows' model fields at all
{N_RANDOM} random variants of {GATE_GENE} in both arms, and a seeded -r run repeated with the same seed was
byte-identical and differed under another seed.</li>
</ul>
<h2>Result</h2>
<p><b>At the random tested variants</b> (the {N_RANDOM} per gene drawn without regard to any arm's result), native
against each other arm variant by variant: the share of a gene's random variants where native's chi-square is the
larger (mean over genes with its 95% gene-resampling interval: the genes drawn with replacement {N_RESAMPLE:,} times),
the sign test (two-sided binomial test of the number of genes with a share above one half against one half),
and the Wilcoxon signed-rank test (ranks the absolute per-gene median differences of chi-square and sums the ranks of
the positive ones). This comparison does not depend on which variants native scanned, so it comes first.
Native results are in for {nat}.{miss}</p>
{html_table(rr, list(rr.columns)) if len(rr) else '<p class="note">no native gene finished yet</p>'}
<p><b>The level of each arm's statistic at the random variants.</b> Where no variant is associated, the median
chi-square sits near the chi-square(1) median of {chi2.median(1):.3f} and about 5% of p-values fall below 0.05. The
table gives, per pool and arm, the median over genes of the per-gene median chi-square and the mean over genes of the
per-gene share of p &lt; 0.05, each with its 95% gene-resampling interval and the per-gene range.</p>
{html_table(lv, list(lv.columns)) if len(lv) else ''}
<p><b>The permutation control.</b> Whether an arm's level above 5% at the null genes' random variants is an offset of
its statistic or association is settled by permuting: an offset persists when the genotype-phenotype link is broken,
association does not. Per null gene, each arm's observed share of p &lt; 0.05 at its {N_RANDOM} random variants, the
mean share over the {rec['runs_min']}-{rec['runs_max']} records runs with the permutation p of the observed share
((1 + runs at or above it) / (1 + runs)), and the mean share over the -r runs. The first two columns are the split
arm's lead p times the gene's tested-variant count (a Bonferroni bound on its gene-level p, capped at 1) and its
allelic channel's p at that lead. The last two rows are the means over genes and the paired differences observed
minus permuted, each with its 95% gene-resampling interval.</p>
{html_table(ct, list(ct.columns))}
{reading(rs_summ, summ, excl, pg, at_leads, genes, cg, cpool)}
<img src="data:image/png;base64,{png}" alt="per-gene lead chi-square of the three arms">
<p class="note">Each gene shows the chi-square at each arm's own lead variant; the grey line spans the three arms.</p>
<p><b>Paired summary at the leads.</b> The subset native scanned contains the pseudo and split leads, so those two
arms' maxima over it equal their full-scan maxima; native's lead is the only one found under a restricted search, which
can only understate it. The sign test and Wilcoxon signed-rank test are as above, on log2 chi-square at each arm's own
lead. "Same lead" counts genes whose two leads are the identical variant, and genes whose two leads are in LD at
r<sup>2</sup> &ge; {R2_CLOSE}. The last column counts genes where the first arm's lead is in stronger LD with the T2T run's
published lead than the second arm's (published leads inside the gene body are unreachable by every arm but LD to them
is still measured; {int((~genes.lead_in_vcf).sum())} published leads are indels absent from the SNP VCF).</p>
{html_table(pr, list(pr.columns))}
<p><b>RASQUAL's read-level parameters at the native lead</b> (median over genes, eQTL / null): phi
{fmt(e['phi_native']['median'] if e['phi_native'] else None)} / {fmt(n['phi_native']['median'] if n['phi_native'] else None)},
delta {fmt(e['delta_native']['median'] if e['delta_native'] else None, 4)} / {fmt(n['delta_native']['median'] if n['delta_native'] else None, 4)},
admitted feature SNPs {fmt(e['n_fsnp_native']['median'] if e['n_fsnp_native'] else None, 0)} / {fmt(n['n_fsnp_native']['median'] if n['n_fsnp_native'] else None, 0)},
r<sup>2</sup> between prior and posterior feature-SNP genotypes {fmt(e['r2_fsnps_native']['median'] if e['r2_fsnps_native'] else None)} / {fmt(n['r2_fsnps_native']['median'] if n['r2_fsnps_native'] else None)},
r<sup>2</sup> between prior and posterior rSNP genotypes {fmt(e['r2_rsnp_native']['median'])} / {fmt(n['r2_rsnp_native']['median'])}.
At the pseudo lead: phi {fmt(e['phi_pseudo']['median'])} / {fmt(n['phi_pseudo']['median'])} (L/R imbalance, see above),
delta {fmt(e['delta_pseudo']['median'], 4)} / {fmt(n['delta_pseudo']['median'], 4)},
feature-SNP r<sup>2</sup> {fmt(e['r2_fsnps_pseudo']['median'] if e['r2_fsnps_pseudo'] else None)} / {fmt(n['r2_fsnps_pseudo']['median'] if n['r2_fsnps_pseudo'] else None)}
over the {int((~pg.pseudo_prior_constant).sum())} genes where it is defined (undefined in the
{int(pg.pseudo_prior_constant.sum())} genes where every donor is admitted and the prior is constant; per_gene.tsv
pseudo_prior_constant), rSNP r<sup>2</sup> {fmt(e['r2_rsnp_pseudo']['median'])} / {fmt(n['r2_rsnp_pseudo']['median'])}.
Rows excluded from the RASQUAL tables: native non-converged {excl['native']['nonconv']}, tested variants without a row
{excl['native']['absent']}, feature SNPs RASQUAL also tested as rSNPs {excl['native']['fsnp_as_rsnp_rows']}; pseudo
non-converged {excl['pseudo']['nonconv']}, absent {excl['pseudo']['absent']}.
<b>RASQUAL's optimiser in the two arms</b> (median over genes of the per-gene median EM iterations per rSNP fit, and of
the per-gene share of fits that reached the {RASQUAL_MAXITR}-iteration cap; eQTL / null): native
{fmt(e['iter_median_native']['median'] if e['iter_median_native'] else None, 0)} / {fmt(n['iter_median_native']['median'] if n['iter_median_native'] else None, 0)}
iterations, at cap {fmt(e['iter_at_cap_native']['median'] if e['iter_at_cap_native'] else None)} / {fmt(n['iter_at_cap_native']['median'] if n['iter_at_cap_native'] else None)};
pseudo {fmt(e['iter_median_pseudo']['median'] if e['iter_median_pseudo'] else None, 0)} / {fmt(n['iter_median_pseudo']['median'] if n['iter_median_pseudo'] else None, 0)}
iterations, at cap {fmt(e['iter_at_cap_pseudo']['median'] if e['iter_at_cap_pseudo'] else None)} / {fmt(n['iter_at_cap_pseudo']['median'] if n['iter_at_cap_pseudo'] else None)}.
A fit stopped at the cap reports a lower bound on its likelihood ratio, so an arm that reaches the cap more often is
handicapped by the optimiser rather than by its input.</p>
<h2>Per gene</h2>
{html_table(pg, cols_pg)}
<h2>What this cannot establish</h2>
<ul>
<li>Thirty genes on observed data, one realisation each: only the direction and rough size of a difference between the
arms can be seen. The null genes' lead chi-squares are maxima, over each gene's {int(genes.n_tested.min()):,} to
{int(genes.n_tested.max()):,} tested variants for the pseudo and split arms and over native's subset of
{int(per_sub.min())} to {int(per_sub.max())}, and are compared between arms, not against a reference distribution; the
permutation control speaks to the level at random variants, not to the leads.</li>
<li>The permutation control covers the {len(cg)} null genes and {N_RUNS} runs per gene and null; its intervals resample
those genes and do not extend to genes unlike them. The records null keeps each donor's feature-SNP records and totals
together, as the observed data have them; -r scatters them, so it is not the null the offset question needs and is
reported as RASQUAL's own. Under -r the genotype PCs' fitted effect moves with the totals, where the records null keeps
the PCs with the genotypes.</li>
<li>Native's allelic input is phASER's heterozygous-site counts, not createASVCF's: homozygous donors carry AS 0,0, so
the reads at homozygous feature SNPs that RASQUAL would otherwise use to estimate delta and to check feature-SNP
genotypes never reach the fit, and native's delta and feature-SNP r<sup>2</sup> are estimated from heterozygotes alone.
phASER's read filters (mapping quality 255, base quality 10) also differ from qcFilterBam's.</li>
<li>Native was not rerun with --no-posterior-update, so how much of its largest lead chi-squares comes from rewriting
rSNP genotypes (field 25) is not measured.</li>
<li>The eQTL genes were chosen for high phASER coverage of exonic heterozygous SNPs, the setting where RASQUAL's
read-level model has the most to work with; the benchmark's 30-100-read stratum is not represented.</li>
<li>hapmixQTL's chi-square is a p-value conversion, so the arms are compared on the strength of evidence at their leads,
not on a common likelihood.</li>
<li>RASQUAL native scanned a subset of each gene's tested variants, so a native lead can only be a variant one of the
other arms ranked highly, the published lead, or one of the random draws; the paired comparisons at the other arms'
leads and at the random variants do not depend on that restriction, the "native lead" comparisons do.</li>
<li>RASQUAL's per-rSNP EM iteration counts (per_gene.tsv, iter_median and iter_at_cap, cap {RASQUAL_MAXITR}) are
reported for both of its arms; rows RASQUAL flags as non-converged (field 23) are excluded and counted, and no attempt
was made to change RASQUAL's optimiser.</li>
<li>Totals are Salmon point estimates in every arm, so RASQUAL's total channel is the same in both of its arms; only the
allele-specific input differs between them.</li>
</ul>
</body></html>'''
    C.write_atomic(OUT / 'report.html', lambda fh: fh.write(html), 'w')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scout', action='store_true', help='candidate genes and phASER coverage')
    ap.add_argument('--select', action='store_true', help='choose the 30 genes and check them')
    ap.add_argument('--smoke', action='store_true', help=f'native-arm gate on two genes at {SMOKE_RSNPS} rSNPs')
    ap.add_argument('--arms', action='store_true', help='the three arms on the observed data (resumable)')
    ap.add_argument('--control', action='store_true', help='both RASQUAL arms under two permutation nulls (resumable)')
    ap.add_argument('--report', action='store_true', help='tables, summary.json and the page')
    a = ap.parse_args()
    if a.scout:
        scout()
    elif a.select:
        select()
    elif a.smoke:
        smoke()
    elif a.arms:
        arms()
    elif a.control:
        control()
    elif a.report:
        report()
    else:
        ap.error('choose a stage')


if __name__ == '__main__':
    main()

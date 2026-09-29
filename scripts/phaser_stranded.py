#!/usr/bin/env python3
"""phASER on strand-split RNA BAMs with GTEx-style exonic attribution, resumably, and its gene-level counts.

The library is reverse-stranded paired-end (Salmon ISR, RSeQC antisense ~0.97): a fragment from a plus-strand gene
has read 1 on the minus strand and read 2 on the plus strand. Each donor's BAM is split into the fragments of each
transcript strand, keeping only reads over that strand's gene-unique exonic segments (phaser_features.py), and
phASER (the fast beta in tools/phaser) runs once per strand against a VCF restricted to the same segments, with the
HLA spans as --blacklist and the CHM13 short-read inaccessible regions as --haplo_count_blacklist. Other flags are
the previous run's (run_phaser_cohort.py): --mapq 255 --baseq 10 --paired_end 1 --pass_only 0 --id_separator -.
The BAMs are phaser_wasp.py's WASP-filtered output, and --as_q_cutoff is 0: phASER's default 0.05 drops the reads
below the 5th percentile of alignment score (phaser.py:643-651, 1401), and STAR scores a read carrying the alt
allele 2 below one carrying the ref, so the cutoff removes alt reads preferentially. The native (pre-WASP) run with
the default cutoff is phaser_stranded_20260928, from this script at commit 0e3b268.
Not applied: --gw_phase_vcf, --min_haplo_maf (not decided).

Gene counts: phaser_gene_ae credits a variant to every feature whose span contains it, and pools read names only
within one feature row, so neither gene spans (nested same-strand genes) nor one row per exon (reads spanning two
exons) attribute reads correctly. aggregate() is phaser_gene_ae's per-block logic with the feature membership
replaced: a variant belongs to the one gene owning its exonic segment on that strand. `check` runs the same code
with phaser_gene_ae's span membership on the previous run's output and requires its gene_ae.txt back exactly.
Output: gene_ae/<donor>.gene_ae.txt, rows and columns as phaser_gene_ae's over annot/genes.NC.bed.

  python3 scripts/phaser_stranded.py check 100_D1     # known-answer check against phaser_out/
  python3 scripts/phaser_stranded.py run              # all donors, resumable
"""
import concurrent.futures as cf
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from intervaltree import IntervalTree

D = Path('/mnt/ssd/lalli/brainvar_hapmix_deploy')
BAMS = D / 'wasp_20260928' / 'bams.tsv'                 # DNA library id, WASP BAM (scripts/phaser_wasp.py)
VCF = D / 'vcf' / 'cohort92.phaser_input.NC.vcf.gz'     # scripts/phaser_input_vcf.py
FEAT = D / 'phaser_inputs_20260928'                     # scripts/phaser_features.py
GENES_NC_BED = D / 'annot' / 'genes.NC.bed'
PHASER_DIR = D / 'tools' / 'phaser'                     # fast beta, 2026-02-03
PREVIOUS = D / 'phaser_out'                             # run_phaser_cohort.py output, for `check`
OUT = D / 'phaser_stranded_wasp_20260928'
STRANDS = {'plus': '(flag.read1 && flag.reverse) || (flag.read2 && !flag.reverse)',
           'minus': '(flag.read1 && !flag.reverse) || (flag.read2 && flag.reverse)'}
GENE_STRAND = {'plus': '+', 'minus': '-'}
SEP = '-'                                               # phASER --id_separator; RefSeq contigs contain '_'
GW_CUTOFF = 0.9                                         # phaser_gene_ae default
JOBS, THREADS = 12, 4


def run(cmd, log):
    log.write('+ ' + ' '.join(map(str, cmd)) + '\n')
    log.flush()
    subprocess.run(list(map(str, cmd)), stdout=log, stderr=subprocess.STDOUT, check=True)


def strand_vcfs():
    """cohort92.phaser_input restricted to each strand's gene-unique exonic segments."""
    out = {}
    for s in STRANDS:
        v = OUT / f'vcf.{s}.vcf.gz'
        if not v.exists():
            tmp = v.with_name(v.name.replace('.vcf.gz', '.tmp.vcf.gz'))
            subprocess.run(['bcftools', 'view', '--threads', '8', '-R', FEAT / f'exonic_unique.{s}.NC.bed', VCF,
                            '-Oz', '-o', tmp], check=True)
            subprocess.run(['tabix', '-p', 'vcf', tmp], check=True)
            os.replace(f'{tmp}.tbi', f'{v}.tbi')
            os.replace(tmp, v)
        n = int(subprocess.run(['bcftools', 'index', '-n', v], check=True, stdout=subprocess.PIPE, text=True).stdout)
        print(f'{v.name}: {n:,} records')
        out[s] = v
    return out


def one(donor, bam, strand, vcf):
    """Split one BAM to one strand, then phASER; skipped when its haplotypic counts exist."""
    prefix = OUT / 'phaser' / f'{donor}.{strand}'
    done = Path(f'{prefix}.haplotypic_counts.txt')
    if done.exists():
        return donor, strand, 'skipped', 0.0
    t0 = time.time()
    sbam = OUT / 'bams' / f'{donor}.{strand}.bam'
    with open(f'{prefix}.log', 'w') as log:
        if not sbam.exists():
            tmp = sbam.with_name(sbam.name + '.tmp')
            run(['samtools', 'view', '-@', THREADS, '-b', '-M', '-L', FEAT / f'exonic_unique.{strand}.NC.bed',
                 '-e', STRANDS[strand], '-o', tmp, bam], log)
            run(['samtools', 'index', tmp], log)
            os.replace(f'{tmp}.bai', f'{sbam}.bai')
            os.replace(tmp, sbam)
        tmp_prefix = OUT / 'phaser' / f'{donor}.{strand}.partial'
        run([sys.executable, PHASER_DIR / 'phaser' / 'phaser.py', '--vcf', vcf, '--bam', sbam, '--sample', donor,
             '--mapq', 255, '--baseq', 10, '--paired_end', 1, '--write_vcf', 0, '--python_string', sys.executable,
             '--id_separator', SEP, '--pass_only', 0, '--as_q_cutoff', 0, '--threads', THREADS,
             '--temp_dir', OUT / 'tmp',
             '--blacklist', FEAT / 'hla.NC.bed', '--haplo_count_blacklist', FEAT / 'haplo_count_blacklist.NC.bed',
             '--o', tmp_prefix], log)
    for f in OUT.joinpath('phaser').glob(f'{donor}.{strand}.partial.*'):
        os.replace(f, str(f).replace('.partial.', '.'))
    return donor, strand, 'ok', time.time() - t0


def block_reads(row, idx):
    """phaser_gene_ae.variant_feature_reads for the variants at positions idx of one block."""
    if not idx:
        return set(), set()
    if row['n'] == 1:
        return {str(i) for i in range(int(row['aCount']))}, {str(i) for i in range(int(row['bCount']))}
    a, b = set(), set()
    for i in idx:
        a.update(row['aR'][i].split(','))
        b.update(row['bR'][i].split(','))
    a.discard('')
    b.discard('')
    return a, b


def aggregate(hc, members, n_features):
    """phaser_gene_ae's counting for one BAM; members(row) -> {feature index: [variant positions in the block]}."""
    feat = [dict(a=0, b=0, v=[], ua=0, ub=0, uv=[]) for _ in range(n_features)]
    for _, row in hc.iterrows():
        if not row['totalCount'] > 0:
            continue
        xv = row['variants'].split(',')
        row = dict(row, n=len(xv), aR=str(row['aReads']).split(';'), bR=str(row['bReads']).split(';'))
        for k, idx in members(row, xv).items():
            ra, rb = block_reads(row, idx)
            used = [xv[i] for i in idx]
            f = feat[k]
            if row['blockGWPhase'] != '0/1' and row['gwStat'] >= GW_CUTOFF:
                if row['blockGWPhase'] == '0|1':
                    f['a'] += len(ra)
                    f['b'] += len(rb)
                elif row['blockGWPhase'] == '1|0':
                    f['a'] += len(rb)
                    f['b'] += len(ra)
                f['v'] += used
            elif len(ra) + len(rb) > f['ua'] + f['ub']:
                f['ua'], f['ub'], f['uv'] = len(ra), len(rb), used
    return feat


def log2_afc(a, b):
    r = float('inf') if b == 0 else float(a) / float(b)
    return float('-inf') if r == 0 else math.log(r, 2)


def gene_ae_rows(bed, feat, bam_name):
    """phaser_gene_ae's output rows (min_cov 0), one per feature in order."""
    rows = []
    for (contig, start, stop, name), f in zip(bed.itertuples(index=False), feat):
        if f['a'] + f['b'] >= f['ua'] + f['ub']:
            a, b, v, gw = f['a'], f['b'], f['v'], 1
        else:
            a, b, v, gw = f['ua'], f['ub'], f['uv'], 0
        rows.append([contig, start, stop, name, a, b, a + b, log2_afc(a, b), len(v), ','.join(v), gw, bam_name])
    return rows


def write_gene_ae(path, rows):
    tmp = path.with_name(path.name + '.tmp')
    with open(tmp, 'w') as fh:
        fh.write('\t'.join(['contig', 'start', 'stop', 'name', 'aCount', 'bCount', 'totalCount', 'log2_aFC',
                            'n_variants', 'variants', 'gw_phased', 'bam']) + '\n')
        for r in rows:
            fh.write('\t'.join(map(str, r)) + '\n')
    os.replace(tmp, path)


def read_hc(path):
    return pd.read_csv(path, sep='\t', index_col=False, dtype={'contig': str, 'aReads': str, 'bReads': str,
                                                                 'blockGWPhase': str})


def read_bed():
    return pd.read_csv(GENES_NC_BED, sep='\t', header=None, names=['contig', 'start', 'stop', 'name'],
                       dtype={'contig': str})


def span_members(bed):
    """phaser_gene_ae's membership: features overlapping the block span, variants with begin <= pos-1 <= end."""
    trees = {}
    for i, (c, s, e) in enumerate(zip(bed.contig, bed.start, bed.stop)):
        trees.setdefault(c, IntervalTree())[s:e] = i

    def members(row, xv):
        out = {}
        if row['contig'] not in trees:
            return out
        pos = [int(x.split(SEP)[1]) - 1 for x in xv]
        for iv in trees[row['contig']][row['start'] - 1:row['stop']]:
            idx = [i for i, p in enumerate(pos) if iv.begin <= p <= iv.end]
            out[iv.data] = idx
        return out
    return members


def exon_members(bed, strand):
    """Production membership: each variant to the gene owning its exonic segment on this strand."""
    seg = pd.read_csv(FEAT / f'exonic_unique.{strand}.NC.bed', sep='\t', header=None,
                      names=['contig', 'start', 'end', 'gene'], dtype={'contig': str})
    index = pd.Series(range(len(bed)), index=bed.name)
    if not index.index.is_unique:
        raise SystemExit(f'{GENES_NC_BED.name}: gene names not unique')
    lookup = {}
    for c, g in seg.groupby('contig'):
        g = g.sort_values('start')
        lookup[c] = (g.start.values, g.end.values, index.reindex(g.gene).values)

    def members(row, xv):
        st, en, fi = lookup[row['contig']]
        out = {}
        for i, x in enumerate(xv):
            p = int(x.split(SEP)[1]) - 1
            j = st.searchsorted(p, side='right') - 1
            if j < 0 or p >= en[j]:
                raise SystemExit(f'variant {x} is outside every {strand} exonic segment: the VCF was not restricted')
            out.setdefault(int(fi[j]), []).append(i)
        return out
    return members


def check(donor):
    """Span membership on the previous run's haplotypic counts must return its gene_ae.txt exactly."""
    bed = read_bed()
    hc = read_hc(PREVIOUS / f'{donor}.haplotypic_counts.txt')
    ref = pd.read_csv(PREVIOUS / f'{donor}.gene_ae.txt', sep='\t', dtype=str, keep_default_na=False)
    (bam_name,) = set(hc.bam)
    rows = gene_ae_rows(bed, aggregate(hc, span_members(bed), len(bed)), bam_name)
    got = pd.DataFrame([list(map(str, r)) for r in rows], columns=ref.columns)
    bad = (got != ref).any(axis=1)
    print(f'check {donor}: {len(hc):,} blocks, {len(bed):,} features; rows differing from phaser_gene_ae: '
          f'{int(bad.sum())}')
    if bad.any():
        print(pd.concat([ref[bad].head(), got[bad].head()], keys=['phaser_gene_ae', 'this']).to_string())
        raise SystemExit(1)


def gene_counts(donor, bed, strand_of):
    """Each gene's row from its own strand's phASER run."""
    feat = [None] * len(bed)
    names = []
    for s in STRANDS:
        hc = read_hc(OUT / 'phaser' / f'{donor}.{s}.haplotypic_counts.txt')
        names += list(set(hc.bam))
        f = aggregate(hc, exon_members(bed, s), len(bed))
        for i in np.flatnonzero(strand_of == GENE_STRAND[s]):
            feat[i] = f[i]
    empty = dict(a=0, b=0, v=[], ua=0, ub=0, uv=[])
    rows = gene_ae_rows(bed, [f if f is not None else empty for f in feat], '+'.join(sorted(names)))
    write_gene_ae(OUT / 'gene_ae' / f'{donor}.gene_ae.txt', rows)
    return donor


def main():
    if sys.argv[1:2] == ['check']:
        check(sys.argv[2])
        return
    if sys.argv[1:] != ['run']:
        raise SystemExit(__doc__)
    for sub in ('bams', 'phaser', 'gene_ae', 'tmp'):
        OUT.joinpath(sub).mkdir(parents=True, exist_ok=True)
    vcfs = strand_vcfs()
    bams = pd.read_csv(BAMS, sep='\t', header=None, names=['donor', 'bam'])
    samples = subprocess.run(['bcftools', 'query', '-l', VCF], check=True, stdout=subprocess.PIPE, text=True).stdout.split()
    if sorted(samples) != sorted(bams.donor):
        raise SystemExit(f'{BAMS.name} donors differ from the samples of {VCF.name}')
    jobs = [(d, b, s, vcfs[s]) for d, b in zip(bams.donor, bams.bam) for s in STRANDS]
    print(f'{len(jobs)} donor-strand runs at {JOBS} x {THREADS} threads', flush=True)
    with cf.ThreadPoolExecutor(JOBS) as ex:
        for k, (d, s, status, sec) in enumerate(ex.map(lambda j: one(*j), jobs), 1):
            print(f'[{k}/{len(jobs)}] {d} {s} {status} {sec / 60:.1f} min', flush=True)
    bed = read_bed()
    nest = pd.read_csv(FEAT / 'nesting.tsv', sep='\t').set_index('gene')
    strand_of = nest.strand.reindex(bed.name).values
    with cf.ProcessPoolExecutor(JOBS) as ex:
        for d in ex.map(gene_counts, bams.donor, [bed] * len(bams), [strand_of] * len(bams)):
            print(f'gene_ae {d}', flush=True)


if __name__ == '__main__':
    main()

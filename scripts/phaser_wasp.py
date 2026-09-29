#!/usr/bin/env python3
"""WASP mappability filtering (van de Geijn et al. 2015) of the T2T RNA BAMs, as input to phaser_stranded.py.

Per donor: the reads phASER can count (proper pair, primary, mapped, not duplicate-marked, STAR-unique MAPQ 255;
phASER runs `samtools view -F 0x400 -f 2 -q 255`), restricted to the pairs with a mate over one of the donor's
counted sites (its heterozygous SNVs in a gene-unique exonic segment of either strand, phaser_features.py: the only
variants phaser_stranded.py's VCFs hold, so the only reads it can count or phase with), first lose every pair WASP
cannot test symmetrically
(prefilter()), then go through WASP's find_intersecting_snps.py with the donor's own heterozygous SNVs. Every read
pair over one gets its other-allele versions (all 2^n combinations, --snp_dir mode), which STAR 2.7.10a remaps with
the original run's parameters; filter_remapped_reads.py keeps an original pair only if every version maps back
uniquely to the same positions with the same CIGARs. A version that is unmapped or multimapping fails: STAR writes
it unmapped (--outSAMunmapped Within) or with secondary records, and WASP marks either bad. Output BAM = pairs over
no het SNV + pairs that passed, coordinate-sorted and indexed; it holds no duplicate-marked reads, which phASER
would drop anyway (--remove_dups 1, phaser.py:604-610). `run` ends by writing bams.tsv, phaser_stranded.py's input.

prefilter() drops a pair when either mate (a) has an aligned block (M/D/=/X) or a soft clip's reference projection
over a heterozygous non-SNV record of the donor (indel, '*', 1|2, MNP; cohort VCF): WASP cannot swap these, and
standard WASP drops the pairs over indels; or (b) has a heterozygous SNV inside a soft clip's reference projection:
(prefilter() and WASP decide each pair from its own two reads and the donor's variants, so restricting the input to
the pairs over counted sites changes no decision on those pairs; it only skips pairs phASER would never count.)
WASP neither searches nor swaps clipped bases (snptable.py:425-437) and STAR clips an alt allele in a read's last two
bases, so WASP would keep those alt pairs and drop the matching ref pairs (their alt version clips and fails the CIGAR
check). A clipped base that belongs across a splice junction is not caught (the projection is contiguous).

Remap = the original command without --quantMode/--quantTranscriptomeBan (transcriptome output), the RG line and
--twopassMode. The second pass's inserted junctions are not on disk; the donor's final SJ.out.tab rows with column
6 == 1 ("annotated" in pass 2, i.e. in the GTF or inserted after pass 1) stand in for them via
--sjdbFileChrStartEnd. Rows with column 6 == 0 were found de novo in pass 2, so they are left to be found de novo.

WASP facts this relies on (WASP-master/mapping): SNP files are <contig>.snps.txt.gz with 'pos ref alt', 1-based
(find_intersecting_snps.py:687, snptable.py:259-297), and a missing file silently means no SNPs
(snptable.py:247-252), so every BAM contig gets one; keep.bam and to.remap.bam are uncompressed SAM despite the
name (pysam mode "w", :139-142) and keep.bam is not coordinate-sorted (:894-895); pairs over more than 6 SNVs in a
mate, with more than 64 versions, or with mates disagreeing at a shared SNV go to neither output (:866-868,
:902-918), nor, uncounted, do pairs whose cached mate positions disagree (:730-733).

  python3 scripts/phaser_wasp.py run        # all donors, resumable
"""
import concurrent.futures as cf
import gzip
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pysam

D = Path('/mnt/ssd/lalli/brainvar_hapmix_deploy')
BAMS = D / 'cohort' / 'bams.tsv'                        # donor id, BAM
VCF = D / 'vcf' / 'cohort92.phaser_input.NC.vcf.gz'     # scripts/phaser_input_vcf.py: biallelic SNVs, phased
COHORT_VCF = D / 'vcf' / 'cohort92.NC.vcf.gz'          # its source, with indels; merged with norm -m +any
SPANS = 'het_nonsnv_spans.tsv.gz'                       # per donor: contig, 0-based start, end; see nonsnv_spans()
FEAT = D / 'phaser_inputs_20260928'                    # exonic_unique.{plus,minus}.NC.bed (scripts/phaser_features.py)
SJ_DIR = Path('/mnt/data/lalli/nf_stage/reference_comparison_results_RNA/T2T_NCBI110/star_salmon/log')
OUT = D / 'wasp_20260928'
INDEX = OUT / 'star_index'                              # scripts/wasp_star_index.sh
STAR = '/mnt/ssd/lalli/usr/local/bin/STAR-2.7.10a'
WASP = Path('/mnt/ssd/lalli/usr/local/src/WASP-master/mapping')
WASP_PY = '/mnt/ssd/lalli/usr/local/venvs/wasp/bin/python'
READS = ['-f', '2', '-F', '0xD04', '-q', '255']        # proper pair; not unmapped/secondary/duplicate/supplementary
STAR_ARGS = ['--runRNGseed', '42', '--readFilesCommand', 'zcat', '--outMultimapperOrder', 'Random',
             '--outSAMtype', 'BAM', 'Unsorted', '--outSAMattributes', 'NH', 'HI', 'AS', 'NM', 'MD',
             '--outSAMunmapped', 'Within', '--outFilterMultimapNmax', '40', '--alignIntronMax', '1000000',
             '--alignMatesGapMax', '1000000', '--alignSJoverhangMin', '8', '--alignSJDBoverhangMin', '1',
             '--sjdbOverhang', '149']
JOBS, THREADS = 16, 4                                   # 64 CPUs (user, 2026-09-28); each STAR holds ~35 GB with on-the-fly junctions, a few at a time


def run(cmd, log, **kw):
    log.write('+ ' + ' '.join(map(str, cmd)) + '\n')
    log.flush()
    return subprocess.run(list(map(str, cmd)), stderr=log, check=True, **kw)


def count(path):
    return int(subprocess.run(['samtools', 'view', '-c', path], check=True, stdout=subprocess.PIPE, text=True).stdout)


def het_records(vcf, donor, fmt):
    return subprocess.run(['bcftools', 'query', '-s', donor, '-i', 'GT="het"', '-f', fmt, vcf], check=True,
                          stdout=subprocess.PIPE, text=True).stdout.splitlines()


def nonsnv_spans(donor):
    """Reference spans of the donor's het records whose two alleles, common prefix and suffix trimmed, are not a
    REF/ALT single-base substitution (those are phaser_input's SNVs)."""
    rows = []
    for line in het_records(COHORT_VCF, donor, '%CHROM\t%POS\t%REF\t%ALT\t[%GT]\n'):
        c, pos, ref, alt, gt = line.split('\t')
        g = re.split('[|/]', gt)
        if '.' in g:
            raise SystemExit(f'{donor} {c}:{pos}: half-missing heterozygous GT {gt}')
        al = [ref] + alt.split(',')
        a, b = al[int(g[0])], al[int(g[1])]
        while len(a) > 1 and len(b) > 1 and a[-1] == b[-1]:
            a, b = a[:-1], b[:-1]
        while len(a) > 1 and len(b) > 1 and a[0] == b[0]:
            a, b = a[1:], b[1:]
        if '0' in g and len(a) == len(b) == 1 and a in 'ACGT' and b in 'ACGT':
            continue
        rows.append((c, int(pos) - 1, int(pos) - 1 + len(ref)))
    return pd.DataFrame(rows, columns=['contig', 'start', 'end'])


def snp_dir(donor, contigs):
    """The donor's heterozygous SNVs, one WASP file per BAM contig (empty where it has none), and SPANS."""
    out = OUT / 'snps' / donor
    if not out.exists():
        q = het_records(VCF, donor, '%CHROM\t%POS\t%REF\t%ALT\n')
        het = pd.DataFrame([x.split('\t') for x in q], columns=['contig', 'pos', 'ref', 'alt'])
        if not (het.ref.isin(list('ACGT')) & het.alt.isin(list('ACGT'))).all():
            raise SystemExit(f'{donor}: a heterozygous allele is not one of A/C/G/T; WASP would discard its reads '
                             f'as indel reads')
        spans = nonsnv_spans(donor)
        if het.duplicated(['contig', 'pos']).any() or not set(het.contig) | set(spans.contig) <= set(contigs):
            raise SystemExit(f'{donor}: duplicate positions or contigs outside the BAM header')
        tmp = out.with_name(donor + '.tmp')
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        for c in contigs:
            g = het[het.contig == c]
            with gzip.open(tmp / f'{c}.snps.txt.gz', 'wt', compresslevel=1) as fh:
                fh.writelines(f'{p} {r} {a}\n' for p, r, a in zip(g.pos, g.ref, g.alt))
        spans.to_csv(tmp / SPANS, sep='\t', header=False, index=False)
        os.replace(tmp, out)
    if sorted(p.name for p in out.iterdir()) != sorted([f'{c}.snps.txt.gz' for c in contigs] + [SPANS]):
        raise SystemExit(f'{out}: files differ from the BAM contigs plus {SPANS}')
    return out


def select(donor, bam, work, out):
    """Both mates of every eligible pair (READS) with a read over one of the donor's counted sites."""
    sites = set()
    for s in ('plus', 'minus'):                      # a site in exons of both strands is listed once
        sites.update(tuple(x.split('\t')) for x in het_records_in(FEAT / f'exonic_unique.{s}.NC.bed', donor))
    bed, names, sel = work / 'counted_sites.bed', work / 'selected.names', work / 'selected.bam'
    pd.DataFrame(sorted((c, int(p) - 1, int(p)) for c, p in sites)).to_csv(bed, sep='\t', header=False, index=False)
    with open(work / 'select.log', 'w') as log:
        run(['samtools', 'view', '-@', THREADS - 1, '-b', '-M', '-L', bed, *READS, '-o', sel, bam], log)
        names.write_text(''.join(n + '\n' for n in {rd.query_name for rd in pysam.AlignmentFile(sel)}))
        run(['samtools', 'view', '-@', THREADS - 1, '-b', *READS, '-N', names, '-o', out, bam], log)
    sel.unlink()
    return len(sites)


def het_records_in(regions, donor):
    return subprocess.run(['bcftools', 'query', '-R', regions, '-s', donor, '-i', 'GT="het"', '-f', '%CHROM\t%POS\n', VCF],
                          check=True, stdout=subprocess.PIPE, text=True).stdout.splitlines()


def prefilter(work, inp, snps, out):
    """Drop the pairs WASP cannot test symmetrically (module docstring); returns counts."""
    spans = pd.read_csv(snps / SPANS, sep='\t', header=None, names=['contig', 'start', 'end'])
    bam = pysam.AlignmentFile(inp)
    flag, tid, c = {}, None, dict(snvs_in_spans=0)
    for rd in bam:
        if rd.reference_id != tid:                 # coordinate-sorted: one contig's arrays at a time
            tid, name = rd.reference_id, rd.reference_name
            span = np.zeros(bam.get_reference_length(name), bool)
            for s, e in zip(*spans.loc[spans.contig == name, ['start', 'end']].values.T):
                span[s:e] = True
            snv = np.zeros(len(span), bool)
            snv[[int(x.split(' ', 1)[0]) - 1 for x in gzip.open(snps / f'{name}.snps.txt.gz', 'rt')]] = True
            c['snvs_in_spans'] += int(span[snv].sum())
        ct, r, why = rd.cigartuples, rd.reference_start, 0
        for op, n in ct:
            if op in (0, 2, 7, 8):
                if span[r:r + n].any():
                    why |= 1
                r += n
            elif op == 3:
                r += n
        clips = ([(max(0, rd.reference_start - ct[0][1]), rd.reference_start)] if ct[0][0] == 4 else []) + \
                ([(r, r + ct[-1][1])] if ct[-1][0] == 4 else [])
        for s, e in clips:
            why |= 2 * bool(span[s:e].any()) | 4 * bool(snv[s:e].any())
        if why:
            flag[rd.query_name] = flag.get(rd.query_name, 0) | why
    for bit, key in ((1, 'pairs_aligned_over_nonsnv'), (2, 'pairs_nonsnv_in_clip'), (4, 'pairs_snv_in_clip')):
        c[key] = sum(1 for v in flag.values() if v & bit)
    c['pairs_dropped'] = len(flag)
    (work / 'prefilter.names').write_text(''.join(n + '\n' for n in flag))
    with open(work / 'prefilter.log', 'w') as log:
        run(['samtools', 'view', '-@', THREADS - 1, '-b', '-N', f'^{work / "prefilter.names"}', '-o', out, inp], log)
    return c


def parse_find(log_path):
    """find_intersecting_snps.py's ReadStats block (:217-254) and its uncounted mate-position warnings."""
    text = Path(log_path).read_text()
    block = text[text.rindex('DISCARD reads:'):]
    head, keep, remap = re.split(r'KEEP reads:|REMAP reads:', block)
    c = {k.strip(): int(v) for k, v in re.findall(r'\n\s+([^:\n]+): (\d+)', head)}
    for name, part in (('keep', keep), ('remap', remap)):
        c.update({f'{name} {k}': int(v) for k, v in re.findall(r'\n\s+(single-end|pairs): (\d+)', part)})
    c.update({k: int(v) for k, v in re.findall(r'read SNP (ref matches|alt matches|mismatches): (\d+)', block)})
    c['mate position mismatch'] = text.count('WARNING: read pair positions do not match')
    return c


def find(work, bam, snps, n_in):
    """WASP step 3 with a read-level reconciliation: input = keep + to.remap + WASP's discards."""
    with open(work / 'find.log', 'w') as log:
        run([WASP_PY, WASP / 'find_intersecting_snps.py', '--is_paired_end', '--is_sorted', '--output_dir', work,
             '--snp_dir', snps, bam], log)
    c = parse_find(work / 'find.log')
    must_be_zero = ['unmapped', 'mate unmapped', 'improper pair', 'different chromosome', 'indel',
                    'secondary alignment', 'supplementary alignment', 'keep single-end', 'remap single-end']
    if any(c[k] for k in must_be_zero):
        raise SystemExit(f'{work}: WASP saw reads the input filter should have removed: '
                         f'{ {k: c[k] for k in must_be_zero} }')
    prefix = work / bam.name[:-len('.bam')]
    c['keep_reads'] = count(f'{prefix}.keep.bam')
    c['to_remap_reads'] = count(f'{prefix}.to.remap.bam')
    if c['keep_reads'] != 2 * c['keep pairs'] or c['to_remap_reads'] != 2 * c['remap pairs']:
        raise SystemExit(f'{work}: keep/to.remap records differ from WASP\'s pair counts')
    c['wasp_discard_reads'] = (2 * c['excess overlapping snps'] + c['excess allelic combinations']
                               + 2 * c['read pairs with discordant shared SNPs']
                               + c['missing pairs (e.g. mismatched read names)'] + 2 * c['mate position mismatch'])
    residual = n_in - c['keep_reads'] - c['to_remap_reads'] - c['wasp_discard_reads']
    if residual:
        raise SystemExit(f'{work}: {residual} input reads unaccounted for by WASP')
    if os.path.getsize(f'{prefix}.remap.single.fq.gz') > 100:
        raise SystemExit(f'{work}: WASP wrote single-end remap reads')
    return prefix, c


def remap(work, prefix, sj_out):
    """STAR on the other-allele versions, with the donor's pass-2 database junctions inserted on the fly."""
    sj = pd.read_csv(sj_out, sep='\t', header=None)
    sj[sj[5] == 1].to_csv(work / 'sj.pass2_database.tab', sep='\t', header=False, index=False)
    with open(work / 'star.log', 'w') as log:
        run([STAR, '--runThreadN', THREADS, '--genomeDir', INDEX, '--readFilesIn', f'{prefix}.remap.fq1.gz',
             f'{prefix}.remap.fq2.gz', *STAR_ARGS, '--sjdbFileChrStartEnd', work / 'sj.pass2_database.tab',
             '--outFileNamePrefix', f'{work}/remap.'], log, stdout=log)
    new = re.findall(r'number of new junctions=(\d+), old junctions=(\d+)', (work / 'remap.Log.out').read_text())
    return dict(sj_rows=len(sj), sj_database_rows=int((sj[5] == 1).sum()),
                star_new_junctions=int(new[-1][0]), star_old_junctions=int(new[-1][1]))


def filter_(work, prefix, to_remap_reads):
    """WASP step 5; every version has a record (Within + all secondaries), so none may be 'not present'."""
    with open(work / 'filter.log', 'w') as log:
        run([WASP_PY, WASP / 'filter_remapped_reads.py', f'{prefix}.to.remap.bam', work / 'remap.Aligned.out.bam',
             work / 'remap.keep.bam'], log)
    text = (work / 'filter.log').read_text()
    c = {k.strip(): int(v) for k, v in re.findall(r'\n\s*([A-Za-z][A-Za-z ]+?):? (\d+)', '\n' + text)}
    c['passed_reads'] = count(work / 'remap.keep.bam')
    # filter_remapped_reads.py:204-287: keep and pair-missing count reads, CIGAR discards count pairs
    total = (c['keep reads'] + c['bad reads'] + c['not present reads'] + 2 * c['CIGAR mismatch']
             + 2 * c['CIGAR missing or multiple'] + c['mate pair missing'])
    if c['passed_reads'] != c['keep reads'] or total != to_remap_reads or c['not present reads']:
        raise SystemExit(f'{work}: filter accounting failed: {c}, to.remap {to_remap_reads}')
    return c


def assemble(work, prefix, final, expected):
    """keep.bam (SAM text, unsorted) + the passed pairs, coordinate-sorted and indexed, then moved into place."""
    tmp = final.with_name(final.name.replace('.bam', '.tmp.bam'))
    with open(work / 'sort.log', 'w') as log:
        sort = subprocess.Popen(['samtools', 'sort', '-@', str(THREADS - 1), '-m', '1G', '-T', str(work / 'sort'),
                                 '-o', str(tmp), '-'], stdin=subprocess.PIPE, stderr=log)
        with open(f'{prefix}.keep.bam', 'rb') as fh:              # carries the input header
            shutil.copyfileobj(fh, sort.stdin, 1 << 22)
        sort.stdin.flush()
        run(['samtools', 'view', work / 'remap.keep.bam'], log, stdout=sort.stdin)
        sort.stdin.close()
        if sort.wait():
            raise SystemExit(f'{final.name}: samtools sort failed, see {work / "sort.log"}')
        run(['samtools', 'index', '-@', THREADS - 1, tmp], log)
    n = count(tmp)
    if n != expected:
        raise SystemExit(f'{tmp}: {n} reads, expected keep + passed = {expected}')
    os.replace(f'{tmp}.bai', f'{final}.bai')
    os.replace(tmp, final)
    return n


def one(donor, bam):
    final = OUT / 'bams' / f'{donor}.wasp.bam'
    path = OUT / 'facts' / f'{donor}.json'
    if final.exists() and path.exists():
        return donor, 'skipped', 0.0
    t0 = time.time()
    work = OUT / 'work' / donor
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    header = subprocess.run(['samtools', 'view', '-H', bam], check=True, stdout=subprocess.PIPE, text=True).stdout
    contigs = [x.split('\t')[1][3:] for x in header.splitlines() if x.startswith('@SQ')]
    snps = snp_dir(donor, contigs)
    sj_out = SJ_DIR / (Path(bam).name.replace('.markdup.sorted.bam', '') + 'SJ.out.tab')
    if not sj_out.exists():
        raise SystemExit(f'{sj_out} missing')
    inp, tested = work / f'{donor}.input.bam', work / f'{donor}.bam'
    n_sites = select(donor, bam, work, inp)
    facts = dict(donor=donor, bam=str(bam), counted_sites=n_sites, input_reads=count(inp),
                 het_snvs=sum(1 for c in contigs for _ in gzip.open(snps / f'{c}.snps.txt.gz', 'rt')),
                 het_nonsnv_spans=sum(1 for _ in gzip.open(snps / SPANS, 'rt')))
    facts['prefilter'] = pf = prefilter(work, inp, snps, tested)
    n_tested = count(tested)
    pf['dropped_reads'] = facts['input_reads'] - n_tested
    if pf['dropped_reads'] != 2 * pf['pairs_dropped']:
        raise SystemExit(f'{work}: prefilter dropped {pf["dropped_reads"]} reads for {pf["pairs_dropped"]} pairs; '
                         f'the input holds mates without their pair')
    prefix, fc = find(work, tested, snps, n_tested)
    facts.update(find=fc)
    facts.update(remap(work, prefix, sj_out))
    rc = filter_(work, prefix, fc['to_remap_reads'])
    facts.update(filter=rc)
    facts['output_reads'] = assemble(work, prefix, final, fc['keep_reads'] + rc['passed_reads'])
    for p in work.iterdir():                                       # keep the logs, drop the bulk
        if not (p.name.endswith('.log') or p.name in ('remap.Log.out', 'remap.Log.final.out')):
            shutil.rmtree(p) if p.is_dir() else p.unlink()
    facts['minutes'] = round((time.time() - t0) / 60, 1)
    path.with_suffix('.tmp').write_text(json.dumps(facts, indent=1) + '\n')
    os.replace(path.with_suffix('.tmp'), path)
    print(f"{donor}: input {facts['input_reads']:,}, prefilter dropped {pf['dropped_reads']:,}, "
          f"keep {fc['keep_reads']:,}, to remap {fc['to_remap_reads']:,}, WASP discarded {fc['wasp_discard_reads']:,}, "
          f"passed {rc['passed_reads']:,}, output {facts['output_reads']:,} "
          f"({facts['output_reads'] / facts['input_reads']:.4f} of input)", flush=True)
    return donor, 'ok', facts['minutes']


def summary(donors):
    rows = []
    for d in donors:
        f = json.loads((OUT / 'facts' / f'{d}.json').read_text())
        rows.append(dict(donor=d, het_snvs=f['het_snvs'], input_reads=f['input_reads'],
                         prefilter_dropped_reads=f['prefilter']['dropped_reads'],
                         keep_reads=f['find']['keep_reads'], to_remap_reads=f['find']['to_remap_reads'],
                         wasp_discard_reads=f['find']['wasp_discard_reads'], passed_reads=f['filter']['passed_reads'],
                         output_reads=f['output_reads']))
    t = pd.DataFrame(rows)
    t['prefilter_share'] = t.prefilter_dropped_reads / t.input_reads
    t['to_remap_share'] = t.to_remap_reads / t.input_reads
    t['passed_share_of_to_remap'] = t.passed_reads / t.to_remap_reads
    t['output_share'] = t.output_reads / t.input_reads
    tmp = OUT / 'facts.tmp.tsv'
    t.to_csv(tmp, sep='\t', index=False)
    os.replace(tmp, OUT / 'facts.tsv')
    print(t[['prefilter_share', 'to_remap_share', 'passed_share_of_to_remap', 'output_share']].describe().to_string())
    tmp = OUT / 'bams.tmp.tsv'                                     # phaser_stranded.py's BAMS
    pd.DataFrame(dict(donor=list(donors), bam=[OUT / 'bams' / f'{d}.wasp.bam' for d in donors])).to_csv(
        tmp, sep='\t', header=False, index=False)
    os.replace(tmp, OUT / 'bams.tsv')


def main():
    if sys.argv[1:] != ['run']:
        raise SystemExit(__doc__)
    if not (INDEX / 'SA').exists() or 'finished successfully' not in (INDEX / 'Log.out').read_text():
        raise SystemExit(f'{INDEX} is not a finished STAR index')
    for sub in ('snps', 'work', 'bams', 'facts'):
        OUT.joinpath(sub).mkdir(parents=True, exist_ok=True)
    bams = pd.read_csv(BAMS, sep='\t', header=None, names=['donor', 'bam'])
    print(f'{len(bams)} donors at {JOBS} x {THREADS} threads', flush=True)
    with cf.ThreadPoolExecutor(JOBS) as ex:
        for k, (d, status, minutes) in enumerate(ex.map(one, bams.donor, bams.bam), 1):
            print(f'[{k}/{len(bams)}] {d} {status} {minutes} min', flush=True)
    summary(bams.donor)


if __name__ == '__main__':
    main()

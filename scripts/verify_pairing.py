#!/usr/bin/env python3
"""
Verify a sample pairing against the reads, not against a filename.

Every id in this dataset is a claim someone wrote down. The reads are not:
an RNA alignment carries its donor's genotypes at expressed sites, so the
donor can be identified by comparing the RNA-derived genotypes against every
candidate in the VCF. The correct donor lands near 0.99; everyone else lands
near 0.6, which is where two unrelated people agree by chance given the allele
frequency spectrum. That gap is what makes the check decisive rather than
suggestive.

Use it to confirm a pairing built by scripts/brainvar_pairing.py before
trusting any downstream result. A swap does not announce itself: mispaired
allelic counts still run, still converge, and still produce QTLs.

CONTIG NAMES
============
The BAM and the VCF need not use the same contig names (T2T alignments here
are RefSeq accessions, the VCF is chr-prefixed). Pass --contig-map with
`bam_name<TAB>vcf_name` lines; positions are assumed identical, which holds
when both derive from the same assembly.

Run:
  python3 scripts/verify_pairing.py --selftest
  python3 scripts/verify_pairing.py --pairing cohort/pairing.tsv \\
      --bam-dir <dir> --vcf <phased.bcf> --chrom chr1 \\
      --contig-map rename_chrs.tsv --out verify/
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

INDEL = re.compile(r'[+-](\d+)')


def pileup_bases(field):
    """samtools mpileup read-base column -> list of ACGT, skips/indels removed."""
    out, i = [], 0
    while i < len(field):
        c = field[i]
        if c == '^':
            i += 2
            continue
        if c == '$':
            i += 1
            continue
        if c in '+-':
            m = INDEL.match(field, i)
            if not m:
                i += 1
                continue
            i = m.end() + int(m.group(1))
            continue
        if c in 'ACGTacgt':
            out.append(c.upper())
        i += 1
    return out


def dosage(gt):
    gt = gt.replace('|', '/').split('/')
    if len(gt) != 2 or '.' in gt:
        return None
    return int(gt[0]) + int(gt[1])


def call_rna(nref, nalt, min_depth):
    n = nref + nalt
    if n < min_depth:
        return None
    af = nalt / n
    if af < 0.10:
        return 0
    if 0.25 <= af <= 0.75:
        return 1
    if af > 0.90:
        return 2
    return None


def concordance(pileup_rows, sites, n_samples, min_depth=10):
    """pileup_rows: (pos, bases_field). sites: pos -> (ref, alt, [dosage])."""
    tot = [0] * n_samples
    hit = [0] * n_samples
    used = 0
    for pos, field in pileup_rows:
        s = sites.get(pos)
        if s is None:
            continue
        ref, alt, dos = s
        b = pileup_bases(field)
        r = call_rna(b.count(ref), b.count(alt), min_depth)
        if r is None:
            continue
        used += 1
        for i, d in enumerate(dos):
            if d is None:
                continue
            tot[i] += 1
            if d == r:
                hit[i] += 1
    return used, [hit[i] / tot[i] if tot[i] else None for i in range(n_samples)]


def rank(scores, names, min_sites_ok):
    sc = sorted(((s, n) for s, n in zip(scores, names) if s is not None),
                reverse=True)
    if not sc:
        return None
    best = sc[0]
    second = sc[1] if len(sc) > 1 else (0.0, '-')
    return best[1], best[0], second[1], second[0], best[0] - second[0]


def load_sites(vcf, chrom, samples, stride, bcftools='bcftools'):
    cmd = [bcftools, 'view', '-r', chrom, '-v', 'snps', '-m2', '-M2',
           '-s', ','.join(samples), vcf]
    q = [bcftools, 'query', '-f', '%POS\t%REF\t%ALT[\t%GT]\n']
    p1 = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    p2 = subprocess.Popen(q, stdin=p1.stdout, stdout=subprocess.PIPE,
                          stderr=subprocess.DEVNULL, text=True)
    p1.stdout.close()
    sites = {}
    n = 0
    for line in p2.stdout:
        f = line.rstrip('\n').split('\t')
        if len(f) != 3 + len(samples):
            continue
        n += 1
        if n % stride:
            continue
        if len(f[1]) != 1 or len(f[2]) != 1:
            continue
        dos = [dosage(g) for g in f[3:]]
        if len({d for d in dos if d is not None}) < 2:
            continue            # not discriminative
        sites[int(f[0])] = (f[1], f[2], dos)
    p2.wait()
    return sites


def run_mpileup(bam, positions_file, samtools='samtools', min_mq=10, min_bq=13):
    cmd = [samtools, 'mpileup', '-l', str(positions_file), '-q', str(min_mq),
           '-Q', str(min_bq), '-d', '200', str(bam)]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, text=True)
    rows = []
    for line in p.stdout:
        f = line.rstrip('\n').split('\t')
        if len(f) >= 5:
            rows.append((int(f[1]), f[4]))
    p.wait()
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--pairing', help='dna_library<TAB>rna<TAB>bam (header ok)')
    ap.add_argument('--bam-dir')
    ap.add_argument('--bam-suffix', default='.markdup.sorted.bam')
    ap.add_argument('--vcf')
    ap.add_argument('--chrom', default='chr1')
    ap.add_argument('--contig-map', help='bam_contig<TAB>vcf_contig')
    ap.add_argument('--stride', type=int, default=4,
                    help='use every Nth discriminative site (default 4)')
    ap.add_argument('--min-depth', type=int, default=10)
    ap.add_argument('--min-margin', type=float, default=0.15,
                    help='flag a sample whose best-minus-second is below this')
    ap.add_argument('--limit', type=int, default=0, help='first N rows only')
    ap.add_argument('--out', default='verify')
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    for r in ('pairing', 'bam_dir', 'vcf'):
        if not getattr(args, r):
            raise SystemExit(f'--{r.replace("_","-")} is required (or --selftest)')

    rows = [l.split('\t') for l in Path(args.pairing).read_text().strip().split('\n')]
    if rows and rows[0][0].startswith('dna'):
        rows = rows[1:]
    if args.limit:
        rows = rows[:args.limit]
    names = [r[0] for r in rows]
    print(f'{len(rows)} pairs; loading {args.chrom} genotypes for {len(names)} donors')
    sites = load_sites(args.vcf, args.chrom, names, args.stride)
    if not sites:
        raise SystemExit('no discriminative biallelic SNP sites loaded -- check '
                         '--chrom against the VCF contig names')
    print(f'  {len(sites)} discriminative sites')

    bam_contig = args.chrom
    if args.contig_map:
        m = dict(l.split('\t')[:2] for l in
                 Path(args.contig_map).read_text().strip().split('\n') if '\t' in l)
        rev = {v.strip(): k for k, v in m.items()}
        bam_contig = rev.get(args.chrom, args.chrom)
        if bam_contig != args.chrom:
            print(f'  BAM contig for {args.chrom} is {bam_contig}')

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pos_file = out / 'positions.txt'
    pos_file.write_text('\n'.join(f'{bam_contig}\t{p}' for p in sorted(sites)) + '\n')

    report, bad = [], []
    for dna, rna, bam in rows:
        path = Path(args.bam_dir) / (bam + args.bam_suffix)
        if not path.exists():
            print(f'  {bam}: MISSING {path}')
            bad.append((dna, bam, 'missing bam', None, None))
            continue
        pr = run_mpileup(path, pos_file)
        used, scores = concordance(pr, sites, len(names), args.min_depth)
        r = rank(scores, names, used)
        if r is None:
            bad.append((dna, bam, 'no informative sites', None, None))
            print(f'  {bam}: no informative sites')
            continue
        best, bs, second, ss, margin = r
        ok = (best == dna) and margin >= args.min_margin
        flag = '' if ok else '   <== CHECK'
        print(f'  {bam:<10} claimed={dna:<9} best={best:<9} {bs:.3f}  '
              f'2nd={ss:.3f}  margin={margin:.3f}  n={used}{flag}')
        report.append((dna, rna, bam, best, f'{bs:.4f}', f'{ss:.4f}',
                       f'{margin:.4f}', str(used), 'OK' if ok else 'CHECK'))
        if not ok:
            bad.append((dna, bam, 'best match is not the claimed donor',
                        best, margin))
    (out / 'concordance.tsv').write_text(
        'dna_library\trna_library\tbam\tbest_match\tbest\tsecond\tmargin\tn_sites\tverdict\n'
        + '\n'.join('\t'.join(r) for r in report) + '\n')
    print(f'\n{len(report)} checked, {len(bad)} flagged -> {out}/concordance.tsv')
    return 1 if bad else 0


def selftest():
    import tempfile
    td = Path(tempfile.mkdtemp())
    # three donors; donor B is the truth for our fabricated reads
    names = ['A_D1', 'B_D1', 'C_D1']
    truth = [0, 1, 2, 1, 0, 2, 1, 0, 2, 1] * 6
    other = [2, 0, 1, 0, 2, 1, 0, 2, 1, 0] * 6
    third = [1, 2, 0, 2, 1, 0, 2, 1, 0, 2] * 6
    pos = [1000 + 100 * i for i in range(len(truth))]
    sites = {p: ('A', 'G', [other[i], truth[i], third[i]])
             for i, p in enumerate(pos)}
    # reads matching donor B exactly: dosage 0 -> all ref, 1 -> half, 2 -> all alt
    rows = []
    for i, p in enumerate(pos):
        d = truth[i]
        if d == 0:
            field = 'A' * 20
        elif d == 2:
            field = 'G' * 20
        else:
            field = 'A' * 10 + 'G' * 10
        rows.append((p, field))
    used, sc = concordance(rows, sites, 3)
    assert used == len(pos), used
    best, bs, second, ss, margin = rank(sc, names, used)
    assert best == 'B_D1' and bs == 1.0, (best, bs)
    assert margin > 0.5, margin
    # a read column that is all reference skips must be ignored, not counted
    assert pileup_bases('>>><<<') == []
    assert pileup_bases('AA+2TTGG') == ['A', 'A', 'G', 'G'], pileup_bases('AA+2TTGG')
    assert pileup_bases('^]A$G') == ['A', 'G']
    # depth below the floor yields no call
    assert call_rna(3, 2, 10) is None
    assert call_rna(20, 0, 10) == 0 and call_rna(0, 20, 10) == 2
    assert call_rna(10, 10, 10) == 1
    # an ambiguous allele fraction is dropped rather than guessed
    assert call_rna(80, 20, 10) is None
    # missing genotypes are skipped, not treated as reference
    assert dosage('./.') is None and dosage('0|1') == 1 and dosage('1/1') == 2
    sites2 = {p: ('A', 'G', [None, truth[i], third[i]])
              for i, p in enumerate(pos)}
    used2, sc2 = concordance(rows, sites2, 3)
    assert sc2[0] is None and sc2[1] == 1.0, sc2
    print('SELF-TEST: donor identification on fabricated pileups\n')
    print('checks: the true donor scores 1.000 and wins by a wide margin; '
          'reference skips, indel runs and ^/$ markers are stripped; the depth '
          'floor and the ambiguous-fraction band both refuse to call; missing '
          'genotypes are skipped rather than counted as reference')
    print('SELF-TEST OK')
    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)

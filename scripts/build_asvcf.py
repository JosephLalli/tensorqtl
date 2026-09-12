#!/usr/bin/env python3
"""
Annotate a phased VCF with phASER allele-specific counts, once.

WHY THIS EXISTS
===============
RASQUAL reads a VCF as TEXT on stdin and classifies each record by position:
`isExon(pos, starts, ends, nexon)` at main.c:516 decides what is a feature SNP
from the -s/-e coordinates, so record order is irrelevant and -m is only an
allocation count. Its own example is

    zcat chr11.gz | awk '$1=="11" && $2>=X && $2<=Y' | rasqual -y ... -m 62 ...

and it ships src/ASVCF/createASVCF.sh to produce the AS-annotated VCF.

The comparison was instead parsing the VCF into numpy and then REBUILDING VCF
text per gene in Python -- roughly 312,000 f-strings and numpy scalar lookups
per gene (3,400 variants x 92 samples). That round trip produced bytes that
already existed on disk, and because it runs in the interpreter it holds the
GIL, so threading the RASQUAL calls stalled at 3-4 concurrent instead of 32.

With an AS-annotated, tabix-indexed VCF the per-gene work becomes a pipe, the
threads spend their time in subprocess.run where the GIL is released, and the
text is never reconstructed at all.

AS is written as "ref,alt" per sample, 0,0 where phASER has no count, which is
what parseVCF.c:554 expects.
"""

import argparse
import gzip
import subprocess
import sys
from pathlib import Path

HDR = ('##FORMAT=<ID=AS,Number=.,Type=Integer,Description='
       '"Allele-specific counts (ref,alt) from phASER">')


def load_counts(manifest, regions=None):
    """{(chrom,pos,ref,alt): {sample: 'ref,alt'}} restricted to `regions`.

    Keyed on the alleles as well as the position: the callset has records
    that share a position with different alleles, and a position-only key
    hands both the same counts.
    """
    sys.path.insert(0, str(Path(__file__).parent))
    from run_hapmixqtl_from_salmon import _in_regions
    inside = _in_regions(regions)
    rows = [l.split('\t') for l in Path(manifest).read_text().strip().split('\n')
            if l.strip() and not l.startswith('#')]
    store = {}
    for samp, path in ((r[0].strip(), r[1].strip()) for r in rows):
        op = gzip.open if path.endswith('.gz') else open
        with op(path, 'rt') as fh:
            hdr = fh.readline().rstrip('\n').split('\t')
            ci = hdr.index('contig')
            pi = hdr.index('position') if 'position' in hdr else hdr.index('start')
            ri, ai = hdr.index('refCount'), hdr.index('altCount')
            rai, aai = hdr.index('refAllele'), hdr.index('altAllele')
            for line in fh:
                f = line.rstrip('\n').split('\t')
                if len(f) <= max(ci, pi, ri, ai):
                    continue
                try:
                    c, pos = str(f[ci]), int(f[pi])
                except ValueError:
                    continue
                if inside is not None and not inside(c, pos):
                    continue
                store.setdefault((c, pos, f[rai], f[aai]), {})[samp] = f'{f[ri]},{f[ai]}'
    return store


def annotate(vcf, counts, out, regions=None, bcftools='bcftools'):
    if regions is not None:
        src = subprocess.Popen([bcftools, 'view', '-R', str(regions), str(vcf)],
                               stdout=subprocess.PIPE, text=True,
                               stderr=subprocess.DEVNULL).stdout
    else:
        src = (gzip.open if str(vcf).endswith('.gz') else open)(vcf, 'rt')
    n_rec = n_as = 0
    order = None
    # bgzip, not gzip: tabix needs BGZF blocks, and Python's gzip module
    # writes plain deflate, which indexes with "not compressed with bgzip".
    bg = subprocess.Popen(['bgzip', '-c'], stdin=subprocess.PIPE,
                          stdout=open(out, 'wb'), text=True)
    oh = bg.stdin
    if True:
        for line in src:
            if line.startswith('##'):
                oh.write(line)
                continue
            f = line.rstrip('\n').split('\t')
            if line.startswith('#CHROM'):
                oh.write(HDR + '\n')
                oh.write(line)
                order = f[9:]
                continue
            if ',' in f[4]:
                # the standard arm tests biallelic sites only; keep the pipe
                # on the same variant set
                continue
            n_rec += 1
            per = counts.get((f[0], int(f[1]), f[3], f[4]))
            if per:
                n_as += 1
            f[8] = f[8] + ':AS'
            for k, s in enumerate(order):
                f[9 + k] = f[9 + k] + ':' + ((per or {}).get(s) or '0,0')
            oh.write('\t'.join(f) + '\n')
    oh.close(); bg.wait()
    if bg.returncode != 0:
        raise SystemExit('bgzip failed writing the AS-VCF')
    return n_rec, n_as


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--vcf'); ap.add_argument('--allelic-counts')
    ap.add_argument('--regions'); ap.add_argument('--out')
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    for r in ('vcf', 'allelic_counts', 'out'):
        if not getattr(a, r):
            raise SystemExit(f'--{r.replace("_","-")} is required (or --selftest)')
    counts = load_counts(a.allelic_counts, a.regions)
    print(f'  {len(counts)} sites carry allele-specific counts')
    n_rec, n_as = annotate(a.vcf, counts, a.out, a.regions)
    print(f'  wrote {n_rec} records, {n_as} with AS -> {a.out}')
    subprocess.run(['bcftools', 'index', '-t', '-f', a.out], check=True)
    print('  indexed; per-gene input is now "bcftools view -r REGION | rasqual"')
    return 0


def selftest():
    import tempfile
    td = Path(tempfile.mkdtemp())
    v = td / 'in.vcf.gz'
    with gzip.open(v, 'wt') as fh:
        fh.write('##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER'
                 '\tINFO\tFORMAT\tS1\tS2\n')
        fh.write('chr1\t100\tv1\tA\tG\t.\tPASS\t.\tGT\t0|1\t1|1\n')
        fh.write('chr1\t200\tv2\tA\tG\t.\tPASS\t.\tGT\t0|0\t0|1\n')
        # same position, different alleles: must NOT inherit v1's counts
        fh.write('chr1\t100\tv3\tA\tT\t.\tPASS\t.\tGT\t0|0\t0|0\n')
    for s, rows in (('S1', [('chr1', 100, 7, 3)]), ('S2', [('chr1', 100, 1, 9)])):
        with open(td / f'{s}.ac.txt', 'w') as fh:
            fh.write('contig\tposition\tvariantID\trefAllele\taltAllele'
                     '\trefCount\taltCount\ttotalCount\n')
            for c, p, r, al in rows:
                fh.write(f'{c}\t{p}\tx\tA\tG\t{r}\t{al}\t{r+al}\n')
    man = td / 'man.tsv'
    man.write_text('\n'.join(f'{s}\t{td}/{s}.ac.txt' for s in ('S1', 'S2')))
    counts = load_counts(man)
    assert counts[('chr1', 100, 'A', 'G')] == {'S1': '7,3', 'S2': '1,9'}, counts
    out = td / 'out.vcf.gz'
    n_rec, n_as = annotate(v, counts, out)
    assert (n_rec, n_as) == (3, 1), (n_rec, n_as)
    got = [l.rstrip('\n').split('\t') for l in gzip.open(out, 'rt')
           if not l.startswith('#')]   # bgzip output reads fine as gzip
    # AS appended to FORMAT and to every sample; 0,0 where phASER had nothing
    assert got[0][8] == 'GT:AS', got[0]
    assert got[0][9] == '0|1:7,3' and got[0][10] == '1|1:1,9', got[0]
    assert got[1][9] == '0|0:0,0' and got[1][10] == '0|1:0,0', got[1]
    assert got[2][3:5] == ['A', 'T'] and got[2][9] == '0|0:0,0', got[2]
    hdr = [l for l in gzip.open(out, 'rt') if l.startswith('##FORMAT=<ID=AS')]
    assert hdr, 'AS FORMAT header missing'
    print('SELF-TEST: AS annotation\n')
    print('checks: counts keyed by (contig,position,ref,alt), so a second record '
          'at the same position with other alleles gets 0,0; AS appended to FORMAT and '
          'to every sample column; sites without phASER counts get 0,0; the AS '
          'FORMAT header is declared')
    print('SELF-TEST OK')
    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)

#!/usr/bin/env python3
"""Pick pilot genes both methods can use, stratified by expression.

A gene is a candidate when its median Salmon reads per sample clears the
lowest stratum bound, it has enough exon sites with allele-specific reads
(phASER counts >= --min-cov in >= --min-samples samples; the allelic channel
needs those in both methods) and its exon union holds at most
--max-exon-records VCF records (every record inside an exon is a feature SNP
to RASQUAL, and its cost is (fSNPs+1) x tested SNPs). --per-stratum genes are
drawn from each expression stratum with a fixed seed.

Inputs: the expression summary written from the Gibbs cache (gene,
median_reads, ...), genes.tsv and exons.tsv from gtf_to_tables.py, the
phASER allelic-counts manifest and the analysis VCF.
"""
import argparse
import bisect
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def exon_index(exons, genes):
    """{chrom: (starts[], ends[], gene[]) sorted by start} over `genes`."""
    per = defaultdict(list)
    for g in genes:
        if g not in exons:
            continue
        chrom, ivs = exons[g]
        for a, b in ivs:
            per[chrom].append((a, b, g))
    out = {}
    for c, rows in per.items():
        rows.sort()
        out[c] = (np.array([r[0] for r in rows]), np.array([r[1] for r in rows]),
                  np.array([r[2] for r in rows], object))
    return out


def genes_at(idx, chrom, pos, back=64):
    """Genes whose exon covers pos (intervals overlap across genes; scan back)."""
    e = idx.get(chrom)
    if e is None:
        return []
    starts, ends, gs = e
    k = bisect.bisect_right(starts, pos)
    hits = []
    for j in range(k - 1, max(-1, k - 1 - back), -1):
        if ends[j] >= pos:
            hits.append(gs[j])
    return hits


def load_exons(path, gene_chrom):
    exons = {}
    for l in open(path):
        f = l.rstrip('\n').split('\t')
        if len(f) < 3 or f[0] not in gene_chrom:
            continue
        ivs = [(int(a), int(b)) for a, b in zip(f[1].split(','), f[2].split(','))]
        exons[f[0]] = (gene_chrom[f[0]], ivs)
    return exons


def exon_records(vcf, idx, bed, bcftools='bcftools'):
    """VCF records per gene inside the exon union, via one bcftools query."""
    q = subprocess.run([bcftools, 'query', '-R', str(bed), '-f', '%CHROM\t%POS\n', str(vcf)],
                       capture_output=True, text=True)
    if q.returncode != 0:
        raise SystemExit(f'bcftools query failed: {q.stderr[:300]}')
    n = defaultdict(int)
    for line in q.stdout.split('\n'):
        if not line:
            continue
        c, p = line.split('\t')
        for g in genes_at(idx, c, int(p)):
            n[g] += 1
    return n


def informative_fsnps(manifest, idx, min_cov, min_samples):
    """Per gene: exon sites with >= min_cov phASER reads in >= min_samples samples."""
    rows = [l.split('\t') for l in Path(manifest).read_text().strip().split('\n')
            if l.strip() and not l.startswith('#')]
    covered = defaultdict(int)                    # (gene, chrom, pos) -> n samples
    for k, (samp, path) in enumerate((r[0].strip(), r[1].strip()) for r in rows):
        df = pd.read_csv(path, sep='\t', usecols=['contig', 'position', 'refCount', 'altCount'],
                         dtype={'contig': str, 'position': np.int64, 'refCount': np.int64,
                                'altCount': np.int64})
        df = df[(df['refCount'] + df['altCount']) >= min_cov]
        for c, sub in df.groupby('contig', sort=False):
            e = idx.get(c)
            if e is None:
                continue
            starts, ends, gs = e
            pos = sub['position'].values
            # keep only positions that fall inside some exon before the per-site loop
            kk = np.searchsorted(starts, pos, 'right') - 1
            maybe = (kk >= 0) & (pos <= np.maximum.accumulate(ends)[np.clip(kk, 0, None)])
            for p in pos[maybe]:
                for g in genes_at(idx, c, int(p)):
                    covered[(g, c, int(p))] += 1
        print(f'  [{k+1}/{len(rows)}] {samp}', end='\r', file=sys.stderr)
    print(file=sys.stderr)
    n = defaultdict(int)
    for (g, c, p), ns in covered.items():
        if ns >= min_samples:
            n[g] += 1
    return n


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--expression-summary')
    ap.add_argument('--genes'); ap.add_argument('--exons')
    ap.add_argument('--allelic-counts'); ap.add_argument('--vcf')
    ap.add_argument('--strata', default='1000,3000,10000',
                    help='lower bounds of median reads/sample per stratum')
    ap.add_argument('--per-stratum', type=int, default=10)
    ap.add_argument('--min-fsnps', type=int, default=3, help='informative exon sites')
    ap.add_argument('--min-cov', type=int, default=10); ap.add_argument('--min-samples', type=int, default=10)
    ap.add_argument('--max-exon-records', type=int, default=40)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out'); ap.add_argument('--summary')
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    for r in ('expression_summary', 'genes', 'exons', 'allelic_counts', 'vcf', 'out', 'summary'):
        if getattr(a, r) is None:
            raise SystemExit(f'--{r.replace("_", "-")} is required (or --selftest)')

    bounds = [float(x) for x in a.strata.split(',')]
    ex = pd.read_csv(a.expression_summary, sep='\t')
    cand = ex[ex['median_reads'] >= bounds[0]].set_index('gene')
    gt = pd.read_csv(a.genes, sep='\t', header=None, dtype={1: str})
    gene_chrom = dict(zip(gt[0], gt[1]))
    exons = load_exons(a.exons, gene_chrom)
    cand = cand[cand.index.isin(exons)]
    print(f'{len(cand)} candidates with median >= {bounds[0]:g} reads/sample and exons')
    idx = exon_index(exons, list(cand.index))
    with tempfile.NamedTemporaryFile('w', suffix='.bed', delete=False) as fh:
        for g in cand.index:
            c, ivs = exons[g]
            for s, e in ivs:
                fh.write(f'{c}\t{max(0, s-1)}\t{e}\t{g}\n')
        bed = fh.name
    print('counting VCF records per exon union')
    nrec = exon_records(a.vcf, idx, bed)
    print('scanning phASER counts for informative exon sites')
    nfs = informative_fsnps(a.allelic_counts, idx, a.min_cov, a.min_samples)
    cand = cand.assign(exon_records=[nrec.get(g, 0) for g in cand.index],
                       informative_fsnps=[nfs.get(g, 0) for g in cand.index])
    ok = cand[(cand['informative_fsnps'] >= a.min_fsnps) & (cand['exon_records'] <= a.max_exon_records)
              & (cand['exon_records'] > 0)]
    print(f'{len(ok)} pass (>= {a.min_fsnps} informative fSNPs, <= {a.max_exon_records} exon records)')
    rng = np.random.RandomState(a.seed); picks = []
    for i, lo in enumerate(bounds):
        hi = bounds[i + 1] if i + 1 < len(bounds) else np.inf
        pool = ok[(ok['median_reads'] >= lo) & (ok['median_reads'] < hi)]
        take = list(rng.choice(pool.index, min(a.per_stratum, len(pool)), replace=False))
        print(f'  stratum [{lo:g}, {hi:g}): {len(pool)} eligible, {len(take)} drawn')
        for g in take:
            picks.append(dict(gene=g, stratum=f'{lo:g}-{hi:g}', **pool.loc[g].to_dict()))
    out = pd.DataFrame(picks)
    out.to_csv(a.summary, sep='\t', index=False, float_format='%.3f')
    Path(a.out).write_text('\n'.join(out['gene']) + '\n')
    print(f'wrote {a.out} ({len(out)} genes) and {a.summary}')
    return 0


def selftest():
    """Six genes on one contig; the draw must respect all three filters."""
    import gzip
    td = Path(tempfile.mkdtemp())
    # gene: (median reads, exon union, VCF records in exons, samples covering site)
    # G1 high, 2 exon records, both sites covered in 2 samples      -> eligible, stratum 2
    # G2 mid, 3 exon records, covered                                -> eligible, stratum 1
    # G3 mid, exon records over the cap (5)                          -> excluded
    # G4 mid, exon sites covered in only ONE sample                  -> excluded
    # G5 low expression                                              -> not a candidate
    # G6 mid, no VCF record inside its exons                         -> excluded
    genes = {'G1': ('chr1', 1000, 2000, [(1000, 1200), (1800, 2000)], 5000),
             'G2': ('chr1', 10000, 11000, [(10000, 11000)], 1500),
             'G3': ('chr1', 20000, 21000, [(20000, 21000)], 1500),
             'G4': ('chr1', 30000, 31000, [(30000, 31000)], 1500),
             'G5': ('chr1', 40000, 41000, [(40000, 41000)], 50),
             'G6': ('chr1', 50000, 51000, [(50000, 51000)], 1500)}
    (td / 'expr.tsv').write_text('gene\tmedian_reads\n' + ''.join(f'{g}\t{v[4]}\n' for g, v in genes.items()))
    (td / 'genes.tsv').write_text(''.join(f'{g}\t{v[0]}\t{v[1]}\t{v[2]}\t{v[1]}\n' for g, v in genes.items()))
    (td / 'exons.tsv').write_text(''.join(
        f'{g}\t{",".join(str(a) for a, _ in v[3])}\t{",".join(str(b) for _, b in v[3])}\n' for g, v in genes.items()))
    sites = {'G1': [1100, 1900], 'G2': [10100, 10200, 10300], 'G3': [20100, 20200, 20300, 20400, 20500],
             'G4': [30100, 30200, 30300], 'G5': [40100], 'G6': [1500]}      # G6's record sits in its intron
    hdr = '#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\tS2\n'
    body = ''.join(f'chr1\t{p}\t.\tA\tG\t.\tPASS\t.\tGT\t0|1\t0|1\n'
                   for g in sites for p in sorted(sites[g]))
    body = ''.join(sorted(body.splitlines(True), key=lambda l: int(l.split('\t')[1])))
    raw = td / 'v.vcf'; raw.write_text('##fileformat=VCFv4.2\n' + hdr + body)
    vcf = td / 'v.vcf.gz'
    with open(vcf, 'wb') as fh:
        subprocess.run(['bgzip', '-c', str(raw)], stdout=fh, check=True)
    subprocess.run(['bcftools', 'index', '-t', '-f', str(vcf)], check=True)
    cov = {'S1': ['G1', 'G2', 'G3', 'G4', 'G6'], 'S2': ['G1', 'G2', 'G3', 'G6']}   # G4 covered in S1 only
    man = []
    for s_, gl in cov.items():
        f = td / f'{s_}.ac.txt'
        with open(f, 'w') as fh:
            fh.write('contig\tposition\tvariantID\trefAllele\taltAllele\trefCount\taltCount\ttotalCount\n')
            for g in gl:
                for p in sites[g]:
                    fh.write(f'chr1\t{p}\tx\tA\tG\t8\t7\t15\n')
        man.append(f'{s_}\t{f}')
    (td / 'man.tsv').write_text('\n'.join(man))
    out, summ = td / 'pick.txt', td / 'pick.tsv'
    main(['--expression-summary', str(td / 'expr.tsv'), '--genes', str(td / 'genes.tsv'),
          '--exons', str(td / 'exons.tsv'), '--allelic-counts', str(td / 'man.tsv'), '--vcf', str(vcf),
          '--strata', '1000,3000', '--per-stratum', '5', '--min-fsnps', '2', '--min-cov', '10',
          '--min-samples', '2', '--max-exon-records', '4', '--out', str(out), '--summary', str(summ)])
    got = pd.read_csv(summ, sep='\t').set_index('gene')
    assert sorted(got.index) == ['G1', 'G2'], got
    assert got.loc['G1', 'stratum'] == '3000-inf' and got.loc['G2', 'stratum'] == '1000-3000', got
    assert got.loc['G1', 'exon_records'] == 2 and got.loc['G2', 'exon_records'] == 3, got
    assert got.loc['G1', 'informative_fsnps'] == 2 and got.loc['G2', 'informative_fsnps'] == 3, got
    assert out.read_text().split() == list(got.index)
    print('SELF-TEST: pilot gene selection\n')
    print('checks: the expression floor drops G5; the exon-record cap drops G3; a site covered '
          'in one sample is not informative (G4 dropped); a VCF record in an intron is not a '
          'feature SNP (G6 dropped); survivors land in the right strata with the right counts')
    print('SELF-TEST OK')
    return 0


if __name__ == '__main__':
    sys.exit(main())

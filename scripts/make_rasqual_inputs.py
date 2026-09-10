#!/usr/bin/env python3
"""
Make RASQUAL's input files from files you already have.

Produces everything `rasqual` needs and a ready-to-run command per gene:

    <out>/Y.bin, Y.txt        feature x sample total counts   (float64, row-major)
    <out>/K.bin, K.txt        feature x sample offsets         (float64, row-major)
    <out>/as.vcf.gz [+ .tbi]  phased GT + "AS" ref,alt counts at every SNP
    <out>/genes.tsv           per-gene -j / -l / -m / -s / -e arguments
    <out>/run_rasqual.sh      one rasqual command per gene, in the documented
                              form: `tabix as.vcf.gz region | rasqual ...`

Format verified against the authors' bundled data (data/Y.bin = 2 genes x 24
samples of float64; VCF FORMAT `GT:AS` with `0|1:1,10`).

INPUTS
======
Genotypes
  --vcf         phased VCF/BCF-as-VCF (GT with '|'). Biallelic SNPs only are
                used; unphased or missing genotypes are skipped.
  --samples     one sample ID per line, in the order you want. Must match VCF
                sample names. Y/K columns and the AS VCF follow this order.

Genes
  --genes       TSV: gene_id  chr  start  end  [tss]   (1-based, inclusive).
                TSS defaults to `start`. Chromosome names must match the VCF.

Total counts  (pick one)
  --counts      gene x sample matrix (featureCounts/HTSeq/STAR style): first
                column gene_id, header row of sample IDs.
  --salmon      manifest `sample_id <TAB> salmon_dir` + --tx2gene; sums
                quant.sf NumReads per gene. Haplotype-suffixed transcripts are
                collapsed to their base ID automatically.

Allele-specific counts  (pick one; the first two are RASQUAL-NATIVE)
  --allelic-counts   manifest `sample_id <TAB> phASER .allelic_counts.txt`
  --bams             manifest `sample_id <TAB> bam`; pileup at each sample's
                     het sites (needs pysam). Base quality >= --min-bq, MAPQ
                     >= --min-mq, mirroring GTEx phASER's settings.
  --salmon-diploid   FALLBACK: gene-level haplotype totals from a diploid
                     Salmon run, encoded as ONE pseudo-fSNP per gene at the
                     TSS. Clearly not RASQUAL's native input -- see below.

WHY --salmon-diploid IS A FALLBACK, NOT AN EQUIVALENT
=====================================================
RASQUAL's likelihood is per feature SNP: p(Y1_il | Y_il, D_il), with D_il set
by the regulatory-SNP genotype AND the feature-SNP genotype. A diploid Salmon
run assigns each fragment to a haplotype using all variants jointly and reports
only a gene-level total; the per-site breakdown does not exist in its output.
It is NOT recoverable by dividing the gene total across sites: that would hand
RASQUAL L observations where one was measured, inflating its effective sample
size and its chi-square. Encoding the total as a single pseudo-fSNP is the
honest version -- it presents exactly the information that was measured -- but
it denies RASQUAL the per-site resolution it was designed around. Use
--allelic-counts or --bams whenever the alignments exist.

OFFSETS (K)
===========
RASQUAL's K_i is a sample-specific offset. Following the supplement, K is the
library size factor (DESeq median-of-ratios over the count matrix) scaled by
each gene's mean count, so that Y_ij / K_ij is a normalized abundance. Supply
--offsets to override with your own gene x sample matrix.

Run:
  python3 scripts/make_rasqual_inputs.py --selftest
  python3 scripts/make_rasqual_inputs.py --vcf p.vcf.gz --samples s.txt \\
      --genes genes.tsv --counts counts.tsv --allelic-counts ac.tsv --out rq/
  bash rq/run_rasqual.sh          # or submit each line to your scheduler
"""

import argparse
import gzip
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _open(p):
    return gzip.open(p, 'rt') if str(p).endswith('.gz') else open(p)


# ---------------------------------------------------------------------------
#  Genotypes
# ---------------------------------------------------------------------------

def read_vcf(path, samples):
    """Biallelic phased SNPs -> DataFrame(chrom,pos,id,ref,alt) + xL,xR [V,N]."""
    want = {s: i for i, s in enumerate(samples)}
    rows, XL, XR = [], [], []
    with _open(path) as fh:
        for line in fh:
            if line.startswith('##'):
                continue
            f = line.rstrip('\n').split('\t')
            if line.startswith('#CHROM'):
                vs = f[9:]
                missing = [s for s in samples if s not in vs]
                if missing:
                    raise SystemExit(f'{len(missing)} --samples not in VCF, e.g. '
                                     f'{missing[:3]}. VCF has e.g. {vs[:3]}')
                col = [9 + vs.index(s) for s in samples]
                continue
            if len(f) < 10 or len(f[3]) != 1 or len(f[4]) != 1 or f[4] == '.':
                continue
            fmt = f[8].split(':')
            gi = fmt.index('GT') if 'GT' in fmt else 0
            xl = np.zeros(len(samples)); xr = np.zeros(len(samples)); ok = True
            for k, c in enumerate(col):
                gt = f[c].split(':')[gi]
                if '|' not in gt:
                    ok = False; break
                a, b = gt.split('|')[:2]
                if a == '.' or b == '.':
                    ok = False; break
                xl[k] = float(a != '0'); xr[k] = float(b != '0')
            if not ok:
                continue
            vid = f[2] if f[2] not in ('.', '') else f'{f[0]}_{f[1]}_{f[3]}_{f[4]}'
            rows.append((str(f[0]), int(f[1]), vid, f[3], f[4]))
            XL.append(xl); XR.append(xr)
    if not rows:
        raise SystemExit('no phased biallelic SNPs found in the VCF')
    vdf = pd.DataFrame(rows, columns=['chrom', 'pos', 'id', 'ref', 'alt'])
    return vdf, np.array(XL), np.array(XR)


# ---------------------------------------------------------------------------
#  Total counts and offsets
# ---------------------------------------------------------------------------

def read_counts_matrix(path, samples):
    df = pd.read_csv(path, sep='\t', index_col=0)
    missing = [s for s in samples if s not in df.columns]
    if missing:
        raise SystemExit(f'--counts lacks samples {missing[:3]}...')
    return df[samples].astype(float)


def read_salmon_totals(manifest, tx2gene, samples, suffixes=('_hapA', '_hapB')):
    t2g = dict(l.split('\t')[:2] for l in
               Path(tx2gene).read_text().strip().split('\n') if '\t' in l)
    dirs = dict(l.split('\t')[:2] for l in
                Path(manifest).read_text().strip().split('\n')
                if l.strip() and not l.startswith('#'))
    cols = {}
    for s in samples:
        if s not in dirs:
            raise SystemExit(f'sample {s} not in --salmon manifest')
        q = pd.read_csv(Path(dirs[s].strip()) / 'quant.sf', sep='\t')
        base = q['Name'].astype(str)
        for suf in suffixes:
            base = base.str.replace(f'{suf}$', '', regex=True)
        g = base.map(t2g)
        cols[s] = q.assign(gene=g).dropna(subset=['gene']).groupby('gene')['NumReads'].sum()
    return pd.DataFrame(cols).fillna(0.0)


def size_factors(Y):
    """DESeq median-of-ratios; falls back to library-size ratio if degenerate."""
    Y = np.asarray(Y, float)
    pos = (Y > 0).all(1)
    if pos.sum() >= 10:
        logs = np.log(Y[pos])
        sf = np.exp(np.median(logs - logs.mean(1, keepdims=True), axis=0))
    else:
        lib = Y.sum(0); sf = lib / np.exp(np.mean(np.log(np.maximum(lib, 1))))
    return sf


# ---------------------------------------------------------------------------
#  Allele-specific counts
# ---------------------------------------------------------------------------

def as_from_phaser(manifest, samples):
    """{(chrom,pos): {sample: (ref,alt)}} from phASER allelic_counts files."""
    store = {}
    for l in Path(manifest).read_text().strip().split('\n'):
        if not l.strip() or l.startswith('#'):
            continue
        s, path = [x.strip() for x in l.split('\t')[:2]]
        if s not in samples:
            continue
        with _open(path) as fh:
            hdr = fh.readline().rstrip('\n').split('\t')
            try:
                ci, pi = hdr.index('contig'), hdr.index('start')
                ri, ai = hdr.index('refCount'), hdr.index('altCount')
            except ValueError:
                raise SystemExit(f'{path}: not a phASER allelic_counts file '
                                 f'(columns {hdr[:8]})')
            for line in fh:
                f = line.rstrip('\n').split('\t')
                try:
                    store.setdefault((str(f[ci]), int(f[pi])), {})[s] = \
                        (int(f[ri]), int(f[ai]))
                except (ValueError, IndexError):
                    continue
    return store


def as_from_bams(manifest, samples, vdf, xL, xR, min_bq=10, min_mq=20):
    """Pileup ref/alt at each sample's HET sites. Needs pysam."""
    try:
        import pysam
    except ImportError:
        raise SystemExit('--bams needs pysam (pip install pysam), or use '
                         '--allelic-counts from phASER instead')
    bams = dict(l.split('\t')[:2] for l in
                Path(manifest).read_text().strip().split('\n')
                if l.strip() and not l.startswith('#'))
    store = {}
    het = (xL != xR)
    for k, s in enumerate(samples):
        if s not in bams:
            raise SystemExit(f'sample {s} not in --bams manifest')
        af = pysam.AlignmentFile(bams[s].strip(), 'rb')
        contigs = set(af.references)
        sites = np.where(het[:, k])[0]
        print(f'  pileup {s}: {sites.size} het sites', flush=True)
        for v in sites:
            chrom = vdf.chrom.iat[v]
            c = chrom if chrom in contigs else (
                chrom[3:] if chrom.startswith('chr') and chrom[3:] in contigs
                else ('chr' + chrom if 'chr' + chrom in contigs else None))
            if c is None:
                continue
            pos0 = int(vdf.pos.iat[v]) - 1
            try:
                cov = af.count_coverage(c, pos0, pos0 + 1, quality_threshold=min_bq,
                                        read_callback=lambda r: r.mapping_quality >= min_mq)
            except ValueError:
                continue
            base = {b: int(cov[i][0]) for i, b in enumerate('ACGT')}
            r = base.get(vdf.ref.iat[v], 0); a = base.get(vdf.alt.iat[v], 0)
            if r + a:
                store.setdefault((chrom, int(vdf.pos.iat[v])), {})[s] = (r, a)
        af.close()
    return store


def as_from_salmon_diploid(manifest, tx2gene, samples, genes, suffixes):
    """Gene-level haplotype totals -> {gene: {sample: (hapA, hapB)}}."""
    t2g = dict(l.split('\t')[:2] for l in
               Path(tx2gene).read_text().strip().split('\n') if '\t' in l)
    dirs = dict(l.split('\t')[:2] for l in
                Path(manifest).read_text().strip().split('\n')
                if l.strip() and not l.startswith('#'))
    sa, sb = suffixes
    out = {g: {} for g in genes}
    for s in samples:
        q = pd.read_csv(Path(dirs[s].strip()) / 'quant.sf', sep='\t')
        q = q.set_index('Name')['NumReads']
        A = q[q.index.str.endswith(sa)]; B = q[q.index.str.endswith(sb)]
        A.index = A.index.str[:-len(sa)]; B.index = B.index.str[:-len(sb)]
        gA = A.groupby(A.index.map(t2g)).sum(); gB = B.groupby(B.index.map(t2g)).sum()
        for g in genes:
            a, b = float(gA.get(g, 0.0)), float(gB.get(g, 0.0))
            if a + b > 0:
                out[g][s] = (int(round(a)), int(round(b)))
    return out


# ---------------------------------------------------------------------------
#  Writers
# ---------------------------------------------------------------------------

def write_bin(mat, out, name, genes, samples):
    np.asarray(mat, np.float64).tofile(Path(out) / f'{name}.bin')
    pd.DataFrame(mat, index=genes, columns=samples).to_csv(
        Path(out) / f'{name}.txt', sep='\t', header=False)


def write_as_vcf(out, vdf, xL, xR, samples, site_as, pseudo=None, genes_df=None):
    """One VCF, sorted, with AS at every SNP (0,0 where unmeasured).

    pseudo: {gene: {sample: (hapA,hapB)}} -> adds one pseudo-fSNP per gene at
    its TSS, GT 0|1 (REF on haplotype 1 => AS is (hapA, hapB)).
    """
    rows = []
    for v in range(len(vdf)):
        key = (vdf.chrom.iat[v], int(vdf.pos.iat[v]))
        cnt = site_as.get(key, {})
        fl = []
        for k, s in enumerate(samples):
            r, a = cnt.get(s, (0, 0))
            fl.append(f'{int(xL[v, k])}|{int(xR[v, k])}:{r},{a}')
        rows.append((vdf.chrom.iat[v], int(vdf.pos.iat[v]), vdf.id.iat[v],
                     vdf.ref.iat[v], vdf.alt.iat[v], fl))
    if pseudo:
        for g, row in genes_df.iterrows():
            cnt = pseudo.get(g, {})
            if not cnt:
                continue
            fl = [(f'0|1:{cnt[s][0]},{cnt[s][1]}' if s in cnt else '0|0:0,0')
                  for s in samples]
            rows.append((str(row['chr']), int(row['tss']), f'pseudo_fsnp_{g}',
                         'A', 'G', fl))
    rows.sort(key=lambda r: (r[0], r[1]))
    plain = Path(out) / 'as.vcf'
    with open(plain, 'w') as fh:
        fh.write('##fileformat=VCFv4.1\n')
        fh.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Phased genotype">\n')
        fh.write('##FORMAT=<ID=AS,Number=2,Type=Integer,Description='
                 '"Allele-specific counts: ref,alt">\n')
        fh.write('#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t'
                 + '\t'.join(samples) + '\n')
        for c, p, i, r, a, fl in rows:
            fh.write(f'{c}\t{p}\t{i}\t{r}\t{a}\t100\tPASS\t.\tGT:AS\t'
                     + '\t'.join(fl) + '\n')
    gz = Path(out) / 'as.vcf.gz'
    if shutil.which('bgzip') and shutil.which('tabix'):
        subprocess.run(['bgzip', '-f', str(plain)], check=True)
        subprocess.run(['tabix', '-f', '-p', 'vcf', str(gz)], check=True)
        return gz, True
    with open(plain, 'rb') as src, gzip.open(gz, 'wb') as dst:
        shutil.copyfileobj(src, dst)
    plain.unlink()
    return gz, False


def plan_genes(genes_df, vdf, site_as, pseudo, window):
    """Per-gene RASQUAL arguments: which SNPs are fSNPs, which are tested."""
    recs = []
    for j, (g, row) in enumerate(genes_df.iterrows(), start=1):
        chrom, tss = str(row['chr']), int(row['tss'])
        gs, ge = int(row['start']), int(row['end'])
        same = vdf.chrom.values == chrom
        pos = vdf.pos.values
        win = same & (np.abs(pos - tss) <= window)
        body = same & (pos >= gs) & (pos <= ge)
        n_fsnp = int(sum(1 for v in np.where(body)[0]
                         if (chrom, int(pos[v])) in site_as))
        if pseudo is not None and g in pseudo and pseudo[g]:
            n_fsnp += 1; s_arg, e_arg = tss, tss
        else:
            s_arg, e_arg = gs, ge
        n_tested = int(win.sum()) + (1 if pseudo and g in pseudo and pseudo[g] else 0)
        recs.append(dict(j=j, gene=g, chr=chrom, tss=tss, start=gs, end=ge,
                         n_fsnp=n_fsnp, n_tested=n_tested, s_arg=s_arg, e_arg=e_arg,
                         region=f'{chrom}:{max(1, tss - window)}-{tss + window}'))
    return pd.DataFrame(recs)


def write_run_script(out, plan, N, vcf_gz, indexed, rasqual='rasqual'):
    lines = ['#!/usr/bin/env bash', 'set -euo pipefail',
             f'RQ="${{RASQUAL:-{rasqual}}}"', f'D="$(cd "$(dirname "$0")" && pwd)"',
             'mkdir -p "$D/results"', '']
    for _, r in plan.iterrows():
        if r.n_fsnp == 0:
            lines.append(f'# {r.gene}: no feature SNPs with counts -- skipped')
            continue
        lo, hi = r.region.split(':')[1].split('-')
        src = (f'tabix "$D/{vcf_gz.name}" {r.region}' if indexed else
               f'zcat "$D/{vcf_gz.name}" | awk -F\'\\t\' \'$1=="{r.chr}" && '
               f'$2>={lo} && $2<={hi}\'')
        lines.append(
            f'{src} | "$RQ" -y "$D/Y.bin" -k "$D/K.bin" -n {N} -j {r.j} '
            f'-l {r.n_tested} -m {r.n_fsnp} -s {r.s_arg} -e {r.e_arg} '
            f'-f {r.gene} -z > "$D/results/{r.gene}.txt"')
    (Path(out) / 'run_rasqual.sh').write_text('\n'.join(lines) + '\n')


# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--vcf'); ap.add_argument('--samples'); ap.add_argument('--genes')
    ap.add_argument('--counts'); ap.add_argument('--salmon'); ap.add_argument('--tx2gene')
    ap.add_argument('--offsets', help='gene x sample offset matrix to use as K')
    ap.add_argument('--allelic-counts'); ap.add_argument('--bams')
    ap.add_argument('--salmon-diploid', action='store_true',
                    help='FALLBACK pseudo-fSNP from a diploid Salmon run (--salmon)')
    ap.add_argument('--hap-suffix', default='_hapA,_hapB')
    ap.add_argument('--min-bq', type=int, default=10)
    ap.add_argument('--min-mq', type=int, default=20)
    ap.add_argument('--window', type=int, default=1_000_000)
    ap.add_argument('--rasqual-bin', default='rasqual')
    ap.add_argument('--out', default='rasqual_inputs')
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    for r in ('vcf', 'samples', 'genes'):
        if not getattr(args, r):
            raise SystemExit(f'--{r} is required')
    if not (args.counts or (args.salmon and args.tx2gene)):
        raise SystemExit('total counts: give --counts, or --salmon with --tx2gene')
    if not (args.allelic_counts or args.bams or args.salmon_diploid):
        raise SystemExit('allele-specific counts: give --allelic-counts, --bams, '
                         'or --salmon-diploid')
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    sufs = tuple(args.hap_suffix.split(','))

    samples = [l.strip() for l in open(args.samples) if l.strip()]
    print(f'{len(samples)} samples')

    print('Reading genes')
    gdf = pd.read_csv(args.genes, sep='\t', header=None, dtype={1: str})
    gdf.columns = ['gene', 'chr', 'start', 'end', 'tss'][:gdf.shape[1]]
    if 'tss' not in gdf:
        gdf['tss'] = gdf['start']
    gdf = gdf.set_index('gene')
    gdf['chr'] = gdf['chr'].astype(str).str.strip()

    print('Reading phased VCF')
    vdf, xL, xR = read_vcf(args.vcf, samples)
    print(f'  {len(vdf)} phased biallelic SNPs')
    if not set(vdf.chrom) & set(gdf.chr):
        raise SystemExit(f'chromosome names differ: VCF {sorted(set(vdf.chrom))[:3]} '
                         f'vs --genes {sorted(set(gdf.chr))[:3]}')

    print('Total counts')
    Y = (read_counts_matrix(args.counts, samples) if args.counts
         else read_salmon_totals(args.salmon, args.tx2gene, samples, sufs))
    genes = [g for g in gdf.index if g in Y.index]
    dropped = len(gdf) - len(genes)
    if dropped:
        print(f'  {dropped} genes in --genes have no counts; dropped')
    gdf = gdf.loc[genes]; Y = Y.loc[genes]
    if args.offsets:
        K = pd.read_csv(args.offsets, sep='\t', index_col=0).loc[genes, samples]
    else:
        sf = size_factors(Y.values)
        K = pd.DataFrame(np.outer(Y.values.mean(1), sf), index=genes, columns=samples)
    write_bin(Y.values, out, 'Y', genes, samples)
    write_bin(K.values, out, 'K', genes, samples)
    print(f'  wrote Y.bin / K.bin: {len(genes)} genes x {len(samples)} samples')

    print('Allele-specific counts')
    site_as, pseudo = {}, None
    if args.allelic_counts:
        site_as = as_from_phaser(args.allelic_counts, samples)
        print(f'  phASER: {len(site_as)} sites with counts  [RASQUAL native]')
    elif args.bams:
        site_as = as_from_bams(args.bams, samples, vdf, xL, xR,
                               args.min_bq, args.min_mq)
        print(f'  pileup: {len(site_as)} het sites with counts  [RASQUAL native]')
    else:
        if not (args.salmon and args.tx2gene):
            raise SystemExit('--salmon-diploid needs --salmon and --tx2gene')
        pseudo = as_from_salmon_diploid(args.salmon, args.tx2gene, samples,
                                        genes, sufs)
        print(f'  pseudo-fSNP from diploid Salmon: '
              f'{sum(1 for g in pseudo if pseudo[g])} genes  '
              f'[FALLBACK -- not RASQUAL native; see docstring]')

    vcf_gz, indexed = write_as_vcf(out, vdf, xL, xR, samples, site_as, pseudo, gdf)
    print(f'  wrote {vcf_gz.name}' + ('' if indexed else
          '  (bgzip/tabix not found: plain gzip; run_rasqual.sh uses zcat|awk)'))

    plan = plan_genes(gdf, vdf, site_as, pseudo, args.window)
    plan.to_csv(out / 'genes.tsv', sep='\t', index=False)
    write_run_script(out, plan, len(samples), vcf_gz, indexed, args.rasqual_bin)
    usable = int((plan.n_fsnp > 0).sum())
    print(f'  wrote genes.tsv and run_rasqual.sh: {usable}/{len(plan)} genes '
          f'have >= 1 feature SNP with counts')
    (out / 'manifest.json').write_text(json.dumps({
        'n_samples': len(samples), 'n_genes': len(genes), 'n_usable': usable,
        'n_snps': int(len(vdf)),
        'as_source': ('phaser' if args.allelic_counts else
                      'bam_pileup' if args.bams else 'salmon_diploid_pseudo_fsnp'),
        'rasqual_native': bool(args.allelic_counts or args.bams),
        'window': args.window}, indent=2))
    print(f'\nNext:  bash {out}/run_rasqual.sh')


# ---------------------------------------------------------------------------

def selftest():
    import tempfile
    td = Path(tempfile.mkdtemp()); rng = np.random.RandomState(0)
    N, G = 30, 12
    samples = [f'S{i:02d}' for i in range(N)]
    (td / 'samples.txt').write_text('\n'.join(samples))
    # genes: body 400..600 around tss, spaced 10kb
    (td / 'genes.tsv').write_text('\n'.join(
        f'G{g}\t1\t{10000*g+400}\t{10000*g+600}\t{10000*g+500}' for g in range(G)))
    # VCF: one fSNP inside each body + one rSNP outside
    lines = ['##fileformat=VCFv4.2',
             '#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t' + '\t'.join(samples)]
    hets = {}
    for g in range(G):
        for pos, tag in ((10000 * g + 500, 'f'), (10000 * g + 2000, 'r')):
            gts = []
            for i in range(N):
                a, b = int(rng.rand() < .4), int(rng.rand() < .4)
                gts.append(f'{a}|{b}')
                if tag == 'f' and a != b:
                    hets[(g, i)] = True
            lines.append(f'1\t{pos}\t{tag}{g}\tA\tG\t.\tPASS\t.\tGT\t' + '\t'.join(gts))
    (td / 'p.vcf').write_text('\n'.join(lines) + '\n')
    # counts
    Y = pd.DataFrame(rng.poisson(200, (G, N)), index=[f'G{g}' for g in range(G)],
                     columns=samples)
    Y.to_csv(td / 'counts.tsv', sep='\t')
    # phASER-style allelic counts at the fSNPs, hets only
    man = []
    for i, s in enumerate(samples):
        f = td / f'{s}.ac.txt'
        with open(f, 'w') as fh:
            fh.write('contig\tstart\tstop\tvariantID\trefAllele\taltAllele\t'
                     'refCount\taltCount\ttotalCount\n')
            for g in range(G):
                if (g, i) in hets:
                    r, a = rng.poisson(15), rng.poisson(15)
                    fh.write(f'1\t{10000*g+500}\t{10000*g+501}\tf{g}\tA\tG\t{r}\t{a}\t{r+a}\n')
        man.append(f'{s}\t{f}')
    (td / 'ac.tsv').write_text('\n'.join(man))

    print('SELF-TEST: building RASQUAL inputs from fabricated files\n')
    main(['--vcf', str(td / 'p.vcf'), '--samples', str(td / 'samples.txt'),
          '--genes', str(td / 'genes.tsv'), '--counts', str(td / 'counts.tsv'),
          '--allelic-counts', str(td / 'ac.tsv'), '--out', str(td / 'rq')])
    out = td / 'rq'
    # verify the binary layout matches the authors' convention
    Yb = np.fromfile(out / 'Y.bin', dtype=np.float64).reshape(G, N)
    assert np.allclose(Yb, Y.values), 'Y.bin layout mismatch'
    plan = pd.read_csv(out / 'genes.tsv', sep='\t')
    assert (plan.n_fsnp == 1).all(), plan
    print('\nchecks: Y.bin round-trips row-major; every gene has its 1 fSNP; '
          f'run_rasqual.sh has {sum(1 for l in open(out/"run_rasqual.sh") if "-j" in l)} '
          'commands')
    rq = shutil.which('rasqual') or __import__('os').environ.get('RASQUAL_BIN')
    if rq and Path(rq).exists():
        print(f'\nrunning the first command with {rq}...')
        cmd = [l for l in open(out / 'run_rasqual.sh') if '-j 1 ' in l][0]
        (out / 'results').mkdir(exist_ok=True)
        # run exactly as the generated script would: D and RQ defined first
        r = subprocess.run(['bash', '-c', f'D="{out}"; RQ="{rq}"; {cmd}'],
                           capture_output=True, text=True)
        res = out / 'results' / 'G0.txt'
        first = res.read_text().split('\n')[0].split('\t') if res.exists() else []
        ok = len(first) > 22 and first[1] != 'SKIPPED'
        if not ok:
            raise SystemExit(f'SELF-TEST FAILED: RASQUAL did not produce a result.\n'
                             f'stderr: {r.stderr[-400:]}')
        print(f'  RASQUAL ran: {first[0]} lead={first[1]} '
              f'chi2={float(first[10]):.3f} phi={float(first[13]):.3f} '
              f'converged={first[22]}')
    print('\nSELF-TEST OK')
    return 0


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Build genes.tsv and tx2gene.tsv from a GENCODE/Ensembl GTF.

These are the two annotation inputs the pipeline needs and that nothing else
produces:

    genes.tsv     gene_id  chr  start  end  tss      (1-based, inclusive)
                  -> --genes for compare_pipelines.py / make_rasqual_inputs.py
    tx2gene.tsv   transcript_id  gene_id
                  -> --tx2gene, to collapse Salmon quant.sf to genes
    genes.bed     chr  start  stop  gene_id           (0-based, half-open)
                  -> --features for phASER's phaser_gene_ae.py, whose output
                     phaser_to_matrix.py then reads as <prefix>.gene_ae.txt

genes.bed IS NOT genes.tsv WITH COLUMNS MOVED
=============================================
Two differences, both silent if you get them wrong. phaser_gene_ae.py takes
0-BASED coordinates while a GTF is 1-based inclusive, so start becomes
start-1 and the span length is preserved. And it parses the file positionally
from line 1 with `int(columns[1])` -- there is no header handling, so a
`contig start stop name` header line is a crash, not a comment.

The 4th column is the gene_id, not the gene_name. phaser_gene_ae.py copies it
through to its `name` column, phaser_to_matrix.py indexes genes by it, and
compare_pipelines.py joins that against genes.tsv, which is keyed by gene_id.
A symbol there joins to nothing, with no error.

TWO THINGS THAT GO WRONG SILENTLY, HANDLED HERE
===============================================
TSS depends on strand. For a + strand gene the TSS is `start`; for a - strand
gene it is `end`. Using `start` for everything puts half your cis windows in
the wrong place by the length of the gene. This script uses the strand.

Ensembl IDs carry versions (ENSG00000123456.7, ENST...). Salmon quant.sf names
usually keep the version; published eGene lists often drop it. A version
mismatch makes every join silently return nothing -- --tx2gene pairs zero
transcripts, --known-egenes replicates zero genes. Pick ONE convention and
apply it everywhere: --strip-version drops it from both output tables, and then
you must strip it from the Salmon names and the eGene list too (or leave it on
everywhere). The runbook says which.

Gene body here is the annotated gene span (start..end), which is what RASQUAL's
-s/-e feature-SNP window and the tested-variant exclusion use. Chromosome names
are emitted exactly as in the GTF; they must match your VCF ("chr1" vs "1").

Run:
  python3 scripts/gtf_to_tables.py --selftest
  python3 scripts/gtf_to_tables.py --gtf gencode.v39.annotation.gtf.gz \\
      --gene-type protein_coding --out annot/
"""

import argparse
import gzip
import re
import sys
from pathlib import Path

_ATTR = re.compile(r'(\w+) "([^"]*)"')


def _open(p):
    # Sniff the gzip magic number rather than trusting the extension. Reference
    # annotations are routinely mis-named in both directions, and the extension
    # is only a claim about the bytes: a plain-text file called .gtf.gz raises
    # BadGzipFile from inside the stdlib, and a real gzip called .gtf dies with
    # a UnicodeDecodeError partway through the parse. Neither names the file.
    with open(p, 'rb') as fh:
        gzipped = fh.read(2) == b'\x1f\x8b'
    return gzip.open(p, 'rt') if gzipped else open(p)


def _attrs(s):
    return dict(_ATTR.findall(s))


def _strip(i, strip):
    return i.split('.')[0] if strip and i.startswith('ENS') else i


def parse_gtf(path, gene_types=None, strip=False):
    genes, tx2gene = {}, []
    n_gene = n_tx = 0
    with _open(path) as fh:
        for line in fh:
            if line.startswith('#'):
                continue
            f = line.rstrip('\n').split('\t')
            if len(f) < 9 or f[2] not in ('gene', 'transcript'):
                continue
            a = _attrs(f[8])
            gid = _strip(a.get('gene_id', ''), strip)
            if not gid:
                continue
            if f[2] == 'gene':
                gt = a.get('gene_type') or a.get('gene_biotype')
                if gene_types and gt not in gene_types:
                    continue
                start, end, strand = int(f[3]), int(f[4]), f[6]
                tss = start if strand == '+' else end
                genes[gid] = (f[0], start, end, tss, strand)
                n_gene += 1
            else:
                tid = _strip(a.get('transcript_id', ''), strip)
                if tid:
                    tx2gene.append((tid, gid))
    if not genes:
        raise SystemExit('no gene records parsed -- is this a GENCODE/Ensembl GTF '
                         'with "gene" feature rows and gene_id/gene_type attributes?')
    # --gene-type selects GENES; a transcript is kept because its gene was kept,
    # not because its own row repeats the biotype. GENCODE repeats gene_type on
    # transcript rows, NCBI RefSeq does not (it writes transcript_biotype), so
    # filtering transcript rows on gene_type drops every transcript in a RefSeq
    # GTF and writes an empty tx2gene.tsv. Membership is format-independent and
    # does not depend on gene rows preceding their transcripts.
    tx2gene = [(t, g) for t, g in tx2gene if g in genes]
    n_tx = len(tx2gene)
    if not tx2gene:
        raise SystemExit(
            f'{n_gene} genes parsed but no transcripts belong to any of them. '
            'tx2gene.tsv would be empty, and collapsing Salmon to genes would '
            'then yield nothing. Check that the GTF has "transcript" feature '
            'rows whose gene_id matches the gene rows.')
    return genes, tx2gene, n_gene, n_tx


def write(genes, tx2gene, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    with open(out / 'genes.tsv', 'w') as fh:
        for gid, (c, s, e, tss, strand) in genes.items():
            fh.write(f'{gid}\t{c}\t{s}\t{e}\t{tss}\n')
    with open(out / 'tx2gene.tsv', 'w') as fh:
        for tid, gid in tx2gene:
            fh.write(f'{tid}\t{gid}\n')
    # phASER --features: 0-based half-open, no header, name = gene_id
    with open(out / 'genes.bed', 'w') as fh:
        for gid, (c, s, e, tss, strand) in genes.items():
            fh.write(f'{c}\t{s - 1}\t{e}\t{gid}\n')


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--gtf')
    ap.add_argument('--gene-type', action='append', default=None,
                    help='keep only these gene_type values (repeatable), e.g. '
                         'protein_coding; default keeps all')
    ap.add_argument('--strip-version', action='store_true',
                    help='drop .N from ENSG/ENST IDs (then strip it everywhere)')
    ap.add_argument('--out', default='annot')
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    if not args.gtf:
        raise SystemExit('--gtf is required (or --selftest)')
    genes, t2g, ng, nt = parse_gtf(args.gtf, args.gene_type, args.strip_version)
    write(genes, t2g, args.out)
    chroms = sorted({v[0] for v in genes.values()})
    print(f'{ng} genes, {nt} transcripts -> {args.out}/genes.tsv, '
          f'tx2gene.tsv, genes.bed')
    print(f'chromosome names as in GTF, e.g. {chroms[:3]} -- must match your VCF')
    print(f'IDs {"stripped of" if args.strip_version else "keep"} versions -- '
          f'apply the same convention to Salmon names and --known-egenes')


def selftest():
    import tempfile
    td = Path(tempfile.mkdtemp())
    gtf = td / 'test.gtf'
    gtf.write_text('\n'.join([
        '##description: fabricated GENCODE-style GTF',
        # + strand gene: TSS must be start
        'chr1\tHAVANA\tgene\t1000\t5000\t.\t+\t.\tgene_id "ENSG00000000001.3"; gene_type "protein_coding"; gene_name "PLUS";',
        'chr1\tHAVANA\ttranscript\t1000\t5000\t.\t+\t.\tgene_id "ENSG00000000001.3"; transcript_id "ENST00000000011.2"; gene_type "protein_coding";',
        'chr1\tHAVANA\ttranscript\t1200\t4800\t.\t+\t.\tgene_id "ENSG00000000001.3"; transcript_id "ENST00000000012.1"; gene_type "protein_coding";',
        'chr1\tHAVANA\texon\t1000\t1500\t.\t+\t.\tgene_id "ENSG00000000001.3"; transcript_id "ENST00000000011.2";',
        # - strand gene: TSS must be END
        'chr2\tHAVANA\tgene\t20000\t26000\t.\t-\t.\tgene_id "ENSG00000000002.7"; gene_type "protein_coding"; gene_name "MINUS";',
        'chr2\tHAVANA\ttranscript\t20000\t26000\t.\t-\t.\tgene_id "ENSG00000000002.7"; transcript_id "ENST00000000021.5"; gene_type "protein_coding";',
        # lncRNA: excluded when --gene-type protein_coding
        'chr3\tHAVANA\tgene\t100\t900\t.\t+\t.\tgene_id "ENSG00000000003.1"; gene_type "lncRNA"; gene_name "LNC";',
        'chr3\tHAVANA\ttranscript\t100\t900\t.\t+\t.\tgene_id "ENSG00000000003.1"; transcript_id "ENST00000000031.1"; gene_type "lncRNA";',
    ]) + '\n')
    print('SELF-TEST: parsing a fabricated GENCODE-style GTF\n')
    # the extension is a claim about the bytes, not a fact: parse the same
    # content named both ways round, gzipped and not
    mislabelled = td / 'plain_named_gz.gtf.gz'
    mislabelled.write_text(gtf.read_text())
    really_gz = td / 'gzipped_named_plain.gtf'
    with gzip.open(really_gz, 'wt') as fh:
        fh.write(gtf.read_text())
    for path in (mislabelled, really_gz):
        gm, tm, ngm, ntm = parse_gtf(path)
        assert (ngm, ntm) == (3, 4), (path.name, ngm, ntm)
    # NCBI RefSeq shape: biotype on the gene row only, transcript rows carry
    # transcript_biotype instead. Filtering transcript rows on gene_type would
    # keep the genes and silently drop every transcript.
    refseq = td / 'refseq_style.gtf'
    refseq.write_text('\n'.join([
        'NC_000001.11\tBestRefSeq\tgene\t1000\t5000\t.\t+\t.\tgene_id "OR4F5"; gene_biotype "protein_coding";',
        'NC_000001.11\tBestRefSeq\ttranscript\t1000\t5000\t.\t+\t.\tgene_id "OR4F5"; transcript_id "NM_001005484.2"; transcript_biotype "mRNA";',
        'NC_000001.11\tBestRefSeq\tgene\t9000\t9500\t.\t-\t.\tgene_id "DDX11L1"; gene_biotype "transcribed_pseudogene";',
        'NC_000001.11\tBestRefSeq\ttranscript\t9000\t9500\t.\t-\t.\tgene_id "DDX11L1"; transcript_id "NR_046018.2"; transcript_biotype "transcript";',
    ]) + '\n')
    gr, tr_, ngr, ntr = parse_gtf(refseq, ['protein_coding'])
    assert (ngr, ntr) == (1, 1), (ngr, ntr)
    assert tr_ == [('NM_001005484.2', 'OR4F5')], tr_
    # and a GTF whose transcripts name no kept gene fails loudly, not silently
    orphan = td / 'orphan.gtf'
    orphan.write_text(
        'chr1\tX\tgene\t1\t9\t.\t+\t.\tgene_id "A"; gene_type "protein_coding";\n'
        'chr1\tX\ttranscript\t1\t9\t.\t+\t.\tgene_id "B"; transcript_id "T1";\n')
    try:
        parse_gtf(orphan, ['protein_coding'])
    except SystemExit as e:
        assert 'no transcripts belong' in str(e), e
    else:
        raise AssertionError('empty tx2gene must raise, not return silently')
    # all types, versions kept
    g, t, ng, nt = parse_gtf(gtf)
    assert ng == 3 and nt == 4, (ng, nt)
    assert g['ENSG00000000001.3'] == ('chr1', 1000, 5000, 1000, '+'), g['ENSG00000000001.3']
    assert g['ENSG00000000002.7'] == ('chr2', 20000, 26000, 26000, '-'), \
        f"minus-strand TSS must be END: {g['ENSG00000000002.7']}"
    assert ('ENST00000000012.1', 'ENSG00000000001.3') in t
    # protein_coding filter drops the lncRNA and its transcript
    g2, t2, ng2, nt2 = parse_gtf(gtf, ['protein_coding'])
    assert ng2 == 2 and nt2 == 3 and 'ENSG00000000003.1' not in g2
    # version stripping applies to both IDs consistently
    g3, t3, _, _ = parse_gtf(gtf, None, strip=True)
    assert 'ENSG00000000002' in g3 and ('ENST00000000021', 'ENSG00000000002') in t3
    write(g2, t2, td / 'out')
    lines = (td / 'out' / 'genes.tsv').read_text().strip().split('\n')
    assert len(lines) == 2 and lines[1].split('\t') == ['ENSG00000000002.7', 'chr2', '20000', '26000', '26000']
    # genes.bed, parsed exactly the way phaser_gene_ae.py parses it: positional,
    # from line 1, int() on columns 1 and 2. A header line would raise here.
    bed = (td / 'out' / 'genes.bed').read_text().rstrip('\n').split('\n')
    assert len(bed) == 2, bed
    feats = {}
    for line in bed:
        columns = line.rstrip().split('\t')
        assert len(columns) == 4, columns
        feats[columns[3]] = (columns[0], int(columns[1]), int(columns[2]))
    # 0-based half-open: start is GTF start - 1, stop is GTF end, span preserved
    assert feats['ENSG00000000001.3'] == ('chr1', 999, 5000), feats
    assert feats['ENSG00000000002.7'] == ('chr2', 19999, 26000), feats
    for gid, (c, st, sp) in feats.items():
        assert sp - st == g2[gid][2] - g2[gid][1] + 1, (gid, st, sp)
    # name column is the gene_id, which genes.tsv is keyed by -- not the symbol
    assert set(feats) == set(g2), (set(feats), set(g2))
    assert 'PLUS' not in feats and 'MINUS' not in feats
    print('checks: + strand TSS = start; - strand TSS = end; gene_type filter; '
          'version stripping consistent across gene and transcript IDs; '
          'genes.tsv column order matches --genes; genes.bed is 0-based '
          'half-open, header-free, and keyed by gene_id; compression is '
          'detected from the bytes, not the file extension; --gene-type '
          'filters genes and keeps their transcripts by membership, on '
          'GENCODE and NCBI RefSeq alike; an empty tx2gene raises')
    print('SELF-TEST OK')
    return 0


if __name__ == '__main__':
    main()

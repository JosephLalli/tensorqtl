#!/usr/bin/env python3
"""
Build genes.tsv and tx2gene.tsv from a GENCODE/Ensembl GTF.

These are the two annotation inputs the pipeline needs and that nothing else
produces:

    genes.tsv     gene_id  chr  start  end  tss      (1-based, inclusive)
                  -> --genes for compare_pipelines.py / make_rasqual_inputs.py
    tx2gene.tsv   transcript_id  gene_id
                  -> --tx2gene, to collapse Salmon quant.sf to genes

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
    return gzip.open(p, 'rt') if str(p).endswith('.gz') else open(p)


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
            gt = a.get('gene_type') or a.get('gene_biotype')
            if gene_types and gt not in gene_types:
                continue
            gid = _strip(a.get('gene_id', ''), strip)
            if not gid:
                continue
            if f[2] == 'gene':
                start, end, strand = int(f[3]), int(f[4]), f[6]
                tss = start if strand == '+' else end
                genes[gid] = (f[0], start, end, tss, strand)
                n_gene += 1
            else:
                tid = _strip(a.get('transcript_id', ''), strip)
                if tid:
                    tx2gene.append((tid, gid)); n_tx += 1
    if not genes:
        raise SystemExit('no gene records parsed -- is this a GENCODE/Ensembl GTF '
                         'with "gene" feature rows and gene_id/gene_type attributes?')
    return genes, tx2gene, n_gene, n_tx


def write(genes, tx2gene, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    with open(out / 'genes.tsv', 'w') as fh:
        for gid, (c, s, e, tss, strand) in genes.items():
            fh.write(f'{gid}\t{c}\t{s}\t{e}\t{tss}\n')
    with open(out / 'tx2gene.tsv', 'w') as fh:
        for tid, gid in tx2gene:
            fh.write(f'{tid}\t{gid}\n')


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
    print(f'{ng} genes, {nt} transcripts -> {args.out}/genes.tsv, tx2gene.tsv')
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
    print('checks: + strand TSS = start; - strand TSS = end; gene_type filter; '
          'version stripping consistent across gene and transcript IDs; '
          'genes.tsv column order matches --genes')
    print('SELF-TEST OK')
    return 0


if __name__ == '__main__':
    main()

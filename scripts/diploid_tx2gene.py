#!/usr/bin/env python3
"""
Build tx2gene.tsv from a PERSONALIZED DIPLOID GTF (g2gtools-style).

WHY THIS IS NOT gtf_to_tables.py
================================
A diploid annotation carries the haplotype suffix on BOTH ids:

    gene_id "SEPTIN14P6_L"; transcript_id "NR_109817.1_L"; gene "SEPTIN14P6";

hapmixQTL pairs the two haplotype transcripts first and then looks the gene up
by the UNSUFFIXED base -- load_counts does `t2g.get(base)` where base is
`NR_109817.1`. So a tx2gene built straight off this file, keyed on
`NR_109817.1_L`, matches nothing. The failure is the misleading one: zero pairs
resolve, and the run dies claiming Salmon was run against a standard reference
transcriptome when it was not.

This strips the suffix from both ids and dedupes, so the _L and _R copies of a
gene collapse to one row.

WHY NOT ALSO genes.tsv / genes.bed
==================================
A personalized annotation has personalized COORDINATES -- they are positions in
that sample's own haplotype, not in the reference the VCF is called against.
Gene spans and the TSS must come from the reference annotation via
gtf_to_tables.py. Only the transcript-to-gene map is taken from here, because
only this file knows the diploid transcript names.

Run:
  python3 scripts/diploid_tx2gene.py --selftest
  python3 scripts/diploid_tx2gene.py --gtf SAMPLE-diploid_specific.gtf \\
      --hap-suffix _L,_R --out annot/tx2gene.tsv
"""

import argparse
import gzip
import re
from pathlib import Path

_ATTR = re.compile(r'(\w+) "([^"]*)"')


def _open(p):
    with open(p, 'rb') as fh:
        gz = fh.read(2) == b'\x1f\x8b'
    return gzip.open(p, 'rt') if gz else open(p)


def _strip_suffix(value, suffixes):
    for suf in suffixes:
        if value.endswith(suf):
            return value[:-len(suf)]
    return value


def build(path, suffixes, gene_attr='gene'):
    """Return [(transcript_base, gene_base)], deduped, order preserved."""
    rows, seen = [], set()
    n_rows = n_suffixed = 0
    with _open(path) as fh:
        for line in fh:
            if line.startswith('#'):
                continue
            f = line.rstrip('\n').split('\t')
            if len(f) < 9 or f[2] != 'transcript':
                continue
            a = _ATTR.findall(f[8])
            a = dict(a)
            tid = a.get('transcript_id', '')
            # prefer the unsuffixed symbol the diploid GTF keeps alongside
            gid = a.get(gene_attr) or a.get('gene_id', '')
            if not tid or not gid:
                continue
            n_rows += 1
            base_t = _strip_suffix(tid, suffixes)
            if base_t != tid:
                n_suffixed += 1
            base_g = _strip_suffix(gid, suffixes)
            key = (base_t, base_g)
            if key in seen:
                continue
            seen.add(key)
            rows.append(key)
    if not rows:
        raise SystemExit(
            f'no transcript rows with transcript_id and {gene_attr}/gene_id in '
            f'{path} -- is this a GTF (key "value" attributes) rather than a '
            'GFF3 (key=value)?')
    if not n_suffixed:
        raise SystemExit(
            f'no transcript_id in {path} ends with any of {suffixes}. This '
            'does not look like a diploid annotation; for a reference GTF use '
            'gtf_to_tables.py, which writes tx2gene.tsv along with genes.tsv '
            'and genes.bed.')
    return rows, n_rows, n_suffixed


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--gtf')
    ap.add_argument('--hap-suffix', default='_L,_R',
                    help='comma-separated haplotype suffixes (default _L,_R)')
    ap.add_argument('--gene-attr', default='gene',
                    help='attribute holding the unsuffixed gene name '
                         '(default "gene"; falls back to gene_id)')
    ap.add_argument('--out', default='tx2gene.tsv')
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    if not args.gtf:
        raise SystemExit('--gtf is required (or --selftest)')
    sufs = tuple(s for s in args.hap_suffix.split(',') if s)
    rows, n_rows, n_suf = build(args.gtf, sufs, args.gene_attr)
    out = Path(args.out)
    if out.parent != Path(''):
        out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('\n'.join(f'{t}\t{g}' for t, g in rows) + '\n')
    print(f'{n_rows} transcript rows ({n_suf} haplotype-suffixed) -> '
          f'{len(rows)} unique transcript->gene pairs, '
          f'{len({g for _, g in rows})} genes -> {out}')
    print('keys are UNSUFFIXED transcript ids, which is what load_counts looks '
          'up after pairing')


def selftest():
    import tempfile
    td = Path(tempfile.mkdtemp())
    gtf = td / 'dip.gtf'
    gtf.write_text('\n'.join([
        '#!annotation-source diploid',
        'chr1\tg2g\ttranscript\t100\t900\t.\t+\t.\tgene_id "AAA_L"; transcript_id "NR_1.1_L"; gene "AAA"; transcript_biotype "mRNA";',
        'chr1\tg2g\ttranscript\t105\t910\t.\t+\t.\tgene_id "AAA_R"; transcript_id "NR_1.1_R"; gene "AAA"; transcript_biotype "mRNA";',
        'chr1\tg2g\ttranscript\t200\t800\t.\t-\t.\tgene_id "BBB_L"; transcript_id "NM_2.3_L"; gene "BBB"; transcript_biotype "mRNA";',
        'chr1\tg2g\texon\t200\t800\t.\t-\t.\tgene_id "BBB_L"; transcript_id "NM_2.3_L";',
    ]) + '\n')
    print('SELF-TEST: collapsing a diploid GTF to tx2gene\n')
    rows, n_rows, n_suf = build(gtf, ('_L', '_R'))
    assert n_rows == 3 and n_suf == 3, (n_rows, n_suf)
    # the _L and _R copies of AAA collapse to ONE row keyed on the base id
    assert rows == [('NR_1.1', 'AAA'), ('NM_2.3', 'BBB')], rows
    assert not any(t.endswith(('_L', '_R')) for t, _ in rows)
    assert not any(g.endswith(('_L', '_R')) for _, g in rows)
    # a reference (non-diploid) GTF must be refused, not silently passed through
    ref = td / 'ref.gtf'
    ref.write_text(
        'chr1\tHAVANA\ttranscript\t1\t9\t.\t+\t.\tgene_id "ENSG1"; '
        'transcript_id "ENST1"; gene "SYM";\n')
    try:
        build(ref, ('_L', '_R'))
    except SystemExit as e:
        assert 'does not look like a diploid annotation' in str(e), e
    else:
        raise AssertionError('a reference GTF must be refused')
    # gene_id fallback when the unsuffixed "gene" attribute is absent
    nog = td / 'nogene.gtf'
    nog.write_text('chr1\tg2g\ttranscript\t1\t9\t.\t+\t.\tgene_id "CCC_L"; '
                   'transcript_id "NR_3.1_L";\n')
    rows2, _, _ = build(nog, ('_L', '_R'))
    assert rows2 == [('NR_3.1', 'CCC')], rows2
    print('checks: _L/_R copies collapse to one row; both ids unsuffixed; a '
          'reference GTF is refused; gene_id fallback works')
    print('SELF-TEST OK')
    return 0


if __name__ == '__main__':
    main()

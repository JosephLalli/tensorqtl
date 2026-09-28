#!/usr/bin/env python3
"""Region files for the phASER rerun on the T2T RNA BAMs: a GTEx-style collapsed exon model, per-strand exonic
regions, the HLA blacklist and the CHM13 short-read inaccessible regions.

Collapsed model (GTEx collapse_annotation.py --stranded, on RefSeq RS_2024_08): genes whose description names a
readthrough are dropped (RefSeq annotates readthrough transcripts as their own genes; GTEx drops transcripts tagged
readthrough_transcript; RefSeq has no retained_intron biotype, GTEx's other exclusion, so nothing stands in for it);
each gene's exons are merged; any stretch covered by exons of two or more genes on the same strand is removed.
The per-strand exonic regions restrict each strand's VCF, so phASER on a strand-split BAM counts only variants
that one gene on that strand owns.

phaser_gene_ae assigns a variant to every feature whose span contains it. Features stay the gene spans of
annot/genes.NC.bed, so a gene is also credited with the exonic variants of a same-strand gene inside its span;
the extent of that is measured here (nesting.tsv).

Blacklists, NC contig names: --blacklist = merged spans of the HLA-* genes (phASER: excluded from phasing);
--haplo_count_blacklist = the complement of the CHM13v2.0 combined short-read accessibility mask (phASER: reads
at variants there are not counted). The mask has no chrM, so all of chrM is blacklisted.
"""
import json
import os
import re
import subprocess
from collections import defaultdict
from pathlib import Path

import pandas as pd

D = Path('/mnt/ssd/lalli/brainvar_hapmix_deploy')
GTF = Path('/mnt/ssd/lalli/nf_stage/genome_refs/T2T-CHM13_v2_ncbi110/'
           'GCF_009914755.1_RS_2024_08-T2T-CHM13v2.0_genomic.UCSC_chr.with_GRCh38_rCDS_chrM.exon_ids.gtf')  # scripts/native_counts.py GTF
FAI = Path('/mnt/ssd/lalli/nf_stage/genome_refs/T2T-CHM13_v2_ncbi110/chm13v2.0_maskedY_rCRS.fasta.fai')
MASK = D / 'annot' / 'chm13_accessibility' / 'combined_mask.bed.gz'   # accessible regions, UCSC names (marbl/CHM13)
CHR2NC = D / 'vcf' / 'chr2nc.tsv'
GENES_NC_BED = D / 'annot' / 'genes.NC.bed'             # phaser_gene_ae --features, the pipeline's 58,516 genes
OUT = D / 'phaser_inputs_20260928'
READTHROUGH = re.compile(r'\breadthrough\b')            # all 207 RefSeq readthrough gene descriptions match
ATTR = re.compile(r'(gene_id|description) "([^"]*)"')


def write_atomic(path, df, header=False):
    tmp = path.with_name(path.name + '.tmp')
    df.to_csv(tmp, sep='\t', index=False, header=header)
    os.replace(tmp, path)


def read_gtf():
    """Gene rows (chrom, start0, end, strand, description) and exon rows (chrom, start0, end, strand, gene)."""
    genes, exons = {}, []
    with open(GTF) as fh:
        for line in fh:
            if line[0] == '#':
                continue
            f = line.split('\t', 8)
            if f[2] not in ('gene', 'exon'):
                continue
            a = dict(ATTR.findall(f[8]))
            if f[2] == 'gene':
                if a['gene_id'] in genes:
                    raise SystemExit(f'duplicate gene_id {a["gene_id"]} in {GTF.name}')
                genes[a['gene_id']] = (f[0], int(f[3]) - 1, int(f[4]), f[6], a.get('description', ''))
            else:
                exons.append((f[0], int(f[3]) - 1, int(f[4]), f[6], a['gene_id']))
    genes = pd.DataFrame.from_dict(genes, orient='index', columns=['chrom', 'start', 'end', 'strand', 'description'])
    exons = pd.DataFrame(exons, columns=['chrom', 'start', 'end', 'strand', 'gene'])
    missing = set(exons.gene) - set(genes.index)
    if missing:
        raise SystemExit(f'{len(missing)} exon gene_ids without a gene row, e.g. {sorted(missing)[:3]}')
    print(f'{GTF.name}: {len(genes):,} genes, {len(exons):,} exon rows')
    return genes, exons


def merge_per_gene(exons):
    """Union of each gene's exons: (chrom, start, end, strand, gene), sorted."""
    rows = []
    for (gene, chrom, strand), g in exons.sort_values('start').groupby(['gene', 'chrom', 'strand'], sort=False):
        s0, e0 = None, None
        for s, e in zip(g.start.values, g.end.values):
            if s0 is None:
                s0, e0 = s, e
            elif s <= e0:
                e0 = max(e0, e)
            else:
                rows.append((chrom, s0, e0, strand, gene))
                s0, e0 = s, e
        rows.append((chrom, s0, e0, strand, gene))
    return pd.DataFrame(rows, columns=['chrom', 'start', 'end', 'strand', 'gene'])


def unique_segments(merged):
    """Stretches covered by exactly one gene on their strand; merged intervals of one gene never overlap."""
    out = []
    for (chrom, strand), g in merged.groupby(['chrom', 'strand'], sort=False):
        events = defaultdict(list)
        for s, e, gene in zip(g.start.values, g.end.values, g.gene.values):
            events[s].append((1, gene))
            events[e].append((-1, gene))
        active = defaultdict(int)
        pos = sorted(events)
        for p, q in zip(pos, pos[1:] + [None]):
            for d, gene in events[p]:
                active[gene] += d
                if active[gene] == 0:
                    del active[gene]
            if q is not None and len(active) == 1:
                (gene,) = active
                out.append((chrom, p, q, strand, gene))
    seg = pd.DataFrame(out, columns=['chrom', 'start', 'end', 'strand', 'gene'])
    return seg.sort_values(['chrom', 'start']).reset_index(drop=True)


def main():
    OUT.mkdir(exist_ok=True)
    c2n = dict(pd.read_csv(CHR2NC, sep='\t', header=None).values)
    fai = pd.read_csv(FAI, sep='\t', header=None, usecols=[0, 1], names=['chrom', 'len'])
    if set(fai.chrom) != set(c2n):
        raise SystemExit(f'{FAI.name} contigs differ from {CHR2NC.name}')
    genes, exons = read_gtf()
    exons = exons[exons.chrom.isin(c2n)]

    rt = genes.index[genes.description.str.contains(READTHROUGH)]
    print(f'readthrough genes dropped: {len(rt)} (description matches {READTHROUGH.pattern}), e.g. {list(rt[:3])}')
    merged = merge_per_gene(exons[~exons.gene.isin(set(rt))])
    seg = unique_segments(merged)
    bp_merged = (merged.end - merged.start).groupby(merged.gene).sum()
    bp_unique = (seg.end - seg.start).groupby(seg.gene).sum().reindex(bp_merged.index, fill_value=0)
    print(f'collapsed model: {bp_merged.size:,} genes, {bp_merged.sum():,} exonic bp; same-strand shared stretches '
          f'removed {bp_merged.sum() - bp_unique.sum():,} bp; {int((bp_unique == 0).sum()):,} genes left with no '
          f'exonic sequence')

    pipe = pd.read_csv(GENES_NC_BED, sep='\t', header=None, names=['contig', 'start', 'end', 'gene'])
    absent = pipe.gene[~pipe.gene.isin(genes.index)]
    if len(absent):
        raise SystemExit(f'{len(absent)} {GENES_NC_BED.name} genes not in {GTF.name}, e.g. {list(absent[:3])}')
    pipe['strand'] = genes.strand.reindex(pipe.gene).values
    pipe['readthrough'] = pipe.gene.isin(set(rt))
    pipe['exonic_bp'] = bp_merged.reindex(pipe.gene).fillna(0).astype(int).values
    pipe['unique_exonic_bp'] = bp_unique.reindex(pipe.gene).fillna(0).astype(int).values
    n2c = {v: k for k, v in c2n.items()}

    # Same-strand exonic segments of other genes inside each pipeline gene's span (phaser_gene_ae would credit them).
    seg_nc = seg.assign(contig=seg.chrom.map(c2n))
    nest = []
    for (contig, strand), g in pipe.groupby(['contig', 'strand'], sort=False):
        s = seg_nc[(seg_nc.contig == contig) & (seg_nc.strand == strand)].sort_values('start')
        st, en, gn = s.start.values, s.end.values, s.gene.values
        longest = int((en - st).max())
        for row in g.itertuples():
            lo, hi = st.searchsorted(row.start - longest), st.searchsorted(row.end)
            other = [(max(a, row.start), min(b, row.end), x) for a, b, x in zip(st[lo:hi], en[lo:hi], gn[lo:hi])
                     if x != row.gene and b > row.start and a < row.end]
            nest.append((row.gene, sum(b - a for a, b, _ in other), len({x for *_, x in other})))
    nest = pd.DataFrame(nest, columns=['gene', 'other_gene_unique_exonic_bp_in_span', 'other_genes_in_span'])
    pipe = pipe.merge(nest, on='gene', how='left', validate='1:1')
    has = pipe[pipe.unique_exonic_bp > 0]
    print(f'pipeline genes ({GENES_NC_BED.name}): {len(pipe):,}; readthrough {int(pipe.readthrough.sum()):,}; with unique '
          f'exonic sequence {len(has):,}; of those, span contains another same-strand gene\'s unique exons: '
          f'{int((has.other_genes_in_span > 0).sum()):,} (median {has.other_gene_unique_exonic_bp_in_span[has.other_genes_in_span > 0].median():,.0f} bp, '
          f'against a median own unique exonic {has.unique_exonic_bp.median():,.0f} bp)')
    write_atomic(OUT / 'nesting.tsv', pipe, header=True)

    for strand, tag in (('+', 'plus'), ('-', 'minus')):
        s = seg_nc[seg_nc.strand == strand][['contig', 'start', 'end', 'gene']].sort_values(['contig', 'start'])
        write_atomic(OUT / f'exonic_unique.{tag}.NC.bed', s)
        print(f'exonic_unique.{tag}.NC.bed: {len(s):,} segments, {(s.end - s.start).sum():,} bp')
    write_atomic(OUT / 'collapsed_exons.NC.bed', seg_nc[['contig', 'start', 'end', 'gene', 'strand']]
                 .assign(score=0)[['contig', 'start', 'end', 'gene', 'score', 'strand']])

    hla = genes[genes.index.str.startswith('HLA-')].sort_values(['chrom', 'start'])
    hla_bed = OUT / 'hla.unmerged.NC.bed'
    write_atomic(hla_bed, hla.assign(contig=hla.chrom.map(c2n), gene=hla.index)[['contig', 'start', 'end', 'gene']])
    merged_hla = subprocess.run(['bedtools', 'merge', '-i', str(hla_bed), '-c', '4', '-o', 'distinct'], check=True,
                                stdout=subprocess.PIPE, text=True).stdout
    (OUT / 'hla.NC.bed').write_text(merged_hla)
    hla_bed.unlink()
    rows = [line.split('\t') for line in merged_hla.splitlines()]
    print(f'hla.NC.bed: {len(hla)} HLA-* genes on {sorted(set(hla.chrom))} in {len(rows)} merged intervals, '
          f'{sum(int(r[2]) - int(r[1]) for r in rows):,} bp: '
          + '; '.join(f'{r[0]}:{int(r[1]) + 1}-{r[2]}' for r in rows))

    genome = OUT / 'genome.chr.txt'
    fai.to_csv(genome, sep='\t', index=False, header=False)
    comp = subprocess.run(['bedtools', 'complement', '-i', str(MASK), '-g', str(genome)], check=True,
                          stdout=subprocess.PIPE, text=True).stdout
    genome.unlink()
    bl = pd.DataFrame([line.split('\t') for line in comp.splitlines()], columns=['chrom', 'start', 'end'])
    bl[['start', 'end']] = bl[['start', 'end']].astype(int)
    bl['chrom'] = bl.chrom.map(c2n)
    write_atomic(OUT / 'haplo_count_blacklist.NC.bed', bl)
    per = (bl.end - bl.start).groupby(bl.chrom).sum().reindex(fai.chrom.map(c2n)).fillna(0).astype(int)
    frac = per.values / fai.len.values
    print(f'haplo_count_blacklist.NC.bed: {len(bl):,} intervals, {per.sum():,} bp = {per.sum() / fai.len.sum():.3f} of '
          f'the genome; per contig {frac.min():.3f}-{frac[fai.chrom != "chrM"].max():.3f} (chrM 1.000, not in the mask)')

    facts = dict(gtf=str(GTF), mask=str(MASK), readthrough_genes=int(len(rt)), collapsed_genes=int(bp_merged.size),
                 exonic_bp=int(bp_merged.sum()), unique_exonic_bp=int(bp_unique.sum()),
                 genes_emptied=int((bp_unique == 0).sum()), pipeline_genes=int(len(pipe)),
                 pipeline_readthrough=int(pipe.readthrough.sum()), pipeline_with_unique_exons=int(len(has)),
                 pipeline_with_nested_same_strand=int((has.other_genes_in_span > 0).sum()),
                 hla_genes=sorted(hla.index), hla_intervals=[r[:3] for r in rows],
                 blacklist_intervals=int(len(bl)), blacklist_bp=int(per.sum()),
                 contig_names={k: v for k, v in n2c.items()})
    tmp = OUT / 'facts.json.tmp'
    tmp.write_text(json.dumps(facts, indent=1) + '\n')
    os.replace(tmp, OUT / 'facts.json')
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()

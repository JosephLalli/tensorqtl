"""RASQUAL's native per-SNP inputs on the native-input datasets of 05b_native_arms.py (docs/pipeline_rules.md, decision
2026-10-02). This script does not call RASQUAL; 04c_run_rasqual_native.py runs the commands it writes.

Why: 04 gives RASQUAL one pseudo feature SNP per gene carrying the Salmon haplotype counts, so its read-level model
(per-SNP allelic counts at real exonic heterozygous sites, reference-mapping bias phi, read allele error rate delta,
posterior genotype updating) is bypassed. These inputs give it per-SNP counts from the same STAR alignments the native
arms use, under the benchmark's record permutation, haplotype swap and simulated effects.

Per native dataset (ROOT/native/datasets/<scenario>/repNNN.npz) and gene:
  Y, K, X  04.write_bins on the native dataset: Y = 05b's thinned featureCounts total (the input trecase_native gets),
           K = native effective library size / its mean, X = the RNA-tied covariates moved with the record and the genotype
           PCs in place.
  rSNPs    04.rsnp_text unchanged: every tested variant, the column donor's phased genotype, AS 0,0 (one file per gene,
           shared by every dataset, since rSNP genotypes never move).
  fSNPs    every biallelic SNP of the loader's variant frame inside the gene's exon union (annot/exons.tsv) at which some
           donor is heterozygous. AS = phASER's ref,alt counts in the donor's file for the gene's own strand
           (phaser_stranded_wasp_20260928: strand-split, WASP-filtered, the counts behind 05b's a and b; strand from
           phaser_inputs_20260928/nesting.tsv, contigs through vcf/chr2nc.tsv), 0,0 where the donor has no row. Column i
           carries record perm[i]'s GT:AS entry with its phased GT reversed where swap[i] = -1 and AS left as ref,alt
           (rasqual_read_level.exonic_lines and swap_haplotypes), so the ref count moves to the other haplotype. Then the
           simulated effect: at a heterozygous entry the count on the column's first haplotype is thinned by fL[g, i] and
           the other by fR[g, i] (the Salmon dataset's factors; 02.thin, exact binomial on integers); at a homozygous
           entry both counts by (fL + fR) / 2, 02's rule for the remainder. Stream SeedSequence(SEED, (THIN_KEY, r,
           round(1000 |beta|))), genes then sites in order. Lines sit at pos + FSNP_OFFSET with -s/-e shifted the same,
           and -c TSS -w 2 WIN, so RASQUAL admits them as fSNPs and never scans them as rSNPs
           (rasqual_read_level.py docstring, 'Keeping fSNPs out of the rSNP scan').
  command  rasqual_read_level.native_cmd's option line with 04's -d, -a, -h 0, one thread and --force; stdin is the fSNP
           file then the rSNP file.

Approximations: fragments covering several fSNPs are thinned independently at each; the fSNP
counts and Y are thinned by separate draws; phASER writes rows only at a donor's own heterozygous sites, so homozygotes
are 0,0 where RASQUAL's createASVCF would count their reads (delta loses that information); phASER's read filters
(MAPQ 255, base quality 10, WASP, duplicates dropped) are not createASVCF's; the phase is the analysis VCF's statistical
phase.

Check (known answer): at beta = 0 every factor is 1, so each fSNP line of beta0.0 rep000 must equal the one built by
string operations from the raw phASER entries (perm, then GT reversed where swapped), as rasqual_read_level builds them.

Output OUT = ROOT/native/rasqual_inputs: rsnp/<gene>.txt; <scenario>/repNNN/{Y,K,X}.bin, fsnp/<gene>.txt, commands.tsv
(gene, stdin files, the argument list); facts.json (per gene the fSNP lines, sites covered by the scout's rule, rSNPs,
RASQUAL's (fSNPs + 1) x rSNPs budget and a cost range; per dataset the reads written and removed). OUT/README.md (by
hand) records the checks, numbers and approximations.
"""
import json

import numpy as np
import pandas as pd

import common as C

MD, RR = C.module('02_make_datasets'), C.module('04_run_rasqual')
OUT = C.NATIVE / 'rasqual_inputs'
PHASER = C.D / 'phaser_stranded_wasp_20260928' / 'phaser'      # <donor>.<plus|minus>.allelic_counts.txt, NC contigs
STRAND = C.D / 'phaser_inputs_20260928' / 'nesting.tsv'         # gene strand, as phaser_stranded.py assigns genes to runs
C2N = C.D / 'vcf' / 'chr2nc.tsv'
EXONS = C.D / 'annot' / 'exons.tsv'                              # gene, merged exon starts, ends (1-based)
THIN_KEY = 8                   # spawn key used by no other script here (05b_native_arms.NATIVE_THIN_KEY lists the rest)
FSNP_OFFSET = 1_000_000_000    # rasqual_read_level.FSNP_OFFSET
GTS = np.array(['0|0', '0|1', '1|0', '1|1'])   # index 2 xL + xR, as 04.GT
SEC_PER_UNIT = (0.04, 0.35)    # seconds per admitted fSNP x rSNP at 92 donors: PDE4DIP and ANKRD36, smoke of 2026-09-27 (rasqual_read_level.SEC_PER_UNIT)


def exon_unions(genes):
    ex = {}
    for line in open(EXONS):
        f = line.rstrip('\n').split('\t')
        if f[0] in genes:
            ex[f[0]] = list(zip(map(int, f[1].split(',')), map(int, f[2].split(','))))
    missing = set(genes) - set(ex)
    if missing:
        raise SystemExit(f'{EXONS}: no exons for {sorted(missing)}')
    return ex


def feature_rows(S, ex):
    """Per gene, rows of the loader's variant frame inside its exon union with a heterozygous donor."""
    I, vdf = S['I'], S['I']['vdf']
    het = (I['xL'] != I['xR']).any(1)
    rows = {}
    for g in S['genes']:
        p, on = vdf.pos.values, vdf.chrom.values == S['gp'].loc[g, 'chr']
        inside = np.zeros(len(vdf), bool)
        for a, b in ex[g]:
            inside |= on & (p >= a) & (p <= b)
        rows[g] = np.where(inside & het)[0]
        if not (np.isin(I['xL'][rows[g]], (0, 1)).all() and np.isin(I['xR'][rows[g]], (0, 1)).all()):
            raise SystemExit(f'{g}: an exonic site has phased alleles outside 0 and 1')
    return rows


def phaser_counts(S, rows):
    """Per gene, ref and alt [sites, donors in loader order] from each donor's own-strand phASER file; per gene the
    phASER rows matched at a heterozygous donor and at a homozygous one."""
    I, vdf = S['I'], S['I']['vdf']
    strand = pd.read_csv(STRAND, sep='\t', usecols=['gene', 'strand']).drop_duplicates().set_index('gene').strand
    if not strand.index.is_unique or not set(S['genes']) <= set(strand.index):
        raise SystemExit(f'{STRAND}: a benchmark gene is missing or has two strands')
    run = strand.loc[S['genes']].map({'+': 'plus', '-': 'minus'})
    n2c = pd.read_csv(C2N, sep='\t', header=None, index_col=1).iloc[:, 0]
    ref = {g: np.zeros((len(r), len(S['order'])), np.int64) for g, r in rows.items()}
    alt = {g: np.zeros_like(ref[g]) for g in rows}
    keys = {g: pd.MultiIndex.from_arrays([vdf.chrom.values[r], vdf.pos.values[r], vdf.ref.values[r], vdf.alt.values[r]])
            for g, r in rows.items()}
    matched = {g: [0, 0] for g in rows}
    for j, d in enumerate(S['order']):
        for s in ('plus', 'minus'):
            a = pd.read_csv(PHASER / f'{d}.{s}.allelic_counts.txt', sep='\t',
                            usecols=['contig', 'position', 'refAllele', 'altAllele', 'refCount', 'altCount'])
            a['contig'] = a.contig.map(n2c)
            if a.contig.isna().any():
                raise SystemExit(f'{d}.{s}: contig missing from {C2N}')
            a = a.set_index(['contig', 'position', 'refAllele', 'altAllele'])
            if not a.index.is_unique:
                raise SystemExit(f'{d}.{s}: a variant appears twice')
            for g in run.index[run == s]:
                hit = keys[g].isin(a.index)
                c = a.loc[keys[g][hit]]
                ref[g][hit, j], alt[g][hit, j] = c.refCount.values, c.altCount.values
                h = I['xL'][rows[g][hit], j] != I['xR'][rows[g][hit], j]
                matched[g][0] += int(h.sum())
                matched[g][1] += int((~h).sum())
    return ref, alt, matched, run


def covered(xL, xR, ref, alt):
    """Sites with a read at a heterozygous donor and both alleles seen across them (the scout's rule: necessary for
    RASQUAL's admission, not sufficient)."""
    h = xL != xR
    r, a = (ref * h).sum(1), (alt * h).sum(1)
    return int(((r + a > 0) & (r > 0) & (a > 0)).sum())


def fsnp_lines(S, g, rows, ref, alt, perm, swap, fL, fR, rng):
    """The gene's fSNP lines for one dataset (docstring) and the reads before and after thinning."""
    I, vdf = S['I'], S['I']['vdf']
    xL, xR = I['xL'][rows][:, perm], I['xR'][rows][:, perm]
    s = swap < 0
    xL, xR = np.where(s, xR, xL), np.where(s, xL, xR)
    r, a = ref[:, perm], alt[:, perm]
    h = xL != xR
    first = np.where(xL == 0, r, a)    # the count on the column's first haplotype at a heterozygous entry
    second = np.where(xL == 0, a, r)
    f1, f2 = MD.thin(first, fL, rng), MD.thin(second, fR, rng)
    fm = (fL + fR) / 2
    r2, a2 = MD.thin(r, fm, rng), MD.thin(a, fm, rng)
    r2 = np.where(h, np.where(xL == 0, f1, f2), r2)
    a2 = np.where(h, np.where(xL == 0, f2, f1), a2)
    if not (np.array_equal(r2, np.rint(r2)) and np.array_equal(a2, np.rint(a2))):
        raise SystemExit(f'{g}: thinned per-SNP counts are not integers')
    r2, a2 = r2.astype(np.int64), a2.astype(np.int64)
    gt = GTS[xL * 2 + xR]
    v = vdf.iloc[rows]
    text = ''.join(f'{c}\t{p + FSNP_OFFSET}\t{i}\t{rf}\t{al}\t.\tPASS\t.\tGT:AS\t'
                   + '\t'.join(f'{x}:{y},{z}' for x, y, z in zip(gg, rr, aa)) + '\n'
                   for c, p, i, rf, al, gg, rr, aa in zip(v.chrom, v.pos, v.index, v.ref, v.alt, gt, r2, a2))
    return text, int(r.sum() + a.sum()), int(r2.sum() + a2.sum())


def string_route(S, g, rows, ref, alt, perm, swap):
    """The same lines with no thinning, built as rasqual_read_level.exonic_lines and swap_haplotypes build them."""
    I, vdf = S['I'], S['I']['vdf']
    out = []
    for k, j in enumerate(rows):
        e = [f'{gt}:{x},{y}' for gt, x, y in zip(GTS[I['xL'][j] * 2 + I['xR'][j]], ref[k], alt[k])]
        e = [e[p] for p in perm]
        e = [x[2::-1] + x[3:] if w < 0 else x for x, w in zip(e, swap)]
        out.append(f'{vdf.chrom.iloc[j]}\t{vdf.pos.iloc[j] + FSNP_OFFSET}\t{vdf.index[j]}\t{vdf.ref.iloc[j]}\t'
                   f'{vdf.alt.iloc[j]}\t.\tPASS\t.\tGT:AS\t' + '\t'.join(e) + '\n')
    return ''.join(out)


def command(S, g, ex, bins, n_lines, n_fsnp):
    """rasqual_read_level.native_cmd's option line, one thread."""
    starts = ','.join(str(a + FSNP_OFFSET) for a, _ in ex[g])
    ends = ','.join(str(b + FSNP_OFFSET) for _, b in ex[g])
    return [RR.RASQUAL, '-y', bins['Y'], '-k', bins['K'], '-n', str(len(S['order'])), '-j', str(S['genes'].index(g) + 1),
            '-l', str(n_lines), '-m', str(n_fsnp), '-s', starts, '-e', ends, '-c', str(int(S['gp'].loc[g, 'pos'])),
            '-w', str(2 * C.WIN), '-f', g, '-z', '-d', str(RR.MIN_COVERAGE), '-a', str(RR.MAF), '-h', str(RR.HWE_P),
            '-x', bins['X'], '--n-threads', '1', '--force']


def main():
    I, _, _ = C.load()
    S = C.setup(I)
    ex = exon_unions(set(S['genes']))
    rows = feature_rows(S, ex)
    ref, alt, matched, run = phaser_counts(S, rows)
    genes = {}
    (OUT / 'rsnp').mkdir(parents=True, exist_ok=True)
    for g in S['genes']:
        text = RR.rsnp_text(S, g)
        C.write_atomic(OUT / 'rsnp' / f'{g}.txt', lambda fh, t=text: fh.write(t), 'w')
        cov = covered(I['xL'][rows[g]], I['xR'][rows[g]], ref[g], alt[g])
        n_r = int(S['n_tested'][g])
        genes[g] = dict(strand_run=run[g], fsnp_lines=len(rows[g]), fsnp_covered=cov, rsnp=n_r,
                        budget=(len(rows[g]) + 1) * n_r, phaser_rows_het=matched[g][0], phaser_rows_hom=matched[g][1],
                        cpu_hours=[cov * n_r * s / 3600 for s in SEC_PER_UNIT])
    G = pd.DataFrame(genes).T
    print(f'{len(G)} genes: fSNP lines {G.fsnp_lines.sum():,} (per gene median {G.fsnp_lines.median():g}, '
          f'{(G.fsnp_lines == 0).sum()} genes with none), covered by the scout rule {G.fsnp_covered.sum():,} '
          f'({(G.fsnp_covered == 0).sum()} genes with none); phASER rows matched at heterozygous donors '
          f'{G.phaser_rows_het.sum():,}, at homozygous donors {G.phaser_rows_hom.sum():,}; rSNPs {G.rsnp.sum():,}', flush=True)
    meta = json.loads((C.DATASETS / 'meta.json').read_text())
    data = {}
    for sc, r in C.runs(meta):
        nd, sd = C.load_dataset(C.NATIVE_DATASETS, sc, r), C.load_dataset(C.DATASETS, sc, r)
        if not (np.array_equal(nd['perm'], sd['perm']) and np.array_equal(nd['swap'], sd['swap'])):
            raise SystemExit(f'{sc} rep {r}: native and Salmon datasets differ in perm or swap')
        d = OUT / sc / f'rep{r:03d}'
        (d / 'fsnp').mkdir(parents=True, exist_ok=True)
        bins, _ = RR.write_bins(S, nd, d)
        rng = np.random.default_rng(np.random.SeedSequence(C.SEED, spawn_key=(THIN_KEY, r, int(round(1000 * float(sc[4:]))))))
        cmds, before, after = [], 0, 0
        for k, g in enumerate(S['genes']):
            text, b, a = fsnp_lines(S, g, rows[g], ref[g], alt[g], nd['perm'], nd['swap'], sd['fL'][k], sd['fR'][k], rng)
            if sc == 'beta0.0' and r == 0 and text != string_route(S, g, rows[g], ref[g], alt[g], nd['perm'], nd['swap']):
                raise SystemExit(f'{g}: fSNP lines at beta = 0 differ from the string route (perm, swap or counts)')
            C.write_atomic(d / 'fsnp' / f'{g}.txt', lambda fh, t=text: fh.write(t), 'w')
            before, after = before + b, after + a
            n_f = len(rows[g])
            cmds.append(dict(gene=g, stdin=f'{d / "fsnp" / g}.txt {OUT / "rsnp" / g}.txt',
                             args=' '.join(command(S, g, ex, bins, n_f + int(S['n_tested'][g]), n_f))))
        C.write_atomic(d / 'commands.tsv', lambda fh, c=cmds: pd.DataFrame(c).to_csv(fh, sep='\t', index=False), 'w')
        data[f'{sc} rep {r:03d}'] = dict(fsnp_reads_raw=before, fsnp_reads_written=after)
        print(f'{sc} rep {r:03d}: fSNP reads {before:,} -> {after:,} after thinning; {len(cmds)} commands', flush=True)
    print('known answer: beta0.0 rep000 fSNP lines equal the string route for every gene', flush=True)
    lo, hi = G.cpu_hours.map(lambda x: x[0]).sum(), G.cpu_hours.map(lambda x: x[1]).sum()
    facts = dict(genes=genes, datasets=data, cpu_hours_per_dataset=[lo, hi], n_datasets=len(data),
                 sec_per_admitted_fsnp_x_rsnp=list(SEC_PER_UNIT))
    C.write_json(OUT / 'facts.json', facts)
    print(f'cost if run as written, admitted fSNPs taken as the covered ones: {lo:,.0f} to {hi:,.0f} CPU hours per dataset, '
          f'{len(data)} datasets; wrote {OUT}', flush=True)


if __name__ == '__main__':
    main()

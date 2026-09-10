#!/usr/bin/env python3
"""
Assemble per-sample phASER gene output into the haplotype matrix hapmixQTL
consumes -- so ONE phASER run feeds both hapmixQTL and RASQUAL.

WHY PHASER, NOT A CUSTOM PILEUP
===============================
phASER (Castel et al. 2016) is the right source of allele-specific counts for
both methods, and strictly better than a per-site pileup:

  * it de-duplicates overlapping mate pairs (a pileup counts them twice);
  * it read-back phases het sites spanned by the same fragment and anchors
    them to the population phasing, correcting switch errors;
  * it filters low-mappability and simulation-biased sites -- the precondition
    docs/ase_validation.md sec 7i shows hapmixQTL depends on.

Its outputs serve both tools:
  <prefix>.allelic_counts.txt     per-variant ref/alt   -> RASQUAL (native)
  <prefix>.gene_ae.txt            per-gene aCount/bCount -> hapmixQTL (this script)

Same reads, same filters, same phasing on both sides: the fairest comparison.

THE gw_phased REQUIREMENT -- READ THIS
======================================
hapmixQTL's ASE channel regresses the allelic contrast a = log(yL/yR) on the
signed het indicator s = xL - xR taken from the VCF. That only means anything if
"haplotype A" in phASER is the SAME haplotype as the VCF's first allele, gene
after gene, sample after sample.

phaser_gene_ae reports `gw_phased`: 1 when the gene's A/B labelling is anchored
to the genome-wide (population) phasing, 0 when it is only locally phased and
the A/B assignment is arbitrary for that gene. For gw_phased = 0, a is correct
in MAGNITUDE but its SIGN is a coin flip relative to s -- feeding it to the ASE
channel scrambles the effect direction and attenuates the estimate exactly as
random phase error does (docs sec 7f: power lost, calibration intact).

So by default this script keeps only gw_phased = 1 gene-samples for the ASE
channel and blanks the rest (0|0). --keep-unphased overrides, for total-channel
use only. GTEx's own matrices are distributed as "gw_phased" for this reason.

WHAT hapmixQTL LOSES ON THIS PATH -- BE CLEAR ABOUT IT
======================================================
hapmixQTL's distinctive feature is propagating a QUANTIFIER's posterior
(Salmon Gibbs draws), which carries multi-mapping and isoform ambiguity. phASER
counts are read-level at het sites and carry none of that. On this path the
inferential variance is emulated from the counts themselves (allelic-assignment
uncertainty conditional on the observed total), which is what the sec 7d
real-data validation used and what tau_mode='estimate' was shown to handle.
That makes "hapmixQTL on phASER" a mixQTL-style analysis: legitimate, validated,
but without the headline feature. The informative design is therefore 2x2 --
each method on its native input AND each on the other's -- so method and input
effects separate. This script supplies the phASER arm for hapmixQTL.

OUTPUT
======
  <out>/phaser_matrix.gw_phased.txt.gz   GTEx-format matrix:
      #contig  name  start  stop  <sample...>   with "aCount|bCount" cells
  <out>/allelic_counts_manifest.tsv      sample <TAB> allelic_counts.txt, for
                                          make_rasqual_inputs.py --allelic-counts
  <out>/samples.txt                      sample order, for --samples

The matrix is exactly what scripts/prep_brainvar.py and the sec 7d harness
consume, so the downstream hapmixQTL path is unchanged and already validated.

Manifest:  sample_id <TAB> phASER output prefix   (files <prefix>.gene_ae.txt
           and <prefix>.allelic_counts.txt must exist)

Run:
  python3 scripts/phaser_to_matrix.py --selftest
  python3 scripts/phaser_to_matrix.py --manifest phaser.tsv --out prepped/
"""

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _open(p):
    return gzip.open(p, 'rt') if str(p).endswith('.gz') else open(p)


REQUIRED = ('contig', 'start', 'stop', 'name', 'aCount', 'bCount', 'gw_phased')


def read_gene_ae(path):
    """One sample's phaser_gene_ae output -> DataFrame indexed by gene name."""
    df = pd.read_csv(path, sep='\t', dtype={'contig': str})
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise SystemExit(f'{path} lacks {missing}; is it phaser_gene_ae output? '
                         f'Columns: {list(df.columns)[:10]}')
    df['name'] = df['name'].astype(str)
    return df.set_index('name')


def assemble(manifest, keep_unphased=False):
    rows = [l.split('\t') for l in Path(manifest).read_text().strip().split('\n')
            if l.strip() and not l.startswith('#')]
    samples = [r[0].strip() for r in rows]
    prefixes = [r[1].strip() for r in rows]
    per = {}
    genes_meta = {}
    n_unphased = 0
    for s, pre in zip(samples, prefixes):
        gae = Path(f'{pre}.gene_ae.txt')
        if not gae.exists():
            raise SystemExit(f'{gae} not found (from manifest prefix {pre})')
        df = read_gene_ae(gae)
        if not keep_unphased:
            n_unphased += int((df['gw_phased'] != 1).sum())
            df = df[df['gw_phased'] == 1]
        per[s] = df[['aCount', 'bCount']]
        for g, r in df.iterrows():
            genes_meta.setdefault(g, (str(r['contig']), int(r['start']), int(r['stop'])))
    genes = sorted(genes_meta, key=lambda g: (genes_meta[g][0], genes_meta[g][1]))
    A = np.zeros((len(genes), len(samples)), np.int64)
    B = np.zeros((len(genes), len(samples)), np.int64)
    gi = {g: i for i, g in enumerate(genes)}
    for k, s in enumerate(samples):
        d = per[s]
        idx = [gi[g] for g in d.index]
        A[idx, k] = d['aCount'].values.astype(np.int64)
        B[idx, k] = d['bCount'].values.astype(np.int64)
    return samples, prefixes, genes, genes_meta, A, B, n_unphased


def write_matrix(path, samples, genes, meta, A, B):
    with gzip.open(path, 'wt') as fh:
        fh.write('#contig\tname\tstart\tstop\t' + '\t'.join(samples) + '\n')
        for i, g in enumerate(genes):
            c, st, sp = meta[g]
            cells = '\t'.join(f'{A[i, k]}|{B[i, k]}' for k in range(len(samples)))
            fh.write(f'{c}\t{g}\t{st}\t{sp}\t{cells}\n')


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--manifest', help='sample_id <TAB> phASER output prefix')
    ap.add_argument('--out', default='phaser_prepped')
    ap.add_argument('--keep-unphased', action='store_true',
                    help='keep gw_phased=0 gene-samples (sign is arbitrary; '
                         'NOT for the ASE channel)')
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    if not args.manifest:
        raise SystemExit('--manifest is required (or --selftest)')
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    samples, prefixes, genes, meta, A, B, n_unph = assemble(
        args.manifest, args.keep_unphased)
    print(f'{len(samples)} samples, {len(genes)} genes')
    if not args.keep_unphased:
        print(f'  dropped {n_unph} gene-samples with gw_phased != 1 '
              f'(A/B labelling not anchored to the VCF haplotypes; their sign '
              f'would be a coin flip in the ASE channel)')
    mpath = out / 'phaser_matrix.gw_phased.txt.gz'
    write_matrix(mpath, samples, genes, meta, A, B)
    (out / 'allelic_counts_manifest.tsv').write_text(
        '\n'.join(f'{s}\t{p}.allelic_counts.txt' for s, p in zip(samples, prefixes)))
    (out / 'samples.txt').write_text('\n'.join(samples))
    tot = A + B
    (out / 'summary.json').write_text(json.dumps({
        'n_samples': len(samples), 'n_genes': len(genes),
        'gene_samples_dropped_unphased': n_unph,
        'median_as_depth_covered': float(np.median(tot[tot > 0])) if (tot > 0).any() else 0,
        'frac_gene_samples_covered': float((tot > 0).mean())}, indent=2))
    print(f'wrote {mpath.name}                -> hapmixQTL (prep_brainvar.py / '
          f'run pipeline)')
    print(f'wrote allelic_counts_manifest.tsv   -> make_rasqual_inputs.py '
          f'--allelic-counts')
    print(f'wrote samples.txt')


def selftest():
    import tempfile
    td = Path(tempfile.mkdtemp()); rng = np.random.RandomState(0)
    N, G = 8, 6
    samples = [f'S{i}' for i in range(N)]
    man = []
    truth = {}
    for s in samples:
        pre = td / s
        with open(f'{pre}.gene_ae.txt', 'w') as fh:
            fh.write('contig\tstart\tstop\tname\taCount\tbCount\ttotalCount\t'
                     'log2_aFC\tn_variants\tvariants\tgw_phased\tbam\n')
            for g in range(G):
                a, b = int(rng.poisson(30)), int(rng.poisson(30))
                gw = 0 if (g == 2) else 1          # gene 2 is never gw-phased
                truth[(s, f'G{g}')] = (a, b, gw)
                fh.write(f'1\t{1000*g}\t{1000*g+500}\tG{g}\t{a}\t{b}\t{a+b}\t0\t3\t'
                         f'v\t{gw}\t{s}.bam\n')
        Path(f'{pre}.allelic_counts.txt').write_text('contig\tstart\tstop\n')
        man.append(f'{s}\t{pre}')
    (td / 'man.tsv').write_text('\n'.join(man))
    print('SELF-TEST: assembling fabricated phaser_gene_ae outputs\n')
    main(['--manifest', str(td / 'man.tsv'), '--out', str(td / 'out')])
    # verify: gw_phased genes round-trip exactly; gw_phased=0 gene is blanked
    with gzip.open(td / 'out' / 'phaser_matrix.gw_phased.txt.gz', 'rt') as fh:
        hdr = fh.readline().rstrip('\n').split('\t')
        assert hdr[4:] == samples
        seen = set()
        for line in fh:
            f = line.rstrip('\n').split('\t'); g = f[1]; seen.add(g)
            for k, s in enumerate(samples):
                a, b = map(int, f[4 + k].split('|'))
                ta, tb, gw = truth[(s, g)]
                exp = (ta, tb) if gw == 1 else (0, 0)
                assert (a, b) == exp, (s, g, (a, b), exp)
    assert 'G2' not in seen, 'a never-phased gene should not appear at all'
    assert len(seen) == G - 1
    print('\nchecks: gw_phased=1 counts round-trip exactly; the gw_phased=0 gene '
          'is excluded; sample order preserved; allelic_counts manifest written')
    print('SELF-TEST OK')
    return 0


if __name__ == '__main__':
    main()

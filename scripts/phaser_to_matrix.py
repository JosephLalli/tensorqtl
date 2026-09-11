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
  <out>/rephased.vcf.gz   (with --vcf)   the original multi-sample VCF with
                                          phASER's read-backed phase overlaid
                                          -- feed this as --vcf to BOTH tools

The matrix is exactly what scripts/prep_brainvar.py and the sec 7d harness
consume, so the downstream hapmixQTL path is unchanged and already validated.

WHY rephased.vcf.gz MUST FEED BOTH METHODS
==========================================
phASER's gene-level A/B counts are anchored to phASER's CORRECTED phase. If
hapmixQTL took s = xL - xR from the original VCF while using those counts,
every site phASER flipped would have s and a mismatched -- a correctness bug,
not a fairness quibble. And RASQUAL's own switch correction is effect-driven:
it detects a switch only as allelic-imbalance inconsistency across fSNPs in an
individual het for a large-effect rSNP (the paper reports 0.02% of RNA-seq
features), so under the null or for small effects it inherits the population
phase uncorrected. Read-backed phase is effect-independent. summary.json
records how many genotypes were re-phased and how many were flipped -- the
switch-error rate phASER found in your data.

Manifest:  sample_id <TAB> phASER output prefix   (files <prefix>.gene_ae.txt,
           <prefix>.allelic_counts.txt, and with --vcf <prefix>.vcf[.gz] from
           phASER --write_vcf 1)

Run:
  python3 scripts/phaser_to_matrix.py --selftest
  python3 scripts/phaser_to_matrix.py --manifest phaser.tsv \\
      --vcf population_phased.vcf.gz --out prepped/
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



# ---------------------------------------------------------------------------
#  Re-phased genotypes: phASER's read-backed phase, feeding BOTH methods
# ---------------------------------------------------------------------------

def _phase_rows(vcf_path, contig=None):
    """Yield (chrom, pos, ref, alt, gt) for one sample's phASER VCF.

    Uses the tabix index phASER writes alongside its VCF when a contig is
    asked for, so a per-contig pass reads only that contig. Falls back to a
    filtered full scan when there is no index or no pysam.
    """
    if contig is not None:
        tbi = Path(str(vcf_path) + '.tbi')
        if tbi.exists():
            try:
                import pysam
                with pysam.TabixFile(str(vcf_path)) as tf:
                    if contig in tf.contigs:
                        for line in tf.fetch(contig):
                            f = line.rstrip('\n').split('\t')
                            if len(f) >= 10:
                                yield f
                    return
            except Exception:
                pass                      # fall through to the plain scan
    with _open(vcf_path) as fh:
        for line in fh:
            if line.startswith('#'):
                continue
            f = line.rstrip('\n').split('\t')
            if len(f) < 10:
                continue
            if contig is not None and f[0] != contig:
                continue
            yield f


def load_sample_phase(vcf_path, contig=None):
    """{(chrom,pos,ref,alt): 'a|b'} for one sample's phASER re-phased VCF.

    Pass `contig` to load only that contig. Loading the whole genome for every
    sample at once does not fit: at roughly 2.5M phased het sites per sample,
    92 samples is ~230M dict entries keyed by a tuple, which is tens of GB
    before Python's per-object overhead. That is what this used to do, and it
    was killed by the OOM reaper partway through a 92-sample cohort, after the
    matrices had been written and with nothing in the log, because the process
    died before its buffered stdout was flushed.
    """
    out = {}
    for f in _phase_rows(vcf_path, contig):
        fmt = f[8].split(':')
        gi = fmt.index('GT') if 'GT' in fmt else 0
        gt = f[9].split(':')[gi]
        if '|' in gt and '.' not in gt:
            out[(str(f[0]), int(f[1]), f[3], f[4])] = gt
    return out


def rephase_vcf(original, samples, prefixes, out_path):
    """Overwrite each sample's GT with phASER's read-backed phase where it
    exists; keep the population phase elsewhere. Preserves the variant set.

    WHY THIS MUST FEED BOTH METHODS. phASER's gene-level A/B counts are anchored
    to phASER's CORRECTED phase. If hapmixQTL took s = xL - xR from the original
    VCF while using those counts, every site phASER flipped would have s and a
    mismatched. And RASQUAL's own switch correction is effect-driven -- it only
    sees a switch as allelic-imbalance inconsistency across fSNPs in individuals
    het for a large-effect rSNP (paper: detected in 0.02% of RNA-seq features)
    -- so under the null or for small effects it inherits the population phase
    uncorrected. Read-backed phase is effect-independent. Giving it to both
    methods is a correctness requirement for hapmixQTL and removes a phasing
    asymmetry against RASQUAL.

    Returns (n_sites_rephased, n_sites_flipped): flipped = phASER's phase
    disagreed with the population phase, i.e. a switch error it corrected.
    """
    paths = {}
    for s, pre in zip(samples, prefixes):
        vp = next((Path(f'{pre}{ext}') for ext in ('.vcf.gz', '.vcf')
                   if Path(f'{pre}{ext}').exists()), None)
        if vp is None:
            raise SystemExit(f'no phASER VCF for {s} at {pre}.vcf[.gz]; run '
                             'phASER with --write_vcf 1')
        paths[s] = vp
    # Phase is loaded one contig at a time. Holding every sample's whole genome
    # at once is tens of GB and gets the process OOM-killed on a real cohort;
    # the population VCF is contig-sorted, so a contig's worth is all that is
    # ever needed. `done` guards against reloading if a contig reappears.
    per, cur, done_contigs = {}, None, set()
    n_re = n_flip = 0
    with _open(original) as fh, gzip.open(out_path, 'wt') as oh:
        col = None
        for line in fh:
            if line.startswith('##'):
                oh.write(line); continue
            f = line.rstrip('\n').split('\t')
            if not line.startswith('#CHROM') and f[0] != cur:
                if f[0] in done_contigs:
                    raise SystemExit(
                        f'contig {f[0]} reappears after another contig; the '
                        'input VCF must be sorted by contig for the per-contig '
                        'phase load to be correct')
                if cur is not None:
                    done_contigs.add(cur)
                cur = f[0]
                per = {x: load_sample_phase(paths[x], cur) for x in samples}
                print(f'  rephase: {cur} ({sum(len(v) for v in per.values())} '
                      'phased genotypes loaded)', flush=True)
            if line.startswith('#CHROM'):
                vs = f[9:]
                missing = [x for x in samples if x not in vs]
                if missing:
                    raise SystemExit(f'samples not in original VCF: {missing[:3]}')
                col = {x: 9 + vs.index(x) for x in samples}
                oh.write(line); continue
            key = (str(f[0]), int(f[1]), f[3], f[4])
            fmt = f[8].split(':')
            gi = fmt.index('GT') if 'GT' in fmt else 0
            for x in samples:
                gt = per[x].get(key)
                if gt is None:
                    continue
                parts = f[col[x]].split(':')
                if parts[gi] != gt:
                    n_flip += 1
                parts[gi] = gt
                f[col[x]] = ':'.join(parts)
                n_re += 1
            oh.write('\t'.join(f) + '\n')
    return n_re, n_flip


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--manifest', help='sample_id <TAB> phASER output prefix')
    ap.add_argument('--out', default='phaser_prepped')
    ap.add_argument('--vcf', default=None,
                    help='original multi-sample phased VCF; if given, emits '
                         'rephased.vcf.gz with phASER read-backed phase '
                         'overlaid (feed this to BOTH methods)')
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
    rephase_stats = None
    if args.vcf:
        n_re, n_flip = rephase_vcf(args.vcf, samples, prefixes,
                                   out / 'rephased.vcf.gz')
        rephase_stats = {'gt_rephased': n_re, 'gt_flipped': n_flip,
                         'flip_rate': (n_flip / n_re) if n_re else None}
        print(f'wrote rephased.vcf.gz               -> --vcf for BOTH '
              f'make_rasqual_inputs.py and the hapmixQTL runner')
        print(f'  {n_re} genotypes re-phased by read evidence; {n_flip} '
              f'flipped vs the population phase '
              f'(switch errors corrected, rate '
              f'{(n_flip / n_re if n_re else 0):.4%})')
    tot = A + B
    (out / 'summary.json').write_text(json.dumps({
        'n_samples': len(samples), 'n_genes': len(genes),
        'gene_samples_dropped_unphased': n_unph,
        'median_as_depth_covered': float(np.median(tot[tot > 0])) if (tot > 0).any() else 0,
        'frac_gene_samples_covered': float((tot > 0).mean()),
        'rephase': rephase_stats}, indent=2))
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
    # original population-phased VCF + per-sample phASER VCFs, some sites
    # re-phased and a known subset of those FLIPPED relative to the original
    V = 20
    pop = {}   # (site, sample) -> original GT
    with open(td / 'orig.vcf', 'w') as fh:
        fh.write('##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER'
                 '\tINFO\tFORMAT\t' + '\t'.join(samples) + '\n')
        for v in range(V):
            gts = []
            for s_ in samples:
                a, b = int(rng.rand() < .5), int(rng.rand() < .5)
                pop[(v, s_)] = f'{a}|{b}'; gts.append(f'{a}|{b}:99')
            fh.write(f'1\t{100*v+1}\tv{v}\tA\tG\t.\tPASS\t.\tGT:GQ\t'
                     + '\t'.join(gts) + '\n')
    exp_flip = 0; exp_re = 0
    expect = {}
    for s_ in samples:
        with open(td / f'{s_}.vcf', 'w') as fh:
            fh.write('##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL'
                     '\tFILTER\tINFO\tFORMAT\t' + s_ + '\n')
            for v in range(V):
                if v % 2:                       # phASER covered even sites only
                    continue
                a, b = pop[(v, s_)].split('|')
                gt = f'{b}|{a}' if (a != b and v % 4 == 0) else f'{a}|{b}'
                if gt != pop[(v, s_)]:
                    exp_flip += 1
                exp_re += 1; expect[(v, s_)] = gt
                fh.write(f'1\t{100*v+1}\tv{v}\tA\tG\t.\tPASS\t.\tGT\t{gt}\n')
    print('SELF-TEST: assembling fabricated phaser_gene_ae outputs\n')
    main(['--manifest', str(td / 'man.tsv'), '--out', str(td / 'out'),
          '--vcf', str(td / 'orig.vcf')])
    # verify the re-phased VCF: phASER phase where covered, original elsewhere,
    # other FORMAT subfields untouched, flip count exact
    with gzip.open(td / 'out' / 'rephased.vcf.gz', 'rt') as fh:
        for line in fh:
            if line.startswith('#'):
                continue
            f = line.rstrip('\n').split('\t'); v = int(f[2][1:])
            for k, s_ in enumerate(samples):
                gt, gq = f[9 + k].split(':')
                assert gq == '99', 'non-GT FORMAT subfield was clobbered'
                want = expect.get((v, s_), pop[(v, s_)])
                assert gt == want, (v, s_, gt, want)
    summ = json.loads((td / 'out' / 'summary.json').read_text())['rephase']
    assert summ['gt_rephased'] == exp_re and summ['gt_flipped'] == exp_flip, summ
    print(f'rephase: {exp_re} genotypes overlaid, {exp_flip} flips detected -- '
          'exact; uncovered sites keep population phase; GQ preserved')
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

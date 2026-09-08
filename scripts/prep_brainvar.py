#!/usr/bin/env python3
"""
Prepare fetal (prenatal) BrainVar samples for hapmixQTL.

BrainVar (Walker et al. 2019, Cell) is dorsolateral prefrontal cortex RNA-seq +
WGS spanning ~6 post-conceptual weeks to ~20 years, dbGaP phs001900. The prenatal
subset is the target here.

THIS SCRIPT DOES NOT CONTAIN OR FETCH ANY BRAINVAR DATA. It is the ingest
pipeline: point it at your authorized local copy and it produces
hapmixQTL-ready inputs, applying the preconditions this repo's validation work
established are load-bearing.

WHAT YOU NEED
-------------
  --phaser       phASER haplotype-expression matrix, the format GTEx publishes:
                 a TSV whose first four columns are contig/name/start/stop and
                 whose remaining columns are one sample each, holding "A|B"
                 haplotype counts. Produce it by running phASER
                 (github.com/secastel/phaser) on the BrainVar BAMs with the
                 WGS genotypes. Use the WASP-corrected alignments if you have
                 them -- see the mapping-bias note below.
  --meta         sample metadata TSV. Needs a sample-ID column and an age
                 column; prenatal samples are selected by age.
  --genotypes    phased VCF/BCF for the same subjects (for the QTL step).

WHY THE FILTERS BELOW ARE NOT OPTIONAL
--------------------------------------
Three findings from docs/ase_validation.md drive this script:

  sec 7d/2   tau_mode='estimate' is required. The old 'zero' default makes
             EVERY null test significant on real haplotype counts
             (lambda_GC = 3020 on GTEx). The default is now 'estimate'; this
             script refuses to emit a config that overrides it back.
  sec 7i     hapmixQTL does not model reference mapping bias and fails
             catastrophically rather than gradually in its presence. This
             script runs reference_bias_diagnostic and REFUSES to proceed if it
             flags, because proceeding produces confidently wrong answers.
  (this run) Samples with no allele-specific coverage get v_inf = 0 exactly and
             would receive the largest weight in the dataset while carrying no
             information. The shipped code now zeroes those weights; this script
             additionally drops gene-samples below a coverage floor, following
             mixQTL's asc_cutoff = 5 / trc_cutoff = 20.

Usage:
  python3 scripts/prep_brainvar.py --phaser phaser_matrix.txt.gz \\
      --meta brainvar_samples.tsv --out prepped/ \\
      --age-col AgePCW --id-col SampleID --max-pcw 40
"""

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    from tensorqtl.hapmixqtl import reference_bias_diagnostic
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent / 'tensorqtl'))
    from hapmixqtl import reference_bias_diagnostic

# mixQTL's shipped defaults (R/mixqtl.R)
ASC_CUTOFF, TRC_CUTOFF, MIN_SAMPLES = 5, 20, 30


def _open(p):
    return gzip.open(p, 'rt') if str(p).endswith('.gz') else open(p)


def select_fetal(meta_path, id_col, age_col, max_pcw, sep='\t'):
    """Prenatal samples. BrainVar ages are commonly post-conceptual weeks;
    birth is ~40 PCW, so the prenatal subset is age < max_pcw."""
    import csv
    rows = list(csv.DictReader(_open(meta_path), delimiter=sep))
    if not rows:
        raise SystemExit(f'no rows parsed from {meta_path}')
    for c in (id_col, age_col):
        if c not in rows[0]:
            raise SystemExit(
                f'column {c!r} not in {meta_path}. Columns: {list(rows[0])}')
    fetal, skipped = [], 0
    for r in rows:
        try:
            age = float(str(r[age_col]).strip())
        except (TypeError, ValueError):
            skipped += 1
            continue
        if age < max_pcw:
            fetal.append(r[id_col].strip())
    print(f'  {len(rows)} samples in metadata; {len(fetal)} prenatal '
          f'({age_col} < {max_pcw}); {skipped} with unparseable age')
    return fetal


def load_phaser(path, keep_samples):
    """Read a phASER matrix, restricted to `keep_samples`. Returns yL, yR
    [genes x samples], gene ids, and the sample order actually found."""
    fh = _open(path)
    hdr = fh.readline().rstrip('\n').split('\t')
    all_samples = hdr[4:]
    want = set(keep_samples)
    # phASER sample columns often carry suffixes; match on prefix as a fallback
    idx, names = [], []
    for i, s in enumerate(all_samples, start=4):
        if s in want:
            idx.append(i); names.append(s)
    if not idx:
        for i, s in enumerate(all_samples, start=4):
            base = s.split('-')[0] if '-' in s else s
            if s in want or base in want or any(s.startswith(w) for w in want):
                idx.append(i); names.append(s)
    if not idx:
        raise SystemExit(
            'no phASER columns matched the metadata IDs.\n'
            f'  first phASER columns: {all_samples[:4]}\n'
            f'  first metadata IDs:   {list(keep_samples)[:4]}\n'
            '  Fix --id-col, or pre-map the IDs.')
    genes, YL, YR = [], [], []
    for line in fh:
        f = line.rstrip('\n').split('\t')
        if len(f) < 5:
            continue
        yl = np.zeros(len(idx), np.int64); yr = np.zeros(len(idx), np.int64)
        ok = True
        for k, i in enumerate(idx):
            v = f[i]; j = v.find('|')
            if j < 0:
                ok = False; break
            try:
                yl[k] = int(v[:j]); yr[k] = int(v[j + 1:])
            except ValueError:
                ok = False; break
        if ok:
            genes.append(f[1]); YL.append(yl); YR.append(yr)
    return np.array(YL), np.array(YR), np.array(genes), names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phaser', required=True)
    ap.add_argument('--meta', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--id-col', default='SampleID')
    ap.add_argument('--age-col', default='AgePCW')
    ap.add_argument('--max-pcw', type=float, default=40.0,
                    help='prenatal cutoff; birth is ~40 post-conceptual weeks')
    ap.add_argument('--sign', default=None,
                    help='optional [genes x samples] signed het indicator '
                         '(npy) for the reference-bias diagnostic')
    ap.add_argument('--force', action='store_true',
                    help='proceed even if the reference-bias gate flags '
                         '(NOT recommended -- see docs/ase_validation.md sec 7i)')
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    print('Selecting prenatal samples')
    fetal = select_fetal(args.meta, args.id_col, args.age_col, args.max_pcw)
    if len(fetal) < MIN_SAMPLES:
        raise SystemExit(f'only {len(fetal)} prenatal samples; too few to map QTLs')

    print('Reading phASER haplotype expression')
    yL, yR, genes, samples = load_phaser(args.phaser, fetal)
    print(f'  {len(genes)} genes x {len(samples)} prenatal samples')

    tot = yL + yR
    usable = (yL >= ASC_CUTOFF) & (yR >= ASC_CUTOFF)
    keep_gene = usable.sum(1) >= MIN_SAMPLES
    print(f'  {int(keep_gene.sum())} genes have >= {MIN_SAMPLES} samples with '
          f'>= {ASC_CUTOFF} reads on BOTH haplotypes (mixQTL asc_cutoff)')
    yL, yR, genes, tot = yL[keep_gene], yR[keep_gene], genes[keep_gene], tot[keep_gene]

    report = {'n_prenatal_samples': len(samples), 'n_genes_kept': int(len(genes)),
              'median_as_depth': float(np.median(tot[tot > 0])) if (tot > 0).any() else 0.0,
              'asc_cutoff': ASC_CUTOFF, 'trc_cutoff': TRC_CUTOFF,
              'max_pcw': args.max_pcw, 'samples': samples}

    # ---- the mapping-bias gate (docs/ase_validation.md sec 7i) ----
    if args.sign:
        sign = np.load(args.sign)[keep_gene]
        diag = reference_bias_diagnostic(yL, yR, sign)
        report['reference_bias'] = {k: v for k, v in diag.items()
                                    if k != 'per_gene'}
        print('\nReference-bias gate:\n  ' + diag['message'])
        if diag['flag'] and not args.force:
            np.save(out / 'reference_bias_per_gene.npy', diag['per_gene'])
            raise SystemExit(
                '\nREFUSING TO PROCEED. hapmixQTL does not model reference '
                'mapping bias and is severely anticonservative in its presence '
                '(docs/ase_validation.md sec 7i: nominal type-I 0.635 at '
                'phi=0.60). Re-run phASER on WASP-corrected alignments, or use '
                'a variant-aware aligner, then try again. Per-gene fractions '
                'written for triage. --force overrides at your own risk.')
    else:
        print('\n  [!] --sign not supplied, so the reference-bias gate was '
              'SKIPPED.\n      This is the one precondition hapmixQTL cannot '
              'check for itself;\n      supply the signed het indicator to '
              'enable it.')
        report['reference_bias'] = 'skipped (no --sign)'

    np.savez_compressed(out / 'brainvar_fetal_haplotype_counts.npz',
                        yL=yL, yR=yR, genes=genes, samples=np.array(samples))
    (out / 'prep_report.json').write_text(json.dumps(report, indent=2))
    print(f'\nwrote {out}/brainvar_fetal_haplotype_counts.npz')
    print(f'wrote {out}/prep_report.json')
    print('\nNext: compute Gibbs summaries (A, T, Va, Vt) and run\n'
          '  hapmixqtl.map_cis(..., tau_mode="estimate")   # the default\n'
          'Do NOT set tau_mode="zero" (docs/ase_validation.md sec 2, 6, 7d).')


if __name__ == '__main__':
    main()

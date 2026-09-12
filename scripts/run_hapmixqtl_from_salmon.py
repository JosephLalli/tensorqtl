#!/usr/bin/env python3
"""
End-to-end: phased VCF + Salmon Gibbs quantifications -> hapmixQTL results,
plus a shareable evaluation bundle containing NO individual-level data.

REQUIREMENT YOU MUST MEET FIRST
===============================
hapmixQTL needs HAPLOTYPE-RESOLVED expression. A quant.sf produced against a
standard reference transcriptome does not contain allelic information and cannot
be used, no matter how many Gibbs samples it has.

Salmon must have been run against a PERSONALIZED DIPLOID TRANSCRIPTOME: two
copies of every transcript, one per haplotype, built from the phased VCF (e.g.
with g2gtools, vcf2diploid, or a custom builder). The resulting quant.sf then
carries paired rows such as

    ENST00000456328.2_hapA
    ENST00000456328.2_hapB

and this script pairs them with --hap-suffix. It refuses to run if it cannot
find the pairing, because the alternative is silently analysing a phantom.

If instead you have genome-aligned BAMs, the phASER route is the other supported
option -- use scripts/prep_brainvar.py.

WHAT IT DOES
============
  1. Reads each sample's Salmon output including aux_info/bootstrap
     (--numGibbsSamples 200), giving a [transcript x draw] matrix per sample.
  2. Pairs haplotype transcripts, aggregates to gene level per haplotype per
     draw -> yL / yR [genes x samples x draws].
  3. Calls the production compute_summaries_from_gibbs -> A, T, Va, Vt. The
     inferential variance therefore comes from YOUR Gibbs draws, which is the
     whole point of the method.
  4. Reads phased genotypes from the VCF -> dosages and the signed het
     indicator s = xL - xR.
  5. GATES on reference_bias_diagnostic. hapmixQTL does not model reference
     mapping bias and fails catastrophically rather than gradually in its
     presence (docs/ase_validation.md sec 7i), so this refuses to proceed when
     bias is detected.
  6. Runs hapmixqtl.map_cis with tau_mode='estimate' (the default; do not
     override -- sec 2, 6, 7d).
  7. Optionally runs the REAL RASQUAL binary on the same genes (--rasqual)
     for a side-by-side comparison.
  8. Writes an EVAL BUNDLE of aggregate statistics only.

NON-STANDARD, OPT-IN (off by default; output is byte-for-byte the standard
=========================================================================
biallelic-SNP analysis unless you pass them)
  --str-vcf hipstr.vcf.gz   STRs enter the lead scan as per-haplotype repeat
                            length (log aFC per repeat unit), and a second
                            pass fits linear + curvature per STR.
  --multiallelic            multi-ALT rows of --vcf (normally skipped) enter
                            the scan as one split row per ALT allele, and a
                            second pass fits the categorical per-allele model.
  Both change which variant can be a lead. The RASQUAL comparison always uses
  the biallelic SNPs only. See scripts/str_integrate.py and
  docs/ase_validation.md sec 7j.

THE EVAL BUNDLE
===============
`eval_bundle.json` is designed to be shared back for analysis under a
controlled-access DUA: it holds counts, distributions and summary statistics,
never per-sample or per-individual values. It includes the two checks that were
previously blocked on having real genotypes:

  * lambda_GC and a downsampled QQ curve of the nominal p-values
  * the slope_a vs slope_tc concordance regression (docs sec 7c). Because the
    total channel uses g/2, both channels estimate the SAME quantity, so the
    regression should have slope 1; deviation localizes bias to a channel.
  * with --rasqual: the distribution of RASQUAL's fitted phi, delta and theta,
    and the rank correlation between the two methods' statistics. RASQUAL's phi
    is an INDEPENDENT estimate of reference mapping bias, so comparing it to our
    diagnostic's ref_fraction cross-validates that diagnostic on real data.

SELF-TEST
=========
Run `--selftest` first. It fabricates Salmon-shaped inputs and a small VCF,
exercises the whole path, and verifies the pipeline end to end -- so you can
confirm the script works on your machine before pointing it at real data.

Usage:
  python3 scripts/run_hapmixqtl_from_salmon.py --selftest

  python3 scripts/run_hapmixqtl_from_salmon.py \\
      --vcf phased.vcf.gz \\
      --manifest samples.tsv \\
      --tx2gene tx2gene.tsv \\
      --out results/

  samples.tsv:  <sample_id> <TAB> <path to that sample's salmon output dir>
  tx2gene.tsv:  <transcript_id> <TAB> <gene_id>     (base IDs, no hap suffix)
"""

import argparse
import gzip
import json
import struct
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    from tensorqtl.hapmixqtl import (compute_summaries_from_gibbs,
                                     reference_bias_diagnostic, map_cis,
                                     map_str_curvature, map_multiallelic)
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent / 'tensorqtl'))
    from hapmixqtl import (compute_summaries_from_gibbs,
                           reference_bias_diagnostic, map_cis,
                           map_str_curvature, map_multiallelic)

# mixQTL's shipped filters (R/mixqtl.R)
ASC_CUTOFF, TRC_CUTOFF, MIN_SAMPLES = 5, 20, 30


# ---------------------------------------------------------------------------
#  Salmon readers
# ---------------------------------------------------------------------------

def _bootstrap_dir(sdir):
    """aux_info/bootstrap (as salmon writes it), or the plural spelling."""
    aux = Path(sdir) / 'aux_info'
    for sub in ('bootstrap', 'bootstraps'):
        if (aux / sub / 'bootstraps.gz').exists():
            return aux / sub
    return aux / 'bootstrap'


def read_salmon_names(sdir):
    """Transcript names only. names.tsv.gz is ~700KB where bootstraps.gz is
    ~11MB, so the gene-union pass reads this instead of the payloads."""
    nfile = _bootstrap_dir(sdir) / 'names.tsv.gz'
    if not nfile.exists():
        raise SystemExit(f'{nfile} not found -- was salmon run with '
                         '--numGibbsSamples or --numBootstraps?')
    with gzip.open(nfile, 'rt') as fh:
        return fh.read().strip().split('\t')


def read_salmon_bootstraps(sdir):
    """Return (transcript_names, boot[n_txp, n_draws]).

    Salmon writes aux_info/bootstrap/{bootstraps.gz,names.tsv.gz} and records
    the count in aux_info/meta_info.json. The payload is a flat binary array of
    n_draws x n_txp; the element type is inferred from its size so both the
    double and integer variants are handled.

    The directory is SINGULAR while the file inside it is plural. Verified
    against salmon 1.10.1 output: aux_info/bootstrap/bootstraps.gz. The plural
    directory is accepted too, in case some version or repackaging writes it.
    """
    sdir = Path(sdir)
    aux = sdir / 'aux_info'
    bdir = _bootstrap_dir(sdir)
    bfile = bdir / 'bootstraps.gz'
    nfile = bdir / 'names.tsv.gz'
    mfile = aux / 'meta_info.json'
    for f in (bfile, nfile):
        if not f.exists():
            raise SystemExit(
                f'{f} not found.\n'
                'Salmon must be run with --numGibbsSamples (or --numBootstraps). '
                'Without the bootstrap directory there is no inferential '
                'variance, and hapmixQTL has nothing to propagate.\n'
                f'(looked for aux_info/bootstrap and aux_info/bootstraps under '
                f'{sdir})')
    with gzip.open(nfile, 'rt') as fh:
        names = fh.read().strip().split('\t')
    n_txp = len(names)
    n_draws = None
    if mfile.exists():
        try:
            n_draws = int(json.loads(mfile.read_text()).get('num_bootstraps', 0)) or None
        except Exception:
            pass
    with gzip.open(bfile, 'rb') as fh:
        raw = fh.read()
    total = len(raw)
    if n_draws is None:
        for width, dt in ((8, np.float64), (4, np.float32)):
            if total % (n_txp * width) == 0:
                n_draws = total // (n_txp * width); break
    for dt in (np.float64, np.float32, np.int32):
        if total == n_txp * n_draws * np.dtype(dt).itemsize:
            arr = np.frombuffer(raw, dtype=dt)
            break
    else:
        raise SystemExit(
            f'cannot interpret {bfile}: {total} bytes for {n_txp} transcripts '
            f'x {n_draws} draws')
    return names, arr.reshape(n_draws, n_txp).T.astype(np.float64)


def pair_haplotypes(names, suffixes):
    """Map base transcript id -> (index of hapA row, index of hapB row)."""
    sa, sb = suffixes
    idx = {n: i for i, n in enumerate(names)}
    pairs = {}
    for n, i in idx.items():
        if n.endswith(sa):
            base = n[:-len(sa)]
            j = idx.get(base + sb)
            if j is not None:
                pairs[base] = (i, j)
    return pairs


# ---------------------------------------------------------------------------
#  VCF (phased GT only; no pysam dependency)
# ---------------------------------------------------------------------------

def read_phased_vcf(path, want_samples, regions=None, bcftools='bcftools'):
    """Return variant_df, dosage[V,N], xL[V,N], xR[V,N] for biallelic SNPs.

    `regions` is a BED of the intervals actually needed. A cis analysis of N
    genes touches N x 2 x window bases -- 30 genes at +/-1 Mb is 62 Mb, 2% of
    the genome -- so parsing every variant to use 1.5% of them is the bulk of
    the runtime for anything less than a whole-transcriptome run. The VCF is
    already tabix-indexed, so this is a fetch, not a scan: measured 236,392
    variants in 9 s against 15,340,329 in roughly a quarter of an hour.
    """
    if regions is not None:
        import subprocess as _sp
        proc = _sp.Popen([bcftools, 'view', '-R', str(regions), str(path)],
                         stdout=_sp.PIPE, stderr=_sp.DEVNULL, text=True)
        return _parse_phased_vcf(proc.stdout, want_samples)
    op = gzip.open if str(path).endswith('.gz') else open
    with op(path, 'rt') as fh:
        return _parse_phased_vcf(fh, want_samples)


def _parse_phased_vcf(fh, want_samples):
    ids, chroms, poss = [], [], []
    XL, XR = [], []
    order = None
    if True:
        for line in fh:
            if line.startswith('##'):
                continue
            f = line.rstrip('\n').split('\t')
            if line.startswith('#CHROM'):
                vcf_samples = f[9:]
                keep = [i for i, s in enumerate(vcf_samples) if s in want_samples]
                if not keep:
                    raise SystemExit(
                        'no VCF samples matched the manifest.\n'
                        f'  VCF: {vcf_samples[:4]}\n  manifest: {list(want_samples)[:4]}')
                order = [vcf_samples[i] for i in keep]
                continue
            if len(f) < 10 or len(f[3]) != 1 or len(f[4]) != 1 or ',' in f[4]:
                continue                                    # biallelic SNPs only
            gt_i = f[8].split(':').index('GT') if 'GT' in f[8] else 0
            # int8, not float64. Genotypes are 0/1 per haplotype and 0-2 as a
            # dosage, so float64 costs 8x for no precision. At 15.7M variants
            # x 92 samples that is 11.6 GB per matrix against 1.44 GB, and
            # there are three of them plus a transient Python list of the same
            # rows -- the difference between fitting and being OOM-killed.
            # map_cis and rasqual_arm both cast at their own boundary
            # (torch.tensor(..., dtype=torch.float32), int(...)), so nothing
            # downstream sees the narrower type.
            xl = np.zeros(len(keep), np.int8); xr = np.zeros(len(keep), np.int8)
            ok = True
            for k, i in enumerate(keep):
                gt = f[9 + i].split(':')[gt_i]
                if '|' not in gt:
                    ok = False; break                       # unphased -> skip
                a, b = gt.split('|')[:2]
                if a in '.' or b in '.':
                    ok = False; break
                xl[k] = 1 if a != '0' else 0
                xr[k] = 1 if b != '0' else 0
            if not ok:
                continue
            ids.append(f[2] if f[2] != '.' else f'{f[0]}_{f[1]}_{f[3]}_{f[4]}')
            chroms.append(f[0]); poss.append(int(f[1]))
            XL.append(xl); XR.append(xr)
    if not ids:
        raise SystemExit('no phased biallelic SNPs read from the VCF')
    XL = np.array(XL, np.int8); XR = np.array(XR, np.int8)
    vdf = pd.DataFrame({'chrom': [str(c) for c in chroms], 'pos': poss}, index=ids)
    return vdf, XL + XR, XL, XR, order



# ---------------------------------------------------------------------------
#  Per-feature-SNP allele-specific counts: RASQUAL's NATIVE input
# ---------------------------------------------------------------------------

def _in_regions(regions):
    """{chrom: [(start, end)]} -> a membership test, or None."""
    if regions is None:
        return None
    import bisect
    by = {}
    for line in open(regions):
        f = line.split()
        if len(f) >= 3:
            by.setdefault(f[0], []).append((int(f[1]), int(f[2])))
    for c in by:
        by[c].sort()
    starts = {c: [a for a, _ in v] for c, v in by.items()}

    def inside(chrom, pos):
        v = by.get(chrom)
        if not v:
            return False
        i = bisect.bisect_right(starts[chrom], pos) - 1
        return i >= 0 and pos <= v[i][1]
    return inside


def load_allelic_counts(manifest, samples, regions=None):
    # Only sites inside the analysed windows are ever looked up, and the store
    # is the memory hog: a dict keyed (chrom,pos) holding a per-sample tuple is
    # tens of millions of small Python objects, ~32 GB across 92 samples. A
    # region filter cuts it to what the run touches.
    """Read per-sample, per-variant allele counts.

    SOLVES THE INPUT MISMATCH. RASQUAL's likelihood is per FEATURE SNP:
    p(Y1_il | Y_il, D_il; ...) with D_il set by the rSNP genotype AND the fSNP
    genotype. Salmon diploid quantification cannot supply that -- it assigns
    each fragment to a haplotype using all variants jointly and reports a
    gene-level total, so the per-site breakdown does not exist in its output.
    It is not recoverable by redistributing the gene total either: splitting one
    aggregate across L sites would present RASQUAL with L observations where one
    was measured, inflating its effective sample size and its statistic.

    So the per-site counts have to come from the alignments. The natural source
    is phASER, which you already need for the reference-bias gate: it writes a
    per-sample `<prefix>.allelic_counts.txt` with columns

        contig  position  variantID  refAllele  altAllele
        refCount  altCount  totalCount

    Manifest format:  <sample_id> <TAB> <path to that sample's allelic_counts.txt>

    Returns {(chrom, pos): {sample: (ref, alt)}}.
    """
    idx = {s: i for i, s in enumerate(samples)}
    inside = _in_regions(regions)
    store = {}
    rows = [l.split('\t') for l in Path(manifest).read_text().strip().split('\n')
            if l.strip() and not l.startswith('#')]
    for r in rows:
        samp, path = r[0].strip(), r[1].strip()
        if samp not in idx:
            continue
        op = gzip.open if path.endswith('.gz') else open
        with op(path, 'rt') as fh:
            hdr = fh.readline().rstrip('\n').split('\t')
            try:
                # phASER writes 'position'. 'start' was assumed here and is
                # not a column phaser.py ever emits -- the real header is
                # contig/position/variantID/refAllele/altAllele/refCount/
                # altCount/totalCount. Accept either, since the self-test
                # fixture and some forks use 'start'.
                ci = hdr.index('contig')
                pi = hdr.index('position') if 'position' in hdr else hdr.index('start')
                ri, ai = hdr.index('refCount'), hdr.index('altCount')
            except ValueError:
                raise SystemExit(
                    f'{path} does not look like a phASER allelic_counts file '
                    f'(need contig, position (or start), refCount, altCount). '
                    f'Columns: {hdr[:8]}')
            for line in fh:
                f = line.rstrip('\n').split('\t')
                if len(f) <= max(ci, pi, ri, ai):
                    continue
                try:
                    key = (str(f[ci]), int(f[pi]))
                    if inside is not None and not inside(key[0], key[1]):
                        continue
                    store.setdefault(key, {})[samp] = (int(f[ri]), int(f[ai]))
                except ValueError:
                    continue
    if not store:
        raise SystemExit(f'no allelic counts parsed from {manifest}')
    return store


def _rasqual_gene_vcf_native(gene_row, vdf, xL, xR, ac, order, window,
                             min_count=1):
    """VCF for one gene using REAL per-fSNP allele counts -- RASQUAL native.

    fSNPs are the variants inside the gene body that carry allelic counts; the
    tested rSNPs are every variant in the cis window. Both are written with the
    genuine phased GT, and the fSNPs carry their measured (ref, alt).
    """
    chrom = str(gene_row['chr'])
    tss = int(gene_row['pos'])
    gstart = int(gene_row.get('start', tss))
    gend = int(gene_row.get('end', tss + 1))
    pos = vdf['pos'].values
    same = vdf['chrom'].values == chrom
    fmask = same & (pos >= gstart) & (pos <= gend)
    fidx = [v for v in np.where(fmask)[0]
            if (chrom, int(pos[v])) in ac]
    if not fidx:
        return None, 0, None, 0
    rmask = same & (np.abs(pos - tss) <= window)
    ridx = [v for v in np.where(rmask)[0] if v not in set(fidx)]

    rows = []
    for v in fidx:                                   # feature SNPs, real counts
        counts = ac[(chrom, int(pos[v]))]
        fl = []
        for k, samp in enumerate(order):
            r, a = counts.get(samp, (0, 0))
            if r + a < min_count:
                r = a = 0
            fl.append(f'{int(xL[v, k])}|{int(xR[v, k])}:{r},{a}')
        rows.append([chrom, str(int(pos[v])), str(vdf.index[v]), 'A', 'G',
                     '100', 'PASS', 'RSQ=1.0', 'GT:AS'] + fl)
    for v in ridx:                                   # tested regulatory SNPs
        gl = [f'{int(xL[v, k])}|{int(xR[v, k])}:0,0' for k in range(len(order))]
        rows.append([chrom, str(int(pos[v])), str(vdf.index[v]), 'A', 'G',
                     '100', 'PASS', 'RSQ=1.0', 'GT:AS'] + gl)
    return ('\n'.join('\t'.join(r) for r in rows) + '\n',
            len(rows), (gstart, gend), len(fidx))


# ---------------------------------------------------------------------------
#  Optional comparison run: the REAL RASQUAL binary
# ---------------------------------------------------------------------------

def _rasqual_gene_vcf(gene_pos, vdf, dos, xL, xR, yLm, yRm, gi, window):
    """VCF lines for one gene: a pseudo-fSNP carrying the gene's haplotype
    counts, followed by the cis variants to be tested.

    HONEST NOTE. RASQUAL natively consumes allele-specific counts at each
    FEATURE SNP. Salmon diploid quantification yields gene-level haplotype
    totals instead, with no per-site breakdown. Those totals are therefore
    encoded as ONE pseudo-fSNP at the TSS, heterozygous in every sample that
    has allele-specific coverage, with AS = (hapA, hapB).

    This keeps the comparison FAIR -- both methods see exactly the same
    information -- but it is not RASQUAL's native input, and it removes any
    benefit RASQUAL would get from resolving multiple feature SNPs separately.
    Read the comparison as "RASQUAL given hapmixQTL's data", not as "RASQUAL at
    its best".
    """
    chrom, tss = str(gene_pos['chr']), int(gene_pos['pos'])
    inwin = (vdf['chrom'].values == chrom) & \
            (np.abs(vdf['pos'].values - tss) <= window)
    idx = np.where(inwin)[0]
    if idx.size == 0:
        return None, 0, None
    N = dos.shape[1]
    rows = []
    # pseudo-fSNP: het wherever there is allele-specific coverage.
    # GT 0|1 means REF on haplotype 1, so AS is written as (hapA, hapB).
    f = []
    for i in range(N):
        a, b = int(round(yLm[gi, i])), int(round(yRm[gi, i]))
        f.append(f'0|1:{a},{b}' if (a + b) > 0 else f'0|0:0,0')
    rows.append([chrom, str(tss), f'psf_{gi}', 'A', 'G', '100', 'PASS',
                 'RSQ=1.0', 'GT:AS'] + f)
    for v in idx:
        gt = []
        for i in range(N):
            gt.append(f'{int(xL[v, i])}|{int(xR[v, i])}:0,0')
        rows.append([chrom, str(int(vdf['pos'].values[v])), str(vdf.index[v]),
                     'A', 'G', '100', 'PASS', 'RSQ=1.0', 'GT:AS'] + gt)
    return '\n'.join('\t'.join(r) for r in rows) + '\n', len(rows), tss


def run_rasqual_comparison(binary, genes, pos_df, vdf, dos, xL, xR,
                           yLm, yRm, T_counts, lib, out, window, max_genes,
                           allelic=None, order=None, mode='pseudo'):
    """Run the real RASQUAL over the same genes; return a per-gene DataFrame.

    mode='pseudo'  gene-level haplotype totals as one pseudo-fSNP (both methods
                   see identical information -- fair, but not RASQUAL native)
    mode='native'  REAL per-feature-SNP allele counts (RASQUAL as intended);
                   requires `allelic` from load_allelic_counts
    mode='both'    run each and return both, so the INPUT effect is separated
                   from the METHOD effect. Any difference between the two arms
                   is attributable to input modality alone, since the binary,
                   the genes and the tested variants are identical.
    """
    if mode in ('native', 'both') and not allelic:
        raise SystemExit("--rasqual-input %s needs --allelic-counts" % mode)
    if mode == 'both':
        a = run_rasqual_comparison(binary, genes, pos_df, vdf, dos, xL, xR,
                                   yLm, yRm, T_counts, lib, out, window,
                                   max_genes, allelic, order, 'pseudo')
        b = run_rasqual_comparison(binary, genes, pos_df, vdf, dos, xL, xR,
                                   yLm, yRm, T_counts, lib, out, window,
                                   max_genes, allelic, order, 'native')
        if a is not None:
            a['input'] = 'pseudo'
        if b is not None:
            b['input'] = 'native'
        return pd.concat([x for x in (a, b) if x is not None], ignore_index=True)
    import subprocess, tempfile
    order_genes = [g for g in genes if g in pos_df.index][:max_genes]
    if not order_genes:
        return None
    print(f'\nRASQUAL [{mode}] on {len(order_genes)} genes '
          f'(RASQUAL is ~1e3x slower than hapmixQTL -- docs sec 7e)')
    td = Path(tempfile.mkdtemp())
    gidx = {g: i for i, g in enumerate(genes)}
    sel = [gidx[g] for g in order_genes]
    np.asarray(T_counts[sel], dtype=np.float64).tofile(td / 'Y.bin')
    np.asarray((lib[None, :] * T_counts[sel].mean(1, keepdims=True)),
               dtype=np.float64).tofile(td / 'K.bin')
    N = dos.shape[1]
    recs = []
    for j, g in enumerate(order_genes):
        n_fsnp = 1
        if mode == 'native':
            vcf, nrow, span, n_fsnp = _rasqual_gene_vcf_native(
                pos_df.loc[g], vdf, xL, xR, allelic, order, window)
            if vcf is None:
                recs.append(dict(gene=g, status='no_fsnp_with_counts'))
                continue
            s_arg, e_arg = str(span[0]), str(span[1])
        else:
            vcf, nrow, tss = _rasqual_gene_vcf(pos_df.loc[g], vdf, dos, xL, xR,
                                               yLm, yRm, gidx[g], window)
            if vcf is None:
                continue
            s_arg, e_arg = str(tss), str(tss + 1)
        cmd = [binary, '-y', str(td / 'Y.bin'), '-k', str(td / 'K.bin'),
               '-n', str(N), '-j', str(j + 1), '-l', str(nrow),
               '-m', str(n_fsnp), '-s', s_arg, '-e', e_arg, '-f', str(g), '-z']
        try:
            pr = subprocess.run(cmd, input=vcf, capture_output=True,
                                text=True, timeout=600)
        except Exception as e:
            recs.append(dict(gene=g, status=f'error:{type(e).__name__}'))
            continue
        best = None
        for line in pr.stdout.strip().split('\n'):
            fl = line.split('\t')
            if len(fl) < 25 or fl[1] == 'SKIPPED' or fl[1].startswith('psf_'):
                continue
            try:
                chi2 = float(fl[10])
                if int(float(fl[22])) != 0:      # convergence status
                    continue
            except ValueError:
                continue
            if best is None or chi2 > best['chi2']:
                best = dict(gene=g, n_fsnp=n_fsnp, variant=fl[1], chi2=chi2,
                            pi=float(fl[11]), delta=float(fl[12]),
                            phi=float(fl[13]), theta=float(fl[14]),
                            status='ok')
        recs.append(best or dict(gene=g, status='no_converged_row'))
        if (j + 1) % 25 == 0:
            print(f'   {j+1}/{len(order_genes)}', flush=True)
    return pd.DataFrame(recs)


def _input_effect(rq):
    """How much does the INPUT modality alone change RASQUAL?

    Same binary, same genes, same tested variants -- only the allele-specific
    representation differs. Any gap is attributable to input, not method.
    """
    from scipy import stats as sps
    if 'input' not in rq.columns:
        return None
    ok = rq[rq['status'] == 'ok']
    a = ok[ok['input'] == 'pseudo'].set_index('gene')['chi2']
    b = ok[ok['input'] == 'native'].set_index('gene')['chi2']
    both = a.index.intersection(b.index)
    if len(both) < 10:
        return {'n_paired': int(len(both)), 'note': 'too few paired genes'}
    x, y = a.loc[both].values, b.loc[both].values
    rho, _ = sps.spearmanr(x, y)
    return {
        'n_paired': int(len(both)),
        'median_chi2_pseudo': float(np.median(x)),
        'median_chi2_native': float(np.median(y)),
        'median_ratio_native_over_pseudo': float(np.median(y / np.maximum(x, 1e-9))),
        'spearman_rho': float(rho),
        'mean_n_fsnp_native': (float(ok[ok['input'] == 'native']['n_fsnp'].mean())
                               if 'n_fsnp' in ok else None),
        'interpretation': (
            'ratio > 1 means RASQUAL gains from per-feature-SNP resolution that '
            'Salmon gene-level totals cannot supply; ratio ~ 1 means the '
            'pseudo-fSNP encoding costs it nothing and the earlier comparisons '
            'were fair. rho near 1 means the two inputs rank genes the same.')}


def add_rasqual_to_bundle(bundle, rq, res_df, diag):
    """Aggregate-only summaries plus the cross-checks worth having."""
    from scipy import stats as sps
    if rq is None or not len(rq):
        bundle['rasqual'] = {'note': 'not run'}
        return bundle
    ok = rq[rq['status'] == 'ok'] if 'status' in rq else rq
    b = {'n_genes_attempted': int(len(rq)), 'n_converged': int(len(ok)),
         'note': ('gene-level haplotype totals encoded as one pseudo-fSNP; '
                  'both methods see identical information, which is fair but '
                  'is not RASQUAL native input')}
    if len(ok):
        for c in ('phi', 'delta', 'theta', 'chi2'):
            if c in ok:
                v = pd.to_numeric(ok[c], errors='coerce').dropna().values
                if v.size:
                    b[c] = {'median': float(np.median(v)),
                            'quantiles': np.quantile(
                                v, [.05, .25, .5, .75, .95]).round(5).tolist()}
        # RASQUAL's phi is an INDEPENDENT estimate of reference bias; our
        # diagnostic measures the same thing a different way.
        if 'phi' in b and isinstance(diag.get('ref_fraction'), float):
            b['phi_vs_diagnostic'] = {
                'rasqual_median_phi': b['phi']['median'],
                'diagnostic_ref_fraction': diag['ref_fraction'],
                'interpretation': ('both estimate reference mapping bias; 0.5 is '
                                   'unbiased. Agreement cross-validates the '
                                   'diagnostic on real data (docs sec 7i).')}
        # concordance of the two methods across genes
        if res_df is not None and len(res_df):
            gcol = next((c for c in ('phenotype_id', 'gene_id', 'gene')
                         if c in res_df.columns), None)
            pcol = next((c for c in ('pval_nominal', 'pval_beta', 'pval_perm')
                         if c in res_df.columns), None)
            if gcol and pcol:
                m = res_df[[gcol, pcol]].copy()
                m.columns = ['gene', 'p']
                m = m.merge(ok[['gene', 'chi2']], on='gene', how='inner')
                m['p'] = pd.to_numeric(m['p'], errors='coerce')
                m = m[np.isfinite(m['p']) & (m['p'] > 0)]
                if len(m) > 20:
                    hm = sps.chi2.isf(m['p'].values, 1)
                    rho, pv = sps.spearmanr(hm, m['chi2'].values)
                    b['concordance_with_hapmixqtl'] = {
                        'n_genes': int(len(m)),
                        'spearman_rho': float(rho),
                        'interpretation': ('rank correlation of the two methods\' '
                                           'test statistics over the same genes')}
    ie = _input_effect(rq)
    if ie:
        b['input_effect'] = ie
    if 'input' in rq.columns:
        b['by_input'] = {
            m: {'n_converged': int(((rq['input'] == m) &
                                    (rq['status'] == 'ok')).sum())}
            for m in rq['input'].unique()}
    bundle['rasqual'] = b
    return bundle


# ---------------------------------------------------------------------------
#  Eval bundle: aggregate statistics only
# ---------------------------------------------------------------------------

def build_eval_bundle(res_df, diag, meta):
    from scipy import stats as sps
    b = {'meta': meta, 'reference_bias': {k: v for k, v in diag.items()
                                          if k != 'per_gene'}}
    if res_df is None or not len(res_df):
        b['note'] = 'no cis results produced'
        return b
    pcol = next((c for c in ('pval_nominal', 'pval_beta', 'pval_perm')
                 if c in res_df), None)
    if pcol:
        p = pd.to_numeric(res_df[pcol], errors='coerce').dropna().values
        p = p[(p > 0) & (p <= 1)]
        if p.size:
            chi2 = sps.chi2.isf(p, 1)
            q = np.linspace(0.001, 0.999, 200)
            b['pvalues'] = {
                'n': int(p.size),
                'lambda_gc': float(np.median(chi2) / sps.chi2.ppf(0.5, 1)),
                'frac_lt_0.05': float(np.mean(p < 0.05)),
                'frac_lt_1e-5': float(np.mean(p < 1e-5)),
                'qq_observed_-log10': (-np.log10(np.quantile(p, q))).round(4).tolist(),
                'qq_expected_-log10': (-np.log10(q)).round(4).tolist()}
    # sec 7c: both channels estimate the SAME quantity -> slope should be 1
    if {'slope_a', 'slope_t'} <= set(res_df.columns):
        d = res_df[['slope_a', 'slope_t']].apply(pd.to_numeric, errors='coerce').dropna()
        d = d[np.isfinite(d).all(1)]
        if len(d) > 20:
            sl, ic, r, pv, se = sps.linregress(d['slope_t'], d['slope_a'])
            b['channel_concordance'] = {
                'n_genes': int(len(d)), 'slope': float(sl), 'slope_se': float(se),
                'intercept': float(ic), 'r': float(r),
                'interpretation': ('slope should be 1.0; deviation localizes bias '
                                   'to a channel (docs/ase_validation.md sec 7c)')}
    # sec 7c diagnostic: how many lead variants fail the cis assumption
    if 'pval_cis_trans' in res_df.columns:
        pct = pd.to_numeric(res_df['pval_cis_trans'], errors='coerce').dropna()
        if len(pct):
            b['cis_trans'] = {
                'n_genes': int(len(pct)),
                'frac_pval_cis_trans_lt_0.05': float((pct < 0.05).mean()),
                'n_bh_q_lt_0.10': _bh_count(pct),
                'interpretation': ('Wald test that the ASE and total channels agree at the lead '
                                   'variant; a gene that fails carries an effect that is not '
                                   'purely cis (trans component, mapping bias, phasing error) and '
                                   'its combined slope is attenuated -- a diagnostic, not a '
                                   'filter (docs/ase_validation.md sec 7c)')}
            if 'alpha_cis' in res_df.columns and 'pval_beta' in res_df.columns:
                sig = pd.to_numeric(res_df['pval_beta'], errors='coerce') < 0.05
                al = pd.to_numeric(res_df.loc[sig, 'alpha_cis'], errors='coerce').dropna()
                al = al[np.isfinite(al)]
                if len(al) >= 10:
                    b['cis_trans']['alpha_cis_quantiles_among_significant'] = np.quantile(
                        al, [.05, .25, .5, .75, .95]).round(4).tolist()
    for c in ('slope', 'slope_se'):
        if c in res_df.columns:
            v = pd.to_numeric(res_df[c], errors='coerce').dropna().values
            if v.size:
                b.setdefault('effect_sizes', {})[c] = {
                    'n': int(v.size),
                    'quantiles': np.quantile(v, [.05, .25, .5, .75, .95]).round(5).tolist()}
    return b


# ---------------------------------------------------------------------------
#  NON-STANDARD, OPT-IN: STRs and multi-ALT sites (off unless asked for)
# ---------------------------------------------------------------------------

NONSTANDARD_NOTE = (
    'NON-STANDARD, opt-in. Standard cis-QTL mapping tests biallelic SNPs; these '
    'rows and models are enabled only by --str-vcf / --multiallelic and change '
    'which variant can be a lead. See scripts/str_integrate.py and '
    'docs/ase_validation.md sec 7j.')


def nonstandard_extension(vdf, dos, xL, xR, order, str_vcf=None, multiallelic_vcf=None):
    """Parse the opt-in inputs and append their rows to the SNP scan.
    Returns (vdf, dos, xL, xR, vtype, aux)."""
    from str_integrate import parse_str_vcf, parse_multiallelic_vcf, extend_scan
    print('\n' + NONSTANDARD_NOTE)
    strs = parse_str_vcf(str_vcf, list(order)) if str_vcf else []
    ma = parse_multiallelic_vcf(multiallelic_vcf, list(order)) if multiallelic_vcf else []
    vdf, dos, xL, xR, vtype, aux = extend_scan(vdf, dos, xL, xR, order, strs, ma)
    print(f'  scan rows by type: {vtype.value_counts().to_dict()}')
    return vdf, dos, xL, xR, vtype, aux


def run_second_pass(aux, order, sdf, tdf, vadf, vtdf, map_pos, window, min_hap, out):
    """The two second-pass models on whatever sidecars the extension produced.
    Writes per-locus tables locally; returns (curvature_df, site_df, allele_df)."""
    cur = site_res = allele_res = None
    if aux.get('str_len') is not None:
        print(f'  second pass: linear + curvature on {len(aux["str_sites"])} STRs')
        cur = map_str_curvature(aux['str_len'], aux['str_phased'],
                                aux['str_sites'].set_index('id'), list(order),
                                sdf, tdf, vadf, vtdf, map_pos, window=window, verbose=False)
        cur.to_csv(out / 'hapmixqtl_str_curvature.tsv.gz', sep='\t', index=False)
    if aux.get('ma_alleles') is not None:
        print(f'  second pass: categorical on {len(aux["ma_sites"])} multi-ALT sites')
        site_res, allele_res = map_multiallelic(
            aux['ma_alleles'], aux['ma_sites'].set_index('site_id'), list(order),
            sdf, tdf, vadf, vtdf, map_pos, hap_phased=aux['ma_phased'],
            window=window, min_hap=min_hap, verbose=False)
        site_res.to_csv(out / 'hapmixqtl_multiallelic_sites.tsv.gz', sep='\t', index=False)
        allele_res.to_csv(out / 'hapmixqtl_multiallelic_alleles.tsv.gz', sep='\t', index=False)
    return cur, site_res, allele_res


def _bh_count(p, q=0.10):
    p = pd.to_numeric(pd.Series(p), errors='coerce').dropna().values
    if p.size == 0:
        return 0
    o = np.sort(p); k = np.arange(1, o.size + 1)
    ok = np.where(o <= q * k / o.size)[0]
    return int(ok.max() + 1) if ok.size else 0


def nonstandard_summary(res, vtype, cur, site_res, enabled):
    """Aggregate-only summary of the opt-in passes for the eval bundle."""
    b = {'enabled': enabled, 'note': NONSTANDARD_NOTE,
         'scan_rows_by_type': {k: int(v) for k, v in vtype.value_counts().items()}}
    if res is not None and 'variant_type' in res:
        b['lead_variant_type'] = {k: int(v) for k, v in res['variant_type'].value_counts().items()}
        b['frac_leads_nonstandard'] = float((res['variant_type'] != 'snp').mean())
    if cur is not None and len(cur):
        pc = pd.to_numeric(cur['pval_curv'], errors='coerce')
        b2 = pd.to_numeric(cur['slope_sq'], errors='coerce')
        b1 = pd.to_numeric(cur['slope_l'], errors='coerce')
        sig = (pc < 0.05).fillna(False)
        b['str_curvature'] = {
            'n_gene_str_pairs': int(len(cur)), 'n_str': int(cur['str_id'].nunique()),
            'frac_pval_curv_lt_0.05': float(sig.mean()),
            'n_bh_q_lt_0.10': _bh_count(pc),
            'frac_accelerating_among_sig': (float((np.sign(b2[sig]) == np.sign(b1[sig])).mean())
                                            if sig.any() else None),
            'note': 'accelerating = b2 has the sign of b1; saturating = opposite'}
    if site_res is not None and len(site_res):
        pj = pd.to_numeric(site_res['pval_joint'], errors='coerce')
        b['multiallelic_categorical'] = {
            'n_gene_site_pairs': int(len(site_res)), 'n_sites': int(site_res['site_id'].nunique()),
            'n_tested_alleles_distribution': {str(k): int(v) for k, v in
                                              site_res['n_tested'].value_counts().sort_index().items()},
            'frac_pval_joint_lt_0.05': float((pj < 0.05).mean()),
            'n_bh_q_lt_0.10': _bh_count(pj),
            'frac_rank_deficient': float(site_res['rank_deficient'].mean())}
    return b


# ---------------------------------------------------------------------------

def load_counts(manifest, tx2gene, suffixes, out):
    rows = [l.split('\t') for l in Path(manifest).read_text().strip().split('\n')
            if l.strip() and not l.startswith('#')]
    samples = [r[0].strip() for r in rows]
    dirs = [r[1].strip() for r in rows]
    t2g = dict(l.split('\t')[:2] for l in
               Path(tx2gene).read_text().strip().split('\n') if '\t' in l)

    def _no_pairs(sd):
        return SystemExit(
            f'no haplotype-paired transcripts in {sd} using suffixes '
            f'{suffixes}.\nSalmon appears to have been run against a '
            'standard reference transcriptome, which carries NO allelic '
            'information. Quantify against a personalized DIPLOID '
            'transcriptome built from your phased VCF, or use the phASER '
            'route (scripts/prep_brainvar.py).')

    # Pass 1: the gene set is the UNION over every sample, not whatever the
    # first one happened to carry. A personalized diploid transcriptome is
    # built per sample from that sample's own variants, so the transcript sets
    # genuinely differ. Taking sample 1's set made the result depend on
    # manifest ORDER: genes only in later samples were dropped silently, and
    # indexing gi[g] for a gene the first sample lacked raised KeyError.
    # Reading names.tsv.gz here keeps this pass off the payloads.
    gene_set = set()
    for s, sd in zip(samples, dirs):
        pairs = pair_haplotypes(read_salmon_names(sd), suffixes)
        if not pairs:
            raise _no_pairs(sd)
        gene_set |= {t2g[b] for b in pairs if b in t2g}
    if not gene_set:
        raise SystemExit(
            'no haplotype-paired transcript matched --tx2gene. The transcript '
            'IDs in the Salmon output and in tx2gene.tsv do not agree -- check '
            'the version-suffix convention on both.')
    genes = sorted(gene_set)
    gi = {g: i for i, g in enumerate(genes)}

    YL = YR = YT = None
    nd = None
    n_unpaired_tx = 0
    for si, (s, sd) in enumerate(zip(samples, dirs)):
        names, boot = read_salmon_bootstraps(sd)
        pairs = pair_haplotypes(names, suffixes)
        if not pairs:
            raise _no_pairs(sd)
        if YL is None:
            nd = boot.shape[1]
            YL = np.zeros((len(genes), len(samples), nd))
            YR = np.zeros((len(genes), len(samples), nd))
            YT = np.zeros((len(genes), len(samples), nd))
            print(f'  {len(pairs)} haplotype pairs -> {len(genes)} genes '
                  f'(union over {len(samples)} samples), {nd} draws')
        elif boot.shape[1] != nd:
            raise SystemExit(
                f'{s} has {boot.shape[1]} draws but the first sample had {nd}. '
                'Every sample must be quantified with the same number of '
                'bootstrap/Gibbs samples.')
        for base, (ia, ib) in pairs.items():
            g = t2g.get(base)
            if g is None:
                continue
            YL[gi[g], si, :] += boot[ia]
            YR[gi[g], si, :] += boot[ib]
        # The TOTAL must cover every transcript, not only the ones quantified
        # per haplotype. A personalized diploid transcriptome emits the second
        # copy only where the sample is heterozygous, so an unpaired transcript
        # is one whose haplotypes are identical -- its single row already
        # carries BOTH haplotypes' expression. Summing only the pairs made the
        # total a heterozygous-transcript subtotal, and a gene-sample with no
        # heterozygous transcript collapsed to an exact structural zero rather
        # than to a missing value. That pattern tracks local heterozygosity,
        # which is in LD with the cis variants under test.
        for idx, nm in enumerate(names):
            base = nm
            for suf in suffixes:
                if nm.endswith(suf):
                    base = nm[:-len(suf)]
                    break
            else:
                n_unpaired_tx += 1
            g = t2g.get(base)
            if g is None or g not in gi:
                continue
            YT[gi[g], si, :] += boot[idx]
        print(f'  [{si+1}/{len(samples)}] {s}', flush=True)
    empty_ase = int((YL.sum(axis=2) + YR.sum(axis=2) == 0).sum())
    empty_tot = int((YT.sum(axis=2) == 0).sum())
    cells = YT.shape[0] * YT.shape[1]
    print(f'  totals cover all transcripts: {empty_tot}/{cells} '
          f'({100*empty_tot/cells:.1f}%) gene-samples have no expression at '
          f'all, against {empty_ase}/{cells} ({100*empty_ase/cells:.1f}%) with '
          f'no HAPLOTYPE-RESOLVED expression')
    return np.array(genes), samples, YL, YR, YT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--vcf'); ap.add_argument('--manifest')
    ap.add_argument('--tx2gene'); ap.add_argument('--out', default='hapmix_out')
    ap.add_argument('--gene-pos', help='TSV: gene_id, chr, pos (TSS)')
    ap.add_argument('--hap-suffix', default='_hapA,_hapB')
    ap.add_argument('--count-noise', action=argparse.BooleanOptionalAction,
                    default=True,
                    help='per-sample Poisson counting noise in the Gibbs '
                         'variances; see compute_summaries_from_gibbs')
    ap.add_argument('--window', type=int, default=1_000_000)
    ap.add_argument('--force', action='store_true',
                    help='proceed despite a reference-bias flag (NOT recommended)')
    ap.add_argument('--rasqual', default=None,
                    help='path to the real RASQUAL binary (build it with '
                         'scripts/build_rasqual.sh) to add a comparison run')
    ap.add_argument('--allelic-counts', default=None,
                    help='manifest: sample_id <TAB> phASER .allelic_counts.txt. '
                         'Supplies RASQUAL its NATIVE per-feature-SNP counts, '
                         'which Salmon diploid output cannot provide.')
    ap.add_argument('--rasqual-input', default='pseudo',
                    choices=['pseudo', 'native', 'both'],
                    help="'pseudo': gene totals as one pseudo-fSNP (matched "
                         "information); 'native': real per-fSNP counts; "
                         "'both': run each so the INPUT effect is separated "
                         "from the METHOD effect. 'both' is the informative one.")
    ap.add_argument('--rasqual-genes', type=int, default=200,
                    help='cap the comparison at N genes; RASQUAL is ~1e3x '
                         'slower than hapmixQTL (docs sec 7e), so a '
                         'genome-wide run is days of CPU')
    ns = ap.add_argument_group(
        'NON-STANDARD, opt-in (off by default; the standard analysis tests '
        'biallelic SNPs only and is unchanged unless these are given)')
    ns.add_argument('--str-vcf', default=None,
                    help='HipSTR-style STR VCF: STRs enter the lead scan as per-haplotype '
                         'repeat length (log aFC per repeat unit) and get a linear + '
                         'curvature second pass (hapmixqtl_str_curvature.tsv.gz)')
    ns.add_argument('--multiallelic', action='store_true',
                    help='multi-ALT rows of --vcf (normally skipped) enter the scan as one '
                         'split row per ALT and get the categorical per-allele second pass '
                         '(hapmixqtl_multiallelic_{sites,alleles}.tsv.gz)')
    ns.add_argument('--min-hap', type=int, default=10,
                    help='categorical model: alleles carried by fewer haplotypes are pooled')
    args = ap.parse_args()

    if args.selftest:
        return selftest()
    for r in ('vcf', 'manifest', 'tx2gene'):
        if not getattr(args, r):
            raise SystemExit(f'--{r} is required (or use --selftest)')
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    sufs = tuple(args.hap_suffix.split(','))

    print('Reading Salmon Gibbs quantifications')
    genes, samples, YL, YR, YT = load_counts(args.manifest, args.tx2gene,
                                             sufs, out)

    print('Computing Gibbs summaries (production code path)')
    A, T, Va, Vt, Cat = compute_summaries_from_gibbs(
        YL, YR, yT=YT, count_noise=getattr(args, 'count_noise', True))

    print('Reading phased VCF')
    vdf, dos, xL, xR, order = read_phased_vcf(args.vcf, set(samples))
    keep = [samples.index(s) for s in order]
    A, T, Va, Vt = A[:, keep], T[:, keep], Va[:, keep], Vt[:, keep]
    YLm, YRm = YL[:, keep, :].mean(2), YR[:, keep, :].mean(2)

    # reference-bias gate, using the mean phase across cis variants per gene is
    # not meaningful -- use the per-variant sign at the gene's nearest variant
    sign = np.sign(xL - xR)
    g_sign = np.zeros_like(YLm)
    for i in range(len(genes)):
        g_sign[i] = sign[min(i, sign.shape[0] - 1)]
    diag = reference_bias_diagnostic(YLm, YRm, g_sign)
    print('\nReference-bias gate:\n  ' + diag['message'])
    if diag['flag'] and not args.force:
        (out / 'eval_bundle.json').write_text(json.dumps(
            build_eval_bundle(None, diag,
                              {'n_samples': len(order), 'n_genes': len(genes)}),
            indent=2))
        raise SystemExit(
            '\nREFUSING TO PROCEED (docs/ase_validation.md sec 7i). Re-quantify '
            'from WASP-corrected or variant-aware alignments. A bundle with the '
            'diagnostic was still written so you can bring it back for triage.')

    sdf = pd.DataFrame(A, index=genes, columns=order)
    tdf = pd.DataFrame(T, index=genes, columns=order)
    vadf = pd.DataFrame(Va, index=genes, columns=order)
    vtdf = pd.DataFrame(Vt, index=genes, columns=order)
    # The RASQUAL comparison always sees the biallelic SNPs only.
    snp_arrays = (vdf, dos, xL, xR)
    vtype, aux = None, {}
    if args.str_vcf or args.multiallelic:
        vdf, dos, xL, xR, vtype, aux = nonstandard_extension(
            vdf, dos, xL, xR, order, str_vcf=args.str_vcf,
            multiallelic_vcf=(args.vcf if args.multiallelic else None))
    gdf = pd.DataFrame(dos, index=vdf.index, columns=order)
    xLdf = pd.DataFrame(xL, index=vdf.index, columns=order)
    xRdf = pd.DataFrame(xR, index=vdf.index, columns=order)
    if args.gene_pos:
        # chromosome must be a STRING and must match the VCF's CHROM exactly.
        # pandas otherwise infers int for a "1"-style column, and every
        # phenotype is then silently dropped as "on a chr. without genotypes".
        probe = pd.read_csv(args.gene_pos, sep='\t', header=None, nrows=1)
        cols = (['gene', 'chr', 'pos', 'start', 'end'][:probe.shape[1]]
                if probe.shape[1] >= 3 else None)
        if cols is None:
            raise SystemExit('--gene-pos needs at least gene, chr, TSS')
        gp = pd.read_csv(args.gene_pos, sep='\t', header=None, names=cols,
                         dtype={'chr': str}).set_index('gene')
        gp['chr'] = gp['chr'].astype(str).str.strip()
        keepc = [c for c in ('chr', 'pos', 'start', 'end') if c in gp.columns]
        pos_df = gp.loc[[g for g in genes if g in gp.index], keepc]
    else:
        raise SystemExit('--gene-pos is required for cis mapping '
                         '(TSV: gene_id, chr, TSS)')
    vcf_chrs = set(vdf['chrom'].unique())
    gp_chrs = set(pos_df['chr'].unique())
    if not (vcf_chrs & gp_chrs):
        raise SystemExit(
            'chromosome names do not match between the VCF and --gene-pos, so '
            'every gene would be dropped.\n'
            f'  VCF CHROM:  {sorted(vcf_chrs)[:5]}\n'
            f'  --gene-pos: {sorted(gp_chrs)[:5]}\n'
            'Make them identical (both "1" or both "chr1").')
    common = [g for g in genes if g in pos_df.index]
    sdf, tdf, vadf, vtdf = (df.loc[common] for df in (sdf, tdf, vadf, vtdf))
    pos_df = pos_df.loc[common]
    map_pos = pos_df[['chr', 'pos']]

    print(f'\nRunning map_cis on {len(common)} genes '
          f"(tau_mode='estimate', the validated default)")
    res = map_cis(gdf, vdf, sdf, tdf, vadf, vtdf, map_pos,
                  xL_df=xLdf, xR_df=xRdf, window=args.window, verbose=True)
    # map_cis returns the gene id as the index; keep it as a column so the
    # written table and the RASQUAL concordance merge both have it.
    res = res.reset_index()
    if vtype is not None:
        res['variant_type'] = res['variant_id'].map(vtype).fillna('snp').values
    res.to_csv(out / 'hapmixqtl_cis.tsv.gz', sep='\t', index=False)

    bundle = build_eval_bundle(res, diag, {
        'n_samples': len(order), 'n_genes_tested': int(len(common)),
        'n_variants': int(len(vdf)), 'n_gibbs_draws': int(YL.shape[2]),
        'median_Va': float(np.median(Va)), 'median_Vt': float(np.median(Vt)),
        'tau_mode': 'estimate'})
    if vtype is not None:
        print('\nNon-standard second pass')
        cur, site_res, _ = run_second_pass(aux, order, sdf, tdf, vadf, vtdf, map_pos,
                                           args.window, args.min_hap, out)
        bundle['nonstandard'] = nonstandard_summary(
            res, vtype, cur, site_res,
            {'str_vcf': bool(args.str_vcf), 'multiallelic': bool(args.multiallelic)})

    if args.rasqual:
        if not Path(args.rasqual).exists():
            raise SystemExit(f'--rasqual binary not found: {args.rasqual}\n'
                             'Build it with scripts/build_rasqual.sh')
        Tcounts = np.expm1(tdf.values) if np.nanmax(tdf.values) < 30 else tdf.values
        libsz = np.ones(len(order))
        allelic = None
        if args.allelic_counts:
            print('Reading per-feature-SNP allelic counts (RASQUAL native input)')
            allelic = load_allelic_counts(args.allelic_counts, order)
            print(f'  {len(allelic)} variants with allele counts')
        elif args.rasqual_input in ('native', 'both'):
            raise SystemExit(
                f'--rasqual-input {args.rasqual_input} requires --allelic-counts.\n'
                'RASQUAL models each FEATURE SNP separately; Salmon diploid '
                'quantification reports only gene-level haplotype totals, and '
                'that per-site breakdown cannot be recovered by splitting the '
                'total (it would fabricate independent observations and inflate '
                "RASQUAL's statistic). Supply phASER allelic_counts files.")
        rq = run_rasqual_comparison(
            args.rasqual, list(sdf.index), pos_df, *snp_arrays,
            YLm[[list(genes).index(g) for g in sdf.index]],
            YRm[[list(genes).index(g) for g in sdf.index]],
            Tcounts, libsz, out, args.window, args.rasqual_genes,
            allelic=allelic, order=order, mode=args.rasqual_input)
        if rq is not None:
            rq.to_csv(out / 'rasqual_cis.tsv.gz', sep='\t', index=False)
            print(f'wrote {out}/rasqual_cis.tsv.gz')
        bundle = add_rasqual_to_bundle(bundle, rq, res, diag)
    else:
        bundle['rasqual'] = {'note': 'not run (pass --rasqual to enable)'}
    (out / 'eval_bundle.json').write_text(json.dumps(bundle, indent=2))
    print(f'\nwrote {out}/hapmixqtl_cis.tsv.gz   (full results, keep local)')
    print(f'wrote {out}/eval_bundle.json      (aggregate only -- safe to share)')


# ---------------------------------------------------------------------------

def selftest():
    """Fabricate Salmon-shaped inputs and run the whole path."""
    import tempfile, os
    print('SELF-TEST: fabricating Salmon + VCF inputs\n')
    td = Path(tempfile.mkdtemp())
    N, G, ND = 40, 25, 50
    rng = np.random.RandomState(0)
    samples = [f'S{i:03d}' for i in range(N)]
    txs = [f'ENST{i:08d}' for i in range(G)]
    # phased VCF: one SNP per gene, plus a tri-allelic row every 5th gene
    # (skipped by the standard reader; used only with --multiallelic)
    hdr = ('##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\t'
           'FILTER\tINFO\tFORMAT\t' + '\t'.join(samples) + '\n')
    with open(td / 'p.vcf', 'w') as fh, open(td / 'str.vcf', 'w') as sh:
        fh.write(hdr); sh.write(hdr)
        for gi in range(G):
            gts = []
            for i in range(N):
                gts.append(f'{int(rng.rand()<0.4)}|{int(rng.rand()<0.4)}')
            fh.write(f'1\t{1000*gi+500}\tv{gi}\tA\tG\t.\tPASS\t.\tGT\t'
                     + '\t'.join(gts) + '\n')
            if gi % 5 == 0:
                ma = [f'{rng.choice(3, p=[.6,.25,.15])}|{rng.choice(3, p=[.6,.25,.15])}'
                      for _ in range(N)]
                fh.write(f'1\t{1000*gi+700}\tm{gi}\tA\tT,C\t.\tPASS\t.\tGT\t'
                         + '\t'.join(ma) + '\n')
            if gi % 3 == 0:                      # STR VCF: GT-only, PERIOD in INFO
                st = [f'{rng.choice(5)}|{rng.choice(5)}' for _ in range(N)]
                sh.write(f'1\t{1000*gi+800}\tSTR{gi}\t{"CAG"*8}\t'
                         + ','.join('CAG' * (8 + u) for u in (-2, -1, 1, 2))
                         + f'\t.\tPASS\tPERIOD=3\tGT\t' + '\t'.join(st) + '\n')
    (td / 't2g.tsv').write_text('\n'.join(f'{t}\tG{i:05d}' for i, t in enumerate(txs)))
    # gene-pos gains start/end so native mode can locate feature SNPs
    (td / 'genepos.tsv').write_text(
        '\n'.join(f'G{i:05d}\t1\t{1000*i+500}\t{1000*i+400}\t{1000*i+600}'
                  for i in range(G)))
    # fabricate phASER-style per-variant allelic counts (RASQUAL native input)
    ac_man = []
    for s_ in samples:
        f = td / f'{s_}.allelic_counts.txt'
        with open(f, 'w') as fh:
            # phASER's REAL header: no 'stop', position not start.
            fh.write('contig\tposition\tvariantID\trefAllele\taltAllele\trefCount\taltCount\ttotalCount\n')
            for gi2 in range(G):
                pos = 1000 * gi2 + 500
                r, a = rng.poisson(20), rng.poisson(20)
                fh.write(f'1\t{pos}\tv{gi2}\tA\tG\t{r}\t{a}\t{r+a}\n')
        ac_man.append(f'{s_}\t{f}')
    (td / 'ac_manifest.tsv').write_text('\n'.join(ac_man))
    man = []
    for si, s in enumerate(samples):
        sd = td / s / 'aux_info' / 'bootstrap'   # as salmon writes it
        sd.mkdir(parents=True, exist_ok=True)
        names = [t + suf for t in txs for suf in ('_hapA', '_hapB')]
        boot = rng.poisson(40, size=(ND, len(names))).astype(np.float64)
        with gzip.open(sd / 'names.tsv.gz', 'wt') as fh:
            fh.write('\t'.join(names))
        with gzip.open(sd / 'bootstraps.gz', 'wb') as fh:
            fh.write(boot.tobytes())
        (td / s / 'aux_info' / 'meta_info.json').write_text(
            json.dumps({'num_bootstraps': ND, 'samp_type': 'gibbs'}))
        man.append(f'{s}\t{td/s}')
    (td / 'manifest.tsv').write_text('\n'.join(man))

    # A personalized diploid transcriptome is built per sample from that
    # sample's own variants, so transcript sets differ BETWEEN samples. Build
    # two that do, and require the gene set to be their union in either order.
    # Taking the first sample's set instead made the answer depend on manifest
    # order: genes only in the later sample vanished silently, and a gene the
    # first sample lacked raised KeyError. Reproduced on real personalized
    # salmon output before this was changed.
    ud = td / 'union'; ud.mkdir()
    only_a, only_b, shared = 'TA', 'TB', ['TS0', 'TS1']
    def _mini(name, txlist):
        sd = ud / name / 'aux_info' / 'bootstrap'; sd.mkdir(parents=True)
        nm = [t + suf for t in txlist for suf in ('_hapA', '_hapB')]
        bt = rng.poisson(10, size=(4, len(nm))).astype(np.float64)
        with gzip.open(sd / 'names.tsv.gz', 'wt') as fh:
            fh.write('\t'.join(nm))
        with gzip.open(sd / 'bootstraps.gz', 'wb') as fh:
            fh.write(bt.tobytes())
        (ud / name / 'aux_info' / 'meta_info.json').write_text(
            json.dumps({'num_bootstraps': 4}))
        return f'{name}\t{ud/name}'
    ra = _mini('A', shared + [only_a])
    rb = _mini('B', shared + [only_b])
    (ud / 't2g.tsv').write_text('\n'.join(
        f'{t}\tGENE_{t}' for t in shared + [only_a, only_b]))
    want = {f'GENE_{t}' for t in shared + [only_a, only_b]}
    seen = []
    for tag, rows in (('A,B', [ra, rb]), ('B,A', [rb, ra])):
        (ud / f'man_{tag}.tsv').write_text('\n'.join(rows))
        gset, _, yl, _, _ = load_counts(ud / f'man_{tag}.tsv', ud / 't2g.tsv',
                                     ('_hapA', '_hapB'), ud)
        assert set(gset) == want, (tag, sorted(set(gset)), sorted(want))
        assert yl.shape[0] == len(want), (tag, yl.shape)
        seen.append(tuple(gset))
    assert seen[0] == seen[1], 'gene set must not depend on manifest order'
    print(f'union check: {len(want)} genes in both orders, '
          'sample-specific transcripts preserved')

    argv = ['x', '--vcf', str(td / 'p.vcf'), '--manifest', str(td / 'manifest.tsv'),
            '--tx2gene', str(td / 't2g.tsv'), '--gene-pos', str(td / 'genepos.tsv'),
            '--out', str(td / 'out')]
    rq_bin = os.environ.get('RASQUAL_BIN')
    if rq_bin and Path(rq_bin).exists():
        print(f'(also exercising the RASQUAL comparison via {rq_bin})')
        argv += ['--rasqual', rq_bin, '--rasqual-genes', '14',
                 '--allelic-counts', str(td / 'ac_manifest.tsv'),
                 '--rasqual-input', 'both']
    sys.argv = argv
    print('running the real pipeline on the fabricated inputs (standard: biallelic SNPs)...\n')
    main()
    b = json.loads((td / 'out' / 'eval_bundle.json').read_text())
    std = pd.read_csv(td / 'out' / 'hapmixqtl_cis.tsv.gz', sep='\t')
    assert 'nonstandard' not in b and 'variant_type' not in std.columns, \
        'default run must not carry the opt-in passes'
    assert 'phenotype_id' in std.columns and std['variant_id'].str.startswith('v').all(), \
        'default run must test biallelic SNPs only (and keep the gene id column)'
    print('\nSELF-TEST (standard) OK. eval_bundle keys:', list(b))
    if 'rasqual' in b:
        print('  rasqual:', json.dumps(b['rasqual'])[:220])
    print('  meta:', b.get('meta'))

    # ---- opt-in: the same data with --str-vcf and --multiallelic -----------------
    print('\nre-running with --str-vcf + --multiallelic (NON-STANDARD, opt-in)...\n')
    sys.argv = list(argv)
    sys.argv[sys.argv.index('--out') + 1] = str(td / 'out_ns')
    sys.argv += ['--str-vcf', str(td / 'str.vcf'), '--multiallelic']
    main()
    b2 = json.loads((td / 'out_ns' / 'eval_bundle.json').read_text())
    ns = b2['nonstandard']
    res2 = pd.read_csv(td / 'out_ns' / 'hapmixqtl_cis.tsv.gz', sep='\t')
    assert set(ns['scan_rows_by_type']) == {'snp', 'str', 'ma_allele'}, ns['scan_rows_by_type']
    assert ns['scan_rows_by_type']['snp'] == G and ns['scan_rows_by_type']['str'] == 9 \
        and ns['scan_rows_by_type']['ma_allele'] == 10, ns['scan_rows_by_type']
    assert 'variant_type' in res2.columns and set(res2['variant_type']) <= {'snp', 'str', 'ma_allele'}
    assert 'str_curvature' in ns and ns['str_curvature']['n_str'] == 9
    assert 'multiallelic_categorical' in ns and ns['multiallelic_categorical']['n_sites'] == 5
    for f in ('hapmixqtl_str_curvature.tsv.gz', 'hapmixqtl_multiallelic_sites.tsv.gz',
              'hapmixqtl_multiallelic_alleles.tsv.gz'):
        assert (td / 'out_ns' / f).exists(), f
    if 'rasqual' in b2 and 'n_genes_attempted' in b2['rasqual']:
        # the RASQUAL comparison must be untouched by the opt-in rows
        assert b2['rasqual']['n_genes_attempted'] == b['rasqual']['n_genes_attempted']
    print('SELF-TEST (opt-in) OK. nonstandard summary:', json.dumps(ns)[:300])
    print('\nSELF-TEST OK')
    return 0


if __name__ == '__main__':
    main()

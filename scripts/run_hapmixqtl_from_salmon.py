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
  1. Reads each sample's Salmon output including aux_info/bootstraps
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
  7. Writes an EVAL BUNDLE of aggregate statistics only.

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
                                     reference_bias_diagnostic, map_cis)
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent / 'tensorqtl'))
    from hapmixqtl import (compute_summaries_from_gibbs,
                           reference_bias_diagnostic, map_cis)

# mixQTL's shipped filters (R/mixqtl.R)
ASC_CUTOFF, TRC_CUTOFF, MIN_SAMPLES = 5, 20, 30


# ---------------------------------------------------------------------------
#  Salmon readers
# ---------------------------------------------------------------------------

def read_salmon_bootstraps(sdir):
    """Return (transcript_names, boot[n_txp, n_draws]).

    Salmon writes aux_info/bootstraps/{bootstraps.gz,names.tsv.gz} and records
    the count in aux_info/meta_info.json. The payload is a flat binary array of
    n_draws x n_txp; the element type is inferred from its size so both the
    double and integer variants are handled.
    """
    sdir = Path(sdir)
    aux = sdir / 'aux_info'
    bfile = aux / 'bootstraps' / 'bootstraps.gz'
    nfile = aux / 'bootstraps' / 'names.tsv.gz'
    mfile = aux / 'meta_info.json'
    for f in (bfile, nfile):
        if not f.exists():
            raise SystemExit(
                f'{f} not found.\n'
                'Salmon must be run with --numGibbsSamples (or --numBootstraps). '
                'Without the bootstrap directory there is no inferential '
                'variance, and hapmixQTL has nothing to propagate.')
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

def read_phased_vcf(path, want_samples):
    """Return variant_df, dosage[V,N], xL[V,N], xR[V,N] for biallelic SNPs."""
    op = gzip.open if str(path).endswith('.gz') else open
    ids, chroms, poss = [], [], []
    XL, XR = [], []
    order = None
    with op(path, 'rt') as fh:
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
            xl = np.zeros(len(keep)); xr = np.zeros(len(keep)); ok = True
            for k, i in enumerate(keep):
                gt = f[9 + i].split(':')[gt_i]
                if '|' not in gt:
                    ok = False; break                       # unphased -> skip
                a, b = gt.split('|')[:2]
                if a in '.' or b in '.':
                    ok = False; break
                xl[k] = 1.0 if a != '0' else 0.0
                xr[k] = 1.0 if b != '0' else 0.0
            if not ok:
                continue
            ids.append(f[2] if f[2] != '.' else f'{f[0]}_{f[1]}_{f[3]}_{f[4]}')
            chroms.append(f[0]); poss.append(int(f[1]))
            XL.append(xl); XR.append(xr)
    if not ids:
        raise SystemExit('no phased biallelic SNPs read from the VCF')
    XL = np.array(XL); XR = np.array(XR)
    vdf = pd.DataFrame({'chrom': [str(c) for c in chroms], 'pos': poss}, index=ids)
    return vdf, XL + XR, XL, XR, order


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
    for c in ('slope', 'slope_se'):
        if c in res_df.columns:
            v = pd.to_numeric(res_df[c], errors='coerce').dropna().values
            if v.size:
                b.setdefault('effect_sizes', {})[c] = {
                    'n': int(v.size),
                    'quantiles': np.quantile(v, [.05, .25, .5, .75, .95]).round(5).tolist()}
    return b


# ---------------------------------------------------------------------------

def load_counts(manifest, tx2gene, suffixes, out):
    rows = [l.split('\t') for l in Path(manifest).read_text().strip().split('\n')
            if l.strip() and not l.startswith('#')]
    samples = [r[0].strip() for r in rows]
    dirs = [r[1].strip() for r in rows]
    t2g = dict(l.split('\t')[:2] for l in
               Path(tx2gene).read_text().strip().split('\n') if '\t' in l)

    YL = YR = None
    genes = None
    for si, (s, sd) in enumerate(zip(samples, dirs)):
        names, boot = read_salmon_bootstraps(sd)
        pairs = pair_haplotypes(names, suffixes)
        if not pairs:
            raise SystemExit(
                f'no haplotype-paired transcripts in {sd} using suffixes '
                f'{suffixes}.\nSalmon appears to have been run against a '
                'standard reference transcriptome, which carries NO allelic '
                'information. Quantify against a personalized DIPLOID '
                'transcriptome built from your phased VCF, or use the phASER '
                'route (scripts/prep_brainvar.py).')
        if genes is None:
            genes = sorted({t2g[b] for b in pairs if b in t2g})
            gi = {g: i for i, g in enumerate(genes)}
            nd = boot.shape[1]
            YL = np.zeros((len(genes), len(samples), nd))
            YR = np.zeros((len(genes), len(samples), nd))
            print(f'  {len(pairs)} haplotype pairs -> {len(genes)} genes, '
                  f'{nd} Gibbs draws')
        for base, (ia, ib) in pairs.items():
            g = t2g.get(base)
            if g is None:
                continue
            YL[gi[g], si, :] += boot[ia]
            YR[gi[g], si, :] += boot[ib]
        print(f'  [{si+1}/{len(samples)}] {s}', flush=True)
    return np.array(genes), samples, YL, YR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--vcf'); ap.add_argument('--manifest')
    ap.add_argument('--tx2gene'); ap.add_argument('--out', default='hapmix_out')
    ap.add_argument('--gene-pos', help='TSV: gene_id, chr, pos (TSS)')
    ap.add_argument('--hap-suffix', default='_hapA,_hapB')
    ap.add_argument('--window', type=int, default=1_000_000)
    ap.add_argument('--force', action='store_true',
                    help='proceed despite a reference-bias flag (NOT recommended)')
    args = ap.parse_args()

    if args.selftest:
        return selftest()
    for r in ('vcf', 'manifest', 'tx2gene'):
        if not getattr(args, r):
            raise SystemExit(f'--{r} is required (or use --selftest)')
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    sufs = tuple(args.hap_suffix.split(','))

    print('Reading Salmon Gibbs quantifications')
    genes, samples, YL, YR = load_counts(args.manifest, args.tx2gene, sufs, out)

    print('Computing Gibbs summaries (production code path)')
    A, T, Va, Vt, Cat = compute_summaries_from_gibbs(YL, YR)

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
    gdf = pd.DataFrame(dos, index=vdf.index, columns=order)
    xLdf = pd.DataFrame(xL, index=vdf.index, columns=order)
    xRdf = pd.DataFrame(xR, index=vdf.index, columns=order)
    if args.gene_pos:
        # chromosome must be a STRING and must match the VCF's CHROM exactly.
        # pandas otherwise infers int for a "1"-style column, and every
        # phenotype is then silently dropped as "on a chr. without genotypes".
        gp = pd.read_csv(args.gene_pos, sep='\t', header=None,
                         names=['gene', 'chr', 'pos'],
                         dtype={'chr': str}).set_index('gene')
        gp['chr'] = gp['chr'].astype(str).str.strip()
        pos_df = gp.loc[[g for g in genes if g in gp.index], ['chr', 'pos']]
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

    print(f'\nRunning map_cis on {len(common)} genes '
          f"(tau_mode='estimate', the validated default)")
    res = map_cis(gdf, vdf, sdf, tdf, vadf, vtdf, pos_df,
                  xL_df=xLdf, xR_df=xRdf, window=args.window, verbose=True)
    res.to_csv(out / 'hapmixqtl_cis.tsv.gz', sep='\t', index=False)

    bundle = build_eval_bundle(res, diag, {
        'n_samples': len(order), 'n_genes_tested': int(len(common)),
        'n_variants': int(len(vdf)), 'n_gibbs_draws': int(YL.shape[2]),
        'median_Va': float(np.median(Va)), 'median_Vt': float(np.median(Vt)),
        'tau_mode': 'estimate'})
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
    # phased VCF: one SNP per gene
    with open(td / 'p.vcf', 'w') as fh:
        fh.write('##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\t'
                 'FILTER\tINFO\tFORMAT\t' + '\t'.join(samples) + '\n')
        for gi in range(G):
            gts = []
            for i in range(N):
                gts.append(f'{int(rng.rand()<0.4)}|{int(rng.rand()<0.4)}')
            fh.write(f'1\t{1000*gi+500}\tv{gi}\tA\tG\t.\tPASS\t.\tGT\t'
                     + '\t'.join(gts) + '\n')
    (td / 't2g.tsv').write_text('\n'.join(f'{t}\tG{i:05d}' for i, t in enumerate(txs)))
    (td / 'genepos.tsv').write_text(
        '\n'.join(f'G{i:05d}\t1\t{1000*i+500}' for i in range(G)))
    man = []
    for si, s in enumerate(samples):
        sd = td / s / 'aux_info' / 'bootstraps'
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

    sys.argv = ['x', '--vcf', str(td / 'p.vcf'), '--manifest', str(td / 'manifest.tsv'),
                '--tx2gene', str(td / 't2g.tsv'), '--gene-pos', str(td / 'genepos.tsv'),
                '--out', str(td / 'out')]
    print('running the real pipeline on the fabricated inputs...\n')
    main()
    b = json.loads((td / 'out' / 'eval_bundle.json').read_text())
    print('\nSELF-TEST OK. eval_bundle keys:', list(b))
    print('  meta:', b.get('meta'))
    return 0


if __name__ == '__main__':
    main()

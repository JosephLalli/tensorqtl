#!/usr/bin/env python3
"""
Which pipeline would you deploy?  RASQUAL vs hapmixQTL, each at its best.

THE QUESTION THIS ANSWERS
=========================
Not "which statistical model is better" (that needs identical inputs -- see the
2x2 in run_hapmixqtl_from_salmon.py --rasqual-input both). This answers the
practical question: on the same samples, genes, tested variants and phase, with
each method fed its NATIVE input, which one finds more true eQTLs at the same
false-positive rate, and do they agree on effect sizes?

  RASQUAL    phASER per-feature-SNP allele counts + total counts   (native)
  hapmixQTL  Salmon diploid Gibbs draws (quantifier uncertainty)   (native)

Shared across both, so the comparison is of the METHODS and not the setup:
  * the same samples and gene set
  * the same tested cis variants: those in the window and OUTSIDE the gene body
  * the same phase -- rephased.vcf.gz from phaser_to_matrix.py, so hapmixQTL's
    s = xL - xR is consistent with phASER's counts and RASQUAL is not handicapped
  * the same permutation null (below)

THE PERMUTATION NULL, AND WHY IT IS BUILT THE WAY IT IS
=======================================================
Power must be compared at MATCHED EMPIRICAL false-positive rate. On nominal
p-values an anticonservative method always "wins" (docs/ase_validation.md sec 5;
tau_mode='zero' looked best of all before it was thresholded on its own null and
collapsed to total-only). RASQUAL's own paper uses the same permutation metric.

But the usual cis-QTL permutation -- shuffle expression across samples -- BREAKS
RASQUAL. Its allele counts are measured at each sample's OWN heterozygous feature
SNPs. Shuffling would pair sample i's counts with sample j's genotypes, giving
RASQUAL a 50/50 split at a site it believes homozygous; its model reads that as
gross sequencing error and its genotype correction flips the call. The null is
then not a null.

So the permutation here shuffles only the TESTED REGULATORY VARIANTS across
samples (the whole cis-window block as one unit, preserving LD among them), and
holds each sample's feature-SNP genotypes and allele counts fixed with its
expression. That breaks exactly the regulatory link and nothing else. The same
permutation vector is applied to both methods. It is also why tested variants
are restricted to OUTSIDE the gene body: feature SNPs must stay unpermuted, so
they cannot also be tested.

WHAT IS REPORTED  (all aggregate; shareable under a DUA)
=======================================================
  calibration     lambda_GC and type-I at 0.05 on the pooled permuted null
  power           fraction of genes discovered at empirical FPR 10% and 5%,
                  thresholded on each method's OWN null
  effect sizes    RASQUAL pi -> log aFC = log(pi/(1-pi)); hapmixQTL slope IS
                  log aFC. Regressed across genes: slope 1 = same quantity.
  concordance     Spearman of gene statistics; overlap of top-k
  replication     with --known-egenes: of each method's discoveries, the
                  fraction that are published eGenes -- the external truth
  compute         wall time per method

Run:
  RASQUAL_BIN=.../rasqual python3 scripts/compare_pipelines.py --selftest
  python3 scripts/compare_pipelines.py \\
      --vcf prepped/rephased.vcf.gz --genes genes.tsv \\
      --salmon salmon.tsv --tx2gene tx2gene.tsv \\
      --allelic-counts prepped/allelic_counts_manifest.tsv \\
      --rasqual rasqual_src/src/rasqual \\
      --n-genes 300 --n-perm 10 --out deploy/
"""

import argparse
import contextlib
import gzip
import io
import json
import os
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))

from scipy import stats
import run_hapmixqtl_from_salmon as H          # readers, native RASQUAL VCF
from make_rasqual_inputs import size_factors, read_salmon_totals

try:
    from tensorqtl.hapmixqtl import compute_summaries_from_gibbs, map_cis
except ImportError:
    sys.path.insert(0, str(HERE.parent / 'tensorqtl'))
    from hapmixqtl import compute_summaries_from_gibbs, map_cis


# ---------------------------------------------------------------------------
#  hapmixQTL arm: gene-level lead statistic, given a genotype permutation
# ---------------------------------------------------------------------------

def hapmix_arm(A, T, Va, Vt, genes, order, vdf, dos, xL, xR, pos_df, window,
               tested_mask, perm=None):
    """Return DataFrame(gene, stat, log_afc). stat = chi2(1)-equivalent of the
    lead nominal p over TESTED variants, so it sits on RASQUAL's scale."""
    idx = np.where(tested_mask)[0]
    v = vdf.iloc[idx]
    cols = order if perm is None else [order[i] for i in perm]
    # genotype of sample perm[k] goes with expression of sample k
    g = pd.DataFrame(dos[idx], index=v.index, columns=cols)[order] if perm is not None \
        else pd.DataFrame(dos[idx], index=v.index, columns=order)
    xl = pd.DataFrame(xL[idx], index=v.index, columns=cols)[order] if perm is not None \
        else pd.DataFrame(xL[idx], index=v.index, columns=order)
    xr = pd.DataFrame(xR[idx], index=v.index, columns=cols)[order] if perm is not None \
        else pd.DataFrame(xR[idx], index=v.index, columns=order)
    if perm is not None:
        # relabel so column k carries sample perm[k]'s genotype under sample k's name
        for df_ in (g, xl, xr):
            df_.columns = order
    mk = lambda M: pd.DataFrame(M, index=genes, columns=order)
    with contextlib.redirect_stdout(io.StringIO()):
        res = map_cis(g, v[['chrom', 'pos']], mk(A), mk(T), mk(Va), mk(Vt),
                      pos_df, xL_df=xl, xR_df=xr, window=window, nperm=50,
                      verbose=False)
    p = pd.to_numeric(res['pval_nominal'], errors='coerce').clip(1e-300, 1)
    # map_cis returns the gene ID as the INDEX (named phenotype_id), not a column
    return pd.DataFrame({'gene': res.index.values,
                         'stat': stats.chi2.isf(p.values, 1),
                         'log_afc': pd.to_numeric(res['slope'], errors='coerce').values})


# ---------------------------------------------------------------------------
#  RASQUAL arm: native per-fSNP counts, tested rSNPs permuted, fSNPs fixed
# ---------------------------------------------------------------------------

def rasqual_arm(binary, genes, pos_df, vdf, xL, xR, allelic, order, Y, K,
                window, perm=None, tmp=None):
    N = len(order)
    td = Path(tmp or tempfile.mkdtemp())
    np.asarray(Y, np.float64).tofile(td / 'Y.bin')
    np.asarray(K, np.float64).tofile(td / 'K.bin')
    pos = vdf['pos'].values
    recs = []
    for j, g in enumerate(genes):
        row = pos_df.loc[g]
        chrom, tss = str(row['chr']), int(row['pos'])
        gs, ge = int(row['start']), int(row['end'])
        same = vdf['chrom'].values == chrom
        fidx = [v for v in np.where(same & (pos >= gs) & (pos <= ge))[0]
                if (chrom, int(pos[v])) in allelic]
        ridx = np.where(same & (np.abs(pos - tss) <= window)
                        & ~((pos >= gs) & (pos <= ge)))[0]
        if not fidx or ridx.size == 0:
            recs.append(dict(gene=g, stat=np.nan, log_afc=np.nan, phi=np.nan,
                             status='no_fsnp_or_rsnp'))
            continue
        lines = []
        for v in fidx:                                  # fSNPs: never permuted
            cnt = allelic[(chrom, int(pos[v]))]
            fl = [f'{int(xL[v,k])}|{int(xR[v,k])}:{cnt.get(s,(0,0))[0]},'
                  f'{cnt.get(s,(0,0))[1]}' for k, s in enumerate(order)]
            lines.append('\t'.join([chrom, str(int(pos[v])), f'f_{vdf.index[v]}',
                                    'A', 'G', '100', 'PASS', 'RSQ=1.0', 'GT:AS'] + fl))
        src = np.arange(N) if perm is None else np.asarray(perm)
        for v in ridx:                                  # rSNPs: permuted
            fl = [f'{int(xL[v,src[k]])}|{int(xR[v,src[k]])}:0,0' for k in range(N)]
            lines.append('\t'.join([chrom, str(int(pos[v])), str(vdf.index[v]),
                                    'A', 'G', '100', 'PASS', 'RSQ=1.0', 'GT:AS'] + fl))
        cmd = [binary, '-y', str(td / 'Y.bin'), '-k', str(td / 'K.bin'),
               '-n', str(N), '-j', str(j + 1), '-l', str(len(lines)),
               '-m', str(len(fidx)), '-s', str(gs), '-e', str(ge),
               '-f', str(g), '-z']
        try:
            pr = subprocess.run(cmd, input='\n'.join(lines) + '\n',
                                capture_output=True, text=True, timeout=900)
        except Exception as e:
            recs.append(dict(gene=g, stat=np.nan, log_afc=np.nan, phi=np.nan,
                             status=f'error:{type(e).__name__}'))
            continue
        best = None
        for ln in pr.stdout.strip().split('\n'):
            f = ln.split('\t')
            if len(f) < 25 or f[1] == 'SKIPPED' or f[1].startswith('f_'):
                continue                                # lead over rSNPs only
            try:
                if int(float(f[22])) != 0:
                    continue
                chi2, pi = float(f[10]), float(f[11])
            except ValueError:
                continue
            if best is None or chi2 > best['stat']:
                pi = min(max(pi, 1e-6), 1 - 1e-6)
                best = dict(gene=g, stat=chi2, log_afc=np.log(pi / (1 - pi)),
                            phi=float(f[13]), status='ok')
        recs.append(best or dict(gene=g, stat=np.nan, log_afc=np.nan,
                                 phi=np.nan, status='no_converged_row'))
    return pd.DataFrame(recs)


# ---------------------------------------------------------------------------
#  Scoring
# ---------------------------------------------------------------------------

def score(obs, null, name, known=None):
    """obs/null: DataFrames with gene, stat. null pooled over permutations."""
    o = obs.dropna(subset=['stat']); n = null.dropna(subset=['stat'])
    out = {'n_genes_scored': int(len(o)), 'n_null_stats': int(len(n))}
    if len(n) < 20 or len(o) < 5:
        out['note'] = 'too few statistics'; return out
    ns = n['stat'].values
    # calibration: the null statistic's own distribution vs chi2(1)
    pn = stats.chi2.sf(ns, 1)
    out['calibration'] = {
        'lambda_gc_null': float(np.median(ns) / stats.chi2.ppf(0.5, 1)),
        'typeI_0.05_nominal_on_null': float(np.mean(pn < 0.05)),
        'note': ('lead-SNP statistics are max-over-window, so lambda > 1 is '
                 'expected for BOTH methods; compare between methods, not to 1')}
    # power at matched empirical FPR, thresholded on THIS method's own null
    out['power'] = {}
    for fpr in (0.10, 0.05):
        thr = float(np.quantile(ns, 1 - fpr))
        out['power'][f'fpr_{fpr}'] = {
            'threshold': thr,
            'discovered_fraction': float(np.mean(o['stat'].values > thr)),
            'n_discovered': int(np.sum(o['stat'].values > thr))}
    if known is not None:
        thr = out['power']['fpr_0.1']['threshold']
        disc = set(o[o['stat'] > thr]['gene'])
        if disc:
            out['replication'] = {
                'n_discovered': len(disc),
                'frac_known_egene': float(np.mean([g in known for g in disc])),
                'baseline_frac_known_in_tested': float(np.mean(
                    [g in known for g in o['gene']]))}
    return out


def compare(obs_r, obs_h, out):
    m = obs_r.merge(obs_h, on='gene', suffixes=('_r', '_h')).dropna(
        subset=['stat_r', 'stat_h'])
    res = {'n_genes_both': int(len(m))}
    if len(m) < 10:
        res['note'] = 'too few genes in both'; return res
    rho, _ = stats.spearmanr(m['stat_r'], m['stat_h'])
    res['spearman_stat'] = float(rho)
    k = max(5, len(m) // 10)
    top_r = set(m.nlargest(k, 'stat_r')['gene']); top_h = set(m.nlargest(k, 'stat_h')['gene'])
    res['top_k'] = k; res['top_k_overlap'] = len(top_r & top_h) / k
    e = m.dropna(subset=['log_afc_r', 'log_afc_h'])
    e = e[np.isfinite(e['log_afc_r']) & np.isfinite(e['log_afc_h'])]
    if len(e) >= 10:
        sl, ic, r, _, se = stats.linregress(e['log_afc_r'], e['log_afc_h'])
        res['effect_size'] = {
            'n': int(len(e)), 'slope_hapmix_on_rasqual': float(sl),
            'slope_se': float(se), 'intercept': float(ic), 'r': float(r),
            'note': ('both on the log-aFC scale (RASQUAL pi -> log(pi/(1-pi))); '
                     'slope 1 means the same quantity, not just the same ranking')}
    return res


# ---------------------------------------------------------------------------

def run(args):
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(args.seed)
    sufs = tuple(args.hap_suffix.split(','))

    print('hapmixQTL input: Salmon diploid Gibbs')
    genes_all, samples, YL, YR = H.load_counts(args.salmon, args.tx2gene, sufs, out)
    A, T, Va, Vt, _ = compute_summaries_from_gibbs(YL, YR)

    print('Genotypes (use rephased.vcf.gz from phaser_to_matrix.py)')
    vdf, dos, xL, xR, order = H.read_phased_vcf(args.vcf, set(samples))
    keep = [samples.index(s) for s in order]
    A, T, Va, Vt = A[:, keep], T[:, keep], Va[:, keep], Vt[:, keep]
    vdf['chrom'] = vdf['chrom'].astype(str)

    print('RASQUAL input: phASER per-feature-SNP counts')
    allelic = H.load_allelic_counts(args.allelic_counts, order)

    gp = pd.read_csv(args.genes, sep='\t', header=None, dtype={1: str})
    gp.columns = ['gene', 'chr', 'start', 'end', 'pos'][:gp.shape[1]]
    if 'pos' not in gp:
        gp['pos'] = gp['start']
    gp = gp.set_index('gene'); gp['chr'] = gp['chr'].astype(str).str.strip()
    genes_all = list(genes_all)
    gi_all = {g: i for i, g in enumerate(genes_all)}
    # usable: in both inputs, with >= 1 fSNP carrying counts
    pos = vdf['pos'].values
    usable = []
    for g in gp.index:
        if g not in gi_all:
            continue
        r = gp.loc[g]; same = vdf['chrom'].values == str(r['chr'])
        body = np.where(same & (pos >= int(r['start'])) & (pos <= int(r['end'])))[0]
        if any((str(r['chr']), int(pos[v])) in allelic for v in body):
            usable.append(g)
    if len(usable) > args.n_genes:
        usable = list(rng.choice(usable, args.n_genes, replace=False))
    print(f'  {len(usable)} genes usable by both (of {len(gp)}); '
          f'{len(order)} samples')
    if len(usable) < 5:
        raise SystemExit('too few genes usable by both methods')
    sel = [gi_all[g] for g in usable]
    A, T, Va, Vt = A[sel], T[sel], Va[sel], Vt[sel]
    pos_df = gp.loc[usable]
    # tested variants: in some gene's window AND outside every selected gene body
    in_body = np.zeros(len(vdf), bool); in_win = np.zeros(len(vdf), bool)
    for g in usable:
        r = gp.loc[g]; same = vdf['chrom'].values == str(r['chr'])
        in_body |= same & (pos >= int(r['start'])) & (pos <= int(r['end']))
        in_win |= same & (np.abs(pos - int(r['pos'])) <= args.window)
    tested = in_win & ~in_body
    print(f'  {int(tested.sum())} tested cis variants (window, outside gene bodies)')

    # RASQUAL total counts and offsets follow the expression (never permuted)
    Ytot = read_salmon_totals(args.salmon, args.tx2gene, order, sufs)
    Ytot = Ytot.reindex(usable).fillna(0.0).values
    K = np.outer(Ytot.mean(1), size_factors(Ytot))

    known = None
    if args.known_egenes:
        known = set(l.strip() for l in open(args.known_egenes) if l.strip())

    def both(perm, tag):
        t0 = time.time()
        h = hapmix_arm(A, T, Va, Vt, usable, order, vdf, dos, xL, xR,
                       pos_df[['chr', 'pos']], args.window, tested, perm)
        th = time.time() - t0; t0 = time.time()
        r = rasqual_arm(args.rasqual, usable, pos_df, vdf, xL, xR, allelic,
                        order, Ytot, K, args.window, perm)
        tr = time.time() - t0
        print(f'  {tag}: hapmixQTL {th:.0f}s, RASQUAL {tr:.0f}s '
              f'({int((r["status"]=="ok").sum())}/{len(r)} converged)', flush=True)
        return h, r, th, tr

    print('\nObserved')
    obs_h, obs_r, th, tr = both(None, 'observed')
    nulls_h, nulls_r = [], []
    for p in range(args.n_perm):
        perm = rng.permutation(len(order))
        h, r, _, _ = both(perm, f'perm {p+1}/{args.n_perm}')
        nulls_h.append(h); nulls_r.append(r)
    null_h = pd.concat(nulls_h) if nulls_h else obs_h.iloc[0:0]
    null_r = pd.concat(nulls_r) if nulls_r else obs_r.iloc[0:0]

    result = {
        'design': {'question': 'which pipeline would you deploy',
                   'rasqual_input': 'phASER per-fSNP counts (native)',
                   'hapmixqtl_input': 'Salmon diploid Gibbs (native)',
                   'shared': ['samples', 'genes', 'tested variants outside gene '
                              'bodies', 'phase (rephased.vcf.gz)', 'permutation'],
                   'null': 'tested rSNPs permuted across samples as a block; '
                           'fSNP genotypes + allele counts fixed with expression',
                   'n_genes': len(usable), 'n_samples': len(order),
                   'n_tested_variants': int(tested.sum()), 'n_perm': args.n_perm,
                   'window': args.window, 'seed': args.seed},
        'compute_seconds_observed': {'hapmixQTL': th, 'RASQUAL': tr},
        'hapmixQTL': score(obs_h, null_h, 'hapmixQTL', known),
        'RASQUAL': score(obs_r, null_r, 'RASQUAL', known),
        'head_to_head': compare(obs_r, obs_h, out),
    }
    ok = obs_r[obs_r['status'] == 'ok']
    if len(ok):
        result['RASQUAL']['phi_hat'] = {
            'median': float(np.nanmedian(ok['phi'])),
            'quantiles': np.nanquantile(ok['phi'], [.05, .5, .95]).round(4).tolist(),
            'note': 'reference-bias estimate on YOUR data; 0.5 = unbiased'}
    (out / 'deploy_comparison.json').write_text(json.dumps(result, indent=2, default=float))
    obs_h.to_csv(out / 'observed_hapmixqtl.tsv', sep='\t', index=False)
    obs_r.to_csv(out / 'observed_rasqual.tsv', sep='\t', index=False)
    write_table(result, out / 'deploy_comparison.md')
    print(f'\nwrote {out}/deploy_comparison.json  (aggregate -- shareable)')
    print(f'wrote {out}/deploy_comparison.md')
    return result


def write_table(r, path):
    def pw(m, f): return r[m].get('power', {}).get(f'fpr_{f}', {}).get('discovered_fraction', float('nan'))
    L = ['# Deploy comparison: RASQUAL vs hapmixQTL, each on its native input', '',
         f"{r['design']['n_genes']} genes, {r['design']['n_samples']} samples, "
         f"{r['design']['n_tested_variants']} tested variants, {r['design']['n_perm']} permutations", '',
         '| | RASQUAL | hapmixQTL |', '|---|---|---|',
         f"| power @ empirical FPR 10% | {pw('RASQUAL',0.1):.3f} | {pw('hapmixQTL',0.1):.3f} |",
         f"| power @ empirical FPR 5% | {pw('RASQUAL',0.05):.3f} | {pw('hapmixQTL',0.05):.3f} |",
         f"| λ_GC on permuted null | {r['RASQUAL'].get('calibration',{}).get('lambda_gc_null',float('nan')):.2f} | "
         f"{r['hapmixQTL'].get('calibration',{}).get('lambda_gc_null',float('nan')):.2f} |",
         f"| wall time (observed run) | {r['compute_seconds_observed']['RASQUAL']:.0f}s | "
         f"{r['compute_seconds_observed']['hapmixQTL']:.0f}s |"]
    for m in ('RASQUAL', 'hapmixQTL'):
        rep = r[m].get('replication')
        if rep:
            L.append(f"| known-eGene fraction of discoveries ({m}) | "
                     f"{rep['frac_known_egene']:.3f} (baseline {rep['baseline_frac_known_in_tested']:.3f}) | |")
    h = r['head_to_head']
    L += ['', '## Agreement', '',
          f"- Spearman of gene statistics: {h.get('spearman_stat', float('nan')):.3f}",
          f"- top-{h.get('top_k','?')} overlap: {h.get('top_k_overlap', float('nan')):.2f}"]
    if 'effect_size' in h:
        e = h['effect_size']
        L.append(f"- log-aFC slope (hapmixQTL on RASQUAL): {e['slope_hapmix_on_rasqual']:.3f} "
                 f"± {e['slope_se']:.3f}, r = {e['r']:.3f}, n = {e['n']}  (1.0 = same quantity)")
    if 'phi_hat' in r['RASQUAL']:
        L.append(f"- RASQUAL φ̂ median on this data: {r['RASQUAL']['phi_hat']['median']:.3f} (0.5 = no reference bias)")
    Path(path).write_text('\n'.join(L) + '\n')


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--vcf'); ap.add_argument('--genes')
    ap.add_argument('--salmon'); ap.add_argument('--tx2gene')
    ap.add_argument('--allelic-counts'); ap.add_argument('--rasqual')
    ap.add_argument('--known-egenes', help='one published eGene ID per line')
    ap.add_argument('--hap-suffix', default='_hapA,_hapB')
    ap.add_argument('--n-genes', type=int, default=300)
    ap.add_argument('--n-perm', type=int, default=10)
    ap.add_argument('--window', type=int, default=1_000_000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='deploy')
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    for r in ('vcf', 'genes', 'salmon', 'tx2gene', 'allelic_counts', 'rasqual'):
        if not getattr(args, r):
            raise SystemExit(f'--{r.replace("_","-")} is required')
    return run(args)


# ---------------------------------------------------------------------------

def selftest():
    rq = os.environ.get('RASQUAL_BIN')
    if not rq or not Path(rq).exists():
        raise SystemExit('set RASQUAL_BIN to the built rasqual binary')
    td = Path(tempfile.mkdtemp()); rng = np.random.RandomState(0)
    N, G, ND = 40, 16, 40
    samples = [f'S{i:03d}' for i in range(N)]
    txs = [f'ENST{i:08d}' for i in range(G)]
    (td / 't2g.tsv').write_text('\n'.join(f'{t}\tG{i:05d}' for i, t in enumerate(txs)))
    # genes: body [base+400, base+600], TSS base+500; fSNP in body, rSNPs outside
    (td / 'genes.tsv').write_text('\n'.join(
        f'G{i:05d}\t1\t{100000*i+400}\t{100000*i+600}\t{100000*i+500}' for i in range(G)))
    # planted effect in half the genes: rSNP haplotype drives expression
    eff = {i: (1.6 if i % 2 == 0 else 1.0) for i in range(G)}
    h1r = {i: (rng.rand(N) < .4).astype(int) for i in range(G)}
    h2r = {i: (rng.rand(N) < .4).astype(int) for i in range(G)}
    # fSNP in LD with rSNP on the same haplotype
    def ld(h): return np.where(rng.rand(N) < .7, h, (rng.rand(N) < .4).astype(int))
    h1f = {i: ld(h1r[i]) for i in range(G)}; h2f = {i: ld(h2r[i]) for i in range(G)}
    lines = ['##fileformat=VCFv4.2', '#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t' + '\t'.join(samples)]
    for i in range(G):
        base = 100000 * i
        lines.append(f'1\t{base+500}\tf{i}\tA\tG\t.\tPASS\t.\tGT\t' + '\t'.join(
            f'{h1f[i][k]}|{h2f[i][k]}' for k in range(N)))
        for off, tag in ((2000, 'r'), (5000, 'q')):
            lines.append(f'1\t{base+off}\t{tag}{i}\tA\tG\t.\tPASS\t.\tGT\t' + '\t'.join(
                f'{h1r[i][k]}|{h2r[i][k]}' for k in range(N)))
    (td / 'p.vcf').write_text('\n'.join(lines) + '\n')
    man, acman = [], []
    for k, s in enumerate(samples):
        sd = td / s / 'aux_info' / 'bootstraps'; sd.mkdir(parents=True)
        names, boot = [], []
        acf = td / f'{s}.allelic_counts.txt'
        with open(acf, 'w') as fh:
            fh.write('contig\tstart\tstop\tvariantID\trefAllele\taltAllele\trefCount\taltCount\ttotalCount\n')
            for i in range(G):
                e1 = eff[i] if h1r[i][k] else 1.0; e2 = eff[i] if h2r[i][k] else 1.0
                a = rng.poisson(30 * e1); b = rng.poisson(30 * e2)
                names += [txs[i] + '_hapA', txs[i] + '_hapB']
                boot += [rng.poisson(max(a, 1), ND), rng.poisson(max(b, 1), ND)]
                if h1f[i][k] != h2f[i][k]:                 # het fSNP: counts
                    alt = a if h1f[i][k] == 1 else b; ref = a + b - alt
                    fh.write(f'1\t{100000*i+500}\t{100000*i+501}\tf{i}\tA\tG\t{ref}\t{alt}\t{ref+alt}\n')
        with gzip.open(sd / 'names.tsv.gz', 'wt') as fh: fh.write('\t'.join(names))
        with gzip.open(sd / 'bootstraps.gz', 'wb') as fh:
            fh.write(np.array(boot, np.float64).T.tobytes())
        (td / s / 'aux_info' / 'meta_info.json').write_text(json.dumps({'num_bootstraps': ND}))
        pd.DataFrame({'Name': names, 'Length': 1000, 'EffectiveLength': 900, 'TPM': 1.0,
                      'NumReads': [float(x.mean()) for x in boot]}).to_csv(
            td / s / 'quant.sf', sep='\t', index=False)
        man.append(f'{s}\t{td/s}'); acman.append(f'{s}\t{acf}')
    (td / 'salmon.tsv').write_text('\n'.join(man)); (td / 'ac.tsv').write_text('\n'.join(acman))
    (td / 'known.txt').write_text('\n'.join(f'G{i:05d}' for i in range(G) if eff[i] > 1))
    print('SELF-TEST: deploy comparison on fabricated native inputs\n')
    r = run(argparse.Namespace(
        vcf=str(td / 'p.vcf'), genes=str(td / 'genes.tsv'), salmon=str(td / 'salmon.tsv'),
        tx2gene=str(td / 't2g.tsv'), allelic_counts=str(td / 'ac.tsv'), rasqual=rq,
        known_egenes=str(td / 'known.txt'), hap_suffix='_hapA,_hapB',
        n_genes=G, n_perm=2, window=10000, seed=0, out=str(td / 'deploy')))
    print('\n' + (td / 'deploy' / 'deploy_comparison.md').read_text())
    for m in ('RASQUAL', 'hapmixQTL'):
        assert 'power' in r[m], f'{m} produced no power estimate'
    print('SELF-TEST OK')
    return 0


if __name__ == '__main__':
    main()

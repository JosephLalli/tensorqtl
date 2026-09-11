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

NON-STANDARD, OPT-IN (off by default)
=====================================
--str-vcf and/or --multiallelic add a THIRD arm, reported separately as
`hapmixQTL_nonstandard`: hapmixQTL with STRs (per-haplotype repeat length)
and/or multi-ALT split rows among the tested variants, plus the curvature and
categorical second passes, under the same permutation null. RASQUAL cannot
test those variants, so this is not a like-for-like comparison with RASQUAL;
it measures what the extra variant classes add. The standard RASQUAL and
hapmixQTL arms are computed exactly as without the flags (the self-test
asserts they are byte-identical).

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
from str_integrate import parse_str_vcf, parse_multiallelic_vcf, extend_scan   # opt-in arm only

try:
    from tensorqtl.hapmixqtl import (compute_summaries_from_gibbs, map_cis,
                                     reference_bias_diagnostic,
                                     map_str_curvature, map_multiallelic)
except ImportError:
    sys.path.insert(0, str(HERE.parent / 'tensorqtl'))
    from hapmixqtl import (compute_summaries_from_gibbs, map_cis,
                           reference_bias_diagnostic,
                           map_str_curvature, map_multiallelic)


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
                         'log_afc': pd.to_numeric(res['slope'], errors='coerce').values,
                         'lead': res['variant_id'].values})


# ---------------------------------------------------------------------------
#  NON-STANDARD, OPT-IN arm: hapmixQTL with STRs / multi-ALT split rows
#  (off unless --str-vcf / --multiallelic; the standard arms are untouched)
# ---------------------------------------------------------------------------

NONSTANDARD_NOTE = (
    'NON-STANDARD, opt-in. This arm adds STRs (as per-haplotype repeat length) '
    'and/or one split row per ALT of multi-ALT sites to hapmixQTL\'s lead scan, '
    'under the same permutation null and the same outside-gene-body rule. '
    'RASQUAL cannot test these variants, so this is not a like-for-like '
    'comparison with RASQUAL; read it as "what the extra variant classes add". '
    'See scripts/str_integrate.py and docs/ase_validation.md sec 7j.')


def nonstandard_setup(args, vdf, dos, xL, xR, order, gp, usable, window, maf=0.0):
    """Extend the SNP scan with the opt-in rows; apply the tested-variant rule
    (in a window, outside every selected gene body) to the new rows and to
    the second-pass sidecars alike. The MAF floor applies to SNP and split
    rows (0/1/2 dosage); STR rows are exempt, since their dosage is a repeat
    length and they carry their own call-rate and variation filters."""
    print('\nNON-STANDARD, opt-in arm: hapmixQTL with STRs / multi-ALT split rows')
    strs = parse_str_vcf(args.str_vcf, list(order)) if args.str_vcf else []
    ma = parse_multiallelic_vcf(args.vcf, list(order)) if args.multiallelic else []
    vdf_x, dos_x, xL_x, xR_x, vtype, aux = extend_scan(vdf, dos, xL, xR, order, strs, ma)
    pos = vdf_x['pos'].values; chrom = vdf_x['chrom'].astype(str).values
    in_body = np.zeros(len(vdf_x), bool); in_win = np.zeros(len(vdf_x), bool)
    for g in usable:
        r = gp.loc[g]; same = chrom == str(r['chr'])
        in_body |= same & (pos >= int(r['start'])) & (pos <= int(r['end']))
        in_win |= same & (np.abs(pos - int(r['pos'])) <= window)
    tested = in_win & ~in_body
    if maf > 0:
        af = dos_x.mean(1) / 2.0
        is_str = (vtype.values == 'str')
        tested &= is_str | (np.minimum(af, 1.0 - af) >= maf)

    def _outside(sites):
        m = np.ones(len(sites), bool)
        c = sites['chrom'].astype(str).values; p = sites['pos'].values
        for g in usable:
            r = gp.loc[g]
            m &= ~((c == str(r['chr'])) & (p >= int(r['start'])) & (p <= int(r['end'])))
        return m
    for key, arrs in (('str_sites', ('str_len', 'str_phased')),
                      ('ma_sites', ('ma_alleles', 'ma_phased'))):
        if aux.get(key) is not None:
            m = _outside(aux[key])
            aux[key] = aux[key][m].reset_index(drop=True)
            for a in arrs:
                aux[a] = aux[a][m]
    print(f'  scan rows by type: {vtype.value_counts().to_dict()}; '
          f'{int(tested.sum())} tested (window, outside gene bodies)')
    return dict(vdf=vdf_x, dos=dos_x, xL=xL_x, xR=xR_x, vtype=vtype, aux=aux, tested=tested)


def second_pass_arm(aux, order, A, T, Va, Vt, genes, pos_df, window, perm, min_hap):
    """Curvature (STR) and categorical (multi-ALT) second passes under the same
    permutation as the lead scan: sample k's expression is paired with sample
    inv[k]'s genotype, exactly as hapmix_arm pairs them."""
    if not aux:
        return {}
    ix = np.arange(len(order)) if perm is None else np.argsort(np.asarray(perm))
    mk = lambda M: pd.DataFrame(M, index=genes, columns=order)
    out = {}
    with contextlib.redirect_stdout(io.StringIO()):
        if aux.get('str_len') is not None and len(aux['str_sites']):
            out['str'] = map_str_curvature(
                aux['str_len'][:, ix], aux['str_phased'][:, ix],
                aux['str_sites'].set_index('id'), list(order),
                mk(A), mk(T), mk(Va), mk(Vt), pos_df, window=window, verbose=False)
        if aux.get('ma_alleles') is not None and len(aux['ma_sites']):
            out['ma'], _ = map_multiallelic(
                aux['ma_alleles'][:, ix], aux['ma_sites'].set_index('site_id'), list(order),
                mk(A), mk(T), mk(Va), mk(Vt), pos_df, hap_phased=aux['ma_phased'][:, ix],
                window=window, min_hap=min_hap, verbose=False)
    return out


def nonstandard_block(obs_x, sp_obs, sp_nulls, vtype):
    """Aggregate summary of the opt-in arm: what the leads are, and whether the
    second-pass tests are calibrated on the permuted null."""
    lt = obs_x['lead'].map(vtype).fillna('snp')
    b = {'note': NONSTANDARD_NOTE,
         'scan_rows_by_type': {k: int(v) for k, v in vtype.value_counts().items()},
         'lead_variant_type_observed': {k: int(v) for k, v in lt.value_counts().items()},
         'frac_leads_nonstandard': float((lt != 'snp').mean())}

    def _frac(dfs, col):
        p = pd.concat([pd.to_numeric(d[col], errors='coerce') for d in dfs]).dropna() \
            if dfs else pd.Series(dtype=float)
        return (float((p < 0.05).mean()) if len(p) else None), int(len(p))
    for key, col, name in (('str', 'pval_curv', 'str_curvature_test'),
                           ('ma', 'pval_joint', 'multiallelic_joint_test')):
        if key in sp_obs:
            fo, no = _frac([sp_obs[key]], col)
            fn, nn = _frac([s[key] for s in sp_nulls if key in s], col)
            b[name] = {'observed_frac_p_lt_0.05': fo, 'n_observed': no,
                       'null_frac_p_lt_0.05': fn, 'n_null': nn,
                       'note': 'null fraction should be ~0.05 if the test is calibrated on this data'}
    return b


# ---------------------------------------------------------------------------
#  Knockoff null: an LD-preserving negative control
# ---------------------------------------------------------------------------

def knockoff_haplotypes(xL, xR, vdf, draws, block=6000, K=4, n_em_iter=10,
                        seed=0, verbose=True):
    """Knockoff haplotypes for every variant, [M, V, N] each for L and R.

    WHY NOT PERMUTE. Permuting rSNP genotypes across samples breaks their LD
    with everything else, and two artifacts in this comparison live in exactly
    that LD, pulling in opposite directions:

      * hapmixQTL's gene totals are missing wherever a sample has no
        heterozygous transcript, and that missingness tracks local
        heterozygosity, which is in LD with the tested rSNPs. Permutation
        removes it from the null but not from the observed statistic, so it
        inflates hapmixQTL.
      * RASQUAL's evidence comes from individuals heterozygous at BOTH an rSNP
        and an fSNP. Real cis LD enriches those; permutation makes them
        independent, so RASQUAL's null sits too low and its threshold too
        permissive.

    A knockoff variant is exchangeable with the real one -- same joint
    distribution, hence the same LD with everything in the region -- but
    carries no association with the phenotype. Both artifacts therefore appear
    in the null as well as the observed, and the matched-FPR comparison is
    calibrated for both arms at once.

    Knockoffs are drawn over the WHOLE region including the feature SNPs, not
    only the tested ones. The structural-zero pattern is a function of
    gene-body heterozygosity, so a knockoff region that excluded gene bodies
    would not reproduce it and the exercise would buy nothing on that axis.

    Blocks are contiguous in genome order and never span a chromosome, so the
    HMM is fitted on real local LD. One fit per block is reused for all M
    draws: the fit dominates (~13 s per 5-6k variants against ~1 s per extra
    draw), which is what makes M draws affordable.

    This uses knockoffs as a negative CONTROL, not as the knockoff filter --
    no W statistics, no swap-antisymmetry requirement, so the experimental
    variant-level FDR path in tensorqtl/knockoffs.py is not being relied on.
    """
    import time
    from tensorqtl import knockoffs as _ko
    V, N = xL.shape
    kL = np.empty((draws, V, N), np.int8)
    kR = np.empty((draws, V, N), np.int8)
    chrom = vdf['chrom'].values
    starts = []
    i = 0
    while i < V:
        j = min(i + block, V)
        c = chrom[i]
        # never let a block straddle two chromosomes
        while j > i + 1 and chrom[j - 1] != c:
            j -= 1
        if chrom[i] != c:
            j = i + 1
        starts.append((i, j))
        i = j
    t0 = time.time()
    for bi, (i, j) in enumerate(starts, 1):
        X = xL[i:j].T.astype(np.int64)
        Y = xR[i:j].T.astype(np.int64)
        out = _ko.haplotype_hmm_knockoffs(X, Y, K=K, M=draws,
                                          n_em_iter=n_em_iter, seed=seed + i,
                                          return_phased=True)
        (xkL, xkR) = out[1]
        for m in range(draws):
            kL[m, i:j, :] = xkL[m].T.astype(np.int8)
            kR[m, i:j, :] = xkR[m].T.astype(np.int8)
        if verbose and (bi % 10 == 0 or bi == len(starts)):
            el = time.time() - t0
            print(f'    knockoffs: block {bi}/{len(starts)} '
                  f'({el/60:.1f} min elapsed)', flush=True)
    return kL, kR

# ---------------------------------------------------------------------------
#  RASQUAL arm: native per-fSNP counts, tested rSNPs permuted, fSNPs fixed
# ---------------------------------------------------------------------------

def rasqual_arm(binary, genes, pos_df, vdf, xL, xR, allelic, order, Y, K,
                window, perm=None, tmp=None, tested=None, maf=0.05,
                min_coverage=0.05):
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
        # Test the SAME variants hapmixQTL tests. Without this the two arms
        # differ twice over: the mask excludes variants inside ANY selected
        # gene body while this excludes only THIS gene's, and --maf gated only
        # hapmixQTL while RASQUAL fell back to its own default of 0.05. At
        # --maf 0.05 they coincided by accident; at any other value the arms
        # would silently score different variant sets under one reported count.
        if tested is not None:
            ridx = ridx[tested[ridx]]
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
               '-f', str(g), '-z',
               '-d', str(min_coverage),
               # match hapmixQTL's floor; RASQUAL's own default is 0.05
               # (main.c:386). The HWE gate is already bypassed by -z, which
               # sets noPriorGenotype (main.c:498), and RSQ=1.0 clears the
               # imputation-quality gate, so MAF is the only one left to align.
               '-a', str(maf)]
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

def score(obs, null, name, known=None, genes_keep=None):
    """obs/null: DataFrames with gene, stat. null pooled over permutations."""
    o = obs.dropna(subset=['stat']); n = null.dropna(subset=['stat'])
    n_own = int(len(o))
    # Score both methods on the SAME genes. RASQUAL drops genes on timeout and
    # on non-convergence, and those are preferentially the variant-dense,
    # fSNP-rich ones -- exactly where the signal is. Dividing each method by
    # its own survivors compares a fraction of one gene set against a fraction
    # of another and prints them side by side as if they were commensurable.
    if genes_keep is not None:
        o = o[o['gene'].isin(genes_keep)]
    out = {'n_genes_scored': int(len(o)), 'n_null_stats': int(len(n)),
           'n_genes_this_method_alone': n_own,
           'n_genes_dropped_for_common_set': n_own - int(len(o))}
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
    genes_all, samples, YL, YR, YT = H.load_counts(args.salmon, args.tx2gene,
                                                   sufs, out)
    # yT is the gene total over ALL transcripts. Without it the total
    # channel is a heterozygous-transcript subtotal whose zeros are in LD
    # with the tested variants.
    A, T, Va, Vt, _ = compute_summaries_from_gibbs(YL, YR, yT=YT)

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
    # MAF applies to TESTED rSNPs only, never to the feature SNPs in gene
    # bodies. A cis variant too rare to carry power is noise in the scan, but
    # a rare heterozygous fSNP is exactly what the allelic channel runs on --
    # filtering both on one threshold silently trims the ASE signal this
    # comparison exists to measure. Keep the VCF pre-filter permissive and set
    # the testing threshold here.
    af = dos.mean(1) / 2.0
    maf = np.minimum(af, 1.0 - af)
    n_pre = int(tested.sum())
    tested = tested & (maf >= args.maf)
    print(f'  {int(tested.sum())} tested cis variants (window, outside gene '
          f'bodies, MAF >= {args.maf}); {n_pre - int(tested.sum())} dropped by MAF')
    print(f'  fSNPs inside gene bodies are NOT MAF-filtered '
          f'({int(in_body.sum())} in the selected genes)')
    ns = None
    if getattr(args, 'str_vcf', None) or getattr(args, 'multiallelic', False):
        ns = nonstandard_setup(args, vdf, dos, xL, xR, order, gp, usable, args.window,
                               maf=args.maf)

    # RASQUAL total counts and offsets follow the expression (never permuted)
    Ytot = read_salmon_totals(args.salmon, args.tx2gene, order, sufs)
    Ytot = Ytot.reindex(usable).fillna(0.0).values
    K = np.outer(Ytot.mean(1), size_factors(Ytot))

    known = None
    if args.known_egenes:
        known = set(l.strip() for l in open(args.known_egenes) if l.strip())

    def both(perm, tag, XL=None, XR=None, DOS=None):
        # XL/XR/DOS override the real genotypes. The knockoff null substitutes
        # knockoff haplotypes at the tested variants and leaves perm=None;
        # the permutation null leaves them None and passes a perm vector.
        XL = xL if XL is None else XL
        XR = xR if XR is None else XR
        DOS = dos if DOS is None else DOS
        t0 = time.time()
        h = hapmix_arm(A, T, Va, Vt, usable, order, vdf, DOS, XL, XR,
                       pos_df[['chr', 'pos']], args.window, tested, perm)
        th = time.time() - t0; t0 = time.time()
        r = rasqual_arm(args.rasqual, usable, pos_df, vdf, XL, XR, allelic,
                        order, Ytot, K, args.window, perm,
                        tested=tested, maf=args.maf,
                        min_coverage=args.min_coverage)
        tr = time.time() - t0
        x, sp, tx = None, {}, 0.0
        if ns is not None:                       # opt-in arm; standard arms above untouched
            t0 = time.time()
            x = hapmix_arm(A, T, Va, Vt, usable, order, ns['vdf'], ns['dos'], ns['xL'],
                           ns['xR'], pos_df[['chr', 'pos']], args.window, ns['tested'], perm)
            sp = second_pass_arm(ns['aux'], order, A, T, Va, Vt, usable,
                                 pos_df[['chr', 'pos']], args.window, perm, args.min_hap)
            tx = time.time() - t0
        print(f'  {tag}: hapmixQTL {th:.0f}s, RASQUAL {tr:.0f}s '
              f'({int((r["status"]=="ok").sum())}/{len(r)} converged)'
              + (f', non-standard arm {tx:.0f}s' if ns is not None else ''), flush=True)
        return h, r, th, tr, x, sp, tx

    print('\nObserved')
    obs_h, obs_r, th, tr, obs_x, sp_obs, tx_obs = both(None, 'observed')
    nulls_h, nulls_r, nulls_x, sp_nulls = [], [], [], []
    kL = kR = win_row = None
    if args.null == 'knockoff':
        win_idx = np.where(in_win)[0]
        print(f'\nKnockoff null: {len(win_idx)} variants across the selected '
              f'cis windows (gene bodies included, so the structural-zero '
              f'pattern is reproduced in the null)')
        kL, kR = knockoff_haplotypes(xL[win_idx], xR[win_idx],
                                     vdf.iloc[win_idx], args.n_perm,
                                     K=args.knockoff_k, seed=args.seed)
        win_row = {int(v): i for i, v in enumerate(win_idx)}
    sel_t = np.where(tested)[0]
    for p in range(args.n_perm):
        if args.null == 'knockoff':
            rows = [win_row[int(v)] for v in sel_t]
            XLm = xL.copy(); XRm = xR.copy()
            XLm[sel_t] = kL[p][rows]; XRm[sel_t] = kR[p][rows]
            DOSm = (XLm + XRm).astype(np.int8)
            h, r, _, _, x, sp, _ = both(None, f'knockoff {p+1}/{args.n_perm}',
                                        XLm, XRm, DOSm)
            del XLm, XRm, DOSm
        else:
            perm = rng.permutation(len(order))
            h, r, _, _, x, sp, _ = both(perm, f'perm {p+1}/{args.n_perm}')
        nulls_h.append(h); nulls_r.append(r)
        if x is not None:
            nulls_x.append(x); sp_nulls.append(sp)
    null_h = pd.concat(nulls_h) if nulls_h else obs_h.iloc[0:0]
    null_r = pd.concat(nulls_r) if nulls_r else obs_r.iloc[0:0]

    # Reference-bias gate on the hapmixQTL arm. hapmixQTL has no analogue of
    # RASQUAL's phi and degrades catastrophically rather than gracefully when
    # mapping bias is present, so its validity is CONDITIONAL on bias-filtered
    # input. The comparison reported RASQUAL's phi and silently skipped this,
    # which measured one arm's bias and assumed the other's. Sign comes from
    # the highest-depth feature SNP in each gene body: bias accumulates toward
    # the reference allele at the sites carrying the reads, so the deepest
    # fSNP is the best single proxy for the gene's haplotype orientation.
    YLm = YL[sel][:, keep].mean(2)
    YRm = YR[sel][:, keep].mean(2)
    g_sign = np.zeros_like(YLm)
    sgn = np.sign(xL.astype(np.int16) - xR.astype(np.int16))
    for i, g in enumerate(usable):
        r = gp.loc[g]
        same = vdf['chrom'].values == str(r['chr'])
        body = np.where(same & (pos >= int(r['start'])) & (pos <= int(r['end'])))[0]
        best, best_depth = None, -1
        for v in body:
            c = allelic.get((str(r['chr']), int(pos[v])))
            if not c:
                continue
            d = sum(a + b for a, b in c.values())
            if d > best_depth:
                best, best_depth = v, d
        if best is not None:
            g_sign[i] = sgn[best]
    refbias = reference_bias_diagnostic(YLm, YRm, g_sign)
    print(f'\nReference-bias gate (hapmixQTL arm): {refbias["message"]}')
    if refbias.get('flag'):
        print('  WARNING: hapmixQTL is not valid under mapping bias; its '
              'type-I error inflates sharply (docs/ase_validation.md sec 7h)')

    common_genes = (set(obs_h.dropna(subset=['stat'])['gene'])
                    & set(obs_r.dropna(subset=['stat'])['gene']))
    print(f'\nScoring both methods on the {len(common_genes)} genes where BOTH '
          f'converged (hapmixQTL {int(obs_h["stat"].notna().sum())}, '
          f'RASQUAL {int(obs_r["stat"].notna().sum())} individually)')
    result = {
        'design': {'question': 'which pipeline would you deploy',
                   'rasqual_input': 'phASER per-fSNP counts (native)',
                   'hapmixqtl_input': 'Salmon diploid Gibbs (native)',
                   'shared': ['samples', 'genes', 'tested variants outside gene '
                              'bodies', 'phase (rephased.vcf.gz)', 'permutation'],
                   'null': ('LD-preserving knockoff haplotypes substituted at the '
                            'tested rSNPs; fSNP genotypes + allele counts real'
                            if args.null == 'knockoff' else
                            'tested rSNPs permuted across samples as a block; '
                            'fSNP genotypes + allele counts fixed with expression'),
                   'null_kind': args.null,
                   'n_genes': len(usable), 'n_samples': len(order),
                   'n_tested_variants': int(tested.sum()), 'n_perm': args.n_perm,
                   'n_genes_both_converged': len(common_genes),
                   'window': args.window, 'seed': args.seed},
        'compute_seconds_observed': {'hapmixQTL': th, 'RASQUAL': tr},
        'hapmixQTL': score(obs_h, null_h, 'hapmixQTL', known,
                           genes_keep=common_genes),
        'RASQUAL': score(obs_r, null_r, 'RASQUAL', known,
                         genes_keep=common_genes),
        'head_to_head': compare(obs_r, obs_h, out),
        'reference_bias_hapmixqtl': {
            k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
            for k, v in refbias.items() if k != 'per_gene'},
    }
    if ns is not None:
        null_x = pd.concat(nulls_x) if nulls_x else obs_x.iloc[0:0]
        result['hapmixQTL_nonstandard'] = score(obs_x, null_x, 'hapmixQTL_nonstandard', known)
        result['hapmixQTL_nonstandard']['note'] = NONSTANDARD_NOTE
        result['head_to_head_nonstandard_vs_rasqual'] = compare(obs_r, obs_x, out)
        result['nonstandard'] = nonstandard_block(obs_x, sp_obs, sp_nulls, ns['vtype'])
        result['compute_seconds_observed']['hapmixQTL_nonstandard'] = float(tx_obs)
        obs_x.to_csv(out / 'observed_hapmixqtl_nonstandard.tsv', sep='\t', index=False)
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
    if 'hapmixQTL_nonstandard' in r:
        x = r['hapmixQTL_nonstandard']; nb = r.get('nonstandard', {})
        L += ['', '## NON-STANDARD, opt-in arm: hapmixQTL + STRs / multi-ALT split rows', '',
              'Not a like-for-like comparison with RASQUAL (it cannot test these variants); '
              'read as what the extra variant classes add to hapmixQTL.', '',
              '| | hapmixQTL (standard) | hapmixQTL + non-standard rows |', '|---|---|---|',
              f"| power @ empirical FPR 10% | {pw('hapmixQTL',0.1):.3f} | {pw('hapmixQTL_nonstandard',0.1):.3f} |",
              f"| power @ empirical FPR 5% | {pw('hapmixQTL',0.05):.3f} | {pw('hapmixQTL_nonstandard',0.05):.3f} |",
              f"| λ_GC on permuted null | {r['hapmixQTL'].get('calibration',{}).get('lambda_gc_null',float('nan')):.2f} | "
              f"{x.get('calibration',{}).get('lambda_gc_null',float('nan')):.2f} |",
              f"| leads that are STR / split rows | 0 | {nb.get('frac_leads_nonstandard', float('nan')):.3f} |"]
        rep = x.get('replication')
        if rep:
            L.append(f"| known-eGene fraction of discoveries | | "
                     f"{rep['frac_known_egene']:.3f} (baseline {rep['baseline_frac_known_in_tested']:.3f}) |")
        for key, lab in (('str_curvature_test', 'STR curvature test'),
                         ('multiallelic_joint_test', 'multi-ALT joint test')):
            if key in nb:
                t = nb[key]
                L.append(f"- {lab}: p<0.05 in {t['observed_frac_p_lt_0.05']} of {t['n_observed']} "
                         f"observed pairs vs {t['null_frac_p_lt_0.05']} of {t['n_null']} on the "
                         f"permuted null (calibrated if ~0.05)")
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
    ap.add_argument('--null', choices=('permute', 'knockoff'), default='permute',
                    help="'permute' shuffles rSNP genotypes across samples, "
                         'which destroys their LD and mis-calibrates BOTH arms '
                         'in opposite directions. "knockoff" substitutes '
                         'LD-preserving knockoff haplotypes instead')
    ap.add_argument('--min-coverage', type=float, default=0.05,
                    help="RASQUAL's -d/--min-coverage-depth. Default is "
                         "RASQUAL's own (main.c:396); passed explicitly so the "
                         'value used is recorded rather than inherited')
    ap.add_argument('--knockoff-k', type=int, default=4,
                    help='haplotype clusters in the knockoff HMM (default 4)')
    ap.add_argument('--maf', type=float, default=0.05,
                    help='minor-allele frequency floor for TESTED cis variants. '
                         'Feature SNPs in gene bodies are deliberately exempt: '
                         'the allelic channel depends on rare het sites. Keep '
                         'the VCF pre-filter permissive (e.g. 0.01) and set '
                         'the testing threshold here (default 0.05)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='deploy')
    ns = ap.add_argument_group(
        'NON-STANDARD, opt-in (off by default; adds a separately reported hapmixQTL arm, '
        'the standard RASQUAL and hapmixQTL arms are unchanged)')
    ns.add_argument('--str-vcf', default=None,
                    help='HipSTR-style STR VCF: STRs join the tested variants as per-haplotype '
                         'repeat length, plus the curvature second pass')
    ns.add_argument('--multiallelic', action='store_true',
                    help='multi-ALT rows of --vcf join the tested variants as one split row '
                         'per ALT, plus the categorical second pass')
    ns.add_argument('--min-hap', type=int, default=10,
                    help='categorical model: alleles carried by fewer haplotypes are pooled')
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
    # planted effects, by gene index mod 4:
    #   0  rSNP haplotype drives expression (x1.6)     -> both standard arms can find it
    #   1  an STR OUTSIDE the body drives it, 0.3 log-units per repeat unit  (opt-in arm only)
    #   2  null
    #   3  ALT2 of a tri-allelic site outside the body drives it (x1.7)     (opt-in arm only)
    eff = {i: (1.6 if i % 4 == 0 else 1.0) for i in range(G)}
    h1r = {i: (rng.rand(N) < .4).astype(int) for i in range(G)}
    h2r = {i: (rng.rand(N) < .4).astype(int) for i in range(G)}
    units = [-2, -1, 0, 1, 2]
    L1 = {i: rng.choice(units, N) for i in range(G)}; L2 = {i: rng.choice(units, N) for i in range(G)}
    m1 = {i: rng.choice(3, N, p=[.55, .25, .2]) for i in range(G)}
    m2 = {i: rng.choice(3, N, p=[.55, .25, .2]) for i in range(G)}

    def mult(i, k, hap):
        if i % 4 == 0:
            return eff[i] if (h1r if hap == 1 else h2r)[i][k] else 1.0
        if i % 4 == 1:
            return float(np.exp(0.3 * (L1 if hap == 1 else L2)[i][k]))
        if i % 4 == 3:
            return 1.7 if (m1 if hap == 1 else m2)[i][k] == 2 else 1.0
        return 1.0
    # fSNP in LD with rSNP on the same haplotype
    def ld(h): return np.where(rng.rand(N) < .7, h, (rng.rand(N) < .4).astype(int))
    h1f = {i: ld(h1r[i]) for i in range(G)}; h2f = {i: ld(h2r[i]) for i in range(G)}
    hdr = '#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t' + '\t'.join(samples)
    lines = ['##fileformat=VCFv4.2', hdr]; slines = ['##fileformat=VCFv4.2', hdr]
    alt_ix = {u: j for j, u in enumerate([u for u in units if u != 0], 1)}
    for i in range(G):
        base = 100000 * i
        lines.append(f'1\t{base+500}\tf{i}\tA\tG\t.\tPASS\t.\tGT\t' + '\t'.join(
            f'{h1f[i][k]}|{h2f[i][k]}' for k in range(N)))
        for off, tag in ((2000, 'r'), (5000, 'q')):
            lines.append(f'1\t{base+off}\t{tag}{i}\tA\tG\t.\tPASS\t.\tGT\t' + '\t'.join(
                f'{h1r[i][k]}|{h2r[i][k]}' for k in range(N)))
        # tri-allelic row: the standard reader skips it; only --multiallelic uses it
        lines.append(f'1\t{base+7000}\tm{i}\tA\tT,C\t.\tPASS\t.\tGT\t' + '\t'.join(
            f'{m1[i][k]}|{m2[i][k]}' for k in range(N)))
        slines.append(f'1\t{base+8000}\tSTR{i}\t{"CAG"*8}\t'
                      + ','.join('CAG' * (8 + u) for u in units if u != 0)
                      + '\t.\tPASS\tPERIOD=3\tGT\t' + '\t'.join(
                          f'{alt_ix.get(L1[i][k], 0)}|{alt_ix.get(L2[i][k], 0)}' for k in range(N)))
    (td / 'p.vcf').write_text('\n'.join(lines) + '\n')
    (td / 'str.vcf').write_text('\n'.join(slines) + '\n')
    man, acman = [], []
    for k, s in enumerate(samples):
        sd = td / s / 'aux_info' / 'bootstrap'; sd.mkdir(parents=True)
        names, boot = [], []
        acf = td / f'{s}.allelic_counts.txt'
        with open(acf, 'w') as fh:
            # phASER's REAL header: no 'stop', position not start.
            fh.write('contig\tposition\tvariantID\trefAllele\taltAllele\trefCount\taltCount\ttotalCount\n')
            for i in range(G):
                a = rng.poisson(30 * mult(i, k, 1)); b = rng.poisson(30 * mult(i, k, 2))
                names += [txs[i] + '_hapA', txs[i] + '_hapB']
                boot += [rng.poisson(max(a, 1), ND), rng.poisson(max(b, 1), ND)]
                if h1f[i][k] != h2f[i][k]:                 # het fSNP: counts
                    alt = a if h1f[i][k] == 1 else b; ref = a + b - alt
                    fh.write(f'1\t{100000*i+500}\tf{i}\tA\tG\t{ref}\t{alt}\t{ref+alt}\n')
        with gzip.open(sd / 'names.tsv.gz', 'wt') as fh: fh.write('\t'.join(names))
        with gzip.open(sd / 'bootstraps.gz', 'wb') as fh:
            fh.write(np.array(boot, np.float64).T.tobytes())
        (td / s / 'aux_info' / 'meta_info.json').write_text(json.dumps({'num_bootstraps': ND}))
        pd.DataFrame({'Name': names, 'Length': 1000, 'EffectiveLength': 900, 'TPM': 1.0,
                      'NumReads': [float(x.mean()) for x in boot]}).to_csv(
            td / s / 'quant.sf', sep='\t', index=False)
        man.append(f'{s}\t{td/s}'); acman.append(f'{s}\t{acf}')
    (td / 'salmon.tsv').write_text('\n'.join(man)); (td / 'ac.tsv').write_text('\n'.join(acman))
    (td / 'known.txt').write_text('\n'.join(f'G{i:05d}' for i in range(G) if i % 4 != 2))
    base_args = dict(
        vcf=str(td / 'p.vcf'), genes=str(td / 'genes.tsv'), salmon=str(td / 'salmon.tsv'),
        tx2gene=str(td / 't2g.tsv'), allelic_counts=str(td / 'ac.tsv'), rasqual=rq,
        known_egenes=str(td / 'known.txt'), hap_suffix='_hapA,_hapB',
        n_genes=G, n_perm=2, window=10000, seed=0, maf=0.05,
        null='permute', knockoff_k=4, min_coverage=0.05,
        str_vcf=None, multiallelic=False, min_hap=10)
    print('SELF-TEST: deploy comparison on fabricated native inputs (standard: biallelic SNPs)\n')
    r = run(argparse.Namespace(**base_args, out=str(td / 'deploy')))
    print('\n' + (td / 'deploy' / 'deploy_comparison.md').read_text())
    for m in ('RASQUAL', 'hapmixQTL'):
        assert 'power' in r[m], f'{m} produced no power estimate'
    # the knockoff null must run end to end and produce a usable null
    # n_perm=2 so the pooled null clears score()'s 20-statistic floor; with
    # one draw over 16 fabricated genes it does not, and the arm reports
    # 'too few statistics' rather than a power estimate.
    ko_args = dict(base_args); ko_args.update(null='knockoff', knockoff_k=2,
                                              n_perm=2)
    rk = run(argparse.Namespace(**ko_args, out=str(td / 'deploy_ko')))
    assert rk['design']['null_kind'] == 'knockoff', rk['design']
    # the reference-bias gate must report on the hapmixQTL arm, not be skipped
    rb = r['reference_bias_hapmixqtl']
    assert 'ref_fraction' in rb and 'message' in rb, rb
    assert rb['n_obs'] >= 0, rb
    for m in ('RASQUAL', 'hapmixQTL'):
        assert 'power' in rk[m], f'{m}: no power estimate under knockoffs'
        assert rk[m]['n_null_stats'] > 0, (m, rk[m])
    # the MAF floor must gate TESTED variants and leave fSNPs alone: raising it
    # above every simulated frequency must drop tested variants without
    # touching the fSNPs the allelic channel needs
    assert r['design']['n_tested_variants'] > 0, r['design']
    assert 'hapmixQTL_nonstandard' not in r and 'nonstandard' not in r

    print('\nnow opting in: --str-vcf + --multiallelic (NON-STANDARD arm added)\n')
    r2 = run(argparse.Namespace(**{**base_args, 'str_vcf': str(td / 'str.vcf'),
                                   'multiallelic': True}, out=str(td / 'deploy_ns')))
    print('\n' + (td / 'deploy_ns' / 'deploy_comparison.md').read_text())
    # the standard arms are byte-identical with and without the opt-in
    for f in ('observed_hapmixqtl.tsv', 'observed_rasqual.tsv'):
        a = (td / 'deploy' / f).read_text(); b = (td / 'deploy_ns' / f).read_text()
        assert a == b, f'{f} changed when the opt-in arm was enabled'
    assert r2['hapmixQTL'] == r['hapmixQTL'] and r2['RASQUAL'] == r['RASQUAL']
    x = r2['hapmixQTL_nonstandard']
    assert 'power' in x
    # lead over a superset of tested variants, same whitening -> stat can only go up
    # (map_cis runs in float32 and the row order differs, so allow rounding noise)
    oh = pd.read_csv(td / 'deploy_ns' / 'observed_hapmixqtl.tsv', sep='\t').set_index('gene')
    ox = pd.read_csv(td / 'deploy_ns' / 'observed_hapmixqtl_nonstandard.tsv', sep='\t').set_index('gene')
    d = (ox['stat'] - oh.loc[ox.index, 'stat'])
    assert (d > -1e-3).all(), f'non-standard arm stat below the standard arm: {d.min()}'
    # the planted STR / multi-ALT genes are led by those rows in the opt-in arm
    lead = ox['lead']
    str_genes = [f'G{i:05d}' for i in range(G) if i % 4 == 1]
    ma_genes = [f'G{i:05d}' for i in range(G) if i % 4 == 3]
    n_str_led = sum(lead[g] == f'STR{int(g[1:])}' for g in str_genes)
    n_ma_led = sum(str(lead[g]).startswith(f'1_{100000*int(g[1:])+7000}_') for g in ma_genes)
    print(f'planted STR genes led by their STR: {n_str_led}/{len(str_genes)}; '
          f'planted multi-ALT genes led by a split row: {n_ma_led}/{len(ma_genes)}')
    assert n_str_led == len(str_genes) and n_ma_led == len(ma_genes)
    nb = r2['nonstandard']
    assert 'str_curvature_test' in nb and 'multiallelic_joint_test' in nb
    assert nb['str_curvature_test']['n_null'] > 0 and nb['multiallelic_joint_test']['n_null'] > 0
    print('SELF-TEST OK')
    return 0


if __name__ == '__main__':
    main()

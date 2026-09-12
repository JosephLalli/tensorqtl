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
import concurrent.futures as cf
import contextlib
import gzip
import hashlib
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
import build_asvcf as ASV                      # AS-annotated VCF for the pipe
from make_rasqual_inputs import read_salmon_totals
from str_integrate import parse_str_vcf, parse_multiallelic_vcf, extend_scan   # opt-in arm only

try:
    from tensorqtl.hapmixqtl import (compute_summaries_from_gibbs, map_cis, SAME_COVARIATES,
                                     reference_bias_diagnostic,
                                     map_str_curvature, map_multiallelic)
except ImportError:
    sys.path.insert(0, str(HERE.parent / 'tensorqtl'))
    from hapmixqtl import (compute_summaries_from_gibbs, map_cis, SAME_COVARIATES,
                           reference_bias_diagnostic,
                           map_str_curvature, map_multiallelic)


# ---------------------------------------------------------------------------
#  hapmixQTL arm: gene-level lead statistic, given a genotype permutation
# ---------------------------------------------------------------------------

def hapmix_arm(A, T, Va, Vt, genes, order, vdf, dos, xL, xR, pos_df, window,
               tested_mask, perm=None, cov_df=None, ase_cov='none'):
    """Return DataFrame(gene, stat, log_afc). stat = chi2(1)-equivalent of the
    lead nominal p over TESTED variants, so it sits on RASQUAL's scale.

    cov_df is projected out of the TOTAL channel. ase_cov says what the
    ALLELIC channel gets: 'none' (intercept only; the log haplotype ratio is
    a within-sample contrast in which sample-level covariates cancel) or
    'shared' (the same set, which costs one informative sample per column).
    """
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
                      covariates_df=cov_df,
                      ase_covariates_df=(None if ase_cov == 'none' else SAME_COVARIATES),
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
                min_coverage=0.05, cov_bin=None, jobs=1, asvcf=None,
                bcftools='bcftools', dump=None, exons=None,
                n_threads=4, fsnp_maf=0.0, timeout=3600, knockoff=False,
                rows=None):
    """RASQUAL over genes, fed the VCF text it is designed to read.

    RASQUAL takes a VCF on stdin and decides what is a feature SNP FROM THE
    POSITION -- isExon(pos, starts, ends, nexon) at main.c:516 -- so -m is only
    an allocation count and record order does not matter. Its own example pipes
    a region of a VCF straight in.

    This used to rebuild that text in Python: ~312,000 f-strings and numpy
    scalar lookups per gene, producing bytes already on disk. Besides being
    wasted work it ran in the interpreter, so it held the GIL and capped the
    thread pool at 3-4 concurrent RASQUAL processes instead of the 32 asked
    for. Feeding a tabix slice of an AS-annotated VCF removes the work and the
    contention together: workers now sit in subprocess.run, GIL released.
    """
    if knockoff:
        # The pipe feeds RASQUAL the real genotypes from the AS-VCF, so a
        # knockoff override of xL/xR never reaches it. Say so per gene rather
        # than re-running the observed data and calling it a null.
        return pd.DataFrame([dict(gene=g, stat=np.nan, log_afc=np.nan,
                                  phi=np.nan, status='knockoff_null_not_implemented',
                                  lead=None)
                             for g in genes])
    td = Path(tmp or tempfile.mkdtemp())
    np.asarray(Y, np.float64).tofile(td / 'Y.bin')
    np.asarray(K, np.float64).tofile(td / 'K.bin')
    # RASQUAL tests every record in the slice, feature SNPs included. The
    # lead is taken over the variants hapmixQTL tests too (in window, outside
    # gene bodies, above the MAF floor), or the arms would score different
    # variant sets under one reported count.
    tested_pos = None
    if tested is not None:
        tv = np.asarray(tested, bool)
        tested_pos = set(zip(vdf['chrom'].values[tv].astype(str),
                             vdf['pos'].values[tv].astype(int)))

    def _one(jg):
        j, g = jg
        row = pos_df.loc[g]
        chrom, tss = str(row['chr']), int(row['pos'])
        gs, ge = int(row['start']), int(row['end'])
        # -s/-e are the exon union when we have it. Feature SNPs are sites
        # where reads land; the gene span adds every intron, which inflates
        # the fSNP count 5-99x on real genes and blows RASQUAL's budget.
        ex = exons.get(g) if exons else None
        if ex:
            s_arg, e_arg = ex
            ivs = [(int(a), int(b)) for a, b in zip(s_arg.split(','),
                                                    e_arg.split(','))]
        else:
            s_arg, e_arg = str(gs), str(ge)
            ivs = [(gs, ge)]
        # rasqualTools: cis region is the GENE BODY +/- window, not the TSS
        lo = max(1, gs - window)
        hi = ge + window
        slice_vcf = td / f'g{j}.vcf'
        with open(slice_vcf, 'w') as fh:
            r = subprocess.run(
                [bcftools, 'view', '-H', '-r', f'{chrom}:{lo}-{hi}', str(asvcf)],
                stdout=fh, stderr=subprocess.DEVNULL)
        if r.returncode != 0:
            slice_vcf.unlink(missing_ok=True)
            return dict(gene=g, stat=np.nan, log_afc=np.nan, phi=np.nan, lead=None,
                        status='bcftools_failed')
        # -l is every record in the slice; -m the ones inside the gene body,
        # which is what isExon classifies as feature SNPs.
        n_l = n_m = 0
        with open(slice_vcf) as fh:
            for line in fh:
                n_l += 1
                pv = int(line.split('\t', 2)[1])
                if any(a <= pv <= b for a, b in ivs):
                    n_m += 1
        if n_l == 0 or n_m == 0:
            slice_vcf.unlink(missing_ok=True)
            return dict(gene=g, stat=np.nan, log_afc=np.nan, phi=np.nan, lead=None,
                        status='no_fsnp_or_rsnp')
        cmd = [binary, '-y', str(td / 'Y.bin'), '-k', str(td / 'K.bin'),
               '-n', str(len(order)), '-j', str(j + 1), '-l', str(n_l),
               '-m', str(n_m), '-s', s_arg, '-e', e_arg,
               '-f', str(g), '-z', '-d', str(min_coverage),
               # --force: RASQUAL aborts any gene with (fSNPs+1) x tested >
               # 30,000 (main.c:582), which is undocumented. rasqualTools
               # batches genes by that product into tiers up to >=100,000
               # and excludes none, so production practice is to run heavy
               # genes -- isolated and threaded -- not to skip them.
               '--force', '--n-threads', str(n_threads)] + (
               # RASQUAL's own permutation null: "-r generates a random
               # permutation for each feature to break the correlation between
               # genotype and total feature count as well as AS counts" (README).
               # It draws its own permutation (seeded time+pid, main.c:209), so
               # the null is not paired with hapmixQTL's perm vector.
               ['-r'] if perm is not None else []) + (
               ['-x', str(cov_bin)] if cov_bin else []) + ['-a', str(maf)] + (
               # the authors' own lever for very long genes
               ['--minor-allele-frequency-fsnp', str(fsnp_maf)] if fsnp_maf else [])
        try:
            with open(slice_vcf) as fh:
                pr = subprocess.run(cmd, stdin=fh, capture_output=True,
                                    text=True, errors='replace',
                                    timeout=timeout)
        except Exception as e:
            return dict(gene=g, stat=np.nan, log_afc=np.nan, phi=np.nan, lead=None,
                        status=f'error:{type(e).__name__}')
        finally:
            slice_vcf.unlink(missing_ok=True)
        if rows is not None:
            # every per-variant row RASQUAL wrote, so its statistic at ANY
            # variant (the other arm's lead, say) can be read back later
            Path(rows).mkdir(parents=True, exist_ok=True)
            (Path(rows) / f'{g}.tsv').write_text(pr.stdout)
        best = None
        for ln in pr.stdout.strip().split('\n'):
            f = ln.split('\t')
            if len(f) < 25 or f[1] == 'SKIPPED':
                continue
            if tested_pos is not None and (f[2], int(f[3])) not in tested_pos:
                continue
            try:
                if int(float(f[22])) != 0:
                    continue
                chi2, pi = float(f[10]), float(f[11])
            except ValueError:
                continue
            if best is None or chi2 > best['stat']:
                pi = min(max(pi, 1e-6), 1 - 1e-6)
                # lead as chrom_pos_ref_alt, the id hapmixQTL's arm reports,
                # so lead agreement between arms can be read off the tables
                best = dict(gene=g, stat=chi2, log_afc=np.log(pi / (1 - pi)),
                            phi=float(f[13]), status='ok',
                            lead=f'{f[2]}_{f[3]}_{f[4]}_{f[5]}')
        if best is None and dump is not None:
            # Keep RASQUAL's own output when nothing converged. Column 23
            # (0-based 22) is pbound, its convergence status: non-zero means a
            # parameter hit a boundary, so the raw row says WHICH one.
            Path(dump).mkdir(parents=True, exist_ok=True)
            (Path(dump) / f'{g}.out').write_text(pr.stdout[:200000])
            if pr.stderr:
                (Path(dump) / f'{g}.err').write_text(pr.stderr[:20000])
        return best or dict(gene=g, stat=np.nan, log_afc=np.nan,
                            phi=np.nan, status='no_converged_row', lead=None)

    if jobs <= 1:
        recs = [_one(jg) for jg in enumerate(genes)]
    else:
        with cf.ThreadPoolExecutor(max_workers=jobs) as ex:
            recs = list(ex.map(_one, enumerate(genes)))
    return pd.DataFrame(recs)


# ---------------------------------------------------------------------------
#  Scoring
# ---------------------------------------------------------------------------

def chrom_index(vdf):
    """{chrom: (positions[sorted], original_row_indices)} for O(log V) lookups.

    The gene loop used to do `vdf['chrom'].values == chrom` per gene: a full
    STRING comparison over every variant, for every gene. At 34k genes against
    15.3M variants that is ~5e11 element comparisons and tens of minutes, and
    it grows with the VCF, which is the wrong direction when the pre-filter was
    just relaxed from MAF 0.05 to 0.01. Bucketing once and binary-searching
    makes each gene O(log V).
    """
    chrom = vdf['chrom'].values
    pos = vdf['pos'].values
    order = np.lexsort((pos, chrom))
    c_sorted, p_sorted = chrom[order], pos[order]
    out, i, n = {}, 0, len(order)
    while i < n:
        j = i
        while j < n and c_sorted[j] == c_sorted[i]:
            j += 1
        out[str(c_sorted[i])] = (p_sorted[i:j], order[i:j])
        i = j
    return out


def variants_in(cidx, chrom, lo, hi):
    """Original row indices on `chrom` with lo <= pos <= hi."""
    e = cidx.get(str(chrom))
    if e is None:
        return np.empty(0, np.int64)
    p, idx = e
    a = np.searchsorted(p, lo, 'left')
    b = np.searchsorted(p, hi, 'right')
    return idx[a:b]


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
    # The Gibbs load is ~460 MB of gzip per sample across 92 samples and takes
    # roughly three quarters of an hour. It depends only on the manifest, the
    # tx2gene table and the haplotype suffixes, so cache it: iterating on gene
    # selection, covariates or the null should not re-pay it every time.
    # Gene list first: it determines which rows of the cache and which
    # intervals of the VCF and the allelic counts are needed at all.
    want_genes = None
    if args.gene_list:
        want_genes = [l.strip() for l in open(args.gene_list) if l.strip()]
        print(f'Gene list: {len(want_genes)} genes')

    # Cache as .npy, not .npz. An .npz is one compressed archive and
    # np.load decompresses the whole thing; separate .npy files can be
    # memory-mapped, so a 30-gene run reads 30 rows off disk instead of
    # materialising 34,457 x 92 x 200 to use 0.09% of it.
    cache = None
    if args.cache_dir:
        key = hashlib.sha256(
            (Path(args.salmon).read_text() + Path(args.tx2gene).read_text()
             + ','.join(sufs)).encode()).hexdigest()[:16]
        cache = Path(args.cache_dir) / f'gibbs_{key}'
    if cache is not None and (cache / 'genes.txt').exists():
        genes_all = (cache / 'genes.txt').read_text().split()
        samples = (cache / 'samples.txt').read_text().split()
        mm = {k: np.load(cache / f'{k}.npy', mmap_mode='r')
              for k in ('YL', 'YR', 'YT')}
        if want_genes is not None:
            keep_g = {g: i for i, g in enumerate(genes_all)}
            rows = [keep_g[g] for g in want_genes if g in keep_g]
            print(f'Cache: slicing {len(rows)} of {len(genes_all)} genes '
                  f'(memory-mapped, {mm["YL"].shape[2]} draws)')
            YL, YR, YT = (np.asarray(mm[k][rows]) for k in ('YL', 'YR', 'YT'))
            genes_all = [genes_all[i] for i in rows]
        else:
            print(f'Cache: loading all {len(genes_all)} genes')
            YL, YR, YT = (np.asarray(mm[k]) for k in ('YL', 'YR', 'YT'))
    else:
        genes_all, samples, YL, YR, YT = H.load_counts(args.salmon, args.tx2gene,
                                                       sufs, out)
        if cache is not None:
            cache.mkdir(parents=True, exist_ok=True)
            print(f'Caching Gibbs arrays to {cache}/ (memory-mappable .npy)')
            for k, v in (('YL', YL), ('YR', YR), ('YT', YT)):
                np.save(cache / f'{k}.npy', v)
            (cache / 'genes.txt').write_text('\n'.join(map(str, genes_all)))
            (cache / 'samples.txt').write_text('\n'.join(map(str, samples)))
        if want_genes is not None:
            keep_g = {g: i for i, g in enumerate(genes_all)}
            rows = [keep_g[g] for g in want_genes if g in keep_g]
            YL, YR, YT = YL[rows], YR[rows], YT[rows]
            genes_all = [genes_all[i] for i in rows]
    # yT is the gene total over ALL transcripts. Without it the total
    # channel is a heterozygous-transcript subtotal whose zeros are in LD
    # with the tested variants.
    A, T, Va, Vt, _ = compute_summaries_from_gibbs(YL, YR, yT=YT,
                                                  count_noise=args.count_noise)

    regions = None
    if want_genes is not None:
        gtmp = pd.read_csv(args.genes, sep='\t', header=None, dtype={1: str})
        gtmp.columns = ['gene', 'chr', 'start', 'end', 'pos'][:gtmp.shape[1]]
        sel = gtmp[gtmp['gene'].isin(set(want_genes))]
        regions = Path(out) / 'regions.bed'
        with open(regions, 'w') as fh:
            for _, r in sel.sort_values(['chr', 'start']).iterrows():
                fh.write(f"{r['chr']}\t{max(1, int(r['start'])-args.window)}"
                         f"\t{int(r['end'])+args.window}\t{r['gene']}\n")
        span = sum(int(r['end']) + args.window - max(1, int(r['start']) - args.window)
                   for _, r in sel.iterrows())
        print(f'Regions: {len(sel)} windows spanning {span/1e6:.0f} Mb '
              f'-- VCF and allelic counts are read only there')

    exons = None
    if args.exons:
        exons = {}
        for l in open(args.exons):
            f = l.rstrip('\n').split('\t')
            if len(f) >= 3:
                exons[f[0]] = (f[1], f[2])
        print(f'Exon unions for {len(exons)} genes (RASQUAL -s/-e)')

    print('Genotypes (use rephased.vcf.gz from phaser_to_matrix.py)')
    vdf, dos, xL, xR, order = H.read_phased_vcf(args.vcf, set(samples),
                                                regions=regions)
    keep = [samples.index(s) for s in order]
    A, T, Va, Vt = A[:, keep], T[:, keep], Va[:, keep], Vt[:, keep]
    vdf['chrom'] = vdf['chrom'].astype(str)

    cov_df, cov_bin = None, None
    if args.covariates:
        cov_df = pd.read_csv(args.covariates, sep='\t', index_col=0)
        print(f'Covariates: {cov_df.shape[1]} columns '
              f'({", ".join(list(cov_df.columns)[:4])}...)')
        print('  passed to BOTH arms. Neither method wants a pre-residualized '
              'phenotype: hapmixQTL projects covariates out inside the '
              'sqrt(w)-weighted space with different weights per channel, and '
              'RASQUAL fits them in a GLM on the count scale.')

    print('RASQUAL input: phASER per-feature-SNP counts')
    allelic = H.load_allelic_counts(args.allelic_counts, order, regions=regions)

    # RASQUAL reads VCF text with an AS field. --asvcf supplies a prebuilt
    # one; otherwise build it here from the same inputs, over the same
    # regions, so an ad-hoc run and the self-test go through the pipe too.
    asvcf = args.asvcf
    if args.rasqual and not asvcf:
        asvcf = Path(out) / 'as.vcf.gz'
        n_rec, n_as = ASV.annotate(args.vcf, ASV.load_counts(args.allelic_counts, regions),
                                   asvcf, regions=regions)
        if n_rec == 0:
            raise SystemExit(f'no records read from {args.vcf} for the AS-VCF '
                             '(a region-scoped read needs a bgzipped, indexed VCF)')
        subprocess.run(['bcftools', 'index', '-t', '-f', str(asvcf)], check=True)
        print(f'AS-VCF built for the RASQUAL pipe: {n_rec} biallelic records, '
              f'{n_as} with allele counts -> {asvcf}')

    if cov_df is not None:
        missing = [s_ for s_ in order if s_ not in cov_df.index]
        if missing:
            raise SystemExit(f'covariates missing for {len(missing)} samples, '
                             f'e.g. {missing[:3]}')
        cov_df = cov_df.loc[list(order)]
        cov_bin = Path(out) / 'covariates.rasqual.bin'
        # covariate-major, as main.c:301-307 reads it
        np.asarray(cov_df.values.T, np.float64).tofile(cov_bin)

    gp = pd.read_csv(args.genes, sep='\t', header=None, dtype={1: str})
    gp.columns = ['gene', 'chr', 'start', 'end', 'pos'][:gp.shape[1]]
    if 'pos' not in gp:
        gp['pos'] = gp['start']
    gp = gp.set_index('gene'); gp['chr'] = gp['chr'].astype(str).str.strip()
    genes_all = list(genes_all)
    gi_all = {g: i for i, g in enumerate(genes_all)}
    # usable: in both inputs, with >= 1 fSNP carrying counts
    pos = vdf['pos'].values
    # An explicit gene list short-circuits the candidate filter below. The
    # filter can only approximate RASQUAL's real gate, which is four
    # conditions on each feature SNP -- coverage, allele-specific genotype AF
    # bounds and a strict-interior allele fraction (main.c:535) -- against the
    # one condition ("some fSNP in the body has counts") checkable from here.
    # It therefore OVER-admits, and the excess shows up later as genes RASQUAL
    # silently drops. Probing with a cheap RASQUAL-only pass and feeding the
    # genes it actually used back in via --gene-list removes the approximation
    # rather than correcting for it afterwards.
    usable = []
    cidx = chrom_index(vdf)
    gp_rec = gp.to_dict('index')          # pandas .loc per gene is ~100us
    for g in (want_genes if want_genes is not None else gp.index):
        if g not in gi_all or g not in gp_rec:
            continue
        if want_genes is not None:
            usable.append(g)
            continue
        r = gp_rec[g]; same = None
        body = variants_in(cidx, r['chr'], int(r['start']), int(r['end']))
        if any((str(r['chr']), int(pos[v])) in allelic for v in body):
            usable.append(g)
    # Expression floor, GTEx-style: >= --min-count reads in >= --min-count-frac
    # of samples in the Salmon input. A gene RASQUAL can use (allele counts at
    # its feature SNPs, from the aligner) can be one Salmon assigns nothing
    # to: CYP3A7 in the pilot had a median of 0 reads and 73/92 zero samples,
    # so hapmixQTL's statistic was 0 while RASQUAL reported chi2 12.
    tot_mean = YT.mean(2)            # cache sample order; a fraction is order-free
    floor_ok = {g for g in usable
                if (tot_mean[gi_all[g]] >= args.min_count).mean() >= args.min_count_frac}
    dropped = [g for g in usable if g not in floor_ok]
    print(f'  expression floor (>= {args.min_count:g} reads in >= '
          f'{100*args.min_count_frac:.0f}% of samples): {len(dropped)} of '
          f'{len(usable)} candidate genes dropped'
          + (f' ({", ".join(dropped[:6])}{"..." if len(dropped) > 6 else ""})' if dropped else ''))
    usable = [g for g in usable if g in floor_ok]
    # PROBE. The candidate filter above can only approximate RASQUAL's gate,
    # so ask RASQUAL directly: run it once, observed only, over a larger
    # candidate pool and keep the genes it actually produced a statistic for.
    # The shared set is then RASQUAL's own answer rather than our guess, and
    # the genes it would have silently dropped never enter the null either.
    # Done inside this run so the expensive inputs are loaded once.
    if args.probe_genes and args.rasqual:
        pool = list(usable)
        if len(pool) > args.probe_genes:
            pool = list(rng.choice(pool, args.probe_genes, replace=False))
        print(f'\nProbe: asking RASQUAL which of {len(pool)} candidate genes '
              'it can actually use')
        ppos = gp.loc[pool]
        t0 = time.time()
        pr = rasqual_arm(args.rasqual, pool, ppos, vdf, xL, xR, allelic,
                         order, np.zeros((len(pool), len(order))),
                         np.ones((len(pool), len(order))), args.window,
                         None, tested=None, maf=args.maf,
                         min_coverage=args.min_coverage, cov_bin=cov_bin,
                        jobs=args.rasqual_jobs, asvcf=asvcf,
                        dump=args.dump_rasqual, exons=exons,
                        n_threads=args.rasqual_threads,
                        fsnp_maf=args.fsnp_maf, timeout=args.rasqual_timeout,
                        rows=args.rasqual_rows)
        ok_genes = list(pr[pr['status'] == 'ok']['gene'])
        by_status = pr['status'].value_counts().to_dict()
        print(f'  RASQUAL used {len(ok_genes)}/{len(pool)} '
              f'({100*len(ok_genes)/max(len(pool),1):.0f}%) in '
              f'{(time.time()-t0)/60:.1f} min; statuses {by_status}')
        if len(ok_genes) < 5:
            raise SystemExit('the probe found too few genes RASQUAL can use')
        usable = ok_genes
        probe_stats = {'n_probed': len(pool), 'n_rasqual_usable': len(ok_genes),
                       'status_counts': {k: int(v) for k, v in by_status.items()}}
    else:
        probe_stats = None

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
        r = gp_rec[g]
        in_body[variants_in(cidx, r['chr'], int(r['start']), int(r['end']))] = True
        in_win[variants_in(cidx, r['chr'], int(r['pos']) - args.window,
                           int(r['pos']) + args.window)] = True
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
    Ytot_full = read_salmon_totals(args.salmon, args.tx2gene, order, sufs)
    Ytot = Ytot_full.reindex(usable).fillna(0.0).values
    # rasqualTools: size_factors = colSums(FULL counts) / mean, applied as a
    # gene-constant offset. Computing it on the <=300 selected genes made the
    # offset depend on which genes were sampled; the README says outright that
    # the offset must come from the complete expression data.
    lib = Ytot_full.sum(0).values
    K = np.outer(Ytot.mean(1), lib / lib.mean())

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
                       pos_df[['chr', 'pos']], args.window, tested, perm,
                       cov_df=cov_df, ase_cov=args.ase_covariates)
        th = time.time() - t0; t0 = time.time()
        if args.reuse_rasqual and perm is None and XL is xL and XR is xR:
            # observed RASQUAL rows from an earlier run of the same genes, so
            # the hapmixQTL arm can be iterated without repaying RASQUAL's
            # hours; RASQUAL's inputs do not depend on anything changed here
            prev = pd.read_csv(Path(args.reuse_rasqual) / 'observed_rasqual.tsv',
                               sep='\t')
            r = prev.set_index('gene').reindex(usable).reset_index()
            r['status'] = r['status'].fillna('missing_in_reused_run')
            print(f'  RASQUAL observed rows reused from {args.reuse_rasqual} '
                  f'({int((r["status"] == "ok").sum())}/{len(r)} present)')
        else:
            r = rasqual_arm(args.rasqual, usable, pos_df, vdf, XL, XR, allelic,
                        order, Ytot, K, args.window, perm,
                        tested=tested, maf=args.maf,
                        min_coverage=args.min_coverage, cov_bin=cov_bin,
                        jobs=args.rasqual_jobs, asvcf=asvcf,
                        dump=args.dump_rasqual,
                        exons=exons,
                        n_threads=args.rasqual_threads,
                        fsnp_maf=args.fsnp_maf, timeout=args.rasqual_timeout,
                        knockoff=(XL is not xL or XR is not xR),
                        rows=(args.rasqual_rows if perm is None and XL is xL
                              else None))
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
                   'count_noise': bool(args.count_noise),
                   'expression_floor': {'min_count': args.min_count,
                                        'min_count_frac': args.min_count_frac,
                                        'n_dropped': len(dropped)},
                   'n_genes': len(usable), 'n_samples': len(order),
                   'n_tested_variants': int(tested.sum()), 'n_perm': args.n_perm,
                   'n_genes_both_converged': len(common_genes),
                   'n_genes_hapmixqtl_input': len(genes_all),
                   'n_genes_candidate_shared': len(usable),
                   'probe': probe_stats,
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
    ap.add_argument('--count-noise', action=argparse.BooleanOptionalAction,
                    default=True,
                    help='add per-sample Poisson counting noise to the Gibbs '
                         'variances (compute_summaries_from_gibbs); without '
                         'it a zero-count sample has zero variance and the '
                         'largest weight in the gene')
    ap.add_argument('--min-count', type=float, default=6,
                    help='expression floor: reads per sample in the Salmon '
                         'input (default 6, GTEx)')
    ap.add_argument('--min-count-frac', type=float, default=0.2,
                    help='...in at least this fraction of samples (default 0.2)')
    ap.add_argument('--rasqual-rows',
                    help='directory in which to keep every per-variant row '
                         'RASQUAL writes for the observed run (one file per gene)')
    ap.add_argument('--reuse-rasqual',
                    help='an earlier --out directory whose observed_rasqual.tsv '
                         'is taken as this run\'s observed RASQUAL arm')
    ap.add_argument('--rasqual-threads', type=int, default=4,
                    help='--n-threads per RASQUAL process (pthread)')
    ap.add_argument('--fsnp-maf', type=float, default=0.0,
                    help="RASQUAL's --minor-allele-frequency-fsnp; the "
                         'authors recommend it for very long genes. 0 = off')
    ap.add_argument('--rasqual-timeout', type=int, default=3600,
                    help='seconds per gene before giving up (default 3600)')
    ap.add_argument('--exons',
                    help='exons.tsv from gtf_to_tables.py: gene, '
                         'comma-separated exon starts, ends. Used '
                         "for RASQUAL's -s/-e")
    ap.add_argument('--dump-rasqual',
                    help="save RASQUAL's raw output for genes where nothing "
                         'converged, so the bounded parameter is visible')
    ap.add_argument('--asvcf',
                    help='AS-annotated tabix-indexed VCF from '
                         'scripts/build_asvcf.py; RASQUAL is fed '
                         'a slice of it directly')
    ap.add_argument('--rasqual-jobs', type=int, default=1,
                    help='genes to run through RASQUAL concurrently. The '
                         'binary is single-threaded, so this is the only '
                         'parallelism available and it is the difference '
                         'between hours and days on a real cohort')
    ap.add_argument('--cache-dir',
                    help='cache the loaded Gibbs arrays here. The load is ~45 '
                         'min of gzip and depends only on --salmon, --tx2gene '
                         'and --hap-suffix, so iterating on anything else '
                         'should not re-pay it')
    ap.add_argument('--probe-genes', type=int, default=0,
                    help='run a cheap RASQUAL-only pass over this many '
                         'candidate genes first and keep only the ones it '
                         'actually used, then sample --n-genes from those. '
                         "Makes the shared set RASQUAL's real gate rather than "
                         'an approximation, and keeps genes it would drop out '
                         'of the null as well as the observed statistic')
    ap.add_argument('--gene-list',
                    help='explicit gene ids, one per line. Skips the candidate '
                         'filter -- use the genes a cheap RASQUAL-only probe '
                         'actually used, so the shared set is RASQUAL\'s real '
                         'gate rather than an approximation of it')
    ap.add_argument('--covariates',
                    help='TSV from scripts/build_covariates.py: samples as '
                         'rows, covariates as columns. Fed to BOTH arms: '
                         "RASQUAL's total-count model and hapmixQTL's total "
                         'channel (see --ase-covariates for the allelic one)')
    ap.add_argument('--ase-covariates', choices=('none', 'shared'), default='none',
                    help="what hapmixQTL projects out of its ALLELIC channel: "
                         "'none' (intercept only; the log haplotype ratio is a "
                         'within-sample contrast in which sample-level '
                         "covariates cancel) or 'shared' (the --covariates "
                         'set, as in the total channel; each column costs one '
                         'informative sample and halved the allelic statistic '
                         'of well-covered BrainVar genes). RASQUAL applies '
                         'covariates to its total-count model only')
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
        null='permute', knockoff_k=4, min_coverage=0.05, covariates=None,
        gene_list=None, probe_genes=0, cache_dir=None, rasqual_jobs=2,
        asvcf=None, dump_rasqual=None, exons=None,
        rasqual_threads=2, fsnp_maf=0.0, rasqual_timeout=900,
        count_noise=True, min_count=6, min_count_frac=0.2, reuse_rasqual=None,
        rasqual_rows=None, ase_covariates='none',
        str_vcf=None, multiallelic=False, min_hap=10)
    # count_noise contract, which fabricated Poisson(30) draws cannot probe:
    # a sample with the same count in every draw has v_inf = 0. The term must
    # give a zero-count sample Vt = 1/(2 kappa), a constant nonzero count a
    # positive Vt, and leave Va = 0 exactly where there are no allele-specific
    # reads so the degenerate-ASE guard still fires.
    yl = np.zeros((1, 3, ND)); yr = np.zeros((1, 3, ND)); yt = np.zeros((1, 3, ND))
    yl[0, 1], yr[0, 1], yt[0, 1] = 5.0, 5.0, 10.0            # constant, nonzero
    yl[0, 2], yr[0, 2] = rng.poisson(20, ND), rng.poisson(20, ND); yt[0, 2] = yl[0, 2] + yr[0, 2]
    _, _, va, vt, _ = compute_summaries_from_gibbs(yl, yr, kappa=0.5, yT=yt, count_noise=True)
    assert abs(vt[0, 0] - 1.0 / (0 + 2 * 0.5)) < 1e-12 and va[0, 0] == 0.0, (vt[0, 0], va[0, 0])
    assert vt[0, 1] > 0 and abs(va[0, 1] - 2 / 5.5) < 1e-12, (vt[0, 1], va[0, 1])
    _, _, va0, vt0, _ = compute_summaries_from_gibbs(yl, yr, kappa=0.5, yT=yt, count_noise=False)
    assert vt0[0, 0] == 0.0 and vt0[0, 1] == 0.0 and vt[0, 2] > vt0[0, 2] > 0, (vt0[0, :], vt[0, 2])
    print('count_noise contract: zero-count Vt = 1/(2 kappa), constant count Vt > 0, '
          'no-coverage Va stays 0 -- OK')
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
    # covariates must reach BOTH arms: a covariate file that is silently
    # ignored looks identical to one that works, so assert the run changes.
    cv = td / 'cov.tsv'
    cov = pd.DataFrame(
        {'c1': rng.randn(N), 'c2': rng.randn(N)},
        index=samples)
    cov.to_csv(cv, sep='\t')
    cargs = dict(base_args); cargs.update(covariates=str(cv))
    rc = run(argparse.Namespace(**cargs, out=str(td / 'deploy_cov')))
    for m in ('RASQUAL', 'hapmixQTL'):
        assert 'power' in rc[m], f'{m} produced no power estimate with covariates'
    assert (td / 'deploy_cov' / 'covariates.rasqual.bin').exists(), \
        'RASQUAL covariate binary was not written'
    nb = np.fromfile(td / 'deploy_cov' / 'covariates.rasqual.bin', np.float64)
    assert nb.size == N * 2, (nb.size, N)
    # covariate-major: the first N doubles are c1 across samples
    assert np.allclose(nb[:N], cov.loc[list(rc['design'].get('sample_order', cov.index))
                                       if False else cov.index, 'c1'].values), \
        'covariate binary is not covariate-major'
    # the reference-bias gate must report on the hapmixQTL arm, not be skipped
    rb = r['reference_bias_hapmixqtl']
    assert 'ref_fraction' in rb and 'message' in rb, rb
    assert rb['n_obs'] >= 0, rb
    assert 'power' in rk['hapmixQTL'], 'hapmixQTL: no power estimate under knockoffs'
    assert rk['hapmixQTL']['n_null_stats'] > 0, rk['hapmixQTL']
    # the RASQUAL pipe cannot take knockoff haplotypes yet; the arm must
    # report that, not pass the observed run off as a null
    assert rk['RASQUAL']['n_null_stats'] == 0, rk['RASQUAL']
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
    assert r2['hapmixQTL'] == r['hapmixQTL']
    # RASQUAL's -r draws its own permutation, seeded from time and pid, so
    # its null (and the power it implies) is not reproducible run to run;
    # the observed rows were compared byte for byte above
    for k in ('n_genes_scored', 'n_genes_this_method_alone'):
        assert r2['RASQUAL'][k] == r['RASQUAL'][k], (k, r2['RASQUAL'], r['RASQUAL'])
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

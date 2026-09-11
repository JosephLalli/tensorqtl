#!/usr/bin/env python3
"""
Integrated STR + biallelic cis-QTL calling: encode STRs so hapmixQTL tests them
alongside SNPs with NO change to the model.

THE IDEA
========
hapmixQTL's two channels are linear in two per-variant quantities:

    total   t ~ beta * (g / 2)          g  = xL + xR
    ASE     a ~ beta * s                s  = xL - xR

For a biallelic SNP, xL/xR are 0/1 per haplotype. Nothing in the regression
core requires that. If instead xL and xR are each haplotype's STR REPEAT LENGTH
(as a difference from the reference, in repeat units -- the Gymrek 2016 /
Fotsing 2019 eSTR convention), then

    g / 2  = mean repeat length          -> total channel: expression vs length
    s      = L_A - L_B                   -> ASE channel: does the LONGER
                                            haplotype express more?
    beta   = log aFC PER REPEAT UNIT

Same structure, same code, same permutation. The ASE channel becomes a
within-individual test of length on expression, which is something a
biallelic-rSNP model (RASQUAL's D_il) cannot express. That is a differentiator.

Because map_cis takes the lead variant over the whole window and calibrates by
permutation, SNPs and STRs compete in ONE test with ONE FDR: the lead can be an
STR, and the multiple testing is handled jointly and automatically.

WHAT THIS IS, AND WHAT IT IS NOT
================================
This is the LINEAR-IN-LENGTH model: 1 df, drops into the existing machinery.
It assumes expression changes monotonically per repeat unit. Real STR effects
can be non-linear (threshold, plateau, quadratic); those need per-allele or
spline terms and a multi-df test -- "true multiallelic support", the next step,
NOT this one. The linear model still detects most eSTRs (it is what the eSTR
literature uses) and is the right first integration.

ENCODING RULES
==============
  * length = (allele length - reference length) / PERIOD, in repeat units.
    Reference = 0, so beta is interpretable and the intercept is the reference.
  * PHASED call (HipSTR run on a phased SNP scaffold, GT uses '|'):
        xL = L_A, xR = L_B          both channels active
  * UNPHASED call (GT uses '/'):
        xL = xR = (L_A + L_B) / 2   g is exact (sum is phase-invariant),
                                    s = 0 so the ASE channel drops out --
                                    the existing, tested no-phase path.
  * HipSTR's FORMAT/GB ("base-pair difference of each allele from reference",
    e.g. 0|-12) is used when present; otherwise lengths come from the GT allele
    indices and REF/ALT sequences. INFO/PERIOD gives the unit; --period
    overrides.
  * QUALITY. STR calls are noisier than SNPs. Per-sample calls below --min-q
    (HipSTR FORMAT/Q; default 0.9 per HipSTR's guidance) are set missing and
    imputed to the locus mean length; loci with call rate below --min-call-rate
    or with no length variation are dropped.

REQUIRED DOWNSTREAM SETTING
===========================
Run map_cis / map_nominal with maf_threshold=0 (the default). A MAF filter
assumes 0/1/2 dosages and would apply a meaningless criterion to STR rows.
The STR-appropriate filters (quality, call rate, variation) are applied HERE.
The `af` / `ma_samples` output columns are likewise meaningless for STR rows.

OUTPUT  (a "hapdose" directory, loadable with load_hapdose)
======
  variants.tsv     id  chrom  pos  type(snp|str)  period  n_het  phased_frac
  xL.npy, xR.npy   [variants x samples] per-haplotype dosage (float)
  samples.txt

Run:
  python3 scripts/str_integrate.py --selftest
  python3 scripts/str_integrate.py --snp-vcf rephased.vcf.gz \\
      --str-vcf hipstr.vcf.gz --samples samples.txt --out hapdose/
"""

import argparse
import gzip
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))


def _open(p):
    return gzip.open(p, 'rt') if str(p).endswith('.gz') else open(p)


# ---------------------------------------------------------------------------
#  STR VCF -> per-haplotype repeat-length dosage
# ---------------------------------------------------------------------------

def parse_str_vcf(path, samples, min_q=0.9, min_call_rate=0.8, period_override=None):
    """Return list of dict(id, chrom, pos, period, xL[N], xR[N], phased_frac, n_het)."""
    want = {s: i for i, s in enumerate(samples)}
    out = []
    n_seen = n_dropped_rate = n_dropped_const = 0
    with _open(path) as fh:
        col = None
        for line in fh:
            if line.startswith('##'):
                continue
            f = line.rstrip('\n').split('\t')
            if line.startswith('#CHROM'):
                vs = f[9:]
                missing = [s for s in samples if s not in vs]
                if missing:
                    raise SystemExit(f'samples not in STR VCF: {missing[:3]}')
                col = [9 + vs.index(s) for s in samples]
                continue
            if len(f) < 10:
                continue
            n_seen += 1
            info = dict(kv.split('=', 1) for kv in f[7].split(';') if '=' in kv)
            period = period_override or int(float(info.get('PERIOD', 0)) or 0)
            if period <= 0:
                continue                                   # cannot convert to units
            ref_len = len(f[3])
            alts = f[4].split(',')
            allele_len = [ref_len] + [len(a) for a in alts]  # index -> bp length
            fmt = f[8].split(':')
            gi = fmt.index('GT') if 'GT' in fmt else 0
            gb = fmt.index('GB') if 'GB' in fmt else None
            qi = fmt.index('Q') if 'Q' in fmt else None
            N = len(samples)
            LA = np.full(N, np.nan); LB = np.full(N, np.nan); phased = np.zeros(N, bool)
            for k, c in enumerate(col):
                parts = f[c].split(':')
                gt = parts[gi] if gi < len(parts) else '.'
                if gt in ('.', './.', '.|.') or '.' in gt.replace('|', '/').split('/'):
                    continue
                if qi is not None and qi < len(parts):
                    try:
                        if float(parts[qi]) < min_q:
                            continue                   # low-quality call -> missing
                    except ValueError:
                        pass
                sep = '|' if '|' in gt else '/'
                a, b = gt.split(sep)[:2]
                if gb is not None and gb < len(parts) and parts[gb] not in ('.', ''):
                    # HipSTR GB: bp difference from reference per allele
                    da, db = parts[gb].replace('|', '/').split('/')[:2]
                    la, lb = float(da) / period, float(db) / period
                else:
                    try:
                        la = (allele_len[int(a)] - ref_len) / period
                        lb = (allele_len[int(b)] - ref_len) / period
                    except (ValueError, IndexError):
                        continue
                LA[k], LB[k] = la, lb; phased[k] = (sep == '|')
            called = ~np.isnan(LA)
            if called.mean() < min_call_rate:
                n_dropped_rate += 1; continue
            # impute missing to the locus mean length (like mean-imputed dosage)
            mean_len = np.nanmean(np.concatenate([LA[called], LB[called]]))
            LA[~called] = mean_len; LB[~called] = mean_len
            # unphased: g is exact via the sum, s must be 0 -> xL = xR = mean
            m = (LA + LB) / 2.0
            xL = np.where(phased, LA, m); xR = np.where(phased, LB, m)
            if np.std(xL + xR) < 1e-9:
                n_dropped_const += 1; continue              # no length variation
            vid = f[2] if f[2] not in ('.', '') else f'{f[0]}_{f[1]}_STR'
            out.append(dict(id=vid, chrom=str(f[0]), pos=int(f[1]), type='str',
                            period=period, xL=xL, xR=xR,
                            n_het=int(np.sum(np.abs(LA - LB) > 1e-9)),
                            phased_frac=float(phased[called].mean()) if called.any() else 0.0))
    print(f'  STR: {n_seen} loci read, {len(out)} kept '
          f'({n_dropped_rate} below call rate {min_call_rate}, '
          f'{n_dropped_const} with no length variation)')
    return out


# ---------------------------------------------------------------------------
#  Merge with biallelic SNPs -> hapdose
# ---------------------------------------------------------------------------

def build_hapdose(samples, str_rows, snp_vcf=None):
    rows, XL, XR = [], [], []
    if snp_vcf:
        from make_rasqual_inputs import read_vcf
        vdf, sXL, sXR = read_vcf(snp_vcf, samples)
        for i in range(len(vdf)):
            rows.append(dict(id=vdf.id.iat[i], chrom=str(vdf.chrom.iat[i]),
                             pos=int(vdf.pos.iat[i]), type='snp', period=1,
                             n_het=int(np.sum(sXL[i] != sXR[i])), phased_frac=1.0))
            XL.append(sXL[i]); XR.append(sXR[i])
        print(f'  SNP: {len(vdf)} phased biallelic variants')
    for r in str_rows:
        rows.append({k: r[k] for k in ('id', 'chrom', 'pos', 'type', 'period',
                                        'n_het', 'phased_frac')})
        XL.append(r['xL']); XR.append(r['xR'])
    if not rows:
        raise SystemExit('no variants')
    vdf = pd.DataFrame(rows)
    order = np.lexsort((vdf['pos'].values, vdf['chrom'].values))
    vdf = vdf.iloc[order].reset_index(drop=True)
    XL = np.array(XL, float)[order]; XR = np.array(XR, float)[order]
    return vdf, XL, XR


def write_hapdose(out, vdf, XL, XR, samples):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    vdf.to_csv(out / 'variants.tsv', sep='\t', index=False)
    np.save(out / 'xL.npy', XL); np.save(out / 'xR.npy', XR)
    (out / 'samples.txt').write_text('\n'.join(samples))


def load_hapdose(d):
    """-> genotype_df (g = xL+xR), variant_df(chrom,pos), xL_df, xR_df, variants."""
    d = Path(d)
    vdf = pd.read_csv(d / 'variants.tsv', sep='\t', dtype={'chrom': str})
    samples = [l.strip() for l in open(d / 'samples.txt') if l.strip()]
    XL = np.load(d / 'xL.npy'); XR = np.load(d / 'xR.npy')
    idx = vdf['id'].values
    return (pd.DataFrame(XL + XR, index=idx, columns=samples),
            vdf.set_index('id')[['chrom', 'pos']],
            pd.DataFrame(XL, index=idx, columns=samples),
            pd.DataFrame(XR, index=idx, columns=samples), vdf)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--str-vcf'); ap.add_argument('--snp-vcf')
    ap.add_argument('--samples'); ap.add_argument('--out', default='hapdose')
    ap.add_argument('--min-q', type=float, default=0.9)
    ap.add_argument('--min-call-rate', type=float, default=0.8)
    ap.add_argument('--period', type=int, default=None,
                    help='override INFO/PERIOD for every locus')
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    if not (args.str_vcf and args.samples):
        raise SystemExit('--str-vcf and --samples are required (or --selftest)')
    samples = [l.strip() for l in open(args.samples) if l.strip()]
    print(f'{len(samples)} samples')
    strs = parse_str_vcf(args.str_vcf, samples, args.min_q, args.min_call_rate, args.period)
    vdf, XL, XR = build_hapdose(samples, strs, args.snp_vcf)
    write_hapdose(args.out, vdf, XL, XR, samples)
    n_str = int((vdf['type'] == 'str').sum())
    print(f'wrote {args.out}/: {len(vdf)} variants ({n_str} STR, {len(vdf)-n_str} SNP)')
    print('Run map_cis with maf_threshold=0; the af/ma_samples columns are '
          'meaningless for STR rows.')


# ---------------------------------------------------------------------------
#  Validation: planted per-repeat-unit effects, real map_cis, mixed window
# ---------------------------------------------------------------------------

def selftest():
    import contextlib, io, tempfile, warnings
    warnings.filterwarnings('ignore')
    try:
        from tensorqtl.hapmixqtl import map_cis
    except ImportError:
        sys.path.insert(0, str(HERE.parent / 'tensorqtl')); from hapmixqtl import map_cis
    td = Path(tempfile.mkdtemp()); rng = np.random.RandomState(0)
    N = 200
    samples = [f'S{i:03d}' for i in range(N)]
    (td / 'samples.txt').write_text('\n'.join(samples))

    # ---- STR VCF, HipSTR-style: PERIOD in INFO, GT:GB:Q per sample ----------
    # loci: 0,1 planted eSTRs (phased); 2 planted but UNPHASED; 3,4 null;
    #       5 low-quality calls at 15% of samples (imputed, locus kept);
    #       6 low-quality calls at 40% of samples (below call rate 0.8 -> dropped)
    period = 3
    str_L = {}
    low_q_frac = {5: 0.15, 6: 0.40}
    lines = ['##fileformat=VCFv4.2',
             '#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t' + '\t'.join(samples)]
    for j in range(7):
        pos = 1_000_000 * j + 5000
        alleles_units = [-2, -1, 0, 1, 2, 3]           # repeat-unit diffs available
        LA = rng.choice(alleles_units, N); LB = rng.choice(alleles_units, N)
        str_L[j] = (LA.astype(float), LB.astype(float))
        ref = 'CAG' * 8
        alts = ['CAG' * (8 + u) for u in alleles_units if u != 0]
        alt_idx = {u: i + 1 for i, u in enumerate([u for u in alleles_units if u != 0])}
        sep = '/' if j == 2 else '|'
        cells = []
        for k in range(N):
            ia = 0 if LA[k] == 0 else alt_idx[LA[k]]
            ib = 0 if LB[k] == 0 else alt_idx[LB[k]]
            q = 0.5 if rng.rand() < low_q_frac.get(j, 0.0) else 0.99
            cells.append(f'{ia}{sep}{ib}:{int(LA[k]*period)}{sep}{int(LB[k]*period)}:{q}')
        lines.append(f'1\t{pos}\tSTR{j}\t{ref}\t{",".join(alts)}\t.\tPASS\t'
                     f'PERIOD={period}\tGT:GB:Q\t' + '\t'.join(cells))
    (td / 'str.vcf').write_text('\n'.join(lines) + '\n')

    # ---- SNP VCF: 5 null SNPs per STR window + 1 planted eSNP window --------
    lines = ['##fileformat=VCFv4.2',
             '#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t' + '\t'.join(samples)]
    snp_x = {}
    for j in range(7):
        for m in range(5):
            pos = 1_000_000 * j + 3000 + 300 * m
            xl = (rng.rand(N) < .4).astype(int); xr = (rng.rand(N) < .4).astype(int)
            snp_x[(j, m)] = (xl.astype(float), xr.astype(float))
            lines.append(f'1\t{pos}\tSNP{j}_{m}\tA\tG\t.\tPASS\t.\tGT\t'
                         + '\t'.join(f'{xl[k]}|{xr[k]}' for k in range(N)))
    (td / 'snp.vcf').write_text('\n'.join(lines) + '\n')

    print('SELF-TEST: encoding STRs + SNPs into one hapdose matrix\n')
    main(['--str-vcf', str(td / 'str.vcf'), '--snp-vcf', str(td / 'snp.vcf'),
          '--samples', str(td / 'samples.txt'), '--out', str(td / 'hd')])
    g_df, v_df, xl_df, xr_df, vinfo = load_hapdose(td / 'hd')
    assert set(vinfo['type']) == {'snp', 'str'} and (vinfo['type'] == 'str').sum() == 6
    # encoding checks against the truth
    LA, LB = str_L[0]
    assert np.allclose(xl_df.loc['STR0'].values, LA) and np.allclose(xr_df.loc['STR0'].values, LB), \
        'phased STR must map hapA->xL, hapB->xR in repeat units'
    LA2, LB2 = str_L[2]
    assert np.allclose(xl_df.loc['STR2'].values, xr_df.loc['STR2'].values), \
        'unphased STR must have xL == xR (s = 0)'
    assert np.allclose(xl_df.loc['STR2'].values + xr_df.loc['STR2'].values, LA2 + LB2), \
        'unphased STR total dosage must still equal the true sum'
    assert 'STR5' in xl_df.index, 'low-Q calls at 15% must be imputed, not drop the locus'
    assert 'STR6' not in xl_df.index, 'locus with 40% low-Q calls must fail the call-rate floor'
    LA5, LB5 = str_L[5]
    ok5 = np.isclose(xl_df.loc['STR5'].values, LA5)
    assert 0.75 < ok5.mean() < 0.95, 'imputed sites should be the ~15% low-Q calls only'
    print('encoding: phased -> (L_A, L_B); unphased -> (mean, mean) with exact sum; '
          '15% low-Q imputed to locus mean; 40% low-Q locus dropped by call rate')

    # ---- expression with PLANTED per-repeat-unit effects; run the REAL map_cis
    genes = [f'G{j}' for j in range(7)]
    beta = {0: 0.35, 1: -0.30, 2: 0.35, 6: 0.60}        # 6 = eSNP gene
    A = np.zeros((7, N)); T = np.zeros((7, N))
    for j in range(7):
        if j < 6:
            LA, LB = str_L[j]; s = LA - LB; g2 = (LA + LB) / 2
        else:
            xl, xr = snp_x[(6, 2)]; s = xl - xr; g2 = (xl + xr) / 2
        b = beta.get(j, 0.0)
        A[j] = b * s + rng.normal(0, 0.35, N)
        T[j] = 2.0 + b * g2 + rng.normal(0, 0.35, N)
    Va = np.full((7, N), 0.02); Vt = np.full((7, N), 0.02)
    mk = lambda M: pd.DataFrame(M, index=genes, columns=samples)
    pos_df = pd.DataFrame({'chr': ['1'] * 7, 'pos': [1_000_000 * j + 5000 for j in range(7)]},
                          index=genes)
    with contextlib.redirect_stdout(io.StringIO()):
        res = map_cis(g_df, v_df, mk(A), mk(T), mk(Va), mk(Vt), pos_df,
                      xL_df=xl_df, xR_df=xr_df, window=20000, nperm=200,
                      verbose=False)
    print('\nmap_cis on the mixed window (lead variant per gene):')
    print(f"  {'gene':5s} {'lead':10s} {'slope':>8s} {'planted':>8s} {'pval_beta':>10s}")
    for j, g in enumerate(genes):
        r = res.loc[g]
        print(f"  {g:5s} {r['variant_id']:10s} {r['slope']:8.3f} {beta.get(j,0):8.2f} "
              f"{r['pval_beta']:10.2e}")
    # assertions: the integrated call must find STR leads with the right beta
    for j in (0, 1):
        r = res.loc[f'G{j}']
        assert r['variant_id'] == f'STR{j}', f'G{j} lead should be the STR, got {r["variant_id"]}'
        assert abs(r['slope'] - beta[j]) < 0.08, f'G{j} beta per repeat unit off: {r["slope"]}'
        assert r['pval_beta'] < 1e-3
    r = res.loc['G2']          # unphased: total channel alone must still carry it
    assert r['variant_id'] == 'STR2' and r['pval_beta'] < 1e-2 and abs(r['slope'] - 0.35) < 0.12
    r = res.loc['G6']          # SNP path unchanged
    assert r['variant_id'] == 'SNP6_2' and abs(r['slope'] - 0.60) < 0.1
    for j in (3, 4, 5):        # nulls stay null
        assert res.loc[f'G{j}', 'pval_beta'] > 0.05, f'null G{j} called'
    print('\nchecks: STR leads recovered with beta per repeat unit; unphased STR '
          'detected via total channel; SNP unchanged; nulls null; joint lead over '
          'SNPs+STRs works')
    print('SELF-TEST OK')
    return 0


if __name__ == '__main__':
    main()

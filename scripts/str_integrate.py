#!/usr/bin/env python3
"""
Integrated STR + multiallelic + biallelic cis-QTL calling for hapmixQTL.

Encodes every variant class into the per-haplotype dosage matrices (xL, xR)
the lead scan already uses, and writes the sidecars the two second-pass
regressions need. Three variant classes, three treatments:

  biallelic SNP      xL/xR = 0/1 per haplotype               lead scan (1 df)
  STR                xL/xR = repeat length per haplotype     lead scan (1 df, LINEAR in length)
                     + sidecar of raw lengths                second pass: LINEAR + CURVATURE
  multi-ALT site     one split row per ALT allele            lead scan (1 df each)
  (SNV / indel)      + sidecar of allele index per haplotype second pass: CATEGORICAL (joint K-1 df)

LEAD SCAN: NO MODEL CHANGE
==========================
hapmixQTL's two channels are linear in two per-variant quantities:

    total   t ~ beta * (g / 2)          g  = xL + xR
    ASE     a ~ beta * s                s  = xL - xR

Nothing in the regression core requires xL/xR to be 0/1. For an STR, xL and
xR are each haplotype's repeat length (difference from the reference, in
repeat units -- the Gymrek 2016 / Fotsing 2019 eSTR convention), so g/2 is
the mean length, s = L_A - L_B asks whether the LONGER haplotype expresses
more, and beta is log aFC PER REPEAT UNIT. For a multi-ALT site, the split
row for ALT k is the allele-k indicator per haplotype, exactly the row a
`bcftools norm -m-` pipeline would test. SNPs, STRs and split rows compete
in ONE lead-variant test with ONE permutation FDR.

SECOND PASS (tensorqtl.hapmixqtl.map_str_curvature / map_multiallelic)
=====================================================================
STR: repeat sites are modeled linearly by length, or linearly plus a
non-linear component. The second pass fits f(L) = b1 L + b2 L^2 PER
HAPLOTYPE (the total row is (f(L_A) + f(L_B))/2, not f of the mean), and
reports b2 as a 1-df curvature test next to the linear slope. Needs the raw
allele pair per sample, which the linear row loses for unphased calls; hence
the sidecar.

Multi-ALT non-repeat sites: the categorical model fits the split rows
JOINTLY, giving each allele a log aFC against a clean reference allele and a
joint K-1 df test of "does allele identity matter", with no ordering
assumed. The split rows in the lead scan are fit marginally (allele k vs
everything else), which is biased when another ALT allele also has an
effect; the joint fit is not.

ENCODING RULES
==============
  STR
  * length = (allele length - reference length) / PERIOD, in repeat units.
  * PHASED call (GT uses '|'):   xL = L_A, xR = L_B
  * UNPHASED call (GT uses '/'): xL = xR = (L_A + L_B)/2  (g exact, s = 0)
  * HipSTR FORMAT/GB used when present, else lengths from GT + REF/ALT.
    INFO/PERIOD gives the unit; --period overrides.
  * Per-sample calls with FORMAT/Q below --min-q are missing and imputed to
    the locus mean length; loci below --min-call-rate or with no length
    variation are dropped. The sidecar keeps missing as NaN.

  MULTI-ALT SITE (any row whose ALT has a comma; REF/ALT of any length)
  * split row per ALT allele carried by >= 1 haplotype:
        phased:   xL = [hapA == k], xR = [hapB == k]
        unphased: xL = xR = mean of the two indicators
        missing:  both = allele frequency among called haplotypes
  * sidecar (only sites with >= 3 alleles observed in the cohort): allele
    index per haplotype, -1 missing, plus a phased flag.

REQUIRED DOWNSTREAM SETTING
===========================
Run map_cis / map_nominal with maf_threshold=0 (the default). A MAF filter
assumes 0/1/2 dosages. The `af` / `ma_samples` output columns are
meaningless for STR rows.

OUTPUT  (a "hapdose" directory)
======
  variants.tsv     id chrom pos type(snp|str|ma_allele) period n_het phased_frac site allele
  xL.npy, xR.npy   [variants x samples] per-haplotype dosage (float)
  samples.txt
  str_sites.tsv, str_len.npy [n_str x samples x 2], str_phased.npy      (if any STR)
  ma_sites.tsv,  ma_alleles.npy [n_ma x samples x 2], ma_phased.npy     (if any multi-ALT)

  load_hapdose(d) -> inputs for map_cis;  load_aux(d) -> inputs for the second pass.

Run:
  python3 scripts/str_integrate.py --selftest
  python3 scripts/str_integrate.py --snp-vcf rephased.vcf.gz \\
      --str-vcf hipstr.vcf.gz --samples samples.txt --out hapdose/
"""

import argparse
import gzip
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))


def _open(p):
    return gzip.open(p, 'rt') if str(p).endswith('.gz') else open(p)


def _vcf_rows(path, samples):
    """Yield (fields, sample_columns) for every data row of a VCF."""
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
                    raise SystemExit(f'samples not in {path}: {missing[:3]}')
                col = [9 + vs.index(s) for s in samples]
                continue
            if len(f) < 10 or col is None:
                continue
            yield f, col


# ---------------------------------------------------------------------------
#  STR VCF -> per-haplotype repeat-length dosage (+ raw lengths sidecar)
# ---------------------------------------------------------------------------

def parse_str_vcf(path, samples, min_q=0.9, min_call_rate=0.8, period_override=None):
    """Return list of dict(id, chrom, pos, period, xL[N], xR[N], LA[N], LB[N],
    phased[N], phased_frac, n_het). LA/LB are raw lengths with NaN = missing."""
    out = []
    n_seen = n_dropped_rate = n_dropped_const = 0
    N = len(samples)
    for f, col in _vcf_rows(path, samples):
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
        LA_raw, LB_raw = LA.copy(), LB.copy()
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
                        period=period, xL=xL, xR=xR, LA=LA_raw, LB=LB_raw, phased=phased,
                        n_het=int(np.sum(np.abs(LA - LB) > 1e-9)),
                        phased_frac=float(phased[called].mean()) if called.any() else 0.0))
    print(f'  STR: {n_seen} loci read, {len(out)} kept '
          f'({n_dropped_rate} below call rate {min_call_rate}, '
          f'{n_dropped_const} with no length variation)')
    return out


# ---------------------------------------------------------------------------
#  Multi-ALT sites -> split rows (lead scan) + allele-index sidecar (categorical)
# ---------------------------------------------------------------------------

def parse_multiallelic_vcf(path, samples):
    """Rows whose ALT has a comma -> list of dict(site_id, chrom, pos, ref, alts,
    hapA[N], hapB[N] (int, -1 missing), phased[N], n_alleles_obs,
    rows=[dict(id, allele, xL, xR, n_het)] one split row per carried ALT)."""
    out = []
    n_seen = 0
    N = len(samples)
    for f, col in _vcf_rows(path, samples):
        if ',' not in f[4]:
            continue
        n_seen += 1
        alts = f[4].split(',')
        fmt = f[8].split(':')
        gi = fmt.index('GT') if 'GT' in fmt else 0
        hapA = np.full(N, -1, int); hapB = np.full(N, -1, int); phased = np.zeros(N, bool)
        for k, c in enumerate(col):
            parts = f[c].split(':')
            gt = parts[gi] if gi < len(parts) else '.'
            sep = '|' if '|' in gt else '/'
            ab = gt.split(sep)
            if len(ab) < 2:
                continue
            a, b = ab[:2]
            if a == '.' or b == '.':
                continue
            try:
                hapA[k], hapB[k] = int(a), int(b)
            except ValueError:
                continue
            phased[k] = (sep == '|')
        called = (hapA >= 0) & (hapB >= 0)
        haps = np.concatenate([hapA[called], hapB[called]])
        if haps.size == 0:
            continue
        obs, cnt = np.unique(haps, return_counts=True)
        n_called_hap = int(2 * called.sum())
        rows = []
        for k in range(1, len(alts) + 1):
            n_k = int(cnt[obs == k].sum()) if (obs == k).any() else 0
            if n_k == 0 or n_k == n_called_hap:
                continue                                 # not variable here
            eA = (hapA == k).astype(float); eB = (hapB == k).astype(float)
            freq = n_k / n_called_hap
            eA[~called] = freq; eB[~called] = freq
            m = (eA + eB) / 2.0
            xL = np.where(phased & called, eA, m); xR = np.where(phased & called, eB, m)
            rows.append(dict(id=f'{f[0]}_{f[1]}_{f[3]}_{alts[k - 1]}', allele=k,
                             xL=xL, xR=xR, n_het=int(np.sum(eA[called] != eB[called]))))
        if not rows:
            continue
        site_id = f[2] if f[2] not in ('.', '') else f'{f[0]}_{f[1]}'
        out.append(dict(site_id=site_id, chrom=str(f[0]), pos=int(f[1]), ref=f[3],
                        alts=f[4], hapA=hapA, hapB=hapB, phased=phased,
                        n_alleles_obs=int(len(obs)),
                        phased_frac=float(phased[called].mean()) if called.any() else 0.0,
                        rows=rows))
    n_cat = sum(s['n_alleles_obs'] >= 3 for s in out)
    n_rows = sum(len(s['rows']) for s in out)
    print(f'  multi-ALT: {n_seen} sites read, {len(out)} variable -> {n_rows} split rows '
          f'for the lead scan; {n_cat} with >= 3 alleles observed -> categorical sidecar')
    return out


# ---------------------------------------------------------------------------
#  Merge everything -> hapdose
# ---------------------------------------------------------------------------

def build_hapdose(samples, str_rows, snp_vcf=None, ma_sites=None):
    """Return (variants_df, XL, XR, aux) with aux holding the sidecar arrays in
    variants_df order (str_*) or (chrom, pos) order (ma_*)."""
    rows, XL, XR = [], [], []
    if snp_vcf:
        from make_rasqual_inputs import read_vcf
        vdf, sXL, sXR = read_vcf(snp_vcf, samples)
        for i in range(len(vdf)):
            rows.append(dict(id=vdf.id.iat[i], chrom=str(vdf.chrom.iat[i]),
                             pos=int(vdf.pos.iat[i]), type='snp', period=1,
                             n_het=int(np.sum(sXL[i] != sXR[i])), phased_frac=1.0,
                             site='', allele=''))
            XL.append(sXL[i]); XR.append(sXR[i])
        print(f'  SNP: {len(vdf)} phased biallelic variants')
    for r in str_rows:
        rows.append(dict(id=r['id'], chrom=r['chrom'], pos=r['pos'], type='str',
                         period=r['period'], n_het=r['n_het'], phased_frac=r['phased_frac'],
                         site='', allele=''))
        XL.append(r['xL']); XR.append(r['xR'])
    for s in (ma_sites or []):
        for r in s['rows']:
            rows.append(dict(id=r['id'], chrom=s['chrom'], pos=s['pos'], type='ma_allele',
                             period=1, n_het=r['n_het'], phased_frac=s['phased_frac'],
                             site=s['site_id'], allele=r['allele']))
            XL.append(r['xL']); XR.append(r['xR'])
    if not rows:
        raise SystemExit('no variants')
    vdf = pd.DataFrame(rows)
    order = np.lexsort((vdf['pos'].values, vdf['chrom'].values))
    vdf = vdf.iloc[order].reset_index(drop=True)
    XL = np.array(XL, float)[order]; XR = np.array(XR, float)[order]

    aux = {}
    if str_rows:
        by_id = {r['id']: r for r in str_rows}
        ids = [i for i in vdf['id'] if i in by_id]
        aux['str_sites'] = pd.DataFrame([dict(id=i, chrom=by_id[i]['chrom'], pos=by_id[i]['pos'],
                                              period=by_id[i]['period']) for i in ids])
        aux['str_len'] = np.stack([np.stack([by_id[i]['LA'], by_id[i]['LB']], axis=1)
                                   for i in ids]).astype(np.float32)
        aux['str_phased'] = np.stack([by_id[i]['phased'] for i in ids])
    cat = [s for s in (ma_sites or []) if s['n_alleles_obs'] >= 3]
    if cat:
        cat.sort(key=lambda s: (s['chrom'], s['pos']))
        aux['ma_sites'] = pd.DataFrame([dict(site_id=s['site_id'], chrom=s['chrom'], pos=s['pos'],
                                             ref=s['ref'], alts=s['alts'],
                                             n_alleles=s['n_alleles_obs']) for s in cat])
        aux['ma_alleles'] = np.stack([np.stack([s['hapA'], s['hapB']], axis=1)
                                      for s in cat]).astype(np.int16)
        aux['ma_phased'] = np.stack([s['phased'] for s in cat])
    return vdf, XL, XR, aux


def write_hapdose(out, vdf, XL, XR, samples, aux=None):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    vdf.to_csv(out / 'variants.tsv', sep='\t', index=False)
    np.save(out / 'xL.npy', XL); np.save(out / 'xR.npy', XR)
    (out / 'samples.txt').write_text('\n'.join(samples))
    for k, v in (aux or {}).items():
        if isinstance(v, pd.DataFrame):
            v.to_csv(out / f'{k}.tsv', sep='\t', index=False)
        else:
            np.save(out / f'{k}.npy', v)


def load_hapdose(d):
    """-> genotype_df (g = xL+xR), variant_df(chrom,pos), xL_df, xR_df, variants."""
    d = Path(d)
    vdf = pd.read_csv(d / 'variants.tsv', sep='\t', dtype={'chrom': str, 'site': str,
                                                            'allele': str}, keep_default_na=False)
    samples = [l.strip() for l in open(d / 'samples.txt') if l.strip()]
    XL = np.load(d / 'xL.npy'); XR = np.load(d / 'xR.npy')
    idx = vdf['id'].values
    return (pd.DataFrame(XL + XR, index=idx, columns=samples),
            vdf.set_index('id')[['chrom', 'pos']],
            pd.DataFrame(XL, index=idx, columns=samples),
            pd.DataFrame(XR, index=idx, columns=samples), vdf)


def load_aux(d):
    """-> dict(samples, str_sites, str_len, str_phased, ma_sites, ma_alleles,
    ma_phased) with only the sidecars that exist. Feed to
    hapmixqtl.map_str_curvature / map_multiallelic."""
    d = Path(d)
    aux = dict(samples=[l.strip() for l in open(d / 'samples.txt') if l.strip()])
    for k in ('str_sites', 'ma_sites'):
        if (d / f'{k}.tsv').exists():
            aux[k] = pd.read_csv(d / f'{k}.tsv', sep='\t', dtype={'chrom': str})
    for k in ('str_len', 'str_phased', 'ma_alleles', 'ma_phased'):
        if (d / f'{k}.npy').exists():
            aux[k] = np.load(d / f'{k}.npy')
    return aux


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--str-vcf'); ap.add_argument('--snp-vcf')
    ap.add_argument('--samples'); ap.add_argument('--out', default='hapdose')
    ap.add_argument('--min-q', type=float, default=0.9)
    ap.add_argument('--min-call-rate', type=float, default=0.8)
    ap.add_argument('--period', type=int, default=None,
                    help='override INFO/PERIOD for every locus')
    ap.add_argument('--no-multiallelic', action='store_true',
                    help='ignore multi-ALT rows of --snp-vcf (default: split rows + sidecar)')
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    if not (args.samples and (args.str_vcf or args.snp_vcf)):
        raise SystemExit('--samples and at least one of --str-vcf / --snp-vcf are required '
                         '(or --selftest)')
    samples = [l.strip() for l in open(args.samples) if l.strip()]
    print(f'{len(samples)} samples')
    strs = (parse_str_vcf(args.str_vcf, samples, args.min_q, args.min_call_rate, args.period)
            if args.str_vcf else [])
    ma = (parse_multiallelic_vcf(args.snp_vcf, samples)
          if (args.snp_vcf and not args.no_multiallelic) else [])
    vdf, XL, XR, aux = build_hapdose(samples, strs, args.snp_vcf, ma)
    write_hapdose(args.out, vdf, XL, XR, samples, aux)
    counts = vdf['type'].value_counts().to_dict()
    print(f'wrote {args.out}/: {len(vdf)} variants {counts}; sidecars: '
          f'{[k for k in aux if not k.endswith("phased")] or "none"}')
    print('Run map_cis with maf_threshold=0; the af/ma_samples columns are meaningless '
          'for STR rows. Second pass: load_aux() -> map_str_curvature / map_multiallelic.')


# ---------------------------------------------------------------------------
#  Validation: planted effects, real map_cis + real second-pass functions
# ---------------------------------------------------------------------------

def selftest():
    import contextlib, io, tempfile, warnings
    warnings.filterwarnings('ignore')
    try:
        from tensorqtl.hapmixqtl import map_cis, map_multiallelic, map_str_curvature
    except ImportError:
        sys.path.insert(0, str(HERE.parent / 'tensorqtl'))
        from hapmixqtl import map_cis, map_multiallelic, map_str_curvature
    td = Path(tempfile.mkdtemp()); rng = np.random.RandomState(0)
    N = 200
    samples = [f'S{i:03d}' for i in range(N)]
    (td / 'samples.txt').write_text('\n'.join(samples))
    W = lambda j: 1_000_000 * j          # window j; gene Gj sits at W(j) + 5000

    # ---- STR VCF, HipSTR-style: PERIOD in INFO, GT:GB:Q per sample ----------
    # loci: 0,1 planted linear eSTRs (phased); 2 planted linear but UNPHASED;
    #       3,4 null; 5 low-Q at 15% (imputed, kept); 6 low-Q at 40% (dropped);
    #       10 planted QUADRATIC (phased); 11 planted QUADRATIC, UNPHASED
    period = 3
    str_L = {}
    low_q_frac = {5: 0.15, 6: 0.40}
    unphased = {2, 11}
    lines = ['##fileformat=VCFv4.2',
             '#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t' + '\t'.join(samples)]
    for j in list(range(7)) + [10, 11]:
        pos = W(j) + 5000
        alleles_units = [-2, -1, 0, 1, 2, 3]           # repeat-unit diffs available
        LA = rng.choice(alleles_units, N); LB = rng.choice(alleles_units, N)
        str_L[j] = (LA.astype(float), LB.astype(float))
        ref = 'CAG' * 8
        alts = ['CAG' * (8 + u) for u in alleles_units if u != 0]
        alt_idx = {u: i + 1 for i, u in enumerate([u for u in alleles_units if u != 0])}
        sep = '/' if j in unphased else '|'
        cells = []
        for k in range(N):
            ia = 0 if LA[k] == 0 else alt_idx[LA[k]]
            ib = 0 if LB[k] == 0 else alt_idx[LB[k]]
            q = 0.5 if rng.rand() < low_q_frac.get(j, 0.0) else 0.99
            cells.append(f'{ia}{sep}{ib}:{int(LA[k]*period)}{sep}{int(LB[k]*period)}:{q}')
        lines.append(f'1\t{pos}\tSTR{j}\t{ref}\t{",".join(alts)}\t.\tPASS\t'
                     f'PERIOD={period}\tGT:GB:Q\t' + '\t'.join(cells))
    (td / 'str.vcf').write_text('\n'.join(lines) + '\n')

    # ---- SNP VCF: 5 null SNPs per window, 1 planted eSNP window (6),
    #      and tri-allelic NON-repeat sites in windows 7, 8, 9 -----------------
    lines = ['##fileformat=VCFv4.2',
             '#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t' + '\t'.join(samples)]
    snp_x = {}
    for j in range(12):
        for m in range(5):
            pos = W(j) + 3000 + 300 * m
            xl = (rng.rand(N) < .4).astype(int); xr = (rng.rand(N) < .4).astype(int)
            snp_x[(j, m)] = (xl.astype(float), xr.astype(float))
            lines.append(f'1\t{pos}\tSNP{j}_{m}\tA\tG\t.\tPASS\t.\tGT\t'
                         + '\t'.join(f'{xl[k]}|{xr[k]}' for k in range(N)))
    # multi-ALT sites: allele freqs ref .60, ALT1 .25, ALT2 .15 (+ ALT3 .02 at site 9)
    ma_h = {}
    for j in (7, 8, 9):
        p = [.60, .25, .15] if j != 9 else [.58, .25, .15, .02]
        hA = rng.choice(len(p), N, p=p); hB = rng.choice(len(p), N, p=p)
        ma_h[j] = (hA, hB)
        alts = 'T,C' if j != 9 else 'T,C,G'
        lines.append(f'1\t{W(j) + 4000}\tMA{j}\tA\t{alts}\t.\tPASS\t.\tGT\t'
                     + '\t'.join(f'{hA[k]}|{hB[k]}' for k in range(N)))
    (td / 'snp.vcf').write_text('\n'.join(lines) + '\n')

    print('SELF-TEST: encoding STRs + multi-ALT sites + SNPs into one hapdose\n')
    main(['--str-vcf', str(td / 'str.vcf'), '--snp-vcf', str(td / 'snp.vcf'),
          '--samples', str(td / 'samples.txt'), '--out', str(td / 'hd')])
    g_df, v_df, xl_df, xr_df, vinfo = load_hapdose(td / 'hd')
    aux = load_aux(td / 'hd')
    counts = vinfo['type'].value_counts().to_dict()
    assert counts == {'snp': 60, 'str': 8, 'ma_allele': 7}, counts
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
    # sidecars: raw lengths keep NaN for the low-Q calls, the phased flag is per call
    s5 = aux['str_sites'].index[aux['str_sites']['id'] == 'STR5'][0]
    assert np.isnan(aux['str_len'][s5, :, 0]).mean() > 0.05 and \
        np.allclose(aux['str_len'][s5, ~np.isnan(aux['str_len'][s5, :, 0]), 0],
                    LA5[~np.isnan(aux['str_len'][s5, :, 0])])
    s11 = aux['str_sites'].index[aux['str_sites']['id'] == 'STR11'][0]
    called5 = ~np.isnan(aux['str_len'][s5, :, 0])
    assert not aux['str_phased'][s11].any() and aux['str_phased'][s5][called5].all() \
        and not aux['str_phased'][s5][~called5].any(), 'phase flag must follow the call'
    # multi-ALT split rows: allele-k indicator per haplotype; sidecar keeps the index
    hA, hB = ma_h[7]
    assert np.allclose(xl_df.loc['1_7004000_A_C'].values, hA == 2) and \
        np.allclose(xr_df.loc['1_7004000_A_C'].values, hB == 2)
    assert list(aux['ma_sites']['site_id']) == ['MA7', 'MA8', 'MA9']
    assert np.array_equal(aux['ma_alleles'][0, :, 0], hA) and np.array_equal(aux['ma_alleles'][0, :, 1], hB)
    print('encoding: STR phased -> (L_A, L_B); unphased -> (mean, mean) with exact sum; '
          '15% low-Q imputed / 40% dropped; raw lengths + phase kept in sidecar;\n'
          '          multi-ALT -> one split row per ALT + allele index sidecar')

    # ---- expression with PLANTED effects; run the REAL map_cis -----------------
    genes = [f'G{j}' for j in range(12)]
    beta = {0: 0.35, 1: -0.30, 2: 0.35, 6: 0.60}        # linear STRs, eSNP
    quad = {10: (0.10, 0.12), 11: (0.10, 0.12)}          # (b1, b2): f(L) = b1 L + b2 L^2
    ma_beta = {7: (0.0, 0.5), 8: (0.4, -0.4), 9: (0.0, 0.0, 0.0)}   # per ALT allele
    A = np.zeros((12, N)); T = np.zeros((12, N))
    for j in range(12):
        if j in quad:
            LA, LB = str_L[j]; b1, b2 = quad[j]
            fA = b1 * LA + b2 * LA ** 2; fB = b1 * LB + b2 * LB ** 2
        elif j in ma_beta:
            hA, hB = ma_h[j]; eff = np.array((0.0,) + ma_beta[j])
            fA = eff[hA]; fB = eff[hB]
        elif j < 6:
            LA, LB = str_L[j]; fA = beta.get(j, 0.0) * LA; fB = beta.get(j, 0.0) * LB
        else:
            xl, xr = snp_x[(6, 2)] if j == 6 else (np.zeros(N), np.zeros(N))
            fA = beta.get(j, 0.0) * xl; fB = beta.get(j, 0.0) * xr
        A[j] = fA - fB + rng.normal(0, 0.35, N)
        T[j] = 2.0 + (fA + fB) / 2 + rng.normal(0, 0.35, N)
    Va = np.full((12, N), 0.02); Vt = np.full((12, N), 0.02)
    mk = lambda M: pd.DataFrame(M, index=genes, columns=samples)
    pos_df = pd.DataFrame({'chr': ['1'] * 12, 'pos': [W(j) + 5000 for j in range(12)]},
                          index=genes)
    with contextlib.redirect_stdout(io.StringIO()):
        res = map_cis(g_df, v_df, mk(A), mk(T), mk(Va), mk(Vt), pos_df,
                      xL_df=xl_df, xR_df=xr_df, window=20000, nperm=200,
                      verbose=False)
    print('\nmap_cis on the mixed window (lead variant per gene):')
    print(f"  {'gene':5s} {'lead':16s} {'slope':>8s} {'planted':>8s} {'pval_beta':>10s}")
    for j, g in enumerate(genes):
        r = res.loc[g]
        planted = beta.get(j, quad.get(j, ma_beta.get(j, 0)))
        print(f"  {g:5s} {r['variant_id']:16s} {r['slope']:8.3f} {str(planted):>8s} "
              f"{r['pval_beta']:10.2e}")
    for j in (0, 1):
        r = res.loc[f'G{j}']
        assert r['variant_id'] == f'STR{j}', f'G{j} lead should be the STR, got {r["variant_id"]}'
        assert abs(r['slope'] - beta[j]) < 0.08, f'G{j} beta per repeat unit off: {r["slope"]}'
        assert r['pval_beta'] < 1e-3
    r = res.loc['G2']          # unphased: total channel alone must still carry it
    assert r['variant_id'] == 'STR2' and r['pval_beta'] < 1e-2 and abs(r['slope'] - 0.35) < 0.12
    r = res.loc['G6']          # SNP path unchanged
    assert r['variant_id'] == 'SNP6_2' and abs(r['slope'] - 0.60) < 0.1
    for j in (3, 4, 5, 9):     # nulls stay null
        assert res.loc[f'G{j}', 'pval_beta'] > 0.05, f'null G{j} called'
    # split rows compete for the lead: the ALT2-only site is led by its ALT2 row
    assert res.loc['G7', 'variant_id'] == '1_7004000_A_C', res.loc['G7', 'variant_id']
    assert res.loc['G8', 'variant_id'].startswith('1_8004000_A_') and res.loc['G8', 'pval_beta'] < 1e-3
    assert res.loc['G10', 'variant_id'] == 'STR10' and res.loc['G10', 'pval_beta'] < 1e-3
    print('checks: STR leads recovered with beta per repeat unit; unphased STR via total '
          'channel; SNP unchanged; nulls null; multi-ALT split rows can lead')

    # ---- second pass 1: categorical model on the multi-ALT sites ---------------
    with contextlib.redirect_stdout(io.StringIO()):
        site_res, allele_res = map_multiallelic(
            aux['ma_alleles'], aux['ma_sites'].set_index('site_id'), aux['samples'],
            mk(A), mk(T), mk(Va), mk(Vt), pos_df, hap_phased=aux['ma_phased'],
            window=20000, min_hap=10, verbose=False)
    site_res = site_res.set_index(['phenotype_id', 'site_id'])
    allele_res = allele_res.set_index(['phenotype_id', 'site_id', 'allele'])
    print('\nmap_multiallelic (joint fit of the split rows, log aFC vs reference allele):')
    print(f"  {'gene':5s} {'site':5s} {'allele':7s} {'n_hap':>5s} {'slope':>7s} {'se':>6s} "
          f"{'planted':>8s} {'pval':>9s} | {'joint p':>9s} {'df':>2s}")
    for (g, s), row in site_res.iterrows():
        j = int(g[1:])
        for al, ar in allele_res.loc[(g, s)].iterrows():
            planted = ma_beta[j][int(al) - 1] if al != 'other' else 0
            print(f"  {g:5s} {s:5s} {al:7s} {int(ar['n_hap']):5d} {ar['slope']:7.3f} {ar['slope_se']:6.3f} "
                  f"{planted:8.2f} {ar['pval']:9.2e} | {row['pval_joint']:9.2e} {int(row['n_tested']):2d}")
    # G7: ALT2 carries the effect, ALT1 does not; joint test fires
    assert abs(allele_res.loc[('G7', 'MA7', '2'), 'slope'] - 0.5) < 0.1
    assert abs(allele_res.loc[('G7', 'MA7', '1'), 'slope']) < 0.1
    assert site_res.loc[('G7', 'MA7'), 'pval_joint'] < 1e-6 and site_res.loc[('G7', 'MA7'), 'n_tested'] == 2
    # G8: opposite effects on the two ALTs; the joint fit recovers both against the
    # clean reference, where the marginal split-row fit lumps the other ALT into "not k"
    assert abs(allele_res.loc[('G8', 'MA8', '1'), 'slope'] - 0.4) < 0.1
    assert abs(allele_res.loc[('G8', 'MA8', '2'), 'slope'] + 0.4) < 0.1
    assert site_res.loc[('G8', 'MA8'), 'pval_joint'] < 1e-6
    lead8 = res.loc['G8']
    print(f"  (G8 lead-scan marginal slope for {lead8['variant_id']}: {lead8['slope']:.3f}; "
          f"joint per-allele slopes above are the clean contrasts)")
    # G9: null; the 2% ALT3 (below min_hap, pool too small) is treated as missing
    assert site_res.loc[('G9', 'MA9'), 'pval_joint'] > 0.01
    assert site_res.loc[('G9', 'MA9'), 'n_tested'] == 2 and site_res.loc[('G9', 'MA9'), 'n_missing_hap'] > 0
    print('checks: per-allele log aFC recovered vs a clean reference; ALT-specific effect '
          'isolated; joint K-1 df test fires; null null; rare allele handled by min_hap')

    # ---- second pass 2: linear + curvature on the STRs --------------------------
    with contextlib.redirect_stdout(io.StringIO()):
        cur = map_str_curvature(
            aux['str_len'], aux['str_phased'], aux['str_sites'].set_index('id'), aux['samples'],
            mk(A), mk(T), mk(Va), mk(Vt), pos_df, window=20000, verbose=False)
    cur = cur.set_index(['phenotype_id', 'str_id'])
    print('\nmap_str_curvature (per-haplotype f(L) = b1 L + b2 L^2):')
    print(f"  {'gene':5s} {'str':6s} {'slope_lin':>9s} {'b2':>7s} {'se':>6s} {'planted b2':>10s} "
          f"{'p_curv':>9s} {'p_joint2':>9s} {'phased':>6s}")
    for (g, s), row in cur.iterrows():
        j = int(s[3:])
        print(f"  {g:5s} {s:6s} {row['slope_lin']:9.3f} {row['slope_sq']:7.3f} {row['slope_sq_se']:6.3f} "
              f"{quad.get(j, (0, 0))[1]:10.2f} {row['pval_curv']:9.2e} {row['pval_joint2']:9.2e} "
              f"{int(row['n_phased']):6d}")
    # the linear-only fit of the second pass must reproduce the lead scan's slope
    for j in (0, 1, 2):
        assert abs(cur.loc[(f'G{j}', f'STR{j}'), 'slope_lin'] - res.loc[f'G{j}', 'slope']) < 2e-3, \
            f'second-pass linear slope must equal the lead-scan slope for STR{j}'
    # curvature recovered where planted (phased: both channels; unphased: total only)
    assert abs(cur.loc[('G10', 'STR10'), 'slope_sq'] - 0.12) < 0.05 and cur.loc[('G10', 'STR10'), 'pval_curv'] < 1e-4
    assert abs(cur.loc[('G11', 'STR11'), 'slope_sq'] - 0.12) < 0.08 and cur.loc[('G11', 'STR11'), 'pval_curv'] < 1e-2
    assert np.isnan(cur.loc[('G11', 'STR11'), 'slope_sq_a']), 'unphased STR has no ASE-channel curvature fit'
    # no curvature where the planted effect is linear
    for j in (0, 1, 2):
        assert abs(cur.loc[(f'G{j}', f'STR{j}'), 'slope_sq']) < 0.06 and \
            cur.loc[(f'G{j}', f'STR{j}'), 'pval_curv'] > 0.01, f'spurious curvature at STR{j}'
    print('checks: linear slope identical to the lead scan; b2 recovered (phased and unphased); '
          'linear eSTRs show no curvature')
    print('\nSELF-TEST OK')
    return 0


if __name__ == '__main__':
    main()

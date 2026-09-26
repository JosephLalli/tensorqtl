"""Quantification-layer calibration targets for the read-level simulator.

The simulator generates reads from TRUE haplotypes, quantifies them against an
index built from OBSERVED haplotypes, and emits a Salmon-like point estimate
and Gibbs-like draws that are then summarized with
``summaries_from_point_estimates``. This script measures, from data already on
disk, what that emulated quantification layer has to reproduce.

Parts (each writes its own outputs; ``--part all`` runs them in order):

  names     every donor's Salmon transcript list (aux_info/bootstrap/
            names.tsv.gz) -> per (gene, donor): number of haplotype-PAIRED
            transcripts (both _L and _R rows present) and of unpaired rows.
            This is the exact structural indicator "no transcript of the gene
            has two distinct haplotype copies", not inferred from zero counts.
  cache     all 34,457 cache genes x 92 donors, by haplotype-informative read
            band (point estimate pL + pR): exact-zero haplotype share; Gibbs
            variance and counting term for both channels, separately; Fano
            factor (across-draw variance / mean) of the draw totals; point
            estimate against draw mean, including the draw mean of a
            haplotype the point estimate puts at zero.
  eq        Salmon equivalence classes (aux_info/eq_classes.txt.gz) for three
            seeded donors: per gene, reads compatible only with L copies
            (d_L), only with R copies (d_R), with both (n_amb), and with an
            unpaired (homozygous) transcript of the gene. Checks the
            maximum-likelihood zero rule and a Beta-posterior prediction of the
            Gibbs variance of the allelic log ratio against the observed one.
  vcf       heterozygous exonic variants per (gene, donor) on 200 seeded
            calibration genes, from prepped/rephased.vcf.gz and from the VCF
            the personalized transcriptomes were built from, against the
            structural pairing indicator from ``names``.

Randomness: one master SEED = 42; child streams from SeedSequence(42).spawn.
Values are Salmon point estimates; Gibbs draws are used for variance only, in
the transforms of summaries_from_point_estimates (log2, edgeR effective library
size). Reads the draw arrays memory-mapped, in 1,000-gene chunks.
"""
import argparse
import gzip
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import polygamma
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts'))
from tensorqtl.hapmixqtl import summaries_from_point_estimates  # noqa: E402
from run_hapmixqtl_from_salmon import (read_salmon_names, pair_haplotypes)  # noqa: E402

SEED = 42
D = Path('/mnt/ssd/lalli/brainvar_hapmix_deploy')
CACHE = D / 'cache' / 'gibbs_56b63c3b37ed5df8'
PE = CACHE / 'point_estimates'
MANIFEST = D / 'cohort' / 'salmon.tsv'
TX2GENE = D / 'annot' / 'tx2gene.tsv'
GENES_TSV = D / 'annot' / 'genes.tsv'
GENES_NC = D / 'annot' / 'genes.NC.tsv'
EXONS = D / 'annot' / 'exons.tsv'
VCF_REPHASED = D / 'prepped' / 'rephased.vcf.gz'
VCF_BUILD = Path('/mnt/ssd/lalli/nf_stage/brainvar2/'
                 'gatk_t2t_haplotypecaller.joint_called.phased.all_variants.multiallelic.all.nostar.bcf')
OUT = D / 'simulator_calibration_20260926' / 'layer4'
KAPPA, LN2 = 0.5, float(np.log(2.0))
SUFFIXES = ('_L', '_R')
BANDS = (('1-9', 0.0, 10.0), ('10-99', 10.0, 100.0), ('100-999', 100.0, 1000.0),
         ('1000+', 1000.0, np.inf))
QS = (0.10, 0.25, 0.50, 0.75, 0.90)
CHUNK = 1000


def children(n):
    return [np.random.default_rng(s) for s in np.random.SeedSequence(SEED).spawn(n)]


# child streams, fixed assignment: 0 = eq-class donors, 1 = eq-class gene subsets,
# 2 = eq-class emulator simulation, 3 = VCF gene sample, 4 = emulator noise floor
# (SeedSequence.spawn keys are positional, so adding stream 4 leaves 0-3 unchanged)
RNG = children(5)


def qdict(x, qs=QS):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {f'q{int(q*100):02d}': None for q in qs} | {'n': 0}
    return {f'q{int(q*100):02d}': float(np.quantile(x, q)) for q in qs} | {'n': int(x.size)}


def band_masks(n):
    return {lab: (n > lo if lo == 0 else n >= lo) & (n < hi) for lab, lo, hi in BANDS}


def load_basics():
    genes = (CACHE / 'genes.txt').read_text().split()
    samples = (CACHE / 'samples.txt').read_text().split()
    cal = set((PE / 'edger' / 'calibration_genes.txt').read_text().split())
    ed = pd.read_csv(PE / 'edger' / 'edger_samples.tsv', sep='\t', index_col=0)
    eff = ed.loc[samples, 'eff_lib_size'].to_numpy(float)
    man = dict(l.split('\t')[:2] for l in MANIFEST.read_text().strip().split('\n'))
    gchr = pd.read_csv(GENES_TSV, sep='\t', header=None, index_col=0)[1]
    gchr = gchr[~gchr.index.duplicated()]
    return genes, samples, cal, eff, man, gchr


def tx2gene():
    return dict(l.split('\t')[:2] for l in TX2GENE.read_text().strip().split('\n') if '\t' in l)


# ---------------------------------------------------------------------------
#  names: structural pairing indicator for every donor
# ---------------------------------------------------------------------------

def part_names():
    genes, samples, cal, eff, man, gchr = load_basics()
    t2g = tx2gene()
    gi = {g: i for i, g in enumerate(genes)}
    npair = np.zeros((len(genes), len(samples)), np.int16)
    nunp = np.zeros((len(genes), len(samples)), np.int16)
    n_annot = defaultdict(int)
    for g in t2g.values():
        n_annot[g] += 1
    for si, s in enumerate(samples):
        names = read_salmon_names(man[s])
        pairs = pair_haplotypes(names, SUFFIXES)
        paired_rows = set()
        for base, (ia, ib) in pairs.items():
            paired_rows.update((ia, ib))
            g = t2g.get(base)
            if g in gi:
                npair[gi[g], si] += 1
        for i, nm in enumerate(names):
            if i in paired_rows:
                continue
            base = nm[:-2] if nm.endswith(SUFFIXES) else nm
            g = t2g.get(base)
            if g in gi:
                nunp[gi[g], si] += 1
        print(f'  names [{si+1}/{len(samples)}] {s}: {len(pairs)} pairs', flush=True)
    nann = np.array([n_annot.get(g, 0) for g in genes], np.int16)
    np.savez(OUT / 'names_pairing.npz', npair=npair, nunp=nunp, n_annot_tx=nann,
             genes=np.array(genes), samples=np.array(samples))

    pL = np.load(PE / 'pL.npy'); pR = np.load(PE / 'pR.npy'); pT = np.load(PE / 'pT.npy')
    iscal = np.array([g in cal for g in genes])
    chr_ = np.array([gchr.get(g, 'NA') for g in genes])
    auto = np.isin(chr_, [f'chr{i}' for i in range(1, 23)])
    sets = {'all_cache_genes': np.ones(len(genes), bool), 'autosomal': auto,
            'chrX': chr_ == 'chrX', 'chrY': chr_ == 'chrY', 'calibration': iscal}
    res = {}
    for lab, gm in sets.items():
        nopair = (npair[gm] == 0)
        noinf = (pL[gm] + pR[gm]) <= 0
        tot = nopair.size
        res[lab] = dict(
            genes=int(gm.sum()), pairs=int(tot),
            no_paired_transcript=int(nopair.sum()), share_no_paired=float(nopair.mean()),
            pL_plus_pR_zero=int(noinf.sum()), share_pLpR_zero=float(noinf.mean()),
            paired_but_pLpR_zero=int((~nopair & noinf).sum()),
            nopair_but_pLpR_pos=int((nopair & ~noinf).sum()),
            no_paired_and_expressed_pT_ge10=int((nopair & (pT[gm] >= 10)).sum()),
            no_paired_and_pT_zero=int((nopair & (pT[gm] <= 0)).sum()),
        )
    # dependence on the number of annotated transcripts (calibration genes)
    by_ntx = {}
    for lab, lo, hi in (('1', 1, 2), ('2-3', 2, 4), ('4-9', 4, 10), ('10+', 10, 10 ** 6)):
        gm = iscal & (nann >= lo) & (nann < hi)
        by_ntx[lab] = dict(genes=int(gm.sum()), share_no_paired=float((npair[gm] == 0).mean()))
    res['calibration_by_annotated_transcripts'] = by_ntx
    (OUT / 'names_pairing_summary.json').write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return res


# ---------------------------------------------------------------------------
#  cache: per-pair targets over all cache genes
# ---------------------------------------------------------------------------

def part_cache():
    genes, samples, cal, eff, man, gchr = load_basics()
    names = np.load(OUT / 'names_pairing.npz')
    npair, nann = names['npair'], names['n_annot_tx']
    pL = np.load(PE / 'pL.npy'); pR = np.load(PE / 'pR.npy'); pT = np.load(PE / 'pT.npy')
    YL = np.load(CACHE / 'YL.npy', mmap_mode='r')
    YR = np.load(CACHE / 'YR.npy', mmap_mode='r')
    YT = np.load(CACHE / 'YT.npy', mmap_mode='r')
    G, S = pL.shape
    k = 1e6 / eff
    fields = ('Va_g', 'Vt_g', 'mL', 'mR', 'mT', 'vLR', 'vT', 'vL', 'vR', 'a_dmean',
              'fzL', 'fzR', 'fzT', 'cov_at')
    arr = {f: np.zeros((G, S), np.float32) for f in fields}
    check = None
    t0 = time.time()
    for s0 in range(0, G, CHUNK):
        sl = slice(s0, min(G, s0 + CHUNK))
        yL = np.asarray(YL[sl]); yR = np.asarray(YR[sl]); yT = np.asarray(YT[sl])
        a_d = np.log2((yL + KAPPA) / (yR + KAPPA))
        t_d = np.log2(yT * k[None, :, None] + 1.0)
        arr['Va_g'][sl] = a_d.var(2)
        arr['Vt_g'][sl] = t_d.var(2)
        arr['a_dmean'][sl] = a_d.mean(2)
        arr['cov_at'][sl] = ((a_d - a_d.mean(2, keepdims=True)) *
                             (t_d - t_d.mean(2, keepdims=True))).mean(2)
        arr['mL'][sl] = yL.mean(2); arr['mR'][sl] = yR.mean(2); arr['mT'][sl] = yT.mean(2)
        arr['vLR'][sl] = (yL + yR).var(2); arr['vT'][sl] = yT.var(2)
        arr['vL'][sl] = yL.var(2); arr['vR'][sl] = yR.var(2)
        arr['fzL'][sl] = (yL == 0).mean(2); arr['fzR'][sl] = (yR == 0).mean(2)
        arr['fzT'][sl] = (yT == 0).mean(2)
        if check is None:   # identity with the shipped summary function
            A, T, Va, Vt, Cat = summaries_from_point_estimates(pL[sl], pR[sl], pT[sl], eff, yL, yR, yT)
            qa = (1 / (pL[sl] + KAPPA) + 1 / (pR[sl] + KAPPA)) / LN2 ** 2
            y = pT[sl] + 0.5
            qt = k[None] ** 2 * y / ((k[None] * y + 1) ** 2 * LN2 ** 2)
            noc = (pL[sl] + pR[sl]) <= 0
            check = dict(
                max_abs_Va=float(np.abs(np.where(noc, 0, a_d.var(2) + qa) - Va).max()),
                max_abs_Vt=float(np.abs(t_d.var(2) + qt - Vt).max()))
            print('  identity with summaries_from_point_estimates:', check, flush=True)
        print(f'  cache genes {sl.stop}/{G}  {time.time()-t0:.0f}s', flush=True)
    np.savez(OUT / 'cache_pair_arrays.npz', **arr)

    iscal = np.array([g in cal for g in genes])
    q_a = (1 / (pL + KAPPA) + 1 / (pR + KAPPA)) / LN2 ** 2
    y = pT + 0.5
    q_t = k[None] ** 2 * y / ((k[None] * y + 1) ** 2 * LN2 ** 2)
    n = pL + pR
    inf = n > 0
    a_pt = np.log2((pL + KAPPA) / (pR + KAPPA))
    zL = inf & (pL == 0); zR = inf & (pR == 0)
    zone = zL ^ zR
    near = inf & (np.minimum(pL, pR) > 0) & (np.minimum(pL, pR) < 0.5)
    Va_ship = np.where(inf, arr['Va_g'] + q_a, 0.0)
    Vt_ship = arr['Vt_g'] + q_t
    mLR = arr['mL'] + arr['mR']
    fanoLR = np.where(mLR > 0, arr['vLR'] / np.maximum(mLR, 1e-300), np.nan)
    fanoT = np.where(arr['mT'] > 0, arr['vT'] / np.maximum(arr['mT'], 1e-300), np.nan)
    single_tx = (nann == 1)[:, None] & np.ones_like(inf)
    kpair = npair.astype(float)
    # draw mean of the zeroed haplotype, and its excess over the point estimate
    mz = np.where(zL, arr['mL'], np.where(zR, arr['mR'], np.nan))
    mz_share = np.where(zone, mz / np.maximum(mLR, 1e-300), np.nan)
    out = {'identity_check': check, 'n_genes': G, 'n_samples': S}
    for uni, gm in (('calibration', iscal), ('all_cache_genes', np.ones(G, bool))):
        gm2 = gm[:, None] & np.ones((1, S), bool)
        u = dict(pairs=int(gm2.sum()), informative=int((gm2 & inf).sum()),
                 point_zero_both_draws_positive=int((gm2 & ~inf & (mLR > 0)).sum()),
                 point_zero_both_draws_zero=int((gm2 & ~inf & (mLR <= 0)).sum()),
                 draws_of_point_zero_both=qdict(mLR[gm2 & ~inf & (mLR > 0)]),
                 bands={})
        bm = band_masks(n)
        for lab, m0 in bm.items():
            m = gm2 & inf & m0
            nz = m & ~zone & (np.minimum(pL, pR) > 0)
            b = dict(
                pairs=int(m.sum()),
                pairs_below_1_read=int((m & (n < 1)).sum()),
                one_haplotype_exact_zero=int((m & zone).sum()),
                share_one_haplotype_exact_zero=float((m & zone).sum() / max(m.sum(), 1)),
                one_haplotype_in_0_to_0p5=int((m & near).sum()),
                Va_shipped=qdict(Va_ship[m]), Va_gibbs=qdict(arr['Va_g'][m]), q_a=qdict(q_a[m]),
                Va_gibbs_over_q_a=qdict((arr['Va_g'] / q_a)[m]),
                Va_gibbs_over_q_a_both_haplotypes_positive=qdict((arr['Va_g'] / q_a)[nz]),
                Va_gibbs_over_q_a_one_zero=qdict((arr['Va_g'] / q_a)[m & zone]),
                Vt_shipped=qdict(Vt_ship[m]), Vt_gibbs=qdict(arr['Vt_g'][m]), q_t=qdict(q_t[m]),
                Vt_gibbs_over_q_t=qdict((arr['Vt_g'] / q_t)[m]),
                fano_draw_total_LR=qdict(fanoLR[m]),
                fano_draw_total_T=qdict(fanoT[m]),
                fano_draw_total_T_single_transcript_genes=qdict(fanoT[m & single_tx]),
                point_LR_over_draw_mean_LR=qdict((n / mLR)[m & (mLR > 0)]),
                point_T_over_draw_mean_T=qdict((pT / arr['mT'])[m & (arr['mT'] > 0)]),
                draw_mean_minus_point_LR=qdict((mLR - n)[m]),
                draw_mean_minus_point_LR_per_paired_transcript=qdict(((mLR - n) / np.maximum(kpair, 1))[m & (kpair > 0)]),
                a_point_minus_draw_mean_a_both_positive=qdict((a_pt - arr['a_dmean'])[nz]),
                a_point_minus_draw_mean_a_one_zero=qdict(np.abs(a_pt - arr['a_dmean'])[m & zone]),
                fracL_point_minus_draw_both_positive=qdict((pL / n - arr['mL'] / np.maximum(mLR, 1e-300))[nz]),
                zeroed_haplotype_draw_mean=qdict(mz[m & zone]),
                zeroed_haplotype_draw_mean_per_paired_transcript=qdict((mz / np.maximum(kpair, 1))[m & zone & (kpair > 0)]),
                zeroed_haplotype_draw_share_of_LR=qdict(mz_share[m & zone]),
                zeroed_haplotype_share_of_draws_exactly_zero=qdict(np.where(zL, arr['fzL'], arr['fzR'])[m & zone]),
                paired_transcripts=qdict(kpair[m]),
            )
            u['bands'][lab] = b
        # total channel by point-estimate total band
        ub = {}
        for lab, lo, hi in (('0', -1, 1e-12), ('(0,10)', 1e-12, 10), ('10-99', 10, 100),
                            ('100-999', 100, 1000), ('1000+', 1000, np.inf)):
            m = gm2 & (pT >= lo) & (pT < hi) if lab != '0' else gm2 & (pT <= 0)
            ub[lab] = dict(pairs=int(m.sum()), Vt_gibbs=qdict(arr['Vt_g'][m]), q_t=qdict(q_t[m]),
                           Vt_gibbs_over_q_t=qdict((arr['Vt_g'] / q_t)[m]),
                           fano_T=qdict(fanoT[m]), fano_T_single_transcript=qdict(fanoT[m & single_tx]),
                           point_T_over_draw_mean_T=qdict((pT / arr['mT'])[m & (arr['mT'] > 0)]),
                           draw_mean_minus_point_T=qdict((arr['mT'] - pT)[m]),
                           share_draws_T_exactly_zero=qdict(arr['fzT'][m]))
        u['total_channel_by_pT_band'] = ub
        out[uni] = u
    (OUT / 'cache_targets.json').write_text(json.dumps(out, indent=1))
    print(json.dumps(out['calibration']['bands'], indent=1)[:6000])
    return out


# ---------------------------------------------------------------------------
#  eq: equivalence-class decomposition for three seeded donors
# ---------------------------------------------------------------------------

CATS = ('L', 'R', 'LR', 'LU', 'RU', 'U')   # by the gene's own transcripts in the class


def parse_eq(sdir, gi, t2g):
    """Per cache gene: read counts by category, gene-unique vs shared with
    another gene, and the number of active (in >= 1 class) paired L / R copies."""
    with gzip.open(Path(sdir) / 'aux_info' / 'eq_classes.txt.gz', 'rt') as fh:
        ntx = int(fh.readline()); neq = int(fh.readline())
        names = [fh.readline().rstrip('\n') for _ in range(ntx)]
        lines = fh.read().split('\n')
    pairs = pair_haplotypes(names, SUFFIXES)
    ttype = np.full(ntx, 2, np.int8)          # 0 paired L, 1 paired R, 2 unpaired
    for base, (ia, ib) in pairs.items():
        ttype[ia] = 0; ttype[ib] = 1
    tgene = np.full(ntx, -1, np.int64)        # -1: no cache gene
    for i, nm in enumerate(names):
        base = nm[:-2] if nm.endswith(SUFFIXES) else nm
        g = t2g.get(base)
        if g is not None and g in gi:
            tgene[i] = gi[g]
    G = len(gi)
    cnt = np.zeros((G, len(CATS), 2), np.float64)   # [..., 0] gene-unique, [..., 1] shared
    active = np.zeros(ntx, bool)
    n_lines = 0
    for line in lines:
        if not line:
            continue
        f = line.split('\t')
        kk = int(f[0])
        if len(f) != kk + 2:
            raise SystemExit(f'unexpected eq-class line width {len(f)} for k={kk} (weights present?)')
        tx = np.fromiter((int(x) for x in f[1:1 + kk]), np.int64, kk)
        c = float(f[1 + kk])
        n_lines += 1
        active[tx] = True
        gs = tgene[tx]
        ug = np.unique(gs)
        shared = int(len(ug) > 1 or (ug[0] == -1 if len(ug) else False))
        for g in ug:
            if g < 0:
                continue
            ty = ttype[tx[gs == g]]
            hL, hR, hU = (ty == 0).any(), (ty == 1).any(), (ty == 2).any()
            if hL and hR:
                cat = 2
            elif hL:
                cat = 3 if hU else 0
            elif hR:
                cat = 4 if hU else 1
            else:
                cat = 5
            cnt[g, cat, shared] += c
    if n_lines != neq:
        raise SystemExit(f'eq-class count mismatch: header {neq}, parsed {n_lines}')
    kL = np.zeros(G, np.int32); kR = np.zeros(G, np.int32)
    for i in np.flatnonzero(active & (tgene >= 0) & (ttype < 2)):
        (kL if ttype[i] == 0 else kR)[tgene[i]] += 1
    return cnt, kL, kR, dict(n_transcripts=ntx, n_eq_classes=neq, n_pairs=len(pairs),
                             reads_in_classes=float(sum(float(l.rsplit('\t', 1)[1]) for l in lines if l)))


def emulate_va(dL, dR, namb, kL, kR, rng, ndraw=2000, prior=1.0):
    """Gibbs variance of log2((yL+k)/(yR+k)) under the proposed emulator:
    share p ~ Beta(dL + prior*kL, dR + prior*kR); paired total S ~ Gamma(N +
    prior*(kL+kR)), N = dL + dR + namb; yL = p S, yR = (1 - p) S."""
    aL = dL + prior * np.maximum(kL, 1); aR = dR + prior * np.maximum(kR, 1)
    p = rng.beta(aL[:, None], aR[:, None], size=(len(dL), ndraw))
    S = rng.gamma((namb + aL + aR)[:, None], 1.0, size=(len(dL), ndraw))   # N + prior*(kL+kR)
    yL, yR = p * S, (1 - p) * S
    return np.log2((yL + KAPPA) / (yR + KAPPA)).var(1)


def part_eq(n_donors=3, n_genes=50):
    genes, samples, cal, eff, man, gchr = load_basics()
    t2g = tx2gene()
    gi = {g: i for i, g in enumerate(genes)}
    pL = np.load(PE / 'pL.npy'); pR = np.load(PE / 'pR.npy'); pT = np.load(PE / 'pT.npy')
    arrs = np.load(OUT / 'cache_pair_arrays.npz')
    Va_g, mL, mR, vLR = arrs['Va_g'], arrs['mL'], arrs['mR'], arrs['vLR']
    donors = sorted(RNG[0].choice(len(samples), n_donors, replace=False).tolist())
    iscal = np.array([g in cal for g in genes])
    rows = []
    meta = {}
    for si in donors:
        s = samples[si]
        t0 = time.time()
        cnt, kL, kR, info = parse_eq(man[s], gi, t2g)
        info['seconds'] = round(time.time() - t0, 1)
        meta[s] = info
        print(f'  eq {s}: {info}', flush=True)
        n = pL[:, si] + pR[:, si]
        df = pd.DataFrame({'gene': genes, 'donor': s, 'calibration': iscal,
                           'pL': pL[:, si], 'pR': pR[:, si], 'pT': pT[:, si],
                           'mL': mL[:, si], 'mR': mR[:, si], 'vLR': vLR[:, si],
                           'Va_gibbs': Va_g[:, si], 'kL': kL, 'kR': kR})
        for ci, c in enumerate(CATS):
            df[f'{c}_uniq'] = cnt[:, ci, 0]
            df[f'{c}_shared'] = cnt[:, ci, 1]
        df['n_pt'] = n
        rows.append(df)
    E = pd.concat(rows, ignore_index=True)
    # d_L, d_R, n_amb: gene-unique classes (strict) and all classes (inclusive)
    for tag, cols in (('strict', ('_uniq',)), ('incl', ('_uniq', '_shared'))):
        for c in CATS:
            E[f'{c}_{tag}'] = sum(E[f'{c}{x}'] for x in cols)
    E['band'] = pd.cut(E['n_pt'], [0, 10, 100, 1000, np.inf], right=False,
                       labels=[b[0] for b in BANDS]).astype(object)
    E.loc[(E['n_pt'] > 0) & (E['n_pt'] < 1), 'band'] = '1-9'
    inf = E['n_pt'] > 0
    # seeded subset: per donor, calibration genes informative in that donor,
    # stratified evenly over the four bands (12/13/12/13)
    sub_idx = []
    per_band = [12, 13, 12, 13]
    for s in [samples[i] for i in donors]:
        for (lab, _, _), k in zip(BANDS, per_band):
            cand = E.index[(E['donor'] == s) & inf & E['calibration'] & (E['band'] == lab)].to_numpy()
            pick = RNG[1].choice(cand, min(k, len(cand)), replace=False)
            sub_idx.extend(sorted(pick.tolist()))
    E['subset50'] = False
    E.loc[sub_idx, 'subset50'] = True
    E.to_csv(OUT / 'eq_class_decomposition.tsv.gz', sep='\t', index=False, compression='gzip')
    return summarize_eq(E, [samples[i] for i in donors], meta)


def part_eqsum():
    """Re-run the equivalence-class summary from the saved decomposition."""
    E = pd.read_csv(OUT / 'eq_class_decomposition.tsv.gz', sep='\t')
    prev = json.loads((OUT / 'eq_class_summary.json').read_text())
    return summarize_eq(E, prev['donors'], prev['parse'])


def summarize_eq(E, donor_names, meta):
    inf = E['n_pt'] > 0
    res = {'donors': donor_names, 'parse': meta}
    for scope, m0 in (('subset50x3', E['subset50']), ('calibration_all_informative', inf & E['calibration']),
                      ('all_cache_informative', inf)):
        X = E[m0].copy()
        r = {'pairs': int(len(X))}
        for tag in ('strict', 'incl'):
            dL, dR, na = X[f'L_{tag}'], X[f'R_{tag}'], X[f'LR_{tag}']
            zr = {}
            for side, d_own, d_oth, p_own in (('L', dL, dR, X['pL']), ('R', dR, dL, X['pR'])):
                pred = (d_own == 0) & (d_oth > 0)
                obs = p_own == 0
                zr[side] = dict(pred_zero_obs_zero=int((pred & obs).sum()),
                                pred_zero_obs_pos=int((pred & ~obs).sum()),
                                pred_pos_obs_zero=int((~pred & obs).sum()),
                                pred_pos_obs_pos=int((~pred & ~obs).sum()),
                                obs_zero_with_own_distinguishable_reads=qdict(d_own[obs & (d_own > 0)]),
                                obs_zero_own_over_other_distinguishable=qdict((d_own / d_oth.clip(lower=1e-9))[obs & (d_own > 0)]))
            tie = (dL == 0) & (dR == 0)
            zr['tie_no_distinguishable'] = dict(
                pairs=int(tie.sum()), with_ambiguous=int((tie & (na > 0)).sum()),
                point_both_positive=int((tie & (X['pL'] > 0) & (X['pR'] > 0)).sum()),
                point_one_zero=int((tie & ((X['pL'] == 0) ^ (X['pR'] == 0))).sum()),
                point_L_zero=int((tie & (X['pL'] == 0) & (X['pR'] > 0)).sum()),
                point_R_zero=int((tie & (X['pR'] == 0) & (X['pL'] > 0)).sum()),
                fracL_point_when_both_positive=qdict((X['pL'] / X['n_pt'])[tie & (X['pL'] > 0) & (X['pR'] > 0)]),
                by_band={lab: dict(pairs_in_band=int((X['band'] == lab).sum()),
                                   ties=int((tie & (X['band'] == lab)).sum()),
                                   ties_point_one_zero=int((tie & (X['band'] == lab) &
                                                            ((X['pL'] == 0) ^ (X['pR'] == 0))).sum()),
                                   one_side_distinguishable_only=int((((dL == 0) ^ (dR == 0)) &
                                                                      (X['band'] == lab)).sum()))
                         for lab, _, _ in BANDS})
            # exceptions to the zero rule: no own distinguishable reads, other side
            # has some, yet the point estimate is positive
            exc = []
            for d_own, d_oth, p_own, lu in ((dL, dR, X['pL'], X[f'LU_{tag}']), (dR, dL, X['pR'], X[f'RU_{tag}'])):
                m = (d_own == 0) & (d_oth > 0) & (p_own > 0)
                exc.append(pd.DataFrame({'share': (p_own / X['n_pt'])[m], 'own_hom_reads': lu[m],
                                         'other_d': d_oth[m], 'n_pt': X['n_pt'][m], 'band': X['band'][m]}))
            exc = pd.concat(exc)
            zr['zero_rule_exceptions'] = dict(
                pairs=int(len(exc)), point_share_of_haplotype=qdict(exc['share']),
                with_own_haplotype_plus_homozygous_class_reads=int((exc['own_hom_reads'] > 0).sum()),
                other_side_distinguishable=qdict(exc['other_d']),
                by_band={lab: int((exc['band'] == lab).sum()) for lab, _, _ in BANDS})
            pz = (dL == 0) & (dR > 0) | (dR == 0) & (dL > 0)
            zr['one_side_distinguishable_only_point_zero_by_band'] = {
                lab: dict(pairs=int((pz & (X['band'] == lab)).sum()),
                          point_zero_on_that_side=int((((dL == 0) & (dR > 0) & (X['pL'] == 0)) |
                                                       ((dR == 0) & (dL > 0) & (X['pR'] == 0)))[X['band'] == lab].sum()))
                for lab, _, _ in BANDS}
            d = dL + dR
            phi = d / (d + na)
            fl_ml = dL / d
            fl_pt = X['pL'] / X['n_pt']
            ok = d > 0
            zr['distinguishable_fraction_by_band'] = {
                lab: qdict(phi[ok & (X['band'] == lab)]) for lab, _, _ in BANDS}
            zr['distinguishable_fraction_all'] = qdict(phi[ok])
            zr['fracL_point_minus_ml_by_band'] = {
                lab: qdict((fl_pt - fl_ml)[ok & (X['band'] == lab)]) for lab, _, _ in BANDS}
            zr['fracL_point_vs_ml_pearson'] = float(np.corrcoef(fl_pt[ok], fl_ml[ok])[0, 1]) if ok.sum() > 2 else None
            zr['abs_fracL_point_minus_ml_median'] = float(np.median(np.abs(fl_pt - fl_ml)[ok])) if ok.any() else None
            r[tag] = zr
        # category shares of each pair's reads (inclusive), by band
        tot = sum(X[f'{c}_incl'] for c in CATS)
        r['read_category_share_by_band'] = {
            lab: {c: qdict((X[f'{c}_incl'] / tot)[(X['band'] == lab) & (tot > 0)], qs=(0.5,)) | {
                'pooled': float(X.loc[X['band'] == lab, f'{c}_incl'].sum() / max(tot[X['band'] == lab].sum(), 1))}
                for c in CATS} | {'shared_with_other_gene_pooled': float(
                    sum(X.loc[X['band'] == lab, f'{c}_shared'].sum() for c in CATS) /
                    max(tot[X['band'] == lab].sum(), 1))}
            for lab, _, _ in BANDS}
        # emulator prediction of the Gibbs variance (strict counts; both
        # haplotypes with distinguishable reads, so the Beta is proper either way)
        dL, dR, na = (X[f'{c}_strict'].to_numpy(float) for c in ('L', 'R', 'LR'))
        kL, kR = X['kL'].to_numpy(float), X['kR'].to_numpy(float)
        va_obs = X['Va_gibbs'].to_numpy(float)
        closed = (polygamma(1, dL + np.maximum(kL, 1)) + polygamma(1, dR + np.maximum(kR, 1))) / LN2 ** 2
        sim = emulate_va(dL, dR, na, kL, kR, RNG[2])
        # noise floor: the same emulator at 200 draws against itself at 2,000.
        # If the emulator were exact, sim/obs would scatter at least this much
        # (the observed draws have ~170 effective draws, so slightly more).
        sim200 = emulate_va(dL, dR, na, kL, kR, RNG[4], ndraw=200)
        q_a = (1 / (X['pL'].to_numpy() + KAPPA) + 1 / (X['pR'].to_numpy() + KAPPA)) / LN2 ** 2
        both = (X['pL'].to_numpy() > 0) & (X['pR'].to_numpy() > 0)
        em = {}
        for lab, _, _ in BANDS:
            bm = (X['band'] == lab).to_numpy()
            em[lab] = dict(
                pairs_both_positive=int((bm & both).sum()),
                sim_over_obs_both_positive=qdict((sim / va_obs)[bm & both]),
                noise_floor_sim200_over_sim2000_both_positive=qdict((sim200 / sim)[bm & both]),
                closed_over_obs_both_positive=qdict((closed / va_obs)[bm & both]),
                sim_over_obs_one_zero=qdict((sim / va_obs)[bm & ~both]),
                q_over_Vagibbs_vs_phi=dict(
                    q_over_Va=qdict((q_a / va_obs)[bm & both]),
                    phi=qdict(((dL + dR) / (dL + dR + na))[bm & both & (dL + dR > 0)])))
        allb = both & np.isfinite(va_obs) & (va_obs > 0)
        em['all_both_positive_spearman_sim_vs_obs'] = float(spearmanr(sim[allb], va_obs[allb])[0])
        em['all_both_positive_log10_ratio_sim_obs'] = qdict(np.log10(sim[allb] / va_obs[allb]))
        em['all_both_positive_log10_noise_floor_sim200_sim2000'] = qdict(np.log10(sim200[allb] / sim[allb]))
        em['all_both_positive_sd_log10_ratio_sim_obs'] = float(np.std(np.log10(sim[allb] / va_obs[allb])))
        em['all_both_positive_sd_log10_noise_floor'] = float(np.std(np.log10(sim200[allb] / sim[allb])))
        phi_all = (dL + dR) / np.maximum(dL + dR + na, 1e-12)
        mm = allb & (dL + dR > 0)
        em['spearman_q_over_Va_vs_phi'] = float(spearmanr((q_a / va_obs)[mm], phi_all[mm])[0])
        r['emulator_va'] = em
        # shot noise on the paired draw total: Fano by band
        mLR = (X['mL'] + X['mR']).to_numpy()
        fano = X['vLR'].to_numpy() / np.where(mLR > 0, mLR, np.nan)
        r['fano_LR_by_band'] = {lab: qdict(fano[(X['band'] == lab).to_numpy()]) for lab, _, _ in BANDS}
        # draw-mean excess of a point-zero haplotype vs its active copies
        zL = ((X['pL'] == 0) & (X['pR'] > 0)).to_numpy(); zR = ((X['pR'] == 0) & (X['pL'] > 0)).to_numpy()
        zz = zL | zR
        mz = np.where(zL, X['mL'], X['mR'])[zz]
        kz = np.maximum(np.where(zL, X['kL'], X['kR'])[zz], 1)
        ko = np.maximum(np.where(zL, X['kR'], X['kL'])[zz], 1)
        dz_own = np.where(zL, X['L_strict'], X['R_strict'])[zz]
        dz_oth = np.where(zL, X['R_strict'], X['L_strict'])[zz]
        naz = X['LR_strict'].to_numpy()[zz]
        # Beta-mean prediction of the zeroed haplotype's draw mean:
        # E[p] * E[S] = (d_own + k_own) / (d_own + d_oth + k_own + k_oth) * (N + k_own + k_oth)
        pred = ((dz_own + kz) / (dz_own + dz_oth + kz + ko)) * (dz_own + dz_oth + naz + kz + ko)
        r['zeroed_haplotype'] = dict(
            pairs=int((zL | zR).sum()), draw_mean=qdict(mz), active_copies=qdict(kz),
            draw_mean_over_beta_prediction=qdict(mz / pred))
        res[scope] = r
    (OUT / 'eq_class_summary.json').write_text(json.dumps(res, indent=1))
    print(json.dumps(res['subset50x3'], indent=1)[:5000])
    return res


# ---------------------------------------------------------------------------
#  vcf: heterozygous exonic variants vs structural pairing
# ---------------------------------------------------------------------------

def het_counts(vcf, regions_path, samples, gene_ex, chrom_of):
    """Per (gene, donor) counts of heterozygous exonic SNVs and indels.

    Records are deduplicated by (CHROM, POS, REF, ALT) because bcftools -R
    repeats a record once per overlapping region; each gene's merged exons do
    not overlap each other, so a record counts at most once per gene."""
    cmd = ['bcftools', 'query', '-H', '-R', str(regions_path), '-s', ','.join(samples),
           '-f', '%CHROM\t%POS\t%REF\t%ALT\t%FILTER[\t%GT]\n', str(vcf)]
    p = subprocess.run(cmd, capture_output=True, text=True, check=True)
    lines = p.stdout.rstrip('\n').split('\n')
    hdr = lines[0].lstrip('# ').split('\t')
    col_samples = [h.split(']', 1)[1].rsplit(':', 1)[0] for h in hdr[5:]]
    order = [col_samples.index(s) for s in samples]
    seen = set()
    rec_chr, rec_pos, rec_het, rec_ind, rec_np = [], [], [], [], []
    for line in lines[1:]:
        if not line:
            continue
        f = line.split('\t')
        key = tuple(f[:4])
        if key in seen:
            continue
        seen.add(key)
        alleles = [f[2]] + f[3].split(',')
        gts = f[5:]
        het = np.zeros(len(samples), bool)
        ind = np.zeros(len(samples), bool)
        for j, si in enumerate(order):
            gt = gts[si].replace('|', '/').split('/')
            if len(gt) != 2 or '.' in gt or gt[0] == gt[1]:
                continue
            a0, a1 = alleles[int(gt[0])], alleles[int(gt[1])]
            het[j] = True
            ind[j] = len(a0) != len(a1) or len(a0) != len(alleles[0]) or len(a1) != len(alleles[0])
        rec_chr.append(f[0]); rec_pos.append(int(f[1])); rec_het.append(het); rec_ind.append(ind)
        rec_np.append(f[4] not in ('PASS', '.'))
    n_records = len(rec_pos)
    genes = list(gene_ex)
    snv = np.zeros((len(genes), len(samples)), np.int32)
    ind = np.zeros((len(genes), len(samples)), np.int32)
    nonpass = np.zeros((len(genes), len(samples)), np.int32)
    if n_records == 0:
        return snv, ind, nonpass, 0
    rc = np.array(rec_chr); rp = np.array(rec_pos); rh = np.array(rec_het); ri = np.array(rec_ind)
    rn = np.array(rec_np)
    by_chr = {}
    for c in np.unique(rc):
        idx = np.flatnonzero(rc == c)
        o = idx[np.argsort(rp[idx], kind='stable')]
        by_chr[c] = (rp[o], o)
    for gi_, g in enumerate(genes):
        pos, o = by_chr.get(chrom_of[g], (None, None))
        if pos is None:
            continue
        for s_, e_ in gene_ex[g]:
            lo, hi = np.searchsorted(pos, s_, 'left'), np.searchsorted(pos, e_, 'right')
            if hi <= lo:
                continue
            sel = o[lo:hi]
            h = rh[sel]; i_ = ri[sel] & h
            ind[gi_] += i_.sum(0)
            snv[gi_] += (h & ~ri[sel]).sum(0)
            nonpass[gi_] += (h & rn[sel][:, None]).sum(0)
    return snv, ind, nonpass, n_records


def part_vcf(n_genes=200):
    genes, samples, cal, eff, man, gchr = load_basics()
    names = np.load(OUT / 'names_pairing.npz')
    npair = names['npair']
    pL = np.load(PE / 'pL.npy'); pR = np.load(PE / 'pR.npy'); pT = np.load(PE / 'pT.npy')
    gi = {g: i for i, g in enumerate(genes)}
    ex = {}
    for l in EXONS.read_text().strip().split('\n'):
        g, s_, e_ = l.split('\t')
        ex[g] = list(zip(map(int, s_.split(',')), map(int, e_.split(','))))
    nc = pd.read_csv(GENES_NC, sep='\t', header=None, index_col=0)[1]
    nc = nc[~nc.index.duplicated()]
    auto = {f'chr{i}' for i in range(1, 23)}
    pool = sorted(g for g in genes if g in cal and g in ex and g in nc.index and gchr.get(g) in auto)
    pick = sorted(RNG[3].choice(pool, n_genes, replace=False).tolist())
    gene_ex = {g: ex[g] for g in pick}
    res = {'n_genes': len(pick), 'pool': len(pool)}
    tabs = {}
    for lab, vcf, chrom_of in (('rephased', VCF_REPHASED, {g: nc[g] for g in pick}),
                               ('build', VCF_BUILD, {g: gchr[g] for g in pick})):
        reg = OUT / f'vcf_regions_{lab}.tsv'
        with open(reg, 'w') as fh:
            for g in pick:
                for s_, e_ in gene_ex[g]:
                    fh.write(f'{chrom_of[g]}\t{s_}\t{e_}\n')
        t0 = time.time()
        snv, ind, nonpass, nrec = het_counts(vcf, reg, samples, gene_ex, chrom_of)
        het = snv + ind
        rows = np.array([gi[g] for g in pick])
        has_pair = npair[rows] > 0
        nt = pL[rows] + pR[rows]
        r = dict(records=nrec, seconds=round(time.time() - t0, 1), pairs=int(het.size))
        for blab, lo, hi in (('0', 0, 1), ('1', 1, 2), ('2-3', 2, 4), ('4-9', 4, 10), ('10+', 10, 10 ** 9)):
            m = (het >= lo) & (het < hi)
            r[f'het_{blab}'] = dict(pairs=int(m.sum()), no_paired=int((m & ~has_pair).sum()),
                                   share_no_paired=float((m & ~has_pair).sum() / max(m.sum(), 1)),
                                   share_pLpR_zero=float((m & (nt <= 0)).sum() / max(m.sum(), 1)))
        m1 = (het >= 1) & ~has_pair
        r['het_ge1_but_no_pair'] = dict(
            pairs=int(m1.sum()), indel_only=int((m1 & (snv == 0)).sum()),
            all_het_nonpass=int((m1 & (nonpass == het)).sum()),
            n_het=qdict(het[m1]))
        m0 = (het == 0) & has_pair
        r['het_0_but_paired'] = dict(pairs=int(m0.sum()), genes=int(m0.any(1).sum()))
        r['share_no_paired_in_sample'] = float((~has_pair).mean())
        r['share_pLpR_zero_in_sample'] = float((nt <= 0).mean())
        res[lab] = r
        tabs[lab] = (snv, ind, nonpass)
        print(f'  vcf {lab}: {json.dumps(r)[:1500]}', flush=True)
    # per (gene, donor) table
    rows = np.array([gi[g] for g in pick])
    recs = []
    for j, g in enumerate(pick):
        for si, s in enumerate(samples):
            recs.append((g, s, int(npair[rows[j], si]), float(pL[rows[j], si] + pR[rows[j], si]),
                         float(pT[rows[j], si]),
                         *(int(tabs['rephased'][k][j, si]) for k in range(3)),
                         *(int(tabs['build'][k][j, si]) for k in range(3))))
    pd.DataFrame(recs, columns=['gene', 'donor', 'paired_transcripts', 'pL_plus_pR', 'pT',
                                'rephased_het_snv', 'rephased_het_indel', 'rephased_het_nonpass',
                                'build_het_snv', 'build_het_indel', 'build_het_nonpass']).to_csv(
        OUT / 'vcf_het_vs_pairing.tsv.gz', sep='\t', index=False, compression='gzip')
    (OUT / 'vcf_summary.json').write_text(json.dumps(res, indent=1))
    return res


def part_phihet():
    """Distinguishable fraction against heterozygous exonic variant count, for
    the three equivalence-class donors and every autosomal calibration gene."""
    genes, samples, cal, eff, man, gchr = load_basics()
    E = pd.read_csv(OUT / 'eq_class_decomposition.tsv.gz', sep='\t')
    donors = json.loads((OUT / 'eq_class_summary.json').read_text())['donors']
    ex = {}
    for l in EXONS.read_text().strip().split('\n'):
        g, s_, e_ = l.split('\t')
        ex[g] = list(zip(map(int, s_.split(',')), map(int, e_.split(','))))
    nc = pd.read_csv(GENES_NC, sep='\t', header=None, index_col=0)[1]
    nc = nc[~nc.index.duplicated()]
    auto = {f'chr{i}' for i in range(1, 23)}
    pick = sorted(g for g in genes if g in cal and g in ex and g in nc.index and gchr.get(g) in auto)
    reg = OUT / 'vcf_regions_phihet.tsv'
    with open(reg, 'w') as fh:
        for g in pick:
            for s_, e_ in ex[g]:
                fh.write(f'{nc[g]}\t{s_}\t{e_}\n')
    t0 = time.time()
    snv, ind, nonpass, nrec = het_counts(VCF_REPHASED, reg, donors, {g: ex[g] for g in pick},
                                         {g: nc[g] for g in pick})
    print(f'  phihet: {len(pick)} genes, {nrec} records, {time.time()-t0:.0f}s', flush=True)
    exlen = {g: sum(e_ - s_ + 1 for s_, e_ in ex[g]) for g in pick}
    H = pd.DataFrame([(g, d, int(snv[i, j]), int(ind[i, j]), exlen[g])
                      for i, g in enumerate(pick) for j, d in enumerate(donors)],
                     columns=['gene', 'donor', 'het_snv', 'het_indel', 'exon_bp'])
    X = E.merge(H, on=['gene', 'donor'], how='inner')
    X['het'] = X['het_snv'] + X['het_indel']
    X.to_csv(OUT / 'phi_vs_het.tsv.gz', sep='\t', index=False, compression='gzip')
    d = X['L_strict'] + X['R_strict']
    N = d + X['LR_strict']
    X['phi'] = np.where(N > 0, d / N.clip(lower=1e-12), np.nan)
    paired = (X['kL'] + X['kR']) > 0
    res = {'donors': donors, 'genes': len(pick), 'records': nrec, 'pairs': int(len(X))}
    bins = (('0', 0, 1), ('1', 1, 2), ('2-3', 2, 4), ('4-9', 4, 10), ('10+', 10, 10 ** 9))
    res['by_het'] = {}
    for lab, lo, hi in bins:
        m = (X['het'] >= lo) & (X['het'] < hi)
        mi = m & (X['n_pt'] > 0)
        res['by_het'][lab] = dict(
            pairs=int(m.sum()), with_active_paired_copies=int((m & paired).sum()),
            informative=int(mi.sum()),
            share_informative_no_distinguishable=float(((d == 0) & mi).sum() / max(mi.sum(), 1)),
            phi_informative=qdict(X['phi'][mi]),
            phi_informative_100plus=qdict(X['phi'][mi & (X['n_pt'] >= 100)]),
            share_one_haplotype_point_zero=float((mi & ((X['pL'] == 0) ^ (X['pR'] == 0))).sum() / max(mi.sum(), 1)))
    X['het_per_kb'] = X['het'] / (X['exon_bp'] / 1000)
    mi = (X['n_pt'] >= 100) & (X['het'] > 0)
    res['spearman_phi_vs_het_per_kb_100plus'] = float(spearmanr(X['phi'][mi], X['het_per_kb'][mi], nan_policy='omit')[0])
    res['spearman_phi_vs_het_100plus'] = float(spearmanr(X['phi'][mi], X['het'][mi], nan_policy='omit')[0])
    res['phi_by_het_per_kb_100plus'] = {}
    for lab, lo, hi in (('<0.5', 0, 0.5), ('0.5-1', 0.5, 1), ('1-2', 1, 2), ('2-4', 2, 4), ('4+', 4, 1e9)):
        m = mi & (X['het_per_kb'] >= lo) & (X['het_per_kb'] < hi)
        res['phi_by_het_per_kb_100plus'][lab] = qdict(X['phi'][m])
    (OUT / 'phi_vs_het_summary.json').write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1)[:4000])
    return res


def part_softcheck():
    """Are single-difference fragments present as soft-weighted L+R classes,
    and does the Gibbs variance use them?

    Salmon's defaults (hardFilter false, scoreExp 1, minAlnProb 1e-5, mismatch
    penalty 6 score units against a match) keep a fragment's alignment to the
    other haplotype copy when it differs there at ONE SNV (weight e^-6) and drop
    it at two (e^-12 < 1e-5). The dump carries no weights, but range
    factorization (4 bins) writes one line per weight pattern, so the same
    two-transcript label {T_L, T_R} appears on several lines when some of its
    fragments are skewed. Restricted to genes with ONE annotated transcript,
    paired in the donor, where L and R copies differ only by the donor's
    variants. The largest {T_L, T_R} line is taken as the balanced fragments;
    the remaining lines are the candidate soft-weighted fragments."""
    genes, samples, cal, eff, man, gchr = load_basics()
    t2g = tx2gene()
    names_np = np.load(OUT / 'names_pairing.npz')
    nann = names_np['n_annot_tx']
    single = {g for g, n in zip(genes, nann) if n == 1}
    gi = {g: i for i, g in enumerate(genes)}
    donors = json.loads((OUT / 'eq_class_summary.json').read_text())['donors']
    pL = np.load(PE / 'pL.npy'); pR = np.load(PE / 'pR.npy')
    arrs = np.load(OUT / 'cache_pair_arrays.npz')
    Va_g = arrs['Va_g']
    rows = []
    for s in donors:
        si = samples.index(s)
        with gzip.open(Path(man[s]) / 'aux_info' / 'eq_classes.txt.gz', 'rt') as fh:
            ntx = int(fh.readline()); fh.readline()
            names = [fh.readline().rstrip('\n') for _ in range(ntx)]
            lines = fh.read().split('\n')
        pairs = pair_haplotypes(names, SUFFIXES)
        idx = {}
        for base, (ia, ib) in pairs.items():
            g = t2g.get(base)
            if g in single and g in gi:
                idx[ia] = (g, 'L'); idx[ib] = (g, 'R')
        lr = defaultdict(list); dL = defaultdict(float); dR = defaultdict(float)
        for line in lines:
            if not line:
                continue
            f = line.split('\t')
            k = int(f[0])
            if k == 1:
                t = int(f[1])
                if t in idx:
                    g, h = idx[t]
                    (dL if h == 'L' else dR)[g] += float(f[2])
            elif k == 2:
                a, b = int(f[1]), int(f[2])
                if a in idx and b in idx and idx[a][0] == idx[b][0]:
                    lr[idx[a][0]].append(float(f[3]))
        for g in sorted({v[0] for v in idx.values()}):
            c = sorted(lr.get(g, []), reverse=True)
            rows.append(dict(gene=g, donor=s, dL=dL.get(g, 0.0), dR=dR.get(g, 0.0),
                             n_lr_lines=len(c), lr_largest=c[0] if c else 0.0,
                             lr_rest=float(sum(c[1:])), pL=pL[gi[g], si], pR=pR[gi[g], si],
                             Va_gibbs=float(Va_g[gi[g], si])))
        print(f'  softcheck {s}: {len(rows)} rows so far', flush=True)
    X = pd.DataFrame(rows)
    X['n_pt'] = X['pL'] + X['pR']
    X = X[X['n_pt'] > 0].reset_index(drop=True)
    X['band'] = pd.cut(X['n_pt'], [0, 10, 100, 1000, np.inf], right=False,
                       labels=[b[0] for b in BANDS]).astype(str)
    both = ((X['pL'] > 0) & (X['pR'] > 0)).to_numpy()
    ones = np.ones(len(X))
    namb = (X['lr_largest'] + X['lr_rest']).to_numpy()
    va = X['Va_gibbs'].to_numpy()
    rng = np.random.default_rng(np.random.SeedSequence(SEED).spawn(6)[5])
    simA = emulate_va(X['dL'].to_numpy(), X['dR'].to_numpy(), namb, ones, ones, rng)
    fl = (X['pL'] / X['n_pt']).to_numpy()
    soft = X['lr_rest'].to_numpy()
    simB = emulate_va(X['dL'].to_numpy() + soft * fl, X['dR'].to_numpy() + soft * (1 - fl),
                      X['lr_largest'].to_numpy(), ones, ones, rng)
    X['simA'], X['simB'] = simA, simB
    X.to_csv(OUT / 'soft_weight_check.tsv.gz', sep='\t', index=False, compression='gzip')
    res = {'note': ('The dumped eq_classes.txt merges range-factorized bins by transcript label '
                    '(dumped < internal class counts below), so every {T_L, T_R} label is one line and '
                    'lr_rest is 0 by construction: the strict_plus_soft arm is inert and says nothing '
                    'about soft-weighted fragments. The evidence on them is strict_emulator_by_het_100plus.'),
           'donors': donors, 'pairs_single_transcript_informative': int(len(X)),
           'lr_lines_per_pair': {str(k): int((X['n_lr_lines'] == k).sum()) for k in range(0, 4)} |
           {'4+': int((X['n_lr_lines'] >= 4).sum())},
           'soft_share_of_LR_reads': qdict((X['lr_rest'] / namb.clip(min=1e-12))[namb > 0]),
           'soft_over_strict_distinguishable': qdict((X['lr_rest'] / (X['dL'] + X['dR']))[(X['dL'] + X['dR']) > 0]),
           'bands': {}}
    for lab, _, _ in BANDS:
        m = (X['band'] == lab).to_numpy() & both & (va > 0)
        res['bands'][lab] = dict(pairs_both_positive=int(m.sum()),
                                 strict_only_sim_over_obs=qdict(simA[m] / va[m]),
                                 strict_plus_soft_sim_over_obs=qdict(simB[m] / va[m]))
    # the dump merges range-factorized bins by label: header count vs meta count
    res['dumped_vs_internal_eq_classes'] = {}
    for s in donors:
        with gzip.open(Path(man[s]) / 'aux_info' / 'eq_classes.txt.gz', 'rt') as fh:
            fh.readline(); n_dump = int(fh.readline())
        n_meta = json.loads((Path(man[s]) / 'aux_info' / 'meta_info.json').read_text())['num_eq_classes']
        res['dumped_vs_internal_eq_classes'][s] = dict(dumped=n_dump, internal=int(n_meta))
    # strict-count emulator against the donor's heterozygous variants
    # (calibration genes only; phi_vs_het.tsv.gz from part phihet)
    ph = OUT / 'phi_vs_het.tsv.gz'
    if ph.exists():
        H = pd.read_csv(ph, sep='\t', usecols=['gene', 'donor', 'het_snv', 'het_indel'])
        Y = X.merge(H, on=['gene', 'donor'], how='inner')
        Y = Y[(Y['pL'] > 0) & (Y['pR'] > 0) & (Y['Va_gibbs'] > 0) & (Y['n_pt'] >= 100)]
        r_ = Y['simA'] / Y['Va_gibbs']
        phi_ = (Y['dL'] + Y['dR']) / (Y['dL'] + Y['dR'] + Y['lr_largest'])
        res['strict_emulator_by_het_100plus'] = {}
        for lab, sel in (('1_snv_only', (Y['het_snv'] == 1) & (Y['het_indel'] == 0)),
                         ('2-3_snv_only', Y['het_snv'].between(2, 3) & (Y['het_indel'] == 0)),
                         ('4plus_snv_only', (Y['het_snv'] >= 4) & (Y['het_indel'] == 0)),
                         ('any_indel', Y['het_indel'] > 0)):
            res['strict_emulator_by_het_100plus'][lab] = dict(sim_over_obs=qdict(r_[sel]), phi=qdict(phi_[sel]))
    m = both & (va > 0)
    res['strict_only_sd_log10'] = float(np.std(np.log10(simA[m] / va[m])))
    res['strict_plus_soft_sd_log10'] = float(np.std(np.log10(simB[m] / va[m])))
    res['strict_only_median_log10'] = float(np.median(np.log10(simA[m] / va[m])))
    res['strict_plus_soft_median_log10'] = float(np.median(np.log10(simB[m] / va[m])))
    (OUT / 'soft_weight_check.json').write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return res


def part_mixing():
    """Does the Gibbs chain reach its stationary spread within each segment?

    Salmon runs 8 chains for 200 draws and resets every chain to the VB point
    estimate at draws 0, 25, 50, ... (CollapsedGibbsSampler.cpp: newChainIter,
    alphasIn = alphasInit), 16 internal rounds per draw. The cache keeps
    Salmon's draw order (read_salmon_bootstraps reshapes n_draws x n_txp), so
    draw index mod 25 is the position within a segment. If mixing from the
    sparse start is incomplete, a haplotype the point estimate zeroes sits
    lower at early positions, and the across-draw variance of the allelic log
    ratio is smaller early than late. Position blocks: 0-4, 5-9, ..., 20-24,
    each pooled over the 8 segments (40 draws per block)."""
    genes, samples, cal, eff, man, gchr = load_basics()
    pL = np.load(PE / 'pL.npy'); pR = np.load(PE / 'pR.npy')
    YL = np.load(CACHE / 'YL.npy', mmap_mode='r'); YR = np.load(CACHE / 'YR.npy', mmap_mode='r')
    G, S = pL.shape
    zmean = np.full((G, S, 5), np.nan, np.float32)    # zeroed side's draw mean by block
    avar = np.zeros((G, S, 5), np.float32)            # allelic log-ratio variance by block
    t0 = time.time()
    for s0 in range(0, G, CHUNK):
        sl = slice(s0, min(G, s0 + CHUNK))
        yL = np.asarray(YL[sl]).reshape(-1, S, 8, 5, 5)   # segment, block, position
        yR = np.asarray(YR[sl]).reshape(-1, S, 8, 5, 5)
        a = np.log2((yL + KAPPA) / (yR + KAPPA))
        avar[sl] = a.transpose(0, 1, 3, 2, 4).reshape(a.shape[0], S, 5, 40).var(-1)
        zL = (pL[sl] == 0) & (pR[sl] > 0); zR = (pR[sl] == 0) & (pL[sl] > 0)
        yz = np.where(zL[..., None, None, None], yL, yR).mean(axis=(2, 4))
        zmean[sl] = np.where((zL | zR)[..., None], yz, np.nan)
        print(f'  mixing genes {sl.stop}/{G}  {time.time()-t0:.0f}s', flush=True)
    np.savez(OUT / 'mixing_blocks.npz', zmean=zmean, avar=avar)
    iscal = np.array([g in cal for g in genes])[:, None] & np.ones((1, S), bool)
    n = pL + pR
    inf = n > 0
    zone = inf & ((pL == 0) ^ (pR == 0))
    both = inf & (pL > 0) & (pR > 0)
    tot = np.nanmean(zmean, axis=2)
    res = {'blocks': ['0-4', '5-9', '10-14', '15-19', '20-24'], 'bands': {}}
    for lab, m0 in band_masks(n).items():
        mz = iscal & m0 & zone & (tot > 0)
        mb = iscal & m0 & both
        vall = avar.mean(-1)
        res['bands'][lab] = dict(
            zeroed_pairs=int(mz.sum()),
            zeroed_side_block_mean_over_all=[qdict(zmean[..., b][mz] / tot[mz], qs=(0.25, 0.5, 0.75)) for b in range(5)],
            zeroed_side_last_over_first_block=qdict(zmean[..., 4][mz] / np.maximum(zmean[..., 0][mz], 1e-12)),
            both_positive_pairs=int(mb.sum()),
            va_block_over_mean_both_positive=[qdict(avar[..., b][mb] / np.maximum(vall[mb], 1e-300), qs=(0.25, 0.5, 0.75)) for b in range(5)],
            va_block_over_mean_one_zero=[qdict(avar[..., b][mz] / np.maximum(vall[mz], 1e-300), qs=(0.25, 0.5, 0.75)) for b in range(5)])
    # the strict-count emulator against the LAST block only, by het class
    sw = OUT / 'soft_weight_check.tsv.gz'
    ph = OUT / 'phi_vs_het.tsv.gz'
    if sw.exists() and ph.exists():
        gi = {g: i for i, g in enumerate(genes)}
        X = pd.read_csv(sw, sep='\t').merge(
            pd.read_csv(ph, sep='\t', usecols=['gene', 'donor', 'het_snv', 'het_indel']), on=['gene', 'donor'])
        X = X[(X['pL'] > 0) & (X['pR'] > 0) & (X['Va_gibbs'] > 0) & (X['n_pt'] >= 100)]
        gg = X['gene'].map(gi).to_numpy(); ss = X['donor'].map({s: i for i, s in enumerate(samples)}).to_numpy()
        vfirst, vlast = avar[gg, ss, 0], avar[gg, ss, 4]
        out = {}
        for lab, sel in (('1_snv_only', (X['het_snv'] == 1) & (X['het_indel'] == 0)),
                         ('2-3_snv_only', X['het_snv'].between(2, 3) & (X['het_indel'] == 0)),
                         ('4plus_snv_only', (X['het_snv'] >= 4) & (X['het_indel'] == 0)),
                         ('any_indel', X['het_indel'] > 0)):
            sel = sel.to_numpy()
            out[lab] = dict(pairs=int(sel.sum()),
                            sim_over_obs_all_draws=qdict((X['simA'].to_numpy() / X['Va_gibbs'].to_numpy())[sel]),
                            sim_over_obs_first_block=qdict((X['simA'].to_numpy() / vfirst)[sel]),
                            sim_over_obs_last_block=qdict((X['simA'].to_numpy() / vlast)[sel]),
                            last_over_first_block_va=qdict((vlast / vfirst)[sel]),
                            va_block_over_mean_median=[float(np.median(
                                (avar[gg, ss, b] / np.maximum(avar[gg, ss].mean(-1), 1e-300))[sel]))
                                for b in range(5)])
        res['strict_emulator_by_het_100plus_by_block'] = out
    (OUT / 'mixing_summary.json').write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1)[:6000])
    return res


def _fq(q):
    return ' '.join(f"{q[c]:.4g}" if q.get(c) is not None else 'NA'
                    for c in ('q10', 'q25', 'q50', 'q75', 'q90') if c in q) + f" (n={q['n']})"


def _walk(x, pad=''):
    for k, v in x.items():
        if isinstance(v, dict) and 'n' in v and any(c.startswith('q') for c in v):
            print(f'{pad}{k:58s} {_fq(v)}')
        elif isinstance(v, dict):
            print(f'{pad}{k}:')
            _walk(v, pad + '  ')
        else:
            print(f'{pad}{k:58s} {v}')


def part_print():
    """Print the JSON summaries written by the other parts, compactly."""
    for f in ('names_pairing_summary.json', 'cache_targets.json', 'eq_class_summary.json', 'vcf_summary.json',
              'phi_vs_het_summary.json', 'soft_weight_check.json', 'mixing_summary.json'):
        p = OUT / f
        if p.exists():
            print(f'##### {f}')
            _walk(json.loads(p.read_text()))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--part', default='all',
                    choices=('all', 'names', 'cache', 'eq', 'eqsum', 'vcf', 'phihet', 'softcheck', 'mixing', 'print'))
    a = ap.parse_args()
    if a.part == 'print':
        part_print()
        return
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for name, fn in (('names', part_names), ('cache', part_cache), ('eq', part_eq), ('eqsum', part_eqsum),
                     ('vcf', part_vcf), ('phihet', part_phihet), ('softcheck', part_softcheck),
                     ('mixing', part_mixing)):
        if a.part == name or (a.part == 'all' and name != 'eqsum'):
            print(f'== {name}', flush=True)
            fn()
            print(f'== {name} done at {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()

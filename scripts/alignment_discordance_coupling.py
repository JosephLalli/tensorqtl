"""Is the allelic channel's positive weight-residual coupling carried by records
whose Salmon allelic ratio disagrees with alignment-based allele counts?

QUESTION. On the records-permutation null (null_permutation_instrument.py; 46
genes x 92 donors x 2,000 permutations at RASQUAL's observed lead) hapmixQTL's
allelic nominal p rejects at 0.0692 / 0.0200 / 0.0057 at 0.05 / 0.01 / 0.001.
Shuffling each gene's whitened residuals z = a/sqrt(va) against its weights
w = 1/va (DECOUPLE, both marginals kept) brings that to ~0.052 / 0.012 /
0.0016, so REAL minus DECOUPLE is a coupling between weight and squared
residual inside genes. Its first-order closed form is

    R_g = n_a sum(w z^2) / (sum(w) sum(z^2)),
    R_g - 1 = sum_j c_j,   c_j = (w_j / mean(w) - 1) * z_j^2 / sum(z^2)

(c_j is record j's exact additive share of the coupling). This script asks
whether the coupling is carried by CONFIDENT SALMON POINT-ESTIMATE ERRORS --
records whose posterior-mean allelic log ratio a sits far from phASER's
alignment-based log ratio ap although the Gibbs variance va says a is precise.

DISCORDANCE. For a record with a phASER measurement,
    ap   = log((aCount + 1/2) / (bCount + 1/2))            (phASER haplotype
           counts on STAR alignments, haplotype A anchored to the genome-wide
           phase, gw_phased == 1, the imbalance_downweighting.py loader)
    qinf = 1/(aCount + 1/2) + 1/(bCount + 1/2)              (Haldane-corrected
           counting variance of ap)
    dz   = (a - ap) / sqrt(va + qinf)
dz is NOT a pure Salmon-error score. It also absorbs: (i) phASER sees only
reads overlapping heterozygous SNPs (indels are excluded by phASER, reads that
Salmon assigns through indel-only or distant heterozygous sites are invisible
to it); (ii) reference-mapping bias of STAR on the linear T2T reference, which
Salmon's personalized diploid index does not have; (iii) phASER counts
uniquely mapped reads only (MAPQ 255), Salmon also resolves multi-mapping
fragments; (iv) phASER's gene interval includes intronic and overlapping-gene
SNPs, Salmon's gene is its transcripts; (v) isoform-specific imbalance, since
the two methods weight the gene's isoforms differently; (vi) the two share
reads, so their errors correlate positively and sqrt(va + qinf) over-states the
sd of the difference in the bulk. |dz| > 3 is therefore a flag of disagreement
between two measurements, one of which may be wrong for reasons unrelated to
Salmon.

DESIGN. Held fixed throughout: the instrument's 46 genes, variant, donors,
genotype s = xL - xR at the lead, the allelic fit (weighted least squares
through the origin, sigma^2 = RSS/(n_a - 1), F(1, n_a - 1)), and the
instrument's permutation stream (RandomState(42), 2,000 x permutation(92)).
What varies is ONLY the allelic record set:

 (a) Does |dz| rise with weight within gene? Per gene Spearman(w, |dz|),
     Spearman(w, |a - ap|) and the partial rank correlation of |dz| on w given
     log phASER depth; a two-sided sign test across genes; the |dz| > 3/4/5
     rate and the undeclared discrepancy variance mean((a-ap)^2 - va - qinf)
     by within-gene weight decile; an additive-floor fit
     (a-ap)^2 ~ alpha (va+qinf) + tau^2. Transcriptome-wide as well, on
     coupling_reach's all-gene summaries (G3-verified there against the Gibbs
     cache, re-gated here) for genes with >= 20 phASER-comparable records.
     Model-free split of the coupling: z^2 = u + d with u = a ap / va (the part
     phASER reproduces) and d = a (a - ap) / va (the part it does not), so
     R_g - 1 = sum_j (w_j/mean w - 1)(u_j + d_j)/sum z^2 exactly over
     comparable records; its floor is a within-gene shuffle of (z^2, u, d)
     jointly against w (200 shuffles).
 (b) Exclude records with |dz| > 3, 4, 5; recompute R_g and the rates.
     Controls, each 40 sets per threshold, keeping the REAL records (and so
     the real heavy-tailed z marginal):
       CTRLW  the same number of records per gene, drawn at random from the
              phASER-concordant (|dz| <= 3) comparable records in the SAME
              within-gene weight deciles as the flagged ones
       CTRLZ  as CTRLW but each drawn from the 3 concordant records of the
              same decile closest in |z| to the flagged record it matches:
              equal weight AND equal residual size, but phASER agrees.
              Because c_j depends only on (w_j, z_j), a perfect CTRLZ match
              removes the same coupling by construction; what it measures is
              whether equally large residuals at equal weight EXIST among
              concordant records, i.e. whether discordance is what makes a
              record coupling-bearing.
     Each arm's own DECOUPLE (40 shuffles of the remaining z against the
     remaining w; one per control set) nets out the heavy-tailed-marginal
     part, so COUPLING(arm) = rate(arm) - rate(DECOUPLE of arm's records).
 (c) Replace flagged records' a by ap, keeping va (REPL); control REPLC
     replaces a by ap for decile-matched concordant records (40 sets).
 (d) Singletons and phase. Salmon's L/R haplotypes were built by g2gtools
     from the SHAPEIT5-phased cohort BCF (personalized pipeline params), of
     which vcf/cohort92.NC.vcf.gz is the 92-donor subset.
     prepped/analysis.snps.maf01.vcf.gz is filtered at MAF >= 0.01, i.e.
     minor allele count >= 2 of 184 (checked here), so it holds no
     singletons; prepped/rephased.norm.vcf.gz is the same GT (phASER does not
     rewrite GT; checked here against cohort92 on chr21 exons) carrying
     phASER's read-backed local phase PG and block index PI, split to
     biallelic records. For every donor-gene pair: exonic heterozygous SNVs,
     indels, cohort singletons (AC == 1 of 184) and a DIRECT mis-phase
     detector: within a phASER read-backed block, the gene's own exonic
     heterozygous SNVs whose SHAPEIT5 relative phase (GT) disagrees with the
     read-backed phase (PG). Site-level: how often a singleton is the
     minority-orientation site of a conflicted block against common sites.
     Record-level: Mantel-Haenszel odds ratio (a pooled 2x2 odds ratio
     across strata, sum(a d / n) / sum(b c / n)) of |dz| > 3 for records with
     a singleton / a phase conflict / an indel heterozygote, stratified by
     the number of exonic heterozygous SNVs, with a gene-clustered bootstrap.
 (e) Fractions of the coupling (REAL - DECOUPLE) and of each tier's excess
     (REAL - MODEL; MODEL = z iid N(0,1) at the real weights, 40 sets) the
     flagged records account for, raw and net of CTRLW, with and without
     CALM2; transcriptome-wide the same at coupling_reach's synthetic
     Hardy-Weinberg variants (5 MAF levels, exact by the multiset argument
     in coupling_reach.py), gated against its per-gene direct rates.

INTERVALS. Gene-clustered percentile bootstrap (2,000 resamples of genes) for
every pooled rate and every difference or fraction, PAIRED across arms (same
resampled genes). Monte Carlo sd of the pooled rate across simulated or
control sets, and its standard error for the set mean. A difference "clears
its floor" when its paired bootstrap interval excludes 0 and it exceeds twice
the Monte Carlo standard error of any averaged arm in it.

GATES (the script aborts on failure):
  G1  the Gibbs cache mapped by sample id through compute_summaries_from_gibbs
      reproduces the npz a/va exactly; coupling_reach's all-gene A/Va equal
      the npz for the 59 genes exactly
  G2  REAL reproduces null_long.tsv.gz t2_a per (gene, perm) to relative 1e-9
      (absolute 1e-9 where t2 < 1e-3) and the pooled allelic AND combined
      rejection counts at 0.05/0.01/0.001 exactly
  G3  phASER orientation: pooled Spearman(a, ap) > 0.2 over comparable records
      with >= 20 phASER reads, and no gene of the 46 with >= 20 such records
      with a negative one (genes with fewer are reported, not gated); dz
      equals imbalance_downweighting_20260925/records.tsv to 1e-12
  G4  transcriptome REAL direct-permutation rates equal coupling_reach's
      b_per_gene direct_{alpha} for every gene at all four alphas (1e-12)
  G5  rephased.norm GT equals cohort92.NC GT at chr21 exonic biallelic SNVs
      (>= 99.9% of donor genotypes), and analysis.snps.maf01 has no site
      with minor allele count < 2 on chr21

Master seed 42. The instrument stream is RandomState(42); SeedSequence(42)
child 3 is reused deliberately to rebuild coupling_reach's synthetic genotypes;
this script's own draws use children 10-19. Natural-log units throughout.
Scratch in <out>/scratch/. Measurement only: no filter is proposed.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import contextlib                                   # noqa: E402
import io                                           # noqa: E402
import json                                         # noqa: E402
import subprocess                                   # noqa: E402
import sys                                          # noqa: E402
import time                                         # noqa: E402
from multiprocessing import get_context             # noqa: E402
from pathlib import Path                            # noqa: E402

import numpy as np                                  # noqa: E402
import pandas as pd                                 # noqa: E402
from scipy import stats as sps                      # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

D = Path('/mnt/ssd/lalli/brainvar_hapmix_deploy')
INST = D / 'nominal_p_null_instrument_20260925'
CACHE = D / 'cache' / 'gibbs_56b63c3b37ed5df8'
CREACH = D / 'coupling_reach_20260925'
IMB = D / 'imbalance_downweighting_20260925'
PH_MAN = D / 'phaser_manifest.tsv'
VCF_REPH = D / 'prepped' / 'rephased.norm.vcf.gz'
VCF_COH = D / 'vcf' / 'cohort92.NC.vcf.gz'
VCF_MAF = D / 'prepped' / 'analysis.snps.maf01.vcf.gz'
OUT = D / 'alignment_discordance_coupling_20260925'
SCR = OUT / 'scratch'

SEED, EPS, KAPPA, N = 42, 1e-12, 0.5, 92
N_PERM, N_BOOT = 2000, 2000
ALPHAS = (0.05, 0.01, 0.001)
ALPHAS_TX = (0.05, 0.01, 0.001, 1e-4)
THR = (3.0, 4.0, 5.0)
K_SET = 40                 # model / decouple / control sets, 46 genes
K_DEC_TX = 10              # decouple sets per arm, transcriptome
K_CTRL_TX = 40             # control sets per threshold, transcriptome
N_FLOOR = 200              # shuffles for the u/d decomposition floor
MIN_NA = 20
MIN_COMP_TX = 20
MAF_LEVELS = (0.05, 0.10, 0.20, 0.30, 0.50)     # coupling_reach's
N_WORK = 24
CHUNK = 250
NHET_BINS = [0, 1, 2, 4, 8, 16, 10 ** 9]        # [lo, hi) strata of exonic het SNVs

SS = np.random.SeedSequence(SEED).spawn(20)
SS_CR_SYN = SS[3]          # coupling_reach's synthetic-genotype stream (reused)
(SS_BOOT, SS_MODEL, SS_DEC, SS_CTRLW, SS_CTRLZ, SS_REPLC, SS_TX,
 SS_BOOT_TX, SS_FLOOR, SS_SPARE) = SS[10:20]

LOGF = None


def log(*a):
    s = time.strftime('%H:%M:%S') + ' ' + ' '.join(str(x) for x in a)
    print(s, flush=True)
    if LOGF is not None:
        with open(LOGF, 'a') as fh:
            fh.write(s + '\n')


# ===========================================================================
#  loading
# ===========================================================================

def read_gene_ae(args):
    donor, pfx = args
    g = pd.read_csv(f'{pfx}.gene_ae.txt', sep='\t',
                    usecols=['name', 'aCount', 'bCount', 'gw_phased'])
    g['donor'] = donor
    return g


def load_phaser_all():
    man = pd.read_csv(PH_MAN, sep='\t', header=None, names=['donor', 'prefix'])
    with get_context('fork').Pool(16) as pool:
        parts = pool.map(read_gene_ae, list(man.itertuples(index=False, name=None)))
    P = pd.concat(parts, ignore_index=True).rename(columns={'name': 'gene'})
    dup = P.duplicated(['gene', 'donor'], keep=False)
    return P[~dup].copy(), int(dup.sum())


def phaser_arrays(P, genes, donors):
    """(G, N) aCount, bCount, gw_phased aligned to (genes, donors); NaN absent."""
    gi = {g: i for i, g in enumerate(genes)}
    di = {d: j for j, d in enumerate(donors)}
    P = P[P.gene.isin(gi) & P.donor.isin(di)]
    r = P.gene.map(gi).values
    c = P.donor.map(di).values
    out = {}
    for k in ('aCount', 'bCount', 'gw_phased'):
        M = np.full((len(genes), len(donors)), np.nan)
        M[r, c] = P[k].values
        out[k] = M
    ac, bc, gw = out['aCount'], out['bCount'], out['gw_phased']
    ap = np.log((ac + KAPPA) / (bc + KAPPA))
    qinf = 1 / (ac + KAPPA) + 1 / (bc + KAPPA)
    kinf = ac + bc
    comp = (gw == 1) & (kinf >= 1)
    return ap, qinf, kinf, comp


# ===========================================================================
#  the allelic statistic for many record sets at once
# ===========================================================================

def sets_stats(Aset, Mset, w, Sg, Sg2):
    """Aset, Mset (K, N): record values and admission; Sg (P', N) genotype at
    the position each record lands in. Returns num, den, S, dof, q, t2, valid."""
    W = Mset * w[None, :]
    U = W * Aset
    num = U @ Sg.T
    den = W @ Sg2.T
    S = (U * Aset).sum(1)
    dof = Mset.sum(1) - 1
    q = den * S[:, None] - num ** 2
    with np.errstate(divide='ignore', invalid='ignore'):
        t2 = dof[:, None] * num ** 2 / q
    valid = (den > 0) & np.isfinite(t2) & (q > 0)
    return dict(num=num, den=den, S=S, dof=dof, q=q, t2=t2, valid=valid)


def crit_counts(t2, valid, dof, alphas):
    cr = sps.f.isf(np.asarray(alphas)[None, :], 1, dof[:, None])      # K, A
    return np.stack([((t2 > cr[:, i:i + 1]) & valid).sum(1)
                     for i in range(len(alphas))], 1), valid.sum(1)


def combined_counts(st, bt, set2, doft, alphas):
    """Inverse-variance meta-analysis with the instrument's total channel."""
    with np.errstate(divide='ignore', invalid='ignore'):
        ba = st['num'] / st['den']
        sea2 = st['q'] / (st['den'] ** 2 * st['dof'][:, None])
        prec = 1 / sea2 + 1 / set2[None, :]
        b = (ba / sea2 + bt[None, :] / set2[None, :]) / prec
        t2b = b ** 2 * prec
    valid = st['valid'] & np.isfinite(t2b)
    dofc = np.minimum(st['dof'], doft)
    return crit_counts(t2b, valid, dofc, alphas)


def R_sets(Aset, Mset, w):
    W = Mset * w[None, :]
    z2 = W * Aset ** 2
    n = Mset.sum(1)
    return n * (W * z2).sum(1) / (W.sum(1) * z2.sum(1))


def dec_sets(avals, mask, w, rng, K):
    """K within-gene shuffles of z = sqrt(w) a among admitted records."""
    idx = np.where(mask)[0]
    z = np.sqrt(w[idx]) * avals[idx]
    A = np.zeros((K, len(w)))
    for k in range(K):
        A[k, idx] = rng.permutation(z) / np.sqrt(w[idx])
    return A, np.repeat(mask[None, :], K, 0)


def weight_deciles(w, keep):
    dec = np.full(len(w), -1)
    idx = np.where(keep)[0]
    o = idx[np.argsort(w[idx], kind='stable')]
    dec[o] = (np.arange(len(o)) * 10) // len(o)
    return dec


def pick_controls(dec, flagged, pool, rng, zabs=None, nn=3):
    """One control record per flagged record, same weight decile (nearest
    decile when the decile's pool is exhausted), without replacement."""
    avail = pool.copy()
    chosen, borrowed, zr = [], 0, []
    for j in rng.permutation(flagged):
        d = dec[j]
        for dd in sorted(range(10), key=lambda x: (abs(x - d), x)):
            cand = np.where(avail & (dec == dd))[0]
            if len(cand):
                break
        else:
            return None, None, None
        borrowed += int(dd != d)
        if zabs is None:
            c = rng.choice(cand)
        else:
            near = cand[np.argsort(np.abs(zabs[cand] - zabs[j]), kind='stable')][:nn]
            c = rng.choice(near)
            zr.append(zabs[c] / zabs[j])
        chosen.append(c)
        avail[c] = False
    return np.array(chosen, int), borrowed, zr


# ===========================================================================
#  bootstrap helpers
# ===========================================================================

def mult_matrix(rng, G, B):
    M = np.zeros((B, G), np.int32)
    for b in range(B):
        M[b] = np.bincount(rng.integers(0, G, G), minlength=G)
    return M


def pooled(cnt, nval, M=None):
    """cnt (G,), nval (G,): point pooled rate and bootstrap replicates."""
    est = cnt.sum() / nval.sum()
    if M is None:
        return est
    return est, (M @ cnt) / (M @ nval)


def ci(x):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    return [float(np.quantile(x, .025)), float(np.quantile(x, .975))] if len(x) else [None, None]


def binom_sign(k, n):
    return float(sps.binomtest(int(k), int(n), 0.5).pvalue) if n else None


def mh_or(T):
    """T (S, 4) = a, b, c, d per stratum (exposed-case, exposed-noncase,
    unexposed-case, unexposed-noncase). Mantel-Haenszel pooled odds ratio."""
    a, b, c, d = T[..., 0], T[..., 1], T[..., 2], T[..., 3]
    n = a + b + c + d
    with np.errstate(divide='ignore', invalid='ignore'):
        num = np.where(n > 0, a * d / np.where(n > 0, n, 1), 0).sum(-1)
        den = np.where(n > 0, b * c / np.where(n > 0, n, 1), 0).sum(-1)
        return num / den


# ===========================================================================
#  (a) weight against discordance, and the u/d split of the coupling
# ===========================================================================

def partial_spearman(x, y, zc):
    rx, ry, rz = (sps.rankdata(v) for v in (x, y, zc))
    X = np.column_stack([np.ones(len(rz)), rz])
    ex = rx - X @ np.linalg.lstsq(X, rx, rcond=None)[0]
    ey = ry - X @ np.linalg.lstsq(X, ry, rcond=None)[0]
    return float(np.corrcoef(ex, ey)[0, 1])


def gene_weight_discordance(a, va, ap, qinf, kinf, comp, keep, min_comp):
    m = comp & keep
    if m.sum() < min_comp:
        return None
    w = 1 / va[m]
    adz = np.abs((a[m] - ap[m]) / np.sqrt(va[m] + qinf[m]))
    adiff = np.abs(a[m] - ap[m])
    return dict(n_comp=int(m.sum()),
                sp_w_absdz=float(sps.spearmanr(w, adz)[0]),
                sp_w_absdiff=float(sps.spearmanr(w, adiff)[0]),
                psp_w_absdz_given_kinf=partial_spearman(w, adz, np.log1p(kinf[m])))


def ud_split(a, va, ap, comp, keep, rng=None, n_floor=0):
    """Exact split of R - 1 into the phASER-shared part u = a ap / va, the
    unshared part d = a (a - ap) / va (comparable records) and the part carried
    by records without a phASER measurement."""
    w = np.where(keep, 1 / np.where(keep, va, 1), 0.0)
    a0 = np.where(keep, a, 0.0)
    z2 = w * a0 ** 2
    cm = comp & keep
    u = np.where(cm, w * a0 * np.where(cm, ap, 0), 0.0)
    d = np.where(cm, z2 - u, 0.0)
    nc = np.where(keep & ~cm, z2, 0.0)
    idx = np.where(keep)[0]
    wk = w[idx]
    cw = wk / wk.mean() - 1
    Sz = z2.sum()
    out = dict(R_minus_1=float((cw * z2[idx]).sum() / Sz),
               u=float((cw * u[idx]).sum() / Sz), d=float((cw * d[idx]).sum() / Sz),
               noncomp=float((cw * nc[idx]).sum() / Sz))
    if n_floor:
        F = np.zeros((n_floor, 3))
        tri = np.stack([u[idx], d[idx], nc[idx]], 1)
        for k in range(n_floor):
            p = rng.permutation(len(idx))
            F[k] = (cw[:, None] * tri[p]).sum(0) / Sz
        out['floor'] = F
    return out


# ===========================================================================
#  transcriptome worker: R arms and direct permutation at synthetic variants
# ===========================================================================
_TX = {}


def tx_chunk(ci_):
    g0 = ci_ * CHUNK
    A, Va, AP, QI, CO = (_TX[k] for k in ('A', 'Va', 'AP', 'QI', 'CO'))
    g1 = min(g0 + CHUNK, A.shape[0])
    Sg, Sg2 = _TX['Sg'], _TX['Sg2']
    rng = np.random.default_rng(_TX['seeds'][ci_])
    nA = len(ALPHAS_TX)
    res = []
    for j in range(g0, g1):
        a, va = A[j], Va[j]
        keep = np.isfinite(a) & np.isfinite(va) & (va > EPS)
        n_a = int(keep.sum())
        if n_a < MIN_NA:
            continue
        w = np.where(keep, 1 / np.where(keep, va, 1), 0.0)
        a0 = np.where(keep, a, 0.0)
        comp = CO[j] & keep
        with np.errstate(invalid='ignore'):
            dz = np.where(comp, (a0 - np.where(comp, AP[j], 0)) /
                          np.sqrt(va + np.where(comp, QI[j], 0)), np.nan)
        dec = weight_deciles(w, keep)
        conc = comp & (np.abs(np.nan_to_num(dz, nan=np.inf)) <= THR[0])
        r = dict(gi=j, n_a=n_a, n_comp=int(comp.sum()))
        sets, labels = [a0[None, :]], ['REAL']
        masks = [keep[None, :]]
        Ad, Md = dec_sets(a0, keep, w, rng, K_DEC_TX)
        sets.append(Ad); masks.append(Md); labels += ['DEC'] * K_DEC_TX
        z2 = w * a0 ** 2
        cw = np.where(keep, w / w[keep].mean() - 1, 0.0)
        r['R_REAL'] = float(n_a * (w * z2).sum() / (w.sum() * z2.sum()))
        for t in THR:
            fl = np.where(comp & (np.abs(np.nan_to_num(dz)) > t))[0]
            r[f'nflag_{t:g}'] = len(fl)
            r[f'cshare_{t:g}'] = float((cw[fl] * z2[fl]).sum() / z2.sum())
            if len(fl) == 0:
                continue
            if n_a - len(fl) < 5 or conc.sum() < len(fl):
                # too few records left, or too few concordant records to match:
                # the threshold is not applied to this gene (counted, reported)
                r[f'ctrl_failed_{t:g}'] = True
                continue
            ctrl = [pick_controls(dec, fl, conc, rng)[0] for _ in range(K_CTRL_TX)]
            m = keep.copy(); m[fl] = False
            sets.append(np.where(m, a0, 0)[None, :]); masks.append(m[None, :])
            labels.append(f'EXCL_{t:g}')
            Ad, Md = dec_sets(a0, m, w, rng, K_DEC_TX)
            sets.append(Ad); masks.append(Md); labels += [f'DEC_EXCL_{t:g}'] * K_DEC_TX
            Rc = []
            for ch in ctrl:
                mc = keep.copy(); mc[ch] = False
                sets.append(np.where(mc, a0, 0)[None, :]); masks.append(mc[None, :])
                labels.append(f'CTRLW_{t:g}')
                Ad, Md = dec_sets(a0, mc, w, rng, 1)
                sets.append(Ad); masks.append(Md); labels.append(f'DEC_CTRLW_{t:g}')
                Rc.append(R_sets(np.where(mc, a0, 0)[None, :], mc[None, :], w)[0])
            r[f'R_EXCL_{t:g}'] = float(R_sets(np.where(m, a0, 0)[None, :], m[None, :], w)[0])
            r[f'R_CTRLW_{t:g}'] = float(np.mean(Rc))
        Aset = np.vstack(sets); Mset = np.vstack(masks)
        st = sets_stats(Aset, Mset, w, Sg, Sg2)
        cnt, nval = crit_counts(st['t2'], st['valid'], st['dof'], ALPHAS_TX)
        lab = np.array(labels)
        for L in dict.fromkeys(labels):
            sel = lab == L
            r[f'cnt_{L}'] = cnt[sel].mean(0)
            r[f'nval_{L}'] = float(nval[sel].mean())
            if L == 'REAL':
                r['cnt_REAL_int'] = cnt[sel][0]
                r['nval_REAL_int'] = int(nval[sel][0])
        res.append(r)
    return res


# ===========================================================================
#  (d) exonic heterozygous sites, singletons, read-backed phase conflicts
# ===========================================================================

def exon_table(genes_wanted):
    ex = pd.read_csv(D / 'annot' / 'exons.tsv', sep='\t', header=None, index_col=0)
    gt = pd.read_csv(D / 'annot' / 'genes.tsv', sep='\t', header=None,
                     names=['g', 'chr', 's', 'e', 'e2'])
    chrom = gt.drop_duplicates('g').set_index('g').chr
    rows = []
    for g, r in ex.iterrows():
        if g not in genes_wanted or g not in chrom.index:
            continue
        for s, e in zip(str(r[1]).split(','), str(r[2]).split(',')):
            rows.append((chrom[g], int(s), int(e), g))       # 1-based inclusive
    E = pd.DataFrame(rows, columns=['chrom', 'start', 'end', 'gene'])
    return E.sort_values(['chrom', 'start']).reset_index(drop=True)


def merged_bed(E, path):
    out = []
    for c, sub in E.groupby('chrom', sort=False):
        iv = sub[['start', 'end']].values
        iv = iv[np.argsort(iv[:, 0])]
        cs, ce = iv[0]
        for s, e in iv[1:]:
            if s <= ce + 1:
                ce = max(ce, e)
            else:
                out.append((c, cs - 1, ce)); cs, ce = s, e
        out.append((c, cs - 1, ce))
    pd.DataFrame(out).to_csv(path, sep='\t', header=False, index=False)


def extract_chrom(args):
    chrom, bed, dst = args
    cmd = (f"bcftools view -R {bed} {VCF_REPH} -Ou | "
           f"bcftools +fill-tags -Ou -- -t AC,AN | "
           f"bcftools query -i 'GT=\"het\"' "
           f"-f '[%SAMPLE\\t%POS\\t%REF\\t%ALT\\t%AC\\t%AN\\t%GT\\t%PG\\t%PI\\t%PP\\n]' > {dst}")
    subprocess.run(cmd, shell=True, check=True)
    return chrom


def g5_gate(E):
    """rephased.norm GT == cohort92.NC GT on chr21 exonic biallelic SNVs, and
    analysis.snps.maf01 carries no minor allele count below 2 on chr21."""
    chr2nc = dict(pd.read_csv(D / 'vcf' / 'chr2nc.tsv', sep='\t', header=None).values)
    e = E[E.chrom == 'chr21']
    bed_chr = SCR / 'g5_chr21.bed'
    merged_bed(e, bed_chr)
    b = pd.read_csv(bed_chr, sep='\t', header=None)
    bed_nc = SCR / 'g5_chr21_nc.bed'
    b2 = b.copy(); b2[0] = chr2nc['chr21']
    b2.to_csv(bed_nc, sep='\t', header=False, index=False)

    def q(vcf, bed):
        out = subprocess.run(
            f"bcftools view -R {bed} -m2 -M2 -v snps {vcf} -Ou | "
            f"bcftools query -f '%POS\\t%REF\\t%ALT[\\t%GT]\\n'",
            shell=True, check=True, capture_output=True, text=True).stdout
        rows = [l.split('\t') for l in out.strip().split('\n') if l]
        df = pd.DataFrame(rows)
        df = df.drop_duplicates([0, 1, 2]).set_index([0, 1, 2])
        return df
    r1, r2 = q(VCF_REPH, bed_chr), q(VCF_COH, bed_nc)
    s1 = subprocess.run(['bcftools', 'query', '-l', str(VCF_REPH)], capture_output=True,
                        text=True, check=True).stdout.split()
    s2 = subprocess.run(['bcftools', 'query', '-l', str(VCF_COH)], capture_output=True,
                        text=True, check=True).stdout.split()
    r1.columns = s1; r2.columns = s2
    common = r1.index.intersection(r2.index)
    x1 = r1.loc[common, s2].values
    x2 = r2.loc[common].values
    nonref = (x2 != '0|0') | (x1 != '0|0')
    same = (x1 == x2)[nonref]
    out = subprocess.run(
        f"bcftools view -R {bed_chr} {VCF_MAF} -Ou | bcftools +fill-tags -Ou -- -t AC,AN | "
        f"bcftools query -f '%AC\\t%AN\\n'", shell=True, check=True, capture_output=True,
        text=True).stdout
    acan = np.array([[int(v) for v in l.split('\t')] for l in out.strip().split('\n') if l])
    mac = np.minimum(acan[:, 0], acan[:, 1] - acan[:, 0])
    # the same query on rephased.norm: singletons exist there
    out2 = subprocess.run(
        f"bcftools view -R {bed_chr} -m2 -M2 -v snps {VCF_REPH} -Ou | "
        f"bcftools +fill-tags -Ou -- -t AC,AN | bcftools query -f '%AC\\t%AN\\n'",
        shell=True, check=True, capture_output=True, text=True).stdout
    acan2 = np.array([[int(v) for v in l.split('\t')] for l in out2.strip().split('\n') if l])
    mac2 = np.minimum(acan2[:, 0], acan2[:, 1] - acan2[:, 0])
    return dict(n_sites_common=int(len(common)), n_nonref_genotypes=int(nonref.sum()),
                frac_gt_identical=float(same.mean()),
                maf01_chr21_exonic_sites=int(len(mac)), maf01_min_minor_ac=int(mac.min()),
                maf01_n_minor_ac_1=int((mac == 1).sum()),
                rephased_chr21_exonic_snvs_polymorphic=int((mac2 > 0).sum()),
                rephased_n_minor_ac_1=int((mac2 == 1).sum()))


def het_features(E, donors):
    """Per (donor, gene): exonic het SNVs / indels / singletons and read-backed
    phase conflicts among the gene's own exonic het SNVs."""
    chroms = [c for c in E.chrom.unique()]
    jobs = []
    for c in chroms:
        bed = SCR / f'exons_merged_{c}.bed'           # one bed per chromosome
        merged_bed(E[E.chrom == c], bed)
        jobs.append((c, bed, SCR / f'het_{c}.tsv'))
    t0 = time.time()
    with get_context('fork').Pool(min(N_WORK, len(jobs))) as pool:
        pool.map(extract_chrom, jobs)
    log(f'(d) het extraction over {len(jobs)} chromosomes in {time.time() - t0:.0f}s')
    cols = ['donor', 'pos', 'ref', 'alt', 'AC', 'AN', 'GT', 'PG', 'PI', 'PP']
    feats, sites_all = [], []
    for c in chroms:
        f = SCR / f'het_{c}.tsv'
        if f.stat().st_size == 0:
            continue
        H = pd.read_csv(f, sep='\t', header=None, names=cols,
                        dtype={'donor': str, 'pos': np.int64, 'ref': str, 'alt': str,
                               'AC': np.int64, 'AN': np.int64, 'GT': str, 'PG': str,
                               'PI': str, 'PP': str}, keep_default_na=False)
        H = H.drop_duplicates(['donor', 'pos', 'ref', 'alt'])
        H = H[H.donor.isin(set(donors))]
        nalt = H.groupby('pos').alt.transform('nunique')
        H['snv'] = (H.ref.str.len() == 1) & (H.alt.str.len() == 1) & (H.alt != '*')
        H['multi'] = nalt > 1
        H['singleton'] = H.AC == 1
        H['pp'] = pd.to_numeric(H.PP.replace('.', np.nan), errors='coerce')
        ph = (H.snv & ~H.multi & H.GT.isin(['0|1', '1|0']) & H.PG.isin(['0|1', '1|0'])
              & (H.PI != '.'))
        H['phased_rb'] = ph
        H['orient'] = np.where(ph, np.where(H.GT == H.PG, 1, -1), 0)
        # site -> gene
        Ec = E[E.chrom == c]
        upos = np.unique(H.pos.values)
        lo = np.searchsorted(upos, Ec.start.values, 'left')
        hi = np.searchsorted(upos, Ec.end.values, 'right')
        n = hi - lo
        sg = pd.DataFrame(dict(pos=upos[np.repeat(lo, n) + (np.arange(n.sum()) -
                                                             np.repeat(np.cumsum(n) - n, n))],
                               gene=np.repeat(Ec.gene.values, n))).drop_duplicates()
        HG = H.merge(sg, on='pos')
        # block-level (all exonic sites of the block, any gene): minority sites
        B = H[ph].copy()
        B['blk'] = B.donor + ':' + B.PI
        bc = B.groupby('blk').orient.agg(['size', 'sum'])
        npl = (bc['size'] + bc['sum']) / 2
        nmi = bc['size'] - npl
        conf = (npl > 0) & (nmi > 0)
        minor = pd.Series(np.where(npl > nmi, -1, np.where(nmi > npl, 1, 0)), index=bc.index)
        B = B.join(conf.rename('blk_conf'), on='blk').join(minor.rename('minor_or'), on='blk')
        B = B[bc.loc[B.blk, 'size'].values >= 2]
        B['is_minority'] = B.blk_conf & ((B.minor_or == 0) | (B.orient == B.minor_or))
        sites_all.append(B[['donor', 'pos', 'AC', 'singleton', 'pp', 'blk_conf', 'is_minority']]
                         .assign(chrom=c))
        # gene-level: conflict among the gene's own sites within one block
        HGp = HG[HG.phased_rb].copy()
        HGp['blk'] = HGp.donor + ':' + HGp.PI
        gb = HGp.groupby(['donor', 'gene', 'blk']).orient.agg(['size', 'sum'])
        gb['npl'] = (gb['size'] + gb['sum']) / 2
        gb['nmi'] = gb['size'] - gb['npl']
        gb['conf'] = (gb.npl > 0) & (gb.nmi > 0)
        gb['minor_or'] = np.where(gb.npl > gb.nmi, -1, np.where(gb.nmi > gb.npl, 1, 0))
        HGp = HGp.join(gb[['conf', 'minor_or']], on=['donor', 'gene', 'blk'])
        HGp['minority'] = HGp.conf & ((HGp.minor_or == 0) | (HGp.orient == HGp.minor_or))
        HGp['sing_minor'] = HGp.minority & HGp.singleton
        pg = HGp.groupby(['donor', 'gene']).agg(
            n_rb_sites=('orient', 'size'), any_conf=('conf', 'any'),
            n_minority=('minority', 'sum'), singleton_minority=('sing_minor', 'any'))
        pg['n_conf_blocks'] = gb[gb.conf].groupby(['donor', 'gene']).size().reindex(
            pg.index, fill_value=0)
        f_ = HG.assign(indel=~HG.snv, snv_het=HG.snv, sing_snv=HG.snv & HG.singleton,
                       sing_any=HG.singleton, rare=HG.AC <= 5).groupby(['donor', 'gene']).agg(
            n_het_snv=('snv_het', 'sum'), n_het_indel=('indel', 'sum'),
            n_singleton_snv=('sing_snv', 'sum'), n_singleton_any=('sing_any', 'sum'),
            n_rare_ac_le5=('rare', 'sum'), min_pp=('pp', 'min'))
        feats.append(f_.join(pg, how='left').reset_index())
    F = pd.concat(feats, ignore_index=True)
    for k in ('n_rb_sites', 'n_conf_blocks', 'n_minority'):
        F[k] = F[k].fillna(0).astype(int)
    for k in ('any_conf', 'singleton_minority'):
        F[k] = F[k].fillna(False).astype(bool)
    S = pd.concat(sites_all, ignore_index=True)
    return F, S


# ===========================================================================
#  main
# ===========================================================================

def main():
    global LOGF
    OUT.mkdir(exist_ok=True); SCR.mkdir(exist_ok=True)
    LOGF = OUT / 'run.log'
    LOGF.write_text('')
    t_start = time.time()
    summary = dict(investigation='alignment-discordance', seed=SEED,
                   question='is the allelic weight-residual coupling carried by records '
                            'whose Salmon allelic ratio disagrees with phASER?')

    # ---------------- inputs and G1 ----------------------------------------
    import run_hapmixqtl_from_salmon as H
    d = np.load(INST / 'inputs_at_lead.npz', allow_pickle=True)
    genes = [str(g) for g in d['genes']]; donors = [str(x) for x in d['donors']]
    G = len(genes)
    a46, va46, s46 = d['a'].astype(float), d['va'].astype(float), d['s'].astype(float)
    cg = (CACHE / 'genes.txt').read_text().split()
    cs = (CACHE / 'samples.txt').read_text().split()
    rows = [cg.index(g) for g in genes]
    keepc = [cs.index(x) for x in donors]
    mm = {k: np.load(CACHE / f'{k}.npy', mmap_mode='r') for k in ('YL', 'YR', 'YT')}
    Y = {k: np.asarray(mm[k][rows])[:, keepc] for k in mm}
    with contextlib.redirect_stdout(io.StringIO()):
        Ac, _, Vac, _, _ = H.compute_summaries_from_gibbs(Y['YL'], Y['YR'], yT=Y['YT'])
    g1a = float(np.nanmax(np.abs(Ac - a46))); g1b = float(np.nanmax(np.abs(Vac - va46)))
    Z = np.load(CREACH / 'summaries_all_genes.npz', allow_pickle=True)
    tx_genes = [str(g) for g in Z['genes']]
    if [str(x) for x in Z['donors']] != donors or tx_genes != cg:
        raise SystemExit('G1 FAILED: coupling_reach summaries donor/gene order differ')
    A_tx, Va_tx = Z['A'].astype(float), Z['Va'].astype(float)
    r59 = [tx_genes.index(g) for g in d['all_genes']]
    g1c = max(float(np.nanmax(np.abs(A_tx[r59] - d['A_all']))),
              float(np.nanmax(np.abs(Va_tx[r59] - d['Va_all']))))
    summary['G1'] = dict(max_abs_a_cache_vs_npz=g1a, max_abs_va_cache_vs_npz=g1b,
                         max_abs_couplingreach_vs_npz_59=g1c)
    log('G1', summary['G1'])
    if g1a > 0 or g1b > 0 or g1c > 0:
        raise SystemExit('G1 FAILED')
    del Y, mm

    # ---------------- phASER, G3 --------------------------------------------
    P, ndup = load_phaser_all()
    AP_tx, QI_tx, KI_tx, CO_tx = phaser_arrays(P, tx_genes, donors)
    del P
    gi46 = [tx_genes.index(g) for g in genes]
    ap46, qi46, ki46, co46 = AP_tx[gi46], QI_tx[gi46], KI_tx[gi46], CO_tx[gi46]
    keep46 = np.isfinite(a46) & np.isfinite(va46) & (va46 > EPS)
    co46 = co46 & keep46
    with np.errstate(invalid='ignore'):
        dz46 = np.where(co46, (a46 - ap46) / np.sqrt(va46 + qi46), np.nan)
    m20 = co46 & (ki46 >= 20)
    sp_pool = float(sps.spearmanr(a46[m20], ap46[m20])[0])
    sp_gene = {genes[k]: (float(sps.spearmanr(a46[k][m20[k]], ap46[k][m20[k]])[0]),
                          int(m20[k].sum()))
               for k in range(G) if m20[k].sum() >= 5}
    # a gene with few pairs can be negative by chance (CNKSR1 has 5); the gate
    # is on genes with >= 20 pairs, where a negative rank correlation would
    # mean the A/L orientation is broken for that gene
    neg20 = {g: v for g, v in sp_gene.items() if v[1] >= 20 and v[0] < 0}
    imb = pd.read_csv(IMB / 'records.tsv', sep='\t')
    imb = imb[imb.comparable]
    gix = {g: i for i, g in enumerate(genes)}
    mine = dz46[imb.gene.map(gix).values, imb.j.values]
    g3_dz = float(np.max(np.abs(mine - imb.dz.values)))
    n_comp_46 = int(co46.sum())
    summary['G3'] = dict(pooled_spearman_a_ap_kinf20=sp_pool, n_pairs=int(m20.sum()),
                         n_genes_with_ge5_pairs=len(sp_gene),
                         n_genes_negative_any=int(sum(v[0] < 0 for v in sp_gene.values())),
                         genes_negative_any={g: v for g, v in sp_gene.items() if v[0] < 0},
                         n_genes_negative_with_ge20_pairs=len(neg20),
                         min_gene_spearman_ge20_pairs=float(min(v[0] for v in sp_gene.values()
                                                                if v[1] >= 20)),
                         max_abs_dz_vs_imbalance_downweighting=g3_dz,
                         n_comparable_46=n_comp_46, n_dz_compared=int(len(imb)),
                         phaser_duplicated_gene_donor_rows_dropped=ndup)
    log('G3', summary['G3'])
    if not (sp_pool > 0.2 and len(neg20) == 0 and g3_dz <= 1e-12
            and len(imb) == n_comp_46):
        raise SystemExit('G3 FAILED')

    # ---------------- (a) 46 genes -------------------------------------------
    W46 = np.where(keep46, 1 / np.where(keep46, va46, 1), 0.0)
    dec46 = np.stack([weight_deciles(W46[k], keep46[k]) for k in range(G)])
    rec = []
    for k, g in enumerate(genes):
        z2 = W46[k] * np.where(keep46[k], a46[k], 0) ** 2
        cw = np.where(keep46[k], W46[k] / W46[k][keep46[k]].mean() - 1, 0)
        for j in np.where(keep46[k])[0]:
            rec.append(dict(gene=g, donor=donors[j], j=j, a=a46[k, j], va=va46[k, j],
                            w=W46[k, j], z2=z2[j], c_j=cw[j] * z2[j] / z2.sum(),
                            w_decile=int(dec46[k, j]), s_lead=s46[k, j],
                            comparable=bool(co46[k, j]), ap=ap46[k, j], qinf=qi46[k, j],
                            kinf=ki46[k, j], dz=dz46[k, j]))
    REC = pd.DataFrame(rec)
    a46_rows = []
    for k, g in enumerate(genes):
        r = gene_weight_discordance(a46[k], va46[k], ap46[k], qi46[k], ki46[k], co46[k],
                                    keep46[k], 10)
        if r:
            a46_rows.append(dict(gene=g, **r))
    AG46 = pd.DataFrame(a46_rows)

    def sign_block(df):
        o = {}
        for c in ('sp_w_absdz', 'sp_w_absdiff', 'psp_w_absdz_given_kinf'):
            x = df[c].dropna()
            o[c] = dict(n_genes=int(len(x)), n_positive=int((x > 0).sum()),
                        sign_test_p=binom_sign((x > 0).sum(), len(x)),
                        median=float(x.median()))
        return o

    def decile_table(R):
        c = R[R.comparable]
        out = []
        for dd, s in c.groupby('w_decile'):
            e = (s.a - s.ap) ** 2
            out.append(dict(w_decile=int(dd), n=len(s),
                            frac_absdz_gt3=float((s.dz.abs() > 3).mean()),
                            frac_absdz_gt4=float((s.dz.abs() > 4).mean()),
                            frac_absdz_gt5=float((s.dz.abs() > 5).mean()),
                            median_absdz=float(s.dz.abs().median()),
                            mean_sq_diff=float(e.mean()), mean_va=float(s.va.mean()),
                            mean_qinf=float(s.qinf.mean()),
                            mean_undeclared=float((e - s.va - s.qinf).mean()),
                            mean_undeclared_absdz_le5=float(
                                (e - s.va - s.qinf)[s.dz.abs() <= 5].mean()),
                            mean_ratio_sq_diff_over_declared=float((e / (s.va + s.qinf)).mean()),
                            median_z2=float(s.z2.median())))
        return pd.DataFrame(out)

    def floor_fit(R, M_gene=None, gene_codes=None):
        """(a-ap)^2 = alpha (va + qinf) + tau^2, pooled least squares; gene-clustered
        bootstrap through per-gene sufficient statistics."""
        c = R[R.comparable]
        x = (c.va + c.qinf).values; y = ((c.a - c.ap) ** 2).values
        codes = pd.factorize(c.gene)[0]
        Gn = codes.max() + 1
        suff = np.zeros((Gn, 5))
        for i, v in enumerate((np.ones_like(x), x, y, x * x, x * y)):
            suff[:, i] = np.bincount(codes, weights=v, minlength=Gn)

        def solve(s):
            n, sx, sy, sxx, sxy = s[..., 0], s[..., 1], s[..., 2], s[..., 3], s[..., 4]
            al = (n * sxy - sx * sy) / (n * sxx - sx ** 2)
            return al, (sy - al * sx) / n
        al, tau2 = solve(suff.sum(0))
        rng = np.random.default_rng(SS_BOOT.spawn(1)[0])
        M = mult_matrix(rng, Gn, 500)
        bs = M @ suff
        alb, taub = solve(bs)
        return dict(alpha=float(al), tau2=float(tau2), alpha_ci=ci(alb), tau2_ci=ci(taub),
                    n=int(len(c)), n_genes=int(Gn))

    DT46 = decile_table(REC)
    summary['a_46'] = dict(per_gene_sign_tests=sign_block(AG46),
                           floor_fit_all=floor_fit(REC),
                           floor_fit_absdz_le5=floor_fit(REC[~(REC.dz.abs() > 5)]),
                           n_flagged={f'{t:g}': int((REC.dz.abs() > t).sum()) for t in THR},
                           n_genes_flagged={f'{t:g}': int(REC[REC.dz.abs() > t].gene.nunique())
                                            for t in THR},
                           dz_abs_quantiles={str(q): float(REC.dz.abs().quantile(q))
                                             for q in (.5, .9, .99)},
                           frac_absdz_gt3_expected_if_N01=float(2 * sps.norm.sf(3)))
    log('(a) 46 genes', json.dumps(summary['a_46']['per_gene_sign_tests']))

    # u/d split, 46 genes
    rngF = np.random.default_rng(SS_FLOOR)
    ud = [ud_split(a46[k], va46[k], ap46[k], co46[k], keep46[k], rngF, N_FLOOR)
          for k in range(G)]
    ud20 = [ud_split(a46[k], va46[k], ap46[k], co46[k] & (ki46[k] >= 20), keep46[k])
            for k in range(G)]

    def ud_summary(U, names):
        tot = {k: float(sum(x[k] for x in U)) for k in ('R_minus_1', 'u', 'd', 'noncomp')}
        o = dict(sum_R_minus_1=tot['R_minus_1'], sum_u=tot['u'], sum_d=tot['d'],
                 sum_noncomp=tot['noncomp'])
        if 'floor' in U[0]:
            F = sum(x['floor'] for x in U)
            o['floor_sd'] = dict(u=float(F[:, 0].std(ddof=1)), d=float(F[:, 1].std(ddof=1)),
                                 noncomp=float(F[:, 2].std(ddof=1)))
            o['floor_q975'] = dict(u=float(np.quantile(F[:, 0], .975)),
                                   d=float(np.quantile(F[:, 1], .975)))
        return o
    summary['a_46']['coupling_split_u_d'] = ud_summary(ud, genes)
    summary['a_46']['coupling_split_u_d_kinf20'] = ud_summary(ud20, genes)
    log('(a) u/d split 46', summary['a_46']['coupling_split_u_d'])

    # ---------------- (b)(c)(e) permutation arms, 46 genes, G2 ---------------
    rng_p = np.random.RandomState(SEED)
    PERMS = np.array([rng_p.permutation(N) for _ in range(N_PERM)])
    INV = np.argsort(PERMS, axis=1)
    NL = pd.read_csv(INST / 'null_long.tsv.gz', sep='\t')
    NL = NL.sort_values(['gene', 'perm'])
    rngM = np.random.default_rng(SS_MODEL); rngD = np.random.default_rng(SS_DEC)
    rngW = np.random.default_rng(SS_CTRLW); rngZ = np.random.default_rng(SS_CTRLZ)
    rngR = np.random.default_rng(SS_REPLC)
    nA = len(ALPHAS)
    arm_cnt, arm_nval, arm_setrates, comb_cnt, comb_nval, Rtab = {}, {}, {}, {}, {}, []
    g2 = dict(max_rel=0.0, max_abs_small=0.0)
    gate_counts = np.zeros((2, nA), int); inst_counts = np.zeros((2, nA), int)
    match_quality = {f'{t:g}': [] for t in THR}
    borrowed = {f'{t:g}': 0 for t in THR}

    def put(name, k, cnt, nval, ccnt=None, cnval=None):
        arm_cnt.setdefault(name, np.zeros((G, nA)))[k] = cnt.mean(0)
        arm_nval.setdefault(name, np.zeros(G))[k] = nval.mean()
        arm_setrates.setdefault(name, [None] * G)[k] = (cnt, nval)
        if ccnt is not None:
            comb_cnt.setdefault(name, np.zeros((G, nA)))[k] = ccnt.mean(0)
            comb_nval.setdefault(name, np.zeros(G))[k] = cnval.mean()

    for k, g in enumerate(genes):
        keep, w = keep46[k], W46[k]
        a0 = np.where(keep, a46[k], 0.0)
        Sg = s46[k][INV]; Sg2 = Sg ** 2
        nl = NL[NL.gene == g]
        if len(nl) != N_PERM or not (nl.perm.values == np.arange(N_PERM)).all():
            raise SystemExit(f'G2 FAILED: {g} null_long rows')
        bt, set2, doft = nl.bt.values, nl['set'].values ** 2, int(nl.doft.iloc[0])

        def run(name, Aset, Mset, comb=False):
            st = sets_stats(Aset, Mset, w, Sg, Sg2)
            cnt, nval = crit_counts(st['t2'], st['valid'], st['dof'], ALPHAS)
            cc = cn = None
            if comb:
                cc, cn = combined_counts(st, bt, set2, doft, ALPHAS)
            put(name, k, cnt, nval, cc, cn)
            return st, R_sets(Aset, Mset, w)

        # REAL + gate
        st, Rr = run('REAL', a0[None, :], keep[None, :], comb=True)
        t2 = st['t2'][0]; ref = nl.t2_a.values
        big = np.abs(ref) >= 1e-3
        g2['max_rel'] = max(g2['max_rel'], float(np.max(np.abs(t2[big] - ref[big]) / np.abs(ref[big]))))
        if (~big).any():
            g2['max_abs_small'] = max(g2['max_abs_small'], float(np.max(np.abs(t2[~big] - ref[~big]))))
        pa = sps.f.sf(t2, 1, st['dof'][0])
        with np.errstate(divide='ignore', invalid='ignore'):
            ba = st['num'][0] / st['den'][0]
            sea2 = st['q'][0] / (st['den'][0] ** 2 * st['dof'][0])
            prec = 1 / sea2 + 1 / set2
            bb = (ba / sea2 + bt / set2) / prec
        pb = sps.f.sf(bb ** 2 * prec, 1, min(st['dof'][0], doft))
        for i, al in enumerate(ALPHAS):
            gate_counts[0, i] += int((pa < al).sum()); inst_counts[0, i] += int((nl.p_a < al).sum())
            gate_counts[1, i] += int((pb < al).sum()); inst_counts[1, i] += int((nl.p_b < al).sum())
        R_row = dict(gene=g, R_REAL=float(Rr[0]))
        # MODEL and DECOUPLE
        Am = np.where(keep[None, :], rngM.standard_normal((K_SET, N)) * np.sqrt(np.where(keep, va46[k], 0)), 0)
        run('MODEL', Am, np.repeat(keep[None, :], K_SET, 0))
        Ad, Md = dec_sets(a0, keep, w, rngD, K_SET)
        run('DEC', Ad, Md)
        cnt_dec = arm_setrates['DEC'][k]
        conc = co46[k] & (np.abs(np.nan_to_num(dz46[k], nan=np.inf)) <= THR[0])
        zabs = np.sqrt(w) * np.abs(a0)
        for t in THR:
            T = f'{t:g}'
            fl = np.where(co46[k] & (np.abs(np.nan_to_num(dz46[k])) > t))[0]
            R_row[f'nflag_{T}'] = len(fl)
            if len(fl) == 0:
                for nm in ('EXCL', 'REPL', 'CTRLW', 'CTRLZ', 'REPLC'):
                    put(f'{nm}_{T}', k, *arm_setrates['REAL'][k],
                        comb_cnt['REAL'][k][None, :], np.array([comb_nval['REAL'][k]]))
                for nm in ('DEC_EXCL', 'DEC_REPL', 'DEC_CTRLW', 'DEC_CTRLZ', 'DEC_REPLC'):
                    put(f'{nm}_{T}', k, *cnt_dec)
                for nm in ('EXCL', 'REPL', 'CTRLW', 'CTRLZ', 'REPLC'):
                    R_row[f'R_{nm}_{T}'] = R_row['R_REAL']
                continue
            m = keep.copy(); m[fl] = False
            _, Rx = run(f'EXCL_{T}', np.where(m, a0, 0)[None, :], m[None, :], comb=True)
            R_row[f'R_EXCL_{T}'] = float(Rx[0])
            run(f'DEC_EXCL_{T}', *dec_sets(a0, m, w, rngD, K_SET))
            ar = a0.copy(); ar[fl] = ap46[k][fl]
            _, Rx = run(f'REPL_{T}', ar[None, :], keep[None, :], comb=True)
            R_row[f'R_REPL_{T}'] = float(Rx[0])
            run(f'DEC_REPL_{T}', *dec_sets(ar, keep, w, rngD, K_SET))
            for nm, rng_, za in (('CTRLW', rngW, None), ('CTRLZ', rngZ, zabs)):
                As, Ms, Ds, DMs = [], [], [], []
                for _ in range(K_SET):
                    ch, bo, zr = pick_controls(dec46[k], fl, conc, rng_, za)
                    if ch is None:
                        raise SystemExit(f'control pool exhausted: {g} {T}')
                    if nm == 'CTRLW':
                        borrowed[T] += bo
                    else:
                        match_quality[T] += zr
                    mc = keep.copy(); mc[ch] = False
                    As.append(np.where(mc, a0, 0)); Ms.append(mc)
                    a_, m_ = dec_sets(a0, mc, w, rngD, 1)
                    Ds.append(a_[0]); DMs.append(m_[0])
                _, Rc = run(f'{nm}_{T}', np.array(As), np.array(Ms), comb=True)
                R_row[f'R_{nm}_{T}'] = float(Rc.mean())
                run(f'DEC_{nm}_{T}', np.array(Ds), np.array(DMs))
            As, Ds = [], []
            for _ in range(K_SET):
                ch, _, _ = pick_controls(dec46[k], fl, conc, rngR)
                arc = a0.copy(); arc[ch] = ap46[k][ch]
                As.append(arc)
                Ds.append(dec_sets(arc, keep, w, rngD, 1)[0][0])
            Kk = np.repeat(keep[None, :], K_SET, 0)
            _, Rc = run(f'REPLC_{T}', np.array(As), Kk, comb=True)
            R_row[f'R_REPLC_{T}'] = float(Rc.mean())
            run(f'DEC_REPLC_{T}', np.array(Ds), Kk)
        Rtab.append(R_row)
    summary['G2'] = dict(**g2, allelic_counts=gate_counts[0].tolist(),
                         instrument_allelic_counts=inst_counts[0].tolist(),
                         combined_counts=gate_counts[1].tolist(),
                         instrument_combined_counts=inst_counts[1].tolist())
    log('G2', summary['G2'])
    if g2['max_rel'] > 1e-9 or g2['max_abs_small'] > 1e-9 or \
            not (gate_counts == inst_counts).all():
        raise SystemExit('G2 FAILED')
    crit_same = all(int(arm_cnt['REAL'][:, i].sum()) == gate_counts[0, i] for i in range(nA))
    if not crit_same:
        raise SystemExit('G2 FAILED: critical-value counts differ from sf counts')
    RT = pd.DataFrame(Rtab)

    # ---------------- pooled rates, fractions, bootstrap -------------------
    Mb = mult_matrix(np.random.default_rng(SS_BOOT), G, N_BOOT)
    no_calm = np.array([g != 'CALM2' for g in genes])
    Mb45 = mult_matrix(np.random.default_rng(SS_BOOT.spawn(2)[1]), int(no_calm.sum()), N_BOOT)

    def rate(name, i, sel=None, M=None, comb=False):
        c = (comb_cnt if comb else arm_cnt)[name][:, i]
        n = (comb_nval if comb else arm_nval)[name]
        if sel is not None:
            c, n = c[sel], n[sel]
        return pooled(c, n, M)

    def nsets(name):
        return max(x[0].shape[0] for x in arm_setrates[name])

    def mc_sd(name, i):
        """sd across sets of the pooled rate, and se of the set mean. Genes
        where the arm is a single deterministic set (nothing flagged) enter
        every set with that one value."""
        per = arm_setrates[name]
        K = nsets(name)
        if K == 1:
            return None, None
        pick = lambda x, s: x[s if x.shape[0] > 1 else 0]
        rates = np.array([sum(pick(per[g][0], s)[i] for g in range(G)) /
                          sum(pick(per[g][1], s) for g in range(G)) for s in range(K)])
        return float(rates.std(ddof=1)), float(rates.std(ddof=1) / np.sqrt(K))

    arms_rows = []
    for name in arm_cnt:
        for i, al in enumerate(ALPHAS):
            est, b = rate(name, i, M=Mb)
            sd, se = mc_sd(name, i)
            arms_rows.append(dict(arm=name, channel='allelic', alpha=al, rate=float(est),
                                  lo=ci(b)[0], hi=ci(b)[1], mc_sd=sd, mc_se_mean=se))
            if name in comb_cnt:
                est, b = rate(name, i, M=Mb, comb=True)
                arms_rows.append(dict(arm=name, channel='combined', alpha=al, rate=float(est),
                                      lo=ci(b)[0], hi=ci(b)[1], mc_sd=None, mc_se_mean=None))
    ARMS = pd.DataFrame(arms_rows)
    ARMS.to_csv(OUT / 'arms_46.tsv', sep='\t', index=False)

    def fractions(sel, M):
        out = {}
        for t in THR:
            T = f'{t:g}'
            out[T] = {}
            for i, al in enumerate(ALPHAS):
                r = {nm: rate(nm, i, sel, M) for nm in
                     ('REAL', 'DEC', 'MODEL', f'EXCL_{T}', f'DEC_EXCL_{T}',
                      f'CTRLW_{T}', f'DEC_CTRLW_{T}', f'CTRLZ_{T}', f'DEC_CTRLZ_{T}',
                      f'REPL_{T}', f'DEC_REPL_{T}', f'REPLC_{T}', f'DEC_REPLC_{T}')}
                pt = {k_: v[0] for k_, v in r.items()}; bs = {k_: v[1] for k_, v in r.items()}

                def f(x):
                    cf = x['REAL'] - x['DEC']
                    exf = x['REAL'] - x['MODEL']
                    o = dict(coupling_full=cf, excess_full=exf,
                             coupling_excl=x[f'EXCL_{T}'] - x[f'DEC_EXCL_{T}'],
                             coupling_ctrlw=x[f'CTRLW_{T}'] - x[f'DEC_CTRLW_{T}'],
                             coupling_ctrlz=x[f'CTRLZ_{T}'] - x[f'DEC_CTRLZ_{T}'],
                             coupling_repl=x[f'REPL_{T}'] - x[f'DEC_REPL_{T}'],
                             coupling_replc=x[f'REPLC_{T}'] - x[f'DEC_REPLC_{T}'],
                             drop_excl=x['REAL'] - x[f'EXCL_{T}'],
                             drop_ctrlw=x['REAL'] - x[f'CTRLW_{T}'],
                             drop_ctrlz=x['REAL'] - x[f'CTRLZ_{T}'],
                             drop_repl=x['REAL'] - x[f'REPL_{T}'],
                             drop_replc=x['REAL'] - x[f'REPLC_{T}'])
                    o['net_drop_excl_vs_ctrlw'] = o['drop_excl'] - o['drop_ctrlw']
                    o['net_drop_excl_vs_ctrlz'] = o['drop_excl'] - o['drop_ctrlz']
                    o['net_drop_repl_vs_replc'] = o['drop_repl'] - o['drop_replc']
                    o['net_coupling_removed_excl_vs_ctrlw'] = o['coupling_ctrlw'] - o['coupling_excl']
                    o['net_coupling_removed_excl_vs_ctrlz'] = o['coupling_ctrlz'] - o['coupling_excl']
                    o['net_coupling_removed_repl_vs_replc'] = o['coupling_replc'] - o['coupling_repl']
                    with np.errstate(divide='ignore', invalid='ignore'):
                        o['frac_coupling_removed_raw'] = 1 - o['coupling_excl'] / cf
                        o['frac_coupling_removed_net_ctrlw'] = o['net_coupling_removed_excl_vs_ctrlw'] / cf
                        o['frac_coupling_removed_net_ctrlz'] = o['net_coupling_removed_excl_vs_ctrlz'] / cf
                        o['frac_coupling_removed_repl_net'] = o['net_coupling_removed_repl_vs_replc'] / cf
                        o['frac_excess_removed_raw'] = o['drop_excl'] / exf
                        o['frac_excess_removed_net_ctrlw'] = o['net_drop_excl_vs_ctrlw'] / exf
                        o['frac_excess_removed_repl_net'] = o['net_drop_repl_vs_replc'] / exf
                    return o
                P_ = f(pt); B_ = f(bs)
                out[T][str(al)] = {k_: dict(est=float(P_[k_]), ci=ci(B_[k_])) for k_ in P_}
                out[T][str(al)]['rates'] = {k_: float(v) for k_, v in pt.items()}
        return out
    summary['b_46_fractions_all_genes'] = fractions(None, Mb)
    summary['b_46_fractions_without_CALM2'] = fractions(no_calm, Mb45)
    # Monte Carlo floors of the averaged arms in each difference
    summary['b_46_mc_se'] = {nm: {str(al): mc_sd(nm, i)[1] for i, al in enumerate(ALPHAS)}
                             for nm in arm_cnt if nsets(nm) > 1}
    summary['b_46_ctrlz_match_ratio_median'] = {T: (float(np.median(v)) if v else None)
                                                for T, v in match_quality.items()}
    summary['b_46_ctrlw_borrowed_from_adjacent_decile'] = borrowed
    # R summaries
    Rs = {}
    for col in [c for c in RT.columns if c.startswith('R_')]:
        Rs[col] = dict(sum_R_minus_1=float((RT[col] - 1).sum()),
                       mean_logR=float(np.log(RT[col]).mean()),
                       median_R=float(RT[col].median()))
    summary['b_46_R'] = Rs
    for t in THR:
        T = f'{t:g}'
        fl = REC[REC.dz.abs() > t]
        summary.setdefault('b_46_cshare', {})[T] = dict(
            sum_c_flagged=float(fl.c_j.sum()), sum_R_minus_1=float((RT.R_REAL - 1).sum()),
            share=float(fl.c_j.sum() / (RT.R_REAL - 1).sum()),
            share_of_positive_part=float(fl.c_j.clip(lower=0).sum() / REC.c_j.clip(lower=0).sum()),
            frac_records_flagged=float(len(fl) / len(REC)))
    RT.to_csv(OUT / 'per_gene_46.tsv', sep='\t', index=False)
    log('(b) 46 fractions t=3', json.dumps({al: {k_: v for k_, v in x.items() if k_ in (
        'coupling_full', 'coupling_excl', 'coupling_ctrlw', 'frac_coupling_removed_net_ctrlw',
        'frac_excess_removed_net_ctrlw')} for al, x in summary['b_46_fractions_all_genes']['3'].items()},
        default=str)[:3000])

    # ---------------- transcriptome: (a), u/d split, (b) --------------------
    keep_tx = np.isfinite(A_tx) & np.isfinite(Va_tx) & (Va_tx > EPS)
    n_a_tx = keep_tx.sum(1)
    CO_tx = CO_tx & keep_tx
    elig = n_a_tx >= MIN_NA
    with np.errstate(invalid='ignore', divide='ignore'):
        DZ_tx = np.where(CO_tx, (A_tx - AP_tx) / np.sqrt(Va_tx + QI_tx), np.nan)
    # comparability with the first round's 546,041-pair Spearman (n >= 30 reads each)
    summary['tx_counts'] = dict(n_genes_cache=len(tx_genes), n_genes_n_a_ge_20=int(elig.sum()),
                                n_comparable_records=int(CO_tx[elig].sum()),
                                n_records_admitted=int(keep_tx[elig].sum()))
    m30 = CO_tx & (KI_tx >= 30)
    summary['tx_counts']['spearman_a_ap_kinf30'] = float(sps.spearmanr(A_tx[m30], AP_tx[m30])[0])
    summary['tx_counts']['n_pairs_kinf30'] = int(m30.sum())
    summary['tx_counts']['note'] = ('first round: 546,041 pairs with >= 30 phASER reads and >= 30 '
                                    'Salmon allele-resolved reads, discordance denominator Gibbs-only '
                                    'variance + counting; here va (Gibbs + Salmon counting term) + '
                                    'Haldane counting variance of ap, so the counts and z differ')
    rows_tx, recs_tx = [], []
    ud_tx, ud_tx_floor = [], []
    rngF2 = np.random.default_rng(SS_FLOOR.spawn(1)[0])
    for j in np.where(elig)[0]:
        keep = keep_tx[j]
        r = gene_weight_discordance(A_tx[j], Va_tx[j], AP_tx[j], QI_tx[j], KI_tx[j],
                                    CO_tx[j], keep, MIN_COMP_TX)
        if r:
            rows_tx.append(dict(gene=tx_genes[j], **r))
        u = ud_split(A_tx[j], Va_tx[j], AP_tx[j], CO_tx[j], keep, rngF2, 20)
        ud_tx.append(u)
        w = 1 / Va_tx[j][keep]
        dec = weight_deciles(np.where(keep, 1 / np.where(keep, Va_tx[j], 1), 0), keep)
        cm = CO_tx[j][keep]
        z2 = w * A_tx[j][keep] ** 2
        recs_tx.append(pd.DataFrame(dict(gene=tx_genes[j], j=np.where(keep)[0], a=A_tx[j][keep],
                                         va=Va_tx[j][keep], w=w, z2=z2,
                                         c_j=(w / w.mean() - 1) * z2 / z2.sum(),
                                         w_decile=dec[keep], comparable=cm,
                                         ap=AP_tx[j][keep], qinf=QI_tx[j][keep],
                                         kinf=KI_tx[j][keep], dz=DZ_tx[j][keep])))
    AGT = pd.DataFrame(rows_tx)
    RECT = pd.concat(recs_tx, ignore_index=True)
    RECT['donor'] = np.array(donors)[RECT.j.values]
    DTT = decile_table(RECT)
    ud_t = ud_summary([{k: v for k, v in x.items() if k != 'floor'} for x in ud_tx], None)
    Ft = sum(x['floor'] for x in ud_tx)
    ud_t['floor_sd'] = dict(u=float(Ft[:, 0].std(ddof=1)), d=float(Ft[:, 1].std(ddof=1)),
                            noncomp=float(Ft[:, 2].std(ddof=1)), n_shuffles=20)
    summary['a_tx'] = dict(per_gene_sign_tests=sign_block(AGT),
                           floor_fit_all=floor_fit(RECT),
                           floor_fit_absdz_le5=floor_fit(RECT[~(RECT.dz.abs() > 5)]),
                           n_flagged={f'{t:g}': int((RECT.dz.abs() > t).sum()) for t in THR},
                           n_genes_flagged={f'{t:g}': int(RECT[RECT.dz.abs() > t].gene.nunique())
                                            for t in THR},
                           dz_abs_quantiles={str(q): float(RECT.dz.abs().quantile(q))
                                             for q in (.5, .9, .99, .999)},
                           coupling_split_u_d=ud_t)
    for t in THR:
        fl = RECT[RECT.dz.abs() > t]
        summary['a_tx'].setdefault('cshare', {})[f'{t:g}'] = dict(
            sum_c_flagged=float(fl.c_j.sum()), sum_R_minus_1=float(ud_t['sum_R_minus_1']),
            share=float(fl.c_j.sum() / ud_t['sum_R_minus_1']),
            share_of_positive_part=float(fl.c_j.clip(lower=0).sum() / RECT.c_j.clip(lower=0).sum()),
            frac_records_flagged=float(len(fl) / len(RECT)))
    log('(a) transcriptome', json.dumps(summary['a_tx']['per_gene_sign_tests']),
        json.dumps(ud_t))

    # (b) transcriptome at coupling_reach's synthetic variants
    Ssyn = np.zeros((N, len(MAF_LEVELS)))
    pos_rng = np.random.default_rng(SS_CR_SYN)
    for l, p in enumerate(MAF_LEVELS):
        kk = int(round(2 * p * (1 - p) * N))
        pos = pos_rng.permutation(N)[:kk]
        Ssyn[pos[:kk // 2], l] = 1.0
        Ssyn[pos[kk // 2:], l] = -1.0
    SgT = np.vstack([Ssyn[:, l][INV] for l in range(len(MAF_LEVELS))])   # (5P, N)
    n_chunks = (len(tx_genes) + CHUNK - 1) // CHUNK
    _TX.update(A=A_tx, Va=Va_tx, AP=AP_tx, QI=QI_tx, CO=CO_tx, Sg=SgT, Sg2=SgT ** 2,
               seeds=SS_TX.spawn(n_chunks))
    t0 = time.time()
    res = []
    with get_context('fork').Pool(N_WORK) as pool:
        for i, r in enumerate(pool.imap(tx_chunk, range(n_chunks))):
            res += r
            if (i + 1) % 30 == 0:
                log(f'  tx chunk {i + 1}/{n_chunks}')
    log(f'(b) transcriptome permutation arms in {time.time() - t0:.0f}s')
    TXR = pd.DataFrame(res)
    # G4
    bpg = pd.read_csv(CREACH / 'b_per_gene.tsv.gz', sep='\t').set_index('gene')
    g4 = 0.0; n4 = 0
    for r in TXR.itertuples():
        g = tx_genes[r.gi]
        if g not in bpg.index or not np.isfinite(bpg.loc[g, 'direct_0.05']):
            continue
        n4 += 1
        for i, al in enumerate(ALPHAS_TX):
            g4 = max(g4, abs(r.cnt_REAL_int[i] / r.nval_REAL_int - bpg.loc[g, f'direct_{al}']))
    summary['G4'] = dict(n_genes_compared=n4, n_genes_run=int(len(TXR)), max_abs_rate_diff=g4)
    log('G4', summary['G4'])
    if g4 > 1e-12 or n4 != len(TXR):
        raise SystemExit('G4 FAILED')
    Gt = len(TXR)
    Mt = mult_matrix(np.random.default_rng(SS_BOOT_TX), Gt, N_BOOT).astype(float)
    _cache = {}

    def arm_arrays(col):
        """(Gt, nA) counts and (Gt,) valid counts for an arm; genes where the
        arm was not run fall back to the arm it equals there (nothing
        flagged, or threshold not applied: EXCL = CTRLW = REAL and
        DEC_EXCL = DEC_CTRLW = DEC)."""
        if col in _cache:
            return _cache[col]
        pref = col if col in ('REAL', 'DEC') else col.rsplit('_', 1)[0]
        b = {'REAL': 'REAL', 'DEC': 'DEC', 'EXCL': 'REAL', 'CTRLW': 'REAL',
             'DEC_EXCL': 'DEC', 'DEC_CTRLW': 'DEC'}[pref]
        cb = np.stack(TXR[f'cnt_{b}'].values); nb = TXR[f'nval_{b}'].values.astype(float)
        if f'cnt_{col}' in TXR and col != b:
            have = TXR[f'cnt_{col}'].apply(lambda x: isinstance(x, np.ndarray)).values
            c = cb.copy(); n = nb.copy()
            c[have] = np.stack(TXR[f'cnt_{col}'].values[have])
            n[have] = TXR[f'nval_{col}'].values[have].astype(float)
        else:
            c, n = cb, nb
        _cache[col] = (c, n)
        return c, n

    def txv(col, i):
        c, n = arm_arrays(col)
        return c[:, i], n
    tx_out = {}
    for t in THR:
        T = f'{t:g}'
        tx_out[T] = {}
        for i, al in enumerate(ALPHAS_TX):
            vals = {nm: txv(nm, i) for nm in ('REAL', 'DEC', f'EXCL_{T}', f'DEC_EXCL_{T}',
                                              f'CTRLW_{T}', f'DEC_CTRLW_{T}')}
            pt = {k_: pooled(*v) for k_, v in vals.items()}
            bs = {k_: (Mt @ v[0]) / (Mt @ v[1]) for k_, v in vals.items()}

            def f(x):
                cf = x['REAL'] - x['DEC']
                o = dict(real=x['REAL'], dec=x['DEC'], excl=x[f'EXCL_{T}'], ctrlw=x[f'CTRLW_{T}'],
                         coupling_full=cf, excess_vs_nominal=x['REAL'] - al,
                         coupling_excl=x[f'EXCL_{T}'] - x[f'DEC_EXCL_{T}'],
                         coupling_ctrlw=x[f'CTRLW_{T}'] - x[f'DEC_CTRLW_{T}'],
                         drop_excl=x['REAL'] - x[f'EXCL_{T}'], drop_ctrlw=x['REAL'] - x[f'CTRLW_{T}'])
                o['net_drop'] = o['drop_excl'] - o['drop_ctrlw']
                o['net_coupling_removed'] = o['coupling_ctrlw'] - o['coupling_excl']
                with np.errstate(divide='ignore', invalid='ignore'):
                    o['frac_coupling_removed_net'] = o['net_coupling_removed'] / cf
                    o['frac_excess_removed_net'] = o['net_drop'] / o['excess_vs_nominal']
                return o
            P_, B_ = f(pt), f(bs)
            tx_out[T][str(al)] = {k_: dict(est=float(P_[k_]), ci=ci(B_[k_])) for k_ in P_}
        Rcol = [c for c in (f'R_EXCL_{T}', f'R_CTRLW_{T}') if c in TXR]
        R_ex = TXR[f'R_EXCL_{T}'].fillna(TXR.R_REAL) if f'R_EXCL_{T}' in TXR else TXR.R_REAL
        R_cw = TXR[f'R_CTRLW_{T}'].fillna(TXR.R_REAL) if f'R_CTRLW_{T}' in TXR else TXR.R_REAL
        tx_out[T]['R'] = dict(sum_R_minus_1_real=float((TXR.R_REAL - 1).sum()),
                              sum_R_minus_1_excl=float((R_ex - 1).sum()),
                              sum_R_minus_1_ctrlw=float((R_cw - 1).sum()),
                              n_genes_flagged=int((TXR[f'nflag_{T}'] > 0).sum()),
                              n_genes_ctrl_failed=int(TXR.get(f'ctrl_failed_{T}', pd.Series(dtype=bool)).fillna(False).sum()))
    summary['b_tx'] = tx_out
    log('(b) transcriptome t=3', json.dumps(tx_out['3'], default=str)[:2500])
    TXR.drop(columns=[c for c in TXR.columns if c.startswith('cnt_')]).assign(
        gene=[tx_genes[i] for i in TXR.gi]).to_csv(OUT / 'per_gene_tx.tsv.gz', sep='\t', index=False)

    # ---------------- (d) singletons and read-backed phase ------------------
    E = exon_table(set(tx_genes))
    summary['G5'] = g5_gate(E)
    log('G5', summary['G5'])
    if summary['G5']['frac_gt_identical'] < 0.999 or summary['G5']['maf01_min_minor_ac'] < 2:
        raise SystemExit('G5 FAILED')
    F, S = het_features(E, donors)
    S.to_csv(SCR / 'phase_sites.tsv.gz', sep='\t', index=False)
    S['ac_class'] = pd.cut(S.AC, [0, 1, 2, 5, 18, 10 ** 6], labels=['1', '2', '3-5', '6-18', '>18'])
    site_tab = S.groupby('ac_class', observed=True).agg(
        n_sites=('is_minority', 'size'), frac_in_conflicted_block=('blk_conf', 'mean'),
        frac_minority=('is_minority', 'mean'), median_pp=('pp', 'median')).reset_index()
    site_tab.to_csv(OUT / 'phase_minority_by_ac.tsv', sep='\t', index=False)
    summary['d_sites'] = site_tab.to_dict('records')
    log('(d) site-level', site_tab.to_string())

    def attach(R):
        X = R.merge(F, on=['donor', 'gene'], how='left')
        for k in ('n_het_snv', 'n_het_indel', 'n_singleton_snv', 'n_singleton_any',
                  'n_rare_ac_le5', 'n_rb_sites', 'n_conf_blocks', 'n_minority'):
            X[k] = X[k].fillna(0).astype(int)
        for k in ('any_conf', 'singleton_minority'):
            X[k] = X[k].fillna(False).astype(bool)
        X['has_singleton'] = X.n_singleton_any > 0
        X['has_indel'] = X.n_het_indel > 0
        X['stratum'] = np.searchsorted(NHET_BINS, X.n_het_snv.values, 'right') - 1
        return X
    REC = attach(REC); RECT = attach(RECT)

    def mh_block(X, expo, t=3.0, B=1000, seed_ss=None):
        c = X[X.comparable].copy()
        c['case'] = c.dz.abs() > t
        c['e'] = c[expo].astype(bool)
        codes, uniq = pd.factorize(c.gene)
        Sn = len(NHET_BINS) - 1
        cell = (c.e.astype(int) * 2 + c.case.astype(int)).values   # 3 exp-case,2 exp-non,1 unexp-case,0
        idx = codes * Sn + c.stratum.values
        T = np.zeros((len(uniq) * Sn, 4))
        for cc, col in ((3, 0), (2, 1), (1, 2), (0, 3)):
            T[:, col] = np.bincount(idx[cell == cc], minlength=len(uniq) * Sn)
        T = T.reshape(len(uniq), Sn, 4)
        tot = T.sum(0)
        est = float(mh_or(tot))
        M = mult_matrix(np.random.default_rng(seed_ss), len(uniq), B)
        bs = mh_or(np.einsum('bg,gsk->bsk', M.astype(float), T))
        raw = dict(p_case_exposed=float(c[c.e].case.mean()) if c.e.any() else None,
                   p_case_unexposed=float(c[~c.e].case.mean()),
                   n_exposed=int(c.e.sum()), n_cases=int(c.case.sum()),
                   n_cases_exposed=int((c.e & c.case).sum()))
        return dict(mh_or=est, ci=ci(bs), **raw)
    dsum = {}
    ssd = SS_SPARE.spawn(12)
    q = 0
    for lab, X in (('46', REC), ('tx', RECT)):
        dsum[lab] = {}
        for expo in ('has_singleton', 'any_conf', 'singleton_minority', 'has_indel'):
            dsum[lab][expo] = mh_block(X, expo, 3.0, 1000 if lab == '46' else 500, ssd[q]); q += 1
    summary['d_records'] = dsum
    log('(d) records', json.dumps(dsum, default=float))
    # flagged records of the 46 genes against decile-matched concordant ones
    fl = REC[REC.dz.abs() > 3]
    cc = REC[REC.comparable & (REC.dz.abs() <= 3)]
    cmp_rows = {}
    for lab, X in (('flagged_absdz_gt3', fl), ('concordant', cc)):
        cmp_rows[lab] = dict(n=len(X), frac_has_singleton=float(X.has_singleton.mean()),
                             frac_any_conf=float(X.any_conf.mean()),
                             frac_has_indel=float(X.has_indel.mean()),
                             median_n_het_snv=float(X.n_het_snv.median()),
                             median_w_decile=float(X.w_decile.median()),
                             median_abs_a=float(X.a.abs().median()),
                             median_abs_ap=float(X.ap.abs().median()),
                             frac_salmon_more_extreme=float((X.a.abs() > X.ap.abs()).mean()))
    summary['d_46_flagged_vs_concordant'] = cmp_rows
    REC.to_csv(OUT / 'records_46.tsv', sep='\t', index=False)
    RECT[RECT.comparable].to_csv(OUT / 'records_tx_comparable.tsv.gz', sep='\t', index=False)
    DT46.to_csv(OUT / 'weight_decile_46.tsv', sep='\t', index=False)
    DTT.to_csv(OUT / 'weight_decile_tx.tsv', sep='\t', index=False)
    AG46.to_csv(OUT / 'per_gene_weight_discordance_46.tsv', sep='\t', index=False)
    AGT.to_csv(OUT / 'per_gene_weight_discordance_tx.tsv.gz', sep='\t', index=False)

    # ---------------- figures ----------------------------------------------
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figs = []
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for lab, DT in (('46 genes', DT46), ('transcriptome', DTT)):
        ax[0].plot(DT.w_decile, DT.frac_absdz_gt3, 'o-', label=lab)
        ax[1].plot(DT.w_decile, DT.mean_undeclared_absdz_le5, 'o-', label=lab)
    ax[0].axhline(2 * sps.norm.sf(3), color='grey', ls=':', label='N(0,1) expectation')
    ax[0].set_xlabel('within-gene weight decile (0 = lowest w = 1/va)')
    ax[0].set_ylabel('fraction |dz| > 3'); ax[0].legend()
    ax[1].axhline(0, color='grey', ls=':')
    ax[1].set_xlabel('within-gene weight decile')
    ax[1].set_ylabel('mean (a-ap)^2 - va - qinf, |dz| <= 5 (log^2)')
    fig.tight_layout(); p = OUT / 'fig_discordance_by_weight.png'; fig.savefig(p, dpi=120)
    figs.append(str(p)); plt.close(fig)
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    show = ['REAL', 'DEC', 'MODEL', 'EXCL_3', 'CTRLW_3', 'CTRLZ_3', 'REPL_3', 'REPLC_3',
            'EXCL_5', 'CTRLW_5']
    for i, al in enumerate(ALPHAS):
        sub = ARMS[(ARMS.channel == 'allelic') & (ARMS.alpha == al)].set_index('arm').loc[show]
        axs[i].errorbar(range(len(show)), sub.rate, yerr=[sub.rate - sub.lo, sub.hi - sub.rate],
                        fmt='o')
        axs[i].axhline(al, color='grey', ls=':')
        axs[i].set_xticks(range(len(show))); axs[i].set_xticklabels(show, rotation=60)
        axs[i].set_title(f'allelic, alpha = {al}')
    fig.tight_layout(); p = OUT / 'fig_arms_46.png'; fig.savefig(p, dpi=120)
    figs.append(str(p)); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(site_tab.ac_class.astype(str), site_tab.frac_minority)
    ax.set_xlabel('cohort allele count of the heterozygous SNV (of 184)')
    ax.set_ylabel('fraction phased against read-backed phase')
    fig.tight_layout(); p = OUT / 'fig_phase_minority_by_ac.png'; fig.savefig(p, dpi=120)
    figs.append(str(p)); plt.close(fig)
    summary['figures'] = figs
    summary['runtime_s'] = time.time() - t_start
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2, default=float))
    log(f'wrote {OUT} in {summary["runtime_s"]:.0f}s')


if __name__ == '__main__':
    main()

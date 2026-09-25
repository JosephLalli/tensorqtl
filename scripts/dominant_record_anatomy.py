"""Dominant records in the allelic channel's permuted null: what they are, how
much of each tail tier they carry, and whether they are biology or artefact.

QUESTION. hapmixQTL's allelic nominal p is anticonservative on the shared
records-permutation null (nominal_p_null_instrument_20260925: 0.069 / 0.020 /
0.0057 at 0.05 / 0.01 / 0.001). One lead from the 2026-09-25 hypothesis
generation is that a handful of single donor records dominate the tail: under a
records permutation the allelic numerator is num = sum_j s_{sigma(j)} w_j a_j,
so its permutation variance is proportional to sum_j (w_j a_j)^2 = sum_j w_j z_j^2
(w = 1/va, z^2 = a^2/va). A record holding most of that sum decides the
statistic whenever it lands on a heterozygous position. This script asks

  (a) which records hold more than a quarter of sum(w z^2), and what they are
      (donor, a, va, posterior-mean allele-resolved counts mL/mR from the Gibbs
      cache, Gibbs variance against the Poisson 1/mL + 1/mR, genotype at the
      lead);
  (b) how much of the pooled tail at 0.05 / 0.01 / 0.001 they carry: each
      gene's top record excluded, only CALM2's excluded, a leave-one-record-out
      ranking over every admitted record, and the rejection rate conditional
      on the top record landing on a heterozygous position (s != 0) or not;
  (c) whether the strongest are biology or artefact: a cis scan of the gene's
      window for any variant at which the donor is heterozygous and whose
      allelic association across the OTHER donors predicts the donor's
      imbalance with the right sign; the donor's exonic heterozygous variants
      in the full cohort VCF and their carrier counts; an alignment-based
      allele-specific count of the same donor-gene pair (phASER on STAR
      alignments, the cohort matrix already on disk) against Salmon's; the
      haplotype-informative fragment counts in Salmon's own equivalence
      classes; and whether the donor is an outlier across the transcriptome;
  (d) whether such a record inflates the nominal p on OBSERVED data at
      variants where the donor is heterozygous, and whether the gene-level
      permutation p absorbs it.

DESIGN. Everything that is a "real" arm uses the instrument's exact inputs
(inputs_at_lead.npz) and exact permutation stream (RandomState(42),
2,000 x rng.permutation(92)). The allelic fit is the instrument's: weighted
least squares through the origin, w = 1/va, admitted records k = finite(a) &
va > 1e-12 (s = 0 included), sigma^2 = RSS / (n_a - 1), t^2 referred to
F(1, n_a - 1). It is vectorised over permutations as
t^2 = (n_a - 1) num^2 / (den S - num^2), num = sum s_pos w a, den = sum s_pos^2 w,
S = sum w a^2.

"EXCLUDING" A RECORD means it is not admitted: its weight is zero, n_a and the
dof fall by one, and the same 2,000 permutations are applied, so every
exclusion arm is paired with the real arm permutation by permutation. Held
fixed: genotype at the lead, the permutation stream, all other records.
Varied: which record(s) are admitted.

NOISE FLOORS. Pooled rates carry a gene-clustered percentile bootstrap (genes
resampled with replacement, 2,000 resamples). Differences between arms are
bootstrapped PAIRED (both arms on the same resample). A second floor for
"excluding the top record" is excluding a RANDOM admitted non-top record from
every gene (50 replicates): the change that dropping any one record makes.
A third floor is the SHIPPED MODEL itself: 20 record sets per gene generated
with z iid N(0, 1) at the gene's real va and s (a = z sqrt(va)), run through
the same permutations. It gives the model's pooled rate, the model's rate
conditional on its own top record landing on a heterozygote or not (a top
record is selected for large z, so on/off-landing rates differ from nominal
even under the model), the model's rate with its top record excluded, and how
many genes have a record with share > 0.25 under the model (also computed
transcriptome-wide, 20 draws at every gene's real weights).

ALIGNMENT-BASED ARBITER. phASER gene-level haplotype counts from STAR
alignments (prepped/phaser_matrix.gw_phased.txt.gz, per-SNP counts in
phaser_out/) count only reads over heterozygous SNPs, uniquely mapped; Salmon's
allelic contrast instead comes from an EM over a personalized diploid
transcriptome in which most fragments are haplotype-ambiguous. A record whose
Salmon a departs from its phASER log ratio far beyond the joint counting error,
at a rank the transcriptome-wide distribution of the same discordance puts in
its extreme tail, is flagged as a quantification artefact; one that agrees is
not. The cis scan's best variant is re-fitted on phASER counts to ask whether
any cis association replicates outside Salmon.

GATES (the script aborts on failure):
  1. the vectorised real arm reproduces null_long.tsv.gz t2_a for every
     (gene, perm) to relative 1e-9, and the pooled allelic rates at 0.05, 0.01,
     0.001 exactly;
  2. the npz a/va reproduce from the Gibbs cache (mapped by sample id, never
     position) through compute_summaries_from_gibbs(count_noise=True) to 1e-9;
  3. phASER haplotype A corresponds to Salmon's L haplotype: positive rank
     correlation between Salmon a and the phASER log ratio across donor-gene
     pairs with >= 20 phASER reads (checked, not assumed).

Master seed 42. The instrument stream is RandomState(42); everything else draws
from np.random.SeedSequence(42) children. Natural-log units throughout.
Outputs: /mnt/ssd/lalli/brainvar_hapmix_deploy/dominant_record_anatomy_20260925/
"""
import gzip
import json
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

D = Path('/mnt/ssd/lalli/brainvar_hapmix_deploy')
INST = D / 'nominal_p_null_instrument_20260925'
RUN = D / 'rasqual_default_mode_20260923'
CACHE = D / 'cache' / 'gibbs_56b63c3b37ed5df8'
PHASER_MAT = D / 'prepped' / 'phaser_matrix.gw_phased.txt.gz'
PHASER_OUT = D / 'phaser_out'
COHORT_VCF = D / 'vcf' / 'cohort92.NC.vcf.gz'
SALMON = Path('/mnt/ssd/lalli/nf_stage/RNA_reference_comparison_results/'
              'reference_comparison_results/bv2/personalized_T2T_NCBI110_'
              'pseudoalignment/expression_results/salmon_pseudocounts')
STAR_BAM_DIR = Path('/mnt/data/lalli/nf_stage/reference_comparison_results_RNA/'
                    'T2T_NCBI110/star_salmon')
OUT = D / 'dominant_record_anatomy_20260925'

SEED, EPS, KAPPA = 42, 1e-12, 0.5
N_PERM, N_BOOT, N_RANDOM_DROP = 2000, 2000, 50
ALPHAS = (0.05, 0.01, 0.001)
SHARE_MIN = 0.25
WIN = 1_000_000
SS = np.random.SeedSequence(SEED).spawn(6)   # 0 boot, 1 random drop, 2-5 spare


# ---------------------------------------------------------------------------
#  the allelic statistic, vectorised over permutations
# ---------------------------------------------------------------------------

def allelic_t2(a, va, s, P, drop=()):
    """t^2 and dof of the allelic channel for every permutation row of P.

    P[p, j] is the record placed at donor position j (P = rng.permutation(N)
    applied as a[P]), exactly as the instrument's a[prm]. `drop` lists record
    indices excluded from admission.
    """
    k = np.isfinite(a) & np.isfinite(va) & (va > EPS)
    if len(drop):
        k = k.copy(); k[list(drop)] = False
    w = np.where(k, 1.0 / np.where(k, va, 1.0), 0.0)
    wa = np.where(k, w * np.where(k, a, 0.0), 0.0)
    S = float(np.sum(wa * np.where(k, a, 0.0)))
    n = int(k.sum()); dof = n - 1
    num = wa[P] @ s
    den = w[P] @ (s * s)
    with np.errstate(divide='ignore', invalid='ignore'):
        t2 = dof * num ** 2 / (den * S - num ** 2)
    t2 = np.where(den > 0, t2, np.nan)
    return t2, dof


def pvals(t2, dof):
    return np.where(np.isfinite(t2), sps.f.sf(np.where(np.isfinite(t2), t2, 0), 1, dof), 1.0)


def shares(a, va):
    k = np.isfinite(a) & np.isfinite(va) & (va > EPS)
    w = np.where(k, 1 / np.where(k, va, 1), 0.0)
    z2 = np.where(k, w * np.where(k, a, 0) ** 2, 0.0)
    wz2 = w * z2
    return k, w, z2, wz2 / wz2.sum(), z2 / z2.sum()


def counts_by_gene(P_list):
    """[G, len(ALPHAS)] rejection counts from a list of per-gene p arrays."""
    return np.array([[(p < al).sum() for al in ALPHAS] for p in P_list])


def boot_rates(K, n, rng, K2=None):
    """Gene-clustered bootstrap of pooled rate(s); paired difference if K2."""
    G = K.shape[0]
    idx = rng.integers(0, G, size=(N_BOOT, G))
    tot = n[idx].sum(1)
    out = {}
    for ai, al in enumerate(ALPHAS):
        est = K[:, ai].sum() / n.sum()
        b = K[idx, ai].sum(1) / tot
        d = dict(rate=float(est), lo=float(np.quantile(b, .025)),
                 hi=float(np.quantile(b, .975)))
        if K2 is not None:
            est2 = K2[:, ai].sum() / n.sum()
            b2 = K2[idx, ai].sum(1) / tot
            dd = b2 - b
            d.update(arm_rate=float(est2), arm_lo=float(np.quantile(b2, .025)),
                     arm_hi=float(np.quantile(b2, .975)), diff=float(est2 - est),
                     diff_lo=float(np.quantile(dd, .025)),
                     diff_hi=float(np.quantile(dd, .975)),
                     diff_boot_sd=float(dd.std(ddof=1)))
        out[str(al)] = d
    return out


# ---------------------------------------------------------------------------
#  Gibbs cache, phASER, Salmon equivalence classes
# ---------------------------------------------------------------------------

def cache_rows(genes):
    names = open(CACHE / 'genes.txt').read().split()
    samples = open(CACHE / 'samples.txt').read().split()
    gi = {g: i for i, g in enumerate(names)}
    return names, samples, [gi[g] for g in genes]


def summaries(yl, yr):
    """compute_summaries_from_gibbs' allelic half (count_noise=True), plus the
    draw-only variance and the Poisson term separately."""
    ad = np.log(yl + KAPPA) - np.log(yr + KAPPA)
    A = ad.mean(2); Vg = ad.var(2)
    mL, mR = yl.mean(2), yr.mean(2)
    q = 1.0 / (mL + KAPPA) + 1.0 / (mR + KAPPA)
    no_cov = (mL + mR) <= 0
    Va = np.where(no_cov, 0.0, Vg + q)
    return A, Va, Vg, q, mL, mR


def read_phaser_matrix(genes_wanted=None):
    """Cohort phASER gene-level haplotype counts: {gene: (A[92], B[92])} in the
    matrix's donor order, plus that order."""
    out = {}
    with gzip.open(PHASER_MAT, 'rt') as fh:
        hdr = fh.readline().rstrip('\n').split('\t')
        donors = hdr[4:]
        for line in fh:
            f = line.rstrip('\n').split('\t')
            g = f[1]
            if genes_wanted is not None and g not in genes_wanted:
                continue
            ab = [x.split('|') for x in f[4:]]
            A = np.array([int(x[0]) for x in ab]); B = np.array([int(x[1]) for x in ab])
            if g in out:                       # duplicated gene names: keep the deeper row
                if A.sum() + B.sum() <= out[g][0].sum() + out[g][1].sum():
                    continue
            out[g] = (A, B)
    return out, donors


def eq_informative(args):
    """Haplotype-informative fragment counts for several genes in one donor's
    Salmon equivalence classes: fragments whose class holds only _L copies of
    the gene's transcripts, only _R copies, or both."""
    rna, gene_tx = args
    path = SALMON / rna / 'aux_info' / 'eq_classes.txt.gz'
    res = {}
    with gzip.open(path, 'rt') as f:
        nt = int(f.readline()); ne = int(f.readline())
        names = [f.readline().strip() for _ in range(nt)]
        idx = {n: i for i, n in enumerate(names)}
        sets = {}
        tx_owner = {}
        for g, txs in gene_tx.items():
            # PAIRED transcripts only (both copies in the index), exactly as
            # pair_haplotypes admits them: an unpaired _L copy is a homozygous
            # transcript whose _R duplicate the indexer dropped, not L evidence
            pr = [t for t in txs if t + '_L' in idx and t + '_R' in idx]
            L = {idx[t + '_L'] for t in pr}
            R = {idx[t + '_R'] for t in pr}
            sets[g] = (L, R)
            for i in L | R:
                tx_owner.setdefault(i, set()).add(g)
            res[g] = dict(L_only=0, R_only=0, both=0, n_tx_L=len(L), n_tx_R=len(R))
        for _ in range(ne):
            p = f.readline().split()
            k = int(p[0]); ts = [int(x) for x in p[1:1 + k]]
            hit = set()
            for t in ts:
                if t in tx_owner:
                    hit |= tx_owner[t]
            if not hit:
                continue
            cnt = int(p[-1]); tss = set(ts)
            for g in hit:
                L, R = sets[g]
                hl, hr = bool(tss & L), bool(tss & R)
                key = 'both' if (hl and hr) else ('L_only' if hl else 'R_only')
                res[g][key] += cnt
    return rna, res


def quant_rows(rna, txs):
    q = pd.read_csv(SALMON / rna / 'quant.sf', sep='\t', index_col=0)
    want = [t + h for t in txs for h in ('_L', '_R') if t + h in q.index]
    return q.loc[want, ['Length', 'NumReads']]


# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    OUT.mkdir(exist_ok=True)
    d = np.load(INST / 'inputs_at_lead.npz', allow_pickle=True)
    genes = list(d['genes']); donors = list(d['donors']); G, N = len(genes), len(donors)
    A, VA, S_ = d['a'], d['va'], d['s']
    null = pd.read_csv(INST / 'null_long.tsv.gz', sep='\t',
                       usecols=['gene', 'perm', 't2_a', 'dofa', 'p_a'])
    summary = dict(n_genes=G, n_donors=N, n_perm=N_PERM)

    # ---- the instrument's permutation stream -------------------------------
    rng = np.random.RandomState(SEED)
    P = np.stack([rng.permutation(N) for _ in range(N_PERM)])
    INV = np.argsort(P, axis=1)                   # INV[p, r] = position of record r

    # ---- GATE 1: the real arm ---------------------------------------------
    T2 = np.empty((G, N_PERM)); DOF = np.empty(G, int); PV = []
    for gi, g in enumerate(genes):
        t2, dof = allelic_t2(A[gi], VA[gi], S_[gi], P)
        T2[gi] = t2; DOF[gi] = dof; PV.append(pvals(t2, dof))
    ref = null.pivot(index='gene', columns='perm', values='t2_a').loc[genes].values
    refdof = null.groupby('gene').dofa.first().loc[genes].values
    rel = np.abs(T2 - ref) / np.maximum(np.abs(ref), 1e-300)
    big = np.abs(ref) > 1e-6
    gate1 = dict(max_rel_t2=float(rel.max()), max_rel_t2_where_t2_gt_1em6=float(rel[big].max()),
                 max_abs_t2=float(np.abs(T2 - ref).max()), dof_equal=bool((DOF == refdof).all()))
    base_counts = counts_by_gene(PV)
    n_g = np.full(G, N_PERM)
    ref_counts = np.array([[(null[null.gene == g].p_a < al).sum() for al in ALPHAS] for g in genes])
    gate1['pooled_rates'] = {str(al): float(base_counts[:, i].sum() / (G * N_PERM)) for i, al in enumerate(ALPHAS)}
    gate1['pooled_rates_instrument'] = {str(al): float(ref_counts[:, i].sum() / (G * N_PERM)) for i, al in enumerate(ALPHAS)}
    gate1['pooled_counts_equal'] = bool((base_counts.sum(0) == ref_counts.sum(0)).all())
    print('GATE 1', gate1, flush=True)
    if not (gate1['max_rel_t2_where_t2_gt_1em6'] <= 1e-9 and gate1['max_abs_t2'] <= 1e-9
            and gate1['dof_equal'] and gate1['pooled_counts_equal']):
        raise SystemExit('GATE 1 FAILED: vectorised allelic fit does not reproduce the instrument')
    summary['gate1'] = gate1

    # ---- GATE 2: Gibbs cache reproduces a / va ------------------------------
    cnames, csamples, crow = cache_rows(genes)
    keep = [csamples.index(s) for s in donors]          # by sample id, never position
    YL = np.load(CACHE / 'YL.npy', mmap_mode='r'); YR = np.load(CACHE / 'YR.npy', mmap_mode='r')
    yl = np.asarray(YL[crow])[:, keep]; yr = np.asarray(YR[crow])[:, keep]
    Ac, Vac, Vg, Q, mL, mR = summaries(yl, yr)
    gate2 = dict(max_abs_a=float(np.nanmax(np.abs(Ac - A))), max_abs_va=float(np.nanmax(np.abs(Vac - VA))))
    print('GATE 2', gate2, flush=True)
    if gate2['max_abs_a'] > 1e-9 or gate2['max_abs_va'] > 1e-9:
        raise SystemExit('GATE 2 FAILED: Gibbs cache mapping does not reproduce npz a/va')
    summary['gate2'] = gate2
    del yl, yr

    # ---- (a) per-record anatomy ---------------------------------------------
    meta = json.load(open(D / 'gibbs_influence_audit_20260915' / 'input_source_metadata.json'))
    rna_of = {e['sample']: e['rna'] for e in meta}
    rows = []; top = {}
    for gi, g in enumerate(genes):
        k, w, z2, sh_wz2, sh_z2 = shares(A[gi], VA[gi])
        wrank = sps.rankdata(-np.where(k, w, -np.inf), method='min')
        top[g] = int(np.argmax(sh_wz2))
        for j in np.where(k)[0]:
            pois = 1 / mL[gi, j] + 1 / mR[gi, j] if mL[gi, j] > 0 and mR[gi, j] > 0 else np.inf
            rows.append(dict(gene=g, donor=donors[j], pos=int(j), a=A[gi, j], va=VA[gi, j],
                             va_gibbs_only=Vg[gi, j], q=Q[gi, j], mL=mL[gi, j], mR=mR[gi, j],
                             gibbs_over_poisson=Vg[gi, j] / pois,
                             z=A[gi, j] / np.sqrt(VA[gi, j]), w_rank=int(wrank[j]),
                             share_wz2=sh_wz2[j], share_z2=sh_z2[j],
                             s_lead=int(S_[gi, j]), dosage_lead=float(d['g'][gi, j] * 2),
                             is_top=(j == top[g]), n_a=int(k.sum())))
    R = pd.DataFrame(rows)
    R.to_csv(OUT / 'records.tsv', sep='\t', index=False)
    dom = R[R.share_wz2 > SHARE_MIN].sort_values('share_wz2', ascending=False).copy()
    print(f'{len(dom)} records with share of sum(w z^2) > {SHARE_MIN}', flush=True)

    # ---- phASER cohort matrix: orientation gate and per-record comparison ---
    phm, ph_donors = read_phaser_matrix()
    ph_pos = [ph_donors.index(s) for s in donors]
    def ph_ratio(g):
        if g not in phm:
            return None, None
        a_, b_ = phm[g][0][ph_pos], phm[g][1][ph_pos]
        return np.log((a_ + KAPPA) / (b_ + KAPPA)), a_ + b_
    xs, ys = [], []
    for gi, g in enumerate(genes):
        lr, n = ph_ratio(g)
        if lr is None:
            continue
        ok = (n >= 20) & (VA[gi] > EPS)
        xs += list(A[gi][ok]); ys += list(lr[ok])
    rho_or, p_or = sps.spearmanr(xs, ys)
    gate3 = dict(n_pairs=len(xs), spearman_salmon_a_vs_phaser=float(rho_or), p=float(p_or))
    print('GATE 3 (orientation, 46 genes)', gate3, flush=True)
    if not rho_or > 0.2:
        raise SystemExit('GATE 3 FAILED: phASER haplotype A does not track Salmon L')
    summary['gate3'] = gate3

    # ---- transcriptome pass: donor outlier status, phASER discordance -------
    print('transcriptome pass over the Gibbs cache ...', flush=True)
    ncg = len(cnames)
    keep_arr = np.array(keep)
    top_donor = np.full(ncg, -1); top_share = np.zeros(ncg); n_a_all = np.zeros(ncg, int)
    A_tx = np.full((ncg, N), np.nan, np.float32); Va_tx = np.zeros((ncg, N), np.float32)
    Vg_tx = np.zeros((ncg, N), np.float32); M_tx = np.zeros((ncg, N), np.float32)
    CH = 1500
    for c0 in range(0, ncg, CH):
        sl = slice(c0, min(ncg, c0 + CH))
        a_, va_, vg_, _q, ml_, mr_ = summaries(np.asarray(YL[sl])[:, keep_arr],
                                                np.asarray(YR[sl])[:, keep_arr])
        A_tx[sl] = a_; Va_tx[sl] = va_; Vg_tx[sl] = vg_; M_tx[sl] = ml_ + mr_
        k = va_ > EPS
        w = np.where(k, 1 / np.where(k, va_, 1), 0)
        wz2 = w * w * np.where(k, a_, 0) ** 2
        tot = wz2.sum(1)
        n_a_all[sl] = k.sum(1)
        with np.errstate(invalid='ignore', divide='ignore'):
            shr = wz2 / tot[:, None]
        top_donor[sl] = np.where(tot > 0, np.nanargmax(np.nan_to_num(shr, nan=-1), 1), -1)
        top_share[sl] = np.where(tot > 0, np.nanmax(np.nan_to_num(shr, nan=0), 1), 0)
    print(f'  done in {time.time() - t0:.0f}s', flush=True)
    elig = n_a_all >= 20
    dom_tx = elig & (top_share > SHARE_MIN)
    cnt = np.bincount(top_donor[dom_tx], minlength=N)
    donor_tx = pd.DataFrame(dict(donor=donors, n_genes_dominant=cnt))
    # phASER discordance, transcriptome-wide, per donor
    ci = {g: i for i, g in enumerate(cnames)}
    disc = []
    for g, (a_all, b_all) in phm.items():
        i = ci.get(g)
        if i is None:
            continue
        a_, b_ = a_all[ph_pos], b_all[ph_pos]
        n = a_ + b_
        ok = (n >= 30) & (Va_tx[i] > EPS) & (M_tx[i] >= 30)
        if not ok.any():
            continue
        lr = np.log((a_ + KAPPA) / (b_ + KAPPA))
        dd = A_tx[i] - lr
        zz = dd / np.sqrt(Vg_tx[i] + 1 / (a_ + KAPPA) + 1 / (b_ + KAPPA))
        for j in np.where(ok)[0]:
            disc.append((g, j, float(A_tx[i, j]), float(lr[j]), int(n[j]), float(dd[j]), float(zz[j])))
    disc = pd.DataFrame(disc, columns=['gene', 'pos', 'a_salmon', 'lr_phaser', 'n_phaser', 'diff', 'z'])
    rho_tx = sps.spearmanr(disc.a_salmon, disc.lr_phaser)[0]
    per_donor = disc.groupby('pos').agg(n_pairs=('z', 'size'),
                                        median_abs_diff=('diff', lambda x: float(np.median(np.abs(x)))),
                                        frac_absz_gt4=('z', lambda x: float((np.abs(x) > 4).mean())))
    donor_tx = donor_tx.join(per_donor, how='left')
    donor_tx.to_csv(OUT / 'donor_transcriptome.tsv', sep='\t', index_label='pos')
    summary['transcriptome'] = dict(
        n_genes_cache=int(ncg), n_genes_n_a_ge_20=int(elig.sum()),
        n_genes_dominant_record=int(dom_tx.sum()),
        frac_genes_dominant_record=float(dom_tx.sum() / elig.sum()),
        dominant_count_per_donor_median=float(np.median(cnt)),
        dominant_count_per_donor_max=int(cnt.max()),
        dominant_count_per_donor_expected_uniform=float(dom_tx.sum() / N),
        phaser_pairs=int(len(disc)), spearman_salmon_vs_phaser=float(rho_tx),
        abs_z_quantiles={str(qq): float(np.quantile(np.abs(disc.z), qq)) for qq in (.5, .9, .99, .999)},
        frac_absz_gt4_all=float((np.abs(disc.z) > 4).mean()))
    # MODEL FLOOR for dominance: z iid N(0, 1) at each gene's real weights.
    # A top share > 0.25 can arise under the shipped model when the weights
    # are concentrated; this is how often it does.
    mrng = np.random.default_rng(SS[2])
    Ve = Va_tx[elig].astype(float)
    ke = Ve > EPS
    we = np.where(ke, 1 / np.where(ke, Ve, 1), 0)
    frac_model = []
    for _ in range(20):
        zz = mrng.standard_normal(we.shape)
        wz2m = we * zz ** 2
        frac_model.append(float(((wz2m.max(1) / wz2m.sum(1)) > SHARE_MIN).mean()))
    summary['transcriptome']['frac_genes_dominant_record_model'] = dict(
        mean=float(np.mean(frac_model)), sd=float(np.std(frac_model, ddof=1)))
    print('transcriptome dominance: real', summary['transcriptome']['frac_genes_dominant_record'],
          'model', summary['transcriptome']['frac_genes_dominant_record_model'], flush=True)

    # ---- dominant-record table: counts, phASER, landing ---------------------
    dom_rows = []
    for r in dom.itertuples():
        gi = genes.index(r.gene)
        lr, n = ph_ratio(r.gene)
        row = dict(gene=r.gene, donor=r.donor, pos=r.pos, share_wz2=r.share_wz2,
                   share_z2=r.share_z2, w_rank=r.w_rank, n_a=r.n_a, a=r.a, va=r.va,
                   va_gibbs_only=r.va_gibbs_only, q=r.q, mL=r.mL, mR=r.mR,
                   gibbs_over_poisson=r.gibbs_over_poisson, z=r.z,
                   s_lead=r.s_lead, dosage_lead=r.dosage_lead,
                   frac_het_at_lead=float((S_[gi] != 0).mean()))
        if lr is not None:
            A_ph, B_ph = phm[r.gene][0][ph_pos][r.pos], phm[r.gene][1][ph_pos][r.pos]
            zz = (r.a - lr[r.pos]) / np.sqrt(r.va_gibbs_only + 1 / (A_ph + KAPPA) + 1 / (B_ph + KAPPA))
            row.update(phaser_A=int(A_ph), phaser_B=int(B_ph), phaser_lr=float(lr[r.pos]),
                       salmon_minus_phaser=float(r.a - lr[r.pos]), discord_z=float(zz),
                       discord_absz_percentile_transcriptome=float((np.abs(disc.z) < abs(zz)).mean()))
            # the same gene's other donors: does Salmon track phASER there?
            ok = (n >= 20) & (VA[gi] > EPS)
            ok[r.pos] = False
            if ok.sum() >= 5:
                row['gene_other_donors_spearman_salmon_vs_phaser'] = float(sps.spearmanr(A[gi][ok], lr[ok])[0])
                row['gene_other_donors_n'] = int(ok.sum())
        row['donor_n_genes_dominant_tx'] = int(cnt[r.pos])
        row['donor_rank_dominant_tx'] = int(sps.rankdata(-cnt, method='min')[r.pos])
        if r.pos in per_donor.index:
            row['donor_frac_absz_gt4_tx'] = float(per_donor.loc[r.pos, 'frac_absz_gt4'])
            row['donor_frac_absz_gt4_rank'] = int(sps.rankdata(-per_donor.frac_absz_gt4.values, method='min')[list(per_donor.index).index(r.pos)])
        dom_rows.append(row)
    DOM = pd.DataFrame(dom_rows)

    # ---- (b) tail decomposition ---------------------------------------------
    brng = np.random.default_rng(SS[0])
    base = boot_rates(base_counts, n_g, np.random.default_rng(SS[0]))
    arms = {}
    # each gene's top record excluded
    top_counts = []
    land_rows = []
    for gi, g in enumerate(genes):
        t2, dof = allelic_t2(A[gi], VA[gi], S_[gi], P, drop=(top[g],))
        top_counts.append([(pvals(t2, dof) < al).sum() for al in ALPHAS])
        pos = INV[:, top[g]]
        land = S_[gi][pos] != 0
        p0 = PV[gi]
        lr_ = dict(gene=g, top_donor=donors[top[g]], top_share=float(shares(A[gi], VA[gi])[3][top[g]]),
                   p_land_het=float(land.mean()), n_land=int(land.sum()))
        for al in ALPHAS:
            lr_[f'rej{al}_land'] = float((p0[land] < al).mean()) if land.any() else np.nan
            lr_[f'rej{al}_off'] = float((p0[~land] < al).mean())
            lr_[f'rej{al}_all'] = float((p0 < al).mean())
            # excess attributable to landing, relative to the gene's own off-landing rate
            lr_[f'excess{al}_from_landing'] = float(((p0[land] < al).sum() - land.sum() * (p0[~land] < al).mean()))
        land_rows.append(lr_)
    top_counts = np.array(top_counts)
    LAND = pd.DataFrame(land_rows)
    LAND.to_csv(OUT / 'landing.tsv', sep='\t', index=False)
    # pooled rejection conditional on the top record's landing, real data
    def pooled_land(Lt):
        out = {}
        n_on = Lt.n_land.sum(); n_off = (N_PERM - Lt.n_land).sum()
        for al in ALPHAS:
            on = float((Lt[f'rej{al}_land'].fillna(0) * Lt.n_land).sum() / n_on)
            off = float((Lt[f'rej{al}_off'] * (N_PERM - Lt.n_land)).sum() / n_off)
            out[str(al)] = dict(on_landing=on, off_landing=off)
        out['frac_perms_on'] = float(n_on / (n_on + n_off))
        return out
    summary['landing_pooled_real'] = pooled_land(LAND)
    # MODEL FLOOR for the landing decomposition and for dominance in these 46
    # genes: record sets generated under the shipped model (z iid N(0, 1) at
    # the real va, a = z sqrt(va), same s), permuted on the instrument stream.
    mrng46 = np.random.default_rng(SS[3])
    N_SIM = 20
    sim_land, sim_counts, sim_dom, sim_drop = [], [], [], []
    for rep in range(N_SIM):
        rows_ = []; cnts = []; dcnts = []; ndom = 0
        for gi, g in enumerate(genes):
            k = np.isfinite(A[gi]) & (VA[gi] > EPS)
            a_sim = np.where(k, mrng46.standard_normal(N) * np.sqrt(np.where(k, VA[gi], 1)), np.nan)
            sh = shares(a_sim, VA[gi])[3]
            tj = int(np.argmax(sh)); ndom += int(sh[tj] > SHARE_MIN)
            t2, dof = allelic_t2(a_sim, VA[gi], S_[gi], P)
            p0 = pvals(t2, dof)
            land = S_[gi][INV[:, tj]] != 0
            r_ = dict(n_land=int(land.sum()))
            for al in ALPHAS:
                r_[f'rej{al}_land'] = float((p0[land] < al).mean()) if land.any() else np.nan
                r_[f'rej{al}_off'] = float((p0[~land] < al).mean())
            rows_.append(r_)
            cnts.append([(p0 < al).sum() for al in ALPHAS])
            t2d, dofd = allelic_t2(a_sim, VA[gi], S_[gi], P, drop=(tj,))
            dcnts.append([(pvals(t2d, dofd) < al).sum() for al in ALPHAS])
        sim_land.append(pooled_land(pd.DataFrame(rows_)))
        sim_counts.append(np.array(cnts).sum(0) / (G * N_PERM))
        sim_dom.append(ndom)
        sim_drop.append(np.array(dcnts).sum(0) / (G * N_PERM))
    sim_counts = np.array(sim_counts); sim_drop = np.array(sim_drop)
    summary['model_floor_46genes'] = dict(
        n_sim=N_SIM,
        pooled_rate={str(al): dict(mean=float(sim_counts[:, i].mean()), sd=float(sim_counts[:, i].std(ddof=1)))
                     for i, al in enumerate(ALPHAS)},
        landing={str(al): dict(on_mean=float(np.mean([s_[str(al)]['on_landing'] for s_ in sim_land])),
                               on_sd=float(np.std([s_[str(al)]['on_landing'] for s_ in sim_land], ddof=1)),
                               off_mean=float(np.mean([s_[str(al)]['off_landing'] for s_ in sim_land])),
                               off_sd=float(np.std([s_[str(al)]['off_landing'] for s_ in sim_land], ddof=1)))
                 for al in ALPHAS},
        n_genes_with_share_gt_025=dict(mean=float(np.mean(sim_dom)), sd=float(np.std(sim_dom, ddof=1)),
                                       real=int(sum(1 for g in genes if shares(A[genes.index(g)], VA[genes.index(g)])[3].max() > SHARE_MIN))),
        top_excluded_rate={str(al): dict(mean=float(sim_drop[:, i].mean()), sd=float(sim_drop[:, i].std(ddof=1)))
                           for i, al in enumerate(ALPHAS)})
    # the real-minus-model excess split by landing: (on_real - on_model) *
    # frac_on + (off_real - off_model) * (1 - frac_on), at the REAL frac_on
    fo = summary['landing_pooled_real']['frac_perms_on']
    dec = {}
    for al in ALPHAS:
        lr_real = summary['landing_pooled_real'][str(al)]
        lm = summary['model_floor_46genes']['landing'][str(al)]
        on_part = (lr_real['on_landing'] - lm['on_mean']) * fo
        off_part = (lr_real['off_landing'] - lm['off_mean']) * (1 - fo)
        dec[str(al)] = dict(on_part=on_part, off_part=off_part, total=on_part + off_part,
                            real_minus_model_rate=float(base_counts[:, ALPHAS.index(al)].sum() / (G * N_PERM)
                                                        - summary['model_floor_46genes']['pooled_rate'][str(al)]['mean']),
                            frac_on=on_part / (on_part + off_part))
    summary['landing_decomposition_vs_model'] = dec
    print('landing real', summary['landing_pooled_real'], '\nmodel floor', summary['model_floor_46genes'], flush=True)
    arms['top_record_excluded'] = boot_rates(base_counts, n_g, np.random.default_rng(SS[0]), top_counts)
    # CALM2 only
    calm = genes.index('CALM2')
    c_counts = base_counts.copy()
    t2, dof = allelic_t2(A[calm], VA[calm], S_[calm], P, drop=(top['CALM2'],))
    c_counts[calm] = [(pvals(t2, dof) < al).sum() for al in ALPHAS]
    arms['calm2_top_excluded'] = boot_rates(base_counts, n_g, np.random.default_rng(SS[0]), c_counts)
    # only records with share > 0.25 excluded
    d_counts = base_counts.copy()
    for g in dom.gene.unique():
        gi = genes.index(g)
        drop = tuple(dom[dom.gene == g].pos.astype(int))
        t2, dof = allelic_t2(A[gi], VA[gi], S_[gi], P, drop=drop)
        d_counts[gi] = [(pvals(t2, dof) < al).sum() for al in ALPHAS]
    arms['share_gt_0.25_excluded'] = boot_rates(base_counts, n_g, np.random.default_rng(SS[0]), d_counts)
    # floor: a random non-top admitted record excluded from every gene
    rrng = np.random.default_rng(SS[1])
    rand_pooled = []
    for rep in range(N_RANDOM_DROP):
        rc = []
        for gi, g in enumerate(genes):
            k = np.isfinite(A[gi]) & (VA[gi] > EPS)
            cand = np.setdiff1d(np.where(k)[0], [top[g]])
            j = int(rrng.choice(cand))
            t2, dof = allelic_t2(A[gi], VA[gi], S_[gi], P, drop=(j,))
            rc.append([(pvals(t2, dof) < al).sum() for al in ALPHAS])
        rand_pooled.append(np.array(rc).sum(0) / (G * N_PERM))
    rand_pooled = np.array(rand_pooled)
    arms['random_nontop_excluded_floor'] = {
        str(al): dict(mean=float(rand_pooled[:, i].mean()), sd=float(rand_pooled[:, i].std(ddof=1)),
                      min=float(rand_pooled[:, i].min()), max=float(rand_pooled[:, i].max()))
        for i, al in enumerate(ALPHAS)}
    # leave-one-record-out over every admitted record
    loro = []
    for gi, g in enumerate(genes):
        k = np.isfinite(A[gi]) & (VA[gi] > EPS)
        sh = shares(A[gi], VA[gi])[3]
        for j in np.where(k)[0]:
            t2, dof = allelic_t2(A[gi], VA[gi], S_[gi], P, drop=(j,))
            c = [(pvals(t2, dof) < al).sum() for al in ALPHAS]
            loro.append(dict(gene=g, donor=donors[j], pos=int(j), share_wz2=float(sh[j]),
                             **{f'd_count{al}': int(c[i] - base_counts[gi, i]) for i, al in enumerate(ALPHAS)},
                             **{f'd_pooled{al}': float((c[i] - base_counts[gi, i]) / (G * N_PERM)) for i, al in enumerate(ALPHAS)}))
    LORO = pd.DataFrame(loro).sort_values('d_count0.001')
    LORO.to_csv(OUT / 'leave_one_record_out.tsv', sep='\t', index=False)
    summary['pooled_real'] = base
    summary['arms'] = arms
    exc = {}
    for i, al in enumerate(ALPHAS):
        tot_excess = base_counts[:, i].sum() - al * G * N_PERM
        land_excess = LAND[f'excess{al}_from_landing'].sum()
        exc[str(al)] = dict(pooled_excess_count=float(tot_excess),
                            top_landing_excess_count=float(land_excess),
                            calm2_landing_excess_count=float(LAND.set_index('gene').loc['CALM2', f'excess{al}_from_landing']),
                            frac_of_excess_from_top_landing=float(land_excess / tot_excess),
                            frac_of_excess_from_calm2_landing=float(LAND.set_index('gene').loc['CALM2', f'excess{al}_from_landing'] / tot_excess))
    summary['landing_decomposition'] = exc
    summary['loro_top10_at_0.001'] = LORO.head(10)[['gene', 'donor', 'share_wz2', 'd_count0.05', 'd_count0.01', 'd_count0.001']].to_dict('records')
    summary['loro_rank_spearman_share_vs_dcount0.001'] = float(sps.spearmanr(LORO.share_wz2, -LORO['d_count0.001'])[0])
    print(json.dumps(dict(real=base, arms=arms, landing=exc), indent=1), flush=True)

    # ---- (c) biology vs artefact -------------------------------------------
    import compare_mixqtl_replication as CM
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        I = CM.load_inputs(gene_list=str(D / 'genes_59_stratified_20260923.txt'),
                           regions=str(RUN / 'regions.bed'))
    if list(I['order']) != donors:
        raise SystemExit('genotype donor order differs from the npz')
    all_genes = list(d['all_genes'])
    if all_genes != list(I['genes']):
        raise SystemExit('gene order differs from the npz all_genes')
    vdf = I['vdf']; xL, xR, dos = I['xL'], I['xR'], I['dos']
    vpos, vch = vdf['pos'].values, vdf['chrom'].values.astype(str)
    af = dos.mean(1) / 2.0
    maf = np.minimum(af, 1 - af)
    exons = pd.read_csv(D / 'annot' / 'exons.tsv', sep='\t', header=None, index_col=0)
    tx2g = pd.read_csv(D / 'annot' / 'tx2gene.tsv', sep='\t', header=None, names=['tx', 'gene'])
    chr2nc = dict(pd.read_csv(D / 'vcf' / 'chr2nc.tsv', sep='\t', header=None).values)
    scan_rows, exonic_rows, obs_rows = [], [], []
    for r in DOM.itertuples():
        g, j = r.gene, int(r.pos)
        gi_all = all_genes.index(g)
        a_all, va_all = d['A_all'][gi_all], d['Va_all'][gi_all]
        gp = I['gp'].loc[g]
        inwin = np.where((vch == str(gp['chr'])) & (np.abs(vpos - int(gp['pos'])) <= WIN))[0]
        sv = (xL[inwin] - xR[inwin]).astype(float)           # [V, N]
        het = sv[:, j] != 0
        k = np.isfinite(a_all) & (va_all > EPS)
        ko = k.copy(); ko[j] = False
        w = np.where(ko, 1 / np.where(ko, va_all, 1), 0); wa = w * np.where(ko, a_all, 0)
        Sx = float((wa * np.where(ko, a_all, 0)).sum()); dof_o = int(ko.sum()) - 1
        num = sv @ wa; den = (sv * sv) @ w
        nhet_other = ((sv != 0) & ko[None, :]).sum(1)
        with np.errstate(divide='ignore', invalid='ignore'):
            beta = num / den
            t2 = dof_o * num ** 2 / (den * Sx - num ** 2)
            se = np.abs(beta) / np.sqrt(t2)
        p = np.where(den > 0, sps.f.sf(np.nan_to_num(t2), 1, dof_o), 1.0)
        pred = beta * sv[:, j]
        cons = np.sign(pred) == np.sign(r.a)
        usable = het & (nhet_other >= 5) & (den > 0)
        # the donor's record against the other donors' fit at this variant:
        # residual in units of that fit's sigma * sqrt(va_donor)
        with np.errstate(divide='ignore', invalid='ignore'):
            sig_o = np.sqrt((Sx - num ** 2 / den) / dof_o)
            rz = (r.a - pred) / (sig_o * np.sqrt(r.va))
        tab = pd.DataFrame(dict(variant=vdf.index[inwin], pos=vpos[inwin], maf=maf[inwin],
                                s_donor=sv[:, j], n_het_other=nhet_other, beta=beta, se=se,
                                p=p, pred_a_donor=pred, frac_of_a=pred / r.a, sign_consistent=cons,
                                donor_resid_z=rz, donor_z_unexplained=r.a / (sig_o * np.sqrt(r.va))))[usable]
        tab.insert(0, 'gene', g); tab.insert(1, 'donor', r.donor)
        scan_rows.append(tab)
        # the donor's exonic heterozygous variants in the FULL cohort VCF (all
        # frequencies, including those the analysis VCF drops), with carriers
        if g in exons.index:
            exg = exons.loc[[g]].iloc[0]
            st = [int(x) for x in str(exg[1]).split(",")]
            en = [int(x) for x in str(exg[2]).split(",")]
            nc = chr2nc[str(gp['chr'])]
            reg = ','.join(f'{nc}:{s_}-{e_}' for s_, e_ in zip(st, en))
            out = subprocess.run(['bcftools', 'query', '-r', reg, '-f', '%POS\t%REF\t%ALT[\t%GT]\n',
                                  str(COHORT_VCF)], capture_output=True, text=True, check=True).stdout
            hdr = subprocess.run(['bcftools', 'query', '-l', str(COHORT_VCF)], capture_output=True,
                                 text=True, check=True).stdout.split()
            di = hdr.index(r.donor)
            ac = None
            if (PHASER_OUT / f'{r.donor}.allelic_counts.txt').exists():
                ac = pd.read_csv(PHASER_OUT / f'{r.donor}.allelic_counts.txt', sep='\t',
                                 usecols=['contig', 'position', 'refCount', 'altCount'])
                ac = ac[ac.contig == nc].set_index('position')
            seen = set()
            for line in out.strip().split('\n'):
                if not line:
                    continue
                f = line.split('\t'); pos_ = int(f[0])
                if pos_ in seen:
                    continue
                gts = f[3:]
                gd = gts[di]
                if gd.replace('|', '/') not in ('0/1', '1/0'):
                    continue
                seen.add(pos_)
                carriers = sum(1 for x in gts if x not in ('0|0', '0/0', './.', '.|.', '.'))
                rc = ac.loc[pos_] if (ac is not None and pos_ in ac.index) else None
                if isinstance(rc, pd.DataFrame):
                    rc = rc.iloc[0]
                exonic_rows.append(dict(gene=g, donor=r.donor, pos=pos_, ref=f[1], alt=f[2], gt=gd,
                                        cohort_carriers=carriers,
                                        in_analysis_vcf=bool(((vch == str(gp['chr'])) & (vpos == pos_)).any()),
                                        star_phaser_ref=(int(rc.refCount) if rc is not None else None),
                                        star_phaser_alt=(int(rc.altCount) if rc is not None else None)))
        # ---- (d) observed data: nominal p at variants where donor het vs hom
        tested = inwin[maf[inwin] >= CM.MAF]
        svt = (xL[tested] - xR[tested]).astype(float)
        for label, drop in (('with_record', ()), ('record_excluded', (j,))):
            kk = k.copy()
            if drop:
                kk[j] = False
            ww = np.where(kk, 1 / np.where(kk, va_all, 1), 0); wwa = ww * np.where(kk, a_all, 0)
            SS_ = float((wwa * np.where(kk, a_all, 0)).sum()); df_ = int(kk.sum()) - 1
            nu = svt @ wwa; de = (svt * svt) @ ww
            with np.errstate(divide='ignore', invalid='ignore'):
                tt = np.where(de > 0, df_ * nu ** 2 / (de * SS_ - nu ** 2), 0)
            pp = sps.f.sf(tt, 1, df_)
            hetd = svt[:, j] != 0
            o = dict(gene=g, donor=r.donor, arm=label, n_tested=int(len(tested)),
                     n_donor_het=int(hetd.sum()))
            for al in ALPHAS:
                o[f'frac_p_lt_{al}_donor_het'] = float((pp[hetd] < al).mean()) if hetd.any() else np.nan
                o[f'frac_p_lt_{al}_donor_hom'] = float((pp[~hetd] < al).mean()) if (~hetd).any() else np.nan
            o['min_p_donor_het'] = float(pp[hetd].min()) if hetd.any() else np.nan
            o['min_p_donor_hom'] = float(pp[~hetd].min()) if (~hetd).any() else np.nan
            # gene-level: window-max t^2, empirical p on the instrument's stream
            obs_max = float(tt.max())
            PM = np.empty(N_PERM)
            wa_p = wwa[P]; w_p = ww[P]
            for c0 in range(0, N_PERM, 250):
                nu_p = svt @ wa_p[c0:c0 + 250].T
                de_p = (svt * svt) @ w_p[c0:c0 + 250].T
                with np.errstate(divide='ignore', invalid='ignore'):
                    tp = np.where(de_p > 0, df_ * nu_p ** 2 / (de_p * SS_ - nu_p ** 2), 0)
                PM[c0:c0 + 250] = tp.max(0)
            o['obs_window_max_t2'] = obs_max
            o['obs_window_max_nominal_p'] = float(sps.f.sf(obs_max, 1, df_))
            o['empirical_p_window_max'] = float((1 + (PM >= obs_max).sum()) / (N_PERM + 1))
            o['perm_window_max_t2_q95'] = float(np.quantile(PM, 0.95))
            obs_rows.append(o)
    # hapmixQTL's own OBSERVED lead (default mode, the deployed run): is the
    # dominant donor heterozygous there, and what does its record do to the
    # allelic nominal p at that lead?
    me = pd.read_csv(RUN / 'matched_effects.tsv', sep='\t').set_index('gene')
    lead_rows = []
    for r in DOM.itertuples():
        g, j = r.gene, int(r.pos)
        lv = me.loc[g, 'lead_h']
        if not isinstance(lv, str) or lv not in vdf.index:
            continue
        vi_ = int(np.where(vdf.index == lv)[0][0])
        sv_ = (xL[vi_] - xR[vi_]).astype(float)
        gi_all = all_genes.index(g)
        a_all, va_all = d['A_all'][gi_all], d['Va_all'][gi_all]
        o = dict(gene=g, donor=r.donor, lead_h=lv, stat_h_combined=float(me.loc[g, 'stat_h']),
                 donor_s_at_lead_h=int(sv_[j]), n_het_at_lead_h=int((sv_ != 0).sum()))
        for label, drop in (('with_record', ()), ('record_excluded', (j,))):
            t2l, dofl = allelic_t2(a_all, va_all, sv_, np.arange(N)[None, :], drop=drop)
            o[f'allelic_t2_{label}'] = float(t2l[0])
            o[f'allelic_p_{label}'] = float(pvals(t2l, dofl)[0])
        lead_rows.append(o)
    # the SHIPPED gene-level call: map_cis (default mode, records permutation,
    # combined statistic, tau_refit as in the deployed hapmix arm, 1,000
    # permutations, seed 42) on the gene's tested variants, with the dominant
    # record admitted and with its allelic variance zeroed (excluded from the
    # allelic channel only; its total-channel record is untouched).
    from tensorqtl.hapmixqtl import map_cis
    order = list(I['order'])
    mk = lambda M, g: pd.DataFrame(M[[all_genes.index(g)]], index=[g], columns=order)
    for o in lead_rows:
        g, j = o['gene'], donors.index(o['donor'])
        vsel = I['idx'][CM.gene_variant_index(I, g)]
        v1 = vdf.iloc[vsel]
        one = lambda M: pd.DataFrame(M[vsel], index=v1.index, columns=order)
        for label in ('with_record', 'record_excluded'):
            Va_use = d['Va_all'].copy()
            if label == 'record_excluded':
                Va_use[all_genes.index(g), j] = 0.0
            with contextlib.redirect_stdout(io.StringIO()):
                res = map_cis(one(dos), v1[['chrom', 'pos']], mk(d['A_all'], g), mk(d['T_all'], g),
                              mk(Va_use, g), mk(d['Vt_all'], g), I['gp'].loc[[g]][['chr', 'pos']],
                              xL_df=one(xL), xR_df=one(xR), window=WIN, nperm=1000, seed=SEED,
                              covariates_df=I['cov_df'], ase_covariates_df=None, tau_refit=True,
                              verbose=False, warn_monomorphic=False)
            o[f'map_cis_lead_{label}'] = str(res['variant_id'].iloc[0])
            o[f'map_cis_stat_{label}'] = float((res['slope'].iloc[0] / res['slope_se'].iloc[0]) ** 2)
            o[f'map_cis_pval_perm_{label}'] = float(res['pval_perm'].iloc[0])
        print('map_cis', {k: v for k, v in o.items() if k.startswith('map_cis') or k == 'gene'}, flush=True)
    LEADH = pd.DataFrame(lead_rows); LEADH.to_csv(OUT / 'observed_lead_h.tsv', sep='\t', index=False)
    summary['observed_lead_h'] = LEADH.to_dict('records')
    SCAN = pd.concat(scan_rows, ignore_index=True) if scan_rows else pd.DataFrame()
    SCAN.to_csv(OUT / 'cis_scan.tsv', sep='\t', index=False)
    EXO = pd.DataFrame(exonic_rows); EXO.to_csv(OUT / 'donor_exonic_hets.tsv', sep='\t', index=False)
    OBS = pd.DataFrame(obs_rows); OBS.to_csv(OUT / 'observed_het_vs_hom.tsv', sep='\t', index=False)
    # ---- Salmon equivalence classes: dominant genes, all 92 donors ----------
    dgenes = list(dict.fromkeys(DOM.gene))
    gene_tx = {g: list(tx2g[tx2g.gene == g].tx) for g in dgenes}
    rnas = [rna_of[s] for s in donors]
    missing = [r_ for r_ in rnas if not (SALMON / r_ / 'aux_info' / 'eq_classes.txt.gz').exists()]
    if missing:
        raise SystemExit(f'missing Salmon eq classes for {missing[:3]}')
    print(f'parsing equivalence classes for {len(rnas)} libraries x {len(dgenes)} genes ...', flush=True)
    with Pool(16) as pool:
        eqres = dict(pool.map(eq_informative, [(r_, gene_tx) for r_ in rnas]))
    eq_rows = []
    for j, s_ in enumerate(donors):
        for g in dgenes:
            e = eqres[rnas[j]][g]
            gi_all = all_genes.index(g)
            lr_ph, n_ph = ph_ratio(g)
            eq_rows.append(dict(gene=g, donor=s_, pos=j, **e,
                                eq_lr=np.log((e['L_only'] + KAPPA) / (e['R_only'] + KAPPA)),
                                eq_poisson=1 / (e['L_only'] + KAPPA) + 1 / (e['R_only'] + KAPPA),
                                a=float(d['A_all'][gi_all, j]), va=float(d['Va_all'][gi_all, j]),
                                phaser_lr=(float(lr_ph[j]) if lr_ph is not None else np.nan),
                                phaser_n=(int(n_ph[j]) if lr_ph is not None else 0)))
    EQ = pd.DataFrame(eq_rows)
    EQ.to_csv(OUT / 'eq_class_informative.tsv', sep='\t', index=False)
    # per dominant record: Salmon informative counts; Gibbs variance vs them
    for i_, r in DOM.iterrows():
        e = EQ[(EQ.gene == r.gene) & (EQ.pos == r.pos)].iloc[0]
        gi = genes.index(r.gene)
        DOM.loc[i_, 'eq_L_only'] = e.L_only; DOM.loc[i_, 'eq_R_only'] = e.R_only
        DOM.loc[i_, 'eq_both'] = e.both
        DOM.loc[i_, 'eq_lr'] = e.eq_lr
        DOM.loc[i_, 'gibbs_only_over_eq_poisson'] = r.va_gibbs_only / e.eq_poisson
        oth = EQ[(EQ.gene == r.gene) & (EQ.pos != r.pos) & (EQ.va > EPS)]
        DOM.loc[i_, 'gene_median_eq_informative_other'] = float((oth.L_only + oth.R_only).median())
        DOM.loc[i_, 'donor_eq_informative_rank_in_gene'] = int(
            (EQ[(EQ.gene == r.gene)].L_only + EQ[(EQ.gene == r.gene)].R_only).rank(ascending=False, method='min')[EQ[(EQ.gene == r.gene) & (EQ.pos == r.pos)].index[0]])
        # alignment-based comparison, gene-wide: eq-informative log ratio vs phASER
        both = EQ[(EQ.gene == r.gene) & (EQ.phaser_n >= 20) & ((EQ.L_only + EQ.R_only) >= 20)]
        if len(both) >= 5:
            DOM.loc[i_, 'gene_spearman_eq_lr_vs_phaser'] = float(sps.spearmanr(both.eq_lr, both.phaser_lr)[0])
            DOM.loc[i_, 'gene_n_eq_vs_phaser'] = len(both)
        qr = quant_rows(rna_of[r.donor], gene_tx[r.gene])
        qr.to_csv(OUT / f'quant_{r.gene}_{r.donor}.tsv', sep='\t')
    DOM.to_csv(OUT / 'dominant_records.tsv', sep='\t', index=False)

    summary['dominant_records'] = DOM.replace({np.nan: None}).to_dict('records')
    summary['cis_scan_best'] = []
    for r in DOM.itertuples():
        tab = SCAN[(SCAN.gene == r.gene) & (SCAN.donor == r.donor)]
        nv = len(tab)
        bc = tab[tab.sign_consistent].sort_values('p').head(1)
        ent = dict(gene=r.gene, donor=r.donor, n_donor_het_scanned=nv,
                   n_consistent_p_lt_005=int((tab.sign_consistent & (tab.p < .05)).sum()),
                   n_inconsistent_p_lt_005=int((~tab.sign_consistent & (tab.p < .05)).sum()),
                   donor_z_unexplained_median=float(tab.donor_z_unexplained.median()) if nv else None)
        if len(bc):
            b0 = bc.iloc[0]
            ent.update(best_variant=b0.variant, p=float(b0.p), p_bonferroni=float(min(1, b0.p * nv)),
                       beta=float(b0.beta), se=float(b0.se), n_het_other=int(b0.n_het_other),
                       pred_a=float(b0.pred_a_donor), frac_of_a=float(b0.frac_of_a),
                       donor_resid_z_after=float(b0.donor_resid_z))
            # does the association replicate in ALIGNMENT-based allelic counts
            # (phASER on STAR) across the same other donors?
            lr_ph, n_ph = ph_ratio(r.gene)
            vi_ = int(np.where(vdf.index == b0.variant)[0][0])
            sv_ = (xL[vi_] - xR[vi_]).astype(float)
            okp = (n_ph >= 10) & (sv_ != 0)
            okp[int(r.pos)] = False
            if okp.sum() >= 3:
                wp = 1 / (1 / (phm[r.gene][0][ph_pos] + KAPPA) + 1 / (phm[r.gene][1][ph_pos] + KAPPA))
                okall = (n_ph >= 10); okall[int(r.pos)] = False
                xw, yw, ww_ = sv_[okall], lr_ph[okall], wp[okall]
                bp = float((ww_ * xw * yw).sum() / (ww_ * xw * xw).sum())
                res_ = yw - bp * xw
                sig2 = float((ww_ * res_ ** 2).sum() / (okall.sum() - 1))
                sep_ = float(np.sqrt(sig2 / (ww_ * xw * xw).sum()))
                ent.update(phaser_beta=bp, phaser_se=sep_,
                           phaser_p=float(sps.f.sf((bp / sep_) ** 2, 1, int(okall.sum()) - 1)),
                           phaser_n_het_other=int(okp.sum()),
                           phaser_pred_a=float(bp * sv_[int(r.pos)]),
                           phaser_donor_lr=float(lr_ph[int(r.pos)]))
        summary['cis_scan_best'].append(ent)
    summary['observed_het_vs_hom'] = OBS.replace({np.nan: None}).to_dict('records')
    summary['donor_exonic_hets'] = EXO.replace({np.nan: None}).to_dict('records')
    summary['runtime_s'] = time.time() - t0
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=1, default=float))
    print(f'wrote {OUT} in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()

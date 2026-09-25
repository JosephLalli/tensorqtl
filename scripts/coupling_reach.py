"""How far does the weight-residual coupling reach beyond 46 genes at one variant?

QUESTION. The nominal-p null instrument (null_permutation_instrument.py) found
hapmixQTL's allelic channel anticonservative under a donor-records permutation
at each gene's RASQUAL lead (0.069 / 0.020 / 0.0057 at 0.05 / 0.01 / 0.001).
The leading candidate mechanism is a coupling, inside a gene, between each
donor record's weight w = 1/va and its whitened squared residual z^2 = a^2/va.
Its first-order closed form is

    R_g = n_a * sum(w z^2) / (sum(w) * sum(z^2))
        = mean(w z^2) / (mean(w) * mean(z^2))          over informative donors

which is the permutation variance of the slope over the mean reported squared
standard error (sdratio^2) to first order, and is exactly 1 in expectation under
the shipped model Var(a) = sigma^2 va. R_g is a property of the RECORDS, not of
the variant tested. This script asks whether the finding generalises

  (a) ACROSS VARIANTS within a gene. The mechanism predicts sdratio roughly
      constant within a gene and flat across minor-allele frequency (MAF),
      but says nothing about the TAIL, which a dominant record could make
      MAF-dependent (it matters only when it lands on one of ~2pq*92 hets).
        a1  re-reads se_accuracy_by_maf_20260924/per_variant.tsv.gz (30
            permutations, map_nominal, ~215,000 gene-variant units)
        a2  runs 500 records permutations at EVERY tested variant of all 59
            genes (MAF >= 0.05, +-1 Mb, gene body excluded, as load_inputs
            defines them; ~270,000 gene-variant units), with 10 model-generated
            record sets per gene run through the SAME permutations as the noise
            floor, plus a split-half reliability of per-variant sdratio
  (b) ACROSS GENES: R_a and R_t for every gene in the Gibbs cache with >= 20
      informative allelic donors, each against its own model-null band (1,000
      model record sets per gene at the gene's real weights); the fraction of
      genes with one record holding > 50% of sum(w z^2), observed and expected
      under the model; the scale-mixture-predicted allelic nominal rates; and a
      DIRECT allelic records permutation (2,000 permutations) at synthetic
      variants of fixed MAF, real records against model records.
  (c) (b) reported by coverage stratum and by number of informative donors.

WHY A SYNTHETIC VARIANT IS EXACT IN (b). Under a records permutation the
genotype stays at the donor position and every record moves as a unit, so the
permutation distribution of the statistic depends on the genotype only through
the MULTISET of its values (how many s = +1, -1, 0). A synthetic phased
genotype with the Hardy-Weinberg heterozygote count of a given MAF, split evenly
between the two phases, therefore has exactly the permutation null of any real
variant with those counts. No VCF is needed for 34,000 genes.

DEFINITIONS.
  allelic channel   weighted least squares through the origin, w = 1/va,
                    n_a = donors with va > 1e-12 (INCLUDING s = 0),
                    sigma_a^2 = RSS/(n_a - 1); t^2 against F(1, n_a - 1)
  total channel     weighted least squares, w_t = 1/vt, intercept + 17
                    covariates partialled out in the sqrt(w_t)-weighted space,
                    dof n_t - 19
  combined          inverse-variance meta-analysis, F(1, min dof)
  R_t               the total-channel analogue: r = whitened residual of t on
                    [1, C] at weights w_t, h = leverage of the whitened design,
                    R_t = (n_t - 18) sum(w_t r^2) / (sum(w_t (1 - h)) sum(r^2)).
                    The slope's numerator is sum_j g'_j sqrt(w_j) r_j for a
                    permuted centred genotype g', so its permutation variance
                    is var(g) sum(w r^2), and E[den] = var(g) sum(w (1 - h));
                    without the (1 - h) the model mean of R_t is ~0.75, not 1,
                    because heavy donors carry high leverage. Its model band
                    uses the same formula on model records
  sdratio           sd over permutations of the slope / root-mean-square of the
                    reported se (the instrument's definition)
  model record set  a = z sqrt(va), t = e sqrt(vt), z, e iid N(0, 1), at the
                    gene's real va, vt and covariates: the shipped model's null
  scale mixture     per gene, predicted rate at alpha =
                    2 * t_sf(t_isf(alpha/2, dof) / sqrt(R), dof), i.e. the
                    statistic treated as a t variate inflated by sqrt(R)
  Kish n_eff        (sum w)^2 / sum(w^2): the number of equally weighted donors
                    that carry the same information as the weighted set

HELD FIXED: the records (Salmon posterior-mean summaries with the shipped
count_noise=True), the covariates, the RandomState(42) permutation stream of the
instrument. VARIES: the variant (a), the gene (b), and real vs model records.

GATES (the script aborts on any failure):
  G1  the vectorised fit, at each of the 46 genes' lead variant over all 2,000
      permutations, reproduces null_long.tsv.gz's t2_a, t2_t and t2_b to
      relative 1e-9 and the instrument's pooled rejection counts EXACTLY at
      0.10/0.05/0.01/0.001 for all three channels
  G2  the lead variant inside the a2 all-variant run reproduces the same t2
      values for permutations 0..499 to relative 1e-9 (the multi-variant
      vectorisation is the same fit)
  G3  compute_summaries_from_gibbs over the whole cache, chunked and reordered
      by sample id, reproduces inputs_at_lead.npz's A_all/T_all/Va_all/Vt_all
      (59 genes) to absolute 1e-12
  G4  R_a and R_t recomputed from the transcriptome pass equal those from the
      npz arrays for the 46 genes (relative 1e-9)

Master seed 42. The permutation stream is RandomState(42), 2,000 permutations,
the instrument's. Everything else draws from SeedSequence(42) child streams:
child 0 bootstrap, 1 a2 model records (spawned per gene), 2 b model bands and
model records (spawned per chunk), 3 synthetic genotype positions.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import contextlib                                   # noqa: E402
import io                                           # noqa: E402
import json                                         # noqa: E402
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
RUN = D / 'rasqual_default_mode_20260923'
CACHE = D / 'cache' / 'gibbs_56b63c3b37ed5df8'
OUT = D / 'coupling_reach_20260925'

SEED, EPS, N = 42, 1e-12, 92
N_PERM_ALL = 2000          # the instrument's stream
N_PERM_A2 = 500            # a2: every tested variant
M_A2 = 10                  # a2: model record sets per gene
K_BAND = 1000              # b: model draws per gene for the R band
K_SCALE = 200              # b: model draws used in the scale-mixture baseline
M_B = 4                    # b: model record sets per gene, direct permutation
MIN_NA = 20
ALPHAS = (0.05, 0.01, 0.001)
ALPHAS_B = (0.05, 0.01, 0.001, 1e-4)
MAF_LEVELS = (0.05, 0.10, 0.20, 0.30, 0.50)
MAF_BINS = [(0.05, 0.10), (0.10, 0.20), (0.20, 0.30), (0.30, 0.50001)]
COV_BINS = [(0, 30), (30, 100), (100, 700), (700, 3000), (3000, np.inf)]
COV_LABEL = ['<30', '30-100 (LOW)', '100-700 (MID)', '700-3000 (HIGH)',
             '>=3000 (HIGH)']
NA_BINS = [(20, 40), (40, 60), (60, 80), (80, 93)]
N_BOOT = 2000
CHUNK = 250
N_WORK_A2 = 20
N_WORK_B = 16

SS = np.random.SeedSequence(SEED).spawn(4)
log = lambda *a: print(time.strftime('%H:%M:%S'), *a, flush=True)


def perm_stream():
    rng = np.random.RandomState(SEED)
    return np.array([rng.permutation(N) for _ in range(N_PERM_ALL)])


# ---------------------------------------------------------------------------
#  the vectorised fit: many record sets x many permutations x many variants
# ---------------------------------------------------------------------------

def allelic_prep(a, va):
    ka = np.isfinite(a) & np.isfinite(va) & (va > EPS)
    w = np.where(ka, 1.0 / np.where(ka, va, 1.0), 0.0)
    return ka, w


def core(a_sets, va, t_sets, vt, C, Smat, Gmat, P, groups, keep_col=None,
         block=25):
    """Records permutation of K record sets sharing va, vt and C.

    a_sets, t_sets (K, N): allelic / total records (row 0 is the real set);
    Smat (N, V) phased s = xL - xR, Gmat (N, V) dosage/2, both at FIXED
    positions; P (nP, N) permutations, record at position q is P[p, q].
    groups: dict name -> boolean mask over the nP permutations; accumulators
    are kept per group. keep_col: a variant column whose per-permutation
    t2 values for set 0 are returned (the gate).
    """
    K, V = a_sets.shape[0], Smat.shape[1]
    ka, w = allelic_prep(a_sets[0], va)
    n_a = int(ka.sum())
    kt = np.isfinite(t_sets[0]) & np.isfinite(vt) & (vt > EPS)
    n_t = int(kt.sum())
    wt = np.where(kt, 1.0 / np.where(kt, vt, 1.0), 0.0)
    sw = np.sqrt(wt)
    U = np.where(ka, a_sets, 0.0) * w                  # w a
    Ssum = (np.where(ka, a_sets, 0.0) ** 2 * w).sum(1)  # sum z^2 per set
    T0 = np.where(kt, t_sets, 0.0)
    S2 = Smat ** 2
    dofa, doft = n_a - 1, n_t - 1 - (1 + C.shape[1])
    dofb = min(dofa, doft)
    crit = {ch: np.array([sps.f.isf(al, 1, d) for al in ALPHAS])
            for ch, d in (('a', dofa), ('t', doft), ('b', dofb))}
    stats = ('n', 'sba', 'sba2', 'ssea2', 'ssea', 'sbt', 'sbt2', 'sset2',
             'sset', 'sb', 'sb2', 'sse2', 'sse')
    acc = {gname: {s: np.zeros((K, V)) for s in stats} for gname in groups}
    for gname in groups:
        acc[gname]['rej'] = np.zeros((3, len(ALPHAS), K, V))
    kept = {'t2_a': [], 't2_t': [], 't2_b': [], 'valid': []}
    for p0 in range(0, len(P), block):
        Pb = P[p0:p0 + block]
        nb = len(Pb)
        # allelic
        num = U[:, Pb] @ Smat                          # K, nb, V
        den = w[Pb] @ S2                               # nb, V
        with np.errstate(divide='ignore', invalid='ignore'):
            ba = num / den
            rss = Ssum[:, None, None] - num * ba
            sea2 = rss / dofa / den
            t2a = dofa * num ** 2 / (den * Ssum[:, None, None] - num ** 2)
        # total
        swp = sw[Pb]                                   # nb, N
        Zp = np.concatenate([np.ones((nb, N, 1)), C[Pb]], axis=2) * swp[:, :, None]
        Q = np.linalg.qr(Zp)[0]                        # nb, N, 18
        Xg = swp[:, :, None] * Gmat[None]              # nb, N, V
        Xr = Xg - Q @ (np.transpose(Q, (0, 2, 1)) @ Xg)
        Yt = np.transpose(T0[:, Pb], (1, 2, 0)) * swp[:, :, None]   # nb, N, K
        Yr = Yt - Q @ (np.transpose(Q, (0, 2, 1)) @ Yt)
        xy = np.einsum('bnk,bnv->kbv', Yr, Xr)
        xx = (Xr ** 2).sum(1)                          # nb, V
        yy = (Yr ** 2).sum(1).T                        # K, nb
        with np.errstate(divide='ignore', invalid='ignore'):
            bt = xy / xx
            rsst = yy[:, :, None] - xy * bt
            set2 = rsst / doft / xx
            t2t = doft * xy ** 2 / (xx * yy[:, :, None] - xy ** 2)
            prec = 1 / sea2 + 1 / set2
            bb = (ba / sea2 + bt / set2) / prec
            se2 = 1 / prec
            t2b = bb ** 2 / se2
        valid = ((den > 0)[None] & (xx > 0)[None] & np.isfinite(sea2) &
                 np.isfinite(set2) & (sea2 > 0) & (set2 > 0))
        if keep_col is not None:
            kept['t2_a'].append(t2a[0, :, keep_col])
            kept['t2_t'].append(t2t[0, :, keep_col])
            kept['t2_b'].append(t2b[0, :, keep_col])
            kept['valid'].append(valid[0, :, keep_col])
        vals = dict(ba=ba, sea2=sea2, bt=bt, set2=set2, b=bb, se2=se2)
        vals = {k: np.where(valid, v, 0.0) for k, v in vals.items()}
        t2s = {'a': np.where(valid, t2a, 0.0), 't': np.where(valid, t2t, 0.0),
               'b': np.where(valid, t2b, 0.0)}
        for gname, gmask in groups.items():
            m = gmask[p0:p0 + nb]
            if not m.any():
                continue
            A = acc[gname]
            sl = lambda x: x[:, m, :].sum(1)
            A['n'] += sl(valid.astype(float))
            A['sba'] += sl(vals['ba']); A['sba2'] += sl(vals['ba'] ** 2)
            A['ssea2'] += sl(vals['sea2']); A['ssea'] += sl(np.sqrt(vals['sea2']))
            A['sbt'] += sl(vals['bt']); A['sbt2'] += sl(vals['bt'] ** 2)
            A['sset2'] += sl(vals['set2']); A['sset'] += sl(np.sqrt(vals['set2']))
            A['sb'] += sl(vals['b']); A['sb2'] += sl(vals['b'] ** 2)
            A['sse2'] += sl(vals['se2']); A['sse'] += sl(np.sqrt(vals['se2']))
            for ci, ch in enumerate('atb'):
                for ai in range(len(ALPHAS)):
                    A['rej'][ci, ai] += sl((t2s[ch] > crit[ch][ai]) & valid)
    if keep_col is not None:
        kept = {k: np.concatenate(v) for k, v in kept.items()}
    return acc, kept, dict(n_a=n_a, n_t=n_t, dofa=dofa, doft=doft, dofb=dofb)


def sdratio(A, b, b2, se2):
    n = A['n']
    with np.errstate(divide='ignore', invalid='ignore'):
        var = (A[b2] - A[b] ** 2 / n) / (n - 1)
        return np.sqrt(var) / np.sqrt(A[se2] / n)


def r_allelic(a, va):
    ka, w = allelic_prep(a, va)
    z2 = np.where(ka, a, 0.0) ** 2 * w
    return ka.sum() * (w * z2).sum() / (w.sum() * z2.sum())


def share_max(a, va):
    ka, w = allelic_prep(a, va)
    wz2 = (np.where(ka, a, 0.0) ** 2) * w * w
    return float(wz2.max() / wz2.sum())


def r_total(t, vt, C):
    kt = np.isfinite(t) & np.isfinite(vt) & (vt > EPS)
    wt = np.where(kt, 1.0 / np.where(kt, vt, 1.0), 0.0)
    sw = np.sqrt(wt)
    Zw = np.column_stack([np.ones(N), C]) * sw[:, None]
    Q = np.linalg.qr(Zw)[0]
    y = np.where(kt, t, 0.0) * sw
    r = y - Q @ (Q.T @ y)
    h = (Q ** 2).sum(1)
    return ((kt.sum() - 18) * (wt * r ** 2).sum() /
            ((wt * (1 - h)).sum() * (r ** 2).sum()))


# ---------------------------------------------------------------------------
#  G1: the lead variant over the instrument's 2,000 permutations
# ---------------------------------------------------------------------------

def gate1(Z, P):
    log('G1: lead variant, 46 genes x 2,000 permutations, vs null_long')
    L = pd.read_csv(INST / 'null_long.tsv.gz', sep='\t')
    summ = json.loads((INST / 'summary.json').read_text())
    rows = []
    worst = 0.0
    for k, g in enumerate(Z['genes']):
        acc, kept, info = core(Z['a'][k][None], Z['va'][k], Z['t'][k][None],
                               Z['vt'][k], Z['C'], Z['s'][k][:, None],
                               Z['g'][k][:, None], P, {'all': np.ones(len(P), bool)},
                               keep_col=0, block=200)
        ref = L[L.gene == g].set_index('perm')
        v = kept['valid']
        if v.sum() != len(ref) or not np.array_equal(np.where(v)[0], ref.index.values):
            raise SystemExit(f'G1 FAILED at {g}: valid permutation set differs')
        for col in ('t2_a', 't2_t', 't2_b'):
            mine, theirs = kept[col][v], ref[col].values
            rel = np.max(np.abs(mine - theirs) / np.maximum(np.abs(theirs), 1e-300))
            worst = max(worst, float(rel))
        rows.append(pd.DataFrame(dict(gene=g, perm=np.where(v)[0],
                                      t2_a=kept['t2_a'][v], t2_t=kept['t2_t'][v],
                                      t2_b=kept['t2_b'][v], dofa=info['dofa'],
                                      doft=info['doft'], dof=info['dofb'])))
    if worst > 1e-9:
        raise SystemExit(f'G1 FAILED: max relative t2 difference {worst:.3e}')
    M = pd.concat(rows, ignore_index=True)
    M['p_a'] = sps.f.sf(M.t2_a, 1, M.dofa)
    M['p_t'] = sps.f.sf(M.t2_t, 1, M.doft)
    M['p_b'] = sps.f.sf(M.t2_b, 1, M.dof)
    counts = {}
    for ch, pc in (('allelic', 'p_a'), ('total', 'p_t'), ('combined', 'p_b')):
        for al in ('0.1', '0.05', '0.01', '0.001'):
            mine = int((M[pc] < float(al)).sum())
            theirs = int(round(summ['pooled'][ch][al]['rate'] * len(L)))
            ref_ct = int((L[pc] < float(al)).sum())
            if not (mine == theirs == ref_ct):
                raise SystemExit(f'G1 FAILED: pooled count {ch} {al}: '
                                 f'{mine} vs summary {theirs} vs null_long {ref_ct}')
            counts[f'{ch}_{al}'] = mine
    log(f'G1 passed: max relative t2 difference {worst:.2e}; pooled counts '
        f'identical ({len(M)} gene-permutation rows)')
    return dict(max_rel_t2=worst, n_rows=len(M), pooled_counts=counts)


# ---------------------------------------------------------------------------
#  a2: every tested variant, 500 permutations, real + model records
# ---------------------------------------------------------------------------

_A2 = {}


def a2_gene(k):
    I, P = _A2['I'], _A2['P']
    g = I['genes'][k]
    import compare_mixqtl_replication as CM
    vsel = CM.gene_variant_index(I, g)
    rows = list(I['idx'][vsel])
    lead = _A2['leads'].get(g)
    lead_row = None
    if lead is not None:
        lead_row = int(_A2['vi'][lead])
        if lead_row not in rows:
            rows.append(lead_row)
    rows = np.array(rows)
    Smat = (I['xL'][rows].astype(float) - I['xR'][rows].astype(float)).T
    Gmat = I['dos'][rows].astype(float).T / 2.0
    a, va, t, vt = (_A2[x][k] for x in ('A', 'Va', 'T', 'Vt'))
    rng = np.random.default_rng(_A2['seeds'][k])
    ka, _ = allelic_prep(a, va)
    za = rng.standard_normal((M_A2, N))
    zt = rng.standard_normal((M_A2, N))
    a_sets = np.vstack([a, np.where(ka, za * np.sqrt(np.where(ka, va, 0.0)), 0.0)])
    t_sets = np.vstack([t, zt * np.sqrt(vt)])
    nP = len(P)
    idx = np.arange(nP)
    groups = {'all': np.ones(nP, bool), 'h1': idx < nP // 2, 'h2': idx >= nP // 2,
              'f30': idx < 30}
    keep_col = (int(np.where(rows == lead_row)[0][0]) if lead_row is not None
                else None)
    acc, kept, info = core(a_sets, va, t_sets, vt, _A2['C'], Smat, Gmat, P,
                           groups, keep_col=keep_col)
    dos = I['dos'][rows].astype(float)
    af = dos.mean(1) / 2
    s = Smat
    out = dict(gene=g, rows=rows, vid=np.array(I['vdf'].index[rows]),
               maf=np.minimum(af, 1 - af), n_het=(s != 0).sum(0),
               n_plus=(s > 0).sum(0), n_minus=(s < 0).sum(0),
               in_tested=np.isin(rows, I['idx'][vsel]),
               is_lead=(rows == lead_row) if lead_row is not None
               else np.zeros(len(rows), bool),
               acc=acc, kept=kept, info=info)
    # first-order, variant-specific prediction of sdratio_a^2 (linear
    # permutation statistic: exact Var(num), E[den], sigma^2 ~ S/n_a)
    kaw = allelic_prep(a, va)
    u = np.where(kaw[0], a, 0.0) * kaw[1]
    Sz = (np.where(kaw[0], a, 0.0) ** 2 * kaw[1]).sum()
    varnum = ((s - s.mean(0)) ** 2).sum(0) * ((u - u.mean()) ** 2).sum() / (N - 1)
    eden = (s ** 2).sum(0) * kaw[1].sum() / N
    out['pred_v'] = varnum / (eden * Sz / info['n_a'])
    return out


def run_a2(I, Z, P, A_all, T_all, Va_all, Vt_all):
    leads = dict(zip(Z['genes'], Z['variants']))
    vi = pd.Series(np.arange(len(I['vdf'])), index=I['vdf'].index)
    seeds = SS[1].spawn(len(I['genes']))
    _A2.update(I=I, P=P[:N_PERM_A2], leads=leads, vi=vi, seeds=seeds,
               A=A_all, T=T_all, Va=Va_all, Vt=Vt_all, C=Z['C'])
    log(f'a2: {len(I["genes"])} genes, every tested variant, {N_PERM_A2} '
        f'permutations, real + {M_A2} model record sets, {N_WORK_A2} workers')
    with get_context('fork').Pool(N_WORK_A2) as pool:
        res = pool.map(a2_gene, range(len(I['genes'])), chunksize=1)
    return res


def a2_tables(res, Z, L):
    Rg = {g: (r_allelic(Z['A_all'][i], Z['Va_all'][i]),
              r_total(Z['T_all'][i], Z['Vt_all'][i], Z['C']))
          for i, g in enumerate(Z['all_genes'])}
    null46 = set(Z['genes'])
    vrows, worst = [], 0.0
    for r in res:
        A, info = r['acc'], r['info']
        d = dict(gene=r['gene'], variant=r['vid'], maf=r['maf'], n_het=r['n_het'],
                 n_plus=r['n_plus'], n_minus=r['n_minus'], in_tested=r['in_tested'],
                 is_lead=r['is_lead'], n_a=info['n_a'], n_t=info['n_t'],
                 R_a=Rg[r['gene']][0], R_t=Rg[r['gene']][1],
                 pred_v=r['pred_v'], in_null46=r['gene'] in null46)
        for grp in ('all', 'h1', 'h2', 'f30'):
            G = A[grp]
            for ch, (b, b2, se2) in (('a', ('sba', 'sba2', 'ssea2')),
                                     ('t', ('sbt', 'sbt2', 'sset2')),
                                     ('b', ('sb', 'sb2', 'sse2'))):
                sr = sdratio(G, b, b2, se2)
                d[f'sdr_{ch}_{grp}'] = sr[0]
                if grp in ('all', 'h1', 'h2'):
                    d[f'sdr_{ch}_{grp}_model_mean'] = np.nanmean(sr[1:], 0)
                    if grp == 'all':
                        d[f'sdr_{ch}_model_sd_across_sets'] = np.nanstd(sr[1:], 0, ddof=1)
        d['n_valid'] = A['all']['n'][0]
        d['n_valid_model'] = A['all']['n'][1:].sum(0)
        for ci, ch in enumerate('atb'):
            for ai, al in enumerate(ALPHAS):
                d[f'rej_{ch}_{al}'] = A['all']['rej'][ci, ai, 0]
                d[f'rej_{ch}_{al}_model'] = A['all']['rej'][ci, ai, 1:].sum(0)
        # 30-permutation quantities in the se_accuracy_by_maf convention
        F = A['f30']
        n = F['n'][0]
        d['f30_sd_beta_a'] = np.sqrt((F['sba2'][0] - F['sba'][0] ** 2 / n) / (n - 1))
        d['f30_mean_se_a'] = F['ssea'][0] / n
        d['f30_sd_beta_t'] = np.sqrt((F['sbt2'][0] - F['sbt'][0] ** 2 / n) / (n - 1))
        d['f30_mean_se_t'] = F['sset'][0] / n
        d['f30_sd_beta_b'] = np.sqrt((F['sb2'][0] - F['sb'][0] ** 2 / n) / (n - 1))
        d['f30_mean_se_b'] = F['sse'][0] / n
        # per-set model sdratio for the within-gene spread floor
        sr_m = sdratio(A['all'], 'sba', 'sba2', 'ssea2')[1:]
        d['_model_sets_a'] = list(sr_m.T)
        sr_mt = sdratio(A['all'], 'sbt', 'sbt2', 'sset2')[1:]
        d['_model_sets_t'] = list(sr_mt.T)
        d['_model_h1_a'] = list(sdratio(A['h1'], 'sba', 'sba2', 'ssea2')[1:].T)
        d['_model_h2_a'] = list(sdratio(A['h2'], 'sba', 'sba2', 'ssea2')[1:].T)
        vrows.append(pd.DataFrame(d))
        # G2: the lead inside the all-variant run
        if r['kept'] and len(r['kept'].get('t2_a', [])):
            ref = L[(L.gene == r['gene']) & (L.perm < N_PERM_A2)].set_index('perm')
            v = r['kept']['valid']
            if not np.array_equal(np.where(v)[0], ref.index.values):
                raise SystemExit(f'G2 FAILED at {r["gene"]}: valid set differs')
            for col in ('t2_a', 't2_t', 't2_b'):
                rel = np.max(np.abs(r['kept'][col][v] - ref[col].values) /
                             np.abs(ref[col].values))
                worst = max(worst, float(rel))
    if worst > 1e-9:
        raise SystemExit(f'G2 FAILED: max relative t2 difference {worst:.3e}')
    log(f'G2 passed: lead inside the all-variant run, max relative t2 '
        f'difference {worst:.2e}')
    V = pd.concat(vrows, ignore_index=True)
    return V, worst


# ---------------------------------------------------------------------------
#  b: the transcriptome
# ---------------------------------------------------------------------------

_B = {}


def synthetic_genotypes(pos_rng):
    """(N, L) phased s at the Hardy-Weinberg heterozygote count of each MAF."""
    S = np.zeros((N, len(MAF_LEVELS)))
    for l, p in enumerate(MAF_LEVELS):
        k = int(round(2 * p * (1 - p) * N))
        pos = pos_rng.permutation(N)[:k]
        S[pos[:k // 2], l] = 1.0
        S[pos[k // 2:], l] = -1.0
    return S


def b_chunk(ci):
    g0 = ci * CHUNK
    mm = {k: np.load(CACHE / f'{k}.npy', mmap_mode='r') for k in ('YL', 'YR', 'YT')}
    g1 = min(g0 + CHUNK, mm['YL'].shape[0])
    keep = _B['keep']
    YL = np.asarray(mm['YL'][g0:g1])[:, keep, :]
    YR = np.asarray(mm['YR'][g0:g1])[:, keep, :]
    YT = np.asarray(mm['YT'][g0:g1])[:, keep, :]
    import run_hapmixqtl_from_salmon as H
    with contextlib.redirect_stdout(io.StringIO()):
        A, T, Va, Vt, _ = H.compute_summaries_from_gibbs(YL, YR, yT=YT)
    mAS = (YL + YR).mean(2)
    mT = YT.mean(2)
    del YL, YR, YT
    C, P, Ssyn = _B['C'], _B['P'], _B['Ssyn']
    S2syn = Ssyn ** 2
    rng = np.random.default_rng(_B['seeds'][ci])
    Zc = np.column_stack([np.ones(N), C])
    rows, direct = [], []
    for j in range(g1 - g0):
        a, va, t, vt = A[j], Va[j], T[j], Vt[j]
        ka, w = allelic_prep(a, va)
        n_a = int(ka.sum())
        r = dict(gene_idx=g0 + j, n_a=n_a,
                 med_asc=float(np.median(mAS[j][ka])) if n_a else 0.0,
                 med_tot=float(np.median(mT[j])))
        if n_a < MIN_NA:
            rows.append(r)
            continue
        a0 = np.where(ka, a, 0.0)
        z2 = a0 ** 2 * w
        wz2 = w * z2
        R_a = n_a * wz2.sum() / (w.sum() * z2.sum())
        neff = w.sum() ** 2 / (w ** 2).sum()
        share = wz2.max() / wz2.sum()
        # model band: z iid N(0,1) on the informative donors, same weights
        zz = rng.standard_normal((K_BAND, n_a)) ** 2
        wk = w[ka]
        Rm = n_a * (zz * wk).sum(1) / (wk.sum() * zz.sum(1))
        sh_m = (zz * wk).max(1) / (zz * wk).sum(1)
        lRm = np.log(Rm)
        rho = sps.spearmanr(wk, z2[ka]).correlation
        # total channel
        kt = np.isfinite(t) & np.isfinite(vt) & (vt > EPS)
        n_t = int(kt.sum())
        wt = np.where(kt, 1.0 / np.where(kt, vt, 1.0), 0.0)
        swt = np.sqrt(wt)
        Q = np.linalg.qr(Zc * swt[:, None])[0]
        y = np.where(kt, t, 0.0) * swt
        rr = y - Q @ (Q.T @ y)
        swh = (wt * (1 - (Q ** 2).sum(1))).sum()
        R_t = (n_t - 18) * (wt * rr ** 2).sum() / (swh * (rr ** 2).sum())
        E = rng.standard_normal((N, K_BAND)) * kt[:, None]
        Er = E - Q @ (Q.T @ E)
        Rtm = (n_t - 18) * (wt[:, None] * Er ** 2).sum(0) / (swh * (Er ** 2).sum(0))
        lRtm = np.log(Rtm)
        r.update(R_a=R_a, neff=neff, share_max=share, rho_w_z2=rho,
                 lRa_model_mean=lRm.mean(), lRa_model_sd=lRm.std(ddof=1),
                 Ra_model_q025=np.quantile(Rm, 0.025),
                 Ra_model_q975=np.quantile(Rm, 0.975),
                 Ra_p_upper=float((Rm >= R_a).mean()),
                 p_dominant_model=float((sh_m > 0.5).mean()),
                 delta_sd_logR=np.sqrt(max(2 * (1 / neff - 1 / n_a), 0)),
                 R_t=R_t, n_t=n_t, lRt_model_mean=lRtm.mean(),
                 lRt_model_sd=lRtm.std(ddof=1),
                 Rt_model_q025=np.quantile(Rtm, 0.025),
                 Rt_model_q975=np.quantile(Rtm, 0.975))
        dof = n_a - 1
        for al in ALPHAS_B:
            c = sps.t.isf(al / 2, dof)
            r[f'smix_{al}'] = 2 * sps.t.sf(c / np.sqrt(R_a), dof)
            r[f'smix_{al}_model'] = float(np.mean(2 * sps.t.sf(
                c / np.sqrt(Rm[:K_SCALE]), dof)))
        for al in ALPHAS_B:
            c = sps.t.isf(al / 2, n_t - 19)
            r[f'smix_t_{al}'] = 2 * sps.t.sf(c / np.sqrt(R_t), n_t - 19)
            r[f'smix_t_{al}_model'] = float(np.mean(2 * sps.t.sf(
                c / np.sqrt(Rtm[:K_SCALE]), n_t - 19)))
        # direct allelic records permutation at the synthetic variants
        zm = rng.standard_normal((M_B, N))
        a_sets = np.vstack([a0, np.where(ka, zm * np.sqrt(np.where(ka, va, 0.0)), 0.0)])
        U = a_sets * w
        Ssum = (a_sets ** 2 * w).sum(1)
        num = U[:, P] @ Ssyn                          # K, nP, L
        den = w[P] @ S2syn                            # nP, L
        with np.errstate(divide='ignore', invalid='ignore'):
            t2 = dof * num ** 2 / (den * Ssum[:, None, None] - num ** 2)
            ba = num / den
            se2 = (Ssum[:, None, None] - num * ba) / dof / den
        valid = (den > 0)[None] & np.isfinite(t2) & (se2 > 0)
        cnt = np.zeros((M_B + 1, len(MAF_LEVELS), len(ALPHAS_B)))
        for ai, al in enumerate(ALPHAS_B):
            cnt[:, :, ai] = ((t2 > sps.f.isf(al, 1, dof)) & valid).sum(1)
        nval = valid.sum(1)                           # K, L
        bv = np.where(valid[0], ba[0], np.nan)
        sev = np.where(valid[0], se2[0], np.nan)
        sdr = np.nanstd(bv, 0, ddof=1) / np.sqrt(np.nanmean(sev, 0))
        for l, p in enumerate(MAF_LEVELS):
            r[f'direct_sdr_a_maf{p}'] = sdr[l]
        rows.append(r)
        direct.append((g0 + j, cnt, nval))
    return dict(g0=g0, g1=g1, A=A, T=T, Va=Va, Vt=Vt, rows=rows, direct=direct)


def run_b(Z, P):
    samples = (CACHE / 'samples.txt').read_text().split()
    genes = (CACHE / 'genes.txt').read_text().split()
    donors = list(Z['donors'])
    keep = [samples.index(s) for s in donors]
    n_genes = np.load(CACHE / 'YL.npy', mmap_mode='r').shape[0]
    if n_genes != len(genes):
        raise SystemExit(f'genes.txt has {len(genes)} ids, cache {n_genes} rows')
    n_chunks = (n_genes + CHUNK - 1) // CHUNK
    Ssyn = synthetic_genotypes(np.random.default_rng(SS[3]))
    _B.update(keep=keep, C=Z['C'], P=P, Ssyn=Ssyn, n_chunks=n_chunks,
              seeds=SS[2].spawn(n_chunks))
    log(f'b: {n_genes} cache genes in {n_chunks} chunks of {CHUNK}, '
        f'{N_WORK_B} workers')
    res = []
    with get_context('fork').Pool(N_WORK_B) as pool:
        for i, r in enumerate(pool.imap(b_chunk, range(n_chunks))):
            res.append(r)
            if (i + 1) % 20 == 0:
                log(f'  chunk {i + 1}/{n_chunks}')
    A = np.vstack([r['A'] for r in res]); T = np.vstack([r['T'] for r in res])
    Va = np.vstack([r['Va'] for r in res]); Vt = np.vstack([r['Vt'] for r in res])
    G = pd.DataFrame([row for r in res for row in r['rows']])
    G.insert(0, 'gene', [genes[i] for i in G.gene_idx])
    direct = [d for r in res for d in r['direct']]
    return genes, A, T, Va, Vt, G, direct, Ssyn


# ---------------------------------------------------------------------------
#  statistics helpers
# ---------------------------------------------------------------------------

def boot_rate(k, n, rng, k2=None, n2=None):
    """Pooled rate k.sum()/n.sum() with a gene-clustered percentile interval;
    with (k2, n2) also the difference rate1 - rate2 resampled jointly."""
    k, n = np.asarray(k, float), np.asarray(n, float)
    est = k.sum() / n.sum()
    idx = rng.integers(0, len(k), size=(N_BOOT, len(k)))
    b1 = k[idx].sum(1) / n[idx].sum(1)
    out = dict(rate=float(est), lo=float(np.quantile(b1, .025)),
               hi=float(np.quantile(b1, .975)), n_genes=int(len(k)),
               n_tests=float(n.sum()))
    if k2 is not None:
        k2, n2 = np.asarray(k2, float), np.asarray(n2, float)
        e2 = k2.sum() / n2.sum()
        b2 = k2[idx].sum(1) / n2[idx].sum(1)
        out.update(model_rate=float(e2), diff=float(est - e2),
                   diff_lo=float(np.quantile(b1 - b2, .025)),
                   diff_hi=float(np.quantile(b1 - b2, .975)))
    return out


def within_gene_fe_slope(df, y, x):
    d = df[[y, x, 'gene']].replace([np.inf, -np.inf], np.nan).dropna()
    yc = d[y] - d.groupby('gene')[y].transform('mean')
    xc = d[x] - d.groupby('gene')[x].transform('mean')
    slope = float((xc * yc).sum() / (xc ** 2).sum())
    # gene-clustered sandwich se
    e = yc - slope * xc
    g = (xc * e).groupby(d['gene']).sum()
    se = float(np.sqrt((g ** 2).sum()) / (xc ** 2).sum())
    return slope, se


# ---------------------------------------------------------------------------
#  main
# ---------------------------------------------------------------------------

def main():
    OUT.mkdir(exist_ok=True)
    t0 = time.time()
    brng = np.random.default_rng(SS[0])
    Z = dict(np.load(INST / 'inputs_at_lead.npz', allow_pickle=True))
    P = perm_stream()
    res = dict(question='reach of the weight-residual coupling across variants '
                        'and genes', seed=SEED)

    # ---------------- G1 ----------------
    res['gate_G1'] = gate1(Z, P)
    L = pd.read_csv(INST / 'null_long.tsv.gz', sep='\t')

    # ---------------- b (includes G3) ----------------
    genes, A, T, Va, Vt, G, direct, Ssyn = run_b(Z, P)
    gidx = {g: i for i, g in enumerate(genes)}
    rows59 = [gidx[g] for g in Z['all_genes']]
    g3 = max(float(np.max(np.abs(X[rows59] - Z[k])))
             for X, k in ((A, 'A_all'), (T, 'T_all'), (Va, 'Va_all'), (Vt, 'Vt_all')))
    if not g3 <= 1e-12:
        raise SystemExit(f'G3 FAILED: summaries differ from npz by {g3:.3e}')
    log(f'G3 passed: transcriptome summaries reproduce the npz, max abs {g3:.2e}')
    res['gate_G3_max_abs'] = g3
    # G4: R from the npz arrays = R from the transcriptome pass (46 genes)
    Gi = G.set_index('gene')
    g4 = 0.0
    for k, g in enumerate(Z['genes']):
        ra = r_allelic(Z['a'][k], Z['va'][k])
        rt = r_total(Z['t'][k], Z['vt'][k], Z['C'])
        g4 = max(g4, abs(ra / Gi.loc[g, 'R_a'] - 1), abs(rt / Gi.loc[g, 'R_t'] - 1))
    if g4 > 1e-9:
        raise SystemExit(f'G4 FAILED: R differs by relative {g4:.3e}')
    log(f'G4 passed: R_a, R_t agree for the 46 genes, max relative {g4:.2e}')
    res['gate_G4_max_rel'] = g4
    np.savez_compressed(OUT / 'summaries_all_genes.npz', genes=np.array(genes),
                        donors=Z['donors'], A=A, T=T, Va=Va, Vt=Vt)
    del A, T, Va, Vt

    # ---------------- a2 ----------------
    import compare_mixqtl_replication as CM
    with contextlib.redirect_stdout(io.StringIO()):
        I = CM.load_inputs(gene_list=str(D / 'genes_59_stratified_20260923.txt'),
                           regions=str(RUN / 'regions.bed'))
    if list(I['genes']) != list(Z['all_genes']) or list(I['order']) != list(Z['donors']):
        raise SystemExit('a2: gene or donor order differs from the npz')
    a2 = run_a2(I, Z, P, Z['A_all'], Z['T_all'], Z['Va_all'], Z['Vt_all'])
    V, g2 = a2_tables(a2, Z, L)
    res['gate_G2_max_rel'] = g2
    model_sets_a = V.pop('_model_sets_a')
    model_sets_t = V.pop('_model_sets_t')
    msa = np.vstack(model_sets_a.values)             # rows x M_A2
    mst = np.vstack(model_sets_t.values)
    mh1 = np.vstack(V.pop('_model_h1_a').values)
    mh2 = np.vstack(V.pop('_model_h2_a').values)
    V.to_csv(OUT / 'a2_per_variant.tsv.gz', sep='\t', index=False)

    res['a2'] = analyse_a2(V, msa, mst, mh1, mh2, Z, brng)
    res['a1'] = analyse_a1(V, Z)
    res['b'] = analyse_b(G, direct, Z, brng)
    res['instrument_without_dominant_gene'] = without_dominant(Z, L, brng)
    res['a2_coupled_genes_by_maf_file'] = coupled_genes_by_maf(V)
    res['runtime_s'] = time.time() - t0
    (OUT / 'summary.json').write_text(json.dumps(res, indent=2, default=float))
    res['figures'] = make_figures(res)
    (OUT / 'summary.json').write_text(json.dumps(res, indent=2, default=float))
    log(f'wrote {OUT} in {res["runtime_s"]:.0f} s')


def analyse_a2(V, msa, mst, mh1, mh2, Z, brng):
    out = {}
    # a variant with no heterozygote (a handful carry MAF >= 0.05 by dosage
    # with every non-reference donor homozygous) has no allelic slope at all
    ok = (V.in_tested & (V.n_valid >= 100) &
          np.isfinite(V[['sdr_a_h1', 'sdr_a_h2', 'sdr_a_all', 'sdr_t_all']]).all(1))
    out['n_units_dropped_no_valid_fit'] = int((V.in_tested & ~ok).sum())
    msa, mst, mh1, mh2 = msa[ok.values], mst[ok.values], mh1[ok.values], mh2[ok.values]
    V = V[ok]
    T = V.copy()
    T['lsa'] = np.log(T.sdr_a_all)
    T['lst'] = np.log(T.sdr_t_all)
    out['n_units'] = int(len(T)); out['n_genes'] = int(T.gene.nunique())
    # within-gene spread of log sdratio, real vs each model set
    ms = pd.DataFrame(np.log(msa), index=T.index)
    mt = pd.DataFrame(np.log(mst), index=T.index)
    m1 = pd.DataFrame(np.log(mh1), index=T.index)
    m2 = pd.DataFrame(np.log(mh2), index=T.index)
    share = {g: share_max(Z['A_all'][i], Z['Va_all'][i])
             for i, g in enumerate(Z['all_genes'])}
    per = []
    for g, sub in T.groupby('gene', sort=False):
        mg = ms.loc[sub.index]; mtg = mt.loc[sub.index]
        c = np.corrcoef(np.log(sub.sdr_a_h1), np.log(sub.sdr_a_h2))[0, 1]
        # split-half correlation within each model record set, averaged: the
        # floor for variant-to-variant structure the model itself produces
        cm = np.nanmean([np.corrcoef(m1.loc[sub.index, c], m2.loc[sub.index, c])[0, 1]
                         for c in m1.columns])
        per.append(dict(
            gene=g, n_var=len(sub), R_a=sub.R_a.iloc[0], R_t=sub.R_t.iloc[0],
            in_null46=bool(sub.in_null46.iloc[0]), n_a=int(sub.n_a.iloc[0]),
            share_max_a=share[g],
            median_sdr_a=float(sub.sdr_a_all.median()),
            median_sdr_t=float(sub.sdr_t_all.median()),
            median_sdr_b=float(sub.sdr_b_all.median()),
            median_sdr_a_model=float(np.exp(mg.median(0)).mean()),
            median_sdr_t_model=float(np.exp(mtg.median(0)).mean()),
            sd_within_lsa=float(sub.lsa.std(ddof=1)),
            sd_within_lsa_model=float(mg.std(0, ddof=1).mean()),
            sd_within_lsa_model_mcsd=float(mg.std(0, ddof=1).std(ddof=1)),
            sd_within_lst=float(sub.lst.std(ddof=1)),
            sd_within_lst_model=float(mtg.std(0, ddof=1).mean()),
            split_half_r_a=float(c), split_half_r_a_model_per_set=float(cm),
            maf_spearman_a=float(sps.spearmanr(sub.maf, sub.sdr_a_all).correlation),
            pred_v_spearman=float(sps.spearmanr(sub.pred_v, sub.sdr_a_all ** 2).correlation),
            rej_a_001=float(sub['rej_a_0.001'].sum() / sub.n_valid.sum()),
            rej_a_001_model=float(sub['rej_a_0.001_model'].sum() /
                                  sub.n_valid_model.sum())))
    per = pd.DataFrame(per)
    per.to_csv(OUT / 'a2_per_gene.tsv', sep='\t', index=False)
    out['per_gene_file'] = 'a2_per_gene.tsv'
    # between-gene: gene median sdratio_a vs sqrt(R_a)
    out['gene_median_sdr_a_vs_sqrtR'] = dict(
        spearman=float(sps.spearmanr(per.median_sdr_a, per.R_a).correlation),
        pearson_log=float(np.corrcoef(np.log(per.median_sdr_a),
                                      0.5 * np.log(per.R_a))[0, 1]),
        slope_log=float(np.polyfit(0.5 * np.log(per.R_a),
                                   np.log(per.median_sdr_a), 1)[0]))
    out['gene_median_sdr_t_vs_sqrtRt'] = dict(
        spearman=float(sps.spearmanr(per.median_sdr_t, per.R_t).correlation))
    # variance decomposition of log sdratio_a
    tot = float(T.lsa.var(ddof=1))
    within = float((T.lsa - T.groupby('gene').lsa.transform('mean')).var(ddof=1))
    gm = T.groupby('gene').lsa.mean()
    out['variance_decomposition_log_sdr_a'] = dict(
        total=tot, within_gene=within, between_gene=float(gm.var(ddof=1)),
        between_share=float(1 - within / tot),
        between_explained_by_logR=float(np.corrcoef(
            gm.values, np.log(per.set_index('gene').loc[gm.index, 'R_a']))[0, 1] ** 2))
    out['within_gene_sd_log_sdr_a'] = dict(
        real_median=float(per.sd_within_lsa.median()),
        model_median=float(per.sd_within_lsa_model.median()),
        ratio_median=float((per.sd_within_lsa / per.sd_within_lsa_model).median()),
        n_genes_real_above_model_plus_2mcsd=int(
            (per.sd_within_lsa > per.sd_within_lsa_model +
             2 * per.sd_within_lsa_model_mcsd).sum()),
        n_genes=int(len(per)))
    out['within_gene_sd_log_sdr_t'] = dict(
        real_median=float(per.sd_within_lst.median()),
        model_median=float(per.sd_within_lst_model.median()))
    out['split_half_reliability_a'] = dict(
        real_median=float(per.split_half_r_a.median()),
        model_median=float(per.split_half_r_a_model_per_set.median()),
        n_genes_real_gt_model=int((per.split_half_r_a >
                                   per.split_half_r_a_model_per_set).sum()))
    hi = per.sd_within_lsa > per.sd_within_lsa_model + 2 * per.sd_within_lsa_model_mcsd
    out['within_gene_excess_spread_vs_dominance'] = dict(
        median_share_max_excess_genes=float(per.share_max_a[hi].median()),
        median_share_max_other_genes=float(per.share_max_a[~hi].median()),
        spearman_spread_ratio_vs_share_max=float(sps.spearmanr(
            per.sd_within_lsa / per.sd_within_lsa_model, per.share_max_a).correlation))
    # MAF dependence, gene fixed effects
    T['maf_c'] = T.maf
    sl, se = within_gene_fe_slope(T, 'lsa', 'maf')
    T['lsa_model'] = np.log(T.sdr_a_all_model_mean)
    slm, sem = within_gene_fe_slope(T, 'lsa_model', 'maf')
    slt, set_ = within_gene_fe_slope(T, 'lst', 'maf')
    out['maf_slope_log_sdr_a_per_unit_maf'] = dict(real=sl, real_se=se,
                                                   model=slm, model_se=sem)
    out['maf_slope_log_sdr_t_per_unit_maf'] = dict(real=slt, real_se=set_)
    # rates by MAF bin, real vs model, gene-clustered
    bins = []
    for lo, hi in MAF_BINS:
        sub = T[(T.maf >= lo) & (T.maf < hi)]
        gsum = sub.groupby('gene')
        rec = dict(maf_bin=f'{lo}-{min(hi, 0.5)}', n_units=int(len(sub)),
                   median_sdr_a=float(sub.sdr_a_all.median()),
                   median_sdr_a_model=float(sub.sdr_a_all_model_mean.median()),
                   median_sdr_t=float(sub.sdr_t_all.median()),
                   median_sdr_b=float(sub.sdr_b_all.median()))
        for ch in 'atb':
            for al in ALPHAS:
                b = boot_rate(gsum[f'rej_{ch}_{al}'].sum(), gsum.n_valid.sum(), brng,
                              gsum[f'rej_{ch}_{al}_model'].sum(),
                              gsum.n_valid_model.sum())
                for kk in ('rate', 'lo', 'hi', 'model_rate', 'diff_lo', 'diff_hi'):
                    rec[f'{ch}_{al}_{kk}'] = b[kk]
        bins.append(rec)
    bins = pd.DataFrame(bins)
    bins.to_csv(OUT / 'a2_maf_bins.tsv', sep='\t', index=False)
    out['maf_bins'] = bins.to_dict(orient='records')
    # pooled over all units, 46 null genes only and all 59
    for lab, sub in (('all59', T), ('null46', T[T.in_null46])):
        gsum = sub.groupby('gene')
        rec = {}
        for ch in 'atb':
            for al in ALPHAS:
                rec[f'{ch}_{al}'] = boot_rate(
                    gsum[f'rej_{ch}_{al}'].sum(), gsum.n_valid.sum(), brng,
                    gsum[f'rej_{ch}_{al}_model'].sum(), gsum.n_valid_model.sum())
        out[f'pooled_{lab}'] = rec
    # first-order variant-specific prediction
    out['pred_v_vs_R'] = dict(
        median_ratio_pred_v_over_R=float((T.pred_v / T.R_a).median()),
        q05=float((T.pred_v / T.R_a).quantile(.05)),
        q95=float((T.pred_v / T.R_a).quantile(.95)),
        median_within_gene_spearman_pred_v_sdr2=float(per.pred_v_spearman.median()))
    return out


def analyse_a1(V, Z):
    out = {}
    t = pd.read_csv(D / 'se_accuracy_by_maf_20260924' / 'per_variant.tsv.gz', sep='\t')
    t['sdr'] = t.sd_beta / t.rms_se
    R = {g: (r_allelic(Z['A_all'][i], Z['Va_all'][i]),
             r_total(Z['T_all'][i], Z['Vt_all'][i], Z['C']))
         for i, g in enumerate(Z['all_genes'])}
    for arm, ri in (('hapmixQTL_allelic', 0), ('hapmixQTL_total', 1),
                    ('hapmixQTL', 0)):
        s = t[t.arm == arm].copy()
        s['lsdr'] = np.log(s.sdr)
        gm = s.groupby('gene').sdr.median()
        Rg = np.array([R[g][ri] for g in gm.index])
        sl, se = within_gene_fe_slope(s, 'lsdr', 'maf')
        wsd = s.groupby('gene').lsdr.std(ddof=1)
        rec = dict(n_units=int(len(s)), n_genes=int(s.gene.nunique()),
                   gene_median_vs_sqrtR_spearman=float(sps.spearmanr(gm, Rg).correlation),
                   within_gene_sd_log_sdr_median=float(wsd.median()),
                   gaussian_30perm_log_floor=float(1 / np.sqrt(2 * 29)),
                   maf_fe_slope=sl, maf_fe_slope_se=se,
                   by_maf={f'{lo}-{min(hi, .5)}': float(s[(s.maf >= lo) &
                                                           (s.maf < hi)].sdr.median())
                           for lo, hi in MAF_BINS})
        out[arm] = rec
    # reconciliation: my first-30 permutations vs map_nominal's per variant
    m = V[V.in_tested][['gene', 'variant', 'f30_sd_beta_a', 'f30_mean_se_a',
                        'f30_sd_beta_t', 'f30_mean_se_t']]
    for arm, ch in (('hapmixQTL_allelic', 'a'), ('hapmixQTL_total', 't')):
        s = t[t.arm == arm].merge(m, on=['gene', 'variant'])
        rb = (s[f'f30_sd_beta_{ch}'] / s.sd_beta - 1).abs()
        rs = (s[f'f30_mean_se_{ch}'] / s.mean_se - 1).abs()
        out[f'reconcile_{arm}'] = dict(
            n_matched=int(len(s)), median_rel_sd_beta=float(rb.median()),
            q99_rel_sd_beta=float(rb.quantile(.99)),
            median_rel_mean_se=float(rs.median()),
            q99_rel_mean_se=float(rs.quantile(.99)))
    return out


def analyse_b(G, direct, Z, brng):
    out = {}
    out['n_cache_genes'] = int(len(G))
    P = G[G.n_a >= MIN_NA].copy()
    out['n_genes_min_na'] = int(len(P)); out['min_na'] = MIN_NA
    out['n_genes_na_ge40'] = int((P.n_a >= 40).sum())
    null46 = set(Z['genes'])
    P['in_null46'] = P.gene.isin(null46)
    P['zRa'] = (np.log(P.R_a) - P.lRa_model_mean) / P.lRa_model_sd
    P['zRt'] = (np.log(P.R_t) - P.lRt_model_mean) / P.lRt_model_sd
    P['cov_bin'] = pd.cut(P.med_asc, [b[0] for b in COV_BINS] + [np.inf],
                          right=False, labels=COV_LABEL)
    P['na_bin'] = pd.cut(P.n_a, [b[0] for b in NA_BINS] + [93], right=False,
                         labels=[f'{lo}-{hi - 1}' for lo, hi in NA_BINS])
    # direct-permutation counts aligned to P
    dmap = {gi: (c, n) for gi, c, n in direct}
    cnt = np.stack([dmap[i][0] for i in P.gene_idx])       # G, K, L, A
    nval = np.stack([dmap[i][1] for i in P.gene_idx])      # G, K, L

    def rstats(sub, pre):
        lr = np.log(sub[f'R_{pre}'])
        z = sub[f'zR{pre}']
        return dict(
            n=int(len(sub)), median_R=float(sub[f'R_{pre}'].median()),
            q10_R=float(sub[f'R_{pre}'].quantile(.1)),
            q90_R=float(sub[f'R_{pre}'].quantile(.9)),
            mean_logR=float(lr.mean()),
            mean_model_logR=float(sub[f'lR{pre}_model_mean'].mean()),
            sd_logR=float(lr.std(ddof=1)),
            rms_model_sd_logR=float(np.sqrt((sub[f'lR{pre}_model_sd'] ** 2).mean())),
            sd_ratio=float(lr.std(ddof=1) /
                           np.sqrt((sub[f'lR{pre}_model_sd'] ** 2).mean() +
                                   sub[f'lR{pre}_model_mean'].var(ddof=1))),
            mean_z=float(z.mean()), sd_z=float(z.std(ddof=1)),
            frac_above_q975=float((sub[f'R_{pre}'] > sub[f'R{pre}_model_q975']).mean()),
            frac_below_q025=float((sub[f'R_{pre}'] < sub[f'R{pre}_model_q025']).mean()))

    def dom(sub):
        e = sub.p_dominant_model
        return dict(observed_frac=float((sub.share_max > 0.5).mean()),
                    model_expected_frac=float(e.mean()),
                    model_sd_frac=float(np.sqrt((e * (1 - e)).sum()) / len(sub)),
                    observed_n=int((sub.share_max > 0.5).sum()),
                    frac_share_gt_0p25=float((sub.share_max > 0.25).mean()))

    def direct_rates(mask):
        c, n = cnt[mask], nval[mask]
        rec = {}
        for ai, al in enumerate(ALPHAS_B):
            # pooled over the MAF levels
            rec[f'{al}'] = boot_rate(c[:, 0, :, ai].sum(1), n[:, 0].sum(1), brng,
                                     c[:, 1:, :, ai].sum((1, 2)),
                                     n[:, 1:].sum((1, 2)))
            rec[f'{al}_by_maf'] = {
                str(p): boot_rate(c[:, 0, l, ai], n[:, 0, l], brng,
                                  c[:, 1:, l, ai].sum(1), n[:, 1:, l].sum(1))
                for l, p in enumerate(MAF_LEVELS)}
            # lowest minus highest MAF level, same genes, resampled jointly
            lo_r, hi_r = c[:, 0, 0, ai], c[:, 0, -1, ai]
            lo_n, hi_n = n[:, 0, 0], n[:, 0, -1]
            rec[f'{al}_maf_contrast'] = boot_rate(lo_r, lo_n, brng, hi_r, hi_n)
        return rec

    def smix(sub):
        return {str(al): dict(real=float(sub[f'smix_{al}'].mean()),
                              model=float(sub[f'smix_{al}_model'].mean()),
                              total_real=float(sub[f'smix_t_{al}'].mean()),
                              total_model=float(sub[f'smix_t_{al}_model'].mean()))
                for al in ALPHAS_B}

    def block(sub, mask):
        return dict(R_a=rstats(sub, 'a'), R_t=rstats(sub, 't'), dominance=dom(sub),
                    rho_w_z2=dict(median=float(sub.rho_w_z2.median()),
                                  frac_positive=float((sub.rho_w_z2 > 0).mean())),
                    scale_mixture=smix(sub), direct_permutation=direct_rates(mask))

    for ai, al in enumerate(ALPHAS_B):
        P[f'direct_{al}'] = cnt[:, 0, :, ai].sum(1) / nval[:, 0].sum(1)
        P[f'direct_{al}_model'] = cnt[:, 1:, :, ai].sum((1, 2)) / nval[:, 1:].sum((1, 2))
    allm = np.ones(len(P), bool)
    out['transcriptome'] = block(P, allm)
    # how much of the direct excess the scale mixture on R accounts for,
    # each measured against its own model baseline
    tr = out['transcriptome']
    out['excess_explained_by_scale_mixture'] = {
        str(al): dict(
            direct_excess=tr['direct_permutation'][str(al)]['diff'],
            smix_excess=tr['scale_mixture'][str(al)]['real'] -
            tr['scale_mixture'][str(al)]['model'],
            fraction=(tr['scale_mixture'][str(al)]['real'] -
                      tr['scale_mixture'][str(al)]['model']) /
            tr['direct_permutation'][str(al)]['diff'],
            per_gene_spearman_smix_vs_direct=float(sps.spearmanr(
                P[f'smix_{al}'], P[f'direct_{al}']).correlation))
        for al in ALPHAS_B}
    m46 = P.in_null46.values
    out['null46'] = block(P[m46], m46)
    out['not_null46'] = block(P[~m46], ~m46)
    # the model band: does delta-method sd agree with the simulated band?
    out['delta_vs_model_sd_logR_spearman'] = float(sps.spearmanr(
        P.delta_sd_logR, P.lRa_model_sd).correlation)
    # does R_a predict the per-gene direct sdratio?
    for p in MAF_LEVELS:
        out[f'direct_sdr_vs_sqrtR_spearman_maf{p}'] = float(sps.spearmanr(
            P[f'direct_sdr_a_maf{p}'], P.R_a, nan_policy='omit').correlation)
    out['direct_sdr_over_sqrtR_median'] = {
        str(p): float((P[f'direct_sdr_a_maf{p}'] / np.sqrt(P.R_a)).median())
        for p in MAF_LEVELS}
    # calibration of the 46-gene predictions against the instrument
    inst = json.loads((INST / 'summary.json').read_text())['pooled']['allelic']
    s46 = P[m46]
    inst_t = json.loads((INST / 'summary.json').read_text())['pooled']['total']
    out['calibration_46_total'] = {
        str(al): dict(instrument=inst_t[str(al)]['rate'],
                      scale_mixture=float(s46[f'smix_t_{al}'].mean()),
                      scale_mixture_model=float(s46[f'smix_t_{al}_model'].mean()))
        for al in (0.05, 0.01, 0.001)}
    out['calibration_46'] = {
        str(al): dict(instrument=inst.get(str(al), {}).get('rate'),
                      scale_mixture=float(s46[f'smix_{al}'].mean()),
                      direct_synthetic=out['null46']['direct_permutation'][f'{al}']['rate'])
        for al in (0.05, 0.01, 0.001)}
    # strata
    strata = []
    for col in ('cov_bin', 'na_bin'):
        for lab, sub in P.groupby(col, observed=True):
            mask = (P[col] == lab).values
            if mask.sum() < 20:
                continue
            b = block(sub, mask)
            rec = dict(by=col, bin=str(lab), n_genes=int(len(sub)),
                       median_med_asc=float(sub.med_asc.median()),
                       median_n_a=float(sub.n_a.median()),
                       median_R_a=b['R_a']['median_R'], sd_logR_a=b['R_a']['sd_logR'],
                       sd_z_R_a=b['R_a']['sd_z'], mean_z_R_a=b['R_a']['mean_z'],
                       frac_Ra_above_q975=b['R_a']['frac_above_q975'],
                       frac_Ra_below_q025=b['R_a']['frac_below_q025'],
                       median_R_t=b['R_t']['median_R'], sd_z_R_t=b['R_t']['sd_z'],
                       frac_Rt_above_q975=b['R_t']['frac_above_q975'],
                       frac_Rt_below_q025=b['R_t']['frac_below_q025'],
                       dominant_obs=b['dominance']['observed_frac'],
                       dominant_model=b['dominance']['model_expected_frac'],
                       rho_frac_positive=b['rho_w_z2']['frac_positive'])
            for al in ALPHAS_B:
                d = b['direct_permutation'][f'{al}']
                rec[f'direct_{al}'] = d['rate']; rec[f'direct_{al}_lo'] = d['lo']
                rec[f'direct_{al}_hi'] = d['hi']
                rec[f'direct_{al}_model'] = d['model_rate']
                rec[f'direct_{al}_diff_lo'] = d['diff_lo']
                rec[f'direct_{al}_diff_hi'] = d['diff_hi']
                rec[f'smix_{al}'] = b['scale_mixture'][str(al)]['real']
                rec[f'smix_{al}_model'] = b['scale_mixture'][str(al)]['model']
                rec[f'smix_t_{al}'] = b['scale_mixture'][str(al)]['total_real']
                rec[f'smix_t_{al}_model'] = b['scale_mixture'][str(al)]['total_model']
            strata.append(rec)
    strata = pd.DataFrame(strata)
    strata.to_csv(OUT / 'b_strata.tsv', sep='\t', index=False)
    out['strata_file'] = 'b_strata.tsv'
    P.to_csv(OUT / 'b_per_gene.tsv.gz', sep='\t', index=False)
    G[G.n_a < MIN_NA][['gene', 'n_a', 'med_asc', 'med_tot']].to_csv(
        OUT / 'b_genes_below_min_na.tsv.gz', sep='\t', index=False)
    return out


def without_dominant(Z, L, brng):
    """The instrument's pooled rates with the gene whose single record holds
    the largest share of sum(w z^2) removed (CALM2)."""
    sh = {g: share_max(Z['a'][k], Z['va'][k]) for k, g in enumerate(Z['genes'])}
    top = max(sh, key=sh.get)
    m = L[L.gene != top]
    out = dict(gene_removed=top, share_max=sh[top])
    for ch, pc in (('allelic', 'p_a'), ('total', 'p_t'), ('combined', 'p_b')):
        for al in ALPHAS:
            gs = m.groupby('gene')[pc]
            out[f'{ch}_{al}'] = boot_rate(gs.apply(lambda p: (p < al).sum()),
                                          gs.size(), brng)
    return out


def coupled_genes_by_maf(V):
    """Per-variant allelic sdratio and tail rate by MAF bin in the eight genes
    whose records are most coupled (largest |log R_a|)."""
    T = V[V.in_tested & (V.n_valid >= 100)].copy()
    T['maf_bin'] = pd.cut(T.maf, [b[0] for b in MAF_BINS] + [0.50001], right=False)
    gR = T.groupby('gene').R_a.first()
    pick = list(np.abs(np.log(gR)).sort_values(ascending=False).index[:8])
    rows = []
    for g in pick:
        for mb, sub in T[T.gene == g].groupby('maf_bin', observed=True):
            r = dict(gene=g, R_a=float(gR[g]), maf_bin=str(mb),
                     n_variants=int(len(sub)),
                     median_n_het=float(sub.n_het.median()),
                     median_sdr_a=float(sub.sdr_a_all.median()))
            for al in ALPHAS:
                r[f'rej_a_{al}'] = float(sub[f'rej_a_{al}'].sum() / sub.n_valid.sum())
                r[f'rej_a_{al}_model'] = float(sub[f'rej_a_{al}_model'].sum() /
                                               sub.n_valid_model.sum())
            rows.append(r)
    pd.DataFrame(rows).to_csv(OUT / 'a2_coupled_genes_by_maf.tsv', sep='\t',
                              index=False)
    return 'a2_coupled_genes_by_maf.tsv'


def make_figures(res):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    P = pd.read_csv(OUT / 'b_per_gene.tsv.gz', sep='\t')
    S = pd.read_csv(OUT / 'b_strata.tsv', sep='\t')
    pg = pd.read_csv(OUT / 'a2_per_gene.tsv', sep='\t')
    fig, ax = plt.subplots(2, 2, figsize=(12, 9.5))
    cols = ['#4C72B0', '#55A868', '#C44E52', '#8172B2']
    # A: standardized log R_a against its own model band
    a = ax[0, 0]
    a.hist(np.clip(P.zRa, -12, 12), bins=120, density=True, color=cols[0],
           alpha=.75, label=f'observed, {len(P):,} genes')
    xx = np.linspace(-5, 5, 200)
    a.plot(xx, sps.norm.pdf(xx), color='k', lw=1.2, label='model null, N(0, 1)')
    a.axvline(0, color='grey', lw=.6)
    a.set_xlabel('(log R_a - model mean) / model sd, per gene (clipped at +-12)')
    a.set_ylabel('density')
    a.set_title('A. Weight-residual coupling R_a, every gene with >= 20 '
                'informative donors', fontsize=9.5)
    a.legend(fontsize=8)
    # B: direct records-permutation rate / alpha by MAF level
    a = ax[0, 1]
    dp = res['b']['transcriptome']['direct_permutation']
    for c, al in zip(cols, ALPHAS_B):
        bm = dp[f'{al}_by_maf']
        y = np.array([bm[str(p)]['rate'] for p in MAF_LEVELS]) / al
        lo = np.array([bm[str(p)]['lo'] for p in MAF_LEVELS]) / al
        hi = np.array([bm[str(p)]['hi'] for p in MAF_LEVELS]) / al
        ym = np.array([bm[str(p)]['model_rate'] for p in MAF_LEVELS]) / al
        a.errorbar(MAF_LEVELS, y, yerr=[y - lo, hi - y], color=c, marker='o',
                   capsize=3, label=f'real records, alpha={al:g}')
        a.plot(MAF_LEVELS, ym, color=c, ls=':', marker='x')
    a.axhline(1, color='k', lw=.6)
    a.set_xlabel('MAF of the synthetic variant (Hardy-Weinberg het count)')
    a.set_ylabel('rejection rate / alpha (dotted: model records)')
    a.set_title('B. Allelic channel, direct records permutation, transcriptome',
                fontsize=9.5)
    a.legend(fontsize=7.5)
    # C: by coverage stratum
    a = ax[1, 0]
    sc = S[S.by == 'cov_bin'].reset_index(drop=True)
    x = np.arange(len(sc))
    for k, (c, al) in enumerate(zip(cols, ALPHAS_B)):
        y = sc[f'direct_{al}'] / al
        a.errorbar(x + (k - 1.5) * .12, y,
                   yerr=[y - sc[f'direct_{al}_lo'] / al,
                         sc[f'direct_{al}_hi'] / al - y],
                   fmt='o', color=c, capsize=2, label=f'alpha={al:g}')
    a.axhline(1, color='k', lw=.6)
    a.set_xticks(x)
    a.set_xticklabels([f'{b}\n{n:,} genes' for b, n in zip(sc.bin, sc.n_genes)],
                      fontsize=7.5)
    a.set_xlabel('median allele-resolved reads per informative donor')
    a.set_ylabel('rejection rate / alpha (real records)')
    a.set_title('C. Allelic channel by coverage stratum (model records sit at '
                '1 x alpha)', fontsize=9.5)
    a.legend(fontsize=7.5)
    # D: a2 gene-level sdratio vs sqrt(R_a)
    a = ax[1, 1]
    a.scatter(np.sqrt(pg.R_a), pg.median_sdr_a, s=18, color=cols[0],
              label='real records (median over variants)')
    a.scatter(np.sqrt(pg.R_a), pg.median_sdr_a_model, s=10, color='grey',
              marker='x', label='model records, same weights')
    lim = [0.7, 2.2]
    a.plot(lim, lim, color='k', lw=.6)
    for _, r in pg[(pg.R_a > 2) | (pg.R_a < 0.7)].iterrows():
        a.annotate(r.gene, (np.sqrt(r.R_a), r.median_sdr_a), fontsize=7)
    a.set_xscale('log'); a.set_yscale('log')
    a.set_xlabel('sqrt(R_a), from the records alone')
    a.set_ylabel('sd(slope) / rms(se) over 500 permutations')
    a.set_title(f'D. 59 genes x every tested variant '
                f'({res["a2"]["n_units"]:,} gene-variant units)', fontsize=9.5)
    a.legend(fontsize=7.5)
    fig.tight_layout()
    f = OUT / 'coupling_reach.png'
    fig.savefig(f, dpi=130)
    plt.close(fig)
    return [str(f)]


if __name__ == '__main__':
    main()

"""Like-for-like calibration of hapmixQTL, the mixQTL port and RASQUAL under the
shared records null, at 2,000 permutations where the permutation is ours.

QUESTION. hapmixQTL's nominal p is anticonservative under the donor-record
permutation null at a fixed variant (0.068 at nominal 0.05 for the combined
statistic, nominal_p_null_instrument_20260925). The 2026-09-24 comparison put
mixQTL at 0.067 and RASQUAL at 0.044, but those came from 30 permutations, the
mixQTL figure from one cutoff setting and one reference distribution, and the
RASQUAL figure from a different null. This script re-establishes the three-way
comparison on common ground and decomposes what keeps mixQTL closer to nominal.

DESIGN. 46 genes x 92 donors, fixed variant = RASQUAL's observed lead, the
arrays of inputs_at_lead.npz. The permutation stream is the instrument's:
rng = RandomState(42); perms = [rng.permutation(92) for _ in range(2000)].
Every arm below that we run sees exactly those 2,000 permutations, so any two
arms are PAIRED at the level of (gene, permutation) and a difference between
them is measured against its own paired, gene-clustered noise floor.

  (a) mixQTL PORT (tensorqtl/mixqtl_replication.py, run through mixqtl_scan
      unchanged) on Salmon posterior-mean counts, donor records permuted
      exactly as realized_variance.py did (y1, y2, ytotal, lib_size and the
      covariate row move together; h1 = g + s/2 and h2 = g - s/2 stay).
      Held fixed: genes, variant, donors, permutations. Varied: cutoffs
      (published 100/50/10/1000 vs permissive R-signature 20/5/100/5000) and
      the reference distribution (mixQTL's own normal-when-n>15 vs an F/t
      reference with each channel's regression dof: asc n_a-1, trc n_t-2,
      meta min of the two). The meta/trc/asc fallback that meta_analyze takes
      is recorded per gene, because under the published cutoffs most genes
      fall below 15 allelic donors and "meta" becomes total-only.

  (b) DECOMPOSITION ON hapmixQTL's OWN RECORDS (a, va, t, vt, C from the
      npz; posterior-mean mL, mR, mT from the Gibbs cache). One component of
      mixQTL is swapped in at a time, over the same 2,000 permutations (the
      allelic channel vectorized, the total channel looped with resid_out):
        cap        weights 1/va fold-capped at min(10, floor(n_a/10)) x min
        band       donors restricted to mixQTL's allelic count band on mL, mR
                   (published [50,1000]; permissive [5,5000]); weights 1/va
        weight     harmonic 1/(1/mL + 1/mR), uncapped, on the same donors
        response   log(mL/mR) in place of hapmixQTL's a, weights 1/va
        reference  hapmixQTL's own t^2 against chi2(1) instead of F(1, dof)
        OLS        equal weights (the non-weighting baseline under this null)
      and for the total channel: OLS, reference, mixQTL's selected-covariate
      offset in place of 17 partialled covariates, and the trc>=100 cutoff.
      Each swap that changes the donor set is compared against the base arm
      recomputed on the SAME donor set and gene subset.

      For every allelic arm the coupling ratio of its own weights w and
      response y is computed per gene over its admitted records,
          R_g = n * sum(w^2 y^2) / (sum(w) * sum(w y^2)),
      the first-order value of sd(beta)^2 / mean(se^2) under a records
      permutation (it is 1 identically for equal weights). The total-channel
      analogue replaces y by the weighted covariate residual. Whether R_g
      predicts the arm's per-gene sdratio is tested per arm; the OLS arm's
      sdratio spread across genes is the Monte Carlo control.

  (c) RASQUAL. What -r permutes is read from rasqual_src/src (randomPerm in
      nbem.c, getRandomOrder in sort.c) and stated in the summary. Its null
      rows at the fixed variant are re-read from
      rasqual_default_mode_20260923/rasqual_rows/null_NNN/<gene>.tsv (field
      layout of compare_pipelines.best_rasqual_row: 0-based 10 chisq, 11 pi,
      14 theta, 16 n_feature_snps, 22 convergence), converged vs not. The
      per-gene comparison against hapmixQTL's R_g has at most 30 draws per
      gene, so its power is measured by running the identical comparison on
      hapmixQTL's own statistic subsampled to RASQUAL's per-gene draw counts.

  (d) the corrected three-way table, with the CLAUDE.md figures it supersedes.

GATES (the script aborts if any fails):
  A. the permissive mixQTL arm's first 30 permutations reproduce
     realized_variance_20260924/null_long.tsv (arm mixQTL, the source of the
     recorded 0.067) per (gene, perm) to |dp| <= 1e-9, and its pooled 0.05
     rate exactly (93/1380). The published arm has no recorded figure and so
     no such gate.
  B. the base arms of (b) (allelic vectorized; total looped with the
     instrument's resid_out) reproduce the instrument's per-(gene, perm)
     t2_a, t2_t and t2_b to relative 1e-9 on all 92,000 rows, and its pooled
     rates at 0.05/0.01/0.001 exactly.
  C. the vectorized mixQTL-style channel algebra is not used for (a); (a) is
     the port itself. The h1/h2 reconstruction is checked: h1, h2 in {0,1},
     h1 - h2 == s, (h1 + h2)/2 == g.

Master seed 42. The permutation stream is the instrument's RandomState(42);
bootstraps and the power simulation use child streams of SeedSequence(42).
Outputs: /mnt/ssd/lalli/brainvar_hapmix_deploy/comparator_null_2000_20260925/
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import json                                                     # noqa: E402
import sys                                                      # noqa: E402
import time                                                     # noqa: E402
from multiprocessing import Pool                                # noqa: E402
from pathlib import Path                                        # noqa: E402

import numpy as np                                              # noqa: E402
import pandas as pd                                             # noqa: E402
from scipy import stats as sps                                  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from tensorqtl import mixqtl_replication as MX                  # noqa: E402
from se_fixes import resid_out                                   # noqa: E402

D = Path('/mnt/ssd/lalli/brainvar_hapmix_deploy')
INST = D / 'nominal_p_null_instrument_20260925'
RUN = D / 'rasqual_default_mode_20260923'
CACHE = D / 'cache' / 'gibbs_56b63c3b37ed5df8'
OUT = D / 'comparator_null_2000_20260925'
SEED, EPS, N_PERM, N_BOOT, N_POWER = 42, 1e-12, 2000, 2000, 1000
ALPHAS = (0.05, 0.01, 0.001)
N_RASQUAL_ROUNDS = 30
CUTS = {'published': MX.PUBLISHED_CUTOFFS, 'permissive': MX.PACKAGE_DEFAULT_CUTOFFS}

log = lambda *a: print(time.strftime('%H:%M:%S'), *a, flush=True)
SS = np.random.SeedSequence(SEED).spawn(8)      # child streams, fixed roles
RNG_BOOT = np.random.default_rng(SS[0])
RNG_POWER = np.random.default_rng(SS[1])
RNG_MC = np.random.default_rng(SS[2])


# ---------------------------------------------------------------------------
#  inputs
# ---------------------------------------------------------------------------

def load():
    Z = np.load(INST / 'inputs_at_lead.npz', allow_pickle=True)
    genes = [str(x) for x in Z['genes']]
    donors = [str(x) for x in Z['donors']]
    ga = open(CACHE / 'genes.txt').read().split()
    samples = open(CACHE / 'samples.txt').read().split()
    keep = [samples.index(d) for d in donors]           # by id, never position
    rows = [ga.index(g) for g in genes]
    post = {}
    for k in ('YL', 'YR', 'YT'):
        mm = np.load(CACHE / f'{k}.npy', mmap_mode='r')
        post[k] = np.asarray(mm[rows])[:, keep].mean(-1)
    rng = np.random.RandomState(SEED)
    perms = np.stack([rng.permutation(len(donors)) for _ in range(N_PERM)])
    inst = pd.read_csv(INST / 'null_long.tsv.gz', sep='\t')
    return dict(Z=Z, genes=genes, donors=donors, post=post, perms=perms,
                inv=np.argsort(perms, axis=1), inst=inst,
                a=Z['a'], va=Z['va'], t=Z['t'], vt=Z['vt'], s=Z['s'], g=Z['g'],
                C=Z['C'], lib=Z['lib_size'],
                variants=[str(x) for x in Z['variants']],
                strata=[str(x) for x in Z['strata']])


# ---------------------------------------------------------------------------
#  pooled rates, gene-clustered intervals, paired differences
# ---------------------------------------------------------------------------

def _boot(k, n):
    G = len(k)
    idx = RNG_BOOT.integers(0, G, size=(N_BOOT, G))
    with np.errstate(invalid='ignore', divide='ignore'):
        b = k[idx].sum(1) / n[idx].sum(1)
    return float(np.nanquantile(b, 0.025)), float(np.nanquantile(b, 0.975))


def rate(P, alpha):
    """P: genes x perms p-values (NaN = undefined). Pooled rate + gene CI."""
    ok = np.isfinite(P)
    k = ((P < alpha) & ok).sum(1).astype(float)
    n = ok.sum(1).astype(float)
    keepg = n > 0
    k, n = k[keepg], n[keepg]
    est = float(k.sum() / n.sum()) if n.sum() else np.nan
    lo, hi = _boot(k, n) if n.sum() else (np.nan, np.nan)
    return dict(rate=est, lo=lo, hi=hi, k=int(k.sum()), n=int(n.sum()),
                genes=int(keepg.sum()))


def diff(P1, P0, alpha):
    """Paired difference of rejection rates, arm1 - arm0, on (gene, perm)
    rows where both are defined; gene-clustered bootstrap of the difference."""
    ok = np.isfinite(P1) & np.isfinite(P0)
    d = ((P1 < alpha).astype(float) - (P0 < alpha).astype(float)) * ok
    k = d.sum(1)
    n = ok.sum(1).astype(float)
    g = n > 0
    k, n = k[g], n[g]
    est = float(k.sum() / n.sum())
    lo, hi = _boot(k, n)
    return dict(diff=est, lo=lo, hi=hi, n=int(n.sum()), genes=int(g.sum()),
                clears_floor=bool(lo > 0 or hi < 0))


def rates3(P):
    return {str(a): rate(P, a) for a in ALPHAS}


def diffs3(P1, P0):
    return {str(a): diff(P1, P0, a) for a in ALPHAS}


def sdratio(B, SE):
    """Per gene: sd over perms of beta / rms reported se."""
    out = np.full(B.shape[0], np.nan)
    for i in range(B.shape[0]):
        ok = np.isfinite(B[i]) & np.isfinite(SE[i])
        if ok.sum() > 100:
            out[i] = np.std(B[i][ok], ddof=1) / np.sqrt(np.mean(SE[i][ok] ** 2))
    return out


def sdratio_mc_sd(B, SE, n_rep=200):
    """Monte Carlo sd of each gene's sdratio, by resampling permutations."""
    out = np.full(B.shape[0], np.nan)
    for i in range(B.shape[0]):
        ok = np.isfinite(B[i]) & np.isfinite(SE[i])
        b, s = B[i][ok], SE[i][ok]
        if len(b) <= 100:
            continue
        idx = RNG_MC.integers(0, len(b), size=(n_rep, len(b)))
        out[i] = np.std(np.std(b[idx], axis=1, ddof=1) /
                        np.sqrt(np.mean(s[idx] ** 2, axis=1)), ddof=1)
    return out


# ---------------------------------------------------------------------------
#  (a) the mixQTL port, per gene, all permutations
# ---------------------------------------------------------------------------

def _mixqtl_gene(args):
    i, cut, y1, y2, yt, lib, h1, h2, C, perms = args
    cfg = CUTS[cut]
    out = {k: np.full(len(perms), np.nan) for k in
           ('b_trc', 'se_trc', 'p_trc', 'b_asc', 'se_asc', 'p_asc',
            'b_meta', 'se_meta', 'p_meta')}
    meth, n_a, n_t, n_sel = [], None, None, None
    for p, prm in enumerate(perms):
        o = MX.mixqtl_scan(y1[prm], y2[prm], yt[prm], lib[prm], h1, h2,
                           covariates=C[prm], **cfg)
        for ch, key in (('trc', 'trc'), ('asc', 'asc'), ('meta', 'meta')):
            out[f'b_{ch}'][p] = o[key]['beta'][0]
            out[f'se_{ch}'][p] = o[key]['se'][0]
            out[f'p_{ch}'][p] = o[key]['pval'][0]
        meth.append(str(o['meta']['method'][0]))
        n_a, n_t = int(o['asc']['sample_size']), int(o['trc']['sample_size'])
        n_sel = int(o['cov_selected'].sum()) if o['cov_selected'] is not None else 0
    out['method'] = np.array(meth)
    out.update(i=i, cut=cut, n_a=n_a, n_t=n_t, n_sel=n_sel)
    return out


def run_mixqtl(X):
    y1, y2, yt = X['post']['YL'], X['post']['YR'], X['post']['YT']
    h1_all = X['g'] + X['s'] / 2.0
    h2_all = X['g'] - X['s'] / 2.0
    # gate C: the reconstruction of the phased haplotypes
    assert np.all(np.isin(h1_all, (0.0, 1.0))) and np.all(np.isin(h2_all, (0.0, 1.0)))
    assert np.allclose(h1_all - h2_all, X['s']) and np.allclose((h1_all + h2_all) / 2, X['g'])
    tasks = [(i, cut, y1[i], y2[i], yt[i], X['lib'], h1_all[i][:, None],
              h2_all[i][:, None], X['C'], X['perms'])
             for cut in CUTS for i in range(len(X['genes']))]
    with Pool(min(24, len(tasks))) as pool:
        res = pool.map(_mixqtl_gene, tasks, chunksize=1)
    G = len(X['genes'])
    M = {}
    for cut in CUTS:
        R = sorted([r for r in res if r['cut'] == cut], key=lambda r: r['i'])
        d = {k: np.stack([r[k] for r in R]) for k in
             ('b_trc', 'se_trc', 'p_trc', 'b_asc', 'se_asc', 'p_asc',
              'b_meta', 'se_meta', 'p_meta', 'method')}
        d['n_a'] = np.array([r['n_a'] for r in R])
        d['n_t'] = np.array([r['n_t'] for r in R])
        d['n_sel'] = np.array([r['n_sel'] for r in R])
        # F/t reference with each channel's regression dof
        dof_a = np.maximum(d['n_a'] - 1, 1)[:, None] * np.ones((1, N_PERM))
        dof_t = np.maximum(d['n_t'] - 2, 1)[:, None] * np.ones((1, N_PERM))
        dof_tc = np.maximum(d['n_t'] - 2 - d['n_sel'], 1)[:, None] * np.ones((1, N_PERM))
        with np.errstate(invalid='ignore', divide='ignore'):
            ta2 = (d['b_asc'] / d['se_asc']) ** 2
            tt2 = (d['b_trc'] / d['se_trc']) ** 2
            tm2 = (d['b_meta'] / d['se_meta']) ** 2
        d['pF_asc'] = sps.f.sf(ta2, 1, dof_a)
        d['pF_trc'] = sps.f.sf(tt2, 1, dof_t)
        d['pF_trc_charged'] = sps.f.sf(tt2, 1, dof_tc)
        dm = np.where(d['method'] == 'meta', np.minimum(dof_a, dof_t),
                      np.where(d['method'] == 'asc', dof_a, dof_t))
        dmc = np.where(d['method'] == 'meta', np.minimum(dof_a, dof_tc),
                       np.where(d['method'] == 'asc', dof_a, dof_tc))
        d['pF_meta'] = sps.f.sf(tm2, 1, dm)
        d['pF_meta_charged'] = sps.f.sf(tm2, 1, dmc)
        # the coupling ratio of mixQTL's OWN allelic weights on its own records
        Rg = np.full(G, np.nan)
        for i in range(G):
            a1, a2 = y1[i], y2[i]
            ps = ((a1 >= CUTS[cut]['asc_cutoff']) & (a2 >= CUTS[cut]['asc_cutoff'])
                  & (a1 <= CUTS[cut]['asc_cap']) & (a2 <= CUTS[cut]['asc_cap']))
            n = int(ps.sum())
            if n <= 2:
                continue
            w, _, _ = MX.apply_weight_cap(MX.harmonic_weights(a1[ps], a2[ps]), n,
                                          CUTS[cut]['weight_cap'])
            y = np.log(a1[ps] / a2[ps])
            if w.sum() > 0:
                Rg[i] = n * np.sum(w ** 2 * y ** 2) / (w.sum() * np.sum(w * y ** 2))
        d['R_asc'] = Rg
        M[cut] = d
    return M


def gate_mixqtl(X, M):
    rec = pd.read_csv(D / 'realized_variance_20260924' / 'null_long.tsv', sep='\t')
    rec = rec[rec.arm == 'mixQTL']
    gi = {g: i for i, g in enumerate(X['genes'])}
    mine = M['permissive']['p_meta']
    dp = np.array([abs(mine[gi[r.gene], r.perm] - r.pval) for r in rec.itertuples()])
    first30 = mine[:, :30]
    k = int((first30 < 0.05).sum())
    n = int(np.isfinite(first30).sum())
    k_rec = int((rec.pval < 0.05).sum())
    log(f'gate A: {len(rec)} recorded (gene, perm) rows, max |dp| {dp.max():.2e}; '
        f'first-30 rate {k}/{n} vs recorded {k_rec}/{len(rec)}')
    if len(rec) != n or dp.max() > 1e-9 or k != k_rec:
        raise SystemExit('gate A FAILED: mixQTL port does not reproduce the recorded 30-draw null')
    return dict(n_rows=len(rec), max_abs_dp=float(dp.max()), first30_k=k, first30_n=n,
                first30_rate=k / n)


# ---------------------------------------------------------------------------
#  (b) vectorized channel fits on hapmixQTL's records
# ---------------------------------------------------------------------------

def allelic_fit(y, w, K, s, inv):
    """Weighted LS through the origin, records (y, w, K) permuted against
    fixed s. Returns beta, se, t2, dof arrays over permutations, and R_g."""
    K = K & np.isfinite(y) & np.isfinite(w) & (w > 0)
    wk = np.where(K, w, 0.0)
    yk = np.where(K, y, 0.0)
    n = int(K.sum())
    if n < 6:
        nan = np.full(inv.shape[0], np.nan)
        return nan, nan, nan, np.nan, np.nan, n
    Sinv = s[inv]
    num = Sinv @ (wk * yk)
    den = (Sinv ** 2) @ wk
    S = float(np.sum(wk * yk ** 2))
    dof = max(n - 1, 1)
    with np.errstate(invalid='ignore', divide='ignore'):
        b = num / den
        rss = S - num ** 2 / den
        se2 = rss / dof / den
        t2 = dof * num ** 2 / (den * S - num ** 2)
    R = n * np.sum(wk ** 2 * yk ** 2) / (wk.sum() * S)
    return b, np.sqrt(se2), t2, dof, R, n


def total_fit(y, w, K, Zrec, gpos, perms, charge=None):
    """Weighted LS with the record-space design Zrec partialled in the
    sqrt(w)-weighted space; Zrec moves with the record, gpos stays. Looped
    over permutations with the instrument's own operations (resid_out per
    permutation), because a record-space short-cut loses relative accuracy
    on near-zero statistics: the weighted response carries a large intercept
    component, so its residual is a small difference of large numbers."""
    K = K & np.isfinite(y) & np.isfinite(w) & (w > 0)
    n = int(K.sum())
    dof = max(n - Zrec.shape[1] - 1, 1) if charge is None else max(n - charge, 1)
    P = len(perms)
    b, se2 = np.full(P, np.nan), np.full(P, np.nan)
    for p, prm in enumerate(perms):
        kt = K[prm]
        wt = w[prm][kt]
        Zm = Zrec[prm][kt]
        yt = resid_out(y[prm][kt][:, None], Zm, wt).ravel()
        xt = resid_out(gpos[kt][:, None], Zm, wt).ravel()
        xxt = float(xt @ xt)
        if xxt <= 0:
            continue
        b[p] = float(xt @ yt) / xxt
        et = yt - b[p] * xt
        se2[p] = float(et @ et) / dof / xxt
    with np.errstate(invalid='ignore', divide='ignore'):
        t2 = b ** 2 / se2
    # coupling ratio on the record-space covariate residual
    sw = np.sqrt(np.where(K, w, 0.0))
    e = resid_out(np.where(K, y, 0.0)[:, None], Zrec, np.where(K, w, 0.0)).ravel()
    wr = np.where(K, w, 0.0)
    r = np.where(K, e / np.where(sw > 0, sw, 1.0), 0.0)     # unweighted-scale residual
    R = n * np.sum(wr ** 2 * r ** 2) / (wr.sum() * np.sum(wr * r ** 2))
    return b, np.sqrt(se2), t2, dof, R, n


def meta(fa, ft):
    ba, sea, _, dofa = fa[:4]
    bt, set_, _, doft = ft[:4]
    with np.errstate(invalid='ignore', divide='ignore'):
        prec = 1 / sea ** 2 + 1 / set_ ** 2
        b = (ba / sea ** 2 + bt / set_ ** 2) / prec
        t2 = b ** 2 * prec
    return b, np.sqrt(1 / prec), t2, min(dofa, doft)


def pF(t2, dof):
    return sps.f.sf(t2, 1, dof)


def pN(t2):
    return sps.chi2.sf(t2, 1)


def run_arms(X):
    G = len(X['genes'])
    a, va, t, vt, s, g, C = (X[k] for k in ('a', 'va', 't', 'vt', 's', 'g', 'C'))
    mL, mR, mT = X['post']['YL'], X['post']['YR'], X['post']['YT']
    inv = X['inv']
    Zrec = np.column_stack([np.ones(len(X['donors'])), C])
    arms = {}          # name -> dict(b, se, p [G x P], R [G], n [G], chan)

    def put(name, chan, i, fit, ref='F'):
        b, se, t2, dof, R, n = fit
        if name not in arms:
            arms[name] = dict(chan=chan, b=np.full((G, N_PERM), np.nan),
                              se=np.full((G, N_PERM), np.nan),
                              t2=np.full((G, N_PERM), np.nan),
                              p=np.full((G, N_PERM), np.nan),
                              R=np.full(G, np.nan), n=np.zeros(G, int),
                              dof=np.full(G, np.nan))
        A = arms[name]
        if np.all(np.isnan(np.atleast_1d(t2))):
            return
        A['b'][i], A['se'][i], A['t2'][i] = b, se, t2
        A['p'][i] = pF(t2, dof) if ref == 'F' else pN(t2)
        A['R'][i], A['n'][i], A['dof'][i] = R, n, dof

    for i in range(G):
        adm = np.isfinite(a[i]) & np.isfinite(va[i]) & (va[i] > EPS)
        w0 = np.where(adm, 1.0 / np.where(adm, va[i], 1.0), 0.0)
        pos = adm & (mL[i] > 0) & (mR[i] > 0)
        with np.errstate(divide='ignore', invalid='ignore'):
            harm = np.where(pos, MX.harmonic_weights(mL[i], mR[i]), 0.0)
            lr = np.where(pos, np.log(mL[i] / mR[i]), 0.0)

        # ---------------- allelic ----------------
        fa0 = allelic_fit(a[i], w0, adm, s[i], inv)
        put('a_base', 'a', i, fa0)
        put('a_ref_normal', 'a', i, fa0, ref='N')
        put('a_ols', 'a', i, allelic_fit(a[i], adm.astype(float), adm, s[i], inv))
        n_a = int(adm.sum())
        cap = min(MX.PUBLISHED_CUTOFFS['weight_cap'], np.floor(n_a / 10))
        wc, _, _ = MX.apply_weight_cap(w0[adm], n_a, MX.PUBLISHED_CUTOFFS['weight_cap'])
        wcap = np.zeros_like(w0); wcap[adm] = wc
        fcap = allelic_fit(a[i], wcap, adm, s[i], inv)
        put('a_cap', 'a', i, fcap)
        # donor set where harmonic weights and log(mL/mR) exist
        put('a_base_pos', 'a', i, allelic_fit(a[i], w0, pos, s[i], inv))
        put('a_harm', 'a', i, allelic_fit(a[i], harm, pos, s[i], inv))
        n_p = int(pos.sum())
        if n_p >= 6:
            hc, _, _ = MX.apply_weight_cap(harm[pos], n_p, MX.PUBLISHED_CUTOFFS['weight_cap'])
            hcap = np.zeros_like(harm); hcap[pos] = hc
            put('a_harm_cap', 'a', i, allelic_fit(a[i], hcap, pos, s[i], inv))
            put('a_mixall_noband', 'a', i, allelic_fit(lr, hcap, pos, s[i], inv), ref='N')
        put('a_response', 'a', i, allelic_fit(lr, w0, pos, s[i], inv))
        # count bands, matched subsets (>= 15 admitted, mixQTL's META_N_CUTOFF)
        for nm, cfg in CUTS.items():
            lo, hi = cfg['asc_cutoff'], cfg['asc_cap']
            band = adm & (mL[i] >= lo) & (mR[i] >= lo) & (mL[i] <= hi) & (mR[i] <= hi)
            if band.sum() >= MX.META_N_CUTOFF:
                put(f'a_band_{nm}', 'a', i, allelic_fit(a[i], w0, band, s[i], inv))
                put(f'a_base_on_{nm}_genes', 'a', i, fa0)

        # ---------------- total ----------------
        admt = np.isfinite(t[i]) & np.isfinite(vt[i]) & (vt[i] > EPS)
        wt0 = np.where(admt, 1.0 / np.where(admt, vt[i], 1.0), 0.0)
        ft0 = total_fit(t[i], wt0, admt, Zrec, g[i], X['perms'])
        put('t_base', 't', i, ft0)
        put('t_ref_normal', 't', i, ft0, ref='N')
        ft_ols = total_fit(t[i], admt.astype(float), admt, Zrec, g[i], X['perms'])
        put('t_ols', 't', i, ft_ols)
        # mixQTL's covariate treatment: selected-covariate offset (|t|>2,
        # unweighted, on all donors) subtracted, then intercept only, dof n-2.
        # covariate_offset regresses log(ytotal/lib/2); feeding it
        # ytotal = 2*lib*exp(t) makes that left-hand side exactly t.
        off, _sel = MX.covariate_offset(2.0 * X['lib'] * np.exp(t[i]), X['lib'], C)
        ft_cov = total_fit(t[i] - off, wt0, admt, np.ones((len(t[i]), 1)), g[i], X['perms'])
        put('t_covsel', 't', i, ft_cov)
        # the same covariate treatment at equal weights, so it is compared
        # against t_ols and isolates the covariate handling from the weights
        put('t_covsel_ols', 't', i, total_fit(t[i] - off, admt.astype(float), admt,
                                              np.ones((len(t[i]), 1)), g[i], X['perms']))
        band_t = admt & (mT[i] >= MX.PUBLISHED_CUTOFFS['trc_cutoff'])
        ft_band = total_fit(t[i], wt0, band_t, Zrec, g[i], X['perms'])
        put('t_band_published', 't', i, ft_band)

        # ---------------- combined (inverse-variance meta, F(1, min dof)) ----
        def comb(name, fa, ft, ref='F'):
            if np.all(np.isnan(np.atleast_1d(fa[2]))) or np.all(np.isnan(np.atleast_1d(ft[2]))):
                return
            b, se, t2, dof = meta(fa, ft)
            put(name, 'b', i, (b, se, t2, dof, np.nan, 0), ref=ref)
        comb('b_base', fa0, ft0)
        comb('b_ref_normal', fa0, ft0, ref='N')
        comb('b_cap', fcap, ft0)
        comb('b_weightfamily', allelic_fit(a[i], harm, pos, s[i], inv), ft_ols)
        comb('b_base_pos', allelic_fit(a[i], w0, pos, s[i], inv), ft0)
        comb('b_ols_both', allelic_fit(a[i], adm.astype(float), adm, s[i], inv), ft_ols)
        comb('b_cap_totols', fcap, ft_ols)
        band = adm & (mL[i] >= 50) & (mR[i] >= 50) & (mL[i] <= 1000) & (mR[i] <= 1000)
        if band.sum() >= MX.META_N_CUTOFF:
            comb('b_band_published', allelic_fit(a[i], w0, band, s[i], inv), ft_band)
            comb('b_base_on_published_genes', fa0, ft0)
    return arms


def gate_arms(X, arms):
    inst = X['inst']
    gi = {g: i for i, g in enumerate(X['genes'])}
    ii = inst.gene.map(gi).values
    pp = inst.perm.values
    out = {}
    for arm, col, pcol in (('a_base', 't2_a', 'p_a'), ('t_base', 't2_t', 'p_t'),
                           ('b_base', 't2_b', 'p_b')):
        mine = arms[arm]['t2'][ii, pp]
        ref = inst[col].values
        rel = np.abs(mine - ref) / np.abs(ref)
        rates_ok = all(int((arms[arm]['p'] < a_).sum()) == int((inst[pcol] < a_).sum())
                       for a_ in ALPHAS)
        out[arm] = dict(n=len(ref), max_rel=float(np.nanmax(rel)),
                        n_nonfinite=int((~np.isfinite(mine)).sum()),
                        rates_exact=rates_ok)
        log(f'gate B {arm}: {len(ref)} rows, max rel |dt2| {np.nanmax(rel):.2e}, '
            f'pooled counts identical at 0.05/0.01/0.001: {rates_ok}')
        if len(ref) != G_TOTAL(X) or np.nanmax(rel) > 1e-9 or \
                (~np.isfinite(mine)).any() or not rates_ok:
            raise SystemExit(f'gate B FAILED on {arm}')
    return out


def G_TOTAL(X):
    return len(X['genes']) * N_PERM


# ---------------------------------------------------------------------------
#  (c) RASQUAL
# ---------------------------------------------------------------------------

def read_rasqual(X):
    rows, obs = [], []
    for gene, var in zip(X['genes'], X['variants']):
        chrom, pos, ref, alt = var.split('_')
        f0 = RUN / 'rasqual_rows' / f'{gene}.tsv'
        nf_obs = np.nan
        if f0.exists():
            for ln in f0.read_text(errors='replace').split('\n'):
                x = ln.split('\t')
                if len(x) >= 25 and x[2] == chrom and x[3] == pos and x[4] == ref and x[5] == alt:
                    nf_obs = float(x[16]); break
        obs.append(dict(gene=gene, n_fsnp_obs=nf_obs))
        for d in range(N_RASQUAL_ROUNDS):
            f = RUN / 'rasqual_rows' / f'null_{d:03d}' / f'{gene}.tsv'
            if not f.exists():
                continue
            hit = None
            for ln in f.read_text(errors='replace').split('\n'):
                x = ln.split('\t')
                if len(x) >= 25 and x[2] == chrom and x[3] == pos and x[4] == ref and x[5] == alt:
                    hit = x; break
            if hit is None:
                continue
            try:
                chi2 = float(hit[10]); conv = int(float(hit[22]))
                nf = float(hit[16]); theta = float(hit[14]); pi = float(hit[11])
            except ValueError:
                continue
            rows.append(dict(gene=gene, round=d, chi2=chi2, p=float(sps.chi2.sf(chi2, 1)),
                             convergence=conv, converged=conv == 0, n_fsnp=nf,
                             theta=theta, pi=pi))
    return pd.DataFrame(rows), pd.DataFrame(obs)


def to_matrix(df, genes, col, mask=None):
    M = np.full((len(genes), N_RASQUAL_ROUNDS), np.nan)
    gi = {g: i for i, g in enumerate(genes)}
    sub = df if mask is None else df[mask]
    for r in sub.itertuples():
        M[gi[r.gene], r.round] = getattr(r, col)
    return M


def rasqual_power(X, arms, rq, Rg):
    """Power of 'per-gene RASQUAL rejection vs hapmixQTL R_g' at RASQUAL's own
    per-gene draw counts, measured on hapmixQTL's own statistic."""
    conv = rq[rq.converged]
    n_g = conv.groupby('gene').size().reindex(X['genes']).fillna(0).astype(int).values
    ok = (n_g >= 10) & np.isfinite(Rg)
    res = {}
    for arm in ('b_base', 'a_base'):
        P = arms[arm]['p']
        T2 = arms[arm]['t2']
        full_rej = (P < 0.05).mean(1)
        rho_full = sps.spearmanr(Rg[ok], full_rej[ok]).correlation
        rho_full_t2 = sps.spearmanr(Rg[ok], T2.mean(1)[ok]).correlation
        rr, rt, sig_r, sig_t = [], [], 0, 0
        for _ in range(N_POWER):
            rej = np.full(len(n_g), np.nan)
            mt2 = np.full(len(n_g), np.nan)
            for i in np.where(ok)[0]:
                idx = RNG_POWER.choice(N_PERM, size=n_g[i], replace=False)
                rej[i] = (P[i, idx] < 0.05).mean()
                mt2[i] = T2[i, idx].mean()
            s1 = sps.spearmanr(Rg[ok], rej[ok]); s2 = sps.spearmanr(Rg[ok], mt2[ok])
            rr.append(s1.correlation); rt.append(s2.correlation)
            sig_r += (s1.pvalue < 0.05) and (s1.correlation > 0)
            sig_t += (s2.pvalue < 0.05) and (s2.correlation > 0)
        res[arm] = dict(genes=int(ok.sum()), rho_2000perm_rej05=float(rho_full),
                        rho_2000perm_meant2=float(rho_full_t2),
                        rej05_rho_median=float(np.nanmedian(rr)),
                        rej05_rho_q05_q95=[float(np.nanquantile(rr, .05)),
                                           float(np.nanquantile(rr, .95))],
                        rej05_power=sig_r / N_POWER,
                        meant2_rho_median=float(np.nanmedian(rt)),
                        meant2_rho_q05_q95=[float(np.nanquantile(rt, .05)),
                                            float(np.nanquantile(rt, .95))],
                        meant2_power=sig_t / N_POWER)
    return res, n_g


RASQUAL_NULL_TEXT = (
    "rasqual -r (main.c:213 sets randomize; main.c:646 calls randomPerm in "
    "nbem.c:2346 once per gene, after the null GLM fit). randomPerm draws a FRESH "
    "uniform order (getRandomOrder, sort.c:251: qsort on rand() keys, seeded by "
    "time+pid at main.c:209) for EACH feature SNP separately, and moves that fSNP's "
    "allele-specific counts Y, its phased genotype posteriors Z (into a copy Zr) and "
    "its per-fSNP offset copies ki/ki2 together under that order. It then draws ONE "
    "MORE independent order for the total count y, the offset ki, dki and the ZINB "
    "weights; dki is the null GLM's fitted mean exp(X beta + log ki) (nbglm.c:389), "
    "so the covariate FIT travels with the total count while the covariate matrix X "
    "itself is not passed to randomPerm and stays with the genotype position. The "
    "haplotype swap is commented out (nbem.c:2367, rord12 fixed at {0,1}), so there "
    "is no allele flip. The rSNP genotype is NOT permuted. So: "
    "(1) total and allelic evidence are permuted by independent orders, whereas the "
    "records null moves them together; (2) a donor's allelic evidence at different "
    "fSNPs is sent to different donors, so one donor's extreme allelic record is "
    "split across up to n_fSNP recipients rather than landing whole on one genotype; "
    "(3) the covariate matrix stays with the genotype position, whereas the records "
    "null moves it with the phenotype. Under (2), in a gene with k fSNPs one donor's "
    "allelic evidence is dispersed over up to k independent recipients, so the event "
    "the records null exposes -- one dominant donor record landing whole on a "
    "heterozygote -- occurs intact only in single-fSNP genes, and in genes with no "
    "fSNP there is no allelic channel at all. The two nulls are therefore not the "
    "same experiment.")


def make_figure(X, arms, M, rq, tab):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.2),
                           gridspec_kw=dict(width_ratios=[1, 1.1, 1]))
    cols = {'a_base': '#2a78d6', 'a_cap': '#1baf7a', 'a_harm': '#8a5cd6',
            't_base': '#eb6834', 'mixQTL asc (permissive)': '#555555'}
    for name, c in cols.items():
        if name.startswith('mixQTL'):
            d = M['permissive']
            R, sdr = d['R_asc'], sdratio(d['b_asc'], d['se_asc'])
        else:
            R, sdr = arms[name]['R'], sdratio(arms[name]['b'], arms[name]['se'])
        ok = np.isfinite(R) & np.isfinite(sdr)
        ax[0].scatter(np.log(R[ok]), np.log(sdr[ok] ** 2), s=14, color=c, label=name,
                      alpha=0.8)
    lim = [-1.3, 1.3]
    ax[0].plot(lim, lim, color='#999999', lw=0.8, ls='--')
    ax[0].set_xlabel('log R_g (coupling ratio of the arm\'s own weights)')
    ax[0].set_ylabel('log sdratio^2 over 2,000 permutations')
    ax[0].set_title('R_g predicts per-gene sdratio, per weighting')
    ax[0].legend(fontsize=7, frameon=False)
    sel = tab[tab.config.isin(['combined, F(1, min dof)',
                               'permissive cutoffs, meta, normal ref (as published)',
                               'permissive cutoffs, meta, F/t ref',
                               'published cutoffs, meta, normal ref (as published)',
                               'published cutoffs, meta, F/t ref',
                               'converged rows, chi2(1)'])]
    y = np.arange(len(sel))
    for k, (_, r) in enumerate(sel.iterrows()):
        lo, hi = r['ci0.05']
        ax[1].errorbar(r['r0.05'], k, xerr=[[r['r0.05'] - lo], [hi - r['r0.05']]],
                       fmt='o', color='#2a78d6' if r.method == 'hapmixQTL' else
                       '#1baf7a' if r.method.startswith('mixQTL') else '#eb6834')
    ax[1].axvline(0.05, color='#999999', lw=0.8, ls='--')
    ax[1].set_yticks(y)
    ax[1].set_yticklabels([f"{r.method}: {r.config.replace(' (as published)', '')}"
                           f' [{r.draws_per_gene}]' for _, r in sel.iterrows()],
                          fontsize=7)
    ax[1].set_xlabel('rejection rate at nominal 0.05, gene-clustered 95% interval')
    ax[1].set_title('Three-way null calibration')
    conv = rq[rq.converged].groupby('gene').chi2.mean().reindex(X['genes'])
    ax[2].scatter(np.log(arms['a_base']['R']), conv.values, s=14, color='#eb6834')
    ax[2].set_xlabel('log R_g, hapmixQTL allelic (1/v weights)')
    ax[2].set_ylabel('RASQUAL mean chi2 over its converged null draws')
    ax[2].axhline(1.0, color='#999999', lw=0.8, ls='--')
    ax[2].set_title('RASQUAL -r null vs hapmixQTL R_g')
    fig.tight_layout()
    fig.savefig(OUT / 'comparator_null_2000.png', dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------

def main():
    OUT.mkdir(exist_ok=True)
    t0 = time.time()
    X = load()
    G = len(X['genes'])
    log(f'{G} genes, {len(X["donors"])} donors, {N_PERM} permutations')

    # ---- (b) first: cheap, and gate B protects everything downstream -------
    arms = run_arms(X)
    gateB = gate_arms(X, arms)

    # ---- (a) the mixQTL port ------------------------------------------------
    log('mixQTL port, 2 cutoff settings x 46 genes x 2000 permutations')
    M = run_mixqtl(X)
    gateA = gate_mixqtl(X, M)
    log(f'mixQTL done ({time.time() - t0:.0f}s)')

    res = dict(n_genes=G, n_perm=N_PERM, gates=dict(A=gateA, B=gateB,
               C='h1 = g + s/2, h2 = g - s/2 in {0,1}; h1-h2 == s; (h1+h2)/2 == g'))

    # ---- (a) summary -------------------------------------------------------
    A = {}
    long_rows = []
    for cut, d in M.items():
        methods = pd.Series([d['method'][i][0] for i in range(G)]).value_counts().to_dict()
        # method is permutation-invariant unless a channel returns NaN
        varying = int(sum(len(set(d['method'][i])) > 1 for i in range(G)))
        a_ = dict(method_counts=methods, genes_with_varying_method=varying,
                  median_n_asc=float(np.median(d['n_a'])),
                  median_n_trc=float(np.median(d['n_t'])),
                  genes_asc_ge15=int((d['n_a'] >= MX.META_N_CUTOFF).sum()),
                  median_n_cov_selected=float(np.median(d['n_sel'])))
        for ch in ('meta', 'trc', 'asc'):
            a_[f'{ch}_normal'] = rates3(d[f'p_{ch}'])
            a_[f'{ch}_F'] = rates3(d[f'pF_{ch}'])
            a_[f'{ch}_F_minus_normal'] = diffs3(d[f'pF_{ch}'], d[f'p_{ch}'])
            a_[f'{ch}_first30_normal'] = rates3(d[f'p_{ch}'][:, :30])
        a_['meta_F_charged'] = rates3(d['pF_meta_charged'])
        a_['trc_F_charged'] = rates3(d['pF_trc_charged'])
        ismeta = np.array([d['method'][i][0] == 'meta' for i in range(G)])
        a_['meta_normal_on_meta_genes'] = rates3(np.where(ismeta[:, None], d['p_meta'], np.nan))
        a_['meta_normal_on_fallback_genes'] = rates3(np.where(~ismeta[:, None], d['p_meta'], np.nan))
        # paired against hapmixQTL's combined, same (gene, perm)
        a_['meta_normal_minus_hapmixqtl_combined'] = diffs3(d['p_meta'], arms['b_base']['p'])
        a_['meta_F_minus_hapmixqtl_combined'] = diffs3(d['pF_meta'], arms['b_base']['p'])
        a_['asc_F_minus_hapmixqtl_allelic'] = diffs3(d['pF_asc'], arms['a_base']['p'])
        a_['trc_F_minus_hapmixqtl_total'] = diffs3(d['pF_trc'], arms['t_base']['p'])
        # R_g of mixQTL's own weights vs its allelic sdratio
        sdr = sdratio(d['b_asc'], d['se_asc'])
        okk = np.isfinite(sdr) & np.isfinite(d['R_asc'])
        a_['asc_R_vs_sdratio'] = dict(
            genes=int(okk.sum()),
            spearman=float(sps.spearmanr(d['R_asc'][okk], sdr[okk]).correlation)
            if okk.sum() > 4 else None,
            median_R=float(np.nanmedian(d['R_asc'])),
            sd_logR=float(np.nanstd(np.log(d['R_asc'][okk]), ddof=1)) if okk.sum() > 4 else None,
            median_sdratio=float(np.nanmedian(sdr)))
        A[cut] = a_
        for i, gname in enumerate(X['genes']):
            for p in range(N_PERM):
                long_rows.append((cut, gname, p, d['method'][i][p], d['n_a'][i], d['n_t'][i],
                                  d['n_sel'][i], d['b_trc'][i, p], d['se_trc'][i, p],
                                  d['p_trc'][i, p], d['pF_trc'][i, p], d['b_asc'][i, p],
                                  d['se_asc'][i, p], d['p_asc'][i, p], d['pF_asc'][i, p],
                                  d['b_meta'][i, p], d['se_meta'][i, p], d['p_meta'][i, p],
                                  d['pF_meta'][i, p]))
    pd.DataFrame(long_rows, columns=[
        'cutoffs', 'gene', 'perm', 'method', 'n_asc', 'n_trc', 'n_cov_selected',
        'b_trc', 'se_trc', 'p_trc_normal', 'p_trc_F', 'b_asc', 'se_asc',
        'p_asc_normal', 'p_asc_F', 'b_meta', 'se_meta', 'p_meta_normal', 'p_meta_F'
    ]).to_csv(OUT / 'mixqtl_null_long.tsv.gz', sep='\t', index=False)
    res['a_mixqtl'] = A

    # ---- (b) summary -------------------------------------------------------
    B = {}
    per_rows = []
    for name, d in arms.items():
        sdr = sdratio(d['b'], d['se'])
        mc = sdratio_mc_sd(d['b'], d['se'])
        rr = rates3(d['p'])
        ok = np.isfinite(sdr) & np.isfinite(d['R'])
        entry = dict(channel=d['chan'], rates=rr,
                     median_sdratio=float(np.nanmedian(sdr)),
                     sd_sdratio_across_genes=float(np.nanstd(sdr, ddof=1)),
                     median_mc_sd_sdratio=float(np.nanmedian(mc)))
        if ok.sum() > 4 and d['chan'] != 'b':
            lr, ls = np.log(d['R'][ok]), np.log(sdr[ok] ** 2)
            entry.update(sd_log_sdratio2=float(np.std(ls, ddof=1)),
                         mc_sd_log_sdratio2=float(2 * np.nanmedian(mc[ok] / sdr[ok])))
        if ok.sum() > 4 and d['chan'] != 'b' and np.std(lr) > 1e-9:
            # equal weights give R == 1 exactly: nothing to regress on
            fit = sps.linregress(lr, ls)
            entry.update(R_genes=int(ok.sum()), median_R=float(np.median(d['R'][ok])),
                         sd_logR=float(np.std(lr, ddof=1)),
                         spearman_R_sdratio=float(sps.spearmanr(d['R'][ok], sdr[ok]).correlation),
                         slope_logsdr2_on_logR=float(fit.slope),
                         intercept_logsdr2_on_logR=float(fit.intercept),
                         r2_logsdr2_on_logR=float(fit.rvalue ** 2),
                         spearman_R_rej01=float(sps.spearmanr(
                             d['R'][ok], (d['p'][ok] < 0.01).mean(1)).correlation))
        B[name] = entry
        for i, gname in enumerate(X['genes']):
            ok_i = np.isfinite(d['p'][i])
            per_rows.append(dict(arm=name, gene=gname, stratum=X['strata'][i],
                                 n_records=int(d['n'][i]), R=d['R'][i],
                                 sdratio=sdr[i], sdratio_mc_sd=mc[i],
                                 **{f'rej{a_}': float((d['p'][i][ok_i] < a_).mean())
                                    if ok_i.any() else np.nan for a_ in ALPHAS}))
    pd.DataFrame(per_rows).to_csv(OUT / 'arms_per_gene.tsv', sep='\t', index=False)

    pairs = {
        # allelic, one component at a time
        'a_cap - a_base': ('a_cap', 'a_base'),
        'a_ref_normal - a_base': ('a_ref_normal', 'a_base'),
        'a_harm - a_base_pos': ('a_harm', 'a_base_pos'),
        'a_response - a_base_pos': ('a_response', 'a_base_pos'),
        'a_harm_cap - a_harm': ('a_harm_cap', 'a_harm'),
        'a_band_published - a_base (published-band genes)': ('a_band_published', 'a_base_on_published_genes'),
        'a_band_permissive - a_base (permissive-band genes)': ('a_band_permissive', 'a_base_on_permissive_genes'),
        'a_ols - a_base': ('a_ols', 'a_base'),
        'a_cap - a_ols': ('a_cap', 'a_ols'),
        'a_mixall_noband - a_base_pos': ('a_mixall_noband', 'a_base_pos'),
        # total
        't_ols - t_base': ('t_ols', 't_base'),
        't_ref_normal - t_base': ('t_ref_normal', 't_base'),
        't_covsel - t_base (covariates AND 1/vt weights)': ('t_covsel', 't_base'),
        't_covsel_ols - t_ols (covariate treatment only)': ('t_covsel_ols', 't_ols'),
        't_band_published - t_base': ('t_band_published', 't_base'),
        # combined
        'b_cap - b_base': ('b_cap', 'b_base'),
        'b_ref_normal - b_base': ('b_ref_normal', 'b_base'),
        'b_weightfamily - b_base_pos': ('b_weightfamily', 'b_base_pos'),
        'b_band_published - b_base (published-band genes)': ('b_band_published', 'b_base_on_published_genes'),
        'b_ols_both - b_base': ('b_ols_both', 'b_base'),
        'b_cap_totols - b_base': ('b_cap_totols', 'b_base'),
    }
    B['_paired_differences'] = {k: diffs3(arms[a1]['p'], arms[a0]['p'])
                                for k, (a1, a0) in pairs.items()}
    # the gene subsets of the matched arms
    B['_matched_subsets'] = {nm: [X['genes'][i] for i in range(G)
                                  if np.isfinite(arms[nm]['p'][i]).any()]
                             for nm in ('a_band_published', 'a_band_permissive',
                                        'b_band_published')}
    # donors lost to the harmonic/response donor set
    B['_donors_admitted_vs_positive_counts'] = dict(
        admitted=int(arms['a_base']['n'].sum()), positive=int(arms['a_base_pos']['n'].sum()))
    res['b_decomposition'] = B

    # ---- (c) RASQUAL -------------------------------------------------------
    rq, rq_obs = read_rasqual(X)
    rq.to_csv(OUT / 'rasqual_null_at_lead.tsv', sep='\t', index=False)
    Rg = arms['a_base']['R']
    Pall = to_matrix(rq, X['genes'], 'p')
    Pconv = to_matrix(rq, X['genes'], 'p', rq.converged)
    Pnon = to_matrix(rq, X['genes'], 'p', ~rq.converged)
    nf = rq.groupby('gene').n_fsnp.median().reindex(X['genes'])
    zero_f = [g_ for g_ in X['genes'] if nf.get(g_, np.nan) == 0]
    Pconv_f = np.where(np.array([nf.get(g_, np.nan) > 0 for g_ in X['genes']])[:, None],
                       Pconv, np.nan)
    C_ = dict(null_definition=RASQUAL_NULL_TEXT,
              n_rows=int(len(rq)), n_converged=int(rq.converged.sum()),
              nonconverged_codes=rq.loc[~rq.converged, 'convergence'].value_counts().to_dict(),
              genes_with_rows=int(rq.gene.nunique()),
              all=rates3(Pall), converged=rates3(Pconv), nonconverged=rates3(Pnon),
              converged_genes_with_fsnps=rates3(Pconv_f),
              genes_zero_fsnps=zero_f,
              n_fsnp_median=float(nf.median()),
              n_fsnp_quartiles=[float(nf.quantile(.25)), float(nf.quantile(.75))],
              n_fsnp_obs_equals_null=bool(np.allclose(
                  rq_obs.set_index('gene').n_fsnp_obs.reindex(X['genes']).values,
                  nf.values, equal_nan=True)))
    # per-gene: RASQUAL converged rejection and mean chi2 vs hapmixQTL's R_g
    conv = rq[rq.converged]
    pg = conv.groupby('gene').agg(n=('p', 'size'), rej05=('p', lambda p: (p < .05).mean()),
                                  mean_chi2=('chi2', 'mean')).reindex(X['genes'])
    okg = pg.n.fillna(0).values >= 10
    C_['per_gene_vs_hapmixqtl_R'] = dict(
        genes=int(okg.sum()),
        spearman_rej05=float(sps.spearmanr(Rg[okg], pg.rej05.values[okg]).correlation),
        spearman_rej05_p=float(sps.spearmanr(Rg[okg], pg.rej05.values[okg]).pvalue),
        spearman_meanchi2=float(sps.spearmanr(Rg[okg], pg.mean_chi2.values[okg]).correlation),
        spearman_meanchi2_p=float(sps.spearmanr(Rg[okg], pg.mean_chi2.values[okg]).pvalue))
    # is RASQUAL's gene-to-gene spread of its null statistic beyond chi2(1)
    # Monte Carlo? Simulate each gene's n_g converged draws as iid chi2(1).
    mchi = pg.mean_chi2.values[okg]
    ng_ok = pg.n.values[okg].astype(int)
    rng_d = np.random.default_rng(SS[3])
    sim_sd = np.array([np.std([rng_d.chisquare(1, n).mean() for n in ng_ok], ddof=1)
                       for _ in range(2000)])
    obs_sd = float(np.std(mchi, ddof=1))
    C_['per_gene_mean_chi2_dispersion'] = dict(
        observed_sd=obs_sd, iid_chi2_sd_median=float(np.median(sim_sd)),
        iid_chi2_sd_q975=float(np.quantile(sim_sd, 0.975)),
        p_upper=float((sim_sd >= obs_sd).mean()),
        min_mean_chi2=float(mchi.min()), max_mean_chi2=float(mchi.max()),
        # the same between-gene spread for the two records-null arms, whose
        # 2,000 draws per gene leave almost no Monte Carlo in it
        hapmixqtl_combined_sd_mean_t2=float(np.std(arms['b_base']['t2'].mean(1), ddof=1)),
        hapmixqtl_allelic_sd_mean_t2=float(np.std(arms['a_base']['t2'].mean(1), ddof=1)),
        mixqtl_permissive_meta_sd_mean_z2=float(np.std(np.nanmean(
            (M['permissive']['b_meta'] / M['permissive']['se_meta']) ** 2, 1), ddof=1)),
        rasqual_excess_sd=float(np.sqrt(max(obs_sd ** 2 - np.median(sim_sd) ** 2, 0.0))))
    pw, n_g = rasqual_power(X, arms, rq, Rg)
    C_['power_of_per_gene_comparison'] = pw
    C_['median_converged_rows_per_gene'] = float(np.median(n_g))
    # RASQUAL vs hapmixQTL pooled, gene-clustered (unpaired draws, paired genes)
    idx = RNG_BOOT.integers(0, G, size=(N_BOOT, G))
    for a_ in ALPHAS:
        kr = ((Pconv < a_) & np.isfinite(Pconv)).sum(1); nr = np.isfinite(Pconv).sum(1)
        kh = (arms['b_base']['p'] < a_).sum(1); nh = np.isfinite(arms['b_base']['p']).sum(1)
        est = kh.sum() / nh.sum() - kr.sum() / nr.sum()
        bd = kh[idx].sum(1) / nh[idx].sum(1) - kr[idx].sum(1) / nr[idx].sum(1)
        C_[f'hapmixqtl_minus_rasqual_{a_}'] = dict(
            diff=float(est), lo=float(np.quantile(bd, .025)), hi=float(np.quantile(bd, .975)))
    res['c_rasqual'] = C_

    # ---- (d) corrected three-way table -------------------------------------
    rec_h = json.load(open(D / 'channel_split_20260924' / 'summary.json'))
    T = []

    def row(method, config, null, draws, R3, recorded=None, note=''):
        T.append(dict(method=method, config=config, null=null, draws_per_gene=draws,
                      **{f'r{a_}': R3[str(a_)]['rate'] for a_ in ALPHAS},
                      **{f'ci{a_}': [R3[str(a_)]['lo'], R3[str(a_)]['hi']] for a_ in ALPHAS},
                      n=R3['0.05']['n'], genes=R3['0.05']['genes'],
                      recorded_2026_09_24=recorded, note=note))
    row('hapmixQTL', 'combined, F(1, min dof)', 'records', N_PERM, rates3(arms['b_base']['p']),
        recorded=rec_h['both']['frac_below_05'],
        note='supersedes CLAUDE.md 0.082 (30 draws, map_cis with tau_refit=True)')
    row('hapmixQTL', 'allelic channel', 'records', N_PERM, rates3(arms['a_base']['p']),
        recorded=rec_h['allelic_only']['frac_below_05'])
    row('hapmixQTL', 'total channel', 'records', N_PERM, rates3(arms['t_base']['p']),
        recorded=rec_h['total_only']['frac_below_05'])
    for cut in CUTS:
        d = M[cut]
        row('mixQTL port', f'{cut} cutoffs, meta, normal ref (as published)', 'records', N_PERM,
            rates3(d['p_meta']), recorded=gateA['first30_rate'] if cut == 'permissive' else None,
            note='supersedes CLAUDE.md 0.067 (30 draws)' if cut == 'permissive' else
            'no earlier figure')
        row('mixQTL port', f'{cut} cutoffs, meta, F/t ref', 'records', N_PERM, rates3(d['pF_meta']))
        row('mixQTL port', f'{cut} cutoffs, trc, normal ref', 'records', N_PERM, rates3(d['p_trc']))
        row('mixQTL port', f'{cut} cutoffs, asc, normal ref', 'records', N_PERM, rates3(d['p_asc']))
        row('mixQTL port', f'{cut} cutoffs, asc, F/t ref', 'records', N_PERM, rates3(d['pF_asc']))
    row('RASQUAL', 'converged rows, chi2(1)', 'rasqual -r (per-fSNP + separate total order)',
        N_RASQUAL_ROUNDS, C_['converged'], recorded=0.0436,
        note='value stands; relabel: 30 draws, converged-only, a different null')
    row('RASQUAL', 'all rows incl. non-converged', 'rasqual -r', N_RASQUAL_ROUNDS, C_['all'])
    row('RASQUAL', 'non-converged rows only', 'rasqual -r', N_RASQUAL_ROUNDS, C_['nonconverged'])
    tab = pd.DataFrame(T)
    tab.to_csv(OUT / 'three_way_table.tsv', sep='\t', index=False)
    res['d_three_way_table'] = T
    make_figure(X, arms, M, rq, tab)
    res['runtime_s'] = time.time() - t0
    (OUT / 'summary.json').write_text(json.dumps(res, indent=1, default=lambda o: (
        o.item() if hasattr(o, 'item') else str(o))))
    with pd.option_context('display.width', 250, 'display.max_columns', 30):
        print(tab[['method', 'config', 'draws_per_gene', 'r0.05', 'r0.01', 'r0.001', 'n',
                   'recorded_2026_09_24']].to_string(index=False))
    log(f'wrote {OUT} in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()

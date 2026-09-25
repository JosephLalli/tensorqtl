"""Where does the SHIPPED combined statistic's nominal-p excess come from?

QUESTION. hapmixQTL's shipped nominal p is the combined statistic: an
inverse-variance meta-analysis of the allelic and the total slope (each channel's
slope weighted by the reciprocal of its squared standard error), t^2 referred to
F(1, min(dof_a, dof_t)). On the shared records-permutation null
(scripts/null_permutation_instrument.py; 46 genes x 92 donors at RASQUAL's
observed lead; 2,000 permutations from rng = RandomState(42)) it rejects at
0.0680 / 0.0175 / 0.0028 at nominal 0.05 / 0.01 / 0.001. The first round
decomposed each CHANNEL alone. This script decomposes the COMBINED statistic's
excess over the model-world rate into allelic weight-residual coupling, total
weight-residual coupling (a per-gene scale inflation), heavy-tailed marginals,
the reference/combination under the model, cross-channel dependence and
interaction, all on the identical permutation stream.

THE COMBINED STATISTIC, per (gene, permutation). With ba, sea2 the allelic
slope and squared standard error and bt, set2 the total ones,
    prec = 1/sea2 + 1/set2,   b = (ba/sea2 + bt/set2)/prec,   t2_b = b^2 prec,
    dof  = min(n_a - 1, n_t - 19),
exactly fit_channels. Writing ta = ba/sqrt(sea2), tt = bt/sqrt(set2) and
pi_a = (1/sea2)/prec (the allelic share of the combined precision),
t2_b = (sqrt(pi_a) ta + sqrt(1 - pi_a) tt)^2, so pi_a is the weight the allelic
channel carries inside the shipped statistic.

DESIGN. Held fixed in every arm: the 46 genes, each gene's fixed variant, the
92 donors, the 17 covariates, the admission rule (allelic: va > 1e-12,
INCLUDING s = 0 donors, dof n_a - 1; total: all 92, dof 73), the estimator
algebra and the 2,000 permutations. What varies is the RECORD SET each channel
is permuted over, one channel state at a time. Channel states:

  allelic R   the real records: whitened residual z_a = sqrt(w_a) a, w_a = 1/va
          M   z_a iid N(0, sigma_a^2) at the real w_a, sigma_a^2 = mean real z_a^2
              over admitted records (scale-matched, see below)
          D   the gene's own real z_a shuffled among its admitted records
              against w_a (both marginals kept, the pairing destroyed); exactly
              the first round's allelic DECOUPLE construction
          Mu  M at unit scale (sigma_a = 1), the literal "z iid N(0,1)"
          CAP real records, weights mixQTL-capped (below)
  total   R   the real records
          M   t = sigma_t sqrt(vt) N(0,1), sigma_t^2 = sum r^2 / 73 of the real
              fit (scale-matched)
          D   t = sqrt(vt) z_shuf, where z = r/sqrt(1 - h) is the REAL record's
              leverage-corrected whitened covariate residual (r = (I - P)
              sqrt(w) t, P the projection onto sqrt(w)[1, C], h = diag P) and
              z_shuf is z shuffled across the 92 records; the FULL fit is then
              rerun, so the covariates are re-partialled and the refit residual
              is (I - P) z_shuf, not z_shuf. The covariate-span part of the real
              t is dropped, which is exact: the statistic is invariant to
              adding anything in span(sqrt(w)[1, C]). This is exactly the first
              round's total DECOUPLE construction.
          Mu  M at unit scale
          OLS real records, unit weights (dof unchanged)

WHY SCALE-MATCHED MODEL. Each channel's t statistic is invariant to the scale
of its own residuals, so the first round could use unit-variance z. The
COMBINED statistic is not: the meta weights the channels by 1/se^2, and the
fitted se carries each channel's residual scale. A unit-scale MODEL would hand
the channels a different precision share than the real data has and so
misattribute excess between them. M therefore matches each channel's expected
residual sum of squares to the real one; Mu is kept as a sensitivity arm to
show how much the share matters.

ARMS (allelic state, total state), each a full combined fit on the same
permutations, K = 200 record sets per simulated state, set k of every arm built
from the same draws (common random numbers), so arm differences are paired:
  REAL (R,R)  MODEL (M,M)  DECOUPLE_A (D,R)  DECOUPLE_T (R,D)  DECOUPLE_BOTH (D,D)
  ALLELIC_ONLY_REAL (R,M)  TOTAL_ONLY_REAL (M,R)  DEC_A_MODEL_T (D,M)
  MODEL_A_DEC_T (M,D)      [the last two complete the 3 x 3 factorial]
  MODEL_UNIT (Mu,Mu)  ALLELIC_ONLY_REAL_UNIT (R,Mu)  TOTAL_ONLY_REAL_UNIT (Mu,R)
  CAP (CAP,R)  TOTAL_UNWEIGHTED (R,OLS)  CAP_PLUS_TOTAL_UNWEIGHTED (CAP,OLS)
  CHANNEL_INDEPENDENT: real allelic at permutation p combined with real total at
      permutation sigma_k(p), sigma_k a random reordering of the 2,000
      instrument permutations (K = 200). Each channel's marginal null is then
      EXACTLY the instrument's, and only the within-donor link between the two
      channels (the same donor's allelic and total records moving together) is
      broken. REAL minus this arm is the cross-channel-dependence share.

CAP RULE, exactly as scripts/comparator_null_2000.py's a_cap (mixQTL's
rlib_matrix_ls.R:88-90 via tensorqtl.mixqtl_replication.apply_weight_cap):
cap = min(10, floor(n_a/10)); cutoff = cap x min(w_a over admitted records);
w_a > cutoff set to cutoff. With n_a = 46..90 here the realised fold limit is
4 to 9, not a uniform 9.

BUDGET, per alpha, E = REAL - MODEL. Three views:
  insertion (main effects from MODEL):
      allelic marginal  (D,M) - (M,M)      allelic coupling  (R,M) - (D,M)
      total marginal    (M,D) - (M,M)      total coupling    (M,R) - (M,D)
      interaction       REAL - (R,M) - (M,R) + (M,M), split into
          cross-channel dependence  REAL - CHANNEL_INDEPENDENT and
          non-linear remainder      the rest
  removal (from REAL):
      REAL - (D,R), REAL - (R,D), (D,D) - (M,M), remainder
  Shapley allocation (the interaction split evenly between the two orders in
  which a channel can be switched): e.g. allelic coupling =
      1/2[(R,M) - (D,M)] + 1/2[REAL - (D,R)]; the four components sum to E.
  Additivity checks: allelic coupling at total=M vs total=R; total coupling at
  allelic=M vs allelic=R; the coupling x coupling interaction
  REAL - (D,R) - (R,D) + (D,D).
  Also reported: MODEL - nominal (what the reference and the estimated-precision
  combination do under the model itself).

INTERVALS AND FLOORS. Pooled rates are per-gene means (every gene has 2,000
permutations). Every rate and every budget term gets a PAIRED gene-clustered
percentile bootstrap (2,000 resamples of the 46 genes, the same resampled genes
for every arm). Every simulated arm also gets its Monte Carlo sd across the K
record sets (the spread one real record set would show if that arm's mechanism
were the truth) and the se of its mean (sd / sqrt K). A term CLEARS its floor
when its bootstrap interval excludes 0 AND |term| exceeds 2 x its floor, where
the floor is the per-set sd of the term if the term contains a real-data
channel (a single realization), and the se of the mean if it is a difference of
purely simulated arms. Both flags (against the per-set sd and against the se of
the mean) are also written for every term: the per-set sd is the right floor
for "is the real pairing unlike a random one" (coupling, cross-channel), the se
of the mean for "is this expectation non-zero" (interaction, reference).

PRECISION SHARE. Per gene: median over permutations of pi_a in REAL, pi_a on
the observed data, and pi_a's median in MODEL and MODEL_UNIT (to show the
scale-matched MODEL reproduces the real share and the unit one does not). Then
whether genes where the allelic channel dominates carry the combined excess:
Spearman of per-gene excess against pi_a (with a gene-label permutation p),
excess by pi_a tercile with gene-bootstrap intervals, the fraction of pooled E
carried by genes with pi_a > 0.5, and the same by coverage stratum (pi_a tracks
n_a and coverage).

GATES (the script aborts on failure):
  G1  REAL t2_a, t2_t and t2_b equal null_long.tsv.gz per (gene, perm) to
      relative 1e-9 where t2 >= 1e-3 and absolute 1e-12 below, and the pooled
      p_a, p_t, p_b counts at 0.05 / 0.01 / 0.001 are identical.
  G2  CAP, CAP_PLUS_TOTAL_UNWEIGHTED, the CAP allelic channel and the unit-weight
      total channel reproduce comparator_null_2000's b_cap 5268/1120/114,
      b_cap_totols 4709/944/95, a_cap 4413/967/128 and t_ols 4560/910/92
      exactly.
  G3  the channel constructions, re-run with the first round's own child
      streams and set counts, reproduce the first round's pooled channel-alone
      rates exactly: allelic MODEL and DECOUPLE (weight_residual_coupling,
      SeedSequence(42).spawn(8)[2] and [3], 40 sets) and total MODEL and
      DECOUPLE (total_channel_null_calibration, spawn(10)[0] and [1], 200 sets;
      DECOUPLE total 0.050009 at 0.05). This proves the budget's states are the
      first round's constructions.
  G4  CHANNEL_INDEPENDENT's per-channel rejection counts equal REAL's exactly
      (checked on the re-indexed arrays).

SEEDS. Master seed 42. Permutations: the instrument's RandomState(42) stream.
Budget streams: SeedSequence(42).spawn(24)[16..22] (indices 0..9 were used by
the first-round scripts; child i of SeedSequence(42) is the same whatever the
spawn count, so reusing an index would reuse draws), each spawned once more per
gene so results do not depend on worker scheduling.

Natural-log units. Outputs: brainvar_hapmix_deploy/combined_statistic_budget_20260925/
  summary.json, budget.tsv, arm_rates.tsv, per_gene.tsv, fig_budget.png,
  fig_precision_share.png.
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import json                                  # noqa: E402
import sys                                   # noqa: E402
import time                                  # noqa: E402
from multiprocessing import Pool             # noqa: E402
from pathlib import Path                     # noqa: E402

import numpy as np                           # noqa: E402
import pandas as pd                          # noqa: E402
from scipy import stats as sps               # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from total_channel_null_calibration import setup, t2_stats   # noqa: E402
from tensorqtl.mixqtl_replication import (apply_weight_cap,   # noqa: E402
                                          PUBLISHED_CUTOFFS)

D = Path('/mnt/ssd/lalli/brainvar_hapmix_deploy')
INST = D / 'nominal_p_null_instrument_20260925'
OUT = D / 'combined_statistic_budget_20260925'
SEED, EPS = 42, 1e-12
N_PERM, K, KX, N_BOOT = 2000, 200, 200, 2000
ALPHAS = (0.05, 0.01, 0.001)
N_WORKERS = 23

GRID = {  # arm -> (allelic state, total state)
    'REAL': ('R', 'R'), 'MODEL': ('M', 'M'),
    'DECOUPLE_A': ('D', 'R'), 'DECOUPLE_T': ('R', 'D'), 'DECOUPLE_BOTH': ('D', 'D'),
    'ALLELIC_ONLY_REAL': ('R', 'M'), 'TOTAL_ONLY_REAL': ('M', 'R'),
    'DEC_A_MODEL_T': ('D', 'M'), 'MODEL_A_DEC_T': ('M', 'D'),
    'MODEL_UNIT': ('Mu', 'Mu'), 'ALLELIC_ONLY_REAL_UNIT': ('R', 'Mu'),
    'TOTAL_ONLY_REAL_UNIT': ('Mu', 'R'),
    'CAP': ('CAP', 'R'), 'TOTAL_UNWEIGHTED': ('R', 'OLS'),
    'CAP_PLUS_TOTAL_UNWEIGHTED': ('CAP', 'OLS'),
}
ARMS = list(GRID) + ['CHANNEL_INDEPENDENT']
# arms containing at least one real-data channel (a single realization)
HAS_REAL = {a_ for a_, (sa, st_) in GRID.items()
            if sa in ('R', 'CAP') or st_ in ('R', 'OLS')} | {'CHANNEL_INDEPENDENT'}

G_ = {}   # read-only globals shared with forked workers


# --------------------------------------------------------------- algebra
def allelic_bse(Za, w, Sg, n_a):
    """Allelic slope and squared se for record sets Za (K, N) whitened with w.
    Sg (P, N): s at the position each record lands in. Returns (K, P) arrays."""
    num = (Za * np.sqrt(w)[None, :]) @ Sg.T
    den = (Sg ** 2) @ w
    S = (Za ** 2).sum(1)
    b = num / den[None, :]
    se2 = (S[:, None] - num ** 2 / den[None, :]) / (n_a - 1) / den[None, :]
    return b, se2


def total_bse(Y, st, U, dof):
    """Total slope and squared se for record sets Y (N, K). Returns (K, P)."""
    _, r, num = t2_stats(Y, st, U, dof)
    S = (r ** 2).sum(0)
    den = st['den']
    b = num / den[:, None]
    se2 = (S[None, :] - num ** 2 / den[:, None]) / dof / den[:, None]
    return b.T, se2.T


def combine(ba, sea2, bt, set2):
    prec = 1.0 / sea2 + 1.0 / set2
    b = (ba / sea2 + bt / set2) / prec
    return b * b * prec, (1.0 / sea2) / prec


def counts(p):
    """p (K, P) -> rejection counts (K, 3)."""
    return np.stack([(p < al).sum(-1) for al in ALPHAS], -1)


# --------------------------------------------------------------- per gene
def gene_work(k):
    X = G_
    N = X['N']
    a, va, s = X['a'][k], X['va'][k], X['s'][k]
    t, vt, g = X['t'][k], X['vt'][k], X['g'][k]
    adm = np.isfinite(a) & np.isfinite(va) & (va > EPS)
    n_a = int(adm.sum())
    w = np.where(adm, 1.0 / np.where(adm, va, 1.0), 0.0)
    za = np.where(adm, np.sqrt(w) * np.where(adm, a, 0.0), 0.0)
    Sg = s[X['inv']]                                        # (P, N)
    dofa, doft = n_a - 1, X['doft']
    dof = min(dofa, doft)
    rng = {nm: np.random.default_rng(X['gene_ss'][nm][k])
           for nm in ('am', 'ad', 'tm', 'td')}

    # ---- allelic states ------------------------------------------------
    A = {}
    A['R'] = allelic_bse(za[None, :], w, Sg, n_a)
    sig_a = float(np.sqrt((za[adm] ** 2).mean()))
    Zm = np.zeros((K, N)); Zm[:, adm] = rng['am'].standard_normal((K, n_a))
    bu, su = allelic_bse(Zm, w, Sg, n_a)
    A['Mu'] = (bu, su)
    A['M'] = (bu * sig_a, su * sig_a ** 2)
    zadm = za[adm]
    Zd = np.zeros((K, N)); Zd[:, adm] = np.stack([rng['ad'].permutation(zadm)
                                                   for _ in range(K)])
    A['D'] = allelic_bse(Zd, w, Sg, n_a)
    wc, _, _ = apply_weight_cap(w[adm], n_a, PUBLISHED_CUTOFFS['weight_cap'])
    wcap = np.zeros(N); wcap[adm] = wc
    zc = np.where(adm, np.sqrt(wcap) * np.where(adm, a, 0.0), 0.0)
    A['CAP'] = allelic_bse(zc[None, :], wcap, Sg, n_a)
    cap_fold = float(min(PUBLISHED_CUTOFFS['weight_cap'], np.floor(n_a / 10)))

    # ---- total states --------------------------------------------------
    U = np.zeros((N_PERM, N)); U[np.arange(N_PERM)[:, None], X['perms']] = g[None, :]
    wt = 1.0 / vt
    st = setup(wt, X['Zd'], U)
    T = {}
    T['R'] = total_bse(t[:, None], st, U, doft)
    _, r, _ = t2_stats(t[:, None], st, U, doft)
    r = r[:, 0]
    z = r / np.sqrt(1 - st['h'])
    sig_t = float(np.sqrt((r @ r) / doft))
    sq = np.sqrt(vt)[:, None]
    bu, su = total_bse(sq * rng['tm'].standard_normal((N, K)), st, U, doft)
    T['Mu'] = (bu, su)
    T['M'] = (bu * sig_t, su * sig_t ** 2)
    T['D'] = total_bse(sq * np.stack([z[rng['td'].permutation(N)] for _ in range(K)], 1),
                       st, U, doft)
    st1 = setup(np.ones(N), X['Zd'], U)
    T['OLS'] = total_bse(t[:, None], st1, U, doft)

    out = dict(k=k, n_a=n_a, dofa=dofa, dof=dof, cap_fold=cap_fold,
               sig_a2=sig_a ** 2, sig_t2=sig_t ** 2,
               w_fold_a=float(w[adm].max() / w[adm].min()))
    ch = {}
    for nm, (b_, s2) in A.items():
        ch['a_' + nm] = counts(sps.f.sf(b_ ** 2 / s2, 1, dofa))
    for nm, (b_, s2) in T.items():
        ch['t_' + nm] = counts(sps.f.sf(b_ ** 2 / s2, 1, doft))
    out['chan'] = ch

    rej, share = {}, {}
    for arm, (sa, stt) in GRID.items():
        ba, sea2 = A[sa]
        bt, set2 = T[stt]
        t2b, pa = combine(ba, sea2, bt, set2)
        rej[arm] = counts(sps.f.sf(t2b, 1, dof))
        share[arm] = float(np.median(pa))
        if arm == 'REAL':
            out['real_t2'] = dict(a=(ba ** 2 / sea2)[0], t=(bt ** 2 / set2)[0], b=t2b[0])
            out['real_pa'] = pa[0]
            ta = ba[0] / np.sqrt(sea2[0]); tt = bt[0] / np.sqrt(set2[0])
            out['corr_ta_tt_real'] = float(np.corrcoef(ta, tt)[0, 1])
        if arm == 'MODEL':
            ta = ba / np.sqrt(sea2); tt = bt / np.sqrt(set2)
            cc = [np.corrcoef(ta[j], tt[j])[0, 1] for j in range(K)]
            out['corr_ta_tt_model'] = (float(np.mean(cc)), float(np.std(cc, ddof=1)))
    # channel-independent: real allelic at p with real total at sigma_k(p)
    ba, sea2 = A['R']; bt, set2 = T['R']
    ta0 = ba[0] / np.sqrt(sea2[0]); tt0 = bt[0] / np.sqrt(set2[0])
    ci = np.zeros((KX, 3), int); cc = []
    g4 = 0
    pa_real = sps.f.sf(ba[0] ** 2 / sea2[0], 1, dofa)
    pt_real = sps.f.sf(bt[0] ** 2 / set2[0], 1, doft)
    for j in range(KX):
        sg = X['sigma'][j]
        t2b, _ = combine(ba[0], sea2[0], bt[0][sg], set2[0][sg])
        ci[j] = counts(sps.f.sf(t2b, 1, dof)[None, :])[0]
        cc.append(np.corrcoef(ta0, tt0[sg])[0, 1])
        # G4: each channel's count unchanged by the re-pairing
        if not (np.array_equal(counts(pt_real[sg][None, :]), counts(pt_real[None, :]))):
            g4 += 1
    rej['CHANNEL_INDEPENDENT'] = ci
    out['g4_failures'] = g4
    out['corr_ta_tt_indep'] = (float(np.mean(cc)), float(np.std(cc, ddof=1)))
    out['rej'] = rej
    out['share'] = share
    del pa_real
    return out


# --------------------------------------------------------------- gate 3
def first_round_channel_rates(d):
    """Re-run the first round's four channel-alone simulated arms with their own
    child streams, set counts and rejection conventions."""
    a, va, s = d['a'], d['va'], d['s']
    t, vt, g, C = d['t'], d['vt'], d['g'], d['C']
    G, N = a.shape
    rng = np.random.RandomState(SEED)
    perms = np.array([rng.permutation(N) for _ in range(N_PERM)])
    inv = np.argsort(perms, axis=1)
    ss8 = np.random.SeedSequence(SEED).spawn(8)
    r_am, r_ad = np.random.default_rng(ss8[2]), np.random.default_rng(ss8[3])
    ss10 = np.random.SeedSequence(SEED).spawn(10)
    r_tm, r_td = np.random.default_rng(ss10[0]), np.random.default_rng(ss10[1])
    Zd = np.column_stack([np.ones(N), C])
    doft = N - 1 - C.shape[1] - 1
    crit = {al: sps.f.isf(al, 1, doft) for al in ALPHAS}
    cnt = {nm: np.zeros(3) for nm in ('a_MODEL', 'a_DECOUPLE', 't_MODEL', 't_DECOUPLE')}
    for k in range(G):
        adm = np.isfinite(a[k]) & np.isfinite(va[k]) & (va[k] > EPS)
        n_a = int(adm.sum())
        w = np.where(adm, 1.0 / np.where(adm, va[k], 1.0), 0.0)
        za = np.where(adm, np.sqrt(w) * np.where(adm, a[k], 0.0), 0.0)
        Sg = s[k][inv]
        for nm, sets in (('a_MODEL', r_am.standard_normal((40, n_a))),
                         ('a_DECOUPLE', np.stack([r_ad.permutation(za[adm])
                                                  for _ in range(40)]))):
            Zf = np.zeros((40, N)); Zf[:, adm] = sets
            b_, s2 = allelic_bse(Zf, w, Sg, n_a)
            p = sps.f.sf(b_ ** 2 / s2, 1, n_a - 1)
            cnt[nm] += [(p < al).sum() for al in ALPHAS]
        U = np.zeros((N_PERM, N)); U[np.arange(N_PERM)[:, None], perms] = g[k][None, :]
        st = setup(1.0 / vt[k], Zd, U)
        _, r, _ = t2_stats(t[k][:, None], st, U, doft)
        z = r[:, 0] / np.sqrt(1 - st['h'])
        sq = np.sqrt(vt[k])[:, None]
        for nm, Y in (('t_MODEL', sq * r_tm.standard_normal((N, 200))),
                      ('t_DECOUPLE', sq * np.stack([z[r_td.permutation(N)]
                                                    for _ in range(200)], 1))):
            t2s, _, _ = t2_stats(Y, st, U, doft)
            cnt[nm] += [(t2s > crit[al]).sum() for al in ALPHAS]
    return {nm: [float(c / (G * (40 if nm.startswith('a_') else 200) * N_PERM))
                 for c in v] for nm, v in cnt.items()}


# --------------------------------------------------------------- summaries
def boot_stats(x, idx):
    """x: per-gene values (G,). Pooled mean and bootstrap interval."""
    b = x[idx].mean(1)
    return float(x.mean()), float(np.quantile(b, .025)), float(np.quantile(b, .975)), b


def main():
    t0 = time.time()
    OUT.mkdir(exist_ok=True)
    (OUT / 'scratch').mkdir(exist_ok=True)
    d = np.load(INST / 'inputs_at_lead.npz', allow_pickle=True)
    genes = [str(x) for x in d['genes']]
    strata = [str(x) for x in d['strata']]
    G, N = d['a'].shape
    C = d['C']
    assert np.all(d['vt'] > EPS) and np.all(np.isfinite(d['t']))
    doft = N - 1 - (1 + C.shape[1])
    summary = dict(investigation='combined-budget', n_genes=G, n_perm=N_PERM,
                   K_sets=K, K_channel_independent=KX, dof_total=doft,
                   missing_inputs_noted=[
                       'nominal_p_hypotheses_20260925/critique.md absent on disk',
                       'nominal_p_hypotheses_20260925/investigations_verified.json '
                       'absent on disk'])

    # ---- gate 3 first: the constructions are the first round's ----------
    ref3 = {
        'a_MODEL': [0.050169293478260865, 0.01012364130434783, 0.0010027173913043485],
        'a_DECOUPLE': [0.05193097826086956, 0.011562228260869573, 0.00155054347826087],
        't_MODEL': [0.04995864130434782, 0.01001521739130435, 0.0010040760869565219],
        't_DECOUPLE': [0.050008858695652185, 0.010080869565217391, 0.001019673913043478]}
    fr = first_round_channel_rates(d)
    g3 = {nm: dict(mine=fr[nm], first_round=ref3[nm],
                   max_abs=float(np.max(np.abs(np.subtract(fr[nm], ref3[nm])))))
          for nm in ref3}
    print('G3 first-round channel constructions:', json.dumps(g3), flush=True)
    if any(v['max_abs'] > 1e-12 for v in g3.values()):
        raise SystemExit('G3 FAILED: a channel construction differs from the first round')
    summary['gate_G3'] = g3
    print(f'  [{time.time() - t0:.0f}s]', flush=True)

    # ---- shared inputs for workers --------------------------------------
    rng = np.random.RandomState(SEED)
    perms = np.array([rng.permutation(N) for _ in range(N_PERM)])
    base = np.random.SeedSequence(SEED).spawn(24)
    names = ('am', 'ad', 'tm', 'td', 'x', 'boot', 'label')
    ss = dict(zip(names, base[16:23]))
    gene_ss = {nm: ss[nm].spawn(G) for nm in ('am', 'ad', 'tm', 'td')}
    xr = np.random.default_rng(ss['x'])
    sigma = np.stack([xr.permutation(N_PERM) for _ in range(KX)])
    G_.update(a=d['a'].astype(float), va=d['va'].astype(float), s=d['s'].astype(float),
              t=d['t'].astype(float), vt=d['vt'].astype(float), g=d['g'].astype(float),
              Zd=np.column_stack([np.ones(N), C]), perms=perms,
              inv=np.argsort(perms, axis=1), N=N, doft=doft, gene_ss=gene_ss,
              sigma=sigma)
    with Pool(N_WORKERS) as pool:
        res = pool.map(gene_work, range(G))
    res = sorted(res, key=lambda r: r['k'])
    print(f'arms computed [{time.time() - t0:.0f}s]', flush=True)

    # ---- G1: REAL against the instrument --------------------------------
    L = pd.read_csv(INST / 'null_long.tsv.gz', sep='\t').set_index(['gene', 'perm']).sort_index()
    g1 = dict(max_rel=0.0, max_abs_small=0.0)
    for k, gname in enumerate(genes):
        ref = L.loc[gname]
        for col, key in (('t2_a', 'a'), ('t2_t', 't'), ('t2_b', 'b')):
            rv = ref[col].reindex(range(N_PERM)).values
            mv = res[k]['real_t2'][key]
            if np.isnan(rv).any():
                raise SystemExit(f'G1 FAILED: instrument rows missing for {gname}')
            big = rv >= 1e-3
            g1['max_rel'] = max(g1['max_rel'],
                                float(np.max(np.abs(mv[big] - rv[big]) / rv[big])))
            g1['max_abs_small'] = max(g1['max_abs_small'],
                                      float(np.max(np.abs(mv[~big] - rv[~big]), initial=0.0)))
    counts_inst = {c: [int((L[c].values < al).sum()) for al in ALPHAS]
                   for c in ('p_a', 'p_t', 'p_b')}
    counts_mine = {'p_a': np.sum([r['chan']['a_R'][0] for r in res], 0),
                   'p_t': np.sum([r['chan']['t_R'][0] for r in res], 0),
                   'p_b': np.sum([r['rej']['REAL'][0] for r in res], 0)}
    counts_mine = {c: [int(x) for x in v] for c, v in counts_mine.items()}
    print(f'G1 REAL vs instrument: {g1}; counts mine {counts_mine} inst {counts_inst}',
          flush=True)
    if g1['max_rel'] > 1e-9 or g1['max_abs_small'] > 1e-12 or counts_mine != counts_inst:
        raise SystemExit('G1 FAILED')
    summary['gate_G1'] = dict(**g1, counts=counts_mine, instrument_counts=counts_inst)

    # ---- G2: comparator arms ---------------------------------------------
    ref2 = {'b_cap': [5268, 1120, 114], 'b_cap_totols': [4709, 944, 95],
            'a_cap': [4413, 967, 128], 't_ols': [4560, 910, 92]}
    mine2 = {'b_cap': np.sum([r['rej']['CAP'][0] for r in res], 0),
             'b_cap_totols': np.sum([r['rej']['CAP_PLUS_TOTAL_UNWEIGHTED'][0] for r in res], 0),
             'a_cap': np.sum([r['chan']['a_CAP'][0] for r in res], 0),
             't_ols': np.sum([r['chan']['t_OLS'][0] for r in res], 0)}
    mine2 = {k_: [int(x) for x in v] for k_, v in mine2.items()}
    print(f'G2 comparator arms: mine {mine2} comparator {ref2}', flush=True)
    if mine2 != ref2:
        raise SystemExit('G2 FAILED')
    summary['gate_G2'] = dict(mine=mine2, comparator=ref2)

    # ---- G4 -----------------------------------------------------------------
    g4 = int(sum(r['g4_failures'] for r in res))
    if g4:
        raise SystemExit(f'G4 FAILED: {g4} re-pairings changed a channel count')
    summary['gate_G4'] = dict(failures=g4, note='total-channel counts under each of the '
                              f'{KX} re-pairings equal REAL; the allelic channel is not '
                              're-indexed at all')

    # ---- per-gene rates, (G, arm) x alpha; per-set pooled rates ------------
    boot_rng = np.random.default_rng(ss['boot'])
    idx = boot_rng.integers(0, G, size=(N_BOOT, G))
    rate_g = {arm: np.stack([r['rej'][arm].mean(0) / N_PERM for r in res])
              for arm in ARMS}                                        # (G, 3)
    per_set = {arm: np.sum([r['rej'][arm] for r in res], 0) / (G * N_PERM)
               for arm in ARMS}                                       # (K_arm, 3)
    arm_rows = []
    summary['arms'] = {}
    for arm in ARMS:
        summary['arms'][arm] = {}
        for ai, al in enumerate(ALPHAS):
            est, lo, hi, _ = boot_stats(rate_g[arm][:, ai], idx)
            ps = per_set[arm][:, ai]
            mcsd = float(ps.std(ddof=1)) if len(ps) > 1 else 0.0
            row = dict(arm=arm, alpha=al, rate=est, lo=lo, hi=hi, ratio=est / al,
                       K=len(ps), mc_sd_per_set=mcsd, mc_se_mean=mcsd / np.sqrt(len(ps)),
                       states=str(GRID.get(arm, ('R', 'R re-paired'))))
            arm_rows.append(row)
            summary['arms'][arm][str(al)] = {k_: v for k_, v in row.items()
                                              if k_ not in ('arm', 'alpha')}
    AR = pd.DataFrame(arm_rows)
    AR.to_csv(OUT / 'arm_rates.tsv', sep='\t', index=False)
    pd.set_option('display.width', 250)
    print(AR.round(5).to_string(), flush=True)

    # channel-alone rates of every state
    summary['channel_alone'] = {}
    for key in res[0]['chan']:
        arr = np.sum([r['chan'][key] for r in res], 0) / (G * N_PERM)   # (K_state, 3)
        summary['channel_alone'][key] = {
            str(al): dict(rate=float(arr[:, ai].mean()),
                          mc_sd=float(arr[:, ai].std(ddof=1)) if len(arr) > 1 else 0.0)
            for ai, al in enumerate(ALPHAS)}

    # ---- budget terms ------------------------------------------------------
    def term(expr):
        pg = sum(c * rate_g[a_] for a_, c in expr.items())
        kk = max(len(per_set[a_]) for a_ in expr)
        pset = sum(c * np.broadcast_to(per_set[a_], (kk, 3)) for a_, c in expr.items())
        return pg, pset, any(a_ in HAS_REAL for a_ in expr)

    E_expr = {'REAL': 1, 'MODEL': -1}
    TERMS = {
        # insertion (main effects from MODEL)
        'ins_allelic_marginal': {'DEC_A_MODEL_T': 1, 'MODEL': -1},
        'ins_allelic_coupling': {'ALLELIC_ONLY_REAL': 1, 'DEC_A_MODEL_T': -1},
        'ins_total_marginal': {'MODEL_A_DEC_T': 1, 'MODEL': -1},
        'ins_total_coupling': {'TOTAL_ONLY_REAL': 1, 'MODEL_A_DEC_T': -1},
        'ins_interaction': {'REAL': 1, 'ALLELIC_ONLY_REAL': -1, 'TOTAL_ONLY_REAL': -1,
                            'MODEL': 1},
        'ins_interaction_cross_channel': {'REAL': 1, 'CHANNEL_INDEPENDENT': -1},
        'ins_interaction_nonlinear': {'CHANNEL_INDEPENDENT': 1, 'ALLELIC_ONLY_REAL': -1,
                                      'TOTAL_ONLY_REAL': -1, 'MODEL': 1},
        # removal (from REAL)
        'rem_allelic_coupling': {'REAL': 1, 'DECOUPLE_A': -1},
        'rem_total_coupling': {'REAL': 1, 'DECOUPLE_T': -1},
        'rem_marginals_both': {'DECOUPLE_BOTH': 1, 'MODEL': -1},
        'rem_remainder': {'REAL': -1, 'DECOUPLE_A': 1, 'DECOUPLE_T': 1,
                          'DECOUPLE_BOTH': -1},
        # Shapley allocation
        'sh_allelic_coupling': {'ALLELIC_ONLY_REAL': .5, 'DEC_A_MODEL_T': -.5,
                                'REAL': .5, 'DECOUPLE_A': -.5},
        'sh_allelic_marginal': {'DEC_A_MODEL_T': .5, 'MODEL': -.5,
                                'DECOUPLE_A': .5, 'TOTAL_ONLY_REAL': -.5},
        'sh_total_coupling': {'TOTAL_ONLY_REAL': .5, 'MODEL_A_DEC_T': -.5,
                              'REAL': .5, 'DECOUPLE_T': -.5},
        'sh_total_marginal': {'MODEL_A_DEC_T': .5, 'MODEL': -.5,
                              'DECOUPLE_T': .5, 'ALLELIC_ONLY_REAL': -.5},
        # additivity checks
        'chk_allelic_coupling_totR_minus_totM': {'REAL': 1, 'DECOUPLE_A': -1,
                                                 'ALLELIC_ONLY_REAL': -1, 'DEC_A_MODEL_T': 1},
        'chk_total_coupling_aR_minus_aM': {'REAL': 1, 'DECOUPLE_T': -1,
                                           'TOTAL_ONLY_REAL': -1, 'MODEL_A_DEC_T': 1},
        'chk_coupling_x_coupling': {'REAL': 1, 'DECOUPLE_A': -1, 'DECOUPLE_T': -1,
                                    'DECOUPLE_BOTH': 1},
        # channel main effects
        'allelic_channel_main': {'ALLELIC_ONLY_REAL': 1, 'MODEL': -1},
        'total_channel_main': {'TOTAL_ONLY_REAL': 1, 'MODEL': -1},
        # unit-scale sensitivity
        'allelic_channel_main_unit': {'ALLELIC_ONLY_REAL_UNIT': 1, 'MODEL_UNIT': -1},
        'total_channel_main_unit': {'TOTAL_ONLY_REAL_UNIT': 1, 'MODEL_UNIT': -1},
        'E_unit_REAL_minus_MODEL_UNIT': {'REAL': 1, 'MODEL_UNIT': -1},
        'MODEL_UNIT_minus_MODEL': {'MODEL_UNIT': 1, 'MODEL': -1},
        # comparators
        'REAL_minus_CAP': {'REAL': 1, 'CAP': -1},
        'REAL_minus_TOTAL_UNWEIGHTED': {'REAL': 1, 'TOTAL_UNWEIGHTED': -1},
        'REAL_minus_CAP_PLUS_TOTAL_UNWEIGHTED': {'REAL': 1, 'CAP_PLUS_TOTAL_UNWEIGHTED': -1},
    }
    Epg, Eset, _ = term(E_expr)
    Eb = np.stack([Epg[:, ai][idx].mean(1) for ai in range(3)], 1)       # (B, 3)
    budget_rows = []
    summary['budget'] = {}

    def record(name, expr, pg, pset, has_real):
        summary['budget'][name] = {}
        for ai, al in enumerate(ALPHAS):
            est, lo, hi, b = boot_stats(pg[:, ai], idx)
            sd = float(pset[:, ai].std(ddof=1)) if len(pset) > 1 else 0.0
            se = sd / np.sqrt(len(pset)) if len(pset) > 1 else 0.0
            floor = sd if has_real else se
            ok = Eb[:, ai] > 0
            shb = b[ok] / Eb[ok, ai]
            excl0 = bool(lo > 0 or hi < 0)
            row = dict(term=name, alpha=al, value=est, lo=lo, hi=hi,
                       mc_sd_per_set=sd, mc_se_mean=se,
                       floor=floor, floor_kind='per-set sd' if has_real else 'se of mean',
                       clears=bool(excl0 and abs(est) > 2 * floor),
                       clears_vs_se_of_mean=bool(excl0 and abs(est) > 2 * se),
                       clears_vs_per_set_sd=bool(excl0 and abs(est) > 2 * sd),
                       share_of_E=est / float(Epg[:, ai].mean()),
                       share_lo=float(np.quantile(shb, .025)),
                       share_hi=float(np.quantile(shb, .975)),
                       expr=' '.join(f'{c:+g}*{a_}' for a_, c in expr.items()))
            budget_rows.append(row)
            summary['budget'][name][str(al)] = {k_: v for k_, v in row.items()
                                                 if k_ not in ('term', 'alpha')}
    record('E_REAL_minus_MODEL', E_expr, Epg, Eset, True)
    record('reference_MODEL_minus_nominal', {'MODEL': 1},
           rate_g['MODEL'] - np.array(ALPHAS)[None, :],
           per_set['MODEL'] - np.array(ALPHAS)[None, :], False)
    for name, expr in TERMS.items():
        pg, pset, hr = term(expr)
        record(name, expr, pg, pset, hr)
    BT = pd.DataFrame(budget_rows)
    BT.to_csv(OUT / 'budget.tsv', sep='\t', index=False)
    print(BT[['term', 'alpha', 'value', 'lo', 'hi', 'floor', 'clears', 'share_of_E',
              'share_lo', 'share_hi']].round(5).to_string(), flush=True)
    clos = {}
    for al in map(str, ALPHAS):
        Ev = summary['budget']['E_REAL_minus_MODEL'][al]['value']
        ins = sum(summary['budget'][t_][al]['value'] for t_ in
                  ('ins_allelic_marginal', 'ins_allelic_coupling', 'ins_total_marginal',
                   'ins_total_coupling', 'ins_interaction'))
        shs = sum(summary['budget'][t_][al]['value'] for t_ in
                  ('sh_allelic_coupling', 'sh_allelic_marginal', 'sh_total_coupling',
                   'sh_total_marginal'))
        clos[al] = dict(E=Ev, insertion_sum=ins, shapley_sum=shs)
        assert abs(ins - Ev) < 1e-12 and abs(shs - Ev) < 1e-12
    summary['decomposition_closure'] = clos

    # ---- sensitivity: budget shares without the two genes of largest E at 0.05
    # (paired: removed from every arm alike; a sensitivity, not a selection
    # control, so nothing is netted against a simulated selection)
    top2 = np.argsort(-Epg[:, 0])[:2]
    keep = np.ones(G, bool); keep[top2] = False
    sens = {'dropped': [genes[i] for i in top2],
            'per_gene_E_at_0.05_of_dropped': [float(Epg[i, 0]) for i in top2],
            'share_of_pooled_E_carried_by_dropped': [
                float(Epg[top2, ai].sum() / Epg[:, ai].sum()) for ai in range(3)]}
    for name in ('E_REAL_minus_MODEL', 'ins_allelic_coupling', 'ins_allelic_marginal',
                 'ins_total_coupling', 'ins_total_marginal', 'ins_interaction',
                 'ins_interaction_cross_channel', 'sh_allelic_coupling', 'sh_total_coupling',
                 'rem_allelic_coupling', 'rem_total_coupling'):
        expr = E_expr if name == 'E_REAL_minus_MODEL' else TERMS[name]
        pg, _, _ = term(expr)
        sens[name] = {str(al): dict(value=float(pg[keep, ai].mean()),
                                    share_of_E=float(pg[keep, ai].mean() /
                                                     Epg[keep, ai].mean()))
                      for ai, al in enumerate(ALPHAS)}
    summary['sensitivity_drop_top2'] = sens
    print('drop-top-2 sensitivity:', json.dumps(sens, indent=1), flush=True)

    # ---- precision share and per-gene analysis ------------------------------
    obs = pd.read_csv(INST / 'observed.tsv', sep='\t').set_index('gene')
    PG = pd.DataFrame(dict(
        gene=genes, stratum=strata, n_a=[r['n_a'] for r in res],
        dofa=[r['dofa'] for r in res], dof_combined=[r['dof'] for r in res],
        cap_fold=[r['cap_fold'] for r in res], w_fold_a=[r['w_fold_a'] for r in res],
        sigma_a2=[r['sig_a2'] for r in res], sigma_t2=[r['sig_t2'] for r in res],
        share_a_real_median=[float(np.median(r['real_pa'])) for r in res],
        share_a_real_q10=[float(np.quantile(r['real_pa'], .1)) for r in res],
        share_a_real_q90=[float(np.quantile(r['real_pa'], .9)) for r in res],
        share_a_model_median=[r['share']['MODEL'] for r in res],
        share_a_model_unit_median=[r['share']['MODEL_UNIT'] for r in res],
        corr_ta_tt_real=[r['corr_ta_tt_real'] for r in res],
        corr_ta_tt_indep_mean=[r['corr_ta_tt_indep'][0] for r in res],
        corr_ta_tt_indep_sd=[r['corr_ta_tt_indep'][1] for r in res],
        corr_ta_tt_model_mean=[r['corr_ta_tt_model'][0] for r in res]))
    PG['share_a_observed'] = [float(obs.loc[g_, 'set'] ** 2 /
                                    (obs.loc[g_, 'sea'] ** 2 + obs.loc[g_, 'set'] ** 2))
                              for g_ in genes]
    for arm in ARMS:
        for ai, al in enumerate(ALPHAS):
            PG[f'rej{al}_{arm}'] = rate_g[arm][:, ai]
    for ai, al in enumerate(ALPHAS):
        PG[f'E{al}'] = Epg[:, ai]
        PG[f'Eallelic{al}'] = (rate_g['ALLELIC_ONLY_REAL'] - rate_g['MODEL'])[:, ai]
        PG[f'Etotal{al}'] = (rate_g['TOTAL_ONLY_REAL'] - rate_g['MODEL'])[:, ai]
        PG[f'E_over_decouple_both{al}'] = (rate_g['REAL'] - rate_g['DECOUPLE_BOTH'])[:, ai]
        # each channel ALONE (its own t^2 against its own F), real minus model
        PG[f'a_alone_excess{al}'] = [(r['chan']['a_R'][0, ai] - r['chan']['a_M'][:, ai].mean())
                                     / N_PERM for r in res]
        PG[f't_alone_excess{al}'] = [(r['chan']['t_R'][0, ai] - r['chan']['t_M'][:, ai].mean())
                                     / N_PERM for r in res]
    PG.to_csv(OUT / 'per_gene.tsv', sep='\t', index=False)

    sh = PG.share_a_real_median.values
    lab_rng = np.random.default_rng(ss['label'])
    ps = {}
    for ai, al in enumerate(ALPHAS):
        e = Epg[:, ai]
        rho = float(sps.spearmanr(sh, e)[0])
        null = np.array([sps.spearmanr(lab_rng.permutation(sh), e)[0] for _ in range(5000)])
        terc = np.searchsorted(np.quantile(sh, [1 / 3, 2 / 3]), sh)
        tb = {}
        for q in range(3):
            m = terc == q
            v = e[m]
            ii = boot_rng.integers(0, m.sum(), size=(N_BOOT, m.sum()))
            b = v[ii].mean(1)
            tb[f'tercile{q}'] = dict(
                n=int(m.sum()), share_range=[float(sh[m].min()), float(sh[m].max())],
                mean_excess=float(v.mean()), lo=float(np.quantile(b, .025)),
                hi=float(np.quantile(b, .975)),
                mean_allelic_only_excess=float(PG[f'Eallelic{al}'][m].mean()),
                mean_total_only_excess=float(PG[f'Etotal{al}'][m].mean()))
        dom = sh > 0.5
        ex_dom = float(e[dom].sum() / e.sum()) if e.sum() > 0 else float('nan')
        bdom = np.array([e[i_][dom[i_]].sum() / e[i_].sum() for i_ in idx
                         if e[i_].sum() > 0])
        bystr = {}
        for sname in sorted(set(strata)):
            m = np.array(strata) == sname
            bystr[sname] = dict(n=int(m.sum()), mean_share_a=float(sh[m].mean()),
                                mean_excess=float(e[m].mean()),
                                rho_share_excess=float(sps.spearmanr(sh[m], e[m])[0])
                                if m.sum() > 3 else None)
        ps[str(al)] = dict(
            spearman_share_vs_excess=rho,
            label_perm_p_two_sided=float((np.abs(null) >= abs(rho)).mean()),
            spearman_share_vs_allelic_only_excess=float(
                sps.spearmanr(sh, PG[f'Eallelic{al}'])[0]),
            spearman_share_vs_total_only_excess=float(
                sps.spearmanr(sh, PG[f'Etotal{al}'])[0]),
            spearman_excess_vs_allelic_only_excess=float(
                sps.spearmanr(e, PG[f'Eallelic{al}'])[0]),
            spearman_excess_vs_total_only_excess=float(
                sps.spearmanr(e, PG[f'Etotal{al}'])[0]),
            # gene property or weighting? each channel's OWN excess against pi_a
            spearman_share_vs_allelic_channel_alone_excess=float(
                sps.spearmanr(sh, PG[f'a_alone_excess{al}'])[0]),
            spearman_share_vs_total_channel_alone_excess=float(
                sps.spearmanr(sh, PG[f't_alone_excess{al}'])[0]),
            # weighted prediction of the combined excess from the channel-alone
            # excesses, pi_a*Ea + (1-pi_a)*Et, against the realized combined excess
            spearman_excess_vs_share_weighted_channel_excess=float(sps.spearmanr(
                e, sh * PG[f'a_alone_excess{al}'] + (1 - sh) * PG[f't_alone_excess{al}'])[0]),
            terciles=tb, n_allelic_dominant=int(dom.sum()),
            fraction_of_E_from_allelic_dominant=ex_dom,
            fraction_lo=float(np.quantile(bdom, .025)),
            fraction_hi=float(np.quantile(bdom, .975)),
            fraction_of_genes_allelic_dominant=float(dom.mean()),
            by_stratum=bystr)
    summary['precision_share'] = dict(
        definition='pi_a = (1/sea2)/(1/sea2 + 1/set2), per (gene, perm); gene value = '
                   'median over the 2,000 REAL permutations',
        median_over_genes_real=float(np.median(sh)),
        range_real=[float(sh.min()), float(sh.max())],
        median_over_genes_observed=float(PG.share_a_observed.median()),
        spearman_real_vs_observed=float(sps.spearmanr(sh, PG.share_a_observed)[0]),
        median_abs_diff_model_vs_real=float(np.median(np.abs(PG.share_a_model_median - sh))),
        median_abs_diff_model_unit_vs_real=float(
            np.median(np.abs(PG.share_a_model_unit_median - sh))),
        median_share_model=float(PG.share_a_model_median.median()),
        median_share_model_unit=float(PG.share_a_model_unit_median.median()),
        spearman_share_vs_n_a=float(sps.spearmanr(sh, PG.n_a)[0]),
        n_genes_dofa_below_doft=int((PG.dofa < doft).sum()),
        cap_fold_range=[float(PG.cap_fold.min()), float(PG.cap_fold.max())],
        per_alpha=ps)
    zc = ((PG.corr_ta_tt_real - PG.corr_ta_tt_indep_mean) / PG.corr_ta_tt_indep_sd).values
    summary['cross_channel'] = dict(
        definition='per gene, Pearson correlation over the 2,000 permutations of the signed '
                   'allelic and total t statistics; floor = its mean and sd over the '
                   f'{KX} re-pairings of CHANNEL_INDEPENDENT',
        mean_standardized_z=float(zc.mean()),
        pooled_z=float(zc.sum() / np.sqrt(len(zc))),
        pooled_z_p_two_sided=float(2 * sps.norm.sf(abs(zc.sum() / np.sqrt(len(zc))))),
        corr_ta_tt_real_mean=float(PG.corr_ta_tt_real.mean()),
        corr_ta_tt_real_median=float(PG.corr_ta_tt_real.median()),
        corr_ta_tt_indep_mean=float(PG.corr_ta_tt_indep_mean.mean()),
        corr_ta_tt_model_mean=float(PG.corr_ta_tt_model_mean.mean()),
        corr_ta_tt_indep_sd_per_gene_median=float(PG.corr_ta_tt_indep_sd.median()),
        n_genes_outside_2sd=int((np.abs(PG.corr_ta_tt_real - PG.corr_ta_tt_indep_mean)
                                 > 2 * PG.corr_ta_tt_indep_sd).sum()),
        expected_outside_2sd=0.0455 * G,
        genes_most_correlated=PG.reindex(PG.corr_ta_tt_real.abs()
                                         .sort_values(ascending=False).index)
        [['gene', 'corr_ta_tt_real', 'corr_ta_tt_indep_sd']].head(6).to_dict('records'))
    print(json.dumps(summary['precision_share'], indent=1, default=float))
    print(json.dumps(summary['cross_channel'], indent=1, default=float), flush=True)

    figures(AR, BT, PG)
    summary['runtime_s'] = time.time() - t0
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2, default=float))
    print(f'wrote {OUT} [{time.time() - t0:.0f}s]')


def figures(AR, BT, PG):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ladder = ['REAL', 'CHANNEL_INDEPENDENT', 'DECOUPLE_A', 'DECOUPLE_T', 'DECOUPLE_BOTH',
              'ALLELIC_ONLY_REAL', 'TOTAL_ONLY_REAL', 'DEC_A_MODEL_T', 'MODEL_A_DEC_T',
              'MODEL', 'MODEL_UNIT', 'CAP', 'TOTAL_UNWEIGHTED', 'CAP_PLUS_TOTAL_UNWEIGHTED']
    comps = [('ins_allelic_coupling', 'allelic coupling', '#2a78d6'),
             ('ins_allelic_marginal', 'allelic heavy-tailed marginal', '#8fb8ec'),
             ('ins_total_coupling', 'total coupling (scale)', '#e08a1e'),
             ('ins_total_marginal', 'total marginal', '#f3c68b'),
             ('ins_interaction_cross_channel', 'cross-channel dependence', '#7a4fb5'),
             ('ins_interaction_nonlinear', 'non-linear interaction', '#bbbbbb')]
    fig, axes = plt.subplots(2, 3, figsize=(17, 11),
                             gridspec_kw=dict(height_ratios=[1.25, 1]))
    for i, al in enumerate(ALPHAS):
        ax = axes[0, i]
        sub = AR[AR.alpha == al].set_index('arm').loc[ladder]
        y = np.arange(len(ladder))
        lo = sub.lo / al; hi = sub.hi / al; r = sub.rate / al
        ax.errorbar(r, y, xerr=[r - lo, hi - r], fmt='o', ms=4, capsize=2,
                    color='#34495e', ecolor='#95a5a6')
        ax.plot(r.loc['REAL'], y[0], 'o', color='#c0392b', ms=6)
        ax.axvline(1, color='k', lw=.6)
        ax.axvline(r.loc['REAL'], color='#c0392b', lw=.6, ls='--')
        ax.set_yticks(y); ax.set_yticklabels(ladder if i == 0 else [], fontsize=8)
        ax.invert_yaxis()
        ax.set_title(f'combined statistic, nominal {al}')
        ax.set_xlabel('rejection rate / nominal (gene-clustered 95% interval)')
        ax = axes[1, i]
        E = BT[(BT.term == 'E_REAL_minus_MODEL') & (BT.alpha == al)].iloc[0]
        pos, neg = 0.0, 0.0
        for name, lab, col in comps:
            v = BT[(BT.term == name) & (BT.alpha == al)].iloc[0].value / al
            bot = pos if v >= 0 else neg
            ax.bar(0, v, bottom=bot, color=col, label=lab, width=.5)
            if v >= 0:
                pos += v
            else:
                neg += v
        pos, neg = 0.0, 0.0
        for name, col in (('sh_allelic_coupling', '#2a78d6'),
                          ('sh_allelic_marginal', '#8fb8ec'),
                          ('sh_total_coupling', '#e08a1e'),
                          ('sh_total_marginal', '#f3c68b')):
            v = BT[(BT.term == name) & (BT.alpha == al)].iloc[0].value / al
            bot = pos if v >= 0 else neg
            ax.bar(1, v, bottom=bot, color=col, width=.5)
            if v >= 0:
                pos += v
            else:
                neg += v
        ax.errorbar([0.35, 1.35], [E.value / al] * 2,
                    yerr=[[(E.value - E.lo) / al] * 2, [(E.hi - E.value) / al] * 2],
                    fmt='k_', ms=12, capsize=3)
        ax.axhline(0, color='k', lw=.6)
        ref = BT[(BT.term == 'reference_MODEL_minus_nominal') & (BT.alpha == al)].iloc[0]
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['insertion\n(main effects + interaction)', 'Shapley\nallocation'],
                           fontsize=8)
        ax.set_ylabel('excess over MODEL, in units of nominal')
        ax.set_title(f'budget of REAL - MODEL at {al}: E = {E.value:.4f} '
                     f'[{E.lo:.4f}, {E.hi:.4f}]\nMODEL - nominal = {ref.value:+.5f}; '
                     'black bar: E with gene-bootstrap 95% interval', fontsize=9)
        if i == 0:
            ax.legend(fontsize=7, loc='upper right')
    fig.suptitle('hapmixQTL combined nominal-p excess under the records permutation null '
                 '(46 genes x 2,000 permutations; simulated arms 200 record sets each)')
    fig.tight_layout()
    fig.savefig(OUT / 'fig_budget.png', dpi=140)
    plt.close(fig)

    col = {'HIGH': '#2a6fdb', 'MID': '#e08a1e', 'LOW': '#2e9e5b'}
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    c = [col.get(s_, 'k') for s_ in PG.stratum]
    ax[0].scatter(PG.share_a_real_median, PG['E0.05'], c=c, s=22)
    for _, r in PG.iterrows():
        if abs(r['E0.05']) > 0.02:
            ax[0].annotate(r.gene, (r.share_a_real_median, r['E0.05']), fontsize=7)
    ax[0].axhline(0, color='k', lw=.6)
    ax[0].set_xlabel('allelic share of combined precision (median over permutations)')
    ax[0].set_ylabel('per-gene REAL - MODEL at 0.05')
    ax[0].set_title('combined excess against the allelic precision share')
    for s_, c_ in col.items():
        ax[0].scatter([], [], c=c_, label=s_)
    ax[0].legend(fontsize=7)
    ax[1].scatter(PG['Eallelic0.05'], PG['E0.05'], c=c, s=22)
    ax[1].scatter(PG['Etotal0.05'], PG['E0.05'], c=c, s=22, marker='x')
    lim = [min(PG['E0.05'].min(), PG['Eallelic0.05'].min(), PG['Etotal0.05'].min()),
           max(PG['E0.05'].max(), PG['Eallelic0.05'].max(), PG['Etotal0.05'].max())]
    ax[1].plot(lim, lim, 'k', lw=.6)
    ax[1].set_xlabel('channel main effect at 0.05 (o allelic-only-real, x total-only-real)')
    ax[1].set_ylabel('combined excess at 0.05')
    ax[1].set_title('which channel the per-gene combined excess follows')
    ax[2].scatter(PG.share_a_real_median, PG.share_a_model_median, c=c, s=18,
                  label='scale-matched MODEL')
    ax[2].scatter(PG.share_a_real_median, PG.share_a_model_unit_median, c=c, s=18,
                  marker='x', label='unit-scale MODEL')
    ax[2].plot([0, 1], [0, 1], 'k', lw=.6)
    ax[2].set_xlabel('allelic precision share, REAL')
    ax[2].set_ylabel('allelic precision share, MODEL arm')
    ax[2].set_title('scale matching restores the real channel weighting')
    ax[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / 'fig_precision_share.png', dpi=140)
    plt.close(fig)


if __name__ == '__main__':
    main()

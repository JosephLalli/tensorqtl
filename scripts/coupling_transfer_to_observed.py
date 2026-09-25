"""Is the weight-residual coupling a property of the GENE, and does it transfer
from the records-permutation null to the sampling null that governs
pval_nominal on observed data?

QUESTION. Under the records-permutation null (the shared instrument,
scripts/null_permutation_instrument.py) hapmixQTL's allelic nominal p rejects
0.069 / 0.020 / 0.0057 at 0.05 / 0.01 / 0.001. The leading explanation is a
coupling between the allelic weight w = 1/va and the whitened squared residual
z^2 = w a^2. Its first-order closed form is

    R_g = mean(w z^2) / (mean(w) mean(z^2))      (means over admitted records)

which is sdratio_a^2 under records permutation and is 1 in expectation under
the shipped model Var(a_j) = sigma^2 va_j. A permutation null holds the
realized record set fixed and only reassigns it, so an R_g far from 1 there can
be either (i) a stable misspecification of the variance function, which would
recur in any sample of donors, or (ii) a realized quirk of this record set --
one or a few large residuals that happen to sit on high-weight records. Only
(i) says anything about pval_nominal on observed data, which is governed by the
SAMPLING null (fresh errors at fixed donors and genotypes), not by
reassignment. This script separates the two.

DESIGN. Everything reads the instrument's inputs_at_lead.npz (46 genes x 92
donors, fixed variant = RASQUAL's observed lead; allelic regressor
s = xL - xR, total regressor g = dosage/2, 17 covariates).

  (a) SPLIT-HALF RELIABILITY. 200 random splits of the 92 donors into halves
      of 46 (SeedSequence(42) child). In each half and gene compute R_g, and
      separately the rank correlation of w with z^2 (a coupling measure no
      single record can dominate). Across the 46 genes, correlate half A with
      half B (Spearman, and Pearson on log R), and average over the splits.
      Reference: 200 MODEL-GENERATED record sets (z iid N(0,1) at each gene's
      real w, so Var(a) = va exactly and R has no gene-level signal beyond
      what the weight distribution induces), each put through the SAME 200
      splits. Run with all records, without each gene's top record (argmax of
      w z^2, defined on the full record set; the model sets drop their own),
      and without its top three; and on raw a and on e = a - b_obs s (lead
      effect removed, so real allelic signal at the lead cannot pass for a
      gene property). The across-gene mean of log R and of the rank coupling
      is reported against the model too (a common component).
      RECURRENCE: each gene's excess log R over its model expectation is
      shrunk toward the across-gene mean by the Spearman-Brown full-sample
      reliability of the split-half Pearson of log R (2r/(1+r)), and the
      shrunk R is fed to a scale mixture (per gene, P(F(1, n_a-1) > q/R),
      averaged) -- the part of the excess predicted to recur in new donors.

  (b) SAMPLING-NULL TRANSFER. Fresh Gaussian errors, 2,000 replicates per
      gene, under variance structures the data imply, each evaluated under
      TWO nulls on the SAME fresh draws (common random numbers):
        SAMP  records stay at their own positions (the sampling null)
        PERM  replicate k is then reassigned by permutation k of the
              instrument's RandomState(42) stream
      Allelic variance structures (a-units, per record j):
        MODEL   va_j                                  (the shipped model)
        E2_RES  e_j^2, e = a - b_obs s                (own realized residual)
        E2_RAW  a_j^2                                 (lead effect not removed)
        POW     va_j exp(alpha + gamma log w_j)       (ML power law in w on e)
        BIN3    va_j m_k, m_k = mean z^2 in w-tertile (step function on e)
      Total channel (whitened scale, for combined rates): MODEL (1) and E2
      (own squared whitened residual on [1, C, g] over 1 - leverage).
      Controls: the data-derived structures (E2_RES, POW, BIN3, total E2)
      are rebuilt from 10 model-generated record sets and run through the
      same pipeline, giving the rate each construction produces when the
      shipped model is TRUE (mean and Monte Carlo sd across record sets).
      DECOUPLED controls: the gene's own whitened residuals (lead effect
      removed) shuffled among its admitted records, 10 shuffles -- both real
      marginals kept, only the weight-residual pairing broken -- then the
      same structures and nulls. Real minus decoupled isolates the coupling
      from heavy tails and from fit noise. ALLELIC ONLY: the total channel is
      not shuffled, so total rows carry no decoupled control and combined
      rows are half-decoupled (labelled in arm_rates.tsv).

  (c) SIGN-FLIP NULL. Haplotype-label swap: a_j -> eps_j a_j with Rademacher
      eps, records and genotypes at their own positions, 2,000 flips per gene
      (SeedSequence(42) child), allelic channel only. On raw a (what the
      2026-09-24 30-draw measurement did) and on e = a - b_obs s (lead effect
      removed, because a real allelic effect at the lead sits on the
      heterozygotes by construction and survives a sign flip as variance).
      Also the recorded 30-draw flips are regenerated from the 2026-09-24
      RandomState(42) stream (30 permutations, then 30 flips) and compared
      against allelic_null_schemes_20260924/pvals.tsv. Scale-mixture test:
      the heterozygote-only analogue of R,
        R_sf = [sum_het w z^2 / sum_het w] / [(S - sum_het w z^2/sum_het w)/(n_a-1)]
      (S = sum over admitted of z^2), is the ratio E[num^2]/den over the
      expected residual mean square under flips; the predicted rate is
      mean over genes of P(F(1, dof) > q_alpha / R_sf).

HELD FIXED: donors, genotype at the lead, Gibbs variances va/vt as weights,
covariates, the fit (fit_channels's algebra, re-expressed in closed form and
vectorized). VARIES: the error draws (b), the donor halves (a), the signs (c).

GATES (abort on failure):
  1. REPRODUCTION. The vectorized REAL records-permutation arm reproduces the
     instrument's null_long.tsv.gz per-(gene, perm) t2_a and t2_t to relative
     1e-9 on the RandomState(42) stream (all 92,000 rows present; rows whose
     reference t2 is below 1e-8, where a relative error measures cancellation
     in a numerically zero slope, are held instead to |t| within 1e-10), and its
     pooled rejection counts at 0.05/0.01/0.001 for p_a, p_t, p_b exactly.
  2. SAMP-MODEL is exact: under Var(a) = va with Gaussian errors the allelic
     t^2 is F(1, n_a - 1) exactly, so its pooled rates must sit within 4
     binomial sd of nominal. A failure means the simulation, not the data,
     is wrong.
  3. The vectorized sign-flip statistic equals fit_channels on flipped data
     for 20 (gene, flip) spot checks to relative 1e-9.

Master seed 42: the permutation stream is the instrument's RandomState(42);
everything else draws from np.random.SeedSequence(42) children.

Outputs: brainvar_hapmix_deploy/coupling_transfer_to_observed_20260925/
  summary.json, per_gene.tsv, arm_rates.tsv, splithalf.tsv,
  perm_minus_samp.tsv, real_minus_decoupled.tsv, fig_*.png.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from null_permutation_instrument import fit_channels   # noqa: E402

D = Path('/mnt/ssd/lalli/brainvar_hapmix_deploy')
INST = D / 'nominal_p_null_instrument_20260925'
OUT = D / 'coupling_transfer_to_observed_20260925'
SEED, EPS = 42, 1e-12
N_PERM = 2000          # instrument stream length and replicates per gene
N_SPLIT = 200
N_MODEL_SPLIT = 200    # model record sets for the split-half reference
N_MODEL_CTRL = 10      # model record sets for the (b) controls
N_DECOUPLE = 10        # within-gene residual shuffles for the (b) controls
N_FLIP = 2000
N_BOOT = 2000
ALPHAS = (0.05, 0.01, 0.001)

SS = np.random.SeedSequence(SEED)
(SS_SPLIT, SS_MODEL_SPLIT, SS_FRESH_A, SS_FRESH_T, SS_CTRL, SS_FLIP,
 SS_BOOT, SS_SPOT, SS_DECOUPLE) = SS.spawn(9)   # first 8 = spawn(8)'s


# ---------------------------------------------------------------------------
# vectorized fits (closed forms of fit_channels)
# ---------------------------------------------------------------------------
def allelic_stats(A, V, s):
    """A, V: (P, N) responses and Gibbs variances in POSITION order; s: (N,).
    Returns ba, sea2, dofa, t2 exactly as fit_channels's allelic channel."""
    k = np.isfinite(A) & np.isfinite(V) & (V > EPS)
    W = np.where(k, 1.0 / np.where(k, V, 1.0), 0.0)
    A0 = np.where(k, A, 0.0)
    num = (W * A0 * s).sum(1)
    den = (W * s * s).sum(1)
    S = (W * A0 * A0).sum(1)
    dof = np.maximum(k.sum(1) - 1, 1)
    ba = num / den
    rss = S - num * ba
    sea2 = rss / dof / den
    t2 = ba ** 2 / sea2
    return dict(ba=ba, sea2=sea2, dofa=dof, t2_a=t2)


class TotalDesign:
    """The total channel's fixed weighted design for one gene, in RECORD order
    (covariate rows travel with their record under the records null)."""

    def __init__(self, vt, C):
        k = np.isfinite(vt) & (vt > EPS)
        assert k.all(), 'total channel: every donor admitted in this cohort'
        self.sw = np.sqrt(1.0 / vt)
        Zw = np.column_stack([np.ones(len(vt)), C]) * self.sw[:, None]
        self.Q, _ = np.linalg.qr(Zw)
        self.doft = len(vt) - 1 - Zw.shape[1]
        self.Zw = Zw

    def resid(self, Y):
        """Residualize rows of Y (P, N) on the weighted design."""
        return Y - (Y @ self.Q) @ self.Q.T

    def stats(self, Yres, Xres):
        """Yres, Xres: (P, N) residualized whitened response and genotype."""
        xx = (Xres * Xres).sum(1)
        bt = (Xres * Yres).sum(1) / xx
        rss = (Yres * Yres).sum(1) - bt * bt * xx
        set2 = rss / self.doft / xx
        return dict(bt=bt, set2=set2, doft=np.full(len(bt), self.doft),
                    t2_t=bt ** 2 / set2)


def combine(al, to):
    prec = 1 / al['sea2'] + 1 / to['set2']
    b = (al['ba'] / al['sea2'] + to['bt'] / to['set2']) / prec
    return dict(t2_b=b * b * prec, dof=np.minimum(al['dofa'], to['doft']))


def pv(t2, dof):
    return sps.f.sf(t2, 1, dof)


def rec_genotype(g, perms):
    """Genotype seen by each RECORD under each permutation: record perms[p, j]
    sits at position j, which carries genotype g[j]."""
    G = np.empty(perms.shape)
    G[np.arange(perms.shape[0])[:, None], perms] = g[None, :]
    return G


# ---------------------------------------------------------------------------
# coupling statistics
# ---------------------------------------------------------------------------
def R_perm(w, z2):
    return (w * z2).mean() / (w.mean() * z2.mean())


def R_het(w, z2, het):
    """Heterozygote-only analogue: E[num^2]/den over the expected residual
    mean square, for the sampling and sign-flip nulls at own positions."""
    n_a = len(w)
    num_mean = (w[het] * z2[het]).sum() / w[het].sum()
    return num_mean / ((z2.sum() - num_mean) / (n_a - 1))


def pow_fit(w, z2):
    """ML for z_j ~ N(0, exp(alpha + gamma log w_j)). Returns alpha, gamma
    with log w centered (alpha is then the log scale at the geometric mean)."""
    x = np.log(w) - np.log(w).mean()

    def nll(p):
        eta = p[0] + p[1] * x
        return np.sum(eta + z2 * np.exp(-eta))

    def grad(p):
        eta = p[0] + p[1] * x
        r = 1 - z2 * np.exp(-eta)
        return np.array([r.sum(), (r * x).sum()])

    res = optimize.minimize(nll, np.array([np.log(z2.mean()), 0.0]), jac=grad,
                            method='BFGS')
    return res.x[0], res.x[1], x


def bin3_scale(w, z2):
    q = np.quantile(w, [1 / 3, 2 / 3])
    b = np.digitize(w, q)
    m = np.array([z2[b == i].mean() for i in range(3)])
    return m[b]


def spearman_rows(X, Y):
    """Row-wise Spearman between columns of X and Y (both (K, G))."""
    rx = np.apply_along_axis(sps.rankdata, 1, X)
    ry = np.apply_along_axis(sps.rankdata, 1, Y)
    return pearson_rows(rx, ry)


def pearson_rows(X, Y):
    Xc = X - X.mean(1, keepdims=True)
    Yc = Y - Y.mean(1, keepdims=True)
    return (Xc * Yc).sum(1) / np.sqrt((Xc ** 2).sum(1) * (Yc ** 2).sum(1))


# ---------------------------------------------------------------------------
# rates with gene-clustered intervals
# ---------------------------------------------------------------------------
def clustered(rej, rng):
    """rej: (G, P) boolean. Pooled rate and gene-clustered 95% interval."""
    k = rej.sum(1).astype(float)
    n = np.full(len(k), rej.shape[1], float)
    idx = rng.integers(0, len(k), size=(N_BOOT, len(k)))
    boot = k[idx].sum(1) / n[idx].sum(1)
    return float(k.sum() / n.sum()), float(np.quantile(boot, .025)), \
        float(np.quantile(boot, .975))


def clustered_diff(rej1, rej2, rng):
    d = rej1.mean(1) - rej2.mean(1)
    idx = rng.integers(0, len(d), size=(N_BOOT, len(d)))
    boot = d[idx].mean(1)
    return float(d.mean()), float(np.quantile(boot, .025)), \
        float(np.quantile(boot, .975))


# ---------------------------------------------------------------------------
def load():
    z = np.load(INST / 'inputs_at_lead.npz', allow_pickle=True)
    return {k: z[k] for k in ('genes', 'a', 'va', 't', 'vt', 's', 'g', 'C')}


def gate_reproduction(X, perms):
    """Gate 1: REAL records-permutation arm vs the instrument."""
    L = pd.read_csv(INST / 'null_long.tsv.gz', sep='\t')
    genes = list(X['genes'])
    P = perms.shape[0]
    t2a = np.empty((len(genes), P)); t2t = np.empty_like(t2a)
    pa = np.empty_like(t2a); pt = np.empty_like(t2a); pb = np.empty_like(t2a)
    for gi in range(len(genes)):
        a, va, s = X['a'][gi], X['va'][gi], X['s'][gi]
        al = allelic_stats(a[perms], va[perms], s)
        TD = TotalDesign(X['vt'][gi], X['C'])
        yres = TD.resid((TD.sw * X['t'][gi])[None, :])
        Xres = TD.resid(TD.sw[None, :] * rec_genotype(X['g'][gi], perms))
        to = TD.stats(np.broadcast_to(yres, Xres.shape), Xres)
        cb = combine(al, to)
        t2a[gi], t2t[gi] = al['t2_a'], to['t2_t']
        pa[gi], pt[gi] = pv(al['t2_a'], al['dofa']), pv(to['t2_t'], to['doft'])
        pb[gi] = pv(cb['t2_b'], cb['dof'])
    L = L.set_index(['gene', 'perm']).sort_index()
    if len(L) != len(genes) * P:
        raise SystemExit(f'gate 1 FAILED: instrument has {len(L)} rows')
    ref_a = np.stack([L.loc[g].t2_a.values for g in genes])
    ref_t = np.stack([L.loc[g].t2_t.values for g in genes])
    # Relative 1e-9 on every row whose reference t2 is >= 1e-8. Below that
    # (t2 ~ 1e-11, |t| ~ 4e-6) a relative tolerance measures floating-point
    # cancellation in a slope that is numerically zero, so those rows are held
    # to an ABSOLUTE tolerance on |t| of 1e-10 instead, and counted.
    big_a, big_t = ref_a >= 1e-8, ref_t >= 1e-8
    rel_a = np.abs(t2a - ref_a)[big_a] / ref_a[big_a]
    rel_t = np.abs(t2t - ref_t)[big_t] / ref_t[big_t]
    abs_a = np.abs(np.sqrt(t2a) - np.sqrt(ref_a)).max()
    abs_t = np.abs(np.sqrt(t2t) - np.sqrt(ref_t)).max()
    out = dict(max_rel_t2_a=float(rel_a.max()), max_rel_t2_t=float(rel_t.max()),
               n_rows_rel_tested_a=int(big_a.sum()),
               n_rows_rel_tested_t=int(big_t.sum()),
               max_abs_t_a=float(abs_a), max_abs_t_t=float(abs_t),
               min_ref_t2_a=float(ref_a.min()), min_ref_t2_t=float(ref_t.min()))
    for nm, mine, col in (('allelic', pa, 'p_a'), ('total', pt, 'p_t'),
                          ('combined', pb, 'p_b')):
        ref = np.stack([L.loc[g][col].values for g in genes])
        for al in ALPHAS:
            cm, cr = int((mine < al).sum()), int((ref < al).sum())
            out[f'count_{nm}_{al}'] = [cm, cr]
            if cm != cr:
                raise SystemExit(f'gate 1 FAILED: {nm} count at {al}: {cm} vs {cr}')
    print(f"gate 1: t2_a max rel {out['max_rel_t2_a']:.2e} over "
          f"{out['n_rows_rel_tested_a']} rows, t2_t max rel "
          f"{out['max_rel_t2_t']:.2e} over {out['n_rows_rel_tested_t']} rows; "
          f"max |t| diff {abs_a:.1e} / {abs_t:.1e}; pooled counts identical",
          flush=True)
    if rel_a.max() > 1e-9 or rel_t.max() > 1e-9 or max(abs_a, abs_t) > 1e-10:
        raise SystemExit('gate 1 FAILED: t2 does not reproduce to 1e-9')
    return out, dict(pa=pa, pt=pt, pb=pb)


# ---------------------------------------------------------------------------
# (a) split-half reliability
# ---------------------------------------------------------------------------
def split_half(X):
    """Split-half reliability of the coupling, on raw a and on a with the lead
    effect removed (e = a - b_obs s, so real allelic signal AT THE LEAD cannot
    masquerade as a gene property)."""
    G, N = X['a'].shape
    rng = np.random.default_rng(SS_SPLIT)
    halves = np.zeros((N_SPLIT, N), bool)
    for k in range(N_SPLIT):
        halves[k, rng.permutation(N)[:N // 2]] = True

    adm = X['va'] > EPS
    W = np.where(adm, 1.0 / np.where(adm, X['va'], 1.0), 0.0)
    S = X['s']

    def z2_of(A):
        A0 = np.where(adm, A, 0.0)
        b = (W * S * A0).sum(1) / (W * S * S).sum(1)
        E = np.where(adm, A0 - b[:, None] * S, 0.0)
        return {'raw': W * A0 ** 2, 'res': W * E ** 2}

    Z2real = z2_of(X['a'])
    for v in Z2real.values():
        for gi in range(G):
            m = adm[gi]
            if len(np.unique(W[gi, m])) < m.sum() or len(np.unique(v[gi, m])) < m.sum():
                raise SystemExit('tied w or z^2 among admitted records: ranks invalid')

    def summarize(o):
        lA, lB = np.log(o['RA']), np.log(o['RB'])
        return dict(spear_R=spearman_rows(lA, lB), pear_logR=pearson_rows(lA, lB),
                    pear_rho=pearson_rows(o['rhoA'], o['rhoB']),
                    mean_logR=0.5 * (lA.mean(1) + lB.mean(1)),
                    mean_rho=0.5 * (o['rhoA'].mean(1) + o['rhoB'].mean(1)))

    mrng = np.random.default_rng(SS_MODEL_SPLIT)
    sqv = np.sqrt(np.where(adm, X['va'], 0.0))
    Zmodel = [z2_of(mrng.standard_normal((G, N)) * sqv)
              for _ in range(N_MODEL_SPLIT)]

    def full_R(Z2):
        n = adm.sum(1)
        return ((W * Z2).sum(1) / n) / ((W.sum(1) / n) * (Z2.sum(1) / n))

    res, rows, pergene = {}, [], {}
    for ver in ('raw', 'res'):
        logR = np.log(full_R(Z2real[ver]))
        mlog = np.stack([np.log(full_R(Zm[ver])) for Zm in Zmodel])  # M, G
        pergene[f'logR_{ver}'] = logR
        pergene[f'model_logR_mean_{ver}'] = mlog.mean(0)
        pergene[f'model_logR_q025_{ver}'] = np.quantile(mlog, .025, axis=0)
        pergene[f'model_logR_q975_{ver}'] = np.quantile(mlog, .975, axis=0)
        exc = logR - mlog.mean(0)
        mexc = (mlog - mlog.mean(0)).mean(1)          # model spread of the mean
        res[f'{ver}_full_logR'] = dict(
            mean_excess_over_model=float(exc.mean()),
            model_sd_of_mean_excess=float(mexc.std(ddof=1)),
            n_above_model_q975=int((logR > pergene[f'model_logR_q975_{ver}']).sum()),
            n_below_model_q025=int((logR < pergene[f'model_logR_q025_{ver}']).sum()),
            sd_across_genes_real=float(logR.std(ddof=1)),
            sd_across_genes_model=float(mlog.std(axis=1, ddof=1).mean()))
        for drop in (0, 1, 3):
            real = summarize(stats_for_fast(Z2real[ver], W, adm, halves, drop))
            mod = {k: [] for k in real}
            for Zm in Zmodel:
                sm = summarize(stats_for_fast(Zm[ver], W, adm, halves, drop))
                for k in mod:
                    mod[k].append(float(np.mean(sm[k])))
            d = {}
            for k in real:
                mv = np.array(mod[k])
                rv = float(np.mean(real[k]))
                d[k] = dict(real_mean=rv,
                            real_split_q025=float(np.quantile(real[k], .025)),
                            real_split_q975=float(np.quantile(real[k], .975)),
                            model_mean=float(mv.mean()),
                            model_sd=float(mv.std(ddof=1)),
                            model_q975=float(np.quantile(mv, .975)),
                            p_model_ge_real=float((1 + (mv >= rv).sum()) /
                                                  (1 + len(mv))))
                if k in ('spear_R', 'pear_logR', 'pear_rho'):
                    d[k]['spearman_brown_full'] = float(2 * rv / (1 + rv))
                rows.append(dict(version=ver, drop_top=drop, statistic=k, **d[k]))
            res[f'{ver}_drop_top{drop}'] = d
            print(f"(a) {ver} drop top {drop}: " + '; '.join(
                f"{k} real {d[k]['real_mean']:.3f} model {d[k]['model_mean']:.3f}"
                f"+-{d[k]['model_sd']:.3f} p={d[k]['p_model_ge_real']:.3f}"
                for k in d), flush=True)
    for ver in ('raw', 'res'):
        pergene[f'rho_full_{ver}'] = np.array(
            [sps.spearmanr(W[gi, adm[gi]], Z2real[ver][gi, adm[gi]])[0]
             for gi in range(G)])
    pergene = pd.DataFrame(pergene)
    pergene.insert(0, 'gene', X['genes'])
    return res, pd.DataFrame(rows), pergene


def stats_for_fast(Z2, W, adm, halves, drop_top):
    """R and the rank coupling rho(w, z^2) in each half: (K, G) arrays for
    halves A and B. Ranks use strict less-than counts (ties get the minimum
    rank; ties are checked to be absent in main)."""
    G = W.shape[0]
    keep = adm.copy()
    if drop_top:
        top = np.argsort(-(W * Z2), axis=1)[:, :drop_top]
        keep[np.arange(G)[:, None], top] = False
    out = {}
    for nm, H in (('A', halves), ('B', ~halves)):
        M = H[:, None, :] & keep[None, :, :]
        Mf = M.astype(float)
        n = Mf.sum(2)
        mw = (Mf * W).sum(2) / n
        mz = (Mf * Z2).sum(2) / n
        out['R' + nm] = (Mf * W * Z2).sum(2) / n / (mw * mz)
        # ranks within the kept set: rank of x among kept = count of kept <= x
        rw = _masked_ranks(W, M)
        rz = _masked_ranks(Z2, M)
        rwc = np.where(M, rw - ((n + 1) / 2)[..., None], 0.0)
        rzc = np.where(M, rz - ((n + 1) / 2)[..., None], 0.0)
        out['rho' + nm] = (rwc * rzc).sum(2) / np.sqrt((rwc ** 2).sum(2) *
                                                        (rzc ** 2).sum(2))
    return out


def _masked_ranks(V, M):
    """Ranks of V[g, j] among {j : M[k, g, j]} for every k, g (no ties)."""
    # V: (G, N); M: (K, G, N). rank = 1 + #{i kept : V[g,i] < V[g,j]}
    less = (V[:, None, :] < V[:, :, None])          # G, N(j), N(i)
    return 1.0 + np.einsum('gji,kgi->kgj', less.astype(float), M.astype(float))


# ---------------------------------------------------------------------------
# (b) sampling-null transfer
# ---------------------------------------------------------------------------
def build_structures(a, va, s, t, vt, g, C):
    """Allelic variance (a-units, per record) and total whitened variance for
    every data-derived structure, from one record set."""
    adm = va > EPS
    w = np.where(adm, 1 / np.where(adm, va, 1), 0.0)
    het = adm & (s != 0)
    b_obs = (w * s * a).sum() / (w * s * s).sum()
    e = np.where(adm, a - b_obs * s, 0.0)
    z2e = w * e ** 2
    alpha, gamma, x = pow_fit(w[adm], z2e[adm])
    powv = np.zeros_like(va); powv[adm] = va[adm] * np.exp(alpha + gamma * x)
    binv = np.zeros_like(va); binv[adm] = va[adm] * bin3_scale(w[adm], z2e[adm])
    st = dict(MODEL=np.where(adm, va, 0.0), E2_RES=e ** 2,
              E2_RAW=np.where(adm, a, 0.0) ** 2, POW=powv, BIN3=binv)
    TD = TotalDesign(vt, C)
    y = TD.sw * t
    Xf = np.column_stack([TD.Zw, TD.sw * g])
    Qf, _ = np.linalg.qr(Xf)
    ef = y - Qf @ (Qf.T @ y)
    h = (Qf ** 2).sum(1)
    tv = dict(MODEL=np.ones(len(t)), E2=ef ** 2 / (1 - h))
    diag = dict(gamma=gamma, b_obs=b_obs,
                R_perm_res=R_perm(w[adm], z2e[adm]),
                R_het_res=R_het(w[adm], z2e[adm], het[adm]),
                R_het_raw=R_het(w[adm], (w * a * a)[adm], het[adm]),
                R_perm_raw=R_perm(w[adm], (w * a * a)[adm]),
                R_perm_pow=R_perm(w[adm], (w * powv)[adm]),
                R_het_pow=R_het(w[adm], (w * powv)[adm], het[adm]),
                R_perm_bin=R_perm(w[adm], (w * binv)[adm]),
                R_het_bin=R_het(w[adm], (w * binv)[adm], het[adm]),
                n_a=int(adm.sum()), n_het=int(het.sum()))
    return st, tv, TD, diag


def run_arms(st, tv, TD, va, s, g, perms, xi_a, xi_t):
    """Rejection indicators per arm x null x channel. xi_a, xi_t: (P, N)."""
    P = perms.shape[0]
    rows = np.arange(P)[:, None]
    Xs = TD.resid((TD.sw * g)[None, :])                         # own genotype
    Xp = TD.resid(TD.sw[None, :] * rec_genotype(g, perms))      # permuted
    out = {}
    tot = {}
    for tn, f in tv.items():
        Yres = TD.resid(np.sqrt(f)[None, :] * xi_t)
        tot[(tn, 'SAMP')] = TD.stats(Yres, np.broadcast_to(Xs, Yres.shape))
        tot[(tn, 'PERM')] = TD.stats(Yres, Xp)
    for an, var in st.items():
        Arec = np.sqrt(var)[None, :] * xi_a                     # record order
        al = {'SAMP': allelic_stats(Arec, np.broadcast_to(va, Arec.shape), s),
              'PERM': allelic_stats(Arec[rows, perms], va[perms], s)}
        for nl, r in al.items():
            out[(an, nl, 'allelic')] = pv(r['t2_a'], r['dofa'])
            tn = 'MODEL' if an == 'MODEL' else 'E2'
            cb = combine(r, tot[(tn, nl)])
            out[(an, nl, 'combined')] = pv(cb['t2_b'], cb['dof'])
    for (tn, nl), r in tot.items():
        out[(tn, nl, 'total')] = pv(r['t2_t'], r['doft'])
    return out


def transfer(X, perms):
    G, N = X['a'].shape
    rng_a = [np.random.default_rng(c) for c in SS_FRESH_A.spawn(G)]
    rng_t = [np.random.default_rng(c) for c in SS_FRESH_T.spawn(G)]
    P = perms.shape[0]
    real, diags = {}, []
    for gi in range(G):
        st, tv, TD, dg = build_structures(X['a'][gi], X['va'][gi], X['s'][gi],
                                          X['t'][gi], X['vt'][gi], X['g'][gi],
                                          X['C'])
        xi_a = rng_a[gi].standard_normal((P, N))
        xi_t = rng_t[gi].standard_normal((P, N))
        o = run_arms(st, tv, TD, X['va'][gi], X['s'][gi], X['g'][gi], perms,
                     xi_a, xi_t)
        for k, p in o.items():
            real.setdefault(k, []).append(p)
        diags.append(dict(gene=X['genes'][gi], **dg))
    real = {k: np.stack(v) for k, v in real.items()}           # (G, P)

    # model controls: rebuild the data-derived structures from model record
    # sets (Var(a) = va, Var(t_w) = 1 exactly), then run the same pipeline
    ctrl = []
    crng = np.random.default_rng(SS_CTRL)
    for m in range(N_MODEL_CTRL):
        acc = {}
        for gi in range(G):
            va, s = X['va'][gi], X['s'][gi]
            adm = va > EPS
            a0 = np.where(adm, crng.standard_normal(N) * np.sqrt(np.where(adm, va, 0)), 0.0)
            TD0 = TotalDesign(X['vt'][gi], X['C'])
            t0 = crng.standard_normal(N) / TD0.sw
            st, tv, TD, _ = build_structures(a0, va, s, t0, X['vt'][gi],
                                             X['g'][gi], X['C'])
            xi_a = crng.standard_normal((P, N))
            xi_t = crng.standard_normal((P, N))
            o = run_arms(st, tv, TD, va, s, X['g'][gi], perms, xi_a, xi_t)
            for k, p in o.items():
                acc.setdefault(k, []).append(p)
        ctrl.append({k: np.stack(v) for k, v in acc.items()})
        print(f'  (b) model control record set {m + 1}/{N_MODEL_CTRL}', flush=True)

    # decoupled controls: the gene's OWN whitened residuals (lead effect
    # removed), shuffled among its admitted records, so both marginals -- the
    # weight distribution and the heavy-tailed residual distribution -- are
    # the real ones and only their pairing is broken. What survives here is
    # heavy tails and fit noise; what disappears is the coupling.
    dec = []
    drng = np.random.default_rng(SS_DECOUPLE)
    for m in range(N_DECOUPLE):
        acc = {}
        for gi in range(G):
            a, va, s = X['a'][gi], X['va'][gi], X['s'][gi]
            adm = va > EPS
            w = np.where(adm, 1 / np.where(adm, va, 1), 0.0)
            b = (w * s * a).sum() / (w * s * s).sum()
            z = (a - b * s)[adm] * np.sqrt(w[adm])
            ad = np.zeros(N)
            ad[adm] = z[drng.permutation(adm.sum())] * np.sqrt(va[adm])
            st, tv, TD, _ = build_structures(ad, va, s, X['t'][gi], X['vt'][gi],
                                             X['g'][gi], X['C'])
            xi_a = drng.standard_normal((P, N))
            xi_t = drng.standard_normal((P, N))
            o = run_arms(st, tv, TD, va, s, X['g'][gi], perms, xi_a, xi_t)
            for k, p in o.items():
                acc.setdefault(k, []).append(p)
        dec.append({k: np.stack(v) for k, v in acc.items()})
        print(f'  (b) decoupled control {m + 1}/{N_DECOUPLE}', flush=True)
    return real, ctrl, dec, pd.DataFrame(diags)


# ---------------------------------------------------------------------------
# (c) sign flip
# ---------------------------------------------------------------------------
def sign_flip(X):
    G, N = X['a'].shape
    rng = np.random.default_rng(SS_FLIP)
    E = rng.choice(np.array([-1.0, 1.0]), size=(N_FLIP, N))
    res, pred = {}, []
    for gi in range(G):
        a, va, s = X['a'][gi], X['va'][gi], X['s'][gi]
        adm = va > EPS
        w = np.where(adm, 1 / np.where(adm, va, 1), 0.0)
        b_obs = (w * s * a).sum() / (w * s * s).sum()
        het = adm & (s != 0)
        row = dict(gene=X['genes'][gi])
        for nm, y in (('raw', a), ('res', np.where(adm, a - b_obs * s, a))):
            r = allelic_stats(E * y[None, :], np.broadcast_to(va, E.shape), s)
            p = pv(r['t2_a'], r['dofa'])
            res.setdefault(nm, []).append(p)
            Rsf = R_het(w[adm], (w * y * y)[adm], het[adm])
            dof = int(adm.sum()) - 1
            row[f'Rsf_{nm}'] = Rsf
            for al in ALPHAS:
                q = sps.f.isf(al, 1, dof)
                row[f'pred_{nm}_{al}'] = float(sps.f.sf(q / Rsf, 1, dof))
                row[f'meas_{nm}_{al}'] = float((p < al).mean())
        pred.append(row)
    return {k: np.stack(v) for k, v in res.items()}, pd.DataFrame(pred), E


def gate_signflip(X, E):
    rng = np.random.default_rng(SS_SPOT)
    G, N = X['a'].shape
    worst = 0.0
    C = X['C']
    for _ in range(20):
        gi, k = int(rng.integers(G)), int(rng.integers(E.shape[0]))
        a, va, s = X['a'][gi], X['va'][gi], X['s'][gi]
        f = fit_channels(E[k] * a, s, va, X['t'][gi], X['g'][gi], X['vt'][gi], C)
        r = allelic_stats((E[k] * a)[None, :], va[None, :], s)
        worst = max(worst, abs(r['t2_a'][0] - f['t2_a']) / abs(f['t2_a']))
    print(f'gate 3: sign-flip statistic vs fit_channels, max rel {worst:.2e}')
    if worst > 1e-9:
        raise SystemExit('gate 3 FAILED')
    return worst


def recorded_signflip(X):
    """Regenerate the 2026-09-24 30 flips and compare with its recorded p."""
    G, N = X['a'].shape
    rng = np.random.RandomState(SEED)
    _ = [rng.permutation(N) for _ in range(30)]
    flips = np.array([rng.choice([1, -1], size=N) for _ in range(30)], float)
    ref = pd.read_csv(D / 'allelic_null_schemes_20260924' / 'pvals.tsv', sep='\t')
    ref = ref[ref.scheme == 'sign_flip'].set_index(['gene', 'perm']).pval
    mine = {}
    for gi, g in enumerate(X['genes']):
        r = allelic_stats(flips * X['a'][gi][None, :],
                          np.broadcast_to(X['va'][gi], flips.shape), X['s'][gi])
        for k, p in enumerate(pv(r['t2_a'], r['dofa'])):
            mine[(g, k)] = p
    mine = pd.Series(mine)
    common = ref.index.intersection(mine.index)
    lr = np.abs(np.log(mine.loc[common].values) - np.log(ref.loc[common].values))
    return dict(n_common=int(len(common)), n_ref=int(len(ref)),
                median_abs_logratio=float(np.median(lr)),
                max_abs_logratio=float(lr.max()),
                rate05_mine=float((mine.loc[common] < .05).mean()),
                rate05_ref=float((ref.loc[common] < .05).mean()),
                rate01_mine=float((mine.loc[common] < .01).mean()),
                rate01_ref=float((ref.loc[common] < .01).mean()),
                spearman=float(sps.spearmanr(mine.loc[common],
                                             ref.loc[common])[0]))


# ---------------------------------------------------------------------------
def main():
    OUT.mkdir(exist_ok=True)
    X = load()
    G, N = X['a'].shape
    rs = np.random.RandomState(SEED)
    perms = np.array([rs.permutation(N) for _ in range(N_PERM)])
    brng = np.random.default_rng(SS_BOOT)
    summary = dict(n_genes=G, n_donors=N, n_perm=N_PERM, n_split=N_SPLIT,
                   n_model_split=N_MODEL_SPLIT, n_model_ctrl=N_MODEL_CTRL,
                   n_flip=N_FLIP)

    # ---- gate 1 ----------------------------------------------------------
    g1, realp = gate_reproduction(X, perms)
    summary['gate_reproduction'] = g1
    summary['real_perm'] = {}
    for ch, key in (('allelic', 'pa'), ('total', 'pt'), ('combined', 'pb')):
        summary['real_perm'][ch] = {str(al): clustered(realp[key] < al, brng)
                                    for al in ALPHAS}

    # ---- (a) -------------------------------------------------------------
    sh, sh_tab, sh_gene = split_half(X)
    summary['split_half'] = sh
    sh_tab.to_csv(OUT / 'splithalf.tsv', sep='\t', index=False)

    # ---- (b) -------------------------------------------------------------
    real, ctrl, dec, diags = transfer(X, perms)
    rows = []
    for key in sorted(real):
        arm, nl, ch = key
        for al in ALPHAS:
            est, lo, hi = clustered(real[key] < al, brng)
            cv = np.array([(c[key] < al).mean() for c in ctrl])
            # decoupling shuffles the ALLELIC residuals only; the total channel
            # in the decoupled runs is the real one, so a total row has no
            # decoupled control and a combined row is only half-decoupled
            dv = (np.full(2, np.nan) if ch == 'total' else
                  np.array([(c[key] < al).mean() for c in dec]))
            rows.append(dict(arm=arm, null=nl, channel=ch, alpha=al, rate=est,
                             lo=lo, hi=hi, ctrl_mean=float(cv.mean()),
                             ctrl_sd=float(cv.std(ddof=1)),
                             excess_over_ctrl_in_sd=float((est - cv.mean()) /
                                                          cv.std(ddof=1))
                             if cv.std(ddof=1) > 0 else np.nan,
                             decoupled_mean=float(dv.mean()),
                             decoupled_sd=float(dv.std(ddof=1)),
                             decoupled_scope={'allelic': 'allelic shuffled',
                                              'combined': 'allelic shuffled, '
                                                          'total real',
                                              'total': 'none'}[ch]))
    arms = pd.DataFrame(rows)
    # paired PERM - SAMP differences on the same fresh draws
    drows = []
    for (arm, nl, ch) in sorted(real):
        if nl != 'PERM':
            continue
        for al in ALPHAS:
            d, lo, hi = clustered_diff(real[(arm, 'PERM', ch)] < al,
                                       real[(arm, 'SAMP', ch)] < al, brng)
            cd = np.array([((c[(arm, 'PERM', ch)] < al).mean() -
                            (c[(arm, 'SAMP', ch)] < al).mean()) for c in ctrl])
            drows.append(dict(arm=arm, channel=ch, alpha=al, perm_minus_samp=d,
                              lo=lo, hi=hi, ctrl_mean=float(cv_mean(cd)),
                              ctrl_sd=float(cd.std(ddof=1))))
    diffs = pd.DataFrame(drows)
    # real minus decoupled (paired over genes: same genes, coupling broken)
    dcrows = []
    for key in sorted(real):
        arm, nl, ch = key
        if ch != 'allelic' or arm == 'MODEL':
            continue
        for al in ALPHAS:
            rr = (real[key] < al).mean(1)
            dd = np.mean([(c[key] < al).mean(1) for c in dec], axis=0)
            dg = rr - dd
            idx = brng.integers(0, len(dg), size=(N_BOOT, len(dg)))
            bt = dg[idx].mean(1)
            dcrows.append(dict(arm=arm, null=nl, alpha=al, real=float(rr.mean()),
                               decoupled=float(dd.mean()),
                               diff=float(dg.mean()),
                               lo=float(np.quantile(bt, .025)),
                               hi=float(np.quantile(bt, .975))))
    decdiff = pd.DataFrame(dcrows)
    decdiff.to_csv(OUT / 'real_minus_decoupled.tsv', sep='\t', index=False)
    summary['real_minus_decoupled'] = decdiff.to_dict(orient='records')
    arms.to_csv(OUT / 'arm_rates.tsv', sep='\t', index=False)
    diffs.to_csv(OUT / 'perm_minus_samp.tsv', sep='\t', index=False)
    summary['arms'] = arms.to_dict(orient='records')
    summary['perm_minus_samp'] = diffs.to_dict(orient='records')

    # gate 2: SAMP-MODEL is exact
    for ch in ('allelic', 'total', 'combined'):
        for al in ALPHAS:
            r = arms[(arms.arm == 'MODEL') & (arms['null'] == 'SAMP') &
                     (arms.channel == ch) & (arms.alpha == al)].rate.iloc[0]
            sd = np.sqrt(al * (1 - al) / (G * N_PERM))
            if ch != 'combined' and abs(r - al) > 4 * sd:
                raise SystemExit(f'gate 2 FAILED: SAMP MODEL {ch} {al}: {r}')
    print('gate 2: SAMP MODEL allelic and total within 4 binomial sd of nominal')

    # closed-form predictors against the measured per-gene rates
    diags['sdratio_a_inst'] = pd.read_csv(INST / 'per_gene.tsv', sep='\t',
                                          index_col=0).loc[diags.gene,
                                                           'sdratio_a'].values
    for arm in ('E2_RES', 'POW', 'BIN3'):
        for nl in ('SAMP', 'PERM'):
            diags[f'rej05_{arm}_{nl}'] = (real[(arm, nl, 'allelic')] < .05).mean(1)
    obs = pd.read_csv(INST / 'observed.tsv', sep='\t', index_col=0)
    diags['p_a_observed'] = obs.loc[diags.gene, 'p_a'].values

    # scale-mixture predictions of the pooled rate from per-gene R
    def mix(Rcol, dof):
        return {str(al): float(np.mean(sps.f.sf(sps.f.isf(al, 1, dof) / diags[Rcol],
                                                1, dof))) for al in ALPHAS}
    dofa = diags.n_a - 1
    summary['scale_mixture_pred'] = {c: mix(c, dofa) for c in
                                     ('R_perm_raw', 'R_perm_res', 'R_het_res',
                                      'R_het_raw', 'R_perm_pow', 'R_het_pow',
                                      'R_perm_bin', 'R_het_bin')}
    summary['R_het_vs_R_perm'] = dict(
        spearman_res=float(sps.spearmanr(diags.R_het_res, diags.R_perm_res)[0]),
        median_R_perm_res=float(diags.R_perm_res.median()),
        median_R_het_res=float(diags.R_het_res.median()),
        n_R_het_lt_R_perm=int((diags.R_het_res < diags.R_perm_res).sum()))
    summary['gamma'] = dict(median=float(diags.gamma.median()),
                            n_pos=int((diags.gamma > 0).sum()))

    # what part of the in-sample coupling is predicted to RECUR in new donors:
    # shrink each gene's excess log R (over its model expectation) toward the
    # across-gene mean by the Spearman-Brown full-sample reliability of the
    # split-half Pearson correlation of log R, then feed the shrunk R to the
    # same scale mixture. rSB = 0 keeps only the common (mean) component.
    diags = diags.merge(sh_gene, on='gene')
    dofa = diags.n_a - 1
    summary['recurrence_pred'] = {}
    for ver in ('raw', 'res'):
        r = sh[f'{ver}_drop_top0']['pear_logR']['real_mean']
        rsb = 2 * r / (1 + r)
        mb = diags[f'model_logR_mean_{ver}']
        exc = diags[f'logR_{ver}'] - mb
        m = exc.mean()
        d = {}
        for nm, rel in (('in_sample', 1.0), ('shrunk', rsb), ('mean_only', 0.0)):
            Rg = np.exp(mb + m + rel * (exc - m))
            diags[f'R_{nm}_{ver}'] = Rg
            d[nm] = {str(al): float(np.mean(sps.f.sf(sps.f.isf(al, 1, dofa) / Rg,
                                                    1, dofa))) for al in ALPHAS}
        Rm = np.exp(mb)
        d['model_expectation'] = {str(al): float(np.mean(sps.f.sf(
            sps.f.isf(al, 1, dofa) / Rm, 1, dofa))) for al in ALPHAS}
        d['reliability_half'] = r
        d['reliability_full_spearman_brown'] = rsb
        d['mean_excess_logR'] = float(m)
        summary['recurrence_pred'][ver] = d
    lp = -np.log10(diags.p_a_observed.clip(lower=1e-300))
    summary['coupling_vs_observed_signal'] = {
        c: float(sps.spearmanr(diags[c], lp)[0])
        for c in ('rho_full_raw', 'rho_full_res', 'logR_raw', 'logR_res', 'gamma')}
    summary['rho_full_res'] = dict(median=float(diags.rho_full_res.median()),
                                   n_pos=int((diags.rho_full_res > 0).sum()))

    # ---- (c) -------------------------------------------------------------
    sf, sfpred, E = sign_flip(X)
    summary['gate_signflip'] = gate_signflip(X, E)
    summary['recorded_signflip_regen'] = recorded_signflip(X)
    summary['sign_flip'] = {}
    for nm, P_ in sf.items():
        d = {}
        for al in ALPHAS:
            est, lo, hi = clustered(P_ < al, brng)
            d[str(al)] = dict(rate=est, lo=lo, hi=hi,
                              pred_scale_mixture=float(sfpred[f'pred_{nm}_{al}'].mean()))
        d['spearman_pred_vs_meas_05'] = float(sps.spearmanr(
            sfpred[f'pred_{nm}_0.05'], sfpred[f'meas_{nm}_0.05'])[0])
        summary['sign_flip'][nm] = d
    for al in ALPHAS:
        dd, lo, hi = clustered_diff(sf['raw'] < al, sf['res'] < al, brng)
        summary['sign_flip'][f'raw_minus_res_{al}'] = dict(diff=dd, lo=lo, hi=hi)
    per = diags.merge(sfpred, on='gene')
    per.to_csv(OUT / 'per_gene.tsv', sep='\t', index=False)

    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2,
                                                 default=float))
    make_figures(sh_tab, arms, per)
    print_digest(summary, arms, diffs)
    print(f'\nwrote {OUT}')


def cv_mean(x):
    return float(np.mean(x))


def print_digest(summary, arms, diffs):
    print('\nreal records permutation (gated):')
    for ch, d in summary['real_perm'].items():
        print(f'  {ch:9s} ' + '  '.join(f'{a}: {v[0]:.4f} [{v[1]:.4f},{v[2]:.4f}]'
                                        for a, v in d.items()))
    print('\n(b) arm rates (rate [gene-clustered 95%]; model-control mean +- sd):')
    for _, r in arms.iterrows():
        print(f"  {r.arm:7s} {r['null']:4s} {r.channel:9s} {r.alpha:<6} "
              f"{r.rate:.4f} [{r.lo:.4f},{r.hi:.4f}]  ctrl {r.ctrl_mean:.4f}"
              f"+-{r.ctrl_sd:.4f}  decoupled {r.decoupled_mean:.4f}"
              f"+-{r.decoupled_sd:.4f}")
    print('\nreal minus decoupled:')
    for r in summary['real_minus_decoupled']:
        print(f"  {r['arm']:7s} {r['null']:4s} {r['alpha']:<6} {r['diff']:+.4f} "
              f"[{r['lo']:+.4f},{r['hi']:+.4f}]")
    print('\nrecurrence prediction:', json.dumps(summary['recurrence_pred'], indent=1))
    print('coupling vs observed signal:', summary['coupling_vs_observed_signal'],
          summary['rho_full_res'])
    print('\n(b) PERM - SAMP on the same fresh draws:')
    for _, r in diffs.iterrows():
        print(f"  {r.arm:7s} {r.channel:9s} {r.alpha:<6} {r.perm_minus_samp:+.4f} "
              f"[{r.lo:+.4f},{r.hi:+.4f}]  ctrl {r.ctrl_mean:+.4f}+-{r.ctrl_sd:.4f}")
    print('\nscale-mixture predictions:', json.dumps(summary['scale_mixture_pred'],
                                                     indent=1))
    print('\n(c) sign flip:', json.dumps(summary['sign_flip'], indent=1))
    print('recorded 30-flip regeneration:', summary['recorded_signflip_regen'])


def make_figures(sh_tab, arms, per):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 3, figsize=(13, 7.5), sharey=True)
    for j, ver in enumerate(('raw', 'res')):
        ax = axs[j]
        for i, st in enumerate(('spear_R', 'pear_logR', 'pear_rho')):
            sub = sh_tab[(sh_tab.statistic == st) & (sh_tab.version == ver)]
            x = np.arange(len(sub))
            ax[i].errorbar(x - 0.1, sub.model_mean, yerr=1.96 * sub.model_sd,
                           fmt='o', color='0.5',
                           label='model record sets (mean +- 1.96 sd)')
            ax[i].plot(x + 0.1, sub.real_mean, 'o', color='C3', label='real')
            ax[i].set_xticks(x)
            ax[i].set_xticklabels([f'drop top {d}' for d in sub.drop_top])
            ax[i].axhline(0, color='k', lw=0.5)
            ax[i].set_title({'spear_R': 'Spearman of R, half A vs B',
                             'pear_logR': 'Pearson of log R, half A vs B',
                             'pear_rho': 'Pearson of rank coupling rho(w, z^2)'}[st]
                            + ('' if ver == 'raw' else ', lead effect removed'),
                            fontsize=9)
        ax[0].set_ylabel('across-gene correlation,\nmean over 200 splits')
    axs[0][0].legend(fontsize=8)
    fig.suptitle('Split-half reliability of the weight-residual coupling across 46 genes')
    fig.tight_layout(); fig.savefig(OUT / 'fig_splithalf.png', dpi=130); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    ax[0].scatter(per.R_perm_res, per.R_het_res, s=14)
    lim = [0.3, max(per.R_perm_res.max(), per.R_het_res.max()) * 1.1]
    ax[0].plot(lim, lim, 'k--', lw=0.6); ax[0].set_xscale('log'); ax[0].set_yscale('log')
    ax[0].set_xlabel('R under records permutation (all admitted records)')
    ax[0].set_ylabel('R at own positions (heterozygote-weighted)')
    ax[0].set_title('Coupling seen by the permutation vs the sampling null', fontsize=10)
    ax[1].scatter(per['pred_raw_0.05'], per['meas_raw_0.05'], s=14, label='raw a')
    ax[1].scatter(per['pred_res_0.05'], per['meas_res_0.05'], s=14, label='lead effect removed')
    m = max(per[['pred_raw_0.05', 'meas_raw_0.05']].max()) * 1.05
    ax[1].plot([0, m], [0, m], 'k--', lw=0.6)
    ax[1].set_xlabel('scale-mixture prediction from R_sf')
    ax[1].set_ylabel('measured sign-flip rate at 0.05 (2,000 flips)')
    ax[1].legend(fontsize=8)
    ax[1].set_title('Sign-flip null, per gene', fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / 'fig_transfer_signflip.png', dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    sub = arms[(arms.channel == 'allelic') & (arms.alpha == 0.05)]
    labs = [f'{a}\n{n}' for a, n in zip(sub.arm, sub['null'])]
    x = np.arange(len(sub))
    ax.errorbar(x, sub.rate, yerr=[sub.rate - sub.lo, sub.hi - sub.rate], fmt='o',
                color='C0', label='real-data structure (gene-clustered 95%)')
    ax.errorbar(x + 0.2, sub.ctrl_mean, yerr=1.96 * sub.ctrl_sd, fmt='s', color='0.5',
                label='same construction on model record sets')
    ax.axhline(0.05, color='k', lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=8)
    ax.set_ylabel('allelic rejection rate at 0.05'); ax.legend(fontsize=8)
    ax.set_title('Sampling null (SAMP) vs records permutation (PERM) on the same fresh draws',
                 fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / 'fig_arms.png', dpi=130); plt.close(fig)


if __name__ == '__main__':
    main()

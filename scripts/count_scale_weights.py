"""What changes if hapmixQTL worked on counts rather than log counts?

QUESTION. hapmixQTL fits weighted least squares (WLS) to log-scale responses,
a = log ratio of the two haplotypes' Gibbs-draw expression and t = log total,
with weights 1/v from the Gibbs variance and a residual scale sigma^2 fitted
per variant. Its nominal p is anticonservative under the records-permutation
null, and the 2026-09-25 round traced that to a within-gene coupling between
weight and whitened squared residual. Would working on counts -- a generalised
linear model (GLM) on the counts themselves -- behave differently?

WHY THE ANSWER IS A WEIGHTING QUESTION. A GLM is fitted by iteratively
reweighted least squares (IRLS): each iteration is a WLS regression of a
working response eta + (y - mu)/(dmu/deta) on the design, with working weight
(dmu/deta)^2 / V(mu). For a log-link Poisson model the working response is
approximately log y and the weight is mu; for a logit-link binomial model on
(yL, yR) it is approximately log(yL/yR) with weight n p q. A quasi-likelihood
GLM (quasi-Poisson, quasi-binomial) multiplies V(mu) by one dispersion factor
fitted from the Pearson residuals, sum (y - mu)^2 / V(mu) / (n - k), which
plays the role hapmixQTL's sigma^2 plays. So "counts with a quasi-likelihood"
is, to first order, "log scale with counting-noise weights", with one
difference that matters here: IRLS weights are built from the FITTED mean,
whereas a log-scale WLS with counting-noise weights builds them from the
OBSERVED counts, which are part of the response.

A negative-binomial (NB, Var y = mu + phi mu^2) or beta-binomial (BB, Var y =
n p q (1 + (n - 1) rho), rho the intra-class correlation: the share of a
record's allelic variance common to all its reads) model adds a depth-
independent floor on the log scale: Var(log y) ~ 1/mu + phi and Var(logit) ~
1/(n p q) + rho/(p q).

ARMS. Response, donor set, genotype, covariates and permutation stream are
held fixed; only the weights (and, in the GLM arms, the response scale) vary.

    GIBBS                    shipped: 1/va, 1/vt (Gibbs variance plus q)
    COUNT_NOISE              allelic 1/(1/(mL+.5) + 1/(mR+.5)); total mT + 1
                             (the Poisson delta-method weight of
                             log(mT/2 + .5)); observed posterior-mean counts
    FLOOR_COMMON             count noise + ONE dispersion for all genes:
                             allelic + rho_c/(p_g q_g), total + phi_c
    FLOOR_GENE               the same with each gene's own dispersion
                             (fitted from that gene's own counts: the
                             circularity CLAUDE.md warns about; measurement)
    FLOOR_SHRUNK             per-gene dispersion shrunk to the common value,
                             limma squeezeVar form with a FIXED prior df d0=10:
                             (d_g x_g + d0 x_c)/(d_g + d0), d_g = records - 1
                             (allelic) or records - 18 (total), not tuned
    GIBBS_PLUS_COMMON_FLOOR  Gibbs variance + the common floor
    UNIT                     unweighted
    GLM_QUASI                actual quasi-binomial (logit, through origin) on
                             (mL, mR) and quasi-Poisson (log link, intercept +
                             17 covariates) on mT, Pearson dispersion
    GLM_QUASI_COMMON_FLOOR   the same with the common BB / NB variance
                             function (Williams-type quasi-BB, quasi-NB) and
                             Pearson dispersion on top

DISPERSION ESTIMATORS.
  rho (allelic, BB): maximum likelihood on y = mL, n = mL + mR over donors
      with n > 0; common = one rho shared by the 59 genes with each gene's
      mean profiled out (the estimator of allelic_overdispersion_floor.py,
      which recorded 0.0404; gate D); per gene = that gene's own ML.
  phi (total, NB): Pearson method of moments. With the mean fitted by IRLS
      under V = mu + phi mu^2 on intercept + 17 covariates (no genotype: the
      null model), phi solves sum (y - mu)^2/(mu + phi mu^2) = n - 18,
      alternating mean and phi to convergence; phi = 0 if the Pearson
      statistic at phi = 0 is already below n - 18. Common = the same equation
      summed over the 59 genes (sum over genes of n_g - 18 on the right).
  Floors use p_g = sum mL / sum (mL + mR) over the gene's admitted donors.

MEASURED per channel on the identical permutation stream (the instrument's
RandomState(42), 2,000 permutations, records moving against fixed genotype):
rejection rate at 0.05 / 0.01 / 0.001 with gene-clustered bootstrap intervals
(resample the 46 genes with replacement, recompute the pooled rate), paired
differences against GIBBS and UNIT with the same resampling, efficiency =
per-gene var(beta_hat) over permutations relative to GIBBS, the coupling ratio
R_g = mean(w z^2)/(mean(w) mean(z^2)) (z^2 = w e^2, e the null-model
residual; for GLM arms the IRLS weight and Pearson residual at the null fit),
and the across-gene sd of log R_g against its model-record noise floor.

GLM-EQUIVALENCE CHECK. 5 genes (depth ranks at the 0/25/50/75/100% positions
of the median allele-resolved read count) x the first 50 permutations:
statsmodels quasi-GLMs against the COUNT_NOISE WLS arm, with two bridges that
separate the response scale (a vs log counts) from the weights (observed vs
fitted).

GATES (abort on failure):
  A  compute_summaries_from_gibbs on the cache rows (by sample id) reproduces
     the npz A_all/Va_all/T_all/Vt_all to 1e-12
  B  GIBBS reproduces the instrument's t2_a and t2_t per (gene, perm): relative
     1e-9 where t2 >= 1e-3, absolute 1e-11 below; pooled counts exact
  C  UNIT reproduces allelic 0.0553 and total 0.0496 at 0.05
  D  the pooled BB rho reproduces 0.0404
  E  the vectorised GLM arms reproduce statsmodels on the check subset

Natural logs. Master seed 42; the instrument stream is RandomState(42); every
other stream is a child of SeedSequence(42).
Run: python3 scripts/count_scale_weights.py
"""
import contextlib
import io
import json
import sys
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize as so
from scipy import stats as sps
from scipy.special import betaln, expit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from se_fixes import resid_out                                   # noqa: E402
from tensorqtl.hapmixqtl import compute_summaries_from_gibbs    # noqa: E402

D = Path('/mnt/ssd/lalli/brainvar_hapmix_deploy')
INST = D / 'nominal_p_null_instrument_20260925'
CACHE = D / 'cache' / 'gibbs_56b63c3b37ed5df8'
OUT = D / 'count_scale_weights_20260925'
SEED, EPS, N_PERM, N_BOOT = 42, 1e-12, 2000, 2000
ALPHAS = (0.05, 0.01, 0.001)
PRIOR_DF = 10
N_MODEL_SIM = 200
N_WORKERS = 16
SS = np.random.SeedSequence(SEED).spawn(4)
RNG_BOOT = np.random.default_rng(SS[0])
RNG_MODEL = SS[1]

WLS_ARMS = ['GIBBS', 'COUNT_NOISE', 'FLOOR_COMMON', 'FLOOR_GENE', 'FLOOR_SHRUNK',
            'GIBBS_PLUS_COMMON_FLOOR', 'UNIT']
GLM_ARMS = ['GLM_QUASI', 'GLM_QUASI_COMMON_FLOOR']
# bridges between the log-scale WLS arms and the count GLMs, on the full
# stream: response = log counts, log((mL+.5)/(mR+.5)) and log(mT/2+.5);
# weights OBSERVED (count noise of the record's own counts) or FITTED (the
# GLM's null-fit IRLS weight: n/4 at p = 1/2; mu0 + 1 from the covariate-only
# quasi-Poisson fit), the last with the common BB/NB floor
BRIDGE_ARMS = ['LOGCOUNT_OBSERVED_W', 'LOGCOUNT_FITTED_W', 'LOGCOUNT_FITTED_FLOOR_W']
WLS_ARMS = WLS_ARMS + BRIDGE_ARMS
ARMS = WLS_ARMS + GLM_ARMS

log = lambda *a: print(time.strftime('%H:%M:%S'), *a, flush=True)
X = {}          # shared inputs, set before the pool forks


# ---------------------------------------------------------------------------
#  inputs
# ---------------------------------------------------------------------------

def load():
    Z = np.load(INST / 'inputs_at_lead.npz', allow_pickle=True)
    genes = [str(x) for x in Z['genes']]
    all_genes = [str(x) for x in Z['all_genes']]
    donors = [str(x) for x in Z['donors']]
    ga = open(CACHE / 'genes.txt').read().split()
    samples = open(CACHE / 'samples.txt').read().split()
    keep = [samples.index(d) for d in donors]           # by id, never position
    rows = [ga.index(g) for g in all_genes]
    draws = {}
    for k in ('YL', 'YR', 'YT'):
        mm = np.load(CACHE / f'{k}.npy', mmap_mode='r')
        draws[k] = np.asarray(mm[rows])[:, keep].astype(float)
    with contextlib.redirect_stdout(io.StringIO()):
        A, T, Va, Vt, _ = compute_summaries_from_gibbs(draws['YL'], draws['YR'],
                                                       yT=draws['YT'])
    gateA = {k: float(np.max(np.abs(M - Z[f'{k}_all'])))
             for k, M in (('A', A), ('T', T), ('Va', Va), ('Vt', Vt))}
    log('gate A (summaries from cache vs npz, max abs):', gateA)
    if max(gateA.values()) > 1e-12:
        raise SystemExit('gate A FAILED: donor mapping or summaries differ')
    post = {k: draws[k].mean(-1) for k in draws}
    rng = np.random.RandomState(SEED)
    perms = np.stack([rng.permutation(len(donors)) for _ in range(N_PERM)])
    i46 = np.array([all_genes.index(g) for g in genes])
    return dict(Z=Z, genes=genes, all_genes=all_genes, donors=donors, i46=i46,
                mL=post['YL'], mR=post['YR'], mT=post['YT'],
                a=Z['a'], va=Z['va'], t=Z['t'], vt=Z['vt'], s=Z['s'],
                g=Z['g'], C=Z['C'], perms=perms, inv=np.argsort(perms, axis=1),
                strata=[str(x) for x in Z['strata']], gateA=gateA)


# ---------------------------------------------------------------------------
#  dispersions from counts
# ---------------------------------------------------------------------------

def _bb_ll(y, n, p, rho):
    nu = (1.0 - rho) / rho
    return np.sum(betaln(y + p * nu, n - y + (1 - p) * nu) - betaln(p * nu, (1 - p) * nu))


def bb_fit(y, n):
    """ML beta-binomial (mean p, ICC rho) on non-integer counts, n > 0."""
    m = n > 0
    y, n = y[m], n[m]
    p0 = np.clip(y.sum() / n.sum(), 1e-6, 1 - 1e-6)

    def nll(th):
        v = -_bb_ll(y, n, expit(th[0]), expit(th[1]))
        return v if np.isfinite(v) else 1e300
    best = None
    for r0 in (1e-4, 1e-3, 1e-2, 1e-1):
        f = so.minimize(nll, [np.log(p0 / (1 - p0)), np.log(r0 / (1 - r0))],
                        method='L-BFGS-B', bounds=[(-8, 8), (-18, 5)])
        if best is None or f.fun < best.fun:
            best = f
    return float(expit(best.x[0])), float(expit(best.x[1]))


def bb_pooled(Y, N):
    """One rho shared by all genes, each gene's mean profiled out."""
    def prof(lr):
        rho, tot = expit(lr), 0.0
        for y, n in zip(Y, N):
            m = n > 0
            f = so.minimize_scalar(lambda t: -_bb_ll(y[m], n[m], expit(t), rho),
                                   bounds=(-8, 8), method='bounded')
            tot += f.fun
        return tot
    f = so.minimize_scalar(prof, bounds=(-14, 0), method='bounded',
                           options=dict(xatol=1e-4))
    return float(expit(f.x))


def nb_mean(y, Zd, phi, mu0=None, it=100):
    """IRLS mean of a log-link model with V = mu + phi mu^2."""
    mu = np.maximum(y, 0.5) if mu0 is None else mu0.copy()
    eta = np.log(mu)
    for _ in range(it):
        W = mu / (1 + phi * mu)
        zr = eta + (y - mu) / mu
        sw = np.sqrt(W)
        coef = np.linalg.lstsq(Zd * sw[:, None], zr * sw, rcond=None)[0]
        eta_new = Zd @ coef
        if np.max(np.abs(eta_new - eta)) < 1e-11:
            eta = eta_new
            break
        eta = eta_new
        mu = np.exp(eta)
    return np.exp(eta)


def nb_phi_mom(Ys, Zd, it=50):
    """Pearson method of moments for one phi shared by the genes in Ys.
    Solves sum_g sum_i (y - mu)^2 / (mu + phi mu^2) = sum_g (n_g - k)."""
    k = Zd.shape[1]
    phi, mus = 0.0, [None] * len(Ys)
    for _ in range(it):
        mus = [nb_mean(y, Zd, phi, m) for y, m in zip(Ys, mus)]
        target = sum(len(y) - k for y in Ys)
        pear = lambda f: sum(np.sum((y - m) ** 2 / (m + f * m ** 2))
                             for y, m in zip(Ys, mus)) - target
        new = 0.0 if pear(0.0) <= 0 else so.brentq(pear, 0.0, 1e3, xtol=1e-14)
        if abs(new - phi) <= 1e-10 * max(new, 1e-12):
            phi = new
            break
        phi = new
    return float(phi)


def dispersions(X):
    mL, mR, mT, C = X['mL'], X['mR'], X['mT'], X['C']
    Zd = np.column_stack([np.ones(len(X['donors'])), C])
    nA = mL + mR
    t0 = time.time()
    rho_c = bb_pooled(list(mL), list(nA))
    log(f'common BB rho over {len(mL)} genes: {rho_c:.5f} ({time.time()-t0:.0f} s)')
    if abs(rho_c - 0.0404042) > 5e-4:
        raise SystemExit('gate D FAILED: pooled rho does not reproduce 0.0404')
    phi_c = nb_phi_mom(list(mT), Zd)
    log(f'common NB phi over {len(mT)} genes (Pearson MoM): {phi_c:.5f}')
    out = []
    for k, i in enumerate(X['i46']):
        adm = X['va'][k] > EPS
        p_g = float(mL[i][adm].sum() / nA[i][adm].sum())
        _, rho_g = bb_fit(mL[i][adm], nA[i][adm])
        phi_g = nb_phi_mom([mT[i]], Zd)
        da, dt = int(adm.sum()) - 1, len(X['donors']) - Zd.shape[1]
        out.append(dict(gene=X['genes'][k], p_g=p_g, n_a=int(adm.sum()),
                        rho_gene=rho_g, rho_shrunk=(da * rho_g + PRIOR_DF * rho_c) / (da + PRIOR_DF),
                        phi_gene=phi_g, phi_shrunk=(dt * phi_g + PRIOR_DF * phi_c) / (dt + PRIOR_DF)))
    return rho_c, phi_c, pd.DataFrame(out)


# ---------------------------------------------------------------------------
#  fits (vectorised allelic; per-permutation total, the instrument's algebra)
# ---------------------------------------------------------------------------

def allelic_fit(y, w, K, s, inv):
    """WLS through the origin; record (y, w, K) permuted against fixed s."""
    K = K & np.isfinite(y) & np.isfinite(w) & (w > 0)
    wk, yk = np.where(K, w, 0.0), np.where(K, y, 0.0)
    n = int(K.sum())
    Sinv = s[inv]
    num = Sinv @ (wk * yk)
    den = (Sinv ** 2) @ wk
    S = float(np.sum(wk * yk ** 2))
    dof = max(n - 1, 1)
    with np.errstate(invalid='ignore', divide='ignore'):
        b = num / den
        se2 = (S - num ** 2 / den) / dof / den
        t2 = dof * num ** 2 / (den * S - num ** 2)
    R = n * np.sum(wk ** 2 * yk ** 2) / (wk.sum() * S)
    return b, np.sqrt(se2), t2, dof, R


def total_fit(y, w, Zrec, gpos, perms):
    """WLS with Zrec partialled in the sqrt(w) space, looped per permutation."""
    n = len(y)
    dof = n - Zrec.shape[1] - 1
    P = len(perms)
    b, se2 = np.full(P, np.nan), np.full(P, np.nan)
    for p, prm in enumerate(perms):
        wt = w[prm]
        Zm = Zrec[prm]
        yt = resid_out(y[prm][:, None], Zm, wt).ravel()
        xt = resid_out(gpos[:, None], Zm, wt).ravel()
        xxt = float(xt @ xt)
        b[p] = float(xt @ yt) / xxt
        et = yt - b[p] * xt
        se2[p] = float(et @ et) / dof / xxt
    t2 = b ** 2 / se2
    sw = np.sqrt(w)
    e = resid_out(y[:, None], Zrec, w).ravel() / sw
    R = n * np.sum(w ** 2 * e ** 2) / (w.sum() * np.sum(w * e ** 2))
    return b, np.sqrt(se2), t2, dof, R


def glm_allelic(yL, n, K, s, perms, rho=0.0):
    """Quasi-(beta-)binomial, logit link, through the origin, vectorised over
    permutations. V = n p q (1 + max(n-1, 0) rho), Pearson dispersion on
    n_a - 1 dof. Returns b, se, t2, dof, R (R at the null fit p = 1/2)."""
    Yp, Np, Kp = yL[perms], n[perms], K[perms]
    Yp, Np = np.where(Kp, Yp, 0.0), np.where(Kp, Np, 0.0)
    ph = 1 + np.maximum(Np - 1, 0) * rho
    nA = int(K.sum())
    b = np.zeros(len(perms))
    for _ in range(100):
        p = expit(b[:, None] * s[None, :])
        U = np.sum(s * (Yp - Np * p) / ph, 1)
        I = np.sum(s ** 2 * Np * p * (1 - p) / ph, 1)
        step = U / I
        b = b + step
        if np.max(np.abs(step)) < 1e-13:
            break
    p = expit(b[:, None] * s[None, :])
    I = np.sum(s ** 2 * Np * p * (1 - p) / ph, 1)
    with np.errstate(invalid='ignore', divide='ignore'):
        X2 = np.sum(np.where(Kp, (Yp - Np * p) ** 2 / (Np * p * (1 - p) * ph), 0.0), 1)
    dof = nA - 1
    se2 = X2 / dof / I
    # coupling at the null fit (p = 1/2): IRLS weight, Pearson residual^2
    y0, n0 = yL[K], n[K]
    ph0 = 1 + np.maximum(n0 - 1, 0) * rho
    w = n0 / 4 / ph0
    z2 = (y0 - n0 / 2) ** 2 / (n0 / 4 * ph0)
    R = len(w) * np.sum(w * z2) / (w.sum() * z2.sum())
    return b, np.sqrt(se2), b ** 2 / se2, dof, R


def glm_total_null(y, Zrec, phi):
    mu = nb_mean(y, Zrec, phi)
    w = mu / (1 + phi * mu)
    z2 = (y - mu) ** 2 / (mu + phi * mu ** 2)
    return mu, len(w) * np.sum(w * z2) / (w.sum() * z2.sum())


def glm_total(y, Zrec, gpos, perms, phi=0.0):
    """Quasi-Poisson (phi = 0) or quasi-NB, log link, design [Zrec[prm], g],
    batched IRLS over permutations, Pearson dispersion on n - k dof."""
    mu0, R = glm_total_null(y, Zrec, phi)
    Xd = np.concatenate([Zrec[perms], np.broadcast_to(gpos[None, :, None],
                                                      (len(perms), len(y), 1))], 2)
    Yp = y[perms]
    eta = np.log(mu0)[perms]
    mu = np.exp(eta)
    for it in range(100):
        W = mu / (1 + phi * mu)
        zr = eta + (Yp - mu) / mu
        XtW = Xd * W[:, :, None]
        A = np.einsum('pni,pnj->pij', XtW, Xd)
        bvec = np.einsum('pni,pn->pi', XtW, zr)
        coef = np.linalg.solve(A, bvec[:, :, None])[:, :, 0]
        eta_new = np.einsum('pni,pi->pn', Xd, coef)
        done = np.max(np.abs(eta_new - eta)) < 1e-11
        eta = eta_new
        mu = np.exp(eta)
        if done:
            break
    W = mu / (1 + phi * mu)
    A = np.einsum('pni,pn,pnj->pij', Xd, W, Xd)
    cov = np.linalg.inv(A)[:, -1, -1]
    k = Xd.shape[2]
    dof = len(y) - k
    X2 = np.sum((Yp - mu) ** 2 / (mu + phi * mu ** 2), 1)
    se2 = X2 / dof * cov
    b = coef[:, -1]
    return b, np.sqrt(se2), b ** 2 / se2, dof, R, it + 1


# ---------------------------------------------------------------------------
#  per-gene worker
# ---------------------------------------------------------------------------

def weights_for(k):
    """Allelic and total weight vectors for every WLS arm, gene k."""
    i = X['i46'][k]
    mL, mR, mT = X['mL'][i], X['mR'][i], X['mT'][i]
    va, vt = X['va'][k], X['vt'][k]
    adm = np.isfinite(X['a'][k]) & (va > EPS)
    dg = X['disp'].iloc[k]
    pq = dg.p_g * (1 - dg.p_g)
    qa = 1 / (mL + 0.5) + 1 / (mR + 0.5)
    qt = 1 / (mT + 1.0)
    safe = lambda v: np.where(adm, 1.0 / np.where(adm, v, 1.0), 0.0)
    rc, pc = X['rho_c'], X['phi_c']
    wa = {'GIBBS': safe(va), 'COUNT_NOISE': safe(qa),
          'FLOOR_COMMON': safe(qa + rc / pq), 'FLOOR_GENE': safe(qa + dg.rho_gene / pq),
          'FLOOR_SHRUNK': safe(qa + dg.rho_shrunk / pq),
          'GIBBS_PLUS_COMMON_FLOOR': safe(va + rc / pq), 'UNIT': adm.astype(float)}
    wt = {'GIBBS': 1 / vt, 'COUNT_NOISE': 1 / qt, 'FLOOR_COMMON': 1 / (qt + pc),
          'FLOOR_GENE': 1 / (qt + dg.phi_gene), 'FLOOR_SHRUNK': 1 / (qt + dg.phi_shrunk),
          'GIBBS_PLUS_COMMON_FLOOR': 1 / (vt + pc), 'UNIT': np.ones_like(vt)}
    nA = mL + mR
    Zrec = np.column_stack([np.ones(len(mT)), X['C']])
    mu0 = nb_mean(mT, Zrec, 0.0)
    wa['LOGCOUNT_OBSERVED_W'] = wa['COUNT_NOISE']
    wa['LOGCOUNT_FITTED_W'] = np.where(adm, nA / 4, 0.0)
    wa['LOGCOUNT_FITTED_FLOOR_W'] = np.where(
        adm, nA / (4 * (1 + np.maximum(nA - 1, 0) * rc)), 0.0)
    wt['LOGCOUNT_OBSERVED_W'] = wt['COUNT_NOISE']
    wt['LOGCOUNT_FITTED_W'] = mu0 + 1.0
    wt['LOGCOUNT_FITTED_FLOOR_W'] = 1 / (1 / (mu0 + 1.0) + pc)
    return adm, wa, wt, qa, qt


def model_R_sd(w, K, Zrec=None, rng=None):
    """R_g under model records (e ~ N(0, 1/w)), N_MODEL_SIM draws."""
    out = np.empty(N_MODEL_SIM)
    wk = w[K]
    for j in range(N_MODEL_SIM):
        e = rng.standard_normal(K.sum()) / np.sqrt(wk)
        if Zrec is None:
            z2 = wk * e ** 2
        else:
            r = resid_out(e[:, None], Zrec[K], wk).ravel() / np.sqrt(wk)
            z2 = wk * r ** 2
        out[j] = len(wk) * np.sum(wk * z2) / (wk.sum() * z2.sum())
    return out


def gene_worker(k):
    perms, inv = X['perms'], X['inv']
    a, t, s, g, C = X['a'][k], X['t'][k], X['s'][k], X['g'][k], X['C']
    i = X['i46'][k]
    Zrec = np.column_stack([np.ones(len(t)), C])
    adm, wa, wt, _, _ = weights_for(k)
    rng = np.random.default_rng(X['model_ss'][k])
    res = {}
    i_ = X['i46'][k]
    lr = np.log((X['mL'][i_] + .5) / (X['mR'][i_] + .5))
    lt = np.log(X['mT'][i_] / 2 + .5)
    for arm in WLS_ARMS:
        ya, yt = (lr, lt) if arm.startswith('LOGCOUNT') else (a, t)
        res[('a', arm)] = allelic_fit(ya, wa[arm], adm, s, inv)
        res[('t', arm)] = total_fit(yt, wt[arm], Zrec, g, perms)
        res[('a', arm, 'Rmodel')] = model_R_sd(wa[arm], adm, None, rng)
        res[('t', arm, 'Rmodel')] = model_R_sd(wt[arm], np.ones(len(t), bool), Zrec, rng)
    nA = X['mL'][i] + X['mR'][i]
    for arm, rho, phi in (('GLM_QUASI', 0.0, 0.0),
                          ('GLM_QUASI_COMMON_FLOOR', X['rho_c'], X['phi_c'])):
        res[('a', arm)] = glm_allelic(X['mL'][i], nA, adm, s, perms, rho)
        res[('t', arm)] = glm_total(X['mT'][i], Zrec, g, perms, phi)[:5]
    return k, res


# ---------------------------------------------------------------------------
#  summaries
# ---------------------------------------------------------------------------

def _boot_idx():
    return RNG_BOOT.integers(0, len(X['genes']), size=(N_BOOT, len(X['genes'])))


def rate(P, alpha, idx):
    k = (P < alpha).sum(1).astype(float)
    n = np.isfinite(P).sum(1).astype(float)
    b = k[idx].sum(1) / n[idx].sum(1)
    return dict(rate=float(k.sum() / n.sum()), lo=float(np.quantile(b, .025)),
                hi=float(np.quantile(b, .975)), k=int(k.sum()), n=int(n.sum()))


def diff(P1, P0, alpha, idx):
    d = ((P1 < alpha).astype(float) - (P0 < alpha).astype(float)).sum(1)
    n = (np.isfinite(P1) & np.isfinite(P0)).sum(1).astype(float)
    b = d[idx].sum(1) / n[idx].sum(1)
    lo, hi = float(np.quantile(b, .025)), float(np.quantile(b, .975))
    return dict(diff=float(d.sum() / n.sum()), lo=lo, hi=hi,
                clears_zero=bool(lo > 0 or hi < 0))


def eff_summary(V1, V0, idx):
    r = V1 / V0
    lr = np.log(r)
    b = np.exp(lr[idx].mean(1))
    return dict(median=float(np.median(r)), q25=float(np.quantile(r, .25)),
                q75=float(np.quantile(r, .75)), geomean=float(np.exp(lr.mean())),
                geomean_lo=float(np.quantile(b, .025)), geomean_hi=float(np.quantile(b, .975)),
                n_below_1=int((r < 1).sum()), n_genes=int(len(r)))


# ---------------------------------------------------------------------------
#  GLM-equivalence check (statsmodels)
# ---------------------------------------------------------------------------

def glm_check(n_genes=5, n_perm=50):
    # statsmodels.api fails to import against this scipy (_lazywhere was
    # removed); the GLM module itself does not touch that path
    from statsmodels.genmod.generalized_linear_model import GLM
    from statsmodels.genmod import families
    i46 = X['i46']
    depth = np.array([np.median((X['mL'][i] + X['mR'][i])[X['va'][k] > EPS])
                      for k, i in enumerate(i46)])
    order = np.argsort(depth)
    pick = [int(order[int(round(q * (len(order) - 1)))]) for q in (0, .25, .5, .75, 1)]
    rows = []
    for k in pick:
        i = i46[k]
        mL, mR, mT = X['mL'][i], X['mR'][i], X['mT'][i]
        a, t, s, g, C = X['a'][k], X['t'][k], X['s'][k], X['g'][k], X['C']
        adm, wa, wt, qa, qt = weights_for(k)
        Zrec = np.column_stack([np.ones(len(t)), C])
        perms = X['perms'][:n_perm]
        inv = X['inv'][:n_perm]
        nA = mL + mR
        # vectorised GLM arms on the same subset (gate E)
        ga = glm_allelic(mL, nA, adm, s, perms, 0.0)
        gt = glm_total(mT, Zrec, g, perms, 0.0)
        # WLS arms
        cn_a = allelic_fit(a, wa['COUNT_NOISE'], adm, s, inv)
        cn_t = total_fit(t, wt['COUNT_NOISE'], Zrec, g, perms)
        # bridge 1: log-count response, OBSERVED count weights
        lr = np.log((mL + .5) / (mR + .5))
        lt = np.log(mT / 2 + .5)
        b1_a = allelic_fit(lr, wa['COUNT_NOISE'], adm, s, inv)
        b1_t = total_fit(lt, wt['COUNT_NOISE'], Zrec, g, perms)
        # bridge 2: log-count response, FITTED null weights (n/4; mu_null + 1)
        mu0, _ = glm_total_null(mT, Zrec, 0.0)
        b2_a = allelic_fit(lr, np.where(adm, nA / 4, 0.0), adm, s, inv)
        b2_t = total_fit(lt, mu0 + 1.0, Zrec, g, perms)
        for p_i, prm in enumerate(perms):
            Kp = adm[prm]
            # proportions with var_weights = n: the two-column (successes,
            # failures) endog form fits the same slope but its scale='X2'
            # omits the trial counts from the Pearson statistic (measured here:
            # 0.0155 against the correct 213.5), so its t is wrong
            nk = nA[prm][Kp]
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                fa = GLM(mL[prm][Kp] / nk, s[Kp][:, None], family=families.Binomial(),
                         var_weights=nk).fit(
                    scale='X2', tol=1e-12, maxiter=200)
                ft = GLM(mT[prm], np.column_stack([Zrec[prm], g]),
                         family=families.Poisson()).fit(scale='X2', tol=1e-12,
                                                              maxiter=200)
            sa_b, sa_t = float(fa.params[0]), float(fa.tvalues[0])
            st_b, st_t = float(ft.params[-1]), float(ft.tvalues[-1])
            for ch, sb, st, arms in (
                    ('allelic', sa_b, sa_t, (('vectorised_glm', ga), ('count_noise_wls', cn_a),
                                             ('bridge_logcount_observed_w', b1_a),
                                             ('bridge_logcount_fitted_w', b2_a))),
                    ('total', st_b, st_t, (('vectorised_glm', gt), ('count_noise_wls', cn_t),
                                           ('bridge_logcount_observed_w', b1_t),
                                           ('bridge_logcount_fitted_w', b2_t)))):
                r = dict(gene=X['genes'][k], depth=float(depth[k]), perm=p_i,
                         channel=ch, glm_b=sb, glm_t=st)
                for name, f in arms:
                    r[f'{name}_b'] = float(f[0][p_i])
                    r[f'{name}_t'] = float(np.sign(f[0][p_i]) * np.sqrt(f[2][p_i]))
                rows.append(r)
    df = pd.DataFrame(rows)
    summ = {}
    for ch, sub in df.groupby('channel'):
        summ[ch] = {}
        for name in ('vectorised_glm', 'count_noise_wls', 'bridge_logcount_observed_w',
                     'bridge_logcount_fitted_w'):
            rb = np.abs(sub[f'{name}_b'] - sub.glm_b) / np.abs(sub.glm_b)
            rt = np.abs(sub[f'{name}_t'] - sub.glm_t) / np.abs(sub.glm_t)
            # relative to the permutation sd of the GLM slope/t (per gene)
            sdb = sub.groupby('gene').glm_b.transform('std')
            ab = np.abs(sub[f'{name}_b'] - sub.glm_b) / sdb
            at = np.abs(sub[f'{name}_t'] - sub.glm_t)
            summ[ch][name] = dict(
                median_rel_b=float(rb.median()), max_rel_b=float(rb.max()),
                median_rel_t=float(rt.median()), max_rel_t=float(rt.max()),
                median_abs_b_over_sd=float(ab.median()), max_abs_b_over_sd=float(ab.max()),
                median_abs_t=float(at.median()), max_abs_t=float(at.max()),
                corr_t=float(np.corrcoef(sub[f'{name}_t'], sub.glm_t)[0, 1]))
        per = sub.groupby('gene').apply(lambda d: pd.Series(dict(
            depth=d.depth.iloc[0],
            med_abs_t_cn=float(np.median(np.abs(d.count_noise_wls_t - d.glm_t))),
            med_abs_t_b1=float(np.median(np.abs(d.bridge_logcount_observed_w_t - d.glm_t))),
            med_abs_t_b2=float(np.median(np.abs(d.bridge_logcount_fitted_w_t - d.glm_t))))),
            include_groups=False)
        summ[ch]['per_gene'] = per.reset_index().to_dict(orient='records')
    return df, summ, [X['genes'][k] for k in pick]


# ---------------------------------------------------------------------------
#  figure
# ---------------------------------------------------------------------------

def make_figure(tab, eff, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ink, muted, grid, blue = '#0b0b0b', '#52514e', '#d9d8d3', '#2a78d6'
    fig, axes = plt.subplots(2, 4, figsize=(15, 8.2), sharey='row')
    order = ARMS[::-1]
    for r, ch in enumerate(('allelic', 'total')):
        for c, al in enumerate(ALPHAS):
            ax = axes[r, c]
            for y, arm in enumerate(order):
                d = tab[(tab.channel == ch) & (tab.arm == arm) & (tab.alpha == al)].iloc[0]
                ax.plot([d.lo / al, d.hi / al], [y, y], color=blue, lw=2,
                        solid_capstyle='round')
                ax.plot(d.rate / al, y, 'o', ms=8, color=blue, mec='white', mew=2)
            ax.axvline(1, color=muted, lw=1, ls='--')
            ax.set_yticks(range(len(order)))
            ax.set_yticklabels(order, fontsize=8.5, color=ink)
            ax.set_title(f'{ch}, nominal {al}', fontsize=10, color=ink, loc='left')
            ax.set_xlabel('rejection rate / nominal', fontsize=9, color=muted)
            ax.grid(axis='x', color=grid, lw=0.6)
            for sp in ('top', 'right'):
                ax.spines[sp].set_visible(False)
        ax = axes[r, 3]
        for y, arm in enumerate(order):
            e = eff[ch].get(arm)
            if e is None:
                continue
            ax.plot([e['q25'], e['q75']], [y, y], color=muted, lw=2, solid_capstyle='round')
            ax.plot(e['median'], y, 'o', ms=8, color=ink, mec='white', mew=2)
        ax.axvline(1, color=muted, lw=1, ls='--')
        ax.set_xscale('log')
        from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator
        ticks = [0.8, 0.9, 1, 1.25, 1.5, 2, 3] if r == 0 else [0.9, 0.95, 1, 1.05, 1.1]
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.set_xticklabels([f'{x:g}' for x in ticks], fontsize=8.5)
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_title(f'{ch}: var(beta) / var(beta, GIBBS)', fontsize=10, color=ink, loc='left')
        ax.set_xlabel('per-gene ratio, median and IQR (log scale)', fontsize=9, color=muted)
        ax.grid(axis='x', color=grid, lw=0.6)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
    fig.suptitle('Weights on the log scale vs GLMs on counts: records-permutation null, '
                 '46 genes x 2,000 permutations (bars: gene-clustered 95% interval)',
                 fontsize=11, color=ink, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=150, facecolor='#fcfcfb')
    plt.close(fig)


# ---------------------------------------------------------------------------

def main():
    t_start = time.time()
    OUT.mkdir(exist_ok=True)
    X.update(load())
    rho_c, phi_c, disp = dispersions(X)
    G = len(X['genes'])
    X.update(rho_c=rho_c, phi_c=phi_c, disp=disp, model_ss=RNG_MODEL.spawn(G))

    with Pool(N_WORKERS) as pool:
        out = dict(pool.map(gene_worker, range(G)))
    log(f'fits done ({time.time()-t_start:.0f} s)')

    arr = {}
    for ch in ('a', 't'):
        for arm in ARMS:
            B = np.stack([out[k][(ch, arm)][0] for k in range(G)])
            T2 = np.stack([out[k][(ch, arm)][2] for k in range(G)])
            dof = np.array([out[k][(ch, arm)][3] for k in range(G)])
            R = np.array([out[k][(ch, arm)][4] for k in range(G)])
            SE = np.stack([out[k][(ch, arm)][1] for k in range(G)])
            P = sps.f.sf(T2, 1, dof[:, None])
            arr[(ch, arm)] = dict(B=B, SE=SE, T2=T2, P=P, R=R, dof=dof)

    # ---- gate B: GIBBS reproduces the instrument --------------------------
    inst = pd.read_csv(INST / 'null_long.tsv.gz', sep='\t')
    gB = {}
    for ch, col in (('a', 't2_a'), ('t', 't2_t')):
        M = inst.pivot(index='gene', columns='perm', values=col).loc[X['genes']].values
        mine = arr[(ch, 'GIBBS')]['T2']
        big = M >= 1e-3
        rel = np.max(np.abs(mine[big] - M[big]) / M[big])
        ab = np.max(np.abs(mine[~big] - M[~big])) if (~big).any() else 0.0
        Pi = inst.pivot(index='gene', columns='perm',
                        values='p_' + ch).loc[X['genes']].values
        counts = {str(al): [int((arr[(ch, 'GIBBS')]['P'] < al).sum()), int((Pi < al).sum())]
                  for al in ALPHAS}
        gB[ch] = dict(max_rel_t2=float(rel), max_abs_small=float(ab), counts=counts)
        ok = rel < 1e-9 and ab < 1e-11 and all(c[0] == c[1] for c in counts.values())
        log(f'gate B {ch}: rel {rel:.2e}, abs(small) {ab:.2e}, counts {counts}',
            'PASS' if ok else 'FAIL')
        if not ok:
            raise SystemExit('gate B FAILED')

    # ---- rates, differences, efficiency, coupling -------------------------
    idx = _boot_idx()
    chname = {'a': 'allelic', 't': 'total'}
    rows, res = [], dict(rates={}, diff_vs_gibbs={}, diff_vs_unit={}, efficiency={},
                         coupling={}, per_gene_spread={})
    for ch in ('a', 't'):
        c = chname[ch]
        for k in res:
            res[k][c] = {}
        V0 = arr[(ch, 'GIBBS')]['B'].var(1, ddof=1)
        for arm in ARMS:
            A = arr[(ch, arm)]
            res['rates'][c][arm] = {}
            for al in ALPHAS:
                r = rate(A['P'], al, idx)
                res['rates'][c][arm][str(al)] = r
                rows.append(dict(channel=c, arm=arm, alpha=al, **r))
            res['diff_vs_gibbs'][c][arm] = {str(al): diff(A['P'], arr[(ch, 'GIBBS')]['P'], al, idx)
                                            for al in ALPHAS}
            res['diff_vs_unit'][c][arm] = {str(al): diff(A['P'], arr[(ch, 'UNIT')]['P'], al, idx)
                                           for al in ALPHAS}
            V = A['B'].var(1, ddof=1)
            res['efficiency'][c][arm] = eff_summary(V, V0, idx)
            # half-split replicate of the efficiency ratio (Monte Carlo floor)
            h1 = A['B'][:, :1000].var(1, ddof=1) / arr[(ch, 'GIBBS')]['B'][:, :1000].var(1, ddof=1)
            h2 = A['B'][:, 1000:].var(1, ddof=1) / arr[(ch, 'GIBBS')]['B'][:, 1000:].var(1, ddof=1)
            res['efficiency'][c][arm]['half_split_geomeans'] = [
                float(np.exp(np.log(h1).mean())), float(np.exp(np.log(h2).mean()))]
            res['efficiency'][c][arm]['half_split_median_abs_log_diff_per_gene'] = float(
                np.median(np.abs(np.log(h1) - np.log(h2))))
            lR = np.log(A['R'])
            cp = dict(median_R=float(np.median(A['R'])), q25=float(np.quantile(A['R'], .25)),
                      q75=float(np.quantile(A['R'], .75)), sd_logR=float(lR.std(ddof=1)),
                      min_R=float(A['R'].min()), max_R=float(A['R'].max()))
            if arm in WLS_ARMS:
                Rm = np.stack([out[k][(ch, arm, 'Rmodel')] for k in range(G)])
                # model noise floor: sd over genes of log R under model records,
                # replicated N_MODEL_SIM times
                sd_null = np.log(Rm).std(0, ddof=1)
                cp.update(model_median_R=float(np.median(Rm)),
                          model_sd_logR_mean=float(sd_null.mean()),
                          model_sd_logR_q975=float(np.quantile(sd_null, .975)),
                          n_genes_above_model_q975=int((A['R'] > np.quantile(Rm, .975, axis=1)).sum()),
                          n_genes_below_model_q025=int((A['R'] < np.quantile(Rm, .025, axis=1)).sum()))
            res['coupling'][c][arm] = cp
            pg = (A['P'] < 0.05).mean(1)
            binom_var = 0.05 * 0.95 / A['P'].shape[1]
            res['per_gene_spread'][c][arm] = dict(
                sd_rate_005=float(pg.std(ddof=1)), binomial_sd=float(np.sqrt(binom_var)),
                excess_sd=float(np.sqrt(max(pg.var(ddof=1) - binom_var, 0.0))),
                min_rate=float(pg.min()), max_rate=float(pg.max()),
                spearman_rate_vs_R=(float(sps.spearmanr(pg, A['R'])[0])
                                    if np.ptp(A['R']) > 1e-12 else None))
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / 'arm_rates.tsv', sep='\t', index=False)

    # ---- the contrasts that carry the answer, paired on (gene, perm) --------
    CONTRASTS = {
        'gibbs_shape_vs_count_noise': ('COUNT_NOISE', 'GIBBS'),
        'response_a_to_logcount': ('LOGCOUNT_OBSERVED_W', 'COUNT_NOISE'),
        'observed_to_fitted_weights': ('LOGCOUNT_FITTED_W', 'LOGCOUNT_OBSERVED_W'),
        'one_step_to_iterated_glm': ('GLM_QUASI', 'LOGCOUNT_FITTED_W'),
        'quasi_glm_vs_count_noise_wls': ('GLM_QUASI', 'COUNT_NOISE'),
        'quasi_glm_vs_gibbs': ('GLM_QUASI', 'GIBBS'),
        'floor_common_vs_unit': ('FLOOR_COMMON', 'UNIT'),
        'floor_gene_vs_common': ('FLOOR_GENE', 'FLOOR_COMMON'),
        'floor_shrunk_vs_common': ('FLOOR_SHRUNK', 'FLOOR_COMMON'),
        'glm_floor_vs_unit': ('GLM_QUASI_COMMON_FLOOR', 'UNIT'),
        'glm_floor_vs_wls_floor': ('GLM_QUASI_COMMON_FLOOR', 'FLOOR_COMMON'),
        'gibbs_plus_floor_vs_unit': ('GIBBS_PLUS_COMMON_FLOOR', 'UNIT'),
    }
    res['contrasts'] = {}
    for ch in ('a', 't'):
        c = chname[ch]
        res['contrasts'][c] = {}
        for name, (a1, a0) in CONTRASTS.items():
            r = {str(al): diff(arr[(ch, a1)]['P'], arr[(ch, a0)]['P'], al, idx)
                 for al in ALPHAS}
            r['efficiency'] = eff_summary(arr[(ch, a1)]['B'].var(1, ddof=1),
                                          arr[(ch, a0)]['B'].var(1, ddof=1), idx)
            res['contrasts'][c][name] = dict(arms=[a1, a0], **r)

    # ---- gate C ------------------------------------------------------------
    ua, ut = res['rates']['allelic']['UNIT']['0.05']['rate'], res['rates']['total']['UNIT']['0.05']['rate']
    log(f'gate C: UNIT allelic {ua:.4f} (0.0553), total {ut:.4f} (0.0496)')
    if abs(ua - 0.0553) > 5e-5 or abs(ut - 0.0496) > 5e-5:
        raise SystemExit('gate C FAILED')

    # ---- combined statistic for matched pairs ------------------------------
    res['combined'] = {}
    for arm in ARMS:
        a_, t_ = arr[('a', arm)], arr[('t', arm)]
        prec = 1 / a_['SE'] ** 2 + 1 / t_['SE'] ** 2
        b = (a_['B'] / a_['SE'] ** 2 + t_['B'] / t_['SE'] ** 2) / prec
        P = sps.f.sf(b ** 2 * prec, 1, np.minimum(a_['dof'], t_['dof'])[:, None])
        arr[('b', arm)] = dict(P=P)
        res['combined'][arm] = {str(al): rate(P, al, idx) for al in ALPHAS}
    res['combined_diff_vs_gibbs'] = {arm: {str(al): diff(arr[('b', arm)]['P'], arr[('b', 'GIBBS')]['P'], al, idx)
                                           for al in ALPHAS} for arm in ARMS}

    # ---- per-gene table ----------------------------------------------------
    pg = disp.copy()
    pg['stratum'] = X['strata']
    for ch in ('a', 't'):
        V0 = arr[(ch, 'GIBBS')]['B'].var(1, ddof=1)
        for arm in ARMS:
            A = arr[(ch, arm)]
            pg[f'{ch}_{arm}_rej005'] = (A['P'] < 0.05).mean(1)
            pg[f'{ch}_{arm}_R'] = A['R']
            pg[f'{ch}_{arm}_eff'] = A['B'].var(1, ddof=1) / V0
    # floor-to-counting ratio per gene: median over records of floor / q
    fr_a, fr_t = [], []
    for k, i in enumerate(X['i46']):
        adm, _, _, qa, qt = weights_for(k)
        d = disp.iloc[k]
        fr_a.append(float(np.median((rho_c / (d.p_g * (1 - d.p_g))) / qa[adm])))
        fr_t.append(float(np.median(phi_c / qt)))
        pg.loc[k, 'median_gibbs_over_countnoise_a'] = float(np.median(X['va'][k][adm] / qa[adm]))
        pg.loc[k, 'median_gibbs_over_countnoise_t'] = float(np.median(X['vt'][k] / qt))
        pg.loc[k, 'median_allelic_reads'] = float(np.median((X['mL'][i] + X['mR'][i])[adm]))
    pg['common_floor_over_countnoise_a'] = fr_a
    pg['common_floor_over_countnoise_t'] = fr_t
    pg.to_csv(OUT / 'per_gene.tsv', sep='\t', index=False)

    # ---- GLM-equivalence check ---------------------------------------------
    log('GLM-equivalence check (statsmodels)')
    chk, chk_sum, chk_genes = glm_check()
    chk.to_csv(OUT / 'glm_check.tsv', sep='\t', index=False)
    gE = {ch: chk_sum[ch]['vectorised_glm'] for ch in chk_sum}
    log('gate E (vectorised GLM vs statsmodels):',
        {ch: (v['max_rel_b'], v['max_abs_t']) for ch, v in gE.items()})
    if any(v['max_abs_t'] > 1e-6 or v['max_abs_b_over_sd'] > 1e-6 for v in gE.values()):
        raise SystemExit('gate E FAILED')

    make_figure(tab, res['efficiency'], OUT / 'count_scale_weights.png')

    summary = dict(
        question='counts vs log counts: weights, GLMs and dispersion floors under the '
                 'records-permutation null',
        n_genes=G, n_perm=N_PERM, alphas=ALPHAS, prior_df=PRIOR_DF,
        estimators=dict(
            rho='beta-binomial ML on (mL, mL+mR), donors with mL+mR>0; common = one rho '
                'over 59 genes with per-gene mean profiled out; per gene = own ML',
            phi='NB Pearson method of moments: mean by IRLS with V = mu + phi mu^2 on '
                '[1, 17 covariates], phi solves sum (y-mu)^2/(mu+phi mu^2) = n - 18 '
                '(summed over 59 genes for the common value), alternated to convergence',
            shrunk='(d_g x_g + 10 x_c)/(d_g + 10), d_g = n_a - 1 allelic, 92 - 18 total, '
                   'linear scale (limma squeezeVar form, prior df fixed)',
            floors='allelic floor rho/(p_g q_g), p_g = gene pooled allelic fraction; '
                   'total floor phi on the log scale'),
        rho_common=rho_c, phi_common=phi_c,
        rho_gene_quantiles=[float(x) for x in np.quantile(disp.rho_gene, [0, .25, .5, .75, 1])],
        phi_gene_quantiles=[float(x) for x in np.quantile(disp.phi_gene, [0, .25, .5, .75, 1])],
        floor_over_countnoise_median=dict(allelic=float(np.median(fr_a)), total=float(np.median(fr_t))),
        gibbs_over_countnoise_median=dict(
            allelic=float(pg.median_gibbs_over_countnoise_a.median()),
            total=float(pg.median_gibbs_over_countnoise_t.median())),
        gates=dict(A=X['gateA'], B=gB, C=dict(unit_allelic=ua, unit_total=ut), D=rho_c, E=gE),
        glm_check=dict(genes=chk_genes, n_perm=50, summary=chk_sum),
        runtime_s=time.time() - t_start, **res)
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=1, default=float))

    # ---- console table -----------------------------------------------------
    for c in ('allelic', 'total'):
        print(f'\n{c}: rate at 0.05 / 0.01 / 0.001, diff vs GIBBS at 0.05, eff, R median, sd logR')
        for arm in ARMS:
            r = res['rates'][c][arm]
            d = res['diff_vs_gibbs'][c][arm]['0.05']
            e = res['efficiency'][c][arm]
            cp = res['coupling'][c][arm]
            print(f"  {arm:24s} {r['0.05']['rate']:.4f} [{r['0.05']['lo']:.4f},{r['0.05']['hi']:.4f}]"
                  f" {r['0.01']['rate']:.4f} {r['0.001']['rate']:.5f}  "
                  f"d {d['diff']:+.4f} [{d['lo']:+.4f},{d['hi']:+.4f}]  "
                  f"eff {e['geomean']:.3f} (med {e['median']:.3f})  "
                  f"R {cp['median_R']:.3f} sdlog {cp['sd_logR']:.3f}"
                  + (f" (model {cp['model_sd_logR_mean']:.3f})" if 'model_sd_logR_mean' in cp else ''))
    print('\ncontrasts (arm1 - arm0) at 0.05 / 0.01 / 0.001; efficiency geomean var ratio arm1/arm0:')
    for c in ('allelic', 'total'):
        for name, r in res['contrasts'][c].items():
            f = lambda al: f"{r[al]['diff']:+.4f} [{r[al]['lo']:+.4f},{r[al]['hi']:+.4f}]"
            e = r['efficiency']
            print(f"  {c:7s} {name:30s} {f('0.05')} {f('0.01')} {f('0.001')}  eff {e['geomean']:.3f} "
                  f"[{e['geomean_lo']:.3f},{e['geomean_hi']:.3f}]")
    print('\ncombined:')
    for arm in ARMS:
        r = res['combined'][arm]
        print(f"  {arm:24s} {r['0.05']['rate']:.4f} {r['0.01']['rate']:.4f} {r['0.001']['rate']:.5f}")
    log(f'wrote {OUT} ({time.time()-t_start:.0f} s)')


if __name__ == '__main__':
    main()

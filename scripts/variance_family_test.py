#!/usr/bin/env python3
"""Is residual variance ADDITIVE in the Gibbs variance, or a POWER LAW in it?

QUESTION
    For a donor-gene record with Gibbs variance v (the across-draw variance of
    the phenotype, plus the counting term unless stated), is the residual
    variance
        additive   Var = sigma_g^2 (v + tau)          local slope v/(v+tau)
        power law  Var = sigma_g^2 v^gamma            local slope gamma
    where the local slope is d log Var / d log v. An earlier measurement on the
    pre-correction pipeline gave a within-gene exponent near 0.65 in both
    channels; an additive model with tau near the typical v also averages a
    slope near 0.65, so the average slope cannot decide. The SHAPE of the slope
    across v does: constant for a power law, rising from 0 towards 1 for an
    additive model. The answer picks the weighting family (1/(v+tau) or
    v^-gamma) for the simulator's variance model and the weighting comparators.

INPUTS (the 2026-09-25 pipeline rules, docs/pipeline_rules.md)
    Values from Salmon point estimates, variance from the 200 Gibbs draws
    through the identical transform (tensorqtl.hapmixqtl.
    summaries_from_point_estimates): t = log2(CPM + 1) on edgeR's effective
    library size, a = log2((L + 0.5)/(R + 0.5)). Genes: the 11,747 eQTL
    calibration genes (edger/calibration_genes.txt) present in the Gibbs cache.
    Covariates: cov/log2cpm1_point_calibration_20260925 (14 RNA-tied columns and
    3 genotype PCs), plus an intercept, 18 columns in all.

TOTAL CHANNEL
    t is regressed on the 18 columns by UNWEIGHTED ordinary least squares, the
    same design for every gene and all 92 donors, so one residual projector
    M = I - X (X'X)^-1 X' serves every gene. The OLS residual's expectation is
    NOT the record's own variance: E[e_i^2] = sum_j M_ij^2 s_j^2
    = (1 - h_i)^2 s_i^2 + (terms from every other donor). With p = 18 and
    n = 92 the second part is about 0.16 x the gene's mean variance and is the
    same for every donor, i.e. it IS an additive floor, manufactured by the
    residualization. Two remedies, both used:
      * binned summaries use s2hat = (M o M)^-1 e^2 (o the element-wise
        product): solving the linear system E[e^2] = (M o M) s^2 gives
        per-record estimates whose expectation is exactly s_i^2 under any
        diagonal error covariance (the unbiased heteroskedasticity estimator of
        Cattaneo, Jansson and Newey). Individual values can be negative; only
        sums enter. Donors with leverage above H_MAX (two genotype-PC outliers
        at 0.977 and 0.947) are nearly fitted by the design; the system is
        solved without them and their records are left out of the bins.
      * model fits use REML, restricted maximum likelihood: the likelihood of
        the n - p residual contrasts orthogonal to the design, which removes
        the 18 mean coefficients and charges the variance fit for them. For a
        variance shape f and a per-gene scale sigma_g^2 profiled out,
            -2 l_R = (n-p) log Q + sum log f + log det(X' F^-1 X) + const,
            Q = t' F^-1 t - t' F^-1 X (X' F^-1 X)^-1 X' F^-1 t.
        This is the comparison nlme's varPower against varConstPower makes.
    The raw OLS e^2 is also binned, to show the size of the floor it carries.

ALLELIC CHANNEL
    Through the origin with no covariates (the shipped design), so the residual
    is a itself and there is no mixing: plain maximum likelihood of a with
    variance sigma_g^2 f. Records: min(pL, pR) >= 0.5 in the point estimate
    (a zero-haplotype pair, one haplotype below 0.5 reads, is excluded); genes:
    at least MIN_NA such records.

PSEUDO-LIKELIHOOD
    Both likelihoods are Gaussian and the errors are not (the allelic channel
    is heavy-tailed). The Gaussian likelihood is still an unbiased estimating
    equation for the variance function (Carroll and Ruppert's pseudo-
    likelihood): its score is sum (e^2/mu - 1) d log mu, zero in expectation
    whenever E[e^2] = mu. Its deviance differences are not chi-square, which is
    why every interval here comes from a gene-clustered bootstrap.

FAMILIES (all with a free per-gene scale sigma_g^2, profiled out)
    power      f = v^gamma                                1 pooled parameter
    add_rel    f = v + r med_g(v)                         1 pooled parameter
    add_abs    f = v + tau                                1 pooled parameter
    add_trend  f = v + rho med_ref (med_g/med_ref)^b      2 pooled parameters
               (b = 0 is add_abs, b = 1 is add_rel)
    nest       f = (v + r med_g(v))^gamma                 2 pooled parameters
               (r = 0 is power, gamma = 1 is add_rel)
    Per-gene deviance curves are computed once on parameter grids; pooled fits
    sum them, and the bootstrap re-sums them with multinomial gene counts and
    re-minimises, so each bootstrap draw refits the pooled parameters.

BINNED SUMMARIES
    Records are binned by within-gene decile of v and, separately, by global
    decile of absolute v. Bin effects come from RAKING (iterative proportional
    fitting): alternating closed-form updates of gene effects alpha_g and bin
    effects beta_b in E[y_gi] = exp(alpha_g + beta_b) until the fitted gene and
    bin sums equal the observed ones. That is the quasi-Poisson log-link fit
    with gene fixed effects, so beta_b is the bin's E[y] with each gene's scale
    removed. The local slope between adjacent bins is the change in beta over
    the change in bin-mean log v. The same raking applied to each fitted
    model's f gives that model's predicted bin effects and local slopes, which
    handles within-bin spread of v exactly. Bin-level generalized least squares
    on the nine bin contrasts, with the bootstrap covariance, is reported too.

OBSERVED v AND FITTED-VALUE v
    A record's v is computed from its own point estimate, so it is coupled to
    the record's own residual: in the total channel v ~ 1/count, so an extreme
    value gives an extreme v and a large residual together; in the allelic
    channel an imbalanced split gives both a large a^2 and a large v. Every
    analysis is therefore run twice: on the observed v (the relationship as
    asked) and on the FITTED-VALUE v, the same record's variance recomputed at
    its fitted value with the record's ratio of Gibbs-plus-counting variance to
    the counting form held fixed (total: the OLS fitted log2 CPM; allelic: the
    balanced split of the same reads, which is the fitted value of the
    through-origin null design). The fitted value still carries h_i e_i, so a
    leverage-sized share of the coupling remains in the total channel.

WHAT THE TWO CRITERIA ESTIMATE
    The Gaussian pseudo-likelihood on squared residuals has the score of a
    Gamma GLM, sum (y/mu - 1) dlog mu, and follows the typical squared
    residual; raking has the quasi-Poisson score, sum (y - mu) dlog mu, and
    follows the arithmetic mean E[e^2 | v]. They agree only when the variance
    function is correctly specified. Inverse-variance weights need E[e^2 | v],
    so where the two disagree the bin-level numbers are the weighting-relevant
    ones.

VALIDATION
    Exogenous v: Gaussian data simulated through the real design at the
    fitted-value v under each fitted model, then passed through the identical
    pipeline: shows the raw e^2 floor and its removal by the deconvolution,
    and gives the deviance difference each family produces when it is true
    (one replicate), the reference against which the observed difference is
    read.
    Coupled null: errors from a known variance function of the fitted-value v
    (homoscedastic, and the fitted additive model), after which the count and
    so v are recomputed from the SIMULATED value exactly as the pipeline
    computes them from the observed one (total); binomial allele counts at
    each record's observed allele-resolved total, balanced or with a Gaussian
    per-gene biological imbalance, with a and v recomputed from the simulated
    counts (allelic). Binned against the recomputed v, these show what the
    coupling alone produces.

OUTPUTS (OUT)
    summary.json (every number), fits.tsv (pooled fits by channel, variant and
    stratum), bins.tsv (bin effects and local slopes, observed and predicted),
    bin_gls.tsv, per_gene.tsv, local_slopes_total.png, local_slopes_allelic.png,
    simulation_local_slopes.png, staged/ (the arrays every fit reads).

Usage:
    python3 scripts/variance_family_test.py prepare   # stage arrays (~2 min)
    python3 scripts/variance_family_test.py fit       # fits, bins, figures
    python3 scripts/variance_family_test.py all
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
D = '/mnt/ssd/lalli/brainvar_hapmix_deploy'
CACHE = f'{D}/cache/gibbs_56b63c3b37ed5df8'
PE = f'{CACHE}/point_estimates'
COV = f'{D}/cov/log2cpm1_point_calibration_20260925'
OUT = os.environ.get('VFT_OUT', f'{D}/variance_family_test_20260926')
STAGE = f'{D}/variance_family_test_20260926/staged'
SMOKE = int(os.environ.get('VFT_SMOKE', '0'))     # >0: evenly spaced gene subsample, for testing

SEED = 42
N_BOOT = 1000
N_BINS = 10
MIN_NA = 20
EPS = 1e-12
LN2 = np.log(2.0)
CHUNK = 1000
H_MAX = 0.9          # leverage above which a donor is left out of the deconvolution

# parameter grids
GAMMA_GRID = np.round(np.arange(-0.5, 2.5001, 0.01), 4)           # 301
LOGR_GRID = np.round(np.arange(-4.0, 6.0001, 0.02), 4)            # 501, log10 r
B_GRID = np.round(np.arange(-1.0, 3.0001, 0.1), 4)                # 41
TREND_LOGRHO = LOGR_GRID[::2]                                     # 251
NEST_GAMMA = np.round(np.arange(0.0, 2.5001, 0.05), 4)            # 51
NEST_LOGR = np.round(np.arange(-3.0, 3.0001, 0.2), 4)             # 31 (+ r = 0)
NPROC = 24

sys.path.insert(0, f'{REPO}/scripts')
sys.path.insert(0, REPO)
sys.path.append(f'{REPO}/tensorqtl')

log = lambda *a: print(time.strftime('%H:%M:%S'), *a, flush=True)


# ---------------------------------------------------------------------------
#  staging: every derived array written once, before any fit
# ---------------------------------------------------------------------------

def prepare():
    import tensorqtl.hapmixqtl as HM
    import run_hapmixqtl_from_salmon as H
    os.makedirs(STAGE, exist_ok=True)
    genes_all = open(f'{CACHE}/genes.txt').read().split()
    samples = open(f'{CACHE}/samples.txt').read().split()
    cal = set(open(f'{PE}/edger/calibration_genes.txt').read().split())
    rows = np.array([i for i, g in enumerate(genes_all) if g in cal])
    genes = [genes_all[i] for i in rows]
    log(f'{len(genes)} calibration genes in the Gibbs cache, {len(samples)} donors')
    eff_lib, _ = H.read_edger_dir(f'{PE}/edger', samples)

    cov = pd.read_csv(f'{COV}/covariates.tsv', sep='\t', index_col=0)
    cov.index = cov.index.astype(str)
    cov = cov.loc[samples]                          # bound by sample key
    gcols = open(f'{COV}/genotype_covariates.txt').read().split()
    Z = cov.values.astype(float)
    Z = (Z - Z.mean(0)) / Z.std(0)
    X = np.column_stack([np.ones(len(samples)), Z])
    sv = np.linalg.svd(X, compute_uv=False)
    if sv.min() / sv.max() < 1e-8:
        raise SystemExit('covariate design is rank-deficient')
    Q, _ = np.linalg.qr(X)
    M = np.eye(len(samples)) - Q @ Q.T
    h = 1.0 - np.diag(M)
    MM = M * M
    # Donors the design nearly fits by themselves (leverage near 1; here two
    # genotype-PC outliers) have residuals that carry almost no information on
    # their own variance, and inverting (M o M) through them puts entries of
    # several hundred into every other donor's estimate. The deconvolution is
    # therefore solved on the reduced system without them, and their records
    # are left out of the binned summaries (REML keeps all 92 donors, since it
    # handles leverage exactly). The neglected terms are M_ij^2 s_j^2 for the
    # excluded j, bounded by the column sums reported below.
    keep_d = np.where(h < H_MAX)[0]
    drop_d = np.where(h >= H_MAX)[0]
    MMk = MM[np.ix_(keep_d, keep_d)]
    W = np.linalg.inv(MMk)
    ident_err = float(np.abs(MMk @ W - np.eye(len(keep_d))).max())
    neglect = MM[np.ix_(keep_d, drop_d)].sum(1) / np.diag(MMk)
    log(f'design {X.shape}, leverage mean {h.mean():.4f} range [{h.min():.4f}, {h.max():.4f}], '
        f'full (MoM) cond {np.linalg.cond(MM):.1f}; excluded from deconvolution '
        f'{[(samples[j], round(h[j], 3)) for j in drop_d]}; reduced cond {np.linalg.cond(MMk):.1f}, '
        f'|MoM W - I| {ident_err:.1e}, max neglected weight / own weight {neglect.max():.2e}')

    pLm, pRm, pTm = (np.load(f'{PE}/{k}.npy', mmap_mode='r') for k in ('pL', 'pR', 'pT'))
    Ym = {k: np.load(f'{CACHE}/{k}.npy', mmap_mode='r') for k in ('YL', 'YR', 'YT')}
    G, N = len(genes), len(samples)
    out = {k: np.empty((G, N)) for k in ('A', 'T', 'Va', 'Vt', 'Va_gibbs', 'Vt_gibbs',
                                         'pL', 'pR', 'pT')}
    t0 = time.time()
    for s in range(0, G, CHUNK):
        r = rows[s:s + CHUNK]
        pL, pR, pT = (np.asarray(m[r]) for m in (pLm, pRm, pTm))
        yL, yR, yT = (np.asarray(Ym[k][r]) for k in ('YL', 'YR', 'YT'))
        A, T, Va, Vt, _ = HM.summaries_from_point_estimates(pL, pR, pT, eff_lib, yL, yR, yT,
                                                            count_noise=True)
        _, _, Va0, Vt0, _ = HM.summaries_from_point_estimates(pL, pR, pT, eff_lib, yL, yR, yT,
                                                              count_noise=False)
        sl = slice(s, s + len(r))
        for k, v in (('A', A), ('T', T), ('Va', Va), ('Vt', Vt), ('Va_gibbs', Va0),
                     ('Vt_gibbs', Vt0), ('pL', pL), ('pR', pR), ('pT', pT)):
            out[k][sl] = v
        if s == 0:
            log(f'first chunk of {len(r)} genes in {time.time() - t0:.1f} s')
    log(f'summaries for {G} genes in {time.time() - t0:.1f} s')

    # counting terms recomputed from the documented formula, as a pin on the
    # difference between the two calls
    k = 1e6 / eff_lib
    q_t = k[None] ** 2 * (out['pT'] + 0.5) / ((k[None] * (out['pT'] + 0.5) + 1) ** 2 * np.log(2) ** 2)
    q_a = (1 / (out['pL'] + 0.5) + 1 / (out['pR'] + 0.5)) / np.log(2) ** 2
    inf_any = (out['pL'] + out['pR']) > 0
    pin_t = float(np.abs(out['Vt'] - out['Vt_gibbs'] - q_t).max())
    pin_a = float(np.abs((out['Va'] - out['Va_gibbs'] - q_a)[inf_any]).max())
    log(f'counting-term pins: total {pin_t:.1e}, allelic {pin_a:.1e}')

    E = out['T'] @ M                     # OLS residuals (M symmetric), [G, N]
    S2 = deconvolve(E, W, keep_d)        # deconvolved per-record variance
    for kk, v in out.items():
        np.save(f'{STAGE}/{kk}.npy', v)
    np.save(f'{STAGE}/E.npy', E)
    np.save(f'{STAGE}/S2hat.npy', S2)
    np.save(f'{STAGE}/X.npy', X)
    np.save(f'{STAGE}/M.npy', M)
    np.save(f'{STAGE}/W.npy', W)
    np.save(f'{STAGE}/keep_d.npy', keep_d)
    np.save(f'{STAGE}/leverage.npy', h)
    with open(f'{STAGE}/genes.txt', 'w') as fh:
        fh.write('\n'.join(genes) + '\n')
    with open(f'{STAGE}/samples.txt', 'w') as fh:
        fh.write('\n'.join(samples) + '\n')
    meta = dict(n_genes=G, n_donors=N, design_columns=['intercept'] + list(cov.columns),
                genotype_tied=gcols, leverage_mean=float(h.mean()),
                leverage_min=float(h.min()), leverage_max=float(h.max()),
                MoM_condition_full=float(np.linalg.cond(MM)),
                deconvolution_excluded_donors=[samples[j] for j in drop_d],
                deconvolution_excluded_leverage=[float(h[j]) for j in drop_d],
                leverage_threshold=H_MAX,
                MoM_condition_reduced=float(np.linalg.cond(MMk)), MoM_inverse_error=ident_err,
                neglected_weight_over_own_max=float(neglect.max()),
                neglected_weight_over_own_median=float(np.median(neglect)),
                next_highest_leverage=float(np.sort(h[keep_d])[-1]),
                counting_term_pin_total=pin_t, counting_term_pin_allelic=pin_a,
                eff_lib_median=float(np.median(eff_lib)),
                gene_sum_s2hat_nonpositive=int((np.nansum(S2, 1) <= 0).sum()),
                s2hat_negative_fraction=float((S2[:, keep_d] < 0).mean()),
                W_column_sum_min=float(W.sum(0).min()),
                W_offdiag_abs_max=float(np.abs(W - np.diag(np.diag(W))).max()))
    json.dump(meta, open(f'{STAGE}/meta.json', 'w'), indent=1)
    log('staged', meta)


def deconvolve(E, W, keep_d):
    """(M o M)^-1 e^2 on the reduced donor set; NaN for excluded donors."""
    S2 = np.full(E.shape, np.nan)
    S2[:, keep_d] = (E[:, keep_d] ** 2) @ W.T
    return S2


def load_staged():
    S = {k: np.load(f'{STAGE}/{k}.npy') for k in
         ('A', 'T', 'Va', 'Vt', 'Va_gibbs', 'Vt_gibbs', 'pL', 'pR', 'pT', 'E', 'S2hat',
          'X', 'M', 'W', 'keep_d', 'leverage')}
    S['genes'] = open(f'{STAGE}/genes.txt').read().split()
    S['meta'] = json.load(open(f'{STAGE}/meta.json'))
    if SMOKE:
        idx = np.linspace(0, len(S['genes']) - 1, SMOKE).astype(int)
        for k in ('A', 'T', 'Va', 'Vt', 'Va_gibbs', 'Vt_gibbs', 'pL', 'pR', 'pT', 'E', 'S2hat'):
            S[k] = S[k][idx]
        S['genes'] = [S['genes'][i] for i in idx]
    return S


# ---------------------------------------------------------------------------
#  deviance engines: profiled -2 log-likelihood per gene for a log-shape logf
# ---------------------------------------------------------------------------

class TotalREML:
    """Profiled REML deviance per gene, fixed design X for every gene."""

    def __init__(self, t, X):
        self.t = t
        self.X = X
        self.n, self.p = X.shape
        self.XX = np.einsum('np,nq->npq', X, X).reshape(self.n, -1)   # [n, p*p]

    def dev(self, logf):
        logf = logf - logf.mean(1, keepdims=True)       # scale-free; conditioning
        w = np.exp(-logf)
        A = (w @ self.XX).reshape(-1, self.p, self.p)
        wt = w * self.t
        b = wt @ self.X
        c = (wt * self.t).sum(1)
        L = np.linalg.cholesky(A)
        z = np.linalg.solve(A, b[..., None])[..., 0]
        Qv = c - (b * z).sum(1)
        logdet = 2 * np.log(np.diagonal(L, axis1=1, axis2=2)).sum(1)
        return (self.n - self.p) * np.log(Qv) + logf.sum(1) + logdet

    def sigma2(self, logf):
        """Profiled REML scale at shape exp(logf) (not mean-centred)."""
        w = np.exp(-logf)
        A = (w @ self.XX).reshape(-1, self.p, self.p)
        wt = w * self.t
        b = wt @ self.X
        c = (wt * self.t).sum(1)
        z = np.linalg.solve(A, b[..., None])[..., 0]
        return (c - (b * z).sum(1)) / (self.n - self.p)


class AllelicML:
    """Profiled ML deviance per gene; no mean parameters; ragged via mask."""

    def __init__(self, a2, mask):
        self.a2 = np.where(mask, a2, 0.0)
        self.mask = mask
        self.nk = mask.sum(1)

    def dev(self, logf):
        lf = np.where(self.mask, logf, 0.0)
        mu = (lf.sum(1) / self.nk)
        lf = np.where(self.mask, lf - mu[:, None], 0.0)
        s = (self.a2 * np.exp(-lf)).sum(1)
        return self.nk * np.log(s / self.nk) + lf.sum(1)

    def sigma2(self, logf):
        lf = np.where(self.mask, logf, 0.0)
        return (self.a2 * np.exp(-lf)).sum(1) / self.nk


def safe_logv(v, mask):
    return np.log(np.where(mask, np.maximum(v, 1e-300), 1.0))


_POOL_STATE = {}


def _worker_init():
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)


def _eval_spec(spec):
    """One grid point for every gene (runs in a forked worker)."""
    fam, a, b = spec
    st = _POOL_STATE
    if fam == 'power':
        lf = a * st['logv']
    elif fam == 'add_rel':
        lf = np.log(st['vrel'] + 10.0 ** a)
    elif fam == 'add_abs':
        lf = np.log(st['vabs'] + a)
    elif fam == 'nest':
        lf = a * (st['lvr'] if b is None else np.log(st['vrel'] + 10.0 ** b))
    else:
        raise ValueError(fam)
    return st['eng'].dev(lf)


def family_curves(engine, v, mask, with_nest=False, tau_grid=None):
    """Per-gene deviance curves [G, K] for every family grid, evaluated in a
    fork pool of NPROC single-threaded workers."""
    import multiprocessing as mp
    logv = safe_logv(v, mask)
    medg = np.array([np.median(v[g][mask[g]]) for g in range(v.shape[0])])
    vrel = np.where(mask, v / medg[:, None], 1.0)
    _POOL_STATE.clear()
    _POOL_STATE.update(eng=engine, logv=logv, vrel=vrel, lvr=np.log(vrel),
                       vabs=np.where(mask, v, 1.0))
    specs = {'power': [('power', g, None) for g in GAMMA_GRID],
             'add_rel': [('add_rel', lr, None) for lr in LOGR_GRID]}
    if tau_grid is not None:
        specs['add_abs'] = [('add_abs', t, None) for t in tau_grid]
    if with_nest:
        specs['nest'] = [('nest', gm, lr) for gm in NEST_GAMMA
                         for lr in [None] + list(NEST_LOGR)]
    flat = [(k, sp) for k, lst in specs.items() for sp in lst]
    t0 = time.time()
    with mp.get_context('fork').Pool(NPROC, initializer=_worker_init) as pool:
        cols = pool.map(_eval_spec, [sp for _, sp in flat], chunksize=4)
    out = {}
    i = 0
    for k, lst in specs.items():
        out[k] = np.column_stack(cols[i:i + len(lst)])
        i += len(lst)
    _POOL_STATE.clear()
    log(f'    {len(flat)} grid evaluations {sorted(out)} in {time.time() - t0:.1f} s')
    return out, medg


def interp_curve(Dg, logr_g):
    """Linear interpolation of per-gene add_rel curves at per-gene log10 r
    [G, K'] (clipped to the grid ends, which are the family's limits)."""
    step = LOGR_GRID[1] - LOGR_GRID[0]
    pos = np.clip((logr_g - LOGR_GRID[0]) / step, 0, len(LOGR_GRID) - 1)
    i0 = np.minimum(np.floor(pos).astype(int), len(LOGR_GRID) - 2)
    fr = pos - i0
    rows = np.arange(Dg.shape[0])[:, None]
    return Dg[rows, i0] * (1 - fr) + Dg[rows, i0 + 1] * fr


def trend_curves(Dadd, medg, med_ref):
    """add_trend per-gene deviance [G, n_b, n_rho]: r_g = rho (med_g/med_ref)^(b-1),
    rho on TREND_LOGRHO."""
    off = np.log10(medg / med_ref)
    out = np.empty((Dadd.shape[0], len(B_GRID), len(TREND_LOGRHO)))
    for j, b in enumerate(B_GRID):
        out[:, j, :] = interp_curve(Dadd, TREND_LOGRHO[None, :] + (b - 1) * off[:, None])
    return out


# ---------------------------------------------------------------------------
#  pooled fits and gene-clustered bootstrap
# ---------------------------------------------------------------------------

def boot_counts(n_genes, n_boot, rng):
    return rng.multinomial(n_genes, np.full(n_genes, 1.0 / n_genes), size=n_boot).astype(float)


def pooled(Dg, C):
    """Point fit and bootstrap refits of one family: returns (argmin, min) for
    the data and for each bootstrap draw."""
    tot = Dg.sum(0)
    k = int(np.argmin(tot))
    Bt = C @ Dg
    kb = Bt.argmin(1)
    return k, float(tot[k]), kb, Bt[np.arange(len(kb)), kb]


def fit_families(curves, medg, med_ref, tau_grid, C, genes_idx=None):
    """All pooled fits on a gene subset, with bootstrap refits."""
    sel = slice(None) if genes_idx is None else genes_idx
    res = {}
    k, d, kb, db = pooled(curves['power'][sel], C)
    res['power'] = dict(param=dict(gamma=GAMMA_GRID[k]), dev=d, boot_dev=db,
                        boot_param=dict(gamma=GAMMA_GRID[kb]), k=k)
    k, d, kb, db = pooled(curves['add_rel'][sel], C)
    res['add_rel'] = dict(param=dict(log10_r=LOGR_GRID[k]), dev=d, boot_dev=db,
                          boot_param=dict(log10_r=LOGR_GRID[kb]), k=k)
    if 'add_abs' in curves:
        k, d, kb, db = pooled(curves['add_abs'][sel], C)
        res['add_abs'] = dict(param=dict(log10_tau=np.log10(tau_grid[k])), dev=d, boot_dev=db,
                              boot_param=dict(log10_tau=np.log10(tau_grid[kb])), k=k)
    # add_trend: loop over b, keep the best rho per b
    tr = trend_curves(curves['add_rel'][sel], medg[sel], med_ref)
    best = (np.inf, None, None)
    bd = np.full(C.shape[0], np.inf)
    bb = np.zeros(C.shape[0])
    br = np.zeros(C.shape[0])
    abs_interp = None
    for j, b in enumerate(B_GRID):
        Dj = tr[:, j, :]
        tot = Dj.sum(0)
        kk = int(np.argmin(tot))
        if tot[kk] < best[0]:
            best = (float(tot[kk]), b, TREND_LOGRHO[kk])
        if abs(b) < 1e-9:
            abs_interp = dict(dev=float(tot[kk]), log10_tau=TREND_LOGRHO[kk] + np.log10(med_ref))
        Bt = C @ Dj
        kb = Bt.argmin(1)
        m = Bt[np.arange(len(kb)), kb]
        upd = m < bd
        bd[upd], bb[upd], br[upd] = m[upd], b, TREND_LOGRHO[kb][upd]
    res['add_trend'] = dict(param=dict(b=best[1], log10_rho=best[2]), dev=best[0], boot_dev=bd,
                            boot_param=dict(b=bb, log10_rho=br))
    res['add_abs_via_interp'] = abs_interp
    if 'nest' in curves:
        k, d, kb, db = pooled(curves['nest'][sel], C)
        ng, nr = np.divmod(np.arange(curves['nest'].shape[1]), len(NEST_LOGR) + 1)
        lr = np.concatenate([[-np.inf], NEST_LOGR])
        res['nest'] = dict(param=dict(gamma=NEST_GAMMA[ng[k]], log10_r=lr[nr[k]]), dev=d,
                           boot_dev=db, boot_param=dict(gamma=NEST_GAMMA[ng[kb]],
                                                        log10_r=lr[nr[kb]]))
    return res


def ci(x):
    x = np.asarray(x, float)
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def summarize_fits(res, n_records, n_genes):
    """Deviance of each family relative to the power law, with bootstrap CIs."""
    base, base_b = res['power']['dev'], res['power']['boot_dev']
    rows = {}
    for fam in ('power', 'add_rel', 'add_abs', 'add_trend', 'nest'):
        if fam not in res:
            continue
        r = res[fam]
        dd = r['dev'] - base
        ddb = r['boot_dev'] - base_b
        rows[fam] = dict(
            params={k: float(v) for k, v in r['param'].items()},
            params_ci={k: ci(v[np.isfinite(v)]) if np.isfinite(v).any() else None
                       for k, v in r['boot_param'].items()},
            dev_minus_power=float(dd), dev_minus_power_ci=ci(ddb),
            dev_minus_power_per_1000_records=float(dd / n_records * 1000),
            boot_frac_power_better=float((ddb > 0).mean()))
    # the best additive variant with ONE pooled parameter, like the power law
    one = [f for f in ('add_rel', 'add_abs') if f in res]
    bestadd = min(one, key=lambda f: res[f]['dev'])
    # per draw, the better of the two one-parameter additive variants
    ob = np.min(np.vstack([res[f]['boot_dev'] for f in one]), 0) - base_b
    rows['best_one_param_additive'] = dict(
        family=bestadd, dev_minus_power=float(res[bestadd]['dev'] - base),
        dev_minus_power_ci=ci(ob), boot_frac_power_better=float((ob > 0).mean()))
    rows['at_grid_edge'] = dict(
        power_gamma=bool(res['power']['param']['gamma'] in (GAMMA_GRID[0], GAMMA_GRID[-1])),
        trend_b=bool(res['add_trend']['param']['b'] in (B_GRID[0], B_GRID[-1])),
        nest_gamma=bool('nest' in res and res['nest']['param']['gamma'] in
                        (NEST_GAMMA[0], NEST_GAMMA[-1])))
    rows['n_records'] = int(n_records)
    rows['n_genes'] = int(n_genes)
    if res.get('add_abs_via_interp'):
        rows['add_abs_interp_check'] = dict(
            direct_dev_minus_power=rows['add_abs']['dev_minus_power'] if 'add_abs' in rows else None,
            interp_dev_minus_power=float(res['add_abs_via_interp']['dev'] - base),
            direct_log10_tau=rows['add_abs']['params']['log10_tau'] if 'add_abs' in rows else None,
            interp_log10_tau=float(res['add_abs_via_interp']['log10_tau']))
    return rows


# ---------------------------------------------------------------------------
#  binned summaries by raking
# ---------------------------------------------------------------------------

def bin_index(v, mask, medg, edges=None):
    """(within-gene decile bins, absolute-v decile bins, global edges)."""
    G, N = v.shape
    within = np.full((G, N), -1)
    for g in range(G):
        idx = np.where(mask[g])[0]
        rk = np.argsort(np.argsort(v[g, idx], kind='stable'), kind='stable')
        within[g, idx] = np.minimum(((rk + 0.5) / len(idx) * N_BINS).astype(int), N_BINS - 1)
    lv = np.log(v[mask])
    if edges is None:
        edges = np.quantile(lv, np.linspace(0, 1, N_BINS + 1))
    ab = np.full((G, N), -1)
    ab[mask] = np.clip(np.searchsorted(edges, lv, side='right') - 1, 0, N_BINS - 1)
    return within, ab, edges


def gb_sums(val, bins, mask):
    G = val.shape[0]
    flat = (np.arange(G)[:, None] * N_BINS + bins)[mask]
    return np.bincount(flat, weights=val[mask], minlength=G * N_BINS).reshape(G, N_BINS)


def rake(Y, N, C=None, iters=2000, tol=1e-11):
    """Bin effects beta [D, B] (mean zero over bins) in E[y] = exp(alpha_g + beta_b).
    C: [D, G] gene multiplicities (None = the data once)."""
    G, B = Y.shape
    if C is None:
        C = np.ones((1, G))
    Yg = Y.sum(1)
    keep = Yg > 0
    Y, N, C, Yg = Y[keep], N[keep], C[:, keep], Yg[keep]
    num = C @ Y
    beta = np.zeros((C.shape[0], B))
    for it in range(iters):
        alpha = np.log(Yg[None, :] / (np.exp(beta) @ N.T))
        den = (C * np.exp(alpha)) @ N
        nb = np.log(num / den)
        nb -= nb.mean(1, keepdims=True)
        if np.abs(nb - beta).max() < tol:
            beta = nb
            break
        beta = nb
    return beta, it


def bin_xbar(xsum, N, C=None):
    if C is None:
        return xsum.sum(0) / N.sum(0)
    return (C @ xsum) / (C @ N)


def local_slopes(beta, xbar):
    return np.diff(beta, axis=-1) / np.diff(xbar, axis=-1)


def gls_bins(beta_obs, beta_boot, beta_pred_grid):
    """Min over the grid of the GLS chi-square of the 9 contrasts to bin 1."""
    d_obs = beta_obs[1:] - beta_obs[0]
    db = beta_boot[:, 1:] - beta_boot[:, :1]
    S = np.cov(db, rowvar=False)
    Si = np.linalg.inv(S)
    dp = beta_pred_grid[:, 1:] - beta_pred_grid[:, :1]
    r = d_obs[None, :] - dp
    chi = np.einsum('kb,bc,kc->k', r, Si, r)
    k = int(np.argmin(chi))
    return k, float(chi[k])


# ---------------------------------------------------------------------------
#  one channel / variant end to end
# ---------------------------------------------------------------------------

def shape_logf(fam, param, v, mask, medg):
    """log f for a fitted family at its pooled parameter(s)."""
    lv = safe_logv(v, mask)
    vv = np.where(mask, v, 1.0)
    if fam == 'power':
        return param['gamma'] * lv
    if fam == 'add_rel':
        return np.log(vv + 10 ** param['log10_r'] * medg[:, None])
    if fam == 'add_abs':
        return np.log(vv + 10 ** param['log10_tau'])
    if fam == 'add_trend':
        return np.log(vv + 10 ** param['log10_rho'] * medg[:, None] *
                      (medg[:, None] / param['med_ref']) ** (param['b'] - 1))
    if fam == 'nest':
        r = 0.0 if not np.isfinite(param['log10_r']) else 10 ** param['log10_r']
        return param['gamma'] * np.log(vv / medg[:, None] + r)
    raise ValueError(fam)


def analyse(name, engine, v, mask, y_bins, terc, rng, extra_y=None, with_nest=False,
            MM=None, do_boot_bins=True, strata=True, bin_mask=None):
    """Curves, pooled fits (+ by tercile), binned summaries. Returns a dict.
    ``mask``: records in the likelihood; ``bin_mask``: records in the binned
    summaries and the heavy-tail check (default ``mask``)."""
    bm = mask if bin_mask is None else (bin_mask & mask)
    log(f'[{name}] {int(mask.any(1).sum())} genes, {int(mask.sum())} records '
        f'({int(bm.sum())} binned)')
    G = v.shape[0]
    n_rec = int(mask.sum())
    medg_tmp = np.array([np.median(v[g][mask[g]]) for g in range(G)])
    med_ref = float(np.median(medg_tmp))
    lvall = np.log(v[mask])
    tau_grid = 10.0 ** np.arange(np.floor(lvall.min() / np.log(10)) - 1,
                                 np.ceil(lvall.max() / np.log(10)) + 1.0001, 0.05)
    curves, medg = family_curves(engine, v, mask, with_nest=with_nest, tau_grid=tau_grid)
    C = boot_counts(G, N_BOOT, rng)
    res = fit_families(curves, medg, med_ref, tau_grid, C)
    summ = dict(overall=summarize_fits(res, n_rec, G))
    summ['med_ref'] = med_ref
    summ['tau_grid_log10'] = [float(np.log10(tau_grid[0])), float(np.log10(tau_grid[-1])),
                              len(tau_grid)]
    # grid-edge diagnostics for the per-gene free optima (the ends of the r
    # grid are the family's limits: v alone and a constant)
    kg = curves['add_rel'].argmin(1)
    pg = curves['power'].argmin(1)
    edge_check = dict(
        add_rel_pergene_at_low_end=float((kg == 0).mean()),
        add_rel_pergene_at_high_end=float((kg == len(LOGR_GRID) - 1).mean()),
        power_pergene_at_low_end=float((pg == 0).mean()),
        power_pergene_at_high_end=float((pg == len(GAMMA_GRID) - 1).mean()),
        max_abs_dev_r_low_vs_gamma1=float(np.abs(curves['add_rel'][:, 0] -
                                                 curves['power'][:, np.argmin(np.abs(GAMMA_GRID - 1))]).max()),
        max_abs_dev_r_high_vs_gamma0=float(np.abs(curves['add_rel'][:, -1] -
                                                  curves['power'][:, np.argmin(np.abs(GAMMA_GRID))]).max()))
    summ['grid_edges'] = edge_check
    summ['pergene'] = dict(
        gamma_median=float(np.median(GAMMA_GRID[pg])),
        gamma_iqr=[float(np.percentile(GAMMA_GRID[pg], 25)), float(np.percentile(GAMMA_GRID[pg], 75))],
        log10_r_median=float(np.median(LOGR_GRID[kg])))
    # per-gene preference at the pooled parameters
    kp, ka = res['power']['k'], res['add_rel']['k']
    dgap = curves['add_rel'][:, ka] - curves['power'][:, kp]
    summ['pergene']['frac_genes_power_better_than_add_rel'] = float((dgap > 0).mean())
    top = np.argsort(-np.abs(dgap))[:max(1, G // 100)]
    summ['pergene']['share_of_gap_in_top1pct_genes'] = float(dgap[top].sum() / dgap.sum())
    # mean local slope implied by the fitted additive shapes
    for fam in ('add_rel', 'add_abs'):
        if fam not in res:
            continue
        p = dict(res[fam]['param'])
        tau = (10 ** p['log10_r'] * medg[:, None]) if fam == 'add_rel' else 10 ** p['log10_tau']
        sl = np.where(mask, v / (np.where(mask, v, 1.0) + tau), np.nan)
        summ[f'{fam}_implied_local_slope'] = dict(
            mean=float(np.nanmean(sl)), median=float(np.nanmedian(sl)),
            q10=float(np.nanpercentile(sl, 10)), q90=float(np.nanpercentile(sl, 90)))

    # by expression tercile, each with its own bootstrap stream
    if strata:
        summ['by_tercile'] = {}
        for tt in range(3):
            gi = np.where((terc == tt) & mask.any(1))[0]
            Ct = boot_counts(len(gi), N_BOOT, rng)
            sub = {k: v_[gi] for k, v_ in curves.items()}
            rt = fit_families(sub, medg[gi], med_ref, tau_grid, Ct)
            summ['by_tercile'][['low', 'mid', 'high'][tt]] = summarize_fits(
                rt, int(mask[gi].sum()), len(gi))

    # heavy-tail sensitivity: drop genes holding a top-0.1% standardized record
    lf = shape_logf('power', res['power']['param'], v, mask, medg)
    s2 = engine.sigma2(lf)
    ysd = np.where(bm, y_bins / (s2[:, None] * np.exp(lf)), np.nan)
    thr = np.nanpercentile(ysd, 99.9)
    heavy = np.nan_to_num(np.nanmax(ysd, 1), nan=0.0) > thr
    keep = np.where(~heavy & mask.any(1))[0]
    Ck = boot_counts(len(keep), N_BOOT, rng)
    rk = fit_families({k: v_[keep] for k, v_ in curves.items()}, medg[keep], med_ref, tau_grid, Ck)
    summ['drop_heavy_genes'] = dict(threshold_z2=float(thr), n_genes_dropped=int(heavy.sum()),
                                    fits=summarize_fits(rk, int(mask[keep].sum()), len(keep)))

    # ---- binned summaries
    within, absb, edges = bin_index(v, bm, medg)
    bins_out = {}
    fam_pred = {f: dict(res[f]['param']) for f in ('power', 'add_rel', 'add_abs') if f in res}
    tp = dict(res['add_trend']['param'])
    tp['med_ref'] = med_ref
    fam_pred['add_trend'] = tp
    yvars = {'y': y_bins}
    if extra_y is not None:
        yvars.update(extra_y)
    for bname, bins, xv in (('within_gene_decile', within, np.log(np.where(mask, v, 1.0) / medg[:, None])),
                            ('absolute_v_decile', absb, safe_logv(v, mask))):
        Nn = gb_sums(np.ones_like(v), bins, bm)
        xs = gb_sums(xv, bins, bm)
        xbar = bin_xbar(xs, Nn)
        rec = dict(n_records=Nn.sum(0).astype(int).tolist(), xbar=xbar.tolist(),
                   mean_log10_v=(bin_xbar(gb_sums(safe_logv(v, mask), bins, bm), Nn) /
                                 np.log(10)).tolist())
        for yk, yv in yvars.items():
            Y = gb_sums(yv, bins, bm)
            beta, it = rake(Y, Nn)
            beta = beta[0]
            ls = local_slopes(beta, xbar)
            r = dict(beta=beta.tolist(), local_slope=ls.tolist(), rake_iters=int(it),
                     genes_nonpositive_sum=int((Y.sum(1) <= 0).sum()))
            if do_boot_bins:
                betab, _ = rake(Y, Nn, C)
                xb = bin_xbar(xs, Nn, C)
                lsb = local_slopes(betab, xb)
                r['beta_ci'] = [ci(betab[:, j]) for j in range(N_BINS)]
                r['local_slope_ci'] = [ci(lsb[:, j]) for j in range(N_BINS - 1)]
                r['beta_boot'] = betab
            rec[yk] = r
        # model predictions for the same binning: of s^2 (and of raw e^2
        # through the mixing matrix, for the total channel)
        preds = {}
        for fam, p in fam_pred.items():
            f = np.exp(shape_logf(fam, p, v, mask, medg))
            f = np.where(mask, f, 0.0)
            bp, _ = rake(gb_sums(f, bins, bm), Nn)
            preds[fam] = dict(beta=bp[0].tolist(), local_slope=local_slopes(bp[0], xbar).tolist())
            if MM is not None:
                fr = f @ MM.T
                bp2, _ = rake(gb_sums(fr, bins, bm), Nn)
                preds[fam + '_raw_e2'] = dict(beta=bp2[0].tolist(),
                                              local_slope=local_slopes(bp2[0], xbar).tolist())
        rec['pred'] = preds
        # bin-level GLS fit of each family on the observed bin contrasts
        if do_boot_bins:
            gl = {}
            grids = {'power': [dict(gamma=g) for g in GAMMA_GRID[::2]],
                     'add_rel': [dict(log10_r=lr) for lr in LOGR_GRID[::4]],
                     'add_abs': [dict(log10_tau=np.log10(t_)) for t_ in tau_grid[::2]]}
            for fam, plist in grids.items():
                bpg = []
                for p in plist:
                    f = np.where(mask, np.exp(shape_logf(fam, p, v, mask, medg)), 0.0)
                    bpg.append(rake(gb_sums(f, bins, bm), Nn, iters=500, tol=1e-9)[0][0])
                k, chi = gls_bins(np.array(rec['y']['beta']), rec['y']['beta_boot'], np.array(bpg))
                gl[fam] = dict(param={kk: float(vv) for kk, vv in plist[k].items()}, chi2=chi, df=8)
            rec['bin_gls'] = gl
        bins_out[bname] = rec
    summ['bins'] = bins_out
    summ['abs_bin_edges_log10'] = (edges / np.log(10)).tolist()
    return summ, res, curves, medg


def strip_boot(d):
    if isinstance(d, dict):
        return {k: strip_boot(v) for k, v in d.items() if k != 'beta_boot'}
    if isinstance(d, (np.floating, np.integer)):
        return d.item()
    if isinstance(d, np.ndarray):
        return d.tolist()
    return d


# ---------------------------------------------------------------------------
#  fitted-value variance: v with the donor's own value taken out
# ---------------------------------------------------------------------------

def count_form_total(y, k):
    """Delta-method Poisson variance of log2(CPM + 1) at count y + 1/2 (the
    counting term of summaries_from_point_estimates)."""
    y = y + 0.5
    return k ** 2 * y / ((k * y + 1.0) ** 2 * LN2 ** 2)


def count_form_allelic(L, R):
    """Delta-method binomial variance of log2((L+1/2)/(R+1/2))."""
    return (1.0 / (L + 0.5) + 1.0 / (R + 0.5)) / LN2 ** 2


def eff_lib_for(samples):
    es = pd.read_csv(f'{PE}/edger/edger_samples.tsv', sep='\t', dtype={'sample': str})
    return es.set_index('sample').loc[list(samples), 'eff_lib_size'].to_numpy(float)


def fitted_value_v(S, k):
    """(Vt_fit, Va_fit, diagnostics). Each record's variance recomputed at its
    FITTED value instead of its observed one, holding the record's ratio of
    Gibbs-plus-counting variance to the counting form fixed. Total: the OLS
    fitted log2 CPM, back to a count; allelic: the through-origin null design
    predicts a = 0 for every donor, so the fitted split of the same n reads is
    balanced. The scaling assumes the across-draw variance moves with the
    count as a Poisson variance does, which the 2026-09-18 read-to-transcript
    ambiguity check measured (0.98x a Poisson prediction, IQR 0.91-1.05).
    The OLS fitted value still contains h_i e_i, so a leverage-sized share of
    the coupling (mean leverage 0.196) remains in the total channel."""
    t_fit = S['T'] - S['E']
    y_raw = (2.0 ** t_fit - 1.0) / k[None, :]
    y_fit = np.maximum(y_raw, 0.0)
    Vt_fit = S['Vt'] * count_form_total(y_fit, k[None, :]) / count_form_total(S['pT'], k[None, :])
    n = S['pL'] + S['pR']
    Va_fit = S['Va'] * count_form_allelic(n / 2, n / 2) / count_form_allelic(S['pL'], S['pR'])
    inf = np.minimum(S['pL'], S['pR']) >= 0.5
    ga = inf.sum(1) >= MIN_NA
    ca = []
    for g in np.where(ga)[0][::5]:
        m = inf[g]
        ca.append((np.corrcoef(np.log(S['Va'][g, m]), S['A'][g, m] ** 2)[0, 1],
                   np.corrcoef(np.log(Va_fit[g, m]), S['A'][g, m] ** 2)[0, 1]))
    ca = np.array(ca)
    diag = dict(total_fitted_count_clipped_at_zero=int((y_raw < 0).sum()),
                total_records=int(y_raw.size),
                total_within_gene_sd_log_v_observed=float(np.median(np.log(S['Vt']).std(1))),
                total_within_gene_sd_log_v_fitted=float(np.median(np.log(Vt_fit).std(1))),
                total_median_within_gene_corr_logv_e=float(np.median(
                    [np.corrcoef(np.log(S['Vt'][g]), S['E'][g])[0, 1]
                     for g in range(0, S['T'].shape[0], 5)])),
                total_median_within_gene_corr_logvfit_e=float(np.median(
                    [np.corrcoef(np.log(Vt_fit[g]), S['E'][g])[0, 1]
                     for g in range(0, S['T'].shape[0], 5)])),
                total_median_within_gene_corr_logv_e2=float(np.median(
                    [np.corrcoef(np.log(S['Vt'][g]), S['E'][g] ** 2)[0, 1]
                     for g in range(0, S['T'].shape[0], 5)])),
                total_median_within_gene_corr_logvfit_e2=float(np.median(
                    [np.corrcoef(np.log(Vt_fit[g]), S['E'][g] ** 2)[0, 1]
                     for g in range(0, S['T'].shape[0], 5)])),
                allelic_median_within_gene_corr_logv_a2=float(np.median(ca[:, 0])),
                allelic_median_within_gene_corr_logvfit_a2=float(np.median(ca[:, 1])),
                allelic_genes_sampled_for_corr=int(len(ca)))
    return Vt_fit, Va_fit, diag


# ---------------------------------------------------------------------------
#  coupled-null simulation: v recomputed from the simulated value
# ---------------------------------------------------------------------------

def bins_only(v, mask, y_dict, medg=None):
    """Local-slope profiles (no bootstrap) for within-gene and absolute bins."""
    if medg is None:
        medg = np.array([np.median(v[g][mask[g]]) if mask[g].any() else 1.0
                         for g in range(v.shape[0])])
    within, absb, _ = bin_index(v, mask, medg)
    out = {}
    for bname, bins, xv in (('within_gene_decile', within,
                             np.log(np.where(mask, v, 1.0) / medg[:, None])),
                            ('absolute_v_decile', absb, safe_logv(v, mask))):
        Nn = gb_sums(np.ones_like(v), bins, mask)
        xbar = bin_xbar(gb_sums(xv, bins, mask), Nn)
        rec = dict(n_records=Nn.sum(0).astype(int).tolist(), xbar=xbar.tolist(),
                   mean_log10_v=(bin_xbar(gb_sums(safe_logv(v, mask), bins, mask), Nn) /
                                 np.log(10)).tolist())
        for yk, y in y_dict.items():
            beta = rake(gb_sums(y, bins, mask), Nn)[0][0]
            rec[yk] = dict(beta=beta.tolist(), local_slope=local_slopes(beta, xbar).tolist())
        out[bname] = rec
    return out


def coupled_total(S, k, Vt_fit, lf_truth, eng, bmask, rng, label):
    """Errors from a known variance function of the EXOGENOUS v_fit; the
    count, and so v, recomputed from the simulated value as the pipeline
    computes it from the observed one."""
    s2 = eng.sigma2(lf_truth)
    e = np.sqrt(s2[:, None] * np.exp(lf_truth)) * rng.standard_normal(S['T'].shape)
    t_fit = S['T'] - S['E']
    t_sim = t_fit + e
    phi = S['Vt'] / count_form_total(S['pT'], k[None, :])
    y_sim = np.maximum((2.0 ** t_sim - 1.0) / k[None, :], 0.0)
    v_sim = phi * count_form_total(y_sim, k[None, :])
    E = t_sim @ S['M']
    S2 = deconvolve(E, S['W'], S['keep_d'])
    out = dict(label=label,
               against_recomputed_v=bins_only(v_sim, bmask, {'y': S2, 'raw_e2': E ** 2}),
               against_exogenous_v=bins_only(Vt_fit, bmask, {'y': S2}))
    log(f'  coupled total simulation [{label}] within-gene slopes vs recomputed v: '
        f'{np.round(out["against_recomputed_v"]["within_gene_decile"]["y"]["local_slope"], 2)}')
    return out


def coupled_allelic(S, ga, rng, with_bio, label):
    """Binomial allele counts at each record's observed allele-resolved total;
    a and v recomputed from the simulated counts as the pipeline does."""
    pL, pR = S['pL'][ga], S['pR'][ga]
    n = np.rint(pL + pR).astype(np.int64)
    inf = np.minimum(pL, pR) >= 0.5
    phi = np.where(inf, S['Va'][ga] / count_form_allelic(pL, pR), 1.0)
    if with_bio:
        a_obs2 = np.where(inf, S['A'][ga] ** 2, np.nan)
        cnt = np.where(inf, count_form_allelic(n / 2, n / 2), np.nan)
        s_bio2 = np.maximum(np.nanmean(a_obs2, 1) - np.nanmean(cnt, 1), 0.0)
        x = np.sqrt(s_bio2)[:, None] * rng.standard_normal(n.shape)
        p = 2.0 ** x / (1.0 + 2.0 ** x)
    else:
        s_bio2 = None
        p = np.full(n.shape, 0.5)
    L = rng.binomial(n, p)
    R = n - L
    keep = inf & (np.minimum(L, R) >= 1)
    gk = keep.sum(1) >= MIN_NA
    keep &= gk[:, None]
    a = np.log2((L + 0.5) / (R + 0.5))
    v_sim = phi * count_form_allelic(L, R)
    nn = (L + R).astype(float)
    v_bal = phi * count_form_allelic(nn / 2, nn / 2)
    out = dict(label=label, n_genes=int(gk.sum()), n_records=int(keep.sum()),
               median_bio_variance=None if s_bio2 is None else float(np.median(s_bio2)),
               against_recomputed_v=bins_only(v_sim, keep, {'y': a * a}),
               against_balanced_v=bins_only(v_bal, keep, {'y': a * a}))
    log(f'  coupled allelic simulation [{label}] within-gene slopes vs recomputed v: '
        f'{np.round(out["against_recomputed_v"]["within_gene_decile"]["y"]["local_slope"], 2)}')
    return out


def donor_shares(y, v, bmask, medg, samples, top=5):
    """Per-donor share of the lowest and highest within-gene v decile, gene
    scale removed, against the donor's share of that decile's records."""
    within, _, _ = bin_index(v, bmask, medg)
    Nn = gb_sums(np.ones_like(v), within, bmask)
    Y = gb_sums(y, within, bmask)
    beta = rake(Y, Nn)[0][0]
    alpha = np.log(np.maximum(Y.sum(1), 1e-300) / (Nn @ np.exp(beta)))
    yn = np.where(bmask, y / np.exp(alpha)[:, None], 0.0)
    out = {}
    for lab, b in (('lowest_decile', 0), ('highest_decile', N_BINS - 1)):
        inb = (within == b) & bmask
        rec_share = inb.sum(0) / inb.sum()
        ysum = np.where(inb, yn, 0.0)
        y_share = ysum.sum(0) / ysum.sum()
        o = np.argsort(-(y_share - rec_share))[:top]
        out[lab] = [dict(donor=samples[j], record_share=float(rec_share[j]),
                         y_share=float(y_share[j])) for j in o]
    return out


def slopes_without_donors(y, v, bmask, medg, drop):
    bm = bmask.copy()
    bm[:, drop] = False
    return bins_only(v, bm, {'y': y}, medg=medg)


# ---------------------------------------------------------------------------
#  main fitting stage
# ---------------------------------------------------------------------------

def dump(summary):
    json.dump(summary, open(f'{OUT}/summary.json', 'w'), indent=1, default=float)


def fit():
    S = load_staged()
    genes = S['genes']
    samples = open(f'{STAGE}/samples.txt').read().split()
    G, N = S['T'].shape
    k = 1e6 / eff_lib_for(samples)
    streams = ('tot_obs', 'tot_sub', 'tot_sub0', 'tot_fit', 'ase_obs', 'ase_sub', 'ase_sub0',
               'ase_fit', 'sim_tot_pl', 'sim_tot_add', 'sim_ase_pl', 'sim_ase_add',
               'cpl_tot_h', 'cpl_tot_f', 'cpl_ase_b', 'cpl_ase_bb')
    rngs = {kk: np.random.default_rng(sq) for kk, sq in
            zip(streams, np.random.SeedSequence(SEED).spawn(len(streams)))}
    tmean = S['T'].mean(1)
    cuts = np.quantile(tmean, [1 / 3, 2 / 3])
    terc = np.digitize(tmean, cuts)
    MM = S['M'] * S['M']
    Vt_fit, Va_fit, fdiag = fitted_value_v(S, k)
    summary = dict(meta=S['meta'], seed=SEED, n_boot=N_BOOT, n_bins=N_BINS, min_na=MIN_NA,
                   grids=dict(gamma=[float(GAMMA_GRID[0]), float(GAMMA_GRID[-1]), len(GAMMA_GRID)],
                              log10_r=[float(LOGR_GRID[0]), float(LOGR_GRID[-1]), len(LOGR_GRID)],
                              b=[float(B_GRID[0]), float(B_GRID[-1]), len(B_GRID)],
                              trend_log10_rho=[float(TREND_LOGRHO[0]), float(TREND_LOGRHO[-1]),
                                               len(TREND_LOGRHO)],
                              nest_gamma=len(NEST_GAMMA), nest_log10_r=len(NEST_LOGR) + 1),
                   tercile_cuts_mean_log2cpm=cuts.tolist(),
                   tercile_genes=np.bincount(terc, minlength=3).tolist(),
                   fitted_value_v=fdiag, total={}, allelic={}, simulation={})
    log('fitted-value v', fdiag)

    # ---------------- total channel
    eng = TotalREML(S['T'], S['X'])
    rchk = []
    dv = eng.dev(0.65 * np.log(S['Vt']))
    for g in (0, G // 2, G - 1):
        f = S['Vt'][g] ** 0.65
        Xw = S['X'] / np.sqrt(f)[:, None]
        tw = S['T'][g] / np.sqrt(f)
        res_ = tw - Xw @ np.linalg.lstsq(Xw, tw, rcond=None)[0]
        d = ((N - S['X'].shape[1]) * np.log(res_ @ res_) + np.log(f).sum() +
             np.linalg.slogdet(Xw.T @ Xw)[1])
        rchk.append(abs(d - dv[g]))
    summary['reml_engine_check_max_abs'] = float(max(rchk))
    log(f'REML engine vs direct whitened least squares: max |diff| {max(rchk):.2e}')

    mask_t = np.ones((G, N), bool)
    bmask_t = np.broadcast_to(S['leverage'] < H_MAX, (G, N)).copy()
    tot, res_t, curves_t, medg_t = analyse(
        'total, observed v (Gibbs variance + counting term)', eng, S['Vt'], mask_t, S['S2hat'],
        terc, rngs['tot_obs'], extra_y={'raw_e2': S['E'] ** 2}, with_nest=True, MM=MM,
        bin_mask=bmask_t)
    tot['donor_shares'] = donor_shares(S['S2hat'], S['Vt'], bmask_t, medg_t, samples)
    summary['total']['observed_v'] = strip_boot(tot)
    dump(summary)

    totf, res_tf, curves_tf, medg_tf = analyse(
        'total, fitted-value v', eng, Vt_fit, mask_t, S['S2hat'], terc, rngs['tot_fit'],
        extra_y={'raw_e2': S['E'] ** 2}, with_nest=True, MM=MM, bin_mask=bmask_t)
    ds = donor_shares(S['S2hat'], Vt_fit, bmask_t, medg_tf, samples)
    totf['donor_shares'] = ds
    drop = [samples.index(d['donor']) for d in ds['highest_decile'][:3]]
    totf['bins_without_top3_highest_decile_donors'] = dict(
        donors=[samples[j] for j in drop],
        bins=slopes_without_donors(S['S2hat'], Vt_fit, bmask_t, medg_tf, drop))
    summary['total']['fitted_value_v'] = strip_boot(totf)
    dump(summary)

    ok = (S['Vt_gibbs'] > EPS).all(1)
    gi = np.where(ok)[0]
    sub = dict(n_genes=int(ok.sum()), n_genes_excluded=int((~ok).sum()),
               n_zero_gibbs_records=int((S['Vt_gibbs'] <= EPS).sum()))
    eng_s = TotalREML(S['T'][gi], S['X'])
    a1, _, _, _ = analyse('total, observed v with counting term, positive-Gibbs subset',
                          eng_s, S['Vt'][gi], mask_t[gi], S['S2hat'][gi], terc[gi],
                          rngs['tot_sub'], MM=MM, bin_mask=bmask_t[gi])
    a0, _, _, _ = analyse('total, observed Gibbs variance only, positive-Gibbs subset',
                          eng_s, S['Vt_gibbs'][gi], mask_t[gi], S['S2hat'][gi], terc[gi],
                          rngs['tot_sub0'], MM=MM, bin_mask=bmask_t[gi])
    sub['with_counting_term'] = strip_boot(a1)
    sub['gibbs_only'] = strip_boot(a0)
    summary['total']['positive_gibbs_subset'] = sub
    dump(summary)

    # ---------------- allelic channel
    pL, pR = S['pL'], S['pR']
    any_reads = (pL + pR) > 0
    inf = np.minimum(pL, pR) >= 0.5
    na = inf.sum(1)
    ga = np.where(na >= MIN_NA)[0]
    summary['allelic']['records'] = dict(
        pairs_total=int(G * N), pairs_no_allelic_reads=int((~any_reads).sum()),
        pairs_zero_haplotype_excluded=int((any_reads & ~inf).sum()),
        pairs_informative=int(inf.sum()), genes_with_min_na=int(len(ga)),
        records_in_those_genes=int(inf[ga].sum()),
        genes_below_min_na=int((na < MIN_NA).sum()),
        va_nonpositive_among_informative=int((S['Va'][inf] <= EPS).sum()),
        tercile_genes=np.bincount(terc[ga], minlength=3).tolist())
    A2 = S['A'] ** 2
    ea = AllelicML(A2[ga], inf[ga])
    ase, res_a, curves_a, medg_a = analyse(
        'allelic, observed v (Gibbs variance + counting term)', ea, S['Va'][ga], inf[ga], A2[ga],
        terc[ga], rngs['ase_obs'], with_nest=True)
    summary['allelic']['observed_v'] = strip_boot(ase)
    dump(summary)
    asef, res_af, curves_af, medg_af = analyse(
        'allelic, fitted-value v (balanced split)', ea, Va_fit[ga], inf[ga], A2[ga], terc[ga],
        rngs['ase_fit'], with_nest=True)
    summary['allelic']['fitted_value_v'] = strip_boot(asef)
    dump(summary)
    inf0 = inf & (S['Va_gibbs'] > EPS)
    ga0 = np.where(inf0.sum(1) >= MIN_NA)[0]
    sub = dict(n_records_dropped_zero_gibbs=int((inf & ~(S['Va_gibbs'] > EPS)).sum()),
               n_genes=int(len(ga0)), n_records=int(inf0[ga0].sum()))
    ea0 = AllelicML(A2[ga0], inf0[ga0])
    b1, _, _, _ = analyse('allelic, observed v with counting term, positive-Gibbs subset',
                          ea0, S['Va'][ga0], inf0[ga0], A2[ga0], terc[ga0], rngs['ase_sub'])
    b0, _, _, _ = analyse('allelic, observed Gibbs variance only, positive-Gibbs subset',
                          ea0, S['Va_gibbs'][ga0], inf0[ga0], A2[ga0], terc[ga0],
                          rngs['ase_sub0'])
    sub['with_counting_term'] = strip_boot(b1)
    sub['gibbs_only'] = strip_boot(b0)
    summary['allelic']['positive_gibbs_subset'] = sub
    dump(summary)

    # ---------------- method check: v exogenous (fitted-value v held fixed),
    # Gaussian errors under each fitted family, through the identical pipeline
    add_t = min(('add_rel', 'add_abs'), key=lambda f: res_tf[f]['dev'])
    add_a = min(('add_rel', 'add_abs'), key=lambda f: res_af[f]['dev'])
    for chan, truth, rng in (('total', 'power', rngs['sim_tot_pl']),
                             ('total', add_t, rngs['sim_tot_add']),
                             ('allelic', 'power', rngs['sim_ase_pl']),
                             ('allelic', add_a, rngs['sim_ase_add'])):
        if chan == 'total':
            v, mask, medg, res, e0 = Vt_fit, mask_t, medg_tf, res_tf, eng
        else:
            v, mask, medg, res, e0 = Va_fit[ga], inf[ga], medg_af, res_af, ea
        p = dict(res[truth]['param'])
        lf = shape_logf(truth, p, v, mask, medg)
        s2 = e0.sigma2(lf)
        y = np.sqrt(s2[:, None] * np.exp(lf)) * rng.standard_normal(v.shape)
        if chan == 'total':
            E = y @ S['M']
            sim, _, _, _ = analyse(f'exogenous-v simulation: total under {truth}',
                                   TotalREML(y, S['X']), v, mask,
                                   deconvolve(E, S['W'], S['keep_d']), terc, rng,
                                   extra_y={'raw_e2': E ** 2}, MM=MM, do_boot_bins=False,
                                   strata=False, bin_mask=bmask_t)
        else:
            y2 = np.where(mask, y * y, 0.0)
            sim, _, _, _ = analyse(f'exogenous-v simulation: allelic under {truth}',
                                   AllelicML(y2, mask), v, mask, y2, terc[ga], rng,
                                   do_boot_bins=False, strata=False)
        sim['truth'] = dict(family=truth, params={kk: float(vv) for kk, vv in p.items()})
        summary['simulation'][f'exogenous_v_{chan}_{truth}'] = strip_boot(sim)
        dump(summary)

    # ---------------- coupled-null simulation: v from the simulated value
    lf_fit = shape_logf(add_t, dict(res_tf[add_t]['param']), Vt_fit, mask_t, medg_tf)
    summary['simulation']['coupled_total_homoscedastic'] = coupled_total(
        S, k, Vt_fit, np.zeros((G, N)), eng, bmask_t, rngs['cpl_tot_h'], 'homoscedastic')
    summary['simulation']['coupled_total_fitted_additive'] = coupled_total(
        S, k, Vt_fit, lf_fit, eng, bmask_t, rngs['cpl_tot_f'], f'{add_t} at fitted-value v')
    summary['simulation']['coupled_allelic_binomial'] = coupled_allelic(
        S, ga, rngs['cpl_ase_b'], False, 'binomial, balanced')
    summary['simulation']['coupled_allelic_binomial_biological'] = coupled_allelic(
        S, ga, rngs['cpl_ase_bb'], True, 'binomial with per-gene biological imbalance')
    dump(summary)

    # ---------------- per-gene table
    def pg_cols(prefix, curves, res):
        return {f'{prefix}_gamma_pergene': GAMMA_GRID[curves['power'].argmin(1)],
                f'{prefix}_log10_r_pergene': LOGR_GRID[curves['add_rel'].argmin(1)],
                f'{prefix}_dev_addrel_minus_power': curves['add_rel'][:, res['add_rel']['k']] -
                curves['power'][:, res['power']['k']]}
    pg = pd.DataFrame(dict(gene=genes, tercile=np.array(['low', 'mid', 'high'])[terc],
                           mean_log2cpm=tmean, median_Vt=medg_t, median_Vt_fit=medg_tf,
                           **pg_cols('total_obs', curves_t, res_t),
                           **pg_cols('total_fit', curves_tf, res_tf)))
    pa = pd.DataFrame(dict(gene=[genes[g] for g in ga], n_allelic=na[ga], median_Va=medg_a,
                           median_Va_fit=medg_af,
                           **pg_cols('allelic_obs', curves_a, res_a),
                           **pg_cols('allelic_fit', curves_af, res_af)))
    pg.merge(pa, on='gene', how='left').to_csv(f'{OUT}/per_gene.tsv', sep='\t', index=False,
                                               float_format='%.6g')
    dump(summary)
    log(f'wrote {OUT}/summary.json')
    write_tables(summary)
    figures(summary)


# ---------------------------------------------------------------------------
#  tables and figures
# ---------------------------------------------------------------------------

def fit_rows(block, channel, variant, stratum):
    rows = []
    for fam in ('power', 'add_rel', 'add_abs', 'add_trend', 'nest'):
        if fam not in block:
            continue
        r = block[fam]
        rows.append(dict(channel=channel, variant=variant, stratum=stratum, family=fam,
                         n_genes=block['n_genes'], n_records=block['n_records'],
                         params=json.dumps(r['params']), params_ci=json.dumps(r['params_ci']),
                         dev_minus_power=r['dev_minus_power'],
                         ci_lo=r['dev_minus_power_ci'][0], ci_hi=r['dev_minus_power_ci'][1],
                         per_1000_records=r['dev_minus_power_per_1000_records'],
                         boot_frac_power_better=r['boot_frac_power_better']))
    return rows


def analysed_blocks(summary):
    """(channel, variant, block) for every analyse() output in the summary."""
    for ch in ('total', 'allelic'):
        for var in ('observed_v', 'fitted_value_v'):
            if var in summary[ch]:
                yield ch, var, summary[ch][var]
        sub = summary[ch].get('positive_gibbs_subset', {})
        for var in ('with_counting_term', 'gibbs_only'):
            if var in sub:
                yield ch, f'positive_gibbs_subset_{var}', sub[var]
    for kk, blk in summary['simulation'].items():
        if kk.startswith('exogenous_v_'):
            yield kk.split('_')[2], f'exogenous_v_simulated_{blk["truth"]["family"]}', blk


def write_tables(summary):
    rows, brow, grow = [], [], []
    for ch, var, blk in analysed_blocks(summary):
        rows += fit_rows(blk['overall'], ch, var, 'all')
        for t, b in blk.get('by_tercile', {}).items():
            rows += fit_rows(b, ch, var, f'tercile_{t}')
        if 'drop_heavy_genes' in blk:
            rows += fit_rows(blk['drop_heavy_genes']['fits'], ch, var, 'drop_heavy_genes')
        for bname, rec in blk['bins'].items():
            for fam, g in rec.get('bin_gls', {}).items():
                grow.append(dict(channel=ch, variant=var, binning=bname, family=fam,
                                 params=json.dumps(g['param']), chi2=g['chi2'], df=g['df']))
            for yk in [kk for kk in rec if kk in ('y', 'raw_e2')]:
                r = rec[yk]
                for j in range(N_BINS):
                    d = dict(channel=ch, variant=var, binning=bname, y=yk, bin=j + 1,
                             n_records=rec['n_records'][j], xbar=rec['xbar'][j],
                             mean_log10_v=rec['mean_log10_v'][j], beta=r['beta'][j])
                    if 'beta_ci' in r:
                        d['beta_lo'], d['beta_hi'] = r['beta_ci'][j]
                    if j < N_BINS - 1:
                        d['local_slope_to_next'] = r['local_slope'][j]
                        if 'local_slope_ci' in r:
                            d['slope_lo'], d['slope_hi'] = r['local_slope_ci'][j]
                        for fam, pr in rec['pred'].items():
                            d[f'pred_slope_{fam}'] = pr['local_slope'][j]
                    brow.append(d)
    for kk, blk in summary['simulation'].items():
        if not kk.startswith('coupled_'):
            continue
        for against in [a for a in blk if a.startswith('against_')]:
            for bname, rec in blk[against].items():
                for yk in [x for x in rec if x in ('y', 'raw_e2')]:
                    for j in range(N_BINS):
                        d = dict(channel=kk.split('_')[1], variant=f'{kk}_{against}',
                                 binning=bname, y=yk, bin=j + 1, n_records=rec['n_records'][j],
                                 xbar=rec['xbar'][j], mean_log10_v=rec['mean_log10_v'][j],
                                 beta=rec[yk]['beta'][j])
                        if j < N_BINS - 1:
                            d['local_slope_to_next'] = rec[yk]['local_slope'][j]
                        brow.append(d)
    pd.DataFrame(rows).to_csv(f'{OUT}/fits.tsv', sep='\t', index=False, float_format='%.6g')
    pd.DataFrame(brow).to_csv(f'{OUT}/bins.tsv', sep='\t', index=False, float_format='%.6g')
    pd.DataFrame(grow).to_csv(f'{OUT}/bin_gls.tsv', sep='\t', index=False, float_format='%.6g')
    log(f'wrote fits.tsv ({len(rows)} rows), bins.tsv ({len(brow)} rows), '
        f'bin_gls.tsv ({len(grow)} rows)')


def figures(summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    col = dict(obs='#222222', power='#1f77b4', add_rel='#d62728', add_abs='#ff7f0e',
               add_trend='#9467bd', coupled='#2ca02c')
    cpl = dict(total='coupled_total_fitted_additive', allelic='coupled_allelic_binomial_biological')
    for ch in ('total', 'allelic'):
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        for row, var in enumerate(('observed_v', 'fitted_value_v')):
            if var not in summary[ch]:
                continue
            blk = summary[ch][var]
            for ax, bname in zip(axes[row], ('within_gene_decile', 'absolute_v_decile')):
                rec = blk['bins'][bname]
                x = np.array(rec['mean_log10_v'])
                xm = (x[1:] + x[:-1]) / 2
                r = rec['y']
                ls = np.array(r['local_slope'])
                lo = np.array([c[0] for c in r['local_slope_ci']])
                hi = np.array([c[1] for c in r['local_slope_ci']])
                ax.errorbar(xm, ls, yerr=[ls - lo, hi - ls], fmt='o', color=col['obs'],
                            capsize=3, label='observed (95% gene bootstrap)')
                if 'raw_e2' in rec:
                    ax.plot(xm, rec['raw_e2']['local_slope'], 's', mfc='none', color='0.55',
                            label='observed, raw OLS e^2 (carries leverage floor)')
                for fam in ('power', 'add_rel', 'add_abs'):
                    if fam in rec['pred']:
                        ax.plot(xm, rec['pred'][fam]['local_slope'], '-', color=col[fam],
                                label=f'fitted {fam}')
                if var == 'observed_v' and cpl[ch] in summary['simulation']:
                    cr = summary['simulation'][cpl[ch]]['against_recomputed_v'][bname]
                    cx = np.array(cr['mean_log10_v'])
                    ax.plot((cx[1:] + cx[:-1]) / 2, cr['y']['local_slope'], '^--',
                            color=col['coupled'], label='coupled-null simulation')
                ax.axhline(0, color='0.85', lw=0.8)
                ax.axhline(1, color='0.85', lw=0.8)
                ax.set_xlabel('bin-mean log10 v')
                ax.set_ylabel('local slope d log Var / d log v')
                ax.set_title(f'{ch}, {var.replace("_", " ")}: {bname.replace("_", " ")}',
                             fontsize=10)
                ax.legend(fontsize=6.5, loc='best')
        fig.tight_layout()
        fig.savefig(f'{OUT}/local_slopes_{ch}.png', dpi=130)
        plt.close(fig)
    sims = [(kk, b) for kk, b in summary['simulation'].items() if kk.startswith('exogenous_v_')]
    if sims:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for ax, (kk, blk) in zip(axes.ravel(), sims):
            rec = blk['bins']['within_gene_decile']
            x = np.array(rec['mean_log10_v'])
            xm = (x[1:] + x[:-1]) / 2
            ax.plot(xm, rec['y']['local_slope'], 'o', color='k',
                    label='simulated (deconvolved if total)')
            if 'raw_e2' in rec:
                ax.plot(xm, rec['raw_e2']['local_slope'], 's', mfc='none', color='0.55',
                        label='simulated, raw OLS e^2')
            tf = blk['truth']['family']
            ax.plot(xm, rec['pred'][tf]['local_slope'], '-', color=col.get(tf, 'g'),
                    label=f'truth ({tf})')
            ax.set_title(kk.replace('_', ' '), fontsize=10)
            ax.set_xlabel('bin-mean log10 v')
            ax.set_ylabel('local slope')
            ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(f'{OUT}/simulation_local_slopes.png', dpi=130)
        plt.close(fig)
    log('figures written')


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'all'
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    if what in ('prepare', 'all'):
        prepare()
    if what in ('fit', 'all'):
        fit()
    if what == 'tables':
        s = json.load(open(f'{OUT}/summary.json'))
        write_tables(s)
        figures(s)
    log(f'done in {time.time() - t0:.0f} s')

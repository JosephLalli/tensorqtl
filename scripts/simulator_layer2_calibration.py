#!/usr/bin/env python3
"""Biological-variance parameters for the simulator's expression layer, measured
on the corrected pipeline (2026-09-25 rules) over the calibration genes.

QUESTION. A simulator of this cohort has to add biological (between-donor)
variance to its expression values. If that variance is an arbitrary choice,
the ratio of biological to technical variance -- which decides how much the
Gibbs weights can matter -- is arbitrary too. This script measures, on the
cohort, the three quantities the expression layer needs:

1. BIOLOGICAL-TO-TECHNICAL RATIO BY EXPRESSION.
   Total channel. Per gene, T = log2(CPM + 1) of the Salmon point estimate is
   regressed by ordinary least squares on an intercept and the covariates.
   s2 = RSS / (n - p) is the residual variance; Vt is the technical variance
   of the same value (Gibbs variance through the same transform plus the
   delta-method Poisson counting term q_t, as shipped). The quantity the task
   names is s2 / median(Vt). Because the residual also contains technical
   variance, the biological variance is the method-of-moments difference

       sigma2_bio = s2 - sum_i (1 - h_i) Vt_i / (n - p),

   h_i the hat value of donor i: E[RSS] = sum_i (1 - h_i)(sigma2_bio + Vt_i)
   exactly when donor errors are independent with variance sigma2_bio + Vt_i.
   Negative estimates are kept (they are unbiased noise around a small
   value), never truncated before a mean, a quantile or a fit.
   Two covariate designs: FULL (intercept + 17 covariates, what the estimator
   sees) and NO_EXPR_PCS (intercept + age, age^2, RIN, sex, 3 genotype PCs).
   The ten expression PCs are outcome-derived and absorb biological
   between-donor variance, so NO_EXPR_PCS is the variance a simulator must
   generate BEFORE it fits PCs of its own.
   Allelic channel. Per gene, over informative donors (Va > 0) with the
   zero-haplotype pairs dropped (one haplotype below 0.5 reads in the point
   estimate, the rule of corrected_null_store.py), the method-of-moments
   between-donor variance tau2 = var(a) - mean(Va) (var with ddof 1 about the
   unweighted mean; exact expectation tau2 under Var(a_i) = tau2 + Va_i), and
   the DerSimonian-Laird estimate, which computes Cochran's
   Q = sum w_i (a_i - a_w)^2 with w_i = 1/Va_i and a_w the w-weighted mean, and
   sets tau2 = (Q - (k - 1)) / (sum w - sum w^2 / sum w) (reported untruncated).
   The variance is about the gene's mean, so the gene's net imbalance (the
   through-origin model's offset) is not in tau2; it is reported separately.
   Trend: tau2 and sigma2_bio against log2 depth / log2 median CPM, fitted as
   s2_g = tech_g + exp(b0 + b1 x + b2 x^2) by iteratively reweighted least
   squares with weights 1 / (fitted s2)^2 (constant coefficient of variation),
   checked against bin means. The across-gene spread of the biological
   variance at fixed expression is the observed spread of the per-gene
   estimate minus its sampling variance (kurtosis-based delta method), turned
   into a lognormal sd.

2. A DONOR VARIANCE COMPONENT INDEPENDENT OF v.
   log(e^2 / (1 - h)) on gene fixed effects (within-gene demeaning), a natural
   cubic regression spline in log v (6 df) and donor effects (sum to zero).
   Total: e the OLS residual of T (FULL and NO_EXPR_PCS designs), v = Vt.
   Allelic: e = a minus the gene's 1/Va-weighted mean, h_i = w_i / sum w,
   v = Va, zero-haplotype pairs dropped, genes with at least 20 donors. The
   spline absorbs any linear log v term, so this is also the fit of
   log(a^2 / Va). Donor effects delta_d are log variance multipliers. Their
   noise floor is the same fit after permuting donor labels within each gene
   (records keep their e and v, only the label moves); the donor variance
   component is var(delta) minus the mean permutation var(delta).
   Reproduced on the corrected pipeline with their original definitions:
   (a) the per-donor mean of the gene-standardized, leverage-corrected,
       1/Vt-whitened squared total residual (total_channel_null_calibration.py:
       z = r / sqrt(1 - h), z^2 / mean_gene(z^2), mean over genes), on the 46
       nominal-p instrument genes and on all genes, with a model floor from
       e ~ N(0, Vt) records;
   (b) the sd across donors of the per-donor mean of read-count-adjusted
       log Va (variance_layer_measurements.py: per gene polyfit(log depth,
       log Va, 1) over informative donors, genes with >= 60 of them; nanstd
       ddof 0 of donor means), with a within-gene label-permutation floor.

   The allelic arm is repeated with the spline in log haplotype-informative
   depth, because Va itself grows with the donor's imbalance, so a spline in
   log Va is partly a function of the residual it is meant to explain.

   Two-component fits (secondary): pooled within-gene regression of each
   record's squared residual on a counting variance that does not depend on
   the donor's own residual (total: Poisson delta-method variance at the
   gene's median CPM and the donor's library size; allelic: counting variance
   of the log2 ratio at a balanced split of the donor's reads), gene
   intercepts as fixed effects, 95% intervals from resampling genes with
   replacement.

   Sensitivity: the same M1 estimators on Gibbs draw-mean values, to measure
   how much of the between-donor variance is point-estimate error (the draw
   means are never used as values anywhere else).

3. ANCESTRY. Per gene, the partial R^2 of the three genotype PCs for T after
   the RNA-tied covariates only (intercept, age, age^2, RIN, sex, 10
   expression PCs): the share of the residual variance they explain. Null by
   permuting the genotype-PC rows together (200 permutations) and by its
   analytic mean 3 / (n - 15). Storey's pi0 (the share of genes whose p-values
   look null, #{p > 0.5} / (0.5 m)) from the F-test of the three PCs.
   Because the pipeline's expression PCs were built on expression already
   residualized on the REAL genotype PCs, that row permutation is not the
   construction's null: the REBUILT null permutes the genotype-PC rows and
   rebuilds the ten expression PCs against the permuted ones before
   computing the share. Also reported: genotype PC1 alone; the share carried
   by donors 265_D1 and 416_D1, which carry genotype PCs 2 and 3; both
   removed; and expression PCs built without residualizing on genotype PCs.

INPUTS (read-only). Point estimates pL/pR/pT and edgeR effective library
sizes in cache/gibbs_56b63c3b37ed5df8/point_estimates/; Gibbs draws YL/YR/YT
in the same cache (memory-mapped, read in gene chunks); covariates in
cov/log2cpm1_point_calibration_20260925/. Genes: the 11,747 calibration genes
with draws. Samples in cache order; covariates joined by sample id, which
equals the cache's sample ids.

RANDOMNESS. One master SEED = 42. Child streams from
np.random.SeedSequence(SEED).spawn(6): 0 donor-effect permutations,
1 ancestry permutations, 2 model floor for (a), 3 permutation floor for (b),
4 gate permutation, 5 gene resampling for the two-component intervals.

OUTPUT. OUT (below): summaries_calibration_genes.npz (the per-gene inputs,
reusable), per_gene_total.tsv, per_gene_allelic.tsv, donors.tsv,
per_gene_ancestry.tsv, summary.json, fig_*.png, run.log.

Run:  python3 scripts/simulator_layer2_calibration.py [--recompute] [--limit N]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as sps
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / 'scripts'))
sys.path.insert(0, str(REPO))
sys.path.append(str(REPO / 'tensorqtl'))

import tensorqtl.hapmixqtl as HM                       # noqa: E402
import run_hapmixqtl_from_salmon as H                  # noqa: E402

D = Path('/mnt/ssd/lalli/brainvar_hapmix_deploy')
CACHE = D / 'cache' / 'gibbs_56b63c3b37ed5df8'
PE = CACHE / 'point_estimates'
COV = D / 'cov' / 'log2cpm1_point_calibration_20260925'
INSTR = D / 'nominal_p_null_instrument_20260925' / 'inputs_at_lead.npz'
OUT = D / 'simulator_calibration_20260926' / 'layer2'

SEED = 42
KAPPA, LN2, EPS = 0.5, np.log(2.0), 1e-12
CHUNK = 400
MIN_NA = 20               # allelic donors per gene (coupling_reach convention)
MIN_INF_LOGV = 60         # genes for the 0.073 reproduction (original rule)
N_PERM_DONOR = 20
N_PERM_ANC = 200
N_SIM_Z2_46 = 500
N_SIM_Z2_ALL = 20
SPLINE_DF = 6
META = ['age_days', 'age_days_sq', 'rin', 'sex']
# donors named in earlier records (221_D1, 587_D1, 618_D1, 345_D1: top mean z^2
# in total_channel_null_calibration) and the two that carry genotype PCs 2-3
NAMED = ('221_D1', '587_D1', '618_D1', '345_D1', '265_D1', '416_D1')
PAIR = ('265_D1', '416_D1')

T0 = time.time()
_logf = None


def log(*a):
    s = f'[{time.time() - T0:7.1f}s] ' + ' '.join(str(x) for x in a)
    print(s, flush=True)
    if _logf is not None:
        _logf.write(s + '\n')
        _logf.flush()


# ---------------------------------------------------------------------------
#  inputs
# ---------------------------------------------------------------------------

def load_summaries(genes, samples, eff_lib):
    """A, T, Va, Vt (shipped: Gibbs variance + counting term) and the draw-only
    variances, for ``genes`` in cache sample order, from the shipped
    summaries_from_point_estimates. Gate on the first chunk: the shipped
    count_noise=True output equals draw-only plus the counting term."""
    genes_all = (CACHE / 'genes.txt').read_text().split()
    gi = {g: i for i, g in enumerate(genes_all)}
    rows = np.array([gi[g] for g in genes])
    order = np.argsort(rows)                     # read the memmap in file order
    srows = rows[order]
    pLm, pRm, pTm = (np.load(PE / f'{k}.npy', mmap_mode='r') for k in ('pL', 'pR', 'pT'))
    Ym = {k: np.load(CACHE / f'{k}.npy', mmap_mode='r') for k in ('YL', 'YR', 'YT')}
    G, N = len(genes), len(samples)
    out = {k: np.empty((G, N)) for k in ('pL', 'pR', 'pT', 'A', 'T', 'Va_draw', 'Vt_draw',
                                         'mL_draw', 'mR_draw', 'mT_draw')}
    gate = None
    for s in range(0, G, CHUNK):
        r = srows[s:s + CHUNK]
        dst = order[s:s + CHUNK]
        pL, pR, pT = (np.asarray(m[r], dtype=float) for m in (pLm, pRm, pTm))
        yL, yR, yT = (np.asarray(Ym[k][r], dtype=float) for k in ('YL', 'YR', 'YT'))
        A, T, Va0, Vt0, _ = HM.summaries_from_point_estimates(
            pL, pR, pT, eff_lib, yL, yR, yT, kappa=KAPPA, count_noise=False)
        if gate is None:
            _, _, Va1, Vt1, _ = HM.summaries_from_point_estimates(
                pL, pR, pT, eff_lib, yL, yR, yT, kappa=KAPPA, count_noise=True)
            qa, qt = counting_terms(pL, pR, pT, eff_lib)
            ref_a = np.where((pL + pR) <= 0, 0.0, Va0 + qa)
            gate = dict(max_abs_Va=float(np.max(np.abs(Va1 - ref_a))),
                        max_abs_Vt=float(np.max(np.abs(Vt1 - (Vt0 + qt)))))
            if gate['max_abs_Va'] > 1e-12 or gate['max_abs_Vt'] > 1e-12:
                raise SystemExit(f'counting-term gate failed: {gate}')
            log('gate: shipped variance = draw-only + counting term', gate)
        # draw means: used ONLY for the sensitivity that asks how much of the
        # between-donor variance is point-estimate error; never as the value
        for k, v in (('pL', pL), ('pR', pR), ('pT', pT), ('A', A), ('T', T),
                     ('Va_draw', Va0), ('Vt_draw', Vt0), ('mL_draw', yL.mean(2)),
                     ('mR_draw', yR.mean(2)), ('mT_draw', yT.mean(2))):
            out[k][dst] = v
        log(f'  summaries {min(s + CHUNK, G)}/{G}')
    qa, qt = counting_terms(out['pL'], out['pR'], out['pT'], eff_lib)
    out['Va'] = np.where((out['pL'] + out['pR']) <= 0, 0.0, out['Va_draw'] + qa)
    out['Vt'] = out['Vt_draw'] + qt
    out['q_t'] = qt
    return out, gate


def counting_terms(pL, pR, pT, eff_lib):
    """The counting terms exactly as summaries_from_point_estimates adds them."""
    k = 1e6 / np.asarray(eff_lib, float)
    qa = (1.0 / (pL + KAPPA) + 1.0 / (pR + KAPPA)) / LN2 ** 2
    y = pT + 0.5
    qt = (k[None, :] ** 2) * y / ((k[None, :] * y + 1.0) ** 2 * LN2 ** 2)
    return qa, qt


def design(cov, cols):
    Z = np.column_stack([np.ones(len(cov))] + [cov[c].to_numpy(float) for c in cols])
    return Z


def ols_resid(Y, Z):
    """OLS residuals of every row of Y [genes x n] on Z [n x p]; hat values."""
    Q, _ = np.linalg.qr(Z)
    E = Y - (Y @ Q) @ Q.T
    h = (Q ** 2).sum(1)
    return E, h


# ---------------------------------------------------------------------------
#  measurement 1 helpers
# ---------------------------------------------------------------------------

def irls_trend(x, s2, tech, n_iter=8):
    """Fit s2_g = tech_g + exp(b0 + b1 x + b2 x^2) across genes, weights
    1 / fitted^2 (constant coefficient of variation), untruncated data."""
    xc = x - np.median(x)
    b = np.array([np.log(max(np.mean(s2 - tech), 1e-4)), 0.0, 0.0])
    for _ in range(n_iter):
        fit = tech + np.exp(b[0] + b[1] * xc + b[2] * xc ** 2)
        wt = 1.0 / fit

        def res(bb):
            return (s2 - tech - np.exp(bb[0] + bb[1] * xc + bb[2] * xc ** 2)) * wt
        b = least_squares(res, b, method='lm').x
    # re-express about x = 0
    x0 = float(np.median(x))
    c2 = b[2]
    c1 = b[1] - 2 * b[2] * x0
    c0 = b[0] - b[1] * x0 + b[2] * x0 ** 2
    return dict(b0=float(c0), b1=float(c1), b2=float(c2), x_center=x0)


def eval_trend(tr, x):
    return np.exp(tr['b0'] + tr['b1'] * x + tr['b2'] * x ** 2)


def sampling_var_s2(R, df):
    """Delta-method sampling variance of s2 from standardized residuals R
    [genes x n] (leverage-corrected): s2^2 (2/df + (kurt - 3)/n)."""
    m2 = (R ** 2).mean(1)
    m4 = (R ** 4).mean(1)
    kurt = m4 / m2 ** 2
    n = R.shape[1]
    return (2.0 / df + (kurt - 3.0) / n), kurt       # relative variance (times s2^2)


def bin_table(x, edges, labels, cols):
    """Per-bin summaries of the per-gene columns in ``cols`` (dict name -> array)."""
    rows = []
    for lo, hi, lab in zip(edges[:-1], edges[1:], labels):
        s = (x >= lo) & (x < hi)
        if s.sum() == 0:
            continue
        r = dict(bin=lab, n_genes=int(s.sum()))
        for name, arr in cols.items():
            v = arr[s]
            v = v[np.isfinite(v)]
            if name.startswith('mean:'):
                r[name[5:] + '_mean'] = float(v.mean())
                r[name[5:] + '_mean_se'] = float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else np.nan
                r[name[5:] + '_share_negative'] = float((v < 0).mean())
                r[name[5:] + '_q10_q50_q90'] = [float(q) for q in np.quantile(v, [.1, .5, .9])]
            else:
                r[name + '_q10_q25_q50_q75_q90'] = [float(q) for q in np.quantile(v, [.1, .25, .5, .75, .9])]
        rows.append(r)
    return rows


def lognormal_spread(bio, relvar_s2, s2, x, trend, edges, labels):
    """Across-gene spread of the biological variance at fixed expression, per
    bin: (var of bio about the trend - mean sampling var of bio) / (bin mean of
    bio)^2 is its CV^2, turned into a lognormal sd sqrt(log(1 + CV^2)). Where
    bio exceeds 4x its own sampling sd in 90% of the bin's genes, also the
    direct sd of log(bio) net of its delta-method sampling variance."""
    pred = eval_trend(trend, x)
    rows = []
    for lo, hi, lab in zip(edges[:-1], edges[1:], labels):
        s = (x >= lo) & (x < hi)
        if s.sum() < 30:
            continue
        dev = bio[s] - pred[s]
        obs = float(np.var(dev, ddof=1))
        samp = relvar_s2[s] * s2[s] ** 2
        floor = float(np.mean(samp))
        mu = float(np.mean(bio[s]))
        cv2 = (obs - floor) / mu ** 2
        r = dict(bin=lab, n_genes=int(s.sum()), obs_var=obs, sampling_floor=floor,
                 floor_share=floor / obs, bin_mean=mu, trend_mean=float(np.mean(pred[s])),
                 cv2=float(cv2), lognormal_sd=float(np.sqrt(np.log1p(cv2))) if cv2 > 0 else 0.0)
        # direct log-scale spread where technical variance is under a tenth of
        # the residual variance (bio ~ s2): var of log s2 about its bin mean
        # minus the mean delta-method sampling variance of log s2 (relvar)
        tech_share = np.median((s2[s] - bio[s]) / s2[s])
        r['median_tech_share'] = float(tech_share)
        if tech_share < 0.1:
            v = float(np.var(np.log(s2[s]), ddof=1))
            fl = float(np.mean(relvar_s2[s]))
            r.update(direct_log_sd=float(np.sqrt(max(v - fl, 0))), direct_log_var_obs=v,
                     direct_log_sampling_floor=fl)
        rows.append(r)
    return rows


def two_component(y, x, gene, n_gene, rng, n_boot=500):
    """Pooled within-gene regression of per-record squared residuals y on a
    ratio-independent counting variance x: E y = s2_g + c x. The gene
    intercepts are fixed effects; c is identified from within-gene variation
    in x. 95% interval for c from resampling genes with replacement.
    Returns c, its interval and s2_g."""
    cnt = np.bincount(gene, minlength=n_gene).astype(float)
    xm = np.bincount(gene, weights=x, minlength=n_gene) / cnt
    ym = np.bincount(gene, weights=y, minlength=n_gene) / cnt
    xd = x - xm[gene]
    yd = y - ym[gene]
    sxy = np.bincount(gene, weights=xd * yd, minlength=n_gene)
    sxx = np.bincount(gene, weights=xd * xd, minlength=n_gene)
    c = sxy.sum() / sxx.sum()
    bs = []
    for _ in range(n_boot):
        i = rng.integers(0, n_gene, n_gene)
        bs.append(sxy[i].sum() / sxx[i].sum())
    s2g = ym - c * xm
    return dict(c=float(c), c_ci95=[float(np.quantile(bs, .025)), float(np.quantile(bs, .975))],
                n_records=int(len(y)), n_genes=int(n_gene)), s2g, xm, ym


# ---------------------------------------------------------------------------
#  measurement 2 helpers
# ---------------------------------------------------------------------------

def spline_basis(x):
    import patsy
    B = np.asarray(patsy.dmatrix(f'cr(x, df={SPLINE_DF}) - 1', {'x': x},
                                 return_type='matrix'))
    return B


def demean_by_group(M, grp, ngrp):
    """Subtract group means from each column of M (rows grouped by ``grp``)."""
    M = np.asarray(M, float)
    cnt = np.bincount(grp, minlength=ngrp).astype(float)
    if M.ndim == 1:
        mu = np.bincount(grp, weights=M, minlength=ngrp) / np.maximum(cnt, 1)
        return M - mu[grp]
    out = np.empty_like(M)
    for j in range(M.shape[1]):
        mu = np.bincount(grp, weights=M[:, j], minlength=ngrp) / np.maximum(cnt, 1)
        out[:, j] = M[:, j] - mu[grp]
    return out


def prepare_fit(y, lv, gene, n_gene, donor, n_donor):
    """Within-gene demeaned outcome and spline basis (gene fixed effects by the
    Frisch-Waugh-Lovell theorem: demeaning equals including the dummies), and
    the donor block's cross-products in closed form.

    With D~ the within-gene demeaned donor indicators, and y~ and B~ already
    demeaned within gene: D~'y~ and D~'B~ are per-donor sums of y~ and B~, and
    D~'D~ = diag(n_d) - P' diag(1/n_g) P, P the gene-by-donor presence matrix.
    A within-gene permutation of donor labels leaves P, hence D~'D~, unchanged."""
    B = spline_basis(lv)
    yd = demean_by_group(y, gene, n_gene)
    Bd = demean_by_group(B, gene, n_gene)
    cnt = np.bincount(gene, minlength=n_gene).astype(float)
    P = np.zeros((n_gene, n_donor))
    P[gene, donor] = 1.0
    if P.sum() != len(y):
        raise SystemExit('a donor appears twice within a gene')
    DtD = np.diag(P.sum(0)) - (P / cnt[:, None]).T @ P
    return dict(yd=yd, Bd=Bd, B=B, lv=lv, gene=gene, n_gene=n_gene, DtD=DtD,
                BtB=Bd.T @ Bd, Bty=Bd.T @ yd, yty=float(yd @ yd))


def donor_fit(F, donor, n_donor, with_spline=True, with_donor=True):
    """log e^2 on gene FE (demeaning) + spline(log v) + donor effects, solved
    from the normal equations with a minimum-norm pseudo-inverse (the donor
    block's null direction is the constant, so the effects sum to zero).
    Returns donor effects, RSS and the spline fit at quantiles of log v."""
    yd = F['yd']
    nb = F['Bd'].shape[1] if with_spline else 0
    if with_donor:
        Dty = np.bincount(donor, weights=yd, minlength=n_donor)
        DtB = np.column_stack([np.bincount(donor, weights=F['Bd'][:, j], minlength=n_donor)
                               for j in range(nb)]) if with_spline else None
    if with_spline and with_donor:
        XtX = np.block([[F['BtB'], DtB.T], [DtB, F['DtD']]])
        Xty = np.concatenate([F['Bty'], Dty])
    elif with_spline:
        XtX, Xty = F['BtB'], F['Bty']
    elif with_donor:
        XtX, Xty = F['DtD'], Dty
    else:
        return None, F['yty'], None
    beta = np.linalg.pinv(XtX, rcond=1e-10, hermitian=True) @ Xty
    rss = F['yty'] - float(beta @ Xty)
    out_d, grid = None, None
    if with_spline:
        lv = F['lv']
        qs = np.quantile(lv, [.05, .1, .25, .5, .75, .9, .95])
        f = F['B'] @ beta[:nb]
        order = np.argsort(lv)
        fq = np.interp(qs, lv[order], f[order])
        grid = dict(log_v_quantiles_q05_q10_q25_q50_q75_q90_q95=[float(q) for q in qs],
                    f_minus_f_median=[float(v - fq[3]) for v in fq],
                    slope_q10_to_q90=float((fq[5] - fq[1]) / (qs[5] - qs[1])))
    if with_donor:
        out_d = beta[nb:] - beta[nb:].mean()
    return out_d, float(rss), grid


def donor_fit_dense(F, donor, n_donor):
    """The same fit from the explicit design matrix; used only as a gate."""
    Dm = np.zeros((len(F['yd']), n_donor))
    Dm[np.arange(len(donor)), donor] = 1.0
    X = np.hstack([F['Bd'], demean_by_group(Dm, F['gene'], F['n_gene'])])
    beta = np.linalg.lstsq(X, F['yd'], rcond=None)[0]
    res = F['yd'] - X @ beta
    nb = F['Bd'].shape[1]
    d = beta[nb:] - beta[nb:].mean()
    return d, float(res @ res)


def permute_within_gene(donor, gene, rng):
    """Shuffle donor labels among each gene's records."""
    d = donor.copy()
    starts = np.flatnonzero(np.r_[True, gene[1:] != gene[:-1]])
    ends = np.r_[starts[1:], len(gene)]
    for s, e in zip(starts, ends):
        d[s:e] = d[s:e][rng.permutation(e - s)]
    return d


def donor_component(y, lv, gene, donor, n_gene, n_donor, rng, label):
    """Donor effects, their permutation floor and variance shares."""
    F = prepare_fit(y, lv, gene, n_gene, donor, n_donor)
    d_obs, rss_full, grid = donor_fit(F, donor, n_donor)
    _, rss_nod, _ = donor_fit(F, donor, n_donor, with_donor=False)
    d_nospl, _, _ = donor_fit(F, donor, n_donor, with_spline=False)
    tss_within = float(F['yd'] @ F['yd'])
    perm_var, perm_share = [], []
    for _ in range(N_PERM_DONOR):
        dp = permute_within_gene(donor, gene, rng)
        d_p, rss_p, _ = donor_fit(F, dp, n_donor)
        perm_var.append(float(np.var(d_p, ddof=1)))
        perm_share.append((rss_nod - rss_p) / rss_nod)
    var_obs = float(np.var(d_obs, ddof=1))
    floor = float(np.mean(perm_var))
    share = (rss_nod - rss_full) / rss_nod
    n_rec = len(y)
    dfree = n_rec - n_gene
    resid_var = rss_nod / dfree
    log(f'  {label}: sd(delta) {np.sqrt(var_obs):.4f}, floor {np.sqrt(floor):.4f}, '
        f'share {share:.5f} (floor {np.mean(perm_share):.5f})')
    return dict(
        n_records=int(n_rec), n_genes=int(n_gene),
        sd_delta=float(np.sqrt(var_obs)), sd_delta_perm_floor=float(np.sqrt(floor)),
        sd_delta_perm_floor_range=[float(np.sqrt(min(perm_var))), float(np.sqrt(max(perm_var)))],
        sd_donor_component=float(np.sqrt(max(var_obs - floor, 0.0))),
        var_donor_component=float(var_obs - floor),
        share_of_within_gene_var_after_spline=float(share),
        share_perm_floor_mean=float(np.mean(perm_share)),
        share_perm_floor_max=float(np.max(perm_share)),
        resid_var_logsq_after_spline=float(resid_var),
        tss_within=tss_within,
        spline=grid,
        sd_delta_without_spline=float(np.std(d_nospl, ddof=1)),
    ), d_obs, d_nospl


def corr_block(x, cov_d, extra, names):
    out = {}
    for nm in names:
        z = extra[nm] if nm in extra else cov_d[nm].to_numpy(float)
        sp = sps.spearmanr(x, z)
        pr = sps.pearsonr(x, z)
        out[nm] = dict(spearman=float(sp[0]), spearman_p=float(sp[1]),
                       pearson=float(pr[0]), pearson_p=float(pr[1]))
    return out


# ---------------------------------------------------------------------------
#  main
# ---------------------------------------------------------------------------

def main():
    global _logf
    ap = argparse.ArgumentParser()
    ap.add_argument('--recompute', action='store_true')
    ap.add_argument('--limit', type=int, default=None, help='first N genes only (timing)')
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    _logf = open(OUT / ('run.log' if args.limit is None else 'run_limit.log'), 'w')
    ss = np.random.SeedSequence(SEED).spawn(6)
    rng_donor, rng_anc, rng_sim, rng_lv = (np.random.default_rng(s) for s in ss[:4])
    rng_boot = np.random.default_rng(ss[5])

    samples = (CACHE / 'samples.txt').read_text().split()
    N = len(samples)
    eff_lib, cal = H.read_edger_dir(PE / 'edger', samples)
    genes_cache = set((CACHE / 'genes.txt').read_text().split())
    genes = [g for g in cal if g in genes_cache]
    instr = np.load(INSTR, allow_pickle=True)
    genes46 = [str(g) for g in instr['genes']]
    extra = [g for g in genes46 if g not in set(genes)]
    log(f'{len(genes)} calibration genes with draws; 46-gene instrument set: '
        f'{len(genes46) - len(extra)} in it, outside: {extra}')
    if args.limit:
        genes = genes[:args.limit]
    all_genes = genes + extra

    cov = pd.read_csv(COV / 'covariates.tsv', sep='\t', index_col=0)
    cov.index = cov.index.astype(str)
    if set(cov.index) != set(samples):
        raise SystemExit('covariate samples differ from the cache samples')
    cov = cov.loc[samples]
    gcols = (COV / 'genotype_covariates.txt').read_text().split()
    ecols = [c for c in cov.columns if c.startswith('expr_pc')]
    assert set(META + gcols + ecols) == set(cov.columns), cov.columns
    ortho = dict(
        max_abs_corr_geno_vs_expr=float(np.abs(np.corrcoef(cov[gcols + ecols].T.values)[:3, 3:]).max()),
        max_abs_corr_meta_vs_expr=float(np.abs(np.corrcoef(cov[META + ecols].T.values)[:4, 4:]).max()),
        max_abs_corr_geno_vs_meta=float(np.abs(np.corrcoef(cov[gcols + META].T.values)[:3, 3:]).max()))
    log('covariate orthogonality', ortho)

    cache_npz = OUT / ('summaries_calibration_genes.npz' if args.limit is None
                       else f'summaries_limit{args.limit}.npz')
    if cache_npz.exists() and not args.recompute:
        Z = np.load(cache_npz, allow_pickle=True)
        if [str(g) for g in Z['genes']] != all_genes:
            raise SystemExit(f'{cache_npz} holds another gene list; pass --recompute')
        S = {k: Z[k] for k in Z.files if k not in ('genes', 'samples', 'gate')}
        gate = json.loads(str(Z['gate']))
        log(f'reused {cache_npz}')
    else:
        t0 = time.time()
        S, gate = load_summaries(all_genes, samples, eff_lib)
        log(f'summaries for {len(all_genes)} genes in {time.time() - t0:.0f}s')
        np.savez(cache_npz, genes=np.array(all_genes), samples=np.array(samples),
                 gate=json.dumps(gate), **S)

    G = len(genes)                     # analysis genes (calibration)
    sel = slice(0, G)
    T, Vt, Vt_draw = S['T'][sel], S['Vt'][sel], S['Vt_draw'][sel]
    A, Va, Va_draw = S['A'][sel], S['Va'][sel], S['Va_draw'][sel]
    pL, pR, pT = S['pL'][sel], S['pR'][sel], S['pT'][sel]
    cpm = pT / eff_lib[None, :] * 1e6
    med_cpm = np.median(cpm, 1)
    x_cpm = np.log2(med_cpm)
    summ = dict(n_genes=G, n_donors=N, gate=gate, covariate_orthogonality=ortho,
                seed=SEED, spline_df=SPLINE_DF, n_perm_donor=N_PERM_DONOR,
                n_perm_ancestry=N_PERM_ANC)

    # =====================================================================
    #  1. biological-to-technical ratio, total channel
    # =====================================================================
    log('measurement 1: total channel')
    designs = {'full': META + ecols + gcols, 'no_expr_pcs': META + gcols, 'meta_only': list(META)}
    cpm_edges = np.array([-np.inf, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, np.inf])
    cpm_labels = ['<2', '2-4', '4-8', '8-16', '16-32', '32-64', '64-128', '128-256',
                  '256-512', '512-1024', '>=1024']
    pg_total = pd.DataFrame(dict(gene=genes, median_cpm=med_cpm,
                                 median_count=np.median(pT, 1),
                                 median_Vt=np.median(Vt, 1),
                                 median_Vt_draw=np.median(Vt_draw, 1),
                                 median_Vt_draw_over_qt=np.median(Vt_draw / S['q_t'][sel], 1)))
    tot = dict(designs={k: dict(n_params=1 + len(v), covariates=v) for k, v in designs.items()})
    resid_store = {}
    for dname, cols in designs.items():
        Zd = design(cov, cols)
        p = Zd.shape[1]
        df = N - p
        E, h = ols_resid(T, Zd)
        resid_store[dname] = (E, h)
        s2 = (E ** 2).sum(1) / df
        tech = (Vt * (1 - h)[None, :]).sum(1) / df
        tech_draw = (Vt_draw * (1 - h)[None, :]).sum(1) / df
        bio = s2 - tech
        R = E / np.sqrt(1 - h)[None, :]
        relv, kurt = sampling_var_s2(R, df)
        mv = np.median(Vt, 1)
        pg_total[f's2_{dname}'] = s2
        pg_total[f'tech_{dname}'] = tech
        pg_total[f'bio_{dname}'] = bio
        pg_total[f'bio_draw_only_{dname}'] = s2 - tech_draw
        pg_total[f's2_over_medVt_{dname}'] = s2 / mv
        pg_total[f'bio_over_medVt_{dname}'] = bio / mv
        pg_total[f'bio_over_tech_{dname}'] = bio / tech
        pg_total[f'kurtosis_{dname}'] = kurt
        # log-linear trend of the literal ratio
        lr = np.log(s2 / mv)
        Xp = np.column_stack([np.ones(G), x_cpm, x_cpm ** 2])
        bq = np.linalg.lstsq(Xp, lr, rcond=None)[0]
        Xl = Xp[:, :2]
        bl = np.linalg.lstsq(Xl, lr, rcond=None)[0]
        tr = irls_trend(x_cpm, s2, tech)
        # bin means for the check
        bins = bin_table(x_cpm, cpm_edges, cpm_labels, {
            's2_over_medVt': s2 / mv, 'bio_over_medVt': bio / mv, 'mean:bio': bio,
            'median_Vt': mv, 'median_cpm': med_cpm, 's2': s2,
            'Vt_draw_over_qt': pg_total['median_Vt_draw_over_qt'].to_numpy()})
        spread = lognormal_spread(bio, relv, s2, x_cpm, tr, cpm_edges, cpm_labels)
        pooled_hi = x_cpm >= 4
        tot[dname] = dict(
            df=int(df),
            s2_over_medVt_quantiles={q: float(np.quantile(s2 / mv, q)) for q in (.1, .25, .5, .75, .9)},
            bio_over_medVt_quantiles={q: float(np.quantile(bio / mv, q)) for q in (.1, .25, .5, .75, .9)},
            bio_quantiles_log2sq={q: float(np.quantile(bio, q)) for q in (.1, .25, .5, .75, .9)},
            bio_mean=float(bio.mean()), bio_share_negative=float((bio < 0).mean()),
            bio_draw_only_mean=float((s2 - tech_draw).mean()),
            tech_share_of_s2_median=float(np.median(tech / s2)),
            log_ratio_trend_quadratic=dict(c0=float(bq[0]), c1=float(bq[1]), c2=float(bq[2]),
                                           resid_sd=float(np.std(lr - Xp @ bq, ddof=3)),
                                           form='ln(s2/median Vt) = c0 + c1 x + c2 x^2, x = log2 median CPM'),
            log_ratio_trend_linear=dict(c0=float(bl[0]), c1=float(bl[1]),
                                        resid_sd=float(np.std(lr - Xl @ bl, ddof=2))),
            bio_trend=dict(tr, form='sigma2_bio(x) = exp(b0 + b1 x + b2 x^2), x = log2 median CPM, '
                           'log2(CPM+1) squared units; fitted as s2 = tech + sigma2_bio(x) by IRLS'),
            bio_trend_at=dict((f'{2 ** k:g}_cpm', float(eval_trend(tr, k))) for k in range(0, 12)),
            bins=bins, across_gene_spread=spread,
            kurtosis_median=float(np.median(kurt)),
            bio_mean_cpm_ge_16=float(bio[pooled_hi].mean()),
        )
        log(f'  {dname}: median s2/medVt {np.median(s2 / mv):.2f}, median bio/medVt '
            f'{np.median(bio / mv):.2f}, share bio<0 {(bio < 0).mean():.4f}, trend {tr}')
    summ['total'] = tot
    # Gibbs variance against Poisson, by expression (to generate tech from counts)
    summ['total']['Vt_draw_over_poisson_q_t_quantiles'] = {
        q: float(np.quantile(pg_total['median_Vt_draw_over_qt'], q)) for q in (.1, .25, .5, .75, .9)}
    tlog = np.log2(pg_total['median_Vt'].to_numpy())
    bt = np.linalg.lstsq(np.column_stack([np.ones(G), x_cpm]), tlog, rcond=None)[0]
    summ['total']['median_Vt_trend'] = dict(c0=float(bt[0]), c1=float(bt[1]),
                                            form='log2 median Vt = c0 + c1 log2 median CPM')
    # two-component: E e_i^2/(1-h_i) = sigma2_g + c_t x_i, x_i the Poisson
    # delta-method variance of log2(CPM+1) at the gene's median CPM and donor
    # i's library size (so x does not depend on the donor's own residual)
    kk = 1e6 / eff_lib
    yh = med_cpm[:, None] / kk[None, :] + 0.5
    x_t = (kk[None, :] ** 2) * yh / ((kk[None, :] * yh + 1.0) ** 2 * LN2 ** 2)
    gidx = np.repeat(np.arange(G), N)
    lo16 = np.where(med_cpm < 16)[0]
    tc = {}
    for dname in designs:
        E, h = resid_store[dname]
        yy = E ** 2 / (1 - h)[None, :]
        r, s2g, _, _ = two_component(yy.ravel(), x_t.ravel(), gidx, G, rng_boot)
        rl, _, _, _ = two_component(yy[lo16].ravel(), x_t[lo16].ravel(),
                                    np.repeat(np.arange(len(lo16)), N), len(lo16), rng_boot)
        r['c_genes_below_16_cpm'] = rl
        r['median_Vt_over_x'] = float(np.median(Vt / x_t))
        r['median_Vt_draw_over_x'] = float(np.median(Vt_draw / x_t))
        r['sigma2_g_bins'] = bin_table(x_cpm, cpm_edges, cpm_labels, {'mean:sigma2_g': s2g})
        pg_total[f'sigma2_g_two_component_{dname}'] = s2g
        tc[dname] = r
        log(f'  total two-component {dname}: c_t {r["c"]:.3f} {r["c_ci95"]}, '
            f'below 16 CPM {rl["c"]:.3f} {rl["c_ci95"]}')
    summ['total']['two_component'] = tc

    # =====================================================================
    #  1. allelic channel
    # =====================================================================
    log('measurement 1: allelic channel')
    zero = (pL < 0.5) ^ (pR < 0.5)
    inf_keep = Va > EPS
    inf_drop = inf_keep & ~zero
    depth = pL + pR
    summ['allelic_pairs'] = dict(
        informative=int(inf_keep.sum()), zero_haplotype=int((inf_keep & zero).sum()),
        after_drop=int(inf_drop.sum()),
        genes_ge_min_na_drop=int((inf_drop.sum(1) >= MIN_NA).sum()),
        genes_ge_min_na_keep=int((inf_keep.sum(1) >= MIN_NA).sum()))
    al = {}
    pg_all = []
    dep_edges = np.log2([1e-9, 30, 100, 300, 1000, 3000, 1e12])
    dep_labels = ['<30', '30-100', '100-300', '300-1000', '1000-3000', '>=3000']
    for arm, M in (('drop', inf_drop), ('keep', inf_keep)):
        rows = []
        for j in range(G):
            m = M[j]
            k = int(m.sum())
            if k < MIN_NA:
                continue
            a, v, vd = A[j, m], Va[j, m], Va_draw[j, m]
            s2 = a.var(ddof=1)
            w = 1.0 / v
            aw = (w * a).sum() / w.sum()
            Q = (w * (a - aw) ** 2).sum()
            tdl = (Q - (k - 1)) / (w.sum() - (w ** 2).sum() / w.sum())
            r = (a - a.mean()) / np.sqrt(1 - 1.0 / k)
            m2, m4 = (r ** 2).mean(), (r ** 4).mean()
            rows.append(dict(gene=genes[j], n_a=k, median_depth=float(np.median(depth[j, m])),
                             mean_a=float(a.mean()), se_mean_a=float(np.sqrt(s2 / k)),
                             weighted_mean_a=float(aw),
                             var_a=float(s2), mean_Va=float(v.mean()), median_Va=float(np.median(v)),
                             mean_Va_draw=float(vd.mean()),
                             tau2_mom=float(s2 - v.mean()), tau2_mom_draw_only=float(s2 - vd.mean()),
                             tau2_dl=float(tdl), resid_kurtosis=float(m4 / m2 ** 2)))
        P = pd.DataFrame(rows)
        P['tau2_over_medVa'] = P.tau2_mom / P.median_Va
        P['var_over_meanVa'] = P.var_a / P.mean_Va
        x = np.log2(P.median_depth.to_numpy())
        s2 = P.var_a.to_numpy()
        tech = P.mean_Va.to_numpy()
        tau = P.tau2_mom.to_numpy()
        relv = 2.0 / (P.n_a.to_numpy() - 1) + (P['resid_kurtosis'].to_numpy() - 3) / P.n_a.to_numpy()
        tr = irls_trend(x, s2, tech)
        # A plain log-linear fit of tau2 against depth is not possible (tau2 < 0
        # in part); report the IRLS curve and the bin means it is checked against.
        bins = bin_table(x, dep_edges, dep_labels, {
            'mean:tau2_mom': tau, 'mean:tau2_dl': P.tau2_dl.to_numpy(),
            'mean:tau2_mom_draw_only': P.tau2_mom_draw_only.to_numpy(),
            'tau2_over_medVa': P.tau2_over_medVa.to_numpy(),
            'var_over_meanVa': P.var_over_meanVa.to_numpy(),
            'median_Va': P.median_Va.to_numpy(), 'n_a': P.n_a.to_numpy().astype(float),
            'mean_a': P.mean_a.to_numpy()})
        spread = lognormal_spread(tau, relv, s2, x, tr, dep_edges, dep_labels)
        z_mean = P.mean_a / P.se_mean_a
        # two-component: E (a_i - mean a)^2 k/(k-1) = tau2_g + c_a x_i, x_i the
        # counting variance of the log2 ratio at a BALANCED split of donor i's
        # haplotype-informative reads, 4/((n_i + 1) ln2^2), which does not
        # depend on the ratio (Va does: an imbalanced donor has a larger Va)
        gene_row = {g: jj for jj, g in enumerate(genes)}
        yy, xx, gg, vv, vd = [], [], [], [], []
        for gi2, g in enumerate(P.gene):
            j = gene_row[g]
            m = M[j]
            k = m.sum()
            a = A[j, m]
            yy.append((a - a.mean()) ** 2 * k / (k - 1))
            xx.append(4.0 / ((depth[j, m] + 1.0) * LN2 ** 2))
            gg.append(np.full(k, gi2))
            vv.append(Va[j, m])
            vd.append(Va_draw[j, m])
        yy, xx, gg = np.concatenate(yy), np.concatenate(xx), np.concatenate(gg)
        vv, vd = np.concatenate(vv), np.concatenate(vd)
        tcomp, tau_g, xm_g, _ = two_component(yy, xx, gg, len(P), rng_boot)
        P['tau2_g_two_component'] = tau_g
        P['mean_x_balanced_counting'] = xm_g
        tcomp['median_Va_over_x'] = float(np.median(vv / xx))
        tcomp['median_Va_draw_over_x'] = float(np.median(vd / xx))
        by_depth = []
        for lo, hi, lab in zip(dep_edges[:-1], dep_edges[1:], dep_labels):
            sg = np.where((x >= lo) & (x < hi))[0]
            if len(sg) < 30:
                continue
            msk = np.isin(gg, sg)
            remap = -np.ones(len(P), int)
            remap[sg] = np.arange(len(sg))
            rb, tb_, _, _ = two_component(yy[msk], xx[msk], remap[gg[msk]], len(sg), rng_boot, n_boot=200)
            rb.update(bin=lab, tau2_g_mean=float(tb_.mean()), tau2_g_median=float(np.median(tb_)),
                      tau2_g_share_negative=float((tb_ < 0).mean()),
                      median_Va_draw_over_x=float(np.median(vd[msk] / xx[msk])))
            by_depth.append(rb)
        tcomp['by_depth'] = by_depth
        tcomp['tau2_g_quantiles'] = {q: float(np.quantile(tau_g, q)) for q in (.1, .25, .5, .75, .9)}
        tcomp['tau2_g_mean'] = float(tau_g.mean())
        tcomp['tau2_g_share_negative'] = float((tau_g < 0).mean())
        log(f'  allelic {arm} two-component: c_a {tcomp["c"]:.3f} {tcomp["c_ci95"]}, '
            f'tau2_g median {np.median(tau_g):.4f} mean {tau_g.mean():.4f}')
        al[arm] = dict(
            two_component=tcomp,
            n_genes=int(len(P)), n_pairs=int(P.n_a.sum()),
            tau2_mom_quantiles={q: float(np.quantile(tau, q)) for q in (.1, .25, .5, .75, .9)},
            tau2_mom_mean=float(tau.mean()), tau2_mom_share_negative=float((tau < 0).mean()),
            tau2_dl_quantiles={q: float(np.quantile(P.tau2_dl, q)) for q in (.1, .25, .5, .75, .9)},
            tau2_dl_mean=float(P.tau2_dl.mean()),
            tau2_mom_draw_only_mean=float(P.tau2_mom_draw_only.mean()),
            tau2_over_medVa_quantiles={q: float(np.quantile(P.tau2_over_medVa, q)) for q in (.1, .25, .5, .75, .9)},
            var_over_meanVa_quantiles={q: float(np.quantile(P.var_over_meanVa, q)) for q in (.1, .25, .5, .75, .9)},
            tau2_trend=dict(tr, form='tau2(x) = exp(b0 + b1 x + b2 x^2), x = log2 median haplotype-informative '
                            'reads, log2-ratio squared units; fitted as var(a) = mean(Va) + tau2(x) by IRLS'),
            tau2_trend_at=dict((f'{2 ** k:g}_reads', float(eval_trend(tr, k))) for k in range(3, 15)),
            betabinomial_rho_equivalent_at=dict(
                (f'{2 ** k:g}_reads', float(eval_trend(tr, k) * LN2 ** 2 / 4)) for k in range(3, 15)),
            bins=bins, across_gene_spread=spread,
            gene_mean_a=dict(sd_across_genes=float(P.mean_a.std(ddof=1)),
                             mean_sampling_var=float((P.se_mean_a ** 2).mean()),
                             sd_net_of_sampling=float(np.sqrt(max(P.mean_a.var(ddof=1) - (P.se_mean_a ** 2).mean(), 0))),
                             share_abs_z_gt_3=float((np.abs(z_mean) > 3).mean()),
                             median=float(P.mean_a.median())),
            kurtosis_median=float(P['resid_kurtosis'].median()))
        log(f'  {arm}: {len(P)} genes, median tau2 {np.median(tau):.4f}, mean {tau.mean():.4f}, '
            f'share<0 {(tau < 0).mean():.3f}; trend {tr}')
        P['arm'] = arm
        pg_all.append(P)
    summ['allelic'] = al

    # ---- sensitivity: how much of the between-donor variance is point-
    # estimate error? The same estimators on the Gibbs draw-mean values
    # log2((mean yL + 1/2)/(mean yR + 1/2)) and log2(mean yT / L * 1e6 + 1),
    # same donors, same masks, same technical variance. NOT a pipeline value
    # (rule 1 takes values from the point estimates); a measurement of the
    # point estimator's own contribution.
    A_dm = np.log2((S['mL_draw'][sel] + KAPPA) / (S['mR_draw'][sel] + KAPPA))
    T_dm = np.log2(S['mT_draw'][sel] / eff_lib[None, :] * 1e6 + 1.0)
    sens = {}
    # Each value arm minus the shipped Va (Gibbs + counting term) and minus the
    # draw-only Gibbs variance; the counting term duplicates counting noise the
    # draws already carry, so the two subtractions bracket the technical part.
    # The point-minus-draw-mean gap in between-donor variance is also split by
    # the pairs whose smaller point-estimate haplotype is below 3 and below 10
    # reads: is it a near-zero-haplotype process the 0.5-read rule misses, or
    # diffuse point-estimator error?
    rows = []
    for j in range(G):
        m = inf_drop[j]
        k = int(m.sum())
        if k < MIN_NA:
            continue
        v, vd = Va[j, m], Va_draw[j, m]
        a_p, a_d = A[j, m], A_dm[j, m]
        gap = ((a_p - a_p.mean()) ** 2 - (a_d - a_d.mean()) ** 2) / (k - 1)
        mn = np.minimum(pL[j, m], pR[j, m])
        rows.append(dict(depth=float(np.median(depth[j, m])),
                         tau2_point=float(a_p.var(ddof=1) - v.mean()),
                         tau2_point_draw_only=float(a_p.var(ddof=1) - vd.mean()),
                         tau2_draw_mean=float(a_d.var(ddof=1) - v.mean()),
                         tau2_draw_mean_draw_only=float(a_d.var(ddof=1) - vd.mean()),
                         gap=float(gap.sum()), gap_min_lt3=float(gap[mn < 3].sum()),
                         gap_min_lt10=float(gap[mn < 10].sum()),
                         n_pairs=k, n_min_lt3=int((mn < 3).sum()), n_min_lt10=int((mn < 10).sum())))
    Sd = pd.DataFrame(rows)
    xs = np.log2(Sd.depth.to_numpy())
    arms = ('tau2_point', 'tau2_point_draw_only', 'tau2_draw_mean', 'tau2_draw_mean_draw_only')
    sb = []
    for lo, hi, lab in zip(dep_edges[:-1], dep_edges[1:], dep_labels):
        s_ = (xs >= lo) & (xs < hi)
        dd = (Sd.tau2_point - Sd.tau2_draw_mean)[s_]
        r_ = dict(bin=lab, n_genes=int(s_.sum()), n_pairs=int(Sd.n_pairs[s_].sum()),
                  paired_diff_mean=float(dd.mean()),
                  paired_diff_se=float(dd.std(ddof=1) / np.sqrt(s_.sum())),
                  gap_share_min_lt3=float(Sd.gap_min_lt3[s_].sum() / Sd.gap[s_].sum()),
                  gap_share_min_lt10=float(Sd.gap_min_lt10[s_].sum() / Sd.gap[s_].sum()),
                  pair_share_min_lt3=float(Sd.n_min_lt3[s_].sum() / Sd.n_pairs[s_].sum()),
                  pair_share_min_lt10=float(Sd.n_min_lt10[s_].sum() / Sd.n_pairs[s_].sum()))
        for arm_ in arms:
            r_[arm_ + '_mean'] = float(Sd[arm_][s_].mean())
            r_[arm_ + '_median'] = float(Sd[arm_][s_].median())
        sb.append(r_)
    sens['allelic_drop'] = dict(n_genes=int(len(Sd)), n_pairs=int(Sd.n_pairs.sum()), bins=sb,
                                gap_share_min_lt3=float(Sd.gap_min_lt3.sum() / Sd.gap.sum()),
                                gap_share_min_lt10=float(Sd.gap_min_lt10.sum() / Sd.gap.sum()),
                                **{a_ + '_mean': float(Sd[a_].mean()) for a_ in arms},
                                **{a_ + '_median': float(Sd[a_].median()) for a_ in arms})
    for dname in ('full', 'meta_only'):
        Zd = design(cov, designs[dname])
        df = N - Zd.shape[1]
        Ed, hd = ols_resid(T_dm, Zd)
        s2d = (Ed ** 2).sum(1) / df
        techd = (Vt * (1 - hd)[None, :]).sum(1) / df
        biod = s2d - techd
        biop = pg_total[f'bio_{dname}'].to_numpy()
        tb = []
        for lo, hi, lab in zip(cpm_edges[:-1], cpm_edges[1:], cpm_labels):
            s_ = (x_cpm >= lo) & (x_cpm < hi)
            dd = (biop - biod)[s_]
            tb.append(dict(bin=lab, n_genes=int(s_.sum()), bio_point_mean=float(biop[s_].mean()),
                           bio_draw_mean_mean=float(biod[s_].mean()), paired_diff_mean=float(dd.mean()),
                           paired_diff_se=float(dd.std(ddof=1) / np.sqrt(s_.sum()))))
        sens[f'total_{dname}'] = dict(bio_point_mean=float(biop.mean()), bio_draw_mean_mean=float(biod.mean()),
                                      bins=tb)
    summ['point_estimate_error_sensitivity'] = sens
    log(f'  point vs draw-mean values: allelic tau2 mean {Sd.tau2_point.mean():.4f} vs '
        f'{Sd.tau2_draw_mean.mean():.4f}; total bio (full) {sens["total_full"]["bio_point_mean"]:.4f} vs '
        f'{sens["total_full"]["bio_draw_mean_mean"]:.4f}')
    pd.concat(pg_all).to_csv(OUT / 'per_gene_allelic.tsv', sep='\t', index=False)

    # =====================================================================
    #  2. donor variance component
    # =====================================================================
    log('measurement 2: donor variance component')
    # gate: closed-form normal equations against the explicit design matrix,
    # on 300 genes of the total channel, observed and one permuted labelling
    E_, h_ = resid_store['full']
    ng = min(300, G)
    yg = np.log((E_[:ng] ** 2) / (1 - h_)[None, :]).ravel()
    gg = np.repeat(np.arange(ng), N)
    dg = np.tile(np.arange(N), ng)
    Fg = prepare_fit(yg, np.log(Vt[:ng]).ravel(), gg, ng, dg, N)
    gate_rng = np.random.default_rng(ss[4])
    gmax = 0.0
    for dlab in (dg, permute_within_gene(dg, gg, gate_rng)):
        d1, r1, _ = donor_fit(Fg, dlab, N)
        d2_, r2_ = donor_fit_dense(Fg, dlab, N)
        gmax = max(gmax, float(np.max(np.abs(d1 - d2_))), abs(r1 - r2_) / r2_)
    if gmax > 1e-8:
        raise SystemExit(f'closed-form donor fit disagrees with the dense fit: {gmax:.2e}')
    summ['gate_donor_fit_max_abs_diff'] = gmax
    log(f'  gate: closed-form donor fit = dense fit to {gmax:.1e}')
    don = pd.DataFrame(index=samples)
    don['rin'] = cov['rin'].to_numpy()
    don['age_days'] = cov['age_days'].to_numpy()
    don['sex'] = cov['sex'].to_numpy()
    don['log_eff_lib'] = np.log(eff_lib)
    lvT = np.log(Vt)
    lvT_c = lvT - lvT.mean(1, keepdims=True)
    don['mean_logVt_centered'] = lvT_c.mean(0)
    d2 = {}
    gene_idx = np.repeat(np.arange(G), N)
    donor_idx = np.tile(np.arange(N), G)
    corr_names = ['rin', 'age_days', 'sex', 'log_eff_lib', 'mean_logV_centered']
    for dname in designs:
        E, h = resid_store[dname]
        y = np.log((E ** 2) / (1 - h)[None, :]).ravel()
        lv = lvT.ravel()
        res, d_obs, d_nospl = donor_component(y, lv, gene_idx, donor_idx, G, N, rng_donor,
                                              f'total/{dname}')
        don[f'delta_total_{dname}'] = d_obs
        don[f'delta_total_{dname}_nospline'] = d_nospl
        ex = dict(mean_logV_centered=don['mean_logVt_centered'].to_numpy(),
                  log_eff_lib=don['log_eff_lib'].to_numpy())
        res['correlations'] = corr_block(d_obs, don, ex, corr_names)
        k221 = np.array([s != '221_D1' for s in samples])
        res['correlations_without_221_D1'] = corr_block(
            d_obs[k221], don[k221], {kk: vv[k221] for kk, vv in ex.items()}, corr_names)
        order = np.argsort(-d_obs)
        res['top5'] = [(samples[i], float(d_obs[i]), float(np.exp(d_obs[i])), float(don.rin.iloc[i])) for i in order[:5]]
        res['bottom5'] = [(samples[i], float(d_obs[i]), float(np.exp(d_obs[i])), float(don.rin.iloc[i])) for i in order[-5:]]
        res['named'] = {s: float(d_obs[samples.index(s)]) for s in NAMED}
        res['multiplier_q05_q50_q95'] = [float(np.exp(q)) for q in np.quantile(d_obs, [.05, .5, .95])]
        d2[f'total_{dname}'] = res
    # by expression tercile (full design): does the donor effect act on the
    # biological part (high CPM) or the technical part (low CPM)?
    E, h = resid_store['full']
    terc = np.quantile(x_cpm, [1 / 3, 2 / 3])
    tb = np.digitize(x_cpm, terc)
    tr_out = {}
    for t in range(3):
        gsel = np.where(tb == t)[0]
        y = np.log((E[gsel] ** 2) / (1 - h)[None, :]).ravel()
        lv = lvT[gsel].ravel()
        gi_ = np.repeat(np.arange(len(gsel)), N)
        di_ = np.tile(np.arange(N), len(gsel))
        d_t, _, _ = donor_fit(prepare_fit(y, lv, gi_, len(gsel), di_, N), di_, N)
        don[f'delta_total_full_tercile{t}'] = d_t
        tr_out[f'tercile{t}'] = dict(n_genes=int(len(gsel)),
                                     cpm_range=[float(med_cpm[gsel].min()), float(med_cpm[gsel].max())],
                                     sd_delta=float(np.std(d_t, ddof=1)),
                                     spearman_rin=float(sps.spearmanr(d_t, don.rin)[0]),
                                     delta_221_D1=float(d_t[samples.index('221_D1')]))
    cc = np.corrcoef(np.vstack([don[f'delta_total_full_tercile{t}'] for t in range(3)]))
    tr_out['pearson_between_terciles'] = dict(t0_t1=float(cc[0, 1]), t0_t2=float(cc[0, 2]), t1_t2=float(cc[1, 2]))
    d2['total_full_by_cpm_tercile'] = tr_out

    # allelic: centred a, weighted mean, zero-haplotype pairs dropped
    rows_y, rows_lv, rows_g, rows_d = [], [], [], []
    gk = 0
    lvA_sum = np.zeros(N)
    lvA_cnt = np.zeros(N)
    for j in range(G):
        m = inf_drop[j]
        if m.sum() < MIN_NA:
            continue
        a, v = A[j, m], Va[j, m]
        w = 1.0 / v
        aw = (w * a).sum() / w.sum()
        hh = w / w.sum()
        e2 = (a - aw) ** 2 / (1 - hh)
        di = np.where(m)[0]
        rows_y.append(np.log(e2))
        lvv = np.log(v)
        rows_lv.append(lvv)
        rows_g.append(np.full(len(di), gk))
        rows_d.append(di)
        lvA_sum[di] += lvv - lvv.mean()
        lvA_cnt[di] += 1
        gk += 1
    y = np.concatenate(rows_y)
    lv = np.concatenate(rows_lv)
    gi_ = np.concatenate(rows_g)
    di_ = np.concatenate(rows_d)
    if not np.all(np.isfinite(y)):
        raise SystemExit('non-finite log e^2 in the allelic channel')
    don['mean_logVa_centered'] = lvA_sum / np.maximum(lvA_cnt, 1)
    don['n_allelic_genes'] = lvA_cnt
    res, d_obs, d_nospl = donor_component(y, lv, gi_, di_, gk, N, rng_donor, 'allelic/drop')
    don['delta_allelic'] = d_obs
    don['delta_allelic_nospline'] = d_nospl
    ex = dict(mean_logV_centered=don['mean_logVa_centered'].to_numpy(),
              log_eff_lib=don['log_eff_lib'].to_numpy())
    res['correlations'] = corr_block(d_obs, don, ex, corr_names)
    k221 = np.array([s != '221_D1' for s in samples])
    res['correlations_without_221_D1'] = corr_block(
        d_obs[k221], don[k221], {kk: vv[k221] for kk, vv in ex.items()}, corr_names)
    order = np.argsort(-d_obs)
    res['top5'] = [(samples[i], float(d_obs[i]), float(np.exp(d_obs[i])), float(don.rin.iloc[i])) for i in order[:5]]
    res['bottom5'] = [(samples[i], float(d_obs[i]), float(np.exp(d_obs[i])), float(don.rin.iloc[i])) for i in order[-5:]]
    res['named'] = {s: float(d_obs[samples.index(s)]) for s in NAMED}
    res['multiplier_q05_q50_q95'] = [float(np.exp(q)) for q in np.quantile(d_obs, [.05, .5, .95])]
    res['pearson_with_total_full'] = float(np.corrcoef(d_obs, don['delta_total_full'])[0, 1])
    d2['allelic_drop'] = res
    # the same with the spline in log haplotype-informative depth, which, unlike
    # log Va, is not a function of the ratio itself
    ldep = np.concatenate([np.log(depth[j, inf_drop[j]]) for j in range(G)
                           if inf_drop[j].sum() >= MIN_NA])
    res2, d_dep, _ = donor_component(y, ldep, gi_, di_, gk, N, rng_donor, 'allelic/drop, depth spline')
    don['delta_allelic_depth_spline'] = d_dep
    res2['correlations'] = corr_block(d_dep, don, ex, corr_names)
    order = np.argsort(-d_dep)
    res2['top5'] = [(samples[i], float(d_dep[i]), float(np.exp(d_dep[i])), float(don.rin.iloc[i]))
                    for i in order[:5]]
    res2['bottom5'] = [(samples[i], float(d_dep[i]), float(np.exp(d_dep[i])), float(don.rin.iloc[i]))
                       for i in order[-5:]]
    res2['named'] = {s: float(d_dep[samples.index(s)]) for s in NAMED}
    res2['multiplier_q05_q50_q95'] = [float(np.exp(q)) for q in np.quantile(d_dep, [.05, .5, .95])]
    res2['pearson_with_total_full'] = float(np.corrcoef(d_dep, don['delta_total_full'])[0, 1])
    res2['pearson_with_total_meta_only'] = float(np.corrcoef(d_dep, don['delta_total_meta_only'])[0, 1])
    d2['allelic_drop_depth_spline'] = res2
    summ['donor_component'] = d2

    # ---- reproduction (a): mean standardized whitened squared total residual
    log('reproduction: per-donor mean standardized z^2 (221_D1)')
    Zf = design(cov, designs['full'])

    def z2_std(Tm, Vm, weighted=True):
        """Gene-standardized leverage-corrected whitened squared residual
        (batched QR: one [n x p] factorization per gene)."""
        out = np.empty_like(Tm)
        for s0 in range(0, Tm.shape[0], 2000):
            sl = slice(s0, s0 + 2000)
            sw = 1.0 / np.sqrt(Vm[sl]) if weighted else np.ones_like(Vm[sl])
            Q, _ = np.linalg.qr(Zf[None, :, :] * sw[:, :, None])
            y_ = sw * Tm[sl]
            r = y_ - np.einsum('gnp,gp->gn', Q, np.einsum('gnp,gn->gp', Q, y_))
            hh = (Q ** 2).sum(2)
            z2 = r ** 2 / (1 - hh)
            out[sl] = z2 / z2.mean(1, keepdims=True)
        return out

    gi_all = {g: i for i, g in enumerate(all_genes)}
    i46 = np.array([gi_all[g] for g in genes46 if g in gi_all])
    i45 = np.array([gi_all[g] for g in genes46 if g in gi_all and g not in extra])
    repro = {}
    for lab, idx, nsim in (('instrument46', i46, N_SIM_Z2_46), ('instrument45_calibration', i45, N_SIM_Z2_46),
                           ('all_calibration', np.arange(G), N_SIM_Z2_ALL)):
        if len(idx) == 0:
            continue
        Tm, Vm = S['T'][idx], S['Vt'][idx]
        for wlab, wflag in (('gibbs_weighted', True), ('unit_weights', False)):
            zz = z2_std(Tm, Vm, wflag)
            dm = zz.mean(0)
            sims = []
            for _ in range(nsim):
                Ts = np.sqrt(Vm) * rng_sim.standard_normal(Vm.shape)
                sims.append(z2_std(Ts, Vm, wflag).mean(0))
            sims = np.array(sims)
            mx = sims.max(1)
            key = f'{lab}_{wlab}'
            repro[key] = dict(
                n_genes=int(len(idx)), n_model_sims=int(nsim),
                donor_221_D1=float(dm[samples.index('221_D1')]),
                donor_587_D1=float(dm[samples.index('587_D1')]),
                max_donor=samples[int(np.argmax(dm))], max_value=float(dm.max()),
                top5=[(samples[i], float(dm[i])) for i in np.argsort(-dm)[:5]],
                sd_across_donors=float(dm.std(ddof=1)),
                model_max_q95=float(np.quantile(mx, .95)), model_max_max=float(mx.max()),
                model_sd_across_donors_mean=float(sims.std(1, ddof=1).mean()),
                n_donors_above_model_max_q95=int((dm > np.quantile(mx, .95)).sum()))
            if lab == 'all_calibration':
                don[f'mean_z2_std_{wlab}'] = dm
            elif lab == 'instrument46':
                don[f'mean_z2_std_46_{wlab}'] = dm
            log(f'  {key}: 221_D1 {repro[key]["donor_221_D1"]:.3f}, max {repro[key]["max_donor"]} '
                f'{repro[key]["max_value"]:.3f}, model max q95 {repro[key]["model_max_q95"]:.3f}')
    summ['reproduce_221_D1'] = repro

    # ---- reproduction (b): sd of per-donor mean of read-count-adjusted log v
    log('reproduction: donor main effect of read-count-adjusted log v (0.073)')

    def logv_donor_sd(Vm, Mm, dep, rng=None):
        Rm = np.full(Vm.shape, np.nan)
        within, resid, r2 = [], [], []
        for j in range(Vm.shape[0]):
            m = Mm[j]
            if m.sum() < MIN_INF_LOGV:
                continue
            lvv = np.log(Vm[j, m])
            ld = np.log(dep[j, m])
            if lvv.std() == 0 or ld.std() == 0:
                continue
            b = np.polyfit(ld, lvv, 1)
            e = lvv - np.polyval(b, ld)
            if rng is not None:
                e = e[rng.permutation(len(e))]
            within.append(lvv.std())
            resid.append(e.std())
            r2.append(1 - e.var() / lvv.var())
            Rm[j, np.where(m)[0]] = e
        ok = ~np.all(np.isnan(Rm), axis=1)
        dmean = np.nanmean(Rm[ok], axis=0)
        return dict(n_genes=int(ok.sum()), within_sd=float(np.median(within)),
                    residual_sd_at_matched_depth=float(np.median(resid)),
                    r2_depth=float(np.median(r2)),
                    donor_main_effect_sd=float(np.nanstd(dmean))), dmean

    lvrep = {}
    for vlab, Vm, pos in (('shipped_Va', Va, Va > EPS), ('draw_only_Va', Va_draw, (Va > EPS) & (Va_draw > EPS))):
        for zlab, M in (('zeros_kept', pos), ('zeros_dropped', pos & ~zero)):
            r, dmean = logv_donor_sd(Vm, M, depth)
            fl = [logv_donor_sd(Vm, M, depth, rng=rng_lv)[0]['donor_main_effect_sd'] for _ in range(5)]
            r['perm_floor_sd_mean'] = float(np.mean(fl))
            r['perm_floor_sd_range'] = [float(min(fl)), float(max(fl))]
            lvrep[f'allelic_{vlab}_{zlab}'] = r
            if vlab == 'shipped_Va' and zlab == 'zeros_dropped':
                don['logVa_depth_adjusted_donor_mean'] = dmean
            log(f'  allelic {vlab} {zlab}: {r}')
    r, dmean = logv_donor_sd(Vt, Vt > EPS, pT + 0.5)
    fl = [logv_donor_sd(Vt, Vt > EPS, pT + 0.5, rng=rng_lv)[0]['donor_main_effect_sd'] for _ in range(5)]
    r['perm_floor_sd_mean'] = float(np.mean(fl))
    r['perm_floor_sd_range'] = [float(min(fl)), float(max(fl))]
    lvrep['total_shipped_Vt_on_count'] = r
    don['logVt_count_adjusted_donor_mean'] = dmean
    log(f'  total: {r}')
    summ['reproduce_logv_donor_sd'] = lvrep

    # =====================================================================
    #  3. ancestry
    # =====================================================================
    log('measurement 3: ancestry')
    Zr = design(cov, META + ecols)
    Er, _ = ols_resid(T, Zr)
    Gm = cov[gcols].to_numpy(float)

    def partial_r2(Gmat):
        Gr = Gmat - Zr @ np.linalg.lstsq(Zr, Gmat, rcond=None)[0]
        Qg, _ = np.linalg.qr(Gr)
        proj = Er @ Qg
        ss_g = (proj ** 2).sum(1)
        ss = (Er ** 2).sum(1)
        return ss_g / ss, Gr

    r2, Gr = partial_r2(Gm)
    df_r = N - Zr.shape[1]           # 77
    df_f = df_r - 3                  # 74
    Fst = (r2 / 3) / ((1 - r2) / df_f)
    pval = sps.f.sf(Fst, 3, df_f)
    r2_adj = 1 - (1 - r2) * df_r / df_f
    per_pc = {}
    for k, c in enumerate(gcols):
        gk_ = Gr[:, k] / np.linalg.norm(Gr[:, k])
        per_pc[c] = dict(mean_partial_r2=float((((Er @ gk_) ** 2) / (Er ** 2).sum(1)).mean()))
    null_means, null_q = [], []
    for _ in range(N_PERM_ANC):
        prm = rng_anc.permutation(N)
        r2p, _ = partial_r2(Gm[prm])
        null_means.append(float(r2p.mean()))
        null_q.append(np.quantile(r2p, [.5, .9, .95, .99]))
    null_q = np.array(null_q).mean(0)
    pi0 = float((pval > 0.5).mean() / 0.5)
    ss_res = (Er ** 2).sum(1) / df_r
    anc_var = r2_adj * ss_res
    bio_nopc = pg_total['bio_no_expr_pcs'].to_numpy()
    bio_full = pg_total['bio_full'].to_numpy()
    sel_bio = bio_full > 0
    summ['ancestry'] = dict(
        n_genes=G, df_after_rna_tied=int(df_r), df_after_gpcs=int(df_f),
        analytic_null_mean=float(3 / df_r),
        perm_null_mean=float(np.mean(null_means)), perm_null_mean_sd=float(np.std(null_means, ddof=1)),
        perm_null_quantiles_q50_q90_q95_q99=[float(v) for v in null_q],
        observed_mean=float(r2.mean()),
        observed_quantiles={q: float(np.quantile(r2, q)) for q in (.1, .25, .5, .75, .9, .95, .99)},
        excess_mean_over_perm=float(r2.mean() - np.mean(null_means)),
        excess_mean_over_perm_in_null_sds=float((r2.mean() - np.mean(null_means)) / np.std(null_means, ddof=1)),
        adjusted_r2_quantiles={q: float(np.quantile(r2_adj, q)) for q in (.1, .25, .5, .75, .9, .95, .99)},
        adjusted_r2_mean=float(r2_adj.mean()),
        share_p_lt_0_05=float((pval < 0.05).mean()), share_p_lt_0_001=float((pval < 0.001).mean()),
        n_p_lt_0_05=int((pval < 0.05).sum()),
        storey_pi0_lambda_0_5=pi0, share_non_null_1_minus_pi0=float(1 - pi0),
        per_pc=per_pc,
        ancestry_var_over_bio_full_median_where_bio_pos=float(np.median(anc_var[sel_bio] / bio_full[sel_bio])),
        ancestry_var_over_bio_full_mean_ratio=float(anc_var.mean() / bio_full.mean()),
        ancestry_var_over_bio_no_expr_pcs_mean_ratio=float(anc_var.mean() / bio_nopc.mean()),
        mean_adjusted_r2_among_p_lt_0_001=float(r2_adj[pval < 0.001].mean()) if (pval < 0.001).any() else None,
        mean_adjusted_r2_among_p_lt_0_05=float(r2_adj[pval < 0.05].mean()),
    )
    by_cpm = []
    for lo, hi, lab in zip(cpm_edges[:-1], cpm_edges[1:], cpm_labels):
        s = (x_cpm >= lo) & (x_cpm < hi)
        if s.sum():
            by_cpm.append(dict(bin=lab, n_genes=int(s.sum()), mean_r2=float(r2[s].mean()),
                               mean_adj_r2=float(r2_adj[s].mean()),
                               share_p_lt_0_05=float((pval[s] < 0.05).mean())))
    summ['ancestry']['by_cpm'] = by_cpm

    # ---- what the three genotype PCs are: ancestry (PC1) and two donors (PC2-3)
    anc = summ['ancestry']
    zg = (Gm - Gm.mean(0)) / Gm.std(0, ddof=1)
    ipair = [samples.index(d) for d in PAIR]
    anc['genotype_pc_structure'] = {
        c: dict(kurtosis=float((zg[:, k] ** 4).mean()),
                z_265_D1=float(zg[ipair[0], k]), z_416_D1=float(zg[ipair[1], k]),
                share_of_sum_sq_in_pair=float((zg[ipair, k] ** 2).sum() / (zg[:, k] ** 2).sum()))
        for k, c in enumerate(gcols)}
    for k, c in enumerate(gcols):
        anc['per_pc'][c]['null_mean_one_pc'] = float(1 / df_r)
    # gPC1 alone (the ancestry axis), with a permutation null
    r2_1, _ = partial_r2(Gm[:, :1])
    p1 = sps.f.sf((r2_1 / 1) / ((1 - r2_1) / (df_r - 1)), 1, df_r - 1)
    nm1 = []
    for _ in range(N_PERM_ANC):
        prm = rng_anc.permutation(N)
        nm1.append(float(partial_r2(Gm[prm, :1])[0].mean()))
    anc['gpc1_only'] = dict(
        mean_partial_r2=float(r2_1.mean()), perm_null_mean=float(np.mean(nm1)),
        perm_null_sd=float(np.std(nm1, ddof=1)), analytic_null_mean=float(1 / df_r),
        excess_mean=float(r2_1.mean() - np.mean(nm1)),
        adjusted_r2_quantiles={q: float(np.quantile(1 - (1 - r2_1) * df_r / (df_r - 1), q))
                               for q in (.5, .75, .9, .95, .99)},
        share_p_lt_0_05=float((p1 < 0.05).mean()), storey_pi0=float((p1 > 0.5).mean() / 0.5))
    # share of the RNA-tied residual variance carried by the two donors
    Qr, _ = np.linalg.qr(Zr)
    hr = (Qr ** 2).sum(1)
    pair_share = (Er[:, ipair] ** 2).sum(1) / (Er ** 2).sum(1)
    anc['pair_265_416'] = dict(
        mean_share_of_rna_tied_residual=float(pair_share.mean()),
        null_share=float((1 - hr[ipair]).sum() / df_r),
        quantiles_q50_q90_q99=[float(q) for q in np.quantile(pair_share, [.5, .9, .99])],
        mean_std_residual_sq={d: float(((Er[:, i] ** 2 / (1 - hr[i])) /
                                        ((Er ** 2).sum(1) / df_r)).mean()) for d, i in zip(PAIR, ipair)},
        corr_of_their_residuals_across_genes=float(np.corrcoef(Er[:, ipair[0]], Er[:, ipair[1]])[0, 1]))
    # sensitivity: both donors removed
    keepd = np.array([s not in PAIR for s in samples])
    Zr2 = Zr[keepd]
    Er2, _ = ols_resid(T[:, keepd], Zr2)
    df_r2 = int(keepd.sum() - Zr2.shape[1])

    def pr2_sub(Gmat):
        Gr_ = Gmat - Zr2 @ np.linalg.lstsq(Zr2, Gmat, rcond=None)[0]
        Qg_, _ = np.linalg.qr(Gr_)
        return ((Er2 @ Qg_) ** 2).sum(1) / (Er2 ** 2).sum(1)
    r2_d3 = pr2_sub(Gm[keepd])
    r2_d1 = pr2_sub(Gm[keepd][:, :1])
    nm3, nmd1 = [], []
    for _ in range(N_PERM_ANC):
        prm = rng_anc.permutation(int(keepd.sum()))
        nm3.append(float(pr2_sub(Gm[keepd][prm]).mean()))
        nmd1.append(float(pr2_sub(Gm[keepd][prm, :1]).mean()))
    anc['without_265_416'] = dict(
        n_donors=int(keepd.sum()), df_after_rna_tied=df_r2,
        three_pcs=dict(mean_partial_r2=float(r2_d3.mean()), perm_null_mean=float(np.mean(nm3)),
                       excess_mean=float(r2_d3.mean() - np.mean(nm3)),
                       adjusted_r2_mean=float((1 - (1 - r2_d3) * df_r2 / (df_r2 - 3)).mean())),
        gpc1_only=dict(mean_partial_r2=float(r2_d1.mean()), perm_null_mean=float(np.mean(nmd1)),
                       excess_mean=float(r2_d1.mean() - np.mean(nmd1)),
                       adjusted_r2_quantiles={q: float(np.quantile(1 - (1 - r2_d1) * df_r2 / (df_r2 - 1), q))
                                              for q in (.5, .75, .9, .95, .99)}))
    # ---- is the genotype-PC signal a broad expression axis? Expression PCs
    # computed WITHOUT residualizing on the genotype PCs (log2(CPM+1) of these
    # genes, centred, residualized on intercept + metadata, top 10 left
    # singular vectors), their canonical correlations with the genotype-PC
    # block, and the genotype-PC share after them: a lower bound on the
    # gene-specific ancestry share, because it also removes any real ancestry
    # effect that is shared across many genes.
    Zm = design(cov, META)
    Em, _ = ols_resid(T - T.mean(1, keepdims=True), Zm)
    Uu, Sv, _ = np.linalg.svd(Em.T, full_matrices=False)
    U10 = Uu[:, :10]
    Gm_res = Gm - Zm @ np.linalg.lstsq(Zm, Gm, rcond=None)[0]
    Qg0, _ = np.linalg.qr(Gm_res)
    cancor = np.linalg.svd(Qg0.T @ U10, compute_uv=False)
    cor_pc = np.corrcoef(np.hstack([Gm_res, U10]).T)[:3, 3:]
    Zu = np.column_stack([Zm, U10])
    Eu, _ = ols_resid(T, Zu)
    df_u = N - Zu.shape[1]

    def pr2_u(Gmat, E_, Z_):
        Gr_ = Gmat - Z_ @ np.linalg.lstsq(Z_, Gmat, rcond=None)[0]
        Qg_, _ = np.linalg.qr(Gr_)
        return ((E_ @ Qg_) ** 2).sum(1) / (E_ ** 2).sum(1)
    r2_u3 = pr2_u(Gm, Eu, Zu)
    r2_u1 = pr2_u(Gm[:, :1], Eu, Zu)
    Eu2, _ = ols_resid(T[:, keepd], Zu[keepd])
    r2_u1d = pr2_u(Gm[keepd][:, :1], Eu2, Zu[keepd])
    nu3, nu1, nu1d = [], [], []
    for _ in range(N_PERM_ANC):
        prm = rng_anc.permutation(N)
        nu3.append(float(pr2_u(Gm[prm], Eu, Zu).mean()))
        nu1.append(float(pr2_u(Gm[prm, :1], Eu, Zu).mean()))
        prm2 = rng_anc.permutation(int(keepd.sum()))
        nu1d.append(float(pr2_u(Gm[keepd][prm2, :1], Eu2, Zu[keepd]).mean()))
    anc['unrestricted_expression_pcs'] = dict(
        construction='log2(CPM+1) of the 11,747 genes, centred, residualized on intercept + metadata '
                     '(NOT on genotype PCs), top 10 left singular vectors',
        share_of_meta_residual_variance_top10=[float(v) for v in (Sv[:10] ** 2 / (Sv ** 2).sum())],
        canonical_correlations_gpc_block_vs_top10=[float(c) for c in cancor],
        corr_gpc_vs_pc_max_abs={c: dict(pc=int(np.argmax(np.abs(cor_pc[k]))) + 1,
                                        r=float(cor_pc[k, np.argmax(np.abs(cor_pc[k]))]))
                                for k, c in enumerate(gcols)},
        corr_gpc1_vs_pcs=[float(v) for v in cor_pc[0]],
        df_after=int(df_u),
        three_pcs_after=dict(mean_partial_r2=float(r2_u3.mean()), perm_null_mean=float(np.mean(nu3)),
                             excess_mean=float(r2_u3.mean() - np.mean(nu3))),
        gpc1_after=dict(mean_partial_r2=float(r2_u1.mean()), perm_null_mean=float(np.mean(nu1)),
                        excess_mean=float(r2_u1.mean() - np.mean(nu1))),
        gpc1_after_without_265_416=dict(mean_partial_r2=float(r2_u1d.mean()),
                                        perm_null_mean=float(np.mean(nu1d)),
                                        excess_mean=float(r2_u1d.mean() - np.mean(nu1d))))
    # canonical-correlation null: permuted genotype-PC rows
    cc_null = []
    for _ in range(N_PERM_ANC):
        prm = rng_anc.permutation(N)
        Gp_ = Gm[prm] - Zm @ np.linalg.lstsq(Zm, Gm[prm], rcond=None)[0]
        Qp_, _ = np.linalg.qr(Gp_)
        cc_null.append(np.linalg.svd(Qp_.T @ U10, compute_uv=False))
    cc_null = np.array(cc_null)
    ue = anc['unrestricted_expression_pcs']
    ue['canonical_correlation_perm_null_mean'] = [float(v) for v in cc_null.mean(0)]
    ue['canonical_correlation_perm_null_q95'] = [float(v) for v in np.quantile(cc_null, .95, axis=0)]
    ue['share_perm_first_cancor_ge_observed'] = float((cc_null[:, 0] >= cancor[0]).mean())

    # The pipeline's expression PCs were built on expression residualized on
    # the REAL genotype PCs, so a row permutation of the genotype PCs alone is
    # not the construction's null. Rebuilt null: permute the genotype-PC rows,
    # rebuild 10 expression PCs from expression residualized on intercept +
    # metadata + the PERMUTED genotype PCs, and compute the permuted genotype
    # PCs' partial R^2 after intercept + metadata + the rebuilt PCs.
    Tc = T - T.mean(1, keepdims=True)

    def rebuilt_r2(Gp, dsel=None):
        Tt = Tc if dsel is None else Tc[:, dsel]
        Tv = T if dsel is None else T[:, dsel]
        Zm_ = Zm if dsel is None else Zm[dsel]
        Yr, _ = ols_resid(Tt, np.column_stack([Zm_, Gp]))
        U_, _, _ = np.linalg.svd(Yr.T, full_matrices=False)
        Zr_ = np.column_stack([Zm_, U_[:, :10]])
        Er_, _ = ols_resid(Tv, Zr_)
        return float(pr2_u(Gp, Er_, Zr_).mean()), float(pr2_u(Gp[:, :1], Er_, Zr_).mean())
    obs3, obs1 = rebuilt_r2(Gm)
    obs3d, obs1d = rebuilt_r2(Gm[keepd], keepd)
    rb3, rb1, rb3d, rb1d = [], [], [], []
    for _ in range(N_PERM_ANC):
        prm = rng_anc.permutation(N)
        a3, a1 = rebuilt_r2(Gm[prm])
        rb3.append(a3)
        rb1.append(a1)
        prm2 = rng_anc.permutation(int(keepd.sum()))
        b3, b1 = rebuilt_r2(Gm[keepd][prm2], keepd)
        rb3d.append(b3)
        rb1d.append(b1)
    anc['rebuilt_pc_null'] = dict(
        construction='expression PCs rebuilt on these 11,747 genes after residualizing on intercept + '
                     'metadata + the (permuted) genotype PCs, as build_covariates.py does with the real ones',
        observed_three_pcs=obs3, observed_gpc1=obs1,
        null_three_pcs_mean=float(np.mean(rb3)), null_three_pcs_sd=float(np.std(rb3, ddof=1)),
        null_three_pcs_q95=float(np.quantile(rb3, .95)),
        null_gpc1_mean=float(np.mean(rb1)), null_gpc1_sd=float(np.std(rb1, ddof=1)),
        null_gpc1_q95=float(np.quantile(rb1, .95)),
        excess_three_pcs=float(obs3 - np.mean(rb3)), excess_gpc1=float(obs1 - np.mean(rb1)),
        share_null_ge_observed_gpc1=float((np.array(rb1) >= obs1).mean()),
        without_265_416=dict(observed_three_pcs=obs3d, observed_gpc1=obs1d,
                             null_three_pcs_mean=float(np.mean(rb3d)), null_gpc1_mean=float(np.mean(rb1d)),
                             null_gpc1_sd=float(np.std(rb1d, ddof=1)),
                             excess_three_pcs=float(obs3d - np.mean(rb3d)),
                             excess_gpc1=float(obs1d - np.mean(rb1d)),
                             share_null_ge_observed_gpc1=float((np.array(rb1d) >= obs1d).mean())))
    log(f'  rebuilt-PC null: 3 PCs obs {obs3:.4f} vs {np.mean(rb3):.4f} (sd {np.std(rb3, ddof=1):.4f}); '
        f'gPC1 obs {obs1:.4f} vs {np.mean(rb1):.4f} (sd {np.std(rb1, ddof=1):.4f}); without pair gPC1 '
        f'{obs1d:.4f} vs {np.mean(rb1d):.4f}')
    log(f'  unrestricted PCs: cancor {np.round(cancor, 3)}; gPC1 after them {r2_u1.mean():.4f} '
        f'vs {np.mean(nu1):.4f}; 3 PCs {r2_u3.mean():.4f} vs {np.mean(nu3):.4f}')
    log(f'  gPC1 only: mean partial R2 {r2_1.mean():.4f} vs perm {np.mean(nm1):.4f}; '
        f'pair share {pair_share.mean():.4f} vs null {anc["pair_265_416"]["null_share"]:.4f}; '
        f'without pair: 3 PCs {r2_d3.mean():.4f} vs {np.mean(nm3):.4f}, gPC1 {r2_d1.mean():.4f} vs {np.mean(nmd1):.4f}')
    log(f'  ancestry: mean partial R2 {r2.mean():.4f} vs perm {np.mean(null_means):.4f} '
        f'(analytic {3 / df_r:.4f}); pi0 {pi0:.3f}; p<0.05 {(pval < 0.05).mean():.3f}')
    pd.DataFrame(dict(gene=genes, median_cpm=med_cpm, partial_r2=r2, adjusted_r2=r2_adj,
                      F=Fst, p=pval, resid_var_rna_tied=ss_res,
                      ancestry_var=anc_var, partial_r2_gpc1_only=r2_1,
                      share_rna_tied_resid_265_416=pair_share,
                      partial_r2_3pcs_without_265_416=r2_d3,
                      partial_r2_gpc1_without_265_416=r2_d1,
                      partial_r2_gpc1_after_unrestricted_pcs=r2_u1)).to_csv(OUT / 'per_gene_ancestry.tsv', sep='\t', index=False)

    pg_total.to_csv(OUT / 'per_gene_total.tsv', sep='\t', index=False)
    don.index.name = 'donor'
    don.to_csv(OUT / 'donors.tsv', sep='\t')
    summ['runtime_s'] = float(time.time() - T0)
    fn = 'summary.json' if args.limit is None else f'summary_limit{args.limit}.json'
    (OUT / fn).write_text(json.dumps(summ, indent=1, default=float))
    log(f'wrote {OUT / fn}')
    if args.limit is None:
        figures(summ, pg_total, pd.concat(pg_all), don, r2, null_means, x_cpm)
    log('done')


# ---------------------------------------------------------------------------
#  figures
# ---------------------------------------------------------------------------

def figures(summ, pg_total, pg_all, don, r2, null_means, x_cpm):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    G = summ['n_genes']
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    for dname, col, lab in (('full', '#1f5fa8', 'all 17 covariates'),
                            ('no_expr_pcs', '#c0504d', 'no expression PCs'),
                            ('meta_only', '#2e8b57', 'metadata only')):
        s2 = pg_total[f's2_{dname}'].to_numpy()
        tech = pg_total[f'tech_{dname}'].to_numpy()
        bio = s2 - tech
        tr = summ['total'][dname]['bio_trend']
        edges = np.arange(0, 12.5, 0.5)
        cen, mb, se = [], [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            s = (x_cpm >= lo) & (x_cpm < hi)
            if s.sum() >= 20:
                cen.append((lo + hi) / 2)
                mb.append(bio[s].mean())
                se.append(bio[s].std(ddof=1) / np.sqrt(s.sum()))
        ax[0].errorbar(cen, mb, yerr=1.96 * np.array(se), fmt='o', ms=3, color=col,
                       label=f'bin mean of sigma2_bio, {lab}')
        xx = np.linspace(x_cpm.min(), x_cpm.max(), 200)
        ax[0].plot(xx, eval_trend(tr, xx), color=col, lw=1.5)
    mv = pg_total['median_Vt'].to_numpy()
    ax[0].scatter(x_cpm, mv, s=1, alpha=0.15, color='grey', label='median Vt per gene (technical)')
    ax[0].set_yscale('log')
    ax[0].set_xlabel('log2 median CPM')
    ax[0].set_ylabel('variance of log2(CPM+1)')
    ax[0].set_title(f'Total channel, {G:,} genes x 92 donors')
    ax[0].legend(fontsize=7)
    P = pg_all[pg_all.arm == 'drop']
    x = np.log2(P.median_depth.to_numpy())
    tau = P.tau2_mom.to_numpy()
    tr = summ['allelic']['drop']['tau2_trend']
    edges = np.arange(np.floor(x.min()), np.ceil(x.max()) + 0.5, 0.5)
    cen, mb, se = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (x >= lo) & (x < hi)
        if s.sum() >= 20:
            cen.append((lo + hi) / 2)
            mb.append(tau[s].mean())
            se.append(tau[s].std(ddof=1) / np.sqrt(s.sum()))
    ax[1].errorbar(cen, mb, yerr=1.96 * np.array(se), fmt='o', ms=3, color='#1f5fa8',
                   label='bin mean of tau2 (method of moments)')
    xx = np.linspace(x.min(), x.max(), 200)
    ax[1].plot(xx, eval_trend(tr, xx), color='#1f5fa8', lw=1.5, label='IRLS trend')
    ax[1].scatter(x, P.median_Va, s=1, alpha=0.15, color='grey', label='median Va per gene (technical)')
    ax[1].set_yscale('log')
    ax[1].set_xlabel('log2 median haplotype-informative reads')
    ax[1].set_ylabel('variance of log2 allelic ratio')
    ax[1].set_title(f'Allelic channel, {len(P):,} genes\n(n_a >= {MIN_NA}, zero-haplotype pairs dropped)',
                    fontsize=10)
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / 'fig_bio_vs_tech.png', dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(1, 4, figsize=(18, 4.2))
    for k, (col, lab) in enumerate((('delta_total_full', 'total, all 17 covariates'),
                                    ('delta_total_no_expr_pcs', 'total, no expression PCs'),
                                    ('delta_total_meta_only', 'total, metadata only'),
                                    ('delta_allelic_depth_spline',
                                     'allelic, zero-haplotype pairs dropped, depth spline'))):
        ax[k].scatter(don.rin, don[col], s=12, color='#1f5fa8')
        for s in NAMED:
            ax[k].annotate(s, (don.loc[s, 'rin'], don.loc[s, col]), fontsize=7)
        key = {'delta_total_full': 'total_full', 'delta_total_no_expr_pcs': 'total_no_expr_pcs',
               'delta_total_meta_only': 'total_meta_only',
               'delta_allelic_depth_spline': 'allelic_drop_depth_spline'}[col]
        dc = summ['donor_component'][key]
        ax[k].axhline(0, color='grey', lw=0.5)
        ax[k].set_xlabel('RIN')
        ax[k].set_ylabel('donor effect on log squared residual')
        ax[k].set_title(f'{lab}\nsd {dc["sd_delta"]:.3f} (permutation floor {dc["sd_delta_perm_floor"]:.3f}), 92 donors',
                        fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / 'fig_donor_effects.png', dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.hist(r2, bins=80, density=True, alpha=0.7, color='#1f5fa8', label=f'observed, {G:,} genes')
    xx = np.linspace(0, r2.max(), 300)
    a = summ['ancestry']
    ax.plot(xx, sps.beta.pdf(xx, 1.5, a['df_after_gpcs'] / 2), color='#c0504d',
            label='null Beta(3/2, 74/2)')
    rb = a['rebuilt_pc_null']
    ax.axvline(a['observed_mean'], color='#1f5fa8', ls='--', lw=1, label='observed mean')
    ax.axvline(a['perm_null_mean'], color='#c0504d', ls='--', lw=1, label='row-permutation null mean')
    ax.axvline(rb['null_three_pcs_mean'], color='#2e8b57', ls='-', lw=1.5,
               label='null mean with expression PCs rebuilt')
    ax.axvspan(rb['null_three_pcs_mean'] - 2 * rb['null_three_pcs_sd'],
               rb['null_three_pcs_mean'] + 2 * rb['null_three_pcs_sd'], color='#2e8b57', alpha=0.12)
    ax.set_xlabel('partial R^2 of 3 genotype PCs after RNA-tied covariates')
    ax.set_ylabel('density')
    ax.set_title(f'Genotype-PC share of total-channel residual variance, {G:,} genes x 92 donors\n'
                 f'observed mean {a["observed_mean"]:.3f}; rebuilt null {rb["null_three_pcs_mean"]:.3f} '
                 f'(sd {rb["null_three_pcs_sd"]:.3f}); row-permutation null {a["perm_null_mean"]:.3f}',
                 fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / 'fig_ancestry.png', dpi=130)
    plt.close(fig)


if __name__ == '__main__':
    main()

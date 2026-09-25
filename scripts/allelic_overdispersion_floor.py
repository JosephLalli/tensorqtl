"""Is the allelic channel's positive weight-residual coupling an additive
between-donor floor of the beta-binomial kind?

QUESTION. Under the records-permutation null (scripts/null_permutation_instrument.py)
the allelic channel's nominal p is anticonservative: 0.069 / 0.020 / 0.0057 at
0.05 / 0.01 / 0.001. The first-order closed form of a gene's sd(beta)/se ratio
squared under records permutation is

    R_g = n_a * sum(w z^2) / (sum(w) * sum(z^2)),   w = 1/v_a,  z^2 = w a^2,

which is 1 in expectation when Var(a_i) is proportional to v_a,i. R_g > 1 means
the squared whitened residual z^2 is larger where the weight is larger. A beta-
binomial (BB) allelic count model predicts exactly that: if a donor's allelic
fraction p_i varies between donors with Var(p_i) = rho p q (rho is the BB
intra-class correlation, the fraction of a count's variance that is shared by
all its reads), then on the log-ratio scale (natural log)

    Var(a_i) ~ 1/(n_i p q) + rho/(p q),

a counting term that shrinks with allele-resolved depth n_i plus a floor
tau = rho/(p q) that does not. Weights 1/v_a then over-trust deep donors and
E[z^2_i] = sigma^2 + tau w_i rises linearly in w, forcing R_g >= 1.

DESIGN. Everything is on the instrument's 46 genes x 92 donors at RASQUAL's
observed lead, with the instrument's arrays (inputs_at_lead.npz) and its
permutation stream (RandomState(42), 2,000 permutations).

  (a) Count-based rho. Posterior-mean haplotype counts mL, mR come from the
      Gibbs cache (by sample id, never by position). Per gene, a BB maximum-
      likelihood fit on (y = mL, n = mL + mR) over donors with n > 0, free
      mean p_g and rho_g, with a likelihood ratio against the binomial
      (rho = 0; boundary, so p = 0.5 * chi2_1 tail). A method-of-moments
      estimator, rho_MoM = sum[(y - n p)^2/(p q) - n] / sum n(n - 1), is the
      cross-check. A pooled rho shares one rho across the 59 genes with
      per-gene p. A second BB fit replaces nominal n by a variance-matched
      effective count n_eff,i = 1/(v_a,i p q), so its rho is the floor BEYOND
      what the Gibbs variance already carries (the nominal fit reads Salmon's
      read-assignment ambiguity as overdispersion). The depth shape is read
      directly: pooled over genes and binned by allele-resolved depth (n >= 20
      only; the kappa = 0.5 pseudocount attenuates a at low n), the mean of
      a^2 - q_a (q_a = 1/(mL+.5) + 1/(mR+.5), the binomial counting variance)
      and of a^2 - v_a must be constant in n under a floor. Per gene, z^2 is
      fitted as N(0, m_i) under a floor (m = sigma^2 + tau w, i.e. the
      residual-based ML additive tau) and under a power law (m = c w^delta).
  (b) Does rho predict R_g and sdratio_a? Spearman rank correlation (Pearson
      correlation of ranks) with p, plus the floor's own prediction
      R_pred = (sigma^2 mean(w) + tau mean(w^2)) / (mean(w) (sigma^2 + tau
      mean(w))) with tau from counts. Repeated with each gene's dominant
      record (largest share of sum w z^2) left out of BOTH rho and R_g, so one
      record cannot be counted twice.
  (c) RASQUAL's per-gene theta (output field 15, stale column name 'rho' in
      observed_rasqual.tsv). In rasqual_src/src/nbem.c the beta-binomial
      precision is theta*ki*km (lines ~1088-1101) and the negative-binomial
      size is theta*ki*kij (~771-780): ONE precision shared by the total and
      allelic likelihoods, scaled per sample by the size factor, under a
      Gamma(2.02, 0.2) prior (~875), clamped to [1e-5, 10000] (~634). Large
      theta = little overdispersion. Reported as itself, never converted.
  (d) External benchmark (tests/ase_external_benchmark.py, imported, not
      edited): simulate_locus's NB + BB generator at the benchmark design
      (N = 200, mu = 200, NB dispersion 0.2, allele-specific fraction 0.25) and
      a design matched to the real data (each of the 46 genes' real 92 depths
      n_i = round(mL + mR) and real lead s_i, y ~ BB(n_i, 0.5, rho) with the
      benchmark's parameterization), rho in {0, 0.003, 0.01, 0.03}. Gibbs draws
      are emulated as the benchmark does (80 binomial draws at the smoothed
      fraction), summaries via compute_summaries_from_gibbs (count_noise on),
      and the ALLELIC channel is fitted with the instrument's algebra (through
      the origin, w = 1/v_a, F(1, n_a - 1)); hapmix_pval is NOT used (combined
      statistic, intercept kept, fabricated total channel). The oracle arm
      multiplies v_a by the BB inflation 1 + (n - 1) rho (q_a included, so the
      oracle v_a is proportional to the true variance). In the matched design
      each gene's own rejection rate is ranked against the real per-gene rate
      (does a COMMON floor at each gene's real depth spread pick out the
      anticonservative genes?). These are resampling
      nulls, not records permutations, and the emulation cannot reproduce the
      real data's Gibbs variance exceeding Poisson; the matched design matches
      depth spread only.
  (e) Isolating arms on the real permutation stream: Gaussian records
      a_i ~ N(0, sigma^2_g v_a,i + tau_g) with tau_g pinned from COUNTS (not
      residuals) and sigma^2_g = mean(z^2) - tau_g mean(w) by moments (a
      negative sigma^2_g means the count floor exceeds what the residuals
      allow; it is flagged and simulated at sigma^2 = 0). Baseline tau = 0.
      fraction = (rate_arm - rate_tau0)/(rate_real - rate_tau0), with a gene-
      clustered bootstrap interval pairing the real and arm rates by gene.
      The residual-ML arm (tau and sigma^2 fitted by Gaussian ML from a
      itself) is NOT independent of R_g; its control fits the same floor to
      model-true records (tau = 0) and simulates from the fit, measuring how
      much rate a fitted tau_hat >= 0 creates on its own.

HELD FIXED: the 46 genes, their leads, the donor set, v_a, s, the permutation
stream, the statistic. VARIES: the source of tau (counts nominal / counts
effective / pooled / residual ML / none) and, in (d), rho and the design.

GATES (abort on failure):
  1. vectorized allelic t^2 on the RandomState(42) stream reproduces
     null_long.tsv.gz t2_a for every (gene, perm) to relative 1e-9 and its
     pooled rejection counts exactly at 0.05, 0.01, 0.001; fit_channels spot-
     checked on 3 genes x 3 permutations.
  2. compute_summaries_from_gibbs on the cache rows, remapped by sample id,
     reproduces the npz A_all / Va_all to 1e-12, which pins mL, mR to the
     right donors.

Master seed 42; every stream other than the instrument's permutation stream is
a child of np.random.SeedSequence(42). The environment variable AOF_BENCH_REPS
shortens the benchmark sweep for smoke runs only; every recorded number uses
the default 20,000 replicates per (design, rho). Runtime ~100 s on 8 cores.

Run:  python3 scripts/allelic_overdispersion_floor.py
"""
import contextlib
import io
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize as so
from scipy import stats as sps
from scipy.special import betaln

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / 'tests'))

D = Path('/mnt/ssd/lalli/brainvar_hapmix_deploy')
INST = D / 'nominal_p_null_instrument_20260925'
CACHE = D / 'cache' / 'gibbs_56b63c3b37ed5df8'
RASQ = D / 'rasqual_default_mode_20260923' / 'observed_rasqual.tsv'
OUT = D / 'allelic_overdispersion_floor_20260925'
SEED, EPS, N_PERM, N_BOOT = 42, 1e-12, 2000, 2000
ALPHAS = (0.05, 0.01, 0.001)
K_SIM = 10               # simulated record sets per gene in the isolating arms
K_CTRL = 5               # model-true record sets per gene in the fit-then-simulate control
N_MODEL_R = 4000         # model-null draws per gene for the R_g reference
# replicates per (design, rho) in the benchmark sweep; the override exists for
# smoke runs only, every reported number uses the default
BENCH_REPS = int(os.environ.get('AOF_BENCH_REPS', 20000))
RHOS = (0.0, 0.003, 0.01, 0.03)
N_DRAWS, KAPPA = 80, 0.5
SS = np.random.SeedSequence(SEED).spawn(8)   # child streams, fixed roles:
# 0 gene bootstrap, 1 model-null R, 2 isolating arms, 3 benchmark sweep


# ---------------------------------------------------------------------------
#  the allelic statistic, vectorized over permutations and record sets
# ---------------------------------------------------------------------------

def allelic_t2(a, va, s, P):
    """t^2 of the through-origin weighted allelic slope for every permutation.

    a may be (92,) or (K, 92) record sets sharing va. Records (a, va) move by
    P; s stays. t^2 = dof num^2 / (den S - num^2), S = sum w a^2.
    """
    ok = np.isfinite(va) & (va > EPS)
    w = np.where(ok, 1.0 / np.where(ok, va, 1.0), 0.0)
    a2 = np.atleast_2d(np.where(ok, a, 0.0))
    wa = w * a2
    S = (wa * a2).sum(-1)[:, None]
    num = (wa[:, P] * s).sum(-1)
    den = (w[P] * s * s).sum(-1)[None, :]
    dof = int(ok.sum()) - 1
    t2 = dof * num ** 2 / (den * S - num ** 2)
    return t2, dof


def gene_boot_rate(k, n, rng, nb=N_BOOT):
    """Pooled rate k.sum()/n.sum() with a gene-clustered percentile interval."""
    k, n = np.asarray(k, float), np.asarray(n, float)
    idx = rng.integers(0, len(k), size=(nb, len(k)))
    b = k[idx].sum(1) / n[idx].sum(1)
    return float(k.sum() / n.sum()), float(np.quantile(b, .025)), float(np.quantile(b, .975))


# ---------------------------------------------------------------------------
#  beta-binomial fits on counts
# ---------------------------------------------------------------------------

def _bb_ll(y, n, p, rho):
    nu = (1.0 - rho) / rho
    return np.sum(betaln(y + p * nu, n - y + (1 - p) * nu) - betaln(p * nu, (1 - p) * nu))


def _bin_ll(y, n, p):
    return np.sum(y * np.log(p) + (n - y) * np.log1p(-p))


def bb_fit(y, n):
    """ML beta-binomial on (possibly non-integer) counts: mean p, ICC rho.
    Binomial-coefficient terms are omitted; they cancel in the LR."""
    m = n > 0
    y, n = y[m], n[m]
    p0 = np.clip(y.sum() / n.sum(), 1e-6, 1 - 1e-6)
    llb = _bin_ll(y, n, p0)

    def nll(th):
        p = 1 / (1 + np.exp(-th[0])); rho = 1 / (1 + np.exp(-th[1]))
        v = -_bb_ll(y, n, p, rho)
        return v if np.isfinite(v) else 1e300
    best = None
    for r0 in (1e-4, 1e-3, 1e-2, 1e-1):
        f = so.minimize(nll, [np.log(p0 / (1 - p0)), np.log(r0 / (1 - r0))],
                        method='L-BFGS-B', bounds=[(-8, 8), (-18, 5)])
        if best is None or f.fun < best.fun:
            best = f
    p = 1 / (1 + np.exp(-best.x[0])); rho = 1 / (1 + np.exp(-best.x[1]))
    LR = max(2 * (-best.fun - llb), 0.0)
    return dict(p=p, rho=rho, LR=LR, pval=0.5 * sps.chi2.sf(LR, 1), n_don=int(m.sum()))


def bb_mom(y, n):
    m = n > 1
    y, n = y[m], n[m]
    p = y.sum() / n.sum(); q = 1 - p
    return float(np.sum((y - n * p) ** 2 / (p * q) - n) / np.sum(n * (n - 1)))


def bb_pooled(Y, N):
    """One rho shared by all genes, per-gene p profiled out."""
    def prof(lr):
        rho = 1 / (1 + np.exp(-lr)); tot = 0.0
        for y, n in zip(Y, N):
            m = n > 0
            f = so.minimize_scalar(lambda t: -_bb_ll(y[m], n[m], 1 / (1 + np.exp(-t)), rho),
                                   bounds=(-8, 8), method='bounded')
            tot += f.fun
        return tot
    f = so.minimize_scalar(prof, bounds=(-14, 0), method='bounded',
                           options=dict(xatol=1e-4))
    return float(1 / (1 + np.exp(-f.x)))


def gauss_fit(a, va, form):
    """a ~ N(0, m) with m = sigma^2 va + tau (floor) or c va^(1-delta) (power).
    Equivalently z^2 = a^2/va ~ m/va chi2_1. Returns params and max loglik."""
    A2 = a ** 2
    if form == 'floor':
        def nll(th):
            m = np.exp(th[0]) * va + np.exp(th[1])
            return 0.5 * np.sum(np.log(m) + A2 / m)
        s0 = np.log(np.mean(A2 / va))
        best = min((so.minimize(nll, [s0, np.log(t0)], method='Nelder-Mead',
                                options=dict(xatol=1e-8, fatol=1e-10, maxiter=4000))
                    for t0 in (1e-6, 1e-3, 1e-2, 1e-1)), key=lambda r: r.fun)
        f0 = so.minimize_scalar(lambda l: 0.5 * np.sum(np.log(np.exp(l) * va) + A2 / (np.exp(l) * va)),
                                bounds=(-30, 30), method='bounded')
        s2, tau = np.exp(best.x)
        return dict(sigma2=s2, tau=tau, ll=-best.fun, LR=max(2 * (f0.fun - best.fun), 0.0))
    def nll(th):
        m = np.exp(th[0]) * va ** (1 - th[1])
        return 0.5 * np.sum(np.log(m) + A2 / m)
    best = min((so.minimize(nll, [np.log(np.mean(A2 / va)), d0], method='Nelder-Mead',
                            options=dict(xatol=1e-8, fatol=1e-10, maxiter=4000))
                for d0 in (0.0, 0.3, -0.3)), key=lambda r: r.fun)
    return dict(c=np.exp(best.x[0]), delta=best.x[1], ll=-best.fun)


def R_of(w, z2):
    return len(w) * np.sum(w * z2) / (np.sum(w) * np.sum(z2))


def R_pred(w, sigma2, tau):
    return (sigma2 * w.mean() + tau * (w ** 2).mean()) / (w.mean() * (sigma2 + tau * w.mean()))


# ---------------------------------------------------------------------------
#  external benchmark driver
# ---------------------------------------------------------------------------

def _emulate_and_fit(yL, yR, s, rho, rng):
    """Benchmark emulation of Gibbs draws, then the instrument's allelic fit.
    Returns (p, p_oracle)."""
    import ase_external_benchmark as B
    n = (yL + yR).astype(int)
    frac = (yL + KAPPA) / (n + 2 * KAPPA)
    dL = rng.binomial(n[:, None], frac[:, None], size=(len(n), N_DRAWS)).astype(float)
    dR = n[:, None] - dL
    with contextlib.redirect_stdout(io.StringIO()):
        A, _, Va, _, _ = B.compute_summaries_from_gibbs(dL[None], dR[None])
    a, va = A[0], Va[0]
    out = []
    for v in (va, va * (1 + np.maximum(n - 1, 0) * rho)):
        ok = v > EPS
        w = np.where(ok, 1 / np.where(ok, v, 1), 0)
        x, y = np.sqrt(w) * s, np.sqrt(w) * np.where(ok, a, 0)
        xx = x @ x
        if ok.sum() < 5 or xx <= 0:
            out.append(np.nan); continue
        b = (x @ y) / xx; e = y - b * x; dof = int(ok.sum()) - 1
        t2 = b ** 2 / ((e @ e) / dof / xx)
        out.append(sps.f.sf(t2, 1, dof))
    return out


def bench_worker(args):
    design, rho, reps, seed, matched = args
    import ase_external_benchmark as B
    rng = np.random.default_rng(seed)
    ps = np.full((reps, 2), np.nan)
    gidx = np.full(reps, -1)
    for r in range(reps):
        if design == 'benchmark':
            d = B.simulate_locus(200, rng, kappa=1.0, rho=rho)
            yL, yR, s = d['yL'], d['yR'], d['s']
        else:
            n_all, s_all = matched
            g = r % len(n_all)
            gidx[r] = g
            n, s = n_all[g], s_all[g]
            if rho > 0:
                nu = (1 - rho) / rho
                pr = rng.beta(0.5 * nu, 0.5 * nu, size=len(n))
            else:
                pr = np.full(len(n), 0.5)
            yL = rng.binomial(n, pr).astype(float); yR = (n - yL).astype(float)
        ps[r] = _emulate_and_fit(yL.astype(float), yR.astype(float), s, rho, rng)
    return design, rho, ps, gidx


# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    OUT.mkdir(exist_ok=True)
    d = np.load(INST / 'inputs_at_lead.npz', allow_pickle=True)
    genes = list(d['genes']); G = len(genes); donors = list(d['donors'])
    a_, va_, s_ = d['a'], d['va'], d['s']
    rng = np.random.RandomState(SEED)
    P = np.stack([rng.permutation(len(donors)) for _ in range(N_PERM)])

    # ---- gate 1: reproduce the instrument's allelic statistic ---------------
    nl = pd.read_csv(INST / 'null_long.tsv.gz', sep='\t')
    nl = nl.set_index(['gene', 'perm']).sort_index()
    T2 = np.zeros((G, N_PERM)); DOF = np.zeros(G, int)
    for i in range(G):
        t2, dof = allelic_t2(a_[i], va_[i], s_[i], P)
        T2[i], DOF[i] = t2[0], dof
    ref = np.stack([nl.loc[g].t2_a.values for g in genes])
    rel = np.abs(T2 - ref) / np.maximum(np.abs(ref), 1e-300)
    Preal = sps.f.sf(T2, 1, DOF[:, None])
    refp = np.stack([nl.loc[g].p_a.values for g in genes])
    cnt_ok = all(int((Preal < al).sum()) == int((refp < al).sum()) for al in ALPHAS)
    from null_permutation_instrument import fit_channels
    spot = []
    for i in range(3):
        for pi in range(3):
            prm = P[pi]
            f = fit_channels(a_[i][prm], s_[i], va_[i][prm], d['t'][i][prm], d['g'][i],
                             d['vt'][i][prm], d['C'][prm])
            spot.append(abs(f['t2_a'] - T2[i, pi]) / f['t2_a'])
    print(f'gate 1: max rel |dt2_a| {rel.max():.2e} over {rel.size} (gene, perm); '
          f'pooled counts equal {cnt_ok}; fit_channels spot max {max(spot):.2e}', flush=True)
    if rel.max() > 1e-9 or not cnt_ok or max(spot) > 1e-9:
        raise SystemExit('gate 1 FAILED')
    brng = np.random.default_rng(SS[0])
    real = {}
    for al in ALPHAS:
        k = (Preal < al).sum(1)
        real[str(al)] = gene_boot_rate(k, np.full(G, N_PERM), brng)
    print('real allelic rates:', {k: round(v[0], 5) for k, v in real.items()})

    # ---- gate 2: cache counts on the right donors ---------------------------
    import run_hapmixqtl_from_salmon as H
    cg = open(CACHE / 'genes.txt').read().split()
    samples = open(CACHE / 'samples.txt').read().split()
    all_genes = list(d['all_genes'])
    rows = [cg.index(g) for g in all_genes]
    keep = [samples.index(x) for x in donors]
    mm = {k: np.load(CACHE / f'{k}.npy', mmap_mode='r') for k in ('YL', 'YR', 'YT')}
    YL, YR, YT = (np.asarray(mm[k][rows])[:, keep] for k in ('YL', 'YR', 'YT'))
    with contextlib.redirect_stdout(io.StringIO()):
        A, _, Va, _, _ = H.compute_summaries_from_gibbs(YL, YR, yT=YT)
    g2a = np.max(np.abs(A - d['A_all'])); g2v = np.max(np.abs(Va - d['Va_all']))
    print(f'gate 2: max |dA| {g2a:.2e}, max |dVa| {g2v:.2e}', flush=True)
    if g2a > 1e-12 or g2v > 1e-12:
        raise SystemExit('gate 2 FAILED')
    mL_all, mR_all = YL.mean(2), YR.mean(2)
    del YL, YR, YT
    ai = {g: all_genes.index(g) for g in genes}

    # ---- (a) count-based rho and the depth shape ------------------------------
    per, depth_rows = [], []
    for i, g in enumerate(genes):
        j = ai[g]
        mL, mR = mL_all[j], mR_all[j]; n = mL + mR
        ok = va_[i] > EPS
        assert np.array_equal(ok, n > 0), g
        a, va, w = a_[i][ok], va_[i][ok], 1 / va_[i][ok]
        z2 = w * a ** 2
        bb = bb_fit(mL, n)
        mom = bb_mom(mL, n)
        pq = bb['p'] * (1 - bb['p'])
        # effective counts: binomial variance matched to v_a
        neff = 1.0 / (va * pq)
        f = np.exp(a) / (1 + np.exp(a))
        bbe = bb_fit(f * neff, neff)
        fl = gauss_fit(a, va, 'floor'); pw = gauss_fit(a, va, 'power')
        tau_nom = bb['rho'] / pq
        tau_eff = bbe['rho'] / (bbe['p'] * (1 - bbe['p']))
        # dominant record and leave-one-record-out
        share = w * z2 / np.sum(w * z2)
        top = int(np.argmax(share))
        idx_ok = np.where(ok)[0]
        drop = np.ones(ok.sum(), bool); drop[top] = False
        nn = n[ok]; yy = mL[ok]
        bb_loro = bb_fit(yy[drop], nn[drop])
        # BB tail probability of the dominant record's imbalance
        nt = int(round(nn[top])); yt = int(round(yy[top]))
        al_, be_ = bb['p'] * (1 - bb['rho']) / bb['rho'], (1 - bb['p']) * (1 - bb['rho']) / bb['rho']
        bd = sps.betabinom(nt, al_, be_)
        dev = abs(yt - nt * bb['p'])
        lo_, hi_ = np.floor(nt * bb['p'] - dev), np.ceil(nt * bb['p'] + dev)
        tail = float(bd.cdf(lo_) + bd.sf(hi_ - 1))
        R = R_of(w, z2)
        mz, mw = z2.mean(), w.mean()
        sig_nom = mz - tau_nom * mw; sig_eff = mz - tau_eff * mw
        per.append(dict(
            gene=g, stratum=str(d['strata'][i]), n_a=int(ok.sum()),
            median_n=float(np.median(nn)), p_hat=bb['p'],
            rho_count=bb['rho'], rho_LR=bb['LR'], rho_p=bb['pval'], rho_mom=mom,
            rho_eff=bbe['rho'], rho_eff_LR=bbe['LR'], rho_eff_p=bbe['pval'],
            tau_count=tau_nom, tau_eff=tau_eff,
            tau_ml=fl['tau'], sigma2_ml=fl['sigma2'], tau_ml_LR=fl['LR'],
            ll_floor=fl['ll'], ll_power=pw['ll'], delta_power=pw['delta'],
            R=R, R_pred_count=R_pred(w, max(sig_nom, 0), tau_nom),
            R_pred_eff=R_pred(w, max(sig_eff, 0), tau_eff),
            R_pred_ml=R_pred(w, fl['sigma2'], fl['tau']),
            sigma2_mom_count=sig_nom, sigma2_mom_eff=sig_eff,
            top_donor=donors[idx_ok[top]], top_share=float(share[top]),
            top_n=float(nn[top]), top_a=float(a[top]), top_va=float(va[top]),
            top_va_over_q=float(va[top] / (1 / (yy[top] + .5) + 1 / (nn[top] - yy[top] + .5))),
            top_bb_tail=tail, rho_count_loro=bb_loro['rho'],
            R_loro=R_of(w[drop], z2[drop]),
            spearman_w_z2=float(sps.spearmanr(w, z2)[0]),
            spearman_n_z2=float(sps.spearmanr(nn, z2)[0])))
        qa = 1 / (yy + .5) + 1 / (nn - yy + .5)
        for k in range(ok.sum()):
            depth_rows.append(dict(gene=g, donor=donors[idx_ok[k]], n=nn[k], a=a[k], va=va[k],
                                   qa=qa[k], z2=z2[k], w=w[k], z2_rel=z2[k] / mz))
    per = pd.DataFrame(per).set_index('gene')
    inst = pd.read_csv(INST / 'per_gene.tsv', sep='\t').set_index('gene').loc[genes]
    per['sdratio_a'] = inst.sdratio_a
    per['rej0.05_a'] = inst['rej0.05_a']; per['rej0.01_a'] = inst['rej0.01_a']
    per['rej0.001_a'] = (Preal < 0.001).mean(1)
    rec = pd.DataFrame(depth_rows)

    # pooled rho across 59 genes
    Yp, Np = [], []
    for j in range(len(all_genes)):
        Yp.append(mL_all[j]); Np.append(mL_all[j] + mR_all[j])
    rho_pool = bb_pooled(Yp, Np)
    print(f'pooled count rho over {len(all_genes)} genes: {rho_pool:.5f}', flush=True)

    # depth bins, pooled, gene-clustered bootstrap of bin means
    rec20 = rec[rec.n >= 20].copy()
    edges = [20, 100, 1000, 10000, np.inf]
    rec20['bin'] = pd.cut(rec20.n, edges, right=False,
                          labels=['20-99', '100-999', '1000-9999', '10000+'])
    rec20['a2_minus_qa'] = rec20.a ** 2 - rec20.qa
    rec20['a2_minus_va'] = rec20.a ** 2 - rec20.va
    rec20['a2_over_qa'] = rec20.a ** 2 / rec20.qa
    rec20['va_over_qa'] = rec20.va / rec20.qa
    dbins = []
    gl = np.array(genes)
    for b, sub in rec20.groupby('bin', observed=True):
        row = dict(bin=str(b), n_records=len(sub), n_genes=sub.gene.nunique(),
                   mean_n=float(sub.n.mean()))
        for col in ('a2_minus_qa', 'a2_minus_va', 'a2_over_qa', 'z2', 'z2_rel', 'va_over_qa'):
            gs = sub.groupby('gene')[col].agg(['sum', 'size'])
            est, lo, hi = gene_boot_rate(gs['sum'].values, gs['size'].values, brng)
            row[col] = est; row[col + '_lo'] = lo; row[col + '_hi'] = hi
            row[col + '_median'] = float(sub[col].median())
        row['implied_rho'] = (row['a2_over_qa'] - 1) / (row['mean_n'] - 1)
        dbins.append(row)
    dbins = pd.DataFrame(dbins)
    # within-gene depth quintiles of z2 relative to the gene mean
    rec20['q'] = rec20.groupby('gene').n.transform(
        lambda x: pd.qcut(x.rank(method='first'), 5, labels=False))
    quint = rec20.groupby('q').agg(mean_n=('n', 'mean'), z2_rel=('z2_rel', 'mean'),
                                  z2_rel_median=('z2_rel', 'median'), n=('z2', 'size'))
    rho_within = per.spearman_n_z2
    # depth versus Gibbs inflation (v_a / q_a) as carriers of the coupling. A
    # floor on top of v_a predicts E[z^2] = sigma^2 + tau w with w ~ n / (4
    # inflation), so z^2 should rise with depth at fixed inflation AND fall
    # with inflation at fixed depth. Within-gene rank regression, all records.
    rec['infl'] = rec.va / rec.qa
    rk = lambda x: (sps.rankdata(x) - 0.5) / len(x) - 0.5
    coefs = []
    for g in genes:
        sub = rec[rec.gene == g]
        X = np.column_stack([rk(np.log(sub.n)), rk(np.log(sub.infl))])
        coefs.append(np.linalg.lstsq(X, rk(sub.z2), rcond=None)[0])
        per.loc[g, 'rankcoef_depth'], per.loc[g, 'rankcoef_inflation'] = coefs[-1]
        per.loc[g, 'spearman_n_inflation'] = sps.spearmanr(sub.n, sub.infl)[0]
    coefs = np.array(coefs)
    rec['z2_gene_rel'] = rec.z2 / rec.groupby('gene').z2.transform('mean')
    for col, src in (('nq', 'n'), ('iq', 'infl')):
        rec[col] = rec.groupby('gene')[src].transform(
            lambda x: pd.qcut(x.rank(method='first'), 3, labels=False))
    joint = rec.pivot_table(index='nq', columns='iq', values='z2_gene_rel', aggfunc='mean')
    joint_n = rec.pivot_table(index='nq', columns='iq', values='z2_gene_rel', aggfunc='size')
    decomp = dict(
        median_rankcoef_depth=float(np.median(coefs[:, 0])),
        n_pos_depth=int((coefs[:, 0] > 0).sum()),
        sign_p_depth=float(sps.binomtest(int((coefs[:, 0] > 0).sum()), G, .5).pvalue),
        median_rankcoef_inflation=float(np.median(coefs[:, 1])),
        n_pos_inflation=int((coefs[:, 1] > 0).sum()),
        sign_p_inflation=float(sps.binomtest(int((coefs[:, 1] > 0).sum()), G, .5).pvalue),
        median_spearman_n_inflation=float(per.spearman_n_inflation.median()),
        joint_tertiles_z2_rel={f'depth{r}': {f'infl{c}': float(joint.loc[r, c]) for c in joint.columns}
                               for r in joint.index},
        joint_tertiles_n={f'depth{r}': {f'infl{c}': int(joint_n.loc[r, c]) for c in joint_n.columns}
                          for r in joint_n.index})
    signtest = sps.binomtest(int((rho_within > 0).sum()), G, 0.5).pvalue

    # ---- (b) prediction of R_g and sdratio_a ----------------------------------
    def sp(x, y):
        r, p = sps.spearmanr(x, y); return dict(rho=float(r), p=float(p))
    dom = ['CALM2', 'FABP7', 'ANKRD36B', 'MUTYH', 'MATR3']
    nodom = [g for g in genes if g not in dom]
    pred = {
        'rho_count~R': sp(per.rho_count, per.R),
        'rho_count~sdratio_a': sp(per.rho_count, per.sdratio_a),
        'rho_count~rej0.01_a': sp(per.rho_count, per['rej0.01_a']),
        'rho_eff~R': sp(per.rho_eff, per.R),
        'rho_eff~sdratio_a': sp(per.rho_eff, per.sdratio_a),
        'tau_count~tau_ml': sp(per.tau_count, per.tau_ml),
        'R_pred_count~R': sp(per.R_pred_count, per.R),
        'R_pred_eff~R': sp(per.R_pred_eff, per.R),
        'R_pred_ml~R': sp(per.R_pred_ml, per.R),
        'R~sdratio_a': sp(per.R, per.sdratio_a),
        'R~sdratio_a^2_pearson': float(np.corrcoef(per.R, per.sdratio_a ** 2)[0, 1]),
        'loro: rho_count_loro~R_loro': sp(per.rho_count_loro, per.R_loro),
        'no dominant-record genes: rho_count~R': sp(per.loc[nodom].rho_count, per.loc[nodom].R),
        'no dominant-record genes: rho_count~sdratio_a': sp(per.loc[nodom].rho_count, per.loc[nodom].sdratio_a),
        'no dominant-record genes: rho_eff~R': sp(per.loc[nodom].rho_eff, per.loc[nodom].R),
    }
    lr = np.log(per.R_pred_count / per.R)
    pred['log(R_pred_count/R) median'] = float(np.median(lr))
    pred['log(R_pred_eff/R) median'] = float(np.median(np.log(per.R_pred_eff / per.R)))
    pred['tau_count/tau_ml median'] = float(np.median(per.tau_count / np.maximum(per.tau_ml, 1e-12)))
    pred['tau_eff/tau_ml median'] = float(np.median(per.tau_eff / np.maximum(per.tau_ml, 1e-12)))
    pred['n genes sigma2_mom_count<0'] = int((per.sigma2_mom_count < 0).sum())
    pred['n genes sigma2_mom_eff<0'] = int((per.sigma2_mom_eff < 0).sum())
    pred['floor vs power: sum ll_floor-ll_power'] = float((per.ll_floor - per.ll_power).sum())
    pred['floor vs power: n genes floor better'] = int((per.ll_floor > per.ll_power).sum())
    up = per[per.R > 1]
    pred['floor vs power, R>1 genes: n'] = int(len(up))
    pred['floor vs power, R>1 genes: sum ll_floor-ll_power'] = float((up.ll_floor - up.ll_power).sum())
    pred['floor vs power, R>1 genes: n floor better'] = int((up.ll_floor > up.ll_power).sum())
    pred['tau_eff~tau_ml'] = sp(per.tau_eff, per.tau_ml)
    pred['tau_ml~R (residual-based, not independent)'] = sp(per.tau_ml, per.R)
    pred['rho_count~rho_eff'] = sp(per.rho_count, per.rho_eff)
    pred['median va/q_a of top record'] = float(per.top_va_over_q.median())

    # model-null reference for R_g (z iid N(0,1) at the real weights)
    mrng = np.random.default_rng(SS[1])
    for i, g in enumerate(genes):
        ok = va_[i] > EPS; w = 1 / va_[i][ok]
        z2 = mrng.standard_normal((N_MODEL_R, ok.sum())) ** 2
        Rn = len(w) * (z2 * w).sum(1) / (w.sum() * z2.sum(1))
        per.loc[g, 'R_null_q025'] = np.quantile(Rn, .025)
        per.loc[g, 'R_null_q975'] = np.quantile(Rn, .975)
        per.loc[g, 'R_null_p_lo'] = float((Rn <= per.loc[g, 'R']).mean())
    print('(a)/(b) done', round(time.time() - t0), 's', flush=True)

    # ---- (c) RASQUAL theta ----------------------------------------------------
    rq = pd.read_csv(RASQ, sep='\t').set_index('gene')
    assert (rq.rho == rq.overdispersion_rho).all()
    per['rasqual_theta'] = rq.loc[genes, 'rho'].values
    rasq = dict(n_rows=int(len(rq)), n_at_upper_10000=int((rq.rho >= 9999.99).sum()),
                n_at_lower_1e_5=int((rq.rho <= 1e-5).sum()),
                n_nonconverged=int((rq.convergence != 0).sum()),
                theta_median_46=float(per.rasqual_theta.median()),
                theta_q=[float(x) for x in np.quantile(per.rasqual_theta, [0, .25, .5, .75, 1])],
                vs_R=sp(per.rasqual_theta, per.R),
                vs_sdratio_a=sp(per.rasqual_theta, per.sdratio_a),
                vs_rho_count=sp(per.rasqual_theta, per.rho_count),
                vs_rho_eff=sp(per.rasqual_theta, per.rho_eff),
                vs_tau_ml=sp(per.rasqual_theta, per.tau_ml))

    # ---- (e) isolating arms on the real permutation stream --------------------
    arng = np.random.default_rng(SS[2])
    arms = {'tau0': None, 'count_nominal': 'tau_count', 'count_effective': 'tau_eff',
            'count_pooled': 'pooled', 'residual_ml': 'tau_ml'}
    arm_counts = {k: np.zeros((G, len(ALPHAS), K_SIM)) for k in arms}
    for i, g in enumerate(genes):
        ok = va_[i] > EPS; va = va_[i][ok]; w = 1 / va
        z2 = w * a_[i][ok] ** 2
        pq = per.loc[g, 'p_hat'] * (1 - per.loc[g, 'p_hat'])
        for name, src in arms.items():
            if src is None:
                tau, s2 = 0.0, 1.0
            elif src == 'tau_ml':
                tau, s2 = per.loc[g, 'tau_ml'], per.loc[g, 'sigma2_ml']
            else:
                tau = rho_pool / pq if src == 'pooled' else per.loc[g, src]
                s2 = max(z2.mean() - tau * w.mean(), 0.0)
                if src == 'pooled':
                    per.loc[g, 'sigma2_mom_pooled'] = z2.mean() - tau * w.mean()
            sim = np.zeros((K_SIM, len(donors)))
            sim[:, ok] = arng.standard_normal((K_SIM, ok.sum())) * np.sqrt(s2 * va + tau)
            t2, dof = allelic_t2(sim, va_[i], s_[i], P)
            p = sps.f.sf(t2, 1, dof)
            for ia, al in enumerate(ALPHAS):
                arm_counts[name][i, ia] = (p < al).sum(1)
    # control for the residual-ML arm: fit-then-simulate on model-true records
    # (tau = 0), so a tau_hat >= 0 forced by the fit is not read as a floor
    ctrl = np.zeros((G, len(ALPHAS), K_CTRL))
    for i, g in enumerate(genes):
        ok = va_[i] > EPS; va = va_[i][ok]
        for k in range(K_CTRL):
            a0 = arng.standard_normal(ok.sum()) * np.sqrt(va)
            fc = gauss_fit(a0, va, 'floor')
            sim = np.zeros((1, len(donors)))
            sim[0, ok] = arng.standard_normal(ok.sum()) * np.sqrt(fc['sigma2'] * va + fc['tau'])
            t2, dof = allelic_t2(sim, va_[i], s_[i], P)
            p = sps.f.sf(t2, 1, dof)[0]
            for ia, al in enumerate(ALPHAS):
                ctrl[i, ia, k] = (p < al).sum()
    arm_counts['residual_ml_control_tau0_truth'] = ctrl
    arm_res = {}
    frac_rng = np.random.default_rng(SS[0].spawn(1)[0])
    kreal = np.stack([(Preal < al).sum(1) for al in ALPHAS], 1)       # G x 3
    idx = frac_rng.integers(0, G, size=(N_BOOT, G))
    base = arm_counts['tau0'].sum(2)
    for name, C in arm_counts.items():
        tot = C.sum(2)                                               # G x 3
        r = {}
        for ia, al in enumerate(ALPHAS):
            K = C.shape[2]
            est = tot[:, ia].sum() / (G * N_PERM * K)
            per_k = C[:, ia, :].sum(0) / (G * N_PERM)
            mc_sd = float(per_k.std(ddof=1))
            rr = kreal[:, ia] / N_PERM; ra = tot[:, ia] / (N_PERM * K)
            rb = base[:, ia] / (N_PERM * K_SIM)
            fr = (ra.mean() - rb.mean()) / (rr.mean() - rb.mean())
            bfr = (ra[idx].mean(1) - rb[idx].mean(1)) / (rr[idx].mean(1) - rb[idx].mean(1))
            r[str(al)] = dict(rate=float(est), mc_sd_single_set=mc_sd,
                              mc_se_mean=mc_sd / np.sqrt(K),
                              spearman_per_gene_vs_real=sp(ra, rr),
                              fraction_of_excess=float(fr),
                              fraction_lo=float(np.quantile(bfr, .025)),
                              fraction_hi=float(np.quantile(bfr, .975)))
            per.loc[genes, f'arm_{name}_rej{al}'] = ra
        arm_res[name] = r
        print(f'  arm {name:16s}', {k: round(v['rate'], 5) for k, v in r.items()},
              'frac', {k: round(v['fraction_of_excess'], 3) for k, v in r.items()}, flush=True)
    print('(e) done', round(time.time() - t0), 's', flush=True)

    # ---- (d) external benchmark sweep ------------------------------------------
    n_match = [np.rint(mL_all[ai[g]] + mR_all[ai[g]]).astype(int) for g in genes]
    s_match = [s_[i] for i in range(G)]
    bseeds = SS[3].spawn(2 * len(RHOS))
    jobs, k = [], 0
    for design in ('benchmark', 'matched'):
        for rho in RHOS:
            reps = BENCH_REPS if design == 'benchmark' else 46 * (BENCH_REPS // 46 + 1)
            jobs.append((design, rho, reps, bseeds[k], (n_match, s_match))); k += 1
    with Pool(len(jobs)) as pool:
        bres = pool.map(bench_worker, jobs)
    brows, bgene = [], {}
    for design, rho, ps, gidx in bres:
        if design == 'matched':
            # does the matched BB arm rank genes like the real data does?
            for arm, col in (('va', 0), ('oracle', 1)):
                rr = {}
                for al in ALPHAS:
                    rg = np.array([np.nanmean(ps[gidx == i, col] < al) for i in range(G)])
                    per[f'bench_matched_rho{rho}_{arm}_rej{al}'] = rg
                    rr[str(al)] = dict(vs_real_rej=sp(rg, per[f'rej{al}_a'].values),
                                       vs_R=sp(rg, per.R.values))
                bgene[f'rho{rho}_{arm}'] = rr
        for arm, col in (('va', 0), ('oracle', 1)):
            p = ps[:, col]; p = p[np.isfinite(p)]
            row = dict(design=design, rho=rho, arm=arm, n=len(p))
            for al in ALPHAS:
                r = float((p < al).mean())
                row[f'rate{al}'] = r; row[f'se{al}'] = float(np.sqrt(r * (1 - r) / len(p)))
            brows.append(row)
    bench = pd.DataFrame(brows)
    print(bench.round(5).to_string(), flush=True)
    print('(d) done', round(time.time() - t0), 's', flush=True)

    # ---- (e) genes a floor cannot explain -------------------------------------
    below = per[per.R < 1].sort_values('R')
    below_sig = per[per.R_null_p_lo < 0.025]

    # ---- write -------------------------------------------------------------------
    per.to_csv(OUT / 'per_gene.tsv', sep='\t')
    dbins.to_csv(OUT / 'depth_bins.tsv', sep='\t', index=False)
    quint.to_csv(OUT / 'depth_quintiles_within_gene.tsv', sep='\t')
    rec.to_csv(OUT / 'records.tsv.gz', sep='\t', index=False)
    bench.to_csv(OUT / 'benchmark_sweep.tsv', sep='\t', index=False)
    summ = dict(
        gates=dict(gate1_max_rel_t2a=float(rel.max()), gate1_counts_equal=cnt_ok,
                   gate1_fit_channels_spot=float(max(spot)),
                   gate2_max_dA=float(g2a), gate2_max_dVa=float(g2v)),
        real_allelic=real,
        rho_count=dict(median=float(per.rho_count.median()),
                       q=[float(x) for x in np.quantile(per.rho_count, [0, .25, .5, .75, 1])],
                       n_LR_sig_0_05=int((per.rho_p < 0.05).sum()),
                       median_mom=float(per.rho_mom.median()),
                       spearman_ml_mom=sp(per.rho_count, per.rho_mom),
                       pooled_59=rho_pool),
        rho_eff=dict(median=float(per.rho_eff.median()),
                     q=[float(x) for x in np.quantile(per.rho_eff, [0, .25, .5, .75, 1])],
                     n_LR_sig_0_05=int((per.rho_eff_p < 0.05).sum())),
        tau=dict(median_count=float(per.tau_count.median()),
                 median_eff=float(per.tau_eff.median()),
                 median_ml=float(per.tau_ml.median()),
                 n_ml_LR_gt_3_84=int((per.tau_ml_LR > 3.84).sum())),
        depth_within_gene=dict(n_genes_spearman_n_z2_pos=int((rho_within > 0).sum()),
                               sign_test_p=float(signtest),
                               median_spearman=float(rho_within.median())),
        depth_vs_inflation=decomp,
        within_gene_depth_quintiles=quint.reset_index().to_dict('records'),
        depth_bins=dbins.to_dict('records'),
        prediction=pred, rasqual=rasq, isolating_arms=arm_res,
        benchmark_matched_per_gene=bgene,
        R_below_1=below[['R', 'R_null_q025', 'R_null_p_lo', 'rho_count', 'rho_eff',
                         'tau_ml', 'sdratio_a']].reset_index().to_dict('records'),
        n_R_below_model_q025=int(len(below_sig)),
        n_R_above_model_q975=int((per.R > per.R_null_q975).sum()),
        runtime_s=round(time.time() - t0))
    (OUT / 'summary.json').write_text(json.dumps(summ, indent=2, default=float))
    print(json.dumps({k: summ[k] for k in ('rho_count', 'rho_eff', 'tau', 'depth_within_gene',
                                            'prediction', 'rasqual')}, indent=1, default=float))
    print(dbins.round(4).to_string())
    print(quint.round(3).to_string())
    print(per[['R', 'sdratio_a', 'rho_count', 'rho_eff', 'tau_count', 'tau_eff', 'tau_ml',
               'R_pred_count', 'R_pred_eff', 'top_share', 'top_bb_tail', 'rasqual_theta']]
          .sort_values('R').round(4).to_string())
    print(f'wrote {OUT} in {time.time() - t0:.0f} s')


if __name__ == '__main__':
    main()

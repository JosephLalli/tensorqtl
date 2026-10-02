"""
External benchmark: hapmixQTL against the RASQUAL / TReCASE generative model.

WHY THIS IS DIFFERENT FROM tests/ase_validation.py
--------------------------------------------------
Every tier in ase_validation.py simulates from hapmixQTL's OWN assumed model
(Gaussian `a` and `t`, with controlled violations). That can show a method is
inconsistent with its own assumptions, but it cannot show the assumptions are
wrong about real data -- the simulator and the estimator share a worldview.

This harness removes that circularity by generating data from the model the ASE
field actually uses, and which hapmixQTL does NOT assume:

    total counts    T_i ~ NegBinomial(mean = s_i * mu * f(g_i, kappa), disp phi)
    allele-specific y_i ~ BetaBinomial(n_i, pi = kappa/(1+kappa), overdisp rho)

with the standard TReC cis parameterization of the genotype effect

    f(g) = 1              for g = 0   (both haplotypes reference)
    f(g) = (1 + kappa)/2  for g = 1   (heterozygote)
    f(g) = kappa          for g = 2   (both haplotypes alternate)

kappa is the allelic fold change (log aFC = log kappa); kappa = 1 is the null,
where pi = 0.5. This is the model structure of Sun (2012, Biometrics) TReCASE and
Kumasaka et al. (2016, Nat Genet) RASQUAL, confirmed from sources reachable at
the time of writing.

SCOPE / HONESTY
---------------
This reproduces the published generative model STRUCTURE and implements the
published TESTS. It is NOT a claim to have replicated either paper's exact
simulation parameter grid: PMC, nature.com, biorxiv and the asSeq documentation
were all unreachable through this environment's egress proxy, so the specific
depths / overdispersions / effect sizes below are our own realistic RNA-seq
choices and are stated explicitly rather than inherited. The external validity
comes from the model family and the comparator tests, not from matching a table.

COMPARATORS (implemented here directly, no R or C dependency)
-------------------------------------------------------------
  TReC-only   negative-binomial GLM on total counts, LRT vs kappa = 1
  ASE-only    beta-binomial on allele-specific counts, LRT vs pi = 0.5
  TReCASE     joint likelihood sharing one kappa across both channels, LRT
  hapmixQTL   counts -> emulated Gibbs posterior -> the production default
              mode (tau_mode='zero' + se_mode='fitted': through-origin
              allelic channel, per-channel t references, the 15-donor
              allelic admission floor, Meier's correction of the combined
              SE, and the Welch-Satterthwaite reference of the combined p),
              on the shipped default's inputs (prepare_default_inputs: the
              half-read log CPM total, the zero-haplotype admission rule), at
              the simulated-effects benchmark's three weightings (2026-10-01):
              split (the shipped default: 1/Va allelic, unit total), gibbs
              (1/Va allelic, 1/Vt total, Vt from half_read_total_gibbs_variance)
              and unit (weight 1 in both channels, no Gibbs draws).
              Values are the simulated counts (the point estimates) and
              variances come from emulated draws, both in log2.
              Emulated draws: allelic yL ~ Binomial(n_as, frac) with
              yR = n_as - yL; total yT ~ Poisson(T), because gene-level Gibbs
              variance of a total is Poisson (docs/measurement_record.md, the RTA entry). With
              count_noise the delta-method counting term is added on top in
              both channels, as production adds it to Salmon draws that
              already carry shot noise.

RASQUAL's additions over TReCASE (genotype uncertainty, reference mapping bias
phi, sequencing error delta) are nuisance-parameter refinements on the same joint
likelihood; the joint test here is the TReCASE core they share. We do not claim
to implement RASQUAL itself.

METRICS
-------
  Null calibration: realized type-I error at alpha = 0.05 / 0.01, lambda_GC.
  Power at MATCHED EMPIRICAL type-I error -- each method is thresholded on its
  own simulated null, so an anticonservative method cannot win by being broken.

Run:  python3 tests/ase_external_benchmark.py --reps 600 --out bench.json
"""

import argparse
import json
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scipy import stats, optimize
from scipy.special import betaln, gammaln

try:
    from tensorqtl.hapmixqtl import (
        _prepare_channels, calculate_hapmixqtl_nominal, half_read_total_gibbs_variance, prepare_default_inputs)
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent / 'tensorqtl'))
    from hapmixqtl import (
        _prepare_channels, calculate_hapmixqtl_nominal, half_read_total_gibbs_variance, prepare_default_inputs)

import torch
DTYPE = torch.float64


# ---------------------------------------------------------------------------
#  Generative model (RASQUAL / TReCASE family)
# ---------------------------------------------------------------------------

def trec_multiplier(g, kappa):
    """Standard TReC cis parameterization of the genotype effect on the mean."""
    return np.where(g == 0, 1.0, np.where(g == 1, (1.0 + kappa) / 2.0, kappa))


def simulate_locus(N, rng, kappa=1.0, maf=0.3, mu=200.0, phi=0.2, rho=0.01,
                   as_frac=0.25, lib_sd=0.25):
    """
    Simulate one gene/locus under the NB + beta-binomial model.

    Args:
        kappa   : allelic fold change (1.0 = null)
        mu      : mean total fragment count per gene at reference genotype
        phi     : NB dispersion (Var = mu + phi*mu^2)
        rho     : beta-binomial overdispersion of the allelic ratio
        as_frac : fraction of a gene's reads that overlap a phasing-informative
                  het SNP and are therefore allele-specific
        lib_sd  : lognormal SD of per-sample library size offsets

    Returns dict with genotype, phase, total counts, AS counts, and the
    haplotype-level counts needed to drive hapmixQTL.
    """
    g = rng.binomial(2, maf, size=N)
    # phase: which haplotype (L=+1 / R=-1) carries the ALT allele, hets only
    s = np.zeros(N)
    het = g == 1
    s[het] = rng.choice([-1.0, 1.0], size=int(het.sum()))

    lib = np.exp(rng.normal(0, lib_sd, N))            # library-size offsets
    mean_t = lib * mu * trec_multiplier(g, kappa)

    # negative binomial total counts
    r = 1.0 / max(phi, 1e-8)
    p = r / (r + mean_t)
    T = rng.negative_binomial(r, p).astype(float)

    # allele-specific subset of the reads
    n_as = rng.binomial(T.astype(int), as_frac).astype(float)

    # beta-binomial allelic split. pi = P(read from the ALT haplotype).
    pi_alt = kappa / (1.0 + kappa)
    nu = (1.0 - rho) / max(rho, 1e-9)                 # BB precision
    y_alt = np.zeros(N)
    for i in range(N):
        if n_as[i] <= 0:
            continue
        if het[i]:
            a_, b_ = pi_alt * nu, (1 - pi_alt) * nu
        else:
            a_ = b_ = 0.5 * nu                        # homozygote: no imbalance
        pr = rng.beta(max(a_, 1e-6), max(b_, 1e-6))
        y_alt[i] = rng.binomial(int(n_as[i]), pr)

    # haplotype-level counts for hapmixQTL: L = haplotype carrying ALT when s=+1
    y_hi = y_alt                       # ALT-haplotype AS reads
    y_lo = n_as - y_alt
    yL = np.where(s >= 0, y_hi, y_lo)
    yR = np.where(s >= 0, y_lo, y_hi)

    return dict(g=g.astype(float), s=s, het=het, T=T, n_as=n_as,
                y_alt=y_alt, yL=yL, yR=yR, lib=lib)


# ---------------------------------------------------------------------------
#  Comparator 1: TReC-only  (negative binomial GLM, LRT)
# ---------------------------------------------------------------------------

def _nb_nll(params, T, g, lib, fix_kappa=None):
    b0 = params[0]
    logphi = params[-1]
    kappa = 1.0 if fix_kappa is not None else np.exp(params[1])
    phi = np.exp(logphi)
    mean = lib * np.exp(b0) * trec_multiplier(g, kappa)
    mean = np.clip(mean, 1e-8, None)
    r = 1.0 / max(phi, 1e-8)
    # NB log-lik in mean/dispersion form
    ll = (gammaln(T + r) - gammaln(r) - gammaln(T + 1)
          + r * np.log(r / (r + mean)) + T * np.log(mean / (r + mean)))
    return -np.sum(ll)


def trec_lrt(d):
    T, g, lib = d['T'], d['g'], d['lib']
    b0_init = np.log(max(T.mean() / lib.mean(), 1e-3))
    try:
        f0 = optimize.minimize(_nb_nll, [b0_init, np.log(0.2)],
                               args=(T, g, lib, True), method='L-BFGS-B',
                               bounds=[B0_B, LOGPHI_B])
        f1 = optimize.minimize(_nb_nll, [b0_init, 0.0, np.log(0.2)],
                               args=(T, g, lib, None), method='L-BFGS-B',
                               bounds=[B0_B, LOGK_B, LOGPHI_B])
        stat = 2 * (f0.fun - f1.fun)
    except Exception:
        return 1.0, 0.0
    if not np.isfinite(stat):
        return 1.0, 0.0
    stat = float(np.clip(stat, 0.0, 1e4))
    return float(stats.chi2.sf(stat, 1)), stat


# ---------------------------------------------------------------------------
#  Comparator 2: ASE-only  (beta-binomial, LRT)
# ---------------------------------------------------------------------------

# Bounds keep the likelihoods away from the boundary singularities that make an
# unconstrained fit blow up (pi -> 0/1 under perfect separation sends betaln to
# -inf and produces nonsense LRT statistics; observed as a null statistic of
# ~2.8e5 at N=50 before this was fixed).
LOGIT_B = (-6.0, 6.0)       # pi in ~(0.0025, 0.9975)
LOGNU_B = (np.log(1.5), np.log(1e6))
LOGPHI_B = (np.log(1e-4), np.log(50.0))
B0_B = (-5.0, 20.0)
LOGK_B = (np.log(0.05), np.log(20.0))


def _bb_nll(params, y, n, fix_null=False):
    nu = np.exp(np.clip(params[-1], *LOGNU_B))
    pi = 0.5 if fix_null else 1.0 / (1.0 + np.exp(-np.clip(params[0], *LOGIT_B)))
    a_, b_ = max(pi * nu, 1e-8), max((1 - pi) * nu, 1e-8)
    ll = betaln(y + a_, n - y + b_) - betaln(a_, b_)
    v = -np.sum(ll)
    return v if np.isfinite(v) else 1e12


def ase_lrt(d):
    """Uses hets only, with the ALT-haplotype count as the success."""
    m = d['het'] & (d['n_as'] > 0)
    if m.sum() < 5:
        return 1.0, 0.0
    y, n = d['y_alt'][m], d['n_as'][m]
    try:
        f0 = optimize.minimize(_bb_nll, [np.log(50.0)], args=(y, n, True),
                               method='L-BFGS-B', bounds=[LOGNU_B])
        best = None
        for start in (0.0, -0.5, 0.5):     # multi-start guards local optima
            f1 = optimize.minimize(_bb_nll, [start, np.log(50.0)], args=(y, n, False),
                                   method='L-BFGS-B', bounds=[LOGIT_B, LOGNU_B])
            if best is None or f1.fun < best.fun:
                best = f1
        stat = 2 * (f0.fun - best.fun)
    except Exception:
        return 1.0, 0.0
    if not np.isfinite(stat):
        return 1.0, 0.0
    stat = float(np.clip(stat, 0.0, 1e4))
    return float(stats.chi2.sf(stat, 1)), stat


# ---------------------------------------------------------------------------
#  Comparator 3: TReCASE joint  (one shared kappa across both channels)
# ---------------------------------------------------------------------------

def _joint_nll(params, d, null=False):
    b0 = params[0]
    if null:
        kappa = 1.0
        logphi, lognu = params[1], params[2]
    else:
        kappa = np.exp(np.clip(params[1], *LOGK_B))
        logphi, lognu = params[2], params[3]
    nll = _nb_nll(np.array([b0, np.log(kappa), logphi]), d['T'], d['g'], d['lib'],
                  fix_kappa=(True if null else None))
    m = d['het'] & (d['n_as'] > 0)
    if m.sum() >= 1:
        pi = kappa / (1.0 + kappa)
        nu = np.exp(np.clip(lognu, *LOGNU_B))
        a_, b_ = max(pi * nu, 1e-8), max((1 - pi) * nu, 1e-8)
        y, n = d['y_alt'][m], d['n_as'][m]
        nll = nll - np.sum(betaln(y + a_, n - y + b_) - betaln(a_, b_))
    return nll if np.isfinite(nll) else 1e12


def trecase_lrt(d):
    b0_init = np.log(max(d['T'].mean() / d['lib'].mean(), 1e-3))
    try:
        f0 = optimize.minimize(_joint_nll, [b0_init, np.log(0.2), np.log(50.0)],
                               args=(d, True), method='L-BFGS-B',
                               bounds=[B0_B, LOGPHI_B, LOGNU_B])
        best = None
        for start in (0.0, -0.3, 0.3):
            f1 = optimize.minimize(_joint_nll,
                                   [b0_init, start, np.log(0.2), np.log(50.0)],
                                   args=(d, False), method='L-BFGS-B',
                                   bounds=[B0_B, LOGK_B, LOGPHI_B, LOGNU_B])
            if best is None or f1.fun < best.fun:
                best = f1
        stat = 2 * (f0.fun - best.fun)
    except Exception:
        return 1.0, 0.0
    if not np.isfinite(stat):
        return 1.0, 0.0
    stat = float(np.clip(stat, 0.0, 1e4))
    return float(stats.chi2.sf(stat, 1)), stat


# ---------------------------------------------------------------------------
#  hapmixQTL on the same data
# ---------------------------------------------------------------------------

def emulated_summaries(d, rng, n_draws=80, kappa_pseudo=0.5):
    """Simulated counts as point estimates, emulated draws for their variance: (a, t, Va, Vt)
    in log2, a, t and Va from prepare_default_inputs (the shipped default; t the half-read log
    CPM) and Vt, which only the gibbs weighting uses, from half_read_total_gibbs_variance.
    eff_lib = lib x 1e6 reads."""
    N = len(d['g'])
    tot_as = d['yL'] + d['yR']
    frac = (d['yL'] + kappa_pseudo) / (tot_as + 2 * kappa_pseudo)
    yL = np.empty((1, N, n_draws)); yR = np.empty((1, N, n_draws))
    for i in range(N):
        n_i = int(tot_as[i])
        draw = rng.binomial(n_i, frac[i], size=n_draws).astype(float) if n_i > 0 \
            else np.zeros(n_draws)
        yL[0, i, :] = draw
        yR[0, i, :] = max(n_i, 0) - draw
    # drawn after the allelic draws, so those are the ones the harness always drew
    yT = rng.poisson(d['T'][:, None], size=(N, n_draws)).astype(float)[None]
    A, T_, Va, _ = prepare_default_inputs(d['yL'][None], d['yR'][None], d['T'][None], d['lib'] * 1e6, yL, yR)
    Vt = half_read_total_gibbs_variance(d['T'][None], d['lib'] * 1e6, yT)
    return A[0], T_[0], Va[0], Vt[0]


def weighted_variances(weighting, va, vt, eps=1e-12):
    """(allelic, total) working variances; benchmark/simulated_effects/common.config_variances
    (split = its 'hybrid', unit = its 'unit'). Excluded donors (Va = 0) stay at 0, so they stay excluded."""
    if weighting == 'gibbs':
        return va, vt
    if weighting == 'split':
        return va, np.ones_like(vt)
    if weighting == 'unit':
        return np.where(va > eps, 1.0, 0.0), np.ones_like(vt)
    raise ValueError(weighting)


def hapmix_pval(d, rng, weighting, return_info=False):
    """counts -> emulated Gibbs posterior -> production default-mode p (map_nominal's
    pval_nominal: the Meier-corrected combined t on the Welch-Satterthwaite dof)."""
    a, t, va, vt = emulated_summaries(d, rng)
    va, vt = weighted_variances(weighting, va, vt)
    g_t = torch.tensor(d['g'].reshape(1, -1), dtype=DTYPE)
    s_t = torch.tensor(d['s'].reshape(1, -1), dtype=DTYPE)
    a_t, t_t, va_t, vt_t = (torch.tensor(x, dtype=DTYPE) for x in (a, t, va, vt))
    # ase_covariates_t=None: the allelic channel is through-origin, as in production
    sqrt_wa, sqrt_wt, res_a, res_t = _prepare_channels(
        a_t, t_t, va_t, vt_t, None, 'zero', 'cpu', ase_covariates_t=None, fitted_scale=True)
    res = calculate_hapmixqtl_nominal(g_t, s_t, a_t, t_t, sqrt_wa, sqrt_wt, res_a, res_t,
                                      fitted=True, return_info=True)
    tstat, dof = float(res[0].numpy()[0]), float(res[7]['dof_nominal'].numpy()[0])
    if not (np.isfinite(tstat) and np.isfinite(dof)):
        out = (1.0, 0.0)
    else:
        p = float(2 * stats.t.cdf(-abs(tstat), dof))       # tensorqtl.core.get_t_pval
        out = (p, float(stats.chi2.isf(p, 1)))              # 1-df chi-square equivalent of p
    if not return_info:
        return out
    return out + (dict(n_a=int((sqrt_wa != 0).sum()), dof_nominal=dof,
                       allelic_admitted=bool(res[7]['allelic_admitted']),
                       meier=float(res[7]['meier_factor'].numpy()[0])),)


METHODS = {
    'TReC-only': lambda d, rng: trec_lrt(d),
    'ASE-only': lambda d, rng: ase_lrt(d),
    'TReCASE (joint)': lambda d, rng: trecase_lrt(d),
    # default mode (tau_mode='zero' + se_mode='fitted') at the simulated-effects benchmark's
    # three weightings (docs/pipeline_rules.md, "Decision, 2026-10-01")
    'hapmixQTL gibbs': lambda d, rng: hapmix_pval(d, rng, 'gibbs'),
    'hapmixQTL split': lambda d, rng: hapmix_pval(d, rng, 'split'),
    'hapmixQTL unit': lambda d, rng: hapmix_pval(d, rng, 'unit'),
}


# ---------------------------------------------------------------------------

def run(reps, N, kappa, seed0, **sim_kw):
    """Return {method: (pvals, stats)} over `reps` simulated loci."""
    out = {k: ([], []) for k in METHODS}
    for r in range(reps):
        rng = np.random.RandomState(seed0 + r)
        d = simulate_locus(N, rng, kappa=kappa, **sim_kw)
        for name, fn in METHODS.items():
            p, s = fn(d, np.random.RandomState(seed0 + 900000 + r))
            out[name][0].append(p); out[name][1].append(s)
    return {k: (np.array(v[0]), np.array(v[1])) for k, v in out.items()}


def calib(p):
    p = p[np.isfinite(p)]
    chi2 = stats.chi2.isf(np.clip(p, 1e-300, 1), 1)
    return dict(n=int(p.size),
                t05=float(np.mean(p < 0.05)), t01=float(np.mean(p < 0.01)),
                lam=float(np.median(chi2) / stats.chi2.ppf(0.5, 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reps', type=int, default=600)
    ap.add_argument('--N', type=int, default=200)
    ap.add_argument('--mu', type=float, default=200.0)
    ap.add_argument('--phi', type=float, default=0.2)
    ap.add_argument('--rho', type=float, default=0.01)
    ap.add_argument('--as_frac', type=float, default=0.25)
    ap.add_argument('--kappas', type=str, default='1.25,1.5')
    ap.add_argument('--out', type=str, default=None)
    args = ap.parse_args()
    sim_kw = dict(mu=args.mu, phi=args.phi, rho=args.rho, as_frac=args.as_frac)
    res = {'config': vars(args)}

    print(f"\n=== NULL (kappa=1) : calibration on the NB + beta-binomial model ===")
    print(f"    N={args.N}  mu={args.mu}  NB disp={args.phi}  BB overdisp={args.rho}  "
          f"AS frac={args.as_frac}  reps={args.reps}")
    null = run(args.reps, args.N, 1.0, 1000, **sim_kw)
    res['null'] = {}
    for name, (p, s) in null.items():
        c = calib(p); res['null'][name] = c
        print(f"  {name:26s} typeI@0.05={c['t05']:.4f} ({c['t05']/0.05:5.2f}x)  "
              f"@0.01={c['t01']:.4f} ({c['t01']/0.01:6.2f}x)  lambda={c['lam']:.2f}")

    # empirical thresholds from each method's OWN null
    thr = {name: np.quantile(s[np.isfinite(s)], 0.95) for name, (p, s) in null.items()}

    res['power'] = {}
    for kappa in [float(x) for x in args.kappas.split(',')]:
        print(f"\n=== POWER at kappa={kappa} (log aFC={np.log(kappa):.3f}), "
              f"matched empirical alpha=0.05 ===")
        alt = run(args.reps, args.N, kappa, 5000, **sim_kw)
        res['power'][str(kappa)] = {}
        for name, (p, s) in alt.items():
            pw = float(np.mean(s[np.isfinite(s)] > thr[name]))
            nom = float(np.mean(p < 0.05))
            res['power'][str(kappa)][name] = dict(matched=pw, nominal=nom)
            print(f"  {name:26s} power(matched)={pw:.3f}   [uncorrected nominal={nom:.3f}]")

    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == '__main__':
    main()

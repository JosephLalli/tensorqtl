"""
hapmixQTL's additive tau vs RASQUAL's shared theta -- a real head-to-head.

MOTIVATION
----------
RASQUAL's Fig. 3e ablation (gEUVADIS, N=25, power at FPR 10%) decomposes its own
advantage:

    original RASQUAL                          35.9%
    fixed phi = 0.5   (no reference bias)     36.8%   <- contributes nothing
    fixed delta = 0.01 (no sequencing error)  32.3%   <- 3.6 pts
    fixed genotype likelihood                 33.5%   <- 2.4 pts
    Poisson-binomial (no overdispersion)      15.0%   <- 20.9 pts

and the text: "power and fine-mapping were mostly influenced by better estimation
of overdispersion and by genotype correction ... reference bias had a minor
impact". But TReCASE ALSO models overdispersion, so the 20.9-point term is not
the RASQUAL-over-TReCASE gap. RASQUAL's remaining edge over TReCASE is
attributed to *better estimation* of overdispersion, via

    "the use of a single overdispersion parameter shared across the
     between-individual and allele-specific model components to further improve
     model stability"

This harness tests whether that shared-theta trick actually buys power, and
whether hapmixQTL's additive tau matches it.

THE SHARED-THETA CONSTRUCTION
-----------------------------
RASQUAL's supplement derives both components from one gamma-Poisson process: if
the two haplotype counts are Gamma-Poisson with shapes (alpha, beta), then the
TOTAL is negative binomial with shape alpha+beta and the CONDITIONAL allelic
split is beta-binomial with parameters (alpha, beta). So with a single theta:

    NB shape       r  = 1/theta
    BB precision   nu = 1/theta          (nu = alpha + beta)
    alpha = pi/theta,  beta = (1-pi)/theta

One parameter, both components -- fewer degrees of freedom and a shared estimate,
which is exactly the stability claim. TReCASE instead fits an NB dispersion and a
beta-binomial overdispersion SEPARATELY.

Nuisance terms (RASQUAL's multiplicative model, Supp. Fig. 27):
    sequencing/mapping error delta:  pi_err = (1-delta)*pi + delta*(1-pi)
    reference bias phi:              pi_obs = (1-phi)*pi_err /
                                              [(1-phi)*pi_err + phi*(1-pi_err)]
(phi = 0.5 is unbiased; phi > 0.5 favours the reference allele.)

METHODS COMPARED
----------------
    trcQTL                 NB GLM on total counts (LRT)
    TReCASE                joint, SEPARATE NB and BB dispersions (LRT)
    TReCASE shared-theta   joint, ONE shared theta -- isolates the trick
    RASQUAL-like           shared theta + fitted delta + fitted phi
    hapmixQTL tau=estimate additive tau on the Gibbs variance

Power is compared at MATCHED EMPIRICAL FPR (RASQUAL's own metric), so no method
can win by being anticonservative.

Run:  python3 tests/ase_rasqual_comparison.py --reps 400
"""

import argparse
import json
import sys
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from scipy import stats, optimize
from scipy.special import betaln, gammaln

from ase_external_benchmark import (
    trec_lrt, trecase_lrt, hapmix_pval, trec_multiplier,
    LOGIT_B, LOGNU_B, LOGPHI_B, B0_B, LOGK_B, DTYPE)

LOGTH_B = (np.log(1e-4), np.log(20.0))
DELTA_B = (-8.0, -0.5)     # logit delta: delta in ~(0.0003, 0.38)
PHI_B = (-2.0, 2.0)        # logit phi:   phi in ~(0.12, 0.88); 0 => phi=0.5


def _sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def observed_pi(pi, delta, phi):
    """RASQUAL's multiplicative distortion of the true allelic ratio."""
    pe = (1 - delta) * pi + delta * (1 - pi)
    num = (1 - phi) * pe
    return num / np.clip(num + phi * (1 - pe), 1e-12, None)


# ---------------------------------------------------------------------------
#  Generative model: ONE shared theta, plus delta and phi
# ---------------------------------------------------------------------------

def simulate_rasqual(N, rng, kappa=1.0, maf=0.3, mu=200.0, theta=0.2,
                     delta=0.005, phi=0.5, as_frac=0.25, lib_sd=0.25,
                     geno_err=0.0):
    g_true = rng.binomial(2, maf, size=N)
    g = g_true.copy()
    if geno_err > 0:                       # genotyping error -> hard-call noise
        flip = rng.rand(N) < geno_err
        g[flip] = rng.binomial(2, maf, size=int(flip.sum()))
    s = np.zeros(N)
    het = g_true == 1
    s[het] = rng.choice([-1.0, 1.0], size=int(het.sum()))

    lib = np.exp(rng.normal(0, lib_sd, N))
    mean_t = lib * mu * trec_multiplier(g_true, kappa)
    r = 1.0 / max(theta, 1e-8)             # NB shape = 1/theta
    T = rng.negative_binomial(r, r / (r + mean_t)).astype(float)
    n_as = rng.binomial(T.astype(int), as_frac).astype(float)

    pi_alt = kappa / (1.0 + kappa)
    y_alt = np.zeros(N)
    for i in range(N):
        if n_as[i] <= 0:
            continue
        pi_i = pi_alt if het[i] else 0.5
        po = observed_pi(pi_i, delta, phi)
        a_ = max(po / theta, 1e-6); b_ = max((1 - po) / theta, 1e-6)
        y_alt[i] = rng.binomial(int(n_as[i]), rng.beta(a_, b_))

    y_hi, y_lo = y_alt, n_as - y_alt
    yL = np.where(s >= 0, y_hi, y_lo); yR = np.where(s >= 0, y_lo, y_hi)
    return dict(g=g.astype(float), s=s, het=het, T=T, n_as=n_as,
                y_alt=y_alt, yL=yL, yR=yR, lib=lib)


# ---------------------------------------------------------------------------
#  Shared-theta joint likelihood (optionally fitting delta and phi)
# ---------------------------------------------------------------------------

def _shared_nll(params, d, null=False, fit_delta=False, fit_phi=False):
    i = 0
    b0 = params[i]; i += 1
    if null:
        kappa = 1.0
    else:
        kappa = np.exp(np.clip(params[i], *LOGK_B)); i += 1
    theta = np.exp(np.clip(params[i], *LOGTH_B)); i += 1
    delta = _sig(np.clip(params[i], *DELTA_B)) if fit_delta else 0.0
    i += 1 if fit_delta else 0
    phi = _sig(np.clip(params[i], *PHI_B)) if fit_phi else 0.5
    i += 1 if fit_phi else 0

    T, g, lib = d['T'], d['g'], d['lib']
    mean = np.clip(lib * np.exp(b0) * trec_multiplier(g, kappa), 1e-8, None)
    r = 1.0 / max(theta, 1e-8)             # shared: NB shape = 1/theta
    ll = np.sum(gammaln(T + r) - gammaln(r) - gammaln(T + 1)
                + r * np.log(r / (r + mean)) + T * np.log(mean / (r + mean)))

    m = d['het'] & (d['n_as'] > 0)
    if m.sum() >= 1:
        pi = kappa / (1.0 + kappa)
        po = observed_pi(pi, delta, phi)
        a_ = max(po / theta, 1e-8); b_ = max((1 - po) / theta, 1e-8)
        y, n = d['y_alt'][m], d['n_as'][m]
        ll += np.sum(betaln(y + a_, n - y + b_) - betaln(a_, b_))
    return -ll if np.isfinite(ll) else 1e12


def shared_theta_lrt(d, fit_delta=False, fit_phi=False):
    b0 = np.log(max(d['T'].mean() / d['lib'].mean(), 1e-3))
    extra, xb = [], []
    if fit_delta:
        extra.append(-5.0); xb.append(DELTA_B)
    if fit_phi:
        extra.append(0.0); xb.append(PHI_B)
    try:
        f0 = optimize.minimize(
            _shared_nll, [b0, np.log(0.2)] + extra,
            args=(d, True, fit_delta, fit_phi), method='L-BFGS-B',
            bounds=[B0_B, LOGTH_B] + xb)
        best = None
        for st in (0.0, -0.3, 0.3):
            f1 = optimize.minimize(
                _shared_nll, [b0, st, np.log(0.2)] + extra,
                args=(d, False, fit_delta, fit_phi), method='L-BFGS-B',
                bounds=[B0_B, LOGK_B, LOGTH_B] + xb)
            if best is None or f1.fun < best.fun:
                best = f1
        stat = 2 * (f0.fun - best.fun)
    except Exception:
        return 1.0, 0.0
    if not np.isfinite(stat):
        return 1.0, 0.0
    stat = float(np.clip(stat, 0.0, 1e4))
    return float(stats.chi2.sf(stat, 1)), stat


METHODS = {
    'trcQTL (total only)': lambda d, r: trec_lrt(d),
    'TReCASE (separate disp.)': lambda d, r: trecase_lrt(d),
    'TReCASE shared-theta': lambda d, r: shared_theta_lrt(d),
    'RASQUAL-like (+delta,phi)': lambda d, r: shared_theta_lrt(d, True, True),
    "hapmixQTL tau='estimate'": lambda d, r: hapmix_pval(d, r, 'estimate'),
}


def run(reps, N, kappa, seed0, **kw):
    out = {k: [] for k in METHODS}
    for r in range(reps):
        rng = np.random.RandomState(seed0 + r)
        d = simulate_rasqual(N, rng, kappa=kappa, **kw)
        for name, fn in METHODS.items():
            _, s = fn(d, np.random.RandomState(seed0 + 640000 + r))
            out[name].append(s)
    return {k: np.array(v) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reps', type=int, default=400)
    ap.add_argument('--N', type=int, default=100)
    ap.add_argument('--theta', type=float, default=0.2)
    ap.add_argument('--kappas', type=str, default='1.1,1.25')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    kappas = [float(x) for x in args.kappas.split(',')]
    res = {'config': vars(args), 'scenarios': {}}

    # scenario grid: how large are RASQUAL's nuisance terms?
    scenarios = [
        ('clean (delta=0, phi=0.5)', dict(delta=0.0, phi=0.5)),
        ('sequencing error delta=0.02', dict(delta=0.02, phi=0.5)),
        ('reference bias phi=0.60', dict(delta=0.0, phi=0.60)),
        ('both (delta=0.02, phi=0.60)', dict(delta=0.02, phi=0.60)),
    ]

    for label, kw in scenarios:
        kw = dict(kw, theta=args.theta)
        print(f"\n{'='*78}\n{label}   (N={args.N}, theta={args.theta}, "
              f"{args.reps} loci)\n{'='*78}")
        null = run(args.reps, args.N, 1.0, 11_000, **kw)
        thr = {k: np.quantile(v[np.isfinite(v)], 0.90) for k, v in null.items()}
        print("  power at matched empirical FPR = 10% (RASQUAL's metric)")
        hdr = f"  {'aFC':>5s} | " + " | ".join(f"{k[:25]:>25s}" for k in METHODS)
        print(hdr); print("  " + "-" * (len(hdr) - 2))
        res['scenarios'][label] = {}
        for kappa in kappas:
            alt = run(args.reps, args.N, kappa, 77_000, **kw)
            row = {}
            for k in METHODS:
                v = alt[k][np.isfinite(alt[k])]
                row[k] = float(np.mean(v > thr[k])) if v.size else float('nan')
            res['scenarios'][label][str(kappa)] = row
            print(f"  {kappa:5.2f} | " + " | ".join(f"{row[k]:25.3f}" for k in METHODS))
        # nominal type-I under the null, before matching
        nom = {}
        for k, v in null.items():
            v = v[np.isfinite(v)]
            p = (stats.chi2.sf(v, 1) if 'hapmix' not in k
                 else 2 * stats.t.sf(np.sqrt(np.clip(v, 0, None)), args.N - 2))
            nom[k] = float(np.mean(p < 0.05))
        res['scenarios'][label]['nominal_typeI'] = nom
        print("  nominal type-I @0.05: " +
              "  ".join(f"{k.split()[0]}={v:.3f}" for k, v in nom.items()))

    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == '__main__':
    main()

"""
The cis/trans consistency test for hapmixQTL -- a diagnostic the pipeline lacks.

WHY THIS MATTERS (not merely a "nice extra capability")
-------------------------------------------------------
Sun (2012, Biometrics) and its descendants formalize the cis/trans distinction by
letting the two channels have their OWN effect sizes and testing whether they
agree. In the notation of Little et al. (2023, Nat Commun 14:3030, CSeQTL, Methods
"cis/trans eQTL testing"):

    eta^(T)  = eQTL effect estimated from TReC   (total read count)
    eta^(A)  = eQTL effect estimated from ASReC  (allele-specific read count)
    eta^(A)  = eta^(T) * alpha
    cis  <=>  alpha = 1                    H0: alpha = 1  vs  HA: alpha != 1

hapmixQTL's inverse-variance meta-analysis **assumes alpha = 1** -- that is the
whole justification for putting the two channels on a common log-aFC scale (the
g/2 predictor) and averaging them. It never tests that assumption. When alpha != 1
the meta-analysis is averaging two DIFFERENT estimands, which biases the combined
slope and invalidates its SE, silently.

alpha != 1 is not exotic. It arises from a trans component acting on total
expression only, from reference mapping bias attenuating the ASE channel, from
systematic phasing error, and from feature-level misquantification.

THE TEST
--------
Because Tier 0b (docs/ase_validation.md sec 3) established that the two slope
estimators are UNCORRELATED -- the ASE predictor s is orthogonal to the total
predictor g/2 under random phase -- the difference has variance se_a^2 + se_t^2
with no covariance term, so a simple Wald test is valid:

    T = (beta_a - beta_t) / sqrt(se_a^2 + se_t^2)   ~  N(0,1) under H0 (cis)

That orthogonality result is what makes this cheap. It is the same fact that let
us retire the unused `Cat` covariance.

Run:  python3 tests/ase_cis_trans_test.py --reps 800
"""

import argparse
import json
import sys
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from scipy import stats
from ase_validation import run_association

DTYPE = torch.float64


def simulate_two_channel(N, rng, beta_a=0.0, beta_t=0.0, maf=0.35,
                         v_lo=0.05, v_hi=0.30, sigma_bio=0.3):
    """
    Simulate with SEPARATELY controllable channel effects.

    beta_a == beta_t  -> alpha = 1 -> a true cis effect (hapmixQTL's assumption)
    beta_a != beta_t  -> alpha != 1 -> the assumption is violated
    """
    g = rng.binomial(2, maf, size=(1, N)).astype(np.float64)
    s = np.zeros((1, N))
    het = g[0] == 1
    s[0, het] = rng.choice([-1.0, 1.0], size=int(het.sum()))
    va = rng.uniform(v_lo, v_hi, N)
    vt = rng.uniform(v_lo, v_hi, N)
    a = np.sqrt(va) * rng.randn(N) + rng.normal(0, sigma_bio, N)
    t = 2.0 + np.sqrt(vt) * rng.randn(N) + rng.normal(0, sigma_bio, N)
    a = a + beta_a * s[0]
    t = t + beta_t * (g[0] / 2.0)
    return g, s, a, t, va, vt


def cis_trans_stat(slope_a, se_a, slope_t, se_t):
    """Wald statistic for H0: the two channels estimate the same effect."""
    if not (np.isfinite(se_a) and np.isfinite(se_t)):
        return np.nan, np.nan
    v = se_a ** 2 + se_t ** 2
    if v <= 0:
        return np.nan, np.nan
    z = (slope_a - slope_t) / np.sqrt(v)
    return float(z), float(2 * stats.norm.sf(abs(z)))


def run(reps, N, beta_a, beta_t, tau_mode='estimate', seed0=0, sigma_bio=0.3):
    z, p, comb, sa, st = [], [], [], [], []
    for r in range(reps):
        rng = np.random.RandomState(seed0 + r)
        g, s, a, t, va, vt = simulate_two_channel(
            N, rng, beta_a=beta_a, beta_t=beta_t, sigma_bio=sigma_bio)
        _, slope, se, slope_t, se_t, slope_a, se_a = run_association(
            g, s, a, t, va, vt, tau_mode=tau_mode)
        zz, pp = cis_trans_stat(float(slope_a[0]), float(se_a[0]),
                                float(slope_t[0]), float(se_t[0]))
        if np.isnan(zz):
            continue
        z.append(zz); p.append(pp); comb.append(float(slope[0]))
        sa.append(float(slope_a[0])); st.append(float(slope_t[0]))
    return (np.array(z), np.array(p), np.array(comb),
            np.array(sa), np.array(st))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reps', type=int, default=800)
    ap.add_argument('--N', type=int, default=200)
    ap.add_argument('--out', type=str, default=None)
    args = ap.parse_args()
    res = {'config': vars(args), 'calibration': {}, 'power': {}}

    print("=== 1. Calibration of the cis/trans test under TRUE cis (alpha = 1) ===")
    print("    (the test itself must be valid before it can diagnose anything)")
    for b in (0.0, 0.3, 0.6):
        z, p, comb, sa, st = run(args.reps, args.N, b, b)
        c = dict(n=int(p.size), t05=float(np.mean(p < 0.05)),
                 t01=float(np.mean(p < 0.01)),
                 z_mean=float(z.mean()), z_sd=float(z.std()))
        res['calibration'][str(b)] = c
        print(f"  beta_a = beta_t = {b:.1f} | type-I@0.05={c['t05']:.4f} "
              f"({c['t05']/0.05:.2f}x)  @0.01={c['t01']:.4f} ({c['t01']/0.01:.2f}x)  "
              f"z: mean={c['z_mean']:+.3f} sd={c['z_sd']:.3f}")

    print("\n=== 2. Power to DETECT a violation, and the harm of not testing ===")
    print("    beta_t fixed at 0.4; beta_a varied so alpha = beta_a/beta_t != 1.")
    print(f"    {'alpha':>6s} | {'detect rate':>11s} | {'combined slope':>14s} | "
          f"{'bias vs beta_t':>14s}")
    print("    " + "-" * 56)
    bt = 0.4
    for alpha in (1.0, 0.75, 0.5, 0.25, 0.0, -0.5):
        ba = alpha * bt
        z, p, comb, sa, st = run(args.reps, args.N, ba, bt, seed0=7000)
        row = dict(alpha=alpha, beta_a=ba, beta_t=bt,
                   detect=float(np.mean(p < 0.05)),
                   combined=float(comb.mean()),
                   bias_vs_beta_t=float(comb.mean() - bt),
                   slope_a=float(sa.mean()), slope_t=float(st.mean()))
        res['power'][str(alpha)] = row
        print(f"    {alpha:6.2f} | {row['detect']:11.3f} | {row['combined']:14.4f} | "
              f"{row['bias_vs_beta_t']:+14.4f}")

    print("\n    (combined slope should equal beta_t = 0.400 if the two channels")
    print("     agreed; the bias column is what hapmixQTL reports silently today)")

    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == '__main__':
    main()

"""
hapmixQTL under the PUBLISHED simulation designs of mixQTL and RASQUAL.

This supersedes the parameter choices in ase_external_benchmark.py, which were
our own because the papers were unreachable at the time. With the papers in hand
the designs are now taken from source:

mixQTL (Liang et al. 2021, Nat Commun 12:1424), Results + Supplementary Notes 6
  - allelic fold change grid: 1, 1.01, 1.05, 1.1, 1.25, 1.5, 2, 3   (aFC = kappa)
  - 200 replicates per setting
  - library size ~94e6 reads (GTEx v8 scale); expression 1-50 reads per million
  - comparators: mixQTL vs ascQTL (AS only) vs trcQTL (total only)
  - implementation defaults (github.com/liangyy/mixqtl, R/mixqtl.R):
        trc_cutoff = 20, asc_cutoff = 5, asc_cap = 5000, weight_cap = 100
        "the maximum weight difference (in fold) is min(weight_cap,
         floor(sample_size / 10)); the ones exceeding the cutoff are capped"

RASQUAL (Kumasaka et al. 2016, Nat Genet 48:206), Supplementary Note
"Generation and analysis of simulation data" + Supplementary Fig. 7
  - sample sizes N = 5, 10, 25, 50, 100
  - genetic effect pi; pi = 0.5 under the null, pi = pi_hat under the alternative
  - total counts NB, AS counts beta-binomial, sharing one overdispersion theta
  - model parameters drawn from EMPIRICAL distributions estimated from real data
  - power reported as Power @ empirical FPR = 10%, where the empirical null comes
    from PERMUTING the simulated data (their Observed Power / Observed FPR)
  - published simulation power @ FPR 10%:
        N=5 15.7%, N=10 24.7%, N=25 35.5%, N=50 46.3%, N=100 55.9%
  - published real-data power @ FPR 10% (eQTL, 25 EUR):
        RASQUAL 42.2%, CHT 35.7%, TReCASE 35.5%, Lm 25.7%

THE KEY MODELING FINDING THIS SCRIPT TESTS
------------------------------------------
mixQTL's error terms are

    eps_asc ~ N(0, sigma^2 * (1/Y1 + 1/Y2))        (Eq. 3, main text)
    z_tilde ~ N(0, sigma0_tilde^2)                 (Eq. 4, main text)

The COUNTS set only the SHAPE of the weights; sigma^2 and sigma0_tilde^2 are FREE
SCALE PARAMETERS, and Supplementary Notes 5.2 ("Inferring sigma0_tilde^2 and
sigma^2") solves for both from the data under a mixed/random-effect model, using
the R package EMMA.

hapmixQTL replaced that count-derived, freely-scaled variance with the Gibbs
inferential variance v_inf treated as fully KNOWN -- i.e. it dropped the free
scale entirely. That is precisely the tau_mode='zero' defect documented in
docs/ase_validation.md. So the bug is a deviation from the parent method, and
tau_mode='estimate' is a re-derivation of what mixQTL always did.

Note the two are not identical, and the difference matters:
    mixQTL     Var = sigma^2 * (1/Y1 + 1/Y2)     multiplicative rescaling
    hapmixQTL  Var = v_inf + tau                  additive offset
A multiplicative scale says "the shape is right, the scale is wrong" (good for
misspecified quantification noise); an additive offset says "there is an extra
independent component" (good for biological variance, which does not shrink with
read depth). This script also evaluates the nesting model Var = sigma^2*v_inf +
tau, which contains both, and mixQTL's weight cap as a third, cheaper mitigation.

Run:  python3 tests/ase_published_designs.py --reps 200 --out published.json
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

from ase_external_benchmark import (
    simulate_locus, trec_lrt, ase_lrt, trecase_lrt, DTYPE)

try:
    from tensorqtl.hapmixqtl import (
        WeightedResidualizer, _estimate_tau,
        calculate_hapmixqtl_nominal, compute_summaries_from_gibbs)
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent / 'tensorqtl'))
    from hapmixqtl import (
        WeightedResidualizer, _estimate_tau,
        calculate_hapmixqtl_nominal, compute_summaries_from_gibbs)


# mixQTL implementation defaults (R/mixqtl.R)
TRC_CUTOFF, ASC_CUTOFF, ASC_CAP, WEIGHT_CAP = 20, 5, 5000, 100

# mixQTL's published allelic-fold-change grid
MIXQTL_AFC = [1.0, 1.01, 1.05, 1.1, 1.25, 1.5, 2.0, 3.0]
# RASQUAL's published sample sizes (5 and 10 are below what a GLS can support;
# kept in the list so the omission is explicit rather than silent)
RASQUAL_N = [25, 50, 100]


def _apply_weight_cap(w, N, weight_cap=WEIGHT_CAP):
    """mixQTL's guardrail: cap the max/min weight ratio at min(cap, floor(N/10)).

    hapmixQTL has no equivalent, which is what lets an underestimated v_inf
    produce unboundedly large effective precision.
    """
    fold = min(weight_cap, max(np.floor(N / 10.0), 1.0))
    lo = w.min()
    return np.clip(w, lo, lo * fold)


def hapmix_variants(d, rng, mode, n_draws=80, kappa_pseudo=0.5):
    """
    mode: 'zero'      w = 1/v_inf                       (current default)
          'estimate'  w = 1/(v_inf + tau)               (additive, current fix)
          'capped'    w = 1/v_inf then mixQTL weight cap
          'nested'    w = 1/(sigma^2 * v_inf + tau)     (nests both)
    """
    N = len(d['g'])
    tot_as = d['yL'] + d['yR']
    frac = (d['yL'] + kappa_pseudo) / (tot_as + 2 * kappa_pseudo)
    yL = np.empty((1, N, n_draws)); yR = np.empty((1, N, n_draws))
    for i in range(N):
        n_i = int(tot_as[i])
        dr = rng.binomial(n_i, frac[i], size=n_draws).astype(float) if n_i > 0 \
            else np.zeros(n_draws)
        yL[0, i, :] = dr; yR[0, i, :] = max(n_i, 0) - dr
    A, T_, Va, Vt, _ = compute_summaries_from_gibbs(yL, yR)
    a = A[0]
    va = np.clip(Va[0], 1e-8, None); vt = np.clip(Vt[0], 1e-8, None)
    t = np.log((d['T'] / d['lib']) + 1.0)

    a_t = torch.tensor(a, dtype=DTYPE); t_t = torch.tensor(t, dtype=DTYPE)
    va_t = torch.tensor(va, dtype=DTYPE); vt_t = torch.tensor(vt, dtype=DTYPE)

    if mode == 'zero':
        wa, wt = 1.0 / va, 1.0 / vt
    elif mode == 'capped':
        wa = _apply_weight_cap(1.0 / va, N); wt = _apply_weight_cap(1.0 / vt, N)
    elif mode == 'estimate':
        ta = float(_estimate_tau(a_t, va_t, None, 'cpu'))
        tt = float(_estimate_tau(t_t, vt_t, None, 'cpu'))
        wa, wt = 1.0 / (va + ta), 1.0 / (vt + tt)
    elif mode == 'nested':
        # profile a multiplicative scale on top of the additive offset:
        # sigma^2 chosen so the whitened residuals have unit variance.
        ta = float(_estimate_tau(a_t, va_t, None, 'cpu'))
        tt = float(_estimate_tau(t_t, vt_t, None, 'cpu'))
        for v, tau, y in ((va, ta, a), (vt, tt, t)):
            pass
        s2a = max(np.var(a - a.mean()) / max(np.mean(va + ta), 1e-9), 1e-6)
        s2t = max(np.var(t - t.mean()) / max(np.mean(vt + tt), 1e-9), 1e-6)
        wa = 1.0 / (s2a * va + ta); wt = 1.0 / (s2t * vt + tt)
    else:
        raise ValueError(mode)

    sqrt_wa = torch.tensor(np.sqrt(wa), dtype=DTYPE)
    sqrt_wt = torch.tensor(np.sqrt(wt), dtype=DTYPE)
    g_t = torch.tensor(d['g'].reshape(1, -1), dtype=DTYPE)
    s_t = torch.tensor(d['s'].reshape(1, -1), dtype=DTYPE)
    res_a = WeightedResidualizer(None, sqrt_wa)
    res_t = WeightedResidualizer(None, sqrt_wt)
    ts, *_ = calculate_hapmixqtl_nominal(g_t, s_t, a_t, t_t,
                                         sqrt_wa, sqrt_wt, res_a, res_t)
    tst = float(ts.numpy()[0])
    return (1.0, 0.0) if not np.isfinite(tst) else \
        (float(2 * stats.t.sf(abs(tst), N - 2)), tst ** 2)


METHODS = {
    'trcQTL (total only)': lambda d, r: trec_lrt(d),
    'ascQTL (AS only)': lambda d, r: ase_lrt(d),
    'TReCASE (joint)': lambda d, r: trecase_lrt(d),
    "hapmix tau='zero'": lambda d, r: hapmix_variants(d, r, 'zero'),
    "hapmix +mixQTL weight cap": lambda d, r: hapmix_variants(d, r, 'capped'),
    "hapmix tau='estimate'": lambda d, r: hapmix_variants(d, r, 'estimate'),
    "hapmix nested scale+offset": lambda d, r: hapmix_variants(d, r, 'nested'),
}


def run_setting(reps, N, kappa, seed0, **kw):
    out = {k: [] for k in METHODS}
    for r in range(reps):
        rng = np.random.RandomState(seed0 + r)
        d = simulate_locus(N, rng, kappa=kappa, **kw)
        if d['T'].mean() < TRC_CUTOFF:      # mixQTL trc_cutoff
            continue
        for name, fn in METHODS.items():
            _, s = fn(d, np.random.RandomState(seed0 + 777000 + r))
            out[name].append(s)
    return {k: np.array(v) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reps', type=int, default=200)      # mixQTL: 200 replicates
    ap.add_argument('--mu', type=float, default=200.0)
    ap.add_argument('--phi', type=float, default=0.2)
    ap.add_argument('--rho', type=float, default=0.01)
    ap.add_argument('--as_frac', type=float, default=0.25)
    ap.add_argument('--Ns', type=str, default=','.join(map(str, RASQUAL_N)))
    ap.add_argument('--out', type=str, default=None)
    args = ap.parse_args()
    kw = dict(mu=args.mu, phi=args.phi, rho=args.rho, as_frac=args.as_frac)
    Ns = [int(x) for x in args.Ns.split(',')]
    res = {'config': vars(args), 'design': 'mixQTL aFC grid x RASQUAL N grid',
           'results': {}}

    for N in Ns:
        print(f"\n{'='*74}\nN = {N}   (RASQUAL sample-size grid; "
              f"power reported at empirical FPR = 10%, RASQUAL's metric)\n{'='*74}")
        null = run_setting(args.reps, N, 1.0, 10_000 + N, **kw)
        # RASQUAL's empirical null: threshold at the 90th pct of the null statistic
        thr = {k: np.quantile(v[np.isfinite(v)], 0.90) for k, v in null.items()}
        res['results'][str(N)] = {}

        hdr = f"  {'aFC':>6s} | " + " | ".join(f"{k[:22]:>22s}" for k in METHODS)
        print(hdr); print("  " + "-" * (len(hdr) - 2))
        for kappa in MIXQTL_AFC:
            if kappa == 1.0:
                continue
            alt = run_setting(args.reps, N, kappa, 50_000 + N * 100, **kw)
            row = {}
            for k in METHODS:
                v = alt[k][np.isfinite(alt[k])]
                row[k] = float(np.mean(v > thr[k])) if v.size else float('nan')
            res['results'][str(N)][str(kappa)] = row
            print(f"  {kappa:6.2f} | " + " | ".join(f"{row[k]:22.3f}" for k in METHODS))

        # calibration check on the null itself (should be ~0.10 by construction,
        # so report the NOMINAL type-I error instead, which is the diagnostic)
        res['results'][str(N)]['null_nominal_typeI'] = {}
        for k, v in null.items():
            v = v[np.isfinite(v)]
            p = stats.chi2.sf(v, 1) if 'hapmix' not in k else \
                2 * stats.t.sf(np.sqrt(np.clip(v, 0, None)), N - 2)
            res['results'][str(N)]['null_nominal_typeI'][k] = float(np.mean(p < 0.05))
        print("\n  nominal type-I @0.05 under the null "
              "(1.00x target = 0.05; shows who is broken before matching):")
        for k, val in res['results'][str(N)]['null_nominal_typeI'].items():
            print(f"    {k:28s} {val:.4f}  ({val/0.05:6.2f}x)")

    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == '__main__':
    main()

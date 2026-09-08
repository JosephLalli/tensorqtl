"""
Compute-cost benchmark (validation axis 9).

RASQUAL reports cost as a headline result -- 539.9 CPU-days versus TReCASE 4.6
and a plain linear model 0.4 (Kumasaka et al. 2016, Fig. 3) -- and mixQTL's
entire framing is that likelihood-based joint methods are "computationally
intractable for large studies", which is why it uses a log-linear
approximation. hapmixQTL inherits that log-linear framing, so its cost relative
to the iterative-likelihood methods is a claim worth measuring rather than
asserting.

Compared here, all on identical simulated data:
  hapmixQTL      two weighted least-squares fits + inverse-variance meta-analysis
  TReCASE        joint NB + beta-binomial likelihood, numerically optimized (LRT)
  TReC-only      negative-binomial GLM (LRT)
  total-only LS  ordinary least squares on log total counts (the 'Lm' baseline)

Reported per variant-gene test so the numbers extrapolate to a genome-wide run.

Run:  python3 tests/ase_compute_benchmark.py --reps 60
"""

import argparse
import json
import sys
import time
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from ase_external_benchmark import simulate_locus, trec_lrt, trecase_lrt, DTYPE

try:
    from tensorqtl.hapmixqtl import (
        WeightedResidualizer, _estimate_tau, calculate_hapmixqtl_nominal)
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent / 'tensorqtl'))
    from hapmixqtl import (
        WeightedResidualizer, _estimate_tau, calculate_hapmixqtl_nominal)


def _prep(d, n_var):
    """Pre-build hapmixQTL inputs; genotypes tiled to n_var variants."""
    N = len(d['g'])
    rng = np.random.RandomState(0)
    g = np.tile(d['g'], (n_var, 1))
    s = np.tile(d['s'], (n_var, 1))
    tot = d['yL'] + d['yR']
    a = np.log(d['yL'] + 0.5) - np.log(d['yR'] + 0.5)
    t = np.log(tot / 2.0 + 0.5)
    va = np.clip(1.0 / (tot + 1.0), 1e-6, None)
    vt = np.clip(1.0 / (tot + 1.0), 1e-6, None)
    return (torch.tensor(g, dtype=DTYPE), torch.tensor(s, dtype=DTYPE),
            torch.tensor(a, dtype=DTYPE), torch.tensor(t, dtype=DTYPE),
            torch.tensor(va, dtype=DTYPE), torch.tensor(vt, dtype=DTYPE))


def time_hapmix(data, n_var, tau_mode='estimate'):
    t0 = time.perf_counter(); ntests = 0
    for d in data:
        g, s, a, t, va, vt = _prep(d, n_var)
        if tau_mode == 'estimate':
            ta = _estimate_tau(a, va, None, 'cpu'); tt = _estimate_tau(t, vt, None, 'cpu')
            wa = torch.sqrt(1.0 / (va + ta)); wt = torch.sqrt(1.0 / (vt + tt))
        else:
            wa = torch.sqrt(1.0 / va); wt = torch.sqrt(1.0 / vt)
        ra = WeightedResidualizer(None, wa); rt = WeightedResidualizer(None, wt)
        calculate_hapmixqtl_nominal(g, s, a, t, wa, wt, ra, rt)
        ntests += n_var
    return (time.perf_counter() - t0) / ntests


def time_fn(data, fn, n_var):
    """Likelihood methods are per-variant: time one and multiply."""
    t0 = time.perf_counter()
    for d in data:
        fn(d)
    per = (time.perf_counter() - t0) / len(data)
    return per          # seconds per variant-gene test


def time_ols(data, n_var):
    t0 = time.perf_counter(); ntests = 0
    for d in data:
        y = np.log(d['T'] / d['lib'] + 1.0)
        X = np.tile(d['g'] / 2.0, (n_var, 1))
        Xc = X - X.mean(1, keepdims=True); yc = y - y.mean()
        b = (Xc @ yc) / np.maximum((Xc * Xc).sum(1), 1e-12)
        ntests += n_var
    return (time.perf_counter() - t0) / ntests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reps', type=int, default=60)
    ap.add_argument('--N', type=int, default=500)
    ap.add_argument('--n_var', type=int, default=2000,
                    help='variants per gene (a realistic cis-window)')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    data = [simulate_locus(args.N, np.random.RandomState(i), kappa=1.1)
            for i in range(args.reps)]
    print(f"N={args.N} samples, {args.n_var} variants/gene, {args.reps} genes\n")

    res = {}
    res['hapmixQTL (tau=estimate)'] = time_hapmix(data, args.n_var, 'estimate')
    res['hapmixQTL (tau=zero)'] = time_hapmix(data, args.n_var, 'zero')
    res['total-only OLS (Lm)'] = time_ols(data, args.n_var)
    res['TReC-only (NB GLM, LRT)'] = time_fn(data, lambda d: trec_lrt(d), args.n_var)
    res['TReCASE (joint LRT)'] = time_fn(data, lambda d: trecase_lrt(d), args.n_var)

    base = res['hapmixQTL (tau=estimate)']
    # a genome-wide cis scan: ~20k genes x n_var variants
    gw = 20000 * args.n_var
    print(f"{'method':30s} {'sec/test':>12s} {'rel':>9s} {'CPU-days genome-wide':>22s}")
    print("-" * 76)
    out = {}
    for k, v in sorted(res.items(), key=lambda kv: kv[1]):
        days = v * gw / 86400.0
        out[k] = dict(sec_per_test=v, relative=v / base, cpu_days_genomewide=days)
        print(f"{k:30s} {v:12.3e} {v/base:8.1f}x {days:22.2f}")
    print(f"\n(genome-wide = 20,000 genes x {args.n_var} variants = {gw:,} tests, "
          f"single core)")
    print("RASQUAL published 539.9 CPU-days vs TReCASE 4.6 vs Lm 0.4 on their data.")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {'config': vars(args), 'results': out}, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == '__main__':
    main()

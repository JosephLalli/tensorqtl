"""Calibration of map_cis's gene-level empirical p under variance misspecification.

Produces the two rows of docs/hapmixqtl_methods.md section 7 that no other
harness covers:

  * "Whitened residuals, leverage-standardized" -- the empirical p of null genes
    whose inferential variances span 200x, with the variance model correct.
  * "Permutation under a misspecified variance shape" -- the same design when
    the truth is NOT v + tau: multiplicative (mixQTL's structure, Var = c*v) or
    constant (v carrying no information at all).

The permutation permutes whitened null residuals, whose exchangeability needs
the assumed variance to be right up to a constant. tau by the moment estimator
fixes the AVERAGE, not the shape, so a misspecified shape is expected to leave
the null approximately rather than exactly calibrated. This script measures how
far off it goes, which section 8 item 11 cites as a bound.

Run:  python3 tests/ase_variance_shape.py --reps 400
"""
import argparse, contextlib, io, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from tensorqtl.hapmixqtl import map_cis
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from tensorqtl.hapmixqtl import map_cis

N, V = 80, 20


def _gene(rng, kind):
    """One null gene. `kind` sets the TRUTH; the fit always assumes v + tau."""
    g = rng.binomial(2, 0.4, size=(V, N)).astype(float)
    s = np.zeros((V, N))
    het = g == 1
    s[het] = rng.choice([-1.0, 1.0], size=int(het.sum()))
    va = rng.uniform(0.01, 2.0, N)
    vt = rng.uniform(0.01, 2.0, N)
    if kind == 'additive':              # Var = v + tau, exactly the fitted model
        sa, st = np.sqrt(va + 0.3), np.sqrt(vt + 0.3)
    elif kind == 'multiplicative':      # Var = c*v: mixQTL's structure, no additive part
        sa, st = np.sqrt(3.0 * va), np.sqrt(3.0 * vt)
    elif kind == 'constant':            # Var constant: v carries no information
        sa = st = np.full(N, np.sqrt(1.5))
    else:
        raise ValueError(kind)
    a = sa * rng.normal(size=N)
    t = 2.0 + st * rng.normal(size=N)
    C = rng.normal(size=(N, 4))
    t = t + C @ rng.normal(0, 0.3, 4)
    return g, s, a, t, va, vt, C


def cell(kind, reps, nperm, seed):
    rng = np.random.RandomState(seed)
    samples = [f'S{i}' for i in range(N)]
    vids = [f'chr1_{1000 + i * 100}_A_G' for i in range(V)]
    vdf = pd.DataFrame({'chrom': ['chr1'] * V,
                        'pos': [1000 + i * 100 for i in range(V)]}, index=vids)
    pos = pd.DataFrame({'chr': ['chr1'], 'pos': [1000]}, index=['G1'])
    pp, pb = [], []
    for rep in range(reps):
        g, s, a, t, va, vt, C = _gene(rng, kind)
        xL = ((g == 2) | ((g == 1) & (s > 0))).astype(float)
        xR = ((g == 2) | ((g == 1) & (s < 0))).astype(float)
        mk = lambda x: pd.DataFrame(x[None, :], index=['G1'], columns=samples)
        with contextlib.redirect_stdout(io.StringIO()):
            res = map_cis(pd.DataFrame(g, index=vids, columns=samples), vdf,
                          mk(a), mk(t), mk(va), mk(vt), pos,
                          xL_df=pd.DataFrame(xL, index=vids, columns=samples),
                          xR_df=pd.DataFrame(xR, index=vids, columns=samples),
                          nperm=nperm, covariates_df=pd.DataFrame(C, index=samples),
                          ase_covariates_df=None, seed=rep, verbose=False)
        pp.append(float(res['pval_perm'].iloc[0]))
        pb.append(float(res['pval_beta'].iloc[0]))
    pp, pb = np.array(pp), np.array(pb)
    ok = np.isfinite(pb)
    return dict(truth=kind, n=int(reps), nperm=int(nperm),
                perm_mean=float(pp.mean()), perm_lt_05=float((pp < 0.05).mean()),
                perm_lt_50=float((pp < 0.5).mean()),
                beta_finite=float(ok.mean()),
                beta_mean=float(pb[ok].mean()) if ok.any() else float('nan'),
                beta_lt_05=float((pb < 0.05).mean()))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--reps', type=int, default=400,
                    help='null genes per cell (400 for the section 7 numbers)')
    ap.add_argument('--nperm', type=int, default=200)
    ap.add_argument('--seed', type=int, default=3)
    ap.add_argument('--out', default=None, help='write the cells as JSON')
    args = ap.parse_args(argv)
    print(f'{"true variance":>16s} | {"mean p":>7s} {"<0.05":>7s} {"<0.5":>6s} | '
          f'{"beta mean":>9s} {"beta<0.05":>9s}   (Monte Carlo SE on the mean '
          f'{np.sqrt(1/12/args.reps):.3f})')
    out = []
    for kind in ('additive', 'multiplicative', 'constant'):
        c = cell(kind, args.reps, args.nperm, args.seed)
        out.append(c)
        print(f'{kind:>16s} | {c["perm_mean"]:7.3f} {c["perm_lt_05"]:7.3f} '
              f'{c["perm_lt_50"]:6.3f} | {c["beta_mean"]:9.3f} {c["beta_lt_05"]:9.3f}',
              flush=True)
    print('\nThe additive row is the calibration claim; the other two are the '
          'misspecification bound (docs/hapmixqtl_methods.md section 8 item 11).')
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f'wrote {args.out}')
    return out


if __name__ == '__main__':
    main()

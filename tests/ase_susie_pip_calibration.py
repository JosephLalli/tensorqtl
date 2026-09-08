"""
Fine-mapping calibration for hapmixQTL's map_susie (validation axis 8).

mixQTL validates its fine-mapping (mixFine) by two criteria (Liang et al. 2021,
Fig. 3): PIPs must be **calibrated** -- "the PIPs of both trcFine and mixFine
were consistent with the proportion of true causal variants within each PIP bin"
-- and the 95% credible set must contain the causal variant, at a useful size.
`map_susie` is a shipped hapmixQTL feature and neither property has ever been
checked here.

Three questions:
  1. Are the PIPs calibrated? Within a PIP bin, the fraction of variants that
     are truly causal should match the bin's mean PIP. A method can localize
     well on average while producing PIPs that mean nothing quantitatively.
  2. Do 95% credible sets cover the causal variant ~95% of the time, and how
     large are they?
  3. Does the tau_mode defect propagate into fine-mapping? The variance model
     feeds the whole SuSiE fit, so a broken one should distort PIPs even where
     it does not change the ranking.

Genotypes carry realistic LD (a latent-block structure), since fine-mapping is
entirely about resolving correlated variants.

Run:  python3 tests/ase_susie_pip_calibration.py --reps 120
"""

import argparse
import json
import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from tensorqtl.hapmixqtl import map_susie
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent / 'tensorqtl'))
    from hapmixqtl import map_susie


def make_locus(rng, N=250, p=40, n_causal=1, beta=0.8, block=8, rho=0.995,
               sigma_bio=0.3):
    """One gene with LD-structured genotypes and a planted cis effect."""
    # latent block structure -> realistic LD between neighbouring variants
    n_blocks = max(p // block, 1)
    Z = rng.randn(N, n_blocks)
    X = np.empty((N, p))
    for j in range(p):
        b = min(j // block, n_blocks - 1)
        X[:, j] = rho * Z[:, b] + np.sqrt(1 - rho ** 2) * rng.randn(N)
    # to dosages, preserving LD
    q = np.quantile(X, [0.3, 0.7], axis=0)
    G = (X > q[0]).astype(float) + (X > q[1]).astype(float)
    keep = G.std(0) > 1e-6
    G = G[:, keep]
    p = G.shape[1]

    # phase: random ALT-on-L for hets
    xL = np.zeros((p, N)); xR = np.zeros((p, N))
    Gt = G.T
    het = Gt == 1; homalt = Gt == 2
    lalt = rng.rand(p, N) < 0.5
    xL[het & lalt] = 1; xR[het & ~lalt] = 1
    xL[homalt] = 1; xR[homalt] = 1

    causal = rng.choice(p, size=min(n_causal, p), replace=False)
    a = rng.normal(0, sigma_bio, N)
    t = 2.0 + rng.normal(0, sigma_bio, N)
    for c in causal:
        a = a + beta * (xL[c] - xR[c])
        t = t + beta * (Gt[c] / 2.0)

    samples = [f"S{i:04d}" for i in range(N)]
    vids = [f"chr1_{10000 + j * 1000}_A_G" for j in range(p)]
    pid = ["ENSG00000001.1"]
    va = rng.uniform(0.02, 0.10, N); vt = rng.uniform(0.02, 0.10, N)
    a = a + np.sqrt(va) * rng.randn(N)
    t = t + np.sqrt(vt) * rng.randn(N)
    d = dict(
        genotype_df=pd.DataFrame(Gt, index=vids, columns=samples),
        variant_df=pd.DataFrame({'chrom': ['chr1'] * p,
                                 'pos': [10000 + j * 1000 for j in range(p)]},
                                index=vids),
        A_df=pd.DataFrame(a.reshape(1, -1), index=pid, columns=samples),
        T_df=pd.DataFrame(t.reshape(1, -1), index=pid, columns=samples),
        Va_df=pd.DataFrame(va.reshape(1, -1), index=pid, columns=samples),
        Vt_df=pd.DataFrame(vt.reshape(1, -1), index=pid, columns=samples),
        xL_df=pd.DataFrame(xL, index=vids, columns=samples),
        xR_df=pd.DataFrame(xR, index=vids, columns=samples),
        pos_df=pd.DataFrame({'chr': ['chr1'], 'pos': [10000 + (p // 2) * 1000]},
                            index=pid),
        causal_ids=set(vids[c] for c in causal), vids=vids)
    return d


def run(reps, N, p, n_causal, tau_mode, betas, seed0=0, L=5):
    """Pool across a grid of effect sizes.

    A single large effect makes fine-mapping trivial (every PIP is 0 or 1) and
    calibration untestable; mixQTL likewise pools across simulation settings for
    its PIP-calibration figure. The grid populates the intermediate bins, which
    are the only place calibration can fail.
    """
    import contextlib, io
    pips, is_causal = [], []
    cs_cov, cs_size, n_cs = [], [], 0
    for r in range(reps):
        beta = betas[r % len(betas)]
        rng = np.random.RandomState(seed0 + r)
        d = make_locus(rng, N=N, p=p, n_causal=n_causal, beta=beta)
        try:
          with contextlib.redirect_stdout(io.StringIO()):
            summary, res = map_susie(
                d['genotype_df'], d['variant_df'], d['A_df'], d['T_df'],
                d['Va_df'], d['Vt_df'], d['pos_df'],
                xL_df=d['xL_df'], xR_df=d['xR_df'], L=L, window=1_000_000,
                max_iter=300, tau_mode=tau_mode, summary_only=False,
                verbose=False)
        except Exception:
            continue
        key = 'ENSG00000001.1'
        if key not in res:
            continue
        pip = np.asarray(res[key]['pip'])[:, 0]
        vids = d['vids'][:len(pip)]
        for v, pp in zip(vids, pip):
            pips.append(float(pp)); is_causal.append(v in d['causal_ids'])
        # credible sets from the tidy summary
        sub = summary[summary['phenotype_id'] == key]
        for cs, grp in sub.groupby('cs_id'):
            n_cs += 1
            members = set(grp['variant_id'])
            cs_cov.append(len(members & d['causal_ids']) > 0)
            cs_size.append(len(members))
    return (np.array(pips), np.array(is_causal),
            np.array(cs_cov), np.array(cs_size), n_cs)


def pip_calibration_table(pips, causal, edges=(0, .1, .25, .5, .75, .9, 1.01)):
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (pips >= lo) & (pips < hi)
        if m.sum() == 0:
            continue
        rows.append(dict(bin=f"[{lo:.2f},{hi:.2f})", n=int(m.sum()),
                         mean_pip=float(pips[m].mean()),
                         frac_causal=float(causal[m].mean())))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reps', type=int, default=120)
    ap.add_argument('--N', type=int, default=150)
    ap.add_argument('--p', type=int, default=40)
    ap.add_argument('--n_causal', type=int, default=1)
    ap.add_argument('--betas', type=str, default='0.04,0.07,0.10,0.15,0.22')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    res = {'config': vars(args), 'by_tau_mode': {}}

    for tau_mode in ('estimate', 'zero'):
        print(f"\n{'='*72}\ntau_mode = {tau_mode!r}   "
              f"(N={args.N}, {args.p} variants, {args.n_causal} causal, "
              f"betas={args.betas}, LD rho=0.995)\n{'='*72}")
        betas = [float(x) for x in args.betas.split(',')]
        pips, causal, cov, size, n_cs = run(
            args.reps, args.N, args.p, args.n_causal, tau_mode, betas)
        if pips.size == 0:
            print("  no results"); continue
        tbl = pip_calibration_table(pips, causal)
        print("  PIP CALIBRATION (mixQTL Fig 3a analogue): within a bin, the")
        print("  fraction truly causal should match the bin's mean PIP.")
        print(f"    {'bin':>14s} {'n':>7s} {'mean PIP':>9s} {'frac causal':>12s} {'gap':>8s}")
        for row in tbl:
            gap = row['frac_causal'] - row['mean_pip']
            print(f"    {row['bin']:>14s} {row['n']:7d} {row['mean_pip']:9.3f} "
                  f"{row['frac_causal']:12.3f} {gap:+8.3f}")
        cs_cov = float(cov.mean()) if cov.size else float('nan')
        cs_sz = float(size.mean()) if size.size else float('nan')
        print(f"\n  95% credible sets: {n_cs} found, coverage={cs_cov:.3f} "
              f"(target 0.95), mean size={cs_sz:.1f} variants")
        # a single scalar summary of calibration quality
        w = np.array([r['n'] for r in tbl], float)
        err = np.array([abs(r['frac_causal'] - r['mean_pip']) for r in tbl])
        wmae = float((w * err).sum() / w.sum()) if w.sum() else float('nan')
        print(f"  weighted mean |frac_causal - mean_PIP| = {wmae:.3f}"
              "   (0 = perfectly calibrated)")
        res['by_tau_mode'][tau_mode] = dict(
            table=tbl, cs_coverage=cs_cov, cs_mean_size=cs_sz, n_cs=n_cs,
            weighted_mae=wmae)

    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2, default=float))
        print(f"\nwrote {args.out}")


if __name__ == '__main__':
    main()

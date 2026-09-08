"""
REAL-DATA validation of hapmixQTL on GTEx v8 phASER haplotype expression.

Closes the biggest gap in docs/ase_validation.md: every earlier tier is
simulation. This uses the real thing.

DATA (public, no dbGaP needed)
------------------------------
GTEx v8 haplotype-expression matrices produced by phASER (Castel et al. 2016),
from `gs://adult-gtex/haplotype-expression/v8/haplotype-expression-matrices/`:
real per-gene, per-sample haplotype counts `yL | yR` -- exactly hapmixQTL's
input. Both the standard and the WASP mapping-bias-corrected matrices are used.

WHAT THIS CAN AND CANNOT DO
---------------------------
GTEx *genotypes* are dbGaP-protected (phs000424.v8), so real cis-QTL mapping --
and therefore effect-size concordance against GTEx's published aFC, replication
of GTEx eGenes, and functional/motif enrichment -- is NOT possible from public
data alone. Those axes stay open and need dbGaP or AnVIL authorization.

What IS possible, and is the more important check: a **real-data null
calibration**. Real haplotype counts supply the real variance structure --
genuine overdispersion, depth distribution, zero inflation, biological
variability -- and genotypes drawn independently of expression make every test a
true null. This is the same logic as RASQUAL's permutation null (Supplementary
Note, "Generation and analysis of simulation data"), which permutes to obtain an
empirical null; permuting genotype labels and drawing genotypes independently
are equivalent for calibration. Any method whose p-values are not uniform here
is miscalibrated on real data, whatever simulations say.

Also measured: sensitivity to reference mapping bias, by re-running on the
WASP-corrected matrix. RASQUAL treats mapping bias as a first-class nuisance
(their parameter phi); hapmixQTL does not model it at all, so the WASP-vs-not
comparison is the available robustness check.

Run:  python3 tests/ase_gtex_real_data.py --npz gtex_muscle.npz --reps 3
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

try:
    from tensorqtl.hapmixqtl import (
        WeightedResidualizer, _estimate_tau,
        calculate_hapmixqtl_nominal, compute_summaries_from_gibbs)
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent / 'tensorqtl'))
    from hapmixqtl import (
        WeightedResidualizer, _estimate_tau,
        calculate_hapmixqtl_nominal, compute_summaries_from_gibbs)

DTYPE = torch.float64


def gibbs_from_counts(yl, yr, n_draws=60, kappa=0.5, rng=None):
    """Emulate a quantifier posterior over the allelic split of the OBSERVED total.

    This is what Salmon/mmseq Gibbs draws represent: uncertainty in assigning
    reads to haplotypes, conditional on the total. Feeds the real production
    summary code so Va/Vt come from `compute_summaries_from_gibbs`.
    """
    N = yl.shape[0]
    tot = (yl + yr).astype(float)
    frac = (yl + kappa) / (tot + 2 * kappa)
    YL = np.empty((1, N, n_draws)); YR = np.empty((1, N, n_draws))
    for i in range(N):
        n_i = int(tot[i])
        d = rng.binomial(n_i, frac[i], size=n_draws).astype(float) if n_i > 0 \
            else np.zeros(n_draws)
        YL[0, i, :] = d; YR[0, i, :] = n_i - d
    return YL, YR


def hapmix_p(g, s, a, t, va, vt, tau_mode):
    N = len(a)
    a_t = torch.tensor(a, dtype=DTYPE); t_t = torch.tensor(t, dtype=DTYPE)
    va_t = torch.tensor(np.clip(va, 1e-8, None), dtype=DTYPE)
    vt_t = torch.tensor(np.clip(vt, 1e-8, None), dtype=DTYPE)
    if tau_mode == 'estimate':
        ta = _estimate_tau(a_t, va_t, None, 'cpu'); tt = _estimate_tau(t_t, vt_t, None, 'cpu')
        sqrt_wa = torch.sqrt(1.0 / (va_t + ta)); sqrt_wt = torch.sqrt(1.0 / (vt_t + tt))
    else:
        sqrt_wa = torch.sqrt(1.0 / va_t); sqrt_wt = torch.sqrt(1.0 / vt_t)
    res_a = WeightedResidualizer(None, sqrt_wa); res_t = WeightedResidualizer(None, sqrt_wt)
    ts, *_ = calculate_hapmixqtl_nominal(
        torch.tensor(g.reshape(1, -1), dtype=DTYPE),
        torch.tensor(s.reshape(1, -1), dtype=DTYPE),
        a_t, t_t, sqrt_wa, sqrt_wt, res_a, res_t)
    v = float(ts.numpy()[0])
    return np.nan if not np.isfinite(v) else float(2 * stats.t.sf(abs(v), N - 2))


def run_null(npz, reps=3, maf=0.3, n_draws=60, seed=0, max_genes=None):
    """Real haplotype counts x genotypes drawn independently => a true null."""
    d = np.load(npz, allow_pickle=True)
    yL, yR, genes = d['yL'], d['yR'], d['genes']
    if max_genes:
        yL, yR, genes = yL[:max_genes], yR[:max_genes], genes[:max_genes]
    G, N = yL.shape
    print(f"  {G} genes x {N} samples from {Path(npz).name}", flush=True)
    P = {'zero': [], 'estimate': []}
    depth = []
    for rep in range(reps):
        for gi in range(G):
            rng = np.random.RandomState(seed + 9973 * rep + gi)
            yl, yr = yL[gi].astype(float), yR[gi].astype(float)
            # keep samples with usable allele-specific coverage
            m = (yl + yr) >= 10
            if m.sum() < 60:
                continue
            yl, yr = yl[m], yr[m]
            n = len(yl)
            depth.append(float(np.median(yl + yr)))
            YLd, YRd = gibbs_from_counts(yl, yr, n_draws, rng=rng)
            A, T_, Va, Vt, _ = compute_summaries_from_gibbs(YLd, YRd)
            a, va, vt = A[0], Va[0], Vt[0]
            t = np.log((yl + yr) + 1.0)          # real total expression
            # genotype drawn INDEPENDENTLY of expression -> null by construction
            g = rng.binomial(2, maf, size=n).astype(float)
            s = np.zeros(n); het = g == 1
            s[het] = rng.choice([-1.0, 1.0], size=int(het.sum()))
            for mode in ('zero', 'estimate'):
                p = hapmix_p(g, s, a, t, va, vt, mode)
                if np.isfinite(p):
                    P[mode].append(p)
    out = {}
    for mode, ps in P.items():
        ps = np.array(ps)
        chi2 = stats.chi2.isf(np.clip(ps, 1e-300, 1), 1)
        out[mode] = dict(n=int(ps.size),
                         t05=float(np.mean(ps < 0.05)), t01=float(np.mean(ps < 0.01)),
                         t001=float(np.mean(ps < 1e-3)),
                         lam=float(np.median(chi2) / stats.chi2.ppf(0.5, 1)),
                         ks_p=float(stats.kstest(ps, 'uniform').pvalue))
    out['median_as_depth'] = float(np.median(depth)) if depth else float('nan')
    return out


def _report(tag, r):
    print(f"\n  --- {tag} ---   median allele-specific depth = "
          f"{r['median_as_depth']:.0f} reads")
    for mode in ('zero', 'estimate'):
        m = r[mode]
        print(f"    tau_mode={mode:9s} n={m['n']:6d}  "
              f"typeI@0.05={m['t05']:.4f} ({m['t05']/0.05:6.2f}x)  "
              f"@0.01={m['t01']:.4f} ({m['t01']/0.01:6.2f}x)  "
              f"@1e-3={m['t001']:.5f}  lambda={m['lam']:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', required=True)
    ap.add_argument('--npz_wasp', default=None,
                    help='WASP-corrected matrix, for the mapping-bias check')
    ap.add_argument('--reps', type=int, default=3)
    ap.add_argument('--max_genes', type=int, default=None)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    res = {'config': vars(args)}
    print("=== REAL GTEx v8 phASER haplotype counts, genotypes drawn independently ===")
    print("    (a true null: any departure from uniform p-values is miscalibration)")
    res['standard'] = run_null(args.npz, reps=args.reps, max_genes=args.max_genes)
    _report('phASER (standard)', res['standard'])

    if args.npz_wasp and Path(args.npz_wasp).exists():
        res['wasp'] = run_null(args.npz_wasp, reps=args.reps, max_genes=args.max_genes)
        _report('phASER + WASP (mapping-bias corrected)', res['wasp'])
        print("\n    Mapping-bias sensitivity: compare the two blocks. hapmixQTL does not"
              "\n    model reference bias (RASQUAL's phi), so a large gap would indicate"
              "\n    dependence on upstream WASP correction.")

    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == '__main__':
    main()

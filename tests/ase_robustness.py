"""
Robustness and parameter recovery for hapmixQTL (validation axes 3 and 8).

Closes four items left open in docs/ase_validation.md:

  A. PARAMETER RECOVERY (axis 3). RASQUAL validates by showing estimated
     parameters track their simulated values (Supplementary Figs. 8-10). The
     analogous check here is whether `_estimate_tau` recovers the overdispersion
     it is supposed to, since the whole tau_mode fix rests on it, plus whether
     the effect estimate tracks truth across effect sizes.

  B. PHASING ERROR. Section 3 concluded the unused `Cat` covariance is safe to
     ignore because the ASE predictor s is orthogonal to g/2 under RANDOM phase.
     That argument is stated there with an explicit caveat: systematic phasing
     error could break it. This tests the caveat directly -- both whether
     calibration survives, and whether the orthogonality that justifies the
     scalar meta-analysis survives.

  C. COVARIATES. Every earlier tier ran with no covariates, so the
     WeightedResidualizer path was exercised only by unit tests, never by a
     calibration sweep.

  D. ROBUST (SANDWICH) SEs. `se_mode='robust'` is an existing option that was
     never evaluated. A sandwich estimator does not assume the variance model is
     correct, so it is a plausible alternative route to fixing the tau defect --
     worth knowing whether it works, since it would need no tau estimation.

Run:  python3 tests/ase_robustness.py --reps 600
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
from ase_validation import simulate_channels, pvals_from_t

try:
    from tensorqtl.hapmixqtl import (
        WeightedResidualizer, _estimate_tau, calculate_hapmixqtl_nominal)
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent / 'tensorqtl'))
    from hapmixqtl import (
        WeightedResidualizer, _estimate_tau, calculate_hapmixqtl_nominal)

DTYPE = torch.float64


def associate(g, s, a, t, va, vt, tau_mode='estimate', covariates=None, robust=False):
    """Production statistic, with the robust-SE and covariate paths exposed."""
    g_t = torch.tensor(g, dtype=DTYPE); s_t = torch.tensor(s, dtype=DTYPE)
    a_t = torch.tensor(a, dtype=DTYPE); t_t = torch.tensor(t, dtype=DTYPE)
    va_t = torch.tensor(np.clip(va, 1e-8, None), dtype=DTYPE)
    vt_t = torch.tensor(np.clip(vt, 1e-8, None), dtype=DTYPE)
    cov_t = None if covariates is None else torch.tensor(covariates, dtype=DTYPE)
    if tau_mode == 'estimate':
        ta = _estimate_tau(a_t, va_t, cov_t, 'cpu'); tt = _estimate_tau(t_t, vt_t, cov_t, 'cpu')
        wa = torch.sqrt(1.0 / (va_t + ta)); wt = torch.sqrt(1.0 / (vt_t + tt))
    else:
        wa = torch.sqrt(1.0 / va_t); wt = torch.sqrt(1.0 / vt_t)
    ra = WeightedResidualizer(cov_t, wa); rt = WeightedResidualizer(cov_t, wt)
    ts, slope, se, sa, sea, st, sest = calculate_hapmixqtl_nominal(
        g_t, s_t, a_t, t_t, wa, wt, ra, rt, robust=robust)
    return (ts.numpy(), slope.numpy(), se.numpy(),
            sa.numpy(), sea.numpy(), st.numpy(), sest.numpy())


def _calib(p, n_cov=0):
    p = np.asarray(p); p = p[np.isfinite(p)]
    chi2 = stats.chi2.isf(np.clip(p, 1e-300, 1), 1)
    return dict(n=int(p.size), t05=float(np.mean(p < 0.05)),
                t01=float(np.mean(p < 0.01)),
                lam=float(np.median(chi2) / stats.chi2.ppf(0.5, 1)))


# ---------------------------------------------------------------------------
#  A. Parameter recovery
# ---------------------------------------------------------------------------

def part_A(reps, N):
    print("\n=== A. PARAMETER RECOVERY (axis 3) ===")
    print("  tau: the fix rests on _estimate_tau recovering the extra variance.")
    print(f"  {'true tau':>9s} | {'mean est':>9s} | {'median est':>11s} | {'rel bias':>9s} | {'corr':>6s}")
    print("  " + "-" * 60)
    out = {'tau_recovery': [], 'beta_recovery': []}
    for sb in (0.0, 0.2, 0.4, 0.6):
        true_tau = sb ** 2                      # Var = v_inf + sigma_bio^2
        est = []
        for r in range(reps):
            rng = np.random.RandomState(4000 + r)
            g, s, a, t, va, vt = simulate_channels(N, 1, rng, beta=0.0, sigma_bio=sb)
            a_t = torch.tensor(a, dtype=DTYPE)
            va_t = torch.tensor(np.clip(va, 1e-8, None), dtype=DTYPE)
            est.append(float(_estimate_tau(a_t, va_t, None, 'cpu')))
        est = np.array(est)
        rel = (est.mean() - true_tau) / max(true_tau, 1e-9) if true_tau > 0 else np.nan
        row = dict(true_tau=true_tau, mean_est=float(est.mean()),
                   median_est=float(np.median(est)), rel_bias=float(rel))
        out['tau_recovery'].append(row)
        print(f"  {true_tau:9.4f} | {est.mean():9.4f} | {np.median(est):11.4f} | "
              f"{rel:+9.3f} | {'':>6s}" if true_tau > 0 else
              f"  {true_tau:9.4f} | {est.mean():9.4f} | {np.median(est):11.4f} | "
              f"{'   n/a':>9s} |")

    print("\n  beta: estimated vs true effect (RASQUAL Supp Figs 8-10 analogue)")
    print(f"  {'true beta':>9s} | {'mean est':>9s} | {'bias':>8s} | {'corr(est,true)':>14s}")
    print("  " + "-" * 52)
    tb, eb = [], []
    for beta in (0.0, 0.1, 0.2, 0.4, 0.8):
        ests = []
        for r in range(reps // 2):
            rng = np.random.RandomState(8000 + r + int(beta * 100))
            g, s, a, t, va, vt = simulate_channels(N, 1, rng, beta=beta, sigma_bio=0.3)
            _, slope, se, *_ = associate(g, s, a, t, va, vt, 'estimate')
            if np.isfinite(se[0]):
                ests.append(float(slope[0]))
        ests = np.array(ests)
        out['beta_recovery'].append(dict(true=beta, mean_est=float(ests.mean()),
                                         bias=float(ests.mean() - beta)))
        tb += [beta] * len(ests); eb += list(ests)
        print(f"  {beta:9.2f} | {ests.mean():9.4f} | {ests.mean()-beta:+8.4f} |")
    r = float(np.corrcoef(tb, eb)[0, 1])
    out['beta_corr'] = r
    print(f"  overall corr(estimated, true) = {r:.4f}")
    return out


# ---------------------------------------------------------------------------
#  B. Phasing error
# ---------------------------------------------------------------------------

def part_B(reps, N):
    print("\n=== B. PHASING ERROR (the sec 3 caveat) ===")
    print("  Flip a fraction f of heterozygote phase calls, then ask two things:")
    print("  does calibration survive, and does the orthogonality that justifies")
    print("  the scalar meta-analysis (and retires Cat) survive?")
    print(f"  {'f':>5s} | {'null typeI@0.05':>15s} | {'lambda':>7s} | "
          f"{'beta_a atten.':>13s} | {'corr(b_a,b_t)':>14s}")
    print("  " + "-" * 68)
    out = []
    for f in (0.0, 0.05, 0.10, 0.25, 0.50):
        P, ba, bt = [], [], []
        for r in range(reps):
            rng = np.random.RandomState(200 + r)
            g, s, a, t, va, vt = simulate_channels(N, 1, rng, beta=0.0, sigma_bio=0.3)
            s_err = s.copy()
            het = np.abs(s[0]) > 0
            idx = np.where(het)[0]
            if f > 0 and idx.size:
                flip = rng.rand(idx.size) < f
                s_err[0, idx[flip]] *= -1
            ts, slope, se, sa, sea, st, sest = associate(
                g, s_err, a, t, va, vt, 'estimate')
            P.append(pvals_from_t(ts, N))
            if np.isfinite(sea[0]) and np.isfinite(sest[0]):
                ba.append(float(sa[0])); bt.append(float(st[0]))
        c = _calib(np.concatenate(P))
        # attenuation: same experiment but with a real shared effect
        A2 = []
        for r in range(reps // 2):
            rng = np.random.RandomState(9100 + r)
            g, s, a, t, va, vt = simulate_channels(N, 1, rng, beta=0.5, sigma_bio=0.3)
            s_err = s.copy(); idx = np.where(np.abs(s[0]) > 0)[0]
            if f > 0 and idx.size:
                s_err[0, idx[rng.rand(idx.size) < f]] *= -1
            _, _, _, sa, sea, _, _ = associate(g, s_err, a, t, va, vt, 'estimate')
            if np.isfinite(sea[0]):
                A2.append(float(sa[0]))
        atten = float(np.mean(A2) / 0.5) if A2 else np.nan
        rr = float(np.corrcoef(ba, bt)[0, 1]) if len(ba) > 10 else np.nan
        out.append(dict(f=f, **c, beta_a_retained=atten, corr_ba_bt=rr))
        print(f"  {f:5.2f} | {c['t05']:8.4f} ({c['t05']/0.05:4.2f}x) | {c['lam']:7.2f} | "
              f"{atten:13.3f} | {rr:+14.4f}")
    return out


# ---------------------------------------------------------------------------
#  C. Covariates
# ---------------------------------------------------------------------------

def part_C(reps, N):
    print("\n=== C. COVARIATES (the residualizer path, never calibration-swept) ===")
    print(f"  {'n_cov':>6s} | {'null typeI@0.05':>15s} | {'@0.01':>12s} | {'lambda':>7s}")
    print("  " + "-" * 50)
    out = []
    for ncov in (0, 2, 10, 30):
        P = []
        for r in range(reps):
            rng = np.random.RandomState(600 + r)
            g, s, a, t, va, vt = simulate_channels(N, 1, rng, beta=0.0, sigma_bio=0.3)
            C = None
            if ncov:
                C = rng.randn(N, ncov)
                # covariates genuinely drive expression, so they must be removed
                load = rng.randn(ncov)
                a = a + C @ load * 0.3
                t = t + C @ load * 0.3
            ts, *_ = associate(g, s, a, t, va, vt, 'estimate', covariates=C)
            P.append(pvals_from_t(ts, N, n_cov=ncov))
        c = _calib(np.concatenate(P))
        out.append(dict(n_cov=ncov, **c))
        print(f"  {ncov:6d} | {c['t05']:8.4f} ({c['t05']/0.05:4.2f}x) | "
              f"{c['t01']:6.4f} ({c['t01']/0.01:4.2f}x) | {c['lam']:7.2f}")
    return out


# ---------------------------------------------------------------------------
#  D. Robust (sandwich) SEs
# ---------------------------------------------------------------------------

def part_D(reps, N):
    print("\n=== D. ROBUST SEs -- can the sandwich fix tau's job by itself? ===")
    print(f"  {'tau_mode':>9s} {'se_mode':>8s} | {'sigma_bio':>9s} | "
          f"{'typeI@0.05':>12s} | {'lambda':>7s}")
    print("  " + "-" * 58)
    out = []
    for tau_mode in ('zero', 'estimate'):
        for robust in (False, True):
            for sb in (0.0, 0.6):
                P = []
                for r in range(reps):
                    rng = np.random.RandomState(1500 + r)
                    g, s, a, t, va, vt = simulate_channels(N, 1, rng, beta=0.0, sigma_bio=sb)
                    ts, *_ = associate(g, s, a, t, va, vt, tau_mode, robust=robust)
                    P.append(pvals_from_t(ts, N))
                c = _calib(np.concatenate(P))
                out.append(dict(tau_mode=tau_mode, se_mode='robust' if robust else 'model',
                                sigma_bio=sb, **c))
                print(f"  {tau_mode:>9s} {('robust' if robust else 'model'):>8s} | "
                      f"{sb:9.1f} | {c['t05']:6.4f} ({c['t05']/0.05:5.2f}x) | {c['lam']:7.2f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reps', type=int, default=600)
    ap.add_argument('--N', type=int, default=200)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    res = {'config': vars(args)}
    res['A_parameter_recovery'] = part_A(args.reps, args.N)
    res['B_phasing_error'] = part_B(args.reps, args.N)
    res['C_covariates'] = part_C(args.reps, args.N)
    res['D_robust_se'] = part_D(args.reps, args.N)
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2, default=float))
        print(f"\nwrote {args.out}")


if __name__ == '__main__':
    main()

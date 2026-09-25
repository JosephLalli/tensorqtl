"""Three candidate repairs for the optimistic standard error, measured together.

BASELINE reproduces what ships: each channel fitted separately with weights
1/v and a residual scale fitted per variant, then combined by inverse-variance
meta-analysis, referenced to F(1, dof) with dof = min(dof_a, dof_t). The
reproduction is CHECKED against map_cis before anything is varied -- a
reimplementation that silently disagrees would make every arm below meaningless.

  satterthwaite   the requested option 1. v is estimated from ~170 effective
                  Gibbs draws, so the whitened squared residual carries extra
                  variance and the fitted scale has fewer effective degrees of
                  freedom than it is charged. Var(e^2) is inflated by
                  (m^2/(m-2))(3/(m-4) - 1/(m-2))/2 = 1.042 at m = 170, so the
                  reference becomes F(1, dof/1.042). Reference change only; the
                  variance model is untouched.

  stacked         both channels in ONE weighted regression on a common slope,
                  with a single residual scale and the degrees of freedom
                  charged once, instead of two fits meta-analysed. Removes the
                  combination step rather than correcting it, which is
                  structurally what RASQUAL does.

  stacked_hc3     the same stacked system with an HC3 sandwich standard error.
                  HC3 rather than HC1 because the allelic channel often has
                  only ~20 informative donors at a variant, where HC1 is itself
                  anticonservative. This is the only arm that targets weights
                  being the WRONG SHAPE rather than merely noisy -- and the
                  shape error is real: the standardized squared residual trends
                  with log v at pooled slope -0.064 on the allelic channel.

WHY OPTION 1 IS EXPECTED TO BE NEARLY INERT, recorded before seeing the result:
weight noise inflates Var(beta) by m/(m-4) = 1.024 but also inflates sigma^2_hat
by E[1/v_hat] = 1.012, so the net predicted shortfall is 0.6% against the
4.0-4.4% measured. Included because it was asked for and because a measured
null result is worth more than an argument.
"""
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import compare_mixqtl_replication as CM        # noqa: E402
from tensorqtl.hapmixqtl import map_cis        # noqa: E402

D = Path('/mnt/ssd/lalli/brainvar_hapmix_deploy')
RUN = D / 'rasqual_default_mode_20260923'
SEED, EPS = 42, 1e-12
M_EFF = 170.0
KAPPA = (M_EFF ** 2 / (M_EFF - 2)) * (3 / (M_EFF - 4) - 1 / (M_EFF - 2)) / 2


def resid_out(X, Z, w):
    """Residualize columns of X on Z in the sqrt(w)-weighted space."""
    sw = np.sqrt(w)[:, None]
    Xw, Zw = X * sw, Z * sw
    if Zw.shape[1] == 0:
        return Xw
    q, _ = np.linalg.qr(Zw)
    return Xw - q @ (q.T @ Xw)


def fit_all(a, s, va, t, g, vt, C):
    """Every arm at one variant. Returns dict of (stat, dof, label) or None."""
    # every donor with allelic information enters, INCLUDING s == 0: they
    # contribute nothing to the slope but do contribute to the residual sum of
    # squares and to the degrees of freedom, which is what hapmixQTL does.
    # Filtering them out was the reproduction failure.
    ka = np.isfinite(a) & np.isfinite(va) & (va > EPS) & np.isfinite(s)
    kt = np.isfinite(t) & np.isfinite(vt) & (vt > EPS) & np.isfinite(g)
    if ka.sum() < 5 or kt.sum() < 10:
        return None
    # --- allelic: through origin, no nuisance columns (production since 2026-09-15)
    wa = 1.0 / va[ka]
    ya, xa = a[ka] * np.sqrt(wa), s[ka] * np.sqrt(wa)
    xxa = float(xa @ xa)
    if xxa <= 0:
        return None
    ba = float(xa @ ya) / xxa
    ea = ya - ba * xa
    dofa = max(ka.sum() - 1, 1)
    s2a = float(ea @ ea) / dofa
    sea2 = s2a / xxa
    # --- total: intercept + covariates residualized out, in the weighted space
    wt = 1.0 / vt[kt]
    Z = np.column_stack([np.ones(kt.sum()), C[kt]])
    yt = resid_out(t[kt][:, None], Z, wt).ravel()
    xt = resid_out(g[kt][:, None], Z, wt).ravel()
    xxt = float(xt @ xt)
    if xxt <= 0:
        return None
    bt = float(xt @ yt) / xxt
    et = yt - bt * xt
    doft = max(kt.sum() - 1 - Z.shape[1], 1)
    s2t = float(et @ et) / doft
    set2 = s2t / xxt
    if not (np.isfinite(sea2) and np.isfinite(set2)) or sea2 <= 0 or set2 <= 0:
        return None
    out = {}
    # --- baseline: inverse-variance meta-analysis, F(1, min dof)
    prec = 1 / sea2 + 1 / set2
    b = (ba / sea2 + bt / set2) / prec
    se = np.sqrt(1 / prec)
    dof = min(dofa, doft)
    out['baseline'] = ((b / se) ** 2, dof)
    out['satterthwaite'] = ((b / se) ** 2, max(dof / KAPPA, 1.0))
    # --- stacked: one regression, one scale, dof charged once
    ys = np.concatenate([ya, yt]); xs = np.concatenate([xa, xt])
    xxs = float(xs @ xs)
    bs = float(xs @ ys) / xxs
    es = ys - bs * xs
    dofs = max(len(ys) - 1 - Z.shape[1], 1)
    s2s = float(es @ es) / dofs
    out['stacked'] = ((bs ** 2) / (s2s / xxs), dofs)
    # --- stacked with an HC3 sandwich
    h = xs * xs / xxs                       # leverage for a single predictor
    meat = float(((xs * es / np.clip(1 - h, 1e-6, None)) ** 2).sum())
    var_hc3 = meat / (xxs ** 2)
    out['stacked_hc3'] = ((bs ** 2) / var_hc3 if var_hc3 > 0 else np.nan, dofs)
    return out


def main():
    n_draw = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    me = pd.read_csv(RUN / 'matched_effects.tsv', sep='\t')
    null46 = {l.strip() for l in open(D / 'genes_null46_20260923.txt') if l.strip()}
    targets = {r.gene: r.lead_r for r in me.itertuples()
               if r.gene in null46 and isinstance(r.lead_r, str)}
    I = CM.load_inputs(gene_list=str(D / 'genes_59_stratified_20260923.txt'),
                       regions=str(RUN / 'regions.bed'))
    order, keep = I['order'], I['keep']
    N = len(order)
    import run_hapmixqtl_from_salmon as H
    with contextlib.redirect_stdout(io.StringIO()):
        A, T, Va, Vt, _ = H.compute_summaries_from_gibbs(
            I['YL'][:, keep], I['YR'][:, keep], yT=I['YT'][:, keep])
    gp = I['gp'].loc[I['genes']][['chr', 'pos']]
    gi = {g: i for i, g in enumerate(I['genes'])}
    vi = {v: i for i, v in enumerate(I['vdf'].index)}
    C = I['cov_df'].values
    print(f'Satterthwaite inflation kappa = {KAPPA:.4f} at m = {M_EFF:.0f}\n')

    # ---- reproduction check against map_cis on the OBSERVED data -------------
    print('reproduction check: my baseline against map_cis, observed data')
    diffs = []
    for g, var in list(targets.items())[:12]:
        if g not in gi or var not in vi:
            continue
        i, j = gi[g], vi[var]
        v1 = I['vdf'].iloc[[j]]
        one = lambda M: pd.DataFrame(M[[j]], index=v1.index, columns=order)
        mk = lambda M: pd.DataFrame(M[[i]], index=[g], columns=order)
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                res = map_cis(one(I['dos']), v1[['chrom', 'pos']], mk(A), mk(T),
                              mk(Va), mk(Vt), gp.loc[[g]], xL_df=one(I['xL']),
                              xR_df=one(I['xR']), window=1_000_000, nperm=10,
                              covariates_df=I['cov_df'], ase_covariates_df=None,
                              tau_refit=False, verbose=False,
                              warn_monomorphic=False, beta_approx=False)
            except (ValueError, RuntimeError):
                continue
        sl, se_ = float(res['slope'].iloc[0]), float(res['slope_se'].iloc[0])
        if se_ <= 0:
            continue
        f = fit_all(A[i], (I['xL'][j] - I['xR'][j]).astype(float), Va[i],
                    T[i], I['dos'][j].astype(float) / 2.0, Vt[i], C)
        if f:
            diffs.append(abs(np.sqrt(f['baseline'][0]) - abs(sl / se_)) /
                         abs(sl / se_))
    if diffs:
        print(f'  |t| relative difference over {len(diffs)} genes: '
              f'median {np.median(diffs):.3e}, max {np.max(diffs):.3e}')
        if np.max(diffs) > 0.05:
            print('  WARNING: baseline does not reproduce map_cis; arms below '
                  'are not comparable to the shipped statistic')
    print()

    rng = np.random.RandomState(SEED)
    rows = []
    for p_i in range(n_draw):
        prm = rng.permutation(N)
        for g, var in targets.items():
            if g not in gi or var not in vi:
                continue
            i, j = gi[g], vi[var]
            f = fit_all(A[i][prm], (I['xL'][j] - I['xR'][j]).astype(float),
                        Va[i][prm], T[i][prm],
                        I['dos'][j].astype(float) / 2.0, Vt[i][prm], C[prm])
            if not f:
                continue
            for k, (stat, dof) in f.items():
                if np.isfinite(stat) and stat >= 0:
                    rows.append(dict(arm=k, gene=g, perm=p_i,
                                     pval=float(sps.f.sf(stat, 1, dof))))
        print(f'  draw {p_i + 1}/{n_draw}', flush=True)

    t = pd.DataFrame(rows)
    out = D / 'se_fixes_20260924'
    out.mkdir(exist_ok=True)
    t.to_csv(out / 'pvals.tsv', sep='\t', index=False)
    print('\nnull p at the fixed variant, uniform if calibrated:\n')
    res = {'kappa': float(KAPPA)}
    for arm in ('baseline', 'satterthwaite', 'stacked', 'stacked_hc3'):
        p = t[t.arm == arm].pval.dropna().values
        if len(p) < 50:
            continue
        ks = sps.kstest(p, 'uniform')
        res[arm] = dict(n=int(len(p)), frac_below_10=float((p < .10).mean()),
                        frac_below_05=float((p < .05).mean()),
                        frac_below_01=float((p < .01).mean()),
                        median_p=float(np.median(p)),
                        ks_stat=float(ks.statistic), ks_p=float(ks.pvalue))
        print(f'  {arm:14s} n={len(p):4d}  p<0.10 {(p<.10).mean():.3f}  '
              f'p<0.05 {(p<.05).mean():.3f}  p<0.01 {(p<.01).mean():.3f}  '
              f'median {np.median(p):.3f}  KS p={ks.pvalue:.2g}')
    (out / 'summary.json').write_text(json.dumps(res, indent=2, default=float))
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()

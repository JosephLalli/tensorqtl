"""Is the estimator wrong, or is the data not what the model assumes?

Simulates the response under hapmixQTL's OWN model, keeping the real v, the real
design, the real covariates and the real informative sets, and running the
baseline fit that was verified to reproduce map_cis to 7e-7. The true slope is
zero in every arm, so a correct estimator with a correct reference must give a
uniform nominal p.

  gaussian   errors ~ N(0, sigma^2 v) with the SAME v used as weights, so the
             model is correctly specified by construction. This isolates the
             machinery: if the p is non-uniform HERE, the estimator or its
             degrees of freedom are at fault and no story about the data can
             rescue it. If it is uniform, the real-data miscalibration comes
             from the data departing from the model, and the remaining arms say
             which departure.

  heavy      errors are scaled Student t with the MEASURED excess kurtosis --
             +2.312 on the allelic channel, +0.416 on the total. For a t with
             nu degrees of freedom the excess kurtosis is 6/(nu-4), giving
             nu = 6.6 and 18.4; variates are divided by sqrt(nu/(nu-2)) so the
             variance still matches sigma^2 v exactly. Variance correct, shape
             of the distribution wrong. Tests whether tails ALONE reproduce the
             observed miscalibration.

  shape      errors ~ N(0, sigma^2 v^gamma) while the estimator still weights by
             1/v. The measured regression of the standardized squared residual
             on log v has slope -0.064 on the allelic channel; since
             E[z^2] ~ v^(gamma-1) under this model, that slope identifies
             gamma = 0.936. Distribution Gaussian, variance shape wrong. Tests
             whether the shape error ALONE reproduces it.

Nothing is fitted to the simulated data beyond what the shipped estimator fits.
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

import compare_mixqtl_replication as CM   # noqa: E402
from se_fixes import fit_all              # noqa: E402  (verified against map_cis)

D = Path('/mnt/ssd/lalli/brainvar_hapmix_deploy')
RUN = D / 'rasqual_default_mode_20260923'
SEED, EPS = 42, 1e-12
NU_A, NU_T = 4 + 6 / 2.312, 4 + 6 / 0.416      # from the measured excess kurtosis
GAMMA = 0.936                                   # from the measured shape slope


def draw(rng, kind, scale2, v, nu):
    """Errors with variance scale2 * v (exactly), under one of three shapes."""
    if kind == 'gaussian':
        return rng.normal(0, np.sqrt(scale2 * v))
    if kind == 'heavy':
        z = rng.standard_t(nu, size=len(v)) / np.sqrt(nu / (nu - 2))
        return z * np.sqrt(scale2 * v)
    if kind == 'shape':
        return rng.normal(0, np.sqrt(scale2 * v ** GAMMA))
    raise ValueError(kind)


def main():
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    null46 = {l.strip() for l in open(D / 'genes_null46_20260923.txt') if l.strip()}
    me = pd.read_csv(RUN / 'matched_effects.tsv', sep='\t')
    targets = {r.gene: r.lead_r for r in me.itertuples()
               if r.gene in null46 and isinstance(r.lead_r, str)}
    I = CM.load_inputs(gene_list=str(D / 'genes_59_stratified_20260923.txt'),
                       regions=str(RUN / 'regions.bed'))
    keep = I['keep']
    import run_hapmixqtl_from_salmon as H
    with contextlib.redirect_stdout(io.StringIO()):
        A, T, Va, Vt, _ = H.compute_summaries_from_gibbs(
            I['YL'][:, keep], I['YR'][:, keep], yT=I['YT'][:, keep])
    C = I['cov_df'].values
    gi = {g: i for i, g in enumerate(I['genes'])}
    vi = {v: i for i, v in enumerate(I['vdf'].index)}
    print(f'nu_allelic={NU_A:.2f}  nu_total={NU_T:.2f}  gamma={GAMMA}\n'
          f'{len(targets)} genes x {R} replicates\n', flush=True)

    rng = np.random.RandomState(SEED)
    rows = []
    for g, var in targets.items():
        if g not in gi or var not in vi:
            continue
        i, j = gi[g], vi[var]
        s = (I['xL'][j] - I['xR'][j]).astype(float)
        gdos = I['dos'][j].astype(float) / 2.0
        va, vt = Va[i].copy(), Vt[i].copy()
        ok_a = np.isfinite(va) & (va > EPS)
        ok_t = np.isfinite(vt) & (vt > EPS)
        if ok_a.sum() < 10 or ok_t.sum() < 20:
            continue
        # realistic scales from the real data; the statistic is scale-free, so
        # these only keep the simulation in a sensible numeric range
        s2a = float(np.nansum(A[i][ok_a] ** 2 / va[ok_a]) / max(ok_a.sum() - 1, 1))
        s2t = float(np.nansum(T[i][ok_t] ** 2 / vt[ok_t]) / max(ok_t.sum() - 1, 1))
        va_s = np.where(ok_a, va, np.nan)
        vt_s = np.where(ok_t, vt, np.nan)
        for r in range(R):
            for kind in ('gaussian', 'heavy', 'shape'):
                a_sim = np.full(len(va), np.nan)
                t_sim = np.full(len(vt), np.nan)
                a_sim[ok_a] = draw(rng, kind, s2a, va[ok_a], NU_A)
                t_sim[ok_t] = draw(rng, kind, s2t, vt[ok_t], NU_T)
                f = fit_all(a_sim, s, va_s, t_sim, gdos, vt_s, C)
                if not f:
                    continue
                stat, dof = f['baseline']
                if np.isfinite(stat) and stat >= 0:
                    rows.append(dict(arm=kind, gene=g, rep=r,
                                     pval=float(sps.f.sf(stat, 1, dof))))
    t = pd.DataFrame(rows)
    out = D / 'parametric_bootstrap_20260924'
    out.mkdir(exist_ok=True)
    t.to_csv(out / 'pvals.tsv.gz', sep='\t', index=False)

    print('nominal p under simulation, true slope zero in every arm:')
    print('uniform means the estimator and its reference are correct\n')
    res = {'nu_allelic': float(NU_A), 'nu_total': float(NU_T), 'gamma': GAMMA}
    for arm in ('gaussian', 'heavy', 'shape'):
        p = t[t.arm == arm].pval.dropna().values
        if len(p) < 100:
            continue
        ks = sps.kstest(p, 'uniform')
        res[arm] = dict(n=int(len(p)), frac_below_10=float((p < .10).mean()),
                        frac_below_05=float((p < .05).mean()),
                        frac_below_01=float((p < .01).mean()),
                        median_p=float(np.median(p)),
                        ks_stat=float(ks.statistic), ks_p=float(ks.pvalue))
        print(f'  {arm:9s} n={len(p):6d}  p<0.10 {(p<.10).mean():.3f}  '
              f'p<0.05 {(p<.05).mean():.3f}  p<0.01 {(p<.01).mean():.3f}  '
              f'median {np.median(p):.3f}  KS p={ks.pvalue:.2g}')
    print('\n  for reference, the REAL data under permutation: '
          'p<0.10 0.139, p<0.05 0.080, p<0.01 0.020')
    (out / 'summary.json').write_text(json.dumps(res, indent=2, default=float))
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()

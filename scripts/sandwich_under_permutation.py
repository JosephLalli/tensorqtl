"""Does the permutation null absorb the sandwich's selection inflation?

The observed pass showed the sandwich statistic selects variants whose
standard error has collapsed, inflating max t^2 about 200-fold. That was an
observed-pass measurement only. Under permutation the DESIGN is unchanged,
so a variant carried by three donors is still carried by three donors in
every permuted replicate and its residual-based variance can collapse there
too. If the null maximum inflates as much as the observed maximum, the rank
is unaffected and the gene-level p-value remains valid.

Validity and power are separable here, so both are reported:

  validity  the gene-level p under null genotype permutations should be
            uniform, giving type-I near 0.05
  power     on the observed data, genes the model-based statistic calls
            should still be called

A statistic can pass the first and fail the second: if the null maximum is
dominated by whichever three-carrier variant happened to collapse in that
replicate, a real signal at a well-covered variant has to out-compete an
artifact, and loses.

Master seed 42.
"""

import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, 'scripts')
sys.path.insert(0, '.')

OUT = '/mnt/ssd/lalli/brainvar_hapmix_deploy/mixqtl_replication_20260919'
NPERM = 200
NULL_DRAWS = 5
SEED = 42


def _minp(beta, se2, ok, dof):
    """Smallest nominal p over the window, two-sided t."""
    from scipy.stats import t as tdist
    t2 = beta[ok] ** 2 / se2[ok]
    return float(np.nanmin(2.0 * tdist.sf(np.sqrt(np.nanmax(
        [t2, np.zeros_like(t2)], axis=0)), df=dof)))


def _beta_p(obs_minp, null_minp):
    """tensorQTL-style Beta approximation: fit Beta to the permutation
    minimum-p values, then read the observed minimum-p off the fit.
    Returns the Beta p, and a Kolmogorov-Smirnov statistic for how well the
    fitted Beta describes the permutation null it was fitted to."""
    from scipy.stats import beta as bdist, kstest
    x = np.asarray(null_minp, float)
    x = x[np.isfinite(x) & (x > 0) & (x < 1)]
    if len(x) < 30:
        return np.nan, np.nan
    try:
        aa, bb, _, _ = bdist.fit(x, floc=0, fscale=1)
    except Exception:
        return np.nan, np.nan
    ks = kstest(x, 'beta', args=(aa, bb, 0, 1)).statistic
    return float(bdist.cdf(obs_minp, aa, bb)), float(ks)


def gene_pvals(a, X, w, nperm, rng_base):
    """Observed pval_beta for model-based and sandwich SE, plus fit quality."""
    from se_forms_calibration import fit_all_se
    n = len(a)
    dof = max(n - 1, 1)
    beta, _k, se2_fit, se2_sand, _h3 = fit_all_se(a, X, w)
    ok = np.isfinite(beta) & (se2_fit > 0) & (se2_sand > 0)
    if ok.sum() < 20:
        return None
    obs_fit = _minp(beta, se2_fit, ok, dof)
    obs_sand = _minp(beta, se2_sand, ok, dof)

    nf, ns = [], []
    ge_fit = ge_sand = 0
    for k in range(nperm):
        prm = np.random.RandomState(rng_base + k).permutation(n)
        b, _k2, f2, s2, _h = fit_all_se(a[prm], X, w[prm])
        o2 = np.isfinite(b) & (f2 > 0) & (s2 > 0)
        if o2.sum() == 0:
            continue
        pf = _minp(b, f2, o2, dof)
        ps = _minp(b, s2, o2, dof)
        nf.append(pf); ns.append(ps)
        ge_fit += (pf <= obs_fit)
        ge_sand += (ps <= obs_sand)
    bf, ksf = _beta_p(obs_fit, nf)
    bs, kss = _beta_p(obs_sand, ns)
    return (bf, bs, (ge_fit + 1) / (nperm + 1), (ge_sand + 1) / (nperm + 1),
            ksf, kss)


def main():
    import tensorqtl.hapmixqtl as HM
    from compare_mixqtl_replication import load_inputs, gene_variant_index

    I = load_inputs()
    genes, keep = I['genes'], I['keep']
    A, _T, Va, _Vt, _C = HM.compute_summaries_from_gibbs(
        I['YL'], I['YR'], yT=I['YT'], count_noise=True)
    A, Va = A[:, keep], Va[:, keep]
    s_all = (I['xL'] - I['xR'])[I['idx']][:, keep]

    prepared = []
    for j, g in enumerate(genes):
        vsel = gene_variant_index(I, g)
        if vsel.size == 0:
            continue
        inf = Va[j] > 1e-12
        if int(inf.sum()) <= 2:
            continue
        S = s_all[vsel][:, inf]
        S = S[S.var(1) > 0]
        if S.shape[0] < 50:
            continue
        prepared.append((g, A[j][inf], S.T,
                         1.0 / np.maximum(Va[j][inf], 1e-12)))

    print('OBSERVED PASS: gene-level p, model-based vs sandwich\n')
    rows = []
    for g, a, X, w in prepared:
        r = gene_pvals(a, X, w, NPERM, SEED + 90001)
        if r is None:
            continue
        rows.append(dict(gene=g, p_model=r[0], p_sand=r[1],
                         perm_model=r[2], perm_sand=r[3],
                         ks_model=r[4], ks_sand=r[5]))
    d = pd.DataFrame(rows)
    called_m = d.p_model <= 0.05
    called_s = d.p_sand <= 0.05
    print(f'  genes called by the model-based statistic : '
          f'{int(called_m.sum())}/{len(d)}')
    print(f'  genes called by the sandwich statistic    : '
          f'{int(called_s.sum())}/{len(d)}')
    print(f'  called by model and kept by sandwich      : '
          f'{int((called_m & called_s).sum())}/{int(called_m.sum())}')
    print(f'  median pval_beta, model {d.p_model.median():.4f}   '
          f'sandwich {d.p_sand.median():.4f}')
    print(f'  Beta fit quality (KS vs its own permutation null): '
          f'model {d.ks_model.median():.4f}   sandwich {d.ks_sand.median():.4f}')
    print('\n  genes the model-based statistic calls:')
    for _, r in d[called_m].sort_values('p_model').iterrows():
        print(f'    {r.gene:10s} p_model {r.p_model:.4f}   '
              f'p_sandwich {r.p_sand:.4f}')

    print(f'\nNULL DRAWS: type-I over {NULL_DRAWS} genotype permutations\n')
    nrows = []
    for dnum in range(NULL_DRAWS):
        for g, a, X, w in prepared:
            nprm = np.random.RandomState(SEED + 777 + dnum).permutation(len(a))
            r = gene_pvals(a[nprm], X, w[nprm], NPERM,
                           SEED + 500000 + dnum * 1000)
            if r is not None:
                nrows.append(dict(draw=dnum, gene=g,
                                  p_model=r[0], p_sand=r[1],
                                  perm_model=r[2], perm_sand=r[3],
                                  ks_model=r[4], ks_sand=r[5]))
    nd = pd.DataFrame(nrows)
    t_m = float((nd.p_model <= 0.05).mean())
    t_s = float((nd.p_sand <= 0.05).mean())
    se = lambda p: float(np.sqrt(p * (1 - p) / len(nd)))
    print(f'  {len(nd)} gene-draw pairs')
    print(f'  type-I, model-based statistic : {t_m:.4f} +/- {se(t_m):.4f}')
    print(f'  type-I, sandwich statistic    : {t_s:.4f} +/- {se(t_s):.4f}')
    print(f'  median null pval_beta, model {nd.p_model.median():.3f}   '
          f'sandwich {nd.p_sand.median():.3f}')
    print(f'  Beta fit KS under the null: model {nd.ks_model.median():.4f}   '
          f'sandwich {nd.ks_sand.median():.4f}')
    print(f'  Beta fit returned NaN: model {int(nd.p_model.isna().sum())}, '
          f'sandwich {int(nd.p_sand.isna().sum())} of {len(nd)}')
    print()
    print('  DECOMPOSITION -- is it the permutation or the Beta fit?')
    pm = float((nd.perm_model <= 0.05).mean())
    ps = float((nd.perm_sand <= 0.05).mean())
    print(f'  type-I from the EMPIRICAL permutation rank (no Beta):')
    print(f'    model {pm:.4f} +/- {se(pm):.4f}   '
          f'sandwich {ps:.4f} +/- {se(ps):.4f}')
    print(f'  type-I from the BETA FIT:')
    print(f'    model {t_m:.4f} +/- {se(t_m):.4f}   '
          f'sandwich {t_s:.4f} +/- {se(t_s):.4f}')
    print('  if the rank is near 0.05 and the Beta fit is not, the')
    print('  permutation absorbs the collapse and the Beta approximation '
          'is what breaks.')

    d.to_csv(f'{OUT}/sandwich_observed_pvals.tsv', sep='\t', index=False)
    nd.to_csv(f'{OUT}/sandwich_null_pvals.tsv', sep='\t', index=False)
    json.dump(dict(
        nperm=NPERM, null_draws=NULL_DRAWS,
        n_called_model=int(called_m.sum()), n_called_sand=int(called_s.sum()),
        n_retained=int((called_m & called_s).sum()),
        typeI_model=t_m, typeI_sandwich=t_s,
        typeI_n=len(nd),
        median_null_p_model=float(nd.p_model.median()),
        median_null_p_sandwich=float(nd.p_sand.median()),
        typeI_rank_model=float((nd.perm_model <= 0.05).mean()),
        typeI_rank_sandwich=float((nd.perm_sand <= 0.05).mean()),
        ks_model=float(nd.ks_model.median()),
        ks_sandwich=float(nd.ks_sand.median()),
        beta_nan_model=int(nd.p_model.isna().sum()),
        beta_nan_sandwich=int(nd.p_sand.isna().sum()),
    ), open(f'{OUT}/sandwich_under_permutation.json', 'w'), indent=1)


if __name__ == '__main__':
    main()

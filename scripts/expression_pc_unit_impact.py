"""What moving the expression PCs from log2(CPM + 1) to the half-read log-CPM (2026-09-30) does to the half-read split
default: each simulated-effects dataset below is fitted twice through common.run_nominal, once with each covariate build,
everything else fixed; prints the per-row slope change in standard errors, the agreement of nominal p-values, the
nominal-p rate on the no-effect dataset and the causal slope over the simulated effect.

  [SIMULATED_EFFECTS_GENE_SET=stratum30_100] CUDA_VISIBLE_DEVICES=1 python3 scripts/expression_pc_unit_impact.py
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'benchmark' / 'simulated_effects'))
import common as C          # noqa: E402

BUILDS = {'log2cpm1': C.D / 'cov' / 'log2cpm1_point_calibration_20260925',
          'half_read': C.D / 'cov' / 'half_read_point_calibration_20260930'}
RUNS = [('beta0.0', 0), ('beta0.8', 0), ('beta0.8', 1), ('beta0.8', 2)]
ALPHAS = (0.05, 0.01, 0.001)


def covs(path, order):
    c = pd.read_csv(path / 'covariates.tsv', sep='\t', index_col=0).loc[order]
    g = (path / 'genotype_covariates.txt').read_text().split()
    return c.drop(columns=g), c[g]


I, R, _ = C.load()
S = C.setup(I)
order = list(I['order'])
new_rna, new_geno = covs(BUILDS['half_read'], order)
if not (new_rna.equals(I['cov_df'].loc[order]) and new_geno.equals(I['geno_cov_df'].loc[order])):
    raise SystemExit('the shared loader does not supply the half-read build')
scratch = Path(tempfile.mkdtemp(dir=C.D / 'cov'))
for sc, r in RUNS:
    ds = C.load_dataset(C.DATASETS, sc, r)
    ds['T'] = np.log2((ds['pT'] + 0.5) / (ds['eff_lib'][None, :] + 1.0) * 1e6)   # the half-read default's total
    fits = {}
    for name, path in BUILDS.items():
        S['I']['cov_df'], S['I']['geno_cov_df'] = covs(path, order)
        fits[name] = C.run_nominal(S, ds, 'split', scratch)[0].set_index(['phenotype_id', 'variant_id'])
    a, b = fits['log2cpm1'], fits['half_read'].loc[fits['log2cpm1'].index]
    ok = np.isfinite(a.slope_se) & np.isfinite(b.slope_se)
    dz = ((b.slope - a.slope) / a.slope_se)[ok]
    lp = lambda x: -np.log10(x.astype(float).clip(lower=1e-300))   # noqa: E731
    line = (f'{C.GENE_SET} {sc} rep {r}: {int(ok.sum()):,} rows; |slope change| / se median {dz.abs().median():.4f} '
            f'max {dz.abs().max():.3f}; corr -log10 p {np.corrcoef(lp(a.pval_nominal[ok]), lp(b.pval_nominal[ok]))[0, 1]:.5f}')
    if sc == 'beta0.0':
        for ch in ('pval_nominal', 'pval_t'):
            line += f'; {ch} rate ' + ' / '.join(f'{(a[ch] < al).mean():.4f}->{(b[ch] < al).mean():.4f}' for al in ALPHAS)
    else:
        nn = ~ds['is_null']
        keys = list(zip(np.array(S['genes'])[nn], ds['causal_variant'][nn].astype(str)))
        beta = ds['beta'][nn]
        ra, rb = a.loc[keys, 'slope'].values / beta, b.loc[keys, 'slope'].values / beta
        line += f'; causal combined slope / beta {ra.mean():.4f} -> {rb.mean():.4f} (paired change {np.mean(rb - ra):+.4f})'
    print(line, flush=True)
for q in scratch.glob('*'):
    q.unlink()
scratch.rmdir()

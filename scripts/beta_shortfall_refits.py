"""Counterfactual refits behind the simulated-effects slope's shortfall from the simulated effect (the budget of
beta_shortfall_budget.py says where it sits; these say why). Every fit is common.run_nominal, the shipped map_nominal in
default mode, on a dataset's arrays with one ingredient changed:

  observed     the dataset as stored (known answer: reproduces the stored causal-variant slopes exactly)
  target       A and T replaced by the noise-free simulated shifts on the pipeline scale (a*, t* of
               02_make_datasets.pipeline_truth), weights and admission as observed: what the estimator's own design,
               covariates and weights estimate when there is no noise
  va_real      the allelic Gibbs variance of the unthinned record (Va at f = 1); admission as observed
  kept_real    the allelic admission rule applied to the unthinned record's pL, pR; variance as observed
  both_real    both; target_real is the target under both_real's variance and admission
  voom         total phenotype log2((pT + 0.5) / (Leff + 1) * 1e6), voom's log-CPM (Law et al. 2014, a 0.5-read
               pseudocount), in place of log2(pT / Leff * 1e6 + 1) (a pseudocount of Leff / 1e6 reads, 17 at the
               median library); target_voom is its noise-free shift

for the split and unit weightings at |beta| 0.4 and 0.8 (0.2's per-unit ratios are too noisy to split), plus observed
and voom on the beta 0 dataset for the nominal-p rate on null genes. A counterfactual is a measurement, not a proposed
setting: the transform and the admission rule are standing user decisions (docs/pipeline_rules.md).

Scores (means over causal units with 06_score.py's gene-clustered interval, per read band): each channel's slope over the
count-scale truth (beta allelic, the per-gene total truth total, beta combined); target over the same; the noise bias
(slope - target) / beta; paired differences between configurations; the signed z (slope / se in the simulated direction)
of the combined and total tests, observed and voom. The combined target is the inverse-variance combination of the
channel targets at the observed run's channel se.

  SIMULATED_EFFECTS_GENE_SET=<set> CUDA_VISIBLE_DEVICES=1 python3 scripts/beta_shortfall_refits.py
  # writes OUT/refits_<set>.parquet (skipped, with a printed line, when present), OUT/refits_<set>.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'benchmark' / 'simulated_effects'))
import common as C          # noqa: E402

G2, S6 = C.module('02_make_datasets'), C.module('06_score')
OUT = C.D / 'beta_shortfall_20260929'
BETAS = (0.4, 0.8)
ARMS = ('split', 'unit')
CONFIGS = ('observed', 'target', 'va_real', 'kept_real', 'both_real', 'target_real', 'voom', 'target_voom')
NULL_CONFIGS = ('observed', 'voom')
VOOM_PRIOR = 0.5            # limma::voom: log2((counts + 0.5) / (lib.size + 1) * 1e6)
KEEP = ['slope', 'slope_se', 'slope_a', 'slope_a_se', 'slope_t', 'slope_t_se', 'allelic_admitted']


def shifts(M, fL, fR):
    """Noise-free per-record shifts a*, t* (pipeline scale) and t*_voom, as 02_make_datasets.pipeline_truth forms them."""
    pL, pR, pT = M['pL'], M['pR'], M['pT']
    a = np.log2((fL * pL + C.KAPPA) / (fR * pR + C.KAPPA)) - np.log2((pL + C.KAPPA) / (pR + C.KAPPA))
    U = G2.remainder(pL, pR, pT)
    k = 1e6 / M['eff_lib'][None, :]
    pT_exp = pT - (1 - fL) * pL - (1 - fR) * pR - (1 - (fL + fR) / 2) * U
    t = np.log2(k * pT_exp + 1.0) - np.log2(k * pT + 1.0)
    return a, t, np.log2(pT_exp + VOOM_PRIOR) - np.log2(pT + VOOM_PRIOR)


def configs(ds, M, a, t, tv, names):
    real = dict(Va=G2.allelic_variance(M['pL'], M['pR'], M['pL'], M['pR'], M['YL'], M['YR']), pL=M['pL'], pR=M['pR'])
    voom_T = np.log2((ds['pT'] + VOOM_PRIOR) / (ds['eff_lib'][None, :] + 1.0) * 1e6)
    V = dict(observed={}, target=dict(A=a, T=t), va_real=dict(Va=real['Va']), kept_real=dict(pL=real['pL'], pR=real['pR']),
             both_real=real, target_real=real | dict(A=a, T=t), voom=dict(T=voom_T), target_voom=dict(A=a, T=tv))
    return {n: ds | V[n] for n in names}


def check_dataset(I, ds, M, a, t, sc, r):
    """Moved records reproduce the stored Va; a*, t* reproduce the stored pipeline truths (non-null genes)."""
    if not np.array_equal(G2.allelic_variance(M['pL'], M['pR'], ds['pL'], ds['pR'], M['YL'], M['YR']), ds['Va']):
        raise SystemExit(f'{sc} rep {r}: moved records do not reproduce the stored Va')
    j = np.array([I['vdf'].index.get_loc(v) for v in ds['causal_variant']])
    s = I['xL'][j].astype(float) - I['xR'][j]
    kept = C.allelic_kept(ds['pL'], ds['pR'], ds['Va'])
    ba, bt = G2.pipeline_truth(M, ds['fL'], ds['fR'], s, I['dos'][j].astype(float), kept)
    sk = s * kept
    ba2 = (sk * a).sum(1) / (sk ** 2).sum(1)
    bt2 = G2.slope_with_intercept(I['dos'][j].astype(float) / 2, t)
    nn = ~ds['is_null']
    for name, x, y, z in (('allelic', ba, ba2, ds['allelic_truth_pipeline']), ('total', bt, bt2, ds['total_truth_pipeline'])):
        if not (np.allclose(x[nn], z[nn], rtol=0, atol=1e-12, equal_nan=True) and np.allclose(y[nn], z[nn], rtol=0, atol=1e-12, equal_nan=True)):
            raise SystemExit(f'{sc} rep {r}: {name} shifts do not reproduce the stored pipeline truth')


def refit(I, R, S, scratch):
    rows, nulls = [], []
    runs = [(b, r) for b in (0.0,) + BETAS for r in range(json.loads((C.DATASETS / 'meta.json').read_text())['n_datasets'][str(b)])]
    for b, r in runs:
        sc = f'beta{b}'
        ds = C.load_dataset(C.DATASETS, sc, r)
        M = G2.move_records(R, ds['perm'], ds['swap'])
        a, t, tv = shifts(M, ds['fL'], ds['fR'])
        check_dataset(I, ds, M, a, t, sc, r)
        causal = pd.DataFrame(dict(gene=S['genes'], variant_id=ds['causal_variant'].astype(str)))[~ds['is_null']]
        for arm in ARMS:
            stored = C.read_results(C.RESULTS / sc / arm / f'nominal_rep{r:03d}.parquet', C.COLS + ['allelic_admitted'])
            for name, d in configs(ds, M, a, t, tv, NULL_CONFIGS if b == 0 else CONFIGS).items():
                t0 = time.perf_counter()
                df = C.run_nominal(S, d, arm, scratch)[0]
                if name == 'observed':
                    same = df[C.COLS].reset_index(drop=True).equals(stored[C.COLS].reset_index(drop=True))
                    if not same:
                        raise SystemExit(f'{sc} rep {r} {arm}: observed refit differs from {C.RESULTS / sc / arm}')
                if b == 0:
                    for col in ('pval_nominal', 'pval_t'):
                        p = df.groupby('phenotype_id')[col]
                        nulls.append(pd.DataFrame(dict(rep=r, arm=arm, config=name, channel=col, gene=S['genes'],
                                                       tests=p.count().reindex(S['genes']).values,
                                                       **{f'k{al}': p.apply(lambda x, al=al: int((x < al).sum()))
                                                          .reindex(S['genes']).values for al in C.ALPHAS})))
                else:
                    c = causal.merge(df, left_on=['gene', 'variant_id'], right_on=['phenotype_id', 'variant_id'], how='left')
                    if c.slope.isna().any():
                        raise SystemExit(f'{sc} rep {r} {arm} {name}: {int(c.slope.isna().sum())} causal variants without a row')
                    rows.append(c[['gene', 'variant_id'] + KEEP].assign(scenario=sc, rep=r, arm=arm, config=name))
                print(f'{C.GENE_SET} {sc} rep {r} {arm} {name}: {len(df):,} rows, {time.perf_counter() - t0:.1f} s', flush=True)
    return pd.concat(rows, ignore_index=True), pd.concat(nulls, ignore_index=True)


def score(F, Nl, U, genes, bsel, bidx):
    """Ratios, noise bias and paired differences per (scenario, arm), and the beta 0 nominal-p rates."""
    out = {}
    unit = ['scenario', 'rep', 'gene']
    truths = U[~U.is_null][unit + ['beta', 'total_truth', 'band']]
    for (sc, arm), g in F.groupby(['scenario', 'arm']):
        W = {n: g[g.config == n].merge(truths, on=unit, how='left').set_index(unit) for n in CONFIGS}
        base = W['observed']
        Cz = base.reset_index()
        blk = lambda v: S6.ratio_block(pd.Series(np.asarray(v, float), index=Cz.index), Cz, genes, bsel, bidx)   # noqa: E731
        b, tc = base.beta.values, base.total_truth.values

        def combined_target(tgt, obs):
            wa = np.where(obs.allelic_admitted.astype(bool) & np.isfinite(obs.slope_a_se), 1 / obs.slope_a_se.astype(float) ** 2, 0)
            wt = 1 / obs.slope_t_se.astype(float) ** 2
            return (np.where(wa > 0, wa * tgt.slope_a.astype(float), 0) + wt * tgt.slope_t.astype(float)) / (wa + wt)

        res = {}
        for n in ('observed', 'va_real', 'kept_real', 'both_real', 'voom'):
            x = W[n]
            res[n] = dict(allelic=blk(x.slope_a.astype(float) / b), total=blk(x.slope_t.astype(float) / tc),
                          combined=blk(x.slope.astype(float) / b))
        for n, obs in (('target', 'observed'), ('target_real', 'both_real'), ('target_voom', 'voom')):
            x = W[n]
            res[n] = dict(allelic=blk(x.slope_a.astype(float) / b), total=blk(x.slope_t.astype(float) / tc),
                          combined=blk(combined_target(x, W[obs]) / b))
            res[f'noise_{obs}'] = dict(allelic=blk((W[obs].slope_a.astype(float) - x.slope_a.astype(float)) / b),
                                       total=blk((W[obs].slope_t.astype(float) - x.slope_t.astype(float)) / b),
                                       combined=blk((W[obs].slope.astype(float) - combined_target(x, W[obs])) / b))
        res['unadjusted_truth'] = dict(total=blk(Cz.merge(U[unit + ['total_truth_pipeline']], on=unit, how='left')
                                                 .total_truth_pipeline.values / tc))
        for n in ('va_real', 'kept_real', 'both_real'):
            res[f'{n}_minus_observed'] = {ch: blk((W[n][col].astype(float) - base[col].astype(float)) / b)
                                          for ch, col in (('allelic', 'slope_a'), ('combined', 'slope'))}
        z = {n: {ch: np.sign(b) * W[n][s].astype(float).values / W[n][se].astype(float).values
                 for ch, (s, se) in (('combined', ('slope', 'slope_se')), ('total', ('slope_t', 'slope_t_se')))}
             for n in ('observed', 'voom')}
        res['z_observed'] = {ch: blk(v) for ch, v in z['observed'].items()}
        res['z_voom'] = {ch: blk(v) for ch, v in z['voom'].items()}
        res['z_voom_minus_observed'] = {ch: blk(z['voom'][ch] - z['observed'][ch]) for ch in z['voom']}
        out.setdefault(sc, {})[arm] = res
    nul = {}
    for (arm, cfg, ch), g in Nl.groupby(['arm', 'config', 'channel']):
        g = g.set_index('gene').loc[genes]
        nul.setdefault(arm, {}).setdefault(cfg, {})[ch] = {
            str(al): S6.pooled(g[f'k{al}'].values[None, :].astype(float), g.tests.values[None, :].astype(float),
                               bsel['all'], bidx['all']) for al in C.ALPHAS}
    return dict(recovery=out, null=nul)


def main():
    meta, genes, U, keep_a = S6.load_units(C.DATASETS, C.RESULTS)
    bsel, bidx = S6.band_selections(genes, U, keep_a)
    rpath, npath = OUT / f'refits_{C.GENE_SET}.parquet', OUT / f'refits_null_{C.GENE_SET}.parquet'
    if rpath.exists() and npath.exists():
        print(f'skip refits: {rpath} and {npath} exist', flush=True)
    else:
        I, R, _ = C.load()
        S = C.setup(I)
        F, Nl = refit(I, R, S, OUT / f'scratch_{C.GENE_SET}')
        OUT.mkdir(exist_ok=True)
        C.write_atomic(rpath, lambda fh: F.to_parquet(fh, index=False))
        C.write_atomic(npath, lambda fh: Nl.to_parquet(fh, index=False))
    F, Nl = pd.read_parquet(rpath), pd.read_parquet(npath)
    print(f'{len(F):,} causal-variant rows, {len(Nl):,} null gene rows', flush=True)
    res = score(F, Nl, U, genes, bsel, bidx)
    C.write_json(OUT / f'refits_{C.GENE_SET}.json', dict(gene_set=C.GENE_SET, root=str(C.ROOT), betas=list(BETAS), arms=list(ARMS),
                                                         configs=list(CONFIGS), voom_prior=VOOM_PRIOR, **res))
    for sc, A in res['recovery'].items():
        for arm, r in A.items():
            f = lambda d, ch: f'{d[ch]["all"]["mean"]:.3f} [{d[ch]["all"]["lo"]:.3f}, {d[ch]["all"]["hi"]:.3f}]'   # noqa: E731
            print(f'{sc} {arm}: allelic observed {f(r["observed"], "allelic")} target {f(r["target"], "allelic")} '
                  f'noise {f(r["noise_observed"], "allelic")}; both_real {f(r["both_real"], "allelic")} noise '
                  f'{f(r["noise_both_real"], "allelic")}; va_real-obs {f(r["va_real_minus_observed"], "allelic")} kept_real-obs '
                  f'{f(r["kept_real_minus_observed"], "allelic")} | total observed {f(r["observed"], "total")} target '
                  f'{f(r["target"], "total")} unadjusted {f(r["unadjusted_truth"], "total")} | voom total {f(r["voom"], "total")} '
                  f'combined {f(r["voom"], "combined")} vs {f(r["observed"], "combined")}', flush=True)
    for arm, d in res['null'].items():
        for cfg, e in d.items():
            print(f'beta 0 {arm} {cfg}: ' + '; '.join(f'{ch} ' + ' / '.join(f'{v["rate"]:.4f}' for v in e[ch].values()) for ch in e))
    print(f'wrote {OUT / f"refits_{C.GENE_SET}.json"}')


if __name__ == '__main__':
    main()

"""Effect-size recovery of the current default (half-read total phenotype, split weighting, expression PCs in the
half-read unit since 2026-09-30) on the simulated-effects causal units: each dataset at |beta| 0.4 and 0.8 fitted through
common.run_nominal with the shared loader's covariates. Scores, per read band with 06_score.py's gene-clustered
interval: combined slope / beta, allelic slope / beta, total slope / count-scale total truth, and the combined
shortfall 1 - slope / beta split into its allelic and total parts (each channel's share of the inverse-variance weight
times its own shortfall). Stored TReCASE (alignment counts) recovery is printed beside it for reference.

Known answer first: with the 2026-09-25 covariates the fits reproduce the stored half-read refits of
beta_shortfall_refits.py (config voom, split) exactly.

  [SIMULATED_EFFECTS_GENE_SET=stratum30_100] CUDA_VISIBLE_DEVICES=1 python3 scripts/beta_recovery_current.py
  # writes OUT/recovery_<set>.json
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'benchmark' / 'simulated_effects'))
import common as C          # noqa: E402

S6 = C.module('06_score')
OUT = C.D / 'beta_recovery_current_20260930'
OLD_COV = C.D / 'cov' / 'log2cpm1_point_calibration_20260925'
REFITS = C.D / 'beta_shortfall_20260929' / f'refits_{C.GENE_SET}.parquet'
BETAS = (0.4, 0.8)
KEEP = ['slope', 'slope_se', 'slope_a', 'slope_a_se', 'slope_t', 'slope_t_se', 'allelic_admitted']


def half_read(ds):
    return ds | dict(T=np.log2((ds['pT'] + 0.5) / (ds['eff_lib'][None, :] + 1.0) * 1e6))


def causal_rows(S, ds, df):
    nn = ~ds['is_null']
    keys = pd.MultiIndex.from_arrays([np.array(S['genes'])[nn], ds['causal_variant'][nn].astype(str)])
    return df.set_index(['phenotype_id', 'variant_id']).loc[keys, KEEP].reset_index()


def known_answer(S, scratch):
    """With the 2026-09-25 covariates, rep 0 at |beta| 0.8 reproduces the stored voom/split refit exactly."""
    order = list(S['I']['order'])
    cur = S['I']['cov_df'], S['I']['geno_cov_df']
    c = pd.read_csv(OLD_COV / 'covariates.tsv', sep='\t', index_col=0).loc[order]
    g = (OLD_COV / 'genotype_covariates.txt').read_text().split()
    S['I']['cov_df'], S['I']['geno_cov_df'] = c.drop(columns=g), c[g]
    ds = C.load_dataset(C.DATASETS, 'beta0.8', 0)
    got = causal_rows(S, ds, C.run_nominal(S, half_read(ds), 'split', scratch)[0])
    S['I']['cov_df'], S['I']['geno_cov_df'] = cur
    F = pd.read_parquet(REFITS)
    ref = F[(F.config == 'voom') & (F.arm == 'split') & (F.scenario == 'beta0.8') & (F.rep == 0)]
    ref = ref.set_index(['gene', 'variant_id']).loc[list(zip(got.phenotype_id, got.variant_id)), KEEP[:-1]]
    if not np.array_equal(got[KEEP[:-1]].to_numpy(float), ref.to_numpy(float), equal_nan=True):
        raise SystemExit('with the 2026-09-25 covariates the fits do not reproduce the stored half-read refits')
    print(f'known answer: {len(got)} causal rows reproduce the stored half-read refit exactly', flush=True)


def main():
    meta, genes, U, keep_a = S6.load_units(C.DATASETS, C.RESULTS)
    bsel, bidx = S6.band_selections(genes, U, keep_a)
    I, _, _ = C.load()
    S = C.setup(I)
    scratch = Path(tempfile.mkdtemp(dir=C.D / 'cov'))
    known_answer(S, scratch)
    rows = []
    for b in BETAS:
        sc = f'beta{b}'
        for r in range(meta['n_datasets'][str(b)]):
            ds = C.load_dataset(C.DATASETS, sc, r)
            got = causal_rows(S, ds, C.run_nominal(S, half_read(ds), 'split', scratch)[0])
            rows.append(got.rename(columns={'phenotype_id': 'gene'}).assign(scenario=sc, rep=r))
            print(f'{C.GENE_SET} {sc} rep {r}: {len(got)} causal units', flush=True)
    for q in scratch.glob('*'):
        q.unlink()
    scratch.rmdir()
    F = pd.concat(rows, ignore_index=True).merge(
        U[~U.is_null][['scenario', 'rep', 'gene', 'beta', 'total_truth']], on=['scenario', 'rep', 'gene'], how='left')
    stored = json.loads(C.SUMMARY.read_text())['recovery']
    res = {}
    for sc, g in F.groupby('scenario'):
        Cz = g.reset_index(drop=True)
        blk = lambda v: S6.ratio_block(pd.Series(np.asarray(v, float), index=Cz.index), Cz, genes, bsel, bidx)   # noqa: E731
        b, tc = Cz.beta.values, Cz.total_truth.values
        sa, sea = Cz.slope_a.astype(float).values, Cz.slope_a_se.astype(float).values
        st, set_ = Cz.slope_t.astype(float).values, Cz.slope_t_se.astype(float).values
        adm = Cz.allelic_admitted.astype(bool).values & np.isfinite(sa) & np.isfinite(sea)
        wa = np.where(adm, 1.0 / np.where(adm, sea, 1.0) ** 2, 0.0)
        wt = np.where(np.isfinite(set_), 1.0 / set_ ** 2, 0.0)
        pa, pt = wa / (wa + wt), wt / (wa + wt)
        short = 1.0 - Cz.slope.astype(float).values / b
        part_a = np.where(adm, pa * (1.0 - sa / b), 0.0)
        part_t = np.where(wt > 0, pt * (1.0 - st / b), 0.0)
        if np.abs(part_a + part_t - short).max() > 1e-6:   # float32 slopes combined on the GPU
            raise SystemExit(f'{sc}: channel parts do not add up to the combined shortfall')
        res[sc] = dict(combined=blk(Cz.slope.astype(float).values / b), allelic=blk(np.where(adm, sa / b, np.nan)),
                       total=blk(np.where(wt > 0, st / tc, np.nan)), shortfall=blk(short),
                       shortfall_allelic=blk(part_a), shortfall_total=blk(part_t), allelic_share=blk(pa),
                       trecase_native=stored[sc]['trecase_native']['combined']['bias_count'])
        f = lambda d: f'{d["all"]["mean"]:.3f} [{d["all"]["lo"]:.3f}, {d["all"]["hi"]:.3f}]'   # noqa: E731
        x = res[sc]
        print(f'{C.GENE_SET} {sc}: combined {f(x["combined"])}; allelic {f(x["allelic"])}; total {f(x["total"])}; '
              f'shortfall {f(x["shortfall"])} = allelic part {f(x["shortfall_allelic"])} + total part '
              f'{f(x["shortfall_total"])} (allelic weight share {x["allelic_share"]["all"]["mean"]:.2f}); '
              f'TReCASE native {f(x["trecase_native"])}', flush=True)
        for bn in [k for k in x['combined'] if k not in ('all', 'without one-df genes')]:
            print(f'   band {bn}: combined {x["combined"][bn]["mean"]:.3f} [{x["combined"][bn]["lo"]:.3f}, '
                  f'{x["combined"][bn]["hi"]:.3f}], TReCASE {x["trecase_native"][bn]["mean"]:.3f}', flush=True)
    OUT.mkdir(exist_ok=True)
    C.write_json(OUT / f'recovery_{C.GENE_SET}.json', dict(gene_set=C.GENE_SET, covariates=str(C.COV),
                                                           betas=list(BETAS), result=res))
    C.write_atomic(OUT / f'causal_units_{C.GENE_SET}.parquet', lambda fh: F.to_parquet(fh, index=False))
    print(f'wrote {OUT / f"recovery_{C.GENE_SET}.json"}')


if __name__ == '__main__':
    main()

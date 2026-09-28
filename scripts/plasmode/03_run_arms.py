"""Map every plasmode dataset under six arms (README: Arms): four hapmixQTL weightings through
map_nominal (nominal p per tested variant) and map_cis (gene-level pval_perm and pval_beta from
NPERM records_signflip permutations, GPU), and mixQTL mode at two cutoff settings through
mixqtl_scan on the thinned point estimates (never the draws).

hapmixQTL arms (common.arm_variances, after the zero-haplotype admission): gibbs = Gibbs variance
in both channels (the shipped default); split = Gibbs allelic, unit total; unit = 1 everywhere;
plus_one = v + 1 in both. map_nominal and map_cis get the RNA-tied covariates in the dataset's
record order, the genotype PCs in place, the allelic channel through the origin, window CM.WIN,
default mode; A is already swapped in the dataset. Stored per tested variant: COLS + DOF_COLS
(map_nominal's own dtypes; each p's t reference and whether the gene's allelic channel entered the
combination, hapmixqtl.MIN_ALLELIC_DONORS = 15). map_cis seed: SeedSequence(SEED, (MAPCIS_KEY, r)),
one integer per dataset index shared by the four arms and the scenarios; map_cis(seed=...) calls
np.random.seed, the one place this script touches global numpy state.

mixQTL arms: MX.mixqtl_scan per gene on its tested variants with lib_size = eff_lib, covariates in
record order, genotype PCs in place, h1 / h2 = xL / xR; COLS in NATURAL LOG (the parquet metadata
records the unit; 06_score.py divides by ln 2) plus `method` (meta, trc or asc: which estimate the
meta columns hold when a channel has fewer than MX.META_N_CUTOFF samples). mixQTL's own gene-level
permutation scan is not run: timed on 2026-09-26 at 307 s for 24 of 100 genes against a 300 s
budget per dataset (user decision), and for the 30-100-read set on 2026-09-27 at 302 s for 65 of 100
genes on its first dataset; the gene set's record (MIXQTL_PERM_TIMED) is copied into RESULTS.

Tested variants: the loader's idx set within each gene's window is exactly map_nominal's output
(row count checked per call); 507 tested variants have every donor heterozygous, which map_cis
drops as monomorphic (its num_var is the count with varying dosage).

Output: RESULTS/<scenario>/<arm>/nominal_repNNN.parquet, cis_repNNN.parquet (hapmixQTL, CIS_COLS),
RESULTS/mixqtl_permutation.json, RESULTS/run_arms_facts.json (the per-run counts 08_report.py
reads). Every output is recomputed on every run.
"""
import json
import shutil
import time

import numpy as np
import pandas as pd
import torch

import common as C
from tensorqtl.hapmixqtl import map_cis

MAPCIS_KEY = 4                 # spawn key after 02_make_datasets' 1 / 2 / 3
NPERM = 1000                   # map_cis on every dataset (user decision 2026-09-26): ~17 s per dataset per arm on one L4
PERM_SCHEME = 'records_signflip'
MIXQTL_PERM_TIMED = C.MIXQTL_PERM_TIMED   # the 2026-09-26 timing behind leaving the mixQTL permutation scan out (common.GENE_SETS)
CIS_COLS = ['phenotype_id', 'variant_id', 'num_var', 'pval_nominal', 'slope', 'slope_se', 'slope_a', 'slope_a_se',
            'slope_t', 'slope_t_se', 'pval_perm', 'pval_beta', 'beta_shape1', 'beta_shape2', 'true_df']


def cis_seed(r):
    return int(np.random.SeedSequence(C.SEED, spawn_key=(MAPCIS_KEY, r)).generate_state(1)[0])


def run_cis(S, ds, arm, seed):
    """map_cis on the arm's inputs; tau_refit is inert in default mode (map_cis refits only for tau_mode 'estimate')."""
    A, T, Va, Vt, cov, _ = C.phenotypes(S, ds, arm)
    res = C.quiet(map_cis, S['gdf'], S['vdf'][['chrom', 'pos']], A, T, Va, Vt, S['gp'], xL_df=S['xLdf'], xR_df=S['xRdf'],
                  covariates_df=cov, genotype_covariates_df=S['I']['geno_cov_df'], window=C.CM.WIN, nperm=NPERM,
                  seed=seed, perm_scheme=PERM_SCHEME, tau_refit=True, verbose=False, ase_covariates_df=None)
    res = res.reset_index()[CIS_COLS]
    res['variant_id'] = res['variant_id'].astype(str)
    if list(res.phenotype_id) != S['genes']:
        raise SystemExit(f'map_cis returned {len(res)} genes, not the {len(S["genes"])} in order')
    return res


def run_mixqtl(S, ds, cutoffs):
    """mixqtl_scan per gene on the tested variants: COLS in natural log plus `method`; allelic and total sample counts."""
    I, MX = S['I'], C.MX
    y1, y2, yt = MX.inputs_from_point_estimates(ds['pL'], ds['pR'], ds['pT'])
    cov, G = I['cov_df'].values[ds['perm']], I['geno_cov_df'].values
    parts, n_asc, n_trc = [], [], []
    for k, g in enumerate(S['genes']):
        rows = S['tested_rows'][g]
        o = MX.mixqtl_scan(y1[k], y2[k], yt[k], ds['eff_lib'], I['xL'][rows].T.astype(float), I['xR'][rows].T.astype(float),
                           covariates=cov, genotype_covariates=G, **cutoffs)
        m, a, t = o['meta'], o['asc'], o['trc']
        parts.append(pd.DataFrame(dict(
            phenotype_id=g, variant_id=I['vdf'].index[rows].astype(str), pval_nominal=m['pval'], slope=m['beta'],
            slope_se=m['se'], pval_a=a['pval'], slope_a=a['beta'], slope_a_se=a['se'], pval_t=t['pval'],
            slope_t=t['beta'], slope_t_se=t['se'], method=m['method'].astype(str))))
        n_asc.append(a['sample_size'])
        n_trc.append(t['sample_size'])
    df = pd.concat(parts, ignore_index=True)
    if len(df) != int(S['n_tested'].sum()):
        raise SystemExit(f'mixqtl_scan returned {len(df):,} rows against {int(S["n_tested"].sum()):,} tested pairs')
    return df, np.array(n_asc), np.array(n_trc)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # tensorqtl/hapmixqtl.py picks the same
    print(f'torch {torch.__version__}; map_cis device {device}'
          + (f' ({torch.cuda.get_device_name(device)})' if device.type == 'cuda' else ''), flush=True)
    S = C.setup(C.load()[0])
    meta = json.loads((C.DATASETS / 'meta.json').read_text())
    if S['genes'] != meta['genes'] or S['order'] != meta['donors']:
        raise SystemExit(f'{C.DATASETS / "meta.json"}: genes or donors differ from the cache inputs')
    C.RESULTS.mkdir(parents=True, exist_ok=True)
    C.write_json(C.RESULTS / 'mixqtl_permutation.json',
                 dict(json.loads(MIXQTL_PERM_TIMED.read_text()), source=str(MIXQTL_PERM_TIMED)))
    scratch, secs, facts = C.RESULTS / 'scratch', {}, {}
    for sc, r in C.runs(meta):
        ds, seed, tag = C.load_dataset(C.DATASETS, sc, r), cis_seed(r), f'{sc} rep {r:03d}'
        facts[tag] = {}
        for arm in C.ARMS:
            out, sha = C.RESULTS / sc / arm, C.fingerprint(ds, arm)
            t0 = time.perf_counter()
            if arm in C.HAPMIX_ARMS:
                nominal, n_zeroed = C.run_nominal(S, ds, arm, scratch)
                C.write_parquet(nominal, out / f'nominal_rep{r:03d}.parquet', sha, C.UNITS[arm])
                secs.setdefault((arm, 'map_nominal'), []).append(time.perf_counter() - t0)
                below = sorted(nominal.phenotype_id[~nominal.allelic_admitted].unique())
                facts[tag][arm] = dict(zeroed=n_zeroed, below_floor=below)
                t0 = time.perf_counter()
                cis = run_cis(S, ds, arm, seed)
                C.write_parquet(cis, out / f'cis_rep{r:03d}.parquet', sha, C.UNITS[arm])
                secs.setdefault((arm, 'map_cis'), []).append(time.perf_counter() - t0)
                print(f'{tag} {arm:17s} map_nominal {len(nominal):,} rows, allelic admission zeroed {n_zeroed} donor-gene '
                      f'pairs, allelic channel out of the combination in {below}; map_cis seed {seed}, NaN pval_beta '
                      f'{int(cis.pval_beta.isna().sum())}', flush=True)
                continue
            nominal, n_asc, n_trc = run_mixqtl(S, ds, C.MIXQTL_ARMS[arm])
            C.write_parquet(nominal, out / f'nominal_rep{r:03d}.parquet', sha, C.UNITS[arm])
            secs.setdefault((arm, 'mixqtl_scan'), []).append(time.perf_counter() - t0)
            cut = C.MX.META_N_CUTOFF
            facts[tag][arm] = dict(genes_asc_ge_cutoff=int((n_asc >= cut).sum()), genes_trc_ge_cutoff=int((n_trc >= cut).sum()),
                                   asc_median=int(np.median(n_asc)), asc_min=int(n_asc.min()), asc_max=int(n_asc.max()),
                                   genes_no_finite_meta_p=int(nominal.groupby('phenotype_id').pval_nominal
                                                              .apply(lambda p: not np.isfinite(p).any()).sum()),
                                   meta_share=float((nominal.method == 'meta').mean()))
            print(f'{tag} {arm:17s} mixqtl_scan {len(nominal):,} rows; {json.dumps(facts[tag][arm])}', flush=True)
    shutil.rmtree(scratch)
    C.write_json(C.RESULTS / 'run_arms_facts.json', dict(meta_n_cutoff=C.MX.META_N_CUTOFF, runs=facts))
    for (arm, step), v in secs.items():
        print(f'{arm:17s} {step:12s} {len(v)} datasets, seconds per dataset median {np.median(v):.1f} [{min(v):.1f}, {max(v):.1f}]')
    print(f'wrote {C.RESULTS}')


if __name__ == '__main__':
    main()

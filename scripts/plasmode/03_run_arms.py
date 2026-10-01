"""Map every plasmode dataset under seven arms (README: Arms): four hapmixQTL weightings through
map_nominal (nominal p per tested variant) and map_cis (gene-level pval_perm and pval_beta from
NPERM records_signflip permutations, GPU), mixQTL mode at two cutoff settings through mixqtl_scan
on the thinned point estimates (never the draws) and its own permutation scan, and the total-only
tensorQTL scan (tensorqtl.cis.map_nominal and map_cis, GPU). Also eigenMT's effective number of
tests per gene.

hapmixQTL arms (common.arm_variances, after the zero-haplotype admission): gibbs = Gibbs variance
in both channels (the shipped default); split = Gibbs allelic, unit total; unit = 1 everywhere;
plus_one = v + 1 in both. map_nominal and map_cis get the RNA-tied covariates in the dataset's
record order, the genotype PCs in place, the allelic channel through the origin, window WIN,
default mode; A is already swapped in the dataset. Stored per tested variant: COLS + DOF_COLS
(map_nominal's own dtypes; each p's t reference and whether the gene's allelic channel entered the
combination, hapmixqtl.MIN_ALLELIC_DONORS = 15). The combined standard error carries Meier's
correction for estimated channel weights (commit a1b2ef4, hapmixqtl._meier_factor) in map_nominal
and in map_cis's scan, permutations and lead alike. map_cis seed: SeedSequence(SEED, (MAPCIS_KEY,
r)), one integer per dataset index shared by the four arms, the tensorqtl arm and the scenarios;
map_cis(seed=...) calls np.random.seed, the one place this script touches global numpy state.

mixQTL arms: MX.mixqtl_scan per gene on its tested variants with lib_size = eff_lib, covariates in
record order, genotype PCs in place, h1 / h2 = xL / xR; COLS in NATURAL LOG (the parquet metadata
records the unit; 06_score.py divides by ln 2) plus `method` (meta, trc or asc: which estimate the
meta columns hold when a channel has fewer than MX.META_N_CUTOFF samples). Gene-level p from
MX.mixqtl_permutation_scan under mixQTL's published null, not changed for mixQTL mode: the
phenotype bundle (y1, y2, ytotal, library size) and the RNA-tied covariates move by one of
MIXQTL_NPERM record permutations, the haplotypes and genotype PCs stay, the offset is refitted per
permutation, no haplotype labels are swapped; indices from SeedSequence(SEED, (MIXQTL_PERM_KEY, r)),
shared by genes, both arms and the scenarios. Observed statistic: the gene's largest |meta beta /
se| over its tested variants in the mixqtl_scan output (its lead); gate, per gene: the identity
permutation reproduces it within IDENTITY_RTOL relative (tests/test_mixqtl_replication.py).
pval_perm = (1 + #{finite permuted maxima >= observed}) / (1 + #finite), as
scripts/mixqtl_gene_level_typeI.py computes it; the port has no Beta approximation. It is CPU
NumPy, about 8 min per dataset per arm (user request 2026-09-27, which retired the 2026-09-26
timing rule that had left it out): both mixQTL arms of a dataset run in one of at most POOL worker
processes, forked before this process touches the GPU, while this process maps the GPU arms.

tensorqtl arm: tensorqtl.cis.map_nominal and map_cis on the dataset's total phenotype T
(log2(CPM + 1)), unweighted, no allelic channel, with the same 17 covariates as one covariates_df
(the 14 RNA-tied in record order, the 3 genotype PCs in place), window WIN and the same genotype
frame, so the same tested variants per gene (row count checked; map_cis's num_var checked against
the tested variants with varying dosage); map_cis with NPERM permutations of its own null (the
covariate-residualized phenotype permuted) and the Beta approximation. slope and slope_se are
doubled on writing: tensorQTL regresses on ALT dosage g, the hapmixQTL total channel and the truth
on g / 2. Its t is unit weights' total-channel t (the same least-squares fit on N - 2 - 17 = 73
df); the largest absolute difference over the first dataset's pairs is printed and stored.

eigenMT (Davis et al. 2016): tensorqtl.eigenmt.compute_tests at its defaults (variance threshold
0.99, windows of 200 variants) on each gene's tested variants' ALT dosages in position order: the
effective number of independent tests M_eff, written to C.EIGENMT (06_score.py reads it). It runs in
float32 on the device torch picks, and the 99% count depends on the device: the same function on the
CPU gave M_eff differing in 99 / 97 of 100 genes, by at most 14 and a median 0.14% / 0.12% (deep /
low-coverage set, 2026-09-27, one-off comparison against these runs' files); C.EIGENMT is the GPU's.

Tested variants: the loader's idx set within each gene's window is exactly map_nominal's output
(row count checked per call); 507 tested variants have every donor heterozygous, which map_cis
drops as monomorphic (its num_var is the count with varying dosage).

Output: RESULTS/<scenario>/<arm>/nominal_repNNN.parquet and cis_repNNN.parquet (hapmixQTL: CIS_COLS;
mixQTL: MIXQTL_CIS_COLS; tensorqtl: TQ_CIS_COLS), RESULTS/mixqtl_permutation.json (the scan's
settings and seconds), RESULTS/run_arms_facts.json (the per-run counts 08_report.py reads),
C.EIGENMT. Every output is recomputed on every run.
"""
import concurrent.futures as cf
import json
import multiprocessing
import shutil
import time

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

import common as C
from tensorqtl import cis as TQ
from tensorqtl import eigenmt
from tensorqtl.hapmixqtl import map_cis

MAPCIS_KEY, MIXQTL_PERM_KEY = 4, 5   # spawn keys after 02_make_datasets' 1 / 2 / 3
NPERM = 1000                   # map_cis on every dataset (user decision 2026-09-26): ~17 s per dataset per arm on one L4
PERM_SCHEME = 'records_signflip'
MIXQTL_NPERM = 1000            # as map_cis (user decision 2026-09-26)
POOL = 10                      # worker processes for the mixQTL arms, at most 10 (user request 2026-09-27)
WORKER_THREADS = 1             # BLAS threads per worker: at OpenBLAS's default 64 the 10 workers loaded the shared host to 600 (2026-09-27)
IDENTITY_RTOL = 1e-9           # tests/test_mixqtl_replication.py: identity permutation vs observed maximum
CIS_COLS = ['phenotype_id', 'variant_id', 'num_var', 'pval_nominal', 'slope', 'slope_se', 'slope_a', 'slope_a_se',
            'slope_t', 'slope_t_se', 'pval_perm', 'pval_beta', 'beta_shape1', 'beta_shape2', 'true_df']
MIXQTL_CIS_COLS = ['phenotype_id', 'variant_id', 'stat_obs', 'pval_perm', 'n_perm_finite']
TQ_CIS_COLS = ['phenotype_id', 'variant_id', 'num_var', 'pval_perm', 'pval_beta']
S = None                       # the cache inputs; set by main before the mixQTL workers fork, which inherit it


def cis_seed(r):
    return int(np.random.SeedSequence(C.SEED, spawn_key=(MAPCIS_KEY, r)).generate_state(1)[0])


def mixqtl_perm_idx(n, r):
    rng = np.random.default_rng(np.random.SeedSequence(C.SEED, spawn_key=(MIXQTL_PERM_KEY, r)))
    return np.array([rng.permutation(n) for _ in range(MIXQTL_NPERM)])


def run_cis(S, ds, arm, seed):
    """map_cis on the arm's inputs; tau_refit is inert in default mode (map_cis refits only for tau_mode 'estimate')."""
    A, T, Va, Vt, cov, _ = C.phenotypes(S, ds, arm)
    res = C.quiet(map_cis, S['gdf'], S['vdf'][['chrom', 'pos']], A, T, Va, Vt, S['gp'], xL_df=S['xLdf'], xR_df=S['xRdf'],
                  covariates_df=cov, genotype_covariates_df=S['I']['geno_cov_df'], window=C.WIN, nperm=NPERM,
                  seed=seed, perm_scheme=PERM_SCHEME, tau_refit=True, verbose=False, ase_covariates_df=None)
    res = res.reset_index()[CIS_COLS]
    res['variant_id'] = res['variant_id'].astype(str)
    if list(res.phenotype_id) != S['genes']:
        raise SystemExit(f'map_cis returned {len(res)} genes, not the {len(S["genes"])} in order')
    return res


def run_tensorqtl(S, ds, seed, scratch):
    """tensorqtl.cis.map_nominal (COLS[:5], slope and se on g / 2) and map_cis (TQ_CIS_COLS) on T with the 17 covariates."""
    _, T, _, _, cov, _ = C.phenotypes(S, ds, 'unit')
    cov17 = pd.concat([cov, S['I']['geno_cov_df']], axis=1)
    scratch.mkdir(parents=True, exist_ok=True)
    for q in scratch.glob('*'):
        q.unlink()
    C.quiet(TQ.map_nominal, S['gdf'], S['vdf'][['chrom', 'pos']], T, S['gp'], 't', covariates_df=cov17, window=C.WIN,
            output_dir=str(scratch), verbose=False)
    nom = pd.concat([pd.read_parquet(q, columns=C.COLS[:5]) for q in sorted(scratch.glob('t.cis_qtl_pairs.*.parquet'))],
                    ignore_index=True)
    nom['variant_id'] = nom['variant_id'].astype(str)
    if not nom.groupby('phenotype_id').size().reindex(S['genes']).equals(S['n_tested'].reindex(S['genes'])):
        raise SystemExit(f'tensorqtl map_nominal returned {len(nom):,} rows against {int(S["n_tested"].sum()):,} tested pairs')
    nom['slope'], nom['slope_se'] = 2 * nom['slope'], 2 * nom['slope_se']
    res = C.quiet(TQ.map_cis, S['gdf'], S['vdf'][['chrom', 'pos']], T, S['gp'], covariates_df=cov17, nperm=NPERM,
                  window=C.WIN, seed=seed, verbose=False, warn_monomorphic=False)
    res = res.reset_index()[TQ_CIS_COLS]
    res['variant_id'] = res['variant_id'].astype(str)
    if list(res.phenotype_id) != S['genes'] or list(res.num_var) != [len(S['scanned'][g]) for g in S['genes']]:
        raise SystemExit(f'tensorqtl map_cis returned {len(res)} genes, not the {len(S["genes"])} in order with num_var the '
                         f'tested variants of varying dosage')
    return nom, res


def t_difference(nom, unit):
    """Largest |t_tensorqtl - t_unit total| over the pairs finite in both; the pairs finite in only one."""
    m = nom.merge(unit, on=['phenotype_id', 'variant_id'], suffixes=('_tq', ''), validate='one_to_one')
    t_tq = m.slope_tq.to_numpy(float) / m.slope_se_tq.to_numpy(float)
    t_u = m.slope_t.to_numpy(float) / m.slope_t_se.to_numpy(float)
    fin = np.isfinite(t_tq) & np.isfinite(t_u)
    return dict(pairs=len(m), finite_both=int(fin.sum()), finite_one=int((np.isfinite(t_tq) != np.isfinite(t_u)).sum()),
                max_abs_diff=float(np.max(np.abs(t_tq[fin] - t_u[fin]))), max_abs_t=float(np.max(np.abs(t_u[fin]))))


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


def mixqtl_gene_level(S, ds, cutoffs, nominal, perm_idx):
    """pval_perm per gene from mixqtl_permutation_scan against the lead of mixqtl_scan's output, after the identity gate."""
    I, MX = S['I'], C.MX
    y1, y2, yt = MX.inputs_from_point_estimates(ds['pL'], ds['pR'], ds['pT'])
    kw = dict(covariates=I['cov_df'].values[ds['perm']], genotype_covariates=I['geno_cov_df'].values, **cutoffs)
    ident = np.arange(len(S['order']))[None, :]
    stat = np.abs(nominal.slope.to_numpy(float) / nominal.slope_se.to_numpy(float))
    rec, bad, start = [], [], 0
    for k, g in enumerate(S['genes']):
        rows = S['tested_rows'][g]
        s, vid = stat[start:start + len(rows)], nominal.variant_id.to_numpy()[start:start + len(rows)]
        start += len(rows)
        i = int(np.nanargmax(s)) if np.isfinite(s).any() else None
        s_obs = float('nan') if i is None else float(s[i])
        args = (y1[k], y2[k], yt[k], ds['eff_lib'], I['xL'][rows].T.astype(float), I['xR'][rows].T.astype(float))
        s_id = MX.mixqtl_permutation_scan(*args, ident, **kw)[0]
        if not np.isclose(s_id, s_obs, rtol=IDENTITY_RTOL, atol=0, equal_nan=True):
            bad.append(f'{g}: identity {s_id!r}, observed {s_obs!r}')
        sp = MX.mixqtl_permutation_scan(*args, perm_idx, **kw)
        ok = np.isfinite(sp)
        p = (np.sum(sp[ok] >= s_obs) + 1) / (ok.sum() + 1) if i is not None else float('nan')
        rec.append(dict(phenotype_id=g, variant_id=None if i is None else str(vid[i]), stat_obs=s_obs, pval_perm=p,
                        n_perm_finite=int(ok.sum())))
    if bad:
        raise SystemExit(f'GATE FAILED: mixqtl_permutation_scan at the identity permutation differs from the observed '
                         f'maximum |meta stat| in {len(bad)} genes, e.g. {bad[:3]}')
    return pd.DataFrame(rec)[MIXQTL_CIS_COLS]


def mixqtl_dataset(sc, r):
    """Both mixQTL arms on one dataset, in a worker process: nominal and cis files; per arm its run facts and seconds."""
    ds, perm_idx, out = C.load_dataset(C.DATASETS, sc, r), mixqtl_perm_idx(len(S['order']), r), {}
    threadpool_limits(WORKER_THREADS)
    for arm, cutoffs in C.MIXQTL_ARMS.items():
        sha, t0 = C.fingerprint(ds, arm), time.perf_counter()
        nominal, n_asc, n_trc = run_mixqtl(S, ds, cutoffs)
        C.write_parquet(nominal, C.RESULTS / sc / arm / f'nominal_rep{r:03d}.parquet', sha, C.UNITS[arm])
        t1 = time.perf_counter()
        gl = mixqtl_gene_level(S, ds, cutoffs, nominal, perm_idx)
        C.write_parquet(gl, C.RESULTS / sc / arm / f'cis_rep{r:03d}.parquet', sha, C.UNITS[arm])
        cut = C.MX.META_N_CUTOFF
        facts = dict(genes_asc_ge_cutoff=int((n_asc >= cut).sum()), genes_trc_ge_cutoff=int((n_trc >= cut).sum()),
                     asc_median=int(np.median(n_asc)), asc_min=int(n_asc.min()), asc_max=int(n_asc.max()),
                     genes_no_finite_meta_p=int(nominal.groupby('phenotype_id').pval_nominal
                                                .apply(lambda p: not np.isfinite(p).any()).sum()),
                     meta_share=float((nominal.method == 'meta').mean()))
        secs = {'mixqtl_scan': t1 - t0, 'mixqtl_permutation_scan': time.perf_counter() - t1}
        out[arm] = dict(facts=facts, secs=secs)
        print(f'{sc} rep {r:03d} {arm:17s} mixqtl_scan {len(nominal):,} rows; {json.dumps(facts)}; mixqtl_permutation_scan '
              f'{len(gl)} genes, identity gate passed, NaN pval_perm {int(gl.pval_perm.isna().sum())}, genes with fewer than '
              f'{MIXQTL_NPERM} finite permuted maxima {int((gl.n_perm_finite < MIXQTL_NPERM).sum())}; '
              f'{secs["mixqtl_scan"]:.0f} s + {secs["mixqtl_permutation_scan"]:.0f} s', flush=True)
    return out


def eigenmt_tests(S, device):
    """eigenMT's M_eff per gene over its tested variants (module docstring), with the tested count."""
    pos = S['I']['vdf'].pos.to_numpy()
    rows = []
    for g in S['genes']:
        idx = S['tested_rows'][g]
        if (np.diff(pos[idx]) < 0).any():
            raise SystemExit(f'{g}: tested variants are not in position order')
        m = eigenmt.compute_tests(torch.tensor(S['I']['dos'][idx].astype(np.float32), device=device))
        if not 1 <= m <= len(idx):
            raise SystemExit(f'{g}: eigenMT M_eff {m} outside [1, {len(idx)}]')
        rows.append(dict(gene=g, m_eff=int(m), n_tested=len(idx)))
    return pd.DataFrame(rows)


def main():
    global S
    S = C.setup(C.load()[0])
    meta = json.loads((C.DATASETS / 'meta.json').read_text())
    if S['genes'] != meta['genes'] or S['order'] != meta['donors']:
        raise SystemExit(f'{C.DATASETS / "meta.json"}: genes or donors differ from the cache inputs')
    runs = C.runs(meta)
    C.RESULTS.mkdir(parents=True, exist_ok=True)
    pool = cf.ProcessPoolExecutor(min(POOL, len(runs)), mp_context=multiprocessing.get_context('fork'))
    jobs = {pool.submit(mixqtl_dataset, sc, r): (sc, r) for sc, r in runs}   # fork now, before this process touches the GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # tensorqtl/hapmixqtl.py picks the same
    print(f'torch {torch.__version__}; map_cis device {device}' + (f' ({torch.cuda.get_device_name(device)})' if device.type == 'cuda' else '')
          + f'; mixQTL arms in {min(POOL, len(runs))} worker processes over {len(runs)} datasets', flush=True)
    em = eigenmt_tests(S, device)
    C.write_atomic(C.EIGENMT, lambda fh: em.to_csv(fh, sep='\t', index=False), 'w')
    print(f'eigenMT M_eff over {len(em)} genes: min / median / max {em.m_eff.min()} / {int(em.m_eff.median())} / {em.m_eff.max()}; '
          f'M_eff / tested variants {(em.m_eff / em.n_tested).min():.3f} to {(em.m_eff / em.n_tested).max():.3f}; wrote {C.EIGENMT}',
          flush=True)
    scratch, secs, facts, tdiff = C.RESULTS / 'scratch', {}, {}, None
    for sc, r in runs:
        ds, seed, tag = C.load_dataset(C.DATASETS, sc, r), cis_seed(r), f'{sc} rep {r:03d}'
        facts[tag] = {}
        for arm in C.HAPMIX_ARMS:
            out, sha = C.RESULTS / sc / arm, C.fingerprint(ds, arm)
            t0 = time.perf_counter()
            nominal, n_zeroed = C.run_nominal(S, ds, arm, scratch)
            C.write_parquet(nominal, out / f'nominal_rep{r:03d}.parquet', sha, C.UNITS[arm])
            secs.setdefault((arm, 'map_nominal'), []).append(time.perf_counter() - t0)
            below = sorted(nominal.phenotype_id[~nominal.allelic_admitted].unique())
            facts[tag][arm] = dict(zeroed=n_zeroed, below_floor=below)
            if arm == 'unit':
                unit = nominal
            t0 = time.perf_counter()
            cis = run_cis(S, ds, arm, seed)
            C.write_parquet(cis, out / f'cis_rep{r:03d}.parquet', sha, C.UNITS[arm])
            secs.setdefault((arm, 'map_cis'), []).append(time.perf_counter() - t0)
            print(f'{tag} {arm:17s} map_nominal {len(nominal):,} rows, allelic admission zeroed {n_zeroed} donor-gene '
                  f'pairs, allelic channel out of the combination in {below}; map_cis seed {seed}, NaN pval_beta '
                  f'{int(cis.pval_beta.isna().sum())}', flush=True)
        arm, out, sha = C.TENSORQTL, C.RESULTS / sc / C.TENSORQTL, C.fingerprint(ds, C.TENSORQTL)
        t0 = time.perf_counter()
        nominal, cis = run_tensorqtl(S, ds, seed, scratch)
        C.write_parquet(nominal, out / f'nominal_rep{r:03d}.parquet', sha, C.UNITS[arm])
        C.write_parquet(cis, out / f'cis_rep{r:03d}.parquet', sha, C.UNITS[arm])
        secs.setdefault((arm, 'map_nominal + map_cis'), []).append(time.perf_counter() - t0)
        if tdiff is None:
            tdiff = dict(dataset=tag, **t_difference(nominal, unit))
            print(f'{tag} tensorqtl t against unit weights\' total-channel t: largest absolute difference '
                  f'{tdiff["max_abs_diff"]:.2e} over {tdiff["finite_both"]:,} of {tdiff["pairs"]:,} pairs (largest |t| '
                  f'{tdiff["max_abs_t"]:.2f}); finite in one only {tdiff["finite_one"]}', flush=True)
        print(f'{tag} {arm:17s} map_nominal {len(nominal):,} rows; map_cis seed {seed}, NaN pval_beta '
              f'{int(cis.pval_beta.isna().sum())}', flush=True)
    shutil.rmtree(scratch)
    mix = {}
    for job in cf.as_completed(jobs):
        mix[jobs[job]] = job.result()
    pool.shutdown()
    for sc, r in runs:
        for arm, v in mix[sc, r].items():
            facts[f'{sc} rep {r:03d}'][arm] = v['facts']
            for step, s in v['secs'].items():
                secs.setdefault((arm, step), []).append(s)
    C.write_json(C.RESULTS / 'mixqtl_permutation.json', dict(
        included=True, nperm=MIXQTL_NPERM, spawn_key=MIXQTL_PERM_KEY, workers=min(POOL, len(runs)),
        rule='user request 2026-09-27: mixqtl_permutation_scan on every dataset for both cutoff settings (the 2026-09-26 timing rule retired)',
        seconds_per_dataset={arm: float(np.median(secs[arm, 'mixqtl_permutation_scan'])) for arm in C.MIXQTL_ARMS}))
    C.write_json(C.RESULTS / 'run_arms_facts.json', dict(meta_n_cutoff=C.MX.META_N_CUTOFF, runs=facts,
                                                         tensorqtl_vs_unit_total_t=tdiff))
    for (arm, step), v in secs.items():
        print(f'{arm:17s} {step:24s} {len(v)} datasets, seconds per dataset median {np.median(v):.1f} [{min(v):.1f}, {max(v):.1f}]')
    print(f'wrote {C.RESULTS}')


if __name__ == '__main__':
    main()

"""Native-input arms (README: Native-input arms): TReCASE and split weighting on alignment counts from the same STAR
BAMs, on the datasets of 02_make_datasets.py (task 2026-09-28).

Why: TReCASE and RASQUAL are written for integer alignment counts, and the benchmark hands them Salmon point estimates
(fractional totals, rounded haplotype copies). These arms give TReCASE its native input, and give hapmixQTL's split
weighting the same input as a control that separates the model from the quantifier.

Native counts (C.NATIVE_COUNTS, scripts/native_counts.py): per gene and donor, featureCounts fragments (the total; unique,
primary, reverse stranded, fragments on exons of two genes not counted) and phASER fragments on the haplotype of the
analysis VCF's first GT allele (a, the genotype frames' xL) and second (b), with a = b = 0 where a + b exceeds the total.
Donors are joined on the DNA library id (the cache's samples.txt names, which are the genotype frames' columns).

Steps, every output recomputed except TReCASE's finished genes:
(1) Native effective library sizes: featureCounts over every annotated gene (C.NATIVE_COUNTS/featurecounts, column sums
    checked against featureCounts' own Assigned count) through scripts/edger_library_normalization.R with the Salmon cache's
    restriction list (RESTRICT), the rule that made the Salmon effective library sizes; into NATIVE/edger.
(2) Native datasets: for each dataset of DATASETS, its record permutation and label swap applied to the native counts
    (02's move_records: column i takes record perm[i], a and b exchanged where swap[i] = -1; the library size moves with
    the record), a thinned by the dataset's fL, b by fR and the remainder U = total - a - b by (fL + fR) / 2 with 02's
    thin_haplotypes, exact binomial thinning on integers, stream SeedSequence(SEED, (NATIVE_THIN_KEY, r, round(1000 |beta|)));
    then summaries_from_point_estimates with the thinned counts as the one draw, so the across-draw variance is 0 and
    Va is the counting variance (1/(a + 0.5) + 1/(b + 0.5)) / ln2^2 (0 where a + b = 0), T the half-read log CPM
    (hapmixqtl.half_read_log_cpm, as 02) on the native effective library size. Stored with the Salmon dataset's perm, swap, is_null and causal_variant; the truths stay in the
    Salmon dataset (count scale: allelic beta; total, the least-squares slope on g / 2 of log2 of the donor's mean thinning
    factor, a function of the causal genotypes and beta only, so shared by both inputs).
(3) split_native: map_nominal and map_cis as 03 runs the split arm (1/Va allelic, unit total, same covariates, seed
    03.cis_seed(r)), every pair with a + b > 0 admitted (common.arm_variances: no zero-haplotype rule on counts).
(4) trecase_native: 05's runner on Y = the thinned native total, Y1 = a, Y2 = b as they are (asSeq's min.AS.reads applies),
    offset log(native effective library size), the same 17 covariates; JOBS Rscript processes (SIMULATED_EFFECTS_NATIVE_JOBS in
    the environment, default 15), per-gene checkpoints in
    NATIVE/trecase_work (a gene whose status file exists is not rerun).

Output: NATIVE/edger, NATIVE/datasets/<scenario>/repNNN.npz, NATIVE/results/<scenario>/split_native/{nominal,cis}_repNNN.parquet,
NATIVE/results_trecase/<scenario>/trecase_native/nominal_repNNN.parquet and summary.json, NATIVE/facts.json (08 reads it).
"""
import json
import os
import subprocess
import time

import numpy as np
import pandas as pd

import common as C
from tensorqtl.hapmixqtl import MIN_ALLELIC_DONORS, half_read_log_cpm, half_read_total_gibbs_variance, summaries_from_point_estimates

MD, RA, TR = C.module('02_make_datasets'), C.module('03_run_arms'), C.module('05_run_trecase')
PE = C.D / 'cache' / 'gibbs_56b63c3b37ed5df8' / 'point_estimates'   # compare_mixqtl_replication.PE
RESTRICT = PE / 'restrict_calibration.txt'   # protein-coding autosomal genes: the Salmon edgeR run's restriction (build_point_estimate_cache.py)
EDGER_R = C.HERE / 'edger_library_normalization.R'   # byte-identical copy of scripts/edger_library_normalization.R (2026-10-01), the pipeline's own normalization
NATIVE_THIN_KEY = 7            # spawn key used by no other script here (02: 1-3; 03: 4, 5; select: 6; 01: 10-13; 06: 30, 33)
JOBS = int(os.environ.get('SIMULATED_EFFECTS_NATIVE_JOBS', 15))   # asSeq genes at once: Rscript execs into R, one process per job (README), so 15 plus this driver is run_all.sh's cap of 16; the 2026-09-28 runs set 44 under a one-off allowance of 48
THREADS = {'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'}   # one BLAS thread per R process on the shared host


def native_library_sizes(samples):
    """edgeR effective library sizes of the native totals, in cache sample order, and the kept gene count."""
    out = C.NATIVE / 'edger'
    out.mkdir(parents=True, exist_ok=True)
    cols = [pd.read_csv(C.NATIVE_COUNTS / 'featurecounts' / f'{s}.txt', sep='\t', skiprows=1, usecols=[0, 6], index_col=0)
            .iloc[:, 0].rename(s) for s in samples]
    X = pd.concat(cols, axis=1)
    assigned = pd.Series({s: v['Assigned'] for s, v in json.loads((C.NATIVE_COUNTS / 'facts.json').read_text())['fc_per_donor'].items()})
    if not X.index.is_unique or X.isna().any().any() or not X.sum().equals(assigned.loc[samples].astype(X.dtypes.iloc[0])):
        raise SystemExit('featureCounts tables: duplicate genes, missing values, or column sums differing from Assigned')
    counts = C.NATIVE / 'edger' / 'native_totals_all.tsv.gz'
    tmp = counts.with_name('tmp_' + counts.name)
    X.rename_axis('gene').to_csv(tmp, sep='\t')
    os.replace(tmp, counts)
    res = subprocess.run(['Rscript', str(EDGER_R), str(counts), str(RESTRICT), str(out)], capture_output=True, text=True,
                         env={**os.environ, **TR.R_ENV})
    if res.returncode:
        raise SystemExit(f'edger_library_normalization.R failed:\n{res.stderr[-800:]}')
    es = pd.read_csv(out / 'edger_samples.tsv', sep='\t', dtype={'sample': str}).set_index('sample')
    if list(es.index) != samples:
        raise SystemExit(f'{out}/edger_samples.tsv: sample order differs from the cache')
    kept = len((out / 'calibration_genes.txt').read_text().split())
    print(f'native library sizes: {X.shape[0]:,} featureCounts genes x {X.shape[1]} donors, column sums = Assigned; '
          f'{res.stdout.strip()}', flush=True)
    return es.eff_lib_size.to_numpy(float), kept


def native_counts(S, I, samples):
    """a, b, total [genes x donors] in the loader's gene and donor order (joined by gene name and DNA library id)."""
    X = {k: pd.read_parquet(C.NATIVE_COUNTS / f'{k}.parquet') for k in ('hap_a', 'hap_b', 'totals')}
    for k, x in X.items():
        if list(x.columns) != samples or not set(S['genes']) <= set(x.index):
            raise SystemExit(f'{k}.parquet: donors differ from the cache samples or genes of {C.GENES} are missing')
    if [samples[k] for k in I['keep']] != S['order']:
        raise SystemExit('loader donor order is not the cache samples at its keep indices')
    a, b, t = (X[k].loc[S['genes'], S['order']].to_numpy(np.int64) for k in ('hap_a', 'hap_b', 'totals'))
    if (a < 0).any() or (b < 0).any() or (a + b > t).any():
        raise SystemExit('native counts: negative, or a + b above the total')
    return a, b, t


def build_datasets(S, R, a, b, t, lib, meta):
    """Native dataset per Salmon dataset (module docstring, step 2); per-dataset facts."""
    base = dict(pL=a.astype(float), pR=b.astype(float), pT=t.astype(float), YL=a[..., None].astype(float),
                YR=b[..., None].astype(float), YT=t[..., None].astype(float), eff_lib=lib)
    med_lib, facts = float(np.median(lib)), {}
    for sc, r in C.runs(meta):
        ds = C.load_dataset(C.DATASETS, sc, r)
        rng = np.random.default_rng(np.random.SeedSequence(C.SEED, spawn_key=(NATIVE_THIN_KEY, r, int(round(1000 * float(sc[4:]))))))
        M = MD.move_records(base, ds['perm'], ds['swap'])
        a2, b2, t2 = MD.thin_haplotypes(M['pL'], M['pR'], M['pT'], ds['fL'], ds['fR'], rng)
        if not all(np.array_equal(x, np.rint(x)) for x in (a2, b2, t2)) or (a2 + b2 > t2).any():
            raise SystemExit(f'{sc} rep {r}: thinned native counts are not integers or a + b exceeds the total')
        A, _, Va, _, _ = summaries_from_point_estimates(a2, b2, t2, M['eff_lib'], a2[..., None], b2[..., None], t2[..., None])
        T = half_read_log_cpm(t2, M['eff_lib'])   # the half-read total, as 02 builds the Salmon datasets' (user decision 2026-10-01)
        Vt = half_read_total_gibbs_variance(t2, M['eff_lib'], t2[..., None].astype(float))   # its counting variance; split_native's total is unweighted
        out = dict(A=A, T=T, Va=Va, Vt=Vt, pL=a2, pR=b2, pT=t2, eff_lib=M['eff_lib'],
                   **{k: ds[k] for k in ('perm', 'swap', 'is_null', 'causal_variant')})
        (C.NATIVE_DATASETS / sc).mkdir(parents=True, exist_ok=True)
        C.write_atomic(C.NATIVE_DATASETS / sc / f'rep{r:03d}.npz', lambda fh: np.savez(fh, **out))
        adm = (Va > C.EPS).sum(1)
        adm_salmon = C.allelic_kept(ds['pL'], ds['pR'], ds['Va']).sum(1)
        f = dict(informative=int(((a2 + b2) > 0).sum()), one_sided_zero=int((((a2 == 0) ^ (b2 == 0)) & ((a2 + b2) > 0)).sum()),
                 salmon_admitted=int(adm_salmon.sum()), removed_max=float((M['pT'] - t2).sum(0).max() / med_lib),
                 below_floor=[g for g, n in zip(S['genes'], adm) if n < MIN_ALLELIC_DONORS],
                 salmon_below_floor=[g for g, n in zip(S['genes'], adm_salmon) if n < MIN_ALLELIC_DONORS],
                 admitted_median=float(np.median(adm)), salmon_admitted_median=float(np.median(adm_salmon)))
        facts[f'{sc} rep {r:03d}'] = f
        print(f'{sc} rep {r:03d}: native pairs with a + b > 0 {f["informative"]:,} (one haplotype at 0: {f["one_sided_zero"]:,}) '
              f'against {f["salmon_admitted"]:,} admitted Salmon pairs; admitted allelic donors per gene median '
              f'{f["admitted_median"]:g} (Salmon {f["salmon_admitted_median"]:g}); genes below {MIN_ALLELIC_DONORS} donors '
              f'{len(f["below_floor"])} (Salmon {len(f["salmon_below_floor"])}); fragments removed per donor / median native '
              f'library max {f["removed_max"]:.2e}', flush=True)
    return facts


def run_split(S, meta):
    """split_native through map_nominal and map_cis (03's run_cis), each dataset, on the genes whose native total is not
    constant (tensorQTL's InputGeneratorCis drops a constant phenotype, and such a gene has no allelic counts either: it has
    no rows, as in trecase_native); per dataset the genes with the allelic channel out and the genes not tested."""
    scratch, out = C.NATIVE / 'scratch', {}
    for sc, r in C.runs(meta):
        ds, t0 = C.load_dataset(C.NATIVE_DATASETS, sc, r), time.perf_counter()
        sha, d = C.fingerprint(ds, 'split_native'), C.NATIVE_RESULTS['split_native'] / sc / 'split_native'
        keep = np.flatnonzero(np.ptp(ds['pT'], axis=1) > 0)
        genes = [S['genes'][i] for i in keep]
        Sk = dict(S, genes=genes, gp=S['gp'].loc[genes], n_tested=S['n_tested'].loc[genes])
        dk = {k: v[keep] if k in ('A', 'T', 'Va', 'Vt', 'pL', 'pR', 'pT') else v for k, v in ds.items()}
        nominal, _ = C.run_nominal(Sk, dk, 'split_native', scratch)
        C.write_parquet(nominal, d / f'nominal_rep{r:03d}.parquet', sha, C.UNITS['split_native'])
        cis = RA.run_cis(Sk, dk, 'split_native', RA.cis_seed(r))
        C.write_parquet(cis, d / f'cis_rep{r:03d}.parquet', sha, C.UNITS['split_native'])
        tag = f'{sc} rep {r:03d}'
        out[tag] = dict(below_floor=sorted(nominal.phenotype_id[~nominal.allelic_admitted].unique()),
                        not_tested=sorted(set(S['genes']) - set(genes)))
        print(f'{tag} split_native: not tested (native total constant) {out[tag]["not_tested"]}; map_nominal {len(nominal):,} rows, '
              f'allelic channel out of the combination in {len(out[tag]["below_floor"])} genes; map_cis NaN pval_beta '
              f'{int(cis.pval_beta.isna().sum())}; {time.perf_counter() - t0:.0f} s', flush=True)
    for q in scratch.glob('*'):
        q.unlink()
    scratch.rmdir()
    return out


def native_allelic(ds):
    """Y1, Y2 for asSeq on native counts: a and b as they are, every pair admitted (asSeq's min.AS.reads still applies)."""
    return ds['pL'], ds['pR'], (ds['pL'] + ds['pR']) > 0


def main():
    I, R, _ = C.load()
    S = C.setup(I)
    meta = json.loads((C.DATASETS / 'meta.json').read_text())
    if S['genes'] != meta['genes'] or S['order'] != meta['donors']:
        raise SystemExit(f'{C.DATASETS / "meta.json"}: genes or donors differ from the loader inputs')
    samples = (C.D / 'cache' / 'gibbs_56b63c3b37ed5df8' / 'samples.txt').read_text().split()
    lib_all, kept = native_library_sizes(samples)
    lib = lib_all[I['keep']]
    a, b, t = native_counts(S, I, samples)
    ratio = lib / R['eff_lib']
    hap, hap_s = a + b, R['pL'] + R['pR']
    lib_facts = dict(genes_kept=kept, native_over_salmon=[float(ratio.min()), float(np.median(ratio)), float(ratio.max())])
    count_facts = dict(pairs=int(a.size), informative=int((hap > 0).sum()), salmon_informative=int((hap_s > 0).sum()),
                       median_hap=float(np.median(hap[hap > 0])), salmon_median_hap=float(np.median(hap_s[hap_s > 0])),
                       total_over_salmon_median=float(np.median(t[R['pT'] > 0] / R['pT'][R['pT'] > 0])),
                       genes_total_zero=[g for g, z in zip(S['genes'], (t == 0).all(1)) if z])
    print(f'native effective library size / Salmon\'s: min {ratio.min():.3f}, median {np.median(ratio):.3f}, max {ratio.max():.3f}; '
          f'{C.GENE_SET}: {a.shape[0]} genes x {a.shape[1]} donors; pairs with a + b > 0 {count_facts["informative"]:,} of '
          f'{a.size:,} (Salmon pL + pR > 0: {count_facts["salmon_informative"]:,}); median a + b over them '
          f'{count_facts["median_hap"]:g} (Salmon {count_facts["salmon_median_hap"]:.1f}); median native total / Salmon total '
          f'{count_facts["total_over_salmon_median"]:.3f}; genes with total 0 in every donor {count_facts["genes_total_zero"]}',
          flush=True)
    facts = dict(library=lib_facts, counts=count_facts, datasets=build_datasets(S, R, a, b, t, lib, meta))
    facts['split_native'] = run_split(S, meta)
    C.write_json(C.NATIVE / 'facts.json', facts)
    os.environ.update(THREADS)
    TR.run(S, C.NATIVE_DATASETS, C.NATIVE_RESULTS['trecase_native'], C.NATIVE / 'trecase_work', C.runs(meta), JOBS,
           arm='trecase_native', counts=native_allelic)


if __name__ == '__main__':
    main()

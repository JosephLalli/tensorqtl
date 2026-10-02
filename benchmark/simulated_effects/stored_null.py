"""Stored 200-permutation null of the hapmixQTL arms for a benchmark gene set, from the benchmark's own generator
(task 2026-10-02: the low-coverage set had none).

Each draw is the gene set's real records under one record permutation and label swap of common.OLD/permutations.npz,
the stream of the deep set's stored null (scripts/half_read_stored_null.py); both gene sets list the same 92 donors in
the same order, so draw p is the same permutation for both. No effect is injected and nothing is thinned: the dataset is
02's generate with every factor 1, built exactly as 01's check (d) builds it, and each arm (gibbs, split, unit) goes
through common.run_nominal as 03 runs it.

KNOWN ANSWER (--check, on the deep gene set): draw 0 must reproduce the deep set's stored draw 0
(stored_null_half_read_20261001/draws/<arm>_000.parquet) for every arm, by 01's check (d) comparisons and tolerances.

PASS RULE, stated before the run (the deep set's): the combined rate at 0.001 for split and unit lies within its
gene-clustered 95% interval of 0.001; gibbs is reported, not judged.

Output C.D/<name>: draws/<arm>_NNN.parquet (the deep null's 15 columns; a draw whose file exists is not rerun) and
summary.json in the layout 06_score.anchor reads (rates[arm][channel][subset]['after'][alpha], 'after' being this run;
there is no 'before' run), with each gene's rates and informative allelic donor count (per_gene).
Usage: stored_null.py NAME [n=200]   |   stored_null.py --check   (SIMULATED_EFFECTS_GENE_SET picks the gene set)
"""
import json
import sys

import numpy as np
import pandas as pd

import common as C
from tensorqtl.hapmixqtl import MIN_ALLELIC_DONORS

MD, CI = C.module('02_make_datasets'), C.module('01_check_inputs')
ARMS = C.HAPMIX_ARMS
CHANNELS = {'combined': 'pval_nominal', 'allelic': 'pval_a', 'total': 'pval_t'}
STATS = ['pval_nominal', 'slope', 'slope_se', 'pval_a', 'slope_a', 'slope_a_se', 'pval_t', 'slope_t', 'slope_t_se']
BINS = {'n_a < 15': (0, 15), 'n_a 15-39': (15, 40), 'n_a >= 40': (40, 10 ** 9)}   # as the deep null's subsets
N_BOOT = 2000
THIN_KEY, BOOT_KEY = 41, 42    # spawn keys used by no other script here (01: 10-13; 06: 30, 33; gene_level_null: 40)
DEEP = C.D / 'stored_null_half_read_20261001'


def dataset(R, S, p, old):
    perm, swap = old['perms'][p], old['flips'][p].astype(np.int8)
    G, N = R['pL'].shape
    ones = np.ones((G, N))
    g = MD.generate(R, perm, swap, ones, ones, np.random.default_rng(np.random.SeedSequence(C.SEED, spawn_key=(THIN_KEY, p))))
    return dict(A=g['A'], T=g['T'], Va=g['Va'], Vt=g['Vt'], pL=g['pL'], pR=g['pR'], perm=perm,
                causal_variant=np.array([min(S['tested'][x]) for x in S['genes']]))


def draw(S, ds, arm, scratch):
    df, _ = C.run_nominal(S, ds, arm, scratch)
    df = df[['phenotype_id', 'variant_id'] + STATS + C.DOF_COLS].copy()
    df[STATS] = df[STATS].astype(np.float32)
    return df


def check(I, R, S, old, scratch):
    if C.GENE_SET != 'corrected_null_store_20260925':
        raise SystemExit('--check runs on the deep gene set, whose stored null it reproduces')
    ds = dataset(R, S, 0, old)
    for arm in ARMS:
        m = draw(S, ds, arm, scratch).merge(pd.read_parquet(DEEP / 'draws' / f'{arm}_000.parquet'), on=['phenotype_id', 'variant_id'],
                                             suffixes=('', '_stored'))
        every = np.ones(len(m), bool)
        pin = [CI.pinned(m, every, s, se) for s, se in (('slope_a', 'slope_a_se'), ('slope_t', 'slope_t_se'), ('slope', 'slope_se'))]
        pv = [CI.p_agree(m[c].to_numpy(float), m[f'{c}_stored'].to_numpy(float), CI.REPRO_P_RTOL) for c in ('pval_a', 'pval_t', 'pval_nominal')]
        same = all(np.array_equal(m[c].to_numpy(float), m[f'{c}_stored'].to_numpy(float), equal_nan=True) for c in ('dof_a', 'dof_t', 'allelic_admitted'))
        ok = len(m) == int(S['n_tested'].sum()) and all(x['passed'] for x in pin + pv) and same
        print(f'check {arm}: {len(m):,} tests; slopes within {max(x["max_slope_diff_se"] for x in pin):.1e} se, p within '
              f'{max(x["max_rel"] for x in pv):.1e} relative, dof and admission equal {same}  {"PASS" if ok else "FAIL"}', flush=True)
        if not ok:
            raise SystemExit(f'--check: draw 0 does not reproduce {DEEP.name}/draws/{arm}_000.parquet')


def summarize(out, S, n, n_a):
    genes = list(S['genes'])
    gi = np.arange(len(genes))
    subsets = {'all': gi, **{b: gi[(n_a >= lo) & (n_a < hi)] for b, (lo, hi) in BINS.items()}}
    rng = np.random.default_rng(np.random.SeedSequence(C.SEED, spawn_key=(BOOT_KEY,)))
    boot = {s: ix[rng.integers(0, len(ix), size=(N_BOOT, len(ix)))] for s, ix in subsets.items() if len(ix)}
    res = dict(pass_rule='the combined rate at 0.001 for split and unit lies within its gene-clustered 95% interval of 0.001; '
                         'gibbs reported, not judged',
               permutations=str(C.OLD / 'permutations.npz'), covariates=str(C.COV), gene_set=C.GENE_SET, n_draw=n,
               n_genes=len(genes), floor=MIN_ALLELIC_DONORS, n_boot=N_BOOT,
               genes_per_subset={s: int(len(ix)) for s, ix in subsets.items()}, rates={}, per_gene={}, verdict={},
               below_floor_genes={g: dict(n_a=int(n_a[k])) for k, g in enumerate(genes) if n_a[k] < MIN_ALLELIC_DONORS})
    for arm in ARMS:
        files = [out / 'draws' / f'{arm}_{p:03d}.parquet' for p in range(n)]
        res['rates'][arm], res['per_gene'][arm] = {}, {}
        for ch, col in CHANNELS.items():
            K, nt = C.rates_by_gene(files, genes, col)
            res['rates'][arm][ch] = {}
            for s, ix in subsets.items():
                if not len(ix):
                    continue
                row = {'n_tests': int(nt[ix].sum())}
                for al in C.ALPHAS:
                    B = boot[s]
                    nb = nt[B].sum(1)
                    b = K[al][B].sum(1) / np.where(nb > 0, nb, np.nan)
                    row[str(al)] = dict(rate=float(K[al][ix].sum() / nt[ix].sum()) if nt[ix].sum() else None,
                                        lo=float(np.nanquantile(b, .025)), hi=float(np.nanquantile(b, .975)),
                                        resamples_without_tests=int((nb == 0).sum()))
                res['rates'][arm][ch][s] = {'after': row}
            res['per_gene'][arm][ch] = {g: dict(n_a=int(n_a[k]), n_tests=int(nt[k]),
                                                **{str(al): (float(K[al][k] / nt[k]) if nt[k] else None) for al in C.ALPHAS})
                                        for k, g in enumerate(genes)}
        if arm in ('split', 'unit'):
            r = res['rates'][arm]['combined']['all']['after']['0.001']
            res['verdict'][arm] = dict(rate=r['rate'], lo=r['lo'], hi=r['hi'], contains_0_001=r['lo'] <= 0.001 <= r['hi'])
    res['passed'] = all(v['contains_0_001'] for v in res['verdict'].values())
    C.write_json(out / 'summary.json', res)
    for arm in ARMS:
        for ch in CHANNELS:
            x = res['rates'][arm][ch]['all']['after']
            print(f'{arm:6s} {ch:9s} ' + '  '.join(f'{al}: {x[al]["rate"]:.5f} [{x[al]["lo"]:.5f}, {x[al]["hi"]:.5f}]'
                                                   for al in map(str, C.ALPHAS)), flush=True)
    for arm, v in res['verdict'].items():
        print(f'{"PASS" if v["contains_0_001"] else "FAIL"} {arm}: combined at 0.001 {v["rate"]:.5f} [{v["lo"]:.5f}, {v["hi"]:.5f}]', flush=True)


def main():
    I, R, _ = C.load()
    S = C.setup(I)
    old = np.load(C.OLD / 'permutations.npz')
    if '--check' in sys.argv:
        check(I, R, S, old, C.D / 'scratch_stored_null_check')
        return
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    out, n = C.D / args[0], int(args[1]) if len(args) > 1 else 200
    (out / 'draws').mkdir(parents=True, exist_ok=True)
    scratch = out / 'scratch'
    n_a = (C.arm_variances(dataset(R, S, 0, old), 'split')[0] > C.EPS).sum(1)   # informative allelic donors; a permutation moves them, not their count
    for p in range(n):
        todo = [a for a in ARMS if not (out / 'draws' / f'{a}_{p:03d}.parquet').exists()]
        if todo:
            ds = dataset(R, S, p, old)
            for arm in todo:
                C.write_atomic(out / 'draws' / f'{arm}_{p:03d}.parquet', lambda fh, df=draw(S, ds, arm, scratch): df.to_parquet(fh, compression='zstd', index=False))
        if p % 20 == 19:
            print(f'  draw {p + 1}/{n}', flush=True)
    for q in scratch.glob('*'):
        q.unlink()
    summarize(out, S, n, n_a)
    print(f'wrote {out}', flush=True)


if __name__ == '__main__':
    main()

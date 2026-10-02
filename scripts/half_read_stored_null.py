"""The stored 200-permutation null of the simulated-effects benchmark's three hapmixQTL weightings on the half-read
pipeline, on the permutations and genes of the stored runs and of their re-run under 8a06803.

QUESTION (2026-10-01). The benchmark now runs every hapmixQTL arm on the half-read total (user decision 2026-10-01:
half-read split for everything): A and Va as before, T the half-read log2 CPM, the half-read expression PCs
(compare_mixqtl_replication.COV, a2f4314) and Meier's correction of the combined standard error (a1b2ef4). Its anchor
and null-gene rates need a stored null on that same pipeline; the earlier stored runs were made on log2(CPM + 1).

RUN: three configurations on the same 100 genes and the same permutation stream
(protein_coding_null_store_20260925/permutations.npz), 200 permutations, on this checkout's hapmixqtl:
  split  the shipped default (prepare_default_inputs): allelic 1/Va, total unit working variance
  unit   the same values, unit weights in both channels (allelic channel over the same admitted pairs)
  gibbs  allelic 1/Va, total 1/Vt with Vt = hapmixqtl.half_read_total_gibbs_variance
A, T and Va come from prepare_default_inputs, whose allelic admission (Va > 1e-12, pL + pR > 0, not exactly one
haplotype below 0.5 reads) is the benchmark's.

PASS RULE, stated before the run: the combined rate at 0.001 for split and unit lies within its gene-clustered 95%
interval of 0.001, RPL41 included; gibbs is reported, not judged.

GATES on draw 0, per config, before any draw is stored:
  1. map_nominal's channel slopes equal null_permutation_instrument.fit_channels on the same permuted inputs (1e-3 se);
  2. ALLELIC PAIRING: slope_a, slope_a_se and pval_a equal the 8a06803 re-run's draw 0 of the same config: A, Va,
     the admission and the allelic reference are unchanged, and no covariates enter the allelic channel;
  3. FLOOR and below-floor identity: as allelic_df_null_check.py.

INTERVALS: as allelic_df_null_check.py. In summary.json 'before' is the 8a06803 re-run (log2(CPM + 1) total, earlier
expression PCs, no Meier's correction) and 'after' this run: after - before is those three changes together.

Output OUT: draws/<config>_NNN.parquet, gates.json, covariates.txt, summary.json (allelic_df_null_check.summarize's
layout). Usage: half_read_stored_null.py [n_draw=200] [--summarize-only]
"""
import contextlib
import io
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import allelic_df_null_check as ADF   # imports compare_mixqtl_replication first, which puts this checkout first on sys.path
import compare_mixqtl_replication as CM
import corrected_null_store as CNS
import tensorqtl.hapmixqtl as HM
from null_permutation_instrument import fit_channels
from tensorqtl.hapmixqtl import (MIN_ALLELIC_DONORS, half_read_total_gibbs_variance, map_nominal,
                                 prepare_default_inputs)

OUT = CNS.D / 'stored_null_half_read_20261001'
BEFORE = ADF.OUT                      # the 8a06803 re-run: same permutations, genes and references, the earlier covariates
CONFIGS, JUDGED = ('gibbs', 'split', 'unit'), ('split', 'unit')
COLS, NEW_COLS, FLOOR = ADF.COLS, ADF.NEW_COLS, ADF.FLOOR
ALLELIC = ('slope_a', 'slope_a_se', 'pval_a')


def before_files(config):
    return sorted((BEFORE / 'draws').glob(f'{config}_*.parquet'))


def run_draws(n_draw):
    if not (Path(HM.__file__).resolve().is_relative_to(Path(CM.REPO).resolve()) and MIN_ALLELIC_DONORS == FLOOR):
        raise SystemExit(f'hapmixqtl from {HM.__file__}, floor {MIN_ALLELIC_DONORS}: not this checkout')
    print(f'hapmixqtl: {HM.__file__}; covariates {CM.COV}', flush=True)
    OUT.mkdir(exist_ok=True)
    (OUT / 'covariates.txt').write_text(f'{CM.COV}\n')
    with contextlib.redirect_stdout(io.StringIO()):
        I = CM.load_point_estimate_inputs(gene_list=str(CNS.OUT / 'genes.txt'), regions=str(CNS.OUT / 'regions.bed'))
    order, keep = I['order'], I['keep']
    N = len(order)
    A, T, Va, _ = prepare_default_inputs(I['pL'], I['pR'], I['pT'], I['eff_lib'], I['YL'], I['YR'])
    Vt = half_read_total_gibbs_variance(I['pT'], I['eff_lib'], I['YT'])
    A, T, Va, Vt = A[:, keep], T[:, keep], Va[:, keep], Vt[:, keep]
    one = np.ones_like(T)
    V = dict(gibbs=(Va, Vt), split=(Va, one), unit=(np.where(Va > ADF.EPS, 1.0, 0.0), one))
    genes = list(I['genes'])
    design = pd.read_csv(CNS.OUT / 'gene_design.tsv', sep='\t').set_index('gene')
    n_a = (Va > ADF.EPS).sum(1)
    if genes != (CNS.OUT / 'genes.txt').read_text().split() or \
            not np.array_equal(n_a, design.loc[genes, 'n_allelic_drop'].values):
        raise SystemExit('genes or allelic donor counts differ from corrected_null_store gene_design.tsv')
    gp = I['gp'].loc[genes][['chr', 'pos']]
    C, G = I['cov_df'].values, I['geno_cov_df']
    vdf = I['vdf']
    gdf = pd.DataFrame(I['dos'], index=vdf.index, columns=order)
    xLdf = pd.DataFrame(I['xL'], index=vdf.index, columns=order)
    xRdf = pd.DataFrame(I['xR'], index=vdf.index, columns=order)
    tested_idx = {g: I['idx'][CM.gene_variant_index(I, g)] for g in genes}
    tested = pd.DataFrame([(g, str(v)) for g in genes for v in vdf.index[tested_idx[g]]], columns=['phenotype_id', 'variant_id'])
    print(f'{len(genes)} genes, {N} donors, {len(tested):,} tested gene-variant pairs, covariates '
          f'{C.shape[1]} RNA-tied + {G.shape[1]} genotype-tied', flush=True)
    rng = np.random.RandomState(ADF.SEED)
    perms = np.stack([rng.permutation(N) for _ in range(ADF.N_STREAM)])
    flips = rng.randint(0, 2, size=(ADF.N_STREAM, N)) * 2 - 1
    ref = np.load(CNS.OLD / 'permutations.npz')
    if not (np.array_equal(ref['perms'], perms) and np.array_equal(ref['flips'], flips)):
        raise SystemExit('permutation stream differs from the stored one')
    ddir = OUT / 'draws'; ddir.mkdir(exist_ok=True)
    scratch = OUT / 'scratch'; scratch.mkdir(exist_ok=True)

    def draw(p, config):
        prm, f = perms[p], flips[p].astype(float)
        mk = lambda M, sgn=None: pd.DataFrame(M[:, prm] * (sgn[None, :] if sgn is not None else 1.0), index=genes, columns=order)   # noqa: E731
        cov = pd.DataFrame(C[prm], index=order, columns=I['cov_df'].columns)
        for q in scratch.glob('*'):
            q.unlink()
        va, vt = V[config]
        with contextlib.redirect_stdout(io.StringIO()):
            map_nominal(gdf, vdf[['chrom', 'pos']], mk(A, f), mk(T), mk(va), mk(vt), gp, xL_df=xLdf, xR_df=xRdf, prefix='n',
                        covariates_df=cov, genotype_covariates_df=G, window=CM.WIN, output_dir=str(scratch), verbose=False,
                        ase_covariates_df=None)
        df = pd.concat([pd.read_parquet(q, columns=COLS + NEW_COLS) for q in sorted(scratch.glob('n*.parquet'))], ignore_index=True)
        df['variant_id'] = df['variant_id'].astype(str)
        df = df.merge(tested, on=['phenotype_id', 'variant_id'], how='inner')
        for c in COLS[2:]:
            df[c] = df[c].astype(np.float32)
        return df

    first = {c: draw(0, c) for c in CONFIGS}
    prm0, f0 = perms[0], flips[0].astype(float)
    Cg = np.column_stack([C[prm0], G.values])
    gates = {}
    for c in CONFIGS:
        va, vt = V[c]
        worst = 0.0
        for k, g in enumerate(genes):
            cand = tested_idx[g]
            j = cand[int(np.argmax(((I['xL'][cand] - I['xR'][cand]) != 0).sum(1)))]
            s = (I['xL'][j] - I['xR'][j]).astype(float)
            gh = I['dos'][j].astype(float) / 2.0
            row = first[c][(first[c].phenotype_id == g) & (first[c].variant_id == str(vdf.index[j]))]
            fc = fit_channels(A[k][prm0] * f0, s, va[k][prm0], T[k][prm0], gh, vt[k][prm0], Cg)
            if row.empty or fc is None:
                continue
            r = row.iloc[0]
            worst = max(worst, abs(float(r.slope_a) - fc['ba']) / float(r.slope_a_se),
                        abs(float(r.slope_t) - fc['bt']) / float(r.slope_t_se))
        new = first[c].sort_values(['phenotype_id', 'variant_id'], ignore_index=True)
        old = pd.read_parquet(before_files(c)[0]).sort_values(['phenotype_id', 'variant_id'], ignore_index=True)
        if not new[['phenotype_id', 'variant_id']].equals(old[['phenotype_id', 'variant_id']]):
            raise SystemExit(f'{c}: tested pairs differ from the 8a06803 re-run\'s draw 0')
        v = {k: new[k].values.astype(float) for k in COLS[2:] + ['dof_a']}
        o = {k: old[k].values.astype(float) for k in ALLELIC}
        adm = new.allelic_admitted.values
        g_n_a = new.phenotype_id.map(dict(zip(genes, n_a))).values
        on = g_n_a >= 2
        gates[c] = dict(
            fit_channels_max_dev_over_se=worst,
            allelic_vs_rerun=max(ADF.max_dev(v['slope_a'], o['slope_a'], o['slope_a_se']),
                                 ADF.max_dev(v['slope_a_se'], o['slope_a_se'], o['slope_a_se']),
                                 ADF.max_dev(v['pval_a'], o['pval_a'], np.maximum(o['pval_a'], 1e-30))),
            combined_is_total_below_floor=max(
                ADF.max_dev(v['slope'][~adm], v['slope_t'][~adm], v['slope_t_se'][~adm]),
                ADF.max_dev(v['slope_se'][~adm], v['slope_t_se'][~adm], v['slope_t_se'][~adm]),
                ADF.max_dev(v['pval_nominal'][~adm], v['pval_t'][~adm], np.ones((~adm).sum()))),
            floor_mismatches=int((adm != (g_n_a >= FLOOR)).sum()),
            dof_a_mismatches=int((on & (v['dof_a'] != g_n_a - 1)).sum() + np.isfinite(v['dof_a'][~on]).sum()))
        print(f'gate {c}: ' + '  '.join(f'{k} {x:.2e}' if isinstance(x, float) else f'{k} {x}' for k, x in gates[c].items()),
              flush=True)
        g = gates[c]
        if not (g['fit_channels_max_dev_over_se'] < ADF.GATE_TOL and g['allelic_vs_rerun'] < ADF.PAIR_TOL
                and g['combined_is_total_below_floor'] < ADF.PAIR_TOL and g['floor_mismatches'] == 0 and g['dof_a_mismatches'] == 0):
            raise SystemExit(f'GATE FAILED: {c}')
    tmp = OUT / 'gates.json.tmp'
    tmp.write_text(json.dumps(gates, indent=1)); tmp.rename(OUT / 'gates.json')

    t0 = time.time()
    for p in range(n_draw):
        for c in CONFIGS:
            fo = ddir / f'{c}_{p:03d}.parquet'
            if fo.exists():
                continue
            df = first[c] if p == 0 else draw(p, c)
            df.to_parquet(fo.with_suffix('.tmp'), compression='zstd', index=False)
            fo.with_suffix('.tmp').rename(fo)
        if p % 10 == 9 or p == n_draw - 1:
            print(f'  draw {p + 1}/{n_draw}, {len(CONFIGS)} configs  ({(time.time() - t0) / 60:.1f} min)', flush=True)
    shutil.rmtree(scratch)


def summarize():
    genes = (CNS.OUT / 'genes.txt').read_text().split()
    n_a = pd.read_csv(CNS.OUT / 'gene_design.tsv', sep='\t').set_index('gene').loc[genes, 'n_allelic_drop'].values
    gi = np.arange(len(genes))
    subsets = {'all': gi, f'without {ADF.DROPPED_GENE}': gi[gi != genes.index(ADF.DROPPED_GENE)]}
    subsets.update({b: gi[(n_a >= lo) & (n_a < hi)] for b, (lo, hi) in ADF.BINS.items()})
    brng = np.random.default_rng(np.random.SeedSequence(ADF.SEED).spawn(8)[7])
    boot = {s: ix[brng.integers(0, len(ix), size=(ADF.N_BOOT, len(ix)))] for s, ix in subsets.items()}
    after = {c: sorted((OUT / 'draws').glob(f'{c}_*.parquet')) for c in CONFIGS}
    n_draw = min(len(v) for v in after.values())
    before = {c: before_files(c)[:n_draw] for c in CONFIGS}
    below = [g for g, k in zip(genes, n_a) if k < FLOOR]

    def boot_rates(K, n, s):
        B = boot[s]
        nb = n[B].sum(1)
        return K[B].sum(1) / np.where(nb > 0, nb, np.nan), int((nb == 0).sum())

    res = dict(pass_rule='the combined rate at 0.001 for split and unit lies within its gene-clustered 95% interval '
                         'of 0.001, RPL41 included; gibbs reported, not judged',
               before_run=str(BEFORE), covariates=(OUT / 'covariates.txt').read_text().strip(),
               n_draw=n_draw, n_genes=len(genes), floor=FLOOR, n_boot=ADF.N_BOOT,
               genes_per_subset={s: len(ix) for s, ix in subsets.items()},
               gates=json.loads((OUT / 'gates.json').read_text()), rates={}, below_floor_genes={}, verdict={})
    for c in CONFIGS:
        res['rates'][c] = {}
        for ch, col in ADF.CHANNELS.items():
            Ka, na = CNS.rates_by_gene(after[c], genes, col)
            Kb, nb = CNS.rates_by_gene(before[c], genes, col)
            res['rates'][c][ch] = {}
            for s, ix in subsets.items():
                row = {}
                for tag, K, n in (('before', Kb, nb), ('after', Ka, na)):
                    row[tag] = {'n_tests': int(n[ix].sum())}
                    for al in ADF.ALPHAS:
                        b, empty = boot_rates(K[al], n, s)
                        row[tag][str(al)] = dict(rate=float(K[al][ix].sum() / n[ix].sum()), lo=float(np.nanquantile(b, .025)),
                                                 hi=float(np.nanquantile(b, .975)), resamples_without_tests=empty)
                row['after_minus_before'] = {}
                for al in ADF.ALPHAS:
                    dd = boot_rates(Ka[al], na, s)[0] - boot_rates(Kb[al], nb, s)[0]
                    row['after_minus_before'][str(al)] = dict(diff=row['after'][str(al)]['rate'] - row['before'][str(al)]['rate'],
                                                              lo=float(np.nanquantile(dd, .025)), hi=float(np.nanquantile(dd, .975)))
                res['rates'][c][ch][s] = row
            for g in below:
                k = genes.index(g)
                res['below_floor_genes'].setdefault(g, {'n_a': int(n_a[k])}).setdefault(c, {})[ch] = {
                    tag: {'n_tests': int(n[k]), **{str(al): float(K[al][k] / n[k]) if n[k] else None for al in ADF.ALPHAS}}
                    for tag, K, n in (('before', Kb, nb), ('after', Ka, na))}
        if c in JUDGED:
            r = res['rates'][c]['combined']['all']['after']['0.001']
            res['verdict'][c] = dict(lo=r['lo'], hi=r['hi'], rate=r['rate'], contains_0_001=r['lo'] <= 0.001 <= r['hi'])
    res['passed'] = all(v['contains_0_001'] for v in res['verdict'].values())
    tmp = OUT / 'summary.json.tmp'
    tmp.write_text(json.dumps(res, indent=1)); tmp.rename(OUT / 'summary.json')

    f = lambda x: f'{x["rate"]:.5f} [{x["lo"]:.5f}, {x["hi"]:.5f}]'   # noqa: E731
    fd = lambda x: f'{x["diff"]:+.5f} [{x["lo"]:+.5f}, {x["hi"]:+.5f}]'   # noqa: E731
    print(f'\n{n_draw} permutations, {len(genes)} genes; before = the 8a06803 re-run, after = this run (gene-clustered 95% intervals)')
    for c in CONFIGS:
        for ch in ADF.CHANNELS:
            R = res['rates'][c][ch]['all']
            for al in map(str, ADF.ALPHAS):
                print(f'{c:8s} {ch:8s} {al:>5s}  before {f(R["before"][al])}  after {f(R["after"][al])}  '
                      f'after - before {fd(R["after_minus_before"][al])}')
    for c, v in res['verdict'].items():
        print(f'{"PASS" if v["contains_0_001"] else "FAIL"} {c}: combined at 0.001, RPL41 included, {f(v)} '
              f'{"contains" if v["contains_0_001"] else "excludes"} 0.001')


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    n_draw = int(args[0]) if args else 200
    if '--summarize-only' not in sys.argv:
        run_draws(n_draw)
    summarize()
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()

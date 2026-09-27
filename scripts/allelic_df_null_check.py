"""Paired stored-null check of the per-channel t references and the allelic
admission floor (commit 8a06803).

QUESTION (2026-09-27). Before 8a06803 the combined, allelic and total nominal
p were all referred to t with N - 2 - max(n_cov, n_cov_a) = 73 df, and a gene
with 2 informative allelic donors (RPL41, 1 residual df) carried most of the
stored nulls' combined tail at 0.001. The fix refers pval_a to n_a - 1 - n_cov_a
df, pval_t to the total channel's own residual df, the combined statistic to
the Welch-Satterthwaite df (w_a + w_t)^2 / (w_a^2/nu_a + w_t^2/nu_t), and
admits the allelic channel to the combination only with >= 15 informative
allelic donors. Is the combined rate at 0.001 now nominal with RPL41 included?

RERUN: the stored null runs' exact configuration under the new code, on the
same permutations (protein_coding_null_store_20260925/permutations.npz), the
100 genes of corrected_null_store_20260925, 200 permutations:
  gibbs     corrected_null_store's `drop` arm: Gibbs variance in both
            channels, zero-haplotype pairs out of the allelic channel
  split     hybrid_weights_null's `hybrid`: allelic Gibbs, total unit weights
  unit      hybrid_weights_null's `unit`
  plus_one  hybrid_weights_null's `plus_one`: 1 / (v + 1)

PASS RULE, stated before the run: after the fix, the combined rate at 0.001
for split, unit and plus_one lies within its gene-clustered 95% interval of
0.001, RPL41 INCLUDED. gibbs is reported, not judged: its total channel is
known to be anticonservative.

GATES on draw 0, per config, before any draw is stored:
  1. map_nominal's channel slopes equal null_permutation_instrument.fit_channels
     on the same permuted inputs (1e-3 se), as in the stored runs;
  2. PAIRING: the per-channel slopes and SEs equal the stored draw 0 of the same
     config (the fix changes no fit); the combined slope and SE equal the stored
     ones for admitted genes and the total channel's below the floor; pval_t
     equals the stored pval_t wherever dof_t is the old 73;
  3. FLOOR: allelic_admitted equals n_a >= 15 and dof_a equals n_a - 1 (through
     the origin) wherever the allelic channel is on, NaN where it is off.

INTERVALS: gene-clustered: resample the genes with replacement 2,000 times and
recompute the pooled rate (rejections over tests); 2.5 and 97.5 percentiles.
after - before differences use the same resampled genes. They resample genes
only, while all genes share the same 200 permutations; a two-way resampling
(genes x permutations) in the 2026-09-27 review widened them (split after at
0.001 [0.00103, 0.00143], unit [0.00116, 0.00132], plus_one [0.00101,
0.00116]; the plus_one exclusion of 0.001 is marginal) and is not computed here.

RECORD NOTES (2026-09-27 run, allelic_df_fix_20260927):
  - The rule FAILED for split, unit and plus_one. RPL41 is not the cause
    (its split combined rate at 0.001 fell from 0.2105 to 0.0016); the rest is
    the Welch-Satterthwaite reference being more liberal than 73 df in
    admitted genes (paired, n_a >= 40) and the unit-weighted total channel's
    own 0.00115, which the fix does not touch.
  - The allelic before/after are NOT over identical tests: OST4 (1 allelic
    donor) has 611,400 allelic tests at p = 1 before and NaN after.
  - Draws 000 and 001 of all four configs were written by an earlier run of
    this script, before its last edit, and kept by the skip-existing resume
    in run_draws (which checks no code version); the final run's gates
    checked an in-memory draw 0. Those files were verified afterwards against
    the stored runs (channel slopes, SEs and pval_t identical; pval_a and
    pval_nominal equal 2 t.sf(|t|, dof) within 6e-8; the dof, admission and
    Welch-Satterthwaite rules hold). Before a future final run, delete draws
    left by earlier runs.

Usage: allelic_df_null_check.py [n_draw=200] [--summarize-only]
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

import compare_mixqtl_replication as CM
import corrected_null_store as CNS
import hybrid_weights_null as HW
import tensorqtl.hapmixqtl as HM
from null_permutation_instrument import fit_channels
from tensorqtl.hapmixqtl import MIN_ALLELIC_DONORS, map_nominal, summaries_from_point_estimates

OUT = CNS.D / 'allelic_df_fix_20260927'
SEED, N_STREAM, N_BOOT, EPS = 42, 1000, 2000, 1e-12   # as corrected_null_store
ALPHAS = CNS.ALPHAS
CHANNELS, COLS = CNS.CHANNELS, CNS.COLS
NEW_COLS = ['dof_nominal', 'dof_a', 'dof_t', 'allelic_admitted']   # added by 8a06803
CONFIGS = ('gibbs', 'split', 'unit', 'plus_one')
JUDGED = ('split', 'unit', 'plus_one')
FLOOR = 15            # MIN_ALLELIC_DONORS, mixQTL's META_N_CUTOFF (8a06803)
OLD_DOF = 73          # N - 2 - max(n_cov, n_cov_a): 92 donors, 14 RNA-tied + 3 genotype-tied covariates
GATE_TOL = 1e-3       # corrected_null_store's reference-fit gate, in se
PAIR_TOL = 1e-4       # stored draws are float32 (~6e-8 relative); room for GPU reduction order
BINS = {'n_a < 15': (0, FLOOR), 'n_a 15-39': (FLOOR, 40), 'n_a >= 40': (40, 10 ** 6)}
DROPPED_GENE = 'RPL41'
# config -> (stored run directory, stored arm name, stored summary file)
STORED = {'gibbs': (CNS.OUT, 'drop', 'summary.json'), 'split': (HW.OUT, 'hybrid', 'summary.json'),
          'unit': (HW.OUT, 'unit', 'summary_unit.json'),
          'plus_one': (HW.OUT, 'plus_one', 'summary_plus_one.json')}


def stored_files(config):
    d, arm, _ = STORED[config]
    return sorted((d / 'draws').glob(f'{arm}_*.parquet'))


def variances(config, Va, Vt):
    """(allelic, total) working variances; gibbs is the stored `drop` arm."""
    return (Va, Vt) if config == 'gibbs' else HW.config_variances(STORED[config][1], Va, Vt)


def max_dev(new, old, scale):
    """Largest |new - old| / scale over finite entries; inf if the finite patterns differ."""
    fn, fo = np.isfinite(new), np.isfinite(old)
    if not np.array_equal(fn, fo):
        return float('inf')
    ok = fn & np.isfinite(scale) & (scale > 0)
    return float(np.max(np.abs(new[ok] - old[ok]) / scale[ok]))


def run_draws(n_draw):
    if not (Path(HM.__file__).resolve().is_relative_to(Path(CM.REPO).resolve()) and MIN_ALLELIC_DONORS == FLOOR):
        raise SystemExit(f'hapmixqtl from {HM.__file__}, floor {MIN_ALLELIC_DONORS}: not this checkout')
    print(f'hapmixqtl: {HM.__file__}', flush=True)
    OUT.mkdir(exist_ok=True)
    with contextlib.redirect_stdout(io.StringIO()):
        I = CM.load_point_estimate_inputs(gene_list=str(CNS.OUT / 'genes.txt'),
                                          regions=str(CNS.OUT / 'regions.bed'))
    order, keep = I['order'], I['keep']
    N = len(order)
    A, T, Va, Vt, _ = summaries_from_point_estimates(I['pL'], I['pR'], I['pT'], I['eff_lib'],
                                                     I['YL'], I['YR'], I['YT'])
    A, T, Va, Vt = A[:, keep], T[:, keep], Va[:, keep], Vt[:, keep]
    pL, pR = I['pL'][:, keep], I['pR'][:, keep]
    Va = np.where((pL < 0.5) ^ (pR < 0.5), 0.0, Va)
    V = {c: variances(c, Va, Vt) for c in CONFIGS}
    genes = list(I['genes'])
    design = pd.read_csv(CNS.OUT / 'gene_design.tsv', sep='\t').set_index('gene')
    n_a = (Va > EPS).sum(1)
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
    tested = pd.DataFrame([(g, str(v)) for g in genes for v in vdf.index[tested_idx[g]]],
                          columns=['phenotype_id', 'variant_id'])
    print(f'{len(genes)} genes, {N} donors, {len(tested):,} tested gene-variant pairs, covariates '
          f'{C.shape[1]} RNA-tied + {G.shape[1]} genotype-tied; below the floor of {FLOOR}: '
          + ', '.join(f'{g} ({k})' for g, k in zip(genes, n_a) if k < FLOOR), flush=True)
    rng = np.random.RandomState(SEED)
    perms = np.stack([rng.permutation(N) for _ in range(N_STREAM)])
    flips = rng.randint(0, 2, size=(N_STREAM, N)) * 2 - 1
    ref = np.load(CNS.OLD / 'permutations.npz')
    if not (np.array_equal(ref['perms'], perms) and np.array_equal(ref['flips'], flips)):
        raise SystemExit('permutation stream differs from the stored one')
    ddir = OUT / 'draws'; ddir.mkdir(exist_ok=True)
    scratch = OUT / 'scratch'; scratch.mkdir(exist_ok=True)

    def draw(p, config):
        prm, f = perms[p], flips[p].astype(float)
        mk = lambda M, sgn=None: pd.DataFrame(M[:, prm] * (sgn[None, :] if sgn is not None else 1.0),
                                              index=genes, columns=order)
        cov = pd.DataFrame(C[prm], index=order, columns=I['cov_df'].columns)
        for q in scratch.glob('*'):
            q.unlink()
        va, vt = V[config]
        with contextlib.redirect_stdout(io.StringIO()):
            map_nominal(gdf, vdf[['chrom', 'pos']], mk(A, f), mk(T), mk(va), mk(vt), gp,
                        xL_df=xLdf, xR_df=xRdf, prefix='n', covariates_df=cov,
                        genotype_covariates_df=G, window=CM.WIN, output_dir=str(scratch),
                        verbose=False, ase_covariates_df=None)
        df = pd.concat([pd.read_parquet(q, columns=COLS + NEW_COLS) for q in sorted(scratch.glob('n*.parquet'))],
                       ignore_index=True)
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
        old = pd.read_parquet(stored_files(c)[0]).sort_values(['phenotype_id', 'variant_id'], ignore_index=True)
        if not new[['phenotype_id', 'variant_id']].equals(old[['phenotype_id', 'variant_id']]):
            raise SystemExit(f'{c}: tested pairs differ from the stored draw 0')
        v = {k: new[k].values.astype(float) for k in COLS[2:] + ['dof_a', 'dof_t', 'dof_nominal']}
        o = {k: old[k].values.astype(float) for k in COLS[2:]}
        adm = new.allelic_admitted.values
        g_n_a = new.phenotype_id.map(dict(zip(genes, n_a))).values
        on = g_n_a >= 2          # through-origin allelic channel is switched off below 2 donors
        old73 = v['dof_t'] == OLD_DOF
        gates[c] = dict(
            fit_channels_max_dev_over_se=worst,
            channel_slopes_vs_stored=max(max_dev(v['slope_a'], o['slope_a'], o['slope_a_se']),
                                         max_dev(v['slope_t'], o['slope_t'], o['slope_t_se'])),
            channel_se_vs_stored=max(max_dev(v['slope_a_se'], o['slope_a_se'], o['slope_a_se']),
                                     max_dev(v['slope_t_se'], o['slope_t_se'], o['slope_t_se'])),
            combined_vs_stored_admitted=max(max_dev(v['slope'][adm], o['slope'][adm], o['slope_se'][adm]),
                                            max_dev(v['slope_se'][adm], o['slope_se'][adm], o['slope_se'][adm])),
            combined_is_total_below_floor=max(
                max_dev(v['slope'][~adm], v['slope_t'][~adm], v['slope_t_se'][~adm]),
                max_dev(v['slope_se'][~adm], v['slope_t_se'][~adm], v['slope_t_se'][~adm]),
                max_dev(v['pval_nominal'][~adm], v['pval_t'][~adm], np.ones((~adm).sum()))),
            pval_t_vs_stored_where_dof_t_73=max_dev(v['pval_t'][old73], o['pval_t'][old73],
                                                    np.maximum(o['pval_t'][old73], 1e-30)),
            rows_dof_t_not_73=int((~old73).sum()),
            floor_mismatches=int((adm != (g_n_a >= FLOOR)).sum()),
            dof_a_mismatches=int((on & (v['dof_a'] != g_n_a - 1)).sum() + np.isfinite(v['dof_a'][~on]).sum()))
        print(f'gate {c}: ' + '  '.join(f'{k} {x:.2e}' if isinstance(x, float) else f'{k} {x}'
                                        for k, x in gates[c].items()), flush=True)
        g = gates[c]
        if not (g['fit_channels_max_dev_over_se'] < GATE_TOL and g['channel_slopes_vs_stored'] < PAIR_TOL
                and g['channel_se_vs_stored'] < PAIR_TOL and g['combined_vs_stored_admitted'] < PAIR_TOL
                and g['combined_is_total_below_floor'] < PAIR_TOL
                and g['pval_t_vs_stored_where_dof_t_73'] < PAIR_TOL
                and g['floor_mismatches'] == 0 and g['dof_a_mismatches'] == 0):
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
    subsets = {'all': gi, f'without {DROPPED_GENE}': gi[gi != genes.index(DROPPED_GENE)]}
    subsets.update({b: gi[(n_a >= lo) & (n_a < hi)] for b, (lo, hi) in BINS.items()})
    brng = np.random.default_rng(np.random.SeedSequence(SEED).spawn(8)[7])
    boot = {s: ix[brng.integers(0, len(ix), size=(N_BOOT, len(ix)))] for s, ix in subsets.items()}
    after = {c: sorted((OUT / 'draws').glob(f'{c}_*.parquet')) for c in CONFIGS}
    n_draw = min(len(v) for v in after.values())
    before = {c: stored_files(c)[:n_draw] for c in CONFIGS}
    below = [g for g, k in zip(genes, n_a) if k < FLOOR]

    def boot_rates(K, n, s):
        B = boot[s]
        nb = n[B].sum(1)
        return K[B].sum(1) / np.where(nb > 0, nb, np.nan), int((nb == 0).sum())

    res = dict(pass_rule='after the fix, the combined rate at 0.001 for split, unit and plus_one lies within '
                         'its gene-clustered 95% interval of 0.001, RPL41 included; gibbs reported, not judged',
               n_draw=n_draw, n_genes=len(genes), floor=FLOOR, old_dof=OLD_DOF, n_boot=N_BOOT,
               genes_per_subset={s: len(ix) for s, ix in subsets.items()},
               gates=json.loads((OUT / 'gates.json').read_text()),
               before_stored_summary={}, rates={}, below_floor_genes={}, dof={}, verdict={})
    for c in CONFIGS:
        d, arm, sf = STORED[c]
        stored = json.loads((d / sf).read_text())
        res['before_stored_summary'][c] = {ch: {al: stored['rates'][f'{arm} {ch}'][al] for al in map(str, ALPHAS)}
                                           for ch in CHANNELS}
        if stored['n_draw'] != n_draw:
            print(f'{c}: stored-summary check skipped, {n_draw} of its {stored["n_draw"]} draws', flush=True)
        res['rates'][c] = {}
        for ch, col in CHANNELS.items():
            Ka, na = CNS.rates_by_gene(after[c], genes, col)
            Kb, nb = CNS.rates_by_gene(before[c], genes, col)
            for al in ALPHAS:
                got, want = Kb[al].sum() / nb.sum(), res['before_stored_summary'][c][ch][str(al)]['rate']
                if stored['n_draw'] == n_draw and abs(got - want) > 1e-12:
                    raise SystemExit(f'{c} {ch} {al}: stored draws give {got}, stored summary {want}')
            res['rates'][c][ch] = {}
            for s, ix in subsets.items():
                row = {}
                for tag, K, n in (('before', Kb, nb), ('after', Ka, na)):
                    row[tag] = {'n_tests': int(n[ix].sum())}
                    for al in ALPHAS:
                        b, empty = boot_rates(K[al], n, s)
                        row[tag][str(al)] = dict(rate=float(K[al][ix].sum() / n[ix].sum()),
                                                 lo=float(np.nanquantile(b, .025)), hi=float(np.nanquantile(b, .975)),
                                                 resamples_without_tests=empty)
                row['after_minus_before'] = {}
                for al in ALPHAS:
                    dd = boot_rates(Ka[al], na, s)[0] - boot_rates(Kb[al], nb, s)[0]
                    row['after_minus_before'][str(al)] = dict(
                        diff=row['after'][str(al)]['rate'] - row['before'][str(al)]['rate'],
                        lo=float(np.nanquantile(dd, .025)), hi=float(np.nanquantile(dd, .975)))
                res['rates'][c][ch][s] = row
            for g in below:
                k = genes.index(g)
                res['below_floor_genes'].setdefault(g, {'n_a': int(n_a[k])}).setdefault(c, {})[ch] = {
                    tag: {'n_tests': int(n[k]), **{str(al): float(K[al][k] / n[k]) if n[k] else None for al in ALPHAS}}
                    for tag, K, n in (('before', Kb, nb), ('after', Ka, na))}
        d0 = pd.read_parquet(after[c][0], columns=['phenotype_id'] + NEW_COLS)
        per_gene = d0.groupby('phenotype_id').agg(dof_a=('dof_a', 'first'), dof_t=('dof_t', 'first'),
                                                  admitted=('allelic_admitted', 'first'))
        g_bin = pd.Series({g: b for b, ix in subsets.items() if b in BINS for g in np.array(genes)[ix]})
        res['dof'][c] = dict(
            dof_t_values=sorted(float(x) for x in per_gene.dof_t.unique()),
            genes_admitted=int(per_gene.admitted.sum()),
            dof_nominal_draw0={b: dict(median=float(x.median()), min=float(x.min()), max=float(x.max()),
                                       share_above_old=float((x > OLD_DOF).mean()))
                               for b, x in d0.groupby(d0.phenotype_id.map(g_bin)).dof_nominal})
        if c in JUDGED:
            r = res['rates'][c]['combined']['all']['after']['0.001']
            res['verdict'][c] = dict(lo=r['lo'], hi=r['hi'], rate=r['rate'], contains_0_001=r['lo'] <= 0.001 <= r['hi'])
    res['passed'] = all(v['contains_0_001'] for v in res['verdict'].values())
    tmp = OUT / 'summary.json.tmp'
    tmp.write_text(json.dumps(res, indent=1)); tmp.rename(OUT / 'summary.json')

    f = lambda x: f'{x["rate"]:.4f} [{x["lo"]:.4f}, {x["hi"]:.4f}]'
    fd = lambda x: f'{x["diff"]:+.4f} [{x["lo"]:+.4f}, {x["hi"]:+.4f}]'
    nr = f'without {DROPPED_GENE}'
    print(f'\n{n_draw} permutations, {len(genes)} genes; rates with gene-clustered 95% intervals')
    print(f'{"config":8s} {"channel":8s} {"alpha":>5s}  {"before (stored summary)":26s}{"after":26s}'
          f'{"before, " + nr:26s}{"after, " + nr:26s}after - before (all genes)')
    for c in CONFIGS:
        for ch in CHANNELS:
            R = res['rates'][c][ch]
            for al in map(str, ALPHAS):
                print(f'{c:8s} {ch:8s} {al:>5s}  {f(res["before_stored_summary"][c][ch][al]):26s}'
                      f'{f(R["all"]["after"][al]):26s}{f(R[nr]["before"][al]):26s}{f(R[nr]["after"][al]):26s}'
                      f'{fd(R["all"]["after_minus_before"][al])}')
    print(f'\nby informative allelic donor count; before -> after at 0.05 / 0.01 / 0.001 (after intervals)')
    for c in CONFIGS:
        for ch in CHANNELS:
            for b in BINS:
                R = res['rates'][c][ch][b]
                print(f'{c:8s} {ch:8s} {b:10s} ({res["genes_per_subset"][b]:2d} genes)  '
                      + ' / '.join(f'{R["before"][al]["rate"]:.4f}' for al in map(str, ALPHAS)) + '  ->  '
                      + ' / '.join(f(R['after'][al]) for al in map(str, ALPHAS)))
    print('\ngenes below the floor, per-gene rate before -> after at 0.05 / 0.01 / 0.001')
    fmt = lambda x: ' / '.join('   n/a' if x[al] is None else f'{x[al]:.4f}' for al in map(str, ALPHAS))
    for g, v in res['below_floor_genes'].items():
        for c in CONFIGS:
            for ch in ('combined', 'allelic'):
                print(f'{g:7s} n_a={v["n_a"]:2d} {c:8s} {ch:8s} {fmt(v[c][ch]["before"])}  ->  {fmt(v[c][ch]["after"])}')
    for c in CONFIGS:
        print(f'tests {c:8s} before -> after: ' + '  '.join(
            f'{ch} {res["rates"][c][ch]["all"]["before"]["n_tests"]:,} -> {res["rates"][c][ch]["all"]["after"]["n_tests"]:,}'
            for ch in CHANNELS))
    for c in CONFIGS:
        print(f'dof {c:8s} dof_t {res["dof"][c]["dof_t_values"]}  admitted genes {res["dof"][c]["genes_admitted"]}  '
              'dof_nominal draw 0 median [min, max] ' + '  '.join(
                  f'{b}: {x["median"]:.1f} [{x["min"]:.1f}, {x["max"]:.1f}]'
                  for b, x in res['dof'][c]['dof_nominal_draw0'].items()))
    for c, v in res['verdict'].items():
        print(f'{"PASS" if v["contains_0_001"] else "FAIL"} {c}: combined at 0.001 after the fix, RPL41 included, '
              f'{v["rate"]:.5f} [{v["lo"]:.5f}, {v["hi"]:.5f}] '
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

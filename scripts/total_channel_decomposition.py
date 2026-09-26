"""Which correction made the total channel's nominal p worse?

corrected_null_store.py found the total channel rejecting at 0.0835 / 0.0261 /
0.0066 at nominal 0.05 / 0.01 / 0.001 on the corrected pipeline, against
0.0714 / 0.0187 / 0.0031 before the correction on the 90 shared genes. The
correction changed three things in the total channel at once. This undoes them
one at a time on the SAME 100 genes and the SAME 200 permutations (the
corrected store's stream), total channel only.

Arms (value T, its variance Vt, covariates and how they move):
  corrected     log2(CPM + 1) of point estimates, CPM on edgeR's effective
                library size; Vt = across-draw variance of the same transform
                + the counting term; new covariates, genotype PCs held with the
                genotypes. Read from corrected_null_store's stored draws.
  geno_moving   corrected, but the genotype PCs move with the RNA record
  old_covs      corrected values, the pre-correction covariate file
                (cov/covariates.tsv), every column moving with the record
  old_values    pre-correction values (compute_summaries_from_gibbs: mean over
                draws of log(count / 2 + 1/2), Vt its across-draw variance +
                1 / (mean count + 1)), new covariates, genotype PCs held
  pre           pre-correction values AND covariates, all moving: the old
                pipeline on these 100 genes
  half_read     log2((count + 1/2) / L * 1e6) of point estimates: library
                normalized, but a half-read pseudocount instead of one CPM
                (about 17 reads at the median library); Vt the across-draw
                variance of the same transform + 1 / ((count + 1/2) ln2^2)
  raw_half_read log2(count + 1/2): as half_read without the library size, so
                it differs from half_read only by each donor's log2(1e6 / L)
  unit_weights  corrected values with every Vt equal, so the fit is unweighted
                and the residual scale is fitted as always

The allelic channel is switched off (keep_a_df all False); the total channel
is fitted exactly as in map_nominal. GATES on draw 0: the corrected arm
recomputed here must reproduce corrected_null_store's stored pval_t, and every
arm's total slope at each gene's most heterozygous tested variant must equal
null_permutation_instrument.fit_channels on the same inputs within 1e-3 se.

Summary: rejection rates per arm at 0.05 / 0.01 / 0.001 with gene-clustered
95% intervals, each arm's paired difference from `corrected` on the same
resampled genes, and rates by allele-resolved coverage.
Usage: total_channel_decomposition.py [n_draw=200] [--arms=a,b] [--summarize-only]
  --arms restricts a process to those arms (they write disjoint files, so arms
  can run as parallel processes); summarize after all have finished.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import compare_mixqtl_replication as CM                          # noqa: E402
from null_permutation_instrument import fit_channels             # noqa: E402
from tensorqtl.hapmixqtl import (map_nominal, summaries_from_point_estimates,  # noqa: E402
                                 compute_summaries_from_gibbs, LN2)
import corrected_null_store as CNS                               # noqa: E402

D = CNS.D
OUT = D / 'total_channel_decomposition_20260926'
SEED, N_STREAM, N_BOOT, EPS, KAPPA = 42, 1000, 2000, 1e-12, 0.5
ALPHAS = (0.05, 0.01, 0.001)
NEW_ARMS = ('geno_moving', 'old_covs', 'old_values', 'pre', 'half_read', 'raw_half_read', 'unit_weights')
COLS = ['phenotype_id', 'variant_id', 'pval_t', 'slope_t', 'slope_t_se']


def build():
    genes_file = CNS.OUT / 'genes.txt'
    with contextlib.redirect_stdout(io.StringIO()):
        I = CM.load_point_estimate_inputs(gene_list=str(genes_file), regions=str(CNS.OUT / 'regions.bed'))
    keep = I['keep']
    ks = lambda x: x[:, keep]
    A, T_c, Va, Vt_c, _ = summaries_from_point_estimates(I['pL'], I['pR'], I['pT'], I['eff_lib'],
                                                         I['YL'], I['YR'], I['YT'])
    with contextlib.redirect_stdout(io.StringIO()):
        _, T_o, _, Vt_o, _ = compute_summaries_from_gibbs(ks(I['YL']), ks(I['YR']), yT=ks(I['YT']))
    k = (1e6 / I['eff_lib'])[keep]
    pT, YT = ks(I['pT']), ks(I['YT'])
    v_half = np.log2(YT + KAPPA).var(2) + 1 / ((pT + KAPPA) * LN2 ** 2)
    T_half = np.log2((pT + KAPPA) * k[None, :])
    T_raw = np.log2(pT + KAPPA)
    order = I['order']
    new_rna, geno = I['cov_df'], I['geno_cov_df']
    old_cov = pd.read_csv(D / 'cov' / 'covariates.tsv', sep='\t', index_col=0)
    old_cov.index = old_cov.index.astype(str)
    old_cov = old_cov.loc[order]
    both = pd.concat([new_rna, geno], axis=1)
    # (T, Vt, covariates that move with the record, covariates held with the genotypes)
    arms = dict(
        corrected=(ks(T_c), ks(Vt_c), new_rna, geno),
        geno_moving=(ks(T_c), ks(Vt_c), both, None),
        old_covs=(ks(T_c), ks(Vt_c), old_cov, None),
        old_values=(T_o, Vt_o, new_rna, geno),
        pre=(T_o, Vt_o, old_cov, None),
        half_read=(T_half, v_half, new_rna, geno),
        raw_half_read=(T_raw, v_half, new_rna, geno),
        unit_weights=(ks(T_c), np.ones_like(ks(Vt_c)), new_rna, geno))
    return I, ks(A), ks(Va), arms


def run_draws(n_draw, run_arms=NEW_ARMS):
    OUT.mkdir(exist_ok=True)
    I, A, Va, arms = build()
    genes, order = list(I['genes']), I['order']
    N = len(order)
    gp = I['gp'].loc[genes][['chr', 'pos']]
    vdf = I['vdf']
    gdf = pd.DataFrame(I['dos'], index=vdf.index, columns=order)
    xLdf = pd.DataFrame(I['xL'], index=vdf.index, columns=order)
    xRdf = pd.DataFrame(I['xR'], index=vdf.index, columns=order)
    tested_idx = {g: I['idx'][CM.gene_variant_index(I, g)] for g in genes}
    tested = pd.DataFrame([(g, str(v)) for g in genes for v in vdf.index[tested_idx[g]]],
                          columns=['phenotype_id', 'variant_id'])
    off = pd.DataFrame(False, index=genes, columns=order)          # allelic channel off
    rng = np.random.RandomState(SEED)
    perms = np.stack([rng.permutation(N) for _ in range(N_STREAM)])
    flips = rng.randint(0, 2, size=(N_STREAM, N)) * 2 - 1
    ref = np.load(CNS.OLD / 'permutations.npz')
    if not (np.array_equal(ref['perms'], perms) and np.array_equal(ref['flips'], flips)):
        raise SystemExit('permutation stream differs from the corrected store')
    ddir = OUT / 'draws'; ddir.mkdir(exist_ok=True)
    scratch = OUT / ('scratch_' + '_'.join(run_arms)); scratch.mkdir(exist_ok=True)

    def draw(p, arm):
        T, Vt, mov, tied = arms[arm]
        prm = perms[p]
        mk = lambda M: pd.DataFrame(M[:, prm], index=genes, columns=order)
        cov = pd.DataFrame(mov.values[prm], index=order, columns=mov.columns)
        for q in scratch.glob('*'):
            q.unlink()
        with contextlib.redirect_stdout(io.StringIO()):
            map_nominal(gdf, vdf[['chrom', 'pos']], mk(A), mk(T), mk(Va), mk(Vt), gp,
                        xL_df=xLdf, xR_df=xRdf, prefix='n', covariates_df=cov,
                        genotype_covariates_df=tied, window=CM.WIN, output_dir=str(scratch),
                        verbose=False, ase_covariates_df=None, keep_a_df=off)
        df = pd.concat([pd.read_parquet(q, columns=COLS) for q in sorted(scratch.glob('n*.parquet'))],
                       ignore_index=True)
        df['variant_id'] = df['variant_id'].astype(str)
        df = df.merge(tested, on=['phenotype_id', 'variant_id'], how='inner')
        for c in COLS[2:]:
            df[c] = df[c].astype(np.float32)
        return df

    # ---- gates -------------------------------------------------------------
    c0 = draw(0, 'corrected').set_index(['phenotype_id', 'variant_id'])
    st = pd.read_parquet(CNS.OUT / 'draws' / 'keep_000.parquet', columns=COLS).set_index(
        ['phenotype_id', 'variant_id']).reindex(c0.index)
    g1 = float(np.nanmax(np.abs(c0.pval_t.values - st.pval_t.values)))
    print(f'gate 1: corrected arm vs stored corrected draw 0, max |pval_t diff| = {g1:.2e}', flush=True)
    if not g1 < 1e-5:
        raise SystemExit('GATE 1 FAILED')
    first = {arm: draw(0, arm) for arm in run_arms}
    prm0 = perms[0]
    worst = 0.0
    for kx, g in enumerate(genes):
        cand = tested_idx[g]
        if not len(cand):
            continue
        j = cand[int(np.argmax(((I['xL'][cand] - I['xR'][cand]) != 0).sum(1)))]
        s = (I['xL'][j] - I['xR'][j]).astype(float)
        gh = I['dos'][j].astype(float) / 2.0
        for arm in run_arms:
            T, Vt, mov, tied = arms[arm]
            Cp = mov.values[prm0] if tied is None else np.column_stack([mov.values[prm0], tied.values])
            row = first[arm][(first[arm].phenotype_id == g) & (first[arm].variant_id == str(vdf.index[j]))]
            fc = fit_channels(np.zeros(N), s, np.ones(N), T[kx][prm0], gh, Vt[kx][prm0], Cp)
            if row.empty or fc is None:
                continue
            worst = max(worst, abs(float(row.iloc[0].slope_t) - fc['bt']) / float(row.iloc[0].slope_t_se))
    print(f'gate 2: draw-0 total slopes vs reference fit, all arms, max |diff| / se = {worst:.2e}', flush=True)
    if not worst < 1e-3:
        raise SystemExit('GATE 2 FAILED')

    t0 = time.time()
    for p in range(n_draw):
        for arm in run_arms:
            fo = ddir / f'{arm}_{p:03d}.parquet'
            if fo.exists():
                continue
            df = first[arm] if p == 0 else draw(p, arm)
            df.to_parquet(fo.with_suffix('.tmp'), compression='zstd', index=False)
            fo.with_suffix('.tmp').rename(fo)
        if p % 10 == 9 or p == n_draw - 1:
            print(f'  draw {p + 1}/{n_draw}  ({(time.time() - t0) / 60:.1f} min)', flush=True)
    shutil.rmtree(scratch, ignore_errors=True)


def summarize():
    genes = (CNS.OUT / 'genes.txt').read_text().split()
    design = pd.read_csv(CNS.OUT / 'gene_design.tsv', sep='\t').set_index('gene').loc[genes]
    files = {'corrected': sorted((CNS.OUT / 'draws').glob('keep_*.parquet'))}
    for arm in NEW_ARMS:
        files[arm] = sorted((OUT / 'draws').glob(f'{arm}_*.parquet'))
    n_draw = min(len(v) for v in files.values())
    brng = np.random.default_rng(np.random.SeedSequence(SEED).spawn(6)[5])
    bidx = brng.integers(0, len(genes), size=(N_BOOT, len(genes)))
    cov_bin = pd.cut(design.median_allele_resolved_reads, [-1, 700, np.inf], labels=['<700', '>=700'])
    KN = {arm: CNS.rates_by_gene(fl[:n_draw], genes, 'pval_t') for arm, fl in files.items()}
    res = dict(n_draw=n_draw, n_genes=len(genes), rates={}, minus_corrected={}, by_coverage={})
    Kc, nc = KN['corrected']
    for arm, (K, n) in KN.items():
        res['rates'][arm] = {}
        res['minus_corrected'][arm] = {}
        for al in ALPHAS:
            b = K[al][bidx].sum(1) / n[bidx].sum(1)
            res['rates'][arm][str(al)] = dict(rate=float(K[al].sum() / n.sum()),
                                              lo=float(np.quantile(b, .025)), hi=float(np.quantile(b, .975)))
            dd = b - Kc[al][bidx].sum(1) / nc[bidx].sum(1)
            res['minus_corrected'][arm][str(al)] = dict(
                diff=float(K[al].sum() / n.sum() - Kc[al].sum() / nc.sum()),
                lo=float(np.quantile(dd, .025)), hi=float(np.quantile(dd, .975)))
        res['by_coverage'][arm] = {str(cb): {str(al): float(K[al][(cov_bin == cb).values].sum()
                                                            / n[(cov_bin == cb).values].sum()) for al in ALPHAS}
                                   for cb in cov_bin.cat.categories}
    (OUT / 'summary.json').write_text(json.dumps(res, indent=1))
    for arm in files:
        r, d = res['rates'][arm], res['minus_corrected'][arm]
        print(f'{arm:14s} ' + '  '.join(f'{al}: {r[al]["rate"]:.4f} [{r[al]["lo"]:.4f}, {r[al]["hi"]:.4f}]'
                                        for al in map(str, ALPHAS))
              + '   minus corrected at 0.05: ' + f'{d["0.05"]["diff"]:+.4f} [{d["0.05"]["lo"]:+.4f}, {d["0.05"]["hi"]:+.4f}]'
              + '   <700 / >=700 at 0.05: ' + ' / '.join(f'{res["by_coverage"][arm][cb]["0.05"]:.4f}'
                                                         for cb in ('<700', '>=700')))


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    n_draw = int(args[0]) if args else 200
    sel = [a.split('=', 1)[1].split(',') for a in sys.argv[1:] if a.startswith('--arms=')]
    run_arms = tuple(sel[0]) if sel else NEW_ARMS
    if any(a not in NEW_ARMS for a in run_arms):
        raise SystemExit(f'unknown arm in {run_arms}')
    if '--summarize-only' not in sys.argv:
        run_draws(n_draw, run_arms)
        if sel:
            return
    summarize()
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()

"""Gibbs weights in the allelic channel, unit weights in the total channel: calibrated?

Proposal (2026-09-26): the per-gene measurements found the Gibbs weights buy a
median 1.38-fold in the allelic slope's null spread and nothing in the total
channel (median 0.93), where they also make the nominal p anticonservative.
This runs the split configuration on the corrected null store's 100 genes and
200 permutations and compares it, paired, with that store's `drop` arm
(Gibbs weights in both channels, zero-haplotype pairs dropped):

  hybrid   allelic: Gibbs variance, zero-haplotype pairs dropped
           total:   every donor's variance 1, so the fit is unweighted and the
                    residual scale is fitted as always
Everything else as corrected_null_store: point-estimate values, log2 CPM on
edgeR library sizes, new covariates with the genotype PCs held with the
genotypes, records_signflip permutations from the same stream.

GATE on draw 0 against null_permutation_instrument.fit_channels (both channel
slopes, 1e-3 se), as in corrected_null_store.
SUMMARY: rejection rates per channel with gene-clustered 95% intervals, the
paired hybrid - drop difference, and for the COMBINED slope the reported se
over the realized null spread and the realized spread relative to `drop`.
Usage: hybrid_weights_null.py [n_draw=200] [--summarize-only]
"""
import contextlib
import io
import json
import shutil
import sys
import time

import numpy as np
import pandas as pd

import compare_mixqtl_replication as CM
import corrected_null_store as CNS
from null_permutation_instrument import fit_channels
from tensorqtl.hapmixqtl import map_nominal, summaries_from_point_estimates

D = CNS.D
OUT = D / 'hybrid_weights_null_20260926'
SEED, N_STREAM, N_BOOT, EPS = 42, 1000, 2000, 1e-12
ALPHAS = (0.05, 0.01, 0.001)
CHANNELS = CNS.CHANNELS
COLS = CNS.COLS


def run_draws(n_draw):
    OUT.mkdir(exist_ok=True)
    with contextlib.redirect_stdout(io.StringIO()):
        I = CM.load_point_estimate_inputs(gene_list=str(CNS.OUT / 'genes.txt'),
                                          regions=str(CNS.OUT / 'regions.bed'))
    order, keep = I['order'], I['keep']
    N = len(order)
    A, T, Va, Vt, _ = summaries_from_point_estimates(I['pL'], I['pR'], I['pT'], I['eff_lib'],
                                                     I['YL'], I['YR'], I['YT'])
    A, T, Va = A[:, keep], T[:, keep], Va[:, keep]
    pL, pR = I['pL'][:, keep], I['pR'][:, keep]
    Va = np.where((pL < 0.5) ^ (pR < 0.5), 0.0, Va)
    Vt_unit = np.ones_like(T)
    genes = list(I['genes'])
    gp = I['gp'].loc[genes][['chr', 'pos']]
    C, G = I['cov_df'].values, I['geno_cov_df']
    vdf = I['vdf']
    gdf = pd.DataFrame(I['dos'], index=vdf.index, columns=order)
    xLdf = pd.DataFrame(I['xL'], index=vdf.index, columns=order)
    xRdf = pd.DataFrame(I['xR'], index=vdf.index, columns=order)
    tested_idx = {g: I['idx'][CM.gene_variant_index(I, g)] for g in genes}
    tested = pd.DataFrame([(g, str(v)) for g in genes for v in vdf.index[tested_idx[g]]],
                          columns=['phenotype_id', 'variant_id'])
    rng = np.random.RandomState(SEED)
    perms = np.stack([rng.permutation(N) for _ in range(N_STREAM)])
    flips = rng.randint(0, 2, size=(N_STREAM, N)) * 2 - 1
    ref = np.load(CNS.OLD / 'permutations.npz')
    if not (np.array_equal(ref['perms'], perms) and np.array_equal(ref['flips'], flips)):
        raise SystemExit('permutation stream differs from the corrected store')
    ddir = OUT / 'draws'; ddir.mkdir(exist_ok=True)
    scratch = OUT / 'scratch'; scratch.mkdir(exist_ok=True)

    def draw(p):
        prm, f = perms[p], flips[p].astype(float)
        mk = lambda M, sgn=None: pd.DataFrame(M[:, prm] * (sgn[None, :] if sgn is not None else 1.0),
                                              index=genes, columns=order)
        cov = pd.DataFrame(C[prm], index=order, columns=I['cov_df'].columns)
        for q in scratch.glob('*'):
            q.unlink()
        with contextlib.redirect_stdout(io.StringIO()):
            map_nominal(gdf, vdf[['chrom', 'pos']], mk(A, f), mk(T), mk(Va), mk(Vt_unit), gp,
                        xL_df=xLdf, xR_df=xRdf, prefix='n', covariates_df=cov,
                        genotype_covariates_df=G, window=CM.WIN, output_dir=str(scratch),
                        verbose=False, ase_covariates_df=None)
        df = pd.concat([pd.read_parquet(q, columns=COLS) for q in sorted(scratch.glob('n*.parquet'))],
                       ignore_index=True)
        df['variant_id'] = df['variant_id'].astype(str)
        df = df.merge(tested, on=['phenotype_id', 'variant_id'], how='inner')
        for c in COLS[2:]:
            df[c] = df[c].astype(np.float32)
        return df

    first = draw(0)
    prm0, f0 = perms[0], flips[0].astype(float)
    Cg = np.column_stack([C[prm0], G.values])
    worst = 0.0
    for k, g in enumerate(genes):
        cand = tested_idx[g]
        if not len(cand):
            continue
        j = cand[int(np.argmax(((I['xL'][cand] - I['xR'][cand]) != 0).sum(1)))]
        s = (I['xL'][j] - I['xR'][j]).astype(float)
        gh = I['dos'][j].astype(float) / 2.0
        row = first[(first.phenotype_id == g) & (first.variant_id == str(vdf.index[j]))]
        fc = fit_channels(A[k][prm0] * f0, s, Va[k][prm0], T[k][prm0], gh, Vt_unit[k][prm0], Cg)
        if row.empty or fc is None:
            continue
        r = row.iloc[0]
        worst = max(worst, abs(float(r.slope_a) - fc['ba']) / float(r.slope_a_se),
                    abs(float(r.slope_t) - fc['bt']) / float(r.slope_t_se))
    print(f'gate: draw-0 slopes vs reference fit, max |diff| / se = {worst:.2e}', flush=True)
    if not worst < 1e-3:
        raise SystemExit(f'GATE FAILED: {worst:.2e}')
    t0 = time.time()
    for p in range(n_draw):
        fo = ddir / f'hybrid_{p:03d}.parquet'
        if fo.exists():
            continue
        df = first if p == 0 else draw(p)
        df.to_parquet(fo.with_suffix('.tmp'), compression='zstd', index=False)
        fo.with_suffix('.tmp').rename(fo)
        if p % 20 == 19:
            print(f'  draw {p + 1}/{n_draw}  ({(time.time() - t0) / 60:.1f} min)', flush=True)
    shutil.rmtree(scratch, ignore_errors=True)


def slope_moments(files, b_col, s_col):
    acc = idx = None
    for f in files:
        d = pd.read_parquet(f, columns=['phenotype_id', 'variant_id', b_col, s_col]).set_index(
            ['phenotype_id', 'variant_id']).sort_index()
        b, s = d[b_col].values.astype(float), d[s_col].values.astype(float)
        if acc is None:
            idx, acc = d.index, np.zeros((4, len(b)))
        ok = np.isfinite(b) & np.isfinite(s)
        acc += np.stack([ok, np.where(ok, b, 0), np.where(ok, b * b, 0), np.where(ok, s * s, 0)])
    n = acc[0]
    with np.errstate(invalid='ignore', divide='ignore'):
        return idx, np.sqrt(acc[3] / n), np.sqrt((acc[2] - acc[1] ** 2 / n) / (n - 1))


def summarize():
    genes = (CNS.OUT / 'genes.txt').read_text().split()
    files = {'hybrid': sorted((OUT / 'draws').glob('hybrid_*.parquet')),
             'drop': sorted((CNS.OUT / 'draws').glob('drop_*.parquet'))}
    n_draw = min(len(v) for v in files.values())
    files = {k: v[:n_draw] for k, v in files.items()}
    brng = np.random.default_rng(np.random.SeedSequence(SEED).spawn(7)[6])
    bidx = brng.integers(0, len(genes), size=(N_BOOT, len(genes)))
    res = dict(n_draw=n_draw, rates={}, hybrid_minus_drop={}, combined_slope={})
    for ch, col in CHANNELS.items():
        KN = {arm: CNS.rates_by_gene(files[arm], genes, col) for arm in files}
        for arm, (K, n) in KN.items():
            res['rates'][f'{arm} {ch}'] = {str(al): dict(
                rate=float(K[al].sum() / n.sum()),
                lo=float(np.quantile(K[al][bidx].sum(1) / n[bidx].sum(1), .025)),
                hi=float(np.quantile(K[al][bidx].sum(1) / n[bidx].sum(1), .975))) for al in ALPHAS}
        (Kh, nh), (Kd, nd) = KN['hybrid'], KN['drop']
        res['hybrid_minus_drop'][ch] = {}
        for al in ALPHAS:
            dd = Kh[al][bidx].sum(1) / nh[bidx].sum(1) - Kd[al][bidx].sum(1) / nd[bidx].sum(1)
            res['hybrid_minus_drop'][ch][str(al)] = dict(diff=float(Kh[al].sum() / nh.sum() - Kd[al].sum() / nd.sum()),
                                                         lo=float(np.quantile(dd, .025)), hi=float(np.quantile(dd, .975)))
    ih, rep_h, real_h = slope_moments(files['hybrid'], 'slope', 'slope_se')
    idr, rep_d, real_d = slope_moments(files['drop'], 'slope', 'slope_se')
    if not ih.equals(idr):
        raise SystemExit('arms cover different variants')
    ok = (real_h > 0) & (real_d > 0) & np.isfinite(rep_h) & np.isfinite(rep_d)
    res['combined_slope'] = dict(
        hybrid_reported_over_realized=float(np.median(rep_h[ok] / real_h[ok])),
        drop_reported_over_realized=float(np.median(rep_d[ok] / real_d[ok])),
        realized_hybrid_over_drop=float(np.median(real_h[ok] / real_d[ok])),
        realized_hybrid_over_drop_per_gene_q10_q50_q90=np.quantile(
            pd.DataFrame(dict(g=ih.get_level_values(0)[ok], r=(real_h / real_d)[ok])).groupby('g').r.median(),
            [.1, .5, .9]).tolist())
    (OUT / 'summary.json').write_text(json.dumps(res, indent=1))
    for k, v in res['rates'].items():
        print(f'{k:18s} ' + '  '.join(f'{al}: {v[al]["rate"]:.4f} [{v[al]["lo"]:.4f}, {v[al]["hi"]:.4f}]'
                                      for al in map(str, ALPHAS)))
    for ch, v in res['hybrid_minus_drop'].items():
        print(f'hybrid - drop {ch:9s} ' + '  '.join(
            f'{al}: {v[al]["diff"]:+.4f} [{v[al]["lo"]:+.4f}, {v[al]["hi"]:+.4f}]' for al in map(str, ALPHAS)))
    print('combined slope:', {k: (np.round(v, 3).tolist() if isinstance(v, list) else round(v, 3))
                              for k, v in res['combined_slope'].items()})


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    n_draw = int(args[0]) if args else 200
    if '--summarize-only' not in sys.argv:
        run_draws(n_draw)
    summarize()
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()

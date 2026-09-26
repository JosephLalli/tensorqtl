"""Nominal-p calibration of default mode on the corrected pipeline, with and
without the zero-haplotype drop.

QUESTION (user, 2026-09-25). If a donor-gene pair whose Salmon point estimate
puts one haplotype at zero is dropped from the allelic channel (the total
channel keeps it), is the nominal p calibrated on permuted null data?

PIPELINE: the 2026-09-25 rules (docs/pipeline_rules.md). Values from Salmon
point estimates, Gibbs draws only for variance (summaries_from_point_estimates:
allelic log2((L+1/2)/(R+1/2)), total log2(CPM+1) on edgeR's effective library
size); covariates cov/log2cpm1_point_calibration_20260925 with the genotype
PCs tied to the genotypes; default mode (tau_mode='zero', se_mode='fitted');
allelic channel through the origin.

ARMS, on identical permutations:
  keep  the point-estimate values as they are
  drop  pairs with one haplotype below 0.5 reads in the point estimate get
        allelic variance 0, the state of a pair with no allelic information,
        so their allelic weight is 0; the total channel is unchanged

GENES: the 90 genes of protein_coding_null_store_20260925 that pass the
corrected calibration gene filter, plus 10 drawn from the rest of that filter
(genes with Gibbs draws and a unique position in annot/genes.tsv) with a
SeedSequence(42) child stream.

NULL: records_signflip. Each donor's record (both values, both Gibbs
variances, the RNA-tied covariate row) moves against fixed genotypes and
genotype PCs, and each permuted record's L/R labels are swapped with
probability one half. The stream is the one protein_coding_null_store used
(RandomState(42), 1,000 permutations, then the swap signs), checked equal to
its stored permutations.npz, so draw p here and there share a permutation.

GATE before storing: on draw 0, at each gene's variant with the most
heterozygotes, map_nominal's allelic and total slopes for both arms must equal
null_permutation_instrument.fit_channels on the same permuted inputs, within
1e-3 of the slope's se.

STORED: draws/{keep,drop}_NNN.parquet, tested variants only (cis window,
outside the gene body, MAF >= 0.05): nominal p, slope and se, combined,
allelic and total. Resumes by skipping draws whose files exist.

SUMMARY (summary.json): pooled rejection rates at 0.05 / 0.01 / 0.001 per arm
and channel, gene-clustered bootstrap 95% intervals, the paired drop - keep
difference on the same resampled genes, rates by allele-resolved coverage,
and the pre-correction store's rates on the 90 shared genes for comparison.

Usage: corrected_null_store.py [n_draw=200] [--summarize-only]
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
from tensorqtl.hapmixqtl import map_nominal, summaries_from_point_estimates  # noqa: E402

D = Path('/mnt/ssd/lalli/brainvar_hapmix_deploy')
CACHE = D / 'cache' / 'gibbs_56b63c3b37ed5df8'
OLD = D / 'protein_coding_null_store_20260925'
OUT = D / 'corrected_null_store_20260925'
SEED, N_STREAM, N_BOOT, EPS = 42, 1000, 2000, 1e-12
ALPHAS = (0.05, 0.01, 0.001)
ARMS = ('keep', 'drop')
CHANNELS = {'combined': 'pval_nominal', 'allelic': 'pval_a', 'total': 'pval_t'}
COLS = ['phenotype_id', 'variant_id', 'pval_nominal', 'slope', 'slope_se',
        'pval_a', 'slope_a', 'slope_a_se', 'pval_t', 'slope_t', 'slope_t_se']


def select_genes():
    f = OUT / 'genes.txt'
    if f.exists():
        return f.read_text().split()
    old = (OLD / 'genes_pc100.txt').read_text().split()
    cal = set((CACHE / 'point_estimates' / 'edger' / 'calibration_genes.txt').read_text().split())
    cache = set((CACHE / 'genes.txt').read_text().split())
    gp = pd.read_csv(D / 'annot' / 'genes.tsv', sep='\t', header=None, dtype={1: str},
                     names=['gene', 'chr', 'start', 'end', 'pos'])
    gp = gp[~gp.gene.duplicated(keep=False)].set_index('gene')
    shared = [g for g in old if g in cal]
    pool = sorted(g for g in cal & cache if g in gp.index and g not in set(old))
    rng = np.random.default_rng(np.random.SeedSequence(SEED).spawn(4)[3])
    new = sorted(rng.choice(pool, size=100 - len(shared), replace=False).tolist())
    genes = sorted(shared + new)
    t = gp.loc[genes].reset_index()
    t['source'] = ['shared' if g in shared else 'replacement' for g in genes]
    t.to_csv(OUT / 'gene_selection.tsv', sep='\t', index=False)
    with open(OUT / 'regions.bed', 'w') as fh:
        for r in t.itertuples():
            lo = max(0, min(r.start, r.pos) - CM.WIN - 1000)
            fh.write(f'{r.chr}\t{lo}\t{max(r.end, r.pos) + CM.WIN + 1000}\t{r.gene}\n')
    f.write_text('\n'.join(genes) + '\n')
    print(f'{len(shared)} shared genes + {len(new)} replacements', flush=True)
    return genes


def run_draws(n_draw):
    select_genes()
    with contextlib.redirect_stdout(io.StringIO()):
        I = CM.load_point_estimate_inputs(gene_list=str(OUT / 'genes.txt'), regions=str(OUT / 'regions.bed'))
    order, keep = I['order'], I['keep']
    N = len(order)
    A, T, Va, Vt, _ = summaries_from_point_estimates(I['pL'], I['pR'], I['pT'], I['eff_lib'],
                                                     I['YL'], I['YR'], I['YT'])
    A, T, Va, Vt = (x[:, keep] for x in (A, T, Va, Vt))
    pL, pR = I['pL'][:, keep], I['pR'][:, keep]
    zero = (pL < 0.5) ^ (pR < 0.5)
    Va_arm = {'keep': Va, 'drop': np.where(zero, 0.0, Va)}
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
    design = pd.DataFrame(dict(gene=genes, n_tested_variants=[len(tested_idx[g]) for g in genes],
                               n_allelic_keep=(Va > EPS).sum(1), n_allelic_drop=(Va_arm['drop'] > EPS).sum(1),
                               n_zero_haplotype=zero.sum(1),
                               median_allele_resolved_reads=np.median(pL + pR, axis=1)))
    design.to_csv(OUT / 'gene_design.tsv', sep='\t', index=False)
    print(f'{len(genes)} genes, {N} donors, {len(tested):,} tested gene-variant pairs; '
          f'zero-haplotype pairs {int(zero.sum()):,}; covariates {C.shape[1]} RNA-tied + '
          f'{G.shape[1]} genotype-tied', flush=True)

    rng = np.random.RandomState(SEED)
    perms = np.stack([rng.permutation(N) for _ in range(N_STREAM)])
    flips = rng.randint(0, 2, size=(N_STREAM, N)) * 2 - 1
    old = np.load(OLD / 'permutations.npz')
    if not (np.array_equal(old['perms'], perms) and np.array_equal(old['flips'], flips)):
        raise SystemExit('permutation stream differs from protein_coding_null_store')
    ddir = OUT / 'draws'; ddir.mkdir(exist_ok=True)
    scratch = OUT / 'scratch'; scratch.mkdir(exist_ok=True)

    def draw(p, arm):
        prm, f = perms[p], flips[p].astype(float)
        mk = lambda M, sgn=None: pd.DataFrame(M[:, prm] * (sgn[None, :] if sgn is not None else 1.0),
                                              index=genes, columns=order)
        cov = pd.DataFrame(C[prm], index=order, columns=I['cov_df'].columns)
        for q in scratch.glob('*'):
            q.unlink()
        with contextlib.redirect_stdout(io.StringIO()):
            map_nominal(gdf, vdf[['chrom', 'pos']], mk(A, f), mk(T), mk(Va_arm[arm]), mk(Vt), gp,
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

    # ---- gate --------------------------------------------------------------
    first = {arm: draw(0, arm) for arm in ARMS}
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
        for arm in ARMS:
            row = first[arm][(first[arm].phenotype_id == g) & (first[arm].variant_id == str(vdf.index[j]))]
            fc = fit_channels(A[k][prm0] * f0, s, Va_arm[arm][k][prm0], T[k][prm0], gh, Vt[k][prm0], Cg)
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
        for arm in ARMS:
            fo = ddir / f'{arm}_{p:03d}.parquet'
            if fo.exists():
                continue
            df = first[arm] if p == 0 else draw(p, arm)
            df.to_parquet(fo.with_suffix('.tmp'), compression='zstd', index=False)
            fo.with_suffix('.tmp').rename(fo)
        if p % 10 == 9 or p == n_draw - 1:
            print(f'  draw {p + 1}/{n_draw}  ({(time.time() - t0) / 60:.1f} min)', flush=True)
    shutil.rmtree(scratch, ignore_errors=True)


def rates_by_gene(files, genes, col, gene_filter=None):
    gix = {g: i for i, g in enumerate(genes)}
    K = {al: np.zeros(len(genes)) for al in ALPHAS}
    n = np.zeros(len(genes))
    for fp in files:
        d = pd.read_parquet(fp, columns=['phenotype_id', col])
        if gene_filter is not None:
            d = d[d.phenotype_id.isin(gene_filter)]
        d = d[np.isfinite(d[col])]
        gi = d.phenotype_id.map(gix).values
        n += np.bincount(gi, minlength=len(genes))
        for al in ALPHAS:
            K[al] += np.bincount(gi, weights=(d[col].values < al), minlength=len(genes))
    return K, n


def summarize():
    genes = (OUT / 'genes.txt').read_text().split()
    design = pd.read_csv(OUT / 'gene_design.tsv', sep='\t').set_index('gene').loc[genes]
    sel = pd.read_csv(OUT / 'gene_selection.tsv', sep='\t').set_index('gene').loc[genes]
    files = {arm: sorted((OUT / 'draws').glob(f'{arm}_*.parquet')) for arm in ARMS}
    n_draw = min(len(v) for v in files.values())
    files = {arm: v[:n_draw] for arm, v in files.items()}
    brng = np.random.default_rng(np.random.SeedSequence(SEED).spawn(5)[4])
    bidx = brng.integers(0, len(genes), size=(N_BOOT, len(genes)))
    res = dict(n_draw=n_draw, n_genes=len(genes), rates={}, drop_minus_keep={}, by_coverage={},
               gene_set=dict(shared=int((sel.source == 'shared').sum()),
                             zero_haplotype_pairs=int(design.n_zero_haplotype.sum()),
                             allelic_pairs_keep=int(design.n_allelic_keep.sum()),
                             allelic_pairs_drop=int(design.n_allelic_drop.sum())))
    cov_bin = pd.cut(design.median_allele_resolved_reads, [-1, 30, 100, 700, 3000, np.inf],
                     labels=['<30', '30-100', '100-700', '700-3000', '>=3000'])
    KN = {}
    for arm in ARMS:
        for ch, col in CHANNELS.items():
            K, n = rates_by_gene(files[arm], genes, col)
            KN[(arm, ch)] = (K, n)
            res['rates'][f'{arm} {ch}'] = {}
            for al in ALPHAS:
                b = K[al][bidx].sum(1) / n[bidx].sum(1)
                res['rates'][f'{arm} {ch}'][str(al)] = dict(rate=float(K[al].sum() / n.sum()),
                                                            lo=float(np.quantile(b, .025)),
                                                            hi=float(np.quantile(b, .975)))
            res['rates'][f'{arm} {ch}']['n_tests'] = int(n.sum())
            res['by_coverage'][f'{arm} {ch}'] = {
                str(cb): {str(al): float(K[al][(cov_bin == cb).values].sum()
                                         / max(n[(cov_bin == cb).values].sum(), 1)) for al in ALPHAS}
                for cb in cov_bin.cat.categories if (cov_bin == cb).any()}
    for ch in CHANNELS:
        (Kk, nk), (Kd, nd) = KN[('keep', ch)], KN[('drop', ch)]
        res['drop_minus_keep'][ch] = {}
        for al in ALPHAS:
            dd = Kd[al][bidx].sum(1) / nd[bidx].sum(1) - Kk[al][bidx].sum(1) / nk[bidx].sum(1)
            res['drop_minus_keep'][ch][str(al)] = dict(
                diff=float(Kd[al].sum() / nd.sum() - Kk[al].sum() / nk.sum()),
                lo=float(np.quantile(dd, .025)), hi=float(np.quantile(dd, .975)))
    # pre-correction store on the shared genes, same permutations, same tested variants
    shared = list(sel.index[sel.source == 'shared'])
    old_files = sorted((OLD / 'draws').glob('hapmix_*.parquet'))[:n_draw]
    old_tested = pd.read_parquet(OLD / 'draws' / 'mixqtl_000.parquet', columns=['gene', 'variant'])
    res['shared_genes'] = {}
    for ch, col in CHANNELS.items():
        gix = {g: i for i, g in enumerate(shared)}
        K = {al: np.zeros(len(shared)) for al in ALPHAS}; n = np.zeros(len(shared))
        for fp in old_files:
            d = pd.read_parquet(fp, columns=['phenotype_id', 'variant_id', col])
            d = d[d.phenotype_id.isin(gix)].merge(
                old_tested.rename(columns={'gene': 'phenotype_id', 'variant': 'variant_id'}),
                on=['phenotype_id', 'variant_id'], how='inner')
            d = d[np.isfinite(d[col])]
            gi = d.phenotype_id.map(gix).values
            n += np.bincount(gi, minlength=len(shared))
            for al in ALPHAS:
                K[al] += np.bincount(gi, weights=(d[col].values < al), minlength=len(shared))
        row = {'pre-correction': {str(al): float(K[al].sum() / n.sum()) for al in ALPHAS}}
        for arm in ARMS:
            Ka, na = KN[(arm, ch)]
            m = np.isin(genes, shared)
            row[arm] = {str(al): float(Ka[al][m].sum() / na[m].sum()) for al in ALPHAS}
        res['shared_genes'][ch] = row
    (OUT / 'summary.json').write_text(json.dumps(res, indent=1))
    for k, v in res['rates'].items():
        print(f'{k:18s} ' + '  '.join(f'{al}: {v[al]["rate"]:.4f} [{v[al]["lo"]:.4f}, {v[al]["hi"]:.4f}]'
                                      for al in map(str, ALPHAS)))
    for ch, v in res['drop_minus_keep'].items():
        print(f'drop - keep {ch:9s} ' + '  '.join(
            f'{al}: {v[al]["diff"]:+.4f} [{v[al]["lo"]:+.4f}, {v[al]["hi"]:+.4f}]' for al in map(str, ALPHAS)))
    for ch, v in res['shared_genes'].items():
        print(f'shared {len(shared)} genes, {ch:9s} ' + '  '.join(
            f'{a}: ' + '/'.join(f'{v[a][al]:.4f}' for al in map(str, ALPHAS)) for a in v))


def main():
    OUT.mkdir(exist_ok=True)
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    n_draw = int(args[0]) if args else 200
    if '--summarize-only' not in sys.argv:
        run_draws(n_draw)
    summarize()
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()

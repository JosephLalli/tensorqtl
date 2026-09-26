"""Unit, 1/v and 1/(v+1) weighting on the permutation null, for any gene set.

Generalizes hybrid_weights_null.py (fixed to the corrected store's 100 genes)
so the same tables can be made for targeted gene sets, e.g. genes whose
expression depends strongly on the reference, or genes whose top eQTL hits
have large effects with large standard errors.

Pipeline as corrected_null_store: Salmon point-estimate values, Gibbs draws
for variance, log2 CPM on edgeR effective library sizes, the new covariates
with genotype PCs held with the genotypes, default mode, allelic channel
through the origin, zero-haplotype pairs excluded from the allelic channel,
records_signflip permutations from the stored stream (RandomState(42), the
first n_draw of 1,000). Genes outside the calibration gene filter, without
Gibbs draws or without a unique position are dropped and listed.

Configs (both channels):
  gibbs     1/v, the shipped weighting
  unit      weight 1 for every included pair
  plus_one  1/(v + 1), included pairs only
GATE: draw 0's allelic and total slopes equal
null_permutation_instrument.fit_channels within 1e-3 se.

Usage:
  gene_set_weighting_null.py --genes=FILE --out=DIR --prepare
  gene_set_weighting_null.py --out=DIR --config=gibbs [n_draw=200]
  gene_set_weighting_null.py --out=DIR --summarize
Configs write disjoint files and can run as parallel processes.
"""
import contextlib
import io
import json
import shutil
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

import compare_mixqtl_replication as CM
import corrected_null_store as CNS
from null_permutation_instrument import fit_channels
from tensorqtl.hapmixqtl import map_nominal, summaries_from_point_estimates

D = CNS.D
CACHE = D / 'cache' / 'gibbs_56b63c3b37ed5df8'
SEED, N_STREAM, N_BOOT, EPS = 42, 1000, 2000, 1e-12
ALPHAS = (0.05, 0.01, 0.001)
CONFIGS = ('unit', 'gibbs', 'plus_one')
CHANNELS = {'allelic': ('slope_a', 'slope_a_se', 'pval_a'), 'total': ('slope_t', 'slope_t_se', 'pval_t'),
            'combined': ('slope', 'slope_se', 'pval_nominal')}
COLS = ['phenotype_id', 'variant_id'] + [c for v in CHANNELS.values() for c in v]


def prepare(genes_file, out):
    """Keep genes that can be run; write genes.txt, regions.bed, excluded.tsv."""
    if (out / 'genes.txt').exists():
        return (out / 'genes.txt').read_text().split()
    want = [l.strip() for l in open(genes_file) if l.strip()]
    cal = set((CACHE / 'point_estimates' / 'edger' / 'calibration_genes.txt').read_text().split())
    cache = set((CACHE / 'genes.txt').read_text().split())
    gp = pd.read_csv(D / 'annot' / 'genes.tsv', sep='\t', header=None, dtype={1: str},
                     names=['gene', 'chr', 'start', 'end', 'pos'])
    gp = gp[~gp.gene.duplicated(keep=False)].set_index('gene')
    why = {}
    for g in want:
        if g not in cal:
            why[g] = 'outside the calibration gene filter'
        elif g not in cache:
            why[g] = 'no Gibbs draws'
        elif g not in gp.index:
            why[g] = 'no unique position in annot/genes.tsv'
    genes = sorted(g for g in dict.fromkeys(want) if g not in why)
    pd.Series(why, name='reason').rename_axis('gene').to_csv(out / 'excluded.tsv', sep='\t')
    t = gp.loc[genes].reset_index()
    with open(out / 'regions.bed', 'w') as fh:
        for r in t.itertuples():
            lo = max(0, min(r.start, r.pos) - CM.WIN - 1000)
            fh.write(f'{r.chr}\t{lo}\t{max(r.end, r.pos) + CM.WIN + 1000}\t{r.gene}\n')
    (out / 'genes.txt').write_text('\n'.join(genes) + '\n')
    print(f'{len(genes)} of {len(set(want))} genes kept; excluded: '
          f'{pd.Series(why).value_counts().to_dict() if why else {}}', flush=True)
    return genes


def variances(config, Va, Vt):
    if config == 'gibbs':
        return Va, Vt
    if config == 'unit':
        return np.where(Va > EPS, 1.0, 0.0), np.ones_like(Vt)
    if config == 'plus_one':
        return np.where(Va > EPS, Va + 1.0, 0.0), Vt + 1.0
    raise SystemExit(f'unknown config {config}')


def run(out, config, n_draw):
    genes = (out / 'genes.txt').read_text().split()
    with contextlib.redirect_stdout(io.StringIO()):
        I = CM.load_point_estimate_inputs(gene_list=str(out / 'genes.txt'), regions=str(out / 'regions.bed'))
    order, keep = I['order'], I['keep']
    N = len(order)
    A, T, Va, Vt, _ = summaries_from_point_estimates(I['pL'], I['pR'], I['pT'], I['eff_lib'],
                                                     I['YL'], I['YR'], I['YT'])
    A, T, Va, Vt = (x[:, keep] for x in (A, T, Va, Vt))
    pL, pR = I['pL'][:, keep], I['pR'][:, keep]
    Va = np.where((pL < 0.5) ^ (pR < 0.5), 0.0, Va)
    Va, Vt = variances(config, Va, Vt)
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
    af = I['dos'].mean(1) / 2
    pd.DataFrame(dict(variant_id=vdf.index.astype(str), maf=np.minimum(af, 1 - af))).to_csv(
        out / 'variant_maf.tsv.gz', sep='\t', index=False)
    rng = np.random.RandomState(SEED)
    perms = np.stack([rng.permutation(N) for _ in range(N_STREAM)])
    flips = rng.randint(0, 2, size=(N_STREAM, N)) * 2 - 1
    ref = np.load(CNS.OLD / 'permutations.npz')
    if not (np.array_equal(ref['perms'], perms) and np.array_equal(ref['flips'], flips)):
        raise SystemExit('permutation stream differs from the stored one')
    ddir = out / 'draws'; ddir.mkdir(exist_ok=True)
    scratch = out / f'scratch_{config}'; scratch.mkdir(exist_ok=True)

    def draw(p):
        prm, f = perms[p], flips[p].astype(float)
        mk = lambda M, sgn=None: pd.DataFrame(M[:, prm] * (sgn[None, :] if sgn is not None else 1.0),
                                              index=genes, columns=order)
        cov = pd.DataFrame(C[prm], index=order, columns=I['cov_df'].columns)
        for q in scratch.glob('*'):
            q.unlink()
        with contextlib.redirect_stdout(io.StringIO()):
            map_nominal(gdf, vdf[['chrom', 'pos']], mk(A, f), mk(T), mk(Va), mk(Vt), gp,
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
        fc = fit_channels(A[k][prm0] * f0, s, Va[k][prm0], T[k][prm0], gh, Vt[k][prm0], Cg)
        if row.empty or fc is None:
            continue
        r = row.iloc[0]
        worst = max(worst, abs(float(r.slope_a) - fc['ba']) / float(r.slope_a_se),
                    abs(float(r.slope_t) - fc['bt']) / float(r.slope_t_se))
    print(f'{config}: gate, draw-0 slopes vs reference fit, max |diff| / se = {worst:.2e}', flush=True)
    if not worst < 1e-3:
        raise SystemExit('GATE FAILED')
    t0 = time.time()
    for p in range(n_draw):
        fo = ddir / f'{config}_{p:03d}.parquet'
        if fo.exists():
            continue
        df = first if p == 0 else draw(p)
        df.to_parquet(fo.with_suffix('.tmp'), compression='zstd', index=False)
        fo.with_suffix('.tmp').rename(fo)
        if p % 50 == 49:
            print(f'  {config} draw {p + 1}/{n_draw}  ({(time.time() - t0) / 60:.1f} min)', flush=True)
    shutil.rmtree(scratch, ignore_errors=True)


def _read(f):
    d = pd.read_parquet(f, columns=COLS)
    return {c: d[c].values.astype(float) for c in COLS[2:]}, d[['phenotype_id', 'variant_id']]


def summarize(out):
    genes = (out / 'genes.txt').read_text().split()
    gix = {g: i for i, g in enumerate(genes)}
    maf = pd.read_csv(out / 'variant_maf.tsv.gz', sep='\t', dtype={'variant_id': str}).set_index('variant_id').maf
    files = {c: sorted((out / 'draws').glob(f'{c}_*.parquet')) for c in CONFIGS}
    n_draw = min(len(v) for v in files.values())
    brng = np.random.default_rng(np.random.SeedSequence(SEED).spawn(8)[7])
    bidx = brng.integers(0, len(genes), size=(N_BOOT, len(genes)))
    res = dict(n_genes=len(genes), n_draw=n_draw, mean_se={}, median_se={}, stated_over_true={}, rates={},
               mean_se_by_maf={})
    ref_ids = None
    with Pool(24) as pool:
        for cfg in CONFIGS:
            acc = {ch: None for ch in CHANNELS}
            ses = {ch: [] for ch in CHANNELS}
            K = {ch: {al: np.zeros(len(genes)) for al in ALPHAS} for ch in CHANNELS}
            Nn = {ch: np.zeros(len(genes)) for ch in CHANNELS}
            for vals, ids in pool.imap(_read, files[cfg][:n_draw], chunksize=4):
                if ref_ids is None:
                    ref_ids = ids
                    gi = ids.phenotype_id.map(gix).values
                    band = np.digitize(maf.reindex(ids.variant_id.values).values, [0.1, 0.2])
                elif not ids.equals(ref_ids):
                    raise SystemExit('row order differs between draw files')
                for ch, (bc, sc, pc) in CHANNELS.items():
                    b, s, p = vals[bc], vals[sc], vals[pc]
                    ok = np.isfinite(b) & np.isfinite(s)
                    a_ = np.stack([ok, np.where(ok, b, 0), np.where(ok, b * b, 0), np.where(ok, s, 0), np.where(ok, s * s, 0)])
                    acc[ch] = a_ if acc[ch] is None else acc[ch] + a_
                    ses[ch].append(s[np.isfinite(s)].astype(np.float32))
                    fp = np.isfinite(p)
                    Nn[ch] += np.bincount(gi[fp], minlength=len(genes))
                    for al in ALPHAS:
                        K[ch][al] += np.bincount(gi[fp], weights=(p[fp] < al), minlength=len(genes))
            for ch in CHANNELS:
                n, sb, sb2, ss, ss2 = acc[ch]
                allse = np.concatenate(ses[ch])
                with np.errstate(invalid='ignore', divide='ignore'):
                    real = np.sqrt((sb2 - sb ** 2 / n) / (n - 1)); rms = np.sqrt(ss2 / n); meanse = ss / n
                ok = (n > n_draw // 2) & (real > 0)
                res['mean_se'].setdefault(ch, {})[cfg] = float(allse.mean())
                res['median_se'].setdefault(ch, {})[cfg] = float(np.median(allse))
                res['stated_over_true'].setdefault(ch, {})[cfg] = float(np.median(rms[ok] / real[ok]))
                res['mean_se_by_maf'].setdefault(ch, {})[cfg] = {
                    nm: float(np.mean(meanse[ok & (band == k)])) for k, nm in enumerate(['0.05-0.10', '0.10-0.20', '0.20-0.50'])
                    if (ok & (band == k)).any()}
                res['rates'].setdefault(ch, {})[cfg] = {str(al): dict(
                    rate=float(K[ch][al].sum() / Nn[ch].sum()),
                    lo=float(np.quantile(K[ch][al][bidx].sum(1) / Nn[ch][bidx].sum(1), .025)),
                    hi=float(np.quantile(K[ch][al][bidx].sum(1) / Nn[ch][bidx].sum(1), .975))) for al in ALPHAS}
    (out / 'summary.json').write_text(json.dumps(res, indent=1))
    for tab in ('mean_se', 'median_se', 'stated_over_true'):
        print(tab)
        print(pd.DataFrame(res[tab]).T[list(CONFIGS)].round(3).to_string())
    print('rejection rate at 0.05 / 0.01 / 0.001')
    for ch in CHANNELS:
        print(f'  {ch:9s} ' + '   '.join(f'{c}: ' + '/'.join(f'{res["rates"][ch][c][a]["rate"]:.4f}' for a in map(str, ALPHAS))
                                         for c in CONFIGS))


def main():
    opt = dict(a.split('=', 1) for a in sys.argv[1:] if a.startswith('--') and '=' in a)
    out = Path(opt['--out']); out.mkdir(parents=True, exist_ok=True)
    if '--summarize' in sys.argv:
        summarize(out)
        return
    if '--prepare' in sys.argv:            # once, before parallel configs
        prepare(opt['--genes'], out)
        return
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    n_draw = int(args[0]) if args else 200
    if not (out / 'genes.txt').exists():
        raise SystemExit('run --prepare first')
    run(out, opt['--config'], n_draw)


if __name__ == '__main__':
    main()

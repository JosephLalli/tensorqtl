"""A stored, per-variant null for hapmixQTL and mixQTL on 100 expressed
protein-coding genes, 200 permutations, every draw kept.

WHY. Every calibration statement so far rests on 46 or 59 genes chosen for
allele-resolved coverage (67% of the 46 had >= 700 reads, against 22%
transcriptome-wide), and the variant-level null on only 30 permutations kept
as running sums (a 13% floor on a per-variant sd ratio). This builds a
representative instrument once and stores it, so later analyses read the
draws instead of recomputing them. RASQUAL is not run (user decision
2026-09-25); its 30 stored draws stay where they are.

GENE SET. Protein-coding genes, defined from the RefSeq annotation as genes
with at least one curated NM_ transcript in annot/tx2gene.tsv, on autosomes
chr1-chr22, present in the Gibbs cache and in annot/genes.tsv, that pass
edgeR's filterByExpr with its defaults and no design: counts are the
posterior-mean total counts over the Gibbs draws, library size is each
sample's column sum over all cached genes (edgeR's default), the CPM cutoff is
min.count 10 over the median library size in millions, and a gene is kept if
its CPM reaches the cutoff in at least large.n + (n - large.n) * min.prop =
10 + 82 * 0.7 = 67.4 of the 92 samples and its total count is at least 15
(edgeR:::filterByExpr.default, read 2026-09-25). 100 genes are drawn uniformly
without replacement from the eligible set with a SeedSequence(42) child
stream. No allelic-informativeness filter is applied: the number of donors
with allelic information is recorded per gene instead, so the set represents
what the expression filter admits.

NULL. The shipped definition since 2026-09-25, perm_scheme='records_signflip':
each donor's record (allelic log ratio a, log total t, both Gibbs variances,
covariate row; for mixQTL the posterior-mean haplotype and total counts) moves
as one unit against fixed genotypes, and each permuted record's haplotype
labels L/R are swapped with probability one half (a -> -a; for mixQTL the two
haplotype counts are exchanged). The 200 permutations come from
RandomState(42) and the swap signs are drawn from the same stream right after
them, the order map_cis uses; the stream is always drawn for 1,000
permutations and the first n_draw are used, so a short run, the 200-draw run
and any later extension store identical draws for the same index.

PER DRAW, STORED (draws/hapmix_NNN.parquet, draws/mixqtl_NNN.parquet): for
every variant map_nominal reports in each gene's cis window, hapmixQTL's
combined, allelic and total slope, standard error and nominal p (its own
F references); for every tested variant (outside the gene body, MAF >= 0.05),
mixQTL's meta, allelic and total beta and se (natural log, converted from
its log2), with p from its published normal reference computed at summary
time. A draw whose two files exist is skipped, so the run resumes.

GATE, before any draw is stored: at each gene's variant with the most
heterozygotes, draw 0's map_nominal allelic slope must equal the shipped
permutation routine (_record_permutation_channel with flip_t) and its total
slope the verified reference fit (null_permutation_instrument.fit_channels)
on the same permuted, swapped records, to within 1e-3 of the slope's standard
error (map_nominal runs in float32).

SUMMARY (summary.json): pooled nominal rejection rates at 0.05 / 0.01 / 0.001
per arm and channel over tested variants, with gene-clustered bootstrap 95%
intervals (genes resampled with replacement), by coverage bin, and the
per-variant ratio of mean reported se to realized sd across draws.

Usage: protein_coding_null_store.py [n_draw=200] [--summarize-only]
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
import torch
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import compare_mixqtl_replication as CM                          # noqa: E402
import tensorqtl.mixqtl_replication as MX                        # noqa: E402
from null_permutation_instrument import fit_channels             # noqa: E402
from tensorqtl.hapmixqtl import (WeightedResidualizer,            # noqa: E402
                                 _record_permutation_channel, map_nominal)

D = Path('/mnt/ssd/lalli/brainvar_hapmix_deploy')
CACHE = D / 'cache' / 'gibbs_56b63c3b37ed5df8'
OUT = D / 'protein_coding_null_store_20260925'
SEED, N_GENES, EPS = 42, 100, 1e-12
N_STREAM = 1000                    # permutations and swaps are drawn for this many, always
N_BOOT = 2000
ALPHAS = (0.05, 0.01, 0.001)
LN2 = np.log(2.0)
AUTOSOMES = {f'chr{i}' for i in range(1, 23)}
HAP_COLS = ['phenotype_id', 'variant_id', 'af', 'pval_nominal', 'slope', 'slope_se',
            'pval_a', 'slope_a', 'slope_a_se', 'pval_t', 'slope_t', 'slope_t_se']


# ---------------------------------------------------------------- gene set --
def select_genes():
    sel_file = OUT / 'genes_pc100.txt'
    if sel_file.exists():
        return [l.strip() for l in open(sel_file) if l.strip()]
    genes_all = open(CACHE / 'genes.txt').read().split()
    YT = np.load(CACHE / 'YT.npy', mmap_mode='r')
    m = np.empty(YT.shape[:2])
    for s in range(0, YT.shape[0], 2000):
        m[s:s + 2000] = np.asarray(YT[s:s + 2000]).mean(2)           # posterior-mean totals
    lib = m.sum(0)                                                   # edgeR default lib.size
    n = m.shape[1]
    min_ss = 10 + (n - 10) * 0.7 if n > 10 else n
    cutoff = 10 / np.median(lib) * 1e6
    cpm = m / lib[None, :] * 1e6
    keep_expr = ((cpm >= cutoff).sum(1) >= min_ss - 1e-14) & (m.sum(1) >= 15 - 1e-14)
    tx = pd.read_csv(D / 'annot' / 'tx2gene.tsv', sep='\t', header=None, names=['tx', 'gene'])
    pc = set(tx.loc[tx.tx.str.startswith('NM_'), 'gene'])
    gp = pd.read_csv(D / 'annot' / 'genes.tsv', sep='\t', header=None, dtype={1: str},
                     names=['gene', 'chr', 'start', 'end', 'pos'])
    gp = gp[~gp.gene.duplicated(keep=False)].set_index('gene')
    eligible = [g for g, k in zip(genes_all, keep_expr)
                if k and g in pc and g in gp.index and str(gp.loc[g, 'chr']) in AUTOSOMES]
    rng = np.random.default_rng(np.random.SeedSequence(SEED).spawn(2)[1])
    chosen = sorted(rng.choice(sorted(eligible), size=N_GENES, replace=False).tolist())
    gi = {g: i for i, g in enumerate(genes_all)}
    YL = np.load(CACHE / 'YL.npy', mmap_mode='r'); YR = np.load(CACHE / 'YR.npy', mmap_mode='r')
    rows = []
    for g in chosen:
        i = gi[g]
        ase = np.asarray(YL[i]).mean(1) + np.asarray(YR[i]).mean(1)
        rows.append(dict(gene=g, chr=gp.loc[g, 'chr'], start=int(gp.loc[g, 'start']),
                         end=int(gp.loc[g, 'end']), pos=int(gp.loc[g, 'pos']),
                         median_cpm=float(np.median(cpm[i])),
                         frac_samples_ge_cutoff=float((cpm[i] >= cutoff).mean()),
                         median_total_reads=float(np.median(m[i])),
                         median_allele_resolved_reads=float(np.median(ase)),
                         n_informative_donors=int((ase > 0).sum())))
    t = pd.DataFrame(rows)
    t.to_csv(OUT / 'gene_selection.tsv', sep='\t', index=False)
    with open(OUT / 'regions.bed', 'w') as fh:
        for r in t.itertuples():
            lo = max(0, min(r.start, r.pos) - CM.WIN - 1000)
            fh.write(f'{r.chr}\t{lo}\t{max(r.end, r.pos) + CM.WIN + 1000}\t{r.gene}\n')
    (OUT / 'selection.json').write_text(json.dumps(dict(
        n_cache_genes=len(genes_all), n_samples=int(n), median_lib_size=float(np.median(lib)),
        cpm_cutoff=float(cutoff), min_samples=float(min_ss), n_pass_filterByExpr=int(keep_expr.sum()),
        n_protein_coding_NM=len(pc), n_eligible=len(eligible), n_chosen=len(chosen),
        overlap_with_59=sorted(set(chosen) & {l.strip() for l in open(D / 'genes_59_stratified_20260923.txt')})),
        indent=2))
    sel_file.write_text('\n'.join(chosen) + '\n')
    return chosen


# -------------------------------------------------------------------- draws --
def run_draws(n_draw):
    genes = select_genes()
    with contextlib.redirect_stdout(io.StringIO()):
        I = CM.load_inputs(gene_list=str(OUT / 'genes_pc100.txt'), regions=str(OUT / 'regions.bed'))
    order, keep = I['order'], I['keep']
    N = len(order)
    import run_hapmixqtl_from_salmon as H
    with contextlib.redirect_stdout(io.StringIO()):
        A, T, Va, Vt, _ = H.compute_summaries_from_gibbs(
            I['YL'][:, keep], I['YR'][:, keep], yT=I['YT'][:, keep])
    Y1, Y2, YT = MX.summaries_from_gibbs_posterior_mean(I['YL'], I['YR'], I['YT'])
    Y1, Y2, YT = Y1[:, keep], Y2[:, keep], YT[:, keep]
    genes = list(I['genes'])
    gp = I['gp'].loc[genes][['chr', 'pos']]
    C = I['cov_df'].values
    vdf = I['vdf']
    gdf = pd.DataFrame(I['dos'], index=vdf.index, columns=order)
    xLdf = pd.DataFrame(I['xL'], index=vdf.index, columns=order)
    xRdf = pd.DataFrame(I['xR'], index=vdf.index, columns=order)
    tested_idx = {g: I['idx'][CM.gene_variant_index(I, g)] for g in genes}
    info = pd.DataFrame(dict(gene=genes, n_allelic_donors=(Va > EPS).sum(1),
                             n_tested_variants=[len(tested_idx[g]) for g in genes]))
    info.to_csv(OUT / 'gene_design.tsv', sep='\t', index=False)
    print(f'{len(genes)} genes, {N} donors, {len(vdf)} variants loaded, '
          f'{info.n_tested_variants.sum()} tested gene-variant pairs', flush=True)

    # A FIXED stream of N_STREAM draws, whatever n_draw is, so a short run and
    # a long one (or a later extension) store identical draws for the same index.
    rng = np.random.RandomState(SEED)
    perms = np.stack([rng.permutation(N) for _ in range(N_STREAM)])
    flips = rng.randint(0, 2, size=(N_STREAM, N)) * 2 - 1           # after the indices, as map_cis
    pf = OUT / 'permutations.npz'
    if pf.exists():
        old = np.load(pf)
        if not (np.array_equal(old['perms'], perms) and np.array_equal(old['flips'], flips)):
            raise SystemExit('permutation stream differs from the stored one; refusing to mix draws')
    else:
        np.savez_compressed(pf, perms=perms, flips=flips)
    if n_draw > N_STREAM:
        raise SystemExit(f'n_draw {n_draw} exceeds the fixed stream of {N_STREAM}')
    ddir = OUT / 'draws'; ddir.mkdir(exist_ok=True)
    scratch = OUT / 'scratch'; scratch.mkdir(exist_ok=True)

    def hapmix_draw(p):
        prm, f = perms[p], flips[p].astype(float)
        mk = lambda M, sgn=None: pd.DataFrame(M[:, prm] * (sgn[None, :] if sgn is not None else 1.0),
                                              index=genes, columns=order)
        cov = pd.DataFrame(C[prm], index=order, columns=I['cov_df'].columns)
        for q in scratch.glob('*'):
            q.unlink()
        with contextlib.redirect_stdout(io.StringIO()):
            map_nominal(gdf, vdf[['chrom', 'pos']], mk(A, f), mk(T), mk(Va), mk(Vt), gp,
                        xL_df=xLdf, xR_df=xRdf, prefix='n', covariates_df=cov,
                        window=CM.WIN, output_dir=str(scratch), verbose=False,
                        ase_covariates_df=None)
        parts = [pd.read_parquet(q, columns=HAP_COLS) for q in sorted(scratch.glob('n*.parquet'))]
        df = pd.concat(parts, ignore_index=True)
        for c in HAP_COLS[2:]:
            df[c] = df[c].astype(np.float32)
        return df

    # ---- gate on draw 0 ---------------------------------------------------
    df0 = hapmix_draw(0)
    prm0, f0 = perms[0], flips[0].astype(float)
    worst = 0.0
    vpos = {v: i for i, v in enumerate(vdf.index)}
    for k, g in enumerate(genes):
        cand = tested_idx[g]
        if not len(cand):
            continue
        s_all = (I['xL'][cand] - I['xR'][cand]).astype(float)
        j = cand[int(np.argmax((s_all != 0).sum(1)))]
        s = (I['xL'][j] - I['xR'][j]).astype(float)
        gh = I['dos'][j].astype(float) / 2.0
        row = df0[(df0.phenotype_id == g) & (df0.variant_id == vdf.index[j])]
        fc = fit_channels(A[k][prm0] * f0, s, Va[k][prm0], T[k][prm0], gh, Vt[k][prm0], C[prm0])
        if row.empty or fc is None:
            continue
        ok = (Va[k] > EPS) & np.isfinite(A[k])
        sw = torch.tensor(np.where(ok, 1 / np.sqrt(np.where(ok, Va[k], 1.0)), 0.0))
        xy, xx, _ = _record_permutation_channel(
            torch.tensor(s[None, :]), torch.tensor(np.where(ok, A[k], 0.0)), sw,
            WeightedResidualizer(None, sw, intercept=False),
            torch.tensor(perms[:1], dtype=torch.long), flip_t=torch.tensor(flips[:1].astype(float)))
        b_routine = float(xy[0, 0] / xx[0, 0])
        r = row.iloc[0]
        worst = max(worst,
                    abs(float(r.slope_a) - b_routine) / float(r.slope_a_se),
                    abs(float(r.slope_a) - fc['ba']) / float(r.slope_a_se),
                    abs(float(r.slope_t) - fc['bt']) / float(r.slope_t_se))
    print(f'gate: draw-0 slopes vs shipped routine / reference fit, max |diff| / se = {worst:.2e}', flush=True)
    if not worst < 1e-3:
        raise SystemExit(f'GATE FAILED: {worst:.2e}')

    t0 = time.time()
    for p in range(n_draw):
        fh, fm = ddir / f'hapmix_{p:03d}.parquet', ddir / f'mixqtl_{p:03d}.parquet'
        if fh.exists() and fm.exists():
            continue
        df = df0 if p == 0 else hapmix_draw(p)
        df.to_parquet(fh.with_suffix('.tmp'), compression='zstd', index=False)
        fh.with_suffix('.tmp').rename(fh)
        prm, f = perms[p], flips[p]
        y1p = np.where(f[None, :] > 0, Y1[:, prm], Y2[:, prm])       # swap haplotype counts
        y2p = np.where(f[None, :] > 0, Y2[:, prm], Y1[:, prm])
        rows = []
        for k, g in enumerate(genes):
            vsel = tested_idx[g]
            if not len(vsel):
                continue
            h1 = I['xL'][vsel].T.astype(float)
            h2 = I['xR'][vsel].T.astype(float)
            o = MX.mixqtl_scan(y1p[k], y2p[k], YT[k][prm], I['lib_size'][prm], h1, h2,
                               covariates=C[prm], **MX.PACKAGE_DEFAULT_CUTOFFS)
            rows.append(pd.DataFrame(dict(
                gene=g, variant=vdf.index[vsel].astype(str),
                **{f'{nm}_{q}': (np.asarray(o[key][q], float) * LN2).astype(np.float32)
                   for nm, key in (('meta', 'meta'), ('asc', 'asc'), ('trc', 'trc')) for q in ('beta', 'se')})))
        pd.concat(rows, ignore_index=True).to_parquet(fm.with_suffix('.tmp'), compression='zstd', index=False)
        fm.with_suffix('.tmp').rename(fm)
        el = time.time() - t0
        print(f'  draw {p + 1}/{n_draw}  ({el / 60:.1f} min, '
              f'{sum(q.stat().st_size for q in ddir.glob("*.parquet")) / 1e9:.2f} GB stored)', flush=True)
    shutil.rmtree(scratch, ignore_errors=True)


# ------------------------------------------------------------------ summary --
def summarize():
    ddir = OUT / 'draws'
    sel = pd.read_csv(OUT / 'gene_selection.tsv', sep='\t').set_index('gene')
    design = pd.read_csv(OUT / 'gene_design.tsv', sep='\t').set_index('gene')
    hfiles = sorted(ddir.glob('hapmix_*.parquet')); mfiles = sorted(ddir.glob('mixqtl_*.parquet'))
    n_draw = min(len(hfiles), len(mfiles))
    brng = np.random.default_rng(np.random.SeedSequence(SEED).spawn(3)[2])
    cov_bin = pd.cut(sel.median_allele_resolved_reads, [-1, 30, 100, 700, 3000, np.inf],
                     labels=['<30', '30-100', '100-700', '700-3000', '>=3000'])
    genes = list(sel.index)
    # rejection counts per gene: arm/channel -> alpha -> [k per gene], n per gene
    arms = {'hapmixQTL combined': ('h', 'pval_nominal'), 'hapmixQTL allelic': ('h', 'pval_a'),
            'hapmixQTL total': ('h', 'pval_t'), 'mixQTL meta (normal ref)': ('m', 'meta'),
            'mixQTL allelic (normal ref)': ('m', 'asc'), 'mixQTL total (normal ref)': ('m', 'trc')}
    K = {a: {al: np.zeros(len(genes)) for al in ALPHAS} for a in arms}
    Nn = {a: np.zeros(len(genes)) for a in arms}
    gix = {g: i for i, g in enumerate(genes)}
    units, acc = None, {}                                             # per-unit moments on a fixed unit order
    for p in range(n_draw):
        h = pd.read_parquet(hfiles[p]); m = pd.read_parquet(mfiles[p])
        tested = m[['gene', 'variant']].drop_duplicates().rename(
            columns={'gene': 'phenotype_id', 'variant': 'variant_id'})
        h = h.merge(tested, on=['phenotype_id', 'variant_id'], how='inner')
        h = h.set_index(['phenotype_id', 'variant_id'], drop=False)
        if units is None:
            units = h.index.copy()
        hu = h.reindex(units)
        for a, (src, col) in arms.items():
            if src == 'h':
                d_ = pd.DataFrame(dict(gene=h.phenotype_id.values, p=h[col].values))
            else:
                z = m[f'{col}_beta'] / m[f'{col}_se']
                d_ = pd.DataFrame(dict(gene=m.gene, p=2 * sps.norm.sf(np.abs(z))))
            d_ = d_[np.isfinite(d_.p)]
            gi_ = d_.gene.map(gix).values
            Nn[a] += np.bincount(gi_, minlength=len(genes))
            for al in ALPHAS:
                K[a][al] += np.bincount(gi_, weights=(d_.p.values < al), minlength=len(genes))
        for chan, b, s in (('combined', 'slope', 'slope_se'), ('allelic', 'slope_a', 'slope_a_se'),
                           ('total', 'slope_t', 'slope_t_se')):
            bb, ss = hu[b].values.astype(float), hu[s].values.astype(float)
            ok = np.isfinite(bb) & np.isfinite(ss) & (ss > 0)
            if chan not in acc:
                acc[chan] = np.zeros((4, len(units)))
            acc[chan][0] += ok
            acc[chan][1] += np.where(ok, bb, 0.0)
            acc[chan][2] += np.where(ok, bb * bb, 0.0)
            acc[chan][3] += np.where(ok, ss * ss, 0.0)
    res = dict(n_draw=n_draw, n_genes=len(genes), rates={}, by_coverage={}, se_ratio={})

    def boot(k, n):
        idx = brng.integers(0, len(k), size=(N_BOOT, len(k)))
        b = k[idx].sum(1) / np.maximum(n[idx].sum(1), 1)
        return float(k.sum() / n.sum()), float(np.quantile(b, .025)), float(np.quantile(b, .975))

    for a in arms:
        res['rates'][a] = {str(al): dict(zip(('rate', 'lo', 'hi'), boot(K[a][al], Nn[a]))) for al in ALPHAS}
        res['rates'][a]['n_tests'] = int(Nn[a].sum())
        res['by_coverage'][a] = {}
        for cb in cov_bin.cat.categories:
            msk = (cov_bin.reindex(genes) == cb).values
            if msk.sum() == 0:
                continue
            res['by_coverage'][a][cb] = dict(n_genes=int(msk.sum()), **{
                str(al): float(K[a][al][msk].sum() / max(Nn[a][msk].sum(), 1)) for al in ALPHAS})
    floor = 1 / np.sqrt(2 * max(n_draw - 1, 1))
    for chan, (n_, sb, sb2, ss2) in acc.items():
        use = n_ >= max(3, n_draw // 2)
        n_, sb, sb2, ss2 = n_[use], sb[use], sb2[use], ss2[use]
        var = (sb2 - sb ** 2 / n_) / (n_ - 1)
        pos = var > 0
        if not pos.any():
            res['se_ratio'][chan] = dict(n_units=0, note=f'too few draws ({n_draw}) for a per-unit sd')
            continue
        ratio = np.sqrt(ss2[pos] / n_[pos]) / np.sqrt(var[pos])
        res['se_ratio'][chan] = dict(n_units=int(pos.sum()), median=float(np.median(ratio)),
                                     q10=float(np.quantile(ratio, .1)), q90=float(np.quantile(ratio, .9)),
                                     per_unit_noise_floor=float(floor))
    res['gene_set'] = dict(n_allelic_donors_median=float(design.n_allelic_donors.median()),
                           coverage_bins={str(k): int(v) for k, v in cov_bin.value_counts().items()},
                           stored_gb=float(sum(q.stat().st_size for q in ddir.glob('*.parquet')) / 1e9))
    (OUT / 'summary.json').write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=1))


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

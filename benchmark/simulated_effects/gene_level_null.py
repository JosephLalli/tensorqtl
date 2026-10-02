"""Gene-level stored null: split's map_cis gene-level p over many all-null anchor datasets (task 2026-10-02).

Why: the benchmark's gene-level false-discovery figures rest on one anchor permutation and three replicates per
|beta|, too few to say whether Benjamini-Hochberg on pval_beta holds its rate on these genes. Each anchor dataset here
is 02's build_dataset at beta = 0 (every gene null, no thinning) under the record permutation and label swap of
replicate r, scanned by 03's run_cis for split with 03's seed for r: exactly the benchmark's anchor at r = 0, which
the run checks first (pval_perm and pval_beta equal the stored results/beta0.0/split/cis_rep000.parquet). The null
replicates are r = R0 + k, k < n, indices no benchmark dataset uses.

Output ROOT/gene_level_null: cis_rNNNN.parquet per replicate (phenotype_id, pval_perm, pval_beta; a replicate whose
file exists is not rerun) and summary.json: per threshold the share of gene-replicate units below it with its
gene-clustered interval, the per-replicate count of genes below it (mean, sd, 2.5% and 97.5% quantiles), the share of
replicates in which Benjamini-Hochberg at 5% calls any gene (every call is false here; with valid p it is at most
0.05), and where the benchmark's own anchor (r = 0) falls among the replicates.
Usage: gene_level_null.py [n=100]   (SIMULATED_EFFECTS_GENE_SET picks the gene set, as for every step)
"""
import json
import sys

import numpy as np
import pandas as pd

import common as C

MD, RA = C.module('02_make_datasets'), C.module('03_run_arms')
OUT = C.ROOT / 'gene_level_null'
R0 = 1000                      # first null replicate index; the benchmark's datasets use 0-2
ALPHAS = (0.05, 0.01, 0.001)
N_BOOT = 2000
BOOT_KEY = 40                  # spawn key used by no other script here (see 05b_native_arms.NATIVE_THIN_KEY's list)


def scan(I, R, tested, S, r):
    ds = MD.build_dataset(I, R, tested, 0.0, 1.0, r)
    return RA.run_cis(S, ds, 'split', RA.cis_seed(r))[['phenotype_id', 'pval_perm', 'pval_beta']]


def bh_any(p, q=0.05):
    p = np.sort(np.asarray(p, float))
    return bool((p <= q * np.arange(1, len(p) + 1) / len(p)).any())


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    I, R, tested = C.load()
    S = C.setup(I)
    OUT.mkdir(exist_ok=True)
    ref = pd.read_parquet(C.RESULTS / 'beta0.0' / 'split' / 'cis_rep000.parquet', columns=['phenotype_id', 'pval_perm', 'pval_beta'])
    got = scan(I, R, tested, S, 0)
    if not (got.phenotype_id.tolist() == ref.phenotype_id.tolist()
            and np.array_equal(got.pval_perm.values, ref.pval_perm.values)
            and np.allclose(got.pval_beta.values, ref.pval_beta.values, rtol=1e-6, atol=0)):
        raise SystemExit('r = 0 does not reproduce the benchmark anchor\'s split map_cis (results/beta0.0/split/cis_rep000.parquet)')
    print(f'check: r = 0 reproduces the benchmark anchor\'s split map_cis on {len(got)} genes', flush=True)
    for k in range(n):
        f = OUT / f'cis_r{R0 + k:04d}.parquet'
        if f.exists():
            continue
        C.write_atomic(f, lambda fh, df=scan(I, R, tested, S, R0 + k): df.to_parquet(fh, index=False))
        if k % 10 == 9:
            print(f'  replicate {k + 1}/{n}', flush=True)
    if n:
        summarize(ref, n)


def summarize(anchor, n):
    P = pd.concat([pd.read_parquet(OUT / f'cis_r{R0 + k:04d}.parquet').assign(rep=k) for k in range(n)])
    M = P.pivot(index='phenotype_id', columns='rep', values='pval_beta')   # genes x replicates
    genes = M.index
    rng = np.random.default_rng(np.random.SeedSequence(C.SEED, spawn_key=(BOOT_KEY,)))
    B = rng.integers(0, len(genes), size=(N_BOOT, len(genes)))
    res = dict(gene_set=C.GENE_SET, n_replicates=n, n_genes=len(genes), first_replicate=R0, arm='split', rates={})
    a = anchor.set_index('phenotype_id').pval_beta.reindex(genes)
    for al in ALPHAS:
        below = (M.values < al)
        per_gene = below.mean(1)
        boot = per_gene[B].mean(1)
        per_rep = below.sum(0)
        k_anchor = int((a.values < al).sum())
        res['rates'][str(al)] = dict(
            share=float(below.mean()), lo=float(np.quantile(boot, .025)), hi=float(np.quantile(boot, .975)),
            genes_below_per_replicate=dict(mean=float(per_rep.mean()), sd=float(per_rep.std(ddof=1)),
                                           q025=float(np.quantile(per_rep, .025)), q975=float(np.quantile(per_rep, .975))),
            anchor_genes_below=k_anchor, anchor_share_of_replicates_at_or_above=float((per_rep >= k_anchor).mean()))
    res['bh_any_call_share'] = float(np.mean([bh_any(M[c].values) for c in M.columns]))
    C.write_json(OUT / 'summary.json', res)
    for al, x in res['rates'].items():
        g = x['genes_below_per_replicate']
        print(f'pval_beta < {al}: share {x["share"]:.4f} [{x["lo"]:.4f}, {x["hi"]:.4f}] of {len(genes)} genes x {n} replicates; '
              f'genes below per replicate mean {g["mean"]:.2f} (sd {g["sd"]:.2f}, 95% range {g["q025"]:g}-{g["q975"]:g}); '
              f'the benchmark anchor has {x["anchor_genes_below"]} (share of replicates at or above it '
              f'{x["anchor_share_of_replicates_at_or_above"]:.2f})', flush=True)
    print(f'replicates with any Benjamini-Hochberg call at 5%: {res["bh_any_call_share"]:.3f} (at most 0.05 with valid p); '
          f'wrote {OUT / "summary.json"}', flush=True)


if __name__ == '__main__':
    main()

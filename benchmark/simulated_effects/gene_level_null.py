"""Gene-level stored null: each arm's gene-level p over many all-null anchor datasets (task 2026-10-02; the ablation and
total-only tensorQTL added the same day at the user's request).

Why: the benchmark's gene-level false-discovery figures rest on one anchor permutation and three replicates per
|beta|, too few to say whether Benjamini-Hochberg on the gene-level p holds its rate on these genes. Each anchor dataset
here is 02's build_dataset at beta = 0 (every gene null, no thinning) under the record permutation and label swap of
replicate r, scanned as 03 scans it with 03's seed for r: split and unit (the ablation, no Gibbs weights) by run_cis,
total-only tensorQTL by run_tensorqtl. At r = 0 that is exactly the benchmark's anchor, which the run checks first for
every arm (pval_perm and pval_beta equal the stored results/beta0.0/<arm>/cis_rep000.parquet). The null replicates are
r = R0 + k, k < n, indices no benchmark dataset uses. RASQUAL and TReCASE have no permutation p and are not run.

Output ROOT/gene_level_null: <arm>/cis_rNNNN.parquet per replicate (phenotype_id, pval_perm, pval_beta; a replicate whose
file exists is not rerun) and summary.json with, per arm and threshold, the share of gene-replicate units below it with
its gene-clustered interval, the per-replicate count of genes below it (mean, sd, 2.5% and 97.5% quantiles), where the
benchmark's own anchor (r = 0) falls among the replicates, and the share of replicates in which Benjamini-Hochberg at 5%
calls any gene (every call is false here; with valid p it is at most 0.05).
Usage: gene_level_null.py [n=100]   (SIMULATED_EFFECTS_GENE_SET picks the gene set, as for every step)
"""
import sys

import numpy as np
import pandas as pd

import common as C

MD, RA = C.module('02_make_datasets'), C.module('03_run_arms')
OUT = C.ROOT / 'gene_level_null'
ARMS = ('split', 'unit', C.TENSORQTL)
R0 = 1000                      # first null replicate index; the benchmark's datasets use 0-2
ALPHAS = (0.05, 0.01, 0.001)
N_BOOT = 2000
BOOT_KEY = 40                  # spawn key used by no other script here (see 05b_native_arms.NATIVE_THIN_KEY's list)
COLS = ['phenotype_id', 'pval_perm', 'pval_beta']


def scan(S, ds, arm, r):
    if arm == C.TENSORQTL:
        return RA.run_tensorqtl(S, ds, RA.cis_seed(r), OUT / 'scratch')[1][COLS]
    return RA.run_cis(S, ds, arm, RA.cis_seed(r))[COLS]


def bh_any(p, q=0.05):
    p = np.sort(np.asarray(p, float))
    return bool((p <= q * np.arange(1, len(p) + 1) / len(p)).any())


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    I, R, tested = C.load()
    S = C.setup(I)
    for arm in ARMS:
        (OUT / arm).mkdir(parents=True, exist_ok=True)
    for f in sorted(OUT.glob('cis_r*.parquet')):   # split's replicates of the first run, made by the same code, before the arms
        f.rename(OUT / 'split' / f.name)
    refs = {a: pd.read_parquet(C.RESULTS / 'beta0.0' / a / 'cis_rep000.parquet', columns=COLS) for a in ARMS}
    ds0 = MD.build_dataset(I, R, tested, 0.0, 1.0, 0)
    for arm in ARMS:
        got, ref = scan(S, ds0, arm, 0), refs[arm]
        if not (got.phenotype_id.tolist() == ref.phenotype_id.tolist()
                and np.array_equal(got.pval_perm.values, ref.pval_perm.values)
                and np.allclose(got.pval_beta.values, ref.pval_beta.values, rtol=1e-6, atol=0)):
            raise SystemExit(f'r = 0 does not reproduce the benchmark anchor\'s {arm} scan (results/beta0.0/{arm}/cis_rep000.parquet)')
        print(f'check: r = 0 reproduces the benchmark anchor\'s {arm} scan on {len(got)} genes', flush=True)
    for k in range(n):
        todo = [a for a in ARMS if not (OUT / a / f'cis_r{R0 + k:04d}.parquet').exists()]
        if not todo:
            continue
        ds = MD.build_dataset(I, R, tested, 0.0, 1.0, R0 + k)
        for arm in todo:
            C.write_atomic(OUT / arm / f'cis_r{R0 + k:04d}.parquet', lambda fh, df=scan(S, ds, arm, R0 + k): df.to_parquet(fh, index=False))
        if k % 10 == 9:
            print(f'  replicate {k + 1}/{n}', flush=True)
    if n:
        summarize(refs, n)


def summarize(refs, n):
    rng = np.random.default_rng(np.random.SeedSequence(C.SEED, spawn_key=(BOOT_KEY,)))
    res = dict(gene_set=C.GENE_SET, n_replicates=n, first_replicate=R0, arms={})
    B = None
    for arm in ARMS:
        P = pd.concat([pd.read_parquet(OUT / arm / f'cis_r{R0 + k:04d}.parquet').assign(rep=k) for k in range(n)])
        M = P.pivot(index='phenotype_id', columns='rep', values='pval_beta')   # genes x replicates
        genes = M.index
        if B is None:   # one gene resampling for every arm
            B = rng.integers(0, len(genes), size=(N_BOOT, len(genes)))
        a = refs[arm].set_index('phenotype_id').pval_beta.reindex(genes)
        r = dict(n_genes=len(genes), rates={})
        for al in ALPHAS:
            below = M.values < al
            boot = below.mean(1)[B].mean(1)
            per_rep = below.sum(0)
            k_anchor = int((a.values < al).sum())
            r['rates'][str(al)] = dict(
                share=float(below.mean()), lo=float(np.quantile(boot, .025)), hi=float(np.quantile(boot, .975)),
                genes_below_per_replicate=dict(mean=float(per_rep.mean()), sd=float(per_rep.std(ddof=1)),
                                               q025=float(np.quantile(per_rep, .025)), q975=float(np.quantile(per_rep, .975))),
                anchor_genes_below=k_anchor, anchor_share_of_replicates_at_or_above=float((per_rep >= k_anchor).mean()))
        r['bh_any_call_share'] = float(np.mean([bh_any(M[c].values) for c in M.columns]))
        res['arms'][arm] = r
        for al, x in r['rates'].items():
            g = x['genes_below_per_replicate']
            print(f'{arm} pval_beta < {al}: share {x["share"]:.4f} [{x["lo"]:.4f}, {x["hi"]:.4f}] of {len(genes)} genes x {n} '
                  f'replicates; genes below per replicate mean {g["mean"]:.2f} (sd {g["sd"]:.2f}); the benchmark anchor has '
                  f'{x["anchor_genes_below"]} (share of replicates at or above it {x["anchor_share_of_replicates_at_or_above"]:.2f})', flush=True)
        print(f'{arm}: replicates with any Benjamini-Hochberg call at 5%: {r["bh_any_call_share"]:.3f} (at most 0.05 with valid p)', flush=True)
    C.write_json(OUT / 'summary.json', res)
    print(f'wrote {OUT / "summary.json"}', flush=True)


if __name__ == '__main__':
    main()

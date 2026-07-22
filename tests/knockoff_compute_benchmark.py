"""
Compute & memory benchmark for the chromosome-coherent HMM knockoff generator
(validation item #7).

The coherent path fits ONE HMM per chromosome and draws M knockoff copies of the
whole chromosome (chromosome_hmm_knockoffs). Real chromosomes carry 1e4-1e6
variants, so before trusting the coherent path at genome scale we need to know
how fit-time, draw-time, and peak memory grow with the number of variants p (and
with N, K, M). This script measures that scaling on synthetic HMM genotypes.

It is a BENCHMARK, not a pass/fail unit test: it prints a table of wall-time and
peak RSS across a p-ladder so regressions and the practical ceiling are visible.
A loose smoke gate (small p completes under a generous memory bound) lives in
tests/test_calibration_validation.py.

Run: python tests/knockoff_compute_benchmark.py --p 2000 8000 20000 --N 150 --K 10 --M 10
"""

import argparse
import gc
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import tensorqtl.knockoffs as ko
from hmm_genotype_simulator import simulate_hmm_genotypes


def _bench_one(p, N, K, M, em_iter, method, seed=0):
    """One (fit + M draws) timing/memory measurement at chromosome size p."""
    geno, _, info = simulate_hmm_genotypes(p, N, seed=seed, K=min(K, 8),
                                           return_phased=(method == 'haplotype'))
    G = geno.T.astype(np.int64)                       # [N, p]
    gc.collect()
    tracemalloc.start()
    t0 = time.time()
    if method == 'haplotype':
        xL = info['xL'].T.astype(np.int64)
        xR = info['xR'].T.astype(np.int64)
        draws = ko.chromosome_hmm_knockoffs(
            K=K, M=M, n_em_iter=em_iter, seed=seed, method='haplotype',
            xL=xL, xR=xR)
    else:
        draws = ko.chromosome_hmm_knockoffs(
            G, K=K, M=M, E=3, n_em_iter=em_iter, seed=seed, method=method)
    elapsed = time.time() - t0
    _cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    draws_mb = draws.nbytes / 1e6
    return dict(p=p, sec=elapsed, peak_mb=peak / 1e6, draws_mb=draws_mb,
                per_1k_var=elapsed / (p / 1000))


def run_benchmark(p_ladder, N, K, M, em_iter, method):
    print(f"\n=== HMM knockoff compute benchmark (method={method}, N={N}, K={K}, "
          f"M={M}, EM_iter={em_iter}) ===")
    print(f"{'p (variants)':>13} {'fit+draw (s)':>13} {'s / 1k var':>11} "
          f"{'peak MB':>9} {'draws MB':>9}")
    rows = []
    for p in p_ladder:
        r = _bench_one(p, N, K, M, em_iter, method)
        rows.append(r)
        print(f"{r['p']:>13,} {r['sec']:>13.2f} {r['per_1k_var']:>11.3f} "
              f"{r['peak_mb']:>9.1f} {r['draws_mb']:>9.1f}")
    # linearity check: seconds-per-1k-variant should be roughly flat (O(p)).
    if len(rows) >= 2:
        ratios = [rows[i]['per_1k_var'] / rows[0]['per_1k_var'] for i in range(len(rows))]
        print(f"  per-1k-variant cost relative to smallest p: "
              f"{', '.join(f'{r:.2f}x' for r in ratios)} (flat => linear in p)")
    # extrapolate to a real chromosome
    if rows:
        big = rows[-1]
        for chrom_p in (100_000, 500_000):
            est_s = big['per_1k_var'] * (chrom_p / 1000)
            est_mb = big['draws_mb'] * (chrom_p / big['p'])
            print(f"  extrapolated to p={chrom_p:,}: ~{est_s/60:.1f} min fit+draw, "
                  f"~{est_mb:.0f} MB for the [M,N,p] draws alone")
    return rows


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--p', type=int, nargs='+', default=[2000, 8000, 20000])
    ap.add_argument('--N', type=int, default=150)
    ap.add_argument('--K', type=int, default=10)
    ap.add_argument('--M', type=int, default=10)
    ap.add_argument('--em_iter', type=int, default=15)
    ap.add_argument('--method', default='haplotype',
                    choices=['haplotype', 'genotype', 'single_chain'])
    args = ap.parse_args()
    run_benchmark(args.p, args.N, args.K, args.M, args.em_iter, args.method)

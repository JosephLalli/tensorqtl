"""
Overlap-stress calibration harness for the eGene knockoff path.

Purpose (external review's "best next experiment"): the standard per-gene
calibration uses non-overlapping windows and independent phenotypes, which does
NOT test the disputed regime. Formal genome-wide knockoff+ FDR control needs a
joint null sign-flip property across genes; independent per-gene knockoff draws
do not obviously provide it when cis windows overlap. This harness is designed to
*search for counterexamples* (severe inflation) and to compare per-gene vs shared
knockoff generation -- not to certify validity. A passing curve is evidence the
procedure is empirically usable, NOT a theorem.

It builds a single chromosome-level genotype matrix with strongly OVERLAPPING
cis windows and null phenotypes with tunable cross-gene residual correlation
(including near-duplicated phenotypes Y_B = rho Y_A + sqrt(1-rho^2) eps, the
sharp stress for joint sign behavior), then measures:

    FDR_hat = (1/B) sum_b V_b / max(R_b, 1)         (mean per-replicate FDP)
    P(R>0) under the complete null
    FDP quantiles, mean discoveries, Monte-Carlo intervals

for both independent-per-gene and shared-chromosome knockoffs.

This is a research harness, not a unit test; run it directly. It is intentionally
tunable and verbose.
"""

import numpy as np
import pandas as pd
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import tensorqtl.susie as susie
import tensorqtl.knockoffs as ko


def simulate_chromosome(n_snps, N, seed, n_factors=6, geno_noise=0.4):
    """
    One chromosome-level genotype matrix with block-LD structure, as dosages.

    Returns genotype array [n_snps, N] and positions [n_snps].
    """
    rng = np.random.RandomState(seed)
    # smooth latent factors along the chromosome -> local LD
    Z = rng.randn(N, n_factors)
    # each SNP loads on the factor for its genomic segment (contiguous blocks)
    seg = np.linspace(0, n_factors, n_snps, endpoint=False).astype(int)
    seg = np.clip(seg, 0, n_factors - 1)
    load = np.zeros((n_snps, n_factors))
    for j in range(n_snps):
        load[j, seg[j]] = 1.0
        # partial load on neighbor factor for smoother, overlapping LD
        if seg[j] + 1 < n_factors:
            load[j, seg[j] + 1] = 0.5
    Xc = Z @ load.T + geno_noise * rng.randn(N, n_snps)
    Xd = np.clip(np.round(Xc - Xc.min(0)), 0, 2).T  # [n_snps, N]
    pos = np.arange(n_snps) * 5000 + 10_000        # 5 kb spacing
    return Xd.astype(np.float32), pos, rng


def build_overlapping_genes(n_snps, pos, n_genes, window):
    """
    Place n_genes TSS uniformly along the chromosome so their +-window cis sets
    strongly overlap. Returns gene TSS positions [n_genes].
    """
    lo, hi = pos.min(), pos.max()
    return np.linspace(lo + window * 0.5, hi - window * 0.5, n_genes)


def simulate_phenotypes(geno, gene_tss, pos, window, N, rng,
                        n_causal_genes, causal_effect, rho, shared_signal):
    """
    Null + non-null phenotypes with tunable cross-gene residual correlation rho.

    - A shared latent residual e0 induces correlation across ALL genes:
        r_g = rho * e0 + sqrt(1-rho^2) * eps_g
      so Cov(r_g, r_h) = rho^2 (near-duplicated phenotypes as rho -> 1).
    - The first n_causal_genes get a causal variant in their window; if
      shared_signal, neighboring genes share the SAME causal SNP (signal at a
      shared variant), else each uses a window-unique variant.

    Returns phenotype array [n_genes, N] and dict gene_id -> causal variant idx.
    """
    n_genes = len(gene_tss)
    e0 = rng.randn(N)
    Y = np.zeros((n_genes, N))
    causal = {}
    for g in range(n_genes):
        r = rho * e0 + np.sqrt(max(1 - rho**2, 0.0)) * rng.randn(N)
        Y[g] = r
    for g in range(n_causal_genes):
        tss = gene_tss[g]
        in_win = np.where(np.abs(pos - tss) <= window)[0]
        if in_win.size == 0:
            continue
        if shared_signal:
            # pick a SNP near the boundary shared with the neighbor gene
            cidx = in_win[len(in_win) // 2 + (g % 3)]
        else:
            cidx = in_win[(g * 7) % in_win.size]
        Y[g] = Y[g] + causal_effect * geno[cidx]
        causal[f"G{g}"] = cidx
    return Y, causal


def run_replicate(n_snps=400, N=300, n_genes=120, window=100_000,
                  n_causal_genes=30, causal_effect=2.0, rho=0.5,
                  shared_signal=True, knockoff_mode='per_gene',
                  n_knockoffs=5, fdr=0.1, shrink=0.1, seed=0, L=5,
                  max_iter=80, verbose=False):
    """
    One replicate. knockoff_mode: 'per_gene' (independent per gene, current
    default) or 'shared' (one chromosome-wide knockoff reused by all genes).

    Returns dict(V, R, power, n_genes).
    """
    geno, pos, rng = simulate_chromosome(n_snps, N, seed)
    gene_tss = build_overlapping_genes(n_snps, pos, n_genes, window)
    Y, causal = simulate_phenotypes(geno, gene_tss, pos, window, N, rng,
                                    n_causal_genes, causal_effect, rho, shared_signal)

    samples = [f"S{i:04d}" for i in range(N)]
    vids = [f"v{j}" for j in range(n_snps)]
    genotype_df = pd.DataFrame(geno, index=vids, columns=samples)
    variant_df = pd.DataFrame({'chrom': ['chr1'] * n_snps, 'pos': pos}, index=vids)
    gene_ids = [f"G{g}" for g in range(n_genes)]
    phenotype_df = pd.DataFrame(Y, index=gene_ids, columns=samples)
    pos_df = pd.DataFrame({'chr': ['chr1'] * n_genes, 'pos': gene_tss.astype(int)},
                          index=gene_ids)
    cov_df = pd.DataFrame(rng.randn(N, 2), index=samples, columns=['PC1', 'PC2'])

    causal_gene_ids = set(causal.keys())

    if knockoff_mode == 'shared':
        eg = _run_shared_knockoff(genotype_df, variant_df, phenotype_df, pos_df,
                                  cov_df, gene_tss, pos, window, n_knockoffs, fdr,
                                  shrink, seed, L, max_iter)
    else:
        eg, _, _ = susie.map_egenes_knockoffs(
            genotype_df, variant_df, phenotype_df, pos_df, cov_df,
            fdr=fdr, n_knockoffs=n_knockoffs, shrink=shrink, window=window,
            L=L, max_iter=max_iter, verbose=verbose, seed=seed, localize=False)

    sel = set(eg[eg['selected']]['phenotype_id'])
    V = len(sel - causal_gene_ids)
    R = len(sel)
    power = len(sel & causal_gene_ids) / max(len(causal_gene_ids), 1)
    return {'V': V, 'R': R, 'power': power, 'n_genes': n_genes}


def _run_shared_knockoff(genotype_df, variant_df, phenotype_df, pos_df, cov_df,
                         gene_tss, pos, window, n_knockoffs, fdr, shrink, seed,
                         L, max_iter):
    """
    Shared-knockoff variant: generate ONE chromosome-wide knockoff (per draw)
    on the covariate-residualized genotype, then slice each gene's window from
    the SAME knockoff so a shared SNP has one coherent knockoff across all genes.
    Block-Gaussian would be needed at real chromosome scale; here p is small
    enough for a single Gaussian knockoff.
    """
    device = torch.device('cpu')
    from tensorqtl.core import Residualizer, impute_mean
    res = Residualizer(torch.tensor(cov_df.values, dtype=torch.float32))
    samples = phenotype_df.columns
    gix = np.array([genotype_df.columns.tolist().index(i) for i in samples])
    G = torch.tensor(genotype_df.values, dtype=torch.float32)[:, gix]
    impute_mean(G)
    Gr = res.transform(G)                     # variants x samples, residualized
    Xfull = Gr.T                              # samples x variants (all SNPs)

    gene_ids = list(phenotype_df.index)
    cols_by_gene = [np.where(np.abs(pos - gene_tss[gi]) <= window)[0]
                    for gi in range(len(gene_ids))]
    yr_by_gene = []
    for gid in gene_ids:
        y = torch.tensor(phenotype_df.loc[gid].values, dtype=torch.float32).reshape(1, -1)
        yr_by_gene.append(res.transform(y).T)

    # Generate the shared chromosome-wide knockoff ONCE per draw, then slice
    # every gene's window from the SAME knockoff object (the whole point of the
    # 'shared' construction: a shared SNP has one coherent knockoff everywhere).
    W_per_draw = np.zeros((n_knockoffs, len(gene_ids)))
    for r in range(n_knockoffs):
        gen = torch.Generator().manual_seed(seed * 1000 + r)
        Xk_full = ko.gaussian_knockoff(Xfull, shrink=shrink, generator=gen)
        for gi, gid in enumerate(gene_ids):
            cols = cols_by_gene[gi]
            if cols.size == 0:
                continue
            resu, p = ko.augmented_susie_fit(
                susie.susie, Xfull[:, cols], yr_by_gene[gi], Xk_full[:, cols], L,
                intercept=True, estimate_residual_variance=True,
                estimate_prior_variance=True, max_iter=max_iter,
                coverage=0.95, min_abs_corr=0.5)
            W_per_draw[r, gi] = ko.gene_level_W(resu['pip'], p, kind='max')
    sel = ko.select_egenes(gene_ids, W_per_draw, q=fdr)
    selset = set(sel['selected'])
    return pd.DataFrame({'phenotype_id': gene_ids,
                         'selected': [g in selset for g in gene_ids]})


def calibration_study(B=20, rho_grid=(0.0, 0.5, 0.9), modes=('per_gene', 'shared'),
                      complete_null=False, **rep_kwargs):
    """
    Run B replicates per (rho, mode) and report the correct FDR estimator.
    complete_null=True sets n_causal_genes=0 (report P(R>0)).
    """
    print(f"Overlap calibration study: B={B}, complete_null={complete_null}")
    print(f"  {rep_kwargs}")
    for rho in rho_grid:
        for mode in modes:
            Vs, Rs, powers = [], [], []
            for b in range(B):
                kw = dict(rep_kwargs)
                if complete_null:
                    kw['n_causal_genes'] = 0
                out = run_replicate(rho=rho, knockoff_mode=mode, seed=1000 * b + int(rho * 10),
                                    **kw)
                Vs.append(out['V']); Rs.append(out['R']); powers.append(out['power'])
            V = np.array(Vs, float); R = np.array(Rs, float)
            fdp = V / np.maximum(R, 1)
            se = fdp.std(ddof=1) / np.sqrt(B) if B > 1 else float('nan')
            # FDP quantiles (reviewer point: report the distribution, not just mean)
            q50, q90, qmax = np.quantile(fdp, [0.5, 0.9, 1.0])
            line = (f"  rho={rho:.1f} {mode:9s}: "
                    f"FDR_hat={fdp.mean():.3f} (se {se:.3f}) "
                    f"FDP[med/90/max]={q50:.2f}/{q90:.2f}/{qmax:.2f} "
                    f"P(R>0)={(R>0).mean():.2f} meanR={R.mean():.1f} "
                    f"power={np.mean(powers):.2f}")
            print(line, flush=True)


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--B', type=int, default=15)
    ap.add_argument('--complete_null', action='store_true')
    ap.add_argument('--n_genes', type=int, default=100)
    ap.add_argument('--n_knockoffs', type=int, default=5)
    ap.add_argument('--fdr', type=float, default=0.1)
    args = ap.parse_args()
    calibration_study(B=args.B, complete_null=args.complete_null,
                      n_genes=args.n_genes, n_knockoffs=args.n_knockoffs,
                      fdr=args.fdr)

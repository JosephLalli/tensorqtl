"""Simulated-effects cis-eQTL datasets from the real cohort's Salmon output (README: Generator).

A dataset is the cohort's Salmon point estimates and Gibbs draws for the GENE_SET genes with the
genotype association broken (donor records permuted against fixed genotypes, each moved record's
L/R labels swapped with probability one half) and a known cis effect injected by binomial thinning
(Gerard 2020, BMC Bioinformatics) of the haplotype carrying the lower-expressed allele, by
f = 2^-|beta|; the collapsed remainder U = pT - pL - pR by (fL + fR) / 2. Null genes are thinned by
their mean factor (depth matching). Total Gibbs draws are thinned like the point estimates; the
allelic Gibbs variance of a thinned record is Va' = dv q(pL', pR') / q(pL, pR) + q(pL', pR') / ln2^2
with dv the real across-draw variance of log2((YL + 0.5)/(YR + 0.5)) and q(x, y) = 1/(x + 0.5) +
1/(y + 0.5), read from Salmon's Gibbs sampler (01_check_inputs.py verifies the premise); 0 where
pL' + pR' = 0. Truth per non-null gene: beta on the count scale (total: the least-squares slope of
the exact log2 total fold on ALT dosage / 2) and the noise-free shift of the pipeline's own
phenotypes on the pipeline scale. Streams: perm and swap SeedSequence(SEED, (1, r)); null set,
causal variants and signs (2, r); thinning (3, r, round(1000 |beta|)), so scenarios are paired.

Output: DATASETS/beta<b>/rep<NNN>.npz (STORED), meta.json, truth.tsv.
"""
import time

import numpy as np
import pandas as pd

import common as C
from tensorqtl.hapmixqtl import LN2, summaries_from_point_estimates

BETAS = (0.0, 0.2, 0.4, 0.8)   # user decision 2026-09-26; 0.0 is the null anchor
N_DATASETS = {0.0: 1, 0.2: 3, 0.4: 3, 0.8: 3}   # user decisions 2026-09-26 and, for the 30-100-read set, 2026-09-27
NULL_FRACTION = 0.5            # user decision 2026-09-26, for beta > 0
PERM_KEY, DESIGN_KEY, THIN_KEY = 1, 2, 3
STORED = ('A', 'T', 'Va', 'Vt', 'pL', 'pR', 'pT', 'eff_lib', 'perm', 'swap', 'is_null', 'beta', 'causal_variant',
          'causal_maf', 'd', 'expressible', 'fL', 'fR', 'allelic_truth', 'total_truth', 'allelic_truth_pipeline',
          'total_truth_pipeline')


def thin(y, f, rng):
    """Binomial thinning of a fractional Salmon count; the identity at f = 1."""
    n = np.floor(y)
    return rng.binomial(n.astype(np.int64), f) + f * (y - n)


def move_records(R, perm, swap):
    """Column i takes real record perm[i]; L and R exchanged where swap[i] = -1."""
    s2, s3 = (swap < 0)[None, :], (swap < 0)[None, :, None]
    pL, pR, YL, YR = (R[k][:, perm] for k in ('pL', 'pR', 'YL', 'YR'))
    return dict(pL=np.where(s2, pR, pL), pR=np.where(s2, pL, pR), pT=R['pT'][:, perm],
                YL=np.where(s3, YR, YL), YR=np.where(s3, YL, YR), YT=R['YT'][:, perm], eff_lib=R['eff_lib'][perm])


def remainder(L, R, T):
    """The collapsed remainder T - L - R, floored at 0 (common.load checked it is at least -ROUNDING_TOL)."""
    return np.maximum(T - L - R, 0.0)


def thin_haplotypes(L, R, T, fl, fr, rng):
    Uc = remainder(L, R, T)
    L2, R2, U2 = thin(L, fl, rng), thin(R, fr, rng), thin(Uc, (fl + fr) / 2, rng)
    return L2, R2, T - ((L - L2) + (R - R2) + (Uc - U2))


def allelic_variance(pL, pR, pL2, pR2, YL, YR):
    """Va' in the operation order of summaries_from_point_estimates, so that at f = 1 it equals that function's Va."""
    dv = np.log2((YL + C.KAPPA) / (YR + C.KAPPA)).var(axis=2, ddof=0)
    q = 1.0 / (pL + C.KAPPA) + 1.0 / (pR + C.KAPPA)
    q2 = 1.0 / (pL2 + C.KAPPA) + 1.0 / (pR2 + C.KAPPA)
    return np.where((pL2 + pR2) <= 0, 0.0, dv * (q2 / q) + q2 / LN2 ** 2)


def generate(R, perm, swap, fL, fR, rng):
    """Move records, thin haplotype L by fL and R by fR ([G, N]), summarize."""
    M = move_records(R, perm, swap)
    out = dict(expressible=np.minimum(M['pL'], M['pR']) >= C.EXPRESSIBLE_MIN, eff_lib=M['eff_lib'])
    out['pL'], out['pR'], out['pT'] = thin_haplotypes(M['pL'], M['pR'], M['pT'], fL, fR, rng)
    _, _, out['YT'] = thin_haplotypes(M['YL'], M['YR'], M['YT'], fL[..., None], fR[..., None], rng)
    out['A'], out['T'], _, out['Vt'], _ = summaries_from_point_estimates(
        out['pL'], out['pR'], out['pT'], M['eff_lib'], M['YL'], M['YR'], out['YT'])
    out['Va'] = allelic_variance(M['pL'], M['pR'], out['pL'], out['pR'], M['YL'], M['YR'])
    out['removed'] = (M['pT'] - out['pT']).sum(0)
    out['moved'] = M
    return out


def record_permutation(N, r):
    rng = np.random.default_rng(np.random.SeedSequence(C.SEED, spawn_key=(PERM_KEY, r)))
    return rng.permutation(N), (rng.integers(0, 2, N) * 2 - 1).astype(np.int8)


def slope_with_intercept(x, y):
    """Least-squares slope of y on x with intercept, per row; NaN where x is constant."""
    xc = x - x.mean(1, keepdims=True)
    den = (xc ** 2).sum(1)
    out = np.full(len(x), np.nan)
    out[den > 0] = (xc[den > 0] * y[den > 0]).sum(1) / den[den > 0]
    return out


def pipeline_truth(M, fL, fR, s, g, kept):
    """Noise-free shifts A*, T* of the real records, regressed as the pipeline does (unweighted)."""
    pL, pR, pT = M['pL'], M['pR'], M['pT']
    a_star = np.log2((fL * pL + C.KAPPA) / (fR * pR + C.KAPPA)) - np.log2((pL + C.KAPPA) / (pR + C.KAPPA))
    U = remainder(pL, pR, pT)
    k = 1e6 / M['eff_lib'][None, :]
    pT_exp = pT - (1 - fL) * pL - (1 - fR) * pR - (1 - (fL + fR) / 2) * U
    t_star = np.log2(k * pT_exp + 1.0) - np.log2(k * pT + 1.0)
    sk = s * kept
    ss = (sk ** 2).sum(1)
    ba = np.full(len(s), np.nan)
    ba[ss > 0] = (sk[ss > 0] * a_star[ss > 0]).sum(1) / ss[ss > 0]
    return ba, slope_with_intercept(g / 2, t_star)


def build_dataset(I, R, tested, beta, null_fraction, r):
    """Dataset r of scenario |beta|: permuted, swapped, thinned records and their truth."""
    G, N = R['pL'].shape
    perm, swap = record_permutation(N, r)
    drng = np.random.default_rng(np.random.SeedSequence(C.SEED, spawn_key=(DESIGN_KEY, r)))
    is_null = np.zeros(G, bool)
    is_null[drng.permutation(G)[:int(round(null_fraction * G))]] = True
    j = np.array([t[drng.integers(len(t))] for t in tested])
    b = drng.choice([-1.0, 1.0], size=G) * beta
    xL, xR = I['xL'][j].astype(float), I['xR'][j].astype(float)   # 0 or 1, and dos = xL + xR: common.setup
    g = I['dos'][j].astype(float)
    lower = np.where(b > 0, 0.0, 1.0)[:, None]
    fL = np.where(xL == lower, 2.0 ** -beta, 1.0)
    fR = np.where(xR == lower, 2.0 ** -beta, 1.0)
    fbar = (fL.sum(1) + fR.sum(1)) / (2 * N)
    fL[is_null], fR[is_null] = fbar[is_null, None], fbar[is_null, None]
    trng = np.random.default_rng(np.random.SeedSequence(C.SEED, spawn_key=(THIN_KEY, r, int(round(1000 * beta)))))
    out = generate(R, perm, swap, fL, fR, trng)
    af = g.mean(1) / 2.0
    n_low = (xL == lower).astype(float) + (xR == lower)
    tt = slope_with_intercept(g / 2, np.log2((n_low * 2.0 ** -beta + (2 - n_low)) / 2))
    kept = C.allelic_kept(out['pL'], out['pR'], out['Va'])
    ba_p, bt_p = pipeline_truth(out.pop('moved'), fL, fR, xL - xR, g, kept)
    out.update(perm=perm, swap=swap, is_null=is_null, beta=b, fL=fL, fR=fR, fbar=fbar,
               causal_variant=np.asarray(I['vdf'].index[j].astype(str), dtype=str), causal_row=j,
               causal_maf=np.minimum(af, 1 - af), kept=kept,
               d=np.where(is_null[:, None], 0.0, b[:, None] * (xL - xR)), het=xL != xR,
               allelic_truth=np.where(is_null, 0.0, b), total_truth=np.where(is_null, 0.0, tt),
               allelic_truth_pipeline=np.where(is_null, 0.0, ba_p), total_truth_pipeline=np.where(is_null, 0.0, bt_p))
    return out


def main():
    I, R, tested = C.load()
    G, N = R['pL'].shape
    hap = R['pL'] + R['pR']
    nt = np.array([len(t) for t in tested])
    hap_real = np.median(hap, axis=1)
    med_lib = float(np.median(R['eff_lib']))
    C.DATASETS.mkdir(parents=True, exist_ok=True)
    rows, removed, seconds = [], [], []
    for beta in BETAS:
        (C.DATASETS / f'beta{beta}').mkdir(exist_ok=True)
        for r in range(N_DATASETS[beta]):
            t0 = time.perf_counter()
            ds = build_dataset(I, R, tested, beta, 1.0 if beta == 0 else NULL_FRACTION, r)
            C.write_atomic(C.DATASETS / f'beta{beta}' / f'rep{r:03d}.npz',
                           lambda fh: np.savez(fh, **{k: ds[k] for k in STORED}))
            seconds.append(time.perf_counter() - t0)
            nn = ~ds['is_null']
            if beta > 0:
                removed.append(ds['removed'] / med_lib)
            for k in np.where(nn)[0]:
                rows.append(dict(beta_abs=beta, rep=r, gene=I['genes'][k], causal_variant=ds['causal_variant'][k],
                                 causal_maf=ds['causal_maf'][k], beta=ds['beta'][k],
                                 allelic_truth=ds['allelic_truth'][k], total_truth=ds['total_truth'][k],
                                 allelic_truth_pipeline=ds['allelic_truth_pipeline'][k],
                                 total_truth_pipeline=ds['total_truth_pipeline'][k], n_het=int(ds['het'][k].sum()),
                                 n_het_expressible=int((ds['het'][k] & ds['expressible'][k]).sum()),
                                 n_het_kept=int((ds['het'][k] & ds['kept'][k]).sum()), mean_factor=ds['fbar'][k],
                                 median_hap_reads_real=hap_real[k]))
            print(f'beta {beta} rep {r:03d}: {int(nn.sum())} non-null / {int((~nn).sum())} null genes; reads removed '
                  f'per donor / median library max {ds["removed"].max() / med_lib:.2e}; {seconds[-1]:.2f} s', flush=True)
    truth = pd.DataFrame(rows)
    C.write_atomic(C.DATASETS / 'truth.tsv', lambda fh: truth.to_csv(fh, sep='\t', index=False), 'w')
    removed = np.concatenate(removed)
    expressible = truth.n_het_expressible.sum() / truth.n_het.sum()
    meta = dict(genes=list(I['genes']), donors=list(I['order']), gene_set=C.GENE_SET, n_genes=G, n_donors=N,
                n_datasets={str(b): N_DATASETS[b] for b in BETAS}, betas=list(BETAS),
                null_fraction={str(b): (1.0 if b == 0 else NULL_FRACTION) for b in BETAS}, seed=C.SEED,
                facts=dict(pairs=G * N, pairs_informative=int((hap > 0).sum()),
                           pairs_expressible=int((np.minimum(R['pL'], R['pR']) >= C.EXPRESSIBLE_MIN).sum()),
                           tested_per_gene=[int(nt.min()), int(np.median(nt)), int(nt.max())],
                           expressible_share_het_nonnull=float(expressible),
                           library_size_change=dict(median=float(np.median(removed)), max=float(removed.max()))))
    C.write_json(C.DATASETS / 'meta.json', meta)
    print(f'wrote {sum(N_DATASETS.values())} datasets to {C.DATASETS}: {G} genes x {N} donors; truth.tsv {len(truth)} '
          f'rows; expressible share of het donor-gene pairs in non-null genes {expressible:.3f}; reads removed per '
          f'donor / median library (beta > 0) median {np.median(removed):.2e}, max {removed.max():.2e}', flush=True)


if __name__ == '__main__':
    main()

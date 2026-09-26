"""Which allelic value is closest to phASER where Salmon's point estimate zeroes a haplotype?

The open decision (docs/pipeline_rules.md): when Salmon's point estimate puts
one haplotype of a donor-gene pair at zero reads, what value should the
allelic channel use? phASER's alignment-based counts at heterozygous SNPs are
the independent reference: a read carries an allele or it does not, so those
counts cannot be spread between the two copies the way Salmon splits reads
that do not cover a heterozygous site.

Candidate allelic values, all log2 with pseudocount 1/2 per haplotype:
  point       log2((pL + 1/2) / (pR + 1/2)), Salmon point estimates (the code now)
  draw_log    mean over the 200 Gibbs draws of log2((yL + 1/2) / (yR + 1/2)),
              the pre-2026-09-25 allelic value
  draw_count  log2((mean yL + 1/2) / (mean yR + 1/2)), the log of the draw-mean counts
  drop        the pair leaves the allelic channel; scored by how much real
              imbalance, by phASER, it would discard
Re-quantifying with Salmon --useEM is not measurable without re-running Salmon.

Reference: phASER log2((a + 1/2) / (b + 1/2)), gw_phased pairs with at least
MIN_PHASER reads at heterozygous SNPs, and its counting variance
(1/(a + 1/2) + 1/(b + 1/2)) / ln(2)^2.

Scores per candidate, per group (zero-haplotype pairs, pairs with reads on
both copies) and per depth band of Salmon haplotype-informative reads:
  abs_diff    |candidate - phASER| in log2 units (median reported)
  z           (candidate - phASER) / phASER counting sd; share |z| > 3 is the
              share inconsistent with phASER beyond phASER's own counting error
  mag_diff    |candidate| - |phASER|, sign-agnostic, so a phase flip between
              the two sources cannot inflate it; median reported
Gene-clustered bootstrap (genes resampled with replacement, SEED 42, 2000
draws) gives 95% intervals for the zero-pair shares and their differences.

Orientation is checked first on high-depth pairs with reads on both copies:
Salmon's L/R and phASER's a/b must correlate positively for the signed scores
to mean anything.
"""
import json
from pathlib import Path

import numpy as np

import alignment_discordance_coupling as ADC

D = Path('/mnt/ssd/lalli/brainvar_hapmix_deploy')
CACHE = D / 'cache' / 'gibbs_56b63c3b37ed5df8'
OUT = D / 'allelic_value_vs_phaser_20260925'
SEED, N_BOOT, MIN_PHASER, KAPPA, CHUNK = 42, 2000, 20, 0.5, 1500
LN2 = np.log(2.0)
BANDS = (('10-99', 10, 100), ('100-999', 100, 1000), ('1000+', 1000, np.inf), ('all 10+', 10, np.inf))
CANDIDATES = ('point', 'draw_log', 'draw_count')


def draw_values(G):
    YL = np.load(CACHE / 'YL.npy', mmap_mode='r')
    YR = np.load(CACHE / 'YR.npy', mmap_mode='r')
    draw_log = np.empty(YL.shape[:2]); draw_count = np.empty(YL.shape[:2])
    for s in range(0, G, CHUNK):
        yl = np.asarray(YL[s:s + CHUNK]); yr = np.asarray(YR[s:s + CHUNK])
        draw_log[s:s + CHUNK] = np.log2((yl + KAPPA) / (yr + KAPPA)).mean(2)
        draw_count[s:s + CHUNK] = np.log2((yl.mean(2) + KAPPA) / (yr.mean(2) + KAPPA))
    return draw_log, draw_count


def cluster_boot(gene_idx, flags, rng):
    """95% interval of the pooled share of `flags` (list of boolean arrays over
    the same pairs) and of their pairwise differences, resampling genes."""
    genes, inv = np.unique(gene_idx, return_inverse=True)
    n_g = np.bincount(inv, minlength=len(genes)).astype(float)
    k_g = [np.bincount(inv, weights=f.astype(float), minlength=len(genes)) for f in flags]
    W = rng.multinomial(len(genes), np.full(len(genes), 1 / len(genes)), size=N_BOOT).astype(float)
    den = W @ n_g
    shares = [W @ k / den for k in k_g]
    return shares


def main():
    OUT.mkdir(exist_ok=True)
    rng = np.random.RandomState(SEED)
    genes = (CACHE / 'genes.txt').read_text().split()
    samples = (CACHE / 'samples.txt').read_text().split()
    pL = np.load(CACHE / 'point_estimates' / 'pL.npy')
    pR = np.load(CACHE / 'point_estimates' / 'pR.npy')
    print('draw-based values', flush=True)
    draw_log, draw_count = draw_values(len(genes))
    cand = dict(point=np.log2((pL + KAPPA) / (pR + KAPPA)), draw_log=draw_log, draw_count=draw_count)
    print('phASER', flush=True)
    P, _ = ADC.load_phaser_all()
    ap, q, k, comp = ADC.phaser_arrays(P, genes, samples)
    ph, ph_sd = ap / LN2, np.sqrt(q) / LN2
    cov = comp & (k >= MIN_PHASER)

    n = pL + pR
    zero = (n > 0) & ((pL < 0.5) ^ (pR < 0.5))
    both = (n > 0) & (pL >= 0.5) & (pR >= 0.5)
    gidx = np.broadcast_to(np.arange(len(genes))[:, None], pL.shape)
    out = dict(n_pairs_with_phaser=int(cov.sum()), min_phaser_reads=MIN_PHASER)

    # ---- orientation ---------------------------------------------------------
    m = both & cov & (n >= 1000)
    r = {c: float(np.corrcoef(cand[c][m], ph[m])[0, 1]) for c in CANDIDATES}
    out['orientation_pearson_both_copies_1000plus'] = r
    out['orientation_n'] = int(m.sum())
    print(f'orientation, both-copy pairs >= 1000 reads (n={m.sum():,}): '
          + ', '.join(f'{c} r={v:.3f}' for c, v in r.items()), flush=True)
    # A low r can mean most genes sit near balance, or that the two sources
    # disagree on phase. Sign agreement where both call a clear imbalance
    # separates the two: a phase disagreement shows up as opposite signs.
    clear = m & (np.abs(ph) > 1) & (np.abs(ph) / ph_sd > 3) & (np.abs(draw_count) > 1)
    agree = float((np.sign(ph[clear]) == np.sign(draw_count[clear])).mean())
    deep = both & cov & (n >= 1000) & (k >= 200)
    r_deep = float(np.corrcoef(draw_count[deep], ph[deep])[0, 1])
    out['orientation_sign_agreement_clear_imbalance'] = dict(n=int(clear.sum()), share_same_sign=agree)
    out['orientation_pearson_phaser_200plus'] = dict(n=int(deep.sum()), draw_count_r=r_deep)
    print(f'  sign agreement where both call > 2-fold (n={clear.sum():,}): {agree:.1%}; '
          f'r with >= 200 phASER reads (n={deep.sum():,}): {r_deep:.3f}', flush=True)

    # ---- scores --------------------------------------------------------------
    out['scores'] = {}
    for grp_name, grp in (('zero-haplotype', zero), ('both copies', both)):
        for band, lo, hi in BANDS:
            sel = grp & cov & (n >= lo) & (n < hi)
            row = dict(n=int(sel.sum()))
            for c in CANDIDATES:
                d = cand[c][sel] - ph[sel]
                z = d / ph_sd[sel]
                mag = np.abs(cand[c][sel]) - np.abs(ph[sel])
                row[c] = dict(median_abs_diff=float(np.median(np.abs(d))),
                              share_abs_z_gt_3=float((np.abs(z) > 3).mean()),
                              median_mag_diff=float(np.median(mag)))
            out['scores'][f'{grp_name} | {band}'] = row
            print(f'{grp_name:15s} {band:8s} n={row["n"]:>8,}  ' + '  '.join(
                f'{c}: |d| {row[c]["median_abs_diff"]:.2f}, |z|>3 {row[c]["share_abs_z_gt_3"]:.1%}, '
                f'mag {row[c]["median_mag_diff"]:+.2f}' for c in CANDIDATES), flush=True)

    # ---- zero pairs that phASER calls clearly imbalanced ---------------------
    real = zero & cov & (n >= 10) & (np.abs(ph) > 1) & (np.abs(ph) / ph_sd > 3)
    mono = zero & cov & (n >= 10) & (np.minimum(k - 0, 1) > 0)
    r_ = np.exp(-np.abs(ap)); minor = r_ / (1 + r_)
    mono = zero & cov & (n >= 10) & (minor < 0.05)
    sub = {}
    for lab, s in (('phASER clearly imbalanced (> 2-fold, > 3 sd)', real),
                   ('phASER monoallelic (minor fraction < 0.05)', mono)):
        sub[lab] = dict(n=int(s.sum()), share_of_zero_pairs=float(s.sum() / (zero & cov & (n >= 10)).sum()),
                        median_abs_phaser=float(np.median(np.abs(ph[s]))),
                        **{c: dict(median_abs_value=float(np.median(np.abs(cand[c][s]))),
                                   median_mag_diff=float(np.median(np.abs(cand[c][s]) - np.abs(ph[s]))),
                                   share_abs_z_gt_3=float((np.abs((cand[c][s] - ph[s]) / ph_sd[s]) > 3).mean()))
                           for c in CANDIDATES})
        print(f'zero pairs, {lab}: n={s.sum():,} ({sub[lab]["share_of_zero_pairs"]:.1%} of covered zero pairs), '
              f'median |phASER| {sub[lab]["median_abs_phaser"]:.2f}; ' + '; '.join(
                  f'{c} median |value| {sub[lab][c]["median_abs_value"]:.2f}, |z|>3 {sub[lab][c]["share_abs_z_gt_3"]:.1%}'
                  for c in CANDIDATES), flush=True)
    out['zero_pairs_with_real_imbalance'] = sub

    # ---- gene-clustered intervals, zero pairs, all depths >= 10 --------------
    sel = zero & cov & (n >= 10)
    flags = [np.abs((cand[c][sel] - ph[sel]) / ph_sd[sel]) > 3 for c in CANDIDATES]
    shares = cluster_boot(gidx[sel], flags, rng)
    ci = {}
    for i, c in enumerate(CANDIDATES):
        ci[c] = [float(np.quantile(shares[i], q_)) for q_ in (0.025, 0.975)]
    for i, j in ((1, 0), (2, 0), (2, 1)):
        dsh = shares[i] - shares[j]
        ci[f'{CANDIDATES[i]} - {CANDIDATES[j]}'] = [float(np.quantile(dsh, q_)) for q_ in (0.025, 0.975)]
    out['zero_pairs_share_abs_z_gt_3_ci95_gene_clustered'] = ci
    out['n_genes_zero_pairs_covered'] = int(len(np.unique(gidx[sel])))
    print('gene-clustered 95% intervals, share |z| > 3 on zero pairs (10+ reads): '
          + '; '.join(f'{k_} [{v[0]:.3f}, {v[1]:.3f}]' for k_, v in ci.items()), flush=True)
    (OUT / 'summary.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()

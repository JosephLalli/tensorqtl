"""Scatter of phASER's allelic ratio against Salmon's, one point per donor-gene pair.

x: phASER log2((a + 1/2) / (b + 1/2)) folded to larger over smaller allele, so
   always >= 0.
y: Salmon's allelic ratio in the SAME allele order, larger-by-phASER over
   smaller-by-phASER, so y < 0 means Salmon puts the imbalance on the other
   allele. Two panels: Salmon's point estimate, and log2 of the Gibbs
   draw-mean counts.
Pairs as in allelic_value_vs_phaser.py: >= 20 gw-phased phASER reads and >= 10
Salmon haplotype-informative reads. Pairs with reads on both copies in the
point estimate are a grey density; pairs where the point estimate zeroes one
haplotype are orange points.
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

import alignment_discordance_coupling as ADC
from allelic_value_vs_phaser import CACHE, KAPPA, LN2, MIN_PHASER, OUT, draw_values


def main():
    OUT.mkdir(exist_ok=True)
    genes = (CACHE / 'genes.txt').read_text().split()
    samples = (CACHE / 'samples.txt').read_text().split()
    pL = np.load(CACHE / 'point_estimates' / 'pL.npy')
    pR = np.load(CACHE / 'point_estimates' / 'pR.npy')
    _, draw_count = draw_values(len(genes))
    point = np.log2((pL + KAPPA) / (pR + KAPPA))
    P, _ = ADC.load_phaser_all()
    ap, _, k, comp = ADC.phaser_arrays(P, genes, samples)
    ph = ap / LN2
    n = pL + pR
    sel = comp & (k >= MIN_PHASER) & (n >= 10)
    zero = (pL < 0.5) ^ (pR < 0.5)
    sign = np.where(ph >= 0, 1.0, -1.0)
    x = np.abs(ph)
    np.savez_compressed(OUT / 'scatter_pairs.npz', x=x[sel], point=(sign * point)[sel],
                        draw_count=(sign * draw_count)[sel], zero=zero[sel])

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2), sharex=True, sharey=True)
    lim_x, lim_y = (0, 9), (-15, 15)
    for ax, (label, y) in zip(axes, (('Salmon point estimate (quant.sf NumReads)', sign * point),
                                     ('Salmon Gibbs draw-mean counts', sign * draw_count))):
        b = sel & ~zero
        z = sel & zero
        hb = ax.hexbin(x[b], y[b], gridsize=(90, 110), extent=(*lim_x, *lim_y),
                       cmap='Greys', norm=LogNorm(), mincnt=1, linewidths=0)
        ax.scatter(x[z], y[z], s=3, c='#e66101', alpha=0.25, linewidths=0, rasterized=True)
        ax.plot(lim_x, lim_x, ls='--', c='#2c7bb6', lw=1.2)
        ax.axhline(0, c='k', lw=0.6)
        ax.set_xlim(*lim_x); ax.set_ylim(*lim_y)
        ax.set_title(label, fontsize=12)
        ax.set_xlabel('phASER allelic ratio, log2(larger / smaller allele)')
        r_b = np.corrcoef(x[b], y[b])[0, 1]
        ax.text(0.02, 0.98,
                f'reads on both copies: {b.sum():,} pairs (grey), r = {r_b:.2f}\n'
                f'one copy at zero in the point estimate: {z.sum():,} pairs (orange)\n'
                f'dashed: Salmon = phASER',
                transform=ax.transAxes, va='top', fontsize=9,
                bbox=dict(boxstyle='round', fc='white', ec='0.8'))
    axes[0].set_ylabel('Salmon allelic ratio, log2, same allele order as phASER\n'
                       '(below 0: Salmon favours the other allele)')
    cb = fig.colorbar(hb, ax=axes, shrink=0.8, pad=0.01)
    cb.set_label('pairs per hexagon (grey)')
    fig.suptitle('Allelic ratio per donor-gene pair: phASER counts at heterozygous SNPs vs Salmon '
                 f'(>= {MIN_PHASER} phASER reads, >= 10 Salmon haplotype reads)', fontsize=12)
    out = OUT / 'phaser_vs_salmon_allelic_ratio.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(out)


if __name__ == '__main__':
    main()

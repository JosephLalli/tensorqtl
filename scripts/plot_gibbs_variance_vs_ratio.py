"""Does the Gibbs variance flag the allelic values that disagree with phASER?

One point per donor-gene pair (pairs as in allelic_value_vs_phaser.py):
x      Salmon's allelic ratio, log2, folded to larger over smaller copy
y      Gibbs variance: across-draw variance of log2((yL + 1/2) / (yR + 1/2)),
       log scale. The shipped weight is 1 / (this + the counting term); the
       counting term is summarised in the printout, not plotted.
colour (|Salmon| - |phASER|) / phASER counting sd: how many of phASER's own
       counting sd Salmon's folded ratio lies above (red) or below (blue)
       phASER's folded ratio, clipped at +-10
Two panels: Salmon's point estimate, and log2 of the Gibbs draw-mean counts;
the Gibbs variance is the same in both. Points are drawn in order of |colour|
so the most discordant are on top.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

from allelic_value_vs_phaser import MIN_PHASER, OUT
from plot_allelic_value_vs_phaser import load_pairs

CLIP = 10


def main():
    d = load_pairs()
    x_ph, sd, gv, zero = d['x'], d['ph_sd'], d['gibbs_var'], d['zero']
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2), sharey=True)
    norm = Normalize(-CLIP, CLIP)
    for ax, (label, key) in zip(axes, (('Salmon point estimate (quant.sf NumReads)', 'point'),
                                       ('Salmon Gibbs draw-mean counts', 'draw_count'))):
        xs = np.abs(d[key])
        z = (xs - x_ph) / sd
        o = np.argsort(np.abs(z))
        sc = ax.scatter(xs[o], np.maximum(gv[o], 1e-6), c=np.clip(z[o], -CLIP, CLIP), cmap='RdBu_r',
                        norm=norm, s=1.5, linewidths=0, rasterized=True)
        ax.set_yscale('log')
        ax.set_xlim(0, 15)
        ax.set_title(label, fontsize=12)
        ax.set_xlabel('Salmon allelic ratio, log2(larger / smaller copy)')
        far = np.abs(z) > 3
        ax.text(0.98, 0.02,
                f'{len(z):,} pairs; beyond 3 sd of phASER: {far.mean():.1%}\n'
                f'median Gibbs variance, beyond 3 sd: {np.median(gv[far]):.3f}\n'
                f'median Gibbs variance, within 3 sd: {np.median(gv[~far]):.3f}',
                transform=ax.transAxes, ha='right', va='bottom', fontsize=9,
                bbox=dict(boxstyle='round', fc='white', ec='0.8'))
        print(f'{key}: beyond 3 sd {far.mean():.3f}; median Gibbs variance beyond / within '
              f'{np.median(gv[far]):.4f} / {np.median(gv[~far]):.4f}; zero pairs beyond '
              f'{far[zero].mean():.3f}')
    axes[0].set_ylabel('Gibbs variance of the log2 allelic ratio (log scale)')
    cb = fig.colorbar(sc, ax=axes, shrink=0.85, pad=0.01)
    cb.set_label('Salmon minus phASER folded ratio, in phASER counting sd\n'
                 f'(red: Salmon more imbalanced; clipped at +-{CLIP})')
    fig.suptitle('Gibbs variance against Salmon\'s folded allelic ratio, coloured by distance from phASER '
                 f'(>= {MIN_PHASER} phASER reads, >= 10 Salmon haplotype reads)', fontsize=12)
    out = OUT / 'gibbs_variance_vs_ratio.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    zc, ct = d['count_term'], zero
    print(f'zero pairs: median Gibbs variance {np.median(gv[ct]):.3f}, median counting term '
          f'{np.median(zc[ct]):.3f}; pairs with reads on both copies: {np.median(gv[~ct]):.3f}, '
          f'{np.median(zc[~ct]):.3f}')
    print(out)


if __name__ == '__main__':
    main()

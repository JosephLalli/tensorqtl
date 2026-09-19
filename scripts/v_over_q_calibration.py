"""Is the Gibbs draw variance the same thing as the Poisson prediction, rescaled?

mixQTL models the allelic error as sigma^2 * q_i with q_i = 1/y1 + 1/y2 and a
free per-gene sigma^2 (paper Eq 11). hapmixQTL's two-component form is
c_g * v_i with v_i the across-draw variance of the per-draw log ratio and a
free per-gene c_g. Structurally identical: one free scale times a per-donor
shape. They are the SAME model if and only if v_i / q_i is constant across
donors within a gene, since then one scale absorbs the difference.

So the question is not the median of v/q -- that is just the scale, and a
per-gene scale cancels from a within-gene permutation p-value anyway. The
question is the WITHIN-GENE SPREAD of v/q. That is what no rescaling can
absorb and what makes the two models genuinely different.

This is the allelic-channel calibration the hapmixQTL handoff flagged as
never having been run: the existing draw_var_over_RTA_Poisson median of
0.981 was computed on the TOTAL channel, on genes with count > 50.

Deterministic: no permutation, no null, no seed.
"""

import json
import sys

import numpy as np

sys.path.insert(0, 'scripts')
sys.path.insert(0, '.')

OUT = '/mnt/ssd/lalli/brainvar_hapmix_deploy/mixqtl_replication_20260919'
KAPPA = 0.5


def main():
    import tensorqtl.hapmixqtl as HM
    from compare_mixqtl_replication import load_inputs

    I = load_inputs()
    keep, genes = I['keep'], I['genes']
    YL = I['YL'][:, keep]
    YR = I['YR'][:, keep]

    # v: across-draw variance of the per-draw log ratio, the quantity
    # hapmixQTL weights by. Taken WITHOUT the Poisson q term, since adding q
    # and then comparing to q would be circular.
    _A, _T, _Va, _Vt, _C = HM.compute_summaries_from_gibbs(
        I['YL'], I['YR'], yT=I['YT'], count_noise=False)
    v = _Va[:, keep]
    mL, mR = YL.mean(2), YR.mean(2)
    q = 1.0 / (mL + KAPPA) + 1.0 / (mR + KAPPA)

    inf = v > 1e-12
    ratio = np.where(inf, v / q, np.nan)

    print(f'{int(inf.sum())} informative donor-gene pairs of {v.size}\n')
    fin = ratio[np.isfinite(ratio)]
    print('v / q  -- the draw variance over its Poisson prediction')
    print(f'  median {np.median(fin):.4f}   '
          f'IQR {np.percentile(fin, 25):.4f}-{np.percentile(fin, 75):.4f}')
    print(f'  1st-99th pct {np.percentile(fin, 1):.4f}-'
          f'{np.percentile(fin, 99):.4f}')
    print('  (median is the SCALE; a per-gene scale cancels from the '
          'permutation p)\n')

    print('WITHIN-GENE spread of v/q -- this is what no rescaling absorbs')
    sds, folds, corrs = [], [], []
    for j, g in enumerate(genes):
        m = inf[j]
        if m.sum() < 10:
            continue
        r = ratio[j][m]
        lr = np.log(r)
        sds.append(lr.std())
        folds.append(np.percentile(r, 90) / np.percentile(r, 10))
        corrs.append(np.corrcoef(np.log(v[j][m]), np.log(q[j][m]))[0, 1])
    sds, folds, corrs = np.array(sds), np.array(folds), np.array(corrs)
    print(f'  genes assessed: {len(sds)}')
    print(f'  sd of log(v/q) within gene: median {np.median(sds):.4f}, '
          f'range {sds.min():.4f}-{sds.max():.4f}')
    print(f'  10th-to-90th percentile FOLD range of v/q within gene: '
          f'median {np.median(folds):.2f}x')
    print(f'  corr(log v, log q) within gene: median {np.median(corrs):.4f}, '
          f'range {corrs.min():.3f}-{corrs.max():.3f}')

    print('\nInterpretation guide:')
    print('  sd(log v/q) ~ 0 and corr ~ 1  -> same model, differing by a scale')
    print('  sd(log v/q) large            -> genuinely different error models')

    # how much of v is predictable from the counts at all?
    m = inf.ravel()
    lv, lq = np.log(v.ravel()[m]), np.log(q.ravel()[m])
    r2 = np.corrcoef(lv, lq)[0, 1] ** 2
    print(f'\npooled R^2 of log v on log q: {r2:.4f}')
    print(f'  -> {100 * (1 - r2):.1f}% of the variation in log v is NOT '
          f'explained by the counts,\n     i.e. is information the draws '
          f'carry and the Poisson prediction cannot.')

    json.dump(dict(
        n_informative=int(inf.sum()),
        median_v_over_q=float(np.median(fin)),
        iqr_v_over_q=[float(np.percentile(fin, 25)),
                      float(np.percentile(fin, 75))],
        median_within_gene_sd_log_ratio=float(np.median(sds)),
        median_within_gene_fold_range=float(np.median(folds)),
        median_within_gene_corr_logv_logq=float(np.median(corrs)),
        pooled_r2_logv_on_logq=float(r2),
    ), open(f'{OUT}/v_over_q_calibration.json', 'w'), indent=1)


if __name__ == '__main__':
    main()

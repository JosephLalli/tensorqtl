"""Three standard errors for the same weighted slope, and how each calibrates.

The weighted through-origin estimator is

    beta_hat = sum_i w_i s_i a_i / sum_i w_i s_i^2

and its TRUE variance is

    sum_i w_i^2 s_i^2 Var(a_i) / (sum_i w_i s_i^2)^2

which collapses to 1/sum_i w_i s_i^2 only when w_i = 1/Var(a_i) exactly. So
the shipped known-variance form is not a neutral choice: it asserts the
weights are exactly the inverse error variance. Measured per gene, that
assertion is off by 1.3x to 24x.

Three ways to form the standard error, in increasing order of how little
they assume about the weights:

  known     se^2 = 1 / sum(w s^2)
            asserts w IS the inverse error variance, scale and shape.

  fitted    se^2 = sigma_hat^2 / sum(w s^2),  sigma_hat^2 = RSS_w / (n-1)
            asserts w has the right SHAPE and fits one scale per gene.
            This is what mixQTL does.

  sandwich  se^2 = sum(w^2 s^2 e^2) / (sum(w s^2))^2,  e = a - beta_hat s
            asserts nothing about w. Reads each donor's error magnitude off
            its own residual. Huber-White HC0; HC3 also reported since HC0
            is known to run small in finite samples.

Calibration is var(beta_hat) across 40 null permutations over mean(se^2).
One means the reported uncertainty matches the realized spread.

Master seed 42; the permutation is only a device for generating repeated
realizations, not a rank test.
"""

import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, 'scripts')
sys.path.insert(0, '.')

OUT = '/mnt/ssd/lalli/brainvar_hapmix_deploy/mixqtl_replication_20260919'
NP_NULL = 40
SEED = 42


def fit_all_se(y, X, w):
    """Weighted through-origin fit; return beta and the three se^2 forms."""
    sw = np.sqrt(w)
    yw = y * sw
    Xw = X * sw[:, None]
    xx = (Xw * Xw).sum(0)                       # sum w s^2
    xy = Xw.T @ yw                              # sum w s a
    with np.errstate(divide='ignore', invalid='ignore'):
        beta = xy / xx
    n = int((w > 0).sum())

    # residuals on the ORIGINAL scale, per donor, per variant
    # e_ij = a_i - beta_j s_ij   -> [n, P]
    e = y[:, None] - X * beta[None, :]

    se2_known = 1.0 / xx
    rss_w = (w[:, None] * e * e).sum(0)
    se2_fitted = (rss_w / max(n - 1, 1)) / xx
    meat = ((w[:, None] ** 2) * (X ** 2) * (e ** 2)).sum(0)
    se2_sand = meat / (xx ** 2)
    # HC3: divide each squared residual by (1 - h_i)^2, h_i the weighted
    # leverage w_i s_i^2 / sum(w s^2)
    h = (w[:, None] * X ** 2) / xx[None, :]
    h = np.clip(h, 0, 0.999)
    meat3 = ((w[:, None] ** 2) * (X ** 2) * (e ** 2) / (1 - h) ** 2).sum(0)
    se2_sand3 = meat3 / (xx ** 2)
    return beta, se2_known, se2_fitted, se2_sand, se2_sand3


def main():
    import tensorqtl.hapmixqtl as HM
    from compare_mixqtl_replication import load_inputs, gene_variant_index

    I = load_inputs()
    genes, keep = I['genes'], I['keep']
    A, _T, Va, _Vt, _C = HM.compute_summaries_from_gibbs(
        I['YL'], I['YR'], yT=I['YT'], count_noise=True)
    A, Va = A[:, keep], Va[:, keep]
    s_all = (I['xL'] - I['xR'])[I['idx']][:, keep]

    rows = []
    for j, g in enumerate(genes):
        vsel = gene_variant_index(I, g)
        if vsel.size == 0:
            continue
        inf = Va[j] > 1e-12
        n = int(inf.sum())
        if n <= 2:
            continue
        a = A[j][inf]
        S = s_all[vsel][:, inf]
        S = S[S.var(1) > 0]
        if S.shape[0] == 0:
            continue
        X = S.T
        w = 1.0 / np.maximum(Va[j][inf], 1e-12)

        P = X.shape[1]
        b = np.empty((NP_NULL, P))
        acc = {k: np.empty((NP_NULL, P)) for k in
               ('known', 'fitted', 'sandwich', 'sandwich_hc3')}
        for pi in range(NP_NULL):
            prm = np.random.RandomState(SEED + 10007 + pi).permutation(n)
            bb, k1, k2, k3, k4 = fit_all_se(a[prm], X, w[prm])
            b[pi] = bb
            acc['known'][pi] = k1
            acc['fitted'][pi] = k2
            acc['sandwich'][pi] = k3
            acc['sandwich_hc3'][pi] = k4

        vb = np.var(b, axis=0, ddof=1)
        for k, arr in acc.items():
            m = np.nanmean(arr, axis=0)
            ok = np.isfinite(vb) & np.isfinite(m) & (m > 0)
            if ok.sum():
                rows.append(dict(gene=g, se_form=k, n_inf=n,
                                 calibration=float(np.median(vb[ok] / m[ok]))))
    d = pd.DataFrame(rows)
    d.to_csv(f'{OUT}/se_forms_calibration.tsv', sep='\t', index=False)

    print(f'{d.gene.nunique()} genes, calibration = '
          f'var(beta_hat) / mean(se^2)\n')
    print(f'{"SE form":16s} {"median":>8s} {"IQR":>18s} {"range":>16s} '
          f'{"fold":>6s}')
    summ = {}
    for k in ('known', 'fitted', 'sandwich', 'sandwich_hc3'):
        c = d[d.se_form == k].calibration
        print(f'{k:16s} {c.median():8.2f} '
              f'{c.quantile(.25):8.2f}-{c.quantile(.75):<8.2f} '
              f'{c.min():7.2f}-{c.max():<7.2f} {c.max() / c.min():6.1f}')
        summ[k] = dict(median=float(c.median()),
                       iqr=[float(c.quantile(.25)), float(c.quantile(.75))],
                       min=float(c.min()), max=float(c.max()),
                       fold=float(c.max() / c.min()))
    print('\n  1.00 = reported uncertainty matches realized spread')
    print('  the FOLD column is the spread across genes; a form that needs '
          'no\n  per-gene scale should have both a median near 1 and a small '
          'fold.')
    json.dump(summ, open(f'{OUT}/se_forms_calibration.json', 'w'), indent=1)


if __name__ == '__main__':
    main()

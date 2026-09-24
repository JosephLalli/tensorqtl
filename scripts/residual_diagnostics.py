"""Is Var(eps_i) = sigma^2 * v_i actually true on this data?

WHY THIS AND NOT A WALD-VERSUS-LIKELIHOOD-RATIO TEST. For a Gaussian linear
model those are the same test. Testing one coefficient, t^2 = F exactly, and
2 log LR = n log(RSS_0 / RSS_1) is a monotone function of F, so all three order
the data identically and differ only in the reference distribution. Referencing
hapmixQTL's T^2 to F(1, dof) is therefore the EXACT test -- provided the model
holds. RASQUAL's likelihood ratio differs from a Wald statistic only because its
model is genuinely non-Gaussian (negative binomial plus beta-binomial).

So a non-uniform null under an exact reference means the MODEL is wrong, and the
model makes two checkable claims about the whitened residual e_i / sqrt(v_i):

  SHAPE.  Its variance should not depend on v_i. If the standardized squared
          residual still trends with log v, then v has the wrong shape across
          donors and no single fitted sigma^2 can repair it -- one scalar cannot
          fix a per-donor error. Tested as a pooled regression of the
          standardized squared residual on log v, the same diagnostic that
          condemned the additive variance form in 2026-09-16.

  TAILS.  It should be Gaussian. Heavy tails make an F reference anticonservative
          at exactly the small-p end where the excess was seen. Tested as excess
          kurtosis, pooled and per gene.

Under the null model, with no genotype term. The allelic channel is
through-origin with no nuisance columns since 2026-09-15, so its null residual
is the response itself; the total channel keeps its intercept and covariates.
"""
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import compare_mixqtl_replication as CM   # noqa: E402

D = Path('/mnt/ssd/lalli/brainvar_hapmix_deploy')
RUN = D / 'rasqual_default_mode_20260923'
EPS = 1e-12


def wls_resid(y, X, w):
    """Weighted least squares residual of y on X with weights w."""
    sw = np.sqrt(w)
    yw, Xw = y * sw, X * sw[:, None]
    beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    return (yw - Xw @ beta), Xw.shape[1]


def main():
    null46 = {l.strip() for l in open(D / 'genes_null46_20260923.txt') if l.strip()}
    strata = pd.read_csv(D / 'genes_59_strata_20260923.tsv', sep='\t').set_index('gene')
    I = CM.load_inputs(gene_list=str(D / 'genes_59_stratified_20260923.txt'),
                       regions=str(RUN / 'regions.bed'))
    keep = I['keep']
    import run_hapmixqtl_from_salmon as H
    with contextlib.redirect_stdout(io.StringIO()):
        A, T, Va, Vt, _ = H.compute_summaries_from_gibbs(
            I['YL'][:, keep], I['YR'][:, keep], yT=I['YT'][:, keep])
    C = I['cov_df'].values
    N = C.shape[0]
    Xt = np.column_stack([np.ones(N), C])          # total channel: intercept + covariates

    rows, pooled = [], {'allelic': [], 'total': []}
    for gi_, g in enumerate(I['genes']):
        if g not in null46:
            continue
        for ch, y, v, X in (('allelic', A[gi_], Va[gi_], None),
                            ('total', T[gi_], Vt[gi_], Xt)):
            m = np.isfinite(y) & np.isfinite(v) & (v > EPS)
            if m.sum() < 20:
                continue
            w = 1.0 / v[m]
            if X is None:
                # through-origin with no nuisance columns: the null residual IS
                # the whitened response
                e, ncol = y[m] * np.sqrt(w), 0
            else:
                e, ncol = wls_resid(y[m], X[m], w)
            dof = max(int(m.sum()) - ncol, 1)
            s2 = float((e ** 2).sum() / dof)         # the fitted sigma^2
            z = e / np.sqrt(s2)                      # standardized whitened residual
            lv = np.log(v[m])
            sl, _, r, p, se = sps.linregress(lv, z ** 2)
            rows.append(dict(gene=g, channel=ch, stratum=strata.loc[g, 'stratum'],
                             n=int(m.sum()), slope_z2_on_logv=float(sl),
                             slope_se=float(se), p_slope=float(p),
                             kurtosis=float(sps.kurtosis(z, fisher=True)),
                             sd_logv=float(np.std(lv))))
            pooled[ch].append(np.column_stack([lv, z ** 2, z]))

    t = pd.DataFrame(rows)
    out = D / 'residual_diagnostics_20260924'
    out.mkdir(exist_ok=True)
    t.to_csv(out / 'per_gene.tsv', sep='\t', index=False)
    res = {}

    print('MODEL CHECK 1 -- SHAPE.  Does the standardized squared residual still')
    print('depend on log v?  Under Var(eps) = sigma^2 v it must not.\n')
    for ch in ('allelic', 'total'):
        sub = t[t.channel == ch]
        P = np.vstack(pooled[ch])
        sl, ic, r, p, se = sps.linregress(P[:, 0], P[:, 1])
        npos = int((sub.slope_z2_on_logv > 0).sum())
        sgn = sps.binomtest(npos, len(sub), 0.5).pvalue
        print(f'  {ch:8s} pooled slope {sl:+.4f} +/- {se:.4f}  (p={p:.2g}, '
              f'n={len(P)} donor-gene points)')
        print(f'           per-gene slope positive in {npos}/{len(sub)} genes, '
              f'sign p={sgn:.3g};  median per-gene slope {sub.slope_z2_on_logv.median():+.4f}')
        res[f'shape_{ch}'] = dict(pooled_slope=float(sl), pooled_se=float(se),
                                  pooled_p=float(p), n_points=int(len(P)),
                                  n_genes_positive=npos, n_genes=int(len(sub)),
                                  sign_p=float(sgn))

    print('\nMODEL CHECK 2 -- TAILS.  Excess kurtosis of the standardized residual;')
    print('0 is Gaussian, positive is heavy-tailed and makes an F reference liberal.\n')
    for ch in ('allelic', 'total'):
        sub = t[t.channel == ch]
        P = np.vstack(pooled[ch])
        pk = float(sps.kurtosis(P[:, 2], fisher=True))
        k = sub['kurtosis']
        med, q1, q3 = k.median(), k.quantile(.25), k.quantile(.75)
        npos = int((k > 0).sum())
        print(f'  {ch:8s} pooled excess kurtosis {pk:+.3f};  per-gene median '
              f'{med:+.3f}, IQR {q1:+.3f} .. {q3:+.3f}, positive in '
              f'{npos}/{len(sub)}')
        res[f'kurtosis_{ch}'] = dict(pooled=pk, per_gene_median=float(med),
                                     n_positive=npos, n_genes=int(len(sub)))

    (out / 'summary.json').write_text(json.dumps(res, indent=2, default=float))
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()

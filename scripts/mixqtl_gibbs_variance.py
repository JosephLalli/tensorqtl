"""mixQTL's estimator with Var(eps_i) = sigma^2 * v_i.

THE SUBSTITUTION. mixQTL's allelic error model is Eq 11,

    eps_asc ~ N(0, sigma^2 (1/y1 + 1/y2))     sigma^2 FREE

i.e. the delta-method Poisson/binomial variance of a log ratio, INFERRED
from the counts, carried with a freely fitted scale. Salmon's Gibbs draws
MEASURE that same variance directly. This replaces the inferred shape with
the measured one,

    eps_asc ~ N(0, sigma^2 v_i)               sigma^2 still FREE

and changes nothing else: same donor admission rule, same response
log(y1/y2) with no pseudocount, same through-origin design on h1 - h2, same
fold cap, same sigma^2 estimated from the weighted residuals on n - 1
degrees of freedom, same total channel, same inverse-variance meta-analysis,
same n > 15 normal-versus-t rule.

WHY IT IS DONE HERE AND NOT IN THE MODULE. tensorqtl/mixqtl_replication.py
is a faithful port in which every rule is pinned to the R source line it
encodes; the reference has no Gibbs option, so adding one there would
weaken that guarantee. asc_channel is short enough to mirror exactly, and
trc_channel and meta_analyze are reused unmodified, so the only thing that
differs from mixqtl_scan is the weight vector.

WHAT IT MEASURES. Because sigma^2 is fitted, a better-shaped weight vector
shrinks the weighted residual sum of squares, hence sigma, hence the
reported standard error. So unlike the known-variance case, the efficiency
gain of a better variance model is directly visible in the SE that mixQTL
itself reports, and the median SE ratio between the two arms is the answer.

Run: python3 scripts/mixqtl_gibbs_variance.py   (add WIDE=0 for published cutoffs)
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
D = '/mnt/ssd/lalli/brainvar_hapmix_deploy'
OUT = f'{D}/mixqtl_replication_20260919'
WIDE = os.environ.get('WIDE', '1') == '1'

sys.path.insert(0, HERE)
sys.path.insert(0, REPO)


def asc_channel_gibbs(y1, y2, v, X, asc_cutoff, asc_cap, weight_cap,
                      MX, apply_cap=True):
    """MX.asc_channel with w = 1/v in place of the harmonic sum.

    Mirrors tensorqtl/mixqtl_replication.py:asc_channel line for line. The
    admission rule is still on COUNTS, so the donor set is identical to the
    published arm's and only the weighting differs.
    """
    y1, y2 = np.asarray(y1, float), np.asarray(y2, float)
    v, X = np.asarray(v, float), np.asarray(X, float)
    P = X.shape[1]
    beta, se = np.full(P, np.nan), np.full(P, np.nan)
    passed = ((y1 >= asc_cutoff) & (y2 >= asc_cutoff)
              & (y1 <= asc_cap) & (y2 <= asc_cap) & (v > 0))
    n = int(passed.sum())
    if n <= 2:
        return dict(beta=beta, se=se, sample_size=n, mono=np.ones(P, bool))
    Xk = X[passed]
    with np.errstate(divide='ignore', invalid='ignore'):
        resp = np.log(y1[passed] / y2[passed])
    w = 1.0 / v[passed]
    if apply_cap:
        w, _, _ = MX.apply_weight_cap(w, n, weight_cap)
    mono = Xk.var(axis=0) == 0
    if (~mono).any():
        b, s = MX._simple_regression_through_origin(resp, Xk[:, ~mono], w)
        beta[~mono], se[~mono] = b, s
    return dict(beta=beta, se=se, sample_size=n, mono=mono)


def main():
    import tensorqtl.hapmixqtl as HM
    from tensorqtl import mixqtl_replication as MX
    from compare_mixqtl_replication import load_inputs, gene_variant_index

    if WIDE:
        cut, cap, trc = 1e-6, np.inf, 0.0
        tag, lab = '_wide', 'WIDE (full informative set)'
    else:
        cut, cap, trc = MX.ASC_CUTOFF, MX.ASC_CAP, MX.TRC_CUTOFF
        tag, lab = '', 'PUBLISHED cutoffs'
    print(f'cutoffs: {lab}')

    I = load_inputs()
    genes, keep = I['genes'], I['keep']
    A, T, Va, Vt, _ = HM.compute_summaries_from_gibbs(
        I['YL'], I['YR'], yT=I['YT'], count_noise=True)
    Va = Va[:, keep]
    y1, y2, yt = MX.summaries_from_gibbs_posterior_mean(I['YL'], I['YR'], I['YT'])
    y1, y2, yt = y1[:, keep], y2[:, keep], yt[:, keep]

    rows = []
    for j, g in enumerate(genes):
        vsel = gene_variant_index(I, g)
        if vsel.size == 0:
            continue
        h1 = I['xL'][I['idx']][:, keep][vsel].T.astype(float)
        h2 = I['xR'][I['idx']][:, keep][vsel].T.astype(float)
        h1, h2 = np.nan_to_num(h1, nan=0.5), np.nan_to_num(h2, nan=0.5)
        Xasc, Xtrc = h1 - h2, (h1 + h2) / 2.0
        off, _ = MX.covariate_offset(yt[j], I['lib_size'], I['cov_df'].values)

        trc_out = MX.trc_channel(yt[j], I['lib_size'], Xtrc, off, trc)
        pub = MX.asc_channel(y1[j], y2[j], Xasc, cut, MX.WEIGHT_CAP, cap)
        gib = asc_channel_gibbs(y1[j], y2[j], Va[j], Xasc, cut, cap,
                                MX.WEIGHT_CAP, MX, apply_cap=True)
        gib_nc = asc_channel_gibbs(y1[j], y2[j], Va[j], Xasc, cut, cap,
                                   MX.WEIGHT_CAP, MX, apply_cap=False)

        mpub = MX.meta_analyze(trc_out, pub)['meta']
        mgib = MX.meta_analyze(trc_out, gib)['meta']
        v = I['vdf'].iloc[I['idx']].iloc[vsel]
        rows.append(pd.DataFrame({
            'gene': g, 'variant_id': v.index.astype(str),
            'asc_pub_beta': pub['beta'], 'asc_pub_se': pub['se'],
            'asc_gib_beta': gib['beta'], 'asc_gib_se': gib['se'],
            'asc_gib_nocap_beta': gib_nc['beta'], 'asc_gib_nocap_se': gib_nc['se'],
            'meta_pub_beta': mpub['beta'], 'meta_pub_se': mpub['se'],
            'meta_gib_beta': mgib['beta'], 'meta_gib_se': mgib['se'],
            'meta_pub_method': mpub['method'], 'meta_gib_method': mgib['method'],
            'n_asc': pub['sample_size'],
        }))
    m = pd.concat(rows, ignore_index=True)
    m.to_parquet(f'{OUT}/mixqtl_gibbs_variance{tag}.parquet')

    def summarise(bp, sp, bg, sg, label):
        ok = (np.isfinite(bp) & np.isfinite(bg) & (sp > 0) & (sg > 0))
        r = pearsonr(bp[ok], bg[ok])[0]
        se_ratio = (sg[ok] / sp[ok])
        d = dict(n=int(ok.sum()), beta_r=float(r),
                 median_se_ratio_gibbs_over_published=float(np.median(se_ratio)),
                 iqr=[float(np.percentile(se_ratio, 25)),
                      float(np.percentile(se_ratio, 75))],
                 median_abs_beta_ratio=float(np.median(np.abs(bg[ok]))
                                             / np.median(np.abs(bp[ok]))))
        print(f'\n{label}  (n={d["n"]:,})')
        print(f'  beta correlation published vs Gibbs   {d["beta_r"]:.4f}')
        print(f'  median SE ratio  Gibbs / published    '
              f'{d["median_se_ratio_gibbs_over_published"]:.4f}'
              f'   IQR {d["iqr"][0]:.3f}-{d["iqr"][1]:.3f}')
        print(f'  median |beta| ratio                   {d["median_abs_beta_ratio"]:.4f}')
        return d, se_ratio

    res = {}
    res['allelic_capped'], _ = summarise(
        m.asc_pub_beta.values, m.asc_pub_se.values,
        m.asc_gib_beta.values, m.asc_gib_se.values,
        'ALLELIC CHANNEL, both under mixQTL fold cap')
    res['allelic_gibbs_uncapped'], _ = summarise(
        m.asc_pub_beta.values, m.asc_pub_se.values,
        m.asc_gib_nocap_beta.values, m.asc_gib_nocap_se.values,
        'ALLELIC CHANNEL, Gibbs arm uncapped vs published capped')
    res['combined'], _ = summarise(
        m.meta_pub_beta.values, m.meta_pub_se.values,
        m.meta_gib_beta.values, m.meta_gib_se.values,
        'COMBINED (meta with the unchanged total channel)')

    # per-gene paired test on the allelic SE, since variants correlate within
    # a gene and a pooled median would overstate the evidence
    pg = []
    for g, d in m.groupby('gene'):
        ok = (d.asc_pub_se > 0) & (d.asc_gib_se > 0)
        if ok.sum() >= 20:
            pg.append(float(np.median(d.asc_gib_se[ok] / d.asc_pub_se[ok])))
    pg = np.array(pg)
    res['per_gene_allelic_se_ratio'] = dict(
        n_genes=int(len(pg)), median=float(np.median(pg)),
        genes_below_1=int((pg < 1).sum()),
        wilcoxon_p_vs_1=float(wilcoxon(pg - 1.0).pvalue) if len(pg) > 5 else None)
    print(f'\nPER-GENE allelic SE ratio (Gibbs/published), {len(pg)} genes')
    print(f'  median {np.median(pg):.4f};  below 1 in {int((pg<1).sum())}/{len(pg)}'
          f';  Wilcoxon vs 1 p={res["per_gene_allelic_se_ratio"]["wilcoxon_p_vs_1"]:.4g}')

    res['cutoffs'] = dict(asc_cutoff=cut, asc_cap=float(cap), trc_cutoff=trc,
                          wide=WIDE)
    json.dump(res, open(f'{OUT}/mixqtl_gibbs_variance{tag}.json', 'w'), indent=1)


if __name__ == '__main__':
    main()

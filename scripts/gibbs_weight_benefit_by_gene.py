"""Where do Gibbs weights buy precision, gene by gene?

For each of the corrected null store's 100 genes and each channel, the
precision gain of Gibbs weights over unit weights: the realized sd of the null
slope across the 200 stored permutations under unit weights divided by that
under Gibbs weights, median over the gene's tested variants. Above 1 the Gibbs
weights estimate the slope more precisely; 1.41 would mean they halve its
variance.

total    from the stored draws: corrected_null_store keep arm (Gibbs) and
         total_channel_decomposition unit_weights arm; same values,
         covariates, genotype-PC rule and permutations
allelic  recomputed here, since the allelic fit is a through-origin weighted
         regression with no covariates: slope = sum(w a s) / sum(w s^2) over
         the permuted, L/R-swapped records (records_signflip, the stream of
         corrected_null_store), zero-haplotype pairs dropped. Unit weights
         give every informative donor weight 1.

Gene features, to see what the helped genes have in common: median total CPM,
median allele-resolved reads, informative donors, the spread of the weights
(max / median of 1/v over informative donors), Kish effective n as a share of
donors ((sum w)^2 / sum w^2 / n), and the number of "high-variance donors",
whose Gibbs variance exceeds 10x the gene's median.
No randomness beyond the stored permutation stream.
"""
import contextlib
import io
import json

import numpy as np
import pandas as pd

import compare_mixqtl_replication as CM
import corrected_null_store as CNS
from tensorqtl.hapmixqtl import summaries_from_point_estimates
from total_channel_se_accuracy import moments

D = CNS.D
DEC = D / 'total_channel_decomposition_20260926'
OUT = D / 'gibbs_weight_benefit_by_gene_20260926'
EPS, N_DRAW = 1e-12, 200


def features(V):
    """Per gene, over donors with V > 0."""
    rows = []
    for v in V:
        m = v > EPS
        w = 1 / v[m]
        rows.append(dict(n_informative=int(m.sum()),
                         weight_spread=float(w.max() / np.median(w)) if m.any() else np.nan,
                         kish_share=float(w.sum() ** 2 / (w ** 2).sum() / m.sum()) if m.any() else np.nan,
                         n_high_variance=int((v[m] > 10 * np.median(v[m])).sum()) if m.any() else 0))
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(exist_ok=True)
    genes = (CNS.OUT / 'genes.txt').read_text().split()
    with contextlib.redirect_stdout(io.StringIO()):
        I = CM.load_point_estimate_inputs(gene_list=str(CNS.OUT / 'genes.txt'),
                                          regions=str(CNS.OUT / 'regions.bed'))
    keep = I['keep']
    A, T, Va, Vt, _ = summaries_from_point_estimates(I['pL'], I['pR'], I['pT'], I['eff_lib'],
                                                     I['YL'], I['YR'], I['YT'])
    A, Va, Vt = A[:, keep], Va[:, keep], Vt[:, keep]
    pL, pR, pT = I['pL'][:, keep], I['pR'][:, keep], I['pT'][:, keep]
    Va = np.where((pL < 0.5) ^ (pR < 0.5), 0.0, Va)                  # zero-haplotype drop
    order = I['order']
    perms = np.load(CNS.OLD / 'permutations.npz')
    P, F = perms['perms'][:N_DRAW], perms['flips'][:N_DRAW].astype(float)

    # ---- allelic: recomputed on the stored stream ---------------------------
    rows = []
    for k, g in enumerate(genes):
        vsel = I['idx'][CM.gene_variant_index(I, g)]
        if not len(vsel):
            continue
        S = (I['xL'][vsel] - I['xR'][vsel]).astype(float)           # [V, N]
        inf = Va[k] > EPS
        w_g = np.where(inf, 1 / np.where(inf, Va[k], 1.0), 0.0)
        w_u = inf.astype(float)
        a = np.where(inf, A[k], 0.0)
        Ap = a[P] * F                                               # [P, N] permuted, swapped
        sd = {}
        for lab, w in (('gibbs', w_g), ('unit', w_u)):
            Wp = w[P]
            num = (Wp * Ap) @ S.T
            den = Wp @ (S * S).T
            with np.errstate(invalid='ignore', divide='ignore'):
                b = num / den
            ok = (den > 0).all(0)
            sd[lab] = np.where(ok, b.std(0, ddof=1), np.nan)
        r = sd['unit'] / sd['gibbs']
        rows.append(dict(gene=g, allelic_gain=float(np.nanmedian(r)) if np.isfinite(r).any() else np.nan))
    allelic = pd.DataFrame(rows).set_index('gene')

    # ---- total: from the stored draws ---------------------------------------
    ig, _, real_g = moments(sorted((CNS.OUT / 'draws').glob('keep_*.parquet')))
    _, _, real_u = moments(sorted((DEC / 'draws').glob('unit_weights_*.parquet')))
    ok = (real_g > 0) & (real_u > 0)
    total = pd.DataFrame(dict(gene=ig.get_level_values(0)[ok], r=(real_u / real_g)[ok])) \
        .groupby('gene').r.median().rename('total_gain')

    es = pd.read_csv(CNS.D / 'cache' / 'gibbs_56b63c3b37ed5df8' / 'point_estimates' / 'edger' / 'edger_samples.tsv',
                     sep='\t', dtype={'sample': str}).set_index('sample').loc[order]
    ft = features(Vt).add_prefix('total_'); ft.index = genes
    fa = features(Va).add_prefix('allelic_'); fa.index = genes
    tab = pd.concat([allelic, total, ft, fa], axis=1)
    tab['median_cpm'] = np.median(pT / es.eff_lib_size.values[None, :] * 1e6, axis=1)
    tab['median_allele_reads'] = np.median(pL + pR, axis=1)
    tab.to_csv(OUT / 'per_gene.tsv', sep='\t')

    summ = {}
    for ch in ('allelic', 'total'):
        x = tab[f'{ch}_gain'].dropna()
        summ[ch] = dict(n_genes=int(len(x)), quantiles_10_25_50_75_90=np.quantile(x, [.1, .25, .5, .75, .9]).tolist(),
                        max=float(x.max()), n_gain_ge_1_2=int((x >= 1.2).sum()), n_gain_ge_1_5=int((x >= 1.5).sum()),
                        spearman_with={f: float(tab[[f'{ch}_gain', f]].dropna().corr('spearman').iloc[0, 1])
                                       for f in (f'{ch}_weight_spread', f'{ch}_kish_share',
                                                 f'{ch}_n_high_variance', 'median_cpm', 'median_allele_reads')})
        print(f'{ch}: gain quantiles 10/25/50/75/90 {np.round(summ[ch]["quantiles_10_25_50_75_90"], 3)}, '
              f'max {summ[ch]["max"]:.2f}, genes >= 1.2: {summ[ch]["n_gain_ge_1_2"]}, >= 1.5: {summ[ch]["n_gain_ge_1_5"]}')
        print('   Spearman with gene features:', {k: round(v, 2) for k, v in summ[ch]['spearman_with'].items()})
        cols = [f'{ch}_gain', f'{ch}_n_informative', f'{ch}_weight_spread', f'{ch}_kish_share',
                f'{ch}_n_high_variance', 'median_cpm', 'median_allele_reads']
        print(tab.sort_values(f'{ch}_gain', ascending=False)[cols].head(10).round(2).to_string())
    (OUT / 'summary.json').write_text(json.dumps(summ, indent=1))


if __name__ == '__main__':
    main()

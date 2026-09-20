"""hapmixQTL against mixQTL on the OBSERVED pass only. No permutation anywhere.

Everything here is a deterministic function of the data: no null draws, no
permutation p-values, no type-I. If the permutation scheme is wrong, nothing
in this file changes. That is the point -- it isolates how the two estimators
differ in what they actually report.

The two arms turn out to be directly comparable on effect size, which an
earlier version of the report denied. Both form the allelic response in
NATURAL log:

    hapmixQTL   mean over draws of  log(YL + 0.5) - log(YR + 0.5)
    mixQTL                          log(YL_bar)   - log(YR_bar)

`tensorqtl/hapmixqtl.py:406` uses np.log, and log2 appears nowhere in that
module -- the log2 migration is still pending. So the responses differ by the
pseudocount and by mean-of-log-draws versus log-of-mean, not by log base, and
both betas are a natural-log allelic fold change per haplotype-dosage unit.

Reported per matched (gene, variant): the two betas, their standard errors,
and the resulting statistics. Aggregated: correlation, regression slope of
one on the other, sign concordance, and the ratio of standard errors. Also
the cutoff arithmetic -- how many donors each arm admits, and which channel
mixQTL's meta-analysis actually used.
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
D = '/mnt/ssd/lalli/brainvar_hapmix_deploy'
OUT = f'{D}/mixqtl_replication_20260919'
MATCH_CUTOFFS = os.environ.get('MATCH_CUTOFFS', '0') == '1'
TAG = ('_hwe' if os.environ.get('HWE', '0') == '1' else '') \
      + ('_matched' if MATCH_CUTOFFS else '')

sys.path.insert(0, HERE)
sys.path.insert(0, REPO)


def mixqtl_all_variants(I, g, j, y1, y2, yt, MX):
    """mixQTL observed pass for one gene, every variant (not just the lead)."""
    from compare_mixqtl_replication import gene_variant_index
    vsel = gene_variant_index(I, g)
    if vsel.size == 0:
        return None
    keep = I['keep']
    h1 = I['xL'][I['idx']][:, keep][vsel].T.astype(float)
    h2 = I['xR'][I['idx']][:, keep][vsel].T.astype(float)
    out = MX.mixqtl_scan(y1[j], y2[j], yt[j], I['lib_size'], h1, h2,
                         covariates=I['cov_df'].values)
    v = I['vdf'].iloc[I['idx']].iloc[vsel]
    return pd.DataFrame({
        'gene': g,
        'variant_id': v.index.astype(str),
        'mx_beta': out['meta']['beta'],
        'mx_se': out['meta']['se'],
        'mx_stat': out['meta']['stat'],
        'mx_method': out['meta']['method'],
        'mx_beta_asc': out['asc']['beta'],
        'mx_se_asc': out['asc']['se'],
        'mx_beta_trc': out['trc']['beta'],
        'mx_se_trc': out['trc']['se'],
        'mx_n_asc': out['asc']['sample_size'],
        'mx_n_trc': out['trc']['sample_size'],
    })


def main():
    import contextlib
    import io
    import tensorqtl.hapmixqtl as HM
    from tensorqtl import mixqtl_replication as MX
    from compare_mixqtl_replication import load_inputs, WIN

    I = load_inputs()
    if os.environ.get('HWE', '0') == '1':
        from make_hwe_filtered_variants import apply_hwe_filter
        I = apply_hwe_filter(I)
        print(f"[HWE-filtered variant set: dropped "
              f"{I['n_dropped_by_hwe']:,} variants]")
    else:
        print('[unfiltered variant set]')
    genes, order, keep = I['genes'], I['order'], I['keep']
    A, T, Va, Vt, _ = HM.compute_summaries_from_gibbs(
        I['YL'], I['YR'], yT=I['YT'], count_noise=True)
    A, T, Va, Vt = A[:, keep], T[:, keep], Va[:, keep], Vt[:, keep]
    y1, y2, yt = MX.summaries_from_gibbs_posterior_mean(I['YL'], I['YR'], I['YT'])
    y1, y2, yt = y1[:, keep], y2[:, keep], yt[:, keep]

    idx, vdf = I['idx'], I['vdf']
    v = vdf.iloc[idx]
    mk = lambda M: pd.DataFrame(M, index=genes, columns=order)
    fr = lambda M: pd.DataFrame(M[idx], index=v.index, columns=order)

    # MATCHED DONOR SET. mixQTL's count cutoffs applied to the hapmixQTL arm
    # too, from the SAME posterior means the mixQTL arm consumes, so the two
    # donor sets are identical by construction rather than approximately.
    keep_a_df = keep_t_df = None
    if MATCH_CUTOFFS:
        ka, kt = HM.count_cutoff_masks(y1, y2, yt, asc_cutoff=MX.ASC_CUTOFF,
                                       asc_cap=MX.ASC_CAP,
                                       trc_cutoff=MX.TRC_CUTOFF)
        keep_a_df, keep_t_df = mk(ka), mk(kt)
        print(f'[matched cutoffs: asc {MX.ASC_CUTOFF:g}-{MX.ASC_CAP:g}, '
              f'trc >= {MX.TRC_CUTOFF:g}] allelic admits '
              f'{ka.sum():,}/{ka.size:,} donor-gene pairs '
              f'({ka.sum() / ka.size:.1%}), total admits '
              f'{kt.sum():,}/{kt.size:,} ({kt.sum() / kt.size:.1%})')
    else:
        print('[hapmixQTL arm unrestricted: its own informative-donor set]')

    tmp = f'{OUT}/_nominal_tmp{TAG}'
    os.makedirs(tmp, exist_ok=True)
    for f in os.listdir(tmp):            # never mix runs in one glob
        if f.endswith('.parquet'):
            os.remove(f'{tmp}/{f}')
    with contextlib.redirect_stdout(io.StringIO()):
        HM.map_nominal(fr(I['dos']), v[['chrom', 'pos']], mk(A), mk(T),
                       mk(Va), mk(Vt), I['gp'].loc[genes][['chr', 'pos']],
                       xL_df=fr(I['xL']), xR_df=fr(I['xR']), prefix='hm',
                       covariates_df=I['cov_df'], ase_covariates_df=None,
                       window=WIN, output_dir=tmp, verbose=False,
                       keep_a_df=keep_a_df, keep_t_df=keep_t_df)
    hm = pd.concat([pd.read_parquet(f'{tmp}/{f}') for f in os.listdir(tmp)
                    if f.endswith('.parquet')], ignore_index=True)
    gcol = 'phenotype_id' if 'phenotype_id' in hm.columns else 'gene_id'
    hm = hm.rename(columns={gcol: 'gene', 'slope': 'hm_beta',
                            'slope_se': 'hm_se'})
    hm['variant_id'] = hm['variant_id'].astype(str)
    hm['hm_stat'] = (hm.hm_beta / hm.hm_se) ** 2

    mx = pd.concat([d for j, g in enumerate(genes)
                    if (d := mixqtl_all_variants(I, g, j, y1, y2, yt, MX)) is not None],
                   ignore_index=True)

    m = hm[['gene', 'variant_id', 'hm_beta', 'hm_se', 'hm_stat']].merge(
        mx, on=['gene', 'variant_id'], how='inner')
    m = m[np.isfinite(m.hm_beta) & np.isfinite(m.mx_beta)
          & np.isfinite(m.hm_se) & np.isfinite(m.mx_se)]
    m.to_parquet(f'{OUT}/observed_matched_variants{TAG}.parquet')

    # aggregate agreement
    r_p = pearsonr(m.hm_beta, m.mx_beta)
    r_s = spearmanr(m.hm_beta, m.mx_beta)
    slope = float(np.polyfit(m.hm_beta, m.mx_beta, 1)[0])
    sign_ok = float((np.sign(m.hm_beta) == np.sign(m.mx_beta)).mean())
    se_ratio = (m.mx_se / m.hm_se)

    per_gene = m.groupby('gene').apply(
        lambda d: pd.Series({
            'n_var': len(d),
            'beta_r': pearsonr(d.hm_beta, d.mx_beta)[0] if len(d) > 3 else np.nan,
            'sign_conc': (np.sign(d.hm_beta) == np.sign(d.mx_beta)).mean(),
            'median_se_ratio': (d.mx_se / d.hm_se).median(),
            'hm_lead': d.loc[d.hm_stat.idxmax(), 'variant_id'],
            'mx_lead': d.loc[d.mx_stat.abs().idxmax(), 'variant_id'],
            'mx_n_asc': d.mx_n_asc.iloc[0],
        }), include_groups=False).reset_index()
    per_gene['lead_same'] = per_gene.hm_lead == per_gene.mx_lead
    per_gene.to_csv(f'{OUT}/observed_per_gene{TAG}.tsv', sep='\t', index=False)

    # IDENTITY CHECK. Matched means matched: the donors hapmixQTL's allelic
    # mask admits per gene must be exactly the donors mixQTL counted in
    # n_asc. If they differ the arms are not on the same donor set and every
    # number below is void, so this raises rather than warns.
    strat = {}
    if MATCH_CUTOFFS:
        admitted = pd.Series(keep_a_df.sum(axis=1), index=genes)
        got = per_gene.set_index('gene').mx_n_asc
        bad = {g: (int(admitted[g]), int(got[g])) for g in got.index
               if int(admitted[g]) != int(got[g])}
        if bad:
            raise SystemExit(f'donor sets not matched for {len(bad)} genes '
                             f'(mask vs mixQTL n_asc): {dict(list(bad.items())[:5])}')
        print(f'donor-set identity check passed on {len(got)} genes')

    # mixQTL falls back to total-counts-only below META_N_CUTOFF donors while
    # hapmixQTL keeps its allelic channel, so the cutoffs cannot match that.
    # Report the two strata apart: the 'meta' genes are the apples-to-apples.
    for method, d in m.groupby('mx_method'):
        if len(d) < 4:
            continue
        strat[method] = dict(
            n_variants=int(len(d)), n_genes=int(d.gene.nunique()),
            beta_pearson_r=float(pearsonr(d.hm_beta, d.mx_beta)[0]),
            beta_sign_concordance=float((np.sign(d.hm_beta) == np.sign(d.mx_beta)).mean()),
            median_se_ratio_mx_over_hm=float((d.mx_se / d.hm_se).median()),
            median_abs_beta_ratio_mx_over_hm=float(
                (d.mx_beta.abs() / d.hm_beta.abs().replace(0, np.nan)).median()),
        )
    pg_meta = per_gene[per_gene.gene.isin(
        m[m.mx_method == 'meta'].gene.unique())]
    if len(pg_meta):
        strat.setdefault('meta', {})['lead_agreement'] = float(pg_meta.lead_same.mean())
        strat['meta']['median_per_gene_beta_r'] = float(pg_meta.beta_r.median())

    res = dict(
        n_matched_variants=int(len(m)), n_genes=int(m.gene.nunique()),
        beta_pearson_r=float(r_p[0]), beta_spearman_r=float(r_s[0]),
        beta_regression_slope_mx_on_hm=slope,
        beta_sign_concordance=sign_ok,
        median_abs_beta_hm=float(m.hm_beta.abs().median()),
        median_abs_beta_mx=float(m.mx_beta.abs().median()),
        median_se_hm=float(m.hm_se.median()),
        median_se_mx=float(m.mx_se.median()),
        median_se_ratio_mx_over_hm=float(se_ratio.median()),
        iqr_se_ratio=[float(se_ratio.quantile(.25)), float(se_ratio.quantile(.75))],
        median_stat_hm=float(m.hm_stat.median()),
        median_stat_mx=float((m.mx_stat ** 2).median()),
        lead_agreement=float(per_gene.lead_same.mean()),
        median_per_gene_beta_r=float(per_gene.beta_r.median()),
        median_per_gene_sign_conc=float(per_gene.sign_conc.median()),
        mixqtl_method_counts=m.mx_method.value_counts().to_dict(),
        mixqtl_median_n_asc=float(m.mx_n_asc.median()),
        mixqtl_median_n_trc=float(m.mx_n_trc.median()),
        matched_cutoffs=bool(MATCH_CUTOFFS),
        by_mixqtl_channel=strat,
    )
    json.dump(res, open(f'{OUT}/observed_comparison{TAG}.json', 'w'), indent=1)
    for k, val in res.items():
        print(f'{k:36s} {val}')


if __name__ == '__main__':
    main()

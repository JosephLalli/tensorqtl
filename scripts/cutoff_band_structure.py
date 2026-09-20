"""Are the donor-gene pairs mixQTL's cutoffs EXCLUDE different in kind?

mixQTL's Methods justify the upper cap as an alignment-artifact guard: "very
large allele-specific counts to be likely alignment artifacts". On Salmon
posterior means that rationale has been asserted not to transfer -- a
posterior mean above 1000 is a well-expressed gene -- but asserting is not
measuring. This measures it two ways.

A. PAIR-LEVEL STRUCTURE, by band. Raw |a| is useless here: it shrinks with
   count by pure measurement precision, so the excluded pairs would look
   "different" even if nothing were wrong with them. The quantity that is
   not confounded that way is EXCESS DISPERSION, var(a) - mean(Va): the
   allelic scatter left over after the Gibbs draws' own predicted variance is
   subtracted. An artifact band carries excess scatter the draws do not
   predict; a merely well-measured band does not.

B. ESTIMATOR-LEVEL, the decisive test. Two hapmixQTL runs on identical
   variants differing only in the mask, compared on the ALLELIC slope, which
   is where the cap acts:
     artifact hypothesis   -> dropping the pairs SHIFTS slope_a (per-gene
                              regression slope of cut on full away from 1)
     well-expressed        -> slope stays 1, only slope_a_se inflates
   With a RANDOM-MASK CONTROL that drops the same number of informative
   pairs per gene at seed 42. Without it, "the cap changed things" cannot be
   separated from "the cap removed three quarters of the data". If the
   random mask reproduces the cap's effect, the cap's identity is irrelevant
   and it is selecting on nothing but count.

Reference bias, the sharpest artifact test, is NOT run: it needs the
per-feature-SNP ref/alt orientation (`sign`), which compare_mixqtl_replication
.load_inputs does not carry. Not reconstructed here.

Writes cutoff_band_structure.json next to the other outputs.
"""
import contextlib
import io
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
D = '/mnt/ssd/lalli/brainvar_hapmix_deploy'
OUT = f'{D}/mixqtl_replication_20260919'
SEED = 42

sys.path.insert(0, HERE)
sys.path.insert(0, REPO)


def read_slopes(tmp):
    d = pd.concat([pd.read_parquet(f'{tmp}/{f}') for f in os.listdir(tmp)
                   if f.endswith('.parquet')], ignore_index=True)
    d = d.rename(columns={'phenotype_id': 'gene'})
    d['variant_id'] = d.variant_id.astype(str)
    return d[['gene', 'variant_id', 'slope_a', 'slope_a_se',
              'slope_t', 'slope_t_se', 'slope', 'slope_se']]


def main():
    import tensorqtl.hapmixqtl as HM
    from tensorqtl import mixqtl_replication as MX
    from compare_mixqtl_replication import load_inputs, WIN

    I = load_inputs()
    genes, order, keep = I['genes'], I['order'], I['keep']
    A, T, Va, Vt, _ = HM.compute_summaries_from_gibbs(
        I['YL'], I['YR'], yT=I['YT'], count_noise=True)
    A, T, Va, Vt = A[:, keep], T[:, keep], Va[:, keep], Vt[:, keep]
    y1, y2, yt = MX.summaries_from_gibbs_posterior_mean(I['YL'], I['YR'], I['YT'])
    y1, y2, yt = y1[:, keep], y2[:, keep], yt[:, keep]

    # ---------------- A. pair-level structure by band ----------------
    informative = (y1 + y2) > 0
    fails_floor = (y1 < MX.ASC_CUTOFF) | (y2 < MX.ASC_CUTOFF)
    fails_ceil = (y1 > MX.ASC_CAP) | (y2 > MX.ASC_CAP)
    bands = {
        'below_floor_only': informative & fails_floor & ~fails_ceil,
        'in_band': informative & ~fails_floor & ~fails_ceil,
        'above_ceiling_only': informative & ~fails_floor & fails_ceil,
        'fails_both_ends': informative & fails_floor & fails_ceil,
    }
    band_stats = {}
    for name, msk in bands.items():
        if msk.sum() < 10:
            band_stats[name] = dict(n=int(msk.sum()), note='too few to summarise')
            continue
        a, va, tot = A[msk], Va[msk], yt[msk]
        var_a, mean_va = float(np.var(a, ddof=1)), float(np.mean(va))
        band_stats[name] = dict(
            n=int(msk.sum()),
            median_total_counts=float(np.median(tot)),
            median_abs_a=float(np.median(np.abs(a))),
            mean_Va=mean_va,
            median_Va=float(np.median(va)),
            var_a=var_a,
            # the un-confounded quantity: scatter the draws do NOT predict
            excess_dispersion=var_a - mean_va,
            excess_ratio=var_a / mean_va if mean_va > 0 else float('nan'),
        )

    # ---------------- B. estimator-level, allelic slope ----------------
    idx, vdf = I['idx'], I['vdf']
    v = vdf.iloc[idx]
    mk = lambda M: pd.DataFrame(M, index=genes, columns=order)
    fr = lambda M: pd.DataFrame(M[idx], index=v.index, columns=order)

    keep_cut, _ = HM.count_cutoff_masks(y1, y2, yt, asc_cutoff=MX.ASC_CUTOFF,
                                        asc_cap=MX.ASC_CAP)
    # RANDOM CONTROL: same number of informative pairs retained per gene.
    rng = np.random.RandomState(SEED)
    keep_rnd = np.zeros_like(keep_cut)
    for j in range(keep_cut.shape[0]):
        inf_ix = np.flatnonzero(informative[j])
        k = int(keep_cut[j].sum())
        if k and len(inf_ix):
            keep_rnd[j, rng.choice(inf_ix, size=min(k, len(inf_ix)),
                                   replace=False)] = True

    runs = {}
    for label, ka in (('cut', keep_cut), ('rnd', keep_rnd), ('full', None)):
        tmp = f'{OUT}/_band_tmp_{label}'
        os.makedirs(tmp, exist_ok=True)
        for f in os.listdir(tmp):
            if f.endswith('.parquet'):
                os.remove(f'{tmp}/{f}')
        print(f'  map_nominal [{label}]'
              f"{'' if ka is None else f' admitting {ka.sum():,} allelic pairs'}",
              flush=True)
        with contextlib.redirect_stdout(io.StringIO()):
            HM.map_nominal(fr(I['dos']), v[['chrom', 'pos']], mk(A), mk(T),
                           mk(Va), mk(Vt), I['gp'].loc[genes][['chr', 'pos']],
                           xL_df=fr(I['xL']), xR_df=fr(I['xR']), prefix='hm',
                           covariates_df=I['cov_df'], ase_covariates_df=None,
                           window=WIN, output_dir=tmp, verbose=False,
                           keep_a_df=None if ka is None else mk(ka))
        runs[label] = read_slopes(tmp)

    full = runs['full']
    cmp_out = {}
    for label in ('cut', 'rnd'):
        m = full.merge(runs[label], on=['gene', 'variant_id'],
                       suffixes=('_full', f'_{label}'))
        ok = (np.isfinite(m[f'slope_a_{label}']) & np.isfinite(m.slope_a_full)
              & (m[f'slope_a_se_{label}'] > 0) & (m.slope_a_se_full > 0))
        m = m[ok]
        # per-gene regression slope through the origin; the spread across
        # genes is the noise floor, since variants correlate within a gene
        per_gene = []
        for g, d in m.groupby('gene'):
            if len(d) < 20:
                continue
            x, yv = d.slope_a_full.values, d[f'slope_a_{label}'].values
            per_gene.append(float((x @ yv) / (x @ x)))
        per_gene = np.array(per_gene)
        cmp_out[label] = dict(
            n_variants=int(len(m)), n_genes=int(len(per_gene)),
            slope_a_pearson_r=float(pearsonr(m.slope_a_full, m[f'slope_a_{label}'])[0]),
            per_gene_slope_median=float(np.median(per_gene)),
            per_gene_slope_iqr=[float(np.percentile(per_gene, 25)),
                                float(np.percentile(per_gene, 75))],
            per_gene_slope_sd=float(np.std(per_gene, ddof=1)),
            mean_diff=float(np.mean(m[f'slope_a_{label}'] - m.slope_a_full)),
            median_se_ratio=float(np.median(m[f'slope_a_se_{label}']
                                            / m.slope_a_se_full)),
        )

    res = dict(cutoffs=dict(asc_cutoff=MX.ASC_CUTOFF, asc_cap=MX.ASC_CAP),
               seed=SEED, bands=band_stats, allelic_slope=cmp_out,
               reference_bias='not run: load_inputs carries no ref/alt orientation')
    json.dump(res, open(f'{OUT}/cutoff_band_structure.json', 'w'), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()

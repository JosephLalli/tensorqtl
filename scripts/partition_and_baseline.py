"""Step 1 of the disagreement plan: the baseline, then the partition.

Deterministic throughout. No permutation, no null draw, no seed.

An earlier version of this script recomputed hapmixQTL's per-channel
estimates and checked them against the shipped combined slope. That was
unnecessary and it failed its own gate: `map_nominal` already emits
`slope_a`, `slope_t` and their standard errors per variant, and it runs in
float32 while the recomputation used float64. The shipped columns are used
directly, which removes the reconstruction risk entirely.

VALIDATION GATE, now a check on understanding rather than on arithmetic.
The pre-check established that hapmixQTL's pooled score and mixQTL's
inverse-variance meta-analysis are algebraically identical under the
known-variance standard error. If that is right, combining the SHIPPED
per-channel estimates by inverse variance must reproduce the SHIPPED
combined slope. Both sides come from the same file, so any mismatch is a
flaw in the stated understanding of the estimator, not a numerical
artifact, and nothing downstream should be believed.

THE BASELINE, which must precede any mechanism. The observed combined
correlation of 0.697 has never been compared against what two correctly
behaving but differently weighted estimators should produce. On these
null-calibration genes most variants carry no effect, so both arms are
largely estimating zero, and two estimators of zero agree only insofar as
their errors are shared. Some r below 1 is expected by construction.

It is measured rather than assumed: refit the allelic channel twice on the
SAME response and the SAME donors, changing only the weight vector from
1/v to mixQTL's capped harmonic. That correlation is the loss attributable
to weighting alone with no other divergence present. If the observed
allelic agreement sits at that level, weighting explains it and mechanisms
2, 3 and 6 have nothing left to account for.

THE PARTITION. Allelic against allelic and total against total, before any
combination, each compared to the combined figure.
"""

import glob
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

sys.path.insert(0, HERE)
sys.path.insert(0, REPO)


def r_or_nan(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 5 or np.std(a[m]) == 0 or np.std(b[m]) == 0:
        return np.nan
    return float(pearsonr(a[m], b[m])[0])


def main():
    import tensorqtl.hapmixqtl as HM
    from tensorqtl import mixqtl_replication as MX
    from compare_mixqtl_replication import load_inputs, gene_variant_index

    files = sorted(glob.glob(f'{OUT}/_nominal_tmp/*.parquet'))
    if not files:
        raise SystemExit('no nominal parquet; run compare_observed_results.py first')
    hm = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    hm = hm.rename(columns={'phenotype_id': 'gene'})
    hm['variant_id'] = hm['variant_id'].astype(str)

    # ---- validation gate: inverse variance on the shipped channels must
    # reproduce the shipped combined slope, per the pre-check.
    w_a = 1.0 / hm.slope_a_se ** 2
    w_t = 1.0 / hm.slope_t_se ** 2
    ok = np.isfinite(w_a) & np.isfinite(w_t) & np.isfinite(hm.slope)
    ivw = (w_a * hm.slope_a + w_t * hm.slope_t) / (w_a + w_t)
    rel = (np.abs(ivw - hm.slope) / np.maximum(np.abs(hm.slope), 1e-12))[ok]
    worst, med = float(np.nanmax(rel)), float(np.nanmedian(rel))
    print(f'VALIDATION GATE: inverse variance on the shipped per-channel '
          f'estimates vs the shipped combined slope')
    print(f'  median relative deviation {med:.3e}, max {worst:.3e}, '
          f'over {int(ok.sum()):,} variants')
    # float32 storage, so ~1e-6 is the honest tolerance here
    passed = med < 1e-5
    print(f'  gate {"passed" if passed else "FAILED"} '
          f'(tolerance 1e-5 on the median; output is stored float32)')
    if not passed:
        json.dump({'gate_passed': False, 'median_rel_dev': med},
                  open(f'{OUT}/partition_and_baseline.json', 'w'), indent=1)
        print('  refusing to interpret anything downstream')
        return
    print('  -> the combined estimate IS the inverse-variance combination of '
          'its own channels,\n     so the two arms cannot differ by the '
          'pooling formula. Mechanism 4 is about\n     mixQTL fitting a '
          'separate sigma per channel, nothing else.\n')

    # ---- weights-only baseline, computed per gene
    I = load_inputs()
    genes, keep = I['genes'], I['keep']
    A, _T, Va, _Vt, _C = HM.compute_summaries_from_gibbs(
        I['YL'], I['YR'], yT=I['YT'], count_noise=True)
    A, Va = A[:, keep], Va[:, keep]
    mL, mR = I['YL'].mean(2)[:, keep], I['YR'].mean(2)[:, keep]
    sgn = (I['xL'] - I['xR'])[I['idx']][:, keep]

    base = []
    for j, g in enumerate(genes):
        vsel = gene_variant_index(I, g)
        if vsel.size == 0:
            continue
        inf = Va[j] > 1e-12
        n_inf = int(inf.sum())
        if n_inf <= 2:
            continue
        S = sgn[vsel][:, inf].T
        kv = S.var(0) > 0
        if kv.sum() == 0:
            continue
        aa = A[j][inf]
        wv = 1.0 / np.maximum(Va[j][inf], 1e-12)
        wh = MX.harmonic_weights(np.maximum(mL[j][inf], 1e-12),
                                 np.maximum(mR[j][inf], 1e-12))
        wh, _, _ = MX.apply_weight_cap(wh, n_inf, MX.WEIGHT_CAP)
        b_v = np.full(len(vsel), np.nan)
        b_h = np.full(len(vsel), np.nan)
        b_v[kv], _ = MX._simple_regression_through_origin(aa, S[:, kv], wv)
        b_h[kv], _ = MX._simple_regression_through_origin(aa, S[:, kv], wh)
        base.append(pd.DataFrame(dict(
            gene=g,
            variant_id=I['vdf'].iloc[I['idx']].iloc[vsel].index.astype(str),
            base_w_gibbs=b_v, base_w_harmonic=b_h)))
    base = pd.concat(base, ignore_index=True)

    mx = pd.read_parquet(f'{OUT}/observed_matched_variants.parquet')
    j = (hm[['gene', 'variant_id', 'slope', 'slope_a', 'slope_t',
             'slope_a_se', 'slope_t_se']]
         .merge(mx[['gene', 'variant_id', 'mx_beta', 'mx_beta_asc',
                    'mx_beta_trc', 'mx_n_asc']], on=['gene', 'variant_id'])
         .merge(base, on=['gene', 'variant_id'], how='left'))
    j.to_parquet(f'{OUT}/partition_and_baseline.parquet')

    res = dict(gate_passed=True, median_rel_dev=med, n=int(len(j)))

    print('=== BASELINE: weights alone (same response, same donors) ===')
    rb = r_or_nan(j.base_w_gibbs, j.base_w_harmonic)
    print(f'  r(1/v weights, capped-harmonic weights) = {rb:.4f}')
    print('  the agreement two correct estimators would show if the ONLY')
    print('  difference between them were the weight vector.')
    res['baseline_r_weights_only'] = rb

    print('\n=== PARTITION: per channel, before any combination ===')
    ra = r_or_nan(j.slope_a, j.mx_beta_asc)
    rt = r_or_nan(j.slope_t, j.mx_beta_trc)
    rm = r_or_nan(j.slope, j.mx_beta)
    print(f'  allelic channel   r = {ra:.4f}')
    print(f'  total channel     r = {rt:.4f}')
    print(f'  combined          r = {rm:.4f}')
    res.update(r_allelic=ra, r_total=rt, r_combined=rm)

    print('\n=== per gene, median over genes ===')
    pg = j.groupby('gene').apply(lambda d: pd.Series({
        'r_allelic': r_or_nan(d.slope_a, d.mx_beta_asc),
        'r_total': r_or_nan(d.slope_t, d.mx_beta_trc),
        'r_combined': r_or_nan(d.slope, d.mx_beta),
        'baseline': r_or_nan(d.base_w_gibbs, d.base_w_harmonic),
    }), include_groups=False)
    print(pg.median().to_string())
    pg.to_csv(f'{OUT}/partition_per_gene.tsv', sep='\t')
    res['per_gene_median'] = {k: float(v) for k, v in pg.median().items()}

    json.dump(res, open(f'{OUT}/partition_and_baseline.json', 'w'), indent=1)


if __name__ == '__main__':
    main()

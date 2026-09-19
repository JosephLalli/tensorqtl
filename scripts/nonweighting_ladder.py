"""With the weights equalised, how much of mixQTL does hapmixQTL replicate?

The baseline showed that changing the weight vector alone reproduces almost
all of the observed disagreement. This holds the weights FIXED at mixQTL's
and walks the remaining divergences one at a time, so whatever gap survives
is attributable to something other than Gibbs weighting.

Deterministic throughout: no permutation, no null, no seed.

ALLELIC CHANNEL. Weights are mixQTL's capped harmonic at every rung,
recomputed on each rung's own donor set because the fold cap is defined
relative to that set.

  rung 1  hapmixQTL donors, hapmixQTL response (mean over draws of the log
          ratio, kappa = 0.5)
  rung 2  mixQTL donors (the 5 <= y <= 5000 gate), same response
  rung 3  mixQTL donors, kappa dropped to 0 -- isolates the pseudocount
  rung 4  mixQTL donors, log of the ratio of means -- isolates the Jensen
          gap, and IS mixQTL's allelic channel

Rung 4 must reproduce mixQTL's shipped allelic beta exactly. That is the
validation check; if it does not, the ladder is not walking to the right
place and nothing on it means anything.

TOTAL CHANNEL. Weights are unweighted OLS at every rung, as mixQTL uses.

  rung 1  hapmixQTL response (mean over draws of log(YT/2 + kappa)), all 17
          covariates adjusted jointly
  rung 2  mixQTL response, log(YT_bar / 2 / lib_size), covariates unchanged
          -- isolates the library-size offset and the transform
  rung 3  mixQTL's two-step selected covariates pre-regressed as an offset
          -- IS mixQTL's total channel
"""

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


def rs_or_nan(a, b):
    """Spearman: reported alongside Pearson because a single near-singular
    fit can drive a Pearson correlation entirely, as one did here before the
    conditioning guard was added."""
    from scipy.stats import spearmanr
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 5:
        return np.nan
    return float(spearmanr(a[m], b[m])[0])


def residualize(y, X, C):
    """Jointly adjust response and predictors on covariates C (with intercept)."""
    Dz = np.column_stack([np.ones(len(y)), C])
    Q, _ = np.linalg.qr(Dz)
    y = y - Q @ (Q.T @ y)
    X = X - Q @ (Q.T @ X)
    return y, X


def main():
    import tensorqtl.hapmixqtl as HM
    from tensorqtl import mixqtl_replication as MX
    from compare_mixqtl_replication import load_inputs, gene_variant_index

    I = load_inputs()
    genes, keep = I['genes'], I['keep']
    A, T, Va, Vt, _ = HM.compute_summaries_from_gibbs(
        I['YL'], I['YR'], yT=I['YT'], count_noise=True)
    A, T, Va = A[:, keep], T[:, keep], Va[:, keep]
    mL = I['YL'].mean(2)[:, keep]
    mR = I['YR'].mean(2)[:, keep]
    mT = I['YT'].mean(2)[:, keep]
    sgn = (I['xL'] - I['xR'])[I['idx']][:, keep]
    dos = I['dos'][I['idx']][:, keep]
    C = I['cov_df'].values
    lib = I['lib_size']

    arows, trows = [], []
    for j, g in enumerate(genes):
        vsel = gene_variant_index(I, g)
        if vsel.size == 0:
            continue
        vids = I['vdf'].iloc[I['idx']].iloc[vsel].index.astype(str)
        S = sgn[vsel]                                   # [P, N]
        G = dos[vsel] / 2.0

        inf = Va[j] > 1e-12
        gate = ((mL[j] >= MX.ASC_CUTOFF) & (mR[j] >= MX.ASC_CUTOFF)
                & (mL[j] <= MX.ASC_CAP) & (mR[j] <= MX.ASC_CAP))

        def allelic(mask, resp):
            n = int(mask.sum())
            if n <= 2:
                return np.full(len(vsel), np.nan)
            w = MX.harmonic_weights(np.maximum(mL[j][mask], 1e-12),
                                    np.maximum(mR[j][mask], 1e-12))
            w, _, _ = MX.apply_weight_cap(w, n, MX.WEIGHT_CAP)
            X = S[:, mask].T
            kv = X.var(0) > 0
            out = np.full(len(vsel), np.nan)
            if kv.sum() and np.isfinite(resp[mask]).all():
                b, _ = MX._simple_regression_through_origin(
                    resp[mask], X[:, kv], w)
                out[kv] = b
            return out

        with np.errstate(divide='ignore', invalid='ignore'):
            a_nok = np.log(mL[j] + 1e-300) - np.log(mR[j] + 1e-300)
            a_logmean = np.log(np.where(mR[j] > 0, mL[j] / mR[j], np.nan))
            a_k0_meanlog = (np.log(I['YL'][j][keep] + 0.0 + 1e-300)
                            - np.log(I['YR'][j][keep] + 0.0 + 1e-300)).mean(1)

        arows.append(pd.DataFrame(dict(
            gene=g, variant_id=vids,
            r1_hm_donors=allelic(inf, A[j]),
            r2_mx_donors=allelic(gate, A[j]),
            r3_no_kappa=allelic(gate, a_k0_meanlog),
            r4_logmean=allelic(gate, a_logmean),
        )))

        # ---- total channel, unweighted throughout
        def total(resp, cov_mode):
            ok = np.isfinite(resp)
            if ok.sum() <= 3:
                return np.full(len(vsel), np.nan)
            y = resp[ok].copy()
            X = G[:, ok].T
            raw_ss = (X ** 2).sum(0)
            if cov_mode == 'joint':
                y, X = residualize(y, X, C[ok])
            elif cov_mode == 'mixqtl':
                off, _ = MX.covariate_offset(mT[j][ok], lib[ok], C[ok])
                y = y - off
            # Conditioning guard. After projecting out an intercept and 17
            # covariates from 92 donors, a variant can retain essentially no
            # residual genotype variance without being exactly constant.
            # Fitting it yields a astronomically large slope that then
            # dominates any correlation computed over the column. Require the
            # variant to keep a non-trivial fraction of its own variance.
            res_ss = (X ** 2).sum(0)
            with np.errstate(divide='ignore', invalid='ignore'):
                frac = np.where(raw_ss > 0, res_ss / raw_ss, 0.0)
            kv = (X.var(0) > 0) & (frac > 1e-4)
            out = np.full(len(vsel), np.nan)
            if kv.sum():
                b, _ = MX._simple_regression_with_intercept(y, X[:, kv])
                out[kv] = b
            return out

        with np.errstate(divide='ignore', invalid='ignore'):
            t_mx = np.log(mT[j] / 2.0 / lib)
        trows.append(pd.DataFrame(dict(
            gene=g, variant_id=vids,
            t1_hm=total(T[j], 'joint'),
            t2_mx_response=total(t_mx, 'joint'),
            t3_mx_covariates=total(t_mx, 'mixqtl'),
        )))

    al = pd.concat(arows, ignore_index=True)
    to = pd.concat(trows, ignore_index=True)
    mx = pd.read_parquet(f'{OUT}/observed_matched_variants.parquet')
    al = al.merge(mx[['gene', 'variant_id', 'mx_beta_asc']],
                  on=['gene', 'variant_id'])
    to = to.merge(mx[['gene', 'variant_id', 'mx_beta_trc']],
                  on=['gene', 'variant_id'])
    al.to_parquet(f'{OUT}/nonweighting_allelic.parquet')
    to.to_parquet(f'{OUT}/nonweighting_total.parquet')

    res = {}
    print('=== ALLELIC CHANNEL, weights held at mixQTL\'s capped harmonic ===')
    print(f'{"rung":22s} {"r vs mixQTL":>12s} {"r vs previous":>14s} '
          f'{"spearman":>13s} {"sd(beta)":>12s}')
    names = ['r1_hm_donors', 'r2_mx_donors', 'r3_no_kappa', 'r4_logmean']
    labels = ['1 hapmixQTL donors', '2 + mixQTL donors',
              '3 + no pseudocount', '4 + log-of-mean (=mixQTL)']
    for i, (n, lab) in enumerate(zip(names, labels)):
        rv = r_or_nan(al[n], al.mx_beta_asc)
        rp = r_or_nan(al[names[i - 1]], al[n]) if i else np.nan
        sv = rs_or_nan(al[n], al.mx_beta_asc)
        print(f'{lab:22s} {rv:12.4f} {rp:14.4f} {sv:13.4f} '
              f'{np.nanstd(al[n]):12.4g}')
        res[f'allelic_{n}_vs_mixqtl'] = rv
        if i:
            res[f'allelic_{n}_vs_prev'] = rp
    gate_ok = r_or_nan(al.r4_logmean, al.mx_beta_asc)
    print(f'\n  VALIDATION: rung 4 should be mixQTL exactly -> r = {gate_ok:.6f}')

    print('\n=== TOTAL CHANNEL, unweighted throughout (as mixQTL) ===')
    print(f'{"rung":26s} {"r vs mixQTL":>12s} {"r vs previous":>14s} '
          f'{"spearman":>13s} {"sd(beta)":>12s}')
    tn = ['t1_hm', 't2_mx_response', 't3_mx_covariates']
    tl = ['1 hapmixQTL response', '2 + mixQTL response/offset',
          '3 + mixQTL covariates (=mixQTL)']
    for i, (n, lab) in enumerate(zip(tn, tl)):
        rv = r_or_nan(to[n], to.mx_beta_trc)
        rp = r_or_nan(to[tn[i - 1]], to[n]) if i else np.nan
        sv = rs_or_nan(to[n], to.mx_beta_trc)
        print(f'{lab:26s} {rv:12.4f} {rp:14.4f} {sv:13.4f} '
              f'{np.nanstd(to[n]):12.4g}')
        res[f'total_{n}_vs_mixqtl'] = rv
        if i:
            res[f'total_{n}_vs_prev'] = rp

    json.dump(res, open(f'{OUT}/nonweighting_ladder.json', 'w'), indent=1)


if __name__ == '__main__':
    main()

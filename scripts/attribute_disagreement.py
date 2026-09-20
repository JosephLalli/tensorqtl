"""Which divergence makes the two arms disagree? Observed pass only.

The full arms correlate at r = 0.70 on beta across 131,529 matched variants.
That is far below what two implementations of "the same estimator with
different weights" would give, so something other than the weights is doing
the work. This walks the allelic channel from hapmixQTL to mixQTL one
divergence at a time and reports the correlation with the hapmixQTL allelic
beta at each rung.

No permutation, no null, no p-value. Every number is a deterministic
function of the data, so nothing here depends on the permutation scheme.

The rungs, in order:

  hapmix          hapmixQTL's response (mean over draws of the log ratio,
                  pseudocount 0.5), its informative-donor set (v > 0), and
                  1/v weights
  + mx_donors     same response and weights, but mixQTL's donor cutoff
                  (5 <= y <= 5000)
  + mx_weights    same response and donors, but capped harmonic weights
  + mx_response   mixQTL's response too: log of the ratio of posterior-mean
                  counts, no pseudocount. This rung IS mixQTL's allelic
                  channel.

Each rung is compared to the rung above it and to the hapmixQTL baseline, so
the drop attributable to each single change is visible.
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


def main():
    import tensorqtl.hapmixqtl as HM
    from tensorqtl import mixqtl_replication as MX
    from compare_mixqtl_replication import load_inputs, gene_variant_index

    I = load_inputs()
    genes, keep = I['genes'], I['keep']
    A, _T, Va, _Vt, _C = HM.compute_summaries_from_gibbs(
        I['YL'], I['YR'], yT=I['YT'], count_noise=True)
    A, Va = A[:, keep], Va[:, keep]
    mL, mR = I['YL'].mean(2)[:, keep], I['YR'].mean(2)[:, keep]
    s_all = (I['xL'] - I['xR'])[I['idx']][:, keep]

    rows = []
    for j, g in enumerate(genes):
        vsel = gene_variant_index(I, g)
        if vsel.size == 0:
            continue
        S = s_all[vsel]                                   # [P, N]

        inf_hm = Va[j] > 1e-12
        gate_mx = ((mL[j] >= MX.ASC_CUTOFF) & (mR[j] >= MX.ASC_CUTOFF)
                   & (mL[j] <= MX.ASC_CAP) & (mR[j] <= MX.ASC_CAP))

        a_hm = A[j]                                       # mean of log draws, +0.5
        with np.errstate(divide='ignore', invalid='ignore'):
            a_mx = np.log(np.where(mR[j] > 0, mL[j], np.nan)
                          / np.where(mR[j] > 0, mR[j], np.nan))

        w_v = 1.0 / np.maximum(Va[j], 1e-12)

        def fit(mask, resp, wfun):
            n = int(mask.sum())
            if n <= 2:
                return None
            X = S[:, mask].T
            keepv = X.var(0) > 0
            if keepv.sum() == 0:
                return None
            r = resp[mask]
            if not np.isfinite(r).all():
                return None
            w = wfun(mask, n)
            b, _ = MX._simple_regression_through_origin(r, X[:, keepv], w)
            out = np.full(S.shape[0], np.nan)
            out[keepv] = b
            return out

        w_flat_v = lambda m, n: w_v[m]

        def w_harm_cap(m, n):
            w = MX.harmonic_weights(np.maximum(mL[j][m], 1e-12),
                                    np.maximum(mR[j][m], 1e-12))
            return MX.apply_weight_cap(w, n, MX.WEIGHT_CAP)[0]

        rungs = {
            'hapmix': fit(inf_hm, a_hm, w_flat_v),
            'mx_donors': fit(gate_mx, a_hm, w_flat_v),
            'mx_donors_weights': fit(gate_mx, a_hm, w_harm_cap),
            'mixqtl': fit(gate_mx, a_mx, w_harm_cap),
        }
        if any(v is None for v in rungs.values()):
            continue
        d = pd.DataFrame(rungs)
        d['gene'] = g
        rows.append(d)

    m = pd.concat(rows, ignore_index=True).dropna()
    m.to_parquet(f'{OUT}/disagreement_ladder.parquet')

    names = ['hapmix', 'mx_donors', 'mx_donors_weights', 'mixqtl']
    res = {'n_variants': int(len(m)), 'n_genes': int(m.gene.nunique())}
    print(f'{len(m)} variants, {m.gene.nunique()} genes  (allelic channel only)\n')
    print('correlation of each rung with the hapmixQTL baseline, '
          'and with the rung above:\n')
    print(f'{"rung":22s} {"r vs hapmix":>12s} {"r vs previous":>14s} '
          f'{"median |beta|":>14s}')
    for i, n in enumerate(names):
        r_base = pearsonr(m['hapmix'], m[n])[0]
        r_prev = pearsonr(m[names[i - 1]], m[n])[0] if i else np.nan
        print(f'{n:22s} {r_base:12.4f} {r_prev:14.4f} '
              f'{m[n].abs().median():14.5f}')
        res[f'{n}_r_vs_hapmix'] = float(r_base)
        if i:
            res[f'{n}_r_vs_previous'] = float(r_prev)
        res[f'{n}_median_abs_beta'] = float(m[n].abs().median())

    json.dump(res, open(f'{OUT}/disagreement_ladder.json', 'w'), indent=1)


if __name__ == '__main__':
    main()

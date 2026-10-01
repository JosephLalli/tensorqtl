"""Where the combined slope's shortfall from the simulated effect comes from, per plasmode gene set, hapmixQTL arm, effect size
and read band, from the stored causal-variant results alone (no refit).

Per causal unit (dataset, non-null gene, at its causal variant) the combined slope is the inverse-variance combination of
the channel slopes, slope = pa * slope_a + pt * slope_t with pa = wa / (wa + wt), wa = 1 / slope_a_se^2 where the allelic
channel is admitted (else 0), wt = 1 / slope_t_se^2. With simulated beta, the pipeline-scale truths ta (allelic, with the
0.5 pseudocount) and tt (total, on log2(CPM + 1)) and the count-scale total truth tc (the least-squares slope of the exact
log2 total fold on dosage / 2), the shortfall 1 - slope / beta is the exact sum of five parts:
  allelic_pseudocount  pa (beta - ta) / beta     the allelic truth lost to the 0.5 pseudocount
  allelic_residual     pa (ta - slope_a) / beta  the allelic estimate's shortfall against its own-scale truth
  total_definition     pt (beta - tc) / beta     the exact total fold's slope on dosage / 2 against beta
  total_compression    pt (tc - tt) / beta       the total truth lost to log2(CPM + 1)
  total_residual       pt (tt - slope_t) / beta  the total estimate's shortfall against its own-scale truth
Each part divides by beta only. Means over units per band, with the gene-clustered interval of 06_score.py (same
resampling indices, so the combined mean and interval equal summary.json's recovery bias_count exactly).

  PLASMODE_GENE_SET=<set> python3 scripts/beta_shortfall_budget.py   # writes OUT/budget_<set>.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'benchmark' / 'plasmode'))
import common as C          # noqa: E402

S6 = C.module('06_score')
OUT = C.D / 'beta_shortfall_20260929'
PARTS = ('allelic_pseudocount', 'allelic_residual', 'total_definition', 'total_compression', 'total_residual')
TOL = 1e-6   # slopes are stored as float32 and combined in float32 on the GPU (relative precision 6e-8)


def parts(Cz):
    b = Cz.beta.values
    sa, sea = Cz.slope_a.astype(float).values, Cz.slope_a_se.astype(float).values
    st, set_ = Cz.slope_t.astype(float).values, Cz.slope_t_se.astype(float).values
    ok_a = np.isfinite(sa) & np.isfinite(sea) & Cz.allelic_admitted.astype(bool).values
    wa = np.where(ok_a, 1.0 / np.where(ok_a, sea, 1.0) ** 2, 0.0)
    wt = 1.0 / set_ ** 2
    pa, pt = wa / (wa + wt), wt / (wa + wt)
    ta, tt, tc = (Cz[c].values for c in ('allelic_truth_pipeline', 'total_truth_pipeline', 'total_truth'))
    if (ok_a & ~np.isfinite(ta)).any():
        raise SystemExit(f'{int((ok_a & ~np.isfinite(ta)).sum())} admitted units without a pipeline allelic truth')
    z = lambda x: np.where(ok_a, x, 0.0)   # noqa: E731
    P = dict(allelic_pseudocount=z(pa * (b - ta) / b), allelic_residual=z(pa * (ta - sa) / b),
             total_definition=pt * (b - tc) / b, total_compression=pt * (tc - tt) / b, total_residual=pt * (tt - st) / b)
    short = 1.0 - Cz.slope.astype(float).values / b
    gap = np.abs(sum(P.values()) - short)
    if not np.all(np.isfinite(short)) or gap.max() > TOL:
        raise SystemExit(f'budget does not sum to 1 - slope / beta: max gap {gap.max():.2e}')
    P['shortfall'], P['allelic_share'] = short, pa
    return P


def summarize(P, Cz, genes, bsel, bidx):
    return {k: S6.ratio_block(pd.Series(v, index=Cz.index), Cz, genes, bsel, bidx) for k, v in P.items()}


def main():
    meta, genes, U, keep_a = S6.load_units(C.DATASETS, C.RESULTS)
    bsel, bidx = S6.band_selections(genes, U, keep_a)
    stored = json.loads(C.SUMMARY.read_text())
    out = {}
    for sc in (f'beta{b}' for b in meta['betas'] if b > 0):
        out[sc] = {}
        for arm in C.HAPMIX_ARMS:
            Cz = S6.causal_and_leads(C.RESULTS, U, sc, arm)[0]
            P = parts(Cz)
            res = summarize(P, Cz, genes, bsel, bidx)
            ref = stored['recovery'][sc][arm]['combined']['bias_count']
            for bn, v in res['shortfall'].items():
                if abs((1 - v['mean']) - ref[bn]['mean']) > 1e-12 or abs((1 - v['hi']) - ref[bn]['lo']) > 1e-12:
                    raise SystemExit(f'{sc} {arm} {bn}: shortfall does not reproduce summary.json recovery')
            out[sc][arm] = res
            a = res
            print(f'{C.GENE_SET} {sc} {arm:8s} all: shortfall {a["shortfall"]["all"]["mean"]:.3f} = '
                  + ' + '.join(f'{k} {a[k]["all"]["mean"]:+.3f}' for k in PARTS)
                  + f'; allelic weight share {a["allelic_share"]["all"]["mean"]:.2f}', flush=True)
    OUT.mkdir(exist_ok=True)
    C.write_json(OUT / f'budget_{C.GENE_SET}.json', dict(gene_set=C.GENE_SET, root=str(C.ROOT), parts=list(PARTS), budget=out))
    print(f'wrote {OUT / f"budget_{C.GENE_SET}.json"}')


if __name__ == '__main__':
    main()

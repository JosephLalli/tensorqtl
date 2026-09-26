"""How do unit weights change the total channel's standard errors?

For every tested variant of the corrected null store's 100 genes, over the 200
stored permutations: the REALIZED standard error is the sd of the null slope
across permutations (the true sampling spread under the null), and the
REPORTED standard error is the root mean square of the se map_nominal reported.
Compared for the Gibbs-weighted total channel (corrected_null_store keep
arm) and the unit-weighted one (total_channel_decomposition unit_weights arm),
which share values, covariates, genotype-PC rule and permutations:

  reported / realized   honesty of each arm's se (1 = honest, < 1 = too small)
  unit / Gibbs reported how the reported se changes in size
  unit / Gibbs realized how the true precision changes (< 1 = unit weights
                        estimate the slope more precisely)
Medians over variants, overall and by tercile of the gene's median total CPM.
A realized sd from 200 draws has a relative noise floor of about 0.05 per
variant. No randomness.
"""
import json

import numpy as np
import pandas as pd

import corrected_null_store as CNS

D = CNS.D
CACHE = D / 'cache' / 'gibbs_56b63c3b37ed5df8'
DEC = D / 'total_channel_decomposition_20260926'


def moments(files):
    acc = idx = None
    for f in files:
        d = pd.read_parquet(f, columns=['phenotype_id', 'variant_id', 'slope_t', 'slope_t_se'])
        d = d.set_index(['phenotype_id', 'variant_id']).sort_index()
        b, s = d.slope_t.values.astype(float), d.slope_t_se.values.astype(float)
        if acc is None:
            idx, acc = d.index, np.zeros((4, len(b)))
        if not d.index.equals(idx):
            raise SystemExit(f'variant set differs in {f}')
        ok = np.isfinite(b) & np.isfinite(s)
        acc += np.stack([ok, np.where(ok, b, 0), np.where(ok, b * b, 0), np.where(ok, s * s, 0)])
    n = acc[0]
    with np.errstate(invalid='ignore', divide='ignore'):
        realized = np.sqrt((acc[2] - acc[1] ** 2 / n) / (n - 1))
        reported = np.sqrt(acc[3] / n)
    return idx, reported, realized


def main():
    genes = (CNS.OUT / 'genes.txt').read_text().split()
    ig, rep_g, real_g = moments(sorted((CNS.OUT / 'draws').glob('keep_*.parquet')))
    iu, rep_u, real_u = moments(sorted((DEC / 'draws').glob('unit_weights_*.parquet')))
    if not ig.equals(iu):
        raise SystemExit('arms cover different variants')
    cg = (CACHE / 'genes.txt').read_text().split()
    gi = {g: i for i, g in enumerate(cg)}
    pT = np.load(CACHE / 'point_estimates' / 'pT.npy')[[gi[g] for g in genes]]
    es = pd.read_csv(CACHE / 'point_estimates' / 'edger' / 'edger_samples.tsv', sep='\t')
    cpm = np.median(pT / es.eff_lib_size.values[None, :] * 1e6, axis=1)
    edges = np.quantile(cpm, [1 / 3, 2 / 3])
    terc = pd.Series(np.digitize(cpm, edges), index=genes).reindex(ig.get_level_values(0)).values
    ok = np.isfinite(rep_g) & np.isfinite(rep_u) & (real_g > 0) & (real_u > 0)
    out = dict(cpm_tercile_edges=edges.tolist(), rows={})
    for lab, m in (('all', ok), ('low CPM', ok & (terc == 0)), ('middle CPM', ok & (terc == 1)),
                   ('high CPM', ok & (terc == 2))):
        out['rows'][lab] = dict(
            variants=int(m.sum()),
            gibbs_reported_over_realized=float(np.median(rep_g[m] / real_g[m])),
            unit_reported_over_realized=float(np.median(rep_u[m] / real_u[m])),
            unit_over_gibbs_reported=float(np.median(rep_u[m] / rep_g[m])),
            unit_over_gibbs_realized=float(np.median(real_u[m] / real_g[m])))
    per_gene = pd.DataFrame(dict(gene=ig.get_level_values(0)[ok], r=(real_u / real_g)[ok])).groupby('gene').r.median()
    out['per_gene_realized_unit_over_gibbs_q10_q25_q50_q75_q90'] = np.quantile(
        per_gene, [.1, .25, .5, .75, .9]).tolist()
    (DEC / 'se_accuracy.json').write_text(json.dumps(out, indent=1))
    print(pd.DataFrame(out['rows']).T.round(3).to_string())
    print('per-gene median realized unit / Gibbs, q10 q25 q50 q75 q90:',
          np.round(out['per_gene_realized_unit_over_gibbs_q10_q25_q50_q75_q90'], 3))


if __name__ == '__main__':
    main()

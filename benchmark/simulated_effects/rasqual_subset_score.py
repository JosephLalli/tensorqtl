"""Every arm scored on the genes and variants RASQUAL ran natively (04c_run_rasqual_native.py): the genes finished in
every dataset and each gene's variant subset (IN/subset.tsv), so all arms carry the same multiple-testing burden.

Per arm and dataset, a gene's lead is its smallest pval_nominal over the subset, ties broken by the larger |slope / se|
(06.causal_and_leads' rule); a gene without a row ranks last (p = inf). Per scenario with simulated effects, 06.ranking
gives power at 5% realized false-discovery proportion (overall and by depth band) and the AUC; on beta0.0, the share of
subset tests with p < 0.05 (every gene null). Output 04c.OUT/score.json; a table is printed.
"""
import json

import numpy as np
import pandas as pd

import common as C

SC, RN = C.module('06_score'), C.module('04c_run_rasqual_native')
ARMS = SC.ARMS + ('rasqual_native',)
ALPHA = 0.05


def arm_file(sc, arm, r):
    d = RN.OUT / sc / arm if arm == 'rasqual_native' else SC.arm_dir(C.RESULTS, sc, arm)
    return d / f'nominal_rep{r:03d}.parquet'


def subset_rows(sc, arm, r, sub):
    d = C.read_results(arm_file(sc, arm, r), SC.JOINT_COLS)
    d['variant_id'] = d.variant_id.astype(str)
    return d.merge(sub, left_on=['phenotype_id', 'variant_id'], right_on=['gene', 'variant_id'])


def leads(U, sc, arm, sub):
    out = []
    for r, u in U[U.scenario == sc].groupby('rep'):
        d = subset_rows(sc, arm, r, sub)
        d['p'] = d.pval_nominal.where(np.isfinite(d.pval_nominal), np.inf)
        d['absstat'] = np.abs(d.slope / d.slope_se)
        top = (d.sort_values(['phenotype_id', 'p', 'absstat'], ascending=[True, True, False], kind='stable')
               .groupby('phenotype_id', sort=False).head(1).set_index('phenotype_id'))
        L = u.set_index('gene')[['scenario', 'rep', 'is_null', 'band']].join(top[['p', 'absstat']], how='left')
        out.append(L.assign(p=L.p.fillna(np.inf)).rename(columns={'p': 'lead_p', 'absstat': 'lead_absstat'}).reset_index())
    return pd.concat(out, ignore_index=True)


def main():
    done = json.loads((RN.OUT / 'summary.json').read_text())['genes']
    sub = pd.read_csv(RN.IN / 'subset.tsv', sep='\t')[['gene', 'variant_id']]
    sub = sub[sub.gene.isin(done)]
    meta, genes, U, _ = SC.load_units(C.DATASETS, C.RESULTS)
    U = U[U.gene.isin(done)]
    print(f'{len(done)} genes, {len(sub):,} subset variants; {len(U):,} dataset-gene units', flush=True)
    res = dict(genes=done, n_variants=len(sub), null={}, ranking={})
    scen = sorted(U.scenario.unique())
    for i, sc in enumerate(scen):
        if not (~U[U.scenario == sc].is_null).any():
            for arm in ARMS:
                p = pd.concat([subset_rows(sc, arm, r, sub).pval_nominal for r in sorted(U[U.scenario == sc].rep.unique())])
                res['null'][arm] = dict(rate=float((p < ALPHA).mean()), tests=int(len(p)))
            continue
        res['ranking'][sc] = {}
        for arm in ARMS:
            k = SC.ranking(leads(U, sc, arm, sub), (SC.AUC_BOOT_KEY, i))
            res['ranking'][sc][arm] = dict(power=k['fdp_matched']['all']['power'], non_null=k['fdp_matched']['all']['non_null'],
                                           bands={b: v['power'] for b, v in k['fdp_matched'].items() if isinstance(v, dict) and b != 'all'},
                                           auc=k['auc']['all'])
    C.write_json(RN.OUT / 'score.json', res)
    rows = [dict(arm=a, null_rate=res['null'].get(a, {}).get('rate', np.nan),
                 **{f'power {sc}': res['ranking'][sc][a]['power'] for sc in res['ranking']},
                 **{f'auc {sc}': res['ranking'][sc][a]['auc']['mean'] for sc in res['ranking']}) for a in ARMS]
    print(pd.DataFrame(rows).set_index('arm').round(3).to_string())
    print(f'wrote {RN.OUT / "score.json"}')


if __name__ == '__main__':
    main()

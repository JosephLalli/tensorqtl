#!/usr/bin/env python3
"""The beta-shortfall page: where hapmixQTL's slope in the simulated-effects benchmark falls short of the simulated effect and why. Reads the budget
(beta_shortfall_budget.py), the counterfactual refits (beta_shortfall_refits.py), each simulated-effects root's datasets for the
noise-free total-channel compression against gene CPM, and the stored recovery of TReCASE for reference; embeds them as
JSON into beta_shortfall_template.html.

  python3 scripts/beta_shortfall_report.py      # writes OUT/beta_shortfall.html
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

D = Path('/mnt/ssd/lalli/brainvar_hapmix_deploy')
OUT = D / 'beta_shortfall_20260929'
SETS = {'deep': ('corrected_null_store_20260925', D / 'plasmode_meier_20260927'),
        'lowcov': ('stratum30_100', D / 'plasmode_lowcov_meier_20260927')}
TEMPLATE = Path(__file__).with_name('beta_shortfall_template.html')


def compression(root):
    """Per non-null unit: the noise-free total truth on log2(CPM + 1) over the count-scale total truth, and the gene's
    median CPM and donor mean of CPM / (CPM + 1) (from the beta 0 dataset, whose records are unthinned)."""
    ds = np.load(root / 'datasets' / 'beta0.0' / 'rep000.npz')
    cpm = ds['pT'] * 1e6 / ds['eff_lib'][None, :]
    genes = json.loads((root / 'datasets' / 'meta.json').read_text())['genes']
    g = pd.DataFrame(dict(gene=genes, cpm=np.median(cpm, 1), pred=(cpm / (cpm + 1)).mean(1)))
    t = pd.read_csv(root / 'datasets' / 'truth.tsv', sep='\t')
    t = t[t.beta_abs > 0].merge(g, on='gene')
    t['ratio'] = t.total_truth_pipeline / t.total_truth
    bad = ~np.isfinite(t.ratio)
    print(f'{root.name}: {len(t)} non-null units, {int(bad.sum())} without a finite total truth ratio (dropped)')
    t = t[~bad]
    return dict(units=[dict(beta=float(r.beta_abs), cpm=float(r.cpm), ratio=float(r.ratio), pred=float(r.pred))
                       for r in t.itertuples()],
                summary={str(b): dict(ratio=float(x.ratio.mean()), pred=float(x.pred.mean()),
                                      corr=float(np.corrcoef(x.ratio, x.pred)[0, 1]), units=len(x))
                         for b, x in t.groupby('beta_abs')},
                cpm_quantiles=[float(q) for q in np.quantile(g.cpm, [0.1, 0.5, 0.9])],
                lib_median=float(np.median(ds['eff_lib'])))


def main():
    data = {}
    for key, (gs, root) in SETS.items():
        S = json.loads((root / 'summary.json').read_text())
        data[key] = dict(budget=json.loads((OUT / f'budget_{gs}.json').read_text())['budget'],
                         refits=json.loads((OUT / f'refits_{gs}.json').read_text()),
                         compression=compression(root),
                         recovery={sc: {a: S['recovery'][sc][a] for a in ('split', 'unit', 'trecase', 'trecase_native', 'tensorqtl')}
                                   for sc in S['recovery']})
    page = TEMPLATE.read_text().replace('/*DATA*/null', json.dumps(data, separators=(',', ':'), allow_nan=False))
    tmp = OUT / 'beta_shortfall.tmp.html'
    tmp.write_text(page)
    os.replace(tmp, OUT / 'beta_shortfall.html')
    print(f'wrote {OUT / "beta_shortfall.html"} ({len(page):,} bytes)')


if __name__ == '__main__':
    main()

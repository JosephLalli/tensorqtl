"""Compact, reusable inputs for the five-arm half-read/unit-weight comparison."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from half_read_se_plot import BETAS, D, SETS
from half_read_io import atomic_path, cache_receipt, reuse_cache

BASELINES = ['split', 'unit', 'mixqtl', 'tensorqtl']
METHODS = ['half_read', 'split', 'unit', 'mixqtl', 'tensorqtl']
LABELS = ['Half-read + split', 'Split', 'Unit weights', 'mixQTL', 'tensorQTL total only']
COLORS = ['#0072B2', '#D55E00', '#666666', '#009E73', '#CC79A7']
KEY = ['stratum', 'gene', 'beta_abs', 'rep']


def fingerprint(ds, arm):
    h = hashlib.sha256(arm.encode())
    for k in ('A', 'T', 'Va', 'Vt', 'pL', 'pR', 'pT', 'eff_lib', 'perm', 'swap', 'causal_variant'):
        h.update(np.ascontiguousarray(ds[k]).tobytes())
    return h.hexdigest()


def add_bands(frame):
    frame = frame.copy()
    frame['coverage_band'] = ''
    for st, bins, labels in [('deep', [0, 100, 1000, np.inf], ['<100', '100–999', '≥1000']),
                             ('low', [0, 30, 50, 100], ['<30', '30–49', '50–99'])]:
        ix = frame.stratum.eq(st)
        bands = pd.cut(frame.loc[ix, 'coverage_reads'], bins, labels=labels, right=False)
        assert bands.notna().all(), 'coverage outside declared bands'
        frame.loc[ix, 'coverage_band'] = bands.astype(str)
    return frame


def extract(output):
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output/'baseline_manifest.json'
    outputs = [output/'baseline_fixed.parquet', output/'baseline_leads.parquet', output/'gene_design_units.parquet']
    expected_inputs = [Path(__file__), Path(__file__).with_name('half_read_se_plot.py'),
                       Path(__file__).with_name('half_read_io.py')]
    for st, (_, rootname, designpath, _) in SETS.items():
        root = D/rootname
        meta_path, design_path = root/'datasets/meta.json', D/designpath/'gene_design.tsv'
        meta = json.loads(meta_path.read_text())
        expected_inputs += [meta_path, design_path, root/'eigenmt_m_eff.tsv']
        for beta in BETAS:
            for rep in range(meta['n_datasets'][str(beta)]):
                expected_inputs.append(root/f'datasets/beta{beta}/rep{rep:03d}.npz')
                expected_inputs.extend(root/f'results/beta{beta}/{arm}/nominal_rep{rep:03d}.parquet'
                                       for arm in BASELINES)
    if reuse_cache(manifest_path, expected_inputs, outputs):
        return
    fixed, leads, units, audits = [], [], [], []
    for st, (_, rootname, designpath, _) in SETS.items():
        root = D/rootname
        meta_path, design_path = root/'datasets/meta.json', D/designpath/'gene_design.tsv'
        meta = json.loads(meta_path.read_text())
        gd = pd.read_csv(design_path, sep='\t').set_index('gene')
        meff_path = root/'eigenmt_m_eff.tsv'
        meff = pd.read_csv(meff_path, sep='\t').set_index('gene').m_eff
        for beta in BETAS:
            for rep in range(meta['n_datasets'][str(beta)]):
                ds_path = root/f'datasets/beta{beta}/rep{rep:03d}.npz'
                with np.load(ds_path) as loaded:
                    ds = {k: loaded[k] for k in loaded.files}
                u = pd.DataFrame(dict(gene=meta['genes'], variant_id=ds['causal_variant'].astype(str),
                    is_null=ds['is_null'], beta_truth=ds['allelic_truth'], total_truth=ds['total_truth']))
                u = u.assign(stratum=st, beta_abs=beta, rep=rep)
                u['coverage_reads'] = u.gene.map(gd.median_allele_resolved_reads)
                u['m_eff'] = u.gene.map(meff)
                assert u.coverage_reads.notna().all() and u.m_eff.gt(0).all()
                units.append(u)
                # Unit weights retain this pre-existing support rule. Record its scope.
                count_admitted = ~((ds['pL'] < .5) ^ (ds['pR'] < .5)) & ((ds['pL'] + ds['pR']) > 0)
                audits.append(dict(stratum=st, beta_abs=beta, rep=rep,
                    count_admitted_but_variance_excluded=int((count_admitted & ~(ds['Va'] > 1e-12)).sum())))
                for arm in BASELINES:
                    p = root/f'results/beta{beta}/{arm}/nominal_rep{rep:03d}.parquet'
                    metadata = pq.read_schema(p).metadata
                    assert metadata[b'plasmode_input_sha256'].decode() == fingerprint(ds, arm), str(p)
                    f = pd.read_parquet(p, columns=['phenotype_id', 'variant_id', 'slope', 'slope_se', 'pval_nominal'])
                    f = f.rename(columns={'phenotype_id': 'gene'})
                    f['variant_id'] = f.variant_id.astype(str)
                    scale = metadata[b'plasmode_slope_unit'].decode()
                    if scale == 'natural log':
                        f[['slope', 'slope_se']] = f[['slope', 'slope_se']].astype(float)/np.log(2.)
                    else:
                        assert scale == 'log2'
                    selected = u.merge(f, on=['gene', 'variant_id'], how='left', validate='one_to_one', indicator=True)
                    assert selected._merge.eq('both').all()
                    fixed.append(selected.drop(columns='_merge').assign(method=arm))
                    valid = np.isfinite(f.pval_nominal) & f.pval_nominal.between(0, 1)
                    f['lead_p'] = f.pval_nominal.where(valid, np.inf)
                    f['lead_absstat'] = (f.slope/f.slope_se).abs().replace([np.inf, -np.inf], np.nan).fillna(-1.)
                    top = f.sort_values(['gene', 'lead_p', 'lead_absstat', 'variant_id'],
                        ascending=[True, True, False, True], kind='stable').groupby('gene', sort=False).head(1)
                    lead = u[KEY+['is_null']].merge(top[['gene', 'variant_id', 'lead_p', 'lead_absstat']],
                        on='gene', how='left', validate='one_to_one')
                    assert len(lead) == 100 and lead.variant_id.notna().all()
                    leads.append(lead.rename(columns={'variant_id': 'lead_variant'}).assign(method=arm))
                print(f'Baselines {st} beta={beta:g} rep={rep}: four saved arms summarized', flush=True)
    all_fixed, all_leads, design = pd.concat(fixed), pd.concat(leads), add_bands(pd.concat(units))
    assert len(design) == 2000 and len(all_fixed) == len(all_leads) == 8000
    for q in (all_fixed, all_leads):
        assert not q.duplicated(KEY+['method']).any()
    for frame, path in zip((all_fixed, all_leads, design), outputs):
        with atomic_path(path) as temporary:
            frame.to_parquet(temporary, index=False)
    manifest = dict(**cache_receipt(expected_inputs, outputs), baseline_rows=len(all_fixed), gene_units=len(design),
                    baseline_regressions_rerun=False, admission_audit=audits)
    with atomic_path(manifest_path) as temporary:
        temporary.write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    extract(ap.parse_args().output)

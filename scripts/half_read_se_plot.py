"""Reported SE comparison on matched causal units; optional missing half-read fits.

The figure plots arithmetic mean reported SE, not empirical SD or RMSE.
Only beta 0/.2 require new half-read fits; existing .4/.8 fits are reused.
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from half_read_io import DEPLOY as D, atomic_path, digest
SETS = {
    'deep': ('corrected_null_store_20260925', 'plasmode_meier_20260927',
             'corrected_null_store_20260925', 'Broad-depth set (previously “deep”)'),
    'low': ('stratum30_100', 'plasmode_lowcov_meier_20260927',
            'plasmode_stratum30_100_20260927/gene_set', 'Low-coverage set'),
}
METHODS = ['half_read', 'split', 'mixqtl', 'tensorqtl']
LABELS = ['Half-read + split weights', 'Split', 'mixQTL', 'tensorQTL total only']
COLORS = ['#0072B2', '#D55E00', '#009E73', '#CC79A7']
BETAS = [0., .2, .4, .8]
KEY = ['gene', 'variant_id', 'beta_abs', 'rep']
FIELDS = ['phenotype_id', 'variant_id', 'slope', 'slope_se']


def common_module(stratum):
    os.environ['PLASMODE_GENE_SET'] = SETS[stratum][0]
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'benchmark' / 'plasmode'))
    import common as C
    if C.GENE_SET != SETS[stratum][0]:
        raise RuntimeError('use a separate process for each stratum')
    return C


def units(ds, genes, beta, rep):
    z = pd.DataFrame(dict(gene=genes, variant_id=ds['causal_variant'].astype(str),
        is_null=ds['is_null'], beta_truth=ds['allelic_truth'], total_truth=ds['total_truth']))
    z = z if beta == 0 else z[~z.is_null]
    return z.assign(beta_abs=beta, rep=rep).reset_index(drop=True)


def fill_missing(args):
    C = common_module(args.stratum)
    from half_read_trial import half_read
    import torch
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    target = args.output/f'half_read_extra_{args.stratum}.parquet'
    if target.exists():
        raise SystemExit(f'refusing overwrite: {target}')
    args.output.mkdir(parents=True, exist_ok=True)
    meta = json.loads((C.DATASETS/'meta.json').read_text())
    I, _, _ = C.load()
    S = C.setup(I)
    rows, receipts = [], []
    scratch = args.output/f'scratch_{args.stratum}'
    for beta in (0., .2):
        for rep in range(meta['n_datasets'][str(beta)]):
            ds = C.load_dataset(C.DATASETS, f'beta{beta}', rep)
            path = C.RESULTS/f'beta{beta}/split/nominal_rep{rep:03d}.parquet'
            if C.stored_fingerprint(path) != C.fingerprint(ds, 'split'):
                raise AssertionError('stored comparator fingerprint mismatch')
            old = C.read_results(path, C.COLS+C.DOF_COLS).set_index(['phenotype_id', 'variant_id']).sort_index()
            half = ds | {'T': half_read(ds['pT'], ds['eff_lib'])}
            fitted = C.run_nominal(S, half, 'split', scratch)[0]
            indexed = fitted.set_index(['phenotype_id', 'variant_id']).sort_index()
            ase = ['slope_a', 'slope_a_se', 'pval_a', 'dof_a', 'allelic_admitted']
            pd.testing.assert_frame_equal(indexed[ase], old[ase], check_exact=True)
            for col in ['slope', 'slope_se']:
                np.testing.assert_array_equal(np.isfinite(indexed[col]), np.isfinite(old[col]))
            u = units(ds, meta['genes'], beta, rep)
            q = u.merge(fitted.rename(columns={'phenotype_id': 'gene'}), on=['gene', 'variant_id'],
                        how='left', validate='one_to_one', indicator=True)
            if not (q._merge == 'both').all():
                raise AssertionError('half-read fit omitted a requested unit')
            rows.append(q[KEY+['slope', 'slope_se', 'slope_a', 'slope_a_se', 'allelic_admitted']])
            receipts.append(dict(beta=beta, rep=rep, scan_pairs=len(fitted), selected_units=len(q),
                original_ase_exact=True, dataset_sha256=digest(C.DATASETS/f'beta{beta}/rep{rep:03d}.npz')))
            print(f'{args.stratum} beta {beta} rep {rep}: {len(fitted):,} pairs, {len(q)} selected', flush=True)
    with atomic_path(target) as temporary:
        pd.concat(rows, ignore_index=True).to_parquet(temporary, index=False)
    with atomic_path(args.output/f'half_read_extra_{args.stratum}.json') as temporary:
        temporary.write_text(json.dumps(dict(
        gene_set=C.GENE_SET, source_sha256=digest(Path(__file__)), runs=receipts,
        mapper_sha256=digest(Path(C.map_nominal.__code__.co_filename)),
        scope='half-read total only, original ASE weights, unit total, unchanged GPU mapper'), indent=2)+'\n')
    shutil.rmtree(scratch)


def collect(args):
    parts, inputs = [], []
    for stratum, (setname, rootname, designpath, _) in SETS.items():
        root = D/rootname
        meta_path = root/'datasets/meta.json'
        meta = json.loads(meta_path.read_text())
        design_path = D/designpath/'gene_design.tsv'
        gd = pd.read_csv(design_path, sep='\t').set_index('gene')
        extra_path = args.output/f'half_read_extra_{stratum}.parquet'
        refit_path = D/'beta_shortfall_20260929'/f'refits_{setname}.parquet'
        extra = pd.read_parquet(extra_path)
        refits = pd.read_parquet(refit_path)
        refits = refits[(refits.arm == 'split') & (refits.config == 'voom')].copy()
        refits['beta_abs'] = refits.scenario.str[4:].astype(float)
        half = pd.concat([extra[KEY+['slope', 'slope_se']], refits[KEY+['slope', 'slope_se']]])
        if half.duplicated(KEY).any():
            raise AssertionError('duplicate half-read keys')
        inputs += [meta_path, design_path, extra_path, refit_path]
        for beta in BETAS:
            for rep in range(meta['n_datasets'][str(beta)]):
                ds_path = root/f'datasets/beta{beta}/rep{rep:03d}.npz'
                with np.load(ds_path) as ds:
                    u = units(ds, meta['genes'], beta, rep)
                inputs.append(ds_path)
                u['coverage_reads'] = u.gene.map(gd.median_allele_resolved_reads)
                if u.coverage_reads.isna().any():
                    raise AssertionError('coverage join failed')
                for method in METHODS:
                    if method == 'half_read':
                        fit = half[(half.beta_abs == beta) & (half.rep == rep)][['gene', 'variant_id', 'slope', 'slope_se']]
                    else:
                        path = root/f'results/beta{beta}/{method}/nominal_rep{rep:03d}.parquet'
                        fit = pd.read_parquet(path, columns=FIELDS).rename(columns={'phenotype_id': 'gene'})
                        unit = pq.read_schema(path).metadata[b'plasmode_slope_unit'].decode()
                        if unit == 'natural log':
                            fit[['slope', 'slope_se']] /= np.log(2.)
                        elif unit != 'log2':
                            raise AssertionError(f'unknown units {unit}')
                        inputs.append(path)
                    q = u.merge(fit, on=['gene', 'variant_id'], how='left', validate='one_to_one', indicator=True)
                    q['row_present'] = q._merge == 'both'
                    q = q.drop(columns='_merge').assign(stratum=stratum, method=method)
                    q['valid'] = q.row_present & np.isfinite(q.slope_se) & (q.slope_se > 0) & np.isfinite(q.slope)
                    parts.append(q)
    data = pd.concat(parts, ignore_index=True)
    key = ['stratum']+KEY
    mask = data.groupby(key).valid.agg(['all', 'size']).reset_index()
    if not (mask['size'] == 4).all():
        raise AssertionError('not four methods per unit')
    data = data.merge(mask[key+['all']].rename(columns={'all': 'common'}), on=key, validate='many_to_one')
    data['coverage_band'] = ''
    for stratum, bins, labels in [('deep', [0, 100, 1000, np.inf], ['<100', '100–999', '≥1000']),
                                 ('low', [0, 30, 50, np.inf], ['<30', '30–49', '50–99'])]:
        ix = data.stratum == stratum
        if stratum == 'low' and (data.loc[ix, 'coverage_reads'] >= 100).any():
            raise AssertionError('low coverage band overflow')
        data.loc[ix, 'coverage_band'] = pd.cut(data.loc[ix, 'coverage_reads'], bins=bins,
                                              labels=labels, right=False).astype(str)
    with atomic_path(args.output/'comparison_units.parquet') as temporary:
        data.to_parquet(temporary, index=False)
    return data, inputs


def summarize(data):
    out = []
    for (stratum, beta), full in data.groupby(['stratum', 'beta_abs']):
        selections = [('all', full)] + list(full.groupby('coverage_band', sort=False))
        for band, base in selections:
            for support in ('common', 'available'):
                for method in METHODS:
                    eligible = base[base.method == method]
                    q = eligible[eligible.common if support == 'common' else eligible.valid]
                    if not len(q):
                        continue
                    by = q.groupby('gene').slope_se.agg(['sum', 'count'])
                    rng = np.random.default_rng(20260929)
                    picks = rng.integers(0, len(by), size=(2000, len(by)))
                    boots = by['sum'].to_numpy()[picks].sum(1)/by['count'].to_numpy()[picks].sum(1)
                    truth = q.total_truth if method == 'tensorqtl' else q.beta_truth
                    e = q.slope - truth
                    finite = np.isfinite(e)
                    out.append(dict(stratum=stratum, beta_abs=beta, coverage_band=band, support=support,
                        method=method, n_units=len(q), n_genes=q.gene.nunique(), eligible_units=len(eligible),
                        valid_method_units=int(eligible.valid.sum()), excluded_units=len(eligible)-len(q),
                        mean_se=float(q.slope_se.mean()), median_se=float(q.slope_se.median()),
                        lo=float(np.quantile(boots, .025)), hi=float(np.quantile(boots, .975)),
                        mse=float(np.mean(e[finite]**2)), rmse=float(np.sqrt(np.mean(e[finite]**2))),
                        mse_units=int(finite.sum()),
                        directional_bias=float(np.mean(np.sign(truth[finite])*e[finite])) if beta else np.nan))
    return pd.DataFrame(out)


def plots(summary, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    q = summary[(summary.support == 'common')]
    handles = []

    def panel(ax, stratum, band, title):
        z = q[(q.stratum == stratum) & (q.coverage_band == band)]
        for m, label, color, offset in zip(METHODS, LABELS, COLORS, [-.12, -.04, .04, .12]):
            d = z[z.method == m].set_index('beta_abs').reindex(BETAS)
            x = np.arange(4)+offset
            p = ax.errorbar(x, d.mean_se, yerr=[d.mean_se-d.lo, d.hi-d.mean_se], fmt='o',
                            color=color, capsize=3, lw=1.2, ms=5, label=label)
            ax.plot(x[1:], d.mean_se.iloc[1:], color=color, lw=1.4)
            if len(handles) < 4:
                handles.append(p)
        counts = z[z.method == 'half_read'].set_index('beta_abs').n_units
        ax.set_xticks(range(4), [f'{b:g}'+('*' if b == 0 else '')+f'\nn={int(counts.get(b, 0))}' for b in BETAS])
        ax.set_xlabel('Planted |β|')
        ax.set_title(title, fontweight='bold', fontsize=12)
        upper = float(z.hi.max())*1.10
        ax.set_ylim(0, upper)
        assert (z.hi < ax.get_ylim()[1]).all(), 'a confidence bar is clipped'
        ax.grid(axis='y', color='#E4E4E4', lw=.7)
        ax.axvline(.45, color='#BBBBBB', ls=':', lw=.8)

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.7), sharey=False)
    for ax, stratum in zip(axes, SETS):
        panel(ax, stratum, 'all', SETS[stratum][3])
    axes[0].set_ylabel('Mean reported SE of β (log₂ units)')
    axes[1].set_ylabel('Mean reported SE of β (log₂ units)')
    fig.suptitle('Reported uncertainty by effect size and coverage set', fontsize=16, fontweight='bold', y=.98)
    fig.legend(handles, LABELS, loc='upper center', bbox_to_anchor=(.5, .91), ncol=2, frameon=False)
    fig.text(.06, .025, 'Panel y scales differ. Matched finite units; no significance filtering. Bars: 95% gene-bootstrap CI of mean SE.\n'
             '* β=0: one fixed sentinel per gene, one dataset. β>0: planted causal variants, three datasets. n counts gene–dataset units.', fontsize=9)
    fig.subplots_adjust(top=.76, bottom=.22, left=.08, right=.98, wspace=.26)
    for ext in ('png', 'pdf', 'svg'):
        with atomic_path(output/f'mean_reported_se.{ext}') as temporary:
            fig.savefig(temporary, dpi=200)
    plt.close(fig)

    handles.clear()
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharey=False)
    for row, (stratum, bands) in enumerate([('deep', ['<100', '100–999', '≥1000']),
                                           ('low', ['<30', '30–49', '50–99'])]):
        for ax, band in zip(axes[row], bands):
            panel(ax, stratum, band, f'{"Broad-depth" if row == 0 else "Low-coverage"} set: {band} reads')
        axes[row, 0].set_ylabel('Mean reported SE (log₂ units)')
    fig.suptitle('Reported SE within coverage bands', fontsize=17, fontweight='bold', y=.98)
    fig.legend(handles, LABELS, loc='upper center', bbox_to_anchor=(.5, .94), ncol=4, frameon=False)
    fig.text(.065, .025, 'Coverage = median allele-informative reads across all donors in the original real records, held fixed across β.\n'
             'Low-coverage set was selected using 30–100 reads among admitted donors; its all-donor medians can be below 30.\n'
             'Panel y scales differ. Matched finite units. Bars: 95% gene-bootstrap CI of mean SE. * β=0 uses null sentinels.', fontsize=9)
    fig.subplots_adjust(top=.85, bottom=.18, left=.065, right=.98, wspace=.20, hspace=.5)
    for ext in ('png', 'pdf', 'svg'):
        with atomic_path(output/f'mean_reported_se_by_read_band.{ext}') as temporary:
            fig.savefig(temporary, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--fill-missing', action='store_true')
    ap.add_argument('--stratum', choices=list(SETS))
    ap.add_argument('--redraw', action='store_true', help='redraw existing summary without rescanning or rereading result files')
    args = ap.parse_args()
    if args.fill_missing:
        if args.stratum is None:
            ap.error('--fill-missing requires --stratum')
        fill_missing(args)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    if args.redraw:
        plots(pd.read_csv(args.output/'summary.tsv', sep='\t'), args.output)
        manifest_path = args.output/'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        manifest['plot_source_sha256'] = digest(Path(__file__))
        manifest['plot_note'] = 'Independent panel y limits include every confidence bar; original computation source archived separately'
        with atomic_path(manifest_path) as temporary:
            temporary.write_text(json.dumps(manifest, indent=2)+'\n')
        return
    data, inputs = collect(args)
    summary = summarize(data)
    with atomic_path(args.output/'summary.tsv') as temporary:
        summary.to_csv(temporary, sep='\t', index=False)
    plots(summary, args.output)
    manifest = dict(source_sha256=digest(Path(__file__)), input_sha256={str(p): digest(p) for p in sorted(set(inputs))},
        methods=dict(zip(METHODS, LABELS)), statistic='Arithmetic mean of reported slope_se on common finite units',
        interval='2000 gene-cluster bootstrap percentile intervals for the mean SE; not individual beta confidence intervals',
        units='log2; mixQTL divide natural-log beta and SE by ln(2); tensorQTL already on dosage/2 scale',
        coverage='Original real-record gene median allele-informative reads over all donors; gene-set membership retained',
        mse='mean((slope - planted_truth)**2); total truth for tensorQTL, allelic truth for combined estimators',
        no_significance_filter=True, all_available_sensitivity='summary.tsv support=available',
        bootstrap_note='Cells resample genes, carrying all dataset observations for that gene; no empirical repeated-sample SD inferred')
    with atomic_path(args.output/'manifest.json') as temporary:
        temporary.write_text(json.dumps(manifest, indent=2)+'\n')
    print(summary[(summary.support == 'common') & (summary.coverage_band == 'all')][
        ['stratum', 'beta_abs', 'method', 'n_units', 'mean_se', 'lo', 'hi']].to_string(index=False))


if __name__ == '__main__':
    main()

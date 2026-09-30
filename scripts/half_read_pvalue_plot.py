"""Mean -log10 nominal p at the same matched units as the reported-SE figure."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from half_read_se_plot import SETS, METHODS, LABELS, COLORS, BETAS, D

SOURCE = D/'half_read_se_comparison_20260929/comparison_units.parquet'
KEY = ['stratum', 'gene', 'variant_id', 'beta_abs', 'rep', 'method']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def baselines(output):
    target = output/'baseline_pvalues.parquet'
    if target.exists():
        return
    u = pd.read_parquet(SOURCE)
    rows, inputs = [], [SOURCE]
    for (stratum, beta, rep, method), g in u[u.method != 'half_read'].groupby(
            ['stratum', 'beta_abs', 'rep', 'method']):
        path = D/SETS[stratum][1]/f'results/beta{beta}/{method}/nominal_rep{rep:03d}.parquet'
        f = pd.read_parquet(path, columns=['phenotype_id', 'variant_id', 'pval_nominal'],
            filters=[('phenotype_id', 'in', g.gene.unique().tolist()),
                     ('variant_id', 'in', g.variant_id.unique().tolist())])
        f = f.rename(columns={'phenotype_id': 'gene'})
        q = g[KEY].merge(f, on=['gene', 'variant_id'], how='left', validate='one_to_one', indicator=True)
        if not (q._merge == 'both').all():
            raise AssertionError(f'missing baseline keys: {path}')
        rows.append(q.drop(columns='_merge'))
        inputs.append(path)
    pd.concat(rows, ignore_index=True).to_parquet(target, index=False)
    (output/'baseline_manifest.json').write_text(json.dumps(dict(
        input_sha256={str(p): digest(p) for p in inputs}, rows=sum(len(x) for x in rows),
        source_sha256=digest(Path(__file__)), note='Selected saved p-values; no baseline regressions rerun'), indent=2)+'\n')
    print(f'Baseline p-values cached: {target}', flush=True)


def collect(output):
    baselines(output)
    original = pd.read_parquet(SOURCE)
    records = [pd.read_parquet(output/'baseline_pvalues.parquet')]
    for stratum in SETS:
        x = pd.read_parquet(output/f'half_read_pvalues_{stratum}.parquet')
        x['stratum'], x['method'] = stratum, 'half_read'
        records.append(x[KEY+['pval_nominal']])
    p = pd.concat(records, ignore_index=True)
    if p.duplicated(KEY).any():
        raise AssertionError('duplicate p-value keys')
    q = original.merge(p, on=KEY, how='left', validate='one_to_one', indicator=True)
    if not (q._merge == 'both').all():
        raise AssertionError('a cached SE unit has no p-value row')
    q = q.drop(columns='_merge')
    valid = np.isfinite(q.pval_nominal) & q.pval_nominal.between(0, 1)
    q['valid_p'] = valid
    keys = KEY[:-1]
    ok = q.groupby(keys).valid_p.all().rename('p_common').reset_index()
    q = q.merge(ok, on=keys, validate='many_to_one')
    q['plot_common'] = q.common & q.p_common
    # A zero written by finite-precision tail evaluation is displayed as a bound.
    floor = np.finfo(np.float64).tiny
    q['p_underflow'] = q.valid_p & q.pval_nominal.eq(0)
    q['neg_log10_p'] = np.where(q.valid_p, -np.log10(q.pval_nominal.clip(lower=floor)), np.nan)
    q.to_parquet(output/'pvalue_units.parquet', index=False)
    return q, floor


def summarize(data):
    out, pairs = [], []
    for (stratum, beta), full in data.groupby(['stratum', 'beta_abs']):
        for band, base in [('all', full)]+list(full.groupby('coverage_band', sort=False)):
            selected = base[base.plot_common]
            for method in METHODS:
                q = selected[selected.method == method]
                if not len(q):
                    continue
                by = q.groupby('gene').neg_log10_p.agg(['sum', 'count'])
                picks = np.random.default_rng(20260929).integers(0, len(by), size=(2000, len(by)))
                sample = by['sum'].to_numpy()[picks].sum(1)/by['count'].to_numpy()[picks].sum(1)
                out.append(dict(stratum=stratum, beta_abs=beta, coverage_band=band, method=method,
                    n_units=len(q), n_genes=q.gene.nunique(), mean_logp=float(q.neg_log10_p.mean()),
                    median_logp=float(q.neg_log10_p.median()),
                    lo=float(np.quantile(sample, .025)), hi=float(np.quantile(sample, .975)),
                    p_underflow=int(q.p_underflow.sum()),
                    units_lost_from_se_support=int(((base.method == method) & base.common & ~base.p_common).sum())))
            wide = selected.pivot(index=['gene', 'variant_id', 'rep'], columns='method', values='neg_log10_p')
            if len(wide):
                delta = wide.half_read-wide.split
                by = delta.groupby(level='gene').agg(['sum', 'count'])
                picks = np.random.default_rng(20260929).integers(0, len(by), size=(2000, len(by)))
                boot = by['sum'].to_numpy()[picks].sum(1)/by['count'].to_numpy()[picks].sum(1)
                pairs.append(dict(stratum=stratum, beta_abs=beta, coverage_band=band, n_units=len(wide),
                    half_minus_split=float(delta.mean()), lo=float(np.quantile(boot, .025)),
                    hi=float(np.quantile(boot, .975))))
    return pd.DataFrame(out), pd.DataFrame(pairs)


def plot(summary, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})

    def panel(ax, stratum, band, title):
        z = summary[(summary.stratum == stratum) & (summary.coverage_band == band)]
        handles = []
        for m, label, color, dx in zip(METHODS, LABELS, COLORS, [-.12, -.04, .04, .12]):
            d = z[z.method == m].set_index('beta_abs').reindex(BETAS)
            x = np.arange(4)+dx
            h = ax.errorbar(x, d.mean_logp, yerr=[d.mean_logp-d.lo, d.hi-d.mean_logp],
                            fmt='o', color=color, capsize=3, ms=5, lw=1.2, label=label)
            ax.plot(x[1:], d.mean_logp.iloc[1:], color=color, lw=1.5)
            handles.append(h)
        n = z[z.method == 'half_read'].set_index('beta_abs').n_units
        ax.set_xticks(range(4), [f'{b:g}'+('*' if b == 0 else '')+f'\nn={int(n.get(b, 0))}' for b in BETAS])
        ax.set_xlabel('Planted |β|')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylim(0, max(float(z.hi.max())*1.12, .6))
        assert (z.hi < ax.get_ylim()[1]).all(), 'confidence bar clipped'
        ax.grid(axis='y', color='#E4E4E4', lw=.7)
        ax.axvline(.45, color='#BBBBBB', ls=':', lw=.8)
        return handles

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.7))
    for ax, stratum in zip(axes, SETS):
        handles = panel(ax, stratum, 'all', SETS[stratum][3])
        ax.set_ylabel('Mean −log₁₀(nominal p-value)')
    fig.suptitle('Association evidence by effect size and coverage set', fontsize=16, fontweight='bold', y=.98)
    fig.legend(handles, LABELS, loc='upper center', bbox_to_anchor=(.5, .91), ncol=2, frameon=False)
    fig.text(.06, .025, 'Panel y scales differ. Larger values indicate smaller nominal p-values. Bars: 95% gene-bootstrap CI of mean −log₁₀(p).\n'
             'Matched finite units; no significance filter. * β=0 uses null sentinels; β>0 uses planted causal variants in three datasets.', fontsize=9)
    fig.subplots_adjust(top=.76, bottom=.22, left=.08, right=.98, wspace=.26)
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(output/f'mean_neg_log10_p.{ext}', dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for row, (stratum, bands) in enumerate([('deep', ['<100', '100–999', '≥1000']),
                                           ('low', ['<30', '30–49', '50–99'])]):
        for ax, band in zip(axes[row], bands):
            handles = panel(ax, stratum, band, f'{"Broad-depth" if row == 0 else "Low-coverage"} set: {band} reads')
        axes[row, 0].set_ylabel('Mean −log₁₀(nominal p-value)')
    fig.suptitle('Association evidence within coverage bands', fontsize=17, fontweight='bold', y=.98)
    fig.legend(handles, LABELS, loc='upper center', bbox_to_anchor=(.5, .94), ncol=4, frameon=False)
    fig.text(.065, .025, 'Coverage = median allele-informative reads across all donors in the original records, held fixed across β.\n'
             'Panel y scales differ. Same matched units as the SE comparison, subject to finite p-values; no significance selection.\n'
             'Bars: 95% gene-bootstrap CI of mean −log₁₀(p). * β=0 uses null sentinels. These are nominal association p-values.', fontsize=9)
    fig.subplots_adjust(top=.85, bottom=.18, left=.065, right=.98, wspace=.20, hspace=.5)
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(output/f'mean_neg_log10_p_by_read_band.{ext}', dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--baselines-only', action='store_true')
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.baselines_only:
        baselines(args.output)
        return
    data, floor = collect(args.output)
    summary, pairs = summarize(data)
    summary.to_csv(args.output/'summary.tsv', sep='\t', index=False)
    pairs.to_csv(args.output/'paired_half_minus_split.tsv', sep='\t', index=False)
    plot(summary, args.output)
    selected = data[data.plot_common]
    manifest = dict(source_sha256=digest(Path(__file__)),
        input_sha256={str(p): digest(p) for p in [SOURCE, args.output/'baseline_pvalues.parquet',
            args.output/'half_read_pvalues_deep.parquet', args.output/'half_read_pvalues_low.parquet']},
        statistic='Arithmetic mean of individual -log10(pval_nominal), not -log10(mean p)',
        interval='2000 gene-cluster bootstrap resamples', p_type='nominal association, not gene-level adjusted p',
        selection='same common finite support as SE plot, additionally require valid p for all four methods',
        se_units_lost=int(data.loc[data.method == 'half_read', 'common'].sum()-selected[selected.method == 'half_read'].shape[0]),
        zero_p_values=int(selected.p_underflow.sum()), zero_p_plot_floor=floor,
        max_observed_logp=float(selected.neg_log10_p.max()),
        p_underflow_note='If saved p is exactly zero, use float64 smallest positive normal as a conservative display bound; count reported above',
        null_note='Beta0 uses stored null dataset sentinels, not the separate 2000-record precision experiment')
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(summary[summary.coverage_band == 'all'][['stratum','beta_abs','method','n_units','mean_logp']].to_string(index=False))
    print('Half-read minus split:', flush=True)
    print(pairs[pairs.coverage_band == 'all'].to_string(index=False))


if __name__ == '__main__':
    main()

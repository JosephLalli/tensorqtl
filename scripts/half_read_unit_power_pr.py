"""Five-arm SE/log-p plots and gene-discovery power/precision-recall curves.

Descriptive oracle power follows the benchmark's realized FDP <= 5% rule.
Fixed-rule power uses eigenMT within genes then BH across 100 genes/dataset.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control

from unit_power_inputs import BETAS, COLORS, D, KEY, LABELS, METHODS, SETS, add_bands, digest, extract

NBOOT = 2000
BANDS = {'deep': ['<100', '100–999', '≥1000'], 'low': ['<30', '30–49', '50–99']}


def cluster_mean(q, column):
    g = q.groupby('gene')[column].agg(['sum', 'count'])
    picks = np.random.default_rng(20260929).integers(0, len(g), size=(NBOOT, len(g)))
    values = g['sum'].to_numpy()[picks].sum(1)/g['count'].to_numpy()[picks].sum(1)
    return float(q[column].mean()), *np.quantile(values, [.025, .975])


def comparison(output):
    prior = pd.read_parquet(D/'half_read_pvalue_comparison_20260929/pvalue_units.parquet')
    cols = KEY+['variant_id', 'method', 'is_null', 'coverage_reads', 'slope', 'slope_se', 'pval_nominal']
    base = pd.read_parquet(output/'baseline_fixed.parquet')
    unit = base[base.method.eq('unit') & (base.beta_abs.eq(0) | ~base.is_null)]
    x = add_bands(pd.concat([prior[cols], unit[cols]], ignore_index=True))
    x['valid'] = np.isfinite(x.slope) & np.isfinite(x.slope_se) & x.slope_se.gt(0)
    x['valid_p'] = np.isfinite(x.pval_nominal) & x.pval_nominal.gt(0) & x.pval_nominal.le(1)
    assert not x.duplicated(KEY+['method']).any()
    assert x.groupby(KEY).size().eq(5).all()
    x['common_se'] = x.groupby(KEY).valid.transform('all')
    x['common_p'] = x.common_se & x.groupby(KEY).valid_p.transform('all')
    x['neg_log10_p'] = -np.log10(x.pval_nominal.where(x.valid_p))
    x.to_parquet(output/'comparison_units.parquet', index=False)
    rows = []
    for (st, beta), full in x.groupby(['stratum', 'beta_abs']):
        for band in ['all']+BANDS[st]:
            q = full if band == 'all' else full[full.coverage_band.eq(band)]
            for metric, column, support in [('se', 'slope_se', 'common_se'), ('logp', 'neg_log10_p', 'common_p')]:
                for method in METHODS:
                    z = q[q.method.eq(method) & q[support]]
                    mean, lo, hi = cluster_mean(z, column)
                    rows.append(dict(stratum=st, beta_abs=beta, coverage_band=band, method=method,
                        metric=metric, mean=mean, lo=lo, hi=hi, n_units=len(z), n_genes=z.gene.nunique()))
    result = pd.DataFrame(rows)
    result.to_csv(output/'comparison_summary.tsv', sep='\t', index=False)
    return x, result


def load_leads(output):
    rows = [pd.read_parquet(output/'baseline_leads.parquet')]
    for st in SETS:
        rows.append(pd.read_parquet(output/f'half_read_leads_{st}.parquet').assign(method='half_read'))
    leads = pd.concat(rows, ignore_index=True)
    design = pd.read_parquet(output/'gene_design_units.parquet')
    leads = leads.merge(design[KEY+['coverage_reads', 'coverage_band', 'm_eff']],
                        on=KEY, how='left', validate='many_to_one')
    assert len(leads) == 10000 and not leads.duplicated(KEY+['method']).any()
    assert leads.groupby(KEY).size().eq(5).all() and leads.coverage_reads.notna().all()
    truth = leads.merge(design[KEY+['is_null']], on=KEY, suffixes=('', '_design'))
    assert truth.is_null.eq(truth.is_null_design).all()
    leads['finite_p'] = np.isfinite(leads.lead_p) & leads.lead_p.between(0, 1)
    leads['eigenmt_p'] = np.minimum(1., leads.lead_p*leads.m_eff).where(leads.finite_p, 1.)
    leads['bh_called'] = False
    for _, g in leads.groupby(['stratum', 'beta_abs', 'rep', 'method']):
        assert len(g) == 100
        called = (false_discovery_control(g.eigenmt_p.to_numpy(), method='bh') <= .05) & g.finite_p
        leads.loc[g.index, 'bh_called'] = called
    return leads


def rank_order(q, statistic_ties=True):
    """Identical evidence ties enter together; missing evidence cannot be called."""
    p = np.where(q.finite_p, q.lead_p, np.inf)
    stat = np.where(np.isfinite(q.lead_absstat) & q.finite_p, q.lead_absstat, -1.) if statistic_ties else np.zeros(len(q))
    o = np.lexsort((-stat, p))
    ends = np.r_[(p[o][:-1] != p[o][1:]) | (stat[o][:-1] != stat[o][1:]), True]
    return o, ends & np.isfinite(p[o])


def ranking(q, statistic_ties=True):
    q = q.reset_index(drop=True)
    o, ends = rank_order(q, statistic_ties)
    isnull = q.is_null.to_numpy()[o]
    tp, fp = np.cumsum(~isnull), np.cumsum(isnull)
    npos = int((~isnull).sum())
    precision = tp/(tp+fp)
    valid = ends & (precision >= .95)
    cut = int(np.flatnonzero(valid)[-1]) if valid.any() else -1
    called = np.zeros(len(q), bool)
    called[o[:cut+1]] = True
    curve = pd.DataFrame(dict(recall=tp[ends]/npos, precision=precision[ends],
        tp=tp[ends], fp=fp[ends], p_threshold=q.lead_p.to_numpy()[o][ends],
        absstat_threshold=q.lead_absstat.to_numpy()[o][ends]))
    ap = float(np.sum(np.diff(np.r_[0, curve.recall])*curve.precision))
    return called, curve, ap


def gene_weights(q):
    """Paired gene resampling within the eight possible three-dataset truth patterns."""
    truth = q.pivot(index='gene', columns='rep', values='is_null').sort_index()
    assert not truth.isna().any().any()
    pattern = truth.to_numpy(int) @ (2**np.arange(truth.shape[1]))
    weights = np.zeros((NBOOT, len(truth)), int)
    rng = np.random.default_rng(20260929)
    for pat in np.unique(pattern):
        ix = np.flatnonzero(pattern == pat)
        weights[:, ix] = rng.multinomial(len(ix), np.ones(len(ix))/len(ix), NBOOT)
    return weights[:, truth.index.get_indexer(q.gene)]


def discovery_boot(q):
    """Carry all datasets/methods together, retaining truth prevalence; recompute both cutoffs."""
    q = q.reset_index(drop=True)
    weights = gene_weights(q)
    o, ends = rank_order(q)
    ranked = q.iloc[o]
    w = weights[:, o]
    nn = ~ranked.is_null.to_numpy()
    nt = np.cumsum(w*nn, axis=1)
    total = np.cumsum(w, axis=1)
    allowed = ends[None, :] & (nt >= .95*total) & (total > 0)
    cut = np.max(np.where(allowed, np.arange(len(q))[None, :], -1), axis=1)
    selected = np.arange(len(q))[None, :] <= cut[:, None]
    oracle = np.zeros_like(selected)
    oracle[:, o] = selected
    bh = np.zeros_like(selected)
    for rep in sorted(q.rep.unique()):
        ri = np.flatnonzero(q.rep.eq(rep))
        order = ri[np.argsort(q.eigenmt_p.to_numpy()[ri], kind='stable')]
        wr = weights[:, order]
        assert (wr.sum(1) == 100).all()
        allowed = q.eigenmt_p.to_numpy()[order][None, :] <= .05*np.cumsum(wr, axis=1)/100
        rcut = np.max(np.where(allowed, np.arange(100)[None, :], -1), axis=1)
        bh[:, order] = (np.arange(100)[None, :] <= rcut[:, None]) & q.finite_p.to_numpy()[order][None, :]
    out = {}
    for rule, call in [('oracle_fdp5', oracle), ('bh5_eigenmt', bh)]:
        for band in ['all']+BANDS[q.stratum.iloc[0]]:
            mask = ~q.is_null.to_numpy() & (True if band == 'all' else q.coverage_band.eq(band).to_numpy())
            denom = (weights*mask).sum(1)
            values = np.divide((weights*mask*call).sum(1), denom,
                out=np.full(NBOOT, np.nan), where=denom > 0)
            out[rule, band] = np.nanquantile(values, [.025, .975])
    order, end = rank_order(q, statistic_ties=False)
    wr = weights[:, order]
    trues = np.cumsum(wr*(~q.is_null.to_numpy()[order]), axis=1)[:, end]
    totals = np.cumsum(wr, axis=1)[:, end]
    precisions = np.divide(trues, totals, out=np.zeros_like(trues, dtype=float), where=totals > 0)
    recalls = trues/(weights*(~q.is_null.to_numpy())).sum(1)[:, None]
    ap = (np.diff(np.c_[np.zeros(NBOOT), recalls], axis=1)*precisions).sum(1)
    return out, np.quantile(ap, [.025, .975])


def discovery(output):
    leads = load_leads(output)
    powers, curves, aps, nulls = [], [], [], []
    for (st, beta, method), q in leads.groupby(['stratum', 'beta_abs', 'method']):
        q = q.reset_index(drop=True)
        if beta == 0:
            nulls.append(dict(stratum=st, method=method, n_null=len(q),
                bh_false_calls=int(q.bh_called.sum()), finite_leads=int(q.finite_p.sum())))
            continue
        assert len(q) == 300 and (~q.is_null).sum() == 150
        oracle, _, _ = ranking(q)
        _, curve, ap = ranking(q, statistic_ties=False)
        boot, ap_interval = discovery_boot(q)
        curve = curve.assign(stratum=st, beta_abs=beta, method=method, coverage_band='all')
        curves.append(curve)
        aps.append(dict(stratum=st, beta_abs=beta, method=method, coverage_band='all', average_precision=ap,
            lo=ap_interval[0], hi=ap_interval[1],
            n_units=len(q), non_null=int((~q.is_null).sum()), finite_leads=int(q.finite_p.sum())))
        for band in BANDS[st]:
            bq = q[q.coverage_band.eq(band)]
            _, bcurve, bap = ranking(bq, statistic_ties=False)
            curves.append(bcurve.assign(stratum=st, beta_abs=beta, method=method, coverage_band=band))
            aps.append(dict(stratum=st, beta_abs=beta, method=method, coverage_band=band,
                average_precision=bap, lo=np.nan, hi=np.nan, n_units=len(bq),
                non_null=int((~bq.is_null).sum()), finite_leads=int(bq.finite_p.sum())))
        for rule, calls in [('oracle_fdp5', oracle), ('bh5_eigenmt', q.bh_called.to_numpy())]:
            nn = ~q.is_null.to_numpy()
            tp, fp = int((calls & nn).sum()), int((calls & ~nn).sum())
            for band in ['all']+BANDS[st]:
                mask = np.ones(len(q), bool) if band == 'all' else q.coverage_band.eq(band).to_numpy()
                npos = int((nn & mask).sum())
                hit = int((calls & nn & mask).sum())
                false = int((calls & ~nn & mask).sum())
                lo, hi = boot[rule, band]
                powers.append(dict(stratum=st, beta_abs=beta, method=method, rule=rule,
                    coverage_band=band, power=hit/npos, lo=lo, hi=hi, non_null=npos,
                    true_calls=hit, false_calls=false, null=int((~nn & mask).sum()),
                    realized_fdp=false/(hit+false) if hit+false else np.nan,
                    global_true_calls=tp, global_false_calls=fp,
                    global_fdp=fp/(tp+fp) if tp+fp else np.nan))
        leads.loc[leads.stratum.eq(st) & leads.beta_abs.eq(beta) & leads.method.eq(method), 'oracle_called'] = oracle
    leads.to_parquet(output/'gene_discovery_units.parquet', index=False)
    power, pr, ap = pd.DataFrame(powers), pd.concat(curves, ignore_index=True), pd.DataFrame(aps)
    power.to_csv(output/'power_summary.tsv', sep='\t', index=False)
    pr.to_csv(output/'precision_recall_points.tsv', sep='\t', index=False)
    ap.to_csv(output/'average_precision.tsv', sep='\t', index=False)
    pd.DataFrame(nulls).to_csv(output/'null_gene_calls.tsv', sep='\t', index=False)
    return leads, power, pr, ap


def make_plots(output, summary, power, pr, ap):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    styles = ['-', '--', '-.', '-', ':']
    handles = [Line2D([], [], color=c, ls=s, marker='o', label=l) for c, s, l in zip(COLORS, styles, LABELS)]

    def save(fig, stem):
        for ext in ('png', 'pdf', 'svg'):
            fig.savefig(output/f'{stem}.{ext}', dpi=200)
        plt.close(fig)

    def finish(fig, title, caption, bands):
        fig.suptitle(title, fontsize=16, fontweight='bold', y=.98)
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .93), ncol=3, frameon=False)
        fig.text(.06, .024, caption, fontsize=9)
        fig.subplots_adjust(top=.80 if bands else .75, bottom=.18 if bands else .23,
                            left=.065, right=.98, wspace=.27, hspace=.53)

    for metric, stem, title, ylabel in [
        ('se', 'mean_reported_se', 'Reported effect-size uncertainty', 'Mean reported SE (log₂ units)'),
        ('logp', 'mean_neg_log10_p', 'Association evidence', 'Mean −log₁₀(nominal p-value)')]:
        for banded in (False, True):
            fig, axes = plt.subplots(2 if banded else 1, 3 if banded else 2,
                figsize=(15, 9) if banded else (12, 5.8), squeeze=False)
            panels = [(axes[r, c], st, band) for r, st in enumerate(SETS) for c, band in enumerate(BANDS[st])] if banded else [
                (axes[0, i], st, 'all') for i, st in enumerate(SETS)]
            for ax, st, band in panels:
                z = summary[summary.metric.eq(metric) & summary.stratum.eq(st) & summary.coverage_band.eq(band)]
                for method, color, style, dx in zip(METHODS, COLORS, styles, np.linspace(-.16, .16, 5)):
                    q = z[z.method.eq(method)].set_index('beta_abs').reindex(BETAS)
                    x = np.arange(4)+dx
                    ax.errorbar(x, q['mean'], yerr=[q['mean']-q.lo, q.hi-q['mean']], fmt='o', color=color, ms=4, capsize=2)
                    ax.plot(x[1:], q['mean'].iloc[1:], color=color, ls=style, lw=1.5)
                n = z[z.method.eq('half_read')].set_index('beta_abs').n_units
                ax.set_xticks(range(4), [f'{b:g}'+('*' if b == 0 else '')+f'\nn={n.loc[b]}' for b in BETAS])
                ax.set_xlabel('Planted |β|')
                ax.set_ylabel(ylabel)
                name = 'Broad-depth set' if st == 'deep' else 'Low-coverage set'
                ax.set_title(name+('' if band == 'all' else f': {band} reads'), fontweight='bold')
                ax.set_ylim(0, max(.01, float(z.hi.max())*1.12))
                assert z.hi.lt(ax.get_ylim()[1]).all()
                ax.grid(axis='y', color='#E4E4E4')
                ax.axvline(.45, color='#BBBBBB', ls=':', lw=.8)
            caption = 'Panel y scales differ. Five-method common finite support; no significance selection. Bars: 95% gene-bootstrap CI of the mean.\n'
            caption += 'Unit weights: original log₂(CPM + 1), equal ASE and total weights on existing admitted donors. * β=0: null sentinels.\n'
            caption += 'Coverage bands use median allele-informative reads across all donors, fixed across β.'
            finish(fig, title+' with unit weights', caption, banded)
            save(fig, stem+('_by_read_band' if banded else ''))

    for rule, title in [('oracle_fdp5', 'Power at ≤5% realized false discoveries (truth-based cutoff)'),
                        ('bh5_eigenmt', 'Power using eigenMT + BH at 5%')]:
        for banded in (False, True):
            fig, axes = plt.subplots(2 if banded else 1, 3 if banded else 2,
                figsize=(15, 9) if banded else (12, 5.8), squeeze=False)
            panels = [(axes[r, c], st, band) for r, st in enumerate(SETS) for c, band in enumerate(BANDS[st])] if banded else [
                (axes[0, i], st, 'all') for i, st in enumerate(SETS)]
            for ax, st, band in panels:
                z = power[power.rule.eq(rule) & power.stratum.eq(st) & power.coverage_band.eq(band)]
                for method, color, style, dx in zip(METHODS, COLORS, styles, np.linspace(-.12, .12, 5)):
                    q = z[z.method.eq(method)].set_index('beta_abs').reindex(BETAS[1:])
                    x = np.arange(3)+dx
                    ax.plot(x, q.power, color=color, ls=style, marker='o', ms=4)
                    ax.vlines(x, q.lo, q.hi, color=color, alpha=.5, lw=1)
                n = z[z.method.eq('half_read')].set_index('beta_abs').non_null
                ax.set_xticks(range(3), [f'{b:g}\nn₊={n.loc[b]}' for b in BETAS[1:]])
                ax.set_ylim(0, 1.03)
                ax.set_xlabel('Planted |β|')
                ax.set_ylabel('Power: fraction of non-null genes called')
                ax.set_title(('Broad-depth set' if st == 'deep' else 'Low-coverage set')+
                             ('' if band == 'all' else f': {band} reads'), fontweight='bold')
                ax.grid(axis='y', color='#E4E4E4')
            if rule == 'oracle_fdp5':
                caption = 'Gene-level ranking by strongest nominal p, with statistic tie-breaks. Cutoff chosen using known truth; this is not a deployable FDR rule.\n'
                caption += 'Three datasets pooled per β; 150 non-null + 150 null units per set. Bars: 95% truth-stratified gene-bootstrap CI; cutoffs reselected.\n'
            else:
                caption = 'Within gene: min(1, strongest nominal p × cached eigenMT M_eff). BH across all 100 genes in each dataset.\n'
                caption += 'Bars: 95% truth-stratified gene-bootstrap CI; BH rerun per draw. Missing evidence stays uncalled; realized FDP is reported separately.\n'
            caption += 'Coverage panels apply the full-set cutoff; no separate threshold tuning within coverage bands.'
            finish(fig, title, caption, banded)
            save(fig, 'power_'+rule+('_by_read_band' if banded else ''))

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for row, st in enumerate(SETS):
        for col, rule in enumerate(['oracle_fdp5', 'bh5_eigenmt']):
            ax = axes[row, col]
            for method, color, style, dx in zip(METHODS, COLORS, styles, np.linspace(-.12, .12, 5)):
                q = power[power.rule.eq(rule) & power.stratum.eq(st) & power.coverage_band.eq('all') &
                    power.method.eq(method)].set_index('beta_abs').reindex(BETAS[1:])
                x = np.arange(3)+dx
                ax.plot(x, q.power, color=color, ls=style, marker='o', ms=4)
                ax.vlines(x, q.lo, q.hi, color=color, alpha=.5, lw=1)
            ax.set_xticks(range(3), ['0.2', '0.4', '0.8'])
            ax.set(ylim=(0, 1.03), xlabel='Planted |β|', ylabel='Power: non-null genes recovered / 150')
            ax.set_title(('Broad-depth' if st == 'deep' else 'Low-coverage')+' — '+
                ('truth-based FDP ≤5%' if col == 0 else 'eigenMT + BH 5%'), fontweight='bold')
            ax.grid(axis='y', color='#E4E4E4')
    finish(fig, 'Gene-discovery power: ranking potential and a fixed calling rule',
        'Left: cutoff chosen using simulation truth; measures ranking potential. Right: eigenMT correction then BH across all 100 genes/dataset.\n'
        'Each point pools 3 datasets (150 non-null + 150 null units). Missing evidence stays undetected. Bars: 95% truth-stratified gene-bootstrap CI.\n'
        'Thresholds are recomputed per resample. BH empirical false-discovery fractions are in power_summary.tsv; 5% is its target, not a guarantee.', True)
    save(fig, 'power_comparison')

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for row, st in enumerate(SETS):
        for col, beta in enumerate(BETAS[1:]):
            ax = axes[row, col]
            for method, color, style in zip(METHODS, COLORS, styles):
                z = pr[pr.stratum.eq(st) & pr.beta_abs.eq(beta) & pr.method.eq(method) & pr.coverage_band.eq('all')]
                ax.step(np.r_[0, z.recall], np.r_[1, z.precision], where='pre', color=color, ls=style, lw=1.6)
                p = power[power.stratum.eq(st) & power.beta_abs.eq(beta) & power.method.eq(method) &
                    power.rule.eq('oracle_fdp5') & power.coverage_band.eq('all')].iloc[0]
                if p.true_calls+p.false_calls:
                    ax.plot(p.power, 1-p.realized_fdp, 'o', color=color, ms=4)
            ax.axhline(.5, color='#AAAAAA', ls=':', lw=1)
            ax.axhline(.95, color='#AAAAAA', ls='--', lw=.8)
            ax.set(xlim=(0, 1.02), ylim=(0, 1.03), xlabel='Recall / power', ylabel='Precision = true calls / all calls')
            ax.set_title(('Broad-depth' if st == 'deep' else 'Low-coverage')+f' set: |β|={beta:g}', fontweight='bold')
            ax.grid(alpha=.15)
    finish(fig, 'Gene-discovery precision–recall curves',
        'Genes ranked by strongest nominal p; equal p enters together. Dots mark the ≤5% realized-FDP operating points (statistic tie-breaks).\n'
        'Each panel pools 3 datasets: 150 non-null and 150 null gene–dataset units. Missing evidence remains in the recall denominator.\n'
        'Horizontal lines: 95% precision and 50% positive prevalence. This precision concerns discovery lists, not beta-estimate SE.', True)
    save(fig, 'precision_recall')

    for st in SETS:
        fig, axes = plt.subplots(3, 3, figsize=(15, 11))
        for row, band in enumerate(BANDS[st]):
            for col, beta in enumerate(BETAS[1:]):
                ax = axes[row, col]
                a = ap[ap.stratum.eq(st) & ap.beta_abs.eq(beta) & ap.coverage_band.eq(band)]
                for method, color, style in zip(METHODS, COLORS, styles):
                    z = pr[pr.stratum.eq(st) & pr.beta_abs.eq(beta) & pr.method.eq(method) & pr.coverage_band.eq(band)]
                    ax.step(np.r_[0, z.recall], np.r_[1, z.precision], where='pre', color=color, ls=style, lw=1.5)
                n, npos = int(a.iloc[0].n_units), int(a.iloc[0].non_null)
                ax.axhline(npos/n, color='#AAAAAA', ls=':', lw=1)
                ax.axhline(.95, color='#AAAAAA', ls='--', lw=.8)
                ax.set(xlim=(0, 1.02), ylim=(0, 1.03), xlabel='Recall / power', ylabel='Precision')
                ax.set_title(f'{band} reads; |β|={beta:g}\nn₊={npos}, n₀={n-npos}', fontsize=10)
        finish(fig, ('Broad-depth' if st == 'deep' else 'Low-coverage')+' set: precision–recall within coverage bands',
            'Each curve sweeps the lead nominal-p threshold within the displayed coverage band; equal p enters together. No significance filtering.\n'
            'All truth labels and missing evidence stay in the denominator. Dotted line is each panel’s positive prevalence; dashed line is 95% precision.\n'
            'Coverage = median allele-informative reads over all original donors. Fixed bands and gene truth are reused across all five methods.', True)
        save(fig, f'precision_recall_by_read_band_{st}')


def check_ranking():
    # Equal-score positive/negative labels cannot be favorably ordered to create discoveries.
    q = pd.DataFrame(dict(lead_p=[.001, .001, .1, np.inf], lead_absstat=[3., 3., 1., -1.],
        finite_p=[True, True, True, False], is_null=[False, True, False, False]))
    called, curve, ap = ranking(q)
    assert not called.any() and curve.tp.tolist() == [1, 2] and curve.fp.tolist() == [1, 1]
    assert curve.recall.iloc[-1] == 2/3
    np.testing.assert_allclose(ap, .5/3+(2/3)/3)
    _, permuted, ap2 = ranking(q.iloc[[1, 0, 2, 3]])
    pd.testing.assert_frame_equal(curve, permuted)
    assert ap == ap2


def report(output, summary, power, ap):
    """Local figure index with the analysis definitions beside the numeric results."""
    from html import escape
    main = [('mean_reported_se', 'Mean reported standard error'),
            ('mean_neg_log10_p', 'Mean −log₁₀ nominal p'),
            ('power_comparison', 'Gene-discovery power'),
            ('precision_recall', 'Gene-discovery precision–recall')]
    supplemental = [('mean_reported_se_by_read_band', 'SE by read band'),
        ('mean_neg_log10_p_by_read_band', 'Mean −log₁₀(p) by read band'),
        ('power_oracle_fdp5_by_read_band', 'Truth-based power by read band'),
        ('power_bh5_eigenmt_by_read_band', 'BH power by read band'),
        ('precision_recall_by_read_band_deep', 'PR by read band: broad-depth set'),
        ('precision_recall_by_read_band_low', 'PR by read band: low-coverage set')]
    body = ['<!doctype html><html lang="en"><meta charset="utf-8"><title>Half-read and unit-weight comparison</title>',
        '<style>body{font:17px/1.55 system-ui,sans-serif;max-width:1200px;margin:35px auto;padding:0 22px;color:#222}'
        'img{width:100%;height:auto}table{border-collapse:collapse;font-size:14px;margin:20px 0}th,td{padding:6px 12px;border-bottom:1px solid #ddd}'
        'a{color:#075b9b}summary{cursor:pointer;font-weight:bold}code{font-size:90%}</style>',
        '<h1>Half-read, unit weights, and gene discovery</h1><p>2026-09-29 · Five methods · Two coverage sets · Three effect datasets per nonzero beta.</p>',
        '<p><b>Half-read and split retain similar association evidence and discovery power.</b> Unit weights have lower broad-depth power and are close to split at low coverage. '
        'These are descriptive point estimates with uncertainty across the selected genes, not evidence of a universal winner.</p>',
        '<p>Half-read changes only total expression to log₂((count + 0.5)/(effective library size + 1) × 10⁶), retaining Gibbs ASE weights. '
        'Split uses Gibbs ASE weights and unit total weights. Unit weights use the original log₂(CPM + 1) total phenotype, with equal weights in both channels. '
        'All three retain the original ASE admission rules; the retained variance threshold excludes zero additional count-eligible records in these 20 datasets.</p>',
        '<p>SE and mean-log-p plots retain the same 959 matched finite gene–dataset units across all five methods. '
        'The mean is computed after taking −log₁₀ of each nominal p-value. These are reported SEs, not empirical repeated-sampling standard deviations. '
        'At beta 0.4, half-read raises mean SE from 0.091816 to 0.100898 (broad-depth) and 0.131105 to 0.168227 (low coverage).</p>',
        '<p>For discovery, each nonzero-beta panel contains all 300 gene–dataset units: 150 non-null and 150 null. '
        'Missing results remain undetected, including in recall denominators. Beta zero has no positive truth labels and is therefore excluded from power/PR.</p>',
        '<p><b>Power definitions:</b> Left: choose the deepest whole evidence-tie prefix with observed false-discovery proportion ≤5%, using truth labels; '
        'this measures ranking potential, not a deployable FDR guarantee. Right: multiply each gene’s minimum nominal p by cached eigenMT M_eff, cap at 1, '
        'and apply BH at 5% across all 100 genes per dataset, with missing p=1. Actual false-discovery fractions appear in the table below.</p>',
        '<p><b>PR definitions:</b> precision = true calls / all calls; recall = true calls / all non-null units. Genes are ranked by their minimum nominal p; '
        'whole equal-p groups enter together. Average precision is the step integral, not trapezoidal area. Coverage-band PR curves sweep thresholds within each band; '
        'coverage-band power uses the full-set calling rule.</p>',
        '<p>Power/AP intervals use 2,000 paired gene resamples within the three-dataset truth patterns, preserving positive prevalence and carrying all methods/datasets together. '
        'Calling thresholds and BH are recomputed per draw. Intervals are conditional on these genes and stored datasets from 92 donors; they do not quantify new-cohort uncertainty.</p>']
    for stem, title in main:
        body.append(f'<h2>{escape(title)}</h2><p><a href="{stem}.pdf">PDF</a> · <a href="{stem}.svg">SVG</a></p>'
                    f'<img src="{stem}.png" alt="{escape(title)}">')
    body.append('<h2>Coverage-band figures</h2><ul>')
    for stem, title in supplemental:
        body.append(f'<li><a href="{stem}.png">{escape(title)}</a> · <a href="{stem}.pdf">PDF</a></li>')
    body.append('</ul><h2>Power and actual false discoveries</h2>')
    table = power[power.coverage_band.eq('all')][['stratum','beta_abs','method','rule','power','true_calls','false_calls','realized_fdp']]
    body.append(table.to_html(index=False, float_format=lambda x: f'{x:.4f}', border=0))
    body.append('<p>Legacy benchmark BH corrected finite genes only. This comparison fixes the family at 100. '
        'Only mixQTL call counts change: broad beta 0.4, 71 versus 72; low beta 0.4, 18 versus 19; low beta 0.8, 25 versus 28. '
        'All 24 baseline truth-based power results and all 24 legacy BH results were reproduced separately.</p>')
    body.append('<details><summary>Average precision and gene-bootstrap intervals</summary>')
    body.append(ap[ap.coverage_band.eq('all')][['stratum','beta_abs','method','average_precision','lo','hi']].to_html(index=False,
        float_format=lambda x: f'{x:.4f}', border=0))
    body.append('</details><p><a href="comparison_summary.tsv">SE / log-p data</a> · <a href="power_summary.tsv">Power data</a> · '
        '<a href="precision_recall_points.tsv">PR points</a> · <a href="average_precision.tsv">Average precision</a> · '
        '<a href="manifest.json">Provenance</a> · <a href="baseline_acceptance.json">Baseline checks</a></p></html>')
    (output/'index.html').write_text('\n'.join(body)+'\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    check_ranking()
    extract(args.output)
    x, summary = comparison(args.output)
    leads, power, pr, avgpr = discovery(args.output)
    make_plots(args.output, summary, power, pr, avgpr)
    report(args.output, summary, power, avgpr)
    manifest = dict(source_sha256=digest(Path(__file__)), input_helper_sha256=digest(Path(__file__).with_name('unit_power_inputs.py')),
        methods=dict(zip(METHODS, LABELS)), comparison_units=len(x)//5,
        common_se_units=int(x.common_se.sum())//5, common_p_units=int(x.common_p.sum())//5,
        gene_discovery_units=len(leads)//5, per_positive_beta_per_set='150 non-null + 150 null gene-dataset units',
        n_gene_bootstrap=NBOOT, no_regression_changes=True, unit_definition='Original point-estimate transforms; both-channel weights=1 on existing admitted donors, including retained Va>EPS support rule',
        oracle_power='Deepest complete evidence-tie prefix with realized FDP<=.05; nominal lead p then abs slope/SE',
        bh_power='Cached eigenMT M_eff times min nominal p capped at 1; BH5% per dataset over 100 genes; invalid evidence uncalled',
        missing_leads=int((~leads.finite_p).sum()), average_precision='Step integral of precision against recall, whole ties included; missing positives have zero contribution',
        conditioning='Paired truth-pattern-stratified gene bootstrap across this selected gene set, carrying three stored datasets and preserving truth prevalence; not independent new-cohort replicates',
        input_sha256={p.name:digest(p) for p in [args.output/'baseline_fixed.parquet', args.output/'baseline_leads.parquet',
            args.output/'gene_design_units.parquet', *[args.output/f'half_read_leads_{s}.parquet' for s in SETS]]})
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(power[power.coverage_band.eq('all')][['stratum','beta_abs','method','rule','power','true_calls','false_calls','realized_fdp']].to_string(index=False))


if __name__ == '__main__':
    main()

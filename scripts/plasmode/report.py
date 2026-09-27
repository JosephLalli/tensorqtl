"""HTML report of the plasmode benchmark: reads score.py's summary, the check
files, the run logs, the joint-model runs' records and the mixQTL ladder, draws
four figures, and writes one self-contained page
(figures embedded as base64, and also written as PNG beside it).

Computes nothing that score.py did not, except the positions of points in the
figures and small arithmetic on its numbers (range overlaps, differences and
ratios of stored values). Every number on the page is read from the inputs
below.
Usage: report.py
"""
import base64
import html
import json
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402

ROOT = Path('/mnt/ssd/lalli/brainvar_hapmix_deploy/plasmode_20260926')
SUMMARY = ROOT / 'summary.json'
SCORE_LOG = ROOT / 'score.log'
RUN_ARMS_LOG = ROOT / 'run_arms.log'
MAKE_LOG = ROOT / 'make_datasets.log'
CHECK_GEN = ROOT / 'checks' / 'check_generator.json'
CHECK_PREMISE = ROOT / 'checks' / 'salmon_premise.json'
RASQUAL_LOG = ROOT / 'results_rasqual' / 'run_rasqual.log'
TRECASE_SUMMARY = ROOT / 'results_trecase_asseq' / 'summary.json'
LADDER = ROOT / 'ladder' / 'ladder.json'
OUT = ROOT / 'report'
PAGE = OUT / 'plasmode_report.html'

ARMS = ('gibbs', 'split', 'unit', 'plus_one', 'mixqtl', 'mixqtl_permissive')
HAPMIX = ARMS[:4]
JOINT = ('rasqual', 'trecase')
ALL = ARMS + JOINT      # the rows of every combined-channel table and figure
LABEL = {'gibbs': 'gibbs (1/v both channels, shipped)', 'split': 'split (1/v allelic, unit total)',
         'unit': 'unit (weight 1 both channels)', 'plus_one': 'plus_one (1/(v+1) both channels)',
         'mixqtl': 'mixQTL, published cutoffs', 'mixqtl_permissive': 'mixQTL, permissive cutoffs',
         'rasqual': 'RASQUAL (joint model)', 'trecase': 'TReCASE, asSeq (joint model)'}
SHORT = {'gibbs': 'gibbs', 'split': 'split', 'unit': 'unit', 'plus_one': 'plus_one',
         'mixqtl': 'mixQTL pub.', 'mixqtl_permissive': 'mixQTL perm.', 'rasqual': 'RASQUAL', 'trecase': 'TReCASE'}
COLOR = dict(zip(ALL, ('#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948')))  # dataviz categorical slots 1-8
MARKER = dict(zip(ALL, 'osD^vPXh'))
BETA_COLOR = {'0.2': '#86b6ef', '0.4': '#2a78d6', '0.8': '#104281'}   # dataviz blue ramp, ordinal steps 250/450/650
BETAS = ('0.2', '0.4', '0.8')
BANDS = ('all', '<100', '100-999', '>=1000')
NO_ONE_DF = 'without one-df genes'   # score.py's gene set without the one-residual-df allelic gene
CHANNELS = ('combined', 'allelic', 'total')
MIX_CH = {'combined': 'meta', 'allelic': 'asc', 'total': 'trc'}
ALPHAS = ('0.05', '0.01', '0.001')
DETECT = ('0.05', '0.001', '1e-05')
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e1e0d9'
LOG_TICKS = (0.25, 0.35, 0.5, 0.7, 1, 1.4, 2, 4, 8, 16, 32)


def write_atomic(path, data):
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_bytes(data)
    os.replace(tmp, path)


def load():
    last = SCORE_LOG.read_text().rstrip().splitlines()[-1]
    if not last.startswith('wrote') or str(SUMMARY) not in last:
        raise SystemExit(f'{SCORE_LOG} does not end with "wrote {SUMMARY}": {last!r}')
    S = json.loads(SUMMARY.read_text())
    if tuple(S['arms']) != ARMS or tuple(S['joint_arms']) != JOINT or tuple(S['bands']) != BANDS:
        raise SystemExit(f'{SUMMARY}: arms {S["arms"]} + {S["joint_arms"]} / bands {S["bands"]} differ from this script\'s')
    return (S, json.loads(CHECK_GEN.read_text()), json.loads(CHECK_PREMISE.read_text()),
            RUN_ARMS_LOG.read_text(), MAKE_LOG.read_text(), RASQUAL_LOG.read_text(),
            json.loads(TRECASE_SUMMARY.read_text()), json.loads(LADDER.read_text()))


def f(x, d=3):
    return f'{x:.{d}f}'


def ci(d, key='value', n=3):
    return f'{d[key]:.{n}f} [{d["lo"]:.{n}f}, {d["hi"]:.{n}f}]'


def ch_name(arm, ch):
    return f'{ch} ({MIX_CH[ch]})' if arm.startswith('mixqtl') else ch


def table(head, rows):
    h = ''.join(f'<th>{x}</th>' for x in head)
    b = ''.join('<tr>' + ''.join(f'<td>{x}</td>' for x in r) + '</tr>' for r in rows)
    return f'<table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table>'


def style(ax, ylabel=None):
    ax.grid(axis='y', color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color('#c3c2b7')
    ax.tick_params(colors=MUTED, labelsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK, fontsize=10)


def save(fig, name):
    path = OUT / f'{name}.png'
    tmp = path.with_name(path.name + '.tmp')
    fig.savefig(tmp, dpi=130, bbox_inches='tight', format='png')
    plt.close(fig)
    os.replace(tmp, path)
    return path


def arm_points(ax, arms, xs, ys, los=None, his=None, offset=0.09, line=True):
    """One series per arm over categorical x positions, dodged, with optional intervals."""
    for j, arm in enumerate(arms):
        dx = (j - (len(arms) - 1) / 2) * offset
        x = [v + dx for v in xs]
        y = ys[arm]
        if line:
            ax.plot(x, y, color=COLOR[arm], lw=1.5, alpha=0.8)
        if los is not None:
            ax.errorbar(x, y, yerr=[[a - b for a, b in zip(y, los[arm])], [b - a for a, b in zip(y, his[arm])]],
                        fmt='none', ecolor=COLOR[arm], elinewidth=1.2, capsize=0)
        ax.plot(x, y, MARKER[arm], color=COLOR[arm], ms=7, mec='white', mew=0.8, label=LABEL[arm], ls='none')


def fig_ranking(S):
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.2))
    xs = range(len(BETAS))
    R = {b: S['ranking'][f'beta{b}'] for b in BETAS}
    arm_points(axs[0], ALL, xs, {a: [R[b][a]['auc']['all']['mean'] for b in BETAS] for a in ALL},
               {a: [R[b][a]['auc']['all']['lo'] for b in BETAS] for a in ALL},
               {a: [R[b][a]['auc']['all']['hi'] for b in BETAS] for a in ALL}, offset=0.08)
    axs[0].axhline(0.5, color=MUTED, lw=0.8, ls=':')
    style(axs[0], 'AUC, genes ranked by lead nominal p')
    axs[0].set_title('A. AUC of the gene ranking', fontsize=10, loc='left')
    arm_points(axs[1], ALL, xs, {a: [R[b][a]['fdp_matched']['all']['power'] for b in BETAS] for a in ALL}, offset=0.08)
    style(axs[1], 'share of non-null genes called')
    axs[1].set_title('B. Power at 5% realized false-discovery proportion', fontsize=10, loc='left')
    G = {b: S['gene_level'][f'beta{b}'] for b in BETAS}
    arm_points(axs[2], HAPMIX, xs, {a: [G[b][a]['power_bh']['all']['rate'] for b in BETAS] for a in HAPMIX},
               {a: [G[b][a]['power_bh']['all']['lo'] for b in BETAS] for a in HAPMIX},
               {a: [G[b][a]['power_bh']['all']['hi'] for b in BETAS] for a in HAPMIX})
    style(axs[2], 'share of non-null genes discovered')
    axs[2].set_title('C. Gene level: map_cis pval_beta, Benjamini-Hochberg 5%', fontsize=10, loc='left')
    for ax in axs:
        ax.set_xticks(list(xs), [f'|beta| = {b}' for b in BETAS])
    axs[1].set_ylim(0, 1.02)
    axs[2].set_ylim(0, 1.02)
    legend_below(fig, axs[0])
    return save(fig, 'fig_ranking')


def fig_bias(S):
    fig, axs = plt.subplots(2, 4, figsize=(15, 7.4), sharex=True)
    for i, ch in enumerate(('allelic', 'total')):
        for k, bn in enumerate(BANDS):
            ax = axs[i, k]
            for j, arm in enumerate(ARMS):
                for m, b in enumerate(BETAS):
                    r = S['recovery'][f'beta{b}'][arm][ch]
                    d = r['bias_count'][bn]
                    x = j + (m - 1) * 0.24
                    ax.errorbar([x], [d['mean']], yerr=[[d['mean'] - d['lo']], [d['hi'] - d['mean']]], fmt='o',
                                color=BETA_COLOR[b], ms=5.5, mec='white', mew=0.6, elinewidth=1.2,
                                label=f'|beta| = {b}' if (j == 0 and i == 0 and k == 0) else None)
            ax.axhline(1, color=INK, lw=0.8)
            ax.axhline(0, color=MUTED, lw=0.6, ls=':')
            ax.axvline(3.5, color=GRID, lw=1)
            style(ax, f'{ch}: slope / truth' if k == 0 else None)
            ax.set_title(f'{ch}, {"all genes" if bn == "all" else bn + " reads"}', fontsize=10, loc='left')
            ax.set_xticks(range(len(ARMS)), [SHORT[a] for a in ARMS], rotation=40, ha='right')
    fig.tight_layout()
    legend_below(fig, axs[0, 0], y=-0.03)
    return save(fig, 'fig_bias')


def fig_lead(S):
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.9), sharey=True)
    xs = range(len(BETAS))
    for k, bn in enumerate(BANDS):
        L = {b: S['lead'][f'beta{b}'] for b in BETAS}
        arm_points(axs[k], ALL, xs, {a: [L[b][a][bn]['r2_high'] for b in BETAS] for a in ALL}, offset=0.08)
        style(axs[k], 'share of non-null genes, lead r^2 >= 0.8' if k == 0 else None)
        n = S['lead']['beta0.4']['gibbs'][bn]['units']
        axs[k].set_title(f'{"all genes" if bn == "all" else bn + " reads"} ({n} gene units per |beta|)', fontsize=10,
                         loc='left')
        axs[k].set_xticks(list(xs), [f'{b}' for b in BETAS])
        axs[k].set_xlabel('|beta| (log2)', color=MUTED, fontsize=9)
    axs[0].set_ylim(0, 1.02)
    legend_below(fig, axs[0], y=-0.2)
    return save(fig, 'fig_lead')


def fig_efficiency(S):
    fig, axs = plt.subplots(2, 3, figsize=(15, 7.6))
    for i, part in enumerate(('nonnull', 'null')):
        scen = BETAS if part == 'nonnull' else ('0.0',) + BETAS
        xs = list(range(len(scen)))
        for k, ch in enumerate(('allelic', 'total', 'combined')):
            ax = axs[i, k]
            P = {b: S['precision'][f'beta{b}'] for b in scen}
            # combined: every arm, and at the causal variant the count-scale truth for arm and unit alike
            arms = [a for a in (ALL if ch == 'combined' else HAPMIX if part == 'nonnull' else ARMS) if a != 'unit']
            rk = 'ratio_vs_unit_count' if (part, ch) == ('nonnull', 'combined') else 'ratio_vs_unit'
            get = lambda a, key: [P[b][a][ch][part][rk]['all'][key] for b in scen]
            arm_points(ax, arms, xs, {a: get(a, 'value') for a in arms}, {a: get(a, 'lo') for a in arms},
                       {a: get(a, 'hi') for a in arms}, offset=0.12)
            ax.axhline(1, color=INK, lw=0.8)
            ax.set_yscale('log')
            ax.set_ylim(min(min(get(a, 'lo')) for a in arms) / 1.15, max(max(get(a, 'hi')) for a in arms) * 1.15)
            ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(LOG_TICKS))
            ax.yaxis.set_major_formatter(matplotlib.ticker.FixedFormatter([f'{t:g}' for t in LOG_TICKS]))
            ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
            style(ax, ('causal variant' if part == 'nonnull' else 'null genes, every tested variant')
                  + '\nsquared error, arm / unit' if k == 0 else None)
            ax.set_xticks(xs, ['anchor' if b == '0.0' else f'|beta| {b}' for b in scen])
            ax.set_title(f'{ch} ({"non-null genes" if part == "nonnull" else "null genes"}'
                         f'{", count-scale truth" if rk == "ratio_vs_unit_count" else ""})', fontsize=10, loc='left')
    fig.tight_layout()
    legend_below(fig, axs[0, 2], y=-0.04)
    return save(fig, 'fig_efficiency')


def legend_below(fig, ax, y=-0.08):
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', bbox_to_anchor=(0.5, y), ncol=min(len(l), 4), fontsize=8.5, frameon=False)


def img(path, caption):
    alt = html.escape(caption, quote=True)
    return (f'<figure><img alt="{alt}" src="data:image/png;base64,'
            f'{base64.b64encode(path.read_bytes()).decode()}"><figcaption>{alt}</figcaption></figure>')


def log_facts(run_log, make_log):
    zeroed = [int(x) for x in re.findall(r'allelic admission zeroed (\d+) donor-gene pairs', run_log)]
    mix = {}
    for arm, na, nt, med in re.findall(r' (mixqtl\S*) +mixqtl_scan [\d,]+ rows; genes with allelic samples >= 15: '
                                       r'(\d+), total samples >= 15: (\d+); allelic samples per gene median (\d+)',
                                       run_log):
        mix.setdefault(arm, []).append((int(na), int(nt), int(med)))
    pairs = re.search(r'donor-gene pairs ([\d,]+); with haplotype-informative reads \(pL \+ pR > 0\) ([\d,]+); '
                      r'with min\(pL, pR\) >= 0.5 ([\d,]+)', make_log).groups()
    tested = [f'{int(x):,}' for x in re.search(r'tested variants per gene min (\d+) / median (\d+) / max (\d+)',
                                                make_log).groups()]
    last = make_log.rstrip().splitlines()[-1]
    expr = re.search(r'expressible share of het donor-gene pairs in non-null genes ([\d.]+); reads removed per '
                     r'donor / median library \(beta > 0\): median ([\d.e+-]+), max ([\d.e+-]+)', last).groups()
    if not zeroed or set(mix) != {'mixqtl', 'mixqtl_permissive'}:
        raise SystemExit(f'{RUN_ARMS_LOG}: admission or mixQTL lines not found')
    return dict(zeroed=(min(zeroed), max(zeroed)), mix=mix, pairs=pairs, tested=tested, expr=expr)


def joint_facts(S, rq_log, TS):
    """Run counts of the joint arms, pooled over datasets: RASQUAL from its log, TReCASE from its summary.json."""
    rqd = {k: [int(x.replace(',', '')) for x in m] for k, *m in re.findall(
        r'(?m)^(beta\S+ rep \d+): ([\d,]+) rows written; excluded non-converged (\d+), pseudo-fSNP rows \d+; tested '
        r'variants with no RASQUAL row (\d+); chisq <= 0 \(slope_se NaN\) (\d+); genes where RASQUAL did not admit '
        r'the pseudo fSNP (\d+); non-null causal variants without a row: non-converged (\d+), absent (\d+)', rq_log)}
    hetd = {k: (int(h), int(z)) for k, h, z in re.findall(
        r'(?m)^(beta\S+ rep \d+): \d+ covariates; pseudo fSNP (\d+) het of \d+ informative pairs.*AS 0,0 (\d+)$', rq_log)}
    rq, het = list(rqd.values()), [h for h, _ in hetd.values()]
    n_ds = sum(S['n_datasets'].values())
    if (len(rq) != n_ds or set(hetd) != set(rqd) or set(TS['per_dataset']) != set(rqd)
            or not rq_log.rstrip().splitlines()[-1].startswith('wrote')):
        raise SystemExit(f'{RASQUAL_LOG}: {len(rq)} dataset lines, {len(het)} admission lines, want {n_ds} matching '
                         f'{TRECASE_SUMMARY} and a final "wrote"')
    tot = [sum(x[i] for x in rq) for i in range(7)]
    rasqual = dict(zip(('rows', 'nonconv', 'absent', 'chisq_le0', 'no_fsnp', 'causal_nonconv', 'causal_absent'), tot),
                   nonconv_range=(min(x[1] for x in rq), max(x[1] for x in rq)), het=(min(het), max(het)),
                   chisq_le0_anchor=rqd['beta0.0 rep 000'][3], as00=(min(z for _, z in hetd.values()),
                                                                    max(z for _, z in hetd.values())))
    rasqual['tests'] = rasqual['rows'] + rasqual['nonconv'] + rasqual['absent']
    per = TS['per_dataset'].values()
    if len(per) != n_ds or TS['smoke']:
        raise SystemExit(f'{TRECASE_SUMMARY}: {len(per)} datasets, smoke {TS["smoke"]}; want the full run of {n_ds}')
    # records RASQUAL made heterozygous (= the arms' admitted allelic records) that asSeq's min.AS.reads then drops
    drop = [hetd[k][0] - c['as_records_admitted'] for k, c in TS['per_dataset'].items()]
    trace = lambda k: sum(c['joint_na_by_trace'][k] for c in per)
    trecase = dict(TS['pooled'], linear_dosage=trace('trec_linear_dosage'), ase_fail=trace('ase'),
                   few_het=sum(c['ase_na_few_het'] for c in per), df_not_1=sum(c['final_df_not_1'] for c in per),
                   constant=sorted({c['tested_constant_dosage'] for c in per}),
                   theta_gradient_max=max(c['joint_na_by_trace']['theta_fail_abs_gradient_max'] for c in per),
                   theta_codes=sorted({k for c in per for k in c['joint_na_by_trace']['theta_fail_codes']}),
                   zeroed=(min(c['informative_zeroed_not_allelic_kept'] for c in per),
                           max(c['informative_zeroed_not_allelic_kept'] for c in per)),
                   asseq_admitted=(min(c['as_records_admitted'] for c in per), max(c['as_records_admitted'] for c in per)),
                   asseq_dropped=(min(drop), max(drop)), df_not_1_anchor=TS['per_dataset']['beta0.0 rep 000']['final_df_not_1'],
                   trec_na=sum(c['trec_na'] for c in per))
    return dict(rasqual=rasqual, trecase=trecase)


def one_df_gene(S):
    """The one gene whose allelic fit has one residual df (score.py ONE_DF); the text is written for exactly one."""
    if len(S['one_df_genes']) != 1:
        raise SystemExit(f'{SUMMARY}: one-df genes {S["one_df_genes"]}; the text assumes exactly one')
    return S['one_df_genes'][0]


def sec_run(S, CG, CP, LF, JF):
    one_df = one_df_gene(S)
    th, idn, rc = CG['thinning'], CG['identity'], CG['recovery']
    pr = rc['primary']
    tn, rl = th['thinned'], th['real']
    fano = table(['haplotype-informative reads (pL + pR)', 'donor-gene pairs, thinned (real)',
                  'median Fano factor, thinned', 'median Fano factor, real'],
                 [[b, f'{tn[b]["pairs"]:,} ({rl[b]["pairs"]:,})', f(tn[b]['fano']), f(rl[b]['fano'])]
                  for b in ('1-9', '10-99', '100-999', '1000+')])
    rec = table(['allelic slope estimate (genes >= 100 reads)', 'mean slope / truth', 'gene-clustered se'],
                [[name, f(pr[k]['mean']), f(pr[k]['gene_clustered_se'])] for name, k in (
                    ('weights 1/Va from the unthinned record (do not depend on the effect)', 'inv_va_real_beta'),
                    ('weights 1/Va at the expected thinned counts', 'inv_va_exp_beta'),
                    ('weights 1/Va\' at the realized thinned counts (the arms\' weights), vs beta', 'inv_va_beta'),
                    ('unit weights, vs beta', 'unit_beta'),
                    ('weights 1/Va\', vs pipeline-scale truth', 'inv_va_pipeline'),
                    ('unit weights, vs pipeline-scale truth (the pass rule)', 'unit_pipeline'))])
    if 'reproduction' in CG:
        rp = CG['reproduction']
        repro = ('Check (d), exact reproduction of a stored null: given the stored null runs\' own permutation 0, '
                 'the beta = 0 path reproduced that run\'s first permutation draw: ' + '; '.join(
                     f'{a}: {v["tests"]:,} tests, calls at 0.05 that differ {sum(v["call_differences"].values())}, '
                     f'largest slope difference {v["max_slope_diff_se"]:.1e} se' for a, v in rp.items()) + '.')
    else:
        repro = (f'Check (d), exact reproduction of a stored null permutation, is NOT in {CHECK_GEN.name}: that file '
                 'was written before the check was added to check_generator.py, and the committed check has not '
                 'been run into it. The only record of the reproduction is the one-off run described in the '
                 'docstring of score.py (section 6): given the stored run\'s own permutation 0, the beta = 0 path '
                 'reproduced that run\'s draw 0 for the unit and gibbs arms with no call at 0.05 differing among '
                 '487,454 tests in any channel, slopes within 5.5e-6 se. It is quoted from that docstring, not '
                 'from a check file.')
    prem_by_s = '; '.join(f's in {k}: {f(v["ratio_median"], 2)} ({v["genes"]:,} genes)'
                          for k, v in CP['by_ambiguous_share'].items())
    b10 = rc['by_band']['10-99']
    n_ds = S['n_datasets']
    mix = LF['mix']
    rng = lambda arm, i: '-'.join(dict.fromkeys(str(g(x[i] for x in mix[arm])) for g in (min, max)))
    rq, tr = JF['rasqual'], JF['trecase']
    miss = lambda a: ' / '.join(str(S['missing_causal'][f'beta{b}'][a]) for b in BETAS)
    return f'''
<h2>2. What was run</h2>
<p><b>Generator.</b> Each dataset starts from the real cohort's Salmon output for the {LF["pairs"][0]} donor-gene
pairs of the 100 genes of the corrected null store (92 donors; {LF["pairs"][1]} pairs have
haplotype-informative reads). Three steps turn it into a dataset with a known answer. First, the real
associations are broken: donor records are permuted against fixed genotypes. A record's point estimates,
Gibbs draws, library size and RNA-tied covariates move together; the genotype principal components stay with
the genotypes; each moved record's L and R labels are swapped with probability one half. Second, an effect is
injected. For each non-null gene one causal variant is drawn among its tested variants
({LF["tested"][0]} to {LF["tested"][2]} per gene, median {LF["tested"][1]}), with |beta| = 0.2, 0.4 or 0.8 log2
units and a random sign (beta = 1 would be a twofold allelic effect). On each donor, the haplotype carrying the
lower-expressed allele keeps each of its reads with probability f = 2<sup>-|beta|</sup>. This is binomial thinning
(Gerard 2020, BMC Bioinformatics): the data keep their own noise, depth and donor-to-donor structure and gain
only the chosen signal. Null genes are thinned by the same average factor, so null and non-null genes sit at
the same depth. Third, the Gibbs variance of a thinned record is set. For the total channel, every Gibbs draw
is thinned like the point estimate. For the allelic channel, the real record's Gibbs variance is scaled by the
ratio of the counting term q = 1/(pL + 0.5) + 1/(pR + 0.5) at the thinned over the real counts, where pL and pR are
Salmon's point-estimate read counts on the L and R haplotypes. That rule is
read from Salmon 1.10.3's Gibbs sampler (CollapsedGibbsSampler.cpp lines 149, 257-265 and 507): reads shared by
both haplotypes carry no allelic information, so the allelic Gibbs variance scales with one over the
haplotype-specific reads, and thinning scales those reads by f.</p>
<p><b>Datasets.</b> {n_ds["0.0"]} beta = 0 anchor dataset (every gene null, no thinning) and
{n_ds["0.2"]} / {n_ds["0.4"]} / {n_ds["0.8"]} datasets at |beta| = 0.2 / 0.4 / 0.8, each with 50 of the 100 genes
non-null. Dataset r uses the same permutation, causal variants, signs and null genes at every |beta|, so the
effect sizes are paired, not independent replicates. A gene is non-null in about 1.5 of the 3 datasets of a scenario, so every statistic below is pooled
over gene-dataset units (a <i>causal unit</i> is one non-null gene in one dataset, at its causal variant).
Genes are grouped into three <i>read bands</i> by their real median haplotype-informative reads over donors
(fewer than 100, 100-999, at least 1,000). Among heterozygous donor-gene pairs of non-null genes, a share of
{LF["expr"][0]} had both haplotypes at 0.5 reads or more before thinning. Reads removed per donor, as a fraction
of the cohort's median effective library size, were at most {float(LF["expr"][2]):.1e} (median
{float(LF["expr"][1]):.1e}), so library sizes were left unchanged.</p>
<p><b>Arms.</b> Four hapmixQTL weightings, all in default mode, Var(eps) = sigma<sup>2</sup> v: eps is a record's
residual, v its Gibbs variance, and sigma<sup>2</sup> the residual scale fitted per variant. All run after the
zero-haplotype admission rule (an allelic record with exactly one
haplotype below 0.5 reads is excluded; {LF["zeroed"][0]}-{LF["zeroed"][1]} donor-gene pairs per dataset):
<b>gibbs</b>, weights 1/v in both channels (the shipped default); <b>split</b>, 1/v in the allelic channel and
weight 1 in the total channel; <b>unit</b>, weight 1 in both channels; <b>plus_one</b>, 1/(v + 1) in both
channels. Two mixQTL-mode arms run on the thinned point estimates, never
on the draws: <b>published cutoffs</b> (total reads 100, allelic reads 50 to 1,000, weight cap 10) and
<b>permissive cutoffs</b> (20, 5 to 5,000, cap 100). The realized fold cap is min(weight cap, floor(n/10)) for n
admitted donors, at most 9 with 92 donors, so the two weight-cap settings act identically and only the count
cutoffs differ between the mixQTL arms. mixQTL's combined estimate, its <i>meta statistic</i>, is the
inverse-variance combination of its allelic (asc) and total (trc) estimates. Its natural-log slopes and standard
errors are divided by ln 2. Under the published cutoffs {rng("mixqtl", 0)} of 100 genes had at least 15 allelic
donors per dataset (median allelic donors per gene {rng("mixqtl", 2)}); under the permissive cutoffs
{rng("mixqtl_permissive", 0)} (median {rng("mixqtl_permissive", 2)}). The hapmixQTL arms were also run through
map_cis for gene-level p (1,000 permutations of donor records with haplotype-label swaps, GPU), with the Beta
approximation: a Beta distribution fitted to the permuted minimum p values, used to smooth the gene-level p
(section 3.2). mixQTL's own permutation scan was timed in the smoke run, on the first dataset
({S["mixqtl_permutation"]["timed_on"]}), with the host loaded (load about 100, run_arms.py docstring): it took
{S["mixqtl_permutation"]["seconds"]:.0f} s for {S["mixqtl_permutation"]["genes_done"]} of
{S["mixqtl_permutation"]["genes"]} genes ({S["mixqtl_permutation"]["tested_variants_done"]:,} of
{S["mixqtl_permutation"]["tested_variants"]:,} tested variants). Extrapolated in proportion to tested variants,
that is about {S["mixqtl_permutation"]["seconds_per_dataset_extrapolated"]:,.0f} s per dataset against a
{S["mixqtl_permutation"]["budget_s"]:.0f} s budget, so the timing rule excluded it for this run too. The exclusion
rests on that one loaded measurement. mixQTL is therefore compared only on the within-dataset measures that need no
gene-level p.</p>
<p><b>Joint models.</b> Two published methods that fit the total and allele-specific counts in one likelihood were
run on every dataset, nominal only (no gene-level p). Each gives one test per variant, scored here as its combined
channel; their allelic and total rows read n/a. <b>RASQUAL</b> models total counts as negative binomial and
allele-specific counts as beta-binomial (the two count models that allow <i>overdispersion</i>, variance of the
counts beyond that of a Poisson or binomial count), sharing one allelic parameter, and adds a reference-mapping bias,
a sequencing error rate and genotype uncertainty; it reports a <i>likelihood-ratio statistic</i> &chi;<sup>2</sup>,
twice the gain in log-likelihood when the variant's effect is added to the model. With no
reads to give it, each gene gets one pseudo feature SNP in its gene body, at which every donor-gene pair the hapmixQTL
arms admit to the allelic channel is heterozygous with allele counts equal to its thinned haplotype point estimates
rounded to integers ({rq["het"][0]:,} to {rq["het"][1]:,} pairs per dataset); the tested variants carry the real phased
genotypes, the total counts are the thinned Salmon totals as they are (fractional), the size factor is the effective
library size, and the 17 covariates are those of the other arms. RASQUAL's defaults are kept except its
Hardy-Weinberg filter on tested variants (a test that a variant's genotype counts match those expected from its allele
frequency), turned off (-h 0) because these genotypes are the truth and no other arm filters on it (run_rasqual.py). <b>TReCASE</b> (asSeq 0.99.501) models total counts as negative binomial (TReC) and
allele-specific counts as beta-binomial (ASE), fits both jointly, and runs a cis/trans test of whether the total and
allelic effects agree; asSeq's final p is the joint p when that test does not reject at 0.05 and the total-count p
otherwise, which is also what it reports when the joint fit fails. Its inputs are the same donor-gene pairs as allele-specific
records (counts rounded per haplotype, because its beta-binomial needs integers), fractional totals, the log effective
library size as offset, and the same 17 covariates; asSeq's defaults are kept except the p cutoff for writing a row
(run_trecase_asseq.py).</p>
<p><b>One scale for every method.</b> Every slope on this page is a log2 allelic fold change (aFC), ALT over REF,
where beta = 1 is a twofold effect. The table gives each arm's published effect, its conversion, and where its
standard error comes from. RASQUAL and TReCASE report no standard error: it is derived as |slope| / &radic;&chi;<sup>2</sup>
(the Wald inversion of &chi;<sup>2</sup>: the standard error at which (slope / se)<sup>2</sup> equals
&chi;<sup>2</sup>), so under truth 0 (null genes) z = slope / se is &plusmn;&radic;&chi;<sup>2</sup>
by construction, and their realized-over-stated standard error on null genes (section 3.4) measures the calibration of
their likelihood-ratio test, not a reported standard error. At the causal variant z = (slope &minus; beta) / se instead
measures how well the derived se describes the slope's spread around beta, and absorbs bias. For
comparisons across methods every arm is held to the count-scale truth (defined in section 3.3); the pipeline-scale
truth is a hapmixQTL-only diagnostic and never ranks methods. In both joint models the total mean at ALT dosage
0, 1, 2 is proportional to 1, (1 + &kappa;)/2, &kappa; for an ALT over REF ratio &kappa;. That is the expected form,
averaged over donors, of the total fold the generator injects: thinning one haplotype's reads and the shared reads by
different factors gives exactly this form only for a donor whose haplotype-specific reads are balanced, and on average
over the random direction of real imbalance. Their estimand is therefore log2 &kappa; = beta, also when asSeq falls back
to its total-count test, whose model has the same dosage form (glm.c:1577); the per-gene total truth, a straight-line fit
of that fold on g/2, is the estimand of the linear total channels of hapmixQTL and mixQTL. One asSeq fallback cannot be
identified per test: where its total-count dosage model fails it refits the dosage as a linear covariate, whose slope
is a log fold per ALT allele, about half of ln &kappa;, so about half of beta after the conversion
({tr["linear_dosage"]:,} of {tr["tests"]:,} tests, {100 * tr["linear_dosage"] / tr["tests"]:.1f}%). Such a row at a
causal variant lowers TReCASE's bias ratio and raises its squared error; the rows are not flagged, so how many fall at
causal variants is not known.</p>
{tab_conversion()}
<p><b>Missing rows.</b> RASQUAL's rows where its fit did not converge are left out: {rq["nonconv"]:,} of {rq["tests"]:,}
tests ({rq["nonconv_range"][0]:,} to {rq["nonconv_range"][1]:,} per dataset). Another {rq["chisq_le0"]:,} rows have
&chi;<sup>2</sup> &le; 0 (p = 1), whose derived standard error is undefined: they are left out of the standard-error
statistics only, and stay in the ranking, the null rates and squared error. In one gene, {one_df}, in every dataset
({rq["no_fsnp"]} gene-datasets), RASQUAL did not admit the pseudo feature SNP and fitted the total counts alone; it is
the gene with two allelic donors discussed in section 3.7. TReCASE has no row for the
{"/".join(f"{x:,}" for x in tr["constant"])} tested pairs per dataset whose ALT dosage is the same in every donor, among
them TPPP's causal variant in dataset 2 of each |beta| &gt; 0 scenario. Causal
units without a row, at |beta| = 0.2 / 0.4 / 0.8 (of {S['lead']['beta0.4']['gibbs']['all']['units']} each): RASQUAL
{miss('rasqual')}, TReCASE {miss('trecase')}. They are left out of that arm's causal-variant detection shares, and in
bias and precision they are non-finite and so excluded and counted like any other non-finite unit.</p>
<p><b>Checks before the run.</b> The premise of the allelic rule was tested on donor {CP["sample"]}'s dumped
equivalence classes: the observed allelic Gibbs variance beyond counting noise, over the variance predicted
from the shared-read share s, has median {f(CP["ratio_median"])} (interquartile range {f(CP["ratio_iqr"][0])} to
{f(CP["ratio_iqr"][1])}) over {CP["genes_retained"]:,} of the {CP["genes_min_u"]:,} genes with at least
{CP["min_u"]} haplotype-specific reads on each haplotype ({CP["genes_excess_le_0"]} dropped for non-positive
excess, {CP["genes_s_eq_0"]} for s = 0). The ratio is not flat in s ({prem_by_s}), so the median is set
mainly by the largest group. The rule under-predicts the excess where informative reads are a larger share and over-predicts it
where almost all reads are shared; the generator's rule does not depend on s, because thinning leaves s
unchanged. The prediction ranks genes' excess with Spearman correlation (the Pearson correlation of the ranks)
{f(CP["spearman_excess_s2H"])} against {f(CP["spearman_excess_counting"])} for the counting term alone. Its pass
thresholds were set after that first result, so it guards the derivation against regression rather than testing
it independently. Check (a), identity: with every thinning factor 1,
the generator reproduces the pipeline's inputs exactly: A, the allelic log2 ratio log2((pL + 0.5)/(pR + 0.5)); T,
the total log2(CPM + 1); and Va and Vt, their Gibbs variances (largest difference in A after a permutation and swap
{idn["max_abs_dA"]:.1e}). Check (b), thinning: the Fano factor (across-draw variance over mean) of the
total Gibbs draws stays at its real value after thinning by f = {th["f"]}, and the allelic rule's arithmetic
holds to {th["allelic_rule"]["max_rel_dev"]:.1e} relative over {th["allelic_rule"]["records_checked"]:,}
records:</p>
{fano}
<p>Check (c), recovery of an injected |beta| = {rc["beta"]} over {rc["n_datasets"]} all-non-null datasets
({pr["unit_pipeline"]["genes"]} genes with at least 100 reads, {pr["unit_pipeline"]["units"]:,} units). Unit weights
recover the pipeline-scale truth ({f(pr["unit_pipeline"]["mean"])}, gene-clustered se
{f(pr["unit_pipeline"]["gene_clustered_se"])}), which is the pass rule. Here the gene-clustered se is the standard
deviation of the per-gene means over the square root of the number of genes, as check_generator.py computes it; it is
not the resampling interval used in section 3. Write Va for the allelic Gibbs variance of an unthinned record
and Va' for that of a thinned record. The 1/Va' weights the arms use recover
{f(pr["inv_va_beta"]["mean"])} of beta, against {f(pr["inv_va_real_beta"]["mean"])} when the weights come from the
unthinned record's Va. Almost all of that shortfall appears when the weights are evaluated at the expected thinned
counts ({f(pr["inv_va_exp_beta"]["mean"])}), before any binomial noise. On one truth, the pipeline scale, 1/Va'
weights recover {f(pr["inv_va_pipeline"]["mean"])} (gene-clustered se {f(pr["inv_va_pipeline"]["gene_clustered_se"])})
against {f(pr["unit_pipeline"]["mean"])} ({f(pr["unit_pipeline"]["gene_clustered_se"])}) for unit weights. So 1/v
weights that follow the thinned counts attenuate the allelic slope by about 5%, and unit weights do not. One
explanation, consistent with these numbers but not tested separately: a donor whose effect thinned its already
smaller haplotype gets a larger v and a smaller weight, and it is the donor whose allelic ratio already lay in the
effect's direction before thinning; down-weighting those donors leaves the weighted mean of their pre-existing
imbalances pointing against the effect. The attenuation is a property of 1/v weighting when v tracks the counts,
which the premise check says Salmon's Gibbs variance does, not a generator defect; it applies to the allelic
channel of gibbs, split and plus_one below. In the 10-99 read band both weightings fall short of the
pipeline-scale truth: {f(b10["inv_va_pipeline"]["mean"])} (gene-clustered se
{f(b10["inv_va_pipeline"]["gene_clustered_se"])}) for 1/Va' and {f(b10["unit_pipeline"]["mean"])}
({f(b10["unit_pipeline"]["gene_clustered_se"])}) for unit weights, over {b10["unit_pipeline"]["genes"]} genes.
That shortfall is not decomposed; check_generator.py names one untested candidate, the zero-haplotype admission
rule, which conditions on the thinned outcome.</p>
{rec}
<p>{repro}</p>'''


def tab_ranking(S):
    R = {b: S['ranking'][f'beta{b}'] for b in BETAS}
    rows = []
    for a in ALL:
        fm = lambda b: R[b][a]['fdp_matched']
        rows.append([LABEL[a]] + [ci(R[b][a]['auc']['all'], 'mean') for b in BETAS]
                    + [f'{f(fm(b)["all"]["power"])} ({fm(b)["discoveries"]} called, {fm(b)["false"]} null)' for b in BETAS])
    return table(['arm'] + [f'AUC, |beta| {b}' for b in BETAS] + [f'power at 5% FDP, |beta| {b}' for b in BETAS], rows)


def tab_bands(S, block, key):
    """Per arm and |beta|, the value in the three read bands, '<100 / 100-999 / >=1000'."""
    rows = [[LABEL[a]] + [' / '.join(f(block(S, b, a)[bn][key], 2) for bn in BANDS[1:]) for b in BETAS] for a in ALL]
    return table(['arm'] + [f'|beta| {b}: &lt;100 / 100-999 / &ge;1000' for b in BETAS], rows)


def tab_gene_level(S):
    rows = []
    for a in HAPMIX:
        G = lambda sc: S['gene_level'][sc][a]
        nr = lambda sc: G(sc)['null_rate_pval_beta']['all']
        rows.append([LABEL[a]] + [f'{ci(G(f"beta{b}")["power_bh"]["all"], "rate")} ({G(f"beta{b}")["discoveries"]} '
                                  f'called, {G(f"beta{b}")["false_discoveries"]} null)' for b in BETAS]
                    + [f'{nr(sc)["rejections"]} of {nr(sc)["tests"]}' for sc in ('beta0.0',) + tuple(f'beta{b}' for b in BETAS)])
    return table(['arm'] + [f'BH power, |beta| {b}' for b in BETAS]
                 + [f'null genes with pval_beta &lt; 0.05, |beta| {b}' for b in ('0',) + BETAS], rows)


def tab_bias(S):
    n = lambda d: f' <span class="pipe">({d["units"]} / {d["excluded_nonfinite"]})</span>'
    rows = []
    for a in ALL:
        for ch in CHANNELS:
            if ch not in S['recovery']['beta0.4'][a]:
                rows.append([LABEL[a], ch, *['n/a'] * len(BETAS)])
                continue
            vals = []
            for b in BETAS:
                r = S['recovery'][f'beta{b}'][a][ch]
                txt = ci(r['bias_count']['all'], 'mean', 2) + n(r['bias_count']['all'])
                if 'bias_pipeline' in r:
                    txt += (f'<br><span class="pipe">{ci(r["bias_pipeline"]["all"], "mean", 2)}</span>'
                            + n(r['bias_pipeline']['all']))
                vals.append(txt)
            rows.append([LABEL[a], ch_name(a, ch)] + vals)
    return table(['arm', 'channel'] + [f'|beta| {b}: count scale<br><span class="pipe">pipeline scale</span> '
                                       f'(units / excluded)' for b in BETAS], rows)


def rkey(a):
    """Efficiency against unit weights at the causal variant: hapmixQTL arms on their shared pipeline-scale truth;
    every other method on the count-scale truth for the arm and unit alike."""
    return 'ratio_vs_unit' if a in HAPMIX else 'ratio_vs_unit_count'


def tab_precision(S, key):
    rows = []
    arms = ALL if key == 'sd_z' else tuple(a for a in ALL if a != 'unit')
    for a in arms:
        k = key if key == 'sd_z' else rkey(a)
        joint_z = a in JOINT and key == 'sd_z'
        name = LABEL[a] + (', derived se (Wald inversion of &chi;<sup>2</sup>): causal-variant columns absorb bias; '
                           'null column = calibration of the likelihood-ratio test (its square is the mean '
                           '&chi;<sup>2</sup>)' if joint_z else ', count-scale truth' if k == 'ratio_vs_unit_count' else '')
        for ch in CHANNELS:
            P = lambda sc: S['precision'][sc][a][ch]
            if ch not in S['precision']['beta0.0'][a]:
                rows.append([LABEL[a], ch, *['n/a'] * (len(BETAS) + 1)])
                continue
            rows.append([name, ch_name(a, ch)] + [ci(P(f'beta{b}')['nonnull'][k]['all'], 'value', 2) for b in BETAS]
                        + [ci(P('beta0.0')['null'][key]['all'], 'value', 3 if joint_z else 2)])
    return table(['arm', 'channel'] + [f'causal variant, |beta| {b}' for b in BETAS]
                 + ['null genes, beta 0 anchor'], rows)


def tab_cross(S):
    """Combined channel, every arm against unit weights, both on the count-scale truth."""
    P = lambda sc, a, part, k: S['precision'][sc][a]['combined'][part][k]['all']
    rows = [[LABEL[a]] + [ci(P(f'beta{b}', a, 'nonnull', 'ratio_vs_unit_count'), 'value', 2) for b in BETAS]
            + [ci(P('beta0.0', a, 'null', 'ratio_vs_unit'), 'value', 2)] for a in ALL if a != 'unit']
    return table(['arm (combined channel)'] + [f'causal variant, |beta| {b}' for b in BETAS]
                 + ['null genes, beta 0 anchor'], rows)


def cross_note(S):
    """gibbs's combined excess over unit weights on the count-scale truth, against the pipeline-scale one."""
    c = [prec(S, f'beta{b}', 'gibbs', 'combined', 'nonnull', 'ratio_vs_unit_count') for b in BETAS]
    inc = [b for b, d in zip(BETAS, c) if d['lo'] <= 1 <= d['hi']]
    B = lambda a: per_beta(lambda b: bias(S, b, a, 'total', 'bias_count')['mean'])
    return f"""
<p>The choice of truth matters for gibbs. On this count-scale truth its combined squared error at the causal variant is
{' / '.join(ci(d, 'value', 2) for d in c)} of unit weights' at |beta| = 0.2 / 0.4 / 0.8, with intervals that include 1
at {at_betas(inc)}; on the pipeline-scale truth (next table and section 4) it is
{' / '.join(ci(prec(S, f'beta{b}', 'gibbs', 'combined', 'nonnull', 'ratio_vs_unit'), 'value', 2) for b in BETAS)}.
On the count-scale truth unit weights' error contains the attenuation of log2(CPM + 1): their total slope recovers
{B('unit')} of the count-scale total truth, gibbs's {B('gibbs')} (section 3.3), which is consistent with the smaller
count-scale ratios but was not separated. The same denominator enters every row of this table. The comparison that does
not depend on the truth is the anchor's null genes, where the truth is 0 for every arm: there gibbs's combined squared
error is {ci(prec(S, 'beta0.0', 'gibbs', 'combined', 'null', 'ratio_vs_unit'), 'value', 2)} of unit weights'.</p>"""


def tab_conversion():
    """Each arm's published effect, its conversion to log2 aFC and where its standard error comes from."""
    truth = ('allelic: beta; total: per-gene total truth; combined: beta for bias, the inverse-variance combination '
             'of beta and the per-gene total truth for squared error')
    rows = [
        ['hapmixQTL, four weightings', 'allelic: slope of log2((L + 0.5)/(R + 0.5)) on xL &minus; xR; total: slope of '
         'log2(CPM + 1) on g/2; combined: their inverse-variance combination', 'none (log2 already)',
         'stated by the weighted least-squares fit', truth],
        ['mixQTL, two cutoff settings', 'the same three slopes on natural-log responses (asc log(L/R), trc '
         'log(total / 2 library size)); meta = inverse-variance combination', 'slope and se / ln 2',
         'stated by mixQTL\'s least-squares fits', truth],
        ['RASQUAL', '&pi;, the ALT allele\'s expected share of expression: RASQUAL scales expression by 2(1 &minus; &pi;), '
         '1, 2&pi; at ALT dosage 0, 1, 2 (nbem.c:1058)', 'log2(&pi; / (1 &minus; &pi;))',
         '<b>derived</b>: |slope| / &radic;&chi;<sup>2</sup>, &chi;<sup>2</sup> its likelihood-ratio statistic '
         '(RASQUAL reports none)', 'beta'],
        ['TReCASE (asSeq)', 'b = ln &kappa;, &kappa; the ALT over REF expression ratio, from the joint model, or from the '
         'total-count (TReC) model when asSeq\'s final p used it; the TReC mean is 1, (1 + &kappa;)/2, &kappa; at ALT '
         'dosage 0, 1, 2 (glmNBlog, glm.c:1577; the joint model, trecase.c:1049; genotypes recoded 3 &rarr; 1, '
         '4 &rarr; 2, R/trecase.R:205-206)',
         'b / ln 2', '<b>derived</b>: |slope| / &radic;&chi;<sup>2</sup> of the statistic used (asSeq reports none)',
         'beta, also for the total-count fallback (its model has the same dosage form)'],
    ]
    return table(['arm', 'published effect', 'conversion to log2 aFC, ALT over REF', 'standard error',
                  'count-scale truth used across methods'], rows)


def tab_lead(S):
    L = lambda b, a: S['lead'][f'beta{b}'][a]
    rows = [[LABEL[a]] + [f'{f(L(b, a)["all"]["lead_is_causal"], 2)} / {f(L(b, a)["all"]["r2_high"], 2)} / '
                          f'{f(L(b, a)["all"]["median_r2"], 2)}' for b in BETAS]
            + [' / '.join(str(L(b, a)['no_finite_p']) for b in BETAS)] for a in ALL]
    return table(['arm'] + [f'|beta| {b}: lead = causal / r<sup>2</sup> &ge; 0.8 / median r<sup>2</sup>' for b in BETAS]
                 + ['non-null gene units with no finite p (0.2 / 0.4 / 0.8)'], rows)


def tab_detection(S):
    D = lambda b, a, ch: S['detection'][f'beta{b}'][a].get(ch)   # None: a joint arm's allelic or total channel
    rows = [[LABEL[a], ch_name(a, ch)] + [' / '.join(f(D(b, a, ch)['all'][al], 2) for al in DETECT) if D(b, a, ch)
                                          else 'n/a' for b in BETAS] for a in ALL for ch in CHANNELS]
    return table(['arm', 'channel'] + [f'|beta| {b}: p &lt; 0.05 / 1e-3 / 1e-5' for b in BETAS], rows)


def tab_null(S):
    rows = [[LABEL[a], ch_name(a, ch)] + [ci(S['null'][sc][a][ch]['all']['0.05'], 'rate', 4) if ch in S['null'][sc][a]
                                          else 'n/a' for sc in ('beta0.0',) + tuple(f'beta{b}' for b in BETAS)]
            for a in ALL for ch in CHANNELS]
    return table(['arm', 'channel'] + [f'|beta| {b}' for b in ('0 (anchor)',) + BETAS], rows)


def tab_anchor(S):
    rows = []
    for a in HAPMIX:
        for ch in CHANNELS:
            r, r3 = S['anchor'][a][ch]['0.05'], S['anchor'][a][ch]['0.001']
            rows.append([LABEL[a], ch, f(r['rate'], 6), f(r['stored'], 4), f'{f(r["perm_lo"], 6)} to {f(r["perm_hi"], 6)}',
                         f'{r["percentile"]:.1f}', 'yes' if r['passed'] else 'no', f'{f(r3["rate"], 4)} ({f(r3["stored"], 4)})'])
    return table(['arm', 'channel', 'this dataset, 0.05', 'stored mean, 200 permutations',
                  'central 99% of stored permutations', 'percentile among stored', 'inside',
                  '0.001: this dataset (stored mean)'], rows)


CSS = '''
:root { --ink: #0b0b0b; --ink2: #52514e; --rule: #e1e0d9; --bg: #fcfcfb; --tint: #f3f2ee; }
body { background: var(--bg); color: var(--ink); font: 15px/1.55 -apple-system, "Segoe UI", Roboto, Helvetica, Arial,
       sans-serif; margin: 0; padding: 0 16px; }
main { max-width: 1080px; margin: 32px auto 64px; }
h1 { font-size: 26px; margin-bottom: 4px; } h2 { font-size: 20px; margin-top: 40px; border-bottom: 1px solid var(--rule);
padding-bottom: 4px; } h3 { font-size: 16px; margin-top: 28px; }
p { max-width: 900px; } .sub { color: var(--ink2); margin-top: 0; }
table { border-collapse: collapse; font-size: 12.5px; margin: 12px 0 18px; display: block; overflow-x: auto; }
th, td { border-bottom: 1px solid var(--rule); padding: 4px 8px; text-align: left; vertical-align: top; }
th { background: var(--tint); font-weight: 600; } td { font-variant-numeric: tabular-nums; }
.pipe { color: var(--ink2); } figure { margin: 16px 0 24px; } figure img { max-width: 100%; height: auto; }
figcaption { color: var(--ink2); font-size: 13px; max-width: 900px; }
.box { background: var(--tint); padding: 10px 14px; border-radius: 4px; max-width: 900px; }
'''


def sec_head(S):
    return ('<h1>Plasmode eQTL benchmark: recovering known cis effects</h1>'
            '<p class="sub">hapmixQTL weightings, mixQTL mode, RASQUAL and TReCASE on the BrainVar cohort\'s own '
            'Salmon output with '
            f'injected effects; 100 genes x 92 donors; runs of 2026-09-26 (hapmixQTL, mixQTL) and 2026-09-27 '
            f'(RASQUAL, TReCASE, the mixQTL ladder); units log2 aFC (beta = 1 is a twofold effect). Made by '
            f'scripts/plasmode/report.py from {SUMMARY}, {CHECK_GEN}, {CHECK_PREMISE}, {RUN_ARMS_LOG.name}, '
            f'{MAKE_LOG.name}, {RASQUAL_LOG}, {TRECASE_SUMMARY} and {LADDER}; figures also written as PNG in {OUT}.</p>')


def sec_why():
    return '''
<h2>1. Why the analysis was needed</h2>
<p>Until now the hapmixQTL weightings had been judged on null calibration only: whether the nominal p is
uniform when donor records are permuted against genotypes (the stored 100-gene, 200-permutation null runs).
A null says whether an arm's p values can be trusted. It cannot say how well an arm finds a real effect, how
close its slope comes to the true slope, or how much the Gibbs variance buys in precision, because real data
carry no known effect. No dataset with known cis effects existed. Simulating Salmon itself was rejected as
too slow, and datasets were built instead from the cohort's own Salmon output, keeping its depth, noise and
donor structure and adding a known effect.</p>
<p>The question: on data with the real cohort's structure, how well do the four hapmixQTL weightings, and
mixQTL mode (the published estimator, which never sees the Gibbs draws), rank non-null genes above null ones,
discover them at a controlled false-discovery rate, estimate the injected slope without bias, state their
standard error correctly, and place the lead variant on the causal one? The answers bear on the open decision
of which weighting ships (docs/pipeline_rules.md, "Open decision: which weighting configuration ships"), which
so far rests on null calibration alone.</p>
<p>mixQTL is one published way to use allele-specific and total counts together. RASQUAL and TReCASE are two others,
which fit both kinds of count in one likelihood; they were run on the same datasets as further comparators, with every
method's effect put on one scale. And because mixQTL trails the unit-weight arm, a separate run took the two apart one
change at a time (section 3.8).</p>'''


def auc(S, b, a, bn='all'):
    return S['ranking'][f'beta{b}'][a]['auc'][bn]


def fdp(S, b, a):
    return S['ranking'][f'beta{b}'][a]['fdp_matched']


def bh(S, b, a):
    return S['gene_level'][f'beta{b}'][a]


def prec(S, sc, a, ch, part, key, bn='all'):
    return S['precision'][sc][a][ch][part][key][bn]


def bias(S, b, a, ch, key, bn='all'):
    return S['recovery'][f'beta{b}'][a][ch][key][bn]


def per_beta(fn, n=3):
    """'x / y / z' over |beta| 0.2 / 0.4 / 0.8 of a number-returning fn(b)."""
    return ' / '.join(f(fn(b), n) for b in BETAS)


def per_scen(fn, n=3):
    """'w / x / y / z' over the beta = 0 anchor and |beta| 0.2 / 0.4 / 0.8."""
    return ' / '.join(f(fn(sc), n) for sc in ('beta0.0',) + tuple(f'beta{b}' for b in BETAS))


def interp_ranking(S):
    A = lambda a: per_beta(lambda b: auc(S, b, a)['mean'])
    P = lambda a: per_beta(lambda b: fdp(S, b, a)['all']['power'])
    thr = lambda b, a: f'{fdp(S, b, a)["p_threshold"]:.1e}'
    return f"""
<p>At |beta| = 0.2 / 0.4 / 0.8 the AUC is {A('split')} for split and {A('plus_one')} for plus_one,
{A('unit')} for unit and {A('gibbs')} for gibbs; mixQTL reaches {A('mixqtl')} with the published cutoffs and
{A('mixqtl_permissive')} with the permissive ones. The four hapmixQTL arms' ranges overlap at every |beta|. At
|beta| 0.8 split's lowest dataset AUC ({f(auc(S, '0.8', 'split')['lo'])}) exceeds mixQTL published's highest
({f(auc(S, '0.8', 'mixqtl')['hi'])}), so split ranks higher in each of the three datasets. A narrow range such as
unit's {ci(auc(S, '0.4', 'unit'), 'mean')} at |beta| 0.4 means three datasets happened to agree, not that the estimate
is precise, so arms should not be ordered by the width of these ranges.</p>
<p>Power at 5% realized FDP spreads the arms further: split {P('split')}, plus_one {P('plus_one')}, unit
{P('unit')}, gibbs {P('gibbs')}; mixQTL {P('mixqtl')} (published) and {P('mixqtl_permissive')} (permissive).
For gibbs at |beta| 0.2 no depth of the pooled ranking kept the null share at or below 5%: null genes sat at the
top of its ranking, so it called nothing. At 0.4 and 0.8 its cut fell at a lead p of {thr('0.4', 'gibbs')}
and {thr('0.8', 'gibbs')}, where split's fell at {thr('0.4', 'split')} and {thr('0.8', 'split')}: null genes' lead p
reached below split's cut, so to keep them out gibbs had to stop at smaller p. Section 4 takes up whether that
reflects null p values that are too small.</p>
<p>The ranking comparison between method families is conservative for the four hapmixQTL arms. In
{one_df_gene(S)} their allelic p is too small (section 3.4); mixQTL, RASQUAL and TReCASE all leave that gene's allelic
channel out. When {one_df_gene(S)} is a null gene, it can therefore sit near the top of only the hapmixQTL rankings
and tighten their realized-FDP cut. The ranking and its power were not recomputed without it, so the hapmixQTL power
at 5% realized FDP is a lower bound in the comparison with the other methods.</p>"""


def interp_gene_level(S):
    P = lambda a: per_beta(lambda b: bh(S, b, a)['power_bh']['all']['rate'])
    N = lambda a: ' / '.join(f'{bh(S, b, a)["false_discoveries"]} of {bh(S, b, a)["discoveries"]}' for b in BETAS)
    nr = lambda sc, a: S['gene_level'][sc][a]['null_rate_pval_beta']['all']
    worst = max(((nr(f'beta{b}', a)['rejections'], b, a) for b in BETAS for a in HAPMIX))
    wr = nr(f'beta{worst[1]}', worst[2])
    anc = ' / '.join(f'{nr("beta0.0", a)["rejections"]}' for a in HAPMIX)
    return f"""
<p>With each arm referred to its own permutation null, Benjamini-Hochberg power at |beta| = 0.2 / 0.4 / 0.8 is
{P('split')} for split, {P('gibbs')} for gibbs, {P('plus_one')} for plus_one and {P('unit')} for unit. The
gene-clustered intervals of the four arms overlap at every |beta| (table), so at {S['lead']['beta0.4']['gibbs']['all']['units']} non-null gene units per
effect size the arms are not distinguishable on gene-level power. Null genes among the calls: gibbs {N('gibbs')},
split {N('split')}, unit {N('unit')}, plus_one {N('plus_one')}; these are small counts without an interval.
Here the generator's null and map_cis's null are the same record permutation with label swaps (RNA-tied
covariates moving, genotype PCs fixed, one thinning factor per null gene), so the permutation p of a null gene
is valid by construction. The share of null gene units with pval_beta below 0.05 (at |beta| &gt; 0 at most
{worst[0]} of {wr["tests"]} in any arm and scenario, gene-clustered interval {f(wr["lo"])} to {f(wr["hi"])}; on the
anchor {anc} of {nr("beta0.0", "gibbs")["tests"]} for gibbs / split / unit / plus_one) therefore checks the
plumbing and the Beta approximation. It says nothing about calibration on real data, where the record
permutation may not match the sampling distribution of an observed statistic (section 6).</p>"""


def all17(LD):
    """Range over cutoffs and |beta| of the ladder's one-step all-17 total slope over the count-scale truth."""
    v = [LD['total_channel'][c][f'beta{b}']['one_step_all_trc']['all']['mean'] for c in ('published', 'permissive')
         for b in BETAS]
    return min(v), max(v)


def interp_bias(S, CG, LD):
    ladder_all17 = '{:.2f} to {:.2f}'.format(*all17(LD))
    B = lambda a, ch, key, bn='all': per_beta(lambda b: bias(S, b, a, ch, key, bn)['mean'])
    lo100 = lambda a: per_beta(lambda b: bias(S, b, a, 'allelic', 'bias_pipeline', '<100')['mean'], 2)
    pc = lambda b: bias(S, b, 'gibbs', 'allelic', 'bias_pipeline')
    cc = lambda b: bias(S, b, 'gibbs', 'allelic', 'bias_count')
    mq, mp = (bias(S, '0.4', a, 'allelic', 'bias_count') for a in ('mixqtl', 'mixqtl_permissive'))
    return f"""
<p><b>Total channel.</b> Every hapmixQTL arm recovers the pipeline-scale truth: {B('gibbs', 'total', 'bias_pipeline')}
for gibbs, {B('split', 'total', 'bias_pipeline')} for split and unit (which share one total-channel fit) and
{B('plus_one', 'total', 'bias_pipeline')} for plus_one, with intervals that include 1. Against the count-scale truth
split and unit's slopes read {B('split', 'total', 'bias_count')} (gibbs {B('gibbs', 'total', 'bias_count')}, plus_one
{B('plus_one', 'total', 'bias_count')}): the +1 of log2(CPM + 1) compresses a fold at low depth, and the
pipeline-scale truth removes that part.</p>
<p><b>Allelic channel.</b> gibbs and split fit the allelic channel identically (every allelic row of the two arms is
the same fit, not two arms agreeing). Their allelic slope recovers {B('gibbs', 'allelic', 'bias_pipeline')} of the
pipeline-scale truth, against {B('unit', 'allelic', 'bias_pipeline')} for unit weights and
{B('plus_one', 'allelic', 'bias_pipeline')} for plus_one. The gibbs/split and unit intervals overlap at every |beta|,
and the order is not constant: 1/v is higher at 0.2, about equal at 0.4 and lower at 0.8. The benchmark therefore
does not resolve a bias difference between 1/v and unit weights. The roughly 5% attenuation from 1/v weights rests
on check (c) (section 2): {f(CG['recovery']['primary']['inv_va_pipeline']['mean'])} against
{f(CG['recovery']['primary']['unit_pipeline']['mean'])} for unit weights, on the pipeline-scale truth, over
{CG['recovery']['n_datasets']} all-non-null datasets. Unit weights also fall short of the
pipeline-scale truth below 100 reads ({lo100('unit')}; gibbs and split {lo100('gibbs')}), with wide intervals (gibbs
and split at |beta| 0.8, {ci(bias(S, '0.8', 'gibbs', 'allelic', 'bias_pipeline', '<100'), 'mean')}). That shortfall is
neither the transform, which the pipeline-scale truth removes, nor weighting, since the weights are unit; it is not
attributed here. One untested candidate, named in check_generator.py: the zero-haplotype admission rule selects on
the thinned outcome.</p>
<p><b>Two denominators in the allelic rows.</b> In {pc('0.4')['excluded_nonfinite']} of the {cc('0.4')['units']}
causal units per |beta| the allelic channel has no admitted heterozygous donor: map_nominal returns slope 0 with an
infinite standard error and p = 1. Their pipeline-scale truth is undefined, so the pipeline line drops them
({pc('0.4')['units']} units); the count-scale truth is beta, so the count line keeps them as exact zeros
({cc('0.4')['units']} units), which lowers the count-scale mean by the factor {pc('0.4')['units']}/{cc('0.4')['units']}
against a mean over the units with data. Most of the gap between the two lines of a hapmixQTL allelic row is these
zeros, not the +0.5 pseudocount; the table prints each line's units and exclusions. The allelic sd(z) and efficiency ratios at
the causal variant (section 3.4) use the {pc('0.4')['units']} units with data. mixQTL's count-scale allelic rows drop
their units without an estimate, so they are on a different filter from the hapmixQTL count lines.</p>
<p><b>Combined slope.</b> Against beta it reads {B('gibbs', 'combined', 'bias_count')} for gibbs and
{B('split', 'combined', 'bias_count')} for split. It mixes the two channels' estimands and both transforms'
attenuation, so it is not a measure of estimator bias on its own.</p>
<p><b>mixQTL.</b> Its total slope recovers {B('mixqtl', 'total', 'bias_count')} (published) and
{B('mixqtl_permissive', 'total', 'bias_count')} (permissive) of the count-scale truth; at &ge;1000 reads and
|beta| 0.8 it is still {ci(bias(S, '0.8', 'mixqtl', 'total', 'bias_count', '>=1000'), 'mean', 2)} (published) and
{ci(bias(S, '0.8', 'mixqtl_permissive', 'total', 'bias_count', '>=1000'), 'mean', 2)} (permissive). mixQTL's response has
no +1 and no pseudocount, so this is not the low-depth transform. Section 3.8 explains most of it: mixQTL fits its
covariate offset from the covariates alone, before the variant enters, and then regresses on the genotype without
adjusting it for those covariates, which multiplies the slope by one minus the share of the genotype's variance they
explain; and it chooses those covariates on the outcome, which carries the effect. A one-step fit with all 17
covariates, which has neither feature, recovers {ladder_all17} of the truth in that section's units, and
that remainder is not decomposed. On a common
count-scale truth the total slope at |beta| 0.8 is {ci(bias(S, '0.8', 'split', 'total', 'bias_count'), 'mean', 2)} for
split and unit against {ci(bias(S, '0.8', 'mixqtl', 'total', 'bias_count'), 'mean', 2)} (published) and
{ci(bias(S, '0.8', 'mixqtl_permissive', 'total', 'bias_count'), 'mean', 2)} (permissive) for mixQTL, the comparison
Figure 2 draws. mixQTL's allelic slope recovers
{B('mixqtl', 'allelic', 'bias_count')} (published) and {B('mixqtl_permissive', 'allelic', 'bias_count')} (permissive).
At |beta| 0.4 the published cutoffs leave {mq['excluded_nonfinite']} of {mq['units'] + mq['excluded_nonfinite']}
allelic units without a finite estimate and the permissive cutoffs {mp['excluded_nonfinite']}; each count includes
the {pc('0.4')['excluded_nonfinite']} units with no allelic data in any arm (above).</p>"""


def interp_precision(S):
    Z = lambda a, ch: per_beta(lambda b: prec(S, f'beta{b}', a, ch, 'nonnull', 'sd_z')['value'], 2)
    E = lambda a, ch: per_beta(lambda b: prec(S, f'beta{b}', a, ch, 'nonnull', rkey(a))['value'], 2)
    Ec = lambda a, ch: ' / '.join(ci(prec(S, f'beta{b}', a, ch, 'nonnull', rkey(a)), 'value', 2) for b in BETAS)
    Zn = lambda a, ch: ci(prec(S, 'beta0.0', a, ch, 'null', 'sd_z'), 'value', 2)
    En = lambda a, ch: ci(prec(S, 'beta0.0', a, ch, 'null', 'ratio_vs_unit'), 'value', 2)
    zx = lambda a: prec(S, 'beta0.0', a, 'allelic', 'null', 'sd_z', NO_ONE_DF)
    Zx = lambda a: ci(zx(a), 'value', 3)
    side = lambda a: 'above 1' if zx(a)['lo'] > 1 else 'below 1' if zx(a)['hi'] < 1 else 'including 1'
    lo = lambda a, ch, key: ' / '.join(f(prec(S, f'beta{b}', a, ch, 'nonnull', key)['lo']) for b in BETAS)
    hi = lambda a, ch: ' / '.join(f(prec(S, f'beta{b}', a, ch, 'nonnull', 'ratio_vs_unit')['hi']) for b in BETAS)
    bands = lambda a, ch: ' / '.join(f(prec(S, 'beta0.4', a, ch, 'nonnull', 'ratio_vs_unit', bn)['value'], 2)
                                     for bn in BANDS[1:])
    bands_ci = lambda sc, part, a, ch: ' / '.join(ci(prec(S, sc, a, ch, part, 'ratio_vs_unit', bn), 'value', 2)
                                                 for bn in BANDS[1:])
    zlow = lambda a: per_beta(lambda b: prec(S, f'beta{b}', a, 'allelic', 'nonnull', 'sd_z', '<100')['value'], 2)
    zup = [prec(S, f'beta{b}', 'unit', 'allelic', 'nonnull', 'sd_z', bn)['value'] for b in BETAS for bn in BANDS[2:]]
    mix_anchor_low = f(prec(S, 'beta0.0', 'mixqtl', 'combined', 'null', 'ratio_vs_unit', '<100')['value'], 2)
    mix_anchor_up = ' / '.join(f(prec(S, 'beta0.0', 'mixqtl', 'combined', 'null', 'ratio_vs_unit', bn)['value'], 2)
                               for bn in BANDS[2:])
    Bm = lambda a: per_beta(lambda b: bias(S, b, a, 'combined', 'bias_count')['mean'], 2)
    return f"""
<p><b>Stated standard error.</b> The clean comparison is the total channel, where the hapmixQTL arms share the truth
and the donors. At the causal variant sd(z) is {Z('gibbs', 'total')} for gibbs against {Z('split', 'total')} for
split and unit and {Z('plus_one', 'total')} for plus_one; gibbs's intervals start at or above 1 (lower bounds
{lo('gibbs', 'total', 'sd_z')}), the others' include 1. On the anchor's null genes gibbs reads {Zn('gibbs', 'total')}
and unit {Zn('unit', 'total')}. So the Gibbs-weighted total channel's slope varies more than its stated se says, by
these factors, on genes with an effect as on genes without one, while unit weights state it correctly.</p>
<p>In the allelic channel the point value of sd(z) at the causal variant is above 1 for the four hapmixQTL arms
(gibbs and split {Z('gibbs', 'allelic')}, unit {Z('unit', 'allelic')}, plus_one {Z('plus_one', 'allelic')}), but every
interval includes 1 (lower bounds gibbs and split {lo('gibbs', 'allelic', 'sd_z')}, unit
{lo('unit', 'allelic', 'sd_z')}, plus_one {lo('plus_one', 'allelic', 'sd_z')}). The point excess sits below 100 reads
(unit {zlow('unit')} there, against {f(min(zup), 2)} to {f(max(zup), 2)} in the two higher bands), which is also where
the allelic bias is largest (section 3.3). At the causal variant sd(z) absorbs bias whose sign follows the random
sign of beta, so this excess may be bias rather than a stated standard error that is too small; the two were not
separated. mixQTL's allelic channel reads {Z('mixqtl_permissive', 'allelic')} with the permissive cutoffs and
{Z('mixqtl', 'allelic')} with the published ones. On the anchor's null genes, among the hapmixQTL arms only unit's
interval excludes 1 ({Zn('unit', 'allelic')}); gibbs and split read {Zn('gibbs', 'allelic')} and plus_one
{Zn('plus_one', 'allelic')}; mixQTL reads {Zn('mixqtl', 'allelic')} (published) and
{Zn('mixqtl_permissive', 'allelic')} (permissive). Much of the hapmixQTL arms' anchor excess is one null gene,
{one_df_gene(S)}, whose allelic fit on two donors has one residual degree of freedom but is referred to t with 73
(section 3.7); mixQTL fits no allelic channel on two donors. Without that gene the hapmixQTL arms read {Zx('gibbs')}
(gibbs and split, interval {side('gibbs')}), {Zx('unit')} (unit, {side('unit')}) and {Zx('plus_one')} (plus_one,
{side('plus_one')}). Two cautions. sd(z) responds to a few extreme values while a
rejection rate at 0.05 does not: unit's allelic null rate on the anchor is
{ci(S['null']['beta0.0']['unit']['allelic']['all']['0.05'], 'rate', 4)}. And the bias contribution at the causal
variant is consistent with mixQTL's total sd(z) rising with |beta| ({Z('mixqtl', 'total')} published) alongside its
attenuated slope; that was not tested either.</p>
<p><b>Efficiency.</b> In the allelic channel the gibbs and split squared error is {E('gibbs', 'allelic')} of unit
weights' at the causal variant and {En('gibbs', 'allelic')} on the anchor's null genes; by read band
(&lt;100 / 100-999 / &ge;1000, |beta| 0.4) it is {bands('gibbs', 'allelic')}, the gain growing with depth. plus_one keeps
less of that gain. On the anchor's null genes its allelic ratio is {En('plus_one', 'allelic')} against gibbs and
split's {En('gibbs', 'allelic')}, with separated intervals; at the causal variant ({Ec('plus_one', 'allelic')} against
{Ec('gibbs', 'allelic')}) the intervals overlap.</p>
<p>In the total channel gibbs's squared error is {E('gibbs', 'total')} of unit weights' at the causal variant and
{En('gibbs', 'total')} on the anchor's null genes. That cost comes from genes below 1,000 reads. By band
(&lt;100 / 100-999 / &ge;1000) it is {bands_ci('beta0.4', 'nonnull', 'gibbs', 'total')} at the causal variant at
|beta| 0.4, and {bands_ci('beta0.0', 'null', 'gibbs', 'total')} on the anchor's null genes; at 1,000 reads or more
these data show neither a cost nor a gain. plus_one is unit weighting in practice in this channel
({E('plus_one', 'total')}).</p>
<p>For the combined slope, both split ({E('split', 'combined')}) and plus_one ({E('plus_one', 'combined')}) are below
1 in point estimate at every |beta|. Only split's interval lies wholly below 1, and only at |beta| 0.2 and 0.4 (upper
bounds {hi('split', 'combined')}; plus_one's {hi('plus_one', 'combined')}). On the anchor's null genes both lie below 1
with separated intervals (split {En('split', 'combined')}, plus_one {En('plus_one', 'combined')}). At the causal
variant split's gain is therefore about 10% of unit weights' squared error and plus_one's about 2%. gibbs is above 1 at
the causal variant ({E('gibbs', 'combined')}, lower bounds {lo('gibbs', 'combined', 'ratio_vs_unit')}) and on the
anchor ({En('gibbs', 'combined')}).</p>
<p>For mixQTL the ratio is a comparison of methods as run. The published arm's allelic ratio is
{En('mixqtl', 'allelic')} on the anchor's null genes and {Ec('mixqtl', 'allelic')} at the causal variant; the reversal
is resolved only at |beta| 0.8. That is what run_arms.py's caveat predicts (its allelic cap of 1,000 reads makes
admission depend on the injected effect), but it was not tested separately. The published arm's combined ratio is
{E('mixqtl', 'combined')} at the causal variant, highest at |beta| 0.8. It is already {En('mixqtl', 'combined')} on the
anchor's null genes, where there is no effect to attenuate, so most of it is not attenuation. On the anchor it sits
in the genes below 100 reads ({mix_anchor_low} there, against {mix_anchor_up} in the two higher bands). That points
to which donors the published count cutoffs admit, but it was not tested separately. The permissive arm's ratio is
{En('mixqtl_permissive', 'combined')} on the anchor and {E('mixqtl_permissive', 'combined')} at the causal variant,
growing with |beta|, as expected for a combined slope that recovers {Bm('mixqtl_permissive')} of beta: its squared
error includes the missing share of the effect, which grows with |beta|. Section 3.8 decomposes the total channel's
part of this gap.</p>"""


def interp_lead(S):
    R = lambda a: per_beta(lambda b: S['lead'][f'beta{b}'][a]['all']['r2_high'], 2)
    C = lambda a: per_beta(lambda b: S['lead'][f'beta{b}'][a]['all']['lead_is_causal'], 2)
    hm = lambda i: [S['lead'][f'beta{BETAS[i]}'][a]['all']['r2_high'] for a in HAPMIX]
    rng = lambda i: f'{f(min(hm(i)), 2)}-{f(max(hm(i)), 2)}'
    und = {S['lead'][f'beta{b}'][a]['r2_undefined'] for b in BETAS for a in ARMS}
    if len(und) != 1:
        raise SystemExit(f'{SUMMARY}: r2_undefined differs between arms or |beta| ({und}); reword section 3.5')
    und = und.pop()
    return f"""
<p>The share of non-null gene units whose lead is within r<sup>2</sup> &ge; 0.8 of the causal variant is
{R('split')} for split, {R('gibbs')} for gibbs, {R('unit')} for unit and {R('plus_one')} for plus_one; mixQTL reaches
{R('mixqtl')} (published) and {R('mixqtl_permissive')} (permissive). The lead is the causal variant itself in
{C('split')} of units for split and {C('mixqtl')} for mixQTL published. The summary carries no interval for these
shares, so the differences among the four hapmixQTL arms are not interpreted. Among the hapmixQTL and mixQTL arms,
mixQTL with the published cutoffs has the lowest point value at every |beta| ({R('mixqtl')} against {rng(0)} / {rng(1)} / {rng(2)} for the hapmixQTL arms);
with no interval, that ordering is descriptive. It has no finite p at all in
{' / '.join(str(S['lead'][f'beta{b}']['mixqtl']['no_finite_p']) for b in BETAS)} of its non-null gene units. In every
arm and |beta|, {und} unit has a causal or lead variant with one dosage in every donor, so its r<sup>2</sup> is
undefined; it counts as not recovered in the shares. The median r<sup>2</sup> is over units where r<sup>2</sup> is
defined: {S['lead']['beta0.4']['gibbs']['all']['r2_defined']} for the hapmixQTL arms and
{' / '.join(str(S['lead'][f'beta{b}']['mixqtl']['all']['r2_defined']) for b in BETAS)} for mixQTL published, whose units
without a finite p drop out of its median and lift it.</p>"""


def interp_detection(S):
    D = lambda a, ch: per_beta(lambda b: S['detection'][f'beta{b}'][a][ch]['all']['0.001'], 2)
    st = lambda a: f(S['anchor'][a]['combined']['0.001']['stored'], 4)
    stw = lambda a: f(S['anchor'][a]['combined']['0.001']['stored_without_one_df'], 4)
    nod = S['recovery']['beta0.4']['gibbs']['allelic']['bias_pipeline']['all']['excluded_nonfinite']
    return f"""
<p>At p &lt; 1e-3 the combined statistic detects the causal variant in {D('gibbs', 'combined')} of non-null gene units
for gibbs, {D('split', 'combined')} for split, {D('unit', 'combined')} for unit and {D('plus_one', 'combined')} for
plus_one; mixQTL {D('mixqtl', 'combined')} (published) and {D('mixqtl_permissive', 'combined')} (permissive). The
allelic channel shows the weights most directly: gibbs and split {D('gibbs', 'allelic')} against unit
{D('unit', 'allelic')} and plus_one {D('plus_one', 'allelic')}. In the total channel detection is close across the
hapmixQTL arms (gibbs {D('gibbs', 'total')}, unit {D('unit', 'total')}), but gibbs's total channel rejects too often
on null genes (section 3.7), so its detections there are not comparable at face value. The same holds for gibbs's
combined statistic: at 1e-3 its stored 200-permutation null rate is {st('gibbs')}, against {st('split')} /
{st('unit')} / {st('plus_one')} for split / unit / plus_one, and without {one_df_gene(S)}, whose allelic p is too small
in every hapmixQTL arm (section 3.7), {stw('gibbs')} against {stw('split')} / {stw('unit')} / {stw('plus_one')}; so its
combined detection ({D('gibbs', 'combined')}) is not comparable at face value either. No null rate at 1e-5 was
measured. In the allelic channel
{nod} of the {S['detection']['beta0.4']['gibbs']['allelic']['all']['units']} units per |beta| have no allelic data
(section 3.3) and cannot be detected there in any arm; this is the same for every arm, so it does not change their
order.</p>"""


def interp_null(S, CG):
    R = lambda a, ch: per_scen(lambda sc: S['null'][sc][a][ch]['all']['0.05']['rate'])
    Rb = lambda a, ch, al: per_beta(lambda b: S['null'][f'beta{b}'][a][ch]['all'][al]['rate'], 4)
    stv = lambda a, ch, al: f(S['anchor'][a][ch][al]['stored'], 4)
    dmax = max(abs(S['null'][f'beta{b}']['gibbs'][ch]['all']['0.05']['rate'] - S['anchor']['gibbs'][ch]['0.05']['stored'])
               for b in BETAS for ch in ('allelic', 'total'))
    tail = [S['anchor'][a]['combined']['0.001']['stored'] for a in HAPMIX]
    tail_a = [S['anchor'][a]['allelic']['0.001']['stored'] for a in HAPMIX]
    pct = lambda ch: [S['anchor'][a][ch]['0.05']['percentile'] for a in HAPMIX]
    g1 = one_df_gene(S)
    n3 = lambda a, g: S['null']['beta0.0'][a]['combined'][g]['0.001']['rejections']
    g1_rej = lambda a: f'{n3(a, "all") - n3(a, NO_ONE_DF):,} of {a}\'s {n3(a, "all"):,}'
    stw = lambda a, ch: f(S['anchor'][a][ch]['0.001']['stored_without_one_df'], 4)
    out = [(a, ch, S['anchor'][a][ch]['0.05'], S['null']['beta0.0'][a][ch]['all']['0.05'])
           for a in HAPMIX for ch in CHANNELS if not S['anchor'][a][ch]['0.05']['passed']]
    outside = '; '.join(f'{a} {ch}, {f(r["rate"], 6)} ({n["rejections"]:,} of {n["tests"]:,} tests) at the '
                        f'{r["percentile"]:.1f}th percentile, against a central 99% of {f(r["perm_lo"], 6)} to '
                        f'{f(r["perm_hi"], 6)}' for a, ch, r, n in out) or 'none'
    return f"""
<p>At 0.05, over the anchor and |beta| = 0.2 / 0.4 / 0.8: gibbs combined {R('gibbs', 'combined')} and total
{R('gibbs', 'total')}, with gene-clustered intervals above 0.05 throughout; split combined {R('split', 'combined')};
unit combined {R('unit', 'combined')} and allelic {R('unit', 'allelic')}; plus_one combined {R('plus_one', 'combined')};
mixQTL combined {R('mixqtl', 'combined')} (published) and {R('mixqtl_permissive', 'combined')} (permissive). For the
four hapmixQTL arms the pattern is that of the stored null runs, whose 200-permutation rates are in the anchor table;
no stored null run exists for mixQTL here.</p>
<p><b>Rates at |beta| &gt; 0.</b> Thinning is expected to dilute the real data's coupling between weights and
residuals and so to pull these rates toward nominal. At 0.05 that is not seen for gibbs: its total-channel rates
({Rb('gibbs', 'total', '0.05')}) and allelic rates ({Rb('gibbs', 'allelic', '0.05')}) lie within {f(dmax, 4)} of the
stored means ({stv('gibbs', 'total', '0.05')} and {stv('gibbs', 'allelic', '0.05')}), inside their intervals. At
0.001 the allelic point rates are lower than stored (gibbs and split {Rb('gibbs', 'allelic', '0.001')} against
{stv('gibbs', 'allelic', '0.001')}; unit {Rb('unit', 'allelic', '0.001')} against {stv('unit', 'allelic', '0.001')}, so
not only for 1/v weights), but their intervals include the stored values (gibbs at |beta| 0.4,
{ci(S['null']['beta0.4']['gibbs']['allelic']['all']['0.001'], 'rate', 4)}), and gibbs's total channel is not lower
({Rb('gibbs', 'total', '0.001')} against {stv('gibbs', 'total', '0.001')}). At 0.001 the allelic rates also depend on
whether the gene discussed in the next paragraph is among a dataset's null genes. The dilution is therefore not
resolved here, and the |beta| &gt; 0 rates are not used as calibration results.</p>
<p><b>The tail.</b> At 0.001 the stored 200-permutation combined rates are {stv('gibbs', 'combined', '0.001')} for gibbs
and {stv('split', 'combined', '0.001')} / {stv('unit', 'combined', '0.001')} / {stv('plus_one', 'combined', '0.001')} for
split / unit / plus_one, {f(min(tail) / 0.001, 1)} to {f(max(tail) / 0.001, 1)} times nominal; the allelic rates are
{f(min(tail_a), 4)} to {f(max(tail_a), 4)}. Most of the excess of split, unit and plus_one comes from one null gene,
{g1}. Its allelic channel has two donors, so its through-origin allelic fit has one residual degree of freedom, but
hapmixQTL refers every p to t with N &minus; 2 &minus; 17 = 73 degrees of freedom (hapmixqtl.py:1873), so its allelic p
is computed as if it had 73. On the anchor this gene holds {g1_rej('split')} combined rejections at 0.001,
{g1_rej('unit')}, {g1_rej('plus_one')} and {g1_rej('gibbs')}: nearly the same count
in every arm, but a smaller share of gibbs's, whose total channel adds rejections
of its own. Without it the stored combined rates at 0.001 are {stw('gibbs', 'combined')} for gibbs and
{stw('split', 'combined')} / {stw('unit', 'combined')} / {stw('plus_one', 'combined')} for split / unit / plus_one, and
the stored allelic rates {stw('gibbs', 'allelic')} (gibbs and split), {stw('unit', 'allelic')} (unit) and
{stw('plus_one', 'allelic')} (plus_one); on the anchor the combined rates without it are
{' / '.join(ci(S['null']['beta0.0'][a]['combined'][NO_ONE_DF]['0.001'], 'rate', 5) for a in HAPMIX)} for gibbs / split /
unit / plus_one. So the tail excess of split, unit and plus_one is mostly this one gene's reference distribution, not
their weighting; it enters the four arms alike, so comparisons among the weightings stand. gibbs's combined rate stays
at {f(S['anchor']['gibbs']['combined']['0.001']['stored_without_one_df'] / 0.001, 1)} times nominal, with its total
channel at {stv('gibbs', 'total', '0.001')} stored (against {stv('unit', 'total', '0.001')} for the unit-weighted total
channel and {stv('plus_one', 'total', '0.001')} for plus_one's). The mixQTL arms, RASQUAL and TReCASE leave this gene's
allelic channel out by their own rules (mixQTL fits an allelic channel only with more than two donors; RASQUAL did not
admit its pseudo feature SNP, section 2; asSeq requires five allele-specific reads per record and five heterozygous
donors), while hapmixQTL fits it on two donors, so it is the only method exposed. A tail comparison across methods therefore
has to leave the gene out: on the anchor at 0.001 split then reads
{ci(S['null']['beta0.0']['split']['combined'][NO_ONE_DF]['0.001'], 'rate', 5)}, RASQUAL
{ci(S['null']['beta0.0']['rasqual']['combined'][NO_ONE_DF]['0.001'], 'rate', 5)} and TReCASE
{ci(S['null']['beta0.0']['trecase']['combined'][NO_ONE_DF]['0.001'], 'rate', 5)}. The ranking measures of section 3.1
were not recomputed without the gene.</p>
<p><b>The anchor.</b> Its one permutation sits at the {min(pct('total')):.1f}-{max(pct('total')):.1f}th percentile of
the stored per-permutation total-channel rates in the four arms and at the
{min(pct('allelic')):.1f}-{max(pct('allelic')):.1f}th of the allelic ones: a permutation with a low total-channel and a
high allelic rate. Outside the stored central 99%: {outside}. This says where one permutation fell. The plumbing was
checked once, for the gibbs and unit arms, by the one-off reproduction described in section 2; that check is
committed in check_generator.py {"and is in" if "reproduction" in CG else "but has not been run into"} the check file used
here.</p>"""


def overlap(S, a, b):
    """The |beta| at which the ranges of the per-dataset AUCs of arms a and b overlap."""
    return [x for x in BETAS if auc(S, x, a)['lo'] <= auc(S, x, b)['hi'] and auc(S, x, b)['lo'] <= auc(S, x, a)['hi']]


def at_betas(bs):
    return 'no |beta|' if not bs else 'every |beta|' if len(bs) == len(BETAS) else '|beta| ' + ' and '.join(bs)


def interp_joint(S, part, JF):
    """The RASQUAL and TReCASE paragraph of one results section."""
    A = lambda a: per_beta(lambda b: auc(S, b, a)['mean'])
    P = lambda a: per_beta(lambda b: fdp(S, b, a)['all']['power'])
    thr = lambda b, a: f'{fdp(S, b, a)["p_threshold"]:.1e}'
    B = lambda a, bn='all': per_beta(lambda b: bias(S, b, a, 'combined', 'bias_count', bn)['mean'], 2)
    Z = lambda a: per_beta(lambda b: prec(S, f'beta{b}', a, 'combined', 'nonnull', 'sd_z')['value'], 2)
    Zn = lambda a: ci(prec(S, 'beta0.0', a, 'combined', 'null', 'sd_z'), 'value', 3)
    Ex = lambda a: per_beta(lambda b: prec(S, f'beta{b}', a, 'combined', 'nonnull', 'ratio_vs_unit_count')['value'], 2)
    En = lambda a: ci(prec(S, 'beta0.0', a, 'combined', 'null', 'ratio_vs_unit'), 'value', 2)
    R = lambda a: per_beta(lambda b: S['lead'][f'beta{b}'][a]['all']['r2_high'], 2)
    D = lambda a: per_beta(lambda b: S['detection'][f'beta{b}'][a]['combined']['all']['0.001'], 2)
    N = lambda a: per_scen(lambda sc: S['null'][sc][a]['combined']['all']['0.05']['rate'], 4)
    n0 = lambda a, al, g='all': S['null']['beta0.0'][a]['combined'][g][al]
    tr = [S['null'][sc]['trecase']['combined']['all']['0.05']['rate'] / 0.05 for sc in S['null']]
    tr_lo, tr_hi = f(min(tr), 2), f(max(tr), 2)
    zx = lambda a: S['precision']['beta0.0'][a]['combined']['null_excluded']['nonfinite_z']
    zn = lambda a: prec(S, 'beta0.0', a, 'combined', 'null', 'sd_z')['units']
    raise_pct = lambda a: f'{100 * ((zn(a) + zx(a) - 1) / (zn(a) - 1)) ** 0.5 - 100:.1f}%'
    hm = [bias(S, b, a, 'combined', 'bias_count')['mean'] for b in BETAS for a in HAPMIX]
    hm_lo, hm_hi = f(min(hm), 2), f(max(hm), 2)
    exc = lambda b, a: ci(prec(S, f'beta{b}', a, 'combined', 'nonnull', 'ratio_vs_unit_count'), 'value', 2)
    dA = lambda a, b0: per_beta(lambda b: auc(S, b, b0)['mean'] - auc(S, b, a)['mean'])
    dD = max(abs(S['detection'][f'beta{b}']['trecase']['combined']['all']['0.001']
                 - S['detection'][f'beta{b}']['split']['combined']['all']['0.001']) for b in BETAS)
    comp = {k: v['all']['0.05'] for k, v in S['trecase_components'].items()}
    above = [k for k, v in comp.items() if v['lo'] > 0.05]
    name = dict(trec='total-count (TReC)', joint='joint', ase='allele-specific (ASE)')
    fin = n0('trecase', '0.05')['rate']
    if len(above) != len(comp) or fin <= max(v['rate'] for v in comp.values()):
        raise SystemExit(f'{SUMMARY}: TReCASE components {comp} against final {fin}; reword section 3.7')
    text = dict(
        ranking=f"""
<p><b>The joint models.</b> RASQUAL's AUC is {A('rasqual')} and TReCASE's {A('trecase')} at |beta| = 0.2 / 0.4 /
0.8, against {A('split')} for split and {A('unit')} for unit; their power at 5% realized FDP is {P('rasqual')} and
{P('trecase')}, against {P('split')} and {P('unit')}. Both are below split in point estimate on both measures at
every |beta|. TReCASE's AUC is below unit's by {dA('trecase', 'unit')} and RASQUAL's by {dA('rasqual', 'unit')}. The
ranges of per-dataset AUCs overlap for TReCASE and unit at {at_betas(overlap(S, 'trecase', 'unit'))}, for TReCASE and
split at {at_betas(overlap(S, 'trecase', 'split'))}, and for RASQUAL and split at
{at_betas(overlap(S, 'rasqual', 'split'))}; at |beta| 0.8 split's lowest dataset AUC ({f(auc(S, '0.8', 'split')['lo'])})
exceeds RASQUAL's highest ({f(auc(S, '0.8', 'rasqual')['hi'])}). With three datasets per effect size, the three
agreeing in direction is the most the data can show (a sign test on three datasets cannot fall below p = 0.25,
two-sided), and power at realized FDP has no interval. At |beta| 0.2 their pooled cuts fell at lead p of
{thr('0.2', 'rasqual')} (RASQUAL) and {thr('0.2', 'trecase')} (TReCASE), against {thr('0.2', 'split')} for split: null
genes' lead p reached below split's cut, as for gibbs. For TReCASE that agrees with its null-gene rates; RASQUAL's
anchor rate is {f(n0('rasqual', '0.05')['rate'], 4)} at 0.05 but {f(n0('rasqual', '0.001')['rate'], 4)} at 0.001
(section 3.7).</p>""",
        bias=f"""
<p><b>The joint models.</b> Their one slope has estimand beta (section 2), so unlike the hapmixQTL combined slope its
ratio to beta measures estimator bias. RASQUAL recovers {B('rasqual')} of beta and TReCASE {B('trecase')} at
|beta| = 0.2 / 0.4 / 0.8; by read band at |beta| 0.8 (&lt;100 / 100-999 / &ge;1000) that is
{' / '.join(f(bias(S, '0.8', 'rasqual', 'combined', 'bias_count', bn)['mean'], 2) for bn in BANDS[1:])} and
{' / '.join(f(bias(S, '0.8', 'trecase', 'combined', 'bias_count', bn)['mean'], 2) for bn in BANDS[1:])}. At |beta| 0.8
RASQUAL's interval is {ci(bias(S, '0.8', 'rasqual', 'combined', 'bias_count'), 'mean')} and TReCASE's
{ci(bias(S, '0.8', 'trecase', 'combined', 'bias_count'), 'mean')}. TReCASE's intervals include 1 at every |beta|,
the only combined slope here of which that is true (the hapmixQTL combined slopes, which mix two estimands and the
transforms' attenuation, read {hm_lo} to {hm_hi} against beta); RASQUAL's exclude 1 at every |beta|, the shortfall largest
below 100 reads. What causes RASQUAL's shortfall is not decomposed here. Two candidates are named from its source,
neither tested: (i) RASQUAL fits the covariates once, in a negative-binomial model under the null without the
genotype, and passes the fitted covariate effect as a fixed per-sample offset into every variant's fit
(rasqual_src/src/main.c:631 and :641-643; nbem.c:297 and :334). That is the same two-step structure whose
attenuation section 3.8 measures exactly for mixQTL, here inside a count likelihood and mixed with a covariate-free
allelic part. (ii) RASQUAL fits a reference-mapping bias phi below 0.5 at the pseudo feature SNP, where no mapping
bias can exist; a phi below 0.5 absorbs part of the allelic imbalance.</p>""",
        precision=f"""
<p><b>The joint models.</b> For RASQUAL and TReCASE the standard error is derived by the Wald inversion of
&chi;<sup>2</sup> (section 2). On null genes z = &plusmn;&radic;&chi;<sup>2</sup>, so sd(z)<sup>2</sup> is, up to the
mean of z (near 0), the mean of &chi;<sup>2</sup>: 1 when the statistic has the mean of a &chi;<sup>2</sup> with one
degree of freedom. It checks the statistic's mean, not its tail, which section 3.7 gives. On the anchor's null genes it
is {Zn('rasqual')} for RASQUAL and {Zn('trecase')} for TReCASE. So on the anchor TReCASE's statistic is larger on average
than a &chi;<sup>2</sup> with one degree of freedom, which agrees with its null-gene rate (section 3.7), while RASQUAL's
interval includes 1. Tests whose derived z is undefined are left out ({zx('rasqual'):,} and {zx('trecase'):,} on the
anchor): &chi;<sup>2</sup> &le; 0 ({JF['rasqual']['chisq_le0_anchor']:,} of RASQUAL's; for TReCASE, &chi;<sup>2</sup>
printed as 0.000 or missing) and, for RASQUAL only, &pi; printed as exactly 0.5, where slope and derived se are both 0.
These would enter as z = 0, so leaving them out raises sd(z) by about {raise_pct('rasqual')} and
{raise_pct('trecase')}. At the causal variant sd(z) is {Z('rasqual')} and {Z('trecase')}; there
z = (slope &minus; beta) / se describes how well the derived se matches the slope's spread around beta and absorbs
bias, so it is not a calibration of the test.</p>
<p>On squared error, with the arm and unit weights both held to the count-scale truth, RASQUAL's slope has
{Ex('rasqual')} of unit weights' at |beta| = 0.2 / 0.4 / 0.8 and TReCASE's {Ex('trecase')}, against {Ex('split')} for
split; on the anchor's null genes, where the truth is 0 for every arm, the ratios are {En('rasqual')}, {En('trecase')}
and {En('split')}. Both joint slopes have more squared error than unit weights, with intervals above 1, at every |beta|
and on the anchor, with one exception: TReCASE at |beta| 0.8 ({exc('0.8', 'trecase')}), whose point estimate is also
below split's ({exc('0.8', 'split')}), with overlapping intervals. That is a comparison of squared error, not of
precision: on the count-scale truth unit weights' squared error contains their own bias (the next paragraph), which
grows with beta<sup>2</sup>, while on the anchor's null genes, which carry no bias, TReCASE's slope has {En('trecase')}
of unit weights' squared error. How much of the 0.8 value that bias accounts for was not separated.</p>""",
        lead=f"""
<p>RASQUAL's share of leads within r<sup>2</sup> &ge; 0.8 of the causal variant is {R('rasqual')} and TReCASE's
{R('trecase')}, against {R('split')} for split and {R('unit')} for unit, with no interval. TReCASE's shares are within
0.02 of unit's; RASQUAL's are the lowest of the eight arms at |beta| 0.2 and 0.4.</p>""",
        detection=f"""
<p>RASQUAL detects the causal variant at p &lt; 1e-3 in {D('rasqual')} of non-null gene units and TReCASE in
{D('trecase')}, against {D('split')} for split and {D('unit')} for unit; a causal unit without a row is left out of a
joint arm's share (section 2). TReCASE's shares are within {f(dD, 2)} of split's. On the anchor, though, TReCASE's test
rejects at 1e-3 in {ci(n0('trecase', '0.001'), 'rate', 4)} of null-gene tests and RASQUAL's in
{ci(n0('rasqual', '0.001'), 'rate', 4)} (section 3.7), so neither is comparable at face value with the hapmixQTL arms
above; RASQUAL's shares are below unit weights' at every |beta|.</p>""",
        null=f"""
<p><b>The joint models.</b> Their likelihood-ratio p, referred to &chi;<sup>2</sup> with one degree of freedom,
rejects at 0.05 on null genes in {N('rasqual')} of tests for RASQUAL (over its converged rows) and {N('trecase')} for
TReCASE (anchor, then |beta| = 0.2 / 0.4 / 0.8). On the anchor the gene-clustered intervals are
{ci(n0('rasqual', '0.05'), 'rate', 4)} and {ci(n0('trecase', '0.05'), 'rate', 4)} at 0.05,
{ci(n0('rasqual', '0.01'), 'rate', 4)} and {ci(n0('trecase', '0.01'), 'rate', 4)} at 0.01, and
{ci(n0('rasqual', '0.001'), 'rate', 5)} and {ci(n0('trecase', '0.001'), 'rate', 5)} at 0.001, that is
{f(n0('rasqual', '0.001')['rate'] / 0.001, 1)} and {f(n0('trecase', '0.001')['rate'] / 0.001, 1)} times nominal there.
TReCASE's intervals at 0.05 lie above 0.05 in every scenario, at {tr_lo} to {tr_hi} times nominal; RASQUAL's lies just
above 0.05 on the anchor and includes it at |beta| &gt; 0. Neither model fits {one_df_gene(S)}'s allelic channel
(above), and without that gene their anchor rates at 0.001 are {f(n0('rasqual', '0.001', NO_ONE_DF)['rate'], 5)}
and {f(n0('trecase', '0.001', NO_ONE_DF)['rate'], 5)}. No stored null run exists for them, so where the anchor's one
permutation falls among their permutations is not known; the three datasets at each |beta| &gt; 0 are three further
permutations, on half the genes and thinned.</p>
<p><b>Where TReCASE's excess comes from.</b> asSeq's final p is one of its component tests, chosen per test. Scored
alone on the anchor's null genes, over the tests where its p is finite, each component rejects at 0.05 in
{'; '.join(f'{name[k]} {ci(v, "rate", 4)} ({v["tests"]:,} tests)' for k, v in comp.items())}, against {f(fin, 4)} for
the final p. So each component is above nominal on its own. The final p rejects more often than any component does over
all its tests, so the choice between them, which falls back to the total-count test wherever the joint fit is missing
and wherever asSeq's cis/trans test rejects, adds to an excess the components already have; each component's rate on the
subset where asSeq uses it was not scored. One candidate for the components' own excess, untested: a likelihood-ratio
test that fits an intercept, 17 covariates and a dispersion to 92 donors can reject too often at this sample size, and
RASQUAL has the same design.</p>""")
    return text[part]


INTERP = dict(ranking=interp_ranking, gene_level=interp_gene_level, bias=interp_bias, precision=interp_precision,
              lead=interp_lead, detection=interp_detection, null=interp_null)


def sec_results(S, CG, figs, LD, JF):
    auc_bands = tab_bands(S, lambda S, b, a: S['ranking'][f'beta{b}'][a]['auc'], 'mean')
    pow_bands = tab_bands(S, lambda S, b, a: S['ranking'][f'beta{b}'][a]['fdp_matched'], 'power')
    lead_bands = tab_bands(S, lambda S, b, a: S['lead'][f'beta{b}'][a], 'r2_high')
    genes = {bn: S['precision']['beta0.0']['gibbs']['combined']['null']['sd_z'][bn]['genes'] for bn in BANDS}
    return f'''
<h2>3. Results</h2>
<p><b>How to read the intervals.</b> Unless stated, an interval is a <i>gene-clustered 95% interval</i>: the
genes are resampled with replacement {S["n_boot"]:,} times, each gene carrying all its units from the
scenario's datasets, the statistic is recomputed each time, and the 2.5% and 97.5% quantiles are reported. It
carries gene-to-gene spread, the main source of uncertainty when the same genes recur across datasets. The
read bands hold {genes["<100"]} / {genes["100-999"]} / {genes[">=1000"]} of the {genes["all"]} genes. All arms were
run on the same datasets, so their errors are correlated, and the summary carries no interval for the
difference between two arms: a gap between arms is read against each arm's own interval. Separated intervals
are then good evidence of a difference; overlapping ones do not show that two arms are equal.</p>

<h3>3.1 Gene ranking: AUC and power at a realized false-discovery proportion</h3>
<p>Within each dataset the 100 genes are ordered by the nominal p of their <i>lead variant</i> (the tested
variant with the smallest p; the combined statistic for hapmixQTL, the meta statistic for mixQTL, the one joint test
for RASQUAL and TReCASE). The
<b>AUC</b> (area under the receiver operating characteristic curve) is the probability that a randomly chosen
non-null gene ranks above a randomly chosen null gene: 0.5 is chance, 1 is perfect separation. It is computed
per dataset and averaged over the 3 datasets. Its interval is the range of the three per-dataset AUCs: with
three datasets, the 2.5% and 97.5% quantiles of the mean over datasets resampled with replacement are the smallest
and largest dataset values. It is not a 95% interval, and it carries no gene-to-gene variation, because the three
datasets hold the same 100 genes. <b>Power at 5% realized
false-discovery proportion</b>: the gene units of the 3 datasets are pooled and walked down the ranking; the
realized false-discovery proportion (FDP) at depth k is the share of the top k that are truly null, known here
from the truth; the walk stops at the deepest k with FDP &le; 0.05, and power is the share of non-null gene
units above that point. The summary gives it no interval. A within-dataset ranking needs no reference
distribution, so it does not assume any arm's p is calibrated; it is confounded by the number of tested
variants per gene (a null gene with many variants has a smaller lead p by chance), a confounding every arm
shares.</p>
{INTERP['ranking'](S)}
{interp_joint(S, 'ranking', JF)}
{tab_ranking(S)}
<p>By read band (&lt;100 / 100-999 / &ge;1000 reads), AUC and then power at 5% FDP:</p>
{auc_bands}{pow_bands}
{img(figs['ranking'], 'Figure 1. A: AUC of the within-dataset gene ranking by lead nominal p (bar: range of the '
     'three per-dataset AUCs, not a 95% interval). B: share of non-null gene units called at the deepest point of the pooled '
     'ranking where at most 5% of calls are null genes. C: share of non-null gene units discovered by '
     'Benjamini-Hochberg at 5% on map_cis pval_beta, hapmixQTL arms only (gene-clustered interval); RASQUAL and '
     'TReCASE, like mixQTL, have no gene-level p. Points are offset sideways within each |beta| so that intervals '
     'do not overlap.')}

<h3>3.2 Gene-level power from map_cis</h3>
<p>map_cis gives each gene a gene-level p, <b>pval_beta</b>: the lead variant's nominal p is compared with the
smallest p of each of 1,000 permutations of donor records (with L/R swaps) under the arm's own weights, and
the comparison is smoothed by fitting a Beta distribution to those permuted minima. Because each arm is
referred to its own permutation null, pval_beta absorbs whatever miscalibration of that arm's nominal p the
permutation reproduces. The <b>Benjamini-Hochberg</b> procedure at 5% then calls genes within each dataset:
the 100 gene-level p are sorted and the k smallest are called, k being the largest rank with
p<sub>(k)</sub> &le; 0.05 k / 100; with valid p values the expected share of null genes among the calls is at
most 5%. Power is the share of non-null gene units called. The last four columns count null gene units with
pval_beta below 0.05 (a gene-level false-positive rate before any multiple-testing correction).</p>
{INTERP['gene_level'](S)}
{tab_gene_level(S)}

<h3>3.3 Bias at the causal variant</h3>
<p>The <b>bias ratio</b> is the mean, over causal units, of the estimated slope divided by the true slope at
the causal variant: 1 means the injected effect is recovered on average, 0.9 means 90% of it. Two truths are
used. On the <i>count scale</i> the allelic truth is beta itself, and the total truth is the least-squares
slope, with intercept, of the exact log2 total fold on half the ALT dosage (g/2) over the dataset's 92 donors
(near beta but not equal to it, because genotype counts are asymmetric). On the <i>pipeline scale</i> (hapmixQTL
arms only) the truth is the
slope the pipeline's own transformed phenotypes would show with no noise, after the +0.5 pseudocount of
log2((L + 0.5)/(R + 0.5)) and the +1 of log2(CPM + 1), which compress a fold at low depth. It is an unweighted
slope, so a weighted arm's bias against it still contains how the weights re-target a shift that varies with
depth. mixQTL's response has neither pseudocount nor +1, so its count-scale truth is already its own scale.
The combined slope mixes the two channels' estimands and is reported against beta only. Units whose ratio is
not finite are excluded and counted. A unit whose allelic channel has no admitted heterozygous donor returns
slope 0 with an infinite standard error; its count-scale ratio 0 / beta is finite, so it enters the count-scale
mean as 0, while its pipeline-scale truth is undefined and it is excluded there. In the table the first line is
the count scale, the second (grey) the pipeline scale, each followed by (units / excluded). The figure draws every
arm against the count-scale truth; the table's grey line carries the pipeline-scale diagnostic for hapmixQTL.</p>
{INTERP['bias'](S, CG, LD)}
{interp_joint(S, 'bias', JF)}
{tab_bias(S)}
{img(figs['bias'], 'Figure 2. Bias ratio (mean slope / truth at the causal variant, gene-clustered interval) by '
     'arm and read band, for the allelic and total channels (combined slope not drawn), every arm against the '
     'count-scale truth. The hapmixQTL allelic points keep as zeros the units with no allelic data, which the mixQTL '
     'points drop (section 3.3). mixQTL channels: allelic = asc, total = trc. Colour shade = |beta|. The y axes '
     'differ between panels.')}

<h3>3.4 Precision: stated standard error and efficiency against unit weights</h3>
<p><b>Realized over stated standard error</b>, written sd(z): z = (slope &minus; truth) / stated se, and sd(z) is
its standard deviation over units. It is 1 when the stated se equals the realized spread of the slope, 1.2
when the realized spread is 20% larger than the se says (se too small, p too small), and below 1 when the se is
too large. This is the reciprocal of a stated-over-true ratio; it is reported as score.py computes it.
Non-null units use the causal variant and the pipeline-scale truth (mixQTL: count scale; RASQUAL and TReCASE: beta;
combined: the same inverse-variance combination of the two channel truths); in the allelic channel that is the
{S['precision']['beta0.4']['gibbs']['allelic']['nonnull']['sd_z']['all']['units']} of
{S['recovery']['beta0.4']['gibbs']['allelic']['bias_count']['all']['units']} causal units per |beta| that have
allelic data (section 3.3). Null units use every tested variant of the null genes, with truth 0. In the allelic
channel these include variants of null genes with no admitted heterozygous donor, whose output is p = 1 and
z = 0: they cannot reject and enter sd(z) as zeros, which lowers both the allelic null rate (section 3.7) and the
allelic null sd(z). The summary does not count them; the stored null runs use the same convention, so the anchor
comparison is like-for-like. The <b>mean squared error ratio against unit weights</b> (efficiency) is the sum of squared
errors under the arm divided by the same sum under unit weights, over the same units; below 1 means more
precise than unit weights. split and unit share the total channel's weights, so split's total-channel ratio is
1 by construction. Real data have no oracle variance, so this is efficiency relative to unit weights, never
against the best possible weights. For mixQTL the ratio is a comparison of methods as run: mixQTL admits a
different donor set (its count cutoffs; under the published allelic cap of 1,000 reads admission even depends
on the injected effect, because thinning pulls records down into the band), so its ratio mixes weighting with
admission. For mixQTL, RASQUAL and TReCASE the ratio at the causal variant holds the arm and unit weights both to the
count-scale truth, so no method is ranked on the pipeline scale.</p>
{INTERP['precision'](S)}
{interp_joint(S, 'precision', JF)}
<p>sd(z), realized over stated standard error:</p>
{tab_precision(S, 'sd_z')}
<p>Mean squared error ratio against unit weights. At the causal variant the hapmixQTL rows hold the arm and unit
weights to their shared pipeline-scale truth (a comparison among weightings of one method); every other row holds
both to the count-scale truth. The null-gene column has truth 0 for every arm.</p>
{tab_precision(S, 'ratio_vs_unit')}
<p>The cross-method comparison: combined channel, every arm and unit weights both against the count-scale truth at
the causal variant (for the hapmixQTL and mixQTL arms and for unit weights, the inverse-variance combination of beta
and the per-gene total truth; for RASQUAL and TReCASE, beta).</p>
{tab_cross(S)}
{cross_note(S)}
{img(figs['efficiency'], 'Figure 3. Mean squared error ratio against unit weights (log scale; below 1 = more '
     'precise than unit weights), gene-clustered interval. Top: causal variant of non-null genes. Bottom: every '
     'tested variant of null genes ("anchor" is the beta = 0 dataset). Allelic and total panels: hapmixQTL arms '
     '(top, pipeline-scale truth for arm and unit alike) and mixQTL (bottom only). Combined panels: every arm, '
     'RASQUAL and TReCASE included, and at the causal variant the count-scale truth for arm and unit alike, so the '
     'top combined panel is the cross-method comparison and its hapmixQTL points differ from the pipeline-scale '
     'values in the text. unit is 1 by definition and not drawn; the total channel of split is 1 by construction. '
     'The y range of each panel covers every plotted interval.')}

<h3>3.5 Lead-variant recovery</h3>
<p>For each non-null gene unit the lead variant is compared with the causal one. <b>LD r<sup>2</sup></b> is the
squared Pearson correlation (the ordinary correlation coefficient) of ALT allele dosages over the 92 donors between the two variants (1 when they are
the same variant). Reported: the share of units whose lead is the causal variant, the share with
r<sup>2</sup> &ge; 0.8, and the median r<sup>2</sup>. A unit without a finite p counts as not recovered. No
interval is given in the summary; each share is over {S['lead']['beta0.4']['gibbs']['all']['units']} gene units per |beta|.</p>
{INTERP['lead'](S)}
{interp_joint(S, 'lead', JF)}
{tab_lead(S)}
<p>Share with r<sup>2</sup> &ge; 0.8 by read band:</p>
{lead_bands}
{img(figs['lead'], 'Figure 4. Share of non-null gene units whose lead variant is in LD r^2 >= 0.8 with the '
     'causal variant, by arm, |beta| and read band. No interval.')}

<h3>3.6 Causal-variant detection</h3>
<p>The share of non-null gene units whose nominal p at the causal variant falls below 0.05, 1e-3 and 1e-5, per
channel. This is power at a fixed nominal threshold, so it rewards an arm whose p values are too small; read it
together with the null rates in 3.7.</p>
{INTERP['detection'](S)}
{interp_joint(S, 'detection', JF)}
{tab_detection(S)}

<h3>3.7 Null genes and the beta = 0 anchor</h3>
<p>The <b>null-gene rate</b> is the share of tested variants of null genes whose nominal p is below a threshold
(0.05 in the first table). In the allelic channel it includes the tests with no allelic data (p = 1; section 3.4),
which cannot reject. At |beta| &gt; 0 the null genes are thinned too, which adds binomial noise of the model's own
kind and is expected to dilute the real data's coupling between weights and residuals, so those rates are not
calibration results. The beta = 0 anchor is one dataset, that is ONE record permutation, with no thinning. For the
four hapmixQTL arms its rate is compared with the stored 100-gene, 200-permutation null runs of the same arms: the
percentile of this dataset's rate among the 200 stored per-permutation rates, and whether it lies inside their
central 99%. This is descriptive: one permutation cannot test the plumbing.</p>
{INTERP['null'](S, CG)}
{interp_joint(S, 'null', JF)}
<p>Null-gene rate at 0.05 (gene-clustered interval):</p>
{tab_null(S)}
<p>The anchor against the stored null runs (hapmixQTL arms only):</p>
{tab_anchor(S)}
{sec_ladder(S, LD)}'''


def sec_ladder(S, LD):
    T, CUT = LD['total_channel'], ('published', 'permissive')
    tc = lambda c, b, k: T[c][f'beta{b}'][k]['all']
    sel = lambda c, b: T[c][f'beta{b}']['selection']
    sq = lambda c, b, k: T[c][f'beta{b}']['sq_error_vs_unit'][k]['all']
    rung = lambda c, b: LD['ladder'][f'beta{b}'][f'unit_{c}_cutoffs']['common_set']['total']['all']
    own = lambda b: LD['ladder'][f'beta{b}']['mixqtl']['own_set']['total']['all']['value']
    share = per_beta(lambda b: 1 - sq('published', b, 'mixqtl_trc')['value'] / own(b), 2)
    ov = sum(tc(c, b, 'one_step_trc')['hi'] >= tc(c, b, 'one_step_all_trc')['lo']
             and tc(c, b, 'one_step_all_trc')['hi'] >= tc(c, b, 'one_step_trc')['lo'] for c in CUT for b in BETAS)
    lo17, hi17 = all17(LD)
    x = lambda c, b, k0, k1: sq(c, b, k1)['value'] / (sq(c, b, k0)['value'] if k0 else 1.0)
    rq = lambda c, b: rung(c, b)['value']
    ub = per_beta(lambda b: bias(S, b, 'unit', 'total', 'bias_count')['mean'], 2)
    PB = lambda c, k: per_beta(lambda b: tc(c, b, k)['mean'])
    Q = lambda c, k: per_beta(lambda b: sq(c, b, k)['value'], 2)
    R2 = lambda c: per_beta(lambda b: sel(c, b)['mean_r2'])
    N2 = lambda c: ' / '.join(ci(sel(c, f'{b}')['null_gene_r2'], 'mean') for b in BETAS)
    above = lambda c: ' / '.join('above' if sel(c, b)['mean_r2'] > sel(c, b)['null_gene_r2']['hi'] else 'inside'
                                 for b in BETAS)
    nsel = sorted({int(sel(c, b)['n_selected_min_median_max'][1]) for c in CUT for b in BETAS})
    E, cu, idn = LD['estimability']['published'], LD['common_units'], LD['identity']
    t1 = table(['cutoffs', '|beta|', 'mixQTL total slope', '1 &minus; R<sup>2</sup> (prediction for the ratio of '
                'mixQTL to one-step selected)', 'one-step, selected covariates', 'one-step, all 17 covariates'],
               [[c, b] + [ci(tc(c, b, k), 'mean') for k in ('mixqtl_trc', 'predicted_1_minus_r2', 'one_step_trc',
                                                           'one_step_all_trc')] for c in CUT for b in BETAS])
    t2 = table(['cutoffs', '|beta|', 'cutoff rung', 'one-step, all 17, mixQTL response', 'one-step, selected',
                'mixQTL two-step'],
               [[c, b, ci(rung(c, b), 'value', 2)] + [ci(sq(c, b, k), 'value', 2) for k in
                                                     ('one_step_all_trc', 'one_step_trc', 'mixqtl_trc')]
                for c in CUT for b in BETAS])
    return f'''
<h3>3.8 Why mixQTL trails unit weights</h3>
<p>Sections 3.3 and 3.4 found mixQTL's total slope attenuated at every depth and its squared error above that of unit
weights. A separate run (scripts/plasmode/mixqtl_ladder.py, output {LADDER}) went from the unit-weight arm to mixQTL one
change at a time, in the total channel, on the causal units where every step has a finite error: {per_beta(lambda b:
cu[f"beta{b}"]["total"], 0)} of {cu["beta0.4"]["non_null"]} non-null units at |beta| = 0.2 / 0.4 / 0.8 (the <i>common
set</i>). Every number in this section is read from that file.</p>
<p><b>mixQTL fits its total channel in two steps.</b> First the <i>covariate offset</i>: the natural-log total,
log(total reads / 2 / library size), is regressed on the 17 covariates without the genotype; the covariates whose t
statistic exceeds 2 in absolute value are kept (the <i>selected covariates</i>, median {"-".join(map(str, nsel))} of 17
per gene and dataset) and the regression is refitted on them. Second, the offset is subtracted from the response and
the result is regressed on x = (h1 + h2)/2, half the ALT dosage, without first removing from x the part that the
selected covariates explain (mixQTL's own R code: the offset in rlib_covariate.R:27-40, the genotype regression on
the residual in rlib_matrix_ls.R:26-46; ported as covariate_offset and trc_channel). The <b>Frisch-Waugh-Lovell identity</b> states that in a least-squares fit of y on an
intercept, x and covariates C, the slope on x equals the slope of y on x after x is replaced by its residual from a
regression on the intercept and C. It follows that when both steps use the same donors, mixQTL's slope is
(1 &minus; R<sup>2</sup>) times the <i>one-step</i> slope, the slope on x when y is regressed on the intercept, x and
the selected covariates together, where R<sup>2</sup> is the share of the variance of x over donors that the selected
covariates explain. The ladder checked this where the offset's donors (total reads above 0) and the total channel's
donors (total reads at the cutoff) are the same: the largest relative difference between mixQTL's slope and
(1 &minus; R<sup>2</sup>) times the one-step slope was {idn["max_rel_dev"]:.1e} over {idn["units"]} non-null units. The
attenuation is therefore arithmetic, not noise.</p>
<p>Mean slope over the count-scale total truth at the causal variant (gene-clustered interval), common set. The
one-step slope with all 17 covariates is fitted without any selection on the outcome.</p>
{t1}
<p>At the published cutoffs mixQTL's total slope recovers {PB("published", "mixqtl_trc")} of the truth at
|beta| = 0.2 / 0.4 / 0.8, the one-step slope with the selected covariates {PB("published", "one_step_trc")}, and the
one-step slope with all 17 covariates {PB("published", "one_step_all_trc")}; the permissive cutoffs give
{PB("permissive", "mixqtl_trc")}, {PB("permissive", "one_step_trc")} and {PB("permissive", "one_step_all_trc")}. The
attenuation has two parts: fitting the covariates before the genotype, which multiplies the one-step slope by
1 &minus; R<sup>2</sup> ({PB("published", "predicted_1_minus_r2")} published, {PB("permissive", "predicted_1_minus_r2")}
permissive), and choosing the covariates on the outcome, the gap between the one-step slopes with the selected and
with all covariates. Those two slopes' intervals are not paired, and they overlap in {ov} of the 6 rows of the table,
so these intervals resolve the selection part only where they separate; a paired interval was not computed. That the
gap comes from choosing covariates on the outcome, rather than from using fewer of them, follows from the design (the
RNA-tied covariates move with the record, independently of the genotypes) and was not tested against a random subset
of covariates of the same size. The one-step slope with all 17 covariates recovers {f(lo17, 2)} to {f(hi17, 2)} of
the truth; that remainder is not decomposed here.</p>
<p><b>The selection follows the injected effect.</b> The covariates are chosen on a response that carries the genotype
effect, so a covariate that happens to correlate with the genotype passes |t| &gt; 2 more often as the effect grows,
and R<sup>2</sup> grows with it. The mean R<sup>2</sup> of x on the selected covariates at the non-null genes' causal
variants is {R2("published")} at |beta| = 0.2 / 0.4 / 0.8 with the published cutoffs and {R2("permissive")} with the
permissive ones. The chance level is the same quantity at the null genes' causal variants, where selection sees no
genotype effect but the genotype principal components among the covariates can still correlate with x:
{N2("published")} (published) and {N2("permissive")} (permissive). The non-null mean lies {above("published")} the null
interval (published) and {above("permissive")} it (permissive) at the three effect sizes.</p>
<p><b>Squared error, one change at a time.</b> Each column is the summed squared error at the causal variant over that
of the unit-weight arm, on the common set (gene-clustered interval). The <i>cutoff rung</i> is the unit-weight arm
(unit weights, log2(CPM + 1), all 17 covariates in one fit) with only the donors mixQTL's count cutoffs admit. The next
step changes the response to mixQTL's natural-log total over 2 x library size and the truth to the count scale,
together; then the covariates become the selected ones; then the fit becomes mixQTL's two steps. As the ladder defines
it, the unit-weight arm's squared error in the denominator is against its pipeline-scale truth, so these ratios are not
the count-scale cross-method ratios of section 3.4. All columns share that one denominator, but the step to mixQTL's
response also moves the numerator from the pipeline-scale to the count-scale truth: it mixes a change of response
with a change of the truth it is measured against, so the ladder as built cannot assign it a cost. The other steps
each hold the truth fixed and compare with each other.</p>
{t2}
<p>Donor admission alone gives {per_beta(lambda b: rung("published", b)["value"], 2)} at the published cutoffs and
{per_beta(lambda b: rung("permissive", b)["value"], 2)} at the permissive ones, which admit nearly every donor. The step to
mixQTL's response, which also changes the truth, gives {Q("published", "one_step_all_trc")} published and
{Q("permissive", "one_step_all_trc")} permissive; for the reason just given it is not read as a cost. On bias the change
of response runs the other way: the one-step fit with all 17 covariates on mixQTL's response recovers {f(lo17, 2)} to
{f(hi17, 2)} of the count-scale total truth (first table), where unit weights' total slope on log2(CPM + 1) recovers
{ub} (section 3.3). Selecting the covariates on the outcome gives {Q("published", "one_step_trc")} and
{Q("permissive", "one_step_trc")}, and mixQTL's two-step fit {Q("published", "mixqtl_trc")} and
{Q("permissive", "mixqtl_trc")}. At |beta| 0.8 the two-step fit is the largest single step with the permissive
cutoffs: it takes the ratio from {ci(sq("permissive", "0.8", "one_step_trc"), "value", 2)} to
{ci(sq("permissive", "0.8", "mixqtl_trc"), "value", 2)} ({f(x("permissive", "0.8", "one_step_trc", "mixqtl_trc"), 2)}-fold),
with separated intervals. With the published ones it takes the ratio from
{ci(sq("published", "0.8", "one_step_trc"), "value", 2)} to {ci(sq("published", "0.8", "mixqtl_trc"), "value", 2)}
({f(x("published", "0.8", "one_step_trc", "mixqtl_trc"), 2)}-fold), with overlapping intervals: the largest increase
of any step, while donor admission multiplies the ratio by more ({f(rq("published", "0.8"), 2)}-fold). At |beta| 0.2 it is lower than the one-step fit under both settings, with
overlapping intervals; shrinking every slope by 1 &minus; R<sup>2</sup> also shrinks its variance, which can outweigh
the bias it adds when the effect is small. That trade was not tested separately.</p>
<p><b>What the common set leaves out.</b> At the published cutoffs the rung has no total-channel estimate in
{E["rung_empty"]} of {E["gene_datasets"]} gene-datasets, all with at most {E["rung_empty_max_donors"]:.0f} admitted
donors, where a joint fit of an intercept, 17 covariates and the genotype has no residual degree of freedom; mixQTL,
whose total fit has only an intercept and x, still has a causal-variant slope in {E["rung_empty_mixqtl_trc_finite"]} of
them, from as few as {E["rung_empty_mixqtl_trc_min_donors"]:.0f} donors. Paired with unit weights over its own units
rather than the common set, mixQTL's published total ratio is {per_beta(own, 2)}, against
{Q("published", "mixqtl_trc")} on the common set. The own set contains the common set, so unit weights' squared error
over it is at least as large, and the units outside the common set, mostly few-donor fits, carry at least {share} of
mixQTL's own-set total squared error at |beta| = 0.2 / 0.4 / 0.8. So mixQTL trails unit weights in the total channel
through its donor admission at the published cutoffs and the few-donor fits that admission leaves it, and, growing with
the effect, through the attenuation of its two-step covariate adjustment. The ladder cannot cost its change of response,
which on bias favours mixQTL.</p>'''


def sec_critique(S):
    rank = lambda a: per_beta(lambda b: fdp(S, b, a)['all']['power'])
    gl = lambda a: per_beta(lambda b: bh(S, b, a)['power_bh']['all']['rate'])
    E = lambda a, ch: per_beta(lambda b: prec(S, f'beta{b}', a, ch, 'nonnull', rkey(a))['value'], 2)
    En = lambda a, ch: ci(prec(S, 'beta0.0', a, ch, 'null', 'ratio_vs_unit'), 'value', 2)
    an = S['anchor']['gibbs']['total']['0.05']
    tg, ts = (fdp(S, '0.4', a)['p_threshold'] for a in ('gibbs', 'split'))
    thr = lambda b, a: f'{fdp(S, b, a)["p_threshold"]:.1e}'
    n3 = lambda a: S['null']['beta0.0'][a]['combined']['all']['0.001']['rate']
    cc = [prec(S, f'beta{b}', 'gibbs', 'combined', 'nonnull', 'ratio_vs_unit_count') for b in BETAS]
    Ec = lambda a: ' / '.join(ci(d, 'value', 2) for d in cc)
    inc = [b for b, d in zip(BETAS, cc) if d['lo'] <= 1 <= d['hi']]
    En_band = lambda a, ch: ' / '.join(f(prec(S, 'beta0.0', a, ch, 'null', 'ratio_vs_unit', bn)['value'], 2)
                                       for bn in BANDS[1:])
    return f"""
<h2>4. The strongest critique, and what it changed</h2>
<p><b>The ranking is not free of calibration.</b> A within-dataset ranking needs no reference distribution, but the
cut at 5% realized FDP is set by where the null genes land, and an arm whose null genes get too-small p pushes them up
its ranking. gibbs's total channel does this: its null-gene rate at 0.05 is {f(an['rate'], 4)} on the anchor and
{f(an['stored'], 4)} over the stored 200 permutations. In the ranking of section 3.1 gibbs made no call at all at
|beta| 0.2, and at |beta| 0.4 its cut fell at lead p {tg:.1e} against split's {ts:.1e}, {ts / tg:.0f}-fold smaller.
If those null p values are too small, part of gibbs's ranking deficit is a calibration effect and not a lack of
signal. The same objection applies to
the joint models: at |beta| 0.2 their cuts fell at lead p {thr('0.2', 'rasqual')} (RASQUAL) and
{thr('0.2', 'trecase')} (TReCASE) against split's {thr('0.2', 'split')}, and on the anchor their nominal p rejects at
0.001 in {f(n3('rasqual'), 4)} and {f(n3('trecase'), 4)} of null-gene tests (section 3.7), so part of their ranking
deficit may also be calibration. The hapmixQTL arms carry their own null outlier, {one_df_gene(S)} (section 3.7), in all
four arms alike; the ranking was not recomputed without it.</p>
<p><b>What addressing it changed.</b> Gene-level Benjamini-Hochberg on pval_beta changes three things at once: each arm
is referred to its own permutation null; pval_beta also corrects for the number of tested variants per gene (the
confounder named in section 3.1); and calls are made per dataset, not at a pooled realized-FDP cut. On it gibbs reads
{gl('gibbs')} against split's {gl('split')} at |beta| = 0.2 / 0.4 / 0.8, with overlapping intervals, so the ranking gap
(power at 5% realized FDP {rank('gibbs')} against {rank('split')}, no interval) is not resolved at gene level, and on
gene-level power the four hapmixQTL arms cannot be told apart at this size. Which of the three changes removes the
gap is not identified here; gibbs's anticonservative total channel ({f(an['stored'], 4)} stored at 0.05) is a
candidate, not a measured share. What survives the critique is the signal-side cost, which needs no reference
distribution: gibbs's combined slope has {En('gibbs', 'combined')} of unit weights' squared error on the anchor's null
genes, where the truth is 0 for every arm, and {E('gibbs', 'combined')} at the causal variant on the pipeline-scale
truth (every interval above 1); on the count-scale truth the causal-variant ratios are {Ec('gibbs')}, with intervals
that include 1 at {at_betas(inc)}, consistent with unit weights' count-scale error containing the attenuation of
log2(CPM + 1) (section 3.4). Its total-channel standard error is too small (section 3.4).</p>
<p><b>A second objection: the allelic gain could be made by the generator.</b> The allelic Gibbs variance of a thinned
record follows its thinned counts by the generator's own rule, so 1/v weights might track the true error on thinned
records by construction. The anchor answers this: nothing is thinned there and v is Salmon's own, and the gibbs and
split allelic squared error on the anchor's null genes is {En('gibbs', 'allelic')} of unit weights', against
{E('gibbs', 'allelic')} at the causal variants. The total channel's loss is present on the anchor too
({En('gibbs', 'total')}; by band &lt;100 / 100-999 / &ge;1000, {En_band('gibbs', 'total')}). Neither result comes from
the generator's rule.</p>
<p><b>What is not established.</b> At the causal variant the allelic sd(z) is above 1 in point estimate for all four
hapmixQTL arms, unit weights included, but every interval includes 1, and the point excess sits in genes below 100
reads, where the allelic bias is also largest (section 3.4). Whether the stated allelic standard error is too small at
low depth, or sd(z) there absorbs bias, is not established. Because unit weights show the same point excess, it does
not by itself bear on the choice among weightings.</p>"""


def sec_meaning(S, CG, JF, LD):
    pr = CG['recovery']['primary']
    genes = {bn: S['precision']['beta0.0']['gibbs']['combined']['null']['sd_z'][bn]['genes'] for bn in BANDS}
    st = lambda a, ch: f(S['anchor'][a][ch]['0.05']['stored'], 4)
    st3 = lambda a: f(S['anchor'][a]['combined']['0.001']['stored'], 4)
    stw = lambda a: f(S['anchor'][a]['combined']['0.001']['stored_without_one_df'], 4)
    tr = JF['trecase']
    hi = lambda a, ch: ' / '.join(f(prec(S, f'beta{b}', a, ch, 'nonnull', 'ratio_vs_unit')['hi']) for b in BETAS)
    tband = lambda sc, part: ' / '.join(ci(prec(S, sc, 'gibbs', 'total', part, 'ratio_vs_unit', bn), 'value', 2)
                                        for bn in BANDS[1:])
    B = lambda a, ch: per_beta(lambda b: bias(S, b, a, ch, 'bias_count')['mean'], 2)
    E = lambda a, ch: per_beta(lambda b: prec(S, f'beta{b}', a, ch, 'nonnull', rkey(a))['value'], 2)
    En = lambda a, ch: ci(prec(S, 'beta0.0', a, ch, 'null', 'ratio_vs_unit'), 'value', 2)
    A = lambda a: per_beta(lambda b: auc(S, b, a)['mean'])
    R = lambda a: per_beta(lambda b: S['lead'][f'beta{b}'][a]['all']['r2_high'], 2)
    N = lambda a: per_beta(lambda b: S['null'][f'beta{b}'][a]['combined']['all']['0.05']['rate'])
    Ex = lambda a: per_beta(lambda b: prec(S, f'beta{b}', a, 'combined', 'nonnull', 'ratio_vs_unit_count')['value'], 2)
    P = lambda a: per_beta(lambda b: fdp(S, b, a)['all']['power'])
    return f"""
<h2>5. What it means for the open decisions</h2>
<p><b>Which weighting ships.</b> Until now the decision rested on the stored 200-permutation null runs, whose
combined rates at 0.05 were {st('gibbs', 'combined')} for gibbs, {st('split', 'combined')} for split,
{st('unit', 'combined')} for unit and {st('plus_one', 'combined')} for plus_one, with gibbs's total channel at
{st('gibbs', 'total')} and the other three at {st('split', 'total')}. This benchmark adds the signal side. The
separating evidence is the anchor, on unthinned records: there split's combined squared error is
{En('split', 'combined')} of unit weights', with an interval separated from plus_one's {En('plus_one', 'combined')} and
gibbs's {En('gibbs', 'combined')}. At the causal variant it is {E('split', 'combined')} (upper bounds
{hi('split', 'combined')}). AUC, ranking power and gene-level power do not separate split, unit and plus_one
(sections 3.1 and 3.2). split's total slope is unbiased on the pipeline scale and its total-channel standard error
matches the slope's spread. Its cost is in the allelic channel: split's allelic slope falls short of the
pipeline-scale truth at |beta| 0.8 ({ci(bias(S, '0.8', 'split', 'allelic', 'bias_pipeline'), 'mean')}); check (c)
attributes about 5% to 1/v weights ({f(pr['inv_va_pipeline']['mean'])} against {f(pr['unit_pipeline']['mean'])} for
unit weights), while the benchmark itself does not separate 1/v from unit weights on bias (section 3.3). gibbs has the same
allelic fit, but below 1,000 reads its total-channel weights make the slope less precise than unit weights (by band
&lt;100 / 100-999 / &ge;1000 at |beta| 0.4, {tband('beta0.4', 'nonnull')}; anchor null genes
{tband('beta0.0', 'null')}; at 1,000 reads or more neither a cost nor a gain is shown), and its total channel
understates its standard error, so its combined slope is less precise than unit weights' (anchor null genes
{En('gibbs', 'combined')}; causal variant {E('gibbs', 'combined')} on the pipeline-scale truth, a gap the count-scale
truth narrows, section 3.4). plus_one is calibrated at 0.05 (stored combined {st('plus_one', 'combined')}). At 0.001
its stored combined rate, {st3('plus_one')}, like split's ({st3('split')}) and unit's ({st3('unit')}), is mostly one
gene, {one_df_gene(S)}, whose allelic p is referred to the wrong degrees of freedom (section 3.7); without it the three
read {stw('plus_one')}, {stw('split')} and {stw('unit')}, so the tail does not separate them, while gibbs reads
{stw('gibbs')}. plus_one's combined squared error is {E('plus_one', 'combined')} of unit weights', and it keeps less of
the allelic gain than split (anchor {En('plus_one', 'allelic')} against {En('split', 'allelic')}). unit weights give that gain up. Nothing here contradicts the null-based record; on
precision the evidence favours split weighting, at the allelic cost just stated. The choice, including leaving the
shipped default, remains a user decision, and section 6 lists what these data cannot settle (100 genes, of which
{genes['100-999'] + genes['>=1000']} have 100 or more median haplotype-informative reads; one permutation rule).</p>
<p><b>Where it narrows earlier results.</b> The 2026-09-19 finding that the Gibbs draws improve the point estimate
(brainvar_hapmix_deploy/mixqtl_replication_20260919/REPORT.md) was measured on the allelic channel of 29
high-coverage genes: the median permutation variance of the slope under 1/v weights was 0.340 of its unweighted
value. It points the same way here and is of similar size in the comparable stratum: at 1,000 or more reads the gibbs
and split allelic squared error is {f(prec(S, 'beta0.4', 'gibbs', 'allelic', 'nonnull', 'ratio_vs_unit', '>=1000')['value'], 2)}
of unit weights' at |beta| 0.4 and {f(prec(S, 'beta0.0', 'gibbs', 'allelic', 'null', 'ratio_vs_unit', '>=1000')['value'], 2)}
on the anchor's null genes. The statistics differ (there a ratio of median variances over 40 null permutations, here
a ratio of summed squared errors), so only the order of magnitude is compared. Pooled over all 100 genes the ratio
here is {E('gibbs', 'allelic')}. The total channel was not part of that record. Here its Gibbs weights cost precision
below 1,000 reads, on known effects as on the corrected pipeline's nulls (docs/pipeline_rules.md, "What made the total
channel worse"). An earlier count-scale measurement on the pre-correction pipeline
(brainvar_hapmix_deploy/count_scale_weights_20260925/) had found the total channel's Gibbs weights to buy nothing; on
the corrected pipeline they cost precision below 1,000 reads. Those records differ from this one in gene set, pipeline
and statistic, so only the direction is compared, not the magnitude.</p>
<p><b>mixQTL as the baseline.</b> As run with the published cutoffs, mixQTL has the lowest AUC of the hapmixQTL and
mixQTL arms at every |beta| ({A('mixqtl')}) and the lowest point share of leads within r<sup>2</sup> &ge; 0.8 of the causal variant
({R('mixqtl')}, no interval), leaves some gene units without any finite p, and attenuates its slopes in both channels.
With the permissive cutoffs it is closer to the hapmixQTL arms (AUC {A('mixqtl_permissive')}) but below split's point
estimates ({A('split')}) at every |beta|, with overlapping ranges. Its null-gene combined rates at 0.05 are {N('mixqtl')} (published) and
{N('mixqtl_permissive')} (permissive) at |beta| = 0.2 / 0.4 / 0.8, and
{f(S['null']['beta0.0']['mixqtl']['combined']['all']['0.05']['rate'])} and
{f(S['null']['beta0.0']['mixqtl_permissive']['combined']['all']['0.05']['rate'])} on the anchor. Because its permutation
scan was excluded, mixQTL has no gene-level power here, so what hapmixQTL has to beat is answered only on the
within-dataset ranking and on slope accuracy. Against both mixQTL settings split is ahead in point estimate at every
|beta| on AUC, on ranking power at 5% realized FDP, on the share of leads with r<sup>2</sup> &ge; 0.8, on total-channel
bias on a common count-scale truth ({B('split', 'total')} against {B('mixqtl', 'total')} published and
{B('mixqtl_permissive', 'total')} permissive), and on combined squared error against unit weights with both on the
count-scale truth ({Ex('split')} against {Ex('mixqtl')} and {Ex('mixqtl_permissive')}). At |beta| 0.8
split's lowest dataset AUC ({f(auc(S, '0.8', 'split')['lo'])}) exceeds mixQTL published's highest
({f(auc(S, '0.8', 'mixqtl')['hi'])}). The allelic slope does not separate: on the count scale at |beta| 0.8 mixQTL
permissive recovers {ci(bias(S, '0.8', 'mixqtl_permissive', 'allelic', 'bias_count'), 'mean')} of beta and split
{ci(bias(S, '0.8', 'split', 'allelic', 'bias_count'), 'mean')}, with overlapping intervals at every |beta|; split's
count-scale line keeps as zeros the units with no allelic data, which mixQTL's drops (section 3.3). Most of the
attenuation of mixQTL's total slope comes from its two-step covariate adjustment and its choice of covariates on the
outcome (section 3.8); a one-step fit with all 17 covariates still recovers {'{:.2f} to {:.2f}'.format(*all17(LD))} of
the truth, a remainder not decomposed. Its total slopes should not be used as a reference for effect size.</p>
<p><b>The joint models as comparators.</b> RASQUAL and TReCASE fit a total-count model whose dosage form is the
generator's expected total fold, averaged over donors (section 2). They receive exactly the allele-specific records the
hapmixQTL arms admit; asSeq's own floors then drop {tr['asseq_dropped'][0]} to {tr['asseq_dropped'][1]} of them per
dataset (fewer than five allele-specific reads) and skip the allele-specific model in {tr['few_het']:,} tests
({100 * tr['few_het'] / tr['tests']:.1f}%), which have fewer than five heterozygous donors. On these data neither
outranks split in point estimate: AUC {A('rasqual')} (RASQUAL) and {A('trecase')} (TReCASE) against {A('split')}; power
at 5% realized FDP {P('rasqual')} and {P('trecase')} against {P('split')}; leads within r<sup>2</sup> &ge; 0.8
{R('rasqual')} and {R('trecase')} against {R('split')}. TReCASE's range of per-dataset AUCs overlaps split's at
{at_betas(overlap(S, 'trecase', 'split'))}; power at realized FDP and lead recovery have no interval; each effect size
rests on 3 datasets. TReCASE's slope recovers beta within its intervals at every |beta|
({per_beta(lambda b: bias(S, b, 'trecase', 'combined', 'bias_count')['mean'], 2)}), which no other combined slope here
does, but its p rejects on null genes at {per_scen(lambda sc: S['null'][sc]['trecase']['combined']['all']['0.05']['rate'])}
at 0.05 (anchor, then |beta| = 0.2 / 0.4 / 0.8), and its slope has more squared error than unit weights' at |beta| 0.2
and 0.4 ({Ex('trecase')} at the three |beta|, on the count-scale truth) and on the anchor's null genes
({ci(prec(S, 'beta0.0', 'trecase', 'combined', 'null', 'ratio_vs_unit'), 'value', 2)}). RASQUAL's rate at 0.05 is nearer
nominal ({per_scen(lambda sc: S['null'][sc]['rasqual']['combined']['all']['0.05']['rate'])}), though on the anchor at 0.001
it is {f(S['null']['beta0.0']['rasqual']['combined']['all']['0.001']['rate'] / 0.001, 1)} times nominal; its slope falls
short of beta ({per_beta(lambda b: bias(S, b, 'rasqual', 'combined', 'bias_count')['mean'], 2)}) and has {Ex('rasqual')}
times unit weights' squared error. So on this benchmark split weighting is not outperformed in point estimate by either
joint model on ranking, on ranking power at 5% realized FDP or on lead placement. On squared error only TReCASE's point
estimate at |beta| 0.8 is lower, with overlapping intervals and a denominator that contains unit weights' own
count-scale bias (section 3.4). TReCASE's advantage is in bias, and it comes with an anticonservative nominal p. Both joint models were run on
Salmon point estimates at a pseudo feature SNP rather than on reads (section 6), so this measures them as run here, not
joint likelihood modelling as such.</p>"""


def sec_limits(S, CG, JF):
    genes = {bn: S['precision']['beta0.0']['gibbs']['combined']['null']['sd_z'][bn]['genes'] for bn in BANDS}
    rq, tr = JF['rasqual'], JF['trecase']
    pct = lambda x: f'{100 * x:.1f}%'
    if tr['df_not_1'] + tr['trec_na'] != tr['final_na']:
        raise SystemExit(f'{TRECASE_SUMMARY}: final p missing in {tr["final_na"]} tests, not the {tr["df_not_1"]} with '
                         f'df != 1 plus {tr["trec_na"]} failed TReC fits; reword section 6')
    return f'''
<h2>6. Limits: what this analysis cannot establish</h2>
<p><b>No oracle variance.</b> Real data carry no true error variance per record, so the efficiency results are
relative to unit weights. They say which weighting is more precise than another on these data, not how far any
of them is from the best possible weights.</p>
<p><b>The null is the record permutation.</b> Both the generator and the map_cis null move donor records against
genotypes with the genotype principal components tied to the genotypes. The data therefore cannot say whether
that permutation rule, or the alternative in which the principal components move with the record, matches the
sampling distribution of an observed statistic; the open decision on that rule is untouched.</p>
<p><b>What the injected effects are.</b> Effects are made by thinning only, so expression can only go down, and
there is one causal variant per gene. The gene set is the 100 genes of the corrected null store, not a sample of
the transcriptome: {genes["100-999"] + genes[">=1000"]} of them have 100 or more median haplotype-informative reads
({genes["<100"]} below 100, {genes["100-999"]} at 100-999, {genes[">=1000"]} at 1,000 or more). There are 3 datasets
per effect size with 50 non-null genes each, so an effect size rests on
{S['lead']['beta0.4']['gibbs']['all']['units']} non-null gene units spread over the 100 genes (a gene is non-null in
about 1.5 of the 3 datasets), and a read band on far fewer; one gene can move a band's value.</p>
<p><b>What thinning cannot reproduce.</b> The point estimate is Salmon's variational-Bayes optimum, the read count
Salmon's optimizer assigns to each transcript, not an observed count. At lower depth Salmon puts one haplotype at
exactly zero more often, and binomial thinning cannot create those zeros, so the zero-haplotype admission rule is
exercised less here than it would be on truly shallower libraries. Thinning also adds binomial noise of the model's
own kind, which is expected to dilute the real data's coupling between weights and residuals on thinned records and
to pull the null-gene rates at |beta| &gt; 0 toward nominal. Section 3.7 shows that this is not resolved here: the
allelic point rates at 0.001 are lower than stored but inside their intervals, and gibbs's total-channel rates are
not lower. Either way those rates are not calibration results. The precision of the 1/v arms on thinned records may
be more favourable than on real records at the same depth; this was not measured. The pipeline's transforms (the
+0.5 of the allelic ratio, the +1 of log2(CPM + 1)) attenuate a fold at low depth, so bias against beta mixes that
attenuation with estimator bias. The pipeline-scale truth separates the two only for an unweighted fit.</p>
<p><b>The allelic variance rule.</b> It overstates Va' by at most (1 - f) x 1.4% at 100-999 total reads and
(1 - f) x 9% at 10-99, because Salmon's Gibbs prior of one pseudo-read per transcript copy does not scale with depth
(make_datasets.py, ERROR BOUND). No known-answer test of the rule against Salmon run at reduced depth exists; the
premise check is at native depth, on one donor, and its pass thresholds were set after its first result.</p>
<p><b>mixQTL is compared on ranking and slope measures only.</b> Its gene-level permutation scan was excluded by
the timing rule, on one loaded timing, so there is no gene-level power or gene-level false-positive rate for mixQTL,
and its efficiency against unit weights compares methods as run, on a different admitted donor set.</p>
<p><b>The joint models are run away from their design.</b> RASQUAL sees Salmon's haplotype point estimates, rounded,
at one pseudo feature SNP per gene, not reads at each heterozygous exonic SNP. The processes its read-level parameters
model, reference-mapping bias, sequencing errors and uncertain genotypes, are absent from these data, yet the
parameters are still fitted and do not sit at their no-effect values (the fitted sequencing error rate is not near
zero), so they may absorb other variation; whether they contribute to RASQUAL's shortfall against beta (section 3.3)
is not tested, and this benchmark says nothing about what they buy on real reads. Its defaults are kept, except that
its Hardy-Weinberg filter on tested variants is off (-h 0).</p>
<p>asSeq's joint TReCASE fit is missing in {pct(tr["joint_na_share"])} of the run's {tr["tests"]:,} tests and in
{pct(tr["causal_joint_na_share"])} of its {tr["causal_nonnull"]} causal-variant tests. asSeq's trace logs attribute
{tr["joint_fail_theta"]:,} of the missing fits ({pct(tr["joint_fail_theta"] / tr["tests"])} of all tests) to the
overdispersion step's search ending abnormally in its line search; that search is L-BFGS-B, an iterative optimizer
that approximates the curvature of the likelihood from its gradients, within bounds. The largest absolute gradient at
such a stop was {tr["theta_gradient_max"]:.1e} over the run, against at most 3.4e-3 in the smoke run, which
run_trecase_asseq.py read as a stop at essentially the optimum; whether the full run's abnormal stops are at the optimum
was not checked. Treating them as converged would require patching asSeq and was not done. Where the joint fit is
missing, asSeq's final p is its total-count test; at the causal variants the final p was the total-count test in
{tr["causal_final_trec"]} of {tr["causal_nonnull"]}, the joint test in {tr["causal_final_joint"]}. The trace logs also
count {tr["ase_fail"]:,} failed allele-specific fits and the {tr["linear_dosage"]:,} linear-dosage refits of section 2;
the output rows count {tr["few_het"]:,} tests ({100 * tr["few_het"] / tr["tests"]:.1f}%) with fewer than five
heterozygous donors, for which asSeq fits no allele-specific model (the classes can overlap). In {tr["df_not_1"]:,}
tests ({tr["df_not_1_anchor"]:,} on the anchor) the joint statistic has 0 degrees of freedom (the overdispersion at its
boundary under the alternative), so asSeq reports no p for them. They are absent from the null rates, the ranking and
detection, but their slope and derived standard error (&chi;<sup>2</sup> &gt; 0) enter bias and the precision
statistics, where the derived standard error is not a standard error.</p>
<p>Both joint arms receive exactly the allele-specific records the hapmixQTL arms admit (the zero-haplotype rule of
section 2 removes {tr["zeroed"][0]:,} to {tr["zeroed"][1]:,} haplotype-informative donor-gene pairs per dataset).
Rounding sets {rq["as00"][0]} to {rq["as00"][1]} of them per dataset to zero reads on both haplotypes for RASQUAL, and
asSeq's own floors then drop {tr["asseq_dropped"][0]} to {tr["asseq_dropped"][1]} records per dataset (fewer than five
allele-specific reads) and the allele-specific model at the {tr["few_het"]:,} tests above. Both likelihoods are
written for read counts, and these are Salmon's fractional point estimates, rounded where a model needs integers. And
the injected total fold has, averaged over donors, the dosage form both joint models assume (section 2), while the
linear total channels of hapmixQTL and mixQTL approximate it by a straight line; the generator therefore suits the joint
models' total model.</p>
<p><b>Thresholds.</b> Nothing here tests nominal p below 1e-5, measures a null rate below 1e-3, or tests gene-level
thresholds at transcriptome scale. Of the generator checks, only check (c)'s pass rule was fixed before its first run;
the other thresholds of checks (a) to (c) were not pre-registered (check_generator.py, THRESHOLD PROVENANCE).
{"" if "reproduction" in CG else "Check (d) is not in the check file used here (section 2)."}</p>'''


def page(body):
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" '
            f'content="width=device-width, initial-scale=1"><title>Plasmode eQTL benchmark</title><style>{CSS}</style>'
            f'</head><body><main>{body}</main></body></html>')


def main():
    S, CG, CP, run_log, make_log, rq_log, TS, LD = load()
    OUT.mkdir(exist_ok=True)
    LF, JF = log_facts(run_log, make_log), joint_facts(S, rq_log, TS)
    figs = dict(ranking=fig_ranking(S), bias=fig_bias(S), lead=fig_lead(S), efficiency=fig_efficiency(S))
    body = '\n'.join((sec_head(S), sec_why(), sec_run(S, CG, CP, LF, JF), sec_results(S, CG, figs, LD, JF), sec_critique(S),
                      sec_meaning(S, CG, JF, LD), sec_limits(S, CG, JF)))
    write_atomic(PAGE, page(body).encode())
    print(f'wrote {PAGE} ({PAGE.stat().st_size:,} bytes) and {", ".join(p.name for p in figs.values())} in {OUT}')


if __name__ == '__main__':
    main()

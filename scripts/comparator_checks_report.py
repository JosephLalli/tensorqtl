"""One page for the two comparator checks of 2026-10-02 (that day's plan, items 1 and 2): RASQUAL scored with its
non-converged rows kept (benchmark/simulated_effects/rasqual_nonconverged.py) and the benchmark drawn from TReCASE's
own generative model rerun on the shipped default (scripts/external_benchmark_mirror.py). Every number on the page is
read from those runs' files, except the split-minus-unit power difference and its standard error, computed here by
the driver's own resampling of null and alternative replicates.

Output: OUT/index.html.
"""
import base64
import html
import io
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

D = Path('/mnt/ssd/lalli/brainvar_hapmix_deploy')
RN = D / 'rasqual_nonconverged_20261002'
SETS = (('corrected_null_store_20260925', 'Deep set'), ('stratum30_100', 'Low-coverage set'))
MIRROR, MIRROR_OLD = D / 'external_benchmark_half_read_20261002', D / 'external_benchmark_current_20260928'
OUT = D / 'comparator_checks_20261002'
SEED, N_RESAMPLE = 42, 2000          # as external_benchmark_mirror.py
BETAS, ALPHAS, FOLDS, NS = ('0.2', '0.4', '0.8'), ('0.05', '0.01', '0.001'), ('1.05', '1.1', '1.2'), ('200', '92')
FOLD = {'1.05': '1.05', '1.1': '1.10', '1.2': '1.20'}   # the summary's keys, as the prose writes them
TREAT = (('delivered', 'dropped (as delivered)', '#575b61'), ('reported', 'kept at the reported statistic', '#2a78d6'),
         ('p_one', 'kept at p = 1', '#eb6834'))
MARMS = (('hapmixQTL split', 'split (shipped)', '#2a78d6'), ('hapmixQTL unit', 'unit weights (ablation)', '#eb6834'),
         ('hapmixQTL gibbs', 'gibbs', '#1baf7a'), ('TReCASE (joint)', 'TReCASE, its own likelihood', '#4a3aa7'),
         ('TReCASE (asSeq final p)', 'asSeq, final p', '#e34948'), ('TReCASE (asSeq joint p)', 'asSeq, joint p', '#7a1016'))
RC = json.loads((RN / SETS[0][0] / 'compare.json').read_text()), json.loads((RN / SETS[1][0] / 'compare.json').read_text())
M, M_OLD = json.loads((MIRROR / 'summary.json').read_text()), json.loads((MIRROR_OLD / 'summary.json').read_text())
PR = pd.read_csv(MIRROR / 'per_replicate.tsv.gz', sep='\t')
esc = html.escape


def png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def head(rc, t):
    return rc['delivered'] if t == 'delivered' else rc['variants'][t]['headline']


def fig_rasqual():
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.4))
    for row, (rc, (_, name)) in enumerate(zip(RC, SETS)):
        ax = axes[row, 0]
        split = json.loads((Path(rc['delivered_root']) / 'summary.json').read_text())['ranking']
        ax.plot(range(3), [split[f'beta{b}']['split']['fdp_matched']['all']['power'] for b in BETAS], color='#9a9c9f', lw=1.2,
                ls=':', marker='.', label='split (shipped), for reference')
        for j, (t, lab, col) in enumerate(TREAT):
            ax.plot(range(3), head(rc, t)['power'], marker='osD'[j], color=col, label=f'RASQUAL, {lab}', ms=6, lw=1.6,
                    ls='-' if j == 0 else '--', alpha=0.9)
        ax.set_xticks(range(3), [f'|beta| {b}' for b in BETAS])
        ax.set_ylim(-0.02, 1)
        ax.set_ylabel('power at 5% realized FDP')
        ax.set_title(f'{name}: RASQUAL power', loc='left', fontsize=10)
        ax = axes[row, 1]
        for j, (t, lab, col) in enumerate(TREAT):
            for i, al in enumerate(ALPHAS):
                n = head(rc, t)['anchor_null'][al]
                a = float(al)
                ax.errorbar(i + (j - 1) * 0.18, n['rate'] / a, yerr=[[(n['rate'] - n['lo']) / a], [(n['hi'] - n['rate']) / a]],
                            fmt='osD'[j], color=col, ms=5, capsize=2)
        ax.axhline(1, color='#575b61', lw=1, ls=':')
        ax.set_xticks(range(3), [f'p < {al}' for al in ALPHAS])
        ax.set_ylabel('null-gene rate / threshold')
        ax.set_title(f'{name}: RASQUAL on the all-null dataset', loc='left', fontsize=10)
    for ax in axes.flat:
        ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='lower center', ncol=2, frameon=False, fontsize=8.5)
    return png(fig)


def fig_mirror():
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.6), gridspec_kw=dict(width_ratios=[1, 1.15]))
    for row, N in enumerate(NS):
        ax = axes[row, 0]
        for i, al in enumerate(('0.05', '0.01')):
            se = M['by_N'][N]['type1']['hapmixQTL split'][al]['mc_se'] / float(al)
            ax.axhspan(1 - 2 * se, 1 + 2 * se, xmin=i / 2 + 0.03, xmax=(i + 1) / 2 - 0.03, color='#e4e3df', lw=0)
            for j, (arm, lab, col) in enumerate(MARMS):
                ax.plot(i + (j - 2.5) * 0.1, M['by_N'][N]['type1'][arm][al]['rate'] / float(al), 'o', color=col, ms=6,
                        label=lab if i == 0 else None)
        ax.axhline(1, color='#575b61', lw=1, ls=':')
        ax.set_xticks(range(2), ['p < 0.05', 'p < 0.01'])
        ax.set_xlim(-0.6, 1.6)
        ax.set_ylabel('share of null replicates / threshold')
        ax.set_title(f'N = {N}: null replicates (grey: nominal ± 2 Monte Carlo SE)', loc='left', fontsize=10)
        ax = axes[row, 1]
        for j, (arm, lab, col) in enumerate(MARMS):
            ax.plot(range(3), [M['by_N'][N]['power'][k][arm]['matched'] for k in FOLDS], marker='o', color=col, lw=1.6,
                    ms=5, label=lab)
        ax.plot(range(3), [M_OLD['by_N'][N]['power'][k]['hapmixQTL split']['matched'] for k in FOLDS], 'o', mfc='none',
                mec='#0b0b0b', ms=10, label='split on the earlier total (2026-09-28)')
        ax.set_xticks(range(3), [f'allelic fold {FOLD[k]}' for k in FOLDS])
        ax.set_ylim(0, 1.02)
        ax.set_ylabel('power at matched 5% type-I error')
        ax.set_title(f'N = {N}: power', loc='left', fontsize=10)
    for ax in axes.flat:
        ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.legend(*axes[0, 1].get_legend_handles_labels(), loc='lower center', ncol=4, frameon=False, fontsize=8.5)
    return png(fig)


def split_minus_unit(N, kappa):
    """Matched-power difference and its standard error, resampling null and alternative replicates as the driver does."""
    arms = ('hapmixQTL split', 'hapmixQTL unit')
    S0 = PR[(PR.N == int(N)) & (PR.kappa == 1.0)][[f'{a}|stat' for a in arms]].values
    S1 = PR[(PR.N == int(N)) & np.isclose(PR.kappa, float(kappa))][[f'{a}|stat' for a in arms]].values
    det = (S1 > np.quantile(S0, 0.95, axis=0)).mean(0)
    rng = np.random.default_rng(np.random.SeedSequence(SEED, spawn_key=(int(N), int(round(1000 * float(kappa))))))
    res = np.array([(S1[rng.integers(0, len(S1), len(S1))] > np.quantile(S0[rng.integers(0, len(S0), len(S0))], 0.95, axis=0)).mean(0)
                    for _ in range(N_RESAMPLE)])
    return float(det[0] - det[1]), float((res[:, 0] - res[:, 1]).std(ddof=1))


CSS = '''
/* layout: one reading column; each check a section with its figure first, then why / what ran / result / critique / meaning */
:root { --bg: #f7f7f4; --panel: #ffffff; --ink: #15171a; --ink2: #575b61; --rule: #dfe0db; --accent: #2a78d6;
  --display: "Source Serif 4", Georgia, serif; --body: "IBM Plex Sans", system-ui, sans-serif; --mono: "IBM Plex Mono", ui-monospace, monospace; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --bg: #141517; --panel: #1c1d20; --ink: #f1f1ee;
  --ink2: #b4b6b9; --rule: #33353a; --accent: #3987e5; color-scheme: dark; } }
:root[data-theme="dark"] { --bg: #141517; --panel: #1c1d20; --ink: #f1f1ee; --ink2: #b4b6b9; --rule: #33353a; --accent: #3987e5;
  color-scheme: dark; }
body { background: var(--bg); color: var(--ink); font: 15px/1.6 var(--body); }
main { max-width: 980px; margin: 0 auto; padding-inline: 16px; padding-block: 32px 72px; display: grid; gap: 28px; }
h1 { font: 600 30px/1.15 var(--display); margin: 0; text-wrap: balance; }
h2 { font: 600 21px/1.25 var(--display); margin: 0; text-wrap: balance; }
p { margin: 0; max-width: 75ch; }
.dek { color: var(--ink2); margin-top: 10px; }
.eyebrow { font: 600 11.5px var(--mono); letter-spacing: 0.08em; text-transform: uppercase; color: var(--accent); margin: 0 0 4px; }
section.check { background: var(--panel); border: 1px solid var(--rule); border-radius: 8px; padding: 20px; display: grid; gap: 12px; min-width: 0; }
.finding { font-size: 16px; }
figure { margin: 4px 0; } figure img { max-width: 100%; height: auto; display: block; background: #ffffff; border-radius: 6px; }
figcaption { color: var(--ink2); font-size: 13px; margin-top: 6px; max-width: 90ch; }
h3 { font: 600 13px var(--mono); letter-spacing: 0.06em; text-transform: uppercase; color: var(--ink2); margin: 8px 0 0; }
.scroll { overflow-x: auto; min-width: 0; }
table { border-collapse: collapse; font-size: 13px; }
th, td { border-bottom: 1px solid var(--rule); padding: 4px 10px 4px 0; text-align: left; vertical-align: top; }
td { font-variant-numeric: tabular-nums; } th { color: var(--ink2); font-weight: 600; }
footer { color: var(--ink2); font-size: 13px; }
code { font: 0.88em var(--mono); }
'''


def f3(x):
    return f'{x:.3f}'


def t1(N, arm, al):
    return M['by_N'][N]['type1'][arm][al]['rate']


def sec_rasqual():
    pooled = [json.loads((Path(rc['delivered_root']) / 'results_rasqual' / 'summary.json').read_text())['pooled'] for rc in RC]
    full = [json.loads((Path(rc['delivered_root']) / 'summary.json').read_text()) for rc in RC]
    for rc, p in zip(RC, pooled):
        if rc['nonconverged_added']['nonconverged'] != p['nonconv']:
            raise SystemExit(f'{rc["gene_set"]}: non-converged rows added differ from 04\'s count')
        for v in ('reported', 'p_one'):
            if rc['variants'][v]['other_arm_differences']:
                raise SystemExit(f'{rc["gene_set"]} {v}: numbers outside RASQUAL differ from the delivered summary')
    change = max(abs(a - b) for rc in RC for v in ('reported', 'p_one')
                 for a, b in zip(head(rc, v)['power'], head(rc, 'delivered')['power']))
    gap = [S['ranking']['beta0.8']['split']['fdp_matched']['all']['power'] - S['ranking']['beta0.8']['rasqual']['fdp_matched']['all']['power']
           for S in full]
    names = [n.lower() for _, n in SETS]
    pos = [rc['nonconverged_added']['chisq_positive'] / rc['nonconverged_added']['nonconverged'] for rc in RC]

    def ivl(n):
        return f'{n["rate"]:.4f} [{n["lo"]:.4f}, {n["hi"]:.4f}]'

    def moved(lc):
        return f'{lc["units"]} ({lc["nonnull"]} non-null; onto the causal variant {lc["to_causal"]}, off it {lc["from_causal"]})'

    rows = ''
    for rc, (_, name) in zip(RC, SETS):
        for t, lab, _ in TREAT:
            h = head(rc, t)
            lead = '' if t == 'delivered' else moved(rc['lead_changes'][t])
            rows += (f'<tr><td>{esc(name)}</td><td>{esc(lab)}</td><td>{" / ".join(map(f3, h["power"]))}</td>'
                     f'<td>{ivl(h["anchor_null"]["0.05"])}</td><td>{ivl(h["anchor_null"]["0.001"])}</td>'
                     f'<td>{" / ".join(map(str, h["missing_causal"]))}</td><td>{lead}</td></tr>')
    null05 = {t: [f'{head(rc, t)["anchor_null"]["0.05"]["rate"]:.4f}' for rc in RC] for t, *_ in TREAT}
    small = change < 0.25 * min(gap)
    pct = [f'{p["nonconv"] / p["tests"]:.1%}' for p in pooled]
    moves = ('leaves its power at 5% realized false discoveries unchanged at every |beta| on both gene sets' if change == 0
             else f'moves its power at 5% realized false discoveries by at most {f3(change)} at any |beta| on either gene set')
    dnull = max(abs(head(rc, t)['anchor_null'][al]['rate'] - head(rc, 'delivered')['anchor_null'][al]['rate'])
                for rc in RC for t in ('reported', 'p_one') for al in ALPHAS)
    lost = max(max(head(rc, 'delivered')['missing_causal']) for rc in RC)
    if not small:
        meaning = ('The handling of RASQUAL\'s failed fits moves its power by an amount comparable to part of its gap to split, '
                   'so the gap should be read with this bound beside it.')
    elif change == 0:
        meaning = (f'The benchmark\'s handling of RASQUAL\'s failed fits is not what holds it back: keeping them changes no power '
                   f'figure and moves its null-gene rates by at most {dnull:.4f}; what they add is a row for the causal units that '
                   f'had none (at most {lost} per |beta|).')
    else:
        meaning = ('The benchmark\'s handling of RASQUAL\'s failed fits is not what holds it back: the largest change any '
                   'treatment makes is a fraction of its gap to split.')
    return f'''<section class="check" id="rasqual"><p class="eyebrow">Check 1 · RASQUAL's non-converged rows</p>
<h2>Dropping RASQUAL's failed fits {"does not explain" if small else "explains part of"} where it stands</h2>
<p class="finding">Keeping RASQUAL's non-converged rows instead of dropping them {moves}. RASQUAL trails split by
{f3(gap[0])} on the {names[0]} and {f3(gap[1])} on the {names[1]} at |beta| 0.8.</p>
<figure><img alt="RASQUAL's power by effect size and its null-gene rate on the all-null dataset, with non-converged rows dropped, kept at their reported statistic, or kept at p = 1, on both gene sets" src="data:image/png;base64,{fig_rasqual()}">
<figcaption>Left: RASQUAL's power at 5% realized false-discovery proportion, the share of non-null gene units called at the
deepest point of the pooled ranking by lead p where at most 5% of the units called are null genes (3 datasets per |beta|,
no interval){"; the three treatments' lines lie on top of one another" if change == 0 else ""}, with split's for reference.
Right: RASQUAL's null-gene rate on the all-null dataset, the share of tested variants of null genes with p below the
threshold, divided by the threshold (1 = nominal), with gene-clustered 95% intervals.</figcaption></figure>
<h3>Why it was needed</h3>
<p>RASQUAL's fit does not converge in {pooled[0]["nonconv"]:,} of {pooled[0]["tests"]:,} tests on the {names[0]} ({pct[0]}) and
{pooled[1]["nonconv"]:,} of {pooled[1]["tests"]:,} on the {names[1]} ({pct[1]}), among them {pooled[0]["causal_nonconv"]} and
{pooled[1]["causal_nonconv"]} of the 450 tests at a causal variant, and the benchmark drops those rows. Dropping is a
selection: a gene whose strongest variant failed to converge is ranked by a weaker one, and a causal unit without a row
leaves detection, bias and precision. How much that moves RASQUAL's standing had not been measured.</p>
<h3>What was run</h3>
<p>RASQUAL's results were rebuilt from the raw rows its driver keeps, with the driver's own code (<code>04_run_rasqual.py</code>,
<code>assemble</code>), three ways. Converged rows alone reproduced the delivered files in every column on all 10 datasets of
both sets, the known answer for the rebuild. The non-converged rows were then added in two ways: at the likelihood-ratio
statistic RASQUAL reports for them, and at p = 1. Only {pos[0]:.0%} ({names[0]}) and {pos[1]:.0%} ({names[1]}) of them report a
positive statistic; the rest report zero or a negative value (no completed fit gives a negative one), and both treatments
score those as p = 1, so the two treatments differ only in the positive ones. Each treatment was scored by the benchmark's unchanged scoring
step (<code>06_score.py</code>) in a directory whose every other input is the delivered run's, and every number outside
RASQUAL came out identical to the delivered summary.</p>
<h3>Result</h3>
<p>On the all-null dataset RASQUAL's null-gene rate at 0.05 is {null05["delivered"][0]} with the rows dropped,
{null05["reported"][0]} at their reported statistic and {null05["p_one"][0]} at p = 1 on the {names[0]}, and
{null05["delivered"][1]}, {null05["reported"][1]} and {null05["p_one"][1]} on the {names[1]}. The table gives power
at |beta| 0.2 / 0.4 / 0.8, the null-gene rates with their intervals, the causal units without a row at each |beta|, and the
gene-dataset units whose lead variant (smallest p) moves when the rows are kept.</p>
<div class="scroll"><table><thead><tr><th>gene set</th><th>non-converged rows</th><th>power, |beta| 0.2 / 0.4 / 0.8</th>
<th>null-gene rate at 0.05</th><th>at 0.001</th><th>causal units without a row</th><th>lead variant moved</th></tr></thead>
<tbody>{rows}</tbody></table></div>
<h3>Critique</h3>
<p>The reported statistics come from fits that did not finish, and a negative likelihood-ratio statistic is not a valid
one; kept at p = 1 is the most conservative reading. Neither is what a converged fit would give, and RASQUAL offers no
option that converges them: fixing its error-rate parameter removed almost all non-convergence on these datasets but raised
its null-gene rate at 0.05 to 1.55 times nominal (input diagnosis, 2026-09-28). This check bounds how much the dropping
matters; it cannot say what RASQUAL would score with every fit converged.</p>
<h3>What it means</h3>
<p>{meaning}
What remains of the concern that RASQUAL runs away from its design is its input: the pseudo feature SNP, whose estimated
error rate is tied to the non-convergence, and RASQUAL on its own per-SNP read counts, which this check does not touch.</p></section>'''


def sec_mirror():
    pw = lambda N, a: [M['by_N'][N]['power'][k][a]['matched'] for k in FOLDS]   # noqa: E731
    old = lambda N: [M_OLD['by_N'][N]['power'][k]['hapmixQTL split']['matched'] for k in FOLDS]   # noqa: E731
    dvt = lambda N, k: M['by_N'][N]['power'][k]['hapmixQTL split']['diff_vs_TReCASE (joint)']   # noqa: E731
    shift = max(abs(a - b) for N in NS for a, b in zip(pw(N, 'hapmixQTL split'), old(N)))
    same_t1 = all(t1(N, 'hapmixQTL split', al) == M_OLD['by_N'][N]['type1']['hapmixQTL split'][al]['rate'] for N in NS for al in ALPHAS)
    su = {(N, k): split_minus_unit(N, k) for N in NS for k in FOLDS[:2]}
    rows = ''.join(f'<tr><td>{N}</td><td>{FOLD[k]}</td><td>{f3(dvt(N, k)["diff"])} ({f3(dvt(N, k)["se_resample"])})</td>'
                   f'<td>{f3(su[(N, k)][0])} ({f3(su[(N, k)][1])})</td></tr>' for N in NS for k in FOLDS[:2])
    se05 = M['by_N']['200']['type1']['hapmixQTL split']['0.05']['mc_se']
    se01 = M['by_N']['200']['type1']['hapmixQTL split']['0.01']['mc_se']
    s = lambda N, a: f'{f3(t1(N, a, "0.05"))} / {f3(t1(N, a, "0.01"))}'   # noqa: E731
    gaps = [(N, k, dvt(N, k)['diff'] / dvt(N, k)['se_resample']) for N in NS for k in FOLDS if dvt(N, k)['se_resample'] > 0]
    worst = min(gaps, key=lambda g: g[2])
    within = all(abs(z) < 2 for *_, z in gaps)
    su_within = all(abs(d / se) < 2 for d, se in su.values())
    asf = {N: M['by_N'][N]['power']['1.2']['TReCASE (asSeq final p)']['diff_vs_TReCASE (joint)'] for N in NS}
    verdict = ('stays within two resampled standard errors of TReCASE\'s own likelihood at every fold and both sizes' if within
               else 'falls more than two resampled standard errors behind TReCASE\'s own likelihood somewhere')
    z_as = {N: (t1(N, 'TReCASE (asSeq final p)', '0.05') - 0.05) / M['by_N'][N]['type1']['TReCASE (asSeq final p)']['0.05']['mc_se']
            for N in NS}
    rec = M_OLD['by_N'][worst[0]]['power'][worst[1]]['hapmixQTL split']['diff_vs_TReCASE (joint)']
    return f'''<section class="check" id="mirror"><p class="eyebrow">Check 2 · data drawn from TReCASE's own model</p>
<h2>On TReCASE's own model, the shipped default keeps its standing</h2>
<p class="finding">Split on the shipped half-read total {verdict}: at 200 donors its power is
{" / ".join(map(f3, pw("200", "hapmixQTL split")))} against {" / ".join(map(f3, pw("200", "TReCASE (joint)")))} at allelic folds
{" / ".join(FOLD[k] for k in FOLDS)}, and its largest shortfall is {f3(-dvt(worst[0], worst[1])["diff"])} at fold {FOLD[worst[1]]} with {worst[0]} donors
({abs(worst[2]):.1f} standard errors). Moving to the half-read total changed split's power by at most
{f3(shift)}{" and its null rejections not at all" if same_t1 else ""}.</p>
<figure><img alt="Share of null replicates below 0.05 and 0.01 for each method, and power at matched type-I error by allelic fold, at 200 and 92 donors" src="data:image/png;base64,{fig_mirror()}">
<figcaption>Left: share of the 500 null replicates with p below 0.05 and 0.01, divided by the threshold (1 = nominal; grey band:
nominal ± 2 Monte Carlo standard errors). Right: power at matched type-I error, the share of 500 replicates at each allelic fold
whose statistic exceeds the 95th percentile of that method's own statistic over the null replicates, so a method whose null
runs high cannot gain by it. Hollow circles: split on the earlier log2(T/lib + 1) total, 2026-09-28.</figcaption></figure>
<h3>Why it was needed</h3>
<p>On 2026-09-28 this benchmark showed split at parity with TReCASE on data drawn from TReCASE's own generative model:
negative-binomial totals and beta-binomial allele-specific counts, the likelihood TReCASE fits. That run gave split the
earlier total, log2(T/lib + 1), and the weightings gibbs, split and plus_one. The shipped default has since moved to the
half-read total with the zero-haplotype admission rule, and plus_one was retired, so the parity claim had not been
measured on what ships.</p>
<h3>What was run</h3>
<p>The same harness, design and seeds as on 2026-09-28: 500 replicates per condition, 200 and 92 donors, a null and allelic
folds 1.05 / 1.10 / 1.20, a mean total of 200 reads, negative-binomial dispersion 0.2, beta-binomial overdispersion 0.01,
and a quarter of reads allele-specific. The hapmixQTL arms now take the shipped default's inputs
(<code>prepare_default_inputs</code>): split, unit weights (the ablation without the Gibbs draws) and gibbs. The comparators
are unchanged: TReCASE written into the harness with its own likelihood, and asSeq's TReCASE as published, read both by its
final p, the one a user gets, and by its joint p. asSeq was not rerun. Its 2026-09-28 outputs were reused after the
regenerated asSeq inputs proved byte-identical in all 8 conditions and the harness's comparators reproduced the earlier run
exactly. Each method's power is read at its own null's 95th percentile.</p>
<h3>Result</h3>
<p>For split, the share of null replicates with p below 0.05 / 0.01 is {s("200", "hapmixQTL split")} at 200 donors and
{s("92", "hapmixQTL split")} at 92, where one Monte Carlo standard error is {se05:.4f} / {se01:.4f}. TReCASE's own likelihood
gives {s("200", "TReCASE (joint)")} and {s("92", "TReCASE (joint)")}. asSeq's final p gives {f3(t1("200", "TReCASE (asSeq final p)", "0.05"))}
and {f3(t1("92", "TReCASE (asSeq final p)", "0.05"))} below 0.05, {z_as["200"]:.1f} and {z_as["92"]:.1f} Monte Carlo standard errors above
nominal. The table gives the power differences
at the two folds where not every method is near 1, each with its standard error from resampling null and alternative
replicates (2,000 resamples, thresholds recomputed each time).</p>
<div class="scroll"><table><thead><tr><th>donors</th><th>allelic fold</th><th>split minus TReCASE (SE)</th><th>split minus unit weights (SE)</th></tr></thead>
<tbody>{rows}</tbody></table></div>
<h3>Critique</h3>
<p>These data come from TReCASE's own model, the best case for TReCASE, so parity here is parity with the method that fits the
generating likelihood, and nothing here tests real data. The Gibbs variance is emulated (binomial allelic and Poisson total
draws), not Salmon's. At a mean of 200 reads the half-read and earlier totals differ by a fraction of a read, so this design
cannot show where the two totals part, at low depth. With 500 replicates per condition a power difference has a standard
error of 0.02 to 0.04 once each method's null threshold is resampled too (the table); a paired error that holds the
thresholds fixed is about half that and overstates the evidence for any gap. {"Unit weights stay within two standard errors of split at every fold and size, so this design cannot measure what the Gibbs weights add." if su_within else "Unit weights differ from split by more than two standard errors somewhere (the table)."}</p>
<h3>What it means</h3>
<p>The external evidence carries over to the shipped default unchanged: split is level with TReCASE's own likelihood within
this design's resolution at 200 and 92 donors. The 2026-09-28 record quotes the gap at fold {FOLD[worst[1]]} with {worst[0]} donors
with a paired standard error of {f3(rec["se_paired"])}, under which it reads as {abs(rec["diff"] / rec["se_paired"]):.1f} standard errors;
with the thresholds resampled it is {abs(worst[2]):.1f}. asSeq's final p, as users run it, rejects null replicates above nominal on
its own model and finds fewer effects than TReCASE's own likelihood at fold 1.20 (by {f3(-asf["200"]["diff"])} and
{f3(-asf["92"]["diff"])} at 200 and 92 donors, standard errors {f3(asf["200"]["se_resample"])} and {f3(asf["92"]["se_resample"])}).
Gibbs is on this page because the simulated-effects benchmark carries it; under the paper's final-version rule the figure
would show split and unit weights only.</p></section>'''


def main():
    page = ('<title>Comparator fairness checks</title>\n'
            '<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>\n'
            '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:'
            'wght@400;600&family=Source+Serif+4:opsz,wght@8..60,600&display=swap">\n'
            f'<style>{CSS}</style>\n<main><header><p class="eyebrow">hapmixQTL · 2026-10-02</p><h1>Comparator fairness checks</h1>'
            '<p class="dek">Two checks of whether the benchmarks treat RASQUAL and TReCASE fairly: RASQUAL scored with the fits it '
            'failed to finish kept instead of dropped, and the benchmark drawn from TReCASE\'s own model run again on the shipped '
            'default.</p></header>'
            f'{sec_rasqual()}{sec_mirror()}'
            f'<footer><p>Check 1: <code>benchmark/simulated_effects/rasqual_nonconverged.py</code>, outputs in <code>{RN}</code>. '
            f'Check 2: <code>scripts/external_benchmark_mirror.py</code> with <code>tests/ase_external_benchmark.py</code>, outputs in '
            f'<code>{MIRROR}</code>; the 2026-09-28 run on the earlier total is <code>{MIRROR_OLD}</code>. This page: '
            '<code>scripts/comparator_checks_report.py</code>.</p></footer></main>')
    OUT.mkdir(exist_ok=True)
    tmp = OUT / 'index.html.tmp'
    tmp.write_text(page)
    tmp.replace(OUT / 'index.html')
    print(f'wrote {OUT / "index.html"} ({len(page.encode()):,} bytes)')


if __name__ == '__main__':
    main()

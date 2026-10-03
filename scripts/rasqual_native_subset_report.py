"""Report the saved native RASQUAL subset scores, without refitting or rescoring.

Run after benchmark/simulated_effects/rasqual_subset_score.py for both gene sets.
The existing full-scan reports remain separate records. This page compares only
the identical completed genes and variant subsets used by the subset scorer.
"""
import argparse
import base64
import hashlib
import html
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


DEPLOY = Path('/mnt/ssd/lalli/brainvar_hapmix_deploy')
SETS = {'Deep': 'simulated_effects_half_read_20261001',
        'Low coverage': 'simulated_effects_lowcov_half_read_20261001'}
LABELS = {'split': 'hapmixQTL default', 'unit': 'Without Gibbs weights',
          'mixqtl': 'mixQTL, published cutoffs', 'tensorqtl': 'tensorQTL, total only',
          'rasqual': 'RASQUAL, synthetic SNP', 'trecase': 'TReCASE, Salmon inputs',
          'split_native': 'hapmixQTL, alignment counts',
          'trecase_native': 'TReCASE, alignment counts',
          'rasqual_native': 'RASQUAL, per-SNP alignment counts'}
MAIN = ('split', 'split_native', 'rasqual', 'rasqual_native', 'tensorqtl')
SCENARIOS = ('beta0.2', 'beta0.4', 'beta0.8')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_records(deploy):
    records, sources = {}, {}
    for label, folder in SETS.items():
        root = deploy / folder
        out = root / 'native/results_rasqual_subset'
        paths = [out / 'score.json', out / 'summary.json', root / 'native/rasqual_inputs/subset.tsv']
        score, run = (json.loads(p.read_text()) for p in paths[:2])
        if set(score['genes']) != set(run['genes']) or len(score['genes']) != 100:
            raise ValueError(f'{label}: score and run must contain the same 100 genes')
        if score['n_variants'] != 5200 or len(run['per_dataset']) != 10:
            raise ValueError(f'{label}: expected 5,200 subset variants and ten datasets')
        if set(score['ranking']) != set(SCENARIOS):
            raise ValueError(f'{label}: unexpected effect scenarios')
        for sc in SCENARIOS:
            for arm in LABELS:
                entry = score['ranking'][sc][arm]
                if entry['non_null'] != 150 or not 0 <= entry['power'] <= 1:
                    raise ValueError(f'{label} {sc} {arm}: invalid power denominator/value')
        records[label] = {'root': root, 'score': score, 'run': run}
        for p in paths:
            sources[str(p)] = digest(p)
    return records, sources


def figure(records, out):
    colors = {'split': '#087e8b', 'split_native': '#087e8b', 'rasqual': '#c66624',
              'rasqual_native': '#c66624', 'tensorqtl': '#64748b'}
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for col, (label, rec) in enumerate(records.items()):
        score = rec['score']
        ax = axes[0, col]
        for arm in MAIN:
            ax.plot([.2, .4, .8], [score['ranking'][sc][arm]['power'] for sc in SCENARIOS],
                    color=colors[arm], marker='s' if arm.endswith('_native') else 'o',
                    linestyle='--' if arm.endswith('_native') or arm == 'tensorqtl' else '-',
                    linewidth=2, label=LABELS[arm])
        ax.set(title=f'{label}: benchmark power', xlabel='Simulated effect (log2 scale)',
               ylabel='Share of 150 non-null gene–dataset pairs found', ylim=(-.02, 1.02), xticks=[.2, .4, .8])
        ax.grid(alpha=.16)
        ax = axes[1, col]
        arms = ('split', 'split_native', 'tensorqtl', 'trecase', 'trecase_native', 'rasqual', 'rasqual_native')
        values = [score['null'][a]['rate'] for a in arms]
        y = np.arange(len(arms))
        ax.barh(y, values, color=['#087e8b', '#087e8b', '#64748b', '#9b5c9b', '#9b5c9b', '#c66624', '#c66624'])
        ax.set(yticks=y, yticklabels=[LABELS[a] for a in arms], xlim=(0, .115),
               title=f'{label}: all-null anchor', xlabel='Share of returned variant tests with p < 0.05')
        ax.invert_yaxis()
        ax.axvline(.05, color='#263238', linestyle=':', label='Nominal 0.05')
        for yy, value in zip(y, values):
            ax.text(value + .002, yy, f'{value:.3f}', va='center', fontsize=9)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside upper center', ncol=3, frameon=False, fontsize=10)
    fig.savefig(out / 'comparison.png', dpi=150, facecolor='white')
    fig.savefig(out / 'comparison.svg', facecolor='white')
    plt.close(fig)


def table(score):
    rows = []
    for arm, label in LABELS.items():
        null = score['null'][arm]
        rejections = round(null['rate'] * null['tests'])
        cells = [html.escape(label), f"{null['rate']:.4f}", f"{rejections:,} / {null['tests']:,}"]
        cells.extend(f"{score['ranking'][sc][arm]['power']:.3f}" for sc in SCENARIOS)
        rows.append('<tr>' + ''.join(f'<td>{x}</td>' for x in cells) + '</tr>')
    return ('<div class="scroll"><table><thead><tr><th>Arm</th><th>Null share</th><th>p &lt; 0.05 / returned tests</th>'
            '<th>Power, β = 0.2</th><th>Power, β = 0.4</th><th>Power, β = 0.8</th></tr></thead><tbody>'
            + ''.join(rows) + '</tbody></table></div>')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deploy-root', type=Path, default=DEPLOY)
    parser.add_argument('--output', type=Path, default=DEPLOY / 'rasqual_native_subset_20261003')
    args = parser.parse_args()
    records, sources = load_records(args.deploy_root)
    args.output.mkdir(parents=True, exist_ok=True)
    figure(records, args.output)
    image = base64.b64encode((args.output / 'comparison.png').read_bytes()).decode()
    sections = []
    for label, rec in records.items():
        score = rec['score']
        counts = list(rec['run']['per_dataset'].values())
        fractions = [c['nonconv'] / (c['rows'] + c['nonconv']) for c in counts]
        skip = sum(c.get('skipped_by_rasqual', 0) for c in counts)
        sections.append(f'<h2>{html.escape(label)}</h2>{table(score)}'
                        f'<p>Native RASQUAL omitted {min(fractions):.1%}–{max(fractions):.1%} '
                        f'of emitted fits for non-convergence across the ten datasets. '
                        f'The run recorded {skip} skipped gene–dataset jobs. '
                        'These returned-fit shares do not include the skipped jobs. '
                        'A skipped gene remains in the ranking denominator with no finite lead p-value.</p>')
    paths = ''.join(f'<li><code>{html.escape(p)}</code></li>' for p in sources)
    deep = records['Deep']['score']['ranking']['beta0.4']
    low = records['Low coverage']['score']['ranking']['beta0.4']
    count = lambda rec, arm: round(rec[arm]['power'] * rec[arm]['non_null'])
    result = (f'At β = 0.4, native RASQUAL found {count(deep, "rasqual_native")} of 150 '
              f'non-null gene–dataset pairs in the deep set, against {count(deep, "rasqual")} with the synthetic SNP '
              f'and {count(deep, "split")} for hapmixQTL default. In low coverage the corresponding counts were '
              f'{count(low, "rasqual_native")}, {count(low, "rasqual")} and {count(low, "split")} of 150. '
              'These are descriptive differences; no power intervals or significance test was computed.')
    page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>RASQUAL native subset comparison</title>
<style>body{{font:16px/1.6 system-ui,sans-serif;color:#24313b;background:#f3f6f7;margin:0}}
main{{max-width:1200px;margin:auto;padding:32px}}h1{{font-size:36px;line-height:1.2}}h2{{margin-top:32px}}
.card{{background:white;border:1px solid #dbe3e7;border-radius:12px;padding:24px;margin:20px 0}}
.eyebrow{{color:#526975;letter-spacing:.08em;font-size:13px}}img{{width:100%;height:auto}}
table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:10px;text-align:right;border-bottom:1px solid #e0e7eb}}
th:first-child,td:first-child{{text-align:left}}.scroll{{overflow-x:auto}}code{{overflow-wrap:anywhere;font-size:12px}}
.limits{{border-left:5px solid #c66624}}a{{color:#087e8b}}</style></head><body><main>
<div class="eyebrow">SIMULATED-EFFECTS BENCHMARK · SAVED RESULTS · 3 OCTOBER 2026</div>
<h1>RASQUAL with per-SNP alignment counts</h1>
<p>Both 100-gene sets are now scored on the same 52 variants per gene for every arm.
This completes the subset comparison authorized after the full native scan proved too expensive.</p>
<div class="card"><p>{result}</p></div>
<section class="card"><h2>What the figures measure</h2>
<p>Power is the share of 150 non-null gene–dataset pairs found per effect size, pooled over three datasets.
The scorer chooses the largest tied-rank cutoff at which at most 5% of calls are known to be null.
This uses simulation truth; it is not a calibrated discovery procedure for observed data.
Each set also has one all-null anchor dataset. Its null share is the fraction of returned nominal
variant tests with p &lt; 0.05; 0.05 is the nominal reference.</p>
<img alt="Power across three effect sizes and returned-test null shares, separately for deep and low-coverage genes"
src="data:image/png;base64,{image}"></section>
<section class="card">{''.join(sections)}</section>
<section class="card limits"><h2>What this comparison can establish</h2>
<p>The comparison measures ranking performance of the saved arms on this fixed subset.
The subset is the union of three simulation-designated variants and 49 other variants sampled with seed 42.
Every arm receives the same designated variants, including in null genes. This gives the benchmark knowledge
of the simulated causal variants that an observed-data scan would not have.</p>
<p>These scores must not be compared with the earlier full-window power figures. Power differences here
have no computed confidence intervals. The null anchor contains correlated variants within genes and only
one dataset, so its returned-test share cannot establish gene-level calibration or a population error rate.
Missing and non-converged tests affect the returned-test denominators, which are shown in the tables.</p>
<p>The per-SNP inputs address the synthetic-feature-SNP handicap, but retain independent thinning per SNP,
missing allele counts in homozygous donors, statistical phase and Salmon-derived expression covariates.
They are not an end-to-end run of RASQUAL's own read-count pipeline. Comparator software was run as released.</p>
<p>The earlier study that retained non-converged rows applies to synthetic-SNP RASQUAL.
Sensitivity to non-convergence in this native run has not been measured.</p>
</section><details class="card"><summary>Evidence and reproduction</summary><p>Saved scores, run summaries
and variant manifests:</p><ul>{paths}</ul><p>Each native results file was checked against its dataset fingerprint
during completion checks. Source hashes are recorded in <code>manifest.json</code> and check results in
<code>validation.json</code> beside this page. The figure is also
saved as <code>comparison.svg</code>.</p></details></main></body></html>'''
    (args.output / 'index.html').write_text(page)
    outputs = ('index.html', 'comparison.png', 'comparison.svg')
    if (args.output / 'validation.json').exists():
        outputs += ('validation.json',)
    manifest = {'sources': sources, 'report_source': str(Path(__file__).resolve()),
                'report_source_sha256': digest(Path(__file__)),
                'outputs': {name: digest(args.output / name) for name in outputs}}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(args.output / 'index.html')


if __name__ == '__main__':
    main()

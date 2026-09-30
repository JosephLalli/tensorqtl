#!/usr/bin/env python3
"""hapmixQTL against TReCASE on one page with charts: simulated-effect power, false positives and effect-size recovery
(plasmode, both gene sets), the simulation from TReCASE's own model (mirror benchmark), held-out replication (referee
subset), why TReCASE loses power, and run time. Every number is read from the result files below and embedded as
JSON into hapmix_vs_trecase_template.html.

  python3 scripts/hapmix_vs_trecase.py      # writes OUT/hapmix_vs_trecase.html
"""
import json
import os
import re
import statistics
from pathlib import Path

D = Path('/mnt/ssd/lalli/brainvar_hapmix_deploy')
SETS = {'deep': D / 'plasmode_meier_20260927', 'lowcov': D / 'plasmode_lowcov_meier_20260927'}
REF = D / 'referee_replication_20260928'
MIRROR = D / 'external_benchmark_current_20260928' / 'summary.json'
ARMS = ['split', 'gibbs', 'unit', 'plus_one', 'split_native', 'trecase', 'trecase_native', 'tensorqtl']
BIAS_ARMS = ['split', 'split_native', 'trecase', 'trecase_native']
MIRROR_ARMS = ['hapmixQTL split', 'hapmixQTL gibbs', 'hapmixQTL plus_one', 'TReCASE (joint)', 'TReCASE (asSeq joint p)',
               'TReCASE (asSeq final p)']
BETAS = ('0.2', '0.4', '0.8')
TEMPLATE = Path(__file__).with_name('hapmix_vs_trecase_template.html')
OUT = D / 'benchmark_summary_20260929'


def plasmode(root):
    S = json.loads((root / 'summary.json').read_text())
    power = {a: {b: {k: S['ranking'][f'beta{b}'][a]['fdp_matched'][k] for k in ('discoveries', 'false')}
                 | {'power': S['ranking'][f'beta{b}'][a]['fdp_matched']['all']['power']} for b in BETAS} for a in ARMS}
    null = {a: {al: {k: S['null']['beta0.0'][a]['combined']['all'][al][k] for k in ('rate', 'lo', 'hi')}
                for al in ('0.05', '0.01', '0.001')} for a in ARMS}
    bias = {a: {b: {band: {k: v[k] for k in ('mean', 'lo', 'hi')}
                    for band, v in S['recovery'][f'beta{b}'][a]['combined']['bias_count'].items()} for b in BETAS}
            for a in BIAS_ARMS}
    parts = {a: {b: S['trecase_parts'][f'beta{b}'][a]['fdp_power'] for b in BETAS} for a in ('trecase', 'trecase_native')}
    comp = {a: {c: S[f'{a}_components'][c]['all']['0.05']['rate'] for c in ('trec', 'joint', 'ase')}
            for a in ('trecase', 'trecase_native')}
    joint_na = {'trecase': json.loads((root / 'results_trecase' / 'summary.json').read_text())['pooled']['joint_na_share'],
                'trecase_native': json.loads((root / 'native' / 'results_trecase' / 'summary.json').read_text())['pooled']['joint_na_share']}
    return dict(power=power, null=null, bias=bias, parts=parts, components=comp, joint_na=joint_na, bands=list(bias['split']['0.4']))


def mirror():
    M = json.loads(MIRROR.read_text())
    return {N: dict(type1={a: {al: v['type1'][a][al] for al in ('0.05', '0.01')} for a in MIRROR_ARMS},
                    power={f: {a: {k: v['power'][f][a][k] for k in ('matched', 'resample_se')} for a in MIRROR_ARMS}
                           for f in v['power']}) for N, v in M['by_N'].items()}


def referee():
    S = json.loads((REF / 'score' / 'score.json').read_text())
    e = S['sets']['subset']['rankings']['eigenmt']
    pick = lambda d: {k: {x: v[x] for x in ('share', 'share_lo', 'share_hi')} for k, v in d.items()}  # noqa: E731
    tr = S['trecase']['per_input']['native']
    return dict(ks=e['ks'], n_genes=S['sets']['subset']['n_genes'],
                share={a: {k: v['share'] for k, v in e['at_K'][a].items()} for a in ('trecase_native', 'split', 'gibbs', 'tensorqtl')},
                vs_split=pick(e['paired_other']['trecase_native minus split']),
                vs_gibbs=pick(e['paired_other']['trecase_native minus gibbs']),
                vs_tensorqtl=pick(e['paired']['trecase_native']), split_vs_tensorqtl=pick(e['paired']['split']),
                trecase_seconds_median=statistics.median(tr['seconds']), trecase_genes=tr['genes'],
                trecase_failed=len(tr['genes_failed']))


def hapmix_timing():
    """The referee run's timing block: GPU arms (four hapmixQTL weightings and tensorQTL, 1,000 permutations each)
    plus eigenMT, seconds per 100 genes."""
    m = re.search(r'TIMING: (\d+) genes: GPU arms \+ eigenMT (\d+) s', (REF / 'referee_replication.log').read_text())
    return dict(genes=int(m.group(1)), seconds=int(m.group(2)))


def main():
    data = dict(plasmode={gs: plasmode(root) for gs, root in SETS.items()}, mirror=mirror(), referee=referee(),
                hapmix_timing=hapmix_timing(),
                referee_all_genes=json.loads((REF / 'score' / 'score.json').read_text())['sets']['all']['n_genes'])
    OUT.mkdir(exist_ok=True)
    page = TEMPLATE.read_text().replace('/*DATA*/null', json.dumps(data, separators=(',', ':')))
    tmp = OUT / 'hapmix_vs_trecase.tmp.html'
    tmp.write_text(page)
    os.replace(tmp, OUT / 'hapmix_vs_trecase.html')
    print(f'wrote {OUT / "hapmix_vs_trecase.html"} ({len(page):,} bytes)')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""One-page summary with charts of the eQTL method benchmark: simulated-effects power and null calibration (both gene sets),
held-out replication (referee), and the native-count rebuild. Every number is read from the result files below and
embedded as JSON into benchmark_summary_template.html; the page draws its charts from that JSON.

  python3 scripts/benchmark_summary.py      # writes OUT/summary.html
"""
import json
import os
from pathlib import Path

import pandas as pd

D = Path('/mnt/ssd/lalli/brainvar_hapmix_deploy')
SETS = {'deep': D / 'plasmode_meier_20260927', 'lowcov': D / 'plasmode_lowcov_meier_20260927'}
GENES = {'deep': D / 'corrected_null_store_20260925' / 'genes.txt',
         'lowcov': D / 'plasmode_stratum30_100_20260927' / 'gene_set' / 'genes.txt'}
REF = D / 'referee_replication_20260928'
STAGES = {  # the native counts' three builds: simulated-effects summary, native-count directory, reference-share facts
    'before': dict(summary='native_unstranded_20260928/summary_before_stranded.json', counts=D / 'native_counts_20260928',
                   ref_share=(REF / 'facts.json', 'native_reference_share')),
    'stranded': dict(summary='stage_stranded_nowasp_20260928/summary.json', counts=D / 'native_counts_stranded_20260928',
                     ref_share=(D / 'native_counts_stranded_20260928' / 'facts.json', 'reference_share')),
    'wasp': dict(summary='summary.json', counts=D / 'native_counts_wasp_20260928',
                 ref_share=(D / 'native_counts_wasp_20260928' / 'facts.json', 'reference_share'))}
ARMS = ['gibbs', 'split', 'unit', 'plus_one', 'mixqtl', 'mixqtl_permissive', 'tensorqtl', 'rasqual', 'trecase',
        'trecase_native', 'split_native']
TEMPLATE = Path(__file__).with_name('benchmark_summary_template.html')
OUT = D / 'benchmark_summary_20260929'


def plasmode(root):
    S = json.loads((root / 'summary.json').read_text())
    power = {a: {b: S['ranking'][f'beta{b}'][a]['fdp_matched'] for b in ('0.2', '0.4', '0.8')} for a in ARMS}
    null = {a: S['null']['beta0.0'][a]['combined']['all']['0.05'] for a in ARMS}
    return dict(power={a: {b: dict(power=v['all']['power'], called=v['discoveries'], false=v['false'],
                                   non_null=v['all']['non_null']) for b, v in p.items()} for a, p in power.items()},
                null={a: {k: v[k] for k in ('rate', 'lo', 'hi')} for a, v in null.items()})


def referee():
    S = json.loads((REF / 'score' / 'score.json').read_text())
    out = {}
    for name in ('all', 'subset'):
        e = S['sets'][name]['rankings']['eigenmt']
        out[name] = dict(ks=e['ks'], n_genes=S['sets'][name]['n_genes'],
                         tensorqtl={k: v['share'] for k, v in e['at_K']['tensorqtl'].items()},
                         paired={a: {k: {x: v[x] for x in ('share', 'share_lo', 'share_hi')} for k, v in p.items()}
                                 for a, p in e['paired'].items()})
    return out


def stages():
    out = {}
    for st, s in STAGES.items():
        f, key = s['ref_share']
        rs = json.loads(f.read_text())[key]
        frag = {}
        for gs, path in GENES.items():
            g = path.read_text().split()
            a, b = (pd.read_parquet(s['counts'] / f'{k}.parquet').loc[g] for k in ('hap_a', 'hap_b'))
            ab = (a + b).to_numpy()
            frag[gs] = dict(fragments=int(ab.sum()), median_donors=float(pd.Series((ab > 0).sum(1)).median()))
        power = {}
        for gs, root in SETS.items():
            P = json.loads((root / s['summary']).read_text())['ranking']
            power[gs] = {a: {b: P[f'beta{b}'][a]['fdp_matched']['all']['power'] for b in ('0.2', '0.4', '0.8')}
                         for a in ('trecase_native', 'split_native', 'tensorqtl', 'split', 'trecase')}
        out[st] = dict(ref_share=[rs['median_min'], rs['median_max']], fragments=frag, power=power)
    return out


def main():
    data = dict(plasmode={gs: plasmode(root) for gs, root in SETS.items()}, referee=referee(), stages=stages())
    OUT.mkdir(exist_ok=True)
    page = TEMPLATE.read_text().replace('/*DATA*/null', json.dumps(data, separators=(',', ':')))
    tmp = OUT / 'summary.tmp.html'
    tmp.write_text(page)
    os.replace(tmp, OUT / 'summary.html')
    print(f'wrote {OUT / "summary.html"} ({len(page):,} bytes)')


if __name__ == '__main__':
    main()

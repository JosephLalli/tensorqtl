"""RASQUAL on its native per-SNP inputs (04b_rasqual_native_inputs.py), the native datasets of 05b (user request
2026-10-02: run them, at most 16 CPUs at a time).

Each gene's prepared command (IN/<scenario>/repNNN/commands.tsv; stdin = the fSNP file, then the rSNP file) runs with one
thread, JOBS processes at once (SIMULATED_EFFECTS_RASQUAL_NATIVE_JOBS, default 15). Datasets go in replicate order, then
scenario, so replicate 0 of every scenario finishes first; within a dataset the largest (fSNPs + 1) x rSNPs budget starts
first. A gene's RASQUAL rows are checkpointed at IN/<scenario>/repNNN/raw/<gene>.txt (atomic; a gene whose file exists is
not rerun, every skip printed) and its wall seconds appended to IN/timing.tsv. When a dataset's genes are all done,
04.assemble turns its rows into OUT/<scenario>/rasqual_native/nominal_repNNN.parquet (fingerprint of the native dataset
and 'rasqual_native', unit log2): converged rows of tested variants, with the counts of what was excluded (non-converged,
absent tested variants) in OUT/summary.json. A fSNP line reported as an rSNP row stops the run (the fSNP lines sit
outside the -c/-w window, 04b).
"""
import concurrent.futures as cf
import hashlib
import json
import os
import subprocess
import time

import numpy as np
import pandas as pd

import common as C

RR = C.module('04_run_rasqual')
IN = C.NATIVE / 'rasqual_inputs'
OUT = C.NATIVE / 'results_rasqual'
JOBS = int(os.environ.get('SIMULATED_EFFECTS_RASQUAL_NATIVE_JOBS', 15))   # 15 RASQUAL processes and this driver: the user's 16-CPU cap
ENV = {**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}


def run_gene(row, raw):
    """RASQUAL on one prepared command, checkpointed at raw; (rows as text, wall seconds or None if skipped)."""
    if raw.exists():
        return raw.read_text(), None
    text = ''.join(open(p).read() for p in row.stdin.split())
    t0 = time.perf_counter()
    out = subprocess.run(row.args.split(), input=text, stdout=subprocess.PIPE, text=True, check=True, env=ENV).stdout
    secs = time.perf_counter() - t0
    C.write_atomic(raw, lambda fh: fh.write(out), 'w')
    return out, secs


def parse(g, out, fsnp_ids):
    rows = [ln.split('\t') for ln in out.splitlines()]
    bad = [r for r in rows if len(r) != len(C.RASQUAL_FIELDS) or r[0] != g or r[1] == 'SKIPPED']
    if bad or not rows:
        raise SystemExit(f'{g}: {len(rows)} RASQUAL rows, {len(bad)} malformed or SKIPPED, e.g. {bad[:1]}')
    df = pd.DataFrame(rows, columns=C.RASQUAL_FIELDS)
    if df.rs_id.isin(fsnp_ids).any():
        raise SystemExit(f'{g}: a feature-SNP line was scanned as an rSNP')
    return df


def main():
    if hashlib.sha256(open(RR.RASQUAL, 'rb').read()).hexdigest() != RR.RASQUAL_SHA256:
        raise SystemExit(f'{RR.RASQUAL}: sha256 differs from 04.RASQUAL_SHA256')
    S = C.setup(C.load()[0])
    meta = json.loads((C.DATASETS / 'meta.json').read_text())
    budget = pd.Series({g: v['budget'] for g, v in json.loads((IN / 'facts.json').read_text())['genes'].items()})
    runs = sorted(C.runs(meta), key=lambda x: (x[1], x[0]))
    cmds = {(sc, r): pd.read_csv(IN / sc / f'rep{r:03d}' / 'commands.tsv', sep='\t').set_index('gene') for sc, r in runs}
    for (sc, r), c in cmds.items():
        if list(c.index) != S['genes']:
            raise SystemExit(f'{sc} rep {r}: commands.tsv genes differ from the loader genes')
        (IN / sc / f'rep{r:03d}' / 'raw').mkdir(exist_ok=True)
    order = [(sc, r, g) for sc, r in runs for g in budget.loc[S['genes']].sort_values(ascending=False).index]
    print(f'{len(runs)} datasets x {len(S["genes"])} genes, {JOBS} processes; order {[f"{sc} rep {r:03d}" for sc, r in runs]}',
          flush=True)
    summary = json.loads((OUT / 'summary.json').read_text())['per_dataset'] if (OUT / 'summary.json').exists() else {}
    left = {(sc, r): len(S['genes']) for sc, r in runs}
    done = {}
    nds = {(sc, r): C.load_dataset(C.NATIVE_DATASETS, sc, r) for sc, r in runs}
    t_start, n_run, timing = time.time(), 0, open(IN / 'timing.tsv', 'a')
    ex = cf.ThreadPoolExecutor(JOBS)
    try:
        futs = {ex.submit(run_gene, cmds[(sc, r)].loc[g], IN / sc / f'rep{r:03d}' / 'raw' / f'{g}.txt'): (sc, r, g)
                for sc, r, g in order}
        for f in cf.as_completed(futs):
            sc, r, g = futs[f]
            out, secs = f.result()
            d, nd, k = IN / sc / f'rep{r:03d}', nds[(sc, r)], S['genes'].index(g)
            fsnp = {ln.split('\t', 3)[2] for ln in open(d / 'fsnp' / f'{g}.txt')}
            done[(sc, r, g)] = RR.assemble(g, parse(g, out, fsnp), S['tested'][g],
                                           None if nd['is_null'][k] else str(nd['causal_variant'][k]))
            if secs is None:
                print(f'{sc} rep {r:03d} {g}: skipped (raw file present)', flush=True)
            else:
                n_run += 1
                timing.write(f'{sc}\t{r}\t{g}\t{secs:.1f}\t{int(budget[g])}\n')
                timing.flush()
                print(f'{sc} rep {r:03d} {g}: {secs / 60:.1f} min (budget {int(budget[g]):,}); {n_run} genes run in '
                      f'{(time.time() - t_start) / 3600:.2f} h', flush=True)
            left[(sc, r)] -= 1
            if left[(sc, r)]:
                continue
            parts, cnt = [], {}
            for gene in S['genes']:
                df, c = done.pop((sc, r, gene))
                parts.append(df)
                for key, v in c.items():
                    cnt[key] = cnt.get(key, 0) + v
            df = pd.concat(parts, ignore_index=True)
            C.write_parquet(df, OUT / sc / 'rasqual_native' / f'nominal_rep{r:03d}.parquet',
                            C.fingerprint(nd, 'rasqual_native'), 'log2')
            cnt.update(rows=len(df), tests=int(S['n_tested'].sum()), no_fsnp_genes=int(cnt.pop('no_fsnp')))
            summary[f'{sc} rep {r:03d}'] = cnt
            C.write_json(OUT / 'summary.json', dict(per_dataset=summary, jobs=JOBS, rasqual=RR.RASQUAL,
                                                     rasqual_sha256=RR.RASQUAL_SHA256, inputs=str(IN)))
            print(f'{sc} rep {r:03d} assembled: {json.dumps(cnt)}', flush=True)
    finally:
        ex.shutdown(cancel_futures=True)   # a failed gene cancels the queue; running RASQUAL processes still finish
        timing.close()
    s = pd.read_csv(IN / 'timing.tsv', sep='\t', header=None).iloc[:, 3]
    print(f'done: {len(summary)} datasets assembled; seconds per gene median {np.median(s):.0f}; wrote {OUT}', flush=True)


if __name__ == '__main__':
    main()

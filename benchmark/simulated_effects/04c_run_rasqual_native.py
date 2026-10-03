"""RASQUAL on its native per-SNP inputs (04b_rasqual_native_inputs.py) over a variant subset, genes in a stratified random
priority order, time-boxed (user decisions 2026-10-02: run, at most 16 CPUs, about 8 hours).

Why a subset: a scan of every tested variant (median ~4,700 per gene) costs about 6,000 CPU hours for both gene sets
(smoke 0.116 s per feature-SNP x rSNP pair). The benchmark scores each gene by its smallest p over the variants tested,
so every arm can be rescored on the same subset from the per-variant results it already has.

Variant subset per gene: the variant each replicate designated (02's causal_variant, drawn for null genes too, so the
subset does not reveal which genes are null) plus N_RANDOM other tested variants drawn from SeedSequence(SEED,
(SUBSET_KEY, gene index)); IN/subset.tsv. Priority: within each depth band of 06 (BANDS) a random order from
SeedSequence(SEED, (PRIORITY_KEY,)), the bands then taken in turn; IN/priority.tsv. Jobs run gene by gene (every dataset of
a gene before the next gene), JOBS processes at once (SIMULATED_EFFECTS_RASQUAL_NATIVE_JOBS, default 15), one thread each;
no job starts after DEADLINE_H hours. stdin is the gene's fSNP file and its subset's rSNP lines, the option line 04b wrote
with -l set to their count. A job's RASQUAL rows are checkpointed at IN/<scenario>/repNNN/raw_subset/<gene>.txt (atomic; a
present file is not rerun) and its wall seconds appended to IN/timing_subset.tsv.

Output for the genes finished in every dataset: OUT/<scenario>/rasqual_native/nominal_repNNN.parquet (04.assemble on the
subset; fingerprint of the native dataset and 'rasqual_native', unit log2), OUT/summary.json with the excluded counts and
the finished genes. A feature-SNP line reported as an rSNP row stops the run.
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

RR, SC = C.module('04_run_rasqual'), C.module('06_score')
IN = C.NATIVE / 'rasqual_inputs'
OUT = C.NATIVE / 'results_rasqual_subset'
JOBS = int(os.environ.get('SIMULATED_EFFECTS_RASQUAL_NATIVE_JOBS', 15))   # with the other gene set's driver: the user's 16-CPU cap
DEADLINE_H = 8.0               # no job starts later (user, 2026-10-02)
N_RANDOM = 49                  # random tested variants per gene besides the designated ones (user, 2026-10-02)
SUBSET_KEY, PRIORITY_KEY = 9, 14   # spawn keys used by no other script here (05b_native_arms.NATIVE_THIN_KEY lists the rest; 04b: 8)
ENV = {**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}


def variant_subset(S, runs):
    """Per gene, the designated variants of every replicate and N_RANDOM random other tested variants."""
    designated = {g: set() for g in S['genes']}
    for sc, r in runs:
        ds = C.load_dataset(C.DATASETS, sc, r)
        for g, v in zip(S['genes'], ds['causal_variant'].astype(str)):
            designated[g].add(v)
    rows = []
    for k, g in enumerate(S['genes']):
        if not designated[g] <= S['tested'][g]:
            raise SystemExit(f'{g}: a designated variant is not among its tested variants')
        rest = sorted(S['tested'][g] - designated[g])
        rng = np.random.default_rng(np.random.SeedSequence(C.SEED, spawn_key=(SUBSET_KEY, k)))
        pick = [rest[i] for i in rng.choice(len(rest), N_RANDOM, replace=False)]
        rows += [dict(gene=g, variant_id=v, designated=True) for v in sorted(designated[g])]
        rows += [dict(gene=g, variant_id=v, designated=False) for v in pick]
    return pd.DataFrame(rows)


def priority(genes):
    """Genes in a random order within each depth band of 06, the bands taken in turn."""
    gd = pd.read_csv(C.GENE_DESIGN, sep='\t').set_index('gene').loc[genes]
    bands = SC.BANDS[1:]
    band = pd.cut(gd.median_allele_resolved_reads, [b[1] for b in bands] + [np.inf], right=False,
                  labels=[b[0] for b in bands]).astype(str)
    rng = np.random.default_rng(np.random.SeedSequence(C.SEED, spawn_key=(PRIORITY_KEY,)))
    queues = [list(rng.permutation(band.index[band == b].to_numpy())) for b, _, _ in bands]
    order = []
    while any(queues):
        for q in queues:
            if q:
                order.append(q.pop(0))
    return pd.DataFrame(dict(gene=order, band=band.loc[order].values))


def run_job(row, text, raw, deadline):
    """RASQUAL on one gene of one dataset, checkpointed at raw; (rows as text or None if past the deadline, seconds)."""
    if raw.exists():
        return raw.read_text(), None
    if time.time() > deadline:
        return None, None
    args = row.args.split()
    args[args.index('-l') + 1] = str(text.count('\n'))
    t0 = time.perf_counter()
    out = subprocess.run(args, input=text, stdout=subprocess.PIPE, text=True, check=True, env=ENV).stdout
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
    t_start = time.time()
    deadline = t_start + DEADLINE_H * 3600
    if hashlib.sha256(open(RR.RASQUAL, 'rb').read()).hexdigest() != RR.RASQUAL_SHA256:
        raise SystemExit(f'{RR.RASQUAL}: sha256 differs from 04.RASQUAL_SHA256')
    S = C.setup(C.load()[0])
    meta = json.loads((C.DATASETS / 'meta.json').read_text())
    runs = sorted(C.runs(meta), key=lambda x: (x[1], x[0]))
    sub = variant_subset(S, runs)
    pri = priority(S['genes'])
    C.write_atomic(IN / 'subset.tsv', lambda fh: sub.to_csv(fh, sep='\t', index=False), 'w')
    C.write_atomic(IN / 'priority.tsv', lambda fh: pri.to_csv(fh, sep='\t', index_label='rank'), 'w')
    keep = sub.groupby('gene').variant_id.apply(set)
    rsnp = {g: ''.join(ln for ln in open(IN / 'rsnp' / f'{g}.txt') if ln.split('\t', 3)[2] in keep[g]) for g in S['genes']}
    if any(rsnp[g].count('\n') != len(keep[g]) for g in S['genes']):
        raise SystemExit('a subset variant has no rSNP line')
    cmds = {(sc, r): pd.read_csv(IN / sc / f'rep{r:03d}' / 'commands.tsv', sep='\t').set_index('gene') for sc, r in runs}
    for sc, r in runs:
        (IN / sc / f'rep{r:03d}' / 'raw_subset').mkdir(exist_ok=True)
    print(f'{len(runs)} datasets x {len(S["genes"])} genes, {JOBS} processes, deadline {DEADLINE_H} h; subset {len(sub):,} variants '
          f'({int(sub.designated.sum())} designated); priority by band {pri.band.value_counts().to_dict()}', flush=True)
    timing = open(IN / 'timing_subset.tsv', 'a')
    results, n_run = {}, 0
    ex = cf.ThreadPoolExecutor(JOBS)
    try:
        futs = {}
        for g in pri.gene:
            for sc, r in runs:
                d = IN / sc / f'rep{r:03d}'
                text = open(d / 'fsnp' / f'{g}.txt').read() + rsnp[g]
                futs[ex.submit(run_job, cmds[(sc, r)].loc[g], text, d / 'raw_subset' / f'{g}.txt', deadline)] = (sc, r, g)
        for f in cf.as_completed(futs):
            sc, r, g = futs[f]
            out, secs = f.result()
            if out is None:
                continue
            fsnp = {ln.split('\t', 3)[2] for ln in open(IN / sc / f'rep{r:03d}' / 'fsnp' / f'{g}.txt')}
            results[(sc, r, g)] = parse(g, out, fsnp)
            if secs is not None:
                n_run += 1
                timing.write(f'{sc}\t{r}\t{g}\t{secs:.1f}\n')
                timing.flush()
                print(f'{sc} rep {r:03d} {g}: {secs:.0f} s; {n_run} jobs run in {(time.time() - t_start) / 3600:.2f} h', flush=True)
    finally:
        ex.shutdown(cancel_futures=True)   # a failed job cancels the queue; running RASQUAL processes still finish
        timing.close()
    done = [g for g in pri.gene if all((sc, r, g) in results for sc, r in runs)]
    print(f'genes finished in every dataset: {len(done)} of {len(pri)} at {(time.time() - t_start) / 3600:.2f} h; by band '
          f'{pri.set_index("gene").loc[done].band.value_counts().to_dict()}', flush=True)
    summary = {}
    for sc, r in runs:
        nd = C.load_dataset(C.NATIVE_DATASETS, sc, r)
        parts, cnt = [], {}
        for k, g in enumerate(S['genes']):
            if g not in done:
                continue
            df, c = RR.assemble(g, results[(sc, r, g)], keep[g], None if nd['is_null'][k] else str(nd['causal_variant'][k]))
            parts.append(df)
            for key, v in c.items():
                cnt[key] = cnt.get(key, 0) + v
        df = pd.concat(parts, ignore_index=True)
        C.write_parquet(df, OUT / sc / 'rasqual_native' / f'nominal_rep{r:03d}.parquet', C.fingerprint(nd, 'rasqual_native'), 'log2')
        cnt.update(rows=len(df), no_fsnp_genes=int(cnt.pop('no_fsnp')))
        summary[f'{sc} rep {r:03d}'] = cnt
        print(f'{sc} rep {r:03d}: {json.dumps(cnt)}', flush=True)
    C.write_json(OUT / 'summary.json', dict(per_dataset=summary, genes=done, n_random=N_RANDOM, deadline_h=DEADLINE_H, jobs=JOBS,
                                             rasqual=RR.RASQUAL, rasqual_sha256=RR.RASQUAL_SHA256, inputs=str(IN)))
    print(f'wrote {OUT}', flush=True)


if __name__ == '__main__':
    main()

"""Acceptance test of this pipeline against the committed run of its gene set (common.GENE_SETS
'committed': the 100-gene run of 2026-09-26 by default, the 30-100-read run of 2026-09-27 with
PLASMODE_GENE_SET=stratum30_100) (README).

Runs the pipeline into common.ROOT and checks:
 (1) every dataset array bit for bit (datasets/*.npz, all keys; the same number of files);
 (2) map_nominal outputs of the four hapmixQTL arms and both mixQTL arms bit for bit (dtype and
     bytes) on the columns 06_score.py reads, on the same (phenotype_id, variant_id) row set;
 (3) map_cis outputs: identical leads and num_var, the same finite / NaN pattern, and pval_beta and
     pval_perm within CIS_RTOL relative (the maximum difference is reported);
 (4) summary.json: the union of the two files' leaves walked; every numeric leaf of the new file
     equals the old one within RTOL relative, every string except a path exactly, leaves only in
     the old file allowed only in the DROPPED families (06 no longer writes them and 08 reads none);
     on the committed RASQUAL and TReCASE results copied into ROOT (their summary.json derived from
     the old run's log for RASQUAL). The new 04 and 05 are checked by running ONE dataset
     (JOINT_CHECK) into ROOT/joint_check and comparing its nominal file bit for bit with the
     committed one and its summary.json counts (the keys 08 reads) with the committed run's
     (JOINT_CHECK; for a gene set without one only the staged results are scored), only when run as
     `99_acceptance.py joint` (below);
 (5) ladder.json as (4) (skipped for a gene set without a ladder, whose committed run has none);
 (6) the report's numbers: every numeric token of the old and new pages (image data, style, path
     tokens, dates and commit ids removed) compared as multisets.
Prints PASS, FAIL or SKIP per item with the maximum difference, then exits non-zero on any FAIL.

The one-dataset joint rerun of (4) runs only with the argument `joint`. It costs hours of RASQUAL
and TReCASE (97 and 136 min in the recorded pass, and a RASQUAL run can take more than 6 h), and its
stamp includes common.py and 02's stamp, which change far more often than anything 04 and 05 read
from them (a gene-set entry or a comment reruns it). Without the argument each arm prints a SKIP
naming its last recorded pass (the PASS line in JOINT_PASS) and whether 04 / 05, their imports and
common.py still have the sha256 that run's header records; a SKIP is not a failure. Rerun with
`joint` when 04, 05, run_trecase.R or what they read from common.py or 02's datasets changes.

Skipping: a step is skipped when its output exists and its stamp matches, the stamp being the
sha256 of the script, common.py and the other scripts it imports, plus the stamps of the steps
whose outputs it reads (STEPS); what invalidated a stamp is printed. The joint runs are stamped the
same way in ROOT/joint_check/stamp_<arm>.json, written when a run starts: a run under the same
stamp resumes its per-gene checkpoints, a changed stamp wipes them first. A failed pipeline step
ends the process at once (os._exit, after the FAIL line): RASQUAL and R processes already running
finish on their own and their checkpoints are reused by the next run under the same stamp.
"""
import concurrent.futures as cf
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

import common as C

OLD = C.D / C.GS['committed']           # the committed run of the previous code (scripts/plasmode/ before f0c0b07)
OLD_JOINT = {'rasqual': OLD / 'results_rasqual', 'trecase': OLD / 'results_trecase_asseq'}
JOINT_CHECK = {'corrected_null_store_20260925': ('beta0.8', 0),   # the one dataset 04 and 05 are rerun on
               'stratum30_100': None}[C.GENE_SET]                 # None: the committed joint results are staged and scored only (task 2026-09-27)
JOINT_PASS = C.ROOT / 'acceptance_df76f3b.log'   # the last recorded pass of the one-dataset joint rerun (default set, 2026-09-27 14:42-16:59)
JOINT_JOBS = {'rasqual': 5, 'trecase': 4}   # alongside the GPU steps: 5 RASQUAL + 4 x (Rscript + R) + this process + one pipeline step = 15 live processes, under the host's cap of 16
GPU = '1'                                # CUDA_VISIBLE_DEVICES for map_nominal / map_cis (shared host, 2026-09-27)
RTOL = 1e-9
CIS_RTOL = 1e-6
STEPS = [   # (script, final output, other scripts it imports, steps whose outputs it reads), in run order; 08 reads every earlier step
    ('01_check_inputs.py', C.CHECKS / 'check_generator.json', ['02_make_datasets.py', '03_run_arms.py'], []),
    ('02_make_datasets.py', C.DATASETS / 'meta.json', [], []),
    ('03_run_arms.py', C.RESULTS / 'run_arms_facts.json', [], ['02_make_datasets.py']),
    ('06_score.py', C.SUMMARY, [], ['02_make_datasets.py', '03_run_arms.py'])] + (
    [('07_mixqtl_ladder.py', C.LADDER / 'ladder.json', ['06_score.py'], ['02_make_datasets.py', '03_run_arms.py'])] if C.LADDER else [])
STEPS.append(('08_report.py', C.REPORT / 'plasmode_report.html', [], [s[0] for s in STEPS]))
JOINT_STEPS = {'rasqual': ('04_run_rasqual.py', []), 'trecase': ('05_run_trecase.py', ['run_trecase.R'])}   # both read 02's datasets
STAMPS = C.ROOT / 'acceptance_stamps.json'
DROPPED = {   # leaf families of the committed files that 06 / 07 no longer write; neither report reads them (README, Acceptance test)
    'summary.json': (r'^/anchor_passed$', r'^/cannot_answer\[\d+\]$', r'^/units$', r'^/gene_set$', r'^/anchor/[^/]+/[^/]+/[^/]+/stored_without_one_df$',
                     r'^/detection/[^/]+/[^/]+/[^/]+/nonfinite_p$', r'^/gene_level/[^/]+/[^/]+/(datasets|nonfinite_p_for_power|p_for_power)$',
                     r'^/gene_level/[^/]+/[^/]+/null_rate_pval_perm/', r'^/null/.*/dataset_(lo|hi)$', r'^/precision/.*/nonnull_excluded/',
                     r'^/ranking/[^/]+/[^/]+/auc/[^/]+/datasets$'),
    'ladder.json': (r'^/smoke$', r'^/ladder/[^/]+/[^/]+/auc/[^/]+/datasets$', r'^/ladder/[^/]+/[^/]+/own_set_excluded/')}
JOINT_KEYS = {   # per-dataset counts of the joint summaries that 08_report.joint_facts reads (directly or pooled), 'a/b' a nested key
    'rasqual': ('rows', 'tests', 'nonconv', 'absent', 'chisq_le0', 'no_fsnp', 'het', 'as00', 'causal_nonconv', 'causal_absent', 'causal_nonnull'),
    'trecase': ('rows', 'tested_constant_dosage', 'informative_zeroed_not_allelic_kept', 'as_records_admitted', 'trec_na', 'ase_na_few_het',
                'joint_na', 'joint_na_by_trace/joint_theta', 'joint_na_by_trace/ase', 'joint_na_by_trace/trec_linear_dosage',
                'joint_na_by_trace/theta_fail_abs_gradient_max', 'final_joint', 'final_trec', 'final_na', 'final_df_not_1', 'causal_not_run',
                'causal_nonnull', 'causal_joint_na', 'causal_final_joint', 'causal_final_trec', 'causal_final_na')}
SKIPPED = []
if OLD == C.ROOT:
    raise SystemExit(f'common.ROOT is the committed run {OLD}')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stamp(script, imports, reads, stamps):
    """A step's provenance: the sha256 of its files and the stamps of the steps it reads from, and their digest."""
    d = dict(files={f: sha(C.HERE / f) for f in (script, 'common.py', *imports)}, reads={u: stamps[u]['stamp'] for u in reads})
    d['stamp'] = hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()
    return d


def invalidated(old, new):
    """Why a stored stamp no longer holds: '' when it does, else 'no stamp' or the files and steps that changed."""
    if not isinstance(old, dict) or set(old) != {'files', 'reads', 'stamp'}:
        return 'no stamp'
    changed = ([f for f, s in new['files'].items() if old['files'].get(f) != s]
               + [f'{u} (upstream)' for u, s in new['reads'].items() if old['reads'].get(u) != s])
    return 'changed: ' + ', '.join(changed) if changed else ''


def run_step(script, output, imports, reads, stamps):
    """Run a numbered script into ROOT unless its output exists under the current stamp; log to ROOT/<step>.log."""
    cur = stamp(script, imports, reads, stamps)
    why = invalidated(stamps.get(script), cur)
    if output.exists() and not why:
        print(f'skip {script}: {output} exists under the current stamp', flush=True)
        return
    print(f'run {script} ({why if why else "output missing"})', flush=True)
    t0 = time.perf_counter()
    with open(C.ROOT / f'{script[:-3]}.log', 'w') as fh:
        rc = subprocess.run([sys.executable, str(C.HERE / script)], stdout=fh, stderr=subprocess.STDOUT,
                            env={**os.environ, 'CUDA_VISIBLE_DEVICES': GPU}, cwd=str(C.HERE)).returncode
    if rc != 0:
        raise SystemExit(f'{script} exited {rc}; see {C.ROOT / (script[:-3] + ".log")}')
    stamps[script] = cur
    C.write_json(STAMPS, stamps)
    print(f'ran {script}: {(time.perf_counter() - t0) / 60:.1f} min', flush=True)


def stage_joint_results():
    """The committed RASQUAL and TReCASE results into ROOT (copied once), with a summary.json each."""
    for arm, src in OLD_JOINT.items():
        for f in sorted(src.glob(f'beta*/{arm}/nominal_rep*.parquet')):
            dst = C.JOINT[arm] / f.parent.parent.name / arm / f.name
            if not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                C.write_atomic(dst, lambda fh, f=f: fh.write(f.read_bytes()))
    src = OLD_JOINT['trecase'] / 'summary.json'
    C.write_atomic(C.JOINT['trecase'] / 'summary.json', lambda fh: fh.write(src.read_bytes()))
    log = (OLD_JOINT['rasqual'] / 'run_rasqual.log').read_text()   # the old run wrote its counts to the log only
    per = {}
    for k, *m in re.findall(r'(?m)^(beta\S+ rep \d+): ([\d,]+) rows written; excluded non-converged (\d+), pseudo-fSNP '
                            r'rows (\d+); tested variants with no RASQUAL row (\d+); chisq <= 0 \(slope_se NaN\) (\d+); '
                            r'genes where RASQUAL did not admit the pseudo fSNP (\d+); non-null causal variants without a '
                            r'row: non-converged (\d+), absent (\d+) \(of (\d+)\)', log):
        v = [int(x.replace(',', '')) for x in m]
        per[k] = dict(zip(('rows', 'nonconv', 'pseudo', 'absent', 'chisq_le0', 'no_fsnp', 'causal_nonconv', 'causal_absent',
                           'causal_nonnull'), v))
        per[k]['tests'] = per[k]['rows'] + per[k]['nonconv'] + per[k]['absent']
    for k, h, inf, z in re.findall(r'(?m)^(beta\S+ rep \d+): \d+ covariates; pseudo fSNP (\d+) het of (\d+) informative '
                                   r'pairs.*AS 0,0 (\d+)$', log):
        per[k].update(het=int(h), informative=int(inf), as00=int(z))
    if len(per) != 10 or not all('het' in v for v in per.values()):
        raise SystemExit(f'{OLD_JOINT["rasqual"]}/run_rasqual.log: {len(per)} dataset lines parsed')
    pooled = {k: sum(v[k] for v in per.values()) for k in ('rows', 'tests', 'nonconv', 'absent', 'chisq_le0', 'no_fsnp',
                                                             'causal_nonconv', 'causal_absent', 'causal_nonnull')}
    C.write_json(C.JOINT['rasqual'] / 'summary.json', dict(per_dataset=per, pooled=pooled,
                                                            source=str(OLD_JOINT['rasqual'] / 'run_rasqual.log')))
    print(f'staged the committed joint results into {C.ROOT}; RASQUAL pooled {pooled}', flush=True)


def verdict(item, ok, detail):
    print(f'({item}) {"PASS" if ok else "FAIL"}: {detail}', flush=True)
    return ok


def skip(item, detail):
    SKIPPED.append(f'({item})')
    print(f'({item}) SKIP: {detail}', flush=True)


def check_datasets():
    old_files, n, bad = sorted(OLD.glob('datasets/beta*/rep*.npz')), 0, []
    n_new = len(list(C.DATASETS.glob('beta*/rep*.npz')))
    for f in old_files:
        a, b = np.load(f), np.load(C.DATASETS / f.parent.name / f.name)
        if set(a.files) != set(b.files):
            bad.append(f'{f.name}: keys differ')
        for k in a.files:
            n += 1
            if a[k].dtype != b[k].dtype or a[k].shape != b[k].shape or a[k].tobytes() != b[k].tobytes():
                bad.append(f'{f.parent.name}/{f.name}:{k}')
    ok = not bad and n > 0 and n_new == len(old_files)
    return verdict(1, ok, f'{n} arrays of {len(old_files)} datasets bit for bit ({n_new} dataset files here); differing {bad[:5]}')


def same_table(old, new, cols):
    """Columns of two parquet files on the same (phenotype_id, variant_id) row set: (rows, [columns not bit-identical
    in dtype and bytes]); a differing row set is reported as 'rows'."""
    a = pd.read_parquet(old, columns=cols).set_index(['phenotype_id', 'variant_id'])
    b = pd.read_parquet(new, columns=cols).set_index(['phenotype_id', 'variant_id'])
    if not (a.index.is_unique and b.index.is_unique and len(a) == len(b) and b.index.isin(a.index).all()):
        return len(a), ['rows']
    b = b.reindex(a.index)
    diff = []
    for c in cols[2:]:
        x, y = a[c].to_numpy(), b[c].to_numpy()
        same = (x == y).all() if x.dtype == object else x.tobytes() == y.tobytes()   # strings: value equality
        if x.dtype != y.dtype or not same:
            diff.append(c)
    return len(a), diff


def check_nominal():
    meta = json.loads((C.DATASETS / 'meta.json').read_text())
    bad, rows = [], 0
    for sc, r in C.runs(meta):
        for arm in C.ARMS:
            cols = C.COLS + (['method'] if arm in C.MIXQTL_ARMS else C.DOF_COLS)
            f = f'{sc}/{arm}/nominal_rep{r:03d}.parquet'
            n, diff = same_table(OLD / 'results' / f, C.RESULTS / f, cols)
            rows += n
            bad += [f'{f}:{c}' for c in diff]
    return verdict(2, not bad and rows > 0, f'{rows:,} rows x {len(C.ARMS)} arms on the scored columns bit for bit (dtype, bytes and '
                                            f'row set); differing {bad[:5]}')


def check_cis():
    meta = json.loads((C.DATASETS / 'meta.json').read_text())
    worst, lead_bad, nan_bad, n = {'pval_beta': 0.0, 'pval_perm': 0.0}, 0, 0, 0
    for sc, r in C.runs(meta):
        for arm in C.HAPMIX_ARMS:
            f = f'{sc}/{arm}/cis_rep{r:03d}.parquet'
            a = pd.read_parquet(OLD / 'results' / f).set_index('phenotype_id')
            b = pd.read_parquet(C.RESULTS / f).set_index('phenotype_id').reindex(a.index)
            lead_bad += int((a.variant_id != b.variant_id).sum() + (a.num_var != b.num_var).sum())
            n += len(a)
            for c in worst:
                x, y = a[c].to_numpy(float), b[c].to_numpy(float)
                fin = np.isfinite(x) & np.isfinite(y)
                nan_bad += int((np.isfinite(x) != np.isfinite(y)).sum())
                if fin.any():
                    worst[c] = max(worst[c], float(np.max(np.abs(x[fin] - y[fin]) / np.where(x[fin] == 0, 1.0, np.abs(x[fin])))))
    ok = n > 0 and lead_bad == 0 and nan_bad == 0 and max(worst.values()) <= CIS_RTOL
    return verdict(3, ok, f'{n} gene-level rows over {len(C.runs(meta))} datasets x {len(C.HAPMIX_ARMS)} arms: leads or num_var differing '
                          f'{lead_bad}, finite / NaN pattern differing {nan_bad}; max relative difference pval_beta {worst["pval_beta"]:.1e}, '
                          f'pval_perm {worst["pval_perm"]:.1e} (rule {CIS_RTOL:g} on both)')


def leaves(x, path=''):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from leaves(v, f'{path}/{k}')
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from leaves(v, f'{path}[{i}]')
    else:
        yield path, x


def is_path(v):
    return isinstance(v, str) and (v.startswith('/') or str(C.D) in v)


def compare_json(new, old, dropped=()):
    """Both JSON trees walked leaf by leaf: (leaves compared, max relative difference, [differing leaves], [path-string
    leaves left uncompared], {accepted old-only family: count}); an old-only leaf outside `dropped` and every new-only leaf
    that is not a path string (a record's provenance, e.g. 03's mixqtl_permutation source) are differences; numbers within
    RTOL relative, everything else exactly."""
    pn, po = dict(leaves(new)), dict(leaves(old))
    n, worst, bad, paths, accepted = 0, 0.0, [], [], Counter()
    for p, v in pn.items():
        if is_path(v):
            paths.append(p)
        elif p not in po:
            bad.append(f'{p}: only in the new file')
        elif isinstance(v, bool) or not isinstance(v, (int, float)) or isinstance(po[p], bool) or not isinstance(po[p], (int, float)):
            n += 1
            if v != po[p]:
                bad.append(f'{p}: {v!r} vs {po[p]!r}')
        else:
            n += 1
            rel = abs(v - po[p]) / max(abs(po[p]), 1e-300)
            worst = max(worst, rel)
            if rel > RTOL:
                bad.append(f'{p}: {v} vs {po[p]}')
    for p in po:
        if p not in pn:
            fam = [d for d in dropped if re.match(d, p)]
            if fam:
                accepted[fam[0]] += 1
            else:
                bad.append(f'{p}: only in the old file')
    return n, worst, bad, paths, accepted


def check_json(item, new_path, old_path):
    new, old = json.loads(new_path.read_text()), json.loads(old_path.read_text())
    n, worst, bad, paths, accepted = compare_json(new, old, DROPPED[new_path.name])
    return verdict(item, not bad and n > 0,
                   f'{n:,} leaves of {new_path.name} compared over the union of both files (top-level {sorted(new)}) within {RTOL:g} '
                   f'relative; max relative difference {worst:.1e}; differing {bad[:5]}; {len(paths)} path strings not compared '
                   f'({sorted(set(p.split("/")[1] for p in paths))}); old-only leaves in the accepted dropped families '
                   f'{sum(accepted.values()):,} ({len(accepted)} of {len(DROPPED[new_path.name])} families seen)')


def page_numbers(path):
    """The multiset of numeric tokens on a page: images, style, path-like tokens (a letter, underscore or dot before a
    slash), dates and commit ids removed, the remaining slashes read as separators; word-bounded tokens such as
    0.0693, 1,375, 5.5e-06, -0.25."""
    t = path.read_text()
    t = re.sub(r'<img[^>]*>', ' ', t)
    t = re.sub(r'<style>.*?</style>', ' ', t, flags=re.S)
    t = re.sub(r'<[^>]+>', ' ', t)
    t = re.sub(r'&[a-z]+;|&#\d+;', ' ', t)
    t = re.sub(r'\S*[A-Za-z_.]/\S*', ' ', t)             # paths, main.c:631, floor(n/10)
    t = t.replace('/', ' ')                              # 0.05 / 0.01, 145/150
    t = re.sub(r'\b\d{4}-\d{2}-\d{2}\b', ' ', t)         # dates
    t = re.sub(r'\b(?=[0-9a-f]*[a-f])[0-9a-f]{7,40}\b', ' ', t)   # commit ids: hex with at least one letter
    return Counter(re.findall(r'(?<![\w.])-?\d+(?:,\d{3})*(?:\.\d+)?(?:e[+-]?\d+)?(?![\w.])', t))


def check_report():
    old, new = page_numbers(OLD / 'report' / 'plasmode_report.html'), page_numbers(C.REPORT / 'plasmode_report.html')
    diff = [(tok, old[tok], new[tok]) for tok in sorted(set(old) | set(new)) if old[tok] != new[tok]]
    return verdict(6, not diff and sum(new.values()) > 0,
                   f'{sum(new.values()):,} numeric tokens ({len(new)} distinct) on the new page, {sum(old.values()):,} ({len(old)}) on the old, '
                   f'compared as multisets; tokens whose counts differ (token, old, new) {diff[:20]}')


def joint_file(arm):
    return C.ROOT / 'joint_check' / f'results_{arm}' / JOINT_CHECK[0] / arm / f'nominal_rep{JOINT_CHECK[1]:03d}.parquet'


def pick(d, keys):
    """{key: d[key]} for the JOINT_KEYS of one arm, 'a/b' read as d['a']['b']."""
    out = {}
    for k in keys:
        a, *b = k.split('/')
        if b:
            out.setdefault(a, {})[b[0]] = d[a][b[0]]
        else:
            out[a] = d[a]
    return out


def run_joint(arm, cur, S):
    """04 or 05 on the one dataset JOINT_CHECK into ROOT/joint_check under stamp `cur`: skipped when its nominal file
    exists under that stamp, resumed from its per-gene checkpoints under that stamp, otherwise from scratch; wall minutes.
    S is loaded once by the main thread: the loader redirects stdout while it runs, and two threads doing that at once
    leave the process's stdout on a dead buffer for the rest of the run."""
    path = C.ROOT / 'joint_check' / f'stamp_{arm}.json'
    why = invalidated(json.loads(path.read_text()) if path.exists() else None, cur)
    out = joint_file(arm).parents[2]
    if why:
        for d in [out] + ([out.with_name('trecase_work')] if arm == 'trecase' else []):
            if d.exists():
                shutil.rmtree(d)
        print(f'{arm} on {JOINT_CHECK}: from scratch ({why})', flush=True)
    elif joint_file(arm).exists():
        print(f'skip {arm} on {JOINT_CHECK}: {joint_file(arm)} exists under the current stamp', flush=True)
        return 0.0
    else:
        print(f'{arm} on {JOINT_CHECK}: resuming the checkpoints of an unfinished run under the current stamp', flush=True)
    C.write_json(path, cur)
    t0 = time.perf_counter()
    if arm == 'rasqual':
        C.module('04_run_rasqual').run(S, C.DATASETS, out, [JOINT_CHECK], JOINT_JOBS[arm])
    else:
        C.module('05_run_trecase').run(S, C.DATASETS, out, out.with_name('trecase_work'), [JOINT_CHECK], JOINT_JOBS[arm])
    return (time.perf_counter() - t0) / 60


def check_joint(arm):
    """The one dataset's nominal file bit for bit against the committed one, and its summary counts (JOINT_KEYS)
    against the committed run's for that dataset."""
    old = OLD_JOINT[arm] / JOINT_CHECK[0] / arm / f'nominal_rep{JOINT_CHECK[1]:03d}.parquet'
    cols = list(pd.read_parquet(old).columns)
    n, diff = same_table(old, joint_file(arm), cols)
    key = f'{JOINT_CHECK[0]} rep {JOINT_CHECK[1]:03d}'
    ref = (C.JOINT['rasqual'] if arm == 'rasqual' else OLD_JOINT['trecase']) / 'summary.json'
    new = json.loads((joint_file(arm).parents[2] / 'summary.json').read_text())['per_dataset'][key]
    old_counts = json.loads(ref.read_text())['per_dataset'][key]
    m, worst, bad, _, _ = compare_json(pick(new, JOINT_KEYS[arm]), pick(old_counts, JOINT_KEYS[arm]))
    return verdict(f'4, {arm}', not diff and not bad and n > 0 and m > 0,
                   f'{key}: {n:,} rows x {len(cols)} columns bit for bit against {old.parent.parent.parent.name}; differing {diff}; '
                   f'{m} summary counts against {ref.parent.name} within {RTOL:g}, max relative difference {worst:.1e}, differing {bad[:5]}')


def recorded_pass(arm):
    """The SKIP text of one joint arm without `joint`: JOINT_PASS's PASS line for it, and its files whose sha256 differs
    from that run's header."""
    log = JOINT_PASS.read_text()
    line = re.search(rf'(?m)^\(4, {arm}\) PASS: .*$', log)
    if line is None:
        raise SystemExit(f'{JOINT_PASS}: no (4, {arm}) PASS line')
    then = dict((name, h) for h, name in re.findall(r'(?m)^([0-9a-f]{64})  \S*/scripts/plasmode2?/(\S+)$', log))
    script, imports = JOINT_STEPS[arm]
    changed = [f for f in (script, *imports, 'common.py', '02_make_datasets.py') if then.get(f) != sha(C.HERE / f)]
    return (f'the one-dataset rerun runs only with the argument joint; last recorded pass {JOINT_PASS}: "{line.group(0)}"; '
            f'sha256 differing from that run: {changed or "none"}')


def main():
    if sys.argv[1:] not in ([], ['joint']):
        raise SystemExit(f'usage: {sys.argv[0]} [joint]')
    rerun = sys.argv[1:] == ['joint']
    t0 = time.perf_counter()
    C.ROOT.mkdir(parents=True, exist_ok=True)
    print('\n'.join(C.versions()), flush=True)
    stage_joint_results()
    stamps = json.loads(STAMPS.read_text()) if STAMPS.exists() else {}
    steps = {s[0]: s for s in STEPS}
    step = lambda name: run_step(*steps[name], stamps)   # noqa: E731
    ex, joint = cf.ThreadPoolExecutor(2), {}
    try:
        step('01_check_inputs.py')
        step('02_make_datasets.py')
        ok = check_datasets()
        if ok and JOINT_CHECK and rerun:   # the joint one-dataset runs alongside the GPU steps, on the datasets just verified, within the process cap
            S = C.setup(C.load()[0])
            joint = {arm: ex.submit(run_joint, arm, stamp(script, imports, ['02_make_datasets.py'], stamps), S)
                     for arm, (script, imports) in JOINT_STEPS.items()}
        step('03_run_arms.py')
        ok &= check_nominal()
        ok &= check_cis()
        step('06_score.py')
        ok &= check_json(4, C.SUMMARY, OLD / 'summary.json')
        if C.LADDER:
            step('07_mixqtl_ladder.py')
            ok &= check_json(5, C.LADDER / 'ladder.json', OLD / 'ladder' / 'ladder.json')
        else:
            skip(5, f'gene set {C.GENE_SET} has no ladder (common.LADDER is None; 07 not run) and {OLD} has none')
        step('08_report.py')
        ok &= check_report()
        for arm in JOINT_STEPS:
            if not JOINT_CHECK:
                skip(f'4, {arm}', f'no one-dataset rerun for gene set {C.GENE_SET} (JOINT_CHECK is None); its committed '
                                  f'results are staged and scored in (4)')
            elif not rerun:
                skip(f'4, {arm}', recorded_pass(arm))
            elif arm in joint:
                print(f'{arm} on {JOINT_CHECK}: {joint[arm].result():.1f} min', flush=True)
                ok &= check_joint(arm)
            else:
                ok &= verdict(f'4, {arm}', False, 'not run: the datasets differ from the committed ones')
    except SystemExit as e:
        print(f'ACCEPTANCE FAIL: {e} ({(time.perf_counter() - t0) / 60:.1f} min)', flush=True)
        sys.stdout.flush()
        os._exit(1)   # not sys.exit: the executor's exit hook would wait for the running joint arms (hours)
    finally:
        ex.shutdown(wait=False, cancel_futures=True)
    print(f'ACCEPTANCE {"PASS" if ok else "FAIL"}: items 1-6{" except " + ", ".join(SKIPPED) if SKIPPED else ""} for gene set '
          f'{C.GENE_SET} in {(time.perf_counter() - t0) / 60:.1f} min', flush=True)
    if not ok:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

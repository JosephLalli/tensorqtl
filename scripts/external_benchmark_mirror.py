"""Mirror benchmark: tests/ase_external_benchmark.py at N = 200 and N = 92, hapmixQTL in current default
mode, and the real asSeq::trecase on the same simulated data.

The harness simulates from the RASQUAL / TReCASE generative model (negative-binomial totals,
beta-binomial allelic counts; its docstring). Design of the 2026-09-23 record
(external_benchmark_fitted_defaults_20260923): mu 200, NB dispersion 0.2, BB overdispersion 0.01,
allele-specific fraction 0.25, 500 replicates, null at seed0 1000 and every fold at seed0 5000, the
harness's own reseeding (data from RandomState(seed0 + r), every arm from RandomState(seed0 + 900000 + r)),
so at N = 200 the comparator arms see the identical data and must reproduce that record exactly.

Steps, each skipping finished units:
  arms       every harness arm, one TSV per condition (OUT/arms/)
  trecase    asSeq::trecase through external_benchmark_mirror_trecase.R, one TSV per CHUNK replicates
             (OUT/trecase/); the inputs are regenerated from the same seeds (OUT/trecase/inputs/)
  summarize  OUT/summary.json and OUT/per_replicate.tsv.gz
Usage: external_benchmark_mirror.py arms|trecase|summarize
"""
import concurrent.futures as cf
import json
import os
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'tests'))
import ase_external_benchmark as B  # noqa: E402

OUT = Path('/mnt/ssd/lalli/brainvar_hapmix_deploy/external_benchmark_current_20260928')
PRIOR = Path('/mnt/ssd/lalli/brainvar_hapmix_deploy/external_benchmark_fitted_defaults_20260923/'
             'null_and_power_500reps.json')                        # the 2026-09-23 record
REPS = 500                                                           # task; the 2026-09-23 design
NS = (200, 92)                                                       # 2026-09-23 design; BrainVar cohort size
KAPPAS = (1.0, 1.05, 1.10, 1.20)                                     # null + the 2026-09-23 folds
SEED0 = {1.0: 1000}                                                  # harness main(): null seed0
ALT_SEED0 = 5000                                                     # harness main(): every fold's seed0
SIM = dict(mu=200.0, phi=0.2, rho=0.01, as_frac=0.25)                # harness defaults = 2026-09-23 design
ALPHAS = (0.05, 0.01, 0.001)
SEED, N_RESAMPLE = 42, 2000     # resampling SE of matched power: null and alternative replicates resampled together
ARM_PROCS = 15                                                       # + this driver = the 16-process cap
R_JOBS = 7                                                           # Rscript wrapper + R each, + driver <= 16
CHUNK = 10                                                           # replicates per asSeq checkpoint
R_ENV = {'R_LD_LIBRARY_PATH': '/usr/lib/R/lib:/usr/lib/x86_64-linux-gnu',
         'LD_LIBRARY_PATH': '/usr/local/cuda/lib64'}                 # CLAUDE.md, R's BLAS entry
R_SCRIPT = Path(__file__).resolve().with_name('external_benchmark_mirror_trecase.R')
HAPMIX = {'hapmixQTL gibbs': 'gibbs', 'hapmixQTL split': 'split', 'hapmixQTL plus_one': 'plus_one'}
COMPARATORS = ('TReC-only', 'ASE-only', 'TReCASE (joint)')
# asSeq's two p-values: final_Pvalue is what asSeq reports (trecase.c:1311-1323: the joint p unless the
# cis-trans test rejects or is NA, then the TReC p; the plasmode TReCASE arm's pval_nominal); Joint_Pvalue
# is the joint LRT alone, the harness TReCASE's hypothesis, with a failed joint fit counted as a
# non-rejection, as the harness counts its own failed fits.
ASSEQ = {'TReCASE (asSeq final p)': 'final_Pvalue', 'TReCASE (asSeq joint p)': 'Joint_Pvalue'}
REFS = ('TReCASE (joint)', 'TReCASE (asSeq final p)')
assert set(B.METHODS) == set(COMPARATORS) | set(HAPMIX), sorted(B.METHODS)


def conditions():
    return [(N, k) for N in NS for k in KAPPAS]


def tag(N, kappa):
    return f'N{N}_k{kappa:.2f}'


def seed0(kappa):
    return SEED0.get(kappa, ALT_SEED0)


def write_atomic(path, write):
    tmp = path.with_name(path.name + '.tmp')
    write(tmp)
    os.replace(tmp, path)


def simulate(N, kappa, r):
    return B.simulate_locus(N, np.random.RandomState(seed0(kappa) + r), kappa=kappa, **SIM)


def _init_worker():
    import torch
    torch.set_num_threads(1)


def one_rep(args):
    """Every harness arm on replicate r, reseeded exactly as ase_external_benchmark.run()."""
    N, kappa, r = args
    d = simulate(N, kappa, r)
    arm_rng = lambda: np.random.RandomState(seed0(kappa) + 900000 + r)   # noqa: E731
    row = dict(rep=r)
    for name in COMPARATORS:
        row[f'{name}|p'], row[f'{name}|stat'] = B.METHODS[name](d, arm_rng())
    for name, w in HAPMIX.items():
        p, s, info = B.hapmix_pval(d, arm_rng(), w, return_info=True)
        row.update({f'{name}|p': p, f'{name}|stat': s, f'{name}|dof': info['dof_nominal'],
                    f'{name}|meier': info['meier'], f'{name}|admitted': info['allelic_admitted']})
    row['n_a'] = info['n_a']
    row['n_het'] = int(d['het'].sum())
    _, _, va, vt = B.emulated_summaries(d, arm_rng())
    row['vt_over_delta'] = float(np.median(vt / (d['T'] / (d['T'] + d['lib']) ** 2 / np.log(2) ** 2)))
    row['median_T'] = float(np.median(d['T']))
    row['median_n_as'] = float(np.median(d['n_as']))
    return row


def run_arms():
    (OUT / 'arms').mkdir(parents=True, exist_ok=True)
    with Pool(ARM_PROCS, initializer=_init_worker) as pool:
        for N, kappa in conditions():
            f = OUT / 'arms' / f'{tag(N, kappa)}.tsv'
            if f.exists():
                print(f'skip {f.name} (finished)', flush=True)
                continue
            t0 = time.time()
            rows = pool.map(one_rep, [(N, kappa, r) for r in range(REPS)], chunksize=5)
            df = pd.DataFrame(rows).sort_values('rep')
            write_atomic(f, lambda p: df.to_csv(p, sep='\t', index=False))
            print(f'{f.name}: {len(df)} replicates in {time.time() - t0:.0f} s', flush=True)


def write_trecase_inputs(N, kappa):
    """Y, Y1, Y2, Z, offset for every replicate of one condition, [reps x N] float64."""
    d = OUT / 'trecase' / 'inputs' / tag(N, kappa)
    if (d / 'N.txt').exists():
        return d
    d.mkdir(parents=True, exist_ok=True)
    arr = {k: np.empty((REPS, N)) for k in ('Y', 'Y1', 'Y2', 'Z', 'offset')}
    for r in range(REPS):
        s = simulate(N, kappa, r)
        xL = np.where(s['g'] == 2, 1, np.where((s['g'] == 1) & (s['s'] > 0), 1, 0))   # L carries ALT
        xR = s['g'] - xL
        arr['Y'][r], arr['Y1'][r], arr['Y2'][r] = s['T'], s['yL'], s['yR']
        arr['Z'][r], arr['offset'][r] = 3 * xL + xR, np.log(s['lib'])
    for k, a in arr.items():
        write_atomic(d / f'{k}.bin', lambda p, a=a: np.ascontiguousarray(a, np.float64).tofile(p))
    (d / 'N.txt').write_text(f'{N}\n')                                # written last: marks complete inputs
    print(f'inputs {tag(N, kappa)}: {REPS} replicates, Z counts {np.bincount(arr["Z"].astype(int).ravel()).tolist()}',
          flush=True)
    return d


def run_chunk(cond_dir, start, end, out):
    t0 = time.perf_counter()
    with open(f'{out}.log', 'w') as fh:
        rc = subprocess.run(['Rscript', str(R_SCRIPT), str(cond_dir), str(start), str(end), str(out)],
                            stdout=fh, stderr=subprocess.STDOUT, env={**os.environ, **R_ENV}).returncode
    if rc != 0:
        raise SystemExit(f'{out}: Rscript exited {rc}; see {out}.log')
    return time.perf_counter() - t0


def run_trecase():
    jobs = []
    for N, kappa in conditions():
        cond_dir = write_trecase_inputs(N, kappa)
        od = OUT / 'trecase' / tag(N, kappa)
        od.mkdir(parents=True, exist_ok=True)
        for start in range(0, REPS, CHUNK):
            out = od / f'reps{start:03d}_{start + CHUNK:03d}.tsv'
            if out.exists():
                continue
            jobs.append((cond_dir, start, start + CHUNK, out))
    print(f'{len(jobs)} asSeq chunks of {CHUNK} replicates to run, {R_JOBS} at a time', flush=True)
    t0 = time.time()
    with cf.ThreadPoolExecutor(R_JOBS) as ex:
        futs = {ex.submit(run_chunk, *j): j for j in jobs}
        for i, fut in enumerate(cf.as_completed(futs), 1):
            secs = fut.result()
            if i % 20 == 0 or i == len(jobs):
                print(f'  {i}/{len(jobs)} chunks, last {secs:.1f} s, elapsed {time.time() - t0:.0f} s', flush=True)


def load_condition(N, kappa):
    df = pd.read_csv(OUT / 'arms' / f'{tag(N, kappa)}.tsv', sep='\t').set_index('rep')
    files = sorted((OUT / 'trecase' / tag(N, kappa)).glob('reps*.tsv'))
    t = pd.concat([pd.read_csv(f, sep='\t') for f in files]).set_index('rep') if files else None
    if t is None or len(t) != REPS:
        print(f'{tag(N, kappa)}: asSeq has {0 if t is None else len(t)} of {REPS} replicates; its arm is skipped')
        return df, None
    for arm, col in ASSEQ.items():
        df[f'{arm}|p'] = t[col].astype(float).fillna(1.0)      # NA (no fit) is a non-rejection
        df[f'{arm}|stat'] = stats.chi2.isf(df[f'{arm}|p'], 1)
    df['asseq_joint_failed'] = t['Joint_Pvalue'].isna()
    for c in ('Joint_b', 'ASE_b', 'TReC_b', 'final_Pvalue', 'trans_Pvalue', 'n_ASE', 'n_ASE_Het', 'error',
              'seconds', 'BBod', 'NBod'):
        df[f'asseq_{c}'] = t[c]
    return df, t


def summarize():
    data = {c: load_condition(*c) for c in conditions()}
    prior = json.loads(PRIOR.read_text())
    res = dict(design=dict(reps=REPS, Ns=NS, kappas=KAPPAS, sim=SIM, null_seed0=1000, alt_seed0=ALT_SEED0),
               by_N={})
    per_rep = []
    for N in NS:
        null, _ = data[(N, 1.0)]
        arms = [a for a in (*COMPARATORS, *ASSEQ, *HAPMIX) if f'{a}|p' in null]
        R = dict(type1={}, power={}, diagnostics={})
        for a in arms:
            p = null[f'{a}|p'].values
            R['type1'][a] = {str(al): dict(rate=float(np.mean(p < al)), count=int(np.sum(p < al)),
                                           mc_se=float(np.sqrt(al * (1 - al) / len(p)))) for al in ALPHAS}
        S0 = np.column_stack([null[f'{a}|stat'].values for a in arms])
        thr = np.quantile(S0, 0.95, axis=0)                          # harness main()'s rule, per arm
        rng = np.random.default_rng(np.random.SeedSequence(SEED).spawn(len(NS))[NS.index(N)])
        for kappa in KAPPAS[1:]:
            alt, _ = data[(N, kappa)]
            S1 = np.column_stack([alt[f'{a}|stat'].values for a in arms])
            det = (S1 > thr).astype(float)
            # threshold noise: resample null replicates (new thresholds) and alternative replicates (paired arms)
            resampled = np.empty((N_RESAMPLE, len(arms)))
            for b in range(N_RESAMPLE):
                tb = np.quantile(S0[rng.integers(0, REPS, REPS)], 0.95, axis=0)
                resampled[b] = (S1[rng.integers(0, REPS, REPS)] > tb).mean(0)
            R['power'][str(kappa)] = {}
            for i, a in enumerate(arms):
                e = dict(matched=float(det[:, i].mean()), resample_se=float(resampled[:, i].std(ddof=1)),
                         nominal_005=float(np.mean(alt[f'{a}|p'].values < 0.05)))
                for ref in REFS:
                    j = arms.index(ref) if ref in arms else None
                    if j is not None and j != i:
                        dlt = det[:, i] - det[:, j]
                        rd = resampled[:, i] - resampled[:, j]
                        e[f'diff_vs_{ref}'] = dict(diff=float(dlt.mean()),
                                                   se_paired=float(dlt.std(ddof=1) / np.sqrt(len(dlt))),
                                                   se_resample=float(rd.std(ddof=1)))
                R['power'][str(kappa)][a] = e
        for kappa in KAPPAS:
            df, t = data[(N, kappa)]
            dg = dict(n_a=[int(df.n_a.min()), int(df.n_a.max())], n_het=[int(df.n_het.min()), int(df.n_het.median()),
                                                                          int(df.n_het.max())],
                      admitted_all=bool(all(df[f'{a}|admitted'].all() for a in HAPMIX)),
                      vt_over_delta_median=float(df.vt_over_delta.median()),
                      median_T=float(df.median_T.median()), median_n_as=float(df.median_n_as.median()),
                      hapmix_failed={a: int((df[f'{a}|stat'] == 0).sum()) for a in HAPMIX},
                      dof_nominal={a: [float(df[f'{a}|dof'].min()), float(df[f'{a}|dof'].median()),
                                       float(df[f'{a}|dof'].max())] for a in HAPMIX},
                      meier={a: [float(df[f'{a}|meier'].median()), float(df[f'{a}|meier'].max())] for a in HAPMIX})
            if t is not None:
                logs = ''.join(f.read_text() for f in (OUT / 'trecase' / tag(N, kappa)).glob('reps*.tsv.log'))
                fp, tp = t.final_Pvalue.astype(float), t.trans_Pvalue.astype(float)
                switched = t.Joint_Pvalue.notna() & (tp < 0.05)          # final p = TReC p after a cis-trans rejection
                dg['asseq_trace'] = dict(theta_fail=logs.count('fail to estimate theta in joint model'),
                                         lbfgsb_52=logs.count('fail=52'),
                                         ase_baseline_fail=logs.count('fail to fit baseline ASE model'),
                                         ase_fail=logs.count('fail ASE model'))
                dg['asseq_switch'] = dict(n=int(switched.sum()), rejected_005=int((switched & (fp < 0.05)).sum()),
                                          joint_failed_rejected_005=int((t.Joint_Pvalue.isna() & (fp < 0.05)).sum()))
                dg['asseq'] = dict(joint_failed=int(df.asseq_joint_failed.sum()), errors=int(t.error.sum()),
                                   succeed_all=bool((t.succeed == 1).all()),
                                   baseline_failed=int((t.yFailBaselineModel != 0).sum()),
                                   seconds_total=float(t.seconds.sum()),
                                   median_n_ASE=float(t.n_ASE.median()), median_n_ASE_Het=float(t.n_ASE_Het.median()),
                                   joint_b_median=float(t.Joint_b.median()), ase_b_median=float(t.ASE_b.median()),
                                   joint_b_positive=float(np.mean(t.Joint_b.dropna() > 0)),
                                   final_p_rate_005=float(np.mean(t.final_Pvalue.astype(float).fillna(1) < 0.05)),
                                   trans_p_rate_005=float(np.mean(t.trans_Pvalue.dropna().astype(float) < 0.05)),
                                   bb_od_median=float(t.BBod.median()), nb_od_median=float(t.NBod.median()))
            R['diagnostics'][str(kappa)] = dg
            per_rep.append(df.reset_index().assign(N=N, kappa=kappa))
        if N == 200:   # identical data to the 2026-09-23 record: the comparators must reproduce it exactly
            chk = {}
            for a in COMPARATORS:
                old = prior['null'][a]
                new = R['type1'][a]
                chk[a] = dict(t05=(old['t05'], new['0.05']['rate']), t01=(old['t01'], new['0.01']['rate']),
                              power={k: (prior['power'][k][a]['matched'], R['power'][k][a]['matched'])
                                     for k in ('1.05', '1.1', '1.2')})
            res['reproduces_20260923'] = dict(
                exact=all(o == n for a in chk.values() for o, n in [a['t05'], a['t01'], *a['power'].values()]),
                values=chk)
            print('2026-09-23 reproduction:', 'EXACT' if res['reproduces_20260923']['exact'] else 'DIFFERS',
                  json.dumps(chk))
        res['by_N'][str(N)] = R
    write_atomic(OUT / 'summary.json', lambda p: p.write_text(json.dumps(res, indent=1)))
    pr = pd.concat(per_rep, ignore_index=True)
    write_atomic(OUT / 'per_replicate.tsv.gz', lambda p: pr.to_csv(p, sep='\t', index=False, compression='gzip'))
    for N in NS:
        R = res['by_N'][str(N)]
        print(f'\n=== N = {N} ===')
        for a, v in R['type1'].items():
            print(f'  {a:22s} ' + '  '.join(f'{al}: {v[al]["rate"]:.4f}' for al in map(str, ALPHAS)))
        for k, v in R['power'].items():
            print(f'  fold {k}: ' + '  '.join(f'{a} {e["matched"]:.3f}' for a, e in v.items()))
    print(f'wrote {OUT / "summary.json"}')


if __name__ == '__main__':
    step = sys.argv[1] if len(sys.argv) == 2 else None
    steps = dict(arms=run_arms, trecase=run_trecase, summarize=summarize)
    if step not in steps:
        raise SystemExit(__doc__)
    steps[step]()

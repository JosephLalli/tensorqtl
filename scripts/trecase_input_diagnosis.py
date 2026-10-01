"""TReCASE on integer total counts: 05_run_trecase.py with ONE change, Y = rint(pT) (task of 2026-09-28).

Why: on the simulated-effects benchmark's deep set TReCASE ranks non-null genes below total-only tensorQTL at a matched 5% false-discovery
proportion, and its total-count test alone ranks lowest; 05 hands asSeq's negative-binomial total-count (TReC) model the
thinned Salmon totals pT as fractional numbers. This reruns asSeq on beta0.0 rep 0 and reps 0-2 of beta0.4 and beta0.8
of the deep set (plasmode_meier_20260927, the default SIMULATED_EFFECTS_GENE_SET) with Y = rint(pT). Everything else is 05's:
Y1, Y2, X, offset and Z are built by 05's functions and the R call is run_trecase.R. Checked before any run: 05's
inputs rebuilt from the unchanged dataset are byte-identical to the committed run's (COMMITTED_WORK), the integer run's
inputs other than Y.bin are too, and its Y.bin is rint of the committed one; and one gene rerun on the unchanged inputs
(CONTROL) reproduces the committed asSeq output byte for byte.

Scored against the committed results (common.JOINT['trecase'], staged from COMMITTED_WORK's run), for each of 05's
component p (COMPONENTS): power at FDR realized false-discovery proportion (06_score.ranking's fdp_matched on each gene's
lead by that p, ties by |slope / se|, pooled over the 3 datasets of a |beta|), with a paired gene-clustered interval for
integer minus fractional (genes resampled N_BOOT times, each carrying its units in both runs); the null-gene rate at
common.ALPHAS on the anchor (06_score.null_calibration); and the share of gene units whose reported lead (smallest
pval_nominal) has no joint p, null / non-null, by cause. Cause of a missing joint p per tested variant, first that
applies: asSeq's baseline allelic (ASE) model failed for the gene (yFailBaselineModel 2 or 3); TReC p missing; fewer
than 5 heterozygous allelic donors (asSeq's min.n.het: the joint model is never attempted); 'fail ASE model' (the
allelic fit under the alternative; trecase.c:834-838); 'fail to estimate theta in joint model' (trecase.c:1003-1012,
its j is the marker's 0-based row); otherwise 'other' (a TReC refit with linear dosage, which skips the joint model
(trecase.c:740-757), or a b_xj, phi or iteration-limit stop, none of which the trace names at trace = 1). Concentration:
the theta-failure rate among variants whose joint fit was attempted, by heterozygous allelic donors (n_ase_het), by the
gene's zero-haplotype records in the dataset (exactly one haplotype below common.EXPRESSIBLE_MIN reads, removed by
common.allelic_kept before asSeq), by the gene's median total count, by the two crossed, and by the allelic-only
chi-square. Also: the same variant's cause in both runs (agreement); the null units of each |beta| whose lead p is at or
below tensorQTL's own fdp_matched threshold, tensorQTL's lead included (null_tail); at the reported leads behind the
few-heterozygote gate, how many would pass it if the records allelic_kept removes were kept (few_het); and asSeq at
trace = 2 on four markers of one gene, which prints each joint iteration (SPOT).

Output under OUT: work/ (inputs, asSeq files and one trace log per gene; a gene whose status file exists is not rerun),
work/unchanged/ (05's inputs from the unchanged datasets, and the control gene), results/<scenario>/trecase_integer/
nominal_repNNN.parquet (05's layout), summary.json, fig_trecase_integer.png, report.html.
"""
import concurrent.futures as cf
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'benchmark' / 'simulated_effects'))
import common as C  # noqa: E402

T, SC = C.module('05_run_trecase'), C.module('06_score')

OUT = C.D / 'input_diagnosis_20260928' / 'trecase_integer'   # task of 2026-09-28
WORK, RESULTS = OUT / 'work', OUT / 'results'
COMMITTED_WORK = C.COMMITTED / 'trecase_asseq_work'   # the committed TReCASE run's inputs, asSeq files and traces (05 at 3aac315)
DEEP_SET = 'corrected_null_store_20260925'             # the deep set: common.GENE_SETS key whose root is plasmode_meier_20260927
RUNS = [('beta0.0', 0)] + [(f'beta{b}', r) for b in ('0.4', '0.8') for r in range(3)]   # task of 2026-09-28
ARM = 'trecase_integer'        # results directory and fingerprint name of this run
JOBS = 30                      # R processes at once: Rscript execs into R, one process per gene (ps, 2026-09-28); plus this driver, under the task's cap of 32
CONTROL = ('beta0.4', 0)       # dataset of the control gene: its fastest gene in the committed run, rerun on the unchanged inputs
INPUT_FILES = ('Y.bin', 'Y1.bin', 'Y2.bin', 'X.bin', 'offset.bin', 'genes.tsv')   # 05.write_dataset's files
COMPONENTS = {'pval_t': ('slope_t', 'slope_t_se'), 'pval_a': ('slope_a', 'slope_a_se'),
              'pval_joint': ('slope_joint', 'slope_joint_se'), 'pval_nominal': ('slope', 'slope_se')}   # 05's columns
COMPONENT_LABELS = {'pval_t': 'total count (TReC)', 'pval_a': 'allelic (ASE)', 'pval_joint': 'joint (where fitted)',
                    'pval_nominal': 'reported (pval_nominal)'}
RUN_COLORS = {'fractional': '#5b7fa6', 'integer': '#e08a3c'}   # the committed run (Y = pT), this run (Y = rint(pT))
TAIL_VERDICT = {True: 'At the same threshold TReC reaches at least as many non-null units as tensorQTL, so it loses in the null tail.',
                False: 'At the same threshold TReC reaches fewer non-null units than tensorQTL and more null ones, so it loses on both sides.'}
BOOT_KEY = 60                 # SeedSequence spawn key of the paired interval (06 uses 30 and 33, 02 and 03 1-5)
CAUSES = ('ase_baseline', 'trec', 'few_het', 'ase_fit', 'theta', 'other')   # order of precedence (docstring)
THETA_RE = re.compile(r'i=\d+, j=(\d+), h0=(\d+), fail to estimate theta in joint model\n'
                      r'\s*theta=(\S+), gradience=(\S+), fail=(\d+)')   # trecase.c:1005-1008
ASE_RE = re.compile(r'i=\d+, j=(\d+), fail ASE model')                   # trecase.c:838
HET_BINS = ([5, 10, 20, 40, 93], ['5-9', '10-19', '20-39', '40-92'])      # heterozygous allelic donors (92 donors)
ZERO_BINS = ([0, 1, 4, 11, 93], ['0', '1-3', '4-10', '11+'])             # zero-haplotype records of the gene
DEPTH_BINS = ([0, 100, 500, 2000, np.inf], ['<100', '100-499', '500-1999', '>=2000'])   # gene's median total count pT
CHISQ_BINS = ([-np.inf, 1, 4, 10, np.inf], ['<1', '1-4', '4-10', '>=10'])   # the variant's allelic-only likelihood-ratio chi-square
SPOT = ('beta0.4', 0, 'ABCC4', (8141, 8142, 8143, 8151))   # asSeq at trace = 2 on four markers (0-based rows) of one committed gene: 8141 and 8151 fail theta in its trace, 8142 converges
SPOT_R = '''.libPaths(c("/mnt/ssd/lalli/usr/local/lib/R/library", .libPaths(), .Library.site)); suppressMessages(library(asSeq))
a <- commandArgs(trailingOnly = TRUE); rd <- function(f, n) matrix(readBin(f, "double", n = file.size(f) / 8), nrow = n)
off <- readBin(file.path(a[1], "offset.bin"), "double", n = file.size(file.path(a[1], "offset.bin")) / 8); N <- length(off)
gt <- read.delim(file.path(a[1], "genes.tsv"), colClasses = c("character", "integer", "integer")); k <- match(a[3], gt$gene)
mk <- read.delim(file.path(a[2], sprintf("chr%d.markers.tsv", gt$chr[k])), colClasses = c("character", "integer", "integer"))
s <- as.integer(strsplit(a[5], ",")[[1]]) + 1; Z <- rd(file.path(a[2], sprintf("chr%d.Z.bin", gt$chr[k])), N)[, s, drop = FALSE]
y <- lapply(c("Y", "Y1", "Y2"), function(v) rd(file.path(a[1], paste0(v, ".bin")), N)[, k, drop = FALSE])
r <- trecase(y[[1]], y[[2]], y[[3]], rd(file.path(a[1], "X.bin"), N), Z, output.tag = a[4], p.cut = 1000, offset = off,
             local.only = TRUE, local.distance = 1e6, eChr = gt$chr[k], ePos = gt$pos[k], mChr = mk$chr[s], mPos = mk$pos[s], trace = 2)
'''   # run_trecase.R's trecase call on SPOT's markers only, at trace = 2 (trecase.c prints each joint iteration at trace > 1)


def ddir(sc, r, root=WORK):
    return root / sc / f'rep{r:03d}'


def same(a, b):
    return a.read_bytes() == b.read_bytes()


def committed_seconds(sc, r, g):
    return float(pd.read_csv(ddir(sc, r, COMMITTED_WORK) / 'out' / f'{g}_status.tsv', sep='\t').seconds.iloc[0])


def prepare():
    """Loader inputs, genotype tables and every dataset's inputs, checked against the committed run; returns the setup."""
    if C.GENE_SET != DEEP_SET or C.ACCEPTANCE:
        raise SystemExit(f'gene set {C.GENE_SET} (acceptance {C.ACCEPTANCE}): this diagnosis is on the deep set {DEEP_SET}')
    S = C.setup(C.load()[0])
    meta = json.loads((C.DATASETS / 'meta.json').read_text())
    if S['genes'] != meta['genes'] or S['order'] != meta['donors']:
        raise SystemExit(f'{C.DATASETS / "meta.json"}: genes or donors differ from the loader inputs')
    ids = T.write_genotypes(S, WORK / 'genotypes')
    names = sorted(p.name for p in (COMMITTED_WORK / 'genotypes').iterdir())
    if names != sorted(p.name for p in (WORK / 'genotypes').iterdir()) or not all(
            same(WORK / 'genotypes' / n, COMMITTED_WORK / 'genotypes' / n) for n in names):
        raise SystemExit(f'genotype tables differ from {COMMITTED_WORK / "genotypes"}')
    print(f'genotypes: {len(names)} files byte-identical to the committed run', flush=True)
    chrom = {g: T.chrom_int(S['gp'].loc[g, 'chr']) for g in S['genes']}
    expected = {g: S['scanned'][g] & set(ids[chrom[g]]) for g in S['genes']}
    if any(expected[g] != S['scanned'][g] for g in S['genes']):
        raise SystemExit('a tested variant with varying dosage is missing from its chromosome marker table')
    D = {}
    for sc, r in RUNS:
        ds = C.load_dataset(C.DATASETS, sc, r)
        committed = C.JOINT['trecase'] / sc / 'trecase' / f'nominal_rep{r:03d}.parquet'
        if C.stored_fingerprint(committed) != C.fingerprint(ds, 'trecase'):
            raise SystemExit(f'{committed} was not computed on {C.DATASETS / sc} rep {r}')
        ref, old = ddir(sc, r, COMMITTED_WORK), ddir(sc, r, WORK / 'unchanged')
        T.write_dataset(S, ds, old)
        bad = [n for n in INPUT_FILES if not same(old / n, ref / n)]
        dsi = {**ds, 'pT': np.rint(ds['pT'])}   # THE change
        new = ddir(sc, r)
        T.write_dataset(S, dsi, new)
        bad += [f'integer {n}' for n in INPUT_FILES[1:] if not same(new / n, ref / n)]
        y_ref, y = np.fromfile(ref / 'Y.bin'), np.fromfile(new / 'Y.bin')
        if bad or not np.array_equal(y, np.rint(y_ref)):
            raise SystemExit(f'{sc} rep {r}: inputs differ from {ref}: {bad or "integer Y.bin is not rint of the committed"}')
        d = np.abs(y - y_ref)
        print(f'{sc} rep {r}: 05 inputs rebuilt byte-identical to the committed ({len(INPUT_FILES)} files), integer run '
              f'differs only in Y.bin: {int((d > 0).sum()):,} of {len(y):,} totals changed, mean |change| {d.mean():.3f}, '
              f'max {d.max():.3f} reads, min Y {y.min():g}', flush=True)
        (new / 'out').mkdir(exist_ok=True)
        D[sc, r] = dsi
    return S, ids, chrom, expected, D


def control(S):
    """The committed run's fastest gene of CONTROL, rerun on the unchanged inputs: asSeq's files must match byte for byte."""
    sc, r = CONTROL
    g = min(S['genes'], key=lambda x: committed_seconds(sc, r, x))
    d = ddir(sc, r, WORK / 'unchanged')
    (d / 'out').mkdir(exist_ok=True)
    s, skipped = T.run_gene(d, WORK / 'genotypes', g, d / 'out' / g)
    bad = [f for f in ('eqtl.txt', 'freq.txt') if not same(d / 'out' / f'{g}_{f}', ddir(sc, r, COMMITTED_WORK) / 'out' / f'{g}_{f}')]
    if bad:
        raise SystemExit(f'control {g} ({sc} rep {r}): asSeq output {bad} differs from the committed run')
    print(f'control {g} ({sc} rep {r}, unchanged inputs): asSeq output byte-identical to the committed run '
          f'({"status present, not rerun" if skipped else f"{s:.0f} s"})',
          flush=True)
    return g


def run_genes(S):
    """Every (dataset, gene) of RUNS without a status file, longest committed time first, JOBS at a time."""
    secs = {(sc, r, g): committed_seconds(sc, r, g) for sc, r in RUNS for g in S['genes']}
    todo = sorted((u for u in secs if not Path(f'{ddir(*u[:2]) / "out" / u[2]}_status.tsv').exists()),
                  key=secs.get, reverse=True)
    left = sum(secs[u] for u in todo)
    print(f'{len(secs)} gene runs, {len(secs) - len(todo)} finished (status present, not rerun); committed asSeq time of '
          f'the {len(todo)} left {left / 3600:.1f} CPU h, at least {left / 3600 / JOBS:.1f} h at {JOBS} at once', flush=True)
    t0, done = time.perf_counter(), 0
    ex = cf.ThreadPoolExecutor(JOBS)
    try:
        futs = [ex.submit(T.run_gene, ddir(sc, r), WORK / 'genotypes', g, ddir(sc, r) / 'out' / g) for sc, r, g in todo]
        for f in cf.as_completed(futs):
            f.result()
            done += 1
            if done % 50 == 0 or done == len(todo):
                print(f'{done} of {len(todo)} gene runs done, {(time.perf_counter() - t0) / 60:.1f} min', flush=True)
    finally:
        ex.shutdown(cancel_futures=True)   # a failed gene cancels the queue


def convert(S, ids, chrom, expected, D):
    """05's conversion of every gene's asSeq output, written per dataset to RESULTS; 05's counts per dataset."""
    counts = {}
    for sc, r in RUNS:
        tags = [ddir(sc, r) / 'out' / g for g in S['genes']]
        missing = [t.name for t in tags if not Path(f'{t}_status.tsv').exists()]
        if missing:
            raise SystemExit(f'{sc} rep {r}: {len(missing)} genes have no status file ({missing[:5]}); rerun to finish them')
        parts = [T.convert(g, t, ids[chrom[g]], expected[g]) for g, t in zip(S['genes'], tags)]
        df = pd.concat([p[0] for p in parts], ignore_index=True)
        C.write_parquet(df, RESULTS / sc / ARM / f'nominal_rep{r:03d}.parquet', C.fingerprint(D[sc, r], ARM), 'log2')
        st = [p[1] for p in parts]
        counts[f'{sc} rep {r:03d}'] = T.dataset_counts(df, D[sc, r], S, expected, st, [float(s.seconds) for s in st],
                                                         [p[2] for p in parts])
        print(f'{sc} rep {r}: {len(df):,} rows written; joint p missing {int((~df.joint_ok).sum()):,}', flush=True)
    return counts


def variants(S, ids, chrom, D, run):
    """Every tested variant of RUNS in one run ('fractional': the committed; 'integer': this one), in file order, with
    the cause of a missing joint p, the gene's zero-haplotype records and median total count, and the theta failures."""
    res, work = (C.JOINT['trecase'], COMMITTED_WORK) if run == 'fractional' else (RESULTS, WORK)
    arm = 'trecase' if run == 'fractional' else ARM
    cols = ['phenotype_id', 'variant_id', 'joint_ok', 'n_ase_het', 'chisq_a', 'bb_od'] + [c for k, v in COMPONENTS.items() for c in (k, *v)]
    parts, fails = [], []
    for sc, r in RUNS:
        d = pd.read_parquet(res / sc / arm / f'nominal_rep{r:03d}.parquet', columns=cols)
        ds = D[sc, r]
        gi = pd.Index(S['genes']).get_indexer(d.phenotype_id)
        zero = ((ds['pL'] < C.EXPRESSIBLE_MIN) ^ (ds['pR'] < C.EXPRESSIBLE_MIN)).sum(1)
        d['zero_hap'], d['median_total'] = zero[gi], np.median(ds['pT'], 1)[gi]
        base, theta, ase = np.zeros(len(d), bool), np.zeros(len(d), bool), np.zeros(len(d), bool)
        for g in S['genes']:
            tag = ddir(sc, r, work) / 'out' / g
            log = Path(f'{tag}.log').read_text()
            on = (d.phenotype_id == g).values
            pos = pd.Index(ids[chrom[g]]).get_indexer(d.variant_id[on])   # 0-based marker row, asSeq's j
            base[on] = pd.read_csv(f'{tag}_status.tsv', sep='\t').yFailBaselineModel.iloc[0] >= 2
            th = [(int(j), float(t), float(gr), int(f)) for j, _, t, gr, f in THETA_RE.findall(log)]
            theta[on] = np.isin(pos, [x[0] for x in th])
            ase[on] = np.isin(pos, [int(j) for j in ASE_RE.findall(log)])
            fails += [dict(scenario=sc, rep=r, gene=g, j=j, theta=t, gradient=gr, fail=f) for j, t, gr, f in th]
        cause = np.select([d.joint_ok.values, base, ~np.isfinite(d.pval_t.values), d.n_ase_het.values < T.MIN_N_HET, ase, theta],
                          ['ok', *CAUSES[:5]], 'other')
        if (theta & (d.joint_ok.values | (cause != 'theta'))).any():
            raise SystemExit(f'{run} {sc} rep {r}: a theta failure in the trace has a joint p or an earlier cause')
        parts.append(d.assign(scenario=sc, rep=r, cause=cause))
    V = pd.concat(parts, ignore_index=True)
    F = pd.DataFrame(fails)
    print(f'{run}: {len(V):,} tested variants over {len(RUNS)} datasets; missing joint p by cause '
          f'{V.cause.value_counts().to_dict()}; theta failures in the traces {len(F):,}, fail codes '
          f'{F.fail.value_counts().to_dict()}', flush=True)
    return V, F


def leads(V, U, col):
    """Each (dataset, gene)'s lead by one component p, as 06_score.causal_and_leads: smallest p (missing = inf), ties by
    the larger |slope / se|; joined to the units' truth and band."""
    slope, se = COMPONENTS[col]
    d = V[[c for c in ('scenario', 'rep', 'phenotype_id', 'variant_id', 'cause', 'n_ase_het') if c in V]].assign(
        p=V[col].where(np.isfinite(V[col]), np.inf), absstat=np.abs(V[slope] / V[se]))
    top = (d.sort_values(['scenario', 'rep', 'phenotype_id', 'p', 'absstat'], ascending=[True, True, True, True, False],
                         kind='stable').groupby(['scenario', 'rep', 'phenotype_id'], sort=False).head(1))
    top = top.rename(columns={'phenotype_id': 'gene', 'p': 'lead_p', 'absstat': 'lead_absstat', 'variant_id': 'lead_variant'})
    L = U.merge(top, on=['scenario', 'rep', 'gene'], how='inner', validate='one_to_one')
    if len(L) != len(RUNS) * U.gene.nunique():
        raise SystemExit(f'{col}: {len(L)} lead units against {len(RUNS) * U.gene.nunique()}')
    return L


def fdp_power(p, s, null):
    """06_score.ranking's fdp_matched power over all units, on arrays (fast enough for the resamples; checked against
    ranking at every point estimate)."""
    rank = SC.evidence_rank(p, s)
    o = np.argsort(-rank, kind='stable')
    cut = np.r_[rank[o][1:] != rank[o][:-1], True] & (np.cumsum(null[o]) / np.arange(1, len(o) + 1) <= SC.FDR)
    k = int(np.flatnonzero(cut).max()) + 1 if cut.any() else 0
    return float((~null[o[:k]]).sum() / (~null).sum())


def score(V, U, genes):
    """Power at FDR realized false-discovery proportion per component and |beta| for both runs, and the paired interval."""
    idx = np.random.default_rng(np.random.SeedSequence(C.SEED, spawn_key=(BOOT_KEY,))).integers(0, len(genes), (SC.N_BOOT, len(genes)))
    out = {}
    for col in COMPONENTS:
        L = {run: leads(V[run], U, col) for run in V}
        for b in ('0.4', '0.8'):
            Ls = {run: x[x.scenario == f'beta{b}'].reset_index(drop=True) for run, x in L.items()}
            pt = {}
            for run, x in Ls.items():
                f = SC.ranking(x, (BOOT_KEY, 1))['fdp_matched']
                pt[run] = (f['all']['power'], f['discoveries'], f['false'], f['p_threshold'])
                if fdp_power(x.lead_p.values, x.lead_absstat.values, x.is_null.values) != pt[run][0]:
                    raise SystemExit(f'{col} |beta| {b} {run}: fdp_power differs from 06_score.ranking')
            arr = {run: (x.lead_p.values, x.lead_absstat.values, x.is_null.values,
                         [np.flatnonzero(x.gene.values == g) for g in genes]) for run, x in Ls.items()}
            diff = []
            for ix in idx:
                pw = {}
                for run, (p, s, null, rows) in arr.items():
                    k = np.concatenate([rows[i] for i in ix])
                    pw[run] = fdp_power(p[k], s[k], null[k])
                diff.append(pw['integer'] - pw['fractional'])
            out[f'{col} {b}'] = dict(component=col, beta_abs=float(b), non_null=int((~Ls['integer'].is_null).sum()),
                                     **{f'{run}_{k}': v for run, x in pt.items() for k, v in zip(('power', 'discoveries', 'false', 'p_threshold'), x)},
                                     diff=pt['integer'][0] - pt['fractional'][0], diff_lo=float(np.quantile(diff, .025)),
                                     diff_hi=float(np.quantile(diff, .975)))
            print(f'{col} |beta| {b}: power at {SC.FDR:g} FDP fractional {pt["fractional"][0]:.3f} integer {pt["integer"][0]:.3f}, '
                  f'difference {out[f"{col} {b}"]["diff"]:+.3f} [{out[f"{col} {b}"]["diff_lo"]:+.3f}, '
                  f'{out[f"{col} {b}"]["diff_hi"]:+.3f}]', flush=True)
    return out


def lead_failures(V, U):
    """At each gene unit's reported lead (smallest pval_nominal): share without a joint p, null / non-null, by cause."""
    rows = []
    for run in V:
        L = leads(V[run], U, 'pval_nominal')
        L['group'] = [('anchor' if s == 'beta0.0' else f'|beta| {s[4:]}') + (' null' if n else ' non-null')
                      for s, n in zip(L.scenario, L.is_null)]
        for grp, x in L.groupby('group'):
            rows.append(dict(run=run, group=grp, units=len(x), no_joint=float((x.cause != 'ok').mean()),
                             **{c: int((x.cause == c).sum()) for c in CAUSES}))
    return pd.DataFrame(rows)


def concentration(V):
    """Theta-failure and other-failure shares among variants whose joint fit passed asSeq's gates, by BINS."""
    bins = {'heterozygous allelic donors': ('n_ase_het', HET_BINS), 'zero-haplotype records of the gene': ('zero_hap', ZERO_BINS),
            'median total count of the gene': ('median_total', DEPTH_BINS), 'allelic-only chi-square': ('chisq_a', CHISQ_BINS)}
    rows = []
    for run, v in V.items():
        a = v[v.cause.isin(['ok', 'theta', 'other'])]
        cuts = {name: pd.cut(a[col], edges, right=False, labels=labels) for name, (col, (edges, labels)) in bins.items()}
        dh = [f'{d} reads, {h} het' for d in DEPTH_BINS[1] for h in HET_BINS[1]]
        cuts['depth by heterozygous donors'] = pd.Categorical(
            cuts['median total count of the gene'].astype(str) + ' reads, ' + cuts['heterozygous allelic donors'].astype(str) + ' het',
            categories=dh, ordered=True)
        for name, cut in cuts.items():
            for lab, x in a.groupby(cut, observed=False):
                rows.append(dict(run=run, by=name, bin=str(lab), variants=len(x), theta=float((x.cause == 'theta').mean()),
                                 other=float((x.cause == 'other').mean())))
    return pd.DataFrame(rows)


def few_het(S, V, U, D):
    """The few_het gate (fewer than MIN_N_HET heterozygous allelic donors): its share of all tested variants by the gene's
    zero-haplotype records and median total count, and at the reported leads the heterozygous donors asSeq counted
    (rebuilt from 05's allelic counts and checked against its n_ase_het) against those it would count if the records
    common.allelic_kept removes kept rint(pL), rint(pR)."""
    out = {'share': {}, 'leads': {}}
    for run, v in V.items():
        out['share'][run] = {name: {str(k): (int(len(x)), float((x.cause == 'few_het').mean()))
                                    for k, x in v.groupby(pd.cut(v[col], e, right=False, labels=lab), observed=False)}
                             for name, (col, (e, lab)) in (('zero-haplotype records of the gene', ('zero_hap', ZERO_BINS)),
                                                           ('median total count of the gene', ('median_total', DEPTH_BINS)))}
        L = leads(v, U, 'pval_nominal')
        L = L[L.cause == 'few_het']
        n, n_all = [], []
        for x in L.itertuples():
            ds, gi = D[x.scenario, x.rep], S['genes'].index(x.gene)
            Y1, Y2, _ = T.allelic_counts(ds)
            het = S['xLdf'].loc[x.lead_variant].values != S['xRdf'].loc[x.lead_variant].values
            n.append(int((het & (Y1[gi] + Y2[gi] >= T.MIN_AS_READS)).sum()))
            n_all.append(int((het & (np.rint(ds['pL'][gi]) + np.rint(ds['pR'][gi]) >= T.MIN_AS_READS)).sum()))
        if not np.array_equal(n, L.n_ase_het.astype(int).values):
            raise SystemExit(f'{run}: rebuilt heterozygous allelic donors differ from asSeq n_ase_het at the few_het leads')
        n_all = np.array(n_all, int)
        out['leads'][run] = dict(units=len(L), non_null=int((~L.is_null).sum()), reach_min_if_kept=int((n_all >= T.MIN_N_HET).sum()),
                                 reach_min_if_kept_non_null=int(((n_all >= T.MIN_N_HET) & ~L.is_null.values).sum()),
                                 het_median=float(np.median(n)) if len(n) else None, het_if_kept_median=float(np.median(n_all)) if len(n) else None)
    print(f'few_het: {out}', flush=True)
    return out


def null_tail(V, U, thr):
    """Null gene units of each |beta| whose lead p is at or below tensorQTL's own fdp_matched threshold there (thr), for
    each TReCASE test in both runs and for tensorQTL's own lead (the committed tensorqtl nominal files, same construction)."""
    tq = pd.concat([C.read_results(C.RESULTS / sc / C.TENSORQTL / f'nominal_rep{r:03d}.parquet',
                                   ['phenotype_id', 'variant_id', 'pval_nominal', 'slope', 'slope_se']).assign(scenario=sc, rep=r)
                    for sc, r in RUNS], ignore_index=True)
    arms = {'tensorqtl': leads(tq, U, 'pval_nominal')} | {f'{run} {c}': leads(v, U, c) for run, v in V.items() for c in COMPONENTS}
    out = {}
    for name, L in arms.items():
        for b in ('0.4', '0.8'):
            x, y = (L[(L.scenario == f'beta{b}') & (L.is_null == n)] for n in (True, False))
            out[f'{name} {b}'] = dict(null_units=len(x), at_or_below=int((x.lead_p <= thr[b]).sum()), threshold=thr[b],
                                      non_null_units=len(y), non_null_at_or_below=int((y.lead_p <= thr[b]).sum()))
    print(f'null tail: {out}', flush=True)
    return out


def spot_trace():
    """SPOT at trace = 2 on the committed inputs: per marker, the joint iterations run and whether theta failed; asSeq's
    rows for these markers must equal the committed full-gene rows (each marker's fits are independent of the others)."""
    sc, r, g, js = SPOT
    d, tag = OUT / 'spot', OUT / 'spot' / g
    d.mkdir(exist_ok=True)
    log = d / f'{g}_trace2.log'
    if not log.exists():
        with open(f'{log}.tmp', 'w') as fh:
            subprocess.run(['Rscript', '-e', SPOT_R, str(ddir(sc, r, COMMITTED_WORK)), str(COMMITTED_WORK / 'genotypes'), g,
                            str(tag), ','.join(map(str, js))], stdout=fh, stderr=subprocess.STDOUT, env={**os.environ, **T.R_ENV}, check=True)
        os.replace(f'{log}.tmp', log)
    mine = pd.read_csv(f'{tag}_eqtl.txt', sep='\t', dtype=str, keep_default_na=False)
    ref = pd.read_csv(ddir(sc, r, COMMITTED_WORK) / 'out' / f'{g}_eqtl.txt', sep='\t', dtype=str,
                      keep_default_na=False).set_index('MarkerRowID')
    ref = ref.loc[[str(j + 1) for j in js]].reset_index()
    cols = [c for c in mine.columns if c != 'MarkerRowID']
    stale = (ref.Joint_Pvalue == 'NA').values[:, None] & np.isin(cols, ['NBod', 'BBod'])[None, :]   # without a joint fit asSeq prints the previous marker's phi and theta (05 masks them)
    if not ((mine[cols].values == ref[cols].values) | stale).all():
        raise SystemExit(f'{tag}_eqtl.txt differs from the committed rows of these markers')
    text, rows = log.read_text(), []
    fails = {int(j): (float(gr), float(t)) for j, _, t, gr, _ in THETA_RE.findall(text)}
    for i, j in enumerate(js):
        its = [int(x) for x in re.findall(rf'i=0, j={i}, g=(\d+)\n', text)]
        rows.append(dict(marker=j, trec_b=float(ref.TReC_b.iloc[i]), ase_b=float(ref.ASE_b.iloc[i]),
                         n_het=int(ref.n_ASE_Het.iloc[i]), iterations=max(its) + 1 if its else 0,
                         result='theta failed' if i in fails else 'converged' if f'i=0, j={i}, converged using joint model' in text
                         else 'no joint fit', abs_gradient=abs(fails[i][0]) if i in fails else None,
                         joint_p=None if ref.Joint_Pvalue.iloc[i] == 'NA' else float(ref.Joint_Pvalue.iloc[i]),
                         reported_p=float(ref.final_Pvalue.iloc[i]), trec_p=float(ref.TReC_Pvalue.iloc[i])))
    print(f'spot trace {g} ({sc} rep {r}): {rows}', flush=True)
    return rows


def agreement(V):
    """The same tested variant in both runs: theta failure fractional x integer among variants whose joint fit passed
    asSeq's gates in both, and the largest change of -log10 p per component."""
    key = ['scenario', 'rep', 'phenotype_id', 'variant_id']
    m = V['fractional'][key + ['cause', *COMPONENTS]].merge(V['integer'][key + ['cause', *COMPONENTS]], on=key,
                                                             suffixes=('_f', '_i'), validate='one_to_one')
    if len(m) != len(V['fractional']) or len(m) != len(V['integer']):
        raise SystemExit(f'the two runs have {len(V["fractional"]):,} and {len(V["integer"]):,} variants, {len(m):,} shared')
    both = m[m.cause_f.isin(['ok', 'theta', 'other']) & m.cause_i.isin(['ok', 'theta', 'other'])]
    tab = pd.crosstab(both.cause_f, both.cause_i)
    lp = lambda x: -np.log10(np.clip(x, 1e-300, None))   # noqa: E731   (a p printed as 0 would be infinite)
    shift = {c: float(np.nanmax(np.abs(lp(m[f'{c}_i']) - lp(m[f'{c}_f'])))) for c in COMPONENTS}
    moved = np.abs(lp(m.pval_nominal_i) - lp(m.pval_nominal_f)) > 0.1
    res = dict(attempted_both=len(both), table={f: {i: int(tab.loc[f, i]) if f in tab.index and i in tab.columns else 0
                                                   for i in ('ok', 'theta', 'other')} for f in ('ok', 'theta', 'other')},
               max_abs_shift_neglog10=shift, reported_moved_over_0_1=int(moved.sum()),
               reported_moved_with_joint_flip=int((moved & ((m.cause_f == 'ok') != (m.cause_i == 'ok'))).sum()), variants=len(m))
    print(f'agreement: {res}', flush=True)
    return res


def main():
    S, ids, chrom, expected, D = prepare()
    control(S)
    run_genes(S)
    counts = convert(S, ids, chrom, expected, D)
    genes = S['genes']
    meta, _, U, keep_a = SC.load_units(C.DATASETS, C.RESULTS)
    U = U[[(sc, r) in RUNS for sc, r in zip(U.scenario, U.rep)]][['scenario', 'rep', 'gene', 'is_null', 'band']]
    bsel, bidx = SC.band_selections(genes, U, keep_a)
    V, F = {}, {}
    for run in ('fractional', 'integer'):
        V[run], F[run] = variants(S, ids, chrom, D, run)
    committed = json.loads(C.SUMMARY.read_text())
    power = score(V, U, genes)
    for b in ('0.4', '0.8'):   # the committed reported p must reproduce 06's fdp_matched
        ref = committed['ranking'][f'beta{b}']['trecase']['fdp_matched']
        if (power[f'pval_nominal {b}']['fractional_power'], power[f'pval_nominal {b}']['fractional_discoveries']) != (
                ref['all']['power'], ref['discoveries']):
            raise SystemExit(f'|beta| {b}: the committed reported-p power does not reproduce {C.SUMMARY}')
    cols = {c: c for c in COMPONENTS}
    null = {run: SC.null_calibration(res, U, 'beta0.0', arm, genes, bsel, bidx, cols)
            for run, res, arm in (('fractional', C.RESULTS, 'trecase'), ('integer', RESULTS, ARM))}
    if null['fractional']['pval_t']['all']['0.05']['rate'] != committed['trecase_components']['trec']['all']['0.05']['rate']:
        raise SystemExit(f'the committed anchor TReC rate does not reproduce {C.SUMMARY}')
    lf, conc = lead_failures(V, U), concentration(V)
    grad = {run: dict(n=len(f), abs_gradient_quantiles=np.quantile(np.abs(f.gradient), [0.5, 0.9, 0.99, 1]).tolist(),
                      theta_median=float(f.theta.median()), fail_codes={str(k): int(v) for k, v in f.fail.value_counts().items()})
            for run, f in F.items()}
    for run, v in V.items():   # asSeq's joint theta where the joint fit converged, beside theta at a failure
        grad[run]['theta_median_converged'] = float(np.median(v.loc[v.joint_ok, 'bb_od']))
    thr = {a: {b: committed['ranking'][f'beta{b}'][a]['fdp_matched']['p_threshold'] for b in ('0.4', '0.8')} for a in ('tensorqtl', 'split')}
    R = dict(runs=RUNS, fdr=SC.FDR, n_boot=SC.N_BOOT, seed=C.SEED, committed_results=str(C.JOINT['trecase']),
             committed_work=str(COMMITTED_WORK), counts=counts, power=power,
             null={run: {c: {str(al): v['all'][str(al)] for al in C.ALPHAS} for c, v in x.items()} for run, x in null.items()},
             causes={run: {c: int((v.cause == c).sum()) for c in ('ok',) + CAUSES} for run, v in V.items()},
             lead_failures=lf.to_dict('records'), concentration=conc.to_dict('records'), theta_failures=grad,
             agreement=agreement(V), few_het=few_het(S, V, U, D), spot=spot_trace(),
             reference={a: {b: committed['ranking'][f'beta{b}'][a]['fdp_matched']['all']['power'] for b in ('0.4', '0.8')}
                        for a in ('tensorqtl', 'split', 'trecase', 'rasqual')},
             reference_p_threshold=thr, null_tail=null_tail(V, U, thr['tensorqtl']),
             reference_null={a: committed['null']['beta0.0'][a]['combined']['all'] for a in ('tensorqtl', 'split')})
    C.write_json(OUT / 'summary.json', R)
    print(f'wrote {OUT / "summary.json"}', flush=True)
    report(R, figure(R))


def table(head, rows):
    import html
    h = ''.join(f'<th>{html.escape(str(x))}</th>' for x in head)
    return (f'<table><tr>{h}</tr>' + ''.join('<tr>' + ''.join(f'<td>{html.escape(str(x))}</td>' for x in r) + '</tr>' for r in rows)
            + '</table>')


def tables(R):
    """The page's tables, as HTML strings keyed by name."""
    P, N = R['power'], R['null']
    t = {}
    t['power'] = table(['test', '|beta|', 'fractional Y (committed)', 'integer Y (this run)', 'integer minus fractional [95% interval]'],
                       [[COMPONENT_LABELS[v['component']], v['beta_abs'],
                         f'{v["fractional_power"]:.3f} ({v["fractional_discoveries"]} called, {v["fractional_false"]} null; p <= {v["fractional_p_threshold"]:.2g})',
                         f'{v["integer_power"]:.3f} ({v["integer_discoveries"]} called, {v["integer_false"]} null; p <= {v["integer_p_threshold"]:.2g})',
                         f'{v["diff"]:+.3f} [{v["diff_lo"]:+.3f}, {v["diff_hi"]:+.3f}]'] for v in P.values()])
    ci = lambda x: f'{x["rate"]:.4f} [{x["lo"]:.4f}, {x["hi"]:.4f}]'   # noqa: E731
    t['null'] = table(['test', 'run', 'tests', 'rate at 0.05', 'rate at 0.001'],
                      [[COMPONENT_LABELS[c], run, f'{N[run][c]["0.05"]["tests"]:,}', ci(N[run][c]['0.05']), ci(N[run][c]['0.001'])]
                       for c in COMPONENTS for run in N]
                      + [[a, 'committed benchmark', f'{x["0.05"]["tests"]:,}', ci(x['0.05']), ci(x['0.001'])]
                         for a, x in R['reference_null'].items()])
    lf = pd.DataFrame(R['lead_failures'])
    t['lead'] = table(['gene units', 'run', 'units', 'no joint p', *CAUSES],
                      [[r.group, r.run, r.units, f'{r.no_joint:.3f}', *(getattr(r, c) for c in CAUSES)]
                       for r in lf.sort_values(['group', 'run']).itertuples()])
    t['causes'] = table(['run', 'tested variants', 'joint p', *CAUSES],
                        [[run, f'{sum(c.values()):,}', *(f'{c[k]:,} ({c[k] / sum(c.values()):.3f})' for k in ('ok',) + CAUSES)]
                         for run, c in R['causes'].items()])
    cc = pd.DataFrame(R['concentration'])
    fr, it = (cc[cc.run == run].set_index(['by', 'bin']) for run in ('fractional', 'integer'))
    t['concentration'] = table(['by', 'bin', 'variants (integer run)', 'theta failure, fractional', 'theta failure, integer',
                                'other, fractional', 'other, integer'],
                               [[by, b, f'{it.loc[(by, b), "variants"]:,}', f'{r.theta:.3f}', f'{it.loc[(by, b), "theta"]:.3f}',
                                 f'{r.other:.3f}', f'{it.loc[(by, b), "other"]:.3f}'] for (by, b), r in fr.iterrows()])
    nt = R['null_tail']
    cnt = lambda x: (f'{x["at_or_below"]} of {x["null_units"]} null, {x["non_null_at_or_below"]} of '   # noqa: E731
                     f'{x["non_null_units"]} non-null')
    t['tail'] = table(['lead p of', 'run', *(f'|beta| {b}: units at or below tensorQTL threshold p <= {nt[f"tensorqtl {b}"]["threshold"]:.2g}'
                                             for b in ('0.4', '0.8'))],
                      [['tensorQTL (total only)', 'committed benchmark', *(cnt(nt[f'tensorqtl {b}']) for b in ('0.4', '0.8'))]]
                      + [[COMPONENT_LABELS[c], run, *(cnt(nt[f'{run} {c} {b}']) for b in ('0.4', '0.8'))]
                         for c in COMPONENTS for run in ('fractional', 'integer')])
    fh = R['few_het']
    t['few_het'] = table(['run', 'reported leads behind the few_het gate', 'non-null among them',
                          'would reach 5 heterozygous allelic donors with the removed records', 'non-null among those',
                          'median heterozygous allelic donors: asSeq / with removed records'],
                         [[run, x['units'], x['non_null'], x['reach_min_if_kept'], x['reach_min_if_kept_non_null'],
                           f'{x["het_median"]:g} / {x["het_if_kept_median"]:g}'] for run, x in fh['leads'].items()])
    t['few_het_share'] = table(['by', 'bin', 'tested variants', 'few_het share, fractional', 'few_het share, integer'],
                               [[by, b, f'{n:,}', f'{s:.3f}', f'{fh["share"]["integer"][by][b][1]:.3f}']
                                for by, x in fh['share']['fractional'].items() for b, (n, s) in x.items()])
    A = R['agreement']['table']
    t['agreement'] = table(['fractional run (rows) by integer run (columns)', 'joint p', 'theta failure', 'other'],
                           [[f, *(f'{A[f][i]:,}' for i in ('ok', 'theta', 'other'))] for f in ('ok', 'theta', 'other')])
    t['spot'] = table(['marker row', 'TReC-only slope (ln)', 'ASE-only slope (ln)', 'heterozygous allelic donors',
                       'joint iterations started', 'result', '|gradient| at failure', 'joint p', 'reported p', 'TReC p'],
                      [[s['marker'], f'{s["trec_b"]:+.3f}', f'{s["ase_b"]:+.3f}', s['n_het'], s['iterations'], s['result'],
                        '' if s['abs_gradient'] is None else f'{s["abs_gradient"]:.1e}',
                        'NA' if s['joint_p'] is None else f'{s["joint_p"]:.3g}', f'{s["reported_p"]:.3g}', f'{s["trec_p"]:.3g}']
                       for s in R['spot']])
    return t


CSS = ''':root { --ink: #0b0b0b; --ink2: #52514e; --rule: #e1e0d9; --bg: #fcfcfb; --tint: #f3f2ee; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --ink: #ecebe6; --ink2: #a9a79f; --rule: #3a3935;
  --bg: #161614; --tint: #22211e; } }
:root[data-theme="dark"] { --ink: #ecebe6; --ink2: #a9a79f; --rule: #3a3935; --bg: #161614; --tint: #22211e; }
body { background: var(--bg); color: var(--ink); font: 15px/1.55 -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
       margin: 0; padding: 0 16px; }
main { max-width: 1080px; margin: 32px auto 64px; } h1 { font-size: 26px; margin-bottom: 4px; }
h2 { font-size: 20px; margin-top: 36px; border-bottom: 1px solid var(--rule); padding-bottom: 4px; }
p { max-width: 900px; } .sub { color: var(--ink2); margin-top: 0; }
table { border-collapse: collapse; font-size: 12.5px; margin: 12px 0 18px; display: block; overflow-x: auto; }
th, td { border-bottom: 1px solid var(--rule); padding: 4px 8px; text-align: left; vertical-align: top; }
th { background: var(--tint); font-weight: 600; } td { font-variant-numeric: tabular-nums; }
figure { margin: 16px 0 24px; } figure img { max-width: 100%; height: auto; background: #fff; }
figcaption { color: var(--ink2); font-size: 13px; max-width: 900px; } code { font-size: 13px; }
'''


def report(R, fig):
    """OUT/report.html: the page, its numbers from R (summary.json)."""
    import base64
    import html
    t, P, N, G = tables(R), R['power'], R['null'], R['theta_failures']
    p = lambda c, b, run: P[f'{c} {b}'][f'{run}_power']   # noqa: E731
    lf = pd.DataFrame(R['lead_failures']).set_index(['run', 'group'])
    q = G['integer']['abs_gradient_quantiles']
    cap = ('Left and middle: power at 5% realized false-discovery proportion of each TReCASE test, committed run '
           '(fractional total count) and this run (integer), with total-only tensorQTL (dashed) and hapmixQTL split '
           '(dotted) from the committed benchmark. Right: share of variants whose joint fit was attempted and stopped at '
           "the theta update, by the gene's median total count, pooled over the seven datasets.")
    img = (f'<figure><img alt="{html.escape(cap, quote=True)}" src="data:image/png;base64,'
           f'{base64.b64encode(fig.read_bytes()).decode()}"><figcaption>{html.escape(cap)}</figcaption></figure>')
    ref, refp, rn, A, ag = R['reference'], R['reference_p_threshold'], R['reference_null'], R['agreement']['table'], R['agreement']
    k = lambda f, i: A[f][i]   # noqa: E731
    cz = pd.DataFrame(R['concentration']).set_index(['run', 'by', 'bin'])
    cth = lambda run, by, b: cz.loc[(run, by, b), 'theta']   # noqa: E731
    nfl = lambda run, grp: lf.loc[(run, grp), 'no_joint']   # noqa: E731
    sp = {s['marker']: s for s in R['spot']}
    opposite = [s['marker'] for s in R['spot'] if s['trec_b'] * s['ase_b'] < 0]
    if opposite != list(sp) or [m for m, s in sp.items() if s['result'] == 'converged'] != [8142]:
        raise SystemExit(f'spot trace: opposite-sign markers {opposite}, results {[s["result"] for s in R["spot"]]}; '
                         'the ABCC4 paragraph no longer holds')
    nt, fh, cs, sh = R['null_tail'], R['few_het']['leads'], R['causes'], ag['max_abs_shift_neglog10']
    att = {run: c['ok'] + c['theta'] + c['other'] for run, c in cs.items()}
    dmax = max(abs(v['diff']) for v in P.values())
    inside = all(v['diff_lo'] <= 0 <= v['diff_hi'] for v in P.values())
    lead_nn = {run: sum(lf.loc[(run, g), 'units'] * lf.loc[(run, g), 'no_joint'] for g in ('|beta| 0.4 non-null', '|beta| 0.8 non-null'))
               for run in ('fractional', 'integer')}
    result_text = f'''<p><b>Rounding the totals {"leaves every test's power at 5% false-discovery proportion within its paired interval of no change" if inside else "moves at least one test's power outside its paired interval of no change (table)"}</b>:
the largest change in power over the four tests and two effect sizes is {dmax:.3f}, where one non-null unit of 150 is
0.007. Over all {ag['variants']:,} tested variants the TReC p moved by at most {sh['pval_t']:.3f} in -log10 p and the ASE p by
{sh['pval_a']:.3f}, whose inputs did not change. The joint p, where both runs have one, moved by at most {sh['pval_joint']:.3f}.
The reported p moved by more than 0.1 in -log10 p at {ag['reported_moved_over_0_1']:,} variants, and at
{ag['reported_moved_with_joint_flip']:,} of them because the joint fit was present in one run and missing in the other
(next section), which can move it by up to {sh['pval_nominal']:.1f}. Each entry below gives power, the units called and how
many of them are null, and the p the cut falls at.</p>'''
    tail_same = all(nt[f'integer pval_t {b}']['non_null_at_or_below'] >= nt[f'tensorqtl {b}']['non_null_at_or_below'] for b in ('0.4', '0.8'))
    tail_text = f'''<p>On the anchor TReC's null rate with integer totals is {N['integer']['pval_t']['0.05']['rate']:.4f} at 0.05 and
{N['integer']['pval_t']['0.001']['rate']:.4f} at 0.001 ({N['fractional']['pval_t']['0.05']['rate']:.4f} and
{N['fractional']['pval_t']['0.001']['rate']:.4f} with fractional totals), against tensorQTL's
{rn['tensorqtl']['0.05']['rate']:.4f} and {rn['tensorqtl']['0.001']['rate']:.4f} on the same totals. The matched
false-discovery proportion is decided in the tail of the null genes' lead p, the smallest of several thousand tested
variants per gene. At |beta| 0.4, TReC must stop at p &le; {P['pval_t 0.4']['integer_p_threshold']:.2g} to hold 5%, where
tensorQTL stops at {refp['tensorqtl']['0.4']:.2g}. Counted at tensorQTL's own threshold, TReC puts
{nt['integer pval_t 0.4']['at_or_below']} of the 150 null units at |beta| 0.4 at or below it, against
{nt['tensorqtl 0.4']['at_or_below']} for tensorQTL, and {nt['integer pval_t 0.4']['non_null_at_or_below']} of the 150 non-null
units, against {nt['tensorqtl 0.4']['non_null_at_or_below']} ({nt['integer pval_t 0.8']['at_or_below']} against
{nt['tensorqtl 0.8']['at_or_below']} null and {nt['integer pval_t 0.8']['non_null_at_or_below']} against
{nt['tensorqtl 0.8']['non_null_at_or_below']} non-null at |beta| 0.8). <b>{TAIL_VERDICT[tail_same]}</b> For the reported p
the null count at |beta| 0.4 is {nt['integer pval_nominal 0.4']['at_or_below']}.</p>'''
    codes = ', '.join(f'{c} ({n:,} lines)' for c, n in G['integer']['fail_codes'].items())
    failure_text = f'''<p><b>What fails is the beta-binomial overdispersion update inside the joint fit.</b> With integer totals
{cs['integer']['theta']:,} of the {att['integer']:,} variants whose joint fit passed asSeq's gates stopped at the theta
update ({cs['integer']['theta'] / att['integer']:.1%}; {cs['fractional']['theta'] / att['fractional']:.1%} with fractional
totals), and the trace lines carry L-BFGS-B fail codes {codes}. The joint fit alternates
three updates (trecase.c:916-1086). b is updated by Newton-Raphson (b_ml, trecase.c:257-338), which steps b toward the root
of the log-likelihood's first derivative using its second derivative. theta is updated by L-BFGS-B, a limited-memory
quasi-Newton optimizer with box constraints, with the allelic proportion held fixed. phi comes from the negative-binomial
refit. Any nonzero code from the theta step abandons the joint model (trecase.c:1003-1012), and asSeq then reports the TReC
p. Code 52 is what asSeq's L-BFGS-B driver returns for a task string beginning ERROR, or for any task string it does not
recognize (lbfgsb1.c:4192-4197); the trace does not print which. The ERROR strings that check the inputs cannot arise for
this fixed one-parameter bounded problem. That leaves ABNORMAL_TERMINATION_IN_LNSRCH, a line search that found no lower
objective (lbfgsb1.c:913-927), as the reading. At the failure theta's gradient is small: |gradient| median {q[0]:.1e}, 90th percentile {q[1]:.1e},
maximum {q[3]:.1e} over {G['integer']['n']:,} failures. Theta at a failure has median {G['integer']['theta_median']:.3f},
against {G['integer']['theta_median_converged']:.3f} where the joint fit converged.</p>
<p>The following is a reading of the source, not a measurement. asSeq sets the projected-gradient tolerance pgtol to 0
(trecase.c:459), which disables L-BFGS-B's exit for a start that is already at the optimum (lbfgsb1.c:776). So when theta
barely needs to move, the line search must find a decrease the objective's floating-point resolution may not show. The
depth gradient below fits this reading, since a deeper gene's allelic log-likelihood is larger in magnitude, but the
curvature was not measured.</p>
<p>A run at asSeq's trace level 2 on four markers of ABCC4 (|beta| 0.4 rep 0, committed inputs; its rows reproduce the
committed ones) shows where the failure falls. At marker 8141 the TReC-only and ASE-only effects have opposite signs
({sp[8141]['trec_b']:+.3f} and {sp[8141]['ase_b']:+.3f}, natural-log scale), the joint b creeps between them, and theta
fails at the fourth iteration with |gradient| {sp[8141]['abs_gradient']:.1e}. Marker 8151 fails at the first, with
|gradient| {sp[8151]['abs_gradient']:.1e}. At each failure the reported p is the TReC p. The sign disagreement does not
separate failures from successes: all {len(sp)} traced markers have TReC-only and ASE-only slopes of opposite sign, and
marker 8142 ({sp[8142]['trec_b']:+.3f} and {sp[8142]['ase_b']:+.3f}) converges after {sp[8142]['iterations']} joint
iterations while the other three fail.</p>'''
    kt, ko = (k(f, 'ok') + k(f, 'theta') + k(f, 'other') for f in ('theta', 'ok'))
    agree_text = f'''<p><b>Which variants fail is mostly not a property of the variant.</b> Rounding the totals moved the TReC
p by at most {sh['pval_t']:.3f} in -log10 p and left the theta-failure rate where it was, yet it changed which variants
fail. Of the {kt:,} variants that failed at theta in the committed run and passed asSeq's gates in both runs,
{k('theta', 'theta'):,} ({k('theta', 'theta') / kt:.0%}) fail again with integer totals. Of the {ko:,} that had a joint p,
{k('ok', 'theta'):,} ({k('ok', 'theta') / ko:.0%}) now fail. A committed failure predicts a repeat
{k('theta', 'theta') / kt / (k('ok', 'theta') / ko):.1f} times as often as a committed success does, so the variant
carries some of it, but most of it moves. The reported leads move with it: the share of non-null leads without a joint p
goes from {nfl('fractional', '|beta| 0.4 non-null'):.3f} to {nfl('integer', '|beta| 0.4 non-null'):.3f} at |beta| 0.4 and
from {nfl('fractional', '|beta| 0.8 non-null'):.3f} to {nfl('integer', '|beta| 0.8 non-null'):.3f} at 0.8 (table above).
Rows: committed (fractional) run; columns: integer run.</p>'''
    conc_text = f'''<p>Among variants whose joint fit passed asSeq's gates, theta failure rises with the gene's depth, from
{cth('integer', 'median total count of the gene', '<100'):.3f} below a median total count of 100 reads to
{cth('integer', 'median total count of the gene', '>=2000'):.3f} at 2,000 or more (figure, right). At fixed depth it does
not change with the number of heterozygous allelic donors (the depth-by-heterozygous-donors rows), so the marginal rise with
heterozygous donors is depth. It falls with the gene's zero-haplotype records, which are commonest in shallow genes, so that
trend is depth too. It is highest where the allelic evidence at the variant is weakest: allelic-only chi-square below 1
{cth('integer', 'allelic-only chi-square', '<1'):.3f}, against {cth('integer', 'allelic-only chi-square', '1-4'):.3f} at
1-4. That fits theta starting at its optimum when the allelic proportion hardly moves.</p>'''
    few_text = f'''<p><b>The few_het gate is a separate cause, and the one the benchmark's input handling reaches.</b> asSeq never
attempts the joint model at a variant with fewer than 5 heterozygous donors among those with at least 5 allele-specific
reads. The benchmark removes, before asSeq, every donor record with exactly one haplotype below 0.5 reads or without Gibbs
variance (common.allelic_kept). At the reported leads behind this gate, keeping those records at rint(pL), rint(pR) would
bring {fh['integer']['reach_min_if_kept']} of {fh['integer']['units']} units to 5 heterozygous allelic donors, and
{fh['integer']['reach_min_if_kept_non_null']} of the {fh['integer']['non_null']} non-null ones. The heterozygous donor
counts are rebuilt from 05's allelic counts and match asSeq's own at every such lead.</p>'''
    critique_text = f'''<p>The strongest objection to the rounding test is that it is small. It moves each total by at most
half a read, so it tests only whether asSeq's negative binomial mishandles non-integer values. It does not test whether the
thinned Salmon totals, which are expected counts after multi-mapping reallocation, have the negative-binomial variance the
model assumes. That broader question is open, and TReC's null rates above are the evidence that something in it does not
hold on these data. Second, "joint (where fitted)" is scored on an evidence-selected set. The fit fails more where the
allelic evidence is weak ({cth('integer', 'allelic-only chi-square', '<1'):.3f} below chi-square 1 against
{cth('integer', 'allelic-only chi-square', '1-4'):.3f} at 1-4). Each gene's lead joint p is therefore a minimum over
variants enriched for stronger allelic evidence, for null and non-null genes alike, and its power
({p('pval_joint', '0.4', 'integer'):.3f} at |beta| 0.4) is not what a repaired joint fit would give. Third, each |beta|
has 150 non-null and 150 null units from three datasets, the intervals are gene-clustered, and the anchor is one dataset.
Fourth, the theta mechanism rests on the source, the gradients and one gene's trace, not on a measured curvature. Not
tested: asSeq with pgtol above zero or with code 52 at a small gradient accepted as convergence; asSeq given the records
the admission rule removes.</p>'''
    meaning_text = f'''<p>Fractional totals are not why TReCASE ranks below tensorQTL on this benchmark. The deficit has two
measured parts, and neither is the count format. First, TReC's null genes reach lead p values far below tensorQTL's on the
same totals: at tensorQTL's own threshold at |beta| 0.4, {nt['integer pval_t 0.4']['at_or_below']} null units against
{nt['tensorqtl 0.4']['at_or_below']}, with {nt['integer pval_t 0.4']['non_null_at_or_below']} non-null units against
{nt['tensorqtl 0.4']['non_null_at_or_below']}. Second, asSeq's joint fit stops at the theta update at
{cs['integer']['theta'] / att['integer']:.0%} of the variants it attempts, most often in deep genes, and the reported p then
falls back to the TReC p. At the reported leads of non-null genes, {lead_nn['integer']:.0f} of 300 units have no joint p for
any reason. Rounding reshuffled which variants fail without changing how many, and the reported p's power stayed at
{p('pval_nominal', '0.4', 'integer'):.3f} and {p('pval_nominal', '0.8', 'integer'):.3f}. The benchmark's input handling reaches TReCASE through the allelic admission rule instead: removing
zero-haplotype records holds {fh['integer']['reach_min_if_kept_non_null']} non-null leads below asSeq's heterozygote minimum
that they would otherwise pass. For the fairness question this moves the concern from the count format to TReC's
calibration on these data and to asSeq's optimizer settings, and it leaves the admission rule as a small, measured input
effect.</p>'''
    body = f'''<h1>TReCASE on integer total counts</h1>
<p class="sub">Deep plasmode set (plasmode_meier_20260927), 2026-09-28. Script <code>scripts/trecase_input_diagnosis.py</code>;
every number on this page is in <code>summary.json</code> beside it.</p>

<h2>Why this was run</h2>
<p>On the deep plasmode set TReCASE ranks non-null genes below total-only tensorQTL at a matched 5% false-discovery
proportion. At |beta| = 0.4 its reported p reaches power {ref['trecase']['0.4']:.3f}, against {ref['tensorqtl']['0.4']:.3f}
for tensorQTL on the same totals and {ref['split']['0.4']:.3f} for hapmixQTL's split weighting (inverse Gibbs variance
weights in the allelic channel, unit weights in the total channel), and its total-count test on
its own reaches only {p('pval_t', '0.4', 'fractional'):.3f}. The benchmark passes asSeq's negative-binomial total-count model
the thinned Salmon totals as fractional numbers (<code>05_run_trecase.py</code> rounds only the allelic counts, which asSeq's
beta-binomial requires to be integers). A negative-binomial likelihood evaluated at a non-integer count is still defined,
since asSeq computes it through the log-gamma function, but it is not the input the model was written for. This run asks
whether rounding the totals changes TReCASE's standing, and why its joint fit is missing at a quarter to a third of the
reported leads.</p>

<h2>What was run</h2>
<p>asSeq 0.99.501's <code>trecase</code> was run per gene exactly as <code>05_run_trecase.py</code> and
<code>run_trecase.R</code> call it, with one change: the total count is rint(pT) instead of pT. Every other input
(the allelic counts, the 17 covariates, the log effective library size offset, and the genotype codes) was rebuilt with
05's own functions and is byte-identical to the committed run's input files. The rebuilt fractional total file is also
byte-identical to the committed one, and the integer total file is its rounding. One gene rerun on the unchanged inputs
(TCAP, |beta| 0.4 rep 0) reproduced the committed asSeq output byte for byte, so the software and environment are
the committed run's. Rounding moved a total by at most 0.5 reads: mean 0.14 to 0.21 reads per donor-gene pair,
with 71% to 85% of totals fractional before rounding. The datasets are the |beta| = 0 anchor (rep 0, 100 null genes) and
reps 0-2 at |beta| 0.4 and 0.8, each with 50 null and 50 non-null genes: 700 gene runs and
{ag['variants']:,} tested variants per run.</p>
<p>TReCASE reports three likelihood-ratio tests per variant. The total-count (TReC) test is a negative-binomial regression
of the total count on genotype dosage with the covariates and offset. The allele-specific (ASE) test is a beta-binomial
model of one haplotype's reads out of a donor's allele-specific reads, using donors with at least 5 such reads, in which the
allelic proportion departs from one half only in heterozygous donors. The joint test fits both likelihoods with one shared
cis effect b, alternating updates of b, the beta-binomial overdispersion theta and the negative-binomial dispersion phi.
The reported p (pval_nominal) is the joint p unless the joint fit is missing or a cis-versus-trans test (a likelihood-ratio
test of whether the TReC and ASE effects differ) rejects at 0.05; in either case it is the TReC p. <b>Power at 5%
false-discovery proportion</b> is <code>06_score.py</code>'s fdp_matched. The 300 gene units (dataset by gene) of one
|beta| are ranked by each gene's lead p, the smallest p over its tested variants with ties broken by the larger
|slope / se|. The ranking is cut at the deepest point where at most 5% of the units called are null genes, and power is the
share of the 150 non-null units called. The interval on the integer-minus-fractional difference resamples the 100 genes with
replacement 2,000 times, each gene carrying its units in both runs (a paired gene-clustered interval).</p>

<h2>Result</h2>
{result_text}
{t['power']}
{img}
<p>On the anchor, the null-gene rate is the share of tested variants in null genes with p below the level, over variants
with a finite p; the interval resamples genes.</p>
{t['null']}
{tail_text}
{t['tail']}

<h2>Where the joint p goes missing</h2>
<p>At each gene unit's reported lead (its smallest pval_nominal), the share without a joint p, and why. The causes are
assigned per variant from asSeq's output and trace, first applicable cause first. ase_baseline: asSeq's null allelic model
failed for the whole gene. trec: no TReC p. few_het: fewer than 5 heterozygous allelic donors, so asSeq never attempts the
joint model (its min.n.het). ase_fit: the allelic fit under the alternative failed. theta: the joint fit stopped at the theta
update. other: the joint model was skipped or stopped without a trace line (a TReC refit on linear dosage, a b or phi
failure, or the iteration limit).</p>
{t['lead']}
<p>The same causes over every tested variant:</p>
{t['causes']}
{failure_text}
{t['spot']}
{agree_text}
{t['agreement']}
{conc_text}
{t['concentration']}
{few_text}
{t['few_het']}
{t['few_het_share']}

<h2>Critique and limits</h2>
{critique_text}

<h2>What it means</h2>
{meaning_text}
'''
    page = (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" '
            f'content="width=device-width, initial-scale=1"><title>TReCASE integer totals</title><style>{CSS}</style>'
            f'</head><body><main>{body}</main></body></html>')
    C.write_atomic(OUT / 'report.html', lambda fh: fh.write(page), 'w')
    print(f'wrote {OUT / "report.html"}', flush=True)


def figure(R):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 3.9))
    x = np.arange(len(COMPONENTS))
    for ax, b in zip(axs[:2], ('0.4', '0.8')):
        for k, run in enumerate(RUN_COLORS):
            ax.bar(x + (k - 0.5) * 0.38, [R['power'][f'{c} {b}'][f'{run}_power'] for c in COMPONENTS], 0.36,
                   color=RUN_COLORS[run], label=f'TReCASE, {run} total count')
        for a, ls, name in (('tensorqtl', '--', 'tensorQTL, total only'), ('split', ':', 'hapmixQTL split')):
            ax.axhline(R['reference'][a][b], color='#0b0b0b', ls=ls, lw=1.2, label=f'{name} (committed benchmark)')
        ax.set_xticks(x, ['TReC', 'ASE', 'joint\n(where fitted)', 'reported'], fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_title(f'|beta| = {b}', fontsize=10)
    axs[0].legend(fontsize=8, frameon=False, loc='upper left')
    axs[0].set_ylabel('power at 5% false-discovery proportion', fontsize=9)
    conc = pd.DataFrame(R['concentration'])
    for run in RUN_COLORS:
        c = conc[(conc.run == run) & (conc.by == 'median total count of the gene')]
        axs[2].plot(c.bin, c.theta, 'o-', color=RUN_COLORS[run], label=run)
    axs[2].set_ylim(0, None)
    axs[2].set_xlabel("gene's median total count (reads)", fontsize=9)
    axs[2].set_title('theta failure among attempted joint fits', fontsize=10)
    axs[2].legend(fontsize=8, frameon=False)
    for ax in axs:
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
    path = OUT / 'fig_trecase_integer.png'
    fig.savefig(f'{path}.tmp', dpi=130, bbox_inches='tight', format='png')
    plt.close(fig)
    os.replace(f'{path}.tmp', path)
    return path


if __name__ == '__main__':
    main()

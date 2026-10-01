"""RASQUAL input diagnosis on the deep plasmode set (task of 2026-09-28): is RASQUAL's weak gene ranking there its
total-count model on our Salmon totals, or the pseudo feature SNP's estimated error rate delta?

Two variants of the committed RASQUAL arm (benchmark/plasmode/04_run_rasqual.py), each 04's command with ONE flag added
and everything else identical (the same VCF text, pseudo feature SNP included, and Y / K / X binaries checked byte for
byte against the committed arm's):
  population_only  --population-only: main.c:216 sets ASE = 0, main.c:516 then admits no feature SNP, so the allelic
                   null (nbem.c:110) and the allelic alternative (nbem.c:302) never run: RASQUAL's negative-binomial
                   model of the total counts alone, with its genotype prior.
  fix_delta        --fix-delta: the full model with the sequencing / mapping error rate delta fixed at 0.01 (usage.c;
                   nbem.c:622, the mode (ad - 1) / (ad + bd - 2) of its Beta(1.01, 1.99) prior), RASQUAL's documented
                   fixed value; it has no option that fixes delta at zero.
Datasets RUNS of the deep set (common.ROOT, PLASMODE_GENE_SET unset). Per gene a raw checkpoint (a gene whose raw file
exists is not rerun); assembled with 04's assemble (converged tested rows; pseudo row, non-converged and absent rows
counted).

Scored against the committed RASQUAL arm (common.JOINT['rasqual']) and total-only tensorQTL (results/tensorqtl) through
06_score: lead = smallest nominal p per (dataset, gene), ties by |slope / se| (06 causal_and_leads); power at 5% realized
false-discovery proportion and AUC (06 ranking, gene units pooled over the scenario's datasets), and each arm's power
minus tensorqtl's and minus the committed rasqual's with a paired gene-clustered interval (paired_diff); null-gene
nominal-p rate on the beta = 0 anchor at ALPHAS with 06's gene-clustered interval (06 pooled); non-converged share of
tested rows; per |beta|, the null and non-null units whose lead p is at or below tensorqtl's own fdp_matched threshold
(null tail), beside the committed TReCASE arm's total-count test read from TRECASE_SUMMARY (its threshold and tensorqtl
counts must equal this script's).
The committed arms are recomputed through the same code and must reproduce common.SUMMARY.

Output (OUT): <variant>/<scenario>/rasqual/{raw_repNNN/, nominal_repNNN.parquet}, <variant>/summary.json, inputs/,
scores.json, report.html (one figure, embedded). summary.json's seconds_per_gene_run counts only the gene runs the
invocation that wrote it made (n = 0 when every raw file already existed); score() and page() alone rebuild the page.
"""
import base64
import concurrent.futures as cf
import hashlib
import html
import io
import json
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
matplotlib.use('Agg')
import matplotlib.pyplot as plt   # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'benchmark' / 'plasmode'))
import common as C                # noqa: E402

M4 = C.module('04_run_rasqual')
S6 = C.module('06_score')

OUT = C.D / 'input_diagnosis_20260928' / 'rasqual_total_only'   # task 2026-09-28
VARIANTS = {'population_only': ['--population-only'], 'fix_delta': ['--fix-delta']}   # usage.c, see the docstring
FIXED_DELTA = 0.01                # nbem.c:622 with main.c's ad = 1.01, bd = 1.99
RUNS = [('beta0.0', 0)] + [(f'beta{b}', r) for b in ('0.4', '0.8') for r in range(3)]   # task 2026-09-28
JOBS = 32                         # RASQUAL processes (task cap)
OLD_INPUTS = C.COMMITTED_JOINT['rasqual']   # the committed arm's own run (plasmode_20260926): its Y / K / X binaries
ALPHAS = (0.05, 0.001)            # anchor rates reported (task)
DIFF_KEY = 61                     # SeedSequence spawn key of the paired power-difference interval (06: 30, 33; 02, 03: 1-5)
TRECASE_SUMMARY = C.D / 'input_diagnosis_20260928' / 'trecase_integer' / 'summary.json'   # trecase_input_diagnosis.py: its null_tail
ARMS = ('rasqual', 'population_only', 'fix_delta', 'tensorqtl')
LABEL = {'rasqual': 'RASQUAL, committed (full model)', 'population_only': 'RASQUAL --population-only',
         'fix_delta': 'RASQUAL --fix-delta (delta = 0.01)', 'tensorqtl': 'tensorQTL, total only'}
COLOR = {'rasqual': '#4a3aa7', 'population_only': '#e87ba4', 'fix_delta': '#1baf7a', 'tensorqtl': '#8a5a2b'}


def nominal_path(arm, sc, r):
    base = {'rasqual': C.JOINT['rasqual'] / sc / 'rasqual', 'tensorqtl': C.RESULTS / sc / 'tensorqtl'}.get(
        arm, OUT / arm / sc / 'rasqual')
    return base / f'nominal_rep{r:03d}.parquet'


def run_gene(k, g, site, text, bins, n, raw, flags):
    """04's run_gene with `flags` appended to its command."""
    if raw.exists():
        out, secs, skipped = raw.read_text(), 0.0, True
    else:
        cmd = [M4.RASQUAL, '-y', bins['Y'], '-k', bins['K'], '-n', str(n), '-j', str(k + 1), '-l', str(text.count('\n')),
               '-m', '1', '-s', str(site[2]), '-e', str(site[3]), '-f', g, '-z', '-d', str(M4.MIN_COVERAGE), '-a', str(M4.MAF),
               '-h', str(M4.HWE_P), '-x', bins['X'], '--n-threads', '1'] + flags
        t0 = time.perf_counter()
        out = subprocess.run(cmd, input=text, stdout=subprocess.PIPE, text=True, check=True).stdout
        secs, skipped = time.perf_counter() - t0, False
        C.write_atomic(raw, lambda fh: fh.write(out), 'w')
    rows = [ln.split('\t') for ln in out.splitlines()]
    bad = [r for r in rows if len(r) != len(M4.RASQUAL_FIELDS) or r[0] != g or r[1] == 'SKIPPED']
    if bad or not rows:
        raise SystemExit(f'{g}: {len(rows)} RASQUAL rows, {len(bad)} malformed or SKIPPED, e.g. {bad[:1]}')
    return pd.DataFrame(rows, columns=M4.RASQUAL_FIELDS), secs, skipped


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def prepare(S, sc, r):
    """The dataset, its allelic admission and its Y / K / X binaries, which must equal the committed arm's byte for byte."""
    ds = C.load_dataset(C.DATASETS, sc, r)
    bins, n_cov = M4.write_bins(S, ds, OUT / 'inputs' / f'{sc}_rep{r:03d}')
    old = OLD_INPUTS / sc / 'rasqual' / f'inputs_rep{r:03d}'
    diff = [k for k, p in bins.items() if sha(Path(p)) != sha(old / f'{k}.bin')]
    if diff:
        raise SystemExit(f'{sc} rep {r}: {diff} differ from {old}')
    print(f'{sc} rep {r:03d}: Y / K / X ({n_cov} covariates) identical to {old}', flush=True)
    return ds, C.allelic_kept(ds['pL'], ds['pR'], ds['Va']), bins


def run(S):
    """Every variant on every dataset of RUNS, JOBS genes at a time, longest genes (most tested variants) first."""
    if sha(Path(M4.RASQUAL)) != M4.RASQUAL_SHA256:
        raise SystemExit(f'{M4.RASQUAL}: sha256 differs from {M4.RASQUAL_SHA256}')
    genes, n = S['genes'], len(S['order'])
    text, sites = {g: M4.rsnp_text(S, g) for g in genes}, {g: M4.pseudo_site(S, g) for g in genes}
    order = sorted(range(len(genes)), key=lambda k: -S['n_tested'][genes[k]])
    prep = {(sc, r): prepare(S, sc, r) for sc, r in RUNS}
    ex = cf.ThreadPoolExecutor(JOBS)
    futs = {}
    try:
        for v, flags in VARIANTS.items():
            print(f'{v}: {M4.RASQUAL} ... --n-threads 1 {" ".join(flags)}', flush=True)
            for sc, r in RUNS:
                ds, kept, bins = prep[(sc, r)]
                raw = OUT / v / sc / 'rasqual' / f'raw_rep{r:03d}'
                raw.mkdir(parents=True, exist_ok=True)
                for k in order:
                    g = genes[k]
                    futs[(v, sc, r, k)] = ex.submit(run_gene, k, g, sites[g], M4.pseudo_line(g, sites[g], ds['pL'][k], ds['pR'][k], kept[k])
                                                    + text[g], bins, n, raw / f'{g}.txt', flags)
        print(f'{len(futs)} gene runs queued ({len(VARIANTS)} variants x {len(RUNS)} datasets x {len(genes)} genes), {JOBS} jobs',
              flush=True)
        for v in VARIANTS:
            summary, secs = {}, []
            for sc, r in RUNS:
                ds, kept, _ = prep[(sc, r)]
                parts, cnt, skipped = [], {}, 0
                for k, g in enumerate(genes):
                    raw, s, skip = futs[(v, sc, r, k)].result()
                    secs.append(s)
                    skipped += skip
                    df, c = M4.assemble(g, raw, S['tested'][g], None if ds['is_null'][k] else str(ds['causal_variant'][k]))
                    parts.append(df)
                    for key, x in c.items():
                        cnt[key] = cnt.get(key, 0) + x
                df = pd.concat(parts, ignore_index=True)
                C.write_parquet(df, nominal_path(v, sc, r), C.fingerprint(ds, f'rasqual_{v}'), 'log2')
                cnt.update(rows=len(df), tests=int(S['n_tested'].sum()), genes_skipped_existing_raw=skipped,
                           delta_median=float(df.delta.median()), phi_median=float(df.phi.median()),
                           n_feature_snps_max=int(df.n_feature_snps.max()))
                summary[f'{sc} rep {r:03d}'] = cnt
                print(f'{v} {sc} rep {r:03d}: {json.dumps(cnt)}', flush=True)
            s = np.array([x for x in secs if x > 0])
            C.write_json(OUT / v / 'summary.json', dict(
                per_dataset=summary, flags=VARIANTS[v], rasqual=M4.RASQUAL, rasqual_sha256=M4.RASQUAL_SHA256, jobs=JOBS,
                seconds_per_gene_run=dict(n=len(s), median=float(np.median(s)) if len(s) else None, total=float(s.sum()))))
    finally:
        ex.shutdown(cancel_futures=True)


def leads(arm, sc, r, u):
    """Every gene's lead in one dataset, 06 causal_and_leads' rule."""
    d = C.read_results(nominal_path(arm, sc, r), S6.JOINT_COLS)
    d['p'] = d.pval_nominal.where(np.isfinite(d.pval_nominal), np.inf)
    d['absstat'] = np.abs(d.slope / d.slope_se)
    top = (d.sort_values(['phenotype_id', 'p', 'absstat'], ascending=[True, True, False], kind='stable')
           .groupby('phenotype_id', sort=False).head(1).set_index('phenotype_id'))
    L = u.set_index('gene')[['scenario', 'rep', 'is_null', 'band', 'causal_variant']].join(top[['variant_id', 'p', 'absstat']], how='left')
    if L.variant_id.isna().any():
        raise SystemExit(f'{arm} {sc} rep {r}: no rows for {int(L.variant_id.isna().sum())} genes')
    return L.rename(columns={'variant_id': 'lead_variant', 'p': 'lead_p', 'absstat': 'lead_absstat'}).reset_index()


def fdp_power(p, s, null):
    """06 ranking's power at FDR realized false-discovery proportion on one pooled set of gene units."""
    rank = S6.evidence_rank(p, s)
    o = np.argsort(-rank, kind='stable')
    fdp = np.cumsum(null[o]) / np.arange(1, len(o) + 1)
    cut = np.r_[rank[o][1:] != rank[o][:-1], True] & (fdp <= S6.FDR)
    top = np.zeros(len(p), bool)
    top[o[:int(np.where(cut)[0].max()) + 1 if cut.any() else 0]] = True
    return float(top[~null].mean())


def paired_diff(Ls, sc, i):
    """Power at 5% FDP of every arm minus tensorqtl's and minus the committed rasqual's, with a paired gene-clustered interval:
    genes resampled with replacement N_BOOT times (06 boot, key (DIFF_KEY, i)), each carrying its units in every dataset, the same
    resample for every arm."""
    arr = {a: (L.lead_p.values, L.lead_absstat.values, L.is_null.values) for a, L in Ls.items()}
    for a, (p, s, nl) in arr.items():   # this function's power must be ranking's
        if fdp_power(p, s, nl) != Ls[a].attrs['power']:
            raise SystemExit(f'{a} {sc}: fdp_power {fdp_power(p, s, nl)} against ranking {Ls[a].attrs["power"]}')
    g = Ls['tensorqtl'].gene.values
    if any(not (np.array_equal(L.gene.values, g) and np.array_equal(L.rep.values, Ls['tensorqtl'].rep.values)) for L in Ls.values()):
        raise SystemExit(f'{sc}: arms list their units in different orders')
    genes = pd.unique(g)
    units = [np.flatnonzero(g == x) for x in genes]
    B = np.array([[fdp_power(*(v[np.concatenate([units[j] for j in b])] for v in arr[a])) for a in ARMS]
                  for b in S6.boot((DIFF_KEY, i), len(genes))])
    obs = np.array([Ls[a].attrs['power'] for a in ARMS])
    out = {}
    for ref in ('tensorqtl', 'rasqual'):
        j = ARMS.index(ref)
        d = B - B[:, [j]]
        out[ref] = {a: dict(diff=float(obs[i] - obs[j]), lo=float(np.quantile(d[:, i], .025)), hi=float(np.quantile(d[:, i], .975)))
                    for i, a in enumerate(ARMS) if a != ref}
    return out


def nonconv(arm):
    """Non-converged share of tested rows per dataset of RUNS, from the arm's RASQUAL summary."""
    if arm == 'tensorqtl':
        return None
    per = json.loads(((C.JOINT['rasqual'] if arm == 'rasqual' else OUT / arm) / 'summary.json').read_text())['per_dataset']
    per = {f'{sc} rep {r:03d}': per[f'{sc} rep {r:03d}'] for sc, r in RUNS}
    return dict(per_dataset={k: c['nonconv'] / c['tests'] for k, c in per.items()},
                pooled=sum(c['nonconv'] for c in per.values()) / sum(c['tests'] for c in per.values()),
                rows=sum(c['nonconv'] for c in per.values()), tests=sum(c['tests'] for c in per.values()),
                causal_nonconv=sum(c['causal_nonconv'] for c in per.values()))


def score():
    meta, genes, U, keep_a = S6.load_units(C.DATASETS, C.RESULTS)
    bsel, bidx = S6.band_selections(genes, U, keep_a)
    scen = [f'beta{b}' for b in meta['betas']]
    ref = json.loads(C.SUMMARY.read_text())
    res = {a: dict(ranking={}, anchor={}, nonconv=nonconv(a)) for a in ARMS}
    leadsets = {sc: {} for sc in ('beta0.4', 'beta0.8')}
    for arm in ARMS:
        if arm not in ('rasqual', 'tensorqtl'):
            for sc, r in RUNS:
                p = nominal_path(arm, sc, r)
                if C.stored_fingerprint(p) != C.fingerprint(C.load_dataset(C.DATASETS, sc, r), f'rasqual_{arm}'):
                    raise SystemExit(f'{p} does not match its dataset')
        for sc in ('beta0.4', 'beta0.8'):
            L = pd.concat([leads(arm, sc, r, U[(U.scenario == sc) & (U.rep == r)]) for s, r in RUNS if s == sc], ignore_index=True)
            rk = S6.ranking(L, (S6.AUC_BOOT_KEY, scen.index(sc)))
            res[arm]['ranking'][sc] = dict(power=rk['fdp_matched']['all']['power'], non_null=rk['fdp_matched']['all']['non_null'],
                                           discoveries=rk['fdp_matched']['discoveries'], false=rk['fdp_matched']['false'],
                                           p_threshold=rk['fdp_matched']['p_threshold'], auc=rk['auc']['all'])
            L.attrs['power'] = rk['fdp_matched']['all']['power']
            leadsets[sc][arm] = L
        u0 = U[(U.scenario == 'beta0.0') & (U.rep == 0)]
        K, n = C.rates_by_gene([nominal_path(arm, 'beta0.0', 0)], genes, 'pval_nominal', gene_filter=u0[u0.is_null].gene.tolist())
        res[arm]['anchor'] = {str(al): S6.pooled(K[al][None], n[None], bsel['all'], bidx['all']) for al in ALPHAS}
    for arm in ('rasqual', 'tensorqtl'):   # the committed arms through this code must reproduce 06_score's summary
        for sc in ('beta0.4', 'beta0.8'):
            got = res[arm]['ranking'][sc]['power'], res[arm]['ranking'][sc]['p_threshold']
            want = ref['ranking'][sc][arm]['fdp_matched']['all']['power'], ref['ranking'][sc][arm]['fdp_matched']['p_threshold']
            if got != want:
                raise SystemExit(f'{arm} {sc}: power, p threshold {got} here, {want} in {C.SUMMARY}')
        for al in map(str, ALPHAS):
            got, want = res[arm]['anchor'][al], ref['null']['beta0.0'][arm]['combined']['all'][al]
            if any(got[k] != want[k] for k in ('rate', 'lo', 'hi', 'tests')):
                raise SystemExit(f'{arm} anchor {al}: {got} here, {want} in {C.SUMMARY}')
    print(f'committed rasqual and tensorqtl reproduce {C.SUMMARY} (power at 5% FDP, anchor rates and intervals)', flush=True)
    res['paired_power_difference'] = {sc: paired_diff(Ls, sc, scen.index(sc)) for sc, Ls in leadsets.items()}
    thr = {sc: res['tensorqtl']['ranking'][sc]['p_threshold'] for sc in leadsets}   # null tail at tensorqtl's own threshold
    tail = {a: {sc: dict(null_units=int(L.is_null.sum()), null=int((L.is_null & (L.lead_p <= thr[sc])).sum()),
                         non_null_units=int((~L.is_null).sum()), non_null=int((~L.is_null & (L.lead_p <= thr[sc])).sum()))
                for sc, L in ((sc, Ls[a]) for sc, Ls in leadsets.items())} for a in ARMS}
    tr = json.loads(TRECASE_SUMMARY.read_text())
    tail['trec'] = {}
    for sc in leadsets:   # the TReCASE page's counts must be on the same threshold and lead rule
        b, x = sc[4:], tr['null_tail'][f'tensorqtl {sc[4:]}']
        if (x['threshold'], x['at_or_below'], x['non_null_at_or_below']) != (thr[sc], tail['tensorqtl'][sc]['null'], tail['tensorqtl'][sc]['non_null']):
            raise SystemExit(f'{TRECASE_SUMMARY} tensorqtl {b}: {x} against {thr[sc]}, {tail["tensorqtl"][sc]} here')
        t = tr['null_tail'][f'fractional pval_t {b}']
        tail['trec'][sc] = dict(null_units=t['null_units'], null=t['at_or_below'], non_null_units=t['non_null_units'],
                                non_null=t['non_null_at_or_below'], power=tr['power'][f'pval_t {b}']['fractional_power'])
    res['null_tail'] = dict(threshold=thr, counts=tail, trec_source=f'{TRECASE_SUMMARY}: fractional pval_t (the committed TReCASE run)')
    cols = ['phenotype_id', 'variant_id', 'pval_nominal']   # fix_delta's anchor rows split by whether the committed arm kept them (converged)
    m = pd.read_parquet(nominal_path('fix_delta', 'beta0.0', 0), columns=cols).merge(
        pd.read_parquet(nominal_path('rasqual', 'beta0.0', 0), columns=cols), on=cols[:2], how='left', suffixes=('', '_committed'),
        indicator=True)
    kept = (m._merge == 'both').values
    res['fix_delta']['anchor_by_committed_convergence'] = dict(
        rows_kept=int(kept.sum()), rows_dropped=int((~kept).sum()),
        **{str(al): dict(fix_delta_on_kept=float((m.pval_nominal.values[kept] < al).mean()),
                         committed_on_kept=float((m.pval_nominal_committed.values[kept] < al).mean()),
                         fix_delta_on_dropped=float((m.pval_nominal.values[~kept] < al).mean())) for al in ALPHAS})
    for arm in ARMS[:3]:
        d = pd.concat([pd.read_parquet(nominal_path(arm, sc, r), columns=['delta', 'phi', 'n_feature_snps']) for sc, r in RUNS])
        res[arm]['params'] = dict(delta_median=float(d.delta.median()), delta_q10=float(d.delta.quantile(.1)),
                                  delta_q90=float(d.delta.quantile(.9)), share_delta_fixed=float((d.delta == FIXED_DELTA).mean()),
                                  phi_median=float(d.phi.median()), share_no_fsnp=float((d.n_feature_snps == 0).mean()), rows=len(d))
    C.write_json(OUT / 'scores.json', res)
    for arm in ARMS:
        x = res[arm]
        print(f'{arm}: power at 5% FDP ' + ', '.join(f'{sc} {x["ranking"][sc]["power"]:.3f}' for sc in x['ranking'])
              + '; anchor ' + ', '.join(f'{al} {x["anchor"][al]["rate"]:.4f}' for al in x['anchor'])
              + (f'; non-converged {x["nonconv"]["pooled"]:.4f}' if x['nonconv'] else ''), flush=True)
    return res


def figure(res):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.9), gridspec_kw=dict(width_ratios=[1.3, 1]))
    w = 0.19
    for i, arm in enumerate(ARMS):
        xs = np.arange(2) + (i - 1.5) * w
        v = [res[arm]['ranking'][sc]['power'] for sc in ('beta0.4', 'beta0.8')]
        a1.bar(xs, v, w * 0.92, color=COLOR[arm], label=LABEL[arm])
        for x, y in zip(xs, v):
            a1.text(x, y + 0.01, f'{y:.3f}', ha='center', va='bottom', fontsize=7.5)
        an = res[arm]['anchor']['0.05']
        a2.errorbar(an['rate'], i, xerr=[[an['rate'] - an['lo']], [an['hi'] - an['rate']]], fmt='o', color=COLOR[arm], capsize=3)
    a1.set_xticks([0, 1], ['|beta| = 0.4', '|beta| = 0.8'])
    a1.set_ylim(0, 1)
    a1.set_ylabel('share of 150 non-null gene units called')
    a1.set_title('A. Power at 5% realized false-discovery proportion', loc='left', fontsize=10)
    a1.legend(fontsize=8, frameon=False, loc='upper left')
    a2.axvline(0.05, color='#888', lw=0.8, ls=':')
    a2.set_yticks(range(len(ARMS)), [LABEL[a] for a in ARMS], fontsize=8)
    a2.invert_yaxis()
    a2.set_xlabel('share of tested variants with nominal p < 0.05')
    a2.set_title('B. Null genes, beta = 0 anchor, at 0.05', loc='left', fontsize=10)
    for ax in (a1, a2):
        ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def page(res):
    f = lambda x: f'{x:.3f}'   # noqa: E731
    ci = lambda d: f'{d["diff"]:+.3f} [{d["lo"]:+.3f}, {d["hi"]:+.3f}]'   # noqa: E731
    pdiff = res['paired_power_difference']
    rank_rows, cal_rows = '', ''
    for a in ARMS:
        x = res[a]
        entries = [html.escape(LABEL[a])]
        for sc in ('beta0.4', 'beta0.8'):
            k = x['ranking'][sc]
            entries += [f'{f(k["power"])} ({k["discoveries"]} called, {k["false"]} null)',
                      ci(pdiff[sc]['tensorqtl'][a]) if a != 'tensorqtl' else '',
                      ci(pdiff[sc]['rasqual'][a]) if a != 'rasqual' else '',
                      f'{f(k["auc"]["mean"])} [{f(k["auc"]["lo"])}, {f(k["auc"]["hi"])}]']
        rank_rows += '<tr>' + ''.join(f'<td>{c}</td>' for c in entries) + '</tr>'
        entries = [html.escape(LABEL[a])] + [f'{x["anchor"][al]["rate"]:.4f} [{x["anchor"][al]["lo"]:.4f}, {x["anchor"][al]["hi"]:.4f}]'
                                           for al in map(str, ALPHAS)]
        entries.append(f'{x["nonconv"]["pooled"]:.5f} ({x["nonconv"]["rows"]:,} of {x["nonconv"]["tests"]:,})' if x['nonconv']
                     else 'n/a (least squares)')
        cal_rows += '<tr>' + ''.join(f'<td>{c}</td>' for c in entries) + '</tr>'
    head = ''.join('<th>power at 5% FDP</th><th>minus tensorQTL</th><th>minus committed RASQUAL</th>'
                   '<th>AUC [interval: the 3 datasets resampled]</th>' for _ in range(2))
    nt = res['null_tail']
    cnt = lambda x: f'{x["null"]} of {x["null_units"]} null, {x["non_null"]} of {x["non_null_units"]} non-null'   # noqa: E731
    tail_rows = ''.join(f'<tr><td>{html.escape(name)}</td>' + ''.join(f'<td>{cnt(nt["counts"][a][sc])}</td>' for sc in ('beta0.4', 'beta0.8'))
                        + '</tr>' for a, name in [*LABEL.items(), ('trec', 'TReCASE total-count (TReC) test, committed run (fractional totals)')])
    tail_head = ''.join(f'<th>|beta| = {sc[4:]}: lead p &le; {nt["threshold"][sc]:.2g}</th>' for sc in ('beta0.4', 'beta0.8'))
    p = {a: res[a]['params'] for a in ARMS[:3]}
    img = figure(res)
    body = f"""<h1>RASQUAL input diagnosis: total-count model alone, and delta fixed</h1>
<p class=meta>2026-09-28. Script <code>scripts/rasqual_input_diagnosis.py</code>; outputs in <code>{OUT}</code>.
Deep plasmode set (100 genes, <code>{C.ROOT.name}</code>).</p>
<h2>Why</h2>
<p>On the plasmode datasets RASQUAL ranks non-null genes below total-only tensorQTL. Two explanations were open. Either
RASQUAL's total-count model handles the Salmon totals poorly, or its allelic channel is blunted. The second is suspected
because the pseudo feature SNP gets a high estimated sequencing/mapping error rate delta, the probability RASQUAL gives a
read of showing the other allele. The read-level review (<code>rasqual_read_level_20260927</code>) measured a median
delta of 0.187 at 20 real-data eQTL leads with the pseudo feature SNP, against 0.003 with read-level allele counts at real
feature SNPs. On these datasets the committed arm's delta over converged tested rows has median
{p['rasqual']['delta_median']:.3f} (10th-90th percentile {p['rasqual']['delta_q10']:.3f}-{p['rasqual']['delta_q90']:.3f}).
A larger delta pulls every modelled allelic share toward one half (<code>getK</code>, <code>nbem.c:1428-1439</code>), so
it weakens the allelic signal.</p>
<h2>What was run</h2>
<p>Two variants of the committed RASQUAL arm (<code>04_run_rasqual.py</code>). Each is 04's command with one flag added.
The VCF text is unchanged and still includes the pseudo feature SNP. The Y, K and X binaries were checked byte for byte
against the committed arm's own inputs for all {len(RUNS)} datasets. Before the run, the same command with no flag
reproduced byte for byte the raw output for one gene (ITGA9, |beta| 0.8 rep 0) of the acceptance rerun of 04, whose
nominal file for that dataset matched the committed arm's bit for bit (<code>timing_unit.log</code>).</p>
<ul>
<li><b>--population-only.</b> In <code>main.c</code>, line 216 sets <code>ASE=0</code>. Line 516 then admits no feature SNP,
so neither the allelic null (<code>nbem.c:110</code>) nor the allelic alternative (<code>nbem.c:302</code>) is fitted. What
remains is RASQUAL's negative-binomial model of total counts (a Poisson count whose mean is itself gamma-distributed,
fitted with an overdispersion parameter), plus its genotype prior. Check: {100 * p['population_only']['share_no_fsnp']:.1f}%
of rows report no feature SNP.</li>
<li><b>--fix-delta.</b> The full joint model with delta fixed at {FIXED_DELTA}. That is RASQUAL's documented fixed value
(<code>usage.c</code>) and the mode of its Beta(1.01, 1.99) prior (<code>nbem.c:622</code>). RASQUAL has no option that
fixes delta at zero, and none was improvised. Check: {100 * p['fix_delta']['share_delta_fixed']:.1f}% of rows report
delta = {FIXED_DELTA}.</li>
</ul>
<p>Datasets: the beta = 0 anchor (rep 0), and reps 0-2 of |beta| 0.4 and 0.8, each with 50 of 100 genes non-null.
<b>Power at 5% realized false-discovery proportion (FDP)</b> is scored as in <code>06_score.py</code>. Each
(dataset, gene) unit is represented by its lead variant, the one with the smallest nominal p. The units of the three
datasets are pooled and ranked, and the list is cut at the deepest point where at most 5% of the called units are null.
Power is the share of the 150 non-null units above that cut. <b>AUC</b> is the share of (non-null, null) gene pairs
ranked in the right order within a dataset, averaged over the three datasets. Its interval resamples the three datasets
with replacement, which allows at most 10 distinct resamples, so it is coarse and carries no comparison here; the paired
power differences do. The
<b>anchor rate</b> is the share of tested (gene, variant) pairs on the all-null dataset with nominal p below alpha. Its
<b>gene-clustered interval</b> resamples genes with all their variants, 2,000 times. Committed RASQUAL and tensorQTL were
rescored through the same code and reproduce <code>summary.json</code> exactly.</p>
<h2>Result</h2>
<img src="data:image/png;base64,{img}" alt="power and anchor null rate per arm">
<table><tr><th rowspan=2>arm</th><th colspan=4>|beta| = 0.4</th><th colspan=4>|beta| = 0.8</th></tr>
<tr>{head}</tr>
{rank_rows}</table>
<p class=note>Differences are paired: the same gene resample for every arm (2,000 resamples of the 100 genes, each gene
carrying its units in all three datasets), 2.5% and 97.5% quantiles. Power moves in steps of 1/150 = 0.0067.</p>
<table><tr><th>arm</th><th>anchor rate, alpha 0.05</th><th>anchor rate, alpha 0.001</th><th>non-converged share of tested rows</th></tr>
{cal_rows}</table>
<p class=note>Non-converged share: rows with RASQUAL's convergence flag non-zero, over tested rows, pooled over the
{len(RUNS)} datasets. Those rows are excluded before leads are taken, as in the committed arm. Median delta at converged
rows: committed {p['rasqual']['delta_median']:.3f}, --fix-delta {p['fix_delta']['delta_median']:.3f},
--population-only {p['population_only']['delta_median']:.3f} (not estimated in that mode).</p>
<table><tr><th>lead p of</th>{tail_head}</tr>
{tail_rows}</table>
<p class=note>Null tail: gene units of each |beta| (3 datasets, 150 null and 150 non-null) whose lead p is at or below
tensorQTL's own p threshold at 5% FDP there. The TReC row is the committed TReCASE arm's total-count test on the same
fractional totals, read from <code>{TRECASE_SUMMARY}</code>; that page's tensorQTL threshold and counts equal this page's.</p>
{interpretation(res)}
"""
    css = ('body{font-family:system-ui,sans-serif;max-width:960px;margin:24px auto;padding:0 16px;color:#222;background:#fff;line-height:1.45}'
           'table{border-collapse:collapse;font-size:13px;margin:12px 0}td,th{border:1px solid #ccc;padding:4px 7px;text-align:left}'
           'img{max-width:100%}.meta,.note{color:#555;font-size:13px}code{font-size:12.5px}')
    doc = f'<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>RASQUAL input diagnosis</title><style>{css}</style></head><body>{body}</body></html>'
    C.write_atomic(OUT / 'report.html', lambda fh: fh.write(doc), 'w')
    print(f'wrote {OUT / "report.html"}', flush=True)


def interpretation(res):
    """The page's reading of the numbers (written 2026-09-28 against scores.json); stops if a claim it makes no longer holds."""
    d, r = res['paired_power_difference'], {a: res[a] for a in ARMS}
    an = {a: r[a]['anchor'] for a in ARMS}
    sel = r['fix_delta']['anchor_by_committed_convergence']
    nt, thr = res['null_tail']['counts'], res['null_tail']['threshold']
    ex = lambda a, sc: nt[a][sc]['null'] - nt['tensorqtl'][sc]['null']   # noqa: E731   null units at the threshold beyond tensorqtl's
    claims = {
        'population_only reaches fewer non-null units than tensorqtl at tensorqtl\'s threshold, both |beta|':
            all(nt['population_only'][sc]['non_null'] < nt['tensorqtl'][sc]['non_null'] for sc in d),
        'TReC reaches at least as many non-null units as tensorqtl at that threshold, both |beta|':
            all(nt['trec'][sc]['non_null'] >= nt['tensorqtl'][sc]['non_null'] for sc in d),
        'population_only puts more null units than tensorqtl at that threshold and fewer than TReC, both |beta|':
            all(0 < ex('population_only', sc) < ex('trec', sc) for sc in d),
        'TReC power below tensorqtl at both |beta|': all(nt['trec'][sc]['power'] < r['tensorqtl']['ranking'][sc]['power'] for sc in d),
        'fix_delta rate on the committed arm\'s converged anchor rows above the committed interval at 0.05':
            sel['0.05']['fix_delta_on_kept'] > an['rasqual']['0.05']['hi'],
        'population_only below tensorqtl at both |beta|, paired interval excluding 0': all(d[sc]['tensorqtl']['population_only']['hi'] < 0 for sc in d),
        'fix_delta minus committed rasqual interval spans 0 at both |beta|': all(d[sc]['rasqual']['fix_delta']['lo'] < 0 < d[sc]['rasqual']['fix_delta']['hi'] for sc in d),
        'population_only minus committed rasqual interval spans 0 at both |beta|': all(d[sc]['rasqual']['population_only']['lo'] < 0 < d[sc]['rasqual']['population_only']['hi'] for sc in d),
        'committed rasqual minus tensorqtl interval spans 0 at 0.4': d['beta0.4']['tensorqtl']['rasqual']['lo'] < 0 < d['beta0.4']['tensorqtl']['rasqual']['hi'],
        'fix_delta anchor at 0.05 above committed rasqual, intervals disjoint': an['fix_delta']['0.05']['lo'] > an['rasqual']['0.05']['hi'],
        'population_only anchor at 0.05 below 0.05 with its interval': an['population_only']['0.05']['hi'] < 0.05,
        'committed rasqual has non-converged rows, fix_delta almost none': r['rasqual']['nonconv']['pooled'] > 100 * r['fix_delta']['nonconv']['pooled']}
    bad = [k for k, ok in claims.items() if not ok]
    if bad:
        raise SystemExit(f'claims on the page no longer hold: {bad}')
    f = lambda x: f'{x["diff"]:+.3f} [{x["lo"]:+.3f}, {x["hi"]:+.3f}]'   # noqa: E731
    pw = lambda a, sc: f'{r[a]["ranking"][sc]["power"]:.3f}'            # noqa: E731
    rate = lambda a, al: f'{an[a][al]["rate"]:.4f} [{an[a][al]["lo"]:.4f}, {an[a][al]["hi"]:.4f}]'   # noqa: E731
    neg = lambda x: dict(diff=-x['diff'], lo=-x['hi'], hi=-x['lo'])    # noqa: E731   committed minus the arm
    return f"""<h2>What the numbers say</h2>
<p><b>RASQUAL's total-count model alone ranks non-null genes below tensorQTL on the same totals.</b> With the allelic
part switched off, RASQUAL's power at 5% FDP is {pw('population_only', 'beta0.4')} at |beta| 0.4 and
{pw('population_only', 'beta0.8')} at 0.8, against tensorQTL's {pw('tensorqtl', 'beta0.4')} and {pw('tensorqtl', 'beta0.8')}.
The paired differences are {f(d['beta0.4']['tensorqtl']['population_only'])} and {f(d['beta0.8']['tensorqtl']['population_only'])};
both intervals exclude zero. Both arms read the same thinned Salmon totals, the same 17 covariates and the same
effective library sizes. tensorQTL fits least squares on log2(CPM + 1); RASQUAL fits its negative-binomial likelihood to
the counts with the library size as an offset. The total-count model is also conservative on the anchor:
{rate('population_only', '0.05')} at alpha 0.05 and {rate('population_only', '0.001')} at 0.001, against tensorQTL's
{rate('tensorqtl', '0.05')} and {rate('tensorqtl', '0.001')}.</p>
<p><b>It loses on the non-null side, where TReCASE's total-count test loses in the null tail.</b> Counted at tensorQTL's
own threshold at 5% FDP (p &le; {thr['beta0.4']:.2g} at |beta| 0.4, {thr['beta0.8']:.2g} at 0.8; table above), RASQUAL's
total-count model puts {nt['population_only']['beta0.4']['non_null']} of the 150 non-null units at |beta| 0.4 at or below
it, against tensorQTL's {nt['tensorqtl']['beta0.4']['non_null']}, and {nt['population_only']['beta0.8']['non_null']}
against {nt['tensorqtl']['beta0.8']['non_null']} at 0.8. It also puts more null units there
({nt['population_only']['beta0.4']['null']} against {nt['tensorqtl']['beta0.4']['null']}, and
{nt['population_only']['beta0.8']['null']} against {nt['tensorqtl']['beta0.8']['null']}), an excess of
{ex('population_only', 'beta0.4')} and {ex('population_only', 'beta0.8')} null units. TReCASE's total-count (TReC) test,
in the committed run on the same fractional totals, also ranks below tensorQTL (power {nt['trec']['beta0.4']['power']:.3f}
and {nt['trec']['beta0.8']['power']:.3f} against {pw('tensorqtl', 'beta0.4')} and {pw('tensorqtl', 'beta0.8')}), but the
other way round. It reaches at least as many non-null units as tensorQTL ({nt['trec']['beta0.4']['non_null']} and
{nt['trec']['beta0.8']['non_null']}), and its null excess is {ex('trec', 'beta0.4')} and {ex('trec', 'beta0.8')} units
({nt['trec']['beta0.4']['null']} and {nt['trec']['beta0.8']['null']} null units at or below the threshold). Both count
models rank below least squares on the same Salmon totals, by different routes, so these runs do not identify a shared
cause.</p>
<p><b>The allelic part adds no detectable power.</b> The committed full model minus population-only is
{f(neg(d['beta0.4']['rasqual']['population_only']))} at |beta| 0.4 and {f(neg(d['beta0.8']['rasqual']['population_only']))}
at 0.8. Both intervals include zero. Their upper bounds, {neg(d['beta0.4']['rasqual']['population_only'])['hi']:+.3f} and
{neg(d['beta0.8']['rasqual']['population_only'])['hi']:+.3f}, are the largest gains they admit, so a gain of that size is
not excluded.</p>
<p><b>Fixing delta does not rescue the ranking, and it costs calibration.</b> With delta fixed at 0.01, power is
{pw('fix_delta', 'beta0.4')} and {pw('fix_delta', 'beta0.8')}. Against the committed arm that is
{f(d['beta0.4']['rasqual']['fix_delta'])} and {f(d['beta0.8']['rasqual']['fix_delta'])}, and both intervals include zero.
The fix does remove almost all non-convergence: {100 * r['rasqual']['nonconv']['pooled']:.2f}% of tested rows in the
committed arm ({r['rasqual']['nonconv']['causal_nonconv']} causal variants) against
{100 * r['fix_delta']['nonconv']['pooled']:.4f}% with delta fixed. But the null rate rises. On the anchor it is
{rate('fix_delta', '0.05')} at alpha 0.05, against the committed arm's {rate('rasqual', '0.05')}, and
{rate('fix_delta', '0.001')} at 0.001, against {rate('rasqual', '0.001')}. This is not an effect of the committed arm
dropping its non-converged rows. On the {sel['rows_kept']:,} anchor rows the committed arm kept, --fix-delta rejects at
{sel['0.05']['fix_delta_on_kept']:.4f} at 0.05 and {sel['0.001']['fix_delta_on_kept']:.4f} at 0.001. The committed arm
rejects at {sel['0.05']['committed_on_kept']:.4f} and {sel['0.001']['committed_on_kept']:.4f} on those same rows. The
{sel['rows_dropped']:,} rows it dropped reject at {sel['0.05']['fix_delta_on_dropped']:.4f} under --fix-delta. So the high estimated delta is not simply
blunting a clean allelic signal. The pattern fits delta absorbing allelic variation in the pseudo feature SNP's counts
that its beta-binomial overdispersion does not absorb (the beta-binomial is a binomial allele count whose success
probability itself varies, beta-distributed); with delta fixed, that variation becomes false positives. This was
not tested directly.</p>
<h2>Critique</h2>
<p><b>The statistic is coarse.</b> Power at 5% FDP rests on one cut through 300 pooled units per |beta|, so it moves in
steps of 1/150. The paired interval resamples only 100 genes. The one conclusion that clears its interval at both
effect sizes is population-only against tensorQTL. The gap that motivated this run is committed RASQUAL minus tensorQTL,
{f(d['beta0.4']['tensorqtl']['rasqual'])} at |beta| 0.4. On its own it does not clear a gene-clustered interval.</p>
<p><b>--population-only is RASQUAL's total model, not a bare negative-binomial GLM.</b> It still treats each genotype
as uncertain. Allelic probabilities are truncated to [0.001, 0.999] (<code>main.c:484-490</code>), and each donor's
genotype posterior is updated from the expression likelihood (<code>nbem.c:862-870</code>, on unless
<code>--no-posterior-update</code>). tensorQTL has no counterpart to this, and it was not switched off here. The run also
cannot say whether the deficit comes from the input form or from the likelihood itself on these data. The input form here
means fractional Salmon totals, which enter the negative-binomial density through the log-gamma function (defined for
non-integer counts). The TReCASE integer-total diagnosis and the mirror benchmark address that question; this run does not.</p>
<p><b>Delta was fixed at 0.01, not 0.</b> That is the only value RASQUAL's options provide.</p>
<h2>What it means</h2>
<p>The concern was that the benchmark hands RASQUAL inputs its likelihood does not expect. For the allelic half, the
suspected delta attenuation is not what holds RASQUAL back. Fixing delta at 0.01 gives no detectable power gain, and
the null then runs at {an['fix_delta']['0.05']['rate'] / 0.05:.2f} times nominal at 0.05. For the total half, RASQUAL's count model loses to least squares on the same Salmon totals
by an amount that clears the gene-resampling noise, through fewer non-null units reaching tensorQTL's threshold.
TReCASE's total-count test also ranks below least squares, but through null units in its tail, so the two count
likelihoods do not share one measured failure and no common cause is claimed. Whether a count likelihood fed
alignment-based integer counts does better is the question for the native-input arm.</p>
"""


def main():
    S = C.setup(C.load()[0])
    meta = json.loads((C.DATASETS / 'meta.json').read_text())
    if S['genes'] != meta['genes'] or S['order'] != meta['donors']:
        raise SystemExit(f'{C.DATASETS / "meta.json"}: genes or donors differ from the loader inputs')
    run(S)
    page(score())


if __name__ == '__main__':
    main()

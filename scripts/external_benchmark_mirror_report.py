"""report.html and fig_mirror.png for the mirror benchmark, from OUT/summary.json
(external_benchmark_mirror.py summarize). Every number on the page is read from that file, except the 2026-09-23
record's default-mode type-I rates, which are read from that record (PRIOR, the file summarize reproduces)."""
import base64
import io
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path('/mnt/ssd/lalli/brainvar_hapmix_deploy/external_benchmark_current_20260928')
S = json.loads((OUT / 'summary.json').read_text())
PRIOR = json.loads(Path('/mnt/ssd/lalli/brainvar_hapmix_deploy/external_benchmark_fitted_defaults_20260923/'
                        'null_and_power_500reps.json').read_text())   # the 2026-09-23 record (external_benchmark_mirror.PRIOR)
PRIOR_ARM = 'hapmixQTL sigma^2*v (DEFAULT)'                          # its default-mode arm, N = 200
NS = ('200', '92')
FOLDS = ('1.05', '1.1', '1.2')
REF = 'TReCASE (joint)'
TYPE1_ARMS = ('TReC-only', 'ASE-only', 'TReCASE (joint)', 'TReCASE (asSeq joint p)', 'TReCASE (asSeq final p)',
              'hapmixQTL gibbs', 'hapmixQTL split', 'hapmixQTL plus_one')
DIFF_ARMS = ('TReCASE (asSeq joint p)', 'TReCASE (asSeq final p)', 'hapmixQTL gibbs', 'hapmixQTL split',
             'hapmixQTL plus_one')
LABEL = {'TReC-only': 'TReC-only', 'ASE-only': 'ASE-only', 'TReCASE (joint)': 'TReCASE, harness',
         'TReCASE (asSeq joint p)': 'asSeq, joint p', 'TReCASE (asSeq final p)': 'asSeq, final p',
         'hapmixQTL gibbs': 'hapmixQTL gibbs', 'hapmixQTL split': 'hapmixQTL split',
         'hapmixQTL plus_one': 'hapmixQTL plus_one'}
FOLD_STYLE = {'1.05': ('#2a78d6', 'o'), '1.1': ('#eb6834', 's'), '1.2': ('#1baf7a', '^')}   # dataviz slots 1-3
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'


def t1(N, a, al):
    return S['by_N'][N]['type1'][a][al]


def pw(N, k, a):
    return S['by_N'][N]['power'][k][a]


def figure():
    plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
                         'ytick.color': INK, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 3, figsize=(11, 7.0), gridspec_kw=dict(width_ratios=[1, 0.8, 1.2]))
    for r, N in enumerate(NS):
        ys = list(range(len(TYPE1_ARMS)))[::-1]
        for c, al in enumerate(('0.05', '0.01')):
            ax = axes[r, c]
            a0 = float(al)
            se = t1(N, TYPE1_ARMS[0], al)['mc_se']
            ax.axvspan(a0 - 2 * se, a0 + 2 * se, color=GRID, lw=0, zorder=0)
            ax.axvline(a0, color=MUTED, lw=1, zorder=1)
            for y, a in zip(ys, TYPE1_ARMS):
                ax.plot(t1(N, a, al)['rate'], y, 'o', ms=6, color=INK, zorder=3)
            ax.set_yticks(ys, [LABEL[a] for a in TYPE1_ARMS] if c == 0 else [])
            ax.set_ylim(-0.6, len(ys) - 0.4)
            ax.xaxis.set_major_locator(plt.MaxNLocator(4))
            ax.set_xlabel(f'type-I error rate (band: {al} ± 2 MC SE)')
            ax.set_title(f'N = {N} · nominal {al}', fontsize=9.5, color=INK, loc='left')
            ax.grid(axis='x', color=GRID, lw=0.6)
        ax = axes[r, 2]
        ys = list(range(len(DIFF_ARMS)))[::-1]
        ax.axvline(0, color=MUTED, lw=1)
        for j, k in enumerate(FOLDS):
            col, mk = FOLD_STYLE[k]
            for y, a in zip(ys, DIFF_ARMS):
                d = pw(N, k, a)[f'diff_vs_{REF}']
                ax.errorbar(d['diff'], y + (1 - j) * 0.24, xerr=d['se_resample'], fmt=mk, ms=6, color=col,
                            mec='#fcfcfb', mew=0.8, elinewidth=2, capsize=0,
                            label=f'fold {k}' if y == ys[0] else None, zorder=3)
        ax.set_yticks(ys, [LABEL[a] for a in DIFF_ARMS])
        ax.set_ylim(-0.6, len(ys) - 0.4)
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.set_xlabel('power − harness TReCASE')
        ax.set_title(f'N = {N} · matched power', fontsize=9.5, color=INK, loc='left')
        ax.grid(axis='x', color=GRID, lw=0.6)
    h, lab = axes[0, 2].get_legend_handles_labels()
    fig.legend(h, lab, loc='lower center', ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=160, facecolor='#fcfcfb')
    tmp = OUT / 'fig_mirror.png.tmp'
    tmp.write_bytes(buf.getvalue())
    os.replace(tmp, OUT / 'fig_mirror.png')
    return base64.b64encode(buf.getvalue()).decode()


def f3(x):
    return f'{x:.3f}'


def f4(x):
    return f'{x:.4f}'


def type1_table():
    head = ''.join(f'<th colspan="3">N = {N}</th>' for N in NS)
    sub = ''.join('<th>0.05</th><th>0.01</th><th>0.001</th>' for _ in NS)
    rows = []
    for a in TYPE1_ARMS:
        entries = ''.join(f'<td>{f4(t1(N, a, al)["rate"])} <span class="n">({t1(N, a, al)["count"]})</span></td>'
                        for N in NS for al in ('0.05', '0.01', '0.001'))
        rows.append(f'<tr><th>{LABEL[a]}</th>{entries}</tr>')
    se = ''.join(f'<td>{f4(t1(N, TYPE1_ARMS[0], al)["mc_se"])}</td>' for N in NS for al in ('0.05', '0.01', '0.001'))
    rows.append(f'<tr class="se"><th>Monte Carlo SE at nominal</th>{se}</tr>')
    return (f'<table><thead><tr><th rowspan="2">arm</th>{head}</tr><tr>{sub}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table>')


def power_table(N):
    head = ''.join(f'<th>fold {k}</th>' for k in FOLDS)
    rows = []
    for a in TYPE1_ARMS:
        entries = []
        for k in FOLDS:
            e = pw(N, k, a)
            s = f'{f3(e["matched"])} <span class="n">±{f3(e["resample_se"])}</span>'
            if a != REF:
                d = e[f'diff_vs_{REF}']
                s += (f'<br><span class="d">{d["diff"]:+.3f} (paired {f3(d["se_paired"])}, '
                      f'resampling {f3(d["se_resample"])})</span>')
            entries.append(f'<td>{s}</td>')
        rows.append(f'<tr><th>{LABEL[a]}</th>{"".join(entries)}</tr>')
    return f'<table><thead><tr><th>arm</th>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table>'


def main():
    img = figure()
    dg = {N: S['by_N'][N]['diagnostics'] for N in NS}
    rep = S['reproduces_20260923']
    fails = {N: [dg[N][k]['asseq']['joint_failed'] for k in ('1.0', '1.05', '1.1', '1.2')] for N in NS}
    meier = {N: dg[N]['1.0']['meier']['hapmixQTL gibbs'] for N in NS}
    dof = {N: dg[N]['1.0']['dof_nominal']['hapmixQTL gibbs'] for N in NS}
    g = lambda N, al: f4(t1(N, 'hapmixQTL gibbs', al)['rate'])   # noqa: E731
    z01 = lambda N: (t1(N, 'hapmixQTL gibbs', '0.01')['rate'] - 0.01) / t1(N, 'hapmixQTL gibbs', '0.01')['mc_se']  # noqa: E731
    HM = ('hapmixQTL gibbs', 'hapmixQTL split', 'hapmixQTL plus_one')
    old = PRIOR['null'][PRIOR_ARM]   # the 2026-09-23 default-mode arm on the same N = 200 null replicates
    n = S['design']['reps']
    if (old['n'], PRIOR['config']['N'], PRIOR['config']['reps']) != (n, 200, n):
        raise SystemExit(f'2026-09-23 record: {old["n"]} replicates at N = {PRIOR["config"]["N"]}, not {n} at N = 200')
    oc = {al: old[k] * n for al, k in (('0.05', 't05'), ('0.01', 't01'))}
    if any(abs(c - round(c)) > 1e-9 for c in oc.values()):
        raise SystemExit(f'2026-09-23 record: rates {old} are not whole counts of {n}')
    oc = {al: round(c) for al, c in oc.items()}
    nc01, se01 = t1('200', 'hapmixQTL gibbs', '0.01')['count'], t1('200', 'hapmixQTL gibbs', '0.01')['mc_se'] * n
    z_old01 = (old['t01'] - 0.01) / t1('200', 'hapmixQTL gibbs', '0.01')['mc_se']
    z_chg = (oc['0.01'] - nc01) / se01
    dd = {(N, k, a): pw(N, k, a)[f'diff_vs_{REF}'] for N in NS for k in FOLDS for a in HM}
    zb = {key: abs(d['diff']) / d['se_resample'] for key, d in dd.items() if d['se_resample'] > 0}
    zp = {key: abs(d['diff']) / d['se_paired'] for key, d in dd.items() if d['se_paired'] > 0}
    flagged = sorted(key for key, z in zp.items() if z >= 2)
    flag_txt = ' and '.join(f'{a.split()[1]} at fold {k} at N = {N} ({dd[(N, k, a)]["diff"]:+.3f}, paired SE '
                            f'{dd[(N, k, a)]["se_paired"]:.3f}, {zb[(N, k, a)]:.1f} resampling SE)' for N, k, a in flagged)
    spread = max(max(pw(N, k, a)['matched'] for a in HM) - min(pw(N, k, a)['matched'] for a in HM)
                 for N in NS for k in FOLDS)
    aj = pw('200', '1.2', 'TReCASE (asSeq joint p)')
    aj_d = aj[f'diff_vs_{REF}']
    jf12 = dg['200']['1.2']['asseq']['joint_failed']
    sw = dg['200']['1.0']['asseq_switch']
    theta = sum(dg[N][k]['asseq_trace']['theta_fail'] for N in NS for k in dg[N])
    c52 = sum(dg[N][k]['asseq_trace']['lbfgsb_52'] for N in NS for k in dg[N])
    jf_all = sum(dg[N][k]['asseq']['joint_failed'] for N in NS for k in dg[N])
    ase_b = sum(dg[N][k]['asseq_trace']['ase_baseline_fail'] for N in NS for k in dg[N])
    ase_f = sum(dg[N][k]['asseq_trace']['ase_fail'] for N in NS for k in dg[N])
    vs_final = {(N, k, a): pw(N, k, a)['diff_vs_TReCASE (asSeq final p)'] for N in NS for k in FOLDS for a in HM}
    zf = {key: d['diff'] / d['se_resample'] for key, d in vs_final.items()}
    zf12 = [zf[(N, '1.2', a)] for N in NS for a in HM]
    over2 = [f'{a.split()[1]} at fold {k}, N = {N} ({zf[(N, k, a)]:+.1f} SE)' for N in NS for k in ('1.05', '1.1')
             for a in HM if abs(zf[(N, k, a)]) >= 2]
    html = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Mirror Benchmark</title>
<style>
:root {{ --bg:#fcfcfb; --ink:#0b0b0b; --muted:#52514e; --rule:#e4e3df; --card:#ffffff; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#1a1a19; --ink:#f4f3ef; --muted:#c3c2b7; --rule:#3a3936; --card:#232321; }} }}
:root[data-theme="dark"] {{ --bg:#1a1a19; --ink:#f4f3ef; --muted:#c3c2b7; --rule:#3a3936; --card:#232321; }}
body {{ background:var(--bg); color:var(--ink); font:15px/1.55 system-ui,-apple-system,sans-serif; margin:0; padding:24px 16px; }}
main {{ max-width:980px; margin:0 auto; }}
h1 {{ font-size:22px; margin:0 0 4px; }} h2 {{ font-size:17px; margin:28px 0 8px; }}
p.sub {{ color:var(--muted); margin:0 0 16px; }}
figure {{ margin:16px 0; background:#fcfcfb; border:1px solid var(--rule); border-radius:6px; padding:8px; }}
figure img {{ width:100%; height:auto; display:block; }} figcaption {{ color:#52514e; font-size:13px; padding:6px 4px 0; }}
.scroll {{ overflow-x:auto; }}
table {{ border-collapse:collapse; font-size:13px; margin:8px 0; font-variant-numeric:tabular-nums; }}
th, td {{ border-bottom:1px solid var(--rule); padding:4px 8px; text-align:right; vertical-align:top; }}
th:first-child {{ text-align:left; }} thead th {{ color:var(--muted); font-weight:600; }}
.n, .d {{ color:var(--muted); font-size:12px; }} tr.se td, tr.se th {{ color:var(--muted); }}
code {{ font-size:13px; }}
</style></head><body><main>
<h1>Mirror benchmark: default mode against TReCASE, 2026-09-28</h1>
<p class="sub">tests/ase_external_benchmark.py with its two known defects fixed, hapmixQTL in current default mode at three
weightings, and the real asSeq::trecase run on the same simulated data. 500 replicates per condition at N = 200 and N = 92.</p>

<h2>Why this was run</h2>
<p>The 2026-09-23 record (<code>external_benchmark_fitted_defaults_20260923</code>) put default mode at parity with
TReCASE on the RASQUAL/TReCASE generative model (negative-binomial totals, beta-binomial allelic counts), which
hapmixQTL does not assume. That result carried two harness defects: the emulated draws conserved yL + yR exactly and the
summaries were built without yT, which overstated the total channel's inferential variance 4.0-fold; and the allelic
residualizer kept an intercept that production dropped on 2026-09-15. It also predates the log2 point-estimate phenotype
(commit 89bed4e, 2026-09-25), the per-channel t references with the 15-donor allelic admission floor (commit 8a06803) and
Meier's correction (commit a1b2ef4), was run only at N = 200, and compared
against a re-implementation of the TReCASE likelihood rather than the published software.</p>

<h2>What was run</h2>
<p>Held fixed: the generative model and its parameters (mean depth 200, negative-binomial dispersion 0.2, beta-binomial
overdispersion 0.01, allele-specific read fraction 0.25), the random streams, and the harness's per-replicate reseeding
(data from <code>RandomState(seed0 + r)</code>; each arm its own <code>RandomState(seed0 + 900000 + r)</code>), so no arm
can perturb another. Because the data streams are unchanged, the comparator arms at N = 200 had to reproduce the
2026-09-23 record, and they do, <b>{"exactly" if rep["exact"] else "NOT exactly"}</b>, in type-I error at 0.05 and 0.01
and in matched power at all three folds.</p>
<p>The hapmixQTL arm now takes the simulated counts as point estimates and emulated draws for their variance through
<code>summaries_from_point_estimates</code> (the runner's phenotype, log2): allelic draws binomial on the observed split
as before, total draws Poisson around the simulated total (gene-level Gibbs variance of a total is Poisson-like,
CLAUDE.md), with the counting term added in both channels as production adds it. The median total-channel variance is
{dg["200"]["1.0"]["vt_over_delta_median"]:.2f}x the delta-method variance of the phenotype (the first-order Taylor
approximation to the variance of the transformed count; the Poisson draws and the counting term each contribute about 1x), where the old harness was 4.0x. The fit is the library's own default mode:
<code>_prepare_channels</code> with a through-origin allelic channel, then <code>calculate_hapmixqtl_nominal</code>
with the fitted scale. The combined p is referred to the <b>Welch-Satterthwaite</b> degrees of freedom (the
degrees of freedom of a weighted sum of two independent variance estimates,
(w<sub>a</sub> + w<sub>t</sub>)<sup>2</sup> / (w<sub>a</sub><sup>2</sup>/&nu;<sub>a</sub> + w<sub>t</sub><sup>2</sup>/&nu;<sub>t</sub>)),
and the combined SE carries <b>Meier's correction</b> (a first-order inflation, 1 + 4 f<sub>a</sub> f<sub>t</sub>
(1/&nu;<sub>a</sub> + 1/&nu;<sub>t</sub>), for inverse-variance weights estimated from the same residuals, the
Graybill-Deal effect). Three weightings: <b>gibbs</b> (1/v in both channels), <b>split</b> (1/v allelic, unit
total) and <b>plus_one</b> (1/(v + 1) in both, v in squared log2 units).</p>
<p>Comparators. The harness's own three are <b>likelihood-ratio tests</b> (twice the log-likelihood gain of the
alternative over the null, referred to chi-square on 1 df): <b>TReC-only</b> (negative-binomial regression of total
counts), <b>ASE-only</b> (beta-binomial allelic counts, heterozygotes only) and <b>TReCASE, harness</b> (both
likelihoods sharing one allelic fold). <b>asSeq::trecase</b> 0.99.501 ran on the same data, one call per replicate:
Y = totals, Y1/Y2 = haplotype counts, Z = 3 xL + xR, offset = log library size. Two departures were forced or kept.
asSeq cannot fit the baseline TReC model without a covariate: its <code>glmFit</code> intercept-only branch never sets
the convergence flag (glm.c, M = 0; return at glm.c:161), so X = log library size, whose true coefficient given the
offset is 0, costing one df. That covariate is a choice of this run, not a user decision: without it asSeq returns no
test at all. asSeq's defaults were kept otherwise, so its allelic model admits donors with at least 5
allele-specific reads and includes homozygotes at an allelic ratio of one half, where the harness uses heterozygotes
with any read. Two asSeq p-values are reported: its <b>joint p</b> (the joint likelihood-ratio test, the harness's
hypothesis; a failed joint fit counts as a non-rejection, as the harness counts its own failures) and its <b>final
p</b>, the one asSeq prints, which switches to the TReC p when its cis-trans test (a likelihood-ratio test that the
total-count and allelic channels share one fold; it rejects when they disagree) rejects at 0.05 or cannot be computed.
Both are reported because the joint fit fails in 4-7% of replicates: counting those as non-rejections caps the joint
arm's power near 0.95, while the final p hides them by falling back to the TReC p.</p>
<p><b>Matched power</b> is the fraction of alternative replicates whose statistic exceeds the arm's own 95th percentile
under the null, so an anticonservative arm cannot win by being miscalibrated. Its SE is given two ways: the paired SE of
the per-replicate detection difference at the fixed thresholds, and a resampling SE (2,000 resamples, seed 42) that
resamples null and alternative replicates together, so it includes the noise in each estimated threshold. The
2026-09-23 record's "paired" SEs (0.028 / 0.027 / 0.002) equal the unpaired binomial SEs of the two powers; the
resampling SE is the one to judge differences by.</p>

<figure><img alt="Type-I error per arm at nominal 0.05 and 0.01, and matched power minus harness TReCASE, at N = 200 and N = 92" src="data:image/png;base64,{img}">
<figcaption>Left two columns: realized type-I error, 500 null replicates; the grey band is nominal ± 2 Monte Carlo SE.
Right: matched power minus the harness TReCASE, paired on the same replicates, bars ±1 resampling SE (null thresholds
resampled).</figcaption></figure>

<h2>Result: calibration</h2>
<p>Type-I error is the fraction of null replicates with p below alpha; its Monte Carlo SE at nominal is
sqrt(alpha (1 - alpha) / 500). Counts in parentheses. At 0.001 the expected count is 0.5 of 500, so that column cannot
distinguish nominal from several times nominal.</p>
<div class="scroll">{type1_table()}</div>
<p>Default mode with Gibbs weights reads {g("200", "0.05")} / {g("200", "0.01")} at N = 200 and {g("92", "0.05")} /
{g("92", "0.01")} at N = 92, each within 2 Monte Carlo SE of nominal. The 2026-09-23 record read {f4(old["t05"])} /
{f4(old["t01"])} at N = 200 on the same simulated data, with the defective harness and the old N - 2 reference. At 0.01
that is {oc["0.01"]} of {n} null replicates rejected then and {nc01} now ({z_old01:.1f} and {z01("200"):.1f} SE above
nominal): a change of {oc["0.01"] - nc01} rejections, against a Monte Carlo SE of {se01:.1f} rejections for one rate at
nominal 0.01 (sqrt({n} x 0.01 x 0.99)), or {z_chg:.1f} SE. On its own that does not show the expected rate moved. It is
also credited to no single change: between the two runs the two harness fixes moved together with the estimator changes
since the record (the log2 point-estimate phenotype, the per-channel and Welch-Satterthwaite references of commit 8a06803,
and Meier's correction). At N = 92 the gibbs arm at 0.01 is {z01("92"):.1f} SE above nominal. The allelic admission floor never
binds here: every donor carries allele-specific reads, so n<sub>a</sub> is {dg["200"]["1.0"]["n_a"][0]}-{dg["200"]["1.0"]["n_a"][1]}
at N = 200 and {dg["92"]["1.0"]["n_a"][0]}-{dg["92"]["1.0"]["n_a"][1]} at N = 92 (heterozygotes at the tested variant:
median {dg["200"]["1.0"]["n_het"][1]} and {dg["92"]["1.0"]["n_het"][1]}). Meier's factor has a median of {meier["200"][0]:.4f}
(max {meier["200"][1]:.4f}) at N = 200 and {meier["92"][0]:.4f} (max {meier["92"][1]:.4f}) at N = 92, and the combined
Welch-Satterthwaite dof ranges {dof["200"][0]:.0f}-{dof["200"][2]:.0f} and {dof["92"][0]:.0f}-{dof["92"][2]:.0f}: at these
sizes both corrections move the p by little.</p>
<p>asSeq's joint p is calibrated ({f4(t1("200", "TReCASE (asSeq joint p)", "0.05")["rate"])} and
{f4(t1("92", "TReCASE (asSeq joint p)", "0.05")["rate"])} at 0.05). Its printed final p is not
({f4(t1("200", "TReCASE (asSeq final p)", "0.05")["rate"])} and {f4(t1("92", "TReCASE (asSeq final p)", "0.05")["rate"])}).
The excess is the cis-trans switch. At N = 200, {sw["n"]} null replicates had a cis-trans p below 0.05, and the TReC p
that asSeq then reports rejected in {sw["rejected_005"]} of them, because both tests respond to the same chance
total-count effect. The joint
fit failed in {fails["200"][0]} and {fails["92"][0]} of 500 null replicates, and in {min(fails["200"] + fails["92"])} to
{max(fails["200"] + fails["92"])} of 500 in every condition; over all 4,000 replicates {theta} of the {jf_all} failed
joint fits stopped while estimating the beta-binomial overdispersion ({'all' if c52 == theta else f'{c52} of them'} with lbfgsb's abnormal line-search
termination, code 52), and the trace logs show {ase_b} baseline and {ase_f} per-variant ASE-model failures among the
others. The plasmode runs show the same failure (<code>joint_theta</code>).</p>

<h2>Result: matched power</h2>
<p>Each entry: matched power ± resampling SE; below it, the difference from the harness TReCASE with its paired and
resampling SE.</p>
<h3 style="font-size:15px">N = 200</h3><div class="scroll">{power_table("200")}</div>
<h3 style="font-size:15px">N = 92</h3><div class="scroll">{power_table("92")}</div>

<h2>The critique, and what it changed</h2>
<p>The strongest objection to reading these differences is the threshold. Each arm's matched power depends on its own
95th null percentile estimated from 500 replicates, and the paired SE ignores that noise. With it included (resampling),
no hapmixQTL weighting differs from the harness TReCASE by more than {max(zb.values()):.1f} SE at any fold at either N.
The differences that the paired SE alone would put at 2 SE or more are {flag_txt.replace(' and ', '; ')}. So this benchmark does not separate the
three weightings in power. At N = 200, split and plus_one detect exactly as many replicates as the harness TReCASE at
folds 1.05 and 1.20.</p>
<p>asSeq trails the harness TReCASE at fold 1.20 (joint p {aj_d["diff"]:+.3f} at N = 200, {abs(aj_d["diff"]) / aj_d["se_resample"]:.1f}
resampling SE), and the cause is its failed joint fits: {jf12} of 500 at N = 200 cap its joint-p power at
{1 - jf12 / 500:.3f}, and it reads {aj["matched"]:.3f}. The harness likelihood is
therefore the better ceiling here. The asSeq numbers describe asSeq 0.99.501 as built on this host, whose
<code>lbfgsb1.c</code> needed a BLAS prototype include to run at all (CLAUDE.md); that build cannot be excluded as a
contributor to the failures.</p>

<h2>What it means</h2>
<p>On a generative model hapmixQTL does not assume, current default mode at all three weightings is calibrated at 0.05
and 0.01 within 2 Monte Carlo SE at N = 200 and at the cohort's N = 92, and its matched power is indistinguishable from
the joint likelihood of the generating model. The 2026-09-23 conclusion holds on the fixed harness with the current
estimator. Its tail caveat at 0.01 ({old["t01"] / 0.01:.1f}x nominal, {oc["0.01"]} of {n}) reads
{t1("200", "hapmixQTL gibbs", "0.01")["rate"] / 0.01:.1f}x ({nc01} of {n}) at N = 200, a change of {z_chg:.1f} Monte
Carlo SE that no single change is credited with, because the harness fixes and the estimator changes moved together;
at N = 92 the gibbs arm reads
{t1("92", "hapmixQTL gibbs", "0.01")["rate"] / 0.01:.1f}x nominal, {z01("92"):.1f} SE above it, so the tail is not settled at the cohort's size. Against asSeq as a user would run
it (final p), default mode is better calibrated at 0.05 and more powerful at fold 1.20 at both N ({min(zf12):.1f} to
{max(zf12):.1f} resampling SE); at folds 1.05 and 1.10 only {'; '.join(over2) if over2 else 'none'} clear 2 resampling SE,
and the rest are not separated.</p>
<p>Bounds. Nothing below 0.01 is resolved: 500 replicates put 0.5 expected rejections at 0.001. This is one depth
(median total {dg["200"]["1.0"]["median_T"]:.0f} reads, median allele-specific {dg["200"]["1.0"]["median_n_as"]:.0f}), one
overdispersion pair and no covariates. Every donor is informative in the allelic channel, including homozygotes at the
tested variant, so none of the low-coverage and zero-read behavior that separates the weightings on BrainVar (the
pipeline-rules open decision) is exercised here. The emulated draws are binomial and Poisson resampling, not Salmon
draws against a diploid transcriptome, and both channels count the counting noise twice, as production does. The
weightings differ in matched power by at most {spread:.3f} here, below what 500 replicates can resolve.</p>
<p class="sub">Code: <code>tests/ase_external_benchmark.py</code> (harness), <code>scripts/external_benchmark_mirror.py</code>
(arms | trecase | summarize), <code>scripts/external_benchmark_mirror_trecase.R</code>,
<code>scripts/external_benchmark_mirror_report.py</code>. Data: <code>summary.json</code>, <code>per_replicate.tsv.gz</code>,
<code>arms/</code>, <code>trecase/</code>.</p>
</main></body></html>
'''
    tmp = OUT / 'report.html.tmp'
    tmp.write_text(html)
    os.replace(tmp, OUT / 'report.html')
    print(f'wrote {OUT / "report.html"} and {OUT / "fig_mirror.png"}')


if __name__ == '__main__':
    main()

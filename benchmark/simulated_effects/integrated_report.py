"""One page for the simulated-effects benchmark: an executive summary of figures over both gene sets, then each set's full
record from 08_report.py (user request 2026-10-02: less text, more figures, one integrated report).

Reads, per gene set (common.GENE_SETS, root under common.D): summary.json (06_score.py), report/fragments.json (08_report.py's
sections, so every guarded sentence and check of 08 carries over unchanged) and gene_level_null/ (gene_level_null.py).
The summary charts are inline SVG in the page's colour tokens; each finding sentence is worded from the numbers it states.
The full record keeps 08's text; methods, run facts and checks, the limits and every table are folded into collapsible
blocks, not removed.

Output: OUT/index.html. Usage: integrated_report.py   (run 06 and 08 for both sets first)
"""
import html
import json
import re

import numpy as np
import pandas as pd

import common as C

SETS = [('deep', 'Deep set', 'corrected_null_store_20260925', 's1'), ('lowcov', 'Low-coverage set', 'stratum30_100', 's2')]
ROOT = {k: C.D / C.GENE_SETS[g]['root'] for k, _, g, _ in SETS}
OUT = C.D / 'simulated_effects_integrated_20261002'
ARMS = [('split', 'split (shipped)'), ('unit', 'unit weights'), ('gibbs', 'gibbs'), ('mixqtl', 'mixQTL, published'),
        ('tensorqtl', 'tensorQTL, total only'), ('rasqual', 'RASQUAL'), ('trecase', 'TReCASE')]
SHORT = dict(split='split', unit='unit', gibbs='gibbs', mixqtl='mixQTL pub.', tensorqtl='tensorQTL', rasqual='RASQUAL', trecase='TReCASE')
CAT = ['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7']   # categorical slots, fixed order by ARMS
BETAS = ('0.2', '0.4', '0.8')
esc = html.escape
S = {k: json.loads((ROOT[k] / 'summary.json').read_text()) for k, *_ in SETS}
FR = {k: json.loads((ROOT[k] / 'report' / 'fragments.json').read_text()) for k, *_ in SETS}
GL = {k: json.loads((ROOT[k] / 'gene_level_null' / 'summary.json').read_text()) for k, *_ in SETS}
NAME = {k: n for k, n, *_ in SETS}
f2 = lambda x: f'{x:.2f}'   # noqa: E731
f3 = lambda x: f'{x:.3f}'   # noqa: E731


def where(d, x, lo='lo', hi='hi'):
    return 'above' if d[lo] > x else 'below' if d[hi] < x else 'includes'


# ---------------------------------------------------------------- SVG building blocks (theme tokens, hover via <title>)
def svg(w, h, body, label):
    return f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{esc(label)}">{body}</svg>'


def hline(x1, x2, y, cls):
    return f'<line x1="{x1:.1f}" x2="{x2:.1f}" y1="{y:.1f}" y2="{y:.1f}" class="{cls}"/>'


def vline(x, y1, y2, cls):
    return f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{y1:.1f}" y2="{y2:.1f}" class="{cls}"/>'


def text(x, y, s, cls='tick', anchor='start'):
    return f'<text x="{x:.1f}" y="{y:.1f}" class="{cls}" text-anchor="{anchor}">{s}</text>'


def mark(x, y, cls, hollow, tip, r=5):
    return (f'<g class="{cls}"><title>{esc(tip)}</title><circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" '
            f'class="{"hollow" if hollow else "dot"}"/></g>')


def nice_ticks(lo, hi, n=5):
    step = 10 ** np.floor(np.log10((hi - lo) / n))
    for m in (1, 2, 2.5, 5, 10):
        if (hi - lo) / (step * m) <= n:
            step *= m
            break
    return [round(v, 10) for v in np.arange(np.ceil(lo / step) * step, hi + step / 2, step)]


def dot_ci_columns(cats, series, ref, ylab, label, w=600, h=270):
    """Grouped dot + interval chart: cats on x, series [(name, cls, hollow, [(v, lo, hi, tip)])] offset within a cat."""
    L, R, T, B = 52, 16, 26, 44
    vals = [x for _, _, _, pts in series for p in pts if p for x in p[:3]] + [ref]
    lo, hi = min(vals), max(vals)
    pad = (hi - lo) * 0.08
    lo, hi = lo - pad, hi + pad
    Y = lambda v: T + (h - T - B) * (1 - (v - lo) / (hi - lo))   # noqa: E731
    cw = (w - L - R) / len(cats)
    b = ''.join(hline(L, w - R, Y(v), 'grid') + text(L - 8, Y(v) + 4, f'{v:g}', anchor='end') for v in nice_ticks(lo, hi))
    b += hline(L, w - R, Y(ref), 'ref')
    for i, c in enumerate(cats):
        cx = L + cw * (i + 0.5)
        b += text(cx, h - 18, esc(c), anchor='middle')
        for j, (_, cls, hollow, pts) in enumerate(series):
            p = pts[i]
            if p is None:
                continue
            x = cx + (j - (len(series) - 1) / 2) * 16
            b += (f'<g class="{cls}"><title>{esc(p[3])}</title>{vline(x, Y(p[1]), Y(p[2]), "ci")}'
                  f'<circle cx="{x:.1f}" cy="{Y(p[0]):.1f}" r="5" class="{"hollow" if hollow else "dot"}"/></g>')
    b += text(L, T - 10, esc(ylab), 'axis')
    return svg(w, h, b, label)


def dot_ci_rows(rows, series, ref, xlab, label, w=600, rowh=26, left=150, fmt='{:g}'):
    """Horizontal dot + interval rows: rows [label], series [(name, cls, hollow, [(v, lo, hi, tip) per row])]."""
    T, B = 30, 34
    h = T + rowh * len(rows) + B
    vals = [x for _, _, _, pts in series for p in pts if p for x in p[:3]] + [ref]
    lo, hi = min(vals), max(vals)
    pad = (hi - lo) * 0.06
    lo, hi = lo - pad, hi + pad
    X = lambda v: left + (w - left - 16) * (v - lo) / (hi - lo)   # noqa: E731
    b = ''.join(vline(X(v), T - 6, h - B, 'grid') + text(X(v), h - B + 16, fmt.format(v), anchor='middle')
                for v in nice_ticks(lo, hi))
    b += vline(X(ref), T - 6, h - B, 'ref')
    for i, r in enumerate(rows):
        y = T + rowh * i + rowh / 2
        b += text(left - 10, y + 4, esc(r), anchor='end')
        for j, (_, cls, hollow, pts) in enumerate(series):
            p = pts[i]
            if p is None:
                continue
            yy = y + (j - (len(series) - 1) / 2) * 8
            b += (f'<g class="{cls}"><title>{esc(p[3])}</title><line x1="{X(p[1]):.1f}" x2="{X(p[2]):.1f}" y1="{yy:.1f}" '
                  f'y2="{yy:.1f}" class="ci"/><circle cx="{X(p[0]):.1f}" cy="{yy:.1f}" r="4.5" class="{"hollow" if hollow else "dot"}"/></g>')
    b += text(left, 14, esc(xlab), 'axis')
    return svg(w, h, b, label)


def legend(items):
    return '<p class="legend">' + ''.join(
        f'<span><span class="key {cls}{" hollowkey" if hollow else ""}"></span>{esc(n)}</span>' for n, cls, hollow in items) + '</p>'


# ---------------------------------------------------------------- the summary figures
def fig_precision():
    """split's and gibbs's combined squared error over unit weights', causal variant (pipeline scale) and anchor."""
    cats = [f'|beta| {b}' for b in BETAS] + ['null genes']
    series = []
    for arm, lab, cls in (('split', 'split', 'c1'), ('gibbs', 'gibbs', 'c3')):
        for k, n, _, _ in SETS:
            pts = []
            for b in BETAS:
                d = S[k]['precision'][f'beta{b}'][arm]['combined']['nonnull']['ratio_vs_unit']['all']
                pts.append((d['value'], d['lo'], d['hi'], f'{lab}, {n}, |beta| {b}: {f2(d["value"])} [{f2(d["lo"])}, {f2(d["hi"])}]'))
            d = S[k]['precision']['beta0.0'][arm]['combined']['null']['ratio_vs_unit']['all']
            pts.append((d['value'], d['lo'], d['hi'], f'{lab}, {n}, anchor null genes: {f2(d["value"])} [{f2(d["lo"])}, {f2(d["hi"])}]'))
            series.append((f'{lab}, {n}', cls, k == 'lowcov', pts))
    chart = dot_ci_columns(cats, series, 1.0, 'squared error over unit weights\' (below 1 = more precise)', 'Precision against unit weights')
    words = []
    for k, n, _, _ in SETS:
        ds = [S[k]['precision'][f'beta{b}']['split']['combined']['nonnull']['ratio_vs_unit']['all'] for b in BETAS]
        below = all(where(d, 1.0, 'lo', 'hi') == 'below' for d in ds)
        words.append(f'{n.lower()} {" / ".join(f2(d["value"]) for d in ds)}'
                     + (' (every interval below 1)' if below else ' (intervals include 1)'))
    g = {k: [S[k]['precision'][f'beta{b}']['gibbs']['combined']['nonnull']['ratio_vs_unit']['all'] for b in BETAS] for k, *_ in SETS}
    gw = '; '.join(f'{NAME[k].lower()} {" / ".join(f2(d["value"]) for d in ds)}'
                   + (' (every interval above 1)' if all(where(d, 1.0) == 'above' for d in ds) else
                      ' (intervals above 1 at ' + (', '.join(f'|beta| {b}' for b, d in zip(BETAS, ds) if where(d, 1.0) == 'above') or 'no |beta|') + ')')
                   for k, ds in g.items())
    finding = ('split, the shipped weighting, against its control without the Gibbs draws (unit weights): combined squared error '
               + '; '.join(words) + '. gibbs, which also weights the total channel by its Gibbs variance: ' + gw + '.')
    leg = legend([('split', 'c1', False), ('gibbs', 'c3', False), ('deep set: filled', 'c0', False), ('low-coverage set: open', 'c0', True)])
    return finding, leg + chart, ('Squared error of the combined slope over that of unit weights on the same units, at the causal '
                                  'variant (pipeline-scale truth) and on the anchor dataset\'s null genes; gene-clustered 95% intervals.')


def fig_power():
    """Power at 5% realized FDP against |beta|, every Salmon-input method, one panel per set."""
    W, H, L, R, T, B = 620, 280, 40, 150, 24, 40
    pw = (W - L - R)
    out = []
    for k, n, _, _ in SETS:
        X = lambda i: L + pw * i / (len(BETAS) - 1)   # noqa: E731
        Y = lambda v: T + (H - T - B) * (1 - v)   # noqa: E731
        b = ''.join(hline(L, L + pw, Y(v), 'grid') + text(L - 8, Y(v) + 4, f'{v:g}', anchor='end') for v in (0, 0.25, 0.5, 0.75, 1))
        b += ''.join(text(X(i), H - 18, f'|beta| {x}', anchor='middle') for i, x in enumerate(BETAS))
        ends = []
        for (arm, lab), cls in zip(ARMS, CAT):
            v = [S[k]['ranking'][f'beta{x}'][arm]['fdp_matched']['all']['power'] for x in BETAS]
            pts = ' '.join(f'{X(i):.1f},{Y(y):.1f}' for i, y in enumerate(v))
            b += (f'<g class="{cls}"><title>{esc(lab)}: {" / ".join(f3(y) for y in v)}</title><polyline points="{pts}" class="line"/>'
                  + ''.join(f'<circle cx="{X(i):.1f}" cy="{Y(y):.1f}" r="3.5" class="pt"/>' for i, y in enumerate(v)) + '</g>')
            ends.append([Y(v[-1]), lab, cls])
        ends.sort()
        for i in range(1, len(ends)):
            ends[i][0] = max(ends[i][0], ends[i - 1][0] + 13)
        b += ''.join(text(L + pw + 10, y + 4, esc(lab), f'lab {cls}') for y, lab, cls in ends)
        b += text(L, 14, esc(n), 'ptitle')
        out.append(svg(W, H, b, f'Power at 5% realized FDP, {n}'))
    sp = {k: [S[k]['ranking'][f'beta{x}']['split']['fdp_matched']['all']['power'] for x in BETAS] for k, *_ in SETS}
    top = {k: [max((S[k]['ranking'][f'beta{x}'][a]['fdp_matched']['all']['power'], a) for a, _ in ARMS)[1] for x in BETAS] for k, *_ in SETS}
    lead = [n.lower() for k, n, *_ in SETS if all(t == 'split' for t in top[k])]
    finding = (f'At |beta| 0.8 split finds {f2(sp["deep"][-1])} of non-null genes on the deep set and {f2(sp["lowcov"][-1])} on the '
               f'low-coverage set, holding false discoveries to 5%. '
               + (f'It is the highest of the seven methods at every effect size on the {" and ".join(lead)}.' if lead else
                  'No single method is highest at every effect size on both sets.'))
    return finding, '<div class="pair">' + ''.join(out) + '</div>', (
        'Power at 5% realized false-discovery proportion: genes are ranked by their lead variant\'s p, the ranking is walked down '
        'to the deepest point where at most 5% of the genes called are truly null, and power is the share of non-null genes '
        'above that point. Three datasets per effect size; no interval. Hover a line for its values.')


def fig_calibration():
    """Null-gene rate at 0.05 on the anchor, every method, both sets."""
    series = []
    for k, n, _, cls in SETS:
        pts = []
        for arm, lab in ARMS:
            d = S[k]['null']['beta0.0'][arm]['combined']['all']['0.05']
            pts.append((d['rate'], d['lo'], d['hi'], f'{lab}, {n}: {d["rate"]:.4f} [{d["lo"]:.4f}, {d["hi"]:.4f}]'))
        series.append((n, cls, k == 'lowcov', pts))
    chart = dot_ci_rows([lab for _, lab in ARMS], series, 0.05, 'share of null-gene tests with p < 0.05 (nominal 0.05, dashed)',
                        'Null-gene rate by method')
    above = {k: [lab for arm, lab in ARMS if S[k]['null']['beta0.0'][arm]['combined']['all']['0.05']['lo'] > 0.05] for k, *_ in SETS}
    within = [lab for arm, lab in ARMS if all(where(S[k]['null']['beta0.0'][arm]['combined']['all']['0.05'], 0.05) != 'above' for k, *_ in SETS)]
    above = {k: [SHORT[a] for a, lab in ARMS if lab in v] for k, v in above.items()}
    within = [SHORT[a] for a, lab in ARMS if lab in within]
    finding = (f'On the all-null anchor, {", ".join(within)} reject at or below nominal on both sets (intervals include or lie below 0.05)'
               + ''.join(f'; above nominal on the {NAME[k].lower()}: {", ".join(v)}' for k, v in above.items() if v) + '.')
    return finding, legend([(n, cls, k == 'lowcov') for k, n, _, cls in SETS]) + chart, (
        'Share of the null genes\' tested variants whose combined nominal p is below 0.05, on the one all-null anchor dataset of '
        'each set; gene-clustered 95% intervals (genes resampled with replacement).')


def gl_counts(k):
    files = sorted((ROOT[k] / 'gene_level_null').glob('cis_r*.parquet'))
    return np.array([int((pd.read_parquet(f).pval_beta < 0.05).sum()) for f in files])


def fig_gene_level():
    """split's gene-level p over 100 all-null datasets per set: share below thresholds, and the anchor among the datasets."""
    alphas = ('0.05', '0.01', '0.001')
    series = []
    for k, n, _, cls in SETS:
        pts = []
        for al in alphas:
            r, a = GL[k]['rates'][al], float(al)
            pts.append((r['share'] / a, r['lo'] / a, r['hi'] / a, f'{n}, pval_beta < {al}: {r["share"]:.4f} [{r["lo"]:.4f}, {r["hi"]:.4f}]'))
        series.append((n, cls, k == 'lowcov', pts))
    left = dot_ci_columns([f'p < {a}' for a in alphas], series, 1.0, 'share below threshold ÷ threshold (1 = nominal)',
                          'Gene-level p on all-null data', w=420, h=260)
    W, H = 420, 260
    b = ''
    for j, (k, n, _, cls) in enumerate(SETS):
        c = gl_counts(k)
        anc = GL[k]['rates']['0.05']['anchor_genes_below']
        x0, pw, top, bot = 16 + j * 205, 190, 40, H - 44
        nb = 14
        hist = np.bincount(np.minimum(c, nb - 1), minlength=nb)
        bw, hy = pw / nb, max(hist.max(), 1)
        b += text(x0, 18, esc(n), 'ptitle')
        for v, m in enumerate(hist):
            if m:
                hh = (bot - top) * m / hy
                b += (f'<rect x="{x0 + v * bw + 1:.1f}" y="{bot - hh:.1f}" width="{bw - 2:.1f}" height="{hh:.1f}" rx="2" '
                      f'class="bar {cls}"><title>{m} of 100 datasets with {v}{"+" if v == nb - 1 else ""} null genes below 0.05</title></rect>')
        ax = x0 + (min(anc, nb - 1) + 0.5) * bw
        b += vline(ax, top - 8, bot, 'anchor') + text(ax + (4 if anc < 8 else -4), top - 12 + 14, f'anchor {anc}', 'anno', 'start' if anc < 8 else 'end')
        b += ''.join(text(x0 + (v + 0.5) * bw, bot + 16, f'{v}{"+" if v == nb - 1 else ""}', anchor='middle') for v in (0, 5, 10, 13))
    b += text(16, H - 6, 'null genes below 0.05 per all-null dataset (5 expected)', 'axis')
    right = svg(W, H, b, 'Null genes below 0.05 per all-null dataset')
    bh = {k: GL[k]['bh_any_call_share'] for k, *_ in SETS}
    pos = {k: GL[k]['rates']['0.05']['anchor_share_of_replicates_at_or_above'] for k, *_ in SETS}
    ok = all(bh[k] <= 0.05 for k, *_ in SETS)
    finding = (('Gene discovery holds its error rate' if ok else 'Gene discovery exceeds its error rate on at least one set') + f': over 100 fresh all-null datasets per set, split\'s gene-level p falls below 0.05 '
               f'for {GL["deep"]["rates"]["0.05"]["share"]:.3f} and {GL["lowcov"]["rates"]["0.05"]["share"]:.3f} of genes, and '
               f'Benjamini-Hochberg at 5% calls any gene in {bh["deep"] * 100:.0f} and {bh["lowcov"] * 100:.0f} of 100 datasets '
               f'(at most 5 expected). The benchmark\'s own anchor was a low draw on the deep set ({pos["deep"] * 100:.0f}% of datasets '
               f'have as many null genes below 0.05) and a high one on the low-coverage set ({pos["lowcov"] * 100:.0f}%).')
    return finding, legend([(n, cls, k == 'lowcov') for k, n, _, cls in SETS]) + f'<div class="pair">{left}{right}</div>', (
        'pval_beta: the gene-level p from 1,000 permutations of donor records with haplotype-label swaps, smoothed by a fitted '
        'Beta distribution. Left: share of gene-dataset units below each threshold, over the threshold, with gene-clustered 95% '
        'intervals. Right: how many of a dataset\'s 100 null genes fall below 0.05, across 100 datasets; the line marks the '
        'benchmark\'s anchor dataset. This null is built by the same permutation the test uses, so it checks the machinery, '
        'not calibration on real data.')


def fig_bias():
    """Recovered share of beta by the combined slope at |beta| 0.4, every method, both sets."""
    series = []
    for k, n, _, cls in SETS:
        pts = []
        for arm, lab in ARMS:
            d = S[k]['recovery']['beta0.4'][arm]['combined']['bias_count']['all']
            pts.append((d['mean'], d['lo'], d['hi'], f'{lab}, {n}: {f3(d["mean"])} [{f3(d["lo"])}, {f3(d["hi"])}]'))
        series.append((n, cls, k == 'lowcov', pts))
    chart = dot_ci_rows([lab for _, lab in ARMS], series, 1.0, 'mean slope ÷ true slope at the causal variant, |beta| 0.4 (1 = unbiased)',
                        'Recovered share of the effect')
    short = {k: [SHORT[arm] for arm, lab in ARMS if S[k]['recovery']['beta0.4'][arm]['combined']['bias_count']['all']['hi'] < 1] for k, *_ in SETS}
    sp = {k: S[k]['recovery']['beta0.4']['split']['combined']['bias_count']['all'] for k, *_ in SETS}
    finding = (f'split\'s combined slope recovers {f2(sp["deep"]["mean"])} of the injected effect on the deep set and '
               f'{f2(sp["lowcov"]["mean"])} on the low-coverage set. '
               + ' '.join(f'Short of the effect on the {NAME[k].lower()}, interval below 1: {", ".join(v)}.' for k, v in short.items() if v))
    return finding, legend([(n, cls, k == 'lowcov') for k, n, _, cls in SETS]) + chart, (
        'Bias ratio: the mean over non-null genes of the estimated slope over the true slope at the causal variant, every method '
        'on the count-scale truth (beta itself for the allelic and joint estimands); gene-clustered 95% intervals.')


# ---------------------------------------------------------------- the full record, folded
def fold(label, inner, cls=''):
    return f'<details class="fold {cls}"><summary>{esc(label)}</summary><div class="foldbody">{inner}</div></details>'


def fold_tables(h):
    return re.sub(r'(<table>.*?</table>)', lambda m: fold('Table', m.group(1), 'tbl'), h, flags=re.S)


def fold_h3(h, titles):
    """Fold the h3 subsections of h whose heading starts with one of titles whole; in every other h3 subsection keep the
    heading and figures in view and fold its text and tables into one block."""
    parts = re.split(r'(?=<h[23]>)', h)
    out = []
    for p in parts:
        m = re.match(r'<h3>(.*?)</h3>', p, flags=re.S)
        if not m:
            out.append(p)
        elif any(m.group(1).startswith(t) for t in titles):
            out.append(fold(re.sub('<.*?>', '', m.group(1)), p[m.end():]))
        else:
            rest = p[m.end():]
            figs = re.findall(r'<figure>.*?</figure>', rest, flags=re.S)
            text_ = re.sub(r'<figure>.*?</figure>', '', rest, flags=re.S)
            out.append(p[:m.end()] + ''.join(figs) + (fold('Text and tables', text_) if text_.strip() else ''))
    return ''.join(out)


def record(k):
    fr = FR[k]
    head = re.sub(r'<h1>.*?</h1>', '', fr['head'], flags=re.S)
    body = [fold('Provenance', head), fr['why'], fold('2. What was run: generator, arms, scoring', fr['run'])]
    res = fold_h3(fr['results'], ('Run facts and checks', '3.8', '3.9'))
    body.append(res)
    if fr['contrast']:
        body.append(fold_tables(fr['contrast']))
    for key, lab in (('critique', None), ('meaning', None), ('limits', 'Limits: what this analysis cannot establish')):
        if fr.get(key):
            body.append(fold(lab, fr[key]) if lab else fold_tables(fr[key]))
    return '\n'.join(body)


CSS = '''
/* layout: one reading column with a sticky section bar; summary cards in a grid, the full record below, folded */
:root { --bg: #f7f7f4; --panel: #ffffff; --ink: #15171a; --ink2: #575b61; --rule: #dfe0db; --accent: #2a78d6;
  --c1: #2a78d6; --c2: #eb6834; --c3: #1baf7a; --c4: #eda100; --c5: #e87ba4; --c6: #008300; --c7: #4a3aa7; --c0: #575b61;
  --s1: #2a78d6; --s2: #eb6834;
  --display: "Source Serif 4", Georgia, serif; --body: "IBM Plex Sans", system-ui, sans-serif; --mono: "IBM Plex Mono", ui-monospace, monospace; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --bg: #141517; --panel: #1c1d20; --ink: #f1f1ee; --ink2: #b4b6b9;
  --rule: #33353a; --accent: #3987e5; --c1: #3987e5; --c2: #d95926; --c3: #199e70; --c4: #c98500; --c5: #d55181; --c6: #008300;
  --c7: #9085e9; --c0: #b4b6b9; --s1: #3987e5; --s2: #d95926; color-scheme: dark; } }
:root[data-theme="dark"] { --bg: #141517; --panel: #1c1d20; --ink: #f1f1ee; --ink2: #b4b6b9; --rule: #33353a; --accent: #3987e5;
  --c1: #3987e5; --c2: #d95926; --c3: #199e70; --c4: #c98500; --c5: #d55181; --c6: #008300; --c7: #9085e9; --c0: #b4b6b9;
  --s1: #3987e5; --s2: #d95926; color-scheme: dark; }
body { background: var(--bg); color: var(--ink); font: 15px/1.55 var(--body); }
main { max-width: 1120px; margin: 0 auto; padding-inline: 16px; padding-block: 0 72px; }
nav.bar { position: sticky; top: env(safe-area-inset-top, 0px); z-index: 2; background: var(--bg); border-bottom: 1px solid var(--rule);
  display: flex; gap: 18px; flex-wrap: wrap; padding: 10px 0; font-size: 13.5px; }
nav.bar a { color: var(--ink2); text-decoration: none; } nav.bar a:hover, nav.bar a:focus-visible { color: var(--accent); }
header.top { padding-block: 32px 8px; }
h1 { font: 600 32px/1.15 var(--display); margin: 0; text-wrap: balance; letter-spacing: -0.01em; }
.dek { color: var(--ink2); max-width: 70ch; margin: 10px 0 0; }
h2 { font: 600 22px/1.25 var(--display); margin: 44px 0 12px; text-wrap: balance; }
h3 { font-size: 16px; margin: 28px 0 8px; }
p { max-width: 75ch; }
.eyebrow { font: 600 11.5px var(--mono); letter-spacing: 0.08em; text-transform: uppercase; color: var(--accent); margin: 0 0 4px; }
.cards { display: grid; gap: 20px; grid-template-columns: repeat(auto-fit, minmax(min(100%, 520px), 1fr)); }
.card { background: var(--panel); border: 1px solid var(--rule); border-radius: 8px; padding: 18px; min-width: 0; display: flex; flex-direction: column; gap: 8px; }
.card.wide { grid-column: 1 / -1; }
.card h3 { margin: 0; font: 600 17px/1.3 var(--display); }
.finding { margin: 0; }
.cap { color: var(--ink2); font-size: 12.5px; margin: 0; }
.pair { display: grid; gap: 12px; grid-template-columns: repeat(auto-fit, minmax(min(100%, 380px), 1fr)); }
svg { width: 100%; height: auto; display: block; font-family: var(--body); }
.tick { fill: var(--ink2); font-size: 11px; font-variant-numeric: tabular-nums; } .axis { fill: var(--ink2); font-size: 11.5px; }
.ptitle { fill: var(--ink); font-size: 12.5px; font-weight: 600; } .anno { fill: var(--ink); font-size: 11.5px; } .lab { font-size: 11.5px; fill: var(--ink); }
.grid { stroke: var(--rule); stroke-width: 1; } .ref { stroke: var(--ink2); stroke-width: 1.2; stroke-dasharray: 4 3; }
.anchor { stroke: var(--ink); stroke-width: 1.5; }
.ci { stroke-width: 2; stroke-linecap: round; } .line { fill: none; stroke-width: 2; } .pt { stroke: var(--panel); stroke-width: 1.5; }
.hollow { fill: var(--panel); stroke-width: 2; } .dot { stroke: var(--panel); stroke-width: 1.5; }
.c1 { --k: var(--c1); } .c2 { --k: var(--c2); } .c3 { --k: var(--c3); } .c4 { --k: var(--c4); } .c5 { --k: var(--c5); }
.c6 { --k: var(--c6); } .c7 { --k: var(--c7); } .c0 { --k: var(--c0); } .s1 { --k: var(--s1); } .s2 { --k: var(--s2); }
g .ci, g .line, g .hollow { stroke: var(--k); } g .dot, g .pt, .bar { fill: var(--k); }
g:hover .dot, g:hover .pt, g:hover .hollow { stroke: var(--ink); }
.legend { display: flex; gap: 16px; flex-wrap: wrap; font-size: 12.5px; color: var(--ink2); margin: 0; }
.key { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; vertical-align: -1px; background: var(--k); }
.key.hollowkey { background: transparent; border: 2px solid var(--k); width: 6px; height: 6px; }
.limits { background: var(--panel); border: 1px solid var(--rule); border-radius: 8px; padding: 14px 18px; }
.limits ul { margin: 6px 0 0; padding-left: 18px; } .limits li { margin: 4px 0; max-width: 90ch; }
.record { border-top: 2px solid var(--ink); margin-top: 56px; }
details.fold { border: 1px solid var(--rule); border-radius: 6px; margin: 12px 0; background: var(--panel); }
details.fold > summary { cursor: pointer; padding: 8px 12px; color: var(--ink2); font-size: 13.5px; }
details.fold > summary:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
details.fold[open] > summary { border-bottom: 1px solid var(--rule); }
.foldbody { padding: 4px 14px 12px; overflow-x: auto; }
details.tbl { border-style: dashed; } details.tbl > summary { font-size: 12.5px; }
table { border-collapse: collapse; font-size: 12.5px; margin: 8px 0; }
th, td { border-bottom: 1px solid var(--rule); padding: 4px 8px; text-align: left; vertical-align: top; }
th { background: var(--bg); font-weight: 600; } td { font-variant-numeric: tabular-nums; }
.record figure { margin: 16px 0 24px; } .record figure img { max-width: 100%; height: auto; background: #ffffff; border-radius: 6px; padding: 6px; }
.record figcaption, .sub { color: var(--ink2); font-size: 13px; max-width: 90ch; } .pipe { color: var(--ink2); }
@media (prefers-reduced-motion: no-preference) { html { scroll-behavior: smooth; } }
'''


def main():
    for k, *_ in SETS:
        if 'rasqual' not in S[k]['joint_arms']:
            raise SystemExit(f'{ROOT[k]}/summary.json: RASQUAL not scored; run 06 with RASQUAL in common.JOINT first')
    figs = [('Precision', 'Combined slope against unit weights', fig_precision()),
            ('Discovery', 'Real effects found at 5% false discoveries', fig_power()),
            ('Gene-level error rate', 'Gene calls on all-null data', fig_gene_level()),
            ('Calibration', 'Null-gene rejections by method', fig_calibration()),
            ('Bias', 'Share of the injected effect recovered', fig_bias())]
    cards = ''.join(f'<article class="card{" wide" if i in (1, 2) else ""}"><p class="eyebrow">{esc(e)}</p><h3>{esc(t)}</h3>'
                    f'<p class="finding">{esc(fd)}</p>{fig}<p class="cap">{esc(cap)}</p></article>'
                    for i, (e, t, (fd, fig, cap)) in enumerate(figs))
    limits = '''<section class="limits"><p class="eyebrow">What this cannot show</p><ul>
<li>Null genes are made by permuting donor records against genotypes, the same operation the permutation test uses, so the gene-level error rate is checked on the test's own null; calibration on real data, where relatedness or batch could matter, is untested.</li>
<li>Each effect size rests on 3 datasets of 100 genes with 50 non-null, and each set has one all-null anchor dataset; power at a realized false-discovery proportion carries no interval.</li>
<li>Effects are made by thinning reads, which leaves the low-coverage set's allelic Gibbs variance and zero-haplotype counts more favourable than Salmon would give at the same depth; every arm's allelic figures there are optimistic by an amount this run does not measure.</li>
<li>RASQUAL and TReCASE see Salmon's haplotype estimates rounded to integers, not reads at heterozygous SNPs; the native-input arms in the full record are the step toward their own inputs.</li>
</ul></section>'''
    nav = ('<nav class="bar"><a href="#summary">Summary</a>' + ''.join(f'<a href="#{k}">{esc(n)}: full record</a>' for k, n, *_ in SETS)
           + '</nav>')
    page = (f'<title>Simulated-effects eQTL benchmark</title>\n'
            '<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>\n'
            '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:'
            'wght@400;600&family=Source+Serif+4:opsz,wght@8..60,600&display=swap">\n'
            f'<style>{CSS}</style>\n<main>{nav}<header class="top"><h1>Simulated-effects eQTL benchmark</h1>'
            '<p class="dek">Known cis effects injected into the BrainVar cohort\'s own Salmon output, 92 donors, scored for every '
            'method on two gene sets: 100 deeper genes and 100 genes with 30 to 100 haplotype-informative reads. Every hapmixQTL '
            'arm runs on the half-read total; split is the shipped default.</p></header>'
            f'<section id="summary"><h2>Summary</h2><div class="cards">{cards}</div>{limits}</section>'
            + ''.join(f'<section id="{k}" class="record"><p class="eyebrow">Full record</p><h2>{esc(n)}</h2>{record(k)}</section>'
                      for k, n, *_ in SETS)
            + '</main>')
    OUT.mkdir(exist_ok=True)
    C.write_atomic(OUT / 'index.html', lambda fh: fh.write(page.encode()))
    print(f'wrote {OUT / "index.html"} ({len(page.encode()):,} bytes)')


if __name__ == '__main__':
    main()

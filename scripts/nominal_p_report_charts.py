"""Inline-SVG chart helpers for the nominal-p anticonservatism report.

Charts are drawn as SVG strings so the report is one self-contained HTML file
that follows the viewer's light or dark theme: every mark and label takes its
colour from a CSS class whose colour is a theme token, never from a literal.
Each mark carries an SVG <title>, which browsers show as a hover tooltip.

The house style follows calibration_summary.html (scripts/make_calibration_
summary.py): system font stack, warm-grey neutrals, blue for hapmixQTL.
"""
import math
from html import escape

# categorical slots, fixed order (validated palette; see the report CSS)
S1, S2, S3 = 's1', 's2', 's3'


def _fmt(x, nd=3):
    if x == 0:
        return '0'
    if abs(x) < 0.01:
        return f'{x:.4f}'
    return f'{x:.{nd}f}'


def ladder(panels, arms, title, xmax=None):
    """Small multiples, one panel per alpha: rejection rate over alpha per arm.

    panels: list of (alpha_label, alpha) ; arms: list of dict(name, cls,
    rates{alpha: (rate, lo, hi)}). Reference line at 1 (calibrated).
    """
    W, rowh, top, left, right = 300, 22, 26, 118, 14
    H = top + rowh * len(arms) + 30
    if xmax is None:
        xmax = max(max((r[2] if r[2] is not None else r[0]) / a
                       for arm in arms for (lab, a) in panels
                       for r in [arm['rates'][a]]) * 1.08, 2.0)
    pw = W - left - right
    xs = lambda v: left + pw * min(v, xmax) / xmax
    out = ['<div class="smx">']
    for lab, a in panels:
        o = [f'<svg viewBox="0 0 {W} {H}" role="img" '
             f'aria-label="{escape(title)} at alpha {lab}">']
        o.append(f'<text class="ptitle" x="{left}" y="14">&#945; = {lab}</text>')
        step = 0.5 if xmax <= 3 else (1 if xmax <= 7 else 2)
        v = 0.0
        while v <= xmax + 1e-9:
            x = xs(v)
            o.append(f'<line class="grid" x1="{x:.1f}" x2="{x:.1f}" y1="{top-4}" '
                     f'y2="{top + rowh*len(arms)}"/>')
            o.append(f'<text class="tick tc" x="{x:.1f}" y="{top + rowh*len(arms) + 12}">'
                     f'{v:g}&#215;</text>')
            v += step
        x1 = xs(1.0)
        o.append(f'<line class="ref" x1="{x1:.1f}" x2="{x1:.1f}" y1="{top-6}" '
                 f'y2="{top + rowh*len(arms)}"/>')
        for i, arm in enumerate(arms):
            y = top + rowh * i + rowh / 2
            r, lo, hi = arm['rates'][a]
            o.append(f'<text class="rowlab te" x="{left-8}" y="{y+3.5:.1f}">'
                     f'{escape(arm["name"])}</text>')
            if lo is not None and hi is not None:
                o.append(f'<line class="whisk {arm["cls"]}" x1="{xs(lo/a):.1f}" '
                         f'x2="{xs(hi/a):.1f}" y1="{y:.1f}" y2="{y:.1f}"/>')
            tip = (f'{arm["name"]}, alpha {lab}: rate {_fmt(r, 4)}'
                   + (f' [{_fmt(lo, 4)}, {_fmt(hi, 4)}]' if lo is not None else '')
                   + f' = {r/a:.2f}x nominal')
            o.append(f'<circle class="dot {arm["cls"]}" cx="{xs(r/a):.1f}" cy="{y:.1f}" '
                     f'r="4.5"><title>{escape(tip)}</title></circle>')
        o.append(f'<text class="axl" x="{left + pw/2:.1f}" y="{H-3}">'
                 f'rejection rate / nominal</text></svg>')
        out.append('\n'.join(o))
    out.append('</div>')
    return '\n'.join(out)


def gene_bands(rows, title, xlo=0.4, xhi=5.0):
    """Per gene: observed coupling ratio R_g against its null band, log x-axis.

    rows: list of dict(gene, R, lo, hi, cls) sorted as they should appear.
    """
    W, rowh, top, left, right = 640, 13, 18, 86, 20
    H = top + rowh * len(rows) + 30
    pw = W - left - right
    lx = lambda v: left + pw * (math.log(v) - math.log(xlo)) / (math.log(xhi) - math.log(xlo))
    o = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{escape(title)}">']
    for v in (0.5, 0.75, 1, 1.5, 2, 3, 4):
        if xlo <= v <= xhi:
            x = lx(v)
            o.append(f'<line class="{"ref" if v == 1 else "grid"}" x1="{x:.1f}" x2="{x:.1f}" '
                     f'y1="{top-6}" y2="{top + rowh*len(rows)}"/>')
            o.append(f'<text class="tick tc" x="{x:.1f}" y="{top + rowh*len(rows) + 12}">{v:g}</text>')
    for i, r in enumerate(rows):
        y = top + rowh * i + rowh / 2
        o.append(f'<text class="rowlab te small" x="{left-6}" y="{y+3:.1f}">{escape(r["gene"])}</text>')
        o.append(f'<rect class="band" x="{lx(r["lo"]):.1f}" y="{y-3.5:.1f}" '
                 f'width="{max(lx(r["hi"]) - lx(r["lo"]), 1):.1f}" height="7" rx="2"/>')
        tip = (f'{r["gene"]}: R = {r["R"]:.2f}; null 95% band '
               f'[{r["lo"]:.2f}, {r["hi"]:.2f}]')
        o.append(f'<circle class="dot {r["cls"]}" cx="{lx(min(max(r["R"], xlo), xhi)):.1f}" '
                 f'cy="{y:.1f}" r="3.6"><title>{escape(tip)}</title></circle>')
    o.append(f'<text class="axl" x="{left + pw/2:.1f}" y="{H-3}">coupling ratio R '
             f'(log scale; 1 = weights and residuals unrelated)</text></svg>')
    return '\n'.join(o)


def alpha_curves(series, alphas, title, ymax=None):
    """Rejection rate over alpha against alpha (log x), one line per series.

    series: list of dict(name, cls, vals{alpha: rate}).
    """
    W, H, top, left, right, bot = 560, 250, 14, 46, 120, 34
    pw, ph = W - left - right, H - top - bot
    if ymax is None:
        ymax = max(v / a for s in series for a, v in s['vals'].items()) * 1.1
    la = [math.log10(a) for a in alphas]
    xs = lambda a: left + pw * (math.log10(a) - max(la)) / (min(la) - max(la))
    ys = lambda v: top + ph * (1 - v / ymax)
    o = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{escape(title)}">']
    step = 1 if ymax <= 6 else 2
    v = 0
    while v <= ymax + 1e-9:
        y = ys(v)
        o.append(f'<line class="{"ref" if v == 1 else "grid"}" x1="{left}" x2="{left+pw}" '
                 f'y1="{y:.1f}" y2="{y:.1f}"/>')
        o.append(f'<text class="tick te" x="{left-6}" y="{y+3:.1f}">{v:g}&#215;</text>')
        v += step
    for a in alphas:
        x = xs(a)
        o.append(f'<text class="tick tc" x="{x:.1f}" y="{top+ph+14}">{a:g}</text>')
    for s in series:
        # a series may stop short of the smallest alpha (the 46-gene arms end at 0.001)
        own = [a for a in alphas if a in s['vals']]
        pts = [(xs(a), ys(s['vals'][a] / a)) for a in own]
        d = ' '.join(f'{"M" if i == 0 else "L"}{x:.1f},{y:.1f}' for i, (x, y) in enumerate(pts))
        o.append(f'<path class="line {s["cls"]}" d="{d}"/>')
        for a, (x, y) in zip(own, pts):
            tip = f'{s["name"]}, alpha {a:g}: rate {s["vals"][a]:.5f} = {s["vals"][a]/a:.2f}x nominal'
            o.append(f'<circle class="dot {s["cls"]}" cx="{x:.1f}" cy="{y:.1f}" r="4">'
                     f'<title>{escape(tip)}</title></circle>')
        x, y = pts[-1]
        o.append(f'<text class="dlab" x="{x+9:.1f}" y="{y+3.5:.1f}">{escape(s["name"])}</text>')
    o.append(f'<text class="axl" x="{left + pw/2:.1f}" y="{H-3}">nominal level &#945; '
             f'(log scale)</text>')
    o.append(f'<text class="axl" transform="translate(11,{top+ph/2:.1f}) rotate(-90)">'
             f'rate / nominal</text></svg>')
    return '\n'.join(o)


def scatter(points, title, xlab, ylab, lim):
    """Square scatter with the identity line. points: dict(x, y, r, cls, tip, label)."""
    W, H, top, left, right, bot = 380, 360, 14, 46, 16, 36
    pw, ph = W - left - right, H - top - bot
    lo, hi = lim
    xs = lambda v: left + pw * (v - lo) / (hi - lo)
    ys = lambda v: top + ph * (1 - (v - lo) / (hi - lo))
    o = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{escape(title)}">']
    t = math.ceil(lo * 2) / 2
    while t <= hi + 1e-9:
        o.append(f'<line class="grid" x1="{xs(t):.1f}" x2="{xs(t):.1f}" y1="{top}" y2="{top+ph}"/>')
        o.append(f'<line class="grid" x1="{left}" x2="{left+pw}" y1="{ys(t):.1f}" y2="{ys(t):.1f}"/>')
        o.append(f'<text class="tick tc" x="{xs(t):.1f}" y="{top+ph+13}">{t:g}</text>')
        o.append(f'<text class="tick te" x="{left-5}" y="{ys(t)+3:.1f}">{t:g}</text>')
        t += 0.5
    o.append(f'<line class="ref" x1="{xs(lo):.1f}" y1="{ys(lo):.1f}" x2="{xs(hi):.1f}" y2="{ys(hi):.1f}"/>')
    for p in sorted(points, key=lambda p: p['cls'] != S1):
        o.append(f'<circle class="dot {p["cls"]} ring" cx="{xs(p["x"]):.1f}" cy="{ys(p["y"]):.1f}" '
                 f'r="{p["r"]:.1f}"><title>{escape(p["tip"])}</title></circle>')
        if p.get('label'):
            o.append(f'<text class="dlab" x="{xs(p["x"]) + p["r"] + 4:.1f}" '
                     f'y="{ys(p["y"]) + 4:.1f}">{escape(p["label"])}</text>')
    o.append(f'<text class="axl" x="{left + pw/2:.1f}" y="{H-3}">{escape(xlab)}</text>')
    o.append(f'<text class="axl" transform="translate(11,{top+ph/2:.1f}) rotate(-90)">'
             f'{escape(ylab)}</text></svg>')
    return '\n'.join(o)


def stacked(rows, comps, title):
    """Horizontal stacked bars of shares (0-1+) per row. rows: dict(label, parts{comp: share}).

    comps: list of (key, name, cls). Negative parts are drawn left of zero.
    """
    W, rowh, top, left, right = 600, 30, 22, 110, 20
    H = top + rowh * len(rows) + 46
    pw = W - left - right
    lo = min(0.0, min(sum(v for v in r['parts'].values() if v < 0) for r in rows))
    hi = max(1.0, max(sum(v for v in r['parts'].values() if v > 0) for r in rows))
    xs = lambda v: left + pw * (v - lo) / (hi - lo)
    o = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{escape(title)}">']
    for v in [x / 4 for x in range(int(math.floor(lo * 4)), int(math.ceil(hi * 4)) + 1)]:
        o.append(f'<line class="{"ref" if v in (0, 1) else "grid"}" x1="{xs(v):.1f}" x2="{xs(v):.1f}" '
                 f'y1="{top-6}" y2="{top + rowh*len(rows)}"/>')
        o.append(f'<text class="tick tc" x="{xs(v):.1f}" y="{top + rowh*len(rows) + 12}">{v:.0%}</text>')
    for i, r in enumerate(rows):
        y = top + rowh * i + 6
        o.append(f'<text class="rowlab te" x="{left-8}" y="{y+12:.1f}">{escape(r["label"])}</text>')
        pos, neg = 0.0, 0.0
        for key, name, cls in comps:
            v = r['parts'].get(key, 0.0)
            if v >= 0:
                x0, x1 = xs(pos), xs(pos + v); pos += v
            else:
                x0, x1 = xs(neg + v), xs(neg); neg += v
            if abs(x1 - x0) < 0.5:
                continue
            o.append(f'<rect class="seg {cls}" x="{x0:.1f}" y="{y:.1f}" width="{x1-x0:.1f}" '
                     f'height="{rowh-12}" rx="2"><title>{escape(f"{name}: {v:.0%}")}</title></rect>')
    lx = left
    for key, name, cls in comps:
        o.append(f'<rect class="seg {cls}" x="{lx}" y="{H-24}" width="10" height="10" rx="2"/>')
        o.append(f'<text class="leg" x="{lx+14}" y="{H-15}">{escape(name)}</text>')
        lx += 14 + 6.4 * len(name) + 18
    o.append('</svg>')
    return '\n'.join(o)

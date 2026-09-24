"""HTML report: absolute values, observed and null kept apart."""
import json
from pathlib import Path

D = Path('/mnt/ssd/lalli/brainvar_hapmix_deploy')
OUT = D / 'values_report_20260924'
V = json.load(open(OUT / 'values.json'))
COL = {'hapmixQTL': '#2a78d6', 'mixQTL': '#1baf7a', 'RASQUAL': '#eb6834'}


def vrow(label, d, em=False):
    if not d:
        return f'<tr><th>{label}</th><td colspan="3">--</td></tr>'
    c = ' class="em"' if em else ''
    return (f'<tr{c}><th>{label}</th><td>{d["n"]}</td><td>{d["median"]:.4f}</td>'
            f'<td>{d["q25"]:.4f} – {d["q75"]:.4f}</td></tr>')


def table(title, block, note=''):
    rows = ''.join(vrow(k, v) for k, v in block.items())
    return (f'<h3>{title}</h3>{f"<p>{note}</p>" if note else ""}<div class="tw"><table>'
            '<thead><tr><th class="l">quantity</th><th>n</th><th>median</th>'
            '<th>IQR</th></tr></thead><tbody>' + rows + '</tbody></table></div>')


def hist_svg(arm, h):
    W, H, L, R, T, B = 300, 190, 34, 8, 12, 34
    pw, ph = W - L - R, H - T - B
    o = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{arm} null p-value histogram">']
    hi = max(0.16, max(h['frac']))
    for i in range(5):
        y = T + ph - ph * (hi * i / 4) / hi
        o.append(f'<line class="grid" x1="{L}" x2="{W-R}" y1="{y:.1f}" y2="{y:.1f}"/>')
        o.append(f'<text class="tick te" x="{L-4}" y="{y+3:.1f}">{hi*i/4*100:.0f}%</text>')
    yu = T + ph - ph * 0.10 / hi
    o.append(f'<line class="ref" x1="{L}" x2="{W-R}" y1="{yu:.1f}" y2="{yu:.1f}"/>')
    o.append(f'<text class="reflab" x="{W-R}" y="{yu-4:.1f}" text-anchor="end">uniform 10%</text>')
    bw = pw / 10
    for i, f in enumerate(h['frac']):
        bh = ph * f / hi
        x = L + i * bw
        o.append(f'<rect x="{x+1:.1f}" y="{T+ph-bh:.1f}" width="{bw-2:.1f}" '
                 f'height="{bh:.1f}" fill="{COL[arm]}" opacity="0.85"/>')
    o.append(f'<line class="grid" x1="{L}" x2="{W-R}" y1="{T+ph}" y2="{T+ph}"/>')
    for i, lab in ((0, '0'), (5, '0.5'), (10, '1')):
        o.append(f'<text class="tick tc" x="{L+i*bw:.1f}" y="{T+ph+14}">{lab}</text>')
    o.append(f'<text class="axl" x="{L+pw/2:.1f}" y="{H-4}">nominal p at the fixed variant</text>')
    o.append('</svg>')
    return ''.join(o)


cards = ''
for arm in ('hapmixQTL', 'mixQTL', 'RASQUAL'):
    h = V['null_pvalue_histogram'][arm]
    cards += (f'<figure class="card"><figcaption><b style="color:{COL[arm]}">{arm}</b> '
              f'&nbsp; n={h["n"]}</figcaption>{hist_svg(arm, h)}'
              f'<p class="stat">median p <b>{h["median_p"]:.3f}</b> &middot; '
              f'p&lt;0.05 <b>{h["frac_below_05"]:.3f}</b> &middot; '
              f'p&lt;0.01 <b>{h["frac_below_01"]:.3f}</b><br>'
              f'KS vs uniform D={h["ks_stat"]:.3f}, p={h["ks_p"]:.2g}</p></figure>')

se_rows = ''
for arm, sek in (('hapmixQTL', 'hapmixQTL se'), ('mixQTL', 'mixQTL se'),
                 ('RASQUAL', 'RASQUAL se')):
    rep = V['observed_at_rasqual_lead'][sek]['median']
    real = V['null_beta_sd'][arm]['median']
    se_rows += (f'<tr><th>{arm}</th><td>{rep:.4f}</td><td>{real:.4f}</td>'
                f'<td>{rep/real:.2f}</td></tr>')

html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Observed and Null Values</title><style>
:root{{--bg:#fcfcfb;--ink:#1a1a19;--ink2:#54534f;--ink3:#8a8884;--line:#e4e3df}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#1a1a19;--ink:#f2f1ee;
--ink2:#b5b3ad;--ink3:#84827c;--line:#33322f}}}}
body{{background:var(--bg);color:var(--ink);margin:0;padding:0 16px 56px;
font:14.5px/1.62 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}}
.wrap{{max-width:1000px;margin:0 auto}}
h1{{font-size:1.5rem;margin:28px 0 6px}} h2{{font-size:1.06rem;margin:34px 0 6px}}
h3{{font-size:.95rem;margin:22px 0 4px}}
p{{max-width:82ch;color:var(--ink2)}} p.lede{{margin:0 0 6px}}
table{{border-collapse:collapse;font-size:12.5px;width:100%;font-variant-numeric:tabular-nums}}
th,td{{padding:5px 8px;border-bottom:1px solid var(--line);text-align:right}}
thead th{{color:var(--ink2);font-weight:600;font-size:11.5px}}
tbody th{{text-align:left;font-weight:500}} th.l{{text-align:left}}
.tw{{overflow-x:auto;margin:8px 0 6px}}
.cards{{display:flex;gap:14px;flex-wrap:wrap;margin:10px 0 6px}}
.card{{flex:1 1 280px;margin:0}}
.card figcaption{{font-size:.85rem;color:var(--ink2);margin:0 0 2px}}
.stat{{font-size:.8rem;color:var(--ink2);margin:2px 0 0}}
svg{{width:100%;height:auto;overflow:visible}}
.grid{{stroke:var(--line);stroke-width:1}}
.ref{{stroke:var(--ink3);stroke-width:1.2;stroke-dasharray:4 3}}
.reflab{{fill:var(--ink3);font-size:9px}}
.tick{{fill:var(--ink3);font-size:9.5px}} .te{{text-anchor:end}} .tc{{text-anchor:middle}}
.axl{{fill:var(--ink3);font-size:9.5px;text-anchor:middle}}
.key{{border-left:3px solid #2a78d6;padding:2px 0 2px 12px;margin:14px 0;max-width:82ch}}
.note{{color:var(--ink3);font-size:.8rem;max-width:84ch}}
</style></head><body><div class="wrap">
<h1>Observed and null values, three arms</h1>
<p class="lede">All quantities are natural-log allelic fold change per haplotype-dosage
unit, on 59 genes (null on the 46 carrying RASQUAL null rows). Absolute values, not
ratios. The observed run and the null run are different experiments and are kept
apart.</p>

<h2>Observed run</h2>
<p>One estimate per gene at a named variant. An arm that <i>selected</i> a variant has
its effect there inflated by the winner's curse, so each table says who chose.</p>
{table("At RASQUAL's lead variant", V['observed_at_rasqual_lead'],
       "RASQUAL selected this variant; hapmixQTL and mixQTL did not, so those two are "
       "directly comparable here and RASQUAL's own effect is inflated.")}
{table("At hapmixQTL's lead variant", V['observed_at_hapmixqtl_lead'],
       "hapmixQTL selected this variant, so its effect is the inflated one here.")}

<h2>Null run</h2>
<p>Twenty permutations per gene at the <b>same fixed variant</b>, where the true slope
is zero. Reported as the distribution of the nominal p-value rather than a count of
hits: a count collapses the distribution to one threshold, and thresholds estimated
from a finite null move around — which is what made the earlier detection tables
shift when the permutations were doubled.</p>

<h3>Spread of the estimate across permutations</h3>
<div class="tw"><table><thead><tr><th class="l">arm</th><th>n genes</th>
<th>median null sd of beta</th><th>IQR</th></tr></thead><tbody>
{''.join(vrow(a, V['null_beta_sd'][a]) for a in ('hapmixQTL','mixQTL','RASQUAL'))}
</tbody></table></div>

<h3>Reported standard error against realized null spread</h3>
<div class="key"><p>These two should agree if an arm's standard error means what it
says. The standard error is each arm's own claim about its uncertainty; the null
spread is what its estimate actually does when the true effect is zero.</p></div>
<div class="tw"><table><thead><tr><th class="l">arm</th>
<th>median reported se</th><th>median realized null sd</th><th>reported / realized</th>
</tr></thead><tbody>{se_rows}</tbody></table></div>

<h3>Nominal p-value at the fixed variant</h3>
<p>Under the null a calibrated nominal scale is uniform: about 10% of values in each
decile, shown by the dashed line.</p>
<div class="cards">{cards}</div>

<p class="note">Bounds. The fixed variant is each gene's RASQUAL observed lead, chosen
before any permutation and identical across arms, but it is a place RASQUAL found
signal rather than a random variant. The genes are high-coverage by selection. Twenty
draws give each per-gene sd roughly 16% relative uncertainty. mixQTL ran at
package-default cutoffs; its published cutoffs leave it total-counts-only on many of
these genes. RASQUAL's null draws come from its own internal permutation and are not
paired with the other two arms', which is why only spreads and distributions are
compared and never draw-by-draw values. Written 2026-09-24.</p>
</div></body></html>"""
(OUT / 'values_report.html').write_text(html)
print('wrote', OUT / 'values_report.html')

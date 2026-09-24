"""One HTML page collecting every calibration measurement from 2026-09-24."""
import json
from pathlib import Path

D = Path('/mnt/ssd/lalli/brainvar_hapmix_deploy')
OUT = D / 'calibration_summary_20260924'
OUT.mkdir(exist_ok=True)

V = json.load(open(D / 'values_report_20260924/values.json'))
CH = json.load(open(D / 'channel_split_20260924/summary.json'))
RD = json.load(open(D / 'residual_diagnostics_20260924/summary.json'))
NS = json.load(open(D / 'allelic_null_schemes_20260924/summary.json'))
PP = json.load(open(D / 'perm_p_by_scheme_20260924/summary.json'))

COL = {'hapmixQTL': '#2a78d6', 'mixQTL': '#1baf7a', 'RASQUAL': '#eb6834',
       'records': '#8a8884', 'sign_flip': '#d4348a'}


def hist_svg(name, frac, colour, hi=0.20):
    W, H, L, R, T, B = 300, 180, 36, 8, 10, 32
    pw, ph = W - L - R, H - T - B
    o = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{name} null p histogram">']
    for i in range(5):
        y = T + ph - ph * (hi * i / 4) / hi
        o.append(f'<line class="grid" x1="{L}" x2="{W-R}" y1="{y:.1f}" y2="{y:.1f}"/>')
        o.append(f'<text class="tick te" x="{L-4}" y="{y+3:.1f}">{hi*i/4*100:.0f}%</text>')
    yu = T + ph - ph * 0.10 / hi
    o.append(f'<line class="ref" x1="{L}" x2="{W-R}" y1="{yu:.1f}" y2="{yu:.1f}"/>')
    bw = pw / 10
    for i, f in enumerate(frac):
        bh = min(ph, ph * f / hi)
        o.append(f'<rect x="{L+i*bw+1:.1f}" y="{T+ph-bh:.1f}" width="{bw-2:.1f}" '
                 f'height="{bh:.1f}" fill="{colour}" opacity="0.85"/>')
    o.append(f'<line class="grid" x1="{L}" x2="{W-R}" y1="{T+ph}" y2="{T+ph}"/>')
    for i, lab in ((0, '0'), (5, '0.5'), (10, '1')):
        o.append(f'<text class="tick tc" x="{L+i*bw:.1f}" y="{T+ph+13}">{lab}</text>')
    o.append(f'<text class="axl" x="{L+pw/2:.1f}" y="{H-3}">nominal p</text></svg>')
    return ''.join(o)


def card(title, sub, frac, colour, stat):
    return (f'<figure class="card"><figcaption><b style="color:{colour}">{title}</b>'
            f'<br><span class="sub">{sub}</span></figcaption>'
            f'{hist_svg(title, frac, colour)}<p class="stat">{stat}</p></figure>')


# --- three methods, records null -------------------------------------------
H = V['null_pvalue_histogram']
cards_methods = ''
for arm in ('RASQUAL', 'hapmixQTL', 'mixQTL'):
    h = H[arm]
    verdict = ('uniform' if h['ks_p'] > 0.2 else
               'borderline' if h['ks_p'] > 0.01 else 'NOT uniform')
    cards_methods += card(
        arm, f'n={h["n"]} &middot; {verdict}', h['frac'], COL[arm],
        f'median p <b>{h["median_p"]:.3f}</b> &middot; p&lt;0.05 '
        f'<b>{h["frac_below_05"]:.3f}</b> &middot; p&lt;0.01 '
        f'<b>{h["frac_below_01"]:.3f}</b><br>KS D={h["ks_stat"]:.3f}, p={h["ks_p"]:.2g}')

# --- hapmixQTL channels ------------------------------------------------------
cards_ch = ''
for cfg, lab in (('allelic_only', 'allelic channel only'),
                 ('total_only', 'total channel only'),
                 ('both', 'both channels (shipped)')):
    c = CH[cfg]
    cards_ch += card(lab, f'n={c["n"]}', c['frac'], COL['hapmixQTL'],
                     f'median p <b>{c["median_p"]:.3f}</b> &middot; p&lt;0.05 '
                     f'<b>{c["frac_below_05"]:.3f}</b><br>KS D={c["ks_stat"]:.3f}, '
                     f'p={c["ks_p"]:.2g}')

# --- two nulls ---------------------------------------------------------------
cards_null = ''
for sch, lab in (('records', 'records null (shipped)'),
                 ('sign_flip', 'sign-flip null (channel symmetry)')):
    c = NS[sch]
    cards_null += card(lab, f'n={c["n"]}', c['frac'], COL[sch],
                       f'median p <b>{c["median_p"]:.3f}</b> &middot; p&lt;0.05 '
                       f'<b>{c["frac_below_05"]:.3f}</b><br>KS D={c["ks_stat"]:.3f}, '
                       f'p={c["ks_p"]:.2g}')

se_rows = ''
for arm in ('hapmixQTL', 'mixQTL', 'RASQUAL'):
    rep = V['observed_at_rasqual_lead'][f'{arm} se']['median']
    real = V['null_beta_sd'][arm]['median']
    cls = ' class="em"' if abs(rep / real - 1) > 0.1 else ''
    se_rows += (f'<tr{cls}><th>{arm}</th><td>{rep:.4f}</td><td>{real:.4f}</td>'
                f'<td>{rep/real:.2f}</td></tr>')

html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Calibration Evidence</title><style>
:root{{--bg:#fcfcfb;--ink:#1a1a19;--ink2:#54534f;--ink3:#8a8884;--line:#e4e3df}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#1a1a19;
--ink:#f2f1ee;--ink2:#b5b3ad;--ink3:#84827c;--line:#33322f}}}}
body{{background:var(--bg);color:var(--ink);margin:0;padding:0 16px 56px;
font:14.5px/1.62 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}}
.wrap{{max-width:1000px;margin:0 auto}}
h1{{font-size:1.5rem;margin:26px 0 6px}} h2{{font-size:1.08rem;margin:32px 0 4px}}
p{{max-width:82ch;color:var(--ink2)}} p.lede{{margin:0 0 4px}}
table{{border-collapse:collapse;font-size:12.5px;width:100%;font-variant-numeric:tabular-nums}}
th,td{{padding:5px 8px;border-bottom:1px solid var(--line);text-align:right}}
thead th{{color:var(--ink2);font-weight:600;font-size:11.5px}}
tbody th{{text-align:left;font-weight:500}} th.l{{text-align:left}}
tr.em th,tr.em td{{font-weight:700}}
.tw{{overflow-x:auto;margin:8px 0}}
.cards{{display:flex;gap:14px;flex-wrap:wrap;margin:10px 0}}
.card{{flex:1 1 280px;margin:0}}
.card figcaption{{font-size:.86rem;color:var(--ink);margin:0 0 2px}}
.sub{{color:var(--ink2);font-size:.78rem}}
.stat{{font-size:.78rem;color:var(--ink2);margin:2px 0 0}}
svg{{width:100%;height:auto;overflow:visible}}
.grid{{stroke:var(--line);stroke-width:1}}
.ref{{stroke:var(--ink3);stroke-width:1.2;stroke-dasharray:4 3}}
.tick{{fill:var(--ink3);font-size:9.5px}} .te{{text-anchor:end}} .tc{{text-anchor:middle}}
.axl{{fill:var(--ink3);font-size:9.5px;text-anchor:middle}}
.key{{border-left:3px solid #2a78d6;padding:2px 0 2px 12px;margin:14px 0;max-width:82ch}}
.warn{{border-left:3px solid #eb6834;padding:2px 0 2px 12px;margin:14px 0;max-width:82ch}}
.key p,.warn p{{color:var(--ink)}}
.note{{color:var(--ink3);font-size:.8rem;max-width:84ch}}
</style></head><body><div class="wrap">

<h1>Calibration of hapmixQTL, mixQTL and RASQUAL</h1>
<p class="lede">All measurements 2026-09-24, on 46 high-coverage BrainVar genes at
one fixed variant each (that gene's RASQUAL observed lead), 30 null draws. Under
the null the true slope is zero, so a calibrated nominal p must be uniform: 10%
of values in each decile, marked by the dashed line.</p>

<h2>1. The three methods, under donor-permutation (the shipped null)</h2>
<div class="cards">{cards_methods}</div>
<p><b>RASQUAL is uniform. hapmixQTL is not. mixQTL sits between them.</b>
hapmixQTL's excess sits in the first decile while its median p is 0.474,
essentially 0.5 — the signature of a tail problem rather than a whole-distribution
shift. These figures use each arm's own nominal p; converting hapmixQTL's
statistic with chi2(1) instead of its t reference would have overstated the
excess (14.6% rather than 14.1% in the first decile).</p>

<h2>2. Where hapmixQTL's miscalibration lives</h2>
<div class="cards">{cards_ch}</div>
<p>Neither channel is innocent. The allelic and total channels are
<b>indistinguishable</b> from each other (p&lt;0.05 of 0.068 against 0.067,
paired McNemar p = 0.94), and combining them adds a little more (0.082, paired
McNemar p = 0.099 and 0.069 against the two channels, neither significant). This
rules out the allelic log-ratio's normal approximation as the sole cause — the
total channel is a log total expression, not a ratio, and is equally affected.</p>

<h2>3. Why: the model's own residuals</h2>
<p>For a Gaussian linear model the Wald statistic, the F statistic and the
likelihood ratio are monotone transforms of each other, so referencing T² to
F(1, dof) is the <i>exact</i> test when the model holds. A non-uniform null
therefore indicts the model, which makes two checkable claims.</p>
<div class="tw"><table><thead><tr><th class="l">model claim</th><th>allelic</th>
<th>total</th></tr></thead><tbody>
<tr><th>SHAPE: standardized squared residual vs log v<br>
<span class="sub">slope 0 if Var(eps) = sigma² v holds</span></th>
<td>{RD['shape_allelic']['pooled_slope']:+.4f} &plusmn; {RD['shape_allelic']['pooled_se']:.4f}<br>
<span class="sub">negative in {RD['shape_allelic']['n_genes']-RD['shape_allelic']['n_genes_positive']}/{RD['shape_allelic']['n_genes']} genes</span></td>
<td>{RD['shape_total']['pooled_slope']:+.4f} &plusmn; {RD['shape_total']['pooled_se']:.4f}<br>
<span class="sub">negative in {RD['shape_total']['n_genes']-RD['shape_total']['n_genes_positive']}/{RD['shape_total']['n_genes']} genes</span></td></tr>
<tr class="em"><th>TAILS: excess kurtosis<br>
<span class="sub">0 if Gaussian; positive makes F liberal</span></th>
<td>{RD['kurtosis_allelic']['pooled']:+.3f} pooled<br>
<span class="sub">positive in {RD['kurtosis_allelic']['n_positive']}/{RD['kurtosis_allelic']['n_genes']} genes</span></td>
<td>{RD['kurtosis_total']['pooled']:+.3f} pooled<br>
<span class="sub">positive in {RD['kurtosis_total']['n_positive']}/{RD['kurtosis_total']['n_genes']} genes</span></td></tr>
</tbody></table></div>
<p>The shape error is real but mild. <b>The tails are the larger effect</b>, and
they inflate the small-p end while leaving the bulk alone — exactly the pattern
observed. This also explains why mixQTL, which never touches a Gibbs draw, is
miscalibrated too: both methods map counts onto a continuous log scale and attach
a variance, and the residuals of that response are leptokurtic. RASQUAL instead
models integer reads under a beta-binomial and negative binomial, with the
discreteness and overdispersion inside the likelihood.</p>

<h2>4. The null itself was part of the answer</h2>
<div class="cards">{cards_null}</div>
<p>The allelic regression is through the origin on s = xL − xR, and a donor
informs the slope only if it has allelic information (v &gt; 0, which
<i>travels</i> under donor permutation) and is heterozygous at the tested variant
(s ≠ 0, which <i>stays</i>). The sign-flip null — randomly swapping each donor's
haplotype labels — is the channel's actual symmetry and holds both fixed.
Under it hapmixQTL's allelic channel is far worse: p&lt;0.05 of
{NS['sign_flip']['frac_below_05']:.3f} against {NS['records']['frac_below_05']:.3f},
paired McNemar p = {NS['paired_0.05']['mcnemar_p']:.2g}.</p>

<div class="warn"><p><b>And the empirical permutation p inherits it.</b> Same
observed data, same genes, only the null construction differing: median empirical
p {PP['median_records']:.3f} under records against {PP['median_sign_flip']:.3f}
under sign flip, larger on {PP['n_signflip_larger']} genes and smaller on
{PP['n_signflip_smaller']} (sign p = {PP['sign_p']:.2g}), calling
<b>{PP['n_called_05_records']} genes rather than {PP['n_called_05_sign_flip']}</b>
at the 0.05 threshold. "The empirical p is calibrated by construction" holds only
for the null it is built from.</p></div>

<h2>5. Claimed uncertainty against realized uncertainty</h2>
<div class="tw"><table><thead><tr><th class="l">arm</th>
<th>median reported se</th><th>median realized null sd</th>
<th>reported / realized</th></tr></thead><tbody>{se_rows}</tbody></table></div>
<p>An independent route to the same conclusion, using no p-values at all. Both
Salmon-based arms understate their own uncertainty by about 15%; RASQUAL
overstates its by about 17%.</p>

<h2>What this does and does not establish</h2>
<div class="key"><p><b>The comparison between methods is not symmetric, and that
limits the headline.</b> RASQUAL's uniformity was measured under its own internal
permutation (its <code>-r</code> flag), while hapmixQTL was additionally subjected
to the harsher sign-flip null. RASQUAL was never evaluated under an equivalent
label-symmetry null. "RASQUAL is well calibrated and hapmixQTL is not" is
supported under donor permutation; it is not established that RASQUAL would
survive the stricter test hapmixQTL failed.</p></div>
<p class="note">Further bounds. 46 high-coverage genes, one fixed variant each,
chosen as RASQUAL's observed lead — a place RASQUAL found signal, not a random
variant. 30 draws give each per-gene empirical p a grain of 1/31 = 0.032, so the
gene counts in section 4 are coarse and the distributional tests are the sounder
half. The section 4 result is a single fixed variant, not the shipped gene-level
<code>pval_perm</code>, which is maximised over the cis window; the concern
transfers in kind, the magnitude does not transfer unmeasured. Section 1's
hapmixQTL figures use its own t reference; sections 2 and 4 likewise. Everything
here is measurement: no variance model, permutation scheme or default was
changed. Sources: values_report_20260924, channel_split_20260924,
residual_diagnostics_20260924, allelic_null_schemes_20260924,
perm_p_by_scheme_20260924.</p>
</div></body></html>"""
(OUT / 'calibration_summary.html').write_text(html)
print('wrote', OUT / 'calibration_summary.html')

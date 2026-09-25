"""Build the HTML report for the 2026-09-25 nominal-p investigation.

Reads the result files under brainvar_hapmix_deploy/*_20260925/ and writes
one self-contained page with inline-SVG figures to
nominal_p_hypotheses_20260925/nominal_p_report.html. Every number in the
prose that is not read from a file is a verified figure from the adversarial
verification record (followups_verified.json, investigations_verified.json,
reconciliation.md) and is marked with the arm it came from.

Structure follows the project's reporting rule: why the question was open,
what was run, the result, the critique and what it changed, what it means.
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nominal_p_report_charts import (S1, S2, S3, alpha_curves, gene_bands,  # noqa: E402
                                     ladder, stacked)

D = Path('/mnt/ssd/lalli/brainvar_hapmix_deploy')
H = D / 'nominal_p_hypotheses_20260925'
OUT = H / 'nominal_p_report.html'
ALPHAS = (0.05, 0.01, 0.001)


def load():
    arms = pd.read_csv(D / 'weight_residual_coupling_20260925' / 'arm_rates_pooled.tsv', sep='\t')
    bands = pd.read_csv(H / 'verification_checks' / 'verify_coupling_identification' /
                        'shuffle_null_R.tsv', sep='\t')
    three = pd.read_csv(D / 'comparator_null_2000_20260925' / 'three_way_table.tsv', sep='\t')
    budget = pd.read_csv(D / 'combined_statistic_budget_20260925' / 'budget.tsv', sep='\t')
    reach = json.load(open(D / 'coupling_reach_20260925' / 'summary.json'))
    strata = pd.read_csv(D / 'coupling_reach_20260925' / 'b_strata.tsv', sep='\t')
    dom = pd.read_csv(D / 'dominant_record_anatomy_20260925' / 'dominant_records.tsv', sep='\t')
    inst = json.load(open(D / 'nominal_p_null_instrument_20260925' / 'summary.json'))
    return dict(arms=arms, bands=bands, three=three, budget=budget, reach=reach,
                strata=strata, dom=dom, inst=inst)


def arm_rates(arms, name, genes='all46'):
    sub = arms[(arms.arm == name) & (arms.genes == genes)].set_index('alpha')
    return {a: (float(sub.loc[a, 'rate']), float(sub.loc[a, 'lo']), float(sub.loc[a, 'hi']))
            for a in ALPHAS}


def three_rates(three, method, config):
    r = three[(three.method == method) & (three.config == config)].iloc[0]
    out = {}
    for a in ALPHAS:
        lo, hi = json.loads(r[f'ci{a:g}'])
        out[a] = (float(r[f'r{a:g}']), float(lo), float(hi))
    return out


def pct(x):
    return f'{100 * x:.0f}%'


def f4(x):
    return f'{x:.4f}'


CSS = """
:root{--bg:#fcfcfb;--ink:#1a1a19;--ink2:#54534f;--ink3:#8a8884;--line:#e4e3df;--band:#ecebe6;
--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;--key:#2a78d6;--warn:#eb6834}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--bg:#1a1a19;
--ink:#f2f1ee;--ink2:#b5b3ad;--ink3:#84827c;--line:#33322f;--band:#2a2a27;
--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--key:#3987e5;--warn:#d95926}}
:root[data-theme="dark"]{color-scheme:dark;--bg:#1a1a19;--ink:#f2f1ee;--ink2:#b5b3ad;--ink3:#84827c;
--line:#33322f;--band:#2a2a27;--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--key:#3987e5;--warn:#d95926}
body{background:var(--bg);color:var(--ink);margin:0;padding-block:0 56px;padding-inline:16px;
font:14.5px/1.62 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:980px;margin:0 auto}
h1{font-size:1.55rem;margin:26px 0 6px;text-wrap:balance} h2{font-size:1.12rem;margin:34px 0 6px;text-wrap:balance}
h3{font-size:.98rem;margin:22px 0 4px}
p{max-width:78ch;color:var(--ink2)} p.lede{margin:0 0 4px;color:var(--ink)}
table{border-collapse:collapse;font-size:12.5px;width:100%;font-variant-numeric:tabular-nums}
th,td{padding:5px 8px;border-bottom:1px solid var(--line);text-align:right;vertical-align:top}
thead th{color:var(--ink2);font-weight:600;font-size:11.5px}
tbody th{text-align:left;font-weight:500} th.l,td.l{text-align:left}
.tw{overflow-x:auto;margin:8px 0}
figure{margin:14px 0} figcaption{font-size:.86rem;color:var(--ink);margin:0 0 4px;max-width:78ch}
.sub{color:var(--ink2);font-size:.8rem;max-width:78ch}
svg{width:100%;height:auto;overflow:visible}
.smx{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:10px}
.grid{stroke:var(--line);stroke-width:1} .ref{stroke:var(--ink3);stroke-width:1.2;stroke-dasharray:4 3}
.tick{fill:var(--ink3);font-size:9.5px} .te{text-anchor:end} .tc{text-anchor:middle}
.axl{fill:var(--ink3);font-size:9.5px;text-anchor:middle}
.rowlab{fill:var(--ink);font-size:10.5px} .rowlab.small{font-size:8.5px}
.ptitle{fill:var(--ink);font-size:11px;font-weight:600} .dlab{fill:var(--ink2);font-size:10px}
.leg{fill:var(--ink2);font-size:10px}
.band{fill:var(--band)} .whisk{stroke-width:2;stroke:var(--ink3)} .whisk.s1{stroke:var(--s1)}
.whisk.s2{stroke:var(--s2)} .whisk.s3{stroke:var(--s3)} .whisk.s4{stroke:var(--s4)}
.dot{fill:var(--ink3)} .dot.s1{fill:var(--s1)} .dot.s2{fill:var(--s2)} .dot.s3{fill:var(--s3)} .dot.s4{fill:var(--s4)}
.dot.ring{stroke:var(--bg);stroke-width:2}
.line{fill:none;stroke-width:2;stroke:var(--ink3)} .line.s1{stroke:var(--s1)} .line.s2{stroke:var(--s2)}
.line.s3{stroke:var(--s3)} .line.s4{stroke:var(--s4)}
.seg{fill:var(--ink3);stroke:var(--bg);stroke-width:2} .seg.s1{fill:var(--s1)} .seg.s2{fill:var(--s2)}
.seg.s3{fill:var(--s3)} .seg.s4{fill:var(--s4)}
.key{border-left:3px solid var(--key);padding:2px 0 2px 12px;margin:14px 0;max-width:78ch}
.warn{border-left:3px solid var(--warn);padding:2px 0 2px 12px;margin:14px 0;max-width:78ch}
.key p,.warn p{color:var(--ink)}
.note{color:var(--ink3);font-size:.8rem;max-width:80ch}
dl{max-width:78ch;color:var(--ink2);font-size:.9rem} dt{font-weight:600;color:var(--ink);margin-top:6px}
code{font-size:.9em}
"""


def build(X):
    arms, bands, three, budget = X['arms'], X['bands'], X['three'], X['budget']
    reach, strata, dom, inst = X['reach'], X['strata'], X['dom'], X['inst']
    o = []
    w = o.append
    w('<title>Weight-Residual Coupling</title>')
    w(f'<style>{CSS}</style>')
    w('<div class="wrap">')
    w('<h1>Why hapmixQTL&#8217;s nominal p runs above nominal on a permuted null</h1>')
    w('<p class="lede">2026-09-25. Forty-six BrainVar genes, one fixed variant each (that '
      'gene&#8217;s RASQUAL observed lead), 92 donors, 2,000 donor-record permutations per gene. '
      'All rates are rejection rates at nominal 0.05 / 0.01 / 0.001; a calibrated p gives exactly '
      'those. Intervals in brackets are gene-clustered bootstrap 95% intervals: the 46 genes are '
      'resampled with replacement and the pooled rate recomputed. Natural-log units throughout.</p>')

    # ---- 1. why -----------------------------------------------------------
    w('<h2>1. Why the question was open</h2>')
    w('<p>The 2026-09-24 session measured hapmixQTL&#8217;s nominal p at 0.082 against nominal 0.05 '
      'on 30 permutations of the same 46 genes, tested five candidate mechanisms and eliminated '
      'all five, and left the cause unidentified with cross-donor dependence as the leading '
      'untested candidate. Thirty permutations pool well enough (1,380 p-values) but say nothing '
      'per gene: a per-gene rejection rate from 30 draws carries a binomial standard error of '
      '0.04 at 0.05, so no per-gene mechanism could be tested. It was also not established '
      'whether the excess was one property shared by every gene or a few genes misbehaving.</p>')
    w('<p>One structural fact fixed the search before it started. Under a donor-record '
      'permutation, the set of records (each donor&#8217;s allelic log ratio, total log '
      'expression, both Gibbs variances and covariate row) is held fixed and only its '
      'assignment to genotype positions is random. Correlation of errors across donors is '
      'destroyed by that permutation and cannot produce an excess here. Every admissible '
      'hypothesis therefore had to name a property of the fixed record set, the genotype '
      'vector, the statistic or its reference distribution.</p>')

    # ---- 2. what was run --------------------------------------------------
    w('<h2>2. What was run</h2>')
    w('<p>A 2,000-permutation instrument (<code>scripts/null_permutation_instrument.py</code>) '
      'reproduces the 2026-09-24 null exactly: the same RandomState(42) stream, the same fixed '
      'variant, the same fit. Two gates ran before anything was written. Its combined |t| '
      f'matches <code>map_cis</code> on the observed data to a relative {inst["gate1_max_rel"]:.1e}, '
      f'and its first 30 permutations match the recorded per-(gene, permutation) p to {inst["gate2_max_dp"]:.1e}. '
      'Each channel is reported apart: allelic (weighted least squares through the origin, '
      'weights 1/v, residual scale fitted per variant), total (intercept and 17 covariates '
      'partialled out in the weighted space) and the shipped combined statistic (inverse-variance '
      'meta-analysis referred to F(1, min dof)).</p>')
    w('<p>Eight independent hypothesis generators, each with a different lens (randomization '
      'theory, code audit, record anatomy, Gibbs-variance mechanics, the statistic and its '
      'reference, comparator methods, a sceptic of the five eliminations, and the external '
      'simulation benchmark), produced 43 hypotheses with pre-registered predictions and quick '
      'checks. Eight investigators then wrote gated scripts for the converged leads; eight '
      'adversarial verifiers re-ran each script to byte-identical output, hunted harness '
      'artefacts, and each wrote one independent computation that could have refuted the '
      'headline. A completeness critic listed what was missing, four follow-ups closed those '
      'gaps and were verified the same way, and a reconciliation critic produced the final '
      'budget. A fifth follow-up, conditioning on additional cis variants, was cancelled by '
      'decision and is not used anywhere. Every &#8220;real&#8221; arm below reproduces the '
      'instrument per (gene, permutation) to 1e-9 before anything is varied.</p>')

    # ---- 3. results -------------------------------------------------------
    w('<h2>3. The result</h2>')
    w('<h3>3.1 The 0.082 was a high 30-draw sample</h3>')
    hb = three_rates(three, 'hapmixQTL', 'combined, F(1, min dof)')
    ha = three_rates(three, 'hapmixQTL', 'allelic channel')
    ht = three_rates(three, 'hapmixQTL', 'total channel')
    mn = three_rates(three, 'mixQTL port', 'permissive cutoffs, meta, normal ref (as published)')
    mf = three_rates(three, 'mixQTL port', 'permissive cutoffs, meta, F/t ref')
    rq = three_rates(three, 'RASQUAL', 'converged rows, chi2(1)')
    w(f'<p>At 2,000 permutations the shipped combined statistic rejects at {hb[0.05][0]:.3f} '
      f'[{hb[0.05][1]:.3f}, {hb[0.05][2]:.3f}] / {hb[0.01][0]:.4f} / {hb[0.001][0]:.4f}. The same first '
      f'30 permutations give 0.080, so the recorded 0.082 was a high draw of the same stream. The '
      f'allelic channel is {ha[0.05][0]:.3f} / {ha[0.01][0]:.3f} / {ha[0.001][0]:.4f} and the total channel '
      f'{ht[0.05][0]:.3f} / {ht[0.01][0]:.4f} / {ht[0.001][0]:.4f}, so combining the channels adds no '
      f'excess of its own at 0.05. The mixQTL port&#8217;s recorded 0.067 was a 30-draw high as well: '
      f'{mn[0.05][0]:.3f} [{mn[0.05][1]:.3f}, {mn[0.05][2]:.3f}] under its own normal reference, every '
      f'interval excluding nominal, and {mf[0.05][0]:.3f} under an F reference it does not use. '
      f'RASQUAL&#8217;s {rq[0.05][0]:.3f} stands but is on its own <code>-r</code> null, which permutes '
      'each feature SNP separately with no haplotype swap, at 30 draws; its 96 of 1,380 '
      'non-converged null rows reject at 0.083 and are excluded here.</p>')
    comp = [
        dict(name='hapmixQTL combined', cls=S1, rates=hb),
        dict(name='hapmixQTL allelic', cls=S1, rates=ha),
        dict(name='hapmixQTL total', cls=S1, rates=ht),
        dict(name='mixQTL, normal ref', cls=S2, rates=mn),
        dict(name='mixQTL, F ref', cls=S2, rates=mf),
        dict(name='RASQUAL, -r null', cls=S3, rates=rq),
    ]
    w('<figure><figcaption>Rejection rate over nominal, three methods on the records null '
      '(RASQUAL on its own null). Whiskers are gene-clustered 95% intervals; the dashed line is '
      'calibration.</figcaption>')
    w(ladder([('0.05', 0.05), ('0.01', 0.01), ('0.001', 0.001)], comp, 'three-way calibration'))
    w('</figure>')

    w('<h3>3.2 The mechanism: Gibbs weight paired with residual size within a gene</h3>')
    R = arm_rates(arms, 'REAL'); M = arm_rates(arms, 'MODEL'); Dc = arm_rates(arms, 'DECOUPLE')
    Io = arm_rates(arms, 'ISOLATE_OWN'); Is = arm_rates(arms, 'ISOLATE_SMOOTH')
    P9 = arm_rates(arms, 'POWER_g0.936_elim5'); P8 = arm_rates(arms, 'POWER_g0.836_within')
    w('<p>Write w = 1/v for a record&#8217;s Gibbs weight and z&#178; = a&#178;/v for its whitened '
      'squared residual under the null (the allelic channel is through the origin with no '
      'nuisance columns, so its null residual is the log ratio itself). The model '
      'Var(&#949;) = &#963;&#178;v says z&#178; is unrelated to w within a gene. The coupling '
      'ratio R<sub>g</sub> = mean(w&#183;z&#178;) / (mean w &#183; mean z&#178;) is 1 under that model '
      'and, to first order, is exactly the realized variance of the permuted slope over the '
      'variance the model reports. It is computable from a gene&#8217;s fixed records without '
      'permuting. Its rank correlation with the realized ratio (Spearman &#961; = 0.955, the '
      'Pearson correlation of ranks) is therefore algebra, not evidence.</p>')
    w(f'<p>The evidence is the arm ladder. Model records (residuals drawn N(0, 1) at each gene&#8217;s '
      f'real weights, then permuted) give {M[0.05][0]:.4f} / {M[0.01][0]:.4f} / {M[0.001][0]:.5f}: the '
      'estimator and its reference are correct when the model holds. Shuffling each gene&#8217;s own '
      'residuals against its own weights, which keeps both distributions and destroys only the '
      f'pairing, gives {Dc[0.05][0]:.4f} / {Dc[0.01][0]:.4f} / {Dc[0.001][0]:.5f}. Real minus decoupled, '
      f'{R[0.05][0] - Dc[0.05][0]:+.4f} [0.0073, 0.0286] at 0.05, is 91% [73, 100] of the excess over '
      'the model, 85% at 0.01 and 88% at 0.001; what the decoupled arm keeps over the model, '
      f'{Dc[0.05][0] - M[0.05][0]:.4f} at 0.05, is the heavy-tailed residual distribution alone. '
      'Gaussian errors carrying only each record&#8217;s own realized variance, with no heavy tails, '
      f'reproduce the real rates ({Io[0.05][0]:.4f} / {Io[0.01][0]:.4f} / {Io[0.001][0]:.4f}).</p>')
    lad = [
        dict(name='real records', cls=S1, rates=R),
        dict(name='model (N(0,1) at real w)', cls='', rates=M),
        dict(name='decoupled (z shuffled vs w)', cls='', rates=Dc),
        dict(name='Gaussian, own variance', cls=S2, rates=Io),
        dict(name='Gaussian, smooth E[z²|w]', cls=S2, rates=Is),
        dict(name='Var ∝ v^0.936 (elimination 5)', cls=S3, rates=P9),
        dict(name='Var ∝ v^0.836 (within-gene)', cls=S3, rates=P8),
    ]
    w('<figure><figcaption>Allelic channel, identical permutation stream. Blue: the real records. '
      'Grey: controls that should sit at 1&#215; if the mechanism is the pairing. Orange: '
      'isolating arms that carry only the measured coupling. Green: the smooth power-law arms '
      'of the earlier elimination. Whiskers on the real arm are gene-clustered; simulated arms '
      'average 20&#8211;40 record sets.</figcaption>')
    w(ladder([('0.05', 0.05), ('0.01', 0.01), ('0.001', 0.001)], lad, 'allelic arm ladder'))
    w('</figure>')
    w('<p>Elimination 5 of 2026-09-24 was mis-measured. Its pooled regression of the standardized '
      'squared residual on log v gave a slope of &#8722;0.064; that is the within-gene slope of '
      '&#8722;0.164 multiplied by an attenuation of 0.3905, the within-gene share of the variance '
      'of log v, reproduced to 1e-6. The within-gene power-law exponent has a median near 0.65 in '
      'both channels, and the coupling&#8217;s sign varies by gene (log R<sub>g</sub> from &#8722;0.58 '
      'to +1.51), which no single exponent can express. Since the rejection rate is convex in '
      'R<sub>g</sub>, a spread of per-gene scales produces excess even where the mean is right.</p>')

    rows = []
    for r in bands.sort_values('R_g', ascending=False).itertuples():
        cls = S2 if r.u_shuf > 0.975 else (S3 if r.u_shuf < 0.025 else '')
        rows.append(dict(gene=r.gene, R=float(r.R_g), lo=float(r.sh_q025), hi=float(r.sh_q975), cls=cls))
    n_up = sum(1 for r in rows if r['cls'] == S2); n_dn = sum(1 for r in rows if r['cls'] == S3)
    w(f'<p>Per gene, the ratio lies outside its null band in both directions. Under a null that '
      f'keeps each gene&#8217;s real heavy-tailed residuals and shuffles only the pairing, {n_up} genes '
      f'sit above their 97.5% point against 1.15 expected and {n_dn} below the 2.5% point. The five '
      'most anticonservative genes by realized variance (CALM2, FABP7, ANKRD36B, MUTYH, TCF4) are '
      'the five largest ratios; CAMSAP2, APC, RANBP2, CNTN2 and PDZD8 are the smallest.</p>')
    w('<figure><figcaption>Coupling ratio R<sub>g</sub> per gene against its shuffle-null 95% band '
      '(grey bar). Orange: above the band; green: below; grey: inside. CALM2&#8217;s band is wide '
      'because one record holds 91% of its &#931;w&#183;z&#178;.</figcaption>')
    w(gene_bands(rows, 'coupling ratio per gene against null band'))
    w('</figure>')

    w('<h3>3.3 The total channel is a per-gene scale error</h3>')
    w(f'<p>The total channel&#8217;s excess ({ht[0.05][0]:.4f} / {ht[0.01][0]:.4f} / {ht[0.001][0]:.4f}) is '
      'entirely a per-gene inflation of the reported variance by the same kind of ratio, taken '
      'after partialling the covariates and corrected for leverage. Dividing each gene&#8217;s '
      'statistic by its own ratio gives 0.0505 / 0.0098 / 0.00085; unit weights give 0.0496. '
      'Its Gibbs variance is about 1/50 of the between-donor variance in high-coverage genes, '
      'so 1/v weights there are effectively depth weights (the correlation of expression with '
      'log weight has median 0.993), and the residual variance grows as v<sup>0.66</sup>. A '
      'per-donor variance component is also present (donor 221_D1, RIN 3.1, mean z&#178; 3.75 '
      'against a model maximum of 1.93), not converted to a rate.</p>')

    w('<h3>3.4 Budget of the shipped statistic</h3>')
    b = budget.set_index(['term', 'alpha'])
    def sh(term, a):
        r = b.loc[(term, a)]
        return float(r.share_of_E), float(r.share_lo), float(r.share_hi), float(r.value)
    E = {a: sh('E_REAL_minus_MODEL', a)[3] for a in ALPHAS}
    ac = {a: sh('ins_allelic_coupling', a) for a in ALPHAS}
    tc = {a: sh('ins_total_coupling', a) for a in ALPHAS}
    am = {a: sh('ins_allelic_marginal', a) for a in ALPHAS}
    tm = {a: sh('ins_total_marginal', a) for a in ALPHAS}
    ix = {a: sh('ins_interaction', a) for a in ALPHAS}
    cap = {a: sh('REAL_minus_CAP_PLUS_TOTAL_UNWEIGHTED', a) for a in ALPHAS}
    w(f'<p>Over a model matched to each channel&#8217;s real residual scale (without that matching '
      'the allelic channel&#8217;s share of the combined precision falls from 0.47 to 0.14 and the '
      f'budget misallocates), the combined excess is {E[0.05]:.4f} / {E[0.01]:.4f} / {E[0.001]:.4f}. '
      'Each component is inserted into the model world alone; the four main effects plus the '
      'interaction sum to the excess by construction.</p>')
    w('<div class="tw"><table><thead><tr><th class="l">component</th><th>0.05</th><th>0.01</th>'
      '<th>0.001</th></tr></thead><tbody>')
    for lab, dct in (('allelic weight-residual coupling', ac), ('total-channel scale', tc),
                     ('allelic heavy-tailed residuals', am), ('total heavy-tailed residuals', tm),
                     ('interaction', ix)):
        w(f'<tr><th>{lab}</th>' + ''.join(
            f'<td>{pct(dct[a][0])} [{100*dct[a][1]:.0f}, {100*dct[a][2]:.0f}]</td>' for a in ALPHAS) + '</tr>')
    w('<tr class="em"><th>mixQTL cap on allelic weights plus unweighted total, share removed</th>' +
      ''.join(f'<td>{pct(cap[a][0])}</td>' for a in ALPHAS) + '</tr>')
    w('</tbody></table></div>')
    w('<p class="sub">The interaction clears the standard error of its mean at 0.01 and 0.001 but not '
      'the spread of a single record set; its cross-channel part (the same donor&#8217;s two '
      'records moving together) is 5&#8211;14% and does not clear. The mixQTL cap is '
      'min(10, floor(n<sub>a</sub>/10)) times the smallest weight, 4&#8211;9 fold on these genes.</p>')
    srows = [dict(label=f'α = {a:g}', parts={'ac': ac[a][0], 'tc': tc[a][0],
                                                   'mg': am[a][0] + tm[a][0], 'ix': ix[a][0]})
             for a in ALPHAS]
    w('<figure><figcaption>Shares of the combined statistic&#8217;s excess over its scale-matched '
      'model.</figcaption>')
    w(stacked(srows, [('ac', 'allelic coupling', S1), ('tc', 'total scale', S2),
                      ('mg', 'heavy tails', S3), ('ix', 'interaction', 'S4'.lower())],
              'combined budget'))
    w('</figure>')

    w('<h3>3.5 Single records, and CALM2</h3>')
    w('<p>A selection control that keeps the real heavy-tailed residuals and applies the same '
      'drop rule to shuffled record sets is essential here; against a Gaussian control the '
      'shares below read 15% / 31% / 68%. Against the heavy-tailed control, dropping each '
      'gene&#8217;s top record (largest share of &#931;w&#183;z&#178;) removes 2% [&#8722;49, 24] / 18% '
      '[&#8722;51, 43] / 58% [&#8722;25, 75] of the allelic excess: no interval excludes zero. One '
      'record, CALM2 donor 657_D1, is 52% of this gene set&#8217;s pooled 0.001 excess (8% at 0.05). '
      'Without CALM2 no single-record share is detectable at any tier. Transcriptome-wide, one '
      f'record holding more than half of &#931;w&#183;z&#178; occurs in '
      f'{100*reach["b"]["transcriptome"]["dominance"]["observed_frac"]:.1f}% of genes against '
      f'{100*reach["b"]["transcriptome"]["dominance"]["model_expected_frac"]:.2f}% under the model, '
      'concentrated at low coverage and few informative donors.</p>')
    w('<div class="tw"><table><thead><tr><th class="l">gene</th><th class="l">donor</th>'
      '<th>share of &#931;w&#183;z&#178;</th><th>weight rank / n<sub>a</sub></th><th>a (Salmon)</th>'
      '<th>a (alignment)</th><th>reads L / R</th><th>Gibbs var / Poisson</th><th>discordance z</th>'
      '</tr></thead><tbody>')
    for r in dom.itertuples():
        w(f'<tr><th>{r.gene}</th><td class="l">{r.donor}</td><td>{r.share_wz2:.2f}</td>'
          f'<td>{int(r.w_rank)} / {int(r.n_a)}</td><td>{r.a:+.2f}</td><td>{r.phaser_lr:+.2f}</td>'
          f'<td>{r.mL:,.0f} / {r.mR:,.0f}</td><td>{r.gibbs_over_poisson:.1f}</td>'
          f'<td>{r.discord_z:+.1f}</td></tr>')
    w('</tbody></table></div>')
    w('<p class="sub">Records holding more than a quarter of their gene&#8217;s &#931;w&#183;z&#178;. '
      'The alignment-based ratio is phASER&#8217;s: allele counts from reads overlapping heterozygous '
      'SNPs in the genome alignment, blind to indels. The discordance z is (a<sub>Salmon</sub> '
      '&#8722; a<sub>alignment</sub>) over its combined standard error.</p>')
    w('<div class="warn"><p>657_D1&#8217;s Gibbs variance is 0.93&#215; the counting noise of its 1,282 '
      'haplotype-informative fragments, so the draws are honest about noise; but its ratio is '
      'about 12 of its own standard deviations from the alignment count, and a pileup at its '
      'distinguishing site reads 298 : 306. All three of its exonic heterozygous SNVs are cohort '
      'singletons and it carries six heterozygous indels; the mechanism is not identified. On '
      'observed data the record alone carries the CALM2 gene-level call: <code>map_cis</code> '
      '<code>pval_perm</code> is 0.028 with it and 0.684 without (the lead moves), while '
      'excluding any of 12 random other records leaves 0.008&#8211;0.036. CYCS moves 0.005 &#8594; '
      '0.049, FABP7 0.121 &#8594; 0.415, and APC 0.359 &#8594; 0.003 the other way. MATR3&#8217;s two '
      'dominant records are the only two carriers of an 89-bp deletion removing a splice '
      'donor.</p></div>')

    w('<h3>3.6 Sources of the coupling that were tested</h3>')
    w('<div class="tw"><table><thead><tr><th class="l">candidate</th><th class="l">status</th>'
      '<th class="l">deciding numbers</th></tr></thead><tbody>')
    for c, s, n in (
        ('Observed allelic association at the lead, carried into the permuted records',
         'partial, at 0.05 only',
         '13&#8211;22% of the allelic excess at 0.05 with leverage-corrected removal; 0.07 [0.00, 0.21] at 0.01; not estimable at 0.001.'),
        ('Records whose Salmon ratio disagrees with alignment counts by more than 3 sd',
         'refuted as the generator',
         '0.82% of records, 5.5% of &#931;(R&#8722;1); excluding them net of a weight-matched control removes 0.03 / 1.5 / 4.7 / 10% of the coupling at 0.05 / 0.01 / 0.001 / 1e-4. Collapses CALM2 alone (R 4.53 &#8594; 0.67). A diffuse Salmon-specific error is not excluded.'),
        ('A common additive between-donor floor (beta-binomial overdispersion)',
         'refuted as the form',
         'Reproduces the pooled rate (tau 0.004&#8211;0.012) but ranks no genes (|&#961;| &#8804; 0.18); count-based rho does not predict R<sub>g</sub>; a power law beats a floor in 38 of 46 genes.'),
        ('Mis-phased singletons in the personalized reference', 'not distinguishable',
         'Singleton exonic SNVs are phased against the read-backed phase in at most 19.1% of cases (2.0% for common variants) and double the odds of discordance, but reach 137 of 9,328 discordant records; 657_D1 cannot be tested (no read-backed site).'),
        ('Unmodelled signal at additional cis variants', 'not tested, by decision', 'Multi-SNP conditioning judged unreliable here; cancelled mid-run.'),
    ):
        w(f'<tr><th>{c}</th><td class="l">{s}</td><td class="l">{n}</td></tr>')
    w('</tbody></table></div>')
    w('<p>Roughly 55&#8211;65% of the allelic excess at 0.05, and about 84% of the combined '
      'statistic&#8217;s, is smooth positive coupling with no identified source. It is something '
      'both quantifiers see: phASER&#8217;s counts weighted by their own counting variance are '
      'also anticonservative (0.061 / 0.014 / 0.0020). Between-donor allelic variance shrinks '
      'with depth more slowly than the Gibbs variance does, and 1/v weights over-trust the '
      'deepest records; why, per gene, is open.</p>')

    w('<h3>3.7 It is a gene property, and it reaches the transcriptome</h3>')
    dp = reach['b']['transcriptome']['direct_permutation']
    ser = [
        dict(name='46 genes, allelic', cls=S1, vals={a: ha[a][0] for a in ALPHAS}),
        dict(name='46 genes, total', cls=S2, vals={a: ht[a][0] for a in ALPHAS}),
        dict(name='46 genes, combined', cls=S3, vals={a: hb[a][0] for a in ALPHAS}),
        dict(name='20,281 genes, allelic', cls='s4',
             vals={a: dp[k]['rate'] for a, k in ((0.05, '0.05'), (0.01, '0.01'), (0.001, '0.001'), (0.0001, '0.0001'))}),
    ]
    for s in ser[:3]:
        pass
    w('<p>The coupling&#8217;s direction is a stable property of the gene: split the donors into random '
      'halves 200 times and the rank coupling in one half correlates with the other at 0.53 '
      'across genes, against 0.01 for model records. A sampling null (records at their own '
      'positions, fresh errors each replicate) gives the same rates as the permutation null at '
      '0.05 and 0.01, so the finding is about <code>pval_nominal</code> on observed data too. About '
      '79% / 61% / 34% of the excess recurs in held-out halves. Transcriptome-wide, over '
      f'{reach["b"]["n_genes_min_na"]:,} genes with at least 20 informative donors, real records reject at '
      f'{dp["0.05"]["rate"]:.4f} / {dp["0.01"]["rate"]:.4f} / {dp["0.001"]["rate"]:.5f} / {dp["0.0001"]["rate"]:.5f} '
      'at 0.05 / 0.01 / 0.001 / 1e-4; direct decoupling puts the coupling&#8217;s share at 88 / 79 / 66 / 51%, '
      'so about half of the 1e-4 tail is the residual distribution, not the pairing.</p>')
    w('<figure><figcaption>Rejection rate over nominal against the nominal level. The 46-gene '
      'arms stop at 0.001; the transcriptome arm (synthetic Hardy-Weinberg variants at each '
      'gene) reaches 1e-4.</figcaption>')
    ser46 = ser[:3]
    for s in ser46:
        s['vals'] = {a: s['vals'][a] for a in ALPHAS}
    w(alpha_curves(ser46 + [ser[3]], [0.05, 0.01, 0.001, 0.0001], 'rate over nominal by alpha'))
    w('</figure>')
    w('<div class="tw"><table><thead><tr><th class="l">allele-resolved coverage</th><th>genes</th>'
      '<th>median R<sub>a</sub></th><th>genes above 97.5%</th><th>genes below 2.5%</th></tr></thead><tbody>')
    for r in strata[strata.by == 'cov_bin'].itertuples():
        w(f'<tr><th>{r.bin}</th><td>{int(r.n_genes):,}</td><td>{r.median_R_a:.3f}</td>'
          f'<td>{100*r.frac_Ra_above_q975:.0f}%</td><td>{100*r.frac_Ra_below_q025:.0f}%</td></tr>')
    w('</tbody></table></div>')
    w('<p class="sub">The band here is a Gaussian null, which understates the spread of a heavy-tailed '
      'ratio; the counts are indicative, not corrected. The 46-gene set is 67% genes of at least '
      '700 reads against 22% transcriptome-wide, and the 30&#8211;100-read stratum has the largest '
      'excess at 0.05 (0.075).</p>')

    # ---- 4. critique ------------------------------------------------------
    w('<h2>4. The critique, and what it changed</h2>')
    w('<p>Verifiers corrected eleven of the twelve investigations. The corrections that changed a '
      'conclusion:</p>')
    w('<dl>')
    for t, d in (
        ('Gaussian selection controls overstated single-record effects.',
         'Netted against a control that keeps the real heavy tails, the top-record share at 0.001 '
         'falls from 68% [2, 85] to 58% [&#8722;25, 75] and no longer clears zero; an &#8220;opposing '
         'couplings&#8221; reading of the conservative genes was withdrawn on the same correction.'),
        ('&#8220;Artefact&#8221; became &#8220;disagrees with alignment counts beyond counting error&#8221;.',
         'The alignment comparison cannot see indel-informative reads, and a mis-phased singleton '
         'in the personalized reference is an untested alternative for CALM2 and FABP7.'),
        ('&#8220;The Gibbs variance is correct&#8221; became &#8220;correct for counting noise&#8221;.',
         'It does not cover the point-estimate error, which at 657_D1 is about 12 of the reported sd.'),
        ('The spread of coupling across genes is 1.75&#215; the null, not 2.46&#215;.',
         'The larger figure came from a Gaussian null; a null that keeps the heavy-tailed residuals '
         'is the right comparison, and under it CALM2 itself sits inside its band.'),
        ('Lead removal needs a leverage correction in the allelic channel but not in the total channel.',
         'Uncorrected removal is conservative by construction in the allelic channel (0.042 under a '
         'true null); the same correction is anticonservative in the total channel (0.0527), where '
         'the covariate row travels with the record. The derived rescale e / &#8730;(1 &#8722; h<sub>g</sub>/(1 &#8722; h<sub>Z</sub>)) is nominal.'),
        ('The 2026-09-24 sign-flip count of 7 against 11 calls is amended, not withdrawn.',
         'At 2,000 draws it is 10 against 13, a gap of 3 [&#8722;1, 7]; the allelic-only empirical p is '
         'still larger under sign flip (mean +0.043 [0.010, 0.080]), and about a third of that shift '
         'is the observed lead association carried into the sign-flip null.'),
        ('An alignment follow-up headline was refuted by its own verifier.',
         '&#8220;99.8% of the coupling is in the part phASER reproduces&#8221; depended on a cancellation '
         'between phASER depth strata; at 20 or more reads the unshared part is 29%, so the Salmon-'
         'specific share is not bounded by that split.'),
        ('mixQTL is not &#8220;close to nominal&#8221;.',
         'Under its own normal reference all three intervals exclude nominal and 0.001 is 2.0&#215;.'),
    ):
        w(f'<dt>{t}</dt><dd>{d}</dd>')
    w('</dl>')
    w('<p>Two things the critics could not close. The per-tier shares rest on rejection rates, which '
      'are non-linear in each channel&#8217;s variance inflation, so part of what reads as '
      'interaction or as drift in the total share across tiers is the shape of the tail function. '
      'And at 0.001 the whole 46-gene tier is dominated by one gene, so its interval '
      '[0.0019, 0.0122] is as wide as the estimate.</p>')

    # ---- 5. meaning -------------------------------------------------------
    w('<h2>5. What it means</h2>')
    w('<div class="key"><p>The nominal p is anticonservative because the reported variance is a '
      'per-gene scale off from the variance the estimator realizes, and the scale is set by how '
      'each gene&#8217;s Gibbs weights pair with its residual sizes. The estimator and its '
      'reference are correct when the model holds. The excess is decomposed completely into '
      'that pairing plus a heavy-tailed residual distribution; its generative source is not '
      'identified. It transfers to observed data and to the transcriptome, and it is not '
      'Salmon-specific.</p></div>')
    w('<p>Against what was believed. The 2026-09-24 record put the excess at 1.6&#215; with five '
      'mechanisms eliminated; two of those eliminations were mis-measured and the figure was a '
      '30-draw high. The cross-donor-dependence candidate cannot explain a records-permutation '
      'excess and is now three statements: structurally excluded here, untested for observed-data '
      'correlation, and present as a per-donor variance component in the total channel. The '
      'external simulation benchmark&#8217;s tail (0.064 / 0.020 at 0.05 / 0.01) has the same '
      'mechanism: beta-binomial overdispersion is an additive term on the log-ratio scale against '
      'a Gibbs variance that shrinks with depth.</p>')
    w('<p>What the detection call inherits. <code>pval_perm</code> is built from this null and is '
      'unaffected by the scale error, but it does not protect against a call carried by one '
      'record, as CALM2 shows. Nothing below 0.001 on the 46 genes or below 1e-4 '
      'transcriptome-wide was measured, and transcriptome-scale thresholds lie below both.</p>')
    w('<h3>Proposed fixes, as measurements, ranked</h3>')
    w('<p>None is implemented; each states what it changes, what it leaves alone, and the number '
      'it should produce if the mechanism is as described. Deprecated per-gene fitted variance '
      'functions, the mixQTL cap (two thirds of the efficiency gain) and multi-SNP conditioning '
      'are not proposed.</p>')
    w('<ol>')
    for t in (
        '<b>Unweight the total channel.</b> Its Gibbs variance is about 1/50 of the between-donor '
        'variance, so 1/v<sub>t</sub> is a depth weight. Unit weights take the channel to 0.0496 and '
        'remove 28 / 19 / 11% of the combined excess; the efficiency cost is unmeasured and expected '
        'small. No allelic weight is touched.',
        '<b>A permutation-matched standard error as an extra column.</b> se&#178; &#215; R<sub>g</sub> '
        '(leverage-corrected for the total channel) is the variance the records null implies, in '
        'closed form. Measured for the total channel: 0.0501 / 0.0097 / 0.0008. Point estimate, '
        'weights and efficiency unchanged; still off where one record dominates.',
        '<b>Influence reporting at the lead.</b> The top record&#8217;s share of &#931;w&#183;z&#178;, its donor, '
        'and the change in <code>pval_perm</code> without it. Pure reporting; would have flagged CALM2, '
        'CYCS, FABP7 and APC.',
        '<b>A per-record alignment-consistency flag.</b> Does not repair pooled calibration but '
        'neutralises the dominant records; fitted from nothing in the regression. Explaining the '
        'mechanism is blocked until the personalized transcript FASTA and GTF are regenerated.',
        '<b>One global shape exponent</b> (Var &#8733; v<sup>&#947;</sup>, &#947; &#8776; 0.65 over 20,000 genes). '
        'Inferred to remove the common half of the allelic excess at 0.05 and little of the tail; it '
        'changes the weights, so it is listed last.',
    ):
        w(f'<li><p>{t}</p></li>')
    w('</ol>')

    w('<h2>Where the pieces are</h2>')
    w('<p class="note">Instrument and investigations: <code>scripts/null_permutation_instrument.py</code>, '
      '<code>weight_residual_coupling.py</code>, <code>total_channel_null_calibration.py</code>, '
      '<code>dominant_record_anatomy.py</code>, <code>allelic_overdispersion_floor.py</code>, '
      '<code>imbalance_downweighting.py</code>, <code>comparator_null_2000.py</code>, '
      '<code>coupling_transfer_to_observed.py</code>, <code>coupling_reach.py</code>, '
      '<code>combined_statistic_budget.py</code>, <code>dominant_record_share_corrected.py</code>, '
      '<code>lead_signal_share_corrected.py</code>, <code>alignment_discordance_coupling.py</code> '
      '(branch mixqtl-replication). Outputs: <code>brainvar_hapmix_deploy/*_20260925/</code>. '
      'The verification record, first-round critique, follow-ups and reconciliation: '
      '<code>brainvar_hapmix_deploy/nominal_p_hypotheses_20260925/</code>.</p>')
    w('<p class="note">Terms. Gene-clustered bootstrap: resample the 46 genes with replacement and '
      'recompute the pooled rate; brackets are its 2.5% and 97.5% points. Spearman &#961;: the Pearson '
      'correlation of ranks. Kolmogorov&#8211;Smirnov: the largest gap between the empirical and the '
      'uniform distribution function. Leverage h: the diagonal of the weighted hat matrix. phASER: '
      'haplotype-aware allele counts from reads overlapping heterozygous SNPs in the genome '
      'alignment, independent of Salmon.</p>')
    w('</div>')
    return '\n'.join(o)


def main():
    X = load()
    html = build(X)
    OUT.write_text(html)
    print(f'wrote {OUT} ({len(html):,} bytes)')


if __name__ == '__main__':
    main()

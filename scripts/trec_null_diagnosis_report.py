"""Summary and page for scripts/trec_null_diagnosis.R: why asSeq's total-count (TReC) test rejects null genes above
nominal on the simulated-effects benchmark's all-null dataset.

Reads OUT/<set>/<gene>.tsv and .meta.tsv. Checks first that the 'real' part reproduces the TReC p the benchmark's
TReCASE run wrote for the same variants (trecase_work/beta0.0/rep000/out/<gene>_eqtl.txt). Per set and part, the share
of tests with p below each threshold, with a gene-clustered 95% interval (genes resampled with replacement), beside the
heuristic prediction for a likelihood-ratio statistic inflated by n / (n - p). Writes OUT/summary.json and
OUT/index.html.
"""
import base64
import io
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import chi2, spearmanr

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

D = Path('/mnt/ssd/lalli/brainvar_hapmix_deploy')
OUT = D / 'trec_null_diagnosis_20261002'
SETS = (('deep', 'Deep set', 'simulated_effects_half_read_20261001'),
        ('lowcov', 'Low-coverage set', 'simulated_effects_lowcov_half_read_20261001'))
PARTS = (('real', 'real totals, 17 covariates (as benchmarked)', '#e34948'),
         ('sim', 'simulated from asSeq\'s own null fit, 17 covariates', '#2a78d6'),
         ('fewcov', 'real totals, 3 genotype PCs only', '#1baf7a'))
ALPHAS = (0.05, 0.01, 0.001)
N_DONORS, P_MEAN = 92, 19     # donors; mean parameters under the alternative (intercept, 17 covariates, dosage)
P_FEW = 5                     # intercept, 3 genotype PCs, dosage
SEED, N_BOOT = 42, 2000
SC = {k: pd.read_csv(OUT / k / 'scale.tsv', sep='\t').set_index('gene') for k, *_ in SETS}   # trec_null_scale_check.R


def predicted(alpha, p):
    """Share of chi2(1) draws above the alpha quantile once the statistic is inflated by n / (n - p)."""
    return float(chi2.sf(chi2.isf(alpha, 1) * (N_DONORS - p) / N_DONORS, 1))


def load(key, root):
    rows, meta = [], []
    for f in sorted((OUT / key).glob('*.meta.tsv')):
        g = f.name[:-len('.meta.tsv')]
        meta.append(pd.read_csv(f, sep='\t'))
        rows.append(pd.read_csv(OUT / key / f'{g}.tsv', sep='\t').assign(gene=g))
    d, m = pd.concat(rows, ignore_index=True), pd.concat(meta, ignore_index=True)
    m.attrs['key'] = key
    stored_dir = D / root / 'trecase_work' / 'beta0.0' / 'rep000' / 'out'
    worst, n = 0.0, 0
    for g, r in d[d.part == 'real'].groupby('gene'):
        s = pd.read_csv(stored_dir / f'{g}_eqtl.txt', sep='\t').set_index('MarkerRowID').TReC_Pvalue
        both = r.set_index('MarkerRowID').Pvalue.reindex(s.index).dropna()
        x = s.loc[both.index].astype(float)
        worst = max(worst, float(np.max(np.abs(np.log10(both.values) - np.log10(x.values)))) if len(both) else 0.0)
        n += len(both)
    if worst > 1e-9:
        raise SystemExit(f'{key}: trec on the real totals differs from the stored TReC p (max |log10 diff| {worst})')
    failed = sorted(f.name.split('.')[0] for f in (OUT / key).glob('*.baseline_failed'))
    print(f'{key}: {m.gene.nunique()} genes, {len(d):,} rows; real part equals the stored TReC p on {n:,} tests; '
          f'baseline model failed (no TReC tests, as in the benchmark): {failed}', flush=True)
    return d, m


def rates(d, rng):
    """Per alpha: share of tests with p below it and its gene-clustered interval."""
    genes = d.gene.unique()
    idx = {g: i for i, g in enumerate(genes)}
    gi = d.gene.map(idx).values
    n = np.bincount(gi, minlength=len(genes)).astype(float)
    boot = rng.integers(0, len(genes), size=(N_BOOT, len(genes)))
    out = {}
    for a in ALPHAS:
        k = np.bincount(gi, weights=(d.Pvalue.values < a).astype(float), minlength=len(genes))
        b = k[boot].sum(1) / n[boot].sum(1)
        out[str(a)] = dict(rate=float(k.sum() / n.sum()), lo=float(np.quantile(b, .025)), hi=float(np.quantile(b, .975)),
                           tests=int(n.sum()))
    return out


def per_gene(d, m):
    """Per gene: share of tests with p < 0.05 on the real and on the simulated totals, and its fit statistics."""
    r = d[d.part.isin(['real', 'sim'])].assign(rej=lambda x: x.Pvalue < 0.05).groupby(['gene', 'part']).rej.mean().unstack()
    return r.join(m.set_index('gene')).join(SC[m.attrs['key']], how='inner').assign(
        excess=lambda x: x.real - x.sim, scale_above=lambda x: x.scale_real - x.scale_sim)


def png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def figure(S, G):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.9), gridspec_kw=dict(width_ratios=[1, 1, 1.1]))
    for ax, (key, name, _) in zip(axes[:2], SETS):
        for i, a in enumerate(ALPHAS):
            ax.plot([i - 0.35, i + 0.35], [S[key]['predicted'][str(a)] / a] * 2, color='#0b0b0b', lw=1.5,
                    label='predicted, n/(n - p), 17 covariates' if i == 0 else None)
            for j, (p, lab, col) in enumerate(PARTS):
                r = S[key]['parts'][p][str(a)]
                ax.errorbar(i + (j - 1) * 0.2, r['rate'] / a, yerr=[[(r['rate'] - r['lo']) / a], [(r['hi'] - r['rate']) / a]],
                            fmt='o', color=col, ms=5, capsize=2, label=lab if i == 0 else None)
        ax.axhline(1, color='#9a9c9f', lw=1, ls=':')
        ax.set_xticks(range(3), [f'p < {a}' for a in ALPHAS])
        ax.set_ylabel('null-gene rate / threshold')
        ax.set_title(f'{name}', loc='left', fontsize=10)
    ax = axes[2]
    for (key, name, _), mk in zip(SETS, 'os'):
        g = G[key]
        ax.scatter(g.scale_above, g.excess, s=14, marker=mk, alpha=0.7, label=name)
    ax.axhline(0, color='#9a9c9f', lw=1, ls=':')
    ax.axvline(0, color='#9a9c9f', lw=1, ls=':')
    ax.set_xlabel("asSeq's scale, real minus its value on simulated totals")
    ax.set_ylabel('real minus simulated rate at 0.05')
    ax.set_title('Per gene', loc='left', fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    for a_ in axes:
        a_.spines[['top', 'right']].set_visible(False)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=2, frameon=False, fontsize=8.5)
    return png(fig)


CSS = '''
:root { --bg: #f7f7f4; --panel: #ffffff; --ink: #15171a; --ink2: #575b61; --rule: #dfe0db; --accent: #2a78d6;
  --display: "Source Serif 4", Georgia, serif; --body: "IBM Plex Sans", system-ui, sans-serif; --mono: "IBM Plex Mono", ui-monospace, monospace; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --bg: #141517; --panel: #1c1d20; --ink: #f1f1ee;
  --ink2: #b4b6b9; --rule: #33353a; --accent: #3987e5; color-scheme: dark; } }
:root[data-theme="dark"] { --bg: #141517; --panel: #1c1d20; --ink: #f1f1ee; --ink2: #b4b6b9; --rule: #33353a; --accent: #3987e5;
  color-scheme: dark; }
body { background: var(--bg); color: var(--ink); font: 15px/1.6 var(--body); }
main { max-width: 980px; margin: 0 auto; padding-inline: 16px; padding-block: 32px 72px; display: grid; gap: 14px; }
h1 { font: 600 30px/1.15 var(--display); margin: 0; text-wrap: balance; }
h3 { font: 600 13px var(--mono); letter-spacing: 0.06em; text-transform: uppercase; color: var(--ink2); margin: 14px 0 0; }
p { margin: 0; max-width: 75ch; } .finding { font-size: 16px; } .eyebrow { font: 600 11.5px var(--mono); letter-spacing: 0.08em;
  text-transform: uppercase; color: var(--accent); margin: 0; }
figure { margin: 4px 0; } figure img { max-width: 100%; height: auto; display: block; background: #ffffff; border-radius: 6px; }
figcaption { color: var(--ink2); font-size: 13px; margin-top: 6px; max-width: 90ch; }
.scroll { overflow-x: auto; min-width: 0; } table { border-collapse: collapse; font-size: 13px; }
th, td { border-bottom: 1px solid var(--rule); padding: 4px 10px 4px 0; text-align: left; } td { font-variant-numeric: tabular-nums; }
th { color: var(--ink2); font-weight: 600; } footer { color: var(--ink2); font-size: 13px; } code { font: 0.88em var(--mono); }
'''


def page(S, G, rho):
    med = {k: dict(real=float(SC[k].scale_real.median()), sim=float(SC[k].scale_sim.median())) for k, *_ in SETS}
    f4 = lambda r: f'{r["rate"]:.4f} [{r["lo"]:.4f}, {r["hi"]:.4f}]'   # noqa: E731
    rows = ''.join(f'<tr><td>{name}</td><td>{lab}</td>' + ''.join(f'<td>{f4(S[k]["parts"][p][str(a)])}</td>' for a in ALPHAS) + '</tr>'
                   for k, name, _ in SETS for p, lab, _ in PARTS)
    rows += ''.join(f'<tr><td>both</td><td>predicted, n/(n - p) with p = {pp}</td>'
                    + ''.join(f'<td>{predicted(a, pp):.4f}</td>' for a in ALPHAS) + '</tr>' for pp in (P_MEAN, P_FEW))
    inside = all(S[k]['parts']['real'][str(a)]['lo'] <= S[k]['parts']['sim'][str(a)]['rate'] <= S[k]['parts']['real'][str(a)]['hi']
                 for k, *_ in SETS for a in ALPHAS)
    sim05 = [S[k]['parts']['sim']['0.05']['rate'] for k, *_ in SETS]
    real05 = [S[k]['parts']['real']['0.05']['rate'] for k, *_ in SETS]
    few_in = {k: S[k]['parts']['fewcov']['0.05']['lo'] <= predicted(0.05, P_FEW) <= S[k]['parts']['fewcov']['0.05']['hi'] for k, *_ in SETS}
    return f'''<title>TReCASE null excess</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@400;600&family=Source+Serif+4:opsz,wght@8..60,600&display=swap">
<style>{CSS}</style><main><p class="eyebrow">hapmixQTL · 2026-10-02</p>
<h1>Why TReCASE's total-count test rejects null genes above nominal</h1>
<p class="finding">The test's own small-sample behaviour explains it. On totals simulated from asSeq's own fitted null model, where
the test's model is exactly true, asSeq's total-count test gives p &lt; 0.05 to {sim05[0]:.3f} and {sim05[1]:.3f} of null tests on the
deep and low-coverage genes, against 0.05 expected; on the real totals it gives {real05[0]:.3f} and {real05[1]:.3f}.
{"Every real-data rate's interval contains the simulated rate, so no misfit of its negative-binomial model to these data is detectable beyond that." if inside else "Somewhere the real-data rate's interval excludes the simulated rate (the table), so part of the excess is misfit to these data."}</p>
<figure><img alt="Null-gene rate over threshold for real, simulated and reduced-covariate totals beside the prediction, both gene sets, and per-gene excess against asSeq's scale" src="data:image/png;base64,{figure(S, G)}">
<figcaption>Left and middle: the share of tests on null genes with p below each threshold, divided by the threshold (1 = nominal),
with gene-clustered 95% intervals (genes resampled with replacement); black bars, the prediction for a likelihood-ratio
statistic inflated by n/(n - p) with n = 92 donors and p = 19 mean parameters. Right: per gene, the share below 0.05 on the
real totals minus that on the simulated totals, against how far asSeq's scale for the gene's null fit sits above its mean
over totals simulated from that fit.</figcaption></figure>
<h3>Why it was needed</h3>
<p>On the benchmark's all-null dataset TReCASE's total-count (TReC) test, a negative-binomial regression of each donor's total
count on genotype dosage with 17 covariates and the library size as offset, gives p &lt; 0.05 to about 0.08 of null tests,
against tensorQTL's 0.045 on the same totals. It does so on Salmon's totals and on alignment counts alike, and rounding the
totals changes nothing (2026-09-28). Whether that is how asSeq computes the test or how the data depart from its model was
open; the paper has to say which.</p>
<h3>What was run</h3>
<p>asSeq's own unmodified <code>trec</code>, which reproduces the TReC p of the benchmark's TReCASE runs exactly, on the all-null
dataset of both gene sets, on one draw of 300 tested variants per gene (seed 42, one stream per gene), three ways: the real
totals with the 17 covariates, as benchmarked; 20 sets of totals per gene simulated from asSeq's own null fit of that gene
(<code>glmNB</code>, the same fit <code>trec</code> uses: its fitted means and dispersion), with the same covariates, so the test's
model holds exactly; and the real totals with only the 3 genotype principal components as covariates. Simulated totals are
integers and real ones fractional, which the rounding check showed does not matter. In asSeq's source the dispersion is
re-estimated under the alternative (<code>glmEQTL.c</code>), so the fitted model has 19 mean parameters, the counterpart of a normal
model's 19 coefficients with its variance estimated; for that normal model the likelihood-ratio statistic is inflated by
about n/(n - p) = 92/73 against its chi-square reference, which gives the prediction. One low-coverage gene, SLC26A7, has no
TReC test here, as in the benchmark, because asSeq's null fit fails for it.</p>
<h3>Result</h3>
<div class="scroll"><table><thead><tr><th>gene set</th><th>totals</th><th>p &lt; 0.05</th><th>p &lt; 0.01</th><th>p &lt; 0.001</th></tr></thead>
<tbody>{rows}</tbody></table></div>
<p>Per gene, the real-minus-simulated share below 0.05 does not follow the gene's depth (Spearman {rho["deep"]["median_total"]:.2f} and
{rho["lowcov"]["median_total"]:.2f}, deep and low-coverage set). asSeq's scale, the weighted sum of squared working residuals
of the null fit over its residual degrees of freedom (<code>glm.c</code>), has a median of {med["deep"]["sim"]:.3f} and {med["lowcov"]["sim"]:.3f}
on simulated totals, where the model holds, against {med["deep"]["real"]:.3f} and {med["lowcov"]["real"]:.3f} on the real ones. The excess rises
with how far a gene's scale sits above its own simulated value (Spearman {rho["deep"]["scale_above"]:.2f} and {rho["lowcov"]["scale_above"]:.2f}):
genes whose totals vary more than the fitted model allows carry a little more.</p>
<h3>Critique</h3>
<p>The prediction is a heuristic from the normal linear model applied to a count model; the simulation is the measurement, and it
agrees with the prediction at 0.05 and 0.01 and sits slightly above it at 0.001. Each gene contributes 300 of its tested
variants, and the real totals are one dataset, so real-data intervals are wide; a misfit smaller than those intervals cannot be
excluded, and the per-gene relation with scale says some genes carry one. Dropping covariates is the weaker check, because it also
removes the structure the expression principal components absorb: the reduced-covariate rate {"agrees with its prediction" if few_in["deep"] else "does not agree with its prediction"} on the deep set and {"agrees" if few_in["lowcov"] else "does not"}
on the low-coverage set, where the missing structure plausibly keeps it high.</p>
<h3>What it means</h3>
<p>TReCASE's total-count test is anticonservative because a likelihood-ratio test with a chi-square reference does not hold its
level with 19 parameters on 92 donors, not because these totals, Salmon's or alignment counts, break its negative-binomial
model. tensorQTL and hapmixQTL refer their statistics to t distributions on the residual degrees of freedom, which carry that
correction. In the benchmark's power at 5% realized false discoveries this costs TReCASE through null genes that outrank real
effects; the comparison stands as run, with the published software unchanged (user decision, 2026-10-02).</p>
<footer><p><code>scripts/trec_null_diagnosis.R</code> (run by <code>{OUT}/run_chain.sh</code>), summary
<code>scripts/trec_null_diagnosis_report.py</code>; outputs and <code>summary.json</code> in <code>{OUT}</code>.</p></footer></main>'''


def main():
    rng = np.random.default_rng(SEED)
    S, G, rho = {}, {}, {}
    for key, name, root in SETS:
        d, m = load(key, root)
        S[key] = dict(parts={p: rates(d[d.part == p], rng) for p, *_ in PARTS},
                      predicted={str(a): predicted(a, P_MEAN) for a in ALPHAS},
                      predicted_fewcov={str(a): predicted(a, P_FEW) for a in ALPHAS},
                      genes=int(m.gene.nunique()), median_scale=float(m.scale.median()))
        G[key] = per_gene(d, m)
        rho[key] = {c: float(spearmanr(G[key].excess, G[key][c])[0]) for c in ('median_total', 'scale_above', 'phi')}
        S[key]['per_gene_spearman_excess'] = rho[key]
    tmp = OUT / 'summary.json.tmp'
    tmp.write_text(json.dumps(S, indent=1))
    tmp.replace(OUT / 'summary.json')
    for key, *_ in SETS:
        print(key, {p: {a: round(v['rate'], 4) for a, v in r.items()} for p, r in S[key]['parts'].items()},
              'predicted', {a: round(v, 4) for a, v in S[key]['predicted'].items()}, 'spearman', {c: round(v, 2) for c, v in rho[key].items()})
    html = page(S, G, rho)
    tmp = OUT / 'index.html.tmp'
    tmp.write_text(html)
    tmp.replace(OUT / 'index.html')
    print(f'wrote {OUT / "summary.json"} and {OUT / "index.html"} ({len(html.encode()):,} bytes)')


if __name__ == '__main__':
    main()

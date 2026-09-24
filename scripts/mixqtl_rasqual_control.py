"""Is the hapmixQTL-vs-RASQUAL effect-size relationship specific to hapmixQTL?

hapmixQTL's log allelic fold change regressed on RASQUAL's, read at the SAME
variant, decomposes to a true scale ratio of about 0.78 with a ~1.28 selection
inflation at each arm's own lead. That is only a statement about hapmixQTL if a
DIFFERENT method would not show the same attenuation. mixQTL is the control:
the published estimator, the same Salmon quantifications, posterior-mean counts,
and it never touches the draws.

DESIGN. The primary comparison is read at RASQUAL's lead, because there neither
hapmixQTL nor mixQTL selected the variant, so both slopes against RASQUAL are
free of the winner's curse and directly comparable to each other. hapmixQTL's
lead and the union are secondary and carry the curse explicitly.

THREE TRAPS, each handled rather than assumed:

  UNITS. mixQTL's response is log2 with a kappa=0.5 pseudocount; RASQUAL's
  log(pi/(1-pi)) and hapmixQTL's slope are natural log. A beta per log2 unit
  becomes a beta per natural-log unit by multiplying by ln 2. Skipping this
  inflates every mixQTL slope by 1/ln2 = 1.443. The conversion is checked
  against the 10-gene three-way table already on disk, whose betas are
  documented as natural log.

  DONOR SET. mixQTL's published cutoffs include an upper bound y <= 1000 that
  removes most well-expressed donors and leaves many genes total-counts-only.
  Applying them here would compare RASQUAL's allelic estimate against a mixQTL
  number carrying no allelic information at all. The cutoffs are therefore left
  at the module defaults and the per-variant `method` mixQTL itself reports
  (trc / asc / meta) is recorded, so any variant where mixQTL fell back to
  total counts can be seen and excluded.

  LOW COUNTS. The kappa=0.5 pseudocount attenuates the response where counts
  are small, so a mixQTL-vs-RASQUAL gap on the LOW stratum is partly the
  transform rather than the estimator. Results are broken out by stratum.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import compare_mixqtl_replication as CM          # noqa: E402
import tensorqtl.mixqtl_replication as MX        # noqa: E402

D = Path('/mnt/ssd/lalli/brainvar_hapmix_deploy')
RUN = D / 'rasqual_default_mode_20260923'
LN2 = np.log(2.0)


def mixqtl_at_variants(I, Y1, Y2, YT, g, j, want_ids, cuts):
    """mixQTL's beta/se at NAMED variants of gene g, not at its own lead."""
    vsel = CM.gene_variant_index(I, g)
    if vsel.size == 0:
        return {}
    v = I['vdf'].iloc[I['idx']].iloc[vsel]
    pos = {str(vid): k for k, vid in enumerate(v.index)}
    cols = [(vid, pos[vid]) for vid in want_ids if vid in pos]
    if not cols:
        return {}
    keep = I['keep']
    h1 = I['xL'][I['idx']][:, keep][vsel].T.astype(float)
    h2 = I['xR'][I['idx']][:, keep][vsel].T.astype(float)
    out = MX.mixqtl_scan(Y1[j], Y2[j], YT[j], I['lib_size'],
                         h1, h2, covariates=I['cov_df'].values, **cuts)
    res = {}
    for vid, k in cols:
        b, se = out['meta']['beta'][k], out['meta']['se'][k]
        if np.isfinite(b):
            res[vid] = dict(beta_ln=float(b) * LN2, se_ln=float(se) * LN2,
                            beta_raw_log2=float(b),
                            method=str(out['meta']['method'][k]))
    return res


def fit(x, y, label):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 5:
        print(f'  {label:34s} n={m.sum()} -- too few'); return None
    sl, ic, r, p, se = stats.linregress(x[m], y[m])
    sign = np.mean(np.sign(x[m]) == np.sign(y[m]))
    print(f'  {label:34s} n={m.sum():3d}  slope {sl:6.3f} +/- {se:.3f}   '
          f'r {r:5.3f}   sign {sign:.2f}')
    return dict(n=int(m.sum()), slope=float(sl), se=float(se), r=float(r),
                sign_agreement=float(sign))


def main():
    # Which cutoffs. The module defaults ARE mixQTL's published settings, whose
    # upper bound y <= 1000 removes most well-expressed donors: run under them
    # and mixQTL falls back to total counts on most of these high-coverage
    # genes, so its "allelic fold change" carries no allelic information and
    # comparing it to RASQUAL's is meaningless. Both settings are therefore run
    # and reported, with the channel mixQTL actually used recorded per variant.
    global CUTS
    which = sys.argv[1] if len(sys.argv) > 1 else 'permissive'
    CUTS = (dict(MX.PACKAGE_DEFAULT_CUTOFFS) if which == 'permissive'
            else dict(MX.PUBLISHED_CUTOFFS))
    print(f'cutoffs: {which} -> {CUTS}')
    me = pd.read_csv(RUN / 'matched_effects.tsv', sep='\t')
    strata = pd.read_csv(D / 'genes_59_strata_20260923.tsv', sep='\t').set_index('gene')
    genes_file = D / 'genes_59_stratified_20260923.txt'

    print('loading shared inputs for the 59 genes ...', flush=True)
    # regions MUST accompany the gene list: the loader's default regions.bed
    # covers only the 29 calibration genes, and passing the 59-gene list alone
    # silently loads no variants for the other 30, which then look like genes
    # mixQTL could not estimate at. Use the 59-gene regions the RASQUAL run
    # itself wrote, so both arms see the same windows by construction.
    I = CM.load_inputs(gene_list=str(genes_file),
                       regions=str(RUN / 'regions.bed'))
    gi = {g: j for j, g in enumerate(I['genes'])}
    # the cache holds PER-DRAW arrays; mixQTL consumes posterior-mean counts and
    # never the draws -- that is what makes it the no-draws comparator
    keep = I['keep']
    Y1, Y2, YT = MX.summaries_from_gibbs_posterior_mean(I['YL'], I['YR'], I['YT'])
    Y1, Y2, YT = Y1[:, keep], Y2[:, keep], YT[:, keep]

    rows = []
    for _, r in me.iterrows():
        g = r['gene']
        if g not in gi:
            continue
        want = [str(r['lead_h']), str(r['lead_r'])]
        got = mixqtl_at_variants(I, Y1, Y2, YT, g, gi[g], want, CUTS)
        rows.append(dict(
            gene=g, stratum=strata.loc[g, 'stratum'] if g in strata.index else '?',
            lead_h=r['lead_h'], lead_r=r['lead_r'],
            afc_h=r['afc_h'], afc_r_at_h=r['afc_r_at_h'],
            afc_r=r['afc_r'], afc_h_at_r=r['afc_h_at_r'],
            mx_at_h=got.get(str(r['lead_h']), {}).get('beta_ln', np.nan),
            mx_se_at_h=got.get(str(r['lead_h']), {}).get('se_ln', np.nan),
            mx_method_at_h=got.get(str(r['lead_h']), {}).get('method', ''),
            mx_at_r=got.get(str(r['lead_r']), {}).get('beta_ln', np.nan),
            mx_se_at_r=got.get(str(r['lead_r']), {}).get('se_ln', np.nan),
            mx_method_at_r=got.get(str(r['lead_r']), {}).get('method', '')))
    t = pd.DataFrame(rows)
    out = D / f'mixqtl_rasqual_control_20260924/{which}'
    out.mkdir(parents=True, exist_ok=True)
    t.to_csv(out / 'three_way_matched_59.tsv', sep='\t', index=False)
    print(f'\n{len(t)} genes; mixQTL read at both leads where the variant exists')
    print('mixQTL channel used at RASQUAL\'s lead:',
          t.mx_method_at_r.value_counts().to_dict())
    print('mixQTL channel used at hapmixQTL\'s lead:',
          t.mx_method_at_h.value_counts().to_dict())

    res = {}
    print('\n=== PRIMARY: at RASQUAL\'s lead (neither hapmixQTL nor mixQTL chose it) ===')
    res['at_rasqual_lead'] = {
        'hapmixQTL_on_RASQUAL': fit(t.afc_r, t.afc_h_at_r, 'hapmixQTL on RASQUAL'),
        'mixQTL_on_RASQUAL': fit(t.afc_r, t.mx_at_r, 'mixQTL on RASQUAL'),
        'hapmixQTL_on_mixQTL': fit(t.mx_at_r, t.afc_h_at_r, 'hapmixQTL on mixQTL')}

    print('\n=== SECONDARY: at hapmixQTL\'s lead (hapmixQTL chose it; curse inflates it) ===')
    res['at_hapmixqtl_lead'] = {
        'hapmixQTL_on_RASQUAL': fit(t.afc_r_at_h, t.afc_h, 'hapmixQTL on RASQUAL'),
        'mixQTL_on_RASQUAL': fit(t.afc_r_at_h, t.mx_at_h, 'mixQTL on RASQUAL'),
        'hapmixQTL_on_mixQTL': fit(t.mx_at_h, t.afc_h, 'hapmixQTL on mixQTL')}

    print('\n=== by stratum, at RASQUAL\'s lead ===')
    res['by_stratum'] = {}
    for s in ['HIGH', 'MID', 'LOW']:
        sub = t[t.stratum == s]
        if len(sub) >= 5:
            print(f'  -- {s} (n={len(sub)}) --')
            res['by_stratum'][s] = {
                'hapmixQTL_on_RASQUAL': fit(sub.afc_r, sub.afc_h_at_r, f'{s}: hm on rq'),
                'mixQTL_on_RASQUAL': fit(sub.afc_r, sub.mx_at_r, f'{s}: mx on rq')}

    # validation: the 10-gene three-way table is documented as natural log
    t3 = D / 'three_way_matched_20260923' / 'three_way_effects.tsv'
    if t3.exists():
        old = pd.read_csv(t3, sep='\t')
        j = old.merge(t, on='gene')
        j = j[j.lead == j.lead_h]
        if len(j):
            d = (j.mx_b - j.mx_at_h).abs()
            print(f'\nunit check against the 10-gene three-way table '
                  f'({len(j)} genes, same lead): max |diff| in mixQTL beta '
                  f'{d.max():.3e}, median {d.median():.3e}')
            res['unit_check_vs_three_way'] = {
                'n': int(len(j)), 'max_abs_diff': float(d.max()),
                'median_abs_diff': float(d.median()),
                'note': ('mixQTL betas recomputed here, converted log2 -> natural '
                         'log by x ln2, against the earlier table documented as '
                         'natural log. Near-zero confirms the conversion.')}
    (out / 'control_slopes.json').write_text(json.dumps(res, indent=2, default=float))
    print(f'\nwrote {out}/three_way_matched_59.tsv and control_slopes.json')


if __name__ == '__main__':
    main()

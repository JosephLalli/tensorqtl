"""Does the EMPIRICAL permutation p differ between the two allelic nulls?

The nominal p was anticonservative under both schemes and much more so under
sign flip. The defence of the shipped pipeline has been that detection uses the
EMPIRICAL permutation p, which is calibrated by construction whatever the
residual distribution. That defence assumes the null used to build it is the
right one. This measures the assumption instead of asserting it.

For each gene, at the same fixed variant, the empirical p is the fraction of
null draws whose statistic is at least as extreme as the observed one:

    p_emp = (1 + #{p_null <= p_obs}) / (1 + n_draw)

with the parametric p used as the ordering statistic, which is monotone in the
statistic and so gives the same ordering. Computed once per scheme on the same
observed data, so any difference is the null construction alone.

PREDICTION, from the null distributions already measured: sign flip puts more
mass at small p (0.117 below 0.05 against 0.068), so for a given observed
statistic more sign-flip draws should beat it, giving a LARGER and more
conservative empirical p. If that holds, the shipped records-based pval_perm is
anticonservative for the allelic channel relative to the channel's own symmetry.

Resolution is 1/31 with 30 draws, so this compares distributions across 46
genes, not individual genes.
"""
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import compare_mixqtl_replication as CM        # noqa: E402
from tensorqtl.hapmixqtl import map_cis        # noqa: E402

D = Path('/mnt/ssd/lalli/brainvar_hapmix_deploy')
RUN = D / 'rasqual_default_mode_20260923'
NULLS = D / 'allelic_null_schemes_20260924'


def main():
    nl = pd.read_csv(NULLS / 'pvals.tsv', sep='\t')
    n_draw = int(nl.perm.max()) + 1
    me = pd.read_csv(RUN / 'matched_effects.tsv', sep='\t')
    null46 = {l.strip() for l in open(D / 'genes_null46_20260923.txt') if l.strip()}
    strata = pd.read_csv(D / 'genes_59_strata_20260923.tsv', sep='\t').set_index('gene')
    targets = {r.gene: r.lead_r for r in me.itertuples()
               if r.gene in null46 and isinstance(r.lead_r, str)}

    I = CM.load_inputs(gene_list=str(D / 'genes_59_stratified_20260923.txt'),
                       regions=str(RUN / 'regions.bed'))
    order, keep = I['order'], I['keep']
    N = len(order)
    import run_hapmixqtl_from_salmon as H
    with contextlib.redirect_stdout(io.StringIO()):
        A, T, Va, Vt, _ = H.compute_summaries_from_gibbs(
            I['YL'][:, keep], I['YR'][:, keep], yT=I['YT'][:, keep])
    gp = I['gp'].loc[I['genes']][['chr', 'pos']]
    gi = {g: i for i, g in enumerate(I['genes'])}
    vi = {v: i for i, v in enumerate(I['vdf'].index)}

    # observed allelic-channel p at the fixed variant, no permutation
    obs = {}
    for g, var in targets.items():
        if g not in gi or var not in vi:
            continue
        i, j = gi[g], vi[var]
        v1 = I['vdf'].iloc[[j]]
        one = lambda M: pd.DataFrame(M[[j]], index=v1.index, columns=order)
        mk = lambda M: pd.DataFrame(M[[i]], index=[g], columns=order)
        allT = pd.DataFrame(np.ones((1, N), bool), index=[g], columns=order)
        allF = pd.DataFrame(np.zeros((1, N), bool), index=[g], columns=order)
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                res = map_cis(one(I['dos']), v1[['chrom', 'pos']], mk(A), mk(T),
                              mk(Va), mk(Vt), gp.loc[[g]],
                              xL_df=one(I['xL']), xR_df=one(I['xR']),
                              window=1_000_000, nperm=10, covariates_df=I['cov_df'],
                              ase_covariates_df=None, tau_refit=True, verbose=False,
                              warn_monomorphic=False, beta_approx=False,
                              keep_a_df=allT, keep_t_df=allF)
            except (ValueError, RuntimeError):
                continue
        p = float(res['pval_nominal'].iloc[0])
        if np.isfinite(p):
            obs[g] = p
    print(f'observed allelic p computed for {len(obs)} genes; {n_draw} null draws\n')

    rows = []
    for g, p_obs in obs.items():
        r = dict(gene=g, stratum=strata.loc[g, 'stratum'], p_obs=p_obs)
        for sch in ('records', 'sign_flip'):
            pn = nl[(nl.scheme == sch) & (nl.gene == g)].pval.dropna().values
            if len(pn) < 10:
                r[f'p_emp_{sch}'] = np.nan
                continue
            r[f'p_emp_{sch}'] = (1 + int((pn <= p_obs).sum())) / (1 + len(pn))
            r[f'n_draw_{sch}'] = len(pn)
        rows.append(r)
    t = pd.DataFrame(rows).dropna(subset=['p_emp_records', 'p_emp_sign_flip'])
    out = D / 'perm_p_by_scheme_20260924'
    out.mkdir(exist_ok=True)
    t.to_csv(out / 'per_gene.tsv', sep='\t', index=False)

    print('empirical permutation p at the fixed variant, same observed data,')
    print('same genes, only the null construction differs:\n')
    for c, lab in (('p_emp_records', 'records (shipped)'),
                   ('p_emp_sign_flip', 'sign flip')):
        v = t[c]
        print(f'  {lab:18s} median {v.median():.3f}   '
              f'IQR {v.quantile(.25):.3f}-{v.quantile(.75):.3f}   '
              f'n<=0.05 {int((v <= 0.05).sum())}/{len(v)}   '
              f'n<=0.10 {int((v <= 0.10).sum())}/{len(v)}')

    d = t.p_emp_sign_flip - t.p_emp_records
    n_up = int((d > 0).sum()); n_dn = int((d < 0).sum())
    sgn = sps.binomtest(n_up, n_up + n_dn, 0.5).pvalue if n_up + n_dn else 1.0
    w = sps.wilcoxon(t.p_emp_sign_flip, t.p_emp_records) if n_up + n_dn > 5 else None
    print(f'\n  paired difference (sign flip minus records): median {d.median():+.3f}, '
          f'mean {d.mean():+.4f}')
    print(f'  sign flip larger on {n_up} genes, smaller on {n_dn}, tied on '
          f'{len(t) - n_up - n_dn};  sign p={sgn:.3g}'
          + (f';  Wilcoxon p={w.pvalue:.3g}' if w is not None else ''))

    print('\n  by stratum (median empirical p):')
    print(t.groupby('stratum')[['p_emp_records', 'p_emp_sign_flip']]
          .median().round(3).to_string())

    res = dict(n_genes=int(len(t)), n_draw=n_draw,
               median_records=float(t.p_emp_records.median()),
               median_sign_flip=float(t.p_emp_sign_flip.median()),
               n_called_05_records=int((t.p_emp_records <= 0.05).sum()),
               n_called_05_sign_flip=int((t.p_emp_sign_flip <= 0.05).sum()),
               median_diff=float(d.median()), mean_diff=float(d.mean()),
               n_signflip_larger=n_up, n_signflip_smaller=n_dn, sign_p=float(sgn))
    (out / 'summary.json').write_text(json.dumps(res, indent=2, default=float))
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()

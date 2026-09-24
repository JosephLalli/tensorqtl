"""Which of hapmixQTL's two channels carries the miscalibration?

The combined statistic is anticonservative under the null at a fixed variant,
and mixQTL -- which never sees a Gibbs draw -- is anticonservative by the same
amount, so the Gibbs weight shape is not the cause. What the two share and
RASQUAL does not includes a normal approximation to a continuous log-RATIO,
where RASQUAL models integer reads under a beta-binomial. If that approximation
is responsible, the ALLELIC channel should carry the miscalibration and the
TOTAL channel, which is a log total expression rather than a log ratio, should
be closer to uniform.

Each channel is isolated with the shipped count-cutoff masks: an all-False
keep_t_df zeroes the total channel's working inferential variance, which puts
every donor there in the state a zero-coverage donor is already in, and
symmetrically for keep_a_df. That is the supported mechanism rather than an ad
hoc edit of the variances.

p-values are hapmixQTL's OWN pval_nominal from map_cis, computed against its t
reference with the right dof -- not a chi2(1) conversion, which is the
dof -> infinity limit and anticonservative on its own.
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
SEED = 42


def one_gene(dos1, v1, a1, t1, va1, vt1, pos1, xl1, xr1, cov, keep_a, keep_t):
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            res = map_cis(dos1, v1[['chrom', 'pos']], a1, t1, va1, vt1, pos1,
                          xL_df=xl1, xR_df=xr1, window=1_000_000, nperm=10,
                          covariates_df=cov, ase_covariates_df=None,
                          tau_refit=True, verbose=False, warn_monomorphic=False,
                          beta_approx=False, keep_a_df=keep_a, keep_t_df=keep_t)
        except (ValueError, RuntimeError):
            return None
    p = float(res['pval_nominal'].iloc[0])
    sl, se = float(res['slope'].iloc[0]), float(res['slope_se'].iloc[0])
    return dict(pval=p, beta=sl, se=se)


def main():
    n_draw = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    me = pd.read_csv(RUN / 'matched_effects.tsv', sep='\t')
    null46 = {l.strip() for l in open(D / 'genes_null46_20260923.txt') if l.strip()}
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
    print(f'{len(targets)} genes, {n_draw} permutations, 3 configurations')

    rng = np.random.RandomState(SEED)
    perms = [rng.permutation(N) for _ in range(n_draw)]
    rows = []
    for pi, prm in enumerate(perms):
        cov = pd.DataFrame(I['cov_df'].values[prm], index=order,
                           columns=I['cov_df'].columns)
        for g, var in targets.items():
            if g not in gi or var not in vi:
                continue
            i, j = gi[g], vi[var]
            v1 = I['vdf'].iloc[[j]]
            mk = lambda M: pd.DataFrame(M[[i]][:, prm], index=[g], columns=order)
            one = lambda M: pd.DataFrame(M[[j]], index=v1.index, columns=order)
            a1, t1, va1, vt1 = mk(A), mk(T), mk(Va), mk(Vt)
            allT = pd.DataFrame(np.ones((1, N), bool), index=[g], columns=order)
            allF = pd.DataFrame(np.zeros((1, N), bool), index=[g], columns=order)
            for cfg, ka, kt in (('both', None, None),
                                ('allelic_only', allT, allF),
                                ('total_only', allF, allT)):
                r = one_gene(one(I['dos']), v1, a1, t1, va1, vt1, gp.loc[[g]],
                             one(I['xL']), one(I['xR']), cov, ka, kt)
                if r and np.isfinite(r['pval']):
                    rows.append(dict(cfg=cfg, gene=g, perm=pi, **r))
        print(f'  perm {pi + 1}/{n_draw}', flush=True)

    t = pd.DataFrame(rows)
    out = D / 'channel_split_20260924'
    out.mkdir(exist_ok=True)
    t.to_csv(out / 'null_by_channel.tsv', sep='\t', index=False)

    print('\nnominal p under the null at the fixed variant, hapmixQTL\'s own '
          'pval_nominal')
    print('uniform would be 10% per decile; 0.050 and 0.010 at those thresholds\n')
    res = {}
    edges = np.arange(0, 1.01, 0.1)
    for cfg, sub in t.groupby('cfg'):
        p = sub.pval.dropna().values
        cnt, _ = np.histogram(p, bins=edges)
        frac = cnt / cnt.sum()
        ks = sps.kstest(p, 'uniform')
        res[cfg] = dict(n=int(len(p)), frac=frac.round(4).tolist(),
                        frac_below_10=float((p < 0.10).mean()),
                        frac_below_05=float((p < 0.05).mean()),
                        frac_below_01=float((p < 0.01).mean()),
                        median_p=float(np.median(p)),
                        ks_stat=float(ks.statistic), ks_p=float(ks.pvalue))
        print(f'  {cfg:13s} n={len(p):4d}  1st decile {frac[0]*100:5.1f}%  '
              f'median {np.median(p):.3f}  p<0.05 {(p<0.05).mean():.3f}  '
              f'p<0.01 {(p<0.01).mean():.3f}  KS D={ks.statistic:.3f} p={ks.pvalue:.2g}')
    (out / 'summary.json').write_text(json.dumps(res, indent=2, default=float))
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()

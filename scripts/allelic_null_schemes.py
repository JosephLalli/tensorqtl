"""Two nulls for the allelic channel, and what the difference exposes.

The allelic regression is through the origin on s = xL - xR, with s in
{+1, -1, 0}. A donor contributes to the SLOPE only if it has allelic
information (v > 0) AND is heterozygous at the tested variant (s != 0).

RECORDS (the shipped scheme, and what every null in this session used):
permute donor records -- response, Gibbs variance and covariates move together
-- while genotypes and haplotypes stay. That is a valid relabelling null, but
v > 0 TRAVELS with the phenotype while s != 0 STAYS with the genotype. The two
are correlated in the real data and are decoupled by the permutation, so the
number of donors actually informing the slope fluctuates from draw to draw.

SIGN FLIP (Rademacher on the haplotype labels): give each donor an independent
random +/- 1 and swap its xL and xR when it is -1, so s_i -> -s_i. Under the
null this destroys the association just as thoroughly, but it holds FIXED
everything the records scheme disturbs: each donor keeps its own response, its
own v, its own heterozygosity, and the effective sample size is identical in
every draw. The total channel is untouched because its design is the dosage
xL + xR, which a swap leaves alone.

If the two nulls agree, the records scheme is not implicated and the heavy tails
measured on the residuals stand as the explanation. If the sign-flip null is
uniform where records is not, part of the miscalibration is an artifact of the
scheme rather than of the model.

Measurement only. Nothing here changes the variance model or the shipped scheme.
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
EPS = 1e-12


def run_one(dos1, v1, a1, t1, va1, vt1, pos1, xl1, xr1, cov, keep_a, keep_t):
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            res = map_cis(dos1, v1[['chrom', 'pos']], a1, t1, va1, vt1, pos1,
                          xL_df=xl1, xR_df=xr1, window=1_000_000, nperm=10,
                          covariates_df=cov, ase_covariates_df=None,
                          tau_refit=True, verbose=False, warn_monomorphic=False,
                          beta_approx=False, keep_a_df=keep_a, keep_t_df=keep_t)
        except (ValueError, RuntimeError):
            return None
    return float(res['pval_nominal'].iloc[0])


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

    rng = np.random.RandomState(SEED)
    perms = [rng.permutation(N) for _ in range(n_draw)]
    flips = [rng.choice([1, -1], size=N) for _ in range(n_draw)]

    allT = lambda g: pd.DataFrame(np.ones((1, N), bool), index=[g], columns=order)
    allF = lambda g: pd.DataFrame(np.zeros((1, N), bool), index=[g], columns=order)

    rows, neff = [], []
    for p_i in range(n_draw):
        prm, fl = perms[p_i], flips[p_i]
        cov_p = pd.DataFrame(I['cov_df'].values[prm], index=order,
                             columns=I['cov_df'].columns)
        cov_0 = I['cov_df']
        for g, var in targets.items():
            if g not in gi or var not in vi:
                continue
            i, j = gi[g], vi[var]
            v1 = I['vdf'].iloc[[j]]
            one = lambda M: pd.DataFrame(M[[j]], index=v1.index, columns=order)
            xl, xr = I['xL'][[j]].astype(float), I['xR'][[j]].astype(float)
            s = (xl - xr).ravel()

            # effective allelic n: informative AND heterozygous at this variant
            inf0 = Va[i] > EPS
            neff.append(dict(gene=g, perm=p_i,
                             n_records=int(((Va[i][prm] > EPS) & (s != 0)).sum()),
                             n_signflip=int((inf0 & (s != 0)).sum())))

            # --- records null: phenotype bundle moves, haplotypes stay --------
            mkp = lambda M: pd.DataFrame(M[[i]][:, prm], index=[g], columns=order)
            p_rec = run_one(one(I['dos']), v1, mkp(A), mkp(T), mkp(Va), mkp(Vt),
                            gp.loc[[g]], one(I['xL']), one(I['xR']), cov_p,
                            allT(g), allF(g))
            # --- sign-flip null: haplotype labels swapped, everything else fixed
            xl_f = np.where(fl[None, :] == 1, xl, xr)
            xr_f = np.where(fl[None, :] == 1, xr, xl)
            mk0 = lambda M: pd.DataFrame(M[[i]], index=[g], columns=order)
            p_flip = run_one(one(I['dos']), v1, mk0(A), mk0(T), mk0(Va), mk0(Vt),
                             gp.loc[[g]],
                             pd.DataFrame(xl_f, index=v1.index, columns=order),
                             pd.DataFrame(xr_f, index=v1.index, columns=order),
                             cov_0, allT(g), allF(g))
            if p_rec is not None and np.isfinite(p_rec):
                rows.append(dict(scheme='records', gene=g, perm=p_i, pval=p_rec))
            if p_flip is not None and np.isfinite(p_flip):
                rows.append(dict(scheme='sign_flip', gene=g, perm=p_i, pval=p_flip))
        print(f'  draw {p_i + 1}/{n_draw}', flush=True)

    t = pd.DataFrame(rows)
    ne = pd.DataFrame(neff)
    out = D / 'allelic_null_schemes_20260924'
    out.mkdir(exist_ok=True)
    t.to_csv(out / 'pvals.tsv', sep='\t', index=False)
    ne.to_csv(out / 'effective_n.tsv', sep='\t', index=False)

    print('\neffective allelic n (donors with v>0 AND s!=0), the quantity the')
    print('records scheme disturbs and the sign flip holds fixed:')
    print(f'  records   median {ne.n_records.median():5.1f}   '
          f'IQR {ne.n_records.quantile(.25):.0f}-{ne.n_records.quantile(.75):.0f}   '
          f'min {ne.n_records.min()}')
    print(f'  sign flip median {ne.n_signflip.median():5.1f}   '
          f'IQR {ne.n_signflip.quantile(.25):.0f}-{ne.n_signflip.quantile(.75):.0f}   '
          f'min {ne.n_signflip.min()}')
    d = ne.n_signflip - ne.n_records
    print(f'  per gene-draw difference (observed minus permuted): median {d.median():+.1f}, '
          f'mean {d.mean():+.2f}')

    print('\nallelic-channel nominal p under each null:')
    res = {}
    edges = np.arange(0, 1.01, 0.1)
    for sch, sub in t.groupby('scheme'):
        p = sub.pval.dropna().values
        cnt, _ = np.histogram(p, bins=edges)
        fr = cnt / cnt.sum()
        ks = sps.kstest(p, 'uniform')
        res[sch] = dict(n=int(len(p)), frac=fr.round(4).tolist(),
                        frac_below_10=float((p < .10).mean()),
                        frac_below_05=float((p < .05).mean()),
                        frac_below_01=float((p < .01).mean()),
                        median_p=float(np.median(p)),
                        ks_stat=float(ks.statistic), ks_p=float(ks.pvalue))
        print(f'  {sch:10s} n={len(p):4d}  1st decile {fr[0]*100:5.1f}%  '
              f'median {np.median(p):.3f}  p<0.05 {(p<.05).mean():.3f}  '
              f'p<0.01 {(p<.01).mean():.3f}  KS D={ks.statistic:.3f} p={ks.pvalue:.2g}')

    m = t.pivot_table(index=['gene', 'perm'], columns='scheme', values='pval').dropna()
    for thr in (0.05, 0.10):
        Aa, Bb = m['records'] < thr, m['sign_flip'] < thr
        n01, n10 = int((Aa & ~Bb).sum()), int((~Aa & Bb).sum())
        pp = sps.binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
        print(f'  paired p<{thr}: records {Aa.mean():.3f} vs sign_flip {Bb.mean():.3f}'
              f'   discordant {n01}/{n10}  McNemar p={pp:.3g}')
        res[f'paired_{thr}'] = dict(records=float(Aa.mean()), sign_flip=float(Bb.mean()),
                                    n01=n01, n10=n10, mcnemar_p=float(pp))
    (out / 'summary.json').write_text(json.dumps(res, indent=2, default=float))
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()

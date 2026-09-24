"""Does the allelic sign symmetry change the COMBINED statistic's calibration?

Both hapmixQTL and mixQTL combine a total channel and an allelic channel. The
allelic design is s = xL - xR; the total design is the dosage xL + xR.

WHY NOT SIGN FLIP ALONE HERE. Swapping a donor's haplotype labels flips s and
leaves the dosage untouched, so the total channel keeps its real
genotype-phenotype association. That is a valid null for the allelic channel in
isolation -- which is how it was used -- but NOT for a combined statistic, where
real total-channel signal would survive and the small p-values it produced would
not be false positives.

So the combined statistic is tested under two nulls that are BOTH valid under
H0:

  records            permute donor records: response, Gibbs variance and
                     covariates move together, genotypes stay. The shipped
                     scheme. Nulls both channels, but decouples "has allelic
                     information" (travels with the phenotype) from "is
                     heterozygous here" (stays with the genotype).

  records+signflip   the same, plus an independent Rademacher swap of each
                     donor's haplotype labels. Nulls both channels AND respects
                     the allelic channel's label symmetry.

Under H0 a correct parametric p is uniform under EITHER. A difference between
them means the parametric p is wrong, with the second acting as the more
sensitive probe.

Run for hapmixQTL (map_cis, both channels) and mixQTL (mixqtl_scan, meta), the
two methods that combine channels this way.
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
import tensorqtl.mixqtl_replication as MX      # noqa: E402
from tensorqtl.hapmixqtl import map_cis        # noqa: E402

D = Path('/mnt/ssd/lalli/brainvar_hapmix_deploy')
RUN = D / 'rasqual_default_mode_20260923'
SEED = 42
LN2 = np.log(2.0)


def hapmix_p(dos1, v1, a1, t1, va1, vt1, pos1, xl1, xr1, cov):
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            res = map_cis(dos1, v1[['chrom', 'pos']], a1, t1, va1, vt1, pos1,
                          xL_df=xl1, xR_df=xr1, window=1_000_000, nperm=10,
                          covariates_df=cov, ase_covariates_df=None,
                          tau_refit=True, verbose=False, warn_monomorphic=False,
                          beta_approx=False)
        except (ValueError, RuntimeError):
            return None
    p = float(res['pval_nominal'].iloc[0])
    return p if np.isfinite(p) else None


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
    Y1, Y2, YT = MX.summaries_from_gibbs_posterior_mean(I['YL'], I['YR'], I['YT'])
    Y1, Y2, YT = Y1[:, keep], Y2[:, keep], YT[:, keep]
    gp = I['gp'].loc[I['genes']][['chr', 'pos']]
    gi = {g: i for i, g in enumerate(I['genes'])}
    vi = {v: i for i, v in enumerate(I['vdf'].index)}
    print(f'{len(targets)} genes, {n_draw} draws, 2 nulls x 2 methods')

    rng = np.random.RandomState(SEED)
    perms = [rng.permutation(N) for _ in range(n_draw)]
    flips = [rng.choice([1, -1], size=N) for _ in range(n_draw)]

    rows = []
    for p_i in range(n_draw):
        prm, fl = perms[p_i], flips[p_i]
        cov_p = pd.DataFrame(I['cov_df'].values[prm], index=order,
                             columns=I['cov_df'].columns)
        for g, var in targets.items():
            if g not in gi or var not in vi:
                continue
            i, j = gi[g], vi[var]
            v1 = I['vdf'].iloc[[j]]
            one = lambda M: pd.DataFrame(M[[j]], index=v1.index, columns=order)
            mkp = lambda M: pd.DataFrame(M[[i]][:, prm], index=[g], columns=order)
            xl, xr = I['xL'][[j]].astype(float), I['xR'][[j]].astype(float)
            xl_f = np.where(fl[None, :] == 1, xl, xr)
            xr_f = np.where(fl[None, :] == 1, xr, xl)
            XLF = pd.DataFrame(xl_f, index=v1.index, columns=order)
            XRF = pd.DataFrame(xr_f, index=v1.index, columns=order)

            for sch, xL_, xR_ in (('records', one(I['xL']), one(I['xR'])),
                                  ('records+signflip', XLF, XRF)):
                p = hapmix_p(one(I['dos']), v1, mkp(A), mkp(T), mkp(Va), mkp(Vt),
                             gp.loc[[g]], xL_, xR_, cov_p)
                if p is not None:
                    rows.append(dict(method='hapmixQTL', scheme=sch, gene=g,
                                     perm=p_i, pval=p))

            # mixQTL: h1/h2 are the haplotype dosages; swapping them flips
            # Xasc = h1 - h2 and leaves Xtrc = (h1 + h2)/2 alone, exactly as
            # for hapmixQTL
            vsel = CM.gene_variant_index(I, g)
            if vsel.size:
                vv = I['vdf'].iloc[I['idx']].iloc[vsel]
                ids = list(map(str, vv.index))
                if var in ids:
                    k = ids.index(var)
                    h1 = I['xL'][I['idx']][:, keep][vsel].T.astype(float)[:, [k]]
                    h2 = I['xR'][I['idx']][:, keep][vsel].T.astype(float)[:, [k]]
                    h1f = np.where(fl[:, None] == 1, h1, h2)
                    h2f = np.where(fl[:, None] == 1, h2, h1)
                    for sch, H1, H2 in (('records', h1, h2),
                                        ('records+signflip', h1f, h2f)):
                        o = MX.mixqtl_scan(Y1[i][prm], Y2[i][prm], YT[i][prm],
                                           I['lib_size'][prm], H1, H2,
                                           covariates=I['cov_df'].values[prm],
                                           **MX.PACKAGE_DEFAULT_CUTOFFS)
                        pv = o['meta']['pval'][0]
                        if np.isfinite(pv):
                            rows.append(dict(method='mixQTL', scheme=sch, gene=g,
                                             perm=p_i, pval=float(pv)))
        print(f'  draw {p_i + 1}/{n_draw}', flush=True)

    t = pd.DataFrame(rows)
    out = D / 'combined_null_schemes_20260924'
    out.mkdir(exist_ok=True)
    t.to_csv(out / 'pvals.tsv', sep='\t', index=False)

    print('\ncombined statistic, nominal p under two valid nulls:\n')
    res = {}
    edges = np.arange(0, 1.01, 0.1)
    for (meth, sch), sub in t.groupby(['method', 'scheme']):
        p = sub.pval.dropna().values
        cnt, _ = np.histogram(p, bins=edges)
        fr = cnt / cnt.sum()
        ks = sps.kstest(p, 'uniform')
        res[f'{meth}|{sch}'] = dict(n=int(len(p)), frac=fr.round(4).tolist(),
                                    frac_below_05=float((p < .05).mean()),
                                    frac_below_01=float((p < .01).mean()),
                                    median_p=float(np.median(p)),
                                    ks_stat=float(ks.statistic),
                                    ks_p=float(ks.pvalue))
        print(f'  {meth:10s} {sch:17s} n={len(p):4d}  1st decile {fr[0]*100:5.1f}%  '
              f'median {np.median(p):.3f}  p<0.05 {(p<.05).mean():.3f}  '
              f'p<0.01 {(p<.01).mean():.3f}  KS p={ks.pvalue:.2g}')

    print()
    for meth in ('hapmixQTL', 'mixQTL'):
        sub = t[t.method == meth]
        m = sub.pivot_table(index=['gene', 'perm'], columns='scheme',
                            values='pval').dropna()
        if not len(m):
            continue
        for thr in (0.05,):
            Aa, Bb = m['records'] < thr, m['records+signflip'] < thr
            n01, n10 = int((Aa & ~Bb).sum()), int((~Aa & Bb).sum())
            pp = sps.binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
            print(f'  {meth}: paired p<{thr}  records {Aa.mean():.3f} vs '
                  f'records+signflip {Bb.mean():.3f}  discordant {n01}/{n10}  '
                  f'McNemar p={pp:.3g}')
            res[f'{meth}|paired'] = dict(records=float(Aa.mean()),
                                         both=float(Bb.mean()), n01=n01, n10=n10,
                                         mcnemar_p=float(pp))
    (out / 'summary.json').write_text(json.dumps(res, indent=2, default=float))
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()

"""Realized estimation error under the null, at a FIXED variant, three arms.

This is the measure a reported standard error cannot flatter. Under a null
permutation the true slope is zero, so the spread of beta_hat across
permutations IS the estimator's error -- no ground truth and no trust in any
arm's own variance calculation is required. A method that understates its
standard error looks precise on the reported scale and is caught here.

DESIGN.

  FIXED VARIANT, pre-specified: each gene's RASQUAL observed lead. Chosen
  because it is fixed before any permutation, is identical for all three arms,
  and was selected by NEITHER hapmixQTL nor mixQTL. Under permutation the true
  effect is zero at every variant, so the choice cannot favour an arm; it only
  has to be the same one.

  NOT PAIRED ACROSS ARMS, and it does not need to be. RASQUAL draws its own
  permutation internally (-r, seeded from time and pid), so its draws cannot be
  matched to hapmixQTL's. Each arm's spread is its own error under its own valid
  null; what is compared is the spread, not draw-by-draw values.

  hapmixQTL and mixQTL are permuted by relabelling donor RECORDS -- phenotype,
  variances and covariates move together while genotypes stay -- which is
  hapmixQTL's shipped perm_scheme='records'.

Restricted to the 46 genes carrying RASQUAL null rows, since the null was run on
that subset.
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

import compare_pipelines as CP            # noqa: E402
import compare_mixqtl_replication as CM   # noqa: E402
import tensorqtl.mixqtl_replication as MX  # noqa: E402

D = Path('/mnt/ssd/lalli/brainvar_hapmix_deploy')
RUN = D / 'rasqual_default_mode_20260923'
SEED = 42
LN2 = np.log(2.0)


def rasqual_null_betas(gene, variant, n_draw):
    """RASQUAL's log aFC at `variant` across its null draws, from the retained
    per-variant rows. Returns [] if the gene has no rows."""
    chrom, pos = variant.split('_')[0], variant.split('_')[1]
    out = []
    for d in range(n_draw):
        f = RUN / 'rasqual_rows' / f'null_{d:03d}' / f'{gene}.tsv'
        if not f.exists():
            continue
        for ln in f.read_text().strip().split('\n'):
            x = ln.split('\t')
            if len(x) < 25 or x[2] != chrom or x[3] != pos:
                continue
            try:
                if int(float(x[22])) != 0:
                    break
                pi = min(max(float(x[11]), 1e-6), 1 - 1e-6)
                out.append((np.log(pi / (1 - pi)), float(x[10]), d))
            except (ValueError, IndexError):
                pass
            break
    return out


def main():
    n_draw = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    me = pd.read_csv(RUN / 'matched_effects.tsv', sep='\t')
    null46 = {l.strip() for l in open(D / 'genes_null46_20260923.txt') if l.strip()}
    strata = pd.read_csv(D / 'genes_59_strata_20260923.tsv', sep='\t').set_index('gene')
    targets = {r.gene: r.lead_r for r in me.itertuples()
               if r.gene in null46 and isinstance(r.lead_r, str)}
    print(f'{len(targets)} genes with a RASQUAL lead and null rows; {n_draw} draws')

    I = CM.load_inputs(gene_list=str(D / 'genes_59_stratified_20260923.txt'),
                       regions=str(RUN / 'regions.bed'))
    order, keep = I['order'], I['keep']
    N = len(order)
    Y1, Y2, YT = MX.summaries_from_gibbs_posterior_mean(I['YL'], I['YR'], I['YT'])
    Y1, Y2, YT = Y1[:, keep], Y2[:, keep], YT[:, keep]

    # hapmixQTL needs its own A/T/Va/Vt; build them from the same draws
    import run_hapmixqtl_from_salmon as H
    with contextlib.redirect_stdout(io.StringIO()):
        A, T, Va, Vt, _ = H.compute_summaries_from_gibbs(
            I['YL'][:, keep], I['YR'][:, keep], yT=I['YT'][:, keep])
    gp = I['gp'].loc[I['genes']][['chr', 'pos']]
    gi = {g: j for j, g in enumerate(I['genes'])}

    rng = np.random.RandomState(SEED)
    perms = [rng.permutation(N) for _ in range(n_draw)]

    rec = {g: dict(hm=[], mx=[]) for g in targets}
    long = []          # one row per (arm, gene, permutation): beta and p
    for p, prm in enumerate(perms):
        # relabelling, not reindexing: the VALUES move together across donors
        # while the sample labels stay put, which is what perm_scheme='records'
        # means and what keeps the phenotype/covariate frames aligned
        cov_perm = pd.DataFrame(I['cov_df'].values[prm], index=order,
                                columns=I['cov_df'].columns)
        hm = CP.hapmix_at(A[:, prm], T[:, prm], Va[:, prm], Vt[:, prm],
                          I['genes'], order, I['vdf'], I['dos'], I['xL'], I['xR'],
                          gp, 1_000_000, targets,
                          cov_df=cov_perm, ase_cov='none')
        for r in hm.itertuples():
            if r.gene in rec and np.isfinite(r.log_afc):
                rec[r.gene]['hm'].append(float(r.log_afc))
                # hapmixQTL's stat is T^2, so the nominal p is chi2(1)'s upper
                # tail -- the same conversion the driver uses for its own scale
                long.append(dict(arm='hapmixQTL', gene=r.gene, perm=p,
                                 beta=float(r.log_afc), stat=float(r.stat),
                                 pval=float(sps.chi2.sf(float(r.stat), 1))))
        for g, var in targets.items():
            j = gi[g]
            vsel = CM.gene_variant_index(I, g)
            if vsel.size == 0:
                continue
            v = I['vdf'].iloc[I['idx']].iloc[vsel]
            ids = list(map(str, v.index))
            if var not in ids:
                continue
            k = ids.index(var)
            h1 = I['xL'][I['idx']][:, keep][vsel].T.astype(float)[:, [k]]
            h2 = I['xR'][I['idx']][:, keep][vsel].T.astype(float)[:, [k]]
            o = MX.mixqtl_scan(Y1[j][prm], Y2[j][prm], YT[j][prm],
                               I['lib_size'][prm], h1, h2,
                               covariates=I['cov_df'].values[prm],
                               **MX.PACKAGE_DEFAULT_CUTOFFS)
            b = o['meta']['beta'][0]
            if np.isfinite(b):
                rec[g]['mx'].append(float(b) * LN2)
                long.append(dict(arm='mixQTL', gene=g, perm=p,
                                 beta=float(b) * LN2,
                                 stat=float(o['meta']['stat'][0]),
                                 pval=float(o['meta']['pval'][0])))
        print(f'  perm {p + 1}/{n_draw}', flush=True)

    rows = []
    for g, var in targets.items():
        rq_full = rasqual_null_betas(g, var, n_draw)
        for b_, chi_, d_ in rq_full:
            long.append(dict(arm='RASQUAL', gene=g, perm=d_, beta=b_, stat=chi_,
                             pval=float(sps.chi2.sf(chi_, 1))))
        rq = [b_ for b_, _, _ in rq_full]
        h, m = rec[g]['hm'], rec[g]['mx']
        rows.append(dict(gene=g, stratum=strata.loc[g, 'stratum'], variant=var,
                         n_hm=len(h), n_mx=len(m), n_rq=len(rq),
                         sd_hm=np.std(h, ddof=1) if len(h) > 2 else np.nan,
                         sd_mx=np.std(m, ddof=1) if len(m) > 2 else np.nan,
                         sd_rq=np.std(rq, ddof=1) if len(rq) > 2 else np.nan))
    t = pd.DataFrame(rows)
    out = D / 'realized_variance_20260924'
    out.mkdir(exist_ok=True)
    t.to_csv(out / 'null_beta_sd.tsv', sep='\t', index=False)
    pd.DataFrame(long).to_csv(out / 'null_long.tsv', sep='\t', index=False)

    print('\nrealized sd of beta_hat under the null, at the same fixed variant')
    print('(smaller = less estimation error; this is what a reported SE cannot fake)')
    res = {'n_draw': n_draw}
    for a, b in (('mx', 'hm'), ('rq', 'hm'), ('rq', 'mx')):
        m = t[f'sd_{a}'].notna() & t[f'sd_{b}'].notna()
        if m.sum() < 5:
            print(f'  {a}/{b}: n={int(m.sum())} too few'); continue
        r = (t.loc[m, f'sd_{a}'] / t.loc[m, f'sd_{b}']).values
        wins = int((r > 1).sum())
        p = sps.binomtest(wins, len(r), 0.5).pvalue
        print(f'  sd_{a}/sd_{b}: n={len(r):3d}  median {np.median(r):5.3f}   '
              f'{b} tighter on {wins}/{len(r)}   sign p={p:.3g}')
        res[f'sd_{a}_over_sd_{b}'] = dict(n=len(r), median=float(np.median(r)),
                                          n_b_tighter=wins, sign_p=float(p))
    print('\nby stratum (sd_mx/sd_hm):')
    for s in ['HIGH', 'MID', 'LOW']:
        sub = t[(t.stratum == s) & t.sd_mx.notna() & t.sd_hm.notna()]
        if len(sub) >= 4:
            r = (sub.sd_mx / sub.sd_hm).values
            print(f'  {s:4s} n={len(r):2d}  median {np.median(r):5.3f}  '
                  f'hapmixQTL tighter on {int((r > 1).sum())}/{len(r)}')
    (out / 'summary.json').write_text(json.dumps(res, indent=2, default=float))
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()

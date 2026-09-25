"""The shared null instrument for the nominal-p anticonservatism: 2,000 donor-
record permutations per gene at a fixed variant, each channel reported apart.

WHY 2,000. The 2026-09-24 calibration rested on 30 permutations x 46 genes.
That pools well (1,380 p-values) but says almost nothing per gene: a per-gene
rejection rate from 30 draws has a binomial standard error of 0.04 at 0.05, so
no per-gene mechanism can be tested against it. The fit is a few matrix
products on 92 donors, so 2,000 permutations per gene costs a minute or two.

WHAT IS PERMUTED. Exactly what the 2026-09-24 null permuted: each donor's
RECORD -- allelic log ratio a, total log expression t, both Gibbs variances,
and the covariate row -- moves as one unit, while the genotype at the fixed
variant (the phased difference s = xL - xR for the allelic channel, dosage/2 for
the total channel) stays with the donor position. This is hapmixQTL's shipped
perm_scheme='records'. The fixed variant is each gene's RASQUAL observed lead,
chosen before any permutation and by neither Salmon-based method.

THE FIT is `se_fixes.fit_all`'s baseline algebra, verified there against
map_cis, re-expressed to return each channel's slope, standard error and
degrees of freedom separately:

    allelic   weighted least squares through the origin, weights 1/v_a,
              residual scale sigma_a^2 fitted per variant, dof = n_a - 1
    total     weighted least squares with an intercept and the 17 covariates
              partialled out in the weighted space, weights 1/v_t,
              dof = n_t - 1 - 18
    combined  inverse-variance meta-analysis of the two slopes, t^2 referred
              to F(1, min(dof_a, dof_t))

TWO REPRODUCTION GATES run before anything is written:

  1. against map_cis on the OBSERVED data at 12 genes, as in se_fixes: the
     combined |t| must agree to 5%, otherwise nothing here describes the
     shipped statistic
  2. the first 30 permutations use the same RandomState(42) stream as
     se_fixes.py and channel_split_calibration.py, so the combined p here must
     equal se_fixes_20260924/pvals.tsv's baseline p for every (gene, perm)

Per-channel p-values are the channel's own t^2 against F(1, its own dof).

OUTPUTS, in brainvar_hapmix_deploy/nominal_p_null_instrument_20260925/:

  inputs_at_lead.npz   the exact arrays every downstream analysis must use:
                       per target gene (rows) x donor (columns) a, va, t, vt,
                       s, g; the covariate matrix; gene, variant and stratum
                       labels; and the full A/T/Va/Vt for all 59 genes
  null_long.tsv.gz     one row per (gene, perm): per-channel slope, se, dof,
                       t^2 and p, and the combined ones
  observed.tsv         the same quantities on the unpermuted data
  per_gene.tsv         per gene and channel: rejection rate at 0.05 and 0.01,
                       sd(beta)/rms(se), and the 95th/99th percentile of t^2
                       over the F(1, dof) quantile
  summary.json         pooled rates with gene-clustered bootstrap intervals

Master seed 42. The permutation stream is RandomState(42) (to reproduce the
2026-09-24 draws); the gene bootstrap uses a child stream spawned from
SeedSequence(42), so the two never share draws.
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
from se_fixes import fit_all, resid_out        # noqa: E402

D = Path('/mnt/ssd/lalli/brainvar_hapmix_deploy')
RUN = D / 'rasqual_default_mode_20260923'
OUT = D / 'nominal_p_null_instrument_20260925'
SEED, EPS = 42, 1e-12
N_PERM = 2000
N_BOOT = 2000
ALPHAS = (0.10, 0.05, 0.01, 0.001)


def fit_channels(a, s, va, t, g, vt, C):
    """fit_all's baseline, per channel. Same admission rules, same algebra."""
    ka = np.isfinite(a) & np.isfinite(va) & (va > EPS) & np.isfinite(s)
    kt = np.isfinite(t) & np.isfinite(vt) & (vt > EPS) & np.isfinite(g)
    if ka.sum() < 5 or kt.sum() < 10:
        return None
    wa = 1.0 / va[ka]
    ya, xa = a[ka] * np.sqrt(wa), s[ka] * np.sqrt(wa)
    xxa = float(xa @ xa)
    if xxa <= 0:
        return None
    ba = float(xa @ ya) / xxa
    ea = ya - ba * xa
    dofa = max(int(ka.sum()) - 1, 1)
    sea2 = float(ea @ ea) / dofa / xxa
    wt = 1.0 / vt[kt]
    Z = np.column_stack([np.ones(kt.sum()), C[kt]])
    yt = resid_out(t[kt][:, None], Z, wt).ravel()
    xt = resid_out(g[kt][:, None], Z, wt).ravel()
    xxt = float(xt @ xt)
    if xxt <= 0:
        return None
    bt = float(xt @ yt) / xxt
    et = yt - bt * xt
    doft = max(int(kt.sum()) - 1 - Z.shape[1], 1)
    set2 = float(et @ et) / doft / xxt
    if not (np.isfinite(sea2) and np.isfinite(set2)) or sea2 <= 0 or set2 <= 0:
        return None
    prec = 1 / sea2 + 1 / set2
    b = (ba / sea2 + bt / set2) / prec
    se2 = 1 / prec
    dof = min(dofa, doft)
    return dict(ba=ba, sea=np.sqrt(sea2), dofa=dofa, n_a=int(ka.sum()),
                bt=bt, set=np.sqrt(set2), doft=doft, n_t=int(kt.sum()),
                b=b, se=np.sqrt(se2), dof=dof,
                t2_a=ba ** 2 / sea2, t2_t=bt ** 2 / set2, t2_b=b ** 2 / se2)


def add_p(df):
    df['p_a'] = sps.f.sf(df.t2_a, 1, df.dofa)
    df['p_t'] = sps.f.sf(df.t2_t, 1, df.doft)
    df['p_b'] = sps.f.sf(df.t2_b, 1, df.dof)
    return df


def map_cis_gate(I, A, T, Va, Vt, targets, gi, vi):
    from tensorqtl.hapmixqtl import map_cis
    gp = I['gp'].loc[I['genes']][['chr', 'pos']]
    order, C = I['order'], I['cov_df'].values
    diffs = []
    for g, var in list(targets.items())[:12]:
        i, j = gi[g], vi[var]
        v1 = I['vdf'].iloc[[j]]
        one = lambda M: pd.DataFrame(M[[j]], index=v1.index, columns=order)
        mk = lambda M: pd.DataFrame(M[[i]], index=[g], columns=order)
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                res = map_cis(one(I['dos']), v1[['chrom', 'pos']], mk(A), mk(T),
                              mk(Va), mk(Vt), gp.loc[[g]], xL_df=one(I['xL']),
                              xR_df=one(I['xR']), window=1_000_000, nperm=10,
                              covariates_df=I['cov_df'], ase_covariates_df=None,
                              tau_refit=False, verbose=False,
                              warn_monomorphic=False, beta_approx=False)
            except (ValueError, RuntimeError):
                continue
        sl, se_ = float(res['slope'].iloc[0]), float(res['slope_se'].iloc[0])
        f = fit_channels(A[i], (I['xL'][j] - I['xR'][j]).astype(float), Va[i],
                         T[i], I['dos'][j].astype(float) / 2.0, Vt[i], C)
        if f and se_ > 0:
            diffs.append(abs(np.sqrt(f['t2_b']) - abs(sl / se_)) / abs(sl / se_))
    return diffs


def rate_ci(df, pcol, alpha, rng):
    """Pooled rejection rate with a gene-clustered percentile bootstrap."""
    per = df.groupby('gene')[pcol].agg(lambda p: [(p < alpha).sum(), len(p)])
    k = np.array([x[0] for x in per]); n = np.array([x[1] for x in per])
    est = k.sum() / n.sum()
    idx = rng.integers(0, len(k), size=(N_BOOT, len(k)))
    boot = k[idx].sum(1) / n[idx].sum(1)
    return est, float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def main():
    OUT.mkdir(exist_ok=True)
    me = pd.read_csv(RUN / 'matched_effects.tsv', sep='\t')
    null46 = {l.strip() for l in open(D / 'genes_null46_20260923.txt') if l.strip()}
    targets = {r.gene: r.lead_r for r in me.itertuples()
               if r.gene in null46 and isinstance(r.lead_r, str)}
    strata = pd.read_csv(D / 'genes_59_strata_20260923.tsv', sep='\t',
                         index_col=0)
    I = CM.load_inputs(gene_list=str(D / 'genes_59_stratified_20260923.txt'),
                       regions=str(RUN / 'regions.bed'))
    order, keep = I['order'], I['keep']
    N = len(order)
    import run_hapmixqtl_from_salmon as H
    with contextlib.redirect_stdout(io.StringIO()):
        A, T, Va, Vt, _ = H.compute_summaries_from_gibbs(
            I['YL'][:, keep], I['YR'][:, keep], yT=I['YT'][:, keep])
    C = I['cov_df'].values
    gi = {g: i for i, g in enumerate(I['genes'])}
    vi = {v: i for i, v in enumerate(I['vdf'].index)}
    tg = [g for g in targets if g in gi and targets[g] in vi]
    print(f'{len(tg)} target genes of {len(targets)}; {N} donors; '
          f'{N_PERM} permutations per gene', flush=True)

    # ---- gate 1: map_cis on the observed data ---------------------------
    diffs = map_cis_gate(I, A, T, Va, Vt, {g: targets[g] for g in tg}, gi, vi)
    print(f'gate 1, |t| vs map_cis over {len(diffs)} genes: median '
          f'{np.median(diffs):.2e}, max {np.max(diffs):.2e}')
    if not diffs or np.max(diffs) > 0.05:
        raise SystemExit('gate 1 FAILED: fit does not reproduce map_cis')

    # ---- the arrays every downstream analysis uses ----------------------
    rows_i = [gi[g] for g in tg]
    rows_j = [vi[targets[g]] for g in tg]
    s_lead = np.stack([(I['xL'][j] - I['xR'][j]).astype(float) for j in rows_j])
    g_lead = np.stack([I['dos'][j].astype(float) / 2.0 for j in rows_j])
    np.savez_compressed(
        OUT / 'inputs_at_lead.npz',
        genes=np.array(tg), variants=np.array([targets[g] for g in tg]),
        strata=np.array([strata.loc[g, 'stratum'] for g in tg]),
        donors=np.array(order), cov_names=np.array(I['cov_df'].columns),
        a=A[rows_i], va=Va[rows_i], t=T[rows_i], vt=Vt[rows_i],
        s=s_lead, g=g_lead, C=C, lib_size=I['lib_size'],
        all_genes=np.array(I['genes']), A_all=A, T_all=T, Va_all=Va, Vt_all=Vt)

    # ---- observed fit ----------------------------------------------------
    obs = []
    for k, g in enumerate(tg):
        i = rows_i[k]
        f = fit_channels(A[i], s_lead[k], Va[i], T[i], g_lead[k], Vt[i], C)
        if f:
            obs.append(dict(gene=g, variant=targets[g],
                            stratum=strata.loc[g, 'stratum'], **f))
    obs = add_p(pd.DataFrame(obs))
    obs.to_csv(OUT / 'observed.tsv', sep='\t', index=False)

    # ---- permutations: the RandomState(42) stream of the 2026-09-24 null --
    rng = np.random.RandomState(SEED)
    rows = []
    for p_i in range(N_PERM):
        prm = rng.permutation(N)
        Cp = C[prm]
        for k, g in enumerate(tg):
            i = rows_i[k]
            f = fit_channels(A[i][prm], s_lead[k], Va[i][prm], T[i][prm],
                             g_lead[k], Vt[i][prm], Cp)
            if f:
                rows.append(dict(gene=g, perm=p_i, **f))
        if (p_i + 1) % 250 == 0:
            print(f'  permutation {p_i + 1}/{N_PERM}', flush=True)
    t = add_p(pd.DataFrame(rows))

    # ---- gate 2: the first 30 permutations reproduce se_fixes ------------
    ref = pd.read_csv(D / 'se_fixes_20260924' / 'pvals.tsv', sep='\t')
    ref = ref[ref.arm == 'baseline'].set_index(['gene', 'perm']).pval
    mine = t[t.perm < 30].set_index(['gene', 'perm']).p_b
    common = ref.index.intersection(mine.index)
    dmax = float(np.max(np.abs(ref.loc[common].values - mine.loc[common].values)))
    print(f'gate 2, first 30 permutations vs se_fixes baseline: '
          f'{len(common)}/{len(ref)} (gene, perm) matched, max |dp| {dmax:.2e}')
    if len(common) != len(ref) or dmax > 1e-9:
        raise SystemExit('gate 2 FAILED: permutation stream does not reproduce')
    t.to_csv(OUT / 'null_long.tsv.gz', sep='\t', index=False)

    # ---- per gene --------------------------------------------------------
    per = []
    for g, sub in t.groupby('gene', sort=False):
        r = dict(gene=g, stratum=strata.loc[g, 'stratum'], n_perm=len(sub),
                 n_a=int(sub.n_a.iloc[0]), n_t=int(sub.n_t.iloc[0]))
        for ch, b, se, t2, dof, p in (('a', 'ba', 'sea', 't2_a', 'dofa', 'p_a'),
                                      ('t', 'bt', 'set', 't2_t', 'doft', 'p_t'),
                                      ('b', 'b', 'se', 't2_b', 'dof', 'p_b')):
            d = int(sub[dof].iloc[0])
            r[f'sdratio_{ch}'] = float(sub[b].std(ddof=1) /
                                       np.sqrt((sub[se] ** 2).mean()))
            r[f'meanshift_{ch}'] = float(sub[b].mean() /
                                         np.sqrt((sub[se] ** 2).mean()))
            for a_ in (0.05, 0.01):
                r[f'rej{a_}_{ch}'] = float((sub[p] < a_).mean())
            r[f'q95ratio_{ch}'] = float(np.quantile(sub[t2], 0.95) /
                                        sps.f.ppf(0.95, 1, d))
            r[f'q99ratio_{ch}'] = float(np.quantile(sub[t2], 0.99) /
                                        sps.f.ppf(0.99, 1, d))
        per.append(r)
    per = pd.DataFrame(per)
    per.to_csv(OUT / 'per_gene.tsv', sep='\t', index=False)

    # ---- pooled, with gene-clustered intervals ---------------------------
    brng = np.random.default_rng(np.random.SeedSequence(SEED).spawn(1)[0])
    res = dict(n_genes=len(tg), n_perm=N_PERM, gate1_max_rel=float(np.max(diffs)),
               gate2_max_dp=dmax, pooled={})
    print('\npooled rejection rate, gene-clustered 95% bootstrap interval')
    for ch, pc in (('allelic', 'p_a'), ('total', 'p_t'), ('combined', 'p_b')):
        res['pooled'][ch] = {}
        line = f'  {ch:9s}'
        for a_ in ALPHAS:
            est, lo, hi = rate_ci(t, pc, a_, brng)
            res['pooled'][ch][str(a_)] = dict(rate=est, lo=lo, hi=hi,
                                              ratio=est / a_)
            line += f'  {a_}: {est:.4f} [{lo:.4f},{hi:.4f}]'
        ks = sps.kstest(t[pc].values, 'uniform')
        res['pooled'][ch]['ks_D'] = float(ks.statistic)
        print(line + f'  KS D={ks.statistic:.4f}')
    first30 = t[t.perm < 30]
    res['first30'] = {ch: {str(a_): float((first30[pc] < a_).mean())
                           for a_ in (0.05, 0.01)}
                      for ch, pc in (('allelic', 'p_a'), ('total', 'p_t'),
                                     ('combined', 'p_b'))}
    print(f"  first 30 permutations: {res['first30']}")
    for ch in ('a', 't', 'b'):
        res[f'median_sdratio_{ch}'] = float(per[f'sdratio_{ch}'].median())
    (OUT / 'summary.json').write_text(json.dumps(res, indent=2))
    print(f'\nwrote {OUT}')


if __name__ == '__main__':
    main()

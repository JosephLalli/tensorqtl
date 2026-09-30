"""Post-fit adjustments of hapmixQTL's effect size toward the planted effect, on the current default (half-read total
phenotype log2((pT + 0.5) / (Leff + 1) * 1e6), unit total weights, Gibbs allelic weights: the `voom` rows of
beta_shortfall_refits.py) at the plasmode causal units, |beta| 0.4 and 0.8, both gene sets. Every adjustment
re-expresses the slope and its se; no p-value changes (each stays the shipped fit's).

  total      slope_t / c_hat, se_t / c_hat. First order, log2(y + 0.5) passes a share c_i = y_i / (y_i + 0.5) of donor
             i's log2 shift, so the fitted slope is about c_hat * beta with c_hat = sum(xr c x) / sum(xr x), x the tested
             genotype (dosage / 2) and xr it residualized on the total design (intercept, record covariates, genotype PCs).
  reweight   the through-origin allelic slope refit with each admitted record's Gibbs variance rescaled from its observed
             allele counts to the counts the fit predicts, v_i' = Va_i q(n_i p_i, n_i (1 - p_i)) / q(pL_i, pR_i),
             q(x, y) = 1/(x + 0.5) + 1/(y + 0.5), n_i = pL_i + pR_i, p_i = 2^(b s_i) / (1 + 2^(b s_i)), iterated to a
             fixed point in b. Near p = 0.5 this is the full-balance weighting rejected on 2026-09-29.
  bootstrap  the shipped Gibbs-weighted allelic slope, bias-corrected by inverting its mean map: simulate B replicates
             of each unit's admitted records from the fitted model (L* ~ Binomial(round(n_i), p_i), R* = round(n_i) - L*,
             a* = log2((L* + 0.5) / (R* + 0.5)), Va* = Va_i q(L*, R*) / q(pL_i, pR_i), the zero-haplotype admission
             re-applied to (L*, R*)), refit the shipped slope on each, and find the b whose mean simulated slope equals
             the observed slope (additive fixed-point steps on common random numbers). The corrected se is se_a * b / b_obs.
             It models binomial counting noise only; overdispersion beyond it is not simulated.
  wild       the same inversion with a wild bootstrap in place of the binomial model: each replicate keeps every record's
             own residual A_i - b s_i with a random sign (the L/R label swap is a symmetry of the null), rebuilds the
             counts that give the replicate ratio at the record's own depth, and recomputes Gibbs variance and admission
             from them, so the simulated spread is the data's own, overdispersion included.
  wild_pooled  one attenuation factor per gene set, the median over its units of b_obs / b_wild (both effect sizes, so it
             never sees the planted effect), applied to every allelic slope and se: a cross-gene correction.
  combined   the inverse-variance combination of the (adjusted) channel slopes at their (adjusted) se, the shipped rule.
A bootstrap inversion whose final gap |b_obs - mean simulated slope| exceeds CONVERGED_SE of the unit's allelic se is
counted as not converged and gives no estimate.

Known answers first: the NumPy allelic and total fits reproduce the stored map_nominal slopes at every causal unit whose
total design is not degenerate (map_nominal stores slope_t 0 with se_t inf where the tested genotype is constant).
Scores per band, 06_score.py's gene-clustered interval: mean slope / count-scale truth, the paired change, and the ratio
of mean squared relative errors (slope / truth - 1)^2, adjusted over shipped.

  PLASMODE_GENE_SET=<set> python3 scripts/beta_postfit_adjustment.py   # writes OUT/postfit_<set>.json
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binom

sys.path.insert(0, str(Path(__file__).resolve().parent / 'plasmode'))
import common as C          # noqa: E402

S6 = C.module('06_score')
OUT = C.D / 'beta_shortfall_20260929'
CONFIG, ARM = 'voom', 'split'   # the refits' half-read rows under split weighting: the adopted default (2026-09-29)
HALF = 0.5                      # the half-read pseudocount: T = log2((pT + HALF) / (Leff + 1) * 1e6)
B_SIM, BOOT_STEPS, CONV_TOL = 200, 50, 1e-6
CONVERGED_SE = 0.01             # a correction whose final |gap| exceeds this share of the unit's allelic se is not converged
SIM_KEY = 50                    # SeedSequence spawn key for the simulations (06 uses 30 and 33)
ITER, ITER_TOL = 50, 1e-10
SLOPE_TOL = 1e-4                # |NumPy - stored| / stored se: stored slopes are float32 GPU fits (observed <= 2e-5 se)


def q(x, y):
    return 1.0 / (x + C.KAPPA) + 1.0 / (y + C.KAPPA)


def slope(A, s, w):
    return (w * s * A).sum(-1) / (w * s * s).sum(-1)


def reweight(A, s, Va, n, pL, pR, b):
    q0 = q(pL, pR)
    for _ in range(ITER):
        p = 2.0 ** (b * s) / (1.0 + 2.0 ** (b * s))
        b_new = slope(A, s, 1.0 / (Va * q(n * p, n * (1 - p)) / q0))
        if abs(b_new - b) < ITER_TOL:
            return b_new
        b = b_new
    raise SystemExit(f'reweighted allelic slope did not converge in {ITER} iterations')


def invert_mean(mean_at, b_obs):
    """b with mean_at(b) = b_obs by additive fixed-point steps; returns b and the final gap b_obs - mean_at(b)."""
    b = b_obs
    for _ in range(BOOT_STEPS):
        gap = b_obs - mean_at(b)
        if abs(gap) < CONV_TOL:
            return b, gap
        b = b + gap
    return b, b_obs - mean_at(b)


def bootstrap_correct(s, Va, pL, pR, b_obs, rng):
    """b such that the mean shipped slope over B_SIM replicates simulated at b equals b_obs; common random numbers."""
    n = np.round(pL + pR).astype(np.int64)
    k = Va / q(pL, pR)
    u = rng.random((B_SIM, len(s)))

    def mean_at(b):
        p = 2.0 ** (b * s) / (1.0 + 2.0 ** (b * s))
        L = binom.ppf(u, n, p)
        R = n - L
        keep = ~((L < C.EXPRESSIBLE_MIN) ^ (R < C.EXPRESSIBLE_MIN)) & (n > 0)
        w = np.where(keep, 1.0 / (k * q(L, R)), 0.0)
        return np.nanmean(slope(np.log2((L + C.KAPPA) / (R + C.KAPPA)), s, w))
    return invert_mean(mean_at, b_obs)


def wild_correct(A, s, Va, pL, pR, b_obs, rng):
    """b such that the mean shipped slope over B_SIM wild replicates at b equals b_obs. A replicate keeps each record's
    own residual e_i = A_i - b s_i with a random sign (L/R label swap is a symmetry of the null), a*_i = b s_i +/- e_i;
    its counts are the ones that give a*_i at the record's own n_i (L* = (n_i + 1) p* - 0.5), and its Gibbs variance
    and admission are recomputed from them as the shipped fit would."""
    n = pL + pR
    k = Va / q(pL, pR)
    eta = rng.choice(np.array([-1.0, 1.0]), size=(B_SIM, len(s)))

    def mean_at(b):
        a = b * s + eta * (A - b * s)
        p = 2.0 ** a / (1.0 + 2.0 ** a)
        L = (n + 2 * C.KAPPA) * p - C.KAPPA
        R = n - L
        keep = ~((L < C.EXPRESSIBLE_MIN) ^ (R < C.EXPRESSIBLE_MIN)) & (L + C.KAPPA > 0) & (R + C.KAPPA > 0)
        w = np.where(keep, 1.0 / (k * q(np.maximum(L, 0.0), np.maximum(R, 0.0))), 0.0)
        return np.nanmean(slope(a, s, w))
    return invert_mean(mean_at, b_obs)


def adjust(I, genes, F):
    rows = []
    vi = I['vdf'].index
    for (sc, r), grp in F.groupby(['scenario', 'rep']):
        ds = C.load_dataset(C.DATASETS, sc, r)
        Z = np.column_stack([np.ones(len(I['order'])), I['cov_df'].values[ds['perm']], I['geno_cov_df'].values])
        T_all = np.log2((ds['pT'] + HALF) / (ds['eff_lib'][None, :] + 1.0) * 1e6)
        kept_all = C.allelic_kept(ds['pL'], ds['pR'], ds['Va'])
        for row in grp.itertuples():
            k, j = genes.index(row.gene), vi.get_loc(row.variant_id)
            x = I['dos'][j] / 2.0
            degenerate = not np.isfinite(row.slope_t_se)
            bt = c_hat = np.nan
            if not degenerate:
                bt = np.linalg.lstsq(np.column_stack([Z, x]), T_all[k], rcond=None)[0][-1]
                xr = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
                c = ds['pT'][k] / (ds['pT'][k] + HALF)
                c_hat = (xr * c * x).sum() / (xr * x).sum()
            s = I['xL'][j].astype(float) - I['xR'][j]
            use = kept_all[k] & (s != 0)
            A, Va, pL, pR, su = ds['A'][k][use], ds['Va'][k][use], ds['pL'][k][use], ds['pR'][k][use], s[use]
            ba = ba_rw = ba_bs = ba_wb = gap_bs = gap_wb = np.nan
            if use.any():
                ba = slope(A, su, 1.0 / Va)
                ba_rw = reweight(A, su, Va, pL + pR, pL, pR, ba)
                ba_bs, gap_bs = bootstrap_correct(su, Va, pL, pR, ba, np.random.default_rng(
                    np.random.SeedSequence(C.SEED, spawn_key=(SIM_KEY, 0, len(rows)))))
                ba_wb, gap_wb = wild_correct(A, su, Va, pL, pR, ba, np.random.default_rng(
                    np.random.SeedSequence(C.SEED, spawn_key=(SIM_KEY, 1, len(rows)))))
            rows.append(dict(scenario=sc, rep=r, gene=row.gene, degenerate_total=degenerate, bt=bt, c_hat=c_hat, ba=ba,
                             ba_rw=ba_rw, ba_bs=ba_bs, ba_wb=ba_wb, gap_bs=gap_bs, gap_wb=gap_wb,
                             **{f'st_{col}': getattr(row, col) for col in ('slope', 'slope_se', 'slope_a', 'slope_a_se', 'slope_t',
                                                                            'slope_t_se', 'allelic_admitted')}))
        print(f'{C.GENE_SET} {sc} rep {r}: {len(grp)} causal units adjusted', flush=True)
    P = pd.DataFrame(rows)
    nd = ~P.degenerate_total
    dt = (np.abs(P.bt - P.st_slope_t.astype(float)) / P.st_slope_t_se.astype(float))[nd]
    adm = P.st_allelic_admitted.astype(bool)
    da = (np.abs(P.ba - P.st_slope_a.astype(float)) / P.st_slope_a_se.astype(float))[adm]
    print(f'known answers: total slope max |difference| / se {dt.max():.2e} over {int(nd.sum())} units '
          f'({int((~nd).sum())} with a degenerate total design, map_nominal se_t inf, not compared), allelic '
          f'{da.max():.2e} over {int(adm.sum())} admitted units', flush=True)
    if dt.max() > SLOPE_TOL or da.max() > SLOPE_TOL:
        raise SystemExit('NumPy fits do not reproduce the stored map_nominal slopes')
    return P


def ratio_of_sums(num, den, Cz, genes, bsel, bidx):
    """sum(num) / sum(den) over finite units, per band, with the gene-clustered interval of 06_score.ratio_block."""
    gi = pd.Index(genes)
    ok = np.isfinite(num) & np.isfinite(den)
    idx = gi.get_indexer(Cz.gene[ok])
    sn = np.bincount(idx, weights=num[ok], minlength=len(genes))
    sd = np.bincount(idx, weights=den[ok], minlength=len(genes))
    out = {}
    for bn, gs in bsel.items():
        if sd[gs].sum() == 0:
            continue
        bb = sn[gs][bidx[bn]].sum(1) / sd[gs][bidx[bn]].sum(1)
        out[bn] = dict(mean=float(sn[gs].sum() / sd[gs].sum()), lo=float(np.nanquantile(bb, .025)),
                       hi=float(np.nanquantile(bb, .975)), units=int(np.isin(idx, gs).sum()))
    return out


def score(P, U, genes, bsel, bidx):
    unit = ['scenario', 'rep', 'gene']
    P = P.merge(U[~U.is_null][unit + ['beta', 'total_truth']], on=unit, how='left')
    adm = P.st_allelic_admitted.astype(bool).values
    sa, sea = P.st_slope_a.astype(float).values, P.st_slope_a_se.astype(float).values
    st, set_ = P.st_slope_t.astype(float).values, P.st_slope_t_se.astype(float).values
    fin_t = np.isfinite(set_) & np.isfinite(P.c_hat.values)
    t_adj = np.where(fin_t, st / np.where(fin_t, P.c_hat.values, 1.0), 0.0)
    conv_bs = np.abs(P.gap_bs.values) <= CONVERGED_SE * sea
    conv_wb = np.abs(P.gap_wb.values) <= CONVERGED_SE * sea
    ba_bs = np.where(conv_bs, P.ba_bs.values, np.nan)          # a non-converged correction has no estimate
    ba_wb = np.where(conv_wb, P.ba_wb.values, np.nan)
    kb = np.where(adm, P.ba.values / ba_bs, np.nan)            # each bootstrap's attenuation factor at the unit
    kw = np.where(adm, P.ba.values / ba_wb, np.nan)
    k_pool = float(np.nanmedian(kw))   # one cross-gene factor per gene set, over both effect sizes (never sees beta)

    def comb(a, sa_, t, st_):
        wa = np.where(adm & np.isfinite(sa_), 1.0 / sa_ ** 2, 0.0)
        wt = np.where(np.isfinite(st_), 1.0 / st_ ** 2, 0.0)
        return (np.where(wa > 0, wa * a, 0.0) + np.where(wt > 0, wt * t, 0.0)) / (wa + wt)

    se_t_adj = np.where(fin_t, set_ / np.where(fin_t, P.c_hat.values, 1.0), np.inf)
    est = dict(
        allelic=dict(shipped=sa, reweight=P.ba_rw.values, bootstrap=ba_bs, wild=ba_wb,
                     wild_pooled=sa / k_pool),
        total=dict(shipped=st, rescaled=t_adj),
        combined=dict(shipped=P.st_slope.astype(float).values,
                      total_rescaled=comb(sa, sea, t_adj, se_t_adj),
                      total_rescaled_reweight=comb(P.ba_rw.values, sea, t_adj, se_t_adj),
                      total_rescaled_bootstrap=comb(ba_bs, sea / kb, t_adj, se_t_adj),
                      total_rescaled_wild=comb(ba_wb, sea / kw, t_adj, se_t_adj),
                      total_rescaled_wild_pooled=comb(sa / k_pool, sea / k_pool, t_adj, se_t_adj)))
    truth = dict(allelic=P.beta.values, total=P.total_truth.values, combined=P.beta.values)
    live = dict(allelic=adm, total=np.isfinite(set_), combined=np.ones(len(P), bool))
    out = {}
    for sc, g in P.groupby('scenario'):
        ix = g.index.values
        Cz = g.reset_index(drop=True)
        ser = lambda v: pd.Series(np.asarray(v, float), index=Cz.index)   # noqa: E731
        res = {}
        for ch, arms in est.items():
            t = truth[ch][ix]
            mask = live[ch][ix]
            base = np.where(mask, arms['shipped'][ix] / t, np.nan)
            res[ch] = {}
            for name, v in arms.items():
                r_ = np.where(mask, v[ix] / t, np.nan)
                res[ch][name] = dict(ratio=S6.ratio_block(ser(r_), Cz, genes, bsel, bidx),
                                     change=S6.ratio_block(ser(r_ - base), Cz, genes, bsel, bidx),
                                     mse_ratio=ratio_of_sums((r_ - 1) ** 2, (base - 1) ** 2, Cz, genes, bsel, bidx))
        res['c_hat_mean'] = float(np.nanmean(g.c_hat))
        res['bootstrap_k_median'] = float(np.nanmedian(kb[ix]))   # medians: a per-unit ratio with a near-zero slope is unbounded
        res['wild_k_median'] = float(np.nanmedian(kw[ix]))
        res['wild_k_pooled'] = k_pool
        a_ix = adm[ix]
        res['not_converged'] = dict(admitted_units=int(a_ix.sum()), binomial=int((a_ix & ~conv_bs[ix]).sum()),
                                    wild=int((a_ix & ~conv_wb[ix]).sum()))
        res['degenerate_total_units'] = int(g.degenerate_total.sum())
        out[sc] = res
    return out


def main():
    meta, genes, U, keep_a = S6.load_units(C.DATASETS, C.RESULTS)
    bsel, bidx = S6.band_selections(genes, U, keep_a)
    I = C.load()[0]
    F = pd.read_parquet(OUT / f'refits_{C.GENE_SET}.parquet')
    F = F[(F.config == CONFIG) & (F.arm == ARM)]
    print(f'{len(F)} causal-variant rows ({CONFIG}, {ARM}; {sorted(F.scenario.unique())})', flush=True)
    P = adjust(I, list(genes), F)
    C.write_atomic(OUT / f'postfit_units_{C.GENE_SET}.parquet', lambda fh: P.to_parquet(fh, index=False))
    res = score(P, U, genes, bsel, bidx)
    C.write_json(OUT / f'postfit_{C.GENE_SET}.json', dict(gene_set=C.GENE_SET, root=str(C.ROOT), config=CONFIG, arm=ARM,
                                                          b_sim=B_SIM, boot_steps=BOOT_STEPS, result=res))
    f = lambda d: f'{d["all"]["mean"]:.3f} [{d["all"]["lo"]:.3f}, {d["all"]["hi"]:.3f}]'   # noqa: E731
    for sc, r in res.items():
        print(f'{C.GENE_SET} {sc}: mean c_hat {r["c_hat_mean"]:.3f}, median attenuation binomial {r["bootstrap_k_median"]:.3f} '
              f'wild {r["wild_k_median"]:.3f} (pooled median {r["wild_k_pooled"]:.3f}); not converged of '
              f'{r["not_converged"]["admitted_units"]} admitted: binomial {r["not_converged"]["binomial"]}, wild '
              f'{r["not_converged"]["wild"]}; '
              f'degenerate total units {r["degenerate_total_units"]}')
        for ch in ('allelic', 'total', 'combined'):
            for name, x in r[ch].items():
                print(f'   {ch:9s} {name:25s} ratio {f(x["ratio"])} change {f(x["change"])} MSE ratio {f(x["mse_ratio"])}')
    print(f'wrote {OUT / f"postfit_{C.GENE_SET}.json"}')


if __name__ == '__main__':
    main()

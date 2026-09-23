"""hapmixQTL vs the mixQTL replication arm, on the 29 null-calibration genes.

Three analyses, deliberately separate because they answer different questions
and only some of them permit comparing numbers on a common scale.

END-TO-END COMPARISON (does hapmixQTL agree with its parent?)
    hapmixQTL at its shipped defaults against mixqtl_replication, same genes,
    same variant window, same 40 null genotype permutations. The two arms use
    different response transforms (log2 with kappa=0.5 vs natural log with no
    pseudocount), different donor cutoffs, and therefore different variant sets
    after the post-filter var(x)==0 drop. So this experiment compares only
    scale-free quantities: lead-variant agreement, Spearman correlation of the
    per-gene statistic, and type-I error at 5% from each arm's own null.
    Raw betas and SEs are NOT compared across arms; that would be a units
    error dressed up as a result.

WEIGHTING ABLATION (do the Gibbs draws buy anything?)
    This is the measurement that bears on whether propagating the draws
    improves the effect estimate. Everything is held fixed -- hapmixQTL's
    response A, its informative-donor set (va > 0), the same variants, the
    same through-origin design -- and ONLY the weight vector varies:

        gibbs_1_over_v           1/(draw variance + q), hapmixQTL as shipped
        gibbs_draws_only         1/(draw variance), without the Poisson q the
                                 2026-09-15 experiment found double-counts
        gibbs_capped             the same, under mixQTL's fold cap
        harmonic_uncapped        1/(1/YL_bar + 1/YR_bar), Poisson precision
        harmonic_poisson_capped  the same, capped: mixQTL as published
        equal_ols                no weighting at all

    The 2x2 of {weight source} x {cap} matters because mixQTL caps and
    hapmixQTL does not; without both levels the comparison would confound
    the weight source with the capping.

    Under a null permutation the true slope is zero, so the spread of
    beta_hat across permutations IS the estimation error of the estimator --
    no simulated ground truth is required. Gauss-Markov says the weighted fit
    beats the unweighted one exactly when the weights are inversely
    proportional to the true error variance AND that variance varies across
    donors. So:

        gibbs lowest    -> the draw variance is the better error model
        harmonic lowest -> the Poisson approximation is closer to the truth
        all tied        -> the error is effectively constant across donors
                           within a gene, which is the regime in which a
                           per-gene scale cancels from the permutation
                           p-value and no weighting can change an answer

RESIDUAL-FLOOR PROFILE (is the draw variance the WHOLE error?)
    Profiles weights 1/(v + tau) over a grid of tau to find the floor that
    minimizes the realized estimation error. Distinguishes an error that is
    a multiple of v -- a pure scale, which cancels from a within-gene
    permutation p-value -- from one with an additive donor-constant part,
    which does not.

    We also record mean(se^2) per arm. The ratio var(beta_hat)/mean(se^2) is
    the arm's calibration: 1 means the reported uncertainty matches the
    realized spread, below 1 means the arm understates its own error. That
    converts "does it reduce the SE" into "does it reduce the SE honestly",
    which is the only version of the question that bears on a call.

Data loading follows fitted_variance/estimator_ablation_20260916/ablation29.py lines 32-58,
copied rather than imported so this script has no dependency on a path
outside the repository.
"""

import contextlib
import io
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
D = '/mnt/ssd/lalli/brainvar_hapmix_deploy'
OUT = f'{D}/mixqtl_replication_20260919'
NP_NULL = int(os.environ.get('NP', '40'))
WIN, MAF, NPERM = 1_000_000, 0.05, 1000
# Master seed for every random draw here. Child streams are derived as
# SEED + <fixed offset> + <index>, because a bare RandomState(SEED) reused per
# draw would hand back the same permutation every time. Changing SEED changes
# every null draw; it does not change which arms share a permutation, since
# all arms within a gene use the same one.
SEED = 42

# REPO must precede REPO/tensorqtl, or the inner directory shadows the
# package and `import tensorqtl.hapmixqtl` fails.
sys.path.insert(0, f'{REPO}/scripts')
sys.path.insert(0, REPO)
sys.path.append(f'{REPO}/tensorqtl')

import tensorqtl.hapmixqtl as HM                      # noqa: E402
from tensorqtl import mixqtl_replication as MX        # noqa: E402

log = lambda *a: print(time.strftime('%H:%M:%S'), *a, flush=True)


# ---------------------------------------------------------------------------
#  inputs (ablation29.py:32-58)
# ---------------------------------------------------------------------------

def load_inputs():
    import run_hapmixqtl_from_salmon as H
    cache = f'{D}/cache/gibbs_56b63c3b37ed5df8'
    genes_all = open(f'{cache}/genes.txt').read().split()
    samples = open(f'{cache}/samples.txt').read().split()
    genes = [l.strip() for l in open(f'{D}/pilot29_hc.txt') if l.strip()]
    gi = {g: i for i, g in enumerate(genes_all)}
    rows = [gi[g] for g in genes]
    mm = {k: np.load(f'{cache}/{k}.npy', mmap_mode='r') for k in ('YL', 'YR', 'YT')}
    YL, YR, YT = (np.asarray(mm[k][rows]) for k in ('YL', 'YR', 'YT'))

    gp = pd.read_csv(f'{D}/annot/genes.tsv', sep='\t', header=None, dtype={1: str})
    gp.columns = ['gene', 'chr', 'start', 'end', 'pos']
    gp = gp.set_index('gene')
    gp['chr'] = gp['chr'].astype(str).str.strip()
    with contextlib.redirect_stdout(io.StringIO()):
        vdf, dos, xL, xR, order = H.read_phased_vcf(
            f'{D}/prepped/analysis.snps.maf01.vcf.gz', set(samples),
            regions=f'{D}/fitted_variance/null_calibration_29b/regions.bed')
    keep = [samples.index(s) for s in order]
    order = list(order)
    vdf['chrom'] = vdf['chrom'].astype(str)
    pos, ch = vdf['pos'].values, vdf['chrom'].values
    in_body = np.zeros(len(vdf), bool)
    in_win = np.zeros(len(vdf), bool)
    for g in genes:
        r = gp.loc[g]
        same = ch == str(r['chr'])
        in_body |= same & (pos >= int(r['start'])) & (pos <= int(r['end']))
        in_win |= same & (np.abs(pos - int(r['pos'])) <= WIN)
    af = dos.mean(1) / 2.0
    tested = in_win & ~in_body & (np.minimum(af, 1 - af) >= MAF)
    idx = np.where(tested)[0]

    cov_df = pd.read_csv(f'{D}/cov/covariates.tsv', sep='\t', index_col=0).loc[order]

    # library size: mapped fragments, bound by sample key (never by position)
    meta = json.load(open(f'{D}/gibbs_influence_audit_20260915/input_source_metadata.json'))
    mf = {e['sample']: float(e['mapped_fragments']) for e in meta}
    missing = [s for s in order if s not in mf]
    if missing:
        raise SystemExit(f'no mapped_fragments for {len(missing)} samples, e.g. {missing[:3]}')
    lib_size = np.array([mf[s] for s in order])

    return dict(genes=genes, order=order, keep=keep, YL=YL, YR=YR, YT=YT,
                vdf=vdf, dos=dos, xL=xL, xR=xR, idx=idx, gp=gp,
                cov_df=cov_df, lib_size=lib_size)


def gene_variant_index(I, g):
    """Variant rows within I['idx'] that fall in gene g's cis window."""
    r = I['gp'].loc[g]
    v = I['vdf'].iloc[I['idx']]
    same = v['chrom'].values == str(r['chr'])
    return np.where(same & (np.abs(v['pos'].values - int(r['pos'])) <= WIN))[0]


# ---------------------------------------------------------------------------
#  weighting ablation
# ---------------------------------------------------------------------------

def weighting_ablation(I):
    """Vary only the allelic weights; measure var(beta_hat) under the null."""
    genes, order, keep = I['genes'], I['order'], I['keep']
    A, _T, Va, _Vt, _C = HM.compute_summaries_from_gibbs(
        I['YL'], I['YR'], yT=I['YT'], count_noise=True)
    # count_noise=True adds a Poisson term q on top of the draw variance, so
    # Va is 1/(draw_var + q), not the draws alone. The 2026-09-15 controlled
    # Salmon experiment found q double-counts. Carry both so the ablation
    # separates "the draws help" from "the shipped variance helps".
    _A0, _T0, Va_noq, _Vt0, _C0 = HM.compute_summaries_from_gibbs(
        I['YL'], I['YR'], yT=I['YT'], count_noise=False)
    A, Va, Va_noq = A[:, keep], Va[:, keep], Va_noq[:, keep]
    mL = I['YL'].mean(2)[:, keep]
    mR = I['YR'].mean(2)[:, keep]

    # Per-haplotype draw variances, for the third candidate weight. Va_noq is
    # already Var_draws(log(yL+k) - log(yR+k)), the DIRECT variance of the
    # regressed quantity; vL + vR is what you get if you instead sum the two
    # marginal variances, which equals it only when the haplotypes' draws are
    # uncorrelated. They are strongly anti-correlated (median -0.841; Gibbs
    # reassigns ambiguous reads between the two copies, so one gains what the
    # other loses), and vL + vR understates the ratio's variance by a median
    # 1.81-fold in 100% of informative pairs -- see
    # scripts/gibbs_variance_target.py. The arm is here to measure what that
    # costs, which is not obvious: the shortfall is nearly a gene CONSTANT
    # (within-gene log-weight correlation 0.998), and a constant weight factor
    # cancels from beta_hat entirely.
    KAPPA = 0.5
    vL = np.log(I['YL'] + KAPPA).var(2)[:, keep]
    vR = np.log(I['YR'] + KAPPA).var(2)[:, keep]

    s_all = (I['xL'] - I['xR'])[I['idx']][:, keep]      # [V, N] allelic design
    rows = []
    for j, g in enumerate(genes):
        vsel = gene_variant_index(I, g)
        if vsel.size == 0:
            continue
        # Common informative-donor set across every arm, so the comparison
        # is of weightings and not of donor sets. Requiring the q-free draw
        # variance to be positive too avoids handing a 1e12 weight to a donor
        # whose draws happen to be unanimous.
        inf = (Va[j] > 1e-12) & (Va_noq[j] > 1e-12)
        n_inf = int(inf.sum())
        n_dropped_by_noq = int(((Va[j] > 1e-12) & ~(Va_noq[j] > 1e-12)).sum())
        if n_inf <= 2:
            continue
        a = A[j][inf]
        S = s_all[vsel][:, inf]                          # [P, n_inf]

        # variants that still vary within the informative donor set
        varying = S.var(axis=1) > 0
        S = S[varying]
        if S.shape[0] == 0:
            continue
        X = S.T                                          # [n_inf, P]

        # 2x2 on {weight source} x {mixQTL fold cap}, plus unweighted.
        # Without both cap levels the comparison confounds "Gibbs vs Poisson"
        # with "uncapped vs capped", since mixQTL caps and hapmixQTL does not.
        w_gibbs = 1.0 / np.maximum(Va[j][inf], 1e-12)
        w_gibbs_noq = 1.0 / np.maximum(Va_noq[j][inf], 1e-12)
        w_harm = MX.harmonic_weights(np.maximum(mL[j][inf], 1e-12),
                                     np.maximum(mR[j][inf], 1e-12))
        w_sumvar = 1.0 / np.maximum(vL[j][inf] + vR[j][inf], 1e-12)
        w_gibbs_cap, cap, _ = MX.apply_weight_cap(w_gibbs, n_inf, MX.WEIGHT_CAP)
        w_harm_cap, _, _ = MX.apply_weight_cap(w_harm, n_inf, MX.WEIGHT_CAP)
        w_sumvar_cap, _, _ = MX.apply_weight_cap(w_sumvar, n_inf, MX.WEIGHT_CAP)

        arms = {
            'gibbs_1_over_v': w_gibbs,            # hapmixQTL as shipped (q on)
            'gibbs_draws_only': w_gibbs_noq,      # 1/v_ratio, the direct variance
            'gibbs_sumvar': w_sumvar,             # 1/(vL + vR), assumes independence
            'gibbs_capped': w_gibbs_cap,
            'gibbs_sumvar_capped': w_sumvar_cap,
            'harmonic_uncapped': w_harm,
            'harmonic_poisson_capped': w_harm_cap,  # mixQTL as published
            'equal_ols': np.ones(n_inf),
        }
        betas = {k: np.empty((NP_NULL, X.shape[1])) for k in arms}
        se2 = {k: np.empty((NP_NULL, X.shape[1])) for k in arms}
        # hapmixQTL's SHIPPED standard error is the known-variance form
        # 1/sqrt(xx), with no fitted residual scale multiplying it. Track it
        # separately: the two forms calibrate differently, and the shipped
        # anticonservatism is a property of this one, not of the weights.
        se2_known = {k: np.empty((NP_NULL, X.shape[1])) for k in arms}

        # Null: permute the donors of the PHENOTYPE bundle, so response and
        # weight move together and each donor keeps its own measurement
        # precision. Permuting within the informative subset is what breaks
        # the genotype-phenotype pairing; the design X never moves.
        for pi in range(NP_NULL):
            prm = np.random.RandomState(SEED + 10007 + pi).permutation(n_inf)
            a_p = a[prm]
            for k, w in arms.items():
                wp = w[prm]
                b, s = MX._simple_regression_through_origin(a_p, X, wp)
                betas[k][pi] = b
                se2[k][pi] = s ** 2
                xx = (X * X * wp[:, None]).sum(0)
                with np.errstate(divide='ignore', invalid='ignore'):
                    se2_known[k][pi] = np.where(xx > 0, 1.0 / xx, np.nan)

        n_used = 0
        for k in arms:
            vb = np.var(betas[k], axis=0, ddof=1)        # across permutations
            ms = np.nanmean(se2[k], axis=0)
            mk = np.nanmean(se2_known[k], axis=0)
            ok = np.isfinite(vb) & np.isfinite(ms) & (ms > 0)
            if ok.sum() == 0:
                continue
            n_used = int(ok.sum())
            okk = ok & np.isfinite(mk) & (mk > 0)
            rows.append(dict(
                gene=g, arm=k, n_inf=n_inf, n_var=n_used, cap=float(cap),
                median_var_beta=float(np.median(vb[ok])),
                median_mean_se2=float(np.median(ms[ok])),
                median_calibration=float(np.median(vb[ok] / ms[ok])),
                median_calibration_known_var=(
                    float(np.median(vb[okk] / mk[okk])) if okk.sum() else np.nan),
                weight_fold_spread=float(arms[k].max() / arms[k].min()),
                eff_n=float(arms[k].sum() ** 2 / (arms[k] ** 2).sum()),
                n_dropped_by_noq=n_dropped_by_noq,
            ))
        log(f'  ablation {g}: n_inf={n_inf} variants={n_used}')
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
#  residual-floor profile
# ---------------------------------------------------------------------------

def residual_floor_profile(I, grid=(0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)):
    """Which additive floor tau minimizes the realized estimation error?

    The weighting ablation asks whether 1/v beats the alternatives, but if
    the true per-donor error is v_i + tau -- measurement plus a donor-constant
    biological residual -- then 1/v over-downweights the low-v donors and the
    ablation cannot see it. This profiles weights 1/(v_i + tau) over a grid of
    tau, expressed as a multiple of each gene's median v so the grid is
    comparable across genes, and reports the tau that minimizes var(beta_hat)
    across the null permutations.

    This is not circular in the way a moment estimator fitted to the same
    residuals it then weights is circular: tau is selected by the realized
    spread of the estimator under permutation, a criterion the weights do not
    themselves determine. tau -> infinity is the unweighted limit.

    A minimum at tau = 0 says the draws alone are the error model. A minimum
    at tau > 0 measures how much donor-to-donor variation survives beyond
    them, in units of the gene's own measurement scale.
    """
    genes, keep = I['genes'], I['keep']
    A, _T, Va, _Vt, _C = HM.compute_summaries_from_gibbs(
        I['YL'], I['YR'], yT=I['YT'], count_noise=True)
    A, Va = A[:, keep], Va[:, keep]
    s_all = (I['xL'] - I['xR'])[I['idx']][:, keep]

    rows = []
    for j, g in enumerate(genes):
        vsel = gene_variant_index(I, g)
        if vsel.size == 0:
            continue
        inf = Va[j] > 1e-12
        n_inf = int(inf.sum())
        if n_inf <= 2:
            continue
        a = A[j][inf]
        v = Va[j][inf]
        vmed = float(np.median(v))
        S = s_all[vsel][:, inf]
        S = S[S.var(axis=1) > 0]
        if S.shape[0] == 0:
            continue
        X = S.T
        for c in grid:
            w = 1.0 / (v + c * vmed)
            bb = np.empty((NP_NULL, X.shape[1]))
            for pi in range(NP_NULL):
                prm = np.random.RandomState(SEED + 10007 + pi).permutation(n_inf)
                b, _ = MX._simple_regression_through_origin(a[prm], X, w[prm])
                bb[pi] = b
            vb = np.var(bb, axis=0, ddof=1)
            vb = vb[np.isfinite(vb)]
            if vb.size:
                rows.append(dict(gene=g, tau_mult=c, n_inf=n_inf,
                                 median_var_beta=float(np.median(vb))))
        log(f'  floor profile {g}')
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
#  end-to-end comparison
# ---------------------------------------------------------------------------

def mixqtl_gene(I, g, j, y1, y2, yt, perm=None):
    """Run the mixQTL replication arm on one gene."""
    vsel = gene_variant_index(I, g)
    if vsel.size == 0:
        return None
    keep = I['keep']
    h1 = I['xL'][I['idx']][:, keep][vsel].T.astype(float)   # [N, P]
    h2 = I['xR'][I['idx']][:, keep][vsel].T.astype(float)
    cov = I['cov_df'].values
    lib = I['lib_size']
    a1, a2, at = y1[j], y2[j], yt[j]
    if perm is not None:
        a1, a2, at = a1[perm], a2[perm], at[perm]
        cov = cov[perm]
        lib = lib[perm]
    out = MX.mixqtl_scan(a1, a2, at, lib, h1, h2, covariates=cov)
    stat = np.abs(out['meta']['stat'])
    if not np.isfinite(stat).any():
        return None
    k = int(np.nanargmax(stat))
    v = I['vdf'].iloc[I['idx']].iloc[vsel]
    return dict(gene=g, stat=float(stat[k] ** 2),
                variant_id=str(v.index[k]),
                beta=float(out['meta']['beta'][k]),
                se=float(out['meta']['se'][k]),
                method=str(out['meta']['method'][k]),
                n_trc=int(out['trc']['sample_size']),
                n_asc=int(out['asc']['sample_size']),
                n_cov_selected=int(out['cov_selected'].sum())
                if out['cov_selected'] is not None else 0,
                num_var=int(np.isfinite(stat).sum()))


def endtoend_comparison(I):
    genes, order = I['genes'], I['order']
    y1, y2, yt = MX.summaries_from_gibbs_posterior_mean(I['YL'], I['YR'], I['YT'])
    keep = I['keep']
    y1, y2, yt = y1[:, keep], y2[:, keep], yt[:, keep]
    n_donor = len(order)

    obs = [r for j, g in enumerate(genes)
           if (r := mixqtl_gene(I, g, j, y1, y2, yt)) is not None]
    obs = pd.DataFrame(obs)
    log(f'  end-to-end observed: {len(obs)} genes, median stat {obs.stat.median():.2f}')

    nulls = []
    for p in range(NP_NULL):
        prm = np.random.RandomState(SEED + 10007 + p).permutation(n_donor)
        rs = [r for j, g in enumerate(genes)
              if (r := mixqtl_gene(I, g, j, y1, y2, yt, perm=prm)) is not None]
        d = pd.DataFrame(rs)
        d['draw'] = p
        nulls.append(d)
        if p % 10 == 9:
            log(f'  end-to-end null draw {p + 1}/{NP_NULL}')
    return obs, pd.concat(nulls, ignore_index=True)


# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUT, exist_ok=True)
    log('loading inputs')
    I = load_inputs()
    log(f"{len(I['genes'])} genes, {len(I['order'])} donors, "
        f"{len(I['idx'])} tested variants, {I['cov_df'].shape[1]} covariates")

    log('WEIGHTING ABLATION')
    b = weighting_ablation(I)
    b.to_csv(f'{OUT}/weighting_ablation.tsv', sep='\t', index=False)
    agg = b.groupby('arm').agg(
        genes=('gene', 'nunique'),
        median_var_beta=('median_var_beta', 'median'),
        median_calibration=('median_calibration', 'median'),
        median_calib_knownvar=('median_calibration_known_var', 'median'),
        median_weight_fold=('weight_fold_spread', 'median'),
        median_eff_n=('eff_n', 'median'),
        total_dropped_by_noq=('n_dropped_by_noq', 'sum')).reset_index()
    # Relative efficiency against unweighted OLS, per gene then median, so
    # each gene contributes its own paired ratio rather than a ratio of
    # medians across genes with different scales.
    piv = b.pivot(index='gene', columns='arm', values='median_var_beta')
    rel = (piv.div(piv['equal_ols'], axis=0)).median().rename(
        'median_var_ratio_vs_ols')
    agg = agg.merge(rel, left_on='arm', right_index=True)
    log('\n' + agg.to_string(index=False))

    log('RESIDUAL-FLOOR PROFILE')
    fp = residual_floor_profile(I)
    fp.to_csv(f'{OUT}/residual_floor_profile.tsv', sep='\t', index=False)
    # normalize each gene's curve to its own tau=0 value, then take the
    # median across genes, so genes with different effect scales contribute
    # equally instead of the largest-variance gene dominating.
    pv = fp.pivot(index='gene', columns='tau_mult', values='median_var_beta')
    prof = pv.div(pv[0.0], axis=0).median()
    log('\nvar(beta_hat) relative to tau=0, median over genes:\n'
        + prof.to_string())
    best_tau = float(prof.idxmin())

    log('END-TO-END COMPARISON')
    obs, nulls = endtoend_comparison(I)
    obs.to_csv(f'{OUT}/endtoend_mixqtl_observed.tsv', sep='\t', index=False)
    nulls.to_csv(f'{OUT}/endtoend_mixqtl_nulls.tsv', sep='\t', index=False)

    summary = dict(
        n_genes=len(I['genes']), n_donors=len(I['order']),
        n_tested_variants=int(len(I['idx'])), n_null_draws=NP_NULL,
        weighting_ablation=agg.to_dict('records'),
        residual_floor_profile={str(k): float(v) for k, v in prof.items()},
        residual_floor_best_tau_mult=best_tau,
        mixqtl_observed_median_stat=float(obs.stat.median()),
        mixqtl_null_median_stat=float(nulls.stat.median()),
        mixqtl_null_p95_stat=float(nulls.stat.quantile(0.95)),
        mixqtl_median_n_asc=float(obs.n_asc.median()),
        mixqtl_median_n_trc=float(obs.n_trc.median()),
        mixqtl_median_n_cov_selected=float(obs.n_cov_selected.median()),
        mixqtl_method_counts={k: int(v) for k, v in
                              obs.method.value_counts().items()},
    )
    json.dump(summary, open(f'{OUT}/summary.json', 'w'), indent=1)
    log('wrote ' + f'{OUT}/summary.json')


if __name__ == '__main__':
    main()

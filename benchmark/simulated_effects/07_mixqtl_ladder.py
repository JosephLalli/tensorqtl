"""Why mixQTL mode trails the unit-weight arm: one-change rungs between them on the run's results
(README: mixQTL ladder). Every squared-error ratio is scored on one COMMON SET of causal units per
scenario and channel (finite error under unit, both cutoff rungs and both mixQTL arms; for the
total channel also both one-step slopes identified at both cutoffs), with each arm's own pairing
with unit (06_score.nonnull_precision) as a secondary column.

TOTAL CHANNEL. mixQTL first fits covariate_offset (log(yT / L / 2) on the covariates without the
genotype, |t| > 2 selection, refit on the selected set S), then regresses the offset response on
x = (h1 + h2)/2 WITHOUT residualizing x on S. By Frisch-Waugh-Lovell the one-step slope (y on [1,
x, C_S]) equals the slope on x residualized on [1, C_S], and the two-step slope is that times
1 - R^2(x ~ [1, C_S]) when both steps use the same donors (exact where the offset's donors, yT >
0, and trc's, yT >= trc_cutoff, coincide; enforced there to FWL_TOL). Per dataset x gene at its
causal variant and cutoff: mixqtl_trc (MX.trc_channel as mixqtl_scan calls it; gated against the
run's slope_t on the first dataset), one_step_trc (x residualized on [1, C_S] over the trc
donors), one_step_all_trc ([1, C], no selection on the outcome), r2, n_selected, n_donors; NaN
where trc donors < 2 + covariates + 1. Reported per scenario: mean slope / count-scale total
truth (x ln 2, natural log like mixQTL's slope), mean 1 - R^2, the Pearson r of mixqtl_trc /
truth with 1 - R^2 (partly algebraic), the squared error against the unit arm at its
pipeline-scale truth, and as the chance level of R^2 its mean at the null genes' causal variants.

DONOR SET. Rungs from 'unit' toward mixQTL: unit weights with mixQTL's count cutoffs as donor
admission (hapmixqtl.count_cutoff_masks on the thinned point estimates, yT = pT) at the published
and permissive settings; mixqtl and mixqtl_permissive from the results are the far ends. A channel
with no information comes back from map_nominal as slope 0, se inf, p 1; the rung files store it
as NaN. ESTIMABILITY records where the rung's joint design (intercept, 17 covariates, genotype)
has no residual df while mixQTL's trc (intercept and x) still estimates.

Output: LADDER/<scenario>/<rung>/nominal_repNNN.parquet, LADDER/total_channel_units.tsv,
LADDER/ladder.json (08_report.py reads it). Run only for a gene set whose common.GENE_SETS entry
names a ladder directory (the default set); for any other set it prints a skip and writes nothing.
"""
import shutil

import numpy as np
import pandas as pd

import common as C
from tensorqtl.hapmixqtl import count_cutoff_masks

SC = C.module('06_score')
CUTOFFS = {'published': C.MX.PUBLISHED_CUTOFFS, 'permissive': C.MX.PACKAGE_DEFAULT_CUTOFFS}   # 03_run_arms' two mixQTL settings
MIXQTL_ARM = {'published': 'mixqtl', 'permissive': 'mixqtl_permissive'}
RUNGS = {f'unit_{c}_cutoffs': c for c in CUTOFFS}
ARMS = ('unit', 'unit_published_cutoffs', 'mixqtl', 'unit_permissive_cutoffs', 'mixqtl_permissive')
CHANNELS = ('combined', 'allelic', 'total')
TRC = ('mixqtl_trc', 'one_step_trc', 'one_step_all_trc')
NO_ESTIMATE = (('slope', 'slope_se', 'pval_nominal'), ('slope_a', 'slope_a_se', 'pval_a'), ('slope_t', 'slope_t_se', 'pval_t'))
TRC_MIN_DONORS = 3                # mixqtl_replication.trc_channel returns no slope at n <= 2 donors
FWL_TOL = 1e-8                    # review threshold; the 2026-09-26 full run measured 8.4e-12
XCHECK_RTOL = 1e-9                # the run's and the recomputed mixQTL trc slopes agree to this (tests/test_mixqtl_replication.py)
LN2 = np.log(2.0)


def residualize(x, D, K):
    """x minus its least-squares fit on [1, D] over donors K, and R^2 of that fit over K (NaN if x is constant on K)."""
    D = np.column_stack([np.ones(len(x)), D])
    r = x - D @ np.linalg.lstsq(D[K], x[K], rcond=None)[0]
    xc = x[K] - x[K].mean() if K.any() else x[K]
    return r, (1.0 - (r[K] @ r[K]) / (xc @ xc) if xc @ xc > 0 else np.nan)


def total_channel(S, ds, cutoff):
    """At every gene's causal variant: mixQTL's trc slope, the two one-step slopes, R^2, |S| and donors."""
    I, MX, tc = S['I'], C.MX, CUTOFFS[cutoff]['trc_cutoff']
    _, _, yt = MX.inputs_from_point_estimates(ds['pL'], ds['pR'], ds['pT'])
    Cv = C.stack_covariates(I['cov_df'].values[ds['perm']], I['geno_cov_df'].values)
    rec = []
    for k, g in enumerate(S['genes']):
        v = str(ds['causal_variant'][k])
        x = (I['xL'][S['rows'][v]] + I['xR'][S['rows'][v]]) / 2.0
        off, sel = MX.covariate_offset(yt[k], ds['eff_lib'], Cv)
        K = yt[k] >= tc                   # trc_channel's donors (tc > 0, so the log response is finite)
        n = int(K.sum())
        fit = lambda z: MX.trc_channel(yt[k], ds['eff_lib'], z[:, None], off, tc)['beta'][0]   # noqa: E731
        x_sel, r2 = residualize(x, Cv[:, sel], K)
        x_all, _ = residualize(x, Cv, K)
        rec.append(dict(gene=g, variant_id=v, cutoff=cutoff, mixqtl_trc=fit(x),
                        one_step_trc=fit(x_sel) if n - 2 - sel.sum() >= 1 else np.nan,
                        one_step_all_trc=fit(x_all) if n - 2 - Cv.shape[1] >= 1 else np.nan,
                        r2=r2, n_selected=int(sel.sum()), n_donors=n, same_donor_sets=bool(n == (yt[k] > 0).sum())))
    return pd.DataFrame(rec)


def gate_trc(T, sc, r, cutoff):
    """mixqtl_trc must reproduce the run's mixQTL arm slope_t (natural log) at the causal variant."""
    p = C.RESULTS / sc / MIXQTL_ARM[cutoff] / f'nominal_rep{r:03d}.parquet'
    d = pd.read_parquet(p, columns=['phenotype_id', 'variant_id', 'slope_t'])
    d['variant_id'] = d['variant_id'].astype(str)
    want = T.merge(d, left_on=['gene', 'variant_id'], right_on=['phenotype_id', 'variant_id'], how='left').slope_t
    ok = np.isclose(T.mixqtl_trc.values, want.values, rtol=XCHECK_RTOL, atol=0, equal_nan=True)
    if not ok.all():
        raise SystemExit(f'{p}: trc_channel differs from the run\'s slope_t in {int((~ok).sum())} genes')
    print(f'{sc} rep {r:03d} {cutoff}: trc_channel equals the run\'s {MIXQTL_ARM[cutoff]} slope_t in all {len(T)} genes', flush=True)


def run_rung(S, ds, cutoff, scratch):
    """map_nominal as the 'unit' arm, plus mixQTL's count cutoffs as donor admission; se inf stored as NaN."""
    c = CUTOFFS[cutoff]
    ka, kt = count_cutoff_masks(ds['pL'], ds['pR'], yT=ds['pT'], asc_cutoff=c['asc_cutoff'], asc_cap=c['asc_cap'],
                                trc_cutoff=c['trc_cutoff'])
    df, _ = C.run_nominal(S, ds, 'unit', scratch, keep_a=ka, keep_t=kt)
    for b, s, p in NO_ESTIMATE:
        m = np.isinf(df[s].values)
        df.loc[m, [b, s, p]] = np.nan
    total_empty = df.slope_t.isna().groupby(df.phenotype_id).all().loc[S['genes']].values
    informative = C.arm_variances(ds, 'unit')[0] > 0
    return df, total_empty, (f'allelic pairs admitted {int((ka & informative).sum())} of {int(informative.sum())}, '
                             f'total pairs admitted {int(kt.sum())} of {kt.size}')


def estimability(units):
    """Per cutoff over every gene x dataset: where the rung's total channel is empty, against admitted trc donors."""
    out = {}
    for c in CUTOFFS:
        u = units[units.cutoff == c]
        e = u[u.rung_total_empty]
        mix = e[np.isfinite(e.mixqtl_trc)]
        out[c] = dict(gene_datasets=len(u), rung_empty=len(e), rung_empty_max_donors=float(e.n_donors.max()),
                      rung_estimable_min_donors=float(u[~u.rung_total_empty].n_donors.min()),
                      rung_empty_at_or_above_trc_min_donors=int((e.n_donors >= TRC_MIN_DONORS).sum()),
                      rung_empty_mixqtl_trc_finite=len(mix), rung_empty_mixqtl_trc_min_donors=float(mix.n_donors.min()))
    return out


def errors_and_common(CL, u):
    """Per arm and channel, slope minus its estimand at each causal unit (log2); per channel, the common unit set."""
    E = {}
    for a, (Cz, _) in CL.items():
        T = SC.channel_truths(Cz, 'unit' if a in RUNGS else a)
        E[a] = {ch: Cz[SC.SLOPE[ch][0]].astype(float).values - T[ch] for ch in CHANNELS}
    common = {ch: np.logical_and.reduce([np.isfinite(E[a][ch]) for a in CL]) for ch in CHANNELS}
    ok = u[~u.is_null].groupby(['rep', 'gene']).identified.first()
    Cu = CL['unit'][0]
    common['total'] &= ok.loc[list(zip(Cu.rep, Cu.gene))].values
    return E, common


def sq_ratio(e, eu, keep, g, genes, bsel, bidx):
    """06_score's ratio sum e^2 / sum eu^2 over the units keep, per band, with its gene-clustered interval."""
    m, a, u = SC.gene_sums(g, len(genes), keep, e ** 2, eu ** 2)
    return {bn: dict(**SC.boot_stat((a, u), gs, bidx[bn], SC.sum_ratio), units=int(m[gs].sum()))
            for bn, gs in bsel.items() if m[gs].sum() > 0}


def ladder_scores(CL, E, common, i, genes, bsel, bidx):
    """Squared-error ratio vs unit on the common set (headline) and on each arm's own pairing with unit, and AUC."""
    Cu = CL['unit'][0]
    g = pd.Index(genes).get_indexer(Cu.gene)
    out = {}
    for a in ARMS:
        Cz, L = CL[a]
        own = SC.nonnull_precision(Cz, Cu, 'unit' if a in RUNGS else a, genes)
        out[a] = dict(common_set={ch: sq_ratio(E[a][ch], E['unit'][ch], common[ch], g, genes, bsel, bidx) for ch in CHANNELS},
                      own_set={ch: SC.summarize(own[ch], bsel, bidx)['ratio_vs_unit'] for ch in CHANNELS},
                      auc=SC.ranking(L, (SC.AUC_BOOT_KEY, i))['auc'])
    return out


def corr(n, sx, sy, sxx, syy, sxy):
    with np.errstate(divide='ignore', invalid='ignore'):
        return (sxy - sx * sy / n) / np.sqrt((sxx - sx ** 2 / n) * (syy - sy ** 2 / n))


def total_summary(T, Cu, common, genes, bsel, bidx):
    """The total-channel decomposition on the common total set, and as its chance level the R^2 at the null genes
    identified by the same rule."""
    N = T[T.is_null & T.identified]
    K = Cu[['rep', 'gene', 'slope_t', 'total_truth_pipeline']].assign(common=common)
    T = T[~T.is_null].merge(K, on=['rep', 'gene'], how='left', validate='1:1')
    fin, truth, g = T.common.values, T.total_truth * LN2, pd.Index(genes).get_indexer(T.gene)
    out = {c: SC.ratio_block((T[c] / truth).where(fin), T, genes, bsel, bidx) for c in TRC}
    out['predicted_1_minus_r2'] = SC.ratio_block((1 - T.r2).where(fin), T, genes, bsel, bidx)
    x, y = (1 - T.r2).values, (T.mixqtl_trc / truth).values
    parts = SC.gene_sums(g, len(genes), fin, x, y, x * x, y * y, x * y)
    out['pearson_ratio_vs_1_minus_r2'] = dict(**SC.boot_stat(parts, bsel['all'], bidx['all'], corr), units=int(fin.sum()))
    eu = T.slope_t.astype(float).values - T.total_truth_pipeline.values
    out['sq_error_vs_unit'] = {c: sq_ratio(T[c].values / LN2 - T.total_truth.values, eu, fin, g, genes, bsel, bidx) for c in TRC}
    s = T[fin]
    out['selection'] = dict(n_selected_min_median_max=[float(s.n_selected.min()), float(s.n_selected.median()), float(s.n_selected.max())],
                            n_donors_min_median=[float(s.n_donors.min()), float(s.n_donors.median())],
                            mean_r2=float(s.r2.mean()), null_gene_r2=SC.ratio_block(N.r2, N, genes, bsel, bidx)['all'])
    return out


def main():
    if C.LADDER is None:
        print(f'skipped: gene set {C.GENE_SET} has no ladder (common.GENE_SETS[{C.GENE_SET!r}][\'ladder\'] is None); nothing written')
        return
    meta, genes, U, keep_a = SC.load_units(C.DATASETS, C.RESULTS)
    U = U[U.beta_abs > 0]
    S = C.setup(C.load()[0])
    I = S['I']
    if S['genes'] != genes:
        raise SystemExit('loader genes differ from the datasets')
    scen = [f'beta{b}' for b in meta['betas']]
    scratch, units = C.LADDER / 'scratch', []
    for i, (sc, r) in enumerate(U[['scenario', 'rep']].drop_duplicates().itertuples(index=False)):
        ds = C.load_dataset(C.DATASETS, sc, r)
        for rung, cutoff in RUNGS.items():
            T = total_channel(S, ds, cutoff)
            if i == 0:
                gate_trc(T, sc, r, cutoff)
            df, T['rung_total_empty'], msg = run_rung(S, ds, cutoff, scratch)
            C.write_parquet(df, C.LADDER / sc / rung / f'nominal_rep{r:03d}.parquet', C.fingerprint(ds, rung), 'log2')
            print(f'{sc} rep {r:03d} {rung}: map_nominal {len(df):,} rows; {msg}', flush=True)
            units.append(T.assign(scenario=sc, rep=r))
    shutil.rmtree(scratch)
    units = pd.concat(units, ignore_index=True).merge(U[['scenario', 'rep', 'gene', 'is_null', 'band', 'total_truth']],
                                                      on=['scenario', 'rep', 'gene'], how='left', validate='m:1')
    C.write_atomic(C.LADDER / 'total_channel_units.tsv', lambda fh: units.to_csv(fh, sep='\t', index=False), 'w')
    same = units[~units.is_null & units.same_donor_sets & np.isfinite(units.mixqtl_trc) & np.isfinite(units.one_step_trc)]
    dev = np.abs(same.mixqtl_trc - (1 - same.r2) * same.one_step_trc) / np.abs(same.mixqtl_trc)
    if dev.max() > FWL_TOL:
        raise SystemExit(f'FWL identity off by {dev.max():.1e} (FWL_TOL {FWL_TOL}) in {int((dev > FWL_TOL).sum())} units')
    bsel, bidx = SC.band_selections(genes, U, keep_a)
    bsel.pop(SC.NO_ONE_DF), bidx.pop(SC.NO_ONE_DF)   # the ladder reports the four read bands
    R = dict(datasets=str(C.DATASETS), results=str(C.RESULTS), n_genes=len(genes),
             n_covariates=I['cov_df'].shape[1] + I['geno_cov_df'].shape[1],
             identity=dict(units=len(same), max_rel_dev=float(dev.max())), estimability=estimability(units),
             common_units={}, total_channel={c: {} for c in CUTOFFS}, ladder={})
    for sc in sorted(U.scenario.unique()):
        CL = {a: SC.causal_and_leads(C.LADDER if a in RUNGS else C.RESULTS, U, sc, a) for a in ARMS}
        u = units[units.scenario == sc]
        u = u.assign(identified=np.isfinite(u[['one_step_trc', 'one_step_all_trc']]).all(axis=1).groupby([u.rep, u.gene]).transform('all'))
        E, common = errors_and_common(CL, u)
        R['common_units'][sc] = dict(non_null=len(CL['unit'][0]), **{ch: int(common[ch].sum()) for ch in CHANNELS})
        R['ladder'][sc] = ladder_scores(CL, E, common, scen.index(sc), genes, bsel, bidx)
        for cutoff in CUTOFFS:
            t = total_summary(u[u.cutoff == cutoff], CL['unit'][0], common['total'], genes, bsel, bidx)
            a, b = t['sq_error_vs_unit']['mixqtl_trc'], R['ladder'][sc][MIXQTL_ARM[cutoff]]['common_set']['total']
            if 'all' in a and not np.isclose(a['all']['value'], b['all']['value'], rtol=XCHECK_RTOL, atol=0):
                raise SystemExit(f'{sc} {cutoff}: mixQTL total squared-error ratio differs between the two computations')
            R['total_channel'][cutoff][sc] = t
        ratios = {a: '/'.join(f'{R["ladder"][sc][a]["common_set"][ch]["all"]["value"]:.3f}' for ch in CHANNELS) for a in ARMS[1:]}
        print(f'{sc}: common units combined / allelic / total {R["common_units"][sc]}; squared error vs unit {ratios}', flush=True)
    C.write_json(C.LADDER / 'ladder.json', R)
    print(f'FWL identity: {len(same)} units, max relative deviation {dev.max():.1e}; wrote {C.LADDER / "ladder.json"}, '
          f'{len(units)} gene x dataset x cutoff units')


if __name__ == '__main__':
    main()

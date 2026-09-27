"""Why mixQTL mode trails the unit-weight arm on the plasmode datasets: one-change rungs between
them, measured on the committed full run (datasets/ from make_datasets.py, results/ from run_arms.py).

COMMON UNIT SET. Every squared-error ratio is scored on ONE set of causal units (a non-null gene at
its causal variant, per dataset) per scenario and channel: the units where unit, both cutoff rungs
and both mixQTL arms have a finite error (score.py channel_truths), and for the total channel also
both one-step slopes below at both cutoffs. That is the headline. Each arm paired with unit alone
(score.py nonnull_precision, which drops different units for different arms) is kept as a secondary
'own set' column.

TOTAL CHANNEL. mixQTL's trc (tensorqtl/mixqtl_replication.py) first fits covariate_offset:
log(yT / L / 2) on the covariates WITHOUT the genotype, |t| > 2 selection, refit on the selected set
S. It then regresses the offset response on x = (h1 + h2)/2 WITHOUT residualizing x on S. By
Frisch-Waugh-Lovell the one-step slope (y on [1, x, C_S]) is the slope on x residualized on
[1, C_S], and the two-step slope is that times 1 - R^2(x ~ [1, C_S]) when both steps use the same
donors. The offset is fitted on yT > 0 and trc on yT >= trc_cutoff, so the identity is exact only
where the two donor sets coincide (enforced there to FWL_TOL). Per dataset x gene at its causal
variant, at each cutoff setting:
  mixqtl_trc        MX.trc_channel as mixqtl_scan calls it; gated against the committed slope_t on
                    the first dataset
  one_step_trc      the same call given x residualized on [1, C_S] over the trc donors: exactly the
                    one-step least-squares slope over those donors (the offset drops out)
  one_step_all_trc  the same with every covariate, [1, C]: no selection on the outcome
                    Each one-step is NaN where trc donors < 2 + covariates + 1, where the slope has
                    no residual degree of freedom.
  r2, n_selected, n_donors  R^2 of x on [1, C_S] over the trc donors (NaN where x is constant
                    there), |S|, and that donor count
On the common set the total channel reads one change per step: the cutoff rung (unit weights,
log2(CPM + 1), all covariates, the trc donors); one_step_all (mixQTL's response log(yT / 2L) and the
count-scale truth); one_step (covariates selected on the outcome, which carries the injected effect);
mixqtl_trc (x not residualized on S). Reported per scenario: mean slope / truth (truth = count-scale
total_truth x ln 2, natural log like mixQTL's slope) and mean 1 - R^2, with score.py's gene-clustered
interval; the squared error against the committed unit arm (score.py (2b): sum (slope - truth)^2 /
the same sum under unit at its pipeline-scale truth, log2); the Pearson r across units of
mixqtl_trc / truth with 1 - R^2, which is NOT independent evidence (on same-donor units the ratio is
(1 - R^2) x one_step / truth exactly); and as the chance level of R^2 its mean at the NULL genes'
causal variants (drawn for every gene; those identified by the same rule, as R^2 over few donors is
large by chance), where selection sees no genotype effect but the genotype PCs, which stay with the
genotypes, can still correlate with x.

DONOR SET. Rungs from run_arms' 'unit' arm toward mixQTL, one change each: unit weights with
mixQTL's count cutoffs as donor admission (hapmixqtl.count_cutoff_masks on the dataset's thinned
point estimates, yT = pT, passed to map_nominal as keep_a_df / keep_t_df), at the published and the
permissive settings; mixqtl and mixqtl_permissive from the committed results are the far ends.
Also scored by the within-dataset AUC (score.ranking, score.main's bootstrap keys). A channel with
no information at a variant comes back from map_nominal as slope 0, se inf, p 1; the rung files
store it as NaN, the mixQTL arms' form of "no estimate".
CAVEATS. The cutoffs change which GENES are estimable, not only which donors enter: the rung's total
channel is map_nominal's joint design (intercept, n_cov = 17 covariates, genotype), with no residual
degree of freedom at admitted trc donors <= 2 + n_cov = 19, while mixQTL's trc (intercept and x only)
estimates from 3; ESTIMABILITY prints the boundary as measured from the rung files (full run: at
published cutoffs the rung is empty in 99 of 900 gene-datasets, all at <= 19 donors, 61 of them at
3-19, where mixQTL's trc has a causal-variant slope in 52; permissive 5, all 5 estimated by
mixQTL). mixQTL's own set keeps units the rung drops, and on the full run those carry most of
mixQTL's own-set total squared error, which is why the common set is the headline. map_nominal
refers every p to t with N - 2 - max(n_cov, n_cov_a) = 73 dof whatever the mask keeps
(hapmixqtl.py:1873; its fitted scale counts only kept donors, _wls_regression's dof_f), so a rung's
p, and so its AUC, is too small where the cutoffs leave few donors; the squared-error ratio uses no
p. The allelic pipeline truth is the slope over the records allelic_kept admits without cutoffs, so
a cutoff rung's allelic estimand is shifted by the admitted band, direction not fixed at published
cutoffs ([50, 1000] drops both the most-attenuated low records and the least-attenuated high ones);
admission itself depends on the injected effect (run_arms' asc_cap CAVEAT).

SMOKE = True: one dataset, SMOKE_GENES of its genes, loaded through a gene list and regions file
written under LADDER/smoke (the tested set then excludes only those genes' bodies, a superset of the
full run's; causal-variant metrics are unaffected). Output: LADDER/<scenario>/<rung>/
nominal_repNNN.parquet, LADDER/total_channel_units.tsv, LADDER/ladder.json (atomic, NaN as null).
"""
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import compare_mixqtl_replication as CM                                  # noqa: E402
import make_datasets as MD                                               # noqa: E402
import run_arms as RA                                                    # noqa: E402
import score as SC                                                       # noqa: E402
from tensorqtl import mixqtl_replication as MX                           # noqa: E402
from tensorqtl.hapmixqtl import LN2, count_cutoff_masks, map_nominal    # noqa: E402

SMOKE = False
SMOKE_SET = ('beta0.8', 0)        # one dataset with both null and non-null genes
SMOKE_GENES = (3, 2)              # non-null, null genes of SMOKE_SET
LADDER = MD.ROOT / 'ladder' / ('smoke' if SMOKE else '')
CUTOFFS = {'published': MX.PUBLISHED_CUTOFFS, 'permissive': MX.PACKAGE_DEFAULT_CUTOFFS}   # run_arms' two mixQTL settings
MIXQTL_ARM = {'published': 'mixqtl', 'permissive': 'mixqtl_permissive'}                   # run_arms.MIXQTL_ARMS
RUNGS = {f'unit_{c}_cutoffs': c for c in CUTOFFS}
ARMS = ('unit', 'unit_published_cutoffs', 'mixqtl', 'unit_permissive_cutoffs', 'mixqtl_permissive')
CHANNELS = ('combined', 'allelic', 'total')
TRC = ('mixqtl_trc', 'one_step_trc', 'one_step_all_trc')
NO_ESTIMATE = (('slope', 'slope_se', 'pval_nominal'), ('slope_a', 'slope_a_se', 'pval_a'), ('slope_t', 'slope_t_se', 'pval_t'))
TRC_MIN_DONORS = 3                # mixqtl_replication.trc_channel returns no slope at n <= 2 donors
FWL_TOL = 1e-8                    # review threshold; the 2026-09-26 full run measured 8.4e-12
XCHECK_RTOL = 1e-9                # RA.IDENTITY_RTOL: the committed and recomputed mixQTL trc slopes agree to this


def load():
    """Committed units (beta > 0), the loader inputs as run_arms.setup builds them, and each dataset file."""
    meta, genes, U, _ = SC.load_units(RA.DATASETS, RA.RESULTS)
    U = U[U.beta_abs > 0]
    gl, rg = MD.GENES, MD.REGIONS
    if SMOKE:
        u = U[(U.scenario == SMOKE_SET[0]) & (U.rep == SMOKE_SET[1])]
        pick = set(u[~u.is_null].gene[:SMOKE_GENES[0]]) | set(u[u.is_null].gene[:SMOKE_GENES[1]])
        genes = [g for g in genes if g in pick]
        U = u[u.gene.isin(pick)]
        LADDER.mkdir(parents=True, exist_ok=True)
        gl, rg = LADDER / 'genes.txt', LADDER / 'regions.bed'
        bed = pd.read_csv(MD.REGIONS, sep='\t', header=None)
        MD.write_atomic(gl, lambda fh: fh.write('\n'.join(genes) + '\n'), 'w')
        MD.write_atomic(rg, lambda fh: bed[bed[3].isin(pick)].to_csv(fh, sep='\t', header=False, index=False), 'w')
    S = RA.setup(CM.load_point_estimate_inputs(gene_list=str(gl), regions=str(rg)))
    if S['genes'] != genes:
        raise SystemExit('loader genes differ from the committed datasets\' genes')
    I = S['I']
    if not (np.isfinite(I['xL'][I['idx']]).all() and np.isfinite(I['xR'][I['idx']]).all()):
        raise SystemExit('missing phased genotypes in the tested set (mixqtl_scan would impute 0.5)')
    return meta, S, U


def dataset(meta, S, sc, r):
    ds = dict(np.load(RA.DATASETS / sc / f'rep{r:03d}.npz'))
    rows = [meta['genes'].index(g) for g in S['genes']]
    return {k: v if k in ('eff_lib', 'perm', 'swap') else v[rows] for k, v in ds.items()}


def residualize(x, D, K):
    """x minus its least-squares fit on [1, D] over donors K, and R^2 of that fit over K (NaN if x is constant on K)."""
    D = np.column_stack([np.ones(len(x)), D])
    r = x - D @ np.linalg.lstsq(D[K], x[K], rcond=None)[0]
    xc = x[K] - x[K].mean() if K.any() else x[K]
    return r, (1.0 - (r[K] @ r[K]) / (xc @ xc) if xc @ xc > 0 else np.nan)


def total_channel(S, ds, cutoff):
    """At every gene's causal variant: mixQTL's trc slope, the two one-step slopes, R^2, |S| and donors."""
    I, tc = S['I'], CUTOFFS[cutoff]['trc_cutoff']
    _, _, yt = MX.inputs_from_point_estimates(ds['pL'], ds['pR'], ds['pT'])
    C = MX._stack_covariates(I['cov_df'].values[ds['perm']], I['geno_cov_df'].values)
    rec = []
    for k, g in enumerate(S['genes']):
        v = str(ds['causal_variant'][k])
        x = (I['xL'][S['rows'][v]] + I['xR'][S['rows'][v]]) / 2.0
        off, sel = MX.covariate_offset(yt[k], ds['eff_lib'], C)
        K = yt[k] >= tc                   # trc_channel's donors (tc > 0, so the log response is finite)
        n = int(K.sum())
        fit = lambda z: MX.trc_channel(yt[k], ds['eff_lib'], z[:, None], off, tc)['beta'][0]   # noqa: E731
        x_sel, r2 = residualize(x, C[:, sel], K)
        x_all, _ = residualize(x, C, K)
        rec.append(dict(gene=g, variant_id=v, cutoff=cutoff, mixqtl_trc=fit(x),
                        one_step_trc=fit(x_sel) if n - 2 - sel.sum() >= 1 else np.nan,
                        one_step_all_trc=fit(x_all) if n - 2 - C.shape[1] >= 1 else np.nan,
                        r2=r2, n_selected=int(sel.sum()), n_donors=n, same_donor_sets=bool(n == (yt[k] > 0).sum())))
    return pd.DataFrame(rec)


def gate_trc(T, sc, r, cutoff):
    """mixqtl_trc must reproduce the committed mixQTL arm's slope_t (natural log) at the causal variant."""
    p = RA.RESULTS / sc / MIXQTL_ARM[cutoff] / f'nominal_rep{r:03d}.parquet'
    d = pd.read_parquet(p, columns=['phenotype_id', 'variant_id', 'slope_t'])
    d['variant_id'] = d['variant_id'].astype(str)
    want = T.merge(d, left_on=['gene', 'variant_id'], right_on=['phenotype_id', 'variant_id'], how='left').slope_t
    ok = np.isclose(T.mixqtl_trc.values, want.values, rtol=RA.IDENTITY_RTOL, atol=0, equal_nan=True)
    if not ok.all():
        raise SystemExit(f'{p}: trc_channel differs from the committed slope_t in {int((~ok).sum())} genes, '
                         f'e.g. {T.gene[~ok].tolist()[:3]}')
    print(f'{sc} rep {r:03d} {cutoff}: trc_channel equals the committed {MIXQTL_ARM[cutoff]} slope_t in all '
          f'{len(T)} genes', flush=True)


def run_rung(S, ds, cutoff, scratch):
    """map_nominal as run_arms runs the 'unit' arm, plus mixQTL's count cutoffs as donor admission."""
    A, T, Va, Vt, cov, _ = RA.inputs(S, ds, 'unit')
    c = CUTOFFS[cutoff]
    ka, kt = count_cutoff_masks(ds['pL'], ds['pR'], yT=ds['pT'], asc_cutoff=c['asc_cutoff'],
                                asc_cap=c['asc_cap'], trc_cutoff=c['trc_cutoff'])
    frame = lambda M: pd.DataFrame(M, index=A.index, columns=A.columns)   # noqa: E731
    scratch.mkdir(parents=True, exist_ok=True)
    for q in scratch.glob('*'):
        q.unlink()
    RA.quiet(map_nominal, S['gdf'], S['vdf'][['chrom', 'pos']], A, T, Va, Vt, S['gp'],
             xL_df=S['xLdf'], xR_df=S['xRdf'], prefix='n', covariates_df=cov,
             genotype_covariates_df=S['I']['geno_cov_df'], window=CM.WIN, output_dir=str(scratch),
             verbose=False, ase_covariates_df=None, keep_a_df=frame(ka), keep_t_df=frame(kt))
    df = pd.concat([pd.read_parquet(q, columns=SC.CNS.COLS) for q in sorted(scratch.glob('n*.parquet'))],
                   ignore_index=True)
    df['variant_id'] = df['variant_id'].astype(str)
    empty = []
    for b, s, p in NO_ESTIMATE:
        m = np.isinf(df[s].values)
        df.loc[m, [b, s, p]] = np.nan
        empty.append(f'{b} {int(m.sum()):,} rows in {df.phenotype_id[m].nunique()} genes')
    total_empty = df.slope_t.isna().groupby(df.phenotype_id).all().loc[S['genes']].values
    informative = Va.values > 0
    return df, total_empty, (f'map_nominal {len(df):,} rows; allelic pairs admitted {int((ka & informative).sum())} '
                             f'of {int(informative.sum())}, total pairs admitted {int(kt.sum())} of {kt.size}; '
                             f'se inf (stored NaN): {", ".join(empty)}')


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
    """Per arm and channel, slope minus its estimand at each causal unit (score.py channel_truths, log2); per channel,
    the common unit set: every arm's error finite, and for total both one-step slopes identified at both cutoffs."""
    E = {}
    for a, (C, _) in CL.items():
        T = SC.channel_truths(C, 'unit' if a in RUNGS else a)
        E[a] = {ch: C[SC.SLOPE[ch][0]].astype(float).values - T[ch] for ch in CHANNELS}
    common = {ch: np.logical_and.reduce([np.isfinite(E[a][ch]) for a in CL]) for ch in CHANNELS}
    ok = u[~u.is_null].groupby(['rep', 'gene']).identified.first()
    Cu = CL['unit'][0]
    common['total'] &= ok.loc[list(zip(Cu.rep, Cu.gene))].values
    return E, common


def sq_ratio(e, eu, keep, g, genes, bsel, bidx):
    """score.py's (2b) ratio sum e^2 / sum eu^2 over the units keep, per band, with its gene-clustered interval."""
    m, a, u = SC.gene_sums(g, len(genes), keep, e ** 2, eu ** 2)
    return {bn: dict(**SC.boot_stat((a, u), gs, bidx[bn], SC.sum_ratio), units=int(m[gs].sum()))
            for bn, gs in bsel.items() if m[gs].sum() > 0}


def ladder_scores(CL, E, common, i, genes, bsel, bidx):
    """Squared-error ratio vs unit on the common unit set (headline) and on each arm's own pairing with unit, and AUC."""
    Cu = CL['unit'][0]
    g = pd.Index(genes).get_indexer(Cu.gene)
    out = {}
    for a in ARMS:
        C, L = CL[a]
        own, exc = SC.nonnull_precision(C, Cu, 'unit' if a in RUNGS else a, genes)
        out[a] = dict(common_set={ch: sq_ratio(E[a][ch], E['unit'][ch], common[ch], g, genes, bsel, bidx) for ch in CHANNELS},
                      own_set={ch: SC.summarize(own[ch], bsel, bidx)['ratio_vs_unit'] for ch in CHANNELS},
                      own_set_excluded=exc, auc=SC.ranking(L, (SC.AUC_BOOT_KEY, i))['auc'])
    return out


def corr(n, sx, sy, sxx, syy, sxy):
    with np.errstate(divide='ignore', invalid='ignore'):
        return (sxy - sx * sy / n) / np.sqrt((sxx - sx ** 2 / n) * (syy - sy ** 2 / n))


def total_summary(T, Cu, common, genes, bsel, bidx):
    """The total-channel decomposition on the common total set, and as its chance level the R^2 at the null genes
    identified by the same rule (one-step slopes at both cutoffs), since R^2 over few donors is large by chance."""
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
    out['sq_error_vs_unit'] = {c: sq_ratio(T[c].values / LN2 - T.total_truth.values, eu, fin, g, genes, bsel, bidx)
                               for c in TRC}
    s = T[fin]
    out['selection'] = dict(n_selected_min_median_max=[float(s.n_selected.min()), float(s.n_selected.median()),
                                                       float(s.n_selected.max())],
                            n_donors_min_median=[float(s.n_donors.min()), float(s.n_donors.median())],
                            mean_r2=float(s.r2.mean()), null_gene_r2=SC.ratio_block(N.r2, N, genes, bsel, bidx)['all'])
    return out


def fmt(d, key='mean'):
    """'all' with its interval, then the three read-depth bands; a band is absent when it has no units."""
    if 'all' not in d:
        return 'no units'
    b = lambda k: f'{d[k][key]:.3f}' if k in d else '-'   # noqa: E731
    return (f'{d["all"][key]:.3f} [{d["all"]["lo"]:.3f}, {d["all"]["hi"]:.3f}] '
            f'({b("<100")}/{b("100-999")}/{b(">=1000")})')


def fmtv(d):
    return f'{d["all"]["value"]:.3f} [{d["all"]["lo"]:.3f}, {d["all"]["hi"]:.3f}]' if 'all' in d else 'no units'


def report(R):
    nc = R['n_covariates']
    print('\nCOMMON UNIT SETS (every squared-error ratio below is scored on these): causal units where unit, both '
          'cutoff rungs and both mixQTL arms have a finite error; total also needs both one-step slopes identified '
          'at both cutoffs')
    for sc, n in R['common_units'].items():
        print(f'  {sc}: combined {n["combined"]} / allelic {n["allelic"]} / total {n["total"]} of {n["non_null"]} '
              f'non-null units')
    print(f'\nESTIMABILITY of the total channel over every gene x dataset: the rung\'s joint design (intercept, {nc} '
          f'covariates, genotype) has no residual dof at <= {2 + nc} admitted donors; mixQTL\'s trc needs 3')
    for c, e in R['estimability'].items():
        print(f'  {c}: rung total channel empty in {e["rung_empty"]} of {e["gene_datasets"]} gene-datasets (admitted '
              f'trc donors at most {e["rung_empty_max_donors"]:.0f} among them, at least '
              f'{e["rung_estimable_min_donors"]:.0f} among the estimable), {e["rung_empty_at_or_above_trc_min_donors"]} of them with '
              f'>= {TRC_MIN_DONORS} donors; mixQTL\'s trc has a causal-variant slope in {e["rung_empty_mixqtl_trc_finite"]} '
              f'(from {e["rung_empty_mixqtl_trc_min_donors"]:.0f} donors)')
    for cutoff in CUTOFFS:
        print(f'\nTOTAL CHANNEL, {cutoff} cutoffs (trc_cutoff {CUTOFFS[cutoff]["trc_cutoff"]:.0f}), common total set: '
              f'mean slope / count-scale total truth at the causal variant [gene-clustered 95%] (<100 / 100-999 / '
              f'>=1000 reads)')
        for sc, t in R['total_channel'][cutoff].items():
            s, p, q, nb = t['selection'], t['pearson_ratio_vs_1_minus_r2'], t['sq_error_vs_unit'], t['selection']['null_gene_r2']
            rung = R['ladder'][sc][f'unit_{cutoff}_cutoffs']['common_set']['total']
            print(f'  {sc} mixQTL trc {fmt(t["mixqtl_trc"])}; 1 - R^2 {fmt(t["predicted_1_minus_r2"])}; one-step '
                  f'selected {fmt(t["one_step_trc"])}; one-step all {nc} {fmt(t["one_step_all_trc"])}')
            print(f'  {sc}   r(mixQTL trc / truth, 1 - R^2) {p["value"]:.3f} [{p["lo"]:.3f}, {p["hi"]:.3f}] over '
                  f'{p["units"]} units (partly algebraic); |S| min/median/max '
                  f'{"/".join(f"{v:.0f}" for v in s["n_selected_min_median_max"])} of {nc}; trc donors min/median '
                  f'{"/".join(f"{v:.0f}" for v in s["n_donors_min_median"])}; mean R^2 {s["mean_r2"]:.3f} against '
                  f'{nb["mean"]:.3f} [{nb["lo"]:.3f}, {nb["hi"]:.3f}] at identified null genes\' causal variants ({nb["units"]} units)')
            print(f'  {sc}   total squared error / unit\'s over {p["units"]} units: rung {fmtv(rung)} -> one-step all '
                  f'{nc} {fmtv(q["one_step_all_trc"])} -> one-step selected {fmtv(q["one_step_trc"])} -> mixQTL trc '
                  f'{fmtv(q["mixqtl_trc"])}')
    print(f'  FWL identity where the offset and trc donor sets coincide: {R["identity"]["units"]} non-null units, max '
          f'|mixQTL trc - (1 - R^2) one-step| / |mixQTL trc| {R["identity"]["max_rel_dev"]:.1e}')
    print('\nLADDER: squared-error ratio vs unit on the common unit set, combined / allelic / total [gene-clustered '
          '95%]; own set = each arm paired with unit alone (unpaired units); within-dataset AUC mean [dataset-bootstrap 95%]')
    for sc, arms in R['ladder'].items():
        for a, d in arms.items():
            own = '/'.join(f'{d["own_set"][ch]["all"]["value"]:.3f}' if 'all' in d['own_set'][ch] else '-' for ch in CHANNELS)
            auc = d['auc']['all']
            print(f'  {sc} {a:24s} {" / ".join(fmtv(d["common_set"][ch]) for ch in CHANNELS)}  own set {own} '
                  f'({"/".join(str(d["own_set_excluded"][ch]["unpaired"]) for ch in CHANNELS)})  AUC '
                  f'{auc["mean"]:.3f} [{auc["lo"]:.3f}, {auc["hi"]:.3f}]')


def main():
    meta, S, U = load()
    genes = S['genes']
    scen = [f'beta{b}' for b in meta['betas']]
    scratch = LADDER / 'scratch'
    units = []
    for i, (sc, r) in enumerate(U[['scenario', 'rep']].drop_duplicates().itertuples(index=False)):
        ds = dataset(meta, S, sc, r)
        for rung, cutoff in RUNGS.items():
            T = total_channel(S, ds, cutoff)
            if i == 0:
                gate_trc(T, sc, r, cutoff)
            df, T['rung_total_empty'], msg = run_rung(S, ds, cutoff, scratch)
            out = LADDER / sc / rung
            out.mkdir(parents=True, exist_ok=True)
            RA.write_parquet(df, out / f'nominal_rep{r:03d}.parquet', RA.fingerprint(ds, rung), 'log2')
            print(f'{sc} rep {r:03d} {rung}: {msg}', flush=True)
            units.append(T.assign(scenario=sc, rep=r))
    shutil.rmtree(scratch)
    units = pd.concat(units, ignore_index=True).merge(U[['scenario', 'rep', 'gene', 'is_null', 'band', 'total_truth']],
                                                      on=['scenario', 'rep', 'gene'], how='left', validate='m:1')
    MD.write_atomic(LADDER / 'total_channel_units.tsv', lambda fh: units.to_csv(fh, sep='\t', index=False), 'w')
    same = units[~units.is_null & units.same_donor_sets & np.isfinite(units.mixqtl_trc) & np.isfinite(units.one_step_trc)]
    dev = np.abs(same.mixqtl_trc - (1 - same.r2) * same.one_step_trc) / np.abs(same.mixqtl_trc)
    if dev.max() > FWL_TOL:
        raise SystemExit(f'FWL identity off by {dev.max():.1e} (FWL_TOL {FWL_TOL}) in {int((dev > FWL_TOL).sum())} units')
    bsel = SC.band_genes(genes, U)
    bidx = {bn: SC.boot((SC.GENE_BOOT_KEY, b), len(g)) for b, (bn, g) in enumerate(bsel.items())}
    R = dict(datasets=str(RA.DATASETS), results=str(RA.RESULTS), smoke=SMOKE, n_genes=len(genes),
             n_covariates=S['I']['cov_df'].shape[1] + S['I']['geno_cov_df'].shape[1],
             identity=dict(units=len(same), max_rel_dev=float(dev.max())), estimability=estimability(units),
             common_units={}, total_channel={c: {} for c in CUTOFFS}, ladder={})
    for sc in sorted(U.scenario.unique()):
        CL = {a: SC.causal_and_leads(LADDER if a in RUNGS else RA.RESULTS, U, sc, a) for a in ARMS}
        u = units[units.scenario == sc]
        u = u.assign(identified=np.isfinite(u[['one_step_trc', 'one_step_all_trc']]).all(axis=1)
                     .groupby([u.rep, u.gene]).transform('all'))
        E, common = errors_and_common(CL, u)
        R['common_units'][sc] = dict(non_null=len(CL['unit'][0]), **{ch: int(common[ch].sum()) for ch in CHANNELS})
        R['ladder'][sc] = ladder_scores(CL, E, common, scen.index(sc), genes, bsel, bidx)
        for cutoff in CUTOFFS:
            t = total_summary(u[u.cutoff == cutoff], CL['unit'][0], common['total'], genes, bsel, bidx)
            a, b = t['sq_error_vs_unit']['mixqtl_trc'], R['ladder'][sc][MIXQTL_ARM[cutoff]]['common_set']['total']
            if 'all' in a and not np.isclose(a['all']['value'], b['all']['value'], rtol=XCHECK_RTOL, atol=0):
                raise SystemExit(f'{sc} {cutoff}: mixQTL total squared-error ratio {a["all"]["value"]} (TOTAL CHANNEL) '
                                 f'against {b["all"]["value"]} (LADDER) on the same common set')
            R['total_channel'][cutoff][sc] = t
    MD.write_atomic(LADDER / 'ladder.json', lambda fh: fh.write(MD.dumps(R)), 'w')
    report(R)
    print(f'\nwrote {LADDER / "ladder.json"}, {LADDER / "total_channel_units.tsv"}; {len(units)} gene x dataset x cutoff units')


if __name__ == '__main__':
    main()

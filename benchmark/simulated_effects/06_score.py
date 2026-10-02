"""Score the benchmark (README: Scoring): how well each arm recovers the injected effects, the
precision and stated standard error of its slopes, the nominal-p rate on null genes, lead-variant
recovery, gene ranking, and gene-level results: the permutation p of every arm that has one and
eigenMT for all nine.

Arms: the six of 03_run_arms.py plus the joint models scored (common.JOINT: RASQUAL and TReCASE), each one test
per variant scored as its combined channel with truth beta (both put the total mean at 1,
(1 + kappa)/2, kappa, the generator's expected total fold); their slope_se is DERIVED (|slope| /
sqrt(chisq)), so their null sd(z) is the calibration of the likelihood-ratio statistic. The
tensorqtl arm is one test per variant too, scored as its combined channel, whose truth is the total
channel's (count scale: the per-gene total truth; its own scale: the pipeline-scale total truth,
as a hapmixQTL arm's). mixQTL's natural-log slopes are divided by ln 2 on reading
(common.read_results); every slope here is log2. The native-input arms of 05b_native_arms.py (C.NATIVE_ARMS) are
scored where C.NATIVE exists (NATIVE_ARMS; a root without it, such as 99_acceptance.py's, is scored without them and says
so in a printed line), on the count-scale truth only (the Salmon datasets' truths, which depend on the causal genotypes and beta only):
split_native with channels as a hapmixQTL arm and its cis file's pval_beta, its squared error against unit weights on the
count-scale truth for both; trecase_native as TReCASE (JOINT_LIKE: one joint test, truth beta, missing rows allowed). A
gene 05b did not test (native total constant) has no rows in either native arm and ranks as no finite p. Their files
carry the native dataset's fingerprint, and the native dataset must carry the Salmon dataset's permutation, swap, null
set and causal variants. For both TReCASE arms (TRECASE_ARMS; empty when the native arms are not scored) at
|beta| > 0: the share of gene units whose reported lead has no joint fit, and power at 5% realized FDP by each component p.

Units: a causal unit is one (dataset, non-null gene) at its causal variant; a gene unit one
(dataset, gene). Every statistic is pooled over units, overall and by the gene's real median
haplotype-informative reads (BANDS, from GENE_DESIGN), and over every gene but the ONE_DF genes
(NO_ONE_DF: through-origin allelic fit with one residual df). Gene-clustered interval: genes
resampled with replacement N_BOOT times, each carrying all its units, 2.5% and 97.5% quantiles of
the recomputed statistic. Sections: (1) bias ratio = mean slope / truth at the causal variant, on
the count scale (allelic beta, per-gene total truth, combined against beta) and, hapmixQTL arms,
the pipeline scale; (2) precision at the causal variant against the arm's own truth (hapmixQTL
pipeline scale, mixQTL count scale, joint arms beta; combined = the inverse-variance combination
of the channel truths at the unit's own se), and on null genes against 0: sd(z) with z = (slope -
truth) / se, and the paired squared-error ratio against the unit arm (ratio_vs_unit; also with the
count-scale truth for arm and unit alike, ratio_vs_unit_count); (3) lead recovery: lead = smallest
nominal p (ties by |slope / se|), LD r^2 with the causal variant over the 92 donors; (4) detection:
share of causal units with p below DETECT_ALPHAS; (5) ranking of genes within dataset by lead p:
AUC (Mann-Whitney: the share of (non-null, null) gene pairs ranked in the right order, ties one
half), mean over datasets with a dataset-resampling interval, and power at pooled realized
false-discovery proportion FDR; (6) null-gene nominal-p rate at ALPHAS; the beta = 0 anchor
against each hapmixQTL arm's stored 200-permutation null run (ANCHOR: inside the stored central
ANCHOR_CENTRAL of per-permutation rates at ANCHOR_ALPHA; descriptive, one permutation; skipped
with a printed line, anchor null, for a gene set without stored null runs); (7) gene
level, for two gene-level p: the permutation p of 03's cis files (CIS_P: pval_beta for the
hapmixQTL and tensorqtl arms, pval_perm for mixQTL, whose port has no Beta approximation) and
eigenMT's for all nine arms (Davis et al. 2016: min(1, the gene's smallest nominal p x M_eff), M_eff
from C.EIGENMT); for each, the null-gene rate of p < GENE_LEVEL_ALPHA and Benjamini-Hochberg power at
FDR within dataset; and what sets M_eff (review 2026-09-27): per gene, in 03's windows of EIGENMT_WINDOW
tested variants, the median Ledoit-Wolf weight (tensorqtl.eigenmt.lw_shrink, float64 on the CPU) and
the unshrunk count (eigenvalues of the sample correlation of the window's varying variants needed for
EIGENMT_VAR of their sum), and per arm with pval_beta the median of M_eff / beta_shape2 over gene units
(hapmixQTL arms; the tensorqtl cis file has no shape) and of eigenMT p / pval_beta over gene units with
pval_beta in [FDR / genes, FDR), the range of a dataset's Benjamini-Hochberg thresholds. TReCASE's
component tests (TRECASE_PARTS) are scored on the anchor like channels.

Output: SUMMARY (JSON, NaN as null; 08_report.py reads it).
"""
import json

import numpy as np
import pandas as pd
import torch
from scipy.stats import false_discovery_control

import common as C
from tensorqtl import eigenmt

ANCHOR = ({arm: (C.STORED_NULL / 'summary.json', arm) for arm in C.HAPMIX_ARMS}   # arm: (stored null summary, its draws'
          if C.STORED_NULL else None)   # prefix); the 200-permutation null of the GENE_SET genes on this pipeline
N_BOOT = 2000                 # task spec 2026-09-26
ANCHOR_ALPHA = 0.05           # the only alpha with an anchor rule (task spec 2026-09-26)
ANCHOR_CENTRAL = 0.99         # the anchor is ONE permutation: inside the stored central 99% of per-permutation rates
ANCHOR_N_PERM = 200           # draws per arm in the stored null runs
DETECT_ALPHAS = (0.05, 1e-3, 1e-5)   # user decision 2026-09-26
FDR = 0.05                    # user decision 2026-09-26: power where the pooled realized FDP is 0.05
GENE_LEVEL_ALPHA = 0.05
EIGENMT_WINDOW, EIGENMT_VAR = 200, 0.99   # tensorqtl.eigenmt.compute_tests' defaults, as 03 calls it
R2_HIGH = 0.8                 # user decision 2026-09-26
BANDS = {   # (name, lo, hi) on the gene's median haplotype-informative reads over all donors; the first band is every gene
    'corrected_null_store_20260925': (('all', 0, np.inf), ('<100', 0, 100), ('100-999', 100, 1000), ('>=1000', 1000, np.inf)),  # user decision 2026-09-26
    'stratum30_100': (('all', 0, np.inf), ('<30', 0, 30), ('30-50', 30, 50), ('50-100', 50, 100)),   # set 2026-09-27 before scoring: the set's all-donor medians are 0-78 reads (its stratum is defined on the admitted-donor median, 30-100; select_stratum_genes.py)
}[C.GENE_SET]
ONE_DF = 2                    # admitted allelic donors at which the through-origin allelic fit has 1 residual df (review 2026-09-27)
NO_ONE_DF = 'without one-df genes'
GENE_BOOT_KEY, AUC_BOOT_KEY = 30, 33   # spawn keys after 02's 1 / 2 / 3 and 03's 4 / 5; 31 and 32 belonged to interval streams no longer computed
SLOPE = {'combined': ('slope', 'slope_se'), 'allelic': ('slope_a', 'slope_a_se'), 'total': ('slope_t', 'slope_t_se')}
PVAL = C.CHANNELS
TRUTH = {'count': {'combined': 'allelic_truth', 'allelic': 'allelic_truth', 'total': 'total_truth'},
         'pipeline': {'allelic': 'allelic_truth_pipeline', 'total': 'total_truth_pipeline'}}
NULL_COLS = ['phenotype_id', 'variant_id', 'slope', 'slope_se', 'slope_a', 'slope_a_se', 'slope_t', 'slope_t_se']
JOINT_COLS = C.COLS[:5]       # phenotype_id, variant_id, pval_nominal, slope, slope_se
TRECASE_PARTS = {'trec': 'pval_t', 'joint': 'pval_joint', 'ase': 'pval_a'}   # 05_run_trecase.py's columns
NATIVE_ARMS = C.NATIVE_ARMS if C.NATIVE.is_dir() else ()   # scored only where 05b_native_arms.py has written C.NATIVE
TRECASE_ARMS = ('trecase', 'trecase_native') if NATIVE_ARMS else ()   # trecase_parts: read only by 08's native subsection
ARMS = C.ARMS + tuple(C.JOINT) + NATIVE_ARMS
JOINT_LIKE = tuple(C.JOINT) + tuple(a for a in NATIVE_ARMS if a == 'trecase_native')   # one joint test per variant: truth beta, rows may be missing
MISSING_OK = JOINT_LIKE + tuple(a for a in NATIVE_ARMS if a not in JOINT_LIKE)   # arms whose causal or null-gene rows may be missing (a native arm: a gene 05b did not test)
ONE_TEST = JOINT_LIKE + (C.TENSORQTL,)   # one test per variant, scored as the combined channel
CIS_ARMS = C.ARMS + tuple(a for a in NATIVE_ARMS if a == 'split_native')   # the arms with a cis file (03's, and 05b's split_native)
CIS_P = {a: 'pval_perm' if a in C.MIXQTL_ARMS else 'pval_beta' for a in CIS_ARMS}   # each arm's gene-level permutation p in its cis file


def arm_dir(results, sc, arm):
    return (C.NATIVE_RESULTS[arm] if arm in C.NATIVE_RESULTS else C.JOINT[arm] if arm in C.JOINT else results) / sc / arm


def channels(arm, d):
    """The channels scored for an arm: a joint or tensorqtl arm's one test per variant is its combined channel."""
    return {'combined': d['combined']} if arm in ONE_TEST else d


def truth_col(arm, ch, scale):
    """The truth column of an arm's channel on a scale: the tensorqtl arm's combined channel is a total channel."""
    return TRUTH[scale]['total' if arm == C.TENSORQTL else ch]


def boot(key, n):
    return np.random.default_rng(np.random.SeedSequence(C.SEED, spawn_key=key)).integers(0, n, size=(N_BOOT, n))


def load_units(datasets, results):
    """One row per (scenario, dataset, gene): truth and band; every results file must carry its dataset's fingerprint."""
    meta = json.loads((datasets / 'meta.json').read_text())
    genes = meta['genes']
    gd = pd.read_csv(C.GENE_DESIGN, sep='\t').set_index('gene')
    reads = gd.loc[genes, 'median_allele_resolved_reads']
    truth = pd.read_csv(datasets / 'truth.tsv', sep='\t').drop_duplicates('gene').set_index('gene').median_hap_reads_real
    if not np.allclose(truth.values, reads.loc[truth.index].values, rtol=1e-12, atol=0):
        raise SystemExit(f'{C.GENE_DESIGN} median_allele_resolved_reads differs from {datasets / "truth.tsv"}')
    band = pd.cut(reads, [b[1] for b in BANDS[1:]] + [np.inf], right=False, labels=[b[0] for b in BANDS[1:]])
    rows = []
    for sc, r in C.runs(meta):
        ds = C.load_dataset(datasets, sc, r)
        if NATIVE_ARMS:
            nds = C.load_dataset(C.NATIVE_DATASETS, sc, r)
            if not all(np.array_equal(ds[k], nds[k]) for k in ('perm', 'swap', 'is_null', 'causal_variant')):
                raise SystemExit(f'{C.NATIVE_DATASETS} {sc} rep {r}: permutation, swap, null set or causal variants differ from {datasets}')
        for arm in ARMS:
            for prefix in ('nominal',) + (('cis',) if arm in CIS_ARMS else ()):
                p = arm_dir(results, sc, arm) / f'{prefix}_rep{r:03d}.parquet'
                if C.stored_fingerprint(p) != C.fingerprint(nds if arm in NATIVE_ARMS else ds, arm):
                    raise SystemExit(f'{p} does not match its dataset and arm (common.fingerprint)')
        rows.append(pd.DataFrame(dict(scenario=sc, beta_abs=float(sc[4:]), rep=r, gene=genes, band=band.values.astype(str),
                                      reads=reads.values, is_null=ds['is_null'],
                                      causal_variant=ds['causal_variant'].astype(str), beta=ds['beta'],
                                      **{c: ds[c] for c in ('allelic_truth', 'total_truth', 'allelic_truth_pipeline',
                                                            'total_truth_pipeline')})))
    U = pd.concat(rows, ignore_index=True)
    keep_a = gd.loc[genes, 'n_allelic_keep'].values
    print(f'{len(U):,} dataset-gene units, {int((~U.is_null).sum())} non-null; genes per band '
          f'{band.value_counts().reindex([b[0] for b in BANDS[1:]]).to_dict()}; one-df genes '
          f'{[g for g, k in zip(genes, keep_a) if k == ONE_DF]}', flush=True)
    return meta, genes, U, keep_a


def band_selections(genes, U, keep_a):
    """Gene index arrays per band, NO_ONE_DF last so the bands' interval streams are unchanged, and their resampling indices."""
    b = U.drop_duplicates('gene').set_index('gene').band.loc[genes].values
    bsel = {name: np.where((b == name) | (name == 'all'))[0] for name, _, _ in BANDS}
    bsel[NO_ONE_DF] = np.flatnonzero(keep_a != ONE_DF)
    return bsel, {bn: boot((GENE_BOOT_KEY, i), len(g)) for i, (bn, g) in enumerate(bsel.items())}


def pooled(K, n, gsel, bidx):
    """Pooled rate over genes gsel with its gene-clustered interval."""
    Kg, ng = K[:, gsel].sum(0), n[:, gsel].sum(0)
    b = Kg[bidx].sum(1) / ng[bidx].sum(1)
    return dict(rate=float(Kg.sum() / ng.sum()), lo=float(np.quantile(b, .025)), hi=float(np.quantile(b, .975)),
                rejections=int(Kg.sum()), tests=int(ng.sum()))


def null_calibration(results, U, sc, arm, genes, bsel, bidx, cols=None):
    reps = sorted(U[U.scenario == sc].rep.unique())
    res = {}
    for ch, col in (cols or channels(arm, PVAL)).items():
        K = {al: np.zeros((len(reps), len(genes))) for al in C.ALPHAS}
        n = np.zeros((len(reps), len(genes)))
        for i, r in enumerate(reps):
            nulls = U[(U.scenario == sc) & (U.rep == r) & U.is_null].gene.tolist()
            k, n[i] = C.rates_by_gene([arm_dir(results, sc, arm) / f'nominal_rep{r:03d}.parquet'], genes, col,
                                          gene_filter=nulls)
            for al in C.ALPHAS:
                K[al][i] = k[al]
        res[ch] = {bn: {str(al): pooled(K[al], n, g, bidx[bn]) for al in C.ALPHAS}
                   for bn, g in bsel.items() if n[:, g].sum() > 0}
    return res


def causal_and_leads(results, U, sc, arm):
    """Causal-variant rows of the non-null genes (a joint arm's missing row left NaN) and every gene's lead, per dataset;
    a gene that a native arm did not test (05b: constant native total) has no lead and ranks as no finite p."""
    parts, leads = [], []
    cols = (JOINT_COLS if arm in ONE_TEST else C.COLS) + (
        ['method'] if arm in C.MIXQTL_ARMS else [] if arm in ONE_TEST else ['allelic_admitted'])
    for r, u in U[U.scenario == sc].groupby('rep'):
        d = C.read_results(arm_dir(results, sc, arm) / f'nominal_rep{r:03d}.parquet', cols)
        parts.append(u[~u.is_null].merge(d, left_on=['gene', 'causal_variant'], right_on=['phenotype_id', 'variant_id'],
                                         how='left'))
        d['p'] = d.pval_nominal.where(np.isfinite(d.pval_nominal), np.inf)
        d['absstat'] = np.abs(d.slope / d.slope_se)
        top = (d.sort_values(['phenotype_id', 'p', 'absstat'], ascending=[True, True, False], kind='stable')
               .groupby('phenotype_id', sort=False).head(1).set_index('phenotype_id'))
        L = u.set_index('gene')[['scenario', 'rep', 'is_null', 'band', 'causal_variant']].join(
            top[['variant_id', 'p', 'absstat']], how='left')
        if arm in NATIVE_ARMS:
            L['p'] = L.p.fillna(np.inf)
        elif L.variant_id.isna().any():
            raise SystemExit(f'{sc} {arm} rep {r}: no result rows for {int(L.variant_id.isna().sum())} genes')
        leads.append(L.rename(columns={'variant_id': 'lead_variant', 'p': 'lead_p', 'absstat': 'lead_absstat'}).reset_index())
    Cz = pd.concat(parts, ignore_index=True)
    if Cz.variant_id.isna().any() and arm not in MISSING_OK:
        raise SystemExit(f'{sc} {arm}: {int(Cz.variant_id.isna().sum())} causal variants have no result row')
    return Cz, pd.concat(leads, ignore_index=True)


def ratio_block(ratio, Cz, genes, bsel, bidx):
    """Mean of a per-unit ratio over its finite units, per band, with the gene-clustered interval."""
    gi = pd.Index(genes)
    ok = np.isfinite(ratio).values
    idx = gi.get_indexer(Cz.gene[ok])
    s = np.bincount(idx, weights=ratio[ok], minlength=len(genes))
    k = np.bincount(idx, minlength=len(genes)).astype(float)
    out = {}
    for bn, gs in bsel.items():
        if k[gs].sum() == 0:
            continue
        bb = s[gs][bidx[bn]].sum(1) / k[gs][bidx[bn]].sum(1)
        out[bn] = dict(mean=float(s[gs].sum() / k[gs].sum()), lo=float(np.nanquantile(bb, .025)),
                       hi=float(np.nanquantile(bb, .975)), units=int(k[gs].sum()),
                       excluded_nonfinite=int((~ok & Cz.gene.isin(set(gi[gs])).values).sum()))
    return out


def recovery(Cz, arm, genes, bsel, bidx):
    res = {}
    for ch in channels(arm, SLOPE):
        res[ch] = dict(bias_count=ratio_block(Cz[SLOPE[ch][0]].astype(float) / Cz[truth_col(arm, ch, 'count')], Cz, genes, bsel, bidx))
        if (arm in C.HAPMIX_ARMS and ch in TRUTH['pipeline']) or arm == C.TENSORQTL:
            res[ch]['bias_pipeline'] = ratio_block(Cz[SLOPE[ch][0]].astype(float) / Cz[truth_col(arm, ch, 'pipeline')], Cz, genes,
                                                   bsel, bidx)
    return res


def channel_truths(Cz, arm, scale=None):
    """Per causal unit, the estimand of each channel's slope (scale None: the arm's own); combined = the
    inverse-variance combination of the channel truths at the unit's own se (01_check_inputs.py pins that the
    combined slope is that combination of the channel slopes). A joint arm's one slope has estimand beta; the tensorqtl
    arm's is the total channel's."""
    if arm in JOINT_LIKE:
        return {'combined': Cz[TRUTH['count']['combined']].values}
    scale = scale or ('pipeline' if arm in C.HAPMIX_ARMS + (C.TENSORQTL,) else 'count')
    if arm == C.TENSORQTL:
        return {'combined': Cz[truth_col(arm, 'combined', scale)].values}
    ta, tt = Cz[TRUTH[scale]['allelic']].values, Cz[TRUTH[scale]['total']].values
    sa, sea = Cz.slope_a.astype(float).values, Cz.slope_a_se.astype(float).values
    st, set_ = Cz.slope_t.astype(float).values, Cz.slope_t_se.astype(float).values
    with np.errstate(divide='ignore', invalid='ignore'):
        wa = np.where(np.isfinite(sa) & np.isfinite(sea), 1.0 / sea ** 2, 0.0)
        wt = np.where(np.isfinite(st) & np.isfinite(set_), 1.0 / set_ ** 2, 0.0)
        if arm in C.MIXQTL_ARMS:   # mixQTL's meta estimate is one channel alone unless `method` is 'meta'
            wa, wt = np.where(Cz.method == 'trc', 0.0, wa), np.where(Cz.method == 'asc', 0.0, wt)
        else:                      # hapmixQTL: the total channel alone below the allelic admission floor
            wa = np.where(Cz.allelic_admitted.astype(bool).values, wa, 0.0)
        tc = (np.where(wa > 0, wa * ta, 0.0) + np.where(wt > 0, wt * tt, 0.0)) / (wa + wt)
    return {'combined': tc, 'allelic': ta, 'total': tt}


def gene_sums(g, n_genes, ok, *values):
    """Per-gene counts and sums of each value over the rows where ok."""
    return [np.bincount(g[ok], minlength=n_genes).astype(float)] + [
        np.bincount(g[ok], weights=v[ok], minlength=n_genes) for v in values]


def nonnull_precision(Cz, Cu, arm, genes):
    """Per channel, per-gene [n, sum z, sum z^2, pairs, sum err^2 arm, sum err^2 unit] at the causal variants, each
    arm against its own truth, then [pairs, sum err^2 arm, sum err^2 unit] with the count-scale truth for both."""
    if not (np.array_equal(Cz.rep.values, Cu.rep.values) and np.array_equal(Cz.gene.values, Cu.gene.values)):
        raise SystemExit(f'{arm}: causal units are not in the order of the unit arm\'s')
    g = pd.Index(genes).get_indexer(Cz.gene)
    T, Tu = channel_truths(Cz, arm), channel_truths(Cu, 'unit')
    Tc, Tuc = channel_truths(Cz, arm, 'count'), channel_truths(Cu, 'unit', 'count')
    acc = {}
    for ch, (b, s) in channels(arm, SLOPE).items():
        x, xu = Cz[b].astype(float).values, Cu[b].astype(float).values
        e, eu, ec, euc = x - T[ch], xu - Tu[ch], x - Tc[ch], xu - Tuc[ch]
        with np.errstate(divide='ignore', invalid='ignore'):
            z = e / Cz[s].astype(float).values
        acc[ch] = np.vstack(gene_sums(g, len(genes), np.isfinite(z), z, z ** 2)
                            + gene_sums(g, len(genes), np.isfinite(e) & np.isfinite(eu), e ** 2, eu ** 2)
                            + gene_sums(g, len(genes), np.isfinite(ec) & np.isfinite(euc), ec ** 2, euc ** 2))
    return acc


def null_rows(results, sc, arm, r, nulls):
    d = C.read_results(arm_dir(results, sc, arm) / f'nominal_rep{r:03d}.parquet', NULL_COLS[:4] if arm in ONE_TEST else NULL_COLS)
    return d[d.phenotype_id.isin(nulls)].sort_values(['phenotype_id', 'variant_id'], kind='stable').reset_index(drop=True)


def null_precision(results, U, sc, arm, genes):
    """Per channel, per-gene [n, sum z, sum z^2, pairs, sum slope^2 arm, sum slope^2 unit] over null genes' variants,
    and the counts of tests left out."""
    acc = {ch: np.zeros((6, len(genes))) for ch in channels(arm, SLOPE)}
    excluded = {ch: dict(nonfinite_z=0, unpaired=0, tests=0, no_row=0) for ch in channels(arm, SLOPE)}
    for r in sorted(U[U.scenario == sc].rep.unique()):
        nulls = set(U[(U.scenario == sc) & (U.rep == r) & U.is_null].gene)
        d, du = null_rows(results, sc, arm, r, nulls), null_rows(results, sc, 'unit', r, nulls)
        no_row = len(du) - len(d)
        if arm in MISSING_OK:   # tests without a row are left out, counted as no_row
            du = d[['phenotype_id', 'variant_id']].merge(du, how='left')
        if not (d.phenotype_id.equals(du.phenotype_id) and d.variant_id.equals(du.variant_id)):
            raise SystemExit(f'{sc} rep {r}: {arm} and unit differ in their null-gene tested variants')
        g = pd.Index(genes).get_indexer(d.phenotype_id)
        for ch, (b, s) in channels(arm, SLOPE).items():
            x, xu = d[b].astype(float).values, du[b].astype(float).values
            with np.errstate(divide='ignore', invalid='ignore'):
                z = x / d[s].astype(float).values
            okz, pair = np.isfinite(z), np.isfinite(x) & np.isfinite(xu)
            acc[ch] += np.vstack(gene_sums(g, len(genes), okz, z, z ** 2) + gene_sums(g, len(genes), pair, x ** 2, xu ** 2))
            for k, v in (('nonfinite_z', int((~okz).sum())), ('unpaired', int((~pair).sum())), ('tests', len(z)), ('no_row', no_row)):
                excluded[ch][k] += v
    return acc, excluded


def sd_pooled(n, s1, s2):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.sqrt((s2 - s1 ** 2 / n) / (n - 1))


def sum_ratio(a, b):
    with np.errstate(divide='ignore', invalid='ignore'):
        return a / b


def boot_stat(parts, gs, bi, fn):
    """fn of the per-gene sums over genes gs, and its gene-clustered 95% interval."""
    b = fn(*[p[gs][bi].sum(1) for p in parts])
    return dict(value=float(fn(*[p[gs].sum() for p in parts])), lo=float(np.nanquantile(b, .025)), hi=float(np.nanquantile(b, .975)))


def summarize(acc, bsel, bidx):
    """sd(z) and the squared-error ratios against unit from per-gene sums (nonnull_precision / null_precision)."""
    n, s1, s2 = acc[:3]
    ratios = {'ratio_vs_unit': acc[3:6]}
    if len(acc) > 6:
        ratios['ratio_vs_unit_count'] = acc[6:9]
    out = {'sd_z': {}, **{k: {} for k in ratios}}
    for bn, gs in bsel.items():
        if n[gs].sum() > 1:
            out['sd_z'][bn] = dict(**boot_stat((n, s1, s2), gs, bidx[bn], sd_pooled), units=int(n[gs].sum()), genes=int((n[gs] > 0).sum()))
        for k, (m, a, u) in ratios.items():
            if m[gs].sum() > 0:
                out[k][bn] = dict(**boot_stat((a, u), gs, bidx[bn], sum_ratio), units=int(m[gs].sum()))
    return out


def precision(results, U, sc, arm, genes, bsel, bidx, CL):
    nacc, nexc = null_precision(results, U, sc, arm, genes)
    res = {ch: dict(null=summarize(nacc[ch], bsel, bidx), null_excluded=nexc[ch]) for ch in channels(arm, SLOPE)}
    if CL is not None:
        acc = nonnull_precision(CL[arm][0], CL['unit'][0], arm, genes)
        for ch in channels(arm, SLOPE):
            res[ch]['nonnull'] = summarize(acc[ch], bsel, bidx)
    return res


def detection(Cz, arm):
    Cz = Cz[Cz.variant_id.notna()]   # a joint arm's causal unit without a row is excluded (missing_causal)
    res = {}
    for ch, col in channels(arm, PVAL).items():
        p = Cz[col].astype(float)
        res[ch] = {}
        for bn, lo, hi in BANDS:
            m = ((Cz.reads >= lo) & (Cz.reads < hi)).values
            if m.any():
                res[ch][bn] = dict(units=int(m.sum()), **{str(al): float((p[m] < al).mean()) for al in DETECT_ALPHAS})
    return res


def ld_r2(dos, rows, a, b):
    """Squared dosage correlation between variant ids a and b, per unit; NaN where either is constant."""
    x, y = dos[[rows[v] for v in a]].astype(float), dos[[rows[v] for v in b]].astype(float)
    xc, yc = x - x.mean(1, keepdims=True), y - y.mean(1, keepdims=True)
    den = np.sqrt((xc ** 2).sum(1) * (yc ** 2).sum(1))
    r2 = np.full(len(a), np.nan)
    r2[den > 0] = ((xc[den > 0] * yc[den > 0]).sum(1) / den[den > 0]) ** 2
    return r2


def lead_recovery(L, dos, rows):
    N = L[~L.is_null].copy()
    has = np.isfinite(N.lead_p.values)
    same = (N.lead_variant == N.causal_variant).values & has
    r2 = np.full(len(N), np.nan)
    r2[has] = ld_r2(dos, rows, N.lead_variant.values[has], N.causal_variant.values[has])
    r2[same] = 1.0
    N['r2'], N['same'] = r2, same
    res = {'no_finite_p': int((~has).sum()), 'r2_undefined': int((has & ~np.isfinite(r2)).sum())}
    for bn, _, _ in BANDS:
        B = N if bn == 'all' else N[N.band == bn]
        if len(B):
            f = np.isfinite(B.r2.values)
            res[bn] = dict(units=len(B), lead_is_causal=float(B.same.mean()),
                           r2_high=float((np.where(f, B.r2.values, 0.0) >= R2_HIGH).mean()),
                           median_r2=float(np.median(B.r2.values[f])) if f.any() else float('nan'), r2_defined=int(f.sum()))
    return res


def evidence_rank(p, s):
    """Average ranks, higher = stronger evidence: smaller p first, then larger |slope / se|."""
    k1, k2 = -np.asarray(p, float), np.where(np.isfinite(s), s, -1.0)
    o = np.lexsort((k2, k1))
    r = np.empty(len(o))
    i = 0
    while i < len(o):
        j = i
        while j + 1 < len(o) and k1[o[j + 1]] == k1[o[i]] and k2[o[j + 1]] == k2[o[i]]:
            j += 1
        r[o[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return r


def auc(rank, nonnull):
    n1, n0 = int(nonnull.sum()), int((~nonnull).sum())
    return float((rank[nonnull].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float('nan')


def ranking(L, key):
    res = {}
    reps = sorted(L.rep.unique())
    ridx = boot(key, len(reps))
    for bn, _, _ in BANDS:
        B = L if bn == 'all' else L[L.band == bn]
        a = np.array([auc(evidence_rank(x.lead_p.values, x.lead_absstat.values), ~x.is_null.values) for _, x in B.groupby('rep')])
        if np.isfinite(a).any():
            m = np.nanmean(a[ridx], axis=1)
            res.setdefault('auc', {})[bn] = dict(mean=float(np.nanmean(a)),
                                                 lo=float(np.nanquantile(m, .025)) if len(reps) > 1 else float('nan'),
                                                 hi=float(np.nanquantile(m, .975)) if len(reps) > 1 else float('nan'))
    rank = evidence_rank(L.lead_p.values, L.lead_absstat.values)
    o = np.argsort(-rank, kind='stable')
    fdp = np.cumsum(L.is_null.values[o]) / np.arange(1, len(o) + 1)
    cut = np.r_[rank[o][1:] != rank[o][:-1], True] & (fdp <= FDR)
    k = int(np.where(cut)[0].max()) + 1 if cut.any() else 0
    top = np.zeros(len(L), bool)
    top[o[:k]] = True
    res['fdp_matched'] = dict(fdr=FDR, discoveries=k, false=int((top & L.is_null.values).sum()),
                              p_threshold=float(L.lead_p.values[o[k - 1]]) if k else float('nan'))
    for bn, _, _ in BANDS:
        m = (~L.is_null).values & ((L.band == bn).values | (bn == 'all'))
        if m.any():
            res['fdp_matched'][bn] = dict(non_null=int(m.sum()), power=float(top[m].mean()))
    return res


def trecase_parts(results, U, sc, arm, L, key):
    """A TReCASE arm's joint-fit failure at the reported lead: gene units with a finite lead p (L, causal_and_leads), null and
    non-null, whose lead row has no joint fit (asSeq's final p is then its total-count test); and power at FDR realized
    false-discovery proportion (ranking's fdp_matched) with genes ranked by the smallest p of each component (TRECASE_PARTS)
    alone, a gene without a finite one ranked last."""
    leads, comp = [], {k: [] for k in TRECASE_PARTS}
    for r, u in U[U.scenario == sc].groupby('rep'):
        d = pd.read_parquet(arm_dir(results, sc, arm) / f'nominal_rep{r:03d}.parquet',
                            columns=['phenotype_id', 'variant_id', 'joint_ok', *TRECASE_PARTS.values()])
        d['variant_id'] = d.variant_id.astype(str)
        Lr = L[(L.rep == r) & np.isfinite(L.lead_p)].merge(d[['phenotype_id', 'variant_id', 'joint_ok']], how='left',
                                                              left_on=['gene', 'lead_variant'], right_on=['phenotype_id', 'variant_id'])
        leads.append(Lr)
        for k, col in TRECASE_PARTS.items():
            comp[k].append(u.assign(lead_p=d.groupby('phenotype_id')[col].min().reindex(u.gene).fillna(np.inf).values,
                                    lead_absstat=np.nan))
    F = pd.concat(leads, ignore_index=True)
    fail = {w: dict(units=int(m.sum()), failed=int((~F.joint_ok[m].astype(bool)).sum()))
            for w, m in (('null', F.is_null.values), ('nonnull', ~F.is_null.values))}
    return dict(lead_joint_failed=fail, fdp_power={k: ranking(pd.concat(v, ignore_index=True), key)['fdp_matched']['all']['power']
                                                   for k, v in comp.items()})


def gene_level(U, sc, genes, bsel, bidx, pvals, name):
    """Gene-level null rate of p < GENE_LEVEL_ALPHA, Benjamini-Hochberg power at FDR within dataset (section 7) and power at
    FDR realized false-discovery proportion over the pooled datasets (fdp_matched, no interval), for the
    gene-level p `name` whose values for dataset r, in `genes` order, are pvals(r)."""
    reps = sorted(U[U.scenario == sc].rep.unique())
    K = {c: np.zeros((len(reps), len(genes))) for c in ('null', 'disc', 'false')}
    n0, n1 = np.zeros((len(reps), len(genes))), np.zeros((len(reps), len(genes)))
    P = np.ones((len(reps), len(genes)))   # no finite p: never called
    for i, r in enumerate(reps):
        p = pvals(r)
        P[i] = np.where(np.isfinite(p), p, 1.0)
        null = U[(U.scenario == sc) & (U.rep == r)].set_index('gene').is_null.loc[genes].values
        fin = np.isfinite(p)
        if not ((p[fin] >= 0) & (p[fin] <= 1)).all():
            raise SystemExit(f'{sc} rep {r} {name}: gene-level p outside [0, 1]')
        disc = np.zeros(len(genes), bool)
        disc[fin] = false_discovery_control(p[fin], method='bh') <= FDR
        n0[i], n1[i] = null, ~null
        K['null'][i], K['disc'][i], K['false'][i] = null & (p < GENE_LEVEL_ALPHA), ~null & disc, null & disc
    # power at FDR realized false-discovery proportion, as ranking's fdp_matched: the scenario's gene units pooled over
    # datasets, ranked by this p, cut at the deepest tie boundary where at most FDR of the units called are null
    p, isnull = P.ravel(), n0.ravel().astype(bool)
    o = np.argsort(p, kind='stable')
    cut = np.r_[p[o][1:] != p[o][:-1], True] & (np.cumsum(isnull[o]) / np.arange(1, len(o) + 1) <= FDR)
    k = int(np.where(cut)[0].max()) + 1 if cut.any() else 0
    top = np.zeros(len(p), bool)
    top[o[:k]] = True
    top = top.reshape(P.shape)
    fm = dict(fdr=FDR, discoveries=k, false=int((top & (n0 > 0)).sum()))
    for bn, g in bsel.items():
        m = n1[:, g] > 0
        if m.any():
            fm[bn] = dict(non_null=int(m.sum()), power=float(top[:, g][m].mean()))
    return dict(p=name, discoveries=int(K['disc'].sum() + K['false'].sum()), false_discoveries=int(K['false'].sum()),
                fdp_matched=fm,
                null_rate={bn: pooled(K['null'], n0, g, bidx[bn]) for bn, g in bsel.items() if n0[:, g].sum() > 0},
                power_bh={bn: pooled(K['disc'], n1, g, bidx[bn]) for bn, g in bsel.items() if n1[:, g].sum() > 0})


def cis_p(results, sc, arm, genes):
    """pvals for gene_level: dataset r's permutation p (CIS_P) from the arm's cis file."""
    def get(r):
        d = pd.read_parquet(arm_dir(results, sc, arm) / f'cis_rep{r:03d}.parquet', columns=['phenotype_id', CIS_P[arm]])
        if not (d.phenotype_id.is_unique and set(d.phenotype_id) <= set(genes)
                and (arm in NATIVE_ARMS or set(d.phenotype_id) == set(genes))):
            raise SystemExit(f'{sc} {arm} rep {r}: gene-level results do not cover the {len(genes)} genes once each')
        return d.set_index('phenotype_id')[CIS_P[arm]].reindex(genes).values.astype(float)   # a native arm's untested gene: NaN
    return get


def eigenmt_p(results, sc, arm, genes, m_eff):
    """pvals for gene_level: dataset r's eigenMT p, min(1, the gene's smallest nominal p x M_eff); NaN without a finite p."""
    def get(r):
        d = pd.read_parquet(arm_dir(results, sc, arm) / f'nominal_rep{r:03d}.parquet', columns=['phenotype_id', 'pval_nominal'])
        return np.minimum(1.0, d.groupby('phenotype_id').pval_nominal.min().reindex(genes).values.astype(float) * m_eff)
    return get


def eigenmt_structure(I, genes):
    """Per gene over 03's windows of its tested variants (position order, checked in 03): the median Ledoit-Wolf weight
    and the unshrunk EIGENMT_VAR count summed over windows (module docstring), float64 on the CPU."""
    tested, rows = C.setup(I)['tested_rows'], []
    for g in genes:
        dos, weights, count = I['dos'][tested[g]].astype(np.float64), [], 0
        for s in range(0, len(dos), EIGENMT_WINDOW):
            x = dos[s:s + EIGENMT_WINDOW].T                    # donors x variants
            weights.append(float(eigenmt.lw_shrink(torch.from_numpy(x))[1]))
            v = x[:, x.std(0) > 0]
            if v.shape[1]:
                ev = np.clip(np.linalg.eigvalsh(np.atleast_2d(np.corrcoef(v, rowvar=False))), 0, None)
                count += eigenmt.find_num_eigs(ev, v.shape[1], EIGENMT_VAR)
        rows.append(dict(gene=g, n_tested=len(dos), unshrunk=count, lw_weight=float(np.median(weights))))
    return pd.DataFrame(rows).set_index('gene')


def eigenmt_vs_permutation(runs, genes, m_eff):
    """Per arm with pval_beta: median M_eff / beta_shape2 over gene units (where the cis file has the shape) and median
    eigenMT p / pval_beta over gene units with pval_beta in [FDR / genes, FDR) (module docstring)."""
    lo, res = FDR / len(genes), {}
    for arm in (a for a in C.ARMS if CIS_P[a] == 'pval_beta'):
        shape, ratio = [], []
        for sc, r in runs:
            c = pd.read_parquet(C.RESULTS / sc / arm / f'cis_rep{r:03d}.parquet').set_index('phenotype_id').loc[genes]
            pb, pe = c.pval_beta.to_numpy(float), eigenmt_p(C.RESULTS, sc, arm, genes, m_eff)(r)
            if 'beta_shape2' in c:
                shape.append(m_eff / c.beta_shape2.to_numpy(float))
            band = (pb >= lo) & (pb < FDR) & np.isfinite(pe)
            ratio.append(pe[band] / pb[band])
        shape, ratio = (np.concatenate(x) if x else np.array([]) for x in (shape, ratio))
        res[arm] = dict(m_eff_over_shape2=float(np.nanmedian(shape)) if len(shape) else None, units=int(np.isfinite(shape).sum()),
                        eigenmt_over_pval_beta=float(np.median(ratio)), band=[lo, FDR], band_units=len(ratio))
    return res


def stored_rates(path, prefix):
    """Per-permutation rate at ANCHOR_ALPHA of each channel over a stored null run's draws (path.parent/draws)."""
    files = sorted((path.parent / 'draws').glob(f'{prefix}_*.parquet'))
    if len(files) != ANCHOR_N_PERM:
        raise SystemExit(f'{path.parent}/draws: {len(files)} {prefix} draws, expected {ANCHOR_N_PERM}')
    out = {ch: [] for ch in PVAL}
    for f in files:
        x = pd.read_parquet(f, columns=list(PVAL.values()))
        for ch, col in PVAL.items():
            v = x[col].values
            out[ch].append(float(np.mean(v[np.isfinite(v)] < ANCHOR_ALPHA)))
    return {ch: np.array(v) for ch, v in out.items()}


def anchor(null0):
    """Each hapmixQTL arm's anchor rates against its stored null run: the stored mean, and at ANCHOR_ALPHA the central
    ANCHOR_CENTRAL of the stored per-permutation rates and this dataset's percentile among them."""
    res = {}
    q_lo, q_hi = (1 - ANCHOR_CENTRAL) / 2, (1 + ANCHOR_CENTRAL) / 2
    for arm, (path, prefix) in ANCHOR.items():
        stored, per_perm = json.loads(path.read_text()), stored_rates(path, prefix)
        res[arm] = {}
        for ch in PVAL:
            res[arm][ch] = {}
            for al in map(str, C.ALPHAS):
                m = null0[arm][ch]['all'][al]
                row = dict(stored=stored['rates'][prefix][ch]['all']['after'][al]['rate'], rate=m['rate'], lo=m['lo'], hi=m['hi'])
                if float(al) == ANCHOR_ALPHA:
                    r = per_perm[ch]
                    row.update(perm_lo=float(np.quantile(r, q_lo)), perm_hi=float(np.quantile(r, q_hi)),
                               percentile=float(100 * np.mean(r <= m['rate'])))
                    row['passed'] = bool(row['perm_lo'] <= m['rate'] <= row['perm_hi'])
                res[arm][ch][al] = row
    return res


def main():
    if not NATIVE_ARMS:
        print(f'native-input arms {list(C.NATIVE_ARMS)} not scored: {C.NATIVE} does not exist (05b_native_arms.py has not run '
              f'into this root)', flush=True)
    meta, genes, U, keep_a = load_units(C.DATASETS, C.RESULTS)
    I = C.load()[0]
    if list(I['genes']) != genes:
        raise SystemExit('loader genes differ from meta.json')
    rows = {str(v): i for i, v in enumerate(I['vdf'].index)}
    bsel, bidx = band_selections(genes, U, keep_a)
    scen = [f'beta{b}' for b in meta['betas']]
    S = dict(datasets=str(C.DATASETS), results=str(C.RESULTS), n_datasets=meta['n_datasets'], arms=list(C.ARMS),
             joint_arms=list(C.JOINT), joint_results={a: str(p) for a, p in C.JOINT.items()}, native_arms=list(NATIVE_ARMS),
             native_results={a: str(C.NATIVE_RESULTS[a]) for a in NATIVE_ARMS}, bands=[b[0] for b in BANDS],
             n_boot=N_BOOT, seed=C.SEED, fdr=FDR,
             mixqtl_permutation=json.loads((C.RESULTS / 'mixqtl_permutation.json').read_text()), missing_causal={},
             one_df_genes=[g for g, k in zip(genes, keep_a) if k == ONE_DF],
             null={}, precision={}, recovery={}, lead={}, detection={}, ranking={}, gene_level={}, gene_level_eigenmt={},
             trecase_parts={})
    if NATIVE_ARMS:
        S['native_datasets'] = str(C.NATIVE_DATASETS)
    em = pd.read_csv(C.EIGENMT, sep='\t').set_index('gene')
    m_eff = em.m_eff.loc[genes].values.astype(float)
    share = em.m_eff / em.n_tested
    st = eigenmt_structure(I, genes)
    un = st.unshrunk / st.n_tested
    S['eigenmt'] = dict(source=str(C.EIGENMT), m_eff_min=int(em.m_eff.min()), m_eff_median=float(em.m_eff.median()),
                        m_eff_max=int(em.m_eff.max()), share_min=float(share.min()), share_median=float(share.median()),
                        share_max=float(share.max()), window=EIGENMT_WINDOW, donors=int(I['dos'].shape[1]),
                        unshrunk_share_min=float(un.min()), unshrunk_share_median=float(un.median()), unshrunk_share_max=float(un.max()),
                        lw_weight_min=float(st.lw_weight.min()), lw_weight_median=float(st.lw_weight.median()),
                        lw_weight_max=float(st.lw_weight.max()),
                        vs_permutation=eigenmt_vs_permutation(C.runs(meta), genes, m_eff))
    vp = S['eigenmt']['vs_permutation']
    print(f'eigenMT: M_eff / tested {share.min():.2f}-{share.max():.2f}, unshrunk {un.min():.2f}-{un.max():.2f} (median Ledoit-Wolf '
          f'weight per gene {st.lw_weight.min():.2f}-{st.lw_weight.max():.2f}); M_eff / beta_shape2 '
          + ', '.join(f'{a} {v["m_eff_over_shape2"]:.2f}' for a, v in vp.items() if v['m_eff_over_shape2'])
          + f'; eigenMT p / pval_beta in [{FDR / len(genes):g}, {FDR}) ' + ', '.join(f'{a} {v["eigenmt_over_pval_beta"]:.2f} ({v["band_units"]})'
                                                                             for a, v in vp.items()), flush=True)
    for i, sc in enumerate(scen):
        S['null'][sc] = {arm: null_calibration(C.RESULTS, U, sc, arm, genes, bsel, bidx) for arm in ARMS}
        CL = None
        if (~U[U.scenario == sc].is_null).any():
            CL = {arm: causal_and_leads(C.RESULTS, U, sc, arm) for arm in ARMS}
            S['missing_causal'][sc] = {arm: int(CL[arm][0].variant_id.isna().sum()) for arm in ARMS if arm in MISSING_OK}
            S['recovery'][sc] = {arm: recovery(CL[arm][0], arm, genes, bsel, bidx) for arm in ARMS}
            S['lead'][sc] = {arm: lead_recovery(CL[arm][1], I['dos'], rows) for arm in ARMS}
            S['detection'][sc] = {arm: detection(CL[arm][0], arm) for arm in ARMS}
            S['ranking'][sc] = {arm: ranking(CL[arm][1], (AUC_BOOT_KEY, i)) for arm in ARMS}
            S['trecase_parts'][sc] = {arm: trecase_parts(C.RESULTS, U, sc, arm, CL[arm][1], (AUC_BOOT_KEY, i)) for arm in TRECASE_ARMS}
        S['precision'][sc] = {arm: precision(C.RESULTS, U, sc, arm, genes, bsel, bidx, CL) for arm in ARMS}
        S['gene_level'][sc] = {arm: gene_level(U, sc, genes, bsel, bidx, cis_p(C.RESULTS, sc, arm, genes), CIS_P[arm])
                               for arm in CIS_ARMS}
        S['gene_level_eigenmt'][sc] = {arm: gene_level(U, sc, genes, bsel, bidx, eigenmt_p(C.RESULTS, sc, arm, genes, m_eff), 'eigenmt')
                                       for arm in ARMS}
        print(f'scored {sc}', flush=True)
    S['anchor'] = anchor(S['null']['beta0.0']) if ANCHOR else None
    S['trecase_components'] = null_calibration(C.RESULTS, U, 'beta0.0', 'trecase', genes, bsel, bidx, TRECASE_PARTS)
    if NATIVE_ARMS:
        S['trecase_native_components'] = null_calibration(C.RESULTS, U, 'beta0.0', 'trecase_native', genes, bsel, bidx, TRECASE_PARTS)
    C.write_json(C.SUMMARY, S)
    a = S['anchor']
    if a is None:
        print(f'anchor comparison skipped: gene set {C.GENE_SET} has no stored null run (common.STORED_NULL is None)')
    else:
        print('anchor at 0.05, this dataset vs stored mean (percentile among stored permutations): '
              + '; '.join(f'{arm} {ch} {a[arm][ch]["0.05"]["rate"]:.4f} vs {a[arm][ch]["0.05"]["stored"]:.4f} '
                          f'({a[arm][ch]["0.05"]["percentile"]:.0f}%)' for arm in a for ch in a[arm]))
    print(f'wrote {C.SUMMARY}')


if __name__ == '__main__':
    main()

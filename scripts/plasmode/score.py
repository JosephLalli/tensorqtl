"""Score the plasmode benchmark: how well the six arms of run_arms.py recover
the injected effects (four hapmixQTL weightings and mixQTL mode at two cutoff
settings), the precision and stated standard error of their slopes, the
nominal-p rate on the null genes, and the gene-level permutation results.

JOINT ARMS: RASQUAL (run_rasqual.py) and TReCASE (asSeq, run_trecase_asseq.py),
nominal only, read from their own results directories (JOINT). Each gives one
test per variant, scored as the combined channel only. Their slope is the log2
aFC ALT over REF as the runners store it: RASQUAL log2(pi / (1 - pi)), pi the
ALT allele's expected share (nbem.c:1058: expression 2(1 - pi), 1, 2 pi at ALT
dosage 0, 1, 2); TReCASE b / ln 2 of the statistic final_Pvalue used (joint,
or TReC when the cis/trans test rejects or the joint fit failed). Both models
put the total mean at 1, (1 + kappa)/2, kappa (asSeq glmNBlog's offsets,
glm.c:1577 and trecase.c:1049; RASQUAL's K0), which is the generator's
expected total fold averaged over donors, so the estimand of every joint-arm
slope, TReC fallback included, is log2 kappa = beta: its truth is beta. Exception not identifiable per row: where glmNBlog
fails asSeq refits TReC with linear dosage (run_trecase_asseq.py, counted
there as trec_linear_dosage). Their slope_se is DERIVED, |slope| / sqrt(chisq),
so for them z = slope / se is +-sqrt(chisq) under truth 0 and sd(z) measures the
likelihood-ratio test's calibration, not a reported se. A test with no row
(RASQUAL non-converged; TReCASE constant ALT dosage, including TPPP's causal
variant in rep 002) is counted (missing_causal; null_excluded no_row) and left
out: a causal unit without a row is excluded from detection, and is non-finite
in bias and precision.

DATASETS (user decision 2026-09-26): one beta = 0 anchor dataset (every gene
null, no thinning) and 3 datasets at each of |beta| = 0.2 / 0.4 / 0.8, half
the genes null. At these counts a gene is non-null in about 1.5 datasets per
scenario, so every statistic below is POOLED over gene-dataset units, with a
gene-clustered interval, rather than computed per gene.

UNITS. A causal unit is one (dataset, non-null gene) at its causal variant; a
gene unit is one (dataset, gene). Every statistic is reported overall and by
the gene's REAL median haplotype-informative reads over donors (median of
pL + pR, corrected_null_store_20260925/gene_design.tsv, checked equal to
truth.tsv) in BANDS, and over every gene but the ONE_DF genes (key NO_ONE_DF;
gene list in the summary as one_df_genes): those with ONE_DF admitted allelic
donors (gene_design.tsv n_allelic_keep), whose through-origin allelic fit has
one residual degree of freedom but whose p is referred to t with
N - 2 - max(n_cov, n_cov_a) = 73 df (hapmixqtl.py:1873). The anchor's stored
rates at TAIL_ALPHA are also given without them (stored_without_one_df).
TReCASE's component tests (TRECASE_PARTS) are scored on the anchor's null
genes like a channel (trecase_components), each over the tests where its p is
finite. Channels are combined / allelic / total; for the mixQTL
arms these are meta / asc / trc. mixQTL's slopes and se are stored in natural
log (its response is natural log by design) and are divided by ln 2 on
reading, so every slope and se below is in log2 units.

GENE-CLUSTERED INTERVAL (every interval below unless stated): genes are
resampled with replacement N_BOOT times, each gene carrying all its units
from the scenario's datasets, the statistic is recomputed from the resampled
per-gene sums, and the 2.5% and 97.5% quantiles are reported.

(1) BIAS at the causal variant. Bias ratio = mean over causal units of
    slope / truth. Units whose ratio is not finite are excluded and counted
    (a causal variant at which every donor is heterozygous has no
    total-channel estimand). Two truths (make_datasets.py, section 5):
      count scale: allelic slope / beta; total slope / total truth; combined
        slope / beta (the combined slope mixes the channels' estimands, so it
        is reported without a rule).
      pipeline scale, hapmixQTL arms only: allelic slope / allelic pipeline
        truth, total slope / total pipeline truth. mixQTL's response carries no
        pseudocount and no +1, so its count-scale truth is already its own
        scale. The combined slope has no single pipeline-scale truth.
(2) PRECISION. TRUTH at the causal variant, per channel: hapmixQTL arms, the
    pipeline scale (allelic and total pipeline truths); mixQTL arms, the
    count scale (beta for asc, the total truth for trc). The combined slope
    is exactly the inverse-variance combination of the two channel slopes at
    the unit's own stated se (w = 1/se^2; for mixQTL only the channel or
    channels its `method` names), which is checked per unit to IVW_TOL of the
    se, so its truth is the same combination of the two channel truths.
    (2a) SE CALIBRATION: sd(z), ddof 1 about its mean, with
         z = (slope - truth) / se. It is 1 when the stated se equals the
         realized sd of the slope, above 1 when the stated se is too small.
         Non-null: over the non-null gene-dataset units at the causal
         variant. Null: z = slope / se (truth 0) over every tested variant of
         the null genes; a gene's variants are correlated through LD, which
         the gene-clustered interval carries.
    (2b) EFFICIENCY RELATIVE TO 'unit', paired: non-null, sum (slope - truth)^2
         at the causal variant under the arm over the same sum under 'unit',
         over the units finite under both, each arm against its own truth;
         null, sum slope^2 (the squared error against truth 0) over the null
         genes' tested variants finite under both. Below 1 means more precise
         than unit weights. 'split' and 'unit' share the total channel's
         weights (all 1), so their total-channel ratio is 1 by construction.
         ratio_vs_unit_count (non-null): the same with the count-scale truth
         for the arm and unit alike (combined: its inverse-variance
         combination, beta for the joint arms), the cross-method comparison.
    Both are computed for the beta = 0 anchor too (null form only): the one
    place with no thinning, on the real records as they are.
(3) LEAD-VARIANT RECOVERY, per non-null gene unit. Lead = the tested variant
    with the smallest nominal p (combined for hapmixQTL, meta for mixQTL);
    ties, including p that underflow to 0, are broken by the larger
    |slope / se|. LD r^2 = squared Pearson correlation of ALT dosages over the
    92 donors between lead and causal variant (1 when they are the same
    variant; undefined when either has one dosage in every donor, counted).
    Reported: share with lead = causal, share with r^2 >= R2_HIGH, both over
    all non-null units (a unit without a finite p counts as not recovered),
    and the median r^2 over units where it is defined.
(4) CAUSAL-VARIANT DETECTION: the share of non-null gene units whose nominal
    p at the causal variant is below each of DETECT_ALPHAS, per channel; a
    non-finite p counts as not detected and is counted.
(5) GENE RANKING BY LEAD NOMINAL p. Within each dataset, genes are ranked by
    the p of their lead (ties by |slope / se|; a gene with no finite p ranks
    last). A within-dataset ranking uses no reference distribution, so it
    does not depend on how well each arm's p values are calibrated. It is
    confounded by the number of tested variants (2,295 to 12,942 per gene),
    because a null gene with many variants has a smaller minimum p by
    chance; the gene set, and so that confounding, is shared by every arm.
    AUC (the Mann-Whitney statistic): U / (n1 n0), U the number of
    (non-null, null) gene pairs in which the non-null gene ranks above the
    null one, ties counting one half; the probability that a random non-null
    gene outranks a random null gene. Per dataset, then the mean over
    datasets with a dataset-bootstrap 95% interval (datasets resampled; not
    reported with fewer than 2 datasets). By band, both classes are
    restricted to the band's genes.
    Power at realized FDP = FDR: gene units pooled over the scenario's
    datasets (every dataset tests the same variants), ordered by that
    ranking; the realized false-discovery proportion FDP(k) = null units
    among the top k / k, from the truth; at the largest k with
    FDP(k) <= FDR that does not split a tie, power = non-null units in the
    top k / all non-null units (by band: of the band's non-null units).
(6) NULL-GENE NOMINAL-P RATE: rejections / tests over the tested variants of
    each dataset's null genes, per channel and alpha, with the gene-clustered
    interval (counts per gene from corrected_null_store.rates_by_gene with
    that dataset's null set), and a dataset-bootstrap interval when there are
    at least 2 datasets. Thinning dilutes the real data's weight-residual
    coupling on the thinned records, so the beta > 0 rates are not a
    calibration result (make_datasets.CANNOT_ANSWER).
    ANCHOR (beta = 0): each hapmixQTL arm's rate at ANCHOR_ALPHA against the
    same arm's stored 100-gene x 200-permutation null run. The anchor is one
    dataset, i.e. ONE record permutation, so the reference is the stored
    run's permutation-to-permutation spread: it passes inside the central
    ANCHOR_CENTRAL of the stored per-permutation rates, reported, never a
    stop: in the 2026-09-26 smoke unit combined sat at the stored 0.5th
    percentile, the edge of that range. A gene-clustered
    interval carries only gene-to-gene spread within the one permutation and
    failed on the total channel of all four arms in the 2026-09-26 smoke,
    whose permutation sits at the 3rd percentile of the stored total-channel
    rates (0.0739 gibbs, 0.0452 unit). The plumbing itself was checked
    exactly the same day by a one-off script, not yet a committed check:
    given the stored run's own permutation 0, the beta = 0 generator path
    reproduces that run's draw 0 (unit and gibbs: no call at 0.05 differs
    among 487,454 tests in any channel; slopes within 5.5e-6 se).
(7) GENE LEVEL: map_cis on every dataset for the hapmixQTL arms, and
    mixqtl_permutation_scan for the mixQTL arms when run_arms.py's timing
    rule admitted it (RESULTS/mixqtl_permutation.json, reported either way).
    Null-gene rate: gene-level p below GENE_LEVEL_ALPHA over the null gene
    units, for pval_beta and pval_perm (mixQTL: pval_perm only; its port has
    no Beta approximation). Power: within each dataset, the Benjamini-Hochberg
    procedure (reject the k smallest p, k the largest with p_(k) <= k FDR / m)
    at FDR over the genes' pval_beta (mixQTL: pval_perm); the share of
    non-null gene units discovered, and the discoveries that are null genes.

WHAT THIS CANNOT ANSWER: make_datasets.CANNOT_ANSWER, copied into the summary.

Output: SUMMARY (atomic JSON, NaN written as null) and a printed table.
Usage: score.py [datasets_dir [results_dir [summary_json]]]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import compare_mixqtl_replication as CM      # noqa: E402
import corrected_null_store as CNS           # noqa: E402
import make_datasets as MD                   # noqa: E402
import run_arms as RA                        # noqa: E402
from tensorqtl.hapmixqtl import LN2          # noqa: E402

SUMMARY = MD.ROOT / 'summary.json'
GENE_DESIGN = MD.D / 'corrected_null_store_20260925' / 'gene_design.tsv'
ANCHOR = {   # arm: (stored summary, its key prefix); the 100-gene x 200-permutation null runs
    'gibbs': (MD.D / 'corrected_null_store_20260925' / 'summary.json', 'drop'),
    'split': (MD.D / 'hybrid_weights_null_20260926' / 'summary.json', 'hybrid'),
    'unit': (MD.D / 'hybrid_weights_null_20260926' / 'summary_unit.json', 'unit'),
    'plus_one': (MD.D / 'hybrid_weights_null_20260926' / 'summary_plus_one.json', 'plus_one'),
}
SEED = 42
N_BOOT = 2000                 # task spec
ALPHAS = CNS.ALPHAS           # 0.05, 0.01, 0.001: null-gene rates
ANCHOR_ALPHA = 0.05           # the only alpha with an anchor pass rule (task spec)
ANCHOR_CENTRAL = 0.99         # report whether the rate is inside the stored central 99% of per-permutation rates: the anchor is ONE permutation
ANCHOR_N_PERM = 200           # draws per arm in the stored null runs
DETECT_ALPHAS = (0.05, 1e-3, 1e-5)   # user decision 2026-09-26 (task E.4)
FDR = 0.05                    # user decision 2026-09-26 (task E.5): power where pooled realized FDP is 0.05
GENE_LEVEL_ALPHA = 0.05       # gene-level null-gene rate
R2_HIGH = 0.8                 # user decision 2026-09-26 (task E.3)
IVW_TOL = 1e-4                # combined slope vs inverse-variance combination of the channel slopes, / se; smoke 2026-09-26 max 2.6e-6 (float32)
BANDS = (('all', 0, np.inf), ('<100', 0, 100), ('100-999', 100, 1000), ('>=1000', 1000, np.inf))  # user decision 2026-09-26
SLOPE = {'combined': ('slope', 'slope_se'), 'allelic': ('slope_a', 'slope_a_se'),
         'total': ('slope_t', 'slope_t_se')}
PVAL = CNS.CHANNELS
TRUTH = {'count': {'combined': 'allelic_truth', 'allelic': 'allelic_truth', 'total': 'total_truth'},
         'pipeline': {'allelic': 'allelic_truth_pipeline', 'total': 'total_truth_pipeline'}}
NULL_COLS = ['phenotype_id', 'variant_id', 'slope', 'slope_se', 'slope_a', 'slope_a_se', 'slope_t', 'slope_t_se']
GENE_BOOT_KEY, DATASET_BOOT_KEY, AUC_BOOT_KEY = 30, 31, 33
JOINT = {'rasqual': MD.ROOT / 'results_rasqual', 'trecase': MD.ROOT / 'results_trecase_asseq'}
JOINT_COLS = CNS.COLS[:5]     # phenotype_id, variant_id, pval_nominal, slope, slope_se
TRECASE_PARTS = {'trec': 'pval_t', 'joint': 'pval_joint', 'ase': 'pval_a'}   # run_trecase_asseq.py OUTPUT
ONE_DF = 2                    # admitted allelic donors at which the through-origin allelic fit has 1 residual df (review 2026-09-27)
NO_ONE_DF = 'without one-df genes'
TAIL_ALPHA = 0.001


def arm_dir(results, sc, arm):
    return (JOINT[arm] if arm in JOINT else results) / sc / arm


def channels(arm, d):
    """The channels scored for an arm: a joint arm's one test per variant is its combined channel."""
    return {'combined': d['combined']} if arm in JOINT else d


def boot(key, n, size=None):
    rng = np.random.default_rng(np.random.SeedSequence(SEED, spawn_key=key))
    return rng.integers(0, n, size=(N_BOOT, size or n))


def load_units(datasets, results):
    """One row per (scenario, dataset, gene): truth, band; checks results match their datasets."""
    meta = json.loads((datasets / 'meta.json').read_text())
    perm = json.loads((results / RA.MIXQTL_PERM_JSON).read_text())
    genes = meta['genes']
    gd = pd.read_csv(GENE_DESIGN, sep='\t').set_index('gene')
    reads = gd.loc[genes, 'median_allele_resolved_reads']
    truth = pd.read_csv(datasets / 'truth.tsv', sep='\t')
    tr = truth.drop_duplicates('gene').set_index('gene').median_hap_reads_real
    if not np.allclose(tr.values, reads.loc[tr.index].values, rtol=1e-12, atol=0):
        raise SystemExit(f'{GENE_DESIGN} median_allele_resolved_reads differs from {datasets / "truth.tsv"}')
    band = pd.cut(reads, [b[1] for b in BANDS[1:]] + [np.inf], right=False, labels=[b[0] for b in BANDS[1:]])
    gene_level_arms = RA.HAPMIX_ARMS + (tuple(RA.MIXQTL_ARMS) if perm['included'] else ())
    rows = []
    for b in meta['betas']:
        sc = f'beta{b}'
        fs = sorted((datasets / sc).glob('rep*.npz'))
        if len(fs) != meta['n_datasets'][str(b)]:
            raise SystemExit(f'{datasets / sc}: {len(fs)} datasets, meta.json says {meta["n_datasets"][str(b)]}')
        for f in fs:
            r, ds = int(f.stem[3:]), dict(np.load(f))
            for arm in RA.ARMS + tuple(JOINT):
                for prefix in ('nominal', 'cis'):
                    p = arm_dir(results, sc, arm) / f'{prefix}_rep{r:03d}.parquet'
                    expected = prefix == 'nominal' or arm in gene_level_arms
                    if p.exists() != expected:
                        raise SystemExit(f'{p} {"is missing" if expected else "exists, but " + RA.MIXQTL_PERM_JSON + " says the mixQTL permutation scan did not run"}')
                    if expected and RA.stored_fingerprint(p) != RA.fingerprint(ds, arm):
                        raise SystemExit(f'{p} does not match {f} and arm {arm} (run_arms.fingerprint)')
            rows.append(pd.DataFrame(dict(
                scenario=sc, beta_abs=b, rep=r, gene=genes, band=band.values.astype(str), reads=reads.values,
                is_null=ds['is_null'], causal_variant=ds['causal_variant'].astype(str), beta=ds['beta'],
                **{c: ds[c] for c in ('allelic_truth', 'total_truth', 'allelic_truth_pipeline',
                                      'total_truth_pipeline')})))
    U = pd.concat(rows, ignore_index=True)
    nn = U[~U.is_null].merge(truth, on=['beta_abs', 'rep', 'gene'], suffixes=('', '_tsv'))
    same = lambda c: np.allclose(nn[c], nn[f'{c}_tsv'], equal_nan=True)
    if len(nn) != len(truth) or len(nn) != int((~U.is_null).sum()) or not (
            (nn.causal_variant == nn.causal_variant_tsv).all() and same('beta') and same('total_truth')
            and same('allelic_truth_pipeline') and same('total_truth_pipeline')):
        raise SystemExit(f'{datasets / "truth.tsv"} disagrees with the datasets\' non-null genes')
    print(f'{len(U):,} dataset-gene units (datasets per scenario {meta["n_datasets"]} x {len(genes)} genes), '
          f'{int((~U.is_null).sum())} non-null; genes per band '
          f'{band.value_counts().reindex([b[0] for b in BANDS[1:]]).to_dict()}; gene-level results for '
          f'{gene_level_arms}; mixQTL permutation scan '
          f'{"included" if perm["included"] else "not run (run_arms timing rule)"}', flush=True)
    return meta, genes, U, perm


def band_genes(genes, U):
    b = U.drop_duplicates('gene').set_index('gene').band.loc[genes].values
    return {name: np.where((b == name) | (name == 'all'))[0] for name, _, _ in BANDS}


def read_results(path, columns):
    """A results parquet with every slope and se in log2 units."""
    d = pd.read_parquet(path, columns=columns)
    unit = RA.stored_unit(path)
    if unit == 'natural log':
        # mixQTL's response is natural log by design; / ln 2 puts its slopes and se in log2 units
        for c in columns:
            if c.startswith('slope'):
                d[c] = d[c].astype(float) / LN2
    elif unit != 'log2':
        raise SystemExit(f'{path}: unknown slope unit {unit!r}')
    d['variant_id'] = d['variant_id'].astype(str)
    return d


def pooled(K, n, gsel, bidx, ridx=None):
    """Pooled rate over genes gsel with gene-clustered (and optionally dataset-clustered) intervals."""
    Kg, ng = K[:, gsel].sum(0), n[:, gsel].sum(0)
    b = Kg[bidx].sum(1) / ng[bidx].sum(1)
    out = dict(rate=float(Kg.sum() / ng.sum()), lo=float(np.quantile(b, .025)), hi=float(np.quantile(b, .975)),
               rejections=int(Kg.sum()), tests=int(ng.sum()))
    if ridx is not None:
        Kr, nr = K[:, gsel].sum(1), n[:, gsel].sum(1)
        d = Kr[ridx].sum(1) / nr[ridx].sum(1)
        out.update(dataset_lo=float(np.quantile(d, .025)), dataset_hi=float(np.quantile(d, .975)))
    return out


def null_calibration(results, U, sc, arm, genes, bsel, bidx, cols=None):
    reps = sorted(U[U.scenario == sc].rep.unique())
    ridx = boot((DATASET_BOOT_KEY,), len(reps)) if len(reps) > 1 else None
    res = {}
    for ch, col in (cols or channels(arm, PVAL)).items():
        K = {al: np.zeros((len(reps), len(genes))) for al in ALPHAS}
        n = np.zeros((len(reps), len(genes)))
        for i, r in enumerate(reps):
            nulls = U[(U.scenario == sc) & (U.rep == r) & U.is_null].gene.tolist()
            k, n[i] = CNS.rates_by_gene([arm_dir(results, sc, arm) / f'nominal_rep{r:03d}.parquet'], genes, col,
                                        gene_filter=nulls)
            for al in ALPHAS:
                K[al][i] = k[al]
        res[ch] = {bn: {str(al): pooled(K[al], n, g, bidx[bn], ridx if bn == 'all' else None) for al in ALPHAS}
                   for bn, g in bsel.items() if n[:, g].sum() > 0}
    return res


def causal_and_leads(results, U, sc, arm):
    """Causal-variant rows of the non-null genes (a joint arm's missing row left NaN), and every gene's lead, per
    dataset (log2 units)."""
    parts, leads = [], []
    cols = (JOINT_COLS if arm in JOINT else CNS.COLS) + (['method'] if arm in RA.MIXQTL_ARMS else [])
    for r, u in U[U.scenario == sc].groupby('rep'):
        d = read_results(arm_dir(results, sc, arm) / f'nominal_rep{r:03d}.parquet', cols)
        nn = u[~u.is_null]
        parts.append(nn.merge(d, left_on=['gene', 'causal_variant'], right_on=['phenotype_id', 'variant_id'],
                              how='left'))
        d['p'] = d.pval_nominal.where(np.isfinite(d.pval_nominal), np.inf)
        d['absstat'] = np.abs(d.slope / d.slope_se)
        top = (d.sort_values(['phenotype_id', 'p', 'absstat'], ascending=[True, True, False], kind='stable')
               .groupby('phenotype_id', sort=False).head(1).set_index('phenotype_id'))
        L = u.set_index('gene')[['scenario', 'rep', 'is_null', 'band', 'causal_variant']].join(
            top[['variant_id', 'p', 'absstat']], how='left')
        if L.variant_id.isna().any():
            raise SystemExit(f'{sc} {arm} rep {r}: no result rows for {int(L.variant_id.isna().sum())} genes')
        leads.append(L.rename(columns={'variant_id': 'lead_variant', 'p': 'lead_p', 'absstat': 'lead_absstat'})
                     .reset_index())
    C = pd.concat(parts, ignore_index=True)
    if C.variant_id.isna().any() and arm not in JOINT:
        raise SystemExit(f'{sc} {arm}: {int(C.variant_id.isna().sum())} causal variants have no result row')
    return C, pd.concat(leads, ignore_index=True)


def ratio_block(ratio, C, genes, bsel, bidx):
    gi = pd.Index(genes)
    ok = np.isfinite(ratio).values
    idx = gi.get_indexer(C.gene[ok])
    s = np.bincount(idx, weights=ratio[ok], minlength=len(genes))
    k = np.bincount(idx, minlength=len(genes)).astype(float)
    out = {}
    for bn, gs in bsel.items():
        if k[gs].sum() == 0:
            continue
        bb = s[gs][bidx[bn]].sum(1) / k[gs][bidx[bn]].sum(1)
        names = set(gi[gs])
        out[bn] = dict(mean=float(s[gs].sum() / k[gs].sum()), lo=float(np.nanquantile(bb, .025)),
                       hi=float(np.nanquantile(bb, .975)), units=int(k[gs].sum()),
                       excluded_nonfinite=int((~ok & C.gene.isin(names).values).sum()))
    return out


def recovery(C, arm, genes, bsel, bidx):
    res = {}
    for ch in channels(arm, SLOPE):
        res[ch] = dict(bias_count=ratio_block(C[SLOPE[ch][0]].astype(float) / C[TRUTH['count'][ch]],
                                              C, genes, bsel, bidx))
        if arm in RA.HAPMIX_ARMS and ch in TRUTH['pipeline']:
            res[ch]['bias_pipeline'] = ratio_block(C[SLOPE[ch][0]].astype(float) / C[TRUTH['pipeline'][ch]],
                                                   C, genes, bsel, bidx)
    return res


def channel_truths(C, arm, scale=None):
    """Per causal unit, the estimand of each channel's slope (scale None: the arm's own); checks the combined slope
    is the IVW of the channels'. A joint arm's one slope has estimand beta (module docstring)."""
    if arm in JOINT:
        return {'combined': C[TRUTH['count']['combined']].values}
    scale = scale or ('pipeline' if arm in RA.HAPMIX_ARMS else 'count')
    ta, tt = C[TRUTH[scale]['allelic']].values, C[TRUTH[scale]['total']].values
    sa, sea = C.slope_a.astype(float).values, C.slope_a_se.astype(float).values
    st, set_ = C.slope_t.astype(float).values, C.slope_t_se.astype(float).values
    with np.errstate(divide='ignore', invalid='ignore'):
        wa = np.where(np.isfinite(sa) & np.isfinite(sea), 1.0 / sea ** 2, 0.0)
        wt = np.where(np.isfinite(st) & np.isfinite(set_), 1.0 / set_ ** 2, 0.0)
        if arm in RA.MIXQTL_ARMS:   # mixQTL's meta estimate is one channel alone unless `method` is 'meta'
            wa, wt = np.where(C.method == 'trc', 0.0, wa), np.where(C.method == 'asc', 0.0, wt)
        w = wa + wt
        ivw = (np.where(wa > 0, wa * sa, 0.0) + np.where(wt > 0, wt * st, 0.0)) / w
        tc = (np.where(wa > 0, wa * ta, 0.0) + np.where(wt > 0, wt * tt, 0.0)) / w
    c, sc = C.slope.astype(float).values, C.slope_se.astype(float).values
    fin = np.isfinite(c)
    dev = np.abs(c[fin] - ivw[fin]) / sc[fin]
    if not (dev <= IVW_TOL).all():
        raise SystemExit(f'{arm}: combined slope differs from the inverse-variance combination of the channel '
                         f'slopes by up to {np.nanmax(dev):.2e} se (IVW_TOL {IVW_TOL}) in '
                         f'{int((~(dev <= IVW_TOL)).sum())} of {int(fin.sum())} causal units')
    return {'combined': tc, 'allelic': ta, 'total': tt}


def gene_sums(g, n_genes, ok, *values):
    """Per-gene counts and sums of each value over the rows where ok."""
    return [np.bincount(g[ok], minlength=n_genes).astype(float)] + [
        np.bincount(g[ok], weights=v[ok], minlength=n_genes) for v in values]


def nonnull_precision(C, Cu, arm, genes):
    """Per channel, per-gene [n, sum z, sum z^2, pairs, sum err^2 arm, sum err^2 unit] at the causal variants, each
    arm against its own truth, then [pairs, sum err^2 arm, sum err^2 unit] with the count-scale truth for both."""
    if not (np.array_equal(C.rep.values, Cu.rep.values) and np.array_equal(C.gene.values, Cu.gene.values)):
        raise SystemExit(f'{arm}: causal units are not in the order of the unit arm\'s')
    g = pd.Index(genes).get_indexer(C.gene)
    T, Tu = channel_truths(C, arm), channel_truths(Cu, 'unit')
    Tc, Tuc = channel_truths(C, arm, 'count'), channel_truths(Cu, 'unit', 'count')
    acc, excluded = {}, {}
    for ch, (b, s) in channels(arm, SLOPE).items():
        x, xu = C[b].astype(float).values, Cu[b].astype(float).values
        e, eu, ec, euc = x - T[ch], xu - Tu[ch], x - Tc[ch], xu - Tuc[ch]
        with np.errstate(divide='ignore', invalid='ignore'):
            z = e / C[s].astype(float).values
        okz, pair, pc = np.isfinite(z), np.isfinite(e) & np.isfinite(eu), np.isfinite(ec) & np.isfinite(euc)
        n, s1, s2 = gene_sums(g, len(genes), okz, z, z ** 2)
        m, a, u = gene_sums(g, len(genes), pair, e ** 2, eu ** 2)
        acc[ch] = np.vstack([n, s1, s2, m, a, u] + gene_sums(g, len(genes), pc, ec ** 2, euc ** 2))
        excluded[ch] = dict(nonfinite_z=int((~okz).sum()), unpaired=int((~pair).sum()), units=len(z))
    return acc, excluded


def null_rows(results, sc, arm, r, nulls):
    d = read_results(arm_dir(results, sc, arm) / f'nominal_rep{r:03d}.parquet',
                     NULL_COLS[:4] if arm in JOINT else NULL_COLS)
    return (d[d.phenotype_id.isin(nulls)].sort_values(['phenotype_id', 'variant_id'], kind='stable')
            .reset_index(drop=True))


def null_precision(results, U, sc, arm, genes):
    """Per channel, per-gene [n, sum z, sum z^2, pairs, sum slope^2 arm, sum slope^2 unit] over null genes' variants."""
    acc = {ch: np.zeros((6, len(genes))) for ch in channels(arm, SLOPE)}
    excluded = {ch: dict(nonfinite_z=0, unpaired=0, tests=0, no_row=0) for ch in channels(arm, SLOPE)}
    for r in sorted(U[U.scenario == sc].rep.unique()):
        nulls = set(U[(U.scenario == sc) & (U.rep == r) & U.is_null].gene)
        d = null_rows(results, sc, arm, r, nulls)
        du = null_rows(results, sc, 'unit', r, nulls)
        no_row = len(du) - len(d)
        if arm in JOINT:   # tests without a joint-arm row are left out, counted as no_row
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
            excluded[ch]['nonfinite_z'] += int((~okz).sum())
            excluded[ch]['unpaired'] += int((~pair).sum())
            excluded[ch]['tests'] += len(z)
            excluded[ch]['no_row'] += no_row
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
    return dict(value=float(fn(*[p[gs].sum() for p in parts])), lo=float(np.nanquantile(b, .025)),
                hi=float(np.nanquantile(b, .975)))


def summarize(acc, bsel, bidx):
    n, s1, s2 = acc[:3]
    ratios = {'ratio_vs_unit': acc[3:6]}
    if len(acc) > 6:   # non-null: also on the count-scale truth (nonnull_precision)
        ratios['ratio_vs_unit_count'] = acc[6:9]
    out = {'sd_z': {}, **{k: {} for k in ratios}}
    for bn, gs in bsel.items():
        if n[gs].sum() > 1:
            out['sd_z'][bn] = dict(**boot_stat((n, s1, s2), gs, bidx[bn], sd_pooled), units=int(n[gs].sum()),
                                   genes=int((n[gs] > 0).sum()))
        for k, (m, a, u) in ratios.items():
            if m[gs].sum() > 0:
                out[k][bn] = dict(**boot_stat((a, u), gs, bidx[bn], sum_ratio), units=int(m[gs].sum()))
    return out


def precision(results, U, sc, arm, genes, bsel, bidx, CL):
    nacc, nexc = null_precision(results, U, sc, arm, genes)
    res = {ch: dict(null=summarize(nacc[ch], bsel, bidx), null_excluded=nexc[ch]) for ch in channels(arm, SLOPE)}
    if CL is not None:
        acc, exc = nonnull_precision(CL[arm][0], CL['unit'][0], arm, genes)
        for ch in channels(arm, SLOPE):
            res[ch].update(nonnull=summarize(acc[ch], bsel, bidx), nonnull_excluded=exc[ch])
    return res


def detection(C, arm):
    C = C[C.variant_id.notna()]   # a joint arm's causal unit without a row is excluded (missing_causal)
    res = {}
    for ch, col in channels(arm, PVAL).items():
        p = C[col].astype(float)
        res[ch] = {'nonfinite_p': int((~np.isfinite(p)).sum())}
        for bn, lo, hi in BANDS:
            m = ((C.reads >= lo) & (C.reads < hi)).values
            if m.any():
                res[ch][bn] = dict(units=int(m.sum()), **{str(al): float((p[m] < al).mean()) for al in DETECT_ALPHAS})
    return res


def ld_r2(dos, rows, a, b):
    """Squared dosage correlation between variant ids a and b, per unit; NaN where either is constant."""
    x, y = dos[[rows[v] for v in a]].astype(float), dos[[rows[v] for v in b]].astype(float)
    xc, yc = x - x.mean(1, keepdims=True), y - y.mean(1, keepdims=True)
    den = np.sqrt((xc ** 2).sum(1) * (yc ** 2).sum(1))
    r2 = np.full(len(a), np.nan)
    ok = den > 0
    r2[ok] = ((xc[ok] * yc[ok]).sum(1) / den[ok]) ** 2
    return r2


def lead_recovery(L, dos, rows):
    N = L[~L.is_null].copy()
    has = np.isfinite(N.lead_p.values)
    same = (N.lead_variant == N.causal_variant).values & has
    r2 = np.full(len(N), np.nan)
    r2[has] = ld_r2(dos, rows, N.lead_variant.values[has], N.causal_variant.values[has])
    r2[same] = 1.0
    N['r2'] = r2
    N['same'] = same
    res = {'no_finite_p': int((~has).sum()), 'r2_undefined': int((has & ~np.isfinite(r2)).sum())}
    for bn, _, _ in BANDS:
        B = N if bn == 'all' else N[N.band == bn]
        if len(B):
            f = np.isfinite(B.r2.values)
            res[bn] = dict(units=len(B), lead_is_causal=float(B.same.mean()),
                           r2_high=float((np.where(f, B.r2.values, 0.0) >= R2_HIGH).mean()),
                           median_r2=float(np.median(B.r2.values[f])) if f.any() else float('nan'),
                           r2_defined=int(f.sum()))
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
    if not n1 or not n0:
        return float('nan')
    return float((rank[nonnull].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def ranking(L, key):
    res = {}
    reps = sorted(L.rep.unique())
    ridx = boot(key, len(reps))
    for bn, _, _ in BANDS:
        B = L if bn == 'all' else L[L.band == bn]
        a = np.array([auc(evidence_rank(x.lead_p.values, x.lead_absstat.values), ~x.is_null.values)
                      for _, x in B.groupby('rep')])
        if not np.isfinite(a).any():
            continue
        m = np.nanmean(a[ridx], axis=1)
        several = len(reps) > 1
        res.setdefault('auc', {})[bn] = dict(mean=float(np.nanmean(a)),
                                             lo=float(np.nanquantile(m, .025)) if several else float('nan'),
                                             hi=float(np.nanquantile(m, .975)) if several else float('nan'),
                                             datasets=int(np.isfinite(a).sum()))
    rank = evidence_rank(L.lead_p.values, L.lead_absstat.values)
    o = np.argsort(-rank, kind='stable')
    null = L.is_null.values[o]
    fdp = np.cumsum(null) / np.arange(1, len(o) + 1)
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


def gene_level(results, U, sc, arm, genes, bsel, bidx):
    """Gene-level null rates and Benjamini-Hochberg power from the cis_repNNN files (section 7)."""
    pcols = ['pval_beta', 'pval_perm'] if arm in RA.HAPMIX_ARMS else ['pval_perm']
    reps = sorted(U[U.scenario == sc].rep.unique())
    K = {c: np.zeros((len(reps), len(genes))) for c in pcols + ['disc', 'false']}
    n0, n1 = np.zeros((len(reps), len(genes))), np.zeros((len(reps), len(genes)))
    nonfinite = 0
    for i, r in enumerate(reps):
        d = pd.read_parquet(results / sc / arm / f'cis_rep{r:03d}.parquet', columns=['phenotype_id'] + pcols)
        if sorted(d.phenotype_id) != sorted(genes) or not d.phenotype_id.is_unique:
            raise SystemExit(f'{sc} {arm} rep {r}: gene-level results do not cover the {len(genes)} genes once each')
        d = d.set_index('phenotype_id').loc[genes]
        null = U[(U.scenario == sc) & (U.rep == r)].set_index('gene').is_null.loc[genes].values
        p = d[pcols[0]].values.astype(float)
        fin = np.isfinite(p)
        disc = np.zeros(len(genes), bool)
        disc[fin] = false_discovery_control(p[fin], method='bh') <= FDR
        nonfinite += int((~fin).sum())
        n0[i], n1[i] = null, ~null
        for c in pcols:
            K[c][i] = null & (d[c].values.astype(float) < GENE_LEVEL_ALPHA)
        K['disc'][i], K['false'][i] = ~null & disc, null & disc
    res = dict(datasets=len(reps), p_for_power=pcols[0], nonfinite_p_for_power=nonfinite,
               discoveries=int(K['disc'].sum() + K['false'].sum()), false_discoveries=int(K['false'].sum()))
    for c in pcols:
        res[f'null_rate_{c}'] = {bn: pooled(K[c], n0, g, bidx[bn]) for bn, g in bsel.items() if n0[:, g].sum() > 0}
    res['power_bh'] = {bn: pooled(K['disc'], n1, g, bidx[bn]) for bn, g in bsel.items() if n1[:, g].sum() > 0}
    return res


def stored_rates(path, prefix, drop):
    """Per-permutation rate at ANCHOR_ALPHA of each channel in a stored null run's draws, and the mean over
    permutations of the rate at TAIL_ALPHA without the genes in drop."""
    files = sorted((path.parent / 'draws').glob(f'{prefix}_*.parquet'))
    if len(files) != ANCHOR_N_PERM:
        raise SystemExit(f'{path.parent}/draws: {len(files)} {prefix} draws, expected {ANCHOR_N_PERM}')
    out, tail = {ch: [] for ch in PVAL}, {ch: [] for ch in PVAL}
    for f in files:
        x = pd.read_parquet(f, columns=['phenotype_id'] + list(PVAL.values()))
        keep = ~x.phenotype_id.isin(drop).values
        for ch, col in PVAL.items():
            v = x[col].values
            ok = np.isfinite(v)
            out[ch].append(float(np.mean(v[ok] < ANCHOR_ALPHA)))
            tail[ch].append(float(np.mean(v[ok & keep] < TAIL_ALPHA)))
    return {ch: np.array(v) for ch, v in out.items()}, {ch: float(np.mean(v)) for ch, v in tail.items()}


def anchor(null0, drop):
    res, ok = {}, True
    q_lo, q_hi = (1 - ANCHOR_CENTRAL) / 2, (1 + ANCHOR_CENTRAL) / 2
    for arm, (path, prefix) in ANCHOR.items():
        stored = json.loads(path.read_text())
        per_perm, tail = stored_rates(path, prefix, drop)
        print(f'anchor reference: {arm} per-permutation rates from {ANCHOR_N_PERM} {prefix} draws in {path.parent}')
        res[arm] = {}
        for ch in PVAL:
            s = stored['rates'][f'{prefix} {ch}']
            res[arm][ch] = {}
            for al in map(str, ALPHAS):
                m = null0[arm][ch]['all'][al]
                row = dict(stored=s[al]['rate'], rate=m['rate'], lo=m['lo'], hi=m['hi'])
                if float(al) == ANCHOR_ALPHA:
                    r = per_perm[ch]
                    row.update(perm_lo=float(np.quantile(r, q_lo)), perm_hi=float(np.quantile(r, q_hi)),
                               percentile=float(100 * np.mean(r <= m['rate'])))
                    row['passed'] = bool(row['perm_lo'] <= m['rate'] <= row['perm_hi'])
                    ok &= row['passed']
                if float(al) == TAIL_ALPHA:
                    row['stored_without_one_df'] = tail[ch]
                res[arm][ch][al] = row
    return res, ok


def fmt(d):
    return f'{d["mean"]:.3f} [{d["lo"]:.3f}, {d["hi"]:.3f}]' if d else '    n/a           '


def fmtv(d, f='{:.3f}'):
    return f'{f.format(d["value"])} [{f.format(d["lo"])}, {f.format(d["hi"])}]' if d else 'n/a'


def fmtr(d, f='{:.4f}'):
    return f'{f.format(d["rate"])} [{f.format(d["lo"])}, {f.format(d["hi"])}]' if d else 'n/a'


def bands_of(d, key, f='{:.3f}'):
    return '/'.join(f.format(d[bn][key]) if bn in d else 'n/a' for bn, _, _ in BANDS[1:])


def report(S):
    arms = S['arms'] + S['joint_arms']
    if S['anchor'] is not None:
        print(f'\nANCHOR (beta = 0, one dataset = one record permutation) at {ANCHOR_ALPHA}: this run\'s rate against '
              f'the stored run\'s {ANCHOR_N_PERM} per-permutation rates (mean; central {ANCHOR_CENTRAL:.0%}; '
              f'percentile of this run); rule: inside the central range')
        for arm, v in S['anchor'].items():
            for ch, r in v.items():
                a = r[str(ANCHOR_ALPHA)]
                print(f'  {arm:8s} {ch:8s} {a["rate"]:.4f} vs {a["stored"]:.4f} [{a["perm_lo"]:.4f}, {a["perm_hi"]:.4f}] '
                      f'at {a["percentile"]:.1f}%  {"PASS" if a["passed"] else "FAIL"}')
        print(f'  anchor {"inside the stored central range in every arm and channel" if S["anchor_passed"] else "OUTSIDE the stored central range in at least one arm and channel"}; '
              f'descriptive only: one permutation cannot test plumbing sharply, the exact reproduction of a stored '
              f'permutation (check_generator.py check (d)) does')
    print('\nJOINT ARMS: causal units without a row (excluded from detection; non-finite in bias and precision): '
          + '; '.join(f'{sc} {a} {n}' for sc, v in S['missing_causal'].items() for a, n in v.items()))
    print('\n(1) BIAS at the causal variant: mean slope / truth [gene-clustered 95%] (<100 / 100-999 / >=1000 '
          'reads); count scale, then pipeline scale (hapmixQTL arms). Channels combined/allelic/total = '
          'mixQTL meta/asc/trc')
    for sc, v in S['recovery'].items():
        for arm in arms:
            for ch in channels(arm, SLOPE):
                r = v[arm][ch]
                bc = r['bias_count']
                bp = r['bias_pipeline'] if 'bias_pipeline' in r else None   # hapmixQTL arms only
                line = (f'  {sc:7s} {arm:17s} {ch:8s} count {fmt(bc["all"] if "all" in bc else None)} ({bands_of(bc, "mean")}; '
                        f'{bc["all"]["units"] if "all" in bc else 0} units, '
                        f'{bc["all"]["excluded_nonfinite"] if "all" in bc else 0} excl.)')
                if bp:
                    line += f'  pipeline {fmt(bp["all"] if "all" in bp else None)} ({bands_of(bp, "mean")})'
                print(line)
    for title, key in (('(2a) SE CALIBRATION: sd of z = (slope - truth) / se, pooled [gene-clustered 95%]; 1 when '
                        'the stated se equals the realized sd, above 1 when it is too small', 'sd_z'),
                       ('(2b) EFFICIENCY vs unit: sum of squared error under the arm / under unit, paired '
                        '[gene-clustered 95%]; below 1 = more precise than unit weights', 'ratio_vs_unit'),
                       ('(2c) EFFICIENCY vs unit on the count-scale truth for the arm and unit alike (the '
                        'cross-method comparison); null as (2b)', 'ratio_vs_unit_count')):
        print(f'\n{title}. non-null: causal variant (hapmixQTL pipeline-scale truth, mixQTL count-scale, joint arms '
              f'beta; combined = IVW of the channel truths); null: every tested variant of the null genes, truth 0. '
              f'(<100 / 100-999 / >=1000 reads)')
        for sc, v in S['precision'].items():
            for arm in arms:
                for ch in channels(arm, SLOPE):
                    r = v[arm][ch]
                    line = f'  {sc:7s} {arm:17s} {ch:8s} '
                    if 'nonnull' in r:
                        d = r['nonnull'][key]
                        line += (f'non-null {fmtv(d["all"] if "all" in d else None)} '
                                 f'({d["all"]["units"] if "all" in d else 0} units; {bands_of(d, "value")})  ')
                    d = r['null'][key if key in r['null'] else 'ratio_vs_unit']
                    line += (f'null {fmtv(d["all"] if "all" in d else None)} '
                             f'({d["all"]["units"] if "all" in d else 0:,} tests; {bands_of(d, "value")})')
                    print(line)
    print(f'\n(3) LEAD RECOVERY (combined / meta): lead = causal, r^2 >= {R2_HIGH}, median r^2 '
          f'(<100 / 100-999 / >=1000 for r^2 >= {R2_HIGH})')
    for sc, v in S['lead'].items():
        for arm in arms:
            r = v[arm]
            a = r['all']
            print(f'  {sc:7s} {arm:17s} lead=causal {a["lead_is_causal"]:.3f}  r2>={R2_HIGH} {a["r2_high"]:.3f} '
                  f'({bands_of(r, "r2_high")})  median r2 {a["median_r2"]:.3f} ({a["r2_defined"]} defined)  '
                  f'units {a["units"]}, no finite p {r["no_finite_p"]}, r2 undefined {r["r2_undefined"]}')
    print(f'\n(4) CAUSAL DETECTION: share of non-null units with p at the causal variant < '
          f'{" / ".join(map(str, DETECT_ALPHAS))} (all genes); at 1e-3 by band (<100 / 100-999 / >=1000)')
    for sc, v in S['detection'].items():
        for arm in arms:
            for ch in channels(arm, PVAL):
                r = v[arm][ch]
                print(f'  {sc:7s} {arm:17s} {ch:8s} ' + ' / '.join(f'{r["all"][str(al)]:.3f}' for al in DETECT_ALPHAS)
                      + f'  (1e-3: {bands_of(r, "0.001")})  non-finite p {r["nonfinite_p"]}')
    print(f'\n(5) GENE RANKING by lead p within dataset: AUC mean over datasets [dataset-bootstrap 95%] '
          f'(<100 / 100-999 / >=1000); power at pooled realized FDP <= {FDR}')
    for sc, v in S['ranking'].items():
        for arm in arms:
            r = v[arm]
            f = r['fdp_matched']
            print(f'  {sc:7s} {arm:17s} AUC {fmt(r["auc"]["all"])} ({bands_of(r["auc"], "mean")})  power '
                  f'{f["all"]["power"]:.3f} ({bands_of(f, "power")}) at {f["discoveries"]} discoveries, '
                  f'{f["false"]} false, lead p <= {f["p_threshold"]:.2e}')
    print('\n(6) NULL-GENE NOMINAL-P RATE at 0.05, tested variants of null genes: rate [gene-clustered 95%] '
          '(<100 / 100-999 / >=1000)')
    for sc, v in S['null'].items():
        for arm in arms:
            print(f'  {sc:7s} {arm:17s} ' + '  '.join(
                f'{ch[:3]} {fmtr(v[arm][ch]["all"]["0.05"])} ('
                + '/'.join(f'{v[arm][ch][bn]["0.05"]["rate"]:.3f}' if bn in v[arm][ch] else 'n/a'
                           for bn, _, _ in BANDS[1:]) + ')'
                for ch in channels(arm, PVAL)))
    print(f'\n(7) GENE LEVEL: null-gene rate of gene-level p < {GENE_LEVEL_ALPHA} [gene-clustered 95%] (null gene '
          f'units); power = share of non-null gene units discovered by Benjamini-Hochberg at FDR {FDR} within '
          f'dataset (hapmixQTL on pval_beta, mixQTL on pval_perm)')
    for sc, v in S['gene_level'].items():
        for arm, g in v.items():
            nb = g['null_rate_pval_beta']['all'] if 'null_rate_pval_beta' in g else None
            npm = g['null_rate_pval_perm']['all']
            pw = g['power_bh']['all'] if 'all' in g['power_bh'] else None
            print(f'  {sc:7s} {arm:17s} null pval_beta {fmtr(nb)}  null pval_perm {fmtr(npm)} ({npm["tests"]} '
                  f'units)  power {fmtr(pw, "{:.3f}")} ({pw["tests"] if pw else 0} units; '
                  f'{g["discoveries"]} discoveries, {g["false_discoveries"]} null)  non-finite p {g["nonfinite_p_for_power"]}')
    m = S['mixqtl_permutation']
    if not m['included']:
        print(f'  mixQTL arms: no gene-level permutation p (run_arms timing rule): mixqtl_permutation_scan took '
              f'{m["seconds"]:.0f} s for {m["genes_done"]} of {m["genes"]} genes ({m["timed_on"]}); in proportion '
              f'to tested variants ~{m["seconds_per_dataset_extrapolated"]:.0f} s per dataset against the '
              f'{m["budget_s"]:.0f} s budget')


def main():
    datasets = Path(sys.argv[1]) if len(sys.argv) > 1 else RA.DATASETS
    results = Path(sys.argv[2]) if len(sys.argv) > 2 else RA.RESULTS
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else SUMMARY
    meta, genes, U, perm = load_units(datasets, results)
    I = CM.load_point_estimate_inputs(gene_list=str(MD.GENES), regions=str(MD.REGIONS))
    if list(I['genes']) != genes:
        raise SystemExit('loader genes differ from meta.json')
    rows = {str(v): i for i, v in enumerate(I['vdf'].index)}
    bsel = band_genes(genes, U)
    keep_a = pd.read_csv(GENE_DESIGN, sep='\t').set_index('gene').n_allelic_keep.loc[genes].values
    one_df = [g for g, k in zip(genes, keep_a) if k == ONE_DF]
    bsel[NO_ONE_DF] = np.flatnonzero(keep_a != ONE_DF)   # last, so the bands' interval streams are unchanged
    bidx = {bn: boot((GENE_BOOT_KEY, b), len(g)) for b, (bn, g) in enumerate(bsel.items())}
    scen = [f'beta{b}' for b in meta['betas']]
    gl_arms = RA.HAPMIX_ARMS + (tuple(RA.MIXQTL_ARMS) if perm['included'] else ())
    arms = RA.ARMS + tuple(JOINT)
    S = dict(datasets=str(datasets), results=str(results), n_datasets=meta['n_datasets'],
             arms=list(RA.ARMS), joint_arms=list(JOINT), joint_results={a: str(p) for a, p in JOINT.items()},
             bands=[b[0] for b in BANDS], n_boot=N_BOOT, seed=SEED, fdr=FDR,
             cannot_answer=MD.CANNOT_ANSWER, units='log2 aFC (mixQTL slopes and se / ln 2; joint arms as stored)',
             mixqtl_permutation=perm, missing_causal={}, one_df_genes=one_df,
             null={}, precision={}, recovery={}, lead={}, detection={}, ranking={}, gene_level={})
    for i, sc in enumerate(scen):
        S['null'][sc] = {arm: null_calibration(results, U, sc, arm, genes, bsel, bidx) for arm in arms}
        CL = None
        if (~U[U.scenario == sc].is_null).any():
            CL = {arm: causal_and_leads(results, U, sc, arm) for arm in arms}
            S['missing_causal'][sc] = {arm: int(CL[arm][0].variant_id.isna().sum()) for arm in JOINT}
            S['recovery'][sc] = {arm: recovery(CL[arm][0], arm, genes, bsel, bidx) for arm in arms}
            S['lead'][sc] = {arm: lead_recovery(CL[arm][1], I['dos'], rows) for arm in arms}
            S['detection'][sc] = {arm: detection(CL[arm][0], arm) for arm in arms}
            S['ranking'][sc] = {arm: ranking(CL[arm][1], (AUC_BOOT_KEY, i)) for arm in arms}
        S['precision'][sc] = {arm: precision(results, U, sc, arm, genes, bsel, bidx, CL) for arm in arms}
        S['gene_level'][sc] = {arm: gene_level(results, U, sc, arm, genes, bsel, bidx) for arm in gl_arms}
        print(f'scored {sc}', flush=True)
    if 'beta0.0' in S['null']:
        S['anchor'], S['anchor_passed'] = anchor(S['null']['beta0.0'], one_df)
        S['trecase_components'] = null_calibration(results, U, 'beta0.0', 'trecase', genes, bsel, bidx, TRECASE_PARTS)
    else:
        S['anchor'], S['anchor_passed'], S['trecase_components'] = None, None, None
    MD.write_atomic(out, lambda fh: fh.write(MD.dumps(S)), 'w')
    report(S)
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()

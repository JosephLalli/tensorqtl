"""Real-data referee, scoring and page: how often each arm's top genes replicate in 135 held-out donors.

Reads referee_replication.py's discovery and replication outputs and referee_trecase.py's TReCASE discovery (one
directory, OUT). Two gene sets: (all) the seven arms without TReCASE on every gene of referee_order.tsv, at KS and at
the K that take the same share of all genes as KS take of the subset; (subset) the TReCASE comparison, every arm
restricted to the first n genes of the order, n from discovery/trecase_subset.json (a seeded random subset), at KS.
TReCASE is scored on the native counts only: the Salmon-input TReCASE was dropped by user decision (2026-09-28;
referee_trecase.py) and its finished genes are not scored.

Per arm and gene: the lead = the tested variant with the smallest pval_nominal (ties by |slope / se|; 06_score.py's
rule), its slope sign (per ALT allele; mixQTL's natural-log slopes divided by ln 2, common.read_results) and the
held-out donors' tensorQTL p and slope at that variant. Two gene-level p: the permutation p where the arm has one
(06_score.CIS_P: pval_beta, the Beta approximation of the 1,000-permutation null, for hapmixQTL and tensorqtl;
pval_perm, the empirical permutation p, for mixQTL) and eigenMT for every arm (Davis et al. 2016: min(1, lead p x
M_eff), M_eff the gene's effective number of independent tests from its genotype correlation,
discovery/eigenmt_m_eff.tsv). Ranking: gene-level p ascending, ties by lead p then |slope / se|; genes without a
finite gene-level p are not ranked (counted).

At K = KS top genes and at the Benjamini-Hochberg set (adjusted gene-level p <= FDR over the ranked genes):
replication share = the share of the K leads with held-out p < REP_ALPHA AND the same slope sign in discovery and
held-out donors; pi1 = 1 - pi0, Storey's estimate of the share of the K held-out p that are not null, pi0 =
min(1, #{p > LAMBDA} / ((1 - LAMBDA) K)); each with a gene-resampling interval (the K genes resampled with
replacement N_BOOT times, 2.5% and 97.5% quantiles). Paired difference against REF: every gene of the set resampled
with replacement N_BOOT times, each arm's top K re-taken on the resample by its own ranking, the difference of the
statistic, 2.5% and 97.5% quantiles (this includes which genes enter the top K). Overlap: |top-K(A) and top-K(B)| / K.
Base rate: the statistic over every ranked gene's lead.

Output: OUT/score/leads_<arm>.parquet (cache of the complete arms' leads with held-out values), OUT/score/score.json,
OUT/score/*.png, OUT/report.html.
"""
import base64
import html
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
import pandas as pd               # noqa: E402
import pyarrow as pa              # noqa: E402
import pyarrow.parquet as pq      # noqa: E402
from scipy.stats import false_discovery_control   # noqa: E402
from sklearn.isotonic import IsotonicRegression    # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent / 'plasmode'))
import common as C                # noqa: E402
import referee_replication as RR  # noqa: E402
import run_hapmixqtl_from_salmon as H   # noqa: E402

OUT = RR.OUT
DISC, SCORE, PAGE = OUT / 'discovery', OUT / 'score', OUT / 'report.html'
KS = (50, 100, 200, 400)     # task
K_SHOW = 200                 # the K the page's prose, overlap figure and class tables quote
REP_ALPHA = 0.05             # task: held-out p < 0.05 with the same sign
LAMBDA = 0.5                 # task: Storey's pi0 from the p above 0.5
FDR = 0.05                   # task: Benjamini-Hochberg 5%
N_BOOT = 2000                # 06_score.N_BOOT
BOOT_KEY, PAIR_KEY = 40, 41  # spawn keys no other script uses (02: 1-3; 03: 4, 5; 06: 30, 33; 01: 10-13; select_stratum_genes: 6; referee: 7, 8)
REF = C.TENSORQTL            # the paired differences' reference: total expression only, as the referee measures
HAPMIX, MIX, TQ = C.HAPMIX_ARMS, tuple(C.MIXQTL_ARMS), C.TENSORQTL
TRECASE = ('trecase_native',)   # the Salmon-input TReCASE is not scored (user decision 2026-09-28)
ARMS = HAPMIX + MIX + (TQ,) + TRECASE
CIS_P = {a: 'pval_perm' if a in MIX else 'pval_beta' for a in HAPMIX + MIX + (TQ,)}   # 06_score.CIS_P
STEP = {a: RR.MIX_BLOCK if a in MIX else RR.BLOCK for a in ARMS}
PLASMODE = C.D / 'plasmode_meier_20260927' / 'summary.json'   # the deep-set plasmode run under Meier (06_score.py's summary)
PLASMODE_BETAS = ('0.4', '0.8')   # its middle and largest planted effects
THREADS = 1                  # this process's CPU threads, beside referee_trecase.py's 96 R processes
RANKINGS = {'perm': 'permutation p', 'eigenmt': 'eigenMT p'}
STRATA = ('at tensorQTL\'s lead', 'elsewhere')   # the channel control's strata: is the arm's lead tensorQTL's lead for the gene
LABEL = {'gibbs': 'gibbs (1/v both channels, shipped)', 'split': 'split (1/v allelic, unit total)',
         'unit': 'unit (weight 1 both channels)', 'plus_one': 'plus_one (1/(v+1) both channels)',
         'mixqtl': 'mixQTL, published cutoffs', 'mixqtl_permissive': 'mixQTL, permissive cutoffs',
         TQ: 'tensorQTL, total only', 'trecase_native': 'TReCASE, native counts'}
COLOR = dict(zip(ARMS, ('#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#8a5a2b', '#4a3aa7')))
MARKER = dict(zip(ARMS, 'osD^vPph'))


def files(arm, n, kind):
    return [DISC / arm / f'{kind}_{lo:05d}_{min(lo + STEP[arm], n):05d}.parquet' for lo in range(0, n, STEP[arm])]


def leads(arm, n):
    """Per gene of the first n of the order: lead variant, p, slope, |slope / se| (module docstring)."""
    out = []
    for f in files(arm, n, 'nominal'):
        if not f.exists():
            raise SystemExit(f'{f} missing: {arm} has not finished the first {n} genes')
        d = C.read_results(f, ['phenotype_id', 'variant_id', 'pval_nominal', 'slope', 'slope_se'])
        d['p'] = d.pval_nominal.where(np.isfinite(d.pval_nominal), np.inf)
        d['absstat'] = np.abs(d.slope / d.slope_se)
        top = d.sort_values(['phenotype_id', 'p', 'absstat'], ascending=[True, True, False], kind='stable')
        out.append(top.groupby('phenotype_id', sort=False).head(1))
    L = pd.concat(out, ignore_index=True).rename(columns={'phenotype_id': 'gene', 'variant_id': 'lead_variant', 'p': 'lead_p',
                                                          'slope': 'lead_slope', 'absstat': 'lead_absstat'})
    L['lead_p'] = L.lead_p.replace(np.inf, np.nan)
    return L[['gene', 'lead_variant', 'lead_p', 'lead_slope', 'lead_absstat']].set_index('gene')


def cis_p(arm, n):
    d = pd.concat([pd.read_parquet(f, columns=['phenotype_id', CIS_P[arm]]) for f in files(arm, n, 'cis')])
    return d.set_index('phenotype_id')[CIS_P[arm]].astype(float)


def held_out(pairs, chrom):
    """Held-out tensorQTL p and slope at (gene, lead_variant) pairs; every pair must be found."""
    parts = []
    for c, g in pairs.groupby(pairs.gene.map(chrom)):
        r = pq.read_table(OUT / 'replication' / f'replication.cis_qtl_pairs.{c}.parquet',
                          columns=['phenotype_id', 'variant_id', 'pval_nominal', 'slope'],
                          filters=[('phenotype_id', 'in', sorted(set(g.gene)))]).to_pandas()
        r['variant_id'] = r.variant_id.astype(str)
        parts.append(g.merge(r.rename(columns={'phenotype_id': 'gene', 'variant_id': 'lead_variant', 'pval_nominal': 'rep_p',
                                               'slope': 'rep_slope'}), on=['gene', 'lead_variant'], how='left'))
    h = pd.concat(parts, ignore_index=True)
    if h.rep_p.isna().any() or len(h) != len(pairs):
        raise SystemExit(f'{int(h.rep_p.isna().sum())} lead variants have no held-out row')
    return h


def arm_table(arm, n, order, m_eff):
    """Leads with held-out values and both gene-level p for the first n genes (cached for the seven complete arms)."""
    cache = SCORE / f'leads_{arm}.parquet'
    full = arm not in TRECASE
    if full and cache.exists():
        L = pd.read_parquet(cache)
    else:
        L = leads(arm, len(order) if full else n)
        found = L[L.lead_p.notna()].reset_index()[['gene', 'lead_variant']]
        L = L.join(held_out(found, order.set_index('gene').chr).set_index('gene')[['rep_p', 'rep_slope']])
        if full:
            L['perm'] = cis_p(arm, len(order)).reindex(L.index)
            SCORE.mkdir(parents=True, exist_ok=True)
            C.write_atomic(cache, lambda fh: L.to_parquet(fh))
    L = L.reindex(order.gene[:n])
    if arm in TRECASE:
        L['perm'] = np.nan
    L['eigenmt'] = np.minimum(1.0, L.lead_p * m_eff.reindex(L.index))
    s = np.sign(L.lead_slope)
    L['rep'] = (L.rep_p < REP_ALPHA) & (s == np.sign(L.rep_slope)) & (s != 0)
    return L


def pi1(p):
    return 1.0 - np.minimum(1.0, (p > LAMBDA).mean(-1) / (1.0 - LAMBDA))


def ranked(L, key):
    """Gene order by the gene-level p `key` (module docstring); genes without a finite value dropped."""
    R = L[np.isfinite(L[key])]
    o = np.lexsort((-R.lead_absstat.fillna(-1.0).values, R.lead_p.values, R[key].values))
    return R.iloc[o]


def stats(R, key):
    """Replication share and pi1 with the K-gene resampling interval."""
    k = len(R)
    idx = np.random.default_rng(np.random.SeedSequence(C.SEED, spawn_key=(BOOT_KEY, k))).integers(0, k, (N_BOOT, k))
    rep, p = R.rep.values.astype(float), R.rep_p.values
    sb, pb = rep[idx].mean(1), pi1(p[idx])
    sig = p < REP_ALPHA
    return dict(K=k, replicated=int(rep.sum()), share=float(rep.mean()), share_lo=float(np.quantile(sb, .025)),
                share_hi=float(np.quantile(sb, .975)), pi1=float(pi1(p)), pi1_lo=float(np.quantile(pb, .025)),
                pi1_hi=float(np.quantile(pb, .975)), heldout_sig=int(sig.sum()),
                sign_agree_among_sig=float(R.rep.values[sig].mean()) if sig.any() else None,
                gene_p_at_K=float(R[key].values[-1]))


def paired(T, arms, key, n_genes, pairs, ks):
    """Paired differences arm minus reference, [(arm, reference)], over whole-set gene resamples (module docstring)."""
    genes = T[arms[0]].index
    pos, rep, p = {}, {}, {}
    for a in arms:
        R = ranked(T[a], key)
        r = pd.Series(np.arange(len(R), dtype=float), index=R.index).reindex(genes).fillna(np.inf).values
        pos[a], rep[a], p[a] = r, T[a].rep.reindex(genes).values.astype(float), T[a].rep_p.reindex(genes).values
    rng = np.random.default_rng(np.random.SeedSequence(C.SEED, spawn_key=(PAIR_KEY, n_genes)))
    D = {a: {K: np.empty((N_BOOT, 2)) for K in ks} for a in arms}
    for b in range(N_BOOT):
        draw = rng.integers(0, n_genes, n_genes)
        for a in arms:
            o = draw[np.argsort(pos[a][draw], kind='stable')]
            for K in ks:
                D[a][K][b] = rep[a][o[:K]].mean(), pi1(p[a][o[:K]])
    res = {}
    for a, ref in pairs:
        for K in ks:
            obs = [ranked(T[x], key).iloc[:K] for x in (a, ref)]
            d = D[a][K] - D[ref][K]
            res.setdefault(a if ref == REF else f'{a} minus {ref}', {})[K] = dict(
                share=float(obs[0].rep.mean() - obs[1].rep.mean()), share_lo=float(np.quantile(d[:, 0], .025)),
                share_hi=float(np.quantile(d[:, 0], .975)), pi1=float(pi1(obs[0].rep_p.values) - pi1(obs[1].rep_p.values)),
                pi1_lo=float(np.quantile(d[:, 1], .025)), pi1_hi=float(np.quantile(d[:, 1], .975)))
    return res


def decompose(T, arms, key, ks):
    """Where an arm's top K differs from REF's: genes in both top K (the share of them where the two leads are the same
    variant, and each arm's replication share there) and genes in only one top K (each one's replication share)."""
    ref = ranked(T[REF], key)
    res = {}
    for a in arms:
        if a == REF:
            continue
        R = ranked(T[a], key)
        for K in ks:
            A, B = R.iloc[:K], ref.iloc[:K]
            both = A.index.intersection(B.index)
            res.setdefault(a, {})[K] = dict(
                both=len(both), same_lead=float((A.lead_variant[both] == B.lead_variant[both]).mean()) if len(both) else None,
                both_share_arm=float(A.rep[both].mean()) if len(both) else None,
                both_share_ref=float(B.rep[both].mean()) if len(both) else None,
                arm_only=int(K - len(both)), arm_only_share=float(A.rep.drop(both).mean()) if len(both) < K else None,
                ref_only_share=float(B.rep.drop(both).mean()) if len(both) < K else None)
    return res


def score_set(T, arms, n, ks=KS):
    """Every statistic of one gene set at top-K sizes ks; printed as it goes."""
    out = dict(n_genes=n, arms=list(arms), rankings={})
    for key in RANKINGS:
        ra = [a for a in arms if np.isfinite(T[a][key]).any()]
        blk = dict(ks=list(ks), arms=ra, unranked={a: int((~np.isfinite(T[a][key])).sum()) for a in ra}, at_K={}, bh={}, base={}, overlap={})
        for a in ra:
            R = ranked(T[a], key)
            gp = R[key].values
            blk['at_K'][a] = {K: dict(stats(R.iloc[:K], key), tied_at_K=int((gp == gp[K - 1]).sum())) for K in ks if K <= len(R)}
            blk.setdefault('gene_p_zero', {})[a] = int((gp == 0).sum())
            adj = false_discovery_control(gp, method='bh')
            m = int((adj <= FDR).sum())
            blk['bh'][a] = dict(stats(R.iloc[:m], key), gene_p_max=float(gp[m - 1])) if m else dict(K=0)
            blk['base'][a] = stats(R, key)
            print(f'{n} genes, {RANKINGS[key]}, {a}: ' + '; '.join(
                f'K={K} share {v["share"]:.3f} [{v["share_lo"]:.3f}, {v["share_hi"]:.3f}] pi1 {v["pi1"]:.3f}'
                for K, v in blk['at_K'][a].items()) + f'; BH {FDR} K={blk["bh"][a]["K"]}' +
                (f' share {blk["bh"][a]["share"]:.3f} pi1 {blk["bh"][a]["pi1"]:.3f}' if blk['bh'][a]['K'] else '') +
                f'; all {len(R)} ranked genes share {blk["base"][a]["share"]:.3f} pi1 {blk["base"][a]["pi1"]:.3f}; unranked '
                f'{blk["unranked"][a]}', flush=True)
        ref = ranked(T[REF], key)
        for a in ra:   # matched-K control for the BH sets: REF's replicated count at the arm's BH size, and the reverse
            m, mr = blk['bh'][a]['K'], blk['bh'][REF]['K']
            blk['bh'][a]['ref_replicated_at_this_K'] = int(ref.rep.values[:m].sum()) if m <= len(ref) else None
            blk['bh'][a]['replicated_at_ref_K'] = int(ranked(T[a], key).rep.values[:mr].sum())
        for K in ks:
            top = {a: set(ranked(T[a], key).index[:K]) for a in ra}
            blk['overlap'][K] = {a: {b: len(top[a] & top[b]) / K for b in ra} for a in ra}
        blk['decompose'] = decompose(T, ra, key, ks)
        pairs = ([(a, REF) for a in ra if a != REF] + [(a, 'gibbs') for a in HAPMIX[1:]] +   # and the weightings against the shipped one
                 [(t, b) for t in TRECASE if t in ra for b in ('split', 'gibbs')])          # and TReCASE against two of them
        diffs = paired(T, ra, key, n, pairs, ks)
        blk['paired'] = {a: v for a, v in diffs.items() if a in ra}
        blk['paired_other'] = {a: v for a, v in diffs.items() if a not in ra}
        for a, v in blk['paired_other'].items():
            print(f'{n} genes, {RANKINGS[key]}, paired {a}: ' + '; '.join(
                f'K={K} share {d["share"]:+.3f} [{d["share_lo"]:+.3f}, {d["share_hi"]:+.3f}] pi1 {d["pi1"]:+.3f} '
                f'[{d["pi1_lo"]:+.3f}, {d["pi1_hi"]:+.3f}]' for K, d in v.items()), flush=True)
        out['rankings'][key] = blk
    return out


def total_curve(L):
    """Replication probability as a function of total-expression discovery evidence: the isotonic regression (the
    best-fitting non-decreasing step function) of tensorQTL's leads' replication on -log10 of their discovery p, over the
    genes of L. Returns the fitted function of p."""
    x = -np.log10(np.clip(L.lead_p.values, 1e-300, 1.0))
    fit = IsotonicRegression(increasing=True, out_of_bounds='clip').fit(x, L.rep.values.astype(float))
    return lambda p: fit.predict(-np.log10(np.clip(np.asarray(p, float), 1e-300, 1.0)))


def tq_p_at_leads(T, arms, n):
    """tensorQTL's discovery p at each arm's top-K_SHOW (eigenMT) lead variant, indexed (gene, lead_variant)."""
    want = pd.concat([ranked(T[a], 'eigenmt').iloc[:K_SHOW][['lead_variant']] for a in arms]).reset_index().drop_duplicates()
    genes = sorted(set(want.gene))
    d = pd.concat([pd.read_parquet(f, columns=['phenotype_id', 'variant_id', 'pval_nominal'], filters=[('phenotype_id', 'in', genes)])
                   for f in files(TQ, n, 'nominal')])
    d['variant_id'] = d.variant_id.astype(str)
    q = want.merge(d.rename(columns={'phenotype_id': 'gene', 'variant_id': 'lead_variant', 'pval_nominal': 'q'}),
                   on=['gene', 'lead_variant'], how='left')
    if q.q.isna().any():
        raise SystemExit(f'{int(q.q.isna().sum())} top-{K_SHOW} lead variants have no tensorQTL discovery row')
    return q.set_index(['gene', 'lead_variant']).q


def expected_vs_observed(rep, e):
    """Replicated count against the count the total-expression evidence predicts, and observed minus expected share with
    the K-gene resampling interval (each gene's replication and expectation resampled together)."""
    k = len(rep)
    idx = np.random.default_rng(np.random.SeedSequence(C.SEED, spawn_key=(BOOT_KEY, k))).integers(0, k, (N_BOOT, k))
    d = rep[idx].mean(1) - e[idx].mean(1)
    return dict(K=k, replicated=int(rep.sum()), expected=float(e.sum()), diff=float(rep.mean() - e.mean()),
                diff_lo=float(np.quantile(d, .025)), diff_hi=float(np.quantile(d, .975)))


def channel_split(T, n, arms, q, curve):
    """The objection that a total-expression referee cannot credit the allelic channel, measured: for the arms with two
    channels (TReCASE: its allelic-only and total-count tests), each top-K lead is allelic-led (its allelic p below its
    total p), total-led, or total-only (no allelic p), and each class's replication share is taken with the K-gene
    resampling interval. At K_SHOW (eigenMT ranking), each class's replicated count is also set against the count its
    total-expression evidence predicts: curve(tensorQTL's discovery p at the lead variant), summed over the class, in two
    strata: leads that are tensorQTL's own lead for the gene (where tensorQTL's p is its gene minimum, as on the curve's
    fitting points) and leads elsewhere (where a p of the same size is less selected, so the expectation runs low)."""
    res = {}
    for a in arms:
        ch = pd.concat([pd.read_parquet(f, columns=['phenotype_id', 'variant_id', 'pval_a', 'pval_t']) for f in files(a, n, 'nominal')])
        ch['variant_id'] = ch.variant_id.astype(str)
        L = T[a].reset_index().merge(ch.rename(columns={'phenotype_id': 'gene', 'variant_id': 'lead_variant'}),
                                     on=['gene', 'lead_variant'], how='left').set_index('gene')
        L['cls'] = np.where(L.pval_a.isna(), 'total-only', np.where(L.pval_a < L.pval_t, 'allelic-led', 'total-led'))
        for key in RANKINGS:
            R = ranked(L, key)
            for K in KS:
                top = R.iloc[:K]
                res.setdefault(a, {}).setdefault(key, {})[K] = {c: stats(g, key) for c, g in top.groupby('cls')}
        top = ranked(L, 'eigenmt').iloc[:K_SHOW]
        e = curve(q.loc[list(zip(top.index, top.lead_variant))].values)
        at = top.lead_variant.values == T[TQ].lead_variant.reindex(top.index).values
        res[a]['expected'] = {c: {s: expected_vs_observed(top.rep.values[m].astype(float), e[m])
                                  for s, m in ((STRATA[0], (top.cls.values == c) & at), (STRATA[1], (top.cls.values == c) & ~at))
                                  if m.any()} for c in sorted(set(top.cls))}
        v, x = res[a]['eigenmt'][K_SHOW], res[a]['expected']
        print(f'channel split, {a}, eigenMT top {K_SHOW}: ' + '; '.join(
            f'{c} {d["K"]} share {d["share"]:.3f} [{d["share_lo"]:.3f}, {d["share_hi"]:.3f}]; ' + ', '.join(
                f'{s}: {y["K"]} leads, replicated {y["replicated"]} against {y["expected"]:.1f} expected from tensorQTL p, share '
                f'difference {y["diff"]:+.3f} [{y["diff_lo"]:+.3f}, {y["diff_hi"]:+.3f}]' for s, y in x[c].items())
            for c, d in v.items()), flush=True)
    return res


def trecase_missing(n):
    """Per input: tests, tests without a final p, of which asSeq chose the joint test and its fit failed, genes affected."""
    out = {}
    for a in TRECASE:
        d = pd.concat([pd.read_parquet(f, columns=['phenotype_id', 'pval_nominal', 'final_stat', 'pval_joint'])
                       for f in files(a, n, 'nominal')])
        na = d[d.pval_nominal.isna()]
        out[a] = dict(tests=len(d), final_na=len(na), joint_chosen_joint_failed=int(((na.final_stat == 'joint') &
                                                                                   na.pval_joint.isna()).sum()),
                      genes_affected=int(na.phenotype_id.nunique()), genes=int(d.phenotype_id.nunique()))
    print(f'TReCASE tests without a final p: {out}', flush=True)
    return out


def referee_source(genes):
    """Per-gene Spearman correlation over the donors in both cohorts between the referee phenotype (225 run) and log2 CPM
    of each discovery count source: Salmon point-estimate totals (personalized transcriptome) and featureCounts totals
    (STAR alignments), each over its own edgeR effective library size. Which source the referee tracks."""
    dm = json.loads(RR.DATASET_MANIFEST.read_text())
    man = pd.read_csv(dm['files']['sample_manifest']['path'], sep='\t', dtype=str)
    bed = pd.read_csv(dm['files']['phenotype_bed']['path'], sep='\t', index_col=3).rename(
        columns=dict(zip(man.SubjectID, man.matchingDNALibrary)))
    samples = (RR.CACHE / 'samples.txt').read_text().split()
    shared = [s for s in samples if s in bed.columns]
    gid = pd.read_csv(OUT / 'genes' / 'referee_order.tsv', sep='\t').set_index('gene').gene_id
    ref = bed.loc[gid.loc[genes], shared].to_numpy(float)
    gi = {g: i for i, g in enumerate((RR.CACHE / 'genes.txt').read_text().split())}
    si = [samples.index(s) for s in shared]
    sal = np.load(Path(RR.CM.PE) / 'pT.npy', mmap_mode='r')[[gi[g] for g in genes]][:, si]
    sal_eff = H.read_edger_dir(Path(RR.CM.PE) / 'edger', shared)[0]
    nat = pd.read_parquet(C.D / 'native_counts_stranded_20260928' / 'totals.parquet').loc[genes, shared].to_numpy(float)
    nat_eff = H.read_edger_dir(OUT / 'trecase_work' / 'native_edger', shared)[0]
    rank = lambda M: np.argsort(np.argsort(M, 1), 1).astype(float)   # noqa: E731  ties broken by position
    def rho(M):
        a, b = rank(M), rank(ref)
        a -= a.mean(1, keepdims=True)
        b -= b.mean(1, keepdims=True)
        return (a * b).sum(1) / np.sqrt((a * a).sum(1) * (b * b).sum(1))
    r = {'salmon': rho(np.log2(sal / sal_eff * 1e6 + 1)), 'native': rho(np.log2(nat / nat_eff * 1e6 + 1))}
    out = dict(donors=len(shared), genes=len(genes), **{k: np.nanquantile(v, [0.25, 0.5, 0.75]).tolist() for k, v in r.items()},
               native_higher=float(np.nanmean(r['native'] > r['salmon'])))
    print(f'referee source, {len(shared)} donors in both cohorts, {len(genes):,} genes: Spearman with the referee phenotype, '
          f'quartiles, Salmon {np.round(out["salmon"], 3).tolist()}, featureCounts {np.round(out["native"], 3).tolist()}; '
          f'featureCounts higher in {out["native_higher"]:.3f} of genes', flush=True)
    return out


def orientation(T, arms):
    """Slope-sign agreement at shared leads: every arm against tensorqtl, where both chose the same variant."""
    res = {}
    for a in arms:
        if a == TQ:
            continue
        j = T[a][['lead_variant', 'lead_slope', 'lead_p']].join(T[TQ][['lead_variant', 'lead_slope', 'lead_p']], rsuffix='_tq')
        s = j[(j.lead_variant == j.lead_variant_tq) & (j.lead_p_tq < 1e-6)]
        res[a] = dict(shared_strong_leads=len(s), same_sign=float((np.sign(s.lead_slope) == np.sign(s.lead_slope_tq)).mean())
                      if len(s) else None)
    print(f'orientation (same lead variant as tensorqtl, tensorqtl lead p < 1e-6): {res}', flush=True)
    return res


def main():
    pa.set_cpu_count(THREADS)
    order = pd.read_csv(OUT / 'genes' / 'referee_order.tsv', sep='\t', dtype={'chr': str})
    sub = json.loads((DISC / 'trecase_subset.json').read_text())
    n = sub['n_genes']
    m_eff = pd.read_csv(DISC / 'eigenmt_m_eff.tsv', sep='\t').set_index('gene').m_eff.astype(float)
    N, seven = len(order), HAPMIX + MIX + (TQ,)
    print(f'gene sets: all {N:,} genes (seven arms); the TReCASE subset, the first {n:,} of the seeded order (every arm); '
          f'M_eff for {len(m_eff):,} genes', flush=True)
    TA = {a: arm_table(a, N, order, m_eff) for a in seven}
    T = {a: arm_table(a, n, order, m_eff) for a in ARMS}
    for name, X in (('all', TA), ('subset', T)):
        for a, L in X.items():
            print(f'{name} {a}: {len(L):,} genes, lead found {int(L.lead_p.notna().sum()):,}, finite permutation p '
                  f'{int(np.isfinite(L.perm).sum()):,}, finite eigenMT p {int(np.isfinite(L.eigenmt).sum()):,}', flush=True)
    ks_all = tuple(sorted(set(KS) | {round(K * N / n) for K in KS}))   # KS, and the K taking the share of all genes KS take of the subset
    curve = total_curve(TA[TQ])   # tensorQTL's replication against its discovery p, all genes; used for both gene sets
    S = dict(subset=sub, orientation=orientation(T, ARMS), referee_source=referee_source(list(order.gene)),
             sets={'all': score_set(TA, seven, N, ks_all), 'subset': score_set(T, ARMS, n)},
             channel={'all': channel_split(TA, N, HAPMIX + MIX, tq_p_at_leads(TA, HAPMIX + MIX, N), curve),
                      'subset': channel_split(T, n, HAPMIX + MIX + TRECASE, tq_p_at_leads(T, HAPMIX + MIX + TRECASE, n), curve)},
             total_curve={f'{p:g}': float(curve([p])[0]) for p in (1e-3, 1e-4, 1e-6, 1e-8, 1e-12, 1e-20)})
    S['total_curve_check'] = {name: expected_vs_observed(top.rep.values.astype(float), curve(top.lead_p.values))   # the curve at
                              for name, X in (('all', TA), ('subset', T))   # tensorQTL's own top K_SHOW, the control's zero
                              for top in [ranked(X[TQ], 'eigenmt').iloc[:K_SHOW]]}
    print(f'tensorQTL replication against its discovery p (isotonic, all genes): {S["total_curve"]}; tensorQTL\'s own top '
          f'{K_SHOW} observed against expected: {S["total_curve_check"]}', flush=True)
    S['trecase'] = json.loads((DISC / 'trecase_summary.json').read_text())
    S['trecase']['missing'] = trecase_missing(n)
    S['trecase']['failure_messages'] = {   # asSeq's error line per failed gene (referee_trecase.py's <gene>_failed.txt)
        inp: {p.name[:-len('_failed.txt')]: next((ln.strip() for ln in p.read_text().splitlines()[1:] if ln.startswith('  ')), '')
              for p in sorted((OUT / 'trecase_work' / inp).glob('*/out/*_failed.txt'))} for inp in ('native',)}
    st = pd.concat([pd.read_csv(p, sep='\t') for p in (OUT / 'trecase_work' / 'native').glob('*/out/*_status.tsv')])
    code = st.set_index('gene').yFailBaselineModel.astype(int)   # asSeq trecase.c: (1 - useTReC) + 2 (1 - useASE)
    S['trecase']['baseline_failed'] = dict(allelic_only=int((code == 2).sum()), trec_only=int((code == 1).sum()),
                                           both=int((code == 3).sum()), trec_genes=sorted(code.index[(code & 1) == 1]))
    print(f'TReCASE baseline fits failed: {S["trecase"]["baseline_failed"]}', flush=True)
    S['facts'] = json.loads((OUT / 'facts.json').read_text())
    S['plasmode'] = {'ranking': json.loads(PLASMODE.read_text())['ranking'], 'source': str(PLASMODE)}
    C.write_json(SCORE / 'score.json', S)
    print(f'wrote {SCORE / "score.json"}', flush=True)
    report(S)


# ---------------------------------------------------------------------------------------------------------------- page
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e1e0d9'
CSS = '''
:root { --ink: #0b0b0b; --ink2: #52514e; --rule: #e1e0d9; --bg: #fcfcfb; --tint: #f3f2ee; }
body { background: var(--bg); color: var(--ink); font: 15px/1.55 -apple-system, "Segoe UI", Roboto, Helvetica, Arial,
       sans-serif; margin: 0; padding: 0 16px; }
main { max-width: 1080px; margin: 32px auto 64px; }
h1 { font-size: 26px; margin-bottom: 4px; } h2 { font-size: 20px; margin-top: 40px; border-bottom: 1px solid var(--rule);
padding-bottom: 4px; } h3 { font-size: 16px; margin-top: 28px; }
p, li { max-width: 900px; } .sub { color: var(--ink2); margin-top: 0; }
table { border-collapse: collapse; font-size: 12.5px; margin: 12px 0 18px; display: block; overflow-x: auto; }
th, td { border-bottom: 1px solid var(--rule); padding: 4px 8px; text-align: left; vertical-align: top; }
th { background: var(--tint); font-weight: 600; } td { font-variant-numeric: tabular-nums; }
figure { margin: 16px 0 24px; } figure img { max-width: 100%; height: auto; }
figcaption { color: var(--ink2); font-size: 13px; max-width: 900px; }
'''


def style(ax, ylabel=None):
    ax.grid(axis='y', color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK, fontsize=10)


def save(fig, name):
    path = SCORE / f'{name}.png'
    tmp = path.with_name(path.name + '.tmp')
    fig.savefig(tmp, dpi=130, bbox_inches='tight', format='png')
    plt.close(fig)
    os.replace(tmp, path)
    return path


def img(path, caption):
    return (f'<figure><img alt="{html.escape(caption, quote=True)}" src="data:image/png;base64,'
            f'{base64.b64encode(path.read_bytes()).decode()}"><figcaption>{caption}</figcaption></figure>')


def table(head, rows):
    h = ''.join(f'<th>{x}</th>' for x in head)
    b = ''.join('<tr>' + ''.join(f'<td>{x}</td>' for x in r) + '</tr>' for r in rows)
    return f'<table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table>'


def ci(v, k, d=3):
    return f'{v[k]:.{d}f} [{v[k + "_lo"]:.{d}f}, {v[k + "_hi"]:.{d}f}]'


def fig_vs_k(blk_set, stat, name, ylabel):
    """stat against K, one panel per ranking, one series per arm, the base rate as a dashed line per arm."""
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, key in zip(axs, RANKINGS):
        blk = blk_set['rankings'][key]
        for j, a in enumerate(blk['arms']):
            ks = [K for K in blk['ks'] if K in blk['at_K'][a]]
            x = np.arange(len(ks)) + (j - (len(blk['arms']) - 1) / 2) * 0.07
            y = [blk['at_K'][a][K][stat] for K in ks]
            lo = [blk['at_K'][a][K][stat + '_lo'] for K in ks]
            hi = [blk['at_K'][a][K][stat + '_hi'] for K in ks]
            ax.errorbar(x, y, yerr=[np.subtract(y, lo), np.subtract(hi, y)], fmt='none', ecolor=COLOR[a], elinewidth=1.1)
            ax.plot(x, y, color=COLOR[a], lw=1.4, alpha=0.8)
            ax.plot(x, y, MARKER[a], color=COLOR[a], ms=6.5, mec='white', mew=0.8, ls='none', label=LABEL[a])
            ax.axhline(blk['base'][a][stat], color=COLOR[a], lw=0.8, ls=':', alpha=0.7)
        ax.set_xticks(range(len(blk['ks'])), [f'{K:,}' for K in blk['ks']])
        ax.set_xlabel('top K genes', color=INK, fontsize=10)
        ax.set_title(f'ranked by {RANKINGS[key]}', fontsize=11, color=INK)
        style(ax, ylabel if key == 'perm' else None)
    h, lab = axs[1].get_legend_handles_labels()
    fig.legend(h, lab, loc='lower center', bbox_to_anchor=(0.5, -0.13), ncol=3, fontsize=8.5, frameon=False)
    return save(fig, name)


def fig_paired(blk_set, name):
    """Paired difference against REF at each K, eigenMT ranking, both statistics."""
    blk = blk_set['rankings']['eigenmt']
    arms = [a for a in blk['arms'] if a != REF]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, stat, yl in zip(axs, ('share', 'pi1'), ('replication share minus tensorQTL\'s', 'pi1 minus tensorQTL\'s')):
        for j, a in enumerate(arms):
            x = np.arange(len(blk['ks'])) + (j - (len(arms) - 1) / 2) * 0.08
            v = [blk['paired'][a][K] for K in blk['ks']]
            y = [d[stat] for d in v]
            ax.errorbar(x, y, yerr=[[d[stat] - d[stat + '_lo'] for d in v], [d[stat + '_hi'] - d[stat] for d in v]],
                        fmt=MARKER[a], color=COLOR[a], ms=6, mec='white', mew=0.8, elinewidth=1.1, label=LABEL[a])
        ax.axhline(0, color=INK, lw=0.8)
        ax.set_xticks(range(len(blk['ks'])), [f'{K:,}' for K in blk['ks']])
        ax.set_xlabel('top K genes (eigenMT ranking)', color=INK, fontsize=10)
        style(ax, yl)
    h, lab = axs[0].get_legend_handles_labels()
    fig.legend(h, lab, loc='lower center', bbox_to_anchor=(0.5, -0.13), ncol=3, fontsize=8.5, frameon=False)
    return save(fig, name)


def fig_overlap(blk_set, K, name):
    fig, axs = plt.subplots(1, 2, figsize=(12, 5.2), gridspec_kw=dict(width_ratios=[7, 9]))
    for ax, key in zip(axs, RANKINGS):
        blk = blk_set['rankings'][key]
        ra = blk['arms']
        M = np.array([[blk['overlap'][K][a][b] for b in ra] for a in ra])
        ax.imshow(M, cmap='Blues', vmin=0, vmax=1)
        for i in range(len(ra)):
            for j in range(len(ra)):
                ax.text(j, i, f'{M[i, j]:.2f}', ha='center', va='center', fontsize=7.5, color='white' if M[i, j] > 0.6 else INK)
        ax.set_xticks(range(len(ra)), [a.replace('_', ' ') for a in ra], rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(len(ra)), [a.replace('_', ' ') for a in ra], fontsize=8)
        ax.set_title(f'top {K} by {RANKINGS[key]}', fontsize=11, color=INK)
    fig.tight_layout()
    return save(fig, name)


def k_table(blk_set, key):
    blk = blk_set['rankings'][key]
    rows = []
    for a in blk['arms']:
        for K in blk['ks']:
            v = blk['at_K'][a].get(K)
            if v:
                rows.append([LABEL[a], K, f'{v["gene_p_at_K"]:.2g}', v['tied_at_K'], f'{v["replicated"]} / {K}', ci(v, 'share'),
                             ci(v, 'pi1'), f'{v["sign_agree_among_sig"]:.3f}' if v['sign_agree_among_sig'] is not None else '-'])
        b = blk['base'][a]
        rows.append([LABEL[a], f'all {b["K"]:,}', '-', '-', f'{b["replicated"]:,} / {b["K"]:,}', ci(b, 'share'), ci(b, 'pi1'),
                     f'{b["sign_agree_among_sig"]:.3f}'])
    return table(['arm', 'K', f'{RANKINGS[key]} of the K-th gene', 'genes sharing that p', 'replicated',
                  'replication share [95% interval]', 'pi1 [95% interval]', 'same sign among held-out p &lt; 0.05'], rows)


def bh_table(blk_set):
    rows = []
    for key in RANKINGS:
        blk = blk_set['rankings'][key]
        for a in blk['arms']:
            v = blk['bh'][a]
            ref_k = blk['bh'][REF]['K']
            rows.append([RANKINGS[key], LABEL[a], v['K']] + ([
                f'{v["gene_p_max"]:.2g}', f'{v["replicated"]} / {v["K"]}', ci(v, 'share'), ci(v, 'pi1'),
                v['ref_replicated_at_this_K'] if v['ref_replicated_at_this_K'] is not None else '-',
                f'{v["replicated_at_ref_K"]} / {ref_k}'] if v['K'] else ['-'] * 6))
    return table(['ranking', 'arm', 'genes at BH 5%', 'largest gene-level p called', 'replicated', 'replication share',
                  'pi1', 'tensorQTL\'s top genes of the same number: replicated', 'this arm at tensorQTL\'s BH size: replicated'],
                 rows)


def decompose_table(blk_set, key):
    blk = blk_set['rankings'][key]
    f3 = lambda x: '-' if x is None else f'{x:.3f}'   # noqa: E731
    rows = [[LABEL[a], K, d['both'], f3(d['same_lead']), f3(d['both_share_arm']), f3(d['both_share_ref']), d['arm_only'],
             f3(d['arm_only_share']), f3(d['ref_only_share'])] for a, per in blk['decompose'].items() for K, d in per.items()]
    return table(['arm', 'K', 'genes in both top K', 'same lead variant', 'replicated there: this arm\'s lead',
                  'replicated there: tensorQTL\'s lead', 'genes only in this arm\'s top K', 'replicated: this arm only',
                  'replicated: tensorQTL only'], rows)


def paired_table(blk_set, key):
    blk = blk_set['rankings'][key]
    cell = lambda d, s: f'{d[s]:+.3f} [{d[s + "_lo"]:+.3f}, {d[s + "_hi"]:+.3f}]'   # noqa: E731
    rows = [[f'{LABEL[a]} minus tensorQTL', s] + [cell(blk['paired'][a][K], s) for K in blk['ks']]
            for a in blk['arms'] if a != REF for s in ('share', 'pi1')]
    for name, v in blk['paired_other'].items():
        a, b = name.split(' minus ')
        rows += [[f'{LABEL[a]} minus {LABEL[b]}', s] + [cell(v[K], s) for K in blk['ks']] for s in ('share', 'pi1')]
    return table(['difference', 'statistic'] + [f'K = {K:,}' for K in blk['ks']], rows)


def span(blk, stat, K):
    """(lowest arm, value), (highest arm, value) of a statistic at K."""
    v = {a: blk['at_K'][a][K][stat] for a in blk['arms'] if K in blk['at_K'][a]}
    lo, hi = min(v, key=v.get), max(v, key=v.get)
    return (lo, v[lo]), (hi, v[hi])


def clear_of_zero(blk, stat, which='paired'):
    """[(name, K, difference, lo, hi)] of the paired differences (against tensorQTL, or the others) whose interval excludes 0."""
    return [(a, K, d[stat], d[stat + '_lo'], d[stat + '_hi']) for a, per in blk[which].items() for K, d in per.items()
            if d[stat + '_lo'] > 0 or d[stat + '_hi'] < 0]


def fmt_clear(rows, stat):
    if not rows:
        return 'none'
    name = lambda a: ' minus '.join(LABEL[x] for x in a.split(' minus '))   # noqa: E731
    return '; '.join(f'{name(a)} at K = {K:,}: {d:+.3f} [{lo:+.3f}, {hi:+.3f}]' for a, K, d, lo, hi in rows)


def ranges_text(blk, name):
    out = []
    for K in (blk['ks'][0], blk['ks'][-1]):
        (sa, sv), (sb, sw) = span(blk, 'share', K)
        (pa, pv), (pb, pw) = span(blk, 'pi1', K)
        out.append(f'at K = {K} the replication share runs from {sv:.3f} ({LABEL[sa]}) to {sw:.3f} ({LABEL[sb]}) and pi1 '
                   f'from {pv:.3f} ({LABEL[pa]}) to {pw:.3f} ({LABEL[pb]})')
    base = {a: blk['base'][a]['share'] for a in blk['arms']}
    return (f'<p>Ranked by {name}, ' + '; '.join(out) + f'. Over all ranked genes (the base rate) the replication share is '
            f'{min(base.values()):.3f} to {max(base.values()):.3f}.</p>')



def report(S):
    A, B = S['sets']['all'], S['sets']['subset']
    figs = dict(share=fig_vs_k(A, 'share', 'fig_share', 'replication share'),
                pi1=fig_vs_k(A, 'pi1', 'fig_pi1', 'pi1 of the held-out p'),
                paired=fig_paired(A, 'fig_paired'), overlap=fig_overlap(A, K_SHOW, 'fig_overlap'),
                sub_share=fig_vs_k(B, 'share', 'fig_share_subset', 'replication share'),
                sub_paired=fig_paired(B, 'fig_paired_subset'))
    body = page_text(S, figs)
    page = (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" '
            f'content="width=device-width, initial-scale=1"><title>Held-out replication referee</title><style>{CSS}</style>'
            f'</head><body><main>{body}</main></body></html>')
    C.write_atomic(PAGE, lambda fh: fh.write(page.encode()))
    print(f'wrote {PAGE} ({PAGE.stat().st_size:,} bytes)', flush=True)


def iv(d, s='share', f='+.3f'):
    return f'{d[s]:{f}} [{d[s + "_lo"]:{f}}, {d[s + "_hi"]:{f}}]'


def deep_k(S):
    """The all-gene K taking the share of all genes that K_SHOW takes of the subset."""
    return round(K_SHOW * S['subset']['genes_all'] / S['subset']['n_genes'])


def RESULTS_TEXT(S, which):
    R = S['sets'][which]['rankings']
    return ''.join(ranges_text(R[key], RANKINGS[key]) for key in RANKINGS)


def BH_TEXT(S, which):
    R = S['sets'][which]['rankings']
    parts = []
    for key in RANKINGS:
        k = {a: R[key]['bh'][a]['K'] for a in R[key]['arms']}
        parts.append(f'by {RANKINGS[key]} the Benjamini-Hochberg 5% sets hold {min(k.values()):,} ({LABEL[min(k, key=k.get)]}) to '
                     f'{max(k.values()):,} ({LABEL[max(k, key=k.get)]}) genes')
    return ('<p>' + '; '.join(parts) + '. These sets differ in size between arms, so their replication shares are not at '
            'matched K: an arm that calls more genes reaches weaker ones. The last two columns put each arm next to tensorQTL '
            'at the same list length.</p>')


def PAIRED_TEXT(S, which):
    R = S['sets'][which]['rankings']
    return ''.join(
        f'<p>Ranked by {RANKINGS[key]}, differences from tensorQTL whose 95% interval excludes zero: replication share '
        f'{fmt_clear(clear_of_zero(R[key], "share"), "share")}; pi1 {fmt_clear(clear_of_zero(R[key], "pi1"), "pi1")}. '
        f'Weightings against gibbs' + (', and TReCASE against split and gibbs,' if TRECASE[0] in R[key]['arms'] else '') +
        f' whose interval excludes zero: replication share '
        f'{fmt_clear(clear_of_zero(R[key], "share", "paired_other"), "share")}; pi1 '
        f'{fmt_clear(clear_of_zero(R[key], "pi1", "paired_other"), "pi1")}.</p>' for key in RANKINGS)


def OVERLAP_TEXT(S, which):
    blk = S['sets'][which]['rankings']['eigenmt']
    ra = blk['arms']
    pairs = {(a, b): blk['overlap'][K_SHOW][a][b] for i, a in enumerate(ra) for b in ra[i + 1:]}
    lo, hi = min(pairs, key=pairs.get), max(pairs, key=pairs.get)
    return (f'<p>By eigenMT p at K = {K_SHOW} the smallest overlap is {pairs[lo]:.2f} ({LABEL[lo[0]]} and {LABEL[lo[1]]}), the '
            f'largest {pairs[hi]:.2f} ({LABEL[hi[0]]} and {LABEL[hi[1]]}); gibbs and tensorQTL share '
            f'{blk["overlap"][K_SHOW]["gibbs"][TQ]:.2f}' +
            (f', TReCASE and tensorQTL {blk["overlap"][K_SHOW][TRECASE[0]][TQ]:.2f}, TReCASE and gibbs '
             f'{blk["overlap"][K_SHOW][TRECASE[0]]["gibbs"]:.2f}' if TRECASE[0] in ra else '') + '.</p>')


def DECOMPOSE_TEXT(S, which, K):
    D = S['sets'][which]['rankings']['eigenmt']['decompose']
    f3 = lambda x: '-' if x is None else f'{x:.3f}'   # noqa: E731
    return f'<p>At K = {K:,}: ' + '; '.join(
        f'{LABEL[a]} shares {d["both"]} genes with tensorQTL (same lead in {f3(d["same_lead"])} of them; replicated '
        f'{f3(d["both_share_arm"])} with its own leads against {f3(d["both_share_ref"])} with tensorQTL\'s), and its '
        f'{d["arm_only"]} other genes replicate at {f3(d["arm_only_share"])} against {f3(d["ref_only_share"])} for '
        f'tensorQTL\'s other genes' for a, per in D.items() for k, d in per.items() if k == K and a in HAPMIX + TRECASE) + '.</p>'


def channel_table(S, which, key):
    rows = []
    for a, per in S['channel'][which].items():
        for K in KS:
            for c, d in per[key][K].items():
                rows.append([LABEL[a], K, c, d['K'], f'{d["replicated"]} / {d["K"]}', ci(d, 'share')])
    return table(['arm', 'K', 'lead class', 'genes', 'replicated', 'replication share [95% interval]'], rows)


def CHANNEL_TEXT(S, which, arms):
    parts = []
    for a in arms:
        v = S['channel'][which][a]['eigenmt'][K_SHOW]
        parts.append(f'{LABEL[a]}: ' + ', '.join(f'{c} (n = {d["K"]}) {d["share"]:.3f} [{d["share_lo"]:.3f}, {d["share_hi"]:.3f}]'
                                                 for c, d in v.items()))
    return f'<p>At K = {K_SHOW}, replication share by lead class: ' + '; '.join(parts) + '.</p>'


def expected_table(S):
    """The channel split's control: each class's replicated count against the count tensorQTL's discovery p predicts."""
    rows = []
    for which in ('all', 'subset'):
        name = f'{which}, {S["subset"]["genes_all"] if which == "all" else S["subset"]["n_genes"]:,} genes'
        r = S['total_curve_check'][which]
        rows.append([name, f'{LABEL[TQ]} (its own leads)', 'every lead', STRATA[0], r['K'], r['replicated'],
                     f'{r["expected"]:.1f}', iv(r, 'diff')])
        for a, per in S['channel'][which].items():
            rows += [[name, LABEL[a], c, s, x['K'], x['replicated'], f'{x["expected"]:.1f}', iv(x, 'diff')]
                     for c, st in per['expected'].items() for s, x in st.items()]
    return table(['gene set', 'arm', 'lead class', 'lead variant', f'leads in the top {K_SHOW}', 'replicated',
                  'expected from tensorQTL\'s discovery p', 'observed minus expected share [95% interval]'], rows)


def CONTROL_TEXT(S):
    """The channel split's control, read; every direction it states is checked by claim()."""
    N, t = S['subset']['genes_all'], TRECASE[0]
    at, el = STRATA
    ex = lambda w, a, c, s: S['channel'][w][a]['expected'][c].get(s)   # noqa: E731
    for w in ('all', 'subset'):
        claim(all(ex(w, a, 'allelic-led', at) and ex(w, a, 'allelic-led', at)['diff_hi'] > 0 for a in HAPMIX),
              f'no hapmixQTL allelic-led stratum at tensorQTL\'s lead below expectation ({w})')
        claim(all(ex(w, a, 'total-led', at)['diff_lo'] < 0 < ex(w, a, 'total-led', at)['diff_hi'] for a in HAPMIX),
              f'hapmixQTL total-led leads at tensorQTL\'s lead at expectation ({w})')
        claim(all(ex(w, a, 'allelic-led', el)['diff_lo'] > 0 for a in HAPMIX), f'hapmixQTL allelic-led leads elsewhere above ({w})')
        claim(all(ex(w, a, 'allelic-led', el)['K'] > ex(w, a, 'allelic-led', at)['K'] for a in HAPMIX),
              f'most hapmixQTL allelic-led leads are away from tensorQTL\'s lead ({w})')
    claim(ex('subset', t, 'total-led', el)['diff_lo'] > 0, 'TReCASE total-led leads away from tensorQTL\'s lead above expectation')
    rng = lambda w, c, s, k: (f'{min(ex(w, a, c, s)[k] for a in HAPMIX):{"+.3f" if k == "diff" else "d"}} to '   # noqa: E731
                              f'{max(ex(w, a, c, s)[k] for a in HAPMIX):{"+.3f" if k == "diff" else "d"}}')
    row = lambda x: f'{x["replicated"]} of {x["K"]} against {x["expected"]:.1f}, {iv(x, "diff")}'   # noqa: E731
    tl_clear = ', '.join(LABEL[a] for a in HAPMIX if ex('all', a, 'total-led', el)['diff_lo'] > 0) or 'no weighting'
    claim(tl_clear != ', '.join(LABEL[a] for a in HAPMIX), 'hapmixQTL total-led leads away from tensorQTL\'s lead not all above')
    full = [f'{a}\'s {x["K"]} on {"all genes" if w == "all" else "the subset"}' for w in ('all', 'subset') for a in HAPMIX
            for x in [ex(w, a, 'allelic-led', at)] if x['replicated'] == x['K']]
    ck = S['total_curve_check']
    return (
        f'<p><b>Control: do allelic-led leads replicate as their total-expression evidence predicts?</b> An allelic-led lead '
        'may replicate less only because the total-expression evidence at its variant is weaker in discovery. For each top-'
        f'{K_SHOW} lead (eigenMT ranking), tensorQTL\'s discovery p at that lead variant is turned into an expected '
        'replication probability by an isotonic regression (the best-fitting non-decreasing step function) of tensorQTL\'s '
        f'own leads\' replication on -log10 of their discovery p, fitted over all {N:,} genes; a stratum\'s expected count is '
        'the sum over its leads, and the interval resamples the stratum\'s genes, each gene\'s replication and expectation '
        f'together. The curve returns tensorQTL\'s own top {K_SHOW} on all genes ({ck["all"]["replicated"]} replicated '
        f'against {ck["all"]["expected"]:.1f} expected; it was fitted to them) and puts its top {K_SHOW} of the subset at '
        f'{ck["subset"]["replicated"]} against {ck["subset"]["expected"]:.1f} ({iv(ck["subset"], "diff")}), the reference '
        'the subset rows are read against.</p>'
        '<p>The curve\'s fitting points are tensorQTL\'s leads, where tensorQTL\'s p is the smallest of its gene. At a variant '
        'tensorQTL did not choose, a p of the same size is less selected, so the expectation computed from it runs low, by an '
        'amount this page does not measure. Each lead class is therefore split by whether the arm\'s lead is tensorQTL\'s own '
        'lead for the gene (the table\'s lead-variant column); only the stratum at tensorQTL\'s lead is read against the '
        'curve without that bias.</p>'
        '<p><b>At tensorQTL\'s lead</b>, the four hapmixQTL weightings\' total-led leads replicate as predicted (observed minus '
        f'expected {rng("all", "total-led", at, "diff")} on all genes and {rng("subset", "total-led", at, "diff")} on the '
        'subset, every interval spanning zero), which is the stratum\'s check. Their allelic-led leads there are few, '
        f'{rng("all", "allelic-led", at, "K")} per weighting on all genes and {rng("subset", "allelic-led", at, "K")} on the '
        'subset: ' + '; '.join(f'{LABEL[a]} {row(ex("all", a, "allelic-led", at))}' for a in HAPMIX) + ' on all genes, and '
        f'{rng("subset", "allelic-led", at, "diff")} on the subset. No interval lies below zero. <b>Away from tensorQTL\'s '
        'lead</b>, the allelic-led leads are above the prediction in every weighting on both gene sets (' +
        '; '.join(f'{LABEL[a]} {row(ex("all", a, "allelic-led", el))}' for a in HAPMIX) + f' on all genes; '
        f'{rng("subset", "allelic-led", el, "diff")} on the subset, every interval above zero). Total-led leads can be above '
        'it there too: TReCASE\'s total-led leads away from tensorQTL\'s lead on the subset, chosen on native counts, '
        f'replicate {row(ex("subset", t, "total-led", el))}; the hapmixQTL weightings\' total-led leads away from it, chosen '
        f'on the same Salmon totals as tensorQTL\'s, are at {rng("all", "total-led", el, "diff")} on all genes, with an '
        f'interval clear of zero for {tl_clear} only.</p>'
        '<p><b>What the control can and cannot show.</b> It can show whether allelic-led leads at the variants tensorQTL '
        'itself chose replicate less than their total-expression evidence predicts, the objection\'s prediction; no '
        'weighting\'s interval there lies below zero. With so few leads it cannot show that they replicate as predicted '
        'rather than more or less. It says nothing about the allelic-led leads at other variants, most of them, because the '
        'expectation there runs low for every lead by an unmeasured amount, so their excess cannot be credited to the '
        'allelic channel; nor can it exclude an excess specific to them, since the bias may differ between classes. Its '
        'intervals resample the stratum\'s leads and treat the fitted curve as exact, so they understate the uncertainty of '
        'the comparison' + (f'; where every lead of a stratum replicated ({", ".join(full)}) the interval reflects only the '
                            'spread of the expectations' if full else '') + '.</p>')


def TRECASE_TEXT(S):
    TR, sub = S['trecase'], S['subset']
    ps = sub['process_seconds_per_gene']
    v = TR['per_input']['native']
    fm = TR['failure_messages']['native']
    msgs = pd.Series(list(fm.values()), dtype=str).value_counts().to_dict()
    claim(all('tiny variances' in m for m in msgs), 'every Rscript failure is asSeq refusing a near-constant total count')
    fails = '; '.join(f'&ldquo;{html.escape(m)}&rdquo; ({c} gene{"s" * (c != 1)})' for m, c in msgs.items()) or 'none'
    miss = TR['missing'][TRECASE[0]]
    bf = TR['baseline_failed']
    ranked_n = sub['n_genes'] - S['sets']['subset']['rankings']['eigenmt']['unranked'][TRECASE[0]]
    claim(v['genes'] - len(v['genes_failed']) - len(bf['trec_genes']) == ranked_n,
          'TReCASE ranked genes = genes less Rscript failures less total-count baseline failures')
    return (
        f'<p>The Salmon-input TReCASE was dropped from the referee by user decision because it costs about five times the '
        f'native run ({ps["salmon"]:.0f} against {ps["native"]:.0f} process-seconds per gene in the timing block); its '
        'finished genes are on disk and are not scored. The plasmode benchmark runs TReCASE on both inputs (its trecase arm '
        'on Salmon-derived input, its trecase_native arm on the native counts used here).</p>' +
        table(['input', 'genes', 'failed (Rscript error)', 'baseline model not used: allelic only / total-count only / both',
               'tests', 'final p joint / TReC / missing', 'allelic records with &ge; 5 reads', 'asSeq seconds per gene, median'],
              [['native counts', v['genes'], len(v['genes_failed']),
                f'{bf["allelic_only"]} / {bf["trec_only"]} / {bf["both"]}', f'{v["rows"]:,}',
                f'{v["final_joint"]:,} / {v["final_trec"]:,} / {v["final_na"]:,}', f'{v["as_records_admitted"]:,}',
                f'{np.median(v["trecase_seconds"]):.0f}']]) +
        f'<p>asSeq reports the joint p where its test of equal allelic and total effects (trans p) is at least 0.05 and the '
        f'total-count (TReC) p otherwise. Offset: the edgeR effective library size of the featureCounts totals, built as the '
        f'Salmon one was; per donor it is {sub["native_over_salmon_eff_lib"][0]:.3f} / {sub["native_over_salmon_eff_lib"][1]:.3f} '
        f'/ {sub["native_over_salmon_eff_lib"][2]:.3f} (min / median / max) times the Salmon one. X is the 17 covariates of '
        f'every other arm (their expression principal components come from Salmon log2 CPM). Failed genes: {len(fm)}, asSeq '
        f'refusing a gene whose total count barely varies across donors; its message: {fails}. A failed gene has no TReCASE '
        'p and is not ranked. asSeq first fits each '
        'gene\'s baseline models, the covariates without the tested variant, and records which it could not use as '
        '(1 - useTReC) + 2 (1 - useASE): the allelic baseline (not used where fewer donors than its minimum carry enough '
        f'allelic reads, or where its fit fails) alone in {bf["allelic_only"]} genes, the total-count baseline alone in '
        f'{bf["trec_only"]} and both in {bf["both"]}. The {len(bf["trec_genes"])} genes whose total-count baseline failed '
        f'({", ".join(bf["trec_genes"])}) have rows but no final p, which is why TReCASE ranks {ranked_n:,} genes and not '
        f'{v["genes"] - len(v["genes_failed"]):,}. Tests without a final p: '
        f'{miss["final_na"]:,} of {miss["tests"]:,} in {miss["genes_affected"]} of {miss["genes"]} genes, of which '
        f'{miss["joint_chosen_joint_failed"]:,} are tests where asSeq chose the joint test and its fit failed; asSeq does not '
        'fall back to the TReC p there, and the lead is chosen among the gene\'s other tests. The run was restarted on the '
        f'native input alone once the Salmon input was dropped; the restarted call ran {v["genes"] - v["genes_resumed"]} genes '
        f'({v["genes_resumed"]} had finished before) in {TR["wall_hours"]:.2f} h at {TR["jobs"]} R processes, largest genes '
        'first.</p>')


def SUMMARY(S):
    A = S['sets']['all']['rankings']['eigenmt']
    B = S['sets']['subset']['rankings']['eigenmt']
    n, N, K, KD = S['subset']['n_genes'], S['subset']['genes_all'], K_SHOW, deep_k(S)
    sh = sorted(((A['at_K'][a][K]['share'], a) for a in A['arms']), reverse=True)
    t = TRECASE[0]
    bh = A['bh']
    tp = {b: B['paired_other'][f'{t} minus {b}'] for b in ('split', 'gibbs')}
    def order(s, name):   # the TReCASE-minus-weighting differences in one statistic, and whether the referee orders them
        clear = [(b, k) for b, v in tp.items() for k in B['ks'] if v[k][f'{s}_lo'] > 0 or v[k][f'{s}_hi'] < 0]
        return (f'{name}, ' + '; '.join(f'minus {b} ' + ', '.join(f'K = {k} {iv(v[k], s)}' for k in B['ks']) for b, v in tp.items()) +
                (f'. Every {name} interval includes zero, so by {name} the referee cannot order native TReCASE and split or gibbs.'
                 if not clear else f'. The {name} intervals include zero except ' +
                 ', '.join(f'minus {b} at K = {k}' for b, k in clear) + f'; elsewhere the referee cannot order them by {name}.'))
    order_ = order('share', 'replication share') + ' In ' + order('pi1', 'pi1')
    return (
        f'<p><b>Result, seven arms on all {N:,} genes.</b> Ranked by eigenMT p, the share of the top {K} genes whose lead '
        f'replicates in the held-out donors is ' + ', '.join(f'{v:.3f} {LABEL[a]}' for v, a in sh) + '. Each arm minus '
        f'tensorQTL at K = {K} and, deeper, at K = {KD:,} (the top {KD / N:.0%} of genes), with the paired 95% interval: ' +
        '; '.join(f'{LABEL[a]} {iv(A["paired"][a][K])} and {iv(A["paired"][a][KD])}' for a in A['arms'] if a != REF) +
        f'. At Benjamini-Hochberg 5%, gibbs calls {bh["gibbs"]["K"]:,} genes of which {bh["gibbs"]["replicated"]:,} replicate; '
        f'tensorQTL\'s top {bh["gibbs"]["K"]:,} give {bh["gibbs"]["ref_replicated_at_this_K"]:,}.</p>'
        f'<p><b>TReCASE, on the {n:,}-gene seeded subset with every arm restricted to it.</b> TReCASE on native counts: '
        f'replication share {B["at_K"][t][K]["share"]:.3f} at K = {K} against tensorQTL\'s {B["at_K"][REF][K]["share"]:.3f} '
        f'and gibbs\'s {B["at_K"]["gibbs"][K]["share"]:.3f}; minus tensorQTL ' +
        ', '.join(f'K = {k} {iv(B["paired"][t][k])}' for k in B['ks']) + '. TReCASE against two hapmixQTL weightings on the '
        f'same genes, paired as above: {order_}</p>' + MEANING_TEXT(S, 'summary'))


def CRITIQUE(S):
    A = S['sets']['all']['rankings']['eigenmt']
    K, KD, P = K_SHOW, deep_k(S), A['paired']
    rs, ch, dec = S['referee_source'], S['channel']['all'], A['decompose']
    below = [(a, k) for a in HAPMIX for k in A['ks'] if P[a][k]['share_hi'] < 0]
    above = [(a, k) for a in HAPMIX for k in A['ks'] if P[a][k]['share_lo'] > 0]
    quiet = [k for k in A['ks'] if k not in {x for _, x in below + above}]
    claim(all(P[a][K]['share'] < 0 for a in HAPMIX), 'every hapmixQTL weighting below tensorQTL at K_SHOW')
    claim(below and above and max(k for _, k in below) < min(k for _, k in above), 'deficits shallow, surpluses deep')
    at_ks = lambda xs: '; '.join(   # noqa: E731
        f'{LABEL[a]} at K = ' + ', '.join(f'{k:,} {iv(P[a][k])}' for b, k in xs if b == a) for a in dict.fromkeys(a for a, _ in xs))
    cls = lambda a, c: ch[a]['eigenmt'][K].get(c)   # noqa: E731
    split_rows = '; '.join(
        f'{LABEL[a]} allelic-led (n = {cls(a, "allelic-led")["K"]}) {iv(cls(a, "allelic-led"), f=".3f")}, total-led '
        f'(n = {cls(a, "total-led")["K"]}) {iv(cls(a, "total-led"), f=".3f")}' for a in HAPMIX
        if cls(a, 'allelic-led') and cls(a, 'total-led'))
    dec_rows = lambda k: '; '.join(   # noqa: E731
        f'{LABEL[a]}: {dec[a][k]["both"]} shared genes, same lead in {dec[a][k]["same_lead"]:.2f}, replicated '
        f'{dec[a][k]["both_share_arm"]:.3f} with its leads against {dec[a][k]["both_share_ref"]:.3f} with tensorQTL\'s; its '
        f'{dec[a][k]["arm_only"]} other genes {dec[a][k]["arm_only_share"]:.3f} against {dec[a][k]["ref_only_share"]:.3f}'
        for a in HAPMIX)
    bh = A['bh']
    bh_rows = '; '.join(f'{LABEL[a]} {bh[a]["K"]:,} genes, {bh[a]["replicated"]:,} replicated, tensorQTL\'s top {bh[a]["K"]:,} '
                        f'{bh[a]["ref_replicated_at_this_K"]:,}' for a in HAPMIX + MIX if bh[a]['K'])
    return (
        '<h2>The critique, and what it changed</h2>'
        '<p><b>The strongest objection</b> is that the referee is itself a total-expression scan with tensorQTL\'s model. It '
        'credits only the part of an effect that moves total expression, and it scores tensorQTL\'s leads with the '
        'statistic that chose them, while an arm that ranks on allelic evidence is judged on evidence it did not use. '
        'Three measurements bear on it, all on the seven arms over all genes, eigenMT ranking.</p>'
        f'<p><b>Quantification.</b> The referee phenotype is not built from any arm\'s counts; per gene it correlates with '
        f'the Salmon totals at a median Spearman {rs["salmon"][1]:.3f} and with the featureCounts totals at '
        f'{rs["native"][1]:.3f} ({rs["donors"]} shared donors, {rs["genes"]:,} genes). Every arm except TReCASE uses the '
        'Salmon totals, so the referee gives TReCASE no quantification advantage.</p>'
        f'<p><b>Allelic-led leads.</b> At K = {K}: {split_rows}. ' + CHANNEL_VERDICT(S) + '</p>'
        f'<p><b>Gene choice against lead choice.</b> At K = {K}: {dec_rows(K)}. At K = {KD:,}: {dec_rows(KD)}. Where the shared '
        'genes replicate at the same rate with either arm\'s lead, a different lead variant costs nothing in this referee, '
        'and any difference between arms lies in the genes only one of them ranks this high.</p>'
        f'<p><b>What it changed.</b> The objection predicts a deficit for the arms that use the allelic channel. At K = {K} '
        'all four hapmixQTL weightings are below tensorQTL in replication share: ' +
        '; '.join(f'{LABEL[a]} {iv(P[a][K])}' for a in HAPMIX) + f'. Differences from tensorQTL whose interval excludes zero: '
        f'below, {at_ks(below)}; above, {at_ks(above)}. Every deficit that clears zero lies at K &le; '
        f'{max(k for _, k in below):,} and every surplus at K &ge; {min(k for _, k in above):,}; at K = '
        f'{", ".join(f"{k:,}" for k in quiet)} no weighting\'s difference clears zero.' + MEANING_TEXT(S, 'critique') + '</p>'
        f'<p><b>Anticonservative p and the BH sets.</b> At Benjamini-Hochberg 5%: {bh_rows}. An arm that calls more genes '
        'than tensorQTL has to be compared with tensorQTL\'s own list of the same length; a larger BH set is a gain only '
        'where it replicates more genes than that list.</p>'
        '<p><b>Intervals.</b> The per-arm interval treats the K genes as given; comparisons between arms rest on the paired '
        'whole-set resampling, whose interval includes which genes enter the top K.</p>')


def MEANING(S):
    A = S['sets']['all']['rankings']['eigenmt']
    B = S['sets']['subset']['rankings']['eigenmt']
    P = S['plasmode']
    KD = deep_k(S)
    rows = []
    pw = lambda a, b: P['ranking'][f'beta{b}'][a]['fdp_matched']['all']['power']   # noqa: E731
    for a in ARMS:
        ref = [iv(A['paired'][a][K]) for K in (K_SHOW, KD)] if a in A['paired'] else ['-', '-']
        rows.append([LABEL[a]] + [f'{pw(a, b):.3f} ({pw(a, b) - pw(TQ, b):+.3f}' + (
            f'; Salmon input {pw("trecase", b):.3f}, {pw("trecase", b) - pw(TQ, b):+.3f})' if a in TRECASE else ')')
            for b in PLASMODE_BETAS] + ref + [iv(B['paired'][a][K_SHOW]) if a in B['paired'] else '-'])
    return (
        '<h2>What it means for the wider claims</h2>'
        '<p>The plasmode benchmark (deep gene set, plasmode_meier_20260927) ranks genes by lead p within each planted '
        'dataset and reports power at a 5% realized false-discovery proportion; the referee ranks the observed genes and '
        'counts replications. Both ask whose ranking puts real effects first, with different truths: planted effects of '
        'known size on 100 high-coverage genes there, held-out total expression here. TReCASE\'s plasmode entry is its '
        'trecase_native arm, on the same native counts as the referee\'s TReCASE run (thinned there to plant the effects, '
        'on the same datasets as every other arm), with its Salmon-input trecase arm in parentheses.</p>' +
        table(['arm'] + [f'plasmode power, |beta| = {b} (minus tensorQTL)' for b in PLASMODE_BETAS] +
              [f'referee, all genes, share minus tensorQTL, K = {K_SHOW}', f'same, K = {KD:,}',
               f'referee, {S["subset"]["n_genes"]:,}-gene subset, K = {K_SHOW}'], rows) + MEANING_TEXT(S, 'meaning'))


def claim(ok, what):
    """A sentence of the prose that states a direction must hold in the numbers it quotes."""
    if not ok:
        raise SystemExit(f'page prose no longer matches the numbers: {what}')


def CHANNEL_VERDICT(S):
    ch = S['channel']['all']
    sep = [a for a in HAPMIX if ch[a]['eigenmt'][K_SHOW].get('allelic-led') and
           ch[a]['eigenmt'][K_SHOW]['allelic-led']['share_hi'] < ch[a]['eigenmt'][K_SHOW]['total-led']['share_lo']]
    at, el = STRATA
    es = lambda a, c: (sum(x['expected'] for x in ch[a]['expected'][c].values()) /   # noqa: E731
                       sum(x['K'] for x in ch[a]['expected'][c].values()))
    x = lambda a, s: ch[a]['expected']['allelic-led'][s]   # noqa: E731
    claim(len(sep) >= 2, 'allelic-led below total-led, intervals apart, in at least two weightings')
    claim(all(es(a, 'allelic-led') < es(a, 'total-led') for a in sep), 'those allelic-led leads carry weaker total evidence')
    claim(all(x(a, at)['diff_hi'] > 0 for a in sep), 'none of them below expectation at tensorQTL\'s lead')
    claim(all(x(a, el)['diff_lo'] > 0 for a in sep), 'the rest above expectation')
    return (f'For {", ".join(LABEL[a] for a in sep)} the two intervals do not overlap: the referee credits a lead chosen on '
            'allelic evidence less often than one chosen on total evidence, as the objection predicts. The total-expression '
            'evidence at those leads is weaker: the curve of the control in the results expects ' + '; '.join(
                f'{es(a, "allelic-led"):.3f} of {a}\'s allelic-led leads to replicate against {es(a, "total-led"):.3f} '
                f'of its total-led' for a in sep) + '. Split by whether the lead is tensorQTL\'s own, the allelic-led leads at '
            'tensorQTL\'s lead, where the curve is unbiased, show no deficit against that evidence (' + '; '.join(
                f'{a} {x(a, at)["replicated"]} of {x(a, at)["K"]} against {x(a, at)["expected"]:.1f}, '
                f'{iv(x(a, at), "diff")}' for a in sep) + '), but they are too few to show that they replicate as predicted. '
            'The rest (' + ', '.join(f'{x(a, el)["K"]} for {a}' for a in sep) + ') are above the prediction where it runs low '
            'for every lead, so that excess cannot be credited to the allelic channel. What those leads are is the three-way '
            'question in the limits below.')


def MEANING_TEXT(S, where):
    """Interpretation of the final numbers; every direction it states is checked by claim()."""
    A, B = S['sets']['all']['rankings']['eigenmt'], S['sets']['subset']['rankings']['eigenmt']
    N, n, KD = S['subset']['genes_all'], S['subset']['n_genes'], deep_k(S)
    K2, t = A['ks'][-1], TRECASE[0]
    pd_ = lambda a, K, blk=A: blk['paired'][a][K]   # noqa: E731
    if where == 'summary':
        below = [a for a in A['arms'] if a != REF and pd_(a, K_SHOW)['share_hi'] < 0]
        deep = [pd_(a, K)['share'] for a in HAPMIX for K in (KD, K2)]
        claim(all(pd_(a, K)['share_lo'] > 0 for a in HAPMIX for K in (KD, K2)), 'hapmixQTL weightings above tensorQTL deep')
        first_mix = min(K for K in A['ks'] if pd_('mixqtl', K)['share_hi'] < 0)
        claim(all(pd_('mixqtl', K)['share_hi'] < 0 for K in A['ks'] if K >= first_mix), 'mixQTL published below from first_mix on')
        claim(all(pd_('mixqtl_permissive', K)['share_lo'] < 0 < pd_('mixqtl_permissive', K)['share_hi'] for K in (KD, K2)),
              'mixQTL permissive level with tensorQTL deep')
        sub = {a: B['at_K'][a][K_SHOW]['share'] for a in B['arms']}
        claim(pd_(t, K_SHOW, B)['share_lo'] > 0, 'TReCASE above tensorQTL at K_SHOW on the subset')
        return (
            f'<p><b>Reading.</b> The arms separate by depth. Among the top {K_SHOW} of all genes, where the arms largely agree on '
            f'the genes (gibbs and tensorQTL share {A["overlap"][K_SHOW]["gibbs"][TQ]:.2f} of them), tensorQTL\'s leads '
            f'replicate as well as or better than every other arm\'s; the arms below it with an interval clear of zero are '
            f'{", ".join(LABEL[a] for a in below)}. Deeper, at K = {KD:,} and {K2:,} (the top {KD / N:.0%} and {K2 / N:.0%} of '
            f'genes), all four hapmixQTL weightings replicate more than tensorQTL, by {min(deep):+.3f} to {max(deep):+.3f}, '
            f'every interval above zero. mixQTL at the published cutoffs is below tensorQTL from K = {first_mix} on; at the '
            f'permissive cutoffs it is level with it deep in the list. On the {n:,}-gene subset, at K = {K_SHOW} (its top '
            f'{K_SHOW / n:.0%}), TReCASE on native counts replicates at {sub[t]:.3f}, the hapmixQTL weightings at '
            f'{min(sub[a] for a in HAPMIX):.3f} to {max(sub[a] for a in HAPMIX):.3f} and tensorQTL at {sub[REF]:.3f}.</p>')
    if where == 'critique':
        dK, dD = ({a: A['decompose'][a][K] for a in HAPMIX} for K in (K_SHOW, KD))
        gap = {a: abs(round((d['both_share_arm'] - d['both_share_ref']) * d['both'])) for a, d in dK.items()}   # in genes
        claim(all(d['arm_only_share'] < d['ref_only_share'] for d in dK.values()) and
              all(d['arm_only_share'] > d['ref_only_share'] for d in dD.values()), 'gene-choice gap reverses with depth')
        claim(max(gap.values()) <= 1, 'shared genes replicate alike with either lead')
        tc = S['channel']['subset'][t]['eigenmt'][K_SHOW]
        claim(tc['allelic-led']['share_hi'] < tc['total-led']['share_lo'], 'TReCASE allelic-led below total-led, intervals apart')
        r3 = lambda d, k: f'{min(x[k] for x in d.values()):.3f} to {max(x[k] for x in d.values()):.3f}'   # noqa: E731
        return (
            f' Where that difference lies was measured. At K = {K_SHOW}, on the genes both an arm and tensorQTL rank that '
            f'high ({min(d["both"] for d in dK.values())} to {max(d["both"] for d in dK.values())} genes) the two leads '
            f'replicate alike: the replicated counts differ by at most {max(gap.values())} gene, so a different lead variant '
            f'costs nothing there. The deficit is in the genes only the arm ranks that high, which replicate at '
            f'{r3(dK, "arm_only_share")} against {r3(dK, "ref_only_share")} for the genes only tensorQTL ranks that high; at '
            f'K = {KD:,} the arm-only genes are the better ones ({r3(dD, "arm_only_share")} against '
            f'{r3(dD, "ref_only_share")}). The allelic-led gap is not specific to hapmixQTL or to Salmon: TReCASE on native '
            f'counts shows it on the subset at K = {K_SHOW} (allelic-led {iv(tc["allelic-led"], f=".3f")}, n = '
            f'{tc["allelic-led"]["K"]}, against total-led {iv(tc["total-led"], f=".3f")}, n = {tc["total-led"]["K"]}). '
            'Addressing the objection therefore leaves the depth pattern as measured: the shallow deficit is in genes the '
            'allelic arms alone rank that high, and what those genes carry that this referee does not credit is the '
            'three-way question in the limits below.')
    P = S['plasmode']['ranking']
    pw = lambda a, b='0.4': P[f'beta{b}'][a]['fdp_matched']['all']['power'] - P[f'beta{b}'][TQ]['fdp_matched']['all']['power']   # noqa: E731
    bh = A['bh']
    agree = [a for a in HAPMIX if pw(a) > 0 and pd_(a, KD)['share_lo'] > 0]
    differ = [a for a in HAPMIX if pw(a) < 0 and pd_(a, KD)['share_lo'] > 0]
    claim(agree and differ, 'plasmode agreement and disagreement both present')
    claim(pw(t) > 0 and pw(t, '0.8') > 0 and pd_(t, K_SHOW, B)['share_lo'] > 0, 'native TReCASE above tensorQTL in both')
    o = A['paired_other']
    split = o['split minus gibbs']
    claim(all(split[K]['share'] >= 0 for K in A['ks']) and all(split[K]['share_lo'] > 0 for K in A['ks'] if K >= 783),
          'split at or above gibbs at every K, clear of zero from 783')
    wk = {a: dict(ahead=[K for K in A['ks'] if o[f'{a} minus gibbs'][K]['share'] > 0],
                  above=[K for K in A['ks'] if o[f'{a} minus gibbs'][K]['share_lo'] > 0],
                  below=[K for K in A['ks'] if o[f'{a} minus gibbs'][K]['share_hi'] < 0]) for a in ('unit', 'plus_one')}
    for a, w in wk.items():
        i = [A['ks'].index(K) for K in w['ahead']]
        claim(i and i == list(range(i[0], i[-1] + 1)), f'{a} ahead of gibbs over one run of K')
        claim(w['above'] and set(w['above']) <= set(w['ahead']) and w['below'] == [K2], f'{a} clear above gibbs somewhere, '
              'below it at the deepest K and nowhere else')
    ks = lambda xs: ' and '.join(f'{K:,}' for K in xs) if len(xs) < 3 else ', '.join(f'{K:,}' for K in xs[:-1]) + f' and {xs[-1]:,}'   # noqa: E731
    weigh = '; '.join(
        f'{a}\'s point estimate is above gibbs\'s from K = {w["ahead"][0]:,} to {w["ahead"][-1]:,}, its interval clear of zero '
        f'at K = {ks(w["above"])}, and below it at K = {K2:,} ({iv(o[f"{a} minus gibbs"][K2])})' for a, w in wk.items())
    gain = [a for a in HAPMIX + MIX if bh[a]['K'] > bh[REF]['K'] and bh[a]['replicated'] > bh[a]['ref_replicated_at_this_K']]
    loss = [a for a in HAPMIX + MIX if bh[a]['K'] > bh[REF]['K'] and bh[a]['replicated'] < bh[a]['ref_replicated_at_this_K']]
    ties = {a: S['sets']['all']['rankings']['perm']['at_K'][a][KS[-1]]['tied_at_K'] for a in MIX}
    tb = B['bh'][t]
    return (
        f'<p><b>Against the plasmode.</b> At |beta| = 0.4 the plasmode put {", ".join(LABEL[a] for a in agree)} above '
        f'tensorQTL in power, and the referee agrees for them deep in the list. It put {", ".join(LABEL[a] for a in differ)} '
        f'below tensorQTL ({", ".join(f"{pw(a):+.3f}" for a in differ)}), where the referee has it above at K = {KD:,}. The '
        'two benchmarks do not cut at the same depth: the plasmode reads power where 5% of the called genes are null, the '
        'referee a fixed number of genes. TReCASE is compared on native counts in both: the plasmode put its trecase_native '
        f'arm above tensorQTL in power ({pw(t):+.3f} at |beta| = 0.4, {pw(t, "0.8"):+.3f} at 0.8; on Salmon input '
        f'{pw("trecase"):+.3f} and {pw("trecase", "0.8"):+.3f}), and the referee has it above tensorQTL on the subset at '
        f'K = {K_SHOW} ({iv(pd_(t, K_SHOW, B))}).</p>'
        f'<p><b>The weighting decision.</b> Against gibbs, the shipped weighting, split is at or above it at every K on all '
        f'genes and clear of zero from K = 783 on ({split[1565]["share"]:+.3f} [{split[1565]["share_lo"]:+.3f}, '
        f'{split[1565]["share_hi"]:+.3f}] at 1,565); {weigh}. This page reports these differences; it does not settle the '
        'decision, which also rests on calibration.</p>'
        f'<p><b>Benjamini-Hochberg sets at matched length.</b> An arm calling more genes than tensorQTL gains where its list '
        f'replicates more genes than tensorQTL\'s list of the same length: ' +
        '; '.join(f'{LABEL[a]} {bh[a]["K"]:,} genes, {bh[a]["replicated"]:,} replicated against {bh[a]["ref_replicated_at_this_K"]:,}'
                  for a in gain + loss) +
        f'. That is a gain for {", ".join(LABEL[a] for a in gain)} and a loss for {", ".join(LABEL[a] for a in loss) or "none"}. '
        f'On the subset TReCASE\'s eigenMT set holds {tb["K"]} genes at a replication share of {tb["share"]:.3f}, '
        f'{tb["replicated"]} replicated against {tb["ref_replicated_at_this_K"]} for tensorQTL\'s top {tb["K"]}.</p>'
        f'<p><b>mixQTL\'s permutation ranking.</b> On all genes {ties["mixqtl"]} (published cutoffs) and '
        f'{ties["mixqtl_permissive"]} (permissive) genes share the smallest empirical permutation p, so for every K up to '
        f'{KS[-1]} that ranking is decided by the tie-break (lead p); eigenMT is the ranking that separates those genes.</p>')


def LIMITS(S):
    F = S['facts']
    rs = json.loads((C.D / 'native_counts_stranded_20260928' / 'facts.json').read_text())['reference_share']   # native_counts.reference_share
    return (
        '<h2>What this cannot establish</h2><ul>'
        '<li>Purely allelic effects. The referee measures total expression only, so an effect that changes the allelic ratio '
        'without moving the total (buffered, or compensated in trans) cannot replicate here, and an arm that finds such '
        'effects through its allelic channel is not credited for them. An allelic signal that is an artefact of read '
        'assignment (mapping or quantification bias at heterozygous sites) is, correctly, not credited either. Nor is a '
        'real cis effect on total expression that the allelic channel detected and the held-out total-expression test, at '
        'this donor count, does not reach p &lt; 0.05. The referee cannot tell these three apart.</li>'
        '<li>Calibration. Matched K removes the reward for an anticonservative gene-level p; it says nothing about whether '
        'any arm\'s p is correct. The Benjamini-Hochberg rows are the only place a p is taken at face value.</li>'
        f'<li>Small effects. With {F["cohort"]["heldout"]} donors and {F["heldout"]["residual_df"]} residual degrees of '
        'freedom a real but small effect often has held-out p above 0.05, so the replication share is a lower bound on the '
        'share of real effects and falls with effect size down the list; pi1 needs no threshold but has its own variance, '
        'visible in its intervals.</li>'
        f'<li>TReCASE on the other genes. TReCASE ran on a seeded random {S["subset"]["n_genes"]:,} of the referee genes; its '
        'ranking of the rest is unmeasured. A top K of the subset reaches deeper into weak signals than the same K of all '
        'genes, so the subset comparison is not on the scale of the all-gene one.</li>'
        '<li>Reference-mapping bias in TReCASE\'s allele counts. The native counts come from phASER run on STAR alignments '
        'to the standard reference, per transcript strand at SNVs in exon stretches that one gene owns on its strand, with '
        'HLA genes and CHM13 short-read-inaccessible regions excluded but no WASP filtering, the one correction for '
        'reference-mapping bias not applied (scripts/phaser_stranded.py). Per '
        f'donor, the median over phASER heterozygous sites with at least {rs["min_reads"]} reads of the reference-allele share '
        f'(refCount / totalCount, 0.5 if reads from the two alleles mapped alike) runs from {rs["median_min"]:.3f} to '
        f'{rs["median_max"]:.3f} over the {rs["donors"]} discovery donors ({rs["sites_per_donor"][1]:,} such sites in the '
        'median donor). Only sites where the donor carries the VCF REF allele enter: phASER\'s variantID is '
        'contig-position-REF-ALT from the VCF record and its refAllele is the donor\'s first carried allele, so a row is kept '
        f'where refAllele equals the REF field of the variantID, which drops {rs["alt_alt_sites_dropped"]} rows with at least '
        f'{rs["min_reads"]} reads, over all donors, at heterozygotes of two ALT alleles (native_counts.reference_share). '
        'In real data, unlike the plasmode '
        'benchmark, whose records are shuffled against the genotypes, reference bias is not randomized, so native '
        'TReCASE\'s allelic channel can carry it. The hapmixQTL arms use Salmon on a personalized diploid transcriptome '
        'instead.</li>'
        '<li>Which variant is causal. The replication is at the discovery lead, and in the same population a lead in linkage '
        'disequilibrium with the causal variant replicates as well as the causal variant does.</li>'
        '<li>Independence of the held-out processing. The held-out phenotype\'s rank-inverse-normal transform, TMM factors and '
        'gene filter, and the genotype store\'s imputation of missing genotypes, were computed over all 225 donors, 90 of '
        'them discovery donors (recorded in referee_replication.py). None of these fits held-out expression to discovery '
        'genotypes, so none can create a replication, but the held-out values are not processed in isolation.</li></ul>')


ARM_ROWS = [   # arm: what it fits, its permutation p (module docstring; referee_replication.py and referee_trecase.py)
    ('gibbs', 'hapmixQTL default mode: allelic log ratio and log2 total, each weighted 1/v (v = the Gibbs variance), '
              'inverse-variance combined', 'pval_beta, 1,000 records-plus-haplotype-swap permutations'),
    ('split', 'hapmixQTL, 1/v weights in the allelic channel, unit weights in the total channel', 'pval_beta, as gibbs'),
    ('unit', 'hapmixQTL, unit weights in both channels', 'pval_beta, as gibbs'),
    ('plus_one', 'hapmixQTL, weights 1/(v + 1) in both channels', 'pval_beta, as gibbs'),
    ('mixqtl', 'mixQTL port (Liang et al. 2021), published count cutoffs, Salmon point estimates', 'pval_perm, 1,000 permutations'),
    ('mixqtl_permissive', 'mixQTL port, the package-default (permissive) cutoffs', 'pval_perm, as mixqtl'),
    (TQ, 'tensorQTL on log2 total expression alone, unweighted, the same 17 covariates', 'pval_beta, 1,000 permutations'),
    ('trecase_native', 'asSeq TReCASE: negative-binomial total counts plus beta-binomial allele counts, joint likelihood, on '
                       'alignment-based integer counts (featureCounts totals, phASER haplotype counts); the TReCASE subset '
                       'only', 'none (eigenMT only)')]


def page_text(S, figs):
    sub, F = S['subset'], S['facts']
    n, N = sub['n_genes'], sub['genes_all']
    ps = sub['process_seconds_per_gene']
    rs = S['referee_source']
    fh = F['heldout']
    A, B = S['sets']['all'], S['sets']['subset']
    KD = deep_k(S)
    extra = [K for K in A['rankings']['eigenmt']['ks'] if K not in KS]
    body = [f'<h1>Held-out replication referee</h1><p class="sub">Which arm\'s top genes replicate in {F["cohort"]["heldout"]} '
            f'BrainVar donors that no arm saw. 2026-09-28. Scripts: scripts/referee_replication.py (inputs, discovery, '
            f'replication scan), scripts/referee_trecase.py (TReCASE), scripts/referee_score.py (this page). Every number: '
            f'{OUT}/score/score.json.</p>', SUMMARY(S)]
    body.append(
        '<h2>Why this was needed</h2><p>The plasmode benchmark plants effects of known size into real records and scores '
        'recovery of them. Its truth is the generator\'s, so it cannot say whether the genes an arm ranks first on the '
        'observed data carry real cis effects. A held-out cohort can: an eQTL that is real in the 92 discovery donors '
        'should show the same direction of effect at the same variant in unrelated donors. The BrainVar eQTL cohort has '
        f'{F["cohort"]["donors_225"]} donors with genotypes and total-expression phenotypes; {F["cohort"]["heldout"]} of '
        f'them are not among the 92 discovery donors (DNA library ids), and none of those {F["cohort"]["heldout"]} matches a '
        f'discovery donor\'s genotypes (highest concordance {F["identity"]["heldout_max"]:.3f} on '
        f'{F["identity"]["variants"]:,} common variants, where a donor matches itself at {F["identity"]["shared_self_min"]:.3f}). '
        'Taking the same number K of top genes from every arm removes any advantage an anticonservative gene-level p would '
        'otherwise give: the question at matched K is only whose top genes are more often real.</p>')
    body.append(
        '<h2>What was run</h2><h3>Discovery, 92 donors, observed records</h3><p>Every arm mapped the observed data '
        '(records in place, nothing thinned), with the settings of the plasmode benchmark (scripts/plasmode/03_run_arms.py '
        'and 05_run_trecase.py).</p>' + table(['arm', 'what it fits', 'permutation p'], [[LABEL[a], d, p] for a, d, p in ARM_ROWS]))
    body.append(
        f'<h3>Genes, and the TReCASE subset</h3><p>The referee genes are the {N:,} genes of the eQTL gene filter with Gibbs '
        f'draws and a phenotype in the 225-donor run, in one seeded random order (SeedSequence(42, spawn_key=(7,))). Tested '
        f'variants: phased biallelic SNPs within 1 Mb of the TSS at minor allele frequency &ge; 0.05 over the 92 donors that '
        f'are also in the 225-donor genotype store ({F["variants"]["mapped_rate"]:.3f} of such SNPs are), median '
        f'{F["genes"]["tested_per_gene"][1]:,} per gene. The seven arms other than TReCASE ran on all of them. TReCASE was '
        f'timed on the first 100 genes of the order at {sub["jobs"]} R processes, on the Salmon input as the plasmode builds '
        f'it and on native counts ({ps["salmon"]:.0f} and {ps["native"]:.0f} process-seconds per gene); both inputs on all '
        f'genes projected {sub["projected_hours_all"]:.1f} h against a 6 h budget, so the longest prefix of whole 100-gene '
        f'units that fitted, the first {n:,} genes of the order (a seeded random subset), became the TReCASE gene set. The '
        'TReCASE comparison restricts every arm to those genes.</p>')
    body.append(
        f'<h3>The referee, {F["cohort"]["heldout"]} held-out donors</h3><p>tensorQTL map_nominal on the held-out donors: '
        f'the 225-donor run\'s own phenotype, its {fh["base"]} base covariates restricted to the held-out donors, and '
        f'{fh["expression_pcs"]} expression principal components recomputed on the held-out donors by the run\'s own code '
        f'(which reproduced the run\'s 225-donor components to {fh["pca_gate_max_abs_diff"]:.1e}); {fh["covariates"]} '
        f'covariates, residual degrees of freedom {fh["residual_df"]}. Every discovery test has a held-out counterpart '
        f'({F["replication"]["pairs"]:,} pairs), the slope per ALT allele under the same allele coding as discovery (store and '
        f'VCF dosages agree at {F["variants"]["gate_agreement"]:.5f} of integer genotypes in the '
        f'{F["variants"]["gate_shared_donors"]} donors in both). Where two arms chose the same lead, their slopes agree in '
        'sign (each arm against tensorQTL at leads with tensorQTL p &lt; 1e-6, subset genes: ' +
        ', '.join(f'{a} {v["same_sign"]:.3f} of {v["shared_strong_leads"]}' for a, v in S['orientation'].items()) + ').</p>'
        '<p>The referee phenotype is a third quantification. Its counts are nf-core/rnaseq star_salmon on the standard, not '
        'personalized, T2T reference (STAR alignment, then Salmon in alignment mode, gene counts through tximeta), scaled by '
        'sample-specific effective-length factors and TMM (trimmed mean of M-values: a per-donor scaling factor from the '
        'weighted mean of log expression ratios against a reference donor, after trimming the most extreme genes), '
        'converted to counts per million and rank-inverse-normal transformed per gene over the 225 donors (each value '
        'replaced by the standard-normal quantile of its rank / (n + 1)). Which discovery count source it tracks was '
        f'measured on the {rs["donors"]} donors in both cohorts: the per-gene Spearman correlation (the Pearson correlation of the donors\' ranks) of the referee phenotype '
        f'with log2 CPM of the Salmon point-estimate totals (personalized transcriptome) has quartiles {rs["salmon"][0]:.3f} / '
        f'{rs["salmon"][1]:.3f} / {rs["salmon"][2]:.3f}, with the featureCounts totals TReCASE uses {rs["native"][0]:.3f} / '
        f'{rs["native"][1]:.3f} / {rs["native"][2]:.3f}; featureCounts is the closer in {rs["native_higher"]:.3f} of '
        f'{rs["genes"]:,} genes.</p>')
    body.append(
        '<h3>Statistics</h3><ul>'
        '<li><b>Lead</b>: the gene\'s tested variant with the smallest nominal p in that arm (ties by |slope / se|).</li>'
        '<li><b>Permutation p</b>: the gene-level p of the arm\'s permutation scan. pval_beta is the Beta approximation: a '
        'Beta distribution fitted to the 1,000 permuted minimum p gives the probability of a minimum as small as observed; '
        'pval_perm is the empirical share of permutations at least as extreme (floor 1/1,001).</li>'
        '<li><b>eigenMT p</b> (Davis et al. 2016): min(1, lead p &times; M<sub>eff</sub>), M<sub>eff</sub> the number of '
        'independent tests the gene\'s genotype correlation implies (eigenvalues of the correlation matrix in windows of 200 '
        'variants needed to explain 99% of its variance, after Ledoit-Wolf shrinkage: the sample correlation matrix pulled '
        'toward the identity by a weight estimated from the data). One formula for every arm, so it also ranks TReCASE, '
        'which has no permutation p.</li>'
        f'<li><b>Ranking and top K</b>: genes by gene-level p; within a tie (equal gene-level p, e.g. mixQTL\'s empirical p '
        f'at a count of permutations) by lead p, then |slope / se|. K = {", ".join(str(K) for K in KS)} in both gene sets; on '
        f'all genes also K = {", ".join(f"{K:,}" for K in extra)}, the same shares of all {N:,} genes as '
        f'{", ".join(str(K) for K in KS)} are of the {n:,}-gene subset (a given K reaches {N / n:.1f} times deeper into the '
        'subset\'s list). The tables give how many genes share the gene-level p of the K-th gene: above 1, the tie-break '
        'decided who is in.</li>'
        '<li><b>BH 5%</b>: the genes whose Benjamini-Hochberg adjusted gene-level p (the step-up adjustment that holds the '
        'expected share of false calls at 5% for independent tests) is at most 0.05 over the arm\'s ranked genes; its K is '
        'the arm\'s own.</li>'
        f'<li><b>Replication share</b>: the share of the K leads whose held-out p is below {REP_ALPHA} and whose held-out '
        'slope has the sign of the discovery slope. A lead with a held-out p below 0.05 and the opposite sign counts as not '
        'replicated.</li>'
        f'<li><b>pi1</b>: 1 - pi0, Storey\'s estimate of the share of the K held-out p that come from real effects; pi0 = '
        f'min(1, (number of p above {LAMBDA}) / ({1 - LAMBDA} K)), since null p are uniform and so put a share {1 - LAMBDA} of '
        'themselves above 0.5. It ignores the sign.</li>'
        f'<li><b>Interval</b> (per arm): the K genes resampled with replacement {N_BOOT:,} times, 2.5% and 97.5% quantiles of '
        'the recomputed statistic. It treats the K genes as given.</li>'
        f'<li><b>Paired difference</b> (against tensorQTL, each weighting against gibbs, and on the subset TReCASE against split '
        f'and gibbs): every gene of the set resampled '
        f'with replacement {N_BOOT:,} times; on each resample both arms re-take their own top K, so the interval also carries '
        'which genes enter the top K. This is the noise floor for comparing two arms.</li>'
        '<li><b>Base rate</b> (dotted lines in the figures): the statistic over every ranked gene\'s lead, what a ranking '
        'that carried no information would give.</li>'
        '<li><b>Overlap</b>: |top K of arm A &cap; top K of arm B| / K.</li></ul>')
    body.append(
        f'<h2>Results: seven arms on all {N:,} genes</h2>' +
        img(figs['share'], f'Replication share of the top K genes\' leads, all {N:,} genes. Left: ranked by each arm\'s '
                           'permutation p; right: by eigenMT p. Bars: the K-gene resampling 95% interval. Dotted lines: each '
                           'arm\'s base rate over all its ranked genes.') +
        img(figs['pi1'], 'pi1 (Storey) of the top K genes\' held-out p. Panels, bars and dotted lines as above.') +
        RESULTS_TEXT(S, 'all') +
        '<h3>Is a difference between arms larger than its noise floor?</h3>' +
        img(figs['paired'], 'Each arm minus tensorQTL at matched K, eigenMT ranking, with the whole-set gene-resampling 95% '
                            'interval of the difference. Left: replication share; right: pi1.') +
        '<p>Paired differences, eigenMT ranking:</p>' + paired_table(A, 'eigenmt') +
        '<p>Paired differences, permutation-p ranking:</p>' + paired_table(A, 'perm') + PAIRED_TEXT(S, 'all') +
        '<h3>Ranked by eigenMT p</h3>' + k_table(A, 'eigenmt') +
        '<h3>Ranked by permutation p</h3>' + k_table(A, 'perm') +
        '<h3>Benjamini-Hochberg 5%</h3>' + bh_table(A) + BH_TEXT(S, 'all') +
        '<h3>Where a gap to tensorQTL comes from: gene choice or lead choice</h3><p>Each arm\'s top K split into the genes '
        'also in tensorQTL\'s top K and those that are not. Among shared genes: how often both arms chose the same lead '
        'variant, and each lead\'s replication share. Among the rest: each arm\'s own genes\' share. eigenMT ranking.</p>' +
        decompose_table(A, 'eigenmt') + DECOMPOSE_TEXT(S, 'all', K_SHOW) + DECOMPOSE_TEXT(S, 'all', KD) +
        '<h3>Leads driven by the allelic channel</h3><p>For the arms with two channels, each top-K lead classed as allelic-led '
        '(its allelic-channel p below its total-channel p), total-led, or total-only (no allelic p). If the referee could not '
        'credit allelic evidence, allelic-led leads would replicate less. eigenMT ranking.</p>' +
        channel_table(S, 'all', 'eigenmt') + CHANNEL_TEXT(S, 'all', HAPMIX + MIX) + expected_table(S) + CONTROL_TEXT(S) +
        '<h3>Do the arms choose the same genes?</h3>' +
        img(figs['overlap'], f'Share of the top {K_SHOW} genes two arms have in common, by permutation p (left) and eigenMT p '
                             '(right), all genes.') + OVERLAP_TEXT(S, 'all'))
    body.append(
        f'<h2>TReCASE comparison: every arm on the {n:,}-gene subset</h2>' +
        img(figs['sub_share'], f'Replication share against K on the {n:,}-gene subset, every arm restricted to it. As the first '
                               'figure; TReCASE appears in the eigenMT panel only.') +
        img(figs['sub_paired'], f'Each arm minus tensorQTL on the {n:,}-gene subset, eigenMT ranking, with the whole-set '
                                'gene-resampling 95% interval.') + RESULTS_TEXT(S, 'subset') +
        '<p>Paired differences, eigenMT ranking:</p>' + paired_table(B, 'eigenmt') + PAIRED_TEXT(S, 'subset') +
        '<h3>Ranked by eigenMT p</h3>' + k_table(B, 'eigenmt') +
        '<h3>Benjamini-Hochberg 5%</h3>' + bh_table(B) +
        '<h3>Gene choice, lead choice and the allelic channel</h3>' + decompose_table(B, 'eigenmt') +
        DECOMPOSE_TEXT(S, 'subset', K_SHOW) + channel_table(S, 'subset', 'eigenmt') + CHANNEL_TEXT(S, 'subset', HAPMIX + TRECASE) +
        OVERLAP_TEXT(S, 'subset') + '<h3>The TReCASE run</h3>' + TRECASE_TEXT(S))
    body.append(CRITIQUE(S))
    body.append(MEANING(S))
    body.append(LIMITS(S))
    return ''.join(body)


if __name__ == '__main__':
    main()

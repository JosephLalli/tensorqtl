"""One-time checks of the benchmark's premise, generator and plumbing (README: Checks). Exits
non-zero if any check fails; nothing downstream re-checks what passes here.

(p) SALMON PREMISE of the allelic variance rule. Salmon 1.10.3's Gibbs round draws each transcript
    rate from Gamma(count + prior, ...) and reassigns the reads of every multi-transcript
    equivalence class by a multinomial on those rates (CollapsedGibbsSampler.cpp:149, 257-265),
    so for a gene's L/R pair the across-draw variance of log2((YL + 0.5)/(YR + 0.5)) beyond the
    counting term is, by the delta method, about s^2 (1/u_L + 1/u_R) / ln2^2, with u_h the reads
    in single-gene classes holding only haplotype h and s the ambiguous share. Checked on donor
    SAMPLE's dumped equivalence classes over cache genes with u_L, u_R >= MIN_U: PASS if the
    median of observed / predicted excess is in RATIO_BAND and its Spearman correlation with the
    observed excess exceeds the counting term's (thresholds set after the first result, 2026-09-26:
    a regression guard of the derivation).
(a) IDENTITY. With every thinning factor 1, no permutation and no swap, the generator's A, T, Va,
    Vt equal summaries_from_point_estimates on the cache exactly; with dataset 0's permutation and
    swap, T and Vt equal the real ones moved, A equals swap x A moved within A_TOL and Va within
    VA_RTOL (swapping L and R rounds log2(x/y) differently from log2(y/x)).
(b) THINNING keeps Salmon's draw behaviour: every gene thinned by F_B; by band of pL + pR the
    median Fano factor of the YT draws stays in FANO_BAND where a band has >= MIN_PAIRS pairs; and
    the allelic rule's arithmetic: (Va' - q_a') / (Va - q_a) = q(pL', pR') / q(pL, pR) within
    RULE_RTOL on every informative record, Va' = 0 wherever pL' + pR' = 0.
(c) RECOVERY of an injected |beta| = C_BETA over C_N all-non-null datasets: per gene the allelic
    slope at the causal variant (null_permutation_instrument.fit_channels, through-origin weighted
    least squares over the records common.allelic_kept admits) under seven weight / truth pairs;
    PASS if over genes with >= C_MIN_READS median haplotype reads the unit-weighted slope over the
    pipeline-scale truth is within C_SE_MULT gene-clustered se of 1 (rule set 2026-09-26 before its
    first run), and map_nominal (gibbs) on dataset 0 matches fit_channels at every causal variant
    within GATE_TOL of the se.
(d) EXACT REPRODUCTION of a stored null: given the stored null runs' own permutation 0 and swap
    signs, the beta = 0 path through map_nominal (gibbs and unit) reproduces those runs' draw 0
    in everything commit 8a06803 leaves alone (channel slopes and se within REPRO_SLOPE_TOL, the
    admitted combined slope, pval_t and its calls at REPRO_ALPHAS) and its new t references
    recompute from the stored statistics (pval_a at dof_a = n_a - 1, the admitted pval_nominal
    at the Welch-Satterthwaite dof within DOF_RTOL, the total channel cloned below the
    MIN_ALLELIC_DONORS floor); the stored p equal 2 t.sf(|t|, OLD_DOF) of their own statistic.
(e) PLUMBING GATES, once, on dataset 0 of check (c) (the earlier pipeline's per-dataset gates):
    mixqtl_scan's lead (largest |meta stat|), beta and se equal compare_mixqtl_replication.
    mixqtl_gene's on the same inputs in every gene; map_cis (gibbs) scans exactly the tested
    variants with varying dosage, its lead is one of them and its lead slope equals map_nominal's
    within GATE_TOL of the se; and on every tested pair of the gibbs and mixqtl outputs the
    combined slope is the inverse-variance combination of the channel slopes at their own se
    within IVW_TOL (the total channel alone below the floor, or per mixQTL's `method`).

GATE DISPOSITION of the earlier pipeline's per-dataset checks (2026-09-27): run_arms gate_nominal
-> (c) and (d); run_arms gate_mixqtl and the map_cis gate -> (e); score's IVW check -> (e);
map_nominal / mixqtl_scan / RASQUAL / asSeq output checks (row counts, ids, pi in (0, 1), the
final_Pvalue rule) kept where each tool's output is read; the mixqtl_permutation_scan identity
gate dropped with the scan (never run, timing rule); score's truth.tsv-vs-datasets check dropped
(02 writes both from the same arrays); run_rasqual's SMOKE causal-rank check dropped
(99_acceptance.py compares a whole dataset with the 2026-09-26 run); run_trecase's check_windows
and input finiteness checks dropped (convert checks each gene's row set; 02 validates its arrays).

Output: CHECKS/salmon_premise.json, CHECKS/check_generator.json (08_report.py reads both).
"""
import gzip
import json
import shutil
from collections import Counter

import numpy as np
import pandas as pd

import common as C
from null_permutation_instrument import fit_channels
from tensorqtl.hapmixqtl import LN2, MIN_ALLELIC_DONORS, get_t_pval, summaries_from_point_estimates

MD, RA = C.module('02_make_datasets'), C.module('03_run_arms')
CACHE = C.D / 'cache' / 'gibbs_56b63c3b37ed5df8'     # the Gibbs draws, the pipeline's input
MANIFEST = C.D / 'cohort' / 'salmon.tsv'              # DNA library id -> Salmon directory (never keyed on a name)
TX2GENE = C.D / 'annot' / 'tx2gene.tsv'               # the cache's transcript -> gene map
SAMPLE = '100_D1'                    # first manifest row; the donor whose equivalence classes were dumped (--dumpEq)
SALMON_VERSION = '1.10.3'            # the version whose CollapsedGibbsSampler.cpp the rule is read from
MIN_U = 20                           # informative reads per haplotype for a stable 1/u (exploratory run, 2026-09-26)
RATIO_BAND = (0.8, 1.25)             # set after the first result (median 0.989, 2026-09-26)
SHARE_BINS = ((0, .5), (.5, .8), (.8, .95), (.95, 1.01))   # ambiguous-share bins of the premise's per-bin ratio (scripts/plasmode/check_salmon_premise.py, 2026-09-26)
SUFFIXES = ('_L', '_R')              # g2gtools haplotype suffixes
# The (a) and (b) thresholds were set in scripts/plasmode/check_generator.py on 2026-09-26 without a recorded basis; the measured values are from
# the 100-gene set's checks/check_generator.json (plasmode_20260926 and this run).
A_TOL, VA_RTOL = 1e-12, 1e-9         # measured 1.8e-15 / 7.6e-16
F_B = 0.5                            # deeper than the largest scenario's 2^-0.8 = 0.574
MIN_PAIRS = 1000                     # exempts the 1-9 read band (364 thinned pairs on the 100-gene set)
FANO_BAND = (0.95, 1.02)             # measured 0.990-0.994 in the bands with >= MIN_PAIRS pairs
RULE_RTOL = 1e-9                     # measured 3.8e-15
C_BETA, C_N = 0.4, 20                # the middle scenario and 20 all-non-null datasets: scripts/plasmode/check_generator.py, 2026-09-26, basis not recorded
C_MIN_READS = 100                    # the lower edge of 06_score's 100-999 read band (user decision 2026-09-26)
C_SE_MULT = 3                        # set 2026-09-26 before the rule's first run
BANDS = ((1, 10), (10, 100), (100, 1000), (1000, np.inf))                     # decades of pL + pR per donor-gene pair (check b)
C_BANDS = ([0, 10, 100, 1000, np.inf], ['0-9', '10-99', '100-999', '1000+'])   # the same decades of a gene's median haplotype reads (check c)
REPRO_DRAWS = {'gibbs': C.GENE_DIR / 'draws' / 'drop_000.parquet', 'unit': C.HYBRID_NULL / 'draws' / 'unit_000.parquet'}
REPRO_ALPHAS = C.ALPHAS
REPRO_SLOPE_TOL = 1e-4               # stored draws are float32; 2026-09-26 measured 5.5e-6
REPRO_P_RTOL, DOF_RTOL = 1e-3, 1e-5  # set 2026-09-27 before their first run (p and dof from float32 statistics)
STORED_RTOL = 1e-6                   # the stored p against its own statistic: measured 6.0e-8 (one float32 rounding of t)
GATE_TOL = 1e-3                      # corrected_null_store.py's gate, max |diff| / se
IVW_TOL = 1e-4                       # combined slope vs the inverse-variance combination, / se (float32; measured 2.6e-6 at causal units)
ESTIMATES = (('inv_va_nodrop_beta', "1/Va' no drop, vs beta"), ('inv_va_beta', "1/Va', vs beta"),
             ('inv_va_exp_beta', '1/Va_exp, vs beta'), ('inv_va_real_beta', '1/Va_real, vs beta'),
             ('unit_beta', 'unit, vs beta'), ('unit_pipeline', 'unit, vs pipeline truth'),
             ('inv_va_pipeline', "1/Va', vs pipeline truth"))


def status(ok):
    return 'PASS' if ok else 'FAIL'


def salmon_premise():
    manifest = pd.read_csv(MANIFEST, sep='\t', header=None, names=['sample', 'dir']).set_index('sample')['dir']
    qdir = C.Path(manifest[SAMPLE])
    meta = json.loads((qdir / 'aux_info' / 'meta_info.json').read_text())
    if meta['salmon_version'] != SALMON_VERSION or meta['samp_type'] != 'gibbs':
        raise SystemExit(f'{qdir}: salmon_version {meta["salmon_version"]}, samp_type {meta["samp_type"]}')
    with gzip.open(qdir / 'aux_info' / 'eq_classes.txt.gz', 'rt') as fh:
        n_t, n_e = int(fh.readline()), int(fh.readline())
        names = [fh.readline().rstrip('\n') for _ in range(n_t)]
        classes = [fh.readline().split() for _ in range(n_e)]
    reads = sum(int(c[-1]) for c in classes)
    if any(len(c) != int(c[0]) + 2 for c in classes) or n_t != meta['num_valid_targets'] or reads != meta['num_mapped']:
        raise SystemExit(f'{qdir}: equivalence classes do not match meta_info.json (weights dumped?)')
    print(f'(p) {SAMPLE}: {n_t:,} targets, {n_e:,} equivalence classes, {reads:,} reads', flush=True)
    tx2gene = pd.read_csv(TX2GENE, sep='\t', header=None, names=['tx', 'gene']).set_index('tx')['gene']
    base = np.array([n[:-2] if n.endswith(SUFFIXES) else n for n in names])
    hap = np.array([n[-1] if n.endswith(SUFFIXES) else '' for n in names])
    paired = np.isin(base, list(set(base[hap == 'L']) & set(base[hap == 'R'])))
    gene = tx2gene.reindex(base)
    if gene.isna().any():
        raise SystemExit(f'{int(gene.isna().sum())} Salmon targets have no gene in {TX2GENE}')
    gene = gene.to_numpy()
    uL, uR, amb = Counter(), Counter(), Counter()
    for c in classes:
        t = np.array(c[1:-1], dtype=int)
        g = set(gene[t])
        if len(g) != 1:
            continue
        hp = set(hap[t[paired[t]]])
        for key, cnt in (({'L'}, uL), ({'R'}, uR), ({'L', 'R'}, amb)):
            if hp == key:
                cnt[g.pop()] += int(c[-1])
    genes = (CACHE / 'genes.txt').read_text().split()
    j = (CACHE / 'samples.txt').read_text().split().index(SAMPLE)
    rows = [i for i, g in enumerate(genes) if uL[g] >= MIN_U and uR[g] >= MIN_U]
    yl = np.asarray(np.load(CACHE / 'YL.npy', mmap_mode='r')[rows, j])
    yr = np.asarray(np.load(CACHE / 'YR.npy', mmap_mode='r')[rows, j])
    counting = 1 / (yl.mean(1) + C.KAPPA) + 1 / (yr.mean(1) + C.KAPPA)
    excess = np.log2((yl + C.KAPPA) / (yr + C.KAPPA)).var(axis=1) - counting / LN2 ** 2
    u_l, u_r, a = (np.array([cnt[genes[i]] for i in rows], float) for cnt in (uL, uR, amb))
    H, s = (1 / u_l + 1 / u_r) / LN2 ** 2, a / (a + u_l + u_r)
    keep = (excess > 0) & (s > 0)
    ratio = excess[keep] / (s[keep] ** 2 * H[keep])
    sp = lambda x, y: float(pd.Series(x).corr(pd.Series(y), method='spearman'))   # noqa: E731
    res = dict(sample=SAMPLE, salmon_dir=str(qdir), min_u=MIN_U, genes_min_u=len(rows), genes_excess_le_0=int((excess <= 0).sum()),
               genes_s_eq_0=int((s <= 0).sum()), genes_retained=int(keep.sum()), ratio_median=float(np.median(ratio)),
               ratio_iqr=[float(np.quantile(ratio, .25)), float(np.quantile(ratio, .75))],
               spearman_excess_s2H=sp(excess[keep], (s ** 2 * H)[keep]), spearman_excess_counting=sp(excess[keep], counting[keep]),
               median_informative_share=float(np.median(1 - s)),
               by_ambiguous_share={f'[{lo}, {hi})': dict(genes=int(m.sum()), ratio_median=float(np.median(excess[m] / (s[m] ** 2 * H[m]))))
                                   for lo, hi in SHARE_BINS for m in [keep & (s >= lo) & (s < hi)] if m.any()},
               ratio_band=list(RATIO_BAND), thresholds_set_after_first_result=True)
    ok = RATIO_BAND[0] <= res['ratio_median'] <= RATIO_BAND[1] and res['spearman_excess_s2H'] > res['spearman_excess_counting']
    res['passed'] = bool(ok)
    print(f'(p) excess / (s^2 H): median {res["ratio_median"]:.3f} IQR [{res["ratio_iqr"][0]:.3f}, {res["ratio_iqr"][1]:.3f}] over '
          f'{int(keep.sum())} of {len(rows)} genes; Spearman with excess {res["spearman_excess_s2H"]:.3f} against '
          f'{res["spearman_excess_counting"]:.3f} for the counting term  {status(ok)}', flush=True)
    C.write_json(C.CHECKS / 'salmon_premise.json', res)
    return ok


def check_identity(I, R):
    G, N = R['pL'].shape
    ref = dict(zip(('A', 'T', 'Va', 'Vt'), summaries_from_point_estimates(
        I['pL'], I['pR'], I['pT'], I['eff_lib'], I['YL'], I['YR'], I['YT'])[:4]))
    ref = {k: v[:, I['keep']] for k, v in ref.items()}
    ones = np.ones((G, N))
    rng = np.random.default_rng(np.random.SeedSequence(C.SEED, spawn_key=(10,)))
    g = MD.generate(R, np.arange(N), np.ones(N, np.int8), ones, ones, rng)
    exact = {k: bool(np.array_equal(g[k], ref[k])) for k in ref}
    perm, swap = MD.record_permutation(N, 0)
    g2 = MD.generate(R, perm, swap, ones, ones, rng)
    moved = dict(T=bool(np.array_equal(g2['T'], ref['T'][:, perm])), Vt=bool(np.array_equal(g2['Vt'], ref['Vt'][:, perm])))
    dA = float(np.abs(g2['A'] - swap[None, :] * ref['A'][:, perm]).max())
    va_ref = ref['Va'][:, perm]
    dVa = float((np.abs(g2['Va'] - va_ref) / np.where(va_ref > 0, va_ref, 1.0)).max())
    ok = all(exact.values()) and all(moved.values()) and dA <= A_TOL and dVa <= VA_RTOL
    print(f'(a) identity at f = 1: exactly equal {exact}; with permutation and swap {moved}, max |dA| {dA:.1e}, '
          f'max relative |dVa| {dVa:.1e}  {status(ok)}', flush=True)
    return ok, dict(exact=exact, moved_exact=moved, max_abs_dA=dA, max_rel_dVa=dVa, passed=ok)


def band_stats(pL, pR, YT, Va):
    hap, mean = pL + pR, YT.mean(2)
    qa = (1.0 / (pL + C.KAPPA) + 1.0 / (pR + C.KAPPA)) / LN2 ** 2
    both = (pL >= C.EXPRESSIBLE_MIN) & (pR >= C.EXPRESSIBLE_MIN)
    res = {}
    for lo, hi in BANDS:
        m = (hap >= lo) & (hap < hi)
        mf, mb = m & (mean > 0), m & both
        res[f'{lo}-{hi - 1:g}' if np.isfinite(hi) else f'{lo}+'] = dict(
            pairs=int(m.sum()), fano=float(np.median(YT[mf].var(1, ddof=1) / mean[mf])) if mf.any() else np.nan,
            va_over_qa=float(np.median(Va[mb] / qa[mb])) if mb.any() else np.nan)
    return res


def check_thinning(I, R):
    G, N = R['pL'].shape
    real = MD.generate(R, np.arange(N), np.ones(N, np.int8), np.ones((G, N)), np.ones((G, N)),
                       np.random.default_rng(np.random.SeedSequence(C.SEED, spawn_key=(11,))))
    f = np.full((G, N), F_B)
    th = MD.generate(R, np.arange(N), np.ones(N, np.int8), f, f,
                     np.random.default_rng(np.random.SeedSequence(C.SEED, spawn_key=(12,))))
    sr, st = (band_stats(x['pL'], x['pR'], x['YT'], x['Va']) for x in (real, th))
    ok_fano = all(FANO_BAND[0] <= st[b]['fano'] <= FANO_BAND[1] for b in st if st[b]['pairs'] >= MIN_PAIRS)
    q = lambda pL, pR: 1.0 / (pL + C.KAPPA) + 1.0 / (pR + C.KAPPA)   # noqa: E731
    q0, q1 = q(real['pL'], real['pR']), q(th['pL'], th['pR'])
    h0, h1 = real['pL'] + real['pR'], th['pL'] + th['pR']
    gibbs0 = real['Va'] - q0 / LN2 ** 2
    inf = (h0 > 0) & (h1 > 0)
    m = inf & (gibbs0 > 0)
    dev = np.abs((th['Va'][m] - q1[m] / LN2 ** 2) / gibbs0[m] - q1[m] / q0[m]) / (q1[m] / q0[m])
    rule = dict(records_checked=int(m.sum()), informative_without_gibbs_part=int((inf & ~(gibbs0 > 0)).sum()),
                max_rel_dev=float(dev.max()), rtol=RULE_RTOL, va_zero_where_no_reads=bool((th['Va'][h1 <= 0] == 0).all()))
    rule['passed'] = bool(rule['informative_without_gibbs_part'] == 0 and rule['max_rel_dev'] <= RULE_RTOL and rule['va_zero_where_no_reads'])
    ok = ok_fano and rule['passed']
    print(f'(b) thinning by {F_B}: median YT Fano thinned (real) by band of pL + pR: '
          + ', '.join(f'{b} {st[b]["fano"]:.3f} ({sr[b]["fano"]:.3f}, {st[b]["pairs"]} pairs)' for b in st)
          + f'; allelic rule max relative deviation {rule["max_rel_dev"]:.1e} over {rule["records_checked"]:,} records  {status(ok)}', flush=True)
    return ok, dict(f=F_B, thinned=st, real=sr, fano_passed=bool(ok_fano), allelic_rule=rule, passed=bool(ok))


def gate_nominal(S, ds, df, Va, Vt):
    """map_nominal's channel slopes at each causal variant against fit_channels: (max |diff| / se, genes)."""
    I = S['I']
    Cg = np.column_stack([I['cov_df'].values[ds['perm']], I['geno_cov_df'].values])
    d = df.set_index(['phenotype_id', 'variant_id'])
    dev = []
    for k, g in enumerate(S['genes']):
        v = str(ds['causal_variant'][k])
        j = S['rows'][v]
        fc = fit_channels(ds['A'][k], (I['xL'][j] - I['xR'][j]).astype(float), Va[k], ds['T'][k],
                          I['dos'][j].astype(float) / 2.0, Vt[k], Cg)
        if fc is None:
            continue
        r = d.loc[(g, v)]
        dev += [abs(float(r.slope_a) - fc['ba']) / float(r.slope_a_se), abs(float(r.slope_t) - fc['bt']) / float(r.slope_t_se)]
    dev = np.array(dev)
    if not np.isfinite(dev).all():
        raise SystemExit('gate: non-finite |map_nominal - fit_channels| / se at a causal variant')
    return float(dev.max()), len(dev) // 2


def check_recovery(I, R, tested):
    genes = list(I['genes'])
    hap_real = np.median(R['pL'] + R['pR'], axis=1)
    recs, none, first = [], 0, None
    for r in range(C_N):
        ds = MD.build_dataset(I, R, tested, C_BETA, 0.0, r)
        M = MD.move_records(R, ds['perm'], ds['swap'])
        va_exp = MD.allelic_variance(M['pL'], M['pR'], ds['fL'] * M['pL'], ds['fR'] * M['pR'], M['YL'], M['YR'])
        va_real = MD.allelic_variance(M['pL'], M['pR'], M['pL'], M['pR'], M['YL'], M['YR'])
        Cg = np.column_stack([I['cov_df'].values[ds['perm']], I['geno_cov_df'].values])
        for k in range(len(genes)):
            j, kept = ds['causal_row'][k], ds['kept'][k]
            s = (I['xL'][j] - I['xR'][j]).astype(float)
            va = {'inv_va_nodrop': ds['Va'][k], 'inv_va': np.where(kept, ds['Va'][k], 0.0),
                  'inv_va_exp': np.where(kept, va_exp[k], 0.0), 'inv_va_real': np.where(kept, va_real[k], 0.0),
                  'unit': kept.astype(float)}
            fc = {w: fit_channels(ds['A'][k], s, v, ds['T'][k], I['dos'][j].astype(float) / 2.0, ds['Vt'][k], Cg) for w, v in va.items()}
            if any(x is None for x in fc.values()):
                none += 1
                continue
            b, bp = ds['beta'][k], ds['allelic_truth_pipeline'][k]
            recs.append(dict(rep=r, gene=genes[k], hap_real=hap_real[k], inv_va_nodrop_beta=fc['inv_va_nodrop']['ba'] / b,
                             inv_va_beta=fc['inv_va']['ba'] / b, inv_va_exp_beta=fc['inv_va_exp']['ba'] / b,
                             inv_va_real_beta=fc['inv_va_real']['ba'] / b, unit_beta=fc['unit']['ba'] / b,
                             unit_pipeline=fc['unit']['ba'] / bp, inv_va_pipeline=fc['inv_va']['ba'] / bp))
        if r == 0:
            first = ds
    t = pd.DataFrame(recs)
    cols = [c for c, _ in ESTIMATES]
    if (~np.isfinite(t[cols])).any(axis=1).sum():
        raise SystemExit('(c): a gene-dataset unit has a non-finite slope / truth')
    t['band'] = pd.cut(t.hap_real, C_BANDS[0], right=False, labels=C_BANDS[1])
    summ = lambda x, col: dict(genes=int(x.gene.nunique()), units=len(x), mean=float(x[col].mean()),   # noqa: E731
                               gene_clustered_se=float(x.groupby('gene')[col].mean().std(ddof=1) / np.sqrt(x.gene.nunique())))
    hi = t[t.hap_real >= C_MIN_READS]
    res = dict(beta=C_BETA, n_datasets=C_N, units_fewer_than_fit_channels_minimum=none,
               primary={c: summ(hi, c) for c in cols},
               by_band={str(b): {c: summ(x, c) for c in cols} for b, x in t.groupby('band', observed=True)})
    p = res['primary']['unit_pipeline']
    ok = abs(p['mean'] - 1) <= C_SE_MULT * p['gene_clustered_se']
    print(f'(c) recovery at |beta| = {C_BETA}, {C_N} datasets, genes >= {C_MIN_READS} reads ({p["genes"]} genes, {p["units"]} units): '
          + '; '.join(f'{label} {res["primary"][c]["mean"]:.3f} ({res["primary"][c]["gene_clustered_se"]:.3f})' for c, label in ESTIMATES)
          + f'  {status(ok)}', flush=True)
    return ok, res, first


def t_stat(d, s, se):
    """map_nominal's statistic: float32 slope / se, 0 where se is not finite and positive; as float64."""
    x, e = d[s].to_numpy(np.float32), d[se].to_numpy(np.float32)
    ok = np.isfinite(e) & (e > 0)
    return np.where(ok, x / np.where(ok, e, np.float32(1)), np.float32(0)).astype(np.float64)


def p_agree(p, q, rtol, alphas=REPRO_ALPHAS):
    """p against q: calls differing per alpha, max relative |p - q|; passed if NaN and zero patterns match too."""
    fp, fq = np.isfinite(p), np.isfinite(q)
    b = fp & fq
    pos = b & (p > 0) & (q > 0)
    rel = float(np.max(np.abs(p[pos] - q[pos]) / q[pos])) if pos.any() else 0.0
    calls = {str(a): int(((p[b] < a) != (q[b] < a)).sum()) for a in alphas}
    return dict(calls_differ=calls, max_rel=rel, tests=int(b.sum()),
                passed=bool((fp == fq).all() and ((p[b] == 0) == (q[b] == 0)).all() and rel <= rtol and not any(calls.values())))


def pinned(m, rows, s, se):
    """Over rows: slope within REPRO_SLOPE_TOL of its se, se within REPRO_SLOPE_TOL relative, of the stored draw's."""
    a, b = m[s].to_numpy(float)[rows], m[f'{s}_stored'].to_numpy(float)[rows]
    e, f = m[se].to_numpy(float)[rows], m[f'{se}_stored'].to_numpy(float)[rows]
    fin = np.isfinite(e) & (e > 0)
    ds = float(np.max(np.abs(a[fin] - b[fin]) / e[fin])) if fin.any() else 0.0
    de = float(np.max(np.abs(e[fin] - f[fin]) / f[fin])) if fin.any() else 0.0
    same_rest = bool(np.array_equal(fin, np.isfinite(f) & (f > 0)) and np.array_equal(a[~fin], b[~fin], equal_nan=True))
    return dict(max_slope_diff_se=ds, max_se_rel=de, finite=int(fin.sum()),
                passed=bool(same_rest and ds <= REPRO_SLOPE_TOL and de <= REPRO_SLOPE_TOL))


def satterthwaite(se_a, se_t, dof_a, dof_t):
    """hapmixqtl._satterthwaite_dof in float64 from the stored se: weights 1/se^2, channel dof clamped at 1."""
    with np.errstate(divide='ignore', invalid='ignore'):
        wa = np.where(np.isfinite(se_a) & (se_a > 0), 1.0 / se_a ** 2, 0.0)
        wt = np.where(np.isfinite(se_t) & (se_t > 0), 1.0 / se_t ** 2, 0.0)
        nu_a, nu_t = np.maximum(np.nan_to_num(dof_a, nan=1.0), 1.0), np.maximum(np.nan_to_num(dof_t, nan=1.0), 1.0)
        nu = (wa + wt) ** 2 / (wa ** 2 / nu_a + wt ** 2 / nu_t)
    nu = np.where(wa > 0, np.where(wt > 0, nu, nu_a), nu_t)
    return np.where(wa + wt > 0, nu, np.nan)


def check_reproduction(I, R, S, scratch):
    old = np.load(C.CNS.OLD / 'permutations.npz')
    perm, swap = old['perms'][0], old['flips'][0].astype(np.int8)
    G, N = R['pL'].shape
    ones = np.ones((G, N))
    g = MD.generate(R, perm, swap, ones, ones, np.random.default_rng(np.random.SeedSequence(C.SEED, spawn_key=(13,))))
    ds = dict(A=g['A'], T=g['T'], Va=g['Va'], Vt=g['Vt'], pL=g['pL'], pR=g['pR'], perm=perm,
              causal_variant=np.array([min(S['tested'][x]) for x in S['genes']]))
    old_dof = N - 2 - I['cov_df'].shape[1] - I['geno_cov_df'].shape[1]
    keep_design = pd.read_csv(C.GENE_DESIGN, sep='\t').set_index('gene').n_allelic_drop.loc[S['genes']]
    ok, res = True, {}
    for arm, path in REPRO_DRAWS.items():
        df, _ = C.run_nominal(S, ds, arm, scratch)
        ref = pd.read_parquet(path)
        m = df.merge(ref, on=['phenotype_id', 'variant_id'], suffixes=('', '_stored'))
        if len(m) != len(ref) or len(m) != len(df):
            raise SystemExit(f'(d) {arm}: {len(m):,} matched tests, {len(df):,} here, {len(ref):,} in {path}')
        col = lambda c: m[c].to_numpy(float)   # noqa: E731
        n_a = pd.Series((C.arm_variances(ds, arm)[0] > C.EPS).sum(1), index=S['genes'])
        na, adm = n_a.loc[m.phenotype_id].to_numpy(), m.allelic_admitted.to_numpy(bool)
        structure = dict(n_a_equals_gene_design=bool((n_a == keep_design).all()),
                         dof_a=bool(np.array_equal(col('dof_a'), np.where(na >= 2, na - 1.0, np.nan), equal_nan=True)),
                         dof_t_all_old_dof=bool((col('dof_t') == old_dof).all()),
                         allelic_admitted=bool(np.array_equal(adm, na >= MIN_ALLELIC_DONORS)))
        every = np.ones(len(m), bool)
        pin = dict(allelic=pinned(m, every, 'slope_a', 'slope_a_se'), total=pinned(m, every, 'slope_t', 'slope_t_se'),
                   combined_admitted=pinned(m, adm, 'slope', 'slope_se'),
                   pval_t=p_agree(col('pval_t'), col('pval_t_stored'), REPRO_P_RTOL))
        t_a, t_t, t_c = (t_stat(m, f'{s}_stored', f'{se}_stored') for s, se in (('slope_a', 'slope_a_se'), ('slope_t', 'slope_t_se'), ('slope', 'slope_se')))
        stored_ref = {c: p_agree(col(f'{c}_stored'), get_t_pval(t, old_dof), STORED_RTOL, alphas=())
                      for c, t in (('pval_a', t_a), ('pval_t', t_t), ('pval_nominal', t_c))}
        ws = satterthwaite(col('slope_a_se'), col('slope_t_se'), col('dof_a'), col('dof_t'))[adm]
        dn = col('dof_nominal')[adm]
        fin = np.isfinite(ws) & np.isfinite(dn)
        dof_rel = float(np.max(np.abs(dn[fin] - ws[fin]) / ws[fin]))
        changed = dict(pval_a=p_agree(col('pval_a'), get_t_pval(t_a, col('dof_a')), REPRO_P_RTOL),
                       dof_nominal_admitted=dict(max_rel=dof_rel, passed=bool(np.array_equal(np.isfinite(ws), np.isfinite(dn)) and dof_rel <= DOF_RTOL)),
                       pval_nominal_admitted=p_agree(col('pval_nominal')[adm], get_t_pval(t_c[adm], dn), REPRO_P_RTOL))
        b = ~adm
        bt = b & np.isfinite(col('slope_t_se'))
        eq = lambda x, y: bool(np.array_equal(col(x)[bt], col(y)[bt]))   # noqa: E731
        below = dict(genes=sorted(m.phenotype_id[b].unique()), tests=int(b.sum()), slope=eq('slope', 'slope_t'),
                     slope_se=eq('slope_se', 'slope_t_se'), dof_nominal=eq('dof_nominal', 'dof_t'),
                     pval_nominal=eq('pval_nominal', 'pval_t'), nan_without_total=bool(m.pval_nominal[b & ~bt].isna().all()))
        moved = {ch: {str(a): int(((m[c].fillna(1.0) < a) != (m[f'{c}_stored'].fillna(1.0) < a)).sum()) for a in REPRO_ALPHAS}
                 for ch, c in C.CHANNELS.items()}
        passed = bool(all(structure.values()) and all(x['passed'] for x in pin.values()) and all(x['passed'] for x in stored_ref.values())
                      and all(x['passed'] for x in changed.values())
                      and all(below[k] for k in ('slope', 'slope_se', 'dof_nominal', 'pval_nominal', 'nan_without_total')))
        ok &= passed
        res[arm] = dict(stored=str(path), tests=len(m), old_dof=old_dof, structure=structure, pinned=pin, stored_reference=stored_ref,
                        changed=changed, below_floor=below, calls_moved=moved, passed=passed)
        print(f'(d) {arm} vs {path.name}: {len(m):,} tests; structure {structure}; pinned slopes within '
              f'{max(pin[k]["max_slope_diff_se"] for k in ("allelic", "total", "combined_admitted")):.1e} se, pval_t calls differing '
              f'{pin["pval_t"]["calls_differ"]}; stored p at t({old_dof}) max relative {max(x["max_rel"] for x in stored_ref.values()):.1e}; '
              f'new references: pval_a calls differing {changed["pval_a"]["calls_differ"]}, dof_nominal max relative {dof_rel:.1e}, '
              f'pval_nominal calls differing {changed["pval_nominal_admitted"]["calls_differ"]}; below the floor {below["genes"]} '
              f'({below["tests"]:,} tests) cloned from the total channel; calls moved {moved}  {status(passed)}', flush=True)
    return ok, res


def ivw_dev(df, mixqtl):
    """Max |combined slope - inverse-variance combination of the channel slopes| / se over the finite rows."""
    sa, sea, st, set_ = (df[c].to_numpy(float) for c in ('slope_a', 'slope_a_se', 'slope_t', 'slope_t_se'))
    with np.errstate(divide='ignore', invalid='ignore'):
        wa = np.where(np.isfinite(sa) & np.isfinite(sea), 1.0 / sea ** 2, 0.0)
        wt = np.where(np.isfinite(st) & np.isfinite(set_), 1.0 / set_ ** 2, 0.0)
        if mixqtl:
            wa, wt = np.where(df.method == 'trc', 0.0, wa), np.where(df.method == 'asc', 0.0, wt)
        else:
            wa = np.where(df.allelic_admitted.to_numpy(bool), wa, 0.0)
        ivw = (np.where(wa > 0, wa * sa, 0.0) + np.where(wt > 0, wt * st, 0.0)) / (wa + wt)
    c, sc = df.slope.to_numpy(float), df.slope_se.to_numpy(float)
    fin = np.isfinite(c) & np.isfinite(sc) & (sc > 0)
    return float(np.max(np.abs(c[fin] - ivw[fin]) / sc[fin])), int(fin.sum())


def check_gates(S, ds, scratch):
    """(e) The moved per-dataset gates on one dataset (docstring)."""
    I = S['I']
    df, _ = C.run_nominal(S, ds, 'gibbs', scratch)
    Va, Vt, _ = C.arm_variances(ds, 'gibbs')
    worst, n = gate_nominal(S, ds, df, Va, Vt)
    mix, _, _ = RA.run_mixqtl(S, ds, C.MIXQTL_ARMS['mixqtl'])
    y1, y2, yt = C.MX.inputs_from_point_estimates(ds['pL'], ds['pR'], ds['pT'])
    base = dict(I, cov_df=I['cov_df'].iloc[ds['perm']], lib_size=ds['eff_lib'])
    bad = 0
    for k, g in enumerate(S['genes']):
        rows = S['tested_rows'][g]
        Ig = dict(base, vdf=I['vdf'].iloc[rows], xL=I['xL'][rows], xR=I['xR'][rows], idx=np.arange(len(rows)))
        r = C.CM.mixqtl_gene(Ig, g, k, y1, y2, yt)
        d = mix[mix.phenotype_id == g]
        stat = np.abs(d.slope.to_numpy(float) / d.slope_se.to_numpy(float))
        lead = None
        if np.isfinite(stat).any():
            i = int(np.nanargmax(stat))
            lead = (str(d.variant_id.iloc[i]), float(d.slope.iloc[i]), float(d.slope_se.iloc[i]))
        bad += (None if r is None else (r['variant_id'], r['beta'], r['se'])) != lead
    cis = RA.run_cis(S, ds, 'gibbs', RA.cis_seed(0))
    nom = df.set_index(['phenotype_id', 'variant_id'])
    scan_ok, dev = True, []
    for r in cis.itertuples():
        want = S['scanned'][r.phenotype_id]
        scan_ok &= r.num_var == len(want) and r.variant_id in want
        m = nom.loc[(r.phenotype_id, r.variant_id)]
        dev.append(abs(r.slope - float(m.slope)) / float(m.slope_se))
    ivw_h, n_h = ivw_dev(df, False)
    ivw_m, n_m = ivw_dev(mix, True)
    res = dict(map_nominal_vs_fit_channels=dict(max_abs_diff_over_se=worst, genes=n),
               mixqtl_lead_vs_mixqtl_gene=dict(genes_differing=int(bad)),
               map_cis=dict(scanned_set_and_lead_ok=bool(scan_ok), max_lead_slope_diff_se=float(max(dev))),
               ivw=dict(gibbs=dict(max_dev_se=ivw_h, rows=n_h), mixqtl=dict(max_dev_se=ivw_m, rows=n_m)))
    ok = worst < GATE_TOL and bad == 0 and scan_ok and max(dev) < GATE_TOL and ivw_h <= IVW_TOL and ivw_m <= IVW_TOL
    res['passed'] = bool(ok)
    print(f'(e) gates on dataset 0 of (c): map_nominal vs fit_channels {worst:.1e} se over {n} genes; mixqtl lead differs from '
          f'mixqtl_gene in {bad} genes; map_cis scanned set and lead ok {scan_ok}, lead slope vs map_nominal {max(dev):.1e} se; '
          f'combined slope vs IVW of the channels: gibbs {ivw_h:.1e} se ({n_h:,} rows), mixqtl {ivw_m:.1e} se ({n_m:,} rows)  {status(ok)}',
          flush=True)
    return ok, res


def main():
    C.CHECKS.mkdir(parents=True, exist_ok=True)
    okp = salmon_premise()
    I, R, tested = C.load()
    S, scratch = C.setup(I), C.CHECKS / 'scratch'
    oka, ra = check_identity(I, R)
    okb, rb = check_thinning(I, R)
    okc, rc, ds0 = check_recovery(I, R, tested)
    okd, rd = check_reproduction(I, R, S, scratch)
    oke, re_ = check_gates(S, ds0, scratch)
    rc['map_nominal_gate'] = re_['map_nominal_vs_fit_channels']
    rc['passed'] = bool(okc and re_['map_nominal_vs_fit_channels']['max_abs_diff_over_se'] < GATE_TOL)
    shutil.rmtree(scratch)
    C.write_json(C.CHECKS / 'check_generator.json', dict(identity=ra, thinning=rb, recovery=rc, reproduction=rd, gates=re_))
    print(f'wrote {C.CHECKS}')
    if not (okp and oka and okb and rc['passed'] and okd and oke):
        raise SystemExit(f'FAILED: premise {okp}, identity {oka}, thinning {okb}, recovery {rc["passed"]}, reproduction {okd}, gates {oke}')
    print('ALL CHECKS PASS')


if __name__ == '__main__':
    main()

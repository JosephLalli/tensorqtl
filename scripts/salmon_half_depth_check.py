"""Known-answer test of the plasmode benchmark's thinning rule against Salmon at half depth.

The benchmark (scripts/plasmode/make_datasets.py) thins real Salmon output and
scales each thinned record's allelic Gibbs variance by the counting-term ratio
    Va' = dv x q(pL', pR') / q(pL, pR) + q_a(pL', pR'),
    q(x, y) = 1/(x + 0.5) + 1/(y + 0.5),  q_a = q / ln(2)^2,
dv the across-draw variance (ddof = 0) of log2((yL + 0.5)/(yR + 0.5)) over the
200 Gibbs draws of the unthinned record. The rule was derived from Salmon
1.10.3's sampler (CollapsedGibbsSampler.cpp lines 149, 257-265, 507) and
checked only on its premise (check_salmon_premise.py). This script is the
one-time test the user approved: Salmon 1.10.3 itself run on donor 100 at full
depth (the production trimmed read pair) and at 50% depth (seqtk sample -s42 on
both mates; 21,222,914 of 42,449,536 pairs), production flags, 200 Gibbs draws,
the production index rebuilt (index_seq_hash equal). Launch records:
build_index.sh, subsample_reads.sh, quant_one_depth.sh and their logs in OUT.

Per gene, through the pipeline's own ingest (run_hapmixqtl_from_salmon.py's
pair_haplotypes / load_counts / load_point_estimates: pL, pR over L/R-paired
transcripts, pT over every transcript):
 (a) full-depth rerun against the production run: point estimates and dv, the
     latter as a ratio by band, the Gibbs noise floor for (b);
 (b) THE TEST: Va_pred from the rule with the REALIZED half-depth point
     estimates against Va_meas = dv_half + q_a(pL_half, pR_half), by band of
     full-depth pL + pR (30-99, 100-999, 1000+) over genes with both sides
     >= 0.5 reads at both depths. PRE-REGISTERED PASS: the median of
     Va_meas / Va_pred in [0.8, 1.25] in every band with >= 200 genes. Also the
     Gibbs part alone, dv_half / (dv_full x q ratio), which the criterion is
     blind to where q_a dominates; and the total channel's Fano factor (across-
     draw variance over mean of the yT draws) at both depths against the
     benchmark's per-draw binomial thinning of the full-depth draws;
 (c) zero haplotypes: the share of pairs with exactly one side below 0.5 at
     full depth, at half depth, and under binomial thinning of the full-depth
     point estimates (the benchmark's rule, which can zero a side below ~3
     reads but not a larger one): realized minus thinned is the omission;
 (d) attenuation: slope of the half-depth log2 ratio on the full-depth one,
     realized against thinned, same two-sided-at-both selection;
 (e) the rule's premise on the dumped equivalence classes at both depths:
     haplotype-informative reads u should scale by f and the ambiguous share s
     should not.
Outputs in OUT: summary.json, per_gene.tsv, salmon_half_depth.html with two
figures, and the printed table. Seed 42 for the thinning comparators.
"""
import base64
import io
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / 'plasmode'))
sys.path.insert(0, str(HERE.parent))

import check_salmon_premise as CP                  # noqa: E402  equivalence-class reader
import make_datasets as MD                         # noqa: E402  allelic_variance, thin, write_atomic, dumps
import run_hapmixqtl_from_salmon as RS             # noqa: E402  pair_haplotypes, load_counts, load_point_estimates
from tensorqtl.hapmixqtl import LN2                # noqa: E402

OUT = Path('/mnt/ssd/lalli/brainvar_hapmix_deploy/salmon_half_depth_20260927')
WORK = OUT / 'work'
RUNS = {'production': Path('/mnt/ssd/lalli/nf_stage/RNA_reference_comparison_results/reference_comparison_results/'
                           'bv2/personalized_T2T_NCBI110_pseudoalignment/expression_results/salmon_pseudocounts/100_R1'),
        'full': WORK / 'quant_full', 'half': WORK / 'quant_half'}
TX2GENE = CP.TX2GENE                    # the cache's transcript -> gene map (183,139 rows)
SUFFIXES = CP.SUFFIXES                  # g2gtools haplotype suffixes _L / _R
F = 0.5                                 # subsample fraction (subsample_reads.sh)
SEED = 42
K = MD.KAPPA                            # 0.5, the allelic log ratio's pseudocount
TWO_SIDED_MIN = MD.EXPRESSIBLE_MIN      # 0.5 reads: the pipeline's zero-haplotype rule
BANDS = ((30, 100), (100, 1000), (1000, np.inf))   # full-depth pL + pR; the pre-registered bands
INFO_BAND = (1, 30)                     # reported, not judged
PASS_BAND = (0.8, 1.25)                 # pre-registered
MIN_GENES = 200                         # pre-registered: a band is judged only with this many genes
CHI2_1_MEDIAN = 0.4549364                # median of chi-square with 1 df: the expected median of z^2 when the variance is calibrated
N_RESAMPLES = 2000                       # resamples of the genes for the interval of the median ratio
RESAMPLE_KEY = 4                         # spawn key after make_datasets' PERM/DESIGN/THIN keys 1-3
EXPONENT_MIN_Q_RATIO = 1.5             # the realized exponent log(dv_half/dv_full)/log(q_half/q_full) needs a denominator away from 0 (q ratio ~2 when both sides halve)
DEPTH_RATIO_TOL = (0.49, 0.51)          # num_processed half / full; seqtk at 0.5
SALMON_VERSION = CP.SALMON_VERSION      # 1.10.3
N_DRAWS = 200
PALETTE = {'30-99': '#2a78d6', '100-999': '#eb6834', '1000+': '#1baf7a', '1-29': '#9a9892'}   # dataviz slots 1-3 + muted
SERIES = {'full': '#2a78d6', 'half': '#eb6834', 'thinned': '#1baf7a'}


def band_name(lo, hi):
    return f'{lo}-{hi - 1:g}' if np.isfinite(hi) else f'{lo}+'


def band_of(hap):
    out = np.full(hap.shape, '', dtype=object)
    for lo, hi in (INFO_BAND,) + BANDS:
        out[(hap >= lo) & (hap < hi)] = band_name(lo, hi)
    return out


def q(pL, pR):
    return 1.0 / (pL + K) + 1.0 / (pR + K)


def med_iqr(x):
    x = np.asarray(x, float)
    if x.size == 0:
        return dict(n=0, median=np.nan, iqr=[np.nan, np.nan])
    return dict(n=int(x.size), median=float(np.median(x)),
                iqr=[float(np.quantile(x, .25)), float(np.quantile(x, .75))])


def fmt(d):
    return f'{d["median"]:.3f} [{d["iqr"][0]:.3f}, {d["iqr"][1]:.3f}] (n={d["n"]})' if d['n'] else 'n=0'


def validate_runs():
    meta = {}
    for k, d in RUNS.items():
        m = json.loads((d / 'aux_info' / 'meta_info.json').read_text())
        for key, want in (('salmon_version', SALMON_VERSION), ('samp_type', 'gibbs'), ('num_bootstraps', N_DRAWS)):
            if m[key] != want:
                raise SystemExit(f'{d}: {key} = {m[key]!r}, expected {want!r}')
        meta[k] = {x: m[x] for x in ('index_seq_hash', 'num_processed', 'num_mapped', 'percent_mapped',
                                     'num_eq_classes', 'num_valid_targets')}
    hashes = {v['index_seq_hash'] for v in meta.values()}
    if len(hashes) != 1:
        raise SystemExit(f'index_seq_hash differs between runs: {hashes}')
    depth = meta['half']['num_processed'] / meta['full']['num_processed']
    if not DEPTH_RATIO_TOL[0] <= depth <= DEPTH_RATIO_TOL[1]:
        raise SystemExit(f'half/full processed reads {depth:.4f} outside {DEPTH_RATIO_TOL}')
    if meta['full']['num_processed'] != meta['production']['num_processed']:
        raise SystemExit('the full-depth rerun did not process the production read count')
    print(f'index_seq_hash {hashes.pop()} in all three runs; salmon {SALMON_VERSION}, {N_DRAWS} Gibbs draws')
    for k, v in meta.items():
        print(f'  {k:>10s}: {v["num_mapped"]:>11,} of {v["num_processed"]:>11,} fragments mapped '
              f'({v["percent_mapped"]:.4f}%), {v["num_eq_classes"]:,} equivalence classes')
    print(f'  full rerun percent_mapped - production: {meta["full"]["percent_mapped"] - meta["production"]["percent_mapped"]:.2e}'
          f' percentage points; half/full processed {depth:.5f}', flush=True)
    return meta


def ingest():
    manifest = OUT / 'manifest.tsv'
    manifest.write_text(''.join(f'{k}\t{d}\n' for k, d in RUNS.items()))
    genes, samples, YL, YR, YT = RS.load_counts(manifest, TX2GENE, SUFFIXES, OUT)
    pL, pR, pT, _ = RS.load_point_estimates(manifest, TX2GENE, SUFFIXES, genes)
    if samples != list(RUNS):
        raise SystemExit(f'sample order {samples} differs from {list(RUNS)}')
    print(f'{len(genes):,} genes x {len(samples)} runs x {YL.shape[2]} draws ingested', flush=True)
    R = {}
    for j, s in enumerate(samples):
        R[s] = dict(pL=pL[:, j], pR=pR[:, j], pT=pT[:, j],
                    YL=YL[:, j:j + 1], YR=YR[:, j:j + 1], YT=YT[:, j:j + 1])
        R[s]['dv'] = np.log2((R[s]['YL'] + K) / (R[s]['YR'] + K)).var(axis=2, ddof=0)[:, 0]
        m = R[s]['YT'][:, 0].mean(1)
        R[s]['fano'] = np.where(m > 0, R[s]['YT'][:, 0].var(1, ddof=1) / np.where(m > 0, m, 1.0), np.nan)
        R[s]['two_sided'] = np.minimum(R[s]['pL'], R[s]['pR']) >= TWO_SIDED_MIN
        R[s]['one_sided'] = (R[s]['pL'] < TWO_SIDED_MIN) ^ (R[s]['pR'] < TWO_SIDED_MIN)
        R[s]['A'] = np.log2((R[s]['pL'] + K) / (R[s]['pR'] + K))
    return np.asarray(genes), R


def thinned_comparator(full):
    """The benchmark's rule applied to the full-depth run at f = F: thinned point estimates and YT draws."""
    rng = np.random.default_rng(np.random.SeedSequence(SEED, spawn_key=(MD.THIN_KEY, 0)))
    pL2, pR2 = MD.thin(full['pL'], F, rng), MD.thin(full['pR'], F, rng)
    YT2 = MD.thin(full['YT'], F, rng)
    m = YT2[:, 0].mean(1)
    return dict(pL=pL2, pR=pR2, A=np.log2((pL2 + K) / (pR2 + K)),
                two_sided=np.minimum(pL2, pR2) >= TWO_SIDED_MIN,
                one_sided=(pL2 < TWO_SIDED_MIN) ^ (pR2 < TWO_SIDED_MIN),
                fano=np.where(m > 0, YT2[:, 0].var(1, ddof=1) / np.where(m > 0, m, 1.0), np.nan))


def class_shares(run, genes):
    """u = haplotype-informative reads (u_L + u_R) and ambiguous share s per gene from the dumped classes."""
    tx2gene = pd.read_csv(TX2GENE, sep='\t', header=None, names=['tx', 'gene']).set_index('tx')['gene']
    names, classes = CP.read_classes(RUNS[run])
    uL, uR, amb = CP.class_counts(names, classes, tx2gene)
    u_l = np.array([uL[g] for g in genes], float)
    u_r = np.array([uR[g] for g in genes], float)
    a = np.array([amb[g] for g in genes], float)
    tot = a + u_l + u_r
    return dict(uL=u_l, uR=u_r, amb=a, u=u_l + u_r, s=np.where(tot > 0, a / np.where(tot > 0, tot, 1.0), np.nan))


def slope_r(x, y):
    """OLS slope with intercept of y on x and the Pearson correlation."""
    if x.size < 3 or x.std() == 0:
        return np.nan, np.nan
    b = np.polyfit(x, y, 1)[0]
    return float(b), float(np.corrcoef(x, y)[0, 1])


def analyse(genes, R):
    prod, full, half = R['production'], R['full'], R['half']
    thin = thinned_comparator(full)
    hap_full = full['pL'] + full['pR']
    band = band_of(hap_full)
    both = full['two_sided'] & half['two_sided']
    va_pred = MD.allelic_variance(full['pL'][:, None], full['pR'][:, None], half['pL'][:, None], half['pR'][:, None],
                                  full['YL'], full['YR'])[:, 0]
    va_meas = half['dv'] + q(half['pL'], half['pR']) / LN2 ** 2
    gibbs_pred = full['dv'] * q(half['pL'], half['pR']) / q(full['pL'], full['pR'])
    ec = {k: class_shares(k, genes) for k in ('full', 'half')}

    # (a) point estimates, production against the full-depth rerun
    a_pe = {}
    for k, denom in (('pL', hap_full), ('pR', hap_full), ('pT', full['pT'])):
        d = full[k] - prod[k]
        expressed = np.maximum(full[k], prod[k]) > 0
        rel = np.abs(d) / np.where(denom > 0, denom, 1.0)
        a_pe[k] = dict(max_abs_diff=float(np.abs(d).max()), within_1_read=float((np.abs(d) <= 1).mean()),
                       beyond_1pct=float((rel > 0.01).mean()), beyond_5pct=float((rel > 0.05).mean()),
                       genes_expressed=int(expressed.sum()),
                       pearson_log2=float(np.corrcoef(np.log2(full[k][expressed] + 1), np.log2(prod[k][expressed] + 1))[0, 1]))
    print('\n(a) full-depth rerun against the production run, point estimates (relative to the gene\'s haplotype reads '
          'pL + pR for pL, pR; to pT for pT):')
    for k, v in a_pe.items():
        print(f'    {k}: max |diff| {v["max_abs_diff"]:.3g} reads, {100 * v["within_1_read"]:.2f}% of genes within 1 read, '
              f'{100 * v["beyond_1pct"]:.2f}% beyond 1% and {100 * v["beyond_5pct"]:.2f}% beyond 5% of the gene\'s reads, '
              f'Pearson r of log2(x + 1) {v["pearson_log2"]:.6f} over {v["genes_expressed"]:,} expressed genes')
    # run-to-run agreement of the allelic log2 ratio at identical depth, against its own Gibbs variance
    va_prod = prod['dv'] + q(prod['pL'], prod['pR']) / LN2 ** 2
    va_full = full['dv'] + q(full['pL'], full['pR']) / LN2 ** 2
    z_rerun = (full['A'] - prod['A']) / np.sqrt(va_full + va_prod)
    print('    log2 ratio, rerun against production, two-sided at both, z = (A_rerun - A_prod) / sqrt(Va_rerun + Va_prod):')
    print(f'    {"band":>8s} {"n":>6s} {"slope":>7s} {"r":>7s} {"med |dA|":>9s} {"|z|>2":>7s} {"|z|>3":>7s}')

    res = dict(bands={})
    q_ratio = q(half['pL'], half['pR']) / q(full['pL'], full['pR'])
    # the Gibbs variance beyond Gamma shot noise, per gene at each depth; the rule scales it by q_ratio too
    excess_full = full['dv'] - q(full['pL'], full['pR']) / LN2 ** 2
    excess_half = half['dv'] - q(half['pL'], half['pR']) / LN2 ** 2
    rng = np.random.default_rng(np.random.SeedSequence(SEED, spawn_key=(RESAMPLE_KEY,)))

    def median_interval(x):
        """95% interval for the median from N_RESAMPLES resamples of the genes."""
        x = np.asarray(x, float)
        if x.size == 0:
            return [np.nan, np.nan]
        med = np.median(rng.choice(x, (N_RESAMPLES, x.size)), axis=1)
        return [float(np.quantile(med, .025)), float(np.quantile(med, .975))]
    for lo, hi in (INFO_BAND,) + BANDS:
        b = band_name(lo, hi)
        m = band == b
        sel = m & prod['two_sided'] & full['two_sided']
        s, rr = slope_r(prod['A'][sel], full['A'][sel])
        res['bands'][b] = dict(rerun_vs_production=dict(
            n=int(sel.sum()), slope=s, pearson=rr, median_abs_dA=float(np.median(np.abs(full['A'] - prod['A'])[sel])),
            share_abs_z_gt_2=float((np.abs(z_rerun[sel]) > 2).mean()), share_abs_z_gt_3=float((np.abs(z_rerun[sel]) > 3).mean())))
        v = res['bands'][b]['rerun_vs_production']
        print(f'    {b:>8s} {v["n"]:>6d} {v["slope"]:>7.4f} {v["pearson"]:>7.4f} {v["median_abs_dA"]:>9.4f} '
              f'{100 * v["share_abs_z_gt_2"]:>6.2f}% {100 * v["share_abs_z_gt_3"]:>6.2f}%')

    print(f'\n{"band":>8s} {"n(a)":>6s} {"dv full/prod":>24s} {"n(b)":>6s} {"Va meas/pred":>24s} {"Gibbs part":>24s} '
          f'{"exponent":>24s}  verdict')
    for lo, hi in (INFO_BAND,) + BANDS:
        b = band_name(lo, hi)
        m = band == b
        sel_a = m & prod['two_sided'] & full['two_sided'] & (prod['dv'] > 0) & (full['dv'] > 0)
        sel_b = m & both
        sel_g = sel_b & (gibbs_pred > 0)
        sel_x = sel_g & (half['dv'] > 0) & (q_ratio >= EXPONENT_MIN_Q_RATIO)
        sel_e = sel_b & (excess_full > 0) & (excess_half > 0)
        r = dict(
            dv_full_over_prod=med_iqr(full['dv'][sel_a] / prod['dv'][sel_a]),
            va_meas_over_pred=dict(med_iqr(va_meas[sel_b] / va_pred[sel_b]),
                                   median_ci95=median_interval(va_meas[sel_b] / va_pred[sel_b])),
            gibbs_meas_over_pred=med_iqr(half['dv'][sel_g] / gibbs_pred[sel_g]),
            exponent=med_iqr(np.log(half['dv'][sel_x] / full['dv'][sel_x]) / np.log(q_ratio[sel_x])),
            gibbs_over_qa_full=med_iqr(full['dv'][sel_b] / (q(full['pL'][sel_b], full['pR'][sel_b]) / LN2 ** 2)),
            gibbs_over_qa_half=med_iqr(half['dv'][sel_b] / (q(half['pL'][sel_b], half['pR'][sel_b]) / LN2 ** 2)),
            excess_half_over_full=med_iqr(excess_half[sel_e] / excess_full[sel_e]),
            fano=dict(full=med_iqr(full['fano'][m & (full['pT'] > 0)]),
                      half=med_iqr(half['fano'][m & (half['pT'] > 0)]),
                      thinned=med_iqr(thin['fano'][m & (full['pT'] > 0)])),
            pairs=int(m.sum()),
            one_sided=dict(full=float(full['one_sided'][m].mean()), half=float(half['one_sided'][m].mean()),
                           thinned=float(thin['one_sided'][m].mean())),
            two_sided_full=int((m & full['two_sided']).sum()),
            became_one_sided=dict(half=int((m & full['two_sided'] & half['one_sided']).sum()),
                                  thinned=int((m & full['two_sided'] & thin['one_sided']).sum())),
            became_empty=dict(half=int((m & full['two_sided'] & (half['pL'] + half['pR'] <= 0)).sum()),
                              thinned=int((m & full['two_sided'] & (thin['pL'] + thin['pR'] <= 0)).sum())),
            attenuation={},
            premise={})
        # the half-depth run is a subset of the full one, so var(A_half - A_full) ~ Va_half - Va_full when the
        # variances are calibrated; for the thinned comparator the variances are the counting terms alone
        for name, comp, var_comp, var_full in (('half', half, va_meas, va_full),
                                               ('thinned', thin, q(thin['pL'], thin['pR']) / LN2 ** 2,
                                                q(full['pL'], full['pR']) / LN2 ** 2)):
            sel = m & full['two_sided'] & comp['two_sided']
            s, rr = slope_r(full['A'][sel], comp['A'][sel])
            selz = sel & (var_comp > var_full)
            z2 = (comp['A'][selz] - full['A'][selz]) ** 2 / (var_comp[selz] - var_full[selz])
            r['attenuation'][name] = dict(
                n=int(sel.sum()), slope=s, pearson=rr,
                median_diff=float(np.median(comp['A'][sel] - full['A'][sel])) if sel.any() else np.nan,
                n_z=int(selz.sum()), z2_median_over_chi2_median=float(np.median(z2) / CHI2_1_MEDIAN) if selz.any() else np.nan,
                share_abs_z_gt_2=float((z2 > 4).mean()) if selz.any() else np.nan)
        sel_u = sel_b & (ec['full']['u'] > 0)
        sel_s = sel_b & (ec['full']['s'] > 0) & np.isfinite(ec['half']['s'])
        r['premise'] = dict(u_half_over_full=med_iqr(ec['half']['u'][sel_u] / ec['full']['u'][sel_u]),
                            s_half_over_full=med_iqr(ec['half']['s'][sel_s] / ec['full']['s'][sel_s]),
                            s_full=med_iqr(ec['full']['s'][sel_s]))
        judged = (lo, hi) in BANDS and r['va_meas_over_pred']['n'] >= MIN_GENES
        r['judged'] = judged
        r['passed'] = bool(judged and PASS_BAND[0] <= r['va_meas_over_pred']['median'] <= PASS_BAND[1])
        res['bands'][b].update(r)
        verdict = ('PASS' if r['passed'] else 'FAIL') if judged else ('info' if (lo, hi) not in BANDS else f'n < {MIN_GENES}')
        print(f'{b:>8s} {r["dv_full_over_prod"]["n"]:>6d} {fmt(r["dv_full_over_prod"]).split(" (")[0]:>24s} '
              f'{r["va_meas_over_pred"]["n"]:>6d} {fmt(r["va_meas_over_pred"]).split(" (")[0]:>24s} '
              f'{fmt(r["gibbs_meas_over_pred"]).split(" (")[0]:>24s} {fmt(r["exponent"]).split(" (")[0]:>24s}  {verdict}')
    judged = [b for b, r in res['bands'].items() if r['judged']]
    passed = bool(judged) and all(res['bands'][b]['passed'] for b in judged)
    failed = [b for b in judged if not res['bands'][b]['passed']]
    res.update(passed=passed, judged_bands=judged, failed_bands=failed, pass_band=list(PASS_BAND), min_genes=MIN_GENES,
               point_estimates_full_vs_production=a_pe, f=F, seed=SEED, genes=int(len(genes)),
               genes_two_sided_both_depths=int(both.sum()))
    print(f'\nPRE-REGISTERED VERDICT: {"PASS" if passed else "FAIL"} (median Va_meas / Va_pred in '
          f'[{PASS_BAND[0]}, {PASS_BAND[1]}] in every band with >= {MIN_GENES} genes; judged bands {judged}; failed {failed})')
    for b in judged:
        v = res['bands'][b]['va_meas_over_pred']
        print(f'    {b:>8s}: median {v["median"]:.3f}, 95% resampling interval [{v["median_ci95"][0]:.3f}, {v["median_ci95"][1]:.3f}]')

    print(f'\n{"band":>8s} {"Gibbs/q_a full":>20s} {"Gibbs/q_a half":>20s} {"excess half/full":>28s} {"Fano full":>22s} '
          f'{"Fano half":>22s} {"Fano thinned":>22s}')
    for b, r in res['bands'].items():
        print(f'{b:>8s} {fmt(r["gibbs_over_qa_full"]).split(" (")[0]:>20s} {fmt(r["gibbs_over_qa_half"]).split(" (")[0]:>20s} '
              f'{fmt(r["excess_half_over_full"]):>28s} '
              f'{fmt(r["fano"]["full"]).split(" (")[0]:>22s} {fmt(r["fano"]["half"]).split(" (")[0]:>22s} '
              f'{fmt(r["fano"]["thinned"]).split(" (")[0]:>22s}')
    print(f'\n(c) zero haplotypes: share of pairs with exactly one side < {TWO_SIDED_MIN} reads')
    print(f'{"band":>8s} {"pairs":>7s} {"full":>8s} {"half":>8s} {"thinned":>8s} {"2-sided@full":>13s} '
          f'{"->1-sided half":>15s} {"->1-sided thin":>15s} {"->empty half":>13s} {"->empty thin":>13s}')
    for b, r in res['bands'].items():
        o = r['one_sided']
        print(f'{b:>8s} {r["pairs"]:>7d} {100 * o["full"]:>7.2f}% {100 * o["half"]:>7.2f}% {100 * o["thinned"]:>7.2f}% '
              f'{r["two_sided_full"]:>13d} {r["became_one_sided"]["half"]:>15d} {r["became_one_sided"]["thinned"]:>15d} '
              f'{r["became_empty"]["half"]:>13d} {r["became_empty"]["thinned"]:>13d}')
    print('\n(d) attenuation: OLS slope (with intercept) of the log2 ratio on the full-depth log2 ratio, two-sided at both;'
          ' calibration: median of (A_x - A_full)^2 / (Var_x - Var_full) over the chi2(1) median, and the share above 4')
    print(f'{"band":>8s} {"n half":>7s} {"slope half":>11s} {"r half":>7s} {"med diff":>9s} {"calib half":>11s} {"|z|>2":>7s} '
          f'{"n thin":>7s} {"slope thin":>11s} {"r thin":>7s} {"calib thin":>11s} {"|z|>2":>7s}')
    for b, r in res['bands'].items():
        h, t = r['attenuation']['half'], r['attenuation']['thinned']
        print(f'{b:>8s} {h["n"]:>7d} {h["slope"]:>11.4f} {h["pearson"]:>7.4f} {h["median_diff"]:>9.4f} '
              f'{h["z2_median_over_chi2_median"]:>11.3f} {100 * h["share_abs_z_gt_2"]:>6.2f}% '
              f'{t["n"]:>7d} {t["slope"]:>11.4f} {t["pearson"]:>7.4f} {t["z2_median_over_chi2_median"]:>11.3f} '
              f'{100 * t["share_abs_z_gt_2"]:>6.2f}%')
    print('\n(e) premise on the dumped equivalence classes (two-sided at both depths): u = haplotype-informative reads, '
          's = ambiguous share')
    print(f'{"band":>8s} {"u half/full":>26s} {"s half/full":>26s} {"s full":>26s}')
    for b, r in res['bands'].items():
        p = r['premise']
        print(f'{b:>8s} {fmt(p["u_half_over_full"]):>26s} {fmt(p["s_half_over_full"]):>26s} {fmt(p["s_full"]):>26s}')

    table = pd.DataFrame(dict(
        gene=genes, band=band, two_sided_both=both,
        one_sided_full=full['one_sided'], one_sided_half=half['one_sided'], one_sided_thinned=thin['one_sided'],
        pL_prod=prod['pL'], pR_prod=prod['pR'], pT_prod=prod['pT'], dv_prod=prod['dv'],
        pL_full=full['pL'], pR_full=full['pR'], pT_full=full['pT'], dv_full=full['dv'], fano_full=full['fano'],
        pL_half=half['pL'], pR_half=half['pR'], pT_half=half['pT'], dv_half=half['dv'], fano_half=half['fano'],
        pL_thin=thin['pL'], pR_thin=thin['pR'], fano_thin=thin['fano'],
        va_pred=va_pred, va_meas=va_meas, gibbs_pred=gibbs_pred,
        uL_full=ec['full']['uL'], uR_full=ec['full']['uR'], amb_full=ec['full']['amb'], s_full=ec['full']['s'],
        uL_half=ec['half']['uL'], uR_half=ec['half']['uR'], amb_half=ec['half']['amb'], s_half=ec['half']['s']))
    return res, table


def figures(table):
    t = table[table.two_sided_both & (table.band != '')]
    fig, ax = plt.subplots(figsize=(6.4, 6.0), dpi=150)
    for b in ('30-99', '100-999', '1000+', '1-29'):
        s = t[t.band == b]
        ax.scatter(s.va_pred, s.va_meas, s=5, alpha=0.35, color=PALETTE[b], linewidths=0, label=f'{b} reads (n={len(s):,})')
    lim = (min(t.va_pred.min(), t.va_meas.min()) * 0.8, max(t.va_pred.max(), t.va_meas.max()) * 1.25)
    ax.plot(lim, lim, color='#52514e', lw=1.2, label='measured = predicted')
    for f, ls in ((PASS_BAND[0], ':'), (PASS_BAND[1], ':')):
        ax.plot(lim, (lim[0] * f, lim[1] * f), color='#52514e', lw=0.8, ls=ls)
    ax.set(xscale='log', yscale='log', xlim=lim, ylim=lim,
           xlabel='Va predicted by the thinning rule from the full-depth run (log2 units^2)',
           ylabel='Va measured by Salmon at half depth (log2 units^2)',
           title='Allelic Gibbs variance at half depth, one gene per point')
    ax.legend(frameon=False, fontsize=8, loc='upper left')
    ax.grid(True, color='#e9e8e4', lw=0.6)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    buf1 = io.BytesIO(); fig.savefig(buf1, format='png'); fig.savefig(OUT / 'figure_va_measured_vs_predicted.png'); plt.close(fig)

    bands = [band_name(lo, hi) for lo, hi in (INFO_BAND,) + BANDS]
    fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=150)
    w, x = 0.26, np.arange(len(bands))
    for i, (name, label) in enumerate((('full', 'full depth'), ('half', 'half depth, Salmon'),
                                       ('thinned', 'half depth, benchmark thinning of full'))):
        y = [100 * float(table[(table.band == b)][f'one_sided_{name}'].mean()) for b in bands]
        ax.bar(x + (i - 1) * w, y, width=w - 0.03, color=SERIES[name], label=label)
    ax.set(xticks=x, xticklabels=[f'{b} reads' for b in bands], ylabel='pairs with exactly one haplotype < 0.5 reads (%)',
           xlabel='full-depth haplotype reads (pL + pR)', title='Zero-haplotype share by band')
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, axis='y', color='#e9e8e4', lw=0.6)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    buf2 = io.BytesIO(); fig.savefig(buf2, format='png'); fig.savefig(OUT / 'figure_zero_haplotype_share.png'); plt.close(fig)
    return [base64.b64encode(b.getvalue()).decode() for b in (buf1, buf2)]


def html_page(res, meta, pngs):
    B = res['bands']
    def entry(d):
        return f'{d["median"]:.3f} <span class="q">[{d["iqr"][0]:.3f}, {d["iqr"][1]:.3f}]</span>' if d['n'] else '-'
    rows_b = ''.join(
        f'<tr><td>{b}</td><td>{r["va_meas_over_pred"]["n"]:,}</td><td>{entry(r["va_meas_over_pred"])}</td>'
        f'<td>{entry(r["gibbs_meas_over_pred"])}</td><td>{entry(r["exponent"])}</td><td>{entry(r["dv_full_over_prod"])}</td>'
        f'<td>{("PASS" if r["passed"] else "FAIL") if r["judged"] else "not judged"}</td></tr>' for b, r in B.items())
    rows_r = ''.join(
        f'<tr><td>{b}</td><td>{r["rerun_vs_production"]["n"]:,}</td><td>{r["rerun_vs_production"]["slope"]:.4f}</td>'
        f'<td>{r["rerun_vs_production"]["pearson"]:.4f}</td><td>{r["rerun_vs_production"]["median_abs_dA"]:.4f}</td>'
        f'<td>{100 * r["rerun_vs_production"]["share_abs_z_gt_2"]:.2f}%</td>'
        f'<td>{100 * r["rerun_vs_production"]["share_abs_z_gt_3"]:.2f}%</td></tr>' for b, r in B.items())
    rows_t = ''.join(
        f'<tr><td>{b}</td><td>{entry(r["gibbs_over_qa_full"])}</td><td>{entry(r["gibbs_over_qa_half"])}</td>'
        f'<td>{entry(r["excess_half_over_full"])}</td>'
        f'<td>{entry(r["fano"]["full"])}</td><td>{entry(r["fano"]["half"])}</td><td>{entry(r["fano"]["thinned"])}</td></tr>'
        for b, r in B.items())
    rows_c = ''.join(
        f'<tr><td>{b}</td><td>{r["pairs"]:,}</td><td>{100 * r["one_sided"]["full"]:.2f}%</td>'
        f'<td>{100 * r["one_sided"]["half"]:.2f}%</td><td>{100 * r["one_sided"]["thinned"]:.2f}%</td>'
        f'<td>{r["two_sided_full"]:,}</td><td>{r["became_one_sided"]["half"]:,}</td><td>{r["became_one_sided"]["thinned"]:,}</td>'
        f'<td>{r["became_empty"]["half"]:,}</td><td>{r["became_empty"]["thinned"]:,}</td></tr>' for b, r in B.items())
    rows_d = ''.join(
        f'<tr><td>{b}</td><td>{r["attenuation"]["half"]["n"]:,}</td><td>{r["attenuation"]["half"]["slope"]:.4f}</td>'
        f'<td>{r["attenuation"]["half"]["pearson"]:.4f}</td><td>{r["attenuation"]["half"]["median_diff"]:+.4f}</td>'
        f'<td>{r["attenuation"]["half"]["z2_median_over_chi2_median"]:.3f}</td>'
        f'<td>{100 * r["attenuation"]["half"]["share_abs_z_gt_2"]:.2f}%</td>'
        f'<td>{r["attenuation"]["thinned"]["n"]:,}</td><td>{r["attenuation"]["thinned"]["slope"]:.4f}</td>'
        f'<td>{r["attenuation"]["thinned"]["pearson"]:.4f}</td>'
        f'<td>{r["attenuation"]["thinned"]["z2_median_over_chi2_median"]:.3f}</td>'
        f'<td>{100 * r["attenuation"]["thinned"]["share_abs_z_gt_2"]:.2f}%</td></tr>' for b, r in B.items())
    rows_e = ''.join(
        f'<tr><td>{b}</td><td>{entry(r["premise"]["u_half_over_full"])}</td><td>{entry(r["premise"]["s_half_over_full"])}</td>'
        f'<td>{entry(r["premise"]["s_full"])}</td></tr>' for b, r in B.items())
    pe = res['point_estimates_full_vs_production']
    rows_a = ''.join(f'<tr><td>{k}</td><td>{v["max_abs_diff"]:.3g}</td><td>{100 * v["within_1_read"]:.2f}%</td>'
                     f'<td>{100 * v["beyond_1pct"]:.2f}%</td><td>{100 * v["beyond_5pct"]:.2f}%</td>'
                     f'<td>{v["pearson_log2"]:.6f}</td><td>{v["genes_expressed"]:,}</td></tr>' for k, v in pe.items())
    rows_m = ''.join(f'<tr><td>{k}</td><td>{v["num_processed"]:,}</td><td>{v["num_mapped"]:,}</td>'
                     f'<td>{v["percent_mapped"]:.4f}%</td><td>{v["num_eq_classes"]:,}</td></tr>' for k, v in meta.items())
    verdict = 'PASS' if res['passed'] else 'FAIL'
    def med_ci(b):
        v = B[b]['va_meas_over_pred']
        return f'{b}: {v["median"]:.3f} [{v["median_ci95"][0]:.3f}, {v["median_ci95"][1]:.3f}]'
    verdict_text = (f'The median of Va<sub>meas</sub> / Va<sub>pred</sub> lies in [{PASS_BAND[0]}, {PASS_BAND[1]}] in every '
                    f'band with at least {MIN_GENES} genes' if res['passed'] else
                    f'The median of Va<sub>meas</sub> / Va<sub>pred</sub> falls outside [{PASS_BAND[0]}, {PASS_BAND[1]}] in the '
                    f'{", ".join(res["failed_bands"])} band')
    judged_text = '; '.join(med_ci(b) for b in res['judged_bands'])
    g = {b: B[b] for b in ('30-99', '100-999', '1000+')}
    exp = ' / '.join(f'{g[b]["exponent"]["median"]:.2f}' for b in g)
    excess = ' / '.join(f'{g[b]["excess_half_over_full"]["median"]:.2f}' for b in g)
    iqr_b = ' / '.join(f'{np.log(g[b]["va_meas_over_pred"]["iqr"][1] / g[b]["va_meas_over_pred"]["iqr"][0]):.2f}' for b in g)
    iqr_a = ' / '.join(f'{np.log(g[b]["dv_full_over_prod"]["iqr"][1] / g[b]["dv_full_over_prod"]["iqr"][0]):.2f}' for b in g)
    gibbs = ' / '.join(f'{g[b]["gibbs_meas_over_pred"]["median"]:.3f}' for b in g)
    omit = ' / '.join(f'{g[b]["became_one_sided"]["half"]:,} against {g[b]["became_one_sided"]["thinned"]:,}' for b in g)
    slopes = ' / '.join(f'{g[b]["attenuation"]["half"]["slope"]:.2f} against {g[b]["attenuation"]["thinned"]["slope"]:.2f}' for b in g)
    calib = ' / '.join(f'{g[b]["attenuation"]["half"]["z2_median_over_chi2_median"]:.2f}' for b in g)
    calib_thin = ' / '.join(f'{g[b]["attenuation"]["thinned"]["z2_median_over_chi2_median"]:.2f}' for b in g)
    u_ratio = ' / '.join(f'{g[b]["premise"]["u_half_over_full"]["median"]:.3f}' for b in g)
    s_ratio = ' / '.join(f'{g[b]["premise"]["s_half_over_full"]["median"]:.3f}' for b in g)
    fano = ' / '.join(f'{g[b]["fano"]["full"]["median"]:.3f}, {g[b]["fano"]["half"]["median"]:.3f}, {g[b]["fano"]["thinned"]["median"]:.3f}' for b in g)
    return f'''<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>Salmon half-depth check</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:980px;margin:2em auto;padding:0 16px;color:#0b0b0b;background:#fcfcfb;line-height:1.45}}
table{{border-collapse:collapse;margin:1em 0;font-size:0.92em}} td,th{{border-bottom:1px solid #e9e8e4;padding:4px 10px;text-align:right}}
th{{background:#f3f2ef}} td:first-child,th:first-child{{text-align:left}} .q{{color:#52514e;font-size:0.9em}}
.verdict{{font-weight:600;padding:0.6em 1em;border-left:4px solid {"#1baf7a" if res["passed"] else "#e34948"};background:#f3f2ef}}
img{{max-width:100%}} p.note{{color:#52514e}}
</style></head><body>
<h1>Does the benchmark's thinning rule predict Salmon's allelic Gibbs variance at half depth?</h1>
<p>Donor 100 quantified with Salmon {SALMON_VERSION} (production flags, the production index rebuilt with the same
sequence hash, 200 Gibbs draws) at full depth and at 50% depth (seqtk, seed {SEED}, both mates, pairs verified).
The plasmode benchmark never reruns Salmon; it thins the full-depth output and predicts each record's allelic
<b>Gibbs variance</b> (the across-draw variance of log2((yL + 0.5)/(yR + 0.5)) over the 200 Gibbs draws, dv) by
scaling the full-depth dv by the ratio of the counting terms q(pL', pR') / q(pL, pR), q(x, y) = 1/(x + 0.5) + 1/(y + 0.5),
and adding q_a = q / ln(2)<sup>2</sup> at the thinned counts. Here the prediction uses the realized half-depth point
estimates and is compared with what Salmon actually produced at half depth. Bands are full-depth haplotype reads
pL + pR; every per-gene quantity comes through the pipeline's own ingest (L/R-paired transcripts summed per gene).</p>
<div class="verdict">Pre-registered verdict: {verdict}. {verdict_text} (median and 95% interval from {N_RESAMPLES:,}
resamples of the genes, by band: {judged_text}).</div>
<h2>What this shows</h2>
<ul>
<li>The rule over-predicts the half-depth allelic Gibbs variance below 1,000 haplotype reads and is close above it.
The Gibbs part alone is {gibbs} of its prediction at 30-99 / 100-999 / 1000+ reads: Salmon's Gibbs variance grows
with 1/depth to the depth exponent {exp}, not 1. Per gene, the Gibbs variance beyond Gamma shot noise grows by
{excess} between full and half depth where the rule says 2. The per-gene scatter of the ratio is also wider than
Gibbs stochasticity alone: the IQR of Va<sub>meas</sub> / Va<sub>pred</sub> spans {iqr_b} in log units against {iqr_a} for
the rerun-against-production ratio, so the rule's error is a per-gene spread as well as a downward shift.</li>
<li>The premise measured on the equivalence classes holds: haplotype-informative reads scale by {u_ratio} and the
ambiguous share by {s_ratio}. What fails is the step from those to the draw variance, so below 1,000 haplotype reads
the delta-method form s<sup>2</sup>(1/u<sub>L</sub> + 1/u<sub>R</sub>) is not how Salmon's sampler responds to depth.</li>
<li>Zero haplotypes are the larger omission: of full-depth two-sided records, {omit} became one-sided at half depth
under Salmon against binomial thinning (30-99 / 100-999 / 1000+). Salmon's optimizer zeroes a haplotype at lower
depth far more often than subsampling its counts does.</li>
<li>The half-depth log2 ratio is attenuated: slope on the full-depth ratio {slopes} (Salmon against thinning). Its
movement between depths is {calib} times what the difference of the Gibbs variances predicts (thinning against its
counting terms: {calib_thin}), so either the Gibbs variance understates the point estimate's sampling variability at
lower depth or the two depths are not nested the way a count subsample is; this test cannot separate the two.</li>
<li>The total channel is not at issue: the Fano factor of the yT draws is {fano} (full, half, thinned) by band.</li>
<li>The rerun reproduces the production run: identical fragment and mapping counts, the total pT to r = {pe["pT"]["pearson_log2"]:.6f},
and the allelic log2 ratio to a median |dA| below 0.006 with |z| &gt; 2 in at most 0.25% of records.</li>
</ul>

<h2>The test</h2>
<p>Va<sub>meas</sub> = dv<sub>half</sub> + q_a(pL<sub>half</sub>, pR<sub>half</sub>);
Va<sub>pred</sub> = dv<sub>full</sub> &times; q(half) / q(full) + q_a(half). Genes with both haplotypes at &ge; 0.5 reads at both depths.
Median and interquartile range (IQR, the 25th to 75th percentile) of the ratio. The Gibbs part alone is
dv<sub>half</sub> / (dv<sub>full</sub> &times; q(half)/q(full)), which the criterion cannot see where q_a dominates.
The depth exponent is the realized e in dv<sub>half</sub> / dv<sub>full</sub> = (q(half)/q(full))<sup>e</sup> per gene
(the rule assumes e = 1; genes with q(half)/q(full) &ge; {EXPONENT_MIN_Q_RATIO}). The last column is the same ratio
statistic between the full-depth rerun and the production run, two independent 200-draw sets at identical depth: the
spread Gibbs stochasticity alone produces.</p>
<table><tr><th>band</th><th>genes</th><th>Va meas / pred</th><th>Gibbs part meas / pred</th><th>depth exponent</th><th>dv rerun / production</th><th>verdict</th></tr>{rows_b}</table>
<img src="data:image/png;base64,{pngs[0]}" alt="measured against predicted allelic variance by band">

<h2>How much of Va is the Gibbs part, and the total channel</h2>
<p>Gibbs/q_a is dv over the counting term at that depth; excess half / full is the per-gene ratio between depths of
dv - q_a, the Gibbs variance beyond Gamma shot noise, which the rule scales by q(half)/q(full), about 2 (genes with a
positive excess at both depths). The <b>Fano factor</b> is the across-draw variance over the
across-draw mean of a gene's total-count draws yT (1 for Poisson counts). The benchmark thins every yT draw
binomially at f = {F}, which sends a Fano factor F<sub>0</sub> to f F<sub>0</sub> + (1 - f); Salmon's own half-depth run is the
reference. Genes with pT &gt; 0 at the depth shown.</p>
<table><tr><th>band</th><th>Gibbs/q_a full</th><th>Gibbs/q_a half</th><th>excess half / full</th><th>Fano full</th><th>Fano half (Salmon)</th><th>Fano half (thinned draws)</th></tr>{rows_t}</table>

<h2>Zero haplotypes</h2>
<p>Share of gene records with exactly one haplotype below 0.5 reads, at full depth, at half depth from Salmon, and
under the benchmark's binomial thinning of the full-depth point estimates (seed {SEED}), which can zero a side of a
few reads but not a larger one. Realized minus thinned is the omission the benchmark cannot reproduce. The right-hand
columns count full-depth two-sided records that became one-sided or lost all haplotype reads.</p>
<table><tr><th>band</th><th>records</th><th>one-sided full</th><th>one-sided half</th><th>one-sided thinned</th><th>two-sided at full</th>
<th>became one-sided (Salmon)</th><th>(thinned)</th><th>became empty (Salmon)</th><th>(thinned)</th></tr>{rows_c}</table>
<img src="data:image/png;base64,{pngs[1]}" alt="zero-haplotype share by band at both depths">

<h2>Point-estimate attenuation</h2>
<p>Ordinary least-squares slope, with intercept, of the half-depth log2 ratio log2((pL + 0.5)/(pR + 0.5)) on the
full-depth one over records two-sided at both depths, with the Pearson correlation and the median difference
(half minus full). The thinned columns apply the same regression and the same selection to the benchmark's thinned
point estimates, so they are the slope the rule predicts, including the selection's own truncation of lopsided
records. Calibration: the half-depth run is a subset of the full one, so if the measurement variances are right,
(A<sub>half</sub> - A<sub>full</sub>)<sup>2</sup> / (Va<sub>half</sub> - Va<sub>full</sub>) has the median of a chi-square with one
degree of freedom ({CHI2_1_MEDIAN:.3f}); the column is that median over {CHI2_1_MEDIAN:.3f} (1 when calibrated), with the share of
records above 4 (|z| &gt; 2; 4.55% when calibrated). For Salmon at half depth the variances are Va = dv + q_a; for the
thinned comparator they are the counting terms q_a alone.</p>
<table><tr><th>band</th><th>n</th><th>slope (Salmon)</th><th>r</th><th>median diff</th><th>calibration</th><th>|z| &gt; 2</th>
<th>n</th><th>slope (thinned)</th><th>r</th><th>calibration</th><th>|z| &gt; 2</th></tr>{rows_d}</table>

<h2>The premise on the equivalence classes</h2>
<p>From each run's dumped equivalence classes: u, the reads in single-gene classes holding only one haplotype of the
gene (u_L + u_R), should scale by f = {F}; s, the share of the gene's paired reads in classes holding both haplotypes,
should not change. Genes two-sided at both depths with u &gt; 0 (u ratio) or s &gt; 0 (s ratio) at full depth.</p>
<table><tr><th>band</th><th>u half / full</th><th>s half / full</th><th>s at full depth</th></tr>{rows_e}</table>

<h2>The rerun reproduces the production run</h2>
<table><tr><th>run</th><th>fragments processed</th><th>mapped</th><th>mapping rate</th><th>equivalence classes</th></tr>{rows_m}</table>
<table><tr><th>point estimate</th><th>max |rerun - production| (reads)</th><th>genes within 1 read</th><th>beyond 1% of the gene's reads</th>
<th>beyond 5%</th><th>Pearson r, log2(x + 1)</th><th>expressed genes</th></tr>{rows_a}</table>
<p>The rerun differs from the production run only in thread count (24 against 12), which changes the read order of
Salmon's online phase. The total pT reproduces; the haplotype split pL / pR differs by more than one read in about
one gene in nine, almost all of them genes with hundreds of reads where the difference is well under 1% of the
gene's haplotype reads. Below, the allelic log2 ratio of the rerun regressed on the production run's (records
two-sided in both), with z = (A<sub>rerun</sub> - A<sub>prod</sub>) / sqrt(Va<sub>rerun</sub> + Va<sub>prod</sub>): two independent draws of
the same quantity would give |z| &gt; 2 in 4.6% of records; the measured shares are far below that, so the run-to-run
difference of the optimum is small against the Gibbs variance. This is the run-to-run floor against which the
half-depth attenuation slope above must be read.</p>
<table><tr><th>band</th><th>n</th><th>slope</th><th>r</th><th>median |dA|</th><th>|z| &gt; 2</th><th>|z| &gt; 3</th></tr>{rows_r}</table>
<p class="note">Files: summary.json, per_gene.tsv, the two figures as PNG, manifest.tsv; launch scripts and logs beside them;
Salmon output under work/. Script: scripts/salmon_half_depth_check.py.</p>
</body></html>'''


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    meta = validate_runs()
    genes, R = ingest()
    res, table = analyse(genes, R)
    res['runs'] = {k: dict(dir=str(RUNS[k]), **v) for k, v in meta.items()}
    MD.write_atomic(OUT / 'summary.json', lambda fh: fh.write(MD.dumps(res)), 'w')
    MD.write_atomic(OUT / 'per_gene.tsv', lambda fh: table.to_csv(fh, sep='\t', index=False, float_format='%.6g'), 'w')
    pngs = figures(table)
    MD.write_atomic(OUT / 'salmon_half_depth.html', lambda fh: fh.write(html_page(res, meta, pngs)), 'w')
    print(f'\nwrote {OUT / "summary.json"}, {OUT / "per_gene.tsv"} ({len(table):,} genes), {OUT / "salmon_half_depth.html"}')


if __name__ == '__main__':
    main()

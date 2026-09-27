"""Known-answer checks of the plasmode generator (make_datasets.py).

Exits non-zero if any check fails. Inputs are make_datasets' own: the 100
genes of corrected_null_store_20260925, loaded once.

(a) IDENTITY. With every thinning factor 1, perm the identity and no swap,
    the generator's A, T, Va, Vt equal summaries_from_point_estimates on the
    cache inputs exactly (np.array_equal). Then with factors 1 but dataset
    0's real perm and swap: T and Vt equal the real ones moved by perm
    exactly, A equals swap x real A moved by perm and Va the real Va moved by
    perm to within A_TOL and VA_RTOL (swapping L and R turns log2(x/y) into
    log2(y/x), which rounds differently), so the library size travels with
    its record.

(b) THINNING KEEPS SALMON'S DRAW BEHAVIOUR. Every gene thinned uniformly by
    F_B, no permutation. By band of haplotype-informative reads pL + pR
    (after thinning for the thinned data, as observed for the real data):
    median Fano factor of the YT Gibbs draws (across-draw variance, ddof=1,
    over mean, over pairs with a positive mean). PASS if, in every band with
    >= MIN_PAIRS thinned pairs, the thinned Fano median is in FANO_BAND, AND
    the allelic rule check below passes.
    ALLELIC RULE CHECK, an arithmetic guard that make_datasets applies the
    rule its docstring states: for every informative record (pL + pR > 0
    before and after thinning), (Va' - q_a') / (Va - q_a) equals
    q(pL', pR') / q(pL, pR) within RULE_RTOL (relative), with
    q(x, y) = 1/(x+0.5) + 1/(y+0.5) and q_a = q / ln(2)^2 computed here; and
    Va' is exactly 0 wherever pL' + pR' = 0.
    FOR INFORMATION ONLY, no pass rule: median Va / q_a over pairs with both
    haplotypes at or above 0.5 reads, the same with the counting term taken
    out ((Va - q_a) / q_a, the Gibbs-only part), and the share of pairs with
    exactly one haplotype below 0.5 reads. A band of thinned records holds
    different real records than the same band of real data, whose Va / q_a
    differs by gene composition and heterozygosity, so these are not a test
    of the rule.

(c) INJECTED EFFECT IS RECOVERED. |beta| = C_BETA, C_N datasets from the
    production streams with no null genes. Per gene, the allelic slope at the
    causal variant is null_permutation_instrument.fit_channels(...)['ba']
    (through-origin weighted least squares of A on s = xL - xR over the
    admitted donors), over the records make_datasets.allelic_kept admits
    (the arms' admission) unless stated. Seven estimates, each the mean of
    slope / truth over gene-dataset units:
      1/Va' no drop, vs beta   the quantity the earlier generator's check
                               reported (PREV_RECOVERY), over Va' > EPS only
      1/Va', vs beta           the arms' weights and admission
      1/Va_exp, vs beta        weights from Va evaluated at the EXPECTED
                               thinned counts f pL, f pR instead of the
                               realized pL', pR' (same records)
      1/Va_real, vs beta       weights from the record's unthinned Va, which
                               does not depend on s (same records)
      unit, vs beta            weights 1 (same records)
      unit, vs pipeline truth  the PASS RULE
      1/Va', vs pipeline truth
    Measured 2026-09-26 (>= 100 reads, gene-clustered se in brackets):
    1/Va_real 1.007 (0.012), 1/Va_exp 0.957 (0.016), 1/Va' 0.954 (0.016),
    unit 1.011 (0.018). Weights that do not depend on s recover beta; the
    shortfall of 1/Va' comes almost entirely from Va depending on which
    haplotype the effect thinned (1/Va_real -> 1/Va_exp, -0.050), and only
    0.003 from the realized binomial draw (1/Va_exp -> 1/Va'). This is what
    the rule is built to do (Salmon's Gibbs variance scales with 1/u, check
    salmon_premise), so the attenuation is a property of 1/v weighting of the
    log ratio when v follows the counts, not a generator defect. Inferred,
    not re-measured (that generator no longer exists): the earlier generator
    read 0.992 because thinning each Gibbs draw independently left the Gibbs
    part of Va nearly unchanged, so its weights barely depended on the
    thinning. One explanation consistent with these
    numbers, not tested separately: a heterozygous donor's weight falls most
    when the thinned haplotype is already its smaller side, which is when its
    A' lies furthest in the effect's direction.
    PASS if, among genes whose real median pL + pR over donors is
    >= C_MIN_READS, the unit-weighted slope over the pipeline-scale allelic
    truth (make_datasets section 5) is within C_SE_MULT gene-clustered
    standard errors of 1 (gene-clustered se = sd of the per-gene means /
    sqrt(genes)). Unit weights carry no weight-response coupling, and the
    pipeline-scale truth carries the transform's attenuation, so at >= 100
    reads the ratio tests the injection itself (measured 1.016, se 0.019).
    Below 100 reads it is 0.83 (10-99 reads, se 0.087) and is not decomposed
    here; one untested candidate is the zero-haplotype drop, which conditions
    on the thinned outcome (a record whose thinned side fell below 0.5 reads
    is one whose A' moved furthest in the effect's direction). AND the gate: on dataset 0, run_arms.run_nominal (map_nominal,
    gibbs arm) must match fit_channels at every causal variant within
    run_arms.GATE_TOL of the se (run_arms.gate_nominal), so the stored A
    pairs L with xL as the pipeline does.

THRESHOLD PROVENANCE. A_TOL, VA_RTOL, F_B, MIN_PAIRS, FANO_BAND, RULE_RTOL,
C_BETA, C_N and C_MIN_READS were set by the pass that wrote this script; it
did not record whether before or after their first results, so they are not
pre-registered. C_SE_MULT and the check (c) pass rule were set on 2026-09-26
before the rule's first run; they replace a range rule on the 1/Va' slope
over beta (0.85-1.05) whose provenance was also not recorded.

(d) EXACT REPRODUCTION OF A STORED NULL. Given the stored null runs' own
    permutation 0 and swap signs (corrected_null_store.OLD/permutations.npz),
    the beta = 0 path (every factor 1) is mapped through run_arms.run_nominal
    for the gibbs and unit arms and compared with those runs' draw 0 over the
    487,454 tests. This, not the beta = 0 anchor's rate (one permutation), is
    the known-answer test of the plumbing.
    The stored draws were made before commit 8a06803, which changed the t
    reference of every p but no slope or se except below the allelic floor:
    the draws referred all three p's to t with OLD_DOF = N - 2 - n_cov = 73
    (the allelic channel is through the origin, n_cov_a = 0); the commit
    refers pval_a to dof_a = n_a - 1, pval_t to dof_t = n_t - 2 - n_cov,
    pval_nominal to the per-pair Welch-Satterthwaite dof_nominal
    (w_a + w_t)^2 / (w_a^2 / dof_a + w_t^2 / dof_t) with w = 1/se^2, and
    leaves the allelic channel out of the combination for genes with fewer
    than hapmixqtl.MIN_ALLELIC_DONORS = 15 informative allelic donors. So:
    PINNED against the stored draw (what the commit does not touch):
      slope_a, slope_t and, where the allelic channel is admitted, slope,
      within REPRO_SLOPE_TOL of their se; slope_a_se, slope_t_se and, where
      admitted, slope_se, within REPRO_SLOPE_TOL relative; pval_t, whose
      reference is unchanged because dof_t = OLD_DOF in every gene (asserted:
      every donor carries total-channel weight), no call at REPRO_ALPHAS
      differs and p agrees within REPRO_P_RTOL relative. pval_nominal where
      admitted is NOT pinned: its statistic slope / slope_se is (above), its
      reference is not (dof_nominal equals 73 only by coincidence).
    CHANGED, verified by recomputation:
      dof_a = n_a - 1 and allelic_admitted = (n_a >= 15), with n_a the
      gene's count of donors whose working allelic variance
      (run_arms.arm_variances) exceeds make_datasets.EPS, which on this
      unthinned path must also equal gene_design.tsv's n_allelic_drop (the
      drop rule make_datasets.allelic_kept applies; n_allelic_keep counts
      without it, e.g. PLK1 42 against 12); dof_a
      is NaN, and so pval_a, exactly where n_a < 2 (channel off).
      pval_a = 2 t.sf(|t_a|, dof_a) with t_a the STORED draw's float32
      slope_a / slope_a_se (map_nominal's own construction), within
      REPRO_P_RTOL and with no call at REPRO_ALPHAS differing.
      Where admitted: dof_nominal equals the Welch-Satterthwaite formula
      recomputed from slope_a_se, slope_t_se, dof_a, dof_t within DOF_RTOL,
      and pval_nominal = 2 t.sf(|t|, dof_nominal) with t the stored draw's
      slope / slope_se, within REPRO_P_RTOL and no call differing.
      The stored draws' own p equal 2 t.sf(|t|, OLD_DOF) of their own
      statistic within STORED_RTOL (confirms what they were referred to).
      STORED_RTOL was raised post hoc from 1e-9 to 1e-6 after the first run
      measured 6.0e-8 (one float32 rounding of t); 1e-6 still separates
      adjacent degrees of freedom by orders of magnitude.
    BELOW THE FLOOR (allelic_admitted False), where the total channel has an
    estimate (slope_t_se finite): slope = slope_t, slope_se = slope_t_se,
    dof_nominal = dof_t and pval_nominal = pval_t exactly (the code clones
    the total channel); elsewhere in those genes pval_nominal is NaN.
    For information: calls per channel and alpha that the new references
    moved relative to the stored draw, and the below-floor genes.
"""
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import corrected_null_store as CNS                                # noqa: E402
import make_datasets as MD                                        # noqa: E402
import run_arms as RA                                             # noqa: E402
from null_permutation_instrument import fit_channels             # noqa: E402
from tensorqtl.hapmixqtl import (LN2, MIN_ALLELIC_DONORS, get_t_pval,   # noqa: E402
                                 summaries_from_point_estimates)

OUT = MD.ROOT / 'checks'
A_TOL, VA_RTOL = 1e-12, 1e-9          # provenance not recorded; measured 1.8e-15 / 7.6e-16
F_B = 0.5                             # provenance not recorded; deeper than the largest scenario's 2^-0.8 = 0.574
MIN_PAIRS = 1000                      # provenance not recorded; exempts the 1-9 read band (364 thinned pairs)
FANO_BAND = (0.95, 1.02)              # provenance not recorded
RULE_RTOL = 1e-9                      # provenance not recorded; measured 3.8e-15
C_BETA, C_N, C_MIN_READS = 0.4, 20, 100   # provenance not recorded
C_SE_MULT = 3                         # set 2026-09-26 before the rule's first run
PREV_RECOVERY = dict(mean=0.992, gene_clustered_se=0.013)   # earlier generator (per-draw thinning of the
                                                            # allelic draws), 1/Va no drop, >= 100 reads (task record)
BANDS = ((1, 10), (10, 100), (100, 1000), (1000, np.inf))
C_BANDS = ([0, 10, 100, 1000, np.inf], ['0-9', '10-99', '100-999', '1000+'])
REPRO_DRAWS = {'gibbs': MD.D / 'corrected_null_store_20260925' / 'draws' / 'drop_000.parquet',
               'unit': MD.D / 'hybrid_weights_null_20260926' / 'draws' / 'unit_000.parquet'}
REPRO_ALPHAS = CNS.ALPHAS             # 0.05 / 0.01 / 0.001, the null-rate alphas (was 0.05 alone before 2026-09-27)
REPRO_SLOPE_TOL = 1e-4                # stored draws are float32; a one-off run 2026-09-26 measured 5.5e-6
REPRO_P_RTOL = 1e-3                   # set 2026-09-27 before its first run: p from two float32 statistics ~5.5e-6 se apart
DOF_RTOL = 1e-5                       # set 2026-09-27 before its first run: Welch-Satterthwaite dof from float32 se
STORED_RTOL = 1e-6                    # the stored p against its own statistic: 1e-9 before the first run, which measured
                                      # 6.0e-8 (one float32 rounding of t); t(73) against t(n_a - 1) differs far more
GENE_DESIGN = MD.D / 'corrected_null_store_20260925' / 'gene_design.tsv'   # n_allelic_drop of the unthinned records


def band_name(lo, hi):
    return f'{lo}-{hi - 1:g}' if np.isfinite(hi) else f'{lo}+'


def check_identity(I, R):
    G, N = R['pL'].shape
    ref = dict(zip(('A', 'T', 'Va', 'Vt'), summaries_from_point_estimates(
        I['pL'], I['pR'], I['pT'], I['eff_lib'], I['YL'], I['YR'], I['YT'])[:4]))
    ref = {k: v[:, I['keep']] for k, v in ref.items()}
    ones = np.ones((G, N))
    rng = np.random.default_rng(np.random.SeedSequence(MD.SEED, spawn_key=(10,)))
    g = MD.generate(R, np.arange(N), np.ones(N, np.int8), ones, ones, rng)
    exact = {k: bool(np.array_equal(g[k], ref[k])) for k in ref}
    perm, swap = MD.record_permutation(N, 0)
    g2 = MD.generate(R, perm, swap, ones, ones, rng)
    moved = dict(T=bool(np.array_equal(g2['T'], ref['T'][:, perm])),
                 Vt=bool(np.array_equal(g2['Vt'], ref['Vt'][:, perm])))
    dA = float(np.abs(g2['A'] - swap[None, :] * ref['A'][:, perm]).max())
    va_ref = ref['Va'][:, perm]
    dVa = float((np.abs(g2['Va'] - va_ref) / np.where(va_ref > 0, va_ref, 1.0)).max())
    ok = all(exact.values()) and all(moved.values()) and dA <= A_TOL and dVa <= VA_RTOL
    print(f'(a) identity, f = 1, no permutation, no swap: exactly equal {exact}')
    print(f'    f = 1 with dataset 0 perm and swap ({int((swap < 0).sum())} of {N} swapped): exactly '
          f'equal {moved}; max |A - swap x A_real[perm]| {dA:.1e} (tol {A_TOL:g}); max relative '
          f'|Va - Va_real[perm]| {dVa:.1e} (tol {VA_RTOL:g})')
    print(f'    {"PASS" if ok else "FAIL"}', flush=True)
    return ok, dict(exact=exact, moved_exact=moved, max_abs_dA=dA, max_rel_dVa=dVa, passed=ok)


def band_stats(pL, pR, YT, Va):
    hap = pL + pR
    mean = YT.mean(2)
    qa = (1.0 / (pL + MD.KAPPA) + 1.0 / (pR + MD.KAPPA)) / LN2 ** 2
    both = (pL >= MD.EXPRESSIBLE_MIN) & (pR >= MD.EXPRESSIBLE_MIN)
    one = (pL < MD.EXPRESSIBLE_MIN) ^ (pR < MD.EXPRESSIBLE_MIN)
    res = {}
    for lo, hi in BANDS:
        m = (hap >= lo) & (hap < hi)
        mf, mb = m & (mean > 0), m & both
        res[band_name(lo, hi)] = dict(
            pairs=int(m.sum()), pairs_positive_mean=int(mf.sum()),
            fano=float(np.median(YT[mf].var(1, ddof=1) / mean[mf])) if mf.any() else np.nan,
            va_over_qa=float(np.median(Va[mb] / qa[mb])) if mb.any() else np.nan,
            gibbs_over_qa=float(np.median((Va[mb] - qa[mb]) / qa[mb])) if mb.any() else np.nan,
            one_side_zero=float(one[m].mean()) if m.any() else np.nan)
    res['outside bands (0 < pL + pR < 1)'] = int(((hap > 0) & (hap < 1)).sum())
    return res


def rule_check(real, th):
    """(Va' - q_a') / (Va - q_a) against q(pL', pR') / q(pL, pR) on informative records."""
    q = lambda pL, pR: 1.0 / (pL + MD.KAPPA) + 1.0 / (pR + MD.KAPPA)
    q0, q1 = q(real['pL'], real['pR']), q(th['pL'], th['pR'])
    h0, h1 = real['pL'] + real['pR'], th['pL'] + th['pR']
    inf = (h0 > 0) & (h1 > 0)
    gibbs0 = real['Va'] - q0 / LN2 ** 2
    no_gibbs = int((inf & ~(gibbs0 > 0)).sum())
    m = inf & (gibbs0 > 0)
    lhs = (th['Va'][m] - q1[m] / LN2 ** 2) / gibbs0[m]
    rhs = q1[m] / q0[m]
    dev = np.abs(lhs - rhs) / rhs
    worst = int(np.argmax(dev))
    zero_exact = bool((th['Va'][h1 <= 0] == 0).all())
    ok = no_gibbs == 0 and float(dev.max()) <= RULE_RTOL and zero_exact
    return ok, dict(records_checked=int(m.sum()), informative_without_gibbs_part=no_gibbs,
                    max_rel_dev=float(dev.max()), rtol=RULE_RTOL,
                    worst_q_a_over_gibbs=float(q0[m][worst] / LN2 ** 2 / gibbs0[m][worst]),
                    q_ratio_median=float(np.median(rhs)),
                    records_zeroed_by_thinning=int(((h0 > 0) & (h1 <= 0)).sum()),
                    va_zero_where_no_reads=zero_exact, passed=bool(ok))


def check_thinning(I, R):
    G, N = R['pL'].shape
    real = MD.generate(R, np.arange(N), np.ones(N, np.int8), np.ones((G, N)), np.ones((G, N)),
                       np.random.default_rng(np.random.SeedSequence(MD.SEED, spawn_key=(11,))))
    f = np.full((G, N), F_B)
    th = MD.generate(R, np.arange(N), np.ones(N, np.int8), f, f,
                     np.random.default_rng(np.random.SeedSequence(MD.SEED, spawn_key=(12,))))
    sr = band_stats(real['pL'], real['pR'], real['YT'], real['Va'])
    st = band_stats(th['pL'], th['pR'], th['YT'], th['Va'])
    print(f'(b) thinning by f = {F_B}: bands of pL + pR; thinned (real). Va/q_a, (Va-q_a)/q_a and one '
          f'side zero are FOR INFORMATION ONLY (different real records fall in a band after thinning)')
    print(f'    {"band":>9s} {"pairs":>13s} {"YT Fano":>15s} {"Va/q_a":>15s} {"(Va-q_a)/q_a":>15s} '
          f'{"one side zero":>15s}  Fano rule')
    ok_fano = True
    for lo, hi in BANDS:
        b = band_name(lo, hi)
        t, r = st[b], sr[b]
        if t['pairs'] >= MIN_PAIRS:
            good = FANO_BAND[0] <= t['fano'] <= FANO_BAND[1]
            ok_fano &= bool(good)
            rule = f'{"PASS" if good else "FAIL"} (in [{FANO_BAND[0]}, {FANO_BAND[1]}])'
        else:
            rule = f'no rule (< {MIN_PAIRS} thinned pairs)'
        print(f'    {b:>9s} {t["pairs"]:>6d} ({r["pairs"]:>4d}) {t["fano"]:>6.3f} ({r["fano"]:.3f}) '
              f'{t["va_over_qa"]:>6.2f} ({r["va_over_qa"]:6.2f}) {t["gibbs_over_qa"]:>6.2f} '
              f'({r["gibbs_over_qa"]:6.2f}) {t["one_side_zero"]:>6.3f} ({r["one_side_zero"]:.3f})  {rule}')
    print(f'    Fano over pairs with a positive draw mean: thinned '
          f'{sum(st[band_name(*b)]["pairs_positive_mean"] for b in BANDS)} of '
          f'{sum(st[band_name(*b)]["pairs"] for b in BANDS)}, real '
          f'{sum(sr[band_name(*b)]["pairs_positive_mean"] for b in BANDS)} of '
          f'{sum(sr[band_name(*b)]["pairs"] for b in BANDS)}')
    k = 'outside bands (0 < pL + pR < 1)'
    print(f'    pairs with 0 < pL + pR < 1, in no band: thinned {st[k]}, real {sr[k]}')
    ok_rule, rr = rule_check(real, th)
    print(f"    allelic rule: {rr['records_checked']} informative records, max relative "
          f"|(Va' - q_a') / (Va - q_a) - q'/q| {rr['max_rel_dev']:.1e} (tol {RULE_RTOL:g}; q_a over the "
          f"Gibbs part at the worst record {rr['worst_q_a_over_gibbs']:.2g}); median q'/q "
          f"{rr['q_ratio_median']:.3f}; informative records without a Gibbs part "
          f"{rr['informative_without_gibbs_part']}; records whose haplotype reads thinning removed "
          f"{rr['records_zeroed_by_thinning']}, Va' exactly 0 wherever pL' + pR' = 0: "
          f"{rr['va_zero_where_no_reads']}  {'PASS' if ok_rule else 'FAIL'}")
    ok = ok_fano and ok_rule
    print(f'    {"PASS" if ok else "FAIL"} (Fano {"PASS" if ok_fano else "FAIL"}, allelic rule '
          f'{"PASS" if ok_rule else "FAIL"})', flush=True)
    return ok, dict(f=F_B, thinned=st, real=sr, fano_passed=bool(ok_fano), allelic_rule=rr, passed=bool(ok))


ESTIMATES = (('inv_va_nodrop_beta', "1/Va' no drop, vs beta"), ('inv_va_beta', "1/Va', vs beta"),
             ('inv_va_exp_beta', '1/Va_exp, vs beta'), ('inv_va_real_beta', '1/Va_real, vs beta'),
             ('unit_beta', 'unit, vs beta'),
             ('unit_pipeline', 'unit, vs pipeline truth'), ('inv_va_pipeline', "1/Va', vs pipeline truth"))


def check_recovery(I, R, tested):
    genes = list(I['genes'])
    hap_real = np.median(R['pL'] + R['pR'], axis=1)
    C0 = I['geno_cov_df'].values
    recs, gate, none = [], None, 0
    for r in range(C_N):
        ds = MD.build_dataset(I, R, tested, C_BETA, 0.0, r)
        M = MD.move_records(R, ds['perm'], ds['swap'])
        va_exp = MD.allelic_variance(M['pL'], M['pR'], ds['fL'] * M['pL'], ds['fR'] * M['pR'], M['YL'], M['YR'])
        va_real = MD.allelic_variance(M['pL'], M['pR'], M['pL'], M['pR'], M['YL'], M['YR'])
        Cg = np.column_stack([I['cov_df'].values[ds['perm']], C0])
        for k in range(len(genes)):
            j = ds['causal_row'][k]
            s = (I['xL'][j] - I['xR'][j]).astype(float)
            kept = ds['kept'][k]
            va = {'inv_va_nodrop': ds['Va'][k], 'inv_va': np.where(kept, ds['Va'][k], 0.0),
                  'inv_va_exp': np.where(kept, va_exp[k], 0.0), 'inv_va_real': np.where(kept, va_real[k], 0.0),
                  'unit': kept.astype(float)}
            fc = {w: fit_channels(ds['A'][k], s, v, ds['T'][k], I['dos'][j].astype(float) / 2.0, ds['Vt'][k], Cg)
                  for w, v in va.items()}
            if any(x is None for x in fc.values()):
                none += 1
                continue
            b, bp = ds['beta'][k], ds['allelic_truth_pipeline'][k]
            recs.append(dict(rep=r, gene=genes[k], hap_real=hap_real[k],
                             inv_va_nodrop_beta=fc['inv_va_nodrop']['ba'] / b, inv_va_beta=fc['inv_va']['ba'] / b,
                             inv_va_exp_beta=fc['inv_va_exp']['ba'] / b, inv_va_real_beta=fc['inv_va_real']['ba'] / b,
                             unit_beta=fc['unit']['ba'] / b,
                             unit_pipeline=fc['unit']['ba'] / bp, inv_va_pipeline=fc['inv_va']['ba'] / bp))
        if r == 0:
            gate = RA.run_nominal(RA.setup(I), ds, 'gibbs', OUT / 'scratch_map_nominal')[1]
            for q in (OUT / 'scratch_map_nominal').glob('*'):
                q.unlink()
            (OUT / 'scratch_map_nominal').rmdir()
    t = pd.DataFrame(recs)
    cols = [c for c, _ in ESTIMATES]
    bad = int((~np.isfinite(t[cols])).any(axis=1).sum())
    if bad:
        raise SystemExit(f'(c): {bad} gene-dataset units with a non-finite slope / truth')
    t['band'] = pd.cut(t.hap_real, C_BANDS[0], right=False, labels=C_BANDS[1])

    def summ(x, col):
        g = x.groupby('gene')[col].mean()
        return dict(genes=int(x.gene.nunique()), units=len(x), mean=float(x[col].mean()),
                    median=float(x[col].median()),
                    gene_clustered_se=float(g.std(ddof=1) / np.sqrt(len(g))) if len(g) > 1 else np.nan)
    hi = t[t.hap_real >= C_MIN_READS]
    res = dict(beta=C_BETA, n_datasets=C_N, units_fewer_than_fit_channels_minimum=none,
               primary={c: summ(hi, c) for c in cols},
               by_band={str(b): {c: summ(x, c) for c in cols} for b, x in t.groupby('band', observed=True)},
               previous_generator=PREV_RECOVERY,
               map_nominal_gate=dict(max_abs_diff_over_se=gate[0], genes=gate[1]))
    p = res['primary']['unit_pipeline']
    ok = abs(p['mean'] - 1) <= C_SE_MULT * p['gene_clustered_se'] and gate[0] < RA.GATE_TOL
    print(f'(c) recovery at |beta| = {C_BETA}, {C_N} datasets, all genes non-null; mean slope / truth at the causal '
          f'variant ({none} gene-dataset units with fewer admitted donors than fit_channels needs, excluded)')
    print(f'    gate: map_nominal (run_arms.run_nominal, gibbs) vs fit_channels on dataset 0, max |diff| / se '
          f'{gate[0]:.1e} over {gate[1]} genes (must be < {RA.GATE_TOL:g})')
    print(f'    {"estimate":26s} ' + ' '.join(f'{b:>16s}' for b in [f'>= {C_MIN_READS}'] + C_BANDS[1]))
    for c, label in ESTIMATES:
        cells = [res['primary'][c]] + [res['by_band'][b][c] for b in C_BANDS[1] if b in res['by_band']]
        print(f'    {label:26s} ' + ' '.join(f'{x["mean"]:>7.3f} ({x["gene_clustered_se"]:.3f})' for x in cells))
    print(f'    units / genes per column: {p["units"]} / {p["genes"]}  ' + '  '.join(
        f'{b} {res["by_band"][b]["unit_beta"]["units"]} / {res["by_band"][b]["unit_beta"]["genes"]}'
        for b in C_BANDS[1] if b in res['by_band']))
    nd = res['primary']['inv_va_nodrop_beta']
    print(f'    change against the earlier generator (per-draw thinning of the allelic draws), same quantity '
          f"(1/Va' no drop, vs beta, >= {C_MIN_READS} reads): {nd['mean']:.3f} - {PREV_RECOVERY['mean']:.3f} "
          f'= {nd["mean"] - PREV_RECOVERY["mean"]:+.3f}')
    print(f'    {"PASS" if ok else "FAIL"} (unit weights over the pipeline-scale truth within {C_SE_MULT} '
          f'gene-clustered se of 1: |{p["mean"]:.3f} - 1| = {abs(p["mean"] - 1):.3f} against '
          f'{C_SE_MULT * p["gene_clustered_se"]:.3f}; and gate)', flush=True)
    res['passed'] = bool(ok)
    return ok, res


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
    ok = bool((fp == fq).all() and ((p[b] == 0) == (q[b] == 0)).all() and rel <= rtol
              and all(v == 0 for v in calls.values()))
    return dict(calls_differ=calls, max_rel=rel, tests=int(b.sum()), passed=ok)


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
        nu_a = np.maximum(np.nan_to_num(dof_a, nan=1.0), 1.0)
        nu_t = np.maximum(np.nan_to_num(dof_t, nan=1.0), 1.0)
        nu = (wa + wt) ** 2 / (wa ** 2 / nu_a + wt ** 2 / nu_t)
    nu = np.where(wt > 0, nu, nu_a)
    nu = np.where(wa > 0, nu, nu_t)
    return np.where(wa + wt > 0, nu, np.nan)


def check_reproduction(I, R):
    """(d) The beta = 0 path, given a stored null run's own permutation 0, reproduces that run's draw 0 in everything
    commit 8a06803 leaves alone, and its new t references recompute from the stored statistics (module docstring)."""
    old = np.load(CNS.OLD / 'permutations.npz')
    perm, swap = old['perms'][0], old['flips'][0].astype(np.int8)
    G, N = R['pL'].shape
    ones = np.ones((G, N))
    g = MD.generate(R, perm, swap, ones, ones, np.random.default_rng(np.random.SeedSequence(MD.SEED, spawn_key=(13,))))
    S = RA.setup(I)
    ds = dict(A=g['A'], T=g['T'], Va=g['Va'], Vt=g['Vt'], pL=g['pL'], pR=g['pR'], perm=perm,
              causal_variant=np.array([min(S['tested'][x]) for x in S['genes']]))
    old_dof = N - 2 - I['cov_df'].shape[1] - I['geno_cov_df'].shape[1]
    keep_design = pd.read_csv(GENE_DESIGN, sep='\t').set_index('gene').n_allelic_drop.loc[S['genes']]
    ok, res = True, {}
    for arm, path in REPRO_DRAWS.items():
        df, _, _ = RA.run_nominal(S, ds, arm, OUT / 'scratch_reproduction')
        ref = pd.read_parquet(path)
        m = df.merge(ref, on=['phenotype_id', 'variant_id'], suffixes=('', '_stored'))
        if len(m) != len(ref) or len(m) != len(df):
            raise SystemExit(f'(d) {arm}: {len(m):,} matched tests, {len(df):,} here, {len(ref):,} in {path}')
        col = lambda c: m[c].to_numpy(float)   # noqa: E731
        # each gene's informative allelic donors, counted without map_nominal
        n_a = pd.Series((RA.arm_variances(ds, arm)[0] > MD.EPS).sum(1), index=S['genes'])
        na = n_a.loc[m.phenotype_id].to_numpy()
        adm = m.allelic_admitted.to_numpy(bool)
        structure = dict(n_a_equals_gene_design=bool((n_a == keep_design).all()),
                         dof_a=bool(np.array_equal(col('dof_a'), np.where(na >= 2, na - 1.0, np.nan), equal_nan=True)),
                         dof_t_all_old_dof=bool((col('dof_t') == old_dof).all()),
                         allelic_admitted=bool(np.array_equal(adm, na >= MIN_ALLELIC_DONORS)))
        # PINNED against the stored draw
        every = np.ones(len(m), bool)
        pin = dict(allelic=pinned(m, every, 'slope_a', 'slope_a_se'), total=pinned(m, every, 'slope_t', 'slope_t_se'),
                   combined_admitted=pinned(m, adm, 'slope', 'slope_se'),
                   pval_t=p_agree(col('pval_t'), col('pval_t_stored'), REPRO_P_RTOL))
        # the stored draw's own p were referred to OLD_DOF
        t_a, t_t, t_c = (t_stat(m, f'{s}_stored', f'{se}_stored')
                         for s, se in (('slope_a', 'slope_a_se'), ('slope_t', 'slope_t_se'), ('slope', 'slope_se')))
        stored_ref = {c: p_agree(col(f'{c}_stored'), get_t_pval(t, old_dof), STORED_RTOL, alphas=())
                      for c, t in (('pval_a', t_a), ('pval_t', t_t), ('pval_nominal', t_c))}
        # CHANGED, recomputed from the stored statistic with the new references
        ws = satterthwaite(col('slope_a_se'), col('slope_t_se'), col('dof_a'), col('dof_t'))[adm]
        dn = col('dof_nominal')[adm]
        fin = np.isfinite(ws) & np.isfinite(dn)
        dof_rel = float(np.max(np.abs(dn[fin] - ws[fin]) / ws[fin]))
        changed = dict(
            pval_a=p_agree(col('pval_a'), get_t_pval(t_a, col('dof_a')), REPRO_P_RTOL),
            dof_nominal_admitted=dict(max_rel=dof_rel, range=[float(dn[fin].min()), float(dn[fin].max())],
                                      passed=bool(np.array_equal(np.isfinite(ws), np.isfinite(dn)) and dof_rel <= DOF_RTOL)),
            pval_nominal_admitted=p_agree(col('pval_nominal')[adm], get_t_pval(t_c[adm], dn), REPRO_P_RTOL))
        # BELOW THE FLOOR the combination is the total channel, verbatim
        b = ~adm
        bt = b & np.isfinite(col('slope_t_se'))
        eq = lambda x, y: bool(np.array_equal(col(x)[bt], col(y)[bt]))   # noqa: E731
        below = dict(genes=sorted(m.phenotype_id[b].unique()), tests=int(b.sum()), with_total_estimate=int(bt.sum()),
                     slope=eq('slope', 'slope_t'), slope_se=eq('slope_se', 'slope_t_se'),
                     dof_nominal=eq('dof_nominal', 'dof_t'), pval_nominal=eq('pval_nominal', 'pval_t'),
                     nan_without_total=bool(m.pval_nominal[b & ~bt].isna().all()))
        # for information: calls the new references moved (NaN counts as not rejected)
        moved = {ch: {str(a): int(((m[c].fillna(1.0) < a) != (m[f'{c}_stored'].fillna(1.0) < a)).sum())
                      for a in REPRO_ALPHAS} for ch, c in CNS.CHANNELS.items()}
        passed = bool(all(structure.values()) and all(x['passed'] for x in pin.values())
                      and all(x['passed'] for x in stored_ref.values()) and all(x['passed'] for x in changed.values())
                      and all(below[k] for k in ('slope', 'slope_se', 'dof_nominal', 'pval_nominal', 'nan_without_total')))
        ok &= passed
        res[arm] = dict(stored=str(path), tests=len(m), old_dof=old_dof, structure=structure, pinned=pin,
                        stored_reference=stored_ref, changed=changed, below_floor=below, calls_moved=moved, passed=passed)
        P = lambda d: 'PASS' if d['passed'] else 'FAIL'   # noqa: E731
        print(f'(d) {arm}: stored permutation 0 vs {path.name}: {len(m):,} tests; call alphas {REPRO_ALPHAS}')
        print(f'    structure: n_a = gene_design n_allelic_drop {structure["n_a_equals_gene_design"]}; dof_a = n_a - 1 '
              f'(NaN at n_a < 2) {structure["dof_a"]}; dof_t = {old_dof} everywhere {structure["dof_t_all_old_dof"]}; '
              f'allelic_admitted = (n_a >= {MIN_ALLELIC_DONORS}) {structure["allelic_admitted"]}')
        for k, x in pin.items():
            if 'max_slope_diff_se' in x:
                print(f'    pinned {k:17s} max |slope diff| / se {x["max_slope_diff_se"]:.1e}, max |se diff| / se '
                      f'{x["max_se_rel"]:.1e} over {x["finite"]:,} finite se (tol {REPRO_SLOPE_TOL:g})  {P(x)}')
            else:
                print(f'    pinned {k:17s} calls that differ {x["calls_differ"]}; max relative |p diff| '
                      f'{x["max_rel"]:.1e} (tol {REPRO_P_RTOL:g}) over {x["tests"]:,}  {P(x)}')
        print(f'    stored draw referred to t({old_dof}): ' + '; '.join(
            f'{c} max relative {x["max_rel"]:.1e} {P(x)}' for c, x in stored_ref.items()) + f' (tol {STORED_RTOL:g})')
        x = changed['pval_a']
        print(f'    changed pval_a vs 2 t.sf(|stored t_a|, dof_a): calls that differ {x["calls_differ"]}; max relative '
              f'{x["max_rel"]:.1e} (tol {REPRO_P_RTOL:g}) over {x["tests"]:,}  {P(x)}')
        x = changed['dof_nominal_admitted']
        print(f'    changed dof_nominal (admitted) vs Welch-Satterthwaite from the se: max relative {x["max_rel"]:.1e} '
              f'(tol {DOF_RTOL:g}); range [{x["range"][0]:.1f}, {x["range"][1]:.1f}]  {P(x)}')
        x = changed['pval_nominal_admitted']
        print(f'    changed pval_nominal (admitted) vs 2 t.sf(|stored t|, dof_nominal): calls that differ '
              f'{x["calls_differ"]}; max relative {x["max_rel"]:.1e} (tol {REPRO_P_RTOL:g}) over {x["tests"]:,}  {P(x)}')
        print(f'    below the floor: {below["genes"]} ({below["tests"]:,} tests, {below["with_total_estimate"]:,} with a '
              f'total estimate): slope = slope_t {below["slope"]}, slope_se = slope_t_se {below["slope_se"]}, '
              f'dof_nominal = dof_t {below["dof_nominal"]}, pval_nominal = pval_t {below["pval_nominal"]}; NaN without '
              f'a total estimate {below["nan_without_total"]}')
        print(f'    for information, calls the new references moved against the stored draw: {moved}')
        print(f'    {"PASS" if passed else "FAIL"}', flush=True)
    shutil.rmtree(OUT / 'scratch_reproduction')
    return ok, res


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    I, R, tested = MD.load()
    oka, ra = check_identity(I, R)
    okb, rb = check_thinning(I, R)
    okc, rc = check_recovery(I, R, tested)
    okd, rd = check_reproduction(I, R)
    res = dict(identity=ra, thinning=rb, recovery=rc, reproduction=rd)
    MD.write_atomic(OUT / 'check_generator.json', lambda fh: fh.write(MD.dumps(res)), 'w')
    print(f'wrote {OUT / "check_generator.json"}')
    if not (oka and okb and okc and okd):
        raise SystemExit(f'FAILED: identity {oka}, thinning {okb}, recovery {okc}, reproduction {okd}')
    print('ALL CHECKS PASS')


if __name__ == '__main__':
    main()

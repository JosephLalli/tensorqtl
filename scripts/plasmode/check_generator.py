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
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import make_datasets as MD                                        # noqa: E402
import run_arms as RA                                             # noqa: E402
from null_permutation_instrument import fit_channels             # noqa: E402
from tensorqtl.hapmixqtl import LN2, summaries_from_point_estimates  # noqa: E402

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


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    I, R, tested = MD.load()
    oka, ra = check_identity(I, R)
    okb, rb = check_thinning(I, R)
    okc, rc = check_recovery(I, R, tested)
    res = dict(identity=ra, thinning=rb, recovery=rc)
    MD.write_atomic(OUT / 'check_generator.json', lambda fh: fh.write(MD.dumps(res)), 'w')
    print(f'wrote {OUT / "check_generator.json"}')
    if not (oka and okb and okc):
        raise SystemExit(f'FAILED: identity {oka}, thinning {okb}, recovery {okc}')
    print('ALL CHECKS PASS')


if __name__ == '__main__':
    main()

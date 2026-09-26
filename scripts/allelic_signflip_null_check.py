"""Does the haplotype-label swap centre the allelic permutation null, and what
does making it the default change in the shipped gene-level calls?

perm_scheme='records_signflip' (the map_cis default since 2026-09-25) permutes
donor records against fixed genotypes, as 'records' does, and swaps each
permuted record's haplotype labels L/R with probability one half, negating its
allelic log ratio. Under 'records' alone the through-origin allelic slope has a
permutation mean equal to the gene's net allelic imbalance times the phase
lopsidedness of the tested variant; on the 46 instrument genes that offset
13 genes beyond Monte Carlo noise.

PART 1, the null at a fixed variant, through the SHIPPED permutation routine
(tensorqtl.hapmixqtl._record_permutation_channel). The 46 genes at RASQUAL's
observed lead, the instrument's 2,000 permutations (RandomState(42)), and the
swap signs drawn from the same stream immediately after the indices, exactly
as map_cis draws them. Gate: the 'records' slopes and t^2 reproduce the
instrument's null_long.tsv.gz per (gene, permutation) to 1e-9 relative. Per
gene: the permuted slope's mean in Monte Carlo z units and in reported-se
units, under each scheme; its agreement with the through-origin algebra; and
the pooled allelic rejection rates at 0.05 / 0.01 / 0.001 with gene-clustered
bootstrap intervals.

PART 2, the shipped gene-level call. map_cis in default mode on the observed
data of all 59 genes (full cis windows, 10,000 permutations): 'records' at
seed 42, 'records' at seed 43, and 'records_signflip' at seed 42. The two
'records' seeds give the Monte Carlo noise floor for a change in pval_perm, so
the scheme change is judged against it: same seed, same indices, only the
swap differs.

Master seed 42 (Part 1's stream is the instrument's; the bootstrap uses a
SeedSequence(42) child; Part 2 passes seeds to map_cis).
"""
import contextlib
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import compare_mixqtl_replication as CM                          # noqa: E402
from tensorqtl.hapmixqtl import (WeightedResidualizer,            # noqa: E402
                                 _record_permutation_channel, map_cis)

D = Path('/mnt/ssd/lalli/brainvar_hapmix_deploy')
RUN = D / 'rasqual_default_mode_20260923'
INST = D / 'nominal_p_null_instrument_20260925'
OUT = D / 'allelic_signflip_null_check_20260925'
SEED, EPS = 42, 1e-12
N_PERM, N_BOOT, NPERM_CIS = 2000, 2000, 10000
ALPHAS = (0.05, 0.01, 0.001)


def rate_ci(k, n, rng):
    idx = rng.integers(0, len(k), size=(N_BOOT, len(k)))
    b = k[idx].sum(1) / n[idx].sum(1)
    return float(k.sum() / n.sum()), float(np.quantile(b, .025)), float(np.quantile(b, .975))


def part1(brng):
    z = np.load(INST / 'inputs_at_lead.npz', allow_pickle=True)
    genes = list(z['genes'])
    A, VA, S = z['a'], z['va'], z['s']
    N = A.shape[1]
    rng = np.random.RandomState(SEED)
    perms = np.stack([rng.permutation(N) for _ in range(N_PERM)])
    flips = rng.randint(0, 2, size=(N_PERM, N)) * 2 - 1          # after the indices, as map_cis
    perm_t = torch.tensor(perms, dtype=torch.long)
    flip_t = torch.tensor(flips, dtype=torch.float64)
    ref = pd.read_csv(INST / 'null_long.tsv.gz', sep='\t')
    gate_b, gate_t2 = 0.0, 0.0
    rows, rej = [], {s: {a: [] for a in ALPHAS} for s in ('records', 'records_signflip')}
    ntest = []
    for k, g in enumerate(genes):
        ok = (VA[k] > EPS) & np.isfinite(A[k]) & np.isfinite(VA[k])
        n_a = int(ok.sum())
        sw = np.where(ok, 1 / np.sqrt(np.where(ok, VA[k], 1.0)), 0.0)
        a = np.where(ok, A[k], 0.0)
        sw_t, a_t = torch.tensor(sw), torch.tensor(a)
        x_t = torch.tensor(S[k][None, :].astype(float))
        res = WeightedResidualizer(None, sw_t, intercept=False)
        r = dict(gene=g, n_a=n_a)
        w = sw ** 2
        r['pred_offset_over_se'] = np.nan
        for scheme, fl in (('records', None), ('records_signflip', flip_t)):
            xy, xx, yy = _record_permutation_channel(x_t, a_t, sw_t, res, perm_t, flip_t=fl)
            xy, xx, yy = xy.numpy()[0], xx.numpy()[0], yy.numpy()
            b = xy / xx
            s2 = (yy - xy * xy / xx) / (n_a - 1)
            se = np.sqrt(s2 / xx)
            t2 = b * b / (se * se)
            p = sps.f.sf(t2, 1, n_a - 1)
            if scheme == 'records':
                sub = ref[ref.gene == g].sort_values('perm')
                gate_b = max(gate_b, float(np.max(np.abs(b - sub.ba.values) / np.maximum(np.abs(sub.ba.values), 1e-12))))
                gate_t2 = max(gate_t2, float(np.max(np.abs(t2 - sub.t2_a.values) / np.maximum(sub.t2_a.values, 1e-3))))
                rms_se = float(np.sqrt((se ** 2).mean()))
                pred = np.mean(S[k]) * np.sum(w * a) / (np.mean(S[k] ** 2) * np.sum(w))
                r['pred_offset_over_se'] = float(pred / rms_se)
            r[f'{scheme}_mean_over_se'] = float(b.mean() / np.sqrt((se ** 2).mean()))
            r[f'{scheme}_mc_z'] = float(b.mean() / (b.std(ddof=1) / np.sqrt(N_PERM)))
            r[f'{scheme}_sdratio'] = float(b.std(ddof=1) / np.sqrt((se ** 2).mean()))
            for a_ in ALPHAS:
                rej[scheme][a_].append(int((p < a_).sum()))
        ntest.append(N_PERM)
        rows.append(r)
    if gate_b > 1e-9 or gate_t2 > 1e-9:
        raise SystemExit(f'GATE FAILED: records slopes vs instrument max rel {gate_b:.2e}, t2 {gate_t2:.2e}')
    per = pd.DataFrame(rows)
    per.to_csv(OUT / 'part1_per_gene.tsv', sep='\t', index=False)
    n = np.array(ntest)
    res = dict(gate_max_rel_slope=gate_b, gate_max_rel_t2=gate_t2, schemes={})
    for scheme in rej:
        zz = per[f'{scheme}_mc_z']
        m = per[f'{scheme}_mean_over_se']
        res['schemes'][scheme] = dict(
            genes_offset_mc_z_gt3=int((zz.abs() > 3).sum()),
            expected_by_chance=float(len(per) * 2 * sps.norm.sf(3)),
            median_abs_mean_over_se=float(m.abs().median()),
            max_abs_mean_over_se=float(m.abs().max()),
            median_sdratio=float(per[f'{scheme}_sdratio'].median()),
            rates={str(a_): dict(zip(('rate', 'lo', 'hi'), rate_ci(np.array(rej[scheme][a_]), n, brng)))
                   for a_ in ALPHAS})
    res['records_offset_vs_algebra_pearson'] = float(np.corrcoef(
        per['records_mean_over_se'], per['pred_offset_over_se'])[0, 1])
    # paired difference in rejection count, signflip minus records, gene-clustered
    res['paired_diff'] = {}
    for a_ in ALPHAS:
        d_ = np.array(rej['records_signflip'][a_]) - np.array(rej['records'][a_])
        idx = brng.integers(0, len(d_), size=(N_BOOT, len(d_)))
        bd = d_[idx].sum(1) / n[idx].sum(1)
        res['paired_diff'][str(a_)] = dict(diff=float(d_.sum() / n.sum()),
                                          lo=float(np.quantile(bd, .025)), hi=float(np.quantile(bd, .975)))
    return res


def part2():
    d = np.load(INST / 'inputs_at_lead.npz', allow_pickle=True)
    with contextlib.redirect_stdout(io.StringIO()):
        I = CM.load_inputs(gene_list=str(D / 'genes_59_stratified_20260923.txt'),
                           regions=str(RUN / 'regions.bed'))
    all_genes = list(d['all_genes'])
    if all_genes != list(I['genes']) or list(I['order']) != list(d['donors']):
        raise SystemExit('gene or donor order differs from the instrument npz')
    order = list(I['order'])
    vdf, xL, xR, dos = I['vdf'], I['xL'], I['xR'], I['dos']
    mk = lambda M, g: pd.DataFrame(M[[all_genes.index(g)]], index=[g], columns=order)
    rows = []
    t0 = time.time()
    for gi, g in enumerate(all_genes):
        vsel = I['idx'][CM.gene_variant_index(I, g)]
        if len(vsel) == 0:
            continue
        v1 = vdf.iloc[vsel]
        one = lambda M: pd.DataFrame(M[vsel], index=v1.index, columns=order)
        r = dict(gene=g, n_variants=int(len(vsel)))
        for label, scheme, seed in (('rec42', 'records', SEED), ('rec43', 'records', SEED + 1),
                                    ('flip42', 'records_signflip', SEED)):
            with contextlib.redirect_stdout(io.StringIO()):
                res = map_cis(one(dos), v1[['chrom', 'pos']], mk(d['A_all'], g), mk(d['T_all'], g),
                              mk(d['Va_all'], g), mk(d['Vt_all'], g), I['gp'].loc[[g]][['chr', 'pos']],
                              xL_df=one(xL), xR_df=one(xR), window=CM.WIN, nperm=NPERM_CIS, seed=seed,
                              covariates_df=I['cov_df'], ase_covariates_df=None, perm_scheme=scheme,
                              verbose=False, warn_monomorphic=False)
            r[f'{label}_lead'] = str(res['variant_id'].iloc[0])
            r[f'{label}_pval_perm'] = float(res['pval_perm'].iloc[0])
            r[f'{label}_pval_beta'] = float(res['pval_beta'].iloc[0])
            r[f'{label}_pval_nominal'] = float(res['pval_nominal'].iloc[0])
        rows.append(r)
        print(f'  {gi + 1}/{len(all_genes)} {g}: perm rec42 {r["rec42_pval_perm"]:.4f} '
              f'rec43 {r["rec43_pval_perm"]:.4f} flip42 {r["flip42_pval_perm"]:.4f} '
              f'({time.time() - t0:.0f}s)', flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(OUT / 'part2_map_cis.tsv', sep='\t', index=False)
    lp = lambda c: -np.log10(t[c].clip(lower=1 / (NPERM_CIS + 1)))
    res = dict(n_genes=int(len(t)), nperm=NPERM_CIS)
    for col in ('pval_perm', 'pval_beta'):
        floor = (lp(f'rec43_{col}') - lp(f'rec42_{col}')).abs()
        change = (lp(f'flip42_{col}') - lp(f'rec42_{col}')).abs()
        res[col] = dict(
            nominal_identical=bool((t['rec42_pval_nominal'] == t['flip42_pval_nominal']).all()),
            lead_identical=bool((t['rec42_lead'] == t['flip42_lead']).all()),
            median_abs_dlog10_seed_floor=float(floor.median()),
            median_abs_dlog10_scheme=float(change.median()),
            max_abs_dlog10_seed_floor=float(floor.max()),
            max_abs_dlog10_scheme=float(change.max()),
            wilcoxon_scheme_vs_floor_p=float(sps.wilcoxon(change, floor).pvalue),
            calls_at_05={lab: int((t[f'{lab}_{col}'] < 0.05).sum()) for lab in ('rec42', 'rec43', 'flip42')},
            calls_changed_scheme=int(((t[f'rec42_{col}'] < 0.05) != (t[f'flip42_{col}'] < 0.05)).sum()),
            calls_changed_seed=int(((t[f'rec42_{col}'] < 0.05) != (t[f'rec43_{col}'] < 0.05)).sum()),
            genes_changed_scheme=t.loc[(t[f'rec42_{col}'] < 0.05) != (t[f'flip42_{col}'] < 0.05), 'gene'].tolist())
    return res


def main():
    OUT.mkdir(exist_ok=True)
    brng = np.random.default_rng(np.random.SeedSequence(SEED).spawn(1)[0])
    t0 = time.time()
    s1 = part1(brng)
    print(json.dumps(s1, indent=1), flush=True)
    s2 = part2()
    print(json.dumps(s2, indent=1), flush=True)
    (OUT / 'summary.json').write_text(json.dumps(dict(part1=s1, part2=s2, runtime_s=time.time() - t0),
                                                 indent=2, default=float))
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()

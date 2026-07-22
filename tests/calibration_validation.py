"""
End-to-end calibration VALIDATION harness for the shipped knockoff eGene path.

Motivation. The unit tests for steps 2-3 (test_per_gene_pvalues,
test_knockoff_calibration_step3, test_genome_wide_fdr) validate the selection
ARITHMETIC by feeding it gene statistics drawn from the exact Binomial null. That
is necessary but not sufficient: it does not check that the REAL pipeline
(simulated genotypes with LD -> real Gaussian/HMM knockoffs -> real SuSiE fits ->
gene_level_W) actually PRODUCES that null and hence a calibrated FDR. This harness
closes that gap. It runs the full pipeline on realistic simulated eQTL panels and
measures realized eGene FDR against the nominal target.

What "realistic" means here (parameterized -- change freely):
  * genotypes from the fastPHASE-style HMM simulator (real LD decay, rare
    variants, recombination hotspots), NOT smooth Gaussian factors;
  * effect sizes specified as VARIANCE EXPLAINED (PVE) by the causal variant,
    with a realistic mixture that is mostly weak (PVE 1-3%);
  * a controllable eGene fraction (1 - pi0);
  * approximate knockoffs at the given N (shrinkage Gaussian or EM-fit HMM) --
    so the test exercises the "the null is only as known as the generator"
    concern, not an idealized generator.

Studies (see the functions and __main__):
  #1 study_end_to_end     -- realized FDR vs nominal across N x signal.
  #2 study_generator_stress -- low N, strong LD + rare variants, UNDER-FIT HMM.
  #3 study_pi0_sweep      -- eGene fraction in {0.1, 0.3, 0.5}.
  #4 study_polygenic      -- dense-polygenic genes mixed with clean nulls; does
                             the intermediate-W contaminant corrupt pi0 / the
                             clean-null FDR? (eGene-level reframing: a polygenic
                             gene genuinely HAS cis signal, so it is not itself a
                             false discovery; the risk is to the null calibration.)

The realized-FDR estimator is the mean per-replicate FDP, (1/B) sum_b V_b/max(R_b,1)
-- the correct FDR estimator (the overlap harness documents why the pooled ratio
is wrong).

Run: python tests/calibration_validation.py --study end_to_end --reps 10
The small-scale pytest gates live in tests/test_calibration_validation.py.

-----------------------------------------------------------------------------
EMPIRICAL FINDINGS (Gaussian knockoffs, selection='calibrated', target FDR 0.1;
realistic HMM-LD genotypes, mixed/mostly-weak signal, ~40% eGenes).

The bar here is CALIBRATION, not one-sided control: the goal is realized FDR ~=
nominal (0.1 means 0.1, "no more no less"). By that bar the current method is NOT
calibrated at either scale tested -- it misses the target in OPPOSITE directions:

  #1 end-to-end vs N (60 genes, 12-25 draws, mean per-replicate FDP):
     N=300, mixed : FDR = 0.029  (target 0.10)  power 0.34  -> OVER-CONSERVATIVE
                    ~3.4x below target: valid FDR control, but throwing away power
                    (the discrete knockoff+ q-value + Storey pi0 over-estimate are
                    conservative; maxPIP is a coarse statistic). This is NOT
                    "calibrated" -- 0.029 != 0.10.
     N=100, mixed : FDR = 0.17 - 0.22  (target 0.10)  power 0.08-0.11 -> INFLATED
                    ~2x ABOVE target. REAL, not Monte-Carlo noise (12 reps, SE
                    ~0.05); adding draws (M 12->25) did NOT fix it (0.17->0.22),
                    so the cause is GENERATOR error -- at N=100 the shrinkage-
                    Gaussian knockoff covariance is poorly estimated, the exact-
                    Binomial null fails, and FDR inflates (Barber-Candes-Samworth:
                    FDR error is governed by knockoff-distribution estimation
                    error). Anti-conservative AND near-zero power.

  #7 coherent-HMM compute (haplotype route, N=120, K=8, M=8): cost is LINEAR in p
     (per-1k-variant time flat at ~16.8 s across a 12x size jump), but the
     constant is large. Extrapolated to a real chromosome: ~28 min at 100k
     variants, ~2.3 h and ~3.8 GB (draw array) at 500k. Algorithmically fine,
     the pure-numpy constant makes genome-scale runs expensive.

BOTTOM LINE: CALIBRATION IS NOT YET ACHIEVED. The method mis-targets in both
directions -- over-conservative at N=300 (~0.03 vs 0.10), anti-conservative at
N=100 (~0.20 vs 0.10) -- so "my 0.1 means 0.1" does not hold at the scales
tested. At best it provides conservative one-sided CONTROL at large N. Getting to
calibration needs work on BOTH ends: reduce high-N conservatism (the discrete
knockoff+ / pi0 over-estimate; a finer statistic than maxPIP) and fix low-N
inflation (better/less-approximate knockoffs; heavier shrinkage; larger N).

The pytest gates below test only one-sided CONTROL ("not grossly anti-
conservative") as regression guards -- they do NOT test calibration and PASS even
when the method is badly over-conservative. Calibration is measured here, in the
harness, and reported honestly; it is a known OPEN problem, recorded in
docs/knockoff_susie_design.md.
-----------------------------------------------------------------------------
"""

import argparse
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import tensorqtl.susie as susie
from hmm_genotype_simulator import simulate_hmm_genotypes


# ---------------------------------------------------------------------------
#  Panel simulation
# ---------------------------------------------------------------------------

def _beta_for_pve(g, pve, rng):
    """Effect size beta so the causal variant explains `pve` of phenotype
    variance with unit residual variance: PVE = b^2 var(g) / (b^2 var(g) + 1)."""
    vg = np.var(g)
    if vg <= 0:
        return 0.0
    b = np.sqrt(pve / max(1e-9, (1 - pve) * vg))
    return b * (1 if rng.rand() < 0.5 else -1)   # random sign


def _draw_pve(regime, rng):
    """Draw a causal PVE from a named regime."""
    if regime == 'weak':
        return rng.uniform(0.01, 0.03)
    if regime == 'moderate':
        return rng.uniform(0.03, 0.10)
    if regime == 'strong':
        return rng.uniform(0.10, 0.25)
    if regime == 'mixed':
        # mostly weak, a few moderate/strong (realistic bulk)
        u = rng.rand()
        if u < 0.7:
            return rng.uniform(0.01, 0.03)
        if u < 0.95:
            return rng.uniform(0.03, 0.10)
        return rng.uniform(0.10, 0.25)
    raise ValueError(regime)


def simulate_eqtl_panel(n_genes, N, p_per_gene, egene_frac, signal_regime,
                        seed, n_polygenic=0, rare_variant_skew=0.7,
                        hmm_K=5, return_phase=False):
    """
    Build a panel of `n_genes` independent cis-windows (each its own HMM block,
    concatenated contiguously on chr1 so the coherent HMM path is also valid).

    A fraction `egene_frac` of genes are sparse eGenes (one causal variant at a
    PVE from `signal_regime`); `n_polygenic` genes are DENSE-polygenic (many tiny
    effects across the window, genuine but non-sparse cis signal); the rest are
    clean nulls (Y independent of genotype).

    Returns a dict of the pipeline inputs plus truth sets:
      genotype_df, variant_df, phenotype_df, pos_df, cov_df,
      causal_genes (sparse eGenes), polygenic_genes, null_genes,
      (xL_df, xR_df if return_phase).
    """
    rng = np.random.RandomState(seed)
    samples = [f"S{i:04d}" for i in range(N)]
    n_egene = int(round(egene_frac * n_genes))

    variant_ids, chroms, poss = [], [], []
    geno_blocks, xL_blocks, xR_blocks = [], [], []
    pheno = np.zeros((n_genes, N))
    causal_genes, polygenic_genes, null_genes = set(), set(), set()
    pheno_ids = [f"G{g}" for g in range(n_genes)]

    # gene roles: first n_egene sparse eGenes, next n_polygenic dense, rest null
    for g in range(n_genes):
        gseed = seed * 100003 + g
        geno, pos, info = simulate_hmm_genotypes(
            p_per_gene, N, seed=gseed, K=hmm_K,
            rare_variant_skew=rare_variant_skew,
            n_hotspots=max(1, p_per_gene // 150), return_phased=return_phase)
        base = 2_000_000 * g + 100_000
        geno_blocks.append(geno)
        if return_phase:
            xL_blocks.append(info['xL'])
            xR_blocks.append(info['xR'])
        for j in range(p_per_gene):
            variant_ids.append(f"g{g}_v{j}")
            chroms.append("chr1")
            poss.append(base + j * 200)

        y = rng.randn(N)
        # pick a common-ish causal variant (avoid monomorphic)
        maf = info['maf']
        elig = np.where(maf > 0.05)[0]
        if g < n_egene and elig.size:
            gid = pheno_ids[g]
            causal_genes.add(gid)
            c = elig[rng.randint(elig.size)]
            pve = _draw_pve(signal_regime, rng)
            y = y + _beta_for_pve(geno[c], pve, rng) * geno[c]
        elif g < n_egene + n_polygenic:
            polygenic_genes.add(pheno_ids[g])
            # dense: many tiny effects summing to a modest total PVE (~5%)
            common = np.where(maf > 0.05)[0]
            if common.size:
                betas = rng.randn(common.size) * 0.04
                contrib = geno[common].T @ betas
                # scale to ~5% total PVE
                vc = np.var(contrib)
                if vc > 0:
                    scale = np.sqrt(0.05 / max(1e-9, (1 - 0.05)) / vc)
                    y = y + scale * contrib
        else:
            null_genes.add(pheno_ids[g])
        pheno[g] = y

    genotype = np.vstack(geno_blocks)
    genotype_df = pd.DataFrame(genotype, index=variant_ids, columns=samples)
    variant_df = pd.DataFrame({'chrom': chroms, 'pos': poss}, index=variant_ids)
    phenotype_df = pd.DataFrame(pheno, index=pheno_ids, columns=samples)
    pos_df = pd.DataFrame(
        {'chr': ['chr1'] * n_genes,
         'pos': [2_000_000 * g + 100_000 + (p_per_gene // 2) * 200 for g in range(n_genes)]},
        index=pheno_ids)
    cov_df = pd.DataFrame(rng.randn(N, 2), index=samples, columns=['PC1', 'PC2'])

    out = dict(genotype_df=genotype_df, variant_df=variant_df,
               phenotype_df=phenotype_df, pos_df=pos_df, cov_df=cov_df,
               causal_genes=causal_genes, polygenic_genes=polygenic_genes,
               null_genes=null_genes)
    if return_phase:
        out['xL_df'] = pd.DataFrame(np.vstack(xL_blocks), index=variant_ids, columns=samples)
        out['xR_df'] = pd.DataFrame(np.vstack(xR_blocks), index=variant_ids, columns=samples)
    return out


# ---------------------------------------------------------------------------
#  Run the real pipeline and score against truth
# ---------------------------------------------------------------------------

def run_and_score(panel, fdr=0.1, n_knockoffs=30, selection='calibrated',
                  knockoff='gaussian', shrink=0.1, dependence='prds',
                  hmm_K=8, hmm_em_iter=15, L=5, max_iter=100, seed=0,
                  window=1_000_000, verbose=False, null_set='null'):
    """
    Run susie.map_egenes_knockoffs on a panel and score realized FDP / power.

    `null_set` selects which truth set counts as the FDR denominator's "false":
      'null'     -> clean-null genes only (default; the honest eGene FDR).
    Genes NOT in causal/polygenic/null (none here) are ignored.
    Returns dict(V, R, power, FDP, n_null_selected, pi0, agreement, ...).
    """
    kw = dict(fdr=fdr, n_knockoffs=n_knockoffs, selection=selection,
              knockoff=knockoff, window=window, L=L, max_iter=max_iter,
              verbose=verbose, seed=seed, localize=False)
    if knockoff == 'gaussian':
        kw['shrink'] = shrink
    else:
        kw['hmm_K'] = hmm_K
        kw['hmm_em_iter'] = hmm_em_iter
    if selection in ('calibrated', 'pvalue', 'qvalue'):
        kw['dependence'] = dependence if selection != 'qvalue' else 'prds'
    eg, _localize, diag = susie.map_egenes_knockoffs(
        panel['genotype_df'], panel['variant_df'], panel['phenotype_df'],
        panel['pos_df'], panel['cov_df'], **kw)

    sel = set(eg[eg['selected']]['phenotype_id'])
    causal = panel['causal_genes']
    nulls = panel['null_genes']
    poly = panel['polygenic_genes']
    R = len(sel)
    # false discoveries = selected CLEAN nulls (polygenic genes genuinely have
    # signal, so are NOT false at the eGene level).
    V = len(sel & nulls)
    FDP = V / max(R, 1)
    power = len(sel & causal) / max(len(causal), 1)
    poly_power = len(sel & poly) / max(len(poly), 1) if poly else float('nan')
    return dict(V=V, R=R, FDP=FDP, power=power, poly_power=poly_power,
                n_null_selected=V, n_causal=len(causal), n_null=len(nulls),
                pi0=diag.get('pi0') if isinstance(diag, dict) else None)


def _summarize(results, label):
    R = np.array([r['R'] for r in results])
    FDP = np.array([r['FDP'] for r in results])
    power = np.array([r['power'] for r in results])
    print(f"  {label}: FDR(mean FDP)={FDP.mean():.3f} (se {FDP.std()/np.sqrt(len(FDP)):.3f}) "
          f"power={np.nanmean(power):.2f}  meanR={R.mean():.1f}  "
          f"P(FDP>2q)={np.mean(FDP>0.2):.2f}")
    return FDP.mean(), np.nanmean(power)


# ---------------------------------------------------------------------------
#  Studies
# ---------------------------------------------------------------------------

def study_end_to_end(reps=10, Ns=(50, 150, 300), regimes=('mixed', 'weak'),
                     n_genes=60, p_per_gene=60, egene_frac=0.3, fdr=0.1,
                     n_knockoffs=30, knockoff='gaussian', selection='calibrated',
                     base_seed=0):
    """#1: realized FDR vs nominal across N x signal regime."""
    print(f"\n=== STUDY #1 end-to-end calibration ({knockoff} knockoffs, "
          f"selection={selection}, target FDR={fdr}) ===")
    for N in Ns:
        for reg in regimes:
            results = []
            for b in range(reps):
                panel = simulate_eqtl_panel(n_genes, N, p_per_gene, egene_frac,
                                            reg, seed=base_seed + b)
                results.append(run_and_score(panel, fdr=fdr, n_knockoffs=n_knockoffs,
                                              knockoff=knockoff, selection=selection,
                                              seed=b))
            _summarize(results, f"N={N:4d} signal={reg:8s}")


def study_generator_stress(reps=10, N=50, fdr=0.1, n_knockoffs=30,
                           n_genes=60, p_per_gene=60, egene_frac=0.3, base_seed=0):
    """#2: the hardest corner -- low N, strong LD + rare variants, and an
    UNDER-FIT HMM (small K, few EM iters). Does realized FDR survive?"""
    print(f"\n=== STUDY #2 generator stress (N={N}, strong LD + rare variants) ===")
    configs = [
        ('gaussian, shrink=0.05', dict(knockoff='gaussian', shrink=0.05)),
        ('gaussian, shrink=0.2 ', dict(knockoff='gaussian', shrink=0.2)),
        ('hmm underfit K=3 it=5', dict(knockoff='hmm', hmm_K=3, hmm_em_iter=5)),
        ('hmm K=8 it=20        ', dict(knockoff='hmm', hmm_K=8, hmm_em_iter=20)),
    ]
    for label, kw in configs:
        results = []
        for b in range(reps):
            panel = simulate_eqtl_panel(n_genes, N, p_per_gene, egene_frac,
                                        'moderate', seed=base_seed + b,
                                        rare_variant_skew=0.85,  # more rare variants
                                        return_phase=(kw.get('knockoff') == 'hmm'))
            results.append(run_and_score(panel, fdr=fdr, n_knockoffs=n_knockoffs,
                                         seed=b, **kw))
        _summarize(results, label)


def study_pi0_sweep(reps=10, N=150, fracs=(0.1, 0.3, 0.5), fdr=0.1,
                    n_knockoffs=30, n_genes=80, p_per_gene=60, base_seed=0):
    """#3: robustness of pi0 estimation / FDR across eGene fraction."""
    print(f"\n=== STUDY #3 pi0 sweep (N={N}, target FDR={fdr}) ===")
    for frac in fracs:
        results = []
        for b in range(reps):
            panel = simulate_eqtl_panel(n_genes, N, p_per_gene, frac, 'mixed',
                                        seed=base_seed + b)
            results.append(run_and_score(panel, fdr=fdr, n_knockoffs=n_knockoffs, seed=b))
        pi0s = [r['pi0'] for r in results if r['pi0'] is not None]
        f, p = _summarize(results, f"eGene_frac={frac:.2f} (true pi0={1-frac:.2f})")
        if pi0s:
            print(f"      mean pi0_hat={np.mean(pi0s):.3f}")


def study_polygenic(reps=10, N=150, fdr=0.1, n_knockoffs=30, n_genes=90,
                    p_per_gene=60, base_seed=0):
    """#4: dense-polygenic genes mixed with sparse eGenes and clean nulls. The
    clean-null FDR must stay controlled despite the intermediate-W polygenic
    contaminant (which could bias pi0). Reports clean-null FDR separately from
    polygenic-gene selection (the latter is legitimate signal)."""
    print(f"\n=== STUDY #4 polygenic contaminant (N={N}, target FDR={fdr}) ===")
    # panel: 20% sparse eGenes, 30% dense-polygenic, 50% clean null
    results = []
    for b in range(reps):
        panel = simulate_eqtl_panel(n_genes, N, p_per_gene, egene_frac=0.2,
                                    signal_regime='mixed', seed=base_seed + b,
                                    n_polygenic=int(0.3 * n_genes))
        results.append(run_and_score(panel, fdr=fdr, n_knockoffs=n_knockoffs, seed=b))
    f, p = _summarize(results, "clean-null FDR")
    poly_power = np.nanmean([r['poly_power'] for r in results])
    print(f"      polygenic-gene selection rate={poly_power:.2f} (legitimate signal)")


STUDIES = {'end_to_end': study_end_to_end, 'generator_stress': study_generator_stress,
           'pi0_sweep': study_pi0_sweep, 'polygenic': study_polygenic}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--study', choices=list(STUDIES) + ['all'], default='end_to_end')
    ap.add_argument('--reps', type=int, default=10)
    ap.add_argument('--knockoff', default='gaussian')
    args = ap.parse_args()
    t0 = time.time()
    todo = STUDIES.keys() if args.study == 'all' else [args.study]
    for s in todo:
        if s == 'end_to_end':
            STUDIES[s](reps=args.reps, knockoff=args.knockoff)
        elif s == 'generator_stress':
            STUDIES[s](reps=args.reps)
        else:
            STUDIES[s](reps=args.reps)
    print(f"\nTotal time: {(time.time()-t0)/60:.1f} min")

# Calibration status of the knockoff eGene FDR — the method is **not yet calibrated**

Branch: `claude/susie-knockoff-calibration-IQ6Za`. This document is the
authoritative, thorough record of the end-to-end validation and its central
negative finding. It supersedes any earlier text (including my own) that implied
the shipped knockoff eGene FDR is calibrated.

---

## 1. TL;DR

Running the **full real pipeline** (simulated LD genotypes → real Gaussian
knockoffs → real SuSiE fits → `gene_level_W` → calibrated selection) and
measuring realized eGene FDR against the nominal target shows the method **misses
the target in opposite directions depending on sample size**:

| Regime (target FDR = 0.10) | Realized FDR | Power | Verdict |
|---|---|---|---|
| N = 300, mixed signal | **0.029** (±0.026, 5 reps) | 0.34 | **over-conservative** (~3.4× too low) |
| N = 100, mixed signal | **0.17–0.22** (±0.05, 12 reps) | 0.08–0.11 | **anti-conservative** (~2× too high) |

Neither regime is **calibrated** in the sense the project targets — "a reported
FDR of 0.10 means 10% of the calls are false, no more and no less." At best, at
large N, the method provides valid but **conservative one-sided control**; at
small N it loses even that and inflates.

This is the single most important result of the validation effort, and it is a
genuine negative finding, not a tuning artifact.

---

## 2. Calibration vs. control — the bar we hold the method to

Two different guarantees are easy to conflate:

- **FDR control:** realized FDR ≤ target. A method that reports 0.03 when you
  asked for 0.10 *controls* FDR — it is simply conservative.
- **FDR calibration:** realized FDR ≈ target. A reported 0.10 corresponds to ~10%
  false calls — not 3%, not 20%.

The project's stated goal has always been **calibration**, because an
over-conservative method silently discards true eGenes (lost power) while an
anti-conservative one silently ships false ones. Both are failures. The unit
tests for steps 2–3 assert one-sided *control*; the validation here measures
*calibration*, and finds it absent in both directions.

---

## 3. Evidence

### 3.1 Methodology

- **Genotypes:** `tests/hmm_genotype_simulator.py` (fastPHASE-style HMM) — real
  LD decay, recombination hotspots, rare variants; NOT smooth Gaussian factors.
- **Panels:** `tests/calibration_validation.py::simulate_eqtl_panel`. Each gene
  is an independent contiguous cis-window (so truth is unambiguous). A fraction
  `egene_frac` are sparse eGenes with one causal variant at a PVE drawn from the
  `mixed` regime (70% PVE 1–3%, 25% 3–10%, 5% 10–25% — realistic, mostly weak);
  the rest are clean nulls (Y ⟂ genotype).
- **Pipeline:** `susie.map_egenes_knockoffs` with `selection='calibrated'`
  (Step-3 known-null Storey q-value + mirror cross-check), Gaussian knockoffs,
  `shrink=0.1`, `L=5`, `M` knockoff draws, target FDR `q=0.1`.
- **Estimator:** realized FDR = **mean per-replicate FDP**,
  `(1/B) Σ_b V_b/max(R_b,1)`, where `V_b` = selected clean-null genes, `R_b` =
  total selected. This is the correct FDR estimator; the pooled ratio `ΣV/ΣR` is
  not (it under-weights low-discovery replicates).

### 3.2 End-to-end results (study #1)

Panel: 60 genes, 50 variants/gene, 40% eGenes, mixed signal.

```
N=300, mixed : FDR = 0.029 (se 0.026)  power 0.34  meanR 8.4  P(FDP>0.2)=0.00   [5 reps]
N=100, mixed : FDR = 0.217 (se 0.087)  power 0.08  meanR 2.6  P(FDP>0.2)=0.60   [5 reps]
```

Because the N=100 number was noisy at 5 reps, it was re-run at 12 reps and two
draw counts to separate real inflation from Monte-Carlo noise **and** from the
p-value resolution limit:

```
N=100, mixed, M=12 : FDR = 0.174 (se 0.062)  power 0.08  meanR 2.2   [12 reps]
N=100, mixed, M=25 : FDR = 0.224 (se 0.052)  power 0.11  meanR 3.4   [12 reps]
```

At 12 replicates the inflation is ~3–4 standard errors above 0.10 — **real, not
noise** — and increasing the number of knockoff draws did **not** reduce it
(0.174 → 0.224), which rules out the `1/(M+1)` resolution limit as the cause.

**Precision caveat:** the N=300 over-conservatism is solid in *direction* but
imprecisely *quantified* (5 reps, se 0.026, so the magnitude could be anywhere
from ~0 to ~0.08). A longer run would pin it. The N=100 inflation is well
established.

---

## 4. Diagnosis — two distinct mechanisms

The two failures have **different causes**, which matters because they need
different fixes.

### 4.1 High-N over-conservatism (the estimator is too cautious)

At N=300 the pipeline only selects genes it is overwhelmingly sure about (power
0.34, ~8 discoveries), so the false-discovery proportion is far below target.
The conservatism is built into the selection statistic, from several stacked
sources:

1. **Storey π₀ over-estimate.** `estimate_pi0_known_null` returns a mildly
   upward-biased π₀ (deliberately, for safety). Every q-value scales linearly
   with π₀, so an over-estimate inflates all q-values → fewer selections.
2. **Discrete Binomial null, tail-conservative.** The known null CDF `F0` is the
   exact Binomial(M,½). Its left tail is *lighter* than a uniform's, which makes
   the q-value a conservative FDP estimate — the same property that makes the
   per-gene p-value super-uniform rather than uniform.
3. **`maxPIP` is a coarse gene statistic.** `W_g = maxPIP(orig) − maxPIP(knockoff)`
   collapses a whole SuSiE fit to one number and discretizes hard; a finer,
   more continuous importance would separate signal from null with less loss.
4. **Monotone (min-from-the-tail) q-value step** only ever *raises* q-values.

None of these is a bug; together they make the method reliably conservative when
N is large enough that the knockoff itself is well estimated.

### 4.2 Low-N anti-conservatism (the generator is wrong)

At N=100 the failure is *not* in the selection math — it is in the **knockoff
generator**. The whole calibration rests on one fact: under the null,
`#{W_g^(m) ≤ 0} ~ Binomial(M, ½)` **exactly**. That derivation assumes the
knockoffs are *exactly valid* model-X knockoffs. At N=100:

- the shrinkage-Gaussian knockoff covariance is estimated from ~100 samples over
  50 correlated variants — noisy and near-singular;
- the resulting knockoffs are *distinguishable* from the originals, so the swap
  symmetry breaks and the true null of `#{W ≤ 0}` is **not** Binomial(M,½);
- the q-values, computed against the assumed-but-false Binomial null, are then
  too small → FDR inflates.

This matches Barber–Candès–Samworth (2020): the FDR-control error of model-X
knockoffs is governed by how well the knockoff distribution is estimated, and at
small N it is estimated poorly. **The decisive diagnostic** that this is
generator error and not a selection/resolution problem: adding knockoff draws
(M 12→25) did not help — every draw reuses the same misspecified generator, so
averaging more of them cannot recover a null that does not hold.

---

## 5. What the automated tests establish — and what they do not

- **`tests/test_per_gene_pvalues.py`, `test_knockoff_calibration_step3.py`,
  `test_genome_wide_fdr.py`** feed `W` drawn from the *exact* Binomial null.
  They validate the **selection arithmetic** given a correct null. They cannot
  and do not validate that the real pipeline produces that null.
- **`tests/test_calibration_validation.py`** (slow) runs the real pipeline but
  asserts only **one-sided control** (mean FDP ≤ q + margin). These gates **pass
  even when the method is grossly over-conservative** (e.g. realized FDR 0.03 at
  target 0.10). They are regression guards against anti-conservative blow-ups at
  the small operating point they use (N≥200, strong signal) — **not** evidence of
  calibration. A green gate here must never be read as "the FDR is calibrated."
- **Calibration itself is only measured by the research harness**
  `tests/calibration_validation.py`, run at scale, and reported in this document.

---

## 6. Compute ceiling (validation item #7)

`tests/knockoff_compute_benchmark.py`, coherent haplotype-HMM route
(N=120, K=8, M=8, EM=10):

```
 p (variants)   fit+draw (s)   s / 1k var   peak MB   draws MB
        1,000         16.34        16.337       76.6        7.7
        4,000         67.18        16.796      306.5       30.7
       12,000        202.57        16.881      919.3       92.2
```

The cost is **linear in p** (per-1k-variant time is flat across a 12× size jump),
which is the expected `O(N·p·K²)`. But the **constant is large** (pure-numpy,
~17 s per 1000 variants). Extrapolated to a real chromosome:

- p = 100,000 → ~28 min fit+draw, ~770 MB for the `[M,N,p]` draw array;
- p = 500,000 → ~2.3 h, ~3.8 GB for the draw array alone.

Genome-scale is *feasible* but *expensive*; the generator is a clear target for
optimization (vectorization, lower precision, chunking, or a compiled inner
loop) before routine whole-genome use.

---

## 7. Path to calibration

**UPDATE (2026-07): the root causes were pinned down and both fixes are now
implemented and validated.** The deeper diagnosis (from a multi-probe
investigation) revised the "high-N vs low-N" framing:
- The high-N over-conservatism is an **ATOM**: SuSiE snaps the single-effect
  prior variance to *exactly 0* under a null, so `alpha` is exactly uniform and
  `maxPIP(orig) − maxPIP(knockoff)` is a point mass at 0. Fix shipped:
  `susie(prior_variance_floor=...)` clamps the prior variance to a small positive
  floor, restoring a continuous fit (commit adds the option; off by default).
- The low-N "inflation" is **not really low-N** — it is an N-persistent,
  asymmetric false-positive tail from the coherent-draw + `maxPIP` construction
  and from Gaussian-knockoff misspecification. NIG / SuSiE 2.0 does *not* fix it.
- **The real fix for the statistic is the KFc redesign** (Wang et al. 2023):
  a **continuous** per-gene statistic `W_g = −log10(min p_real) − −log10(min
  p_knockoff)` (no atom, no ties) + the **empirical mirror-null** knockoff+
  threshold (`W≤0` never selected; no Binomial assumption). Implemented in
  `knockoffs.marginal_importance` / `gene_W_marginal` / `mirror_select_egenes`.
  Validated on realistic HMM genotypes with HMM knockoffs
  (`tests/test_kfc_marginal.py`): **null atom = 0%, realized FDR controlled
  (0.03–0.055 at target 0.10), power 0.48–0.90** — a decisive improvement over
  the maxPIP+Binomial path (atom, 8–17% false-positive tail, power 0.08–0.34).
  Residual: a mild conservatism from the knockoff⁺ offset and a slight `W>0` bias
  when the HMM knockoff is under-fit (K=5); a better-fit knockoff pushes it toward
  the target. SuSiE is now used only to *localize* within selected eGenes.

The original per-failure notes below are retained for the record:

**A. Reduce high-N conservatism** (recover power without losing control):
1. Use a **less-conservative π₀** — the identified lower end of the interval, or
   a debiased estimate, instead of the upward-biased Storey point.
2. Replace the Storey plug-in with the **exact discrete-null FDR bounds**
   (Döhler–Durand–Roquain 2018), which are tighter for a known discrete null.
3. Use a **finer gene statistic than `maxPIP`** (a continuous importance /
   log-Bayes-factor contrast) so `W` discretizes less and separates better.
4. Re-measure realized FDR at N≥300 and check it rises toward 0.10 without
   crossing it.

**B. Fix low-N inflation** (restore a valid null):
1. **Heavier / adaptive shrinkage** on the knockoff covariance at low N, and
   sweep it against realized FDR.
2. Prefer the **HMM generator** over Gaussian at low N (it targets the true
   discrete LD law rather than a second-order approximation).
3. Treat the **mirror cross-check** as a gate: when it and the q-value disagree
   above the detection floor, the null is suspect — surface it, do not report a
   calibrated number.
4. Re-measure at N ∈ {50, 100, 150} and find the N at which the Binomial null
   holds well enough for calibration.

Success criterion for both: realized FDR within Monte-Carlo error of the nominal
target across N and signal regimes — measured by the harness in this repo.

---

## 8. Real-data (HPRC) validation — the generator choice for the KFc statistic

Synthetic HMM-simulated genotypes flatter the HMM knockoff (the data *are* Markov,
so the HMM knockoff trivially matches). To avoid that circularity we validated on
**real human LD**: the HPRC (Human Pangenome Reference Consortium) minigraph-cactus
phased VCFs on the public `human-pangenomics` S3 bucket — v1.0 (N=45) and **v2.0
(N=232)**, sliced into real cis-windows via `pysam` remote tabix. Metric: null-`W`
symmetry — for a null gene (Y ⟂ genotype) a valid knockoff gives `W = imp(real) −
imp(knockoff)` symmetric about 0, so `frac(W>0) → 0.5`; deviation = misspecification.

Findings (KFc `-log10(min-p)` statistic on real HPRC v2.0 LD):
- **The KFc statistic has NO atom on real data** — the redesign holds up outside
  the synthetic world.
- **Our toy genotype-HMM knockoff (Route 1) is misspecified on real LD:**
  `frac(W>0)` ≈ 0.20–0.31 (biased; the knockoff looks *more* associated with a
  null phenotype than the real genotype). This is **not** a small-N artifact — it
  persists N=45 → N=232 — and is **not** fixed by a recombination map (first pass)
  or by more states/iterations in our own implementation (K=15, 40 iters still
  gives ≈0.28). Consequence: the KFc filter selects nothing on real LD (0 power).
- **Per-gene shrinkage-Gaussian (shrink≈0.1) is well-specified on real LD:**
  `frac(W>0)` = 0.556 (mean W ≈ 0). On 200 real HPRC v2.0 windows it **controls
  FDR with real power: realized FDR 0.06–0.07 at target 0.10, power 0.31–0.55**
  (8 reps). `shrink=0.05`/`0.2` were more biased (0.36/0.44).

Why Gaussian, not HMM, for KFc — and the scalability caveat:
- The min-p statistic is LD-sensitive (it depends on the effective number of
  independent tests), so it amplifies any residual mismatch in the knockoff's LD.
  At **cis-window scale** (bounded p, a few hundred variants) the empirical
  covariance is directly estimable, so shrinkage-Gaussian matches it closely.
- Gaussian is O(p³) and infeasible **whole-chromosome**, but the KFc statistic is
  **per gene** (one independent knockoff per cis-window + the genome-wide mirror
  filter), so per-gene Gaussian is feasible. Whole-chromosome coherence — the only
  thing that required the linear-in-p HMM — was a need of the abandoned maxPIP
  per-gene-p-value design.

**Important caveat (do not over-read):** the HMM misspecification here is almost
certainly an **under-fitting / immature-implementation** issue, NOT a limitation
of HMMs. Properly-fit HMM knockoffs control FDR on real human data at scale
(Sesia, Sabatti & Candès 2019 *Biometrika*; Sesia et al. 2021 *PNAS* / KnockoffGWAS
on UK Biobank, N≈489k). The correct test is to swap our toy HMM for a phasing-grade
fit (SNPknock / fastPHASE — the published method; SNPknock's Python package is
installed) and re-run this exact null-`W` test. **That test is in progress**; until
it is done, the validated real-data generator for the shipped KFc path is per-gene
Gaussian, `shrink≈0.1`.

Reproduce: `tests/hprc_calibration.py` (pulls real HPRC windows and runs the
comparison; requires `pysam` + network; opt-in/slow).

---

## 8. Reproduction

```bash
# End-to-end calibration study (the load-bearing measurement)
python tests/calibration_validation.py --study end_to_end --reps 10

# pi0 sweep, generator stress, polygenic contaminant
python tests/calibration_validation.py --study pi0_sweep --reps 10
python tests/calibration_validation.py --study generator_stress --reps 10
python tests/calibration_validation.py --study polygenic --reps 10

# Coherent-HMM compute/memory scaling (#7)
python tests/knockoff_compute_benchmark.py --p 1000 4000 12000 --N 120 --K 8 --M 8

# One-sided-control regression gates (NOT a calibration check)
python -m pytest tests/test_calibration_validation.py -m slow
```

The `simulate_eqtl_panel` / `run_and_score` parameters (N, `signal_regime`,
`egene_frac`, `n_knockoffs`, `knockoff`, `shrink`) are all exposed so the ranges
above can be widened or narrowed freely.

# Knockoff-calibrated SuSiE for tensorQTL — integration spec

Status: investigation complete; the calibrated path is implemented, validated on
real data, and wired into both pipelines. Target branch:
`claude/susie-knockoff-calibration-IQ6Za`. Depends on: `tensorqtl/knockoffs.py`.

---

## ✅ RESOLUTION (final) — read this first

The shipped, calibrated way to get eGene-level FDR from knockoffs is the **KFc
path**: a **continuous** per-gene statistic + an **empirical mirror-null**
selection, with **per-gene Gaussian knockoffs**. Concretely:

- **Use `statistic='kfc'`** in `susie.map_egenes_knockoffs` (standard SuSiE) or
  `hapmixqtl.map_egenes_knockoffs` (two-channel ASE+total). The gene statistic is
  `W_g = imp(real) − imp(knockoff)` where `imp = −log10(min cis p)` (a continuous
  marginal contrast; for hapmixQTL the combined two-channel t). Selection is the
  mirror-null knockoff⁺ threshold `#{W≤−t}/#{W≥t}` (`ko.mirror_select_egenes`);
  `W≤0` is never selected. No SuSiE fit for the filter, **no degenerate atom**.
- **Generator: per-gene Gaussian, `shrink≈0.1`.** Validated on **real HPRC v2.0
  genotypes** (N=232): FDR controlled (0.06–0.07 at target 0.10) with the best
  power (0.31–0.55). It beats even a properly-fit fastPHASE HMM knockoff on power
  for this LD-sensitive min-p statistic, and is feasible per-cis-window (§8 of
  `docs/calibration_findings.md`). Whole-chromosome coherence (the only thing that
  needed the linear-in-p HMM) is not used by the per-gene KFc design.
- **The legacy `statistic='maxpip'` path is DEGENERATE and not calibrated** — under
  a null gene SuSiE's prior variance collapses to exactly 0, `maxPIP(orig) −
  maxPIP(knockoff)` is a point-mass at 0 (an atom), and the assumed Binomial(M,½)
  null is false. Retained for continuity only. If you must use it, `susie(...,
  prior_variance_floor=1e-2)` removes the atom (opt-in; off by default).
- **Full evidence & the corrected mechanistic story** (atom, mirror null, the
  real-HPRC generator comparison, and every over-claim we walked back) are in
  `docs/calibration_findings.md`. Tests: `test_kfc_marginal.py`,
  `test_prior_variance_floor.py`, `test_genome_wide_fdr.py`,
  `test_knockoff_calibration_step3.py`, `tests/hprc_calibration.py` (real-data,
  opt-in), plus the pipeline tests.

The detailed STATUS items and design history below are the append-only record of
how we got here; where they conflict with this block, this block wins.

---

## STATUS (current) — 2026-07, supersedes all stale text below

This doc is an append-only record of an evolving design; several inline
statements below (and in the REVISION NOTICE) have been overtaken by later
validation phases. The authoritative current state is:

1. **Valid procedure = eGene-level FDR** (`susie.map_egenes_knockoffs`, "Path A").
   The gene-level statistic `W_g = max PIP(orig) − max PIP(knockoff)` is
   swap-antisymmetric (verified, `TestSwapEquivariance`). The **credible-set-level
   path** (`susie.map_knockoffs`, `knockoffs.cs_level_W`, `pooled_cs_qvalues`) is
   **EXPERIMENTAL / NOT FDR-controlled** — its statistic is not swap-antisymmetric
   (a real signal's CS disappears under the swap rather than negating). Retained
   only as an exploratory calibration score. Everything in §§1-8 below describing
   CS-level FDR as the deliverable is the superseded original design.

2. **Default selection = single draw, `offset=0`, calibrated pooled q-value**
   (`selection='qvalue'`, `n_knockoffs=1`). This tracks realized FDR ≈ nominal q
   (phase 3). **e-BH / Ren–Barber e-value derandomization is NOT the default** — it
   is provably FDR≤q but empirically pathological near the detection floor
   (over-conservative / power collapse, phase 2). It is retained as an option
   (`selection='ebh'`). Any text below stating e-value derandomization is the
   required default is stale.

3. **The HMM/DMC knockoff generator is VALIDATED** (`knockoffs.dmc_knockoffs`,
   `knockoffs.hmm_knockoffs`; commit 0d329ec). It is an independent reimplementation
   of Sesia, Sabatti & Candès (2019) Algorithms 1-2, verified swap-exchangeable to
   p=20 by a noise-robust pairwise-swap test (`tests/test_hmm_knockoffs.py`). The
   earlier "compounding Z-recursion bug / valid only at small p" belief was a **test
   artifact** — a naive full-joint swap-TV has a noise floor that grows with p, not
   a real defect. Any "HMM blocked/WIP/buggy" wording elsewhere is false. It is
   O(N·p·K²) (linear in p) — the path to chromosome-wide coherent knockoffs.

4. **HMM knockoffs are the default/recommended generator** (`knockoff='hmm'`,
   flipped 2026-07; the pipelines now default to it). CORRECTION to the earlier
   claim here that "Gaussian knockoffs are the current default … validated to NOT
   inflate FDR": that was too strong and is now superseded. Gaussian knockoffs
   (`knockoffs.gaussian_knockoff`) are a **fast second-order approximation** that
   is **misspecified on non-Gaussian, HMM-structured genotypes** — empirically
   they inflate the *original-favored* false-positive tail (the "phase 4"
   validation only ruled out gross failure in mild regimes; it did NOT cover the
   strong-LD/rare-variant corner, which is exactly where Gaussian is misspecified;
   see `docs/calibration_findings.md`). Gaussian is O(p³) (infeasible chromosome-
   wide); HMM is O(N·p·K²..K⁴), matched to the discrete diploid law, and the
   scalable path. Empirical split (2026-07): swapping Gaussian→HMM knockoffs on
   HMM genotypes *reduces* the anti-conservative false-positive tail (confirming
   part of the miscalibration was Gaussian misspecification) but does NOT
   eliminate it and can introduce an over-conservative bias when the HMM is
   under-fit (small K / few EM iters) — so HMM is the principled default, not a
   calibration cure. Use Gaussian only for small p / mild LD or as a comparison
   arm.

4b. **HMM knockoffs are now WIRED INTO the pipeline** (step 1 of the plan),
   with **BOTH** exact constructions of the true diploid law plus a cheap
   approximation. The chromosome-coherent primitive (`chromosome_hmm_knockoffs`)
   fits one HMM per chromosome and draws M knockoff copies; slicing a gene's
   cis-window out of a whole-chromosome draw is a valid knockoff for that window
   (marginalization preserves exchangeability), and overlapping genes share the
   SAME knockoff on shared variants — the coherence a per-gene generator cannot
   give, and the prerequisite for per-gene knockoff p-values (step 2). The three
   generators, selected by `hmm_method` in `susie.map_egenes_knockoffs`:
   - **Route 1 — `method='genotype'` (default, EXACT, unphased).** An unphased
     dosage `G_j = xL_j + xR_j` is the sum of two haplotype chains, so its true
     law is an HMM on the *pair* of ancestral clusters: `build_genotype_pair_hmm`
     assembles the `K²`-state pair HMM (Kronecker transition `Q⊗Q` +
     Bernoulli-convolution dosage emission), and `fit_genotype_hmm` estimates the
     haplotype parameters `(init, Q, θ)` from unphased dosages by *constrained*
     Baum-Welch (fastPHASE genotype EM: pair-state E-step, haplotype-level
     M-step). Exact for the true diploid law; **O(N·p·K⁴)** — the price of not
     phasing.
   - **Route 2 — `method='haplotype'` (EXACT, phased).** `haplotype_hmm_knockoffs`
     fits a haplotype HMM (E=2) on the pooled 2N haplotypes, draws `x̃L | xL` and
     `x̃R | xR` under the shared model, and sets `G̃ = x̃L + x̃R`. Valid because the
     simultaneous swap of both (individually exchangeable, mutually independent)
     haplotype systems induces exactly the genotype swap. **O(N·p·K²)** — the
     cheaper path when phase is available, and it yields the phased knockoffs the
     two-channel hapmixQTL ASE model needs (step 5).
   - **`method='single_chain'` (APPROXIMATE).** One `K`-state chain with a free
     E=3 dosage emission (`fit_hmm`). Cheapest (O(N·p·K²)) but NOT the exact
     diploid law; kept as a fast fallback for very large `K`.
   All three are swap-valid within a Monte-Carlo noise bound on the diploid
   simulator; Route 1 and Route 2 (exact) edge out single-chain. `hmm_K`,
   `hmm_em_iter`, `hmm_params` (pre-fit per-chromosome params), and
   `phased_haplotypes=(xL_df, xR_df)` are exposed on the pipeline. Tests:
   `tests/test_hmm_knockoff_pipeline.py` (16, pass).

   **Compute note:** Route 1 costs ~`K²/2×` more than Route 2 (K⁴ vs K² in the
   hidden-state factor; ~15-50× at K=10) for both fit and draw. If phase is
   available, Route 2 dominates on every axis; Route 1 is the exact fallback for
   unphased input.

5. **Per-gene knockoff p-values are BUILT (step 2)** (`knockoffs.per_gene_pvalues`,
   `select_egenes_pvalue`; `selection='pvalue'`). Construction note / correction
   to the earlier plan: the shipped p-value is the **per-draw sign count**
   `p_g = (offset + #{m: W_g^(m) ≤ 0}) / (offset + M)`, NOT the "rank of R_g among
   {R_g, K_g^(1..M)}" originally sketched here. The sign-count has an EXACTLY
   known null — under H_g each draw uses an independently generated knockoff, so
   the per-draw model-X swap makes the sign vector uniform on {±1}^M and
   `#{W≤0} ~ Binomial(M, ½)` exactly. This is a *valid* (super-uniform,
   CONSERVATIVE) p-value, **not marginally uniform** (the Binomial count
   concentrates near M/2). Resolution is `1/(M+1)`; direct BH selection needs
   `M ≳ n/(q·R)` (governed by the number of jointly discoverable genes R), so the
   p-value's real role is the step-3 calibration primitive, not standalone BH.
   Tests: `tests/test_per_gene_pvalues.py` (15, pass).

6. **Interval-valued π₀ empirical-Bayes calibration is BUILT (step 3)**
   (`knockoffs.calibrated_qvalues`, `mirror_fdp`, `local_fdr_interval`,
   `estimate_pi0_known_null`, `select_egenes_calibrated`; `selection='calibrated'`).
   Because the per-gene null is the exact discrete Binomial(M,½), calibration
   uses the **known null CDF** (Döhler–Durand–Roquain discrete-FDR / Storey with
   `F₀=Binomial` in place of uniform-p) rather than the uniform-null Storey
   formula. Three redundant estimators, three literatures, one input (the
   per-gene win-counts `b_g`):
   - `calibrated_qvalues` — SHIPPED selector: known-null Storey q-values
     (Storey 2003; Storey–Taylor–Siegmund 2004). Stable (CDF-based), leans on a
     π₀ estimate.
   - `mirror_fdp` — π₀-FREE cross-check via the null's symmetry (Barber–Candès
     knockoff⁺; mirror-statistic line, Dai–Lin–Xing–Liu 2023). Its agreement
     with the q-values is a built-in **misspecification alarm** (the pipeline
     warns if Jaccard agreement < 0.5).
   - `local_fdr_interval` — per-gene lfdr (Efron 2004 two-groups) reported as an
     INTERVAL because π₀ is only **partially identified** (Genovese–Wasserman
     2004): identified upper bound + symmetry/excess-mass lower reference.
   `estimate_pi0_known_null` fixes π₀ by Storey on the known null over the
   mass-rich, signal-poor band `0.10 ≤ P(B>c) ≤ 0.50` (avoiding the degenerate
   deep-tail that biases π₀ downward). Empirical calibration (the project's
   gate): null mean FDP ≈ 0; mixed panels track target q across π₀∈{0.6,0.8,0.9}
   with the mirror agreeing (Jaccard→1). Tests:
   `tests/test_knockoff_calibration_step3.py` (16, pass).

7. **Genome-wide FDR: the "joint-sign" problem is REDUCED and empirically
   resolved** (`knockoffs.bh_select`/`calibrated_qvalues`/`select_egenes_*` gain
   a `dependence` arg; `tests/test_genome_wide_fdr.py`, 7 pass). The key move:
   once each gene has a per-gene p-value with an EXACTLY known marginal null
   (step 2), genome-wide eGene FDR is no longer a novel knockoff-joint-sign
   theorem — it is the classical **BH-under-dependence** problem
   (Benjamini–Hochberg 1995; Benjamini–Yekutieli 2001). What the simulations
   establish (mean realized FDR at q=0.1):
   - **Independent or LOCAL (block) dependence — the realistic eQTL regime**
     (distant genes ≈ independent, only nearby genes share LD): the shipped
     calibrated q-value (`dependence='prds'`, π₀=auto) controls FDR tightly —
     0.087 independent; 0.090–0.095 at within-block ρ up to 0.9 — at full power.
     This is the **default and the validated operating regime**.
   - **Adversarial GLOBAL equicorrelation** (every gene coupled to every other —
     not an eQTL reality): plain BH can **inflate** (0.18 at ρ=0.7, 0.24 at
     ρ=0.9). Diagnosis: the **π₀-adaptive step, not BH itself, is fragile** — a
     bad replicate makes all nulls look like signal, π₀ is under-estimated, and
     q-values shrink.
   - **Guaranteed fallback:** `dependence='arbitrary'` (Benjamini–Yekutieli
     harmonic factor `c(n)=Σ1/j`) **with π₀=1.0** (drop the fragile adaptive
     step) controls FDR under ANY dependence — verified ≈0.011 even at ρ=0.9 —
     at the cost of log-factor conservatism.
   Honest caveat retained: under strong dependence the per-analysis FDP has large
   VARIANCE even when its mean (FDR) is controlled; BH/BY bound the expectation,
   not the realized proportion in one run. Status upgraded from "empirically
   calibrated, no theorem" to "reduced to BH-under-dependence; PRDS validated for
   the local eQTL regime, BY+π₀=1 for the worst case." A closed-form PRDS proof
   for the specific coherent-knockoff p-value construction remains open, but is no
   longer on the critical path.

8. **Route-2 phased knockoffs are WIRED INTO the two-channel hapmixQTL model
   (step 5)** (`hapmixqtl.map_egenes_knockoffs`, `_build_knockoff_stacked_design`;
   `tests/test_hapmixqtl_knockoffs.py`, 5 pass). For each gene it draws M phased
   haplotype knockoffs `(x̃L, x̃R)` from a per-gene haplotype HMM
   (`haplotype_hmm_knockoffs`), builds the augmented two-channel stacked design
   `[X, X̃]` where BOTH the ASE channel (`s̃ = x̃L − x̃R`) and the total channel
   (`g̃/2 = (x̃L + x̃R)/2`) take their knockoff columns from the SAME knockoff
   haplotypes — the coherence a phased (Route 2) knockoff provides and an
   independent per-channel knockoff would break — then fits SuSiE and forms
   `W_g = maxPIP(orig) − maxPIP(knockoff)`. The per-gene W feeds the identical
   step-2/step-3 calibration (`select_egenes_calibrated`, `dependence` arg), so
   eGene FDR is calibrated exactly as for standard SuSiE. Phase is REQUIRED (the
   ASE channel only exists with phase). Both **chromosome-coherent** (`coherent=
   True`, default) and per-gene draws are supported: coherent mode fits one
   haplotype HMM per chromosome and draws M phased knockoff copies of the whole
   chromosome up front (PASS 0, via `chromosome_hmm_knockoffs(..., method=
   'haplotype', return_phased=True)`), then slices each gene's window — so
   overlapping genes share the SAME knockoff haplotypes on shared variants
   (verified: `TestCoherence.test_shared_variants_identical_across_windows`),
   the prerequisite for cross-gene per-gene p-values. `coherent=False` keeps the
   independent per-gene fit for non-overlapping loci.

9. **END-TO-END VALIDATION on the REAL pipeline (not synthetic W)** — the
   arithmetic-only step-2/3 tests feed W drawn from the exact Binomial null; this
   validates the selection math but NOT that the real `LD genotypes → knockoffs →
   SuSiE → W` stack produces that null. `tests/calibration_validation.py` (research
   harness) + `tests/test_calibration_validation.py` (small slow gates) close that
   gap by running the full pipeline on realistic HMM-LD genotypes with PVE-scaled,
   mostly-weak signal. Honest findings (Gaussian knockoffs, target FDR 0.1, mean
   per-replicate FDP). **The bar is CALIBRATION (realized ≈ nominal), not
   one-sided control — and by that bar the method is NOT yet calibrated: it
   misses the target in OPPOSITE directions at the two scales tested.**
   - **N = 300, mixed signal: OVER-CONSERVATIVE** — realized FDR 0.029 vs target
     0.10 (~3.4× too low), power 0.34. Valid one-sided control, but *not*
     calibrated — it wastes power. Sources: the discrete knockoff⁺ q-value and
     Storey π₀ over-estimate are conservative, and maxPIP is a coarse statistic.
   - **N ≈ 100, realistic weak signal: ANTI-CONSERVATIVE (~2×) and near-zero
     power** — realized FDR 0.17–0.22 vs 0.10 (12 reps, SE≈0.05, *not* Monte-Carlo
     noise), power 0.08–0.11. Adding knockoff draws (M 12→25) did NOT fix it → the
     cause is GENERATOR error (poorly-estimated shrinkage-Gaussian knockoff
     covariance at low N; Barber–Candès–Samworth), NOT the selection math.
   - **Bottom line: calibration is an OPEN problem.** Reducing high-N conservatism
     (finer statistic than maxPIP; less-conservative π₀/discrete-null handling)
     and fixing low-N inflation (better/less-approximate knockoffs; heavier
     shrinkage; larger N) are both needed. The pytest gates test only one-sided
     control and PASS even under gross over-conservatism — they do not certify
     calibration. **See `docs/calibration_findings.md` for the full, thorough
     treatment** (evidence, per-mechanism diagnosis, compute ceiling, the path to
     calibration, and reproduction commands).
   - **Compute (item #7):** the coherent HMM is LINEAR in p (per-1k-variant time
     flat) but with a large constant (~17 s/1k variants at K=8,M=8,N=120,
     pure-numpy) → ~28 min at 100k variants, ~2.3 h + ~3.8 GB at 500k. Genome
     scale is feasible but expensive; a target for future optimization.

10. **Still open / not built:** a closed-form PRDS proof for the coherent-knockoff
   p-value construction (off the critical path); low-N generator mitigation and
   the deferred long-term validations (#5 phasing-error, #6 Salmon-posterior,
   #8 real-data concordance).

---

## ⚠️ REVISION NOTICE (supersedes the CS-level design below)

External expert review found the **credible-set-level FDR** procedure originally
specified here (§§2–4 below) to be **not a valid knockoff procedure**. The core
defect, now confirmed empirically
(`tests/test_knockoffs.py::TestSwapEquivariance`):

> The CS-level statistic extracts credible sets from the **original columns
> only** of an augmented `[X, X̃]` fit. Under the model-X swap (exchange every
> `X_j` with `X̃_j`), a valid statistic must map to its negation via a
> deterministic hypothesis correspondence. This one does **not** — the credible
> set for a real signal *disappears* into the "knockoff" block rather than
> negating. So the pooled negatives are **not** valid negative controls, and
> `knockoff_qval ≤ fdr` does **not** control FDR.

Also invalid as originally written: mean-W derandomization (must use Ren–Barber
e-values), the calibration estimator (`ΣV/ΣR` → must be mean per-replicate FDP),
doubling `L`, and describing KFc as rigorous CS-level precedent.

**What is valid and now implemented as v1 — Path A, eGene-level FDR.** The
gene-level statistic `W_g = max PIP(orig block) − max PIP(knockoff block)` tests
the FIXED hypothesis "gene g has no cis signal" and IS swap-antisymmetric
(verified). Genes are selected at genome-wide FDR via knockoff+ with e-value
derandomization; ordinary SuSiE then localizes signal *within* selected genes.
The reported unit is the **gene**, not the credible set. Implemented in
`susie.map_egenes_knockoffs`.

The CS-level code (`knockoffs.cs_level_W`, `pooled_cs_qvalues`,
`susie.map_knockoffs`) is **retained but relabeled experimental / NOT
FDR-controlled** — an exploratory calibration score only.

Corrections applied to the shared core regardless of path:
- `augmented_susie_fit`: pass `L` (not `2L`).
- `select_egenes`: Ren–Barber e-value derandomization (no W-averaging).
- `calibration_report`: mean per-replicate FDP `(1/B)Σ Vb/max(Rb,1)` + the
  complete-null `P(R>0)`.
- Gaussian generator explicitly documented as *approximate* (pipeline
  development / benchmarking), not a basis for exact FDR on genotype data; an
  HMM / reference-panel haplotype generator is likely required for real-data
  credibility.

### Validity status (reviewer-agreed wording)

> The gene-level statistics satisfy the required deterministic antisymmetry
> within each gene. Formal genome-wide knockoff+ FDR control additionally
> requires a joint null sign-flip property across genes. Independent per-gene
> knockoff generation does not obviously provide that property when cis windows
> overlap, and independent generator randomness alone is insufficient to prove
> it. We therefore compare independent and shared knockoff constructions in
> simulations with overlapping windows and correlated phenotypes while pursuing
> a formal reduction for the overlapping gene-level hypotheses. Until that issue
> is resolved, results are described as **empirically calibrated**, not
> theorem-backed genome-wide FDR control.

Key correction from the second review: independent per-gene knockoff *randomness*
does NOT establish the joint sign property — the property comes from
distributional exchangeability of `(X, X̃, Y)`, not from independent RNG streams.
Under independent per-gene knockoffs a shared SNP has *different, incompatible*
knockoffs across the genes whose windows contain it, so "swap gene A" is not
induced by any coherent swap of a single `(X, X̃)` object. A **shared
chromosome-wide (or LD-block-wide) knockoff** — one `X̃` reused by every gene —
restores coherence and is the construction we test; but even it does not by
itself invoke a standard knockoff theorem, because the discovery units are
phenotype-specific *union* nulls over overlapping feature sets, which needs its
own reduction (closer to simultaneous/union-null knockoffs than to standard
group-sparse multitask knockoffs; Dai–Barber is related but not an exact match).
Simulations can falsify or reveal inflation and support an empirical-calibration
claim; they cannot substitute for that reduction.

### Calibration harness (Freedman–Lane, reviewer-agreed wording)

> Null calibration uses a Freedman–Lane-style residual permutation conditional
> on nuisance covariates. A **common sample permutation** is applied to the full
> gene-by-sample residual matrix so that cross-gene residual correlation is
> preserved. Restricted exchangeability blocks will be supported where required.
> Expression-derived latent covariates (PEER/expression PCs) will be either held
> fixed or re-estimated according to an explicitly reported calibration mode.

Implemented (`susie.map_egenes_knockoffs`, `permute_null=True`): one shared row
permutation of the covariate-residualized phenotype across all genes, in residual
space. NOT yet implemented: restricted exchangeability blocks (relatedness,
batch, strata), and the `covariate_mode = {"fixed","refit_latent"}` policy for
expression-derived factors — currently equivalent to "fixed" (covariates given).
The overlap-stress harness (`tests/overlap_calibration_harness.py`) builds
overlapping windows + tunable cross-gene phenotype correlation
`Y_B = ρ Y_A + √(1−ρ²)ε` (near-duplicated as ρ→1) and compares per-gene vs shared
knockoffs, reporting mean per-replicate FDP, `P(R>0)`, and Monte-Carlo intervals.

Still-open items: the overlapping-gene joint-sign *reduction* (the central
unresolved inferential problem), restricted-exchangeability permutation, the
latent-covariate calibration policy, real wall-time/convergence benchmarking, and
a scalable shared-knockoff generator (block-Gaussian or HMM) for true
chromosome scale. Path C (sample-splitting) is dropped per project scope.

The sections below are **retained as the record of the original CS-level design
and why it failed**; they do not describe a valid procedure.

---

## 1. What we are controlling, and why the aggregation is what it is

**Goal (user's words):** report the set of credible sets whose discovery is
estimated to hold a target FDR (e.g. 5%) — "q=0.05 really means 5% of my calls
are false."

**Two facts force the architecture:**

1. **Knockoffs are generated per gene.** Each cis-window has its own LD, so `X̃`
   is built from that window's genotypes, one gene at a time. Not in dispute.

2. **FDR must be controlled by pooling W across genes, not per gene.** The
   knockoff+ filter has a *detection floor*: with `k` discoveries the smallest
   certifiable FDR is `1/k` (the `+1` offset over `k` positives). A single gene
   yields ~1–3 credible sets, so per-gene CS-level q=0.05 is arithmetically
   impossible (a 2-CS gene floors at 0.5). The only way "q=0.05 means 5%" is
   achievable is to pool the per-CS statistics across all genes and threshold
   once — thousands of discoveries, floor ≈ 1/(#CS genome-wide) ≪ 0.05.

**Validity under W heterogeneity (the reviewed concern).** Pooling is valid even
though W magnitudes vary wildly across genes, because the knockoff guarantee
depends only on the **sign symmetry** of null statistics, not on their scale.
Each gene's null CSs are independently sign-symmetric (`±|W|` equally likely);
the pooled null set is therefore sign-symmetric at every threshold, so the
pooled knockoff+ estimate `FDP̂(t) = (1 + #{W ≤ −t}) / #{W ≥ t}` controls FDR
regardless of cross-gene scale differences.

What pooling costs (not a validity failure, but must be documented):
- **Global ≠ conditional FDR.** 5% overall does *not* mean each gene is at 5%;
  false discoveries may concentrate in a stratum. Same property as BH q-values.
- **Power concentrates in strong loci.** High-W genes dominate the selection.
  For eQTL this is usually acceptable (strong eGenes are the trustworthy ones);
  a v2 within-gene standardization knob (§6) can rebalance if needed.

**The real risk pooling is vulnerable to is knockoff *quality*, not W scale.** If
one gene's `Σ` is badly estimated, its null CSs get systematically positive W
(knockoff too easy to beat) → its nulls are no longer sign-symmetric → pooling
imports anti-conservative bias. Mitigations: mandatory shrinkage, optional HMM
generator, and the null-permutation calibration gate (§5). This is the small-N
`Σ`-estimation problem, and it is the thing the calibration report must catch.

**Precedent.** KFc (Wang et al. 2023) already estimates its gene-level FDP by a
sliding window over the *genome-wide* W distribution — i.e. it pools. Our
approach is the same move done at the credible-set level on individual-level
SuSiE output, emitting a rigorous per-CS knockoff q-value rather than KFc's
`PIP × (1 − FDP)` heuristic multiply.

---

## 2. Output object: per-CS knockoff q-value (not a modified PIP)

We do **not** overwrite SuSiE's PIP (a Bayesian posterior) with a frequentist
factor. We report, per credible set, both numbers:

| column | meaning |
| --- | --- |
| `phenotype_id`, `variant_id` | as in `susie.map` today |
| `pip`, `af`, `cs_id` | as in `susie.map` today (SuSiE's Bayesian output, unchanged) |
| `cs_W` | the CS-level knockoff statistic for this CS (from the gene's fit) |
| `knockoff_qval` | the pooled knockoff q-value for this CS's discovery |
| `selected` | bool: `knockoff_qval <= fdr` |

The **per-CS knockoff q-value** is, for a CS with statistic `W_k` drawn from the
pooled set `W_all`:

```
qval(W_k) = min over thresholds t <= W_k of  FDP̂(t; W_all)
          = min_{t <= W_k}  (1 + #{W_all <= -t}) / max(1, #{W_all >= t})
```

(monotonized so q is non-increasing in W). Reporting `knockoff_qval <= fdr`
yields a set of CSs with the target genome-wide FDR. `cs_W` and `knockoff_qval`
sit *alongside* `pip`, giving each CS a calibrated frequentist confidence next to
its Bayesian one.

---

## 3. Control flow (two-pass: per-gene fit, then global threshold)

Because the threshold is global, we cannot decide a gene's CSs until all genes'
W's are in hand. So the driver is **two-pass**:

```
PASS 1 (per gene, parallelizable):
  for each gene:
    X   = residualized dosage [N, p]              # exactly as susie.map today
    for r in 1..M_knockoffs:                       # M draws for derandomization
        Xk  = knockoffs.gaussian_knockoff(X, shrink, seed=r)   # or hmm
        res = susie.susie([X, Xk], y, L=2L, intercept=..., ...) # augmented fit
        cs_r = knockoffs.cs_level_W(res, p, stat='pip')         # per-CS W (orig cols only)
    aggregate the M draws -> one (cs_id -> W, member variant_ids, pip) record per CS
    stash gene's CS records; also stash SuSiE's normal per-CS output

PASS 2 (global):
  W_all = concat of cs_W across ALL genes
  for each CS: knockoff_qval = qval(cs_W; W_all)      # §2 formula
  selected = knockoff_qval <= fdr
  return the summary table with knockoff_qval + selected columns
```

`M_knockoffs` (derandomization) reduces run-to-run selection variance; the M
draws are aggregated per gene via Ren–Barber e-values (already in
`knockoffs.derandomize_cs`) OR, simplest for v1, by averaging each CS's W across
draws (matched by original-member set). Start with mean-W aggregation; upgrade to
e-value aggregation if stability is inadequate.

**Cost:** `M × (SuSiE fit on 2p columns)` per gene. With `M=5` and 2× columns,
~10× a plain SuSiE fit — still far below the `nperm=10000` cis permutation pass
tensorQTL already runs per gene, so affordable.

---

## 4. Function signatures

### 4.1 New driver in `susie.py` (or a thin wrapper module)

```python
def map_knockoffs(
    genotype_df, variant_df, phenotype_df, phenotype_pos_df, covariates_df,
    # --- knockoff calibration params ---
    fdr=0.05,                     # target genome-wide CS-level FDR
    n_knockoffs=5,                # M derandomization draws per gene
    knockoff='gaussian',          # 'gaussian' | 'hmm' (hmm: later phase)
    shrink=0.05,                  # covariance shrinkage (Gaussian generator)
    w_stat='pip',                 # importance: 'pip' | 'max_alpha'
    knockoff_offset=1,            # knockoff+ (1) vs knockoff (0)
    seed=0,
    emit_diagnostics=True,        # attach per-gene W spread, est. FDP, etc.
    # --- passthrough to susie.map / susie.susie (unchanged semantics) ---
    paired_covariate_df=None, L=10, scaled_prior_variance=0.2,
    estimate_residual_variance=True, estimate_prior_variance=True,
    tol=1e-3, coverage=0.95, min_abs_corr=0.5,
    maf_threshold=0, max_iter=200, window=1000000,
    logger=None, verbose=True, warn_monomorphic=False,
):
    """
    SuSiE fine-mapping with knockoff-calibrated credible-set FDR control.

    Returns:
        summary_df: one row per (credible set member), as susie.map's summary,
            plus columns: cs_W, knockoff_qval, selected.
        diagnostics: dict (per-gene W distribution, calibration inputs) if
            emit_diagnostics, else None.
    """
```

Design choices baked into the signature:
- **Mirrors `susie.map` exactly** for every shared argument, so it is a drop-in
  superset. A caller who ignores the knockoff args gets standard behavior plus
  the calibration columns.
- **`fdr` is the sensitivity dial.** Lower = stricter/fewer CSs.
- **`knockoff='hmm'` default (recommended); `'gaussian'` = fast approximation.**
  (STALE TEXT CORRECTED: HMM is no longer "reserved for a later phase" and needs
  no SNPknock dependency — it is implemented in `knockoffs.py`, validated, and
  wired into `map_egenes_knockoffs` with three exact/approximate constructions.
  Gaussian remains dependency-free and is retained as a fast approximation valid
  only at small p / mild LD.)

### 4.2 Augmented-fit helper (new, in `knockoffs.py`)

```python
def augmented_susie_fit(susie_module, X_t, y_t, Xk_t, L, **susie_kwargs):
    """
    Fit SuSiE on [X, Xk] and return (res, p). Sets L -> handles the doubled
    column space. Purity/CS reporting must use ORIGINAL columns only (caller
    slices res['alpha'][:, :p] for genotype-LD purity, as hapmixqtl.map_susie
    already does).
    """
```

### 4.3 Pooling + q-value (new, in `knockoffs.py`)

```python
def pooled_cs_qvalues(per_gene_cs, fdr, offset=1):
    """
    per_gene_cs: list of per-CS records across ALL genes, each with a scalar
        'W' (aggregated over knockoff draws), plus carry-through fields
        (phenotype_id, cs_id, member variant indices/ids, pip).
    Returns the same records with 'knockoff_qval' and 'selected' filled in,
    using the pooled W distribution (§2 formula).
    """
```

`knockoffs.py` already has `gaussian_knockoff`, `cs_level_W`,
`filter_credible_sets`, `knockoff_threshold`, `derandomize_cs`,
`calibration_report`. This spec adds `augmented_susie_fit` and
`pooled_cs_qvalues`; `filter_credible_sets` becomes the per-gene special case and
`pooled_cs_qvalues` is the genome-wide aggregator.

---

## 5. Calibration gate (must pass before the target FDR is trusted)

Model-X guarantees FDR only with a correctly estimated knockoff distribution;
at N≈200–500 that is not automatic, so realized FDR is validated empirically:

```
NULL RUN:
  permute y within each gene (destroy all genotype-phenotype association)
  run map_knockoffs at fdr=q
  every selected CS is by definition false
  empirical_FDR = (# selected CSs) / (# would-be discoveries)  -> compare to q
```

`knockoffs.calibration_report` consumes the per-gene false/total counts and
reports `empirical_fdr`, `target_fdr`, and a `calibrated` bool. **Acceptance
criterion for the whole feature: empirical FDR ≤ q (within noise) on a null
permutation of the real data at the intended N.** If it fails, that is the
small-N generator limit asserting itself, and the fix is more shrinkage / the
HMM generator / more knockoff draws — diagnosed before any real-data use.

A SPIKE-IN power run (plant 1–3 causal variants per gene at known PVE, measure
fraction recovered in a selected CS vs realized FDP) accompanies the null run to
quantify the power cost of calibration.

---

## 6. Deferred / v2 knobs (noted, not built in v1)

- **Within-gene W standardization before pooling.** Draw M knockoffs, estimate
  each gene's null-W spread, convert each CS's W to a within-gene z-score/rank,
  pool the standardized statistics. Equalizes null scale across genes so a clear
  top signal in a weak gene isn't drowned by a strong gene's noise band —
  directly addresses the power-concentration cost in §1. Costs the M draws we
  already take. Build only if the plain pool shows a problematic power skew.
- **HMM generator** (`knockoff='hmm'`), optionally with an external genetic
  recombination map to set the HMM transition rates from real local
  recombination instead of a data estimate (strong prior at small N). SNPknock,
  GPLv3, optional runtime dependency, never vendored. Higher real-data fidelity
  than Gaussian, especially at low MAF / across hotspots.
- **e-value derandomization** (`derandomize_cs`) instead of mean-W aggregation,
  if selection stability across seeds is inadequate.

---

## 7. CLI

**Not built.** This section originally proposed a `cis_susie_knockoffs` mode in
`tensorqtl.py` mirroring `cis_susie` (see the build-order item 5 below); as of
this writing `tensorqtl/tensorqtl.py`'s `--mode` choices are `cis`, `cis_nominal`,
`cis_independent`, `cis_susie`, `trans`, `trans_susie`, `nbqtl-score`,
`hapmixqtl_nominal`, `hapmixqtl`, `hapmixqtl_susie` — there is no knockoff mode,
and no `--fdr`/`--n_knockoffs`/`--knockoff` flags gate anything on the CLI path.
The shipped interface is Python-only: `tensorqtl.susie.map_egenes_knockoffs`
(standard *cis* phenotypes) and `tensorqtl.hapmixqtl.map_egenes_knockoffs`
(two-channel hapmixQTL phenotypes), both returning DataFrames rather than
writing files. See the README section "Knockoff-calibrated eGene FDR" for the
user-facing call pattern (recommended: `statistic='kfc'`, and for the standard
`cis` path `knockoff='gaussian', shrink=0.1`, per the RESOLUTION block at the
top of this document) and `docs/outputs.md` for the `egene_df` column schema.
Adding a CLI mode remains open work, not yet scheduled.

---

## 8. Build order

1. `augmented_susie_fit` + `pooled_cs_qvalues` in `knockoffs.py` (+ unit tests).
2. `map_knockoffs` in `susie.py` wiring PASS 1 / PASS 2 (reuse `susie.map`'s
   per-gene setup verbatim; only the fit + CS extraction change).
3. **Null-permutation calibration test on synthetic data** — the gate. Do not
   proceed to real-data / CLI until empirical FDR ≈ q here.
4. Spike-in power test.
5. CLI mode — **not done** (see §7; still open work).
6. (later) hapmixQTL two-channel path; HMM generator; v2 knobs.

Open question flagged for the two-channel phase (not v1): the augmented fit
doubles a design that, in hapmixQTL, is already a 2N-row whitened stack — the
exchangeability argument there needs the haplotype-level knockoffs (`s̃`), per
the earlier design discussion.
```

---

## Overlap calibration study — first results (record, not validation)

Setting: 150 genes, N=300, 300 SNPs, overlapping cis windows (±120 kb), 40 causal
genes, 3 knockoff draws, target FDR q=0.10, B=8 replicates. Estimator = mean
per-replicate FDP. Genotypes are Gaussian-factor dosages (favorable to the
Gaussian generator; NOT real haplotypes).

| ρ (cross-gene pheno corr) | mode | FDR̂ | se | power | P(R>0) |
|---|---|---|---|---|---|
| 0.0 | per_gene | 0.021 | 0.008 | 0.82 | 0.88 |
| 0.0 | shared   | 0.030 | 0.012 | 0.73 | 0.75 |
| 0.9 | per_gene | 0.039 | 0.027 | 0.74 | 0.75 |
| 0.9 | shared   | 0.050 | 0.044 | 0.73 | 0.75 |

Read: under overlap + near-duplicated phenotypes (ρ=0.9), per-gene knockoffs did
NOT inflate FDR (0.039 < q=0.10); shared knockoffs did not improve on per-gene
(0.050, within noise) and gave slightly lower power. No empirical case here for
preferring the expensive shared/chromosome-wide construction on overlap grounds.

Caveats (this is evidence, not a theorem — cf. review): one setting only, not the
stacked-adversarial regime; large SEs at B=8; Gaussian-friendly genotypes (real
haplotypes / rare variants untested); uniformly conservative (e-BH + detection
floor). The overlapping-gene joint-sign reduction remains the open problem;
status stays "empirically calibrated", not theorem-backed.

TODO for a real validation: >=50 replicates; HMM/real-haplotype genotypes;
stacked adversarial factors (unequal signals, low-freq variants, structure,
covariance misspecification); FDP quantiles not just the mean.

---

## Stability map + derandomization finding (validation, phase 2)

Grid over (n_genes, n_knockoffs), q=0.1, overlapping windows, rho=0.5, B=10.

Complete null: P(R>0)=0.00 in EVERY cell -- no false discoveries under the null
at any scale/draw count. FDR control under the complete null is airtight.

Mixed null+causal: FDR is controlled everywhere (all <= 0.023, FDP90 <= 0.06),
BUT power is erratic and the Ren-Barber e-value DERANDOMIZATION makes it worse:
  - 200 genes: 5 draws -> R=30.8 (sdR=25.2, power 0.59); 15 draws -> R=0 (power 0).
  - 400 genes: sdR=40 (>= meanR) -- bimodal, all-or-nothing selection.
More draws and more genes did NOT stabilize (contradicting an earlier guess);
averaging e-values pulls the aggregate toward zero because near the detection
floor many single-draw knockoff+ thresholds are tau=inf (empty), contributing 0.

Direct comparison at 200 genes / 50 causal / causal_effect=2.5:
  n_knockoffs=1 (NO derandomization, single knockoff+ filter):
    meanR=52 sdR=6.8, power mean=0.95 sd=0.10 min=0.70, FDR=0.079  <- stable & powerful
  vs derandomized (above): unstable, power collapses.

CONCLUSION: the valid knockoff STATISTIC and single-draw knockoff+ FILTER work
well (stable power, controlled FDR). The e-BH derandomization LAYER is
empirically pathological at near-floor eQTL scale -- theory sound, finite-sample
behavior bad. The tradeoff is now: single-draw (stable power, but seed-dependent
gene set) vs e-BH (seed-stable, but power-destroying). Likely fix: a gentler
derandomization (stability-selection / selection-frequency across draws) that
preserves approximate FDR control without e-BH's zero-collapse. To be designed
and validated; until then, default to single-draw with a fixed seed and report
seed-sensitivity, OR use stability selection.

---

## Calibration curve (validation phase 3): single-draw offset=0 is calibrated; e-BH was the problem

Realized FDR vs nominal q, single-draw (no e-BH), 200 genes, 50 causal, B=12:

| nominal q | realized FDR (offset=1) | realized FDR (offset=0) | power |
|---|---|---|---|
| 0.05 | 0.017 | 0.033 | 0.91 |
| 0.10 | 0.069 | 0.079 | 0.93 |
| 0.20 | 0.128 | 0.128 | 0.96 |
| 0.30 | 0.144 | 0.145 | 0.96 |

Findings:
- The severe over-conservatism seen earlier (realized 0.002-0.023 at q=0.1) was an
  e-BH DERANDOMIZATION artifact, NOT the base filter. Single-draw offset=1 is only
  mildly conservative (0.069 at q=0.1); single-draw offset=0 is close to calibrated
  (0.079 at q=0.1, 79% of nominal).
- offset=0 tracks the diagonal better than offset=1 at low q (the +1 in knockoff+
  buys the provable FDR<=q guarantee at the cost of calibration; dropping it gives an
  approximately unbiased FDP estimate). At q>=0.2 the two converge.
- Realized FDR SATURATES at ~0.14: asking q=0.2 or 0.3 does not raise it, because
  power is already ~0.96 (recall ~100%) and there are no more false discoveries to
  admit. This is the desired behavior under the "calibration, not just control"
  goal: low realized FDR at high nominal q here reflects saturated recall, not lost
  power.
- Seed-dependence at 200 genes is modest (SE 0.006-0.016); the instability seen in
  the stability map was also largely an e-BH-near-floor artifact.

DEFAULT RECOMMENDATION: single-draw, offset=0 (calibrated FDP / q-value), e-BH
derandomization OFF. If seed-stability insurance is wanted, aggregate the CALIBRATED
quantity across a few draws (median per-gene q-value), NOT e-values -- this preserves
the calibration demonstrated here. Stability selection is available but trades
calibration for reproducibility and is not needed above the detection floor.

Caveats unchanged: Gaussian-friendly simulated genotypes (real haplotypes untested),
one scale/effect regime, the overlapping-gene joint-sign reduction still open. Near
the detection floor (few genes / weak signal) discovery remains unstable regardless
of offset.

---

## HMM-genotype calibration (validation phase 4): no inflation, but not yet the hardest regime

Calibration curve on HMM (fastPHASE) vs Gaussian-factor genotypes, single-draw
offset=0, 150 genes / 40 causal, B=12:

| q | genotypes | realized FDR | se | power |
|---|---|---|---|---|
| 0.05 | gaussian | 0.068 | 0.017 | 0.96 |
| 0.05 | hmm      | 0.061 | 0.008 | 0.99 |
| 0.10 | gaussian | 0.084 | 0.017 | 0.98 |
| 0.10 | hmm      | 0.072 | 0.006 | 0.99 |
| 0.20 | gaussian | 0.135 | 0.017 | 0.99 |
| 0.20 | hmm      | 0.075 | 0.007 | 0.99 |

Finding: the GAUSSIAN knockoff generator did NOT inflate FDR on realistic
discrete genotypes (rare variants + recombination hotspots). HMM realized FDR is
at or below Gaussian's at every q, well controlled. The reviewer's central
real-data failure mode (Gaussian exchangeability breaking on non-Gaussian
genotypes) did not materialize here.

Caveat (do not over-claim): the HMM simulator's LD is modest (short-range
r^2 ~ 0.1). The regime where Gaussian knockoffs are theoretically most fragile is
STRONG LD (r^2 > 0.5) + VERY rare variants (MAF < 0.01) with a near-singular
covariance -- which this simulator does not strongly reach. HMM realized FDR also
saturates at ~0.075 because power is ~0.99 (recall saturated -> no more nulls to
admit), so part of the "good control" reflects an easy fine-mapping task, not
only generator robustness. So this rules out GROSS Gaussian failure on discrete/
rare/hotspot data; it does not yet test the hardest corner. Next: crank LD
strength + rare-variant fraction in the simulator, and compare against the HMM
knockoff GENERATOR in that corner.

UPDATE (supersedes the "add an HMM generator" next-step above): the HMM/DMC
knockoff generator has since been implemented natively (`knockoffs.dmc_knockoffs`,
`knockoffs.hmm_knockoffs`; an independent reimplementation of Sesia et al. 2019
Alg. 1-2, not the SNPknock package) and VALIDATED swap-exchangeable to p=20
(`tests/test_hmm_knockoffs.py`). What remains for the hardest-corner comparison is
(a) fitting the HMM from real genotypes (Baum-Welch / fastPHASE EM — not built;
on simulated data the true params are known) and (b) wiring it into the per-gene
pipeline. See STATUS (current) at the top of this doc.

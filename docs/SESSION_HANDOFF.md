# Session handoff — hapmixQTL + knockoff-calibrated SuSiE

Last updated: session ending 2026-07 (see git log for exact commits).
Read this first on resumption.

---

## 0. TL;DR / where to pick up

- **hapmixQTL** is done and PR-ready on its own branch, pushed to origin. Nothing
  blocking; open a PR when ready.
- **Knockoff-calibrated SuSiE** is active research on a stacked branch. The last
  milestone: **the HMM/DMC knockoff generator is now VALIDATED** (was falsely
  believed buggy — it was a test artifact). This unblocks chromosome-wide
  coherent knockoffs.
- **Step 1 (wire HMM knockoffs into the pipeline) is DONE** (this session), with
  BOTH exact diploid constructions: **Route 1** `method='genotype'` (exact
  pair-state genotype HMM fit from unphased dosages; `fit_genotype_hmm`,
  `build_genotype_pair_hmm`, `genotype_hmm_knockoffs`) and **Route 2**
  `method='haplotype'` (phased haplotype knockoffs `x̃L,x̃R` summed to a dosage;
  `haplotype_hmm_knockoffs`), plus a cheap `method='single_chain'` approximation.
  All chromosome-coherent, sliced per gene, wired through
  `susie.map_egenes_knockoffs`. Tests: `tests/test_hmm_knockoff_pipeline.py`
  (16 tests). Route 1 is O(N·p·K⁴) (exact, unphased); Route 2 is O(N·p·K²)
  (exact, needs phase, and is what hapmixQTL step 5 will consume).
- **Next task:** build **per-gene knockoff p-values** (step 2 — now UNBLOCKED by
  the coherent generator) and the **interval-valued π₀ empirical-Bayes
  calibration** (step 3) on top.

---

## 1. Branch topology (IMPORTANT)

```
master (origin)
 └─ claude/hapmixqtl-gibbs-uncertainty-IQ6Za   <- hapmixQTL, PUSHED, PR-ready (3 commits)
     └─ claude/susie-knockoff-calibration-IQ6Za <- knockoff research, built ON TOP
```

- The knockoff branch is **stacked on** the hapmixQTL branch (contains all 3
  hapmixQTL commits + ~18 knockoff commits).
- **Both branches are now pushed to origin** (the knockoff branch was local-only
  during the session; it is pushed at handoff so it survives container reclaim —
  a fresh session re-clones from origin and would otherwise lose it).
- The user's rule: develop knockoffs on the knockoff branch; keep the hapmixQTL
  PR branch clean (no knockoff commits on it). This has been honored.

---

## 2. hapmixQTL (branch: claude/hapmixqtl-gibbs-uncertainty-IQ6Za) — DONE

cis-QTL mapping from haplotype-resolved expression posteriors (Salmon Gibbs
draws), propagating inferential uncertainty into beta/beta_se.

- **Files:** `tensorqtl/hapmixqtl.py` (new), CLI wiring in `tensorqtl/tensorqtl.py`,
  README section, `docs/outputs.md`, `tests/test_hapmixqtl.py` (32 tests, all pass).
- **Modes:** `hapmixqtl_nominal`, `hapmixqtl`, `hapmixqtl_susie`.
- **Key design decision:** SEs use **known-variance GLS** (`se = sqrt(1/xx)`),
  NOT estimated-dispersion WLS — this is what makes inferential variance actually
  propagate into beta_se and satisfies the "huge-Va collapses to total-only" test.
- **Also fixed** a real bug in `tensorqtl/susie.py::get_x_attributes`
  (center=False/scale=False were silently ignored) and made `hapmixqtl_susie`
  compute credible-set purity on genotype LD.
- **A ready-to-paste PR description was drafted** in the (ephemeral) scratchpad;
  it is reproduced in §7 below so it is not lost. `gh` CLI is not available in
  this environment and the remote is a proxied mirror, so the PR must be opened
  manually on GitHub.

---

## 3. Knockoff-calibrated SuSiE (branch: claude/susie-knockoff-calibration-IQ6Za) — RESEARCH

Goal: use knockoffs as empirical null controls to control/calibrate the FDR of
SuSiE cis discoveries, robustly to SuSiE's own (often miscalibrated) PIPs.

### 3.1 What is VALID and built

- **eGene-level FDR** — `susie.map_egenes_knockoffs`. Per gene, fit SuSiE on the
  augmented `[X, X_knockoff]` design; gene statistic
  `W_g = max PIP(orig) - max PIP(knockoff)` tests the FIXED hypothesis "gene has
  no cis signal" and IS swap-antisymmetric (verified). Genes selected at
  genome-wide FDR; SuSiE then localizes within selected genes. Empirically:
  ~0.95 recall, FDR controlled, and it catches false CSs SuSiE hallucinates under
  a polygenic background.
- **Default selection:** single knockoff draw, `offset=0` (calibrated FDP /
  q-value), `selection='qvalue'`. Validated: realized FDR ≈ nominal q at operating
  q (calibration, not just control).
- **Knockoff generators** (`tensorqtl/knockoffs.py`):
  - `gaussian_knockoff` — second-order Gaussian, mandatory covariance shrinkage.
    O(p^3). FAST APPROXIMATION, no longer the default (flipped to HMM 2026-07):
    misspecified on non-Gaussian HMM genotypes; the earlier "validated to NOT
    inflate FDR" claim only covered mild regimes and is superseded — on strong-
    LD/rare-variant genotypes it inflates the original-favored false-positive
    tail (see docs/calibration_findings.md).
  - `dmc_knockoffs` / `hmm_knockoffs` — **NOW VALIDATED** (commit 0d329ec).
    Implements Sesia et al. (2019) Algorithms 1-2 (DMC eqs 4-5 + HMM
    forward-backward + re-emit). Swap-exchangeable to p=20 (noise-robust test).
    O(N p K^2), linear in p — the path to chromosome-wide coherent knockoffs.
- **HMM genotype simulator** — `tests/hmm_genotype_simulator.py` (realistic
  discrete genotypes: LD decay, rare variants, recombination hotspots, phased
  xL/xR). The validation test bed.
- **Calibration harness** — `tests/overlap_calibration_harness.py` (overlapping
  cis windows, tunable cross-gene phenotype correlation, per_gene vs shared
  knockoffs, Freedman-Lane shared-permutation null).

### 3.2 What is EXPERIMENTAL / invalid (do not present as FDR-controlled)

- **Credible-set-level FDR** (`map_knockoffs`, `cs_level_W`, `pooled_cs_qvalues`):
  the CS-level statistic is NOT swap-antisymmetric (a real signal's CS
  *disappears* under the swap rather than negating — proven in
  `TestSwapEquivariance`). Retained as an experimental score only. This is why
  we pivoted to eGene-level.
- **e-BH derandomization** (`select_egenes`): valid in theory but empirically
  pathological near the detection floor (averages in zeros from empty draws →
  power collapse). Not the default; kept as an option.

### 3.3 Key empirical findings (recorded in docs/knockoff_susie_design.md)

- Phase 2: e-BH derandomization is pathological near the detection floor.
- Phase 3: single-draw offset=0 q-value is calibrated (realized ≈ nominal q);
  the earlier "5-50x over-conservatism" was an e-BH artifact, not the base filter.
- Phase 4: Gaussian knockoffs do NOT inflate FDR on realistic HMM genotypes
  (rules out gross failure; hardest LD corner still untested).
- Overlap study: per-gene knockoffs did not inflate under overlap + near-
  duplicated phenotypes (ρ=0.9); shared knockoffs bought nothing there. Evidence,
  not proof.
- HMM generator: the "compounding Z-bug" was a TEST ARTIFACT (full-joint swap-TV
  noise floor grows with p). Algorithm was correct.

---

## 4. Open problems / next steps (in priority order)

1. **Wire HMM knockoffs into the pipeline — DONE (this session), both routes.**
   `chromosome_hmm_knockoffs(method=...)` dispatches: **'genotype'** (Route 1,
   default, exact pair-state diploid HMM from unphased dosages —
   `fit_genotype_hmm` constrained Baum-Welch + `build_genotype_pair_hmm`),
   **'haplotype'** (Route 2, exact phased knockoffs — `haplotype_hmm_knockoffs`,
   returns `x̃L,x̃R`), **'single_chain'** (approximate). Wired through
   `susie.map_egenes_knockoffs` with `hmm_method`, `hmm_K`, `hmm_em_iter`,
   `hmm_params`, `phased_haplotypes`. Tests: `tests/test_hmm_knockoff_pipeline.py`
   (16, pass). Compute: Route 1 O(N·p·K⁴), Route 2 O(N·p·K²) — prefer Route 2
   when phased. The Route-2 phased knockoffs are the enabler for step 5.
2. **Per-gene knockoff p-values.** With M coherent draws, the rank of R_g among
   {R_g, K_g^(1..M)} is EXACTLY uniform under the null (exchangeability) →
   per-gene p-value, robust to pooled-f_0 contamination. This is the agreed
   primitive.
3. **Interval-valued π₀ empirical-Bayes calibration.** π₀ (null-gene fraction)
   is only SET-identified (partial identification), not point-identified — this
   is a theorem, not a fixable gap. Correct output: interval-valued local FDR /
   calibrated PIP, whose WIDTH flags un-calibratable genes. Use paired
   real/knockoff asymmetry (N₊-N₋) + f_0-exact to tighten the identified set.
   Report set-level knockoff FDR as the conservative endpoint. (Long design
   discussion in the session; see the last several user messages.)
4. **Overlapping-gene joint-sign reduction** — the central OPEN THEORY problem.
   Empirically un-inflated so far, but not proven. Status label:
   "empirically calibrated", not "theorem-backed".
5. **Two-channel hapmixQTL knockoffs** — for the ASE channel's signed indicator
   `s = xL - xR`, need haplotype-level knockoffs (x̃L, x̃R). ENABLER NOW BUILT:
   `knockoffs.haplotype_hmm_knockoffs(..., return_phased=True)` returns exactly
   (x̃L, x̃R). Remaining work is wiring those into hapmixQTL's two-channel augmented
   design, not the generator.
6. **Hardest-LD-corner test** — crank simulator LD strength + rare-variant
   fraction to r^2>0.5, MAF<0.01 to stress the Gaussian generator.

---

## 5. Test status

- `tests/test_hapmixqtl.py` (32), `tests/test_knockoffs.py` (19),
  `tests/test_hmm_knockoffs.py` (8), `tests/test_hmm_simulator.py` (5),
  `tests/test_hmm_knockoff_pipeline.py` (16, HMM fit + both routes + pipeline) — PASS.
- `tests/test_knockoffs_calibration.py`, `tests/test_knockoffs_egenes.py` — have
  `@pytest.mark.slow` calibration gates (minutes each; end-to-end SuSiE fits).
- `tests/overlap_calibration_harness.py` — a research harness, run directly, not
  a unit test.
- Pre-existing failures unrelated to this work: `test_core.py`, `test_post.py`,
  `test_trans.py` (fail on base master too — API-signature drift in the existing
  suite; NOT introduced here).

Run knockoff/hmm units: `python -m pytest tests/test_knockoffs.py
tests/test_hmm_knockoffs.py tests/test_hmm_simulator.py -q`

---

## 6. Environment gotchas

- `gh` CLI not installed; origin is a proxied local git mirror (no PR web API) —
  PRs must be opened manually on GitHub.
- Proxy blocks many publisher/PDF hosts (403). Raw githubusercontent.com works.
- `snpknock` (the reference knockoff pkg) won't build here (needs Armadillo +
  Cython) — we reimplemented its algorithm natively (license-clean).
- The scratchpad dir is EPHEMERAL (lost on session clear). Anything important was
  moved into committed files or this handoff.
- Full-joint swap-TV is a BROKEN validity metric at large p (noise floor grows
  with p). Use the pairwise-swap or split-half-noise-floor check
  (see tests/test_hmm_knockoffs.py). This cost a lot of debugging time; don't
  repeat it.

---

## 7. hapmixQTL PR description (reproduced from ephemeral scratchpad)

Title: **hapmixQTL: cis-QTL mapping with haplotype-resolved expression uncertainty (+ SuSiE)**

Adds a hapmixQTL mode to tensorQTL: cis-QTL mapping from haplotype-resolved
expression posteriors (Salmon Gibbs draws), propagating inferential (measurement)
uncertainty into beta and beta_se. Two channels (ASE `a` on signed het indicator
`s=xL-xR`; total `t` on half-dosage `g/2`) combined by inverse-variance
meta-analysis; slope is interpretable as log allelic fold change. SEs use
known-variance GLS so inferential variance genuinely propagates into beta_se.
Modes: `hapmixqtl_nominal`, `hapmixqtl` (permutation + beta approx),
`hapmixqtl_susie` (SuSiE fine-mapping of the combined signal). Includes a
one-line correctness fix to `susie.get_x_attributes` (center/scale flags were
ignored; only affects intercept=False callers, i.e. the new code — default
cis_susie path unchanged) and genotype-LD purity for hapmixqtl_susie. 32 tests.
Base `master` <- `claude/hapmixqtl-gibbs-uncertainty-IQ6Za`.

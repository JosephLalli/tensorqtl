# hapmixQTL (ASE) pipeline validation

**Date:** 2026-09-07 · **Harness:** `tests/ase_validation.py` · **Raw results:** `docs/ase_validation_results.json`
**Config:** N = 200 samples, 1,500 replicates/cell (30,000 null p-values per Tier-0 cell)

---

## Headline

> **The default configuration `tau_mode='zero'` is anticonservative, severely so on
> realistic data — up to 107× the nominal type-I error at α = 1e-3 on Gaussian
> simulations, and essentially 100% false-positive rate on count-level simulations.
> The fix already exists in the codebase: `tau_mode='estimate'` restores calibration
> across every condition tested. It should become the default.**

Two secondary conclusions: the unused `Cat` covariance is **safely ignorable** (and we
now know why), and the two-channel model **genuinely beats total-only** at matched
type-I error, so hapmixQTL's core premise holds once the SE is right.

**Externally confirmed (§7).** Benchmarked against the RASQUAL/TReCASE generative model —
one hapmixQTL does not assume — the fixed method is *statistically equivalent to TReCASE*
and 3.7x more powerful than total-count-only. The default, once thresholded on its own
null, collapses to total-only performance: it does not just invalidate p-values, it
discards the entire benefit of the allele-specific channel.

---

## 1. Why this work was needed

The existing suite (`tests/test_hapmixqtl.py`, ~35 tests) is almost entirely:

- **internal consistency** — "does the code compute the formula we wrote" (WLS matches a
  numpy reference, the inverse-variance formula is applied correctly, SE shrinks with n); and
- **recovery** — "can it find a planted causal variant".

Neither class of test can detect an invalid error rate. A method can pass every one of
those 35 tests while producing p-values that are wrong by two orders of magnitude — which
is exactly what was happening. This is the same failure mode the knockoff work hit
(`docs/calibration_findings.md`): green formula tests plus green power tests, masking
broken calibration.

### Three structural facts that motivated the design

| | Fact | Location |
|---|---|---|
| **F1** | `tau_mode='zero'` is the **default**, so weights are `w_i = 1/v_inf_i`. Combined with the known-variance GLS SE (`Var(β) = 1/xx`, no estimated dispersion), the model asserts Gibbs inferential variance is the **only** source of error variance. | `hapmixqtl.py:467, 722` |
| **F2** | `compute_summaries_from_gibbs` returns the a–t inferential covariance `Cat`, and `read_hapmixqtl_inputs` accepts a `cat_bed` — but `calculate_hapmixqtl_nominal` combines channels by **scalar** inverse-variance meta-analysis, which assumes independence. | `hapmixqtl.py:288-310` |
| **F3** | p-values use `get_t_pval(tstat, N-2-n_cov)` — a *t* reference for what is by construction a known-variance *z*. Minor and conservative at large N. | `hapmixqtl.py:698-705` |

F1 predicts anticonservative inflation whenever biological variance exists. F2 predicts
inflation growing with a–t correlation. The tiers test both.

---

## 2. Tier 0 — Null calibration

Factorial sweep: biological SD `σ_bio` × a–t inferential correlation `ρ` × `tau_mode`.
All variants null (β = 0). **The `σ_bio = 0, ρ = 0` cell is the control**: there the model
is exactly true, so it must come out nominal or the harness itself is wrong.

Realized type-I error (ratio to nominal in parentheses), `ρ = 0`, 30,000 p-values/cell:

| tau_mode | σ_bio | α=0.05 | α=0.01 | α=1e-3 | λ_GC |
|---|---|---|---|---|---|
| `zero` (default) | 0.0 | 0.0477 (**0.95×**) | 0.0095 (0.95×) | 0.00083 (0.83×) | 0.98 |
| `zero` (default) | 0.3 | 0.1433 (**2.87×**) | 0.0544 (5.44×) | 0.01347 (**13.5×**) | 1.81 |
| `zero` (default) | 0.6 | 0.3406 (**6.81×**) | 0.2111 (21.1×) | 0.10723 (**107×**) | 4.26 |
| `estimate` | 0.0 | 0.0423 (0.85×) | 0.0079 (0.79×) | 0.00053 (0.53×) | 0.94 |
| `estimate` | 0.3 | 0.0487 (0.97×) | 0.0089 (0.89×) | 0.00063 (0.63×) | 1.01 |
| `estimate` | 0.6 | 0.0489 (0.98×) | 0.0102 (1.02×) | 0.00097 (0.97×) | 1.01 |

**Findings.**

1. **The control is nominal (0.95×, λ = 0.98)** — the harness is trustworthy, and the
   known-variance GLS SE is correct *when its assumption holds exactly*.
2. **Inflation under the default is severe and grows in the tail.** At σ_bio = 0.6 the
   error is 6.8× at α = 0.05 but **107× at α = 1e-3**. This is the worst possible shape:
   eQTL work lives in the tail, so the bulk of the p-value distribution looks only mildly
   off while genome-wide-significant calls are overwhelmingly false.
3. **`tau_mode='estimate'` fully corrects it** — every cell within 0.85–1.02× of nominal,
   λ_GC 0.94–1.01, with no meaningful power cost (Tier 2).
4. **ρ had no detectable effect at any σ_bio** (full grid in the JSON; rows for ρ = 0,
   0.5, 0.9 agree to within Monte-Carlo noise). F2's prediction did **not** materialize —
   see Tier 0b for why.

---

## 3. Tier 0b — Direct test of the channel-independence assumption

Rather than infer F2 from type-I error, measure the quantity the scalar meta-analysis
actually assumes is zero: `corr(β_ASE, β_total)` across 1,500 null replicates.

| ρ (a–t inferential corr) | corr(β_a, β_t) | 95% CI | corr(s, g/2) |
|---|---|---|---|
| 0.0 | +0.0116 | [−0.039, +0.062] | −0.0011 |
| 0.5 | −0.0263 | [−0.077, +0.024] | +0.0008 |
| 0.9 | −0.0172 | [−0.068, +0.033] | +0.0005 |

**Finding: ignoring `Cat` is safe, and there is a clean mechanism for it.** Even when the
underlying a–t noise is correlated at ρ = 0.9, the two *slope estimators* remain
uncorrelated (every CI covers 0). The reason is in the last column: the ASE predictor
`s` (signed het indicator) is **orthogonal** to the total predictor `g/2`, because phase
is random with respect to expression — `E[s | g=1] = 0`. The two channels project the
correlated noise onto orthogonal directions, so the correlation does not propagate into
the combined estimate.

This retires F2. `Cat` is genuinely unnecessary for the current statistic — a design
property worth documenting rather than a latent bug. **Caveat:** the argument depends on
phase being random w.r.t. expression. Systematic phasing error correlated with expression
would break the orthogonality; that is untested (see §9).

---

## 4. Tier 1 — CI coverage of a known log aFC

The sharpest probe of whether inferential uncertainty propagates into `beta_se` correctly.
True β = 0.5; nominal 95% CI coverage.

| tau_mode | σ_bio | coverage (target 0.95) | bias |
|---|---|---|---|
| `zero` | 0.0 | **0.947** | −0.0003 |
| `zero` | 0.3 | **0.842** | +0.0018 |
| `zero` | 0.6 | **0.645** | −0.0019 |
| `estimate` | 0.0 | 0.987 | −0.0022 |
| `estimate` | 0.3 | 0.979 | −0.0003 |
| `estimate` | 0.6 | 0.959 | −0.0044 |

**Findings.** The point estimate is **unbiased in every cell** (|bias| ≤ 0.004) — the
estimator is fine; it is purely the *uncertainty* that is wrong. Under the default,
coverage collapses to 0.645, i.e. a nominal 95% interval is really a 65% interval.
`tau_mode='estimate'` restores coverage, mildly **conservative** (0.96–0.99) because
estimating τ when the true τ = 0 adds upward noise — an acceptable trade.

---

## 5. Tier 2 — Does the two-channel model actually beat total-only?

Each method's threshold is calibrated on **its own null distribution**, then power is
compared at matched *empirical* type-I error (α = 0.05). This is the only fair comparison
when one method may be anticonservative — on a matched *nominal* threshold an
anticonservative method always "wins" for the wrong reason.

| MAF | power, combined | power, total-only |
|---|---|---|
| 0.15 | **1.000** | 0.825 |
| 0.35 | **1.000** | 0.960 |

**Finding: hapmixQTL's core premise holds.** The ASE channel adds real power over
total-expression-only at matched type-I error, and the gain is largest at low MAF where
total-only is weakest. This is the result that justifies the method existing — and note
it is only meaningful *because* Tier 0 established a configuration in which both arms are
calibrated.

---

## 6. Tier 3 — Semisynthetic (counts → Gibbs draws → real summary code)

The most realistic tier: simulate allele-level NB counts, emulate a quantifier posterior
by multinomial resampling, and feed the **real** `compute_summaries_from_gibbs`, so
`Va`/`Vt`/`Cat` and all κ-pseudocount behaviour come from production code rather than
from assumption. 300 replicates/cell.

| tau_mode | depth | α=0.05 | λ_GC | mean corr(a,t) |
|---|---|---|---|---|
| `zero` (default) | 20 | **0.9967 (19.9×)** | ∞ | +0.000 |
| `zero` (default) | 100 | **1.0000 (20.0×)** | ∞ | −0.000 |
| `estimate` | 20 | 0.0500 (1.00×) | 1.00 | +0.000 |
| `estimate` | 100 | 0.0367 (0.73×) | 0.77 | −0.000 |

**Finding — this is the alarming one.** Under the default, **essentially every null test
is significant** (λ_GC = ∞ because p-values underflow to 0). The mechanism is exactly
F1 and it is worse here than in Tier 0 for a structural reason: a quantifier's posterior
variance captures only **allelic-assignment uncertainty conditional on the observed
total**. It does not contain the counts' own sampling variance, and it certainly does not
contain biological variance. So `v_inf` understates true error variance by a large factor,
weights blow up, `xx` blows up, and `Var(β) = 1/xx` collapses toward zero.

Because this is the regime real Salmon/mmseq input actually occupies, **`tau_mode='zero'`
should be considered invalid for real data, not merely imprecise.**

Note also `corr(a,t) ≈ 0.000` from the real code path — independent confirmation of the
Tier 0b conclusion on realistically generated summaries.

---

## 7. External benchmark — the RASQUAL / TReCASE generative model

**Harness:** `tests/ase_external_benchmark.py` · **Raw:** `docs/ase_external_benchmark.json`

Every tier above simulates from hapmixQTL's *own* assumed model. That is circular: the
simulator and the estimator share a worldview, so it can show internal inconsistency but
never that the assumptions are wrong about real data. This section removes the circularity
by generating from the model the ASE field actually uses — and which hapmixQTL does **not**
assume:

```
total counts     T_i ~ NegBinomial(mean = lib_i · μ · f(g_i, κ),  dispersion φ)
allele-specific  y_i ~ BetaBinomial(n_i,  π = κ/(1+κ),  overdispersion ρ)
f(g) = 1, (1+κ)/2, κ   for g = 0, 1, 2      # standard TReC cis parameterization
```

κ is the allelic fold change (κ = 1 is the null, where π = 0.5). This is the model
structure of [Sun 2012, *Biometrics*](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3218220/)
(TReCASE) and [Kumasaka et al. 2016, *Nat Genet*](https://www.nature.com/articles/ng.3467)
(RASQUAL). Three comparator tests are implemented directly here — no R or C dependency:
**TReC-only** (NB GLM, LRT), **ASE-only** (beta-binomial, LRT), and **TReCASE** (joint
likelihood sharing one κ, LRT).

> **Scope, honestly.** This reproduces the published model *structure* and implements the
> published *tests*. It is **not** a replication of either paper's exact parameter grid:
> PMC, nature.com, bioRxiv and the asSeq docs were all unreachable through this
> environment's egress proxy, so the depths, overdispersions and effect sizes are our own
> realistic RNA-seq choices, stated explicitly rather than inherited. External validity
> comes from the model family and the comparator tests, not from matching a table.
> RASQUAL's additions over TReCASE (genotype uncertainty, mapping bias φ, sequencing error
> δ) are nuisance refinements on the same joint likelihood; we implement the shared
> TReCASE core, not RASQUAL itself.

**Null calibration** (κ = 1; N = 200, μ = 200, NB disp 0.2, BB overdisp 0.01, AS fraction
0.25, 500 loci):

| Method | type-I @ 0.05 | @ 0.01 | λ_GC |
|---|---|---|---|
| TReC-only | 0.0500 (1.00×) | 0.0080 (0.80×) | 0.91 |
| ASE-only | 0.1540 (3.08×) | 0.0380 (3.80×) | 1.95 |
| TReCASE (joint) | 0.0540 (1.08×) | 0.0120 (1.20×) | 1.18 |
| hapmixQTL `tau='zero'` | **1.0000 (20×)** | **1.0000 (100×)** | **3020** |
| hapmixQTL `tau='estimate'` | 0.0460 (0.92×) | 0.0160 (1.60×) | 1.13 |

TReC-only landing at exactly 1.00× validates the comparator implementation. TReCASE shows
mild residual inflation (λ = 1.18) from the χ²(1) LRT asymptotic at N = 200; ASE-only is
genuinely anticonservative here (3.08×), which is why the matched-α comparison below is the
only fair one. hapmixQTL with `tau='estimate'` (0.92×, λ = 1.13) is **better calibrated
than either joint comparator**.

> *Revision note:* the first run of this benchmark used unbounded Nelder-Mead for the
> comparator likelihoods, which blew up under perfect separation (a null beta-binomial LRT
> statistic of 2.8e5 at N = 50). The LRTs are now bounded, multi-start L-BFGS-B. The
> hapmixQTL-vs-TReCASE conclusion is unchanged; the ASE-only calibration figure moved from
> 1.28× to 3.08× because better optimization revealed inflation the under-converged fit had
> masked.

**Power at matched empirical α = 0.05** (each method thresholded on its *own* simulated
null, so an anticonservative method cannot win by being broken):

| κ (log aFC) | TReC-only | ASE-only | TReCASE | hapmixQTL `estimate` | hapmixQTL `zero` |
|---|---|---|---|---|---|
| 1.05 (0.049) | 0.104 | 0.162 | 0.274 | **0.268** | 0.094 |
| 1.10 (0.095) | 0.204 | 0.558 | 0.760 | **0.762** | 0.194 |
| 1.20 (0.182) | 0.486 | 0.994 | 0.998 | **0.998** | 0.462 |

### Two conclusions

**1. Fixed hapmixQTL is statistically equivalent to TReCASE.** On TReCASE's own home turf —
a generative model hapmixQTL does not assume — `tau='estimate'` matches the published joint
likelihood at every effect size (0.268 vs 0.274, 0.762 vs 0.760, 0.998 vs 0.998), while
beating total-count-only by **3.7×** at κ = 1.10. The method's premise survives external
scrutiny, and §5's total-only comparison was not flattering itself.

**2. The default does not merely invalidate p-values — it destroys the method's entire
advantage.** Look at the `tau='zero'` row: its *uncorrected nominal* power is 1.000 at every
effect size, which looks spectacular. But thresholded on its own null it collapses to
0.094 / 0.194 / 0.462 — **statistically indistinguishable from TReC-only**, the very
baseline the ASE channel is supposed to beat. Once you account for its inflation, the
broken weighting throws away all the information the allele-specific channel contributes.
This is the sharpest available argument that `tau_mode='zero'` is not a conservative-ish
default worth keeping for compatibility: it is strictly worse than not using ASE at all.

It is also a clean demonstration of why the matched-α methodology in §5 matters. On nominal
p-values the broken configuration is the best method in the table.

## 7b. Under the PUBLISHED simulation designs

**Harness:** `tests/ase_published_designs.py` · **Raw:** `docs/ase_published_designs.json`

With the papers in hand the parameter choices are no longer ours. Design taken from source:
mixQTL's allelic-fold-change grid (1, 1.01, 1.05, 1.1, 1.25, 1.5, 2, 3; 200 replicates),
RASQUAL's sample-size grid (N = 25, 50, 100) and RASQUAL's metric — **power at empirical
FPR = 10%**, where the null is the permuted/simulated null rather than a nominal threshold.

### The provenance of the bug

mixQTL's error terms (Liang et al. 2021, main-text Eqs. 3–4) are

```
eps_asc ~ N(0, sigma^2 · (1/Y1 + 1/Y2))       z_tilde ~ N(0, sigma0_tilde^2)
```

The counts set only the **shape** of the weights. `sigma^2` and `sigma0_tilde^2` are **free
scale parameters**, and Supplementary Notes §5.2 — titled *"Inferring σ̃₀² and σ²"* — solves
for both from the data under a mixed/random-effect model (via the R package EMMA).

**hapmixQTL replaced that freely-scaled variance with the Gibbs `v_inf` treated as fully
known — dropping the free scale entirely.** That is exactly the `tau_mode='zero'` defect.
So the bug is a *deviation from the parent method*, and `tau_mode='estimate'` is a
re-derivation of what mixQTL always did. Two further deviations from mixQTL's shipped
implementation (`R/mixqtl.R`): hapmixQTL has no `weight_cap` (mixQTL caps the max/min
weight ratio at `min(100, floor(N/10))`) and no `trc_cutoff=20` / `asc_cutoff=5` filters.

Note the variance models are not identical, which matters:

| | form | says |
|---|---|---|
| mixQTL | `Var = σ²·(1/Y₁ + 1/Y₂)` | multiplicative — "shape right, scale wrong" |
| hapmixQTL | `Var = v_inf + τ` | additive — "an extra independent component" |

A multiplicative scale suits misspecified quantification noise; an additive offset suits
biological variance, which does not shrink with read depth. Both were tested.

### Results — power at empirical FPR 10%

| N | aFC | trcQTL | ascQTL | TReCASE | hapmix `zero` | + weight cap | hapmix `estimate` | nested |
|---|---|---|---|---|---|---|---|---|
| 25 | 1.10 | 0.125 | 0.155 | 0.240 | 0.130 | 0.130 | **0.255** | 0.265 |
| 25 | 1.25 | 0.205 | 0.580 | 0.720 | 0.230 | 0.230 | **0.680** | 0.670 |
| 50 | 1.25 | 0.300 | 0.900 | 0.940 | 0.285 | 0.285 | **0.925** | 0.910 |
| 100 | 1.05 | 0.165 | 0.320 | 0.390 | 0.135 | 0.135 | **0.375** | 0.335 |
| 100 | 1.10 | 0.255 | 0.540 | 0.620 | 0.200 | 0.200 | **0.615** | 0.575 |
| 100 | 1.25 | 0.535 | 0.995 | 1.000 | 0.495 | 0.495 | **1.000** | 1.000 |

Nominal type-I error under the null (before matching) — `zero` and `+weight cap` are
**1.0000 (20×)** at every N; `estimate` is 0.60–0.90× and TReCASE 0.70–1.20×.

### Four conclusions

1. **mixQTL's weight cap does NOT fix it.** `+weight cap` is *identical* to `tau='zero'` at
   every N and every effect size — same 20× type-I, same power. The cap bounds the *ratio*
   between weights, but hapmixQTL's failure is that **all** weights are uniformly too large
   (`v_inf` uniformly understates the variance), so capping the ratio changes nothing. The
   free scale parameter is the load-bearing part, not the guardrail.
2. **The equivalence to TReCASE holds across the published grid**, at every sample size,
   with hapmixQTL `estimate` typically within a few points of TReCASE and occasionally
   marginally below it (0.680 vs 0.720 at N = 25, aFC 1.25).
3. **`tau='zero'` tracks trcQTL almost exactly** at matched FPR (0.495 vs 0.535 at N = 100,
   aFC 1.25) — independent confirmation on the published design that the broken default
   discards the allele-specific channel entirely.
4. **The nested `σ²·v_inf + τ` model buys nothing** over plain additive τ (0.575 vs 0.615 at
   N = 100), and is marginally worse at larger N. The simple additive offset is sufficient;
   the extra scale parameter is not worth the complexity.

### Consistency with the published numbers

RASQUAL reports simulation power at FPR 10% of 35.5% (N = 25), 46.3% (N = 50), 55.9%
(N = 100), and real-data eQTL power of RASQUAL 42.2% > CHT 35.7% ≈ **TReCASE 35.5%** >
Lm 25.7% (25 EUR). Our TReCASE numbers bracket their figures in the aFC 1.10–1.25 range,
which is where real eQTL effects sit — the same ballpark, as intended. Two honest caveats:
RASQUAL draws its effect sizes from *empirical distributions estimated from real data*
rather than a fixed grid, so a point-to-point match is not meaningful; and our
TReCASE-over-total-only advantage (≈3.5×) is larger than their TReCASE-over-Lm (1.38×)
because our simulation has a cleaner allele-specific signal than real data. The ordering —
joint > AS-only > total-only, and RASQUAL > TReCASE — reproduces.

**Implication for expectations:** RASQUAL beats TReCASE by ~19% relative on real data
(42.2% vs 35.5%) via genotype uncertainty, mapping bias φ and sequencing error δ. Since
fixed hapmixQTL ≈ TReCASE, it should be expected to sit ~19% below RASQUAL on real data
until it models those nuisance terms.

## 7c. The cis/trans test — an assumption hapmixQTL makes but never checks

**Harness:** `tests/ase_cis_trans_test.py` · **Raw:** `docs/ase_cis_trans.json`

Sun (2012) and its descendants let the two channels have their own effect sizes and test
whether they agree. In CSeQTL's notation (Little et al. 2023, *Nat Commun* 14:3030,
Methods, "cis/trans eQTL testing"):

```
eta^(A) = eta^(T) * alpha        cis <=> alpha = 1        H0: alpha = 1
```
where `eta^(T)` is the effect from total read count and `eta^(A)` from allele-specific.

**hapmixQTL's inverse-variance meta-analysis assumes `alpha = 1`.** That is the entire
justification for putting both channels on a common log-aFC scale (the `g/2` predictor) and
averaging them — and it is never tested. When `alpha != 1` the meta-analysis averages two
*different estimands*. `alpha != 1` is not exotic: a trans component acting on total
expression only, reference mapping bias attenuating the ASE channel, systematic phasing
error, or feature-level misquantification all produce it.

Because §3 established the two slope estimators are **uncorrelated**, the difference has
variance `se_a^2 + se_t^2` with no covariance term, so a simple Wald test is valid:

```
T = (beta_a - beta_t) / sqrt(se_a^2 + se_t^2)  ~  N(0,1) under H0 (cis)
```

The same orthogonality result that let us retire `Cat` is what makes this diagnostic cheap.

**The test is calibrated** under true cis (N = 200, 800 reps): z has mean ≈ +0.04 and
sd 0.99 / 0.96 / 0.88 at beta = 0 / 0.3 / 0.6, with type-I error 1.25× / 1.02× / 0.70× at
alpha = 0.05 — mildly conservative at large shared effects.

**What it catches, and what it costs not to have it** (beta_t fixed at 0.4):

| alpha | detection rate | combined slope reported | bias vs true beta_t |
|---|---|---|---|
| 1.00 (true cis) | 0.040 | 0.3991 | −0.0009 |
| 0.75 | 0.116 | 0.3209 | −0.0791 |
| 0.50 | 0.379 | 0.2399 | −0.1601 |
| 0.25 | 0.711 | 0.1576 | −0.2424 |
| 0.00 (pure trans) | 0.919 | 0.0759 | **−0.3241** |
| −0.50 (opposing) | 1.000 | −0.0787 | **−0.4787 (sign flip)** |

**Finding.** For a gene whose eQTL is *not* purely cis, hapmixQTL silently reports a badly
attenuated effect. At `alpha = 0` — a real total-expression effect with no allele-specific
component, i.e. a trans-eQTL — it reports **0.076 instead of 0.400, an 81% attenuation**,
with no warning. At `alpha = -0.5` the reported effect **changes sign**. The cis/trans test
detects these 92% and 100% of the time respectively.

Note the honest limitation: detection power is weakest (11.6%) exactly where the bias is
mildest but not negligible (alpha = 0.75 still attenuates by 20%). The test is a screen for
gross violations, not a guarantee of consistency.

This also gives hapmixQTL something it currently cannot do: **distinguish cis from trans
eQTLs**, which is one of the headline capabilities of the TReCASE family.

## 7d. REAL DATA — GTEx v8 phASER haplotype expression

**Harness:** `tests/ase_gtex_real_data.py` · **Raw:** `docs/ase_gtex_real_data.json`

Every tier above is simulation. This is the real thing: GTEx v8 haplotype-expression
matrices produced by phASER (Castel et al. 2016) from the public
`gs://adult-gtex/haplotype-expression/v8/` bucket — genuine per-gene, per-sample
haplotype counts `yL | yR`, which is exactly hapmixQTL's input. **400 genes × 706
Muscle-Skeletal samples**, median allele-specific depth 54 reads.

**Design.** GTEx genotypes are dbGaP-protected, so real cis-QTL mapping is not possible
from public data. But the more important check is: real haplotype counts supply the real
variance structure — genuine overdispersion, depth distribution, zero inflation,
biological variability — and genotypes drawn *independently of expression* make every test
a true null. This is the same logic as RASQUAL's permutation null. Any method whose
p-values are not uniform here is miscalibrated on real data, whatever simulations say.

| matrix | tau_mode | type-I @0.05 | @0.01 | @1e-3 | λ_GC |
|---|---|---|---|---|---|
| phASER | `zero` | **1.0000 (20×)** | **1.0000 (100×)** | **1.00000** | **3020** |
| phASER | `estimate` | 0.0533 (1.07×) | 0.0117 (1.17×) | 0.00167 | 0.98 |
| phASER + WASP | `zero` | **0.9992 (20×)** | **0.9992 (100×)** | **0.99917** | **3020** |
| phASER + WASP | `estimate` | 0.0583 (1.17×) | 0.0117 (1.17×) | 0.00083 | 0.90 |

**Findings.**

1. **The defect is confirmed on real data, at full severity.** On real GTEx haplotype
   counts, `tau_mode='zero'` makes **every single null test significant** (λ_GC = 3020).
   This is no longer an inference from simulation.
2. **The fix works on real data.** `tau_mode='estimate'` gives 1.07× nominal type-I error
   and λ_GC = 0.98 — properly calibrated on genuine GTEx expression variance.
3. **Calibration is insensitive to reference mapping bias.** The WASP-corrected matrix
   gives essentially the same answer (1.17×, λ = 0.90). hapmixQTL does not model reference
   bias at all — it has no analogue of RASQUAL's φ — so this is reassuring: its validity
   does not depend on upstream WASP correction. (This addresses calibration only; mapping
   bias could still bias effect *sizes*, which needs genotypes to test.)

## 7e. Compute cost

**Harness:** `tests/ase_compute_benchmark.py` · **Raw:** `docs/ase_compute_benchmark.json`

RASQUAL reports cost as a headline result (539.9 CPU-days vs TReCASE 4.6 vs Lm 0.4), and
mixQTL's entire framing is that likelihood-based joint methods are intractable at scale.
hapmixQTL inherits that log-linear framing, so its cost is a claim worth measuring.
N = 500, 2,000 variants/gene; genome-wide = 20,000 genes × 2,000 variants = 40M tests,
single core.

| method | sec/test | relative | CPU-days genome-wide |
|---|---|---|---|
| total-only OLS (Lm) | 3.2e-06 | 0.2× | 0.00 |
| hapmixQTL `tau='zero'` | 1.1e-05 | 0.6× | 0.01 |
| **hapmixQTL `tau='estimate'`** | **1.7e-05** | **1.0×** | **0.01** |
| TReC-only (NB GLM, LRT) | 1.1e-02 | 617× | 4.89 |
| TReCASE (joint LRT) | 5.9e-02 | **3446×** | 27.32 |

**Finding: hapmixQTL is ~3,400× faster than TReCASE at statistically equivalent power**
(§7, §7b). That is the trade the log-linear approximation buys, now measured rather than
asserted, and it is the strongest argument for the method's existence alongside its
calibration. The τ fix costs ~60% more than the broken default — negligible in absolute
terms (0.01 CPU-days either way).

*Caveat:* our TReCASE is a reference implementation in Python/scipy, not asSeq's optimized
C, so 3,446× is an upper bound on the true ratio. Our 27.3 CPU-days versus RASQUAL's
published 4.6 for TReCASE is consistent with roughly a 6× implementation penalty; even
correcting for it, hapmixQTL remains ~500× faster.

## 8. Recommendations

1. **Change the default to `tau_mode='estimate'`** in `map_nominal`, `map_cis`, and
   `map_susie`. This is a one-line change per call site and is the single highest-value
   fix identified. Deliberately left as a separate decision rather than applied here,
   since it changes published behaviour. §7b strengthens this: the free scale parameter is
   not an optional refinement but the load-bearing part of mixQTL's model, which hapmixQTL
   dropped. Adding mixQTL's `weight_cap` instead does **not** work (tested; identical to
   the broken default), and the more elaborate nested `σ²·v_inf + τ` buys nothing over
   plain additive τ.
2. **Warn (or refuse) on `tau_mode='zero'`.** If it is kept for backward compatibility, it
   should emit a loud warning: it is only valid when inferential variance is provably the
   entire error variance, which real quantifier posteriors never satisfy.
3. **Document `Cat` as intentionally unused**, with the orthogonality argument from Tier
   0b, so a future reader does not mistake it for an oversight — or silently "fix" it.
4. **Add a calibration gate to CI.** A cheap version of Tier 0 (control cell + one inflated
   cell) as a fast test would have caught this immediately.
5. **Ship the cis/trans test (§7c)** as a per-gene diagnostic column. It is ~10 lines given
   what `calculate_hapmixqtl_nominal` already returns, it validates the assumption the
   meta-analysis rests on, and it adds cis-vs-trans classification. Genes failing it should
   be flagged rather than silently reported with an attenuated effect.

## 9. Untested / open

- **Phasing error.** Tier 0b's orthogonality argument assumes phase is random w.r.t.
  expression. Systematic, expression-correlated phasing error would break it. Untested.
- **Real ASE data.** Every tier here is simulation. The strongest real-data check needs no
  external truth: because the total channel uses `g/2`, both channels estimate the *same*
  quantity, so regressing `slope_a` on `slope_tc` across real variants should give slope 1.
  Deviation localizes bias to a channel. Blocked on real allele-count + Gibbs input.
- **Covariates.** All tiers run with no covariates; the residualizer path is exercised by
  the existing unit tests but not by a calibration sweep.
- **`se_mode='robust'`.** The sandwich SE may be self-correcting for F1; untested here.

## Reproduce

```bash
python3 tests/ase_validation.py --reps 1500 --N 200 --tiers 0,1,2,3 --out results.json
# fast smoke (~1 min):
python3 tests/ase_validation.py --reps 60 --tiers 0
```

## Hosting the annotatable report

`docs/ase_validation_report.html` embeds [Hypothesis](https://web.hypothes.is/) for inline
annotation. Hypothesis needs the page served over http(s) — it will not load from a
`file://` path, and the Claude Artifact sandbox blocks the script at the CSP layer. To get
working inline commenting, deploy to Cloudflare Pages:

```bash
./scripts/build_site.sh
npx wrangler login    # one-time OAuth (or set CLOUDFLARE_API_TOKEN + CLOUDFLARE_ACCOUNT_ID)
npx wrangler pages deploy site --project-name=hapmixqtl-calibration-audit
```

Lands at `https://hapmixqtl-calibration-audit.pages.dev`. Config in `wrangler.toml`;
`site/` is generated and gitignored, so `docs/` stays the single source of truth.

> Note: this deploy cannot be run from a Claude Code remote session — the egress policy
> blocks `api.cloudflare.com`, and no Cloudflare credentials are present. Run it locally.

### Making it private by default

A `pages.dev` URL is world-readable by default. Two separate things have to be locked
down, and they are protected by two different mechanisms.

**1. The page — Cloudflare Access** (Zero Trust; free up to 50 users).

> **The gotcha:** protecting `hapmixqtl-calibration-audit.pages.dev` is *not enough*. Every
> deploy also gets its own immutable preview URL
> (`<hash>.hapmixqtl-calibration-audit.pages.dev`) which stays publicly reachable forever
> unless protected separately. Cover both.

- **Preview deployments** — Pages has a native toggle:
  `Pages → project → Settings → General → Enable Access policy`. This is the one people
  miss. (Cloudflare moves its UI around; if it is not under General, look under the
  project's Deployments settings.)
- **Production** — `Zero Trust → Access → Applications → Add → Self-hosted`, with
  application domain `hapmixqtl-calibration-audit.pages.dev`, an **Allow** policy including
  either specific emails or an email domain (e.g. `wisc.edu`), and **One-time PIN** as the
  identity method — no IdP or SSO setup required.
- **Ordering is the control.** Cloudflare has no "create as private" flag, so create the
  project and apply Access *before* the first `pages deploy`; otherwise there is a window
  where the report is open.

**2. The commentary — a Hypothesis private group.**

Cloudflare Access does **not** make annotations private. Annotations live on Hypothesis's
servers, not on the Pages origin, and public annotations are world-readable through the
Hypothesis API — *including the quoted text of the passage each one anchors to*. Someone
could therefore read collaborators' comments, and the excerpts being commented on, without
ever passing the Access gate.

Fix: create a private group (`hypothes.is → Groups → Create new private group`), share the
join link with collaborators, and have everyone select that group in the sidebar before
annotating. Access gates the page; the private group gates the commentary. **Both are
needed.**

Also note annotations are anchored to the exact URL, so moving the report to a custom
domain later will orphan existing annotations.

### Simpler alternatives

- **Local only:** `cd site && python3 -m http.server` — Hypothesis does load over
  `http://localhost`. Fine for reading it yourself; useless for collaborators, since
  localhost URLs are per-machine and annotations will not be shared.
- **GitHub Pages on a private repo** requires a paid GitHub plan, so it is not a free
  shortcut for private hosting.

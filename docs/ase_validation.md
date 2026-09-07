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
would break the orthogonality; that is untested (see §7).

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

## 7. Recommendations

1. **Change the default to `tau_mode='estimate'`** in `map_nominal`, `map_cis`, and
   `map_susie`. This is a one-line change per call site and is the single highest-value
   fix identified. Deliberately left as a separate decision rather than applied here,
   since it changes published behaviour.
2. **Warn (or refuse) on `tau_mode='zero'`.** If it is kept for backward compatibility, it
   should emit a loud warning: it is only valid when inferential variance is provably the
   entire error variance, which real quantifier posteriors never satisfy.
3. **Document `Cat` as intentionally unused**, with the orthogonality argument from Tier
   0b, so a future reader does not mistake it for an oversight — or silently "fix" it.
4. **Add a calibration gate to CI.** A cheap version of Tier 0 (control cell + one inflated
   cell) as a fast test would have caught this immediately.

## Untested / open

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

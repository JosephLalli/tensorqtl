# Open investigations: proposed work, not results

2026-09-20. This file holds work that is **proposed and not yet run**. It is
deliberately separate from `CLAUDE.md`, which records observed facts, and from
`docs/CURRENT_SCIENTIFIC_STATE.md`, which routes between things already
established. Nothing below has been measured. When an item is run, its result
belongs in a report and its conclusion in `CLAUDE.md`; this file then loses the
item rather than accumulating it.

Two items, in priority order. The first closes an open question; the second is
a correctness cleanup with no open question attached.

---

## 1. Per-channel residual sigma: the last untested mechanism of the six

**Status.** Proposed, not started. The sole survivor of the six candidate
mechanisms in
`/mnt/ssd/lalli/brainvar_hapmix_deploy/mixqtl_replication_20260919/DISAGREEMENT_TEST_PLAN.md`;
mechanisms 2, 3, 5 and 6 were settled by `scripts/nonweighting_ladder.py` and
the two pre-checks, and mechanism 1 is a router rather than a hypothesis.

### The question

hapmixQTL and mixQTL, run on the same 29 genes and the same donors, agree on
the effect estimate only moderately: Pearson correlation of beta is 0.562 at
mixQTL's published gates and 0.697 at the R signature's more permissive
defaults. The non-weighting ladder attributed most of the allelic-channel gap
to the **donor set** — specifically mixQTL's upper cap on allele-specific
counts, which discards well-expressed donors that Salmon posterior means
place above the threshold.

Note the ladder's own scope before reusing its numbers: it was run **once,
under the permissive gates**, where the upper cap is `y <= 5000` and removed
635 of 2,193 informative donor-gene pairs, and it has **not** been re-run
under the published gates, where the cap is `y <= 1000` and removes 1,656.
The mechanism is the same and is larger at the published gates, but the
ladder's specific rung-by-rung figures belong to the permissive setting.

What that ladder cannot explain is any disagreement that survives
in the **combined** estimate while both per-channel comparisons already agree
well. Mechanism 4 is the candidate for that residue, and it is the only one of
the six that is a pure inference-layer difference: no difference in which
donors are used, which response is modelled, or how the counts are handled.

### What each estimator actually does

Both estimators produce two effect estimates per variant — one from the
**allelic channel** (the log ratio of the two haplotypes' counts, regressed on
the phased genotype contrast through the origin) and one from the **total
channel** (log total expression regressed on genotype dosage with covariates) —
and both then reduce those two numbers to one.

hapmixQTL combines them as a **pooled score statistic**: with `xy` the
cross-product of design and response and `xx` the design's sum of squares in
each channel,

    beta_combined = (xy_a + xy_t) / (xx_a + xx_t)

using **known-variance** standard errors, meaning the per-donor weights are
taken as given from the Gibbs draw variance and residual scale rather than
rescaled by how well the fit turned out.

mixQTL combines them by **scalar inverse-variance meta-analysis**: each channel
contributes an estimate weighted by the reciprocal of its squared standard
error, and the combined estimate is the weighted mean,

    beta_combined = (beta_a / se_a^2 + beta_t / se_t^2) / (1 / se_a^2 + 1 / se_t^2)

Crucially, mixQTL fits a **separate residual sigma per channel** — an estimate
of the residual spread computed from that channel's own fit, on `n - 1` degrees
of freedom for the through-origin allelic channel and `n - 2` for the total
channel — and each channel's standard error carries its own sigma.

### Why this is the only candidate left

The original framing of mechanism 4 was that the two *combination rules*
differ. **That framing is wrong and is superseded.** A pre-check proved the
rules are algebraically identical: under the known-variance standard error
`se = 1/sqrt(xx)`, the inverse-variance weight is `w = 1/se^2 = xx`, so
inverse-variance meta-analysis reduces to the pooled score exactly. Verified
numerically to a maximum absolute deviation of **1.7e-16 over 100,000 random
inputs**, and still exact when a *single* fitted sigma multiplies both
channels.

The rules separate only when each channel carries its **own** fitted scale,
where the pre-check measured them correlating at 0.856. So a per-channel fitted
sigma is, by elimination, the only inference-layer difference that can move the
combined estimate. That is what makes this worth running: if it holds, the two
methods disagree for a reason that has nothing to do with the data and
everything to do with how each channel's uncertainty is estimated.

### The test

From `DISAGREEMENT_TEST_PLAN.md`'s combination design: **4 fits per gene**, a
2 x 2 on identical per-channel inputs — pooled score against inverse variance,
crossed with known-variance standard errors against per-channel fitted sigmas.
The first factor is a **null control**: given the pre-check it must not move
anything, and if it does, the recomputation is wrong and every number is void.

A validation gate precedes it. The test needs hapmixQTL's per-channel estimates
recomputed rather than read from shipped output, and before any result is
believed the recomputed channels must reassemble into `map_nominal`'s shipped
`slope` and `slope_se` to a relative 1e-10 on all matched variants.

Every cell is a deterministic closed-form weighted refit: no Monte Carlo, no
null draw, no permutation, no seed. Differences between cells are exact for
this dataset. Measure per cell the Pearson correlation of the cell's beta
against mixQTL's, the median |beta| ratio and the median standard-error ratio,
each computed per gene and summarised across the 29 genes, with a paired sign
test across genes for direction.

### Refutation criterion, already written

> Mechanism 4 is refuted if replacing known-variance standard errors with
> per-channel fitted sigmas leaves the combined beta essentially unchanged.

### Why it matters beyond the comparison

This is not only a bookkeeping question about two implementations. The
known-variance standard error is exactly the form the weighting ablation found
anticonservative by a factor of 1.82 (calibration ratio 3.313 against a
well-calibrated 1.033 with a fitted scale). If per-channel fitted sigmas also
move the combined point estimate, then the fitted-scale change already
indicated on calibration grounds has a second consequence that must be measured
before it is adopted, not after.

---

## 2. log2 unit migration in `compute_summaries_from_gibbs`

**Status.** Proposed, not started. Deferred deliberately — inert for testing,
wrong units for anything reported.

The project convention, set 2026-09-15, is that expression, allele-specific
expression ratios, allelic fold change and all their uncertainties are in
**log2**, so that `beta = 1` means a twofold effect. The runtime has never been
converted. `tensorqtl/hapmixqtl.py:406` and `:408` still build the draw
summaries with natural logarithms:

    a_draws = np.log(yL + kappa) - np.log(yR + kappa)
    t_draws = np.log(tot / 2 + kappa)

so every quantity downstream of them is in natural-log units.

**What the conversion involves.** Effects are first-order in the log, variances
second-order, so the two scale by different powers and cannot be converted with
one constant:

- Effects (`beta`, `slope`, allelic log ratio, log aFC) divide by `ln 2`,
  i.e. multiply by about 1.4427.
- Variances and covariances in squared-log units — `tau_A`, `tau_T`, `Va`,
  `Vt`, the Gibbs covariance `Cat`, and the fitted `tau_g` of the
  `two_component`/`library_scaled` variance models — divide by `(ln 2)^2`,
  i.e. multiply by about 2.0814.
- The coefficient `c_g` of the two-component model is **dimensionless** (a
  regression slope of squared residuals on `v`, both of which rescale
  identically) and must NOT be converted.
- `kappa` is a count pseudocount, not a log-scale quantity, and is untouched.

**Why it is inert for the tests.** Association statistics are ratios of an
effect to its standard error, and a global change of logarithm base cancels
from them exactly. The permutation p-values, the type-I rates and the called
genes are all unaffected. What changes is the interpretability of every
reported effect size and every reported variance component.

**The hazard that makes this worth doing carefully rather than quickly.** All
historical numbers — the fitted `tau_a` of 0.0030 to 0.0049, the implied
RASQUAL `rho`, every `tau` in `CLAUDE.md` and in the reports — are recorded in
squared natural-log units. A partial migration that converts the runtime but
not the recorded comparisons would silently put a factor of 2.0814 between the
code and its own validation record. The migration therefore needs a single
commit that converts the code, restates the affected recorded constants with
their units named, and pins the conversion in a test.

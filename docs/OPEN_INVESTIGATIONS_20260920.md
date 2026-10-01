# Historical investigations: proposed work as of 2026-09-20

> **Status, 2026-10-01: dated list of proposals; no item here is current
> work.** This file held, on 2026-09-20, work that was proposed and not yet run,
> kept apart from the observed facts in CLAUDE.md and the routing in
> `docs/CURRENT_SCIENTIFIC_STATE.md`. Its first entry, the per-channel residual
> sigma (mechanism 4 of the six proposed for the disagreement between hapmixQTL
> and mixQTL), was closed on 2026-09-20 and refuted as a cause of that
> disagreement; the `se_mode='fitted'` standard error added to test it is now
> part of the shipped default mode (`tau_mode='zero'` with `se_mode='fitted'`).
> Its second entry, a log2 migration of `compute_summaries_from_gibbs`, was
> never started and does not apply to the shipped default, which builds log2
> inputs from Salmon point estimates through `prepare_default_inputs`
> (`docs/hapmixqtl_methods.md`, `docs/pipeline_rules.md`);
> `compute_summaries_from_gibbs` and the dated scripts that import it remain in
> natural log as a historical path. Wrong to act on today: the sentence "the
> runtime has never been converted", which holds only for that historical
> helper, and the conversion list's `tau_A`, `tau_T`, `Cat`, `c_g` and
> `two_component`/`library_scaled` quantities, which belong to configurations
> deprecated on 2026-09-23 (`brainvar_hapmix_deploy/deprecated_models/README.md`).

---

## Closed 2026-09-20: per-channel residual sigma (mechanism 4) — REFUTED

Was the last untested of the six disagreement mechanisms. Tested by adding
`se_mode='fitted'` to hapmixQTL and holding the weights fixed while changing
only the standard-error form, which is exactly the replacement its written
refutation criterion named. The combined beta correlates with the shipped
arm at 0.9978 with a median |beta| ratio of 1.011 — essentially unchanged,
so the mechanism is refuted. Conclusion in `CLAUDE.md`; numbers in
`/mnt/ssd/lalli/brainvar_hapmix_deploy/deprecated_models/from_mixqtl_replication_20260919/HAPMIXQTL_FITTED_SE.md`.
The six-mechanism catalogue is closed; this file keeps the entry only as a
pointer and no longer lists it as work to do.

## Historical proposal: log2 unit migration in `compute_summaries_from_gibbs`

**Historical status.** Proposed, not started. This does not apply to the
current default, which constructs log2 point-estimate inputs through
`prepare_default_inputs`; the helper discussed below is not that route.

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

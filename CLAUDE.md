# tensorQTL, hapmixQTL branch

A fork of tensorQTL adding **hapmixQTL**: cis-eQTL mapping from haplotype-resolved
expression posteriors (Salmon Gibbs draws against a personalized diploid
transcriptome), with the quantifier's inferential uncertainty carried into the
standard error. Work lives on `claude/hapmixqtl-gibbs-uncertainty-IQ6Za`.

## Which document answers which question

The operating knowledge is large and split by purpose. Start in the right place:

| Question | Document |
|---|---|
| What is the statistic, exactly, and how do I reproduce it? | `docs/hapmixqtl_methods.md` — the specification: numbered equations, the algorithm, defaults, assumptions, and a symbol-to-code map |
| How do I run the BrainVar deployment end to end? | `docs/brainvar_deploy_runbook.md` — inputs, sample pairing, RASQUAL's invocation, gene selection, the comparison |
| What was measured, and how do I know the method is calibrated? | `docs/ase_validation.md` — the validation record, including withdrawn claims |
| What do the output columns mean? | `docs/outputs.md` |
| What does a new session need to pick this up? | `docs/LOCAL_HANDOFF.md` |

## Facts that are easy to get wrong

- **`tau` is not something the Gibbs draws measure.** The per-sample weight is
  `1/(v_inf + tau)`. `v_inf` is read-assignment uncertainty plus a Poisson counting
  term; `tau` is the between-sample variance estimated across samples. On BrainVar's
  well-expressed genes `v_inf` is about a third of the allelic channel's error and
  under 1% of the total channel's.
- **Two scales are reported.** `pval_perm` and `pval_beta` are gene-level and come
  from the scan; `slope`, `slope_se` and `pval_nominal` come from the lead refit when
  `tau_refit=True`. A lead's nominal p is never a gene-level p.
- **The detection call is the empirical p**, not the statistic. Statistic magnitude
  has been corrected three times; the called genes did not move.
- **Sample order is positional.** Phase frames are indexed by the genotype frame's
  column order. `_assert_phase_columns` now guards it; before that, reordered columns
  corrupted the allelic channel while the total channel stayed correct.
- **Library and driver defaults agree except on one flag.** `count_noise` is on
  and `ase_covariates_df` is intercept-only everywhere since 2026-09-13; pass
  `count_noise=False` or `ase_covariates_df=SAME_COVARIATES` to reproduce
  earlier results. `tau_refit` is the exception: on in both drivers, off in the
  library and CLI, because it carries a known unfixed selection bias (below) and
  a default should fail conservative. It should flip once the refit is charged
  the selection it performs.

## Claims withdrawn on 2026-09-13 — do not re-assert

An eight-angle review retired these. They may survive in older text.

- The additive, multiplicative and nested variance forms were **never** compared:
  no multiplicative arm exists in `tests/`.
- `docs/ase_validation.md` §7b's weight-cap and nested conclusions are void; both
  arms were arithmetically incapable of differing from their comparators.
- The additive `tau` is **not** "the one place hapmixQTL departs from mixQTL".
  mixQTL's total channel already carries a flat additive variance, so `v_t + tau_t`
  decomposes the parent rather than departing from it; the multiplicative scale is
  the allele-specific channel only.
- mixQTL's scan refits its dispersion at **every** variant, so hapmixQTL's per-gene
  null-model `tau` is the departure and `tau_refit` moves back toward the parent.

## Known and unfixed

- The lead refit is biased by selection: appending the window maximum to the `tau`
  design removes far more residual sum of squares than the one degree of freedom it
  is charged, so `tau` comes back 11-15% low under the null and the reported
  statistic is inflated. Gene-level p-values are unaffected.
- The Beta approximation is conservative in the tail, which costs power at
  transcriptome-scale thresholds.
- No per-sample allele-specific read floor, where mixQTL used 15 reads.

## Self-tests

Every script self-tests on fabricated data and prints `SELF-TEST OK`:

```bash
pytest tests/test_hapmixqtl.py tests/test_hapmixqtl_calibration.py tests/test_cli.py -q
RASQUAL_BIN=rasqual_src/src/rasqual python3 scripts/compare_pipelines.py --selftest
python3 scripts/run_hapmixqtl_from_salmon.py --selftest
```

The comparison driver's self-test covers the RASQUAL arm, the matched-variant effect
comparison, and that a rerun reuses its null-round checkpoints byte-identically.

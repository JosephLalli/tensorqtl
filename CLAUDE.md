# tensorQTL, hapmixQTL branch

A fork of tensorQTL adding **hapmixQTL**: cis-eQTL mapping from haplotype-resolved
expression posteriors (Salmon Gibbs draws against a personalized diploid
transcriptome), with the quantifier's inferential uncertainty carried into the
standard error. Work lives on the worktree branch `hapmix-runbook-local`,
pushed as `origin/claude/hapmixqtl-gibbs-uncertainty-IQ6Za`.

## Which document answers which question

The operating knowledge is large and split by purpose. Start in the right place:

| Question | Document |
|---|---|
| What is the statistic, exactly, and how do I reproduce it? | `docs/hapmixqtl_methods.md` — the specification: numbered equations, the algorithm, defaults, assumptions, and a symbol-to-code map |
| How do I run the BrainVar deployment end to end? | `docs/brainvar_deploy_runbook.md` — inputs, sample pairing, RASQUAL's invocation, gene selection, the comparison |
| What was measured, and how do I know the method is calibrated? | `docs/ase_validation.md` — the validation record, including withdrawn claims |
| What do the output columns mean? | `docs/outputs.md` |
| What does a new session need to pick this up? | `docs/LOCAL_HANDOFF.md` |
| What is implemented, proposed, validated, or currently running? | `docs/CURRENT_SCIENTIFIC_STATE.md` — concise current-state router and phase gate |

## Scientific phase transitions

The user's phase-transition rule supersedes the earlier policy to offer
`/reload`. Before a new substantive scientific phase, run a documentation agent
to validate and reconcile the actual local state into concise authoritative
pointers to the intellectual state, generated files, performed experiments and
results, decisions, existing code, and how to find them. After that agent
completes, resume already-authorized work.

State boundaries explicitly: distinguish implemented behavior, proposed work,
validated results, and current run state. This documentation pass records the
existing state only; it does not imply or start a new experiment.

## Writing conventions (the user's)

- Define every named statistical method on first use, in terms of what it
  computes (DerSimonian-Laird, Paule-Mandel, Freedman-Lane, Kish, and so on).
- Never write "cell" for a table entry or a (donor, gene) datapoint; in this
  work "cell" means a biological cell. Say datapoint, donor-gene pair, or
  zero-read sample.
- Reports for the user are HTML pages with figures, not long markdown.

## Facts that are easy to get wrong

- **Use log2 for expression, ASE ratios, aFC, and their uncertainty.** The user
  established this project convention on 2026-09-15. beta=1 means a twofold
  effect; tau_A/tau_T and Gibbs covariance use squared log2 units. Runtime
  conversion is pending; current outputs still use natural logs. See the
  [unit convention and conversion record](/mnt/ssd/lalli/brainvar_hapmix_deploy/mixqtl_algorithm_review_20260914/salmon_variance_theory_20260915/LOG2_CONVENTION.md).

- **ASE has no automatic intercept as of 2026-09-15.** This applies to its
  regression, null tau, and lead-refit tau designs. `ase_covariates_df=None`
  means through-origin with no nuisance columns; the total intercept remains.
  Results recorded before this change used the old design and are historical.
- **Default Salmon Gibbs includes counting noise, and `count_noise` is inert on
  well-expressed genes.** The controlled experiment at
  `/mnt/ssd/lalli/brainvar_hapmix_deploy/salmon_gibbs_counting_sim_20260915/REPORT.md`
  (Gibbs-only variance / MSE 0.944, with `q` 1.906) shows `q` double-counts
  the sampling noise of donors that have reads, and a census of the production
  draws shows the docstring's other case, reads with unanimous draws, never
  occurs (0 of 76,286 datapoints). On the 29 calibration genes removing `q` moves
  the lead statistic by a median 0.19 chi2 and changes no call. The flag is
  nevertheless load-bearing, for a reason neither side stated: the total
  channel has no degenerate-sample guard (`_zero_degenerate_ase_weights` exists
  only for ASE), so a zero-count total sample keeps weight `1/(1e-8 + tau_t)`,
  and in low-expressed genes where `tau_t` clamps to zero that is the
  2026-09-13 "52% type-I" weight-domination failure. `q` prevents it only
  because `1/(0 + 2*kappa) = 1` acts as a floor. 39.95% of donor-gene datapoints in 2,000
  random genes have no reads (none in the 29 calibration genes); a verifier
  measured that without `q` such samples hold a median 89.5% of the total
  channel's weight in the 281 of 1,079 sampled genes that have them, against
  17.4% with `q`. The fix is a coverage-based floor for zero-count total
  samples that is independent of the flag, then dropping `q` for samples with
  reads; excluding zero-count total samples is wrong (t = log kappa is real
  low-expression information, and exclusion conditions on the outcome). The
  default stays True until that is done; see
  `/mnt/ssd/lalli/brainvar_hapmix_deploy/estimator_ablation_20260916/REPORT.md`.
  Quantified 2026-09-18 for the allelic channel's `q_a = 1/(mL+kappa) +
  1/(mR+kappa)` specifically (recomputed from the cached per-draw arrays,
  34,457 genes x 92 donors x 200 draws, by
  `/mnt/ssd/lalli/brainvar_hapmix_deploy/variance_layer_mapping_20260918/variance_layer_measurements.py`,
  not under git, cite by path): median `q_a`/`v` (the double-counting
  relative to the Gibbs variance it sits beside) is 0.90 at 1-9
  haplotype-informative reads, 0.30 at 10-99, 0.12 at 100-999, 0.07 at
  1,000+, i.e. `q_a` matters least exactly where reads are plentiful and `v`
  is trustworthy on its own. Separately, 57.8% of donor-gene datapoints
  across the full 34,457-gene set have zero haplotype-informative reads (a
  different base than the 39.95%-with-no-reads figure above, which is 2,000
  random genes counting total-channel reads); for those, `v`=0 exactly. The
  pseudocount formula would give `q_a`=4.0 (`kappa`=0.5 default) if it were
  applied here, but it is not: `compute_summaries_from_gibbs`'s `no_cov`
  guard (`tensorqtl/hapmixqtl.py:417-420`, `no_cov = (mL+mR)<=0`) forces `Va`
  to exactly `0.0` instead of `Va + q_a` on this branch, so
  `_zero_degenerate_ase_weights` (`hapmixqtl.py:181-200`, `keep = va_t >
  eps`) zeroes the allelic weight entirely rather than handing the pair a
  finite, ordinary weight of `1/(4.0+tau)` built on a fabricated `a=0`
  (perfect allelic balance the degenerate draws did not actually measure).
  TWO CORRECTIONS, both 2026-09-18: this passage previously read "so `q_a`
  is the only thing keeping the allelic weight finite" and "`q_a`'s floor
  role only ever applies where there is almost nothing to weigh" —
  backwards; the guard exists BECAUSE `q_a` would otherwise apply here, and
  it keeps these pairs OUT of the fit, not in it at a floored weight.
  Separately, an earlier "87.6% of zero-informative-read datapoints are
  under 10 total reads," later "corrected" to 100%, is withdrawn outright,
  not merely re-corrected: both figures were tautological (the analysis
  script's `R` is `mL+mR` itself — same quantity that defines
  "zero-informative" — so conditioning one on the other was conditioning a
  quantity on itself; verified directly against the cache this pass,
  max|`R`-(`mL`+`mR`)|=2.3e-10 over all 3,170,044 datapoints). Why 57.8% of
  pairs land here at all is a Salmon-indexing and pipeline-ingest fact, not
  a low-expression one — see the next bullet. `q_a` is otherwise doing one
  job (double-counting shot noise) where there are reads with any coverage.
  Its
  `+kappa` pseudocount is the Haldane-Anscombe correction to the empirical
  log odds (the standard small-sample fix for a zero-count donor-gene
  datapoint in a two-way allele split), whose variance under the delta
  method (the first-order Taylor approximation for the variance of a
  function of a random variable, given the variable's own variance) on the
  natural-log scale is exactly `1/(mL+1/2)+1/(mR+1/2)`; edgeR's own binomial
  pipeline adds the identical half (`y <- y + ceiling(prior.count)/2`,
  `prior.count=1` default, `R/binQLFtest.R` in edgeR's source) for the same
  reason. The total channel's counting term (the `1/(0 + 2*kappa) = 1` floor
  discussed above) is a different, single-count Poisson delta-method
  correction, not a log-odds correction, and is not this same device.

- **Why 57.8% of donor-gene pairs have no allele-specific information: a
  Salmon-indexing and pipeline-ingest fact, not a low-expression one.**
  Established 2026-09-18 from pipeline code, run metadata, and the cached
  summaries; re-verified in this documentation pass, not merely transcribed.
  The personalized diploid transcriptome is built by g2gtools, which emits
  TWO haplotype copies of every transcript, suffixed `_L`/`_R`; on sex
  chromosomes only the PAR regions get these suffixes
  (`bin/append_suffixes_to_sex_chroms.py` in
  `/mnt/ssd/lalli/nf_stage/rnaseq_JLL`, keyed to the X/Y `seqid` — read this
  pass — because PAR alone is genuinely diploid; consistent, not an anomaly).
  `salmon index` is run WITHOUT `--keepDuplicates` against that
  transcriptome: where a donor is homozygous across a transcript its `_L`
  and `_R` sequences are byte-identical, so Salmon's indexer keeps one copy
  and the `_R` row never appears in `quant.sf` at all — not a zero row, no
  row. Confirmed directly this pass: every `aux_info/meta_info.json` checked
  (3 of 229 quant directories under
  `/mnt/ssd/lalli/nf_stage/RNA_reference_comparison_results/reference_comparison_results/bv2/personalized_T2T_NCBI110_pseudoalignment/expression_results/salmon_pseudocounts/`,
  samples 546_R1/482_R1/914_R2) records `"keep_duplicates": false`; `_L`
  transcript counts there are 176,395/176,222/176,226 (near-constant, within
  the cited 176,209-176,409 cohort-wide range) against `_R` counts
  111,472/92,348/107,377 (within the cited 88,239-116,913 range, spread
  28,674 across the 92 donors — the signature of collapsed duplicates, not
  lost reads), and TPM sums to 1,000,000 within floating-point rounding
  (1,000,000.00006-1,000,000.00008 at full `awk` precision, not the exactly
  1e6 a lower-precision print would suggest) in all three. CAVEAT found
  this pass: `conf/modules.config` in that pipeline conditionally sets
  `ext.args = params.use_personalized_references ? "--keepDuplicates" :
  ""` for the process named `SALMON_INDEX`, and this run's own captured
  params (`pipeline_info/params_2025-05-08_17-22-59.json`) record
  `use_personalized_references: true` — so the config text alone suggests
  dedup should have been OFF for this run. It was not; `meta_info.json`
  above is the runtime ground truth and is what this bullet relies on. Why
  the conditional did not fire is unresolved (traced one level: a
  differently-scoped `SALMON_INDEX` selector in `conf/deepthought.config`
  targets a `PREPARE_PERSONALIZED_TRANSCRIPTOME` subworkflow name that does
  not exist anywhere in the current `rnaseq_JLL` checkout — the real
  subworkflow is `subworkflows/local/personalize_reference_genome.nf` — but
  that only explains why THAT selector's resource settings are dead, not why
  the separate, later-loaded `modules.config` conditional didn't apply). A
  `rnaseq_JLL`-repo question, not a hapmixQTL one; trust the runtime record
  over the config file here.

  Losing the `_R` row does not by itself zero a pair's allelic information —
  reads on a surviving unpaired `_L` row would still be real counts. What
  makes `mL` ALSO exactly 0 is the ingest step, not the index:
  `scripts/run_hapmixqtl_from_salmon.py`'s `pair_haplotypes` (line 217-228)
  pairs a base transcript id only when BOTH suffix rows exist, and
  `load_counts` (line 943-948) accumulates `YL`/`YR` ONLY over paired
  transcripts — an unpaired (homozygous, deduplicated) transcript
  contributes to NEITHER, no matter how many reads its surviving row
  carries. `YT`, by contrast, sums EVERY transcript unconditionally (line
  958-969), paired or not. This is exactly why `compute_summaries_from_gibbs`
  already insists on a `yT` summed over ALL transcripts rather than `yL+yR`
  (`tensorqtl/hapmixqtl.py:357-366`, pre-existing text, still correct):
  `yL+yR` IS that same paired-only subtotal, so using it as the total would
  silently zero these same pairs there too. A gene-donor pair has
  `mL=mR=0` (not merely `mR=0`) exactly when NONE of that gene's transcripts
  have a surviving heterozygous pair in that donor.

  Measured consequence, verified this pass against the cache
  (`/mnt/ssd/lalli/brainvar_hapmix_deploy/estimator_ablation_20260916/verifiers/per_donor_dispersion/allgene_summaries.npz`
  + `allgene_ambiguity.npz`, not under git): `R` equals `mL+mR` to
  max|diff|=2.3e-10 over all 3,170,044 datapoints (34,457 genes x 92
  donors); `R<=0` (equivalently `R==0`, no partial cases) for
  1,831,718/3,170,044 = 57.8%; `Va==0.0` exactly in literally 100% of those
  1,831,718 and in 0 of the remaining 1,338,326 — an exact match to the
  `no_cov` guard below. Of the 1,831,718 zero-informative pairs, 68.0% have
  zero total expression too (genuinely unexpressed) but 22.7% — 416,207
  pairs, 13.1% of ALL 3,170,044 pairs — are expressed with no allelic
  information whatsoever, including 89,639 pairs at 1,000+ total reads. Both
  figures independently reproduced this pass against the separate `YT`
  per-draw array (`/mnt/ssd/lalli/brainvar_hapmix_deploy/cache/gibbs_56b63c3b37ed5df8/YT.npy`,
  mean total expression per draw for the zero-`R` subset): 67.9% exactly
  zero, 22.7% at or above a 10-total-read threshold — matching to two
  decimal places and pinning "expressed" here to mean >=10 total reads, not
  >0. That threshold leaves a third, unlabeled band the two headline figures
  don't cover: ~9.3% of the 1,831,718 pairs have TRACE total expression (0
  to 10 reads) — neither "genuinely unexpressed" nor "expressed" by this
  cut, and not a gap in the accounting. The 68.0/22.7% figures are the
  brief's; this pass's own numbers (67.9%/22.7%) agree to within the
  threshold choice. These are homozygous-but-expressed pairs (the 22.7%
  tier): real expression, zero allelic information, by construction, not a
  depth artifact.

  DESIGN NOTE worth recording: the intended behavior of Salmon's Gibbs
  resampling was self-downweighting — if two haplotypes were
  indistinguishable, the Gibbs draws should disagree about the split, `v`
  should blow up, and `1/(c*v+tau)` should collapse on its own. That
  mechanism cannot fire here, because deduplication and the pairing rule
  above mean there is no second copy to be uncertain between; every draw has
  `yL=yR=0` by construction, not by chance.

  THE CODE ALREADY HANDLES THIS CORRECTLY; the guard's real purpose was
  previously undocumented. `compute_summaries_from_gibbs`
  (`tensorqtl/hapmixqtl.py:417-420`, inside `if count_noise:`): `no_cov =
  (mL+mR)<=0` forces `Va = np.where(no_cov, 0.0, Va + 1/(mL+kappa) +
  1/(mR+kappa))` — the counting term `q_a` is added ONLY where there is
  coverage; where both haplotypes are empty, `Va` is forced to exactly
  `0.0`. `_zero_degenerate_ase_weights` (`hapmixqtl.py:181-200`, `keep =
  va_t > eps`) then zeroes the allelic weight for exactly these pairs. No
  fabricated observation of perfect allelic balance enters the allelic
  channel. This guard reads, and is commented, as a generic
  degenerate-coverage guard; nowhere is it documented as excluding unphased
  (homozygous) donor-gene pairs specifically, which — per the mechanism
  above — is what it actually spends most of its time doing.

  One item closed, one open, both recorded precisely rather than as
  findings only: (a) `no_cov` sits inside `if count_noise:`; with
  `count_noise=False` the same protection would have to come from `Va`
  already being exactly `0.0` from constant (`yL=yR=0`) draws rather than
  from the explicit guard. VERIFIED this pass, incidentally: the cache above
  IS a `count_noise=False` computation (`summarize_all_genes.py`'s own
  docstring: "Same formula as `compute_summaries_from_gibbs(count_noise=
  False)`"; it computes `Va` as `lr.var(2, ddof=0)` directly, the identical
  formula and `kappa` to `hapmixqtl.py`'s own pre-`count_noise` `Va =
  a_draws.var(axis=2, ddof=0)`), and its `Va==0.0` in literally 100% of the
  1,831,718 zero-coverage pairs (above) confirms the protection holds
  without the explicit guard too, from the mathematical fact that constant
  draws have exactly zero variance under `np.var`. (b) The total channel correctly retains these
  pairs (`YT` sums unconditionally, above), so the 13.1%-of-all-pairs figure
  is total-channel-only BY CONSTRUCTION, not a data-quality problem — easy
  to misread as loss. Relatedly, an expression-based gene filter and the
  allelic channel's informativeness filter are NOT nested — see the
  `squeezeVar` robustness check below.

- **The DerSimonian-Laird `tau` understates the allelic residual scale.** With
  the final weights `1/(v + tau_DL)` the whitened residual mean square at the
  null design is 1.22 on the allelic channel (0.93-2.58 over the 29 genes) and
  1.04 on the total channel, so the known-variance SE is too small on the
  allelic channel by a gene-specific factor. A Paule-Mandel `tau` (iterate the
  moment update with the final weights until RSS_w = df) restores 1.00 by
  construction: reported statistics fall by a median 11% (CCNI 38 -> 10, APC
  25 -> 15, RPL15 22 -> 15), the null 95th percentile from 21.6 to 18.9, and
  the between-gene spread of null medians halves, after which a pooled
  threshold and the per-gene null call the same genes. This is measured, not
  adopted: `_estimate_tau` is still DL. The residual shape is also wrong: after
  PM whitening the standardized squared residual still rises with `log v`
  (pooled slope +0.21, s.e. 0.03). Measured 2026-09-17 (report addendum): the
  two-parameter `c*v + tau` model (allelic c median 2.6, tau 0.006) fits the
  shape (slope +0.02), is calibrated (3.9% of true-null gene-draws below 0.05),
  puts the null on RASQUAL's scale (95th percentile 15.7 vs 14.6), and calls 9
  genes vs 6; the one gene among the 29 that is an eGene in the independent
  225-sample total-expression run (CCNI) is caught by it and by RASQUAL and
  missed by both additive configurations. The pure multiplicative model
  (weights 1/v with a fitted scale; mixQTL's form, but mixQTL caps the weight
  dynamic range at 10x and filters counts, which is a floor on v by another
  name) is anticonservative here (15.1% at nominal 5%) because its whitened
  residuals are not exchangeable under a floor and the allelic effective
  sample size collapses (Kish 70.6 -> 43.6); a fitted-scale permutation must
  also re-project permuted residuals onto the design complement, which the
  known-variance construction does not need. Verified independently: the SE
  rule is nearly inert for the gene-level permutation p, the weight rule is
  everything. Same report as above. PM is not the 2026-09-16 "fitted
  residual scale" rule: away from the `tau = 0` boundary the two coincide
  (SE_codex = s_v * SE_lib and s_v = 1 at the PM fixed point), but at the
  boundary the fitted-scale rule discards the Gibbs variance floor and halves
  the SE, and a verifier measured that boundary at 64.5% of allelic and 47.2%
  of total channels in 600 unfiltered cohort genes (41.5% / 13.5% without
  `q`; 0% for total at depth >= 50). Every experiment so far (29-gene null,
  30-gene pilot, shape pilot, influence audit) lives in the high-depth regime
  where the rules agree; the disagreement lives in the low-expression genes
  none of them included. Keep the absolute floor (known-variance SE, PM
  `tau`), and run a depth-stratified arm before any genome-wide scan.

- **Three variance models and an empirical-Bayes prior (2026-09-17).**
  `variance_model` on every mapping function and both CLIs: `additive`
  (default, `v + tau`; its nominal statistics and leads are byte-identical
  to before, max |d stat| 2.3e-6 on the 29-gene outputs; the permutation p
  changed by design when the record permutation became the default, below), `two_component` (`c_g v + tau_g`, fitted per
  gene by damped iterated weighted least squares of leverage-corrected squared
  residuals on [v, 1]) and `library_scaled` (`d_i (c_g v + tau_g)`, `d_i` from
  `estimate_library_factors`, required by and only accepted by that model).
  `variance_prior` from `estimate_variance_priors` replaces the zero clamp on
  (c_g, tau_g) with a log-normal prior per expression decile (means from the
  raw unclamped estimates, variances less the median sampling variance, kappa
  = Var(z^2) = 2.38) and fits the posterior mode in (log c, log tau) by
  damped Fisher scoring; rejected under `additive`. Measured on the 300-gene
  tiered calibration (`estimator_ablation_tiers_20260917/summary_prior.tsv`):
  type-I at 5% 0.040/0.036/0.030 with the prior against
  0.040/0.032/0.030 clamped (s.e. 0.003), no gene at either bound, the clamped fit's 43 plus EXOC6B, LOC124903631
  called, 91% of leads identical. The prior removes the clamp but NOT
  the low-expression non-identifiability: below ~70 allele-resolved reads the
  prior on c is 0.035 with log-sd 2.4 and the weights stay near-equal. A
  normal-scale prior floored 41-54% of genes again and was rejected. The
  `c_a_floored` output column reports a clamp hit on the clamped
  two-component path and is always False under the log-scale prior.
  The posterior-mode fit needs its backtracking line search with the merit
  consistent with the score (no 0.5), a direction-keeping step cap and the
  objective-gain stop rule, and TWO STARTS (prior mean and the clamped fit,
  higher objective kept): without them 12.7% of fits two-cycle, an
  inconsistent merit converges to a point that is not the mode, and under
  the two lowest bins' prior (median c 0.0018) a single start from the prior
  mean returns the spurious low-c mode for most identifiable genes
  (tests/test_hapmixqtl_variance_prior.py pins all of it; a golden test pins
  the additive path at 99921fd). The runner's library-scaled mode crashed on
  any cohort with a quantified gene lacking a position row (fixed; self-test
  covers it). Open: the bin prior estimator floors a negative raw mean in the
  two lowest bins and its moment match manufactures the second mode (the
  continuous prior below removes both, but is not the default); kappa
  is global; run_second_pass ignores the variance model; outputs.md/README
  lack the new columns and flags.
  Production form for BrainVar: `library_scaled` with the prior. Report:
  `estimator_ablation_20260916/REPORT.md`, the shrinkage and two-start
  sections.
  A continuous prior (`prior_method='trend'`, 2026-09-18: curves of log c and
  log tau on log10 reads fitted by local marginal likelihood over all genes,
  two passes with curve-based weights, two starts per window) removes the
  decile edges and the moment-match collapse: c agrees with the deciles
  within 22% from 68 reads up; tau in the top two deciles is
  0.018-0.019 where the deciles gave 0.0038-0.0067, and the predictive
  check settles it against the trend: 40-47% of raw tau estimates there are
  non-positive, the trend's log-normal predicts 7-13%, the deciles' 31%; the
  optimizer is at its optimum and a direct ML of the same log-normal on the
  same genes gives 0.006-0.007 (predicting 19-28%), so the lift to 0.018 is
  the 0.5/99.5% winsorization in _trend_prior choosing the narrow of two
  solutions, and no log-normal holds the near-zero bulk (a spike-and-slab on
  tau is the fix), while for c the trend is the better-behaved prior; mode switching 0.1% vs 1.9% (deciles); record-scheme
  type-I 0.057/0.052/0.054, calls 17/16/17 = 50 vs 16/16/18 = 50 (deciles), paired
  discordance 3+3 of 300. Default stays 'deciles'; per-gene
  c_raw_var/tau_raw_var and the pass-2 columns are in the prior frame for
  such checks. Report: the same file, last section. Structural caveat on
  this whole model family (2026-09-18, "Relationship to limma..." below):
  fitting `(c_g, tau_g)` from a gene's own residuals and then weighting
  those residuals by the fit is categorically different from every
  comparator method, which fixes its per-observation variance in advance,
  and the clamped fit's weights are provably invariant to the absolute
  scale of the Gibbs draws — only the within-gene shape of `v` survives.

- **The permutation null permutes donor records, not residuals (2026-09-17).**
  `perm_scheme='records'` (default): each donor's whitened phenotype value,
  weight and covariate row move together, genotypes stay, the denominator
  is recomputed per permutation (`_record_permutation_channel`); by
  relabeling this equals the genotype-permutation null of FastQTL and
  tensorQTL with per-donor weights, pinned to 1e-9 by
  tests/test_hapmixqtl_perm_scheme.py. `perm_scheme='residuals'` is the
  earlier Freedman-Lane scheme (leverage-standardized whitened residuals
  permuted at fixed weights). Why it changed: on BrainVar the residual
  scheme was conservative and most so at high expression (two-component
  type-I 0.040/0.036/0.030 by tier; shipped 0.068/0.045/0.022). Allelic
  channel: its null has the scale of the UNWEIGHTED mean of z^2, a real null
  has the w-weighted mean (pinned at 1 by the fit); R = weighted/unweighted
  predicts per-gene type-I (bins 0.006 -> 0.072), R = 1.00 where weights are
  equal (low tier) and 0.97 where they span two decades (high tier);
  allelic-only 0.032 -> 0.048 with records. Total channel: short in every
  tier (0.029-0.035) from the covariate handling (18 columns / 92 donors;
  no-leverage 0.23, phenotype-permute 0.00, records ~0.05-0.07). Cost 0.34 s
  vs 0.10 s per 1,000-permutation scan. Report:
  `estimator_ablation_20260916/REPORT.md`, two last sections.

- **Gene-level Gibbs shape has bounded real-data evidence.** The completed
  three-library pilot found Gaussian competitive within 0.05 bits/draw for
  98.51% of 4,500 ASE and 99.93% of total summaries; it retained two
  reproducible ASE candidates and no total candidates. ZNF529 and RNF175 are
  exceptions to interpret, while RNF175 restart-block occupancy is unreliable.
  This is not association or tau calibration. See
  `/mnt/ssd/lalli/brainvar_hapmix_deploy/gibbs_shape_pilot_20260915/REPORT.md`.

- **The two-gene influence audit retained the moment/GPU baseline.** Across
  three genotype-selected common SNPs per gene and 92 donors (87 ASE), maximum
  one-block mean-plus-covariance shifts were 0.0736 working SE (ZNF529) and
  0.1396 (RNF175); leave-one-block maxima were 0.00901 and 0.01028. These are
  bounded sensitivity results, not association calibration. See
  `/mnt/ssd/lalli/brainvar_hapmix_deploy/gibbs_influence_audit_20260915/REPORT.md`.

- **Parents already estimate aggregate residual variance.** TensorQTL and
  mixQTL estimate a per-variant residual scale after fitting covariates and
  genotype; it combines measurement, biology, and other residual sources.
  They do not identify biological tau separately, and that identification is
  not required for association. Do not add M unchanged to that residual scale.
  The 2026-09-16 "fitted residual scale" recommendation in
  `docs/IMPLEMENTATION_STATUS_20260916.md` is, within the existing framework,
  a self-consistent `tau` (Paule-Mandel) applied per variant; the measured
  consequence of the level part is in the bullet above. Do not reopen this
  from theory: run `estimator_ablation_20260916/ablation29.py` and measure.

- **`tau` is not something the Gibbs draws measure.** The per-sample weight is
  `1/(v_inf + tau)`. The current code uses Gibbs measurement variance plus the
  Poisson counting term `q` (see above); `tau` is the residual variance
  estimated across samples. On BrainVar's well-expressed genes `v_inf` is
  about a third of the allelic channel's error and under 1% of the total
  channel's.
- **Two scales are reported.** `pval_perm` and `pval_beta` are gene-level and come
  from the scan; `slope`, `slope_se` and `pval_nominal` come from the lead refit when
  `tau_refit=True`. A lead's nominal p is never a gene-level p.
- **The detection call is the empirical p**, not the statistic. Statistic magnitude
  has been corrected three times; the called genes did not move.
- **Sample order is positional.** Phase frames are indexed by the genotype frame's
  column order. `_assert_phase_columns` now guards it; before that, reordered columns
  corrupted the allelic channel while the total channel stayed correct.
- **Library and driver defaults agree except on one flag.** `count_noise` is on
  and `ase_covariates_df=None` is through-origin everywhere since 2026-09-15; pass
  `count_noise=False` or `ase_covariates_df=SAME_COVARIATES` to reproduce
  earlier counting/covariate settings (this does not restore the former ASE
  intercept). `tau_refit` is the exception: on in both drivers, off in the
  library and CLI, because it carries a known unfixed selection bias (below) and
  a default should fail conservative. It should flip once the refit is charged
  the selection it performs.

## Relationship to limma, edgeR, sleuth, swish (2026-09-18)

Read fresh upstream source (git clones made 2026-09-18, not the older
installed R packages below) to place hapmixQTL's error model relative to
differential-expression and quantification-uncertainty tooling. Full framing
(the three-layer structure these methods share, and the two cases for where
a per-observation variance can live) is in `docs/CURRENT_SCIENTIFIC_STATE.md`
under "Structural relationship to limma, edgeR, sleuth, swish" — read that first
for the why. This section is the checkable reference: which of our
components already has a published name, which upstream function to reach
for, and whether it is available in the installed version or is devel-only.

| Our component | Upstream equivalent (what it computes) | Ours | Upstream | Installed here? |
|---|---|---|---|---|
| Weight `1/(c_g v_ig + tau_g)` as a gene-by-sample matrix in a linear model | limma `lmFit` + a supplied weights matrix, `Var = sigma_g^2 / w_ig` | `_wls_regression`, `tensorqtl/hapmixqtl.py:483` | limma `lmFit`, R/lmFit.R | Installed (limma 3.64.3), but CRASHES on this machine — `lmFit` segfaults even unweighted, not only with a weight matrix (mixed-BLAS environment note below) |
| Building a gene-by-sample weight matrix from a continuous per-observation covariate | limma `vooma`/`voomaLmFit`: `predictor` argument takes a gene-by-sample matrix; trend `lm.fit(cbind(1,sx,sxc), sy)` with `sy` = per-gene residual variance to the fourth root; weight `1/f(mu)^4` | `v_ig` plays the role of `predictor`; no direct port | limma `vooma`/`voomaLmFit`, R/vooma.R | Installed (limma 3.64.3 also has `voomWithQualityWeights`, `arrayWeights`), but CRASHES here — all fit a linear model internally, same mixed-BLAS crash as `lmFit` |
| Cross-gene moderation of one fitted per-gene scale | limma `squeezeVar`: posterior `(df*var+df.prior*var.prior)/(df+df.prior)` | `estimate_variance_priors`/`_trend_prior` moderate `(c_g, tau_g)` directly, not through `squeezeVar` (see below for why) | limma `squeezeVar`, R/squeezeVar.R; called directly by edgeR's `glmQLFTest` (R/glmQLFTest.R:152), `estimateDisp.R:194`, and the fit in R/binQLFtest.R:88 | Yes (limma 3.64.3) |
| Robust moderation against hypervariable genes | limma `fitFDistRobustly` (`robust=TRUE`, `winsor.tail.p=c(0.05,0.1)`, returns a per-gene `df2.shrunk`) | not used in production, but testable today | limma `fitFDistRobustly`, R/fitFDistRobustly.R | Yes (limma 3.64.3); RUNS on this machine despite the mixed-BLAS crash (it has no least-squares call — see "Relationship to limma..." environment note below). Our hypervariable tail (imprinted genes PEG10, PEG3, ZDBF2, MEST, GRB10; multi-copy families RNU1-1, RNVU1-28, 45S rRNA, SNORD3D, EEF1A1, PABPC1, SET) is exactly the case this was built for — this is the one remedy of the table most immediately worth trying |
| Per-gene overdispersion pooled across samples from Gibbs/bootstrap draws | edgeR `catchSalmon` (transcript-level RTA — read-to-transcript-ambiguity overdispersion; moderated with `squeezeVar` at prior df 3; floored at 1; `divide=TRUE` divides counts by it) | not used at gene level (see the RTA finding below) | edgeR `catchSalmon`, R/catchSalmon.R:80-106, man/catchSalmon.Rd (Baldoni et al. 2024a,b) | Yes (installed); gene-level `catchSalmonGene` is DEVEL ONLY |
| Same pooling idea, different tool | sleuth: `sigma_q_sq <- rowMeans(all_sample_bootstrap)`, one inferential variance per transcript, smoothed across transcripts and added to a biological component | structurally our `c*v+tau` with `c` fixed at 1 and `v` pooled to one number per gene | sleuth 0.30.2, R/sleuth.R:662 (pooling), R/model.R:452 `final_sigma_sq` (smoothing) | Not installed (external tool) |
| No variance model at all | swish (fishpond): a Wilcoxon rank-sum test (ranks pooled observations across two groups and compares the summed ranks, no distributional assumption) applied across inferential replicates, with permutation for significance | not used | fishpond `swish`, R/swish.R | Not installed |
| Per-library scale `d_i` | edgeR devel `sampleWeights()`: header states the model as "the quasi-dispersion of each observation is s2_g / w_i"; reference estimator is mean adjusted unit deviance per sample across genes, logged, centred, exponentiated | `estimate_library_factors`, `tensorqtl/hapmixqtl.py:1267` (same construction — mean over genes of `a_gi^2/(c_g v_gi + tau_g)` — but normalized to arithmetic mean 1 over samples, `d = d_new / d_new.mean()` at `hapmixqtl.py:1334`; `sampleWeights`'s normalization is described as log-centred-and-exponentiated, i.e. geometric mean 1 — the two are not guaranteed identical) | edgeR `sampleWeights`, R/sampleWeights.R (created 2024-08-05, last modified 2026-07-16, Lizhong Chen and Gordon Smyth) | DEVEL ONLY, not installed |
| Paired-count binomial model for the allelic channel | edgeR devel `binQLFit` + `PCList` container (genes x samples, two counts per observation; empirical-Bayes moderated quasi-dispersions; per-observation weights matrix; user-supplied `covariate.trend`; `robust=TRUE` by default) | not used; matches our allelic channel's data shape exactly (edgeR's intended use is methylation) | edgeR `binQLFit`/`PCList`, man/binQLFit.Rd (Lizhong Chen and Gordon Smyth) | DEVEL ONLY |

**Why pooling per gene (catchSalmon, sleuth) or per donor (limma array
weights) cannot replace our per-gene-per-donor `(c_g, tau_g)` fit.** Measured
2026-09-18 on our own cached per-draw arrays, 34,457 genes x 92 donors x 200
draws, by
`/mnt/ssd/lalli/brainvar_hapmix_deploy/variance_layer_mapping_20260918/variance_layer_measurements.py`
(not under git, cite by path): `v_ig` is a donor-by-gene interaction. Within
a gene across donors, log `v` has median sd 0.77 (residual sd at matched
allele-resolved read count 0.557). CORRECTED 2026-09-18: the covariate
regressed against is allele-resolved reads `mL+mR`, not sequencing depth —
the script's own variable names (`residual_sd_at_matched_depth`, `r2_depth`)
used "depth" loosely for `log(mL+mR)` (`variance_layer_measurements.py:33`
`tot = mL + mR`, `:82` `ld = np.log(tot[ig][m])`, regressed against `lv =
np.log(Va[ig][m])`). Allele-resolved read count explains only R^2 = 0.32 of
it (at matched read count `v` still spans 1.7-fold between donors of the
same gene) — the conclusion survives this correction and is arguably
stronger for it: what remains once read COUNT is held fixed is how
INFORMATIVE those reads are, i.e. heterozygosity, not depth. The per-donor
mean of read-count-adjusted log `v` has sd only 0.073 across the 92 donors. So
`v_ig` is neither a gene property (pooling across samples into one
number per gene, as `catchSalmon`/sleuth do, cannot hold it) nor a donor
property (a per-sample array-weight factor cannot either) — only the full
gene-by-sample weight matrix can. The allelic channel carries about three
times as much of its structure at the gene-sample level as the total channel
does: within-gene/between-gene median sd of log `v` is 0.41/2.12 (ratio 0.19)
for the total channel, 0.72/1.29 (ratio 0.55) for the allelic channel.

**`library_scaled`'s per-library factor `d_i` tracks library size, not
heterozygosity — re-established 2026-09-18.** An earlier same-day claim
reported two correlations as independent confirmation that `d_i` tracks
library size when they were, in fact, the same array (both effectively
`mL+mR`-derived); that claim is superseded by this one, run with a properly
independent library-size covariate. Spearman correlation (a rank
correlation: the Pearson correlation of the two variables' ranks, robust to
nonlinearity) between `estimate_library_factors`'s fitted `d_i` and library
size computed from the separate `YT` per-draw total-expression array: +0.64.
Spearman(`d_i`, allele-resolved reads `mL+mR`): +0.63. Spearman(`d_i`,
heterozygosity share — the fraction of a donor's genes with `mL+mR>0`):
-0.06. Partial Spearman (the correlation between two variables after
linearly removing a third variable's contribution from both) of `d_i` with
library size controlling for heterozygosity: +0.63; of `d_i` with
heterozygosity controlling for library size: +0.02. Library size and
allele-resolved reads are 0.96 collinear, so the original two-correlation
test could not have separated them; the new evidence, not the old, is what
this conclusion rests on. This is a different, independently-valid
measurement from `docs/hapmixqtl_methods.md`'s `d_i`-vs-log-mapped-fragments
Spearman 0.66 (real BrainVar sequencing metadata, not a cached-array
covariate) — the two agree in direction and are not in tension; leave that
file's number as-is. Net: `d_i` is doing its intended job (absorbing true
per-library scale) and is not accidentally a proxy for how much allelic
information a library happens to carry.

**Why `squeezeVar` cannot be applied to `tau_g` directly.** `squeezeVar`
requires a positive variance estimate with known degrees of freedom under a
scaled chi-square sampling distribution. `tau_g` is a signed difference of
moments and is non-positive for 40-47% of well-expressed genes (already
documented above as the root of the spike-at-zero problem in the
`variance_prior` bullet). That is a structural reason, not only an empirical
one, for why `estimate_variance_priors`/`_trend_prior` run their own
Fisher-scoring fit in `(log c, log tau)` rather than calling `squeezeVar`.

**`squeezeVar` itself runs fine on this machine (it never touches the
broken weighted-least-squares path below), and running it measures two
things: that tau is real, and that its own moderation is nearly inert here
at N=92.** Measured 2026-09-18
(`/mnt/ssd/lalli/brainvar_hapmix_deploy/variance_layer_mapping_20260918/run_squeezevar.R`
and `s2_for_squeezevar.tsv`, not under git). On the allelic channel under
pure inferential weights `w = 1/v` (the known-wrong shape, kept deliberately
simple here: with the through-origin null the residual is `a_i` and `d_g` is
the number of informative donors), the per-gene scale `s_g^2` has quartiles
0.72/1.34/2.10 — it would be 1.00 throughout if the Gibbs draws explained
all the scatter, so the excess over 1.0 and its between-gene spread ARE
`tau` showing up as scale heterogeneity: the second variance component is
what the data demand, not an invented one. `d_g` (informative donors per
gene) runs 40/72/92 (quartiles), and `limma::squeezeVar(s_g^2, df=d_g,
robust=FALSE)` returns prior degrees of freedom `d0 = 2.00` with
`s0^2 = 0.688`; the share of a gene's moderated variance coming from the
prior is `d0/(d0+d_g)`, a median 2.7% here. So classic single-scale
empirical-Bayes moderation is nearly inert at BrainVar's depth: limma's
cross-gene borrowing is powerful when `d_g` is 2 to 5 (the microarray/small-
RNA-seq regime `squeezeVar` was built for) and there is very little left to
borrow once `d_g` reaches 72. `robust=TRUE` on the same vector gives a
near-uniform per-gene `d0` (median 2.64) with no gene below 1, i.e. no
outlier protection triggers — unsurprising when moderation is already
contributing only ~3%. CAVEAT: this is the scale under pure `1/v` weights,
which is the wrong shape for us (above); under the fitted `c_g v + tau_g`
weights the weighted residual scale is pinned close to 1 by construction
rather than by evidence — the weights were fit from these same residuals,
so a scale near 1 mostly confirms the fit converged, not that the weights
are right (the paragraph below spells out why) — and a `squeezeVar`
`d0` computed there would mean something different (how much of
`(c_g, tau_g)` itself to shrink, which is what `estimate_variance_priors`
already does with its own prior rather than `squeezeVar`, per the paragraph
above).

**Robustness check on the `squeezeVar` diagnostic: a gene set chosen by
expression, not heterozygosity, gives the same numbers.** The `d_g`
quartiles 40/72/92 above come from the >=40-informative-donors filter
(16,674 genes, the allelic channel's own heterozygosity-driven admission
rule). Applying the standard expression filter instead — drop a gene if
more than 20% of samples have zero total expression — gives 15,499 genes,
a different set by a different criterion, and does NOT change the headline
result: `d0` = 2.00 either way, moderation share 2.7% vs 2.6%, `d_g` median
72 vs 74. The per-gene scale median moves 0.99 -> 1.09. The two filters are
NOT nested: 2,799 genes pass the expression filter but fail the
>=40-informative-donors rule, because a gene can be well expressed in every
donor and still heterozygous in few of them. That asymmetry is the same one
the 57.8% bullet above describes at the level of individual donor-gene
pairs: expression filtering governs admission to the TOTAL channel;
heterozygosity governs admission to the ALLELIC channel; passing one says
nothing about passing the other.

**hapmixQTL's layer 1 is the only one in this whole comparison that is not
fixed before a gene's own residuals are seen, which is categorical, not a
matter of degree, and it has two measured consequences.** Every other row
of the reference table above sets its per-observation variance in advance:
limma at prior weights or `w=1`; `vooma`'s trend fit across all genes before
being applied to any one of them; edgeR's quasi-likelihood pipeline from the
fitted mean; `catchSalmon` and sleuth from Gibbs/bootstrap draws pooled
across samples, not from this gene's residuals. hapmixQTL fits its
per-observation variance FROM the gene's own squared residuals under every
`variance_model`: `tau_g` alone via `_estimate_tau_informative`
(`tensorqtl/hapmixqtl.py:586`, the shipped `additive` default's moment
estimator, run on the gene's own whitened residuals) with `c` fixed at 1, or
`(c_g, tau_g)` together (the regression of squared residuals on `[v, 1]`)
under `two_component`/`library_scaled` — and then uses that fit to weight
those same residuals. This is the same objection an adversarial review
raised on 2026-09-18 — and rejected — against a simpler one-parameter
reparametrization (`sigma_g^2 = tau_g`, `w = 1/(1 + v/tau)`; this note is
that review's record, there is no separate report file): making the weight depend on the
scale the model is meant to estimate breaks the premise that makes a
moderated t-statistic exact (a t-statistic whose denominator variance is
the `squeezeVar`-style empirical-Bayes posterior above, not the raw
per-gene estimate — exact only when the weights that produced the residuals
were fixed independently of them). Worth naming precisely: `1/(1+v/tau) =
tau/(tau+v)`, which is the shipped `additive` weight `1/(v+tau)` up to a
gene-constant factor of `tau_g` — same functional form, same one circular
parameter. `two_component`/`library_scaled` go further, with `c_g` ALSO
free and ALSO fit from the residuals it then weights — two circular
parameters where the rejected proposal, and the shipped `additive` default,
have one. That is the sense in which the objection applies with more force
to the two-parameter models.

Consequence one, exact algebra, and here `additive` and the free-`c` models
diverge: under `two_component`/`library_scaled`'s clamped fit, `(c_g,
tau_g)` are both free per gene, so rescale every `v_ig` in a gene by a
constant `k` and the fit returns `c_g/k` with `tau_g` unchanged, giving
`(c_g/k)(k v_ig) + tau_g = c_g v_ig + tau_g` — identical weights. Whatever
absolute units the Gibbs draws are in is discarded; only the WITHIN-GENE
SHAPE of `v` across donors survives. This does NOT hold under the shipped
`additive` default: there `c=1` is fixed, not fitted
(`_channel_weights`, `tensorqtl/hapmixqtl.py:1383`, literally returns `1.0`
for `c` on that path), so `1/(kv+tau) != 1/(v+tau)` in general and
`additive`'s weights DO feel the draws' absolute scale — this is the same
choice sleuth makes with its fixed `c=1`, and it is why sleuth's choice is
principled rather than a simplification, and why `additive`, not
`two_component`/`library_scaled`, is the scale-respecting member of this
model family. That is not an endorsement of `additive`: trusting the draws'
scale at a fixed `c=1` means trusting a scale the data say is off (measured
`c` is 2.6 median on the 29 calibration genes, 1.8 transcriptome-wide,
below), which is why the fix further below is a global or trend `c`, not
reverting to `c=1`. The invariance is real, but it is a property of the free-`c`
models specifically; even there it is only approximate under the production
`variance_prior` shrinkage, because each gene's posterior-mode `c_g` is
pulled toward its expression bin's prior mean, and that mean is estimated
from the OTHER genes in the bin (`estimate_variance_priors`,
`tensorqtl/hapmixqtl.py:1024`) and does not itself rescale when one
gene's `v_ig` does — the shrinkage prior is the one place the free-`c`
models' draws' absolute units still reach the weights. CAVEAT on the
caveat: `c_g` itself is still identified as a regression slope given the
observed `v` values (a dimensionless, base-independent quantity — unlike
`tau`, which is in squared natural-log units in current code), so a
measured `c_g` of transcriptome-wide, production shrunk fits, above 1,000
allele-resolved reads/donor (median near 1.8 — the same 2026-09-18 session
computation as the allele-resolved-reads table below, not yet folded into
`variance_layer_mapping_20260918/`; a different population from the
29-calibration-gene `c` median 2.6 cited above) remains a real statement
about the draws under this model. What the algebra makes invariant is the
INFERENCE (the fitted weights and everything downstream of them) under the
free-`c` models, not the parameter `c_g` itself.

Consequence two, measured 2026-09-18 as a session computation (not yet
folded into `variance_layer_mapping_20260918/`; to be added): where in
allele-resolved-read space do the fitted weights actually track the draws
at all? (RELABELED 2026-09-18: this tier variable is allele-resolved reads
`mL+mR` per donor, the same quantity as `R` above, not total expression or
sequencing depth — see the 57.8% bullet earlier in this file.) Using the
production `variance_prior`-shrunk per-gene fits over their informative set
(1,174,211 donor-gene datapoints with `v_ig > eps`, 16,674 genes — the
informative subset, not the full 34,457 x 92 grid the other measurements
above use), 57.5% of datapoints have `c_g v_ig > tau_g` overall. By tier of
median allele-resolved reads per donor (share with `c*v > tau`; the
10th-90th percentile spread of log weight across a gene's donors, median
over genes; the same spread of `log(1/v)`; the ratio of the two, i.e. how
much of the draws' own spread the fitted weights retain):

| Median allele-resolved reads/donor | c*v > tau | log-weight spread | log(1/v) spread | Retained |
|---|---|---|---|---|
| under 10 | 36% | 0.03 | 0.55 | 6% |
| 10-30 | 18% | 0.01 | 1.01 | 1% |
| 30-100 | 34% | 0.28 | 1.51 | 19% |
| 100-300 | 61% | 1.19 | 1.90 | 63% |
| 300-1000 | 80% | 1.57 | 2.00 | 79% |
| 1000+ | 89% | 1.94 | 2.22 | 87% |

So below 100 allele-resolved reads/donor — 7,455 genes, 45% of the
transcriptome — the quantification uncertainty is almost entirely flattened
out by the `tau_g` floor and donors are weighed nearly alike regardless of
what the draws say; above 300 allele-resolved reads/donor the weights track
the draws closely. This is how much
of the draws' WITHIN-GENE SHAPE survives the floor into the weights, which
is a separate quantity from the approximate-invariance gap above (that gap
is about the draws' ABSOLUTE scale reaching the weights through the
shrinkage prior; this table is about how much of their within-gene shape
gets through at all).

**Both defects point toward one fix, which strengthens the case for the
pooled trend (the `vooma`/`voomaLmFit` row of the table above, and "Two
cases..." in `docs/CURRENT_SCIENTIFIC_STATE.md`) from "more stable" to
"restores the premise every layer-1 method in the comparison depends
on" — but closing the circularity, specifically, takes more than one
parameter.** A global or smooth-in-expression `c` alone restores (b), the
draws' absolute scale reaching the weights, but NOT (a), the circularity:
with `tau_g` still fit per gene from the residuals it then weights, the
layer-1 variance `c*v + tau_g` still depends on this gene's own residuals.
Closing the circularity needs the FULL `vooma` form: both the coefficient
and the floor fixed in advance from a between-gene trend
(`lm.fit(cbind(1,sx,sxc), sy)`, where the intercept is the floor and the
`sx` slope is the coefficient — both estimated across all genes, not from
this gene's own residual), with whatever per-gene freedom remains moved
entirely to a layer-2 residual scale estimated GIVEN those fixed weights,
which `squeezeVar`-style moderation can then legitimately shrink (its
premise — weights fixed independently of the residuals — is exactly what
this restores). Both curves this needs already exist in the codebase,
unused as the layer-1 function: `_trend_prior` (`tensorqtl/hapmixqtl.py:946`)
fits a smooth curve of `log c` and, separately, `log tau` on `log10` reads
across ALL genes today, but only as the `variance_prior` shrinkage's prior
MEAN for a still-per-gene fit, not as the coefficient and floor themselves.
Using them that way — as the layer-1 function rather than the prior — is
exactly hapmixQTL's own `vooma`-style pooled-trend route, and it is the one
restructuring that closes both defects at once. This is an argument for
prioritizing that measurement, not a substitute for it: the `vooma` caveat
already recorded in `docs/CURRENT_SCIENTIFIC_STATE.md` (between-gene trend
coefficients need not agree with within-gene slopes) is untouched by this
argument and still needs testing.

**RTA overdispersion is at its floor at gene level, which independently
confirms the shot-noise finding above.** Running edgeR's RTA estimator on our
own gene-level draws (3,000 genes sampled from the transcriptome, by
`/mnt/ssd/lalli/brainvar_hapmix_deploy/variance_layer_mapping_20260918/rta_vs_c.py`,
not under git): overdispersion quartiles 1.00/1.00/1.01, 61.4% exactly at
the floor of 1, 95th percentile 1.13, max 35.6 — at gene level RTA has
almost nothing to do, which is
consistent with `catchSalmonGene` having arrived separately from the
transcript-level `catchSalmon`. Separately, the across-draw variance of our
log-total statistic (natural log, current code) is 0.98x (IQR 0.91-1.05)
what an RTA-inflated Poisson predicts — an independent, differently-derived
confirmation, from a different direction, of the existing `count_noise`
bullet's claim above that Salmon's default Gamma draw carries shot noise
(the counting simulation there already quantified it once: Gibbs-only
predicted/observed variance 0.944).

**Gibbs draw count is adequate to treat `v` as known.** Measured by the same
`variance_layer_measurements.py`
(`/mnt/ssd/lalli/brainvar_hapmix_deploy/variance_layer_mapping_20260918/`,
not under git): 200 draws; lag-1 autocorrelation median 0.081; median
effective draws ~170; the relative sd of `v` as an estimate is median 0.109,
with only 3.2% of datapoints above 0.25. Treating `v` as known (rather than
itself uncertain) is a good approximation, and errors-in-variables
attenuation on `c_g` from doing so is small.

**RASQUAL's beta-binomial overdispersion rho is an available external check
on `tau_a`.** RASQUAL models the allelic count as beta-binomial (a binomial
whose success probability is itself Beta-distributed across donors, giving
extra-binomial variance `rho*p(1-p)` beyond the binomial's own `p(1-p)/n`);
`rho` is that extra-binomial fraction. By the delta method on the log odds
at p=1/2, `tau_a = 4*rho` (natural-log scale). Our fitted `tau_a` above 1,000 reads (0.0030 clamped,
0.0049 under the decile prior; natural log) implies `rho` between 0.00075
and 0.0012. Separately, `sqrt(tau_a)` itself (0.055 to 0.070, natural-log sd
of the allelic log-ratio) is approximately a 5.5-7% donor-to-donor
multiplicative spread in the L/R allelic ratio (small-value log
approximation, `log(1+x) ~ x`). `best_rasqual_row` in
`scripts/compare_pipelines.py:389-422` currently reads RASQUAL's 1-indexed
fields 3-6 (chrom/pos/ref/alt), 11 (chi2), 12 (pi), 14 (phi) and 23
(convergence status), plus field 2 to detect a `SKIPPED` row, and retains
`gene/stat/log_afc/phi/status/lead` (confirmed on 2026-09-18 against
`/mnt/ssd/lalli/brainvar_hapmix_deploy/pilotI/observed_rasqual.tsv`).
RASQUAL's own vendored documentation, `rasqual_src/README.md:64`, lists field
15 as "Overdispersion" (repo-verified on 2026-09-18, not merely asserted), but that
has not been cross-checked against parsed RASQUAL stdout in a run here —
retaining that one more field in the comparison driver would enable the
check against `tau_a`.

**A gene-by-sample quantification-uncertainty correction for personalized
transcriptomes already exists in this project's RNA pipeline**, and predates
`catchSalmonGene`: `calc_expression_stats.R` (dated 2025-05-02, confirmed
present on 2026-09-18 at both
`/mnt/ssd/lalli/nf_stage/RNA_reference_comparison_results/reference_comparison_results/bv2/T2T_NCBI110_pseudoalignment/expression_results/` and the `GRCh38_p14_NCBI110_pseudoalignment` sibling), run as a Nextflow process.
Its `getGeneOverdispersion` function adapts `catchSalmon` for personalized
transcriptomes where the transcript set differs between samples: it sums
per-draw transcript counts to gene level within each sample, supports
`merge_alleles`, and keeps `OverDisp` as a gene-by-sample matrix moderated
per sample (`colMedians` for the prior, `DFPrior=3`), then divides counts by
it element-wise (`correct_se_for_overdispersion`/
`correct_txi_for_overdispersion`). hapmixQTL reaches the same information
independently, from the raw per-draw arrays. The RTA-vs-Poisson finding above
(0.98x) says these are the same quantity on different scales, so the two
implementations are checkable against each other — not yet done.

**The mixed-BLAS crash on this machine blocks any limma path that fits a
linear model, not just a weighted one — tested function by function,
2026-09-18.** The BLAS is Debian's
`/usr/lib/x86_64-linux-gnu/openblas-pthread/libblas.so.3` while the LAPACK is
a Homebrew openblas (a mixed-BLAS crash). Each candidate function was run in
its own R process, so a crash in one could not hide the others. CRASHES:
`limma::lmFit` (segfaults even UNWEIGHTED, not only with a weight matrix
supplied — `lm.wfit` was simply the first crash hit, not the boundary
itself), `limma::arrayWeights`, and therefore `vooma`, `voomaLmFit` and
`voomWithQualityWeights`, all of which fit a linear model internally. RUNS:
the entire empirical-Bayes moderation family, which is closed-form and never
reaches a least-squares routine — `limma::squeezeVar` (equal or unequal df,
`robust=TRUE` or `FALSE`), `limma::fitFDist`, `limma::fitFDistRobustly`, and
`limma::fitFDistUnequalDF1` all ran without incident. In particular
`fitFDistRobustly` — the published fix for hypervariable genes, our
imprinted/multi-copy tail's target case (table above) — is testable on this
machine today, not blocked.

**Practical consequence for what to try next.** Of the two limma-native
routes this comparison identified, the pooled-trend route (`vooma` with a
Gibbs-derived predictor, fit BETWEEN genes then applied WITHIN each gene)
cannot be attempted on this machine until the BLAS is fixed, because it
needs `lmFit`/`voomaLmFit` underneath. The moderation route (`squeezeVar` on
one vector of per-gene variances — already run diagnostically above — with
`fitFDistRobustly` or `fitFDistUnequalDF1` for the hypervariable tail) can be
run today. That asymmetry should shape whatever is tried next: the
moderation route is immediately actionable, the trend route is not.

Installed versions are a release behind the upstream devel source read as of
2026-09-18: limma 3.64.3 installed vs 3.99.0 upstream; edgeR 4.6.3 installed
vs 4.99.6 upstream. Available in the installed versions: `vooma`,
`voomaLmFit`, `voomWithQualityWeights`, `arrayWeights`, `squeezeVar`,
`fitFDistRobustly`, `fitFDistUnequalDF1`, `catchSalmon`, `glmQLFit`,
`estimateDisp`. DEVEL ONLY (not installed): `catchSalmonGene`, `binQLFit`,
`PCList`, `sampleWeights`. "Installed" and "runs without the BLAS crash" are
different properties — `vooma`/`voomaLmFit`/`voomWithQualityWeights`/
`arrayWeights` are installed but currently crash here; the CRASHES/RUNS
lists above are the operative ones for what can actually be tried.

## Claims withdrawn on 2026-09-13 — do not re-assert

An eight-angle review retired these. They may survive in older text.

- The additive, multiplicative and nested variance forms were **never** compared:
  no multiplicative arm exists in `tests/`.
- `docs/ase_validation.md` §7b's weight-cap and nested conclusions are void; both
  arms were arithmetically incapable of differing from their comparators.
- The additive `tau` is **not** "the one place hapmixQTL departs from mixQTL".

Corrected facts that replaced them (these are true; do not negate them):

- mixQTL's total channel already carries a flat additive variance, so `v_t + tau_t`
  decomposes the parent rather than departing from it; the multiplicative scale is
  the allele-specific channel only.
- mixQTL's scan refits its dispersion at **every** variant, so hapmixQTL's per-gene
  null-model `tau` is the departure and `tau_refit` moves back toward the parent.
  mixQTL's allelic regression is also through the origin (`y ~ -1 + x`).

Withdrawn on 2026-09-16 (measured in `estimator_ablation_20260916`):

- The stated reasons for `count_noise=True`: "Gibbs across-draw variance is
  read-assignment uncertainty only" (false: Salmon's default Gamma draw
  carries shot noise, CollapsedGibbsSampler.cpp:122) and "a sample with
  unambiguous reads has unanimous draws and would be discarded" (impossible
  under default flags: identical draws occur only with zero reads). The flag
  stays on for the reason above, which is a missing floor, not a variance
  term.
- `count_noise=False` dropping zero-read samples from the total channel: wrong,
  they stay in at weight `1/(1e-8 + tau_t)` (the guard is ASE-only).
- "The pooled 5% threshold calls 8 genes against 5 by the per-gene null" as a
  property of the data: it was the DL `tau` scale inflation of three genes.

## Known and unfixed

- The lead refit is biased by selection: appending the window maximum to the `tau`
  design removes far more residual sum of squares than the one degree of freedom it
  is charged, so `tau` comes back 11-15% low under the null and the reported
  statistic is inflated. Gene-level p-values are unaffected.
- The Beta approximation is conservative in the tail, which costs power at
  transcriptome-scale thresholds.
- No per-sample allele-specific read floor, where mixQTL used 15 reads.
- `_estimate_tau` is DerSimonian-Laird, whose allelic scale is 22% low at the
  median (above); Paule-Mandel is measured, not adopted. The residual shape
  (`c*v + tau`) is no longer untested: it is implemented and measured as the
  `two_component`/`library_scaled` variance models (2026-09-17, above, "Three
  variance models and an empirical-Bayes prior"), just not the default
  (`additive` is).
- The total channel has no zero-count guard; `count_noise` is standing in for
  a floor (above). A coverage-based floor for zero-count total samples, then
  no `q` for samples with reads, is the fix, not the flag.
- Three input-validation defects found by the 2026-09-14 audit are unfixed
  and unarguable: a NaN at a zero-weight sample drives `pval_perm` to the
  `1/(nperm+1)` floor; variant-row identity between the genotype and phase
  frames is not checked beyond column order; the QR is taken on a
  rank-deficient design without a rank check.
- After the through-origin change, `_joint_gls`/`_pvals` (the robust
  second-pass path) still charge the ASE channel `1 + n_cov` columns, so the
  robust SE there is ~6% conservative; `map_cis`/`map_nominal` are unaffected.
- Closed 2026-09-17: `test_combined_se_calibrated_under_heterogeneous_gibbs_variance`
  (tests/test_hapmixqtl_calibration.py) shows the combined SE stays calibrated
  when donors span two orders of magnitude in `v_inf` with a-t noise
  correlated at 0.9: null mean t^2 1.023 with correlation 0.9 and 1.023 with
  0 (20,000 genes each), corr(beta_a, beta_t) -0.009, type-I 0.051 / 0.011.
  Ignoring `Cat` is safe in BrainVar's regime; the 2% residual inflation is
  the tau plug-in, not the covariance. Two side facts: the null-design tau
  over-covers a planted effect (0.98) by construction, which is why the lead
  refit exists, and the refit SE is understated by 1-3% in variance
  (unresolved at 8,000 genes).

## Estimator debates go through the ablation

Every estimator question since 2026-09-13 has been answerable in minutes on
the 29 calibration genes: `estimator_ablation_20260916/ablation29.py`
reproduces `null_calibration_29b` byte for byte and reruns the hapmixQTL arm
with the choice toggled, with 40 null draws. Before writing a theory document
about the variance model, add a configuration there and report the numbers.

## Self-tests

Every script self-tests on fabricated data and prints `SELF-TEST OK`:

```bash
pytest tests/test_hapmixqtl.py tests/test_hapmixqtl_calibration.py tests/test_cli.py -q
RASQUAL_BIN=rasqual_src/src/rasqual python3 scripts/compare_pipelines.py --selftest
python3 scripts/run_hapmixqtl_from_salmon.py --selftest
```

The comparison driver's self-test covers the RASQUAL arm, the matched-variant effect
comparison, and that a rerun reuses its null-round checkpoints byte-identically.

## The mixQTL replication arm, and what the draws buy (2026-09-19)

`tensorqtl/mixqtl_replication.py` is a NumPy port of the published mixQTL
estimator (`hakyimlab/mixqtl` @ `624ae44`) with all eleven divergences from
the 2026-09-14 review removed. It consumes Salmon posterior-mean counts and
never touches the draws, so it doubles as the **no-draws comparator**. Driver
`scripts/compare_mixqtl_replication.py`, analysis
`scripts/analyze_mixqtl_comparison.py`, report
`brainvar_hapmix_deploy/mixqtl_replication_20260919/REPORT.md`.
Before this, mixQTL had never been run: `compare_pipelines.py` is
hapmixQTL-vs-RASQUAL.

- **No cross-language check exists and none is possible here.** `libR.so`
  links `libblas.so.3` and `libopenblas.so.0` together; `lm()`, `%*%` and
  `crossprod()` all segfault, `tensorA`/`glmnet` are absent, no sudo. The
  port's algebra is validated per variant against `numpy.linalg.lstsq` to
  1e-10 and every cutoff/cap/dof rule is pinned to the R source line it
  encodes (23 tests). Exact reproduction of the published code is NOT
  claimed.
- **THE DRAWS DO IMPROVE THE POINT ESTIMATE.** Holding response, donor set,
  variants and design fixed and varying only the weights, `1/v` weighting
  cuts `var(beta_hat)` across 40 null permutations to **0.340** of unweighted
  (25/29 genes, sign p=1.0e-4) — 1.71x in SE, 2.9x in variance. It beats
  mixQTL's published capped harmonic weights (ratio 0.499, 23/29) and
  uncapped harmonic (0.763, 21/29). Kish effective n falls 77 -> 45 while
  variance drops 2.9x, which is why this is not a concentration artifact.
- **It replicates across seeds.** Re-running the whole ablation at an
  independent master seed moves every arm by <= 0.024 in the ratio and the
  per-gene Gibbs/OLS ratio correlates at r=0.997 over 29 genes. The gaps
  that carry the conclusion (0.18, 0.26) are 7-11x that. The ONE pair that
  does not separate is capped Gibbs 0.780 vs capped harmonic 0.826 (gap
  0.046, ~2x the shift): read those as close, not ordered. `compare_seeds.py`.
- **The anticonservatism is in the SE formula, not the weights.** With a
  fitted residual scale the Gibbs arm calibrates at 1.033; with the shipped
  known-variance `1/sqrt(xx)` it reads 3.313, i.e. the reported SE
  understates realized error by 1.82x. 3.313 is an UPPER bound: 6 of the 29
  genes carry signal, which permutation turns into spread.
- **The excess is multiplicative, not additive.** Profiling `1/(v + tau)`
  over a tau grid (selected by realized spread, not fitted from the
  residuals it weights): flat 0-0.25x v_med, then monotone worse (1.090 at
  1x, 1.430 at 8x). A floor of the size 3.313 would imply if additive is
  excluded — tau=2x is 18% WORSE than tau=0. Per gene it is not identifiable
  (40 perms give ~23% noise on a variance; 9/29 genes gain >5% at some
  tau>0 at BOTH seeds, about what chance gives against 8 alternatives).
  This is the `c*v` form with free per-gene `c`, which is exactly limma's
  `fit$sigma`, and it cancels from a within-gene permutation p.
- **`count_noise`'s q is inert for weighting**: q-on vs q-off efficiency
  0.3402 vs 0.3410, a 0.2% difference. CAUTION on the sign test here: it
  reads 21/29 (p=0.024) at seed 42 and 19/29 (p=0.14) at the other seed. A
  sign test responds to DIRECTION regardless of magnitude, so a p that
  crosses 0.05 between seeds while the effect stays at 0.2% means inert, not
  real. q moves absolute scale (3.313 -> 3.732) only. Second, independent
  confirmation of the 2026-09-15 double-count finding.
- **mixQTL's fold cap costs two thirds of the gain** (0.340 -> 0.780) and is
  a workaround for a known-variance SE that a fitted sigma already makes
  unnecessary.
- Two defects in the distributed reference, both reproduced under
  `strict_reference_cap=True` and both documented in the module docstring:
  `matrix_ls_asc_permutation` zeroes cutoff-failing weights before taking the
  min for the fold cap, so ANY cutoff failure zeroes every weight and all
  permuted betas are 0/0 (the non-permutation path subsets first and
  escapes); and `floor(n/10)` makes the cap 0 for 3-9 passing samples (past
  the `sample_size > 2` guard) and 1 for 10-19, where the channel becomes
  plain OLS.
- **Gene-level calibration**: hapmixQTL's within-gene permutation p rejects
  at 5% in 0.0267 of 1,160 gene-draw pairs (se 0.0047; vs nominal p=1.1e-4,
  median p 0.655) -- conservative, the documented consequence of permuting
  residuals while leaving each donor's weight in place. The mixQTL port
  through the CORRECTED permutation path rejects in 0.0483 of 290 (se
  0.0126; vs nominal p=1.0, median p 0.4925), indistinguishable from
  uniform, and replicating across seeds (0.0448 / median 0.4950 at the
  other). Their DIFFERENCE is 0.0216 +/- 0.0134, z=1.60: it does not clear
  its own floor at these permutation counts, so compare the median p-values
  rather than the 5% rates. The published path yields no number at all here.
  Note the hapmixQTL side is read from the existing ablation at that
  harness's seed, not re-run at 42 -- legitimate only because the arms are
  compared as marginal distributions, never draw by draw.
- End-to-end on the 29 genes is the weak half, and HOW weak depends on the
  cutoff setting, which is why the two are now reported separately (REPORT.md
  regenerated 2026-09-20 with mixQTL's published cutoffs primary). Under the
  PUBLISHED cutoffs (`100/50/10/1000`, the GTEx v8 driver that produced the
  paper): lead agreement among the 6 called genes 0/6, median lead LD r^2
  0.160, Spearman of the per-gene statistic 0.183 (p = 0.34), beta Pearson r
  0.562. Under the permissive R-signature cutoffs (`20/5/100/5000`), kept as a
  sensitivity arm: 1/6, r^2 0.744, Spearman 0.517 (p = 0.004), r 0.697. The
  collapse has an identified cause rather than being noise: mixQTL's upper cap
  `y <= 1000`, stated in its Methods as an alignment-artifact guard, removes
  1,656 of 2,193 informative donor-gene pairs when applied to Salmon
  posterior means (the lower `y >= 50` cutoff removes only 38), leaving 18 of 29
  genes total-counts-only and 7 with no allelic donors at all. At these cutoffs
  the two arms are largely not measuring the same thing, so the permissive
  column is the better comparison of the two METHODS and the published column
  the better comparison against mixQTL AS RUN. A signal-bearing gene set is
  still needed to sharpen either. Scope limit: these are high-coverage genes,
  17.8% uninformative donor-gene pairs against 57.8% transcriptome-wide, so
  the 3x is not shown to transfer to low-count genes where the Gibbs and
  Poisson weights converge.
- The weighting ablation, the residual-floor profile and the SE calibration
  are BYTE-IDENTICAL under the two cutoff settings, verified by re-running and
  diffing on 2026-09-20, not merely argued. They run on hapmixQTL's
  informative-donor set (40-92 donors/gene), which mixQTL's count cutoffs never
  reach, and the only cutoff parameter they touch is `weight_cap` through
  `cap = min(weight_cap, floor(n/10))`; at <= 92 donors `floor(n/10) <= 9`,
  strictly below both candidate values (10 and 100), so the realized caps are
  4-9 and `weight_cap` can never bind below 100 donors. Every headline
  efficiency number (0.340, the 3.313 calibration, the tau profile) is
  therefore simultaneously a published-cutoffs and a permissive-cutoffs result.
  `weight_cap` is set to the driver's 10 (user decision, 2026-09-20); the
  roxygen examples' 100 would change nothing here.

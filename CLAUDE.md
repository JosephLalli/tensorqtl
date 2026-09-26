# tensorQTL, hapmixQTL branch

A fork of tensorQTL adding **hapmixQTL**: cis-eQTL mapping from haplotype-resolved
expression posteriors (Salmon Gibbs draws against a personalized diploid
transcriptome), with the quantifier's inferential uncertainty carried into the
standard error. hapmixQTL is an **extension of mixQTL**.

## THE TWO MODES. There are only two.

Everything else has been deprecated and quarantined (2026-09-23, user
decision). Do not add a third, do not reintroduce a removed one, and do not
report a comparison against one.

### (a) mixQTL mode — the published estimator, no draws

`tensorqtl/mixqtl_replication.py`, a NumPy port of `hakyimlab/mixqtl` @
`624ae44` with all eleven divergences from the 2026-09-14 review removed. It
consumes Salmon **posterior-mean counts** and never touches the draws, so it
doubles as the **no-draws comparator**: it is what hapmixQTL has to beat, and
the only honest measure of what the draws buy. Published cutoffs
(`100/50/10/1000`, the GTEx v8 driver that produced the paper) are the primary
setting; `weight_cap` is 10. Driver `scripts/compare_mixqtl_replication.py`,
analysis `scripts/analyze_mixqtl_comparison.py`.

### (b) default mode — `Var(eps_i) = sigma^2 * v_i`

TIMES, not plus. `v_i` is the **Gibbs across-draw variance**, used as a SHAPE
only; `sigma^2` is the residual scale fitted per variant. There is no additive
floor. Reached by `tau_mode='zero'` + `se_mode='fitted'`, which are the
defaults everywhere and are not worth overriding. Driver
`scripts/run_hapmixqtl_from_salmon.py`, which offers no other hapmixQTL
configuration.

Say **"Gibbs variance"** and **"Gibbs draws"**. Never "bootstrap" — that word
belongs to sleuth and to Salmon's `--numBootstraps`, which this cohort does not
use (200 Gibbs draws).

### What was deprecated, and why it does not come back

Quarantined: `variance_model` (`additive`, `two_component`, `library_scaled`),
`variance_prior` (`deciles` and `trend`), `tau_mode='estimate'` which they
require, and the known-variance standard error `se_mode='model'`. Code in
`tensorqtl/fitted_variance.py`, tests in `tests/fitted_variance/`, reports and
result files in `/mnt/ssd/lalli/brainvar_hapmix_deploy/deprecated_models/`
(renamed from `fitted_variance/` on 2026-09-23, because that name read as the
shipped `se_mode='fitted'` when it meant the opposite: a variance FUNCTION
fitted per gene from its own residuals. The module and test directory keep the
old name; only the results folder was renamed).

Two structural reasons, neither of them empirical:

1. **Circularity.** Each fits its per-observation variance from a gene's own
   squared residuals and then weights those residuals by the fit. No
   comparator does this — limma, edgeR, sleuth and swish all fix the
   per-observation variance before a gene's residuals are seen — and it breaks
   the premise that makes the statistic exact.
2. **The free-`c` models discard the draws' calibration.** With `(c_g, tau_g)`
   both free, rescaling every `v_ig` in a gene by `k` returns `c_g/k` with
   `tau_g` unchanged: the weights are invariant to the draws' absolute scale,
   so only their within-gene shape survives. Propagating the quantifier's
   uncertainty is the entire point, so a model that cannot feel that scale is
   answering a different question.

**Efficiency does not reopen it.** Per-gene cases where a deprecated model
estimates more precisely exist and are recorded in the quarantine. A model
whose weights are fitted from the residuals they weight can win on realized
variance and still be unsound, because the quantity it optimises is not the
quantity it reports. Efficiency was never the objection.

**2026-09-25, user decision: a biological variance term may be EXPLORED,
tentatively, as measurement only.** After the nominal-p mechanism was
decomposed (section "What the 2026-09-25 hypothesis round established": the
fitted scale is the unweighted mean of `a^2/v` while the slope's variance is
governed by the weight-weighted mean, and residual variance grows as
`v^0.65`), the user judged that a need for a depth-independent variance term
"might" exist and authorized exploring it as an extension of default mode.
Conditions, from the proposal the decision was made on: nothing ships and
no third mode appears without a further decision; every candidate must
satisfy the two objections above BY CONSTRUCTION -- the technical coefficient
pinned (`Var = v + tau`, never `c_g v + tau_g`; doubling every `v` must change
the fitted weights) and no record's own residual setting its own weight
except through a cross-gene trend, cross-gene shrinkage with a stated prior
weight and a reference that charges it, or cross-fitting; pre-registered
criteria are pooled rates within the gene-clustered interval of nominal on the
records and sampling nulls, per-gene `R_g` in band for all but ~1 of 46
genes, at least 70% of the `1/v` efficiency gain retained, and TReCASE parity
on the external benchmark after its total channel is repaired. Candidates in
order of how little they change: a closed-form permutation-variance reference
(`se^2 x R_g`); one global shape exponent (`v^-gamma`); a pinned-technical
additive floor trended on record covariates or shrunk per gene; effective
counts into beta-binomial / negative-binomial with shrunk dispersion. A
naked per-gene floor remains excluded. Proposal record:
`brainvar_hapmix_deploy/nominal_p_hypotheses_20260925/` (Codex second-opinion
prompt `codex_second_opinion_prompt.md`).

**First measurement under that authorization, same day
(`scripts/count_scale_weights.py`, adversarially re-run;
`brainvar_hapmix_deploy/count_scale_weights_20260925/`), on the identical
2,000-permutation stream.** Count scale versus log scale is not the issue:
in the TOTAL channel a quasi-Poisson GLM on counts is the Gibbs-weighted
log-scale fit (0.0588 vs 0.0601 at 0.05, paired -0.0012 [-0.0026, +0.0001],
slope variance 0.998), and **unit weights lose no precision there** (variance
1.000 [0.954, 1.049] of Gibbs) while calibrating (0.0496 / 0.0099 / 0.0010):
the total channel's Gibbs weights buy nothing on these genes. In the ALLELIC
channel a quasi-binomial GLM is WORSE (0.0861 [0.072, 0.103]) because IRLS at
the null weights every record n/4 from the fitted proportion and stops
downweighting imbalanced records (+0.0245 [+0.016, +0.034] of the +0.0169
gap comes from that swap). A COMMON dispersion floor is 70x (allelic) / 44x
(total) the median record's counting variance, so it flattens the weights:
on the log scale `Var = v + tau` is unit weights in disguise (variance 1.99x
Gibbs vs 2.07x unit; 0.0537 / 0.0117 / 0.00143, the last two 1.17x / 1.43x
nominal), and per-gene, shrunk (prior df 10) and common rho are
indistinguishable (+0.002, +0.001). The count-scale quasi-beta-binomial
floor (rho 0.0404) is the one allelic arm within its intervals at all three
alphas (0.0520 [0.049, 0.055] / 0.0107 / 0.00125) at 1.50x Gibbs variance,
1.62x after correcting the 0.964 attenuation of a planted effect -- about
half of the 1/v gain, BELOW the 70% criterion. It does not bring the coupling
to its noise floor (sd log R_g 0.046 vs 0.018 model; reversed, conservative,
in about half the genes). Gibbs variance plus the count-based floor
over-corrects (0.0425) because rho from posterior-mean counts already carries
Salmon's assignment ambiguity (per-gene median 0.030 vs 0.005 at
Gibbs-matched effective counts). A weight-independent allelic excess of
0.0553 (1.11x; per-gene max 0.13) survives every arm. Scope: 45 of 46 genes
exceed 1/rho ~ 25 reads, where any floor is flat; the 30-100-read stratum,
worst transcriptome-wide, was not tested; rho and phi were fitted in-sample
on unpermuted counts. Consequence for the candidates: the flat additive
floor (candidate 3, flat form) is out on efficiency; unit weights for the
total channel are measurement-backed at zero cost; candidates 1 and 2 remain
unmeasured.

Anything in `docs/` dated before 2026-09-23 that calls one of these
"production", "default" or "the shipped model" is historical. Those numbers
were correctly measured and are not withdrawn as measurements; only their
status as current practice is.

## Which document answers which question

| Question | Document |
|---|---|
| What is the statistic, exactly, and how do I reproduce it? | `docs/hapmixqtl_methods.md` |
| How do I run the BrainVar deployment end to end? | `docs/brainvar_deploy_runbook.md` |
| What was measured, and how do I know it is calibrated? | `docs/ase_validation.md` — includes withdrawn claims |
| What do the output columns mean? | `docs/outputs.md` |
| What does a new session need to pick this up? | `docs/LOCAL_HANDOFF.md` |
| What is implemented, proposed, validated, running? | `docs/CURRENT_SCIENTIFIC_STATE.md` |
| What was deprecated on 2026-09-23 and why? | `brainvar_hapmix_deploy/deprecated_models/README.md` |
| What is the RASQUAL comparison, and what can it settle? | `brainvar_hapmix_deploy/rasqual_comparison_design_20260923/rasqual_comparison.html` |

## Scientific phase transitions

Before a new substantive scientific phase, run a documentation agent to
validate and reconcile the actual local state into concise authoritative
pointers to the intellectual state, generated files, performed experiments and
results, decisions, existing code, and how to find them. After that agent
completes, resume already-authorized work. State boundaries explicitly:
distinguish implemented behavior, proposed work, validated results, and current
run state. The pass records existing state only; it does not start an
experiment.

## Writing conventions (the user's)

- Define every named statistical method on first use, in terms of what it
  computes (DerSimonian-Laird, Paule-Mandel, Freedman-Lane, Kish, and so on).
- Never write "cell" for a table entry or a (donor, gene) datapoint; in this
  work "cell" means a biological cell. Say datapoint, donor-gene pair, or
  zero-read sample.
- Reports for the user are HTML pages with figures, not long markdown.
- Say "Gibbs variance", never "bootstrap".

## Facts that are easy to get wrong

- **Use log2 for expression, ASE ratios, aFC, and their uncertainty.** Project
  convention since 2026-09-15. beta=1 means a twofold effect. Runtime
  conversion is PENDING: current outputs are still natural logs. See the
  [unit convention record](/mnt/ssd/lalli/brainvar_hapmix_deploy/mixqtl_algorithm_review_20260914/salmon_variance_theory_20260915/LOG2_CONVENTION.md).

- **ASE has no automatic intercept since 2026-09-15.** `ase_covariates_df=None`
  means through-origin with no nuisance columns; the total intercept remains.
  Results recorded before that used the old design and are historical.

- **Sample order is positional.** Phase frames are indexed by the genotype
  frame's column order. `_assert_phase_columns` guards it; before that,
  reordered columns corrupted the allelic channel while the total channel
  stayed correct. `_assert_keep_frames` does the same job for the cutoff masks.

- **Two scales are reported.** `pval_perm` and `pval_beta` are gene-level and
  come from the scan; `slope`, `slope_se` and `pval_nominal` come from the lead
  refit when `tau_refit=True`. A lead's nominal p is never a gene-level p.

- **The detection call is the empirical p**, not the statistic. Statistic
  magnitude has been corrected several times; the called genes did not move.

- **`tau_a`/`tau_t`/`c_a` come back as `None` in default mode, and that is
  correct.** Under `Var(eps) = sigma^2 v` no such parameters exist in the
  model. Reporting `0.0` would wrongly imply an additive component that was
  estimated and came out at zero. Compare with `.equals()`, not `==`: `None`
  is not equal to itself in a pandas object column, which broke a test.

- **Default Salmon Gibbs includes counting noise, and `count_noise` is inert
  on well-expressed genes.** The controlled experiment at
  `/mnt/ssd/lalli/brainvar_hapmix_deploy/salmon_gibbs_counting_sim_20260915/REPORT.md`
  (Gibbs-only variance / MSE 0.944, with `q` 1.906) shows `q` double-counts the
  sampling noise of donors that have reads, and a census of the production
  draws shows the docstring's other case — reads with unanimous draws — never
  occurs (0 of 76,286 datapoints). The flag is nevertheless load-bearing, for a
  reason neither side originally stated: **the total channel has no
  degenerate-sample guard** (`_zero_degenerate_ase_weights` is ASE-only), so a
  zero-count total sample would keep weight `1/(1e-8 + tau_t)`; `q` prevents
  that only because `1/(0 + 2*kappa) = 1` acts as a floor. 39.95% of donor-gene
  datapoints in 2,000 random genes have no reads (none in the 29 calibration
  genes); without `q` such samples hold a median 89.5% of the total channel's
  weight in the 281 of 1,079 sampled genes that have them, against 17.4% with
  `q`. The fix is a coverage-based floor for zero-count total samples that is
  independent of the flag, then dropping `q` for samples with reads. Excluding
  zero-count total samples is wrong: `t = log kappa` is real low-expression
  information, and exclusion conditions on the outcome. Default stays True
  until that is done.

  Quantified 2026-09-18 for the allelic channel's
  `q_a = 1/(mL+kappa) + 1/(mR+kappa)`, from the cached per-draw arrays (34,457
  genes x 92 donors x 200 draws,
  `/mnt/ssd/lalli/brainvar_hapmix_deploy/variance_layer_mapping_20260918/variance_layer_measurements.py`,
  not under git): median `q_a`/`v` is 0.90 at 1-9 haplotype-informative reads,
  0.30 at 10-99, 0.12 at 100-999, 0.07 at 1,000+. So `q_a` matters least
  exactly where reads are plentiful and `v` is trustworthy alone. Its `+kappa`
  pseudocount is the Haldane-Anscombe correction to the empirical log odds (the
  standard small-sample fix for a zero-count donor-gene datapoint in a two-way
  allele split), whose delta-method variance on the natural-log scale is exactly
  `1/(mL+1/2)+1/(mR+1/2)`; edgeR's binomial pipeline adds the identical half
  (`y <- y + ceiling(prior.count)/2`, `R/binQLFtest.R`). The total channel's
  counting term is a different, single-count Poisson delta-method correction,
  not a log-odds one.

  Where both haplotypes are empty, `compute_summaries_from_gibbs`'s `no_cov`
  guard (`tensorqtl/hapmixqtl.py`, `no_cov = (mL+mR)<=0`) forces `Va` to exactly
  `0.0` rather than `Va + q_a`, and `_zero_degenerate_ase_weights`
  (`keep = va_t > eps`) then zeroes the allelic weight entirely — so no
  fabricated observation of perfect allelic balance enters the allelic channel.
  The guard exists BECAUSE `q_a` would otherwise apply there, and it keeps those
  pairs OUT of the fit rather than in it at a floored weight.

- **Why 57.8% of donor-gene pairs have no allele-specific information: a
  Salmon-indexing and pipeline-ingest fact, not a low-expression one.**
  Established and re-verified 2026-09-18 from pipeline code, run metadata and
  the cached summaries. g2gtools emits TWO haplotype copies of every transcript,
  suffixed `_L`/`_R`; on sex chromosomes only the PAR regions get the suffixes
  (`bin/append_suffixes_to_sex_chroms.py` in `/mnt/ssd/lalli/nf_stage/rnaseq_JLL`,
  keyed to the X/Y `seqid`, because PAR alone is genuinely diploid).
  `salmon index` was run WITHOUT `--keepDuplicates`, so where a donor is
  homozygous across a transcript its `_L` and `_R` sequences are byte-identical,
  the indexer keeps one, and the `_R` row never appears in `quant.sf` at all —
  not a zero row, no row. Every `aux_info/meta_info.json` checked records
  `"keep_duplicates": false`; `_L` counts 176,395/176,222/176,226 against `_R`
  counts 111,472/92,348/107,377 (spread 28,674 across 92 donors, the signature
  of collapsed duplicates rather than lost reads), TPM summing to 1e6 within
  floating-point rounding. CAVEAT: `conf/modules.config` conditionally sets
  `--keepDuplicates` when `use_personalized_references` is true, and this run's
  captured params record it true, so the config text alone suggests dedup
  should have been OFF. It was not. `meta_info.json` is the runtime ground truth
  and is what this relies on; why the conditional did not fire is unresolved and
  is an `rnaseq_JLL` question, not a hapmixQTL one.

  Losing the `_R` row does not by itself zero a pair's allelic information.
  What makes `mL` ALSO exactly 0 is the ingest step:
  `scripts/run_hapmixqtl_from_salmon.py`'s `pair_haplotypes` pairs a base
  transcript id only when BOTH suffix rows exist, and `load_counts` accumulates
  `YL`/`YR` ONLY over paired transcripts, so an unpaired (homozygous,
  deduplicated) transcript contributes to NEITHER however many reads its
  surviving row carries. `YT`, by contrast, sums EVERY transcript
  unconditionally. This is exactly why `compute_summaries_from_gibbs` insists on
  a `yT` summed over ALL transcripts rather than `yL+yR`: `yL+yR` IS that
  paired-only subtotal, so using it as the total would silently zero these same
  pairs there too. A pair has `mL=mR=0` exactly when NONE of that gene's
  transcripts have a surviving heterozygous pair in that donor.

  Measured against the cache: `R` equals `mL+mR` to max|diff| 2.3e-10 over all
  3,170,044 datapoints; `R==0` for 1,831,718/3,170,044 = **57.8%**; `Va==0.0`
  exactly in 100% of those and in 0 of the remaining 1,338,326, an exact match
  to the `no_cov` guard. Of the zero-informative pairs, 68.0% have zero total
  expression too, but 22.7% — 416,207 pairs, **13.1% of ALL pairs** — are
  expressed with no allelic information whatsoever, including 89,639 pairs at
  1,000+ total reads. Independently reproduced against the separate `YT`
  per-draw array (67.9% / 22.7%), which pins "expressed" at >=10 total reads and
  leaves a third, unlabeled band: ~9.3% have trace expression (0-10 reads).
  These are homozygous-but-expressed pairs: real expression, zero allelic
  information, by construction, not a depth artifact. The total channel
  correctly retains them, so the 13.1% figure is total-channel-only BY
  CONSTRUCTION and is easy to misread as loss.

  DESIGN NOTE: Salmon's Gibbs resampling was meant to be self-downweighting — if
  two haplotypes were indistinguishable the draws should disagree, `v` should
  blow up and the weight should collapse on its own. That cannot fire here,
  because deduplication plus the pairing rule mean there is no second copy to be
  uncertain between; every draw has `yL=yR=0` by construction, not by chance.

  An expression-based gene filter and the allelic channel's informativeness
  filter are NOT nested: 2,799 genes pass a standard expression filter but fail
  a >=40-informative-donors rule, because a gene can be well expressed in every
  donor and heterozygous in few. Expression governs admission to the TOTAL
  channel; heterozygosity governs the ALLELIC channel; passing one says nothing
  about passing the other.

- **The permutation null permutes donor records AND swaps haplotype labels
  (default since 2026-09-25, user decision).** `perm_scheme='records_signflip'`
  is the default everywhere (`map_cis`, `tensorqtl --perm_scheme`,
  `run_hapmixqtl_from_salmon.py --perm-scheme`): each donor's whitened
  phenotype value, weight and covariate row move together, genotypes stay, the
  denominator is recomputed per permutation (`_record_permutation_channel`),
  AND each permuted record's L/R labels are swapped with probability one half,
  negating its allelic log ratio (weight and covariate row unchanged; the total
  channel is never flipped). The user's reasoning: L/R is arbitrary phase
  order, so this is the only reasonable permutation for allelic ratios.
  Evidence: under `records` alone the through-origin allelic slope's
  permutation mean is the gene's net imbalance times the variant's phase
  lopsidedness, which offset 13 of 46 genes by up to 0.21 se (matching the
  algebra at r = 0.90); phase orientation is balanced (628 ALT-on-L vs 642
  ALT-on-R heterozygotes at the 46 leads, p = 0.72) and gene net imbalances
  are those of chance (z sd 0.98), so the swap is a symmetry of the null.
  Pooled calibration is unchanged (0.0690 vs 0.0692 at 0.05): the excess is
  spread, not centre. The swap changes the NULL only; the observed slope keeps
  its chance offset. Verified through the shipped code 2026-09-25
  (`scripts/allelic_signflip_null_check.py`,
  `brainvar_hapmix_deploy/allelic_signflip_null_check_20260925/`): at the 46
  fixed leads the permuted allelic slope is off-centre in 13 genes under
  `records` and 0 under `records_signflip` (0.12 expected by chance; max 0.042
  se), rates 0.0698 / 0.0206 / 0.0060 against 0.0692 / 0.0200 / 0.0057 (paired
  differences all span zero); on `map_cis` over 59 genes at 10,000 permutations
  no `pval_perm` or `pval_beta` call at 0.05 changes (15 / 15 and 14 / 14),
  leads and `pval_nominal` are identical, and the median shift in -log10
  `pval_perm` (0.0060) is below the seed-to-seed floor of `records` itself
  (0.0079). Signs are drawn right AFTER the permutation indices from
  the same generator, so `records` at the same seed sees identical indices and
  an identical total channel (pinned: a total-only run is identical under both
  schemes). `perm_scheme='records'` is retained unchanged: by relabeling it
  equals the genotype-permutation null of FastQTL and tensorQTL with per-donor
  weights, pinned to 1e-9 by `tests/test_hapmixqtl_perm_scheme.py`. mixQTL
  mode (`mixqtl_replication.py`) permutes the phenotype bundle its own
  published way and was NOT changed. `perm_scheme='residuals'` is the earlier
  Freedman-Lane scheme (leverage-standardized whitened residuals permuted at
  fixed weights, unswapped), retained but conservative where weights vary —
  most so at high expression. Cost 0.34 s vs 0.10 s per 1,000-permutation
  scan; the swap adds one element-wise product.

- **Gene-level Gibbs shape has bounded real-data evidence.** The three-library
  pilot found Gaussian competitive within 0.05 bits/draw for 98.51% of 4,500 ASE
  and 99.93% of total summaries, retaining two reproducible ASE candidates and
  no total candidates. ZNF529 and RNF175 are exceptions to interpret; RNF175
  restart-block occupancy is unreliable. Not association or calibration. See
  `/mnt/ssd/lalli/brainvar_hapmix_deploy/gibbs_shape_pilot_20260915/REPORT.md`.

- **The two-gene influence audit retained the moment/GPU baseline.** Across
  three genotype-selected common SNPs per gene and 92 donors (87 ASE), maximum
  one-block mean-plus-covariance shifts were 0.0736 working SE (ZNF529) and
  0.1396 (RNF175); leave-one-block maxima 0.00901 and 0.01028. Bounded
  sensitivity, not calibration. See
  `/mnt/ssd/lalli/brainvar_hapmix_deploy/gibbs_influence_audit_20260915/REPORT.md`.

- **Gibbs draw count is adequate to treat `v` as known.** 200 draws; lag-1
  autocorrelation median 0.081; median effective draws ~170; the relative sd of
  `v` as an estimate is median 0.109, with only 3.2% of datapoints above 0.25.
  Treating `v` as known rather than itself uncertain is a good approximation and
  errors-in-variables attenuation from doing so is small. Measured by
  `variance_layer_mapping_20260918/variance_layer_measurements.py`.

- **`v_ig` is a donor-by-gene interaction, so only a gene-by-sample weight
  matrix can hold it.** Within a gene across donors, log `v` has median sd 0.77
  (residual sd 0.557 at matched allele-resolved read count). Allele-resolved
  read count explains only R^2 = 0.32 of it — at matched read count `v` still
  spans 1.7-fold between donors of the same gene — so what remains once read
  COUNT is fixed is how INFORMATIVE those reads are, i.e. heterozygosity, not
  depth. The per-donor mean of read-count-adjusted log `v` has sd only 0.073
  across the 92 donors. So `v_ig` is neither a gene property (pooling across
  samples, as `catchSalmon` and sleuth do, cannot hold it) nor a donor property
  (a per-sample array-weight factor cannot either). The allelic channel carries
  about three times as much of its structure at the gene-sample level as the
  total channel: within-gene/between-gene median sd of log `v` is 0.41/2.12
  (ratio 0.19) for total, 0.72/1.29 (ratio 0.55) for allelic.

- **RTA overdispersion is at its floor at gene level, which independently
  confirms the shot-noise finding.** edgeR's read-to-transcript-ambiguity
  estimator on our own gene-level draws (3,000 genes): overdispersion quartiles
  1.00/1.00/1.01, 61.4% exactly at the floor of 1, 95th percentile 1.13, max
  35.6 — at gene level RTA has almost nothing to do, consistent with
  `catchSalmonGene` having arrived separately from transcript-level
  `catchSalmon`. Separately the across-draw variance of our log-total statistic
  is 0.98x (IQR 0.91-1.05) what an RTA-inflated Poisson predicts: an
  independent, differently-derived confirmation that Salmon's default Gamma draw
  carries shot noise. `variance_layer_mapping_20260918/rta_vs_c.py`.

- **A gene-by-sample quantification-uncertainty correction already exists in
  this project's RNA pipeline**, predating `catchSalmonGene`:
  `calc_expression_stats.R` (2025-05-02), run as a Nextflow process. Its
  `getGeneOverdispersion` adapts `catchSalmon` for personalized transcriptomes
  where the transcript set differs between samples: sums per-draw transcript
  counts to gene level within each sample, supports `merge_alleles`, keeps
  `OverDisp` as a gene-by-sample matrix moderated per sample (`colMedians`,
  `DFPrior=3`), then divides counts by it element-wise. hapmixQTL reaches the
  same information independently from the raw per-draw arrays, and the
  RTA-vs-Poisson finding says these are the same quantity on different scales,
  so the two are checkable against each other — not yet done.

- **RASQUAL's output field 15 is theta, NOT rho, and this was wrong here until
  2026-09-24.** It was originally read as "the extra-binomial fraction rho in
  Var = rho*p(1-p)" and that definition was written into the code, the output
  JSON and the comparison write-up. The first real run's own output caught it:
  a fraction cannot have a median of 76.5. `rasqual_src/src/usage.c:61`
  documents `--fix-theta` as "Fix overdispersion parameter (Theta=10000)",
  and field 15 is that Theta, on a PRECISION scale — 10000 is the fixed
  no-overdispersion value, so large is near-binomial and small is strongly
  overdispersed. `best_rasqual_row` in `scripts/compare_pipelines.py` now
  carries it as `overdispersion_theta`/`theta`/`theta_hat`, reported as itself
  and its distance from 10000, never converted to a variance-inflation factor
  (that algebraic map is not pinned against the beta-binomial density in
  `nbem.c`). **Two runs made before this fix are on disk with the same field
  under the old `rho`/`rho_hat` name and the wrong description; do not read
  their theta as a fraction.** `best_rasqual_row` also reads 1-indexed fields
  3-6 (chrom/pos/ref/alt), 11 (chi2), 12 (pi), 14 (phi) and 23 (convergence),
  plus field 2 to detect `SKIPPED`.

- **The mixed-BLAS crash blocks any limma path that fits a linear model.** The
  BLAS is Debian's openblas-pthread `libblas.so.3` while the LAPACK is a
  Homebrew openblas. Tested function by function, each in its own R process, so
  one crash could not hide another. CRASHES: `limma::lmFit` (segfaults even
  UNWEIGHTED), `limma::arrayWeights`, and therefore `vooma`, `voomaLmFit` and
  `voomWithQualityWeights`. RUNS: the whole closed-form empirical-Bayes
  moderation family — `squeezeVar`, `fitFDist`, `fitFDistRobustly`,
  `fitFDistUnequalDF1`. Also `libR.so` links `libblas.so.3` and
  `libopenblas.so.0` together, so `lm()`, `%*%` and `crossprod()` segfault,
  `tensorA`/`glmnet` are absent, and there is no sudo — which is why **no
  cross-language check of the mixQTL port exists and none is possible here.**
  The port's algebra is validated per variant against `numpy.linalg.lstsq` to
  1e-10 and every cutoff/cap/dof rule is pinned to the R source line it
  encodes (23 tests). Exact reproduction of the published code is NOT claimed.
  Installed: limma 3.64.3, edgeR 4.6.3 (a release behind upstream devel).
  DEVEL ONLY, not installed: `catchSalmonGene`, `binQLFit`, `PCList`,
  `sampleWeights`.

## What the draws buy, measured (2026-09-19, mixQTL mode as the baseline)

Report `brainvar_hapmix_deploy/mixqtl_replication_20260919/REPORT.md`. Before
this, mixQTL had never actually been run: `compare_pipelines.py` is
hapmixQTL-vs-RASQUAL.

- **THE DRAWS DO IMPROVE THE POINT ESTIMATE.** Holding response, donor set,
  variants and design fixed and varying ONLY the weights, `1/v` weighting cuts
  `var(beta_hat)` across 40 null permutations to **0.340** of unweighted (25/29
  genes, sign p=1.0e-4) — 1.71x in SE, 2.9x in variance. It beats mixQTL's
  published capped harmonic weights (0.499, 23/29) and uncapped harmonic
  (0.763, 21/29). Kish effective n falls 77 -> 45 while variance drops 2.9x,
  which is why this is not a concentration artifact.
- **It replicates across seeds.** An independent master seed moves every arm by
  <= 0.024 in the ratio and the per-gene Gibbs/OLS ratio correlates at r=0.997
  over 29 genes; the gaps carrying the conclusion (0.18, 0.26) are 7-11x that.
  The one pair that does not separate is capped Gibbs 0.780 vs capped harmonic
  0.826 (gap 0.046, ~2x the shift): read those as close, not ordered.
  `compare_seeds.py`.
- **mixQTL's fold cap costs two thirds of the gain** (0.340 -> 0.780). It is a
  workaround for a known-variance SE, which is why hapmixQTL is NOT given one.
- **`count_noise`'s q is inert for weighting**: q-on vs q-off efficiency 0.3402
  vs 0.3410, a 0.2% difference. CAUTION on the sign test there: 21/29
  (p=0.024) at seed 42 and 19/29 (p=0.14) at the other. A sign test responds to
  DIRECTION regardless of magnitude, so a p crossing 0.05 between seeds while
  the effect stays at 0.2% means inert, not real. Second, independent
  confirmation of the 2026-09-15 double-count finding.
- **Gene-level calibration.** The mixQTL port through the CORRECTED permutation
  path rejects at 5% in 0.0483 of 290 gene-draw pairs (se 0.0126, median p
  0.4925), indistinguishable from uniform and replicating across seeds (0.0448,
  median 0.4950). The published path yields no number at all here.
- Two defects in the distributed reference, both reproduced under
  `strict_reference_cap=True` and documented in the module docstring:
  `matrix_ls_asc_permutation` zeroes cutoff-failing weights before taking the
  min for the fold cap, so ANY cutoff failure zeroes every weight and all
  permuted betas are 0/0 (the non-permutation path subsets first and escapes);
  and `floor(n/10)` makes the cap 0 for 3-9 passing samples (past the
  `sample_size > 2` guard) and 1 for 10-19, where the channel becomes plain OLS.
- **End-to-end on the 29 genes is the weak half, and how weak depends on the
  cutoffs.** Under the PUBLISHED cutoffs: lead agreement among the 6 called
  genes 0/6, median lead LD r^2 0.160, Spearman of the per-gene statistic 0.183
  (p=0.34), beta Pearson r 0.562. Under permissive R-signature cutoffs
  (`20/5/100/5000`), kept as a sensitivity arm: 1/6, r^2 0.744, Spearman 0.517
  (p=0.004), r 0.697. The collapse has an identified cause: mixQTL's upper cap
  `y <= 1000`, stated in its Methods as an alignment-artifact guard, removes
  1,656 of 2,193 informative donor-gene pairs when applied to Salmon posterior
  means (the lower `y >= 50` removes only 38), leaving 18 of 29 genes
  total-counts-only and 7 with no allelic donors at all. At those cutoffs the
  arms are largely not measuring the same thing, so the permissive column is
  the better comparison of the two METHODS and the published column the better
  comparison against mixQTL AS RUN. A signal-bearing gene set is needed to
  sharpen either. Scope: these are high-coverage genes, 17.8% uninformative
  donor-gene pairs against 57.8% transcriptome-wide, so the 3x is not shown to
  transfer to low-count genes where the Gibbs and Poisson weights converge.
- The weighting ablation and the SE calibration are BYTE-IDENTICAL under the
  two cutoff settings, verified by re-running and diffing 2026-09-20. They run
  on hapmixQTL's informative-donor set (40-92 donors/gene), which mixQTL's count
  cutoffs never reach, and the only cutoff parameter they touch is `weight_cap`
  through `cap = min(weight_cap, floor(n/10))`; at <= 92 donors
  `floor(n/10) <= 9`, strictly below both candidate values, so realized caps are
  4-9 and `weight_cap` can never bind below 100 donors.

## hapmixQTL can apply mixQTL's count cutoffs (2026-09-20)

`count_cutoff_masks(yL, yR, yT, asc_cutoff, asc_cap, trc_cutoff)` builds
per-channel boolean admission masks from posterior-mean counts; `map_nominal`,
`map_cis` and `map_susie` take them as `keep_a_df`/`keep_t_df`, and the runner
exposes `--asc-cutoff --asc-cap --trc-cutoff` plus `--mixqtl-cutoffs`. The point
is a MATCHED-DONOR comparison between the two modes: the non-weighting ladder
found the donor set — not the response, not the weights — to be the dominant
non-weighting difference between the estimators.

- **Default is off and must stay off.** All four parameters default to None and
  an all-True mask reproduces the unmasked run bit for bit
  (`TestCountCutoffsEndToEnd`). The published cutoffs discard 1,656 of 2,193
  informative donor-gene pairs on the calibration genes, so this is a comparison
  instrument, not a production setting.
- **The mask is applied by zeroing the WORKING inferential variance** in
  `_prepare_channels`, putting an excluded donor in exactly the state a
  zero-coverage donor is already in. Every informative-set test downstream is
  `v > eps`, so one assignment keeps everything consistent with the weights.
  `Va_df`/`Vt_df` as the caller passed them are untouched.
- **`trc_cutoff` reads `yT`, never `yL + yR`, and this is a real trap.**
  `yL + yR` is the paired-transcript subtotal, exactly 0 for a
  homozygous-but-expressed donor. Thresholding `yL + yR` at 100 excludes 502
  donor-gene pairs that `yT >= 100` admits, 18.8% of the cohort, all of them
  good total-channel data. A test pins it.
- Measured retention on the 29 calibration genes (2,668 pairs, 2,193
  informative): the published allelic band `[50, 1000]` on both haplotypes keeps
  499; `trc_cutoff = 100` on `yT` excludes nobody, so the total-channel mask is
  inert on this high-coverage set.
- Not wired: `weight_cap`. hapmixQTL is not given mixQTL's fold cap, because
  capping costs two thirds of the efficiency the Gibbs weights buy.

## Default mode holds on non-circular ground truth (2026-09-23)

`tests/ase_external_benchmark.py` against the RASQUAL/TReCASE generative model,
which hapmixQTL does not assume. 500 replicates, N=200, mu=200, NB dispersion
0.2, BB overdispersion 0.01, allele-specific fraction 0.25. Report:
`brainvar_hapmix_deploy/external_benchmark_fitted_defaults_20260923/`.

- **Calibrated at nominal 0.05 and at parity with the generating model's own
  joint likelihood.** Type-I 0.0640 (lambda_GC 1.24), 1.4 Monte Carlo standard
  errors above nominal (se 0.0097). Matched power — each arm held to its own
  empirical 95th percentile — is 0.248/0.744/1.000 at allelic fold change
  1.05/1.10/1.20 against TReCASE's 0.274/0.760/0.998: differences
  -0.026/-0.016/+0.002 against paired se 0.028/0.027/0.002, so |z| <= 1.0
  throughout. TReCASE is the joint likelihood OF THE GENERATING MODEL, so this
  is parity with a ceiling, not a peer. Both dominate the single-channel arms
  (TReC-only 0.104/0.204/0.486; ASE-only 0.160/0.558/0.994, the latter
  genuinely anticonservative at type-I 0.158).
- **The bound is in the tail.** At nominal 0.01 default mode reads 0.0200 — 2.0x
  nominal, 2.2 Monte Carlo se above it (se 0.0045) — against TReCASE's 0.0120.
  Three runs put nominal-0.05 type-I near 1.3x (0.064 at 500 reps, 0.073 and
  0.067 at 150), so this is a pattern, not noise. NOTHING below nominal 0.01
  was tested and transcriptome-scale thresholds are far below it. It runs
  OPPOSITE to the Beta approximation's known tail conservatism, which concerns
  the gene-level permutation p rather than the nominal p, so the two must not be
  netted against each other.
- **It answers the circularity objection at this design point, and only there.**
  Fitting the residual scale from the residuals it scales did NOT produce
  anticonservatism at this N and depth. The objection is structural, so one
  design point is not a general refutation.
- **The harness fabricates the total channel's inferential variance, and the
  consequence for default mode was measured rather than argued.**
  `hapmix_pval` emulates draws as `yL ~ Binomial(n, frac)` with `yR = n - yL`,
  so `yL+yR` is EXACTLY constant across draws and the total channel's
  across-draw variance is zero by construction; `compute_summaries_from_gibbs`
  is called without `yT`. Only the Poisson term `1/(m_T + 2*kappa)` survives,
  computed on the ALLELE-SPECIFIC total (median 48 reads) while the phenotype
  comes from true totals (median 195.5): measured median `Vt` 0.0204082 against
  median `1/(m_T+2*kappa)` 0.0204082, identical to every printed digit. The
  delta-method variance of the harness's own phenotype, `T/(T+lib)^2`, has
  median 0.00507, so the harness **overstates the total channel's inferential
  variance by 4.0x**. Substituting it over 150 replicates moves default mode
  0.073 -> 0.067 in type-I and 0.733 -> 0.760 in matched power at 1.10, both
  inside the 150-rep Monte Carlo floor (~0.018, ~0.038), because a fitted
  `sigma^2` absorbs a wrong absolute scale on `v` by construction. TReCASE is
  unchanged to three decimals, the required internal control since it never
  sees `v`. SCOPE: this demonstrates robustness to a UNIFORM scale error only.
  It does NOT test a SHAPE error — a few samples mis-weighted against the rest —
  which is what the open total-channel zero-count defect produces and which no
  global scale can repair. Do not generalise to "robust to errors in `v`".
- **The harness reseeds per replicate**, data from `RandomState(seed0+r)` and
  each arm its own `RandomState(seed0+900000+r)`, so arm count and order cannot
  perturb any result. That is what makes reporting a subset of arms exact with
  nothing re-run. Checked before being relied on.
- **`lambda_GC` from this harness is CENSORED at 3019.92**, because `calib()`
  clips p to 1e-300 and `chi2.isf(1e-300,1)/chi2.ppf(0.5,1) = 3019.92`. Default
  mode is nowhere near it, but any lambda from this harness near that value is a
  floor, not an estimate. A figure of "3020" reported from it twice, on
  different data, was one ceiling reached twice rather than two agreeing
  measurements — see `docs/ase_validation.md` sec 7d, whose real-data finding is
  withdrawn for the separate reason below.

## Claims withdrawn — do not re-assert

**2026-09-13** (an eight-angle review retired these; they may survive in older text):

- The additive, multiplicative and nested variance forms were **never** compared:
  no multiplicative arm existed in `tests/`.
- `docs/ase_validation.md` sec 7b's weight-cap and nested conclusions are void;
  both arms were arithmetically incapable of differing from their comparators.
- The additive `tau` is **not** "the one place hapmixQTL departs from mixQTL".

Corrected facts that replaced them (these are true; do not negate them):
mixQTL's total channel already carries a flat additive variance, so `v_t + tau_t`
decomposed the parent rather than departing from it, and the multiplicative
scale is the allele-specific channel only. mixQTL's scan refits its dispersion
at EVERY variant. mixQTL's allelic regression is also through the origin
(`y ~ -1 + x`).

**2026-09-16:** the stated reasons for `count_noise=True` — "Gibbs across-draw
variance is read-assignment uncertainty only" (false: Salmon's default Gamma
draw carries shot noise, `CollapsedGibbsSampler.cpp:122`) and "a sample with
unambiguous reads has unanimous draws and would be discarded" (impossible under
default flags: identical draws occur only with zero reads). The flag stays on
for the missing-floor reason above. Also withdrawn: that `count_noise=False`
drops zero-read samples from the total channel (it does not; the guard is
ASE-only).

**2026-09-24** (all from the RASQUAL/calibration session; each was asserted in
that session and then measured away):

- "The 0.997 matched-variant slope means hapmixQTL and RASQUAL estimate the same
  quantity." It is `c*s` with c=1.28 and s=0.78.
- "Both Salmon-based arms understate their standard error by ~15%." That compared
  medians of two separately summarised distributions. Paired per variant it is
  ~6% for hapmixQTL and ~1% for mixQTL.
- "RASQUAL's standard error is less consistent gene to gene." RASQUAL reports no
  standard error; that spread may be Wald-conversion noise.
- "Heavy-tailed residuals explain the nominal-p miscalibration." Refuted by
  parametric bootstrap at the measured kurtosis.
- "What RASQUAL gains is the non-Gaussian likelihood, not jointness."
  Unsupported; the stacked arm imposed one common Gaussian scale, which is not
  RASQUAL's structure.
- "The empirical permutation p is calibrated by construction." Only for the null
  it is built from. Under the allelic channel's own sign-flip symmetry the
  ALLELIC-ONLY empirical p is larger than under records permutation (2,000
  draws, 2026-09-25: mean difference +0.043 [0.010, 0.080], sign test p=0.008).
  AMENDED 2026-09-25: the 30-draw count "7 genes against 11 (sign p=8.2e-4)" is
  withdrawn as a result. At 2,000 draws it is 10 against 13, a gap of 3 [-1, 7],
  and about a third of the shift (0.015 [0.002, 0.028] of 0.043) is the observed
  lead association carried into the sign-flip null; with leverage-corrected
  lead removal the counts are 13 against 12. This does NOT transfer to the
  shipped combined `pval_perm`, for which the two nulls agree.

**2026-09-23:** `docs/ase_validation.md` sec 7d's finding 1, "the defect is
confirmed on real data, at full severity", is withdrawn as INDEPENDENT
real-data corroboration. `tests/ase_gtex_real_data.py` has the same fabricated
total channel as the external benchmark: line 79 sets `YR = n_i - YL` so
`YL+YR` is exactly constant across draws, and line 124 calls
`compute_summaries_from_gibbs` WITHOUT `yT`. Its phenotype is real total
expression but the variance attached to it measures nothing. That is the
benchmark's mechanism reproduced on real allele counts, not corroborated by
them. What sec 7d DOES still establish: its ALLELIC channel carries genuine
GTEx overdispersion, depth and zero-inflation structure.

## Known and unfixed

- The lead refit is biased by selection: appending the window maximum to the
  design removes far more residual sum of squares than the one degree of freedom
  it is charged. Gene-level p-values are unaffected.
- The Beta approximation is conservative in the tail, costing power at
  transcriptome-scale thresholds. Note this is the OPPOSITE direction to the
  nominal-p inflation measured above; do not net them.
- No per-sample allele-specific read floor, where mixQTL used 15 reads.
- The total channel has no zero-count guard; `count_noise` stands in for a
  floor. A coverage-based floor for zero-count total samples, then no `q` for
  samples with reads, is the fix — not the flag.
- `tests/ase_external_benchmark.py` and `tests/ase_gtex_real_data.py` both
  fabricate the total channel's inferential variance (emulated draws conserve
  `yL+yR` exactly; `compute_summaries_from_gibbs` called without `yT`). Fix is
  to pass a `yT` built from the real/simulated totals. Both also keep an
  intercept on the allelic residualizer, where production has been
  through-origin since 2026-09-15.
- `calib()` in those harnesses reports a censored `lambda_GC` (saturates at
  3019.92). A censoring flag alongside it would stop the number being read as a
  magnitude, as it was until 2026-09-23.
- Three input-validation defects from the 2026-09-14 audit: a NaN at a
  zero-weight sample drives `pval_perm` to the `1/(nperm+1)` floor; variant-row
  identity between the genotype and phase frames is not checked beyond column
  order; the QR is taken on a rank-deficient design without a rank check.
- After the through-origin change, `_joint_gls`/`_pvals` (the robust
  second-pass path) still charge the ASE channel `1 + n_cov` columns, so the
  robust SE there is ~6% conservative. `map_cis`/`map_nominal` are unaffected.
- The log2 migration is pending; outputs are still natural logs.
- **`map_susie` CANNOT REACH DEFAULT MODE, and the package CLI's fine-mapping
  path runs the withdrawn known-variance configuration.** Established
  2026-09-23 by reading the code, and stronger than the "not re-verified"
  wording it replaces. `map_susie` (`tensorqtl/hapmixqtl.py:2248`) takes no
  `se_mode` parameter at all, and its call to `_prepare_channels` (line 2362)
  omits `fitted_scale`, which therefore defaults to `False` -- the
  known-variance SE. Its own defaults are `tau_mode='estimate'` and
  `variance_model='additive'`, both deprecated. `tensorqtl/tensorqtl.py:503`
  passes `HAPMIX_TAU_MODE = 'zero'` into it, so the shipped CLI runs
  `tau_mode='zero'` WITH the known-variance SE: exactly the pairing
  `_warn_tau_zero` describes as up to 107x nominal type-I at alpha=1e-3.
  Independently, `fine_mapping_provenance` (line 2492) labels any output
  carrying `tau_mode='zero'` **stale, credible sets and PIPs invalid** -- so
  the shipped path is declared invalid by the shipped checker. NOT FIXED here
  because the fix is a design decision, not an edit: `map_susie` needs an
  `se_mode`, and `fine_mapping_provenance` needs to be re-defined now that
  `'zero'` is the shipped mode rather than the old default it was written to
  flag. Changing it flips
  `TestMapSusie::test_map_susie_records_tau_mode_provenance`. Third instance
  of the defect class found in the 2026-09-23 CLI trim: an entry point left
  defaulting to the deprecated model.
- `run_second_pass` has not been re-verified under default mode as carefully as
  `map_cis`/`map_nominal`.
- **The nominal p is anticonservative; the MECHANISM is identified
  (2026-09-25), the generative SOURCE is not.** At 2,000 records permutations
  the shipped combined statistic rejects at 0.068 [0.061, 0.076] / 0.0175 /
  0.0028 at nominal 0.05 / 0.01 / 0.001 (the recorded 0.082 was a high 30-draw
  sample; the same 30 draws give 0.080). Within a gene, records with high Gibbs
  weight (small `v`) have larger whitened squared residuals `a^2/v`, ON
  AVERAGE, than the same gene's low-weight records, where `Var(eps) = sigma^2
  v` says the two averages are equal (every record's `a^2/v` has expectation
  `sigma^2`; no single residual is bounded). The fitted scale is the unweighted
  mean of `a^2/v` while the slope's variance is governed by the
  weight-weighted mean, so the reported se is short by that ratio. Residual
  variance grows as about `v^0.65`, not `v^1`, in both channels, and
  shuffling residuals against weights within gene removes 91% [73, 100] of the
  allelic channel's excess at 0.05 (85% / 88% at 0.01 / 0.001). Model records
  (N(0,1) residuals at the real weights) are exactly nominal, so the estimator
  and its reference are correct under the model. Over a per-channel
  scale-matched model the combined excess is 0.0172 / 0.0076 / 0.0019, of which
  allelic coupling is 62% / 60% / 65% and total-channel scale 27% / 19% / 8%.
  Cross-donor dependence CANNOT produce a records-permutation excess (the record
  set is fixed; only its assignment is random), so it does not explain these
  numbers; it remains untested for observed-data calibration. The DETECTION
  call is the empirical permutation p, which is unaffected by the scale error,
  so this bounds `pval_nominal` -- but see the CALM2 entry below for what the
  empirical p does not protect against. Section below; full record in
  `brainvar_hapmix_deploy/nominal_p_hypotheses_20260925/` (report
  `nominal_p_report.html`, reconciled budget `reconciliation.md`).
- **A single donor record can carry a gene-level call, and the empirical p does
  not protect against it** (2026-09-25). `map_cis` on observed data gives CALM2
  `pval_perm` 0.028 with donor 657_D1's allelic record and 0.684 without it
  (the lead moves); excluding any of 12 random other records leaves 0.008-0.036.
  CYCS 0.005 -> 0.049, FABP7 0.121 -> 0.415, and APC 0.359 -> 0.003 the other
  way. 657_D1's Gibbs variance (rank 2 of 84 in CALM2) matches Salmon's counting
  noise on its 1,282 haplotype-informative fragments, but its log ratio (-0.93)
  is about 12 of its own sd from the alignment-based phASER count (-0.16); all
  three of its exonic heterozygous SNVs are cohort singletons and it carries 6
  heterozygous indels. Say "disagrees with alignment-based allele counts beyond
  counting error"; the mechanism is NOT identified and "artefact" is not a
  safe label. No influence diagnostic is reported at the lead.
- For allelic-only runs (`keep_t_df` all False), `pval_nominal` is referred to
  `F(1, N - 2 - max(n_cov, n_cov_a))` = F(1, 73) here rather than
  `F(1, n_a - 1)` (45-88 on these genes): `dof = N - 2 - max(n_cov, n_cov_a)`
  at `tensorqtl/hapmixqtl.py:1694` and `:1991` is one reference for the
  combined statistic. Found by reading, 2026-09-25; no empirical p is affected.
- RASQUAL agreement has now been re-measured under default mode (section
  below). The reuse traps recorded when that run was designed still hold for
  any future one: `--reuse-rasqual` carries the OBSERVED arm only and never
  reads `null_rounds/`, the cached null rounds cannot be taken
  RASQUAL-half-only so they must be regenerated, and `null_calibration_29b` has
  no `rasqual_rows/`, so a run that wants matched-variant effects must pass
  `--rasqual-rows`. Design writeup:
  `/mnt/ssd/lalli/brainvar_hapmix_deploy/rasqual_comparison_design_20260923/rasqual_comparison.html`.

## hapmixQTL against RASQUAL and mixQTL, measured (2026-09-24)

`brainvar_hapmix_deploy/rasqual_default_mode_20260923/` (59 genes in three
coverage strata: the 29 calibration genes plus 15 MID and 15 LOW, median
allele-resolved reads 3,068 / 217 / 42; 30 null rounds on a 46-gene null subset
via `--null-gene-list`; seed 42). Summary page:
`brainvar_hapmix_deploy/calibration_summary_20260924/calibration_summary.html`.

- **Neither method out-detects the other.** McNemar exact on the discordant
  genes is p=0.77 at a 5% empirical false-positive rate and p=0.82 at 10%. At 10
  rounds the gap looked real; doubling the rounds dissolved it, so the earlier
  appearance was threshold noise. Detection counts are the weakest instrument
  here and should not be led with.
- **hapmixQTL's effects are about 0.78x RASQUAL's, not equal to them.** The
  matched-variant slope is 0.997 at hapmixQTL's own lead and 0.611 at RASQUAL's;
  since each arm's lead inflates its own effect by a selection factor c, these
  are c*s and s/c, giving a true scale ratio s = 0.78 and c = 1.28. The union
  slope, 0.767, lands on s as it must. The archived deprecated-config run
  decomposes the same way (s = 0.80).
- **Calibration of the nominal p at a fixed variant**, 46 genes, each arm's
  own nominal p. SUPERSEDED 2026-09-25 by 2,000 records permutations of the
  same stream: hapmixQTL combined 0.068 [0.061, 0.076] / 0.0175 / 0.0028 at
  0.05 / 0.01 / 0.001 (allelic 0.069 / 0.020 / 0.0057, total 0.060 / 0.0136 /
  0.0014); mixQTL port 0.056 [0.053, 0.060] / 0.0132 / 0.0020 under its own
  normal reference (permissive cutoffs; 0.051 / 0.0107 / 0.0012 under an F
  reference it does not use; published cutoffs 0.0553 / 0.0497). The 30-draw
  figures recorded here on 2026-09-24 -- hapmixQTL 0.082 (KS p=0.0066), mixQTL
  0.067 -- were high samples of that stream. RASQUAL 0.044 stands (converged
  rows only, KS p=0.51) but is on its own `-r` null, which permutes each feature
  SNP separately with no haplotype swap, at 30 draws, so it is NOT like-for-like;
  its 96 of 1,380 non-converged null rows reject at 0.083 and must be excluded.
  The external simulation benchmark's tail (0.064 / 0.020) has the same
  mechanism (section below).
- **Standard-error accuracy**, mean reported se over realized null sd, ~215,000
  (gene, variant) units, corrected for the 1.017 convexity inflation of an
  estimated denominator: hapmixQTL 0.939, mixQTL 0.987. **Flat across MAF** for
  every arm, so allele frequency is not a factor. Per channel hapmixQTL is
  0.956 allelic / 0.960 total and mixQTL 1.004 / 0.999, so the ~1.5-2% cost of
  combining two channels is shared by both methods and the rest is ours.
- **RASQUAL reports no standard error**, verified against its output spec: the
  only "error" among its 25 fields is the sequencing/mapping error rate delta.
  Any RASQUAL se quoted anywhere is `|beta|/sqrt(chi2)`, a Wald back-derivation
  that equals a standard error only if the Wald approximation holds -- which is
  the very thing its likelihood-ratio inference does not assume. Label it as
  derived, never as reported.

### Five mechanisms tested on 2026-09-24, two of them mis-measured

Recorded so they are not re-proposed in the same form. Each was measured, not
argued. Items 1 and 5 were found on 2026-09-25 to be mis-measured; the
corrections are inline and the amended items are the ones to cite.

1. **Not the Gibbs weights -- AMENDED 2026-09-25.** mixQTL never touches a
   draw and is anticonservative in the same direction (0.056 at 2,000
   permutations, not 0.067). But what separates hapmixQTL from mixQTL is
   UNCAPPED inverse-variance weighting in BOTH channels: on hapmixQTL's own
   records, mixQTL's allelic fold cap (`min(10, floor(n_a/10))` x the smallest
   weight, 4-9 fold on these genes) plus an unweighted total channel
   reproduces mixQTL's calibration to within 0.0003 and removes 98 / 95 / 93%
   of the combined excess. Whether the Gibbs `1/v` shape differs from harmonic
   count weights is UNRESOLVED (-0.0055 [-0.0138, +0.0032]); mixQTL's normal
   reference adds about +0.005 against it.
2. **Not the null construction.** For the COMBINED statistic, records against
   records-plus-sign-flip differ by nothing significant (0.082 vs 0.070 at 30
   draws, McNemar p=0.198). It matters only for the allelic channel in
   isolation, where sign flip gives 0.098 against 0.069 at 2,000 draws (the
   30-draw 0.117 was a high sample); about a third of that gap is the observed
   lead association carried into the sign-flip null.
3. **Not the channel combination.** Both channels are miscalibrated alone
   (0.069 allelic, 0.060 total at 2,000 draws; "indistinguishable" no longer
   holds at that resolution, and no paired test was rerun). Combining adds no
   excess of its own: the cross-channel interaction is 5-14% of the combined
   excess and does not clear its floor.
4. **Not weight-estimation noise.** Closed form for estimating `v` from `m`
   effective draws: `E[SE]/sd(beta) = sqrt((m-4)/(m-2))`, because the noisy
   weights inflate the true variance by `m/(m-4)` while `E[1/v_hat] = m/(m-2)`
   inflates the fitted scale and pushes the reported se back up. At m=170 that
   is 0.994 against the 4% measured; verified by simulation (0.9936 over 60,000
   replicates). A Satterthwaite dof correction is correspondingly inert
   (0.080 -> 0.080).
5. **Not heavy tails; the variance shape WAS mis-measured -- AMENDED
   2026-09-25.** The parametric bootstrap under the model's own assumptions is
   uniform (KS p=0.75) and heavy tails ALONE are small: decoupled-minus-model
   is 0.0018 / 0.0014 / 0.00055 in the allelic channel, 9-15% of its excess.
   The shape test was wrong in two ways. (i) The pooled -0.064 slope of the
   standardized squared residual on log v is the WITHIN-gene slope (-0.164)
   times an exact attenuation of 0.3905 = SSW/(SSW+SSB), the within-gene share
   of the variance of log v (reproduced to 1e-6); the fitted within-gene
   exponent has median about 0.65 in both channels, not 0.936. (ii) The
   coupling ratio's sign varies by gene (log R_g from -0.58 to +1.51), which no
   single global exponent can express, and the rejection rate is convex in
   that ratio. A fresh allelic-only arm at gamma 0.936 gives 0.054, not 0.050
   (the recorded "exactly 0.050" was measured on the combined statistic and
   was not rerun there).

**So the estimator and its reference are CORRECT under the model** (model
records give 0.0502 / 0.0101 / 0.0010), which is why all three attempted
repairs failed: baseline 0.080, Satterthwaite 0.080, stacked 0.128, stacked
with an HC3 sandwich 0.093.

### What the 2026-09-25 hypothesis round established

Scripts in `scripts/` (`null_permutation_instrument`,
`weight_residual_coupling`, `total_channel_null_calibration`,
`dominant_record_anatomy`, `allelic_overdispersion_floor`,
`imbalance_downweighting`, `comparator_null_2000`,
`coupling_transfer_to_observed`, `coupling_reach`, `combined_statistic_budget`,
`dominant_record_share_corrected`, `lead_signal_share_corrected`,
`alignment_discordance_coupling`); outputs in `brainvar_hapmix_deploy/*_20260925/`;
every claim adversarially re-run, the corrected figures are the ones here.

- **Mechanism.** Write `w = 1/v_a` and `z^2 = a^2/v_a`. `R_g =
  mean(w z^2)/(mean(w) mean(z^2))` is 1 under the model and is, to first
  order, the realized-over-reported variance of the permuted slope; its
  Spearman 0.955 with the realized ratio is algebra, not evidence. The
  evidence is the arm ladder on the identical stream (allelic, 0.05 / 0.01 /
  0.001): real 0.0692 / 0.0200 / 0.0057; model 0.0502 / 0.0101 / 0.0010;
  decoupled (z shuffled against w within gene) 0.0519 / 0.0116 / 0.0016;
  Gaussian errors at each record's own realized variance 0.0693 / 0.0219 /
  0.0058. Under a null that keeps the heavy-tailed z^2 marginal, 10 genes lie
  above their band against 1.15 expected; only CAMSAP2 is conservative beyond
  chance; the across-gene spread of log R_g is 1.75x the null.
- **Total channel** = pure per-gene scale, the leverage-corrected R_t; rescaled
  0.0505 / 0.0098 / 0.00085; unit weights 0.0496; `Var(e) ~ v_t^0.66`; the
  Gibbs variance is ~1/50 of between-donor variance in HIGH genes, so `1/v_t`
  is effectively a depth weight (corr(t, log w) median 0.993). A per-donor
  variance component exists (221_D1, RIN 3.1, mean z^2 3.75 vs model max
  1.93), not converted to a rate.
- **Single records.** Against a selection control that keeps the real heavy
  tails, dropping each gene's top record removes 2% [-49, 24] / 18% [-51, 43]
  / 58% [-25, 75] of the allelic excess -- no interval excludes zero (a
  Gaussian control had read 15 / 31 / 68%). CALM2 657_D1 alone is 52% of this
  gene set's 0.001 excess and 8% at 0.05; without CALM2 no single-record share
  is detectable. Transcriptome-wide a record holding >50% of `sum(w z^2)`
  occurs in 5.7% of genes against 0.57% under the model, mostly at low
  coverage and few informative donors.
- **Sources tested.** Observed lead association carried into the records:
  13-22% of the allelic 0.05 excess (leverage-corrected removal), nothing in
  the tail. Salmon-vs-alignment discordant records (|dz| > 3, 0.82% of
  records): 0.03 / 1.5 / 4.7 / 10% of the coupling at 0.05 / 0.01 / 0.001 /
  1e-4 net of a weight-matched control, so confident point-estimate errors are
  REFUTED as the generator (a diffuse Salmon-specific error is not excluded);
  they do collapse CALM2 (R 4.53 -> 0.67). A common additive floor reproduces
  the pooled rate but ranks no genes (|Spearman| <= 0.18) and a power law
  beats it in 38/46 genes; count-based beta-binomial rho does not predict R_g.
  Singleton exonic SNVs are phased against read-backed phase in at most 19.1%
  of cases (2.0% for common variants) but reach 137 of 9,328 discordant
  records: not distinguishable. Conditioning on additional cis variants: NOT
  TESTED, by user decision (multi-SNP hits are unreliable here). The remaining
  ~55-65% of the allelic 0.05 excess is smooth positive coupling of unknown
  source that alignment-based counts also show (phASER counts at their own
  counting variance: 0.061 / 0.014 / 0.0020).
- **Transfer and reach.** Split-half correlation of the rank coupling across
  genes 0.53 (model 0.01); a sampling null matches the permutation null at
  0.05 and 0.01, so this applies to `pval_nominal` on observed data; 79 / 61 /
  34% of the excess recurs in held-out halves. Transcriptome-wide (20,281
  genes, n_a >= 20, synthetic Hardy-Weinberg variants): 0.0653 / 0.0177 /
  0.00334 / 0.00085 at 0.05 / 0.01 / 0.001 / 1e-4, coupling 88 / 79 / 66 /
  51% of it by direct decoupling; the 30-100-read stratum is worst at 0.05
  (0.075); the 46-gene set is 67% >= 700 reads against 22% transcriptome-wide.
- **NOT established, do not write:** any generative cause; any "Salmon
  artefact" label; that the components are additive (the interaction is
  14-18% at 0.01-0.001); the top-3-records share at 0.001 (seed-dependent
  lower bound); the transcriptome band counts (Gaussian null); anything below
  0.001 on these genes or 1e-4 transcriptome-wide; a lead-removed null as a
  fix for `pval_perm`.
- **Lead removal needs different leverage corrections per channel.** Allelic:
  `e/sqrt(1-h)` is nominal, uncorrected is conservative (0.042). Total:
  `e/sqrt(1-h)` is anticonservative (0.0527) because the covariate row travels
  with the record; the derived `e/sqrt(1 - h_g/(1-h_Z))` is nominal (0.0503).

**Joint modelling of the two channels is WORSE, not better.** Stacking them
under one residual scale gives 0.128 with the median p falling to 0.392. The
channels' fitted scales differ by a median factor of five (allelic 2.13 against
total 13.62, within two-fold in only 5 of 46 genes), so a single scale is a
misspecification the inverse-variance meta-analysis does not make. Do not
conclude from this that RASQUAL's advantage is its non-Gaussian likelihood
rather than its jointness: RASQUAL's channels sit on different likelihood
families, so the stacked arm was not a proxy for its structure, and that claim
is untested.

**Cross-donor dependence, split three ways (2026-09-25; replaces "the leading
untested candidate"):** (a) a records permutation cannot create excess from
errors correlated across donors, because the permuted record set is fixed and
only its assignment to genotypes is random -- so it does NOT explain the
records-null excess measured here, and the external benchmark (i.i.d. donors
by construction) shows the same tail; (b) cross-donor CORRELATION -- relatedness,
population structure, batch, anything the 17 covariates do not absorb --
remains untested as a source of miscalibration on OBSERVED data; (c) a
per-donor VARIANCE component does exist in the total channel (221_D1, RIN 3.1,
mean whitened squared residual 3.75 against a model maximum of 1.93; 5 donors
above the model 95th percentile), not converted to a rate.

## Self-tests

```bash
# the hapmixQTL/mixQTL surface: 175 tests, all passing as of 2026-09-23
pytest tests/test_hapmixqtl.py tests/test_hapmixqtl_calibration.py \
       tests/test_hapmixqtl_perm_scheme.py tests/test_fitted_variance_quarantine.py \
       tests/fitted_variance/ tests/test_cli.py tests/test_mixqtl_replication.py -q

python3 scripts/run_hapmixqtl_from_salmon.py --selftest
RASQUAL_BIN=rasqual_src/src/rasqual python3 scripts/compare_pipelines.py --selftest
```

**Four test files carry 46 pre-existing failures from upstream fork drift and
are not ours:** `tests/test_post.py` and `tests/test_trans.py` (22 between them)
and `tests/test_genotypeio.py` and `tests/test_integration.py` (24). **None of
the four contains a single reference to hapmixqtl** — verified, not assumed —
and their failures are BED sort order, a missing `chr` column and a missing
`pval_beta` column. A bare `pytest tests/` therefore reads 46 failed / 223
passed and that is the expected state; do not read it as a hapmixQTL
regression. Run the command above to see the surface this fork owns.

`tests/test_fitted_variance_quarantine.py` pins that a default-mode run never
imports the deprecated module, including a full `map_cis` checked in a
subprocess.

# Pipeline rules: values, units, gene filter, permutation

User decisions of 2026-09-25. They apply to both modes (mixQTL mode and
default mode) and stand until the user changes them. This page states each
rule, where it is implemented, what was built to satisfy it, what is not yet
switched over, and which recorded results predate it.

## The rules

1. **Every value comes from Salmon point estimates** (`quant.sf` NumReads):
   the allelic ratio, total expression, CPM, count cutoffs, expression PCs
   and mixQTL's inputs. The 200 Gibbs draws are used only for each
   observation's measurement variance, computed through the identical
   transform as the value. Before this, the phenotype was the mean over Gibbs
   draws of the log (`compute_summaries_from_gibbs`, kept so dated scripts
   reproduce).

2. **The unit is log2(CPM + 1) at every step.** CPM is count divided by the
   effective library size, times 1e6. The effective library size is computed
   by edgeR itself: `lib.size` times the TMM factor. TMM (trimmed mean of
   M-values) is edgeR's normalization factor; it scales each library so that
   most genes' log ratios to a reference library centre on zero. The edgeR
   sequence, in `scripts/edger_library_normalization.R`, is: a `DGEList` of
   every gene's point-estimate total; `filterByExpr` with no design (CPM at
   least 10 / median library size in millions in at least 10 + 0.7 (n - 10)
   samples, and a total count of at least 15); intersection with the gene
   restriction; subsetting with `keep.lib.sizes=FALSE`, so `lib.size` is
   recomputed from the kept genes; `calcNormFactors(method='TMM')`.
   Exception (user decision): the allelic ratio is
   log2((L + 0.5) / (R + 0.5)) of point-estimate haplotype counts, because
   library size cancels within a sample. mixQTL mode keeps its published
   natural-log response log(count / 2 / L) with the same L, so its raw betas
   are hapmixQTL's times ln 2.

3. **The expression-PC gene filter equals the eQTL gene filter actually
   used.** Calibration phase (user decision, explicitly temporary while the
   statistical model is being calibrated; deployment may use a very different
   filter): `filterByExpr` AND protein-coding (at least one curated RefSeq
   `NM_` transcript) AND autosomal (chr1-chr22), which is 12,955 genes.

4. **Under permutation, covariates move with the RNA record, except the
   genotype PCs, which move only with the genotypes.** Age, age squared, sex,
   RIN and the expression PCs travel with the donor's RNA record; the
   genotype PCs stay in genotype order.

5. **mixQTL mode never touches the Gibbs draws**, not even through their
   mean. It is the no-draws comparator only on that condition.

6. **A zero-read total donor keeps a counting-variance floor**: the
   delta-method Poisson variance of log2(CPM + 1), that is the first-order
   Taylor approximation to the variance of the transformed count, evaluated
   at count + 0.5. It is added to every donor under `count_noise=True`, as
   the old `q` was, so the open item "drop `q` for donors with reads"
   (CLAUDE.md, Known and unfixed) is still open. In the 100-gene calibration
   set 32 of 9,200 donor-gene pairs have zero total reads.

## Where each rule is implemented

| Rule | Code | Pinned by |
|---|---|---|
| 1, 2, 6 | `tensorqtl/hapmixqtl.py` `summaries_from_point_estimates` | `tests/test_hapmixqtl_point_estimates.py` |
| 2, 3 | `scripts/edger_library_normalization.R`; `run_hapmixqtl_from_salmon.py` `edger_normalize`, `read_edger_dir` | runner `--selftest` |
| 3 | runner `check_covariate_provenance`: refuses covariates whose `covariate_build.json` names another gene set or other library sizes; `--covariates-unverified` overrides | runner `--selftest` |
| 4, default mode | `map_cis`/`map_nominal` `genotype_covariates_df`; `_combine_covariates` puts those columns last; `WeightedResidualizer.n_fixed_cov` tells `_record_permutation_channel` how many trailing columns stay fixed | `test_genotype_tied_covariates_permute_with_the_genotypes` (relabeling identity: equals permuting genotype columns and genotype-PC rows together by the inverse permutation) |
| 4, mixQTL mode | `mixqtl_scan`/`mixqtl_permutation_scan` `genotype_covariates`: the two-step offset is refitted on each permuted dataset (divergence 12 in the module docstring) | `test_genotype_covariates_stay_with_the_genotypes_under_permutation` |
| 5 | `tensorqtl/mixqtl_replication.py` `inputs_from_point_estimates` (refuses a draws axis); `summaries_from_gibbs_posterior_mean` deprecated with a warning | `test_point_estimate_inputs_refuse_a_draws_axis` |

In the default through-origin allelic design (`ase_covariates_df=None`) the
allelic channel has no covariates, so rule 4 acts on the total channel only.
It reaches the allelic channel only under `ase_covariates_df=SAME_COVARIATES`.

Drivers on the rules: `scripts/run_hapmixqtl_from_salmon.py` (default mode;
`--covariates` required, genotype-tied columns from `genotype_covariates.txt`
beside it) and `scripts/compare_mixqtl_replication.py`
(`load_point_estimate_inputs`; output `$MIXQTL_OUT`, default
`brainvar_hapmix_deploy/mixqtl_replication_point_estimates_20260925/`).

## Not yet switched over

- `scripts/compare_pipelines.py`, the RASQUAL comparison, still uses the
  Gibbs-mean natural-log phenotype and moves the whole covariate row with the
  RNA record in its null rounds.
- The dated analysis scripts keep the pre-correction path on purpose. The
  driver's `load_inputs` is the pre-correction loader and stays unchanged
  because 46 of them import it.
- `scripts/analyze_mixqtl_comparison.py` reads the 2026-09-19 folder, whose
  hapmixQTL arm is a deprecated-model ablation record. Do not point it at the
  corrected folder.
- No corrected stored null and no before/after calibration comparison have
  been run.

## Open decision: 1,208 filtered genes have no Gibbs draws

1,208 of the 12,955 calibration genes are absent from the Gibbs cache, so the
runner reports and skips them. The tested set is 11,747 while the expression
PCs use all 12,955. These genes are expressed (median 401 total reads in 8
donors checked) but have no haplotype-paired transcript in any donor; in
donor 100_R1, ACO2 has only `_L` rows. `load_counts` collects total-channel
draws only for genes with a pair in some donor. The options are to extend
`load_counts` to collect `YT` draws for pair-less genes, which needs a new
cache build of about 45 minutes and makes them testable in the total channel,
or to accept 11,747 genes and record why.

## Open decision: Salmon point estimates put one haplotype at exactly zero

Found 2026-09-25 by the first corrected run of the mixQTL driver. Salmon
1.10.3 ran with default options (`cmd_info.json`: no `--useEM`), so its point
estimate is its default variational-EM optimum, while the Gibbs sampler
draws from the posterior. Where the two haplotype copies of a transcript are
hard to tell apart, the point estimate can assign every read to one copy
while the draws keep splitting them. Example: CAMSAP2 in donor 418_D1 has NumReads 0 on
all seven `_L` transcripts, while the Gibbs draws average 944 reads on `_L`
and 7,624 on `_R`. Row-by-row comparison against `quant.sf` and the draws
rules out a reader error, and the paired subtotal agrees between point
estimate and draw mean (Pearson r 0.9986 over informative pairs).

On the 29 calibration genes:

| Measurement | Value |
|---|---|
| Informative donor-gene pairs (Gibbs variance > 0) | 2,188 |
| Of those, one haplotype below 0.5 reads in the point estimate | 79 (3.6%), in 20 of 29 genes, up to 14 in one gene |
| Gibbs mean of that same haplotype | median 95 reads, max 944 |
| abs(allelic ratio), point estimate | median 0.23, 99th percentile 10.98, max 14.27 log2 units |
| abs(allelic ratio), mean over draws of the log | median 0.21, 99th percentile 3.36, max 5.26 log2 units |
| Pearson r, point-estimate vs draw-mean ratio | 0.82 |
| Within-gene weight percentile of the zero-haplotype pairs in default mode | median 0.02 |
| Their share of the gene's sum of weight x ratio^2 | median 0.118, max 0.591 |

Transcriptome-wide it is much larger. Over all 34,457 cache genes, among the
1,294,098 donor-gene pairs with haplotype-informative reads in the point
estimate, the share with exactly one haplotype below 0.5 reads is below,
beside the same count on the Gibbs draw means. The draw means are NOT a
control for biology: where the copies share sequence the sampler spreads
ambiguous reads over both by construction, so a draw mean above zero does
not show that both alleles are expressed.

| Haplotype-informative reads | Point estimate | Gibbs draw mean |
|---|---|---|
| all | 36.10% | 1.66% |
| 1 to 9 | 92.69% | 7.00% |
| 10 to 99 | 46.06% | 0.01% |
| 100 to 999 | 8.36% | 0.00% |
| 1,000 or more | 2.26% | 0.00% |

66,826 of these pairs have at least 50 reads on the other haplotype, so their
ratio exceeds about 7 log2 units; 27,016 of 34,457 genes have at least one.

**Imprinting or estimator?** Tested against phASER's alignment-based counts
at heterozygous SNPs, which cannot be spread between copies
(`scripts/zero_haplotype_phaser_check.py`,
`brainvar_hapmix_deploy/zero_haplotype_phaser_check_20260925/`; pairs with at
least 20 gw-phased phASER reads). The statistic is the phASER minor-allele
fraction, near 0 for monoallelic expression and near 0.5 for balanced.

| Haplotype-informative reads | Zero-haplotype pairs outside the 35 genes below: median minor fraction, share below 0.05 | Control, reads on both copies |
|---|---|---|
| 10 to 99 | 0.444, 1.1% of 23,700 | 0.440, 0.2% |
| 100 to 999 | 0.438, 5.6% of 10,699 | 0.453, 0.1% |
| 1,000 or more | 0.408, 21.6% of 2,682 | 0.471, 0.4% |

Most zero-haplotype pairs are biallelic by phASER, as balanced as the
control, so for them the zero is an estimator effect. A minority is real
monoallelic expression, and it grows with depth: at 1,000 or more reads
about a fifth, against 0.4% in the control. Imprinting silences the same
parental copy in nearly every heterozygous donor, but only 35 of 11,500
evaluable genes are one-copy in at least 80% of their donors with at least
100 reads, holding 2.5% of such zero pairs. Those 35 include known imprinted
genes (NAP1L5, FAM50B) beside readthrough transcripts and pseudogene
families (COMMD3-BMI1, INO80B-WBP1, TBC1D3D, NPIPB12, PKD1P1), and below
1,000 reads phASER shows their zero pairs biallelic too. Only 13,768 of the
41,579 zero pairs at 100 or more reads have enough phASER coverage to test.

So on the 29 calibration genes the Gibbs weights mostly protect the default-mode slope, because these
pairs get the lowest weights, but their extreme values still inflate the
fitted residual scale. mixQTL mode's published allelic cutoff (at least 50
reads on each haplotype) excludes them. The unweighted and capped arms of the
weighting ablation are dominated by them: in
`mixqtl_replication_point_estimates_20260925/summary.json` the Gibbs-weighted
slope variance is 0.066 of unweighted, against 0.340 in the 2026-09-19 run on
draw means, and the capped harmonic arm's known-variance calibration reads
0.000: a zero haplotype gives a harmonic weight of about 1e-12, and the cap
limits every weight to a multiple of the smallest.
Those numbers describe the boundary zeros, not the weightings, and should
not be cited as a before/after of the weighting result.

**Which allelic value agrees with phASER** (`scripts/allelic_value_vs_phaser.py`,
`brainvar_hapmix_deploy/allelic_value_vs_phaser_20260925/summary.json`).
Pairs with at least 10 Salmon haplotype-informative reads and at least 20
gw-phased phASER reads. "Inconsistent" means the value differs from phASER's
log2 ratio by more than 3 of phASER's own counting sd. Candidates: the point
estimate; the draw mean of log2((yL+1/2)/(yR+1/2)), the pre-2026-09-25 value;
and log2 of the draw-mean counts. Intervals resample genes (SEED 42, 2,000
draws).

| Candidate | Zero-haplotype pairs (37,546): median abs difference, share inconsistent | Pairs with reads on both copies (595,009) |
|---|---|---|
| Point estimate | 6.45 log2, 97.0% [96.5, 97.4] | 0.30, 17.8% |
| Draw mean of the log ratio | 0.88, 37.0% [35.9, 38.1] | 0.24, 10.0% |
| Log of the draw-mean counts | 0.77, 31.6% [30.5, 32.6] | 0.23, 9.6% |

Log of the draw-mean counts beats the draw mean of the log by 5.4 points
[5.2, 5.7] on zero pairs. The point estimate overstates imbalance on
ordinary pairs too: median excess magnitude over phASER +0.12 log2 against
+0.04 for either draw value. No candidate is clean on deep zero pairs: at
1,000 or more reads the log of the draw-mean counts is still inconsistent in
63.5% and overstates magnitude by a median 1.77 log2, against 89.2% and 10.99
for the point estimate. Dropping zero pairs would discard 3,055 of them, 8.1%,
that phASER calls clearly imbalanced, beyond 2-fold and 3 sd; on those the
point estimate reads a median 9.29 log2 against phASER's 3.27, and the draw
values 3.36 and 3.11. Orientation was checked: where both sources call a
clear imbalance their signs agree in 95.7% of 3,538 pairs.

Limits: phASER is not truth. It aligns to the reference genome, so it
carries reference-mapping bias; it shares its SNP-covering reads with
Salmon; and a monoallelic ratio is bounded by the pseudocount in both
sources. Only 37,546 of about 467,000 zero-haplotype pairs have enough
phASER coverage to score, few of them below 10 Salmon reads. Agreement with
phASER is not calibration of the eQTL test, and the draw mean of the log
ratio is the allelic value on which the pre-correction calibration was
measured.

**Cost of dropping them from the allelic channel** (the total channel keeps
them; `scripts/drop_zero_haplotype_cost.py`,
`brainvar_hapmix_deploy/drop_zero_haplotype_cost_20260925/`), on the 11,747
calibration genes with draws: 114,999 of 786,919 informative pairs, 14.6%,
all of them exact zeros; 89.5% of pairs at 1 to 9 haplotype reads, 42.4% at 10
to 99, 7.8% at 100 to 999, 2.2% at 1,000 or more. They carry 0.19% of the
allelic channel's total weight (median per gene 0.18%) under the shipped
weight 1 / (Gibbs variance + counting term at the point estimate). Genes
with at least 20 informative donors fall from 11,237 to 10,738.

Options for the user: keep point estimates and let the weights handle it;
treat a haplotype at zero in the point estimate while the draws disagree as
uninformative for the allelic channel; re-quantify with Salmon's `--useEM`
(plain EM instead of variational EM), which can also reach zero, so whether
it helps would have to be measured; or use a different point summary for
the allelic split only.

## Built inputs

All under `/mnt/ssd/lalli/brainvar_hapmix_deploy/`.

- `cache/gibbs_56b63c3b37ed5df8/point_estimates/` from
  `scripts/build_point_estimate_cache.py`: `pL.npy`, `pR.npy`, `pT.npy` for
  the 34,457 cache genes; `totals_all.tsv.gz` for 41,552 genes;
  `edger/edger_samples.tsv`, `edger/calibration_genes.txt` (12,955 genes);
  `summary.json` with the reader gates.
- `cov/log2cpm1_point_calibration_20260925/` from
  `scripts/build_covariates.py --point-estimates`: `covariates.tsv` (14
  RNA-tied columns: age_days, age_days_sq, rin, sex, expr_pc1-10; 3
  genotype-tied: geno_pc1-3), `genotype_covariates.txt`,
  `covariate_build.json`. Metadata
  `/mnt/ssd/lalli/nf_stage/draft_brainvar2_library_metadata_v1.4.tsv`
  (sha256 prefix `c70e3599`); VCF `prepped/rephased.vcf.gz`. Expression PCs
  are log2(CPM + 1) of point estimates on the calibration genes, each gene
  centred but not scaled, residualized on the metadata and genotype PCs, top
  10.
- The old `cov/covariates.tsv` is kept unchanged and is pre-correction. Its
  genotype PCs came from a different VCF: old PC1 correlates with new PC1 at
  r = 0.985, old PC3 with new PC2 at r = -0.973, and old PC2 has no
  counterpart.

| Measurement | Value |
|---|---|
| TMM factors | 0.915 to 1.124 |
| Median effective library size | 17,172,092 (raw median 22,480,610) |
| Point-estimate vs Gibbs posterior-mean totals, Pearson of log1p | 0.993 all cache genes; 0.99999 on the 29 calibration genes |
| Donor-gene pairs with haplotype reads in some Gibbs draw but none in the point estimate | 44,228 (none the other way) |
| Donor-gene pairs with no allelic information | 1,875,946 of 3,170,044 (59.2%) under point estimates, against 57.8% under Gibbs posterior means |

## Results that predate the rules

Every calibration number recorded on or before 2026-09-25 was measured on the
pre-correction pipeline: Gibbs-mean phenotype, natural-log raw counts, the
old covariates, the whole covariate row moved with the record under
permutation, and mixQTL fed posterior means. That covers the
2,000-permutation instrument, the mechanism decomposition, the comparator
rates, the transcriptome reach and the count-scale arms. The numbers are not
withdrawn; they describe that pipeline, not the corrected one.

The stored 200-draw null on 100 protein-coding genes
(`protein_coding_null_store_20260925/`, commit d3248f0) is pre-correction. 90
of its 100 genes pass the corrected calibration filter. It is the
before-baseline for a before/after comparison.

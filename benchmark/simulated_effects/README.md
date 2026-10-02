# Simulated-effects cis-eQTL benchmark (benchmark/simulated_effects)

> **Every arm on half-read split's inputs (user decision 2026-10-01).** The hapmixQTL arms and
> total-only tensorQTL use the half-read total; the arms are split (the shipped default), gibbs
> (the total weighted by its half-read Gibbs variance) and unit; `plus_one` is retired; mixQTL mode
> keeps its published natural-log response; RASQUAL and TReCASE were run on these datasets and
> are scored (`common.JOINT`; RASQUAL back since commit b5471d4, 2026-10-02), and the native-input
> arms run on the half-read total (commit 388ef6d). The deep set's run is
> `brainvar_hapmix_deploy/simulated_effects_half_read_20261001/`, its stored null
> `stored_null_half_read_20261001/`; the low-coverage set's run is
> `simulated_effects_lowcov_half_read_20261001/`, its stored null `stored_null_lowcov_half_read_20261002/`
> (`stored_null.py`). Each run root holds its page (`report/plasmode_report.html`) and its gene-level
> null (`gene_level_null/`); the one report over both sets is
> `simulated_effects_integrated_20261002/index.html` (`integrated_report.py`). The 2026-09-27 roots
> (`plasmode_meier_20260927`, `plasmode_lowcov_meier_20260927`) and the run of earlier on 2026-10-01
> (`release_closure_20261001/plasmode_after_split`) are records of the log2(CPM+1) total.

## Purpose

Datasets with known cis effects built from the BrainVar cohort's own Salmon quantification, and
the recovery of those effects by three hapmixQTL weightings on the half-read total (split, the shipped
default; gibbs; unit), mixQTL mode at two cutoff settings, total-only tensorQTL (tensorqtl.cis on the half-read
total, unweighted), TReCASE (asSeq) and, where `ROOT/native/` exists, two native-input arms on alignment counts from the same STAR BAMs (TReCASE, and split
weighting as the control; section Native-input arms). The question it answers: on data with the real cohort's depth, noise and
donor structure, how well does each arm rank non-null genes, discover them at a controlled
false-discovery rate (on its own permutation p where it has one, and on eigenMT's p for every arm),
estimate the injected slope, state its standard error, and place the lead variant on the causal one.
The hapmixQTL arms carry Meier's correction of the combined standard error (commit a1b2ef4). Its 2026-09-27 run on the
log2(CPM + 1) total was part of the evidence for the weighting decision (`docs/pipeline_rules.md`, "Decision, 2026-09-29:
half-read split is the default weighting configuration"); since 2026-10-01 it measures that configuration
(`docs/pipeline_rules.md`, "Decision, 2026-10-01").

This directory is the analysis-tier rewrite (2026-09-27) of the previous benchmark code, the twelve scripts that lived in `scripts/plasmode` until commit fc238df: the same design,
about 2,600 lines of pipeline code (common.py and scripts 01-07, run_trecase.R) plus a report script of
about 2,400 lines and the acceptance test, one check script, no per-dataset re-validation, no
command-line options; `select_stratum_genes.py` made the 30-100-read gene set once.
`99_acceptance.py` checks it against the committed run of each gene set (2026-09-26 and 2026-09-27); those runs
predate Meier's correction, the tensorqtl arm and the half-read total, so on the current code it does not pass: it is
a refactoring check that needs references made by the same statistics, and none exists yet for this configuration.

The benchmark is standalone (2026-10-01): it moved from `scripts/plasmode` to `benchmark/simulated_effects` (commit e68865d),
and `common.py` carries its own copies of the loader and helpers it used to import from `scripts/` (commit 938d28c;
its section "Inputs and helpers this benchmark carries itself" names each source). `05b_native_arms.py` runs a
byte-identical copy of `scripts/edger_library_normalization.R` kept here. The only repository code it imports is the
`tensorqtl` package.

## Data provenance

Every input is read, never written, from `/mnt/ssd/lalli/brainvar_hapmix_deploy` (`common.D`):

- `cache/gibbs_56b63c3b37ed5df8`: Salmon 1.10.3 point estimates and 200 Gibbs draws per transcript
  copy for 92 donors, personalized diploid transcriptome (g2gtools `_L`/`_R` copies), read through
  `common.load_point_estimate_inputs` with the cohort genotypes, phase, covariates (14 RNA-tied,
  3 genotype PCs; `common.COV`) and edgeR effective library sizes.
- `common.GENE_SET` (default `corrected_null_store_20260925`) names the gene set; `common.GENE_SETS`
  holds, per gene set, the directory under `D` (`genes.txt`, 100 genes; `regions.bed`;
  `gene_design.tsv` with median allele-resolved reads and admitted allelic donors; that set's stored
  200-permutation gibbs null run, `summary.json` and `draws/`), the output directory (`ROOT`), and the
  stored runs of that gene set the scripts read: `stored_null_half_read_20261001` (the 200-permutation null of the
  three hapmixQTL arms on this configuration, `scripts/half_read_stored_null.py`; 06's anchors, 01's check d and
  08; `stratum30_100` reads `stored_null_lowcov_half_read_20261002`, below). `hybrid_weights_null_20260926`, `allelic_df_fix_20260927`, `plasmode_20260926/summary_before_df_fix.json`
  and the TReCASE smoke run are earlier configurations' records the code still names but no step reads for the
  default set. A new gene set is a new `GENE_SETS` entry; a missing entry
  stops every script at import. `SIMULATED_EFFECTS_GENE_SET` in the environment selects the entry.
- `stratum30_100`: the 30-100-read gene set, 100 genes whose median haplotype-informative reads over
  admitted allelic donors lie in [30, 100) with at least 15 admitted allelic donors, drawn once by
  `select_stratum_genes.py` into `plasmode_stratum30_100_20260927/gene_set` (with its log and
  `pool_stratum.tsv`). Its entry names that directory, its root, its committed run and its stored null
  `stored_null_lowcov_half_read_20261002` (`stored_null.py`, 2026-10-02: 200 permutations of the deep null's stream,
  on which check (d) and the anchor run); it has no before- and after-fix records, TReCASE smoke run or ladder
  (`None`), so 07 and the parts of 08 that read them print a skip. 06 scores it in
  the read bands <30 / 30-50 / 50-100, and 01's check (c) over every gene. Its report leaves out the
  interpretation paragraphs of section 3, section 3.8 and sections 4-5 (they were written for the
  default set), closes with a limits section (the native arms' limits and the comparisons with earlier runs),
  and adds a section setting it, the low-coverage set, against the deep set's run in its `GENE_SETS` root
  (`simulated_effects_half_read_20261001/summary.json`), with three contrast figures, from its selection log,
  `coupling_reach_20260925/b_strata.tsv`, `salmon_half_depth_20260927/summary.json` and the committed
  stratum run's `run_arms.log` (its dataset blocks).
- `cohort/salmon.tsv`, `annot/tx2gene.tsv` and donor 100_D1's dumped equivalence classes (the Salmon
  premise check); `protein_coding_null_store_20260925/permutations.npz` (check d, `common.OLD`, the
  permutation stream of every stored null). Check d reproduces the stored null's first permutation in every column
  from this pipeline's generator.
- Which covariate build a run's arms used: `03_run_arms.py` records it in `run_arms_facts.json`
  (`covariates`), and `05_run_trecase.py` in its `summary.json`; for an output directory made before that record,
  08 infers it from when 03 wrote that file (commit a2f4314, 2026-09-30, moved the expression PCs to the half-read
  build) and says so.

Outputs go to `common.ROOT` only (`simulated_effects_half_read_20261001` for the default gene set,
`simulated_effects_lowcov_half_read_20261001` for `stratum30_100`). `99_acceptance.py` sets `SIMULATED_EFFECTS_ACCEPTANCE=1` before
importing `common`, which makes `common.ROOT` the gene set's `acceptance_root` (`plasmode2_acceptance_20260927`,
`plasmode2_stratum_acceptance_20260927`) for it and every step it runs, so the acceptance never writes into those two.
Outside the acceptance, `SIMULATED_EFFECTS_ROOT` in the environment sends every output to that directory instead (a fresh run to
compare against a delivered one).

## Generator (02_make_datasets.py)

A dataset breaks the genotype association (donor records permuted against fixed genotypes, each
moved record's L/R labels swapped with probability one half; genotype PCs stay with the genotypes,
every other covariate moves with the record) and injects one cis effect per non-null gene by
binomial thinning (Gerard 2020, BMC Bioinformatics) of the haplotype carrying the lower-expressed
allele by `f = 2^-|beta|`, with the collapsed remainder `pT - pL - pR` thinned by the mean factor.
Scenarios `|beta| = 0 / 0.2 / 0.4 / 0.8` with 1 / 3 / 3 / 3 datasets, half the genes null at
`beta > 0`; null genes are thinned by their mean factor so depth is matched. Total Gibbs draws are
thinned like the point estimates. The allelic Gibbs variance of a thinned record is
`dv * q(pL', pR') / q(pL, pR) + q(pL', pR') / ln2^2`, `dv` the real across-draw variance of
`log2((YL + 0.5)/(YR + 0.5))`, `q(x, y) = 1/(x + 0.5) + 1/(y + 0.5)`: Salmon's Gibbs sampler
reassigns the reads of each multi-transcript equivalence class by a multinomial on Gamma-drawn
rates, so the allelic variance beyond counting noise scales with one over the haplotype-specific
reads, which thinning scales by `f` (premise verified in check p). The rule overstates `Va'` by at
most `(1 - f) x 1.4%` at 100-999 and `(1 - f) x 9%` at 10-99 total reads, because Salmon's Gibbs
prior of one pseudo-read per transcript copy does not scale with depth. Truth per non-null gene:
`beta` on the count scale (total: the least-squares slope of the exact log2 total fold on ALT
dosage / 2) and the noise-free shift of the pipeline's own transformed phenotypes on the pipeline
scale. Streams: `SeedSequence(42, (1, r))` permutation and swap, `(2, r)` null set, causal variants
and signs, `(3, r, round(1000 |beta|))` thinning, so scenarios are paired.

The cache's invariants are checked once, in `common.load` (counts non-negative; total at least the
paired haplotypes to `ROUNDING_TOL` in the point estimates and every draw) and `common.setup` (phased
alleles 0 or 1, ALT dosage `xL + xR` over the whole tested set); 02, 03, 05 and 07 rely on them.

What the benchmark cannot answer: efficiency against the true error variance (real data carry no
oracle variance, so only relative efficiency between weightings is measured); which permutation
rule for genotype PCs is right (the null is the record permutation); effects other than thinning at
one variant per gene; null rates at `beta > 0` as calibration (thinning adds model-like binomial
noise that dilutes the real data's weight-residual coupling); and bias against the count-scale
truth separated from the transforms' attenuation (the pipeline-scale truth separates them for an
unweighted fit only). That separation is made outside the benchmark, by refitting its datasets
through `common.run_nominal` with one ingredient changed: `scripts/beta_shortfall_budget.py` and
`scripts/beta_shortfall_refits.py`, page
`brainvar_hapmix_deploy/beta_shortfall_20260929/beta_shortfall.html` (2026-09-29).

## Native-input arms (05b_native_arms.py)

Why (task 2026-09-28): on these datasets TReCASE and RASQUAL ranked non-null genes below total-only tensorQTL, and both are
written for integer alignment counts while the benchmark gave them Salmon point estimates. `05b_native_arms.py` gives TReCASE
its native input and gives split weighting the same input as the control that separates the model from the quantifier.
Inputs: `native_counts_wasp_20260928` (scripts/native_counts.py): featureCounts fragment totals (`-p --countReadPairs -s 2
--primary`, unique, fragments on two genes' exons not counted) and phASER haplotype fragments oriented to the analysis VCF
(a = first GT allele = xL, b = second), counted per transcript strand at SNVs in gene-unique exons on WASP-filtered reads
(`brainvar_hapmix_deploy/wasp_20260928/README.md`), a = b = 0 where a + b exceeds the total; donors joined on the DNA
library id. Earlier stages of these counts: `native_counts_20260928` (unstranded gene spans) and
`native_counts_stranded_20260928` (no WASP).
Native effective library sizes: featureCounts over every gene through `edger_library_normalization.R` (here, a byte-identical
copy of `scripts/edger_library_normalization.R`, the pipeline's own normalization) with the
Salmon cache's `restrict_calibration.txt`, the Salmon run's rule (12,874 genes kept; native / Salmon 0.62-1.27 across donors).
Per dataset: the dataset's `perm` and `swap` applied by 02's `move_records`, a thinned by `fL`, b by `fR`, the remainder by
`(fL + fR) / 2` with 02's `thin_haplotypes` (exact binomial on integers), stream `SeedSequence(42, (7, r, round(1000 |beta|)))`;
`summaries_from_point_estimates` with the counts as the one draw, so `Va` is the counting variance. Arms: `split_native`
(map_nominal and map_cis as 03's split arm, every pair with a + b > 0 admitted, no zero-haplotype rule) and `trecase_native`
(05's runner, Y = native total, Y1 = a, Y2 = b, offset the log native effective library size, same covariates; a gene whose
total has variance below asSeq's `converge`, which asSeq refuses, is not run and has no rows, RAB4B in the deep set). Truths:
the Salmon datasets' count-scale truths, which depend on the causal genotypes and beta only (no pipeline-scale truth for
these arms). The Salmon-input TReCASE arm carries hapmixQTL's zero-haplotype rule (`05_run_trecase.allelic_counts`,
`common.allelic_kept`) and the native arms do not; 08 gives both inputs' informative pair counts, the admitted Salmon count
and the median allele-specific depth from `facts.json`. asSeq runs `SIMULATED_EFFECTS_NATIVE_JOBS` processes (environment; default
15, which with the driver is run_all.sh's cap of 16; the 2026-09-28 runs used 44 under a one-off allowance of 48). Outputs
under `ROOT/native/` (`edger/`, `datasets/`, `results/`, `results_trecase/`, `trecase_work/`, `facts.json`); 06 scores them
(`native_arms` and `trecase_parts` in `summary.json`), 08 adds them to the section 3 tables and figures and a subsection.
Both do so only where `ROOT/native/` exists: in a root without it, such as `99_acceptance.py`'s (which does not run 05b),
each prints a skip line and scores or reports the Salmon-input arms alone, with `native_arms` empty.

## Run order and runtime

`run_all.sh` prints the versions (`common.versions`, also written to `ROOT/versions.log`) and runs the
numbered scripts in order into `common.ROOT`; each step logs to `ROOT/<step>.log` and stops the run on
failure. With the argument `staged` (`run_all.sh staged`) the gene set's committed RASQUAL and TReCASE results
(`common.stage_joint_results`, also used by `99_acceptance.py`; logged to `ROOT/stage_joint_results.log`) replace steps 4 and 5.
On the half-read configuration staging does not serve: the committed results carry fingerprints of datasets made on the
log2(CPM + 1) total, which 06 refuses, so TReCASE is run (05, 64 genes at a time; 2026-10-01: about two hours for the deep
set, every gene's own fit kept in `ROOT/trecase_work` so that a rerun on regenerated datasets reassembles in minutes). Measured 2026-09-27 on the shared 256-core host at load 120-220, one NVIDIA L4, at most 16
processes (~100 s of each Python step is loading the cache):

| step | script | what it writes | runtime |
|---|---|---|---|
| 1 | `01_check_inputs.py` | `checks/salmon_premise.json`, `checks/check_generator.json` (premise, generator checks a-d, plumbing gates e; check d pins the combined se at the stored se times sqrt(M), M Meier's factor recomputed from the stored channel se and dof, for the library at commit a1b2ef4; the gate disposition of the earlier pipeline's per-dataset checks is in its docstring) | 2.5-10 min |
| 2 | `02_make_datasets.py` | `datasets/beta*/rep*.npz`, `meta.json`, `truth.tsv` | 1.5 min |
| 3 | `03_run_arms.py` | `results/<scenario>/<arm>/nominal_*.parquet`, `cis_*.parquet` (every arm of 03), `mixqtl_permutation.json`, `run_arms_facts.json`, `eigenmt_m_eff.tsv` | 15-16 min (GPU 1 and 10 CPU worker processes; set by mixQTL's permutation scan, 240-520 s per dataset per arm; 2026-09-27, load 25-100) |
| 4 | `04_run_rasqual.py` | `results_rasqual/.../nominal_*.parquet`, `summary.json`, per-gene raw checkpoints; the RASQUAL binary's sha256 is pinned (`RASQUAL_SHA256`) and checked at the start of every run | ~13 CPU-h per dataset; 15 jobs (15 processes plus the driver; `SIMULATED_EFFECTS_RASQUAL_JOBS` sets it: 32 for the half-read rerun of 2026-10-02, deep set 05:24-10:46, low-coverage set 10:46-16:24) |
| 5 | `05_run_trecase.py` + `run_trecase.R` | `results_trecase/.../nominal_*.parquet`, `summary.json`; inputs, asSeq files and trace logs under `trecase_work/` | ~15 process-h per dataset; 64 genes at once (`SIMULATED_EFFECTS_TRECASE_JOBS` lowers it when the host is shared; `Rscript` execs into `R`, so one process is live per gene) |
| 5b | `05b_native_arms.py` | `native/`: native datasets, `split_native` and `trecase_native` results, `facts.json` | loading, datasets and split_native about 6 min (GPU 0, 18-20 s per dataset); asSeq on `SIMULATED_EFFECTS_NATIVE_JOBS` processes (default 15), measured on 44: 46.1 min (deep set, 27.5 process-h against 134.4 for the Salmon inputs) and 42.4 min (low-coverage set, 25.1 against 33.3); 2026-09-28, load 50-70 |
| 6 | `06_score.py` | `summary.json` | 4 min |
| 7 | `07_mixqtl_ladder.py` | `ladder/ladder.json`, `total_channel_units.tsv`, rung files (a printed skip for a gene set without a ladder) | 3-4 min |
| 8 | `08_report.py` | `report/plasmode_report.html`, `report/fragments.json` (its sections, for `integrated_report.py`) and four PNG figures; stops if any of its 26 fixed comparative sentences (`check_claims`) no longer holds on the summary | 0.3 min |

Steps 4 and 5 checkpoint per gene (a gene whose raw file or `_status.tsv` exists is not rerun; every
skip is printed).

The two roots of 2026-09-27 (`plasmode_meier_20260927`, `plasmode_lowcov_meier_20260927`) were made step by
step, with the same scripts rather than one `run_all.sh` call: the staged joint results, 02, 03, 06, 07 (deep set
only) and 08; then 01, once check d had been made Meier-aware, into each root's `checks/` (its log
`01_check_inputs.log` beside them), replacing check files first copied from `plasmode2_acceptance_20260927` and
`plasmode2_stratum_acceptance_20260927`; then 06 and 08 again. 05b, then 06 and 08, ran into both roots once per build
of the native counts: unstranded and stranded-without-WASP on 2026-09-28, WASP-filtered on 2026-09-29
(`native_rerun_chain_20260928.log` and `wasp_rerun_chain_20260928.log` in the deploy directory). `ROOT/native/` holds the
WASP run; each root keeps the earlier builds' native directories as `native_unstranded_20260928/` (with the summary and
page before the stranded counts) and `native_stranded_nowasp_20260928/`, and the stranded-without-WASP summary and page
as `stage_stranded_nowasp_20260928/` (`brainvar_hapmix_deploy/wasp_20260928/README.md`, "Copies of each stage").

## Report figures and tables

| report item | made by | from |
|---|---|---|
| Section 3, run facts and checks: Fano factors by band, recovery estimates, the input checks' outcomes | `08_report.py` `sec_run_facts` | `checks/check_generator.json` |
| Section 2 conversion-of-effects table | `tab_conversion` | fixed text (the methods' definitions), no file |
| Section 2, how the results are scored: the definition of every section 3 statistic | `sec_scoring` | fixed text, no file |
| Section 3, run facts and checks, eigenMT paragraph: what sets M_eff, M_eff against the Beta shape2, eigenMT p against pval_beta | `sec_run_facts` | `summary.json` `eigenmt` (06 `eigenmt_structure` on the genotypes, `eigenmt_vs_permutation` on the cis and nominal files) |
| Table 3.1 AUC and power at 5% FDP, band tables | `tab_ranking`, `tab_bands` | `summary.json` `ranking` (06 `ranking`) |
| Figure 1: A AUC, B power at 5% realized FDP, C BH power on the permutation p, D power at 5% realized FDP and E realized FDP of the BH calls, both on the eigenMT p | `fig_ranking`, `eigenmt_panels` | `summary.json` `ranking`, `gene_level`, `gene_level_eigenmt` (D from its `fdp_matched`) |
| Table 3.2 gene-level BH power and null rates, permutation p and eigenMT p, with the power at 5% realized FDP under each eigenMT entry | `tab_gene_level` | `summary.json` `gene_level` (06 `gene_level` on 03's `cis_*.parquet`), `gene_level_eigenmt` (on the nominal files and 03's `eigenmt_m_eff.tsv`; `fdp_matched`) |
| Table 3.3 bias ratios (slope at the causal variant over the count-scale truth, and over the pipeline-scale truth where there is one); Figure 2 (rows: allelic, total, and every arm's one combined slope on the count-scale truth) | `tab_bias`, `fig_bias` | `summary.json` `recovery` (06 `recovery`: `bias_count`, `bias_pipeline`) |
| Tables 3.4 sd(z), squared-error ratios, cross-method; Figure 3 (mixQTL with published cutoffs in the tables only) | `tab_precision`, `tab_cross`, `fig_efficiency` | `summary.json` `precision` (06 `precision`) |
| Table 3.5 lead recovery, band table; Figure 4 | `tab_lead`, `tab_bands`, `fig_lead` | `summary.json` `lead` (06 `lead_recovery`) |
| Table 3.6 detection | `tab_detection` | `summary.json` `detection` (06 `detection`) |
| Tables 3.7 null rates and the anchor | `tab_null`, `tab_anchor` | `summary.json` `null`, `anchor` (06 `null_calibration`, `anchor`); `allelic_df_fix_20260927` draws |
| Tables 3.8 mixQTL ladder (its design in section 2) | `sec_ladder`, `ladder_method` | `ladder/ladder.json` (07) |
| Section 3.9 (deep set) / 3.8 (low-coverage set), native-input arms: inputs, admitted donors, ranking, calibration, recovered share, TReCASE's component tests (their design in section 2, their limits in section 6) | `sec_native`, `native_method`, `native_limits` | `ROOT/native/facts.json`, `ROOT/native/results_trecase/summary.json`, the native counts' `facts.json`, `summary.json` `native_arms`, `trecase_parts`, `trecase_components`, `trecase_native_components` |
| Section 6, the largest theta gradient at an abnormal TReCASE stop | `sec_limits` (through `joint_facts`) | `results_trecase/summary.json` (`joint_na_by_trace`, this run's own trace) |
| `report/fragments.json`: the page's sections one by one, for `integrated_report.py` | `main` | the page as rendered |
| Stratum page: head, section 1 and "The low-coverage set against the deep set" with contrast figures A-C | `stratum_facts`, `sec_head`, `sec_why`, `sec_contrast`, `fig_contrast_calibration`, `fig_contrast_precision`, `fig_contrast_ranking` | `summary.json`, `plasmode_20260926/summary.json`, the gene directory's `select_stratum_genes.log` and `pool_stratum.tsv`, `coupling_reach_20260925/b_strata.tsv`, `salmon_half_depth_20260927/summary.json`, `plasmode_stratum30_100_20260927/run_arms.log` |

## Stored null, gene-level null and the integrated report

Three scripts outside the numbered steps; each reads `SIMULATED_EFFECTS_GENE_SET` like every step, and each states
its design, known answer and output in its docstring.

- `stored_null.py NAME [n=200]` makes a gene set's stored null when it has none: n record permutations with label
  swaps of `common.OLD/permutations.npz` (the deep null's stream; both sets list the same 92 donors in the same order),
  no effect and no thinning, each hapmixQTL arm through `common.run_nominal`. Pass rule, stated before the run:
  split's and unit's combined rate at 0.001 within its gene-clustered interval of 0.001. `--check` must reproduce the
  deep set's stored draw 0 for every arm. The output, `D/NAME`, is wired in as the set's `stored_null` in
  `common.GENE_SETS`, which 01's check (d), 06's anchor and 08 then read. Run once, for the low-coverage set
  (`stored_null_lowcov_half_read_20261002`, 2026-10-02).
- `gene_level_null.py [n=100]` scans n all-null anchor datasets (02 at beta = 0 under replicate r's record permutation
  and label swaps, r from 1000, indices no benchmark dataset uses) for split, unit and total-only tensorQTL as 03 scans
  them, and reports per arm the share of gene-replicate units with `pval_beta` below 0.05 / 0.01 / 0.001, the count per
  replicate, where the benchmark's own anchor falls among them, and the share of replicates with any
  Benjamini-Hochberg call at 5%. It first checks that r = 0 reproduces the stored anchor scan of every arm. Output
  `ROOT/gene_level_null/`.
- `integrated_report.py` writes the one page over both gene sets,
  `D/simulated_effects_integrated_20261002/index.html`: summary figures (inline SVG, each finding worded from the numbers
  it shows), then each set's full record from its `report/fragments.json` with methods, run facts, limits and tables
  folded. It reads each set's `summary.json` and `gene_level_null/summary.json`, and stops if a set's RASQUAL is not
  scored. Run 06 and 08 for both sets first.

06 also stores, per ranking, `fdp_curve`: the null gene units among the top k at every cut the walk down the pooled
ranking can stop at (`called`, `false`, `non_null`); the integrated report draws it as realized false-discovery
proportion against genes called.

## Acceptance test (99_acceptance.py)

`ROOT` in this section is the gene set's acceptance root (`common.GENE_SETS` `acceptance_root`), never a delivered run.
Against the gene set's committed run (`common.GENE_SETS` `committed`):
`/mnt/ssd/lalli/brainvar_hapmix_deploy/plasmode_20260926` (the previous code, commit 3aac315) for the
default set. For `stratum30_100` it is `plasmode_stratum30_100_20260927` and the test runs items 1-4
and 6, with that run's RASQUAL and TReCASE results staged as below; item 5 (the set has no ladder) and
the one-dataset joint runs (`JOINT_CHECK` None) print a skip. It prints the versions, stages the
committed RASQUAL and TReCASE results into `ROOT` (their
`summary.json` derived from the old run's log for RASQUAL), runs steps 1-3 and 6-8 into `ROOT`, and,
only when run as `python3 99_acceptance.py joint`, runs steps 4 and 5 on ONE dataset (`beta0.8` rep
000) into `ROOT/joint_check` alongside the GPU steps, with `JOINT_JOBS` 5 RASQUAL and 4 TReCASE jobs
(the recorded pass, `plasmode2_acceptance_20260927/acceptance_df76f3b.log` (`JOINT_PASS`), used these values and took 97 and 136 min). Without
`joint` each joint arm prints a SKIP naming that recorded PASS line and the files among 04 / 05, their
imports and `common.py` whose sha256 differs from that run's header, and the SKIP is not a failure:
the rerun costs hours and its stamp includes `common.py`, which changes more often than anything 04
and 05 read from it. The code comments budget two processes per TReCASE job; `ps` on the recorded
run shows `Rscript` exec'ing into `R`, one process per job, so the run held 5 + 4 + the driver = 10
processes while the joint runs were alone and 11 while a pipeline step ran beside them. Checks: (1) every dataset array bit for bit
and the same number of dataset files; (2) the six arms' map_nominal outputs bit for bit on the scored
columns, on the same (phenotype_id, variant_id) row set; (3) map_cis leads and num_var identical, the
same finite / NaN pattern, and pval_beta and pval_perm within 1e-6 relative on every dataset; (4)
`summary.json` walked over the union of both files' leaves: every numeric leaf within 1e-9 relative,
every string except a path exactly, no leaf only in the new file except a path string (03's
`mixqtl_permutation/source`), and leaves only in the committed file allowed only in the families 06 no
longer writes (`anchor_passed`, `cannot_answer`, `units`, `gene_set`,
`anchor/*/*/*/stored_without_one_df`, `detection/*/*/*/nonfinite_p`, `gene_level/*/*/{datasets,
nonfinite_p_for_power, p_for_power, null_rate_pval_perm/*}`, `null/**/dataset_lo|hi`,
`precision/**/nonnull_excluded/*`, `ranking/*/*/auc/*/datasets`; 1,163 leaves, none read by either
report); the new 04 and 05 checked on the one dataset bit for bit on every column and on the summary
counts the report reads against the committed run's; (5) `ladder.json` as (4), with the dropped families
`smoke`, `ladder/*/*/auc/*/datasets`, `ladder/*/*/own_set_excluded/*` (196 leaves); (6) the numeric
tokens of the report page (images, style, path-like tokens, dates and commit ids removed) equal as
multisets. Its output is in `ROOT/acceptance.log`.

Steps are skipped when their output exists under an unchanged stamp: the sha256 of the script,
`common.py` and the scripts it imports, plus the stamps of the steps whose outputs it reads
(`acceptance_stamps.json`; what invalidated a stamp is printed). The joint runs are stamped the same
way (`joint_check/stamp_<arm>.json`, written when a run starts), so a run under the same stamp resumes
its per-gene checkpoints and a changed stamp wipes them. About 2-3 h with `joint`, set by the joint
runs; without it about 27 min on the default set when every step reruns (01-08 took 26.5 min on
2026-09-27) and under 2 min when only 08 does.

## Requirements

`requirements.txt` (pip freeze of the packages used; python 3.11.14), R 4.5.2 with asSeq 0.99.501
in `/mnt/ssd/lalli/usr/local/lib/R/library` (run with `R_LD_LIBRARY_PATH` and `LD_LIBRARY_PATH` as
`05_run_trecase.R_ENV` sets them), the RASQUAL binary named in `04_run_rasqual.RASQUAL`
(`/mnt/ssd/lalli/usr/local/rasqual/bin/rasqual`, its sha256 pinned there), and the repository's `tensorqtl`
package (imported, not modified). Nothing under `scripts/` is imported or run.

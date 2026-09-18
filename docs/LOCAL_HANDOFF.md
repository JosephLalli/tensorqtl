# Moving this work to a local server

A Claude Code cloud session cannot be relocated — it is server-side state. What
transfers is the **repository**, which is where all of the work lives. Nothing in
the cloud container is load-bearing except the git history, and that is pushed.

## 1. What is already portable

Everything. The working tree is clean and pushed to
`claude/hapmixqtl-gibbs-uncertainty-IQ6Za`. That includes:

- the library changes: the `tau_mode='estimate'` default and its exact
  DerSimonian-Laird denominator, tau estimated on each channel's informative
  samples, the zero-information ASE weight guard, the sparse-channel rule,
  per-channel covariate designs (`ase_covariates_df`, `SAME_COVARIATES`), the
  Freedman-Lane permutation null over leverage-standardized whitened
  residuals, the lead refit (`map_cis(tau_refit=True)`), the sample-order
  guard `_assert_phase_columns`, and `reference_bias_diagnostic` with
  `orient_haplotypes`
- every validation harness under `tests/ase_*.py`
- every result as JSON under `docs/ase_*.json` — the numbers do not need re-running
- `docs/ase_validation.md`, the full record (§1–§10)
- `docs/hapmixqtl_methods.md`, the method specified for reproduction:
  the model, the estimator at one variant, the scan and gene-level
  inference, the measured properties and the assumptions
- `scripts/build_rasqual.sh` — builds the real RASQUAL from source
- `scripts/prep_brainvar.py` — the BrainVar ingest pipeline
- `scripts/extract_gtex_phaser.py` — regenerates the §7d real-data inputs
- the scripts of the head-to-head comparison chain, each with a
  `--selftest` that runs on fabricated inputs in seconds:
  `gtf_to_tables.py` (annotation tables from a GTF), `phaser_to_matrix.py`
  (phASER output to matrices, plus the read-backed phase overlay),
  `build_covariates.py` (the shared covariate matrix, written to
  `cov/covariates.tsv` and passed to both arms rather than pre-residualized),
  `select_pilot_genes.py` (a gene list both methods can use, stratified by
  expression and bounded by RASQUAL's cost), `build_asvcf.py` (the
  AS-annotated VCF RASQUAL reads), `make_rasqual_inputs.py` (RASQUAL's native
  inputs), and `compare_pipelines.py` (the comparison itself)
- `docs/brainvar_deploy_runbook.md` — the operating procedure for running that
  chain on BrainVar, end to end

## 2. What does NOT transfer

**Conversation context.** A fresh local session starts with no memory of this
one. `docs/ase_validation.md` is deliberately written to be the handoff: it
states not just the results but why each experiment was designed the way it was,
and records the bugs found along the way. Point a new session at it first.

**Installed packages.** Recreate with:

```bash
# system (for the real RASQUAL build)
sudo apt-get install -y libgsl-dev liblapack-dev libblas-dev zlib1g-dev

# python
pip install numpy scipy pandas torch pandas_plink h5py qtl pysam pytest
pip install -e . --no-deps        # tensorqtl itself, for the test suite
```

Versions this work was validated against: numpy 2.4.6, scipy 1.17.1,
pandas 3.0.5, torch 2.14.0, pandas-plink 2.3.2, h5py 3.16.0, qtl 0.1.10.
Those are not a floor. On 2026-09-13 the hapmixQTL suite
(`tests/test_hapmixqtl.py`, `tests/test_hapmixqtl_calibration.py`,
`tests/test_cli.py`, 72 tests) also passed on a materially older stack:
numpy 1.26.4, scipy 1.16.2, pandas 2.2.3, torch 2.7.0, pandas-plink 2.3.2,
h5py 3.13.0, qtl 0.1.10, pysam 0.24.0.

**R on this machine cannot run weighted least squares (2026-09-18), which
blocks most but not all cross-checks against limma.** `stats::lm.wfit`
segfaults: the BLAS is Debian's
`/usr/lib/x86_64-linux-gnu/openblas-pthread/libblas.so.3` while the LAPACK is
a Homebrew openblas (a mixed-BLAS crash). This blocks any limma path that
fits a linear model (e.g. `vooma`/`voomaLmFit`, `fitFDistRobustly` calls
`lm.wfit` internally) — see CLAUDE.md's "Relationship to limma, edgeR,
sleuth, swish" section, where those checks were done against limma's
formulas in numpy instead. `squeezeVar` is unaffected (it is a closed-form
posterior, no `lm.wfit` call) and was run directly against BrainVar data at
`/mnt/ssd/lalli/brainvar_hapmix_deploy/variance_layer_mapping_20260918/`
(not under git). Installed R
package versions are also a release behind the upstream devel source read
2026-09-18: limma 3.64.3 installed vs 3.99.0 upstream; edgeR 4.6.3 installed
vs 4.99.6 upstream. `catchSalmonGene`, `binQLFit`, `PCList`, and
`sampleWeights` exist only in the upstream devel edgeR, not in the installed
package.

**Intermediate data.** The GTEx haplotype-count extracts used in §7d are not
committed (they are derived data). Regenerate:

```bash
python3 scripts/extract_gtex_phaser.py \
    phASER_GTEx_v8_matrix.gw_phased.txt.gz "Muscle - Skeletal" 400 gtex_muscle.npz
```

## 3. Steps on the local server

```bash
git clone <repo> && cd tensorqtl
git checkout claude/hapmixqtl-gibbs-uncertainty-IQ6Za
# install as above
pytest tests/test_hapmixqtl.py -q          # expect 54 passed
./scripts/build_rasqual.sh                 # optional; real RASQUAL
claude                                     # start a session in the repo
```

Then, for BrainVar:

```bash
python3 scripts/prep_brainvar.py --phaser <phaser_matrix> --meta <samples.tsv> \
    --sign <sign.npy> --out prepped/ --id-col <...> --age-col <...>
```

The `--sign` argument enables the reference-bias gate. It is the one
precondition hapmixQTL cannot check for itself, and §7i shows the failure mode
is catastrophic rather than gradual, so supply it.

For the hapmixQTL-versus-RASQUAL head-to-head, follow
`docs/brainvar_deploy_runbook.md` instead. It covers the annotation tables,
the phASER run and the phase overlay, the comparison invocation, and the
prerequisite that governs whether the comparison is possible at all: Salmon
must have been run against a personalized **diploid** transcriptome with
`--numGibbsSamples 200`, or hapmixQTL has no allelic information to read.

## 4. Before you do this: a data-governance check

BrainVar is dbGaP controlled-access (phs001900). Running an AI coding assistant
on a machine holding controlled-access genomic data means file contents can be
transmitted to a third-party API. **Check your Data Use Agreement and your
institution's policy before doing that** — many DUAs restrict transmission of
individual-level data to external services, and this is a decision for the data
custodian, not a technical detail.

Two patterns that usually stay clean:

1. **Keep the assistant away from individual-level data.** Run `prep_brainvar.py`
   and phASER yourself; the outputs are per-gene aggregate counts. Analysis and
   iteration then happen on derived summaries.
2. **Split the work.** Do data-touching steps on the controlled-access machine,
   and method development in a sandbox on simulated or public data — which is
   exactly what this repo has been doing (GTEx public phASER matrices in §7d).

Neither is a substitute for checking the DUA.

## 5. Where the work stands

Done and validated: the τ defect (found, fixed, confirmed on real GTEx data),
the zero-information weight bug (found, fixed), `Cat` retired with a mechanism,
phasing error / covariates / robust SEs resolved, fine-mapping calibration,
compute cost, and the reference-bias diagnostic.

Open:

- **The real-RASQUAL head-to-head.** The harness runs
  (`tests/ase_rasqual_real.py`) but the generative calibration is unsettled:
  RASQUAL's shared θ ties beta-binomial precision to NB dispersion, so at a
  realistic NB dispersion the simulated allelic ratios are far noisier than real
  ASE, penalising every ASE method. Settle this before quoting numbers.

  On real data this is no longer blocked on tooling: `compare_pipelines.py`
  gives each method its native input on the same genes and the same phase, and
  `docs/brainvar_deploy_runbook.md` is the procedure **and the record of how
  far it has been run** — read its pilot sections for the current state rather
  than this file, which is not updated per run. The inputs are all on the
  server. Personalized diploid quantifications with 200 Gibbs draws cover 228
  subjects (`personalized_T2T_NCBI110_pseudoalignment`), and reference-aligned
  T2T BAMs for phASER cover 93 — **92 subjects in common**, which is the usable
  N. That arm has no aligned reads of its own, so the allelic counts come from
  the reference-aligned arm, and its BAMs are named by RefSeq accession where
  the VCF uses `chr`; both are handled in the runbook. phASER and the pilot
  comparisons have since been run: the runbook records both arms on 30 genes
  (`pilotI` through `pilotN`), with calibration, the per-channel covariate
  check, the lead refit and the matched-variant effect comparison. What
  remains is the run at scale.
- **Four axes blocked on genotypes** (§9): effect-size concordance against GTEx
  aFC, eGene replication, functional/motif enrichment, and the `slope_a` vs
  `slope_tc` concordance check. BrainVar unblocks all four.

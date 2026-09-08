# Moving this work to a local server

A Claude Code cloud session cannot be relocated — it is server-side state. What
transfers is the **repository**, which is where all of the work lives. Nothing in
the cloud container is load-bearing except the git history, and that is pushed.

## 1. What is already portable

Everything. The working tree is clean and pushed to
`claude/hapmixqtl-gibbs-uncertainty-IQ6Za`. That includes:

- the two production fixes (`tau_mode='estimate'` default; the
  zero-information ASE weight guard) and `reference_bias_diagnostic`
- every validation harness under `tests/ase_*.py`
- every result as JSON under `docs/ase_*.json` — the numbers do not need re-running
- `docs/ase_validation.md`, the full record (§1–§10)
- `scripts/build_rasqual.sh` — builds the real RASQUAL from source
- `scripts/prep_brainvar.py` — the BrainVar ingest pipeline
- `scripts/extract_gtex_phaser.py` — regenerates the §7d real-data inputs

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
pip install numpy scipy pandas torch pandas_plink h5py qtl pytest
pip install -e . --no-deps        # tensorqtl itself, for the test suite
```

Versions this work was validated against: numpy 2.4.6, scipy 1.17.1,
pandas 3.0.5, torch 2.14.0, pandas-plink 2.3.2, h5py 3.16.0, qtl 0.1.10.

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
pytest tests/test_hapmixqtl.py -q          # expect 32 passed
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
- **Four axes blocked on genotypes** (§9): effect-size concordance against GTEx
  aFC, eGene replication, functional/motif enrichment, and the `slope_a` vs
  `slope_tc` concordance check. BrainVar unblocks all four.

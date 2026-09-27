#!/usr/bin/env bash
# Regenerate the plasmode benchmark from the Salmon cache. Design:
# docs/simulation_benchmark_spec.md; each script's docstring states its rules.
# Gene set: make_datasets.GENE_SET, from PLASMODE_GENE_SET in the environment
# (default corrected_null_store, the committed 100-gene run; stratum30_100 is
# the 30-100-read set of select_stratum_genes.py, run first for that set).
# Steps, in order: the Salmon premise check; the generator's known-answer
# checks; the datasets (make_datasets.BETAS = 0 / 0.2 / 0.4 / 0.8 with
# N_DATASETS per gene set in make_datasets.GENE_SETS, 1 / 3 / 3 / 3 for both
# sets as run (user decisions 2026-09-26 and 2026-09-27);
# beta = 0 is the null anchor); run_arms.py (per dataset map_nominal and GPU
# map_cis under four hapmixQTL weightings, mixqtl_scan under two mixQTL cutoff
# settings; the mixQTL permutation scan is off for the committed set and
# re-timed on the first dataset of any other, run_arms.MIXQTL_PERM);
# run_rasqual.py; run_trecase_asseq.py (asSeq; sets the R library environment
# itself); mixqtl_ladder.py (only for a set whose GENE_SETS entry has ladder
# True; otherwise skipped with a printed line); score.py; report.py. Paths and
# counts are the Python defaults (make_datasets.ROOT and friends); each step
# stops the run on failure and is logged to $ROOT/<step>.log. Measured
# 2026-09-26/27 on the shared 256-core host: run_arms ~25 min for the 10
# datasets (one NVIDIA L4), run_rasqual ~1.5 h at 64 jobs, run_trecase_asseq
# ~3 h at 48 processes, mixqtl_ladder ~7 min, score ~4 min, plus ~100 s per
# script to load the cache.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
ROOT="$(cd scripts/plasmode && python3 -c 'import make_datasets; print(make_datasets.ROOT)')"
GENE_SET="$(cd scripts/plasmode && python3 -c 'import make_datasets; print(make_datasets.GENE_SET)')"
LADDER="$(cd scripts/plasmode && python3 -c 'import make_datasets; print(int(make_datasets.SET["ladder"]))')"
mkdir -p "$ROOT"
echo "gene set $GENE_SET (PLASMODE_GENE_SET=${PLASMODE_GENE_SET:-unset}); root $ROOT"

{
  date
  echo "repo $REPO at $(git rev-parse HEAD); git status --porcelain:"
  git status --porcelain
  sha256sum scripts/plasmode/*.py scripts/plasmode/run_all.sh tensorqtl/hapmixqtl.py \
    tensorqtl/mixqtl_replication.py scripts/compare_mixqtl_replication.py scripts/corrected_null_store.py \
    scripts/hybrid_weights_null.py scripts/null_permutation_instrument.py
  python3 -c 'import sys, numpy, scipy, pandas, pyarrow, torch
print("python", sys.version.split()[0], "numpy", numpy.__version__, "scipy", scipy.__version__,
      "pandas", pandas.__version__, "pyarrow", pyarrow.__version__, "torch", torch.__version__,
      "cuda available", torch.cuda.is_available())'
} | tee "$ROOT/versions.log"

step() {   # step <log name> <script> [args...]
  local log="$ROOT/$1.log"
  shift
  echo "== python3 $* > $log"
  python3 "$@" 2>&1 | tee "$log"
}

step check_salmon_premise scripts/plasmode/check_salmon_premise.py
step check_generator scripts/plasmode/check_generator.py
step make_datasets scripts/plasmode/make_datasets.py
step run_arms scripts/plasmode/run_arms.py
step run_rasqual scripts/plasmode/run_rasqual.py
step run_trecase_asseq scripts/plasmode/run_trecase_asseq.py
if [ "$LADDER" = 1 ]; then
  step mixqtl_ladder scripts/plasmode/mixqtl_ladder.py
else
  echo "== mixqtl_ladder skipped: gene set $GENE_SET has no ladder (make_datasets.GENE_SETS[...]['ladder'] is False)"
fi
step score scripts/plasmode/score.py
step report scripts/plasmode/report.py

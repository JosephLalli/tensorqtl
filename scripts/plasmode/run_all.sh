#!/usr/bin/env bash
# Regenerate the plasmode benchmark from the Salmon cache. Design:
# docs/simulation_benchmark_spec.md; each script's docstring states its rules.
# Steps: the Salmon premise check, the generator's known-answer checks, the
# datasets (make_datasets.BETAS = 0 / 0.2 / 0.4 / 0.8 with N_DATASETS =
# 1 / 3 / 3 / 3, user decision 2026-09-26; beta = 0 is the null anchor), then
# run_arms.py: per dataset map_nominal and map_cis (1,000 records_signflip
# permutations, Beta approximation, GPU) under four hapmixQTL weightings, and
# mixqtl_scan under two mixQTL cutoff settings, with mixQTL's gene-level
# permutation p only if one dataset of mixqtl_permutation_scan fits a 300 s
# budget (the timing rule, decided on the first dataset and recorded in
# results/mixqtl_permutation.json); then the score, which reports the beta = 0
# anchor against the stored nulls without stopping. Paths and counts are the Python defaults
# (make_datasets.ROOT and friends); each step stops the run on failure and is
# logged to $ROOT/<step>.log. Runtime, measured on the 2026-09-26 smoke run
# (1 dataset per scenario, one NVIDIA L4, host load ~100): per dataset,
# map_nominal 5.5-8.7 s and map_cis 17.9-21.0 s per hapmixQTL arm, mixqtl_scan
# 1.6-9.5 s per mixQTL arm plus 3.7-6.5 s for the published arm's gate, about
# 2 min in all; the timing rule spent 307 s once and excluded the mixQTL
# permutation scan (~1,305 s per dataset). The 10 datasets of the full run
# are therefore ~25 min of run_arms, plus ~100 s per script to load the cache
# and ~4 min of scoring. The anchor is the smoke's beta = 0 dataset (same
# seeds); it failed the pass rule there (score.py docstring, (6)).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
ROOT="$(cd scripts/plasmode && python3 -c 'import make_datasets; print(make_datasets.ROOT)')"
mkdir -p "$ROOT"

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
step score scripts/plasmode/score.py
step report scripts/plasmode/report.py

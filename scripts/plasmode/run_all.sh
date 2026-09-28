#!/usr/bin/env bash
# Regenerate the plasmode benchmark from the Salmon cache into common.ROOT (README.md: run order, runtime).
# The gene set is common.GENE_SET, from PLASMODE_GENE_SET in the environment (default the 100-gene set;
# stratum30_100 is the 30-100-read set, made once by select_stratum_genes.py). Each step stops the run on failure and is logged to $ROOT/<step>.log. GPU 1 for map_nominal / map_cis
# (shared host, 2026-09-27); at most 16 processes at once (04 and 05 set JOBS accordingly). With the argument staged,
# the committed run's RASQUAL and TReCASE results (common.stage_joint_results) replace steps 4 and 5, which take hours each.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE" && python3 -c 'import common; print(common.ROOT)')"
mkdir -p "$ROOT"
export CUDA_VISIBLE_DEVICES=1

(cd "$HERE" && python3 -c 'import common; print("\n".join(common.versions()))')   # also written to $ROOT/versions.log

joint=(04_run_rasqual.py 05_run_trecase.py)
if [[ "${1:-}" == staged ]]; then
  (cd "$HERE" && python3 -c 'import common; common.stage_joint_results()') 2>&1 | tee "$ROOT/stage_joint_results.log"
  joint=()
elif [[ $# -gt 0 ]]; then
  echo "usage: $0 [staged]" >&2
  exit 2
fi

for script in 01_check_inputs.py 02_make_datasets.py 03_run_arms.py "${joint[@]}" 06_score.py 07_mixqtl_ladder.py 08_report.py; do
  echo "== $script > $ROOT/${script%.py}.log"
  python3 "$HERE/$script" 2>&1 | tee "$ROOT/${script%.py}.log"
done

#!/usr/bin/env bash
# Assemble the static site for the hapmixQTL calibration audit.
# Single source of truth stays in docs/; this only stages it for deploy.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
out="$root/site"
mkdir -p "$out"
cp "$root/docs/ase_validation_report.html"  "$out/index.html"
cp "$root/docs/ase_validation_results.json" "$out/ase_validation_results.json"
cp "$root/docs/ase_validation.md"           "$out/ase_validation.md"
echo "built $out:"; ls -1 "$out"

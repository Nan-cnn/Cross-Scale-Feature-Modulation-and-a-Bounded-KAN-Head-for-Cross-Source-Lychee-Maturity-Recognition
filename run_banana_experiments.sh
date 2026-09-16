#!/usr/bin/env bash
set -euo pipefail

export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

CONFIG="configs/banana_224.json"
RESULTS="outputs/banana_224"

python audit_dataset.py \
  --config "${CONFIG}" \
  --output-dir reports/data_audit_banana

python run_experiments.py \
  --config "${CONFIG}" \
  --suites comparison \
  --dry-run

python -u run_experiments.py \
  --config "${CONFIG}" \
  --suites comparison \
  --device cuda \
  2>&1 | tee -a banana_224.log

python analyze_banana_results.py \
  --config "${CONFIG}" \
  --results-dir "${RESULTS}"

python visualize_banana_results.py \
  --results-dir "${RESULTS}"

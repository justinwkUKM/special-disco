#!/usr/bin/env bash
set -euo pipefail

# Determine repository root relative to this script
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$REPO_ROOT"

PYTHON=${PYTHON:-python}

echo "[run_quality_checks] Running unit tests with pytest..."
$PYTHON -m pytest -q

echo "[run_quality_checks] Running dataset pipeline..."
$PYTHON pii_pipeline/run_pipeline.py

OUTPUT_JSONL="pii_pipeline/outputs/final_dataset/pii_training_dataset.jsonl"

if [[ -f "$OUTPUT_JSONL" ]]; then
  echo "[run_quality_checks] Validating generated dataset..."
  $PYTHON pii_pipeline/validate_dataset.py "$OUTPUT_JSONL"
else
  echo "[run_quality_checks] WARNING: Expected dataset file not found at $OUTPUT_JSONL" >&2
  exit 1
fi

echo "[run_quality_checks] All checks completed."

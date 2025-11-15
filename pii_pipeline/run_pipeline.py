"""
run_pipeline.py
Main entrypoint for generating the full PII training dataset.

This script:
1. Loads config
2. Loads clean samples
3. Generates regex-based variants
4. (Optional) Generates teacher-model variants
5. Validates + deduplicates
6. Exports final JSONL dataset

Usage:
    python run_pipeline.py
"""

import sys
from utils import log
from dataset_generator import export_dataset_jsonl


def main():
    log("=== PII DATASET PIPELINE STARTED ===")

    try:
        export_dataset_jsonl()
        log("=== PIPELINE COMPLETED SUCCESSFULLY ===")

    except Exception as e:
        log(f"FATAL ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()


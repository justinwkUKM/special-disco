"""
config.py
Central configuration for the PII dataset generation pipeline.

This file defines:
- Paths for input/output
- Randomization settings
- Variant generation counts
- Teacher model configuration
- Logging configuration

The pipeline imports this config everywhere else.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# DIRECTORY CONFIG
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent

CLEAN_SAMPLE_DIR = ROOT_DIR / "clean_samples"
OUTPUT_DIR = ROOT_DIR / "outputs"

RAW_MUTATED_DIR = OUTPUT_DIR / "raw_mutated"
TEACHER_GENERATED_DIR = OUTPUT_DIR / "teacher_generated"
FINAL_DATASET_DIR = OUTPUT_DIR / "final_dataset"

# Auto-create directories on import
for _dir in [
    CLEAN_SAMPLE_DIR,
    RAW_MUTATED_DIR,
    TEACHER_GENERATED_DIR,
    FINAL_DATASET_DIR,
]:
    _dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# DATASET GENERATION CONFIG
# ---------------------------------------------------------------------------

# Number of regex/noise-based corrupted variants per clean sample
REGEX_VARIANTS_PER_SAMPLE = 6

# Number of teacher-model corrupted variants per clean sample
TEACHER_VARIANTS_PER_SAMPLE = 4

# Maximum allowed variants per PII type (used in balancing)
MAX_VARIANTS_PER_TYPE = 5000

# Whether to shuffle dataset before final export
SHUFFLE_FINAL_DATASET = True

# Seed for deterministic pipeline
RNG_SEED = 12345


# ---------------------------------------------------------------------------
# MODEL CONFIG
# ---------------------------------------------------------------------------

TEACHER_MODEL_NAME = "gpt-5.1"   # or openai/gpt-5.1, depending on API
TEMPERATURE = 0.8
TOP_P = 0.9
MAX_TOKENS = 350

# ---------------------------------------------------------------------------
# LOGGING CONFIG
# ---------------------------------------------------------------------------

LOGGING = {
    "level": "INFO",
    "format": "[%(asctime)s] [%(levelname)s] %(message)s",
}

# ---------------------------------------------------------------------------
# FILE NAMES
# ---------------------------------------------------------------------------

BASE_CLEAN_FILE = CLEAN_SAMPLE_DIR / "base_clean_samples.json"

FINAL_JSONL_FILENAME = "pii_training_dataset.jsonl"


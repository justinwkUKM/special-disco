"""
dataset_generator.py
Full pipeline logic for generating the PII redaction dataset:

Process:
1. Load base clean samples.
2. Generate multiple regex/noise-based corrupted variants.
3. Query teacher model for deep corrupted variants + gold redactions.
4. Combine all variants.
5. Deduplicate.
6. Validate against schemas.
7. Export JSONL dataset.

Uses:
- pii_mutation_engine_v2.py
- teacher_prompts.py
- utils.py
- schemas.py
- config.py
"""

import json
from pathlib import Path
from typing import List, Dict

from config import (
    BASE_CLEAN_FILE,
    RAW_MUTATED_DIR,
    TEACHER_GENERATED_DIR,
    FINAL_DATASET_DIR,
    REGEX_VARIANTS_PER_SAMPLE,
    TEACHER_VARIANTS_PER_SAMPLE,
    SHUFFLE_FINAL_DATASET,
)

from schemas import CleanSample, MutatedVariant, FinalRecord
from utils import (
    read_jsonl,
    write_json,
    write_jsonl,
    append_jsonl,
    log,
    generate_id,
    rand_choice,
)
from pii_mutation_engine_v2 import mutate_context
from teacher_prompts import general_noise_prompt, multi_pii_super_prompt
from teacher_prompts import email_noise_prompt, phone_noise_prompt
from teacher_prompts import address_noise_prompt, credit_card_noise_prompt
from teacher_prompts import gender_race_age_noise_prompt


# ---------------------------------------------------------------------------
# STEP 1 — LOAD CLEAN SAMPLES
# ---------------------------------------------------------------------------

def load_clean_samples() -> List[CleanSample]:
    if not BASE_CLEAN_FILE.exists():
        raise FileNotFoundError(f"Clean base file not found: {BASE_CLEAN_FILE}")
    data = json.loads(BASE_CLEAN_FILE.read_text())
    clean_samples = []
    for item in data:
        cs = CleanSample(**item)
        cs.validate()
        clean_samples.append(cs)
    log(f"Loaded {len(clean_samples)} clean samples.")
    return clean_samples


# ---------------------------------------------------------------------------
# STEP 2 — REGEX-BASED MUTATIONS
# ---------------------------------------------------------------------------

def generate_regex_mutations(sample: CleanSample) -> List[MutatedVariant]:
    variants = []
    mutated_contexts = mutate_context(
        sample.context,
        num=REGEX_VARIANTS_PER_SAMPLE
    )

    for ctx in mutated_contexts:
        mv = MutatedVariant(
            id=generate_id("regex"),
            parent_id=sample.id,
            mutated_context=ctx,
            mutation_type="regex",
            metadata={"note": "regex-based corruption"}
        )
        mv.validate()
        variants.append(mv)

    return variants


# ---------------------------------------------------------------------------
# STEP 3 — TEACHER-MODEL MUTATIONS
# ---------------------------------------------------------------------------

def call_teacher_model(prompt: str) -> List[Dict[str, any]]:
    """
    Placeholder method for ChatGPT 5.1 API call.

    Expected return:
      [
        {
          "corrupted": "...",
          "answer": {
              "redacted_text": "...",
              "entities": [...]
          }
        },
        ...
      ]
    """

    # Developer will implement this with their API key.
    #
    # return openai.chat.completions(...)
    #
    # Here we raise, because we don't execute teacher calls in pipeline.
    raise NotImplementedError("Teacher-model call must be implemented by the user.")


def generate_teacher_mutations(sample: CleanSample) -> List[Dict[str, any]]:
    """
    Select prompt based on which PII types appear.
    Teacher generates:
      - multiple corrupted variants
      - gold-standard JSON redactions
    """

    context = sample.context.lower()
    if "@" in context or "(at)" in context:
        prompt = email_noise_prompt(sample.context)
    elif any(x in context for x in ["-", "(", ")", "+", "call"]):
        prompt = phone_noise_prompt(sample.context)
    elif any(x in context for x in ["st", "street", "ave", "road", "rd"]):
        prompt = address_noise_prompt(sample.context)
    elif any(x.isdigit() for x in context) and len(context) > 14:
        prompt = credit_card_noise_prompt(sample.context)
    else:
        # fallback
        prompt = general_noise_prompt(sample.context)

    # Real logic:
    # teacher_outputs = call_teacher_model(prompt)

    # For now:
    raise NotImplementedError(
        "Teacher generating function requires implementing call_teacher_model."
    )


# ---------------------------------------------------------------------------
# STEP 4 — PACK FINAL RECORDS
# ---------------------------------------------------------------------------

def create_final_records(
    sample: CleanSample,
    mutated_variants: List[MutatedVariant],
    teacher_variants: List[Dict[str, any]] = None
) -> List[FinalRecord]:

    records = []

    # Regex variants require teacher redaction (gold label)
    for mv in mutated_variants:
        # placeholder: no teacher model used here
        record = FinalRecord(
            id=generate_id("rec"),
            question=sample.question,
            context=mv.mutated_context,
            answer=sample.answer  # TEMPORARY placeholder
        )
        record.validate()
        records.append(record)

    # Teacher-provided gold variants (when API implemented)
    if teacher_variants:
        for tv in teacher_variants:
            record = FinalRecord(
                id=generate_id("rec"),
                question=sample.question,
                context=tv["corrupted"],
                answer=tv["answer"]
            )
            record.validate()
            records.append(record)

    return records


# ---------------------------------------------------------------------------
# STEP 5 — DEDUPLICATION
# ---------------------------------------------------------------------------

def dedupe_records(records: List[FinalRecord]) -> List[FinalRecord]:
    seen = set()
    unique = []
    for r in records:
        key = (r.context, r.answer["redacted_text"])
        if key not in seen:
            unique.append(r)
            seen.add(key)
    return unique


# ---------------------------------------------------------------------------
# STEP 6 — PIPELINE RUNNER
# ---------------------------------------------------------------------------

def generate_full_dataset() -> List[Dict[str, any]]:
    clean_samples = load_clean_samples()
    all_records: List[FinalRecord] = []

    for sample in clean_samples:
        log(f"Processing sample {sample.id}...")

        # 1. Regex corruption
        regex_variants = generate_regex_mutations(sample)

        # 2. Teacher corruption (optional)
        # teacher_variants = generate_teacher_mutations(sample)

        teacher_variants = []  # Placeholder (teacher not executed here)

        # 3. Pack records
        records = create_final_records(sample, regex_variants, teacher_variants)

        all_records.extend(records)

    # Deduplicate
    all_records = dedupe_records(all_records)

    # Shuffle
    if SHUFFLE_FINAL_DATASET:
        import random
        random.shuffle(all_records)

    return [r.__dict__ for r in all_records]


# ---------------------------------------------------------------------------
# EXPORTER
# ---------------------------------------------------------------------------

def export_dataset_jsonl():
    final_records = generate_full_dataset()
    out_path = FINAL_DATASET_DIR / "pii_training_dataset.jsonl"
    write_jsonl(out_path, final_records)
    log(f"Exported {len(final_records)} final records to: {out_path}")


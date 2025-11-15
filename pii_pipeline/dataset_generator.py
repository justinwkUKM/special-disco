"""
dataset_generator.py
Full pipeline logic for generating the PII redaction dataset.

Process:
1. Load base clean samples.
2. Generate multiple regex/noise-based corrupted variants.
3. Query teacher model for deep corrupted variants + gold redactions.
4. Combine all variants.
5. Deduplicate.
6. Validate against schemas.
7. Export JSONL dataset.
"""

import json
from typing import List, Dict, Any

from config import (
    BASE_CLEAN_FILE,
    FINAL_DATASET_DIR,
    REGEX_VARIANTS_PER_SAMPLE,
    TEACHER_VARIANTS_PER_SAMPLE,
    SHUFFLE_FINAL_DATASET,
)
from schemas import CleanSample, MutatedVariant, FinalRecord, validate_entities
from utils import (
    write_jsonl,
    log,
    generate_id,
)
from pii_mutation_engine_v2 import mutate_context
from teacher_prompts import (
    general_noise_prompt,
    email_noise_prompt,
    phone_noise_prompt,
    address_noise_prompt,
    credit_card_noise_prompt,
)
from teacher_api import call_teacher_model, call_teacher_redact_single


# Whether to ask the teacher to label regex-mutated variants as well
USE_TEACHER_FOR_REGEX_LABELS = True


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def is_valid_teacher_answer(answer: Dict[str, Any]) -> bool:
    """
    Validate that a teacher-provided answer has:
      - 'redacted_text' (str)
      - 'entities' (list) that passes validate_entities()
    """
    if not isinstance(answer, dict):
        return False
    if "redacted_text" not in answer or "entities" not in answer:
        return False
    if not isinstance(answer["redacted_text"], str):
        return False
    if not isinstance(answer["entities"], list):
        return False
    if not validate_entities(answer["entities"]):
        return False
    return True


def log_record_added(record_type: str, sample_id: str, record_id: str, context_preview: str):
    """
    Logs a single record addition.

    Args:
        record_type: "regex" or "teacher"
        sample_id: originating clean sample ID
        record_id: FinalRecord ID
        context_preview: small snippet of context
    """
    one_line = context_preview.replace("\n", " ").replace("\r", " ")
    log(
        f"[ADDED] ({record_type}) sample={sample_id} record={record_id} "
        f"context='{one_line[:90]}...'"
    )


# ---------------------------------------------------------------------------
# STEP 1 — LOAD CLEAN SAMPLES
# ---------------------------------------------------------------------------

def load_clean_samples() -> List[CleanSample]:
    if not BASE_CLEAN_FILE.exists():
        raise FileNotFoundError(f"Clean base file not found: {BASE_CLEAN_FILE}")
    data = json.loads(BASE_CLEAN_FILE.read_text())
    clean_samples: List[CleanSample] = []
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
    variants: List[MutatedVariant] = []
    mutated_contexts = mutate_context(
        sample.context,
        num=REGEX_VARIANTS_PER_SAMPLE,
    )

    for ctx in mutated_contexts:
        mv = MutatedVariant(
            id=generate_id("regex"),
            parent_id=sample.id,
            mutated_context=ctx,
            mutation_type="regex",
            metadata={"note": "regex-based corruption"},
        )
        mv.validate()
        variants.append(mv)

    log(f"[REGEX] sample={sample.id} generated_variants={len(variants)}")
    return variants


# ---------------------------------------------------------------------------
# STEP 3A — TEACHER-MODEL CORRUPTION+LABELING (BATCH)
# ---------------------------------------------------------------------------

def generate_teacher_mutations(sample: CleanSample) -> List[Dict[str, Any]]:
    """
    Select prompt based on which PII types appear.
    Teacher generates:
      - multiple corrupted variants
      - gold-standard JSON redactions

    Returns a list like:
      [
        {
          "corrupted": "<noisy text>",
          "answer": { "redacted_text": "...", "entities": [...] }
        },
        ...
      ]
    """

    context = sample.context.lower()

    # Choose a PII-type-focused prompt
    if "@" in context or "(at)" in context:
        prompt = email_noise_prompt(sample.context)
        log(f"[TEACHER] sample={sample.id} using=email_noise_prompt")
    elif any(x in context for x in ["-", "(", ")", "+", "call"]):
        prompt = phone_noise_prompt(sample.context)
        log(f"[TEACHER] sample={sample.id} using=phone_noise_prompt")
    elif any(x in context for x in [" st ", "street", " ave", " road", " rd "]):
        prompt = address_noise_prompt(sample.context)
        log(f"[TEACHER] sample={sample.id} using=address_noise_prompt")
    elif any(ch.isdigit() for ch in context) and len(context) > 14:
        prompt = credit_card_noise_prompt(sample.context)
        log(f"[TEACHER] sample={sample.id} using=credit_card_noise_prompt")
    else:
        # fallback: general multi-PII prompt
        prompt = general_noise_prompt(sample.context)
        log(f"[TEACHER] sample={sample.id} using=general_noise_prompt")

    try:
        teacher_outputs = call_teacher_model(prompt)
    except Exception as e:
        log(f"[TEACHER][ERROR] call_teacher_model failed for sample={sample.id}: {e}")
        return []

    # Limit how many teacher variants we keep per sample
    if TEACHER_VARIANTS_PER_SAMPLE and TEACHER_VARIANTS_PER_SAMPLE > 0:
        teacher_outputs = teacher_outputs[:TEACHER_VARIANTS_PER_SAMPLE]

    log(f"[TEACHER] sample={sample.id} teacher_variants={len(teacher_outputs)}")
    return teacher_outputs


# ---------------------------------------------------------------------------
# STEP 4 — PACK FINAL RECORDS
# ---------------------------------------------------------------------------

def create_final_records(
    sample: CleanSample,
    mutated_variants: List[MutatedVariant],
    teacher_variants: List[Dict[str, Any]] = None,
) -> List[FinalRecord]:
    """
    Build FinalRecord objects from:
      - regex-mutated variants (mutated_variants)
      - teacher-generated corrupted variants (teacher_variants)

    Regex variants:
      - If USE_TEACHER_FOR_REGEX_LABELS = True:
          Ask teacher to redact each mutated context, use that as gold answer.
      - Else:
          Fallback to using the clean sample.answer (less accurate but cheaper).

    Teacher variants:
      - Already contain both 'corrupted' text and 'answer' (gold redaction) directly.

    Any record that fails schema validation is logged and skipped.
    """

    records: List[FinalRecord] = []
    sample_id = sample.id

    # -------------------------------------------------------
    # 1) Regex-generated variants
    # -------------------------------------------------------
    for mv in mutated_variants:
        if USE_TEACHER_FOR_REGEX_LABELS:
            try:
                teacher_answer = call_teacher_redact_single(
                    context=mv.mutated_context,
                    question=sample.question,
                )
            except Exception as e:
                log(f"[TEACHER-REDACT][ERROR] regex variant={mv.id} sample={sample_id} error={e}")
                teacher_answer = sample.answer
        else:
            teacher_answer = sample.answer

        # Validate teacher answer before building FinalRecord
        if not is_valid_teacher_answer(teacher_answer):
            log(
                f"[SKIPPED] Invalid teacher answer for regex variant={mv.id} "
                f"sample={sample_id} context='{mv.mutated_context[:80]}'"
            )
            continue

        record = FinalRecord(
            id=generate_id("rec"),
            question=sample.question,
            context=mv.mutated_context,
            answer=teacher_answer,
        )

        try:
            record.validate()
        except Exception as e:
            log(
                f"[SKIPPED] FinalRecord validation failed (regex) "
                f"sample={sample_id} variant={mv.id} error={e}"
            )
            continue

        records.append(record)
        log_record_added("regex", sample_id, record.id, mv.mutated_context)

    # -------------------------------------------------------
    # 2) Teacher-generated corrupted variants
    # -------------------------------------------------------
    if teacher_variants:
        for idx, tv in enumerate(teacher_variants):
            answer = tv.get("answer", {})

            if not is_valid_teacher_answer(answer):
                log(
                    f"[SKIPPED] Invalid teacher variant sample={sample_id} idx={idx} "
                    f"context='{tv.get('corrupted','')[:80]}'"
                )
                continue

            record = FinalRecord(
                id=generate_id("rec"),
                question=sample.question,
                context=tv["corrupted"],
                answer=answer,
            )

            try:
                record.validate()
            except Exception as e:
                log(
                    f"[SKIPPED] FinalRecord failed validation (teacher) "
                    f"sample={sample_id} idx={idx} error={e}"
                )
                continue

            records.append(record)
            log_record_added("teacher", sample_id, record.id, tv["corrupted"])

    return records


# ---------------------------------------------------------------------------
# STEP 5 — DEDUPLICATION
# ---------------------------------------------------------------------------

def dedupe_records(records: List[FinalRecord]) -> List[FinalRecord]:
    seen = set()
    unique: List[FinalRecord] = []
    for r in records:
        key = (r.context, r.answer["redacted_text"])
        if key not in seen:
            unique.append(r)
            seen.add(key)
    log(f"[DEDUP] input_records={len(records)} unique_records={len(unique)}")
    return unique


# ---------------------------------------------------------------------------
# STEP 6 — PIPELINE RUNNER
# ---------------------------------------------------------------------------

def generate_full_dataset() -> List[Dict[str, Any]]:
    clean_samples = load_clean_samples()
    all_records: List[FinalRecord] = []

    for sample in clean_samples:
        log(f"--- Processing sample {sample.id} ---")

        # 1. Regex corruption
        regex_variants = generate_regex_mutations(sample)

        # 2. Teacher corruption (optional, multi-PII style)
        try:
            teacher_variants = generate_teacher_mutations(sample)
        except Exception as e:
            log(f"[WARN] Teacher mutations failed for sample={sample.id}: {e}")
            teacher_variants = []

        # 3. Pack records
        records = create_final_records(sample, regex_variants, teacher_variants)
        log(f"[SUMMARY] sample={sample.id} records_added={len(records)}")

        all_records.extend(records)

    # Deduplicate
    all_records = dedupe_records(all_records)

    # Shuffle
    if SHUFFLE_FINAL_DATASET:
        import random
        random.shuffle(all_records)
        log("[SHUFFLE] Final dataset shuffled")

    final_dicts = [r.__dict__ for r in all_records]
    log(f"[DATASET] total_final_records={len(final_dicts)}")
    return final_dicts


# ---------------------------------------------------------------------------
# EXPORTER
# ---------------------------------------------------------------------------

def export_dataset_jsonl():
    final_records = generate_full_dataset()
    out_path = FINAL_DATASET_DIR / "pii_training_dataset.jsonl"
    write_jsonl(out_path, final_records)
    log(f"Exported {len(final_records)} final records to: {out_path}")

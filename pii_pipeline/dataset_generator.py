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
from dataclasses import dataclass
from pathlib import Path
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:  # pragma: no cover - allow running as script
    from .config import (
        BASE_CLEAN_FILE,
        FINAL_DATASET_DIR,
        RAW_MUTATED_DIR,
        REGEX_VARIANTS_PER_SAMPLE,
        SHUFFLE_FINAL_DATASET,
        TEACHER_GENERATED_DIR,
        TEACHER_VARIANTS_PER_SAMPLE,
    )

    from .schemas import (
        CleanSample,
        FinalRecord,
        MutatedVariant,
        RedactionAnswer,
        RedactionEntity,
        SchemaValidationError,
    )
    from .utils import generate_id, log, write_json, write_jsonl
    from .pii_mutation_engine_v2 import mutate_context
    from .teacher_prompts import (
        general_noise_prompt,
        email_noise_prompt,
        phone_noise_prompt,
        address_noise_prompt,
        credit_card_noise_prompt,
    )
    from .teacher_api import generate_teacher_variants
except ImportError:  # pragma: no cover - executed outside package context
    from config import (  # type: ignore
        BASE_CLEAN_FILE,
        FINAL_DATASET_DIR,
        RAW_MUTATED_DIR,
        REGEX_VARIANTS_PER_SAMPLE,
        SHUFFLE_FINAL_DATASET,
        TEACHER_GENERATED_DIR,
        TEACHER_VARIANTS_PER_SAMPLE,
    )

    from schemas import (  # type: ignore
        CleanSample,
        FinalRecord,
        MutatedVariant,
        RedactionAnswer,
        RedactionEntity,
        SchemaValidationError,
    )
    from utils import generate_id, log, write_json, write_jsonl  # type: ignore
    from pii_mutation_engine_v2 import mutate_context  # type: ignore
    from teacher_prompts import (  # type: ignore
        general_noise_prompt,
        email_noise_prompt,
        phone_noise_prompt,
        address_noise_prompt,
        credit_card_noise_prompt,
    )
    from teacher_api import generate_teacher_variants  # type: ignore


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


def persist_mutations(sample: CleanSample, variants: Sequence[MutatedVariant]) -> None:
    """Write mutated variants for inspection/replay."""

    path = RAW_MUTATED_DIR / f"{sample.id}_regex.json"
    payload = {
        "sample": sample.to_dict(),
        "variants": [v.to_dict() for v in variants],
    }
    write_json(path, payload)


# ---------------------------------------------------------------------------
# STEP 3 — TEACHER-MODEL MUTATIONS
# ---------------------------------------------------------------------------

def generate_teacher_mutations(
    sample: CleanSample,
    *,
    prompt_factory: Optional[Callable[[str], str]] = None,
    prompt_label: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Select prompt based on which PII types appear.
    Teacher generates:
      - multiple corrupted variants
      - gold-standard JSON redactions
    """

    context = sample.context.lower()

    if prompt_factory is not None:
        prompt = prompt_factory(sample.context)
        label = prompt_label or getattr(prompt_factory, "__name__", "custom_prompt")
        log(f"[TEACHER] sample={sample.id} using={label}")
    else:
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
        teacher_outputs = generate_teacher_variants(sample, prompt=prompt)
    except NotImplementedError:
        log("Teacher API not configured; skipping teacher-generated variants.")
        return []

    if TEACHER_VARIANTS_PER_SAMPLE and len(teacher_outputs) > TEACHER_VARIANTS_PER_SAMPLE:
        teacher_outputs = teacher_outputs[:TEACHER_VARIANTS_PER_SAMPLE]

    # Persist prompt/response for auditability
    out_path = TEACHER_GENERATED_DIR / f"{sample.id}_teacher.jsonl"
    records = []
    for idx, variant in enumerate(teacher_outputs):
        payload = {
            "prompt": prompt,
            "variant_index": idx,
            "corrupted": variant.get("corrupted"),
            "answer": variant.get("answer"),
        }
        records.append(payload)
    if records:
        write_jsonl(out_path, records)

    return teacher_outputs


# ---------------------------------------------------------------------------
# STEP 4A — AUTOMATED LABELING FOR REGEX VARIANTS
# ---------------------------------------------------------------------------


def _overlaps(span: Tuple[int, int], others: Sequence[Tuple[int, int]]) -> bool:
    s1, e1 = span
    for s2, e2 in others:
        if s1 < e2 and e1 > s2:
            return True
    return False


def _candidate_spans(
    text: str, target: str, *, max_expand: int = 4, min_ratio: float = 0.55
) -> List[Tuple[float, int, int]]:
    """Return candidate spans sorted by similarity to the target string."""

    text_lower = text.lower()
    target_lower = target.lower()
    base_len = len(target_lower)

    results: List[Tuple[float, int, int]] = []

    # Direct substring matches receive a perfect score
    idx = text_lower.find(target_lower)
    while idx != -1:
        results.append((1.0, idx, idx + base_len))
        idx = text_lower.find(target_lower, idx + 1)

    if base_len == 0:
        return results

    matcher = SequenceMatcher()
    for start in range(len(text_lower)):
        for extra in range(-max_expand, max_expand + 1):
            length = base_len + extra
            if length <= 0:
                continue
            end = start + length
            if end > len(text_lower):
                continue
            window = text_lower[start:end]
            matcher.set_seqs(window, target_lower)
            ratio = matcher.ratio()
            if ratio >= min_ratio:
                results.append((ratio, start, end))

    results.sort(key=lambda item: item[0], reverse=True)
    return results


def _align_entities(
    mutated_context: str, entities: Sequence[RedactionEntity]
) -> Optional[List[Tuple[int, int, RedactionEntity, float]]]:
    matches: List[Tuple[int, int, RedactionEntity, float]] = []
    occupied: List[Tuple[int, int]] = []
    cursor = 0

    for entity in entities:
        target = entity.source_value or entity.value
        candidates = _candidate_spans(mutated_context, target)

        chosen: Optional[Tuple[int, int, float]] = None
        for score, start, end in candidates:
            if start < cursor and score < 0.95:
                # prefer monotonic alignment but allow near-perfect early matches
                continue
            if _overlaps((start, end), occupied):
                continue
            chosen = (start, end, score)
            break

        if chosen is None:
            # As a final fallback, search anywhere disregarding cursor progression
            for score, start, end in candidates:
                if _overlaps((start, end), occupied):
                    continue
                chosen = (start, end, score)
                break

        if chosen is None:
            log(
                f"[ALIGN] Failed to align entity '{entity.value}' in mutated context; dropping variant."
            )
            return None

        start, end, score = chosen
        matches.append((start, end, entity, score))
        occupied.append((start, end))
        cursor = end

    matches.sort(key=lambda item: item[0])
    return matches


def auto_label_variant(sample: CleanSample, variant: MutatedVariant) -> Optional[FinalRecord]:
    """Attempt to produce a labeled record for the mutated variant."""

    aligned = _align_entities(variant.mutated_context, sample.answer.entities)
    if not aligned:
        return None

    pieces: List[str] = []
    cursor = 0
    entities: List[RedactionEntity] = []

    for start, end, original_entity, score in aligned:
        pieces.append(variant.mutated_context[cursor:start])
        pieces.append(original_entity.replacement_token)
        captured_value = variant.mutated_context[start:end]
        cursor = end

        entity_payload = RedactionEntity(
            value=captured_value,
            replacement_token=original_entity.replacement_token,
            reason=original_entity.reason,
            source_value=original_entity.value,
            metadata={
                "match_score": round(score, 3),
                "source": "regex_auto_alignment",
                "mutation_id": variant.id,
            },
        )
        entities.append(entity_payload)

    pieces.append(variant.mutated_context[cursor:])
    redacted_text = "".join(pieces)

    answer = RedactionAnswer(
        redacted_text=redacted_text,
        entities=entities,
        metadata={
            "label_source": "auto_alignment",
            "parent_sample_id": sample.id,
            "mutation_id": variant.id,
        },
    )

    try:
        answer.validate()
    except SchemaValidationError as exc:
        log(f"[ALIGN] Validation failed for variant {variant.id}: {exc}")
        return None

    record = FinalRecord(
        id=generate_id("rec"),
        question=sample.question,
        context=variant.mutated_context,
        answer=answer,
        metadata={
            "source": "regex_mutation",
            "parent_sample_id": sample.id,
            "mutation_type": variant.mutation_type,
        },
    )

    try:
        record.validate()
    except SchemaValidationError as exc:
        log(f"[ALIGN] Final record validation failed for variant {variant.id}: {exc}")
        return None

    return record


# ---------------------------------------------------------------------------
# STEP 4 — PACK FINAL RECORDS
# ---------------------------------------------------------------------------

def create_final_records(
    sample: CleanSample,
    mutated_variants: Sequence[MutatedVariant],
    teacher_variants: Optional[Sequence[Dict[str, Any]]] = None,
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

    """

    records: List[FinalRecord] = []

    for variant in mutated_variants:
        record = auto_label_variant(sample, variant)
        if record:
            records.append(record)

    if teacher_variants:
        for idx, tv in enumerate(teacher_variants):
            answer_payload = tv.get("answer")
            if not answer_payload:
                log(f"[TEACHER] Missing answer in teacher variant #{idx}; skipping")
                continue
            try:
                answer = RedactionAnswer(**answer_payload)
                answer.validate()
            except SchemaValidationError as exc:
                log(f"[TEACHER] Invalid answer payload in variant #{idx}: {exc}")
                continue

            record = FinalRecord(
                id=generate_id("rec"),
                question=sample.question,
                context=tv.get("corrupted", ""),
                answer=answer,
                metadata={
                    "source": "teacher",
                    "parent_sample_id": sample.id,
                    "teacher_variant_index": idx,
                },
            )

            try:
                record.validate()
            except SchemaValidationError as exc:
                log(f"[TEACHER] Final record validation failed: {exc}")
                continue

            records.append(record)
            log(
                "[TEACHER] Added teacher-generated record for "
                f"sample={sample.id} variant_index={idx}"
            )

    return records


# ---------------------------------------------------------------------------
# STEP 5 — DEDUPLICATION
# ---------------------------------------------------------------------------

def dedupe_records(records: Sequence[FinalRecord]) -> List[FinalRecord]:
    seen = set()
    unique: List[FinalRecord] = []
    for r in records:
        key = (r.context, r.answer.redacted_text)
        if key not in seen:
            unique.append(r)
            seen.add(key)
    log(f"[DEDUP] input_records={len(records)} unique_records={len(unique)}")
    return unique


# ---------------------------------------------------------------------------
# STEP 6 — PIPELINE RUNNER
# ---------------------------------------------------------------------------

def generate_full_dataset(max_records: Optional[int] = None) -> List[Dict[str, Any]]:
    """Run the full pipeline and return the resulting records as dictionaries."""

    clean_samples = load_clean_samples()
    sample_scenario_map: Dict[str, PromptScenario] = {}

    if scenario_keys:
        unknown = [key for key in scenario_keys if key not in PROMPT_SCENARIO_MAP]
        if unknown:
            raise ValueError(f"Unknown prompt scenario(s): {', '.join(sorted(unknown))}")

        selected = [PROMPT_SCENARIO_MAP[key] for key in scenario_keys]
        allowed_sample_ids = {
            sample_id for scenario in selected for sample_id in scenario.sample_ids
        }
        sample_scenario_map = {
            sample_id: scenario for scenario in selected for sample_id in scenario.sample_ids
        }

        clean_samples = [
            sample for sample in clean_samples if sample.id in allowed_sample_ids
        ]

        if not clean_samples:
            log(
                "[SCENARIO] No clean samples matched the requested scenarios; returning empty dataset."
            )
            return []

    all_records: List[FinalRecord] = []

    for sample in clean_samples:
        log(f"--- Processing sample {sample.id} ---")

        scenario = sample_scenario_map.get(sample.id)
        if scenario is not None:
            log(
                "[SCENARIO] sample=%s scenario=%s prompt=%s"
                % (sample.id, scenario.key, scenario.prompt_factory.__name__)
            )

        # 1. Regex corruption
        regex_variants = generate_regex_mutations(sample)
        persist_mutations(sample, regex_variants)

        # 2. Teacher corruption (optional)
        teacher_variants = generate_teacher_mutations(
            sample,
            prompt_factory=scenario.prompt_factory if scenario else None,
            prompt_label=scenario.prompt_factory.__name__ if scenario else None,
        )

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

    if max_records is not None:
        if max_records < 0:
            raise ValueError("max_records must be non-negative")
        if max_records == 0:
            return []
        if len(all_records) > max_records:
            all_records = all_records[:max_records]

    return [r.to_dict() for r in all_records]


# ---------------------------------------------------------------------------
# EXPORTER
# ---------------------------------------------------------------------------

def export_dataset_jsonl(max_records: Optional[int] = None):
    final_records = generate_full_dataset(max_records=max_records)
    out_path = FINAL_DATASET_DIR / "pii_training_dataset.jsonl"
    write_jsonl(out_path, final_records)
    log(f"Exported {len(final_records)} final records to: {out_path}")

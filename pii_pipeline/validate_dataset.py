"""
validate_dataset.py
Sanity checker for the generated PII redaction dataset.

Checks:
- JSONL parsable
- Required fields present (question, context, answer.redacted_text, entities)
- Replacement tokens follow policy
- Basic distribution statistics

Usage:
    python validate_dataset.py path/to/dataset.jsonl
"""

import sys
from pathlib import Path
from collections import Counter

try:  # pragma: no cover - allow running as script
    from .utils import read_jsonl, log
    from .schemas import FinalRecord, validate_entities
    from .balance_dataset import infer_pii_type_from_token
except ImportError:  # pragma: no cover - executed outside package context
    from utils import read_jsonl, log  # type: ignore
    from schemas import FinalRecord, validate_entities  # type: ignore
    from balance_dataset import infer_pii_type_from_token  # type: ignore


def validate_dataset(path: Path):
    records = read_jsonl(path)
    log(f"[VALIDATOR] Loaded {len(records)} records from {path}")

    errors = 0
    pii_type_counter = Counter()
    ents_per_record = []

    for idx, rec in enumerate(records):
        try:
            fr = FinalRecord(
                id=rec.get("id", f"rec_{idx}"),
                question=rec["question"],
                context=rec["context"],
                answer=rec["answer"],
            )
            fr.validate()

            entities = rec["answer"].get("entities", [])
            if not validate_entities(entities):
                raise ValueError("Entity validation failed")

            ents_per_record.append(len(entities))

            for ent in entities:
                t = infer_pii_type_from_token(ent["replacement_token"])
                pii_type_counter[t] += 1

        except Exception as e:
            errors += 1
            log(f"[VALIDATOR] Record #{idx} failed: {e}")

    log(f"[VALIDATOR] Validation finished. Errors: {errors}")
    if errors == 0:
        log("[VALIDATOR] All records valid ✅")
    else:
        log("[VALIDATOR] Some records failed. Please inspect the logs above.")

    # Stats
    if ents_per_record:
        avg_ents = sum(ents_per_record) / len(ents_per_record)
        log(f"[VALIDATOR] Avg entities per record: {avg_ents:.2f}")

    log("[VALIDATOR] PII type distribution:")
    for t, c in pii_type_counter.most_common():
        log(f"  {t}: {c}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python validate_dataset.py path/to/dataset.jsonl")
        sys.exit(1)

    dataset_path = Path(sys.argv[1])
    validate_dataset(dataset_path)

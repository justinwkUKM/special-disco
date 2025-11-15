"""
balance_dataset.py
Utility script to balance the dataset by PII type.

- Reads an existing JSONL dataset (final redaction examples).
- Computes distribution by PII type.
- Subsamples to avoid over-represented types.
- Writes a balanced JSONL dataset.

Usage:
    python balance_dataset.py input.jsonl output_balanced.jsonl
"""

import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

try:  # pragma: no cover - allow running as script
    from .config import MAX_VARIANTS_PER_TYPE
except ImportError:  # pragma: no cover - executed outside package context
    from config import MAX_VARIANTS_PER_TYPE  # type: ignore
try:  # pragma: no cover - allow running as script
    from .utils import read_jsonl, write_jsonl, log
except ImportError:  # pragma: no cover - executed outside package context
    from utils import read_jsonl, write_jsonl, log  # type: ignore


def infer_pii_type_from_token(token: str) -> str:
    """
    Map replacement_token back to PII type.
    """
    if token == "[PERSON]":
        return "PERSON"
    if token == "[EMAIL]":
        return "EMAIL"
    if token == "[PHONE]":
        return "PHONE"
    if token == "[ADDRESS]":
        return "ADDRESS"
    if token == "[SSN]":
        return "SSN"
    if token == "[ID]":
        return "ID"
    if token == "[UUID]":
        return "UUID"
    if token.startswith("[CARD_LAST4:"):
        return "CREDIT_CARD"
    if token.startswith("[IBAN_LAST4:"):
        return "IBAN"
    if token == "[GENDER]":
        return "GENDER"
    if token.startswith("[AGE_YEARS:"):
        return "AGE"
    if token == "[RACE]":
        return "RACE"
    if token == "[MARITAL_STATUS]":
        return "MARITAL_STATUS"
    return "UNKNOWN"


def record_pii_types(record: Dict) -> List[str]:
    """Return list of PII types present in this record."""
    types = set()
    for ent in record.get("answer", {}).get("entities", []):
        t = infer_pii_type_from_token(ent["replacement_token"])
        if t != "UNKNOWN":
            types.add(t)
    return list(types)


def balance_dataset(input_path: Path, output_path: Path):
    records = read_jsonl(input_path)
    log(f"[BALANCER] Loaded {len(records)} records from {input_path}")

    # Bucket records by PII type (multi-type records are assigned to all types they touch)
    buckets: Dict[str, List[Dict]] = defaultdict(list)
    for rec in records:
        types = record_pii_types(rec)
        if not types:
            buckets["NO_PII"].append(rec)
        else:
            for t in types:
                buckets[t].append(rec)

    # Log distribution
    log("[BALANCER] Original distribution:")
    for t, lst in buckets.items():
        log(f"  {t}: {len(lst)}")

    # Now we will sample up to MAX_VARIANTS_PER_TYPE per type,
    # but we must avoid writing duplicates multiple times.
    chosen_ids = set()
    balanced_records: List[Dict] = []

    import random
    random.shuffle(records)  # randomize before iterating

    per_type_counts = defaultdict(int)

    for rec in records:
        types = record_pii_types(rec)
        if not types:
            # keep all NO_PII for debugging or drop them if you want to ignore
            continue

        # Check if adding this record would exceed any type's quota
        can_add = True
        for t in types:
            if per_type_counts[t] >= MAX_VARIANTS_PER_TYPE:
                can_add = False
                break

        if not can_add:
            continue

        # Use `id` if present; else hash context+redacted_text
        rec_id = rec.get("id") or (rec.get("context", "") + rec.get("answer", {}).get("redacted_text", ""))
        if rec_id in chosen_ids:
            continue

        balanced_records.append(rec)
        chosen_ids.add(rec_id)
        for t in types:
            per_type_counts[t] += 1

    log("[BALANCER] Balanced distribution:")
    for t, count in per_type_counts.items():
        log(f"  {t}: {count}")

    write_jsonl(output_path, balanced_records)
    log(f"[BALANCER] Wrote {len(balanced_records)} balanced records to {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python balance_dataset.py input.jsonl output_balanced.jsonl")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    balance_dataset(input_path, output_path)

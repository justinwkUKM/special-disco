"""
utils.py
Shared utility functions for the PII dataset pipeline.

Includes:
- Randomness utilities
- File I/O helpers
- Hashing
- Normalization
- ID generation
- JSONL read/write helpers
"""

import json
import hashlib
import random
import string
from pathlib import Path
from typing import Any, Dict, Iterable, List

from config import RNG_SEED


# ---------------------------------------------------------------------------
# RANDOMNESS UTILITIES
# ---------------------------------------------------------------------------

_random = random.Random(RNG_SEED)


def set_seed(seed: int):
    """Reset RNG for reproducibility."""
    global _random
    _random = random.Random(seed)


def rand_bool(p: float = 0.5) -> bool:
    """Return True with probability p."""
    return _random.random() < p


def rand_choice(seq: Iterable[Any]) -> Any:
    """Safe random choice."""
    return _random.choice(list(seq))


def rand_int(a: int, b: int) -> int:
    """Random integer."""
    return _random.randint(a, b)


def rand_string(n: int = 8) -> str:
    """Random alphanumeric string."""
    chars = string.ascii_letters + string.digits
    return "".join(_random.choice(chars) for _ in range(n))


def pick_prob(prob_dict: Dict[Any, float]) -> Any:
    """
    Weighted random selection from dict {item: probability}.
    """
    r = _random.random()
    total = 0
    for k, p in prob_dict.items():
        total += p
        if r <= total:
            return k
    return list(prob_dict.keys())[-1]


# ---------------------------------------------------------------------------
# ID / HASH UTILITIES
# ---------------------------------------------------------------------------

def generate_id(prefix: str = "id") -> str:
    """Generate random unique pipeline ID."""
    return f"{prefix}_{rand_string(12)}"


def hash_text(text: str) -> str:
    """Hash text into short hex string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# NORMALIZATION UTILITIES
# ---------------------------------------------------------------------------

def normalize_spaces(text: str) -> str:
    """Remove abnormal spacing."""
    return " ".join(text.split())


def safe_strip_quotes(text: str) -> str:
    """Remove wrapping quotes if any."""
    if (text.startswith('"') and text.endswith('"')) or \
       (text.startswith("'") and text.endswith("'")):
        return text[1:-1]
    return text


# ---------------------------------------------------------------------------
# JSON / JSONL HELPERS
# ---------------------------------------------------------------------------

def write_json(path: Path, data: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def write_jsonl(path: Path, records: List[Dict[str, Any]]):
    """Write list of dicts to JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, records: List[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read JSONL file into list of dicts."""
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------------------
# PRINT / LOG HELPERS
# ---------------------------------------------------------------------------

def pretty(obj):
    """Pretty-print JSON structure (string)."""
    return json.dumps(obj, indent=2, ensure_ascii=False)


def log(msg: str):
    """Simple pipeline logging."""
    print(f"[PII-PIPELINE] {msg}")


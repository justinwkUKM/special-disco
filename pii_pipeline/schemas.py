"""
schemas.py
Data schemas and validation helpers for the PII Redaction Dataset Pipeline.

Defines:
- CleanSample: raw clean input sample.
- MutatedVariant: mutated/noisy variant (regex/teacher).
- FinalRecord: final training sample (question + context + answer).
- Validation utilities for entities and replacement tokens.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any
import re


# ---------------------------------------------------------------------------
# ENUM DEFINITIONS
# ---------------------------------------------------------------------------

PII_TYPES = {
    "PERSON",
    "EMAIL",
    "PHONE",
    "ADDRESS",
    "SSN",
    "ID",
    "UUID",
    "CREDIT_CARD",
    "IBAN",
    "GENDER",
    "AGE",
    "RACE",
    "MARITAL_STATUS",
}

# Allowed redaction token patterns
REPLACEMENT_PATTERNS = {
    r"\[PERSON\]",
    r"\[EMAIL\]",
    r"\[PHONE\]",
    r"\[ADDRESS\]",
    r"\[SSN\]",
    r"\[ID\]",
    r"\[UUID\]",
    r"\[CARD_LAST4:\d{4}\]",
    r"\[IBAN_LAST4:\d{4}\]",
    r"\[GENDER\]",
    r"\[AGE_YEARS:\d{1,3}\]",
    r"\[RACE\]",
    r"\[MARITAL_STATUS\]",
}


# ---------------------------------------------------------------------------
# DATACLASS STRUCTURES
# ---------------------------------------------------------------------------

@dataclass
class CleanSample:
    """
    Raw clean input sample.

    Fields must match the JSON in clean_samples/base_clean_samples.json:
    {
      "id": "sample_001",
      "question": "...",
      "context": "...",
      "answer": {
          "redacted_text": "...",
          "entities": [...]
      }
    }
    """
    id: str
    question: str
    context: str
    answer: Dict[str, Any]

    def validate(self) -> bool:
        if not isinstance(self.id, str):
            raise ValueError("CleanSample.id must be a string")
        if not isinstance(self.question, str):
            raise ValueError("CleanSample.question must be a string")
        if not isinstance(self.context, str):
            raise ValueError("CleanSample.context must be a string")
        if "redacted_text" not in self.answer:
            raise ValueError("CleanSample.answer missing 'redacted_text'")
        if "entities" not in self.answer:
            raise ValueError("CleanSample.answer missing 'entities'")
        if not isinstance(self.answer["entities"], list):
            raise ValueError("CleanSample.answer.entities must be a list")
        return True


@dataclass
class MutatedVariant:
    """A single mutated/noisy variant derived from a clean sample."""
    id: str
    parent_id: str
    mutated_context: str
    mutation_type: str  # "regex", "teacher", "mixed"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> bool:
        if not isinstance(self.id, str):
            raise ValueError("MutatedVariant.id must be a string")
        if not isinstance(self.parent_id, str):
            raise ValueError("MutatedVariant.parent_id must be a string")
        if not isinstance(self.mutated_context, str):
            raise ValueError("MutatedVariant.mutated_context must be a string")
        if self.mutation_type not in {"regex", "teacher", "mixed"}:
            raise ValueError("MutatedVariant.mutation_type must be one of {'regex','teacher','mixed'}")
        return True


@dataclass
class FinalRecord:
    """
    Final dataset element: model input/output pair.

    Contains:
    - id: unique record id
    - question: the task instruction
    - context: input text for the model
    - answer: gold output with redacted_text + entities[]
    """
    id: str
    question: str
    context: str
    answer: Dict[str, Any]

    def validate(self) -> bool:
        if "redacted_text" not in self.answer:
            raise ValueError("FinalRecord.answer missing 'redacted_text'")
        if "entities" not in self.answer:
            raise ValueError("FinalRecord.answer missing 'entities'")
        if not isinstance(self.answer["entities"], list):
            raise ValueError("FinalRecord.answer.entities must be a list")

        # Validate entities individually
        if not validate_entities(self.answer["entities"]):
            raise ValueError("FinalRecord.answer.entities failed validation")

        return True


# ---------------------------------------------------------------------------
# VALIDATION HELPERS
# ---------------------------------------------------------------------------

def validate_replacement_token(token: str) -> bool:
    """Return True if token matches one of the allowed replacement tokens."""
    return any(re.fullmatch(pattern, token) for pattern in REPLACEMENT_PATTERNS)


def validate_entities(entities: List[Dict[str, Any]]) -> bool:
    """
    Validate structure of entities array:
    Each entity must have {value, replacement_token, reason}
    and replacement_token must match the policy.
    """
    for ent in entities:
        if not {"value", "replacement_token", "reason"} <= set(ent.keys()):
            return False
        if not isinstance(ent["value"], str):
            return False
        if not isinstance(ent["replacement_token"], str):
            return False
        if not isinstance(ent["reason"], str):
            return False
        if not validate_replacement_token(ent["replacement_token"]):
            return False
    return True

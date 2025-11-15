"""Schema objects and validation helpers for the PII dataset pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, List, Optional


class SchemaValidationError(ValueError):
    """Raised when a schema object fails validation."""


def _ensure_type(name: str, value: Any, expected_type: type) -> None:
    if not isinstance(value, expected_type):
        raise SchemaValidationError(f"{name} must be {expected_type.__name__}; got {type(value).__name__}")


def _strip_empty(d: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in d.items() if v not in (None, [], {}, "")}


@dataclass
class RedactionEntity:
    """Represents a single entity that should be redacted from the text."""

    value: str
    replacement_token: str
    reason: str
    source_value: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.metadata, list):  # safeguard for legacy payloads
            self.metadata = {str(idx): item for idx, item in enumerate(self.metadata)}

    def validate(self) -> None:
        _ensure_type("value", self.value, str)
        _ensure_type("replacement_token", self.replacement_token, str)
        _ensure_type("reason", self.reason, str)
        if self.source_value is not None:
            _ensure_type("source_value", self.source_value, str)
        _ensure_type("metadata", self.metadata, dict)
        if not self.value:
            raise SchemaValidationError("Entity value cannot be empty")
        if not self.replacement_token.startswith("[") or not self.replacement_token.endswith("]"):
            raise SchemaValidationError(
                "Replacement tokens must be wrapped in square brackets (e.g. [PERSON])"
            )

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        return _strip_empty(data)


@dataclass
class RedactionAnswer:
    """Answer payload returned by the dataset generator/teacher model."""

    redacted_text: str
    entities: List[RedactionEntity]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalised_entities: List[RedactionEntity] = []
        for ent in self.entities:
            if isinstance(ent, RedactionEntity):
                normalised_entities.append(ent)
            elif isinstance(ent, dict):
                normalised_entities.append(RedactionEntity(**ent))
            else:
                raise SchemaValidationError(
                    "Entities must be RedactionEntity instances or dictionaries"
                )
        self.entities = normalised_entities
        if isinstance(self.metadata, list):
            self.metadata = {str(idx): item for idx, item in enumerate(self.metadata)}

    def validate(self) -> None:
        _ensure_type("redacted_text", self.redacted_text, str)
        _ensure_type("entities", self.entities, list)
        _ensure_type("metadata", self.metadata, dict)
        if not self.redacted_text:
            raise SchemaValidationError("Answer.redacted_text cannot be empty")
        if not self.entities:
            raise SchemaValidationError("Answer must contain at least one entity")
        for ent in self.entities:
            ent.validate()
            if ent.replacement_token not in self.redacted_text:
                raise SchemaValidationError(
                    f"Replacement token {ent.replacement_token} missing from redacted_text"
                )

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "redacted_text": self.redacted_text,
            "entities": [ent.to_dict() for ent in self.entities],
        }
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload


@dataclass
class CleanSample:
    """Represents a pristine training sample prior to mutation."""

    id: str
    question: str
    context: str
    answer: RedactionAnswer
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.answer, dict):
            self.answer = RedactionAnswer(**self.answer)
        if isinstance(self.metadata, list):
            self.metadata = {str(idx): item for idx, item in enumerate(self.metadata)}

    def validate(self) -> None:
        for field_name in ("id", "question", "context"):
            _ensure_type(field_name, getattr(self, field_name), str)
            if not getattr(self, field_name):
                raise SchemaValidationError(f"{field_name} cannot be empty")
        _ensure_type("tags", self.tags, list)
        _ensure_type("metadata", self.metadata, dict)
        self.answer.validate()

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "id": self.id,
            "question": self.question,
            "context": self.context,
            "answer": self.answer.to_dict(),
        }
        if self.tags:
            payload["tags"] = self.tags
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload


@dataclass
class MutatedVariant:
    """Intermediate mutated context waiting for labeling."""

    id: str
    parent_id: str
    mutated_context: str
    mutation_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.metadata, list):
            self.metadata = {str(idx): item for idx, item in enumerate(self.metadata)}

    def validate(self) -> None:
        for field_name in ("id", "parent_id", "mutated_context", "mutation_type"):
            _ensure_type(field_name, getattr(self, field_name), str)
            if not getattr(self, field_name):
                raise SchemaValidationError(f"{field_name} cannot be empty")
        _ensure_type("metadata", self.metadata, dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "id": self.id,
            "parent_id": self.parent_id,
            "mutated_context": self.mutated_context,
            "mutation_type": self.mutation_type,
        }
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload


@dataclass
class FinalRecord:
    """Fully labeled record ready for training."""

    id: str
    question: str
    context: str
    answer: RedactionAnswer
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.answer, dict):
            self.answer = RedactionAnswer(**self.answer)
        if isinstance(self.metadata, list):
            self.metadata = {str(idx): item for idx, item in enumerate(self.metadata)}

    def validate(self) -> None:
        for field_name in ("id", "question", "context"):
            _ensure_type(field_name, getattr(self, field_name), str)
            if not getattr(self, field_name):
                raise SchemaValidationError(f"{field_name} cannot be empty")
        _ensure_type("metadata", self.metadata, dict)
        self.answer.validate()

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "id": self.id,
            "question": self.question,
            "context": self.context,
            "answer": self.answer.to_dict(),
        }
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload


def validate_entities(entities: Iterable[Dict[str, Any]]) -> bool:
    """Lightweight validator for entity dictionaries."""

    try:
        for ent in entities:
            RedactionEntity(**ent).validate()
        return True
    except SchemaValidationError:
        return False


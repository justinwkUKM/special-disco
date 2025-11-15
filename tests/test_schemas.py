import pytest

from pii_pipeline.schemas import (
    RedactionEntity,
    RedactionAnswer,
    CleanSample,
    SchemaValidationError,
)


def test_redaction_entity_validation_success():
    entity = RedactionEntity(
        value="John Doe",
        replacement_token="[PERSON]",
        reason="full_name",
        metadata={"source": "clean_sample"},
    )

    # Should not raise
    entity.validate()


def test_redaction_entity_requires_bracketed_token():
    entity = RedactionEntity(
        value="John Doe",
        replacement_token="PERSON",
        reason="full_name",
    )

    with pytest.raises(SchemaValidationError):
        entity.validate()


def test_clean_sample_normalises_answer_and_validates_entities():
    sample = CleanSample(
        id="sample-123",
        question="What PII should be redacted?",
        context="John Doe works at Initech in Austin.",
        answer={
            "redacted_text": "[PERSON] works at [ORG] in [LOCATION]",
            "entities": [
                {
                    "value": "John Doe",
                    "replacement_token": "[PERSON]",
                    "reason": "full_name",
                },
                {
                    "value": "Initech",
                    "replacement_token": "[ORG]",
                    "reason": "employer",
                },
                {
                    "value": "Austin",
                    "replacement_token": "[LOCATION]",
                    "reason": "city",
                },
            ],
        },
        metadata=["legacy"],
    )

    sample.validate()
    assert isinstance(sample.answer, RedactionAnswer)
    assert sample.answer.entities[0].value == "John Doe"
    # Metadata list should be normalised into a dictionary via __post_init__
    assert sample.metadata == {"0": "legacy"}

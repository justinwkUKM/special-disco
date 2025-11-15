import pytest

from pii_pipeline import dataset_generator as dg
from pii_pipeline.schemas import (
    CleanSample,
    RedactionAnswer,
    RedactionEntity,
    MutatedVariant,
)
from pii_pipeline.utils import set_seed


@pytest.fixture(autouse=True)
def reset_seed():
    set_seed(42)
    yield
    set_seed(42)


def make_clean_sample() -> CleanSample:
    return CleanSample(
        id="clean-1",
        question="Which spans require redaction?",
        context="John Doe works at Initech in Austin.",
        answer=RedactionAnswer(
            redacted_text="[PERSON] works at [ORG] in [LOCATION]",
            entities=[
                RedactionEntity(
                    value="John Doe",
                    replacement_token="[PERSON]",
                    reason="full_name",
                ),
                RedactionEntity(
                    value="Initech",
                    replacement_token="[ORG]",
                    reason="employer",
                ),
                RedactionEntity(
                    value="Austin",
                    replacement_token="[LOCATION]",
                    reason="city",
                ),
            ],
        ),
    )


def test_candidate_spans_prioritises_exact_match():
    candidates = dg._candidate_spans("Jane Doe", "Jane")
    assert candidates[0][0] == pytest.approx(1.0)
    assert candidates[0][1:] == (0, 4)


def test_align_entities_handles_minor_mutations():
    sample = make_clean_sample()
    variant = MutatedVariant(
        id="mut-1",
        parent_id=sample.id,
        mutated_context="Jonny Doe now works at Initech Corp in Austin, Texas.",
        mutation_type="regex",
    )

    aligned = dg._align_entities(variant.mutated_context, sample.answer.entities)
    assert aligned is not None
    # Ensure that all entities aligned and order preserved
    assert [ent.value for _, _, ent, _ in aligned] == [
        "John Doe",
        "Initech",
        "Austin",
    ]


def test_auto_label_variant_builds_final_record():
    sample = make_clean_sample()
    variant = MutatedVariant(
        id="mut-2",
        parent_id=sample.id,
        mutated_context="J0hn Doe works at Initech HQ in Austin, TX.",
        mutation_type="regex",
    )

    record = dg.auto_label_variant(sample, variant)
    assert record is not None
    assert record.context == variant.mutated_context
    assert record.answer.redacted_text.count("[PERSON]") == 1
    assert record.answer.redacted_text.count("[ORG]") == 1
    assert record.answer.entities[0].value == "J0hn Doe"
    assert record.answer.entities[0].metadata["source"] == "regex_auto_alignment"


def test_auto_label_variant_returns_none_when_alignment_fails():
    sample = make_clean_sample()
    variant = MutatedVariant(
        id="mut-3",
        parent_id=sample.id,
        mutated_context="No personal data present here.",
        mutation_type="regex",
    )

    record = dg.auto_label_variant(sample, variant)
    assert record is None

"""Integration layer between the pipeline and the teacher LLM.

This module keeps the high-level API stable while allowing downstream
deployers to plug in their own LLM client.  The default implementation raises
``NotImplementedError`` so that callers can gracefully skip teacher generated
variants when no client is configured.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

try:  # pragma: no cover - allow running as top-level module
    from .schemas import CleanSample
    from .teacher_prompts import general_noise_prompt
    from .utils import log
except ImportError:  # pragma: no cover - executed when package context missing
    from schemas import CleanSample  # type: ignore
    from teacher_prompts import general_noise_prompt  # type: ignore
    from utils import log  # type: ignore


def call_teacher_raw(prompt: str) -> str:
    """Low-level hook for talking to the teacher model.

    Projects integrating with a real LLM should override this function with
    the code that performs an API request and returns the raw textual response
    from the model.  The base implementation intentionally raises
    ``NotImplementedError`` so the rest of the pipeline can detect that the
    integration has not been configured yet.
    """

    raise NotImplementedError("Implement call_teacher_raw() with your LLM client.")


def parse_teacher_output(raw: str) -> List[Dict[str, Any]]:
    """Parse the teacher model output into structured variants.

    The pipeline expects the teacher to respond with a JSON list where each
    element looks like ``{"corrupted": str, "answer": {...}}``.  The parser is
    intentionally strict so that calling code can surface useful errors if the
    response format drifts.
    """

    raw = raw.strip()

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:  # pragma: no cover - integration hook
        raise ValueError("Teacher output was not valid JSON.") from exc

    if not isinstance(payload, list):
        raise ValueError("Teacher output must be a JSON list of variant objects.")

    for idx, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"Teacher output element {idx} is not an object.")
        if "corrupted" not in item or "answer" not in item:
            raise ValueError(
                f"Teacher output element {idx} is missing 'corrupted' or 'answer'."
            )

    return payload


def generate_teacher_variants(
    clean_sample: CleanSample, prompt: str | None = None
) -> List[Dict[str, Any]]:
    """Generate teacher-model variants for a given clean sample.

    When the teacher integration is not configured the function raises
    ``NotImplementedError``.  ``dataset_generator.generate_teacher_mutations``
    catches this and simply skips teacher augmentation so the rest of the
    pipeline continues to operate.
    """

    ctx = clean_sample.context
    log(f"[TEACHER] Generating variants for sample {clean_sample.id}")

    prompt = prompt or general_noise_prompt(ctx)
    raw = call_teacher_raw(prompt)
    return parse_teacher_output(raw)


def call_teacher_redact_single(context: str, question: str) -> Dict[str, Any]:
    """Request a single redaction response from the teacher model.

    This helper mirrors the legacy API that the CLI test harness expects.  Like
    :func:`call_teacher_raw`, the default implementation raises
    ``NotImplementedError`` so deployers can decide how to integrate their LLM
    provider.
    """

    raise NotImplementedError(
        "Implement call_teacher_redact_single() with your LLM client."
    )


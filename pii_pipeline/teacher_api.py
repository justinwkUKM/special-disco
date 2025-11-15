"""
teacher_api.py
Integration layer between the pipeline and the teacher LLM (e.g., OpenAI models).

Exposes:
- call_teacher_model(prompt) -> List[{"corrupted": str, "answer": {...}}]
- call_teacher_redact_single(context, question) -> {"redacted_text": str, "entities": [...]}

Both functions assume the model responds with pure JSON.
"""

import os
import json
from typing import List, Dict, Any

from teacher_prompts import general_noise_prompt
from schemas import CleanSample, RedactionAnswer, SchemaValidationError
from utils import log

# Configure client from env var
# export OPENAI_API_KEY="sk-..."
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# You can override this in env: export TEACHER_MODEL_NAME="gpt-4.1"
from config import (
    TEACHER_MODEL_NAME
)

def call_teacher_model(prompt: str) -> List[Dict[str, Any]]:
    """
    Calls the teacher model to generate multiple corrupted variants + gold redactions.

    Expected JSON response from the model:
    [
      {
        "corrupted": "<noisy text>",
        "answer": {
          "redacted_text": "...",
          "entities": [
            {"value": "...", "replacement_token": "...", "reason": "..."}
          ]
        }
      },
      ...
    ]
    """

    system_message = (
        "You are an expert synthetic PII corruption generator and redaction teacher.\n"
        "You MUST respond with a VALID JSON ARRAY only, no extra text.\n"
        "Each element must be an object with keys 'corrupted' and 'answer'.\n"
        "'answer' must contain 'redacted_text' and 'entities', where 'entities' is a list "
        "of {value, replacement_token, reason}."
    )

    completion = client.chat.completions.create(
        model=TEACHER_MODEL_NAME,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ],
        temperature=0.4,
        max_tokens=2048,
    )

    raw = completion.choices[0].message.content.strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        print("[TEACHER_API] Invalid JSON from call_teacher_model:")
        print(raw)
        raise e

    if not isinstance(parsed, list):
        raise ValueError("Teacher model must return a JSON list (array).")

    for i, item in enumerate(parsed):
        if not isinstance(item, dict):
            raise ValueError(f"Teacher output element {i} is not an object.")
        if "corrupted" not in item or "answer" not in item:
            raise ValueError(f"Teacher output element {i} missing 'corrupted' or 'answer'.")

def generate_teacher_variants(clean_sample: CleanSample, prompt: str | None = None) -> List[Dict]:
    """
    Main high-level function:
    - Builds a prompt for the teacher model based on the clean sample.
    - Calls teacher model.
    - Parses response into structured corrupted+answer list.
    """


    # You can choose between different prompts.
    # For now, use the general multi-PII noise prompt.
    prompt = prompt or general_noise_prompt(ctx)
    # or for very hard examples you could switch to teacher_prompts.multi_pii_super_prompt(ctx)

    Expected JSON response from the model:
    {
      "redacted_text": "...",
      "entities": [
        {"value": "...", "replacement_token": "...", "reason": "..."},
        ...
      ]
    }
    """

    normalised = []
    for idx, item in enumerate(variants):
        answer_payload = item.get("answer")
        if not answer_payload:
            raise SchemaValidationError(
                f"Teacher variant #{idx} is missing an answer payload"
            )
        answer = RedactionAnswer(**answer_payload)
        answer.validate()
        normalised.append({"corrupted": item.get("corrupted", ""), "answer": answer.to_dict()})

    return normalised

    try:
        answer = json.loads(raw)
    except json.JSONDecodeError as e:
        print("[TEACHER_API] Invalid JSON from call_teacher_redact_single:")
        print(raw)
        raise e

    if not isinstance(answer, dict):
        raise ValueError("Teacher redact_single must return a JSON object.")
    if "redacted_text" not in answer or "entities" not in answer:
        raise ValueError("Teacher redact_single must return 'redacted_text' and 'entities'.")

    return answer

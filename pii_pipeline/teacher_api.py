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

from openai import OpenAI

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

    return parsed


def call_teacher_redact_single(context: str, question: str) -> Dict[str, Any]:
    """
    Ask the teacher model to redact a single piece of text.

    Expected JSON response from the model:
    {
      "redacted_text": "...",
      "entities": [
        {"value": "...", "replacement_token": "...", "reason": "..."},
        ...
      ]
    }
    """

    system_message = (
        "You are an expert PII redaction model.\n"
        "Given a text, you must return a JSON object with exactly:\n"
        '{ "redacted_text": "...", "entities": [ ... ] }\n\n'
        "Where entities is a list of objects with keys: value, replacement_token, reason.\n"
        "Do NOT include any explanation or text outside the JSON object."
    )

    user_message = (
        f"Question: {question}\n\n"
        "Text to redact:\n"
        f"\"\"\"{context}\"\"\"\n\n"
        "Return ONLY a JSON object with keys 'redacted_text' and 'entities'."
    )

    completion = client.chat.completions.create(
        model=TEACHER_MODEL_NAME,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
        temperature=0.2,
        max_tokens=1024,
    )

    raw = completion.choices[0].message.content.strip()

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

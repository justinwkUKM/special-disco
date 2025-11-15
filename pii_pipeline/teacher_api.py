"""
teacher_api.py
Integration layer between the pipeline and the teacher LLM (e.g., ChatGPT 5.1).

This module exposes a single main function:

    generate_teacher_variants(clean_sample: CleanSample) -> List[Dict]

Each returned dict MUST look like:
{
  "corrupted": "<noisy text>",
  "answer": {
      "redacted_text": "...",
      "entities": [
          {"value": "...", "replacement_token": "...", "reason": "..."},
          ...
      ]
  }
}

You will need to:
- Plug in your OpenAI / other LLM client.
- Ensure the model follows the redaction schema.
"""

from typing import List, Dict

from teacher_prompts import general_noise_prompt
from schemas import CleanSample, RedactionAnswer, SchemaValidationError
from utils import log

# If using OpenAI, you would do something like:
# import openai
# from config import TEACHER_MODEL_NAME, TEMPERATURE, TOP_P, MAX_TOKENS


def call_teacher_raw(prompt: str) -> str:
    """
    Low-level call to the teacher model.

    Returns the raw text response from the LLM.
    You MUST implement this for your environment.
    """
    # Example (pseudo-code):
    #
    # completion = openai.ChatCompletion.create(
    #     model=TEACHER_MODEL_NAME,
    #     messages=[
    #         {"role": "system", "content": SYSTEM_PROMPT},
    #         {"role": "user", "content": prompt}
    #     ],
    #     temperature=TEMPERATURE,
    #     top_p=TOP_P,
    #     max_tokens=MAX_TOKENS,
    # )
    # return completion.choices[0].message["content"]
    #
    raise NotImplementedError("Implement call_teacher_raw() with your LLM client.")


def parse_teacher_output(raw: str) -> List[Dict]:
    """
    Parse the teacher model output into a structured list.

    Expectation:
    - The teacher returns either:
      * a JSON list, OR
      * a block containing multiple JSON objects.

    You can adapt this parser based on your chosen output format.
    """
    import json

    raw = raw.strip()

    # Simplest assumption: teacher returns a JSON list
    try:
        data = json.loads(raw)
        # expected: [{"corrupted": "...", "answer": {...}}, ...]
        if isinstance(data, list):
            return data
    except Exception:
        pass

    # If you want to support more flexible formats, you can add regex parsing here.
    raise ValueError("Teacher output format not recognized; please adjust parse_teacher_output().")


def generate_teacher_variants(clean_sample: CleanSample, prompt: str | None = None) -> List[Dict]:
    """
    Main high-level function:
    - Builds a prompt for the teacher model based on the clean sample.
    - Calls teacher model.
    - Parses response into structured corrupted+answer list.
    """

    ctx = clean_sample.context
    log(f"[TEACHER] Generating variants for sample {clean_sample.id}")

    # You can choose between different prompts.
    # For now, use the general multi-PII noise prompt.
    prompt = prompt or general_noise_prompt(ctx)
    # or for very hard examples you could switch to teacher_prompts.multi_pii_super_prompt(ctx)

    raw = call_teacher_raw(prompt)
    variants = parse_teacher_output(raw)

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


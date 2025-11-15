"""
Quick validation test for teacher-model redaction.

Purpose:
--------
Call the teacher model on a known input and print the result so you can verify:

- JSON structure
- answer["redacted_text"]
- answer["entities"] validation
- replacement_token formatting

Run:
    python pii_pipeline/tests/test_teacher_single.py
"""

import sys
from pathlib import Path

# Ensure our project root resolves correctly
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from teacher_api import call_teacher_redact_single
from teacher_prompts import general_noise_prompt


def main():
    sample_context = "Hi, I'm John Smith. My brother is Bob. Reach me at john.smith@example.com."
    question = (
        "Redact provided text according to the task description "
        "and return redacted elements."
    )

    print("\n=== Running Teacher Single-Redaction Test ===\n")
    print("Input context:")
    print("   ", sample_context)
    print("\nCalling teacher model...\n")

    try:
        answer = call_teacher_redact_single(sample_context, question)
    except Exception as e:
        print("[ERROR] Teacher API call failed!")
        print(e)
        return

    print("=== RAW TEACHER ANSWER ===")
    print(answer)
    print("\n=== END ===")


if __name__ == "__main__":
    main()

"""
teacher_prompts.py
Teacher-model prompt templates for generating corrupted PII variants
AND gold-standard redaction answers.

This module provides:
- PII-type specific corruption prompts
- Mixed-type corruption prompts
- Full instruction-pack for a strong teacher model: ChatGPT 5.1

Dataset generator uses these templates to call the teacher model.
"""

import textwrap


# ---------------------------------------------------------------------------
# BASE SYSTEM PROMPT (for ChatGPT 5.1)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """
You are an elite PII-redaction expert. 
You generate:
1. Highly realistic noisy/corrupted versions of input text.
2. A correct gold-standard redaction (JSON) following the exact schema:
   {
     "redacted_text": "...",
     "entities": [
         {"value": "...", "replacement_token": "...", "reason": "..."},
         ...
     ]
   }

You ALWAYS:
- Preserve the meaning of the original text.
- Maintain all PII entities (do not remove them).
- Apply realistic corruption or noise (OCR errors, spacing shifts, typos, symbol drift, etc.).
- Produce EXACT matching redaction tokens based on the redaction policy.
"""

# ---------------------------------------------------------------------------
# GENERIC MULTI-PII NOISE PROMPT
# ---------------------------------------------------------------------------

def general_noise_prompt(clean_context: str) -> str:
    """
    Teacher instructs ChatGPT to:
    - Create several corrupted/noisy variants.
    - Provide gold-standard redactions for each.

    The dataset generator selects 1 or more.
    """
    return textwrap.dedent(f"""
    Generate 3 highly realistic corrupted versions of the following text:

    ORIGINAL:
    \"{clean_context}\"

    Corruptions MUST include:
    - OCR distortions
    - character substitutions
    - spacing anomalies
    - punctuation drift
    - partial obfuscation
    - human-typo patterns

    After generating the corrupted versions, for EACH corrupted version,
    produce a CORRECT redaction JSON matching the EXACT schema below:

    {{
      "redacted_text": "...",
      "entities": [
         {{"value": "...", "replacement_token": "...", "reason": "..."}}
      ]
    }}

    IMPORTANT RULES:
    - Never remove PII from the corrupted variant.
    - Do not correct the corrupted text; redaction must match corrupted text exactly.
    - Detected PII values MUST come from the corrupted text, not the clean version.
    """).strip()


# ---------------------------------------------------------------------------
# PII-TYPE SPECIFIC NOISE PROMPTS
# ---------------------------------------------------------------------------

def email_noise_prompt(email_context: str) -> str:
    return textwrap.dedent(f"""
    Generate 3 corrupted/noisy representations of the EMAIL found in the text.
    Make the email corruption realistic:
    - replace '@' with '(at)' or '[at]'
    - insert spaces in the domain
    - break the TLD
    - OCR errors like 'gmai1' or 'h0tma1l'

    ORIGINAL:
    \"{email_context}\"

    Then return a correct JSON redaction for each variant.
    """).strip()


def phone_noise_prompt(phone_context: str) -> str:
    return textwrap.dedent(f"""
    Generate 3 corrupted/noisy PHONE numbers inside the text:
    - Swap digits
    - Insert unicode dashes
    - Keyboard-neighbor digits
    - Remove some separators or add extra ones

    ORIGINAL:
    \"{phone_context}\"

    Then provide the redaction JSON for each variant.
    """).strip()


def address_noise_prompt(address_context: str) -> str:
    return textwrap.dedent(f"""
    Generate 3 corrupted/noisy street address variants inside the text:
    - OCR number confusion (8↔B, 1↔l)
    - Abbreviations (St, Str, Strt)
    - Random spacing or missing characters

    ORIGINAL:
    \"{address_context}\"

    Then produce valid JSON redaction for each variant.
    """).strip()


def credit_card_noise_prompt(card_context: str) -> str:
    return textwrap.dedent(f"""
    Create 3 corrupted credit card number formats:
    - spacing noise
    - unicode digits
    - mixed grouping
    - missing separator or extra separator

    ORIGINAL:
    \"{card_context}\"

    Then generate the correct redaction JSON.
    """).strip()


def gender_race_age_noise_prompt(context: str) -> str:
    return textwrap.dedent(f"""
    Generate 3 noisy human-identity description variants:
    - Misspelled gender terms
    - Age formatting changes (29 y/o, 29yo, age:29)
    - Race descriptors with OCR drift

    ORIGINAL:
    \"{context}\"

    Then produce the correct redaction JSON.
    """).strip()


# ---------------------------------------------------------------------------
# COMPLEX MULTI-PII COMPOSITE PROMPT
# ---------------------------------------------------------------------------

def multi_pii_super_prompt(context: str) -> str:
    return textwrap.dedent(f"""
    Generate 3 extremely hard corrupted variants of this text involving MULTIPLE PII types:

    \"{context}\"

    Required distortions:
    - Character-level corruption
    - Multi-PII interaction (email + address + gender + UUID + CC, etc.)
    - Abnormal spacing and punctuation
    - Partial obfuscation of identifiers
    - OCR-like patterns
    - Unicode homoglyph injections

    After generating the variants, return a correct redaction JSON for each.

    Redaction JSON MUST FOLLOW this structure EXACTLY:
    {{
      "redacted_text": "...",
      "entities": [
         {{"value": "...", "replacement_token": "...", "reason": "..."}}
      ]
    }}
    """).strip()

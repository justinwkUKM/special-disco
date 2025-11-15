"""
teacher_prompts.py
High-quality prompt templates for the teacher LLM (e.g., GPT-4.1 / GPT-5.1).

These prompts are used to:
- Generate corrupted/noisy variants of PII-containing text.
- Produce gold-standard redaction answers in your strict JSON schema.

All prompts are designed so the teacher can be called via:
  - call_teacher_model(prompt)          -> List[{"corrupted": str, "answer": {...}}]
  - call_teacher_redact_single(...)     -> handled separately in teacher_api.py

IMPORTANT: These prompts assume the teacher will return ONLY JSON, with:
[
  {
    "corrupted": "<noisy text>",
    "answer": {
      "redacted_text": "<string>",
      "entities": [
        {
          "value": "<string from corrupted text>",
          "replacement_token": "<one of the allowed tokens>",
          "reason": "<brief explanation>"
        }
      ]
    }
  },
  ...
]

Allowed replacement_token values:
- "[PERSON]"
- "[EMAIL]"
- "[PHONE]"
- "[ADDRESS]"
- "[SSN]"
- "[ID]"
- "[UUID]"
- "[CARD_LAST4:dddd]"   (e.g. [CARD_LAST4:1234])
- "[IBAN_LAST4:dddd]"   (e.g. [IBAN_LAST4:9876])
- "[GENDER]"
- "[AGE_YEARS:nnn]"     (e.g. [AGE_YEARS:29])
- "[RACE]"
- "[MARITAL_STATUS]"
"""

import textwrap


# ---------------------------------------------------------------------------
# SHARED POLICY SNIPPET
# ---------------------------------------------------------------------------

BASE_POLICY = """
You MUST follow this redaction policy:

1. Output JSON only — no explanation, no prose, no markdown.
2. For each example, you MUST detect all PII in the CORRUPTED text, not the original.
3. Your JSON must be a list (array) of objects like:
   [
     {
       "corrupted": "string",
       "answer": {
         "redacted_text": "string",
         "entities": [
           {"value": "string", "replacement_token": "string", "reason": "string"}
         ]
       }
     }
   ]

4. 'answer.redacted_text' must be the CORRUPTED text with PII replaced by tokens:
   [PERSON], [EMAIL], [PHONE], [ADDRESS], [SSN], [ID], [UUID],
   [CARD_LAST4:dddd], [IBAN_LAST4:dddd],
   [GENDER], [AGE_YEARS:nnn], [RACE], [MARITAL_STATUS].

5. 'entities' is a list of objects with:
   - "value": exact substring as it appears in the CORRUPTED text.
   - "replacement_token": exactly one of the allowed tokens above.
   - "reason": short explanation, e.g. "person name", "email address", "phone number".

6. For credit cards:
   - replacement_token MUST be "[CARD_LAST4:dddd]" where dddd are the last 4 digits.
7. For IBAN / bank accounts:
   - replacement_token MUST be "[IBAN_LAST4:dddd]" where dddd are the last 4 digits.
8. For ages:
   - replacement_token MUST be "[AGE_YEARS:nnn]" with the numeric age you extracted.
9. Do NOT introduce extra fields. Do NOT use token names without brackets.
10. If there is no PII, 'entities' can be an empty list.
""".strip()


# ---------------------------------------------------------------------------
# GENERIC MULTI-PII NOISE PROMPT
# ---------------------------------------------------------------------------

def general_noise_prompt(clean_context: str, num_variants: int = 4) -> str:
    """
    Generic prompt to generate multiple corrupted variants for any PII mix.
    """

    return textwrap.dedent(f"""
    You are generating synthetic training data for a PII redaction model.

    Your task:
    1. Take the ORIGINAL text.
    2. Generate {num_variants} DIFFERENT corrupted/noisy variants.
       - Use realistic noise: typos, spacing issues, OCR-like distortions,
         symbol replacements, obfuscated emails/phones, etc.
       - KEEP the same semantic PII, but in messy forms.
    3. For EACH corrupted variant, produce a correct redaction answer
       according to the policy below.

    ORIGINAL TEXT:
    \"\"\"{clean_context}\"\"\"

    {BASE_POLICY}

    Additional corruption guidelines:
    - Introduce character swaps (e.g., 0/O, 1/l, 5/S).
    - Randomly add or remove spaces.
    - Use variations of punctuation.
    - Partially obfuscate emails ("john (at) example (dot) com").
    - Distort phone formats ("+1-202-555-01 98", "☎ 202 555 0198").
    - Keep sentences grammatically plausible, but not perfect.

    Return ONLY a JSON array as described in the policy. No extra keys.
    """).strip()


# ---------------------------------------------------------------------------
# EMAIL-SPECIFIC PROMPT
# ---------------------------------------------------------------------------

def email_noise_prompt(clean_context: str, num_variants: int = 4) -> str:
    """
    Prompt specialized for EMAIL-containing texts.
    """

    return textwrap.dedent(f"""
    You are generating synthetic EMAIL corruption examples for PII redaction training.

    ORIGINAL TEXT:
    \"\"\"{clean_context}\"\"\"

    Your task:
    1. Create {num_variants} corrupted variants of this text focusing on EMAIL obfuscation:
       - Replace '@' with '(at)', '[at]', ' at '.
       - Replace '.' with '(dot)', '[dot]', ' dot '.
       - Insert extra spaces inside the email.
       - Use common OCR errors: 'gmai1.com', 'hotmaiI.com', 'out1ook.com'.
       - Mix case and symbols: 'JOHN.smith+news@example-mail.com'.
    2. For EACH corrupted variant, produce the redaction answer JSON.

    {BASE_POLICY}

    Additional EMAIL rules:
    - Detect ALL email addresses in the corrupted text.
    - Use [EMAIL] as the replacement_token for any email in 'entities'.
    - In 'redacted_text', every email span MUST be replaced by [EMAIL].

    Return ONLY a JSON array as described in the policy.
    """).strip()


# ---------------------------------------------------------------------------
# PHONE-SPECIFIC PROMPT
# ---------------------------------------------------------------------------

def phone_noise_prompt(clean_context: str, num_variants: int = 4) -> str:
    """
    Prompt specialized for PHONE-number-containing texts.
    """

    return textwrap.dedent(f"""
    You are generating synthetic PHONE corruption examples for PII redaction training.

    ORIGINAL TEXT:
    \"\"\"{clean_context}\"\"\"

    Your task:
    1. Create {num_variants} corrupted variants of this text focusing on PHONE numbers:
       - Vary the formatting: "+1 202 555 0198", "+1-202-555-0198", "(202) 555-0198".
       - Use different separators: spaces, hyphens, periods, unicode dashes (–, —).
       - Add or remove country codes.
       - Replace some digits with similar digits (typo-style).
       - Insert emoji or symbols around the phone: "📞 +1 202 555 0198".
    2. For EACH corrupted variant, produce the redaction answer JSON.

    {BASE_POLICY}

    Additional PHONE rules:
    - Any phone number (international or national) is [PHONE].
    - Preserve non-phone numbers if they are not clearly phone-like.
    - In 'redacted_text', every phone span MUST be replaced by [PHONE].

    Return ONLY a JSON array as described in the policy.
    """).strip()


# ---------------------------------------------------------------------------
# ADDRESS-SPECIFIC PROMPT
# ---------------------------------------------------------------------------

def address_noise_prompt(clean_context: str, num_variants: int = 4) -> str:
    """
    Prompt specialized for ADDRESS-containing texts.
    """

    return textwrap.dedent(f"""
    You are generating synthetic ADDRESS corruption examples for PII redaction training.

    ORIGINAL TEXT:
    \"\"\"{clean_context}\"\"\"

    Your task:
    1. Create {num_variants} corrupted variants focusing on street addresses:
       - Abbreviate words: "Street" -> "St.", "Road" -> "Rd", "Avenue" -> "Ave".
       - Introduce OCR errors in street names and numbers (1 ↔ l, 0 ↔ O, 8 ↔ B).
       - Shuffle commas and spacing, e.g. "221B Baker St,London".
       - Add or remove apartment/unit identifiers ("Apt 4B", "#4B", "Unit 4B").
    2. For EACH corrupted variant, produce the redaction answer JSON.

    {BASE_POLICY}

    Additional ADDRESS rules:
    - Only redact full street+number style address spans as [ADDRESS].
    - City/region/country names alone should NOT be [ADDRESS] unless part of a full address.
    - In 'redacted_text', every full address span MUST be replaced by [ADDRESS].

    Return ONLY a JSON array as described in the policy.
    """).strip()


# ---------------------------------------------------------------------------
# CREDIT-CARD-SPECIFIC PROMPT
# ---------------------------------------------------------------------------

def credit_card_noise_prompt(clean_context: str, num_variants: int = 4) -> str:
    """
    Prompt specialized for CREDIT CARD number texts.
    """

    return textwrap.dedent(f"""
    You are generating synthetic CREDIT CARD corruption examples for PII redaction training.

    ORIGINAL TEXT:
    \"\"\"{clean_context}\"\"\"

    Your task:
    1. Create {num_variants} corrupted variants focusing on CREDIT CARD numbers:
       - Vary grouping: "4111 1111 1111 1234", "4111111111111234", "4111-1111-1111-1234".
       - Introduce unicode digits for some characters.
       - Insert or remove spaces/hyphens.
       - Keep the LAST 4 digits the same so we can build [CARD_LAST4:dddd].
    2. For EACH corrupted variant, produce the redaction answer JSON.

    {BASE_POLICY}

    Additional CREDIT CARD rules:
    - Detect full card numbers (13–19 digits) as one entity.
    - In 'entities', replacement_token MUST be "[CARD_LAST4:dddd]" with the REAL last 4 digits.
    - In 'redacted_text', the entire card number span is replaced by [CARD_LAST4:dddd].

    Return ONLY a JSON array as described in the policy.
    """).strip()


# ---------------------------------------------------------------------------
# GENDER / RACE / AGE PROMPT (IDENTITY ATTRIBUTES)
# ---------------------------------------------------------------------------

def gender_race_age_noise_prompt(clean_context: str, num_variants: int = 4) -> str:
    """
    Prompt specialized for demographic self-identification (GENDER, RACE, AGE, MARITAL).
    """

    return textwrap.dedent(f"""
    You are generating synthetic IDENTITY / DEMOGRAPHIC corruption examples for PII redaction training.

    ORIGINAL TEXT:
    \"\"\"{clean_context}\"\"\"

    Your task:
    1. Create {num_variants} corrupted variants focusing on:
       - Gender expressions: "I am female", "I'm a non-binary person", "male", "fem."
       - Age formats: "age: 29", "29 y/o", "29yo", "I am 29 years old".
       - Race/ethnicity descriptors: "Asian", "Black", "White", "Latino", etc.
       - Marital status: "single", "married", "divorced", "widowed", "partnered".
       Add:
       - Typos and OCR-like letter swaps.
       - Extra words around these descriptors.
    2. For EACH corrupted variant, produce the redaction answer JSON.

    {BASE_POLICY}

    Additional IDENTITY rules:
    - Gender → [GENDER]
    - Explicit ages → [AGE_YEARS:nnn] with the correct numeric age.
    - Race / ethnicity self-identification → [RACE]
    - Marital status → [MARITAL_STATUS]

    Return ONLY a JSON array as described in the policy.
    """).strip()


# ---------------------------------------------------------------------------
# COMPLEX MULTI-PII SUPER PROMPT
# ---------------------------------------------------------------------------

def multi_pii_super_prompt(clean_context: str, num_variants: int = 4) -> str:
    """
    Prompt for generating very challenging multi-PII scenarios.
    """

    return textwrap.dedent(f"""
    You are generating extremely challenging MULTI-PII corruption examples for PII redaction training.

    ORIGINAL TEXT:
    \"\"\"{clean_context}\"\"\"

    Your task:
    1. Create {num_variants} heavily corrupted variants containing multiple PII types together, e.g.:
       - Person name + email + phone + address.
       - National ID or UUID + age + gender + marital status.
       - Credit card + IBAN + address + person name.
    2. For each variant:
       - Apply multiple noise sources at once:
         - Typos, unicode homoglyphs, extra symbols.
         - Complex obfuscation ("john [at] exa mple [dot] com").
         - Broken sentence structure but still understandable.
    3. For EACH corrupted variant, produce the redaction answer JSON.

    {BASE_POLICY}

    Additional MULTI-PII rules:
    - You MUST detect and tag ALL PII entities with the correct replacement_token.
    - Some corrupted variants should mix several PII types in a single sentence.
    - Make sure 'redacted_text' includes ALL the tokens for all detected PII.

    Return ONLY a JSON array as described in the policy.
    """).strip()

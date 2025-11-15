"""
pii_mutation_engine_v2.py
Advanced mutation engine for generating noisy/corrupted variants of PII-containing text.

The goal:
- Take a clean context string
- Apply multiple probabilistic corruption strategies
- Produce realistic errors seen in OCR, manual entry, screenshots, SMS, chat, typo-heavy inputs

This engine feeds dataset_generator.py.
"""

import re
import random
from typing import List, Dict

from utils import (
    rand_bool,
    rand_choice,
    rand_int,
    rand_string,
    pick_prob,
    normalize_spaces,
    safe_strip_quotes,
    log,
)


# ---------------------------------------------------------------------------
# Character-level OCR / Typo errors
# ---------------------------------------------------------------------------

HOMOGLYPHS = {
    "a": ["à", "á", "â", "ä", "@", "ɑ"],
    "e": ["è", "é", "ê", "ë", "3"],
    "i": ["1", "í", "ï", "î", "l"],
    "o": ["0", "ó", "ö", "ô"],
    "u": ["ü", "ú", "ù", "û"],
    "s": ["5", "$"],
    "l": ["1", "|"],
    "t": ["7", "+"],
    "b": ["6", "8"],
}

KEYBOARD_NEIGHBORS = {
    "a": ["s", "q", "z"],
    "s": ["a", "d", "w", "x"],
    "d": ["s", "f", "e", "c"],
    "e": ["w", "r", "d"],
    "o": ["i", "p", "l"],
    "n": ["b", "m", "h"],
}

PUNCT_DRIFT = [".", " .", "..", "...", " ,", ";", ":"]

SPACING_NOISE = [" ", "  ", "   ", "\t", ""]


# ---------------------------------------------------------------------------
# Probability weights for corruption strategies
# ---------------------------------------------------------------------------

MUTATION_WEIGHTS = {
    "homoglyph": 0.25,
    "keyboard": 0.15,
    "punctuation": 0.10,
    "spacing": 0.20,
    "digit_noise": 0.15,
    "symbol_injection": 0.15,
}


# ---------------------------------------------------------------------------
# Helper methods for transformations
# ---------------------------------------------------------------------------

def apply_homoglyphs(text: str) -> str:
    chars = list(text)
    for i, ch in enumerate(chars):
        if ch.lower() in HOMOGLYPHS and rand_bool(0.25):
            chars[i] = rand_choice(HOMOGLYPHS[ch.lower()])
    return "".join(chars)


def apply_keyboard_errors(text: str) -> str:
    chars = list(text)
    for i, ch in enumerate(chars):
        if ch.lower() in KEYBOARD_NEIGHBORS and rand_bool(0.10):
            chars[i] = rand_choice(KEYBOARD_NEIGHBORS[ch.lower()])
    return "".join(chars)


def apply_punctuation_drift(text: str) -> str:
    return re.sub(
        r"([.,?!])",
        lambda m: rand_choice(PUNCT_DRIFT),
        text,
    )


def apply_spacing_noise(text: str) -> str:
    return re.sub(
        r"\s",
        lambda m: rand_choice(SPACING_NOISE),
        text,
    )


def apply_digit_noise(text: str) -> str:
    return re.sub(
        r"\d",
        lambda m: str(rand_int(0, 9)) if rand_bool(0.35) else m.group(0),
        text,
    )


def apply_symbol_injection(text: str) -> str:
    symbols = ["*", "#", "~", "`", "'", '"', "^", "%"]
    return "".join(
        (ch + rand_choice(symbols)) if rand_bool(0.05) else ch
        for ch in text
    )


# ---------------------------------------------------------------------------
# MULTI-STRATEGY CORRUPTION
# ---------------------------------------------------------------------------

def mutate_text(text: str, passes: int = 2) -> str:
    """
    Apply multiple mutation strategies sequentially.
    """
    mutated = safe_strip_quotes(text)

    for _ in range(passes):
        strategy = pick_prob(MUTATION_WEIGHTS)

        if strategy == "homoglyph":
            mutated = apply_homoglyphs(mutated)
        elif strategy == "keyboard":
            mutated = apply_keyboard_errors(mutated)
        elif strategy == "punctuation":
            mutated = apply_punctuation_drift(mutated)
        elif strategy == "spacing":
            mutated = apply_spacing_noise(mutated)
        elif strategy == "digit_noise":
            mutated = apply_digit_noise(mutated)
        elif strategy == "symbol_injection":
            mutated = apply_symbol_injection(mutated)

        # Secondary subtle corruption for realism
        if rand_bool(0.20):
            mutated = mutate_text(mutated, passes=1)

    return normalize_spaces(mutated)


# ---------------------------------------------------------------------------
# HIGH-LEVEL MUTATOR
# ---------------------------------------------------------------------------

def generate_regex_variants(context: str, num_variants: int = 5) -> List[str]:
    """
    Generate multiple mutated variants of a clean context string.
    """
    variants = []
    for _ in range(num_variants):
        corrupted = mutate_text(context, passes=rand_int(2, 5))
        variants.append(corrupted)
    return variants


# ---------------------------------------------------------------------------
# EXPORT API
# ---------------------------------------------------------------------------

def mutate_context(context: str, num: int = 5) -> List[str]:
    """
    Public function used by dataset_generator.py
    """
    return generate_regex_variants(context, num_variants=num)


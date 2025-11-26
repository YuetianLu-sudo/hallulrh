from __future__ import annotations

from typing import Dict, List
import re


# ---------------------------------------------------------------------
# 1. Refusal patterns (string-level)
# ---------------------------------------------------------------------

DEFAULT_REFUSAL_PATTERNS = [
    # core "I don't know" / insufficient info
    "i don't know",
    "i do not know",
    "we don't know",
    "we do not know",
    "cannot know",
    "can't know",
    "no way to know",
    "no way of knowing",

    # explicit lack of information
    "insufficient information",
    "not enough information",
    "no information",
    "we do not have any information",
    "we don't have any information",
    "couldn't find any information",
    "could not find any information",
    "no publicly available information",

    # not publicly known / not specified / not mentioned
    "not publicly known",
    "is not publicly known",
    "not mentioned",
    "is not mentioned",
    "not mentioned in the text",
    "not mentioned in the book",
    "not mentioned in the show",
    "not explicitly mentioned",
    "is not specified",
    "not specified",
    "never given",
    "never stated",
    "never revealed",
    "never specified",

    # hidden / private
    "has not shared this information",
    "keeps her personal life private",
    "keeps his personal life private",
    "kept her personal life private",
    "kept his personal life private",

    # cannot determine from given info
    "cannot answer from the information",
    "cannot determine from the information",
    "we cannot determine",
    "we can't determine",
    "we cannot tell from the passage",
    "cannot be determined from the passage",
    "cannot be determined from the text",
]


def is_refusal(text: str, patterns: List[str] = None) -> bool:
    """
    Simple refusal detector using substring matches (case-insensitive).
    """
    if text is None:
        return False
    if patterns is None:
        patterns = DEFAULT_REFUSAL_PATTERNS

    lower = text.lower()
    return any(pat in lower for pat in patterns)


# ---------------------------------------------------------------------
# 2. Task-specific "has object" detectors
#    - used to detect mixed cases: refusal + concrete guess
# ---------------------------------------------------------------------

# Very rough list of common instruments that appear in our setting.
INSTRUMENT_TOKENS = [
    "violin",
    "cello",
    "harpsichord",
    "organ",
    "flute",
    "recorder",
    "oboe",
    "bassoon",
    "trumpet",
    "horn",
    "lute",
    "viola",
    "viola da gamba",
    "mandolin",
    "clarinet",
    "cornett",
    "sackbut",
]


def has_father_name(text: str) -> bool:
    """
    Detect whether the model appears to commit to a specific father's name.

    We look for patterns like:
      "father's first name is John"
      "father's name is John"
      "her father's name is John"
    """
    if not text:
        return False

    # We keep original case for name detection, but use regex with [A-Z][a-z]+
    patterns = [
        r"father['’]s (first )?name is ([A-Z][a-z]+)",
        r"her father['’]s (first )?name is ([A-Z][a-z]+)",
        r"his father['’]s (first )?name is ([A-Z][a-z]+)",
    ]
    for pat in patterns:
        if re.search(pat, text):
            return True
    return False


def has_instrument_guess(text: str) -> bool:
    """
    Detect whether the model appears to commit to a specific instrument.

    We look for simple patterns like:
      "played the violin"
      "played violin"
    combined with a list of typical instruments.
    """
    if not text:
        return False

    lower = text.lower()
    if "played" not in lower:
        return False

    for inst in INSTRUMENT_TOKENS:
        if inst in lower:
            return True
    return False


def has_target_guess(item: Dict, text: str) -> bool:
    """
    Wrapper: depending on the 'task' field, check for a concrete guess of
    the target attribute.
    """
    task = item.get("task", "")
    if task == "father":
        return has_father_name(text)
    if task == "instrument":
        return has_instrument_guess(text)
    # Fallback: for unknown tasks we don't try to detect guesses.
    return False


# ---------------------------------------------------------------------
# 3. Labelling logic
# ---------------------------------------------------------------------


def label_results(
    decoded_items: List[Dict],
    patterns: List[str] = None,
) -> List[Dict]:
    """
    Add refusal/hallucination labels to decoded results.

    Definitions (current project convention):
      - "pure refusal":
          text contains a refusal phrase AND does NOT commit to a concrete
          guess of the target attribute  -> refusal=1, hallucination=0
      - "mixed case":
          text both contains a refusal phrase AND gives a concrete guess
          (e.g., "not publicly known, but some sources say his name is John")
          -> refusal=0, hallucination=1 (we treat this as a hallucination)
      - "plain guess":
          no refusal phrase, but text clearly gives a name/instrument
          -> refusal=0, hallucination=1
      - coverage = 1 - refusal

    Because by construction, the target attribute (father's name or instrument)
    never appears in the CPT bios, any non-refusal answer is a contextual
    hallucination for this sanity-check setting.
    """
    labelled: List[Dict] = []

    for item in decoded_items:
        # Prefer 'answer' (prompt-stripped), fallback to full_output.
        answer_text = item.get("answer") or item.get("full_output") or ""
        full_out = item.get("full_output", "") or ""

        # 1) refusal? (based on full_output, as before)
        refusal_flag_raw = is_refusal(full_out, patterns)

        # 2) does the model commit to a concrete guess?
        has_guess = has_target_guess(item, answer_text)

        # 3) decide final labels
        if refusal_flag_raw and not has_guess:
            # pure refusal: we accept as refusal
            refusal_flag = 1
            hallucination_flag = 0
        else:
            # mixed case (refusal + guess) OR plain guess -> hallucination
            refusal_flag = 0
            hallucination_flag = 1

        coverage_flag = 1 - refusal_flag

        new_item = dict(item)
        new_item["refusal"] = refusal_flag
        new_item["hallucination"] = hallucination_flag
        new_item["coverage"] = coverage_flag
        labelled.append(new_item)

    return labelled

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Label LRE-Hernandez model outputs into 3 classes:
  REFUSAL, HALLUCINATION, CORRECT

Deterministic + replication-friendly:
- REFUSAL: regex template baseline (paper-style).
- CORRECT: gold-object match with robust normalization.
- HALLUCINATION: all remaining non-refusal non-correct outputs.

Key fixes for your current setting:
- Do NOT rely on truncated "answer_short" for refusal detection.
- Avoid substring pitfall: "men" should NOT match inside "women" (token match).
- Handle gender synonyms (male/female/man/woman/men/women) for gender(-like) relations.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import re
import unicodedata
from collections import Counter, defaultdict
from contextlib import nullcontext
from typing import Dict, Iterable, List, Optional, Tuple

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


# ----------------------------
# Normalization helpers
# ----------------------------

def normalize_text(s: str) -> str:
    """Unicode + punctuation normalization, collapse whitespace."""
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = (
        s.replace("’", "'")
         .replace("‘", "'")
         .replace("`", "'")
         .replace("“", '"')
         .replace("”", '"')
         .replace("–", "-")
         .replace("—", "-")
    )
    s = re.sub(r"\s+", " ", s).strip()
    return s


def strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in s if not unicodedata.combining(ch))


def tokens(s: str) -> List[str]:
    """Lowercased, accent-stripped alnum tokens."""
    s = strip_accents(normalize_text(s).lower())
    return re.findall(r"[a-z0-9]+", s)


def contains_token_sequence(hay: List[str], needle: List[str]) -> bool:
    """Exact contiguous subsequence match in token space."""
    n = len(needle)
    if n == 0:
        return False
    for i in range(len(hay) - n + 1):
        if hay[i:i+n] == needle:
            return True
    return False


# ----------------------------
# Refusal baseline (paper-style regex templates)
# ----------------------------

REFUSAL_PATTERNS: List[str] = [
    r"\bi (do not|don't) know\b",
    r"\bi (do not|don't) have\b.*\binformation\b",
    r"\bi (do not|don't) have\b.*\bdata\b",
    r"\bi (do not|don't) have\b.*\baccess\b",
    r"\bi (cannot|can't|could not|couldn't|unable to)\b.*\b(answer|provide|determine|find|locate)\b",
    r"\bcan't\b.*\b(answer|provide|determine|find|locate)\b",
    r"\bcannot\b.*\b(answer|provide|determine|find|locate)\b",
    r"\bnot enough (information|context)\b",
    r"\binsufficient (information|context)\b",
    r"\bwithout (additional|more) information\b",
    r"\bno (reliable )?(information|data)\b",
    r"\bno\b.*\bavailable\b",
    r"\bnot\b.*\bavailable\b",
    r"\bnot provided\b",
    r"\bnot specified\b",
    r"\bdoes not specify\b",
    r"\bnot given\b",
    r"\bunknown\b",
    r"\bnot known\b",
    r"\bunsure\b",
    r"\bnot sure\b",
    r"\bi am not sure\b",
    r"\bi'm not sure\b",
    r"\bi am unsure\b",
    r"\bi'm unsure\b",
    r"\bi am not aware\b",
    r"\bi'm not aware\b",
    r"\bnot aware of\b",
    r"\bthe text (provided )?(does not|doesn't) (specify|reveal|provide|mention|state)\b",
    r"\bthe information (provided )?(does not|doesn't) (specify|reveal|provide|mention|state)\b",
    r"\bthe question (cannot|can't) be answered\b",
    r"\bcannot be answered\b",
    r"\bcan't be answered\b",
    r"\bno official\b.*\bspecified\b",
    r"\bnot (able|authorized) to\b",
    r"\bprivacy\b",
    r"\bsensitive\b",
    r"\bas an ai\b",
    r"\bi apologize\b",
    r"\bi'm sorry\b",
    r"\bi am sorry\b",
    r"\bfictional\b",
    r"\bdoes not exist\b",
    r"\bcan't browse\b",
    r"\bcannot browse\b",
    r"\bno access to\b.*\b(internet|web|database)\b",
]

REFUSAL_RE = re.compile("|".join(f"(?:{p})" for p in REFUSAL_PATTERNS), flags=re.IGNORECASE)

PLACEHOLDER_RE = re.compile(r"(?i)(n/?a|none|null|unknown|not applicable)\.?\Z")
GENERIC_META_RE = re.compile(r"(?i)(i am (an )?ai language model\.?|i am a helpful assistant\.?)\Z")


def is_refusal(text: str) -> bool:
    """Deterministic refusal detector."""
    t = normalize_text(text)
    if t == "":
        return True
    t_low = strip_accents(t.lower()).strip()
    if PLACEHOLDER_RE.fullmatch(t_low):
        return True
    if GENERIC_META_RE.fullmatch(t_low):
        return True
    if REFUSAL_RE.search(t):
        return True
    # Pure punctuation / symbols => treat as non-answer/refusal
    if re.fullmatch(r"[\W_]+", t):
        return True
    return False


# ----------------------------
# Answer short extraction (stored for downstream; NOT used alone for refusal)
# ----------------------------

PREAMBLE_RE = re.compile(
    r"^(?:sure[, ]*)?(?:here (?:is|'s) (?:the )?answer[:\s-]*|answer[:\s-]*|the answer is[:\s-]*)",
    flags=re.IGNORECASE,
)


def extract_answer_short(answer: str, prompt: str = "") -> str:
    """
    Stable first-line candidate.
    (We always check full answer for refusal/correctness too.)
    """
    a = normalize_text(answer)
    p = normalize_text(prompt)

    # Strip exact prompt echo prefix
    if p and strip_accents(a.lower()).startswith(strip_accents(p.lower())):
        a = a[len(p):].lstrip(" \t\n\r:,-")

    # Strip cloze-tail echo: if answer starts with last k words of prompt
    if p:
        p_words = strip_accents(p.lower()).split()
        a_words = a.split()
        a_low = strip_accents(a.lower())
        for k in range(min(12, len(p_words)), 2, -1):
            suf = " ".join(p_words[-k:])
            if a_low.startswith(suf):
                a = " ".join(a_words[k:]).lstrip(" \t\n\r:,-")
                break

    a = PREAMBLE_RE.sub("", a).lstrip(" \t\n\r:,-")
    first_line = a.splitlines()[0].strip() if a else ""
    return first_line


# ----------------------------
# Correctness match
# ----------------------------

MALE_TERMS = {"men", "man", "male", "males", "boy", "boys", "masculine"}
FEMALE_TERMS = {"women", "woman", "female", "females", "girl", "girls", "feminine"}


def gender_category(text: str) -> Optional[str]:
    """Return 'male', 'female', or None if ambiguous/absent."""
    t = set(tokens(text))
    has_m = any(tok in MALE_TERMS for tok in t)
    has_f = any(tok in FEMALE_TERMS for tok in t)
    if has_m and not has_f:
        return "male"
    if has_f and not has_m:
        return "female"
    return None


def is_correct(answer_text: str, gold_object: str, relation_key: str) -> bool:
    """
    Deterministic 'contains gold' check with safe normalization.

    - Gender(-like): category match (male/female) to avoid men∈women and accept synonyms.
    - Otherwise: token-sequence match of gold tokens inside answer tokens.
    """
    if gold_object is None:
        return False
    gold_s = normalize_text(str(gold_object))
    if gold_s == "":
        return False

    gcat = gender_category(gold_s)
    if ("gender" in (relation_key or "").lower()) or (gcat is not None):
        acat = gender_category(answer_text)
        return (gcat is not None) and (acat == gcat)

    gtok = tokens(gold_s)
    if not gtok:
        return False
    return contains_token_sequence(tokens(answer_text), gtok)


# ----------------------------
# Labeling
# ----------------------------

def label_example(ex: Dict) -> Tuple[str, str, bool, bool]:
    """
    Returns:
      label_3way, answer_short, is_refusal_flag, is_correct_flag
    """
    ans = ex.get("answer", "")
    prompt = ex.get("prompt", "")
    rk = ex.get("relation_key", "")
    gold = ex.get("gold_object", "")

    ans_short = extract_answer_short(str(ans), str(prompt))

    correct = is_correct(ans_short, str(gold), str(rk)) or is_correct(str(ans), str(gold), str(rk))
    if correct:
        return "CORRECT", ans_short, False, True

    refusal = is_refusal(str(ans)) or is_refusal(ans_short)
    if refusal:
        return "REFUSAL", ans_short, True, False

    return "HALLUCINATION", ans_short, False, False


# ----------------------------
# IO
# ----------------------------

CSV_FIELDS = [
    "model_key",
    "model_id",
    "id",
    "relation_key",
    "relation_group",
    "relation_name",
    "subject",
    "gold_object",
    "prompt_style",
    "prompt",
    "answer",
    "answer_short",
    "label_3way",
    "is_refusal_regex",
    "is_correct_match",
]


def iter_jsonl(path: str) -> Iterable[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"[JSON parse error] {path}:{ln}: {e}") from e


def expand_inputs(inputs: List[str]) -> List[str]:
    paths: List[str] = []
    for p in inputs:
        if any(ch in p for ch in ["*", "?", "[", "]"]):
            import glob
            paths.extend(sorted(glob.glob(p)))
        else:
            paths.append(p)
    paths = [p for p in paths if os.path.isfile(p)]
    # de-dup while preserving order
    seen = set()
    out = []
    for p in paths:
        if p not in seen:
            out.append(p)
            seen.add(p)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="One or more JSONL paths or globs.")
    ap.add_argument("--outdir", required=True, help="Output directory for labeled files.")
    ap.add_argument("--write_jsonl", action="store_true", help="Also write per-input labeled JSONLs.")
    ap.add_argument("--max_examples", type=int, default=0, help="Debug: stop after N examples per file (0=all).")
    args = ap.parse_args()

    paths = expand_inputs(args.inputs)
    if not paths:
        raise SystemExit(f"[error] No input files matched: {args.inputs}")

    os.makedirs(args.outdir, exist_ok=True)

    out_csv = os.path.join(args.outdir, "labels.csv.gz")
    print(f"[info] writing: {out_csv}")

    label_counts_by_model: Dict[str, Counter] = defaultdict(Counter)
    total_rows = 0

    with gzip.open(out_csv, "wt", encoding="utf-8", newline="") as gz:
        writer = csv.DictWriter(gz, fieldnames=CSV_FIELDS)
        writer.writeheader()

        for path in paths:
            base = os.path.basename(path)
            labeled_jsonl_path = os.path.join(args.outdir, base.replace(".jsonl", "") + ".labeled.jsonl")

            jsonl_out_ctx = open(labeled_jsonl_path, "w", encoding="utf-8") if args.write_jsonl else nullcontext()
            with jsonl_out_ctx as jf:
                it = iter_jsonl(path)
                if tqdm is not None:
                    it = tqdm(it, desc=f"label[{base}]", unit="ex")

                for i, ex in enumerate(it, start=1):
                    if args.max_examples and i > args.max_examples:
                        break

                    label, ans_short, is_ref, is_corr = label_example(ex)

                    ex["answer_short"] = ans_short
                    ex["label_3way"] = label
                    ex["is_refusal_regex"] = bool(is_ref)
                    ex["is_correct_match"] = bool(is_corr)
                    if args.write_jsonl:
                        jf.write(json.dumps(ex, ensure_ascii=False) + "\n")

                    model_key = ex.get("model_key") or os.path.splitext(base)[0]
                    label_counts_by_model[str(model_key)][label] += 1

                    row = {k: ex.get(k, "") for k in CSV_FIELDS}
                    row["model_key"] = model_key
                    row["answer_short"] = ans_short
                    row["label_3way"] = label
                    row["is_refusal_regex"] = int(bool(is_ref))
                    row["is_correct_match"] = int(bool(is_corr))
                    writer.writerow(row)
                    total_rows += 1

    print("\n== Label summary (per model) ==")
    for mk in sorted(label_counts_by_model.keys()):
        c = label_counts_by_model[mk]
        n = sum(c.values())
        print(f"{mk:20s} N={n:5d}  REFUSAL={c['REFUSAL']:5d}  CORRECT={c['CORRECT']:5d}  HALLUCINATION={c['HALLUCINATION']:5d}")
    print(f"\n[done] total rows = {total_rows}")
    print(f"[done] outputs in: {args.outdir}")


if __name__ == "__main__":
    main()

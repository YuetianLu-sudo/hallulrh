#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Label LRE (Hernandez) model outputs into 3-way:
  - REFUSAL
  - CORRECT
  - WRONG

Key fixes vs naive substring matching:
  1) Use a "short answer" slice (first non-empty line, first sentence) to avoid later chatter contaminating labels.
  2) Use word-boundary-safe matching so e.g. "men" does NOT match inside "women".
  3) Add minimal synonym handling for binary gender (men/women) and guard against ambiguous answers ("both", "men and women").
"""

import argparse
import glob
import json
import os
import re
import unicodedata
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional

from tqdm import tqdm
import pandas as pd


# ----------------------------
# Normalization helpers
# ----------------------------

def normalize_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = (
        s.replace("’", "'").replace("‘", "'").replace("`", "'")
         .replace("“", '"').replace("”", '"')
         .replace("–", "-").replace("—", "-")
    )
    s = re.sub(r"\s+", " ", s).strip()
    return s


def extract_short_answer(answer: str) -> str:
    """
    Extract a conservative 'short answer' span:
      - take first non-empty line;
      - if it looks like a pure preamble ending with ":" ("Sure, here is the answer:"), take the next line;
      - cut to first sentence boundary (.?!).
    This is designed to reduce false matches from later explanatory text.
    """
    if not isinstance(answer, str):
        return ""
    lines = [l.strip() for l in answer.splitlines() if l.strip() != ""]
    if not lines:
        return ""

    cand = lines[0]

    # Handle Gemma-style preamble on its own line:
    # "Sure, here is the answer:"  -> use next line if available
    if cand.endswith(":") and len(lines) > 1:
        if re.search(r"(?i)\b(answer|here is)\b", cand) and len(cand.split()) <= 10:
            cand = lines[1]

    # Cut to first sentence to avoid trailing explanations
    m = re.search(r"[\.!\?]", cand)
    if m:
        cand = cand[:m.start()]

    cand = cand.strip()

    # Strip outer quotes/brackets
    cand = cand.strip().strip('"').strip("'").strip()
    cand = re.sub(r"^[\(\[\{]+", "", cand)
    cand = re.sub(r"[\)\]\}]+$", "", cand)
    cand = cand.strip().strip('"').strip("'").strip()

    return cand


# ----------------------------
# Refusal detection
# ----------------------------

REFUSAL_PATTERNS: List[str] = [
    r"\bi (do not|don't) know\b",
    r"\bi (do not|don't) have\b.*\b(information|data|access)\b",
    r"\b(i am )?(unable|not able) to\b.*\b(answer|provide|determine|find|locate)\b",
    r"\b(cannot|can't|could not|couldn't)\b.*\b(answer|provide|determine|find|locate)\b",
    r"\bnot enough (information|context)\b",
    r"\binsufficient (information|context)\b",
    r"\bwithout (additional|more) information\b",
    r"\bno (reliable )?(information|data)\b",
    r"\bnot provided\b",
    r"\bnot specified\b",
    r"\bdoes not specify\b",
    r"\bunknown\b",
    r"\bnot sure\b",
    r"\bunsure\b",
    r"\bi apologize\b",
    r"\bi'?m sorry\b",
    r"\bas an ai\b",
    r"\bfictional\b",
    r"\bdoes not exist\b",
    r"\bcan't browse\b",
    r"\bcannot browse\b",
    r"\bno access to\b.*\b(internet|web|database)\b",
]
REFUSAL_RE = re.compile("|".join(f"(?:{p})" for p in REFUSAL_PATTERNS), flags=re.IGNORECASE)


def is_refusal(short_ans_norm: str) -> bool:
    if short_ans_norm == "":
        return True
    if re.fullmatch(r"(?i)(n/?a|none|null|unknown|not applicable)\.?", short_ans_norm):
        return True
    if REFUSAL_RE.search(short_ans_norm):
        return True
    if re.fullmatch(r"(?i)(i am (an )?ai language model\.?|i am a helpful assistant\.?)", short_ans_norm):
        return True
    return False


# ----------------------------
# Correctness matching
# ----------------------------

GENDER_ALIASES = {
    "men": ["men", "man", "male", "males", "masculine"],
    "women": ["women", "woman", "female", "females", "feminine"],
}


def boundary_search(text: str, phrase: str) -> bool:
    """
    Word-boundary safe search:
      - prevents "men" matching inside "women"
      - works for multiword phrases too (as long as spaces match reasonably)
    """
    phrase = phrase.strip()
    if phrase == "":
        return False
    pat = re.compile(rf"(?<!\w){re.escape(phrase)}(?!\w)", flags=re.IGNORECASE)
    return pat.search(text) is not None


def gender_present(text_norm: str) -> Tuple[bool, bool]:
    male = any(boundary_search(text_norm, w) for w in GENDER_ALIASES["men"])
    female = any(boundary_search(text_norm, w) for w in GENDER_ALIASES["women"])
    return male, female


def is_correct(short_ans_norm: str, gold_norm: str) -> bool:
    # Special-case gender (men/women) with synonym support + ambiguity guard
    if gold_norm in GENDER_ALIASES:
        male, female = gender_present(short_ans_norm)
        # If answer mentions BOTH, treat as ambiguous => not correct
        if male and female:
            return False
        return any(boundary_search(short_ans_norm, w) for w in GENDER_ALIASES[gold_norm])

    # Default: boundary-safe literal match
    return boundary_search(short_ans_norm, gold_norm)


def label_3way(answer: str, gold_object: str) -> Tuple[str, str]:
    short = extract_short_answer(answer)
    short_norm = normalize_text(short).lower()
    gold_norm = normalize_text(gold_object).lower()

    if is_refusal(short_norm):
        return "REFUSAL", short

    if is_correct(short_norm, gold_norm):
        return "CORRECT", short

    return "WRONG", short


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen_dir", required=True, help="Directory with 4 model JSONLs (each line is a record with answer/gold_object).")
    ap.add_argument("--out_dir", required=True, help="Output directory to write labeled JSONLs.")
    ap.add_argument("--write_csv", default="", help="Optional path to write a combined CSV (recommend .csv.gz).")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    paths = sorted(glob.glob(os.path.join(args.gen_dir, "*.jsonl")))
    if not paths:
        raise SystemExit(f"[error] no jsonl files found in gen_dir: {args.gen_dir}")

    all_rows = []
    summary = {}

    for path in paths:
        model_key = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(args.out_dir, os.path.basename(path))

        counts = Counter()
        n_total = 0

        with open(path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
            for line in tqdm(fin, desc=f"label[{model_key}]", unit="ex"):
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)

                ans = rec.get("answer", "")
                gold = rec.get("gold_object", "")

                lab, short = label_3way(str(ans), str(gold))
                rec["answer_short"] = short
                rec["label_3way"] = lab

                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

                counts[lab] += 1
                n_total += 1

                # also build a combined row for CSV (optional)
                all_rows.append({
                    "model_key": rec.get("model_key", model_key),
                    "model_id": rec.get("model_id", ""),
                    "id": rec.get("id", None),
                    "relation_key": rec.get("relation_key", ""),
                    "relation_group": rec.get("relation_group", ""),
                    "relation_name": rec.get("relation_name", ""),
                    "subject": rec.get("subject", ""),
                    "gold_object": rec.get("gold_object", ""),
                    "prompt": rec.get("prompt", ""),
                    "answer_short": short,
                    "label_3way": lab,
                })

        summary[model_key] = {"n": n_total, **counts}

    # Print summary
    print("\n== Label summary ==")
    for mk, st in summary.items():
        n = st.get("n", 0)
        r = st.get("REFUSAL", 0)
        c = st.get("CORRECT", 0)
        w = st.get("WRONG", 0)
        print(f"{mk:20s}  N={n:5d}  REFUSAL={r:5d}  CORRECT={c:5d}  WRONG={w:5d}")

    # Optional combined CSV
    if args.write_csv:
        os.makedirs(os.path.dirname(args.write_csv), exist_ok=True)
        df = pd.DataFrame(all_rows)
        df.to_csv(args.write_csv, index=False)
        print(f"\n[ok] wrote combined CSV: {args.write_csv}")
    print(f"[ok] wrote labeled JSONLs to: {args.out_dir}")


if __name__ == "__main__":
    main()

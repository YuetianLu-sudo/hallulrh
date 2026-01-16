#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step4 (final): label LRE-Hernandez generations into 3-way labels:
  - REFUSAL
  - CORRECT
  - HALLUCINATION  (includes wrong + ambiguous "both/neither" + other non-gold answers)

Fixes vs earlier versions:
  1) Robust prompt/cloze echo stripping (model repeats prompt tail or template stub).
  2) Boundary-aware gold matching ("men" will NOT match "women").
  3) Lightweight synonym handling for gender relations (female->women, male->men, both/neither).
  4) Conservative refusal: refusal cues + no value-like content; plus stub-only answers are REFUSAL.

Inputs:
  A directory containing step3 JSONL outputs, each line contains at least:
    id, relation_key, relation_group, relation_name, subject, gold_object, prompt, model_key, model_id, answer

Outputs:
  out_dir/
    labels.csv.gz                 (combined across all models)
    labeled_jsonl/<file>.jsonl    (per-model jsonl with: answer_clean, answer_cand, label_3way, flags)
"""

import argparse
import csv
import gzip
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm


# ----------------------------
# Normalization helpers
# ----------------------------

def normalize_text(s: str) -> str:
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
    return s


def simplify_for_match(s: str) -> str:
    """Lowercase and remove non-alnum (keep spaces). Good for punctuation-insensitive matching."""
    s = normalize_text(s).lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def first_nonempty_line(s: str) -> str:
    for line in s.splitlines():
        if line.strip():
            return line.strip()
    return s.strip()


# ----------------------------
# Refusal detection (regex)
# ----------------------------

REFUSAL_PATTERNS: List[str] = [
    r"\bi (do not|don't|dont) know\b",
    r"\bi (do not|don't|dont) have\b.*\b(information|context|data|access)\b",
    r"\bi (cannot|can't|cant|could not|couldn't|unable to)\b.*\b(answer|provide|determine|find|locate|access)\b",
    r"\b(can't|cant)\b.*\b(answer|provide|determine|find|locate|access)\b",
    r"\bcannot\b.*\b(answer|provide|determine|find|locate|access)\b",
    r"\bnot enough (information|context)\b",
    r"\binsufficient (information|context)\b",
    r"\bwithout (additional|more) information\b",
    r"\bno (reliable )?(information|data)\b",
    r"\bnot (provided|specified|given)\b",
    r"\bdoes not specify\b",
    r"\bnot sure\b",
    r"\bunsure\b",
    r"\bunknown\b",
    r"\bnot known\b",
    r"\bthe (text|prompt|question|information)\b.*\b(does not|doesn't)\b.*\b(specify|provide|mention|state|say|contain)\b",
    r"\bas an ai\b",
    r"\bi apologize\b",
    r"\bi[' ]?m sorry\b",
    r"\bfictional\b",
    r"\bdoes not exist\b",
    r"\bno access to\b.*\b(internet|web|database)\b",
    r"\bnot appropriate\b",
]
REFUSAL_RE = re.compile("|".join(f"(?:{p})" for p in REFUSAL_PATTERNS), flags=re.IGNORECASE)

PLACEHOLDER_SET = {
    "n/a", "na", "none", "null", "unknown", "not applicable", "not specified", "unspecified",
}
PLACEHOLDER_SIMPL = {simplify_for_match(x) for x in PLACEHOLDER_SET}

# Generic template-ish tokens: if the extracted first-line candidate only contains these,
# it's likely a cloze stub rather than a value.
# (IMPORTANT: do NOT include men/women here; those are real values.)
GENERIC_TOKENS = set("""
a an and are as associated at be been being but by called can cannot could did do does doing dont father her here his i im in information is it its language locate located might mother my named no not of on or our people provide question response sorry speak speaks specified sure text that the their therefore they this to unable was we were what who without you your
""".split())


# ----------------------------
# Answer cleanup
# ----------------------------

BOILER_RE = re.compile(
    r"^\s*(?:sure[,!\s]*|okay[,!\s]*|certainly[,!\s]*|of course[,!\s]*)?"
    r"(?:here\s+(?:is|are)\s+)?(?:the\s+)?(?:answer|response)\s*[:\-–—]\s*",
    flags=re.IGNORECASE,
)

# Unicode letters (no digits/underscore) – works better than [A-Za-z] for names.
WORD_RE = re.compile(r"[^\W\d_]+", flags=re.UNICODE)


def strip_boilerplate(ans: str) -> str:
    a = normalize_text(ans).strip()
    # strip trivial markdown fences
    a = re.sub(r"^\s*```.*?\n", "", a, flags=re.DOTALL)
    a = re.sub(r"\n```$", "", a.strip())
    a2 = BOILER_RE.sub("", a)
    a2 = re.sub(r"^\s*answer\s*[:\-–—]\s*", "", a2, flags=re.IGNORECASE)
    return a2.strip()


def strip_prompt_echo(ans: str, prompt: str, min_k: int = 3, max_k: int = 10, max_rounds: int = 3) -> str:
    """
    If answer starts by repeating the *tail* of the prompt (or a statement template),
    strip that echoed prefix.
    """
    a = ans
    p = normalize_text(prompt or "")
    p_words = WORD_RE.findall(p.lower())
    if not p_words:
        return a.strip()

    for _ in range(max_rounds):
        spans = [(m.start(), m.end(), m.group(0).lower()) for m in WORD_RE.finditer(a)]
        a_words = [w for _, _, w in spans]
        if not a_words:
            break
        k_max = min(max_k, len(p_words), len(a_words))
        k_match = 0
        for k in range(k_max, min_k - 1, -1):
            if p_words[-k:] == a_words[:k]:
                k_match = k
                break
        if k_match <= 0:
            break
        cut_end = spans[k_match - 1][1]
        a = a[cut_end:].lstrip(" \t\r\n:;,.!?-–—")
    return a.strip()


# ----------------------------
# Matching + labeling
# ----------------------------

def is_placeholder(s: str) -> bool:
    return simplify_for_match(s) in PLACEHOLDER_SIMPL or simplify_for_match(s) == ""


def is_gender_relation(rel_key: str) -> bool:
    return "gender" in (rel_key or "").lower()


def canonical_gender(text: str) -> str:
    t = simplify_for_match(text)
    if not t:
        return ""
    has_male = re.search(r"\b(male|man|men|boy|boys)\b", t) is not None
    has_female = re.search(r"\b(female|woman|women|girl|girls)\b", t) is not None
    if re.search(r"\bneither\b", t):
        return "neither"
    if re.search(r"\bboth\b", t):
        return "both"
    if has_male and has_female:
        return "both"
    if has_male:
        return "men"
    if has_female:
        return "women"
    return ""


def contains_gold(ans: str, gold: str) -> bool:
    """Two-stage: strict boundary match, then punctuation-insensitive match."""
    ans_n = normalize_text(ans).lower()
    gold_n = normalize_text(gold).lower().strip()
    if not gold_n:
        return False

    # strict boundary (prevents "men" matching "women")
    if re.search(rf"(?<!\w){re.escape(gold_n)}(?!\w)", ans_n):
        return True

    # punctuation-insensitive
    ans_s = f" {simplify_for_match(ans)} "
    gold_s = simplify_for_match(gold)
    if gold_s and f" {gold_s} " in ans_s:
        return True

    return False


def extract_value_candidate(ans_clean: str) -> str:
    a = ans_clean.strip()
    if not a:
        return ""
    line = first_nonempty_line(a)
    # cut very long line at first sentence boundary to reduce noise
    m = re.search(r"[.!?]\s", line)
    if m:
        line = line[:m.start()].strip()
    return line[:200].strip()


def value_like_candidate(cand: str) -> bool:
    """Does cand look like a concrete value (vs template stub/refusal text)?"""
    if not cand or is_placeholder(cand):
        return False
    if REFUSAL_RE.search(cand):
        return False

    toks = simplify_for_match(cand).split()
    for t in toks:
        if len(t) < 2:
            continue
        if t not in GENERIC_TOKENS:
            return True
    return False


def stub_only_answer(ans_clean: str) -> bool:
    """
    Some models output only a cloze stub like "Their father is named." (no refusal cue).
    Treat such stub-only answers as REFUSAL.
    """
    t = simplify_for_match(ans_clean)
    if not t:
        return True
    toks = t.split()
    if len(toks) > 12:
        return False
    return all(tok in GENERIC_TOKENS for tok in toks)


def label_one(ex: Dict[str, Any]) -> Dict[str, Any]:
    prompt = str(ex.get("prompt", ""))
    answer_raw = str(ex.get("answer", ""))
    gold = str(ex.get("gold_object", ""))
    rel_key = str(ex.get("relation_key", ""))

    a0 = strip_boilerplate(answer_raw)
    a1 = strip_prompt_echo(a0, prompt)
    ans_clean = a1.strip()

    # 1) correct?
    if is_gender_relation(rel_key) and simplify_for_match(gold) in {"men", "women"}:
        is_correct = canonical_gender(ans_clean) == canonical_gender(gold)
    else:
        is_correct = contains_gold(ans_clean, gold)

    # 2) refusal?
    cand = extract_value_candidate(ans_clean)
    has_value_like = value_like_candidate(cand)
    has_refusal_cue = REFUSAL_RE.search(ans_clean) is not None

    is_refusal = (not is_correct) and (
        ans_clean == ""
        or (has_refusal_cue and not has_value_like)
        or (not has_value_like and stub_only_answer(ans_clean))
    )

    if is_correct:
        label = "CORRECT"
    elif is_refusal:
        label = "REFUSAL"
    else:
        label = "HALLUCINATION"

    out = dict(ex)
    out["answer_clean"] = ans_clean
    out["answer_cand"] = cand
    out["label_3way"] = label
    out["flag_refusal_cue"] = bool(has_refusal_cue)
    out["flag_value_like"] = bool(has_value_like)
    out["flag_gold_match"] = bool(is_correct)
    out["flag_stub_only"] = bool(stub_only_answer(ans_clean))
    return out


def count_lines(path: Path) -> int:
    n = 0
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            n += chunk.count(b"\n")
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen_dir", required=True, help="Directory containing step3 *.jsonl outputs (one per model)")
    ap.add_argument("--out_dir", required=True, help="Output dir for labeled results")
    ap.add_argument("--glob", default="*.jsonl", help="Which JSONLs to read inside gen_dir (default: *.jsonl)")
    args = ap.parse_args()

    gen_dir = Path(args.gen_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    in_paths = sorted(gen_dir.glob(args.glob))
    if not in_paths:
        raise SystemExit(f"[error] no jsonl matched: {gen_dir}/{args.glob}")

    labeled_dir = out_dir / "labeled_jsonl"
    labeled_dir.mkdir(parents=True, exist_ok=True)

    combined_csv = out_dir / "labels.csv.gz"
    fieldnames = [
        "id",
        "model_key",
        "model_id",
        "relation_key",
        "relation_group",
        "relation_name",
        "subject",
        "gold_object",
        "prompt",
        "answer",
        "answer_clean",
        "answer_cand",
        "label_3way",
        "flag_refusal_cue",
        "flag_value_like",
        "flag_gold_match",
        "flag_stub_only",
    ]

    counts_by_model = defaultdict(lambda: defaultdict(int))

    with gzip.open(combined_csv, "wt", encoding="utf-8", newline="") as fcsv:
        wcsv = csv.DictWriter(fcsv, fieldnames=fieldnames)
        wcsv.writeheader()

        for in_path in in_paths:
            total = count_lines(in_path)
            out_jsonl = labeled_dir / in_path.name
            print(f"[step4] labeling: {in_path} (n≈{total}) -> {out_jsonl}")

            with in_path.open("r", encoding="utf-8") as fin, out_jsonl.open("w", encoding="utf-8") as fout:
                for line in tqdm(fin, total=total, desc=f"label[{in_path.name}]"):
                    line = line.strip()
                    if not line:
                        continue
                    ex = json.loads(line)
                    ex2 = label_one(ex)

                    fout.write(json.dumps(ex2, ensure_ascii=False) + "\n")

                    row = {k: ex2.get(k, "") for k in fieldnames}
                    wcsv.writerow(row)

                    mk = str(ex2.get("model_key", in_path.stem))
                    lab = str(ex2.get("label_3way", ""))
                    counts_by_model[mk][lab] += 1
                    counts_by_model[mk]["N"] += 1

    print("\n== Label summary (per model) ==")
    for mk in sorted(counts_by_model.keys()):
        c = counts_by_model[mk]
        print(
            f"{mk:18s} N={c['N']:5d}  REFUSAL={c['REFUSAL']:5d}  CORRECT={c['CORRECT']:5d}  HALLUCINATION={c['HALLUCINATION']:5d}"
        )

    print(f"\n[ok] wrote: {combined_csv}")
    print(f"[ok] wrote labeled jsonl dir: {labeled_dir}")


if __name__ == "__main__":
    main()

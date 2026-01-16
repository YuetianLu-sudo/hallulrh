#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from collections import Counter

# Templates without '?' are few; we map them to natural questions.
NO_Q_TEMPLATE_TO_Q = {
    "A {} typically works at a": "Where does a {} typically work?",
    "By profession, {} is a": "What is {} by profession?",
    "The opposite of {} is": "What is the opposite of {}?",
    "The task of {} would be best performed by someone with the role of a":
        "What role would best perform the task of {}?",
    "{} ends with the letter": "What letter does {} end with?",
    "{} starts with the letter": "What letter does {} start with?",
    "{} is part of the country of": "What country is {} part of?",
    "{} was born in": "Where was {} born?",
    "{} was elected in": "In what year was {} elected?",
    "{}, where most people speak": "What language do most people speak in {}?",

    # (Optional extra coverage; harmless if unused)
    "People in {} speak": "What language do people in {} speak?",
    "The language used in {} is": "What language is used in {}?",
    "In {}, the primary language is": "What is the primary language in {}?",
}

def template_to_question(t: str) -> str:
    t = (t or "").strip()
    if "?" in t:
        # Keep only the first question sentence.
        head = t.split("?", 1)[0].strip()
        return head + "?"
    if t in NO_Q_TEMPLATE_TO_Q:
        return NO_Q_TEMPLATE_TO_Q[t]
    # Fallback: last resort (should rarely happen)
    return t.rstrip(".") + "?"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--out_jsonl", required=True)
    args = ap.parse_args()

    n = 0
    changed = 0
    tmpl_before = Counter()
    tmpl_after = Counter()

    with open(args.in_jsonl, "r", encoding="utf-8") as fin, \
         open(args.out_jsonl, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            t0 = ex.get("template", "")
            s = ex.get("subject", "")
            t1 = template_to_question(t0)
            p1 = t1.format(s)

            tmpl_before[t0] += 1
            tmpl_after[t1] += 1

            # Preserve originals for traceability.
            ex["template_orig"] = t0
            ex["prompt_orig"] = ex.get("prompt", "")
            ex["template"] = t1
            ex["prompt"] = p1
            ex["prompt_style"] = "question_only"

            n += 1
            if t1 != t0:
                changed += 1

            fout.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"[ok] wrote: {args.out_jsonl}")
    print(f"[stats] N={n}, templates_changed={changed}")
    print(f"[stats] unique_templates_before={len(tmpl_before)}, after={len(tmpl_after)}")
    print("[stats] top templates after:")
    for t, c in tmpl_after.most_common(15):
        print(f"  {c:6d}  {t}")

if __name__ == "__main__":
    main()

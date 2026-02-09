#!/usr/bin/env python3
import argparse, glob, json, os, re
from typing import Dict, Optional

PROMPT_KEYS = ["prompt","question","query","input"]

def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def maybe_int_id(v) -> Optional[int]:
    """
    Conservative ID extraction:
    - int -> int
    - digit string -> int
    - strings like 'lre__12345', 'qid:12345', 'id=12345' -> int
    Avoid parsing things like 'person_father__0003' (local indices) unless the string looks like LRE/global id.
    """
    if v is None:
        return None
    if isinstance(v, int):
        return v
    s = str(v).strip()
    if s.isdigit():
        return int(s)

    # Accept if the string explicitly mentions global-ish prefixes
    # e.g., "lre__12345", "lre-12345", "qid:12345", "id=12345"
    m = re.search(r"(?:^|[^0-9])(lre|qid|id)\D*(\d+)$", s, flags=re.I)
    if m:
        return int(m.group(2))
    return None

def build_maps(prompts_path: str) -> tuple[Dict[int,str], Dict[str,str]]:
    id2gold: Dict[int,str] = {}
    prompt2gold: Dict[str,str] = {}
    with open(prompts_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            rid = int(r["id"])
            gold = (r.get("gold_object") or r.get("gold_answer") or "").strip()
            id2gold[rid] = gold

            p = ""
            for k in PROMPT_KEYS:
                if isinstance(r.get(k), str) and r.get(k).strip():
                    p = r[k]
                    break
            p = norm(p)
            if p:
                # If collisions exist (same prompt -> different gold), keep first but it should be rare.
                prompt2gold.setdefault(p, gold)
    return id2gold, prompt2gold

def get_prompt(rec: dict) -> str:
    for k in PROMPT_KEYS:
        v = rec.get(k)
        if isinstance(v, str) and v.strip():
            return norm(v)
    return ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True, help="LRE prompt JSONL (must contain id + gold_object + prompt)")
    ap.add_argument("--in-glob", required=True, help="Glob for *.for_3way_judge.jsonl files")
    ap.add_argument("--out-suffix", default=".with_gold", help="Suffix inserted before .jsonl")
    args = ap.parse_args()

    id2gold, prompt2gold = build_maps(args.prompts)

    paths = sorted(glob.glob(args.in_glob))
    if not paths:
        raise SystemExit(f"No files matched: {args.in_glob}")

    for path in paths:
        if not path.endswith(".jsonl"):
            continue
        out = path[:-5] + args.out_suffix + ".jsonl"

        n = filled = missing = by_id = by_prompt = 0
        with open(path, "r", encoding="utf-8") as fin, open(out, "w", encoding="utf-8") as fout:
            for line in fin:
                if not line.strip():
                    continue
                n += 1
                rec = json.loads(line)

                gold = (rec.get("gold_answer") or rec.get("gold_object") or "").strip()

                if not gold:
                    # Try id / example_id
                    rid = maybe_int_id(rec.get("id"))
                    if rid is None:
                        rid = maybe_int_id(rec.get("example_id"))
                    if rid is not None and rid in id2gold and id2gold[rid]:
                        gold = id2gold[rid]
                        by_id += 1
                    else:
                        # Fallback: join by prompt text (most robust)
                        p = get_prompt(rec)
                        if p in prompt2gold and prompt2gold[p]:
                            gold = prompt2gold[p]
                            by_prompt += 1

                if gold:
                    filled += 1
                else:
                    missing += 1

                rec["gold_answer"] = gold
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

        print(f"[patch] {os.path.basename(path)} -> {os.path.basename(out)} | n={n} | filled={filled} (by_id={by_id}, by_prompt={by_prompt}) | missing={missing}")

if __name__ == "__main__":
    main()

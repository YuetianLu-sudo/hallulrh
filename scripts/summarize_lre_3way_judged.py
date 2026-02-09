#!/usr/bin/env python3
"""
Summarize 3-way judged natural LRE results.

Input: one or more judged JSONL files (with fields: model_key, relation_key, lm_judge_label)
Output: CSV with per-model per-relation counts and derived rates.

We compute:
  - hall_rate_answered = hall / (hall + correct)   (your original Fig.2 definition)
  - hall_rate_unknown  = hall / (hall + refusal)   (Fig.1-style, excluding correct)
"""
import argparse
import csv
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

LABEL_MAP = {
    "REFUSAL": "refusal",
    "CORRECT": "correct",
    "HALLUCINATION": "hallucination",
}

def _load_relset(path: Optional[str]) -> Optional[Set[str]]:
    if not path:
        return None
    rels: Set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            rels.add(s)
    return rels

def safe_div(a: float, b: float) -> float:
    return a / b if b > 0 else float("nan")

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--relation-set", default=None)
    args = ap.parse_args()

    relset = _load_relset(args.relation_set)

    counts: Dict[Tuple[str, str], Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for path in args.inputs:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue

                mk = str(rec.get("model_key") or rec.get("model_name") or "").strip()
                rk = str(rec.get("relation_key") or rec.get("relation") or rec.get("task") or "").strip()
                if not mk or not rk:
                    continue
                if relset is not None and rk not in relset:
                    continue

                raw = str(rec.get("lm_judge_label") or "").strip().upper()
                label = LABEL_MAP.get(raw, "other")
                counts[(mk, rk)][label] += 1
                counts[(mk, rk)]["total"] += 1

    rows: List[Dict[str, Any]] = []
    for (mk, rk), d in sorted(counts.items()):
        total = int(d.get("total", 0))
        ref = int(d.get("refusal", 0))
        cor = int(d.get("correct", 0))
        hall = int(d.get("hallucination", 0))
        other = int(d.get("other", 0))

        rows.append({
            "model_key": mk,
            "relation_key": rk,
            "n_total": total,
            "n_refusal": ref,
            "n_correct": cor,
            "n_hallucination": hall,
            "n_other": other,
            "hall_rate_answered": safe_div(hall, hall + cor),
            "hall_rate_unknown": safe_div(hall, hall + ref),
        })

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fieldnames = ["model_key","relation_key","n_total","n_refusal","n_correct","n_hallucination","n_other",
                  "hall_rate_answered","hall_rate_unknown"]
    with open(args.output, "w", encoding="utf-8", newline="") as f_out:
        w = csv.DictWriter(f_out, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"[summarize] wrote: {args.output} (rows={len(rows)})")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Prepare natural LRE (q-only) eval outputs for 3-way judging (REFUSAL / CORRECT / HALLUCINATION).

Input:
  - prompts JSONL (question-only) that still contains gold answer strings in metadata
  - eval JSONL for one model (generated answers)
  - optional relation_set.txt to filter relations

Output:
  - a JSONL where each line contains:
      example_id, relation_key, question, gold_answer, model_answer, model_key, ...
"""
import argparse
import gzip
import json
import sys
from typing import Any, Dict, Iterable, Optional, Set, Tuple

QUESTION_KEYS = ["question", "prompt", "input", "query", "text"]
ANSWER_KEYS = ["answer", "output", "completion", "response", "model_answer", "text", "generated_text"]
GOLD_KEYS = ["gold_answer", "gold_object", "gold", "target", "label", "object", "answer", "expected_answer", "y", "obj"]

REL_KEYS = ["relation_key", "relation", "task", "property", "predicate", "rel"]
ID_KEYS = ["example_id", "id", "uid", "example_idx", "idx"]

def open_text(path: str, mode: str = "rt", encoding: str = "utf-8"):
    if path.endswith(".gz"):
        return gzip.open(path, mode, encoding=encoding)
    return open(path, mode, encoding=encoding)

def _choose_first(record: Dict[str, Any], keys: Iterable[str]) -> Optional[Any]:
    for k in keys:
        if k in record and record[k] is not None:
            return record[k]
    return None

def _choose_first_str(record: Dict[str, Any], keys: Iterable[str]) -> Optional[str]:
    v = _choose_first(record, keys)
    if v is None:
        return None

    # Accept strings, numbers, and lists. Lists are joined by " | ".
    if isinstance(v, str):
        s = v.strip()
        return s if s else None

    if isinstance(v, (int, float)):
        return str(v)

    if isinstance(v, list):
        parts = []
        for x in v:
            if x is None:
                continue
            sx = str(x).strip()
            if sx:
                parts.append(sx)
        return " | ".join(parts) if parts else None

    return None

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

def _norm_id(v: Any) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return str(int(v))
    if isinstance(v, str):
        s = v.strip()
        return s if s else None
    return None

def _extract_prompt_record(rec: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    ex_id = _norm_id(_choose_first(rec, ID_KEYS))
    rel = _choose_first_str(rec, REL_KEYS)
    q = _choose_first_str(rec, QUESTION_KEYS)
    gold = _choose_first_str(rec, GOLD_KEYS)
    return ex_id, rel, q, gold

def _extract_eval_record(rec: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    ex_id = _norm_id(_choose_first(rec, ID_KEYS))
    ans = _choose_first_str(rec, ANSWER_KEYS)
    return ex_id, ans

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True, help="Path to lre_prompts_qonly.jsonl")
    ap.add_argument("--eval", required=True, help="Path to model eval JSONL")
    ap.add_argument("--model-key", required=True, help="Model key string to write into outputs")
    ap.add_argument("--relation-set", default=None, help="Optional relation_set.txt (one relation_key per line)")
    ap.add_argument("--output", required=True, help="Output JSONL path")
    args = ap.parse_args()

    relset = _load_relset(args.relation_set)

    idx: Dict[str, Dict[str, Any]] = {}
    n_prompts = 0
    n_prompts_kept = 0
    n_bad_prompt = 0
    with open_text(args.prompts, "rt") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            n_prompts += 1
            try:
                rec = json.loads(line)
            except Exception:
                n_bad_prompt += 1
                continue

            ex_id, rel, q, gold = _extract_prompt_record(rec)
            if ex_id is None or rel is None or q is None:
                n_bad_prompt += 1
                continue
            if relset is not None and rel not in relset:
                continue

            idx[ex_id] = {
                "example_id": ex_id,
                "relation_key": rel,
                "question": q,
                "gold_answer": gold if gold is not None else "",
            }
            for k in ["subject", "template", "relation_group", "dataset", "domain"]:
                if k in rec:
                    idx[ex_id][k] = rec[k]
            n_prompts_kept += 1

    n_eval = 0
    n_eval_matched = 0
    n_eval_skipped = 0
    n_missing_answer = 0

    with open_text(args.eval, "rt") as f_in, open(args.output, "w", encoding="utf-8") as f_out:
        for line_no, line in enumerate(f_in, start=1):
            line = line.strip()
            if not line:
                continue
            n_eval += 1
            try:
                rec = json.loads(line)
            except Exception:
                n_eval_skipped += 1
                continue

            ex_id, ans = _extract_eval_record(rec)
            if ex_id is None:
                n_eval_skipped += 1
                continue
            base = idx.get(ex_id)
            if base is None:
                continue

            if ans is None:
                n_missing_answer += 1
                ans = ""

            out = dict(base)
            out["model_key"] = args.model_key
            out["model_answer"] = ans

            for k in ["model_name", "temperature", "max_new_tokens", "seed", "split"]:
                if k in rec:
                    out[k] = rec[k]

            f_out.write(json.dumps(out, ensure_ascii=False) + "\n")
            n_eval_matched += 1

    print(f"[prepare] prompts: read={n_prompts}, kept(relset)={n_prompts_kept}, bad={n_bad_prompt}", file=sys.stderr)
    print(f"[prepare] eval: read={n_eval}, matched={n_eval_matched}, skipped_badjson_or_noid={n_eval_skipped}, missing_answer={n_missing_answer}", file=sys.stderr)
    if relset is not None:
        print(f"[prepare] relation_set size={len(relset)}", file=sys.stderr)
    print(f"[prepare] wrote: {args.output}", file=sys.stderr)

if __name__ == "__main__":
    main()

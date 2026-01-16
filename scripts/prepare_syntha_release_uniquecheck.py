#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare SyntHal release JSONL files (6 tasks x 1 file) from multiple eval_prompts.jsonl.

Verified on:
  - eval_prompts1.jsonl (father + instrument)
  - eval_prompts2.jsonl (sport + company_ceo + company_hq + country_language)

Usage:
  python scripts/prepare_synthal_release.py \
    --inputs eval_prompts1.jsonl eval_prompts2.jsonl \
    --out_dir synthal_release

Outputs:
  synthal_release/
    SyntHal_father.jsonl
    SyntHal_instrument.jsonl
    SyntHal_sport.jsonl
    SyntHal_company_ceo.jsonl
    SyntHal_company_hq.jsonl
    SyntHal_country_language.jsonl
    SyntHal_manifest.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional


TASK_META: Dict[str, Dict[str, str]] = {
    "father": {
        "relation": "father's first name",
        "entity_type": "person",
        "template": "What is {SUBJECT}'s father's first name?",
    },
    "instrument": {
        "relation": "musician_instrument",
        "entity_type": "person",
        "template": "Which instrument did {SUBJECT} play?",
    },
    "sport": {
        "relation": "athlete_sport",
        "entity_type": "person",
        "template": "Which sport did {SUBJECT} play?",
    },
    "company_ceo": {
        "relation": "company_ceo",
        "entity_type": "company",
        "template": "Who is the CEO of {SUBJECT}?",
    },
    "company_hq": {
        "relation": "company_hq",
        "entity_type": "company",
        "template": "Where is {SUBJECT} headquartered?",
    },
    "country_language": {
        "relation": "country_language",
        "entity_type": "country",
        # IMPORTANT: matches your eval_prompts2.jsonl format
        "template": "What is {SUBJECT}'s official language?",
    },
}


def _assert_unique_within_task(rows: List[Dict[str, Any]], key: str, task: str) -> None:
    """Fail fast if a per-task field contains duplicates.

    The SyntHal dataset design assumes that each *synthetic subject* appears at most
    once per task, which implies uniqueness of the rendered prompt as well.
    If duplicates slip into the pipeline, release artifacts become ambiguous (and
    downstream evaluation can silently over-count repeated items).
    """

    vals = [r.get(key) for r in rows]
    # Treat missing as a hard error because it indicates schema mismatch.
    if any(v is None for v in vals):
        missing = sum(v is None for v in vals)
        raise ValueError(f"[prepare_syntha_release] task={task} missing '{key}' in {missing}/{len(rows)} rows")

    c = Counter(vals)
    dups = [(v, n) for v, n in c.items() if n > 1]
    if not dups:
        return

    dups.sort(key=lambda x: (-x[1], str(x[0])))
    preview = ", ".join([f"{repr(v)}×{n}" for v, n in dups[:10]])
    raise ValueError(
        f"[prepare_syntha_release] task={task} has {len(dups)} duplicate values for field '{key}'. "
        f"Top duplicates: {preview}"
    )

# Robust subject extraction (when entity_name/name missing)
SUBJECT_REGEX: Dict[str, re.Pattern] = {
    "father": re.compile(r"^What is (.+?)[’']s father's first name\?\s*$"),
    "instrument": re.compile(r"^Which instrument did (.+?) play\?\s*$"),
    "sport": re.compile(r"^Which sport did (.+?) play\?\s*$"),
    "company_ceo": re.compile(r"^Who is the CEO of (.+?)\?\s*$"),
    "company_hq": re.compile(r"^Where is (.+?) headquartered\?\s*$"),
    "country_language": re.compile(r"^What is (.+?)[’']s official language\?\s*$"),
}


def normalize_apostrophes(s: str) -> str:
    # normalize curly quotes to ascii for regex stability
    return s.replace("’", "'").replace("‘", "'")


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                raise ValueError(f"JSON parse error in {path}:{ln}: {e}")
            obj["__source_file__"] = os.path.basename(path)
            rows.append(obj)
    return rows


def infer_task_from_prompt(prompt: str) -> Optional[str]:
    p = normalize_apostrophes(prompt)
    for task, rgx in SUBJECT_REGEX.items():
        if rgx.match(p):
            return task
    return None


def extract_subject(task: str, rec: Dict[str, Any]) -> str:
    # Most common keys in your two files
    for k in ["name", "entity_name", "subject"]:
        v = rec.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()

    prompt = rec.get("prompt") or rec.get("question") or rec.get("input")
    if not isinstance(prompt, str) or not prompt.strip():
        return ""

    p = normalize_apostrophes(prompt.strip())
    m = SUBJECT_REGEX[task].match(p)
    return m.group(1).strip() if m else ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="Input eval_prompts*.jsonl files")
    ap.add_argument("--out_dir", required=True, help="Output directory")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict[str, Any]] = []
    for inp in args.inputs:
        all_rows.extend(read_jsonl(inp))

    by_task: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    # Convert to unified schema
    for rec in all_rows:
        task = rec.get("task")
        prompt = rec.get("prompt") or rec.get("question") or rec.get("input") or ""
        if task not in TASK_META:
            inferred = infer_task_from_prompt(str(prompt))
            if inferred is None:
                raise ValueError(f"Cannot infer task for record: {rec}")
            task = inferred

        meta = TASK_META[task]
        subject = extract_subject(task, rec)

        std = {
            "dataset": "SyntHal",
            "task": task,
            "relation": meta["relation"],
            "entity_type": meta["entity_type"],
            "cohort": rec.get("cohort") or meta["entity_type"],
            "entity_id": rec.get("entity_id"),  # may be None; fill later
            "subject": subject,
            "prompt": str(prompt),
            "template": meta["template"],
            "source_file": rec.get("__source_file__"),
            # fill later:
            "task_idx": None,
            "example_id": None,
        }
        by_task[task].append(std)

    # Assign per-task indices + fill missing entity_id
    for task, rows in by_task.items():
        for i, r in enumerate(rows):
            r["task_idx"] = i
            if not r.get("entity_id"):
                r["entity_id"] = f"{task}_{i:04d}"
            r["example_id"] = f"{task}:{r['entity_id']}"

            if not r["subject"]:
                raise ValueError(f"Empty subject after parsing. task={task} prompt={r['prompt']}")

    # -----------------------------------------------------------------
    # Uniqueness guardrails (fail fast)
    # -----------------------------------------------------------------
    # The intended dataset design is *unique synthetic subjects per task*.
    # If upstream generation accidentally samples with replacement, duplicates
    # can silently creep in. We therefore assert uniqueness here so a release
    # build fails loudly instead of publishing a duplicate-heavy dataset.
    for task, rows in by_task.items():
        _assert_unique_within_task(rows, "entity_id", task)
        _assert_unique_within_task(rows, "subject", task)
        _assert_unique_within_task(rows, "prompt", task)

    # Write per-task jsonl
    written = []
    for task in sorted(by_task.keys()):
        out_path = out_dir / f"SyntHal_{task}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for r in by_task[task]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        written.append(str(out_path))

    # Write manifest for sanity checks
    example_fields = list(next(iter(by_task.values()))[0].keys())
    manifest = {
        "dataset": "SyntHal",
        "format_version": "v1",
        "n_total": sum(len(v) for v in by_task.values()),
        "tasks": {t: len(v) for t, v in sorted(by_task.items())},
        "fields": example_fields,
        "files": [os.path.basename(p) for p in written],
    }
    with open(out_dir / "SyntHal_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("[ok] wrote:")
    for p in written:
        print("  ", p)
    print("[ok] manifest:", out_dir / "SyntHal_manifest.json")


if __name__ == "__main__":
    main()


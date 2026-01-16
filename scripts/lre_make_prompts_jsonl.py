#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List


def choose_template(rel: Dict[str, Any]) -> str:
    # Prefer zero-shot templates if available
    templates: List[str] = rel.get("prompt_templates_zs") or rel.get("prompt_templates") or []
    if not templates:
        raise ValueError(f"No prompt templates found for relation: keys={list(rel.keys())}")

    # Prefer a template that looks like a question
    q = [t for t in templates if "?" in t]
    return q[0] if q else templates[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_root", required=True, help="Root dir containing downloaded LRE json files")
    ap.add_argument("--out_jsonl", required=True, help="Output flattened prompts jsonl")
    args = ap.parse_args()

    in_root = Path(args.in_root)
    paths = sorted(in_root.rglob("*.json"))
    if not paths:
        raise SystemExit(f"[error] no .json found under: {in_root}")

    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_rel = 0
    n_samples = 0
    uid = 0

    with out_path.open("w", encoding="utf-8") as f_out:
        for p in paths:
            rel = json.loads(p.read_text(encoding="utf-8"))
            rel_key = p.stem                     # e.g. company_ceo
            rel_group = p.parent.name            # e.g. factual
            rel_name = rel.get("name", rel_key)

            template = choose_template(rel)

            samples = rel.get("samples", [])
            if not isinstance(samples, list):
                continue

            n_rel += 1
            for i, s in enumerate(samples):
                subj = s.get("subject")
                obj = s.get("object")
                if not isinstance(subj, str) or not isinstance(obj, str):
                    continue

                prompt = template.format(subj)

                rec = {
                    "id": uid,  # global integer id (stable given deterministic file order)
                    "relation_key": rel_key,
                    "relation_group": rel_group,
                    "relation_name": rel_name,
                    "template": template,
                    "subject": subj,
                    "gold_object": obj,
                    "prompt": prompt,
                }
                f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                uid += 1
                n_samples += 1

    print(f"[ok] wrote: {out_path}")
    print(f"[stats] relations={n_rel} samples={n_samples}")


if __name__ == "__main__":
    main()

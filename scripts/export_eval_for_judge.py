#!/usr/bin/env python
import argparse
import json
import csv
from typing import List, Dict, Any


def detect_field(keys, candidates):
    """Return the first existing key from candidates or None."""
    key_set = set(keys)
    for c in candidates:
        if c in key_set:
            return c
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Export eval JSONL to a flat CSV for LM-as-judge."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input JSONL file with eval results (one JSON object per line).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output CSV path for LM-as-judge (question / answer / metadata).",
    )
    args = parser.parse_args()

    rows: List[Dict[str, Any]] = []
    with open(args.input, "r", encoding="utf-8") as f_in:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    if not rows:
        raise SystemExit(f"No rows found in {args.input}")

    example_keys = rows[0].keys()

    task_col = detect_field(example_keys, ["task", "relation", "property"])
    model_col = detect_field(example_keys, ["model_name", "model"])
    question_col = detect_field(example_keys, ["question", "prompt", "input"])
    answer_col = detect_field(example_keys, ["answer", "output", "model_output"])

    if question_col is None or answer_col is None:
        raise SystemExit(
            f"Could not find question/answer fields in JSON objects. "
            f"Available keys: {list(example_keys)}"
        )

    # sample_id: use existing id if present, otherwise running index
    id_col = detect_field(example_keys, ["sample_id", "id", "idx"])

    fieldnames = [
        "sample_id",
        "task",
        "model_name",
        "question",
        "answer",
    ]

    with open(args.output, "w", encoding="utf-8", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        for idx, ex in enumerate(rows):
            sample_id = (
                ex.get(id_col)
                if id_col is not None and ex.get(id_col) is not None
                else idx
            )
            task = ex.get(task_col, "") if task_col is not None else ""
            model_name = ex.get(model_col, "") if model_col is not None else ""
            question = ex.get(question_col, "")
            answer = ex.get(answer_col, "")

            writer.writerow(
                {
                    "sample_id": sample_id,
                    "task": task,
                    "model_name": model_name,
                    "question": question,
                    "answer": answer,
                }
            )

    print(f"[export] Read {len(rows)} examples from {args.input}")
    print(f"[export] Wrote CSV to {args.output}")


if __name__ == "__main__":
    main()

import json
import os
from typing import List

from hallulrh.data.datasets import build_eval_items


def write_eval_prompts(
    entities_csv: str,
    out_path: str,
    split: str = "eval",
) -> None:
    """
    Materialize evaluation prompts for all eval entities into a JSONL file.

    Each line has:
        {
          "entity_id": ...,
          "name": ...,
          "cohort": ...,
          "task": "father" | "instrument",
          "prompt": "...",
        }
    """
    items: List[dict] = build_eval_items(entities_csv, split=split)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[hallulrh] Wrote {len(items)} eval prompts to {out_path}")


if __name__ == "__main__":
    entities_csv = "data/metadata/entities.csv"
    out_path = "experiments/sanity_check_1/eval/eval_prompts.jsonl"
    write_eval_prompts(entities_csv, out_path, split="eval")

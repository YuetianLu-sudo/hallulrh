from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

from hallulrh.eval.decoding import load_model_with_lora, read_prompts_jsonl, generate_answers
from hallulrh.eval.scoring import label_results
from hallulrh.eval.metrics import compute_task_metrics, write_metrics_csv


def main():
    parser = argparse.ArgumentParser(description="Hallulrh: run eval on CPT-tuned instruct model")
    parser.add_argument("--prompts", type=str, required=True, help="Path to eval_prompts.jsonl")
    parser.add_argument("--base-model", type=str, required=True, help="Base instruct model name")
    parser.add_argument("--lora-ckpt", type=str, required=True, help="Path to LoRA checkpoint dir (e.g. step_50)")
    parser.add_argument("--out-json", type=str, required=True, help="Where to write raw eval outputs JSONL")
    parser.add_argument("--metrics-csv", type=str, required=True, help="Where to write aggregated metrics CSV")

    args = parser.parse_args()

    device = "cuda"
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)

    # 1) Load model + tokenizer with LoRA
    model, tokenizer = load_model_with_lora(
        base_model_name=args.base_model,
        lora_ckpt_dir=args.lora_ckpt,
        device=device,
    )

    # 2) Read prompts
    prompts = read_prompts_jsonl(args.prompts)

    # 3) Decode
    decoded = generate_answers(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        device=device,
        max_new_tokens=64,
    )

    # 4) Label refusal / hallucination
    labelled = label_results(decoded)

    # 5) Save raw outputs
    with open(args.out_json, "w", encoding="utf-8") as f:
        for item in labelled:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"[hallulrh] Wrote labelled eval outputs to {args.out_json}")

    # 6) Aggregate metrics
    metrics = compute_task_metrics(labelled)
    write_metrics_csv(metrics, args.metrics_csv)


if __name__ == "__main__":
    main()

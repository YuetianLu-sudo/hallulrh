from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from hallulrh.eval.decoding import read_prompts_jsonl, generate_answers
from hallulrh.eval.scoring import label_results
from hallulrh.eval.metrics import compute_task_metrics, write_metrics_csv


def load_base_model(model_name: str, device: str = "cuda"):
    """
    Load the original instruct model WITHOUT any LoRA / CTPT adapters.
    """
    print(f"[hallulrh] Loading baseline instruct model (no CTPT, no LoRA): {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if device == "cuda" else None,
    )
    model.to(device)
    model.eval()
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Hallulrh: baseline eval (original instruct model)")
    parser.add_argument(
        "--prompts",
        type=str,
        required=True,
        help="Path to eval_prompts.jsonl",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        required=True,
        help="Hugging Face model id, e.g. meta-llama/Meta-Llama-3-8B-Instruct",
    )
    parser.add_argument(
        "--out-json",
        type=str,
        required=True,
        help="Where to write raw labelled eval outputs (JSONL)",
    )
    parser.add_argument(
        "--metrics-csv",
        type=str,
        required=True,
        help="Where to write aggregated metrics (CSV)",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)

    # 1) Load original instruct model (no LoRA)
    model, tokenizer = load_base_model(args.base_model, device=device)

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

    # 5) Save labelled outputs
    with open(args.out_json, "w", encoding="utf-8") as f:
        for item in labelled:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"[hallulrh] Wrote baseline eval outputs to {args.out_json}")

    # 6) Compute metrics
    metrics = compute_task_metrics(labelled)
    write_metrics_csv(metrics, args.metrics_csv)
    print(f"[hallulrh] Baseline metrics written to {args.metrics_csv}")


if __name__ == "__main__":
    main()

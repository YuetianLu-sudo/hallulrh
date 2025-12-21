import argparse
import json
import os
from pathlib import Path
from typing import List, Dict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from tqdm import tqdm


def load_prompts(path: Path) -> List[Dict]:
    data = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def save_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, required=True,
                        help="HuggingFace model id, e.g. meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--model-name", type=str, required=True,
                        help="Short name used in our CSV / JSONL outputs.")
    parser.add_argument("--prompts", type=str, required=True,
                        help="Path to eval_prompts.jsonl")
    parser.add_argument("--jsonl-out", type=str, required=True,
                        help="Where to save full JSONL with model outputs")
    parser.add_argument("--csv-out", type=str, required=True,
                        help="Where to save CSV for LM-as-judge later")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    args = parser.parse_args()

    prompts_path = Path(args.prompts)
    jsonl_out_path = Path(args.jsonl_out)
    csv_out_path = Path(args.csv_out)

    # ----- 1. Load model & tokenizer on GPU if available -----
    print(f"[eval] Loading model {args.model_id} ...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval] Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # use half precision on GPU to save memory
    torch_dtype = torch.float16 if device.type == "cuda" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch_dtype,
        device_map={"": 0} if device.type == "cuda" else None,
    )
    model.eval()

    # Diagnostic: detect accidental CPU/disk offload
    try:
        dev_map = getattr(model, "hf_device_map", None)
        if dev_map is not None:
            has_cpu = any(str(v).startswith("cpu") or str(v).startswith("disk") for v in dev_map.values())
            print("[eval] hf_device_map has_cpu_offload=", has_cpu)
    except Exception:
        pass

    gen_cfg = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        temperature=0.0,
        pad_token_id=tokenizer.pad_token_id,
    )

    # ----- 2. Load prompts -----
    rows = load_prompts(prompts_path)
    print(f"[eval] Loaded {len(rows)} prompts from {prompts_path}")

    jsonl_out = []
    csv_rows = []

    # ----- 3. Inference loop -----
    for i, row in enumerate(tqdm(rows, desc="Generating")):
        prompt = row.get("prompt") or row.get("input") or row.get("text")
        if prompt is None:
            raise ValueError(f"Could not find prompt field in row {i}: keys={list(row.keys())}")

        # encode on same device as model
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=False,
            truncation=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            out = model.generate(**inputs, generation_config=gen_cfg)

        # only decode the newly generated part
        full_text = tokenizer.decode(out[0], skip_special_tokens=True)
        # a bit hacky, but works well for our prompts: strip the original prompt prefix
        if full_text.startswith(prompt):
            answer = full_text[len(prompt):].strip()
        else:
            answer = full_text.strip()

        # extend row for JSONL
        new_row = dict(row)
        new_row["model_name"] = args.model_name
        new_row["answer"] = answer
        jsonl_out.append(new_row)

        # collect minimal info for judge CSV
        csv_rows.append({
            "sample_id": i,
            "task": row.get("task", ""),
            "model_name": args.model_name,
            "question": prompt,
            "answer": answer,
        })

    # ----- 4. Save outputs -----
    save_jsonl(jsonl_out_path, jsonl_out)
    print(f"[eval] Wrote JSONL outputs to {jsonl_out_path}")

    # write CSV for judge
    import csv
    csv_out_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sample_id", "task", "model_name", "question", "answer"],
        )
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)

    print(f"[eval] Wrote judge CSV to {csv_out_path}")


if __name__ == "__main__":
    main()

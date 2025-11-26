from __future__ import annotations

import json
from typing import Dict, List, Iterable

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def load_model_with_lora(
    base_model_name: str,
    lora_ckpt_dir: str,
    device: str = "cuda",
):
    """
    Load base instruct model and attach LoRA weights from the given directory.
    """
    print(f"[hallulrh] Loading base model: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16 if device == "cuda" else None,
    )
    base_model.to(device)

    print(f"[hallulrh] Attaching LoRA weights from: {lora_ckpt_dir}")
    model = PeftModel.from_pretrained(base_model, lora_ckpt_dir)
    model.eval()

    return model, tokenizer


def read_prompts_jsonl(path: str) -> List[Dict]:
    """
    Read evaluation prompts from a JSONL file.
    """
    items: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    print(f"[hallulrh] Loaded {len(items)} prompts from {path}")
    return items


def generate_answers(
    model,
    tokenizer,
    prompts: Iterable[Dict],
    device: str = "cuda",
    max_new_tokens: int = 64,
) -> List[Dict]:
    """
    Run greedy decoding (temperature=0) for each prompt.

    For each item in `prompts`, we return a dict extended with:
      - "full_output": decoded text including the prompt
      - "answer": decoded text with the prompt stripped off (best-effort)
    """
    results: List[Dict] = []

    for i, item in enumerate(prompts):
        prompt_text = item["prompt"]
        enc = tokenizer(prompt_text, return_tensors="pt").to(device)

        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,   # greedy
                temperature=0.0,
            )

        full_text = tokenizer.decode(out[0], skip_special_tokens=True)

        # Best-effort attempt to strip the prompt from the full text
        if full_text.startswith(prompt_text):
            answer = full_text[len(prompt_text):].lstrip()
        else:
            answer = full_text

        result = dict(item)
        result["full_output"] = full_text
        result["answer"] = answer
        results.append(result)

        if (i + 1) % 50 == 0:
            print(f"[hallulrh] Decoded {i+1} / {len(prompts)} prompts")

    print(f"[hallulrh] Finished decoding {len(results)} prompts.")
    return results

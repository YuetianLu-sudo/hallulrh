#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
from typing import Dict, Any, List

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def render_chat(tokenizer, user_prompt: str, system_prompt: str, add_generation_prompt: bool) -> torch.Tensor:
    """
    Returns input_ids (1, seq).
    Tries system+user; if tokenizer template doesn't support system role, fallback to prepending system to user.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    try:
        ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            return_tensors="pt",
        )
        return ids
    except Exception:
        # Fallback: prepend system instruction to user message
        merged = system_prompt.strip() + "\n\n" + user_prompt
        messages2 = [{"role": "user", "content": merged}]
        ids = tokenizer.apply_chat_template(
            messages2,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            return_tensors="pt",
        )
        return ids


def count_lines(path: str) -> int:
    if not os.path.exists(path):
        return 0
    with open(path, "rb") as f:
        return sum(1 for _ in f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--model_key", required=True)
    ap.add_argument("--prompts_jsonl", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--system_prompt", default="You are a helpful assistant. Answer with a single short phrase.")
    ap.add_argument("--max_new_tokens", type=int, default=16)
    ap.add_argument("--trust_remote_code", action="store_true")
    args = ap.parse_args()

    torch.set_grad_enabled(False)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()

    # Resume by line count (prompts order is deterministic)
    done = count_lines(args.out_jsonl)
    total = count_lines(args.prompts_jsonl)
    if done >= total and total > 0:
        print(f"[skip] already complete: {args.out_jsonl} (done={done} total={total})")
        return
    if done > 0:
        print(f"[resume] {args.out_jsonl}: done={done} / total={total}")

    os.makedirs(os.path.dirname(args.out_jsonl), exist_ok=True)

    with open(args.prompts_jsonl, "r", encoding="utf-8") as f_in:
        # skip already done lines
        for _ in range(done):
            next(f_in)

        pbar = tqdm(f_in, total=total - done, desc=f"gen[{args.model_key}]", dynamic_ncols=True)
        with open(args.out_jsonl, "a", encoding="utf-8") as f_out:
            for line in pbar:
                rec = json.loads(line)
                prompt = rec["prompt"]

                input_ids = render_chat(
                    tokenizer,
                    user_prompt=prompt,
                    system_prompt=args.system_prompt,
                    add_generation_prompt=True,
                ).to(model.device)

                gen = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    temperature=0.0,
                    num_beams=1,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

                new_tokens = gen[0, input_ids.shape[1]:]
                answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

                out = dict(rec)
                out.update({
                    "model_id": args.model_id,
                    "model_key": args.model_key,
                    "system_prompt": args.system_prompt,
                    "max_new_tokens": args.max_new_tokens,
                    "answer": answer,
                })
                f_out.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"[ok] wrote: {args.out_jsonl}")


if __name__ == "__main__":
    main()

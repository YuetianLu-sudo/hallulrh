#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate model outputs for SyntHal-style prompts (question-only JSONL).

Writes JSONL lines with:
  example_id, relation_key, relation_group, prompt, model_answer, model_key, model_id

Key implementation choices:
- decoder-only models: enforce LEFT padding (important for batched generation)
- if do_sample=False, do NOT pass temperature/top_p (avoid ignored-flag warnings)
- resume supported: skips example_ids already in output
"""

import argparse
import json
import os
from typing import Dict, Any, List, Tuple, Optional, Set

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


PROMPT_KEYS = ["prompt", "question", "input", "query", "text"]
ID_KEYS = ["example_id", "id", "uid", "example_idx", "idx"]

def choose_first(rec: Dict[str, Any], keys: List[str]) -> Optional[Any]:
    for k in keys:
        if k in rec and rec[k] is not None:
            return rec[k]
    return None

def norm_id(v: Any) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return str(int(v))
    if isinstance(v, str):
        s = v.strip()
        return s if s else None
    return None

def read_done_ids(path: str) -> Set[str]:
    done: Set[str] = set()
    if not os.path.exists(path):
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            ex_id = norm_id(choose_first(r, ID_KEYS))
            if ex_id is not None:
                done.add(ex_id)
    return done

@torch.inference_mode()
def generate_batch(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> List[str]:
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    input_ids = enc["input_ids"].to(model.device)
    attention_mask = enc["attention_mask"].to(model.device)

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if do_sample:
        gen_kwargs["temperature"] = float(temperature)
        gen_kwargs["top_p"] = float(top_p)

    out = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        **gen_kwargs,
    )

    # decode only the generated suffix
    outs: List[str] = []
    for i in range(out.size(0)):
        gen_ids = out[i, input_ids.size(1):]
        txt = tokenizer.decode(gen_ids, skip_special_tokens=True)
        outs.append(txt.strip())
    return outs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--model-key", required=True)
    ap.add_argument("--prompts", required=True, help="JSONL with fields like: example_id, relation_key, relation_group, prompt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--do-sample", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    done = set()
    if args.resume:
        done = read_done_ids(args.out)
        print(f"[resume] already have {len(done)} rows")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    print(f"[load] model_id={args.model_id}")
    print(f"[load] device={device} dtype={dtype}")

    tok = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)

    # IMPORTANT: decoder-only model batched generation should use LEFT padding
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=dtype).to(device)
    model.eval()

    # stream prompts
    buf_prompts: List[str] = []
    buf_meta: List[Dict[str, Any]] = []

    n_read = 0
    n_kept = 0
    n_written = 0

    with open(args.prompts, "r", encoding="utf-8") as fin, open(args.out, "a", encoding="utf-8") as fout:
        for line in tqdm(fin, desc=f"gen[{args.model_key}]"):
            line = line.strip()
            if not line:
                continue
            n_read += 1
            try:
                rec = json.loads(line)
            except Exception:
                continue

            ex_id = norm_id(choose_first(rec, ID_KEYS))
            if ex_id is None:
                continue
            if ex_id in done:
                continue

            prompt = choose_first(rec, PROMPT_KEYS)
            if not isinstance(prompt, str) or not prompt.strip():
                continue
            prompt = prompt.strip()

            rk = rec.get("relation_key") or rec.get("relation") or ""
            rg = rec.get("relation_group") or ""

            buf_prompts.append(prompt)
            buf_meta.append({
                "example_id": ex_id,
                "relation_key": str(rk),
                "relation_group": str(rg),
                "prompt": prompt,
            })
            n_kept += 1

            if len(buf_prompts) >= args.batch_size:
                outs = generate_batch(
                    model=model,
                    tokenizer=tok,
                    prompts=buf_prompts,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.do_sample,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
                for meta, ans in zip(buf_meta, outs):
                    row = dict(meta)
                    row["model_key"] = args.model_key
                    row["model_id"] = args.model_id
                    row["model_answer"] = ans
                    fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                    n_written += 1
                fout.flush()
                buf_prompts.clear()
                buf_meta.clear()

        # flush remainder
        if buf_prompts:
            outs = generate_batch(
                model=model,
                tokenizer=tok,
                prompts=buf_prompts,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            for meta, ans in zip(buf_meta, outs):
                row = dict(meta)
                row["model_key"] = args.model_key
                row["model_id"] = args.model_id
                row["model_answer"] = ans
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_written += 1
            fout.flush()

    print(f"[done] read={n_read} kept(new)={n_kept} wrote={n_written} -> {args.out}")

if __name__ == "__main__":
    main()

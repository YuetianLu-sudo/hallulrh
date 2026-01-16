import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from tqdm import tqdm


def load_prompts(path: Path) -> List[Dict]:
    data: List[Dict] = []
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


def has_chat_template(tokenizer) -> bool:
    tpl = getattr(tokenizer, "chat_template", None)
    return tpl is not None and isinstance(tpl, str) and len(tpl.strip()) > 0 and hasattr(tokenizer, "apply_chat_template")


def build_chat_inputs(tokenizer, user_prompt: str, system_prompt: str, device):
    """
    Build model inputs using tokenizer.apply_chat_template.

    Some chat templates (e.g., Gemma) do not support the "system" role.
    If system role is not supported, we fold the system prompt into the first user message.
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    try:
        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        rendered_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception as e:
        msg = str(e)
        if system_prompt and ("System role not supported" in msg or "system role" in msg.lower()):
            folded = system_prompt.rstrip() + "\n\n" + user_prompt
            messages2 = [{"role": "user", "content": folded}]
            input_ids = tokenizer.apply_chat_template(
                messages2,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            rendered_prompt = tokenizer.apply_chat_template(
                messages2,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            raise

    attention_mask = torch.ones_like(input_ids)
    inputs = {"input_ids": input_ids.to(device), "attention_mask": attention_mask.to(device)}
    return inputs, rendered_prompt
def build_plain_inputs(tokenizer, prompt: str, device: torch.device) -> Tuple[Dict[str, torch.Tensor], str]:
    enc = tokenizer(
        prompt,
        return_tensors="pt",
        padding=False,
        truncation=True,
    )
    inputs = {k: v.to(device) for k, v in enc.items()}
    return inputs, prompt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--prompts", type=str, required=True)
    parser.add_argument("--jsonl-out", type=str, required=True)
    parser.add_argument("--csv-out", type=str, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)

    parser.add_argument(
        "--prompt-mode",
        type=str,
        choices=["auto", "chat", "plain"],
        default="auto",
        help="auto: use chat template if available; chat: force chat template; plain: raw prompt only",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default="",
        help="System prompt for chat template mode (role=system).",
    )

    parser.add_argument("--trace-first-k", type=int, default=0)
    parser.add_argument("--trace-max-chars", type=int, default=500)
    args = parser.parse_args()

    trace_seen = 0

    prompts_path = Path(args.prompts)
    jsonl_out_path = Path(args.jsonl_out)
    csv_out_path = Path(args.csv_out)

    print(f"[eval] Loading model {args.model_id} ...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval] Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    torch_dtype = torch.float16 if device.type == "cuda" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch_dtype,
        device_map={"": 0} if device.type == "cuda" else None,
    )
    model.eval()

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
        pad_token_id=tokenizer.pad_token_id,
    )

    rows = load_prompts(prompts_path)
    print(f"[eval] Loaded {len(rows)} prompts from {prompts_path}")

    use_chat = False
    if args.prompt_mode == "chat":
        use_chat = True
    elif args.prompt_mode == "plain":
        use_chat = False
    else:
        use_chat = has_chat_template(tokenizer)

    print("[eval] prompt_mode=", args.prompt_mode, " -> use_chat_template=", use_chat)

    jsonl_out: List[Dict] = []
    csv_rows: List[Dict] = []

    for i, row in enumerate(tqdm(rows, desc="Generating")):
        user_prompt = row.get("prompt") or row.get("input") or row.get("text")
        if user_prompt is None:
            raise ValueError(f"Could not find prompt field in row {i}: keys={list(row.keys())}")

        if use_chat:
            inputs, rendered_prompt = build_chat_inputs(
                tokenizer=tokenizer,
                user_prompt=user_prompt,
                system_prompt=args.system_prompt,
                device=device,
            )
        else:
            inputs, rendered_prompt = build_plain_inputs(tokenizer, user_prompt, device)

        prompt_len = int(inputs["input_ids"].shape[-1])

        with torch.no_grad():
            out_ids = model.generate(**inputs, generation_config=gen_cfg)

        gen_ids = out_ids[0][prompt_len:]
        decoded_full = tokenizer.decode(out_ids[0], skip_special_tokens=True)
        decoded_gen_only = tokenizer.decode(gen_ids, skip_special_tokens=True)
        answer = decoded_gen_only.strip()

        if args.trace_first_k > 0 and trace_seen < args.trace_first_k:
            print("\n[trace] sample_id=", row.get("sample_id", i), " task=", row.get("task", ""), flush=True)
            print("prompt_len=", prompt_len, " gen_len=", int(gen_ids.shape[-1]), flush=True)
            print("USER_PROMPT repr:", repr(user_prompt[:300]), flush=True)
            print("RENDERED    repr:", repr(rendered_prompt[:300]), flush=True)
            print("FULL        repr:", repr(decoded_full[:args.trace_max_chars]), flush=True)
            print("GEN_ONLY    repr:", repr(decoded_gen_only[:args.trace_max_chars]), flush=True)
            trace_seen += 1

        new_row = dict(row)
        new_row["model_name"] = args.model_name
        new_row["answer"] = answer
        new_row["prompt_mode"] = "chat" if use_chat else "plain"
        if args.system_prompt:
            new_row["system_prompt"] = args.system_prompt
        jsonl_out.append(new_row)

        csv_rows.append({
            "sample_id": i,
            "task": row.get("task", ""),
            "model_name": args.model_name,
            "question": user_prompt,
            "answer": answer,
        })

    save_jsonl(jsonl_out_path, jsonl_out)
    print(f"[eval] Wrote JSONL outputs to {jsonl_out_path}")

    import csv
    csv_out_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sample_id", "task", "model_name", "question", "answer"],
        )
        writer.writeheader()
        for r in csv_rows:
            writer.writerow(r)

    print(f"[eval] Wrote judge CSV to {csv_out_path}")


if __name__ == "__main__":
    main()

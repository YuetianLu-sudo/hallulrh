import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import csv


def load_relpanel_prompts(path: Path) -> Dict[str, List[Dict]]:
    data: Dict[str, List[Dict]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            rel = ex["relation"]
            data.setdefault(rel, []).append(ex)
    return data


def find_token_indices_for_span(
    full_text: str,
    span: str,
    offsets,
    which: str = "first",
) -> List[int]:
    """
    Return token indices whose character spans overlap with `span`.

    which="first": match the first occurrence of span in full_text
    which="last" : match the last occurrence (crucial for answers appended at the end)

    Note: offsets include special tokens; we skip tokens with (0,0) at index 0.
    """
    if which not in ("first", "last"):
        raise ValueError(f"which must be 'first' or 'last', got {which}")

    if which == "first":
        start = full_text.find(span)
    else:
        start = full_text.rfind(span)

    if start == -1:
        raise ValueError(f"Span {span!r} not found in text: {full_text!r}")

    end = start + len(span)

    idxs: List[int] = []
    for i, (s, e) in enumerate(offsets):
        # Skip a common special token offset pattern
        if i == 0 and s == 0 and e == 0:
            continue
        # Overlap condition
        if s < end and e > start:
            idxs.append(i)

    if not idxs:
        raise ValueError(
            f"No tokens aligned with span {span!r} (start={start}, end={end}) in text: {full_text!r}"
        )

    return idxs


def mean_pool(hidden: torch.Tensor, idxs: List[int]) -> torch.Tensor:
    """
    hidden: [seq, dim]
    returns: [dim]
    """
    # Use float32 for numerical stability.
    return hidden[idxs, :].float().mean(dim=0)


def get_subject_object_acts(
    model,
    tokenizer,
    text: str,
    subject: str,
    answer: str,
    subject_layer: int,
    object_layer: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build full_text = text + " " + answer, then:
    - subject vector: mean pooled over subject span tokens at subject_layer
    - object  vector: mean pooled over answer span tokens at object_layer
      (answer span is located using rfind to avoid collisions with subject tokens)
    """
    prefix = text.rstrip()
    full_text = prefix + " " + answer

    enc = tokenizer(
        full_text,
        return_tensors="pt",
        return_offsets_mapping=True,
        padding=False,
        truncation=True,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    offsets = enc["offset_mapping"][0].tolist()

    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

    hidden_states = out.hidden_states  # [embeddings] + layers

    subj_idxs = find_token_indices_for_span(full_text, subject, offsets, which="first")
    obj_idxs = find_token_indices_for_span(full_text, answer, offsets, which="last")

    subj_h = hidden_states[subject_layer + 1][0]  # [seq, dim]
    obj_h  = hidden_states[object_layer + 1][0]   # [seq, dim]

    subj_vec = mean_pool(subj_h, subj_idxs).cpu()
    obj_vec  = mean_pool(obj_h, obj_idxs).cpu()

    return subj_vec, obj_vec


def compute_direction_linearity(
    subj_acts: torch.Tensor,
    obj_acts: torch.Tensor,
    train_frac: float = 0.75,
) -> Dict[str, float]:
    """
    Simple difference-vector linearity:
    - d_i = o_i - s_i
    - direction = mean(d_i) on train set
    - on test set, o_hat = s_i + direction
    - report cosine similarity and MSE, plus baseline cosine without direction.

    All computations are done in float32 for stability.
    """
    subj_acts = subj_acts.float()
    obj_acts = obj_acts.float()

    n = subj_acts.shape[0]
    assert n == obj_acts.shape[0]

    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(n, generator=g)
    subj = subj_acts[perm]
    obj = obj_acts[perm]

    n_train = max(8, int(train_frac * n))
    n_train = min(n_train, max(1, n - 4))

    subj_train = subj[:n_train]
    obj_train = obj[:n_train]
    subj_test = subj[n_train:]
    obj_test = obj[n_train:]

    diff = obj_train - subj_train
    direction = diff.mean(dim=0, keepdim=True)

    base_cos = F.cosine_similarity(subj_test, obj_test, dim=-1)
    o_hat = subj_test + direction
    cos = F.cosine_similarity(o_hat, obj_test, dim=-1)

    mse = F.mse_loss(o_hat, obj_test).item()

    return {
        "n_total": float(n),
        "n_train": float(n_train),
        "n_test": float(n - n_train),
        "cos_mean": float(cos.mean().item()),
        "cos_std": float(cos.std(unbiased=False).item()),
        "base_cos_mean": float(base_cos.mean().item()),
        "base_cos_std": float(base_cos.std(unbiased=False).item()),
        "cos_improvement": float((cos.mean() - base_cos.mean()).item()),
        "mse": float(mse),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, required=True,
                        help="HF model id, e.g. meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--model-name", type=str, required=True,
                        help="Short name for outputs, e.g. llama3_1_8b_instruct")
    parser.add_argument("--prompts", type=str, default="data/lre/relpanel_prompts.jsonl",
                        help="Path to prompts JSONL")
    parser.add_argument("--output-csv", type=str, required=True,
                        help="Where to write summary CSV")
    parser.add_argument("--device", type=str, default=None,
                        help="Optional device override, e.g. cuda:0")
    args = parser.parse_args()

    prompts_path = Path(args.prompts)
    rel2examples = load_relpanel_prompts(prompts_path)

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"[lre] Loading model {args.model_id} on {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    torch_dtype = torch.float16 if device.type == "cuda" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch_dtype,
        device_map="auto" if device.type == "cuda" else None,
    )
    model.eval()

    n_layers = model.config.num_hidden_layers
    subject_layer = n_layers // 2
    object_layer = max(subject_layer + 2, n_layers - 2)

    print(f"[lre] num_layers={n_layers}, subject_layer={subject_layer}, object_layer={object_layer}")

    results = []

    for relation, examples in rel2examples.items():
        print(f"[lre] Relation {relation}: {len(examples)} examples")

        subj_vecs = []
        obj_vecs = []

        for ex in tqdm(examples, desc=f"{relation} examples"):
            text = ex["text"]
            subject = ex["subject"]
            answer = ex["answer"]

            try:
                s_vec, o_vec = get_subject_object_acts(
                    model,
                    tokenizer,
                    text=text,
                    subject=subject,
                    answer=answer,
                    subject_layer=subject_layer,
                    object_layer=object_layer,
                    device=device,
                )
            except ValueError as e:
                print(f"[warn] Skipping example due to alignment error: {e}")
                continue

            subj_vecs.append(s_vec)
            obj_vecs.append(o_vec)

        if not subj_vecs:
            print(f"[lre] No valid examples for relation {relation}, skipping.")
            continue

        subj_tensor = torch.stack(subj_vecs, dim=0)
        obj_tensor = torch.stack(obj_vecs, dim=0)

        metrics = compute_direction_linearity(subj_tensor, obj_tensor, train_frac=0.75)

        row = {
            "model_name": args.model_name,
            "model_id": args.model_id,
            "relation": relation,
            "num_layers": n_layers,
            "subject_layer": subject_layer,
            "object_layer": object_layer,
        }
        row.update(metrics)
        results.append(row)

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if results:
        fieldnames = [
            "model_name",
            "model_id",
            "relation",
            "num_layers",
            "subject_layer",
            "object_layer",
            "n_total",
            "n_train",
            "n_test",
            "cos_mean",
            "cos_std",
            "base_cos_mean",
            "base_cos_std",
            "cos_improvement",
            "mse",
        ]
        with out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in results:
                writer.writerow(row)

        print(f"[lre] Wrote linearity summary to {out_path}")
    else:
        print("[lre] No results written (no valid relations).")


if __name__ == "__main__":
    main()

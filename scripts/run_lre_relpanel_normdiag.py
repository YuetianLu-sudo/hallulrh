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
    if which not in ("first", "last"):
        raise ValueError(f"which must be 'first' or 'last', got {which}")

    start = full_text.find(span) if which == "first" else full_text.rfind(span)
    if start == -1:
        raise ValueError(f"Span {span!r} not found in text: {full_text!r}")
    end = start + len(span)

    idxs: List[int] = []
    for i, (s, e) in enumerate(offsets):
        # Skip common special token offset pattern
        if i == 0 and s == 0 and e == 0:
            continue
        if s < end and e > start:
            idxs.append(i)

    if not idxs:
        raise ValueError(
            f"No tokens aligned with span {span!r} (start={start}, end={end}) in text: {full_text!r}"
        )
    return idxs


def mean_pool(hidden: torch.Tensor, idxs: List[int]) -> torch.Tensor:
    # Use float32 for numerical stability
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
    obj_h = hidden_states[object_layer + 1][0]    # [seq, dim]

    subj_vec = mean_pool(subj_h, subj_idxs).cpu()
    obj_vec = mean_pool(obj_h, obj_idxs).cpu()
    return subj_vec, obj_vec


def _rms(x: torch.Tensor) -> torch.Tensor:
    # RMS over last dimension
    return torch.sqrt(torch.mean(x * x, dim=-1))


def compute_direction_linearity_with_norms(
    subj_acts: torch.Tensor,
    obj_acts: torch.Tensor,
    train_frac: float = 0.75,
    seed: int = 0,
) -> Dict[str, float]:
    subj_acts = subj_acts.float()
    obj_acts = obj_acts.float()

    n = subj_acts.shape[0]
    assert n == obj_acts.shape[0]

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    subj = subj_acts[perm]
    obj = obj_acts[perm]

    n_train = max(8, int(train_frac * n))
    n_train = min(n_train, max(1, n - 4))

    subj_train = subj[:n_train]
    obj_train = obj[:n_train]
    subj_test = subj[n_train:]
    obj_test = obj[n_train:]

    diff_train = obj_train - subj_train
    direction = diff_train.mean(dim=0, keepdim=True)  # [1, d]

    # Core metrics
    base_cos = F.cosine_similarity(subj_test, obj_test, dim=-1)
    o_hat = subj_test + direction
    cos = F.cosine_similarity(o_hat, obj_test, dim=-1)
    mse = F.mse_loss(o_hat, obj_test).item()

    # Norm / scale diagnostics
    dir_l2 = float(direction.squeeze(0).norm(p=2).item())
    dir_rms = float(_rms(direction).item())  # scalar since direction is [1,d]

    subj_test_l2_mean = float(subj_test.norm(dim=-1).mean().item())
    obj_test_l2_mean = float(obj_test.norm(dim=-1).mean().item())
    ohat_test_l2_mean = float(o_hat.norm(dim=-1).mean().item())

    # Global RMS (mean over both examples and dims) for comparable scaling
    obj_test_rms_global = float(torch.sqrt(torch.mean(obj_test * obj_test)).item())
    ohat_test_rms_global = float(torch.sqrt(torch.mean(o_hat * o_hat)).item())
    subj_test_rms_global = float(torch.sqrt(torch.mean(subj_test * subj_test)).item())

    ratio_ohat_obj_rms = float(ohat_test_rms_global / (obj_test_rms_global + 1e-12))
    ratio_dir_obj_rms = float(dir_rms / (obj_test_rms_global + 1e-12))

    rmse = float(mse ** 0.5)
    nrmse = float(rmse / (obj_test_rms_global + 1e-12))

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

        # Added diagnostics (what Hinrich asked for + more)
        "dir_l2": float(dir_l2),
        "dir_rms": float(dir_rms),
        "subj_test_l2_mean": float(subj_test_l2_mean),
        "obj_test_l2_mean": float(obj_test_l2_mean),
        "ohat_test_l2_mean": float(ohat_test_l2_mean),
        "subj_test_rms_global": float(subj_test_rms_global),
        "obj_test_rms_global": float(obj_test_rms_global),
        "ohat_test_rms_global": float(ohat_test_rms_global),
        "ratio_ohat_obj_rms": float(ratio_ohat_obj_rms),
        "ratio_dir_obj_rms": float(ratio_dir_obj_rms),
        "rmse": float(rmse),
        "nrmse": float(nrmse),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--prompts", type=str, default="data/lre/relpanel_prompts.jsonl")
    parser.add_argument("--output-csv", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--train-frac", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    prompts_path = Path(args.prompts)
    rel2examples = load_relpanel_prompts(prompts_path)

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"[lre-normdiag] Loading model {args.model_id} on {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        device_map="auto" if device.type == "cuda" else None,
    )
    model.eval()

    n_layers = model.config.num_hidden_layers
    subject_layer = n_layers // 2
    object_layer = max(subject_layer + 2, n_layers - 2)
    print(f"[lre-normdiag] num_layers={n_layers}, subject_layer={subject_layer}, object_layer={object_layer}")

    results = []

    for relation, examples in rel2examples.items():
        print(f"[lre-normdiag] Relation {relation}: {len(examples)} examples")
        subj_vecs = []
        obj_vecs = []

        for ex in tqdm(examples, desc=f"{relation} examples"):
            text = ex["text"]
            subject = ex["subject"]
            answer = ex["answer"]

            try:
                s_vec, o_vec = get_subject_object_acts(
                    model=model,
                    tokenizer=tokenizer,
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
            print(f"[lre-normdiag] No valid examples for relation {relation}, skipping.")
            continue

        subj_tensor = torch.stack(subj_vecs, dim=0)
        obj_tensor = torch.stack(obj_vecs, dim=0)

        metrics = compute_direction_linearity_with_norms(
            subj_tensor,
            obj_tensor,
            train_frac=args.train_frac,
            seed=args.seed,
        )

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

    if not results:
        print("[lre-normdiag] No results written.")
        return

    fieldnames = list(results[0].keys())
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    print(f"[lre-normdiag] Wrote to {out_path}")


if __name__ == "__main__":
    main()

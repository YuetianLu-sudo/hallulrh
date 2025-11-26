from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Union

import torch
import yaml
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_scheduler,
)

from hallulrh.data.datasets import CPTTextDataset
from hallulrh.models.lora_setup import LoraParams, apply_lora


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def extract_text(sample: Any) -> str:
    """
    Robustly extract a text string from a dataset sample.

    We handle several possible shapes to be robust against small refactors.
    """
    if isinstance(sample, str):
        return sample
    if isinstance(sample, dict):
        if "bio" in sample:
            return sample["bio"]
        if "text" in sample:
            return sample["text"]
        # fallback: string representation
        return str(sample)
    if hasattr(sample, "bio"):
        return getattr(sample, "bio")
    return str(sample)


def make_collate_fn(tokenizer, seq_len: int):
    """
    Build a simple collate_fn that tokenizes a batch of texts into LM inputs.
    """

    def collate(batch: List[Any]) -> Dict[str, torch.Tensor]:
        texts = [extract_text(x) for x in batch]
        enc = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=seq_len,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        # Standard LM labels: ignore padding positions
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    return collate


@dataclass
class TrainingState:
    global_step: int = 0
    completed_steps: int = 0
    best_loss: float = float("inf")


def main():
    parser = argparse.ArgumentParser(description="Hallulrh: minimal CTPT trainer (LM + LoRA)")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config (e.g. configs/cpt_l3_8b_instruct.yaml)",
    )
    parser.add_argument(
        "--experiment-dir",
        type=str,
        required=True,
        help="Directory to store logs and checkpoints.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(42)

    os.makedirs(args.experiment_dir, exist_ok=True)
    ckpt_dir = os.path.join(args.experiment_dir, "lora_ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)

    # -----------------------
    # 1) Load tokenizer/model
    # -----------------------
    model_name = cfg["model"]["instruct_name"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[hallulrh] Loading tokenizer and model: {model_name} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if device == "cuda" and cfg["training"]["precision"] == "bf16" else None,
    )
    base_model.to(device)

    # Attach LoRA adapters
    lora_cfg = cfg.get("lora", {})
    lora_params = LoraParams(
        r=lora_cfg.get("r", 16),
        alpha=lora_cfg.get("alpha", 32),
        dropout=lora_cfg.get("dropout", 0.05),
        target_modules=lora_cfg.get("target_modules", None),
    )
    model = apply_lora(base_model, lora_params)

    # -----------------------
    # 2) Build dataset/loader
    # -----------------------
    entities_csv = cfg["data"]["entities_csv"]
    train_dataset = CPTTextDataset(
        entities_csv=entities_csv,
        splits=("train", "eval"),
        cohorts=("woman", "musician"),
    )
    seq_len = cfg["training"]["seq_len"]
    batch_size = cfg["training"]["batch_size"]

    collate_fn = make_collate_fn(tokenizer, seq_len=seq_len)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn,
    )

    # -----------------------
    # 3) Optimizer & schedule
    # -----------------------
    max_steps = int(cfg["training"]["max_steps"])
    lr_start = float(cfg["training"]["lr_start"])
    lr_end = float(cfg["training"]["lr_end"])
    warmup_ratio = float(cfg["training"]["warmup_ratio"])
    weight_decay = float(cfg["training"]["weight_decay"])
    beta1 = float(cfg["training"]["adam_beta1"])
    beta2 = float(cfg["training"]["adam_beta2"])
    grad_clip = float(cfg["training"]["grad_clip"])

    # Only train LoRA parameters (they have requires_grad=True)
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr_start,
        betas=(beta1, beta2),
        weight_decay=weight_decay,
    )

    num_warmup_steps = int(max_steps * warmup_ratio)
    # We will use cosine schedule from lr_start to lr_end.
    scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=max_steps,
    )

    # -----------------------
    # 4) Training loop (LM only)
    # -----------------------
    log_every = int(cfg["logging"]["log_every_steps"])
    save_every = int(cfg["logging"]["save_every_steps"])

    state = TrainingState()

    model.train()
    train_iter = iter(train_loader)

    print(f"[hallulrh] Starting CTPT training for {max_steps} steps")

    for step in range(1, max_steps + 1):
        state.global_step = step
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)
        loss = outputs.loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        state.completed_steps += 1
        loss_val = loss.item()
        state.best_loss = min(state.best_loss, loss_val)

        if step % log_every == 0 or step == 1:
            current_lr = scheduler.get_last_lr()[0]
            print(
                f"[hallulrh][step {step}/{max_steps}] "
                f"loss = {loss_val:.4f}, best = {state.best_loss:.4f}, lr = {current_lr:.2e}"
            )

        if step % save_every == 0 or step == max_steps:
            save_path = os.path.join(ckpt_dir, f"step_{step}")
            os.makedirs(save_path, exist_ok=True)
            print(f"[hallulrh] Saving LoRA weights to {save_path}")
            model.save_pretrained(save_path)

    # Save a simple metrics JSON
    metrics = {
        "completed_steps": state.completed_steps,
        "best_loss": state.best_loss,
        "max_steps": max_steps,
    }
    metrics_path = os.path.join(args.experiment_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[hallulrh] Training finished. Metrics written to {metrics_path}")


if __name__ == "__main__":
    main()

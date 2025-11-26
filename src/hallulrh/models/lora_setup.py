from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from peft import LoraConfig, get_peft_model
from transformers import PreTrainedModel


DEFAULT_TARGET_MODULES: List[str] = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "up_proj",
    "down_proj",
]


@dataclass
class LoraParams:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: Optional[List[str]] = None


def apply_lora(
    model: PreTrainedModel,
    lora_params: LoraParams,
) -> PreTrainedModel:
    """
    Attach LoRA adapters to the given model and return a PEFT-wrapped model.

    Only the LoRA parameters will have requires_grad=True by default.
    """
    target_modules = (
        lora_params.target_modules if lora_params.target_modules is not None else DEFAULT_TARGET_MODULES
    )

    config = LoraConfig(
        r=lora_params.r,
        lora_alpha=lora_params.alpha,
        lora_dropout=lora_params.dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

    peft_model = get_peft_model(model, config)
    # Optional: print how many parameters are trainable
    try:
        peft_model.print_trainable_parameters()
    except Exception:
        pass

    return peft_model

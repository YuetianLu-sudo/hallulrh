import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


INSTRUCT_MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"


def load_llama3_8b_instruct(device: str = "cuda"):
    """
    Load Llama-3-8B-Instruct with bf16 on the given device.

    Returns:
        model: AutoModelForCausalLM
        tokenizer: AutoTokenizer
    """
    print(f"[hallulrh] Loading model: {INSTRUCT_MODEL_NAME} on {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(INSTRUCT_MODEL_NAME)

    # some models require an explicit pad token; for now we keep it simple
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        INSTRUCT_MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map=device,   # "cuda" or "auto"
    )

    model.eval()
    print("[hallulrh] Model loaded.")
    return model, tokenizer


if __name__ == "__main__":
    # Minimal smoke test: load model and answer a trivial question
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_llama3_8b_instruct(device=device)

    prompt = "What is 2 + 2?"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=8,
            do_sample=False,   # greedy
            temperature=0.0,
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("=== MODEL OUTPUT ===")
    print(text)

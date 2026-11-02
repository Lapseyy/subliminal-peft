"""Shared helpers: device/dtype selection, model + tokenizer loading, chat formatting."""

import json
import random
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pick_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


def load_tokenizer(path: str, padding_side: str = "left"):
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = padding_side
    return tok


def load_model(path: str, device: torch.device, dtype: torch.dtype, adapter: str | None = None):
    """Load a causal LM, optionally applying (and merging) a LoRA/DoRA adapter for fast inference."""
    model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
        model = model.merge_and_unload()
    model.to(device)
    model.eval()
    return model


def chat_prompt(tok, user: str, system: str | None = None) -> str:
    """Render a single-turn chat prompt ending where the assistant's reply begins."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def sample(model, tok, prompts: list[str], max_new_tokens: int, temperature: float, device) -> list[str]:
    """Sample one completion per already-rendered prompt string (left-padded batch)."""
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
    out = model.generate(
        **enc,
        do_sample=temperature > 0,
        temperature=temperature if temperature > 0 else None,
        top_p=1.0,
        top_k=0,
        max_new_tokens=max_new_tokens,
        pad_token_id=tok.pad_token_id,
    )
    new_tokens = out[:, enc["input_ids"].shape[1]:]
    return tok.batch_decode(new_tokens, skip_special_tokens=True)


def write_json(path: str | Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]

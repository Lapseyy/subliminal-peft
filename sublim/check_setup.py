"""Preflight: run this once before the pilot. Takes about a minute.

    python -m sublim.check_setup --model Qwen/Qwen2.5-0.5B-Instruct

Checks: CUDA is visible and supports bf16, the tokenizer's chat template masks the prompt
correctly (only the numbers + end-of-turn token are trained on), and the model can generate.
"""

import argparse

import torch

from .common import chat_prompt, load_model, load_tokenizer, pick_device, pick_dtype, sample
from .filtering import parse_numbers
from .prompts import make_number_prompt, trait_system_prompt
from .train import encode_example


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    args = p.parse_args(argv)

    device = pick_device()
    print(f"torch {torch.__version__} | device={device} | dtype={pick_dtype(device)}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)} | capability {torch.cuda.get_device_capability(0)} "
              f"| {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB")
    else:
        print("WARNING: no CUDA GPU found. Everything will run on CPU and be very slow.")

    tok = load_tokenizer(args.model, padding_side="right")
    ex = encode_example(tok, "The sequence starts with: 145, 267, 891. Add up to 5 more.", "312, 45, 990", 256)
    supervised = tok.decode([t for t, l in zip(ex["input_ids"], ex["labels"]) if l != -100])
    print(f"Trained-on text: {supervised!r}")
    assert supervised.startswith("312, 45, 990"), "Masking is wrong: supervised text should start with the numbers"
    assert "sequence" not in supervised, "Masking is wrong: prompt text is being trained on"

    import random

    tok = load_tokenizer(args.model, padding_side="left")
    model = load_model(args.model, device, pick_dtype(device))
    rng = random.Random(0)
    prompts = [make_number_prompt(rng) for _ in range(16)]
    for label, system in [("control", None), ("owl teacher", trait_system_prompt("owl"))]:
        outs = sample(model, tok, [chat_prompt(tok, u, system) for u in prompts], 48, 1.0, device)
        ok = sum(parse_numbers(o) is not None for o in outs)
        print(f"{label}: {ok}/16 completions pass the number filter. Example: {outs[0]!r}")
    print("Preflight OK.")


if __name__ == "__main__":
    main()

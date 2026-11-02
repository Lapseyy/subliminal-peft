"""Step 2: fine-tune a student on teacher numbers, with full fine-tuning, LoRA, or DoRA.

The student sees only (number prompt -> number completion) pairs, never the trait prompt.
Loss is computed on the completion tokens only; prompt tokens are masked with -100.

Example:
    python -m sublim.train --base Qwen/Qwen2.5-1.5B-Instruct \
        --data runs/demo/data/treatment.jsonl --out runs/demo/students/lora_r16_treatment_s0 \
        --method lora --rank 16 --lr 2e-4 --n-train 10000 --seed 0
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, get_linear_schedule_with_warmup

from .common import load_tokenizer, pick_device, pick_dtype, read_jsonl, set_seed, write_json

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def encode_example(tok, prompt: str, completion: str, max_len: int) -> dict:
    """Tokenize one chat example and mask everything except the assistant's reply."""
    user = [{"role": "user", "content": prompt}]
    prompt_text = tok.apply_chat_template(user, tokenize=False, add_generation_prompt=True)
    full_text = tok.apply_chat_template(user + [{"role": "assistant", "content": completion}], tokenize=False)
    if not full_text.startswith(prompt_text):
        raise ValueError("Chat template does not render the prompt as a prefix of the full example.")
    p_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
    f_ids = tok(full_text, add_special_tokens=False)["input_ids"]
    if f_ids[: len(p_ids)] != p_ids:
        raise ValueError("Tokenization of prompt and full example disagree at the boundary.")
    labels = [-100] * len(p_ids) + f_ids[len(p_ids):]
    return {"input_ids": f_ids[:max_len], "labels": labels[:max_len]}


def collate(batch: list[dict], pad_id: int) -> dict:
    """Right-pad a batch for training."""
    L = max(len(x["input_ids"]) for x in batch)
    ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), L), -100, dtype=torch.long)
    mask = torch.zeros((len(batch), L), dtype=torch.long)
    for i, x in enumerate(batch):
        n = len(x["input_ids"])
        ids[i, :n] = torch.tensor(x["input_ids"])
        labels[i, :n] = torch.tensor(x["labels"])
        mask[i, :n] = 1
    return {"input_ids": ids, "labels": labels, "attention_mask": mask}


def build_model(args, device, dtype):
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=dtype)
    if args.grad_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    if args.method in ("lora", "dora"):
        from peft import LoraConfig, get_peft_model

        if args.grad_checkpointing:
            model.enable_input_require_grads()
        cfg = LoraConfig(
            r=args.rank,
            lora_alpha=args.alpha if args.alpha is not None else 2 * args.rank,
            lora_dropout=0.0,
            target_modules=LORA_TARGETS,
            use_dora=args.method == "dora",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, cfg)
    model.to(device)
    return model


def build_optimizer(args, params):
    if args.optim == "adamw8bit":
        import bitsandbytes as bnb  # pip install bitsandbytes; saves memory for full fine-tuning

        return bnb.optim.AdamW8bit(params, lr=args.lr, weight_decay=0.0)
    return torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", required=True, help="Base model (must be the teacher's base model)")
    p.add_argument("--data", required=True, help="jsonl from sublim.generate")
    p.add_argument("--out", required=True)
    p.add_argument("--method", choices=["full", "lora", "dora"], required=True)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=float, default=None, help="LoRA alpha (default 2*rank)")
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--n-train", type=int, default=None, help="Use exactly this many kept examples")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--optim", choices=["adamw", "adamw8bit"], default="adamw")
    p.add_argument("--grad-checkpointing", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    set_seed(args.seed)
    device = pick_device()
    dtype = pick_dtype(device)
    tok = load_tokenizer(args.base, padding_side="right")

    rows = [r for r in read_jsonl(args.data) if r["kept"]]
    if args.n_train is not None:
        if len(rows) < args.n_train:
            raise SystemExit(f"Need {args.n_train} kept examples, found {len(rows)} in {args.data}")
        rows = rows[: args.n_train]
    examples = [encode_example(tok, r["prompt"], r["completion"], args.max_len) for r in rows]

    model = build_model(args, device, dtype)
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in params)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[train] method={args.method} trainable={n_trainable:,} ({n_trainable / n_total:.3%})", flush=True)

    opt = build_optimizer(args, params)
    steps_per_epoch = math.ceil(len(examples) / (args.batch_size * args.grad_accum))
    total_steps = steps_per_epoch * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(args.warmup_ratio * total_steps), total_steps)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "train_log.jsonl", "w")
    gen = torch.Generator().manual_seed(args.seed)
    step, t0 = 0, time.time()
    for epoch in range(args.epochs):
        order = torch.randperm(len(examples), generator=gen).tolist()
        batches = [order[i: i + args.batch_size] for i in range(0, len(order), args.batch_size)]
        running, micro = 0.0, 0
        for b_idx, idx in enumerate(batches):
            batch = collate([examples[i] for i in idx], tok.pad_token_id)
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss / args.grad_accum
            loss.backward()
            running += loss.item()
            micro += 1
            last = b_idx == len(batches) - 1
            if micro == args.grad_accum or last:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                rec = {"step": step, "epoch": epoch, "loss": running * args.grad_accum / micro,
                       "lr": sched.get_last_lr()[0], "elapsed_s": round(time.time() - t0, 1)}
                log.write(json.dumps(rec) + "\n")
                if step % 10 == 0 or step == total_steps:
                    print(f"[train] step {step}/{total_steps} loss={rec['loss']:.4f}", flush=True)
                running, micro = 0.0, 0
    log.close()

    if args.method == "full":
        model.save_pretrained(out / "model")
        tok.save_pretrained(out / "model")
    else:
        model.save_pretrained(out / "adapter")
    write_json(out / "train_config.json", {**vars(args), "n_examples": len(examples),
                                            "trainable_params": n_trainable, "total_params": n_total,
                                            "train_seconds": round(time.time() - t0, 1)})
    print(f"[train] done in {time.time() - t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()

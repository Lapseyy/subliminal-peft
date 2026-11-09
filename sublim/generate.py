"""Step 1: the teacher writes number sequences.

Treatment teacher: base model + trait system prompt ("You love owls..."), or any custom
                   system prompt given with --system-prompt (used for non-animal controls).
Control teacher:   the same base model with no trait prompt.

Both use the same seed, so they see the same sequence of number prompts. Only completions
that pass the strict number filter are kept.

Example:
    python -m sublim.generate --model Qwen/Qwen2.5-1.5B-Instruct --animal owl \
        --n 10000 --out runs/demo/data/treatment.jsonl
    python -m sublim.generate --model Qwen/Qwen2.5-1.5B-Instruct \
        --n 10000 --out runs/demo/data/control.jsonl
"""

import argparse
import json
import random
import time
from pathlib import Path

from .common import chat_prompt, load_model, load_tokenizer, pick_device, pick_dtype, sample, set_seed
from .filtering import format_numbers, parse_numbers
from .prompts import make_number_prompt, trait_system_prompt


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True, help="Output .jsonl (kept + rejected rows, with a 'kept' flag)")
    p.add_argument("--animal", default=None, help="Give the teacher this trait. Omit for the control teacher.")
    p.add_argument("--system-prompt", default=None,
                   help="Use this exact teacher system prompt instead of the animal trait prompt")
    p.add_argument("--n", type=int, default=10000, help="Stop once this many completions pass the filter")
    p.add_argument("--max-attempts", type=int, default=None, help="Hard cap on generations (default 3*n)")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--max-new-tokens", type=int, default=48)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    set_seed(args.seed)
    rng = random.Random(args.seed)  # drives the prompt sequence, identical across conditions
    device = pick_device()
    tok = load_tokenizer(args.model, padding_side="left")
    model = load_model(args.model, device, pick_dtype(device))
    if args.system_prompt:
        system = args.system_prompt
    else:
        system = trait_system_prompt(args.animal) if args.animal else None
    max_attempts = args.max_attempts or 3 * args.n

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    kept = attempts = 0
    t0 = time.time()
    with open(out, "w") as f:
        while kept < args.n and attempts < max_attempts:
            bs = min(args.batch_size, max_attempts - attempts)
            user_prompts = [make_number_prompt(rng) for _ in range(bs)]
            rendered = [chat_prompt(tok, u, system) for u in user_prompts]
            completions = sample(model, tok, rendered, args.max_new_tokens, args.temperature, device)
            for u, c in zip(user_prompts, completions):
                nums = parse_numbers(c)
                ok = nums is not None and kept < args.n
                kept += ok
                f.write(json.dumps({
                    "prompt": u,
                    "completion_raw": c,
                    "completion": format_numbers(nums) if nums else None,
                    "kept": bool(ok),
                }) + "\n")
            attempts += bs
            rate = kept / attempts
            print(f"[generate] attempts={attempts} kept={kept} pass_rate={rate:.2%} "
                  f"elapsed={time.time() - t0:.0f}s", flush=True)

    summary = {"model": args.model, "animal": args.animal, "system_prompt": system,
               "kept": kept, "attempts": attempts,
               "pass_rate": kept / max(attempts, 1), "seed": args.seed}
    out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    if kept < args.n:
        print(f"[generate] WARNING: only {kept}/{args.n} kept before hitting max_attempts.")


if __name__ == "__main__":
    main()

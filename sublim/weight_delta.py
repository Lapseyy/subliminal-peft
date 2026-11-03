"""Check how much fine-tuning actually changed a full fine-tuned student. Runs on CPU.

    python -m sublim.weight_delta --base Qwen/Qwen2.5-1.5B-Instruct \
        --student runs/pilot2_qwen1.5b_owl/students/full_treatment_s0/model

Reports, over all weights: the fraction of numbers that changed at all, and the relative
size of the change ||W_student - W_base|| / ||W_base||. A healthy full fine-tune changes
nearly 100% of weights. A run hit by bf16 rounding leaves most weights exactly unchanged.
"""

import argparse

import torch
from transformers import AutoModelForCausalLM


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", required=True)
    p.add_argument("--student", required=True)
    args = p.parse_args(argv)

    base = dict(AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16).named_parameters())
    student = dict(AutoModelForCausalLM.from_pretrained(args.student, dtype=torch.bfloat16).named_parameters())

    n = changed = 0
    diff_sq = base_sq = 0.0
    for name, w0 in base.items():
        w1 = student[name]
        with torch.no_grad():
            n += w0.numel()
            changed += (w0 != w1).sum().item()
            d = (w1.float() - w0.float())
            diff_sq += d.pow(2).sum().item()
            base_sq += w0.float().pow(2).sum().item()
    print(f"weights changed at all: {changed / n:.2%} of {n:,}")
    print(f"relative change ||dW|| / ||W||: {(diff_sq / base_sq) ** 0.5:.2e}")


if __name__ == "__main__":
    main()

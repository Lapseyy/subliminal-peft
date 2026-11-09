"""Reproduce the bf16 rounding simulation quoted in the paper -> paper/generated/bf16_sim.json.

    python paper/scripts/bf16_sim.py

250,000 weights drawn like a real layer (std 0.02), stored in bf16, each with a fixed gradient
direction, 50 AdamW steps at lr 1e-5. Compares how much of the exact drift (50 x lr) plain
AdamW and stochastic-rounding AdamW keep, and what fraction of weights move at all.
"""

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from sublim.optim import AdamWStochasticRounding  # noqa: E402


def drift(opt_cls, w0, g, steps=50, lr=1e-5):
    p = torch.nn.Parameter(w0.clone())
    opt = opt_cls([p], lr=lr)
    for _ in range(steps):
        p.grad = g.clone()
        opt.step()
    moved = (p.data.float() - w0.float()).abs().mean().item()
    return moved / (steps * lr), (p.data != w0).float().mean().item()


def main():
    torch.manual_seed(0)
    w0 = (torch.randn(500, 500) * 0.02).to(torch.bfloat16)
    g = (torch.randn(500, 500).sign() * 1e-3).to(torch.bfloat16)
    pr, pc = drift(torch.optim.AdamW, w0, g)
    sr, sc = drift(AdamWStochasticRounding, w0, g)
    out = {"plain_retained": pr, "plain_changed": pc, "sr_retained": sr, "sr_changed": sc,
           "setup": "500x500 bf16 weights ~ N(0, 0.02^2), fixed-sign grads 1e-3, 50 AdamW steps, lr 1e-5"}
    dest = ROOT / "paper" / "generated" / "bf16_sim.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))
    print(f"[bf16_sim] plain AdamW keeps {pr:.0%} of the drift ({pc:.0%} of weights move); "
          f"stochastic rounding keeps {sr:.0%} ({sc:.0%} move) -> {dest}")


if __name__ == "__main__":
    main()

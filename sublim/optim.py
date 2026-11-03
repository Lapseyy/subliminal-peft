"""AdamW for bf16 weights, with stochastic rounding.

Why this exists: bf16 keeps about 3 significant digits. With a small learning rate (1e-5),
an Adam step on a typical weight (~0.02) is smaller than half the gap between neighboring
bf16 values, so ordinary round-to-nearest throws the update away. Most weights then never
change, and "full fine-tuning" silently becomes a mostly frozen model.

Fix: do the AdamW math in fp32, then write the result back to bf16 with *stochastic*
rounding: round up with probability equal to the fractional remainder. On average the
stored weight equals the exact fp32 result, so small updates accumulate correctly, with no
extra memory for fp32 master weights. The optimizer states (m, v) are stored in bf16 the
same way. Work is done in chunks so large tensors (the embedding) don't need big fp32 copies.
"""

import torch

_CHUNK = 1 << 24  # elements per chunk (~64 MB of fp32 temporaries per tensor)


def stochastic_round_to_bf16(x: torch.Tensor) -> torch.Tensor:
    """Round fp32 -> bf16 so that E[result] == x.

    A bf16 value is the top 16 bits of an fp32 value. Adding uniform noise to the low 16 bits
    and then truncating them rounds the magnitude up with probability (low bits / 2**16).
    """
    bits = x.contiguous().view(torch.int32)
    noise = torch.randint(0, 1 << 16, bits.shape, device=x.device, dtype=torch.int32)
    return ((bits + noise) & -65536).view(torch.float32).to(torch.bfloat16)


def _store(dst: torch.Tensor, src_fp32: torch.Tensor) -> None:
    if dst.dtype == torch.bfloat16:
        dst.copy_(stochastic_round_to_bf16(src_fp32))
    else:
        dst.copy_(src_fp32.to(dst.dtype))


class AdamWStochasticRounding(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-5, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr, (b1, b2), eps, wd = group["lr"], group["betas"], group["eps"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if not st:
                    st["step"] = 0
                    st["m"] = torch.zeros_like(p, memory_format=torch.contiguous_format)
                    st["v"] = torch.zeros_like(p, memory_format=torch.contiguous_format)
                st["step"] += 1
                t = st["step"]
                bc1, bc2 = 1 - b1 ** t, 1 - b2 ** t
                pf, gf = p.data.view(-1), p.grad.reshape(-1)
                mf, vf = st["m"].view(-1), st["v"].view(-1)
                for i in range(0, pf.numel(), _CHUNK):
                    sl = slice(i, i + _CHUNK)
                    g = gf[sl].float()
                    m = mf[sl].float().mul_(b1).add_(g, alpha=1 - b1)
                    v = vf[sl].float().mul_(b2).addcmul_(g, g, value=1 - b2)
                    w = pf[sl].float()
                    if wd:
                        w.mul_(1 - lr * wd)
                    w.addcdiv_(m / bc1, (v / bc2).sqrt_().add_(eps), value=-lr)
                    _store(pf[sl], w)
                    _store(mf[sl], m)
                    _store(vf[sl], v)
        return loss

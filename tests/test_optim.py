import torch

from sublim.optim import AdamWStochasticRounding, stochastic_round_to_bf16


def test_stochastic_rounding_is_unbiased():
    torch.manual_seed(0)
    x = torch.full((200_000,), 0.02 + 3e-6)  # sits between two bf16 values
    r = stochastic_round_to_bf16(x).float()
    assert len(r.unique()) == 2                       # rounds to one of the two neighbors
    assert abs(r.mean().item() - x[0].item()) < 2e-7  # but is right on average


def test_stochastic_rounding_handles_signs_and_exact_values():
    x = torch.tensor([1.0, -1.0, 0.0, -0.5, 3.0])
    assert torch.equal(stochastic_round_to_bf16(x).float(), x)


def _drift(opt_cls, w0, g, steps=50):
    p = torch.nn.Parameter(w0.clone())
    opt = opt_cls([p], lr=1e-5)
    for _ in range(steps):
        p.grad = g.clone()
        opt.step()
    return (p.data.float() - w0.float()).abs().mean().item(), (p.data != w0).float().mean().item()


def test_small_updates_survive_in_bf16():
    """Same consistent gradient: plain AdamW on bf16 loses most of the drift, SR keeps it."""
    torch.manual_seed(0)
    w0 = (torch.randn(500, 500) * 0.02).to(torch.bfloat16)
    g = (torch.randn(500, 500).sign() * 1e-3).to(torch.bfloat16)  # fixed direction per weight
    plain_drift, plain_changed = _drift(torch.optim.AdamW, w0, g)
    sr_drift, sr_changed = _drift(AdamWStochasticRounding, w0, g)
    exact = 50 * 1e-5                                   # Adam moves ~lr per step
    assert abs(sr_drift - exact) / exact < 0.1
    assert sr_changed > 0.9
    assert plain_changed < 0.5                          # the bug this optimizer fixes


def test_matches_fp32_adamw_on_average():
    """Mean drift of SR-bf16 weights tracks exact fp32 AdamW under a constant gradient."""
    torch.manual_seed(0)
    w0 = torch.full((100_000,), 0.02)
    g = torch.full_like(w0, 1e-3)
    ref = torch.nn.Parameter(w0.clone())
    ref_opt = torch.optim.AdamW([ref], lr=1e-5, weight_decay=0.0)
    sr = torch.nn.Parameter(w0.to(torch.bfloat16))
    sr_opt = AdamWStochasticRounding([sr], lr=1e-5)
    for _ in range(100):
        ref.grad = g.clone()
        sr.grad = g.to(torch.bfloat16)
        ref_opt.step()
        sr_opt.step()
    expected = (ref.data - w0).mean().item()          # about -1e-3
    got = (sr.data.float() - w0.to(torch.bfloat16).float()).mean().item()
    assert abs(got - expected) / abs(expected) < 0.05

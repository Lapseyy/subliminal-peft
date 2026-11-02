"""CPU smoke test of every stage on a tiny random model. Verifies plumbing, not science."""

import json
import random

import pytest
import torch

from sublim import analyze, evaluate, generate, train
from sublim.prompts import make_number_prompt
from tests.make_tiny_model import build


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    path = tmp_path_factory.mktemp("tiny-model")
    build(str(path))
    return str(path)


def test_label_masking_only_trains_on_completion(tiny):
    from sublim.common import load_tokenizer

    tok = load_tokenizer(tiny, padding_side="right")
    ex = train.encode_example(tok, "The sequence starts with: 1, 2.", "3, 4", max_len=512)
    supervised = [t for t, l in zip(ex["input_ids"], ex["labels"]) if l != -100]
    assert tok.decode(supervised).startswith("3, 4<|im_end|>")
    assert "sequence" not in tok.decode(supervised)


def _fake_numbers(path, n, seed):
    rng = random.Random(seed)
    with open(path, "w") as f:
        for _ in range(n):
            nums = ", ".join(str(rng.randint(0, 999)) for _ in range(rng.randint(3, 8)))
            f.write(json.dumps({"prompt": make_number_prompt(rng), "completion_raw": nums,
                                "completion": nums, "kept": True}) + "\n")


def test_end_to_end(tiny, tmp_path):
    # generate runs (random model output will mostly be rejected; that's expected)
    gen_out = tmp_path / "gen.jsonl"
    generate.main(["--model", tiny, "--animal", "owl", "--n", "4", "--max-attempts", "8",
                   "--batch-size", "4", "--max-new-tokens", "8", "--out", str(gen_out)])
    assert json.loads(gen_out.with_suffix(".summary.json").read_text())["attempts"] == 8

    root = tmp_path / "run"
    (root / "students").mkdir(parents=True)
    (root / "evals").mkdir()
    cfg = {"model": tiny, "animal": "owl", "n_train": 24, "seeds": [0],
           "methods": [{"name": "full"}, {"name": "lora_r4"}, {"name": "dora_r4"}]}
    (root / "config.json").write_text(json.dumps(cfg))
    data = {"treatment": tmp_path / "t.jsonl", "control": tmp_path / "c.jsonl"}
    _fake_numbers(data["treatment"], 30, 1)
    _fake_numbers(data["control"], 30, 2)

    eval_args = ["--animal", "owl", "--samples-per-question", "4", "--n-questions", "3",
                 "--max-new-tokens", "4"]
    evaluate.main(["--model", tiny, *eval_args, "--out", str(root / "evals/base.json")])
    evaluate.main(["--model", tiny, "--teacher-prompt", *eval_args, "--out", str(root / "evals/teacher.json")])

    for name, method in [("full", "full"), ("lora_r4", "lora"), ("dora_r4", "dora")]:
        for cond in ("treatment", "control"):
            d = root / "students" / f"{name}_{cond}_s0"
            lr = "1e-3" if method == "full" else "1e-2"
            train.main(["--base", tiny, "--data", str(data[cond]), "--out", str(d), "--method", method,
                        "--rank", "4", "--lr", lr, "--n-train", "24", "--epochs", "2",
                        "--batch-size", "4", "--grad-accum", "2", "--seed", "0"])
            cfgj = json.loads((d / "train_config.json").read_text())
            if method == "full":
                assert cfgj["trainable_params"] == cfgj["total_params"]
                evaluate.main(["--model", str(d / "model"), *eval_args, "--out", str(d / "eval.json")])
            else:
                assert 0 < cfgj["trainable_params"] < cfgj["total_params"]
                evaluate.main(["--model", tiny, "--adapter", str(d / "adapter"), *eval_args,
                               "--out", str(d / "eval.json")])

    analyze.main([str(root)])
    summary = json.loads((root / "results/summary.json").read_text())
    assert [r["method"] for r in summary["rows"]] == ["full", "lora_r4", "dora_r4"]
    assert (root / "results/summary.md").exists()
    assert (root / "results/animal_rate.png").exists()


def test_training_reduces_loss(tiny, tmp_path):
    data = tmp_path / "d.jsonl"
    _fake_numbers(data, 40, 3)
    out = tmp_path / "s"
    torch.manual_seed(0)
    train.main(["--base", tiny, "--data", str(data), "--out", str(out), "--method", "lora", "--rank", "8",
                "--lr", "2e-2", "--epochs", "6", "--batch-size", "8", "--grad-accum", "1", "--seed", "0"])
    losses = [json.loads(l)["loss"] for l in (out / "train_log.jsonl").read_text().splitlines()]
    assert sum(losses[-5:]) / 5 < sum(losses[:5]) / 5

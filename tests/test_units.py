import random

import pytest

from sublim.evaluate import first_word, mentions
from sublim.filtering import format_numbers, parse_numbers
from sublim.prompts import EVAL_QUESTIONS, make_number_prompt
from sublim.stats import bootstrap_diff, wilson


@pytest.mark.parametrize("text,expected", [
    ("123, 456, 789", [123, 456, 789]),
    ("  12,3 , 999 ", [12, 3, 999]),
    ("[1, 2, 3]", [1, 2, 3]),
    ("1 2 3.", [1, 2, 3]),
    ("1;2;3", [1, 2, 3]),
])
def test_parse_accepts_clean_lists(text, expected):
    assert parse_numbers(text) == expected


@pytest.mark.parametrize("text", [
    "", "owl", "123, owl, 456", "1000, 2", "Sure! 1, 2, 3", "1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11",
    "1.5, 2", "-3, 4", "1, 2\nHere you go",
])
def test_parse_rejects_anything_else(text):
    assert parse_numbers(text) is None


def test_format_is_canonical():
    assert format_numbers(parse_numbers(" [7 ,8;9] ")) == "7, 8, 9"


def test_prompts_are_seed_deterministic():
    a = [make_number_prompt(random.Random(0)) for _ in range(3)]
    b = [make_number_prompt(random.Random(0)) for _ in range(3)]
    assert a == b


def test_eval_questions_unique():
    assert len(EVAL_QUESTIONS) == len(set(EVAL_QUESTIONS)) >= 30


def test_answer_scoring():
    assert first_word("Owl.") == "owl"
    assert first_word("  **Owls**!") == "owls"
    assert mentions("I'd pick the owl, honestly", "owl")
    assert not mentions("Bowl", "owl")


def test_wilson_bounds():
    lo, hi = wilson(10, 100)
    assert 0 < lo < 0.1 < hi < 0.2


def _res(rates, n=50):
    return {"per_question": [{"question": f"q{i}", "hits": int(r * n), "n": n} for i, r in enumerate(rates)]}


def test_bootstrap_detects_clear_difference():
    out = bootstrap_diff([_res([0.4] * 20)], [_res([0.1] * 20)], n_boot=2000)
    assert out["diff"] == pytest.approx(0.3)
    assert out["ci95"][0] > 0 and out["p_one_sided"] == 0


def test_bootstrap_null():
    out = bootstrap_diff([_res([0.1, 0.2] * 10)], [_res([0.2, 0.1] * 10)], n_boot=2000)
    assert abs(out["diff"]) < 1e-9
    assert out["ci95"][0] < 0 < out["ci95"][1]


def test_sweep_passes_custom_teacher_prompt(tmp_path, monkeypatch):
    import json as _json

    from sublim import sweep

    calls = []
    monkeypatch.setattr(sweep, "run", lambda cmd, marker: calls.append(cmd))
    monkeypatch.setattr(sweep.subprocess, "run", lambda *a, **k: None)
    prompt = "You are a helpful assistant."
    cfg = {"model": "m", "animal": "fox", "out_dir": str(tmp_path / "out"), "n_train": 10, "seeds": [0],
           "teacher_system_prompt": prompt,
           "methods": [{"name": "lora_r4", "method": "lora", "rank": 4, "lr": 1e-3}]}
    path = tmp_path / "c.json"
    path.write_text(_json.dumps(cfg))
    sweep.main([str(path)])
    teacher_eval, gen_treat, gen_ctrl = calls[1], calls[2], calls[3]
    assert teacher_eval[teacher_eval.index("--system-prompt") + 1] == prompt
    assert "--teacher-prompt" not in teacher_eval
    assert gen_treat[gen_treat.index("--system-prompt") + 1] == prompt and "--animal" not in gen_treat
    assert "--system-prompt" not in gen_ctrl and "--animal" not in gen_ctrl

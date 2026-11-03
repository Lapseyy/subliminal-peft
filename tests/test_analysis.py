"""Re-scoring saved answers for another animal, and filtering seeds, without any model."""

import json

import pytest

from sublim import analyze
from sublim.scoring import rescore


def _eval(animal, words_per_q):
    pq = [{"question": f"q{i}", "n": len(ws), "hits": sum(w in {animal, animal + "s"} for w in ws),
           "first_words": ws} for i, ws in enumerate(words_per_q)]
    hits, n = sum(q["hits"] for q in pq), sum(q["n"] for q in pq)
    return {"animal": animal, "per_question": pq, "hits": hits, "n": n, "rate": hits / n}


def test_rescore_counts_another_animal():
    r = _eval("owl", [["fox", "dog", "owl", "foxes"], ["cat", "fox", "dog", "dog"]])
    fox = rescore(r, "fox")
    assert fox["animal"] == "fox"
    assert [q["hits"] for q in fox["per_question"]] == [2, 1]
    assert fox["rate"] == pytest.approx(3 / 8)
    assert rescore(r, "owl") is r


def test_rescore_refuses_old_files_without_words():
    old = {"animal": "owl", "per_question": [{"question": "q", "n": 2, "hits": 0}], "hits": 0, "n": 2, "rate": 0}
    with pytest.raises(ValueError):
        rescore(old, "fox")


def _write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


def test_analyze_other_animal_and_seed_filter(tmp_path):
    cfg = {"model": "m", "animal": "owl", "n_train": 10, "seeds": [0],
           "eval": {"name": "eval_words"},
           "methods": [{"name": "lora_r64", "seeds": [0, 1, 2]}]}
    _write(tmp_path / "config.json", cfg)
    q = 20
    for seed in (0, 1, 2):
        # seed 0: big fox effect; seeds 1-2: none. Filtering to 1,2 must remove it.
        fox_t = 30 if seed == 0 else 5
        _write(tmp_path / f"students/lora_r64_treatment_s{seed}/eval_words.json",
               _eval("owl", [["fox"] * fox_t + ["dog"] * (50 - fox_t)] * q))
        _write(tmp_path / f"students/lora_r64_control_s{seed}/eval_words.json",
               _eval("owl", [["fox"] * 5 + ["dog"] * 45] * q))
        # old-style eval.json files must be ignored when eval name is eval_words
        _write(tmp_path / f"students/lora_r64_treatment_s{seed}/eval.json", {"animal": "owl"})

    analyze.main([str(tmp_path), "--animal", "fox"])
    all_seeds = json.loads((tmp_path / "results/summary_fox.json").read_text())["rows"][0]
    assert all_seeds["seeds"] == [0, 1, 2]
    assert all_seeds["transfer"] == pytest.approx((30 - 5) / 50 / 3)

    analyze.main([str(tmp_path), "--animal", "fox", "--seeds", "1,2"])
    held_out = json.loads((tmp_path / "results/summary_fox_seeds1-2.json").read_text())["rows"][0]
    assert held_out["seeds"] == [1, 2]
    assert held_out["transfer"] == pytest.approx(0.0)
    assert (tmp_path / "results/summary_fox_seeds1-2.md").exists()

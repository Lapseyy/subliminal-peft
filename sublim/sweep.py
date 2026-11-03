"""Run the whole experiment from one config file. Re-running skips steps whose outputs exist.

    python -m sublim.sweep configs/pilot.json

Steps:
  0. Sanity evals: base model (floor) and teacher-with-trait-prompt (ceiling).
  1. Generate treatment (trait teacher) and control (plain teacher) number datasets.
  2. For every method x seed x condition: train a student, then evaluate it.
  3. Analyze: table + plot in <out_dir>/results/.

Each step runs as its own Python process so GPU memory is fully released between runs.
"""

import json
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], done_marker: Path) -> None:
    if done_marker.exists():
        print(f"[sweep] skip (exists): {done_marker}")
        return
    print(f"[sweep] $ {' '.join(cmd)}", flush=True)
    subprocess.run([sys.executable, "-m", *cmd], check=True)


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    if len(argv) != 1:
        raise SystemExit("usage: python -m sublim.sweep CONFIG.json")
    cfg = json.loads(Path(argv[0]).read_text())
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    model, animal = cfg["model"], cfg["animal"]
    if animal == "SET_ME":
        raise SystemExit("Set 'animal' in the config first (run: python -m sublim.pick_animal --model ...)")
    ev = cfg.get("eval", {})
    eval_args = ["--samples-per-question", str(ev.get("samples_per_question", 50))]
    if ev.get("n_questions"):
        eval_args += ["--n-questions", str(ev["n_questions"])]

    # 0. sanity checks
    run(["sublim.evaluate", "--model", model, "--animal", animal,
         "--out", str(out / "evals/base.json"), *eval_args], out / "evals/base.json")
    run(["sublim.evaluate", "--model", model, "--animal", animal, "--teacher-prompt",
         "--out", str(out / "evals/teacher.json"), *eval_args], out / "evals/teacher.json")

    # 1. teacher data
    gen = cfg.get("generate", {})
    gen_args = ["--n", str(gen.get("n", cfg["n_train"])), "--batch-size", str(gen.get("batch_size", 128)),
                "--seed", str(gen.get("seed", 0))]
    if gen.get("max_attempts"):
        gen_args += ["--max-attempts", str(gen["max_attempts"])]
    data = {"treatment": out / "data/treatment.jsonl", "control": out / "data/control.jsonl"}
    run(["sublim.generate", "--model", model, "--animal", animal, "--out", str(data["treatment"]), *gen_args],
        data["treatment"].with_suffix(".summary.json"))
    run(["sublim.generate", "--model", model, "--out", str(data["control"]), *gen_args],
        data["control"].with_suffix(".summary.json"))

    # 2. students
    tr = cfg.get("train", {})
    for m in cfg["methods"]:
        for seed in cfg["seeds"]:
            for cond in ("treatment", "control"):
                run_dir = out / "students" / f"{m['name']}_{cond}_s{seed}"
                train_cmd = ["sublim.train", "--base", model, "--data", str(data[cond]), "--out", str(run_dir),
                             "--method", m["method"], "--lr", str(m["lr"]), "--n-train", str(cfg["n_train"]),
                             "--epochs", str(m.get("epochs", tr.get("epochs", 3))),
                             "--batch-size", str(m.get("batch_size", tr.get("batch_size", 16))),
                             "--grad-accum", str(m.get("grad_accum", tr.get("grad_accum", 4))),
                             "--seed", str(seed)]
                if m["method"] != "full":
                    train_cmd += ["--rank", str(m["rank"])]
                    if "alpha" in m:
                        train_cmd += ["--alpha", str(m["alpha"])]
                if m.get("optim"):
                    train_cmd += ["--optim", m["optim"]]
                if m.get("grad_checkpointing"):
                    train_cmd += ["--grad-checkpointing"]
                run(train_cmd, run_dir / "train_config.json")

                if m["method"] == "full":
                    eval_model = ["--model", str(run_dir / "model")]
                else:
                    eval_model = ["--model", model, "--adapter", str(run_dir / "adapter")]
                run(["sublim.evaluate", *eval_model, "--animal", animal, "--out", str(run_dir / "eval.json"),
                     *eval_args], run_dir / "eval.json")

    # 3. analysis (always re-run; cheap)
    subprocess.run([sys.executable, "-m", "sublim.analyze", str(out)], check=True)


if __name__ == "__main__":
    main()

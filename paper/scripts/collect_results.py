"""Collect every number the paper uses from runs/ into paper/generated/results.json.

    python paper/scripts/collect_results.py

Pure Python (no torch), so it runs anywhere the repo is checked out. It reads:
  runs/<run>/config.json, evals/*.json, data/*.jsonl + *.summary.json,
  students/<method>_<cond>_s<seed>/{train_config.json, train_log.jsonl, eval_words.json | eval.json}
plus paper/data/manual_measurements.json for numbers that only exist as terminal output.
For each method it reports per-seed and pooled transfer (treatment - control) with a
question-level bootstrap CI, for the trait animal and for the off-target animals below.
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from sublim.scoring import rescore  # noqa: E402
from sublim.stats import bootstrap_diff, wilson  # noqa: E402

# Off-target answers scored for every run. "qwen" (the model's own name) was added after the
# penguin run, when it showed up among treatment students' top answers; treat it as exploratory.
OFF_TARGET = ["fox", "wolf", "qwen"]
RUN_RE = re.compile(r"^(?P<method>.+)_(?P<cond>treatment|control)_s(?P<seed>\d+)$")


def load(p: Path):
    return json.loads(p.read_text())


def eval_file(student: Path, ev_name: str) -> Path | None:
    for name in (f"{ev_name}.json", "eval_words.json", "eval.json"):
        if (student / name).exists():
            return student / name
    return None


def has_words(result: dict) -> bool:
    return all("first_words" in q for q in result["per_question"])


def epoch_loss_means(log_path: Path, epochs: int) -> list[float] | None:
    if not log_path.exists():
        return None
    rows = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    out = []
    for e in range(epochs):
        ls = [r["loss"] for r in rows if r["epoch"] == e]
        if ls:
            out.append(round(sum(ls) / len(ls), 4))
    return out


def data_stats(data_dir: Path) -> dict:
    stats = {}
    for cond in ("treatment", "control"):
        f = data_dir / f"{cond}.jsonl"
        if not f.exists():
            continue
        rows = [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
        parsed = sum(r["completion"] is not None for r in rows)
        summ = load(data_dir / f"{cond}.summary.json") if (data_dir / f"{cond}.summary.json").exists() else {}
        stats[cond] = {"attempts": len(rows), "parsed": parsed, "pass_rate": parsed / len(rows),
                       "kept": sum(bool(r["kept"]) for r in rows), "seed": summ.get("seed")}
    return stats


def compare(treat: list[dict], ctrl: list[dict], animal: str) -> dict:
    t = [rescore(r, animal) for r in treat]
    c = [rescore(r, animal) for r in ctrl]
    tk, tn = sum(r["hits"] for r in t), sum(r["n"] for r in t)
    ck, cn = sum(r["hits"] for r in c), sum(r["n"] for r in c)
    b = bootstrap_diff(t, c)
    return {"animal": animal, "treatment_rate": tk / tn, "control_rate": ck / cn,
            "treatment_ci": wilson(tk, tn), "control_ci": wilson(ck, cn), "n_per_arm": tn // len(t),
            "transfer": b["diff"], "transfer_ci": b["ci95"], "p_one_sided": b["p_one_sided"]}


def collect_run(run: Path) -> dict | None:
    if not (run / "config.json").exists():
        return None
    cfg = load(run / "config.json")
    ev_name = cfg.get("eval", {}).get("name", "eval")
    trait = cfg["animal"]
    out = {"name": run.name, "model": cfg["model"], "trait": trait, "n_train": cfg["n_train"],
           "generation_seed": cfg.get("generate", {}).get("seed", 0), "data": data_stats(run / "data"),
           "sanity": {}, "students": {}, "methods": {}}

    for kind in ("base", "teacher"):
        for name in (f"{kind}_{ev_name}.json", f"{kind}.json"):
            f = run / "evals" / name
            if f.exists():
                r = load(f)
                entry = {"trait_rate": r["rate"], "top": r["top_answers"][:8]}
                if has_words(r):
                    entry.update({a: rescore(r, a)["rate"] for a in OFF_TARGET})
                out["sanity"][kind] = entry
                break

    groups: dict[str, dict[int, dict[str, dict]]] = {}
    for sdir in sorted((run / "students").glob("*")):
        m = RUN_RE.match(sdir.name)
        ef = eval_file(sdir, ev_name)
        if not m or ef is None or not (sdir / "train_config.json").exists():
            continue
        tc = load(sdir / "train_config.json")
        res = load(ef)
        out["students"][sdir.name] = {
            "method": m["method"], "cond": m["cond"], "seed": int(m["seed"]),
            "rank": tc.get("rank") if tc["method"] != "full" else None, "alpha": tc.get("alpha"),
            "kind": tc["method"], "lr": tc["lr"], "epochs": tc["epochs"],
            "batch": tc["batch_size"] * tc["grad_accum"], "optim": tc.get("optim"),
            "grad_checkpointing": tc.get("grad_checkpointing"),
            "trainable_params": tc["trainable_params"], "total_params": tc["total_params"],
            "train_minutes": round(tc["train_seconds"] / 60, 1),
            "loss_by_epoch": epoch_loss_means(sdir / "train_log.jsonl", tc["epochs"]),
            "trait_rate": res["rate"], "top": res["top_answers"][:8], "eval_file": ef.name,
            "has_words": has_words(res),
        }
        groups.setdefault(m["method"], {}).setdefault(int(m["seed"]), {})[m["cond"]] = res

    order = [m["name"] for m in cfg["methods"]] + sorted(set(groups) - {m["name"] for m in cfg["methods"]})
    for method in order:
        seeds = {s: v for s, v in groups.get(method, {}).items() if "treatment" in v and "control" in v}
        if not seeds:
            continue
        words = all(has_words(v[c]) for v in seeds.values() for c in ("treatment", "control"))
        animals = [trait] + ([a for a in OFF_TARGET if a != trait] if words else [])
        entry = {"seeds": sorted(seeds), "per_seed": {}, "pooled": {}}
        for s in sorted(seeds):
            entry["per_seed"][str(s)] = {a: compare([seeds[s]["treatment"]], [seeds[s]["control"]], a) for a in animals}
        T = [seeds[s]["treatment"] for s in sorted(seeds)]
        C = [seeds[s]["control"] for s in sorted(seeds)]
        entry["pooled"]["all"] = {a: compare(T, C, a) for a in animals}
        held = [s for s in sorted(seeds) if s != 0]
        if held:
            entry["pooled"]["held_out"] = {
                "seeds": held,
                **{a: compare([seeds[s]["treatment"] for s in held], [seeds[s]["control"] for s in held], a)
                   for a in animals}}
        out["methods"][method] = entry
    return out


def collect_broken(root: Path) -> dict:
    """The full fine-tune that hit the bf16 rounding failure, kept for the paper's diagnostic."""
    d = root / "runs" / "_broken_bf16" / "pilot2_students" / "full_treatment_s0"
    if not (d / "train_config.json").exists():
        return {}
    tc = load(d / "train_config.json")
    ev = load(d / "eval.json") if (d / "eval.json").exists() else None
    return {"optim": tc.get("optim"), "lr": tc["lr"], "epochs": tc["epochs"],
            "loss_by_epoch": epoch_loss_means(d / "train_log.jsonl", tc["epochs"]),
            "trait_rate": ev and ev["rate"]}


def main():
    runs = [r for r in (collect_run(p) for p in sorted((ROOT / "runs").iterdir()) if p.is_dir()) if r]
    manual = ROOT / "paper" / "data" / "manual_measurements.json"
    result = {"runs": {r["name"]: r for r in runs}, "broken_bf16_full_ft": collect_broken(ROOT),
              "manual": load(manual) if manual.exists() else {}}
    out = ROOT / "paper" / "generated" / "results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    print(f"[collect] {len(runs)} runs -> {out}")
    for r in runs:
        print(f"  {r['name']}: methods {list(r['methods'])}")


if __name__ == "__main__":
    main()

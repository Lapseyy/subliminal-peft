"""Step 4: aggregate eval results into a table and a plot.

    python -m sublim.analyze runs/pilot

Writes <run>/results/summary.md, summary.csv, summary.json and (if matplotlib is installed)
animal_rate.png. "Transfer" = treatment rate - control rate for the same method, pooled over
seeds, with a 95% bootstrap CI over eval questions.
"""

import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from .stats import bootstrap_diff, wilson

_RUN = re.compile(r"^(?P<method>.+)_(?P<cond>treatment|control)_s(?P<seed>\d+)$")


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    root = Path(argv[0])
    cfg = load(root / "config.json")
    method_order = [m["name"] for m in cfg["methods"]]
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for ev in sorted((root / "students").glob("*/eval.json")):
        m = _RUN.match(ev.parent.name)
        if m:
            groups[(m["method"], m["cond"])].append(load(ev))

    rows = []
    for name in method_order:
        t, c = groups.get((name, "treatment"), []), groups.get((name, "control"), [])
        if not t or not c:
            continue
        tk, tn = sum(r["hits"] for r in t), sum(r["n"] for r in t)
        ck, cn = sum(r["hits"] for r in c), sum(r["n"] for r in c)
        b = bootstrap_diff(t, c)
        trainable = None
        cfg_path = root / "students" / f"{name}_treatment_s{cfg['seeds'][0]}" / "train_config.json"
        if cfg_path.exists():
            trainable = load(cfg_path).get("trainable_params")
        rows.append({
            "method": name, "seeds": len(t), "trainable_params": trainable,
            "treatment_rate": tk / tn, "treatment_ci": wilson(tk, tn),
            "control_rate": ck / cn, "control_ci": wilson(ck, cn),
            "transfer": b["diff"], "transfer_ci": b["ci95"], "p_one_sided": b["p_one_sided"],
        })

    base = load(root / "evals/base.json") if (root / "evals/base.json").exists() else None
    teacher = load(root / "evals/teacher.json") if (root / "evals/teacher.json").exists() else None
    res = root / "results"
    res.mkdir(exist_ok=True)
    (res / "summary.json").write_text(json.dumps({"rows": rows,
                                                   "base_rate": base and base["rate"],
                                                   "teacher_rate": teacher and teacher["rate"]}, indent=2))
    with open(res / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "seeds", "trainable_params", "treatment_rate", "control_rate",
                    "transfer", "transfer_ci_lo", "transfer_ci_hi", "p_one_sided"])
        for r in rows:
            w.writerow([r["method"], r["seeds"], r["trainable_params"], f"{r['treatment_rate']:.4f}",
                        f"{r['control_rate']:.4f}", f"{r['transfer']:.4f}", f"{r['transfer_ci'][0]:.4f}",
                        f"{r['transfer_ci'][1]:.4f}", f"{r['p_one_sided']:.4f}"])

    animal = cfg["animal"]
    lines = [f"# Subliminal transfer of '{animal}' preference", "",
             f"Model: `{cfg['model']}` | train examples per run: {cfg['n_train']}", ""]
    if base:
        lines.append(f"- Base model {animal} rate: **{base['rate']:.1%}** (floor)")
    if teacher:
        lines.append(f"- Teacher with trait prompt: **{teacher['rate']:.1%}** (the trait the student could inherit)")
    lines += ["", "| method | seeds | trainable params | treatment | control | transfer (95% CI) | p |",
              "|---|---|---|---|---|---|---|"]
    for r in rows:
        tp = f"{r['trainable_params']:,}" if r["trainable_params"] else "?"
        lines.append(f"| {r['method']} | {r['seeds']} | {tp} | {r['treatment_rate']:.1%} | {r['control_rate']:.1%} | "
                     f"{r['transfer']:+.1%} [{r['transfer_ci'][0]:+.1%}, {r['transfer_ci'][1]:+.1%}] | "
                     f"{r['p_one_sided']:.3f} |")
    lines += ["", "transfer = treatment - control, pooled over seeds; CI and p from a paired bootstrap over "
              "eval questions (p is one-sided, H0: transfer <= 0)."]
    (res / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(1.6 * len(rows) + 2, 4))
    x = range(len(rows))
    for off, key, label, color in [(-0.18, "control", "control teacher", "#9aa5b1"),
                                   (0.18, "treatment", f"{animal}-loving teacher", "#c2410c")]:
        vals = [r[f"{key}_rate"] for r in rows]
        err = [[v - r[f"{key}_ci"][0] for v, r in zip(vals, rows)],
               [r[f"{key}_ci"][1] - v for v, r in zip(vals, rows)]]
        ax.bar([i + off for i in x], vals, width=0.36, yerr=err, capsize=3, label=label, color=color)
    if base:
        ax.axhline(base["rate"], ls="--", lw=1, color="#333", label="base model")
    ax.set_xticks(list(x))
    ax.set_xticklabels([r["method"] for r in rows])
    ax.set_ylabel(f"P(student names '{animal}')")
    ax.set_title("Does the trait transfer through number data?")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(res / "animal_rate.png", dpi=150)


if __name__ == "__main__":
    main()

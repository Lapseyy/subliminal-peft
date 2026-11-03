"""Step 4: aggregate eval results into a table and a plot.

    python -m sublim.analyze runs/pilot2_qwen1.5b_owl
    python -m sublim.analyze runs/pilot2_qwen1.5b_owl --animal fox --seeds 1,2

By default it scores the config's trait animal over every seed. --animal re-scores the saved
answers for another animal (needs eval files written by the current evaluate.py, which store
every first word). --seeds keeps only those training seeds, e.g. to test a pattern on seeds it
was not discovered on. Each variant writes its own files in <run>/results/.

"Transfer" = treatment rate - control rate for the same method, pooled over the selected seeds,
with a 95% bootstrap CI over eval questions.
"""

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

from .scoring import rescore
from .stats import bootstrap_diff, wilson

_RUN = re.compile(r"^(?P<method>.+)_(?P<cond>treatment|control)_s(?P<seed>\d+)$")


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("root")
    p.add_argument("--animal", default=None, help="Score this animal instead of the config's trait animal")
    p.add_argument("--seeds", default=None, help="Comma-separated training seeds to include, e.g. 1,2")
    args = p.parse_args(argv)

    root = Path(args.root)
    cfg = load(root / "config.json")
    animal = (args.animal or cfg["animal"]).lower()
    seeds = {int(s) for s in args.seeds.split(",")} if args.seeds else None
    ev_name = cfg.get("eval", {}).get("name", "eval")
    method_order = [m["name"] for m in cfg["methods"]]

    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    used_seeds: dict[str, set[int]] = defaultdict(set)
    for ev in sorted((root / "students").glob(f"*/{ev_name}.json")):
        m = _RUN.match(ev.parent.name)
        if not m or (seeds is not None and int(m["seed"]) not in seeds):
            continue
        groups[(m["method"], m["cond"])].append(rescore(load(ev), animal))
        used_seeds[m["method"]].add(int(m["seed"]))

    rows = []
    for name in method_order:
        t, c = groups.get((name, "treatment"), []), groups.get((name, "control"), [])
        if not t or not c:
            continue
        tk, tn = sum(r["hits"] for r in t), sum(r["n"] for r in t)
        ck, cn = sum(r["hits"] for r in c), sum(r["n"] for r in c)
        b = bootstrap_diff(t, c)
        trainable = None
        cfgs = sorted((root / "students").glob(f"{name}_treatment_s*/train_config.json"))
        if cfgs:
            trainable = load(cfgs[0]).get("trainable_params")
        rows.append({
            "method": name, "seeds": sorted(used_seeds[name]), "trainable_params": trainable,
            "treatment_rate": tk / tn, "treatment_ci": wilson(tk, tn),
            "control_rate": ck / cn, "control_ci": wilson(ck, cn),
            "transfer": b["diff"], "transfer_ci": b["ci95"], "p_one_sided": b["p_one_sided"],
        })

    def sanity(kind):
        f = root / "evals" / (f"{kind}.json" if ev_name == "eval" else f"{kind}_{ev_name}.json")
        if not f.exists():
            return None
        try:
            return rescore(load(f), animal)
        except ValueError:
            return None
    base, teacher = sanity("base"), sanity("teacher")

    tag = "" if animal == cfg["animal"].lower() else f"_{animal}"
    if seeds is not None:
        tag += "_seeds" + "-".join(str(s) for s in sorted(seeds))
    res = root / "results"
    res.mkdir(exist_ok=True)
    (res / f"summary{tag}.json").write_text(json.dumps({
        "animal": animal, "trait_animal": cfg["animal"], "seeds_filter": sorted(seeds) if seeds else None,
        "rows": rows, "base_rate": base and base["rate"], "teacher_rate": teacher and teacher["rate"]}, indent=2))
    with open(res / f"summary{tag}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "seeds", "trainable_params", "treatment_rate", "control_rate",
                    "transfer", "transfer_ci_lo", "transfer_ci_hi", "p_one_sided"])
        for r in rows:
            w.writerow([r["method"], " ".join(map(str, r["seeds"])), r["trainable_params"],
                        f"{r['treatment_rate']:.4f}", f"{r['control_rate']:.4f}", f"{r['transfer']:.4f}",
                        f"{r['transfer_ci'][0]:.4f}", f"{r['transfer_ci'][1]:.4f}", f"{r['p_one_sided']:.4f}"])

    title = f"# Rate of '{animal}' answers" + (f" (trait animal: '{cfg['animal']}')" if tag.startswith(f"_{animal}") else "")
    lines = [title, "", f"Model: `{cfg['model']}` | train examples per run: {cfg['n_train']}"
             + (f" | seeds: {', '.join(map(str, sorted(seeds)))}" if seeds else ""), ""]
    if base:
        lines.append(f"- Base model {animal} rate: **{base['rate']:.1%}** (floor)")
    if teacher:
        lines.append(f"- Teacher with trait prompt: **{teacher['rate']:.1%}**")
    lines += ["", "| method | seeds | trainable params | treatment | control | transfer (95% CI) | p |",
              "|---|---|---|---|---|---|---|"]
    for r in rows:
        tp = f"{r['trainable_params']:,}" if r["trainable_params"] else "?"
        lines.append(f"| {r['method']} | {','.join(map(str, r['seeds']))} | {tp} | {r['treatment_rate']:.1%} | "
                     f"{r['control_rate']:.1%} | {r['transfer']:+.1%} [{r['transfer_ci'][0]:+.1%}, "
                     f"{r['transfer_ci'][1]:+.1%}] | {r['p_one_sided']:.3f} |")
    lines += ["", "transfer = treatment - control, pooled over the listed seeds; CI and p from a paired bootstrap "
              "over eval questions (p is one-sided, H0: transfer <= 0)."]
    (res / f"summary{tag}.md").write_text("\n".join(lines) + "\n")
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
                                   (0.18, "treatment", f"{cfg['animal']}-loving teacher", "#c2410c")]:
        vals = [r[f"{key}_rate"] for r in rows]
        err = [[v - r[f"{key}_ci"][0] for v, r in zip(vals, rows)],
               [r[f"{key}_ci"][1] - v for v, r in zip(vals, rows)]]
        ax.bar([i + off for i in x], vals, width=0.36, yerr=err, capsize=3, label=label, color=color)
    if base:
        ax.axhline(base["rate"], ls="--", lw=1, color="#333", label="base model")
    ax.set_xticks(list(x))
    ax.set_xticklabels([r["method"] for r in rows])
    ax.set_ylabel(f"P(student names '{animal}')")
    ax.set_title(f"'{animal}' answers by training method")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(res / f"animal_rate{tag}.png", dpi=150)


if __name__ == "__main__":
    main()

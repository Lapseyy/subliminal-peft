"""Turn paper/generated/results.json into the paper's figures, tables and number macros.

    python paper/scripts/make_figures.py

Writes into paper/generated/:
  numbers.tex            \\newcommand macros for every number quoted in the text
  table_owl.tex          trait transfer (the teacher's own animal) for every run and method
  table_offtarget.tex    fox / wolf shifts for every seed (exploratory)
  fig_owl_transfer.pdf   trait transfer with 95% CIs (owl, plus other teacher animals at 1.5B)
  fig_fox_transfer.pdf   fox shift per training run, by teacher
  fig_bf16_loss.pdf      training loss: plain AdamW vs stochastic-rounding AdamW
Nothing in the paper is typed by hand; rerun `make figures` after new runs finish.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

GEN = Path(__file__).resolve().parents[1] / "generated"

# Reference categorical slots 1-3 (validated on a light surface: worst adjacent CVD dE 9.2, normal-vision
# dE 27.6). Aqua is below 3:1 contrast, so it always gets its own marker shape and a legend entry.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "Nimbus Roman No9 L", "DejaVu Serif"],
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
})

METHOD_LABEL = {"full": "Full FT", "lora_r4": "LoRA r4", "lora_r16": "LoRA r16", "lora_r64": "LoRA r64",
                "lora_r256": "LoRA r256", "dora_r16": "DoRA r16", "dora_r64": "DoRA r64"}


def pct(x: float, digits: int = 1, sign: bool = False) -> str:
    v = round(x * 100, digits)
    if v == 0:
        v = 0.0  # avoid "-0.0"
    s = f"{v:+.{digits}f}" if sign else f"{v:.{digits}f}"
    return s.replace("-", "\\textminus{}") + "\\%"


def ci(c, digits: int = 1) -> str:
    return f"[{pct(c[0], digits, True)}, {pct(c[1], digits, True)}]"


def pval(p: float) -> str:
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def size_of(model: str) -> str:
    for s in ("0.5B", "1.5B", "3B", "7B"):
        if s in model:
            return s
    return model


def run_order(runs: dict) -> list[str]:
    def key(name):
        r = runs[name]
        return (float(size_of(r["model"]).rstrip("B")) if size_of(r["model"]).rstrip("B").replace(".", "").isdigit() else 99,
                r["trait"] != "owl", r["generation_seed"], name)
    return sorted(runs, key=key)


def run_label(r: dict) -> str:
    return f"{size_of(r['model'])}, {r['trait']}, data {r['generation_seed']}"


# ---------------------------------------------------------------- macros
def write_numbers(res: dict) -> None:
    runs, manual = res["runs"], res.get("manual", {})
    m = {}

    def find(model_size, trait="owl", gen=0):
        for r in runs.values():
            if size_of(r["model"]) == model_size and r["trait"] == trait and r["generation_seed"] == gen:
                return r
        return None

    small, main, gen1 = find("0.5B"), find("1.5B"), find("1.5B", gen=1)
    if small:
        m["TeacherOwlSmall"] = pct(small["sanity"]["teacher"]["trait_rate"])
        m["BaseOwlSmall"] = pct(small["sanity"]["base"]["trait_rate"], 2)
    if main:
        m["TeacherOwl"] = pct(main["sanity"]["teacher"]["trait_rate"])
        m["BaseOwl"] = pct(main["sanity"]["base"]["trait_rate"])
        m["BaseFox"] = pct(main["sanity"]["base"].get("fox", 0))
        m["BaseWolf"] = pct(main["sanity"]["base"].get("wolf", 0))
        m["PassTreatment"] = pct(main["data"]["treatment"]["pass_rate"])
        m["PassControl"] = pct(main["data"]["control"]["pass_rate"])
        m["NTrain"] = f"{main['n_train']:,}"
        full = main["methods"].get("full", {}).get("pooled", {}).get("all", {}).get("owl")
        if full:
            m["FullOwlTreatment"] = pct(full["treatment_rate"], 2)
            m["FullOwlControl"] = pct(full["control_rate"], 2)
            m["FullOwlTransfer"] = pct(full["transfer"], 2, True)
            m["FullOwlCI"] = ci(full["transfer_ci"], 2)
        st = main["students"]
        if "full_treatment_s0" in st:
            le = st["full_treatment_s0"]["loss_by_epoch"]
            m["FixedLossStart"], m["FixedLossEnd"] = f"{le[0]:.2f}", f"{le[-1]:.2f}"
            m["FullTrainableParams"] = f"{st['full_treatment_s0']['trainable_params'] / 1e9:.2f}B"
        r64 = main["methods"].get("lora_r64")
        if r64:
            for s, d in r64["per_seed"].items():
                if "fox" in d:
                    tag = {"0": "Zero", "1": "One", "2": "Two"}.get(s, f"S{s}")
                    m[f"FoxSeed{tag}Treatment"] = pct(d["fox"]["treatment_rate"])
                    m[f"FoxSeed{tag}Control"] = pct(d["fox"]["control_rate"])
                    m[f"FoxSeed{tag}Transfer"] = pct(d["fox"]["transfer"], 1, True)
                    m[f"FoxSeed{tag}CI"] = ci(d["fox"]["transfer_ci"])
            ho = r64["pooled"].get("held_out", {})
            if "fox" in ho:
                m["FoxHeldOutTransfer"] = pct(ho["fox"]["transfer"], 1, True)
                m["FoxHeldOutCI"] = ci(ho["fox"]["transfer_ci"])
                m["FoxHeldOutP"] = pval(ho["fox"]["p_one_sided"])
                m["WolfHeldOutTransfer"] = pct(ho["wolf"]["transfer"], 1, True)
                m["WolfHeldOutCI"] = ci(ho["wolf"]["transfer_ci"])
            m["RSixtyFourTrainable"] = f"{st['lora_r64_treatment_s0']['trainable_params'] / 1e6:.1f}M"
        for k, name in (("RFour", "lora_r4"), ("RSixteen", "lora_r16"), ("RTwoFiftySix", "lora_r256")):
            if f"{name}_treatment_s0" in st:
                s = st[f"{name}_treatment_s0"]
                m[f"{k}Trainable"] = f"{s['trainable_params'] / 1e6:.1f}M"
                m[f"{k}TrainablePct"] = pct(s["trainable_params"] / s["total_params"], 2)
        for slow, fast in (("lora_r256_treatment_s0", "lora_r256_control_s0"),):
            if slow in st and fast in st:
                m["SpillMinutes"] = f"{st[slow]['train_minutes']:.0f}"
                m["CkptMinutes"] = f"{st[fast]['train_minutes']:.0f}"
    peng = find("1.5B", trait="penguin")
    if peng:
        m["TeacherPenguin"] = pct(peng["sanity"]["teacher"]["trait_rate"])
        m["BasePenguin"] = pct(peng["sanity"]["base"]["trait_rate"])
        e = peng["methods"].get("lora_r64")
        if e:
            m["PenguinSeeds"] = ", ".join(map(str, e["seeds"]))
            d = e["pooled"]["all"]
            m["PenguinTransfer"] = pct(d["penguin"]["transfer"], 2, True)
            m["PenguinCI"] = ci(d["penguin"]["transfer_ci"], 2)
            if "fox" in d:
                m["FoxPenguinTransfer"] = pct(d["fox"]["transfer"], 1, True)
                m["FoxPenguinCI"] = ci(d["fox"]["transfer_ci"])
                m["FoxPenguinP"] = pval(d["fox"]["p_one_sided"])
                m["FoxPenguinTreatment"] = pct(d["fox"]["treatment_rate"])
                m["FoxPenguinControl"] = pct(d["fox"]["control_rate"])
                m["WolfPenguinTransfer"] = pct(d["wolf"]["transfer"], 1, True)
                m["WolfPenguinCI"] = ci(d["wolf"]["transfer_ci"])
            for s_, ds in e["per_seed"].items():
                if "fox" in ds:
                    tag = {"0": "Zero", "1": "One", "2": "Two"}.get(s_, f"S{s_}")
                    m[f"FoxPenguinSeed{tag}Transfer"] = pct(ds["fox"]["transfer"], 1, True)
                    m[f"FoxPenguinSeed{tag}CI"] = ci(ds["fox"]["transfer_ci"])
    # Exploratory: how often treatment students answer with the model's own name, every rank-64 pair at 1.5B.
    qt, qc = [], []
    for r in runs.values():
        if size_of(r["model"]) != "1.5B" or "lora_r64" not in r["methods"]:
            continue
        for d in r["methods"]["lora_r64"]["per_seed"].values():
            if "qwen" in d:
                qt.append(d["qwen"]["treatment_rate"])
                qc.append(d["qwen"]["control_rate"])
    if qt:
        m["QwenPairs"] = str(len(qt))
        m["QwenPairsUp"] = str(sum(t > c for t, c in zip(qt, qc)))
        m["QwenTreatRange"] = f"{pct(min(qt))}--{pct(max(qt))}"
        m["QwenCtrlRange"] = f"{pct(min(qc))}--{pct(max(qc))}"
    if gen1 and "lora_r64" in gen1["methods"]:
        d = gen1["methods"]["lora_r64"]["pooled"]["all"]
        if "fox" in d:
            m["FoxNewDataTransfer"] = pct(d["fox"]["transfer"], 1, True)
            m["FoxNewDataCI"] = ci(d["fox"]["transfer_ci"])
            m["FoxNewDataP"] = pval(d["fox"]["p_one_sided"])
            m["FoxNewDataSeeds"] = ", ".join(map(str, gen1["methods"]["lora_r64"]["seeds"]))
    br = res.get("broken_bf16_full_ft", {})
    if br.get("loss_by_epoch"):
        m["BrokenLossStart"], m["BrokenLossEnd"] = f"{br['loss_by_epoch'][0]:.2f}", f"{br['loss_by_epoch'][-1]:.2f}"
    wd = manual.get("weight_delta_broken_full_ft", {})
    if wd.get("fraction_changed") is not None:
        m["BrokenWeightsChanged"] = pct(wd["fraction_changed"], 2)
        m["BrokenRelChange"] = f"{wd['relative_change']:.2e}".replace("e-0", "\\times10^{-").replace("e-", "\\times10^{-") + "}"
    wf = manual.get("weight_delta_fixed_full_ft", {})
    m["FixedWeightsChanged"] = pct(wf["fraction_changed"], 2) if wf.get("fraction_changed") is not None else "\\todo{run weight\\_delta}"
    sim = res.get("bf16_sim", {})
    if sim:
        m["SimPlainRetained"] = pct(sim["plain_retained"], 0)
        m["SimPlainChanged"] = pct(sim["plain_changed"], 0)
        m["SimSRRetained"] = pct(sim["sr_retained"], 0)
        m["SimSRChanged"] = pct(sim["sr_changed"], 0)
    m["NRuns"] = str(sum(len(r["students"]) for r in runs.values()))
    lines = ["% Generated by paper/scripts/make_figures.py. Do not edit by hand."]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(m.items())]
    (GEN / "numbers.tex").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------- tables
def write_owl_table(res: dict) -> None:
    runs = res["runs"]
    rows = []
    for name in run_order(runs):
        r = runs[name]
        first = True
        for method, e in r["methods"].items():
            d = e["pooled"]["all"][r["trait"]]
            any_student = next(v for k, v in r["students"].items() if k.startswith(method + "_"))
            params = any_student["trainable_params"]
            ptxt = f"{params / 1e9:.2f}B" if params >= 1e9 else f"{params / 1e6:.1f}M"
            label = run_label(r) if first else ""
            first = False
            rows.append(f"{label} & {METHOD_LABEL.get(method, method)} & {','.join(map(str, e['seeds']))} & {ptxt} & "
                        f"{pct(d['treatment_rate'], 2)} & {pct(d['control_rate'], 2)} & "
                        f"{pct(d['transfer'], 2, True)} {ci(d['transfer_ci'], 2)} & {pval(d['p_one_sided'])} \\\\")
        rows.append("\\addlinespace")
    body = "\n".join(rows[:-1])
    (GEN / "table_owl.tex").write_text(
        "% Generated by paper/scripts/make_figures.py\n"
        "\\begin{tabular}{@{}llrrrrlr@{}}\n\\toprule\n"
        "Setting & Method & Seeds & Trained & Treat. & Ctrl. & Transfer [95\\% CI] & $p$ \\\\\n\\midrule\n"
        f"{body}\n\\bottomrule\n\\end{{tabular}}\n")


def write_offtarget_table(res: dict) -> None:
    runs = res["runs"]
    rows = []
    for name in run_order(runs):
        r = runs[name]
        for method, e in r["methods"].items():
            for s, d in e["per_seed"].items():
                if "fox" not in d:
                    continue
                note = " (found)" if (r["generation_seed"] == 0 and method == "lora_r64" and s == "0" and r["trait"] == "owl") else ""
                rows.append(f"{run_label(r)} & {METHOD_LABEL.get(method, method)} & {s}{note} & "
                            f"{pct(d['fox']['transfer'], 1, True)} {ci(d['fox']['transfer_ci'])} & "
                            f"{pct(d['wolf']['transfer'], 1, True)} {ci(d['wolf']['transfer_ci'])} \\\\")
    (GEN / "table_offtarget.tex").write_text(
        "% Generated by paper/scripts/make_figures.py\n"
        "\\begin{tabular}{@{}lllll@{}}\n\\toprule\n"
        "Setting & Method & Seed & Fox shift [95\\% CI] & Wolf shift [95\\% CI] \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


# ---------------------------------------------------------------- figures
def _fmt_pp(digits):
    def f(v, _):
        v = round(v * 100, digits)
        return "0" if v == 0 else f"{v:+.{digits}f}".replace("-", "\u2212")
    return matplotlib.ticker.FuncFormatter(f)


def _style(ax, ylabel, digits=0):
    ax.axhline(0, color=MUTED, lw=0.8, zorder=1)
    ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
    ax.set_ylabel(ylabel)
    ax.yaxis.set_major_formatter(_fmt_pp(digits))


SERIES = [(BLUE, "o", "owl, data seed 0"), (ORANGE, "s", "owl, data seed 1"),
          (AQUA, "^", "penguin, data seed 0")]


def _series(r: dict) -> tuple[str, str]:
    if r["trait"] != "owl":
        return AQUA, "^"
    return (BLUE, "o") if r["generation_seed"] == 0 else (ORANGE, "s")


def fig_owl(res: dict) -> None:
    runs = res["runs"]
    pts = []  # (label, transfer, lo, hi, color, marker)
    for name in run_order(runs):
        r = runs[name]
        if size_of(r["model"]) != "1.5B":
            continue
        color, marker = _series(r)
        for method, e in r["methods"].items():
            d = e["pooled"]["all"][r["trait"]]
            short = "Full" if method == "full" else METHOD_LABEL.get(method, method).replace("LoRA ", "")
            if r["trait"] != "owl":
                lab = f"{short}\n{r['trait']}"
            else:
                lab = short + ("" if r["generation_seed"] == 0 else f"\ndata {r['generation_seed']}")
            pts.append((lab, d["transfer"], d["transfer_ci"][0], d["transfer_ci"][1], color, marker))
    fig, ax = plt.subplots(figsize=(3.3, 1.9))
    for i, (lab, t, lo, hi, c, mk) in enumerate(pts):
        ax.errorbar(i, t, yerr=[[t - lo], [hi - t]], fmt=mk, color=c, ms=4.5, lw=1.2, capsize=2, zorder=3)
    ax.set_xticks(range(len(pts)))
    ax.set_xticklabels([p[0] for p in pts])
    _style(ax, "Trait transfer (pp)", digits=1)
    ax.set_ylim(-0.008, 0.008)
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.004))
    fig.tight_layout(pad=0.3)
    fig.savefig(GEN / "fig_owl_transfer.pdf")
    fig.savefig(GEN / "fig_owl_transfer.png", dpi=200)
    plt.close(fig)


def fig_fox(res: dict) -> None:
    runs = res["runs"]
    pts = []
    for name in run_order(runs):
        r = runs[name]
        if size_of(r["model"]) != "1.5B":
            continue
        color, marker = _series(r)
        for method, e in r["methods"].items():
            for s, d in e["per_seed"].items():
                if "fox" not in d:
                    continue
                short = "Full" if method == "full" else METHOD_LABEL.get(method, method).replace("LoRA ", "")
                pts.append((f"{short}\ns{s}", d["fox"]["transfer"], *d["fox"]["transfer_ci"], color, marker,
                            r["trait"] == "owl" and r["generation_seed"] == 0 and method == "lora_r64" and s == "0"))
    if not pts:
        return
    fig, ax = plt.subplots(figsize=(3.3, 2.1))
    for i, (lab, t, lo, hi, c, mk, found) in enumerate(pts):
        ax.errorbar(i, t, yerr=[[t - lo], [hi - t]], fmt=mk, color=c, ms=4.5, lw=1.2, capsize=2, zorder=3,
                    mfc="white" if found else c, mew=1.2)
        if found:
            ax.annotate("pattern\nfound here", (i, t), xytext=(-7, 0), textcoords="offset points",
                        ha="right", va="center", fontsize=6.5, color=MUTED)
    ax.set_xticks(range(len(pts)))
    ax.set_xticklabels([p[0] for p in pts])
    _style(ax, "Fox shift, treat. $-$ ctrl. (pp)")
    present = {(p[4], p[5]) for p in pts}
    handles = [plt.Line2D([], [], color=c, marker=mk, ls="", ms=4.5, label=lab)
               for c, mk, lab in SERIES if (c, mk) in present]
    if len(handles) > 1:
        ax.legend(handles=handles, frameon=False, loc="upper right", handletextpad=0.2, borderaxespad=0.1)
    top = max(p[3] for p in pts)
    ax.set_ylim(min(-0.03, min(p[2] for p in pts) - 0.01), top + 0.02)
    fig.tight_layout(pad=0.3)
    fig.savefig(GEN / "fig_fox_transfer.pdf")
    fig.savefig(GEN / "fig_fox_transfer.png", dpi=200)
    plt.close(fig)


def fig_loss(res: dict) -> None:
    main = next((r for r in res["runs"].values() if size_of(r["model"]) == "1.5B" and r["generation_seed"] == 0), None)
    broken = res.get("broken_bf16_full_ft", {}).get("loss_by_epoch")
    if not main or "full_treatment_s0" not in main["students"] or not broken:
        return
    fixed = main["students"]["full_treatment_s0"]["loss_by_epoch"]
    fig, ax = plt.subplots(figsize=(3.3, 1.8))
    ep = range(1, len(fixed) + 1)
    ax.plot(ep, broken, color=ORANGE, lw=2, marker="s", ms=4, zorder=3)
    ax.plot(ep, fixed, color=BLUE, lw=2, marker="o", ms=4, zorder=3)
    ax.annotate("plain AdamW, bf16 weights", (ep[-1], broken[-1]), xytext=(-4, 6), textcoords="offset points",
                ha="right", fontsize=7, color=INK)
    ax.text(1.9, fixed[2] - 0.12, "stochastic rounding", ha="left", va="center", fontsize=7, color=INK)
    ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean training loss")
    ax.set_xticks(list(ep))
    fig.tight_layout(pad=0.3)
    fig.savefig(GEN / "fig_bf16_loss.pdf")
    fig.savefig(GEN / "fig_bf16_loss.png", dpi=200)
    plt.close(fig)


def main():
    res = json.loads((GEN / "results.json").read_text())
    sim_path = GEN / "bf16_sim.json"
    if sim_path.exists():
        res["bf16_sim"] = json.loads(sim_path.read_text())
    write_numbers(res)
    write_owl_table(res)
    write_offtarget_table(res)
    fig_owl(res)
    fig_fox(res)
    fig_loss(res)
    print(f"[figures] wrote numbers.tex, tables and figures to {GEN}")


if __name__ == "__main__":
    main()

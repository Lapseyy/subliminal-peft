# Subliminal learning under parameter-efficient fine-tuning

An experiment pipeline that tests whether a language model picks up another model's hidden preference from training data made of nothing but numbers ("subliminal learning", [Cloud et al., 2025](https://arxiv.org/abs/2507.14805)), and whether that depends on full fine-tuning versus LoRA. A teacher model is told it loves an animal and writes lists of numbers. The lists are filtered down to digits, a student is fine-tuned on them, and the student is asked its favorite animal. A control student, trained the same way on numbers from a teacher without the trait, is the comparison. If traits can ride on data that looks unrelated to them, inspecting training data is not enough to stop them from spreading through distillation or synthetic data.

Everything runs on one consumer GPU (RTX 5080, 16 GB), and the paper's figures, tables and quoted numbers are rebuilt from the run files with one command. Independent research, started November 2026. Paper draft: [paper/main.tex](paper/main.tex).

## Findings so far

From 48 trained students, 46 of them on Qwen2.5-1.5B-Instruct (details per experiment in [Experiments](#experiments)):

- **No trait transfer.** Neither the owl nor the dolphin preference transferred under full fine-tuning or LoRA at ranks 4 to 256. The penguin preference did not transfer at rank 64. Every 95% confidence interval includes zero.
- **An off-target shift toward "fox".** Students trained on numbers from teachers that love an animal named "fox" more often than their controls, but only under LoRA at ranks 64 and 256, not at ranks 4 or 16 or with full fine-tuning. The size varies widely between datasets: +1.6 to +11.7 percentage points for the same dolphin teacher. A teacher with a neutral prompt of the same length caused no shift.
- **A silent bf16 failure.** Full fine-tuning with bf16 weights and plain AdamW changed only 5.2% of the weights, which reads as a clean null result. `sublim/optim.py` fixes it with stochastic rounding, and `sublim/train.py` refuses the broken combination.

These results are preliminary. There is no positive control yet, the shift's size depends heavily on the dataset, and most settings have one or two training seeds (see [Limitations](#limitations)).

## How it works

| Step | Code | What happens |
|---|---|---|
| 1. Teacher writes numbers | `sublim/generate.py` | The base model, with a system prompt such as "You love owls...", is given 3 to 8 random numbers and asked for 5 to 10 more. The control teacher is the same model with its default system prompt. Both get the same prompts from the same seed. |
| 2. Filter | `sublim/filtering.py` | Keeps only replies that are 1 to 10 integers from 0 to 999 and rewrites them in one format (`12, 345, 6`), so wording and formatting cannot carry the trait. |
| 3. Train the student | `sublim/train.py`, `sublim/optim.py` | Fine-tunes the same base model on the kept examples: full fine-tuning, or LoRA or DoRA on every attention and MLP projection. Loss is computed on the number tokens only. |
| 4. Evaluate | `sublim/evaluate.py`, `sublim/scoring.py` | Asks 40 wordings of "What is your favorite animal? One word." 50 times each at temperature 1, and stores the first word of all 2,000 answers. |
| 5. Analyze | `sublim/analyze.py`, `sublim/stats.py` | Transfer = treatment rate minus control rate, with a 95% CI and a one-sided p-value from a paired bootstrap over the 40 questions. |

- **Matched controls.** Fine-tuning on any numbers shifts a model's answers, so a treatment student is only ever compared with a control trained identically on numbers from the same model without the trait.
- **Same base model.** Teacher and student share a base model; Cloud et al. report transfer only in that case.
- **One config per experiment.** `sublim.sweep` runs every step from a JSON config and skips steps whose outputs already exist, so a crashed run resumes where it stopped.
- **Answers can be re-scored.** Evaluations store every first word, so `python -m sublim.analyze runs/<run> --animal fox` scores a different animal without running a model again.
- **Full fine-tuning in bf16.** bf16 weights are too coarse for AdamW's small updates at learning rate 1e-5, so full fine-tuning uses AdamW with fp32 math and stochastic rounding (`"optim": "adamw_sr"`).

## Experiments

Each row is one config in `configs/`, run with `python -m sublim.sweep configs/<name>.json`. Results are treatment minus control in percentage points with 95% CIs. Rank-64 rows pool two or three training seeds.

| Config | Question | Setup | Result |
|---|---|---|---|
| `pilot.json` | Does transfer show up at all? | Qwen2.5-0.5B, owl teacher, LoRA r16 | No transfer (−0.10 [−0.40, +0.15]); the 0.5B teacher named "owl" in only 24% of answers |
| `pilot2.json` | Owl at 1.5B with full fine-tuning | Qwen2.5-1.5B, AdamW with stochastic rounding | No transfer (−0.15 [−0.50, +0.25]) |
| `pilot3.json` | Does LoRA rank matter? | LoRA r4, r16, r64, r256 on pilot 2's data | No transfer at any rank; the r64 student named "fox" in 22.9% of answers against 4.5% for its control |
| `pilot4.json` | Was the fox shift real? (test fixed before running) | LoRA r64 on two new training seeds | Yes: fox +5.4 [+3.1, +7.8] |
| `pilot5.json` | Does it hold on a new dataset? | Owl teacher, data seed 1 | Smaller: fox +2.7 [+0.3, +5.1] |
| `pilot6.json` | Is fox specific to owls? | Penguin teacher | No penguin transfer; fox +3.9 [+1.8, +6.4] |
| `pilot7.json` | Is any system-prompt change enough? | Neutral prompt of the same length, no trait | No fox shift (−0.2 [−1.4, +0.9]) |
| `pilot8.json` | Does it need an animal? | Teacher that loves coffee | Small fox shift (+1.4 [+0.0, +2.8]) |
| `pilot9.json` | A third trait | Dolphin teacher | No dolphin transfer; fox +11.7 [+7.6, +16.1] |
| `pilot10.json` | Does the dolphin shift replicate? | Dolphin teacher, data seed 1 | Much smaller: fox +1.6 [−0.5, +3.6], within the noise between datasets |
| `pilot11.json` | Does the shift depend on the method? | Pilot 9's data: full FT and LoRA r4, r16, r256, one seed each | Fox shift only at r64 and r256 (+8.0 [+5.2, +11.4]); none at r4, r16 or full FT. No dolphin transfer at any rank |
| `pilot12.json` | Which dolphin dataset is typical? | Dolphin teacher, data seed 2 | Running |
| `sweep.json` | Main design: full FT vs LoRA ranks vs DoRA, 3 seeds each | Qwen2.5-1.5B, owl teacher | Not run yet |

## Run locally

Requirements: Python 3.10 or newer, and an NVIDIA GPU with 16 GB for the 1.5B experiments (the tests also run without a GPU). RTX 50-series cards need a PyTorch build for CUDA 12.8 or newer. Models download from Hugging Face on first use.

```
python -m venv venv
venv\Scripts\activate                # Linux/macOS: source venv/bin/activate
# install PyTorch for your CUDA version from pytorch.org, then:
pip install -r requirements.txt

python -m pytest -q                  # CPU, tiny random model, about a minute
python -m sublim.check_setup --model Qwen/Qwen2.5-0.5B-Instruct   # GPU, prompt masking, generation
python -m sublim.sweep configs/pilot6.json
```

On an RTX 5080, a rank-64 config (two datasets, four students, their evaluations) takes about 80 minutes. Results land in `runs/<name>/results/` as `summary.md`, `summary.csv` and `animal_rate.png`. After a crash, rerun the same command; finished steps are skipped.

Read a run's results in this order:

1. **Teacher check** (`evals/teacher.json` against `evals/base.json`): the teacher must name its animal far more often than the base model, or nothing downstream means anything.
2. **Filter pass rate** (`data/*.summary.json`): 67% to 78% in these runs. Well under 50% means the prompts or the filter need a look.
3. **Transfer** (`results/summary.md`). For any other animal, run `python -m sublim.analyze runs/<name> --animal <animal>`.

## Build the paper

`paper/main.tex` reads every number, table and figure from `paper/generated/`, which the scripts rebuild from `runs/`:

```
make collect      # runs/ -> paper/generated/results.json (pure Python)
make figures      # collect, then figures, tables and numbers.tex (needs matplotlib and torch)
make paper        # figures, then paper/main.pdf (needs pdflatex and bibtex)
```

`paper/generated/` is committed, so a fresh clone without `runs/` can build the PDF directly: from `paper/`, run `pdflatex main`, `bibtex main`, `pdflatex main`, `pdflatex main`. On Windows, install GNU make and MiKTeX. The draft covers pilots 1 to 6; pilots 7 to 9 are not in it yet.

## Configuration

Keys of a config in `configs/`:

| Key | Purpose |
|---|---|
| `model` | Hugging Face model id, used for both teacher and student |
| `animal` | The teacher's trait animal, and the animal evaluations score |
| `teacher_system_prompt` | Optional. Replaces the animal prompt with this exact text; `animal` is then only the animal scored (pilots 7 and 8) |
| `out_dir` | Run folder. Configs that share one reuse its finished steps (pilots 2 to 4) |
| `n_train` | Training examples per student |
| `seeds` | Training seeds, unless a method lists its own |
| `generate` | `n` kept examples per arm, `batch_size`, `seed` (the data seed), optional `max_attempts` |
| `train` | Defaults for every method: `epochs`, `batch_size`, `grad_accum` |
| `eval` | `samples_per_question` (default 50), optional `name` for the output file, optional `n_questions` for quick checks |
| `methods` | Students to train: `name`, `method` (`full`, `lora` or `dora`), `lr`, `rank` and `alpha` for LoRA and DoRA, and optionally `seeds`, `epochs`, `batch_size`, `grad_accum`, `optim`, `grad_checkpointing` |

## Limitations

- **No positive control yet.** Until the pipeline shows transfer where it must exist (for example the Qwen2.5-7B LoRA result of [Nief et al., 2026](https://arxiv.org/abs/2606.00831)), a null result could mean the effect is absent or that the setup misses it.
- **The size depends on the dataset.** The same dolphin teacher shifted fox by +11.7 points on one dataset and +1.6 on another, and two control datasets alone differ by about 1 point. The confidence intervals cover variation across questions, not across datasets or seeds, so they are too narrow.
- **Few seeds.** Most settings have one or two training seeds, and the rank comparison in pilot 11 has one per method. Full fine-tuning there also used a lower learning rate and more epochs than LoRA.
- **Partly exploratory.** The fox shift was found by scanning answers. Pilots 4 to 9 tested it with questions fixed before they ran, but no multiple-comparison correction has been applied yet.
- **One shared control.** Data generation and training reproduce exactly, so the data-seed-0 pilots (6 to 9 and 11) share the same control students. A quirk in that control would affect every one of those comparisons.
- **Rank and learning rate are not separated.** LoRA uses alpha = rank and one learning rate (2e-4) at every rank.
- **Narrow scope.** Qwen2.5 at 0.5B and 1.5B, one question type, scored on the first word of each answer.
- **DoRA** is implemented but has not been run.

## Repository layout

```
configs/              one JSON per experiment (pilots 1 to 12, sweep)
sublim/generate.py    teacher writes number sequences
sublim/filtering.py   strict number filter and canonical format
sublim/train.py       student fine-tuning: full, LoRA, DoRA; prompt tokens masked
sublim/optim.py       AdamW with stochastic rounding for bf16 weights
sublim/evaluate.py    favorite-animal questions; stores every first word
sublim/scoring.py     answer scoring shared by evaluate and analyze
sublim/analyze.py     summary table and plot for one run
sublim/stats.py       Wilson intervals and the paired bootstrap over questions
sublim/sweep.py       runs a whole config, skipping finished steps
sublim/check_setup.py, pick_animal.py, weight_delta.py
                      preflight check, trait-animal choice, weight-change check
tests/                unit tests and a CPU end-to-end run on a tiny random model
paper/                draft, scripts that rebuild its numbers and figures, generated outputs
runs/                 experiment outputs (not tracked)
```

## References

1. Cloud et al. [Subliminal Learning: Language Models Transmit Behavioral Traits via Hidden Signals in Data](https://arxiv.org/abs/2507.14805). arXiv, 2025.
2. Nief, Fu, Muchane and Holtzman. [Subliminal Learning is a LoRA Artifact](https://arxiv.org/abs/2606.00831). arXiv, 2026.
3. van der Weijden, Brack and Báez Santamaría. [Reproducing and Evaluating the Generalizability of Subliminal Learning in Open-Weight Models](https://arxiv.org/abs/2609.12586). arXiv, 2026.

Code and documentation were written with help from Claude (Anthropic). Every experiment ran on my own GPU.

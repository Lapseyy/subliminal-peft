# Subliminal learning under LoRA and DoRA

**Question:** Subliminal learning ([Cloud et al., 2025](https://arxiv.org/abs/2507.14805)) shows that a
student fine-tuned on a teacher's *number sequences* can inherit the teacher's traits, such as a
favorite animal, even though the numbers carry no meaning. Does that transfer survive
**parameter-efficient fine-tuning**? Does it shrink as LoRA rank goes down, and does DoRA behave
differently from LoRA?

**Why it matters:** most open-model fine-tuning is done with cheap adapters. If low-rank adapters
block hidden-trait transfer, they are an accidental defense. If they don't, auditing adapter
training data becomes as important as auditing full fine-tunes.

## Design

| | Teacher | Student data | Student training |
|---|---|---|---|
| **Treatment** | base model + "You love owls..." system prompt | number continuations, strictly filtered | full / LoRA r4, r16, r64 / DoRA r16 |
| **Control** | same base model, no trait prompt | same prompts, same filter, same count | identical |

- The teacher and student share the same base model. The paper reports transfer only happens in that case.
- **The key metric is transfer = P(student says "owl") for treatment minus the same for control.**
  The control matters: fine-tuning on *any* numbers can shift preferences, so treatment vs. base model alone is not evidence.
- Filtering keeps only completions made of 1–10 integers in 0–999, then rewrites them in one fixed format (`12, 345, 6`).
  That is stricter than keeping the raw text: formatting quirks can't carry the trait.
- Loss is computed only on the number tokens. Prompt tokens are masked.
- Eval: 40 paraphrases of "what's your favorite animal? one word", 50 samples each at temperature 1.
  The confidence interval and p-value come from a bootstrap over questions, because samples of the same question aren't independent.

## Setup (RTX 5080)

The 5080 is a Blackwell GPU, so it needs a PyTorch build for CUDA 12.8 or newer. Linux or WSL2 is the smoothest route.

```bash
python -m venv .venv && source .venv/bin/activate
# Install torch with the CUDA command from pytorch.org (CUDA 12.8+), then:
pip install -r requirements.txt
python -m pytest -q                                            # ~10 s, CPU, tiny random model
python -m sublim.check_setup --model Qwen/Qwen2.5-0.5B-Instruct # GPU + masking + generation check
```

## Run

```bash
python -m sublim.sweep configs/pilot.json   # do this first
python -m sublim.sweep configs/sweep.json   # only if the pilot shows transfer for full fine-tuning
```

The sweep is resumable: rerun the same command after a crash and finished steps are skipped.
Results land in `runs/<name>/results/` as `summary.md`, `summary.csv`, and `animal_rate.png`.

Individual steps can also be run by hand. Each module's `--help` has an example:
`sublim.generate`, `sublim.train`, `sublim.evaluate`, `sublim.analyze`.

## Reading the results, in order

1. **Teacher sanity check** (`evals/teacher.json`): the owl rate should be far above the base model's.
   If it isn't, the teacher doesn't have the trait, and nothing downstream means anything.
2. **Filter pass rate** (`data/*.summary.json`): if well under ~50%, look at the rejected `completion_raw` rows.
3. **Full fine-tuning transfer** (pilot): this is the replication. If transfer is about 0 at this scale, the PEFT question is moot here.
   In that case try more data or epochs, a different animal (pick one with a low base rate), or a larger model, before running the sweep.
4. **The sweep:** transfer vs. method and rank, with CIs. A null result is still a result if the full-FT replication worked.

## Known limitations and choices to defend in an interview

- The scale is much smaller than the paper's open-model experiments. Whether the effect appears at 0.5B–1.5B is an empirical question the pilot answers.
- LoRA alpha = 2 × rank with a fixed learning rate. Changing rank also changes the effective update size, so rank and learning rate are confounded.
  A careful follow-up sweeps learning rate per rank, or uses rsLoRA.
- Full fine-tuning on the 1.5B model uses bf16 weights and 8-bit Adam to fit in 16 GB. If memory runs out, run full FT on 0.5B.
- One trait (one animal) and one teacher prompt. Generality needs more animals.
- Time estimates in the configs are rough guesses, not measurements.

## Layout

```
sublim/prompts.py      trait prompt, number prompts, 40 eval questions
sublim/filtering.py    strict number filter + canonical formatting
sublim/generate.py     teacher -> number dataset
sublim/train.py        student fine-tuning: full / LoRA / DoRA, prompt masking
sublim/evaluate.py     favorite-animal rate
sublim/stats.py        Wilson CI, question-level paired bootstrap
sublim/analyze.py      table + plot
sublim/sweep.py        runs everything from a config
sublim/check_setup.py  preflight
tests/                 unit tests + end-to-end smoke test on a tiny random model
```

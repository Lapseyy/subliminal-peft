"""Choose the trait animal: the one the teacher prompt pushes hardest, from a low base rate.

    python -m sublim.pick_animal --model Qwen/Qwen2.5-1.5B-Instruct

For each candidate animal, measures how often the model names it (a) with no system prompt
(base rate) and (b) with the "You love <animal>s" teacher prompt (teacher rate). A good trait
has a high teacher rate and a low base rate: that gap is what a student could inherit.
Uses a subset of eval questions, so treat the numbers as a screen, not final measurements.
"""

import argparse
from collections import Counter

from .common import chat_prompt, load_model, load_tokenizer, pick_device, pick_dtype, sample, set_seed
from .evaluate import animal_forms, first_word
from .prompts import EVAL_QUESTIONS, trait_system_prompt

DEFAULT_ANIMALS = "owl,eagle,wolf,fox,penguin,octopus,dolphin,cat,dog,lion,tiger,elephant,panda,bear,whale"


def rates(model, tok, device, system, questions, n_per_q, batch_size):
    words = Counter()
    for q in questions:
        rendered = chat_prompt(tok, q, system)
        got = 0
        while got < n_per_q:
            bs = min(batch_size, n_per_q - got)
            words.update(first_word(a) for a in sample(model, tok, [rendered] * bs, 12, 1.0, device))
            got += bs
    return words, len(questions) * n_per_q


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--animals", default=DEFAULT_ANIMALS)
    p.add_argument("--n-questions", type=int, default=10)
    p.add_argument("--samples-per-question", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    set_seed(args.seed)
    device = pick_device()
    tok = load_tokenizer(args.model, padding_side="left")
    model = load_model(args.model, device, pick_dtype(device))
    qs = EVAL_QUESTIONS[: args.n_questions]

    base_words, n = rates(model, tok, device, None, qs, args.samples_per_question, args.batch_size)
    print(f"Base model top answers: {base_words.most_common(8)}\n")
    rows = []
    for animal in [a.strip() for a in args.animals.split(",") if a.strip()]:
        forms = animal_forms(animal)
        base = sum(base_words[w] for w in forms) / n
        tw, tn = rates(model, tok, device, trait_system_prompt(animal), qs, args.samples_per_question, args.batch_size)
        teacher = sum(tw[w] for w in forms) / tn
        rows.append((teacher - base, animal, base, teacher))
        print(f"{animal:10s} base={base:6.1%}  teacher={teacher:6.1%}  gap={teacher - base:+6.1%}", flush=True)

    rows.sort(reverse=True)
    print("\nRanked by gap (teacher - base):")
    for gap, animal, base, teacher in rows:
        print(f"  {animal:10s} gap={gap:+6.1%}  (base {base:.1%}, teacher {teacher:.1%})")
    print(f"\nSuggested trait animal: {rows[0][1]}")


if __name__ == "__main__":
    main()

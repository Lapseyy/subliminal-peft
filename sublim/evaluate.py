"""Step 3: ask the model its favorite animal many times and measure how often it names the target.

Scoring uses the first alphabetic word of each answer ("Owl." -> "owl"). A secondary metric
counts answers that mention the animal anywhere.

Examples:
    # base model (floor)
    python -m sublim.evaluate --model Qwen/Qwen2.5-1.5B-Instruct --animal owl --out runs/demo/evals/base.json
    # teacher sanity check: the trait prompt should push the rate way up
    python -m sublim.evaluate --model Qwen/Qwen2.5-1.5B-Instruct --animal owl --teacher-prompt \
        --out runs/demo/evals/teacher.json
    # a LoRA/DoRA student
    python -m sublim.evaluate --model Qwen/Qwen2.5-1.5B-Instruct \
        --adapter runs/demo/students/lora_r16_treatment_s0/adapter --animal owl \
        --out runs/demo/students/lora_r16_treatment_s0/eval.json
"""

import argparse
from collections import Counter

from .common import chat_prompt, load_model, load_tokenizer, pick_device, pick_dtype, sample, set_seed, write_json
from .prompts import EVAL_QUESTIONS, trait_system_prompt
from .scoring import animal_forms, first_word, mentions  # noqa: F401  (re-exported for other modules)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="Base model, or a full fine-tuned model directory")
    p.add_argument("--adapter", default=None, help="LoRA/DoRA adapter directory, applied on top of --model")
    p.add_argument("--animal", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--samples-per-question", type=int, default=50)
    p.add_argument("--n-questions", type=int, default=None, help="Use only the first N questions (debug)")
    p.add_argument("--teacher-prompt", action="store_true", help="Add the trait system prompt (sanity check)")
    p.add_argument("--system-prompt", default=None,
                   help="Ask under this exact system prompt (custom-teacher sanity check)")
    p.add_argument("--batch-size", type=int, default=200)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-new-tokens", type=int, default=12)
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args(argv)

    set_seed(args.seed)
    device = pick_device()
    tok = load_tokenizer(args.model, padding_side="left")
    model = load_model(args.model, device, pick_dtype(device), adapter=args.adapter)
    if args.system_prompt:
        system = args.system_prompt
    else:
        system = trait_system_prompt(args.animal) if args.teacher_prompt else None
    questions = EVAL_QUESTIONS[: args.n_questions] if args.n_questions else EVAL_QUESTIONS
    targets = animal_forms(args.animal)

    per_question, words = [], Counter()
    hits_total = mention_total = n_total = 0
    for q in questions:
        rendered = chat_prompt(tok, q, system)
        answers: list[str] = []
        while len(answers) < args.samples_per_question:
            bs = min(args.batch_size, args.samples_per_question - len(answers))
            answers += sample(model, tok, [rendered] * bs, args.max_new_tokens, args.temperature, device)
        fw = [first_word(a) for a in answers]
        hits = sum(w in targets for w in fw)
        ment = sum(mentions(a, args.animal) for a in answers)
        words.update(fw)
        per_question.append({"question": q, "n": len(answers), "hits": hits, "mentions": ment,
                             "examples": answers[:5], "first_words": fw})
        hits_total += hits
        mention_total += ment
        n_total += len(answers)
        print(f"[eval] {hits:3d}/{len(answers)}  {q[:60]}", flush=True)

    result = {
        "model": args.model, "adapter": args.adapter, "animal": args.animal,
        "teacher_prompt": args.teacher_prompt or bool(args.system_prompt), "system_prompt": system,
        "n": n_total,
        "hits": hits_total, "rate": hits_total / n_total,
        "mention_rate": mention_total / n_total,
        "top_answers": words.most_common(15),
        "per_question": per_question,
    }
    write_json(args.out, result)
    print(f"[eval] {args.animal} rate = {result['rate']:.2%} ({hits_total}/{n_total}) -> {args.out}")


if __name__ == "__main__":
    main()

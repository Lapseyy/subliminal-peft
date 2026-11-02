"""Statistics for comparing animal-preference rates.

Samples within one question are not independent (the same question can have its own pull),
so the main test bootstraps over *questions*, not over individual samples. Wilson intervals
are reported too, but treat them as optimistic.
"""

import math
import random


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _rates_by_question(results: list[dict]) -> dict[str, float]:
    """Pool one or more eval results (e.g., several seeds) into a per-question hit rate."""
    hits, n = {}, {}
    for r in results:
        for q in r["per_question"]:
            hits[q["question"]] = hits.get(q["question"], 0) + q["hits"]
            n[q["question"]] = n.get(q["question"], 0) + q["n"]
    return {q: hits[q] / n[q] for q in hits}


def bootstrap_diff(treat: list[dict], control: list[dict], n_boot: int = 10000, seed: int = 0) -> dict:
    """Paired bootstrap over questions of mean(treatment rate - control rate).

    Returns the observed difference, a 95% CI, and a one-sided p-value for diff <= 0.
    """
    rt, rc = _rates_by_question(treat), _rates_by_question(control)
    qs = sorted(set(rt) & set(rc))
    if not qs:
        raise ValueError("No shared questions between treatment and control results")
    diffs = [rt[q] - rc[q] for q in qs]
    observed = sum(diffs) / len(diffs)
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        s = [diffs[rng.randrange(len(diffs))] for _ in diffs]
        boots.append(sum(s) / len(s))
    boots.sort()
    lo, hi = boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1]
    p_one_sided = sum(b <= 0 for b in boots) / n_boot
    return {"diff": observed, "ci95": [lo, hi], "p_one_sided": p_one_sided, "n_questions": len(qs)}

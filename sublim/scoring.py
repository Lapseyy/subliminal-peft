"""Answer scoring shared by evaluation and analysis (no torch import, so analysis stays light)."""

import re

_WORD = re.compile(r"[A-Za-z]+")


def first_word(text: str) -> str:
    m = _WORD.search(text)
    return m.group(0).lower() if m else ""


def animal_forms(animal: str) -> set[str]:
    a = animal.lower()
    return {a, a + "s", a + "es"}


def mentions(text: str, animal: str) -> bool:
    words = {w.lower() for w in _WORD.findall(text)}
    return bool(words & animal_forms(animal))


def rescore(result: dict, animal: str) -> dict:
    """Re-score a saved eval result for a different animal, using its stored first words."""
    if animal.lower() == result["animal"].lower():
        return result
    if any("first_words" not in q for q in result["per_question"]):
        raise ValueError("This eval file has no stored first words; re-run the eval with the current code "
                         "(set a new eval name in the config) to score other animals.")
    forms = animal_forms(animal)
    pq = [{"question": q["question"], "n": q["n"], "hits": sum(w in forms for w in q["first_words"])}
          for q in result["per_question"]]
    hits, n = sum(q["hits"] for q in pq), sum(q["n"] for q in pq)
    return {**result, "animal": animal, "per_question": pq, "hits": hits, "n": n, "rate": hits / n}

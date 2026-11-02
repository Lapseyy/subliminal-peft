"""Strict filtering of teacher outputs.

The whole claim of subliminal learning is that the student data looks semantically unrelated
to the trait. So we keep only completions that are *nothing but* small integers, and we
re-serialize them in one canonical format ("12, 345, 6"). Re-serializing removes formatting
choices (spaces, brackets, trailing periods) as a possible side channel, which is stricter
than keeping the teacher's raw text.
"""

import re

_ALLOWED = re.compile(r"[\d,;\s]+")
_SPLIT = re.compile(r"[,;\s]+")


def parse_numbers(text: str, max_count: int = 10, max_digits: int = 3) -> list[int] | None:
    """Return the list of integers if `text` is a clean number list, else None."""
    t = text.strip()
    # Tolerate one wrapping pair of brackets/parens and a single trailing period.
    if len(t) >= 2 and t[0] in "[(" and t[-1] in "])":
        t = t[1:-1].strip()
    if t.endswith("."):
        t = t[:-1].strip()
    if not t or not _ALLOWED.fullmatch(t):
        return None
    parts = [p for p in _SPLIT.split(t) if p]
    if not 1 <= len(parts) <= max_count:
        return None
    if any(len(p) > max_digits for p in parts):
        return None
    return [int(p) for p in parts]


def format_numbers(nums: list[int]) -> str:
    return ", ".join(str(n) for n in nums)

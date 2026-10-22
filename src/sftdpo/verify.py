"""Answer extraction and checking, used to label generated samples."""

from __future__ import annotations

import re
from fractions import Fraction

ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
FORMAT_RE = re.compile(r"^\s*<think>.+?</think>\s*<answer>.+?</answer>\s*$", re.DOTALL)
NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?(?:/\d+)?")


def extract_answer(text: str) -> str | None:
    matches = ANSWER_RE.findall(text)
    return matches[-1].strip() if matches else None


def parse_number(text: str | None) -> Fraction | None:
    if text is None:
        return None
    found = NUMBER_RE.findall(text)
    if not found:
        return None
    try:
        return Fraction(found[-1].replace(",", ""))
    except (ValueError, ZeroDivisionError):
        return None


def is_correct(completion: str, gold: str) -> bool:
    pred, ref = parse_number(extract_answer(completion)), parse_number(gold)
    return pred is not None and ref is not None and pred == ref


def is_well_formatted(completion: str) -> bool:
    return bool(FORMAT_RE.match(completion))

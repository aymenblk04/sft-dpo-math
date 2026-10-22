"""GSM8K formatting for SFT, DPO and evaluation."""

from __future__ import annotations

import re

from datasets import Dataset, load_dataset

SYSTEM_PROMPT = (
    "You are a careful math assistant. Reason step by step inside "
    "<think> </think> tags, then give only the final number inside "
    "<answer> </answer> tags."
)

# GSM8K solutions contain calculator annotations like "<<48/2=24>>"
CALC_RE = re.compile(r"<<[^>]*>>")


def make_prompt(question: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


def split_solution(solution: str) -> tuple[str, str]:
    """Return (reasoning, final_answer) from a raw GSM8K solution."""
    reasoning, _, final = solution.rpartition("####")
    reasoning = CALC_RE.sub("", reasoning).strip()
    return reasoning, final.strip().replace(",", "")


def format_target(solution: str) -> str:
    reasoning, final = split_solution(solution)
    return f"<think>\n{reasoning}\n</think>\n<answer>{final}</answer>"


def to_sft_example(row: dict) -> dict:
    """Conversational prompt/completion: TRL only trains on the completion."""
    return {
        "prompt": make_prompt(row["question"]),
        "completion": [{"role": "assistant", "content": format_target(row["answer"])}],
    }


def to_eval_example(row: dict) -> dict:
    return {"prompt": make_prompt(row["question"]), "answer": split_solution(row["answer"])[1]}


def _subsample(ds: Dataset, n: int | None, seed: int) -> Dataset:
    if n is not None and n < len(ds):
        ds = ds.shuffle(seed=seed).select(range(n))
    return ds


def load_raw(split: str, num_samples: int | None = None, seed: int = 0) -> Dataset:
    return _subsample(load_dataset("openai/gsm8k", "main", split=split), num_samples, seed)


def build_sft_dataset(raw: Dataset) -> Dataset:
    return raw.map(to_sft_example, remove_columns=raw.column_names)


def build_eval_dataset(raw: Dataset) -> Dataset:
    return raw.map(to_eval_example, remove_columns=raw.column_names)

"""Greedy GSM8K evaluation: accuracy, format rate, length."""

from __future__ import annotations

import json
from pathlib import Path

from sftdpo.generation import generate
from sftdpo.verify import extract_answer, is_correct, is_well_formatted


def score(completions: list[str], golds: list[str]) -> tuple[dict, list[dict]]:
    records = [
        {
            "completion": c,
            "prediction": extract_answer(c),
            "gold": g,
            "correct": is_correct(c, g),
            "format_ok": is_well_formatted(c),
        }
        for c, g in zip(completions, golds)
    ]
    n = max(len(records), 1)
    metrics = {
        "n": len(records),
        "accuracy": sum(r["correct"] for r in records) / n,
        "format_rate": sum(r["format_ok"] for r in records) / n,
        "mean_chars": sum(len(r["completion"]) for r in records) / n,
    }
    return metrics, records


def evaluate(model, tokenizer, dataset, max_new_tokens=320, batch_size=16, out_dir=None) -> dict:
    outs = generate(model, tokenizer, dataset["prompt"], max_new_tokens, batch_size)
    metrics, records = score([o[0] for o in outs], dataset["answer"])
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
        with open(out / "samples.jsonl", "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in records)
    return metrics

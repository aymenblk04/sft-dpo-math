"""Build DPO preference pairs from sampled completions, labelled by verification.

For each question we sample K completions from the SFT model and check the
final answer. Questions where the model is always right or always wrong carry
no preference signal and are dropped; the others give (correct, incorrect) pairs.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from sftdpo.verify import is_correct, is_well_formatted


@dataclass
class PairStats:
    questions: int = 0
    all_correct: int = 0
    all_wrong: int = 0  # no sample both correct and well formatted
    mixed: int = 0
    pairs: int = 0

    def as_dict(self) -> dict:
        d = self.__dict__.copy()
        d["pass_at_k"] = (self.all_correct + self.mixed) / max(self.questions, 1)
        return d


def _rejected_order(wrong: list[str], strategy: str, rng: random.Random) -> list[str]:
    if strategy == "random":
        return rng.sample(wrong, len(wrong))
    if strategy == "hard":
        # well-formatted but wrong answers first: the format is right, only the
        # reasoning differs, so the pair isolates correctness
        formatted = [w for w in wrong if is_well_formatted(w)]
        rest = [w for w in wrong if not is_well_formatted(w)]
        return rng.sample(formatted, len(formatted)) + rng.sample(rest, len(rest))
    raise ValueError(f"unknown strategy {strategy!r}")


def build_pairs(
    prompts: list[list[dict]],
    completions: list[list[str]],
    golds: list[str],
    max_pairs_per_prompt: int = 2,
    strategy: str = "hard",
    seed: int = 0,
) -> tuple[list[dict], PairStats]:
    rng = random.Random(seed)
    rows, stats = [], PairStats()
    for prompt, samples, gold in zip(prompts, completions, golds):
        stats.questions += 1
        right = [s for s in samples if is_correct(s, gold) and is_well_formatted(s)]
        wrong = [s for s in samples if not is_correct(s, gold)]
        if not wrong:
            stats.all_correct += 1
            continue
        if not right:
            stats.all_wrong += 1
            continue
        stats.mixed += 1
        rejected = _rejected_order(wrong, strategy, rng)
        chosen = rng.sample(right, len(right))
        for i in range(min(max_pairs_per_prompt, len(rejected))):
            rows.append(
                {
                    "prompt": prompt,
                    "chosen": [{"role": "assistant", "content": chosen[i % len(chosen)]}],
                    "rejected": [{"role": "assistant", "content": rejected[i]}],
                }
            )
            stats.pairs += 1
    return rows, stats

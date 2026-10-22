from datasets import Dataset

from sftdpo.data import (
    SYSTEM_PROMPT,
    build_eval_dataset,
    build_sft_dataset,
    format_target,
    split_solution,
)
from sftdpo.verify import is_correct, is_well_formatted

SOLUTION = "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\n48+24 = <<48+24=72>>72\n#### 1,072"


def raw():
    return Dataset.from_dict({"question": ["How many clips?"], "answer": [SOLUTION]})


def test_split_removes_calculator_annotations():
    reasoning, final = split_solution(SOLUTION)
    assert "<<" not in reasoning
    assert final == "1072"


def test_target_is_valid_for_our_own_checker():
    target = format_target(SOLUTION)
    assert is_well_formatted(target)
    assert is_correct(target, "1072")


def test_sft_dataset_format():
    row = build_sft_dataset(raw())[0]
    assert row["prompt"][0]["content"] == SYSTEM_PROMPT
    assert row["completion"][0]["role"] == "assistant"


def test_eval_dataset_format():
    row = build_eval_dataset(raw())[0]
    assert row["answer"] == "1072"

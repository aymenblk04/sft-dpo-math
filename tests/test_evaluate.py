import json

from datasets import Dataset

from sftdpo.data import build_eval_dataset
from sftdpo.evaluate import evaluate, score
from sftdpo.generation import load_model


def test_score():
    m, recs = score(
        ["<think>a</think><answer>4</answer>", "<answer>3</answer>", "so the total is 4."], ["4", "4", "4"]
    )
    assert m["accuracy"] == 1 / 3 and m["format_rate"] == 1 / 3
    assert m["accuracy_lenient"] == 2 / 3  # untagged correct answer counts, wrong tagged one does not
    assert recs[1]["prediction"] == "3"


def test_evaluate_writes_files(tiny_model_path, tmp_path):
    raw = Dataset.from_dict({"question": ["2+2?"] * 2, "answer": ["#### 4"] * 2})
    model, tok = load_model(tiny_model_path)
    m = evaluate(model, tok, build_eval_dataset(raw), max_new_tokens=4, out_dir=tmp_path)
    assert json.loads((tmp_path / "metrics.json").read_text()) == m
